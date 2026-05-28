/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/token_generator.h>
#include <executorch/extension/module/module.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace example {

/**
 * @class EagleTokenGenerator
 * @brief Speculative decoding via the EAGLE-3 head.
 *
 * Phase 3 (chain): per generation step, the head produces `draft_len` candidate
 * tokens; the target is invoked once to verify all of them at
 * `target_ar_len = next_pow2(draft_len + 1)` positions. Accepted prefix tokens
 * are committed; both target and head KV caches are rolled back to the last
 * accepted boundary.
 *
 * Phase 4 (tree): the chain is replaced by a static balanced tree of size
 * `next_pow2(max_tree_size)`. See `tree_attention.h`.
 *
 * NOTE: This generator owns NEITHER the target nor the head module. It assumes
 * the parent Runner has loaded both modules and constructed two independent
 * `KVManager`s (one for target, one for head, both with `num_layers=1` for the
 * head).
 */
class EagleTokenGenerator : public TokenGenerator {
 public:
  struct Metadata {
    // Inherits / mirrors TokenGenerator::Metadata fields for the TARGET model.
    int32_t target_context_len;
    int64_t target_num_heads;
    int64_t target_num_layers;
    int32_t target_ar_len;       // next_pow2(draft_len+1) in Phase 3
    int32_t target_vocab_size;
    bool use_int64_token;
    int target_sliding_window;
    CacheMode target_cache_mode;

    // Head-specific.
    int32_t hidden_dim;          // = target hidden size, matches eagle config
    int32_t draft_len;           // chain proposal length (Phase 3)
    int32_t draft_vocab_size;    // 32000 for the provided ckpt
    int32_t head_n_kv_heads;     // 8 for Qwen3 head
    int32_t head_head_dim;       // 128
    int32_t max_tree_size;       // Phase 4: total nodes in tree

    // Hidden-state layer indices (low/mid/high) — informational only.
    int low_layer_idx;
    int mid_layer_idx;
    int high_layer_idx;
  };

  EagleTokenGenerator(
      tokenizers::Tokenizer* tokenizer,
      DecoderRunner* target_runner,
      KVManager* target_kv_manager,
      executorch::extension::Module* head_module,
      KVManager* head_kv_manager,
      const std::string& target_kv_method_name,
      std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids,
      Metadata metadata,
      executorch::llm::Stats* stats,
      std::unique_ptr<executorch::runtime::MethodMeta> target_method_meta,
      std::unique_ptr<executorch::runtime::MethodMeta> head_kv_method_meta,
      std::unique_ptr<executorch::runtime::MethodMeta> head_prefill_method_meta);

  ~EagleTokenGenerator() override = default;

  // Bind both target and head IO buffers.
  void init_io(
      IMemAlloc* buffer_manager,
      executorch::runtime::Result<executorch::runtime::MethodMeta> target_method_meta)
      override;

  // Speculative decoding generation loop.
  executorch::runtime::Result<int64_t> generate(
      std::vector<uint64_t> tokens,
      int64_t start_pos,
      int32_t seq_len,
      std::function<void(const std::string&)> token_callback,
      bool dump_logits,
      AttentionSinkRopeRunner* attention_sink_rope_runner) override;

  // Read d2t mapping from head pte's `get_d2t` constant_method.
  // Must be called after construction, before generate().
  void set_d2t(std::vector<int64_t> d2t) { d2t_ = std::move(d2t); }

  // Phase 4: configure tree topology. If `branching_per_depth` is empty, the
  // generator runs in chain mode. Otherwise it builds a static tree of size
  // <= max_tree_size at each verify step.
  // TODO(phase-4): implement the tree-mode generate() branch.
  void set_tree_topology(std::vector<int> branching_per_depth) {
    tree_branching_per_depth_ = std::move(branching_per_depth);
  }

 private:
  // Map a draft-vocab id back to a target-vocab id.
  inline uint64_t draft_to_target(uint64_t draft_id) const {
    return static_cast<uint64_t>(
        static_cast<int64_t>(draft_id) + d2t_[draft_id]);
  }

  // ---- Draft path (head module) ----
  // Run head.prefill_forward with a [3H] hidden vector to seed `prev_a` and
  // produce the first draft logits.
  // TODO(phase-3): bind head input buffers (hidden_LMH, tok_emb, attn_mask,
  // pos, kv_cache_in/out), call head_module_->execute("prefill_forward").
  void head_prefill_step(
      const float* hidden_LMH,    // [3 * hidden_dim], host fp16/fp32
      uint64_t prev_token,
      int64_t pos);

  // Run head.kv_forward(prev_a, emb(prev_token)) for one step and return the
  // sampled draft id. Writes the new `prev_a` to `prev_a_buffer`.
  // TODO(phase-3): full implementation.
  uint64_t head_decode_step(
      const float* prev_a,
      uint64_t prev_token,
      int64_t pos,
      float* prev_a_buffer);

  // ---- Verify path (target module) ----
  // Pack [last_committed, draft1, ..., draftN] into target ar slot, fill the
  // tree-attention mask (chain mask in Phase 3, real tree in Phase 4),
  // execute target.kv_forward, and return target-sampled tokens at every slot.
  // `out_hidden_LMH_per_slot[k]` is filled with the hidden state at slot k
  // (used to refresh head's `prev_g` after accept).
  // TODO(phase-3): full implementation.
  void target_verify(
      const std::vector<uint64_t>& packed_tokens,
      int64_t cur_pos,
      std::vector<uint64_t>* target_sampled_tokens,
      std::vector<float>* out_hidden_LMH_per_slot);

  // ---- Sampling ----
  // 32000-way argmax over the head's draft logits.
  uint64_t sample_draft(const std::byte* draft_logits_buf);

  // ---- Members ----
  executorch::extension::Module* head_module_;
  KVManager* head_kv_manager_;
  std::unique_ptr<executorch::runtime::MethodMeta> head_kv_method_meta_;
  std::unique_ptr<executorch::runtime::MethodMeta> head_prefill_method_meta_;
  Metadata eagle_meta_;

  // d2t mapping (loaded from head pte's get_d2t constant_method)
  std::vector<int64_t> d2t_;

  // Head IO buffers.
  // TODO(phase-3): allocate via buffer_manager and wire up TensorImpls.
  std::vector<std::byte> head_prev_feature_buf_;   // [hidden_dim], holds prev_a
  std::vector<std::byte> head_tok_emb_buf_;        // [hidden_dim]
  std::vector<std::byte> head_attn_mask_buf_;      // [1, target_context_len]
  std::vector<std::byte> head_pos_buf_;            // [1] int32
  // Head KV in/out per layer (1 layer for current Qwen3-1.7B head):
  std::vector<std::byte> head_k_cache_buf_;
  std::vector<std::byte> head_v_cache_buf_;
  std::vector<std::byte> head_logits_buf_;         // [draft_vocab_size]

  // Target hidden output binding pointers (filled in init_io).
  std::byte* target_hidden_low_;
  std::byte* target_hidden_mid_;
  std::byte* target_hidden_high_;

  // Stats accumulator.
  uint64_t total_drafted_{0};
  uint64_t total_accepted_{0};

  // Phase 4 tree config. Empty => chain mode.
  std::vector<int> tree_branching_per_depth_;
};

} // namespace example
