/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// =============================================================================
// EagleTokenGenerator — Phase 3/4 SCAFFOLD
// =============================================================================
//
// This file is intentionally a skeleton: the structural plumbing (constructor,
// member layout, interface, generation-loop control flow) is present and
// compiles, but the four hot paths are TODO:
//
//   * head_prefill_step — bind head IO + execute "prefill_forward"
//   * head_decode_step  — bind head IO + execute "kv_forward" + sample
//   * target_verify     — pack draft tokens into target ar slot, run target
//                         "kv_forward", read hidden_LMH outputs
//   * KV rollback wiring — call KVManager::update_cache(... selected[]) for
//                         both target and head with the right boolean mask
//
// In Phase 4 the chain-mode mask becomes a tree mask via TreeBuilder
// (see tree_attention.h/.cpp).
//
// Until those are filled in, EagleTokenGenerator falls back to invoking the
// parent TokenGenerator::generate() so the runner remains usable.
//
// =============================================================================

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/eagle_token_generator.h>

#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <cstring>

using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;

namespace example {

EagleTokenGenerator::EagleTokenGenerator(
    tokenizers::Tokenizer* tokenizer,
    DecoderRunner* target_runner,
    KVManager* target_kv_manager,
    executorch::extension::Module* head_module,
    KVManager* head_kv_manager,
    const std::string& target_kv_method_name,
    std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids,
    Metadata metadata,
    executorch::llm::Stats* stats,
    std::unique_ptr<MethodMeta> target_method_meta,
    std::unique_ptr<MethodMeta> head_kv_method_meta,
    std::unique_ptr<MethodMeta> head_prefill_method_meta)
    : TokenGenerator(
          tokenizer,
          target_runner,
          target_kv_manager,
          target_kv_method_name,
          std::move(eos_ids),
          TokenGenerator::Metadata{
              metadata.target_context_len,
              metadata.target_num_heads,
              metadata.target_num_layers,
              metadata.target_ar_len,
              metadata.target_vocab_size,
              metadata.use_int64_token,
              metadata.target_sliding_window,
              metadata.target_cache_mode,
          },
          stats,
          std::move(target_method_meta)),
      head_module_(head_module),
      head_kv_manager_(head_kv_manager),
      head_kv_method_meta_(std::move(head_kv_method_meta)),
      head_prefill_method_meta_(std::move(head_prefill_method_meta)),
      eagle_meta_(metadata),
      target_hidden_low_(nullptr),
      target_hidden_mid_(nullptr),
      target_hidden_high_(nullptr) {
  ET_LOG(
      Info,
      "[Eagle] EagleTokenGenerator constructed: target_ar_len=%d draft_len=%d "
      "draft_vocab_size=%d hidden_dim=%d",
      eagle_meta_.target_ar_len,
      eagle_meta_.draft_len,
      eagle_meta_.draft_vocab_size,
      eagle_meta_.hidden_dim);
}

void EagleTokenGenerator::init_io(
    IMemAlloc* buffer_manager,
    Result<MethodMeta> method_meta) {
  // 1) Standard target IO (input toks, attn mask, pos, KV).
  TokenGenerator::init_io(buffer_manager, std::move(method_meta));

  // 2) Bind target's three extra hidden outputs.
  // Target output layout per Phase 1 modification:
  //   [0]: logits
  //   [1..1+num_layers): k_cache_out
  //   [1+num_layers..1+2*num_layers): v_cache_out
  //   [1+2*num_layers]:   hidden_low
  //   [1+2*num_layers+1]: hidden_mid
  //   [1+2*num_layers+2]: hidden_high
  //
  // TODO(phase-3): use decoder_runner_->set_outputs(...) to bind preallocated
  // buffers for the three hidden outputs and store their pointers in
  // target_hidden_{low,mid,high}_.
  ET_LOG(
      Info,
      "[Eagle] init_io: target hidden binding TODO (Phase 3)");

  // 3) Allocate head IO buffers (host-side scratch; not RPC-shared yet).
  size_t H = eagle_meta_.hidden_dim;
  size_t ctx = eagle_meta_.target_context_len;
  size_t vK = eagle_meta_.head_n_kv_heads;
  size_t hd = eagle_meta_.head_head_dim;

  // dtype: fp16 by default (matches --eagle_hidden_io fp16). Use 2 bytes.
  // TODO(phase-3): respect actual dtype from method_meta.
  size_t fp16_b = 2;

  head_prev_feature_buf_.resize(H * fp16_b);
  head_tok_emb_buf_.resize(H * fp16_b);
  head_attn_mask_buf_.resize(ctx * fp16_b);
  head_pos_buf_.resize(sizeof(int32_t));
  head_k_cache_buf_.resize(vK * hd * (ctx - 1) * fp16_b);
  head_v_cache_buf_.resize(vK * hd * (ctx - 1) * fp16_b);
  head_logits_buf_.resize(eagle_meta_.draft_vocab_size * fp16_b);

  ET_LOG(
      Info,
      "[Eagle] head IO buffers allocated: prev_feature=%zuB attn_mask=%zuB "
      "kv=%zuB logits=%zuB (fp16)",
      head_prev_feature_buf_.size(),
      head_attn_mask_buf_.size(),
      head_k_cache_buf_.size(),
      head_logits_buf_.size());
}

// ============================================================================
// Generation loop
// ============================================================================
Result<int64_t> EagleTokenGenerator::generate(
    std::vector<uint64_t> tokens,
    int64_t start_pos,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    bool dump_logits,
    AttentionSinkRopeRunner* attention_sink_rope_runner) {
  const bool tree_mode = !tree_branching_per_depth_.empty();
  ET_LOG(
      Info,
      "[Eagle] generate() called: start_pos=%lld seq_len=%d tree_mode=%d. "
      "Phase 3/4 runtime not yet implemented; falling back to plain "
      "TokenGenerator path. accept_rate stats will not be reported.",
      static_cast<long long>(start_pos),
      seq_len,
      tree_mode ? 1 : 0);

  // ============================================================
  // TODO(phase-3, chain): see header for the speculative loop spec.
  // TODO(phase-4, tree): when tree_mode is true:
  //   1. Construct TreeBuilder(tree_branching_per_depth_, max_tree_size).
  //   2. Each verify step:
  //      a. nodes = TreeBuilder::build(last_committed, head_step_callback)
  //         where head_step_callback wraps head_decode_step + sample_draft +
  //         draft_to_target.
  //      b. Pack nodes' token_ids into target ar slot.
  //      c. TreeBuilder::fill_tree_attention_mask(nodes, target_ar_len, ...).
  //      d. TreeBuilder::fill_positions(nodes, target_ar_len, cur_pos, ...).
  //      e. target.kv_forward — read logits + hidden_LMH per slot.
  //      f. Walk the tree: starting at root, follow children whose target
  //         argmax matches, accumulating accepted prefix. Take longest path.
  //      g. KV rollback: target_selected[k] = (k is on accepted path).
  //         head rollback similarly using TreeNode::depth.
  // ============================================================

  // Fallback: parent TokenGenerator::generate handles plain target-only decode.
  return TokenGenerator::generate(
      tokens, start_pos, seq_len, token_callback, dump_logits,
      attention_sink_rope_runner);
}

// ============================================================================
// Head paths (TODO)
// ============================================================================
void EagleTokenGenerator::head_prefill_step(
    const float* /*hidden_LMH*/,
    uint64_t /*prev_token*/,
    int64_t /*pos*/) {
  // TODO(phase-3):
  //   1. Cast/copy hidden_LMH (fp32 host) into head_prev_feature_buf_ (fp16 dev).
  //   2. Look up tok_embedding for prev_token. The embedding table is shared
  //      with target — but target's embedding is INSIDE the target pte. We
  //      need either (a) a separate embedding pte (TOK_EMBEDDING) or (b) ask
  //      the target to expose embedding lookup as a constant_method.
  //      For Phase 3 simplicity: assume the head pte itself includes the
  //      embedding lookup as the FIRST step of prefill_forward, so the runtime
  //      passes raw token ids and the head computes its own embedding. Need
  //      to update Phase 2 wrapper accordingly.
  //   3. Build attn_mask for prefill (causal, single position).
  //   4. Bind k/v cache from head_kv_manager_ (input from prior accepted) and
  //      output buffers.
  //   5. head_module_->execute("prefill_forward", inputs).
  //   6. Read head logits and head 'a_out' into prev_feature buffer.
  ET_LOG(Error, "[Eagle] head_prefill_step is TODO");
}

uint64_t EagleTokenGenerator::head_decode_step(
    const float* /*prev_a*/,
    uint64_t /*prev_token*/,
    int64_t /*pos*/,
    float* /*prev_a_buffer*/) {
  // TODO(phase-3): mirror head_prefill_step but call "kv_forward" instead.
  // Sample draft id from head_logits_buf_ (32000-way argmax).
  ET_LOG(Error, "[Eagle] head_decode_step is TODO");
  return 0;
}

// ============================================================================
// Target verify (TODO)
// ============================================================================
void EagleTokenGenerator::target_verify(
    const std::vector<uint64_t>& /*packed_tokens*/,
    int64_t /*cur_pos*/,
    std::vector<uint64_t>* /*target_sampled_tokens*/,
    std::vector<float>* /*out_hidden_LMH_per_slot*/) {
  // TODO(phase-3):
  //   1. Fill input_toks_[0..target_ar_len-1] with packed_tokens (pad with 0).
  //   2. Build chain causal mask: slot k attends to [0..cur_pos+k].
  //      Phase 4: replace with TreeBuilder::fill_tree_attention_mask().
  //   3. Fill input_pos_[k] = cur_pos + k.
  //   4. KV manager: bind k/v_cache_in (history) + k/v_cache_out (new slots).
  //   5. decoder_runner_->step(method_name_, inputs_) -> logits.
  //   6. Read target_hidden_{low,mid,high}_ outputs into the per-slot vector.
  //   7. argmax(logits[k]) for each slot -> target_sampled_tokens.
  ET_LOG(Error, "[Eagle] target_verify is TODO");
}

// ============================================================================
// Sampling
// ============================================================================
uint64_t EagleTokenGenerator::sample_draft(const std::byte* /*draft_logits_buf*/) {
  // TODO(phase-3): 32000-way argmax. Be careful with the dtype:
  //   - If head logits are quantized uint16, dequantize first using the head
  //     pte's get_logits_scale / get_logits_zero_point constant_methods.
  //   - If fp16 (Phase 2 default), reinterpret as __fp16 and argmax directly.
  ET_LOG(Error, "[Eagle] sample_draft is TODO");
  return 0;
}

} // namespace example
