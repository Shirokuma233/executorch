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

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace example {

/**
 * @class DFlashTokenGenerator
 * @brief Block-diffusion speculative decoding (z-lab DFlash).
 *
 * Per step the draft graph runs ONCE and drafts a whole block of
 * `block_size - 1` tokens in parallel (bidirectional attention over
 * [cached-context ‖ new-context ‖ noise-block]). The target then verifies the
 * block at `target_ar_len = block_size` positions and the longest matching
 * prefix is accepted (chain verification — no tree).
 *
 * TWO CACHES
 *   A. hidden staging (`stage_buf_`) — the target's 5 selected hidden layers,
 *      concatenated to one 5H row per token. PromptProcessor's `extra_outputs_`
 *      is a single ar_len-row buffer that every prefill iteration overwrites, so
 *      each chunk must be drained before the next step. Rows are consumed by the
 *      draft as soon as `dflash_prefill_ar` of them accumulate, so this buffer
 *      never exceeds one draft-graph call's worth.
 *   B. draft KV cache (`draft_kv_manager_`) — 5 layers of the draft's own K/V,
 *      derived pointwise from A. This is the only long-lived state; its length
 *      is invariably equal to the number of committed tokens.
 *
 * The reference's `past_key_values_draft.crop(start)` has no counterpart here:
 * the block's K/V never enter the cache in the first place.
 *
 * Draft IO (two graphs sharing weights, differing only in the append length AR):
 *   prefill_forward : AR = dflash_prefill_ar_len   seeds the cache from the prompt
 *   kv_forward      : AR = block_size              appends the accepted rows
 *   in : noise_embedding [1,B,H], atten_mask [1,B,Cc+AR+B],
 *        new_context [1,AR,L*H], context_pos [1,AR], block_pos [1,B],
 *        past_k[Ld] [1,nKV,D,Cc], past_v[Ld] [1,nKV,Cc,D]
 *   out: hidden [1,B,H], k_new[Ld] [1,nKV,D,AR], v_new[Ld] [1,nKV,AR,D]
 * where Cc + AR == max_context_len for both graphs.
 *
 * The draft owns NO embed_tokens and NO lm_head:
 *   - input embedding: host gather from `embed_table_` (embed.bin).
 *   - output logits:   host matmul `lm_head_table_ @ hidden` then argmax over the
 *                      full target vocab. With tie_word_embeddings the two tables
 *                      are identical (one embed.bin serves both).
 *
 * This generator owns neither the target nor the draft module nor either
 * KVManager; the parent Runner constructs them all.
 */
class DFlashTokenGenerator : public TokenGenerator {
 public:
  struct Metadata {
    // ---- target ----
    int32_t target_context_len;
    int64_t target_num_heads;
    int64_t target_num_layers;
    int32_t target_ar_len; // = block_size
    int32_t target_vocab_size;
    bool use_int64_token;
    int target_sliding_window;
    CacheMode target_cache_mode;
    // ---- dflash draft ----
    int32_t block_size; // drafts block_size-1 tokens per block
    int32_t hidden_dim; // draft/target hidden size (2560)
    int32_t num_ctx_layers; // target hidden layers fused by fc (5)
    int32_t num_draft_layers; // draft transformer layers == KV cache layers (5)
    int32_t max_context_len; // draft KV cache context length
    int32_t prefill_ar_len; // context rows prefill_forward appends per call
    int64_t mask_token_id; // 151669
  };

  DFlashTokenGenerator(
      tokenizers::Tokenizer* tokenizer,
      DecoderRunner* target_runner,
      KVManager* target_kv_manager,
      executorch::extension::Module* draft_module,
      KVManager* draft_kv_manager,
      const std::string& target_kv_method_name,
      std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids,
      Metadata metadata,
      executorch::llm::Stats* stats,
      std::unique_ptr<executorch::runtime::MethodMeta> target_method_meta,
      std::unique_ptr<executorch::runtime::MethodMeta> draft_kv_meta,
      std::unique_ptr<executorch::runtime::MethodMeta> draft_prefill_meta);

  ~DFlashTokenGenerator() override = default;

  // Bind target IO (incl. the L extra hidden outputs) + draft scratch buffers.
  void init_io(
      IMemAlloc* buffer_manager,
      executorch::runtime::Result<executorch::runtime::MethodMeta>
          target_method_meta) override;

  // The draft's five non-K/V inputs: noise, mask, new_context, context_pos,
  // block_pos.
  static constexpr size_t kNumDraftNonKvInputs = 5;

  // Capped at the big-core count: the lm_head sweep is DDR-bound on a 742 MiB
  // table, so oversubscribing onto the little cores only adds contention.
  static constexpr size_t kLmHeadMaxThreads = 8;

  // Bytes the draft's non-K/V IO needs from the shared buffer (widest of the two
  // graphs per tensor). Runner must reserve this in the RpcMem region, which is a
  // bump allocator with no bounds check.
  static size_t draft_io_size_in_bytes(
      const executorch::runtime::MethodMeta& kv_meta,
      const executorch::runtime::MethodMeta& prefill_meta);

  executorch::runtime::Result<int64_t> generate(
      std::vector<uint64_t> tokens,
      int64_t start_pos,
      int32_t seq_len,
      std::function<void(const std::string&)> token_callback,
      bool dump_logits,
      AttentionSinkRopeRunner* attention_sink_rope_runner) override;

  // Target embedding table (fp16[vocab*hidden], row-major) for input embedding.
  void set_embed_table(
      std::vector<uint16_t> embed_table,
      int32_t vocab_size,
      int32_t hidden_size) {
    embed_table_ = std::move(embed_table);
    embed_vocab_size_ = vocab_size;
    embed_hidden_size_ = hidden_size;
  }

  // lm_head table (fp16[vocab*hidden]) for the host logits matmul. If the target
  // ties embed/lm_head, pass the same buffer as the embed table.
  void set_lm_head_table(std::vector<uint16_t> lm_head_table) {
    lm_head_table_ = std::move(lm_head_table);
  }

  // PromptProcessor observer: drain one prefill chunk's hidden into the staging
  // buffer, running prefill_forward whenever `prefill_ar_len` rows accumulate.
  void stage_prompt_hidden(
      const std::vector<TensorStructRaw>& extra_outputs,
      int32_t n_valid,
      int64_t pos_base);

  // Flush the staging remainder after the last prefill chunk. Leaves the draft
  // KV cache holding exactly one row per prompt token.
  void finish_prompt_seeding();

 private:
  // One draft graph call. Appends `n_new` context rows (read from `stage_buf_`,
  // RoPE positions `ctx_pos_base + j`) to the KV cache and drafts the block.
  // `out_hidden` may be null when only the cache write matters.
  void run_draft(
      const std::string& method,
      const executorch::runtime::MethodMeta& meta,
      int32_t ar,
      int32_t n_new,
      int64_t ctx_pos_base,
      const std::vector<uint64_t>& block_tokens,
      int64_t block_pos_base,
      std::vector<float>* out_hidden);

  // Run one prefill_forward over the staged rows, then reset the staging count.
  void flush_stage();

  // Host lm_head + argmax over the full target vocab for `count` hidden rows.
  void lm_head_argmax(
      const float* hidden,
      int32_t count,
      std::vector<uint64_t>* out_tokens) const;

  // Gather fp16 embedding row for token_id into dst[hidden_dim].
  void lookup_embedding(uint64_t token_id, uint16_t* dst) const;

  // Target verify: run target kv_forward over the packed block, read logits and
  // the L extra hidden outputs. Returns the target argmax per slot.
  void target_verify_block(
      const std::vector<uint64_t>& packed_tokens,
      int64_t cur_pos,
      std::vector<uint64_t>* target_sampled);

  executorch::extension::Module* draft_module_;
  KVManager* draft_kv_manager_;
  std::unique_ptr<executorch::runtime::MethodMeta> draft_kv_meta_;
  std::unique_ptr<executorch::runtime::MethodMeta> draft_prefill_meta_;
  Metadata dflash_meta_;

  // Reused target embedding (input) + lm_head (output) tables (fp16).
  std::vector<uint16_t> embed_table_;
  std::vector<uint16_t> lm_head_table_;
  int32_t embed_vocab_size_ = 0;
  int32_t embed_hidden_size_ = 0;

  // Draft IO, carved from the shared (ION) region and registered with QNN. An
  // unregistered pointer falls back to a raw FastRPC copy of the whole tensor;
  // with ~21MB of K/V that overruns the transport (QNN 1003 skelExecute failed).
  // Sized for the wider of the two graphs and reused across calls.
  std::array<std::byte*, kNumDraftNonKvInputs> draft_in_bufs_{};
  std::array<size_t, kNumDraftNonKvInputs> draft_in_nbytes_{};
  std::byte* draft_hidden_buf_ = nullptr;
  size_t draft_hidden_nbytes_ = 0;

  // Cache A: hidden staging. max(prefill_ar_len, block_size) rows of L*H fp32.
  std::vector<float> stage_buf_;
  int32_t stage_count_ = 0;
  int64_t stage_pos_base_ = 0;

  // Cache B length == committed token count == the reference's `start`.
  int32_t draft_ctx_len_ = 0;

  // Target extra hidden outputs (L layers, L <= 8) captured during verify.
  static constexpr int kMaxCtxLayers = 8;
  std::array<std::vector<std::byte>, kMaxCtxLayers> target_hidden_bufs_;
  std::array<executorch::aten::ScalarType, kMaxCtxLayers> target_hidden_dtypes_;
  std::array<std::vector<executorch::aten::TensorImpl::SizesType>, kMaxCtxLayers>
      target_hidden_sizes_;
  std::array<
      std::vector<executorch::aten::TensorImpl::DimOrderType>,
      kMaxCtxLayers>
      target_hidden_dim_orders_;
  std::array<std::unique_ptr<executorch::aten::TensorImpl>, kMaxCtxLayers>
      target_hidden_impls_;

  // stats
  uint64_t total_drafted_{0};
  uint64_t total_accepted_{0};
  uint64_t draft_calls_{0};
  uint64_t draft_prefill_calls_{0};
  uint64_t target_verify_calls_{0};
  double draft_time_ms_{0.0};
  double draft_prefill_time_ms_{0.0};
  double target_verify_time_ms_{0.0};
  double lm_head_time_ms_{0.0};
};

} // namespace example
