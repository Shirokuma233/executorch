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
 * The recompiled target decoder is HEADLESS: it consumes embeds (u16) and emits
 * hidden (u16), not tokens/logits. The embedding and vocab projection are their
 * own ptes, shared by the target (u16 side, host-quantized at the boundary) and
 * the draft (f32 side, direct):
 *   - input embedding: emb.pte kv_forward (tokens -> f32 embeds).
 *   - output logits:   lm_head.pte kv_forward (u16 hidden -> f32 logits), argmax
 *                      on the host (QNN's own is off by +/-65536 this wide).
 *
 * This generator owns neither the target nor the draft/emb/lm_head modules nor
 * either KVManager; the parent Runner constructs them all.
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
    // Shifted (DSpark/DeepSpec): block row k predicts the NEXT token, so all B rows
    // carry one and the verify block is [confirmed | B drafted] = B+1 tokens.
    // Aligned (DFlash b16): row k fills slot k, row 0 is discarded -> B-1 drafted,
    // verify block = B. Reading a shifted draft with the aligned rule puts every
    // token one slot early and acceptance collapses to zero.
    bool shifted_decode;
    // Skip the sink token's hidden when seeding, because the draft graph was
    // calibrated without it (DFlashConfig.drop_sink). Read from the pte rather
    // than configured here: seeding a row the quantizer never saw silently costs
    // acceptance instead of failing.
    bool drop_sink;
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
      std::unique_ptr<executorch::runtime::MethodMeta> draft_prefill_meta,
      executorch::extension::Module* emb_module,
      executorch::extension::Module* lm_head_module,
      std::unique_ptr<executorch::runtime::MethodMeta> emb_kv_meta,
      std::unique_ptr<executorch::runtime::MethodMeta> lm_head_kv_meta,
      float embeds_scale,
      int32_t embeds_zero_point,
      float logits_scale,
      int32_t logits_zero_point);

  ~DFlashTokenGenerator() override = default;

  // Bind target IO (incl. the L extra hidden outputs) + draft scratch buffers.
  void init_io(
      IMemAlloc* buffer_manager,
      executorch::runtime::Result<executorch::runtime::MethodMeta>
          target_method_meta) override;

  // The draft's five non-K/V inputs: noise, mask, new_context, context_pos,
  // block_pos.
  static constexpr size_t kNumDraftNonKvInputs = 5;

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
  // `out_hidden` may be null when only the cache write matters. The noise
  // embedding is produced on the fly by emb.pte over `block_tokens`.
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

  // Run emb.pte kv over `n_tokens` ids (padded to the graph's AR). Output f32
  // embeds land in `emb_out_buf_`.
  void run_embedding(const uint64_t* tokens, int32_t n_tokens);

  // Run lm_head.pte kv over the u16 hidden buffer. Logits land in
  // `lm_head_logits_buf_`.
  void run_lm_head(std::byte* hidden_u16);

  // Quantize `count` hidden rows (logits scale/zp) -> u16 -> lm_head.pte kv ->
  // host argmax over the full target vocab, filling `out_tokens`.
  void decode_lm_head(
      const float* hidden,
      int32_t count,
      std::vector<uint64_t>* out_tokens);

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

  // Headless-decoder companions. The decode path only ever touches the kv views
  // (verify runs 8 tokens, draft runs block_size); the prompt processor owns the
  // prefill views separately.
  executorch::extension::Module* emb_module_;
  executorch::extension::Module* lm_head_module_;
  std::unique_ptr<executorch::runtime::MethodMeta> emb_kv_meta_;
  std::unique_ptr<executorch::runtime::MethodMeta> lm_head_kv_meta_;
  // Boundary QDQ, read off the decoder pte getters at load time.
  float embeds_scale_;
  int32_t embeds_zero_point_;
  float logits_scale_;
  int32_t logits_zero_point_;
  int32_t lm_head_vocab_size_ = 0;

  // emb/lm_head IO, carved from the shared (ION) region and registered with QNN.
  std::byte* emb_tok_buf_ = nullptr; // i32 token ids into emb.pte
  size_t emb_tok_nbytes_ = 0;
  // TODO(dflash-uint16): 临时桥接。emb输出/draft-IO 目前是 f32(编译端边界没做成
  // uint16),故此 buffer 是 f32、消费端 host quantize。日后编译端把 emb 输出 tag 成
  // uint16 后,此 buffer 改 u16、可与 decoder embeds 直传。详见
  // dflash/RUNNER_M5_PLAN.md "技术债" 节。
  std::byte* emb_out_buf_ = nullptr; // f32 embeds out of emb.pte
  size_t emb_out_nbytes_ = 0;
  std::byte* lm_head_logits_buf_ = nullptr; // f32 logits out of lm_head.pte
  size_t lm_head_logits_nbytes_ = 0;
  // TODO(dflash-uint16): 临时桥接。draft block_hidden 目前是 f32(编译端边界没做成
  // uint16),故先 host quantize 进此 u16 buffer 再喂 lm_head。日后编译端把 draft
  // hidden tag 成 uint16 后,可删此 buffer、draft 输出直传 lm_head。详见
  // dflash/RUNNER_M5_PLAN.md "技术债" 节。
  std::byte* draft_hidden_u16_buf_ = nullptr; // quantized draft hidden -> lm_head
  size_t draft_hidden_u16_nbytes_ = 0;

  // Draft IO, carved from the shared (ION) region and registered with QNN. An
  // unregistered pointer falls back to a raw FastRPC copy of the whole tensor;
  // with ~21MB of K/V that overruns the transport (QNN 1003 skelExecute failed).
  // Sized for the wider of the two graphs and reused across calls.
  std::array<std::byte*, kNumDraftNonKvInputs> draft_in_bufs_{};
  std::array<size_t, kNumDraftNonKvInputs> draft_in_nbytes_{};

  // Set when the draft pte carries an lm_head, i.e. when its output count exceeds
  // hidden + 2*num_draft_layers. Then the graph returns the block's logits and the
  // host never scans the 743 MB vocab table -- only the argmax stays here, because
  // QNN's own is off by +/-65536 on an axis this wide.
  // kv_forward's context-append length (new_context's middle dim). Equals
  // block_size + 1 for shifted checkpoints, block_size for aligned ones.
  int32_t draft_decode_ar_ = 0;
  bool draft_has_lm_head_ = false;
  std::byte* draft_logits_buf_ = nullptr;
  size_t draft_logits_nbytes_ = 0;
  size_t draft_vocab_size_ = 0;
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

  // A pte's program may declare an output Float while the QNN delegate actually
  // writes uint16 codes; both probe tooling and method_meta report the declared
  // value, so the mismatch is invisible until the numbers come out as garbage.
  // Called once per buffer after its first execute: reinterprets the payload as
  // both f32 and u16 and logs an error when the declared dtype looks wrong.
  static void check_payload_dtype(
      const char* name,
      const std::byte* buf,
      size_t n_elems,
      executorch::aten::ScalarType declared);
  bool dtype_checked_{false};

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

  // Per-pte breakdown. draft_time_ms_/target_verify_time_ms_ above are the
  // coarse per-phase totals; these split them across the individual graphs so
  // the cost of the four-way split is attributable.
  double emb_exec_ms_{0.0}; // emb.pte  (verify tokens + draft noise)
  double decoder_exec_ms_{0.0}; // decoder.pte (headless target)
  double lm_head_exec_ms_{0.0}; // lm_head.pte (verify + draft)
  double sample_ms_{0.0}; // host argmax over the vocab
  // draft.pte's own execute time is draft_time_ms_ above.
  uint64_t emb_calls_{0};
  uint64_t lm_head_calls_{0};
};

} // namespace example
