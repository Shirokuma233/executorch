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

namespace example {

/**
 * @class DraftTokenGenerator
 * @brief Speculative decoding token generator.
 *
 * A small draft model (CPU, fp32) proposes `draft_len` candidate tokens per
 * step.  The large target model (NPU, quantized) then verifies all candidates
 * in a single forward pass of length `draft_len + 1`.  Accepted tokens are
 * committed; the first rejected position restarts the draft loop.
 *
 * The target model is the same QNN-compiled model used by the standard
 * TokenGenerator.  The draft model is a separate CPU ExecuTorch module loaded
 * from `draft_model_path`.
 */
template <typename T>
class DraftTokenGenerator : public TokenGenerator<T> {
 public:
  struct Metadata {
    int32_t context_len;
    int64_t num_heads;
    int64_t num_layers;
    int32_t ar_len;   // target model ar_len (draft_len + 1, rounded up)
    int32_t vocab_size;
    bool use_int64_token;
    int32_t draft_len; // number of tokens the draft model proposes per step
    int sliding_window;
    CacheMode cache_mode;
  };

  DraftTokenGenerator(
      tokenizers::Tokenizer* tokenizer,
      DecoderRunner* decoder_runner,
      KVManager<T>* kv_manager,
      const std::string& forward_name,
      std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids,
      Metadata metadata,
      executorch::llm::Stats* stats,
      std::unique_ptr<executorch::extension::Module> draft_module);

  ~DraftTokenGenerator() = default;

  /**
   * @brief Generate tokens using speculative decoding.
   *
   * Each iteration:
   *   1. Draft model auto-regressively generates `draft_len` tokens on CPU.
   *   2. Target model verifies all `draft_len + 1` positions in one NPU call.
   *   3. Accepted tokens (up to the first mismatch) are committed.
   */
  executorch::runtime::Result<int64_t> generate(
      std::vector<uint64_t> tokens,
      int64_t start_pos,
      int32_t seq_len,
      std::function<void(const std::string&)> token_callback,
      bool dump_logits,
      AttentionSinkRopeRunner* attention_sink_rope_runner) override;

 private:
  Metadata metadata_;

  // Draft model (CPU, fp32 ExecuTorch module)
  std::unique_ptr<executorch::extension::Module> draft_module_;

  // Draft model I/O buffers (ar_len=1, CPU tensors)
  std::vector<int64_t> draft_input_toks_;
  std::vector<int32_t> draft_input_pos_;
  // draft attention mask: [1, 1, context_len]
  std::vector<float> draft_attention_mask_;

  /**
   * @brief Run the draft model for one step and return the sampled token.
   */
  uint64_t draft_step(uint64_t cur_token, int32_t pos);

  /**
   * @brief Prepare target model I/O for verifying `draft_tokens`.
   *
   * Fills input_toks_ with [cur_token, draft_tokens[0..draft_len-1]] and
   * input_pos_ with [pos, pos+1, ..., pos+draft_len].
   */
  void prepare_verify_io(
      uint64_t cur_token,
      const std::vector<uint64_t>& draft_tokens,
      int32_t pos);
};

} // namespace example
