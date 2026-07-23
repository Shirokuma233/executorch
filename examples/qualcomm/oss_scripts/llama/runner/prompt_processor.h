/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/attention_sink_rope_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/cache_utils.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/utils.h>
#include <functional>
#include <memory>
#include <string>

namespace example {
/**
 * @class PromptProcessor
 * @brief Class for processing prompts using decoder and key-value manager.
 */

class PromptProcessor {
 public:
  struct Metadata {
    int32_t context_len;
    int64_t num_heads;
    int64_t num_layers;
    int32_t ar_len;
    int32_t vocab_size;
    bool use_int64_token;
    int sliding_window;
    CacheMode cache_mode;
  };
  PromptProcessor(
      DecoderRunner* decoder_runner,
      KVManager* kv_manager,
      const std::string& method_name,
      Metadata metadata,
      std::unique_ptr<executorch::extension::MethodMeta> method_meta,
      // DFlash headless-decoder companions. Null for every other eval mode, in
      // which case the prompt processor behaves exactly as before.
      executorch::extension::Module* emb_module = nullptr,
      executorch::extension::Module* lm_head_module = nullptr,
      std::unique_ptr<executorch::extension::MethodMeta> emb_prefill_meta =
          nullptr,
      std::unique_ptr<executorch::extension::MethodMeta> lm_head_prefill_meta =
          nullptr,
      float embeds_scale = 1.0f,
      int32_t embeds_zero_point = 0,
      float logits_scale = 1.0f,
      int32_t logits_zero_point = 0);

  virtual ~PromptProcessor() = default;

  /**
   * @brief Initialize I/O tensor and allocate I/O data buffer.
   * @param buffer_manager Pointer to IMemAlloc instance; by default, it uses a
   * shared buffer with RPC memory.
   * @param method_meta Method metadata.
   */
  void init_io(
      IMemAlloc* buffer_manager,
      executorch::runtime::Result<executorch::runtime::MethodMeta> method_meta);

  /**
   * @brief Get the all logits generated
   *
   * @return std::vector<std::byte>& all the logits generated
   */
  virtual const std::vector<std::byte>& get_all_logits();

  const std::vector<TensorStructRaw>& get_extra_outputs() const {
    return extra_outputs_;
  }

  // `extra_outputs_` is a single ar_len-row buffer that every prefill iteration
  // overwrites, so a consumer that needs the whole prompt's hidden states must
  // drain it per chunk. The observer runs right after each graph step with the
  // number of meaningful rows and the absolute position of row 0.
  using ExtraOutputObserver = std::function<
      void(const std::vector<TensorStructRaw>&, int32_t n_valid, int64_t pos_base)>;
  void set_extra_output_observer(ExtraOutputObserver observer) {
    extra_output_observer_ = std::move(observer);
  }

  /**
   * Prefill an LLM Module with the given text input.
   * @param prompt_tokens The text prompt tokens to the LLM Module. Encoded by
   * tokenizer.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * Module.
   * @param dump_logits Used to save all logits. Only enable when analyzing
   * accuracy.
   * @return The next token of the LLM Module after prefill.
   */
  executorch::runtime::Result<uint64_t> prefill(
      std::vector<uint64_t> prompt_tokens,
      int64_t start_pos,
      bool dump_logits,
      AttentionSinkRopeRunner* attention_sink_rope_runner);
  /**
   * @brief Get total I/O size in bytes (excluding the KV cache size)
   * @return Total I/O size in bytes.
   */
  inline const size_t total_prompt_processor_io_size_in_bytes() const {
    return input_toks_.size + input_pos_.size + attention_mask_.size +
        window_attention_mask_.size + logits_.size + extra_outputs_size_;
  }

  uint64_t graph_execute_calls() const {
    return graph_execute_calls_;
  }

  double graph_execute_time_ms() const {
    return graph_execute_time_ms_;
  }

 protected:
  // If the cache length is zero, it indicates a BERT model, which does not use
  // position ids or KV cache inputs.
  bool is_bert() const {
    return metadata_.context_len == metadata_.ar_len;
  }
  /**
   * @brief Fill in I/O buffers with prompt token and position.
   * @param prompt_tokens Vector of prompt tokens.
   * @param prompt_pos Position of the prompt.
   * @param start_pos Starting position.
   */
  void prepare_io(
      const std::vector<uint64_t>& prompt_tokens,
      int64_t prompt_pos,
      int64_t start_pos);

  // DFlash headless helpers: emb.pte prefill (tokens in emb_tok_buf_ -> f32
  // embeds -> quantize into inputs_[0]) and lm_head.pte prefill (u16 hidden ->
  // f32 logits in lm_head_logits_buf_). No-ops unless emb_module_ is set.
  void run_embedding_prefill();
  void run_lm_head_prefill(std::byte* hidden_u16);

  DecoderRunner* decoder_runner_;
  KVManager* kv_manager_;
  std::string method_name_;

  // metadata
  Metadata metadata_;

  // inputs and outputs
  TensorStruct<int64_t> input_toks_;
  TensorStruct<int32_t> input_pos_;
  TensorStructRaw attention_mask_;
  TensorStructRaw window_attention_mask_;
  TensorStructRaw logits_;
  std::vector<TensorStructRaw> extra_outputs_;
  size_t extra_outputs_size_{0};
  ExtraOutputObserver extra_output_observer_;

  // DFlash headless-decoder companions. Null for every other eval mode.
  executorch::extension::Module* emb_module_ = nullptr;
  executorch::extension::Module* lm_head_module_ = nullptr;
  std::unique_ptr<executorch::extension::MethodMeta> emb_prefill_meta_;
  std::unique_ptr<executorch::extension::MethodMeta> lm_head_prefill_meta_;
  float embeds_scale_ = 1.0f;
  int32_t embeds_zero_point_ = 0;
  float logits_scale_ = 1.0f;
  int32_t logits_zero_point_ = 0;
  bool emb_tok_i64_ = false;
  int32_t lm_head_vocab_size_ = 0;
  // emb/lm_head IO, carved from the shared (ION) region + QNN-registered, sized
  // for the prefill views. lm_head's input is the decoder hidden buffer
  // (logits_.data) directly, so no separate quantized-hidden buffer is needed.
  std::byte* emb_tok_buf_ = nullptr; // i32/i64 token ids into emb.pte
  size_t emb_tok_nbytes_ = 0;
  // TODO(dflash-uint16): 临时桥接。emb输出目前是 f32(编译端边界没做成 uint16),
  // 消费端 host quantize 进 decoder embeds(u16)。日后编译端把 emb 输出 tag 成
  // uint16 后,此 buffer 改 u16、可直传。详见 dflash/RUNNER_M5_PLAN.md "技术债" 节。
  std::byte* emb_out_buf_ = nullptr; // f32 embeds out of emb.pte
  size_t emb_out_nbytes_ = 0;
  std::byte* lm_head_logits_buf_ = nullptr; // f32 logits out of lm_head.pte
  size_t lm_head_logits_nbytes_ = 0;

  // layer -> TensorImpl
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> k_cache_in_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> v_cache_in_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> k_cache_out_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> v_cache_out_;

  std::vector<executorch::runtime::EValue> inputs_;
  std::vector<executorch::aten::Tensor> input_tensors_;
  std::vector<executorch::aten::Tensor> output_tensors_;
  // Used for attention sink to evict KV cache.
  std::vector<executorch::runtime::EValue> cache_inputs_;

  // Unused by default, only used when dump_logits_path is provided.
  std::vector<std::byte> prompt_all_logits_;

  uint64_t graph_execute_calls_{0};
  double graph_execute_time_ms_{0.0};

  // Per-pte prefill breakdown (DFlash split only; graph_execute_time_ms_ above
  // is decoder.pte). Lets the cost of the four-way split be attributed instead
  // of guessed -- prefill is where the split pays extra emb/lm_head executes.
  double emb_exec_ms_{0.0}; // emb.pte prefill, once per chunk
  double lm_head_exec_ms_{0.0}; // lm_head.pte prefill, once at the end
  double embeds_copy_ms_{0.0}; // memcpy of the u16 embeds into the decoder input
  double sample_ms_{0.0}; // host argmax for the first token
  uint64_t emb_calls_{0};
};
} // namespace example
