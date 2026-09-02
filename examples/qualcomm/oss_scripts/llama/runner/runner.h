/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama3.2 runner that includes preprocessing and post processing
// logic. The module takes in a string as input and emits a string as output.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/attention_sink_rope_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/cache_utils.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/token_generator.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace example {

enum DecoderModelVersion {
  kLlama2 = 0,
  kLlama3,
  kGemma,
  kGemma3,
  kGranite,
  kPhi4,
  kQwen2_5,
  kQwen3,
  kSmollm2_135m,
  kSmollm3,
  kCodegen,
  kGlm,
  kGemma2,
};

class Runner : public executorch::extension::llm::IRunner {
 public:
  explicit Runner(
      std::unique_ptr<executorch::extension::Module> module,
      const std::string& decoder_model,
      const std::string& model_path,
      const std::string& tokenizer_path,
      const std::string& performance_output_path,
      const std::string& dump_logits_path,
      const float temperature = 0.8f,
      const int eval_mode = EvalMode::kHybrid,
      const bool shared_buffer = false,
      const int ngram = 0,
      const int window = 0,
      const int gcap = 0,
      std::unique_ptr<tokenizers::Tokenizer> tokenizer = nullptr,
      std::unique_ptr<executorch::extension::Module>
          attention_sink_rope_module = nullptr,
      std::unique_ptr<executorch::extension::Module> eagle_head_module =
          nullptr,
      int max_tree_size = 0,
      int draft_len = 0,
      int tree_depth = 4,
      int tree_topk = 4,
      const std::string& eagle_d2t_path = "",
      const std::string& eagle_t2d_path = "",
      const std::string& eagle_embed_path = "",
      std::unique_ptr<executorch::extension::Module> dflash_draft_module =
          nullptr,
      int block_size = 16,
      int dflash_max_context_len = 0,
      std::unique_ptr<executorch::extension::Module> dflash_emb_module = nullptr,
      std::unique_ptr<executorch::extension::Module> dflash_lm_head_module =
          nullptr,
      int dflash_tree_budget = 0,
      float dflash_logit_out_scale = 0.0f,
      bool dflash_repeat_calib = true,
      float dflash_ctx_scale = 1.0f,
      float dflash_draft_mask_neg = -65504.0f);

  bool is_loaded() const override;
  executorch::runtime::Error load() override;
  // TODO: Support echo and warming
  executorch::runtime::Error generate(
      const std::string& prompt,
      const executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const executorch::llm::Stats&)> stats_callback = {})
      override;

  executorch::runtime::Error generate_from_prompt_or_file(
      const std::string& prompt,
      bool tokenized_prompt,
      const executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const executorch::llm::Stats&)> stats_callback = {});
  void stop() override {};
  void reset() override {};
  executorch::runtime::Result<DecoderModelVersion> get_decoder_model_version();

 private:
  enum EvalMode {
    kKVCached = 0,
    kHybrid,
    kLookaheadDecoding,
    kEagleDecoding,
    kDFlashDecoding,
    kUnsupported,
  };

  std::unique_ptr<executorch::extension::Module> module_;
  std::unique_ptr<executorch::extension::Module> attention_sink_rope_module_;
  std::unique_ptr<executorch::extension::Module> eagle_head_module_;
  std::unique_ptr<KVManager> eagle_kv_manager_;
  std::unique_ptr<KVManager> dflash_kv_manager_;
  // Draft non-K/V IO bytes to reserve in the shared buffer (see RpcMem).
  size_t dflash_draft_io_size_ = 0;
  // emb + lm_head pte IO bytes (both PromptProcessor and DFlashTokenGenerator
  // carve their own copies from the shared buffer). Same RpcMem-budget rule.
  size_t dflash_aux_io_size_ = 0;
  int max_tree_size_{0};
  int draft_len_{0};
  int tree_depth_{4};
  int tree_topk_{4};
  std::string eagle_d2t_path_;
  std::string eagle_t2d_path_;
  std::string eagle_embed_path_;
  int32_t context_len_{0};

  // DFlash
  std::unique_ptr<executorch::extension::Module> dflash_draft_module_;
  std::unique_ptr<executorch::extension::Module> dflash_emb_module_;
  std::unique_ptr<executorch::extension::Module> dflash_lm_head_module_;
  int block_size_{16};
  int dflash_max_context_len_{0};
  int dflash_tree_budget_{0};
  float dflash_logit_out_scale_{0.0f};
  bool dflash_repeat_calib_{true};
  float dflash_ctx_scale_{1.0f};
  float dflash_draft_mask_neg_{-65504.0f};

  int ngram_{0};
  int window_{0};
  int gcap_{0};

  // Defaults to StaticCahce, indicating that the model does not use a
  // global/local architecture.
  CacheMode cache_mode_{CacheMode::StaticCahce};
  int64_t cur_pos_{0};

  std::string tokenizer_path_;
  std::string performance_output_path_;
  std::string dump_logits_path_;
  float temperature_;
  EvalMode eval_mode_;
  bool shared_buffer_;

  DecoderModelVersion decoder_model_version_;
  std::unique_ptr<IMemAlloc> buffer_manager_;
  std::unique_ptr<KVManager> kv_manager_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<DecoderRunner> decoder_runner_;
  std::unique_ptr<AttentionSinkRopeRunner> attention_sink_rope_runner_;
  std::unique_ptr<PromptProcessor> prompt_processor_;
  std::unique_ptr<TokenGenerator> token_generator_;

  // stats
  executorch::llm::Stats stats_;
};

} // namespace example
