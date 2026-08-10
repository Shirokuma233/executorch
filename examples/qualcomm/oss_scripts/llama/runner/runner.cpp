/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A llama 3.2 runner that includes preprocessing and post processing
// logic. The module takes in a string as input and emits a string as output.

#include <c10/util/safe_numerics.h>
#include <executorch/examples/models/llama/runner/runner.h>
#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/client_mem.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/dflash_token_generator.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/eagle_token_generator.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/lhd_token_generator.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/rpc_mem.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <algorithm>
#include <cmath>
#include <fstream>

using executorch::extension::Module;
using executorch::extension::llm::get_rss_bytes;
using executorch::extension::llm::print_report;
using executorch::extension::llm::Stats;
using executorch::extension::llm::time_in_ms;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
namespace llm = ::executorch::extension::llm;

namespace example {
namespace {
void print_performance_report(
    const Stats& stats,
    const std::string& performance_output_path) {
  // For now, we just print the total inference time for CI, can save more info
  // in future if needed.
  std::ofstream outfile(performance_output_path.c_str());
  if (outfile.is_open()) {
    double num_tok = 0;
    if (stats.num_generated_tokens == 0) {
      // For cases like evaluate perplexity where prompt_len == cache_len
      num_tok = ((stats.num_prompt_tokens)) /
          (double)(stats.prompt_eval_end_ms - stats.inference_start_ms) *
          stats.SCALING_FACTOR_UNITS_PER_SECOND;
    } else {
      num_tok = (stats.num_generated_tokens) /
          (double)(stats.inference_end_ms - stats.inference_start_ms) *
          stats.SCALING_FACTOR_UNITS_PER_SECOND;
    }

    outfile << num_tok;
    outfile.close();
  } else {
    ET_LOG(Error, "Error saving the inference speed file");
  }
}

void save_logits(
    const std::string& dump_logits_path,
    const std::vector<std::byte>& prefill_logits,
    const std::vector<std::byte>& decode_logits) {
  std::ofstream outFile(dump_logits_path.c_str(), std::ios::binary);
  if (outFile.is_open()) {
    outFile.write(
        reinterpret_cast<const char*>(prefill_logits.data()),
        prefill_logits.size());

    outFile.write(
        reinterpret_cast<const char*>(decode_logits.data()),
        decode_logits.size());
    outFile.close();
  } else {
    ET_CHECK_MSG(false, "Error saving the dump logits file");
  }
}

} // namespace

Runner::Runner(
    std::unique_ptr<executorch::extension::Module> module,
    const std::string& decoder_model_version,
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& dump_logits_path,
    const std::string& performance_output_path,
    const float temperature,
    const int eval_mode,
    const bool shared_buffer,
    const int ngram,
    const int window,
    const int gcap,
    std::unique_ptr<tokenizers::Tokenizer> tokenizer,
    std::unique_ptr<executorch::extension::Module> attention_sink_rope_module,
    std::unique_ptr<executorch::extension::Module> eagle_head_module,
    int max_tree_size,
    int draft_len,
    int tree_depth,
    int tree_topk,
    const std::string& eagle_d2t_path,
    const std::string& eagle_t2d_path,
    const std::string& eagle_embed_path,
    std::unique_ptr<executorch::extension::Module> dflash_draft_module,
    int block_size,
    int dflash_max_context_len,
    std::unique_ptr<executorch::extension::Module> dflash_emb_module,
    std::unique_ptr<executorch::extension::Module> dflash_lm_head_module)
    : module_(std::move(module)),
      attention_sink_rope_module_(std::move(attention_sink_rope_module)),
      eagle_head_module_(std::move(eagle_head_module)),
      max_tree_size_(max_tree_size),
      draft_len_(draft_len),
      tree_depth_(tree_depth),
      tree_topk_(tree_topk),
      eagle_d2t_path_(eagle_d2t_path),
      eagle_t2d_path_(eagle_t2d_path),
      eagle_embed_path_(eagle_embed_path),
      dflash_draft_module_(std::move(dflash_draft_module)),
      dflash_emb_module_(std::move(dflash_emb_module)),
      dflash_lm_head_module_(std::move(dflash_lm_head_module)),
      block_size_(block_size),
      dflash_max_context_len_(dflash_max_context_len),
      ngram_(ngram),
      window_(window),
      gcap_(gcap),
      tokenizer_path_(tokenizer_path),
      performance_output_path_(performance_output_path),
      dump_logits_path_(dump_logits_path),
      temperature_(temperature),
      eval_mode_(static_cast<EvalMode>(eval_mode)),
      shared_buffer_(shared_buffer),
      tokenizer_(std::move(tokenizer)) {
  stats_.reset();

  if (decoder_model_version == "llama2") {
    decoder_model_version_ = DecoderModelVersion::kLlama2;
  } else if (decoder_model_version == "llama3") {
    decoder_model_version_ = DecoderModelVersion::kLlama3;
  } else if (decoder_model_version == "gemma") {
    decoder_model_version_ = DecoderModelVersion::kGemma;
  } else if (decoder_model_version == "gemma2") {
    decoder_model_version_ = DecoderModelVersion::kGemma2;
    cache_mode_ = CacheMode::HybridCache;
  } else if (decoder_model_version == "gemma3") {
    decoder_model_version_ = DecoderModelVersion::kGemma3;
    cache_mode_ = CacheMode::HybridCache;
  } else if (decoder_model_version == "granite") {
    decoder_model_version_ = DecoderModelVersion::kGranite;
  } else if (decoder_model_version == "phi_4_mini") {
    decoder_model_version_ = DecoderModelVersion::kPhi4;
  } else if (decoder_model_version == "qwen2_5") {
    decoder_model_version_ = DecoderModelVersion::kQwen2_5;
  } else if (decoder_model_version == "qwen3") {
    decoder_model_version_ = DecoderModelVersion::kQwen3;
  } else if (decoder_model_version == "smollm2_135m") {
    decoder_model_version_ = DecoderModelVersion::kSmollm2_135m;
  } else if (decoder_model_version == "smollm3") {
    decoder_model_version_ = DecoderModelVersion::kSmollm3;
  } else if (decoder_model_version == "codegen") {
    decoder_model_version_ = DecoderModelVersion::kCodegen;
  } else if (decoder_model_version == "glm") {
    decoder_model_version_ = DecoderModelVersion::kGlm;
  } else {
    ET_CHECK_MSG(false, "Unsupported Decoder Model");
  }

  ET_LOG(Info, "creating module: model_path=%s", model_path.c_str());
  ET_LOG(Info, "creating runner: tokenizer_path=%s", tokenizer_path_.c_str());
  ET_LOG(Info, "eval mode=%d", eval_mode_);
}

bool Runner::is_loaded() const {
  return module_->is_loaded() && tokenizer_ && decoder_runner_ &&
      prompt_processor_ && token_generator_ && kv_manager_ && buffer_manager_;
}

Error Runner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }

  std::string token_generator_method_name, prompt_processor_method_name;
  std::vector<std::string> method_names;
  switch (eval_mode_) {
    case EvalMode::kKVCached:
      prompt_processor_method_name = "kv_forward";
      token_generator_method_name = "kv_forward";
      method_names.emplace_back(token_generator_method_name);
      break;
    case EvalMode::kHybrid:
    case EvalMode::kLookaheadDecoding:
    case EvalMode::kEagleDecoding:
    case EvalMode::kDFlashDecoding:
      prompt_processor_method_name = "prefill_forward";
      token_generator_method_name = "kv_forward";
      method_names.emplace_back(prompt_processor_method_name);
      method_names.emplace_back(token_generator_method_name);
      break;
    case EvalMode::kUnsupported:
      ET_CHECK_MSG(false, "Unsupported llama evaluation mode");
      break;
  }
  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>();
  if (tokenizer_ == nullptr) {
    tokenizer_ = llm::load_tokenizer(tokenizer_path_);
    if (tokenizer_ == nullptr) {
      ET_LOG(
          Error, "Failed to load tokenizer with %s", tokenizer_path_.c_str());
      return Error::Internal;
    }
  }
  // A marker this tokenizer really knows encodes to exactly one id. One it does not
  // know gets BPE'd into pieces, and taking [0] of that hands back an unrelated
  // ordinary token -- Qwen has no <|eot_id|>, and the first piece it fell apart into
  // was '!', so every exclamation mark silently ended generation.
  auto add_eos = [&](const char* marker) {
    auto r = tokenizer_->encode(marker, 0, 0);
    if (r.ok() && r.get().size() == 1) {
      eos_ids->insert(r.get()[0]);
    }
  };
  switch (decoder_model_version_) {
    case DecoderModelVersion::kLlama3:
      add_eos("<|eot_id|>");
      add_eos("<|eot|>");
      add_eos("<|end_of_text|>");
      break;
    case DecoderModelVersion::kPhi4:
      add_eos("<|end|>");
      break;
    case DecoderModelVersion::kQwen3:
    case DecoderModelVersion::kSmollm2_135m:
    case DecoderModelVersion::kSmollm3:
      add_eos("<|im_end|>");
      add_eos("<|endoftext|>");
      break;
    case DecoderModelVersion::kGemma:
    case DecoderModelVersion::kGemma2:
    case DecoderModelVersion::kGemma3:
      add_eos("<end_of_turn>");
      break;
    case DecoderModelVersion::kCodegen:
      add_eos("<|endoftext|>");
      break;
    case DecoderModelVersion::kGlm:
      add_eos("<|user|>");
      break;
    default:
      break;
  }
  // eos_tok() only as a last resort: it reads tokenizer.json, and Qwen declares its
  // eos in tokenizer_config.json instead, so it hands back an unset 0 -- i.e. '!'.
  if (eos_ids->empty()) {
    eos_ids->insert(tokenizer_->eos_tok());
  }
  {
    std::string s;
    for (uint64_t e : *eos_ids) {
      s += std::to_string(e) + " ";
    }
    ET_LOG(Info, "eos ids: %s", s.c_str());
  }

  Result<MethodMeta> method_meta =
      module_->method_meta(token_generator_method_name);

  // For some tokenizer.json, runtime vocab_size might be different, use output
  // shape to get vocab size.
  int32_t vocab_size = method_meta->output_tensor_meta(0)->sizes()[2];

  // DFlash: the recompiled decoder is headless -- input[0] is embeds (u16) and
  // output[0] is hidden (u16), NOT tokens/logits. The embedding and vocab
  // projection are their own ptes. Load them up front so the sampler is sized to
  // the true vocab and both the prompt processor and the generator can bind them.
  const bool dflash = (eval_mode_ == EvalMode::kDFlashDecoding);
  std::unique_ptr<MethodMeta> dflash_emb_kv_meta, dflash_emb_prefill_meta,
      dflash_lm_kv_meta, dflash_lm_prefill_meta;
  float dflash_embeds_scale = 1.0f, dflash_logits_scale = 1.0f;
  int32_t dflash_embeds_zp = 0, dflash_logits_zp = 0;
  if (dflash) {
    ET_CHECK_MSG(
        dflash_emb_module_ != nullptr && dflash_lm_head_module_ != nullptr,
        "DFlash mode requires --dflash_emb_pte_path and --dflash_lm_head_pte_path "
        "(auto-derived as siblings of --dflash_draft_path).");
    ET_CHECK_OK_OR_RETURN_ERROR(
        dflash_emb_module_->load_method("tok_embedding_kv_forward"));
    ET_CHECK_OK_OR_RETURN_ERROR(
        dflash_emb_module_->load_method("tok_embedding_prefill_forward"));
    ET_CHECK_OK_OR_RETURN_ERROR(
        dflash_lm_head_module_->load_method("lm_head_kv_forward"));
    ET_CHECK_OK_OR_RETURN_ERROR(
        dflash_lm_head_module_->load_method("lm_head_prefill_forward"));
    dflash_emb_kv_meta = std::make_unique<MethodMeta>(
        ET_UNWRAP(dflash_emb_module_->method_meta("tok_embedding_kv_forward")));
    dflash_emb_prefill_meta = std::make_unique<MethodMeta>(ET_UNWRAP(
        dflash_emb_module_->method_meta("tok_embedding_prefill_forward")));
    dflash_lm_kv_meta = std::make_unique<MethodMeta>(
        ET_UNWRAP(dflash_lm_head_module_->method_meta("lm_head_kv_forward")));
    dflash_lm_prefill_meta = std::make_unique<MethodMeta>(
        ET_UNWRAP(dflash_lm_head_module_->method_meta("lm_head_prefill_forward")));
    // The true target vocab lives on the lm_head output, not decoder output[0].
    vocab_size =
        static_cast<int32_t>(dflash_lm_kv_meta->output_tensor_meta(0)->sizes()[2]);

    // Boundary QDQ params, self-calibrated into the decoder graph.
    auto ev_to_double = [](const executorch::runtime::EValue& ev,
                           double fb) -> double {
      if (ev.isDouble()) {
        return ev.toDouble();
      }
      if (ev.isInt()) {
        return static_cast<double>(ev.toInt());
      }
      if (ev.isTensor()) {
        auto t = ev.toTensor();
        if (t.numel() >= 1) {
          switch (t.scalar_type()) {
            case executorch::aten::ScalarType::Float:
              return t.const_data_ptr<float>()[0];
            case executorch::aten::ScalarType::Double:
              return t.const_data_ptr<double>()[0];
            case executorch::aten::ScalarType::Int:
              return t.const_data_ptr<int32_t>()[0];
            case executorch::aten::ScalarType::Long:
              return t.const_data_ptr<int64_t>()[0];
            default:
              break;
          }
        }
      }
      return fb;
    };
    auto read_dec_double = [&](const char* name, double fb) -> double {
      if (module_->method_names()->count(name) > 0) {
        auto res = module_->get(name);
        if (res.ok()) {
          return ev_to_double(res.get(), fb);
        }
      }
      ET_LOG(Error, "[DFlash] decoder getter %s missing; using %f", name, fb);
      return fb;
    };
    dflash_embeds_scale =
        static_cast<float>(read_dec_double("get_embeds_scale", 1.0));
    dflash_embeds_zp = static_cast<int32_t>(
        std::lround(read_dec_double("get_embeds_zero_point", 0.0)));
    dflash_logits_scale =
        static_cast<float>(read_dec_double("get_logits_scale", 1.0));
    dflash_logits_zp = static_cast<int32_t>(
        std::lround(read_dec_double("get_logits_zero_point", 0.0)));

    // Aux ION budget: emb/lm_head IO. The token generator binds the kv views
    // (its buffers never see prefill), the prompt processor the prefill views;
    // each buffer is single-view, so size it for exactly the graph it feeds.
    // The prompt processor reuses the decoder hidden buffer as the lm_head input,
    // so it needs no separate quantized-hidden buffer.
    auto in0 = [](const MethodMeta& m) {
      return m.input_tensor_meta(0)->nbytes();
    };
    auto out0 = [](const MethodMeta& m) {
      return m.output_tensor_meta(0)->nbytes();
    };
    const size_t tok_gen_aux = in0(*dflash_emb_kv_meta) +
        out0(*dflash_emb_kv_meta) + out0(*dflash_lm_kv_meta) +
        in0(*dflash_lm_kv_meta);
    const size_t prompt_aux = in0(*dflash_emb_prefill_meta) +
        out0(*dflash_emb_prefill_meta) + out0(*dflash_lm_prefill_meta);
    dflash_aux_io_size_ = tok_gen_aux + prompt_aux;
    ET_LOG(
        Info,
        "[DFlash] headless decoder: vocab=%d | embeds scale=%g zp=%d | logits "
        "scale=%g zp=%d | aux IO %.1f MB",
        vocab_size,
        dflash_embeds_scale,
        dflash_embeds_zp,
        dflash_logits_scale,
        dflash_logits_zp,
        static_cast<double>(dflash_aux_io_size_) / (1 << 20));
  }

  decoder_runner_ =
      std::make_unique<DecoderRunner>(module_.get(), vocab_size, temperature_);

  ET_CHECK_OK_OR_RETURN_ERROR(decoder_runner_->load(method_names));

  ET_LOG(Info, "Reading metadata from model");
  // retrieve any method meta, can be either prefill or kv
  int64_t num_layers =
      ET_UNWRAP(module_->get("get_n_layers")).toScalar().to<int64_t>();

  ET_CHECK_MSG(num_layers != -1, "Could not retrieve num layers");
  // k_cache: [1, n_heads, head_dim, seq_len]
  auto k_cache_shape = method_meta->output_tensor_meta(1)->sizes();
  int64_t num_heads = k_cache_shape[1];
  int64_t head_dim = k_cache_shape[2];
  bool use_int64_token = method_meta->input_tensor_meta(0)->scalar_type() ==
      executorch::aten::ScalarType::Long;

  // Use attention mask length to retrieve AR length and context length
  // Cache len equals to context_len - ar_len
  int32_t prompt_processor_ar_len = 0;
  int32_t token_generator_ar_len = 0;
  int32_t max_cache_len = 0;
  int32_t max_ar_len = 0;
  // atten mask: [1, AR-N, CL]
  auto atten_mask_meta_token = method_meta->input_tensor_meta(1);
  token_generator_ar_len = atten_mask_meta_token->sizes()[1];
  context_len_ = atten_mask_meta_token->sizes()[2];
  if (eval_mode_ == EvalMode::kKVCached) {
    prompt_processor_ar_len = token_generator_ar_len;
  } else if (
      eval_mode_ == EvalMode::kHybrid ||
      eval_mode_ == EvalMode::kLookaheadDecoding ||
      eval_mode_ == EvalMode::kEagleDecoding ||
      eval_mode_ == EvalMode::kDFlashDecoding) {
    auto atten_mask_meta_prompt =
        module_->method_meta(prompt_processor_method_name)
            ->input_tensor_meta(1);
    prompt_processor_ar_len = atten_mask_meta_prompt->sizes()[1];
  }
  if (prompt_processor_ar_len == context_len_)
    max_cache_len = context_len_;
  else
    max_cache_len = context_len_ -
        std::min(token_generator_ar_len, prompt_processor_ar_len);
  max_ar_len = std::max(token_generator_ar_len, prompt_processor_ar_len);

  // Load the sliding window size if the model supports it.
  // This is used to configure the attention mask for models with window
  // attention
  int32_t sliding_window = context_len_;
  if (module_->method_names()->count("get_sliding_window") > 0) {
    sliding_window = ET_UNWRAP(module_->get("get_sliding_window")).toInt();
  }
  kv_manager_ = std::make_unique<KVManager>(
      KVManager::Metadata{
          context_len_,
          head_dim,
          max_ar_len,
          max_cache_len,
          num_heads,
          num_layers},
      std::make_unique<MethodMeta>(
          std::move(module_->method_meta(token_generator_method_name).get())));

  if (attention_sink_rope_module_ != nullptr) {
    attention_sink_rope_runner_ = std::make_unique<AttentionSinkRopeRunner>(
        attention_sink_rope_module_.get());
    ET_CHECK_OK_OR_RETURN_ERROR(
        attention_sink_rope_runner_->load(method_names));
  }

  prompt_processor_ = std::make_unique<PromptProcessor>(
      decoder_runner_.get(),
      kv_manager_.get(),
      prompt_processor_method_name,
      PromptProcessor::Metadata{
          context_len_,
          num_heads,
          num_layers,
          prompt_processor_ar_len,
          vocab_size,
          use_int64_token,
          sliding_window,
          cache_mode_},
      std::make_unique<MethodMeta>(
          std::move(module_->method_meta(prompt_processor_method_name).get())),
      dflash ? dflash_emb_module_.get() : nullptr,
      dflash ? dflash_lm_head_module_.get() : nullptr,
      std::move(dflash_emb_prefill_meta),
      std::move(dflash_lm_prefill_meta),
      dflash_embeds_scale,
      dflash_embeds_zp,
      dflash_logits_scale,
      dflash_logits_zp);
  if (eval_mode_ == EvalMode::kLookaheadDecoding) {
    token_generator_ = std::make_unique<LhdTokenGenerator>(
        tokenizer_.get(),
        decoder_runner_.get(),
        kv_manager_.get(),
        token_generator_method_name,
        std::move(eos_ids),
        LhdTokenGenerator::Metadata{
            context_len_,
            num_heads,
            num_layers,
            token_generator_ar_len,
            vocab_size,
            use_int64_token,
            ngram_,
            window_,
            gcap_,
            sliding_window,
            cache_mode_},
        &stats_,
        std::make_unique<MethodMeta>(std::move(
            module_->method_meta(token_generator_method_name).get())));
  } else if (eval_mode_ == EvalMode::kEagleDecoding) {
    ET_CHECK_MSG(
        eagle_head_module_ != nullptr,
        "Eagle mode requires --eagle_head_path to be set.");
    // Load the head module's methods.
    ET_CHECK_OK_OR_RETURN_ERROR(eagle_head_module_->load_method("kv_forward"));
    ET_CHECK_OK_OR_RETURN_ERROR(
        eagle_head_module_->load_method("prefill_forward"));

    // Build a KV manager dedicated to the head (single layer).
    auto head_kv_meta = ET_UNWRAP(eagle_head_module_->method_meta("kv_forward"));
    auto head_prefill_meta =
        ET_UNWRAP(eagle_head_module_->method_meta("prefill_forward"));
    // head k_cache shape: [1, n_kv, head_dim, ctx-1]
    auto head_k_shape = head_kv_meta.input_tensor_meta(/*input_pos+kv*/ 4)->sizes();
    int64_t head_n_kv = head_k_shape[1];
    int64_t head_head_dim = head_k_shape[2];
    int64_t head_max_cache_len = head_k_shape[3];
    auto head_kv_out_shape = head_kv_meta.output_tensor_meta(/*k_out*/ 2)->sizes();
    auto head_prefill_out_shape =
        head_prefill_meta.output_tensor_meta(/*k_out*/ 2)->sizes();
    int32_t head_kv_ar_len = static_cast<int32_t>(
        head_kv_out_shape[head_kv_out_shape.size() - 1]);
    int32_t head_prefill_ar_len = static_cast<int32_t>(
        head_prefill_out_shape[head_prefill_out_shape.size() - 1]);
    int32_t head_max_ar_len = std::max(head_kv_ar_len, head_prefill_ar_len);
    ET_LOG(
        Info,
        "[Eagle] head compiled ar: kv=%d prefill=%d",
        head_kv_ar_len,
        head_prefill_ar_len);

    eagle_kv_manager_ = std::make_unique<KVManager>(
        KVManager::Metadata{
            context_len_,
            head_head_dim,
            /*max_ar_len=*/head_max_ar_len,
            static_cast<int32_t>(head_max_cache_len),
            head_n_kv,
            /*num_layers=*/1},
        std::make_unique<MethodMeta>(std::move(head_kv_meta)));

    // Load d2t mapping from sibling .bin file (populated from --eagle_d2t_path
    // or defaulted to the same dir as the head pte by the caller).
    std::vector<int64_t> d2t;
    if (!eagle_d2t_path_.empty()) {
      std::ifstream f(eagle_d2t_path_, std::ios::binary | std::ios::ate);
      if (f.is_open()) {
        auto sz = f.tellg();
        f.seekg(0);
        size_t n = static_cast<size_t>(sz) / sizeof(int64_t);
        d2t.resize(n);
        f.read(reinterpret_cast<char*>(d2t.data()), sz);
        ET_LOG(
            Info,
            "[Eagle] d2t loaded from %s: %zu entries",
            eagle_d2t_path_.c_str(),
            n);
      } else {
        ET_LOG(
            Error,
            "[Eagle] Cannot open d2t file: %s. Falling back to pte "
            "constant_method or identity.",
            eagle_d2t_path_.c_str());
      }
    }
    // Fallback: try head pte constant_method.
    if (d2t.empty() &&
        eagle_head_module_->method_names()->count("get_d2t") > 0) {
      auto res = eagle_head_module_->get("get_d2t");
      if (res.ok()) {
        auto& ev = res.get();
        if (ev.isTensor()) {
          auto t = ev.toTensor();
          const auto* ptr = t.const_data_ptr<int64_t>();
          d2t.assign(ptr, ptr + t.numel());
          ET_LOG(
              Info,
              "[Eagle] d2t loaded from head pte constant_method: %zu entries",
              d2t.size());
        }
      }
    }

    // Read head hyperparams from constant_methods (populated by EagleHead.get_metadata()).
    auto read_int_meta = [&](const std::string& name, int32_t fallback) -> int32_t {
      if (eagle_head_module_->method_names()->count(name) > 0) {
        auto res = eagle_head_module_->get(name);
        if (res.ok() && res.get().isInt()) {
          return static_cast<int32_t>(res.get().toInt());
        }
      }
      ET_LOG(Info, "[Eagle] constant_method '%s' not found, using fallback %d",
             name.c_str(), fallback);
      return fallback;
    };
    int32_t head_hidden_size = read_int_meta(
        "get_eagle_hidden_size",
        static_cast<int32_t>(head_head_dim * head_n_kv));
    int32_t head_draft_vocab = read_int_meta(
        "get_eagle_draft_vocab_size",
        static_cast<int32_t>(d2t.size()));
    if (head_draft_vocab == 0 && !d2t.empty()) {
      head_draft_vocab = static_cast<int32_t>(d2t.size());
    }

    // Load embed table from sibling embed.bin (option B for head embedding).
    std::vector<uint16_t> embed_table;  // fp16[vocab_size * hidden_size]
    if (!eagle_embed_path_.empty()) {
      std::ifstream ef(eagle_embed_path_, std::ios::binary | std::ios::ate);
      if (ef.is_open()) {
        auto esz = ef.tellg();
        ef.seekg(0);
        embed_table.resize(static_cast<size_t>(esz) / sizeof(uint16_t));
        ef.read(reinterpret_cast<char*>(embed_table.data()), esz);
        ET_LOG(
            Info,
            "[Eagle] embed.bin loaded from %s: %zu fp16 values",
            eagle_embed_path_.c_str(),
            embed_table.size());
      } else {
        ET_LOG(
            Error,
            "[Eagle] Cannot open embed.bin: %s. Head will not have embeddings.",
            eagle_embed_path_.c_str());
      }
    }

    EagleTokenGenerator::Metadata eagle_md{
        /*target_context_len=*/context_len_,
        /*target_num_heads=*/num_heads,
        /*target_num_layers=*/num_layers,
        /*target_ar_len=*/token_generator_ar_len,
        /*target_vocab_size=*/vocab_size,
        /*use_int64_token=*/use_int64_token,
        /*target_sliding_window=*/sliding_window,
        /*target_cache_mode=*/cache_mode_,
        /*hidden_dim=*/head_hidden_size,
        /*draft_len=*/draft_len_,
        /*draft_vocab_size=*/head_draft_vocab,
        /*head_n_kv_heads=*/static_cast<int32_t>(head_n_kv),
        /*head_head_dim=*/static_cast<int32_t>(head_head_dim),
        /*max_tree_size=*/max_tree_size_,
        /*tree_depth=*/tree_depth_,
        /*tree_topk=*/tree_topk_,
        /*low_layer_idx=*/-1,
        /*mid_layer_idx=*/-1,
        /*high_layer_idx=*/-1,
    };

    auto eagle_gen = std::make_unique<EagleTokenGenerator>(
        tokenizer_.get(),
        decoder_runner_.get(),
        kv_manager_.get(),
        eagle_head_module_.get(),
        eagle_kv_manager_.get(),
        token_generator_method_name,
        std::move(eos_ids),
        eagle_md,
        &stats_,
        std::make_unique<MethodMeta>(std::move(
            module_->method_meta(token_generator_method_name).get())),
        std::make_unique<MethodMeta>(std::move(
            eagle_head_module_->method_meta("kv_forward").get())),
        std::make_unique<MethodMeta>(std::move(head_prefill_meta)));
    eagle_gen->set_d2t(std::move(d2t));
    if (!embed_table.empty()) {
      eagle_gen->set_embed_table(
          std::move(embed_table),
          eagle_md.target_vocab_size,
          head_hidden_size);
    }
    token_generator_ = std::move(eagle_gen);
  } else if (eval_mode_ == EvalMode::kDFlashDecoding) {
    ET_CHECK_MSG(
        dflash_draft_module_ != nullptr,
        "DFlash mode requires --dflash_draft_path to be set.");
    ET_CHECK_OK_OR_RETURN_ERROR(dflash_draft_module_->load_method("kv_forward"));
    ET_CHECK_OK_OR_RETURN_ERROR(
        dflash_draft_module_->load_method("prefill_forward"));
    auto draft_kv_meta = ET_UNWRAP(dflash_draft_module_->method_meta("kv_forward"));
    auto draft_prefill_meta =
        ET_UNWRAP(dflash_draft_module_->method_meta("prefill_forward"));
    dflash_draft_io_size_ = DFlashTokenGenerator::draft_io_size_in_bytes(
        draft_kv_meta, draft_prefill_meta);

    auto read_int_meta = [&](const std::string& name, int32_t fb) -> int32_t {
      if (dflash_draft_module_->method_names()->count(name) > 0) {
        auto res = dflash_draft_module_->get(name);
        if (res.ok() && res.get().isInt()) {
          return static_cast<int32_t>(res.get().toInt());
        }
      }
      return fb;
    };
    int32_t block_size = read_int_meta("get_dflash_block_size", block_size_);
    bool shifted_decode = read_int_meta("get_dflash_shifted_decode", 0) != 0;
    bool drop_sink = read_int_meta("get_dflash_drop_sink", 0) != 0;
    int32_t hidden_dim = read_int_meta("get_dflash_hidden_size", 0);
    int32_t num_ctx_layers = read_int_meta("get_dflash_num_ctx_layers", 0);
    int32_t num_draft_layers = read_int_meta(
        "get_dflash_num_draft_layers",
        static_cast<int32_t>((draft_kv_meta.num_outputs() - 1) / 2));
    int32_t max_ctx = read_int_meta("get_dflash_max_context_len", context_len_);
    int32_t dflash_prefill_ar =
        read_int_meta("get_dflash_prefill_ar_len", prompt_processor_ar_len);
    int64_t mask_token_id = read_int_meta("get_dflash_mask_token_id", 0);

    // Cache B: the draft's own K/V, one row per committed token. Its geometry
    // comes from kv_forward's first cache input [1, nKV, head_dim, ctx - block].
    auto draft_k_shape = draft_kv_meta.input_tensor_meta(5)->sizes();
    int64_t draft_n_kv = draft_k_shape[1];
    int64_t draft_head_dim = draft_k_shape[2];
    int32_t draft_max_cache_len = static_cast<int32_t>(draft_k_shape[3]);
    ET_LOG(
        Info,
        "[DFlash] draft meta: block=%d hidden=%d ctx_layers=%d draft_layers=%d "
        "max_ctx=%d prefill_ar=%d mask=%lld | kv cache %lldx%lldx%d",
        block_size,
        hidden_dim,
        num_ctx_layers,
        num_draft_layers,
        max_ctx,
        dflash_prefill_ar,
        static_cast<long long>(mask_token_id),
        static_cast<long long>(draft_n_kv),
        static_cast<long long>(draft_head_dim),
        draft_max_cache_len);

    dflash_kv_manager_ = std::make_unique<KVManager>(
        KVManager::Metadata{
            max_ctx,
            draft_head_dim,
            /*max_ar_len=*/std::max(dflash_prefill_ar, block_size),
            draft_max_cache_len,
            draft_n_kv,
            num_draft_layers},
        std::make_unique<MethodMeta>(std::move(draft_kv_meta)));
    // The MethodMeta above was consumed; fetch a fresh one for the generator.
    auto draft_kv_meta2 = ET_UNWRAP(dflash_draft_module_->method_meta("kv_forward"));

    DFlashTokenGenerator::Metadata dflash_md{
        /*target_context_len=*/context_len_,
        /*target_num_heads=*/num_heads,
        /*target_num_layers=*/num_layers,
        /*target_ar_len=*/token_generator_ar_len,
        /*target_vocab_size=*/vocab_size,
        /*use_int64_token=*/use_int64_token,
        /*target_sliding_window=*/sliding_window,
        /*target_cache_mode=*/cache_mode_,
        /*block_size=*/block_size,
        /*hidden_dim=*/hidden_dim,
        /*num_ctx_layers=*/num_ctx_layers,
        /*num_draft_layers=*/num_draft_layers,
        /*max_context_len=*/max_ctx,
        /*prefill_ar_len=*/dflash_prefill_ar,
        /*mask_token_id=*/mask_token_id,
        /*shifted_decode=*/shifted_decode,
        /*drop_sink=*/drop_sink,
    };

    auto dflash_gen = std::make_unique<DFlashTokenGenerator>(
        tokenizer_.get(),
        decoder_runner_.get(),
        kv_manager_.get(),
        dflash_draft_module_.get(),
        dflash_kv_manager_.get(),
        token_generator_method_name,
        std::move(eos_ids),
        dflash_md,
        &stats_,
        std::make_unique<MethodMeta>(std::move(
            module_->method_meta(token_generator_method_name).get())),
        std::make_unique<MethodMeta>(std::move(draft_kv_meta2)),
        std::make_unique<MethodMeta>(std::move(draft_prefill_meta)),
        dflash_emb_module_.get(),
        dflash_lm_head_module_.get(),
        std::move(dflash_emb_kv_meta),
        std::move(dflash_lm_kv_meta),
        dflash_embeds_scale,
        dflash_embeds_zp,
        dflash_logits_scale,
        dflash_logits_zp);
    token_generator_ = std::move(dflash_gen);
  } else {
    token_generator_ = std::make_unique<TokenGenerator>(
        tokenizer_.get(),
        decoder_runner_.get(),
        kv_manager_.get(),
        token_generator_method_name,
        std::move(eos_ids),
        TokenGenerator::Metadata{
            context_len_,
            num_heads,
            num_layers,
            token_generator_ar_len,
            vocab_size,
            use_int64_token,
            sliding_window,
            cache_mode_},
        &stats_,
        std::make_unique<MethodMeta>(std::move(
            module_->method_meta(token_generator_method_name).get())));
  }

  buffer_manager_ = std::make_unique<ClientMem>();
  if (shared_buffer_) {
    // RpcMem is a bump allocator with no bounds check: every cache that draws
    // from buffer_manager must be counted here, or its buffers land outside the
    // ION region and the DSP fails the DMA (QNN 6002).
    size_t cache_size = kv_manager_->total_cache_size_in_bytes();
    if (eagle_kv_manager_ != nullptr) {
      cache_size += eagle_kv_manager_->total_cache_size_in_bytes();
    }
    if (dflash_kv_manager_ != nullptr) {
      cache_size += dflash_kv_manager_->total_cache_size_in_bytes() +
          dflash_draft_io_size_ + dflash_aux_io_size_;
    }
    buffer_manager_ = std::make_unique<RpcMem>(
        cache_size,
        prompt_processor_->total_prompt_processor_io_size_in_bytes(),
        token_generator_->total_token_generator_io_size_in_bytes());
  }
  ET_LOG(Info, "creating io_memory");
  // prepare io
  kv_manager_->init_cache(buffer_manager_.get(), prompt_processor_ar_len);
  if (eagle_kv_manager_ != nullptr) {
    eagle_kv_manager_->init_cache(buffer_manager_.get(), 1);
  }
  prompt_processor_->init_io(
      buffer_manager_.get(),
      module_->method_meta(prompt_processor_method_name));
  token_generator_->init_io(
      buffer_manager_.get(), module_->method_meta(token_generator_method_name));
  ET_LOG(
      Info,
      "RSS after io_memory init: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);
  return Error::Ok;
}

Error Runner::generate(
    const std::string& prompt,
    const llm::GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  return generate_from_prompt_or_file(
      prompt, false, config, token_callback, stats_callback);
}

Error Runner::generate_from_prompt_or_file(
    const std::string& prompt,
    bool tokenized_prompt,
    const llm::GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  ET_CHECK_MSG(!prompt.empty(), "prompt cannot be null");
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = time_in_ms();
  }
  stats_.inference_start_ms = time_in_ms();

  int32_t seq_len = config.seq_len;
  if (attention_sink_rope_runner_ == nullptr && seq_len > context_len_) {
    ET_LOG(
        Info,
        "Warning: Requested seq_len (%d) exceeds compiled max_context_len (%d) without attention sink. Clamping to %d.",
        seq_len,
        context_len_,
        context_len_);
    seq_len = context_len_;
  } else if (seq_len <= 0) {
    ET_LOG(
        Info,
        "Warning: Invalid seq_len (%d). Using compiled max_context_len (%d).",
        seq_len,
        context_len_);
    seq_len = context_len_;
  }
  int32_t n_bos = (cur_pos_ == 0) ? 1 : 0;

  // encode the (string) prompt into tokens sequence
  std::vector<uint64_t> prompt_tokens;
  if (tokenized_prompt) {
    std::ifstream inFile(prompt, std::ios::binary);
    if (inFile.is_open()) {
      // Get file size
      inFile.seekg(0, std::ios::end);
      size_t fileSize = inFile.tellg();
      inFile.seekg(0, std::ios::beg);

      // Resize vector and read raw data
      prompt_tokens.resize(fileSize / sizeof(uint64_t));

      inFile.read(reinterpret_cast<char*>(prompt_tokens.data()), fileSize);
      inFile.close();
    } else {
      ET_CHECK_MSG(
          false,
          "Unable to read tokenized prompt from file: %s",
          prompt.c_str());
    }
  } else {
    tokenizers::Result<std::vector<uint64_t>> encode_res =
        tokenizer_->encode(prompt, n_bos, 0);
    ET_CHECK_TK_OK_OR_RETURN_ERROR(
        encode_res.error(), "failed to encode prompt %s", prompt.c_str());
    prompt_tokens = encode_res.get();
  }
  int num_prompt_tokens = prompt_tokens.size();
  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  int64_t end_pos = 0;
  ET_CHECK_MSG(
      !c10::add_overflows(
          cur_pos_, static_cast<int64_t>(num_prompt_tokens), &end_pos) &&
          end_pos < static_cast<int64_t>(seq_len),
      "sequence length exceeded - please increase the seq_len value");

  // Prompt Processor first
  if (token_callback && config.echo) {
    token_callback(prompt);
  }
  bool dump_logits = dump_logits_path_.empty() ? false : true;
  // The prefill graph's hidden outputs live in one ar_len-row buffer that each
  // iteration overwrites, so DFlash must drain them per chunk to seed its draft
  // KV cache with the whole prompt rather than just the last chunk.
  DFlashTokenGenerator* dflash_generator =
      eval_mode_ == EvalMode::kDFlashDecoding
      ? dynamic_cast<DFlashTokenGenerator*>(token_generator_.get())
      : nullptr;
  if (dflash_generator != nullptr) {
    prompt_processor_->set_extra_output_observer(
        [dflash_generator](
            const std::vector<TensorStructRaw>& extra_outputs,
            int32_t n_valid,
            int64_t pos_base) {
          dflash_generator->stage_prompt_hidden(extra_outputs, n_valid, pos_base);
        });
  }
  auto prefill_res = prompt_processor_->prefill(
      prompt_tokens, cur_pos_, dump_logits, attention_sink_rope_runner_.get());
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  uint64_t cur_token = prefill_res.get();
  cur_pos_ += num_prompt_tokens;
  stats_.first_token_ms = time_in_ms();
  stats_.prompt_eval_end_ms = time_in_ms();

  // print the first token from prefill. No prev_token so use cur_token for
  // it.
  if (token_callback) {
    token_callback(
        ET_UNWRAP_TOKENIZER(tokenizer_->decode(cur_token, cur_token)));
  }
  ET_LOG(
      Info,
      "RSS after prompt prefill: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

  if (eval_mode_ == EvalMode::kEagleDecoding) {
    if (auto* eagle_generator =
            dynamic_cast<EagleTokenGenerator*>(token_generator_.get())) {
      eagle_generator->set_prompt_prefill_hidden(
          prompt_processor_->get_extra_outputs(), num_prompt_tokens);
    }
  } else if (dflash_generator != nullptr) {
    dflash_generator->finish_prompt_seeding();
  }

  // start the main loop
  prompt_tokens.push_back(cur_token);
  int64_t num_generated_tokens = ET_UNWRAP(token_generator_->generate(
      prompt_tokens,
      cur_pos_,
      seq_len,
      token_callback,
      dump_logits,
      attention_sink_rope_runner_.get()));
  stats_.inference_end_ms = time_in_ms();
  ET_LOG(
      Info,
      "RSS after finishing text generation: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);
  cur_pos_ += num_generated_tokens;
  if (cur_pos_ == seq_len) {
    ET_LOG(Info, "Sequence length (%i tokens) reached!", seq_len);
  }

  stats_.num_prompt_tokens = num_prompt_tokens;
  stats_.num_generated_tokens = num_generated_tokens;
  print_report(stats_);
  if (eval_mode_ == EvalMode::kHybrid) {
    const uint64_t prefill_calls = prompt_processor_->graph_execute_calls();
    const double prefill_time_ms = prompt_processor_->graph_execute_time_ms();
    ET_LOG(
        Info,
        "[Hybrid] profile prefill_forward: calls=%llu total=%.3fms avg=%.3fms",
        static_cast<unsigned long long>(prefill_calls),
        prefill_time_ms,
        prefill_calls == 0
            ? 0.0
            : prefill_time_ms / static_cast<double>(prefill_calls));
    const uint64_t kv_calls = token_generator_->graph_execute_calls();
    const double kv_time_ms = token_generator_->graph_execute_time_ms();
    ET_LOG(
        Info,
        "[Hybrid] profile kv_forward: calls=%llu total=%.3fms avg=%.3fms",
        static_cast<unsigned long long>(kv_calls),
        kv_time_ms,
        kv_calls == 0 ? 0.0 : kv_time_ms / static_cast<double>(kv_calls));
  }
  print_performance_report(stats_, performance_output_path_);
  if (dump_logits) {
    save_logits(
        dump_logits_path_,
        prompt_processor_->get_all_logits(),
        token_generator_->get_all_logits());
  }
  if (stats_callback) {
    stats_callback(stats_);
  }
  return Error::Ok;
}

Result<DecoderModelVersion> Runner::get_decoder_model_version() {
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = time_in_ms();
  }
  return decoder_model_version_;
}

} // namespace example
