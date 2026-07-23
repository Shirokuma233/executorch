/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/safe_numerics.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.h>
#include <executorch/extension/llm/runner/util.h>
#include <numeric>
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::llm::time_in_ms;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;
namespace example {

namespace {
// Wrap an externally owned buffer as a Tensor matching `info`.
inline TensorImpl make_impl(const TensorInfo& info, void* data) {
  return TensorImpl(
      info.scalar_type(),
      info.sizes().size(),
      const_cast<TensorImpl::SizesType*>(info.sizes().data()),
      data,
      const_cast<TensorImpl::DimOrderType*>(info.dim_order().data()));
}
} // namespace

PromptProcessor::PromptProcessor(
    DecoderRunner* decoder_runner,
    KVManager* kv_manager,
    const std::string& method_name,
    Metadata metadata,
    std::unique_ptr<MethodMeta> method_meta,
    executorch::extension::Module* emb_module,
    executorch::extension::Module* lm_head_module,
    std::unique_ptr<MethodMeta> emb_prefill_meta,
    std::unique_ptr<MethodMeta> lm_head_prefill_meta,
    float embeds_scale,
    int32_t embeds_zero_point,
    float logits_scale,
    int32_t logits_zero_point)
    : decoder_runner_(decoder_runner),
      kv_manager_(kv_manager),
      method_name_(method_name),
      metadata_(metadata),
      emb_module_(emb_module),
      lm_head_module_(lm_head_module),
      emb_prefill_meta_(std::move(emb_prefill_meta)),
      lm_head_prefill_meta_(std::move(lm_head_prefill_meta)),
      embeds_scale_(embeds_scale),
      embeds_zero_point_(embeds_zero_point),
      logits_scale_(logits_scale),
      logits_zero_point_(logits_zero_point) {
  k_cache_in_.resize(metadata_.num_layers);
  v_cache_in_.resize(metadata_.num_layers);
  k_cache_out_.resize(metadata_.num_layers);
  v_cache_out_.resize(metadata_.num_layers);
  // Calculate I/O size
  Result<TensorInfo> attention_mask = method_meta->input_tensor_meta(1);
  Result<TensorInfo> logits = method_meta->output_tensor_meta(0);
  input_toks_.size = metadata_.ar_len * sizeof(int64_t);
  if (is_bert()) {
    input_pos_.size = 0;
  } else {
    input_pos_.size = metadata_.ar_len * sizeof(int32_t);
  }

  attention_mask_.dtype = attention_mask->scalar_type();
  attention_mask_.size = metadata_.ar_len * metadata_.context_len *
      attention_mask_.getElementSize();
  switch (metadata_.cache_mode) {
    case CacheMode::StaticCahce:
      window_attention_mask_.size = 0;
      break;
    case CacheMode::HybridCache: {
      Result<TensorInfo> window_attention_mask =
          method_meta->input_tensor_meta(2);
      window_attention_mask_.dtype = window_attention_mask->scalar_type();
      window_attention_mask_.size = metadata_.ar_len * metadata_.context_len *
          window_attention_mask_.getElementSize();
      break;
    }
    default:
      ET_CHECK_MSG(false, "Unsupported llama cache mode");
      break;
  }

  logits_.dtype = logits->scalar_type();
  logits_.size =
      metadata_.ar_len * metadata_.vocab_size * logits_.getElementSize();

  size_t first_extra_output = 1;
  if (!is_bert()) {
    first_extra_output += 2 * static_cast<size_t>(metadata_.num_layers);
  }
  if (method_meta->num_outputs() > first_extra_output) {
    extra_outputs_.reserve(method_meta->num_outputs() - first_extra_output);
  }
  for (size_t output_idx = first_extra_output;
       output_idx < method_meta->num_outputs();
       ++output_idx) {
    Result<TensorInfo> extra_output = method_meta->output_tensor_meta(output_idx);
    TensorStructRaw extra;
    extra.dtype = extra_output->scalar_type();
    extra.size = extra.getElementSize();
    for (const auto dim : extra_output->sizes()) {
      extra.size *= static_cast<size_t>(dim);
    }
    extra_outputs_size_ += extra.size;
    extra_outputs_.emplace_back(std::move(extra));
  }

  // Headless technique. When the decoder is headless (DFlash), input[0] is embeds
  // (u16) and output[0] is hidden (u16), not tokens/logits. init_io allocates
  // input[0] with input_toks_.size bytes and output[0] with logits_.size bytes
  // while taking dtype/shape from method_meta, so resizing just these two byte
  // counts makes it carve correctly-sized embeds/hidden buffers. Doing it here
  // (not in init_io) also keeps the RpcMem budget from over-reserving vocab-wide.
  if (emb_module_ != nullptr) {
    input_toks_.size = method_meta->input_tensor_meta(0)->nbytes();
    logits_.size = method_meta->output_tensor_meta(0)->nbytes();
  }
};

void PromptProcessor::init_io(
    IMemAlloc* buffer_manager,
    Result<MethodMeta> method_meta) {
  size_t idx = 0;
  input_tensors_.reserve(method_meta->num_inputs());
  output_tensors_.reserve(method_meta->num_outputs());
  // [I]: input_tokens
  Result<TensorInfo> input_toks = method_meta->input_tensor_meta(idx++);
  input_toks_.data =
      reinterpret_cast<int64_t*>(buffer_manager->allocate(input_toks_.size));
  input_toks_.tensor = std::make_unique<TensorImpl>(
      input_toks->scalar_type(),
      input_toks->sizes().size(),
      const_cast<TensorImpl::SizesType*>(input_toks->sizes().data()),
      input_toks_.data,
      const_cast<TensorImpl::DimOrderType*>(input_toks->dim_order().data()));
  input_tensors_.emplace_back(input_toks_.tensor.get());
  buffer_manager->add_memory_info(
      input_toks_.data, input_toks_.size, input_toks.get());

  // [I]: attention_mask
  Result<TensorInfo> attention_mask = method_meta->input_tensor_meta(idx++);
  attention_mask_.data = buffer_manager->allocate(attention_mask_.size);
  attention_mask_.tensor = std::make_unique<TensorImpl>(
      attention_mask->scalar_type(),
      attention_mask->sizes().size(),
      const_cast<TensorImpl::SizesType*>(attention_mask->sizes().data()),
      attention_mask_.data,
      const_cast<TensorImpl::DimOrderType*>(
          attention_mask->dim_order().data()));
  input_tensors_.emplace_back(attention_mask_.tensor.get());
  buffer_manager->add_memory_info(
      attention_mask_.data, attention_mask_.size, attention_mask.get());

  // [I]: sliding window attention_mask
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    Result<TensorInfo> window_attention_mask =
        method_meta->input_tensor_meta(idx++);
    window_attention_mask_.data =
        buffer_manager->allocate(window_attention_mask_.size);
    window_attention_mask_.tensor = std::make_unique<TensorImpl>(
        window_attention_mask->scalar_type(),
        window_attention_mask->sizes().size(),
        const_cast<TensorImpl::SizesType*>(
            window_attention_mask->sizes().data()),
        window_attention_mask_.data,
        const_cast<TensorImpl::DimOrderType*>(
            window_attention_mask->dim_order().data()));
    input_tensors_.emplace_back(window_attention_mask_.tensor.get());
    buffer_manager->add_memory_info(
        window_attention_mask_.data,
        window_attention_mask_.size,
        window_attention_mask.get());
  }

  if (!is_bert()) {
    // [I]: input_pos
    Result<TensorInfo> input_pos = method_meta->input_tensor_meta(idx++);
    input_pos_.data =
        reinterpret_cast<int32_t*>(buffer_manager->allocate(input_pos_.size));
    input_pos_.tensor = std::make_unique<TensorImpl>(
        input_pos->scalar_type(),
        input_pos->sizes().size(),
        const_cast<TensorImpl::SizesType*>(input_pos->sizes().data()),
        input_pos_.data,
        const_cast<TensorImpl::DimOrderType*>(input_pos->dim_order().data()));
    input_tensors_.emplace_back(input_pos_.tensor.get());
    buffer_manager->add_memory_info(
        input_pos_.data, input_pos_.size, input_pos.get());

    // [I] kv_cache
    // Prepare the vector of EValue for kv cache to evict token
    cache_inputs_.reserve(2 * metadata_.num_layers);
    size_t index = idx; // bypass input_tokens, atten_mask, input_pos
    for (int cache_group = 0; cache_group < 2; ++cache_group) {
      std::vector<std::unique_ptr<TensorImpl>>& cache =
          (cache_group == 0 ? k_cache_in_ : v_cache_in_);
      std::vector<KVCache> cache_ptrs = (cache_group == 0)
          ? kv_manager_->get_k_cache_()
          : kv_manager_->get_v_cache_();
      for (int layer = 0; layer < metadata_.num_layers; ++layer, ++index) {
        Result<TensorInfo> kv_cache = method_meta->input_tensor_meta(index);

        cache[layer] = std::make_unique<TensorImpl>(
            kv_cache->scalar_type(),
            kv_cache->sizes().size(),
            const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
            cache_ptrs[layer].buffer,
            const_cast<TensorImpl::DimOrderType*>(
                kv_cache->dim_order().data()));
        input_tensors_.emplace_back(cache[layer].get());
        cache_inputs_.emplace_back(input_tensors_.back());
        buffer_manager->add_memory_info(
            cache_ptrs[layer].buffer, cache[layer]->nbytes(), kv_cache.get());
      }
    }
  }

  // [O]: logits
  Result<TensorInfo> logits = method_meta->output_tensor_meta(0);
  logits_.data = buffer_manager->allocate(logits_.size);
  logits_.tensor = std::make_unique<TensorImpl>(
      logits->scalar_type(),
      logits->sizes().size(),
      const_cast<TensorImpl::SizesType*>(logits->sizes().data()),
      logits_.data,
      const_cast<TensorImpl::DimOrderType*>(logits->dim_order().data()));
  output_tensors_.emplace_back(logits_.tensor.get());
  buffer_manager->add_memory_info(logits_.data, logits_.size, logits.get());

  // [O] kv_cache
  size_t index = 1;
  for (int cache_group = 0; cache_group < 2; ++cache_group) {
    std::vector<std::unique_ptr<TensorImpl>>& cache =
        (cache_group == 0 ? k_cache_out_ : v_cache_out_);
    std::vector<KVCache> cache_ptrs = (cache_group == 0)
        ? kv_manager_->get_k_cache_()
        : kv_manager_->get_v_cache_();
    for (int layer = 0; layer < metadata_.num_layers; ++layer, ++index) {
      Result<TensorInfo> kv_cache = method_meta->output_tensor_meta(index);
      cache[layer] = std::make_unique<TensorImpl>(
          kv_cache->scalar_type(),
          kv_cache->sizes().size(),
          const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
          cache_ptrs[layer].output_buffer,
          const_cast<TensorImpl::DimOrderType*>(kv_cache->dim_order().data()));
      output_tensors_.emplace_back(cache[layer].get());
      buffer_manager->add_memory_info(
          cache_ptrs[layer].output_buffer,
          cache[layer]->nbytes(),
          kv_cache.get());
    }
  }

  for (auto& extra_output : extra_outputs_) {
    Result<TensorInfo> extra_meta = method_meta->output_tensor_meta(index++);
    extra_output.data = buffer_manager->allocate(extra_output.size);
    extra_output.tensor = std::make_unique<TensorImpl>(
        extra_meta->scalar_type(),
        extra_meta->sizes().size(),
        const_cast<TensorImpl::SizesType*>(extra_meta->sizes().data()),
        extra_output.data,
        const_cast<TensorImpl::DimOrderType*>(
            extra_meta->dim_order().data()));
    output_tensors_.emplace_back(extra_output.tensor.get());
    buffer_manager->add_memory_info(
        extra_output.data, extra_output.size, extra_meta.get());
  }

  // DFlash headless companions: emb.pte token input + f32 embeds output, lm_head
  // f32 logits output. lm_head's input is the decoder hidden buffer (logits_.data)
  // directly. All carved from the shared region + QNN-registered, sized for the
  // prefill views this processor binds.
  if (emb_module_ != nullptr) {
    emb_tok_i64_ =
        emb_prefill_meta_->input_tensor_meta(0)->scalar_type() ==
        executorch::aten::ScalarType::Long;
    emb_tok_nbytes_ = emb_prefill_meta_->input_tensor_meta(0)->nbytes();
    emb_tok_buf_ = buffer_manager->allocate(emb_tok_nbytes_);
    buffer_manager->add_memory_info(
        emb_tok_buf_,
        emb_tok_nbytes_,
        emb_prefill_meta_->input_tensor_meta(0).get());

    emb_out_nbytes_ = emb_prefill_meta_->output_tensor_meta(0)->nbytes();
    emb_out_buf_ = buffer_manager->allocate(emb_out_nbytes_);
    buffer_manager->add_memory_info(
        emb_out_buf_,
        emb_out_nbytes_,
        emb_prefill_meta_->output_tensor_meta(0).get());

    lm_head_logits_nbytes_ =
        lm_head_prefill_meta_->output_tensor_meta(0)->nbytes();
    lm_head_logits_buf_ = buffer_manager->allocate(lm_head_logits_nbytes_);
    buffer_manager->add_memory_info(
        lm_head_logits_buf_,
        lm_head_logits_nbytes_,
        lm_head_prefill_meta_->output_tensor_meta(0).get());

    lm_head_vocab_size_ = static_cast<int32_t>(
        lm_head_prefill_meta_->output_tensor_meta(0)->sizes()[2]);
  }

  // Prepare the vector of EValue to run inference
  inputs_.reserve(input_tensors_.size());
  for (auto& input_tensor : input_tensors_) {
    inputs_.emplace_back(std::move(input_tensor));
  }
}

const std::vector<std::byte>& PromptProcessor::get_all_logits() {
  return prompt_all_logits_;
}

void PromptProcessor::prepare_io(
    const std::vector<uint64_t>& prompt_tokens,
    int64_t prompt_pos,
    int64_t start_pos) {
  const bool dflash = emb_module_ != nullptr;
  if (dflash) {
    // Padded slots must embed token 0, so clear before writing the valid ids.
    std::memset(emb_tok_buf_, 0, emb_tok_nbytes_);
  }
  for (int i = 0; i < metadata_.ar_len; i++) {
    if (!is_bert()) {
      // Prepare pos data
      input_pos_.data[i] = start_pos + i;
    }

    // Prepare input token data
    if (prompt_pos + i < prompt_tokens.size()) {
      const uint64_t tok = prompt_tokens[prompt_pos + i];
      if (dflash) {
        // Headless decoder: token ids feed emb.pte, not the decoder directly.
        if (emb_tok_i64_) {
          reinterpret_cast<int64_t*>(emb_tok_buf_)[i] =
              static_cast<int64_t>(tok);
        } else {
          reinterpret_cast<int32_t*>(emb_tok_buf_)[i] =
              static_cast<int32_t>(tok);
        }
      } else if (metadata_.use_int64_token) {
        // Support CPU 4-bit embedding, which requires int64 input.
        // However, for QNN embedding, only int32 input is needed.
        // Therefore, we need to cast to the correct type to write the data.
        input_toks_.data[i] = tok;
      } else {
        int32_t* input_toks_ptr = reinterpret_cast<int32_t*>(input_toks_.data);
        input_toks_ptr[i] = static_cast<int32_t>(tok);
      }
    }
  }
}

Result<uint64_t> PromptProcessor::prefill(
    std::vector<uint64_t> prompt_tokens,
    int64_t start_pos,
    bool dump_logits,
    AttentionSinkRopeRunner* attention_sink_rope_runner) {
  ET_CHECK_MSG(!prompt_tokens.empty(), "Prompt cannot be null");

  int64_t shifted_pos = start_pos;
  bool enable_attention_sink = attention_sink_rope_runner != nullptr;

  // Calculate number of blocks
  int32_t num_prompt_tokens = prompt_tokens.size();
  if (is_bert()) {
    ET_CHECK_MSG(
        start_pos == 0, "Bert model doesn't support multi-turn conversation.");
  } else if (!enable_attention_sink) {
    int64_t end_pos = 0;
    ET_CHECK_MSG(
        !c10::add_overflows(
            start_pos, static_cast<int64_t>(num_prompt_tokens), &end_pos) &&
            end_pos <= static_cast<int64_t>(metadata_.context_len) -
                    static_cast<int64_t>(metadata_.ar_len),
        "The sequence length exceeds the maximum limit that the prompt processor can handle.");
  }

  // store the token
  int64_t cur_token;
  int64_t prompt_pos = 0;
  int32_t n_update = metadata_.ar_len;
  int num_iters = 1 + ((num_prompt_tokens - 1) / metadata_.ar_len);
  graph_execute_calls_ = 0;
  graph_execute_time_ms_ = 0.0;
  ET_LOG(
      Info,
      "Prompt Processor: total %d prompt tokens (AR-%d * %d iters)",
      num_prompt_tokens,
      metadata_.ar_len,
      num_iters);

  // Initialize attention sink rope runner if given and update position
  // accordingly
  if (enable_attention_sink) {
    ET_CHECK_MSG(
        attention_sink_rope_runner->set_outputs(method_name_, cache_inputs_) ==
            executorch::runtime::Error::Ok,
        "Failed to set output tensor for module %s",
        method_name_.c_str());
    shifted_pos =
        shifted_pos - attention_sink_rope_runner->get_position_shift();
  }

  // Rearrange KV cache first
  kv_manager_->rearrange_cache(metadata_.ar_len);
  std::vector<int32_t> attention_map(metadata_.ar_len);
  std::iota(attention_map.begin(), attention_map.end(), -1);
  // Initialize attention mask with current position
  kv_manager_->init_attention_mask(
      attention_mask_.data, attention_map, metadata_.ar_len, shifted_pos);
  // Initialize window attention mask with current position
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    kv_manager_->init_attention_mask(
        window_attention_mask_.data,
        attention_map,
        metadata_.ar_len,
        shifted_pos,
        metadata_.sliding_window);
  }

  // Initialize the output of the module
  ET_CHECK_MSG(
      decoder_runner_->set_outputs(method_name_, output_tensors_) ==
          executorch::runtime::Error::Ok,
      "Failed to set output tensor for module %s",
      method_name_.c_str());

  for (int i = 0; i < num_iters; ++i) {
    // The current position plus the future generated cache exceeds the cache
    // size, which means we need to remove eviction_batch_size key-value cache
    // entries to make room for new tokens.
    if (enable_attention_sink &&
        shifted_pos + metadata_.ar_len >
            metadata_.context_len - metadata_.ar_len) {
      attention_sink_rope_runner->evict_token(method_name_, cache_inputs_);
      shifted_pos =
          shifted_pos - attention_sink_rope_runner->get_eviction_batch_size();
      // Initialize attention mask with current position
      kv_manager_->init_attention_mask(
          attention_mask_.data, attention_map, metadata_.ar_len, shifted_pos);
      // Initialize window attention mask with current position
      if (metadata_.cache_mode == CacheMode::HybridCache) {
        kv_manager_->init_attention_mask(
            window_attention_mask_.data,
            attention_map,
            metadata_.ar_len,
            shifted_pos,
            metadata_.sliding_window);
      }
    }

    // Fill in the token and position data
    prepare_io(prompt_tokens, prompt_pos, shifted_pos);

    // Headless decoder: turn token ids into quantized embeds (inputs_[0]) via
    // emb.pte before the decoder step. No-op for every other eval mode.
    if (emb_module_ != nullptr) {
      run_embedding_prefill();
    }

    // Run inference
    const uint64_t graph_start_ms = time_in_ms();
    decoder_runner_->step(method_name_, inputs_);
    graph_execute_time_ms_ +=
        static_cast<double>(time_in_ms() - graph_start_ms);
    ++graph_execute_calls_;
    if (dump_logits) {
      prompt_all_logits_.insert(
          prompt_all_logits_.end(),
          logits_.data,
          logits_.data +
              metadata_.ar_len * metadata_.vocab_size *
                  logits_.getElementSize());
    }
    // In the last run, offset to the meaningful logits.
    if (i == num_iters - 1) {
      n_update = 1 + ((num_prompt_tokens - 1) % metadata_.ar_len);
    }
    if (extra_output_observer_) {
      extra_output_observer_(extra_outputs_, n_update, shifted_pos);
    }
    // Update KV Cache with the output results
    kv_manager_->update_cache(metadata_.ar_len, shifted_pos, n_update, {});

    // Update attention mask with current position
    kv_manager_->update_attention_mask(
        attention_mask_.data, metadata_.ar_len, shifted_pos, n_update);
    if (metadata_.cache_mode == CacheMode::HybridCache) {
      kv_manager_->update_attention_mask(
          window_attention_mask_.data,
          metadata_.ar_len,
          shifted_pos,
          n_update,
          metadata_.sliding_window);
    }
    prompt_pos += metadata_.ar_len;
    shifted_pos += metadata_.ar_len;
  }

  const int64_t last_row =
      (num_prompt_tokens + metadata_.ar_len - 1) % metadata_.ar_len;
  if (emb_module_ != nullptr) {
    // Headless: output_tensors_[0] is hidden. Project the last chunk through
    // lm_head.pte and sample the last valid row of its logits.
    run_lm_head_prefill(logits_.data);
    // lm_head.pte writes uint16 logits even though the program declares the
    // output Float. Reading them as f32 yields zeros; bind as UInt16 instead.
    // argmax is unaffected by skipping dequantization (order is preserved).
    auto lm_out_meta = lm_head_prefill_meta_->output_tensor_meta(0).get();
    std::vector<executorch::aten::TensorImpl::SizesType> lm_sizes(
        lm_out_meta.sizes().begin(), lm_out_meta.sizes().end());
    std::vector<executorch::aten::TensorImpl::DimOrderType> lm_dimo(
        lm_out_meta.dim_order().begin(), lm_out_meta.dim_order().end());
    TensorImpl lm_impl(
        executorch::aten::ScalarType::UInt16,
        lm_sizes.size(),
        lm_sizes.data(),
        lm_head_logits_buf_,
        lm_dimo.data());
    Tensor lm_logits(&lm_impl);
    const long t_smp = time_in_ms();
    cur_token = decoder_runner_->logits_to_token(lm_logits, last_row);
    sample_ms_ += static_cast<double>(time_in_ms() - t_smp);
    ET_LOG(
        Info,
        "[DFlash] prefill per-pte ms: emb=%.1f (%llu chunks) decoder=%.1f (%llu) "
        "lm_head=%.1f embeds_copy=%.1f host_argmax=%.1f | total graph=%.1f",
        emb_exec_ms_,
        static_cast<unsigned long long>(emb_calls_),
        graph_execute_time_ms_,
        static_cast<unsigned long long>(graph_execute_calls_),
        lm_head_exec_ms_,
        embeds_copy_ms_,
        sample_ms_,
        emb_exec_ms_ + graph_execute_time_ms_ + lm_head_exec_ms_ +
            embeds_copy_ms_ + sample_ms_);
  } else {
    cur_token = decoder_runner_->logits_to_token(output_tensors_[0], last_row);
  }
  return cur_token;
}

void PromptProcessor::run_embedding_prefill() {
  auto tok_meta = emb_prefill_meta_->input_tensor_meta(0).get();
  auto out_meta = emb_prefill_meta_->output_tensor_meta(0).get();
  TensorImpl tok_impl = make_impl(tok_meta, emb_tok_buf_);
  TensorImpl out_impl = make_impl(out_meta, emb_out_buf_);
  std::vector<EValue> ins{EValue(Tensor(&tok_impl))};
  std::vector<EValue> outs{EValue(Tensor(&out_impl))};
  ET_CHECK_MSG(
      emb_module_->set_output(
          "tok_embedding_prefill_forward", outs[0], 0) == Error::Ok,
      "[DFlash] emb prefill set_output failed");
  const long t_emb = time_in_ms();
  auto res = emb_module_->execute("tok_embedding_prefill_forward", ins);
  emb_exec_ms_ += static_cast<double>(time_in_ms() - t_emb);
  ++emb_calls_;
  ET_CHECK_MSG(res.ok(), "[DFlash] emb prefill execute failed");
  // emb.pte's output is ALREADY uint16 in the decoder's own embeds encoding
  // (scale/zp injected at compile time), even though the program declares the
  // tensor Float. So this boundary is a lossless uint16 hand-off -- copy the
  // bytes straight into the decoder's embeds input; quantizing here would
  // reinterpret u16 payload as f32 and destroy it.
  const long t_cp = time_in_ms();
  std::memcpy(input_toks_.data, emb_out_buf_, input_toks_.size);
  embeds_copy_ms_ += static_cast<double>(time_in_ms() - t_cp);
}

void PromptProcessor::run_lm_head_prefill(std::byte* hidden_u16) {
  auto in_meta = lm_head_prefill_meta_->input_tensor_meta(0).get();
  auto out_meta = lm_head_prefill_meta_->output_tensor_meta(0).get();
  TensorImpl in_impl = make_impl(in_meta, hidden_u16);
  TensorImpl out_impl = make_impl(out_meta, lm_head_logits_buf_);
  std::vector<EValue> ins{EValue(Tensor(&in_impl))};
  std::vector<EValue> outs{EValue(Tensor(&out_impl))};
  ET_CHECK_MSG(
      lm_head_module_->set_outputs("lm_head_prefill_forward", outs) == Error::Ok,
      "[DFlash] lm_head prefill set_outputs failed");
  const long t_lm = time_in_ms();
  auto res = lm_head_module_->execute("lm_head_prefill_forward", ins);
  lm_head_exec_ms_ += static_cast<double>(time_in_ms() - t_lm);
  ET_CHECK_MSG(res.ok(), "[DFlash] lm_head prefill execute failed");
}

} // namespace example
