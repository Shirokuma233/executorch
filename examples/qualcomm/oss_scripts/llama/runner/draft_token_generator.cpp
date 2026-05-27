/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/draft_token_generator.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <algorithm>
#include <numeric>

using executorch::extension::Module;
using executorch::runtime::EValue;
using executorch::runtime::Result;

namespace {

template <typename T>
uint64_t argmax_ptr(const T* data, size_t numel) {
  T max_val = data[0];
  uint64_t max_idx = 0;
  for (size_t i = 1; i < numel; ++i) {
    if (data[i] > max_val) {
      max_val = data[i];
      max_idx = static_cast<uint64_t>(i);
    }
  }
  return max_idx;
}

template <typename T>
uint64_t argmax_ptr_cast(const T* data, size_t numel) {
  float max_val = static_cast<float>(data[0]);
  uint64_t max_idx = 0;
  for (size_t i = 1; i < numel; ++i) {
    const float cur = static_cast<float>(data[i]);
    if (cur > max_val) {
      max_val = cur;
      max_idx = static_cast<uint64_t>(i);
    }
  }
  return max_idx;
}

uint64_t argmax_logits(const executorch::aten::Tensor& logits) {
  const auto numel = static_cast<size_t>(logits.numel());
  ET_CHECK_MSG(numel > 0, "DraftTokenGenerator: empty logits tensor");

  switch (logits.scalar_type()) {
    case executorch::aten::ScalarType::Float: {
      const auto* data = logits.const_data_ptr<float>();
      return argmax_ptr<float>(data, numel);
    }
    case executorch::aten::ScalarType::Half: {
      const auto* data = logits.const_data_ptr<executorch::aten::Half>();
      return argmax_ptr_cast<executorch::aten::Half>(data, numel);
    }
    case executorch::aten::ScalarType::BFloat16: {
      const auto* data = logits.const_data_ptr<executorch::aten::BFloat16>();
      return argmax_ptr_cast<executorch::aten::BFloat16>(data, numel);
    }
    default:
      ET_CHECK_MSG(false, "DraftTokenGenerator: unsupported logits dtype");
      return 0;
  }
}

} // namespace

namespace example {

template <typename T>
DraftTokenGenerator<T>::DraftTokenGenerator(
    tokenizers::Tokenizer* tokenizer,
    DecoderRunner* decoder_runner,
    KVManager<T>* kv_manager,
    const std::string& forward_name,
    std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids,
    Metadata metadata,
    executorch::llm::Stats* stats,
    std::unique_ptr<Module> draft_module)
    : TokenGenerator<T>(
          tokenizer,
          decoder_runner,
          kv_manager,
          forward_name,
          std::move(eos_ids),
          typename TokenGenerator<T>::Metadata{
              metadata.context_len,
              metadata.num_heads,
              metadata.num_layers,
              metadata.ar_len,
              metadata.vocab_size,
              metadata.use_int64_token,
              metadata.sliding_window,
              metadata.cache_mode},
          stats),
      metadata_(metadata),
      draft_module_(std::move(draft_module)) {
  ET_LOG(
      Info,
      "DraftTokenGenerator: draft_len=%d, target_ar_len=%d",
      metadata_.draft_len,
      metadata_.ar_len);

  // Allocate draft model I/O buffers (single-token, CPU)
  draft_input_toks_.resize(1, 0);
  draft_input_pos_.resize(1, 0);
  // Attention mask for draft: [1, context_len], initialised to -inf (masked)
  draft_attention_mask_.resize(metadata_.context_len, -65504.0f);

  ET_CHECK_MSG(
      draft_module_ != nullptr, "DraftTokenGenerator: draft_module is null");
  auto load_err = draft_module_->load();
  ET_CHECK_MSG(
      load_err == executorch::runtime::Error::Ok,
      "DraftTokenGenerator: failed to load draft module");
}

template <typename T>
uint64_t DraftTokenGenerator<T>::draft_step(uint64_t cur_token, int32_t pos) {
  // Update draft input buffers
  draft_input_toks_[0] = static_cast<int64_t>(cur_token);
  draft_input_pos_[0] = pos;

  // Update attention mask: unmask positions [0, pos]
  std::fill(draft_attention_mask_.begin(), draft_attention_mask_.end(), -65504.0f);
  for (int32_t i = 0; i <= pos && i < metadata_.context_len; ++i) {
    draft_attention_mask_[i] = 0.0f;
  }

  // Build EValue inputs for the draft model's "kv_forward" method.
  // The draft model has the same interface as the target kv model:
  //   inputs:  tokens[1], attn_mask[1,1,CL], pos[1], k_caches..., v_caches...
  //   outputs: logits[1,1,V], k_cache_out..., v_cache_out...
  //
  // For simplicity we use the module's execute() with named tensors.
  // We pass only the logits output and ignore KV cache outputs (the draft
  // model's KV state is managed implicitly via stateful execution).
  //
  // NOTE: This is a simplified CPU-only execution path.  The draft module
  // was exported without KV cache (use_kv_cache=False, ar_len=context_len)
  // so it takes the full token history each call.  A future optimisation
  // could export a KV-cached draft model for lower latency.

      auto tok_tensor = executorch::extension::from_blob(
      draft_input_toks_.data(), {1, 1}, executorch::aten::ScalarType::Long);
      auto mask_tensor = executorch::extension::from_blob(
      draft_attention_mask_.data(),
      {1, 1, metadata_.context_len},
      executorch::aten::ScalarType::Float);
      auto pos_tensor = executorch::extension::from_blob(
      draft_input_pos_.data(), {1}, executorch::aten::ScalarType::Int);

  auto result = draft_module_->execute(
      "kv_forward",
      {EValue(*tok_tensor), EValue(*mask_tensor), EValue(*pos_tensor)});

  ET_CHECK_MSG(result.ok(), "DraftTokenGenerator: draft_module execute failed");
  auto& outputs = result.get();
  ET_CHECK_MSG(!outputs.empty(), "DraftTokenGenerator: no outputs from draft");

  // outputs[0] is logits [1, 1, vocab_size]
  const auto logits = outputs[0].toTensor();
  // Greedy argmax over the full logits buffer (shape [1, 1, V]).
  return argmax_logits(logits);
}

template <typename T>
void DraftTokenGenerator<T>::prepare_verify_io(
    uint64_t cur_token,
    const std::vector<uint64_t>& draft_tokens,
    int32_t pos) {
  // Fill target model input: [cur_token, draft_tokens...]
  // ar_len = draft_len + 1
  int32_t n = static_cast<int32_t>(draft_tokens.size()) + 1;
  ET_CHECK_MSG(n <= metadata_.ar_len, "prepare_verify_io: n > ar_len");

  auto fill_tok = [&](int idx, uint64_t tok) {
    if (metadata_.use_int64_token) {
      this->input_toks_.data[idx] = static_cast<int64_t>(tok);
    } else {
      reinterpret_cast<int32_t*>(this->input_toks_.data)[idx] =
          static_cast<int32_t>(tok);
    }
    this->input_pos_.data[idx] = pos + idx;
  };

  fill_tok(0, cur_token);
  for (int i = 0; i < static_cast<int>(draft_tokens.size()); ++i) {
    fill_tok(i + 1, draft_tokens[i]);
  }
}

template <typename T>
Result<int64_t> DraftTokenGenerator<T>::generate(
    std::vector<uint64_t> tokens,
    int64_t start_pos,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    bool dump_logits,
    AttentionSinkRopeRunner* attention_sink_rope_runner) {
  ET_CHECK_MSG(
      !tokens.empty(), "DraftTokenGenerator: empty tokens");

  int64_t pos = start_pos;
  int32_t shifted_pos = static_cast<int32_t>(start_pos);
  bool enable_attention_sink = attention_sink_rope_runner != nullptr;

  uint64_t cur_token = tokens.back();
  uint64_t prev_token;

  int64_t n_accepted_total = 0;
  int64_t n_draft_total = 0;

  // Rearrange KV cache and set up attention mask (same as TokenGenerator)
  this->kv_manager_->rearrange_cache(metadata_.ar_len);
  std::vector<int32_t> attention_map(metadata_.ar_len);
  std::iota(attention_map.begin(), attention_map.end(), -1);

  if (enable_attention_sink) {
    ET_CHECK_MSG(
        attention_sink_rope_runner->set_outputs(
            this->method_name_, this->cache_inputs_) ==
            executorch::runtime::Error::Ok,
        "DraftTokenGenerator: failed to set attention sink outputs");
    shifted_pos =
        static_cast<int32_t>(pos) -
        attention_sink_rope_runner->get_position_shift();
  }

  this->kv_manager_->init_attention_mask(
      this->attention_mask_.data, attention_map, metadata_.ar_len, shifted_pos);
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    this->kv_manager_->init_attention_mask(
        this->window_attention_mask_.data,
        attention_map,
        metadata_.ar_len,
        shifted_pos,
        metadata_.sliding_window);
  }

  ET_CHECK_MSG(
      this->decoder_runner_->set_outputs(
          this->method_name_, this->output_tensors_) ==
          executorch::runtime::Error::Ok,
      "DraftTokenGenerator: failed to set target outputs");

  while (pos < seq_len - 1) {
    // --- Attention sink eviction ---
    if (enable_attention_sink &&
        shifted_pos + metadata_.ar_len >
            metadata_.context_len - metadata_.ar_len) {
      attention_sink_rope_runner->evict_token(
          this->method_name_, this->cache_inputs_);
      shifted_pos -=
          attention_sink_rope_runner->get_eviction_batch_size();
      this->kv_manager_->init_attention_mask(
          this->attention_mask_.data,
          attention_map,
          metadata_.ar_len,
          shifted_pos);
      if (metadata_.cache_mode == CacheMode::HybridCache) {
        this->kv_manager_->init_attention_mask(
            this->window_attention_mask_.data,
            attention_map,
            metadata_.ar_len,
            shifted_pos,
            metadata_.sliding_window);
      }
    }

    // --- Draft phase: generate draft_len tokens on CPU ---
    std::vector<uint64_t> draft_tokens;
    draft_tokens.reserve(metadata_.draft_len);
    uint64_t draft_cur = cur_token;
    for (int d = 0; d < metadata_.draft_len && pos + d < seq_len - 1; ++d) {
      draft_cur = draft_step(draft_cur, shifted_pos + d);
      draft_tokens.push_back(draft_cur);
      n_draft_total++;
    }

    // --- Verify phase: target model checks cur_token + draft_tokens ---
    prepare_verify_io(cur_token, draft_tokens, shifted_pos);

    auto logits_res =
        this->decoder_runner_->step(this->method_name_, this->inputs_);
    ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
    executorch::aten::Tensor& logits_tensor = logits_res.get();

    // --- Accept/reject loop ---
    // Position 0 in logits corresponds to cur_token's prediction.
    // Position i+1 corresponds to draft_tokens[i]'s prediction.
    int32_t n_accepted = 0;
    std::vector<bool> selected(metadata_.ar_len, false);

    // Always accept the token predicted at position 0 (cur_token's output)
    selected[0] = true;
    this->stats_->on_sampling_begin();
    uint64_t verified_token =
        this->decoder_runner_->logits_to_token(logits_tensor, 0);
    this->stats_->on_sampling_end();

    prev_token = cur_token;
    cur_token = verified_token;
    pos++;
    shifted_pos++;
    n_accepted++;

    token_callback(
        ET_UNWRAP_TOKENIZER(this->tokenizer_->decode(prev_token, cur_token)));

    if (this->eos_ids_->count(cur_token) > 0) {
      printf("\n");
      ET_LOG(Info, "\nReached end of generation (EOS after verify pos 0)");
      break;
    }

    // Check each draft token against the target's prediction
    for (int d = 0;
         d < static_cast<int>(draft_tokens.size()) && pos < seq_len - 1;
         ++d) {
      this->stats_->on_sampling_begin();
      uint64_t target_tok =
          this->decoder_runner_->logits_to_token(logits_tensor, d + 1);
      this->stats_->on_sampling_end();

      if (target_tok == draft_tokens[d]) {
        // Accept
        selected[d + 1] = true;
        prev_token = cur_token;
        cur_token = target_tok;
        pos++;
        shifted_pos++;
        n_accepted++;
        n_accepted_total++;

        token_callback(ET_UNWRAP_TOKENIZER(
            this->tokenizer_->decode(prev_token, cur_token)));

        if (this->eos_ids_->count(cur_token) > 0) {
          printf("\n");
          ET_LOG(Info, "\nReached end of generation (EOS in draft accept)");
          goto done;
        }
      } else {
        // Reject: use target's token and restart draft
        prev_token = cur_token;
        cur_token = target_tok;
        pos++;
        shifted_pos++;

        token_callback(ET_UNWRAP_TOKENIZER(
            this->tokenizer_->decode(prev_token, cur_token)));

        if (this->eos_ids_->count(cur_token) > 0) {
          printf("\n");
          ET_LOG(Info, "\nReached end of generation (EOS after reject)");
          goto done;
        }
        break;
      }
    }

    // Update KV cache with accepted positions
    int32_t n_update = shifted_pos - static_cast<int32_t>(start_pos + (pos - start_pos - n_accepted));
    // Simpler: n_update = number of positions we advanced this iteration
    n_update = n_accepted;
    this->kv_manager_->update_cache(
        metadata_.ar_len,
        shifted_pos - n_update,
        n_update,
        selected);
    this->kv_manager_->update_attention_mask(
        this->attention_mask_.data,
        metadata_.ar_len,
        shifted_pos - n_update,
        n_update);
    if (metadata_.cache_mode == CacheMode::HybridCache) {
      this->kv_manager_->update_attention_mask(
          this->window_attention_mask_.data,
          metadata_.ar_len,
          shifted_pos - n_update,
          n_update,
          metadata_.sliding_window);
    }

    if (!enable_attention_sink &&
        pos > metadata_.context_len - metadata_.ar_len) {
      printf("\n");
      ET_LOG(Info, "\nReached maximum sequence length");
      break;
    }
  }

done:
  ET_LOG(
      Info,
      "Speculative Decoding: n_generated=%ld, n_draft=%ld, n_accepted=%ld, accept_rate=%.2f%%",
      pos - start_pos,
      n_draft_total,
      n_accepted_total,
      n_draft_total > 0
          ? 100.0 * n_accepted_total / n_draft_total
          : 0.0);

  return pos - start_pos;
}

// Explicit instantiations
template class DraftTokenGenerator<uint16_t>;
template class DraftTokenGenerator<uint8_t>;

} // namespace example
