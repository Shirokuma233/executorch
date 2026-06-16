/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// EagleTokenGenerator — Phase 3 chain-mode speculative decoding.

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/eagle_token_generator.h>

#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_set>

using executorch::aten::ScalarType;
using executorch::aten::TensorImpl;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;
using executorch::extension::llm::time_in_ms;

namespace example {

// ---------------------------------------------------------------------------
// fp16 helpers
// ---------------------------------------------------------------------------
namespace {

// Convert a raw uint16 (IEEE 754 fp16) to float32.
inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1u;
  uint32_t exp  = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign << 31;
    } else {
      exp = 1;
      while (!(mant & 0x400)) { mant <<= 1; exp--; }
      mant &= 0x3ff;
      f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = (sign << 31) | 0x7f800000u | (mant << 13);
  } else {
    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float result;
  memcpy(&result, &f, sizeof(result));
  return result;
}

// Convert float32 to IEEE 754 fp16 uint16.
inline uint16_t fp32_to_fp16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(bits));
  uint16_t sign = (bits >> 31) & 0x1u;
  int32_t  exp  = ((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = bits & 0x7fffffu;
  if (exp <= 0) {
    return sign << 15;
  } else if (exp >= 31) {
    return (sign << 15) | 0x7c00u;
  }
  return static_cast<uint16_t>(
      (sign << 15) | (static_cast<uint16_t>(exp) << 10) | (mant >> 13));
}

inline float read_hidden_scalar(
    const std::byte* data,
    ScalarType dtype,
    size_t index) {
  switch (dtype) {
    case ScalarType::Float:
      return reinterpret_cast<const float*>(data)[index];
    case ScalarType::Half:
    case ScalarType::UInt16:
      return fp16_to_fp32(reinterpret_cast<const uint16_t*>(data)[index]);
    default:
      ET_CHECK_MSG(false, "[Eagle] unsupported hidden dtype");
      return 0.0f;
  }
}

inline float read_hidden_scalar(
    const std::vector<std::byte>& buffer,
    ScalarType dtype,
    size_t index) {
  return read_hidden_scalar(buffer.data(), dtype, index);
}

inline size_t dtype_size(ScalarType dtype) {
  switch (dtype) {
    case ScalarType::Float:
      return sizeof(float);
    case ScalarType::Half:
    case ScalarType::UInt16:
      return sizeof(uint16_t);
    case ScalarType::Int:
      return sizeof(int32_t);
    case ScalarType::Long:
      return sizeof(int64_t);
    case ScalarType::Byte:
      return sizeof(uint8_t);
    default:
      ET_CHECK_MSG(false, "[Eagle] unsupported hidden dtype");
      return 0;
  }
}

inline void write_float_value(std::byte* dst, ScalarType dtype, size_t index, float value) {
  switch (dtype) {
    case ScalarType::Float:
      reinterpret_cast<float*>(dst)[index] = value;
      break;
    case ScalarType::Half:
    case ScalarType::UInt16:
      reinterpret_cast<uint16_t*>(dst)[index] = fp32_to_fp16(value);
      break;
    default:
      ET_CHECK_MSG(false, "[Eagle] unsupported head IO dtype");
      break;
  }
}

inline void write_fp16_value(
    std::byte* dst,
    ScalarType dtype,
    size_t index,
    uint16_t value) {
  switch (dtype) {
    case ScalarType::Float:
      reinterpret_cast<float*>(dst)[index] = fp16_to_fp32(value);
      break;
    case ScalarType::Half:
    case ScalarType::UInt16:
      reinterpret_cast<uint16_t*>(dst)[index] = value;
      break;
    default:
      ET_CHECK_MSG(false, "[Eagle] unsupported head IO dtype");
      break;
  }
}

inline void write_mask_value(
    std::byte* dst,
    ScalarType dtype,
    size_t index,
    bool visible) {
  switch (dtype) {
    case ScalarType::Float:
      reinterpret_cast<float*>(dst)[index] = visible ? 0.0f : -65535.0f;
      break;
    case ScalarType::Half:
    case ScalarType::UInt16:
      reinterpret_cast<uint16_t*>(dst)[index] = visible ? 0x0000 : 0xFC00;
      break;
    default:
      ET_CHECK_MSG(false, "[Eagle] unsupported head mask dtype");
      break;
  }
}

inline size_t tensor_nbytes(const TensorInfo& info) {
  size_t nbytes = dtype_size(info.scalar_type());
  for (const auto dim : info.sizes()) {
    nbytes *= static_cast<size_t>(dim);
  }
  return nbytes;
}

} // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
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
      eagle_meta_(metadata) {
  ET_LOG(
      Info,
      "[Eagle] constructed: target_ar_len=%d draft_len=%d "
      "tree_depth=%d tree_topk=%d max_tree_size=%d "
      "draft_vocab=%d hidden_dim=%d",
      eagle_meta_.target_ar_len,
      eagle_meta_.draft_len,
      eagle_meta_.tree_depth,
      eagle_meta_.tree_topk,
      eagle_meta_.max_tree_size,
      eagle_meta_.draft_vocab_size,
      eagle_meta_.hidden_dim);
}

void EagleTokenGenerator::set_prompt_prefill_hidden(
    const std::vector<TensorStructRaw>& extra_outputs,
    int32_t num_prompt_tokens) {
  prompt_hidden_LMH_.clear();
  prompt_hidden_tokens_ = 0;
  prompt_hidden_token_start_ = 0;
  if (extra_outputs.size() < 3 || num_prompt_tokens <= 0) {
    return;
  }

  const int H = eagle_meta_.hidden_dim;
  const int H3 = 3 * H;
  const size_t slots = extra_outputs[0].size /
      (static_cast<size_t>(H) * extra_outputs[0].getElementSize());
  int32_t n = std::min<int32_t>(num_prompt_tokens, static_cast<int32_t>(slots));
  if (num_prompt_tokens > static_cast<int32_t>(slots)) {
    n = 1 + ((num_prompt_tokens - 1) % static_cast<int32_t>(slots));
    prompt_hidden_token_start_ = num_prompt_tokens - n;
  }
  if (n <= 0) {
    return;
  }
  if (num_prompt_tokens > static_cast<int32_t>(slots)) {
    ET_LOG(
        Info,
        "[Eagle] prompt hidden only has %zu slots for %d prompt tokens; "
        "chain init will use the visible prefill chunk",
        slots,
        num_prompt_tokens);
  }

  prompt_hidden_LMH_.assign(
      static_cast<size_t>(n) * static_cast<size_t>(H3), 0.0f);
  for (int k = 0; k < n; ++k) {
    for (int h = 0; h < H; ++h) {
      const size_t src_index = static_cast<size_t>(k) * H + h;
      const size_t dst_base = static_cast<size_t>(k) * H3;
      prompt_hidden_LMH_[dst_base + h] = read_hidden_scalar(
          extra_outputs[0].data, extra_outputs[0].dtype, src_index);
      prompt_hidden_LMH_[dst_base + H + h] = read_hidden_scalar(
          extra_outputs[1].data, extra_outputs[1].dtype, src_index);
      prompt_hidden_LMH_[dst_base + 2 * H + h] = read_hidden_scalar(
          extra_outputs[2].data, extra_outputs[2].dtype, src_index);
    }
  }
  prompt_hidden_tokens_ = n;
}

// ---------------------------------------------------------------------------
// init_io
// ---------------------------------------------------------------------------
void EagleTokenGenerator::init_io(
    IMemAlloc* buffer_manager,
    Result<MethodMeta> method_meta) {

  // Count target outputs to detect hidden state export.
  size_t target_num_outputs = method_meta->num_outputs();
  size_t expected_kv = 1 + 2 * static_cast<size_t>(eagle_meta_.target_num_layers);
  bool target_has_hidden = (target_num_outputs >= expected_kv + 3);
  ET_LOG(
      Info,
      "[Eagle] init_io: target pte outputs=%zu (kv+logits=%zu hidden=%s)",
      target_num_outputs,
      expected_kv,
      target_has_hidden ? "yes" : "NO — head refresh disabled");

  std::array<size_t, 3> target_hidden_nbytes{};
  if (target_has_hidden) {
    size_t L = static_cast<size_t>(eagle_meta_.target_num_layers);
    for (size_t i = 0; i < 3; ++i) {
      Result<TensorInfo> hidden_meta =
          method_meta->output_tensor_meta(1 + 2 * L + i);
      target_hidden_dtypes_[i] = hidden_meta->scalar_type();
      target_hidden_sizes_[i].assign(
          hidden_meta->sizes().begin(), hidden_meta->sizes().end());
      target_hidden_dim_orders_[i].assign(
          hidden_meta->dim_order().begin(), hidden_meta->dim_order().end());
      target_hidden_nbytes[i] = dtype_size(target_hidden_dtypes_[i]);
      for (const auto dim : target_hidden_sizes_[i]) {
        target_hidden_nbytes[i] *= static_cast<size_t>(dim);
      }
    }
  }

  // Standard target IO.
  TokenGenerator::init_io(buffer_manager, std::move(method_meta));

  // ----- Target hidden output binding -----
  if (target_has_hidden) {
    target_hidden_low_buf_.resize(target_hidden_nbytes[0]);
    target_hidden_mid_buf_.resize(target_hidden_nbytes[1]);
    target_hidden_high_buf_.resize(target_hidden_nbytes[2]);

    hidden_low_impl_ = std::make_unique<TensorImpl>(
        target_hidden_dtypes_[0],
        target_hidden_sizes_[0].size(),
        target_hidden_sizes_[0].data(),
        target_hidden_low_buf_.data(),
        target_hidden_dim_orders_[0].data());
    hidden_mid_impl_ = std::make_unique<TensorImpl>(
        target_hidden_dtypes_[1],
        target_hidden_sizes_[1].size(),
        target_hidden_sizes_[1].data(),
        target_hidden_mid_buf_.data(),
        target_hidden_dim_orders_[1].data());
    hidden_high_impl_ = std::make_unique<TensorImpl>(
        target_hidden_dtypes_[2],
        target_hidden_sizes_[2].size(),
        target_hidden_sizes_[2].data(),
        target_hidden_high_buf_.data(),
        target_hidden_dim_orders_[2].data());

    // Append to output_tensors_ so decoder_runner_->set_outputs covers them.
    output_tensors_.emplace_back(hidden_low_impl_.get());
    output_tensors_.emplace_back(hidden_mid_impl_.get());
    output_tensors_.emplace_back(hidden_high_impl_.get());

    ET_LOG(
        Info,
        "[Eagle] target hidden buffers: %zuB/%zuB/%zuB",
        target_hidden_nbytes[0],
        target_hidden_nbytes[1],
        target_hidden_nbytes[2]);
  }

  // ----- Head IO buffers (host-side scratch) -----
  auto kv_prev = head_kv_method_meta_->input_tensor_meta(0).get();
  auto kv_emb = head_kv_method_meta_->input_tensor_meta(1).get();
  auto kv_mask = head_kv_method_meta_->input_tensor_meta(2).get();
  auto kv_k = head_kv_method_meta_->input_tensor_meta(4).get();
  auto kv_logits = head_kv_method_meta_->output_tensor_meta(0).get();
  auto kv_prev_out = head_kv_method_meta_->output_tensor_meta(1).get();

  head_prev_feature_dtype_ = kv_prev.scalar_type();
  head_tok_emb_dtype_ = kv_emb.scalar_type();
  head_attn_mask_dtype_ = kv_mask.scalar_type();
  head_kv_dtype_ = kv_k.scalar_type();
  head_logits_dtype_ = kv_logits.scalar_type();

  head_prev_feature_buf_.resize(
      std::max(tensor_nbytes(kv_prev), tensor_nbytes(kv_prev_out)));
  head_tok_emb_buf_.resize(tensor_nbytes(kv_emb));
  head_attn_mask_buf_.resize(tensor_nbytes(kv_mask));
  head_pos_buf_.resize(sizeof(int32_t));
  head_logits_buf_.resize(tensor_nbytes(kv_logits));

  ET_LOG(
      Info,
      "[Eagle] head IO: prev_feat=%zuB tok_emb=%zuB attn_mask=%zuB logits=%zuB",
      head_prev_feature_buf_.size(),
      head_tok_emb_buf_.size(),
      head_attn_mask_buf_.size(),
      head_logits_buf_.size());

  // ----- EagleSampler -----
  if (d2t_.empty()) {
    d2t_.assign(eagle_meta_.draft_vocab_size, 0);
    ET_LOG(Info, "[Eagle] d2t not set — using identity mapping");
  }
  sampler_ = std::make_unique<EagleSampler>(
      head_logits_dtype_ == ScalarType::Float ? EagleSampler::Dtype::kFp32
                                              : EagleSampler::Dtype::kFp16,
      eagle_meta_.draft_vocab_size,
      d2t_);
}

// ---------------------------------------------------------------------------
// Embed lookup helper — returns fp16 row for token_id into dst[hidden_dim].
// ---------------------------------------------------------------------------
void EagleTokenGenerator::lookup_embedding(uint64_t token_id, uint16_t* dst) const {
  if (embed_table_.empty() || static_cast<int64_t>(token_id) >= embed_vocab_size_) {
    memset(dst, 0, static_cast<size_t>(embed_hidden_size_) * sizeof(uint16_t));
    return;
  }
  const uint16_t* row =
      embed_table_.data() +
      static_cast<size_t>(token_id) * static_cast<size_t>(embed_hidden_size_);
  memcpy(dst, row, static_cast<size_t>(embed_hidden_size_) * sizeof(uint16_t));
}

// ---------------------------------------------------------------------------
// head_prefill_step
// ---------------------------------------------------------------------------
void EagleTokenGenerator::head_prefill_step(
    const float* hidden_LMH,
    uint64_t prev_token,
    int64_t rope_pos,
    int64_t cache_pos) {
  head_prefill_batch(hidden_LMH, &prev_token, 1, rope_pos, cache_pos);
}

void EagleTokenGenerator::head_prefill_batch(
    const float* hidden_LMH,
    const uint64_t* prev_tokens,
    int32_t valid_count,
    int64_t rope_pos,
    int64_t cache_pos) {
  if (valid_count <= 0) {
    return;
  }
  const int H = eagle_meta_.hidden_dim;
  const int H3 = 3 * H;
  auto h_meta = head_prefill_method_meta_->input_tensor_meta(0).get();
  auto e_meta = head_prefill_method_meta_->input_tensor_meta(1).get();
  auto m_meta = head_prefill_method_meta_->input_tensor_meta(2).get();
  auto p_meta = head_prefill_method_meta_->input_tensor_meta(3).get();
  auto k_meta = head_prefill_method_meta_->input_tensor_meta(4).get();
  auto v_meta = head_prefill_method_meta_->input_tensor_meta(5).get();
  const int32_t ar = static_cast<int32_t>(h_meta.sizes()[1]);
  ET_CHECK_MSG(
      valid_count <= ar,
      "[Eagle] head prefill valid_count %d exceeds compiled ar %d",
      valid_count,
      ar);

  std::vector<std::byte> hidden_buf(tensor_nbytes(h_meta));
  std::vector<std::byte> tok_emb_buf(tensor_nbytes(e_meta));
  std::vector<std::byte> attn_mask_buf(tensor_nbytes(m_meta));
  std::vector<std::byte> pos_buf(tensor_nbytes(p_meta));
  std::fill(hidden_buf.begin(), hidden_buf.end(), std::byte{0});
  std::fill(tok_emb_buf.begin(), tok_emb_buf.end(), std::byte{0});
  std::fill(pos_buf.begin(), pos_buf.end(), std::byte{0});

  for (int k = 0; k < valid_count; ++k) {
    for (int i = 0; i < H3; ++i) {
      write_float_value(
          hidden_buf.data(),
          h_meta.scalar_type(),
          static_cast<size_t>(k) * H3 + i,
          hidden_LMH[static_cast<size_t>(k) * H3 + i]);
    }
    std::vector<uint16_t> emb_fp16(static_cast<size_t>(H));
    lookup_embedding(prev_tokens[k], emb_fp16.data());
    for (int i = 0; i < H; ++i) {
      write_fp16_value(
          tok_emb_buf.data(),
          e_meta.scalar_type(),
          static_cast<size_t>(k) * H + i,
          emb_fp16[static_cast<size_t>(i)]);
    }
    reinterpret_cast<int32_t*>(pos_buf.data())[k] =
        static_cast<int32_t>(rope_pos + k);
  }
  for (int k = valid_count; k < ar; ++k) {
    reinterpret_cast<int32_t*>(pos_buf.data())[k] =
        static_cast<int32_t>(rope_pos + valid_count - 1);
  }

  std::vector<int32_t> attention_map(static_cast<size_t>(ar), -1);
  for (int k = 1; k < valid_count; ++k) {
    attention_map[static_cast<size_t>(k)] = k - 1;
  }
  head_kv_manager_->init_attention_mask(
      attn_mask_buf.data(),
      attention_map,
      ar,
      static_cast<int32_t>(cache_pos));

  head_kv_manager_->rearrange_cache(ar);
  auto k_caches = head_kv_manager_->get_k_cache_();
  auto v_caches = head_kv_manager_->get_v_cache_();

  TensorImpl h_impl(
      h_meta.scalar_type(),
      h_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(h_meta.sizes().data()),
      hidden_buf.data(),
      const_cast<TensorImpl::DimOrderType*>(h_meta.dim_order().data()));
  TensorImpl e_impl(
      e_meta.scalar_type(),
      e_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(e_meta.sizes().data()),
      tok_emb_buf.data(),
      const_cast<TensorImpl::DimOrderType*>(e_meta.dim_order().data()));
  TensorImpl m_impl(
      m_meta.scalar_type(),
      m_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(m_meta.sizes().data()),
      attn_mask_buf.data(),
      const_cast<TensorImpl::DimOrderType*>(m_meta.dim_order().data()));
  TensorImpl p_impl(
      p_meta.scalar_type(),
      p_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(p_meta.sizes().data()),
      pos_buf.data(),
      const_cast<TensorImpl::DimOrderType*>(p_meta.dim_order().data()));
  TensorImpl k_impl(
      k_meta.scalar_type(),
      k_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(k_meta.sizes().data()),
      k_caches[0].buffer,
      const_cast<TensorImpl::DimOrderType*>(k_meta.dim_order().data()));
  TensorImpl v_impl(
      v_meta.scalar_type(),
      v_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(v_meta.sizes().data()),
      v_caches[0].buffer,
      const_cast<TensorImpl::DimOrderType*>(v_meta.dim_order().data()));

  std::vector<EValue> head_inputs{
      EValue(executorch::aten::Tensor(&h_impl)),
      EValue(executorch::aten::Tensor(&e_impl)),
      EValue(executorch::aten::Tensor(&m_impl)),
      EValue(executorch::aten::Tensor(&p_impl)),
      EValue(executorch::aten::Tensor(&k_impl)),
      EValue(executorch::aten::Tensor(&v_impl)),
  };

  auto logits_meta = head_prefill_method_meta_->output_tensor_meta(0).get();
  auto pf_meta = head_prefill_method_meta_->output_tensor_meta(1).get();
  auto ko_meta = head_prefill_method_meta_->output_tensor_meta(2).get();
  auto vo_meta = head_prefill_method_meta_->output_tensor_meta(3).get();
  std::vector<std::byte> logits_buf(tensor_nbytes(logits_meta));
  std::vector<std::byte> pf_buf(tensor_nbytes(pf_meta));

  TensorImpl logits_out(
      logits_meta.scalar_type(),
      logits_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(logits_meta.sizes().data()),
      logits_buf.data(),
      const_cast<TensorImpl::DimOrderType*>(logits_meta.dim_order().data()));
  TensorImpl pf_out(
      pf_meta.scalar_type(),
      pf_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(pf_meta.sizes().data()),
      pf_buf.data(),
      const_cast<TensorImpl::DimOrderType*>(pf_meta.dim_order().data()));
  TensorImpl k_out(
      ko_meta.scalar_type(),
      ko_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(ko_meta.sizes().data()),
      k_caches[0].output_buffer,
      const_cast<TensorImpl::DimOrderType*>(ko_meta.dim_order().data()));
  TensorImpl v_out(
      vo_meta.scalar_type(),
      vo_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(vo_meta.sizes().data()),
      v_caches[0].output_buffer,
      const_cast<TensorImpl::DimOrderType*>(vo_meta.dim_order().data()));

  // Output order from _shared_block: (logits, h/prev_feature, k_new, v_new)
  std::vector<EValue> head_outputs_ev{
      EValue(executorch::aten::Tensor(&logits_out)),
      EValue(executorch::aten::Tensor(&pf_out)),
      EValue(executorch::aten::Tensor(&k_out)),
      EValue(executorch::aten::Tensor(&v_out)),
  };

  auto set_err = head_module_->set_outputs("prefill_forward", head_outputs_ev);
  ET_CHECK_MSG(
      set_err == Error::Ok,
      "[Eagle] head prefill_forward set_outputs failed");

  ET_LOG(
      Info,
      "[Eagle] exec head prefill_forward rope=%lld cache=%lld ar=%d n=%d",
      static_cast<long long>(rope_pos),
      static_cast<long long>(cache_pos),
      ar,
      valid_count);
  long start_ms = time_in_ms();
  auto res = head_module_->execute("prefill_forward", head_inputs);
  head_prefill_time_ms_ += static_cast<double>(time_in_ms() - start_ms);
  ++head_prefill_calls_;
  ET_CHECK_MSG(res.ok(), "[Eagle] head prefill_forward execute failed");

  std::vector<bool> selected(static_cast<size_t>(ar), false);
  for (int k = 0; k < valid_count; ++k) {
    selected[static_cast<size_t>(k)] = true;
  }
  head_kv_manager_->update_cache(
      ar, static_cast<int32_t>(cache_pos), valid_count, selected);

  const size_t logits_stride =
      static_cast<size_t>(eagle_meta_.draft_vocab_size) * dtype_size(logits_meta.scalar_type());
  const size_t prev_stride = static_cast<size_t>(H) * dtype_size(pf_meta.scalar_type());
  const int32_t last = valid_count - 1;
  std::memcpy(
      head_logits_buf_.data(),
      logits_buf.data() + static_cast<size_t>(last) * logits_stride,
      logits_stride);
  std::memcpy(
      head_prev_feature_buf_.data(),
      pf_buf.data() + static_cast<size_t>(last) * prev_stride,
      prev_stride);
}

// ---------------------------------------------------------------------------
// head_decode_step
// ---------------------------------------------------------------------------
uint64_t EagleTokenGenerator::head_decode_step(
    const float* prev_a,
    uint64_t prev_token,
    int64_t rope_pos,
    int64_t cache_pos,
    float* prev_a_buffer) {
  const int H = eagle_meta_.hidden_dim;
  std::vector<float> prev_batch(prev_a, prev_a + H);
  std::vector<uint64_t> token_batch{prev_token};
  std::vector<std::vector<int32_t>> visible(1);
  std::vector<float> next_batch;
  std::vector<std::vector<DraftChoice>> choices_by_parent;
  head_decode_batch(
      prev_batch,
      token_batch,
      visible,
      cache_pos,
      rope_pos,
      cache_pos,
      1,
      &next_batch,
      &choices_by_parent);
  std::memcpy(
      prev_a_buffer,
      next_batch.data(),
      static_cast<size_t>(H) * sizeof(float));
  return choices_by_parent.empty() || choices_by_parent[0].empty()
      ? 0
      : choices_by_parent[0][0].draft_id;
}

void EagleTokenGenerator::head_decode_batch(
    const std::vector<float>& prev_a_batch,
    const std::vector<uint64_t>& prev_tokens,
    const std::vector<std::vector<int32_t>>& visible_past_slots,
    int64_t stable_cache_len,
    int64_t rope_pos,
    int64_t cache_pos,
    int32_t topk,
    std::vector<float>* next_prev_batch,
    std::vector<std::vector<DraftChoice>>* choices_by_parent) {
  const int H = eagle_meta_.hidden_dim;
  auto pf_in_meta = head_kv_method_meta_->input_tensor_meta(0).get();
  auto e_meta = head_kv_method_meta_->input_tensor_meta(1).get();
  auto m_meta = head_kv_method_meta_->input_tensor_meta(2).get();
  auto p_meta = head_kv_method_meta_->input_tensor_meta(3).get();
  auto k_meta = head_kv_method_meta_->input_tensor_meta(4).get();
  auto v_meta = head_kv_method_meta_->input_tensor_meta(5).get();
  const int32_t ar = static_cast<int32_t>(pf_in_meta.sizes()[1]);
  const int32_t valid_count = static_cast<int32_t>(prev_tokens.size());
  ET_CHECK_MSG(
      valid_count <= ar,
      "[Eagle] head decode valid_count %d exceeds compiled ar %d",
      valid_count,
      ar);
  ET_CHECK_MSG(
      prev_a_batch.size() >= static_cast<size_t>(valid_count) * H,
      "[Eagle] prev_a_batch is smaller than valid_count * hidden_dim");

  std::vector<std::byte> prev_buf(tensor_nbytes(pf_in_meta));
  std::vector<std::byte> tok_emb_buf(tensor_nbytes(e_meta));
  std::vector<std::byte> attn_mask_buf(tensor_nbytes(m_meta));
  std::vector<std::byte> pos_buf(tensor_nbytes(p_meta));
  std::fill(prev_buf.begin(), prev_buf.end(), std::byte{0});
  std::fill(tok_emb_buf.begin(), tok_emb_buf.end(), std::byte{0});
  std::fill(attn_mask_buf.begin(), attn_mask_buf.end(), std::byte{0});
  std::fill(pos_buf.begin(), pos_buf.end(), std::byte{0});

  for (int k = 0; k < valid_count; ++k) {
    for (int i = 0; i < H; ++i) {
      write_float_value(
          prev_buf.data(),
          pf_in_meta.scalar_type(),
          static_cast<size_t>(k) * H + i,
          prev_a_batch[static_cast<size_t>(k) * H + i]);
    }
    std::vector<uint16_t> emb_fp16(static_cast<size_t>(H));
    lookup_embedding(prev_tokens[static_cast<size_t>(k)], emb_fp16.data());
    for (int i = 0; i < H; ++i) {
      write_fp16_value(
          tok_emb_buf.data(),
          e_meta.scalar_type(),
          static_cast<size_t>(k) * H + i,
          emb_fp16[static_cast<size_t>(i)]);
    }
    reinterpret_cast<int32_t*>(pos_buf.data())[k] =
        static_cast<int32_t>(rope_pos);
  }
  for (int k = valid_count; k < ar; ++k) {
    reinterpret_cast<int32_t*>(pos_buf.data())[k] =
        static_cast<int32_t>(rope_pos);
  }

  const size_t ctx = static_cast<size_t>(eagle_meta_.target_context_len);
  for (int row = 0; row < ar; ++row) {
    for (size_t col = 0; col < ctx; ++col) {
      write_mask_value(
          attn_mask_buf.data(),
          m_meta.scalar_type(),
          static_cast<size_t>(row) * ctx + col,
          false);
    }
  }
  for (int row = 0; row < valid_count; ++row) {
    for (int64_t col = 0; col < stable_cache_len; ++col) {
      if (col >= 0 && col < static_cast<int64_t>(ctx)) {
        write_mask_value(
            attn_mask_buf.data(),
            m_meta.scalar_type(),
            static_cast<size_t>(row) * ctx + static_cast<size_t>(col),
            true);
      }
    }
    if (row < static_cast<int>(visible_past_slots.size())) {
      for (int32_t col : visible_past_slots[static_cast<size_t>(row)]) {
        if (col >= 0 && col < static_cast<int32_t>(ctx)) {
          write_mask_value(
              attn_mask_buf.data(),
              m_meta.scalar_type(),
              static_cast<size_t>(row) * ctx + static_cast<size_t>(col),
              true);
        }
      }
    }
    const int32_t self_col = static_cast<int32_t>(ctx) - ar + row;
    if (self_col >= 0 && self_col < static_cast<int32_t>(ctx)) {
      write_mask_value(
          attn_mask_buf.data(),
          m_meta.scalar_type(),
          static_cast<size_t>(row) * ctx + static_cast<size_t>(self_col),
          true);
    }
  }

  head_kv_manager_->rearrange_cache(ar);
  auto k_caches = head_kv_manager_->get_k_cache_();
  auto v_caches = head_kv_manager_->get_v_cache_();

  TensorImpl pf_in(
      pf_in_meta.scalar_type(),
      pf_in_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(pf_in_meta.sizes().data()),
      prev_buf.data(),
      const_cast<TensorImpl::DimOrderType*>(pf_in_meta.dim_order().data()));
  TensorImpl e_impl(
      e_meta.scalar_type(),
      e_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(e_meta.sizes().data()),
      tok_emb_buf.data(),
      const_cast<TensorImpl::DimOrderType*>(e_meta.dim_order().data()));
  TensorImpl m_impl(
      m_meta.scalar_type(),
      m_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(m_meta.sizes().data()),
      attn_mask_buf.data(),
      const_cast<TensorImpl::DimOrderType*>(m_meta.dim_order().data()));
  TensorImpl p_impl(
      p_meta.scalar_type(),
      p_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(p_meta.sizes().data()),
      pos_buf.data(),
      const_cast<TensorImpl::DimOrderType*>(p_meta.dim_order().data()));
  TensorImpl k_impl(
      k_meta.scalar_type(),
      k_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(k_meta.sizes().data()),
      k_caches[0].buffer,
      const_cast<TensorImpl::DimOrderType*>(k_meta.dim_order().data()));
  TensorImpl v_impl(
      v_meta.scalar_type(),
      v_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(v_meta.sizes().data()),
      v_caches[0].buffer,
      const_cast<TensorImpl::DimOrderType*>(v_meta.dim_order().data()));

  std::vector<EValue> head_inputs{
      EValue(executorch::aten::Tensor(&pf_in)),
      EValue(executorch::aten::Tensor(&e_impl)),
      EValue(executorch::aten::Tensor(&m_impl)),
      EValue(executorch::aten::Tensor(&p_impl)),
      EValue(executorch::aten::Tensor(&k_impl)),
      EValue(executorch::aten::Tensor(&v_impl)),
  };

  auto logits_meta = head_kv_method_meta_->output_tensor_meta(0).get();
  auto pf_meta = head_kv_method_meta_->output_tensor_meta(1).get();
  auto ko_meta = head_kv_method_meta_->output_tensor_meta(2).get();
  auto vo_meta = head_kv_method_meta_->output_tensor_meta(3).get();

  std::vector<std::byte> logits_buf(tensor_nbytes(logits_meta));
  std::vector<std::byte> pf_out_buf(tensor_nbytes(pf_meta));
  TensorImpl logits_out(
      logits_meta.scalar_type(),
      logits_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(logits_meta.sizes().data()),
      logits_buf.data(),
      const_cast<TensorImpl::DimOrderType*>(logits_meta.dim_order().data()));
  TensorImpl pf_out(
      pf_meta.scalar_type(),
      pf_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(pf_meta.sizes().data()),
      pf_out_buf.data(),
      const_cast<TensorImpl::DimOrderType*>(pf_meta.dim_order().data()));
  TensorImpl k_out(
      ko_meta.scalar_type(),
      ko_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(ko_meta.sizes().data()),
      k_caches[0].output_buffer,
      const_cast<TensorImpl::DimOrderType*>(ko_meta.dim_order().data()));
  TensorImpl v_out(
      vo_meta.scalar_type(),
      vo_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(vo_meta.sizes().data()),
      v_caches[0].output_buffer,
      const_cast<TensorImpl::DimOrderType*>(vo_meta.dim_order().data()));

  // Output order: (logits, prev_feature, k_new, v_new)
  std::vector<EValue> head_outputs_ev2{
      EValue(executorch::aten::Tensor(&logits_out)),
      EValue(executorch::aten::Tensor(&pf_out)),
      EValue(executorch::aten::Tensor(&k_out)),
      EValue(executorch::aten::Tensor(&v_out)),
  };

  auto set_err = head_module_->set_outputs("kv_forward", head_outputs_ev2);
  ET_CHECK_MSG(set_err == Error::Ok, "[Eagle] head kv_forward set_outputs failed");

  ET_LOG(
      Info,
      "[Eagle] exec head kv_forward rope=%lld cache=%lld ar=%d n=%d",
      static_cast<long long>(rope_pos),
      static_cast<long long>(cache_pos),
      ar,
      valid_count);
  long start_ms = time_in_ms();
  auto res = head_module_->execute("kv_forward", head_inputs);
  head_decode_time_ms_ += static_cast<double>(time_in_ms() - start_ms);
  ++head_decode_calls_;
  ET_CHECK_MSG(res.ok(), "[Eagle] head kv_forward execute failed");

  std::vector<bool> selected(static_cast<size_t>(ar), false);
  for (int k = 0; k < valid_count; ++k) {
    selected[static_cast<size_t>(k)] = true;
  }
  head_kv_manager_->update_cache(
      ar, static_cast<int32_t>(cache_pos), valid_count, selected);

  next_prev_batch->assign(static_cast<size_t>(valid_count) * H, 0.0f);
  for (int k = 0; k < valid_count; ++k) {
    for (int i = 0; i < H; ++i) {
      (*next_prev_batch)[static_cast<size_t>(k) * H + i] =
          read_hidden_scalar(
              pf_out_buf,
              pf_meta.scalar_type(),
              static_cast<size_t>(k) * H + i);
    }
  }

  const size_t logits_stride =
      static_cast<size_t>(eagle_meta_.draft_vocab_size) * dtype_size(logits_meta.scalar_type());
  if (valid_count > 0) {
    std::memcpy(head_logits_buf_.data(), logits_buf.data(), logits_stride);
  }
  choices_by_parent->clear();
  choices_by_parent->reserve(static_cast<size_t>(valid_count));
  for (int k = 0; k < valid_count; ++k) {
    choices_by_parent->push_back(
        topk_from_logits_slot(logits_buf.data(), logits_meta.scalar_type(), k, topk));
  }
}

// ---------------------------------------------------------------------------
// target_verify (chain mode)
// ---------------------------------------------------------------------------
void EagleTokenGenerator::target_verify(
    const std::vector<uint64_t>& packed_tokens,
    int64_t cur_pos,
    std::vector<uint64_t>* target_sampled_tokens,
    std::vector<float>* out_hidden_LMH_per_slot) {
  int ar = eagle_meta_.target_ar_len;
  int n  = static_cast<int>(packed_tokens.size());  // typically draft_len+1

  // 1. Fill input tokens (ar slots, pad with 0).
  for (int k = 0; k < ar; ++k) {
    int64_t tok = (k < n) ? static_cast<int64_t>(packed_tokens[k]) : 0;
    if (eagle_meta_.use_int64_token) {
      input_toks_.data[k] = tok;
    } else {
      reinterpret_cast<int32_t*>(input_toks_.data)[k] =
          static_cast<int32_t>(tok);
    }
  }

  // 2. Positions: [cur_pos, cur_pos+1, ..., cur_pos+n-1, cur_pos+n-1, ...]
  for (int k = 0; k < ar; ++k) {
    int pos_k = static_cast<int>(cur_pos) + std::min(k, n - 1);
    input_pos_.data[k] = static_cast<int32_t>(pos_k);
  }

  // 3. Chain causal mask: slot k attends to [0..cur_pos+k].
  //    Mask layout [ar, ctx]: row k is the attention mask for slot k.
  //    KVManager fills this via init_attention_mask with attention_map.
  std::vector<int32_t> attention_map(ar);
  for (int k = 0; k < ar; ++k) {
    attention_map[k] = (k == 0) ? -1 : std::min(k - 1, n - 1);
  }
  kv_manager_->init_attention_mask(
      attention_mask_.data, attention_map, ar, static_cast<int32_t>(cur_pos));

  // 4. Run target.
  ET_LOG(
      Info,
      "[Eagle] exec target %s cur_pos=%lld ar=%d n=%d first=%llu",
      method_name_.c_str(),
      static_cast<long long>(cur_pos),
      ar,
      n,
      packed_tokens.empty()
          ? 0ULL
          : static_cast<unsigned long long>(packed_tokens[0]));
  long start_ms = time_in_ms();
  auto logits_res = decoder_runner_->step(method_name_, inputs_);
  target_verify_time_ms_ += static_cast<double>(time_in_ms() - start_ms);
  ++target_verify_calls_;
  ET_CHECK_MSG(logits_res.ok(), "[Eagle] target verify step failed");
  executorch::aten::Tensor& logits_tensor = logits_res.get();

  // 5. argmax per slot → target_sampled_tokens.
  target_sampled_tokens->resize(static_cast<size_t>(ar));
  for (int k = 0; k < ar; ++k) {
    int32_t tok = decoder_runner_->logits_to_token(logits_tensor, k);
    (*target_sampled_tokens)[k] = static_cast<uint64_t>(tok);
  }

  // 6. Read hidden_LMH per slot (fp16 → fp32, cat low/mid/high per slot).
  //    out_hidden_LMH_per_slot[k] = float[3*H] for accepted slot k.
  int H  = eagle_meta_.hidden_dim;
  int H3 = 3 * H;
  out_hidden_LMH_per_slot->assign(
      static_cast<size_t>(ar) * static_cast<size_t>(H3), 0.0f);

  auto copy_hidden = [&](
                         const std::vector<std::byte>& buf,
                         ScalarType dtype,
                         int offset_in_H3) {
    // buf is [1, ar, H] flat.
    for (int k = 0; k < ar; ++k) {
      for (int h = 0; h < H; ++h) {
        (*out_hidden_LMH_per_slot)[k * H3 + offset_in_H3 + h] =
            read_hidden_scalar(buf, dtype, static_cast<size_t>(k * H + h));
      }
    }
  };
  if (!target_hidden_low_buf_.empty()) {
    copy_hidden(target_hidden_low_buf_, target_hidden_dtypes_[0], 0);
  }
  if (!target_hidden_mid_buf_.empty()) {
    copy_hidden(target_hidden_mid_buf_, target_hidden_dtypes_[1], H);
  }
  if (!target_hidden_high_buf_.empty()) {
    copy_hidden(target_hidden_high_buf_, target_hidden_dtypes_[2], 2 * H);
  }

  // KV is committed by generate() after it knows how many slots were accepted.
}

// ---------------------------------------------------------------------------
// sample_draft
// ---------------------------------------------------------------------------
uint64_t EagleTokenGenerator::sample_draft(const std::byte* draft_logits_buf) {
  if (!sampler_) {
    ET_LOG(Error, "[Eagle] sample_draft called before init_io");
    return 0;
  }
  return sampler_->argmax_draft(draft_logits_buf);
}

std::vector<EagleTokenGenerator::DraftChoice>
EagleTokenGenerator::topk_from_head_logits(int32_t topk) const {
  return topk_from_logits_slot(
      head_logits_buf_.data(), head_logits_dtype_, /*slot=*/0, topk);
}

std::vector<EagleTokenGenerator::DraftChoice>
EagleTokenGenerator::topk_from_logits_slot(
    const std::byte* logits_buf,
    ScalarType dtype,
    int32_t slot,
    int32_t topk) const {
  const int32_t vocab = eagle_meta_.draft_vocab_size;
  topk = std::max<int32_t>(1, std::min<int32_t>(topk, vocab));
  std::vector<double> logits(static_cast<size_t>(vocab));
  double max_logit = -std::numeric_limits<double>::infinity();
  const size_t base = static_cast<size_t>(slot) * static_cast<size_t>(vocab);
  for (int32_t i = 0; i < vocab; ++i) {
    double value = 0.0;
    if (dtype == ScalarType::Float) {
      value = reinterpret_cast<const float*>(logits_buf)[base + i];
    } else {
      value = fp16_to_fp32(
          reinterpret_cast<const uint16_t*>(logits_buf)[base + i]);
    }
    logits[static_cast<size_t>(i)] = value;
    max_logit = std::max(max_logit, value);
  }

  double sum_exp = 0.0;
  for (double value : logits) {
    sum_exp += std::exp(value - max_logit);
  }
  const double logsum = max_logit + std::log(sum_exp);

  std::vector<int32_t> indices(static_cast<size_t>(vocab));
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(
      indices.begin(),
      indices.begin() + topk,
      indices.end(),
      [&](int32_t a, int32_t b) {
        const double la = logits[static_cast<size_t>(a)];
        const double lb = logits[static_cast<size_t>(b)];
        if (la == lb) {
          return a < b;
        }
        return la > lb;
      });

  std::vector<DraftChoice> out;
  out.reserve(static_cast<size_t>(topk));
  for (int32_t i = 0; i < topk; ++i) {
    const uint64_t draft_id = static_cast<uint64_t>(indices[static_cast<size_t>(i)]);
    out.push_back(DraftChoice{
        draft_id,
        draft_to_target(draft_id),
        logits[static_cast<size_t>(draft_id)] - logsum});
  }
  return out;
}

void EagleTokenGenerator::replay_head_path(
    const std::vector<float>& stable_prev_a,
    const std::vector<uint64_t>& path,
    int64_t stable_head_cache_pos,
    std::vector<float>* out_prev_a) {
  const int H = eagle_meta_.hidden_dim;
  std::vector<float> cur_prev = stable_prev_a;
  std::vector<float> next_prev(static_cast<size_t>(H), 0.0f);
  for (size_t i = 0; i < path.size(); ++i) {
    (void)head_decode_step(
        cur_prev.data(),
        path[i],
        stable_head_cache_pos + static_cast<int64_t>(i),
        stable_head_cache_pos + static_cast<int64_t>(i),
        next_prev.data());
    cur_prev.swap(next_prev);
  }
  if (out_prev_a != nullptr) {
    *out_prev_a = std::move(cur_prev);
  }
}

EagleTokenGenerator::TreeProposal EagleTokenGenerator::build_tree_proposal(
    uint64_t root_token,
    const std::vector<float>& stable_prev_a,
    int64_t stable_head_cache_pos) {
  const int32_t depth = std::max<int32_t>(0, eagle_meta_.tree_depth);
  const int32_t topk = std::max<int32_t>(1, eagle_meta_.tree_topk);
  int32_t max_tree_nodes = eagle_meta_.max_tree_size > 0
      ? eagle_meta_.max_tree_size
      : 1 + topk * (depth + 1);
  max_tree_nodes = std::max<int32_t>(2, max_tree_nodes);
  if (max_tree_nodes > eagle_meta_.target_ar_len) {
    ET_LOG(
        Info,
        "[Eagle] tree nodes %d exceed target_ar_len %d; truncating tree",
        max_tree_nodes,
        eagle_meta_.target_ar_len);
    max_tree_nodes = eagle_meta_.target_ar_len;
  }
  const int32_t max_draft_nodes = max_tree_nodes - 1;

  std::vector<TreeNode> all_nodes;
  std::vector<int32_t> active;
  auto initial = topk_from_head_logits(topk);
  all_nodes.reserve(
      static_cast<size_t>(topk + depth * topk * topk));
  active.reserve(static_cast<size_t>(topk));
  for (const DraftChoice& choice : initial) {
    all_nodes.push_back(TreeNode{
        /*parent=*/-1,
        /*depth=*/1,
        /*cache_slot=*/-1,
        choice.target_id,
        choice.logp,
        std::vector<uint64_t>{choice.target_id},
        std::vector<int32_t>{}});
    active.push_back(static_cast<int32_t>(all_nodes.size() - 1));
  }

  std::vector<std::vector<float>> node_prev_features(all_nodes.size());
  for (auto& prev : node_prev_features) {
    prev = stable_prev_a;
  }
  int64_t generated_cache_slots = 0;
  for (int32_t level = 0; level < depth && !active.empty(); ++level) {
    const int32_t batch = static_cast<int32_t>(active.size());
    std::vector<float> prev_batch(static_cast<size_t>(batch) * eagle_meta_.hidden_dim);
    std::vector<uint64_t> token_batch(static_cast<size_t>(batch));
    std::vector<std::vector<int32_t>> visible_slots(static_cast<size_t>(batch));
    for (int32_t row = 0; row < batch; ++row) {
      const int32_t parent_index = active[static_cast<size_t>(row)];
      const TreeNode& parent = all_nodes[static_cast<size_t>(parent_index)];
      const std::vector<float>& prev = node_prev_features[static_cast<size_t>(parent_index)];
      std::memcpy(
          prev_batch.data() + static_cast<size_t>(row) * eagle_meta_.hidden_dim,
          prev.data(),
          static_cast<size_t>(eagle_meta_.hidden_dim) * sizeof(float));
      token_batch[static_cast<size_t>(row)] = parent.target_id;
      visible_slots[static_cast<size_t>(row)] = parent.visible_past_slots;
    }

    std::vector<float> next_prev_batch;
    std::vector<std::vector<DraftChoice>> choices_by_parent;
    const int64_t batch_cache_pos = stable_head_cache_pos + generated_cache_slots;
    head_decode_batch(
        prev_batch,
        token_batch,
        visible_slots,
        stable_head_cache_pos,
        stable_head_cache_pos + level,
        batch_cache_pos,
        topk,
        &next_prev_batch,
        &choices_by_parent);
    for (int32_t row = 0; row < batch; ++row) {
      all_nodes[static_cast<size_t>(active[static_cast<size_t>(row)])].cache_slot =
          static_cast<int32_t>(batch_cache_pos + row);
    }
    generated_cache_slots += batch;

    std::vector<int32_t> children;
    children.reserve(static_cast<size_t>(topk * topk));
    for (int32_t row = 0; row < batch; ++row) {
      const int32_t parent_index = active[static_cast<size_t>(row)];
      const TreeNode& parent = all_nodes[static_cast<size_t>(parent_index)];
      std::vector<int32_t> child_visible = parent.visible_past_slots;
      if (parent.cache_slot >= 0) {
        child_visible.push_back(parent.cache_slot);
      }
      const std::vector<DraftChoice>& child_topk =
          choices_by_parent[static_cast<size_t>(row)];
      for (const DraftChoice& choice : child_topk) {
        std::vector<uint64_t> child_path = parent.path;
        child_path.push_back(choice.target_id);
        all_nodes.push_back(TreeNode{
            parent_index,
            parent.depth + 1,
            /*cache_slot=*/-1,
            choice.target_id,
            parent.score + choice.logp,
            std::move(child_path),
            child_visible});
        node_prev_features.push_back(std::vector<float>(
            next_prev_batch.begin() +
                static_cast<ptrdiff_t>(row) * eagle_meta_.hidden_dim,
            next_prev_batch.begin() +
                static_cast<ptrdiff_t>(row + 1) * eagle_meta_.hidden_dim));
        children.push_back(static_cast<int32_t>(all_nodes.size() - 1));
      }
    }
    std::partial_sort(
        children.begin(),
        children.begin() +
            std::min<int32_t>(topk, static_cast<int32_t>(children.size())),
        children.end(),
        [&](int32_t a, int32_t b) {
          const double sa = all_nodes[static_cast<size_t>(a)].score;
          const double sb = all_nodes[static_cast<size_t>(b)].score;
          if (sa == sb) {
            return a < b;
          }
          return sa > sb;
        });
    if (static_cast<int32_t>(children.size()) > topk) {
      children.resize(static_cast<size_t>(topk));
    }
    active = std::move(children);
  }

  std::vector<int32_t> ranked(static_cast<size_t>(all_nodes.size()));
  std::iota(ranked.begin(), ranked.end(), 0);
  std::partial_sort(
      ranked.begin(),
      ranked.begin() +
          std::min<int32_t>(max_draft_nodes, static_cast<int32_t>(ranked.size())),
      ranked.end(),
      [&](int32_t a, int32_t b) {
        const double sa = all_nodes[static_cast<size_t>(a)].score;
        const double sb = all_nodes[static_cast<size_t>(b)].score;
        if (sa == sb) {
          return a < b;
        }
        return sa > sb;
      });
  std::vector<int32_t> selected;
  std::vector<bool> included(all_nodes.size(), false);
  for (int32_t node_index : ranked) {
    if (static_cast<int32_t>(selected.size()) >= max_draft_nodes) {
      break;
    }
    std::vector<int32_t> missing;
    for (int32_t cur = node_index; cur >= 0;
         cur = all_nodes[static_cast<size_t>(cur)].parent) {
      if (included[static_cast<size_t>(cur)]) {
        break;
      }
      missing.push_back(cur);
    }
    if (selected.size() + missing.size() >
        static_cast<size_t>(max_draft_nodes)) {
      continue;
    }
    for (auto it = missing.rbegin(); it != missing.rend(); ++it) {
      included[static_cast<size_t>(*it)] = true;
      selected.push_back(*it);
    }
  }
  std::sort(selected.begin(), selected.end());

  std::vector<int32_t> slot_for_node(all_nodes.size(), -1);
  TreeProposal proposal;
  proposal.nodes.reserve(selected.size());
  proposal.packed_tokens.reserve(selected.size() + 1);
  proposal.parent_slots.reserve(selected.size() + 1);
  proposal.position_offsets.reserve(selected.size() + 1);
  proposal.packed_tokens.push_back(root_token);
  proposal.parent_slots.push_back(-1);
  proposal.position_offsets.push_back(0);

  for (int32_t node_index : selected) {
    slot_for_node[static_cast<size_t>(node_index)] =
        static_cast<int32_t>(proposal.packed_tokens.size());
    const TreeNode& node = all_nodes[static_cast<size_t>(node_index)];
    proposal.nodes.push_back(node);
    proposal.packed_tokens.push_back(node.target_id);
    proposal.position_offsets.push_back(node.depth);
  }
  for (int32_t node_index : selected) {
    const TreeNode& node = all_nodes[static_cast<size_t>(node_index)];
    int32_t parent_slot = 0;
    if (node.parent >= 0) {
      parent_slot = slot_for_node[static_cast<size_t>(node.parent)];
      ET_CHECK_MSG(
          parent_slot > 0,
          "[Eagle] selected tree node is missing its parent");
    }
    proposal.parent_slots.push_back(parent_slot);
  }

  std::unordered_set<int32_t> parent_slots;
  for (int32_t parent_slot : proposal.parent_slots) {
    if (parent_slot >= 0) {
      parent_slots.insert(parent_slot);
    }
  }
  for (int32_t slot = 1; slot < static_cast<int32_t>(proposal.packed_tokens.size());
       ++slot) {
    if (parent_slots.count(slot) != 0) {
      continue;
    }
    std::vector<int32_t> path;
    int32_t cur = slot;
    while (cur >= 0) {
      path.push_back(cur);
      cur = proposal.parent_slots[static_cast<size_t>(cur)];
    }
    std::reverse(path.begin(), path.end());
    proposal.retrieve_indices.push_back(std::move(path));
  }
  if (proposal.retrieve_indices.empty()) {
    proposal.retrieve_indices.push_back(std::vector<int32_t>{0});
  }

  ET_LOG(
      Debug,
      "[Eagle] tree proposal nodes=%zu leaves=%zu depth=%d topk=%d",
      proposal.packed_tokens.size(),
      proposal.retrieve_indices.size(),
      depth,
      topk);
  return proposal;
}

void EagleTokenGenerator::target_verify_tree(
    const TreeProposal& proposal,
    int64_t cur_pos,
    std::vector<uint64_t>* target_sampled_tokens,
    std::vector<float>* out_hidden_LMH_per_slot) {
  const int ar = eagle_meta_.target_ar_len;
  const int n = static_cast<int>(proposal.packed_tokens.size());
  ET_CHECK_MSG(n <= ar, "[Eagle] tree proposal is larger than target ar_len");

  for (int k = 0; k < ar; ++k) {
    int64_t tok = (k < n) ? static_cast<int64_t>(proposal.packed_tokens[k]) : 0;
    if (eagle_meta_.use_int64_token) {
      input_toks_.data[k] = tok;
    } else {
      reinterpret_cast<int32_t*>(input_toks_.data)[k] =
          static_cast<int32_t>(tok);
    }
  }

  const int32_t last_offset =
      proposal.position_offsets.empty() ? 0 : proposal.position_offsets.back();
  for (int k = 0; k < ar; ++k) {
    const int32_t offset = k < n ? proposal.position_offsets[k] : last_offset;
    input_pos_.data[k] = static_cast<int32_t>(cur_pos + offset);
  }

  std::vector<int32_t> attention_map(static_cast<size_t>(ar), 0);
  for (int k = 0; k < n; ++k) {
    attention_map[static_cast<size_t>(k)] = proposal.parent_slots[k];
  }
  kv_manager_->init_attention_mask(
      attention_mask_.data, attention_map, ar, static_cast<int32_t>(cur_pos));

  ET_LOG(
      Info,
      "[Eagle] exec target tree %s cur_pos=%lld ar=%d n=%d candidate_paths=%zu",
      method_name_.c_str(),
      static_cast<long long>(cur_pos),
      ar,
      n,
      proposal.retrieve_indices.size());
  long start_ms = time_in_ms();
  auto logits_res = decoder_runner_->step(method_name_, inputs_);
  target_verify_time_ms_ += static_cast<double>(time_in_ms() - start_ms);
  ++target_verify_calls_;
  ET_CHECK_MSG(logits_res.ok(), "[Eagle] target tree verify step failed");
  executorch::aten::Tensor& logits_tensor = logits_res.get();

  target_sampled_tokens->resize(static_cast<size_t>(ar));
  for (int k = 0; k < ar; ++k) {
    int32_t tok = decoder_runner_->logits_to_token(logits_tensor, k);
    (*target_sampled_tokens)[k] = static_cast<uint64_t>(tok);
  }

  const int H = eagle_meta_.hidden_dim;
  const int H3 = 3 * H;
  out_hidden_LMH_per_slot->assign(
      static_cast<size_t>(ar) * static_cast<size_t>(H3), 0.0f);

  auto copy_hidden = [&](
                         const std::vector<std::byte>& buf,
                         ScalarType dtype,
                         int offset_in_H3) {
    for (int k = 0; k < ar; ++k) {
      for (int h = 0; h < H; ++h) {
        (*out_hidden_LMH_per_slot)[k * H3 + offset_in_H3 + h] =
            read_hidden_scalar(buf, dtype, static_cast<size_t>(k * H + h));
      }
    }
  };
  if (!target_hidden_low_buf_.empty()) {
    copy_hidden(target_hidden_low_buf_, target_hidden_dtypes_[0], 0);
  }
  if (!target_hidden_mid_buf_.empty()) {
    copy_hidden(target_hidden_mid_buf_, target_hidden_dtypes_[1], H);
  }
  if (!target_hidden_high_buf_.empty()) {
    copy_hidden(target_hidden_high_buf_, target_hidden_dtypes_[2], 2 * H);
  }
}

// ---------------------------------------------------------------------------
// generate — chain mode speculative decoding loop
// ---------------------------------------------------------------------------
Result<int64_t> EagleTokenGenerator::generate(
    std::vector<uint64_t> tokens,
    int64_t start_pos,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    bool dump_logits,
    AttentionSinkRopeRunner* attention_sink_rope_runner) {
  ET_CHECK_MSG(!tokens.empty(), "tokens must not be empty");
  total_drafted_ = 0;
  total_accepted_ = 0;
  target_verify_calls_ = 0;
  head_decode_calls_ = 0;
  head_prefill_calls_ = 0;
  target_verify_time_ms_ = 0.0;
  head_decode_time_ms_ = 0.0;
  head_prefill_time_ms_ = 0.0;

  const bool tree_mode = eagle_meta_.tree_depth > 0 && eagle_meta_.tree_topk > 0;

  if (target_hidden_low_buf_.empty()) {
    ET_LOG(
        Error,
        "[Eagle] Target hidden states not available (pte built without "
        "output_hidden_layers). Falling back to plain decode.");
    return TokenGenerator::generate(
        tokens, start_pos, seq_len, token_callback, dump_logits,
        attention_sink_rope_runner);
  }

  // ---- Setup ----
  int draft_len = eagle_meta_.draft_len;
  int H3        = 3 * eagle_meta_.hidden_dim;
  int H         = eagle_meta_.hidden_dim;
  auto head_prefill_h_meta = head_prefill_method_meta_->input_tensor_meta(0).get();
  const int32_t head_prefill_ar =
      static_cast<int32_t>(head_prefill_h_meta.sizes()[1]);

  // prev_a: fp32[H] — prev_feature fed back into head each step.
  std::vector<float> prev_a(static_cast<size_t>(H), 0.0f);
  std::vector<float> prev_a_next(static_cast<size_t>(H), 0.0f);

  // hidden_LMH_per_slot: fp32[ar * 3H] from target_verify.
  std::vector<float> hidden_LMH_per_slot;

  int64_t cur_pos         = start_pos;
  uint64_t last_committed = tokens.back();
  int64_t head_cache_pos  = 0;

  kv_manager_->rearrange_cache(eagle_meta_.target_ar_len);

  // Set up target output tensors (including hiddens).
  ET_CHECK_MSG(
      decoder_runner_->set_outputs(method_name_, output_tensors_) ==
          Error::Ok,
      "[Eagle] set_outputs failed");

  // Initialize the head like SafeAILab topK_genrate():
  //   hidden = target(prompt), input_ids = prompt[1:] + first target token.
  // The first generated token has already been sampled by prompt prefill and is
  // passed as tokens.back(); it is not in the target KV cache yet.
  if (!prompt_hidden_LMH_.empty() &&
      tokens.size() >=
          static_cast<size_t>(prompt_hidden_token_start_ + prompt_hidden_tokens_ + 1)) {
    for (int base = 0; base < prompt_hidden_tokens_; base += head_prefill_ar) {
      const int count = std::min(head_prefill_ar, prompt_hidden_tokens_ - base);
      std::vector<uint64_t> shifted_tokens(static_cast<size_t>(count));
      for (int k = 0; k < count; ++k) {
        shifted_tokens[static_cast<size_t>(k)] =
            tokens[static_cast<size_t>(
                prompt_hidden_token_start_ + base + k + 1)];
      }
      const float* hidden =
          prompt_hidden_LMH_.data() + static_cast<size_t>(base) * H3;
      head_prefill_batch(
          hidden,
          shifted_tokens.data(),
          count,
          head_cache_pos,
          head_cache_pos);
      head_cache_pos += count;
    }
    for (int i = 0; i < H; ++i) {
      prev_a[i] = read_hidden_scalar(
          head_prev_feature_buf_, head_prev_feature_dtype_, static_cast<size_t>(i));
    }
  } else {
    std::vector<uint64_t> bootstrap_sampled;
    target_verify(
        {last_committed}, cur_pos, &bootstrap_sampled, &hidden_LMH_per_slot);

    head_prefill_step(
        hidden_LMH_per_slot.data(), last_committed, head_cache_pos, head_cache_pos);
    for (int i = 0; i < H; ++i) {
      prev_a[i] = read_hidden_scalar(
          head_prev_feature_buf_, head_prev_feature_dtype_, static_cast<size_t>(i));
    }
    ++head_cache_pos;
  }

  // ---- Main loop ----
  while (cur_pos < static_cast<int64_t>(seq_len) - 1) {
    ET_LOG(
        Info,
        "[Eagle] loop cur_pos=%lld",
        static_cast<long long>(cur_pos));
    if (tree_mode) {
      const int32_t max_tree_offset = eagle_meta_.tree_depth + 1;
      if (cur_pos + max_tree_offset >= static_cast<int64_t>(seq_len)) {
        ET_LOG(
            Info,
            "[Eagle] stop before tree verify: cur_pos=%lld max_offset=%d seq_len=%d",
            static_cast<long long>(cur_pos),
            max_tree_offset,
            seq_len);
        break;
      }

      const int64_t stable_head_cache_pos = head_cache_pos;
      const std::vector<float> stable_prev_a = prev_a;
      TreeProposal proposal =
          build_tree_proposal(last_committed, stable_prev_a, stable_head_cache_pos);
      total_drafted_ += static_cast<uint64_t>(
          proposal.packed_tokens.size() > 0 ? proposal.packed_tokens.size() - 1 : 0);

      std::vector<uint64_t> target_sampled;
      target_verify_tree(
          proposal, cur_pos, &target_sampled, &hidden_LMH_per_slot);

      int best_leaf = 0;
      int best_accept = 0;
      for (int leaf = 0;
           leaf < static_cast<int>(proposal.retrieve_indices.size());
           ++leaf) {
        const std::vector<int32_t>& path =
            proposal.retrieve_indices[static_cast<size_t>(leaf)];
        int accept = 0;
        for (int j = 0; j + 1 < static_cast<int>(path.size()); ++j) {
          const int32_t cur_slot = path[static_cast<size_t>(j)];
          const int32_t next_slot = path[static_cast<size_t>(j + 1)];
          if (target_sampled[static_cast<size_t>(cur_slot)] !=
              proposal.packed_tokens[static_cast<size_t>(next_slot)]) {
            break;
          }
          ++accept;
        }
        if (accept > best_accept) {
          best_accept = accept;
          best_leaf = leaf;
        }
      }

      const std::vector<int32_t>& best_path =
          proposal.retrieve_indices[static_cast<size_t>(best_leaf)];
      const int32_t bonus_slot = best_path[static_cast<size_t>(best_accept)];
      const uint64_t committed_tok =
          target_sampled[static_cast<size_t>(bonus_slot)];
      total_accepted_ += static_cast<uint64_t>(best_accept);
      const int drafted_this_round =
          proposal.packed_tokens.size() > 0
          ? static_cast<int>(proposal.packed_tokens.size() - 1)
          : 0;
      const double accept_per_draft = drafted_this_round == 0
          ? 0.0
          : static_cast<double>(best_accept) /
              static_cast<double>(drafted_this_round);
      const double accept_per_depth = eagle_meta_.tree_depth <= 0
          ? 0.0
          : static_cast<double>(best_accept) /
              static_cast<double>(eagle_meta_.tree_depth);

      ET_LOG(
          Info,
          "[Eagle] tree accept cur_pos=%lld accepted_drafts=%d/%d "
          "accept_per_draft=%.3f accept_per_depth=%d/%d=%.3f "
          "chosen_path_len=%zu candidate_paths=%zu committed=%llu",
          static_cast<long long>(cur_pos),
          best_accept,
          drafted_this_round,
          accept_per_draft,
          best_accept,
          eagle_meta_.tree_depth,
          accept_per_depth,
          best_path.size(),
          proposal.retrieve_indices.size(),
          static_cast<unsigned long long>(committed_tok));

      uint64_t prev_out = last_committed;
      for (int k = 1; k <= best_accept; ++k) {
        const uint64_t tok =
            proposal.packed_tokens[static_cast<size_t>(best_path[k])];
        token_callback(ET_UNWRAP_TOKENIZER(tokenizer_->decode(prev_out, tok)));
        if (eos_ids_->count(tok) > 0) {
          goto done;
        }
        prev_out = tok;
      }
      token_callback(
          ET_UNWRAP_TOKENIZER(tokenizer_->decode(prev_out, committed_tok)));
      if (eos_ids_->count(committed_tok) > 0) {
        goto done;
      }

      std::vector<bool> target_selected(
          static_cast<size_t>(eagle_meta_.target_ar_len), false);
      for (int k = 0; k <= best_accept; ++k) {
        target_selected[static_cast<size_t>(best_path[k])] = true;
      }
      kv_manager_->update_cache(
          eagle_meta_.target_ar_len,
          static_cast<int32_t>(cur_pos),
          best_accept + 1,
          target_selected);
      kv_manager_->update_attention_mask(
          attention_mask_.data,
          eagle_meta_.target_ar_len,
          static_cast<int32_t>(cur_pos),
          best_accept + 1);

      cur_pos += best_accept + 1;
      last_committed = committed_tok;

      head_cache_pos = stable_head_cache_pos;
      const int refresh_count = best_accept + 1;
      std::vector<float> refresh_hidden(
          static_cast<size_t>(refresh_count) * static_cast<size_t>(H3));
      std::vector<uint64_t> refresh_tokens(static_cast<size_t>(refresh_count));
      for (int k = 0; k < refresh_count; ++k) {
        const int32_t slot = best_path[static_cast<size_t>(k)];
        const float* slot_base =
            hidden_LMH_per_slot.data() +
            static_cast<size_t>(slot) * static_cast<size_t>(H3);
        std::memcpy(
            refresh_hidden.data() + static_cast<size_t>(k) * H3,
            slot_base,
            static_cast<size_t>(H3) * sizeof(float));
        refresh_tokens[static_cast<size_t>(k)] = (k < best_accept)
            ? proposal.packed_tokens[static_cast<size_t>(best_path[k + 1])]
            : committed_tok;
      }
      for (int base = 0; base < refresh_count; base += head_prefill_ar) {
        const int count = std::min(head_prefill_ar, refresh_count - base);
        head_prefill_batch(
            refresh_hidden.data() + static_cast<size_t>(base) * H3,
            refresh_tokens.data() + base,
            count,
            head_cache_pos,
            head_cache_pos);
        head_cache_pos += count;
      }
      for (int i = 0; i < H; ++i) {
        prev_a[i] = read_hidden_scalar(
            head_prev_feature_buf_,
            head_prev_feature_dtype_,
            static_cast<size_t>(i));
      }
      continue;
    }

    if (cur_pos + draft_len >= static_cast<int64_t>(seq_len)) {
      ET_LOG(
          Info,
          "[Eagle] stop before verify: cur_pos=%lld draft_len=%d seq_len=%d",
          static_cast<long long>(cur_pos),
          draft_len,
          seq_len);
      break;
    }

    // --- DRAFT ---
    int64_t stable_head_cache_pos = head_cache_pos;
    std::vector<uint64_t> draft_target_ids;  // target-vocab ids proposed by head
    uint64_t draft_id = sample_draft(head_logits_buf_.data());
    uint64_t prev_token = draft_to_target(draft_id);
    draft_target_ids.push_back(prev_token);

    for (int k = 1; k < draft_len; ++k) {
      draft_id = head_decode_step(
          prev_a.data(),
          prev_token,
          head_cache_pos,
          head_cache_pos,
          prev_a_next.data());
      uint64_t target_id = draft_to_target(draft_id);
      draft_target_ids.push_back(target_id);
      prev_token = target_id;
      memcpy(prev_a.data(), prev_a_next.data(), static_cast<size_t>(H) * sizeof(float));
      ++head_cache_pos;
    }
    total_drafted_ += static_cast<uint64_t>(draft_len);

    // --- VERIFY ---
    // packed = [last_committed, draft[0], ..., draft[draft_len-1]]
    std::vector<uint64_t> packed_tokens;
    packed_tokens.push_back(last_committed);
    packed_tokens.insert(
        packed_tokens.end(), draft_target_ids.begin(), draft_target_ids.end());

    std::vector<uint64_t> target_sampled;
    target_verify(packed_tokens, cur_pos, &target_sampled, &hidden_LMH_per_slot);

    // --- ACCEPT ---
    // Greedy: target_sampled[k] is argmax of target logits at slot k.
    // draft_target_ids[k] was proposed for position cur_pos+k+1.
    // We accept draft[k] if target_sampled[k] == draft_target_ids[k].
    int accepted = 0;
    for (int k = 0; k < draft_len; ++k) {
      if (target_sampled[k] == draft_target_ids[k]) {
        ++accepted;
      } else {
        break;
      }
    }
    uint64_t committed_tok = target_sampled[accepted];  // bonus token
    total_accepted_ += static_cast<uint64_t>(accepted);

    ET_LOG(
        Debug,
        "[Eagle] pos=%lld accept=%d/%d committed=%llu",
        static_cast<long long>(cur_pos),
        accepted,
        draft_len,
        static_cast<unsigned long long>(committed_tok));

    // Emit accepted draft tokens + bonus.
    for (int k = 0; k < accepted; ++k) {
      token_callback(ET_UNWRAP_TOKENIZER(
          tokenizer_->decode(
              k == 0 ? last_committed : draft_target_ids[k - 1],
              draft_target_ids[k])));
      if (eos_ids_->count(draft_target_ids[k]) > 0) goto done;
    }
    token_callback(ET_UNWRAP_TOKENIZER(
        tokenizer_->decode(
            accepted > 0 ? draft_target_ids[accepted - 1] : last_committed,
            committed_tok)));
    if (eos_ids_->count(committed_tok) > 0) goto done;

    // --- KV ROLLBACK ---
    {
      // Target: commit slots [0..accepted] (accepted+1 tokens consumed).
      std::vector<bool> target_selected(
          static_cast<size_t>(eagle_meta_.target_ar_len), false);
      for (int k = 0; k <= accepted; ++k) {
        target_selected[static_cast<size_t>(k)] = true;
      }
      kv_manager_->update_cache(
          eagle_meta_.target_ar_len,
          static_cast<int32_t>(cur_pos),
          accepted + 1,
          target_selected);
      kv_manager_->update_attention_mask(
          attention_mask_.data,
          eagle_meta_.target_ar_len,
          static_cast<int32_t>(cur_pos),
          accepted + 1);
    }

    cur_pos  += accepted + 1;
    last_committed = committed_tok;

    // --- REFRESH HEAD ---
    // Append the accepted target-hidden path to the head's stable KV. The token
    // fed with hidden slot k is the next token on the committed path; the last
    // slot is paired with the newly sampled bonus token.
    {
      head_cache_pos = stable_head_cache_pos;
      const int refresh_count = accepted + 1;
      std::vector<uint64_t> refresh_tokens(static_cast<size_t>(refresh_count));
      for (int k = 0; k < refresh_count; ++k) {
        refresh_tokens[static_cast<size_t>(k)] =
            (k < accepted) ? draft_target_ids[static_cast<size_t>(k)]
                           : committed_tok;
      }
      for (int base = 0; base < refresh_count; base += head_prefill_ar) {
        const int count = std::min(head_prefill_ar, refresh_count - base);
        head_prefill_batch(
            hidden_LMH_per_slot.data() + static_cast<size_t>(base) * H3,
            refresh_tokens.data() + base,
            count,
            head_cache_pos,
            head_cache_pos);
        head_cache_pos += count;
      }
      for (int i = 0; i < H; ++i) {
        prev_a[i] = read_hidden_scalar(
            head_prev_feature_buf_,
            head_prev_feature_dtype_,
            static_cast<size_t>(i));
      }
    }
  }

done:
  if (total_drafted_ > 0) {
    ET_LOG(
        Info,
        "[Eagle] accept_rate=%.3f (%llu/%llu)",
        static_cast<double>(total_accepted_) /
            static_cast<double>(total_drafted_),
        static_cast<unsigned long long>(total_accepted_),
        static_cast<unsigned long long>(total_drafted_));
  }
  ET_LOG(
      Info,
      "[Eagle] profile target_verify: calls=%llu total=%.3fms avg=%.3fms",
      static_cast<unsigned long long>(target_verify_calls_),
      target_verify_time_ms_,
      target_verify_calls_ == 0
          ? 0.0
          : target_verify_time_ms_ /
              static_cast<double>(target_verify_calls_));
  ET_LOG(
      Info,
      "[Eagle] profile head_decode: calls=%llu total=%.3fms avg=%.3fms",
      static_cast<unsigned long long>(head_decode_calls_),
      head_decode_time_ms_,
      head_decode_calls_ == 0
          ? 0.0
          : head_decode_time_ms_ / static_cast<double>(head_decode_calls_));
  ET_LOG(
      Info,
      "[Eagle] profile head_prefill: calls=%llu total=%.3fms avg=%.3fms",
      static_cast<unsigned long long>(head_prefill_calls_),
      head_prefill_time_ms_,
      head_prefill_calls_ == 0
          ? 0.0
          : head_prefill_time_ms_ /
              static_cast<double>(head_prefill_calls_));
  return cur_pos - start_pos;
}

} // namespace example
