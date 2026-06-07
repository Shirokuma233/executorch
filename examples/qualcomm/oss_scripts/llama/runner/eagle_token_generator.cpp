/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// EagleTokenGenerator — Phase 3 chain-mode speculative decoding.

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/eagle_token_generator.h>

#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <cstring>

using executorch::aten::ScalarType;
using executorch::aten::TensorImpl;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

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
      "draft_vocab=%d hidden_dim=%d",
      eagle_meta_.target_ar_len,
      eagle_meta_.draft_len,
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
  // 1. Convert hidden_LMH (fp32, 3*H or already H after fc — here it IS 3H
  //    because we feed the concat directly; the head's prefill_forward does fc).
  //    Actually the Python wrapper prefill_forward takes [B,S,3H] as hidden_LMH
  //    and the EagleHead.fc maps it to H internally. So we pass [1,1,3H].
  //    But EagleTokenGenerator::head_prefill_step signature says hidden_LMH is
  //    [3*hidden_dim]. We fp32->fp16 cast it into the prefill_buf.

  int H3 = 3 * eagle_meta_.hidden_dim;
  auto h_meta = head_prefill_method_meta_->input_tensor_meta(0).get();
  std::vector<std::byte> hidden_buf(tensor_nbytes(h_meta));
  for (int i = 0; i < H3; ++i) {
    write_float_value(hidden_buf.data(), h_meta.scalar_type(), i, hidden_LMH[i]);
  }

  // 2. Token embedding lookup.
  std::vector<uint16_t> emb_fp16(static_cast<size_t>(eagle_meta_.hidden_dim));
  lookup_embedding(prev_token, emb_fp16.data());
  for (int i = 0; i < eagle_meta_.hidden_dim; ++i) {
    write_fp16_value(
        head_tok_emb_buf_.data(), head_tok_emb_dtype_, i, emb_fp16[i]);
  }

  std::vector<int32_t> attention_map{-1};
  head_kv_manager_->init_attention_mask(
      head_attn_mask_buf_.data(),
      attention_map,
      1,
      static_cast<int32_t>(cache_pos));

  // 4. Position.
  *reinterpret_cast<int32_t*>(head_pos_buf_.data()) =
      static_cast<int32_t>(rope_pos);

  // 5. KV cache init: head_kv_manager_ rearranges for ar=1.
  head_kv_manager_->rearrange_cache(1);

  auto k_caches = head_kv_manager_->get_k_cache_();
  auto v_caches = head_kv_manager_->get_v_cache_();
  auto e_meta = head_prefill_method_meta_->input_tensor_meta(1).get();
  auto m_meta = head_prefill_method_meta_->input_tensor_meta(2).get();
  auto p_meta = head_prefill_method_meta_->input_tensor_meta(3).get();
  auto k_meta = head_prefill_method_meta_->input_tensor_meta(4).get();
  auto v_meta = head_prefill_method_meta_->input_tensor_meta(5).get();

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
      head_tok_emb_buf_.data(),
      const_cast<TensorImpl::DimOrderType*>(e_meta.dim_order().data()));
  TensorImpl m_impl(
      m_meta.scalar_type(),
      m_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(m_meta.sizes().data()),
      head_attn_mask_buf_.data(),
      const_cast<TensorImpl::DimOrderType*>(m_meta.dim_order().data()));
  TensorImpl p_impl(
      p_meta.scalar_type(),
      p_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(p_meta.sizes().data()),
      head_pos_buf_.data(),
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

  TensorImpl logits_out(
      logits_meta.scalar_type(),
      logits_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(logits_meta.sizes().data()),
      head_logits_buf_.data(),
      const_cast<TensorImpl::DimOrderType*>(logits_meta.dim_order().data()));
  TensorImpl pf_out(
      pf_meta.scalar_type(),
      pf_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(pf_meta.sizes().data()),
      head_prev_feature_buf_.data(),
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
      "[Eagle] exec head prefill_forward rope=%lld cache=%lld",
      static_cast<long long>(rope_pos),
      static_cast<long long>(cache_pos));
  auto res = head_module_->execute("prefill_forward", head_inputs);
  ET_CHECK_MSG(res.ok(), "[Eagle] head prefill_forward execute failed");

  // 8. Advance head KV.
  head_kv_manager_->update_cache(1, static_cast<int32_t>(cache_pos), 1, {});
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
  // prev_a is [H] fp32 — convert to fp16 for head input.
  int H = eagle_meta_.hidden_dim;
  for (int i = 0; i < H; ++i) {
    write_float_value(
        head_prev_feature_buf_.data(), head_prev_feature_dtype_, i, prev_a[i]);
  }

  // Token embedding.
  std::vector<uint16_t> emb_fp16(static_cast<size_t>(H));
  lookup_embedding(prev_token, emb_fp16.data());
  for (int i = 0; i < H; ++i) {
    write_fp16_value(
        head_tok_emb_buf_.data(), head_tok_emb_dtype_, i, emb_fp16[i]);
  }

  std::vector<int32_t> attention_map{-1};
  head_kv_manager_->init_attention_mask(
      head_attn_mask_buf_.data(),
      attention_map,
      1,
      static_cast<int32_t>(cache_pos));

  *reinterpret_cast<int32_t*>(head_pos_buf_.data()) =
      static_cast<int32_t>(rope_pos);

  auto k_caches = head_kv_manager_->get_k_cache_();
  auto v_caches = head_kv_manager_->get_v_cache_();
  auto pf_in_meta = head_kv_method_meta_->input_tensor_meta(0).get();
  auto e_meta = head_kv_method_meta_->input_tensor_meta(1).get();
  auto m_meta = head_kv_method_meta_->input_tensor_meta(2).get();
  auto p_meta = head_kv_method_meta_->input_tensor_meta(3).get();
  auto k_meta = head_kv_method_meta_->input_tensor_meta(4).get();
  auto v_meta = head_kv_method_meta_->input_tensor_meta(5).get();

  TensorImpl pf_in(
      pf_in_meta.scalar_type(),
      pf_in_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(pf_in_meta.sizes().data()),
      head_prev_feature_buf_.data(),
      const_cast<TensorImpl::DimOrderType*>(pf_in_meta.dim_order().data()));
  TensorImpl e_impl(
      e_meta.scalar_type(),
      e_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(e_meta.sizes().data()),
      head_tok_emb_buf_.data(),
      const_cast<TensorImpl::DimOrderType*>(e_meta.dim_order().data()));
  TensorImpl m_impl(
      m_meta.scalar_type(),
      m_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(m_meta.sizes().data()),
      head_attn_mask_buf_.data(),
      const_cast<TensorImpl::DimOrderType*>(m_meta.dim_order().data()));
  TensorImpl p_impl(
      p_meta.scalar_type(),
      p_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(p_meta.sizes().data()),
      head_pos_buf_.data(),
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

  std::vector<std::byte> pf_out_buf(tensor_nbytes(pf_meta));
  TensorImpl logits_out(
      logits_meta.scalar_type(),
      logits_meta.sizes().size(),
      const_cast<TensorImpl::SizesType*>(logits_meta.sizes().data()),
      head_logits_buf_.data(),
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
      "[Eagle] exec head kv_forward rope=%lld cache=%lld token=%llu",
      static_cast<long long>(rope_pos),
      static_cast<long long>(cache_pos),
      static_cast<unsigned long long>(prev_token));
  auto res = head_module_->execute("kv_forward", head_inputs);
  ET_CHECK_MSG(res.ok(), "[Eagle] head kv_forward execute failed");

  head_kv_manager_->update_cache(1, static_cast<int32_t>(cache_pos), 1, {});

  memcpy(head_prev_feature_buf_.data(), pf_out_buf.data(), pf_out_buf.size());
  for (int i = 0; i < H; ++i) {
    prev_a_buffer[i] = read_hidden_scalar(
        pf_out_buf, pf_meta.scalar_type(), static_cast<size_t>(i));
  }

  // Sample: 32000-way argmax over draft logits.
  return sampler_->argmax_draft(head_logits_buf_.data());
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
  auto logits_res = decoder_runner_->step(method_name_, inputs_);
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

  const bool tree_mode = !tree_branching_per_depth_.empty();
  if (tree_mode) {
    ET_LOG(
        Error,
        "[Eagle] Tree mode not yet implemented; falling back to plain decode.");
    return TokenGenerator::generate(
        tokens, start_pos, seq_len, token_callback, dump_logits,
        attention_sink_rope_runner);
  }

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
    for (int k = 0; k < prompt_hidden_tokens_; ++k) {
      const float* hidden =
          prompt_hidden_LMH_.data() + static_cast<size_t>(k) * H3;
      uint64_t shifted_token =
          tokens[static_cast<size_t>(prompt_hidden_token_start_ + k + 1)];
      head_prefill_step(hidden, shifted_token, head_cache_pos, head_cache_pos);
      ++head_cache_pos;
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
      for (int k = 0; k <= accepted; ++k) {
        const float* slot_base =
            hidden_LMH_per_slot.data() +
            static_cast<size_t>(k) * static_cast<size_t>(H3);
        uint64_t shifted_token =
            (k < accepted) ? draft_target_ids[static_cast<size_t>(k)]
                           : committed_tok;
        head_prefill_step(slot_base, shifted_token, head_cache_pos, head_cache_pos);
        ++head_cache_pos;
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
  return cur_pos - start_pos;
}

} // namespace example
