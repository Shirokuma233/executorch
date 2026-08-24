/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// DFlashTokenGenerator — block-diffusion speculative decoding.

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/dflash_token_generator.h>

#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/log.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::llm::time_in_ms;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

namespace example {

const char* const DFlashTokenGenerator::kStageNames[
    DFlashTokenGenerator::kNumStages] = {
    "draft_prep",
    "draft_exec",
    "draft_post",
    "draft_head",
    "draft_pick",
    "tree_build",
    "verify_prep",
    "verify_exec",
    "verify_head",
    "verify_pick",
    "stage_copy",
    "commit",
    "emit",
};

namespace {

// time_in_ms() rounds to whole milliseconds, so a stage that costs 40 us reads as
// 0 or 1 and its total is quantization noise rather than a measurement.
inline int64_t time_in_us() {
  struct timespec t;
#if defined(__ANDROID_API__)
  clock_gettime(CLOCK_MONOTONIC, &t);
#else
  timespec_get(&t, TIME_UTC);
#endif
  return static_cast<int64_t>(t.tv_sec) * 1000000 + t.tv_nsec / 1000;
}

// Argmax per row over raw uint16 logit codes. The lm_head's encoding is a
// monotone affine map, so the winning index is the same as on dequantized
// values, and strict `>` breaks ties toward the lower id exactly as
// Sampler::sample_argmax does.
//
// This exists because DecoderRunner::logits_to_token cannot be used cheaply: for
// every row it expands all 151936 codes into a float vector (608 KB of stores)
// through a per-element dtype switch, then rescans that vector. Measured on
// device at 0.42 ms/row against 0.080 ms/row here.
// The scalar form below carries the index in a loop-carried dependency behind a
// data-dependent branch, which no compiler will vectorise. Splitting it into a
// max reduction and a first-match search makes both halves pure SIMD and keeps
// the result identical: strict `>` means the answer is the FIRST position
// holding the maximum, which is what the second pass looks for. Starting the
// reduction at 0 mirrors `best = 0`, so an all-zero row still answers 0.
inline uint64_t argmax_u16_row(const uint16_t* row, size_t vocab) {
#if defined(__ARM_NEON)
  uint16x8_t acc = vdupq_n_u16(0);
  size_t v = 0;
  for (; v + 8 <= vocab; v += 8) {
    acc = vmaxq_u16(acc, vld1q_u16(row + v));
  }
  uint16_t best = vmaxvq_u16(acc);
  for (size_t t = v; t < vocab; ++t) {
    best = row[t] > best ? row[t] : best;
  }
  const uint16x8_t want = vdupq_n_u16(best);
  for (v = 0; v + 8 <= vocab; v += 8) {
    if (vmaxvq_u16(vceqq_u16(vld1q_u16(row + v), want)) != 0) {
      for (size_t t = v; t < v + 8; ++t) {
        if (row[t] == best) {
          return static_cast<uint64_t>(t);
        }
      }
    }
  }
  for (; v < vocab; ++v) {
    if (row[v] == best) {
      return static_cast<uint64_t>(v);
    }
  }
  return 0;
#else
  uint16_t best = 0;
  uint64_t arg = 0;
  for (size_t v = 0; v < vocab; ++v) {
    if (row[v] > best) {
      best = row[v];
      arg = static_cast<uint64_t>(v);
    }
  }
  return arg;
#endif
}

// Top-k of one row, descending, into `codes`/`ids` (both at least k long).
// Seeded with the first k entries so the threshold is valid from the start, then
// one pass where the common case is a single comparison against it.
inline void topk_u16_row(
    const uint16_t* row,
    size_t vocab,
    int k,
    uint16_t* codes,
    uint64_t* ids) {
  for (int i = 0; i < k; ++i) {
    codes[i] = row[i];
    ids[i] = static_cast<uint64_t>(i);
  }
  for (int i = 1; i < k; ++i) {
    uint16_t c = codes[i];
    uint64_t d = ids[i];
    int j = i - 1;
    for (; j >= 0 && codes[j] < c; --j) {
      codes[j + 1] = codes[j];
      ids[j + 1] = ids[j];
    }
    codes[j + 1] = c;
    ids[j + 1] = d;
  }
  size_t v = static_cast<size_t>(k);
#if defined(__ARM_NEON)
  // Almost every element loses to the threshold, so the scan is one compare and
  // a branch per element -- exactly what a SIMD pre-filter is for. A block that
  // holds no candidate is skipped whole; a block that does falls through to the
  // scalar body below, element by element in increasing v, re-reading the
  // threshold each time. That keeps insertion order and tie-breaking identical
  // to the pure scalar loop, which matters because the threshold rises as
  // elements go in.
  uint16x8_t thr = vdupq_n_u16(codes[k - 1]);
  for (; v + 8 <= vocab; v += 8) {
    if (vmaxvq_u16(vcgtq_u16(vld1q_u16(row + v), thr)) == 0) {
      continue;
    }
    for (size_t u = v; u < v + 8; ++u) {
      const uint16_t c = row[u];
      if (c <= codes[k - 1]) {
        continue;
      }
      int j = k - 2;
      for (; j >= 0 && codes[j] < c; --j) {
        codes[j + 1] = codes[j];
        ids[j + 1] = ids[j];
      }
      codes[j + 1] = c;
      ids[j + 1] = static_cast<uint64_t>(u);
    }
    thr = vdupq_n_u16(codes[k - 1]);
  }
#endif
  for (; v < vocab; ++v) {
    const uint16_t c = row[v];
    if (c <= codes[k - 1]) {
      continue;
    }
    int j = k - 2;
    for (; j >= 0 && codes[j] < c; --j) {
      codes[j + 1] = codes[j];
      ids[j + 1] = ids[j];
    }
    codes[j + 1] = c;
    ids[j + 1] = static_cast<uint64_t>(v);
  }
}

// fp16_to_fp32 lives in runner/utils.h (shared with the EAGLE sampler).

inline uint16_t fp32_to_fp16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(bits));
  uint16_t sign = (bits >> 31) & 0x1u;
  int32_t exp = ((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = bits & 0x7fffffu;
  if (exp <= 0) {
    return sign << 15;
  } else if (exp >= 31) {
    return (sign << 15) | 0x7c00u;
  }
  return static_cast<uint16_t>(
      (sign << 15) | (static_cast<uint16_t>(exp) << 10) | (mant >> 13));
}

// In this codebase UInt16 means "quantized integer", not fp16 bits (see
// DecoderRunner::logits_to_token). A quantized tensor is never valid here: we
// have no scale/zero_point to undo it, and reading the bits as fp16 would
// silently feed the draft garbage.
inline float read_scalar(const std::byte* data, ScalarType dtype, size_t i) {
  switch (dtype) {
    case ScalarType::Float:
      return reinterpret_cast<const float*>(data)[i];
    case ScalarType::Half:
      return fp16_to_fp32(reinterpret_cast<const uint16_t*>(data)[i]);
    case ScalarType::UInt16:
      ET_CHECK_MSG(
          false,
          "[DFlash] target hidden output is quantized (uint16). A captured layer "
          "sits on a graph-sharding boundary; re-export with a --num_sharding "
          "that avoids it.");
      return 0.0f;
    default:
      ET_CHECK_MSG(false, "[DFlash] unsupported read dtype");
      return 0.0f;
  }
}

inline void write_float(std::byte* dst, ScalarType dtype, size_t i, float v) {
  switch (dtype) {
    case ScalarType::Float:
      reinterpret_cast<float*>(dst)[i] = v;
      break;
    case ScalarType::Half:
    case ScalarType::UInt16:
      reinterpret_cast<uint16_t*>(dst)[i] = fp32_to_fp16(v);
      break;
    default:
      ET_CHECK_MSG(false, "[DFlash] unsupported write dtype");
      break;
  }
}

// 0xFBFF is -65504, the most negative finite fp16. 0xFC00 would be -inf, which
// makes softmax produce NaN whenever a query row is fully masked.
// Bulk forms of the two accessors above. The scalar versions carry a runtime
// dtype switch, and the callers use them over runs of thousands of elements --
// one draft prep writes n_new * 12800 of them. Deciding the conversion once per
// run and then walking a typed pointer is the same arithmetic with the dispatch
// hoisted; the default arms fall back element by element so the ET_CHECK on an
// unsupported dtype still fires exactly where it used to.
inline void read_scalars(
    const std::byte* data,
    ScalarType dtype,
    size_t off,
    size_t n,
    float* dst) {
  switch (dtype) {
    case ScalarType::Float:
      std::memcpy(dst, reinterpret_cast<const float*>(data) + off, n * sizeof(float));
      break;
    case ScalarType::Half: {
      const uint16_t* src = reinterpret_cast<const uint16_t*>(data) + off;
      for (size_t i = 0; i < n; ++i) {
        dst[i] = fp16_to_fp32(src[i]);
      }
      break;
    }
    default:
      for (size_t i = 0; i < n; ++i) {
        dst[i] = read_scalar(data, dtype, off + i);
      }
      break;
  }
}

inline void write_floats(
    std::byte* dst,
    ScalarType dtype,
    size_t off,
    size_t n,
    const float* src) {
  switch (dtype) {
    case ScalarType::Float:
      std::memcpy(reinterpret_cast<float*>(dst) + off, src, n * sizeof(float));
      break;
    case ScalarType::Half:
    case ScalarType::UInt16: {
      uint16_t* d = reinterpret_cast<uint16_t*>(dst) + off;
      for (size_t i = 0; i < n; ++i) {
        d[i] = fp32_to_fp16(src[i]);
      }
      break;
    }
    default:
      for (size_t i = 0; i < n; ++i) {
        write_float(dst, dtype, off + i, src[i]);
      }
      break;
  }
}

inline void write_mask(std::byte* dst, ScalarType dtype, size_t i, bool vis) {
  switch (dtype) {
    case ScalarType::Float:
      reinterpret_cast<float*>(dst)[i] = vis ? 0.0f : -65504.0f;
      break;
    case ScalarType::Half:
    case ScalarType::UInt16:
      reinterpret_cast<uint16_t*>(dst)[i] = vis ? 0x0000 : 0xFBFF;
      break;
    default:
      ET_CHECK_MSG(false, "[DFlash] unsupported mask dtype");
      break;
  }
}

inline size_t dtype_size(ScalarType dtype) {
  switch (dtype) {
    case ScalarType::Float:
      return 4;
    case ScalarType::Half:
    case ScalarType::UInt16:
      return 2;
    case ScalarType::Int:
      return 4;
    case ScalarType::Long:
      return 8;
    case ScalarType::Byte:
      return 1;
    default:
      ET_CHECK_MSG(false, "[DFlash] unsupported dtype");
      return 0;
  }
}

inline size_t tensor_nbytes(const TensorInfo& info) {
  size_t n = dtype_size(info.scalar_type());
  for (auto d : info.sizes()) {
    n *= static_cast<size_t>(d);
  }
  return n;
}

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

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
DFlashTokenGenerator::DFlashTokenGenerator(
    tokenizers::Tokenizer* tokenizer,
    DecoderRunner* target_runner,
    KVManager* target_kv_manager,
    executorch::extension::Module* draft_module,
    KVManager* draft_kv_manager,
    const std::string& target_kv_method_name,
    std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids,
    Metadata metadata,
    executorch::llm::Stats* stats,
    std::unique_ptr<MethodMeta> target_method_meta,
    std::unique_ptr<MethodMeta> draft_kv_meta,
    std::unique_ptr<MethodMeta> draft_prefill_meta,
    executorch::extension::Module* emb_module,
    executorch::extension::Module* lm_head_module,
    std::unique_ptr<MethodMeta> emb_kv_meta,
    std::unique_ptr<MethodMeta> lm_head_kv_meta,
    std::unique_ptr<MethodMeta> emb_draft_meta,
    std::unique_ptr<MethodMeta> lm_head_draft_meta,
    float embeds_scale,
    int32_t embeds_zero_point,
    float logits_scale,
    int32_t logits_zero_point)
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
      draft_module_(draft_module),
      draft_kv_manager_(draft_kv_manager),
      draft_kv_meta_(std::move(draft_kv_meta)),
      draft_prefill_meta_(std::move(draft_prefill_meta)),
      dflash_meta_(metadata),
      emb_module_(emb_module),
      lm_head_module_(lm_head_module),
      emb_kv_meta_(std::move(emb_kv_meta)),
      lm_head_kv_meta_(std::move(lm_head_kv_meta)),
      emb_draft_meta_(std::move(emb_draft_meta)),
      lm_head_draft_meta_(std::move(lm_head_draft_meta)),
      embeds_scale_(embeds_scale),
      embeds_zero_point_(embeds_zero_point),
      logits_scale_(logits_scale),
      logits_zero_point_(logits_zero_point) {
  ET_CHECK_MSG(
      emb_module_ != nullptr && lm_head_module_ != nullptr,
      "[DFlash] headless decoder requires emb + lm_head modules");
  // The draft KVManager sizes its buffer as context_len - min(ar), and its
  // dtype scan matches kv_forward's cache length. Both assume kv_forward has
  // the shorter append length.
  ET_CHECK_MSG(
      dflash_meta_.prefill_ar_len >= dflash_meta_.block_size,
      "[DFlash] dflash_prefill_ar_len (%d) must be >= block_size (%d)",
      dflash_meta_.prefill_ar_len,
      dflash_meta_.block_size);
  ET_LOG(
      Info,
      "[DFlash] constructed: block=%d hidden=%d ctx_layers=%d draft_layers=%d "
      "max_ctx=%d prefill_ar=%d target_ar=%d mask_tok=%lld",
      dflash_meta_.block_size,
      dflash_meta_.hidden_dim,
      dflash_meta_.num_ctx_layers,
      dflash_meta_.num_draft_layers,
      dflash_meta_.max_context_len,
      dflash_meta_.prefill_ar_len,
      dflash_meta_.target_ar_len,
      static_cast<long long>(dflash_meta_.mask_token_id));
}

// ---------------------------------------------------------------------------
// init_io
// ---------------------------------------------------------------------------
size_t DFlashTokenGenerator::draft_io_size_in_bytes(
    const MethodMeta& kv_meta,
    const MethodMeta& prefill_meta) {
  size_t total = 0;
  for (size_t i = 0; i < kNumDraftNonKvInputs; ++i) {
    total += std::max(
        tensor_nbytes(kv_meta.input_tensor_meta(i).get()),
        tensor_nbytes(prefill_meta.input_tensor_meta(i).get()));
  }
  total += std::max(
      tensor_nbytes(kv_meta.output_tensor_meta(0).get()),
      tensor_nbytes(prefill_meta.output_tensor_meta(0).get()));
  // The in-graph lm_head's token ids, when present. Small, but RpcMem is a bump
  // allocator with no bounds check: anything init_io allocates and this function
  // forgets lands outside the ION region and the DSP fails the DMA (QNN 6002).
  const size_t Ld = (kv_meta.num_inputs() - kNumDraftNonKvInputs) / 2;
  const size_t kv_outs = 1 + 2 * Ld;
  if (kv_meta.num_outputs() > kv_outs) {
    total += tensor_nbytes(kv_meta.output_tensor_meta(kv_outs).get());
  }
  return total;
}

void DFlashTokenGenerator::init_io(
    IMemAlloc* buffer_manager,
    Result<MethodMeta> method_meta) {
  const int L = dflash_meta_.num_ctx_layers;
  const size_t expected_kv =
      1 + 2 * static_cast<size_t>(dflash_meta_.target_num_layers);
  const size_t num_outputs = method_meta->num_outputs();
  ET_CHECK_MSG(
      num_outputs >= expected_kv + static_cast<size_t>(L),
      "[DFlash] target pte missing %d hidden outputs (have %zu, need >=%zu)",
      L,
      num_outputs,
      expected_kv + static_cast<size_t>(L));
  // hidden + K/V, plus one optional logits output when the draft owns an lm_head.
  const size_t draft_kv_outs =
      1 + 2 * static_cast<size_t>(dflash_meta_.num_draft_layers);
  ET_CHECK_MSG(
      draft_kv_meta_->num_outputs() == draft_kv_outs ||
          draft_kv_meta_->num_outputs() == draft_kv_outs + 1,
      "[DFlash] draft kv_forward has %zu outputs, expected %zu (or %zu with an "
      "in-graph lm_head)",
      draft_kv_meta_->num_outputs(),
      draft_kv_outs,
      draft_kv_outs + 1);

  // Capture the L extra hidden output metas (order = ascending layer index).
  std::array<size_t, kMaxCtxLayers> hidden_nbytes{};
  const size_t Ltgt = static_cast<size_t>(dflash_meta_.target_num_layers);
  for (int i = 0; i < L; ++i) {
    Result<TensorInfo> hm = method_meta->output_tensor_meta(1 + 2 * Ltgt + i);
    target_hidden_dtypes_[i] = hm->scalar_type();
    target_hidden_sizes_[i].assign(hm->sizes().begin(), hm->sizes().end());
    target_hidden_dim_orders_[i].assign(
        hm->dim_order().begin(), hm->dim_order().end());
    hidden_nbytes[i] = tensor_nbytes(hm.get());
  }

  // Headless technique. The base allocates input[0] with `input_toks_.size`
  // bytes and output[0] with `logits_.size` bytes, but builds the TensorImpls
  // from the decoder's method_meta. For this headless decoder input[0] is embeds
  // (u16[1,ar,H]) and output[0] is hidden (u16[1,ar,H]), so overriding just the
  // two byte counts makes the base carve correctly-sized embeds/hidden buffers:
  // inputs_[0].data becomes the embeds-u16 buffer and output_tensors_[0].data the
  // hidden-u16 buffer, with mask/pos/KV bound as usual.
  input_toks_.size = method_meta->input_tensor_meta(0)->nbytes();
  logits_.size = method_meta->output_tensor_meta(0)->nbytes();

  // Standard target IO.
  TokenGenerator::init_io(buffer_manager, std::move(method_meta));

  // Bind the L hidden output tensors.
  for (int i = 0; i < L; ++i) {
    target_hidden_bufs_[i].resize(hidden_nbytes[i]);
    target_hidden_impls_[i] = std::make_unique<TensorImpl>(
        target_hidden_dtypes_[i],
        target_hidden_sizes_[i].size(),
        target_hidden_sizes_[i].data(),
        target_hidden_bufs_[i].data(),
        target_hidden_dim_orders_[i].data());
    output_tensors_.emplace_back(target_hidden_impls_[i].get());
  }

  // Cache B. prefill_forward runs first, so lay the cache out for its AR.
  draft_kv_manager_->init_cache(buffer_manager, dflash_meta_.prefill_ar_len);
  // The attention mask is additive, so it cannot neutralize a NaN read out of an
  // uninitialized fp16 cache slot. The target gets away with it because its cache
  // is quantized; this one is not.
  // Both graphs share these buffers, so each must be sized and registered for the
  // wider of the two views. k_new/v_new are [1, nKV, D, AR], and AR differs:
  // block_size for kv_forward, prefill_ar_len for prefill_forward. Registering only
  // kv_forward's view leaves the DSP writing prefill's larger tensor into a region
  // QNN was told is smaller -- a DMA overrun that surfaces as skelExecute 1003.
  const size_t cache_bytes = std::max(
      tensor_nbytes(draft_kv_meta_->input_tensor_meta(kNumDraftNonKvInputs).get()),
      tensor_nbytes(
          draft_prefill_meta_->input_tensor_meta(kNumDraftNonKvInputs).get()));
  const size_t cache_out_bytes = std::max(
      tensor_nbytes(draft_kv_meta_->output_tensor_meta(1).get()),
      tensor_nbytes(draft_prefill_meta_->output_tensor_meta(1).get()));
  const int Ld = dflash_meta_.num_draft_layers;
  for (int l = 0; l < Ld; ++l) {
    std::memset(draft_kv_manager_->get_k_cache_()[l].buffer, 0, cache_bytes);
    std::memset(draft_kv_manager_->get_v_cache_()[l].buffer, 0, cache_bytes);
  }

  // KVManager only carves the buffers out of the shared region; binding them as
  // QNN shared-memory tensors is the consumer's job (see TokenGenerator::init_io).
  // Skipping it makes QnnManager::RegisterMem fall back to a raw FastRPC copy of
  // every cache tensor, and the draft's ~21MB overruns the transport (QNN 1003).
  const std::vector<KVCache>& kc = draft_kv_manager_->get_k_cache_();
  const std::vector<KVCache>& vc = draft_kv_manager_->get_v_cache_();
  // Register each buffer under its widest view: the cache inputs are widest in
  // kv_forward (Cc = context_len - block_size), the cache outputs in
  // prefill_forward (AR = prefill_ar_len). Same rule as the non-K/V IO below.
  for (int l = 0; l < Ld; ++l) {
    auto k_in = draft_kv_meta_->input_tensor_meta(kNumDraftNonKvInputs + l).get();
    auto v_in =
        draft_kv_meta_->input_tensor_meta(kNumDraftNonKvInputs + Ld + l).get();
    auto k_out = draft_prefill_meta_->output_tensor_meta(1 + l).get();
    auto v_out = draft_prefill_meta_->output_tensor_meta(1 + Ld + l).get();
    buffer_manager->add_memory_info(kc[l].buffer, cache_bytes, k_in);
    buffer_manager->add_memory_info(vc[l].buffer, cache_bytes, v_in);
    buffer_manager->add_memory_info(kc[l].output_buffer, cache_out_bytes, k_out);
    buffer_manager->add_memory_info(vc[l].output_buffer, cache_out_bytes, v_out);
  }

  // kv_forward's append length, read off new_context [1, AR, L*H]. It is B+1 under
  // the shifted convention and B under the aligned one, so it cannot be derived
  // from block_size alone.
  draft_decode_ar_ =
      static_cast<int32_t>(draft_kv_meta_->input_tensor_meta(2)->sizes()[1]);

  // The draft's non-K/V IO, likewise shared + registered. Each buffer takes the
  // wider of the two graphs' views so both can bind it.
  for (size_t i = 0; i < kNumDraftNonKvInputs; ++i) {
    draft_in_nbytes_[i] = std::max(
        tensor_nbytes(draft_kv_meta_->input_tensor_meta(i).get()),
        tensor_nbytes(draft_prefill_meta_->input_tensor_meta(i).get()));
    draft_in_bufs_[i] = buffer_manager->allocate(draft_in_nbytes_[i]);
    buffer_manager->add_memory_info(
        draft_in_bufs_[i],
        draft_in_nbytes_[i],
        draft_prefill_meta_->input_tensor_meta(i).get());
  }
  draft_hidden_nbytes_ = std::max(
      tensor_nbytes(draft_kv_meta_->output_tensor_meta(0).get()),
      tensor_nbytes(draft_prefill_meta_->output_tensor_meta(0).get()));
  draft_hidden_buf_ = buffer_manager->allocate(draft_hidden_nbytes_);
  buffer_manager->add_memory_info(
      draft_hidden_buf_,
      draft_hidden_nbytes_,
      draft_prefill_meta_->output_tensor_meta(0).get());

  // A draft built from a checkpoint with an lm_head appends the block's logits as
  // one extra output, so the vocab projection runs on HTP instead of scanning the
  // 743 MB embedding table on the CPU.
  const size_t kv_outs = 1 + 2 * static_cast<size_t>(Ld);
  draft_has_lm_head_ = draft_kv_meta_->num_outputs() > kv_outs;
  if (draft_has_lm_head_) {
    auto l_meta = draft_kv_meta_->output_tensor_meta(kv_outs).get();
    draft_logits_nbytes_ = tensor_nbytes(l_meta);
    auto l_sizes = l_meta.sizes();
    draft_vocab_size_ = static_cast<size_t>(l_sizes[l_sizes.size() - 1]);
    draft_logits_buf_ = buffer_manager->allocate(draft_logits_nbytes_);
    buffer_manager->add_memory_info(
        draft_logits_buf_, draft_logits_nbytes_, l_meta);
  }
  ET_LOG(
      Info,
      "[DFlash] draft lm_head: %s",
      draft_has_lm_head_ ? "in-graph (HTP)" : "external lm_head.pte");

  // Headless companions: emb.pte token input + f32 embeds output, lm_head.pte
  // f32 logits output, and a u16 buffer for the draft's quantized hidden (the
  // verify path feeds lm_head the decoder hidden buffer directly, so only the
  // draft needs its own). All carved from the shared region + QNN-registered;
  // sized for the kv views this generator binds (decode never runs prefill).
  emb_tok_nbytes_ = emb_kv_meta_->input_tensor_meta(0)->nbytes();
  emb_tok_buf_ = buffer_manager->allocate(emb_tok_nbytes_);
  buffer_manager->add_memory_info(
      emb_tok_buf_, emb_tok_nbytes_, emb_kv_meta_->input_tensor_meta(0).get());

  emb_out_nbytes_ = emb_kv_meta_->output_tensor_meta(0)->nbytes();
  emb_out_buf_ = buffer_manager->allocate(emb_out_nbytes_);
  buffer_manager->add_memory_info(
      emb_out_buf_, emb_out_nbytes_, emb_kv_meta_->output_tensor_meta(0).get());

  lm_head_logits_nbytes_ = lm_head_kv_meta_->output_tensor_meta(0)->nbytes();
  lm_head_logits_buf_ = buffer_manager->allocate(lm_head_logits_nbytes_);
  buffer_manager->add_memory_info(
      lm_head_logits_buf_,
      lm_head_logits_nbytes_,
      lm_head_kv_meta_->output_tensor_meta(0).get());

  draft_hidden_u16_nbytes_ = lm_head_kv_meta_->input_tensor_meta(0)->nbytes();
  draft_hidden_u16_buf_ = buffer_manager->allocate(draft_hidden_u16_nbytes_);
  buffer_manager->add_memory_info(
      draft_hidden_u16_buf_,
      draft_hidden_u16_nbytes_,
      lm_head_kv_meta_->input_tensor_meta(0).get());

  // The block_size-wide views share these buffers -- they are strictly smaller
  // than the kv views -- but QNN registers memory per tensor meta, so each view
  // has to be declared or the DSP has no handle for it.
  if (emb_draft_meta_ != nullptr) {
    buffer_manager->add_memory_info(
        emb_tok_buf_,
        emb_tok_nbytes_,
        emb_draft_meta_->input_tensor_meta(0).get());
    buffer_manager->add_memory_info(
        emb_out_buf_,
        emb_out_nbytes_,
        emb_draft_meta_->output_tensor_meta(0).get());
  }
  if (lm_head_draft_meta_ != nullptr) {
    buffer_manager->add_memory_info(
        draft_hidden_u16_buf_,
        draft_hidden_u16_nbytes_,
        lm_head_draft_meta_->input_tensor_meta(0).get());
    buffer_manager->add_memory_info(
        lm_head_logits_buf_,
        lm_head_logits_nbytes_,
        lm_head_draft_meta_->output_tensor_meta(0).get());
  }
  ET_LOG(
      Info,
      "[DFlash] draft-width heads: emb %s, lm_head %s",
      emb_draft_meta_ ? "yes" : "no (falls back to the tree-width graph)",
      lm_head_draft_meta_ ? "yes" : "no (falls back to the tree-width graph)");

  lm_head_vocab_size_ =
      static_cast<int32_t>(lm_head_kv_meta_->output_tensor_meta(0)->sizes()[2]);
  ET_LOG(
      Info,
      "[DFlash] aux IO: emb_out %.2f MB, lm_head_logits %.2f MB (vocab %d)",
      static_cast<double>(emb_out_nbytes_) / (1 << 20),
      static_cast<double>(lm_head_logits_nbytes_) / (1 << 20),
      lm_head_vocab_size_);

  // Cache A: one draft-graph call's worth of fused 5H rows.
  const size_t LH = static_cast<size_t>(L) * dflash_meta_.hidden_dim;
  const int32_t stage_rows =
      std::max(dflash_meta_.prefill_ar_len, dflash_meta_.block_size);
  stage_buf_.assign(static_cast<size_t>(stage_rows) * LH, 0.0f);
  stage_count_ = 0;
  draft_ctx_len_ = 0;

  ET_LOG(
      Info,
      "[DFlash] init_io: staging %d rows x %zu fp32 (%.1f MB), draft kv cache "
      "%.1f MB",
      stage_rows,
      LH,
      static_cast<double>(stage_buf_.size() * sizeof(float)) / (1 << 20),
      static_cast<double>(draft_kv_manager_->total_cache_size_in_bytes()) /
          (1 << 20));
}

// ---------------------------------------------------------------------------
// emb.pte / lm_head.pte — the headless decoder's split-off projections.
// Both are separate QNN contexts sharing the same ION region; their IO is
// pre-registered in init_io, so a call just rebinds the TensorImpls and runs.
// ---------------------------------------------------------------------------
void DFlashTokenGenerator::run_embedding(
    const uint64_t* tokens,
    int32_t n_tokens,
    bool draft) {
  const bool use_draft = draft && emb_draft_meta_ != nullptr;
  MethodMeta* meta = use_draft ? emb_draft_meta_.get() : emb_kv_meta_.get();
  const char* method = use_draft ? "tok_embedding_draft_forward"
                                 : "tok_embedding_kv_forward";
  auto tok_meta = meta->input_tensor_meta(0).get();
  auto out_meta = meta->output_tensor_meta(0).get();
  const int32_t ar = static_cast<int32_t>(tok_meta.sizes()[1]);
  std::memset(emb_tok_buf_, 0, emb_tok_nbytes_);
  const bool tok_i64 = tok_meta.scalar_type() == ScalarType::Long;
  for (int32_t k = 0; k < n_tokens && k < ar; ++k) {
    if (tok_i64) {
      reinterpret_cast<int64_t*>(emb_tok_buf_)[k] =
          static_cast<int64_t>(tokens[k]);
    } else {
      reinterpret_cast<int32_t*>(emb_tok_buf_)[k] =
          static_cast<int32_t>(tokens[k]);
    }
  }
  TensorImpl tok_impl = make_impl(tok_meta, emb_tok_buf_);
  TensorImpl out_impl = make_impl(out_meta, emb_out_buf_);
  std::vector<EValue> ins{EValue(Tensor(&tok_impl))};
  std::vector<EValue> outs{EValue(Tensor(&out_impl))};
  ET_CHECK_MSG(
      emb_module_->set_outputs(method, outs) == Error::Ok,
      "[DFlash] emb set_outputs failed");
  const long t0 = time_in_ms();
  auto res = emb_module_->execute(method, ins);
  emb_exec_ms_ += static_cast<double>(time_in_ms() - t0);
  ++emb_calls_;
  ET_CHECK_MSG(res.ok(), "[DFlash] emb execute failed");
  if (!dtype_checked_) {
    dtype_checked_ = true;
    check_payload_dtype(
        "emb.pte out",
        emb_out_buf_,
        static_cast<size_t>(ar) * dflash_meta_.hidden_dim,
        out_meta.scalar_type());
  }
}

void DFlashTokenGenerator::run_lm_head(std::byte* hidden_u16, bool draft) {
  const bool use_draft = draft && lm_head_draft_meta_ != nullptr;
  MethodMeta* meta =
      use_draft ? lm_head_draft_meta_.get() : lm_head_kv_meta_.get();
  const char* method =
      use_draft ? "lm_head_draft_forward" : "lm_head_kv_forward";
  auto in_meta = meta->input_tensor_meta(0).get();
  auto out_meta = meta->output_tensor_meta(0).get();
  TensorImpl in_impl = make_impl(in_meta, hidden_u16);
  TensorImpl out_impl = make_impl(out_meta, lm_head_logits_buf_);
  std::vector<EValue> ins{EValue(Tensor(&in_impl))};
  std::vector<EValue> outs{EValue(Tensor(&out_impl))};
  ET_CHECK_MSG(
      lm_head_module_->set_outputs(method, outs) == Error::Ok,
      "[DFlash] lm_head set_outputs failed");
  const long t0 = time_in_ms();
  auto res = lm_head_module_->execute(method, ins);
  lm_head_exec_ms_ += static_cast<double>(time_in_ms() - t0);
  ++lm_head_calls_;
  ET_CHECK_MSG(res.ok(), "[DFlash] lm_head execute failed");
  static bool lm_checked = false;
  if (!lm_checked) {
    lm_checked = true;
    check_payload_dtype(
        "lm_head.pte out",
        lm_head_logits_buf_,
        static_cast<size_t>(out_meta.sizes()[1]) * lm_head_vocab_size_,
        out_meta.scalar_type());
  }
}

// ---------------------------------------------------------------------------
// A pte's program can declare an output Float while the delegate writes uint16
// codes. probe tooling and method_meta both report the DECLARED value, so the
// only reliable check is to look at the payload itself: u16 codes read as f32
// produce absurd magnitudes (two ~30000 codes concatenate into ~1e30+), while
// genuine f32 activations stay within a sane range.
// ---------------------------------------------------------------------------
void DFlashTokenGenerator::check_payload_dtype(
    const char* name,
    const std::byte* buf,
    size_t n_elems,
    ScalarType declared) {
  if (declared != ScalarType::Float || n_elems == 0) {
    return; // only Float declarations can hide a u16 payload
  }
  const size_t n = std::min<size_t>(n_elems, 4096);
  const float* f = reinterpret_cast<const float*>(buf);
  const uint16_t* u = reinterpret_cast<const uint16_t*>(buf);
  size_t absurd = 0, nonzero = 0;
  double u_sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const float x = f[i];
    if (std::isfinite(x)) {
      const float a = std::fabs(x);
      if (a > 1e20f) {
        ++absurd;
      }
      if (a > 0.0f) {
        ++nonzero;
      }
    } else {
      ++absurd;
    }
    u_sum += static_cast<double>(u[i]);
  }
  const double absurd_frac = static_cast<double>(absurd) / static_cast<double>(n);
  if (absurd_frac > 0.10) {
    ET_LOG(
        Error,
        "[DFlash][dtype] %s declares Float but %.0f%% of the payload is absurd "
        "as f32 (|x|>1e20) -- it is almost certainly uint16. Mean u16 code "
        "%.0f. Reading it as f32 will corrupt everything downstream.",
        name,
        absurd_frac * 100.0,
        u_sum / static_cast<double>(n));
  } else {
    ET_LOG(
        Info,
        "[DFlash][dtype] %s: payload consistent with declared Float (%.0f%% "
        "nonzero, no absurd magnitudes).",
        name,
        100.0 * static_cast<double>(nonzero) / static_cast<double>(n));
  }
}

// Quantize `rows` draft hidden rows to the lm_head's u16 input encoding, run
// lm_head.pte, then keep each row's top-k as log-probs.
void DFlashTokenGenerator::draft_topk(
    const float* hidden,
    int32_t rows,
    int32_t k) {
  const int H = dflash_meta_.hidden_dim;
  const long lm_start_ms = time_in_ms();
  const int64_t t_head = time_in_us();
  std::memset(draft_hidden_u16_buf_, 0, draft_hidden_u16_nbytes_);
  // TODO(dflash-uint16): 临时桥接。emb输出/draft-IO 目前是 f32(编译端边界没做成
  // uint16),故此处 host quantize。日后编译端把 emb 输出 + draft noise/hidden tag
  // 成 uint16 后,此 quantize 可删、改 buffer 直传。详见 dflash/RUNNER_M5_PLAN.md
  // "技术债" 节。
  // Its own encoding when the draft-width graph exists: that one calibrated on
  // draft hidden, which spans a fifth of the target's, so the target's scale
  // would spend 2.3 bit of the u16 range on values that never appear here.
  const bool own_head = lm_head_draft_meta_ != nullptr;
  quantize_f32_to_u16(
      hidden,
      reinterpret_cast<uint16_t*>(draft_hidden_u16_buf_),
      static_cast<size_t>(rows) * H,
      own_head ? dflash_meta_.draft_hidden_scale : logits_scale_,
      own_head ? dflash_meta_.draft_hidden_zero_point : logits_zero_point_);
  run_lm_head(draft_hidden_u16_buf_, /*draft=*/true);
  stage_us_[kDraftHead] += time_in_us() - t_head;

  const int64_t t_pick = time_in_us();
  const long t_smp = time_in_ms();
  const size_t V = static_cast<size_t>(lm_head_vocab_size_);
  const uint16_t* lg = reinterpret_cast<const uint16_t*>(lm_head_logits_buf_);
  // These are the DRAFT's code gaps, so they want the scale the draft-width graph
  // calibrated on draft logits. The other one is dominated by the target's.
  const float s = own_head ? dflash_meta_.draft_logit_out_scale
                           : dflash_meta_.logit_out_scale;
  draft_ids_.assign(static_cast<size_t>(rows) * k, 0);
  draft_logp_.assign(static_cast<size_t>(rows) * k, 0.0f);
  std::vector<uint16_t> codes(static_cast<size_t>(k));
  for (int d = 0; d < rows; ++d) {
    uint64_t* ids = draft_ids_.data() + static_cast<size_t>(d) * k;
    float* logp = draft_logp_.data() + static_cast<size_t>(d) * k;
    topk_u16_row(lg + static_cast<size_t>(d) * V, V, k, codes.data(), ids);
    // The zero point cancels in a difference, so it never has to be known here;
    // only the scale survives, and it is what turns a code gap into nats.
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
      logp[i] = s * (static_cast<float>(codes[i]) - static_cast<float>(codes[0]));
      sum += std::exp(static_cast<double>(logp[i]));
    }
    const float log_z = static_cast<float>(std::log(sum));
    for (int i = 0; i < k; ++i) {
      logp[i] -= log_z;
    }
  }
  sample_ms_ += static_cast<double>(time_in_ms() - t_smp);
  stage_us_[kDraftPick] += time_in_us() - t_pick;
  lm_head_time_ms_ += static_cast<double>(time_in_ms() - lm_start_ms);
}

// ---------------------------------------------------------------------------
// DDTree — best-first over the product of the per-depth distributions.
//
// A node at depth d draws its token from depth d's own top-k, independent of
// what was chosen above it. That independence is the whole reason this is free:
// block diffusion already produced every depth's distribution in the one draft
// forward, so a wider tree never costs another one (EAGLE-style trees need a
// draft call per level -- see eagle_token_generator.cpp:1118).
//
// Popping order gives prefix closure for nothing: extending a path adds a
// log-prob, which is <= 0, so a parent always outranks its children and is
// always popped first. init_attention_mask depends on exactly that -- it builds
// a node's mask row by copying its parent's, which must already be written.
//
// With k == 1 the sibling push never fires and this degenerates into the chain:
// one argmax per depth, `depth_limit` nodes, single path. That is the whole
// tree_budget == 0 path, so the chain runs the same code as the tree.
// ---------------------------------------------------------------------------
void DFlashTokenGenerator::note_committed(uint64_t token) {
  const int32_t n = static_cast<int32_t>(committed_tail_.size());
  for (int32_t lag = 1; lag <= std::min(n, kRepLag); ++lag) {
    if (committed_tail_[n - lag] == token) {
      repeat_seen_[lag].insert(token);
    }
  }
  committed_tail_.push_back(token);
  if (static_cast<int32_t>(committed_tail_.size()) > kRepLag) {
    committed_tail_.erase(committed_tail_.begin());
  }
}

bool DFlashTokenGenerator::token_is_numeric(uint64_t token) {
  const auto it = numeric_cache_.find(token);
  if (it != numeric_cache_.end()) {
    return it->second;
  }
  bool numeric = false;
  auto piece = tokenizer_->decode(token, token);
  if (piece.ok()) {
    for (const char ch : piece.get()) {
      if (ch >= '0' && ch <= '9') {
        numeric = true;
        break;
      }
    }
  }
  numeric_cache_.emplace(token, numeric);
  return numeric;
}

float DFlashTokenGenerator::repeat_penalty(
    uint64_t token,
    const uint64_t* path,
    int32_t path_len) {
  // Nearest occurrence wins: a token two slots back that is also five slots back
  // is charged as lag 2, matching how the table was measured.
  ++rep_scored_;
  for (int32_t lag = 1; lag <= std::min(path_len, kRepLag); ++lag) {
    if (path[path_len - lag] == token) {
      const bool gated = repeat_seen_[lag].count(token) > 0;
      ++rep_charged_;
      rep_lag1_ += (lag == 1);
      rep_gated_ += gated;
      return kRepPenalty[gated ? 1 : 0][lag - 1];
    }
  }
  return 0.0f;
}

void DFlashTokenGenerator::fill_candidates(
    int32_t node,
    int32_t depth,
    int32_t k,
    bool calib) {
  const size_t base = static_cast<size_t>(depth - 1) * k;
  const float* lp = draft_logp_.data() + base;
  const uint64_t* id = draft_ids_.data() + base;
  float* sc = cand_score_.data() + static_cast<size_t>(node) * k;
  uint64_t* tk = cand_token_.data() + static_cast<size_t>(node) * k;
  const uint64_t* path = path_tail_.data() + static_cast<size_t>(node) * kRepLag;
  const int32_t plen = path_len_[node];
  bool charged = false;
  for (int32_t c = 0; c < k; ++c) {
    const float pen = calib ? repeat_penalty(id[c], path, plen) : 0.0f;
    charged |= (pen != 0.0f);
    sc[c] = lp[c] - pen;
    tk[c] = id[c];
  }
  if (!charged) {
    return; // draft_topk left the row descending, so it is already sorted
  }
  cand_order_.resize(k);
  cand_tmp_score_.resize(k);
  cand_tmp_token_.resize(k);
  for (int32_t c = 0; c < k; ++c) {
    cand_order_[c] = c;
  }
  std::stable_sort(
      cand_order_.begin(), cand_order_.end(), [sc](int32_t a, int32_t b) {
        return sc[a] > sc[b];
      });
  for (int32_t c = 0; c < k; ++c) {
    cand_tmp_score_[c] = sc[cand_order_[c]];
    cand_tmp_token_[c] = tk[cand_order_[c]];
  }
  std::copy(cand_tmp_score_.begin(), cand_tmp_score_.end(), sc);
  std::copy(cand_tmp_token_.begin(), cand_tmp_token_.end(), tk);
}

void DFlashTokenGenerator::probe_round(
    const std::vector<uint64_t>& true_tokens,
    int32_t n_draft) {
  const int32_t k = draft_topk_;
  std::vector<uint64_t> tail(committed_tail_);
  for (size_t d = 1; d <= true_tokens.size() && d <= static_cast<size_t>(n_draft);
       ++d) {
    const uint64_t truth = true_tokens[d - 1];
    const size_t base = (d - 1) * static_cast<size_t>(k);
    const int32_t plen = static_cast<int32_t>(tail.size());
    for (int32_t c = 0; c < k; ++c) {
      const uint64_t z = draft_ids_[base + c];
      const double q = std::exp(static_cast<double>(draft_logp_[base + c]));
      size_t bucket = 0;
      for (int32_t lag = 1; lag <= std::min(plen, kRepLag); ++lag) {
        if (tail[plen - lag] == z) {
          const size_t gated = repeat_seen_[lag].count(z) ? 1 : 0;
          const size_t numeric = token_is_numeric(z) ? 1 : 0;
          bucket = 1 + (gated * 2 + numeric) * kRepLag + (lag - 1);
          break;
        }
      }
      probe_q_[bucket] += q;
      probe_hit_[bucket] += (z == truth) ? 1.0 : 0.0;
      ++probe_n_[bucket];
    }
    tail.push_back(truth);
    if (static_cast<int32_t>(tail.size()) > kRepLag) {
      tail.erase(tail.begin());
    }
  }
}

void DFlashTokenGenerator::build_tree(int32_t budget, int32_t depth_limit) {
  struct Cand {
    float logw;
    int32_t parent; // slot
    int32_t ptr; // position in the PARENT's candidate ordering
    bool operator<(const Cand& o) const {
      return logw < o.logw; // priority_queue pops the greatest
    }
  };
  const int32_t k = draft_topk_;
  const bool calib = dflash_meta_.repeat_calib && dflash_meta_.tree_budget > 0;
  const size_t nslot = static_cast<size_t>(budget) + 1;
  tree_.clear();
  tree_children_.assign(nslot, {});
  cand_score_.resize(nslot * k);
  cand_token_.resize(nslot * k);
  path_tail_.resize(nslot * kRepLag);
  path_len_.resize(nslot);

  // Node 0 is the confirmed root, and its path is what has already been
  // committed -- a depth-1 candidate equal to the root token is the single most
  // common repeat, so this seeding is not an edge case.
  path_len_[0] = static_cast<int32_t>(committed_tail_.size());
  std::copy(committed_tail_.begin(), committed_tail_.end(), path_tail_.begin());

  std::priority_queue<Cand> heap;
  fill_candidates(0, 1, k, calib);
  heap.push(Cand{cand_score_[0], 0, 0});
  while (!heap.empty() && static_cast<int32_t>(tree_.size()) < budget) {
    const Cand c = heap.top();
    heap.pop();
    const size_t pbase = static_cast<size_t>(c.parent) * k;
    const uint64_t token = cand_token_[pbase + c.ptr];
    const int32_t slot = static_cast<int32_t>(tree_.size()) + 1;
    const int32_t depth =
        c.parent == 0 ? 1 : tree_[static_cast<size_t>(c.parent) - 1].depth + 1;
    tree_.push_back(TreeNode{token, c.parent, depth});
    tree_children_[c.parent].emplace_back(token, slot);

    const uint64_t* ppath =
        path_tail_.data() + static_cast<size_t>(c.parent) * kRepLag;
    uint64_t* npath = path_tail_.data() + static_cast<size_t>(slot) * kRepLag;
    const int32_t plen = path_len_[c.parent];
    if (plen < kRepLag) {
      std::copy(ppath, ppath + plen, npath);
      npath[plen] = token;
      path_len_[slot] = plen + 1;
    } else {
      std::copy(ppath + 1, ppath + kRepLag, npath);
      npath[kRepLag - 1] = token;
      path_len_[slot] = kRepLag;
    }

    if (c.ptr + 1 < k) {
      heap.push(Cand{
          c.logw - cand_score_[pbase + c.ptr] + cand_score_[pbase + c.ptr + 1],
          c.parent,
          c.ptr + 1});
    }
    if (depth < depth_limit) {
      fill_candidates(slot, depth + 1, k, calib);
      heap.push(
          Cand{c.logw + cand_score_[static_cast<size_t>(slot) * k], slot, 0});
    }
  }
}

int32_t DFlashTokenGenerator::tree_child(int32_t parent, uint64_t token) const {
  for (const auto& [tok, slot] : tree_children_[parent]) {
    if (tok == token) {
      return slot;
    }
  }
  return -1;
}

// ---------------------------------------------------------------------------
// One draft graph call (prefill_forward or kv_forward).
//
// Cache B holds `draft_ctx_len_` committed rows. This call appends `n_new` more
// (staged in `stage_buf_`, RoPE positions ctx_pos_base + j) and drafts the block
// against [cached ‖ new ‖ block]. Block K/V are transient by construction — the
// graph simply does not emit them, which is what the reference achieves with
// past_key_values_draft.crop(start).
// ---------------------------------------------------------------------------
void DFlashTokenGenerator::run_draft(
    const std::string& method,
    const MethodMeta& meta,
    int32_t ar,
    int32_t n_new,
    int64_t ctx_pos_base,
    const std::vector<uint64_t>& block_tokens,
    int64_t block_pos_base,
    std::vector<float>* out_hidden) {
  const int64_t t_prep = time_in_us();
  const int B = dflash_meta_.block_size;
  const int H = dflash_meta_.hidden_dim;
  const int Ld = dflash_meta_.num_draft_layers;
  const size_t LH =
      static_cast<size_t>(dflash_meta_.num_ctx_layers) * dflash_meta_.hidden_dim;
  const int32_t n_past = draft_ctx_len_;
  const int32_t Cc = dflash_meta_.max_context_len - ar;
  const int32_t mask_w = Cc + ar + B;

  // KVManager strides each cache row by (context_len - ar), so a graph with
  // append length `ar` can only ever address `Cc` committed rows.
  ET_CHECK_MSG(
      n_past + n_new <= Cc,
      "[DFlash] draft cache overflow: %d + %d > %d (ar=%d)",
      n_past,
      n_new,
      Cc,
      ar);
  // new_context is [1, ar, L*H]. Rows past `ar` would be written into the host
  // staging buffer (sized for the wider graph) and then silently dropped by the
  // bind, leaving draft_ctx_len_ ahead of what the cache actually holds -- a
  // desync the draft never recovers from.
  ET_CHECK_MSG(
      n_new <= ar,
      "[DFlash] %s appends %d context rows but its graph takes only %d",
      method.c_str(),
      n_new,
      ar);

  auto n_meta = meta.input_tensor_meta(0).get();
  auto m_meta = meta.input_tensor_meta(1).get();
  auto c_meta = meta.input_tensor_meta(2).get();
  auto cp_meta = meta.input_tensor_meta(3).get();
  auto bp_meta = meta.input_tensor_meta(4).get();
  auto h_meta = meta.output_tensor_meta(0).get();

  // Shared + QNN-registered buffers (see init_io); a heap buffer here would be
  // copied through FastRPC on every call.
  std::byte* noise_buf = draft_in_bufs_[0];
  std::byte* mask_buf = draft_in_bufs_[1];
  std::byte* ctx_buf = draft_in_bufs_[2];
  std::byte* cpos_buf = draft_in_bufs_[3];
  std::byte* bpos_buf = draft_in_bufs_[4];
  std::byte* hidden_buf = draft_hidden_buf_;
  // Padded rows must read as zero; mask is fully rewritten below.
  std::memset(noise_buf, 0, draft_in_nbytes_[0]);
  std::memset(ctx_buf, 0, draft_in_nbytes_[2]);
  std::memset(cpos_buf, 0, draft_in_nbytes_[3]);
  std::memset(bpos_buf, 0, draft_in_nbytes_[4]);

  // noise_embedding: emb.pte over the block tokens (<= B <= emb kv AR).
  // emb.pte emits uint16 in the decoder's embeds encoding (the program declares
  // the output Float, but the payload is u16 -- see target_verify_block). The
  // draft's IO is genuinely f32, so dequantize across this boundary.
  const int32_t n_block =
      std::min(static_cast<int32_t>(block_tokens.size()), B);
  run_embedding(block_tokens.data(), n_block, /*draft=*/true);
  const uint16_t* emb_rows = reinterpret_cast<const uint16_t*>(emb_out_buf_);
  {
    const ScalarType ndt = n_meta.scalar_type();
    const size_t n_elem = static_cast<size_t>(n_block) * H;
    const float zp = static_cast<float>(embeds_zero_point_);
    if (ndt == ScalarType::Float) {
      float* d = reinterpret_cast<float*>(noise_buf);
      for (size_t idx = 0; idx < n_elem; ++idx) {
        d[idx] = (static_cast<float>(emb_rows[idx]) - zp) * embeds_scale_;
      }
    } else {
      for (size_t idx = 0; idx < n_elem; ++idx) {
        write_float(
            noise_buf,
            ndt,
            idx,
            (static_cast<float>(emb_rows[idx]) - zp) * embeds_scale_);
      }
    }
  }

  // atten_mask [B, Cc + ar + B]: every block query sees the committed cache, the
  // valid new-context rows, and the whole block (bidirectional denoising).
  // Visibility depends only on the column, so every query row is the same row of
  // bytes: lay down five contiguous runs once and copy it B-1 times, instead of
  // B * mask_w calls each re-deciding the dtype.
  {
    const ScalarType mdt = m_meta.scalar_type();
    const size_t row_bytes = static_cast<size_t>(mask_w) * dtype_size(mdt);
    auto fill = [&](int from, int to, bool vis) {
      if (to <= from) {
        return;
      }
      switch (mdt) {
        case ScalarType::Float: {
          float* r = reinterpret_cast<float*>(mask_buf);
          std::fill(r + from, r + to, vis ? 0.0f : -65504.0f);
          break;
        }
        case ScalarType::Half:
        case ScalarType::UInt16: {
          uint16_t* r = reinterpret_cast<uint16_t*>(mask_buf);
          std::fill(
              r + from, r + to, static_cast<uint16_t>(vis ? 0x0000 : 0xFBFF));
          break;
        }
        default:
          for (int c = from; c < to; ++c) {
            write_mask(mask_buf, mdt, static_cast<size_t>(c), vis);
          }
          break;
      }
    };
    const int vis_ctx = std::min(n_past, Cc);
    const int vis_new = Cc + std::min(n_new, ar);
    fill(0, vis_ctx, true);
    fill(vis_ctx, Cc, false);
    fill(Cc, vis_new, true);
    fill(vis_new, Cc + ar, false);
    fill(Cc + ar, mask_w, true);
    for (int q = 1; q < B; ++q) {
      std::memcpy(
          mask_buf + static_cast<size_t>(q) * row_bytes, mask_buf, row_bytes);
    }
  }

  // new_context + its RoPE positions. Padded rows stay zero and are masked out;
  // update_cache only commits the first n_new of them.
  for (int j = 0; j < n_new; ++j) {
    write_floats(
        ctx_buf,
        c_meta.scalar_type(),
        static_cast<size_t>(j) * LH,
        LH,
        stage_buf_.data() + static_cast<size_t>(j) * LH);
    reinterpret_cast<int32_t*>(cpos_buf)[j] =
        static_cast<int32_t>(ctx_pos_base + j);
  }
  for (int k = 0; k < B; ++k) {
    reinterpret_cast<int32_t*>(bpos_buf)[k] =
        static_cast<int32_t>(block_pos_base + k);
  }

  draft_kv_manager_->rearrange_cache(ar);
  const std::vector<KVCache>& k_caches = draft_kv_manager_->get_k_cache_();
  const std::vector<KVCache>& v_caches = draft_kv_manager_->get_v_cache_();

  TensorImpl n_impl = make_impl(n_meta, noise_buf);
  TensorImpl m_impl = make_impl(m_meta, mask_buf);
  TensorImpl c_impl = make_impl(c_meta, ctx_buf);
  TensorImpl cp_impl = make_impl(cp_meta, cpos_buf);
  TensorImpl bp_impl = make_impl(bp_meta, bpos_buf);

  std::vector<TensorInfo> kv_in_meta, kv_out_meta;
  kv_in_meta.reserve(2 * Ld);
  kv_out_meta.reserve(2 * Ld);
  for (int l = 0; l < Ld; ++l) {
    kv_in_meta.emplace_back(meta.input_tensor_meta(5 + l).get());
    kv_out_meta.emplace_back(meta.output_tensor_meta(1 + l).get());
  }
  for (int l = 0; l < Ld; ++l) {
    kv_in_meta.emplace_back(meta.input_tensor_meta(5 + Ld + l).get());
    kv_out_meta.emplace_back(meta.output_tensor_meta(1 + Ld + l).get());
  }

  std::vector<TensorImpl> kv_in_impls;
  kv_in_impls.reserve(2 * Ld);
  for (int l = 0; l < Ld; ++l) {
    kv_in_impls.emplace_back(make_impl(kv_in_meta[l], k_caches[l].buffer));
  }
  for (int l = 0; l < Ld; ++l) {
    kv_in_impls.emplace_back(make_impl(kv_in_meta[Ld + l], v_caches[l].buffer));
  }

  std::vector<EValue> inputs{
      EValue(Tensor(&n_impl)),
      EValue(Tensor(&m_impl)),
      EValue(Tensor(&c_impl)),
      EValue(Tensor(&cp_impl)),
      EValue(Tensor(&bp_impl)),
  };
  for (auto& impl : kv_in_impls) {
    inputs.emplace_back(EValue(Tensor(&impl)));
  }

  TensorImpl h_impl = make_impl(h_meta, hidden_buf);
  std::vector<TensorImpl> kv_out_impls;
  kv_out_impls.reserve(2 * Ld);
  for (int l = 0; l < Ld; ++l) {
    kv_out_impls.emplace_back(
        make_impl(kv_out_meta[l], k_caches[l].output_buffer));
  }
  for (int l = 0; l < Ld; ++l) {
    kv_out_impls.emplace_back(
        make_impl(kv_out_meta[Ld + l], v_caches[l].output_buffer));
  }
  std::vector<EValue> outputs{EValue(Tensor(&h_impl))};
  for (auto& impl : kv_out_impls) {
    outputs.emplace_back(EValue(Tensor(&impl)));
  }
  // TensorImpl is neither default-constructible nor assignable, so it has to be
  // built in place; the vector just gives it a stable address to hand to EValue.
  std::vector<TensorImpl> logits_impl;
  if (draft_has_lm_head_) {
    logits_impl.reserve(1);
    logits_impl.emplace_back(
        make_impl(meta.output_tensor_meta(1 + 2 * Ld).get(), draft_logits_buf_));
    outputs.emplace_back(EValue(Tensor(&logits_impl[0])));
  }

  ET_CHECK_MSG(
      draft_module_->set_outputs(method, outputs) == Error::Ok,
      "[DFlash] draft set_outputs failed for %s",
      method.c_str());

  stage_us_[kDraftPrep] += time_in_us() - t_prep;
  const int64_t t_exec = time_in_us();
  const long start_ms = time_in_ms();
  auto res = draft_module_->execute(method, inputs);
  const double elapsed = static_cast<double>(time_in_ms() - start_ms);
  stage_us_[kDraftExec] += time_in_us() - t_exec;
  const int64_t t_post = time_in_us();
  ET_CHECK_MSG(res.ok(), "[DFlash] draft execute failed for %s", method.c_str());

  if (out_hidden != nullptr) {
    draft_time_ms_ += elapsed;
    ++draft_calls_;
  } else {
    draft_prefill_time_ms_ += elapsed;
    ++draft_prefill_calls_;
  }

  // Commit the new context rows. Nothing else enters Cache B, ever.
  if (n_new > 0) {
    draft_kv_manager_->update_cache(ar, n_past, n_new, {});
    draft_ctx_len_ += n_new;
  }

  if (out_hidden != nullptr) {
    out_hidden->assign(static_cast<size_t>(B) * H, 0.0f);
    read_scalars(
        hidden_buf,
        h_meta.scalar_type(),
        0,
        static_cast<size_t>(B) * H,
        out_hidden->data());
  }
  // The block's logits are no longer produced in-graph (this build's draft emits
  // only hidden + K/V). The caller runs lm_head.pte over the returned hidden via
  // decode_lm_head, which also handles the shifted/aligned row offset.
  stage_us_[kDraftPost] += time_in_us() - t_post;
}

// ---------------------------------------------------------------------------
// Prompt seeding — Cache A -> draft graph -> Cache B, one chunk at a time.
// ---------------------------------------------------------------------------
void DFlashTokenGenerator::flush_stage() {
  if (stage_count_ == 0) {
    return;
  }
  // Block half is unused here; feed mask tokens so the graph sees valid ids.
  const std::vector<uint64_t> block(
      static_cast<size_t>(dflash_meta_.block_size),
      static_cast<uint64_t>(dflash_meta_.mask_token_id));
  run_draft(
      "prefill_forward",
      *draft_prefill_meta_,
      dflash_meta_.prefill_ar_len,
      stage_count_,
      stage_pos_base_,
      block,
      stage_pos_base_ + stage_count_,
      /*out_hidden=*/nullptr);
  stage_count_ = 0;
}

void DFlashTokenGenerator::stage_prompt_hidden(
    const std::vector<TensorStructRaw>& extra_outputs,
    int32_t n_valid,
    int64_t pos_base) {
  const int L = dflash_meta_.num_ctx_layers;
  const int H = dflash_meta_.hidden_dim;
  const size_t LH = static_cast<size_t>(L) * H;
  if (static_cast<int>(extra_outputs.size()) < L || n_valid <= 0) {
    return;
  }
  for (int j = 0; j < n_valid; ++j) {
    // The graph was calibrated without the sink row, so it must not be sent.
    // Nothing else needs adjusting: stage_pos_base_ picks up pos_base + j, which
    // is still the row's true position, and draft_ctx_len_ counts cache slots
    // rather than positions, so it simply ends up one shorter.
    if (dflash_meta_.drop_sink && pos_base + j == 0) {
      continue;
    }
    if (stage_count_ == 0) {
      stage_pos_base_ = pos_base + j;
    }
    float* dst = stage_buf_.data() + static_cast<size_t>(stage_count_) * LH;
    const int64_t t_cp = time_in_us();
    for (int l = 0; l < L; ++l) {
      read_scalars(
          extra_outputs[l].data,
          extra_outputs[l].dtype,
          static_cast<size_t>(j) * H,
          H,
          dst + static_cast<size_t>(l) * H);
    }
    stage_us_[kStageCopy] += time_in_us() - t_cp;
    if (++stage_count_ == dflash_meta_.prefill_ar_len) {
      flush_stage();
    }
  }
}

void DFlashTokenGenerator::finish_prompt_seeding() {
  flush_stage();
  ET_LOG(
      Info,
      "[DFlash] seeded draft cache with %d prompt rows in %llu prefill calls "
      "(%.1f ms)",
      draft_ctx_len_,
      static_cast<unsigned long long>(draft_prefill_calls_),
      draft_prefill_time_ms_);
}

// ---------------------------------------------------------------------------
// Target verify a block (chain / causal), read logits argmax + L hidden.
// ---------------------------------------------------------------------------
void DFlashTokenGenerator::target_verify_tree(
    int64_t cur_pos,
    uint64_t root_token) {
  const int64_t t_prep = time_in_us();
  const int ar = dflash_meta_.target_ar_len;
  const int n = 1 + static_cast<int>(tree_.size());
  ET_CHECK_MSG(n <= ar, "[DFlash] tree has %d nodes but ar_len is %d", n, ar);

  std::vector<uint64_t> packed_tokens(static_cast<size_t>(n));
  packed_tokens[0] = root_token;
  for (size_t i = 0; i < tree_.size(); ++i) {
    packed_tokens[i + 1] = tree_[i].token;
  }

  // Headless decoder input[0] is embeds (u16), not token ids. Run emb.pte over
  // the packed block (padded rows read as token 0, exactly as the old raw-token
  // path fed the decoder a 0), then quantize into inputs_[0].
  run_embedding(packed_tokens.data(), n);
  // emb.pte's output is ALREADY uint16 in the decoder's own embeds encoding
  // (compile-time injected scale/zp), despite the program declaring it Float.
  // Lossless hand-off: copy the bytes. Quantizing here would reinterpret the
  // u16 payload as f32 and destroy it.
  std::memcpy(input_toks_.data, emb_out_buf_, input_toks_.size);

  // Siblings share a position: a node sits at cur_pos + its depth, so the tree's
  // branches all overlay the same stretch of the sequence. Padded slots repeat
  // the last real position rather than running off the RoPE table.
  const int32_t last_depth = tree_.empty() ? 0 : tree_.back().depth;
  for (int k = 0; k < ar; ++k) {
    const int32_t depth = (k == 0) ? 0 : (k < n ? tree_[k - 1].depth : last_depth);
    input_pos_.data[k] = static_cast<int32_t>(cur_pos) + depth;
  }
  // attention_map[k] = k's parent slot; init_attention_mask copies the parent's
  // mask row and lights up k itself, which is exactly "see your ancestors and
  // nobody else". Padded slots point at the root -- their outputs are dropped.
  std::vector<int32_t> attention_map(ar, 0);
  attention_map[0] = -1;
  for (int k = 1; k < n; ++k) {
    attention_map[k] = tree_[k - 1].parent;
  }
  kv_manager_->init_attention_mask(
      attention_mask_.data, attention_map, ar, static_cast<int32_t>(cur_pos));
  stage_us_[kVerifyPrep] += time_in_us() - t_prep;

  long start_ms = time_in_ms();
  // Headless: step returns hidden in output_tensors_[0] (== logits_.data); the L
  // captured layers land in target_hidden_bufs_. We ignore the return tensor and
  // project the hidden through lm_head.pte to get the verify logits.
  const int64_t t_exec = time_in_us();
  const long t_dec = time_in_ms();
  auto step_res = decoder_runner_->step(method_name_, inputs_);
  decoder_exec_ms_ += static_cast<double>(time_in_ms() - t_dec);
  stage_us_[kVerifyExec] += time_in_us() - t_exec;
  ET_CHECK_MSG(step_res.ok(), "[DFlash] target verify step failed");
  const int64_t t_head = time_in_us();
  run_lm_head(logits_.data);
  stage_us_[kVerifyHead] += time_in_us() - t_head;
  target_verify_time_ms_ += static_cast<double>(time_in_ms() - start_ms);
  ++target_verify_calls_;

  // The logits stay in lm_head_logits_buf_. The caller walks the tree one node
  // at a time and argmaxes only the rows it lands on -- typically ~4 of the 16,
  // because a round commits accept_len tokens and the rest of the window is
  // branches the target never reached.
}

// ---------------------------------------------------------------------------
// generate — block-diffusion main loop
// ---------------------------------------------------------------------------
Result<int64_t> DFlashTokenGenerator::generate(
    std::vector<uint64_t> tokens,
    int64_t start_pos,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    bool /*dump_logits*/,
    AttentionSinkRopeRunner* /*attention_sink_rope_runner*/) {
  ET_CHECK_MSG(!tokens.empty(), "[DFlash] empty tokens");
  // Everything the stage timers hold right now was spent seeding the draft from
  // the prompt; the decode phase is whatever accumulates after this line.
  prefill_stage_us_ = stage_us_;
  const int B = dflash_meta_.block_size;
  const int H = dflash_meta_.hidden_dim;
  const int L = dflash_meta_.num_ctx_layers;
  const size_t LH = static_cast<size_t>(L) * H;
  const uint64_t mask_tok = static_cast<uint64_t>(dflash_meta_.mask_token_id);

  kv_manager_->rearrange_cache(dflash_meta_.target_ar_len);
  ET_CHECK_MSG(
      decoder_runner_->set_outputs(method_name_, output_tensors_) == Error::Ok,
      "[DFlash] set_outputs failed");

  int64_t cur_pos = start_pos; // position of the block's slot 0
  uint64_t last_committed = tokens.back();

  // The repeat gate learns from the prompt too -- a code prompt hands it the
  // indentation and separator tokens that legitimately double before the first
  // round runs, which is exactly where a blanket repeat penalty would misfire.
  for (auto& seen : repeat_seen_) {
    seen.clear();
  }
  committed_tail_.clear();
  for (uint64_t t : tokens) {
    note_committed(t);
  }

  // Rows staged for the next draft call: the hidden of the tokens committed by
  // the previous round. Empty on the first round — the prompt already seeded
  // Cache B, so that call appends nothing.
  int32_t pending_rows = 0;
  int64_t pending_pos_base = 0;

  std::vector<float> block_hidden;

  // See Metadata::shifted_decode. Shifted draws a prediction from every block row
  // and hands the target one more token to check; aligned throws row 0 away.
  const bool shifted = dflash_meta_.shifted_decode;
  const int n_draft = shifted ? B : B - 1;
  const int hidden_row0 = shifted ? 0 : H; // first hidden row that predicts
  ET_CHECK_MSG(
      draft_decode_ar_ >= n_draft + 1,
      "[DFlash] kv_forward appends %d rows/call but a round commits up to %d",
      draft_decode_ar_,
      n_draft + 1);

  // The chain is the k=1 tree: no siblings to push, so the heap can only go
  // deeper and lays down n_draft nodes in a single path. Running both through
  // build_tree keeps one code path instead of two that must be kept in step.
  const int32_t budget = dflash_meta_.tree_budget > 0
      ? std::min(dflash_meta_.tree_budget, dflash_meta_.target_ar_len - 1)
      : n_draft;
  draft_topk_ = dflash_meta_.tree_budget > 0 ? std::min(budget, 64) : 1;
  ET_CHECK_MSG(
      dflash_meta_.tree_budget <= 0 || dflash_meta_.logit_out_scale > 0.0f,
      "[DFlash] tree needs --dflash_logit_out_scale (lm_head output encoding)");
  ET_LOG(
      Info,
      "[DFlash] draft %s: budget=%d topk=%d depth_limit=%d ar=%d logit_scale=%g "
      "repeat_calib=%d",
      dflash_meta_.tree_budget > 0 ? "tree" : "chain",
      budget,
      draft_topk_,
      n_draft,
      dflash_meta_.target_ar_len,
      dflash_meta_.logit_out_scale,
      static_cast<int>(dflash_meta_.repeat_calib && dflash_meta_.tree_budget > 0));

  repeat_probe_ = getenv("DFLASH_REPEAT_PROBE") != nullptr;
  numeric_cache_.clear();

  const int64_t t_decode = time_in_us();
  while (cur_pos < static_cast<int64_t>(seq_len) - 1) {
    // --- DRAFT: append last round's accepted hidden, then denoise the block ---
    // The draft always sees B slots: the confirmed token, then masks.
    std::vector<uint64_t> noise(static_cast<size_t>(B), mask_tok);
    noise[0] = last_committed;
    run_draft(
        "kv_forward",
        *draft_kv_meta_,
        draft_decode_ar_,
        pending_rows,
        pending_pos_base,
        noise,
        cur_pos,
        &block_hidden);

    // Project the drafted hidden through lm_head.pte. hidden_row0 selects the
    // first predicting row (0 shifted, H aligned) and n_draft the count.
    draft_topk(block_hidden.data() + hidden_row0, n_draft, draft_topk_);

    const int64_t t_tree = time_in_us();
    build_tree(budget, n_draft);
    stage_us_[kTreeBuild] += time_in_us() - t_tree;
    total_drafted_ += tree_.size();
    total_tree_nodes_ += tree_.size();
    total_tree_depth_ += tree_.empty() ? 0 : tree_.back().depth;

    // --- VERIFY: one target forward over the whole tree ---
    target_verify_tree(cur_pos, last_committed);

    // Accept the longest prefix of the target's own continuation that the tree
    // happens to contain: read the target's argmax at a node, step to that child
    // if it exists, stop when it does not. Lazy on purpose -- only the nodes on
    // this path get argmaxed, not all ar of them.
    const int64_t t_acc = time_in_us();
    const size_t V = static_cast<size_t>(lm_head_vocab_size_);
    const uint16_t* lg = reinterpret_cast<const uint16_t*>(lm_head_logits_buf_);
    std::vector<int32_t> path{0};
    uint64_t bonus = argmax_u16_row(lg, V);
    for (int32_t child = tree_child(0, bonus); child >= 0;
         child = tree_child(path.back(), bonus)) {
      path.push_back(child);
      bonus = argmax_u16_row(lg + static_cast<size_t>(child) * V, V);
    }
    const int accepted = static_cast<int>(path.size()) - 1;
    stage_us_[kVerifyPick] += time_in_us() - t_acc;

    // --- COMMIT the accepted slots' target KV ---
    // A tree path lands on scattered columns, not 0..accepted, so update_cache
    // gathers them -- the same job the reference calls compact_dynamic_cache.
    const int64_t t_commit = time_in_us();
    std::vector<bool> selected(static_cast<size_t>(dflash_meta_.target_ar_len), false);
    for (int32_t slot : path) {
      selected[static_cast<size_t>(slot)] = true;
    }
    kv_manager_->update_cache(
        dflash_meta_.target_ar_len,
        static_cast<int32_t>(cur_pos),
        accepted + 1,
        selected);
    kv_manager_->update_attention_mask(
        attention_mask_.data,
        dflash_meta_.target_ar_len,
        static_cast<int32_t>(cur_pos),
        accepted + 1);
    stage_us_[kCommit] += time_in_us() - t_commit;

    // --- stage the accepted tokens' hidden for the next draft call ---
    // Same scatter as the KV: read the path's slots, not the leading rows. What
    // the draft receives is still a chain at consecutive positions from cur_pos,
    // so nothing downstream of here knows a tree was involved.
    const int64_t t_cp = time_in_us();
    pending_rows = accepted + 1;
    pending_pos_base = cur_pos;
    for (int k = 0; k < pending_rows; ++k) {
      const size_t src_row = static_cast<size_t>(path[k]);
      float* dst = stage_buf_.data() + static_cast<size_t>(k) * LH;
      for (int l = 0; l < L; ++l) {
        read_scalars(
            target_hidden_bufs_[l].data(),
            target_hidden_dtypes_[l],
            src_row * H,
            H,
            dst + static_cast<size_t>(l) * H);
      }
    }

    stage_us_[kStageCopy] += time_in_us() - t_cp;

    // The gate must see committed tokens in order, and must see them before the
    // next tree is built. The emit loop below can exit early on EOS, so this
    // cannot ride along with it.
    std::vector<uint64_t> committed_now;
    for (int k = 1; k <= accepted; ++k) {
      committed_now.push_back(tree_[static_cast<size_t>(path[k]) - 1].token);
    }
    committed_now.push_back(bonus);
    if (repeat_probe_) {
      probe_round(committed_now, n_draft);
    }
    for (uint64_t t : committed_now) {
      note_committed(t);
    }

    total_accepted_ += static_cast<uint64_t>(accepted);
    ET_LOG(
        Debug,
        "[DFlash] cur_pos=%lld accepted=%d/%d bonus=%llu draft_ctx=%d",
        static_cast<long long>(cur_pos),
        accepted,
        n_draft,
        static_cast<unsigned long long>(bonus),
        draft_ctx_len_);

    // --- emit accepted draft tokens + bonus ---
    // An EOS among the accepted tokens ends generation at that token, but must not
    // skip the summary below, so it leaves through the same exit as everything else.
    const int64_t t_emit = time_in_us();
    uint64_t prev = last_committed;
    bool hit_eos = false;
    for (int k = 1; k <= accepted; ++k) {
      const uint64_t tok = tree_[static_cast<size_t>(path[k]) - 1].token;
      token_callback(ET_UNWRAP_TOKENIZER(tokenizer_->decode(prev, tok)));
      if (eos_ids_->count(tok) > 0) {
        cur_pos += k;
        hit_eos = true;
        break;
      }
      prev = tok;
    }
    if (hit_eos) {
      stage_us_[kEmit] += time_in_us() - t_emit;
      break;
    }
    token_callback(ET_UNWRAP_TOKENIZER(tokenizer_->decode(prev, bonus)));
    stage_us_[kEmit] += time_in_us() - t_emit;

    cur_pos += accepted + 1;
    last_committed = bonus;
    if (eos_ids_->count(bonus) > 0) {
      break;
    }
    // kv_forward can only address max_context_len - block_size committed rows.
    if (draft_ctx_len_ + pending_rows > dflash_meta_.max_context_len - B) {
      ET_LOG(Info, "[DFlash] draft context full at %d rows", draft_ctx_len_);
      break;
    }
  }
  decode_wall_us_ = time_in_us() - t_decode;

  if (target_verify_calls_ > 0) {
    // accept_len is the metric that matters: tokens committed per target forward.
    // 1.0 means the draft bought nothing. accept_rate (accepted/drafted) makes the
    // same run look bad simply because block_size is large.
    ET_LOG(
        Info,
        "[DFlash] done: drafted=%llu accepted=%llu accept_rate=%.3f "
        "accept_len=%.2f tok/target_call | phase: draft=%.1f (%llu) "
        "verify=%.1f (%llu) draft_lm_head=%.1f | per-pte ms: emb=%.1f (%llu) "
        "decoder=%.1f draft=%.1f lm_head=%.1f (%llu) host_argmax=%.1f",
        static_cast<unsigned long long>(total_drafted_),
        static_cast<unsigned long long>(total_accepted_),
        total_drafted_ ? static_cast<double>(total_accepted_) /
                static_cast<double>(total_drafted_)
                       : 0.0,
        static_cast<double>(total_accepted_ + target_verify_calls_) /
            static_cast<double>(target_verify_calls_),
        draft_time_ms_,
        static_cast<unsigned long long>(draft_calls_),
        target_verify_time_ms_,
        static_cast<unsigned long long>(target_verify_calls_),
        lm_head_time_ms_,
        // per-pte breakdown
        emb_exec_ms_,
        static_cast<unsigned long long>(emb_calls_),
        decoder_exec_ms_,
        draft_time_ms_,
        lm_head_exec_ms_,
        static_cast<unsigned long long>(lm_head_calls_),
        sample_ms_);

    // Tree shape. Depth well below the budget means breadth is winning the
    // allocation, which is the whole point; depth pinned at the block horizon
    // means the budget is too small to branch and the tree is just a chain.
    ET_LOG(
        Info,
        "[DFlash] tree: budget=%d topk=%d nodes/round=%.2f max_depth/round=%.2f"
        " | repeat: scored=%llu charged=%llu (%.2f%%) lag1=%llu gated=%llu",
        budget,
        draft_topk_,
        static_cast<double>(total_tree_nodes_) /
            static_cast<double>(target_verify_calls_),
        static_cast<double>(total_tree_depth_) /
            static_cast<double>(target_verify_calls_),
        static_cast<unsigned long long>(rep_scored_),
        static_cast<unsigned long long>(rep_charged_),
        rep_scored_ ? 100.0 * static_cast<double>(rep_charged_) /
                static_cast<double>(rep_scored_)
                    : 0.0,
        static_cast<unsigned long long>(rep_lag1_),
        static_cast<unsigned long long>(rep_gated_));

    if (repeat_probe_) {
      for (size_t g = 0; g < 2; ++g) {
        for (size_t nm = 0; nm < 2; ++nm) {
          for (int32_t lag = 1; lag <= kRepLag; ++lag) {
            const size_t b = 1 + (g * 2 + nm) * kRepLag + (lag - 1);
            if (probe_n_[b] == 0) {
              continue;
            }
            ET_LOG(
                Info,
                "[DFlash][probe] gated=%zu numeric=%zu lag=%d n=%llu sumq=%.2f "
                "hit=%.0f R=%.3f",
                g,
                nm,
                lag,
                static_cast<unsigned long long>(probe_n_[b]),
                probe_q_[b],
                probe_hit_[b],
                probe_hit_[b] / std::max(probe_q_[b], 1e-9));
          }
        }
      }
      ET_LOG(
          Info,
          "[DFlash][probe] norepeat n=%llu sumq=%.2f hit=%.0f R=%.3f",
          static_cast<unsigned long long>(probe_n_[0]),
          probe_q_[0],
          probe_hit_[0],
          probe_hit_[0] / std::max(probe_q_[0], 1e-9));
    }

    // Per-round stage breakdown. One line per stage so it parses with a regex,
    // and an `other` row carrying whatever the stages failed to claim -- without
    // it a breakdown silently reads as complete when it is not.
    const double rounds = static_cast<double>(target_verify_calls_);
    const double wall_ms = static_cast<double>(decode_wall_us_) / 1000.0;
    int64_t claimed_us = 0;
    for (int i = 0; i < kNumStages; ++i) {
      claimed_us += stage_us_[i] - prefill_stage_us_[i];
    }
    ET_LOG(
        Info,
        "[DFlash][stage] phase=decode rounds=%.0f wall_ms=%.1f round_ms=%.3f",
        rounds,
        wall_ms,
        wall_ms / rounds);
    for (int i = 0; i < kNumStages; ++i) {
      const double ms =
          static_cast<double>(stage_us_[i] - prefill_stage_us_[i]) / 1000.0;
      ET_LOG(
          Info,
          "[DFlash][stage] decode %-11s total_ms=%9.1f round_ms=%7.3f pct=%5.1f",
          kStageNames[i],
          ms,
          ms / rounds,
          100.0 * ms / wall_ms);
    }
    const double other_ms =
        (static_cast<double>(decode_wall_us_ - claimed_us)) / 1000.0;
    ET_LOG(
        Info,
        "[DFlash][stage] decode %-11s total_ms=%9.1f round_ms=%7.3f pct=%5.1f",
        "other",
        other_ms,
        other_ms / rounds,
        100.0 * other_ms / wall_ms);
    for (int i = 0; i < kNumStages; ++i) {
      if (prefill_stage_us_[i] == 0) {
        continue;
      }
      ET_LOG(
          Info,
          "[DFlash][stage] prefill %-11s total_ms=%9.1f",
          kStageNames[i],
          static_cast<double>(prefill_stage_us_[i]) / 1000.0);
    }
  }
  return static_cast<int64_t>(cur_pos - start_pos);
}

} // namespace example
