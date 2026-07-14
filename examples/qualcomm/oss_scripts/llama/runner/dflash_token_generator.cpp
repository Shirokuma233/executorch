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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
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

namespace {

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
    std::unique_ptr<MethodMeta> draft_prefill_meta)
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
      dflash_meta_(metadata) {
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
      "[DFlash] lm_head: %s (vocab %zu)",
      draft_has_lm_head_ ? "in-graph (HTP)" : "host scan of the embedding table",
      draft_vocab_size_);

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
// Embedding lookup (host gather)
// ---------------------------------------------------------------------------
void DFlashTokenGenerator::lookup_embedding(uint64_t token_id, uint16_t* dst)
    const {
  const int H = embed_hidden_size_;
  if (embed_table_.empty() ||
      static_cast<int64_t>(token_id) >= embed_vocab_size_) {
    memset(dst, 0, static_cast<size_t>(H) * sizeof(uint16_t));
    return;
  }
  const uint16_t* row = embed_table_.data() + static_cast<size_t>(token_id) * H;
  memcpy(dst, row, static_cast<size_t>(H) * sizeof(uint16_t));
}

// ---------------------------------------------------------------------------
// Host lm_head + argmax over the full target vocab
// ---------------------------------------------------------------------------
// V*H*count is ~5.8 GMAC per block (151936 x 2560 x 15). Vocab is the outer loop
// so the 742 MiB tied embed/lm_head table streams from DDR exactly once per block
// instead of once per drafted row, and each fp16 row is widened once rather than
// `count` times.
//
// Threads own contiguous, increasing vocab slices and the reduction walks them in
// order under a strict `>`, so ties still resolve to the lowest token id — the
// same answer the single-threaded loop gave.
void DFlashTokenGenerator::lm_head_argmax(
    const float* hidden,
    int32_t count,
    std::vector<uint64_t>* out_tokens) const {
  const int H = dflash_meta_.hidden_dim;
  const int V = embed_vocab_size_;
  const std::vector<uint16_t>& W =
      lm_head_table_.empty() ? embed_table_ : lm_head_table_;
  const size_t n = static_cast<size_t>(count);
  out_tokens->assign(n, 0);

  unsigned hw = std::thread::hardware_concurrency();
  const size_t nthreads =
      std::max<size_t>(1, std::min<size_t>(hw ? hw : 1, kLmHeadMaxThreads));

  std::vector<float> best(nthreads * n, -std::numeric_limits<float>::infinity());
  std::vector<uint64_t> arg(nthreads * n, 0);

  auto slice = [&](size_t t) {
    const int v0 = static_cast<int>((static_cast<int64_t>(V) * t) / nthreads);
    const int v1 = static_cast<int>((static_cast<int64_t>(V) * (t + 1)) / nthreads);
    float* tbest = best.data() + t * n;
    uint64_t* targ = arg.data() + t * n;
    std::vector<float> wrow(static_cast<size_t>(H));
    for (int v = v0; v < v1; ++v) {
      const uint16_t* src = W.data() + static_cast<size_t>(v) * H;
      for (int i = 0; i < H; ++i) {
        wrow[i] = fp16_to_fp32(src[i]);
      }
      for (size_t s = 0; s < n; ++s) {
        const float* h = hidden + s * static_cast<size_t>(H);
        float dot = 0.0f;
        for (int i = 0; i < H; ++i) {
          dot += wrow[i] * h[i];
        }
        if (dot > tbest[s]) {
          tbest[s] = dot;
          targ[s] = static_cast<uint64_t>(v);
        }
      }
    }
  };

  std::vector<std::thread> pool;
  pool.reserve(nthreads - 1);
  for (size_t t = 1; t < nthreads; ++t) {
    pool.emplace_back(slice, t);
  }
  slice(0);
  for (auto& th : pool) {
    th.join();
  }

  std::vector<float> gbest(n, -std::numeric_limits<float>::infinity());
  for (size_t t = 0; t < nthreads; ++t) {
    for (size_t s = 0; s < n; ++s) {
      if (best[t * n + s] > gbest[s]) {
        gbest[s] = best[t * n + s];
        (*out_tokens)[s] = arg[t * n + s];
      }
    }
  }
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
    std::vector<float>* out_hidden,
    std::vector<uint64_t>* out_tokens) {
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

  // noise_embedding: host gather of the target's embedding table.
  std::vector<uint16_t> emb(H);
  for (size_t k = 0; k < block_tokens.size() && k < static_cast<size_t>(B); ++k) {
    lookup_embedding(block_tokens[k], emb.data());
    for (int i = 0; i < H; ++i) {
      write_float(
          noise_buf, n_meta.scalar_type(), k * H + i, fp16_to_fp32(emb[i]));
    }
  }

  // atten_mask [B, Cc + ar + B]: every block query sees the committed cache, the
  // valid new-context rows, and the whole block (bidirectional denoising).
  for (int q = 0; q < B; ++q) {
    std::byte* row = mask_buf +
        static_cast<size_t>(q) * mask_w * dtype_size(m_meta.scalar_type());
    for (int col = 0; col < mask_w; ++col) {
      bool vis;
      if (col < Cc) {
        vis = col < n_past;
      } else if (col < Cc + ar) {
        vis = (col - Cc) < n_new;
      } else {
        vis = true;
      }
      write_mask(row, m_meta.scalar_type(), static_cast<size_t>(col), vis);
    }
  }

  // new_context + its RoPE positions. Padded rows stay zero and are masked out;
  // update_cache only commits the first n_new of them.
  for (int j = 0; j < n_new; ++j) {
    const float* src = stage_buf_.data() + static_cast<size_t>(j) * LH;
    for (size_t i = 0; i < LH; ++i) {
      write_float(
          ctx_buf, c_meta.scalar_type(), static_cast<size_t>(j) * LH + i, src[i]);
    }
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

  const long start_ms = time_in_ms();
  auto res = draft_module_->execute(method, inputs);
  const double elapsed = static_cast<double>(time_in_ms() - start_ms);
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
    for (int k = 0; k < B; ++k) {
      for (int i = 0; i < H; ++i) {
        (*out_hidden)[static_cast<size_t>(k) * H + i] = read_scalar(
            hidden_buf, h_meta.scalar_type(), static_cast<size_t>(k) * H + i);
      }
    }
  }

  // The graph did the 2.3 GMAC vocab projection; all that is left is picking the max
  // of each row. QNN's own argmax is off by +/-65536 on an axis this wide, so it
  // stays here.
  if (out_tokens != nullptr && draft_has_lm_head_) {
    const size_t V = draft_vocab_size_;
    auto logits = meta.output_tensor_meta(1 + 2 * Ld).get();
    // Rows, not B-1: the shifted convention predicts from every block row.
    const int rows = static_cast<int>(logits.sizes()[1]);
    out_tokens->assign(static_cast<size_t>(rows), 0);
    for (int k = 0; k < rows; ++k) {
      float best = -std::numeric_limits<float>::infinity();
      for (size_t v = 0; v < V; ++v) {
        const float x = read_scalar(
            draft_logits_buf_, logits.scalar_type(), static_cast<size_t>(k) * V + v);
        if (x > best) {
          best = x;
          (*out_tokens)[k] = static_cast<uint64_t>(v);
        }
      }
    }
  }
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
    if (stage_count_ == 0) {
      stage_pos_base_ = pos_base + j;
    }
    float* dst = stage_buf_.data() + static_cast<size_t>(stage_count_) * LH;
    for (int l = 0; l < L; ++l) {
      for (int h = 0; h < H; ++h) {
        dst[l * H + h] = read_scalar(
            extra_outputs[l].data,
            extra_outputs[l].dtype,
            static_cast<size_t>(j) * H + h);
      }
    }
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
void DFlashTokenGenerator::target_verify_block(
    const std::vector<uint64_t>& packed_tokens,
    int64_t cur_pos,
    std::vector<uint64_t>* target_sampled) {
  const int ar = dflash_meta_.target_ar_len;
  const int n = static_cast<int>(packed_tokens.size());

  for (int k = 0; k < ar; ++k) {
    int64_t tok = (k < n) ? static_cast<int64_t>(packed_tokens[k]) : 0;
    if (dflash_meta_.use_int64_token) {
      input_toks_.data[k] = tok;
    } else {
      reinterpret_cast<int32_t*>(input_toks_.data)[k] =
          static_cast<int32_t>(tok);
    }
  }
  for (int k = 0; k < ar; ++k) {
    input_pos_.data[k] = static_cast<int32_t>(cur_pos) + std::min(k, n - 1);
  }
  std::vector<int32_t> attention_map(ar);
  for (int k = 0; k < ar; ++k) {
    attention_map[k] = (k == 0) ? -1 : std::min(k - 1, n - 1);
  }
  kv_manager_->init_attention_mask(
      attention_mask_.data, attention_map, ar, static_cast<int32_t>(cur_pos));

  long start_ms = time_in_ms();
  auto logits_res = decoder_runner_->step(method_name_, inputs_);
  target_verify_time_ms_ += static_cast<double>(time_in_ms() - start_ms);
  ++target_verify_calls_;
  ET_CHECK_MSG(logits_res.ok(), "[DFlash] target verify step failed");
  Tensor& logits = logits_res.get();

  target_sampled->resize(static_cast<size_t>(ar));
  for (int k = 0; k < ar; ++k) {
    (*target_sampled)[k] =
        static_cast<uint64_t>(decoder_runner_->logits_to_token(logits, k));
  }
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

  // Rows staged for the next draft call: the hidden of the tokens committed by
  // the previous round. Empty on the first round — the prompt already seeded
  // Cache B, so that call appends nothing.
  int32_t pending_rows = 0;
  int64_t pending_pos_base = 0;

  std::vector<float> block_hidden;
  std::vector<uint64_t> drafted;
  std::vector<uint64_t> target_sampled;

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
        &block_hidden,
        &drafted);

    // With an in-graph lm_head the logits already came back from HTP; otherwise
    // scan the embedding table here.
    if (!draft_has_lm_head_) {
      const long lm_start_ms = time_in_ms();
      lm_head_argmax(block_hidden.data() + hidden_row0, n_draft, &drafted);
      lm_head_time_ms_ += static_cast<double>(time_in_ms() - lm_start_ms);
    }

    // What the target checks: the confirmed token followed by the draft's guesses.
    std::vector<uint64_t> verify(static_cast<size_t>(n_draft) + 1);
    verify[0] = last_committed;
    for (int k = 0; k < n_draft; ++k) {
      verify[k + 1] = drafted[k];
    }
    total_drafted_ += static_cast<uint64_t>(n_draft);

    // --- VERIFY: target runs the whole block causally ---
    target_verify_block(verify, cur_pos, &target_sampled);

    // Verify slot k predicts slot k+1, so target_sampled[k] is what the target
    // would have put where the draft put verify[k+1].
    int accepted = 0;
    for (int k = 0; k < n_draft; ++k) {
      if (verify[k + 1] == target_sampled[k]) {
        ++accepted;
      } else {
        break;
      }
    }
    uint64_t bonus = target_sampled[accepted];

    // --- COMMIT target KV for slots [0..accepted] ---
    std::vector<bool> selected(static_cast<size_t>(dflash_meta_.target_ar_len), false);
    for (int k = 0; k <= accepted; ++k) {
      selected[static_cast<size_t>(k)] = true;
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

    // --- stage the accepted tokens' hidden for the next draft call ---
    pending_rows = accepted + 1;
    pending_pos_base = cur_pos;
    for (int k = 0; k < pending_rows; ++k) {
      float* dst = stage_buf_.data() + static_cast<size_t>(k) * LH;
      for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h) {
          dst[l * H + h] = read_scalar(
              target_hidden_bufs_[l].data(),
              target_hidden_dtypes_[l],
              static_cast<size_t>(k) * H + h);
        }
      }
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
    uint64_t prev = last_committed;
    bool hit_eos = false;
    for (int k = 1; k <= accepted; ++k) {
      token_callback(ET_UNWRAP_TOKENIZER(tokenizer_->decode(prev, verify[k])));
      if (eos_ids_->count(verify[k]) > 0) {
        cur_pos += k;
        hit_eos = true;
        break;
      }
      prev = verify[k];
    }
    if (hit_eos) {
      break;
    }
    token_callback(ET_UNWRAP_TOKENIZER(tokenizer_->decode(prev, bonus)));

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

  if (target_verify_calls_ > 0) {
    // accept_len is the metric that matters: tokens committed per target forward.
    // 1.0 means the draft bought nothing. accept_rate (accepted/drafted) makes the
    // same run look bad simply because block_size is large.
    ET_LOG(
        Info,
        "[DFlash] done: drafted=%llu accepted=%llu accept_rate=%.3f "
        "accept_len=%.2f tok/target_call | "
        "draft_ms=%.1f (%llu calls) verify_ms=%.1f lm_head_ms=%.1f",
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
        lm_head_time_ms_);
  }
  return static_cast<int64_t>(cur_pos - start_pos);
}

} // namespace example
