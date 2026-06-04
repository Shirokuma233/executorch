/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/eagle_sampler.h>

#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <limits>

namespace example {

EagleSampler::EagleSampler(
    Dtype dtype,
    int32_t draft_vocab_size,
    const std::vector<int64_t>& d2t,
    float logits_scale,
    int32_t logits_zero_point)
    : dtype_(dtype),
      draft_vocab_size_(draft_vocab_size),
      d2t_(d2t),
      logits_scale_(logits_scale),
      logits_zero_point_(logits_zero_point) {
  if (static_cast<int32_t>(d2t_.size()) != draft_vocab_size_) {
    ET_LOG(
        Error,
        "[EagleSampler] d2t size %zu != draft_vocab_size %d. "
        "draft_to_target mapping will fall back to identity for OOR ids.",
        d2t_.size(),
        draft_vocab_size_);
  }
}

uint64_t EagleSampler::argmax_draft(const std::byte* logits_buf) const {
  if (dtype_ == Dtype::kFp16) {
    // __fp16 may not be portable across all compilers; we use uint16 + manual
    // bitcast guarded by sizeof check. On clang/gcc on aarch64/x86_64 with the
    // -mfp16-format=ieee flag, __fp16 is the IEEE binary16. Our QNN runtime
    // is built that way.
    static_assert(sizeof(__fp16) == 2, "expected __fp16 to be 2 bytes");
    const __fp16* p = reinterpret_cast<const __fp16*>(logits_buf);
    int best = 0;
    __fp16 best_val = p[0];
    for (int i = 1; i < draft_vocab_size_; ++i) {
      if (p[i] > best_val) {
        best_val = p[i];
        best = i;
      }
    }
    return static_cast<uint64_t>(best);
  }

  if (dtype_ == Dtype::kFp32) {
    const float* p = reinterpret_cast<const float*>(logits_buf);
    int best = 0;
    float best_val = p[0];
    for (int i = 1; i < draft_vocab_size_; ++i) {
      if (p[i] > best_val) {
        best_val = p[i];
        best = i;
      }
    }
    return static_cast<uint64_t>(best);
  }

  // kQuant16: dequantize then argmax. Strictly we don't need to dequantize for
  // argmax because (x - zp) * scale is monotonic in x when scale > 0 — so the
  // argmax in uint16 space equals the argmax in real space. We just argmax
  // over uint16 directly.
  const uint16_t* p = reinterpret_cast<const uint16_t*>(logits_buf);
  int best = 0;
  uint16_t best_val = p[0];
  for (int i = 1; i < draft_vocab_size_; ++i) {
    if (p[i] > best_val) {
      best_val = p[i];
      best = i;
    }
  }
  return static_cast<uint64_t>(best);
}

} // namespace example
