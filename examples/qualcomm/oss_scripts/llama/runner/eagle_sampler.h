/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace example {

/**
 * @class EagleSampler
 * @brief Sample a draft token from EAGLE head logits and map back to target
 * vocabulary id via the d2t (draft-to-target) lookup.
 *
 * Two dtypes are supported (EAGLE head IO is fp16 by default in our build):
 *   - kFp16:    raw __fp16 logits, host argmax over draft_vocab_size entries
   *   - kFp32:    raw float logits
   *   - kQuant16: uint16 quantized logits, dequantized via head pte's
 *               get_logits_scale / get_logits_zero_point constant_methods
 *               before argmax. (TODO; not used in Phase 2 fp16 build.)
 */
class EagleSampler {
 public:
  enum class Dtype { kFp16, kFp32, kQuant16 };

  EagleSampler(
      Dtype dtype,
      int32_t draft_vocab_size,
      const std::vector<int64_t>& d2t,
      float logits_scale = 1.0f,
      int32_t logits_zero_point = 0);

  // 32000-way argmax over `logits_buf` (length = draft_vocab_size * sizeof(dtype)).
  // Returns the draft vocabulary id (in [0, draft_vocab_size)).
  uint64_t argmax_draft(const std::byte* logits_buf) const;

  // Map a draft vocabulary id back to a target vocabulary id:
  //   target_id = draft_id + d2t[draft_id]
  inline uint64_t draft_to_target(uint64_t draft_id) const {
    if (draft_id >= d2t_.size()) {
      return draft_id;  // identity if no mapping (shouldn't happen)
    }
    return static_cast<uint64_t>(
        static_cast<int64_t>(draft_id) + d2t_[draft_id]);
  }

  // Convenience: argmax + map in one call.
  uint64_t sample_target_from_logits(const std::byte* logits_buf) const {
    return draft_to_target(argmax_draft(logits_buf));
  }

  int32_t draft_vocab_size() const { return draft_vocab_size_; }

 private:
  Dtype dtype_;
  int32_t draft_vocab_size_;
  std::vector<int64_t> d2t_;       // [draft_vocab_size]
  float logits_scale_;
  int32_t logits_zero_point_;
};

} // namespace example
