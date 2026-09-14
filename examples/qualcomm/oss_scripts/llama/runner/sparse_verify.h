/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace example {

/**
 * Gathers the drafter's candidate rows out of a row-major int8 lm_head table
 * into the buffer the sparse verify head reads.
 *
 * The full verify head streams all 389 MB of lm_head weights every round
 * (10.18 ms measured, 40 GB/s -- it is bandwidth-bound, not compute-bound), but
 * verification only has to rank the tokens the drafter proposed. Reading just
 * those D*K rows turns that into 2.1 ms.
 *
 * The gather itself is free: the candidate ids are known once draft_topk fills
 * draft_ids_, which is a whole tree_build plus verify_exec (46 ms of NPU time)
 * before the head runs. So it goes on worker threads and joins just before the
 * head is dispatched.
 *
 * The table must be row-major, which lm_head_qnn.pte is NOT -- QNN interleaves
 * it for the HTP dataflow (a single row is spread over 163840 bytes, 40 pages,
 * and gathering in that layout measures 12x slower and touches 13x the pages).
 * sparse/deinterleave_lm_head.py produces the row-major copy.
 */
class SparseGather {
 public:
  ~SparseGather();

  /**
   * @param hot_rows Rows pinned resident with mlock. The rest are ordinary
   *   file-backed pages: warm they cost 0.35 ms/round, fully cold 33 ms at 16
   *   threads -- both inside the 46 ms window, so residency is a memory knob
   *   rather than a correctness one.
   */
  bool open(
      const char* table_path,
      const char* scale_path,
      int32_t vocab,
      int32_t dim,
      int32_t hot_rows,
      int32_t threads);

  /** Kicks off the gather and returns. `ids` must outlive the wait(). */
  void start(const uint64_t* ids, int32_t n, uint16_t* dst_w);

  /** The encoding the graph was built with. Both sides derive it the same way,
   *  so they agree by construction rather than by a shared file. */
  float w_scale() const {
    return w_scale_;
  }
  void wait();

  bool ready() const {
    return table_ != nullptr;
  }
  int32_t vocab() const {
    return vocab_;
  }

 private:
  void worker(int32_t slot);
  void run_slice(const uint64_t* ids, int32_t lo, int32_t hi, uint16_t* dst_w);

  const int8_t* table_ = nullptr;
  const float* scales_ = nullptr;
  float w_scale_ = 0.0f;
  float inv_w_scale_ = 0.0f;
  size_t table_bytes_ = 0;
  size_t scale_bytes_ = 0;
  int32_t vocab_ = 0;
  int32_t dim_ = 0;
  int32_t nthreads_ = 0;

  std::vector<std::thread> pool_;
  std::mutex mu_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  const uint64_t* ids_ = nullptr;
  uint16_t* dst_w_ = nullptr;
  int32_t n_ = 0;
  uint64_t epoch_ = 0;
  int32_t outstanding_ = 0;
  bool stop_ = false;
};

} // namespace example
