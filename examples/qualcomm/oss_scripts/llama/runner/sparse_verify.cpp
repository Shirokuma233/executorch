/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/sparse_verify.h>

#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <cmath>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace example {

namespace {

const void* map_file(const char* path, size_t want, size_t* mapped) {
  const int fd = ::open(path, O_RDONLY);
  if (fd < 0) {
    ET_LOG(Error, "[DFlash] sparse: cannot open %s", path);
    return nullptr;
  }
  struct stat st {};
  if (::fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) < want) {
    ET_LOG(
        Error,
        "[DFlash] sparse: %s is %lld bytes, need %zu",
        path,
        static_cast<long long>(st.st_size),
        want);
    ::close(fd);
    return nullptr;
  }
  void* p = ::mmap(nullptr, want, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd); // the mapping keeps its own reference
  if (p == MAP_FAILED) {
    ET_LOG(Error, "[DFlash] sparse: mmap %s failed", path);
    return nullptr;
  }
  *mapped = want;
  return p;
}

} // namespace

SparseGather::~SparseGather() {
  if (!pool_.empty()) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_start_.notify_all();
    for (auto& t : pool_) {
      t.join();
    }
  }
  if (table_ != nullptr) {
    ::munmap(const_cast<int8_t*>(table_), table_bytes_);
  }
  if (scales_ != nullptr) {
    ::munmap(const_cast<float*>(scales_), scale_bytes_);
  }
}

bool SparseGather::open(
    const char* table_path,
    const char* scale_path,
    int32_t vocab,
    int32_t dim,
    int32_t hot_rows,
    int32_t threads) {
  vocab_ = vocab;
  dim_ = dim;
  const size_t want_table = static_cast<size_t>(vocab) * dim;
  const size_t want_scale = static_cast<size_t>(vocab) * sizeof(float);
  table_ = static_cast<const int8_t*>(
      map_file(table_path, want_table, &table_bytes_));
  scales_ =
      static_cast<const float*>(map_file(scale_path, want_scale, &scale_bytes_));
  if (table_ == nullptr || scales_ == nullptr) {
    return false;
  }

  // Pin the low-id prefix. Token ids are roughly frequency-ordered in a BPE
  // vocab -- 90% of real candidate references land in the first 32k -- so the
  // prefix is a decent stand-in for a frequency-ordered layout without needing
  // an id->row permutation.
  const size_t hot = static_cast<size_t>(
                         hot_rows > 0 && hot_rows < vocab ? hot_rows : 0) *
      dim;
  if (hot > 0 && ::mlock(table_, hot) != 0) {
    ET_LOG(Info, "[DFlash] sparse: mlock %zu bytes refused, continuing", hot);
  }

  // Every row's int8 uses the full +-127 -- that is how a per-channel symmetric
  // scale gets chosen -- so the dequantized global max is exactly
  // 127 * max(scale). The export derives the graph's encoding from the same
  // expression, so the two agree by construction instead of via a shared file.
  float mx = 0.0f;
  for (int32_t v = 0; v < vocab; ++v) {
    mx = std::max(mx, scales_[v]);
  }
  w_scale_ = mx * 127.0f / 32767.0f;
  inv_w_scale_ = 1.0f / w_scale_;

  outstanding_ = 0;
  nthreads_ = std::max(threads, 1);
  pool_.reserve(nthreads_);
  for (int32_t i = 0; i < nthreads_; ++i) {
    pool_.emplace_back(&SparseGather::worker, this, i);
  }
  ET_LOG(
      Info,
      "[DFlash] sparse gather: vocab=%d dim=%d hot=%d rows threads=%zu",
      vocab,
      dim,
      hot_rows,
      static_cast<size_t>(nthreads_));
  return true;
}

void SparseGather::run_slice(
    const uint64_t* ids,
    int32_t lo,
    int32_t hi,
    uint16_t* dst_w) {
  const size_t dim = static_cast<size_t>(dim_);
  for (int32_t i = lo; i < hi; ++i) {
    const uint64_t id = ids[i];
    const int8_t* src = table_ + static_cast<size_t>(id) * dim;
    uint16_t* out = dst_w + static_cast<size_t>(i) * dim;
    // The per-channel scale folds in here rather than in the graph, which keeps
    // the graph's only quantized intermediate equal to the logits -- an unscaled
    // dot product would need calibration data matching the drafter's candidate
    // distribution exactly. Requantizing to uint16 on the way out halves what
    // this loop writes; measured, that write competes with the NPU for the bus.
    const float sc = scales_[id] * inv_w_scale_;
    for (size_t c = 0; c < dim; ++c) {
      out[c] = static_cast<uint16_t>(
          lrintf(static_cast<float>(src[c]) * sc) + 32768);
    }
  }
}

void SparseGather::worker(int32_t slot) {
  uint64_t seen = 0;
  while (true) {
    int32_t n = 0;
    const uint64_t* ids = nullptr;
    uint16_t* dst_w = nullptr;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_start_.wait(lk, [&] { return stop_ || epoch_ != seen; });
      if (stop_) {
        return;
      }
      seen = epoch_;
      n = n_;
      ids = ids_;
      dst_w = dst_w_;
    }
    run_slice(ids, n * slot / nthreads_, n * (slot + 1) / nthreads_, dst_w);
    {
      std::lock_guard<std::mutex> lk(mu_);
      --outstanding_;
    }
    cv_done_.notify_one();
  }
}

void SparseGather::start(const uint64_t* ids, int32_t n, uint16_t* dst_w) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    ids_ = ids;
    n_ = n;
    dst_w_ = dst_w;
    outstanding_ = nthreads_;
    ++epoch_;
  }
  cv_start_.notify_all();
}

void SparseGather::wait() {
  std::unique_lock<std::mutex> lk(mu_);
  cv_done_.wait(lk, [&] { return outstanding_ == 0; });
}

} // namespace example
