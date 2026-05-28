/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// =============================================================================
// TreeBuilder — Phase 4 SCAFFOLD
// =============================================================================
// All three public methods are TODO. The header is fully fleshed out so that
// downstream callers can be implemented in parallel.
// =============================================================================

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/tree_attention.h>

#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <numeric>

namespace example {

TreeBuilder::TreeBuilder(std::vector<int> branching_per_depth, int max_size)
    : branching_(std::move(branching_per_depth)) {
  // total_nodes_ = product-prefix sum of branching_, capped at max_size.
  // For "1,2,2,2,2": layer0=1, layer1=2, layer2=4, layer3=8, layer4=16 -> 31.
  // We accumulate node count layer by layer.
  size_t count = 0;
  size_t layer_size = 1;
  for (size_t d = 0; d < branching_.size() && count < static_cast<size_t>(max_size); ++d) {
    if (d == 0) {
      layer_size = static_cast<size_t>(branching_[0]);
    } else {
      layer_size *= static_cast<size_t>(branching_[d]);
    }
    if (count + layer_size > static_cast<size_t>(max_size)) {
      layer_size = max_size - count;
    }
    count += layer_size;
  }
  total_nodes_ = count;
  ET_LOG(
      Info,
      "[TreeBuilder] depth=%zu max_size=%d total_nodes=%zu",
      branching_.size(),
      max_size,
      total_nodes_);
}

void TreeBuilder::build(
    uint64_t root_token,
    const std::function<void(uint64_t, std::vector<float>*)>& /*head_step*/,
    std::vector<TreeNode>* out_nodes) {
  // TODO(phase-4):
  //   1. Initialize root node at slot 0 with depth 0 and root_token.
  //   2. BFS: for each parent in BFS order, call head_step(parent.token_id,
  //      &logits) to obtain candidate distribution for that parent's children.
  //      Take top-k where k = branching_[parent.depth + 1].
  //   3. Assign each child a contiguous slot_idx and link parent_idx.
  //   4. Stop when total_nodes_ slots filled.
  //
  // For now we emit a single-node tree (root only) so downstream code can run.
  out_nodes->clear();
  out_nodes->push_back(TreeNode{/*parent=*/-1, /*depth=*/0, /*slot=*/0, root_token});
  ET_LOG(Error, "[TreeBuilder::build] is TODO — emitted root only");
}

void TreeBuilder::fill_tree_attention_mask(
    const std::vector<TreeNode>& nodes,
    int target_ar_len,
    int /*n_past*/,
    int ctx_len,
    std::byte* mask_buffer) {
  // TODO(phase-4):
  //   For each row k in [0, target_ar_len):
  //     If k >= nodes.size(): write -inf everywhere (slot inactive).
  //     Else:
  //       - For col c in [0, n_past): write 0 (allow attend to past KV).
  //       - For col c in [n_past, n_past + target_ar_len): write 0 iff c-n_past
  //         is an ancestor of k in `nodes` (or c-n_past == k itself), else -inf.
  //       - Beyond that: -inf.
  //   Dtype: the parent target was compiled with a specific mask dtype (fp16
  //   typically). Use the same TensorImpl helpers that TokenGenerator uses.
  std::memset(mask_buffer, 0, target_ar_len * ctx_len * 2 /* fp16 */);
  ET_LOG(
      Error,
      "[TreeBuilder::fill_tree_attention_mask] is TODO — wrote zeros (broken)");
  (void)nodes;
}

void TreeBuilder::fill_positions(
    const std::vector<TreeNode>& nodes,
    int target_ar_len,
    int cur_pos,
    int32_t* pos_buffer) {
  for (int k = 0; k < target_ar_len; ++k) {
    if (k < static_cast<int>(nodes.size())) {
      pos_buffer[k] = cur_pos + nodes[k].depth;
    } else {
      pos_buffer[k] = cur_pos;  // pad with cur_pos; mask masks them anyway
    }
  }
}

} // namespace example
