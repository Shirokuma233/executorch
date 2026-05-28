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
#include <functional>
#include <vector>

namespace example {

/**
 * @file tree_attention.h
 * @brief Static balanced tree topology for EAGLE-3 verification (Phase 4).
 *
 * The tree size is fixed at compile time to `next_pow2(max_tree_size)`. The
 * topology is described by a "branching per depth" vector, e.g. `{1,2,2,2,2}`
 * means: 1 root, 2 children per depth-1 node, 2 per depth-2, etc.
 *
 * The flattened node order is BFS so that the parent index of a child is
 * always a smaller index — required for the DAG-walking accept logic.
 */
struct TreeNode {
  int16_t parent_idx;     // -1 for root
  int16_t depth;
  int16_t slot_idx;       // position in target ar slot
  uint64_t token_id;      // committed (target) token id at this node
};

class TreeBuilder {
 public:
  // branching_per_depth: e.g. {1, 2, 2, 2, 2, 2, 2, 4} -> 16 nodes total.
  TreeBuilder(std::vector<int> branching_per_depth, int max_size);

  size_t total_nodes() const { return total_nodes_; }
  int max_depth() const { return static_cast<int>(branching_.size()); }

  /**
   * Build the tree by repeatedly invoking `head_step(parent_token, &logits)`
   * to obtain children's logits, then taking top-k of `logits` according to
   * the topology.
   *
   * `out_nodes` is populated with `total_nodes_` entries in BFS order.
   *
   * TODO(phase-4): full implementation — currently stubbed.
   */
  void build(
      uint64_t root_token,
      const std::function<void(uint64_t parent, std::vector<float>* logits)>& head_step,
      std::vector<TreeNode>* out_nodes);

  /**
   * Fill the target's tree-attention mask for `target_ar_len` slots.
   * Slot `k` attends to the union of:
   *   - all positions in [0, n_past)  (past KV)
   *   - all ancestors of node `k` in the tree
   * Inactive slots (k >= total_nodes_) attend to nothing.
   *
   * `mask_buffer` is `[ar_len, ctx_len]` of dtype matching the model. Phase 4
   * scaffold writes 0.0 / -inf as fp16 (TODO: respect actual dtype).
   */
  void fill_tree_attention_mask(
      const std::vector<TreeNode>& nodes,
      int target_ar_len,
      int n_past,
      int ctx_len,
      std::byte* mask_buffer);

  /**
   * Fill `pos_buffer[k] = cur_pos + nodes[k].depth` for active slots, 0 else.
   */
  void fill_positions(
      const std::vector<TreeNode>& nodes,
      int target_ar_len,
      int cur_pos,
      int32_t* pos_buffer);

 private:
  std::vector<int> branching_;
  size_t total_nodes_{0};
};

} // namespace example
