# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DFlash-draft MHA -> SHA, the draft's counterpart of ConvertMhaToSha.

It reuses ConvertMhaToSha's recursive split (which already handles the bmm / view /
permute decomposition of the batched attention) and only turns on two draft-specific
behaviors via dflash_mode (see convert_mha_to_sha.py):

  * K/V cat: keep the batched ``cat([past_k, k_new, kb])`` as one tensor and slice its
    OUTPUT per head (cat->slice), so ``past_k`` / ``past_v`` stay single 4-D graph
    inputs. The generic pass slices those uint8 inputs into per-head cats (slice->cat),
    which the HTP skel aborts on for the non-causal 3-way cat (QNN 1003).
  * GQA via ``index_select`` (the draft uses index_select, not repeat_kv): descend into
    the source cat with ``kv_sha`` and replicate ``n_rep``, mirroring ``_visit_reshape``.

Quantization stays MHA (the forward is batched), so the scores / softmax / attn-out
activations are calibrated as MHA; ``_copy_meta`` then copies that single shared MHA
scale onto every per-head clone -- exactly what the target's ConvertMhaToSha does.
"""

from executorch.backends.qualcomm._passes.convert_mha_to_sha import ConvertMhaToSha


class DFlashMhaToSha(ConvertMhaToSha):
    def __init__(self, edge_program, verbose=False):
        super().__init__(edge_program, verbose=verbose, dflash_mode=True)
