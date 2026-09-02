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


class DFlashMhaToShaDeep(ConvertMhaToSha):
    """照 target 的拆法拆草稿。设备上会 1007 —— 本分支就是在查为什么。

    与 DFlashMhaToSha 的差别只有两处，其余（递归骨架、_copy_meta 复制共享 MHA
    scale、_concat_sha_nodes 把图输出拼回去）完全共用：

      * matmul/bmm 交回 _visit_default，两个操作数都递归 —— 于是 Q 一路走到
        q_proj，_visit_linear_conv 按输出通道劈开它的权重，q_norm / RoPE 跟着
        碎成逐头。
      * cat 交回 _visit_cat，递归进三个输入 —— past_k/past_v 先按头切片，再逐头
        各拼一次三段。

    index_select 仍走 _visit_index_select：通用 pass 没有这个 visitor。

    等价性已验证：.docs/mha2sha_target_vs_dflash.py 把浅拆、深拆、以及批量前向
    三者对同一组输入断言逐元素相等（max|delta| = 0，含 k_new / v_new 图输出）。
    """

    def __init__(self, edge_program, verbose=False):
        super().__init__(
            edge_program, verbose=verbose, dflash_mode=True, dflash_deep=True
        )


class DFlashMhaToShaDeepProj(ConvertMhaToSha):
    """草稿专用的半深拆：劈投影，但保住批量三段 cat。

    动机：直接拿 target 的通用 pass 去拆草稿是不成立的 —— 两张图的 attention
    结构不一样。深拆 K/V 投影必须递归穿过那个**三段非因果 cat**，而通用
    _visit_cat 会把 past_k / past_v 这两个 uint8 图输入切成逐头 slice_copy 再
    逐头拼；那正是 _visit_cat_dflash 存在的理由。

    于是这一版只往 Q 那条腿深拆（投影权重按头劈开、q_norm / RoPE 跟着逐头），
    K/V 侧维持批量、在 cat 输出上切片。代价是拓扑不对称，而"不对称必崩"那条
    旧归因已经不可靠：1003 和 1007 都是 QNN_COMMON_ERROR_SYSTEM*（"与平台/OS
    服务通信失败"），属 common 段而非 graph 段，不是拓扑校验的结果。

    另外草稿有一处 target 从来没有的形态：q_norm / k_norm 作用在**折叠过的 3D**
    张量上（[1, B*nH, D]，头和块位置折在一起），target 是在 4D [b,nH,s,D] 上做。
    深拆递归穿过那个 reshape 时面对的就是这种折叠形态。
    """

    def __init__(self, edge_program, verbose=False):
        super().__init__(
            edge_program,
            verbose=verbose,
            dflash_mode=True,
            dflash_deep=True,
            dflash_batched_cat=True,
        )
