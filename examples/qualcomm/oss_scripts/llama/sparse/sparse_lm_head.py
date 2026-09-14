#!/usr/bin/env python3
"""导出稀疏验证用的 lm_head 图:候选行由 CPU 在运行时喂进来，图内零权重。

现有 lm_head 的代价是把 389 MB 权重流过 DRAM 一遍（设备实测 10.18 ms，
389MB / 38.7GB/s = 10.05 ms，对得上到 1%），而验证只需要在草稿提名的候选里排序。
所以这张图不带权重，候选行作为运行时输入。

    logits[b, ar, N] = (hidden[b, ar, H] @ cand_w[N, H].T) * cand_scale[N]

`cand_scale` 乘在输出侧而不是权重侧：输出是 32x1920 = 6 万个元素，权重是
1920x2560 = 490 万个；而且大操作数保持纯整数，边界才有机会压到 8bit。
这一步不能省 —— lm_head 是 per-channel 权重量化（_scale_0 是 151936 维），
候选之间比大小依赖各自的 scale，忽略它必然排错。

hidden 的输入编码必须和现有 lm_head 逐位一致（否则 pte 边界解释错），
所以从 build 自己的 qdq_lm_head.pt2 里读出来注入，不靠校准碰运气。

    python sparse_lm_head.py <build_dir> -o out/ --n-cand 1920 --ar 32
"""
from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path

import numpy as np
import torch
import sys
import torch.ao.quantization.fx._decomposed  # noqa: F401  注册 qdq 图里的算子
import torchao.quantization.pt2e  # noqa: F401
from torch import nn

from executorch.backends.qualcomm._passes import TagQuantIO
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.extension.export_util.utils import save_pte_program
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm_head_ref import read_reference  # noqa: E402

Q_OP = torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ_OP = torch.ops.quantized_decomposed.dequantize_per_tensor.default


class SparseLmHead(nn.Module):
    """logits[1, AR, N] = hidden[1, AR, H] @ cand_w[N, H].T

    `cand_w` arrives already dequantized -- the CPU folds each row's
    per-channel scale in while it gathers, which it can do for free because it
    is converting int8 to float anyway. Keeping the scale out of the graph is
    what makes this numerically stable: an unscaled dot product has a range that
    swings with the candidate set (the drafter's top-K score far higher than
    random rows), so quantizing it needs calibration data that matches runtime
    exactly. The scaled product is just the logits, whose encoding lm_head
    already pins down.
    """

    def __init__(self, ar: int, dim: int, n_cand: int):
        super().__init__()
        self.ar, self.dim, self.n_cand = ar, dim, n_cand

    def forward(self, hidden, cand_w):
        return torch.matmul(hidden, cand_w.transpose(0, 1))

    def get_example_input(self):
        return (
            torch.randn(1, self.ar, self.dim),
            torch.randn(self.n_cand, self.dim) * 1e-2,
        )


def tag_cand_w(node, cand_shape):
    """只标候选行这一个输入为 uint16，halving the biggest buffer in the round.

    Deliberately narrow. Tagging the output too is a silent trap: uint16 codes
    read back as float32 are denormals, which passes every "is this payload
    absurd" check while the ranking is noise.

    int8 would halve it again but is not safe here -- cand_w holds already
    dequantized rows, and the per-channel scales span 5.5x, so one per-tensor
    8-bit encoding over that range loses about 2.5 bits against the original
    8-bit-per-channel. 16 bits per tensor is strictly better than what the
    weights started as.
    """
    if node.op == "placeholder" and tuple(node.meta["val"].size()) == cand_shape:
        return torch.uint16
    return None


def override_placeholder_scale(gm, shape, scale, zero_point) -> int:
    """把某个输入的量化参数钉死。照抄 llm_wrappers._override_lm_head_input_scale。"""
    patched = 0
    for node in gm.graph.nodes:
        if node.op != "placeholder" or tuple(node.meta["val"].size()) != tuple(shape):
            continue
        for user in list(node.users):
            if user.target is Q_OP:
                user.args = (user.args[0], scale, zero_point, *user.args[3:])
                patched += 1
                for dq in list(user.users):
                    if dq.target is DQ_OP:
                        dq.args = (dq.args[0], scale, zero_point, *dq.args[3:])
    gm.recompile()
    return patched


def override_output_scale(gm, scale, zero_point) -> int:
    """输出就是 logits，所以直接用 lm_head 自己的编码，不靠校准去猜量程。"""
    out = next(n for n in gm.graph.nodes if n.op == "output")
    patched = 0
    for node in list(out.args[0]):
        if not isinstance(node, torch.fx.Node):
            continue
        if node.target is DQ_OP:
            node.args = (node.args[0], scale, zero_point, *node.args[3:])
            q = node.args[0]
            if isinstance(q, torch.fx.Node) and q.target is Q_OP:
                q.args = (q.args[0], scale, zero_point, *q.args[3:])
            patched += 1
    gm.recompile()
    return patched


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--n-cand", type=int, default=1920, help="15 深度 x K")
    ap.add_argument("--ar", type=int, default=32)
    ap.add_argument("--soc", default="SM8750")
    ap.add_argument("--calib", type=int, default=8, help="校准批数")
    ap.add_argument(
        "--calib-taps",
        default=None,
        help="真实 logits tap（verify_taps.pt）。缺省用合成数据，但那会低估中间点积的"
        "量程：真实候选是草稿的 top-K，与 hidden 的点积远大于随机词表行。",
    )
    a = ap.parse_args()

    build = Path(a.build_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    w, pc_scale, (in_s, in_zp), (out_s, out_zp) = read_reference(build)
    vocab, dim = w.shape
    logging.info(
        "参考: 权重 %s  per-channel scale %s  hidden 编码 %.12g/%d  logits 编码 %.12g/%d",
        tuple(w.shape), tuple(pc_scale.shape), in_s, in_zp, out_s, out_zp,
    )

    model = SparseLmHead(a.ar, dim, a.n_cand).eval()
    example = model.get_example_input()

    chipset = get_soc_to_chipset_map()[a.soc]
    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend, soc_model=chipset
    )
    quantizer.set_default_quant_config(QuantDtype.use_16a8w)

    gm = torch.export.export(model, example, strict=True).module()
    gm = prepare_pt2e(gm, quantizer)

    # 用真实的候选行和真实 scale 校准，让 observer 看到部署时的量程。
    # hidden 那一路的编码后面会被钉死，所以它的范围只影响不到最终结果。
    rng = np.random.RandomState(0)
    wf = w.float()
    hiddens, cands = [], []
    if a.calib_taps:
        # tap 存的是 logits，用正规方程解回 hidden；候选取该行 logits 的 top-K，
        # 也就是草稿会提名的那批高分行。
        L = torch.load(a.calib_taps, weights_only=False)["hidden"]
        G = (wf.T @ wf).double()
        for call in range(0, len(L), max(len(L) // a.calib, 1)):
            if len(hiddens) >= a.calib:
                break
            lg = L[call].float()
            hh = torch.linalg.solve(G, (wf.T @ lg.T).double()).float().T
            q = torch.clamp(torch.round(hh / in_s) + in_zp, 0, 65535)
            hiddens.append(((q - in_zp) * in_s).unsqueeze(0)[:, : a.ar])
            top = torch.topk(lg[0], a.n_cand).indices
            cands.append(top)
        logging.info("用 %d 批真实 hidden + 真实候选校准", len(hiddens))
    if not hiddens:
        lo, hi = (0 - in_zp) * in_s, (65535 - in_zp) * in_s
        for _ in range(a.calib):
            hiddens.append(torch.empty(1, a.ar, dim).uniform_(lo / 8, hi / 8))
            cands.append(torch.from_numpy(rng.randint(0, vocab, a.n_cand)))
        logging.warning("用合成数据校准 —— 会低估中间点积的量程，建议传 --calib-taps")
    for h, rows in zip(hiddens, cands):
        gm(h, wf[rows] * pc_scale[rows][:, None])
    gm = convert_pt2e(gm)

    n = override_placeholder_scale(gm, (1, a.ar, dim), in_s, in_zp)
    if n == 0:
        raise SystemExit("hidden 输入的 quantize 节点没找到，边界编码没注入")
    # 每行的 int8 都用满 ±127（per-channel 对称量化就是这么定 scale 的），
    # 所以反量化后的全局最大值恰好是 127 * max(pc_scale)。runner 用同一个公式，
    # 不需要额外传一个文件。
    w_scale = float(pc_scale.max()) * 127.0 / 32767.0
    n2 = override_placeholder_scale(gm, (a.n_cand, dim), w_scale, 32768)
    if n2 == 0:
        raise SystemExit("cand_w 的 quantize 节点没找到")
    logging.info(
        "cand_w 编码: scale=%.12g zp=32768  (= 127 * max_pc_scale %.6g / 32767)",
        w_scale, float(pc_scale.max()),
    )
    m = override_output_scale(gm, out_s, out_zp)
    if m == 0:
        raise SystemExit("输出的 dequantize 节点没找到，logits 编码没注入")
    logging.info("编码已注入: hidden 输入 %d 处, logits 输出 %d 处", n, m)

    spec = generate_qnn_executorch_compiler_spec(
        soc_model=chipset,
        backend_options=generate_htp_compiler_spec(use_fp16=False),
    )
    # cand_w is the round's largest buffer, and the CPU gather writing it competes with
    # the NPU for the memory bus: measured, verify_exec runs 2.2 ms/round slower
    # with the gather alongside. Halving it is the point of tagging.
    # hidden and the logits output stay plain float32 on purpose.
    passes = get_capture_program_passes()
    passes[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
    passes[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY]["get_quant_io_dtype_fn"] = (
        partial(tag_cand_w, cand_shape=(a.n_cand, dim))
    )
    prog = to_edge_transform_and_lower_to_qnn(
        gm, example, spec, passes_job=passes
    ).to_executorch(
        ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
        )
    )
    name = f"sparse_lm_head_n{a.n_cand}_ar{a.ar}"
    save_pte_program(prog, name, str(out))

    # runner 要的 per-channel scale 表：按 token id 索引，和行主序表同序
    pc_scale.numpy().astype(np.float32).tofile(out / "lm_head_pc_scale.f32")
    logging.info(
        "写出 %s.pte (%.2f MiB) + lm_head_pc_scale.f32 (%d 项)",
        name, (out / f"{name}.pte").stat().st_size / 2**20, pc_scale.numel(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
