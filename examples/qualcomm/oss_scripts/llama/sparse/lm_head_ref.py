"""从 build 自己的 qdq_lm_head.pt2 里读权重、per-channel scale 和两端编码。

单独一个模块是因为它不该依赖 QNN —— 主机侧的数值校验用得上它，而那个环境
装不了 backends.qualcomm。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401  注册 qdq 图里的算子
import torchao.quantization.pt2e  # noqa: F401

DQ_OP = torch.ops.quantized_decomposed.dequantize_per_tensor.default


def read_reference(build_dir: Path):
    gm = torch.export.load(str(Path(build_dir) / "qdq_lm_head.pt2")).module()
    w = pc_scale = None
    for _, b in list(gm.named_buffers()) + list(gm.named_parameters()):
        if b.numel() > 1e8:
            w = b.detach().squeeze()
        elif b.dim() == 1 and b.numel() > 1000 and b.dtype == torch.float32:
            pc_scale = b.detach()
    enc = {}
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target is DQ_OP:
            enc["in" if "in" not in enc else "out"] = (float(n.args[1]), int(n.args[2]))
    if w is None or pc_scale is None or "out" not in enc:
        raise SystemExit("qdq_lm_head.pt2 里读不全权重/scale/编码")
    return w, pc_scale, enc["in"], enc["out"]
