#!/usr/bin/env python3
"""把 lm_head_qnn.pte 里的权重反交织成行主序的 int8 表。

稀疏验证要 CPU 按行取候选，但 QNN 为 HTP 的数据流把权重排成了 depth32 四层交织，
一行的 2560 字节摊在 163840 字节里（40 页）。CPU 直接按那个布局取行会慢 12 倍、
多占 13 倍的页。所以离线反交织一次，产出一份行主序表给 CPU 用。

布局（QNN 2.45 / SM8750 / Linear 转 1x1 Conv 后）:

    offset(v,c) = BASE + (v//64)*163840      每 64 行一个大块
                       + (c//32)*2048        每 32 列一个单元
                       + ((v%64)//32)*1024   块内上/下 32 行
                       + ((c%32)//4)*128     单元内 8 个 128B 子块
                       + (v%32)*4            子块内 32 行各占 4 字节
                       + (c%4)               4 个连续通道

BASE 由与 qdq 图的逐字节比对确定，不硬编码。

    python deinterleave_lm_head.py <build_dir> -o lm_head_rowmajor.bin
"""
from __future__ import annotations

import argparse
import mmap
import sys
from pathlib import Path

import numpy as np
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401  注册 qdq 图里的算子
import torchao.quantization.pt2e  # noqa: F401


def layout_offsets(v: np.ndarray, hidden: int) -> np.ndarray:
    """给定行号数组，返回每行 hidden 个元素在权重区内的字节偏移。"""
    c = np.arange(hidden)
    return (v[:, None] // 64) * 163840 \
        + (c // 32) * 2048 \
        + ((v[:, None] % 64) // 32) * 1024 \
        + ((c % 32) // 4) * 128 \
        + (v[:, None] % 32) * 4 \
        + (c % 4)


def find_base(blob: np.ndarray, ref: np.ndarray, hidden: int, probes=(0, 1, 1000)) -> int:
    """靠已知行在文件里定位权重区起点。"""
    pat = np.ascontiguousarray(ref[0, :4]).tobytes()
    start = 0
    while True:
        pos = blob.tobytes().find(pat, start) if False else _find(blob, pat, start)
        if pos < 0:
            raise SystemExit("定位不到权重区起点：布局可能随 QNN 版本变了，先跑 verify 模式")
        off = layout_offsets(np.array(probes), hidden) + pos
        if off.max() < len(blob) and all(
            np.array_equal(blob[off[i]], ref[p]) for i, p in enumerate(probes)
        ):
            return pos
        start = pos + 1


def _find(blob: np.ndarray, pat: bytes, start: int) -> int:
    mv = memoryview(blob.data).cast("B")
    return bytes(mv).find(pat, start)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--chunk", type=int, default=4096, help="每次反交织的行数")
    a = ap.parse_args()

    d = Path(a.build_dir)
    gm = torch.export.load(str(d / "qdq_lm_head.pt2")).module()
    ref = None
    for _, b in list(gm.named_buffers()) + list(gm.named_parameters()):
        if b.numel() > 1e8:
            ref = b.detach().squeeze().numpy()
            break
    if ref is None:
        raise SystemExit("qdq_lm_head.pt2 里找不到权重")
    vocab, hidden = ref.shape
    print(f"参考权重 {ref.shape} {ref.dtype}")

    fh = open(d / "lm_head_qnn.pte", "rb")
    blob = np.frombuffer(mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ), dtype=np.int8)
    base = find_base(blob, ref, hidden)
    print(f"权重区起点 base = {base}  区长 {(vocab // 64) * 163840} 字节")

    rows = np.random.RandomState(0).randint(0, vocab, 500)
    off = layout_offsets(rows, hidden) + base
    bad = int((blob[off] != ref[rows]).any(axis=1).sum())
    print(f"抽验 500 行: 不符 {bad}")
    if bad:
        raise SystemExit("布局对不上，不要产出表")
    if a.verify_only:
        return 0

    with open(a.out, "wb") as f:
        for lo in range(0, vocab, a.chunk):
            hi = min(lo + a.chunk, vocab)
            v = np.arange(lo, hi)
            f.write(np.ascontiguousarray(blob[layout_offsets(v, hidden) + base]).tobytes())
            if lo % (a.chunk * 8) == 0:
                print(f"  {hi}/{vocab}", end="\r", flush=True)
    print(f"\n写出 {a.out}  {Path(a.out).stat().st_size} 字节")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
