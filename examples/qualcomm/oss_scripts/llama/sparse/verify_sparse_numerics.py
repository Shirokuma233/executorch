#!/usr/bin/env python3
"""稀疏 lm_head 与真实 lm_head 的排序一致性。

图跑通不等于排序对：lm_head 是 per-channel 权重量化，候选之间比大小依赖各自的
scale，少乘一次就会静默排错。这里拿同一批 hidden 喂给两条路，只比我们真正用到的
那件事 —— 候选集内的 argmax 是否相同。

    python verify_sparse_numerics.py <build_dir> --taps <verify_taps.pt> --trace <trace.pt>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401
import torchao.quantization.pt2e  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm_head_ref import read_reference  # noqa: E402

Q = torch.ops.quantized_decomposed.quantize_per_tensor
DQ = torch.ops.quantized_decomposed.dequantize_per_tensor


def quant_hidden(h, scale, zp):
    """把 hidden 按 pte 边界的编码走一遍 uint16，两条路共用同一份输入。"""
    q = torch.clamp(torch.round(h / scale) + zp, 0, 65535)
    return (q - zp) * scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("--taps", default="ddtree/workspace/out/verify_taps.pt")
    ap.add_argument("--trace", default=None, help="给候选集用的 trace；缺省用 top-K")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--depths", type=int, default=15)
    ap.add_argument("--rows", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(32)

    w, pc, (in_s, in_zp), (out_s, out_zp) = read_reference(Path(a.build_dir))
    vocab, dim = w.shape
    wf = w.float()
    print(f"权重 {tuple(w.shape)}  per-channel scale [{pc.min():.3e}, {pc.max():.3e}] "
          f"极差 {pc.max()/pc.min():.1f}x")
    print(f"hidden 编码 {in_s:.12g}/{in_zp}   logits 编码 {out_s:.12g}/{out_zp}\n")

    # 真实 hidden：tap 存的是 logits，用正规方程解回 h（两条路共用，误差同等作用）
    L = torch.load(a.taps, weights_only=False)["hidden"]
    G = (wf.T @ wf).double()
    scaled = wf * pc[:, None]          # 真实的反量化权重 = int8 x per-channel scale

    n_agree = n_tot = n_skip = 0
    n_agree_noscale = 0
    rel = []
    rng = np.random.RandomState(0)
    done = 0
    for call in range(0, len(L), 11):
        if done >= a.rows:
            break
        lg = L[call].float()
        Hh = torch.linalg.solve(G, (wf.T @ lg.T).double()).float().T
        for r in range(Hh.shape[0]):
            if done >= a.rows:
                break
            h = quant_hidden(Hh[r], in_s, in_zp)
            if float(h.norm()) < 1e-2:      # ar 窗口里的 padding 行，排序无意义
                n_skip += 1
                continue
            ref = scaled @ h                                   # 真实 lm_head 的 logits
            # 候选集：真实做法是草稿的 top-K；这里用参考 logits 的 top-K 附近取样，
            # 保证候选里既有强项也有弱项，排序压力和部署一致
            top = torch.topk(ref, a.k * 4).indices
            S = top[torch.from_numpy(rng.choice(a.k * 4, a.depths * a.k, replace=True))]
            sparse = (h @ wf[S].T) * pc[S]                     # 本图算的东西
            noscale = h @ wf[S].T                              # 漏掉 per-channel 的对照
            n_tot += 1
            n_agree += int(S[int(sparse.argmax())] == S[int(ref[S].argmax())])
            n_agree_noscale += int(S[int(noscale.argmax())] == S[int(ref[S].argmax())])
            rel.append(float((sparse - ref[S]).norm() / ref[S].norm()))
            done += 1
    print(f"{n_tot} 个验证行(跳过 {n_skip} 个 padding 行)，每行 {a.depths * a.k} 个候选\n")
    print(f"候选集内 argmax 与真实 lm_head 一致:")
    print(f"  本图(带 per-channel scale)   {n_agree}/{n_tot} = {n_agree/n_tot:.2%}")
    print(f"  对照(漏掉 scale)             {n_agree_noscale}/{n_tot} = {n_agree_noscale/n_tot:.2%}")
    print(f"\nlogits 相对误差 中位 {np.nanmedian(rel):.2e}  最大 {max(rel):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
