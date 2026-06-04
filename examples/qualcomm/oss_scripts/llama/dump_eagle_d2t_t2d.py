#!/usr/bin/env python3
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dump d2t / t2d binaries from an EAGLE-3 checkpoint without recompiling.

Reads SafeAILab official EAGLE-3 checkpoint (`pytorch_model.bin` or
`model.safetensors`) and writes raw binary files consumable by the QNN
runtime's EagleSampler:

    d2t.bin  : int64[draft_vocab_size]    target_id = draft_id + d2t[draft_id]
    t2d.bin  : bool[target_vocab_size]    is target_id in draft subset

Usage:
    python dump_eagle_d2t_t2d.py \\
        --eagle_ckpt /path/to/eagle/ \\
        --artifact   /path/to/eagel_qnn/

This is the manual fallback for `eagle.md` §7.1.A (saves ~22 minutes vs
re-running the full compile pipeline that emits these as a side effect).
"""

import argparse
import logging
import os
import sys

import torch


def load_state_dict(ckpt_dir: str) -> dict:
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    st_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True, mmap=True)
    if os.path.exists(st_path):
        from safetensors.torch import load_file

        return load_file(st_path)
    raise FileNotFoundError(
        f"No pytorch_model.bin or model.safetensors found in {ckpt_dir}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eagle_ckpt",
        required=True,
        help="Directory holding the EAGLE-3 head checkpoint.",
    )
    parser.add_argument(
        "--artifact",
        required=True,
        help="Output directory; d2t.bin / t2d.bin will be written here.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not os.path.isdir(args.artifact):
        os.makedirs(args.artifact, exist_ok=True)

    sd = load_state_dict(args.eagle_ckpt)
    if "d2t" not in sd or "t2d" not in sd:
        logging.error(
            "Checkpoint missing d2t/t2d (vocab_size == draft_vocab_size?). "
            "Available keys: %s",
            sorted(k for k in sd.keys() if k in ("d2t", "t2d")),
        )
        return 1

    d2t = sd["d2t"].detach().cpu().to(torch.int64).contiguous()
    t2d = sd["t2d"].detach().cpu().to(torch.bool).contiguous()

    if d2t.dim() != 1 or t2d.dim() != 1:
        logging.error(
            "Unexpected shapes: d2t %s, t2d %s (both must be 1-D).",
            tuple(d2t.shape),
            tuple(t2d.shape),
        )
        return 1

    d2t_path = os.path.join(args.artifact, "d2t.bin")
    t2d_path = os.path.join(args.artifact, "t2d.bin")

    with open(d2t_path, "wb") as f:
        f.write(d2t.numpy().tobytes())
    with open(t2d_path, "wb") as f:
        f.write(t2d.numpy().tobytes())

    logging.info(
        "Dumped:\n"
        "  d2t -> %s   (%d int64, %d bytes; draft_vocab_size = %d)\n"
        "  t2d -> %s   (%d bool,  %d bytes; target_vocab_size = %d)",
        d2t_path,
        d2t.numel(),
        os.path.getsize(d2t_path),
        d2t.numel(),
        t2d_path,
        t2d.numel(),
        os.path.getsize(t2d_path),
        t2d.numel(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
