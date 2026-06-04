#!/usr/bin/env python3
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dump the target model's token embedding table to embed.bin for the EAGLE runtime.

The EAGLE-3 head shares the target's token embeddings (the head checkpoint does
NOT include embed_tokens.weight). To keep the head forward pass self-contained
at runtime, we extract ``model.embed_tokens.weight`` from the HuggingFace target
checkpoint and save it as raw fp16 row-major binary.

Layout of embed.bin:
    dtype:  float16 (IEEE 754 half-precision)
    shape:  [vocab_size, hidden_size]   (row-major)
    size:   vocab_size * hidden_size * 2 bytes

    e.g. Qwen3-1.7B: 151936 * 2048 * 2 = ~596 MB

Usage:
    python dump_eagle_embed.py \\
        --target_ckpt /path/to/Qwen3-1.7B/ \\
        --artifact    /path/to/eagel_qnn/
"""

import argparse
import json
import logging
import os
import sys

import torch


def load_embed_weight(ckpt_dir: str) -> torch.Tensor:
    """Return embed_tokens.weight as fp16 CPU tensor."""
    # Prefer safetensors shards (faster mmap), fall back to .bin.
    index_json = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(index_json):
        with open(index_json) as f:
            index = json.load(f)
        shard_file = index["weight_map"]["model.embed_tokens.weight"]
        shard_path = os.path.join(ckpt_dir, shard_file)
        from safetensors import safe_open

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            w = f.get_tensor("model.embed_tokens.weight")
        return w.to(torch.float16)

    # Single safetensors file
    st_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(st_path):
        from safetensors.torch import load_file

        sd = load_file(st_path)
        return sd["model.embed_tokens.weight"].to(torch.float16)

    # pytorch_model.bin shards
    bin_index = os.path.join(ckpt_dir, "pytorch_model.bin.index.json")
    if os.path.exists(bin_index):
        with open(bin_index) as f:
            index = json.load(f)
        shard_file = index["weight_map"]["model.embed_tokens.weight"]
        shard_path = os.path.join(ckpt_dir, shard_file)
        sd = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
        return sd["model.embed_tokens.weight"].to(torch.float16)

    # Single pytorch_model.bin
    single_bin = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.exists(single_bin):
        sd = torch.load(single_bin, map_location="cpu", weights_only=True, mmap=True)
        return sd["model.embed_tokens.weight"].to(torch.float16)

    raise FileNotFoundError(
        f"Cannot find model.embed_tokens.weight in {ckpt_dir}. "
        "Make sure the directory contains a HuggingFace checkpoint."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target_ckpt",
        required=True,
        help="Directory of the HuggingFace target model (e.g. Qwen3-1.7B).",
    )
    parser.add_argument(
        "--artifact",
        required=True,
        help="Output directory; embed.bin will be written here.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    os.makedirs(args.artifact, exist_ok=True)

    logging.info("Loading embed_tokens.weight from %s ...", args.target_ckpt)
    w = load_embed_weight(args.target_ckpt)  # [vocab, hidden], fp16, contiguous

    if not w.is_contiguous():
        w = w.contiguous()

    vocab_size, hidden_size = w.shape
    out_path = os.path.join(args.artifact, "embed.bin")
    with open(out_path, "wb") as f:
        f.write(w.numpy().tobytes())

    file_size = os.path.getsize(out_path)
    logging.info(
        "Dumped embed.bin -> %s\n"
        "  dtype:  float16\n"
        "  shape:  [%d, %d]  (vocab_size x hidden_size)\n"
        "  size:   %d bytes (%.1f MB)",
        out_path,
        vocab_size,
        hidden_size,
        file_size,
        file_size / (1024**2),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
