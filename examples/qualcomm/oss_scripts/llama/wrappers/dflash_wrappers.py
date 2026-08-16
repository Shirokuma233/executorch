# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DFlash block-diffusion draft model — QNN static-graph reimplementation.

Reference: dflash/draft_models/*/modeling_dflash.py (z-lab DFlash).

The upstream draft is a small Qwen3 transformer (a few full-attention layers)
that drafts a whole block of `block_size - 1` tokens in ONE forward via
bidirectional (non-causal) attention over:
    K/V = [ k_proj(context) ‖ k_proj(noise_block) ]
where `context` = fc(concat of target hidden at selected layers) and
`noise_block` = target.embed_tokens(block) with mask tokens in the slots to
be drafted.

Differences vs upstream, required for a static QNN graph:
  * The variable-length draft KV-cache is replaced by a FIXED-size context
    input (`target_hidden`, padded to `max_context_len`) plus an additive
    attention mask. Functionally identical (attention over the same
    context+block), just recomputes the context projection each block instead
    of caching it.
  * RoPE uses a precomputed cos/sin table gathered by an explicit position id
    input (mirrors static_llama), so the graph has no dynamic rotary op.

The draft has NO embed_tokens and NO lm_head of its own — both are reused from
the target (embed via the runtime `embed.bin`, lm_head applied by the runtime).
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Qwen3-equivalent RMSNorm, written exactly as HF's Qwen2RMSNorm so that
    RecomposeRmsNorm fuses it into the QNN native OpRmsNorm.

    Two things must hold, and each one alone is enough to break it.

    `self.weight` must reach the final mul as a bare parameter: the pass wants a
    get_attr/placeholder/dq there (recompose_rms_norm.py:122) and silently gives up
    otherwise, so a `weight.float()` is enough to hide it.

    And there must be no dtype cast anywhere. An `x.float()` on the input does get
    folded into the fused node, but it lands as an OpRmsNorm with a FLOAT_32 input
    and an fp16 output, and HTP has no op config for that shape -- all 54 norms come
    back `Failed to validate op ... error 0xc26` (3110) and fall out of the delegate.
    (RecomposeRmsNorm._get_input_node would step over the cast, except its skip list
    names aten.to.dtype while export now emits dim_order_ops._to_dim_order_copy.)

    Fusing is the whole ballgame on HTP. The native op holds the sum of squares at
    fp32 range -- device-verified clean at |x| = 65504, the fp16 ceiling -- while the
    unfused primitives square in fp16 and blow up past sqrt(65504) ~= 256, which the
    draft's residual stream clears immediately. Writing `x.float().pow(2)` does not
    save those either: the cast is declared FLOAT_32 in the QNN graph and overflows
    on device anyway. See dflash/tools/probe_rmsnorm_accum.py.

    The price is that eager fp16 now overflows where the device does not, so parity
    must be checked fp32-host against fp16-device, never fp16-host."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class DFlashConfig:
    """Distilled config for the DFlash draft, from the checkpoint config.json."""

    def __init__(self, raw: dict):
        self.hidden_size: int = raw["hidden_size"]
        self.num_attention_heads: int = raw["num_attention_heads"]
        self.num_key_value_heads: int = raw["num_key_value_heads"]
        self.head_dim: int = raw.get(
            "head_dim", self.hidden_size // self.num_attention_heads
        )
        self.intermediate_size: int = raw["intermediate_size"]
        self.num_hidden_layers: int = raw["num_hidden_layers"]
        self.num_target_layers: int = raw["num_target_layers"]
        self.vocab_size: int = raw["vocab_size"]
        # Newer checkpoints nest rope under rope_parameters and hoist the dflash
        # fields to the top level; older ones keep rope_theta flat and the dflash
        # fields under dflash_config.
        rope = raw.get("rope_parameters", raw)
        self.rope_theta: float = float(rope.get("rope_theta", 1000000.0))
        self.rms_norm_eps: float = float(raw.get("rms_norm_eps", 1e-6))
        self.block_size: int = raw["block_size"]
        self.attention_bias: bool = raw.get("attention_bias", False)
        dflash_cfg = raw.get("dflash_config", raw)
        self.mask_token_id: Optional[int] = dflash_cfg.get("mask_token_id", None)
        self.target_layer_ids: List[int] = dflash_cfg.get(
            "target_layer_ids",
            build_target_layer_ids(self.num_target_layers, self.num_hidden_layers),
        )
        # Whether the sink token is excluded. Deliberately NOT read from the
        # checkpoint or any config file: it was, and a stale `drop_sink: true`
        # sitting at the top level of an untracked checkpoint silently decided it
        # for every build while the command line looked like it was in charge.
        # Set from --no_sink by whoever constructs this.
        self.no_sink: bool = False
        # --sink_split: keep feeding the sink row to the draft, but calibrate the
        # draft-facing encodings PER PHASE -- prefill's range covers the sink (it is
        # the only graph that ever processes position 0), decode's is banked fresh
        # from the generation rows. Mutually exclusive with no_sink.
        self.sink_split: bool = False
        # Two decode conventions, and they are off by one from each other:
        #
        #   aligned (DFlashDraftModel, e.g. Qwen3-4B-DFlash-b16)
        #       row k of the block fills slot k, row 0 is discarded.
        #       -> B-1 tokens from hidden[1:], verify [confirmed, d1..d_{B-1}] (B)
        #
        #   shifted (Qwen3DSparkModel / DeepSpec, e.g. dflash_qwen3_4b_block7)
        #       row k predicts the NEXT token, row 0 included.
        #       -> B tokens from hidden[:B], verify [confirmed, s0..s_{B-1}] (B+1)
        #
        # Run a shifted checkpoint under the aligned convention and every drafted
        # token lands one slot early: acceptance collapses to zero.
        # See DeepSpec/deepspec/eval/dspark/draft_ops.py:build_dspark_proposal.
        arch = (raw.get("architectures") or [""])[0]
        self.shifted_decode: bool = "DSpark" in arch
        # transformers HF-style Qwen3MLP expects these attribute names.
        self.hidden_act = raw.get("hidden_act", "silu")
        self.mlp_bias = raw.get("mlp_bias", False)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """Mirror modeling_dflash.build_target_layer_ids (evenly spaced [1, n-3])."""
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


# ---------------------------------------------------------------------------
# RoPE helpers (HF Qwen3 style, precomputed table)
# ---------------------------------------------------------------------------
def _precompute_rope(head_dim: int, max_pos: int, theta: float):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [max_pos, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_pos, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, nH, S, D];  cos/sin: [S, D] -> broadcast [1,1,S,D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Static bidirectional cross+self attention
# ---------------------------------------------------------------------------
class DFlashAttention(nn.Module):
    """Block attends [cached-context ‖ new-context ‖ block] bidirectionally.

    KV-cache design (mirrors EAGLE head / the reference past_key_values_draft):
    the CONTEXT K/V live in a cache; each call only projects the small `new`
    context rows (<= ar) + the block. This keeps the wide 5H tensor tiny and
    lets the long context be tiled like a normal KV cache — no HTP TCM overflow.
    Returns (attn_out, k_new, v_new) where k_new/v_new are the NEW context K/V
    to append to the cache (transposed layout, matching KVManager).
    """

    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        self.nH = cfg.num_attention_heads
        self.nKV = cfg.num_key_value_heads
        self.D = cfg.head_dim
        self.scaling = self.D**-0.5
        H = cfg.hidden_size
        self.q_proj = nn.Linear(H, self.nH * self.D, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(H, self.nKV * self.D, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(H, self.nKV * self.D, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.nH * self.D, H, bias=cfg.attention_bias)
        self.q_norm = RMSNorm(self.D, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.D, eps=cfg.rms_norm_eps)
        # GQA head-expansion index, precomputed as a constant buffer. Gathering K/V
        # with index_select(1, this) == repeat_interleave(n_rep, dim=1) but stays 4D
        # (repeat_interleave's 5D expand/view won't delegate once quantized). Storing
        # the index as a buffer keeps arange/div out of the graph, which also removes
        # the "inDims is NULL" warnings from the .tensor-variant dequant they cause.
        n_rep = self.nH // self.nKV
        self.register_buffer(
            "kv_rep_idx", torch.arange(self.nKV * n_rep) // n_rep, persistent=False
        )

    def forward(
        self,
        block_h: torch.Tensor,   # [1, B, H]      noise block
        ctx_h: torch.Tensor,     # [1, AR, H]     NEW fused context rows (fc+norm'd)
        cos_ctx, sin_ctx,        # [AR, D]        new-context RoPE
        cos_blk, sin_blk,        # [B, D]         block RoPE
        atten_mask,              # [1, B, Cc+AR+B] additive
        past_k, past_v,          # [1,nKV,D,Cc], [1,nKV,Cc,D]  cached context
    ):
        bsz, B = block_h.shape[:2]
        AR = ctx_h.shape[1]
        # queries from the block
        q = self.q_proj(block_h).reshape(bsz, B * self.nH, self.D)
        q = self.q_norm(q).reshape(bsz, B, self.nH, self.D).transpose(1, 2)  # [1,nH,B,D]
        q = _apply_rope(q, cos_blk, sin_blk)

        # NEW context K/V (to cache)
        kc = self.k_norm(self.k_proj(ctx_h).reshape(bsz, AR * self.nKV, self.D))
        kc = kc.reshape(bsz, AR, self.nKV, self.D).transpose(1, 2)           # [1,nKV,AR,D]
        kc = _apply_rope(kc, cos_ctx, sin_ctx)
        k_new = kc.transpose(2, 3)                                           # [1,nKV,D,AR]
        v_new = self.v_proj(ctx_h).reshape(bsz, AR, self.nKV, self.D).transpose(1, 2)  # [1,nKV,AR,D]

        # block K/V (transient, not cached)
        kb = self.k_norm(self.k_proj(block_h).reshape(bsz, B * self.nKV, self.D))
        kb = kb.reshape(bsz, B, self.nKV, self.D).transpose(1, 2)
        kb = _apply_rope(kb, cos_blk, sin_blk).transpose(2, 3)               # [1,nKV,D,B]
        vb = self.v_proj(block_h).reshape(bsz, B, self.nKV, self.D).transpose(1, 2)  # [1,nKV,B,D]

        # assemble full K/V: [cached ctx ; new ctx ; block]
        k = torch.cat([past_k, k_new, kb], dim=3)   # [1,nKV,D, Cc+AR+B]
        v = torch.cat([past_v, v_new, vb], dim=2)   # [1,nKV, Cc+AR+B, D]
        k = k.index_select(1, self.kv_rep_idx)
        v = v.index_select(1, self.kv_rep_idx)
        scores = torch.matmul(q, k) * self.scaling  # q[1,nH,B,D] @ k[1,nH,D,T] -> [1,nH,B,T]
        scores = scores + atten_mask.unsqueeze(1)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [1, nH, B, D]
        out = out.transpose(1, 2).reshape(bsz, B, self.nH * self.D)
        return self.o_proj(out), k_new, v_new


class DFlashDecoderLayer(nn.Module):
    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        H = cfg.hidden_size
        self.self_attn = DFlashAttention(cfg)
        self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(H, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(H, eps=cfg.rms_norm_eps)

    def forward(self, block_h, ctx_h, cos_ctx, sin_ctx, cos_blk, sin_blk, atten_mask, past_k, past_v):
        residual = block_h
        x = self.input_layernorm(block_h)
        x, k_new, v_new = self.self_attn(
            x, ctx_h, cos_ctx, sin_ctx, cos_blk, sin_blk, atten_mask, past_k, past_v
        )
        block_h = residual + x
        residual = block_h
        x = self.post_attention_layernorm(block_h)
        block_h = residual + self.mlp(x)
        return block_h, k_new, v_new


# ---------------------------------------------------------------------------
# Draft model (one block-draft forward)
# ---------------------------------------------------------------------------
class DFlashDraftModel(nn.Module):
    """KV-cached block-draft, exported once per context-append length `ar_len`.

    Each call appends `AR` NEW context rows to a per-layer context K/V cache and
    drafts the block against [cached-context ‖ new-context ‖ block]. The wide 5H
    tensor is only the AR new rows (small); the long context lives in the cache.

    Two graphs share weights in one pte:
      prefill_forward : AR = prefill_ar_len   (seed the cache from the prompt)
      kv_forward      : AR = block_size       (append the accepted rows)
    `Cc + AR == max_context_len` for both, so the mask width is invariant.

    Inputs:
      noise_embedding : [1, B, H]
      atten_mask      : [1, B, Cc + AR + B]  additive
      new_context     : [1, AR, num_ctx_layers*H]   NEW context rows (raw 5H)
      context_pos     : [1, AR]   RoPE positions of new context
      block_pos       : [1, B]    RoPE positions of block
      past_k[L]       : [1, nKV, D, Cc]   per layer   (Cc = max_context_len - AR)
      past_v[L]       : [1, nKV, Cc, D]   per layer
    Outputs:
      block hidden [1, B, H]; then per layer k_new [1,nKV,D,AR], v_new [1,nKV,AR,D]
    """

    def __init__(
        self,
        cfg: DFlashConfig,
        max_context_len: int,
        ar_len: int,
        with_lm_head: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.H = cfg.hidden_size
        self.B = cfg.block_size
        self.AR = ar_len  # new-context append length per call
        self.nKV = cfg.num_key_value_heads
        self.D = cfg.head_dim
        self.n_layers = cfg.num_hidden_layers
        self.max_context_len = max_context_len
        self.cache_len = max_context_len - self.AR  # Cc
        self.num_ctx_layers = len(cfg.target_layer_ids)
        # Checkpoints that ship an lm_head can do the vocab projection and the argmax
        # on HTP, turning a 2.3 GMAC host scan of a 743 MB table (1.3 s/block, 94% of
        # runtime) into B-1 token ids coming straight out of the graph.
        self.lm_head = (
            nn.Linear(self.H, cfg.vocab_size, bias=False) if with_lm_head else None
        )
        self.fc = nn.Linear(self.num_ctx_layers * self.H, self.H, bias=False)
        self.hidden_norm = RMSNorm(self.H, eps=cfg.rms_norm_eps)
        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(self.H, eps=cfg.rms_norm_eps)
        # Block slots sit past the last context position, hence the +B.
        cos, sin = _precompute_rope(
            cfg.head_dim, max_context_len + self.B, cfg.rope_theta
        )
        self.register_buffer("freqs_cos", cos, persistent=True)
        self.register_buffer("freqs_sin", sin, persistent=True)

    def forward(
        self,
        noise_embedding: torch.Tensor,  # [1, B, H]
        atten_mask: torch.Tensor,       # [1, B, Cc+AR+B]
        new_context: torch.Tensor,      # [1, AR, num_ctx_layers*H]
        context_pos: torch.Tensor,      # [1, AR]
        block_pos: torch.Tensor,        # [1, B]
        *past_kv,                       # k0..k{L-1}, v0..v{L-1}
    ):
        L = self.n_layers
        past_k = past_kv[:L]
        past_v = past_kv[L:]
        ctx_h = self.hidden_norm(self.fc(new_context))  # [1, AR, H]
        cos_ctx = self.freqs_cos[context_pos[0]]
        sin_ctx = self.freqs_sin[context_pos[0]]
        cos_blk = self.freqs_cos[block_pos[0]]
        sin_blk = self.freqs_sin[block_pos[0]]
        block_h = noise_embedding
        k_outs, v_outs = [], []
        for i, layer in enumerate(self.layers):
            block_h, k_new, v_new = layer(
                block_h, ctx_h, cos_ctx, sin_ctx, cos_blk, sin_blk,
                atten_mask, past_k[i], past_v[i],
            )
            k_outs.append(k_new)
            v_outs.append(v_new)
        h = self.norm(block_h)
        if self.lm_head is None:
            return (h, *k_outs, *v_outs)
        # Which rows carry a prediction depends on the checkpoint's convention; see
        # DFlashConfig.shifted_decode. Appended last so the runner's existing output
        # indices do not shift.
        #
        # Logits, not the argmax: QNN's OpArgmax gets the index wrong by +/-65536 on
        # an axis this wide (device-verified, deterministic, and it happily reports
        # aten.argmax.default as supported). The runner takes the max instead -- a few
        # x 151936 compares next to the 2.3 GMAC projection that just moved to HTP.
        rows = h if self.cfg.shifted_decode else h[:, 1:, :]
        return (h, *k_outs, *v_outs, self.lm_head(rows))

    # -- export helpers ---------------------------------------------------
    def get_example_inputs(self, dtype=torch.float32):
        B, H, AR = self.B, self.H, self.AR
        Lc = self.num_ctx_layers
        Cc, nKV, D = self.cache_len, self.nKV, self.D
        base = [
            torch.zeros(1, B, H, dtype=dtype),
            torch.zeros(1, B, Cc + AR + B, dtype=dtype),
            torch.zeros(1, AR, Lc * H, dtype=dtype),
            torch.zeros(1, AR, dtype=torch.int32),
            torch.zeros(1, B, dtype=torch.int32),
        ]
        past_k = [torch.zeros(1, nKV, D, Cc, dtype=dtype) for _ in range(self.n_layers)]
        past_v = [torch.zeros(1, nKV, Cc, D, dtype=dtype) for _ in range(self.n_layers)]
        return tuple(base + past_k + past_v)

    def get_metadata(self, prefill_ar: int) -> dict:
        # Scalar constant_methods only (QNN cannot serialize list constants).
        # The runtime recovers the context layers from the target's ordered
        # hidden outputs, so target_layer_ids need not be embedded here.
        return {
            "get_dflash_block_size": self.B,
            "get_dflash_hidden_size": self.H,
            "get_dflash_num_ctx_layers": self.num_ctx_layers,
            "get_dflash_num_draft_layers": self.n_layers,
            "get_dflash_max_context_len": self.max_context_len,
            "get_dflash_prefill_ar_len": prefill_ar,
            "get_dflash_mask_token_id": int(self.cfg.mask_token_id or 0),
            "get_dflash_shifted_decode": int(self.cfg.shifted_decode),
            "get_dflash_drop_sink": int(self.cfg.no_sink),
        }

    # -- weight loading ---------------------------------------------------
    def load_dflash_state_dict(self, sd: dict):
        """Map the DFlash checkpoint keys into this module (float upcast).

        Checkpoints that ship their own lm_head / embed_tokens carry those keys too;
        the draft graph does not use them, so they are simply not read here.
        """
        with torch.no_grad():
            # RMSNorm and RMSNorm is scale-invariant.
            self.fc.weight.copy_(sd["fc.weight"].float())
            if self.lm_head is not None:
                self.lm_head.weight.copy_(sd["lm_head.weight"].float())
            self.hidden_norm.weight.copy_(sd["hidden_norm.weight"].float())
            self.norm.weight.copy_(sd["norm.weight"].float())
            for i, layer in enumerate(self.layers):
                p = f"layers.{i}."
                a = layer.self_attn
                a.q_proj.weight.copy_(sd[p + "self_attn.q_proj.weight"].float())
                a.k_proj.weight.copy_(sd[p + "self_attn.k_proj.weight"].float())
                a.v_proj.weight.copy_(sd[p + "self_attn.v_proj.weight"].float())
                a.o_proj.weight.copy_(sd[p + "self_attn.o_proj.weight"].float())
                a.q_norm.weight.copy_(sd[p + "self_attn.q_norm.weight"].float())
                a.k_norm.weight.copy_(sd[p + "self_attn.k_norm.weight"].float())
                layer.input_layernorm.weight.copy_(sd[p + "input_layernorm.weight"].float())
                layer.post_attention_layernorm.weight.copy_(
                    sd[p + "post_attention_layernorm.weight"].float()
                )
                layer.mlp.gate_proj.weight.copy_(sd[p + "mlp.gate_proj.weight"].float())
                layer.mlp.up_proj.weight.copy_(sd[p + "mlp.up_proj.weight"].float())
                layer.mlp.down_proj.weight.copy_(sd[p + "mlp.down_proj.weight"].float())


def load_dflash_config(ckpt_dir: str) -> DFlashConfig:
    return DFlashConfig(json.load(open(os.path.join(ckpt_dir, "config.json"))))


def load_dflash_checkpoint(ckpt_dir: str) -> Tuple[dict, DFlashConfig]:
    from safetensors.torch import load_file

    cfg = load_dflash_config(ckpt_dir)
    st = os.path.join(ckpt_dir, "model.safetensors")
    sd = load_file(st) if os.path.exists(st) else torch.load(
        os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu"
    )
    return sd, cfg


# ---------------------------------------------------------------------------
# Compile pipeline (mirrors EagleManager / EagleHeadDualCompiler)
# ---------------------------------------------------------------------------
import argparse  # noqa: E402
import logging  # noqa: E402

from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (  # noqa: E402
    DFLASH_DRAFT,
    DFLASH_GRAPH_NAMES,
    DFLASH_TARGET,
)
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import (  # noqa: E402
    Component,
    log_info,
    Processor,
    Request,
)


def _check_hidden_capture_not_sharded(capture_ids, num_layers: int, num_sharding: int):
    """A captured hidden that sits on a graph-sharding boundary is emitted as a
    quantized uint16 IO (the shards exchange it without a dequant), and the
    runtime has no scale/zero_point to undo that. Everything still compiles and
    runs; the draft just silently reads garbage for that layer."""
    if num_sharding <= 1:
        return
    boundaries = set(range(0, num_layers, num_layers // num_sharding))
    clash = sorted(set(capture_ids) & boundaries)
    if not clash:
        return
    # get_split_graph_pass strides by num_layers // shares, so only divisors give
    # the requested number of shards.
    safe = [
        s
        for s in range(1, num_sharding + 1)
        if num_layers % s == 0
        and not (set(capture_ids) & set(range(0, num_layers, num_layers // s)))
    ]
    raise ValueError(
        f"DFlash hidden captures {clash} land on the num_sharding={num_sharding} "
        f"boundaries {sorted(boundaries)}; those outputs would come out quantized "
        f"(uint16) and the draft would read them as garbage. "
        f"Re-run with --num_sharding {safe[-1]}."
    )


def _export_dflash_cpu_pte(modules, example_inputs, constant_methods, out_path):
    """Export the draft as a portable-CPU pte (fp32, no QNN)."""
    from executorch.exir import EdgeCompileConfig, to_edge
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

    programs = {
        name: torch.export.export(m, inp, strict=True)
        for (name, m), inp in zip(modules.items(), example_inputs)
    }
    edge = to_edge(
        programs,
        constant_methods=constant_methods,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    exec_prog = edge.to_executorch(
        ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False, alloc_graph_output=False
            ),
        )
    )
    with open(out_path, "wb") as f:
        exec_prog.write_to_file(f)


_MASK_NEG = -65504.0  # most-negative finite fp16, the runner's additive mask fill


def _draft_atten_mask(B, Cc, AR, n_past, n_new):
    """Additive draft mask [1,B,Cc+AR+B], run_draft layout (dflash_token_generator
    write_mask): all B block rows share one mask (non-causal block). visible=0,
    masked=_MASK_NEG. cached-context col visible iff col<n_past; new-context slot
    visible iff (col-Cc)<n_new; block slots (col>=Cc+AR) always visible."""
    W = Cc + AR + B
    row = torch.full((W,), _MASK_NEG)
    row[:n_past] = 0.0
    row[Cc : Cc + n_new] = 0.0
    row[Cc + AR :] = 0.0
    return row.view(1, 1, W).expand(1, B, W).contiguous()


def _stamp_input_encoding(model, ph_idx, qparams, what):
    """Force the encoding of placeholder `ph_idx`'s quantize/dequantize pair.

    The mirror of HybridTextDecoder._stamp_tail_output_encoding, for an INPUT rather
    than an output. Runs after _encoding_override, which would otherwise give the
    prefill graph decode's (sink-free) range for a tensor that does carry the sink
    during seeding.
    """
    import logging

    phs = [n for n in model.graph.nodes if n.op == "placeholder"]
    q_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
    dq_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    scale, zp = qparams
    patched = 0
    for user in list(phs[ph_idx].users):
        if user.target is q_op:
            user.args = (user.args[0], scale, zp, *user.args[3:])
            patched += 1
            for dq in list(user.users):
                if dq.target is dq_op:
                    dq.args = (dq.args[0], scale, zp, *dq.args[3:])
    model.recompile()
    logging.info(
        "sink_split: stamped prefill encoding onto %s (patched=%d scale=%.8f zp=%d)",
        what, patched, scale, zp,
    )
    if patched == 0:
        raise RuntimeError(f"sink_split found no quantize node on {what}")


class DFlashCalibrator:
    """Drives the draft inside the target's calibration loop, so both graphs'
    observers see the traffic they actually get at inference.

    The previous path (`_dump_target_hidden`) hand-rolled a single degenerate
    forward instead of reusing kv_inference: it read the target's PREFILL graph,
    whose activations at that point are calibrated on `torch.randn` and are only
    overwritten by `_encoding_override` later, in compile. The draft therefore
    learned a new_context range of +3369 for hidden that really reaches +15852 --
    4.7x off -- from 13 rows of one prompt, with an empty KV cache and pos_ids all
    zero.

    This mirrors dflash_token_generator.cpp instead. Three details that path got
    wrong and that only agreeing with the runner fixes:
      - run_draft takes TWO independent position bases; context rows RoPE at
        ctx_pos_base + j while the block sits at block_pos_base + k.
      - the seeding block is all mask tokens (flush_stage), not a real token.
      - the decode block leads with the last committed token, i.e. the previous
        round's bonus -- not the token at the head of the appended rows.

    Owns the draft's context K/V; block K/V are transient by construction (the
    graph does not emit them).
    """

    def __init__(self, draft, cfg, tok_embedding, lm_head, max_context_len, ar,
                 head_ar, token_dtype, tree=False):
        self.draft = draft
        self.cfg = cfg
        self.tok_embedding = tok_embedding
        self.lm_head = lm_head
        self.ar = ar
        # Width of the emb/lm_head graphs the draft calls. With the draft-width pair
        # compiled this is block_size, so the block fills the graph exactly and the
        # observers see no padding; without it the draft borrows the target's decode
        # graphs and everything past block_size is zero-padded.
        self.head_ar = head_ar
        self.token_dtype = token_dtype
        # Whether the target verifies a tree. The draft is unaffected either way --
        # it only ever sees the accepted chain -- but the target's observers must
        # see the tree-shaped mask and repeated positions it will get at inference.
        self.tree = tree
        self.B = self.block_size = cfg.block_size
        self.no_sink = cfg.no_sink
        self.sink_split = cfg.sink_split
        # --no_sink withholds the row from the draft entirely; --sink_split feeds it
        # and separates the encodings by phase instead.
        self.drop_sink_row = cfg.no_sink
        self.prefill_qparams = None        # target's 5 captured, banked at the boundary
        self.draft_prefill_qparams = None  # the draft's new_context, same boundary
        self.n_draft = self.B if cfg.shifted_decode else self.B - 1
        self.hidden_row0 = 0 if cfg.shifted_decode else 1
        self.Cc = max_context_len - ar
        L, nKV, D = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
        self.past_k = [torch.zeros(1, nKV, D, self.Cc) for _ in range(L)]
        self.past_v = [torch.zeros(1, nKV, self.Cc, D) for _ in range(L)]
        self.ctx_len = 0
        self.rounds = 0
        self.accepted_total = 0
        # First block's worth of (rows, committed) kept for the QDQ-vs-fp32 verify,
        # which used to re-derive them from the dumped hidden.
        self.verify_sample = None

    def new_context_observer(self):
        """The observer on the draft's new_context input.

        Placeholder index 2 -- the draft graph takes
        (noise, atten_mask, new_context, context_pos, block_pos, past_k..., past_v...),
        see DFlashDraftModel.get_example_inputs. prepare_pt2e inserts the observer as
        that placeholder's call_module user.
        """
        phs = [n for n in self.draft.graph.nodes if n.op == "placeholder"]
        for user in phs[2].users:
            if user.op == "call_module":
                obs = getattr(self.draft, user.target, None)
                if obs is not None:
                    return obs
        raise RuntimeError("draft new_context observer not found")

    def bank_draft_prefill_qparams(self):
        """Bank + reset the draft's new_context encoding at the prompt/generation
        boundary, mirroring what kv_inference does for the target's captured five."""
        if not self.sink_split:
            return
        from executorch.examples.qualcomm.oss_scripts.llama.decoder_utils import (
            _bank_and_reset,
        )
        self.draft_prefill_qparams = _bank_and_reset([self.new_context_observer()])[0]

    def can_append(self, n_new: int) -> bool:
        return self.ctx_len + n_new <= self.Cc

    def _forward(self, block_tokens, rows, n_new, ctx_pos_base, block_pos_base):
        cfg = self.cfg
        B, ar = self.B, self.ar
        ids = torch.zeros(1, self.head_ar, dtype=self.token_dtype)
        ids[0, :B] = torch.tensor(block_tokens, dtype=self.token_dtype)
        noise = self.tok_embedding(ids)[:, :B, :]
        atten = _draft_atten_mask(B, self.Cc, ar, self.ctx_len, n_new)
        new_ctx = torch.zeros(1, ar, len(cfg.target_layer_ids) * cfg.hidden_size)
        if n_new:
            new_ctx[0, :n_new] = rows
        ctx_pos = (ctx_pos_base + torch.arange(ar)).view(1, ar).to(torch.int32)
        blk_pos = (block_pos_base + torch.arange(B)).view(1, B).to(torch.int32)
        inp = (noise, atten, new_ctx, ctx_pos, blk_pos)
        out = self.draft(*inp, *self.past_k, *self.past_v)
        L = cfg.num_hidden_layers
        k_new, v_new = out[1 : 1 + L], out[1 + L : 1 + 2 * L]
        if n_new:
            for i in range(L):
                self.past_k[i][:, :, :, self.ctx_len : self.ctx_len + n_new] = k_new[
                    i
                ][:, :, :, :n_new]
                self.past_v[i][:, :, self.ctx_len : self.ctx_len + n_new, :] = v_new[
                    i
                ][:, :, :n_new, :]
            self.ctx_len += n_new
        if self.verify_sample is None and n_new:
            # Taken while the cache was still empty, so the verify can replay it
            # against fresh zero caches. The inputs are never mutated after this.
            self.verify_sample = inp
        return out[0]

    def seed(self, rows, n_new, ctx_pos_base):
        """One prompt chunk: append its context rows, block half unused."""
        block = [self.cfg.mask_token_id or 0] * self.B
        self._forward(block, rows, n_new, ctx_pos_base, ctx_pos_base + n_new)

    def draft_block(self, committed, rows, n_new, ctx_pos_base, cur_pos, topk=1):
        """One speculative round: append last round's accepted rows, then draft.

        Returns each drafted depth's top-k as (log-probs, ids). k=1 is the chain.
        Log-probs rather than logits because the tree compares paths of different
        depths; the lm_head here is fake-quantized but still float, so this can
        normalize over the full vocab instead of the runner's top-k approximation.
        """
        block = [self.cfg.mask_token_id or 0] * self.B
        block[0] = committed
        hidden = self._forward(block, rows, n_new, ctx_pos_base, cur_pos)
        padded = torch.zeros(1, self.head_ar, self.cfg.hidden_size)
        padded[0, : self.B] = hidden[0]
        logits = self.lm_head(padded)
        self.rounds += 1
        rows_out = logits[0, self.hidden_row0 : self.hidden_row0 + self.n_draft]
        return rows_out.float().log_softmax(-1).topk(topk, dim=-1)


class DFlashDraftCompiler(Component):
    """Lowers {kv_forward, prefill_forward} of the DFlash draft into one pte.

    The two graphs differ only in the context-append length AR; QNN shares their
    weights, so the wide `fc` (59M params on Qwen3-4B) is stored once.
    """

    @log_info
    def __init__(
        self,
        control_args: argparse.Namespace,
        config,
        ckpt_dir: str,
        max_context_len: int,
        prefill_ar: int,
        share_lm_head: bool = False,
    ):
        self.control_args = control_args
        self.config = config
        sd, cfg = load_dflash_checkpoint(ckpt_dir)
        cfg.no_sink = control_args.no_sink
        cfg.sink_split = getattr(control_args, "sink_split", False)
        self.dflash_cfg = cfg
        self.prefill_ar = prefill_ar

        self.backend = getattr(control_args, "dflash_draft_backend", "htp")
        # PTQ needs an fp32 graph to export/prepare. For the no-quant path fp32 is
        # equally safe on HTP: no fp32 precision mode (only kHtpQuantized/kHtpFp16),
        # conv runs on HMX with no fp32 datapath -- an fp32 fc produced byte-identical
        # device output to the fp16 one. RMSNorm fuses into the QNN native OpRmsNorm
        #.
        dtype = torch.float32

        # Shared architecture: the draft drops any in-graph lm_head and shares the
        # target's lm_head.pte, outputting hidden only. Even checkpoints that ship
        # lm_head.weight (block7) get it stripped here. Without sharing, a checkpoint
        # that carries lm_head.weight keeps the vocab projection in-graph (HTP) and
        # older ones leave it to the runner's host scan of the target embedding table.
        self.with_lm_head = (not share_lm_head) and "lm_head.weight" in sd
        log_msg = (
            "in-graph (HTP)"
            if self.with_lm_head
            else ("shared lm_head.pte" if share_lm_head else "host (runner scan)")
        )
        logging.info("DFlash draft lm_head: %s", log_msg)

        def build(ar_len):
            m = DFlashDraftModel(cfg, max_context_len, ar_len, self.with_lm_head)
            m.load_dflash_state_dict(sd)
            m.eval()
            if self.backend == "htp":
                # HTP prefers conv2d over aten.linear (matches the TextDecoder /
                # EAGLE head convention). The draft's post-projection reshapes use
                # .reshape to tolerate the conv2d layout.
                from executorch.backends.qualcomm.utils.utils import (
                    convert_linear_to_conv2d,
                )

                m = convert_linear_to_conv2d(m)
            return m.to(dtype)

        # kv_forward's AR is how many context rows one round can append, which is
        # how many tokens that round commits: accepted + 1. The shifted convention
        # can accept all B drafted tokens, so it commits up to B+1; the aligned one
        # drafts B-1 and commits at most B. Size the graph for the worst case or the
        # extra row is silently dropped and the draft's cache desyncs for good.
        decode_ar = cfg.block_size + 1 if cfg.shifted_decode else cfg.block_size
        self.decode = build(decode_ar)
        self.prefill = build(prefill_ar)
        self.decode_input = self.decode.get_example_inputs(dtype=dtype)
        self.prefill_input = self.prefill.get_example_inputs(dtype=dtype)
        self.meta = DFlashDraftModel.get_metadata(self.decode, prefill_ar)

        from executorch.backends.qualcomm._passes.qnn_pass_manager import (
            get_capture_program_passes,
        )
        from executorch.backends.qualcomm._passes.utils import (
            get_passes_dependency_for_capture_program,
        )

        self.passes_job = get_capture_program_passes()
        self.dep_table = get_passes_dependency_for_capture_program()

        # Set by DFlashManager so draft calibration can drive the quantized target.
        # embed_weight is the tied target embedding (Qwen3-4B: target.embed ==
        # draft.embed, bit-identical); used to build the block noise_embedding during
        # calibration. The draft graph itself has no embedding.
        self.max_context_len = max_context_len
        self.target = None
        # Checkpoints with tie_word_embeddings (Qwen3-4B-DFlash-b16) ship no
        # embed_tokens: the table IS the target's. Resolved lazily from the target,
        # which DFlashManager wires in after this constructor returns.
        self._embed_weight = (
            sd["embed_tokens.weight"].float()
            if "embed_tokens.weight" in sd
            else None
        )
        # Set by prepare_graphs/make_calibrator, which DFlashManager runs before the
        # target calibrates so the draft can ride along in the same loop.
        self.prepared = False
        self.calibrator = None
        self.fp_ref = None

    @property
    def embed_weight(self):
        if self._embed_weight is None:
            emb = self.target.prefill.tok_embedding.input_embedding_module
            self._embed_weight = emb.weight.detach().float()
        return self._embed_weight

    @log_info
    def prepare_graphs(self, data):
        """prepare_pt2e both draft graphs BEFORE the target calibrates.

        Only example inputs are needed to prepare, so the draft can be observed
        inside the target's own kv_inference loop instead of being calibrated
        afterwards on a hand-rolled forward. Returns nothing; `self.decode` and
        `self.prefill` become observed GraphModules and `self.fp_ref` keeps the
        fp32 originals for the QDQ-vs-fp32 verify.
        """
        if data.skip_quantize:
            return  # draft runs fp16, no PTQ

        from executorch.backends.qualcomm.export_utils import make_quantizer
        from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
        from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
            DFlashDraftQuantRecipe,
        )
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e

        self.fp_ref = {}
        for name, ex in (("decode", self.decode_input), ("prefill", self.prefill_input)):
            model = getattr(self, name).float()
            self.fp_ref[name] = model
            quantizer = make_quantizer(
                quant_dtype=QuantDtype.use_16a4w,
                per_channel_conv=True,
                per_channel_linear=True,
                backend=data.backend,
                soc_model=data.soc_model,
            )
            quantizer.set_recipe(DFlashDraftQuantRecipe().recipe)
            setattr(
                self,
                name,
                prepare_pt2e(torch.export.export(model, ex).module(), quantizer),
            )
        self.prepared = True

    def make_calibrator(self, tok_embedding, lm_head, head_ar, token_dtype, tree=False):
        """Bind the prepared decode graph to the emb/lm_head graphs the draft calls,
        for the joint loop. Only the decode graph is driven: prefill's encodings are
        overwritten from decode in `quantize`, exactly as the target does."""
        if not self.prepared:
            return None
        self.calibrator = DFlashCalibrator(
            self.decode,
            self.dflash_cfg,
            tok_embedding,
            lm_head,
            self.max_context_len,
            self.decode_input[2].shape[1],
            head_ar,
            token_dtype,
            tree,
        )
        return self.calibrator

    @log_info
    def quantize(self, request: Request):
        data = request.method_data[DFLASH_DRAFT]
        if data.skip_quantize:
            return  # draft runs fp16, no PTQ
        if self.calibrator is None or self.calibrator.rounds == 0:
            raise RuntimeError(
                "DFlash draft was never driven by the target's calibration loop; "
                "DFlashManager must call prepare_graphs/make_calibrator first"
            )

        from executorch.examples.qualcomm.oss_scripts.llama.wrappers.llm_wrappers import (
            HybridTextDecoder,
        )
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e

        # The decode graph's observers were filled by the joint loop. prefill was
        # never driven -- like the target's, its encodings are copied from decode
        # below -- but convert_pt2e still needs one forward to clear the affine
        # observer, so give it the export sample.
        with torch.no_grad():
            self.prefill(*self.prefill_input)
        self.decode = convert_pt2e(self.decode)
        self.prefill = convert_pt2e(self.prefill)
        # prefill seeds the context cache that decode then appends to, through one
        # shared draft_kv_manager_ (dflash_token_generator.h) whose K/V cross the pte
        # boundary as uint8. Two independently calibrated scales there would make
        # decode misread every row prefill wrote.
        HybridTextDecoder._encoding_override(
            decode_model=self.decode, prefill_model=self.prefill
        )
        # Same exception as the target's captured hiddens: new_context is an f32
        # INPUT, not shared state, so the two graphs may carry different ranges --
        # prefill's covers the sink, decode's was banked fresh from generation rows.
        # The KV encodings above must stay identical and are untouched.
        qp = getattr(self.calibrator, "draft_prefill_qparams", None)
        if qp is not None:
            _stamp_input_encoding(self.prefill, 2, qp, "draft new_context")
        logging.info(
            "DFlash draft quantized over %d speculative rounds "
            "(16a4w; fc/down_proj/lm_head 16a8w protected)",
            self.calibrator.rounds,
        )

        # Dump the decode (block) QDQ graph so the host accept loop can drive the
        # draft on CPU (QDQ == pte). Same op namespace as the decoder QDQ; load with
        # `import torch.ao.quantization.fx._decomposed` first.
        if getattr(self.control_args, "dump_qdq", False):
            ep = torch.export.export(self.decode, self.decode_input, strict=True)
            path = f"{self.control_args.artifact}/qdq_draft.pt2"
            torch.export.save(ep, path)
            # new_context AR dim (self.decode is now a GraphModule, no .AR attribute)
            logging.info(
                "dumped QDQ qdq_draft (ar=%d) -> %s",
                self.decode_input[2].shape[1],
                path,
            )

        # Verify the EXACT QDQ decode graph (the one lowered into the pte) against
        # the fp32 draft on the real dumped target hidden: same drafted tokens =>
        # no accept loss from draft quantization. QDQ == the device pte numerically.
        if getattr(self.control_args, "verify_split", False):
            self._verify_draft_quant()

    def _verify_draft_quant(self):
        """Compare the EXACT QDQ decode graph (lowered into the pte) against the fp32
        draft on one real block from the joint loop, and report the drafted-token
        argmax (top1) + logit error. With an in-graph lm_head the last output is block
        logits; top1==1 means the quantized draft drafts identical tokens => no accept
        loss from quantization."""
        import torch.nn.functional as F

        inp = self.calibrator.verify_sample
        if inp is None:
            return
        cfg = self.dflash_cfg
        nKV, D, L = cfg.num_key_value_heads, cfg.head_dim, cfg.num_hidden_layers
        Cc = self.calibrator.Cc
        # The sample is the loop's first block, taken when the cache was still empty.
        past_k = [torch.zeros(1, nKV, D, Cc) for _ in range(L)]
        past_v = [torch.zeros(1, nKV, Cc, D) for _ in range(L)]
        idx = -1 if self.with_lm_head else 0
        with torch.no_grad():
            fp_o = self.fp_ref["decode"](*inp, *past_k, *past_v)[idx]
            q_o = self.decode(*inp, *past_k, *past_v)[idx]
        top1 = float("nan")
        if self.with_lm_head:
            top1 = (fp_o.argmax(-1) == q_o.argmax(-1)).float().mean().item()
        logging.info(
            "DFlash draft QDQ-vs-fp32 verify (block %s): top1=%.4f cos=%.4f mae=%.6f",
            "logits" if self.with_lm_head else "hidden",
            top1,
            F.cosine_similarity(fp_o.flatten(), q_o.flatten(), dim=0).item(),
            (fp_o - q_o).abs().mean().item(),
        )

    @log_info
    def compile(self, request: Request):
        data = request.method_data[DFLASH_DRAFT]
        out_path = f"{self.control_args.artifact}/{data.pte_filename}.pte"
        graph_names = list(DFLASH_GRAPH_NAMES)
        modules = [self.decode, self.prefill]
        inputs = [self.decode_input, self.prefill_input]

        if self.backend == "cpu":
            _export_dflash_cpu_pte(
                dict(zip(graph_names, modules)), inputs, {**self.meta}, out_path
            )
            logging.info("DFlash CPU fp32 draft pte written to %s", out_path)
            return

        from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo
        from executorch.backends.qualcomm.utils.utils import (
            to_edge_transform_and_lower_to_qnn,
        )
        from executorch.exir.capture._config import ExecutorchBackendConfig
        from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

        compile_spec = data.compile_spec
        if not isinstance(compile_spec, list):
            compile_spec = [compile_spec, compile_spec]
        elif len(compile_spec) == 1:
            compile_spec = [compile_spec[0], compile_spec[0]]

        # Expose the K/V caches at the pte boundary as 8-bit instead of letting the
        # delegate dequantize them back to f32. The encoding already exists --
        # _calibrate_draft steps through the sequence writing k_new/v_new back into
        # past_k/past_v, so the observers saw the real accumulated cache -- it was
        # simply never surfaced. The draft's K/V are the only 4-D tensors in the
        # graph (noise/mask/new_context/hidden are 2-D or 3-D), so rank identifies
        # them. Cuts draft KV traffic 4x (39.7MB -> ~9.9MB), which is what actually
        # bounds draft.pte: 5 layers cost 27.4ms vs the 36-layer target's 41.6ms,
        # i.e. 4.7x per layer, tracking its 4x larger per-layer KV footprint.
        from executorch.backends.qualcomm._passes import TagQuantIO
        from executorch.backends.qualcomm.builders.utils import is_graph_output
        from executorch.backends.qualcomm.utils.constants import (
            QCOM_PASS_ACTIVATE_KEY,
            QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
        )

        def _tag_draft_kv(node, kv_dtype=torch.uint8):
            if node.op != "placeholder" and not is_graph_output(node):
                return None
            val = node.meta.get("val", None)
            # Multi-output nodes carry a tuple/list of FakeTensors (no .size());
            # the K/V IO never does, so skip anything that is not a lone tensor.
            if val is None or isinstance(val, (tuple, list)):
                return None
            size_fn = getattr(val, "size", None)
            if size_fn is None:
                return None
            return kv_dtype if len(size_fn()) == 4 else None

        self.passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        self.passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
            "get_quant_io_dtype_fn"
        ] = _tag_draft_kv

        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=dict(zip(graph_names, modules)),
            inputs=dict(zip(graph_names, inputs)),
            compiler_specs=dict(zip(graph_names, compile_spec)),
            constant_methods={**self.meta},
            dep_table={n: self.dep_table for n in graph_names},
            passes_job={n: self.passes_job for n in graph_names},
            skip_node_op_set={"llama.fallback.default"},
        )
        exec_prog_mgr = edge_prog_mgr.to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                ),
                passes=[BuildQuantIo()],
            )
        )
        with open(out_path, "wb") as f:
            exec_prog_mgr.write_to_file(f)
        logging.info("DFlash draft pte written to %s", out_path)


class DFlashManager(Component):
    """Compiles the target (with selected-layer hidden export) + the DFlash draft."""

    @log_info
    def __init__(self, control_args: argparse.Namespace, config):
        from executorch.examples.qualcomm.oss_scripts.llama.wrappers.llm_wrappers import (
            HybridTextDecoder,
        )

        ckpt = control_args.dflash_draft_checkpoint
        cfg = load_dflash_config(ckpt)
        cfg.no_sink = control_args.no_sink
        cfg.sink_split = getattr(control_args, "sink_split", False)
        max_ctx = (
            getattr(control_args, "dflash_max_context_len", 0)
            or control_args.max_context_len
        )
        prefill_ar = (
            getattr(control_args, "dflash_prefill_ar_len", 0)
            or control_args.prefill_ar_len
        )
        # `fc` sees prefill_ar rows of the 5H context at once; that tensor is the
        # widest thing in the graph and must fit HTP's ~8MB VTCM after the
        # channels-first transpose convert_linear_to_conv2d inserts.
        ctx_bytes = prefill_ar * len(cfg.target_layer_ids) * cfg.hidden_size * 2
        if ctx_bytes > 6 << 20:
            raise ValueError(
                f"dflash_prefill_ar_len={prefill_ar} makes the fc input "
                f"{ctx_bytes / (1 << 20):.1f}MB, which overflows HTP TCM. "
                f"Lower it (128 is safe for Qwen3-4B)."
            )

        # static_llama appends `hidden_states` BEFORE running layer `ind`, so a
        # capture at `ind` yields HF's hidden_states[ind] = the OUTPUT of layer
        # ind-1. The reference reads hidden_states[layer_id + 1], i.e. the output
        # of layer `layer_id`, so every capture index shifts by one.
        capture_ids = [lid + 1 for lid in cfg.target_layer_ids]
        _check_hidden_capture_not_sharded(
            capture_ids, cfg.num_target_layers, config.num_sharding
        )

        # Target is split into three shared pte's: token embedding (emb.pte), the
        # headless decoder (embeds -> final hidden + selected-layer hiddens), and the
        # vocab projection (lm_head.pte). The draft reuses emb.pte and lm_head.pte.
        self.target = HybridTextDecoder(
            control_args,
            config,
            apply_embedding=True,  # split token embedding into a shared emb.pte
            output_hidden_layers=capture_ids,
            apply_output=False,  # headless: lm_head moves to a shared lm_head.pte
            # Third emb/lm_head width for the draft: its noise block and its hidden
            # are exactly block_size rows, so borrowing the target's decode graphs
            # both wasted the padding and mixed the two models' statistics.
            draft_head_ar=cfg.block_size,
            # Only --no_sink narrows the captured encodings, and only then do they
            # need to be off the residual's quantization point.
            fork_captured_hiddens=cfg.no_sink or cfg.sink_split,
        )
        self.draft_compiler = DFlashDraftCompiler(
            control_args, config, ckpt, max_ctx, prefill_ar, share_lm_head=True
        )

        self.control_args = control_args
        self.config = config

        # Splice draft_compiler at the tail of target's worker chain.
        self.set_next(self.target)
        tail = self.target
        while tail._next_handler is not None:
            tail = tail._next_handler
        tail.set_next(self.draft_compiler)

        # Draft PTQ calibrates on the quantized target's real hidden (path B); the
        # draft runs after the target in the chain, so target is converted by then.
        self.draft_compiler.target = self.target
        # Tied-embedding drafts read the token table off the target. Resolve it now so
        # a wiring error surfaces here, not an hour later inside draft calibration.
        assert self.draft_compiler.embed_weight is not None

    def process(self, request: Request) -> Request:
        Processor.process(self, request)

    @log_info
    def quantize(
        self, calibration_data, skip_quantize_target, tokenizer, backend, soc_model
    ):
        from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
            AUDIO_ENCODER,
            TEXT_DECODER,
            TEXT_ENCODER,
            TOK_EMBEDDING,
            VISION_ENCODER,
        )

        cal = (
            calibration_data
            if isinstance(calibration_data, dict)
            else {TEXT_DECODER: calibration_data}
        )
        method_data = {
            TEXT_DECODER: Request.Data(
                calibration_data=Request.CalibrationData(
                    datasets=cal.get(TEXT_DECODER, [])
                ),
                skip_quantize=skip_quantize_target,
                tokenizer=tokenizer,
                backend=backend,
                soc_model=soc_model,
            ),
            AUDIO_ENCODER: Request.Data(calibration_data=Request.CalibrationData()),
            VISION_ENCODER: Request.Data(calibration_data=Request.CalibrationData()),
            TEXT_ENCODER: Request.Data(),
            TOK_EMBEDDING: Request.Data(),
            DFLASH_DRAFT: Request.Data(
                calibration_data=Request.CalibrationData(
                    datasets=cal.get(TEXT_DECODER, [])
                ),
                skip_quantize=getattr(self.control_args, "dflash_draft_no_ptq", False),
                tokenizer=tokenizer,
                backend=backend,
                soc_model=soc_model,
            ),
            DFLASH_TARGET: Request.Data(),
        }
        # Observe the draft BEFORE the target calibrates, so it can be driven inside
        # the target's own kv_inference loop: the target's decode graph is the only
        # one carrying real encodings (prefill's are copied in at compile), and the
        # draft consumes exactly its hidden. prepare_pt2e needs example inputs only.
        self.draft_compiler.prepare_graphs(method_data[DFLASH_DRAFT])
        self.target.decode.dflash_draft = self.draft_compiler
        self.process(Request(method_name="quantize", method_data=method_data))

    @log_info
    def compile(
        self,
        compile_spec,
        pte_filenames,
        skip_quantize_target=False,
        draft_compile_spec=None,
        tok_embedding_compile_spec=None,
        lm_head_compile_spec=None,
    ):
        from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
            AUDIO_ENCODER,
            LM_HEAD,
            TEXT_DECODER,
            TEXT_ENCODER,
            TOK_EMBEDDING,
            VISION_ENCODER,
        )

        # The draft needs its own spec: ConvertMhaToSha cannot handle its
        # non-causal block attention (see compile_dflash).
        draft_compile_spec = draft_compile_spec or compile_spec
        # emb.pte / lm_head.pte are weight-shared across their kv/prefill AR graphs;
        # HybridTextDecoder.compile lowers them when their spec + pte_filename are set.
        tok_embedding_compile_spec = tok_embedding_compile_spec or compile_spec
        lm_head_compile_spec = lm_head_compile_spec or compile_spec
        method_data = {
            TEXT_DECODER: Request.Data(
                compile_spec=compile_spec,
                pte_filename=pte_filenames[DFLASH_TARGET],
                skip_quantize=skip_quantize_target,
            ),
            DFLASH_DRAFT: Request.Data(
                compile_spec=draft_compile_spec,
                pte_filename=pte_filenames[DFLASH_DRAFT],
                skip_quantize=True,
            ),
            AUDIO_ENCODER: Request.Data(compile_spec=compile_spec, pte_filename="unused"),
            VISION_ENCODER: Request.Data(compile_spec=compile_spec, pte_filename="unused"),
            TEXT_ENCODER: Request.Data(compile_spec=compile_spec, pte_filename="unused"),
            TOK_EMBEDDING: Request.Data(
                compile_spec=tok_embedding_compile_spec,
                pte_filename=pte_filenames[TOK_EMBEDDING],
            ),
            LM_HEAD: Request.Data(
                compile_spec=lm_head_compile_spec,
                pte_filename=pte_filenames[LM_HEAD],
            ),
            DFLASH_TARGET: Request.Data(
                compile_spec=compile_spec, pte_filename=pte_filenames[DFLASH_TARGET]
            ),
        }
        self.process(Request(method_name="compile", method_data=method_data))
