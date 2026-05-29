# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""EAGLE-3 head wrapper for QNN compilation.

The head architecture (per checkpoint at /mnt/hdd-ws/users/zqchen/study/executorch/eagle):

    fc.weight                         (H,   3H)        # only used in PREFILL
    midlayer.input_layernorm.weight   (H,)             # applied to embedding
    midlayer.hidden_norm.weight       (H,)             # applied to prev_feature
    midlayer.self_attn.q_proj.weight  (n_heads*hd,  2H)
    midlayer.self_attn.k_proj.weight  (n_kv*hd,     2H)
    midlayer.self_attn.v_proj.weight  (n_kv*hd,     2H)
    midlayer.self_attn.o_proj.weight  (H,           n_heads*hd)
    midlayer.post_attention_layernorm.weight (H,)
    midlayer.mlp.gate_proj.weight     (intermediate, H)
    midlayer.mlp.up_proj.weight       (intermediate, H)
    midlayer.mlp.down_proj.weight     (H, intermediate)
    norm.weight                       (H,)
    lm_head.weight                    (draft_vocab, H)

    d2t (draft_vocab, int64)   target_id = draft_id + d2t[draft_id]
    t2d (target_vocab, bool)

The forward path (DECODE / kv_forward):

    e   = input_layernorm(emb)              # [B,1,H]
    h_p = hidden_norm(prev_feature)         # [B,1,H]
    x   = cat([e, h_p], dim=-1)             # [B,1,2H]
    qkv = q/k/v_proj(x)                     # GQA, RoPE
    a   = self_attn(qkv, kv_cache)          # [B,1,H]
    a   = prev_feature + a                  # residual on prev_feature
    a   = a + mlp(post_attention_layernorm(a))   # SwiGLU
    out_a = norm(a)                         # final RMS, fed to lm_head & to next step
    logits_draft = lm_head(out_a)           # [B,1,draft_vocab]

The PREFILL path differs only in: prev_feature := fc(cat(h_low, h_mid, h_high, -1)).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm._passes.utils import (
    get_passes_dependency_for_capture_program,
)
from executorch.backends.qualcomm.utils.utils import (
    convert_linear_to_conv2d,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.examples.qualcomm.oss_scripts.llama import LLMModelConfig
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    EAGLE_HEAD,
    EAGLE_TARGET,
)
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import (
    Component,
    log_info,
    Mode,
    Processor,
    Request,
)
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.llm_wrappers import (
    HybridTextDecoder,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

# SafeAILab official EAGLE-3 checkpoint key conventions
EAGLE_CKPT_KEYS = {
    "fc",
    "midlayer.self_attn.q_proj.weight",
    "midlayer.self_attn.k_proj.weight",
    "midlayer.self_attn.v_proj.weight",
    "midlayer.self_attn.o_proj.weight",
    "midlayer.mlp.gate_proj.weight",
    "midlayer.mlp.up_proj.weight",
    "midlayer.mlp.down_proj.weight",
    "midlayer.input_layernorm.weight",
    "midlayer.hidden_norm.weight",
    "midlayer.post_attention_layernorm.weight",
    "norm.weight",
    "lm_head.weight",
    "d2t",
    "t2d",
}


def load_eagle_head_state_dict(ckpt_path: str) -> dict:
    """Load EAGLE head weights from a directory or a file.

    Supports `.bin` (torch.load) and `.safetensors`. If `ckpt_path` is a
    directory, looks for `pytorch_model.bin` then `model.safetensors`.
    """
    if os.path.isdir(ckpt_path):
        bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
        st_path = os.path.join(ckpt_path, "model.safetensors")
        if os.path.exists(bin_path):
            ckpt_path = bin_path
        elif os.path.exists(st_path):
            ckpt_path = st_path
        else:
            raise FileNotFoundError(
                f"No pytorch_model.bin or model.safetensors found in {ckpt_path}"
            )

    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(ckpt_path)
    else:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)

    return sd


def load_eagle_head_config(ckpt_path: str, override_path: Optional[str] = None) -> dict:
    import json

    if override_path is not None:
        cfg_path = override_path
    elif os.path.isdir(ckpt_path):
        cfg_path = os.path.join(ckpt_path, "config.json")
    else:
        cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")

    with open(cfg_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# RoPE / RMSNorm helpers (Qwen3-style; HF rope, head_dim halved for cos/sin)
# ---------------------------------------------------------------------------


def _hf_precompute_rope(
    head_dim: int, max_pos: int, theta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HuggingFace-style RoPE precompute.

    Produces cos / sin of shape ``[max_pos, head_dim]`` with the
    "duplicate-and-concat" layout (matches transformers' LlamaRotaryEmbedding):

        inv_freq = 1 / (theta ** (arange(0, dim, 2) / dim))     # [dim/2]
        freqs    = outer(t, inv_freq)                           # [max_pos, dim/2]
        emb      = concat([freqs, freqs], -1)                   # [max_pos, dim]

    Used together with ``_apply_rope_hf`` below.
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                            # [max_pos, dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)                     # [max_pos, dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF-style rotate_half: cat([-x2, x1], -1) where x = [x1 | x2]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HuggingFace-style RoPE apply.

    Equivalent to ``q * cos + rotate_half(q) * sin``. Inputs:
      * ``x``   : [..., head_dim]
      * ``cos`` / ``sin`` : [..., head_dim]   (NOT head_dim/2; pre-duplicated)
    """
    return x * cos + _rotate_half(x) * sin



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.weight.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# EagleHead nn.Module
# ---------------------------------------------------------------------------


class EagleHeadConfig:
    """Distilled config for EagleHead, derived from EAGLE checkpoint config.json."""

    def __init__(self, raw: dict):
        self.hidden_size: int = raw["hidden_size"]
        # If target and draft have different hidden sizes, the ckpt's
        # config carries `target_hidden_size`. EAGLE's fc collapses
        # 3 * target_hidden_size -> hidden_size.
        self.target_hidden_size: int = raw.get("target_hidden_size", self.hidden_size)
        self.num_attention_heads: int = raw["num_attention_heads"]
        self.num_key_value_heads: int = raw["num_key_value_heads"]
        self.head_dim: int = raw["head_dim"]
        self.intermediate_size: int = raw["intermediate_size"]
        self.draft_vocab_size: int = raw.get("draft_vocab_size", raw["vocab_size"])
        self.target_vocab_size: int = raw["vocab_size"]
        self.rope_theta: float = float(raw.get("rope_theta", 10000.0))
        self.rms_norm_eps: float = float(raw.get("rms_norm_eps", 1e-6))
        self.num_hidden_layers: int = raw.get("num_hidden_layers", 1)
        # EAGLE-3 always uses 3 hidden inputs in prefill (low/mid/high) — fc(3H→H).
        self.num_hidden_inputs: int = 3


class EagleAttention(nn.Module):
    """GQA attention with 2H input feature dim (cat(emb, prev_feature))."""

    def __init__(self, cfg: EagleHeadConfig, max_context_len: int):
        super().__init__()
        self.cfg = cfg
        self.max_context_len = max_context_len
        H = cfg.hidden_size
        self.q_proj = nn.Linear(
            2 * H, cfg.num_attention_heads * cfg.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            2 * H, cfg.num_key_value_heads * cfg.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            2 * H, cfg.num_key_value_heads * cfg.head_dim, bias=False
        )
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, H, bias=False)

    def forward(
        self,
        x: torch.Tensor,  # [B, S, 2H]
        freqs_cos: torch.Tensor,  # [S, head_dim]   (HF-style, pre-duplicated)
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,  # [B, S, ctx]
        k_cache: Optional[torch.Tensor],  # [B, n_kv, head_dim, ctx-S]
        v_cache: Optional[torch.Tensor],  # [B, n_kv, ctx-S, head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        B, S, _ = x.shape
        Hq = cfg.num_attention_heads
        Hk = cfg.num_key_value_heads
        D = cfg.head_dim

        q = self.q_proj(x).view(B, S, Hq, D).transpose(1, 2)  # [B, Hq, S, D]
        k = self.k_proj(x).view(B, S, Hk, D).transpose(1, 2)  # [B, Hk, S, D]
        v = self.v_proj(x).view(B, S, Hk, D).transpose(1, 2)

        # apply HF-style RoPE on q, k. cos/sin: [S, head_dim] -> broadcast to [1,1,S,D]
        cos = freqs_cos.view(1, 1, S, D)
        sin = freqs_sin.view(1, 1, S, D)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # k cache layout matches static_llama: [B, Hk, D, ctx-S], so transpose
        k_new = k.transpose(2, 3)  # [B, Hk, D, S]
        v_new = v  # [B, Hk, S, D]

        if k_cache is not None:
            k_full = torch.cat([k_cache, k_new], dim=-1)  # [B, Hk, D, ctx]
            v_full = torch.cat([v_cache, v_new], dim=-2)  # [B, Hk, ctx, D]
        else:
            k_full = k_new
            v_full = v_new

        # GQA: repeat kv to match Hq
        rep = Hq // Hk
        if rep > 1:
            k_full = k_full.repeat_interleave(rep, dim=1)
            v_full = v_full.repeat_interleave(rep, dim=1)

        # attention scores: q [B,Hq,S,D] @ k_full [B,Hq,D,ctx] -> [B,Hq,S,ctx]
        scores = torch.matmul(q, k_full) / math.sqrt(D)
        scores = scores + atten_mask.unsqueeze(1)
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v_full)  # [B,Hq,S,D]
        out = out.transpose(1, 2).contiguous().view(B, S, Hq * D)
        out = self.o_proj(out)  # [B, S, H]

        # return new k/v slices (only the S-newest positions, matching output_new_cache_only)
        return out, k_new, v_new


class EagleMLP(nn.Module):
    def __init__(self, cfg: EagleHeadConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class EagleHead(nn.Module):
    """Single-layer EAGLE-3 head.

    `mode` controls the forward signature:
      * Mode.PREFILL: forward(hidden_LMH, tok_emb, atten_mask, input_pos, *kv)
        — first applies fc(3H→H) to hidden_LMH, then runs the transformer.
      * Mode.DECODE:  forward(prev_feature, tok_emb, atten_mask, input_pos, *kv)
        — skips fc.
    """

    def __init__(
        self, cfg: EagleHeadConfig, mode: Mode, ar_len: int, max_context_len: int
    ):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.ar_len = ar_len
        self.max_context_len = max_context_len
        self.use_kv_cache = max_context_len != ar_len

        H = cfg.hidden_size

        # midlayer
        self.input_layernorm = RMSNorm(H, cfg.rms_norm_eps)
        self.hidden_norm = RMSNorm(H, cfg.rms_norm_eps)
        self.self_attn = EagleAttention(cfg, max_context_len)
        self.post_attention_layernorm = RMSNorm(H, cfg.rms_norm_eps)
        self.mlp = EagleMLP(cfg)

        # final
        self.norm = RMSNorm(H, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(H, cfg.draft_vocab_size, bias=False)

        # PREFILL only: fc(3 * target_H -> H). For Qwen3-1.7B target_H == H == 2048.
        if mode == Mode.PREFILL:
            self.fc = nn.Linear(
                cfg.num_hidden_inputs * cfg.target_hidden_size, H, bias=False
            )
        else:
            self.fc = None

        # rope buffers — HuggingFace style: cos/sin have shape [max_ctx, head_dim]
        # NOTE: persistent=True so torch.export's convert_constant_dim_order_pass
        # can find input specs for these lifted tensors.
        cos, sin = _hf_precompute_rope(cfg.head_dim, max_context_len, cfg.rope_theta)
        self.register_buffer("freqs_cos", cos, persistent=True)
        self.register_buffer("freqs_sin", sin, persistent=True)

        # d2t / t2d are saved as buffers but NOT used in forward (host-side at runtime)
        self.register_buffer(
            "d2t",
            torch.zeros(cfg.draft_vocab_size, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "t2d",
            torch.zeros(cfg.target_vocab_size, dtype=torch.bool),
            persistent=True,
        )

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(self, *inputs):
        if self.mode == Mode.PREFILL:
            return self._forward_prefill(*inputs)
        return self._forward_decode(*inputs)

    def _shared_block(
        self,
        prev_feature: torch.Tensor,  # [B, S, H]
        tok_emb: torch.Tensor,  # [B, S, H]
        atten_mask: torch.Tensor,
        input_pos: torch.Tensor,
        k_cache: Optional[torch.Tensor],
        v_cache: Optional[torch.Tensor],
    ):
        B, S, _ = tok_emb.shape

        # rope frequencies sliced by input_pos (mirrors static_llama)
        freqs_cos = (
            self.freqs_cos[input_pos][0] if self.use_kv_cache else self.freqs_cos
        )
        freqs_sin = (
            self.freqs_sin[input_pos][0] if self.use_kv_cache else self.freqs_sin
        )

        e = self.input_layernorm(tok_emb)
        h_p = self.hidden_norm(prev_feature)
        x = torch.cat([e, h_p], dim=-1)  # [B, S, 2H]

        attn_out, k_out, v_out = self.self_attn(
            x, freqs_cos, freqs_sin, atten_mask, k_cache, v_cache
        )
        h = prev_feature + attn_out  # 1st residual on prev_feature
        h = h + self.mlp(self.post_attention_layernorm(h))  # 2nd residual

        # IMPORTANT: `h` (post-MLP, pre-final-norm) is what gets fed back as
        # `prev_feature` in the next decode step — `hidden_norm` will norm it
        # again, so we must NOT pre-norm here. The final RMS is only applied
        # in the lm_head path, matching the official drafter where
        # `forward()` returns raw hidden and `_get_topk_tokens` does
        # `lm_head(self.norm(hidden))`.
        logits = self.lm_head(self.norm(h))  # [B, S, draft_vocab]
        return logits, h, k_out, v_out

    def _forward_prefill(
        self,
        hidden_LMH: torch.Tensor,  # [B, S, 3H]
        tok_emb: torch.Tensor,  # [B, S, H]
        atten_mask: torch.Tensor,
        input_pos: torch.Tensor,
        *kv_caches,
    ):
        prev_feature = self.fc(hidden_LMH)  # [B, S, H]
        k_cache = kv_caches[0] if self.use_kv_cache else None
        v_cache = kv_caches[1] if self.use_kv_cache else None
        return self._shared_block(
            prev_feature, tok_emb, atten_mask, input_pos, k_cache, v_cache
        )

    def _forward_decode(
        self,
        prev_feature: torch.Tensor,  # [B, S, H]
        tok_emb: torch.Tensor,  # [B, S, H]
        atten_mask: torch.Tensor,
        input_pos: torch.Tensor,
        *kv_caches,
    ):
        k_cache = kv_caches[0] if self.use_kv_cache else None
        v_cache = kv_caches[1] if self.use_kv_cache else None
        return self._shared_block(
            prev_feature, tok_emb, atten_mask, input_pos, k_cache, v_cache
        )

    # -----------------------------------------------------------------
    # Helpers expected by the QNN compile pipeline
    # -----------------------------------------------------------------
    def get_example_inputs(self):
        cfg = self.cfg
        H = cfg.hidden_size
        S = self.ar_len
        ctx = self.max_context_len

        if self.mode == Mode.PREFILL:
            # Prefill input is hidden_LMH = cat(low, mid, high) along feat dim
            prev_in = torch.zeros(
                (1, S, cfg.num_hidden_inputs * cfg.target_hidden_size),
                dtype=torch.float32,
            )
        else:
            prev_in = torch.zeros((1, S, H), dtype=torch.float32)
        tok_emb = torch.zeros((1, S, H), dtype=torch.float32)
        atten_mask = torch.zeros((1, S, ctx), dtype=torch.float32)
        input_pos = torch.zeros((1, S), dtype=torch.int32)

        if self.use_kv_cache:
            k_cache = torch.zeros(
                (1, cfg.num_key_value_heads, cfg.head_dim, ctx - S), dtype=torch.float32
            )
            v_cache = torch.zeros(
                (1, cfg.num_key_value_heads, ctx - S, cfg.head_dim), dtype=torch.float32
            )
            return (prev_in, tok_emb, [atten_mask], input_pos, [k_cache], [v_cache])

        return (prev_in, tok_emb, [atten_mask], input_pos)

    def get_metadata(self):
        cfg = self.cfg
        return {
            "get_eagle_ar_len": self.ar_len,
            "get_eagle_hidden_size": cfg.hidden_size,
            "get_eagle_n_kv_heads": cfg.num_key_value_heads,
            "get_eagle_n_heads": cfg.num_attention_heads,
            "get_eagle_head_dim": cfg.head_dim,
            "get_eagle_max_context_len": self.max_context_len,
            "get_eagle_draft_vocab_size": cfg.draft_vocab_size,
            "get_eagle_target_vocab_size": cfg.target_vocab_size,
            "get_eagle_n_layers": cfg.num_hidden_layers,
        }

    # -----------------------------------------------------------------
    # Weight loading from SafeAILab EAGLE-3 checkpoint
    # -----------------------------------------------------------------
    def load_eagle_state_dict(self, sd: dict):
        """Map the official ckpt key naming into our nn.Module."""
        with torch.no_grad():
            if self.fc is not None:
                # ckpt fc.weight shape (H, 3H) — matches nn.Linear(3H -> H)
                self.fc.weight.copy_(sd["fc.weight"].float())

            self.input_layernorm.weight.copy_(
                sd["midlayer.input_layernorm.weight"].float()
            )
            self.hidden_norm.weight.copy_(sd["midlayer.hidden_norm.weight"].float())
            self.post_attention_layernorm.weight.copy_(
                sd["midlayer.post_attention_layernorm.weight"].float()
            )

            self.self_attn.q_proj.weight.copy_(
                sd["midlayer.self_attn.q_proj.weight"].float()
            )
            self.self_attn.k_proj.weight.copy_(
                sd["midlayer.self_attn.k_proj.weight"].float()
            )
            self.self_attn.v_proj.weight.copy_(
                sd["midlayer.self_attn.v_proj.weight"].float()
            )
            self.self_attn.o_proj.weight.copy_(
                sd["midlayer.self_attn.o_proj.weight"].float()
            )

            self.mlp.gate_proj.weight.copy_(sd["midlayer.mlp.gate_proj.weight"].float())
            self.mlp.up_proj.weight.copy_(sd["midlayer.mlp.up_proj.weight"].float())
            self.mlp.down_proj.weight.copy_(sd["midlayer.mlp.down_proj.weight"].float())

            self.norm.weight.copy_(sd["norm.weight"].float())
            self.lm_head.weight.copy_(sd["lm_head.weight"].float())

            if "d2t" in sd:
                self.d2t.copy_(sd["d2t"].long())
            if "t2d" in sd:
                self.t2d.copy_(sd["t2d"].bool())


# ---------------------------------------------------------------------------
# EagleDecoderWrapper (Component) — mirrors TextDecoder, simplified
# ---------------------------------------------------------------------------


def parse_layer_indices(s: Optional[str], n_layers: int = 28) -> List[int]:
    """Parse `--eagle_layer_indices` like '1,14,27'. Default uniform sample."""
    if s is None or s == "":
        # Default: uniformly sample low/mid/high (Qwen3-1.7B has 28 layers)
        return [1, n_layers // 2, n_layers - 1]
    parts = [int(x.strip()) for x in s.split(",")]
    assert len(parts) == 3, f"--eagle_layer_indices needs 3 ints, got {parts}"
    return parts


class EagleDecoderWrapper(Component):
    """Compiles a single EAGLE head graph (PREFILL or DECODE).

    Phase 2 simplification: we run the head in fp32 and lower it to QNN HTP
    without per-layer quantization (skip_quantize=True path). True 16a8w
    quantization for the head is a follow-up (TODO).
    """

    @log_info
    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        head_cfg: EagleHeadConfig,
        mode: Mode,
        head_state_dict: dict,
    ):
        self.control_args = control_args
        self.config = config
        self.mode = mode
        self.head_cfg = head_cfg
        self.passes_job = get_capture_program_passes()
        self.dep_table = get_passes_dependency_for_capture_program()
        self.meta = {}

        # ar_len strategy (mirrors process_model_args):
        #   PREFILL: ar=1 (head prefill consumes a single hidden vector after target verify)
        #   DECODE:  ar=1 (chain head step)
        # NOTE: EAGLE head always processes 1 token per call. There is no need
        # to expose prefill_ar_len here.
        ar_len = 1
        max_ctx = control_args.max_context_len

        self.head = EagleHead(head_cfg, mode, ar_len, max_ctx)
        self.head.load_eagle_state_dict(head_state_dict)
        self.head.eval()

        # convert linear -> conv2d (matches TextDecoder convention for HTP perf)
        self.head = convert_linear_to_conv2d(self.head)

        if control_args.dtype_override == "fp16":
            self.head = self.head.to(torch.float16)

        self.example_input = self.head.get_example_inputs()
        # flatten example_input to torch.export.export-style tuple
        if self.head.use_kv_cache:
            prev_in, tok_emb, atten_masks, input_pos, k_caches, v_caches = (
                self.example_input
            )
            self.export_input = (
                prev_in,
                tok_emb,
                *atten_masks,
                input_pos,
                *k_caches,
                *v_caches,
            )
        else:
            prev_in, tok_emb, atten_masks, input_pos = self.example_input
            self.export_input = (prev_in, tok_emb, *atten_masks, input_pos)

        self.meta = self.head.get_metadata()

    # ------------------------------------------------------------------
    # Component.quantize / Component.compile take a Request object
    # ------------------------------------------------------------------
    @log_info
    def quantize(self, request: Request):  # noqa: D401
        """Phase 2: skip quantization (run head in fp32/fp16).

        TODO(phase-2-quant): port TextDecoder.quantize's prepare_pt2e/convert_pt2e
        flow here. Need to carefully tag KV/IO bit widths and avoid quantizing
        the prev_feature input when --eagle_hidden_io fp16.
        """
        data = request.method_data.get(EAGLE_HEAD)
        if data is not None and data.skip_quantize:
            return
        logging.warning(
            "EagleDecoderWrapper.quantize is a stub (Phase 2). Head will be "
            "lowered without PTQ — fp16 IO. Expect lower accept rate vs full "
            "quantized path."
        )

    @log_info
    def compile(self, request: Request):
        """No-op: EagleHeadDualCompiler handles the actual lowering for both
        DECODE and PREFILL graphs in a single .pte."""
        return


class EagleHeadDualCompiler(Component):
    """Compiles BOTH `prefill_forward` and `kv_forward` into a single pte.

    Mirrors the HybridTextDecoder pattern (which puts decode + prefill in one
    pte under DECODER_GRAPH_NAMES).
    """

    @log_info
    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        ckpt_path: str,
    ):
        self.control_args = control_args
        self.config = config

        sd = load_eagle_head_state_dict(ckpt_path)
        raw_cfg = load_eagle_head_config(ckpt_path, control_args.eagle_head_config)
        self.head_cfg = EagleHeadConfig(raw_cfg)

        self.decode = EagleDecoderWrapper(
            control_args, config, self.head_cfg, Mode.DECODE, sd
        )
        self.prefill = EagleDecoderWrapper(
            control_args, config, self.head_cfg, Mode.PREFILL, sd
        )
        # daisy chain so .quantize() / .compile() walks both
        self.set_next(self.decode).set_next(self.prefill)

    @log_info
    def quantize(self, request: Request):
        # delegate to children via Processor chain
        return

    @log_info
    def compile(self, request: Request):  # noqa: C901
        """Lower {kv_forward, prefill_forward} into a single pte file."""
        data = request.method_data[EAGLE_HEAD]
        compile_spec = data.compile_spec
        if not isinstance(compile_spec, list):
            compile_spec = [compile_spec, compile_spec]
        elif len(compile_spec) == 1:
            compile_spec = [compile_spec[0], compile_spec[0]]

        graph_names = ["kv_forward", "prefill_forward"]
        modules = [self.decode.head, self.prefill.head]
        inputs = [self.decode.export_input, self.prefill.export_input]

        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=dict(zip(graph_names, modules)),
            inputs=dict(zip(graph_names, inputs)),
            compiler_specs=dict(zip(graph_names, compile_spec)),
            # Use decode meta as canonical (both contain identical metadata)
            constant_methods={**self.decode.meta},
            dep_table=dict(
                zip(graph_names, [self.decode.dep_table, self.prefill.dep_table])
            ),
            passes_job=dict(
                zip(graph_names, [self.decode.passes_job, self.prefill.passes_job])
            ),
            skip_node_op_set={"llama.fallback.default"},
        )

        executorch_config = ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
            passes=[BuildQuantIo()],
        )
        exec_prog_mgr = edge_prog_mgr.to_executorch(executorch_config)

        out_path = f"{self.control_args.artifact}/{data.pte_filename}.pte"
        with open(out_path, "wb") as f:
            exec_prog_mgr.write_to_file(f)
        logging.info(f"Eagle head pte written to {out_path}")


# ---------------------------------------------------------------------------
# EagleManager — top-level Component that compiles target + head together
# ---------------------------------------------------------------------------


class EagleManager(Component):
    """Top-level manager for `--model_mode eagle`.

    Compiles two pte's:
      1. Target text decoder pte (with output_hidden_layers enabled)
      2. EAGLE head pte (kv_forward + prefill_forward)
    """

    @log_info
    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
    ):
        # Hidden-layer indices for target's MultiScopeAware/Llama forward.
        # Default: uniform over n_layers; override via --eagle_layer_indices.
        # NOTE: Qwen3-1.7B has 28 layers; we don't know other models' n_layers
        # at this moment without loading the params json. We just default to
        # [1, 14, 27]; user can override.
        layer_indices = parse_layer_indices(
            getattr(control_args, "eagle_layer_indices", None)
        )

        # Phase 2 first-pass: keep target's forward bit-identical to hybrid mode
        # (output_hidden_layers=None) so target compile uses the well-tested
        # path. Hidden export is a later optimization; for now Phase 3 runtime
        # will not have hidden_LMH from target — it must be computed offline
        # or the pipeline must fall back to standard verify (TODO).
        # TODO(phase-2): re-enable hidden export once we resolve the
        # "Missing input spec for lifted tensor freqs_cos" issue triggered by
        # the modified forward returning extra tensors.
        target_hidden_layers = None  # was: layer_indices

        # Target decoder: HybridTextDecoder with output_hidden_layers wired
        self.target = HybridTextDecoder(
            control_args,
            config,
            apply_embedding=False,
            output_hidden_layers=target_hidden_layers,
        )

        # Head dual-graph compiler
        self.head_compiler = EagleHeadDualCompiler(
            control_args, config, control_args.eagle_head_checkpoint
        )

        self.control_args = control_args
        self.config = config
        self._sub_components = [self.target, self.head_compiler]

        # daisy chain
        self.set_next(self.target).set_next(self.head_compiler)

    def process(self, request: Request) -> Request:
        Processor.process(self, request)

    @log_info
    def quantize(
        self,
        calibration_data,
        skip_quantize_target,
        tokenizer,
        backend,
        soc_model,
    ):
        """Drive target quantization. Head is fp16 (Phase 2)."""
        from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
            AUDIO_ENCODER,
            TEXT_DECODER,
            TEXT_ENCODER,
            TOK_EMBEDDING,
            VISION_ENCODER,
        )

        # Build a Request that satisfies HybridTextDecoder.quantize. It looks
        # up TEXT_DECODER / AUDIO_ENCODER / VISION_ENCODER calibration data.
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
            AUDIO_ENCODER: Request.Data(
                calibration_data=Request.CalibrationData(),
            ),
            VISION_ENCODER: Request.Data(
                calibration_data=Request.CalibrationData(),
            ),
            TEXT_ENCODER: Request.Data(),
            TOK_EMBEDDING: Request.Data(),
            EAGLE_HEAD: Request.Data(skip_quantize=True),
            EAGLE_TARGET: Request.Data(),
        }
        request = Request(method_name="quantize", method_data=method_data)
        self.process(request)

    @log_info
    def compile(
        self,
        compile_spec,
        pte_filenames,
        skip_quantize_target=False,
    ):
        from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
            AUDIO_ENCODER,
            TEXT_DECODER,
            TEXT_ENCODER,
            TOK_EMBEDDING,
            VISION_ENCODER,
        )

        method_data = {
            TEXT_DECODER: Request.Data(
                compile_spec=compile_spec,
                pte_filename=pte_filenames[EAGLE_TARGET],
                skip_quantize=skip_quantize_target,
            ),
            EAGLE_HEAD: Request.Data(
                compile_spec=compile_spec,
                pte_filename=pte_filenames[EAGLE_HEAD],
                skip_quantize=True,
            ),
            AUDIO_ENCODER: Request.Data(
                compile_spec=compile_spec, pte_filename="unused"
            ),
            VISION_ENCODER: Request.Data(
                compile_spec=compile_spec, pte_filename="unused"
            ),
            TEXT_ENCODER: Request.Data(
                compile_spec=compile_spec, pte_filename="unused"
            ),
            TOK_EMBEDDING: Request.Data(
                compile_spec=compile_spec, pte_filename="unused"
            ),
            EAGLE_TARGET: Request.Data(
                compile_spec=compile_spec,
                pte_filename=pte_filenames[EAGLE_TARGET],
            ),
        }
        request = Request(method_name="compile", method_data=method_data)
        self.process(request)
