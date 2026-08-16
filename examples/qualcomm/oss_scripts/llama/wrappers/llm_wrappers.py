# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import copy
import inspect
import json
import logging
import os
import re
import time
import types

from functools import partial
from typing import Any, Dict, List, Optional

import torch

from executorch.backends.qualcomm._passes import FoldQDQ, I64toI32, TagQuantIO
from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm._passes.utils import (
    get_passes_dependency_for_capture_program,
)
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.export_utils import make_quantizer
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.backends.qualcomm.utils.utils import (
    convert_linear_to_conv2d,
    to_edge_transform_and_lower_to_qnn,
    update_spill_fill_size,
)
from executorch.devtools.backend_debug import print_delegation_info
from executorch.examples.models.llama.hf_download import (
    download_and_convert_hf_checkpoint,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_embedding_transform,
)
from executorch.examples.qualcomm.oss_scripts.llama import (
    LLM_VARIANT_ARCHS,
    LLMModelConfig,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    DECODE_QDQ_FILENAME,
    DECODER_GRAPH_NAMES,
    LM_HEAD,
    LM_HEAD_GRAPH_NAMES,
    TEXT_DECODER,
    TEXT_ENCODER,
    TOK_EMBEDDING,
    TOK_EMBEDDING_GRAPH_NAMES,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_utils import (
    encode_prompt,
    graph_module_inference,
)
from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_config import (
    GraniteSpeechEncoder,
)
from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_quant_recipe import (
    EncoderQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.mix_precision_analyzer import (
    PerLayerSqnrAnalyzer,
    save_suggest_recipes,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.embedding import (
    LmHead,
    TokenEmbedding,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaModel,
    LlamaModelWithoutEmbedding,
    ModelArgs,
)
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
    StaticLLMQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import (
    Component,
    get_model_specific_kwargs,
    log_info,
    Mode,
    process_model_args,
    Processor,
    Request,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.extension.llm.custom_ops import model_sharding
from executorch.extension.llm.export.builder import DType
from torchao.prototype.spinquant import apply_spinquant
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from transformers import AutoModel, AutoModelForSpeechSeq2Seq


class _SplitEval(torch.nn.Module):
    """Runs the full split path (tokens -> [emb] -> decoder -> [lm_head] -> logits)
    as one token-in/logits-out module, so a real KV generation can verify the split
    end-to-end (coherent text == split preserves the model). tok_embedding/lm_head are
    None when that piece is still in-graph."""

    def __init__(self, tok_embedding, decoder, lm_head):
        super().__init__()
        self.tok_embedding = tok_embedding
        self.decoder = decoder
        self.lm_head = lm_head

    def forward(self, x, *rest):
        if self.tok_embedding is not None:
            x = self.tok_embedding(x)
        out = self.decoder(x, *rest)
        logits = self.lm_head(out[0]) if self.lm_head is not None else out[0]
        return (logits, *out[1:])


class TextDecoder(Component):

    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        mode: Mode,
        apply_embedding: bool = False,
        output_hidden_layers: Optional[List[int]] = None,
        apply_output: bool = True,
        draft_head_ar: Optional[int] = None,
        fork_captured_hiddens: bool = False,
    ):
        self.control_args = control_args
        self.config = config
        self.mode = mode
        self.passes_job = get_capture_program_passes()
        self.dep_table = get_passes_dependency_for_capture_program()
        self.meta = {}
        self.output_hidden_layers = output_hidden_layers
        # Only --no_sink needs the captured hiddens on their own quantization point,
        # and the fork is not free (five elementwise muls plus a requant per forward),
        # so keep-sink builds export the residual tensor directly as they always did.
        self.fork_captured_hiddens = fork_captured_hiddens
        # DFlash: a third emb/lm_head width, block_size wide, for the draft. The draft
        # produces exactly block_size rows per round, so sharing the target's
        # decode-width graphs made the vocab projection scale with the tree size while
        # the draft filled a fixed 16 rows of it -- measured +0.193 ms per tree node,
        # all of it padding. Separate graphs also give the draft its own boundary
        # encodings, calibrated on draft traffic instead of the target's.
        self.draft_head_ar = draft_head_ar
        self.draft_tok_embedding = None
        self.draft_lm_head = None
        self.draft_tok_embedding_passes_job = (
            get_capture_program_passes() if draft_head_ar else None
        )
        self.draft_tok_embedding_dep_table = (
            get_passes_dependency_for_capture_program() if draft_head_ar else None
        )
        self.draft_lm_head_passes_job = (
            get_capture_program_passes() if draft_head_ar else None
        )
        self.draft_lm_head_dep_table = (
            get_passes_dependency_for_capture_program() if draft_head_ar else None
        )
        # DFlashManager sets this on the decode wrapper so the draft rides along in
        # this graph's calibration loop instead of being calibrated afterwards on a
        # separate forward through the (uncalibrated) prefill graph.
        self.dflash_draft = None
        # apply_output=False -> headless decoder (no lm_head, output = final hidden);
        # lm_head lives in a separate shared pte. Flips the IO tagging to hidden-shaped.
        self.apply_output = apply_output
        self.lm_head = None  # built in _get_model_instance when headless
        self.lm_head_passes_job = (
            get_capture_program_passes() if not apply_output else None
        )
        self.lm_head_dep_table = (
            get_passes_dependency_for_capture_program() if not apply_output else None
        )
        self.quant_recipe: StaticLLMQuantRecipe = (
            self.config.quant_recipe(True) if self.config.quant_recipe else None
        )

        # For multimodal embedding
        self.apply_embedding = apply_embedding
        self.tok_embedding_passes_job = (
            get_capture_program_passes() if apply_embedding else None
        )
        self.tok_embedding_dep_table = (
            get_passes_dependency_for_capture_program() if apply_embedding else None
        )

        # load static llama model args
        params_path = (
            config.params_path if control_args.params is None else control_args.params
        )
        with open(params_path) as f:
            self.model_args = process_model_args(
                control_args, ModelArgs(**json.load(f)), self.quant_recipe, config, mode
            )
        # prepare instance
        self.tok_embedding, self.decoder = self._prepare_model()

        # check if sharding required
        if self.decoder and self.config.num_sharding > 1:
            SplitGraph, setting = model_sharding.get_split_graph_pass(
                self.meta["get_n_layers"],
                shares=self.config.num_sharding,
            )
            self.passes_job[SplitGraph] = setting
            self.dep_table[SplitGraph] = [FoldQDQ]
            self.dep_table[TagQuantIO] = [SplitGraph]

    def _prepare_model(self):  # noqa: C901
        if (instance := self._get_model_instance()) is None:
            return None, None
        tok_embedding, decoder = instance
        # load parameters for HF models
        if self.control_args.checkpoint is None:
            checkpoint = download_and_convert_hf_checkpoint(
                self.config.repo_id,
                self.config.convert_weights.__func__,
            )
            state_dict = torch.load(
                checkpoint, weights_only=True, map_location="cpu", mmap=True
            )
            if self.control_args.decoder_model in {
                "gemma-2b",
                "gemma2-2b",
                "gemma3-1b",
            }:
                for k, v in state_dict.items():
                    if "norm" not in k:
                        continue
                    # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
                    # See https://github.com/huggingface/transformers/pull/29402
                    state_dict[k] = v.float() + torch.ones(v.shape, dtype=torch.float32)
        else:
            state_dict = torch.load(
                self.control_args.checkpoint,
                weights_only=True,
                map_location="cpu",
                mmap=True,
            )
            if "model" in state_dict:
                state_dict = state_dict["model"]

            if self.control_args.decoder_model == "stories260k":
                state_dict = {
                    k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
                }

        # change to HF weight to improve the performance of RoPE in HTP backend.
        if self.config.transform_weight:

            def permute(w, heads, partial_rotary_dim):
                dim_0 = w.size(0)
                dim_1 = w.size(1)
                transformed_weight = (
                    w.view(
                        heads, -1, dim_0 // heads // 2 // partial_rotary_dim, 2, dim_1
                    )
                    .transpose(2, 3)
                    .reshape(dim_0, dim_1)
                )
                return transformed_weight

            # TODO: handle cases where input size isn't divisible.
            partial_rotary_dim = int(1 // self.model_args.partial_rotary_factor)
            for layer_i in range(decoder.n_layers):
                state_dict[f"layers.{layer_i}.attention.wq.weight"] = permute(
                    state_dict[f"layers.{layer_i}.attention.wq.weight"],
                    decoder.n_heads,
                    partial_rotary_dim,
                )
                state_dict[f"layers.{layer_i}.attention.wk.weight"] = permute(
                    state_dict[f"layers.{layer_i}.attention.wk.weight"],
                    decoder.n_kv_heads,
                    partial_rotary_dim,
                )

        decoder.load_state_dict(state_dict, strict=True, assign=True)

        # apply spin quant if required
        if any([self.config.r1, self.config.r2]):
            decoder.config = types.SimpleNamespace(
                dim=decoder.dim,
                head_dim=decoder.dim // decoder.n_heads,
                n_local_heads=decoder.n_heads,
                intermediate_size=4 * decoder.dim,
            )
            apply_spinquant(
                decoder,
                use_r1=self.config.r1,
                use_r2=self.config.r2,
                use_r4=False,
                pretrained_rotation_path=None,
                qkv_split=True,
            )

        # perform model transformation
        for layer in decoder.layers:
            if getattr(layer.attention, "prepare_attention_conv", None):
                layer.attention.prepare_attention_conv()
            if getattr(layer.feed_forward, "prepare_feedfoward_conv", None):
                layer.feed_forward.prepare_feedfoward_conv()

        decoder = convert_linear_to_conv2d(decoder)

        # convert_linear_to_conv2d replaces decoder.output in place with a new
        # Conv2D; the LmHead built earlier still references the old nn.Linear, so
        # re-point it at the converted module for op/quant parity with the in-graph
        # lm_head.
        if self.lm_head is not None:
            self.lm_head.output = decoder.output
        if self.draft_lm_head is not None:
            self.draft_lm_head.output = decoder.output

        # check dtype override
        if self.control_args.dtype_override is not None:
            dtype_override = DType[self.control_args.dtype_override]
            decoder = decoder.to(dtype_override.to_torch_dtype())

        # check embedding fallback
        if self.control_args.embedding_quantize:
            decoder = get_quant_embedding_transform(
                embedding_quantize=self.control_args.embedding_quantize
            )(decoder)
            self.passes_job[I64toI32][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                "skip_node"
            ] = {"tokens"}
            if self.apply_embedding:
                tok_embedding = get_quant_embedding_transform(
                    embedding_quantize=self.control_args.embedding_quantize
                )(tok_embedding)
                self.tok_embedding_passes_job[I64toI32][
                    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY
                ]["skip_node"] = {"tokens"}

        if tok_embedding is not None:
            tok_embedding = tok_embedding.eval()

        return tok_embedding, decoder.eval()

    def _get_model_instance(self) -> LlamaModel:
        if self.mode == Mode.PREFILL and self.control_args.model_mode == "kv":
            return None
        use_i64_token = self.control_args.embedding_quantize is not None
        self.is_multimodal = hasattr(self.config, AUDIO_ENCODER) or hasattr(
            self.config, VISION_ENCODER
        )

        # get embedding model (multimodal: from the HF encoder model). Text
        # emb-split builds its TokenEmbedding from the decoder's own table below.
        tok_embedding = None
        if self.apply_embedding and self.is_multimodal:
            if hasattr(self.config, AUDIO_ENCODER):
                auto_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.config.repo_id, _attn_implementation="eager"
                )
            elif hasattr(self.config, VISION_ENCODER):
                auto_model = AutoModel.from_pretrained(
                    self.config.repo_id, _attn_implementation="eager"
                )
            tok_embedding = TokenEmbedding(
                auto_model.get_input_embeddings().to(torch.float32),
                self.model_args.max_batch_size,
                self.model_args.ar_len,
                self.model_args.vocab_size,
                self.model_args.dim,
                use_i64_token,
            )
        # get decoder model
        if self.control_args.decoder_model in {"gemma-2b", "gemma3-1b"}:
            # For gemma, we have preprocessed the weight of rmsnorm
            self.model_args.norm_type = "rmsnorm"

        # emb-split text decoder takes inputs_embeds (LlamaModelWithoutEmbedding) so
        # the embedding table can move to a separate shared pte. Multimodal models are
        # already mapped to WithoutEmbedding in LLM_VARIANT_ARCHS.
        arch = LLM_VARIANT_ARCHS.get(self.control_args.decoder_model, LlamaModel)
        if self.apply_embedding and not self.is_multimodal:
            arch = LlamaModelWithoutEmbedding
        decoder: LlamaModel = arch(
            self.model_args,
            ar_len=self.model_args.ar_len,
            output_new_cache_only=True,
            output_cache=True,
            use_i64_token=use_i64_token,
            output_hidden_layers=self.output_hidden_layers,
            fork_captured_hiddens=self.fork_captured_hiddens,
            apply_output=self.apply_output,
            **get_model_specific_kwargs(self.control_args, self.config),
        )

        # text emb-split: build a standalone TokenEmbedding from the decoder's own
        # embedding table (mirrors LmHead). It references decoder.tok_embeddings, which
        # load_state_dict populates later; nn.Embedding is untouched by
        # convert_linear_to_conv2d so no re-point is needed.
        if self.apply_embedding and not self.is_multimodal:
            tok_embedding = TokenEmbedding(
                decoder.tok_embeddings,
                self.model_args.max_batch_size,
                self.model_args.ar_len,
                self.model_args.vocab_size,
                self.model_args.dim,
                use_i64_token,
            )

        self.meta = decoder.get_metadata()
        # get example input
        self.example_input = decoder.get_example_inputs()
        self.get_example_inputs = decoder.get_example_inputs
        self.export_input = (
            self.example_input[0],  # tokens or hidden_states
            *self.example_input[1],  # attn_mask
            *((self.example_input[2],) if decoder.use_kv_cache else []),  # pos_ids
            *(self.example_input[3] if decoder.use_kv_cache else []),  # k_caches
            *(self.example_input[4] if decoder.use_kv_cache else []),  # v_caches
        )
        self.io_shape = {
            # graph output: logits (vocab) normally, or the final hidden (dim) when
            # headless (apply_output=False) — lm_head then lives in a separate pte.
            (
                decoder.max_batch_size,
                decoder.ar_len,
                decoder.vocab_size if self.apply_output else decoder.dim,
            ),
        }
        # shape of k caches and v caches
        self.kv_cache_shape = {
            # single head, kv input
            (self.meta["get_head_dim"], self.meta["get_max_context_len"]),
            (self.meta["get_max_context_len"], self.meta["get_head_dim"]),
            # single head, kv output
            (self.meta["get_head_dim"], self.meta["get_ar_len"]),
            (self.meta["get_ar_len"], self.meta["get_head_dim"]),
        }

        if self.apply_embedding:
            self.tok_embedding_export_input = (
                tok_embedding.get_example_input()
            )  # tokens
            # headless emb-split: the emb.pte output (embeds [b, ar, dim]) is tagged
            # uint16 so the emb->decoder boundary is lossless uint16 (the emb-output
            # scale is injected onto the decoder input in compile). tokens input stays
            # int and is not tagged.
            self.emb_io_shape = {
                (decoder.max_batch_size, decoder.ar_len, decoder.dim)
            }

        # headless: build a standalone lm_head (hidden -> logits) so the vocab
        # projection can be lowered to a separate shared pte. Quantize/lower are
        # wired in quantize()/compile(); here we just construct it + its export input.
        self.lm_head = None
        if not self.apply_output:
            self.lm_head = LmHead(
                decoder.output,
                decoder.max_batch_size,
                decoder.ar_len,
                decoder.vocab_size,
                decoder.dim,
            ).eval()
            self.lm_head_export_input = self.lm_head.get_example_input()
            # input = final hidden [b, ar, dim]; output = logits [b, ar, vocab] —
            # both quantized uint16 at the pte boundary.
            self.lm_head_io_shape = {
                (decoder.max_batch_size, decoder.ar_len, decoder.dim),
                (decoder.max_batch_size, decoder.ar_len, decoder.vocab_size),
            }

        # The draft-width pair. Same underlying modules as above -- only the AR
        # differs -- so the pte weight-shares all three graphs.
        if self.draft_head_ar:
            if self.control_args.embedding_quantize:
                raise RuntimeError(
                    "--embedding_quantize with a DFlash draft-width emb graph: both "
                    "wrap the same nn.Embedding and get_quant_embedding_transform "
                    "would run over it twice"
                )
            B = self.draft_head_ar
            self.draft_tok_embedding = TokenEmbedding(
                decoder.tok_embeddings,
                decoder.max_batch_size,
                B,
                decoder.vocab_size,
                decoder.dim,
                self.control_args.embedding_quantize is not None,
            ).eval()
            self.draft_tok_embedding_export_input = (
                self.draft_tok_embedding.get_example_input()
            )
            self.draft_emb_io_shape = {(decoder.max_batch_size, B, decoder.dim)}
            self.draft_lm_head = LmHead(
                decoder.output,
                decoder.max_batch_size,
                B,
                decoder.vocab_size,
                decoder.dim,
            ).eval()
            self.draft_lm_head_export_input = (
                self.draft_lm_head.get_example_input()
            )
            self.draft_lm_head_io_shape = {
                (decoder.max_batch_size, B, decoder.dim),
                (decoder.max_batch_size, B, decoder.vocab_size),
            }

        return tok_embedding, decoder

    def _save_logits_quant_attrs(self):
        for node in self.decoder.graph.nodes:
            if node.op == "output":
                for output_node in node.args[0]:
                    if (
                        output_node.target
                        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                    ):
                        source_node = output_node.args[0].args[0]
                        if source_node.meta["val"].size() in self.io_shape:
                            self.meta["get_logits_scale"] = output_node.args[1]
                            self.meta["get_logits_zero_point"] = output_node.args[2]
                            break

    def _save_lm_head_output_quant_attrs(self):
        """The lm_head's OUTPUT (logits) encoding, so the runner is not told it by hand.

        Only the draft tree reads it: a tree scores paths by summing log-probs across
        DEPTHS, and this scale is what turns a code gap into nats -- i.e. it decides
        how the node budget splits between depth and breadth. Chain decoding never
        touches it, because argmax is scale-invariant. That asymmetry is exactly why
        it has to ship in the pte: a wrong value cannot fail, it can only quietly cost
        acceptance, and it moves per build (0.001559 for b16_joint, 0.001439 for
        tree32).
        """
        for node in self.lm_head.graph.nodes:
            if node.op != "output":
                continue
            for out in node.args[0]:
                if (
                    out.target
                    == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                ):
                    self.meta["get_logits_out_scale"] = out.args[1]
                    self.meta["get_logits_out_zero_point"] = out.args[2]
                    logging.info(
                        "lm_head output encoding: scale=%s zp=%s",
                        out.args[1],
                        out.args[2],
                    )
                    return

    def _save_draft_head_quant_attrs(self):
        """The draft-width lm_head's INPUT and OUTPUT encodings, for the runner.

        It needs both because the draft's hidden crosses to lm_head.pte as f32 and is
        quantized on the host, and because the tree turns code gaps into nats.

        Called twice: once here in quantize(), which records what this graph calibrated
        on draft traffic, and again in compile() after those encodings are overridden
        from decode. The second call wins; the first survives only as a log line, which
        is how the two are compared.
        """
        q_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
        dq_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default
        for node in self.draft_lm_head.graph.nodes:
            if node.op == "placeholder":
                for user in node.users:
                    if user.target is q_op:
                        self.meta["get_draft_hidden_scale"] = user.args[1]
                        self.meta["get_draft_hidden_zero_point"] = user.args[2]
                        break
            if node.op == "output":
                for out in node.args[0]:
                    if out.target is dq_op:
                        self.meta["get_draft_logits_out_scale"] = out.args[1]
                        self.meta["get_draft_logits_out_zero_point"] = out.args[2]
                        break
        logging.info(
            "draft-width lm_head encodings: hidden in scale=%s zp=%s, "
            "logits out scale=%s zp=%s",
            self.meta.get("get_draft_hidden_scale"),
            self.meta.get("get_draft_hidden_zero_point"),
            self.meta.get("get_draft_logits_out_scale"),
            self.meta.get("get_draft_logits_out_zero_point"),
        )
        if "get_draft_hidden_scale" not in self.meta:
            raise RuntimeError("draft lm_head input quant params not found")
        if "get_draft_logits_out_scale" not in self.meta:
            raise RuntimeError("draft lm_head output quant params not found")

    def _save_embeds_quant_attrs(self):
        # headless emb-split: read the tok_embedding output (embeds) quant scale/zp so it
        # can be injected onto the decoder's inputs_embeds input, making emb->decoder a
        # lossless uint16 boundary. Mirror of _save_logits_quant_attrs, on tok_embedding.
        for node in self.tok_embedding.graph.nodes:
            if node.op == "output":
                for output_node in node.args[0]:
                    if (
                        output_node.target
                        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                    ):
                        source_node = output_node.args[0].args[0]
                        if source_node.meta["val"].size() in self.emb_io_shape:
                            self.meta["get_embeds_scale"] = output_node.args[1]
                            self.meta["get_embeds_zero_point"] = output_node.args[2]
                            break

    def _save_output_kv_cache_quant_attrs(self):
        kv_idx = 0
        for node in self.decoder.graph.nodes:
            if not is_graph_output(node):
                continue
            cache_output_node = node.args[0].args[0]
            if cache_output_node.meta["val"].size()[-2:] in self.kv_cache_shape:
                # [QCOM_SCALE, QCOM_ZERO_POINT, QCOM_QUANT_MIN, QCOM_QUANT_MAX, QCOM_DTYPE]
                # This meta is for attention sink feature
                self.meta[f"get_kv_output_{kv_idx}_quant_attr"] = [
                    node.args[1],
                    node.args[2],
                    node.args[3],
                    node.args[4],
                    str(node.args[5]),
                ]
                kv_idx += 1

    def _tag_ios(self, node, fixed_point_type):
        atten_mask_shape = {
            (
                self.meta["get_max_batch_size"],
                self.meta["get_ar_len"],
                self.meta["get_max_context_len"],
            ),
        }

        freq_shape = {
            (self.meta["get_ar_len"], self.meta["get_head_dim"] // 2),
        }

        freq_op = {
            exir_ops.edge.aten.select.int,
        }
        quant_io_type = None

        if node.op == "placeholder":
            if (
                len(users := list(node.users)) == 1
                and users[0].meta["val"].size()[-2:] in self.kv_cache_shape
            ):
                quant_io_type = fixed_point_type["kv_type"]
            elif node.meta["val"].size() in self.io_shape:
                # Tag the [b, ar, dim] IO uint16. For headless emb-split this is the
                # inputs_embeds placeholder — tagging it makes the emb->decoder boundary
                # uint16 (the emb-output scale is injected onto it in compile). For
                # emb-split-ONLY (apply_output=True) io_shape is vocab-width, so
                # inputs_embeds (dim-width) never matches here and stays fp32 (unchanged).
                quant_io_type = fixed_point_type["io_type"]
            elif node.meta["val"].size() in atten_mask_shape:
                quant_io_type = fixed_point_type["io_type"]

        if is_graph_output(node):
            if node.meta["val"].size()[-2:] in self.kv_cache_shape:
                quant_io_type = fixed_point_type["kv_type"]
            elif node.meta["val"].size() in self.io_shape:
                # A headless decoder that also exports selected-layer hiddens (DFlash)
                # emits captured hiddens at the SAME [b, ar, dim] shape as the final
                # hidden. Only the final hidden (output position 0) feeds lm_head.pte
                # and must be uint16; the captured hiddens go to the draft as fp, so
                # leave them un-tagged (the runtime rejects quantized captured hiddens).
                if self.output_hidden_layers is None or self._is_primary_output(node):
                    quant_io_type = fixed_point_type["io_type"]

        # tag sharding io
        if exir_ops.edge.llama.fallback.default in [
            u.target for u in list(node.users.keys())
        ] + [node.target]:
            quant_io_type = fixed_point_type["io_type"]

        # tag select op as quantized tensors for freq_sin and freq_cos. It is caused by sharding
        if node.target in freq_op and node.meta["val"].size() in freq_shape:
            quant_io_type = fixed_point_type["io_type"]

        return quant_io_type

    def _is_primary_output(self, node):
        # forward returns (logits/hidden, k_caches..., v_caches..., *captured_hiddens),
        # so the flattened graph output at position 0 is the primary hidden; the trailing
        # captured hiddens share its [b, ar, dim] shape but must not be tagged.
        out_node = next(n for n in node.graph.nodes if n.op == "output")
        flat = list(out_node.args[0])
        return node in flat and flat.index(node) == 0

    def _tag_lm_head_ios(self, node, fixed_point_type, io_shape=None):
        # standalone lm_head graph: input hidden [b, ar, dim] and output logits
        # [b, ar, vocab] are both quantized uint16 at the pte boundary. io_shape is
        # explicit because the draft-width graph has the same roles at a different AR.
        io_shape = self.lm_head_io_shape if io_shape is None else io_shape
        quant_io_type = None
        if node.op == "placeholder" and node.meta["val"].size() in io_shape:
            quant_io_type = fixed_point_type["io_type"]
        if is_graph_output(node) and node.meta["val"].size() in io_shape:
            quant_io_type = fixed_point_type["io_type"]
        return quant_io_type

    def _tag_emb_ios(self, node, fixed_point_type, io_shape=None):
        # standalone tok_embedding graph: the tokens input stays int (untagged); only
        # the embeds output [b, ar, dim] is quantized uint16 at the pte boundary, so
        # emb->decoder is a lossless uint16 boundary (scale injected onto decoder input).
        io_shape = self.emb_io_shape if io_shape is None else io_shape
        quant_io_type = None
        if is_graph_output(node) and node.meta["val"].size() in io_shape:
            quant_io_type = fixed_point_type["io_type"]
        return quant_io_type

    def _quant_recipe_suggestion(
        self,
        fp32_gm: torch.fx.GraphModule,
        qdq_gm: torch.fx.GraphModule,
        input_sample: tuple,
        recipe: StaticLLMQuantRecipe,
    ):
        """
        Compare fp32 vs QDQ intermediate outputs and write SQNR reports.

        fp32_gm: Fp32 exported GraphModule (before prepare_pt2e).
        qdq_gm: QDQ GraphModule (after convert_pt2e).

        Output files:
          ``{model_name}_quantization_error.csv``: per-group statistics
          ``{model_name}_suggest_recipe.py``: Python script containing quantization recipe classes
        based on the suggested quant recipe overrides.
        """
        model_name = self.control_args.decoder_model
        report = PerLayerSqnrAnalyzer(
            model_name=model_name,
            num_layers=self.meta["get_n_layers"],
            fp32_gm=fp32_gm,
            qdq_gm=qdq_gm,
            analysis_recipe=recipe,
        ).analyze(input_sample)
        report.save_analysis_summary()
        suggest_recipe_overrides = report.suggest_recipe_overrides()
        save_suggest_recipes(report, suggest_recipe_overrides)

    def _auto_tune_calibration_threads(self):
        """Find the optimal thread count for calibration via quick microbenchmark.

        AR1 decode calibration is SGEMV-dominated (memory-bandwidth-bound).
        The default thread count (os.cpu_count()) is typically far too high,
        causing massive OpenMP sync overhead. This runs a few forward passes
        at candidate thread counts and picks the fastest.
        """
        # Use sched_getaffinity when available — it respects cgroup/taskset
        # constraints (e.g. containers), unlike os.cpu_count() which returns
        # the host total regardless of pinning.
        available = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
        baseline = min(torch.get_num_threads(), available)
        # Sample fractions of the thread ceiling from low through the
        # bandwidth-saturation knee up to the current default.
        fractions = (1 / 8, 1 / 4, 3 / 8, 1 / 2, 2 / 3, 3 / 4, 1.0)
        candidates = sorted(
            {1, baseline} | {max(1, round(baseline * f)) for f in fractions}
        )
        original = torch.get_num_threads()
        best_threads, best_time = original, float("inf")
        try:
            for n_threads in candidates:
                torch.set_num_threads(n_threads)
                try:
                    with torch.no_grad():
                        self.decoder(*self.export_input)  # warmup
                        t0 = time.perf_counter()
                        for _ in range(3):
                            self.decoder(*self.export_input)
                        elapsed = time.perf_counter() - t0
                    if elapsed < best_time:
                        best_threads, best_time = n_threads, elapsed
                except Exception:
                    logging.debug("Auto-tune: threads=%d failed, skipping", n_threads)
                    continue
        finally:
            torch.set_num_threads(original)
        if best_time == float("inf"):
            logging.warning(
                "Auto-tune: all candidates %s failed, falling back to %d threads",
                candidates,
                baseline,
            )
            return baseline
        logging.info(
            "Auto-tune calibration threads: tested %s, best=%d (%.1fms/fwd)",
            candidates,
            best_threads,
            best_time / 3 * 1000,
        )
        return best_threads

    def _calibrate(
        self,
        model,
        tokenizer,
        event,
        user_calibration_data,
        tok_embedding=None,
        intermediate_outputs=None,
    ):
        """
        Calibrate the model using either task-based evaluation or prompt-based inference.

        This method performs Post-Training Quantization (PTQ) calibration by running inference
        on the model with either:
        1. Task-based datasets by lm_eval for text-only models in perplexity evaluation
        2. User-provided prompts for both text-only and multimodal models

        Args:
            model: The decoder model to calibrate (GraphModule after prepare_pt2e)
            tokenizer: Tokenizer for encoding text inputs
            event: Event name for logging (e.g., "prepare_pt2e", "convert_pt2e")
            tok_embedding: Optional text embedding module (required only for multimodal models)
            intermediate_outputs: Optional pre-computed embeddings from vision/audio encoder
                                 (required only for multimodal models)
        """
        # Determine if this is a multimodal model
        is_multimodal = tok_embedding is not None

        # Determine if task-based calibration is requested
        has_task_calibration = self.control_args.tasks is not None

        # Task-based calibration: Only for text-only LLMs
        # Multimodal models (VLMs) cannot use task-based evaluation currently.
        input_samples = []
        if has_task_calibration and not is_multimodal:
            input_sample = graph_module_inference(
                use_kv_cache=self.meta["get_use_kv_cache"],
                get_example_inputs=self.get_example_inputs,
                module=model,
                tokenizer=tokenizer,
                ar_len=self.meta["get_ar_len"],
                max_seq_len=self.meta["get_max_context_len"],
                tasks=self.control_args.tasks,
                tasks_limit=self.control_args.limit,
                num_fewshot=self.control_args.num_fewshot,
                use_i64_token=self.control_args.embedding_quantize is not None,
                event_name=f"{event}_tasks",
                seq_mse_candidates=self.config.seq_mse_candidates,
            )
            input_samples.extend(input_sample)

        # prepare lookahead config if applicable
        lookahead_config = (
            (self.control_args.window, self.control_args.ngram, self.control_args.gcap)
            if (
                self.mode == Mode.DECODE and self.control_args.model_mode == "lookahead"
            )
            else None
        )
        # prepare dflash config if applicable: the draft is observed inside this
        # graph's loop, so its new_context range comes from the hidden this graph
        # actually emits and its block traffic comes from real accept lengths.
        dflash_config = (
            self.dflash_draft.make_calibrator(
                # Deliberately the decode-width graphs, NOT the draft-width pair, even
                # though the pair is what the runner calls. The pair takes decode's
                # encodings by override in compile(), so it needs no traffic of its
                # own -- and routing this loop through it perturbed the target:
                # decoder.output is a Conv2d after convert_linear_to_conv2d, a 16-wide
                # fp32 conv rounds differently from a 32-wide one, and an occasional
                # flipped token in a REJECTED tree slot still trains the decoder's
                # observers (the tree feeds the target 32 rows; ~6 are walked, all 32
                # are observed). Measured: identical accept (211/248 over 37 rounds)
                # but get_logits_scale moved 5.6%, which is enough to move device
                # acceptance ~9% and make the two builds incomparable.
                self.tok_embedding,
                self.lm_head,
                self.meta["get_ar_len"],
                torch.int64
                if self.control_args.embedding_quantize is not None
                else torch.int32,
                self.control_args.dflash_tree_nodes > 0,
            )
            # `event` gates the post-convert re-run: that pass exists for error
            # analysis and would drive the draft's cache a second time.
            if self.dflash_draft is not None
            and self.mode == Mode.DECODE
            and event == "prepare_pt2e"
            else None
        )
        # check user's prompt which helps calibrate special token
        for turn in zip(intermediate_outputs, user_calibration_data):
            hidden_states, prompt = turn
            input_sample = graph_module_inference(
                use_kv_cache=self.meta["get_use_kv_cache"],
                get_example_inputs=self.get_example_inputs,
                hidden_states=hidden_states,  # hidden_states for multimodal
                module=model,
                tok_embedding=tok_embedding,
                audio_token_id=self.meta.get("audio_token_id", None),
                image_token_id=self.meta.get("image_token_id", None),
                tokenizer=tokenizer,
                ar_len=self.meta["get_ar_len"],
                max_seq_len=self.meta["get_max_context_len"],
                prompt=prompt,
                use_i64_token=self.control_args.embedding_quantize is not None,
                event_name=f"{event}_prompt",
                lookahead_config=lookahead_config,
                dflash_config=dflash_config,
            )
            input_samples.extend(input_sample)
        return input_samples

    def _override_lm_head_input_scale(self, scale, zero_point):
        # Force the lm_head input-activation quant params to the decoder's
        # output-hidden encoding so decoder(uint16 hidden) -> lm_head(uint16
        # hidden) requant is lossless across the pte boundary.
        q_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
        dq_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default
        patched = 0
        for node in self.lm_head.graph.nodes:
            if node.op != "placeholder":
                continue
            for user in list(node.users):
                if user.target is q_op:
                    user.args = (user.args[0], scale, zero_point, *user.args[3:])
                    patched += 1
                    for dq in list(user.users):
                        if dq.target is dq_op:
                            dq.args = (dq.args[0], scale, zero_point, *dq.args[3:])
        self.lm_head.recompile()
        logging.info(
            "lm_head input scale injected: patched=%d scale=%s zp=%s",
            patched,
            scale,
            zero_point,
        )
        if patched == 0:
            raise RuntimeError(
                "lm_head input quantize node not found; boundary scale not injected"
            )

    def _override_decoder_input_scale(self, scale, zero_point):
        # Force the decoder's inputs_embeds input-activation quant params to the emb
        # output encoding so emb(uint16 embeds) -> decoder(uint16 inputs_embeds) requant
        # is lossless across the pte boundary. Mirror of _override_lm_head_input_scale
        # with the direction reversed (emb output is the source). Only the inputs_embeds
        # placeholder matches io_shape (dim-width, headless); tokens/mask/pos/kv differ.
        q_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
        dq_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default
        patched = 0
        for node in self.decoder.graph.nodes:
            if node.op != "placeholder":
                continue
            if node.meta["val"].size() not in self.io_shape:
                continue
            for user in list(node.users):
                if user.target is q_op:
                    user.args = (user.args[0], scale, zero_point, *user.args[3:])
                    patched += 1
                    for dq in list(user.users):
                        if dq.target is dq_op:
                            dq.args = (dq.args[0], scale, zero_point, *dq.args[3:])
        self.decoder.recompile()
        logging.info(
            "decoder input scale injected: patched=%d scale=%s zp=%s",
            patched,
            scale,
            zero_point,
        )
        if patched == 0:
            raise RuntimeError(
                "decoder inputs_embeds quantize node not found; boundary scale not injected"
            )

    def _verify_generate(self):
        # Verify the split end-to-end with a REAL KV generation on the calibration
        # prompt: build the full split path (tokens -> [emb] -> decoder -> [lm_head])
        # as one module and run it through kv_inference, which logs the decoded text.
        # Coherent text (matching the monolithic run) == the split preserves the model.
        # (A single-token / no-context dump is useless here: headless decoders output
        # hidden, and gibberish tokens saturate the logits.)
        # no .eval() — submodules are exported QDQ GraphModules (already eval);
        # torchao forbids train()/eval() on exported models.
        combined = _SplitEval(
            self.tok_embedding if self.apply_embedding else None,
            self.decoder,
            self.lm_head if not self.apply_output else None,
        )
        # run a few diverse prompts so the mono-vs-split comparison isn't a single point
        prompts = [
            self._verify_prompt,
            "The capital of France is",
        ]
        for i, p in enumerate(prompts):
            graph_module_inference(
                use_kv_cache=self.meta["get_use_kv_cache"],
                get_example_inputs=self.get_example_inputs,
                module=combined,
                tokenizer=self._verify_tokenizer,
                ar_len=self.meta["get_ar_len"],
                # Cap GENERATION so a broken split doesn't emit a full context of
                # garbage. max_seq_len bounds prompt+generation together, so it has to
                # clear the prompt first -- a flat 64 silently worked only while every
                # calibration prompt was shorter than that.
                max_seq_len=min(
                    self.meta["get_max_context_len"],
                    len(encode_prompt(self._verify_tokenizer, p)) + 64,
                ),
                prompt=p,
                use_i64_token=self.control_args.embedding_quantize is not None,
                event_name=f"verify_split_{i}",
            )

    def _dump_split_qdq(self, split_lm_head):
        """Export + save the converted (QDQ) decode graphs of the split modules so a
        host script can load them and run the DFlash accept loop on CPU. QDQ graphs
        are fp32 weights + quantize/dequantize nodes, so they run on CPU and match the
        pte numerically. Load them back with `import torch.ao.quantization.fx._decomposed`
        first (registers the quantized_decomposed ops the graph references)."""
        art = self.control_args.artifact

        def save(module, example_input, name):
            ep = torch.export.export(module, example_input, strict=True)
            path = f"{art}/{name}.pt2"
            torch.export.save(ep, path)
            logging.info(
                "dumped QDQ %s (ar=%d) -> %s", name, self.meta["get_ar_len"], path
            )

        save(self.decoder, self.export_input, "qdq_decoder")
        if self.apply_embedding:
            save(self.tok_embedding, self.tok_embedding_export_input, "qdq_tok_embedding")
        if split_lm_head:
            save(self.lm_head, self.lm_head_export_input, "qdq_lm_head")

    @log_info
    def quantize(self, request: Request):  # noqa: C901
        if self.quant_recipe is None:
            return

        if self.decoder is None or (
            self.apply_embedding and self.tok_embedding is None
        ):
            return

        data = request.method_data[TEXT_DECODER]
        # check bit width graph io
        fixed_point_type = {"kv_type": torch.float32, "io_type": torch.float32}
        if data.skip_quantize:
            # already init as float32
            return
        else:
            if self.quant_recipe.get_kv_io_bit_width() == 8:
                fixed_point_type["kv_type"] = torch.uint8
            elif self.quant_recipe.get_kv_io_bit_width() == 16:
                fixed_point_type["kv_type"] = torch.uint16
            else:
                raise RuntimeError(
                    f"unknown kv io bit width {self.quant_recipe.get_kv_io_bit_width()}"
                )

            if self.quant_recipe.get_logits_output_bit_width() == 16:
                fixed_point_type["io_type"] = torch.uint16
            else:
                raise RuntimeError(
                    f"unknown logits io bit width {self.quant_recipe.get_logits_output_bit_width()}"
                )

        data = request.method_data[TEXT_DECODER]
        audio_turns = request.method_data[
            AUDIO_ENCODER
        ].calibration_data.intermediate_outputs
        vision_turns = request.method_data[
            VISION_ENCODER
        ].calibration_data.intermediate_outputs
        if audio_turns is None:
            audio_turns = [[] for _ in range(len(data.calibration_data.datasets))]
        if vision_turns is None:
            vision_turns = [[] for _ in range(len(data.calibration_data.datasets))]
        intermediate_outputs = [
            [*audio_turn, *vision_turn]
            for audio_turn, vision_turn in zip(audio_turns, vision_turns)
        ]

        quantizer = make_quantizer(backend=data.backend, soc_model=data.soc_model)
        quantizer.set_recipe(self.quant_recipe.recipe)

        tok_embedding_quantizer = make_quantizer(
            quant_dtype=QuantDtype.use_16a8w,
            per_channel_conv=True,
            per_channel_linear=True,
            act_observer=MinMaxObserver,
            backend=data.backend,
            soc_model=data.soc_model,
        )

        with torch.no_grad():
            # prepare tok embedding model for ptq
            if self.apply_embedding:
                self.tok_embedding = torch.export.export(
                    self.tok_embedding,
                    self.tok_embedding.get_example_input(),
                    strict=True,
                ).module()

            # prepare decoder model for ptq
            self.decoder = torch.export.export(
                self.decoder, self.export_input, strict=True
            ).module()
            if self.control_args.quant_recipe_suggestion:
                graph_module = copy.deepcopy(self.decoder)

            # real decoder inputs collected during calibration (used by
            # quant_recipe_suggestion below).
            input_samples = None

            # Auto-tune thread count BEFORE prepare_pt2e so the benchmark
            # runs on the exported model without observers — no risk of
            # polluting observer state with synthetic inputs.
            if self.mode == Mode.DECODE or not self.model_args.use_kv_cache:
                calib_threads = getattr(self.control_args, "calibration_num_threads", 0)
                if calib_threads <= 0:
                    calib_threads = self._auto_tune_calibration_threads()

            self.decoder = prepare_pt2e(self.decoder, quantizer)
            if self.apply_embedding:
                self.tok_embedding = prepare_pt2e(
                    self.tok_embedding, tok_embedding_quantizer
                )

            # 路B (joint calibration): when the lm_head is split out, calibrate the
            # WHOLE chain together on the REAL trajectory. A headless decoder alone
            # argmaxes its hidden output -> garbage tokens -> garbage-trajectory
            # calibration (transformer + scales polluted). Preparing the lm_head now
            # and generating through the combined module makes argmax use the real
            # lm_head logits -> real tokens -> real hidden, so every observer sees the
            # real distribution. = "一起量化，切分pte的时候分开".
            split_lm_head = not self.apply_output and self.lm_head is not None
            calib_model = self.decoder
            if split_lm_head:
                lm_head_quantizer = make_quantizer(
                    quant_dtype=QuantDtype.use_16a8w,
                    per_channel_conv=True,
                    per_channel_linear=True,
                    act_observer=MinMaxObserver,
                    backend=data.backend,
                    soc_model=data.soc_model,
                )
                self.lm_head = torch.export.export(
                    self.lm_head, self.lm_head_export_input, strict=True
                ).module()
                self.lm_head = prepare_pt2e(self.lm_head, lm_head_quantizer)
                calib_model = _SplitEval(
                    self.tok_embedding if self.apply_embedding else None,
                    self.decoder,
                    self.lm_head,
                )
                # The draft-width pair rides the same loop, driven by DFlashCalibrator
                # rather than by _SplitEval: its observers must see draft traffic only.
                if self.draft_head_ar:
                    self.draft_tok_embedding = prepare_pt2e(
                        torch.export.export(
                            self.draft_tok_embedding,
                            self.draft_tok_embedding_export_input,
                            strict=True,
                        ).module(),
                        tok_embedding_quantizer,
                    )
                    self.draft_lm_head = prepare_pt2e(
                        torch.export.export(
                            self.draft_lm_head,
                            self.draft_lm_head_export_input,
                            strict=True,
                        ).module(),
                        lm_head_quantizer,
                    )

            # start calibration (only for kv mode or prefill mode without kv cache)
            if self.mode == Mode.DECODE or not self.model_args.use_kv_cache:
                original_threads = torch.get_num_threads()
                torch.set_num_threads(calib_threads)
                logging.info(
                    "Calibration using %d threads (was %d)",
                    calib_threads,
                    original_threads,
                )
                try:
                    input_samples = self._calibrate(
                        model=calib_model,
                        tokenizer=data.tokenizer,
                        event="prepare_pt2e",
                        user_calibration_data=data.calibration_data.datasets,
                        # combined model embeds internally -> feed tokens (None)
                        tok_embedding=None if split_lm_head else self.tok_embedding,
                        intermediate_outputs=intermediate_outputs,
                    )
                finally:
                    torch.set_num_threads(original_threads)
            else:
                # one dummy inference to remove affine observer
                # error happened in convert_pt2e
                self.decoder(*self.export_input)
                if split_lm_head:
                    self.lm_head(*self.lm_head_export_input)

            self.decoder = convert_pt2e(self.decoder)

            if self.control_args.quant_recipe_suggestion:
                self._quant_recipe_suggestion(
                    graph_module,
                    self.decoder,
                    input_samples,
                    self.quant_recipe.recipe,
                )

            # Saving Decode QDQ Model EP for SQNR evaluation
            if self.mode == Mode.DECODE:
                qdq_ep = torch.export.export(
                    self.decoder, self.export_input, strict=True
                )
                qdq_ep_path = f"{self.control_args.artifact}/{DECODE_QDQ_FILENAME}"
                torch.export.save(qdq_ep, qdq_ep_path)
                logging.info(f"QDQ EP saved to {qdq_ep_path}")

            if self.apply_embedding:
                self.tok_embedding = convert_pt2e(self.tok_embedding)
                # headless emb-split: read emb output scale for the uint16 emb->decoder
                # boundary (injected onto the decoder input in compile).
                if not self.apply_output:
                    self._save_embeds_quant_attrs()

            if split_lm_head:
                self.lm_head = convert_pt2e(self.lm_head)
                self._save_lm_head_output_quant_attrs()

            if self.draft_head_ar:
                # Never driven by the calibration loop, so convert_pt2e still needs one
                # forward apiece to clear the affine observer. Their encodings come
                # from decode by override in compile(), so what these see is irrelevant.
                with torch.no_grad():
                    self.draft_tok_embedding(*self.draft_tok_embedding_export_input)
                    self.draft_lm_head(*self.draft_lm_head_export_input)
                self.draft_tok_embedding = convert_pt2e(self.draft_tok_embedding)
                self.draft_lm_head = convert_pt2e(self.draft_lm_head)
                self._save_draft_head_quant_attrs()

            # Dump the decode (AR) QDQ graphs so a host script can run the full
            # DFlash accept loop on CPU. Only the decode graphs carry the real
            # joint-calibration encoding (prefill scales are copied in at compile),
            # and the host loop runs entirely at decode AR, so one graph per module.
            if getattr(self.control_args, "dump_qdq", False) and self.mode == Mode.DECODE:
                self._dump_split_qdq(split_lm_head)

            if self.control_args.verbose and self.mode == Mode.DECODE:
                audio_turns = request.method_data[
                    AUDIO_ENCODER
                ].calibration_data.qdq_intermediate_outputs
                vision_turns = request.method_data[
                    VISION_ENCODER
                ].calibration_data.qdq_intermediate_outputs
                if audio_turns is None:
                    audio_turns = [
                        [] for _ in range(len(data.calibration_data.datasets))
                    ]
                if vision_turns is None:
                    vision_turns = [
                        [] for _ in range(len(data.calibration_data.datasets))
                    ]
                qdq_intermediate_outputs = [
                    [*audio_turn, *vision_turn]
                    for audio_turn, vision_turn in zip(audio_turns, vision_turns)
                ]
                self._calibrate(
                    model=self.decoder,
                    tokenizer=data.tokenizer,
                    event="convert_pt2e",
                    user_calibration_data=data.calibration_data.datasets,
                    tok_embedding=self.tok_embedding,
                    intermediate_outputs=qdq_intermediate_outputs,
                )

        # save logit's quantization attributes to meta
        self._save_logits_quant_attrs()

        # lm_head is now quantized jointly with the decoder above (real-trajectory
        # calibration via the combined module); the boundary scale is injected in compile.

        # split verification: stash the real calibration prompt/tokenizer (compile has
        # no calibration data). apply_output=True paths (monolithic, emb-only split)
        # run the generation here; headless (apply_output=False) runs it in compile
        # after the lm_head scale injection.
        if getattr(self.control_args, "verify_split", False):
            self._verify_tokenizer = data.tokenizer
            self._verify_prompt = data.calibration_data.datasets[0]
            if self.apply_output:
                self._verify_generate()

        # save output KV cache's quantization attributes to meta for attention sink
        self._save_output_kv_cache_quant_attrs()

        # setup quantized IO
        self.passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        self.passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
            "get_quant_io_dtype_fn"
        ] = partial(self._tag_ios, fixed_point_type=fixed_point_type)
        # Activate the emb-pte IO tagging. apply_output=True (multimodal / emb-split-only):
        # _tag_ios with vocab-width io_shape leaves the emb output (dim-width) fp32 —
        # unchanged option-a. Headless DFlash (apply_output=False): _tag_emb_ios tags the
        # emb output uint16 for a lossless uint16 emb->decoder boundary (the emb-output
        # scale is injected onto the decoder input in compile).
        if self.tok_embedding_passes_job is not None:
            self.tok_embedding_passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
            if self.apply_output:
                emb_tag_fn = partial(self._tag_ios, fixed_point_type=fixed_point_type)
            else:
                emb_tag_fn = partial(
                    self._tag_emb_ios,
                    fixed_point_type={"io_type": fixed_point_type["io_type"]},
                )
            self.tok_embedding_passes_job[TagQuantIO][
                QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY
            ]["get_quant_io_dtype_fn"] = emb_tag_fn
        if self.lm_head_passes_job is not None:
            self.lm_head_passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
            self.lm_head_passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                "get_quant_io_dtype_fn"
            ] = partial(
                self._tag_lm_head_ios,
                fixed_point_type={"io_type": fixed_point_type["io_type"]},
            )
        # Same tagging at the draft width. The shapes differ, so these need their own
        # partials -- the default io_shape would silently match nothing.
        if self.draft_tok_embedding_passes_job is not None:
            job = self.draft_tok_embedding_passes_job
            job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
            job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                "get_quant_io_dtype_fn"
            ] = partial(
                self._tag_emb_ios,
                fixed_point_type={"io_type": fixed_point_type["io_type"]},
                io_shape=self.draft_emb_io_shape,
            )
            job = self.draft_lm_head_passes_job
            job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
            job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                "get_quant_io_dtype_fn"
            ] = partial(
                self._tag_lm_head_ios,
                fixed_point_type={"io_type": fixed_point_type["io_type"]},
                io_shape=self.draft_lm_head_io_shape,
            )


class HybridTextDecoder(Component):
    @log_info
    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        apply_embedding: bool = False,
        output_hidden_layers: Optional[List[int]] = None,
        apply_output: bool = True,
        draft_head_ar: Optional[int] = None,
        fork_captured_hiddens: bool = False,
    ):
        # The draft-width emb/lm_head hang off the decode wrapper: that is the one
        # whose calibration loop drives the draft, and the one whose encodings the
        # prefill graphs are copied from.
        self.decode = TextDecoder(
            control_args,
            config,
            Mode.DECODE,
            apply_embedding=apply_embedding,
            output_hidden_layers=output_hidden_layers,
            apply_output=apply_output,
            draft_head_ar=draft_head_ar,
            fork_captured_hiddens=fork_captured_hiddens,
        )
        self.prefill = TextDecoder(
            control_args,
            config,
            Mode.PREFILL,
            apply_embedding=apply_embedding,
            output_hidden_layers=output_hidden_layers,
            apply_output=apply_output,
            fork_captured_hiddens=fork_captured_hiddens,
        )
        self.control_args = control_args
        self.config = config
        self.set_next(self.decode).set_next(self.prefill)

        self.apply_embedding = apply_embedding
        self.apply_output = apply_output

    # Static so the DFlash draft can reuse it: its two graphs share one KV cache
    # too, so they need identical encodings for the same reason hybrid does.
    @staticmethod
    def _encoding_override(decode_model, prefill_model):  # noqa: C901
        pbq_target = {
            torch.ops.torchao.dequantize_affine,
            torch.ops.torchao.quantize_affine,
        }
        pcq_target = {
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.quantize_per_channel.default,
        }
        ptq_target = {
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        }
        qdq_target = pbq_target | pcq_target | ptq_target

        def compare_nodes(decode_node, prefill_node):
            def info(node):
                return node.name + (
                    str(node.meta["nn_module_stack"].values())
                    if node.op == "call_function"
                    else ""
                )

            assert info(decode_node) == info(
                prefill_node
            ), f"found unmatched order for ops: {decode_node} va {prefill_node}"

        def resolve_param_target(node):
            return (
                node
                if node.op == "call_function" and node.target not in qdq_target
                else resolve_param_target(list(node.users)[0])
            )

        def activation_override(decode_node, prefill_node):
            for decode_user, prefill_user in zip(
                list(decode_node.users), list(prefill_node.users)
            ):
                assert decode_user.target == prefill_user.target, (
                    "found unmatched targets: "
                    f"{decode_user.target} vs {prefill_user.target}"
                )
                if decode_user.target in qdq_target:
                    prefill_user.args = (prefill_user.args[0], *decode_user.args[1:])
                    activation_override(decode_user, prefill_user)

        def parameter_override(decode_node, prefill_node):
            # get_attr targets can be dotted paths into submodules -- the DFlash
            # draft keeps buffers like `layers.0.self_attn.kv_rep_idx` there, where a
            # plain getattr raises and a plain setattr would silently bind the literal
            # dotted name, leaving the real buffer untouched. The lifted-constant
            # graphs this ran on before only ever had flat names.
            src = decode_model
            for part in decode_node.target.split("."):
                src = getattr(src, part)
            dst, *rest = prefill_node.target.split(".")
            obj = prefill_model
            while rest:
                obj = getattr(obj, dst)
                dst, *rest = rest
            setattr(obj, dst, src)
            # scale / zero point are part of op's attributes
            if list(decode_node.users)[0].target in ptq_target:
                activation_override(decode_node, prefill_node)

        # copy encoding for hybrid mode
        parameters = [
            {
                n: resolve_param_target(n)
                for n in model.graph.nodes
                if n.op == "get_attr"
            }
            for model in (decode_model, prefill_model)
        ]
        activations = [
            [
                n
                for n in model.graph.nodes
                if n.target not in qdq_target
                and n.op in {"call_function", "placeholder"}
            ]
            for model in (decode_model, prefill_model)
        ]
        # check topology order by node name & nn_module_stack
        for act_decode, act_prefill in zip(*activations):
            compare_nodes(act_decode, act_prefill)

        for op_decode, op_prefill in zip(*[p.values() for p in parameters]):
            compare_nodes(op_decode, op_prefill)
        # perform encoding override
        for act_decode, act_prefill in zip(*activations):
            activation_override(act_decode, act_prefill)

        for param_decode, param_prefill in zip(*[p.keys() for p in parameters]):
            parameter_override(param_decode, param_prefill)

        prefill_model.recompile()

    @log_info
    def compile(self, request: Request):  # noqa: C901
        # perform encoding override for hybrid mode
        # ---
        # theoretically decode & prefill model should share the same encoding
        # given that they are using the identical calibration dataset.
        #
        # however, pytorch will use different computaion kernels for different
        # workloads (AR1 vs ARN) which will introduce some numerical discrepancy.
        #
        # here we use a mechanism to make sure the encoding align correctly and
        # save AoT quantization time as well.
        # ---
        if (
            self.prefill.decoder is not None
            and self.prefill.model_args.use_kv_cache
            and not request.method_data[TEXT_DECODER].skip_quantize
        ):
            self._encoding_override(
                decode_model=self.decode.decoder,
                prefill_model=self.prefill.decoder,
            )
            if self.apply_embedding:
                self._encoding_override(
                    decode_model=self.decode.tok_embedding,
                    prefill_model=self.prefill.tok_embedding,
                )
                # The draft-width graph reads the same table, so the right output range
                # is the table's -- not the dozen-odd rows the draft's noise block
                # happens to touch during calibration (mask token plus one committed
                # token per round). Copying decode's encodings uses the larger sample
                # and keeps a single embeds scale for the runner.
                if self.decode.draft_tok_embedding is not None:
                    self._encoding_override(
                        decode_model=self.decode.tok_embedding,
                        prefill_model=self.decode.draft_tok_embedding,
                    )
            # lm_head had no override at all: its prefill graph carried the encodings
            # of one dummy randn forward (get_logits_out_scale differed 7.5x between
            # the two graphs). The input scale is forced below either way; this fixes
            # the weights and the output.
            if self.decode.lm_head is not None and self.prefill.lm_head is not None:
                self._encoding_override(
                    decode_model=self.decode.lm_head,
                    prefill_model=self.prefill.lm_head,
                )

        # prepare lowering tok_embedding if applicable
        if self.apply_embedding:
            tok_embedding_data = request.method_data[TOK_EMBEDDING]
            emb_models = [
                d for d in [self.decode, self.prefill] if d.tok_embedding is not None
            ]
            tok_embedding_modules = [m.tok_embedding for m in emb_models]
            tok_embedding_example_inputs = [
                m.tok_embedding_export_input for m in emb_models
            ]  # tokens
            tok_embedding_dep_tables = [m.tok_embedding_dep_table for m in emb_models]
            tok_embedding_passes_jobs = [m.tok_embedding_passes_job for m in emb_models]
            if self.decode.draft_tok_embedding is not None:
                tok_embedding_modules.append(self.decode.draft_tok_embedding)
                tok_embedding_example_inputs.append(
                    self.decode.draft_tok_embedding_export_input
                )
                tok_embedding_dep_tables.append(
                    self.decode.draft_tok_embedding_dep_table
                )
                tok_embedding_passes_jobs.append(
                    self.decode.draft_tok_embedding_passes_job
                )
            tok_embedding_graph_names = TOK_EMBEDDING_GRAPH_NAMES[
                : len(tok_embedding_modules)
            ]

        # prepare lowering decoder
        data = request.method_data[TEXT_DECODER]
        models = [d for d in [self.decode, self.prefill] if d.decoder is not None]
        example_inputs = [m.export_input for m in models if m is not None]
        graph_names = DECODER_GRAPH_NAMES[: len(models)]

        # start lowering
        if self.apply_embedding:
            tok_embedding_edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                module=dict(zip(tok_embedding_graph_names, tok_embedding_modules)),
                inputs=dict(
                    zip(tok_embedding_graph_names, tok_embedding_example_inputs)
                ),
                compiler_specs=dict(
                    zip(tok_embedding_graph_names, tok_embedding_data.compile_spec)
                ),
                dep_table=dict(
                    zip(tok_embedding_graph_names, tok_embedding_dep_tables)
                ),
                passes_job=dict(
                    zip(tok_embedding_graph_names, tok_embedding_passes_jobs)
                ),
            )
            if self.control_args.verbose:
                for ep in tok_embedding_edge_prog_mgr._edge_programs.values():
                    print_delegation_info(ep.graph_module)

            executorch_config = ExecutorchBackendConfig(
                # For shared buffer, user must pass the memory address
                # which is allocated by RPC memory to executor runner
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False,
                    alloc_graph_output=False,
                ),
                passes=[BuildQuantIo()],
            )
            tok_embedding_exec_prog_mgr = tok_embedding_edge_prog_mgr.to_executorch(
                executorch_config
            )
            data = request.method_data[TOK_EMBEDDING]
            with open(
                f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
            ) as file:
                tok_embedding_exec_prog_mgr.write_to_file(file)

        # prepare lowering lm_head (headless: vocab projection in a separate shared
        # pte). Force its input-activation scale to the decoder's output-hidden
        # scale so the decoder -> lm_head pte boundary is lossless (both uint16).
        # DFlash headless targets don't pass LM_HEAD (the runner projects via the
        # embed table / a shared lm_head.pte), so skip the split there.
        if not self.apply_output and LM_HEAD in request.method_data:
            lm_head_data = request.method_data[LM_HEAD]
            lm_head_models = [
                d for d in [self.decode, self.prefill] if d.lm_head is not None
            ]
            hidden_scale = self.decode.meta["get_logits_scale"]
            hidden_zp = self.decode.meta["get_logits_zero_point"]
            lm_head_modules = [m.lm_head for m in lm_head_models]
            lm_head_example_inputs = [m.lm_head_export_input for m in lm_head_models]
            lm_head_dep_tables = [m.lm_head_dep_table for m in lm_head_models]
            lm_head_passes_jobs = [m.lm_head_passes_job for m in lm_head_models]
            for m in lm_head_models:
                m._override_lm_head_input_scale(hidden_scale, hidden_zp)
            if self.decode.draft_lm_head is not None:
                # The draft-width graph takes decode's encodings wholesale rather than
                # keeping the ones it calibrated on draft traffic. Its own were finer
                # (hidden 0.58x, logits 0.65x of the target's), but device A/B showed
                # them to be decision-neutral -- same pte, 72 rounds and 437 tokens
                # either way -- so keeping them only adds a variable to every
                # comparison. Run AFTER the input injection above so the draft graph
                # inherits that too, and re-read the getters, which quantize() filled
                # from the pre-override graph.
                self._encoding_override(
                    decode_model=self.decode.lm_head,
                    prefill_model=self.decode.draft_lm_head,
                )
                self.decode._save_draft_head_quant_attrs()
                lm_head_modules.append(self.decode.draft_lm_head)
                lm_head_example_inputs.append(self.decode.draft_lm_head_export_input)
                lm_head_dep_tables.append(self.decode.draft_lm_head_dep_table)
                lm_head_passes_jobs.append(self.decode.draft_lm_head_passes_job)
            lm_head_graph_names = LM_HEAD_GRAPH_NAMES[: len(lm_head_modules)]
            if getattr(self.control_args, "verify_split", False):
                # headless split path: real generation tokens -> [emb] -> decoder ->
                # lm_head -> logits, logging the decoded text for coherence check.
                self.decode._verify_generate()

            lm_head_edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                module=dict(zip(lm_head_graph_names, lm_head_modules)),
                inputs=dict(zip(lm_head_graph_names, lm_head_example_inputs)),
                compiler_specs=dict(
                    zip(lm_head_graph_names, lm_head_data.compile_spec)
                ),
                dep_table=dict(zip(lm_head_graph_names, lm_head_dep_tables)),
                passes_job=dict(zip(lm_head_graph_names, lm_head_passes_jobs)),
            )
            if self.control_args.verbose:
                for ep in lm_head_edge_prog_mgr._edge_programs.values():
                    print_delegation_info(ep.graph_module)

            executorch_config = ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False,
                    alloc_graph_output=False,
                ),
                passes=[BuildQuantIo()],
            )
            lm_head_exec_prog_mgr = lm_head_edge_prog_mgr.to_executorch(
                executorch_config
            )
            with open(
                f"{self.control_args.artifact}/{lm_head_data.pte_filename}.pte", "wb"
            ) as file:
                lm_head_exec_prog_mgr.write_to_file(file)

        # emb -> decoder boundary: inject the emb-output scale onto both decoders'
        # inputs_embeds so the uint16 boundary is lossless (mirror of the lm_head
        # injection above, direction reversed: emb output is the source). Headless
        # emb-split only; _encoding_override already ran, so patch decode + prefill.
        if self.apply_embedding and not self.apply_output:
            embeds_scale = self.decode.meta["get_embeds_scale"]
            embeds_zp = self.decode.meta["get_embeds_zero_point"]
            for m in models:
                m._override_decoder_input_scale(embeds_scale, embeds_zp)

        # decoder lowering
        data = request.method_data[TEXT_DECODER]
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=dict(zip(graph_names, [model.decoder for model in models])),
            inputs=dict(zip(graph_names, example_inputs)),
            compiler_specs=dict(zip(graph_names, data.compile_spec)),
            constant_methods={**self.decode.meta},
            dep_table=dict(zip(graph_names, [model.dep_table for model in models])),
            passes_job=dict(zip(graph_names, [model.passes_job for model in models])),
            skip_node_op_set={"llama.fallback.default"},
        )

        if self.config.num_sharding > 1:
            for graph_name in graph_names:
                update_spill_fill_size(edge_prog_mgr.exported_program(graph_name))

        if self.control_args.verbose:
            for ep in edge_prog_mgr._edge_programs.values():
                print_delegation_info(ep.graph_module)

        executorch_config = ExecutorchBackendConfig(
            # For shared buffer, user must pass the memory address
            # which is allocated by RPC memory to executor runner
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
            passes=[BuildQuantIo()],
        )
        exec_prog_mgr = edge_prog_mgr.to_executorch(executorch_config)
        data = request.method_data[TEXT_DECODER]
        with open(
            f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
        ) as file:
            exec_prog_mgr.write_to_file(file)


class Modality(Component):
    def __init__(
        self, control_args: argparse.Namespace, config: LLMModelConfig, modality
    ):
        self.control_args = control_args
        self.model = None
        self.modality = modality
        repo_id = config.repo_id

        if config := getattr(config, modality, None):
            if modality == AUDIO_ENCODER:
                auto_model = AutoModelForSpeechSeq2Seq.from_pretrained(repo_id)
                self.num_layers = auto_model.config.encoder_config.num_layers
                self.ctx_size = auto_model.config.encoder_config.context_size
            elif modality == TEXT_ENCODER:
                raise NotImplementedError(f"{modality} is under development")
            elif modality == VISION_ENCODER:
                auto_model = AutoModel.from_pretrained(
                    repo_id, _attn_implementation="eager"
                )
                self.num_layers = auto_model.config.vision_config.num_hidden_layers
            else:
                raise NotImplementedError(f"Find no {modality}")

            auto_model = auto_model.to(torch.float32).eval()
            self.model = config().create_encoder(auto_model.config).eval()
            self.model.load_state_dict(
                auto_model.state_dict(), strict=False
            )  # set strict to false to simplify parameter loading for non-text models
            self.example_input = self.model.get_example_inputs()

            # set quant recipe
            self.quant_recipe: EncoderQuantRecipe = (
                config.quant_recipe(True) if config.quant_recipe else None
            )

            # metadata
            self.config = config

        self.passes_job = get_capture_program_passes()
        self.dep_table = get_passes_dependency_for_capture_program()

    def _tag_ios(self, node, fixed_point_type):
        quant_io_type = None

        # tag sharding io
        if exir_ops.edge.llama.fallback.default in [
            u.target for u in list(node.users.keys())
        ] + [node.target]:
            quant_io_type = fixed_point_type["io_type"]

        # GraniteSpeech: tag _to_copy op as quantized tensors for attn dist. It is caused by sharding
        if (
            issubclass(self.config, GraniteSpeechEncoder)
            and node.target == exir_ops.edge.aten._to_copy.default
            and node.meta["val"].size() == (self.ctx_size, self.ctx_size)
        ):
            quant_io_type = torch.int32

        return quant_io_type

    def _get_sharding_get_pattern(self):
        prefixes = [
            "encoder.layers",
            "vision_tower.encoder.layer",
            "vision_model.encoder.layers",
        ]
        prefix_alt = "|".join(re.escape(p) for p in prefixes)
        return rf"^(?:{prefix_alt})\.(\d+)"

    def compile(self, request: Request):
        if self.model is None:
            return

        request_data = request.method_data[self.modality]
        # check if sharding required
        if self.config.num_sharding > 1:
            SplitGraph, setting = model_sharding.get_split_graph_pass(
                self.num_layers,
                shares=self.config.num_sharding,
                pattern=self._get_sharding_get_pattern(),
            )
            self.passes_job[SplitGraph] = setting
            self.dep_table[SplitGraph] = [FoldQDQ]
            self.dep_table[TagQuantIO] = [SplitGraph]

            if not request_data.skip_quantize:
                fixed_point_type = {"io_type": torch.uint16}

                # setup quantized IO
                self.passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
                self.passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                    "get_quant_io_dtype_fn"
                ] = partial(self._tag_ios, fixed_point_type=fixed_point_type)

        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=self.model,
            inputs=self.example_input,
            compiler_specs=request_data.compile_spec,
            dep_table=self.dep_table,
            passes_job=self.passes_job,
            skip_node_op_set={"llama.fallback.default"},
        )

        if self.config.num_sharding > 1:
            update_spill_fill_size(edge_prog_mgr.exported_program())
        if self.control_args.verbose:
            print_delegation_info(edge_prog_mgr.exported_program().graph_module)

        exec_prog_mgr = edge_prog_mgr.to_executorch(
            ExecutorchBackendConfig(passes=[BuildQuantIo()])
        )
        data = request.method_data[self.modality]
        with open(
            f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
        ) as file:
            exec_prog_mgr.write_to_file(file)

    def _calibrate(self, model, calibration_datasets):
        outputs = []
        for turn in calibration_datasets:
            outputs_each_turn = [model(*data) for data in turn]
            outputs.append(outputs_each_turn)
        return outputs

    def quantize(self, request: Request):
        if self.model is None:
            return

        request_data = request.method_data[self.modality]
        calibration_datasets = request_data.calibration_data.datasets

        with torch.no_grad():
            self.model = torch.export.export(self.model, self.example_input).module()

            if request_data.skip_quantize:
                logging.info(f"skipping encoder quantization for {self.modality}")
                intermediate_outputs = self._calibrate(self.model, calibration_datasets)
                request_data.calibration_data.intermediate_outputs = (
                    intermediate_outputs
                )
                return

            quantizer = make_quantizer(
                backend=request_data.backend, soc_model=request_data.soc_model
            )
            quantizer.set_recipe(self.quant_recipe.recipe)
            self.model = prepare_pt2e(self.model, quantizer)

            # start calibration
            intermediate_outputs = self._calibrate(self.model, calibration_datasets)
            request_data.calibration_data.intermediate_outputs = intermediate_outputs

            self.model = convert_pt2e(self.model)

            # QDQ inference
            if self.control_args.verbose:
                qdq_intermediate_outputs = self._calibrate(
                    self.model, calibration_datasets
                )
                request_data.calibration_data.qdq_intermediate_outputs = (
                    qdq_intermediate_outputs
                )


class MultiModalManager(Component):
    def __init__(self, control_args: argparse.Namespace, config: LLMModelConfig):
        self.audio_encoder = Modality(
            control_args,
            config,
            AUDIO_ENCODER,
        )
        self.text_encoder = Modality(
            control_args,
            config,
            TEXT_ENCODER,
        )
        self.vision_encoder = Modality(
            control_args,
            config,
            VISION_ENCODER,
        )
        self.text_decoder = HybridTextDecoder(
            control_args,
            config,
            # multimodal always splits the embedding; --split_embedding is the text
            # dev/test hook that moves the embedding to a separate shared pte.
            apply_embedding=bool(self.audio_encoder.model or self.vision_encoder.model)
            or getattr(control_args, "split_embedding", False),
            # dev/test hook: --headless_decoder makes a plain text compile drop the
            # in-graph lm_head (output = final hidden) to exercise the headless path.
            apply_output=not getattr(control_args, "headless_decoder", False),
        )
        self._modalities = [
            AUDIO_ENCODER,
            TEXT_ENCODER,
            VISION_ENCODER,
            TOK_EMBEDDING,
            LM_HEAD,
            TEXT_DECODER,
        ]
        # build dependency chain
        self.set_next(self.vision_encoder).set_next(self.audio_encoder).set_next(
            self.text_decoder
        )

    def process(self, request: Request) -> Request:
        Processor.process(self, request)

    @log_info
    def compile(
        self,
        compile_specs: Dict[str, List[CompileSpec]],
        pte_filenames: Dict[str, str],
        skip_quantize: Dict[str, bool],
    ):
        compile_request = Request(
            inspect.currentframe().f_code.co_name,
            {
                m: Request.Data(
                    compile_spec=compile_specs[m],
                    pte_filename=pte_filenames[m],
                    skip_quantize=skip_quantize[m] if m in skip_quantize else False,
                )
                for m in self._modalities
            },
        )
        self.process(compile_request)

    @log_info
    def quantize(
        self,
        calibration_data: Dict[str, List[Any]],
        skip_quantize: Dict[str, bool],
        tokenizer,
        backend,
        soc_model,
    ):
        quantize_request = Request(
            inspect.currentframe().f_code.co_name,
            {
                m: Request.Data(
                    calibration_data=Request.CalibrationData(
                        datasets=calibration_data[m]
                    ),
                    skip_quantize=skip_quantize.get(m, False),
                    tokenizer=tokenizer,
                    backend=backend,
                    soc_model=soc_model,
                )
                for m in self._modalities
            },
        )
        self.process(quantize_request)
