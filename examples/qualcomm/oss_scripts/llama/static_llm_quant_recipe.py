# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from executorch.backends.qualcomm.quantizer.custom_annotation import (
    annotate_draft_shared_head,
    annotate_kv_8bit,
)
from executorch.backends.qualcomm.quantizer.quant_recipe import (
    QuantGranularity,
    QuantRecipe,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from torchao.quantization.pt2e import MinMaxObserver


class StaticLLMQuantRecipe:
    """
    Qualcomm's static LLaMA quantization recipe.
    """

    def __init__(self):
        self.recipe: Optional[QuantRecipe] = None

        # For IO bitwidth
        self.default_quant_dtype = getattr(self, "default_quant_dtype", None)
        if self.default_quant_dtype is None:
            raise ValueError("default_quant_dtype must be defined in the recipe.")

    def get_kv_io_bit_width(self) -> int:
        if self.default_quant_dtype is None:
            return 32
        elif (
            self.default_quant_dtype == QuantDtype.use_8a8w
            or annotate_kv_8bit in self.recipe.custom_quant_annotations
        ):
            return 8
        else:
            # If quantized but not 8a8w or mix_quantization, it has to be 16bit kv io.
            return 16

    def get_lm_head_quant_dtype(self) -> QuantDtype:
        """Weight/activation width for a SPLIT lm_head.

        When the decoder is headless the lm_head is a separate module with its own
        quantizer (llm_wrappers), so it never sees `self.recipe` -- the `output.conv`
        regex only applies when the head is in-graph. This hook is the only knob
        that reaches it.
        """
        return QuantDtype.use_16a8w

    def get_logits_output_bit_width(self) -> int:
        # We use 16bit logits for all quant config
        return 32 if self.default_quant_dtype is None else 16


class LlamaStories260KQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()
        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
            .add_regex(
                {r"output\.conv"},
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class LlamaStories110MQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
            .add_regex(
                {r"layers\..*\.attention\.wv.*"},
                QuantDtype.use_8a4w,
                False,
                act_observer=MinMaxObserver,
                act_symmetric=True,
                granularity=QuantGranularity.PER_CHANNEL,
            )
            .add_regex(
                {r"output\.conv"},
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Llama3_1BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w_block

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
                note="default with 16bit activation",
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
                note="Annotate with 16a4w block quantization since these layers are not sensitive.",
            )
            .add_regex(
                {
                    r"output\.conv",
                    r"layers\.[0-3]\.feed_forward\.w2_conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
                note="Down proj layer is sensitive and should be annotated with 16a8w.",
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Llama3_3BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w_block

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
            )
            .add_regex(
                {
                    r"output\.conv",
                    r"layers\.2[1-7]\.feed_forward\.w2_conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class CodegenQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        ).add_node_target(
            {
                torch.ops.aten.conv2d.default,
            },
            QuantDtype.use_16a8w,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
        )


class Gemma_2BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 64, 1, 1)},
            )
            .add_regex(
                {
                    r"layers\..*\.attention\.wv.*",
                    r"output\.conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Gemma2QuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
            )
            .add_regex(
                {
                    r"layers\..*\.attention\.wv.*",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Gemma3QuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 64, 1, 1)},
            )
            .add_regex(
                {
                    r"layers\..*\.attention\.wv.*",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class GLM_1_5B_InstructQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
            )
            .add_regex(
                {r"output\.conv"},
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Granite_3_3_2B_InstructQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 64, 1, 1)},
            )
            .add_regex(
                {
                    r"layers\..*\.attention\.wv.*",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class GraniteSpeech_3_3_2B_InstructQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 64, 1, 1)},
            )
            .add_regex(
                {
                    r"layers\..*\.attention\.wv.*",
                    r"layers\..*\.feed_forward\.w3_conv",
                    r"layers\..*\.feed_forward\.w2_conv",
                    r"output\.conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )


class InternVL3_1B_QuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        ).add_node_target(
            {
                torch.ops.aten.conv2d.default,
            },
            QuantDtype.use_16a8w,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
        )


class Phi4MiniQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 16, 1, 1)},
            )
            .add_regex(
                {r"layers\..*\.attention\.wv.*"},
                QuantDtype.use_8a4w,
                False,
                act_observer=MinMaxObserver,
                act_symmetric=True,
                granularity=QuantGranularity.PER_CHANNEL,
            )
            .add_regex(
                {r"output\.conv"},
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Qwen2_5_0_5BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        ).add_node_target(
            {
                torch.ops.aten.conv2d.default,
            },
            QuantDtype.use_16a4w_block,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_BLOCK,
            extra_kwargs={"block_size": (1, 16, 1, 1)},
        )


class Qwen2_5_1_5BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 16, 1, 1)},
            )
            .add_regex(
                {r"output\.conv"},
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )


class Qwen3_0_6BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
            )
            .add_regex(
                {
                    r"layers\..*\.feed_forward\.w2_conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )


class Qwen3_1_7BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 16, 1, 1)},
            )
            .add_regex(
                {
                    r"output\.conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Qwen3_4BQuantRecipe(StaticLLMQuantRecipe):
    """16a4w recipe for Qwen3-4B (36 layers, hidden 2560).

    Based on the 1.7B recipe (16a4w default, per-block conv2d, 16a8w KV) but,
    for the larger model, additionally keeps the MLP down_proj (feed_forward.w2)
    at 16a8w per-channel. down_proj writes into the residual stream, so this both
    improves general 4B accuracy (it is the classic outlier-heavy PTQ layer) and
    preserves the residual-stream hidden states that the DFlash draft consumes
    (layers 1/9/17/25/33) — those hidden states directly drive draft accept.
    If accept is still low, next levers: protect attention.wo, or shrink the
    conv2d block_size (finer per-block scales).
    """

    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 16, 1, 1)},
            )
            .add_regex(
                {
                    r"output\.conv",
                    r"layers\..*\.feed_forward\.w2_conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Qwen3_4B_Head4wQuantRecipe(StaticLLMQuantRecipe):
    """Qwen3-4B with lm_head at 16a4w instead of 16a8w.

    Identical to Qwen3_4BQuantRecipe except that `output.conv` keeps 4-bit
    weights. It is annotated explicitly rather than dropped from the 16a8w regex:
    dropping it would fall through to the conv2d PER_BLOCK rule, which adds 160
    scales per row (~48.6 MB) and changes granularity as well as bit width.
    Keeping PER_CHANNEL makes bit width the only variable against the 16a8w build.

    lm_head is 9.7% of the parameters but 19.5% of the weight bytes, because it
    is the one large tensor the recipe protects at 8-bit while the body runs at
    4-bit. DFlash reads it twice per round (draft marginals, then verification),
    so this build exists to price that asymmetry against sparse verification,
    which removes the same bytes without touching accuracy.
    """

    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 16, 1, 1)},
            )
            .add_regex(
                {
                    r"layers\..*\.feed_forward\.w2_conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
            .add_regex(
                {
                    r"output\.conv",
                },
                QuantDtype.use_16a4w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)

    def get_lm_head_quant_dtype(self) -> QuantDtype:
        return QuantDtype.use_16a4w


class Qwen3_8BQuantRecipe(Qwen3_4BQuantRecipe):
    """Qwen3-8B (36 layers, hidden 4096) takes the 4B rules unchanged: 16a4w LPBQ block 16,
    MLP down_proj and lm_head 16a8w per-channel, 8-bit KV."""


class DFlashDraftQuantRecipe(StaticLLMQuantRecipe):
    """Uniform-width recipe for the DFlash draft (block-speculative head).

    Every conv2d (q/k/v/o/gate/up/down_proj and fc) gets `w_bits` weights and
    `a_bits` activations; fc alone can be given its own width via `fc_bits` (it sums
    num_ctx_layers*H target-hidden values including the attention-sink outlier, host
    measured peak |fc(x)| up to ~281k). The draft shares the target's emb/lm_head, so
    there is nothing else to quantize. convert_linear_to_conv2d turns each nn.Linear
    into a Conv2D wrapper whose inner conv is `<name>.conv`, hence the regex names.

    4-bit weights are per-block (LPBQ, block 16) when activations are 16-bit, or
    per-channel with `w4_per_channel`; QNN has no 8-bit-activation block scheme and no
    8a16w, so 8-bit activations are always per-channel and 16-bit weights with 8-bit
    activations is rejected.
    """

    CONV_DTYPE = {
        (16, 4): (QuantDtype.use_16a4w_block, QuantGranularity.PER_BLOCK),
        (16, 8): (QuantDtype.use_16a8w, QuantGranularity.PER_CHANNEL),
        (16, 16): (QuantDtype.use_16a16w, QuantGranularity.PER_CHANNEL),
        (8, 4): (QuantDtype.use_8a4w, QuantGranularity.PER_CHANNEL),
        (8, 8): (QuantDtype.use_8a8w, QuantGranularity.PER_CHANNEL),
    }

    def __init__(
        self,
        verbose: bool = False,
        w_bits: int = 4,
        fc_bits: Optional[int] = None,
        a_bits: int = 16,
        w4_per_channel: bool = False,
        fc_per_channel: Optional[bool] = None,
        fc_block_size: Optional[int] = None,
        lpbq_groups: Optional[tuple] = None,
        pc_modules: Optional[tuple] = None,
    ):
        self.default_quant_dtype = (
            QuantDtype.use_16a4w if a_bits == 16 else QuantDtype.use_8a8w
        )
        super().__init__()
        fc_bits = w_bits if fc_bits is None else fc_bits

        def conv_rule(w, per_channel=False, block=16):
            if (a_bits, w) not in self.CONV_DTYPE:
                raise ValueError(f"QNN has no {a_bits}a{w}w quantization")
            dtype, granularity = self.CONV_DTYPE[(a_bits, w)]
            if w == 4 and per_channel and a_bits == 16:
                dtype, granularity = QuantDtype.use_16a4w, QuantGranularity.PER_CHANNEL
            extra = (
                {"block_size": (1, block, 1, 1)}
                if granularity == QuantGranularity.PER_BLOCK
                else None
            )
            return dtype, False, MinMaxObserver, granularity, extra

        def add(builder, target, w, per_channel=False, block=16):
            dtype, qat, observer, granularity, extra = conv_rule(w, per_channel, block)
            return builder(
                target,
                dtype,
                qat,
                act_observer=observer,
                granularity=granularity,
                extra_kwargs=extra,
            )

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        )
        add(self.recipe.add_node_target, {torch.ops.aten.conv2d.default}, w_bits, w4_per_channel)
        fc_pc = w4_per_channel if fc_per_channel is None else fc_per_channel
        if fc_bits != w_bits or fc_pc != w4_per_channel or fc_block_size is not None:
            add(self.recipe.add_regex, {r"fc\.conv"}, fc_bits, fc_pc, fc_block_size or 16)
        # Diagnostics: the body is per-channel except for the named module groups, which stay LPBQ.
        group_regex = {
            "attn": r"layers\.\d+\.self_attn\.(q|k|v|o)_proj\.conv",
            "gate_up": r"layers\.\d+\.mlp\.(gate|up)_proj\.conv",
            "down": r"layers\.\d+\.mlp\.down_proj\.conv",
        }
        for g in lpbq_groups or ():
            add(self.recipe.add_regex, {group_regex[g]}, w_bits, False)
        for rx in pc_modules or ():     # diagnostics: force these modules to per-channel
            add(self.recipe.add_regex, {rx}, w_bits, True)
        # Only present when the draft keeps an in-graph lm_head; harmless (no match)
        # when it shares the target's lm_head.pte.
        add(self.recipe.add_regex, {r"lm_head\.conv"}, 8)
        # Quantize the K/V cache to 8-bit IN-GRAPH (same as Qwen3_4BQuantRecipe).
        # Without this the draft's KV is 16a4w's default 16-bit activation, and
        # tagging the pte's KV IO as u8 at lower time then reuses the 16-bit scale
        # on an 8-bit tensor -- 255/256 of the range is clipped and acceptance
        # collapses. With it, the observers calibrate KV at 8-bit, so the in-graph
        # math and the u8 boundary agree and the calibration/inference trajectories
        # match. get_kv_io_bit_width() also flips 16 -> 8 accordingly.
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)
        # [shared-head experiment] gate behind DFLASH_SHARE_HEAD: default keeps per-head
        # scales; set the env to make each layer's 32 per-head attention activations
        # share one scale (recovers MHA accept on the SHA graph, no QNN 1003).
        import os

        if os.environ.get("DFLASH_SHARE_HEAD"):
            self.recipe.custom_quant_annotations.append(annotate_draft_shared_head)


class Smollm2QuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        ).add_node_target(
            {
                torch.ops.aten.conv2d.default,
            },
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
        )


class Smollm3QuantRecipe(StaticLLMQuantRecipe):

    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
            )
            .add_regex(
                {
                    r"layers\..*\.attention\.wq.*",
                    r"layers\..*\.attention\.wk.*",
                    r"layers\..*\.attention\.wv.*",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
            .add_regex(
                {
                    r"output\.conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class SmolVLMQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        ).add_node_target(
            {
                torch.ops.aten.conv2d.default,
            },
            QuantDtype.use_16a8w,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
        )
