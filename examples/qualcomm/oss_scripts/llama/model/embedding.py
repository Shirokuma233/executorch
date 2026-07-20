# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        input_embedding_module,
        max_batch_size: int,
        ar_len: int,
        vocab_size: int,
        dim: int,
        use_i64_token: bool,
    ):
        super().__init__()
        self.input_embedding_module = input_embedding_module
        self.max_batch_size = max_batch_size
        self.ar_len = ar_len
        self.vocab_size = vocab_size
        self.dim = dim
        self.use_i64_token = use_i64_token

    def get_example_input(self):
        return (
            torch.randint(
                self.vocab_size,
                (self.max_batch_size, self.ar_len),
                dtype=torch.int64 if self.use_i64_token else torch.int32,
            ),
        )

    def forward(self, tokens):
        return self.input_embedding_module(tokens)


class LmHead(nn.Module):
    """Standalone lm_head (hidden -> logits), the counterpart to TokenEmbedding, so
    the vocab projection can live in a separate shared pte. `output_module` is the
    decoder's `self.output` (nn.Linear(dim, vocab)); it goes through the same
    convert_linear_to_conv2d + 16a8w-per-channel path as the in-graph lm_head, so
    the split reuses the identical quantization. Its input (hidden) must be quantized
    with the headless decoder's output-hidden scale for the boundary to be lossless."""

    def __init__(
        self,
        output_module,
        max_batch_size: int,
        ar_len: int,
        vocab_size: int,
        dim: int,
    ):
        super().__init__()
        self.output = output_module
        self.max_batch_size = max_batch_size
        self.ar_len = ar_len
        self.vocab_size = vocab_size
        self.dim = dim

    def get_example_input(self):
        return (
            torch.randn(
                (self.max_batch_size, self.ar_len, self.dim), dtype=torch.float32
            ),
        )

    def forward(self, hidden_states):
        return self.output(hidden_states)
