# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import heapq
import logging
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
from executorch.backends.qualcomm._passes import SeqMSE
from executorch.examples.models.llama.evaluate.eager_eval import EagerEvalWrapper
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import AttentionMask

from executorch.exir._serialize._program import deserialize_pte_binary
from pytorch_tokenizers.hf_tokenizer import HuggingFaceTokenizer
from pytorch_tokenizers.llama2c import Llama2cTokenizer as SentencePieceTokenizer
from pytorch_tokenizers.tiktoken import TiktokenTokenizer

try:
    from lm_eval.evaluator import simple_evaluate
except ImportError:
    raise ImportError(
        "Please install the llm eval dependency via examples/models/llama/install_requirements.sh"
    )


INFERENCE_REGISTRY = {}


def register_inference(use_kv_cache: bool):
    def decorator(func):
        INFERENCE_REGISTRY[use_kv_cache] = func

    return decorator


def _modality_inputs_merger(
    input_ids: torch.LongTensor,
    inputs_embeds: torch.Tensor,
    image_hidden_states: torch.Tensor,
    image_token_id,
):
    """
    This method aims at merging the token embeddings with the image hidden states into one single sequence of vectors that are fed to the transformer LM.
    The merging happens as follows:
    - The text token sequence is: `tok_1 tok_2 tok_3 <fake_token_around_image> <image> <image> ... <image> <fake_token_around_image> tok_4`.
    - We get the image hidden states for the image through the vision encoder and that hidden state, after a pixel shuffle operation, is then projected into the text embedding space.
    We thus have a sequence of image hidden states of size (1, image_seq_len, hidden_dim), where 1 is for batch_size of 1 image and hidden_dim is the hidden_dim of the LM transformer.
    - The merging happens so that we obtain the following sequence: `vector_tok_1 vector_tok_2 vector_tok_3 vector_fake_tok_around_image {sequence of image_seq_len image hidden states} vector_fake_toke_around_image vector_tok_4`. That sequence is fed to the LM.
    - To fit the format of that sequence, `input_ids`, `input_embeds`, `attention_mask` are all 3 adapted to insert the image hidden states.
    """

    special_image_mask = input_ids == image_token_id
    special_image_mask = (
        special_image_mask.unsqueeze(-1)
        .expand_as(inputs_embeds)
        .to(inputs_embeds.device)
    )
    image_hidden_states = image_hidden_states.to(
        inputs_embeds.device, inputs_embeds.dtype
    )
    inputs_embeds = inputs_embeds.masked_scatter(
        special_image_mask, image_hidden_states
    )
    return inputs_embeds


@dataclass
class DecoderInputs:
    all_pos: torch.Tensor
    atten_mask: AttentionMask
    input_ids: Optional[torch.Tensor] = None
    input_ids_dtype: Optional[torch.dtype] = None
    embedding: Optional[torch.Tensor] = None


class GraphModuleCalibrationWrapper(EagerEvalWrapper):
    """
    A wrapper class for calibration
    """

    def __init__(  # noqa: C901
        self,
        model: torch.fx.GraphModule,
        tokenizer: Union[
            SentencePieceTokenizer, TiktokenTokenizer, HuggingFaceTokenizer
        ],
        max_seq_length: int,
        ar_len: int,
        use_kv_cache: bool,
        get_example_inputs: Callable,
        use_i64_token: bool,
        seq_mse_candidates: int,
    ):
        # n seq len = n-1 cache len, so we len(inps) = n-1 during _model_call
        assert max_seq_length is not None, "max_seq_length must be provided"
        super().__init__(
            model=model, tokenizer=tokenizer, max_seq_length=max_seq_length - 1
        )
        self._model = model.to(self.device)
        self.ar_len = ar_len
        self._use_kv_cache = use_kv_cache
        self.get_example_inputs = get_example_inputs
        self.max_seq_length = max_seq_length
        self.use_i64_token = use_i64_token
        self.seq_mse_candidates = seq_mse_candidates
        self._input_samples = None

    def get_input_samples(self):
        return self._input_samples

    def _model_call(self, inps):
        all_logits = None
        kwargs = {}
        if self._use_kv_cache:
            kwargs["ar_len"] = self.ar_len
            kwargs["seq_mse_candidates"] = self.seq_mse_candidates

        all_logits, self._input_samples = INFERENCE_REGISTRY[self._use_kv_cache](
            self.get_example_inputs,
            inps,
            self._model,
            self._tokenizer,
            max_seq_len=self.max_seq_length,
            use_i64_token=self.use_i64_token,
            collect_logits=True,
            **kwargs,
        )
        # one shot is enough for seq mse
        self.seq_mse_candidates = 0
        return all_logits


class LookaheadDecoder:
    """
    Lookahead decoding to speed up calibration
    """

    class NgramPool:
        def __init__(self, num_verifications: int):
            self.pool = defaultdict(OrderedDict)
            # keep the amount of ngrams as number of verification branches for simplicity
            self.num_verifications = num_verifications

        def add(self, ngram: Tuple[int]):
            key = ngram[0]
            # since there is no OrderedSet in python, use OrderedDict with dummy value 1
            self.pool[key][ngram[1:]] = 1
            if len(self.pool[key]) > self.num_verifications:
                # remove cache in FIFO fashion
                self.pool[key].popitem(last=False)

        def __getitem__(self, key):
            return self.pool[key]

        def __iter__(self):
            return iter(self.pool)

    def __init__(
        self,
        window_size: int,
        ngram_size: int,
        num_verifications: int,
        ar_size: int,
        mask_value: int,
    ):
        if ar_size < (ngram_size - 1) * (window_size + num_verifications):
            raise ValueError(
                "AR length is not enough to meet requirement. "
                "Should be at least (ngram_size - 1) * (window_size + num_verifications)."
            )

        self.window_size = window_size
        self.ngram_size = ngram_size
        self.ngram_pool = self.NgramPool(num_verifications)
        self.num_verifications = num_verifications
        self.verification_offset = window_size * (ngram_size - 1)
        self.ar_size = ar_size
        self.mask_value = mask_value

    @property
    def attention_mask(self) -> torch.Tensor:
        mask = torch.full((self.ar_size,) * 2, self.mask_value)
        lookahead_branch_mask = torch.triu(
            torch.full((self.window_size,) * 2, self.mask_value),
            diagonal=1,
        )
        for i in range(self.ngram_size - 1):
            mask[
                i * self.window_size : (i + 1) * self.window_size,
                : self.window_size,
            ] = lookahead_branch_mask
            for j in range(1, i + 1):
                mask[
                    i * self.window_size : (i + 1) * self.window_size,
                    j * self.window_size : (j + 1) * self.window_size,
                ].fill_diagonal_(0)

        verification_branch_mask = torch.triu(
            torch.full((self.ngram_size - 1,) * 2, self.mask_value),
            diagonal=1,
        )
        for i in range(self.num_verifications):
            indices = [i * (self.ngram_size - 1), (i + 1) * (self.ngram_size - 1)]
            slices = (slice(*[ind + self.verification_offset for ind in indices]),) * 2
            mask[slices] = verification_branch_mask
        mask[
            : self.verification_offset + (self.ngram_size - 1) * self.num_verifications,
            0,
        ] = 0

        return mask

    @property
    def position_offset(self) -> torch.Tensor:
        offsets = torch.zeros(self.ar_size, dtype=torch.int32)
        idx = 0
        # lookahead branches
        for i in range(self.ngram_size - 1):
            for j in range(self.window_size):
                offsets[idx] = i + j
                idx += 1

        # verification branches
        for _ in range(self.num_verifications):
            for j in range(1, self.ngram_size):
                offsets[idx] = j
                idx += 1

        return offsets

    def update_verification_branch(self, guess_token: int, inputs: List[int]) -> None:
        for branch, ngram in enumerate(self.ngram_pool[guess_token]):
            verification_offset = self.verification_offset + branch * (
                self.ngram_size - 1
            )
            for i, token in enumerate(ngram):
                inputs[verification_offset + i] = token

    def update_lookahead_branch(self, inputs: List[int], outputs: List[int]) -> None:
        # 1 level shifting
        for i in range(self.ngram_size - 2):
            for j in range(self.window_size):
                inputs[self.window_size * i + j] = inputs[
                    self.window_size * (i + 1) + j
                ]

        last_ngram_offset = self.window_size * (self.ngram_size - 2)
        for i in range(self.window_size):
            inputs[last_ngram_offset + i] = outputs[last_ngram_offset + i]

    def update_ngram_pool(self, inputs: List[int], outputs: List[int]) -> None:
        for i in range(self.window_size):
            ngram = [inputs[i]]
            for j in range(1, self.ngram_size - 1):
                ngram.append(inputs[i + j * self.window_size])

            ngram.append(outputs[i + self.window_size * (self.ngram_size - 2)])
            self.ngram_pool.add(tuple(ngram))

    def verify(
        self, inputs: List[int], outputs: List[int]
    ) -> Tuple[List[int], Optional[int]]:
        best_match, branch = [], None
        for i in range(self.num_verifications):
            current_match = [outputs[0]]
            verification_branch_offset = (
                self.verification_offset + (self.ngram_size - 1) * i
            )
            for j in range(self.ngram_size - 1):
                if inputs[verification_branch_offset + j] == current_match[-1]:
                    current_match.append(outputs[verification_branch_offset + j])
                else:
                    break

            if len(current_match[1:]) > len(best_match):
                best_match = current_match[1:]
                branch = i

        return best_match, branch


def retrieve_info_from_pte(pte_path: str) -> dict:
    # Retrieve vocab_size from get_metadata under static_llama that is passed to edge manager
    output_vocab_size = None
    pte_max_context_len = None
    pte_max_seq_len = None
    logits_scale = None
    logits_zero_point = None
    kv_io_bit_width = 32

    with open(pte_path, "rb") as f:
        program_data = f.read()
        program = deserialize_pte_binary(program_data).program

    for method in program.execution_plan:
        # Don't use tokenizer.n_words, the numbers are off once calling get_tokenizer()
        if method.name == "get_vocab_size":
            # pyre-ignore
            output_vocab_size = method.values[0].val.int_val
        if method.name == "get_max_seq_len":
            # pyre-ignore
            pte_max_seq_len = method.values[0].val.int_val
        if method.name == "get_max_context_len":
            # pyre-ignore
            pte_max_context_len = method.values[0].val.int_val
        if method.name == "get_logits_scale":
            logits_scale = method.values[0].val.double_val
        if method.name == "get_logits_zero_point":
            logits_zero_point = method.values[0].val.int_val
        if method.name == "get_kv_io_bit_width":
            kv_io_bit_width = method.values[0].val.int_val
    if pte_max_context_len is None:
        pte_max_context_len = pte_max_seq_len

    # FP has no scale/zero_point, use following values, which is equivalent to not performing dequantize.
    if kv_io_bit_width == 32 or (logits_scale is None or logits_zero_point is None):
        logits_scale = 1
        logits_zero_point = 0
    assert output_vocab_size is not None, "Couldn't find the vocab size"
    assert pte_max_seq_len is not None, "Couldn't find the max_seq_len from pte"
    meta_info = {
        "output_vocab_size": output_vocab_size,
        "pte_max_context_len": pte_max_context_len,
        "pte_max_seq_len": pte_max_seq_len,
        "logits_scale": logits_scale,
        "logits_zero_point": logits_zero_point,
        "kv_io_bit_width": kv_io_bit_width,
    }
    return meta_info


def encode_prompt(tokenizer, prompt) -> List[int]:
    """Prompt (or an already-tokenized tensor) to token ids."""
    if not isinstance(prompt, str):
        return prompt.flatten().tolist()
    # Llama2 tokenizer has no special tokens
    if isinstance(tokenizer, (SentencePieceTokenizer, HuggingFaceTokenizer)):
        return tokenizer.encode(prompt, bos=True, eos=False)
    if isinstance(tokenizer, TiktokenTokenizer):
        return tokenizer.encode(prompt, bos=True, eos=False, allowed_special="all")
    raise RuntimeError("Unknown tokenizer")


def smart_mask_updater(
    n_updates: int,
    atten_mask: AttentionMask,
    pos,
    k_caches,
    v_caches,
    new_k_caches,
    new_v_caches,
    # lookahead decoding related
    lade_token_offset=None,
    lade_pos_offset=None,
    position_shift=0,
):
    max_cache_len = k_caches[0].size(-1)

    shifted_pos = pos + position_shift
    if shifted_pos + n_updates <= max_cache_len:
        if lade_token_offset is not None:
            # lookahead decode update
            for i, offset in enumerate(lade_token_offset):
                current_pos = shifted_pos + i
                for j, (k_cache, v_cache) in enumerate(zip(k_caches, v_caches)):
                    k_cache[:, :, :, current_pos] = new_k_caches[j][:, :, :, offset]
                    v_cache[:, :, current_pos, :] = new_v_caches[j][:, :, offset, :]
        else:
            for i, k_cache in enumerate(k_caches):
                k_cache[:, :, :, shifted_pos : shifted_pos + n_updates] = new_k_caches[
                    i
                ][:, :, :, :n_updates]
            for i, v_cache in enumerate(v_caches):
                v_cache[:, :, shifted_pos : shifted_pos + n_updates, :] = new_v_caches[
                    i
                ][:, :, :n_updates, :]

        atten_mask.smart_mask_update(shifted_pos, n_updates, lade_pos_offset)

    pos += n_updates
    return pos, k_caches, v_caches


def evict_tokens(
    ar_len: int,
    atten_mask: AttentionMask,
    pos,
    k_caches,
    v_caches,
    rope_module,
    position_shift,
):
    max_cache_len = k_caches[0].size(-1)
    shifted_pos = pos + position_shift
    if shifted_pos + ar_len > max_cache_len:
        num_to_evict = rope_module.eviction_batch_size
        k_caches, v_caches = rope_module(k_caches, v_caches)
        position_shift -= num_to_evict
        shifted_pos -= num_to_evict
        atten_mask.smart_mask_init(shifted_pos)
    return k_caches, v_caches, position_shift


def _prefill_chunking(
    inputs: DecoderInputs,
    module: torch.fx.GraphModule,
    ar_len: int,
    collect_logits,
    result_logits,
    seq_mse_candidates,
    k_caches,
    v_caches,
    total_token_list,
    last_input_sample=None,
    dflash_config=None,
):
    with torch.no_grad():
        num_prompt_tokens = len(total_token_list)
        pos = 0  # Tracks how many prompt tokens have been processed.
        # --sink_split keeps position 0 in the draft's context but stops it from
        # setting the range the GENERATION rows are quantized to. Measured on
        # Qwen3-4B the sink row runs 58x the next largest (16391.8 vs 280.3), and a
        # per-tensor encoding that has to cover it costs every other row ~4.4 bits.
        #
        # The split is possible because prefill and decode are different graphs and
        # only prefill ever sees position 0 at runtime (eval_mode 4 wires
        # prompt_processor=prefill_forward, token_generator=kv_forward). So the phase
        # boundary here IS the graph boundary: whatever the captured observers hold
        # when the prompt finishes is what the PREFILL graph should carry, and they
        # are then reset so the speculative loop alone calibrates the DECODE graph.
        # kv_inference snapshots and resets them between the two phases; nothing is
        # needed inside this loop.
        #
        # Splitting by phase rather than by "exclude row 0" also means a sink that
        # re-fires mid-sequence (documented on the 8B) widens the decode range by
        # itself instead of being clipped.
        while pos < num_prompt_tokens:
            chunk_start_idx, chunk_end_idx = pos, min(num_prompt_tokens, pos + ar_len)

            # Take a chunk of prompt tokens, up to ar_len length.
            if inputs.input_ids is not None:
                actual_chunk_tokens = inputs.input_ids[chunk_start_idx:chunk_end_idx]
                num_tokens_in_chunk = len(actual_chunk_tokens)
                # Prepare tmp_token_list (padded with zeros).
                tmp_token_list = torch.zeros((1, ar_len), dtype=inputs.input_ids_dtype)
                tmp_token_list[0, :num_tokens_in_chunk] = torch.tensor(
                    actual_chunk_tokens, dtype=inputs.input_ids_dtype
                )
            else:
                actual_chunk_tokens = inputs.embedding[
                    :, chunk_start_idx:chunk_end_idx, :
                ]
                num_tokens_in_chunk = actual_chunk_tokens.shape[1]
                # Prepare tmp_token_list (padded with zeros).
                tmp_embedding = torch.zeros((1, ar_len, inputs.embedding.shape[-1]))
                tmp_embedding[0, :num_tokens_in_chunk, :] = torch.tensor(
                    actual_chunk_tokens
                )

            # Prepare tmp_pos (padded with zeros).
            tmp_pos = torch.zeros((1, ar_len), dtype=torch.int32)
            tmp_pos[0, :num_tokens_in_chunk] = inputs.all_pos[
                0,
                pos : pos + num_tokens_in_chunk,
            ]

            # Run inference.
            if inputs.input_ids is not None:
                logits, new_k_caches, new_v_caches, *_eagle_extra = module(
                    tmp_token_list,
                    *inputs.atten_mask,
                    tmp_pos,
                    *k_caches,
                    *v_caches,
                )
                last_input_sample = (
                    tmp_token_list,
                    *inputs.atten_mask,
                    tmp_pos,
                    *k_caches,
                    *v_caches,
                )
            else:
                logits, new_k_caches, new_v_caches, *_eagle_extra = module(
                    tmp_embedding,
                    *inputs.atten_mask,
                    tmp_pos,
                    *k_caches,
                    *v_caches,
                )
                last_input_sample = (
                    tmp_embedding,
                    *inputs.atten_mask,
                    tmp_pos,
                    *k_caches,
                    *v_caches,
                )
            if collect_logits:
                result_logits.append(logits[:, :num_tokens_in_chunk])

            # We should have enough calibration data when generating last token if task was specified
            if seq_mse_candidates != 0 and pos == num_prompt_tokens - 1:
                with SeqMSE(module, seq_mse_candidates):
                    if inputs.input_ids is not None:
                        module(
                            tmp_token_list,
                            *inputs.atten_mask,
                            tmp_pos,
                            *k_caches,
                            *v_caches,
                        )
                    else:
                        module(
                            tmp_embedding,
                            *inputs.atten_mask,
                            tmp_pos,
                            *k_caches,
                            *v_caches,
                        )

            # Seed the DFlash draft from the same chunk. `_eagle_extra` carries the
            # selected hidden layers this graph exports; they were being dropped on
            # the floor while the draft recomputed a worse copy of them elsewhere.
            # no_sink drops the sink row from the draft's context but keeps every
            # surviving row's real position, so slot i holds token i+1 rather than
            # renumbering RoPE.
            if dflash_config is not None:
                rows = torch.cat(
                    [h[0, :num_tokens_in_chunk] for h in _eagle_extra], dim=-1
                )
                skip = 1 if (dflash_config.drop_sink_row and pos == 0) else 0
                # The draft's new_context is [1, draft_ar, .], so it cannot swallow
                # a target chunk in one call once the two ar's diverge -- which they
                # do as soon as the target widens for a tree.
                base = pos + skip
                for off in range(0, num_tokens_in_chunk - skip, dflash_config.ar):
                    n_ctx = min(dflash_config.ar, num_tokens_in_chunk - skip - off)
                    if n_ctx > 0 and dflash_config.can_append(n_ctx):
                        dflash_config.seed(
                            rows[skip + off : skip + off + n_ctx], n_ctx, base + off
                        )

            # Update the pos, KV cache and attention mask.
            pos, k_caches, v_caches = smart_mask_updater(
                num_tokens_in_chunk,
                inputs.atten_mask,
                pos,
                k_caches,
                v_caches,
                new_k_caches,
                new_v_caches,
            )

        # Append the last run logits to the total_token_list.
        total_token_list.append(
            torch.argmax(logits[:, num_tokens_in_chunk - 1], dim=-1).item()
        )

        return pos, last_input_sample


def _captured_hidden_observers(module, n):
    """The observers on the LAST `n` graph outputs of the decoder -- the captured
    hiddens the draft consumes, and nothing else.

    Positional on purpose. The decoder returns
    (final_hidden, k_caches..., v_caches..., *captured), so the captured ones are the
    tail -- the same slice the caller unpacks as `*_eagle_extra`, which is what makes
    the two zip correctly. `n` comes from that convention, not from inspection.

    Selecting by rank instead (rank-3 outputs past index 0) found ZERO on the real
    graph: prepare_pt2e's inserted observer nodes carry no meta['val'], so every
    candidate was skipped. It passed a unit test only because the hand-built graphs
    there set meta['val'] explicitly. Shape metadata is not dependable on inserted
    nodes; output position is.
    """
    # _SplitEval holds three graphs; the decoder is the one with the many outputs.
    best, best_len = None, 0
    for sub in module.modules():
        g = getattr(sub, "graph", None)
        if g is None:
            continue
        out_node = next((x for x in g.nodes if x.op == "output"), None)
        if out_node is None:
            continue
        outs = list(out_node.args[0])
        if len(outs) > best_len:
            best, best_len = (sub, outs), len(outs)
    if best is None or best_len <= n:
        return []
    sub, outs = best
    found = []
    for out in outs[-n:]:
        obs = getattr(sub, out.target, None) if out.op == "call_module" else None
        if obs is not None:
            found.append(obs)
    return found


def _bank_and_reset(observers):
    """Read each observer's qparams, then clear it back to its initial state.

    Used at the prompt/generation boundary: the range accumulated so far belongs to
    the prefill graph (the only one that sees position 0 at runtime), and the range
    accumulated after belongs to the decode graph. Splitting by phase rather than by
    "exclude row 0" also lets a sink that re-fires mid-sequence widen the decode
    range on its own instead of being clipped.
    """
    banked = []
    for obs in observers:
        scale, zp = obs.calculate_qparams()
        banked.append((float(scale.reshape(-1)[0]), int(zp.reshape(-1)[0])))
        obs.reset_min_max_vals()
    return banked


def _observer_snapshot(observers):
    """Every buffer of the given observers, cloned.

    Used to run a forward whose ACTIVATIONS must not widen these ranges while its
    side effects (KV cache, attention mask) must still happen. Restoring buffers is
    type-agnostic, unlike toggling an `observer_enabled` flag that only FakeQuantize
    has -- prepare_pt2e inserts plain observers.
    """
    return [
        (obs, {n: b.detach().clone() for n, b in obs.named_buffers(recurse=False)})
        for obs in observers
    ]


def _observer_restore(snap):
    for obs, bufs in snap:
        for n, b in obs.named_buffers(recurse=False):
            if n in bufs:
                b.copy_(bufs[n])


def build_ddtree(logp, ids, budget: int, depth_limit: int):
    """DDTree: best-first over the product of the per-depth draft distributions.

    A node at depth d draws from depth d's own top-k regardless of what was chosen
    above it -- block diffusion produced every depth in the one draft forward, so
    the tree is free no matter how wide it gets. Adding a depth only ever lowers a
    path's score, so a parent is always popped before its children, which is what
    lets slot k copy its parent's attention row.

    With k == 1 no sibling is ever pushed and this lays down `depth_limit` nodes in
    a single path: the plain chain, same code.

    Returns (tokens, parents, depths, children) with one entry per non-root node;
    slot indices are 1-based because slot 0 is the confirmed token.
    """
    k = logp.shape[1]
    heap = [(-float(logp[0, 0]), 0, 1, 0)]  # (-logw, parent slot, depth, rank)
    tokens, parents, depths = [], [], []
    children = [dict()]
    while heap and len(tokens) < budget:
        neg_w, parent, depth, rank = heapq.heappop(heap)
        token = int(ids[depth - 1, rank])
        slot = len(tokens) + 1
        tokens.append(token)
        parents.append(parent)
        depths.append(depth)
        children.append(dict())
        children[parent][token] = slot
        if rank + 1 < k:
            sib = -neg_w - float(logp[depth - 1, rank]) + float(logp[depth - 1, rank + 1])
            heapq.heappush(heap, (-sib, parent, depth, rank + 1))
        if depth < depth_limit:
            heapq.heappush(heap, (neg_w - float(logp[depth, 0]), slot, depth + 1, 0))
    return tokens, parents, depths, children


def _write_tree_mask(atten_mask, parents, n_nodes: int, ar_len: int):
    """Overwrite the new-token block of the attention mask with the tree topology.

    Only the last ar_len columns -- the cache half is what smart_mask_update
    maintains and it stays "everything committed so far". A node sees its
    ancestors and itself, built by inheriting its parent's row, which is legal
    because build_ddtree emits parents before children. Padded slots see the root
    alone; their outputs are dropped either way.
    """
    vis = torch.zeros(ar_len, ar_len, dtype=torch.bool)
    vis[0, 0] = True
    for slot in range(1, n_nodes):
        vis[slot] = vis[parents[slot - 1]]
        vis[slot, slot] = True
    for slot in range(n_nodes, ar_len):
        vis[slot] = vis[0]
    block = torch.where(vis, 0.0, -255.0)
    for mask in atten_mask.masks:
        mask.mask[:, :, -ar_len:] = block


def _generate_dflash(
    inputs: DecoderInputs,
    pos,
    module: torch.fx.GraphModule,
    tokenizer,
    ar_len: int,
    k_caches,
    v_caches,
    total_token_list,
    dflash_config,
    last_input_sample=None,
):
    """Calibrate through real speculative decoding: the draft proposes a block, this
    graph verifies it, and both sets of observers see exactly the traffic they get at
    inference. Mirrors the decode loop in dflash_token_generator.cpp.

    The plain branch below would drive this graph with one real row and ar_len-1
    zero-padded ones per call, while at inference dflash hands it a full block of
    real tokens -- and would leave the draft to be calibrated afterwards, out of the
    loop, on hidden that no graph ever emits.
    """
    if inputs.input_ids is None:
        raise RuntimeError("DFlash calibration expects the token-input decoder path")
    B, n_draft = dflash_config.block_size, dflash_config.n_draft
    max_cache_len = k_caches[0].size(-1)
    cur_pos = pos
    last_committed = total_token_list[-1]
    pending = (None, 0, 0)  # rows, count, position of the first row
    accepted_total = 0
    # The verify window is ar_len wide, so it holds the confirmed token plus
    # ar_len - 1 drafted nodes. Chain calibration is the k=1 case of the same
    # tree, which is why there is only one loop below.
    budget = ar_len - 1
    topk = min(budget, 64) if dflash_config.tree else 1
    logging.info(
        "dflash calibration: %s ar_len=%d budget=%d topk=%d depth_limit=%d",
        "tree" if dflash_config.tree else "chain",
        ar_len,
        budget,
        topk,
        n_draft,
    )
    with torch.no_grad():
        while cur_pos + B < max_cache_len:
            rows, n_new, pos_base = pending
            logp, ids = dflash_config.draft_block(
                last_committed, rows, n_new, pos_base, cur_pos, topk
            )
            tokens, parents, depths, children = build_ddtree(
                logp, ids, budget, n_draft
            )
            n_nodes = 1 + len(tokens)

            vtok = torch.zeros((1, ar_len), dtype=inputs.input_ids_dtype)
            vtok[0, 0] = last_committed
            if tokens:
                vtok[0, 1:n_nodes] = torch.tensor(
                    tokens, dtype=inputs.input_ids_dtype
                )
            # Siblings share a position -- a node sits at cur_pos + its depth.
            # Padded slots repeat the last real one: an out-of-range RoPE index
            # would be a second, invisible difference from the device.
            vpos = torch.zeros((1, ar_len), dtype=torch.int32)
            last_depth = depths[-1] if depths else 0
            for k in range(ar_len):
                d = 0 if k == 0 else (depths[k - 1] if k < n_nodes else last_depth)
                vpos[0, k] = cur_pos + d
            _write_tree_mask(inputs.atten_mask, parents, n_nodes, ar_len)

            logits, new_k_caches, new_v_caches, *captured = module(
                vtok, *inputs.atten_mask, vpos, *k_caches, *v_caches
            )
            last_input_sample = (
                vtok,
                *inputs.atten_mask,
                vpos,
                *k_caches,
                *v_caches,
            )
            # Walk the target's own continuation down the tree for as long as the
            # tree has it, the same rule the runner applies.
            path = [0]
            bonus = int(logits[0, 0].argmax())
            while bonus in children[path[-1]]:
                path.append(children[path[-1]][bonus])
                bonus = int(logits[0, path[-1]].argmax())
            accepted = len(path) - 1
            n_commit = accepted + 1

            pos_base = cur_pos
            # A tree path lands on scattered slots, so the KV rows have to be
            # gathered rather than sliced -- lade_token_offset is the same hook
            # lookahead uses for the same reason.
            cur_pos, k_caches, v_caches = smart_mask_updater(
                n_commit,
                inputs.atten_mask,
                cur_pos,
                k_caches,
                v_caches,
                new_k_caches,
                new_v_caches,
                lade_token_offset=path,
            )
            pending = (
                torch.cat([h[0, path] for h in captured], dim=-1),
                n_commit,
                pos_base,
            )
            accepted_total += accepted
            hit_eos = False
            for k in range(1, n_commit):
                tok = tokens[path[k] - 1]
                total_token_list.append(tok)
                if tok == tokenizer.eos_id:
                    hit_eos = True
                    break
            if hit_eos:
                break
            last_committed = bonus
            total_token_list.append(last_committed)
            if last_committed == tokenizer.eos_id:
                break
            if not dflash_config.can_append(n_commit):
                logging.info("dflash calibration: draft context full")
                break
    logging.info(
        "dflash calibration accepted / total generated: %d / %d over %d rounds",
        accepted_total,
        cur_pos - pos,
        dflash_config.rounds,
    )
    return last_input_sample


def _generate(
    inputs: DecoderInputs,
    pos,
    module: torch.fx.GraphModule,
    tokenizer,
    tok_embedding,
    ar_len: int,
    max_seq_len: int,
    k_caches,
    v_caches,
    total_token_list,
    lookahead_config,
    last_input_sample=None,
    dflash_config=None,
):
    max_cache_len = max_seq_len - ar_len
    num_tokens = len(total_token_list)
    if dflash_config is not None:
        return _generate_dflash(
            inputs,
            pos,
            module,
            tokenizer,
            ar_len,
            k_caches,
            v_caches,
            total_token_list,
            dflash_config,
            last_input_sample,
        )
    if lookahead_config is None:
        while total_token_list[-1] != tokenizer.eos_id and num_tokens < max_seq_len:
            chunk_start_idx = min(pos, max_cache_len)
            # Take a chunk of generated tokens, up to ar_len length.
            chunk_end_idx = num_tokens
            actual_chunk_tokens = total_token_list[chunk_start_idx:chunk_end_idx]
            num_tokens_in_chunk = len(actual_chunk_tokens)

            # Prepare tmp_token_list (padded with zeros).
            tmp_token_list = torch.zeros((1, ar_len), dtype=inputs.input_ids_dtype)
            tmp_token_list[0, :num_tokens_in_chunk] = torch.tensor(
                actual_chunk_tokens, dtype=inputs.input_ids_dtype
            )

            if inputs.input_ids is None:
                # Get text_embedding
                embedding = tok_embedding(tmp_token_list)

            # Prepare tmp_pos (padded with zeros).
            tmp_pos = torch.zeros((1, ar_len), dtype=torch.int32)
            tmp_pos[0, :num_tokens_in_chunk] = inputs.all_pos[
                0, chunk_start_idx:chunk_end_idx
            ]

            if inputs.input_ids is not None:
                logits, new_k_caches, new_v_caches, *_eagle_extra = module(
                    tmp_token_list,
                    *inputs.atten_mask,
                    tmp_pos,
                    *k_caches,
                    *v_caches,
                )
                last_input_sample = (
                    tmp_token_list,
                    *inputs.atten_mask,
                    tmp_pos,
                    *k_caches,
                    *v_caches,
                )
            else:
                logits, new_k_caches, new_v_caches, *_eagle_extra = module(
                    embedding,
                    *inputs.atten_mask,
                    tmp_pos,
                    *k_caches,
                    *v_caches,
                )
                last_input_sample = (
                    embedding,
                    *inputs.atten_mask,
                    tmp_pos,
                    *k_caches,
                    *v_caches,
                )

            pos, k_caches, v_caches = smart_mask_updater(
                1,
                inputs.atten_mask,
                pos,
                k_caches,
                v_caches,
                new_k_caches,
                new_v_caches,
            )
            total_token_list.append(
                torch.argmax(logits[:, num_tokens_in_chunk - 1], dim=-1).item()
            )
            num_tokens = len(total_token_list)
    else:
        # TODO: support batch decode if necessary
        # variable declaration
        window, ngram, gcap = lookahead_config
        lade = LookaheadDecoder(
            window_size=window,
            ngram_size=ngram,
            num_verifications=gcap,
            ar_size=ar_len,
            mask_value=next(iter(inputs.atten_mask)).min().item(),
        )
        generated_tokens, accepted_tokens = 0, 0
        input_tokens = [total_token_list[-1]] * ar_len
        pos_offsets = lade.position_offset.unsqueeze(0)
        pos_offsets_list = pos_offsets.flatten().tolist()
        # replace ar attention mask to lookahead version
        for mask in inputs.atten_mask:
            mask[:, :, -ar_len:] = lade.attention_mask.unsqueeze(0)
        # start decoding
        while (
            total_token_list[-1] != tokenizer.eos_id
            and len(total_token_list) < max_cache_len
        ):
            # populate verification branch
            lade.update_verification_branch(
                guess_token=input_tokens[0],
                inputs=input_tokens,
            )
            # inference
            if inputs.input_ids is not None:
                logits, new_k_caches, new_v_caches, *_eagle_extra = module(
                    torch.tensor(input_tokens, dtype=inputs.input_ids_dtype).unsqueeze(
                        0
                    ),
                    *inputs.atten_mask,
                    pos_offsets + pos,
                    *k_caches,
                    *v_caches,
                )
                last_input_sample = (
                    torch.tensor(input_tokens, dtype=inputs.input_ids_dtype).unsqueeze(
                        0
                    ),
                    *inputs.atten_mask,
                    pos_offsets + pos,
                    *k_caches,
                    *v_caches,
                )
            else:
                logits, new_k_caches, new_v_caches, *_eagle_extra = module(
                    tok_embedding(
                        torch.tensor(
                            input_tokens, dtype=inputs.input_ids_dtype
                        ).unsqueeze(0)
                    ),
                    *inputs.atten_mask,
                    pos_offsets + pos,
                    *k_caches,
                    *v_caches,
                )
                last_input_sample = (
                    tok_embedding(
                        torch.tensor(
                            input_tokens, dtype=inputs.input_ids_dtype
                        ).unsqueeze(0)
                    ),
                    *inputs.atten_mask,
                    pos_offsets + pos,
                    *k_caches,
                    *v_caches,
                )
            # collect outputs
            output_tokens = torch.argmax(logits, dim=-1).flatten().tolist()
            # update ngram pool
            lade.update_ngram_pool(inputs=input_tokens, outputs=output_tokens)
            # try matching verification branches
            best_match, branch_no = lade.verify(
                inputs=input_tokens, outputs=output_tokens
            )
            # check if any match was found
            lade_token_offset, num_match = [0], len(best_match)
            if num_match > 0:
                accepted_tokens += num_match
                lade_token_offset += [
                    e + lade.verification_offset + branch_no * (ngram - 1)
                    for e in range(num_match)
                ]
            # update kv cache
            pos, k_caches, v_caches = smart_mask_updater(
                len(lade_token_offset),
                inputs.atten_mask,
                pos,
                k_caches,
                v_caches,
                new_k_caches,
                new_v_caches,
                lade_token_offset,
                pos_offsets_list,
            )
            generated_tokens += len(lade_token_offset)
            # update lookahead branch
            lade.update_lookahead_branch(inputs=input_tokens, outputs=output_tokens)
            # update token list
            for token in [output_tokens[0], *best_match]:
                total_token_list.append(token)
                if token == tokenizer.eos_id:
                    break
            # fill next input token
            input_tokens[0] = total_token_list[-1]

        logging.info(
            f"lookahead accepted / total generated: {accepted_tokens} / {generated_tokens}"
        )
    return last_input_sample


@register_inference(use_kv_cache=True)
def kv_inference(  # noqa: C901
    get_example_inputs: Callable,
    prompt: Union[str, list],
    module: torch.fx.GraphModule,
    tokenizer,
    tok_embedding=None,
    hidden_states: Tuple = (),
    audio_token_id=None,
    image_token_id=None,
    ar_len=1,
    max_seq_len=512,
    use_i64_token=False,
    collect_logits=False,
    seq_mse_candidates=0,
    lookahead_config=None,
    dflash_config=None,
):
    input_samples = []  # Record input sample for quantization error analysis
    # external embedding: the decoder takes inputs_embeds instead of token ids —
    # multimodal (image/audio) OR the text emb-split (tok_embedding in its own pte).
    # The actual multimodal merge below is separately gated on `hidden_states`.
    use_external_embedding = tok_embedding is not None
    # max_seq_len bounds prompt AND generation, so a prompt longer than it walks
    # `all_pos` off the end and _prefill_chunking assigns from an empty slice.
    prompt_len = len(encode_prompt(tokenizer, prompt))
    if prompt_len > max_seq_len:
        raise RuntimeError(
            f"prompt is {prompt_len} tokens but max_seq_len is {max_seq_len}"
        )

    _, atten_mask, _, k_caches, v_caches = get_example_inputs()

    # TODO: change criteria & support batch inputs if necessary
    # Sized by the context, not by max_seq_len: max_seq_len bounds how much to
    # GENERATE, but the loop commits a chunk at a time and so overshoots it by up to
    # ar_len, and then indexes positions that exist in the sequence but not in this
    # table. An arange costs nothing, so give it the whole context and let the
    # generation bound be enforced where it is actually meant -- the loop condition.
    max_cache_len = k_caches[0].size(-1)
    all_pos = torch.arange(
        0, max(max_seq_len, max_cache_len + ar_len), 1, dtype=torch.int32
    ).unsqueeze(0)

    prompt_token_list, total_token_list, result_logits = [], [], []

    # 1. prepare token ids
    prompt_token_list = encode_prompt(tokenizer, prompt)

    # 2. process embedding
    if use_external_embedding:
        # 2.1 forward text embedding
        input_ids = torch.tensor([prompt_token_list])
        input_ids = (
            input_ids.to(torch.int64) if use_i64_token else input_ids.to(torch.int32)
        )
        input_ids_len = input_ids.shape[-1]
        padded_seq_len = max(input_ids_len, ar_len)
        padded_seq_len = ((padded_seq_len + ar_len - 1) // ar_len) * ar_len

        embedding_dim = [p for _, p in tok_embedding.named_parameters()][0].shape[-1]
        text_embeddings = torch.zeros(
            (
                1,
                padded_seq_len,
                embedding_dim,
            ),
            dtype=torch.float32,
        )

        with torch.no_grad():
            # Calculate number of chunks needed
            num_chunks = (input_ids_len + ar_len - 1) // ar_len

            # Prefill embeddings in chunks
            for chunk_id in range(num_chunks):
                chunk_start_idx = chunk_id * ar_len
                chunk_end_idx = chunk_start_idx + ar_len

                # Only process if there are tokens in this chunk
                if chunk_start_idx < input_ids_len:
                    embedding = tok_embedding(
                        input_ids[:, chunk_start_idx:chunk_end_idx]
                    )
                    # Put embedding in the correct position
                    actual_chunk_len = embedding.shape[1]
                    text_embeddings[
                        :, chunk_start_idx : chunk_start_idx + actual_chunk_len, :
                    ] = embedding

            # 2.2 merge text and multimodality embedding
            if hidden_states:
                multimodal_embedding = _modality_inputs_merger(
                    input_ids,
                    text_embeddings[
                        :, :input_ids_len, :
                    ],  # Only use actual prompt length
                    torch.cat(hidden_states, dim=1),
                    audio_token_id or image_token_id,
                )
            else:
                multimodal_embedding = text_embeddings[:, :input_ids_len, :]

    # record total input tokens and generated tokens
    total_token_list = prompt_token_list

    # 3. prepare decoder inputs
    inputs = DecoderInputs(
        all_pos,
        atten_mask,
        input_ids=prompt_token_list if not use_external_embedding else None,
        input_ids_dtype=torch.int64 if use_i64_token else torch.int32,
        embedding=multimodal_embedding if use_external_embedding else None,
    )

    # 4. decoder forward
    with torch.no_grad():
        # Phase 1: Prefill the prompt in ar_len chunks.
        cur_pos, prefill_input_sample = _prefill_chunking(
            inputs,
            module,
            ar_len,
            collect_logits,
            result_logits,
            seq_mse_candidates,
            k_caches,
            v_caches,
            total_token_list,
            dflash_config=dflash_config,
        )

        # The prompt is done, so whatever the draft-facing observers hold right now is
        # exactly what the PREFILL graphs will carry at runtime -- prefill is the only
        # graph that ever processes position 0. Bank those qparams and reset, so the
        # speculative loop below calibrates the DECODE graphs on generation rows alone.
        # compile() stamps the banked values back onto the prefill graphs after the
        # usual decode->prefill override has run.
        if dflash_config is not None and dflash_config.sink_split:
            n_cap = len(dflash_config.cfg.target_layer_ids)
            obs = _captured_hidden_observers(module, n_cap)
            if len(obs) != n_cap:
                raise RuntimeError(
                    f"sink_split found {len(obs)} captured-hidden observers, "
                    f"expected {n_cap}"
                )
            dflash_config.prefill_qparams = _bank_and_reset(obs)
            dflash_config.bank_draft_prefill_qparams()
            logging.info(
                "sink_split: banked prefill qparams for %d captured hiddens "
                "+ the draft's new_context, observers reset for the decode phase",
                len(obs),
            )

        # Phase 2: Generate tokens until the EOS token is generated or max_seq_len is reached.
        # When run on wikitext for ppl evaluation, this while-loop is not expected to run.
        generate_input_sample = _generate(
            inputs,
            cur_pos,
            module,
            tokenizer,
            tok_embedding,
            ar_len,
            max_seq_len,
            k_caches,
            v_caches,
            total_token_list,
            lookahead_config,
            dflash_config=dflash_config,
        )
        if generate_input_sample is not None:
            input_samples.append(generate_input_sample)
        else:
            input_samples.append(prefill_input_sample)

    logging.info(f"kv inference result:\n{tokenizer.decode(total_token_list)}")
    if collect_logits:
        result_logits = torch.cat(result_logits, dim=1)
    return result_logits, input_samples


@register_inference(use_kv_cache=False)
def prefill_inference(
    get_example_inputs: Callable,
    prompt: Union[str, list],
    module: torch.fx.GraphModule,
    tokenizer,
    tok_embedding=None,
    hidden_states=None,
    audio_token_id=None,
    image_token_id=None,
    max_seq_len=512,
    use_i64_token=False,
    collect_logits=False,
):
    input_samples = None  # Record input sample for quantization error analysis
    is_multimodal = all(
        [
            tok_embedding,
            audio_token_id or image_token_id,
        ]
    )

    _, atten_mask = get_example_inputs()

    # TODO: change criteria & support batch inputs if necessary

    result_logits = []
    token_list = encode_prompt(tokenizer, prompt)

    pos = len(token_list)
    dtype = torch.int64 if use_i64_token else torch.int32

    with torch.no_grad():
        while token_list[-1] != tokenizer.eos_id and pos < max_seq_len:
            tmp_token_list = torch.tensor(token_list, dtype=dtype).reshape(1, -1)
            if pos < max_seq_len:
                tmp_token_list = torch.cat(
                    [
                        tmp_token_list,
                        torch.zeros((1, max_seq_len - pos), dtype=dtype),
                    ],
                    dim=1,
                )

            if is_multimodal:
                text_embeddings = tok_embedding(tmp_token_list)
                multimodal_embedding = _modality_inputs_merger(
                    tmp_token_list,
                    text_embeddings,
                    torch.cat(hidden_states, dim=1),
                    audio_token_id or image_token_id,
                )
                results = module(multimodal_embedding, *atten_mask)
                input_samples = (multimodal_embedding, *atten_mask)
            else:
                results = module(tmp_token_list, *atten_mask)
                input_samples = (tmp_token_list, *atten_mask)
            if len(results) == 3:
                logits, _, _ = results
            elif len(results) == 1:
                logits = results
            token = torch.argmax(logits[:, pos - 1], dim=-1).item()
            token_list.append(token)
            if collect_logits:
                result_logits = logits[:, :pos]
            pos += 1
    if isinstance(prompt, str):
        logging.info(f"prefill inference result:\n{tokenizer.decode(token_list)}")
    return result_logits, [input_samples]


def graph_module_inference(
    use_kv_cache: bool,
    get_example_inputs: Callable,
    module: torch.fx.GraphModule,
    tokenizer,
    ar_len=1,
    max_seq_len=512,
    prompt=None,
    tok_embedding=None,
    hidden_states: Tuple = (),
    audio_token_id=None,
    image_token_id=None,
    tasks=None,
    tasks_limit=1,
    num_fewshot=None,
    use_i64_token=False,
    event_name: Optional[str] = None,
    seq_mse_candidates: int = 0,
    lookahead_config: Optional[Tuple[int]] = None,
    dflash_config=None,
):
    """
    This function supports model execution from static nn.Module decoder model
    all the way to edge program.
    Users could choose to provide either the prompt or tasks for execution but not both.
    """
    # Checks 1 and only 1 is provided.
    assert (tasks is None) != (
        prompt is None
    ), "Please provide either tasks or prompt/input_ids - not both or neither"
    if tasks is None:
        kwargs = {}
        if use_kv_cache:
            kwargs["ar_len"] = ar_len
            kwargs["lookahead_config"] = lookahead_config
            kwargs["dflash_config"] = dflash_config

        _, input_samples = INFERENCE_REGISTRY[use_kv_cache](
            get_example_inputs,
            prompt,
            module,
            tokenizer,
            tok_embedding=tok_embedding,
            hidden_states=hidden_states,
            audio_token_id=audio_token_id,
            image_token_id=image_token_id,
            max_seq_len=max_seq_len,
            use_i64_token=use_i64_token,
            collect_logits=False,
            **kwargs,
        )
        logging.info(f"Prompt summary for {event_name}")
        return input_samples
    else:
        calibration_wrapper = GraphModuleCalibrationWrapper(
            model=module,
            tokenizer=tokenizer,
            max_seq_length=max_seq_len,
            ar_len=ar_len,
            use_kv_cache=use_kv_cache,
            get_example_inputs=get_example_inputs,
            use_i64_token=use_i64_token,
            seq_mse_candidates=seq_mse_candidates,
        )
        with torch.no_grad():
            eval_results = simple_evaluate(
                model=calibration_wrapper,
                tasks=tasks,
                num_fewshot=num_fewshot,
                limit=tasks_limit,
            )
        logging.info(f"Evaluation summary for {event_name}")
        for task, res in eval_results["results"].items():
            logging.info(f"{task}: {res}")

        return calibration_wrapper.get_input_samples()
