# EAGLE Mode for QNN LLM Runner — Hand-off Plan

> Self-contained plan for migrating the work to a fresh checkout (e.g. `sbo`
> server: `zqchen@47.99.72.246:40600`). On the new machine **nothing in this
> plan has been started yet** — start from Phase 0.
>
> Read this whole document first; it captures the full background discussion
> that led to the chosen design.

---

## 1. Context & Background

### 1.1 What we are building

A new compile-and-inference mode `--model_mode eagle` for the ExecuTorch QNN
llama pipeline (`examples/qualcomm/oss_scripts/llama`). It implements
**EAGLE speculative decoding** with these properties:

- Target model: **Qwen3-1.7B**, runs on Qualcomm HTP (NPU), quantized.
- Draft model: an **EAGLE head** (single-layer transformer, ≈100M params)
  trained by the user from Qwen3-1.7B hidden states. Weights are in HF
  safetensors format. Tokenizer / embedding / lm_head are shared with target.
- Both target and head are compiled into separate `.pte` files.
- Runtime loads both modules; per generation step the head proposes a draft
  tree, target verifies in one forward pass.
- Supports heterogeneous backends for the head: `htp` / `vulkan` / `xnnpack`,
  selected via CLI. Default `htp`.

Existing modes (`kv`, `hybrid`, `lookahead`) **must remain bit-identical**.
Every new code path is gated by `model_mode == "eagle"`.

### 1.2 Why this design (decisions discussed with user)

| Decision | Rationale |
|---|---|
| Use **EAGLE** (not vanilla speculative decoding with a small LLM) | User has trained EAGLE head weights for Qwen3-1.7B already. Provides higher accept rate (~3×) than a vanilla draft LLM (~1.5×). |
| **Both models on NPU first**, GPU/CPU as Layer 3 | Layer 1 baseline must exist before measuring heterogeneous-pipeline speedup. Layer 3 (Vulkan / XNNPACK draft) is added behind a CLI flag. |
| **Draft and target use independent KV caches** | They have different layer counts, head dims, weights — KV reuse is impossible regardless of architecture. |
| **Tree verification with max-padded ar_len** | EAGLE-2/3 uses dynamic trees, but QNN compiles ar_len statically. Solution: compile target `kv_forward` with `ar_len = next_pow2(max_tree_size)`; runtime fills tree nodes and pads unused slots via the tree-attention mask. |
| **Hidden-state IO precision configurable** (default fp16) | Target exports low/mid/high hidden states for the head. fp16 IO preserves accept rate; quant16 saves bandwidth but may degrade accept rate. |
| **Tokenizer shared with target** | EAGLE head reuses target's embedding + lm_head, so vocab is automatically aligned. No vocab translation needed. |
| **Heterogeneous draft via `--eagle_draft_backend`** | `htp` for baseline; `vulkan` for true GPU offload (Adreno fp16, no quant); `xnnpack` for CPU fallback (quantized). QNN GPU backend was investigated and is unsuitable for LLM draft (no quant support, requires online_prepare, incomplete operator coverage). |

### 1.3 Why the previous attempt is being thrown out

A prior incomplete design left these artifacts:

- `examples/qualcomm/oss_scripts/llama/draft_mode_design.md`
- `examples/qualcomm/oss_scripts/llama/runner/draft_token_generator.{h,cpp}`

That design assumed:

1. The codebase used `TokenGenerator<T>` and `KVManager<T>` templates parameterised on cache dtype. **It does not** — these classes are non-template in the actual code. The `.cpp` will not compile.
2. The draft model would be a vanilla small LLM running on **CPU fp32**. The user wants EAGLE head on **NPU/GPU/CPU quantized**.
3. Token-id-only verification (no hidden-state passing). EAGLE needs hidden-state shuttling between target and draft.

→ **Delete those two files** as part of Phase 0 cleanup (see §3.0). The design
doc will be rewritten as `eagle_mode_design.md` in Phase 6.

### 1.4 EAGLE-3 inference recap (reference)

For implementers unfamiliar with EAGLE-3 (Li et al., 2025, arXiv 2503.01840):

- During target prefill / verify, target exports **3 hidden tensors** captured
  from low / mid / high transformer layers. Each shape `[B, S, k]` where
  `k = target.hidden_size` (Qwen3-1.7B → k=2048).
- Draft head per step:
  - Inputs: previous fused feature `prev_g` or previous draft output `prev_a`
    of shape `[1, 1, k]`; new token's embedding `e` of shape `[1, 1, k]`.
  - `concat([prev_*, e], dim=-1)` → `[1, 1, 2k]` → FC(2k→k) → single-layer
    transformer → `a` of shape `[1, 1, k]` and KV-cache update.
  - Logits computed by passing `a` through target's lm_head (weight shared).
- For step 1 (right after target prefill or verify), `prev_g_t` is computed
  from concat(low_t, mid_t, high_t) of shape `[1, S, 3k]` followed by
  FC(3k→k) (the FC weight is part of the EAGLE head checkpoint).
- For steps ≥ 2 within a draft phase, `prev_g` is replaced by the head's own
  `prev_a` from the previous step (the unique EAGLE-3 trick).
- After verify, accept up to first mismatch in the tree; rollback both
  target and draft KV caches; refresh `prev_g` from the accepted-tail
  position's hidden states output by target.

### 1.5 Locked-in answers from user

| Item | Value |
|---|---|
| Target model | `qwen3-1_7b` (already registered in `__init__.py`) |
| EAGLE version | EAGLE-3 (assume 3 hidden layers); runtime should auto-adapt to EAGLE-1/2 if config says only 1 hidden layer |
| Head weights | HF safetensors + config.json, path supplied via `--eagle_head_checkpoint` |
| Head layers | Read from head config (typically 1 transformer layer) |
| Tokenizer | Shared with target |
| Hidden IO precision | Configurable via `--eagle_hidden_io {fp16,quant16}`, default fp16 |
| Verify topology | Tree, max-padded; `ar_len = next_pow2(max_tree_size)` |
| Draft backend | CLI choice `--eagle_draft_backend {htp,vulkan,xnnpack}`, default `htp` |
| Hidden source layers | From head config; overridable via `--eagle_layer_indices L,M,H` |
| Multimodal + EAGLE | Not supported, fail-fast |
| Attention sink + EAGLE | Not supported, fail-fast |
| Multi-batch | Out of scope |

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Compile (Python) — model_mode = "eagle"                          │
│                                                                  │
│   eagle_target_qnn.pte (HTP, quantized)                          │
│     prefill_forward(ar=N)   → logits, kv_out, hidden_{L,M,H}     │
│     kv_forward(ar=max_tree) → logits, kv_out, hidden_{L,M,H}     │
│                                                                  │
│   eagle_head_<backend>.pte (HTP / Vulkan / XNNPACK)              │
│     prefill_forward(ar=N)   → a_out, kv_out                      │
│     kv_forward(ar=1)        → a_out, kv_out                      │
│       inputs:                                                    │
│         prev_feature [1,1,k]   (g_t from target or a_t from head)│
│         tok_emb       [1,1,k]                                    │
│         pos          [1]                                         │
│         attn_mask     [1,1,context_len]                          │
│         k/v_cache     (1 layer)                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Runtime (C++) — EagleTokenGenerator                              │
│                                                                  │
│  loop until EOS / seq_len:                                       │
│    1. Build draft tree                                           │
│       (Phase 3 = chain;  Phase 4 = static balanced tree)         │
│    2. For each tree position:                                    │
│         draft.kv_forward(prev_g_or_a, e_tok, pos)                │
│           → a_t, sampled token                                   │
│    3. Pack [cur_token, ...tree_tokens] into target ar buffer;    │
│       fill tree-attention mask;                                  │
│       pad unused slots                                           │
│    4. target.kv_forward(tokens, tree_mask, pos)                  │
│           → logits[max_tree], hidden_{L,M,H}[max_tree], kv_out   │
│    5. Walk longest accepted prefix in tree (greedy or            │
│       probability-ratio acceptance);                             │
│       commit accepted tokens via token_callback;                 │
│       rollback target+draft KV to last-accepted-pos              │
│    6. Refresh draft prev_g from target hidden of accepted tail   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Phased Implementation Plan

Six phases. Acceptance criteria per phase. **Total estimate: 18-22 working days.**

---

### Phase 0 — Plumbing (1 day)

Goal: add CLI flags, EVAL_MODE entry, runner gflags, pte filename routing,
runner.h/cpp scaffolding, no-op stubs. After Phase 0, `kv | hybrid |
lookahead` modes are bit-identical to before; `--model_mode eagle` parses
without crashing.

#### 3.0 Cleanup of stale files (do this FIRST, then Phase 0 edits)

```bash
rm examples/qualcomm/oss_scripts/llama/runner/draft_token_generator.cpp
rm examples/qualcomm/oss_scripts/llama/runner/draft_token_generator.h
# draft_mode_design.md will be rewritten in Phase 6 — leave for now
```

Verify nothing in `CMakeLists.txt` / `targets.bzl` / `BUCK` references them
(they were never added there).

#### 3.1 `decoder_constants.py`

Add `"eagle": 3` to `EVAL_MODE` and two new component identifiers:

```python
# evaluation mode
EVAL_MODE = {
    "kv": 0,
    "hybrid": 1,
    "lookahead": 2,
    "eagle": 3,
}

# Eagle mode component identifiers
EAGLE_TARGET = "eagle_target"
EAGLE_HEAD = "eagle_head"
```

#### 3.2 `llama.py`

In `_build_parser()` add (right after `--gcap`, before `--use_attention_sink`):

```python
# ---- Eagle speculative decoding (model_mode == "eagle") ----
parser.add_argument("--eagle_head_checkpoint", default=None, type=str,
    help="[Eagle mode] Path to the EAGLE head weights (HF safetensors file or directory).")
parser.add_argument("--eagle_head_config", default=None, type=str,
    help="[Eagle mode] Path to the EAGLE head config json. If omitted and "
         "--eagle_head_checkpoint is a directory, config.json from that directory is used.")
parser.add_argument("--max_tree_size", default=16, type=int,
    help="[Eagle mode] Max nodes in draft tree. Target kv_forward ar_len = next_power_of_two(max_tree_size).")
parser.add_argument("--draft_len", default=4, type=int,
    help="[Eagle mode] Chain-mode draft length (used in Phase 3 chain fallback).")
parser.add_argument("--eagle_draft_backend", default="htp",
    choices=["htp", "vulkan", "xnnpack"], type=str,
    help="[Eagle mode] Backend on which the EAGLE draft head executes.")
parser.add_argument("--eagle_hidden_io", default="fp16",
    choices=["fp16", "quant16"], type=str,
    help="[Eagle mode] Precision of the hidden-state IO that flows from target to draft.")
parser.add_argument("--eagle_layer_indices", default=None, type=str,
    help="[Eagle mode] Comma-separated indices of target layers to export (low,mid,high). "
         "Overrides EAGLE head config. Example: '2,16,30'.")
```

In `export_llama()` extend the `model_mode` branch:

```python
elif args.model_mode == "eagle":
    assert args.max_context_len >= args.prefill_ar_len, \
        "Please ensure max_context_len is >= prefill_ar_len"
    assert args.use_attention_sink is None, \
        "Eagle mode is not compatible with attention sink in v1."
    assert args.eagle_head_checkpoint is not None or args.pre_gen_pte, (
        "Eagle mode requires --eagle_head_checkpoint or --pre_gen_pte.")
    pte_filename = "eagle_target_qnn"
```

Update `pte_filenames` dict to include `EAGLE_HEAD: f"eagle_head_{args.eagle_draft_backend}"`.

Add `eagle_head_pte_path = f"{args.artifact}/{pte_filenames[EAGLE_HEAD]}.pte"`
(and the `pre_gen_pte` mirror).

Update `inference()` signature with `eagle_head_pte_path: str = None` and
add the eagle-mode branch:

```python
if args.model_mode == "eagle":
    assert eagle_head_pte_path is not None and os.path.exists(eagle_head_pte_path), \
        f"Eagle mode requires compiled head at {eagle_head_pte_path}"
    pte_paths.update({EAGLE_HEAD: eagle_head_pte_path})
    eval_results.update({"eagle_head_pte_size": os.path.getsize(eagle_head_pte_path)})
```

Add `compile_eagle_head()` stub above `compile_attention_sink_evictor()`:

```python
def compile_eagle_head(args, decoder_model_config, text_decoder_pte_path,
                       eagle_head_pte_path, tokenizer, calibration_data):
    """[Eagle mode] Phase 0 stub. Real implementation in Phase 2."""
    logging.warning("compile_eagle_head is a stub (Phase 0). Provide --pre_gen_pte to run.")
```

Wire it into `export_llama()` after the regular `compile()` call:

```python
if args.model_mode == "eagle":
    compile_eagle_head(args, decoder_model_config,
        text_decoder_pte_path, eagle_head_pte_path,
        tokenizer, calibration_data)
```

Update both `inference(...)` call sites to pass
`eagle_head_pte_path=eagle_head_pte_path`.

Update top imports: add `EAGLE_HEAD` to the
`from ...decoder_constants import (...)` block.

#### 3.3 `decoder_runtime_evaluator.py`

Add `EAGLE_HEAD` to the import block. In `DefaultEval.__init__()` after the
existing `lookahead_args = ...` block, build eagle args and merge into
`runner_args`:

```python
eagle_args = ""
if args.model_mode == "eagle":
    eagle_head_basename = os.path.basename(self.pte_paths.get(EAGLE_HEAD, ""))
    eagle_head_path = (eagle_head_basename if not args.enable_x86_64
                       else self.pte_paths.get(EAGLE_HEAD, ""))
    eagle_args = " ".join([
        f"--eagle_head_path {eagle_head_path}",
        f"--max_tree_size {args.max_tree_size}",
        f"--draft_len {args.draft_len}",
    ])
runner_args = " ".join([
    f"--eval_mode {EVAL_MODE[args.model_mode]}",
    f"--temperature {args.temperature}",
    f"--system_prompt '{args.system_prompt}'",
    lookahead_args if args.model_mode == "lookahead" else "",
    eagle_args,
])
```

#### 3.4 `runner/runner.h`

Append three default-valued params to the constructor:

```cpp
explicit Runner(
    /* ... existing params unchanged ... */
    std::unique_ptr<tokenizers::Tokenizer> tokenizer = nullptr,
    std::unique_ptr<executorch::extension::Module>
        attention_sink_rope_module = nullptr,
    // ---- Eagle mode (model_mode == "eagle") ----
    std::unique_ptr<executorch::extension::Module>
        eagle_head_module = nullptr,
    int max_tree_size = 0,
    int draft_len = 0);
```

In the private `EvalMode` enum, insert `kEagleDecoding` before `kUnsupported`:

```cpp
enum EvalMode {
  kKVCached = 0,
  kHybrid,
  kLookaheadDecoding,
  kEagleDecoding,
  kUnsupported,
};
```

Add new member fields:

```cpp
std::unique_ptr<executorch::extension::Module> eagle_head_module_;
int max_tree_size_{0};
int draft_len_{0};
```

#### 3.5 `runner/runner.cpp`

Update constructor's initializer list to capture the new params.

In `load()`'s switch over `eval_mode_`, treat eagle the same as hybrid for
prefill+kv method loading:

```cpp
case EvalMode::kHybrid:
case EvalMode::kLookaheadDecoding:
case EvalMode::kEagleDecoding:
  prompt_processor_method_name = "prefill_forward";
  token_generator_method_name = "kv_forward";
  /* ... */
```

Apply the same to the `prompt_processor_ar_len` detection conditional.

For Phase 0 only, the eagle branch falls back to plain `TokenGenerator` with
a warning, so the pte at least loads:

```cpp
} else if (eval_mode_ == EvalMode::kEagleDecoding) {
  ET_LOG(Info,
    "[Eagle mode] Phase 0 stub: running as standard TokenGenerator. "
    "max_tree_size=%d draft_len=%d eagle_head_loaded=%d",
    max_tree_size_, draft_len_, eagle_head_module_ != nullptr);
  token_generator_ = std::make_unique<TokenGenerator>(/* ... same as else branch ... */);
}
```

#### 3.6 `qnn_llama_runner.cpp`

Add three gflags:

```cpp
DEFINE_string(eagle_head_path, "",
  "[Eagle Decoding] Path to the compiled EAGLE draft head pte.");
DEFINE_int32(max_tree_size, 0,
  "[Eagle Decoding] Max nodes in draft tree (must match target compiled ar_len).");
DEFINE_int32(draft_len, 0,
  "[Eagle Decoding] Chain-mode draft length (Phase 3).");
```

Update `--eval_mode` help text to include `3: Eagle Decoding`.

Update `start_runner()` to take `std::unique_ptr<...> eagle_head_module` and
pass through to `Runner(..., FLAGS_max_tree_size, FLAGS_draft_len)`.

In `main()` load the head module if path is provided:

```cpp
std::unique_ptr<executorch::extension::Module> eagle_head_module;
if (!FLAGS_eagle_head_path.empty()) {
  eagle_head_module = std::make_unique<executorch::extension::Module>(
      FLAGS_eagle_head_path.c_str(),
      executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
}
start_runner(std::move(module), prompts,
             std::move(attention_sink_rope_module),
             std::move(eagle_head_module));
```

#### Phase 0 Acceptance

- `python -c "import ast; ast.parse(open(p).read())"` parses cleanly for
  `decoder_constants.py`, `llama.py`, `decoder_runtime_evaluator.py`.
- `lintrunner -a` clean.
- Existing CI / smoke tests for `--model_mode kv|hybrid|lookahead` produce
  byte-identical pte files compared to a pre-Phase-0 commit (compare with
  `cmp` or sha256).
- `--model_mode eagle --eagle_head_checkpoint /tmp/dummy.safetensors --compile_only`
  exits cleanly with the `compile_eagle_head` stub warning (no crash).

---

### Phase 1 — Target hidden state export (2 days)

Goal: Target model emits low/mid/high hidden states as additional outputs.
Default off → existing modes unchanged.

#### Files

- `model/static_llama.py` (the Qwen3 forward path is in the class around
  line 998 — `forward(self, tokens, atten_mask, window_atten_mask, input_pos, *args)`).

#### Edits

Add a class-level field `output_hidden_layers: Optional[List[int]] = None`
to the `LlamaModel` ctor (or whichever class implements Qwen3 forward).
When `None`, behaviour is identical to today.

Inside `forward()`, capture hidden state at each requested layer:

```python
captured_hiddens = [] if self.output_hidden_layers is not None else None
for ind, decoder_layer in enumerate(self.layers):
    # ... existing decoder_layer call ...
    if captured_hiddens is not None and ind in self.output_hidden_layers:
        captured_hiddens.append(hidden_states)

# existing norm + lm_head + return path
if self.output_cache:
    if captured_hiddens is not None:
        return logits, output_k_cache, output_v_cache, *captured_hiddens
    return logits, output_k_cache, output_v_cache
return logits
```

#### Wrapper / quantization plumbing

In `wrappers/llm_wrappers.py`:

- `TextDecoder.__init__` accepts an optional `output_hidden_layers` kwarg
  and forwards it to the model.
- `HybridTextDecoder.__init__` likewise (passes through to both `decode` and
  `prefill` instances).
- For quantization: tag the new outputs as quant IO via `TagQuantIO` only
  when `args.eagle_hidden_io == "quant16"`; otherwise leave fp16. The
  `BuildQuantIo` pass should be updated similarly, see how
  `attention_mask` quant IO is handled today as reference.

#### Acceptance

- Compile Qwen3-1.7B with hidden outputs (`--model_mode eagle --compile_only`,
  combined with the Phase 2 head wiring later — for Phase 1 alone, run a
  unit test importing `LlamaModel(... output_hidden_layers=[2,16,30])` and
  check `model.forward(...)` returns 3 extra tensors of shape `[1, ar, k]`).
- Compile + load `eagle_target_qnn.pte` standalone; query method_meta for
  `kv_forward` and assert `num_outputs == 1 (logits) + 2*num_layers (kv) + 3 (hidden)`.
- SQNR of fp32 hidden vs PTQ hidden ≥ 25 dB on calibration prompts.

---

### Phase 2 — EAGLE head Python wrapper + HF loader + compilation (3 days)

Goal: produce `eagle_head_<backend>.pte`. Standalone Python script can load
both pte's, run end-to-end, and reproduce the EAGLE Python reference's first
generated token within tolerance.

#### New file: `wrappers/eagle_wrappers.py`

```python
class EagleHead(nn.Module):
    """Single-layer transformer + (3k→k) FC + reuse target lm_head.

    Auto-detects EAGLE-1/2 (single hidden) vs EAGLE-3 (3 hidden) from
    config. The forward() signature for EAGLE-3 is:

        forward(prev_feature[B,S,k], tok_emb[B,S,k], pos[S], attn_mask,
                k_cache, v_cache)
            -> (a_out[B,S,k], k_cache_out, v_cache_out)

    The fusion FC for hidden_low/mid/high → k is applied OUTSIDE this
    module by EagleManager when assembling the prefill input from target's
    multi-hidden output. At runtime the C++ side does the same fusion.
    """

class EagleDecoderWrapper(Component):
    """Mirror of TextDecoder for the head: produces prefill_forward(ar=N)
    and kv_forward(ar=1) graphs."""

class EagleManager(Component):
    """Orchestrates target + head compilation:
       - target = HybridTextDecoder with output_hidden_layers set
       - head   = EagleDecoderWrapper compiled to args.eagle_draft_backend
    """
```

Loader: parse `args.eagle_head_checkpoint`. If it's a directory, expect
`config.json` + one or more `*.safetensors` files. Else expect a single
`*.safetensors` next to a config json (path from `--eagle_head_config`).

Important — checkpoint key name conventions for SafeAILab/EAGLE official
training repo (subject to verification when user provides actual ckpt):
`fc.weight`, `fc.bias`, `midlayer.self_attn.{q,k,v,o}_proj.weight`,
`midlayer.mlp.{gate,up,down}_proj.weight`, `midlayer.input_layernorm.weight`,
`midlayer.post_attention_layernorm.weight`, `midlayer.hidden_norm.weight`.
See <https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py>
for ground truth.

If a different training repo (HASS, SpecForge) was used, key names differ
— add a config-driven key map.

#### `wrappers/__init__.py`

Export `EagleManager`.

#### `llama.py`

In `compile()` (the regular function), branch:

```python
if args.model_mode == "eagle":
    eagle_mgr = EagleManager(control_args=args, config=decoder_model_config)
    eagle_mgr.quantize(...)
    eagle_mgr.compile(compile_specs=compile_specs, pte_filenames=pte_filenames,
                      skip_quantize=skip_quantize)
    return  # eagle has its own pipeline; don't run multimodal_mgr
```

Replace the `compile_eagle_head` stub from Phase 0 — this work now happens
inside `compile()` itself.

#### Layer-3 backend selection (xnnpack/vulkan)

Compiling the head to xnnpack:

```python
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

edge_program = to_edge(exported_program)
delegated = edge_program.to_backend(XnnpackPartitioner())
exec_program = delegated.to_executorch(...)
```

Compiling to vulkan:

```python
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
delegated = edge_program.to_backend(VulkanPartitioner())
```

(Verify exact API on current main; may use `to_edge_transform_and_lower`
helpers that wrap this.)

#### Acceptance

- Standalone Python script `tools/eagle_pyref.py` (helper, optional):
  - loads HF Qwen3-1.7B + EAGLE head safetensors with the EAGLE official
    `cnets.py` reference impl;
  - runs a fixed prompt;
  - logs the first 5 sampled tokens.
- Equivalent path through ExecuTorch (compile + run via Module API on x86):
  reproduces same 5 tokens at temperature 0 within argmax-equality
  (allowing for fp16 numerics, top-1 must match).

---

### Phase 3 — Runtime chain spec decoding (3 days)

Goal: end-to-end correctness with **chain** verification (no tree). Output
text bit-identical to non-spec target output at temperature 0.

#### New files

- `runner/eagle_kv_manager.{h,cpp}` — 1-layer specialization. Reuses the
  same `KVCache` struct from `kv_manager.h`, but constructs only
  `head_num_layers` (= 1) entries. Mostly a copy-paste-and-trim of
  `KVManager` parameterised by num_layers (you can also just instantiate
  `KVManager` with `num_layers=1` if its interface allows it without
  baking in target-specific assumptions — verify first).
- `runner/eagle_token_generator.{h,cpp}` — extends `TokenGenerator`. New
  fields: `draft_module_`, `eagle_kv_manager_`, hidden-state output tensor
  handles into target outputs, `prev_feature_buffer_`. `generate()`
  override implements the diagram in §2 with chain (`draft_len` proposals,
  `draft_len + 1` verify positions).

#### Edits to `runner/runner.cpp`

Replace the Phase-0 stub branch with real `EagleTokenGenerator`
construction. Wire `draft_module_` from the loaded `eagle_head_module_`.

#### KV rollback

`KVManager::update_cache(ar_len, pos, n_update, selected)` already takes a
`selected[]` boolean vector. Pass `selected[i] = (i < n_accepted)` for
both the target manager and the eagle KV manager. This commits only the
accepted prefix of the verify ar slot.

#### Hidden-state shuttling

After target verify:

1. Read 3 hidden tensors from target outputs (already wired by
   `decoder_runner_->set_outputs()` plumbing — extend it to also bind
   hidden output buffers).
2. For position `accepted_tail`, compute `g = FC_3kx_k(concat(L,M,H))`.
   The FC weight is part of the EAGLE head and is already loaded in the
   head pte's prefill_forward graph — but in chain runtime we need it
   on the host. **Decision:** include the 3k→k FC as a separate small
   tensor in the head pte's `get_attribute("fusion_fc")`, or alternately
   bake it into a trivial 1-token "fuse" graph. Easiest: bake fusion as
   the first op of the head's `prefill_forward`, expose `kv_forward`
   that takes already-fused features. Then runtime doesn't see the FC
   at all.

   → **Recommended:** dual head graphs:
     - `prefill_forward`: input is `(hidden_LMH[B,S,3k], tok_emb[B,S,k], pos, mask, kv)`
       → first op is fuse-FC, then transformer.
     - `kv_forward`: input is `(prev_a[B,1,k], tok_emb[B,1,k], pos, mask, kv)`
       → no FC, just transformer.

   This keeps runtime simple.

3. Memcpy the resulting `g` into the head's `kv_forward` `prev_feature`
   input buffer for the next draft phase's first step.

#### Acceptance

- For temperature 0 + greedy acceptance: runtime emits the same token
  sequence as `--model_mode hybrid` on the same target pte (without
  hidden outputs). I.e. target-equivalence holds.
- Logged "accept rate" per generation > 0.5 on MT-bench short prompts.
- KV rollback verified by running with `draft_len = 1` and asserting
  identical output to chain mode + `draft_len = 4`.

---

### Phase 4 — Tree verification (3 days)

#### New files

- `runner/tree_attention.{h,cpp}` — tree topology + tree-attention mask.

For Phase 4 v1, use a **static balanced tree** of fixed shape — the
exact shape (depths × branching factor) is config-driven via
`--tree_topology` (e.g. `1-2-2-2` = root has 1 child, depth 1 has 2
each, etc.). EAGLE-2's dynamic confidence-pruned tree is a future
improvement (Phase 4.1, optional).

#### Edits

- `EagleTokenGenerator::generate()`: tree-mode branch when
  `max_tree_size > 0` (compile-time budget). Build node list, populate
  target ar buffer with `[cur_token, ...tree_tokens]`, fill tree-attention
  mask `[ar_len, ar_len]` per the tree topology, fill positions per
  node depth.
- Path walking: scan the tree breadth-first; at each parent with
  accepted token, check children; longest accepted path wins. Commit
  exactly that path's KV positions.

#### Acceptance

- Accept rate on MT-bench ≥ chain version's accept rate.
- Output bit-identical to chain version at temperature 0.

---

### Phase 5 — Heterogeneous draft (1 week)

Two pieces of work.

#### 5.1 Build path: lower head to vulkan / xnnpack (3 days)

Already partially set up by Phase 2's `--eagle_draft_backend` plumbing.
Concretely test:

- `--eagle_draft_backend xnnpack`: head goes through XNNPACK partitioner.
- `--eagle_draft_backend vulkan`: head goes through Vulkan partitioner.

Risks (cover with skip-and-warn fallbacks):
- Vulkan may not support all Qwen3 ops (rope, rmsnorm); fallback to
  partial delegation — XNNPACK / portable kernels handle the rest.
- Vulkan's KV-cache handling for stateful AR may require workarounds;
  XNNPACK is safer.

#### 5.2 Async pipeline scheduling (4 days)

Currently runtime is fully serial: target verify → draft N steps → target
verify. With heterogeneous backends we can overlap.

Add a `std::thread` worker in `EagleTokenGenerator`:

```
main thread:                 worker thread:
  target verify (NPU)     ←→  draft N steps (GPU/CPU) for NEXT iteration
  accept/reject               ↑
  (use precomputed draft) ←───┘
```

This is **speculative-on-speculative**: we precompute next draft tree based
on most-likely accepted token from current verify. If verify rejects,
worker's tree is discarded. On accept rates ≥ 70%, expected wasted compute
< 30%.

Add CLI flag `--eagle_pipeline {sync,async}`, default `sync`.

#### Acceptance

- `--eagle_draft_backend {htp,vulkan,xnnpack}` all produce valid pte.
- end-to-end token rate measured on Adreno + Hexagon vs HTP-only baseline
  for both `sync` and `async` modes.

---

### Phase 6 — Benchmark harness + design doc (2 days)

#### Files

- Rewrite `examples/qualcomm/oss_scripts/llama/draft_mode_design.md` →
  `eagle_mode_design.md` with the architecture diagram from §2 of this
  plan plus EAGLE-specific protocol.
- Add `tools/eagle_benchmark.py` (or a section in `decoder_runtime_evaluator.py`)
  that sweeps `(max_tree_size, draft_len, eagle_hidden_io, eagle_draft_backend)`
  and emits a CSV: `accept_rate, prefill_ms, decode_ms, tok/s, pte_size`.

---

## 4. Files at a Glance

| Path | Phase touched | Notes |
|---|---|---|
| `decoder_constants.py` | 0 | Add `EVAL_MODE["eagle"]`, `EAGLE_TARGET`, `EAGLE_HEAD` |
| `llama.py` | 0, 2 | CLI flags, eagle compile branch, `compile_eagle_head` |
| `wrappers/__init__.py` | 2 | Export `EagleManager` |
| `wrappers/eagle_wrappers.py` (NEW) | 2 | `EagleHead`, `EagleDecoderWrapper`, `EagleManager` |
| `wrappers/base_component.py` | 1 | `process_model_args` eagle ar_len = next_pow2(max_tree_size) |
| `model/static_llama.py` | 1 | `output_hidden_layers` optional ctor arg |
| `wrappers/llm_wrappers.py` | 1 | Plumb `output_hidden_layers` through `TextDecoder` / `HybridTextDecoder` |
| `decoder_runtime_evaluator.py` | 0 | Append eagle args to runner cmd |
| `runner/runner.h` | 0, 3 | `kEagleDecoding` enum, eagle fields, ctor params |
| `runner/runner.cpp` | 0, 3 | Eagle method routing, eagle generator construction |
| `runner/eagle_kv_manager.{h,cpp}` (NEW) | 3 | 1-layer KV manager |
| `runner/eagle_token_generator.{h,cpp}` (NEW) | 3, 4 | Speculative loop, tree mode |
| `runner/tree_attention.{h,cpp}` (NEW) | 4 | Tree topology + mask |
| `qnn_llama_runner.cpp` | 0 | Gflags + module load |
| `CMakeLists.txt` | 3, 4 | Add new sources |
| `runner/draft_token_generator.{h,cpp}` | 0 | **DELETE** |
| `draft_mode_design.md` | 6 | Replace with `eagle_mode_design.md` |

---

## 5. Risks / Notes

- **Qwen3 r3 optimisation.** `Qwen3_1_7B.r3 = True` in `__init__.py`. The r3
  pass alters attention shapes; verify hidden-state capture happens
  **after** the original layer output (pre-r3 transform) or **after** r3
  inverse — whichever matches the EAGLE head's training-time expectation.
  Confirm with the user / training script.
- **Quantizing hidden IO** can drop accept rate sharply. Default fp16.
- **Vulkan support for Qwen3 ops** — verify before committing to that path.
  If vulkan can't lower, fall back to xnnpack messaging at compile time.
- **EAGLE head ckpt key naming** — verify against actual file before
  finalising loader. Most likely matches SafeAILab official naming.
- **Tokenizer drift between Qwen3 and EAGLE training** — should be
  identical (head shares lm_head + embedding) but worth a sanity assert
  at compile time.

---

## 6. How to resume after migration

On the new server (`sbo`):

```bash
ssh sbo  # zqchen@47.99.72.246:40600 from your ~/.ssh/config
cd /path/to/executorch/checkout
git status                                            # confirm clean
git log --oneline -5                                  # confirm starting commit

# Read this plan
$EDITOR examples/qualcomm/oss_scripts/llama/eagle_mode_plan.md

# Start from §3.0 cleanup, then §3.1 onward.
# Use the TaskCreate tool (or your own todo file) to track Phase 0..6.
# Acceptance criteria per phase are the gating signals.
```

Hand-off complete. Good luck.
