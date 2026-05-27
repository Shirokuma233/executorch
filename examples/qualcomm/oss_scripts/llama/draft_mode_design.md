# Draft Mode — Speculative Decoding for QNN LLM Runner

## Overview

Draft mode adds a fourth inference mode (`--model_mode draft`) to the existing
KV Cache / Hybrid / Lookahead pipeline.  It implements **speculative decoding**:

- A **small draft model** (CPU, fp32) proposes `draft_len` candidate tokens per
  step.
- The **large target model** (QNN NPU, quantized) verifies all candidates in a
  single forward pass of length `draft_len + 1`.
- Accepted tokens are committed; the first rejected position restarts the draft
  loop.

Because the target model runs only once per `draft_len` tokens (instead of once
per token), throughput improves when the draft model's acceptance rate is high.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Draft Mode Speculative Decoding Loop                           │
│                                                                 │
│  cur_token                                                      │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Draft Phase  (CPU, fp32, draft_len steps)               │  │
│  │  draft_model.kv_forward(tok, mask, pos) → next_tok       │  │
│  │  Repeat draft_len times → draft_tokens[0..draft_len-1]   │  │
│  └──────────────────────────────────────────────────────────┘  │
│      │                                                          │
│      ▼  [cur_token, draft_tokens...]  (ar_len = draft_len+1)   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Verify Phase  (QNN NPU, quantized, 1 forward pass)      │  │
│  │  target_model.kv_forward(tokens, mask, pos) → logits     │  │
│  └──────────────────────────────────────────────────────────┘  │
│      │                                                          │
│      ▼                                                          │
│  Accept/Reject:                                                 │
│    pos 0 → always accept (target's prediction for cur_token)   │
│    pos i → accept if target[i] == draft_tokens[i-1]            │
│             reject otherwise → use target[i], restart draft    │
└─────────────────────────────────────────────────────────────────┘
```

### Comparison with other modes

| Mode       | Prefill model | Decode model | Tokens/NPU call |
|------------|---------------|--------------|-----------------|
| kv         | —             | NPU (ar=1)   | 1               |
| hybrid     | NPU (ar=N)    | NPU (ar=1)   | 1               |
| lookahead  | NPU (ar=N)    | NPU (ar=W)   | up to N         |
| **draft**  | NPU (ar=N)    | NPU (ar=D+1) | up to D+1       |

`D = draft_len`, `N = prefill_ar_len`, `W = lookahead window`.

---

## Files Changed

### Python (export / quantization / compilation)

| File | Change |
|------|--------|
| `decoder_constants.py` | Added `DRAFT_MODEL = "draft_model"` constant; added `"draft": 3` to `EVAL_MODE` |
| `wrappers/__init__.py` | Exported `DraftManager` |
| `wrappers/llm_wrappers.py` | Added `DraftDecoder` and `DraftManager` classes |
| `llama.py` | Added `--draft_len`, `--draft_model_checkpoint`, `--draft_model_params` args; draft pte filename; draft compile/inference wiring |
| `decoder_runtime_evaluator.py` | Added `draft_model_pte_path` param to `DefaultEval`; draft runner args (`--draft_len`, `--draft_model_path`); push draft pte to device |

### C++ (runtime)

| File | Change |
|------|--------|
| `runner/draft_token_generator.h` | New: `DraftTokenGenerator<T>` class declaration |
| `runner/draft_token_generator.cpp` | New: speculative decoding loop implementation |
| `runner/runner.h` | Added `kDraftDecoding=3` to `EvalMode`; added `draft_len_` and `draft_module_` fields; updated constructor signature |
| `runner/runner.cpp` | Updated constructor; `load()` instantiates `DraftTokenGenerator` for `kDraftDecoding`; includes `draft_token_generator.h` |
| `qnn_llama_runner.cpp` | Added `--draft_len` and `--draft_model_path` gflags; loads draft module; passes to `start_runner` |

---

## New Classes

### `DraftDecoder` (Python, `llm_wrappers.py`)

Wraps a small model for CPU-only export.  It reuses `TextDecoder` with
`model_mode="kv"` (ar_len=1) and exports to a standard ExecuTorch `.pte` via
the CPU backend (no QNN lowering).

### `DraftManager` (Python, `llm_wrappers.py`)

Orchestrates the two-model compilation:
- `HybridTextDecoder` → target model on QNN NPU (same as hybrid mode)
- `DraftDecoder` → draft model on CPU

### `DraftTokenGenerator<T>` (C++, `runner/`)

Inherits from `TokenGenerator<T>`.  Overrides `generate()` with the
speculative decoding loop:
1. `draft_step()` — runs the CPU draft module one token at a time.
2. `prepare_verify_io()` — fills the target model's input buffer with
   `[cur_token, draft_tokens...]`.
3. Accept/reject loop — commits accepted tokens, restarts on first mismatch.

---

## Usage

### Step 1 — Compile

```bash
python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android -m SM8750 \
  --checkpoint models/Llama-3.2-1B-Instruct/consolidated.00.pth \
  --params models/Llama-3.2-1B-Instruct/params.json \
  --tokenizer_model models/Llama-3.2-1B-Instruct/tokenizer.model \
  --decoder_model llama3_2-1b_instruct \
  --model_mode draft \
  --prefill_ar_len 128 \
  --max_seq_len 1024 \
  --draft_len 4 \
  --draft_model_checkpoint models/SmolLM2-135M/consolidated.00.pth \
  --draft_model_params models/SmolLM2-135M/params.json \
  --prompt "I would like to learn python, could you teach me with a simple example?" \
  --artifact models/draft_output \
  --compile_only
```

This produces:
- `models/draft_output/draft_llama_qnn.pte` — target model (QNN NPU)
- `models/draft_output/draft_model.pte` — draft model (CPU fp32)

### Step 2 — Run inference

```bash
python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android -s ${SERIAL_NUM} -m SM8750 \
  --checkpoint models/Llama-3.2-1B-Instruct/consolidated.00.pth \
  --params models/Llama-3.2-1B-Instruct/params.json \
  --tokenizer_model models/Llama-3.2-1B-Instruct/tokenizer.model \
  --decoder_model llama3_2-1b_instruct \
  --model_mode draft \
  --prefill_ar_len 128 \
  --max_seq_len 1024 \
  --draft_len 4 \
  --prompt "I would like to learn python, could you teach me with a simple example?" \
  --pre_gen_pte models/draft_output
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--draft_len` | 4 | Tokens the draft model proposes per step |
| `--draft_model_checkpoint` | None | Draft model checkpoint path (falls back to target checkpoint) |
| `--draft_model_params` | None | Draft model params.json (falls back to target params) |

---

## Design Decisions

**Draft model runs on CPU (fp32)**  
The draft model is exported without QNN lowering.  This avoids the complexity
of managing two separate NPU contexts and KV caches on the NPU.  CPU execution
is acceptable because the draft model is small (e.g. 135M parameters) and its
latency is dominated by the NPU target model call.

**Target model ar_len = draft_len + 1**  
The target model's `kv_forward` method is compiled with `ar_len = draft_len + 1`
so it can verify all draft tokens plus the current token in one NPU call.  This
is set automatically by `process_model_args` when `model_mode == "draft"`.

**Draft model uses no KV cache (simplified)**  
The current implementation passes the full attention mask each step.  A future
optimisation is to export the draft model with a KV cache for lower CPU latency.

**Acceptance is greedy**  
The current implementation uses greedy argmax for both draft and target
sampling.  Stochastic acceptance (as in the original speculative decoding paper)
can be added by replacing the equality check with a probability-ratio test.
