"""
Wrapper to run llama.py over all prompts in Spec-Bench question.jsonl.

Usage:
    python run_specbench.py --prompt_file data/spec_bench/question.jsonl [options]

All other flags are forwarded verbatim to llama.py. --prompt is replaced by
the prompts loaded from --prompt_file. For multi-turn questions, all turns are
passed as multiple --prompt arguments (llama.py nargs="+").

Example (mirrors the original command, minus --prompt):
    python run_specbench.py \
        --prompt_file /mnt/hdd-ws/users/zqchen/study/device_llm_datasets/Spec-Bench/data/spec_bench/question.jsonl \
        -b build-android -s 10.87.61.11:41153 -m SM8750 \
        --checkpoint models/Llama-3.2-1B-Instruct/consolidated.00.pth \
        --params models/Llama-3.2-1B-Instruct/params.json \
        --tokenizer_model models/Llama-3.2-1B-Instruct/tokenizer.model \
        --decoder_model llama3_2-3b_instruct \
        --model_mode hybrid \
        --prefill_ar_len 32 \
        --max_seq_len 1024 \
        --pre_gen_pte models/Llama-3.2-1B-Instruct/llama3_2-3b_instruct_hybrid_32_1024 \
        --skip_push \
        --output_dir outputs \
        --limit 5

Optional filters:
    --category translation       # only run one category
    --single_turn_only           # skip multi-turn questions
    --limit 10                   # stop after N questions
    --output_dir results/        # save per-question stdout to files
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys


LLAMA_SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "/mnt/hdd-ws/users/zqchen/study/executorch/examples/qualcomm/oss_scripts/llama/llama.py",
)


def load_questions(path, category=None, single_turn_only=False, limit=None):
    questions = []
    with open(path) as f:
        for line in f:
            q = json.loads(line)
            if category and q["category"] != category:
                continue
            if single_turn_only and len(q["turns"]) > 1:
                continue
            questions.append(q)
            if limit and len(questions) >= limit:
                break
    return questions


def build_base_cmd(extra_args):
    """Build the base command from forwarded args (everything except --prompt)."""
    cmd = [sys.executable, LLAMA_SCRIPT]
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg == "--prompt":
            # skip --prompt and its values from forwarded args
            i += 1
            while i < len(extra_args) and not extra_args[i].startswith("-"):
                i += 1
        else:
            cmd.append(arg)
            i += 1
    return cmd


def run_question(base_cmd, question, output_dir=None):
    qid = question["question_id"]
    category = question["category"]
    turns = question["turns"]

    cmd = base_cmd + ["--prompt"] + turns

    print(f"\n[{qid}] category={category}  turns={len(turns)}")
    print("  prompt[0]:", turns[0][:80] + ("..." if len(turns[0]) > 80 else ""))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{qid}_{category}.txt")
        with open(out_path, "w") as f:
            f.write(f"question_id: {qid}\ncategory: {category}\n")
            f.write(f"turns:\n")
            for i, t in enumerate(turns):
                f.write(f"  [{i}] {t}\n")
            f.write("\n--- stdout ---\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- stderr ---\n")
                f.write(result.stderr)

    if result.returncode != 0:
        print(f"  [ERROR] exit code {result.returncode}")
        if result.stderr:
            print(result.stderr[-500:])
    else:
        # print last few lines of output as a quick preview
        lines = result.stdout.strip().splitlines()
        for line in lines[-5:]:
            print(" ", line)

    return result.returncode


def extract_latencies(output_dir):
    """Extract latency metrics from all txt files and compute geometric means."""
    if not output_dir or not os.path.exists(output_dir):
        return

    total_inference_times = []
    prompt_eval_times = []
    token_gen_times = []

    # Regex patterns to extract the time values
    total_pattern = re.compile(r"Total inference time:\s+([\d.]+)\s+\(seconds\)")
    prompt_pattern = re.compile(r"Prompt evaluation:\s+([\d.]+)\s+\(seconds\)")
    token_pattern = re.compile(r"Generated \d+ tokens:\s+([\d.]+)\s+\(seconds\)")

    # Process all txt files in output_dir
    for filename in os.listdir(output_dir):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(output_dir, filename)
        with open(filepath, "r") as f:
            content = f.read()

            # Extract values
            total_match = total_pattern.search(content)
            prompt_match = prompt_pattern.search(content)
            token_match = token_pattern.search(content)

            if total_match:
                total_inference_times.append(float(total_match.group(1)))
            if prompt_match:
                prompt_eval_times.append(float(prompt_match.group(1)))
            if token_match:
                token_gen_times.append(float(token_match.group(1)))

    # Compute geometric means
    def geometric_mean(values):
        if not values:
            return None
        product = 1.0
        for v in values:
            product *= v
        return product ** (1.0 / len(values))

    total_gmean = geometric_mean(total_inference_times)
    prompt_gmean = geometric_mean(prompt_eval_times)
    token_gmean = geometric_mean(token_gen_times)

    # Write results
    output_path = os.path.join(output_dir, "inference_latency.txt")
    with open(output_path, "w") as f:
        f.write("Geometric Mean of Latency Metrics\n")
        f.write("=" * 50 + "\n\n")

        if total_gmean is not None:
            f.write(f"Total inference time:     {total_gmean:.6f} seconds\n")
            f.write(f"  (computed from {len(total_inference_times)} samples)\n\n")

        if prompt_gmean is not None:
            f.write(f"Prompt evaluation:        {prompt_gmean:.6f} seconds\n")
            f.write(f"  (computed from {len(prompt_eval_times)} samples)\n\n")

        if token_gmean is not None:
            f.write(f"Generated tokens:         {token_gmean:.6f} seconds\n")
            f.write(f"  (computed from {len(token_gen_times)} samples)\n")

    print(f"\nLatency statistics written to {output_path}")
    if total_gmean:
        print(f"  Total inference time (geometric mean): {total_gmean:.6f}s")
    if prompt_gmean:
        print(f"  Prompt evaluation (geometric mean):    {prompt_gmean:.6f}s")
    if token_gmean:
        print(f"  Token generation (geometric mean):     {token_gmean:.6f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Run llama.py over Spec-Bench questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt_file",
        required=True,
        help="Path to question.jsonl",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Only run questions of this category",
    )
    parser.add_argument(
        "--single_turn_only",
        action="store_true",
        help="Skip multi-turn questions",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many questions",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save per-question output files",
    )
    parser.add_argument(
        "--extract_only",
        action="store_true",
        help="Only extract latency metrics from existing txt files, don't run questions",
    )

    # parse only our flags; everything else is forwarded to llama.py
    args, extra_args = parser.parse_known_args()

    # If extract_only mode, just process existing files and exit
    if args.extract_only:
        if not args.output_dir:
            print("Error: --extract_only requires --output_dir")
            return 1
        extract_latencies(args.output_dir)
        return 0

    questions = load_questions(
        args.prompt_file,
        category=args.category,
        single_turn_only=args.single_turn_only,
        limit=args.limit,
    )
    print(f"Loaded {len(questions)} questions from {args.prompt_file}")

    base_cmd = build_base_cmd(extra_args)

    errors = 0
    for q in questions:
        rc = run_question(base_cmd, q, output_dir=args.output_dir)
        if rc != 0:
            errors += 1

    print(f"\nDone. {len(questions)} questions, {errors} errors.")

    # Extract latency metrics and compute geometric means
    extract_latencies(args.output_dir)


if __name__ == "__main__":
    main()
