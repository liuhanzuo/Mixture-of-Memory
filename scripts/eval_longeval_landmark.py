"""LongEval lines-retrieval evaluation for faithful Landmark Attention.

Run with external/landmark_venv/bin/python on an H20 node. Kept separate from
mem_space LongEval so Landmark's torch2.1/transformers4.28 environment stays
isolated from the Llama-3 mem_space code path.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDMARK_LLAMA_DIR = os.path.join(PROJECT_ROOT, "external", "landmark-attention", "llama")
for _p in (PROJECT_ROOT, LANDMARK_LLAMA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.run_babilong_landmark import (  # noqa: E402
    generate_landmark,
    insert_landmark_tokens,
    load_landmark_pipeline,
)

_LENGTH_TOKENS = {"1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768}
_PROMPT_HEADER = (
    "Below is a record of lines I want you to remember. Each line begins with "
    "'line <line index>' and contains a '<REGISTER_CONTENT>' at the end of the "
    "line as a numerical value. For each line index, memorize its corresponding "
    "<REGISTER_CONTENT>. At the end of the record, I will ask you to retrieve the "
    "corresponding <REGISTER_CONTENT> of a certain line index. Now the record "
    "start:\n\n"
)
_NUM_RE = re.compile(r"\d{4,}")


def _stable_hash(text: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(text)) % 100000


def _random_label(rng: random.Random) -> str:
    def word() -> str:
        return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(5, 9)))
    return f"{word()}-{word()}"


def build_lines_prompt(target_tokens: int, tokenizer, rng: random.Random):
    labels: list[str] = []
    lines: list[str] = []
    values: list[str] = []

    def render(query_label: str) -> str:
        query = (
            f"\nNow the record is over. Tell me what is the <REGISTER_CONTENT> in "
            f"line {query_label}? I need the number.\nThe <REGISTER_CONTENT> in line "
            f"{query_label} is"
        )
        return _PROMPT_HEADER + "".join(lines) + query

    while True:
        for _ in range(64):
            label = _random_label(rng)
            value = str(rng.randint(100000, 999999))
            labels.append(label)
            values.append(value)
            lines.append(f"line {label}: REGISTER_CONTENT is <{value}>\n")
        n_tok = len(tokenizer.encode(render(labels[len(labels) // 2]), add_special_tokens=True))
        if n_tok >= target_tokens:
            break

    ti = rng.randrange(len(labels))
    target_label = labels[ti]
    return render(target_label), values[ti], target_label, len(labels)


def extract_prediction(output: str) -> str:
    m = _NUM_RE.search(output)
    return m.group(0) if m else ""


def main():
    p = argparse.ArgumentParser(description="LongEval lines-retrieval eval for Landmark Attention")
    p.add_argument("--ckpt_path", type=str, default=os.path.join(PROJECT_ROOT, "external", "landmark_ckpts", "landmark_tuned"))
    p.add_argument("--results_folder", type=str, default="./longeval_results")
    p.add_argument("--output_name", type=str, required=True)
    p.add_argument("--lengths", nargs="+", default=["4k", "8k", "16k", "32k"])
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--insert_landmarks", action="store_true", default=True)
    p.add_argument("--no_insert_landmarks", dest="insert_landmarks", action="store_false")
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[longeval-landmark] ckpt={args.ckpt_path} lengths={args.lengths} n={args.num_samples} top_k={args.top_k}")
    pipe, tokenizer = load_landmark_pipeline(args.ckpt_path, args.top_k, args.device, dtype=dtype)
    mem_freq = pipe.model.config.mem_freq

    outdir = Path(args.results_folder) / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    for length in args.lengths:
        if length not in _LENGTH_TOKENS:
            print(f"[WARN] unknown length {length}, skipping")
            continue
        rng = random.Random(args.seed + _stable_hash(length))
        target_tokens = _LENGTH_TOKENS[length]
        records = []
        correct = 0
        n_lines_seen = 0
        n_tok_seen = 0
        for i in tqdm(range(args.num_samples), desc=length, leave=False):
            prompt, expected, label, n_lines = build_lines_prompt(target_tokens, tokenizer, rng)
            n_lines_seen = n_lines
            n_tok_seen = len(tokenizer.encode(prompt, add_special_tokens=True))
            if args.insert_landmarks:
                prompt = insert_landmark_tokens(prompt, tokenizer, mem_freq)
            try:
                out = generate_landmark(pipe, prompt, args.max_new_tokens)
            except Exception as exc:
                print(f"[ERROR] {length} sample={i} generation failed: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                out = ""
            pred = extract_prediction(out)
            ok = pred == expected
            correct += int(ok)
            records.append({
                "label": label,
                "expected": expected,
                "output": out,
                "pred": pred,
                "correct": ok,
                "n_lines": n_lines,
                "n_tokens": int(n_tok_seen),
            })
        acc = correct / len(records) if records else 0.0
        summary[length] = {"accuracy": acc, "correct": correct, "total": len(records), "approx_lines": n_lines_seen, "approx_tokens": n_tok_seen}
        outfile = outdir / f"longeval_{length}.json"
        with open(outfile, "w") as f:
            json.dump({"length": length, "summary": summary[length], "records": records}, f, indent=2)
        print(f"[longeval-landmark] {length}: acc={acc:.3f} ({correct}/{len(records)}) ~{n_lines_seen} lines / {n_tok_seen} tok -> {outfile}")
    print("\n[longeval-landmark] SUMMARY")
    for length, s in summary.items():
        print(f"  {length:>4}: acc={s['accuracy']:.3f} ({s['correct']}/{s['total']}) ~{s['approx_tokens']} tok")
    with open(outdir / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
