#!/usr/bin/env python3
"""Evaluate Qwen3-8B base model on RULER CWE task at 64K context length.

Uses lm-eval-harness (v0.4.11) with built-in RULER tasks.
CWE (Common Word Extraction): given a long context with words at varying
frequencies, the model must identify the most frequent words.

Reference: https://arxiv.org/abs/2404.06654

Usage:
    python eval_ruler_baseline.py                        # Full CWE eval at 64K
    python eval_ruler_baseline.py --limit 5              # Quick sanity check
    python eval_ruler_baseline.py --task ruler           # All 13 RULER tasks
"""

import subprocess
import sys
import os
import json
import argparse
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "Qwen--Qwen3-8b")


def patch_config_if_needed(model_path: str, context_length: int) -> bool:
    """Patch max_position_embeddings in config.json if < context_length.
    Returns True if patched (caller should restore after eval)."""
    config_path = os.path.join(model_path, "config.json")
    backup_path = config_path + ".ruler_backup"
    with open(config_path) as f:
        cfg = json.load(f)
    cur = cfg.get("max_position_embeddings", 0)
    if cur >= context_length:
        return False
    print(f"[INFO] Patching max_position_embeddings: {cur} -> {context_length}")
    shutil.copy2(config_path, backup_path)
    cfg["max_position_embeddings"] = context_length
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return True


def restore_config(model_path: str):
    config_path = os.path.join(model_path, "config.json")
    backup_path = config_path + ".ruler_backup"
    if os.path.exists(backup_path):
        print("[INFO] Restoring original config.json")
        shutil.move(backup_path, config_path)


def run_eval(
    model_path: str,
    output_dir: str,
    task: str = "ruler_cwe",
    context_length: int = 65536,
    num_fewshot: int = 0,
    batch_size: int = 1,
    seed: int = 42,
    limit: int | None = None,
):
    """Run lm-eval on a RULER task."""
    # NOTE: lm-eval 0.4.11 has a bug where it passes `dtype` (not `torch_dtype`)
    # to from_pretrained. We patched huggingface.py to use torch_dtype instead.
    model_args = (
        f"pretrained={model_path},"
        f"max_length={context_length},"
        f"dtype=bfloat16,"
        f"trust_remote_code=True,"
        f"use_fast_tokenizer=False"
    )

    cmd = [
        "lm-eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", task,
        "--batch_size", str(batch_size),
        "--output_path", output_dir,
        "--log_samples",
        "--seed", str(seed),
        "--metadata", json.dumps({"max_seq_lengths": [context_length]}),
    ]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if num_fewshot > 0:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    print(f"\nRunning:\n  {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def main():
    parser = argparse.ArgumentParser(description="RULER baseline eval for Qwen3-8B")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--output_dir",
                        default=os.path.join(PROJECT_ROOT,
                                             "eval_results/ruler_cwe_qwen3-8b_64k"))
    parser.add_argument("--task", default="ruler_cwe",
                        choices=["ruler_cwe", "ruler_fwe", "ruler_vt",
                                 "niah_single_1", "niah_single_2",
                                 "ruler_qa_squad", "ruler_qa_hotpot", "ruler"])
    parser.add_argument("--context_length", type=int, default=65536)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit examples (for debugging)")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    output_dir = os.path.abspath(args.output_dir)

    print("=== RULER Evaluation ===")
    print(f"  Model:    {model_path}")
    print(f"  Task:     {args.task}")
    print(f"  Context:  {args.context_length}")
    print(f"  Output:   {output_dir}")

    patched = patch_config_if_needed(model_path, args.context_length)
    try:
        rc = run_eval(
            model_path=model_path,
            output_dir=output_dir,
            task=args.task,
            context_length=args.context_length,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            seed=args.seed,
            limit=args.limit,
        )
    finally:
        if patched:
            restore_config(model_path)

    sys.exit(rc)


if __name__ == "__main__":
    main()
