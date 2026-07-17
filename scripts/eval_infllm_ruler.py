#!/usr/bin/env python
"""InfLLM baseline — RULER (NIAH / variable_tracking) eval driver.

Head-to-head training-free peer of ``scripts/eval_ruler_qcmem.py``: SAME RULER
sample set (``ruler._build_sample`` + per-(task,length,i) seeds + shard filter),
SAME scoring (``ruler._string_match_all_one``), SAME on-disk layout (per-cell CSV
+ per-cell JSON with ``summary.score`` / ``summary.n`` and ``_shard{i}of{N}``
tags), so the existing merge/score path treats InfLLM and QCMem identically.
ONLY the model forward differs — InfLLM training-free memory attention
(``scripts/infllm_qwen3.py``) instead of QCMem write/read resume.

Example (full eval on node .73, run per shard/group by _eval_taskpool_2group.sh
style scheduling):
    python scripts/eval_infllm_ruler.py \
        --model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --ruler_tasks niah_single niah_multi vt --lengths 8k 16k 32k \
        --limit 100 --use_chat_template \
        --output_name ruler_infllm --results_folder ruler_results/infllm \
        --num_shards 4 --shard_index 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# RULER task framework (generation + scoring) — reused verbatim, unmodified.
import scripts.eval_ruler_mem_space as ruler  # noqa: E402
# QCMem babilong module — only for the shared QUOTE_ALL CSV writer.
import scripts.eval_qcmem_babilong as qcb  # noqa: E402
import scripts.infllm_qwen3 as infllm  # noqa: E402

_TASK_ALIAS = {
    "niah_single": "niah_single_2",
    "niah_single_noise": "niah_single_1",
    "niah_single_essay": "niah_single_2",
    "niah_multi": "niah_multikey_1",
    "niah_multikey": "niah_multikey_1",
    "vt": "variable_tracking",
}
_CANONICAL_TASKS = {
    "niah_single_1", "niah_single_2", "niah_multikey_1", "variable_tracking",
}


def _resolve_task(name: str) -> str:
    if name in _CANONICAL_TASKS:
        return name
    if name in _TASK_ALIAS:
        return _TASK_ALIAS[name]
    raise ValueError(f"unknown ruler task {name!r}")


def _im_end_ids(tokenizer):
    """Chat end tokens so decode stops at <|im_end|> (Qwen3 chat)."""
    ids = []
    try:
        tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(tid, int) and tid >= 0:
            ids.append(tid)
    except Exception:
        pass
    return ids


def main():
    parser = argparse.ArgumentParser(description="InfLLM baseline — RULER eval")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--results_folder", type=str, default="./ruler_results")
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--ruler_tasks", type=str, nargs="+",
                        default=["niah_single", "niah_multi", "vt"])
    parser.add_argument("--lengths", type=str, nargs="+",
                        default=["8k", "16k", "32k"])
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_chat_template", action="store_true", default=False)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    # InfLLM memory-config overrides (defaults = infllm.DEFAULT_MEM_CONFIG)
    parser.add_argument("--n_local", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--n_init", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="InfLLM prefill chunk size (execution granularity).")
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        parser.error("bad shard config")

    tasks = [_resolve_task(t) for t in args.ruler_tasks]
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    mem_override = {}
    for k in ("n_local", "topk", "block_size", "n_init", "chunk_size"):
        v = getattr(args, k)
        if v is not None:
            mem_override[k] = v

    print(f"[InfLLM-RULER] model_path={args.model_path}")
    print(f"[InfLLM-RULER] tasks={tasks} lengths={args.lengths} limit={args.limit} "
          f"chat={args.use_chat_template} think={args.enable_thinking}")

    model, tokenizer, searcher, cfg = infllm.load_infllm_qwen3(
        args.model_path, device=str(device), dtype=dtype, mem_config=mem_override)
    print(f"[InfLLM-RULER] mem_config={cfg}")

    L = int(model.config.num_hidden_layers)
    end_ids = _im_end_ids(tokenizer) if args.use_chat_template else []
    prefill_chunk = int(cfg["chunk_size"])

    outdir = Path(args.results_folder) / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""

    summary: dict = {}
    for task in tqdm(tasks, desc="tasks"):
        summary[task] = {}
        for length in tqdm(args.lengths, desc="lengths", leave=False):
            cell_started = time.time()
            if length not in ruler._LENGTH_TOKENS:
                print(f"[WARN] unknown length {length}, skipping")
                continue
            target_tokens = ruler._LENGTH_TOKENS[length]
            base_seed = args.seed + (hash((task, length)) % 100000)

            vt_icl = None
            if task == "variable_tracking":
                vt_icl = ruler._make_vt_icl(random.Random(base_seed + 777), 4)

            sample_indices = set(
                list(range(args.limit))[args.shard_index::args.num_shards])

            df = pd.DataFrame({"target": [], "output": [], "question": [],
                               "recall": []})
            recall_sum = 0.0
            total = 0
            n_tok_seen = 0
            oom_count = 0
            mnt = args.max_new_tokens if task != "variable_tracking" \
                else max(args.max_new_tokens, 60)

            for i in tqdm(range(args.limit), desc=f"{task}/{length}", leave=False):
                rng = random.Random(base_seed * 1000 + i)
                prompt, answers, gold_needle = ruler._build_sample(
                    task, target_tokens, tokenizer, rng, vt_icl)
                if i not in sample_indices:
                    continue

                bare_q = prompt[prompt.rfind("\n") + 1:].strip()

                model_prompt = prompt
                if args.use_chat_template:
                    messages = [{"role": "user", "content": prompt}]
                    try:
                        model_prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=args.enable_thinking)
                    except TypeError:
                        model_prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True)

                ids = tokenizer.encode(model_prompt, add_special_tokens=True,
                                       return_tensors="pt")
                if isinstance(ids, list):
                    ids = torch.tensor([ids], dtype=torch.long)
                input_ids = ids.to(device)
                n_tok_seen = int(input_ids.shape[1])

                try:
                    output = infllm.infllm_generate(
                        searcher, input_ids, max_new_tokens=mnt,
                        chunk_size=prefill_chunk, extra_end_token_ids=end_ids)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    output = "[OOM]"
                    oom_count += 1
                    print(f"[OOM] i={i} {task}/{length}: {e}", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                rec = ruler._string_match_all_one(output, answers)
                recall_sum += rec
                total += 1
                df.loc[len(df)] = [" | ".join(answers), output, bare_q, rec]
                if len(df) % 10 == 0:
                    qcb.harness._write_results_csv(
                        df, outdir / f"{task}_{length}{shard_tag}.csv")

            score = (recall_sum / total * 100.0) if total else 0.0
            summary[task][length] = {
                "score": round(score, 2), "n": total,
                "approx_tokens": n_tok_seen,
            }
            outfile = outdir / f"{task}_{length}{shard_tag}.csv"
            qcb.harness._write_results_csv(df, outfile)
            cfg_file = outdir / f"{task}_{length}{shard_tag}.json"
            json.dump(
                {
                    "status": "completed" if oom_count == 0 else "failed",
                    "task": task, "length": length,
                    "n_requested": args.limit,
                    "sharding": {"num_shards": args.num_shards,
                                 "shard_index": args.shard_index},
                    "summary": summary[task][length],
                    "score": summary[task][length]["score"],
                    "oom_count": oom_count,
                    "elapsed_seconds": round(time.time() - cell_started, 3),
                    "runtime": {
                        "node": socket.gethostname(),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "device": args.device, "seed": args.seed,
                        "dtype": args.dtype, "attn_implementation": "eager",
                    },
                    "chat_template": bool(args.use_chat_template),
                    "enable_thinking": bool(args.enable_thinking),
                    "scoring": "scripts.eval_ruler_mem_space._string_match_all_one",
                    "baseline": "infllm",
                    "infllm": {"mem_config": cfg, "num_layers": L},
                    "model": {"model_path": args.model_path,
                              "num_hidden_layers": L},
                },
                open(cfg_file, "w"), indent=2,
            )
            print(f"[InfLLM-RULER] {task}/{length}: recall={score:.2f} "
                  f"({total} samples, ~{n_tok_seen} tok) -> {outfile}")

    print("\n[InfLLM-RULER] SUMMARY")
    for task in summary:
        row = "  ".join(
            f"{ln}={summary[task][ln]['score']:.1f}" for ln in summary[task])
        print(f"  {task:>18}: {row}")
    with open(outdir / f"_summary{shard_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[InfLLM-RULER] Evaluation complete!")


if __name__ == "__main__":
    main()
