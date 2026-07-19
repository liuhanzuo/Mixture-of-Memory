#!/usr/bin/env python
"""StreamingLLM (equal-budget, truncation-approx) baseline — BABILong eval driver.

Recency-budget peer of ``scripts/eval_qcmem_babilong.py`` and
``scripts/eval_infllm_babilong.py``: SAME BABILong data (``babilong``
``get_formatted_input`` + ``DEFAULT_PROMPTS`` + ``DEFAULT_TEMPLATE`` via the
shared ``scripts/eval_qcmem_babilong._load_babilong_task`` loader), SAME scoring
(official ``babilong.metrics.compare_answers`` + ``TASK_LABELS`` — NO ``re.search``),
SAME flat sharded CSV layout (``<root>/<run>/{task}_{length}_{prompt}{shard}.csv``
with columns ``target,output,question`` written by the harness ``_write_results_csv``,
QUOTE_ALL) so StreamingLLM, InfLLM and QCMem are graded identically. ONLY the
model forward differs — StreamingLLM's equal-budget sink+window truncation
(``scripts/streamingllm_backbone.py``), the SAME mechanism proven in
``scripts/eval_ruler_streamingllm.py``.

Equal budget = sink 4 + window 6653 = 6657 tokens ~= CoMem constant read. The
BABILong prompt is ``instruction + examples + <context> + question + post_prompt``
(+ chat generation prefix at the tail when --use_chat_template). StreamingLLM
keeps the first ``sink_size`` tokens (attention sink over the leading instruction)
+ the last ``window_size`` tokens (the trailing question / post_prompt / assistant
prefix) and DROPS the middle context, so a supporting fact buried in the dropped
middle is structurally unrecoverable — exactly the recency-budget failure mode
this row exposes vs CoMem's relevance-based retrieval.

Example (full eval on node .73, qa1/qa2/qa5 x 0k-32k, n=100, 8-shard):
    python scripts/eval_streamingllm_babilong.py \
        --model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --tasks qa1 qa2 qa5 --lengths 0k 1k 2k 4k 8k 16k 32k \
        --limit 100 --use_chat_template --max_new_tokens 20 \
        --sink_size 4 --window_size 6653 \
        --results_folder babilong_results --output_name streamingllm_n100 \
        --num_shards 8 --shard_index 0
    # Score only (merge all shards per cell, official compare_answers):
    python scripts/eval_streamingllm_babilong.py --score_only \
        --tasks qa1 qa2 qa5 --lengths 0k 1k 2k 4k 8k 16k 32k \
        --results_folder babilong_results --output_name streamingllm_n100 \
        --num_shards 8
"""
from __future__ import annotations

import argparse
import json
import os
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

from babilong.metrics import TASK_LABELS, compare_answers  # noqa: E402
from babilong.prompts import (DEFAULT_PROMPTS, DEFAULT_TEMPLATE,  # noqa: E402
                              get_formatted_input)

import scripts.eval_qcmem_babilong as qcb  # noqa: E402  (CSV writer + data loader)
import scripts.streamingllm_backbone as slm  # noqa: E402


def _prompt_name(use_instruction, use_examples, use_post_prompt, use_chat_template):
    """Reconstruct the on-disk prompt tag EXACTLY as eval_qcmem_babilong /
    eval_infllm_babilong build it (order: instruction, examples, post_prompt,
    chat_template, system_prompt; system_prompt is always empty -> _no)."""
    prompt_cfg = {
        "instruction": use_instruction,
        "examples": use_examples,
        "post_prompt": use_post_prompt,
        "chat_template": use_chat_template,
        "system_prompt": "",
    }
    return "_".join(f"{k}_yes" if prompt_cfg[k] else f"{k}_no" for k in prompt_cfg)


def _score_only(args, prompt_name):
    """Merge the (disjoint, strided) per-shard CSVs of every (task, length) cell
    and recompute the official BABILong accuracy with
    ``babilong.metrics.compare_answers`` + ``TASK_LABELS[task]`` (NO re.search).

    Shards write ``[shard_index::num_shards]`` disjoint sample subsets, so simply
    concatenating all shard CSVs for a cell reconstructs the full n=limit set
    (no dedup needed). Writes ``scores.json`` and prints the qa x length grid.
    """
    outdir = Path(args.results_folder) / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    print(f"[SLM-BABILong][score] dir={outdir}")
    for task in args.tasks:
        results[task] = {}
        for length in args.lengths:
            pattern = f"{task}_{length}_{prompt_name}"
            # match both sharded (..._shardXofN.csv) and single-shard (....csv).
            csvs = sorted(outdir.glob(f"{pattern}_shard*of*.csv"))
            if not csvs:
                single = outdir / f"{pattern}.csv"
                csvs = [single] if single.exists() else []
            if not csvs:
                print(f"[SLM-BABILong][score] MISSING {task}/{length} "
                      f"(no CSV matching {pattern})")
                results[task][length] = {"n": 0, "correct": 0, "score": None}
                continue
            frames = []
            for c in csvs:
                try:
                    frames.append(pd.read_csv(c, dtype=str, keep_default_na=False))
                except Exception as e:
                    print(f"[SLM-BABILong][score] failed reading {c}: {e}")
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            n = len(df)
            labels = TASK_LABELS[task]
            correct = sum(
                bool(compare_answers(row.target, row.output, row.question, labels))
                for row in df.itertuples(index=False))
            empty = int((df["output"].str.strip() == "").sum()) if n else 0
            score = round(100.0 * correct / n, 2) if n else None
            results[task][length] = {"n": n, "correct": correct, "score": score,
                                     "empty_output": empty,
                                     "n_shards": len(csvs)}
    # print grid + persist
    print("\n[SLM-BABILong] score grid (official compare_answers):")
    header = "task  " + "  ".join(f"{L:>6}" for L in args.lengths)
    print(header)
    for task in args.tasks:
        cells = []
        for L in args.lengths:
            s = results[task][L]["score"]
            cells.append(f"{s:6.1f}" if s is not None else "  n/a ")
        print(f"{task:5s} " + "  ".join(cells))
    with open(outdir / "scores.json", "w") as f:
        json.dump({"benchmark": "babilong", "scoring":
                   "babilong.metrics.TASK_LABELS+compare_answers",
                   "results": results}, f, indent=2)
    print(f"\n[SLM-BABILong][score] saved {outdir / 'scores.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="StreamingLLM (equal-budget) baseline — BABILong eval")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--results_folder", type=str, default="./babilong_results")
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="RMT-team/babilong")
    parser.add_argument("--tasks", type=str, nargs="+", default=["qa1", "qa2", "qa5"])
    parser.add_argument("--lengths", type=str, nargs="+",
                        default=["0k", "1k", "2k", "4k", "8k", "16k", "32k"])
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_instruction", action="store_true", default=True)
    parser.add_argument("--use_examples", action="store_true", default=True)
    parser.add_argument("--use_post_prompt", action="store_true", default=True)
    parser.add_argument("--use_chat_template", action="store_true", default=False)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--score_only", action="store_true",
                        help="Merge existing per-shard CSVs + recompute official "
                             "compare_answers accuracy per (task,length). No model.")
    # StreamingLLM equal-budget knobs (defaults = ruler_streamingllm budget).
    parser.add_argument("--sink_size", type=int, default=slm.DEFAULT_SINK,
                        help="Attention-sink KV kept (StreamingLLM default 4).")
    parser.add_argument("--window_size", type=int, default=slm.DEFAULT_WINDOW,
                        help="Recent-window KV kept. sink+window is the fixed KV "
                             "budget; default 4+6653=6657 ~ CoMem read length.")
    args = parser.parse_args()

    prompt_name = _prompt_name(bool(args.use_instruction), bool(args.use_examples),
                               bool(args.use_post_prompt),
                               bool(args.use_chat_template))

    # --- score-only: merge shards + official compare_answers scoring, exit ------
    if args.score_only:
        _score_only(args, prompt_name)
        return

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        parser.error("bad shard config")
    if not args.model_path:
        parser.error("--model_path is required unless --score_only")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)

    # Data preflight (mirror eval_qcmem/eval_infllm): fail before model load, and
    # never silently downgrade n=100 to a partial mirror count.
    preloaded = {}
    for task in args.tasks:
        for split_name in args.lengths:
            try:
                task_data = qcb._load_babilong_task(
                    args.dataset_name, split_name, task)
            except Exception as e:
                parser.error(f"data preflight failed {task}/{split_name}: {e}")
            available = len(task_data)
            required = args.limit if args.limit > 0 else available
            if available < required:
                parser.error(f"data preflight {task}/{split_name}: "
                             f"available={available} < requested={required}")
            preloaded[(task, split_name)] = task_data

    budget = args.sink_size + args.window_size
    print(f"[SLM-BABILong] model_path={model_path}")
    print(f"[SLM-BABILong] TRUNCATION APPROX: sink_size={args.sink_size} "
          f"window_size={args.window_size} budget={budget} dtype={dtype} "
          f"attn_impl={args.attn_impl}")
    print(f"[SLM-BABILong] tasks={args.tasks} lengths={args.lengths} "
          f"limit={args.limit} chat={args.use_chat_template} "
          f"think={args.enable_thinking} shard={args.shard_index}/{args.num_shards}")

    model, tokenizer = slm.load_backbone(model_path, device, dtype, args.attn_impl)
    L = int(model.config.num_hidden_layers)
    end_ids = slm.im_end_ids(tokenizer) if args.use_chat_template else []

    for task in tqdm(args.tasks, desc="tasks"):
        if task not in DEFAULT_PROMPTS:
            print(f"[WARNING] Task {task} not in DEFAULT_PROMPTS, skipping.")
            continue
        prompt_cfg = {
            "instruction": DEFAULT_PROMPTS[task]["instruction"] if args.use_instruction else "",
            "examples":    DEFAULT_PROMPTS[task]["examples"]    if args.use_examples    else "",
            "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"] if args.use_post_prompt else "",
            "template":    DEFAULT_TEMPLATE,
            "chat_template": args.use_chat_template,
            "system_prompt": "",
        }

        for split_name in tqdm(args.lengths, desc="lengths", leave=False):
            cell_started = time.time()
            task_data = preloaded[(task, split_name)]
            available_count = len(task_data)

            outdir = Path(args.results_folder) / args.output_name
            outdir.mkdir(parents=True, exist_ok=True)
            sharded = args.num_shards > 1
            shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""
            outfile = outdir / f"{task}_{split_name}_{prompt_name}{shard_tag}.csv"
            cfg_file = outdir / f"{task}_{split_name}_{prompt_name}{shard_tag}.json"

            cell_config = {
                "status": "running", "task": task, "length": split_name,
                "n_requested": args.limit,
                "sharding": {"num_shards": args.num_shards,
                             "shard_index": args.shard_index},
                "prompt": prompt_cfg,
                "generate_kwargs": {"max_new_tokens": args.max_new_tokens,
                                    "do_sample": False, "num_beams": 1},
                "baseline": "streamingllm",
                "streamingllm": {"method": "truncation_approx",
                                 "sink_size": args.sink_size,
                                 "window_size": args.window_size,
                                 "budget": budget, "num_layers": L},
                "model": {"model_path": model_path, "num_hidden_layers": L},
                "dataset_name": args.dataset_name,
                "available_count": available_count,
                "runtime": {
                    "node": socket.gethostname(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "device": args.device, "seed": args.seed,
                    "dtype": args.dtype, "attn_implementation": args.attn_impl,
                },
                "chat_template": bool(args.use_chat_template),
                "enable_thinking": bool(args.enable_thinking),
                "scoring": "babilong.metrics.TASK_LABELS+compare_answers",
            }
            json.dump(cell_config, open(cfg_file, "w"), indent=4)

            df = pd.DataFrame({"target": [], "output": [], "question": []})
            oom_count = 0
            kept_last = 0

            num_samples = len(task_data)
            if args.limit > 0:
                num_samples = min(num_samples, args.limit)
            sample_indices = list(range(num_samples))[args.shard_index::args.num_shards]

            for idx in tqdm(sample_indices, desc=f"{task}/{split_name}", leave=False):
                sample = task_data[idx]
                target = sample["target"]
                question = sample["question"]
                input_text = get_formatted_input(
                    sample["input"], sample["question"],
                    prompt_cfg["examples"], prompt_cfg["instruction"],
                    prompt_cfg["post_prompt"], template=prompt_cfg["template"])
                if args.use_chat_template:
                    messages = [{"role": "user", "content": input_text}]
                    # add_generation_prompt=True: StreamingLLM feeds a normal
                    # full-attention model, so the assistant generation prefix
                    # (no-think block) sits at the tail and is kept by the recent
                    # window — identical protocol to eval_infllm_babilong.
                    try:
                        input_text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=args.enable_thinking)
                    except TypeError:
                        input_text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True)

                ids = tokenizer.encode(input_text, add_special_tokens=True,
                                       return_tensors="pt")
                if isinstance(ids, list):
                    ids = torch.tensor([ids], dtype=torch.long)
                input_ids = ids.to(device)

                try:
                    output, kept_last, _orig = slm.streaming_generate(
                        model, tokenizer, input_ids,
                        sink_size=args.sink_size, window_size=args.window_size,
                        max_new_tokens=args.max_new_tokens, device=device,
                        extra_end_token_ids=end_ids)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    output = "[OOM]"
                    oom_count += 1
                    print(f"[OOM] idx={idx} {task}/{split_name}: {e}", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                df.loc[len(df)] = [target, output, question]
                if len(df) % 10 == 0 or idx == sample_indices[-1]:
                    qcb.harness._write_results_csv(df, outfile)

            qcb.harness._write_results_csv(df, outfile)
            correct = sum(
                bool(compare_answers(row.target, row.output, row.question,
                                     TASK_LABELS[task]))
                for row in df.itertuples(index=False))
            cell_config.update({
                "status": "completed" if oom_count == 0 else "failed",
                "n": len(df), "oom_count": oom_count, "correct": correct,
                "score": round(100.0 * correct / len(df), 2) if len(df) else 0.0,
                "kept_tokens": kept_last,
                "elapsed_seconds": round(time.time() - cell_started, 3),
            })
            json.dump(cell_config, open(cfg_file, "w"), indent=4)
            print(f"[SLM-BABILong] {task}/{split_name}: "
                  f"score={cell_config['score']} ({len(df)}, kept~{kept_last}) "
                  f"-> {outfile}")

    print("\n[SLM-BABILong] Evaluation complete!")


if __name__ == "__main__":
    main()
