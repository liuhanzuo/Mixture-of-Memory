#!/usr/bin/env python
"""InfLLM baseline — BABILong (qa1/qa2/qa5) eval driver.

Head-to-head training-free peer of ``scripts/eval_qcmem_babilong.py``: SAME
BABILong data (``babilong`` ``get_formatted_input`` + ``DEFAULT_PROMPTS`` +
``DEFAULT_TEMPLATE``), SAME scoring (``babilong.metrics.compare_answers`` +
``TASK_LABELS``), SAME nested CSV layout (``<root>/<run>/{task}_{length}_{prompt}
{shard}.csv`` with columns ``target,output,question``) so
``scripts/score_nested_babilong.py`` scores it unchanged. ONLY the model forward
differs — InfLLM training-free memory attention (``scripts/infllm_qwen3.py``).

Example (full eval on node .73):
    python scripts/eval_infllm_babilong.py \
        --model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --tasks qa1 qa2 qa5 --lengths 8k 16k 32k \
        --limit 100 --use_chat_template \
        --output_name babilong_infllm --results_folder babilong_results/infllm \
        --num_shards 4 --shard_index 0
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
import scripts.infllm_qwen3 as infllm  # noqa: E402


def _im_end_ids(tokenizer):
    ids = []
    try:
        tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(tid, int) and tid >= 0:
            ids.append(tid)
    except Exception:
        pass
    return ids


def main():
    parser = argparse.ArgumentParser(description="InfLLM baseline — BABILong eval")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--results_folder", type=str, default="./babilong_results")
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="RMT-team/babilong")
    parser.add_argument("--tasks", type=str, nargs="+", default=["qa1", "qa2", "qa5"])
    parser.add_argument("--lengths", type=str, nargs="+",
                        default=["8k", "16k", "32k"])
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_instruction", action="store_true", default=True)
    parser.add_argument("--use_examples", action="store_true", default=True)
    parser.add_argument("--use_post_prompt", action="store_true", default=True)
    parser.add_argument("--use_chat_template", action="store_true", default=False)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--n_local", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--n_init", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        parser.error("bad shard config")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    # Data preflight (mirror eval_qcmem_babilong): fail before model load.
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

    mem_override = {}
    for k in ("n_local", "topk", "block_size", "n_init", "chunk_size"):
        v = getattr(args, k)
        if v is not None:
            mem_override[k] = v

    print(f"[InfLLM-BABILong] model_path={args.model_path}")
    model, tokenizer, searcher, cfg = infllm.load_infllm_qwen3(
        args.model_path, device=str(device), dtype=dtype, mem_config=mem_override)
    print(f"[InfLLM-BABILong] mem_config={cfg}")

    L = int(model.config.num_hidden_layers)
    end_ids = _im_end_ids(tokenizer) if args.use_chat_template else []
    prefill_chunk = int(cfg["chunk_size"])

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
        prompt_name = "_".join(
            [f"{k}_yes" if prompt_cfg[k] else f"{k}_no"
             for k in prompt_cfg if k != "template"])

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
                "baseline": "infllm",
                "infllm": {"mem_config": cfg, "num_layers": L},
                "model": {"model_path": args.model_path, "num_hidden_layers": L},
                "dataset_name": args.dataset_name,
                "available_count": available_count,
                "runtime": {
                    "node": socket.gethostname(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "device": args.device, "seed": args.seed,
                    "dtype": args.dtype, "attn_implementation": "eager",
                },
                "chat_template": bool(args.use_chat_template),
                "enable_thinking": bool(args.enable_thinking),
                "scoring": "babilong.metrics.TASK_LABELS+compare_answers",
            }
            json.dump(cell_config, open(cfg_file, "w"), indent=4)

            df = pd.DataFrame({"target": [], "output": [], "question": []})
            oom_count = 0

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
                    output = infllm.infllm_generate(
                        searcher, input_ids, max_new_tokens=args.max_new_tokens,
                        chunk_size=prefill_chunk, extra_end_token_ids=end_ids)
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
                "elapsed_seconds": round(time.time() - cell_started, 3),
            })
            json.dump(cell_config, open(cfg_file, "w"), indent=4)
            print(f"[InfLLM-BABILong] {task}/{split_name}: "
                  f"score={cell_config['score']} ({len(df)}) -> {outfile}")

    print("\n[InfLLM-BABILong] Evaluation complete!")


if __name__ == "__main__":
    main()
