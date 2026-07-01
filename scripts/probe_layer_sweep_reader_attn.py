#!/usr/bin/env python3
"""Layer-sweep reader-attn recall probe (diagnostic, 2026-06-29).

Sweeps --select_layer over a set of layers, running S1 (reader-attn) and S2
(BM25) recall on the same qa5 16k samples for each layer, to see if some layer
has significantly better needle-finding salience than the default L16.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/probe_layer_sweep_reader_attn.py \
    --model_path models/Meta-Llama-3-8B \
    --checkpoint outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt \
    --adapter_config outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json \
    --layers 8 12 16 20 24 \
    --tasks qa5 --lengths 16k --limit 20 --device cuda:0

Reads babilong locally (offline). Writes per-layer CSV to ./layer_sweep_results/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_BABILONG_PKG = os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")
if os.path.isdir(_BABILONG_PKG) and _BABILONG_PKG not in sys.path:
    sys.path.insert(0, _BABILONG_PKG)

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
from src.memory.mem_space import MemorySpaceConfig, _reset_fifo_memory
from scripts.run_babilong_mem_space import (
    build_mem_space_config,
    load_babilong_dataset,
    load_mem_space_model,
    _reset_banks,
    _reset_l2,
    _locate_needle_chunks,
    _set_fifo_keep_all_buffer,
)
from scripts.e2_multiscorer_probe import (
    _build_prompt_cfg,
    _encode_sample,
    _stream_context,
    _freeze,
    _unfreeze,
    _pool_from_buffer,
    score_S1_reader_attn,
    score_S2_bm25,
    _rank_recall,
    RECALL_KS,
)


def probe_layer(model, input_ids, target, question, tokenizer, chunk_size, device, layer):
    """Score one sample with S1 (at given layer) and S2."""
    _reset_banks(model)
    _reset_l2(model)
    _reset_fifo_memory(model)

    tokens = input_ids[0]
    chunks = list(tokens.split(chunk_size))
    n_chunks = len(chunks)
    ingested = n_chunks - 1

    needle = _locate_needle_chunks(input_ids, target, tokenizer, chunk_size)

    _stream_context(model, chunks, device)
    _freeze(model)
    offset, C = _pool_from_buffer(model, n_chunks, layer)

    result = {"n_chunks": n_chunks, "n_candidates": C, "needle_in_buffer": False,
              "needle_evicted": False}
    for k in RECALL_KS:
        result[f"S1_recall@{k}"] = ""
        result[f"S2_recall@{k}"] = ""
    result["S1_rank"] = ""
    result["S2_rank"] = ""

    if C <= 0 or offset is None:
        _unfreeze(model)
        return result

    # needle status
    needle_local = None
    if needle is not None:
        streamable = sorted(c for c in needle if 0 <= c < ingested)
        if streamable:
            nl = [c - offset for c in streamable if 0 <= (c - offset) < C]
            if nl:
                result["needle_in_buffer"] = True
                needle_local = nl
            else:
                result["needle_evicted"] = True

    last_chunk = chunks[-1]
    sal_s1 = score_S1_reader_attn(model, last_chunk, device, layer)
    q_ids = tokenizer.encode((question or "").strip(), add_special_tokens=False)
    sal_s2 = score_S2_bm25(chunks, offset, C, q_ids)

    _unfreeze(model)

    if needle_local is not None:
        if sal_s1 is not None:
            rank, recalls = _rank_recall(sal_s1, needle_local)
            result["S1_rank"] = rank
            for k in RECALL_KS:
                result[f"S1_recall@{k}"] = recalls[k]
        if sal_s2 is not None:
            rank2, recalls2 = _rank_recall(sal_s2, needle_local)
            result["S2_rank"] = rank2
            for k in RECALL_KS:
                result[f"S2_recall@{k}"] = recalls2[k]
    return result


def main():
    ap = argparse.ArgumentParser(description="Layer-sweep reader-attn recall probe")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adapter_config", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=[8, 12, 16, 20, 24])
    ap.add_argument("--tasks", nargs="+", default=["qa5"])
    ap.add_argument("--lengths", nargs="+", default=["16k"])
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn_impl", default="sdpa")
    ap.add_argument("--dataset_name", default="RMT-team/babilong")
    ap.add_argument("--output_dir", default="./layer_sweep_results")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    with open(args.adapter_config) as f:
        adapter_cfg = json.load(f)
    mem_config: MemorySpaceConfig = build_mem_space_config(adapter_cfg)
    mem_config.l3_recon_max_positions = args.chunk_size

    print(f"[LayerSweep] Loading model from {args.checkpoint}")
    model = load_mem_space_model(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        mem_config=mem_config,
        device=device,
        dtype=dtype,
        attn_impl=args.attn_impl,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-layer results: layer -> {S1_recall@k, S2_recall@k}
    summary = {}  # layer -> {metric: [values]}

    for task in args.tasks:
        if task not in DEFAULT_PROMPTS:
            print(f"Skipping unknown task {task}")
            continue
        prompt_cfg = _build_prompt_cfg(task)
        for length in args.lengths:
            try:
                data = load_babilong_dataset(args.dataset_name, length)
                task_data = data[task]
            except Exception as e:
                print(f"ERROR loading {task}/{length}: {e}")
                continue
            n = min(len(task_data), args.limit) if args.limit > 0 else len(task_data)
            print(f"\n[LayerSweep] task={task} length={length}: n={n}")

            # Per-sample data: encode once, reuse for all layers
            samples = []
            for idx in range(n):
                try:
                    target, question, input_ids = _encode_sample(
                        task_data[idx], prompt_cfg, tokenizer
                    )
                    samples.append((idx, target, question, input_ids.to(device)))
                except Exception as e:
                    print(f"  sample {idx} encode error: {e}")

            for layer in args.layers:
                print(f"  Layer {layer} ...")
                rows = []
                for idx, target, question, input_ids in tqdm(samples, desc=f"L{layer}", leave=False):
                    with torch.amp.autocast(device_type="cuda", dtype=dtype):
                        r = probe_layer(model, input_ids, target, question, tokenizer,
                                        args.chunk_size, device, layer)
                    r["idx"] = idx
                    r["task"] = task
                    r["length"] = length
                    r["layer"] = layer
                    rows.append(r)

                # Save CSV
                csv_path = out_dir / f"layer{layer}_{task}_{length}.csv"
                fieldnames = ["idx", "task", "length", "layer", "n_chunks", "n_candidates",
                              "needle_in_buffer", "needle_evicted", "S1_rank", "S2_rank"] + \
                             [f"S1_recall@{k}" for k in RECALL_KS] + \
                             [f"S2_recall@{k}" for k in RECALL_KS]
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)

                # Summarize
                vals = defaultdict(list)
                for r in rows:
                    for k in RECALL_KS:
                        try: vals[f"S1_recall@{k}"].append(float(r[f"S1_recall@{k}"]))
                        except: pass
                        try: vals[f"S2_recall@{k}"].append(float(r[f"S2_recall@{k}"]))
                        except: pass
                key = f"{task}/{length}/L{layer}"
                summary[key] = {m: sum(v)/len(v) for m, v in vals.items() if v}
                r4_s1 = summary[key].get("S1_recall@4", float("nan"))
                r4_s2 = summary[key].get("S2_recall@4", float("nan"))
                print(f"    L{layer}: S1_recall@4={r4_s1:.3f}  S2_recall@4={r4_s2:.3f}  n={len(samples)}")

    # Print summary table
    print("\n=== LAYER SWEEP SUMMARY ===")
    print(f"{'Key':<25} {'S1_r@1':>7} {'S1_r@4':>7} {'S1_r@8':>7} {'S2_r@4':>7}")
    for key, vals in sorted(summary.items()):
        s1r1 = vals.get("S1_recall@1", float("nan"))
        s1r4 = vals.get("S1_recall@4", float("nan"))
        s1r8 = vals.get("S1_recall@8", float("nan"))
        s2r4 = vals.get("S2_recall@4", float("nan"))
        print(f"{key:<25} {s1r1:>7.3f} {s1r4:>7.3f} {s1r8:>7.3f} {s2r4:>7.3f}")


if __name__ == "__main__":
    main()
