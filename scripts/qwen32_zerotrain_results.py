#!/usr/bin/env python3
"""Validate and aggregate the fixed Qwen3-32B zero-training QCMem protocol."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RULER = {
    "niah_single_2": ["8k", "16k", "32k", "64k", "128k"],
    "niah_multikey_1": ["8k", "16k", "32k", "64k", "128k"],
    "variable_tracking": ["8k", "16k", "32k"],
}
BABI = {task: ["0k", "1k", "2k", "4k", "8k", "16k", "32k"]
        for task in ("qa1", "qa2", "qa5")}


def expected(bench):
    return RULER if bench == "ruler" else BABI


def find_cell(root: Path, task: str, length: str):
    matches = []
    for p in root.rglob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            cfg = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if cfg.get("task") == task and cfg.get("length") == length:
            matches.append((p, cfg))
    return matches


def validate(root: Path, bench: str, task: str, length: str, shard_index=None):
    found = find_cell(root, task, length)
    if shard_index is not None:
        found = [(p, c) for p, c in found
                 if c.get("sharding", {}).get("shard_index") == shard_index]
    if len(found) not in (1, 4):
        return None, [f"expected one config or four shards, found {len(found)}"]
    errors = []
    aggregate_n = aggregate_correct = 0
    aggregate_recall = elapsed = 0.0
    nodes, gpus = [], []
    shard_indices = set()
    first_cfg = None
    for cfg_path, cfg in found:
      first_cfg = first_cfg or cfg
      q = cfg.get("qcmem", {})
      model = cfg.get("model", {})
      runtime = cfg.get("runtime", {})
      sharding = cfg.get("sharding", {"num_shards": 1, "shard_index": 0})
      nshard, sidx = sharding.get("num_shards"), sharding.get("shard_index")
      expected_rows = len(list(range(500))[sidx::nshard]) \
          if isinstance(nshard, int) and isinstance(sidx, int) and nshard > 0 else -1
      shard_indices.add(sidx)
      n = cfg.get("n", cfg.get("summary", {}).get("n"))
      score = cfg.get("score", cfg.get("summary", {}).get("score"))
      checks = {
        "status": cfg.get("status") == "completed",
        "n": n == expected_rows,
        "n_requested": cfg.get("n_requested") == 500,
        "resume_j": q.get("resume_j") == 16,
        "chunk_size": q.get("chunk_size", model.get("chunk_size")) == 512,
        "sink": q.get("sink_tokens") == "bos",
        "lora_adapter": q.get("lora_adapter") is None,
        "bottleneck_ckpt": model.get("bottleneck_ckpt") is None,
        "layers": model.get("num_hidden_layers", q.get("num_layers")) == 64,
        "no_chat_template": cfg.get("chat_template") is False,
        "zero_training_no_adapter": cfg.get("zero_training_no_adapter") is True,
        "dtype": runtime.get("dtype") == "bfloat16",
        "attn": runtime.get("attn_implementation") == "sdpa",
        "pythonhashseed": runtime.get("pythonhashseed") == "0",
        "elapsed": isinstance(cfg.get("elapsed_seconds"), (int, float)),
        "no_oom": cfg.get("oom_count") == 0,
        "score": isinstance(cfg.get("score", cfg.get("summary", {}).get("score")),
                            (int, float)),
      }
      if bench == "ruler" and task == "variable_tracking":
        checks.update({
            "selector": q.get("selector") == "iter_bm25",
            "topk": q.get("topk") == 16,
            "iter_rounds": (q.get("iter") or {}).get("rounds") == 4,
            "iter_hop_topk": (q.get("iter") or {}).get("hop_topk") == 4,
        })
      else:
        checks.update({"selector": q.get("selector") == "bm25",
                       "topk": q.get("topk") == 12})
      if bench == "babilong":
        checks["official_scoring"] = (
            cfg.get("scoring") == "babilong.metrics.TASK_LABELS+compare_answers")
        checks["available_count"] = cfg.get("available_count", 0) >= 500
      else:
        checks["official_scoring"] = (
            cfg.get("scoring") ==
            "scripts.eval_ruler_mem_space._string_match_all_one")
      errors.extend(f"shard{sidx}:{k}" for k, ok in checks.items() if not ok)
      csv_path = cfg_path.with_suffix(".csv")
      if not csv_path.is_file():
        errors.append(f"shard{sidx}:missing csv")
      else:
        with csv_path.open(newline="") as f:
            csv_rows = list(csv.DictReader(f))
        rows = len(csv_rows)
        if rows != expected_rows:
            errors.append(f"shard{sidx}:csv_rows={rows}")
        if bench == "ruler":
            try:
                aggregate_recall += sum(float(row["recall"]) for row in csv_rows)
            except (KeyError, TypeError, ValueError):
                errors.append(f"shard{sidx}:invalid recall column")
      if isinstance(n, int) and isinstance(score, (int, float)):
        aggregate_n += n
      aggregate_correct += int(cfg.get("correct", 0))
      elapsed += float(cfg.get("elapsed_seconds", 0))
      nodes.append(runtime.get("node")); gpus.append(runtime.get("cuda_visible_devices"))
    expected_shards = ({shard_index} if shard_index is not None
                       else set(range(len(found))))
    if shard_indices != expected_shards:
        errors.append(f"shard_indices={sorted(shard_indices, key=str)}")
    merged = dict(first_cfg or {})
    merged["n"] = aggregate_n
    merged["score"] = (round(100.0 * aggregate_correct / aggregate_n, 4)
                       if bench == "babilong" and aggregate_n else
                       round(100.0 * aggregate_recall / aggregate_n, 4)
                       if aggregate_n else None)
    merged["elapsed_seconds"] = elapsed
    merged["runtime"] = {"node": nodes, "cuda_visible_devices": gpus}
    return merged, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler-root", type=Path)
    ap.add_argument("--babilong-root", type=Path)
    ap.add_argument("--is-complete", nargs="+")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.is_complete:
        if len(args.is_complete) not in (4, 5):
            ap.error("--is-complete BENCH ROOT TASK LENGTH [SHARD_INDEX]")
        bench, root, task, length = args.is_complete[:4]
        shard = int(args.is_complete[4]) if len(args.is_complete) == 5 else None
        cfg, errors = validate(Path(root), bench, task, length, shard)
        if errors:
            print("; ".join(errors))
            raise SystemExit(1)
        print(cfg.get("score", cfg.get("summary", {}).get("score")))
        return

    roots = {"ruler": args.ruler_root, "babilong": args.babilong_root}
    if not all(roots.values()):
        ap.error("aggregation requires --ruler-root and --babilong-root")
    report = {"protocol": "Qwen3-32B QCMem zero-training/no-adapter",
              "cells": [], "missing_or_invalid": []}
    by_task = {}
    for bench, root in roots.items():
        for task, lengths in expected(bench).items():
            for length in lengths:
                cfg, errors = validate(root, bench, task, length)
                if errors:
                    report["missing_or_invalid"].append(
                        {"benchmark": bench, "task": task, "length": length,
                         "errors": errors})
                    continue
                score = cfg.get("score", cfg.get("summary", {}).get("score"))
                row = {"benchmark": bench, "task": task, "length": length,
                       "score": score, "elapsed_seconds": cfg["elapsed_seconds"],
                       "node": cfg["runtime"]["node"],
                       "gpu": cfg["runtime"]["cuda_visible_devices"]}
                report["cells"].append(row)
                by_task.setdefault(f"{bench}/{task}", []).append(score)
    report["completed_cells"] = len(report["cells"])
    report["expected_cells"] = 34
    report["macro"] = (round(sum(x["score"] for x in report["cells"])
                             / len(report["cells"]), 4)
                       if report["cells"] else None)
    report["task_macro"] = {k: round(sum(v) / len(v), 4)
                            for k, v in by_task.items()}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if report["completed_cells"] == 34 else 2)


if __name__ == "__main__":
    main()
