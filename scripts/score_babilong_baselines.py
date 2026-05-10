"""Score BABILong CSV outputs and emit aggregated JSON.

Usage:
    python score_babilong_baselines.py \\
        --results_folder /path/to/babilong_results \\
        --baselines MemoryLLM-8B-chat Beacon-Qwen2-7B Llama-3.2-1B-Instruct \\
        --out /path/to/babilong_results.json

Each baseline is a sub-folder of results_folder containing
<task>_<length>_<suffix>.csv files. We compute per-task accuracy with
babilong/babilong/metrics.py, then AVG across tasks per length.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# babilong package on PYTHONPATH
for cand in (
    "/apdcephfs_zwfy6/share_304376610/pighzliu_code/babilong-pkg",
    "/apdcephfs_wzc1/share_303098609/pighzliu_code/babilong",
):
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)

from babilong.metrics import TASK_LABELS, compare_answers


def score_baseline(results_folder: Path, baseline: str, tasks: list[str], lengths: list[str], suffix_parts: list[str] | None = None) -> dict:
    bdir = results_folder / baseline
    if not bdir.is_dir():
        return {"_error": f"directory not found: {bdir}"}

    out: dict = {}
    suffix_glob = "*" if suffix_parts is None else "_".join(suffix_parts)
    for length in lengths:
        per_task_acc = {}
        nsamp_set = set()
        for task in tasks:
            # find a CSV that matches
            cands = list(bdir.glob(f"{task}_{length}_*.csv"))
            if not cands:
                per_task_acc[task] = None
                continue
            csv = sorted(cands)[0]
            try:
                df = pd.read_csv(csv)
            except Exception as e:
                per_task_acc[task] = None
                continue
            if "target" not in df.columns or "output" not in df.columns:
                per_task_acc[task] = None
                continue
            # filter ERROR rows
            df = df.dropna(subset=["target"])
            if len(df) == 0:
                per_task_acc[task] = None
                continue
            labels = TASK_LABELS.get(task, [])
            correct = 0
            for _, row in df.iterrows():
                target = str(row["target"])
                output = str(row.get("output", ""))
                question = str(row.get("question", ""))
                if output.startswith("<<ERROR"):
                    continue
                try:
                    if compare_answers(target, output, question, labels):
                        correct += 1
                except Exception:
                    pass
            acc = correct / len(df)
            per_task_acc[task] = round(acc, 4)
            nsamp_set.add(len(df))

        # AVG across present tasks
        valid = [v for v in per_task_acc.values() if isinstance(v, (int, float))]
        if valid:
            per_task_acc["AVG"] = round(sum(valid) / len(valid), 4)
        else:
            per_task_acc["AVG"] = None
        if nsamp_set:
            per_task_acc["num_samples"] = max(nsamp_set)
        out[length] = per_task_acc
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_folder", type=str, required=True)
    p.add_argument("--baselines", nargs="+", required=True)
    p.add_argument(
        "--tasks", nargs="+",
        default=["qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10"],
    )
    p.add_argument(
        "--lengths", nargs="+",
        default=["1k", "2k", "4k", "8k", "16k", "32k"],
    )
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    rf = Path(args.results_folder)
    aggregated = {}
    for b in args.baselines:
        aggregated[b] = score_baseline(rf, b, args.tasks, args.lengths)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"[score] wrote {args.out}")
    print(json.dumps(aggregated, indent=2))


if __name__ == "__main__":
    main()
