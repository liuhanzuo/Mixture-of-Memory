"""Aggregate BABILong-100 baseline results into babilong_results.json.

Reads CSV files from results_folder, computes per-task accuracy via babilong.metrics,
and emits a structured JSON: {model: {length: {task: acc, AVG: avg}}}.

Designed to:
- Work with both CSV formats (with/without pandas row index)
- Skip tasks/lengths with < threshold rows (default 100)
- Re-use the canonical output schema in status/babilong_results.json

Usage:
    python scripts/aggregate_babilong_baselines.py \
        --results_folder <path> \
        --output_json <path> \
        --models Llama-3.2-1B-Instruct Beacon-Qwen2-7B MemoryLLM-8B-chat \
        --lengths 1k 2k 4k 8k 16k 32k \
        --tasks qa1 qa2 qa3 qa4 qa5 qa6 qa7 qa8 qa9 qa10 \
        --min_rows 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd

# Add babilong package to path (for metrics)
for p in (
    "/apdcephfs_wzc1/share_303098609/pighzliu_code/babilong",
    "/apdcephfs_zwfy6/share_304376610/pighzliu_code/babilong",
):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
from babilong.metrics import TASK_LABELS, compare_answers  # noqa: E402


def find_csv(base: Path, task: str, length: str) -> Path | None:
    suffixes = [
        f"{task}_{length}_instruction_yes_examples_yes_post_prompt_yes_chat_template_yes_system_prompt_no.csv",
        f"{task}_{length}_instruction_yes_examples_yes_post_prompt_yes_chat_template_no_system_prompt_no.csv",
    ]
    for s in suffixes:
        p = base / s
        if p.exists():
            return p
    return None


def score_csv(csv: Path, task: str, min_rows: int) -> dict | None:
    try:
        df = pd.read_csv(csv)
    except Exception as e:
        print(f"[score] {csv} unreadable: {e}", file=sys.stderr)
        return None
    if df.empty:
        return None
    # Drop unnamed index column if present
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    if not all(c in df.columns for c in ("target", "output", "question")):
        return None
    labels = TASK_LABELS[task]
    correct = 0
    total = 0
    errors = 0
    for _, row in df.iterrows():
        tgt = row["target"]
        q = row["question"]
        out = row.get("output")
        if not isinstance(tgt, str) or not isinstance(q, str):
            continue
        if not isinstance(out, str):
            out = ""
        if out.startswith("<<ERROR"):
            errors += 1
        total += 1
        try:
            if compare_answers(tgt, out, q, labels):
                correct += 1
        except Exception:
            pass
    return {
        "correct": correct,
        "total": total,
        "errors": errors,
        "acc": correct / total if total else 0.0,
        "complete": total >= min_rows,
    }


def aggregate(args) -> dict:
    base_folder = Path(args.results_folder)
    out = {
        "evaluator": "babilong-100 baseline aggregation (eval_baseline_babilong.py + run_model_on_babilong.py)",
        "ts": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "tasks": list(args.tasks),
        "lengths": list(args.lengths),
        "min_rows_for_AVG": args.min_rows,
        "models": {},
    }

    for model_name in args.models:
        model_dir = base_folder / model_name
        if not model_dir.exists():
            print(f"[aggregate] missing dir: {model_dir}", file=sys.stderr)
            continue
        per_length: dict[str, dict] = {}
        for length in args.lengths:
            entry: dict = {}
            accs: list[float] = []
            n_complete = 0
            for task in args.tasks:
                csv = find_csv(model_dir, task, length)
                if csv is None:
                    entry[task] = None
                    continue
                rec = score_csv(csv, task, args.min_rows)
                if rec is None:
                    entry[task] = None
                    continue
                entry[task] = round(rec["acc"], 4)
                if rec["complete"]:
                    accs.append(rec["acc"])
                    n_complete += 1
                else:
                    entry[f"{task}__partial"] = rec["total"]
            entry["AVG"] = round(sum(accs) / len(accs), 4) if accs else None
            entry["AVG_n_tasks_complete"] = n_complete
            per_length[length] = entry
        # Compute completion summary
        complete_count = sum(
            1
            for length in args.lengths
            if per_length.get(length, {}).get("AVG_n_tasks_complete", 0) == len(args.tasks)
        )
        out["models"][model_name] = {
            "complete_lengths": complete_count,
            "total_lengths": len(args.lengths),
            "per_length": per_length,
            "AVG_per_length": {
                length: per_length[length].get("AVG")
                for length in args.lengths
                if per_length.get(length, {}).get("AVG") is not None
            },
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_folder",
        type=str,
        default="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/babilong_results",
    )
    p.add_argument(
        "--output_json",
        type=str,
        default="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/status/babilong_results_cluster3.json",
    )
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "Llama-3.2-1B-Instruct",
            "Beacon-Qwen2-7B",
            "MemoryLLM-8B-chat",
        ],
    )
    p.add_argument(
        "--lengths",
        type=str,
        nargs="+",
        default=["1k", "2k", "4k", "8k", "16k", "32k"],
    )
    p.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10"],
    )
    p.add_argument("--min_rows", type=int, default=100)
    p.add_argument("--print_table", action="store_true", default=True)
    args = p.parse_args()

    result = aggregate(args)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[aggregate] wrote {out_path}")

    if args.print_table:
        for model, data in result["models"].items():
            print(f"\n=== {model} ===")
            print(f"complete_lengths: {data['complete_lengths']} / {data['total_lengths']}")
            header = "task    " + "  ".join(f"{l:>5s}" for l in args.lengths)
            print(header)
            for task in args.tasks:
                row = []
                for length in args.lengths:
                    v = data["per_length"].get(length, {}).get(task)
                    row.append(f"{v*100:>4.0f}%" if isinstance(v, (int, float)) else f"{'-':>5s}")
                print(f"{task:6s}  " + "  ".join(f"{c:>5s}" for c in row))
            avg_row = []
            for length in args.lengths:
                v = data["per_length"].get(length, {}).get("AVG")
                avg_row.append(f"{v*100:>4.0f}%" if isinstance(v, (int, float)) else f"{'-':>5s}")
            print(f"{'AVG':6s}  " + "  ".join(f"{c:>5s}" for c in avg_row))


if __name__ == "__main__":
    main()
