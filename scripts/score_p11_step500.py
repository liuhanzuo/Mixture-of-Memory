#!/usr/bin/env python
"""Score outputs/eval_p11_step500/ in the same way as the P11 final + step4500
evals (`third_party/babilong-pkg/babilong/metrics.py`).
Prints a 21-cell table + comparison rows.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "babilong-pkg"))
from babilong.metrics import TASK_LABELS, compare_answers  # type: ignore  # noqa: E402

RESULTS = ROOT / "outputs" / "eval_p11_step500"
TASKS = ["qa1", "qa2", "qa5"]
LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k"]
SHORT = {"0k", "1k", "2k", "4k"}
LONG = {"8k", "16k", "32k"}
SUFFIX = "_instruction_yes_examples_yes_post_prompt_yes_chat_template_yes_system_prompt_no.csv"


def score_cell(path: Path, task: str) -> tuple[int, int]:
    correct = 0
    total = 0
    labels = TASK_LABELS[task]
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            target = (row.get("target") or "").strip()
            output = (row.get("output") or "").strip()
            question = (row.get("question") or "").strip()
            if compare_answers(target, output, question, labels):
                correct += 1
    return correct, total


def main() -> int:
    grid: dict[tuple[str, str], tuple[int, int]] = {}
    for task in TASKS:
        for split, length_set in (("short", SHORT), ("long", LONG)):
            d = RESULTS / f"p11step500_{task}_{split}"
            for length in length_set:
                csv_path = d / f"{task}_{length}{SUFFIX}"
                if not csv_path.exists():
                    grid[(task, length)] = (-1, 0)
                    continue
                grid[(task, length)] = score_cell(csv_path, task)

    # Print 21-cell table
    header = " task  " + "".join(f"  {l:>4}" for l in LENGTHS) + "   | mean"
    print(header)
    print("-" * len(header))
    overall_sum = 0.0
    overall_n = 0
    short_sum, short_n = 0.0, 0
    long_sum, long_n = 0.0, 0
    for task in TASKS:
        cells = []
        task_sum, task_n = 0.0, 0
        for length in LENGTHS:
            c, t = grid[(task, length)]
            if t == 0 or c < 0:
                cells.append("  --")
            else:
                pct = 100.0 * c / t
                cells.append(f"{pct:5.1f}")
                task_sum += pct
                task_n += 1
                overall_sum += pct
                overall_n += 1
                if length in SHORT:
                    short_sum += pct
                    short_n += 1
                else:
                    long_sum += pct
                    long_n += 1
        task_mean = task_sum / task_n if task_n else float("nan")
        print(f" {task}   " + "".join(f"  {x:>4}" for x in cells) + f"   | {task_mean:5.2f}")

    print("-" * len(header))
    overall = overall_sum / overall_n if overall_n else float("nan")
    short_avg = short_sum / short_n if short_n else float("nan")
    long_avg = long_sum / long_n if long_n else float("nan")
    print(f"\nOverall 21-cell mean:  {overall:5.2f}")
    print(f"Short avg (0k/1k/2k/4k):  {short_avg:5.2f}")
    print(f"Long avg  (8k/16k/32k):    {long_avg:5.2f}")
    print()
    print("Comparison:")
    print(f"  P11 step500 (this run)  : {overall:5.2f}")
    print(f"  P11 final (step5000)    : 26.33")
    print(f"  P8 8B (DDP, 500 steps)  : 59.14")
    print(f"  P-1B v2 final           : 37.43")
    print()
    print(f"  vs P11 final: {overall - 26.33:+.2f}pp")
    print(f"  vs P8       : {overall - 59.14:+.2f}pp")
    print(f"  vs P-1B v2  : {overall - 37.43:+.2f}pp")
    print()
    if overall >= 50:
        verdict = (
            "VERDICT: long-training overfit is the dominant cause. "
            "The FSDP gradient bug exists but is NOT catastrophic at short training horizons."
        )
    elif overall <= 35:
        verdict = (
            "VERDICT: FSDP's gradient bug damaged the run from t=0. "
            "The freeze of L3/gates is fundamentally breaking the architecture even at 500 steps."
        )
    else:
        verdict = (
            "VERDICT: BOTH factors contribute. "
            "FSDP gradient bug measurably hurts step-500 vs DDP P8, and additional long-horizon overfit drives the further drop to 26.33."
        )
    print(verdict)

    # Also write a CSV
    out = RESULTS / "p11_step500_score.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "length", "correct", "total", "accuracy_pct"])
        for task in TASKS:
            for length in LENGTHS:
                c, t = grid[(task, length)]
                if t == 0 or c < 0:
                    w.writerow([task, length, "", "", ""])
                else:
                    w.writerow([task, length, c, t, f"{100.0 * c / t:.1f}"])
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
