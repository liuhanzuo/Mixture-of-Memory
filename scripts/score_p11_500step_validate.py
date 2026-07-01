#!/usr/bin/env python
"""Score outputs/eval_p11_500step_validate/ — the FSDP-fix validation run.
Mirrors scripts/score_p11_step500.py but points at the validate output dir.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "babilong-pkg"))
from babilong.metrics import TASK_LABELS, compare_answers  # type: ignore  # noqa: E402

RESULTS = ROOT / "outputs" / "eval_p11_500step_validate"
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
            d = RESULTS / f"p11_500step_validate_{task}_{split}"
            for length in length_set:
                csv_path = d / f"{task}_{length}{SUFFIX}"
                if not csv_path.exists():
                    grid[(task, length)] = (-1, 0)
                    continue
                grid[(task, length)] = score_cell(csv_path, task)

    header = " task  " + "".join(f"  {l:>4}" for l in LENGTHS) + "   | mean"
    print(header)
    print("-" * len(header))
    overall_sum = 0.0
    overall_n = 0
    short_sum, short_n = 0.0, 0
    long_sum, long_n = 0.0, 0
    task_means: dict[str, float] = {}
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
        task_means[task] = task_mean
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
    print(f"  P8 (DDP, 500 steps)            : 59.14   qa1=68.0 qa2=39.4 qa5=70.0")
    print(f"  P11 step500 (FSDP, broken)     : 33.81   (qa1=35.7 qa2=13.3 qa5=52.4 — pre-fix per spec)")
    print(f"  P11 fixed-FSDP 500-step (this) : {overall:5.2f}   qa1={task_means.get('qa1', float('nan')):.1f} qa2={task_means.get('qa2', float('nan')):.1f} qa5={task_means.get('qa5', float('nan')):.1f}")
    print()
    print(f"  vs P8 (DDP 500): {overall - 59.14:+.2f}pp")
    print(f"  vs P11 broken : {overall - 33.81:+.2f}pp")
    print()
    if overall >= 55:
        verdict = "VERDICT: bug fix CONFIRMED — architecture works under FSDP."
    elif overall >= 40:
        verdict = "VERDICT: fix PARTIALLY works, more investigation needed."
    else:
        verdict = "VERDICT: fix did NOT help, deeper issue."
    print(verdict)

    out = RESULTS / "p11_500step_validate_score.csv"
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
