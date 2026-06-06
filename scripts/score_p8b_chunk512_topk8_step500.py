#!/usr/bin/env python
"""Score the P8b chunk512 top_k8 step500 BABILong eval using babilong.metrics
(compare_answers) — IDENTICAL metric to the chunk512 top_k16 baseline scoring.

Reads babilong_results/p8b_chunk512_topk8_step500/<output_name>/<qa>_<len>_...csv
where output_name = p8b_chunk512_topk8_step500_<len> (one dir per length, as
written by eval_p8b_chunk512_topk8_step500.sh).

Prints a qa1/qa2/qa5 x 0k-32k accuracy table and a CSV.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "babilong-pkg"))
from babilong.metrics import TASK_LABELS, compare_answers  # noqa: E402

RESULTS = ROOT / "babilong_results" / "p8b_chunk512_topk8_step500"
TASKS = ["qa1", "qa2", "qa5"]
LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k"]
SUFFIX = "_instruction_yes_examples_yes_post_prompt_yes_chat_template_no_system_prompt_no.csv"


def score_cell(path: Path, task: str) -> tuple[int, int]:
    correct = 0
    total = 0
    labels = TASK_LABELS[task]
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            target = (row.get("target") or "").strip()
            output = (row.get("output") or "").strip()
            question = (row.get("question") or "").strip()
            if not target or not question:
                continue
            total += 1
            if compare_answers(target, output, question, labels):
                correct += 1
    return correct, total


def main() -> int:
    grid: dict[tuple[str, str], tuple[int, int]] = {}
    for length in LENGTHS:
        d = RESULTS / f"p8b_chunk512_topk8_step500_{length}"
        for task in TASKS:
            csv_path = d / f"{task}_{length}{SUFFIX}"
            grid[(task, length)] = score_cell(csv_path, task) if csv_path.exists() else (-1, 0)

    header = " task  " + "".join(f"  {l:>4}" for l in LENGTHS)
    print(header)
    print("-" * len(header))
    for task in TASKS:
        cells = []
        for length in LENGTHS:
            c, t = grid[(task, length)]
            cells.append("  --" if (t == 0 or c < 0) else f"{100.0 * c / t:4.0f}")
        print(f" {task}   " + "".join(f"  {x:>4}" for x in cells))

    out = RESULTS / "p8b_chunk512_topk8_step500_score.csv"
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
