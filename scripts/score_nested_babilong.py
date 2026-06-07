#!/usr/bin/env python
"""Score a BABILong result dir with the nested layout
  <root>/<run>/<run>_<length>/<task>_<length>_<suffix>.csv
using third_party/babilong-pkg/babilong/metrics.compare_answers.

Usage: python scripts/score_nested_babilong.py <result_dir>
Prints a qa x length accuracy grid (correct/total).
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "babilong-pkg"))
from babilong.metrics import TASK_LABELS, compare_answers  # noqa: E402

TASKS = ["qa1", "qa2", "qa5"]
LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k"]


def score_cell(path: Path, task: str):
    correct = total = 0
    labels = TASK_LABELS[task]
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            target = (row.get("target") or "").strip()
            output = (row.get("output") or "").strip()
            question = (row.get("question") or "").strip()
            if compare_answers(target, output, question, labels):
                correct += 1
    return correct, total


def main():
    rdir = Path(sys.argv[1])
    run = rdir.name
    print(f"=== {run} ===")
    print("task   " + "  ".join(f"{l:>6s}" for l in LENGTHS))
    for t in TASKS:
        cells = []
        for L in LENGTHS:
            sub = rdir / f"{run}_{L}"
            hit = None
            if sub.exists():
                for f in sub.glob(f"{t}_{L}_*.csv"):
                    hit = f
                    break
            if hit is None:
                cells.append("   -- ")
            else:
                c, n = score_cell(hit, t)
                pct = 100.0 * c / n if n else 0.0
                cells.append(f"{pct:5.0f} ")
        print(f"{t:5s}  " + " ".join(cells))


if __name__ == "__main__":
    main()
