#!/usr/bin/env python
"""Score the 3 mem_space step500 ablations (P10/P11/topk8) using babilong.metrics
(compare_answers) — IDENTICAL metric to the chunk512 top_k16 baseline scoring.

Layout per prefix: babilong_results/<prefix>/<prefix>_<len>/<qa>_<len>_<SUFFIX>.csv
Writes babilong_results/<prefix>/<prefix>_score.csv for each.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "babilong-pkg"))
from babilong.metrics import TASK_LABELS, compare_answers  # noqa: E402

PREFIXES = [
    "p10_keyrep005_stgumbel_step500",
    "p11_deltarule_normreadout_step500",
    "p8b_chunk512_topk8_step500_v2",
]
TASKS = ["qa1", "qa2", "qa5"]
LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k"]
SUFFIX = "_instruction_yes_examples_yes_post_prompt_yes_chat_template_no_system_prompt_no.csv"


def score_cell(path: Path, task: str):
    correct = total = 0
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


def score_prefix(prefix: str) -> bool:
    base = ROOT / "babilong_results" / prefix
    grid = {}
    any_data = False
    for length in LENGTHS:
        d = base / f"{prefix}_{length}"
        for task in TASKS:
            csv_path = d / f"{task}_{length}{SUFFIX}"
            if csv_path.exists():
                c, t = score_cell(csv_path, task)
                grid[(task, length)] = (c, t)
                if t > 0:
                    any_data = True
            else:
                grid[(task, length)] = (-1, 0)

    header = f"=== {prefix} ===\n task  " + "".join(f"  {l:>4}" for l in LENGTHS)
    print(header)
    print("-" * len(header.split("\n")[-1]))
    for task in TASKS:
        cells = []
        for length in LENGTHS:
            c, t = grid[(task, length)]
            cells.append("  --" if (t == 0 or c < 0) else f"{100.0 * c / t:4.0f}")
        print(f" {task}   " + "".join(f"  {x:>4}" for x in cells))

    out = base / f"{prefix}_score.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
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
    print(f"Wrote {out}  (has_data={any_data})\n")
    return any_data


def main() -> int:
    targets = sys.argv[1:] if len(sys.argv) > 1 else PREFIXES
    ok = True
    for p in targets:
        if not score_prefix(p):
            ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
