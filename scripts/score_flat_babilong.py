#!/usr/bin/env python
"""Score a FLAT sample-sharded BABILong result dir with the OFFICIAL
``babilong.metrics.compare_answers`` + ``TASK_LABELS`` (NO re.search).

Layout handled (flat, sample-level shards directly under <dir>):
    <dir>/<task>_<length>_..._shard{i}of{N}.csv
Each cell's shards are globbed and their rows summed (the shard row-sets
partition the full sample set, so summing correct/total == scoring the full
cell). This complements ``score_nested_babilong.py`` (which expects a nested
``<run>/<run>_<length>/`` layout).

Iron-Law-2 per (task, length) cell:
  * N shards present (default 8);
  * total rows == --expect (default 100);
  * empty_output count (output.strip() == "").

Usage:
    python scripts/score_flat_babilong.py <dir> [--tasks qa2 qa5] \
        [--lengths 8k 16k] [--num_shards 8] [--expect 100]
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "babilong-pkg"))
from babilong.metrics import TASK_LABELS, compare_answers  # noqa: E402


def score_cell(folder, task, length, num_shards, expect):
    pat = os.path.join(folder, f"{task}_{length}_*shard*of{num_shards}.csv")
    files = sorted(glob.glob(pat))
    labels = TASK_LABELS[task]
    shard_ids = set()
    correct = total = empty = 0
    for f in files:
        m = re.search(rf"_shard(\d+)of{num_shards}\.csv$", os.path.basename(f))
        if m:
            shard_ids.add(int(m.group(1)))
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                total += 1
                target = (row.get("target") or "").strip()
                output = (row.get("output") or "").strip()
                question = (row.get("question") or "").strip()
                if output == "":
                    empty += 1
                if compare_answers(target, output, question, labels):
                    correct += 1
    if total == 0:
        return None
    shard_ok = len(shard_ids) == num_shards
    n_ok = (expect <= 0) or (total == expect)
    il2 = shard_ok and n_ok and empty == 0
    return {
        "task": task, "length": length, "n": total, "correct": correct,
        "empty": empty, "shards": len(shard_ids), "shard_ok": shard_ok,
        "score": round(100.0 * correct / total, 2), "il2": il2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--tasks", nargs="+", default=["qa2", "qa5"])
    ap.add_argument("--lengths", nargs="+", default=["8k", "16k"])
    ap.add_argument("--num_shards", type=int, default=8)
    ap.add_argument("--expect", type=int, default=100)
    args = ap.parse_args()

    print(f"# folder: {args.folder}")
    print(f"{'task':<6}{'len':>5}{'n':>5}{'corr':>6}{'empty':>7}"
          f"{'shards':>8}{'score':>8}{'IL2':>5}")
    all_ok = True
    grid = {}
    for task in args.tasks:
        for length in args.lengths:
            c = score_cell(args.folder, task, length, args.num_shards, args.expect)
            if c is None:
                print(f"{task:<6}{length:>5}{'--- not present ---':>36}")
                all_ok = False
                continue
            grid[(task, length)] = c
            all_ok = all_ok and c["il2"]
            print(f"{task:<6}{length:>5}{c['n']:>5}{c['correct']:>6}"
                  f"{c['empty']:>7}{c['shards']:>8}{c['score']:>8}"
                  f"{'OK' if c['il2'] else 'BAD':>5}")
    print(f"\n# IRON-LAW-2 ALL CELLS OK: {all_ok}")
    print("\n# markdown (compare_answers %):")
    print("| task | " + " | ".join(args.lengths) + " |")
    print("|---|" + "|".join(["---:"] * len(args.lengths)) + "|")
    for task in args.tasks:
        cells = [f"{grid[(task,l)]['score']:.0f}" if (task, l) in grid else "—"
                 for l in args.lengths]
        print(f"| {task} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
