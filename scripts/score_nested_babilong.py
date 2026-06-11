#!/usr/bin/env python
"""Score a BABILong result dir with the nested layout
  <root>/<run>/<run>_<length>/<task>_<length>_<suffix>.csv
using third_party/babilong-pkg/babilong/metrics.compare_answers.

Usage: python scripts/score_nested_babilong.py <result_dir> [--expect N]
Prints a qa x length accuracy grid (correct/total).

A row-count guard (default --expect 100) prints a loud WARNING to stderr when a
cell parses to a different number of records than expected. This surfaces CSV
corruption (e.g. multi-line model outputs that an upstream writer failed to
quote, splitting one record across several physical lines) instead of letting
it silently skew the score. Pass ``--expect -1`` to disable the check (e.g.
when cells were produced with a non-default ``--limit``).
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "babilong-pkg"))
from babilong.metrics import TASK_LABELS, compare_answers  # noqa: E402

TASKS = ["qa1", "qa2", "qa5"]
LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k"]


def score_cell(paths, task: str, expect: int = 100):
    """Score one cell. ``paths`` is a list of CSV files whose rows are summed
    together — this is what makes sample-level shards (``..._shard{i}of{N}.csv``)
    merge transparently into a single cell score (their row sets partition the
    full sample set, so summing correct/total over all shards == scoring the
    full cell).
    """
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    correct = total = 0
    labels = TASK_LABELS[task]
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                total += 1
                target = (row.get("target") or "").strip()
                output = (row.get("output") or "").strip()
                question = (row.get("question") or "").strip()
                if compare_answers(target, output, question, labels):
                    correct += 1
    if expect is not None and expect > 0 and total != expect:
        shown = paths[0] if len(paths) == 1 else f"{len(paths)} shard files for cell"
        print(
            f"WARNING: {shown} parsed {total} records, expected {expect}. "
            f"Possible CSV corruption (unquoted multi-line output?) or an "
            f"incomplete shard set — the score for this cell may be unreliable.",
            file=sys.stderr,
        )
    return correct, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=str, help="Nested BABILong result dir")
    parser.add_argument(
        "--expect", type=int, default=100,
        help="Expected records per cell (default 100). Mismatch prints a "
             "stderr WARNING. Pass -1 to disable the check.",
    )
    args = parser.parse_args()

    rdir = Path(args.result_dir)
    run = rdir.name
    print(f"=== {run} ===")
    print("task   " + "  ".join(f"{l:>6s}" for l in LENGTHS))
    for t in TASKS:
        cells = []
        for L in LENGTHS:
            sub = rdir / f"{run}_{L}"
            hits = []
            if sub.exists():
                # Collect ALL CSVs for this cell. With sample-level sharding the
                # cell is split across ``{task}_{L}_..._shard{i}of{N}.csv`` files;
                # globbing + summing merges them. A non-sharded run yields exactly
                # one file here, so behaviour is unchanged.
                hits = sorted(sub.glob(f"{t}_{L}_*.csv"))
            if not hits:
                cells.append("   -- ")
            else:
                c, n = score_cell(hits, t, expect=args.expect)
                pct = 100.0 * c / n if n else 0.0
                cells.append(f"{pct:5.0f} ")
        print(f"{t:5s}  " + " ".join(cells))


if __name__ == "__main__":
    main()
