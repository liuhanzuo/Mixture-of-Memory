#!/usr/bin/env python
"""Position-stratified BABILong scorer for the HNST decisive test.

Reads result CSVs that carry the extra ``needle_chunks`` / ``n_chunks`` columns
(written by ``run_babilong_mem_space.py --record_needle_pos``) and buckets each
sample's accuracy by WHERE the needle chunk sits in the document:

  frac = min(needle_chunks) / max(1, n_chunks - 1)      # earliest mention
  early : frac <  1/3
  mid   : 1/3 <= frac < 2/3
  late  : frac >= 2/3

Scoring uses the official ``babilong.metrics.compare_answers``. Samples whose
needle could not be located (empty ``needle_chunks``) go to an ``unloc`` bucket
and are reported separately (never silently counted as early/mid/late).

Usage:
  python scripts/score_position_stratified.py <dir_glob> --task qa5 [--label NAME]

``<dir_glob>`` may be a directory (all ``*.csv`` under it are summed) or a glob
pattern. Multiple --dir args can be passed to compare arms side by side.
"""
from __future__ import annotations
import argparse
import csv
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "babilong-pkg"))
from babilong.metrics import TASK_LABELS, compare_answers  # noqa: E402


def _iter_csvs(spec: str):
    p = Path(spec)
    if p.is_dir():
        yield from sorted(p.glob("*.csv"))
    else:
        yield from (Path(x) for x in sorted(glob.glob(spec)))


def score_dir(spec: str, task: str):
    labels = TASK_LABELS[task]
    buckets = {"early": [0, 0], "mid": [0, 0], "late": [0, 0], "unloc": [0, 0]}
    overall = [0, 0]
    files = list(_iter_csvs(spec))
    for path in files:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                target = (row.get("target") or "").strip()
                output = (row.get("output") or "").strip()
                question = (row.get("question") or "").strip()
                ok = compare_answers(target, output, question, labels)
                overall[1] += 1
                overall[0] += int(ok)
                nc_str = (row.get("needle_chunks") or "").strip()
                n_chunks = row.get("n_chunks")
                try:
                    n_chunks = int(float(n_chunks)) if n_chunks not in (None, "") else 0
                except (ValueError, TypeError):
                    n_chunks = 0
                if not nc_str or n_chunks <= 0:
                    b = "unloc"
                else:
                    needle = [int(x) for x in nc_str.split(";") if x != ""]
                    frac = min(needle) / max(1, n_chunks - 1)
                    b = "early" if frac < 1 / 3 else ("mid" if frac < 2 / 3 else "late")
                buckets[b][1] += 1
                buckets[b][0] += int(ok)
    return buckets, overall, len(files)


def _pct(c, n):
    return f"{100.0 * c / n:5.1f}" if n else "  -- "


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="qa5")
    ap.add_argument("--dir", action="append", required=True,
                    help="dir or glob; repeat for multiple arms. "
                         "Format NAME=path to label the arm.")
    args = ap.parse_args()

    print(f"=== position-stratified {args.task} "
          f"(early<1/3, 1/3<=mid<2/3, late>=2/3 of doc) ===")
    hdr = f"{'arm':<22} {'early':>12} {'mid':>12} {'late':>12} {'overall':>12} {'unloc':>10}"
    print(hdr)
    print("-" * len(hdr))
    for spec in args.dir:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name, path = Path(spec).name, spec
        buckets, overall, nfiles = score_dir(path, args.task)
        def cell(b):
            c, n = buckets[b]
            return f"{_pct(c,n)}({c}/{n})"
        row = (f"{name:<22} "
               f"{cell('early'):>12} {cell('mid'):>12} {cell('late'):>12} "
               f"{_pct(*overall)+'('+str(overall[0])+'/'+str(overall[1])+')':>12} "
               f"{buckets['unloc'][1]:>10}")
        print(row)


if __name__ == "__main__":
    main()
