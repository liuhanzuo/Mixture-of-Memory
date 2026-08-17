#!/usr/bin/env python
"""B01 four-arm collector: assert n_scored == expected PER CATEGORY, not just n_nan.

WHY THIS SHAPE
--------------
The task brief names a real accident from this repo: an eval produced 8 structurally
complete shards where every task was `skipped:true / n=0` while `n_nan=0`, so a
NaN-only aggregator passed it. So this asserts:

  1. per-arm total n == expected (1986)
  2. per-CATEGORY n == expected, and IDENTICAL across arms (a paired comparison is
     only meaningful on the same sample set)
  3. the two arms answered the SAME question ids (set equality on sample_id), not
     merely the same COUNT -- equal counts over different samples is the failure mode
     a count-only check waves through
  4. no shard contributed 0 rows, and shard row counts sum to the total
  5. predictions are not degenerately empty (would make F1 meaningless)

Exits non-zero on any violation. Writes one JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict


def read_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arms", required=True, help="comma-separated arm dir names")
    ap.add_argument("--num_shards", type=int, default=4)
    ap.add_argument("--expected_n", type=int, default=1986)
    ap.add_argument("--json_out", required=True)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    problems = []
    per_arm = {}

    for arm in arms:
        d = os.path.join(args.root, arm)
        shard_counts = {}
        rows = []
        for s in range(args.num_shards):
            p = os.path.join(d, f"preds_shard{s}of{args.num_shards}.jsonl")
            if not os.path.exists(p):
                problems.append(f"{arm}: MISSING shard file {p}")
                shard_counts[s] = None
                continue
            r = read_jsonl(p)
            shard_counts[s] = len(r)
            if len(r) == 0:
                problems.append(f"{arm}: shard {s} contributed 0 rows")
            rows.extend(r)

        # identify the per-sample key and the category field
        key = None
        for cand in ("sample_id", "global_id", "id", "qid"):
            if rows and cand in rows[0]:
                key = cand
                break
        if key is None and rows:
            problems.append(f"{arm}: no recognised sample-id field in {sorted(rows[0])[:12]}")

        catf = None
        for cand in ("category", "cat", "category_id"):
            if rows and cand in rows[0]:
                catf = cand
                break

        ids = [r.get(key) for r in rows] if key else []
        cats = Counter(r.get(catf) for r in rows) if catf else Counter()

        n_empty_pred = sum(1 for r in rows
                           if not str(r.get("pred", r.get("prediction", ""))).strip())

        scores_path = os.path.join(d, "scores.json")
        scores = json.load(open(scores_path)) if os.path.exists(scores_path) else None
        if scores is None:
            problems.append(f"{arm}: no scores.json")

        # ---- assertion 1: total n
        if len(rows) != args.expected_n:
            problems.append(f"{arm}: total n={len(rows)} != expected {args.expected_n}")
        # ---- assertion 4: shards sum
        ssum = sum(v for v in shard_counts.values() if v is not None)
        if ssum != len(rows):
            problems.append(f"{arm}: shard sum {ssum} != merged {len(rows)}")
        # ---- duplicate ids would inflate n
        if ids:
            dup = len(ids) - len(set(ids))
            if dup:
                problems.append(f"{arm}: {dup} duplicate sample ids")
        # ---- assertion 5
        if rows and n_empty_pred == len(rows):
            problems.append(f"{arm}: ALL {len(rows)} predictions are empty")

        per_arm[arm] = {
            "n_rows": len(rows),
            "shard_counts": shard_counts,
            "shard_sum": ssum,
            "id_field": key,
            "n_unique_ids": len(set(ids)) if ids else None,
            "category_field": catf,
            "per_category_n": {str(k): v for k, v in sorted(cats.items(), key=lambda x: str(x[0]))},
            "n_empty_predictions": n_empty_pred,
            "scores_overall": (scores or {}).get("overall"),
            "scores_per_category": (scores or {}).get("per_category"),
            "_id_set": set(ids),
        }

    # ---- assertion 2 + 3: cross-arm identity
    cross = {}
    if len(arms) >= 2:
        a0 = arms[0]
        for a in arms[1:]:
            same_cat = per_arm[a0]["per_category_n"] == per_arm[a]["per_category_n"]
            s0, s1 = per_arm[a0]["_id_set"], per_arm[a]["_id_set"]
            same_ids = s0 == s1
            cross[f"{a0}_vs_{a}"] = {
                "per_category_n_identical": same_cat,
                "sample_id_sets_identical": same_ids,
                "n_only_in_first": len(s0 - s1),
                "n_only_in_second": len(s1 - s0),
            }
            if not same_cat:
                problems.append(f"{a0} vs {a}: per-category n DIFFER -> not a paired comparison")
            if not same_ids:
                problems.append(f"{a0} vs {a}: sample id SETS differ "
                                f"({len(s0 - s1)} / {len(s1 - s0)} exclusive) -> "
                                f"equal counts over different samples")

    for v in per_arm.values():
        v.pop("_id_set", None)

    out = {
        "root": args.root,
        "arms": arms,
        "expected_n": args.expected_n,
        "num_shards": args.num_shards,
        "per_arm": per_arm,
        "cross_arm": cross,
        "problems": problems,
        "verdict": "PASS" if not problems else "FAIL",
        "what_was_asserted": [
            "per-arm total n == expected_n",
            "per-category n == identical across arms",
            "sample_id SETS identical across arms (not just counts)",
            "no shard contributed 0 rows; shard counts sum to merged n",
            "no duplicate sample ids",
            "predictions not all-empty",
        ],
    }
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("verdict", "problems", "cross_arm")}, indent=2))
    for a in arms:
        v = per_arm[a]
        print(f"\n[{a}] n={v['n_rows']} shards={v['shard_counts']} "
              f"empty_preds={v['n_empty_predictions']}")
        print(f"  per_category_n = {v['per_category_n']}")
    print(f"\nwrote {args.json_out}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
