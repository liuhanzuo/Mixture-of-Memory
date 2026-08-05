#!/usr/bin/env python3
"""MANDATORY-REPORT baseline for any Paper C SQuAD-format eval jsonl.

Prints the score a MODEL THAT NEVER READS ITS INPUT would obtain, so that every
Paper C table can carry it as a floor row.  Any arm scoring below this floor is
NOT demonstrating capability -- it is losing to a constant.

Why this exists: `data/squad_val.jsonl` has 997/2000 = 49.85% of its
`target_text` set to the single Chinese refusal string, so a constant emitter
scores EM 49.85 -- above A4_hero (29.30), A3_fromscratch (26.05) and BASE_ref
(33.85).  That invalidated the entire #92/#133 SQuAD slice as capability
evidence.  See versions/paperC_scoping.md (2026-08-05 reviewer verdict).

Baselines reported
------------------
  constant-majority : always emit argmax_t count(target_text == t)
  constant-refusal  : always emit the Chinese refusal string
  empty             : always emit "" (F1 convention from eval_olmo2_closedbook_qa:
                      empty-vs-empty scores 1.0, so this is non-trivial when the
                      set has empty golds)
  stratified        : the majority baseline evaluated separately on the refusal
                      and answerable strata (this is what a table must show to
                      prove an arm is not just riding the label prior)

Scoring reuses scripts/eval_olmo2_closedbook_qa.score_prediction VERBATIM (the
same normalize_answer / token-F1 that eval_paperC_squad_emf1.py uses), so the
numbers are directly comparable to any arm's summary.json.

Usage
-----
    python scripts/report_constant_baseline.py data/squad_val.jsonl
    python scripts/report_constant_baseline.py data/paperC_squad_v2/*.jsonl
    python scripts/report_constant_baseline.py --json out.json data/squad_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eval_olmo2_closedbook_qa import score_prediction  # noqa: E402

REFUSAL_TEXT = "根据提供的信息无法回答这个问题"


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def const_score(rows, pred):
    """EM/F1 of emitting `pred` for every row."""
    em = 0
    f1 = 0.0
    for r in rows:
        sc = score_prediction(pred, [(r.get("target_text") or "").strip()])
        em += sc["em"]
        f1 += sc["f1"]
    n = max(len(rows), 1)
    return {"n": len(rows), "em": em / n, "f1": f1 / n, "em_hits": em}


def report(path, as_json=False):
    rows = load(path)
    n = len(rows)
    if n == 0:
        print(f"{path}: EMPTY")
        return None
    tgt = Counter((r.get("target_text") or "").strip() for r in rows)
    maj, maj_n = tgt.most_common(1)[0]

    ref_rows = [r for r in rows if (r.get("target_text") or "").strip() == REFUSAL_TEXT]
    ans_rows = [r for r in rows if (r.get("target_text") or "").strip() != REFUSAL_TEXT]

    out = {
        "path": path,
        "n": n,
        "n_unique_targets": len(tgt),
        "majority_target": maj,
        "majority_count": maj_n,
        "majority_frac": maj_n / n,
        "refusal_rate": len(ref_rows) / n,
        "n_refusal": len(ref_rows),
        "n_answerable": len(ans_rows),
        "baseline_constant_majority": const_score(rows, maj),
        "baseline_constant_refusal": const_score(rows, REFUSAL_TEXT),
        "baseline_empty": const_score(rows, ""),
        "top10_targets": [{"target": t, "count": c, "frac": c / n}
                          for t, c in tgt.most_common(10)],
        "stratified": {
            "refusal_stratum": {
                "n": len(ref_rows), "frac": len(ref_rows) / n,
                "constant_majority_here": const_score(ref_rows, maj) if ref_rows else None,
            },
            "answerable_stratum": {
                "n": len(ans_rows), "frac": len(ans_rows) / n,
                "constant_majority_here": const_score(ans_rows, maj) if ans_rows else None,
            },
        },
    }

    if as_json:
        return out

    b = out["baseline_constant_majority"]
    print("=" * 78)
    print(f"FILE  {path}")
    print(f"  n={n}  unique targets={len(tgt)}  refusal_rate={out['refusal_rate']*100:.2f}%"
          f"  ({len(ref_rows)} refusal / {len(ans_rows)} answerable)")
    print(f"  majority target ({maj_n}/{n} = {maj_n/n*100:.2f}%): {maj[:60]!r}")
    print()
    print("  *** MANDATORY FLOOR ROW (input-blind constant model) ***")
    print(f"    constant-majority    EM={b['em']*100:6.2f}   F1={b['f1']*100:6.2f}")
    r = out["baseline_constant_refusal"]
    print(f"    constant-refusal     EM={r['em']*100:6.2f}   F1={r['f1']*100:6.2f}")
    e = out["baseline_empty"]
    print(f"    empty-string         EM={e['em']*100:6.2f}   F1={e['f1']*100:6.2f}")
    print("    => any arm below the constant-majority row is NOT showing capability.")
    print()
    print("  stratified (a table must report these, not just the pooled number):")
    for k in ("refusal_stratum", "answerable_stratum"):
        s = out["stratified"][k]
        cm = s["constant_majority_here"]
        cmtxt = (f"EM={cm['em']*100:6.2f} F1={cm['f1']*100:6.2f}" if cm else "n/a")
        print(f"    {k:20s} n={s['n']:5d} ({s['frac']*100:5.2f}%)  const-majority {cmtxt}")
    print()
    print("  top-10 targets:")
    for t in out["top10_targets"]:
        print(f"    {t['count']:6d} ({t['frac']*100:6.2f}%)  {t['target'][:62]!r}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="eval jsonl file(s)")
    ap.add_argument("--json", default="", help="also dump all reports to this json path")
    args = ap.parse_args()

    reports = []
    for p in args.paths:
        r = report(p, as_json=False)
        if r:
            reports.append(r)

    if len(reports) > 1:
        print("=" * 78)
        print("SUMMARY (constant-majority floor per file)")
        print(f"  {'file':46s} {'n':>6s} {'refus%':>7s} {'EM':>7s} {'F1':>7s}")
        for r in reports:
            b = r["baseline_constant_majority"]
            print(f"  {os.path.basename(r['path']):46s} {r['n']:6d} "
                  f"{r['refusal_rate']*100:7.2f} {b['em']*100:7.2f} {b['f1']*100:7.2f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)
        print(f"\n[json] {args.json}")


if __name__ == "__main__":
    sys.exit(main())
