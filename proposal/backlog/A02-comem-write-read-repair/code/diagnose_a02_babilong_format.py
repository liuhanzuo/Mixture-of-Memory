#!/usr/bin/env python
"""A02 — diagnose WHY the BABILong A4-vs-A5 ordering inverts.  Pure CPU, zero GPU.

This is the mechanism probe for A02_BABILONG_MISORDER_PREREG.md 1.2/F3. The
pre-registered mechanism was RETRIEVAL DOMINATION; the conditional analysis
refuted it (the inversion is present and LARGER on retrieval-HIT items). This
script tests the remaining candidate that the conditional analysis pointed to:

  **FORMAT/SCORER INTERACTION** — `babilong.metrics.compare_answers` requires the
  target to be the *only* task label in `preprocess_output(output)`, and
  `preprocess_output` truncates at the FIRST period. A base LM that answers with a
  multiple-choice list ("Choices: A. In the kitchen B. ...") is truncated to
  "choices: a" and scores 0 **regardless of whether it located the fact**. So the
  metric measures OUTPUT FORMAT as much as retrieval+reading, and two arms can be
  ordered by which one happens to emit a terse declarative sentence.

Metrics reported per (cell, arm), all computed with the CANONICAL scorer imported
from babilong.metrics (never reimplemented):

  score            canonical compare_answers accuracy (the published number)
  listfmt          fraction of outputs that are a multiple-choice enumeration
  trunc_kills      fraction where the target IS present in the raw output but is
                   REMOVED by preprocess_output's first-period truncation
  no_label         fraction with zero task labels surviving truncation (auto-0)
  multi_label      fraction with >=2 labels surviving (auto-0 even if target is one)
  target_in_raw    fraction whose RAW output contains the target string anywhere
                   -- a format-insensitive LOWER BOUND on "knew the answer"

`target_in_raw` is deliberately lenient (it would also credit a lucky mention);
it is used only to show that the canonical score and a format-insensitive read of
the same generations can ORDER TWO ARMS DIFFERENTLY. It is NOT proposed as a
replacement metric and no headline number is computed from it.

Usage:  python diagnose_a02_babilong_format.py [--out <dir>]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

from babilong.metrics import TASK_LABELS, compare_answers, preprocess_output

W = Path(os.environ.get(
    "A02_W", "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"))

# arm -> babilong_results subdir (A0/A4 are the dvr/phase anchors; rest are read-tax)
ARMS = {
    "A0": "a02_dvr_babilong_j0_top12",
    "A1": "a02_rtax_babilong_A1_j0control",
    "A2": "a02_rtax_babilong_A2_j6",
    "A3": "a02_rtax_babilong_A3_j9",
    "A4": "a02_babilong_c2_j12_readlora",
    "A5": "a02_rtax_babilong_A5_j18",
    "A6": "a02_rtax_babilong_A6_j12_r40",
}
NSHARD = 8


def load_cell(subdir, task, length):
    files = sorted(glob.glob(str(W / "babilong_results" / subdir
                                 / f"{task}_{length}_*shard*of{NSHARD}.csv")))
    if len(files) != NSHARD:
        return None, f"SHARD_INCOMPLETE {len(files)}/{NSHARD}"
    rows = []
    for f in files:
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    if len(rows) != 100:
        return None, f"N_MISMATCH {len(rows)}!=100"
    return rows, None


def is_list_format(raw: str) -> bool:
    low = raw.strip().lower()
    if low.startswith(("choices", "options")):
        return True
    head = low[:60]
    return ("a." in head and "b." in head) or ("a)" in head and "b)" in head)


def main(out_dir: Path):
    res, errs = {}, []
    for task in ("qa1", "qa2", "qa5"):
        labs = {l.lower() for l in TASK_LABELS[task]}
        for length in ("16k", "32k"):
            cell = f"babilong|{task}|{length}"
            res[cell] = {}
            for arm, sub in ARMS.items():
                rows, err = load_cell(sub, task, length)
                if err:
                    errs.append(f"{cell}/{arm}: {err}")
                    continue
                n = len(rows)
                sc = lst = trunc = nolab = multi = inraw = 0
                for r in rows:
                    raw, tgt, q = r["output"], r["target"].lower(), r["question"].lower()
                    po = preprocess_output(raw)
                    if compare_answers(r["target"], raw, r["question"], TASK_LABELS[task]):
                        sc += 1
                    if is_list_format(raw):
                        lst += 1
                    if tgt in raw.lower():
                        inraw += 1
                    if tgt in raw.lower() and tgt not in po:
                        trunc += 1
                    inq = {l for l in labs if l in q}
                    surv = {l for l in labs if l in po} - inq
                    if not surv:
                        nolab += 1
                    elif len(surv) >= 2:
                        multi += 1
                res[cell][arm] = {
                    "n": n,
                    "score": round(100 * sc / n, 1),
                    "listfmt": round(100 * lst / n, 1),
                    "trunc_kills": round(100 * trunc / n, 1),
                    "no_label": round(100 * nolab / n, 1),
                    "multi_label": round(100 * multi / n, 1),
                    "target_in_raw": round(100 * inraw / n, 1),
                }

    hdr = (f"{'cell':20s}{'arm':4s} {'score':>6s} {'listfmt':>8s} {'truncK':>7s} "
           f"{'noLab':>6s} {'multiL':>7s} {'tgtInRaw':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for cell, per in res.items():
        for arm in ARMS:
            if arm not in per:
                continue
            m = per[arm]
            print(f"{cell:20s}{arm:4s} {m['score']:6.1f} {m['listfmt']:7.1f}% "
                  f"{m['trunc_kills']:6.1f}% {m['no_label']:5.1f}% "
                  f"{m['multi_label']:6.1f}% {m['target_in_raw']:8.1f}%")
        print()

    print("=== A4 vs A5 under the canonical metric vs a format-insensitive read ===")
    print(f"{'cell':20s} {'canon A4':>9s} {'canon A5':>9s} {'canon d':>8s} "
          f"{'raw A4':>7s} {'raw A5':>7s} {'raw d':>7s}  agree?")
    flips = []
    for cell, per in res.items():
        if "A4" not in per or "A5" not in per:
            continue
        c4, c5 = per["A4"]["score"], per["A5"]["score"]
        r4, r5 = per["A4"]["target_in_raw"], per["A5"]["target_in_raw"]
        cd, rd = c4 - c5, r4 - r5
        agree = (cd > 0) == (rd > 0) or (cd == 0 and rd == 0)
        if not agree:
            flips.append(cell)
        print(f"{cell:20s} {c4:9.1f} {c5:9.1f} {cd:+8.1f} {r4:7.1f} {r5:7.1f} "
              f"{rd:+7.1f}  {'yes' if agree else 'NO -- SIGN FLIP'}")
    print(f"\nsign flips between canonical and format-insensitive: {len(flips)}/6 {flips}")

    if errs:
        print("\nERRORS:")
        for e in errs:
            print("  ", e)

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "a02_babilong_format_diagnosis.json"
    json.dump({
        "generated_by": "diagnose_a02_babilong_format.py",
        "prereg": "A02_BABILONG_MISORDER_PREREG.md (1.2 mechanism / F3)",
        "gpu_spent": "ZERO -- re-scores generations already on disk",
        "scorer": "babilong.metrics.compare_answers + preprocess_output, IMPORTED",
        "caveat_target_in_raw": (
            "lenient LOWER BOUND on 'knew the answer'; would also credit an "
            "incidental mention. Used ONLY to show the canonical metric and a "
            "format-insensitive read of the SAME generations order arms "
            "differently. NOT proposed as a replacement metric."),
        "per_cell": res, "canonical_vs_raw_sign_flips": flips, "errors": errs,
    }, open(dst, "w"), indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        W / "proposal/backlog/A02-comem-write-read-repair/evidence/babilong_misorder"))
    main(Path(ap.parse_args().out))
