#!/usr/bin/env python
"""A02 — per-item statistics for the FORMAT mechanism behind the BABILong misordering.

Zero GPU: re-scores generations already on disk with the CANONICAL scorer.

CONTEXT. A02_BABILONG_MISORDER_PREREG.md 1.2 pre-registered RETRIEVAL DOMINATION as
the mechanism for the A4-vs-A5 inversion. The conditional analysis REFUTED it: the
inversion is present and LARGER on retrieval-HIT items. This script tests the
mechanism the conditional analysis pointed to instead.

MECHANISM UNDER TEST. `babilong.metrics.preprocess_output` truncates the generation at
the FIRST period, and `compare_answers` then requires the target to be the ONLY task
label surviving. A base LM that answers with a multiple-choice enumeration
("Choices: A. In the kitchen B. In the office ...") is truncated to "choices: a" and
scores 0 *whether or not it located the fact*. If two arms emit that format at
different rates, the metric can order them by FORMAT rather than by reading ability.

STATISTICS (all paired, per cell, canonical scorer imported)
  1. is_list rate, A4 vs A5: McNemar exact -- is the format asymmetry real?
  2. trunc_kill rate (target in raw, destroyed by truncation): McNemar exact
  3. THE KEY CONTRAST: canonical A4-vs-A5 restricted to items where NEITHER arm
     suffered a truncation kill, i.e. items the metric could score on its merits.
     *** This is POST-TREATMENT conditioning (format is an outcome of the arm), so it
     is a COLLIDER risk and is reported as DIAGNOSTIC, not causal. *** It answers a
     narrow, legitimate question: on the items the metric did not auto-zero, does the
     published ordering persist?
  4. cell-level dissociation: Fisher exact on (sign flip) x (high vs low list format)

PAIRING. BABILong CSVs are sharded; the dvr analyzer's convention is
`row r of shard s == dataset index s + r*NSHARD`. That convention is reproduced here
and VERIFIED by asserting (question, target) equality across arms per index -- the
same G2 pairing assertion the dvr gate used. A mismatch refuses the cell.

Usage: python analyze_a02_format_mechanism.py [--out <dir>]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path

from babilong.metrics import TASK_LABELS, compare_answers, preprocess_output

W = Path(os.environ.get(
    "A02_W", "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"))

ARMS = {
    "A0": "a02_dvr_babilong_j0_top12",
    "A4": "a02_babilong_c2_j12_readlora",
    "A5": "a02_rtax_babilong_A5_j18",
}
NSHARD = 8
N_BOOT = 5000
SEED = 42


def _log_binom_coef(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def mcnemar_exact(a, b):
    bb = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    cc = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = bb + cc
    if n == 0:
        return bb, cc, 0, 1.0
    obs = abs(bb - n / 2.0)
    tot = sum(math.exp(_log_binom_coef(n, k) - n * math.log(2.0))
              for k in range(n + 1) if abs(k - n / 2.0) >= obs - 1e-12)
    return bb, cc, n, float(min(1.0, tot))


def paired_boot(a, b, n_boot=N_BOOT, seed=SEED):
    import numpy as np
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    boots = 100.0 * (a[idx].mean(axis=1) - b[idx].mean(axis=1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(100.0 * (a.mean() - b.mean())), float(lo), float(hi)


def fisher_exact_2x2(t):
    (a, b), (c, d) = t
    n = a + b + c + d

    def p_tab(a_):
        b_ = a + b - a_
        c_ = a + c - a_
        d_ = d - (a_ - a)
        if min(a_, b_, c_, d_) < 0:
            return 0.0
        return math.exp(_log_binom_coef(a + b, a_) + _log_binom_coef(c + d, c_)
                        - _log_binom_coef(n, a + c))
    p_obs = p_tab(a)
    lo = max(0, a - d)
    hi = min(a + b, a + c)
    return sum(p_tab(x) for x in range(lo, hi + 1) if p_tab(x) <= p_obs + 1e-12)


def is_list_format(raw: str) -> bool:
    low = raw.strip().lower()
    if low.startswith(("choices", "options")):
        return True
    head = low[:60]
    return ("a." in head and "b." in head) or ("a)" in head and "b)" in head)


def load_indexed(subdir, task, length):
    """Return {dataset_index: row} using the dvr convention idx = shard + row*NSHARD."""
    out = {}
    files = glob.glob(str(W / "babilong_results" / subdir
                          / f"{task}_{length}_*shard*of{NSHARD}.csv"))
    if len(files) != NSHARD:
        return None, f"SHARD_INCOMPLETE {len(files)}/{NSHARD}"
    for f in files:
        s = int(Path(f).stem.split("shard")[1].split("of")[0])
        with open(f) as fh:
            for r, row in enumerate(csv.DictReader(fh)):
                out[s + r * NSHARD] = row
    if len(out) != 100:
        return None, f"N_MISMATCH {len(out)}!=100"
    return out, None


def main(out_dir: Path):
    res, errs = {}, []
    for task in ("qa1", "qa2", "qa5"):
        for length in ("16k", "32k"):
            cell = f"babilong|{task}|{length}"
            per_arm, bad = {}, False
            for arm, sub in ARMS.items():
                d, e = load_indexed(sub, task, length)
                if e:
                    errs.append(f"{cell}/{arm}: {e}")
                    bad = True
                else:
                    per_arm[arm] = d
            if bad:
                continue
            idx = sorted(set.intersection(*[set(d) for d in per_arm.values()]))
            if len(idx) != 100:
                errs.append(f"{cell}: common idx {len(idx)}!=100")
                continue
            # G2 pairing: (question,target) must agree across arms per index
            pair_fail = [i for i in idx
                         if len({(per_arm[a][i]["question"].strip(),
                                  per_arm[a][i]["target"].strip()) for a in per_arm}) > 1]
            if pair_fail:
                errs.append(f"{cell}: G2 pairing failed on {len(pair_fail)} idx")
                continue

            feat = {a: {} for a in per_arm}
            for a, d in per_arm.items():
                for i in idx:
                    raw, tgt, q = d[i]["output"], d[i]["target"].lower(), d[i]["question"]
                    po = preprocess_output(raw)
                    feat[a][i] = {
                        "correct": int(bool(compare_answers(d[i]["target"], raw, q,
                                                            TASK_LABELS[task]))),
                        "is_list": int(is_list_format(raw)),
                        "tgt_in_raw": int(tgt in raw.lower()),
                        "trunc_kill": int(tgt in raw.lower() and tgt not in po),
                    }

            def v(a, k, sub=None):
                return [feat[a][i][k] for i in (sub if sub is not None else idx)]

            entry = {"n": len(idx)}
            # 1 + 2: format asymmetry A4 vs A5
            for k in ("is_list", "trunc_kill"):
                b_, c_, nd, p = mcnemar_exact(v("A4", k), v("A5", k))
                entry[k] = {
                    "A4_rate": round(100 * sum(v("A4", k)) / len(idx), 1),
                    "A5_rate": round(100 * sum(v("A5", k)) / len(idx), 1),
                    "mcnemar": {"b_A4only": b_, "c_A5only": c_, "n_discordant": nd,
                                "p_two_sided_exact": round(p, 6)},
                    "significant": bool(p < 0.05),
                }
            # 3: canonical contrast on items neither arm auto-zeroed by truncation
            keep = [i for i in idx
                    if not feat["A4"][i]["trunc_kill"] and not feat["A5"][i]["trunc_kill"]]
            full_d, full_lo, full_hi = paired_boot(v("A4", "correct"), v("A5", "correct"))
            entry["canonical_full"] = {
                "A4": round(100 * sum(v("A4", "correct")) / len(idx), 1),
                "A5": round(100 * sum(v("A5", "correct")) / len(idx), 1),
                "delta_pp": round(full_d, 2), "ci95": [round(full_lo, 2), round(full_hi, 2)],
                "point_inverted": bool(sum(v("A5", "correct")) >= sum(v("A4", "correct"))),
            }
            if len(keep) >= 10:
                d3, lo3, hi3 = paired_boot(v("A4", "correct", keep), v("A5", "correct", keep))
                b3, c3, nd3, p3 = mcnemar_exact(v("A4", "correct", keep),
                                                v("A5", "correct", keep))
                entry["canonical_no_trunc_kill"] = {
                    "n": len(keep),
                    "A4": round(100 * sum(v("A4", "correct", keep)) / len(keep), 1),
                    "A5": round(100 * sum(v("A5", "correct", keep)) / len(keep), 1),
                    "delta_pp": round(d3, 2), "ci95": [round(lo3, 2), round(hi3, 2)],
                    "mcnemar_p": round(p3, 6),
                    "point_inverted": bool(sum(v("A5", "correct", keep))
                                           >= sum(v("A4", "correct", keep))),
                    "CAVEAT": "POST-TREATMENT conditioning (collider risk) -- DIAGNOSTIC ONLY",
                }
            else:
                entry["canonical_no_trunc_kill"] = {"n": len(keep), "status": "TOO_SMALL"}
            entry["tgt_in_raw"] = {
                "A4": round(100 * sum(v("A4", "tgt_in_raw")) / len(idx), 1),
                "A5": round(100 * sum(v("A5", "tgt_in_raw")) / len(idx), 1),
            }
            res[cell] = entry

    # 4: cell-level dissociation -- sign flip vs list-format prevalence
    flip, hi_list = {}, {}
    for cell, e in res.items():
        cd = e["canonical_full"]["delta_pp"]
        rd = e["tgt_in_raw"]["A4"] - e["tgt_in_raw"]["A5"]
        flip[cell] = bool((cd <= 0) and (rd > 0))
        hi_list[cell] = bool(e["is_list"]["A4_rate"] >= 25.0)
    a = sum(1 for c in res if flip[c] and hi_list[c])
    b = sum(1 for c in res if flip[c] and not hi_list[c])
    c_ = sum(1 for c in res if not flip[c] and hi_list[c])
    d_ = sum(1 for c in res if not flip[c] and not hi_list[c])
    diss = {
        "table_flip_x_highlist": [[a, b], [c_, d_]],
        "fisher_exact_p_two_sided": round(fisher_exact_2x2([[a, b], [c_, d_]]), 6),
        "cells_flipped": [c for c in res if flip[c]],
        "cells_high_list": [c for c in res if hi_list[c]],
        "note": ("6 cells is the whole family, so the minimum attainable Fisher p is "
                 "bounded; report as descriptive dissociation, not as a powered test"),
    }

    print(f"{'cell':20s} {'listA4':>7s} {'listA5':>7s} {'p':>8s} | "
          f"{'tkA4':>6s} {'tkA5':>6s} {'p':>8s}")
    for cell, e in res.items():
        L, T = e["is_list"], e["trunc_kill"]
        print(f"{cell:20s} {L['A4_rate']:6.1f}% {L['A5_rate']:6.1f}% "
              f"{L['mcnemar']['p_two_sided_exact']:8.5f} | "
              f"{T['A4_rate']:5.1f}% {T['A5_rate']:5.1f}% "
              f"{T['mcnemar']['p_two_sided_exact']:8.5f}")

    print(f"\n{'cell':20s} {'full d':>8s} {'inv':>4s} | {'no-trunc-kill n':>15s} "
          f"{'A4':>6s} {'A5':>6s} {'d':>8s} {'inv':>4s}")
    for cell, e in res.items():
        f_, k = e["canonical_full"], e["canonical_no_trunc_kill"]
        if k.get("status"):
            print(f"{cell:20s} {f_['delta_pp']:+8.2f} {'YES' if f_['point_inverted'] else 'no':>4s} | {k['status']}")
            continue
        print(f"{cell:20s} {f_['delta_pp']:+8.2f} {'YES' if f_['point_inverted'] else 'no':>4s} | "
              f"{k['n']:15d} {k['A4']:6.1f} {k['A5']:6.1f} {k['delta_pp']:+8.2f} "
              f"{'YES' if k['point_inverted'] else 'no':>4s}")

    print(f"\ndissociation flip x high-list: {diss['table_flip_x_highlist']} "
          f"Fisher p={diss['fisher_exact_p_two_sided']}")
    print(f"  flipped: {diss['cells_flipped']}")
    print(f"  high-list: {diss['cells_high_list']}")
    if errs:
        print("\nERRORS:", *errs, sep="\n  ")

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "a02_format_mechanism.json"
    json.dump({
        "generated_by": "analyze_a02_format_mechanism.py",
        "prereg": "A02_BABILONG_MISORDER_PREREG.md (1.2 mechanism, F3)",
        "gpu_spent": "ZERO",
        "scorer": "babilong.metrics.compare_answers + preprocess_output, IMPORTED",
        "pairing": ("idx = shard + row*8 (dvr convention), VERIFIED by (question,target) "
                    "equality across arms per index; cell refused on mismatch"),
        "post_treatment_warning": (
            "canonical_no_trunc_kill conditions on a variable produced BY the arm "
            "(output format) -- collider risk, DIAGNOSTIC not causal"),
        "per_cell": res, "dissociation": diss, "errors": errs,
    }, open(dst, "w"), indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        W / "proposal/backlog/A02-comem-write-read-repair/evidence/babilong_misorder"))
    main(Path(ap.parse_args().out))
