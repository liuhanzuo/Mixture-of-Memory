#!/usr/bin/env python
"""A02 — the ONE-OPERATION ablation of babilong.metrics: remove first-period truncation.

Zero GPU. Re-scores the SAME on-disk generations; no conditioning of any kind, so this
is free of the collider risk that makes `canonical_no_trunc_kill` diagnostic-only.

MOTIVATION. `diagnose_a02_babilong_format.py` used `target_in_raw` (target substring
anywhere in the raw output) as a format-insensitive read. That is LENIENT and, worse,
it is **inflated by chance for multiple-choice outputs**: a model that lists 3 of the 6
bAbI rooms has a ~50 % chance of containing the target incidentally. Since A4 emits the
list format far more often than A5, `target_in_raw` could favour A4 for a spurious
reason. This script removes that objection.

THE ABLATION. `compare_answers` = (a) `preprocess_output` truncates at the FIRST period,
then (b) the target must be the ONLY task label present. Variant scored here:

    score_notrunc  = step (b) applied to the FULL output, WITHOUT step (a)

This keeps the uniqueness requirement, so a multiple-choice list mentioning >=2 rooms
still scores 0 -- chance inflation is removed by construction. It differs from the
canonical metric in exactly ONE operation, so any ordering difference is attributable
to that operation.

Also reported, as a second independent guard:
    non_list_only  = the canonical metric restricted to outputs that are NOT a
                     multiple-choice enumeration (post-treatment, so diagnostic only)
    n_labels_in_raw= mean number of distinct task labels in the raw output, which is
                     the quantity that drives chance inflation

Usage: python analyze_a02_truncation_ablation.py [--out <dir>]
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
    "A2": "a02_rtax_babilong_A2_j6",
    "A3": "a02_rtax_babilong_A3_j9",
    "A4": "a02_babilong_c2_j12_readlora",
    "A5": "a02_rtax_babilong_A5_j18",
}
LADDER = [("A0", 0), ("A2", 6), ("A3", 9), ("A4", 12), ("A5", 18)]
NSHARD, N_BOOT, SEED = 8, 5000, 42


def _lb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def mcnemar_exact(a, b):
    bb = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    cc = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = bb + cc
    if n == 0:
        return bb, cc, 0, 1.0
    obs = abs(bb - n / 2.0)
    return bb, cc, n, float(min(1.0, sum(
        math.exp(_lb(n, k) - n * math.log(2.0))
        for k in range(n + 1) if abs(k - n / 2.0) >= obs - 1e-12)))


def paired_boot(a, b):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    bo = 100.0 * (a[idx].mean(axis=1) - b[idx].mean(axis=1))
    lo, hi = np.percentile(bo, [2.5, 97.5])
    return float(100.0 * (a.mean() - b.mean())), float(lo), float(hi)


def _rank(a):
    import numpy as np
    a = np.asarray(a, float)
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[o] = np.arange(1, len(a) + 1, dtype=float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            m = inv == i
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    import numpy as np
    rx, ry = _rank(x) - _rank(x).mean(), _rank(y) - _rank(y).mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return 0.0 if den == 0 else float((rx * ry).sum() / den)


def spearman_perm(x, y):
    import itertools
    rho = spearman(x, y)
    tot = cnt = 0
    for p in itertools.permutations(range(len(x))):
        tot += 1
        if abs(spearman(x, [y[i] for i in p])) >= abs(rho) - 1e-12:
            cnt += 1
    return rho, cnt / tot


def is_list_format(raw):
    low = raw.strip().lower()
    if low.startswith(("choices", "options")):
        return True
    h = low[:60]
    return ("a." in h and "b." in h) or ("a)" in h and "b)" in h)


def score_notrunc(target, output, question, labels):
    """compare_answers with the first-period truncation REMOVED (one-operation ablation)."""
    out = output.lower()
    tgt = target.lower()
    labs = {l.lower() for l in labels}
    inq = {l for l in labs if l in question.lower()}
    ino = {l for l in labs if l in out} - inq
    if "," in tgt and len(tgt) > 3:
        subs = tgt.split(",")
        return all(t in ino for t in subs) and len(ino) == len(subs)
    return tgt in ino and len(ino) == 1


def load_indexed(sub, task, length):
    out = {}
    files = glob.glob(str(W / "babilong_results" / sub
                          / f"{task}_{length}_*shard*of{NSHARD}.csv"))
    if len(files) != NSHARD:
        return None, f"SHARD_INCOMPLETE {len(files)}/{NSHARD}"
    for f in files:
        s = int(Path(f).stem.split("shard")[1].split("of")[0])
        for r, row in enumerate(csv.DictReader(open(f))):
            out[s + r * NSHARD] = row
    return (out, None) if len(out) == 100 else (None, f"N={len(out)}")


def main(out_dir: Path):
    res, errs = {}, []
    for task in ("qa1", "qa2", "qa5"):
        labs = {l.lower() for l in TASK_LABELS[task]}
        for length in ("16k", "32k"):
            cell = f"babilong|{task}|{length}"
            per, bad = {}, False
            for arm, sub in ARMS.items():
                d, e = load_indexed(sub, task, length)
                if e:
                    errs.append(f"{cell}/{arm}: {e}")
                    bad = True
                else:
                    per[arm] = d
            if bad:
                continue
            idx = sorted(set.intersection(*[set(d) for d in per.values()]))
            if len(idx) != 100:
                errs.append(f"{cell}: common {len(idx)}")
                continue
            if [i for i in idx if len({(per[a][i]["question"].strip(),
                                        per[a][i]["target"].strip()) for a in per}) > 1]:
                errs.append(f"{cell}: G2 pairing failed")
                continue
            F = {a: {} for a in per}
            for a, d in per.items():
                for i in idx:
                    raw, q, tg = d[i]["output"], d[i]["question"], d[i]["target"]
                    inq = {l for l in labs if l in q.lower()}
                    F[a][i] = {
                        "canon": int(bool(compare_answers(tg, raw, q, TASK_LABELS[task]))),
                        "notrunc": int(bool(score_notrunc(tg, raw, q, TASK_LABELS[task]))),
                        "is_list": int(is_list_format(raw)),
                        "n_lab_raw": len({l for l in labs if l in raw.lower()} - inq),
                    }

            def v(a, k, sub=None):
                return [F[a][i][k] for i in (sub or idx)]

            e = {"n": 100, "acc": {}, "mean_n_labels_raw": {}, "listfmt": {}}
            for a in per:
                e["acc"][a] = {
                    "canonical": round(100 * sum(v(a, "canon")) / 100, 1),
                    "notrunc": round(100 * sum(v(a, "notrunc")) / 100, 1),
                }
                e["mean_n_labels_raw"][a] = round(sum(v(a, "n_lab_raw")) / 100, 2)
                e["listfmt"][a] = round(100 * sum(v(a, "is_list")) / 100, 1)

            for nm, key in (("canonical", "canon"), ("notrunc", "notrunc")):
                d_, lo, hi = paired_boot(v("A4", key), v("A5", key))
                b_, c_, nd, p = mcnemar_exact(v("A4", key), v("A5", key))
                e[f"a4_vs_a5_{nm}"] = {
                    "delta_pp": round(d_, 2), "ci95": [round(lo, 2), round(hi, 2)],
                    "mcnemar": {"b_A4only": b_, "c_A5only": c_, "n_discordant": nd,
                                "p_two_sided_exact": round(p, 6)},
                    "point_inverted": bool(sum(v("A5", key)) >= sum(v("A4", key))),
                    "sig": bool(lo > 0 or hi < 0),
                }
            # ladder rank test under each metric
            for nm, key in (("canonical", "canon"), ("notrunc", "notrunc")):
                accs = [100 * sum(v(a, key)) / 100 for a, _ in LADDER]
                rho, pp = spearman_perm([j for _, j in LADDER], accs)
                e[f"ladder_{nm}"] = {
                    "acc": [round(x, 1) for x in accs], "spearman_rho": round(rho, 4),
                    "p_exact_perm": round(pp, 6),
                    "recovers": bool(rho <= -0.9 and pp < 0.05)}
            # second guard: canonical restricted to non-list outputs (post-treatment)
            keep = [i for i in idx if not F["A4"][i]["is_list"] and not F["A5"][i]["is_list"]]
            if len(keep) >= 10:
                d_, lo, hi = paired_boot(v("A4", "canon", keep), v("A5", "canon", keep))
                e["a4_vs_a5_canonical_nonlist_only"] = {
                    "n": len(keep), "delta_pp": round(d_, 2),
                    "ci95": [round(lo, 2), round(hi, 2)],
                    "point_inverted": bool(sum(v("A5", "canon", keep))
                                           >= sum(v("A4", "canon", keep))),
                    "CAVEAT": "post-treatment conditioning -- diagnostic only"}
            else:
                e["a4_vs_a5_canonical_nonlist_only"] = {"n": len(keep), "status": "TOO_SMALL"}
            res[cell] = e

    print("=== chance-inflation guard: mean # distinct task labels in raw output ===")
    print(f"{'cell':20s}" + "".join(f"{a:>8s}" for a in ARMS))
    for c, e in res.items():
        print(f"{c:20s}" + "".join(f"{e['mean_n_labels_raw'][a]:8.2f}" for a in ARMS))

    print("\n=== A4 vs A5: canonical  vs  SAME METRIC minus first-period truncation ===")
    print(f"{'cell':20s} {'canon d':>9s} {'inv':>4s} {'p':>8s} | "
          f"{'notrunc d':>10s} {'inv':>4s} {'p':>8s} {'CI95':>18s}")
    flips = []
    for c, e in res.items():
        a, b = e["a4_vs_a5_canonical"], e["a4_vs_a5_notrunc"]
        if a["point_inverted"] and not b["point_inverted"]:
            flips.append(c)
        print(f"{c:20s} {a['delta_pp']:+9.2f} {'YES' if a['point_inverted'] else 'no':>4s} "
              f"{a['mcnemar']['p_two_sided_exact']:8.4f} | {b['delta_pp']:+10.2f} "
              f"{'YES' if b['point_inverted'] else 'no':>4s} "
              f"{b['mcnemar']['p_two_sided_exact']:8.4f} "
              f"[{b['ci95'][0]:+7.2f},{b['ci95'][1]:+7.2f}]")
    print(f"\ncells whose INVERSION IS REPAIRED by removing truncation alone: "
          f"{len(flips)}/6 {flips}")

    print("\n=== per-arm accuracy under both metrics (ladder) ===")
    for c, e in res.items():
        print(f"{c}:")
        print("   canonical:", {a: e["acc"][a]["canonical"] for a in ARMS},
              f"rho={e['ladder_canonical']['spearman_rho']:+.3f} "
              f"p={e['ladder_canonical']['p_exact_perm']:.4f} "
              f"recovers={e['ladder_canonical']['recovers']}")
        print("   notrunc  :", {a: e["acc"][a]["notrunc"] for a in ARMS},
              f"rho={e['ladder_notrunc']['spearman_rho']:+.3f} "
              f"p={e['ladder_notrunc']['p_exact_perm']:.4f} "
              f"recovers={e['ladder_notrunc']['recovers']}")

    print("\n=== guard 2: canonical restricted to non-list outputs (diagnostic) ===")
    for c, e in res.items():
        g = e["a4_vs_a5_canonical_nonlist_only"]
        if g.get("status"):
            print(f"{c:20s} {g['status']} n={g['n']}")
        else:
            print(f"{c:20s} n={g['n']:3d} d={g['delta_pp']:+7.2f} "
                  f"[{g['ci95'][0]:+7.2f},{g['ci95'][1]:+7.2f}] "
                  f"inv={'YES' if g['point_inverted'] else 'no'}")
    if errs:
        print("\nERRORS:", *errs, sep="\n  ")

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "a02_truncation_ablation.json"
    json.dump({
        "generated_by": "analyze_a02_truncation_ablation.py",
        "gpu_spent": "ZERO",
        "ablation": ("score_notrunc = babilong compare_answers WITHOUT preprocess_output's "
                     "first-period truncation; uniqueness requirement RETAINED so "
                     "multiple-choice lists still score 0 (removes chance inflation)"),
        "no_conditioning": ("the notrunc contrast uses ALL 100 items in every cell -- no "
                            "subsetting, hence no collider risk"),
        "per_cell": res, "inversion_repaired_by_removing_truncation": flips,
        "errors": errs,
    }, open(dst, "w"), indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        W / "proposal/backlog/A02-comem-write-read-repair/evidence/babilong_misorder"))
    main(Path(ap.parse_args().out))
