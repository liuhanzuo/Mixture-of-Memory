#!/usr/bin/env python3
"""Statistics for the k-span ladder: interaction test + the pre-registered gates.

Everything here operates on the per-task rows written by score_kspan.py, so the
unit of analysis is a (spec_id, arm) pair and paired/nested structure is
respected rather than assumed away.

Tests
-----
interaction
    delta(k_hi) - delta(k_lo) where delta = pass@1(diffusion) - pass@1(AR),
    with an unpooled binomial SE across the four independent cell proportions.
    This is the quantity the pilot reported as +.525 (SE .109).
slope_logistic
    Logistic regression of `passed` on k, per arm. Fit by Newton-Raphson (no
    scipy/sklearn dependency), reported with the Wald z and p.
    Run on: ALL rows, and on the EM-false subset (pre-registered gate 1).
nested
    Pre-registered gate 3: restrict to task_ids present at EVERY k, so the
    k-ladder is not confounded with "which tasks survive to k=4". This is a
    within-task comparison; also reported as a paired McNemar-style delta.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load(run_dir: str) -> list[dict]:
    p = Path(run_dir) / "score.json"
    return json.load(p.open())["rows"]


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p(z: float) -> float:
    return 2.0 * (1.0 - phi(abs(z)))


def logistic_slope(xs: list[float], ys: list[int], *, ridge: float = 1e-6):
    """Newton-Raphson logistic fit of y ~ b0 + b1*x. Returns (b1, se, z, p)."""
    b0, b1 = 0.0, 0.0
    n = len(xs)
    if n == 0 or len(set(ys)) < 2:
        return None
    for _ in range(200):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            eta = b0 + b1 * x
            pr = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, eta))))
            r = y - pr
            w = pr * (1.0 - pr)
            g0 += r; g1 += r * x
            h00 += w; h01 += w * x; h11 += w * x * x
        h00 += ridge; h11 += ridge
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-14:
            return None
        d0 = (h11 * g0 - h01 * g1) / det
        d1 = (h00 * g1 - h01 * g0) / det
        b0 += d0; b1 += d1
        if max(abs(d0), abs(d1)) < 1e-10:
            break
    # covariance = inverse Fisher information
    h00 = h01 = h11 = 0.0
    for x in xs:
        eta = b0 + b1 * x
        pr = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, eta))))
        w = pr * (1.0 - pr)
        h00 += w; h01 += w * x; h11 += w * x * x
    h00 += ridge; h11 += ridge
    det = h00 * h11 - h01 * h01
    if det <= 0:
        return None
    var_b1 = h00 / det
    if var_b1 <= 0:
        return None
    se = math.sqrt(var_b1)
    z = b1 / se
    return b1, se, z, two_sided_p(z)


def cell(rows: list[dict], k: int) -> tuple[int, int]:
    v = [r for r in rows if r["k"] == k]
    return sum(r["passed"] for r in v), len(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion", required=True)
    ap.add_argument("--ar", required=True)
    ap.add_argument("--ar-fair", default=None)
    ap.add_argument("--label", default="main")
    ap.add_argument("--ks", default="1,2,3,4")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    arms = {"diffusion": load(args.diffusion), "ar_fim": load(args.ar)}
    if args.ar_fair:
        arms["ar_fim_fair"] = load(args.ar_fair)

    print("=" * 78)
    print(f"LADDER  [{args.label}]")
    print("=" * 78)
    hdr = f"{'k':>3} {'n':>5} " + " ".join(f"{a:>13}" for a in arms)
    print(hdr)
    for k in ks:
        n = cell(arms["diffusion"], k)[1]
        cells = []
        for a, rows in arms.items():
            p, nn = cell(rows, k)
            cells.append(f"{p/nn:>7.3f}({nn:>3})" if nn else f"{'--':>13}")
        print(f"{k:>3} {n:>5} " + " ".join(f"{c:>13}" for c in cells))

    # ---------------------------------------------------------------- interaction
    print()
    print("INTERACTION  delta = pass@1(diffusion) - pass@1(AR-FIM)")
    lo, hi = ks[0], ks[-1]
    stats = {}
    for k in ks:
        pd, nd = cell(arms["diffusion"], k)
        pa, na = cell(arms["ar_fim"], k)
        d = pd / nd - pa / na
        vd = (pd / nd) * (1 - pd / nd) / nd + (pa / na) * (1 - pa / na) / na
        stats[k] = (d, vd)
        print(f"  k={k}  delta={d:+.3f}  (SE {math.sqrt(vd):.3f})")
    inter = stats[hi][0] - stats[lo][0]
    se = math.sqrt(stats[hi][1] + stats[lo][1])
    z = inter / se if se else float("nan")
    print(f"  interaction (delta@k{hi} - delta@k{lo}) = {inter:+.3f}  "
          f"SE {se:.3f}  z={z:.2f}  p={two_sided_p(z):.3g}")

    # ------------------------------------------------- pre-registered gates 1 & 2
    print()
    print("GATE 1/2  logistic slope of pass@1 on k")
    print(f"  {'arm':<14} {'subset':<16} {'n':>5} {'beta_k':>8} {'SE':>7} {'z':>7} {'p':>10}")
    for a, rows in arms.items():
        for label, sub in (("all", rows),
                           ("EM-false", [r for r in rows if not r["em_all_stripped"]]),
                           ("EM-true", [r for r in rows if r["em_all_stripped"]])):
            sub = [r for r in sub if r["k"] in ks]
            fit = logistic_slope([float(r["k"]) for r in sub],
                                 [int(r["passed"]) for r in sub])
            if fit is None:
                print(f"  {a:<14} {label:<16} {len(sub):>5} {'n/a (degenerate)':>34}")
                continue
            b, s, z2, p = fit
            print(f"  {a:<14} {label:<16} {len(sub):>5} {b:>+8.3f} {s:>7.3f} "
                  f"{z2:>+7.2f} {p:>10.3g}")

    # --------------------------------------------------- pre-registered gate 3
    print()
    print("GATE 3  nested subset: task_ids present at EVERY k")
    common = None
    for k in ks:
        ids = {r["spec_id"].rsplit("/", 1)[0] for r in arms["diffusion"] if r["k"] == k}
        common = ids if common is None else (common & ids)
    print(f"  |common tasks| = {len(common)}")
    for a, rows in arms.items():
        line = []
        for k in ks:
            v = [r for r in rows
                 if r["k"] == k and r["spec_id"].rsplit("/", 1)[0] in common]
            line.append(f"k{k}={sum(r['passed'] for r in v)/len(v):.3f}" if v else f"k{k}=--")
        sub = [r for r in rows
               if r["k"] in ks and r["spec_id"].rsplit("/", 1)[0] in common]
        fit = logistic_slope([float(r["k"]) for r in sub],
                             [int(r["passed"]) for r in sub])
        tail = ""
        if fit:
            b, s, z2, p = fit
            tail = f"   beta_k={b:+.3f} SE={s:.3f} z={z2:+.2f} p={p:.3g}"
        print(f"  {a:<14} " + "  ".join(line) + tail)

    # nested interaction, same tasks at both ends
    pd_hi = [r for r in arms["diffusion"]
             if r["k"] == hi and r["spec_id"].rsplit("/", 1)[0] in common]
    pd_lo = [r for r in arms["diffusion"]
             if r["k"] == lo and r["spec_id"].rsplit("/", 1)[0] in common]
    pa_hi = [r for r in arms["ar_fim"]
             if r["k"] == hi and r["spec_id"].rsplit("/", 1)[0] in common]
    pa_lo = [r for r in arms["ar_fim"]
             if r["k"] == lo and r["spec_id"].rsplit("/", 1)[0] in common]

    def rate(v): return sum(r["passed"] for r in v) / len(v)
    d_hi = rate(pd_hi) - rate(pa_hi)
    d_lo = rate(pd_lo) - rate(pa_lo)
    n = len(common)
    var = sum(rate(v) * (1 - rate(v)) / n for v in (pd_hi, pa_hi, pd_lo, pa_lo))
    se2 = math.sqrt(var)
    z3 = (d_hi - d_lo) / se2 if se2 else float("nan")
    print(f"  nested interaction = {d_hi - d_lo:+.3f}  SE {se2:.3f}  "
          f"z={z3:.2f}  p={two_sided_p(z3):.3g}   (delta@k{lo}={d_lo:+.3f}, delta@k{hi}={d_hi:+.3f})")

    print()
    print("EM-to-gold by cell (stripped)")
    for a, rows in arms.items():
        line = []
        for k in ks:
            v = [r for r in rows if r["k"] == k]
            line.append(f"k{k}={sum(r['em_all_stripped'] for r in v)/len(v):.3f}")
        print(f"  {a:<14} " + "  ".join(line))

    print()
    print("COST (mean over ALL tasks, incl. truncated/aborted)")
    print(f"  {'arm':<14} " + " ".join(f"{'k'+str(k):>22}" for k in ks))
    for a, rows in arms.items():
        cells = []
        for k in ks:
            v = [r for r in rows if r["k"] == k]
            tf = sum(r["tokens_fed"] for r in v) / len(v)
            at = sum(r["attended"] for r in v) / len(v)
            cells.append(f"{tf:>9.0f}/{at:>11.0f}")
        print(f"  {a:<14} " + " ".join(f"{c:>22}" for c in cells))
    print("  (format: tokens_fed / attended_context_sum)")

    print()
    print("TERMINATION (disclosed separately from failures)")
    for a, rows in arms.items():
        for k in ks:
            v = [r for r in rows if r["k"] == k]
            t = sum(1 for r in v if r["truncated_holes"] > 0)
            ab = sum(1 for r in v if r["aborted_holes"] > 0)
            er = sum(1 for r in v if r["error"])
            if t or ab or er:
                print(f"  {a:<14} k={k}  truncated_tasks={t}/{len(v)}  "
                      f"aborted={ab}  errors={er}")
    print("  (blank => zero truncation/abort/error in that arm)")


if __name__ == "__main__":
    main()
