#!/usr/bin/env python3
"""Same-model de-oracle control: does Retraction 7 survive without the model swap?

Retraction 7 compared a DreamOn-v0-7B NON-ORACLE arm against a
Dream-Coder-v0-Instruct-7B ORACLE arm, changing TWO variables at once (model AND
length provisioning). This script adds the one-variable control: the SAME
Dream-Coder model run with a FIXED canvas (8 or 12 mask tokens per hole) instead
of the oracle per-hole gold token counts. The fixed-canvas specs differ from
`data/kspan/kspan_spec_v1.jsonl` in exactly `hole_token_lengths` (and the
derived `total_masked_tokens`); holes, segments and gold are byte-identical.

Reports, on the BALANCED subset (tasks present at every k):
  * per-arm logistic slope of `passed` on k, with BOTH a naive Wald SE and a
    CLUSTER-ROBUST (sandwich, clustered by task_id) SE, because each task
    contributes up to 4 correlated observations;
  * the raw k=1 -> k=4 within-task drop;
  * output-sanity axes (parseable / EM / truncation / abort) so a fixed-canvas
    arm that fell out of infilling mode cannot be reported as a valid control.

Cluster-robust machinery is imported from stats_kspan_interaction.py -- not
reimplemented.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_kspan_interaction import fit_logistic, task_of, two_sided  # noqa: E402

KS = (1, 2, 3, 4)

ARMS = {
    "diffusion_oracle": "runs/kspan_diffusion",
    "diffusion_fix8": "runs/kspan_diffusion_fix8",
    "diffusion_fix12": "runs/kspan_diffusion_fix12",
    "dreamon_nonoracle": "runs/kspan_diffusion_nonoracle",
    "ar_fim": "runs/kspan_ar_fim",
    "ar_fim_fair": "runs/kspan_ar_fim_fair",
}


def load(run_dir: str) -> dict:
    return json.load((Path(run_dir) / "score.json").open())


def slope(rows: list[dict]) -> dict:
    """logistic passed ~ 1 + k ; naive + cluster-robust(by task) SE on b_k."""
    X = [[1.0, float(r["k"])] for r in rows]
    y = [int(r["passed"]) for r in rows]
    cl = [task_of(r["spec_id"]) for r in rows]
    rob = fit_logistic(X, y, cl)
    # naive SE = each row its own cluster (sandwich collapses to inverse-Fisher
    # up to the finite-cluster scale factor); fit again with unique clusters.
    nai = fit_logistic(X, y, [f"row{i}" for i in range(len(y))])
    if rob is None or nai is None:
        return {}
    b, se_r = rob
    _, se_n = nai
    bk = b[1]
    return {
        "beta_k": bk,
        "se_robust": se_r[1], "z_robust": bk / se_r[1],
        "p_robust": two_sided(bk / se_r[1]),
        "se_naive": se_n[1], "z_naive": bk / se_n[1],
        "p_naive": two_sided(bk / se_n[1]),
        "n_obs": len(y), "n_clusters": len(set(cl)),
    }


def cells(rows, key="passed"):
    out = {}
    for k in KS:
        v = [r for r in rows if r["k"] == k]
        out[k] = (sum(bool(r[key]) for r in v) / len(v), len(v)) if v else (float("nan"), 0)
    return out


def main() -> None:
    scored = {}
    for name, d in ARMS.items():
        p = Path(d) / "score.json"
        if not p.exists():
            print(f"MISSING: {p}")
            continue
        scored[name] = load(d)

    # ---- balanced subset: tasks present at EVERY k, defined on the oracle arm
    common = None
    for k in KS:
        ids = {task_of(r["spec_id"]) for r in scored["diffusion_oracle"]["rows"] if r["k"] == k}
        common = ids if common is None else (common & ids)
    common = frozenset(common)
    print("=" * 92)
    print(f"BALANCED SUBSET: {len(common)} tasks present at every k "
          f"(defined on runs/kspan_diffusion)")
    print("=" * 92)

    # sanity: the fixed-canvas arms should already BE that subset
    for name in ("diffusion_fix8", "diffusion_fix12"):
        if name not in scored:
            continue
        ids = {task_of(r["spec_id"]) for r in scored[name]["rows"]}
        print(f"  {name}: {len(ids)} tasks, identical to balanced subset: {ids == common}")

    print()
    print("PROVENANCE (grading axis must match across arms)")
    print(f"  {'arm':<20} {'which':>6} {'rows':>6} {'shards':>7}  spec_sha256[:16]")
    for name, s in scored.items():
        print(f"  {name:<20} {s.get('which','?'):>6} {s.get('n_solutions',0):>6} "
              f"{s.get('n_solution_shards','?'):>7}  "
              f"{(s.get('spec_sha256') or 'not-recorded')[:16]}")

    print()
    print("=" * 92)
    print("BALANCED LADDER  pass@1 (n)")
    print("=" * 92)
    print("  " + "arm".ljust(20) + "".join(f"{'k='+str(k):>16}" for k in KS) + f"{'k1-k4 drop':>13}")
    bal = {}
    for name, s in scored.items():
        rows = [r for r in s["rows"] if task_of(r["spec_id"]) in common and r["k"] in KS]
        bal[name] = rows
        c = cells(rows)
        line = "  " + name.ljust(20)
        for k in KS:
            line += f"   {c[k][0]:.3f} (n={c[k][1]:>3})"
        line += f"{c[1][0]-c[4][0]:>+13.3f}"
        print(line)

    print()
    print("=" * 92)
    print("BALANCED WITHIN-TASK SLOPE  logistic passed ~ k   (negative = degrades with k)")
    print("=" * 92)
    print(f"  {'arm':<20} {'n_obs':>6} {'ntask':>6} {'beta_k':>8} "
          f"{'SEclu':>7} {'zclu':>7} {'p_clu':>10}   {'SEnaive':>8} {'znaive':>7} {'p_naive':>10}")
    fits = {}
    for name, rows in bal.items():
        f = slope(rows)
        if not f:
            print(f"  {name:<20} fit failed")
            continue
        fits[name] = f
        print(f"  {name:<20} {f['n_obs']:>6} {f['n_clusters']:>6} {f['beta_k']:>+8.3f} "
              f"{f['se_robust']:>7.3f} {f['z_robust']:>+7.2f} {f['p_robust']:>10.3g}   "
              f"{f['se_naive']:>8.3f} {f['z_naive']:>+7.2f} {f['p_naive']:>10.3g}")

    print()
    print("=" * 92)
    print("k:arm INTERACTION vs each AR arm (balanced, SE clustered by task)")
    print("negative => AR degrades FASTER than the diffusion-side arm")
    print("=" * 92)
    for ar_name in ("ar_fim", "ar_fim_fair"):
        if ar_name not in bal:
            continue
        for d_name in ("diffusion_oracle", "diffusion_fix8", "diffusion_fix12",
                       "dreamon_nonoracle"):
            if d_name not in bal:
                continue
            X, y, cl = [], [], []
            for flag, rows in ((0.0, bal[d_name]), (1.0, bal[ar_name])):
                for r in rows:
                    t = task_of(r["spec_id"])
                    k = float(r["k"])
                    X.append([1.0, k, flag, k * flag])
                    y.append(int(r["passed"]))
                    cl.append(t)
            fit = fit_logistic(X, y, cl)
            if fit is None:
                print(f"  {d_name} vs {ar_name}: fit failed")
                continue
            b, se = fit
            z = b[3] / se[3]
            print(f"  {d_name:<20} vs {ar_name:<12} k:arm={b[3]:>+7.3f} "
                  f"SE {se[3]:.3f}  z {z:>+6.2f}  p {two_sided(z):>9.3g}  "
                  f"(nclust={len(set(cl))})")

    print()
    print("=" * 92)
    print("OUTPUT SANITY on the balanced subset (is the fixed-canvas arm still infilling?)")
    print("=" * 92)
    print(f"  {'arm':<20} " + "".join(f"{'k='+str(k):>10}" for k in KS) + "   axis")
    for key, label in (("parseable", "parseable"),
                       ("em_all_stripped", "EM-to-gold(strip)")):
        for name, rows in bal.items():
            c = cells(rows, key=key)
            print(f"  {name:<20} " + "".join(f"{c[k][0]:>10.3f}" for k in KS)
                  + f"   {label}")
        print()
    for name, rows in bal.items():
        tr = {k: sum(1 for r in rows if r["k"] == k and r["truncated_holes"] > 0) for k in KS}
        ab = {k: sum(1 for r in rows if r["k"] == k and r["aborted_holes"] > 0) for k in KS}
        er = {k: sum(1 for r in rows if r["k"] == k and r["error"]) for k in KS}
        print(f"  {name:<20} trunc={[tr[k] for k in KS]} abort={[ab[k] for k in KS]} "
              f"err={[er[k] for k in KS]}")

    print()
    print("=" * 92)
    print("COST (mean tokens_fed / forward_passes, balanced subset)")
    print("=" * 92)
    for name, rows in bal.items():
        line = f"  {name:<20}"
        for k in KS:
            v = [r for r in rows if r["k"] == k]
            line += (f"  k{k}: tok={sum(r['tokens_fed'] for r in v)/len(v):>6.0f} "
                     f"fwd={sum(r['forward_passes'] for r in v)/len(v):>4.1f}")
        print(line)


if __name__ == "__main__":
    main()
