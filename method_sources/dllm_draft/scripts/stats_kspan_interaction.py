#!/usr/bin/env python3
"""Pre-registered gate arbitration: the pooled and nested interactions disagree.

The pooled 4-cell interaction (+0.297, z=3.10) and the nested within-task
interaction (+0.153, z=1.57) tell different stories. They differ because the
k-cells are NESTED task sets: only tasks with >=4 available non-adjacent body
lines appear at k=4, and those are systematically longer functions. So a
cell-to-cell slope mixes "more holes" with "which tasks survived".

This script settles it with the analysis that respects the design:
  * logistic  passed ~ k + arm + k:arm  on the COMMON task set, with
    CLUSTER-ROBUST (by task_id) standard errors, because each task contributes
    up to 4 correlated observations per arm;
  * the same model on the full (unbalanced) data, for contrast;
  * a decomposition showing how much of the pooled interaction is composition.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load(d: str) -> list[dict]:
    return json.load((Path(d) / "score.json").open())["rows"]


def phi(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
def two_sided(z): return 2.0 * (1.0 - phi(abs(z)))


def fit_logistic(X: list[list[float]], y: list[int], clusters: list[str],
                 ridge: float = 1e-6):
    """Logistic MLE + cluster-robust (sandwich) covariance."""
    p = len(X[0])
    b = [0.0] * p
    for _ in range(300):
        g = [0.0] * p
        H = [[0.0] * p for _ in range(p)]
        for xi, yi in zip(X, y):
            eta = sum(bj * xj for bj, xj in zip(b, xi))
            pr = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, eta))))
            r, w = yi - pr, pr * (1.0 - pr)
            for a in range(p):
                g[a] += r * xi[a]
                for c in range(p):
                    H[a][c] += w * xi[a] * xi[c]
        for a in range(p):
            H[a][a] += ridge
        step = solve(H, g)
        if step is None:
            return None
        b = [bj + sj for bj, sj in zip(b, step)]
        if max(abs(s) for s in step) < 1e-10:
            break

    # bread
    H = [[0.0] * p for _ in range(p)]
    for xi in X:
        eta = sum(bj * xj for bj, xj in zip(b, xi))
        pr = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, eta))))
        w = pr * (1.0 - pr)
        for a in range(p):
            for c in range(p):
                H[a][c] += w * xi[a] * xi[c]
    for a in range(p):
        H[a][a] += ridge
    Hinv = inv(H)
    if Hinv is None:
        return None

    # meat: sum over clusters of (sum_i score_i)(sum_i score_i)'
    per: dict[str, list[float]] = defaultdict(lambda: [0.0] * p)
    for xi, yi, ci in zip(X, y, clusters):
        eta = sum(bj * xj for bj, xj in zip(b, xi))
        pr = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, eta))))
        r = yi - pr
        for a in range(p):
            per[ci][a] += r * xi[a]
    M = [[0.0] * p for _ in range(p)]
    for s in per.values():
        for a in range(p):
            for c in range(p):
                M[a][c] += s[a] * s[c]
    nc = len(per)
    scale = nc / max(1, nc - 1)
    V = matmul(matmul(Hinv, M), Hinv)
    se = [math.sqrt(max(1e-300, V[a][a] * scale)) for a in range(p)]
    return b, se


def solve(A, bvec):
    n = len(A)
    M = [row[:] + [bvec[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-14:
            return None
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        for j in range(c, n + 1):
            M[c][j] /= d
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                for j in range(c, n + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][n] for i in range(n)]


def inv(A):
    n = len(A)
    out = []
    for i in range(n):
        e = [1.0 if j == i else 0.0 for j in range(n)]
        col = solve(A, e)
        if col is None:
            return None
        out.append(col)
    return [[out[j][i] for j in range(n)] for i in range(n)]


def matmul(A, B):
    n, m, p = len(A), len(B[0]), len(B)
    return [[sum(A[i][k] * B[k][j] for k in range(p)) for j in range(m)]
            for i in range(n)]


def task_of(spec_id: str) -> str:
    return spec_id.rsplit("/", 1)[0]


def report(diff_rows, ar_rows, label, ks, restrict=None):
    X, y, cl = [], [], []
    for arm_flag, rows in ((0.0, diff_rows), (1.0, ar_rows)):
        for r in rows:
            if r["k"] not in ks:
                continue
            t = task_of(r["spec_id"])
            if restrict is not None and t not in restrict:
                continue
            k = float(r["k"])
            X.append([1.0, k, arm_flag, k * arm_flag])
            y.append(int(r["passed"]))
            cl.append(t)
    fit = fit_logistic(X, y, cl)
    if fit is None:
        print(f"  {label}: fit failed")
        return
    b, se = fit
    names = ["intercept", "k", "arm(AR)", "k:arm  <-- INTERACTION"]
    ntask = len(set(cl))
    print(f"  {label}   n_obs={len(y)}  n_clusters={ntask}")
    for nm, bb, ss in zip(names, b, se):
        z = bb / ss if ss else float("nan")
        print(f"     {nm:<26} {bb:>+8.3f}  SE {ss:>6.3f}  z {z:>+6.2f}  "
              f"p {two_sided(z):>9.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion", required=True)
    ap.add_argument("--ar", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--ks", default="1,2,3,4")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    d, a = load(args.diffusion), load(args.ar)
    common = None
    for k in ks:
        ids = {task_of(r["spec_id"]) for r in d if r["k"] == k}
        common = ids if common is None else (common & ids)

    print("=" * 78)
    print(f"CLUSTER-ROBUST INTERACTION  [{args.label}]")
    print("=" * 78)
    print("logistic passed ~ k + arm + k:arm ; SE clustered by task_id")
    print("positive k:arm => AR's slope in k is HIGHER than diffusion's;")
    print("negative k:arm => AR degrades FASTER than diffusion (the claim).")
    print()
    report(d, a, "FULL (unbalanced, nested cells)", ks)
    print()
    report(d, a, "COMMON tasks only (balanced)    ", ks, restrict=common)

    # ---- composition decomposition
    print()
    print("COMPOSITION CHECK: pass@1 at k=1 only, split by how many holes the")
    print("task can support (i.e. by whether it appears at k=4)")
    for nm, rows in (("diffusion", d), ("ar_fim", a)):
        v1 = [r for r in rows if r["k"] == 1]
        inc = [r for r in v1 if task_of(r["spec_id"]) in common]
        exc = [r for r in v1 if task_of(r["spec_id"]) not in common]
        f = lambda v: sum(r["passed"] for r in v) / len(v) if v else float("nan")
        print(f"  {nm:<10} k=1 on k4-capable tasks: {f(inc):.3f} (n={len(inc)})   "
              f"k=1 on short tasks: {f(exc):.3f} (n={len(exc)})")


if __name__ == "__main__":
    main()
