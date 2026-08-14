#!/usr/bin/env python
"""B02 Stage-0 analyzer — per-example oracle headroom with a pre-registered null floor.

Implements FIXED_SAMPLE_PROTOCOL.md sections 4 (fail-closed integrity assertions)
and 5 (estimator + null floor + decision rule).

The load-bearing point: with C configs, `oracle = mean_i max_j correct(i,j)` is
upward-biased by construction. Even with ZERO exploitable per-item structure,
independent columns with marginals p_j give E[oracle] = 1 - prod_j (1 - p_j),
which can sit far above best_fixed. So the gate statistic is the EXCESS over a
null, not the raw oracle minus best-fixed.

Two nulls:
  A (column-margins only): independently permute each config's correctness column.
     Destroys item x config coupling, preserves every p_j.
  B (both margins, PRIMARY): curveball / swap randomisation of the binary
     item x config matrix, preserving BOTH item margins and config margins.
     Primary because null A would credit a router for the trivially predictable
     fact that easy items are easy under every j.

Usage:
  python analyze_b02_oracle.py --cells <dir_glob> ... --length 16k --out evidence.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------- integrity

class IntegrityError(RuntimeError):
    """Fail-closed: never repaired by dropping items."""


def load_cell(cell_dir: Path, task: str, length: str, expect_n: int) -> dict:
    """Load one (resume_j, length) cell, merging shards if present.

    Asserts protocol section 4 items 1-4 and 6-7. Returns a dict with the
    per-item correctness keyed by sample_index plus the identity axis.
    """
    pat = str(cell_dir / f"{task}_{length}*.records.json")
    files = sorted(glob.glob(pat))
    if not files:
        raise IntegrityError(f"no records.json under {pat}")

    # --- assertion 1: shard index SET == {0..S-1} exactly (a set, not a count) ---
    shard_ids: list[int] = []
    n_shards_declared: set[int] = set()
    recs: dict[int, dict] = {}
    meta = None
    dup = 0
    for f in files:
        with open(f) as fh:
            blob = json.load(fh)
        sh = blob.get("sharding", {}) or {}
        n_shards_declared.add(int(sh.get("num_shards", 1)))
        shard_ids.append(int(sh.get("shard_index", 0)))
        if meta is None:
            meta = blob
        for r in blob["records"]:
            si = int(r["sample_index"])
            if si in recs:
                dup += 1
            recs[si] = r
    if len(n_shards_declared) != 1:
        raise IntegrityError(f"{cell_dir}: inconsistent num_shards {n_shards_declared}")
    S = n_shards_declared.pop()
    if set(shard_ids) != set(range(S)):
        raise IntegrityError(
            f"{cell_dir}: shard index SET {sorted(set(shard_ids))} != {list(range(S))} "
            f"(silent partial merge would corrupt the口径)")
    if len(shard_ids) != S:
        raise IntegrityError(f"{cell_dir}: {len(shard_ids)} shard files for num_shards={S}")

    # --- assertion 3: zero duplicates ---
    if dup:
        raise IntegrityError(f"{cell_dir}: {dup} duplicate sample_index")
    # --- assertion 2: exact item count ---
    if len(recs) != expect_n:
        raise IntegrityError(f"{cell_dir}: {len(recs)} items != expected {expect_n}")

    # --- assertion 4: zero NaN, recall in [0,1] ---
    for si, r in recs.items():
        v = float(r["recall"])
        if not np.isfinite(v) or not (0.0 - 1e-9 <= v <= 1.0 + 1e-9):
            raise IntegrityError(f"{cell_dir}: sample {si} recall={v} not finite in [0,1]")

    # --- assertion 6:口径. NOTE the direction: `is not False`, never `is not True`. ---
    cfg_files = sorted(glob.glob(str(cell_dir / f"{task}_{length}*.json")))
    cfg_files = [f for f in cfg_files if not f.endswith(".records.json")]
    if not cfg_files:
        raise IntegrityError(f"{cell_dir}: no cell config json")
    with open(cfg_files[0]) as fh:
        cfg = json.load(fh)
    chat = cfg.get("chat_template")
    if chat is not False:
        raise IntegrityError(
            f"{cell_dir}: chat_template={chat!r}, protocol requires False")
    if cfg.get("enable_thinking") is not False:
        raise IntegrityError(f"{cell_dir}: enable_thinking={cfg.get('enable_thinking')!r}")
    q = cfg.get("qcmem", {})
    if q.get("selector") != "iter_bm25":
        raise IntegrityError(f"{cell_dir}: selector={q.get('selector')!r} != iter_bm25")
    if q.get("lora_adapter") is not None:
        raise IntegrityError(f"{cell_dir}: lora_adapter={q.get('lora_adapter')!r} != None")
    if cfg.get("status") != "completed" or cfg.get("oom_count", 0) != 0:
        raise IntegrityError(
            f"{cell_dir}: status={cfg.get('status')} oom_count={cfg.get('oom_count')}")

    return {
        "dir": str(cell_dir),
        "resume_j": int(q.get("resume_j")),
        "n": len(recs),
        "num_shards": S,
        "n_dup": dup,
        "chat_template": chat,
        "enable_thinking": cfg.get("enable_thinking"),
        "selector": q.get("selector"),
        "topk": q.get("topk"),
        "chunk_size": q.get("chunk_size"),
        "seed": cfg.get("runtime", {}).get("seed"),
        "model_path": cfg.get("model", {}).get("model_path"),
        "num_hidden_layers": cfg.get("model", {}).get("num_hidden_layers"),
        "score": cfg.get("score"),
        "sha": {si: r["input_ids_sha256"] for si, r in recs.items()},
        "correct": {si: int(r["correct"]) for si, r in recs.items()},
        "recall": {si: float(r["recall"]) for si, r in recs.items()},
    }


def assert_pairing(cells: list[dict]) -> list[int]:
    """Protocol assertion 5 + 7. Returns the common sample_index list.

    Byte-identity across arms is THE assertion whose absence produced the T21
    defect. A mismatch aborts; it is never repaired by dropping items.
    """
    idx_sets = [set(c["sha"]) for c in cells]
    common = set.intersection(*idx_sets)
    union = set.union(*idx_sets)
    if common != union:
        raise IntegrityError(
            f"arms do not cover the same sample_index set: "
            f"|common|={len(common)} |union|={len(union)}")
    mism = []
    ref = cells[0]
    for c in cells[1:]:
        for si in sorted(common):
            if c["sha"][si] != ref["sha"][si]:
                mism.append((si, ref["resume_j"], c["resume_j"]))
    if mism:
        raise IntegrityError(
            f"input_ids_sha256 MISMATCH on {len(mism)} (item,arm) pairs, "
            f"first={mism[:3]} -- arms are NOT paired (this is the T21 defect)")
    # identity axis: only resume_j may differ
    for key in ("seed", "topk", "chunk_size", "model_path", "num_hidden_layers",
                "selector", "chat_template", "enable_thinking", "n"):
        vals = {c[key] for c in cells}
        if len(vals) != 1:
            raise IntegrityError(f"identity axis '{key}' differs across arms: {vals}")
    js = [c["resume_j"] for c in cells]
    if len(set(js)) != len(js):
        raise IntegrityError(f"duplicate resume_j across arms: {js}")
    return sorted(common)


# ----------------------------------------------------------------- estimator

def curveball(M: np.ndarray, rng: np.random.Generator, n_swap: int) -> np.ndarray:
    """Both-margins-preserving randomisation of a binary matrix (null B).

    Repeatedly pick two rows and swap a random subset of the positions where
    they differ, which preserves every row sum and every column sum.
    """
    A = M.copy()
    n_rows = A.shape[0]
    if n_rows < 2:
        return A
    for _ in range(n_swap):
        i, k = rng.integers(0, n_rows, size=2)
        if i == k:
            continue
        a, b = A[i], A[k]
        # positions where exactly one of the two rows is 1
        only_a = np.flatnonzero((a == 1) & (b == 0))
        only_b = np.flatnonzero((a == 0) & (b == 1))
        m = min(len(only_a), len(only_b))
        if m == 0:
            continue
        t = int(rng.integers(1, m + 1))
        pick_a = rng.choice(only_a, size=t, replace=False)
        pick_b = rng.choice(only_b, size=t, replace=False)
        a[pick_a] = 0
        b[pick_a] = 1
        a[pick_b] = 1
        b[pick_b] = 0
    return A


def analyze(M: np.ndarray, n_perm: int, seed: int, scale: str = "binary") -> dict:
    """M: (n_items, n_configs) correctness -- binary {0,1} or fractional recall.

    `scale` only labels the output; the arithmetic is identical.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    n_items, n_cfg = M.shape
    p = M.mean(axis=0)
    oracle_obs = float(M.max(axis=1).mean())
    best_fixed = float(p.max())
    raw_headroom = oracle_obs - best_fixed
    # analytic independence expectation -- exact for binary, indicative otherwise
    e_indep = float(1.0 - np.prod(1.0 - p)) if scale == "binary" else None

    # null A (PRIMARY as of protocol v0.2): permute each column independently.
    # Column margins preserved, item x config coupling destroyed. Does NOT preserve
    # row margins -- which is exactly why it can test the oracle at all.
    nullA = np.empty(n_perm)
    for b in range(n_perm):
        P = np.empty_like(M)
        for j in range(n_cfg):
            P[:, j] = rng.permutation(M[:, j])
        nullA[b] = P.max(axis=1).mean()

    # null B: both margins preserved. RETIRED for the binary oracle -- it is
    # PROVABLY INVARIANT there (max_j M[i,j] = 1[rowsum_i>=1] is a function of the
    # row margins alone, which curveball preserves exactly). Still computed, but
    # only so the degeneracy is measured and reported rather than assumed away.
    nullB = None
    if scale == "binary":
        n_swap = max(50, 5 * n_items)
        nullB = np.empty(n_perm)
        cur = M.copy()
        for b in range(n_perm):
            cur = curveball(cur, rng, n_swap)
            nullB[b] = cur.max(axis=1).mean()

    def summarize(null, name, primary):
        d = oracle_obs - float(null.mean())
        lo, hi = np.percentile(oracle_obs - null, [2.5, 97.5])
        ge = float(np.mean(null >= oracle_obs))
        le = float(np.mean(null <= oracle_obs))
        pval = float(min(1.0, 2.0 * min(ge, le)))
        sd = float(null.std(ddof=1))
        out = {
            "null": name,
            "primary": primary,
            "null_mean": float(null.mean()),
            "null_sd": sd,
            "delta_excess": float(d),
            "ci95_excess": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "p_two_sided": pval,
        }
        # Degeneracy guard: a null with ~zero variance is not a test. Refusing to
        # read a verdict off it is the whole point of protocol v0.2 section 5b.
        if sd < 1e-12:
            out["DEGENERATE"] = True
            out["degenerate_reason"] = (
                "null sd is numerically zero: the statistic is invariant under this "
                "randomisation, so it cannot test anything. For the binary oracle, "
                "max_j M[i,j] = 1[rowsum_i>=1] depends on row margins alone and "
                "curveball preserves them exactly. NOT a verdict.")
        return out

    A = summarize(nullA, "A_column_margins", primary=True)
    res_nulls = {"null_A_PRIMARY": A}
    if nullB is not None:
        B = summarize(nullB, "B_both_margins_RETIRED_v0.2", primary=False)
        res_nulls["null_B_RETIRED"] = B

    # verdict reads ONLY off the primary, non-degenerate null
    if A.get("DEGENERATE"):
        verdict = "UNDECIDABLE: primary null is degenerate; do not read a verdict"
    elif not A["excludes_zero"]:
        verdict = ("KILL_CLAUSE_FIRES: no per-item interaction beyond chance; "
                   "the oracle is a max-over-noise artefact")
    elif A["delta_excess"] > 0:
        verdict = "PROCEED: configs are complementary per item; real router headroom"
    else:
        verdict = ("CLOSE: configs positively coupled (hard items hard everywhere); "
                   "worse than no-signal")

    out = {
        "outcome_scale": scale,
        "n_items": int(n_items),
        "n_configs": int(n_cfg),
        "per_config_marginal": [float(x) for x in p],
        "oracle_obs": oracle_obs,
        "best_fixed": best_fixed,
        "raw_headroom_REPORTED_NOT_GATE": raw_headroom,
        "E_oracle_under_independence_closed_form": e_indep,
        "regret_vs_best_fixed": raw_headroom,
        "verdict": verdict,
        "n_perm": int(n_perm),
        "perm_seed": int(seed),
    }
    out.update(res_nulls)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True,
                    help="cell directories containing *.records.json")
    ap.add_argument("--task", default="variable_tracking")
    ap.add_argument("--length", required=True)
    ap.add_argument("--expect_n", type=int, required=True)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--scale", choices=["binary", "fractional", "both"],
                    default="both",
                    help="binary = RULER correct (recall==1.0); fractional = recall. "
                         "Protocol v0.2 makes FRACTIONAL the confirmatory scale.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cells = [load_cell(Path(d), a.task, a.length, a.expect_n) for d in a.cells]
    cells.sort(key=lambda c: c["resume_j"])
    common = assert_pairing(cells)

    Mb = np.array([[c["correct"][si] for c in cells] for si in common], dtype=np.int8)
    Mf = np.array([[c["recall"][si] for c in cells] for si in common], dtype=float)

    res: dict = {}
    if a.scale in ("binary", "both"):
        res["binary"] = analyze(Mb, a.n_perm, a.seed, scale="binary")
    if a.scale in ("fractional", "both"):
        res["fractional_CONFIRMATORY"] = analyze(Mf, a.n_perm, a.seed, scale="fractional")
    res["protocol_version"] = "0.2"
    res["primary_null"] = "A_column_margins (null B retired: provably invariant, see 5b)"
    res["task"] = a.task
    res["length"] = a.length
    res["resume_j"] = [c["resume_j"] for c in cells]
    res["cell_scores_pct"] = [c["score"] for c in cells]
    res["integrity"] = {
        "shard_sets_complete": True,
        "item_counts_exact": True,
        "n_duplicates": 0,
        "n_nan": 0,
        "cross_arm_input_ids_sha256_identical": True,
        "chat_template": cells[0]["chat_template"],
        "enable_thinking": cells[0]["enable_thinking"],
        "selector": cells[0]["selector"],
        "n_paired_items": len(common),
    }
    # bootstrap determinism provenance: three numpy versions live on this cluster,
    # so pin the node and record it (same-seed multinomial diverges across them).
    res["provenance"] = {
        "node": socket.gethostname(),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "cells": [c["dir"] for c in cells],
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
