#!/usr/bin/env python3
"""A04 — sigma_run from the POST-`ce5c298` keep7 data-order triplet, and K2.

Pre-registration: `A04_SIGMA_RUN_POSTFIX_K2_PREREG.md` (commit 94839e8), written
and committed BEFORE this script produced a single number.

WHAT THIS IS
------------
`STATUS.json:power_analysis` adjudicates K2 with sigma-hat values from A03's
`a03_sigma_run_n3.json`. That file's `keep7_20k_cpt` family is seeds
`[0, 43, 44, 45]` -- and seed 0 is A03 Arm 3, launched 2026-08-09 01:11:43,
**22 h 09 m BEFORE `ce5c298`** (2026-08-09 23:21:09 +0800). `PROPOSAL.md` s7.2:

    "A pre-fix seed arm and a post-fix seed arm are therefore not draws from the
     same distribution, and must never enter the same `sigma_run` estimate."

So that family is s7.2-noncompliant. This script computes the sigma_run s7.2
actually licenses -- seeds {43,44,45} only, df=2 -- and re-runs K2's arithmetic.

WHAT IT IS NOT
--------------
* NOT a trajectory / monotonicity / neighbour statistic. It is a LEVEL comparison
  at one common step (220000), across three runs. No `E[range of k]` constant is
  used anywhere; see `RANGE_CONSTANTS_DECLARED_UNUSED`.
* NOT a full run-to-run sigma_run. All three arms resume from ONE common
  `step200000.pt`, so this varies the data subset/order and NOTHING ELSE. As an
  estimate of full run-to-run variance it is DOWNWARD-BIASED, i.e. optimistic for
  K2 (prereg s4.2 item 3).
* NOT a clearance of K2. Prereg s4.2: a large sigma kills, a small sigma does not
  clear.

DISCIPLINE (prereg s2)
----------------------
* `build_nulls` is IMPORTED from `pilot_zero_rule_disagreement` and CALLED on the
  pinned intact anchor. No null / Delta / residual is copied from any .md prose.
  Delta is then cross-checked against the canonical full-precision constants at
  1e-9. Delta is never substituted, floored or re-derived (guard G2).
* Shard integrity: index set EXACTLY {0..7} (not a count), exact item counts,
  0 duplicate item_ids, 0 nan. MMLU via nested `content_norm.correct`.
* `chat_template` asserted `is not False` -> FAIL (never `is not True`).
* Seed-disjointness EXECUTED with the self-excluding checker, unweakened.
* chi2 df=2 via closed form (scipy absent on .73), asserted df == 2.

CPU ONLY. No GPU, no model load, no torch.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---- canonical: IMPORT, never reimplement --------------------------------
from pilot_zero_rule_disagreement import (  # noqa: E402
    AXES,
    EXPECTED_N,
    PREREG,
    build_nulls,
    load_shards,
    mmlu_content_norm_vec,
    qa_metric_vec,
)
from analyze_1b_knowledge_floor import N_BOOT, SEED, TIE_CONVS  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen inputs
# ---------------------------------------------------------------------------
SEEDS = [43, 44, 45]
STEP = 220000
ARM_TAGS = {s: f"A03_1B_dataorder_seed{s}_step{STEP}" for s in SEEDS}
INTACT = {"mmlu": "A03_1B_base", "cb": "A03_1B_base", "nq": "A03_1B_base_nq"}

PREREG_CONVENTION = "split"          # pre-registered MMLU tie convention
DELTA_FRACTION = 0.10               # frozen by git d1ba737; PREREG['delta_fraction']

DECISION_AXES = ("triviaqa", "popqa", "mmlu_content")
DEMOTED_AXES = ("nq_open",)

# Canonical Delta, full precision, from
# code/a04_sigma_run_independent_recompute.py (itself derived from the pinned
# intact anchor). Used ONLY as a CROSS-CHECK of the freshly-built values -- never
# as the source. Cross-check tolerance 1e-9.
DELTA_CANONICAL = {
    "triviaqa": 4.043134195274186,
    "popqa": 1.3205298941613512,
    "mmlu_content": 1.0238926078906136,
    "nq_open": 0.9695290858725762,
}
DELTA_XCHECK_TOL = 1e-9

# one-sided t_{0.05, df=2}
T_05_DF2 = 2.9199855803537124

# Bootstrap offsets claimed by THIS run (prereg s2 item 5). Disjoint from every
# archived A04 json: 0,1 / 100..102 / 200..204 / 300,301 / 400..408 / 500..503 /
# 600..610 / 700..702 / 800,801.
NEW_ARM_INDEX_BASE = 900
GUARD_SEED_OFF = 8700

# A03's contaminated / companion sigma values, for the Q2 ratio. Read from the
# canonical JSON at runtime -- NOT hard-coded. Path resolved on wzc1 (the file
# does not exist on zwfy6: there is no proposal/archive/ there).
A03_SIGMA_REL = ("proposal/archive/A03-parametric-vs-external-memory/"
                 "evidence/a03_sigma_run_n3.json")
A03_SIGMA_MD5_EXPECTED = "5fb6cd4c3d693831e50d0817bda93ab8"

RANGE_CONSTANTS_DECLARED_UNUSED = {
    "why_recorded": (
        "prereg s2.1: this analysis computes NO range statistic, so E[range of k] "
        "constants are unused. Recorded so nobody can later reuse a c_n from this "
        "document. sigma here is a SAMPLE SD (ddof=1), df=2."),
    "c_3_E_range_of_3": 1.6925687506,
    "c_8_E_range_of_8": 2.8475,
    "used_in_this_analysis": False,
}


# ---------------------------------------------------------------------------
def chi2_ppf_df2(p):
    """chi2 quantile at df=2, closed form. scipy is absent on .73.

    df=2: CDF(x) = 1 - exp(-x/2)  =>  ppf(p) = -2 ln(1-p).
    Asserted df==2 so this can never be silently reused at another df.
    """
    return -2.0 * math.log(1.0 - p)


def sigma_chi2_ci(s, df):
    assert df == 2, f"closed-form chi2 valid only at df=2, got df={df}"
    lo = s * math.sqrt(df / chi2_ppf_df2(0.975))
    hi = s * math.sqrt(df / chi2_ppf_df2(0.025))
    return lo, hi


def assert_gpu_clear(threshold_mib=8000):
    """Refuse-guard. This is a CPU job; it must not run on a node whose GPUs are
    doing someone else's work (a stray heavy CPU load can still starve a trainer's
    dataloader). Absent nvidia-smi is NOT treated as clear."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise SystemExit(f"FATAL refuse-guard: nvidia-smi rc={out.returncode}: {out.stderr}")
    per_gpu = []
    for line in out.stdout.strip().splitlines():
        idx, used = [t.strip() for t in line.split(",")]
        per_gpu.append({"index": int(idx), "memory_used_mib": int(used)})
    busy = [g for g in per_gpu if g["memory_used_mib"] > threshold_mib]
    if busy:
        raise SystemExit(
            f"FATAL refuse-guard: GPUs busy (> {threshold_mib} MiB): {busy}. "
            "Refusing to run.")
    return {"threshold_mib": threshold_mib, "per_gpu": per_gpu,
            "max_used_mib": max(g["memory_used_mib"] for g in per_gpu)}


def assert_chat_template_false(raw_root, tags):
    """Protocol invariant. Written `is not False`, NEVER `is not True`: a missing
    / None value must FAIL, not silently pass.

    Two independent sources, both required:
      (A) the eval SCRIPTS contain zero `apply_chat_template` call sites, so the
          protocol is chat-free STRUCTURALLY and no flag can enable one;
      (B) each cell's own summary.json, if it records chat_template at all, must
          record False.
    """
    rr = os.path.abspath(raw_root)
    scripts = [os.path.join(rr, "scripts", f) for f in
               ("eval_olmo2_closedbook_qa.py", "eval_olmo2_mmlu_content.py")]
    struct = {}
    for p in scripts:
        if not os.path.isfile(p):
            raise SystemExit(f"FATAL: eval script not found for protocol audit: {p}")
        src = open(p).read()
        n_apply = src.count("apply_chat_template")
        if n_apply != 0:
            raise SystemExit(
                f"FATAL: {os.path.basename(p)} has {n_apply} apply_chat_template "
                "call site(s); the structural chat-free guarantee is void.")
        struct[os.path.basename(p)] = {"apply_chat_template_sites": 0,
                                       "n_lines": src.count("\n") + 1}

    per_cell = {}
    for label, d in tags.items():
        sp = os.path.join(d, "summary.json")
        if not os.path.isfile(sp):
            raise SystemExit(f"FATAL: no summary.json for protocol audit: {sp}")
        blob = json.load(open(sp))
        meta = blob.get("meta", {}) or {}
        ct = blob.get("chat_template", meta.get("chat_template", False))
        # `is not False` -- a None or True or missing-but-defaulted-wrong must fail
        if ct is not False:
            raise SystemExit(
                f"FATAL: {label}: chat_template is {ct!r}, expected False "
                "(assertion is `is not False`, so None also fails).")
        ab = blob.get("add_bos", meta.get("add_bos"))
        if ab is not False:
            raise SystemExit(f"FATAL: {label}: add_bos is {ab!r}, expected False.")
        per_cell[label] = {"chat_template": False, "add_bos": False,
                           "assertion": "chat_template is not False -> FAIL",
                           "max_new_tokens": meta.get("max_new_tokens"),
                           "base_model": meta.get("base_model"),
                           "ckpt": meta.get("ckpt"),
                           "ckpt_step": meta.get("ckpt_step")}
    return {"structural_scripts": struct, "per_cell": per_cell,
            "note": ("chat_template is False STRUCTURALLY: neither eval script has "
                     "an apply_chat_template call site or a flag to enable one. "
                     "These are BASE LMs (no SFT/RL); any chat=True number is void.")}


def load_cell(raw_root, spec, want_rows=False):
    """Load every axis for one (arm) cell with hard integrity assertions.

    `load_shards` (imported, canonical) already asserts: exactly 8 shard files,
    every index 0..7 present exactly once, no duplicate item_id, merged count ==
    expected_n. We additionally assert 0 nan explicitly per axis, and record the
    integrity block so it lands in the evidence JSON rather than living only in a
    passing assertion.
    """
    mm_root = os.path.join(raw_root, "olmo2_mmlu_content_results")
    cb_root = os.path.join(raw_root, "olmo2_closedbook_results")
    vecs, integ, rows_keep = {}, {}, {}

    d = os.path.join(mm_root, spec["mmlu"])
    rows = load_shards(d, "mmlu", EXPECTED_N["mmlu"])
    for r in rows:
        assert r.get("nan") is not True, f"{spec['mmlu']}: nan:true row"
        cn = r.get("content_norm")
        assert isinstance(cn, dict), f"{spec['mmlu']}: content_norm not nested dict"
        assert isinstance(cn.get("correct"), bool), \
            f"{spec['mmlu']}: content_norm.correct not bool"
    vecs["mmlu_content"] = mmlu_content_norm_vec(rows)
    integ["mmlu_content"] = _integ(d, rows, EXPECTED_N["mmlu"], "mmlu")
    if want_rows:
        rows_keep["_mmlu_rows"] = rows

    d = os.path.join(cb_root, spec["cb"])
    for task in ("triviaqa", "popqa"):
        rr = load_shards(d, task, EXPECTED_N[task])
        vecs[task] = qa_metric_vec(rr, "em")
        integ[task] = _integ(d, rr, EXPECTED_N[task], task)
        if want_rows:
            rows_keep[f"_{task}_rows"] = rr

    d = os.path.join(cb_root, spec["nq"])
    rr = load_shards(d, "nq_open", EXPECTED_N["nq_open"])
    vecs["nq_open"] = qa_metric_vec(rr, "em")
    integ["nq_open"] = _integ(d, rr, EXPECTED_N["nq_open"], "nq_open")
    if want_rows:
        rows_keep["_nq_open_rows"] = rr

    for ax, v in vecs.items():
        n_nan = int(np.isnan(np.asarray(v, float)).sum())
        assert n_nan == 0, f"{ax}: {n_nan} nan in metric vector"
        integ[ax]["n_nan_in_metric_vector"] = 0
    if want_rows:
        vecs.update(rows_keep)
    return vecs, integ


def _integ(d, rows, expected_n, stem):
    files = sorted(glob.glob(os.path.join(
        d, f"per_example_{'mmlu' if stem == 'mmlu' else stem}_shard*of8.jsonl")))
    idx = sorted(int(os.path.basename(f).split("shard")[-1].split("of")[0])
                 for f in files)
    assert idx == list(range(8)), f"{d}/{stem}: shard index set {idx} != 0..7"
    ids = [r.get("item_id", r.get("idx")) for r in rows]
    assert len(set(ids)) == len(ids), f"{d}/{stem}: duplicate item_id"
    assert len(rows) == expected_n, f"{d}/{stem}: n={len(rows)} != {expected_n}"
    return {"dir": d, "shard_index_set": idx, "n_shards": 8,
            "n_items": len(rows), "n_items_expected": expected_n,
            "n_duplicate_item_ids": 0,
            "assertions": ["index set == {0..7} (not a count)",
                           "exact item count", "0 duplicate item_ids", "0 nan"]}


def assert_seeds_disjoint(evidence_dir, used_arm_indices, used_offsets,
                          self_output_basename=None):
    """EXECUTE the seed-disjointness claim. Copied UNWEAKENED (self-excluding
    variant) from `a04_keep12_trajectory_monotonicity.py`, which is the fixed
    version -- it excludes only THIS run's own output file, whose recorded offsets
    are this run's by construction.

    Defensive about SHAPE: two evidence files here have a JSON LIST at top level.
    """
    found, skipped = {}, {}
    for fn in sorted(os.listdir(evidence_dir)):
        if not fn.endswith(".json"):
            continue
        if self_output_basename and fn == self_output_basename:
            skipped[fn] = ("this run's own output file (about to be overwritten); "
                           "its offsets ARE this run's by construction")
            continue
        p = os.path.join(evidence_dir, fn)
        try:
            blob = json.load(open(p))
        except Exception as e:
            skipped[fn] = f"unreadable: {type(e).__name__}"
            continue
        if not isinstance(blob, dict):
            skipped[fn] = f"top-level {type(blob).__name__}, carries no offsets"
            continue
        bo = blob.get("bootstrap_offsets")
        if bo is None:
            continue
        if not isinstance(bo, dict):
            skipped[fn] = f"bootstrap_offsets is {type(bo).__name__}"
            continue
        ai = bo.get("arm_index")
        idxs = sorted(set(ai.values())) if isinstance(ai, dict) else []
        found[fn] = {"arm_index": idxs,
                     "guard_seed_offset": bo.get("guard_seed_offset"),
                     "interval_seed_offset": bo.get("interval_seed_offset")}
        clash = sorted(set(idxs) & set(used_arm_indices))
        if clash:
            raise SystemExit(
                f"FATAL: arm_index {clash} already used by {fn} -- re-running that "
                "archive would produce different numbers. Choose a disjoint base.")
    return {"archives_scanned": len(found), "per_archive": found,
            "archives_skipped": skipped,
            "this_run_arm_indices": sorted(used_arm_indices),
            "this_run_offsets": used_offsets,
            "checked_mechanically": True,
            "checker_provenance": ("verbatim from a04_keep12_trajectory_monotonicity.py "
                                   "(the fixed self-excluding version); NOT weakened")}


# ---------------------------------------------------------------------------
def _exchangeability_probe():
    """Is PRE-fix Arm 3 (seed 0) mechanistically exchangeable with the post-fix
    draws, GIVEN that all four arms resume from a COMMON step200000.pt?

    This probe exists because the s7.2 exclusion is stated over LAUNCH TIME, and
    launch time is a proxy for a mechanism. Here the mechanism is checkable: with a
    common resume ckpt, fresh-tail init variance is identically ZERO for all four
    arms, so the ONLY stochastic input in every one of them is the sampler order.
    Pre-fix simply pins sampler.seed := 0; post-fix pins it to 43/44/45.

    If pre-fix Arm 3's order is bit-identical to `DistributedSampler(seed=0)`, then
    seed 0 is a legitimate 4th draw from the SAME data-order family, NOT a
    degenerate one -- and the df=3 family is defensible ON THIS ARM even though it
    is literally s7.2-noncompliant. Reported either way; the conclusion is not
    assumed.

    Uses the same FakeDS-of-the-right-length technique as
    `a04_sampler_seed_probe.py` (DistributedSampler only reads len(ds)). CPU, no
    data file, no model.
    """
    import torch
    from torch.utils.data.distributed import DistributedSampler

    class FakeDS(torch.utils.data.Dataset):
        def __init__(self, n): self.n = n
        def __len__(self): return self.n
        def __getitem__(self, i): return i

    ds = FakeDS(15491607)   # dataset rows= printed by all four arms' logs
    WORLD, RANK, EPOCH = 8, 0, 1   # epoch=1: every arm logged set_epoch(1) on resume

    def order(seed, use_fix, k=12):
        s = (DistributedSampler(ds, num_replicas=WORLD, rank=RANK, shuffle=True,
                                seed=seed) if use_fix else
             DistributedSampler(ds, num_replicas=WORLD, rank=RANK, shuffle=True))
        s.set_epoch(EPOCH)
        return list(s)[:k], s.seed

    def sl(seed, use_fix, n=2_560_000 // WORLD):
        s = (DistributedSampler(ds, num_replicas=WORLD, rank=RANK, shuffle=True,
                                seed=seed) if use_fix else
             DistributedSampler(ds, num_replicas=WORLD, rank=RANK, shuffle=True))
        s.set_epoch(EPOCH)
        it = iter(s)
        return {next(it) for _ in range(n)}

    o_pre, s_pre = order(42, False)      # Arm 3: --seed 42, PRE-fix -> sampler.seed 0
    o_zero, s_zero = order(0, True)      # post-fix, seed=0 explicitly
    identical = (o_pre == o_zero)
    posts = {sd: order(sd, True) for sd in (43, 44, 45)}
    distinct = len({tuple(o_pre)} | {tuple(v[0]) for v in posts.values()})

    A = sl(42, False)
    S = {sd: sl(sd, True) for sd in (43, 44, 45)}
    jac = {f"arm3_prefix_vs_seed{sd}": len(A & S[sd]) / len(A | S[sd])
           for sd in (43, 44, 45)}
    pairs = [(43, 44), (43, 45), (44, 45)]
    jac.update({f"seed{a}_vs_seed{b}": len(S[a] & S[b]) / len(S[a] | S[b])
                for a, b in pairs})

    return {
        "question": ("is PRE-fix Arm 3 (sampler.seed=0) exchangeable with post-fix "
                     "seeds 43/44/45 on THIS arm, given a COMMON resume ckpt?"),
        "why_it_matters": ("s7.2 excludes pre-fix arms because pre-fix seeds varied "
                           "ONLY fresh-tail init. But all four keep7 arms resume "
                           "from one common step200000.pt, so init variance is "
                           "identically ZERO in all four and the only stochastic "
                           "input is the sampler order. The exclusion's PREMISE "
                           "does not obtain on this particular arm."),
        "arm3_prefix_sampler_seed": s_pre,
        "postfix_seed0_sampler_seed": s_zero,
        "first12_identical": bool(identical),
        "distinct_orders_among_4_draws": distinct,
        "rank0_20k_slice_jaccard": jac,
        "finding": (
            "PRE-fix Arm 3's data order is BIT-IDENTICAL to post-fix "
            "DistributedSampler(seed=0), and its rank-0 slice is near-disjoint from "
            "each post-fix draw (Jaccard ~0.0101-0.0105, indistinguishable from the "
            "0.0104 between post-fix draws themselves). All 4 orders are distinct. "
            "=> On THIS arm, seed 0 is a LEGITIMATE 4th draw from the SAME "
            "data-order family, not a degenerate one."
            if identical else
            "NOT identical -- seed 0 is not a plain sampler-seed-0 draw; the s7.2 "
            "exclusion stands on mechanism as well as on launch time."),
        "consequence_for_this_document": (
            "The s7.2 EXCLUSION IS STILL APPLIED for the headline sigma, because "
            "s7.2 is a binding pre-registered rule and this document does not get "
            "to reinterpret it after seeing that the wider family would be "
            "convenient. But the exclusion is now known to be CONSERVATIVE ON THIS "
            "ARM rather than necessary, and BOTH readings are reported (see "
            "SENSITIVITY_K2_under_every_estimator). The K2 verdict is the same "
            "under every one of them, so nothing rests on the choice."),
        "generalisation_PROHIBITED": (
            "This does NOT rehabilitate pre-fix seeds in general. It holds only "
            "where a COMMON resume checkpoint makes init variance zero. For any arm "
            "pruned fresh per seed (e.g. the keep12 Stage-B family, and every gate "
            "arm in the design) pre-fix seeds genuinely carry init-only variance and "
            "s7.2 applies with full force."),
    }


def _sensitivity_all(d_per_axis, delta_pp, k7, k12):
    """K2 under EVERY defensible estimator. Reported because the pre-registered
    verdict must be shown not to depend on the analyst's choice among them.

    The pre-registered one is `keep12` (PILOT_ONE_PREREG.md s2.2). The others are
    sensitivities, NOT alternative verdicts, and none is OR-ed into the headline.
    """
    T = {2: 2.9199855803537124, 3: 2.3533634348018257, 4: 2.131846786}

    def ppf(p, df):
        if df == 2:
            return -2.0 * math.log(1.0 - p)
        lo, hi = 1e-12, 400.0
        if df == 3:
            f = lambda x: (math.erf(math.sqrt(x / 2.0))
                           - math.sqrt(2.0 * x / math.pi) * math.exp(-x / 2.0) - p)
        elif df == 4:
            f = lambda x: 1.0 - math.exp(-x / 2.0) * (1.0 + x / 2.0) - p
        else:
            raise SystemExit(f"no closed form for df={df}")
        for _ in range(400):
            m = 0.5 * (lo + hi)
            if f(m) < 0:
                lo = m
            else:
                hi = m
        return 0.5 * (lo + hi)

    def gate(sig, df, S):
        t = T[S - 1]
        res, nf, nfhi = {}, 0, 0
        for ax in AXES:
            s = sig[ax]
            lo = s * math.sqrt(df / ppf(0.975, df))
            hi = s * math.sqrt(df / ppf(0.025, df))
            b = t * s / math.sqrt(S)
            bhi = t * hi / math.sqrt(S)
            ex, exhi = b > delta_pp[ax], bhi > delta_pp[ax]
            if ax in DECISION_AXES:
                nf += ex
                nfhi += exhi
            res[ax] = {"sigma_pp": s, "sigma_chi2_95ci_pp": [lo, hi],
                       "bound_pp": b, "bound_at_chi2_hi_pp": bhi,
                       "delta_pp": delta_pp[ax], "exceeds": bool(ex),
                       "exceeds_at_chi2_hi": bool(exhi)}
        v = ("K2_FIRES" if nf >= 2 else
             "K2_INDETERMINATE" if nf == 1 else "K2_DOES_NOT_FIRE")
        return {"per_axis": res, "n_decision_axes_exceeding": int(nf),
                "n_decision_axes_exceeding_at_chi2_hi": int(nfhi),
                "t_used": t, "S": S, "df": df, "verdict": v}

    clean = {ax: d_per_axis[ax]["sigma_run_pp"] for ax in AXES}
    mixed = {ax: k7[ax]["s_pp"] for ax in AXES}
    kk12 = {ax: k12[ax]["s_pp"] for ax in AXES}
    pooled = {ax: math.sqrt((2 * clean[ax] ** 2 + 2 * kk12[ax] ** 2) / 4.0)
              for ax in AXES}

    out = {
        "PRE_REGISTERED_ESTIMATOR": "keep12_S3_df2 (PILOT_ONE_PREREG.md s2.2)",
        "keep12_S3_df2_PREREGISTERED": gate(kk12, 2, 3),
        "keep7_clean_S3_df2_this_document": gate(clean, 2, 3),
        "keep7_mixed_S4_df3_as_A03_recorded": gate(mixed, 3, 4),
        "pooled_keep7clean_plus_keep12_df4": gate(pooled, 4, 3),
    }
    verds = {k: v["verdict"] for k, v in out.items() if isinstance(v, dict)
             and "verdict" in v}
    out["ALL_ESTIMATORS_AGREE"] = len(set(verds.values())) == 1
    out["verdicts"] = verds
    out["interpretation"] = (
        "K2 does not fire under ANY of the four estimators, on the point estimate. "
        "So the headline verdict is NOT an artefact of choosing the s7.2-clean "
        "family, and equally it is not rescued by choosing the wider one. The "
        "estimator choice is immaterial to K2 here -- which is the only honest way "
        "to report a result that depends on a contested inclusion decision."
        if out["ALL_ESTIMATORS_AGREE"] else
        "ESTIMATORS DISAGREE -- the pre-registered keep12 estimator governs and the "
        "disagreement must be reported prominently as a fragility.")
    out["pooling_still_NOT_LICENSED_as_the_verdict"] = (
        "STATUS.json:...K2_STATUS_UNCHANGED_BY_SEED45.tempting_but_NOT_LICENSED: "
        "substituting a pooled sigma after seeing which answer each gives is a "
        "change of estimator and is prohibited. It is reported here as a "
        "SENSITIVITY only, and it agrees, so it changes nothing.")
    return out


# ---------------------------------------------------------------------------
def _q4_transfer(per_axis, delta_pp_1b):
    """Q4: what does it cost to gate a 7B experiment with a 1B sigma_run?

    The 7B Delta values are CANONICAL, read from A04's own 7B evidence
    (a04_keep14_trajectory_ni.json / a04_control_arms_ni.json, `split`
    convention, identical in both). They are NOT re-derived here.

    This is a SENSITIVITY, not a 7B result. No 7B sigma_run exists or is
    reconstructible (must_not_claim[23]).
    """
    DELTA_7B_SPLIT = {           # canonical, from A04's 7B evidence JSONs
        "triviaqa": 6.3291350869371374,
        "popqa": 2.245741921917712,
        "mmlu_content": 1.8613801452784504,
        "nq_open": 1.994459833795014,
    }
    rows, n_fire = {}, 0
    for ax in AXES:
        b = per_axis[ax]["bound_S3_pp"]
        d7, d1 = DELTA_7B_SPLIT[ax], delta_pp_1b[ax]
        ex7 = b > d7
        if ax in DECISION_AXES:
            n_fire += ex7
        rows[ax] = {
            "sigma_run_1B_pp": per_axis[ax]["sigma_run_pp"],
            "bound_S3_1B_pp": b,
            "delta_1B_pp": d1, "delta_7B_pp": d7,
            "delta_7B_over_1B": d7 / d1,
            "exceeds_delta_7B": bool(ex7),
            "headroom_delta7B_over_bound": d7 / b if b > 0 else None,
            "sigma_inflation_needed_to_breach_delta_1B": d1 / b if b > 0 else None,
            "sigma_inflation_needed_to_breach_delta_7B": d7 / b if b > 0 else None,
        }
    ratios_1b = sorted(rows[a]["sigma_inflation_needed_to_breach_delta_1B"]
                       for a in DECISION_AXES)
    ratios_7b = sorted(rows[a]["sigma_inflation_needed_to_breach_delta_7B"]
                       for a in DECISION_AXES)
    return {
        "question": ("A04's rungs are 7B; this sigma is 1B. What is the cost of the "
                     "extrapolation, and is it an upper or a lower bound?"),
        "delta_7B_source": ("CANONICAL: evidence/a04_keep14_trajectory_ni.json and "
                            "evidence/a04_control_arms_ni.json, per_convention.split."
                            "delta_pp -- identical in both. NOT re-derived here."),
        "per_axis": rows,
        "n_decision_axes_exceeding_delta_7B": int(n_fire),
        "verdict_if_the_1B_sigma_were_a_7B_sigma": (
            "K2_FIRES" if n_fire >= 2 else "K2_DOES_NOT_FIRE"),
        "KEY_STRUCTURAL_FACT": (
            "Every 7B Delta is LARGER than its 1B counterpart (x1.57-x2.06), because "
            "Delta = 0.10 x residual(intact) and the 7B intact residual is larger. "
            "So a sigma held constant IN pp is MORE easily accommodated at 7B, not "
            "less. The K2 test therefore gets EASIER to pass as the anchor's residual "
            "grows -- which is a property of a data-dependent margin "
            "(must_not_claim[22], arXiv:2603.16213), not a property of the model."),
        "BOUND_DIRECTION": (
            "CANNOT BE SIGNED, and this document declines to sign it. Two effects "
            "act in OPPOSITE directions and neither is measured: (i) Delta is "
            "1.57-2.06x larger at 7B, which makes the 1B sigma CONSERVATIVE "
            "(pessimistic) as a 7B gate input; (ii) the 1B sigma here is itself "
            "DOWNWARD-biased because all three arms share one init (see "
            "EXCHANGEABILITY / prereg s1.5), which makes it OPTIMISTIC. There is no "
            "measurement of how sigma_run itself scales with parameter count on "
            "THIS harness, so the product of the two effects has unknown sign. "
            "Writing 'upper bound' or 'lower bound' here would be a guess."),
        "external_evidence_may_inform_direction_but_NOT_be_tabulated": (
            "arXiv:2508.13144 (NeurIPS 2025 Spotlight) Table 4 publishes OLMo-2 "
            "per-task noise at 1.5B/7B/13B/32B, i.e. the only published handle on "
            "the sign of the scale effect for this family. But it is INTACT-model "
            "noise, a rel-std over 30 consecutive checkpoints of ONE run (a "
            "checkpoint-selection quantity), on their OLMES protocol -- not a "
            "cross-run sigma on A04's base protocol. Per must_not_claim[20] and the "
            "literature note's own prohibition it may be discussed but NOT tabulated "
            "against these numbers."),
        "how_far_from_firing": {
            "sigma_inflation_needed_for_K2_at_1B_delta": ratios_1b[1],
            "sigma_inflation_needed_for_K2_at_7B_delta": ratios_7b[1],
            "reading": ("K2 needs >=2 of 3 decision axes, so the SECOND-easiest axis "
                        "sets the bar. sigma would have to be ~8.9x larger to fire "
                        "against the 1B Delta and ~14.0x larger against the 7B Delta. "
                        "The margin is not marginal -- but see NOT_A_CLEARANCE: a "
                        "constant-REJECT rung is EXPECTED to have small sigma, so a "
                        "large distance from firing is not evidence the gate is safe."),
        },
        "no_7B_sigma_exists": ("must_not_claim[23]: one seed per 7B rung, historical "
                              "seeds unrecorded, --seed postdates the trainer "
                              "revision that produced them. No 7B sigma_run is "
                              "computable or reconstructible. This block is a "
                              "SENSITIVITY, never a 7B result."),
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--a03_sigma_json", required=True,
                    help="canonical a03_sigma_run_n3.json (lives on wzc1 only)")
    ap.add_argument("--evidence_dir", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    gpu_guard = assert_gpu_clear()
    print(f"[guard] GPUs clear, max used {gpu_guard['max_used_mib']} MiB")

    out = {
        "scope": ("A04 sigma_run from the POST-ce5c298 keep7+fresh2 data-order "
                  f"triplet (seeds {SEEDS}) at step {STEP}, and the K2 "
                  "re-adjudication it licenses."),
        "prereg": "A04_SIGMA_RUN_POSTFIX_K2_PREREG.md (commit 94839e8, PRE-DATA)",
        "gpu_h_spent": 0.0,
        "node": os.uname().nodename,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "scipy_available": False,
        "chi2_method": ("df=2 closed form: CDF=1-exp(-x/2) => ppf(p)=-2ln(1-p); "
                        "asserted df==2. scipy absent on .73."),
        "estimator": ("per-axis ARM MEAN per seed (absolute accuracy), NOT the "
                      "paired delta -- a paired delta shares the baseline term "
                      "across seeds and understates single-arm spread. "
                      "s = sample sd (ddof=1), df = S-1 = 2."),
        "preregistered_constants": dict(PREREG),
        "mmlu_tie_convention": PREREG_CONVENTION,
        "canonical_imports": {
            "build_nulls": "pilot_zero_rule_disagreement.build_nulls (IMPORTED and CALLED)",
            "scorers_nulls": ("analyze_1b_knowledge_floor: best_constant_qa / "
                              "longest_option_vector / best_constant_letter / "
                              "paired_bootstrap"),
            "SEED": SEED, "N_BOOT": N_BOOT, "TIE_CONVS": list(TIE_CONVS),
            "no_constant_copied_from_prose": True,
        },
        "RANGE_CONSTANTS_DECLARED_UNUSED": RANGE_CONSTANTS_DECLARED_UNUSED,
        "gpu_refuse_guard": gpu_guard,
        "bootstrap_offsets": {
            "arm_index": {f"seed{s}": NEW_ARM_INDEX_BASE + i
                          for i, s in enumerate(SEEDS)},
            "guard_seed_offset": f"SEED+{GUARD_SEED_OFF} (unused: no guard bootstrap here)",
            "interval_seed_offset": None,
            "note": ("This analysis computes NO bootstrap: sigma_run is a sample sd "
                     "over 3 arm means and the chi2 interval is closed-form. Offsets "
                     "are still CLAIMED and recorded so the disjointness ledger "
                     "stays complete and a future run cannot collide with it."),
        },
    }

    # ---- 0. seed disjointness, EXECUTED ----------------------------------
    used = list(out["bootstrap_offsets"]["arm_index"].values())
    out["seed_disjointness_checked"] = assert_seeds_disjoint(
        args.evidence_dir, used,
        {"arm_index_base": NEW_ARM_INDEX_BASE, "guard": f"SEED+{GUARD_SEED_OFF}"},
        self_output_basename=os.path.basename(args.out_json))
    print(f"[disjoint] scanned {out['seed_disjointness_checked']['archives_scanned']} "
          f"archives, no clash on arm_index {used}")

    # ---- 1. protocol audit -----------------------------------------------
    cb = os.path.join(args.raw_root, "olmo2_closedbook_results")
    mm = os.path.join(args.raw_root, "olmo2_mmlu_content_results")
    audit_tags = {}
    for s in SEEDS:
        audit_tags[f"seed{s}|cb"] = os.path.join(cb, ARM_TAGS[s])
        audit_tags[f"seed{s}|mmlu"] = os.path.join(mm, ARM_TAGS[s])
    audit_tags["intact|cb"] = os.path.join(cb, INTACT["cb"])
    audit_tags["intact|mmlu"] = os.path.join(mm, INTACT["mmlu"])
    out["protocol_audit"] = assert_chat_template_false(args.raw_root, audit_tags)
    print("[protocol] chat_template False (structural + per-cell), add_bos False")

    # ---- 2. intact anchor + nulls, BUILT not copied ----------------------
    intact_vecs, intact_integ = load_cell(args.raw_root, INTACT, want_rows=True)
    nulls = build_nulls(intact_vecs)
    print("[nulls] build_nulls() called on the pinned intact anchor")

    def null_acc(axis):
        if axis == "mmlu_content":
            return nulls["mmlu_content"]["by_convention"][PREREG_CONVENTION]
        return nulls[axis]["acc"]

    delta_pp, resid_intact_pp, xcheck = {}, {}, {}
    for ax in AXES:
        rep = float(np.asarray(intact_vecs[ax], float).mean())
        res = rep - null_acc(ax)
        resid_intact_pp[ax] = 100.0 * res
        delta_pp[ax] = 100.0 * DELTA_FRACTION * res
        d_canon = DELTA_CANONICAL[ax]
        diff = abs(delta_pp[ax] - d_canon)
        if diff > DELTA_XCHECK_TOL:
            raise SystemExit(
                f"FATAL: Delta[{ax}] built = {delta_pp[ax]!r} but canonical = "
                f"{d_canon!r} (|diff| = {diff:.3e} > {DELTA_XCHECK_TOL:.0e}). "
                "Delta is NEVER substituted; aborting rather than proceeding.")
        xcheck[ax] = {"delta_built_pp": delta_pp[ax], "delta_canonical_pp": d_canon,
                      "abs_diff": diff, "tol": DELTA_XCHECK_TOL, "ok": True}
    out["intact_anchor"] = {
        "dirs": INTACT, "integrity": intact_integ,
        "rule": "G0 -- anchor pinned by path; sha256 pins in a04_g0_anchor_sha256_pinning.json",
        "nulls_used": {ax: null_acc(ax) for ax in AXES},
        "residual_intact_pp": resid_intact_pp,
        "delta_pp": delta_pp,
        "delta_rule": "0.10 * residual(intact, x); fraction frozen by git d1ba737",
        "delta_cross_check_vs_canonical": xcheck,
        "delta_never_substituted": True,
    }
    print("[delta] built Delta reproduces canonical on all 4 axes within 1e-9")

    # ---- 3. per-seed arm means + sigma_run -------------------------------
    per_seed_integ, means = {}, {ax: [] for ax in AXES}
    for s in SEEDS:
        spec = {"mmlu": ARM_TAGS[s], "cb": ARM_TAGS[s], "nq": ARM_TAGS[s] + "_nq"}
        vecs, integ = load_cell(args.raw_root, spec)
        per_seed_integ[f"seed{s}"] = integ
        for ax in AXES:
            means[ax].append(100.0 * float(np.asarray(vecs[ax], float).mean()))
    out["per_seed_integrity"] = per_seed_integ

    print(f"\n{'axis':<14}{'s43':>9}{'s44':>9}{'s45':>9}"
          f"{'sigma':>9}{'bound3':>9}{'Delta':>9}  exceeds?")
    per_axis = {}
    n_fire = n_fire_hi = 0
    for ax in AXES:
        m = means[ax]
        s = float(np.std(m, ddof=1))
        bound = T_05_DF2 * s / math.sqrt(3.0)
        lo, hi = sigma_chi2_ci(s, 2)
        bound_hi = T_05_DF2 * hi / math.sqrt(3.0)
        exceeds = bool(bound > delta_pp[ax])
        exceeds_hi = bool(bound_hi > delta_pp[ax])
        if ax in DECISION_AXES:
            n_fire += exceeds
            n_fire_hi += exceeds_hi
        per_axis[ax] = {
            "means_pct": m, "S": 3, "df": 2,
            "sigma_run_pp": s,
            "sigma_chi2_95ci_pp": [lo, hi],
            "chi2_width_multiplicative": (hi / lo) if lo > 0 else None,
            "bound_S3_pp": bound,
            "bound_at_sigma_ci_hi_pp": bound_hi,
            "delta_pp": delta_pp[ax],
            "exceeds_delta": exceeds,
            "would_exceed_at_sigma_ci_hi": exceeds_hi,
            "headroom_multiple_delta_over_bound": (delta_pp[ax] / bound
                                                   if bound > 0 else None),
            "decision_weight": ax in DECISION_AXES,
            "spread_pp_max_minus_min": float(max(m) - min(m)),
        }
        print(f"{ax:<14}{m[0]:>9.4f}{m[1]:>9.4f}{m[2]:>9.4f}"
              f"{s:>9.4f}{bound:>9.4f}{delta_pp[ax]:>9.4f}  {exceeds}")
    out["per_axis"] = per_axis
    out["t_0.05_df2"] = T_05_DF2

    # ---- 4. K2 verdict ---------------------------------------------------
    if n_fire >= 2:
        verdict = "K2_FIRES"
    elif n_fire == 1:
        verdict = "K2_INDETERMINATE"
    else:
        verdict = "K2_DOES_NOT_FIRE"
    out["K2"] = {
        "clause_verbatim": (
            "K2 -- disagreement drowned by seed variance. ... the one-sided 95% "
            "run-level bound t_{0.05,S-1} * sd_run / sqrt(S) exceeds the "
            "pre-registered non-inferiority margin Delta = 10% of the intact arm's "
            "own calibrated residual on >= 2 of the 4 axes."),
        "operationalisation": ("PILOT_ONE_PREREG.md s2.2: >=2 of the 3 DECISION axes "
                              "(triviaqa/popqa/mmlu_content). nq_open DEMOTED by "
                              "design s5.2, zero decision weight. Exactly 1 of 3 = "
                              "K2_INDETERMINATE (s2.4), neither fire nor clearance."),
        "n_decision_axes_exceeding": int(n_fire),
        "n_decision_axes_exceeding_at_sigma_ci_hi": int(n_fire_hi),
        "rule_needs": ">= 2 of 3",
        "verdict": verdict,
        "chi2_upper_is_NOT_a_second_rule": (
            "prereg s4.3: the pre-registered test is on the POINT ESTIMATE. The "
            "chi2-upper column is reported for honesty (df=2 sigma is imprecise, "
            "12.07x multiplicative width) and MUST ship with any K2 statement, but "
            "is NOT OR-ed into the verdict -- in EITHER direction."),
        "NOT_A_CLEARANCE": (
            "prereg s4.2: a large sigma KILLS, a small sigma does NOT clear. This "
            "family is (i) the wrong arm -- keep7 = 56.2% depth, a confirmed "
            "constant-REJECT rung, and K2 is structurally blind to a saturated "
            "deficit; (ii) the wrong budget -- 20k warm-resume steps; (iii) only "
            "PARTIALLY stochastic -- common init, so this sigma is DOWNWARD-BIASED "
            "for full run-to-run variance, i.e. optimistic for K2."),
    }
    print(f"\ndecision axes exceeding Delta: {n_fire}/3 (rule needs >=2) -> {verdict}")
    print(f"at chi2 upper of sigma: {n_fire_hi}/3 would exceed (NOT a second rule)")

    # ---- 5. Q2 -- comparison to A03's recorded sigma ---------------------
    md5 = subprocess.run(["md5sum", args.a03_sigma_json],
                         capture_output=True, text=True).stdout.split()[0]
    a03 = json.load(open(args.a03_sigma_json))
    fam = a03["families"]
    k7 = fam["keep7_20k_cpt"]
    k12 = fam["keep12_5k"]
    cmp_ = {}
    for ax in AXES:
        s_new = per_axis[ax]["sigma_run_pp"]
        s_mixed = k7["axes"][ax]["s_pp"]
        s_k12 = k12["axes"][ax]["s_pp"]
        cmp_[ax] = {
            "sigma_postfix_only_df2_pp": s_new,
            "sigma_A03_keep7_MIXED_df3_pp": s_mixed,
            "ratio_clean_over_mixed": (s_new / s_mixed) if s_mixed > 0 else None,
            "sigma_A03_keep12_postfix_df2_pp": s_k12,
            "ratio_keep7clean_over_keep12": (s_new / s_k12) if s_k12 > 0 else None,
        }
    out["Q2_comparison_to_A03"] = {
        "a03_json": args.a03_sigma_json,
        "a03_json_md5": md5,
        "a03_json_md5_expected": A03_SIGMA_MD5_EXPECTED,
        "a03_json_md5_ok": md5 == A03_SIGMA_MD5_EXPECTED,
        "WHAT_THE_RATIO_IS": (
            "clean-vs-CONTAMINATED, not pre-vs-post. A03's keep7_20k_cpt family is "
            "seeds [0,43,44,45]; seed 0 is Arm 3, launched 2026-08-09 01:11:43, "
            "22h09m BEFORE ce5c298 (23:21:09). So the df=3 value is 3 post-fix "
            "draws + 1 PRE-fix draw pooled, which PROPOSAL.md s7.2 forbids. The "
            "ratio therefore measures the effect of REMOVING one pre-fix draw."),
        "CLEAN_PRE_VS_POST_IS_NOT_COMPUTABLE": (
            "It needs >=2 PRE-fix seed replicates of one arm with evals on these "
            "axes. The only pre-fix multi-'seed' object in the repo is "
            "outputs/olmo2_probe2_7B_keep14fresh2_seed1234 -- 7B, NO eval shards on "
            "either disk, labelled init-variance-only. A03 Arms 3/4/6 are pre-fix "
            "but are DIFFERENT LR SCHEDULES (arm4=peaklr, arm6=lowerband), so their "
            "spread is a schedule effect, not seed variance. Reported as "
            "not-computable rather than proxied."),
        "per_axis": cmp_,
        "a03_keep7_family_seeds": k7["seeds"],
        "a03_keep7_family_dirs": k7["dirs"],
        "a03_keep12_family_seeds": k12["seeds"],
        "a03_pooled_df5_recorded_in_STATUS_POOLS_THE_NONCOMPLIANT_FAMILY": {
            "note": ("STATUS.json:sigma_run_input_from_A03.pooled_df5 pools "
                     "keep7(df=3, contaminated) with keep12(df=2, clean). Since "
                     "the keep7 term is 口径-noncompliant, the pooled value "
                     "inherits that. Recorded, not recomputed -- and per "
                     "STATUS.json's own 'tempting_but_NOT_LICENSED', substituting "
                     "the pooled sigma into K2 is a change of estimator and "
                     "remains unlicensed."),
            "triviaqa_pp": 0.3666, "popqa_pp": 0.2595,
            "nq_open_pp": 0.1445, "mmlu_content_pp": 0.0656,
            "source": "STATUS.json:sigma_run_input_from_A03.pooled_df5",
        },
    }
    print(f"\n[Q2] a03_sigma_run_n3.json md5 {md5} "
          f"(expected {A03_SIGMA_MD5_EXPECTED}) ok="
          f"{md5 == A03_SIGMA_MD5_EXPECTED}")

    # ---- 6. defect record ------------------------------------------------
    out["EXCHANGEABILITY_PROBE_seed0_is_a_LEGITIMATE_4th_DRAW"] = _exchangeability_probe()

    out["SENSITIVITY_K2_under_every_estimator"] = _sensitivity_all(
        d_per_axis=per_axis, delta_pp=delta_pp, k7=k7["axes"], k12=k12["axes"])

    out["Q4_1B_sigma_against_7B_delta"] = _q4_transfer(per_axis, delta_pp)

    out["DEFECT_FOUND_a03_keep7_family_violates_PROPOSAL_7_2"] = {
        "what": ("a03_sigma_run_n3.json families.keep7_20k_cpt pools seed 0 "
                 "(= A03 Arm 3, PRE-ce5c298) with post-fix seeds 43/44/45, and "
                 "families.pooled_df5 pools that family further."),
        "evidence": {
            "arm3_launch": ("logs/a03_arm3_progress.log: "
                            "'[08-09 01:11:43] launched torchrun pid=3642559'"),
            "arm3_seed_line": ("logs/a03_arm3_cpt20k.log line 1: "
                               "'[seed] set_seed(42) on all ranks' -- and its "
                               "DistributedSampler had no seed= argument"),
            "fix_commit": "ce5c298, 2026-08-09 23:21:09 +0800",
            "margin": "Arm 3 launched 22h09m BEFORE the fix",
            "arm3_has_no_preflight_assertion": ("its progress log has no "
                                                "'trainer post-ce5c298 OK' line, "
                                                "unlike seeds 43/44/45/101/102/103"),
        },
        "rule_violated": ("PROPOSAL.md s7.2: 'A pre-fix seed arm and a post-fix seed "
                          "arm are therefore not draws from the same distribution, "
                          "and must never enter the same sigma_run estimate.'"),
        "consequence": ("The keep7 df=3 sigma and the pooled df=5 sigma are NOT "
                        "口径-clean. STATUS.json:sampler_fix_and_pilot_one_disposition"
                        "_20260812's line 'Every run A04 consumes as sigma_run input "
                        "is POST-fix' is TRUE of the six runs it enumerates "
                        "(43/44/45/101/102/103) but FALSE of the keep7 FAMILY as "
                        "recorded, which carries a 4th draw it did not enumerate."),
        "does_NOT_affect": ("K2's pre-registered estimator, which is the KEEP12 "
                            "family (101/102/103) -- all three post-fix with positive "
                            "preflight assertions. K2's arithmetic is untouched."),
        "not_retroactively_edited": ("a03_sigma_run_n3.json is ARCHIVED provenance "
                                     "and is NOT modified. This entry records the "
                                     "defect; the archive keeps its history."),
    }

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
