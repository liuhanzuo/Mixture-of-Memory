#!/usr/bin/env python3
"""B10 Gate 1 -- paired statistics for the base-axis contrast.

Pre-registered KILL condition (PROPOSAL.md S5 / STATUS.json kill_gate.gate_1):
  "on the base axis (gold ceiling 0.9894, so >=98% of items feasible), the
   qwen_fim vs dreamon_oracle paired contrast is NOT significant at alpha=0.05
   AND |delta| < 0.02"

Both conditions must hold for KILL.

Statistics, as mandated by the task spec:
  * exact McNemar on the 0/1 per-item correctness (primary; discordant pairs
    reported), computed exactly via the binomial two-sided test -- NO chi-square
    approximation, NO continuity correction.
  * paired bootstrap on delta with >=10000 resamples, fixed seed, reported.
  * delta point estimate + 95% CI. Two CIs are given: the paired bootstrap
    percentile CI, and the exact-binomial-derived CI on the discordant split
    (Clopper-Pearson on b/(b+c) mapped to delta), which is the CI that matches
    the McNemar test.

delta is defined as  pass@1(qwen_fim) - pass@1(dreamon_oracle)
i.e. AR minus best-diffusion. POSITIVE delta = AR advantage.

ZERO GPU. Pure numpy/stdlib.
"""
from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

ARMS = ["dream_fim", "dreamon_fim", "dreamon_oracle",
        "dream_prefix", "qwen_fim", "qwen_prefix"]
AR_ARM = "qwen_fim"
DIFF_ARM = "dreamon_oracle"   # pre-registered "strongest diffusion arm"
N_BOOT = 10000
SEED = 20260815


def load_scores(root: Path, fname: str):
    out = {}
    for arm in ARMS:
        p = root / arm / fname
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d["which_tests"] == ("base" if "base" in fname else "plus"), \
            f"{arm}: which_tests={d['which_tests']} unexpected for {fname}"
        per = {t["task_id"]: bool(t["pass"]) for t in d["per_task"]}
        assert len(per) == d["n"], f"{arm}: dup task_id in per_task"
        out[arm] = {"path": str(p), "n": d["n"], "pass_at_1": d["pass_at_1"],
                    "n_pass": d["n_pass"], "per_item": per,
                    "self_test": d["grader_self_test"],
                    "exact_match_rate": d["exact_match_rate"]}
    return out


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar = two-sided binomial test on b of n=b+c, p=0.5.

    Implemented as the standard "sum of probabilities <= observed" two-sided
    binomial p-value, clipped at 1.0.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _binom_sf_ge(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binom(n, p). Exact sum with math.comb, no scipy."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    return min(1.0, sum(math.exp(math.log(math.comb(n, i)) + i * lp
                                 + (n - i) * lq) for i in range(k, n + 1)))


def _binom_cdf_le(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binom(n, p). Exact sum with math.comb, no scipy."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    return min(1.0, sum(math.exp(math.log(math.comb(n, i)) + i * lp
                                 + (n - i) * lq) for i in range(0, k + 1)))


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact (Clopper-Pearson) binomial CI, by bisection on the exact tails.

    lower = p solving  P(X >= k | p) = alpha/2
    upper = p solving  P(X <= k | p) = alpha/2
    Both tails are exact binomial sums (math.comb), so this needs no scipy and
    no Beta-function approximation. Self-validated in _selfcheck() below against
    the textbook CP values for k=0/1/5 of n=10 and k=2 of n=20.
    """
    if n == 0:
        return (0.0, 1.0)

    def bisect(fn, target):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if fn(mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lo = 0.0 if k == 0 else bisect(lambda p: _binom_sf_ge(k, n, p), alpha / 2)
    hi = (1.0 if k == n else
          bisect(lambda p: 1.0 - _binom_cdf_le(k, n, p), 1.0 - alpha / 2))
    return (lo, hi)


def _selfcheck():
    """Validate the hand-rolled exact stats against textbook reference values.

    No scipy exists on either disk (checked 2026-08-15 on .73 and on wzc1
    LOCAL), so the exact-test implementations are validated numerically here
    and the result is embedded in the report. Reference values are the standard
    Clopper-Pearson / binomial-test numbers.
    """
    checks = []

    def close(a, b, tol=1e-6):
        return abs(a - b) < tol

    # Clopper-Pearson reference values (exact, well known)
    for (k, n, ref_lo, ref_hi) in [
        (0, 10, 0.0, 0.30849678),
        (1, 10, 0.00252858, 0.44501612),
        (5, 10, 0.18708603, 0.81291397),
    ]:
        lo, hi = clopper_pearson(k, n)
        checks.append({"test": f"clopper_pearson({k},{n})",
                       "got": [lo, hi], "ref": [ref_lo, ref_hi],
                       "ok": close(lo, ref_lo, 1e-6) and close(hi, ref_hi, 1e-6)})

    # Definitional invariant (stronger than a memorised digit): the CP bounds
    # must satisfy P(X>=k | lo) == alpha/2 and P(X<=k | hi) == alpha/2 exactly.
    # Includes the (b, b+c) actually used by the primary contrast.
    for (k, n) in [(2, 20), (31, 63), (55, 108), (100, 400)]:
        lo, hi = clopper_pearson(k, n)
        got_lo = _binom_sf_ge(k, n, lo)
        got_hi = _binom_cdf_le(k, n, hi)
        checks.append({"test": f"clopper_pearson_invariant({k},{n})",
                       "got": [got_lo, got_hi], "ref": [0.025, 0.025],
                       "ok": close(got_lo, 0.025, 1e-9)
                             and close(got_hi, 0.025, 1e-9)})

    # exact two-sided binomial (== exact McNemar) reference values
    for (b, c, ref) in [
        (0, 0, 1.0),
        (5, 5, 1.0),
        (10, 0, 2.0 / 1024),        # 2 * 0.5^10
        (9, 1, 2 * 11 / 1024),      # 2 * (C(10,0)+C(10,1))/2^10
        (31, 32, 1.0),
    ]:
        got = exact_mcnemar(b, c)
        checks.append({"test": f"exact_mcnemar({b},{c})", "got": got,
                       "ref": ref, "ok": close(got, ref, 1e-9)})

    return {"all_ok": all(c["ok"] for c in checks), "checks": checks}


def paired_bootstrap(x, y, n_boot=N_BOOT, seed=SEED):
    """Percentile CI + two-sided p for delta = mean(x) - mean(y), items paired.

    Resamples ITEMS (so the pairing is preserved), which is the paired bootstrap
    the task spec asks for. The p-value is the standard bootstrap-percentile
    two-sided p: 2*min(P(delta*<=0), P(delta*>=0)), clipped at 1.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    d_obs = float(x.mean() - y.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_le = float((boots <= 0).mean())
    p_ge = float((boots >= 0).mean())
    p = min(1.0, 2.0 * min(p_le, p_ge))
    return {"delta": d_obs, "ci95": [float(lo), float(hi)],
            "p_two_sided": p, "n_boot": n_boot, "seed": seed,
            "numpy_version": np.__version__}


def contrast(a_name, b_name, scores, ids):
    """Paired contrast a - b over the given item ids."""
    A = scores[a_name]["per_item"]
    B = scores[b_name]["per_item"]
    xa = [1.0 if A[i] else 0.0 for i in ids]
    xb = [1.0 if B[i] else 0.0 for i in ids]
    n = len(ids)
    b = sum(1 for i in ids if A[i] and not B[i])       # a wins
    c = sum(1 for i in ids if B[i] and not A[i])       # b wins
    both = sum(1 for i in ids if A[i] and B[i])
    neither = n - b - c - both
    p_mc = exact_mcnemar(b, c)
    disc = b + c
    # CI matching McNemar: Clopper-Pearson on b/(b+c), mapped to delta scale
    if disc:
        plo, phi = clopper_pearson(b, disc)
        ci_mc = [(2 * plo - 1) * disc / n, (2 * phi - 1) * disc / n]
    else:
        ci_mc = [0.0, 0.0]
    boot = paired_bootstrap(xa, xb)
    return {
        "arm_a": a_name, "arm_b": b_name, "n_items": n,
        "pass_a": sum(xa) / n, "pass_b": sum(xb) / n,
        "n_pass_a": int(sum(xa)), "n_pass_b": int(sum(xb)),
        "delta_a_minus_b": sum(xa) / n - sum(xb) / n,
        "mcnemar": {"b_a_only": b, "c_b_only": c, "discordant": disc,
                    "both_pass": both, "both_fail": neither,
                    "p_exact_two_sided": p_mc,
                    "delta_ci95_clopper_pearson": ci_mc},
        "paired_bootstrap": boot,
    }


def main() -> None:
    root = Path(sys.argv[1])
    gold_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    base = load_scores(root, "score_base.json")
    plus = load_scores(root, "score.json")   # pre-existing plus axis, for record

    sc = _selfcheck()
    assert sc["all_ok"], f"exact-stat self-check FAILED: {json.dumps(sc)}"

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_base = {r["task_id"]: r["gold_base_pass"] for r in gold["per_row"]}
    gold_plus = {r["task_id"]: r["gold_plus_pass"] for r in gold["per_row"]}

    ids_all = sorted(base[ARMS[0]]["per_item"].keys())
    for arm in ARMS:
        assert sorted(base[arm]["per_item"].keys()) == ids_all, \
            f"{arm}: base per_task id set mismatch"
        assert base[arm]["self_test"]["trustworthy"], \
            f"{arm}: grader self-test NOT trustworthy"
    assert set(ids_all) == set(gold_base.keys()), "gold id set mismatch"

    ids_feasible_base = [i for i in ids_all if gold_base[i]]
    ids_feasible_plus = [i for i in ids_all if gold_plus[i]]

    report = {
        "gate": "B10 Gate 1 -- base-axis re-score",
        "primary_contrast": f"{AR_ARM} minus {DIFF_ARM} (AR minus best diffusion)",
        "kill_condition_verbatim": (
            "on the base axis (gold ceiling 0.9894, so >=98% of items feasible), "
            "the qwen_fim vs dreamon_oracle paired contrast is not significant "
            "at alpha=0.05 AND |delta| < 0.02"),
        "n_items_all": len(ids_all),
        "gold_ceiling_base_measured": gold["gold_ceiling_base"],
        "gold_ceiling_plus_measured": gold["gold_ceiling_plus"],
        "gold_ceiling_base_prereg_quoted": 0.9894,
        "n_gold_base_infeasible": len(ids_all) - len(ids_feasible_base),
        "n_gold_plus_infeasible": len(ids_all) - len(ids_feasible_plus),
        "arm_pass_at_1_base": {a: base[a]["pass_at_1"] for a in ARMS},
        "arm_n_pass_base": {a: base[a]["n_pass"] for a in ARMS},
        "arm_pass_at_1_plus_for_reference": {a: plus[a]["pass_at_1"] for a in ARMS},
        "score_files_base": {a: base[a]["path"] for a in ARMS},
        "grader_self_test_base": {a: base[a]["self_test"] for a in ARMS},
        "exact_stat_implementation_selfcheck": sc,
    }

    report["PRIMARY_base_axis_all_items"] = contrast(AR_ARM, DIFF_ARM, base, ids_all)
    report["SECONDARY_base_axis_gold_feasible_subset"] = contrast(
        AR_ARM, DIFF_ARM, base, ids_feasible_base)

    # ROBUSTNESS: the pre-registered condition quotes gold ceiling 0.9894, which
    # was measured on wzc1 and marks 11 items (all HumanEval/32, `find_zero`,
    # special oracle) base-infeasible. This disk measures 1.0 on the same split
    # (identical md5 for both the split file and HumanEvalPlus-v0.1.10.jsonl, and
    # identical vendored evalplus md5), so the feasible subset above equals the
    # full set. Repeat the contrast on the *wzc1* feasible set too, so the
    # adjudication does not depend on which of the two ceiling measurements is
    # taken as authoritative.
    if len(sys.argv) > 4:
        alt = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
        alt_base = {r["task_id"]: r["gold_base_pass"] for r in alt["per_row"]}
        assert set(alt_base) == set(ids_all), "alt gold id set mismatch"
        alt_ids = [i for i in ids_all if alt_base[i]]
        report["alt_gold_ceiling_source"] = str(sys.argv[4])
        report["alt_gold_ceiling_base"] = alt["overall"]["gold_ceiling_base"] \
            if "overall" in alt else alt["gold_ceiling_base"]
        report["alt_n_gold_base_infeasible"] = len(ids_all) - len(alt_ids)
        report["alt_gold_base_infeasible_ids"] = [
            i for i in ids_all if not alt_base[i]]
        report["ROBUSTNESS_base_axis_wzc1_gold_feasible_subset"] = contrast(
            AR_ARM, DIFF_ARM, base, alt_ids)

    # sensitivity: is dreamon_oracle actually the strongest diffusion arm on base?
    diff_arms = ["dreamon_oracle", "dreamon_fim", "dream_fim", "dream_prefix"]
    strongest = max(diff_arms, key=lambda a: base[a]["pass_at_1"])
    report["strongest_diffusion_arm_on_base_axis"] = strongest
    report["SENSITIVITY_vs_strongest_diffusion_arm"] = contrast(
        AR_ARM, strongest, base, ids_all)

    # full pairwise base-axis matrix, for the record (NOT used for adjudication)
    report["all_pairwise_base_axis"] = [
        {"pair": f"{a}-{b}",
         "delta": base[a]["pass_at_1"] - base[b]["pass_at_1"],
         "p_exact_mcnemar": exact_mcnemar(
             sum(1 for i in ids_all
                 if base[a]["per_item"][i] and not base[b]["per_item"][i]),
             sum(1 for i in ids_all
                 if base[b]["per_item"][i] and not base[a]["per_item"][i]))}
        for a, b in combinations(ARMS, 2)]

    pc = report["PRIMARY_base_axis_all_items"]
    p_mc = pc["mcnemar"]["p_exact_two_sided"]
    p_bs = pc["paired_bootstrap"]["p_two_sided"]
    delta = pc["delta_a_minus_b"]
    cond_not_sig = (p_mc >= 0.05)
    cond_small = (abs(delta) < 0.02)
    report["adjudication"] = {
        "delta_a_minus_b": delta,
        "abs_delta": abs(delta),
        "p_exact_mcnemar": p_mc,
        "p_paired_bootstrap": p_bs,
        "condition_1_not_significant_at_alpha_0.05": cond_not_sig,
        "condition_2_abs_delta_lt_0.02": cond_small,
        "both_kill_conditions_hold": bool(cond_not_sig and cond_small),
        "VERDICT": "KILL" if (cond_not_sig and cond_small) else "PROCEED",
    }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "gold_ceiling_base_measured": report["gold_ceiling_base_measured"],
        "arm_pass_at_1_base": report["arm_pass_at_1_base"],
        "strongest_diffusion_arm_on_base_axis": strongest,
        "primary": {k: pc[k] for k in
                    ("n_items", "pass_a", "pass_b", "delta_a_minus_b")},
        "mcnemar": pc["mcnemar"],
        "bootstrap": pc["paired_bootstrap"],
        "adjudication": report["adjudication"],
    }, indent=2))


if __name__ == "__main__":
    main()
