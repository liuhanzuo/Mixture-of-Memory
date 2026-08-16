#!/usr/bin/env python3
"""paperC read-out v2 — the PREDICTION-PERMUTATION null (letter-collapse-proof).

WHY THIS EXISTS
---------------
`HEAL_TRAJECTORY_READOUT_1.md` §4 showed that §8's `H_heal` criterion, as
operationalised on "letter accuracy vs the best-constant floor", is satisfiable
by changing WHICH letter a degenerate model collapses onto:

  * `always-<L>` accuracy is a non-flat DATASET property (A .1166 ... J .0785);
  * so `always-A` IS the floor by construction (it is the argmax over letters);
  * a model that emits A for 82-91% of items therefore scores AT floor;
  * its un-healed twin emits E for 94.5% and scores BELOW floor;
  * an independence model `acc_hat = sum_L P(pred=L)*P(gold=L)` -- containing NO
    competence term -- explains every damaged cell to within +0.07..+1.13 pp.

THE FIX, IN ONE LINE
--------------------
Promote that independence model from a *diagnostic* to *the null*. It is exactly
the expected accuracy under a random permutation of the arm's own prediction
vector across items, so:

  Delta_perm := acc - acc_hat  =  the item-level-alignment component of accuracy,

and a pure letter-collapse has `Delta_perm == 0` EXACTLY, for ANY collapse
letter, by construction. `always-<L>` is the special case `P(pred) = delta_L`,
for which `acc == acc_hat == m_L` and `Delta_perm == 0` identically. That is
requirement (1) discharged as an algebraic identity, not an empirical hope.

RELATION TO A01 / paperC's HEADLINE NULL -- READ THIS BEFORE CALLING IT A
CONTRADICTION
------------------------------------------------------------------------
`build_null_calibration_table.py:596-600` states the design rule this appears to
violate: "the pre-registered null is the BEST constant (always-D) because a floor
must not depend on the arm being tested". That rule is CORRECT for the construct
it serves and this script does not repeal it. The two nulls answer two questions:

  * best-constant floor (arm-INdependent)  -> INSTRUMENT VALIDITY.
    "Does this interface beat the best input-blind predictor?" paperC's headline
    claim. Must be arm-independent or arms are not mutually comparable.
  * permutation null (arm-CONDITIONAL)     -> COMPETENCE OF ONE ARM.
    "Do this arm's predictions carry item-level information?" prereg §8's
    `H_heal`. Must condition on the arm's own prediction marginal, otherwise the
    statistic confounds "which letter it collapsed onto" with "what it knows" --
    which is the observed defect.

So this is a SCOPE PARTITION of A01, not a replacement. Both are reported for
every cell and a sign disagreement between them is DISCLOSED, never resolved
silently (gate G5).

⚠️ The permutation null is <= the best-constant floor always (`sum_L p_L m_L <=
max_L m_L`), with equality iff all prediction mass sits on the argmax letter. It
is therefore a LOWER bar in absolute terms. That is not a loophole -- it is a
different question -- but it must never be quoted as "the arm cleared paperC's
floor".

STRATIFICATION
--------------
MMLU-Pro's `n_opt` is NOT constant: {3:21, 4:606, 5:52, 6:93, 7:158, 8:320,
9:801, 10:9981}. An unstratified permutation can assign `pred=J` to a 4-option
item, where J was never available, and the gold marginal differs sharply by
stratum (`n_opt=3` has gold A at 42.9% vs 10.5% at `n_opt=10`). The CANONICAL
null is therefore permutation WITHIN `n_opt` strata; the unstratified variant is
reported as a sensitivity, in the same discipline A01 applies to tie conventions.

ADMISSIBILITY GATES (evaluated BEFORE the verdict, A04 D1-D6 style)
-------------------------------------------------------------------
G1 CAPACITY  : `Delta_max <= hw95`. `Delta_max` is the largest `Delta_perm` any
               re-assignment of the observed prediction MULTISET could achieve,
               `= sum_L min(n_pred_L, n_gold_L)/n - acc_hat`. If the ceiling is
               under the instrument's resolution the cell cannot express signal
               and is NOT_MEASURABLE. (D6-analogue.)
G2 DEGENERATE: `Delta_max == 0`. A single-letter emitter: the null IS the
               observation, so the test has zero power by construction. (Named
               separately from G1 because it is the historically real case --
               A03's barely-healed keep7@500 emitted letter A on 14042/14042.)
G3 INTEGRITY : shard index set {0..7}, n == 12032, 0 dup, 0 nan, 0 trunc,
               `chat_template is False` (asserted with `is False`; `is not True`
               passes on None and is banned).
G4 NULL-ADMIS: at least one arm on this axis must clear the null, else the null
               is above the whole arm population and measures nothing.
               (D4-analogue.) Checked at roll-up over all cells.
G5 COHERENCE : the best-constant-floor verdict is reported alongside; a sign
               disagreement is flagged, not silently resolved.

Estimators are IMPORTED from `mmlu_pro_power_nulls.py` (which itself asserts
bit-identity against A01's), so nothing is re-implemented.

CPU only. 0 GPU. Usage:
  python heal_readout_v2_permutation_null.py <out_json> [--root R] ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from mmlu_pro_power_nulls import (  # noqa: E402
    ESTIMATOR_SOURCE,
    LETTERS,
    N_BOOT,
    best_constant_letter,
    two_sided_boot_p,
)
from heal_trajectory_nulls import (  # noqa: E402
    PREREG_FLOOR_ACC,
    PREREG_FLOOR_LETTER,
    PREREG_N,
    _integrity,
    _is_scored,
)

MMLU_REFERENCE_EFFECT_PP = 1.389
N_PERM = 10000
PERM_SEED = 7          # same seed family as #248/#250/#251 so tables compare
BOOT_SEED = 7
NL = len(LETTERS)

# MATERIALITY, fixed here BEFORE re-judging any cell.
#
# Significance alone is NOT capability at n=12032: `qwen3/k14` reaches p=0.0066
# at Delta_perm=+0.267 pp while emitting `A` on 94.6% of items. So a magnitude
# scale is required alongside the test, or the defect is re-imported one level up.
#
# The scale is the RECOVERY FRACTION, `Delta_perm / Delta_max`, i.e. how much of
# the item-level alignment that the arm's own prediction marginal could possibly
# express is actually expressed. Then materiality is A04's own pre-registered
# constant -- `Delta = 0.10 * <intact anchor>` (A04_MARGIN_GUARD_PREREG.md §4) --
# applied WITHIN FAMILY:
#
#     material  <=>  recovery_fraction >= 0.10 * recovery_fraction(intact, same family)
#
# Within-family is not a stylistic choice. A01 established that cross-family
# floors are not commensurable (GATE3_CONVENTIONS §3: the content null moves up
# to 10.6 pp across tokenizers on identical items), and paperC's README bans
# family-general orderings on these very rungs. A relative-recovery claim must
# therefore be anchored to the SAME family's intact model.
MATERIAL_FRAC_OF_INTACT = 0.10

# G6 ANCHOR ADMISSIBILITY. A within-family relative claim is only meaningful if
# that family's intact anchor is itself competent on this benchmark. MMLU-Pro is
# 10-way and hard: Llama-2-7B intact scores 0.1320 against a 0.1166 floor, a
# recovery fraction of 0.0545 -- i.e. the "anchor" is itself within a whisker of
# no item-level signal. Taking 10% of that would make the bar 0.0055, which any
# noise clears. So a family whose intact anchor is below this bar yields
# NOT_ANCHORED for relative claims (its absolute Delta_perm is still reported).
MIN_ANCHOR_RECOVERY = 0.10


# ---------------------------------------------------------------------------
# the null itself
# ---------------------------------------------------------------------------
def _encode(recs):
    """-> (pred, gold, stratum) integer arrays, plus the stratum key list."""
    pred = np.array([r["letter"]["pred"] for r in recs], dtype=np.int64)
    gold = np.array([r["gold"] for r in recs], dtype=np.int64)
    nopt = np.array([r["n_opt"] for r in recs], dtype=np.int64)
    keys = sorted(set(nopt.tolist()))
    strat = np.searchsorted(np.array(keys), nopt)
    return pred, gold, strat, keys


def _acc_hat(pred, gold, strat, n_strat):
    """E[acc] under a uniformly random permutation of `pred` WITHIN each stratum.

    Exactly `sum_s (1/(n*n_s)) sum_L cnt_pred[s,L]*cnt_gold[s,L]`. With
    `n_strat == 1` this reduces to the unstratified independence model
    `sum_L P(pred=L)*P(gold=L)`.
    """
    n = pred.size
    cp = np.bincount(strat * NL + pred, minlength=n_strat * NL).reshape(n_strat, NL)
    cg = np.bincount(strat * NL + gold, minlength=n_strat * NL).reshape(n_strat, NL)
    ns = cp.sum(axis=1)
    tot = 0.0
    for s in range(n_strat):
        if ns[s]:
            tot += float(np.dot(cp[s], cg[s])) / ns[s]
    return tot / n


def _capacity(pred, gold, strat, n_strat):
    """Delta_max: the best accuracy ANY re-assignment of the observed prediction
    multiset to items could reach (a transportation upper bound, attained by
    matching each letter's predictions onto that letter's gold items), minus
    acc_hat. Zero for a single-letter emitter -> the test has no power.
    """
    n = pred.size
    cp = np.bincount(strat * NL + pred, minlength=n_strat * NL).reshape(n_strat, NL)
    cg = np.bincount(strat * NL + gold, minlength=n_strat * NL).reshape(n_strat, NL)
    best = float(np.minimum(cp, cg).sum()) / n
    return best, best - _acc_hat(pred, gold, strat, n_strat)


def _perm_test(pred, gold, strat, n_strat, n_perm=N_PERM, seed=PERM_SEED):
    """Two-sided permutation test of `acc` against within-stratum permutation of
    `pred`. Returns (p_two_sided, p_greater, mean_perm_acc, sd_perm_acc)."""
    rng = np.random.default_rng(seed)
    obs = float((pred == gold).mean())
    order = np.argsort(strat, kind="stable")
    bounds = np.searchsorted(strat[order], np.arange(n_strat + 1))
    pred_s, gold_s = pred[order].copy(), gold[order]
    accs = np.empty(n_perm)
    work = pred_s.copy()
    for i in range(n_perm):
        for s in range(n_strat):
            a, b = bounds[s], bounds[s + 1]
            if b - a > 1:
                work[a:b] = rng.permutation(pred_s[a:b])
        accs[i] = (work == gold_s).mean()
    # mid-p on both tails, same convention as the project's bootstrap p
    tie = float((accs == obs).mean())
    p_ge = float((accs > obs).mean()) + 0.5 * tie
    p_le = float((accs < obs).mean()) + 0.5 * tie
    p2 = min(1.0, max(2.0 * min(p_ge, p_le), 1.0 / n_perm))
    return p2, p_ge, float(accs.mean()), float(accs.std(ddof=1))


def _boot_delta(pred, gold, strat, n_strat, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap CI on Delta_perm = acc - acc_hat, RECOMPUTING acc_hat inside
    each resample so both terms move together (an acc-only CI would understate
    the uncertainty by treating the null as fixed when it is estimated from the
    same data)."""
    n = pred.size
    rng = np.random.default_rng(seed)
    corr = (pred == gold).astype(np.float64)
    code_p, code_g = strat * NL + pred, strat * NL + gold
    M = n_strat * NL
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        cp = np.bincount(code_p[idx], minlength=M).reshape(n_strat, NL)
        cg = np.bincount(code_g[idx], minlength=M).reshape(n_strat, NL)
        ns = cp.sum(axis=1)
        tot = 0.0
        for s in range(n_strat):
            if ns[s]:
                tot += float(np.dot(cp[s], cg[s])) / ns[s]
        out[i] = corr[idx].mean() - tot / n
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi), two_sided_boot_p(out, n_boot)


# ---------------------------------------------------------------------------
def analyse(root, arm_dir, label, step, family, regime,
            expect_keep=None, expect_fresh=None):
    recs, integ = _integrity(root, arm_dir, expect_keep, expect_fresh)

    # arm-INdependent floor (paperC's headline null), asserted bit-identical.
    bc_letter, bc_vec, bc_diag = best_constant_letter(recs)
    assert bc_letter == PREREG_FLOOR_LETTER, f"{arm_dir}: floor letter {bc_letter}"
    assert abs(bc_diag["best_constant_acc"] - PREREG_FLOOR_ACC) < 5e-7, \
        f"{arm_dir}: floor {bc_diag['best_constant_acc']:.9f}"

    pred, gold, strat, keys = _encode(recs)
    n, n_strat = pred.size, len(keys)
    acc = float((pred == gold).mean())

    ah_s = _acc_hat(pred, gold, strat, n_strat)
    zero = np.zeros_like(strat)
    ah_u = _acc_hat(pred, gold, zero, 1)                      # sensitivity
    d_s, d_u = acc - ah_s, acc - ah_u

    p2, p_ge, mperm, sdperm = _perm_test(pred, gold, strat, n_strat)
    lo, hi, pboot = _boot_delta(pred, gold, strat, n_strat)
    hw = 100 * (hi - lo) / 2

    cap_acc, cap_delta = _capacity(pred, gold, strat, n_strat)

    # Recovery fraction: Delta_perm normalised by the largest Delta_perm the
    # observed prediction MULTISET could possibly attain. 0 = no item-level
    # information; 1 = perfectly aligned given the marginal the arm emits. This
    # is the magnitude scale -- `p < 0.05` at n=12032 is NOT a capability claim
    # (qwen3/k14 reaches p=0.0066 at +0.267 pp while emitting A on 94.6% of
    # items), so a significance-only criterion would re-import the defect it is
    # meant to remove, one level up.
    recov = (d_s / cap_delta) if cap_delta > 1e-12 else float("nan")

    hist = Counter(LETTERS[p] for p in pred.tolist())
    modal, mhits = hist.most_common(1)[0]
    ent = -sum((v / n) * math.log(v / n) for v in hist.values()) / math.log(NL)

    # ---- admissibility gates, evaluated BEFORE the verdict
    g2 = bool(abs(cap_delta) < 1e-12)
    g1 = bool(100 * cap_delta <= hw)
    admissible = not (g1 or g2)

    if not admissible:
        verd = "NOT_MEASURABLE"
    elif pboot >= 0.05 or p2 >= 0.05:
        verd = "NO_ITEM_LEVEL_SIGNAL"
    elif d_s < 0:
        verd = "ANTI_SIGNAL"
    else:
        # significant and positive. Materiality is decided in pass 2, once the
        # family's intact anchor is known (see `_apply_materiality`).
        verd = "SIGNAL_PENDING_MATERIALITY"

    # ---- G5: the old, arm-independent criterion, reported for coherence
    from mmlu_pro_power_nulls import paired_boot
    lvec = np.array([1.0 if r["letter"]["correct"] else 0.0 for r in recs])
    om, olo, ohi, op = paired_boot(lvec - bc_vec)
    old_verd = ("BELOW floor" if (op < 0.05 and om < 0)
                else "ABOVE floor" if (op < 0.05 and om > 0) else "AT floor")

    return {
        "label": label, "arm_dir": arm_dir, "results_root": root, "step": step,
        "family": family, "regime": regime,
        "letter_acc": acc,
        # --- v2 criterion
        "null_perm_stratified": ah_s,
        "delta_perm_pp": 100 * d_s,
        "ci95_lo_pp": 100 * lo, "ci95_hi_pp": 100 * hi,
        "ci95_half_width_pp": hw,
        "boot_p": pboot,
        "perm_p_two_sided": p2, "perm_p_greater": p_ge,
        "perm_null_mean_acc": mperm, "perm_null_sd_acc": sdperm,
        "n_perm": N_PERM, "n_boot": N_BOOT, "perm_seed": PERM_SEED,
        "verdict_v2": verd,
        # --- sensitivity: unstratified independence model (the §4 diagnostic)
        "null_perm_unstratified": ah_u,
        "delta_perm_unstratified_pp": 100 * d_u,
        "stratification_effect_pp": 100 * (d_s - d_u),
        # --- admissibility
        "capacity_max_acc": cap_acc,
        "capacity_delta_max_pp": 100 * cap_delta,
        "recovery_fraction": recov,
        "G1_capacity_below_resolution": g1,
        "G2_degenerate_zero_capacity": g2,
        "admissible": admissible,
        "powered_vs_mmlu_effect": bool(hw < MMLU_REFERENCE_EFFECT_PP),
        # --- degeneracy descriptors (now descriptive only, not load-bearing)
        "modal_pred_letter": modal, "modal_pred_share": mhits / n,
        "n_distinct_pred_letters": len(hist),
        "pred_entropy_normalised": ent,
        "pred_hist": dict(sorted(hist.items())),
        # --- G5: the v1 criterion side by side
        "v1_floor_letter": bc_letter, "v1_floor_acc": bc_diag["best_constant_acc"],
        "v1_delta_vs_floor_pp": 100 * om,
        "v1_ci95_pp": [100 * olo, 100 * ohi], "v1_boot_p": op,
        "v1_verdict": old_verd,
        "n_opt_strata": {str(k): int((strat == i).sum()) for i, k in enumerate(keys)},
        "integrity": integ,
    }


# ---------------------------------------------------------------------------
def _apply_materiality(cells):
    """PASS 2. Turn `SIGNAL_PENDING_MATERIALITY` into a final label using the
    arm's OWN FAMILY intact anchor. Also sets G6 (anchor admissibility).

    Runs after every cell is measured so the anchor is a measured quantity, not
    a hard-coded one -- but the RULE (`0.10 x intact`, within family) is fixed in
    the pre-registration before any of these numbers were looked at.
    """
    anchors = {c["family"]: c for c in cells if c["regime"] == "intact"}
    for c in cells:
        a = anchors.get(c["family"])
        ar = a["recovery_fraction"] if a else float("nan")
        c["family_intact_recovery_fraction"] = ar
        c["family_intact_delta_perm_pp"] = a["delta_perm_pp"] if a else None
        # G6: is the family's own intact model competent enough to anchor to?
        g6 = bool(a is None or not (ar >= MIN_ANCHOR_RECOVERY))
        c["G6_anchor_not_competent"] = g6
        thr = MATERIAL_FRAC_OF_INTACT * ar if a else float("nan")
        c["material_recovery_threshold"] = thr
        c["relative_recovery"] = (c["recovery_fraction"] / ar
                                  if a and ar > 1e-12 else float("nan"))
        if c["verdict_v2"] != "SIGNAL_PENDING_MATERIALITY":
            continue
        if c["regime"] == "intact":
            c["verdict_v2"] = "ITEM_LEVEL_SIGNAL"      # the anchor itself
        elif g6:
            # cannot make a relative claim; the signal is real but unscaled
            c["verdict_v2"] = "SIGNAL_NOT_ANCHORED"
        elif c["recovery_fraction"] >= thr:
            c["verdict_v2"] = "ITEM_LEVEL_SIGNAL"
        else:
            c["verdict_v2"] = "TRACE_SIGNAL"
    return anchors


# ---------------------------------------------------------------------------
CELLS = [
    # (root_key, arm_dir, label, step, family, regime, keep, fresh)
    ("heal", "qwen3base_heal_k8f2_step5000", "qwen3/k8+f2 heal@5000", 5000,
     "qwen3_8b_base", "prune_then_heal", 8, 2),
    ("heal", "qwen3base_heal_k8f2_step5500", "qwen3/k8+f2 heal@5500", 5500,
     "qwen3_8b_base", "prune_then_heal", 8, 2),
    ("heal", "qwen3base_heal_k8f2_step6000", "qwen3/k8+f2 heal@6000", 6000,
     "qwen3_8b_base", "prune_then_heal", 8, 2),
    ("heal", "qwen3base_heal_k8f2_step6500", "qwen3/k8+f2 heal@6500", 6500,
     "qwen3_8b_base", "prune_then_heal", 8, 2),
    ("heal", "qwen3base_heal_k8f2_step7000", "qwen3/k8+f2 heal@7000", 7000,
     "qwen3_8b_base", "prune_then_heal", 8, 2),
    ("heal", "7B_keep8_step45000", "olmo2/keep8+f2 heal@45000", 45000,
     "olmo2_7b", "prune_then_heal", 8, 2),
    ("olmo", "7B_keep8_step121000", "olmo2/keep8+f2 heal@121000 [P2]", 121000,
     "olmo2_7b", "prune_then_heal", 8, 2),
    ("xf", "qwen3_8b_base_k8", "qwen3/k8 UN-healed [P1]", 0,
     "qwen3_8b_base", "trunc_no_heal", 8, 0),
    ("xf", "qwen3_8b_base_base", "qwen3 INTACT", None, "qwen3_8b_base", "intact",
     None, None),
    ("olmo", "7B_base", "olmo2 INTACT", None, "olmo2_7b", "intact", None, None),
    # --- the remaining archived cells: free, and they are what establishes
    #     requirement (3), discrimination at the intact end.
    ("olmo", "7B_keep10_step83500", "olmo2/keep10+f2 heal@83500", 83500,
     "olmo2_7b", "prune_then_heal", 10, 2),
    ("olmo", "7B_keep12_step124000", "olmo2/keep12+f2 heal@124000", 124000,
     "olmo2_7b", "prune_then_heal", 12, 2),
    ("olmo", "7B_keep14_step200000", "olmo2/keep14+f2 heal@200000", 200000,
     "olmo2_7b", "prune_then_heal", 14, 2),
    ("olmo", "7B_shortgpt16_step200000", "olmo2/shortgpt16 heal@200000", 200000,
     "olmo2_7b", "prune_then_heal", None, None),
    ("xf", "qwen3_8b_base_k10", "qwen3/k10 UN-healed", 0, "qwen3_8b_base",
     "trunc_no_heal", 10, 0),
    ("xf", "qwen3_8b_base_k12", "qwen3/k12 UN-healed", 0, "qwen3_8b_base",
     "trunc_no_heal", 12, 0),
    ("xf", "qwen3_8b_base_k14", "qwen3/k14 UN-healed", 0, "qwen3_8b_base",
     "trunc_no_heal", 14, 0),
    ("xf", "llama2_7b_base", "llama2 INTACT", None, "llama2_7b", "intact",
     None, None),
    ("xf", "llama2_7b_k8", "llama2/k8 UN-healed", 0, "llama2_7b",
     "trunc_no_heal", 8, 0),
    ("xf", "llama2_7b_k10", "llama2/k10 UN-healed", 0, "llama2_7b",
     "trunc_no_heal", 10, 0),
    ("xf", "llama2_7b_k12", "llama2/k12 UN-healed", 0, "llama2_7b",
     "trunc_no_heal", 12, 0),
    ("xf", "llama2_7b_k14", "llama2/k14 UN-healed", 0, "llama2_7b",
     "trunc_no_heal", 14, 0),
    ("xf", "llama3_8b_base", "llama3 INTACT", None, "llama3_8b", "intact",
     None, None),
    ("xf", "llama3_8b_k8", "llama3/k8 UN-healed", 0, "llama3_8b",
     "trunc_no_heal", 8, 0),
    ("xf", "llama3_8b_k10", "llama3/k10 UN-healed", 0, "llama3_8b",
     "trunc_no_heal", 10, 0),
    ("xf", "llama3_8b_k12", "llama3/k12 UN-healed", 0, "llama3_8b",
     "trunc_no_heal", 12, 0),
    ("xf", "llama3_8b_k14", "llama3/k14 UN-healed", 0, "llama3_8b",
     "trunc_no_heal", 14, 0),
]


# ---------------------------------------------------------------------------
# SELF-TEST: requirement (1) as an algebraic identity, not an empirical hope.
# ---------------------------------------------------------------------------
def selftest_collapse_invariance(gold, nopt, strat, n_strat, verbose=True):
    """For EVERY collapse letter L that is legal in a stratum, an `always-L`
    emitter must give Delta_perm == 0 and capacity Delta_max == 0.

    This is the formal discharge of requirement (1): the v1 criterion's value
    swings 2.11 pp across collapse letters (always-A .1166 vs always-E .0955);
    the v2 criterion is IDENTICALLY zero for all of them.
    """
    rows = []
    for L in range(NL):
        # a real emitter can only emit L on items where L is actually an option
        ok = nopt > L
        if ok.sum() == 0:
            continue
        p = np.full(int(ok.sum()), L, dtype=np.int64)
        g, s = gold[ok], strat[ok]
        ah = _acc_hat(p, g, s, n_strat)
        acc = float((p == g).mean())
        _, cap = _capacity(p, g, s, n_strat)
        rows.append({"letter": LETTERS[L], "n_items": int(ok.sum()),
                     "always_L_acc": acc, "acc_hat": ah,
                     "delta_perm_pp": 100 * (acc - ah),
                     "capacity_delta_max_pp": 100 * cap})
        assert abs(acc - ah) < 1e-12, \
            f"always-{LETTERS[L]}: Delta_perm={100*(acc-ah):.9f} pp != 0"
        assert abs(cap) < 1e-12, f"always-{LETTERS[L]}: capacity != 0"
    if verbose:
        print("SELFTEST collapse-invariance: Delta_perm == 0 EXACTLY for all "
              f"{len(rows)} always-<L> emitters")
        span = max(r["always_L_acc"] for r in rows) - \
            min(r["always_L_acc"] for r in rows)
        print(f"  v1 statistic (always-L acc) spans {100*span:.3f} pp across L; "
              f"v2 statistic spans 0.000 pp (identically zero)")
    return rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_json")
    ap.add_argument("--heal_root", default="mmlu_pro_lc_paperC_heal_results")
    ap.add_argument("--olmo_root", default="mmlu_pro_letter_content_results")
    ap.add_argument("--xf_root", default="mmlu_pro_lc_crossfamily_results_fix")
    args = ap.parse_args()
    roots = {"heal": args.heal_root, "olmo": args.olmo_root, "xf": args.xf_root}

    cells = []
    selftest = None
    for rk, arm, label, step, fam, reg, keep, fresh in CELLS:
        root = roots[rk]
        if not _is_scored(root, arm):
            print(f"[skip] {root}/{arm}: not scored")
            continue
        c = analyse(root, arm, label, step, fam, reg, keep, fresh)
        if selftest is None:
            # gold/n_opt are DATASET properties (asserted identical across cells
            # via the bit-identical floor), so one cell suffices to run the
            # invariance self-test.
            recs, _ = _integrity(root, arm, keep, fresh)
            pred, gold, strat, keys = _encode(recs)
            nopt = np.array([r["n_opt"] for r in recs], dtype=np.int64)
            selftest = selftest_collapse_invariance(gold, nopt, strat, len(keys))
            print()
        cells.append(c)

    # ---- PASS 2: materiality, against each family's own intact anchor.
    anchors = _apply_materiality(cells)

    # ---- G4 (D4-analogue): the null must not sit above the whole population.
    n_above = sum(1 for c in cells if c["verdict_v2"] == "ITEM_LEVEL_SIGNAL")
    n_trace = sum(1 for c in cells if c["verdict_v2"] == "TRACE_SIGNAL")
    n_unanch = sum(1 for c in cells if c["verdict_v2"] == "SIGNAL_NOT_ANCHORED")
    g4_ok = n_above > 0

    out = {
        "what": "paperC read-out v2: prediction-permutation null, invariant to "
                "WHICH letter a degenerate arm collapses onto",
        "prereg_doc": "paperC/READOUT_V2_PREREGISTRATION.md",
        "estimator_source": ESTIMATOR_SOURCE,
        "null_definition": "acc_hat = E[acc | pred permuted uniformly at random "
                           "WITHIN n_opt strata] = sum_s (1/(n*n_s)) sum_L "
                           "cnt_pred[s,L]*cnt_gold[s,L]; Delta_perm = acc - acc_hat",
        "invariance_identity": "for a pure always-<L> emitter acc == acc_hat == "
                               "m_L so Delta_perm == 0 EXACTLY for every L; the "
                               "criterion cannot be satisfied by letter collapse",
        "v1_null_retained_for": "instrument-validity claims (paperC headline). "
                                "always-A 0.116606, arm-independent. NOT repealed.",
        "n_perm": N_PERM, "n_boot": N_BOOT, "seed": PERM_SEED,
        "G4_null_admissible": g4_ok,
        "G4_n_cells_with_signal": n_above,
        "n_cells_trace_signal": n_trace,
        "n_cells_signal_not_anchored": n_unanch,
        "material_frac_of_intact": MATERIAL_FRAC_OF_INTACT,
        "min_anchor_recovery": MIN_ANCHOR_RECOVERY,
        "materiality_rule":
            "material <=> recovery_fraction >= 0.10 * recovery_fraction(intact, "
            "SAME family), i.e. A04_MARGIN_GUARD_PREREG.md §4's constant applied "
            "within family. Significance alone is not capability at n=12032.",
        "family_anchors": {f: {"label": a["label"],
                               "recovery_fraction": a["recovery_fraction"],
                               "delta_perm_pp": a["delta_perm_pp"],
                               "admissible_anchor":
                                   bool(a["recovery_fraction"] >= MIN_ANCHOR_RECOVERY)}
                           for f, a in anchors.items()},
        "selftest_collapse_invariance": selftest,
        "n_cells": len(cells),
        "cells": cells,
    }
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"estimators: {ESTIMATOR_SOURCE}")
    print(f"n_perm={N_PERM} n_boot={N_BOOT} seed={PERM_SEED}\n")
    hdr = (f"{'label':34s} {'acc':>8s} {'null':>8s} {'d_perm':>8s} {'hw':>6s} "
           f"{'bootp':>7s} {'cap':>7s} {'recov':>7s} {'rel':>7s} "
           f"{'verdict_v2':>21s} {'|':>1s} {'d_v1':>7s} {'v1':>11s} "
           f"{'modal':>5s} {'share':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        print(f"{c['label']:34s} {c['letter_acc']:8.6f} "
              f"{c['null_perm_stratified']:8.6f} {c['delta_perm_pp']:+8.3f} "
              f"{c['ci95_half_width_pp']:6.3f} {c['boot_p']:7.4f} "
              f"{c['capacity_delta_max_pp']:7.3f} "
              f"{c['recovery_fraction']:7.4f} {c['relative_recovery']:7.3f} "
              f"{c['verdict_v2']:>21s} | {c['v1_delta_vs_floor_pp']:+7.3f} "
              f"{c['v1_verdict']:>11s} {c['modal_pred_letter']:>5s} "
              f"{c['modal_pred_share']:6.3f}")
    print()
    print("family intact anchors (G6: admissible iff recovery_fraction >= "
          f"{MIN_ANCHOR_RECOVERY}):")
    for f, a in anchors.items():
        ok = a["recovery_fraction"] >= MIN_ANCHOR_RECOVERY
        print(f"  {f:14s} recov={a['recovery_fraction']:.4f} "
              f"d_perm={a['delta_perm_pp']:+.3f} pp  "
              f"{'ADMISSIBLE' if ok else 'NOT COMPETENT -> relative claims blocked'}")
    print()

    # hard post-conditions
    for c in cells:
        assert c["integrity"]["n_scored"] == PREREG_N
        assert c["integrity"]["n_nan"] == 0
        assert c["integrity"]["n_trunc"] == 0
        assert c["integrity"]["chat_template"] is False, \
            f"{c['arm_dir']}: chat_template must be exactly False"
    assert g4_ok, "G4: no cell clears the permutation null -> null inadmissible"

    print("INTEGRITY OK for all %d cells: shard set {0..7}, n==%d, 0 dup, "
          "0 nan, 0 trunc, chat_template is False" % (len(cells), PREREG_N))
    # G5 coherence: does v2 change the scientific reading of a cell? "v1 says
    # the arm is distinguishable from the floor" vs "v2 says the arm carries
    # material item-level signal". TRACE_SIGNAL counts as NO capability.
    v2_capable = {"ITEM_LEVEL_SIGNAL"}
    v1_capable = {"ABOVE floor"}
    flips = [c for c in cells if (c["v1_verdict"] in v1_capable)
             != (c["verdict_v2"] in v2_capable)]
    print(f"G5 coherence: {len(flips)}/{len(cells)} cells change capability "
          f"reading between v1 and v2")
    for c in flips:
        print(f"  FLIP {c['label']:34s} v1={c['v1_verdict']:>11s} "
              f"({c['v1_delta_vs_floor_pp']:+.3f} pp) -> "
              f"v2={c['verdict_v2']} ({c['delta_perm_pp']:+.3f} pp)")
    print(f"n_signal={n_above} n_trace={n_trace} n_unanchored={n_unanch} "
          f"G4_admissible={g4_ok}")
    for c in cells:
        assert c["verdict_v2"] != "SIGNAL_PENDING_MATERIALITY", \
            f"{c['label']}: pass-2 materiality never applied"
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
