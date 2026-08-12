#!/usr/bin/env python3
"""paperC heal-confound -- the DEGENERACY DECOMPOSITION.

The trajectory read-out (heal_trajectory_nulls.py) says every healed milestone is
"AT floor". This script asks the question that makes that reading interpretable
or not: IS "AT FLOOR" A SIGN OF RECOVERED COMPETENCE, OR IS IT AN ARTEFACT OF
*WHICH LETTER* A DEGENERATE MODEL COLLAPSES ONTO?

The mechanism. `always-<L>` accuracy is a pure dataset property (the gold-letter
marginal), and on MMLU-Pro it is NOT flat:
    A .1166  B .1124  D .1110  C .1092  G .0982  E .0955  F .0938  H .0927
    I .0921  J .0785
Spread A-to-J = 3.81 pp. So a model that emits one letter almost always scores
that letter's marginal -- and the *identity* of the letter, not the presence or
absence of competence, decides whether it lands AT the floor (always-A, the
argmax = the floor by construction) or SIGNIFICANTLY BELOW it (always-E, -2.11 pp).

Test: predict each cell's letter accuracy using ONLY its prediction histogram and
the dataset marginal -- no competence term at all:
    acc_hat = sum_L  P(pred = L) * P(gold = L)
This is what the cell would score if its predictions were INDEPENDENT of the
gold answer. Residual = actual - acc_hat is the part attributable to actual
item-level signal. A near-zero residual means the cell is fully explained as
"degenerate emitter + dataset marginal".
"""
import collections
import json
import os
import sys

L = "ABCDEFGHIJ"
ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2]

CASES = [
    ("qwen3_heal@5000", "mmlu_pro_lc_paperC_heal_results", "qwen3base_heal_k8f2_step5000", "prune+heal 5000", 5000),
    ("qwen3_heal@5500", "mmlu_pro_lc_paperC_heal_results", "qwen3base_heal_k8f2_step5500", "prune+heal 5500", 5500),
    ("qwen3_heal@6000", "mmlu_pro_lc_paperC_heal_results", "qwen3base_heal_k8f2_step6000", "prune+heal 6000", 6000),
    ("qwen3_heal@6500", "mmlu_pro_lc_paperC_heal_results", "qwen3base_heal_k8f2_step6500", "prune+heal 6500", 6500),
    ("qwen3_heal@7000", "mmlu_pro_lc_paperC_heal_results", "qwen3base_heal_k8f2_step7000", "prune+heal 7000", 7000),
    ("olmo2_heal@45000", "mmlu_pro_lc_paperC_heal_results", "7B_keep8_step45000", "prune+heal 45000", 45000),
    ("olmo2_heal@121000", "mmlu_pro_letter_content_results", "7B_keep8_step121000", "prune+heal 121000", 121000),
    ("qwen3_k8_UNHEALED", "mmlu_pro_lc_crossfamily_results_fix", "qwen3_8b_base_k8", "trunc, NO heal", 0),
    ("qwen3_INTACT", "mmlu_pro_lc_crossfamily_results_fix", "qwen3_8b_base_base", "intact", None),
    ("olmo2_INTACT", "mmlu_pro_letter_content_results", "7B_base", "intact", None),
]


def load(root, arm):
    p = os.path.join(ROOT, root, arm, "per_example_mmlu_pro.jsonl")
    return [json.loads(x) for x in open(p)]


marg = None
rows = []
for name, root, arm, regime, step in CASES:
    recs = load(root, arm)
    n = len(recs)
    assert n == 12032, (name, n)
    m = {l: sum(1 for r in recs if r["gold_letter"] == l) / n for l in L}
    if marg is None:
        marg = m
    else:
        assert all(abs(m[l] - marg[l]) < 1e-12 for l in L), \
            f"{name}: gold marginal drifted -> item sets differ, not comparable"

    hist = collections.Counter(L[r["letter"]["pred"]] for r in recs)
    acc = sum(1 for r in recs if r["letter"]["correct"]) / n
    # independence prediction: no competence term whatsoever
    acc_hat = sum((hist[l] / n) * marg[l] for l in L)
    modal, mc = hist.most_common(1)[0]
    rows.append({
        "cell": name, "regime": regime, "heal_steps": step,
        "letter_acc": acc,
        "modal_pred_letter": modal, "modal_pred_share": mc / n,
        "n_distinct_pred_letters": len(hist),
        "always_modal_acc": marg[modal],
        "acc_hat_independence": acc_hat,
        "residual_pp": 100 * (acc - acc_hat),
        "delta_vs_floor_pp": 100 * (acc - marg["A"]),
        "pred_hist": dict(sorted(hist.items())),
    })

out = {
    "what": "paperC heal-confound: is 'AT floor' recovered competence, or an "
            "artefact of WHICH letter a degenerate model collapses onto?",
    "mechanism": "always-<L> accuracy is a dataset property and is NOT flat on "
                 "MMLU-Pro (A .1166 ... J .0785, spread 3.81 pp). A degenerate "
                 "emitter scores its letter's marginal, so the letter IDENTITY "
                 "decides AT-floor (always-A == the floor by construction) vs "
                 "BELOW-floor (always-E = -2.11 pp), with no competence involved.",
    "independence_model": "acc_hat = sum_L P(pred=L)*P(gold=L); residual = "
                          "actual - acc_hat is the item-level-signal part.",
    "gold_letter_marginal": marg,
    "always_letter_acc_sorted": dict(sorted(marg.items(), key=lambda kv: -kv[1])),
    "marginal_spread_A_to_J_pp": 100 * (marg["A"] - marg["J"]),
    "always_E_minus_always_A_pp": 100 * (marg["E"] - marg["A"]),
    "cells": rows,
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)

hdr = (f"{'cell':20s} {'regime':18s} {'letter':>8s} {'modal':>5s} {'share':>6s} "
       f"{'nL':>3s} {'acc_hat':>8s} {'resid_pp':>9s} {'d_floor':>8s}")
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['cell']:20s} {r['regime']:18s} {r['letter_acc']:8.6f} "
          f"{r['modal_pred_letter']:>5s} {r['modal_pred_share']:6.3f} "
          f"{r['n_distinct_pred_letters']:3d} {r['acc_hat_independence']:8.6f} "
          f"{r['residual_pp']:+9.3f} {r['delta_vs_floor_pp']:+8.3f}")
print(f"\nwrote {OUT}")
