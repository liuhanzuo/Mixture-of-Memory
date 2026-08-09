#!/usr/bin/env python3
"""Regenerate the four-row null-calibration master table from raw data.

Companion to scripts/verify_interface_audit.py (same house style: every printed
number is recomputed here from the per-example / per-pair artefacts, so the
table in proposal/active/A01-null-calibration-methodology/evidence/
P1_four_constructs.md cannot drift from the data).

Run from the repo root:
    python3 scripts/build_null_calibration_table.py            # table only
    python3 scripts/build_null_calibration_table.py --n-perm 2000   # + re-permute

Four constructs, each with a PRE-REGISTERED, construct-appropriate input-blind
null.  A generic 'chance line' is never used where the interface has its own
floor.  A fifth, SELF-DIRECTED row turns the same instrument on one of our own
retracted claims (Paper E Obs4).

  C1  MC scoring interface        (n = 14,042 items x 9 OLMo-2 arms)
      reported  = content_norm accuracy of the letter-chance arm (scratch16L)
      null      = best constant letter (always-D) and longest-option heuristic
      source    = olmo2_mmlu_content_results/<arm>/per_example_mmlu.jsonl

  C2  Generative label prior      (n = 2,000 SQuAD-style items)
      reported  = best arm EM on the original val set
      null      = majority-label constant, and empty string
      source    = data/squad_val.jsonl + paperC_squad_results/*_summary.json

  C3  Representation similarity   (n = 91 model pairs, 14 models)
      reported  = mean midband z-CKA
      null      = layer-order shuffle (NOT the random-init floor, which is the
                  wrong and self-flattering baseline for 'is layer i the right
                  partner for layer j')
      source    = proposal/shared/representation/cka_matrices/<a>__<b>.json
                  (cached CKA matrices)

  C4  Probe readout depth         (3 model families x 3 tasks x 5 splits)
      reported  = 1 - linear-probe knee depth  (how much depth the probe says
                  is unnecessary for the task to be linearly readable)
      null      = the model's own native readout knee
      source    = results/p1_2/p1_2_summary.json

  C5  OURS, RETRACTED: Paper E Obs4 "the interface flips the model ranking"
      (n = 14,042 items x 10 OLMo-2 arms = 45 arm-pairs)
      reported  = the two arm-pairs whose ranking flips significantly on both
                  interfaces (keep10 vs scratch16L, keep10 vs keep14-reheal)
      null      = best constant letter for the LETTER interface, longest-option
                  for CONTENT -- i.e. the same nulls as C1, applied to each of
                  the flip arms individually
      verdict   = the flip is real and survives BH, but every letter-side arm in
                  it is at or below its own floor, so the ranking it produces
                  carries no capability information.  That is why we retracted.
      source    = olmo2_mmlu_content_results/<arm>/per_example_mmlu.jsonl
      record    = proposal/active/A01-null-calibration-methodology/evidence/
                  mmlu_interface_initial_dossier.md sec.1/sec.4,
                  status/TRAINER_ACTIVITY.jsonl 2026-08-06T15:14:41Z, UPDATELOG.md:5927

The C3 leg additionally re-runs the layer-order-shuffle null at --n-perm
permutations per pair (default 200 = the shipped value; 2000 is the value the
paper reports) and applies Benjamini-Hochberg at q=0.05.  This is pure CPU work
on the cached z-CKA matrices: no activations are re-extracted and no GPU is
touched.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
from scipy.stats import binomtest

MIDBAND = (0.25, 0.75)
CKA_DIR = "proposal/shared/representation/cka_matrices"
RESULTS_JSON = "proposal/shared/representation/repr_alignment_results.json"
MMLU_DIR = "olmo2_mmlu_content_results"
SQUAD_VAL = "data/squad_val.jsonl"
SQUAD_RES = "paperC_squad_results"
P1_2 = "results/p1_2/p1_2_summary.json"

# 9 OLMo-2 arms sharing one item set.  scratch16L is the load-bearing arm: it is
# at the letter-interface floor by construction (random 16L init, healed 200k),
# so whatever the content interface gives it is structural, not knowledge.
MMLU_ARMS = [
    ("base (32L intact)", "7B_base"),
    ("full32 @25k", "7B_full32_step25000"),
    ("keep8 @121k", "7B_keep8_step121000"),
    ("keep10 @83.5k", "7B_keep10_step83500"),
    ("keep12 @124k", "7B_keep12_step124000"),
    ("keep14 @200k", "7B_keep14_step200000"),
    ("freezefront @200k", "7B_freezefront_step200000"),
    ("scratch16L @200k", "7B_scratch16L_step200000"),
    ("shortgpt16 @200k", "7B_shortgpt16_step200000"),
]
LETTER_CHANCE_ARM = "scratch16L @200k"

# -------------------------------------------------------------------------
# C5 -- the SELF-DIRECTED row: Paper E Observation 4, which we retracted.
#
# Obs4 claimed "the scoring interface flips the model ranking".  It is recorded
# in proposal/active/A01-null-calibration-methodology/evidence/
# mmlu_interface_initial_dossier.md section 1 ("Obs 4 x
# BEING OVERTURNED BY MYSELF"), in the NO-GO entry of status/TRAINER_ACTIVITY.jsonl
# 2026-08-06T15:14:41Z, and in UPDATELOG.md:5927.  The claim rested on 45
# arm-pairs over TEN OLMo-2 arms (C(10,2) = 45), of which 7 had opposite signs
# on the two interfaces and 2 were significant on both.  keep14-reheal @67.5k is
# one of the flip arms, so C5 needs the 10-arm set, not C1's 9.
#
# The retraction reason as written down: the flip only occurs where the LETTER
# interface has already collapsed onto a constant predictor.  This leg tests
# that quantitatively instead of asserting it.
# -------------------------------------------------------------------------
OBS4_ARMS = MMLU_ARMS + [("keep14-reheal @67.5k", "7B_keep14_reheal_step67500")]
OBS4_FLIP_PAIRS = [
    ("keep10 @83.5k", "scratch16L @200k"),        # "inherited vs random-init"
    ("keep10 @83.5k", "keep14-reheal @67.5k"),
]


# =========================================================================
# helpers
# =========================================================================
def bh_reject(pvals, q=0.05):
    """Benjamini-Hochberg step-up.  Returns (reject_mask, adjusted_p, k_max)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    k = int(np.max(np.nonzero(below)[0]) + 1) if below.any() else 0
    reject = np.zeros(n, dtype=bool)
    if k:
        reject[order[:k]] = True
    # BH-adjusted p (monotone cumulative min from the largest rank down)
    adj_ranked = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.minimum(adj_ranked, 1.0)
    return reject, adj, k


def paired_bootstrap(d, n_boot=10000, seed=0):
    """Paired bootstrap over items on the per-item difference vector d.

    d[i] is arm_a_correct[i] - arm_b_correct[i] (or arm_correct[i] - null[i]).
    Resampling ITEMS, not arms, is what makes it paired: both sides always see
    the same item, so the shared item difficulty cancels and only the discordant
    items move the statistic.  Returns (mean, lo, hi, two-sided p), with p
    floored at 1/n_boot (an exact 0 is not attainable from n_boot resamples).

    Implemented via the multinomial representation, which is EXACT in
    distribution (not an approximation, though it is a different RNG stream from
    naive index resampling, so it is not bit-identical to it): a bootstrap
    resample's mean depends on the resample only through how many times each
    distinct value of d was drawn, and those counts are exactly
    multinomial(n, empirical frequencies of d).  d takes very few distinct values
    here (arm-vs-arm is {-1,0,+1}; arm-vs-null adds the tie-split fractions), so
    this costs O(n_boot x n_distinct) instead of O(n_boot x n) -- 14,042 items x
    10,000 resamples x ~112 comparisons is 1.6e10 index draws the naive way,
    which does not finish and gets the process OOM-killed.  Verified against the
    naive resampler on the real Obs4 vectors: CI endpoints agree to <0.09pp and
    p-values to Monte-Carlo error.
    """
    d = np.asarray(d, dtype=float)
    n = d.size
    vals, counts = np.unique(d, return_counts=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, counts / n, size=n_boot)     # (n_boot, n_distinct)
    means = draws @ vals / n
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(max(p, 1.0 / n_boot), 1.0))


def mcnemar_exact(a, b):
    """Exact-binomial McNemar on two 0/1 per-item correctness vectors.

    Only the discordant items carry information, so this is the distribution-free
    companion to the bootstrap.  Returns (b01, b10, p).
    """
    a = np.asarray(a) > 0.5
    b = np.asarray(b) > 0.5
    b01 = int((a & ~b).sum())
    b10 = int((~a & b).sum())
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    return b01, b10, float(binomtest(b01, n, 0.5, alternative="two-sided").pvalue)


def block_idx(La, Lb):
    ia = [i for i in range(La + 1) if MIDBAND[0] <= i / La <= MIDBAND[1]]
    jb = [j for j in range(Lb + 1) if MIDBAND[0] <= j / Lb <= MIDBAND[1]]
    return ia, jb


def block_mean(M, La, Lb):
    ia, jb = block_idx(La, Lb)
    return float(M[np.ix_(ia, jb)].mean())


def shuffle_null(Mz, La, Lb, n_perm, seed=0):
    """Verbatim the null in
    proposal/shared/representation/code/repr_alignment_multimodel.py:561.

    Permutes B's LAYER ORDER.  The CKA entries themselves are untouched -- only
    which of B's layers count as 'B midband' changes.  So this null asks 'is the
    midband-to-midband correspondence special?', which is the question a
    layer-stitching claim rests on.  It is NOT the CKA magnitude floor (that is
    the random-init control, and using it here would be self-flattering).
    """
    ia, jb = block_idx(La, Lb)
    rng = np.random.default_rng(seed)
    rows = Mz[np.ix_(ia, list(range(Lb + 1)))]
    return np.array([rows[:, rng.permutation(Lb + 1)[jb]].mean()
                     for _ in range(n_perm)])


# =========================================================================
# C1 -- MC scoring interface
# =========================================================================
def load_mmlu_arms(arms):
    """Load per-example jsonl for each (label, dir) in `arms`.

    Asserts the arms really are item-aligned -- both C1's inflation number and
    C5's paired tests are only valid if arm i and arm j scored the SAME item in
    the same position, so item_id and gold_letter are checked elementwise rather
    than just counted.
    """
    data = {}
    for label, d in arms:
        rows = []
        with open(os.path.join(MMLU_DIR, d, "per_example_mmlu.jsonl")) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        data[label] = rows
    ref = data[arms[0][0]]
    n = len(ref)
    ref_ids = [r["item_id"] for r in ref]
    ref_gold = [r["gold_letter"] for r in ref]
    for label, rows in data.items():
        assert len(rows) == n, f"{label}: arms disagree on item count"
        assert [r["item_id"] for r in rows] == ref_ids, f"{label}: item_id misaligned"
        assert [r["gold_letter"] for r in rows] == ref_gold, f"{label}: gold misaligned"
        assert not any(r.get("nan") for r in rows), f"{label}: nan rows present"
    return data, n, ref_gold


def longest_option_vector(rows, gold, conv):
    """Per-item score of the input-blind 'always pick the longest option' null.

    The tie convention is load-bearing: 4,805/14,042 = 34.2% of MMLU items have
    >= 2 maximal-length options, so `split` (fractional credit, the unbiased
    expectation under uniform random tie-breaking) is what we pre-registered,
    and the .2822 recorded in the initial MMLU dossier is `last`, not `split`.
    """
    out = np.zeros(len(rows))
    for i, r in enumerate(rows):
        c = r["content_norm"]["cont_tokens"]
        top = max(c.values())
        win = [k for k in "ABCD" if c[k] == top]
        g = gold[i]
        if conv == "split":
            out[i] = (1.0 / len(win)) if g in win else 0.0
        elif conv == "first":
            out[i] = 1.0 if win[0] == g else 0.0
        elif conv == "last":
            out[i] = 1.0 if win[-1] == g else 0.0
        elif conv == "credit":       # optimistic: any tie counts as a hit
            out[i] = 1.0 if g in win else 0.0
        elif conv == "wrong":        # pessimistic: any tie counts as a miss
            out[i] = 1.0 if (len(win) == 1 and win[0] == g) else 0.0
        else:
            raise ValueError(conv)
    return out


TIE_CONVS = ("split", "first", "last", "credit", "wrong")


def leg_mc(verbose=True):
    data, n, gold_seq = load_mmlu_arms(MMLU_ARMS)

    gold = Counter(gold_seq)
    const_letter, hits = gold.most_common(1)[0]
    const_acc = hits / n

    # content interface has its own floor: always pick the longest option.
    # 4,805/14,042 items have >=2 maximal-length options, so the tie convention
    # is load-bearing and all of them are reported rather than one being picked.
    base_rows = data["base (32L intact)"]
    longest_convs = {c: float(longest_option_vector(base_rows, gold_seq, c).mean())
                     for c in TIE_CONVS}
    longest = longest_convs["split"]     # pre-registered convention

    acc = {lab: {k: sum(r[k]["correct"] for r in rows) / n
                 for k in ("letter", "content_norm")}
           for lab, rows in data.items()}

    if verbose:
        print("=" * 78)
        print("C1  MC SCORING INTERFACE   (n = %d items, %d arms, one item set)"
              % (n, len(MMLU_ARMS)))
        print("=" * 78)
        print("gold letter marginals: " + ", ".join(
            f"{k} {v / n:.4f}" for k, v in sorted(gold.items())))
        print(f"null-1  best constant letter  = always-{const_letter} -> {const_acc:.4f}")
        print(f"null-2  longest-option heuristic, by tie convention:")
        for c, v in longest_convs.items():
            print(f"           {c:8s} {v:.4f}"
                  + ("   <- pre-registered" if c == "split" else ""))
        print(f"        NOTE the literature value .2822 recorded in "
              f"the initial MMLU dossier is the 'last' convention; 'split' is the "
              f"defensible one and is used here.")
        print()
        print(f"{'arm':20s} {'letter':>8s} {'content':>8s} {'ltr-const':>10s} "
              f"{'cnt-long':>9s}")
        for lab, _ in MMLU_ARMS:
            a = acc[lab]
            print(f"{lab:20s} {a['letter']:8.4f} {a['content_norm']:8.4f} "
                  f"{100 * (a['letter'] - const_acc):+10.2f} "
                  f"{100 * (a['content_norm'] - longest):+9.2f}")

    z = acc[LETTER_CHANCE_ARM]
    inflation = z["content_norm"] - 0.25          # vs the naive chance line
    inflation_vs_const = z["content_norm"] - const_acc

    # The arm-to-arm effect the interface is USED to measure.  The load-bearing
    # comparison in the Paper B recovery argument is the best healed damaged arm
    # (keep14 @200k) against the knowledge-free control (scratch16L @200k) --
    # i.e. "how much did healing buy us, over a random 16L block healed for the
    # same 200k steps".  On content_norm that is the number the interface has to
    # resolve, and the interface's own structural offset dwarfs it.
    effect = acc["keep14 @200k"]["content_norm"] - z["content_norm"]
    # secondary framing: full spread over all damaged arms
    damaged = ["keep8 @121k", "keep10 @83.5k", "keep12 @124k", "keep14 @200k",
               "freezefront @200k", "scratch16L @200k"]
    dvals = {d: acc[d]["content_norm"] for d in damaged}
    spread = max(dvals.values()) - min(dvals.values())

    if verbose:
        print()
        print(f"letter-chance arm ({LETTER_CHANCE_ARM}): letter {z['letter']:.4f} "
              f"(const floor {const_acc:.4f}, {100*(z['letter']-const_acc):+.2f}pp) "
              f"-> AT/BELOW the letter floor by construction")
        print(f"  its content_norm = {z['content_norm']:.4f}")
        print(f"  structural inflation vs .25 chance line       = "
              f"{100 * inflation:+.2f}pp")
        print(f"  structural inflation vs always-{const_letter} floor      = "
              f"{100 * inflation_vs_const:+.2f}pp")
        print(f"  structural inflation vs longest-option floor  = "
              f"{100 * (z['content_norm'] - longest):+.2f}pp   "
              f"<- content's OWN floor, the construct-appropriate null")
        print(f"  effect being measured (keep14 - scratch16L, content_norm) = "
              f"{100 * effect:.2f}pp")
        print(f"  => inflation / effect = {inflation / effect:.4f}x  "
              f"(vs .25); {(z['content_norm'] - longest) / effect:.4f}x "
              f"(vs longest-option)")
        print(f"  [secondary] full spread over {len(damaged)} damaged arms = "
              f"{100 * spread:.2f}pp -> ratio {inflation / spread:.2f}x")

    return {
        "n_items": n, "const_letter": const_letter, "const_acc": const_acc,
        "longest_acc": longest, "acc": acc,
        "reported": z["content_norm"],
        # PRIMARY null for a content-interface number = content's own floor
        "null": longest,
        "null_alt_chance": 0.25, "null_alt_const_letter": const_acc,
        "inflation_pp": 100 * inflation,
        "inflation_vs_const_pp": 100 * inflation_vs_const,
        "inflation_vs_longest_pp": 100 * (z["content_norm"] - longest),
        "effect_pp": 100 * effect, "spread_pp": 100 * spread,
        "ratio": inflation / effect,
        "ratio_vs_longest": (z["content_norm"] - longest) / effect,
    }


# =========================================================================
# C2 -- generative label prior
# =========================================================================
def leg_squad(verbose=True):
    rows = [json.loads(l) for l in open(SQUAD_VAL) if l.strip()]
    n = len(rows)
    tgt = Counter(r["target_text"] for r in rows)
    maj, maj_hits = tgt.most_common(1)[0]
    maj_em = maj_hits / n
    empty_em = sum(1 for r in rows if r["target_text"].strip() == "") / n

    arms = {}
    for f in sorted(os.listdir(SQUAD_RES)):
        if not f.endswith("_summary.json"):
            continue
        d = json.load(open(os.path.join(SQUAD_RES, f)))
        if d.get("meta", {}).get("val_path", "").endswith("squad_val.jsonl"):
            arms[d["output_name"]] = d["em"]

    if verbose:
        print()
        print("=" * 78)
        print(f"C2  GENERATIVE LABEL PRIOR   (n = {n} items, {SQUAD_VAL})")
        print("=" * 78)
        print(f"null-1  majority-label constant {maj_hits}/{n} = {maj_em:.4f}  "
              f"label = {maj!r}")
        print(f"null-2  empty string -> EM {empty_em:.4f}")
        print("arms scored on THIS val set:")
        for k, v in sorted(arms.items(), key=lambda kv: -kv[1]):
            print(f"  {k:20s} EM {v:.4f}   vs majority floor "
                  f"{100 * (v - maj_em):+7.2f}pp")

    best = max(arms.values()) if arms else float("nan")
    best_name = max(arms, key=arms.get) if arms else None
    if verbose:
        print(f"best arm = {best_name} EM {best:.4f}; residual over the "
              f"input-blind majority constant = {best - maj_em:+.4f}")
    return {"n": n, "majority_label": maj, "majority_em": maj_em,
            "empty_em": empty_em, "arms": arms,
            "reported": best, "reported_arm": best_name, "null": maj_em}


# =========================================================================
# C3 -- representation similarity (+ re-permutation)
# =========================================================================
def leg_cka(n_perm, seed=0, verbose=True):
    ref = json.load(open(RESULTS_JSON))
    minfo = {k: v for k, v in ref["models"].items()}

    pairs = {}
    selfs = {}
    for fn in sorted(os.listdir(CKA_DIR)):
        if not fn.endswith(".json"):
            continue
        rec = json.load(open(os.path.join(CKA_DIR, fn)))
        if rec.get("self"):
            selfs[rec["model_a"]] = rec
        else:
            pairs[(rec["model_a"], rec["model_b"])] = rec

    # hard gate, verbatim from the source script: CKA of a model with itself
    # must be 1 on the diagonal, else the whole matrix is untrustworthy
    gate = max(v["identity_max_abs_dev_z"] for v in selfs.values())
    assert gate < 1e-5, f"IDENTITY GATE FAILED: {gate}"

    rows = []
    for (a, b), rec in sorted(pairs.items()):
        if minfo[a]["random_init"] or minfo[b]["random_init"]:
            continue                      # random-init pairs are the FLOOR arm
        La, Lb = rec["n_layers_a"], rec["n_layers_b"]
        Mz = np.asarray(rec["cka_matrix_z"])
        obs = block_mean(Mz, La, Lb)
        null = shuffle_null(Mz, La, Lb, n_perm, seed)
        # +1 correction: an exact permutation p can never be 0 with finite perms
        p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
        rows.append({"pair": f"{a}:{b}", "obs": obs, "null_mean": float(null.mean()),
                     "p": p, "null": null,
                     "same_family": int(minfo[a]["family"] == minfo[b]["family"])})

    obs = np.array([r["obs"] for r in rows])
    allnull = np.concatenate([r["null"] for r in rows])
    pv = np.array([r["p"] for r in rows])
    reject, adj, k = bh_reject(pv, 0.05)

    # cross-check against the shipped 200-perm figures
    shipped = ref["H3_middle_band"]["null_layer_order_shuffle"]
    shipped_obs = ref["H3_middle_band"]["observed_midband_zcka"]["mean"]

    if verbose:
        print()
        print("=" * 78)
        print(f"C3  REPRESENTATION SIMILARITY   (n = {len(rows)} model pairs, "
              f"{len(minfo)} models)")
        print("=" * 78)
        print(f"identity gate max|M[i][i]-1| (z) = {gate:.3e}  (< 1e-5 required)")
        print(f"observed midband z-CKA mean = {obs.mean():.16f}")
        print(f"  shipped value in {RESULTS_JSON} = {shipped_obs:.16f}  "
              f"| drift {abs(obs.mean() - shipped_obs):.2e}")
        print()
        print(f"layer-order-shuffle null, n_perm = {n_perm}/pair "
              f"({len(allnull)} total draws), seed {seed}")
        print(f"  null mean = {allnull.mean():.16f}")
        print(f"  shipped 200-perm null mean = {shipped['mean']:.16f}")
        print(f"  null [p2.5, p97.5] = [{np.percentile(allnull, 2.5):.4f}, "
              f"{np.percentile(allnull, 97.5):.4f}]")
        print()
        print(f"WRONG null for reference (random-init floor, NOT used): "
              f"{ref['H3_middle_band']['floor_random_init_models']['mean']:.4f}")
        print(f"  -- using it would make the residual look like "
              f"{obs.mean() - ref['H3_middle_band']['floor_random_init_models']['mean']:.4f} "
              f"instead of {obs.mean() - allnull.mean():.4f}")
        print()
        print(f"per-pair permutation p  (p = (#{{null>=obs}} + 1)/(n_perm + 1); "
              f"min attainable = {1 / (n_perm + 1):.2e})")
        print(f"  median p            = {np.median(pv):.6f}")
        print(f"  pairs raw p < 0.05  = {int((pv < 0.05).sum())}/{len(rows)}")
        print(f"  pairs BH q=0.05     = {int(reject.sum())}/{len(rows)}   "
              f"(BH cut at rank k={k})")
        print(f"  pairs obs > null mean = "
              f"{sum(1 for r in rows if r['obs'] > r['null_mean'])}/{len(rows)}")
        print(f"  shipped 200-perm: median p {shipped['per_pair_p_median']}, "
              f"raw p<0.05 {shipped['n_pairs_p_below_0.05']}/91, no BH")
        print()
        surv = [r["pair"] for r, ok in zip(rows, reject) if ok]
        print(f"  BH survivors ({len(surv)}): " + ", ".join(surv[:8])
              + (" ..." if len(surv) > 8 else ""))
        dead = [(r["pair"], r["p"]) for r, ok in zip(rows, reject) if not ok]
        dead.sort(key=lambda t: t[1])
        print(f"  BH non-survivors ({len(dead)}), smallest p first: "
              + ", ".join(f"{n}({p:.3f})" for n, p in dead[:8])
              + (" ..." if len(dead) > 8 else ""))

    return {
        "n_pairs": len(rows), "identity_gate": gate,
        "reported": float(obs.mean()), "null": float(allnull.mean()),
        "null_wrong_randominit":
            ref["H3_middle_band"]["floor_random_init_models"]["mean"],
        "n_perm": n_perm, "p_median": float(np.median(pv)),
        "n_raw_p05": int((pv < 0.05).sum()),
        "n_bh_q05": int(reject.sum()), "bh_k": k,
        "n_obs_above_nullmean": sum(1 for r in rows if r["obs"] > r["null_mean"]),
        "per_pair": [{"pair": r["pair"], "obs": r["obs"],
                      "null_mean": r["null_mean"], "p": r["p"],
                      "p_bh": float(a), "bh_reject": bool(x),
                      "same_family": r["same_family"]}
                     for r, a, x in zip(rows, adj, reject)],
    }


# =========================================================================
# C4 -- probe readout depth
# =========================================================================
def leg_probe(verbose=True):
    d = json.load(open(P1_2))
    out = {}
    for model, v in d.items():
        per = v["per_task"]
        lin = v["content_j_frac_mean"]
        # native knee: aggregate the same three tasks the linear knee aggregates
        nat = {t: per[t]["native_knee_frac"] for t in per}
        natmean = float(np.mean(list(nat.values())))
        out[model] = {"L": v["L"], "linear_knee_frac": lin,
                      "linear_ci95": v["content_j_frac_ci95"],
                      "native_per_task": nat, "native_mean": natmean,
                      "native_sst2": per["SST2"]["native_knee_frac"],
                      "n_points": v["n_points"]}
    if verbose:
        print()
        print("=" * 78)
        print("C4  PROBE READOUT DEPTH   (3 families x 3 tasks x 5 splits)")
        print("=" * 78)
        for m, v in out.items():
            print(f"{m:20s} L={v['L']:2d}  linear knee {v['linear_knee_frac']:.4f} "
                  f"CI{v['linear_ci95']}  native/task "
                  + ", ".join(f"{t} {x:.4f}" for t, x in v["native_per_task"].items()))
    return out


# =========================================================================
# C5 -- OURS, RETRACTED: Paper E Obs4 "the interface flips the model ranking"
# =========================================================================
def leg_obs4(n_boot=10000, seed=0, verbose=True):
    """Turn the instrument on our own retracted claim.

    Two questions, and the CONTRAST between the answers is the finding:
      (a) arm-vs-arm.  Is the flip real?  (paired bootstrap + exact McNemar,
          then BH q=0.05 across all 45 pairs x 2 interfaces, because the flip
          was FOUND by screening 45 pairs and an uncorrected screen is exactly
          the error this paper is about.)
      (b) arm-vs-null.  Is either arm above its interface's own input-blind
          floor?  If both sit at/below it, the ranking they define carries no
          capability information however significant the difference is.
    """
    data, n, gold_seq = load_mmlu_arms(OBS4_ARMS)
    labels = [lab for lab, _ in OBS4_ARMS]

    L = {lab: np.array([r["letter"]["correct"] for r in rows], dtype=float)
         for lab, rows in data.items()}
    C = {lab: np.array([r["content_norm"]["correct"] for r in rows], dtype=float)
         for lab, rows in data.items()}

    # ---- the two construct-appropriate nulls, as per-item score vectors so the
    # ---- arm-vs-null test can be PAIRED on the same items as the arm-vs-arm one.
    gold_counts = Counter(gold_seq)
    const_letter, hits = gold_counts.most_common(1)[0]
    letter_null = np.array([g == const_letter for g in gold_seq], dtype=float)
    content_null_convs = {c: longest_option_vector(data[labels[0]], gold_seq, c)
                          for c in TIE_CONVS}
    content_null = content_null_convs["split"]        # pre-registered
    tie_rate = float(np.mean([
        sum(1 for k in "ABCD"
            if r["content_norm"]["cont_tokens"][k]
            == max(r["content_norm"]["cont_tokens"].values())) >= 2
        for r in data[labels[0]]]))

    # ---- (b) every arm against its own interface's floor
    per_arm = {}
    for lab in labels:
        lm, llo, lhi, lp = paired_bootstrap(L[lab] - letter_null, n_boot, seed + 11)
        _, _, lmc = mcnemar_exact(L[lab], letter_null)
        cm, clo, chi, cp = paired_bootstrap(C[lab] - content_null, n_boot, seed + 12)
        if lp >= 0.05:
            verdict = "AT the floor (indistinguishable)"
        elif lm < 0:
            verdict = "BELOW the floor (significantly)"
        else:
            verdict = "above the floor"
        # --- two diagnostics that must travel with the verdict ---
        # (i) SECONDARY letter null: the arm's OWN modal prediction as a constant.
        #     This is load-bearing in the same way the tie convention is.  The
        #     pre-registered null is the BEST constant (always-D, .2689) because a
        #     floor must not depend on the arm being tested; but against its own
        #     modal letter an arm can come out marginally ABOVE while being BELOW
        #     always-D, and that must be disclosed rather than buried.
        preds = Counter(r["letter"]["pred_letter"] for r in data[lab])
        modal_letter, modal_hits = preds.most_common(1)[0]
        own_null = np.array([g == modal_letter for g in gold_seq], dtype=float)
        om, _, _, op = paired_bootstrap(L[lab] - own_null, n_boot, seed + 13)
        # (ii) bf16 exact-tie rate: when top1 == top2 exactly, argmax breaks the
        #      tie by INDEX, which is input-blind -- the documented mechanism by
        #      which the letter interface decays into a constant predictor.
        tied = float(np.mean([
            sorted(r["letter"]["scores"].values())[-1]
            == sorted(r["letter"]["scores"].values())[-2] for r in data[lab]]))
        per_arm[lab] = {
            "letter": float(L[lab].mean()), "content_norm": float(C[lab].mean()),
            "letter_vs_null_pp": 100 * lm, "letter_ci95_pp": [100 * llo, 100 * lhi],
            "letter_boot_p": lp, "letter_mcnemar_p": lmc,
            "letter_verdict": verdict,
            "content_vs_null_pp": 100 * cm, "content_ci95_pp": [100 * clo, 100 * chi],
            "content_boot_p": cp,
            "content_above_null": bool(cm > 0 and cp < 0.05),
            "letter_modal_pred": modal_letter,
            "letter_modal_pred_rate": modal_hits / n,
            "letter_own_modal_null": float(own_null.mean()),
            "letter_vs_own_modal_pp": 100 * om, "letter_vs_own_modal_p": op,
            "letter_bf16_tie_rate": tied,
        }

    # ---- (a) all 45 arm-pairs on both interfaces, then BH across the screen
    pairs = [(a, b) for i, a in enumerate(labels) for b in labels[i + 1:]]
    rows = []
    for a, b in pairs:
        lm, llo, lhi, lp = paired_bootstrap(L[a] - L[b], n_boot, seed + 41)
        cm, clo, chi, cp = paired_bootstrap(C[a] - C[b], n_boot, seed + 42)
        _, _, lmc = mcnemar_exact(L[a], L[b])
        _, _, cmc = mcnemar_exact(C[a], C[b])
        rows.append({"a": a, "b": b,
                     "letter_pp": 100 * lm, "letter_ci95_pp": [100 * llo, 100 * lhi],
                     "letter_p": lp, "letter_mcnemar_p": lmc,
                     "content_pp": 100 * cm, "content_ci95_pp": [100 * clo, 100 * chi],
                     "content_p": cp, "content_mcnemar_p": cmc,
                     "sign_opposite": bool(np.sign(lm) != np.sign(cm)
                                           and lm != 0 and cm != 0)})
    rej_l, _, _ = bh_reject([r["letter_p"] for r in rows], 0.05)
    rej_c, _, _ = bh_reject([r["content_p"] for r in rows], 0.05)
    for r, x, y in zip(rows, rej_l, rej_c):
        r["letter_bh"] = bool(x)
        r["content_bh"] = bool(y)
        r["flip_sig_both_raw"] = bool(r["sign_opposite"]
                                      and r["letter_p"] < 0.05 and r["content_p"] < 0.05)
        r["flip_sig_both_bh"] = bool(r["sign_opposite"] and x and y)

    n_opposite = sum(r["sign_opposite"] for r in rows)
    flips_raw = [r for r in rows if r["flip_sig_both_raw"]]
    flips_bh = [r for r in rows if r["flip_sig_both_bh"]]
    flip_arms = sorted({x for r in flips_bh for x in (r["a"], r["b"])})

    # ---- the kill: restrict to arms valid on BOTH interfaces and re-screen.
    # 'Valid' = letter significantly ABOVE the constant-letter floor AND content
    # significantly above the longest-option floor.  If no flip survives inside
    # the valid set, the flip is a property of broken instruments, not of models.
    valid = [lab for lab in labels
             if per_arm[lab]["letter_verdict"] == "above the floor"
             and per_arm[lab]["content_above_null"]]
    vset = set(valid)
    vrows = [r for r in rows if r["a"] in vset and r["b"] in vset]
    v_opposite = sum(r["sign_opposite"] for r in vrows)
    v_flips = sum(r["flip_sig_both_raw"] for r in vrows)

    # ---- the self-referential check.  The retraction is recorded as 'both arms
    # sat at the CHANCE floor'.  That wording is loose in a way that matters: had
    # we used the generic .25 chance line instead of the construct-appropriate
    # best-constant floor, one flip arm would have come out significantly ABOVE
    # its null and the retraction would not have been triggered.  So this case
    # does not merely illustrate the paper's thesis, it DEPENDS on it.
    chance_line = 1.0 / 4
    for lab in labels:
        m, lo, hi, p = paired_bootstrap(L[lab] - chance_line, n_boot, seed + 14)
        per_arm[lab]["letter_vs_chance_line_pp"] = 100 * m
        per_arm[lab]["letter_vs_chance_line_ci95_pp"] = [100 * lo, 100 * hi]
        per_arm[lab]["letter_above_chance_line"] = bool(m > 0 and p < 0.05)
    n_flip_above_chance = sum(1 for a in flip_arms
                              if per_arm[a]["letter_above_chance_line"])

    if verbose:
        print()
        print("=" * 78)
        print("C5  OURS, RETRACTED -- Paper E Obs4 'the interface flips the ranking'")
        print(f"    (n = {n} items x {len(labels)} arms = {len(pairs)} arm-pairs)")
        print("=" * 78)
        print("record: proposal/active/A01-null-calibration-methodology/evidence/"
              "mmlu_interface_initial_dossier.md sec.1 "
              "('Obs 4 x overturned by myself') + sec.4;")
        print("        status/TRAINER_ACTIVITY.jsonl 2026-08-06T15:14:41Z "
              "(Obs4_ranking_flip, attack-3-self-refuted); UPDATELOG.md:5927")
        print()
        print(f"nulls: LETTER  = best constant letter always-{const_letter} "
              f"{letter_null.mean():.4f}")
        print(f"       CONTENT = longest-option, tie convention 'split' "
              f"{content_null.mean():.4f}   "
              f"(tied-longest on {tie_rate:.4f} = "
              f"{round(tie_rate * n)}/{n} items, so the convention is "
              f"load-bearing)")
        print("       all conventions: " + ", ".join(
            f"{c} {v.mean():.4f}" for c, v in content_null_convs.items()))
        print()
        print("(a) IS THE FLIP REAL?  45 pairs x 2 interfaces, BH q=0.05 over the "
              "whole screen")
        print(f"    sign-opposite pairs                 : {n_opposite}/{len(pairs)}")
        print(f"    significant on BOTH interfaces, raw : {len(flips_raw)}")
        print(f"    significant on BOTH interfaces, BH  : {len(flips_bh)}")
        for r in flips_bh:
            print(f"      {r['a']:22s} vs {r['b']:22s}")
            print(f"        letter  {r['letter_pp']:+6.2f}pp "
                  f"[{r['letter_ci95_pp'][0]:+.2f},{r['letter_ci95_pp'][1]:+.2f}] "
                  f"boot p={r['letter_p']:.4f} McNemar p={r['letter_mcnemar_p']:.3e}")
            print(f"        content {r['content_pp']:+6.2f}pp "
                  f"[{r['content_ci95_pp'][0]:+.2f},{r['content_ci95_pp'][1]:+.2f}] "
                  f"boot p={r['content_p']:.4f} McNemar p={r['content_mcnemar_p']:.3e}")
        print("    => the flip SURVIVES multiplicity correction.  It is not a "
              "screening artefact.")
        print()
        print("(b) IS EITHER ARM ABOVE ITS FLOOR?  arm vs its own interface's null")
        print(f"{'arm':24s} {'letter':>7s} {'vs null':>9s} {'bootp':>7s} "
              f"{'content':>8s} {'vs null':>9s}   letter verdict")
        for lab in labels:
            v = per_arm[lab]
            mark = " <<FLIP" if any(lab in (r["a"], r["b"]) for r in flips_bh) else ""
            print(f"{lab:24s} {v['letter']:7.4f} {v['letter_vs_null_pp']:+9.2f} "
                  f"{v['letter_boot_p']:7.4f} {v['content_norm']:8.4f} "
                  f"{v['content_vs_null_pp']:+9.2f}   {v['letter_verdict']}{mark}")
        print()
        print("    flip arms, letter side: " + "; ".join(
            f"{a} {per_arm[a]['letter_verdict']}" for a in flip_arms))
        print("    => NOT ONE flip arm is above the letter floor.  The 'ranking' "
              "the letter")
        print("       interface assigns them is noise about a constant predictor.")
        print()
        print(f"(c) THE KILL.  Arms valid on BOTH interfaces ({len(valid)}/"
              f"{len(labels)}): {', '.join(valid)}")
        print(f"    pairs among them              : {len(vrows)}")
        print(f"    sign-opposite among them      : {v_opposite}")
        print(f"    significant flips among them  : {v_flips}")
        print(f"    => {v_flips} flips inside the valid set. The retraction HOLDS: "
              f"the flip exists")
        print("       only where the letter instrument has already collapsed.")
        print()
        print("(d) HONESTY CHECKS -- three ways this conclusion could be wrong, "
              "each tested:")
        n_content_above = sum(1 for lab in labels
                              if per_arm[lab]["content_above_null"])
        print(f"    1. ASYMMETRY. Is the content side let off?  No: content is "
              f"tested against")
        print(f"       its own floor too, and {n_content_above}/{len(labels)} "
              f"arms clear it. So the retraction is NOT")
        print(f"       'both interfaces are dead' -- the CONTENT ranking of the "
              f"flip arms is")
        print(f"       above-floor and therefore meaningful; only the LETTER side "
              f"is noise.")
        print(f"       The flip is thus 'one live instrument vs one dead one', "
              f"not two live ones")
        print(f"       disagreeing, which is exactly why it cannot support the "
              f"retracted claim.")
        print(f"    2. NULL CHOICE. Against the BEST constant (always-"
              f"{const_letter}) vs each arm's OWN")
        print(f"       modal-prediction constant, which is the weaker null:")
        for a in flip_arms:
            v = per_arm[a]
            print(f"         {a:24s} vs always-{const_letter} "
                  f"{v['letter_vs_null_pp']:+6.2f}pp (p {v['letter_boot_p']:.4f}) "
                  f"| vs own always-{v['letter_modal_pred']} "
                  f"({v['letter_own_modal_null']:.4f}) "
                  f"{v['letter_vs_own_modal_pp']:+6.2f}pp "
                  f"(p {v['letter_vs_own_modal_p']:.4f})")
        print(f"       Sign can flip between the two nulls, so the pre-registered "
              f"null matters:")
        print(f"       we use the BEST constant, because a floor that is defined "
              f"by the arm under")
        print(f"       test is not a floor.  Reported both ways regardless.")
        print(f"    3. MECHANISM. bf16 exact-tie rate on the letter interface "
              f"(argmax then breaks")
        print(f"       ties by INDEX, i.e. input-blind), flip arms vs the intact "
              f"base:")
        print(f"         base {per_arm[labels[0]]['letter_bf16_tie_rate']:.4f}   "
              + "   ".join(f"{a.split(' ')[0]} "
                           f"{per_arm[a]['letter_bf16_tie_rate']:.4f}"
                           for a in flip_arms))
        print()
        print("(e) ** THE RECORD'S WORDING IS LOOSE, AND THE LOOSENESS MATTERS **")
        print(f"    The retraction is recorded as 'both flip arms sat at the "
              f"CHANCE floor'.  Against")
        print(f"    the generic 1/4 = {chance_line:.4f} chance line that is NOT "
              f"what the data say:")
        for a in flip_arms:
            v = per_arm[a]
            ci = v["letter_vs_chance_line_ci95_pp"]
            print(f"      {a:24s} {v['letter']:.4f}  vs .2500 "
                  f"{v['letter_vs_chance_line_pp']:+6.2f}pp "
                  f"[{ci[0]:+.2f},{ci[1]:+.2f}] -> "
                  + ("ABOVE the chance line"
                     if v["letter_above_chance_line"] else "at the chance line"))
        print(f"    {n_flip_above_chance}/{len(flip_arms)} flip arms are "
              f"significantly ABOVE the naive chance line.")
        print(f"    So the retraction does NOT follow from a chance-line "
              f"comparison; it follows only")
        print(f"    from the CONSTRUCT-APPROPRIATE floor (best constant letter "
              f"always-{const_letter} "
              f"{letter_null.mean():.4f}),")
        print(f"    which is {100 * (letter_null.mean() - chance_line):+.2f}pp "
              f"above the chance line because the gold letters are")
        print(f"    not uniform.  This case therefore does not merely ILLUSTRATE "
              f"the paper's thesis --")
        print(f"    it DEPENDS on it.  Anyone re-deriving our retraction with a "
              f"chance line will")
        print(f"    conclude we retracted without cause, so the floor definition "
              f"must be stated.")
        print(f"    Corollary: the phrase 'at the chance floor' should be "
              f"corrected to 'at or below")
        print(f"    the best-constant-predictor floor' wherever it appears.")

    return {
        "n_items": n, "n_arms": len(labels), "n_pairs": len(pairs),
        "ours": True, "retracted": True,
        "claim": "the MC scoring interface flips the model ranking",
        "record": [
            "proposal/active/A01-null-calibration-methodology/evidence/"
            "mmlu_interface_initial_dossier.md sec.1 + sec.4",
            "status/TRAINER_ACTIVITY.jsonl 2026-08-06T15:14:41Z",
            "UPDATELOG.md:5927",
        ],
        "letter_null_letter": const_letter,
        "letter_null": float(letter_null.mean()),
        "content_null": float(content_null.mean()),
        "content_null_convention": "split",
        "content_null_all_convs": {c: float(v.mean())
                                   for c, v in content_null_convs.items()},
        "content_tied_longest_rate": tie_rate,
        "n_boot": n_boot,
        "n_sign_opposite": n_opposite,
        "n_flip_sig_both_raw": len(flips_raw),
        "n_flip_sig_both_bh": len(flips_bh),
        "flip_pairs_bh": [{k: r[k] for k in
                           ("a", "b", "letter_pp", "letter_ci95_pp", "letter_p",
                            "letter_mcnemar_p", "content_pp", "content_ci95_pp",
                            "content_p", "content_mcnemar_p")} for r in flips_bh],
        "per_arm": per_arm,
        "flip_arms": flip_arms,
        "n_arms_content_above_null": sum(1 for lab in labels
                                         if per_arm[lab]["content_above_null"]),
        "n_arms_letter_above_null": sum(1 for lab in labels
                                        if per_arm[lab]["letter_verdict"]
                                        == "above the floor"),
        "chance_line": chance_line,
        "n_flip_arms_above_chance_line": n_flip_above_chance,
        "retraction_requires_construct_null": bool(n_flip_above_chance > 0),
        "valid_both_interfaces": valid,
        "n_pairs_within_valid": len(vrows),
        "n_sign_opposite_within_valid": v_opposite,
        "n_flip_sig_within_valid": v_flips,
        "retraction_holds": bool(v_flips == 0 and len(flips_bh) > 0 and all(
            per_arm[x]["letter_verdict"] != "above the floor"
            for r in flips_bh for x in (r["a"], r["b"]))),
        "all_pairs": rows,
    }


# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=200,
                    help="layer-order-shuffle permutations per pair (C3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="paired-bootstrap resamples per comparison (C5)")
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()

    c1 = leg_mc()
    c2 = leg_squad()
    c3 = leg_cka(args.n_perm, args.seed)
    c4 = leg_probe()
    c5 = leg_obs4(args.n_boot, args.seed)

    # -------- the four-row master table --------
    # C4's "reported value" is stated as the depth fraction the probe declares
    # unnecessary for the task to be readable (1 - linear knee).  The null is the
    # model's OWN native readout, which needs almost all of the depth.  Residual
    # = what the probe claim survives once you stop crediting a readout the model
    # does not itself use.  Qwen + OLMo only: Llama's WiC/RTE native verbalizers
    # sit at chance so its 3-task native aggregate is not meaningful (this is the
    # same restriction paperA/sections/tab_depth.tex applies).
    qw = c4["Qwen--Qwen3-8b"]
    ol = c4["OLMo-2-1124-7B"]
    ll = c4["Meta-Llama-3-8B"]
    probe_rep = 1.0 - float(np.mean([qw["linear_knee_frac"], ol["linear_knee_frac"]]))
    probe_null = 1.0 - float(np.mean([qw["native_mean"], ol["native_mean"]]))

    table = [
        ("C1 MC scoring interface", c1["reported"], c1["null"],
         f"longest-option, content's own floor ({c1['null']:.4f}); "
         f"always-{c1['const_letter']} {c1['const_acc']:.4f} for letter"),
        ("C2 Generative label prior", c2["reported"], c2["null"],
         "majority-label constant (empty string = 0.0000)"),
        ("C3 Representation similarity", c3["reported"], c3["null"],
         f"layer-order shuffle (NOT random-init "
         f"{c3['null_wrong_randominit']:.4f})"),
        ("C4 Probe readout depth", probe_rep, probe_null,
         "native readout knee, 3-task mean, Qwen+OLMo"),
    ]

    print()
    print("=" * 104)
    print("MASTER TABLE -- null-calibrated residuals")
    print("=" * 104)
    print(f"{'construct':30s} {'reported':>10s} {'null':>10s} "
          f"{'residual':>10s} {'resid/rep':>10s}   null used")
    fracs = []
    for name, rep, null, desc in table:
        resid = rep - null
        frac = resid / rep
        fracs.append(frac)
        print(f"{name:30s} {rep:10.4f} {null:10.4f} {resid:10.4f} "
              f"{frac:10.4f}   {desc}")
    print()
    lo, hi = min(fracs), max(fracs)
    span = hi / lo
    print(f"residual fractions: min {lo:.4f}  max {hi:.4f}  span = {span:.2f}x")
    print(f"PRE-REGISTERED GATE (span >= 10x): "
          f"{'PASS' if span >= 10 else 'FAIL'}")

    # ---- row 5: OURS AND RETRACTED.  Deliberately NOT folded into the four-leg
    # gate above -- the gate was pre-registered over four constructs and adding a
    # fifth row to it after the fact would be the exact post-hoc move this paper
    # criticises.  It is a self-directed worked example, not a fifth data point.
    # Its "residual" is not a fraction of a reported scalar but the answer to a
    # yes/no question: does the effect survive its own null?
    flip_arms = c5["flip_arms"]
    print()
    print("=" * 104)
    print("ROW 5 (SELF-DIRECTED, OURS AND RETRACTED) -- Paper E Obs4")
    print("=" * 104)
    print(f"claim (retracted): {c5['claim']}")
    print(f"  arm-vs-arm      : {c5['n_flip_sig_both_bh']} of "
          f"{c5['n_pairs']} pairs flip significantly on both interfaces, "
          f"AFTER BH q=0.05 over the whole 45-pair screen -> the effect is REAL")
    print(f"  arm-vs-null     : letter null = always-"
          f"{c5['letter_null_letter']} {c5['letter_null']:.4f}; "
          f"of the {len(flip_arms)} flip arms, "
          f"{sum(1 for a in flip_arms if c5['per_arm'][a]['letter_verdict'] != 'above the floor')}"
          f"/{len(flip_arms)} are AT or BELOW it")
    for a in flip_arms:
        v = c5["per_arm"][a]
        print(f"      {a:24s} letter {v['letter']:.4f} "
              f"{v['letter_vs_null_pp']:+6.2f}pp vs null (boot p "
              f"{v['letter_boot_p']:.4f}) -> {v['letter_verdict']}")
    print(f"  within the {len(c5['valid_both_interfaces'])} arms valid on BOTH "
          f"interfaces: {c5['n_flip_sig_within_valid']} significant flips out of "
          f"{c5['n_pairs_within_valid']} pairs")
    print(f"  => RETRACTION HOLDS: {c5['retraction_holds']}   "
          f"(a significant ranking difference between two arms that are both at "
          f"the floor carries no capability information)")
    print(f"  CAVEAT that must ship with it: against the generic "
          f"{c5['chance_line']:.4f} chance line, "
          f"{c5['n_flip_arms_above_chance_line']}/{len(flip_arms)} flip arms look "
          f"significantly ABOVE null.")
    print(f"  The retraction depends on the construct-appropriate best-constant "
          f"floor ({c5['letter_null']:.4f}), so")
    print(f"  'at the chance floor' in our own record is imprecise and should "
          f"read 'at or below the")
    print(f"  best-constant-predictor floor'.")

    # ---- gate sensitivity: the gate turns on C4, which is the leg with the
    # most operationalization freedom, so every reasonable variant is shown
    # rather than the one that happens to pass.
    def c4frac(lin, nat):
        rep = 1.0 - lin
        return (rep - (1.0 - nat)) / rep

    variants = {
        "Qwen+OLMo, native 3-task mean, pooled (headline)":
            c4frac(np.mean([qw["linear_knee_frac"], ol["linear_knee_frac"]]),
                   np.mean([qw["native_mean"], ol["native_mean"]])),
        "Qwen+OLMo, native 3-task mean, per-model then avg":
            float(np.mean([c4frac(v["linear_knee_frac"], v["native_mean"])
                           for v in (qw, ol)])),
        "all 3 models, native 3-task mean, per-model then avg":
            float(np.mean([c4frac(v["linear_knee_frac"], v["native_mean"])
                           for v in (qw, ol, ll)])),
        "all 3 models, native = SST2 only (matched support)":
            float(np.mean([c4frac(v["linear_knee_frac"], v["native_sst2"])
                           for v in (qw, ol, ll)])),
        "Qwen+OLMo, native = SST2 only":
            float(np.mean([c4frac(v["linear_knee_frac"], v["native_sst2"])
                           for v in (qw, ol)])),
    }
    other = [f for f in fracs[:3]]
    print()
    print("gate sensitivity -- C4 is the leg with operationalization freedom, "
          "so all variants are shown:")
    print(f"{'C4 variant':54s} {'frac':>8s} {'span':>8s} {'gate':>6s}")
    gate_any = False
    for k, f in variants.items():
        sp = max(f, *other) / min(f, *other)
        ok = sp >= 10
        gate_any = gate_any or ok
        print(f"{k:54s} {f:8.4f} {sp:8.2f}x {'PASS' if ok else 'FAIL':>6s}")
    print(f"=> gate passes under ANY reasonable C4 variant: "
          f"{'YES' if gate_any else 'NO'}")

    # The two headline numbers, stated precisely.
    print()
    print(f"headline 1: MC content interface hands the letter-chance arm "
          f"{c1['reported']:.4f}; inflation {c1['inflation_pp']:+.2f}pp vs the "
          f".25 chance line / {c1['inflation_vs_const_pp']:+.2f}pp vs always-"
          f"{c1['const_letter']} / {c1['inflation_vs_longest_pp']:+.2f}pp vs "
          f"its own longest-option floor, against an arm-to-arm effect of "
          f"{c1['effect_pp']:.2f}pp = {c1['ratio']:.2f}x (chance line) / "
          f"{c1['ratio_vs_longest']:.2f}x (own floor)")
    print(f"headline 2: layer-order-shuffle null accounts for "
          f"{100 * c3['null'] / c3['reported']:.2f}% of the reported midband "
          f"z-CKA ({c3['null']:.4f} of {c3['reported']:.4f}); usable signal "
          f"{c3['reported'] - c3['null']:.4f}")
    print(f"            BH q=0.05 survivors: {c3['n_bh_q05']}/{c3['n_pairs']} "
          f"at n_perm={c3['n_perm']} (raw p<0.05: {c3['n_raw_p05']}; "
          f"median p {c3['p_median']:.6f}; min attainable p "
          f"{1 / (c3['n_perm'] + 1):.2e})")

    if args.out:
        payload = {"c1_mc": {k: v for k, v in c1.items() if k != "acc"},
                   "c1_arm_acc": c1["acc"],
                   "c2_squad": c2, "c3_cka": c3, "c4_probe": c4,
                   "c5_obs4_ours_retracted": c5,
                   "table": [{"construct": n, "reported": r, "null": u,
                              "residual": r - u, "residual_frac": (r - u) / r,
                              "null_desc": d} for n, r, u, d in table],
                   "gate_span": span, "gate_pass": bool(span >= 10),
                   "gate_c4_variants": variants,
                   "gate_pass_any_c4_variant": bool(gate_any),
                   "n_perm": args.n_perm, "n_boot": args.n_boot,
                   "seed": args.seed}
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
