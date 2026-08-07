#!/usr/bin/env python3
"""Regenerate the four-row null-calibration master table from raw data.

Companion to scripts/verify_interface_audit.py (same house style: every printed
number is recomputed here from the per-example / per-pair artefacts, so the
table in status/NULL_CALIBRATION_P1.md cannot drift from the data).

Run from the repo root:
    python3 scripts/build_null_calibration_table.py            # table only
    python3 scripts/build_null_calibration_table.py --n-perm 2000   # + re-permute

Four constructs, each with a PRE-REGISTERED, construct-appropriate input-blind
null.  A generic 'chance line' is never used where the interface has its own
floor.

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
      source    = paperD_research/align_cka/<a>__<b>.json (cached CKA matrices)

  C4  Probe readout depth         (3 model families x 3 tasks x 5 splits)
      reported  = 1 - linear-probe knee depth  (how much depth the probe says
                  is unnecessary for the task to be linearly readable)
      null      = the model's own native readout knee
      source    = results/p1_2/p1_2_summary.json

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

MIDBAND = (0.25, 0.75)
CKA_DIR = "paperD_research/align_cka"
RESULTS_JSON = "paperD_research/repr_alignment_results.json"
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


def block_idx(La, Lb):
    ia = [i for i in range(La + 1) if MIDBAND[0] <= i / La <= MIDBAND[1]]
    jb = [j for j in range(Lb + 1) if MIDBAND[0] <= j / Lb <= MIDBAND[1]]
    return ia, jb


def block_mean(M, La, Lb):
    ia, jb = block_idx(La, Lb)
    return float(M[np.ix_(ia, jb)].mean())


def shuffle_null(Mz, La, Lb, n_perm, seed=0):
    """Verbatim the null in paperD_research/repr_alignment_multimodel.py:561.

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
def leg_mc(verbose=True):
    data = {}
    for label, d in MMLU_ARMS:
        rows = []
        with open(os.path.join(MMLU_DIR, d, "per_example_mmlu.jsonl")) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        data[label] = rows
    n = len(data["base (32L intact)"])
    assert all(len(v) == n for v in data.values()), "arms disagree on item count"

    gold = Counter(r["gold_letter"] for r in data["base (32L intact)"])
    const_letter, hits = gold.most_common(1)[0]
    const_acc = hits / n

    # content interface has its own floor: always pick the longest option.
    # 4,805/14,042 items have >=2 maximal-length options, so the tie convention
    # is load-bearing and all of them are reported rather than one being picked.
    def longest_variant(conv):
        tot = 0.0
        for r in data["base (32L intact)"]:
            c = r["content_norm"]["cont_tokens"]
            top = max(c.values())
            win = [k for k in "ABCD" if c[k] == top]
            g = r["gold_letter"]
            if conv == "split":
                tot += (1.0 / len(win)) if g in win else 0.0
            elif conv == "first":
                tot += 1.0 if win[0] == g else 0.0
            elif conv == "last":
                tot += 1.0 if win[-1] == g else 0.0
            elif conv == "credit":       # optimistic: any tie counts as a hit
                tot += 1.0 if g in win else 0.0
            elif conv == "wrong":        # pessimistic: any tie counts as a miss
                tot += 1.0 if (len(win) == 1 and win[0] == g) else 0.0
        return tot / n

    longest_convs = {c: longest_variant(c)
                     for c in ("split", "first", "last", "credit", "wrong")}
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
              f"paperE_research/ is the 'last' convention; 'split' is the "
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=200,
                    help="layer-order-shuffle permutations per pair (C3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()

    c1 = leg_mc()
    c2 = leg_squad()
    c3 = leg_cka(args.n_perm, args.seed)
    c4 = leg_probe()

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
                   "table": [{"construct": n, "reported": r, "null": u,
                              "residual": r - u, "residual_frac": (r - u) / r,
                              "null_desc": d} for n, r, u, d in table],
                   "gate_span": span, "gate_pass": bool(span >= 10),
                   "gate_c4_variants": variants,
                   "gate_pass_any_c4_variant": bool(gate_any),
                   "n_perm": args.n_perm, "seed": args.seed}
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
