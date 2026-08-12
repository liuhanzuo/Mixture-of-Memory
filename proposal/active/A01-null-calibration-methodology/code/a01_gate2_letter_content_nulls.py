#!/usr/bin/env python3
"""paperC gate-2 FULL replication: construct-appropriate nulls for the
letter-vs-content MC interface on NON-MMLU benchmarks.

What this closes
----------------
`paperC/README.md` open defect 2 / `STATUS.json:gate_results.gate2_second_mc_benchmark`.
The 2026-08-08 gate-2 leg reproduced the wrong-null problem off MMLU, but its two
"interfaces" were **raw sum-LL vs length-normalised acc_norm**, which is analogous
to but NOT identical with MMLU's **letter vs content**. This script consumes the
records written by `scripts/eval_olmo2_mc_letter_content.py` — which builds MMLU's
exact letter/content contrast on non-MMLU tasks — and computes, per task and per
arm, every null the protocol requires:

  letter interface  -> BEST-CONSTANT LETTER (argmax of the gold-letter marginal).
                       This is the non-MMLU analogue of MMLU's always-D = 0.2689.
                       It is NOT 1/n_opt; the whole point is that gold marginals
                       are skewed.
  content interface -> LONGEST-OPTION null under all FIVE tie conventions
                       (split / first / last / credit / wrong), because A01 已经
                       established that the convention is a real degree of freedom
                       that can reverse verdicts (GATE3_CONVENTIONS_VERDICT.md).

Statistics
----------
Paired throughout: the null is an input-blind PER-ITEM score vector, so
`model_correct - null_score` is a paired difference over the identical item set.
  * paired bootstrap, n_boot = 10000, on that difference;
  * two-sided bootstrap p via `two_sided_boot_p` — the R-7-FIXED mid-p estimator
    imported verbatim from `a01_gate3_fp32_vs_bf16.py`, NOT the old
    `2*min((bs<=0).mean(),(bs>=0).mean())` which can exceed 1;
  * exact McNemar against the DETERMINISTIC constant predictor (always-<best
    letter>), which is a genuine per-item 0/1 vector so the discordant-pair test
    is well defined. For the letter interface this is the primary significance
    test, matching how gate-2 reported BoolQ.

Reported alongside (they are DECOUPLED from the floor verdict — see
`STATUS.json:claim_scope_after_gates.NARROWED_20260810_constant_predictor`):
modal prediction share, exact-tie rate, and the acc-vs-acc_norm sign flips
(the latter reframed as a REPLICATION of Oostermeijer ICML 2026 under damage, not
as an A01 finding).

CPU only. No GPU, no model load.

Usage:
  python a01_gate2_letter_content_nulls.py <results_root> <out_json> [out_csv]
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

LETTERS = "ABCDEFGHIJ"
CONVS = ["split", "first", "last", "credit", "wrong"]
N_BOOT = 10000
BOOT_SEED = 7  # same seed as a01_gate3_content_conventions.py

# increasing damage, matching STATUS.json's six-arm ordering
ARM_ORDER = [
    "7B_base",
    "7B_shortgpt16_step200000",
    "7B_keep14_step200000",
    "7B_keep12_step124000",
    "7B_keep10_step83500",
    "7B_keep8_step121000",
]
TASKS = ["arc_challenge", "arc_easy", "openbookqa", "commonsense_qa",
         "piqa", "winogrande"]
EXPECTED_N = {"arc_challenge": 1172, "arc_easy": 2376, "openbookqa": 500,
              "commonsense_qa": 1221, "piqa": 1838, "winogrande": 1267}
# winogrande is carried as a NEGATIVE CONTROL, never counted as interface evidence
CONTROLS = {"winogrande"}


# ---------------------------------------------------------------------------
# estimators (R-7-fixed mid-p bootstrap + exact McNemar), imported verbatim
# ---------------------------------------------------------------------------
_A01 = os.path.dirname(os.path.abspath(__file__))
if _A01 not in sys.path:
    sys.path.insert(0, _A01)
from a01_gate3_fp32_vs_bf16 import (  # noqa: E402
    mcnemar_exact_p,
    two_sided_boot_p,
)


def paired_boot(d, n_boot=N_BOOT, seed=BOOT_SEED):
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        bs[i] = d[rng.integers(0, d.size, d.size)].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), two_sided_boot_p(bs, n_boot)


def verdict(mean, p):
    if p >= 0.05:
        return "AT the floor (n.s.)"
    return "BELOW the floor" if mean < 0 else "above the floor"


# ---------------------------------------------------------------------------
# load, WITH shard-integrity asserts (never analyse a partial set)
# ---------------------------------------------------------------------------
def load_arm_task(root, arm, task, num_shards=8):
    d = os.path.join(root, arm)
    shards = sorted(glob.glob(
        os.path.join(d, f"per_example_{task}_shard*of{num_shards}.jsonl")))
    idx = {int(os.path.basename(s).split("_shard")[1].split("of")[0])
           for s in shards}
    missing = sorted(set(range(num_shards)) - idx)
    assert not missing, f"{arm}/{task}: MISSING shards {missing} of {num_shards}"
    recs = []
    for s in shards:
        with open(s) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    ids = [r["item_id"] for r in recs]
    assert len(set(ids)) == len(ids), f"{arm}/{task}: duplicate item_id"
    exp = EXPECTED_N[task]
    assert len(recs) == exp, f"{arm}/{task}: n_scored={len(recs)} != {exp}"
    n_nan = sum(1 for r in recs if r["nan"])
    assert n_nan == 0, f"{arm}/{task}: n_nan={n_nan} != 0"
    recs.sort(key=lambda r: r["item_id"])
    return recs, {"n_shards": len(shards), "n_scored": len(recs),
                  "expected_n": exp, "n_nan": n_nan}


# ---------------------------------------------------------------------------
# nulls
# ---------------------------------------------------------------------------
def best_constant_letter(recs):
    """Best-constant (input-blind) letter predictor: emit the modal gold letter
    on every item. Per-item score vector + the marginal it came from.

    This is the non-MMLU analogue of MMLU's always-D = 0.2689. NOTE it is only
    admissible on items where that letter EXISTS: ARC has a handful of 3-option
    and 5-option items, so a constant emitter of "D" scores 0 (not undefined) on
    a 3-option item -- which is what a real always-D predictor would do."""
    gold = Counter(r["gold_letter"] for r in recs)
    best, cnt = max(gold.items(), key=lambda kv: (kv[1], -LETTERS.index(kv[0])))
    vec = np.array([1.0 if r["gold_letter"] == best else 0.0 for r in recs])
    return best, vec, {
        "gold_letter_marginal": dict(sorted(gold.items())),
        "best_constant_letter": best,
        "best_constant_acc": float(vec.mean()),
        "n_items_where_letter_exists": int(
            sum(1 for r in recs if LETTERS.index(best) < r["n_opt"])),
        "chance_1_over_nopt": float(np.mean([1.0 / r["n_opt"] for r in recs])),
        "delta_vs_chance_pp": 100 * (float(vec.mean())
                                     - float(np.mean([1.0 / r["n_opt"] for r in recs]))),
        "sanity_best_constant_eq_count_over_n": abs(
            float(vec.mean()) - cnt / len(recs)) < 1e-12,
    }


def longest_option_nulls(recs):
    """Longest-option (input-blind) content null under five tie conventions.
    W = argmax-set of continuation-token counts. Conventions verbatim from
    a01_gate3_fp32_vs_bf16.py::longest_floor."""
    out = {c: np.zeros(len(recs)) for c in CONVS}
    mult = Counter()
    gold_in_win = 0
    for i, r in enumerate(recs):
        ct = r["content_norm"]["cont_tokens"]
        keys = [LETTERS[k] for k in range(r["n_opt"])]
        top = max(ct[k] for k in keys)
        W = [k for k in keys if ct[k] == top]
        g = r["gold_letter"]
        mult[len(W)] += 1
        if g in W:
            gold_in_win += 1
        out["split"][i] = (1.0 / len(W)) if g in W else 0.0
        out["first"][i] = 1.0 if W[0] == g else 0.0
        out["last"][i] = 1.0 if W[-1] == g else 0.0
        out["credit"][i] = 1.0 if g in W else 0.0
        out["wrong"][i] = 1.0 if (len(W) == 1 and W[0] == g) else 0.0
    diag = {
        "winner_set_size_hist": {str(k): mult[k] for k in sorted(mult)},
        "frac_items_with_tied_longest": 1.0 - mult[1] / len(recs),
        "frac_items_gold_in_winner_set": gold_in_win / len(recs),
        "frac_items_ALL_options_tied": mult[max(mult)] / len(recs)
        if len(mult) == 1 else sum(v for k, v in mult.items()
                                   if k == max(r["n_opt"] for r in recs)) / len(recs),
    }
    return out, diag


def correct_vec(recs, interface):
    return np.array([1.0 if r[interface]["correct"] else 0.0 for r in recs])


def tie_rate(recs, interface):
    n = 0
    for r in recs:
        v = [r[interface]["scores"][LETTERS[k]] for k in range(r["n_opt"])]
        m = max(v)
        if sum(1 for x in v if x == m) > 1:
            n += 1
    return n / len(recs)


# ---------------------------------------------------------------------------
def main():
    root, out_json = sys.argv[1], sys.argv[2]
    out_csv = sys.argv[3] if len(sys.argv) > 3 else None

    arms = [a for a in ARM_ORDER if os.path.isdir(os.path.join(root, a))]
    assert len(arms) == 6, f"expected 6 arms, found {len(arms)}: {arms}"

    res = {
        "what": "paperC gate-2 FULL replication: MMLU's letter-vs-content "
                "interface contrast on non-MMLU MC, vs construct-appropriate nulls",
        "closes": "paperC/README.md open defect 2; STATUS.json "
                  "gate_results.gate2_second_mc_benchmark caveat "
                  "('raw sum-LL vs acc_norm is analogous but not identical')",
        "protocol": {
            "chat_template": False,
            "add_bos": 0,
            "weights": "fp32 master, bf16-autocast forward",
            "letter_prompt": "Question: <q>\\nA. <t0>\\nB. <t1>...\\nAnswer:  "
                             "cand=' A'..  (MMLU-identical construction)",
            "content_prompt": "Question: <q>\\nAnswer:  cand=' <option text>'",
            "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "boot_p": "two_sided_boot_p (R-7 mid-p fix, 2026-08-11) imported "
                      "from a01_gate3_fp32_vs_bf16.py",
        },
        "nulls": {
            "letter": "best-constant letter = argmax of the gold-letter marginal "
                      "(NOT 1/n_opt)",
            "content": "longest-option, five tie conventions (split canonical)",
        },
        "negative_controls": sorted(CONTROLS),
        "tasks": {},
    }
    rows = []

    for task in TASKS:
        if not os.path.exists(os.path.join(root, arms[0],
                                           f"per_example_{task}_shard0of8.jsonl")):
            continue
        t_out = {"is_negative_control": task in CONTROLS, "arms": {}}
        ref_bc = ref_lo = None

        for arm in arms:
            recs, integ = load_arm_task(root, arm, task)
            bc_letter, bc_vec, bc_diag = best_constant_letter(recs)
            lo_vecs, lo_diag = longest_option_nulls(recs)

            # the nulls are DATASET properties: identical across arms by construction
            if ref_bc is None:
                ref_bc = (bc_letter, float(bc_vec.mean()))
                ref_lo = {c: float(lo_vecs[c].mean()) for c in CONVS}
                t_out["letter_null"] = bc_diag
                t_out["content_null_longest_option"] = {
                    "floor_by_convention": ref_lo, "diagnostics": lo_diag}
            else:
                assert (bc_letter, float(bc_vec.mean())) == ref_bc, \
                    f"{task}/{arm}: letter null drifted -> item sets differ"
                for c in CONVS:
                    assert abs(float(lo_vecs[c].mean()) - ref_lo[c]) < 1e-12, \
                        f"{task}/{arm}: content null '{c}' drifted"

            a_out = {"shard_integrity": integ, "interfaces": {}}
            for interface, nulls in (
                    ("letter", {"best_constant_" + bc_letter: bc_vec}),
                    ("content_raw", {"longest_" + c: lo_vecs[c] for c in CONVS}),
                    ("content_norm", {"longest_" + c: lo_vecs[c] for c in CONVS})):
                cv = correct_vec(recs, interface)
                ent = {"acc": float(cv.mean()),
                       "modal_pred_share": max(Counter(
                           r[interface]["pred_letter"] for r in recs).values()) / len(recs),
                       "pred_hist": dict(sorted(Counter(
                           r[interface]["pred_letter"] for r in recs).items())),
                       "exact_tie_rate": tie_rate(recs, interface),
                       "vs_null": {}}
                for nname, nvec in nulls.items():
                    m, lo, hi, p = paired_boot(cv - nvec)
                    # exact McNemar vs the deterministic constant predictor. Only
                    # meaningful when the null is itself a 0/1 per-item decision
                    # (best-constant letter; 'first'/'last'/'credit'/'wrong' are
                    # 0/1 too, 'split' is fractional -> McNemar not defined).
                    mc = None
                    if set(np.unique(nvec)) <= {0.0, 1.0}:
                        b = int(np.sum((cv == 1) & (nvec == 0)))
                        c_ = int(np.sum((cv == 0) & (nvec == 1)))
                        mc = {"arm_right_null_wrong": b, "arm_wrong_null_right": c_,
                              "mcnemar_exact_p": mcnemar_exact_p(b, c_)}
                    ent["vs_null"][nname] = {
                        "null": float(nvec.mean()),
                        "delta_pp": 100 * m, "ci95_pp": [100 * lo, 100 * hi],
                        "boot_p": p, "verdict": verdict(m, p),
                        "mcnemar": mc,
                        "residual_fraction": ((float(cv.mean()) - float(nvec.mean()))
                                              / float(cv.mean()))
                        if cv.mean() > 0 else None,
                        "residual_fraction_vs_chance": (
                            (float(cv.mean()) - bc_diag["chance_1_over_nopt"])
                            / float(cv.mean())) if cv.mean() > 0 else None,
                    }
                    rows.append({
                        "task": task, "is_control": task in CONTROLS, "arm": arm,
                        "n": integ["n_scored"], "interface": interface,
                        "null_name": nname,
                        "acc": round(float(cv.mean()), 6),
                        "null": round(float(nvec.mean()), 6),
                        "delta_pp": round(100 * m, 4),
                        "ci95_lo_pp": round(100 * lo, 4),
                        "ci95_hi_pp": round(100 * hi, 4),
                        "boot_p": p,
                        "mcnemar_p": (mc["mcnemar_exact_p"] if mc else ""),
                        "verdict": verdict(m, p),
                        "modal_pred_share": round(ent["modal_pred_share"], 6),
                        "exact_tie_rate": round(ent["exact_tie_rate"], 6),
                        "residual_fraction": round(
                            ent["vs_null"][nname]["residual_fraction"], 6)
                        if ent["vs_null"][nname]["residual_fraction"] is not None else "",
                        "residual_fraction_vs_chance": round(
                            ent["vs_null"][nname]["residual_fraction_vs_chance"], 6)
                        if ent["vs_null"][nname]["residual_fraction_vs_chance"] is not None else "",
                    })
                a_out["interfaces"][interface] = ent

            # within-arm letter-vs-content pairing (the MMLU headline pair)
            Lc = correct_vec(recs, "letter")
            CN = correct_vec(recs, "content_norm")
            b = int(np.sum((Lc == 1) & (CN == 0)))
            c_ = int(np.sum((Lc == 0) & (CN == 1)))
            m, lo, hi, p = paired_boot(CN - Lc)
            a_out["content_norm_minus_letter"] = {
                "delta_pp": 100 * m, "ci95_pp": [100 * lo, 100 * hi],
                "boot_p": p, "mcnemar_exact_p": mcnemar_exact_p(b, c_),
                "letter_only_correct": b, "content_only_correct": c_,
                "agreement": float(np.mean(Lc == CN)),
            }
            t_out["arms"][arm] = a_out

        # acc vs acc_norm sign flips over all arm pairs (REPLICATION of
        # Oostermeijer ICML 2026 under damage -- NOT an A01 finding)
        flips = []
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                ai, aj = arms[i], arms[j]
                r_i = t_out["arms"][ai]["interfaces"]["content_raw"]["acc"]
                r_j = t_out["arms"][aj]["interfaces"]["content_raw"]["acc"]
                n_i = t_out["arms"][ai]["interfaces"]["content_norm"]["acc"]
                n_j = t_out["arms"][aj]["interfaces"]["content_norm"]["acc"]
                if (r_i - r_j) * (n_i - n_j) < 0:
                    flips.append(f"{ai} vs {aj}")
        t_out["content_raw_vs_norm_sign_flips"] = {
            "n_flips": len(flips), "n_pairs": len(arms) * (len(arms) - 1) // 2,
            "pairs": flips,
            "attribution": "REPLICATION under damage of Oostermeijer, ICML 2026 "
                           "(arXiv:2607.12767). NOT an A01/paperC finding.",
        }
        res["tasks"][task] = t_out
        print(f"[done] {task}")

    # ---- cross-task roll-up: how many arms sit at/below their letter floor ----
    roll = {}
    for task, t in res["tasks"].items():
        bcname = [k for k in
                  next(iter(t["arms"].values()))["interfaces"]["letter"]["vs_null"]][0]
        per_arm = {a: t["arms"][a]["interfaces"]["letter"]["vs_null"][bcname]["verdict"]
                   for a in t["arms"]}
        roll[task] = {
            "is_negative_control": t["is_negative_control"],
            "letter_null": t["letter_null"]["best_constant_acc"],
            "letter_null_name": bcname,
            "chance": t["letter_null"]["chance_1_over_nopt"],
            "n_arms_below_or_at": sum(1 for v in per_arm.values()
                                      if not v.startswith("above")),
            "n_arms_strictly_below_sig": sum(1 for v in per_arm.values()
                                             if v.startswith("BELOW")),
            "per_arm_verdict": per_arm,
        }
    res["rollup_letter_floor"] = roll

    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out_json}")
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
