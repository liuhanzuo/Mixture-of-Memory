#!/usr/bin/env python3
"""A01 audit-response recompute — CPU only, ZERO GPU.

Recomputes, from on-disk records only, the three numbers the TCODEX audit
(proposal/active/A03-parametric-vs-external-memory/evidence/TCODEX_AUDIT_20260810.md
sections 2.1 + 7) says A01 got wrong:

  R1  the Llama-2 gate-1 letter depth curve INCLUDING the gap-fill arms
      (k=8,12,18,22,26), with paired McNemar on every adjacent step and BH
      correction across the 14 steps -> how many times does letter reverse
      direction, and is the reversal statistical or noise?

  R2  whether Llama-2's *content* curve is "strictly monotone" as printed in
      GATE1_DEPTHCURVE_VERDICT.md's table (full k4..k31 range).

  R3  the longest-option tie-convention spread restricted to the three
      EXECUTABLE (input-blind-realisable) conventions split/first/last, vs the
      two BOUNDS credit/wrong -> how many arm verdicts actually move.

  R4  the tokenizer leg the audit did NOT attack: per-family longest-option
      nulls, and whether using the shared OLMo-2 null 0.284450 instead of a
      family's own null changes any arm's verdict.

Thresholds and conventions used here are NOT new: alpha=0.05 and the
"above / AT / BELOW the null" trichotomy are the same ones every earlier A01
gate used (see code/a01_gate3_fp32_vs_bf16.py, code/a01_gate1_verdict.py).
The one thing added is BH across the adjacent-step family in R1, which is
stated in the output and was chosen before looking at the step p-values
(the audit's complaint was precisely that no change-point / multiplicity
inference had been done at all).

Usage:  python a01_audit_response_recompute.py [--out <json>]
Inputs (wzc1): olmo2_mmlu_content_results/gate1_dmg_llama2_7b*/per_example_mmlu_shard*of8.jsonl
               proposal/active/A01-.../evidence/gate3_content_null_conventions.csv
               proposal/active/A01-.../evidence/a01_gate1_third_family.json
Cross-disk (zwfy6, read-only over ssh, optional): the other three families'
               olmo2_mmlu_content_results/gate1_dmg_*/summary.json
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from fractions import Fraction
from math import comb

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
A01 = os.path.join(REPO, "proposal", "active", "A01-null-calibration-methodology")
LETTER_FLOOR = 0.2689075630252101  # always-D on cais/mmlu test, n=14042
ALPHA = 0.05

# The three executable conventions, and the two bounds that are NOT policies.
EXECUTABLE = ("split", "first", "last")
BOUNDS = ("credit", "wrong")


def mcnemar_exact(b, c):
    """Two-sided exact binomial McNemar on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    # exact rational then float: n can be ~6000, so 2**n overflows a C double
    return min(1.0, float(Fraction(2 * sum(comb(n, i) for i in range(lo + 1)), 2 ** n)))


def binom_two_sided(x, n, p0):
    """Exact two-sided binomial test, scipy if available else point-prob method."""
    try:
        from scipy.stats import binomtest
        return binomtest(x, n, p0).pvalue
    except Exception:
        from math import isclose
        d = comb(n, x) * p0 ** x * (1 - p0) ** (n - x)
        tot = 0.0
        for i in range(n + 1):
            pi = comb(n, i) * p0 ** i * (1 - p0) ** (n - i)
            if pi < d or isclose(pi, d, rel_tol=1e-12):
                tot += pi
        return min(1.0, tot)


def bh(pvals, alpha=ALPHA):
    """Benjamini-Hochberg step-up; returns a bool mask."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    kmax = 0
    for rank, i in enumerate(order, 1):
        if pvals[i] <= alpha * rank / m:
            kmax = rank
    keep = [False] * m
    for rank, i in enumerate(order, 1):
        if rank <= kmax:
            keep[i] = True
    return keep


def verdict(acc, null, p, alpha=ALPHA):
    if p > alpha:
        return "AT"
    return "above" if acc > null else "BELOW"


# ---------------------------------------------------------------- R1 / R2

def load_llama2_arms():
    """Return {k: (dirname, {item_id: record})} for the Llama-2 depth curve.

    Several k have two directories (the original depth_k* and the gap/gap2
    re-run). They are asserted byte-equal on the two accuracies before one is
    kept, so the duplicate is a re-run of the same arm, not a second arm.
    """
    by_k = {}
    for d in sorted(glob.glob(os.path.join(REPO, "olmo2_mmlu_content_results",
                                           "gate1_dmg_llama2_7b*"))):
        s = os.path.join(d, "summary.json")
        if not os.path.exists(s):
            continue
        j = json.load(open(s))
        k = j["meta"]["keep_front_layers"]
        by_k.setdefault(k, []).append((os.path.basename(d), j))
    dups = {}
    for k, v in by_k.items():
        if len(v) > 1:
            accs = {(round(x[1]["letter_acc"], 12), round(x[1]["content_norm_acc"], 12))
                    for x in v}
            assert len(accs) == 1, ("duplicate dirs disagree at k=%d: %s" % (k, v))
            dups[k] = [x[0] for x in v]
    out = {}
    for k, v in sorted(by_k.items()):
        name = v[0][0]
        recs = {}
        shards = sorted(glob.glob(os.path.join(
            REPO, "olmo2_mmlu_content_results", name, "per_example_mmlu_shard*of8.jsonl")))
        assert len(shards) == 8, "k=%d has %d shards, expected 8" % (k, len(shards))
        for sh in shards:
            for line in open(sh):
                r = json.loads(line)
                assert r["item_id"] not in recs, "dup item_id in %s" % name
                recs[r["item_id"]] = r
        assert len(recs) == 14042, "k=%d merged to n=%d, expected 14042" % (k, len(recs))
        out[k] = (name, recs)
    return out, dups


def curve(arms, readout):
    ks = sorted(arms)
    ids = sorted(arms[ks[0]][1])
    accs, steps = [], []
    for k in ks:
        recs = arms[k][1]
        accs.append(sum(1 for i in ids if recs[i][readout]["correct"]) / len(ids))
    for a, b in zip(ks, ks[1:]):
        A = [arms[a][1][i][readout]["correct"] for i in ids]
        B = [arms[b][1][i][readout]["correct"] for i in ids]
        nb = sum(1 for x, y in zip(A, B) if x and not y)
        nc = sum(1 for x, y in zip(A, B) if y and not x)
        steps.append({"from_k": a, "to_k": b,
                      "delta_pp": (sum(B) - sum(A)) / len(ids) * 100.0,
                      "mcnemar_b": nb, "mcnemar_c": nc,
                      "mcnemar_exact_p": mcnemar_exact(nb, nc)})
    keep = bh([s["mcnemar_exact_p"] for s in steps])
    for s, k in zip(steps, keep):
        s["bh_significant"] = bool(k)
    return ks, accs, steps


def runs(steps, sig_only=True):
    """Maximal same-direction runs over the (optionally BH-significant) steps."""
    seq = [(1 if s["delta_pp"] > 0 else -1)
           for s in steps if (s["bh_significant"] or not sig_only)]
    out, cur = [], None
    for x in seq:
        if x != cur:
            out.append(x)
            cur = x
    return out


# ---------------------------------------------------------------- R3

def r3_conventions():
    import csv
    path = os.path.join(A01, "evidence", "gate3_content_null_conventions.csv")
    rows = [r for r in csv.DictReader(open(path))
            if r["dtype"] == "bf16" and r["readout"] == "content_norm"]
    nulls, counts, resid = {}, {}, {}
    for c in EXECUTABLE + BOUNDS:
        sub = [r for r in rows if r["convention"] == c]
        nv = {float(r["null"]) for r in sub}
        assert len(nv) == 1, "null not constant across arms for %s" % c
        nulls[c] = nv.pop()
        counts[c] = {
            "above": sum(1 for r in sub if r["verdict"] == "above the null"),
            "below": sum(1 for r in sub if "BELOW" in r["verdict"]),
            "at": sum(1 for r in sub if r["verdict"] not in
                      ("above the null",) and "BELOW" not in r["verdict"]),
        }
        resid[c] = {r["arm"]: float(r["residual_fraction"]) for r in sub}
    arms = sorted(resid[EXECUTABLE[0]])
    ex_span = {}
    for a in arms:
        v = [resid[c][a] for c in EXECUTABLE]
        allv = [resid[c][a] for c in EXECUTABLE + BOUNDS]
        ex_span[a] = {
            "exec_only_residual_fraction": {c: resid[c][a] for c in EXECUTABLE},
            "exec_only_abs_span": max(v) - min(v),
            "exec_only_ratio": max(v) / min(v) if min(v) > 0 else None,
            "all_five_abs_span": max(allv) - min(allv),
            "all_five_ratio": (max(allv) / min(allv)) if min(allv) > 0 else None,
        }
    exn = [nulls[c] for c in EXECUTABLE]
    alln = [nulls[c] for c in EXECUTABLE + BOUNDS]
    return {
        "why_credit_and_wrong_are_not_conventions":
            "credit scores 1 whenever gold is in the token-count winner set, i.e. it "
            "assumes the tie is broken in gold's favour -- an input-blind baseline "
            "cannot know the gold letter, so no executable policy attains it; it is an "
            "ORACLE UPPER BOUND. wrong scores 0 on every tie, i.e. it assumes the tie "
            "is always broken against gold; a baseline that must emit an answer gets "
            "1/|W| in expectation, so it is a PESSIMISTIC LOWER BOUND. Together they "
            "bound the identified set of the null under an unstated policy; they are "
            "not five equally defensible readings.",
        "nulls": nulls,
        "executable_null_span_pp": (max(exn) - min(exn)) * 100.0,
        "all_five_null_span_pp": (max(alln) - min(alln)) * 100.0,
        "verdict_counts_of_6_arms": counts,
        "per_arm_residual_fraction_spans": ex_span,
    }


# ---------------------------------------------------------------- R4

def r4_tokenizer(zwfy6_json=None):
    tf = json.load(open(os.path.join(A01, "evidence", "a01_gate1_third_family.json")))
    fam_null = {}
    for a in tf["arms"]:
        lab = a["label"]
        fam = ("llama2_7b" if "Llama-2" in lab else
               "llama3_8b" if "Llama-3" in lab else
               "qwen3_8b" if "Qwen3" in lab else "olmo2")
        fam_null.setdefault(fam, a["longest_option_split_tie_null"])
    shared = fam_null["olmo2"]
    rows = []
    for d in sorted(glob.glob(os.path.join(REPO, "olmo2_mmlu_content_results",
                                           "gate1_dmg_*"))):
        s = os.path.join(d, "summary.json")
        if not os.path.exists(s):
            continue
        j = json.load(open(s))
        k = j["meta"].get("keep_front_layers")
        if k is None:
            continue
        rows.append((os.path.basename(d), k, j["content_norm_acc"], j["n_valid"]))
    if zwfy6_json and os.path.exists(zwfy6_json):
        for name, k, l, c, n, _bm in json.load(open(zwfy6_json)):
            if k is not None:
                rows.append((name, k, c, n))
    seen, flips, tested = {}, [], 0
    for name, k, c, n in rows:
        if name in seen:
            continue
        seen[name] = True
        fam = ("llama2_7b" if "llama2" in name else
               "llama3_8b" if "llama3" in name else
               "qwen3_8b" if "qwen3" in name else "olmo2")
        if fam == "olmo2":
            continue  # shared null IS olmo2's own null
        own = fam_null[fam]
        x = round(c * n)
        p_own = binom_two_sided(x, n, own)
        p_sh = binom_two_sided(x, n, shared)
        v_own, v_sh = verdict(c, own, p_own), verdict(c, shared, p_sh)
        tested += 1
        if v_own != v_sh:
            flips.append({"arm": name, "keep": k, "family": fam,
                          "content_norm": c,
                          "own_null": own, "own_p": p_own, "own_verdict": v_own,
                          "shared_null": shared, "shared_p": p_sh,
                          "shared_verdict": v_sh,
                          "robust": bool(min(p_own, p_sh) < 0.005)})
    nv = list(fam_null.values())
    return {
        "per_family_longest_option_split_tie_null": fam_null,
        "null_span_pp": (max(nv) - min(nv)) * 100.0,
        "n_non_olmo_arms_tested": tested,
        "n_verdict_flips": len(flips),
        "n_robust_flips": sum(1 for f in flips if f["robust"]),
        "flips": flips,
        "multiplicity_note":
            "These are uncorrected two-sided exact binomial tests, one per arm-row, "
            "reusing A01's existing alpha=0.05 above/AT/BELOW trichotomy. A flip whose "
            "two p-values straddle 0.05 (e.g. 0.0507 vs 0.0443) is a boundary artifact "
            "and is marked robust=false; only flips where one side is p<0.005 are "
            "counted as robust.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(A01, "evidence",
                                                  "a01_audit_response_recompute.json"))
    ap.add_argument("--zwfy6_json", default="/tmp/a01_zwfy6_gate1.json",
                    help="optional cached read-only dump of the zwfy6 gate1 summaries")
    a = ap.parse_args()

    arms, dups = load_llama2_arms()
    res = {"generated_by": "code/a01_audit_response_recompute.py",
           "responds_to": "proposal/active/A03-parametric-vs-external-memory/evidence/"
                          "TCODEX_AUDIT_20260810.md sections 2.1 and 7",
           "compute": "CPU only, ZERO GPU",
           "alpha": ALPHA,
           "letter_best_constant_floor_always_D": LETTER_FLOOR}

    res["R1_R2_llama2_depth_curve"] = {
        "n_unique_keep_depths": len(arms),
        "keep_depths": sorted(arms),
        "duplicate_dirs_asserted_identical": dups,
        "gap_fill_arms_are_ON_DISK_not_in_flight": sorted(
            k for k, (nm, _) in arms.items() if "gap" in nm),
    }
    for readout in ("letter", "content_norm"):
        ks, accs, steps = curve(arms, readout)
        rr = runs(steps, sig_only=True)
        rr_all = runs(steps, sig_only=False)
        block = {
            "acc_by_keep": dict(zip(map(str, ks), accs)),
            "adjacent_steps": steps,
            "n_steps": len(steps),
            "n_bh_significant_steps": sum(1 for s in steps if s["bh_significant"]),
            "n_decreases_any": sum(1 for s in steps if s["delta_pp"] < 0),
            "n_decreases_bh_significant": sum(
                1 for s in steps if s["delta_pp"] < 0 and s["bh_significant"]),
            "raw_adjacent_direction_reversals": len(rr_all) - 1,
            "bh_significant_direction_reversals": len(rr) - 1,
            "n_maximal_descending_runs_bh_significant": sum(1 for x in rr if x < 0),
            "monotone_as_printed": all(
                s["delta_pp"] >= 0 for s in steps) or all(
                s["delta_pp"] <= 0 for s in steps),
        }
        if readout == "letter":
            fl = []
            for k in ks:
                recs = arms[k][1]
                ids = sorted(recs)
                acc = sum(1 for i in ids if recs[i]["letter"]["correct"]) / len(ids)
                x = round(acc * len(ids))
                p = binom_two_sided(x, len(ids), LETTER_FLOOR)
                lp = [recs[i]["letter"]["pred_letter"] for i in ids]
                fl.append({"keep": k, "letter_acc": acc,
                           "residual_pp_vs_floor": (acc - LETTER_FLOOR) * 100.0,
                           "binom_p_vs_floor": p,
                           "verdict": verdict(acc, LETTER_FLOOR, p),
                           "modal_pred": Counter(lp).most_common(1)[0][0],
                           "modal_share": Counter(lp).most_common(1)[0][1] / len(ids)})
            block["floor_verdict_by_keep"] = fl
            block["floor_verdict_sequence"] = [r["verdict"] for r in fl]
            block["n_floor_crossings"] = sum(
                1 for x, y in zip(fl, fl[1:])
                if x["verdict"] != y["verdict"])
        res["R1_R2_llama2_depth_curve"][readout] = block

    res["R3_tie_conventions"] = r3_conventions()
    res["R4_tokenizer_null"] = r4_tokenizer(a.zwfy6_json)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=False)
    print("wrote", a.out)

    L = res["R1_R2_llama2_depth_curve"]["letter"]
    C = res["R1_R2_llama2_depth_curve"]["content_norm"]
    print("\nR1 Llama-2 letter: %d unique depths, %d/%d steps BH-sig, "
          "%d raw reversals, %d BH-sig reversals, %d descending runs"
          % (res["R1_R2_llama2_depth_curve"]["n_unique_keep_depths"],
             L["n_bh_significant_steps"], L["n_steps"],
             L["raw_adjacent_direction_reversals"],
             L["bh_significant_direction_reversals"],
             L["n_maximal_descending_runs_bh_significant"]))
    print("   floor verdicts:", " ".join(L["floor_verdict_sequence"]))
    print("R2 Llama-2 content: monotone over full range = %s "
          "(%d decreases, %d BH-significant)"
          % (C["monotone_as_printed"], C["n_decreases_any"],
             C["n_decreases_bh_significant"]))
    print("R3 executable null span = %.4f pp; all-five span = %.4f pp"
          % (res["R3_tie_conventions"]["executable_null_span_pp"],
             res["R3_tie_conventions"]["all_five_null_span_pp"]))
    for c in EXECUTABLE + BOUNDS:
        v = res["R3_tie_conventions"]["verdict_counts_of_6_arms"][c]
        print("   %-7s above=%d at=%d below=%d" % (c, v["above"], v["at"], v["below"]))
    print("R4 tokenizer null span = %.4f pp; %d/%d arm verdicts flip (%d robust)"
          % (res["R4_tokenizer_null"]["null_span_pp"],
             res["R4_tokenizer_null"]["n_verdict_flips"],
             res["R4_tokenizer_null"]["n_non_olmo_arms_tested"],
             res["R4_tokenizer_null"]["n_robust_flips"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
