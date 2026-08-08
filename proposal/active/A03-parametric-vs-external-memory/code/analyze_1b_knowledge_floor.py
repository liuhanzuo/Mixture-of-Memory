#!/usr/bin/env python3
"""A03 kill-condition analysis: is the OLMo-2 1B pilot measurable above its own
knowledge-interface floors?

A03's third Kill condition is "1B pilot 所有知识指标均处于 floor, 无法测量", and its
first key control is "每个接口必须高于自己的 null floor". This script takes the
per-example dumps produced by the two existing harnesses on the 1B arms

    olmo2_mmlu_content_results/<arm>/per_example_mmlu.jsonl    (MMLU letter+content)
    olmo2_closedbook_results/<arm>/per_example_{popqa,triviaqa}.jsonl

and reports, per (arm x task x interface), the A01 four-tuple

    reported / construct-appropriate null / calibrated residual / residual fraction

plus a paired significance test of (arm - null).

Nulls are CONSTRUCT-APPROPRIATE and input-blind, never a generic chance line:

  MMLU letter interface   best-constant letter (always-X for the X maximising
                          accuracy over the gold distribution). NOT .25 --- A01
                          documents that on this 14,042-item set the best constant
                          is always-D and that using .25 wrongly credits arms.
  MMLU content interface  longest-option heuristic (pick the option with the most
                          continuation tokens), split-tie convention = fractional
                          credit under uniform random tie-breaking, which is the
                          unbiased expectation. All five tie conventions are
                          reported since 1/3 of MMLU items have >=2 maximal options.
  PopQA / TriviaQA        best-constant answer string: every candidate constant
                          (the K most frequent normalised gold strings + the empty
                          string / a refusal string) is scored as if the model had
                          emitted it for EVERY question, under the SAME metric, and
                          the maximum is the floor. This is the generative analogue
                          of the best-constant MC floor and dominates the harness's
                          own majority_em (which fixes the constant a priori
                          instead of maximising over it). The empty/refusal
                          constant is the A01 SQuAD-style refusal baseline; here it
                          is ~0 because neither set has unanswerable items, so the
                          majority-prior constant is the binding floor.

Statistics (identical conventions to A01's build_null_calibration_table.py):
  * paired bootstrap over ITEMS on the per-item difference vector (n_boot=10000,
    multinomial representation, two-sided p floored at 1/n_boot);
  * exact-binomial McNemar on the discordant items whenever the null is binary;
  * Benjamini-Hochberg q=.05 across the whole (arm x task x interface) family,
    because the kill decision reads every cell at once.

Verdict per cell: "above floor" (residual > 0 and BH-significant), "AT floor"
(not significant), "BELOW floor" (residual < 0 and significant).
A03 KILL_1B_PILOT iff the pruned+healed arm is not above floor on ANY axis.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from collections import Counter

import numpy as np

try:
    from scipy.stats import binomtest
except Exception:  # pragma: no cover
    binomtest = None

N_BOOT = 10000
SEED = 0
TIE_CONVS = ("split", "first", "last", "credit", "wrong")

# ---------------------------------------------------------------------------
# answer normalisation: byte-identical to eval_olmo2_closedbook_qa.normalize_answer
# (re-implemented here rather than imported so the analysis has no torch dep;
#  a self-test below asserts agreement with the harness on the real dumps).
# ---------------------------------------------------------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    s = " ".join(s.split())
    return s


def _f1(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def score_prediction(pred: str, golds) -> dict:
    np_pred = normalize_answer(pred)
    em = contains = 0
    f1 = 0.0
    for g in golds:
        ng = normalize_answer(g)
        if not ng:
            continue
        if np_pred == ng:
            em = 1
        if ng in np_pred:
            contains = 1
        f1 = max(f1, _f1(pred, g))
    return {"em": em, "contains": contains, "f1": f1}


# ---------------------------------------------------------------------------
# statistics (A01 conventions)
# ---------------------------------------------------------------------------
def paired_bootstrap(d, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap over items on per-item difference vector d."""
    d = np.asarray(d, dtype=float)
    n = d.size
    vals, counts = np.unique(d, return_counts=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, counts / n, size=n_boot)
    means = draws @ vals / n
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(max(p, 1.0 / n_boot), 1.0))


def mcnemar_exact(a, b):
    """Exact-binomial McNemar on two 0/1 per-item correctness vectors."""
    a = np.asarray(a) > 0.5
    b = np.asarray(b) > 0.5
    b01 = int((a & ~b).sum())
    b10 = int((~a & b).sum())
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    if binomtest is None:
        return b01, b10, float("nan")
    return b01, b10, float(binomtest(b01, n, 0.5, alternative="two-sided").pvalue)


def bh_reject(pvals, q=0.05):
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
    adj_ranked = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.minimum(adj_ranked, 1.0)
    return reject, adj, k


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------
def load_mmlu_arm(root, arm):
    path = os.path.join(root, arm, "per_example_mmlu.jsonl")
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["item_id"])
    return rows


def longest_option_vector(rows, gold_letters, conv):
    """Input-blind 'always pick the longest option' null, per item."""
    out = np.zeros(len(rows))
    for i, r in enumerate(rows):
        c = r["content_norm"]["cont_tokens"]
        keys = [k for k in "ABCDEFGHIJKLMNOP" if k in c]
        top = max(c[k] for k in keys)
        win = [k for k in keys if c[k] == top]
        g = gold_letters[i]
        if conv == "split":
            out[i] = (1.0 / len(win)) if g in win else 0.0
        elif conv == "first":
            out[i] = 1.0 if win[0] == g else 0.0
        elif conv == "last":
            out[i] = 1.0 if win[-1] == g else 0.0
        elif conv == "credit":
            out[i] = 1.0 if g in win else 0.0
        elif conv == "wrong":
            out[i] = 1.0 if (len(win) == 1 and win[0] == g) else 0.0
        else:
            raise ValueError(conv)
    return out


def best_constant_letter(gold_letters):
    """Best input-blind constant letter + per-item hit vector."""
    cnt = Counter(gold_letters)
    letter, hits = cnt.most_common(1)[0]
    vec = np.array([1.0 if g == letter else 0.0 for g in gold_letters])
    return letter, hits / len(gold_letters), vec, dict(sorted(cnt.items()))


# ---------------------------------------------------------------------------
# closed-book QA
# ---------------------------------------------------------------------------
def load_cb_arm(root, arm, task):
    path = os.path.join(root, arm, f"per_example_{task}.jsonl")
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["item_id"])
    return rows


def best_constant_qa(rows, metric, top_k=300, extra=("", "unknown", "i dont know")):
    """Best input-blind CONSTANT prediction under `metric` ("em" or "contains").

    Candidate constants = the top_k most frequent normalised gold strings (over
    the whole eval set) + refusal/empty strings. Each is scored as if emitted for
    every item; the maximum is the floor. Returns (best_string, acc, per-item
    vector, diagnostics).

    Scoring is the em/contains half of score_prediction() specialised for a
    constant prediction: the candidate is already normalised, so
      em[i]       = candidate in {normalised golds of item i}
      contains[i] = some normalised gold of item i is a substring of the candidate
    which is what score_prediction computes (it never truncates or re-cases the
    prediction beyond normalize_answer). f1 is deliberately not computed here --
    it is not one of the reported interfaces and dominates the runtime.
    """
    gold_norms = [[normalize_answer(g) for g in r["gold"]] for r in rows]
    gold_norms = [[g for g in gs if g] for gs in gold_norms]
    gold_sets = [set(gs) for gs in gold_norms]

    freq = Counter(gs[0] for gs in gold_norms if gs)
    cands = [c for c, _ in freq.most_common(top_k)] + [
        normalize_answer(e) for e in extra]
    seen = set()
    cands = [c for c in cands if not (c in seen or seen.add(c))]

    best = (None, -1.0, None)
    table = []
    for c in cands:
        if metric == "em":
            vec = np.fromiter((1.0 if c in s else 0.0 for s in gold_sets),
                              dtype=float, count=len(rows))
        elif metric == "contains":
            vec = np.fromiter((1.0 if any(g in c for g in gs) else 0.0
                               for gs in gold_norms),
                              dtype=float, count=len(rows))
        else:
            raise ValueError(metric)
        m = float(vec.mean())
        table.append((c, m))
        if m > best[1]:
            best = (c, m, vec)
    table.sort(key=lambda t: -t[1])
    # cross-check the winner against the FULL harness scorer (cheap: one pass)
    ref = float(np.mean([score_prediction(best[0], r["gold"])[metric]
                         for r in rows]))
    assert abs(ref - best[1]) < 1e-12, \
        f"fast constant scorer disagrees with harness scorer: {ref} vs {best[1]}"
    return best[0], best[1], best[2], {
        "n_candidates": len(cands),
        "top5": [{"constant": c, metric: round(m, 6)} for c, m in table[:5]],
        "majority_gold_string": freq.most_common(1)[0][0] if freq else "",
    }


# ---------------------------------------------------------------------------
def lengthmatched_contains_null(rows, target_chars, top_k=4000):
    """Input-blind null for the `contains` metric that is allowed to be VERBOSE.

    `contains` credits an item whenever ANY normalised gold alias is a substring
    of the normalised prediction, so it rewards length: a long prediction has more
    chances to swallow a gold string. A single-word constant therefore understates
    the floor for an arm whose predictions are much longer than the reference
    arm's (here the pruned+healed arm emits ~6x more characters than the intact
    base, purely as a decoding-style artefact of healing).

    This null is still fully input-blind -- ONE fixed string emitted for every
    question -- but is length-matched to the arm being tested: the most frequent
    normalised gold answers are concatenated in frequency order until the string
    reaches the arm's own mean prediction length. Any `contains` credit it earns
    is available to a model that has learned nothing except "be verbose and name
    frequent entities", which is exactly the confound to rule out.
    """
    gold_norms = [[normalize_answer(g) for g in r["gold"]] for r in rows]
    gold_norms = [[g for g in gs if g] for gs in gold_norms]
    freq = Counter(gs[0] for gs in gold_norms if gs)
    parts = []
    total = 0
    for c, _ in freq.most_common(top_k):
        if total >= target_chars:
            break
        parts.append(c)
        total += len(c) + 1
    const = " ".join(parts)
    vec = np.fromiter((1.0 if any(g in const for g in gs) else 0.0
                       for gs in gold_norms), dtype=float, count=len(rows))
    return const, float(vec.mean()), vec


def cell(arm, task, interface, reported_vec, null_vec, null_name, seed_off=0,
         binary_null=True):
    rep = float(np.asarray(reported_vec, dtype=float).mean())
    nul = float(np.asarray(null_vec, dtype=float).mean())
    d = np.asarray(reported_vec, dtype=float) - np.asarray(null_vec, dtype=float)
    mean, lo, hi, p = paired_bootstrap(d, seed=SEED + seed_off)
    out = {
        "arm": arm, "task": task, "interface": interface,
        "n": int(np.asarray(reported_vec).size),
        "reported": rep,
        "null_name": null_name,
        "null": nul,
        "residual": rep - nul,
        "residual_fraction": (rep - nul) / rep if rep > 0 else None,
        "boot_mean_pp": 100 * mean,
        "boot_ci95_pp": [100 * lo, 100 * hi],
        "boot_p": p,
    }
    if binary_null and set(np.unique(np.asarray(null_vec, dtype=float))) <= {0.0, 1.0}:
        b01, b10, mp = mcnemar_exact(reported_vec, null_vec)
        out.update({"mcnemar_arm_only": b01, "mcnemar_null_only": b10,
                    "mcnemar_p": mp})
    else:
        out["mcnemar_p"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu_root", default="olmo2_mmlu_content_results")
    ap.add_argument("--cb_root", default="olmo2_closedbook_results")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="label=dirname, e.g. 'intact=A03_1B_base'")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    arms = []
    for a in args.arms:
        lab, d = a.split("=", 1)
        arms.append((lab, d))

    cells = []
    diag = {}

    # ---------------- MMLU ----------------
    mmlu_rows = {}
    for lab, d in arms:
        mmlu_rows[lab] = load_mmlu_arm(args.mmlu_root, d)
    ref = mmlu_rows[arms[0][0]]
    n_mmlu = len(ref)
    assert n_mmlu == 14042, f"MMLU n={n_mmlu} != 14042 (incomplete merge?)"
    ref_ids = [r["item_id"] for r in ref]
    gold_letters = [r["gold_letter"] for r in ref]
    for lab, rows in mmlu_rows.items():
        assert len(rows) == n_mmlu, f"{lab}: MMLU n={len(rows)} != {n_mmlu}"
        assert [r["item_id"] for r in rows] == ref_ids, f"{lab}: item_id misaligned"
        assert [r["gold_letter"] for r in rows] == gold_letters, f"{lab}: gold misaligned"
        assert not any(r.get("nan") for r in rows), f"{lab}: nan rows present"

    const_letter, const_acc, const_vec, gold_dist = best_constant_letter(gold_letters)
    longest = {c: longest_option_vector(ref, gold_letters, c) for c in TIE_CONVS}
    # the longest-option null is a property of tokenizer+dataset, not of the arm:
    # assert it is identical when computed from every arm's own dump.
    for lab, rows in mmlu_rows.items():
        v = longest_option_vector(rows, gold_letters, "split")
        assert np.allclose(v, longest["split"]), f"{lab}: longest-option null drifted"
    tie_rate = float(np.mean([
        sum(1 for k, val in r["content_norm"]["cont_tokens"].items()
            if val == max(r["content_norm"]["cont_tokens"].values())) >= 2
        for r in ref]))

    diag["mmlu"] = {
        "n": n_mmlu,
        "gold_letter_distribution": gold_dist,
        "best_constant_letter": const_letter,
        "best_constant_acc": const_acc,
        "chance_line_for_reference_only": 0.25,
        "longest_option_by_tie_convention": {c: float(v.mean()) for c, v in longest.items()},
        "preregistered_tie_convention": "split",
        "tied_longest_rate": tie_rate,
    }

    for i, (lab, rows) in enumerate(mmlu_rows.items()):
        L = np.array([1.0 if r["letter"]["correct"] else 0.0 for r in rows])
        CN = np.array([1.0 if r["content_norm"]["correct"] else 0.0 for r in rows])
        cells.append(cell(lab, "mmlu", "letter", L, const_vec,
                          f"best-constant always-{const_letter}", 100 + i))
        cells.append(cell(lab, "mmlu_content", "content_norm", CN, longest["split"],
                          "longest-option (split-tie)", 200 + i, binary_null=False))
        # letter-interface degeneration diagnostics (A01's documented mechanism for
        # an arm landing AT/BELOW the constant floor): how concentrated the arm's
        # letter predictions are, and how often the top-2 letter scores are exactly
        # tied so that argmax breaks the tie by INDEX (input-blind).
        preds = Counter(r["letter"]["pred_letter"] for r in rows)
        modal_letter, modal_hits = preds.most_common(1)[0]
        own_null = np.array([1.0 if g == modal_letter else 0.0
                             for g in gold_letters])
        om, _, _, op = paired_bootstrap(L - own_null, seed=SEED + 400 + i)
        tied = float(np.mean([
            sorted(r["letter"]["scores"].values())[-1]
            == sorted(r["letter"]["scores"].values())[-2] for r in rows]))
        diag.setdefault("letter_degeneration", {})[lab] = {
            "letter_pred_distribution": dict(sorted(preds.items())),
            "modal_pred_letter": modal_letter,
            "modal_pred_share": modal_hits / len(rows),
            "vs_own_modal_null_pp": 100 * om,
            "vs_own_modal_null_p": op,
            "bf16_exact_top2_tie_rate": tied,
        }

    # ---------------- closed-book QA ----------------
    for task, expected_n, headline in (("popqa", 14267, "contains"),
                                       ("triviaqa", 17944, "em")):
        rows_by_arm = {}
        ok = True
        for lab, d in arms:
            p = os.path.join(args.cb_root, d, f"per_example_{task}.jsonl")
            if not os.path.exists(p):
                print(f"[skip] {lab} {task}: {p} missing", file=sys.stderr)
                ok = False
                break
            rows_by_arm[lab] = load_cb_arm(args.cb_root, d, task)
        if not ok:
            continue
        ref_t = rows_by_arm[arms[0][0]]
        n_t = len(ref_t)
        assert n_t == expected_n, f"{task}: n={n_t} != expected {expected_n}"
        ref_ids_t = [r["item_id"] for r in ref_t]
        for lab, rows in rows_by_arm.items():
            assert len(rows) == n_t, f"{lab} {task}: n={len(rows)} != {n_t}"
            assert [r["item_id"] for r in rows] == ref_ids_t, \
                f"{lab} {task}: item_id misaligned"
            assert [r["gold"] for r in rows] == [r["gold"] for r in ref_t], \
                f"{lab} {task}: gold misaligned"
        # self-test: our re-implemented scorer reproduces the harness's stored metric
        for r in ref_t[:500]:
            sc = score_prediction(r["pred"], r["gold"])
            assert sc["em"] == r["em"] and sc["contains"] == r["contains"], \
                f"{task}: scorer disagrees with harness on item {r['item_id']}"

        dg = {}
        for metric in ("em", "contains"):
            bstr, bacc, bvec, bdiag = best_constant_qa(ref_t, metric)
            dg[metric] = {"best_constant": bstr, "acc": bacc, **bdiag}
            for i, (lab, rows) in enumerate(rows_by_arm.items()):
                V = np.array([float(r[metric]) for r in rows])
                cells.append(cell(lab, task, metric, V, bvec,
                                  f"best-constant answer '{bstr}'",
                                  300 + 50 * (metric == "contains") + i))
        dg["headline_metric"] = headline
        # generative degeneration diagnostics: an arm can sit at the floor either
        # because it answers plausibly but wrongly, or because it emits (near-)
        # constant / empty text. Distinguishing these changes the A03 reading.
        dg["arm_prediction_shape"] = {}
        for lab, rows in rows_by_arm.items():
            pnorm = [normalize_answer(r["pred"]) for r in rows]
            pc = Counter(pnorm)
            top, tophits = pc.most_common(1)[0]
            mean_chars = float(np.mean([len(r["pred"]) for r in rows]))
            dg["arm_prediction_shape"][lab] = {
                "n_distinct_normalised_preds": len(pc),
                "modal_pred": top[:80],
                "modal_pred_share": tophits / len(rows),
                "empty_pred_rate": sum(1 for p in pnorm if not p) / len(rows),
                "mean_pred_chars": mean_chars,
            }
        # length-matched verbose `contains` null, per arm (the arms differ ~6x in
        # prediction length, and `contains` is length-sensitive).
        dg["lengthmatched_contains_null"] = {}
        for i, (lab, rows) in enumerate(rows_by_arm.items()):
            tgt = dg["arm_prediction_shape"][lab]["mean_pred_chars"]
            const, acc, nvec = lengthmatched_contains_null(ref_t, tgt)
            dg["lengthmatched_contains_null"][lab] = {
                "target_chars": tgt, "null_string_chars": len(const),
                "acc": acc, "null_string_head": const[:120]}
            V = np.array([float(r["contains"]) for r in rows])
            cells.append(cell(lab, task, "contains_lenmatched", V, nvec,
                              f"length-matched verbose constant "
                              f"({len(const)} chars)", 400 + i))
        diag[task] = {"n": n_t, **dg}

    # ---------------- BH over the whole family ----------------
    pv = [c["boot_p"] for c in cells]
    rej, adj, k = bh_reject(pv, q=0.05)
    for c, r, a in zip(cells, rej, adj):
        c["bh_reject_q05"] = bool(r)
        c["bh_adj_p"] = float(a)
        if not r:
            c["verdict"] = "AT floor (not distinguishable)"
        elif c["residual"] > 0:
            c["verdict"] = "ABOVE floor"
        else:
            c["verdict"] = "BELOW floor"

    result = {
        "n_boot": N_BOOT, "seed": SEED, "bh_q": 0.05, "bh_k": k,
        "n_cells": len(cells),
        "arms": {lab: d for lab, d in arms},
        "nulls": diag,
        "cells": cells,
    }

    # ---------------- print ----------------
    print(f"\nMMLU nulls: best-constant = always-{const_letter} {const_acc:.4f} "
          f"(chance .25 is NOT the floor); longest-option split-tie = "
          f"{longest['split'].mean():.4f} (tied on {tie_rate:.4f} of items)")
    for t in ("popqa", "triviaqa"):
        if t in diag:
            print(f"{t} nulls: best-constant em='{diag[t]['em']['best_constant']}' "
                  f"{diag[t]['em']['acc']:.4f} | contains="
                  f"'{diag[t]['contains']['best_constant']}' "
                  f"{diag[t]['contains']['acc']:.4f}")
    hdr = (f"\n{'arm':<22}{'task':<14}{'iface':<13}{'reported':>9}{'null':>9}"
           f"{'resid':>9}{'frac':>8}{'boot_p':>10}{'bh_p':>10}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        fr = "n/a" if c["residual_fraction"] is None else f"{100*c['residual_fraction']:.1f}%"
        print(f"{c['arm']:<22}{c['task']:<14}{c['interface']:<13}"
              f"{c['reported']:>9.4f}{c['null']:>9.4f}{c['residual']:>+9.4f}"
              f"{fr:>8}{c['boot_p']:>10.2e}{c['bh_adj_p']:>10.2e}  {c['verdict']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"\nwrote {args.out}")
    return result


if __name__ == "__main__":
    main()
