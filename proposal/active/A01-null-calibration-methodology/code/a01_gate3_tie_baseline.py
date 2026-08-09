#!/usr/bin/env python3
"""A01 gate-3, step 1: the bf16 exact-tie BASELINE, recomputed from scratch.

Everything printed here is recomputed from the raw per-example jsonl
(`<results_root>/<arm>/per_example_mmlu.jsonl`).  Nothing is copied from any
markdown.  This file is the *control* against which the fp32 forward
(gate-3 step 2) is compared.

What it computes, per arm
------------------------
* letter / content_norm accuracy over the valid item set
* exact-tie statistics on the LETTER interface:
    - tie2 = top1 == top2 exactly  (argmax then breaks the tie by INDEX, which
      is input-blind -> this is the claimed mechanism)
    - the full multiplicity histogram (how many of the 4 letters are tied at
      the maximum)
    - all-4-tied rate (the fully degenerate case)
    - "drop-tied-items" letter accuracy (accuracy restricted to items with a
      strict argmax), and the "index-luck" accuracy contributed by tied items
* the two construct-appropriate nulls, INDEPENDENTLY re-derived:
    - best-constant letter floor (gold-letter marginals; dataset-determined)
    - longest-option floor, all five tie conventions (tokenizer-determined)
* a quantisation audit: the histogram of the top1-top2 gap, to show that
  the stored 6-dp rounding cannot manufacture ties (bf16 logit spacing at
  these magnitudes is ~1e-2, four orders of magnitude above the rounding).

Usage
-----
  python3 a01_gate3_tie_baseline.py [--results-root olmo2_mmlu_content_results]
                                    [--arms tag1,tag2,...]
                                    [--out out.json]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

LETTERS = "ABCD"
TIE_CONVS = ("split", "first", "last", "credit", "wrong")

# canonical 10-arm OLMo-2 set (the C5/Obs4 set: C1's 9 arms + keep14-reheal)
DEFAULT_ARMS = [
    ("base (32L intact)", "7B_base"),
    ("full32 @25k", "7B_full32_step25000"),
    ("keep8 @121k", "7B_keep8_step121000"),
    ("keep10 @83.5k", "7B_keep10_step83500"),
    ("keep12 @124k", "7B_keep12_step124000"),
    ("keep14 @200k", "7B_keep14_step200000"),
    ("freezefront @200k", "7B_freezefront_step200000"),
    ("scratch16L @200k", "7B_scratch16L_step200000"),
    ("shortgpt16 @200k", "7B_shortgpt16_step200000"),
    ("keep14-reheal @67.5k", "7B_keep14_reheal_step67500"),
]


def load_arm(root, tag):
    p = os.path.join(root, tag, "per_example_mmlu.jsonl")
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["item_id"])
    return rows


def longest_option_vector(rows, gold, conv):
    """input-blind 'always pick the longest option' null, per item.

    'longest' = most CONTINUATION TOKENS under this model's tokenizer
    (content_norm.cont_tokens), which is the quantity the length-normalised
    content interface divides by.  Tokenizer-dependent, hence family-dependent.
    """
    out = np.zeros(len(rows))
    for i, r in enumerate(rows):
        c = r["content_norm"]["cont_tokens"]
        keys = [k for k in LETTERS if k in c]
        top = max(c[k] for k in keys)
        win = [k for k in keys if c[k] == top]
        g = gold[i]
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


def tie_stats(rows):
    """Exact-tie statistics on the letter interface + the gap histogram."""
    n = len(rows)
    mult_hist = Counter()
    tie2 = 0
    tie_correct = 0        # tied item where argmax(index) happened to be right
    strict_n = 0
    strict_correct = 0
    gaps = []
    gold_in_tieset = 0
    for r in rows:
        sc = r["letter"]["scores"]
        keys = [k for k in LETTERS if k in sc and sc[k] is not None]
        vals = [sc[k] for k in keys]
        mx = max(vals)
        win = [k for k, v in zip(keys, vals) if v == mx]
        m = len(win)
        mult_hist[m] += 1
        srt = sorted(vals, reverse=True)
        gap = srt[0] - srt[1] if len(srt) > 1 else float("inf")
        gaps.append(gap)
        if m >= 2:
            tie2 += 1
            if r["letter"]["correct"]:
                tie_correct += 1
            if r["gold_letter"] in win:
                gold_in_tieset += 1
        else:
            strict_n += 1
            if r["letter"]["correct"]:
                strict_correct += 1
    gaps = np.asarray(gaps, dtype=float)
    finite = gaps[np.isfinite(gaps)]
    pos = finite[finite > 0]
    return {
        "n": n,
        "tie2_count": tie2,
        "tie2_rate": tie2 / n,
        "multiplicity_hist": {str(k): mult_hist[k] for k in sorted(mult_hist)},
        "all4_tied_count": mult_hist.get(4, 0),
        "all4_tied_rate": mult_hist.get(4, 0) / n,
        "strict_n": strict_n,
        "strict_letter_acc": (strict_correct / strict_n) if strict_n else None,
        "tied_item_index_luck_acc": (tie_correct / tie2) if tie2 else None,
        "gold_in_tieset_rate_among_tied": (gold_in_tieset / tie2) if tie2 else None,
        # quantisation audit: how many *positive* gaps sit near the 6-dp rounding
        "positive_gap_min": float(pos.min()) if pos.size else None,
        "positive_gap_lt_1e5": int((pos < 1e-5).sum()),
        "positive_gap_lt_1e3": int((pos < 1e-3).sum()),
        "positive_gap_p05": float(np.percentile(pos, 5)) if pos.size else None,
        "positive_gap_median": float(np.median(pos)) if pos.size else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="olmo2_mmlu_content_results")
    ap.add_argument("--arms", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.arms:
        arms = [(t, t) for t in a.arms.split(",") if t]
    else:
        arms = DEFAULT_ARMS

    data = {}
    for label, tag in arms:
        data[label] = load_arm(a.results_root, tag)

    # ---- alignment gate (never merge half a set) ----
    ref = data[arms[0][0]]
    n = len(ref)
    ref_ids = [r["item_id"] for r in ref]
    gold_seq = [r["gold_letter"] for r in ref]
    for label, rows in data.items():
        assert len(rows) == n, f"{label}: n={len(rows)} != {n}"
        assert [r["item_id"] for r in rows] == ref_ids, f"{label}: item_id misaligned"
        assert [r["gold_letter"] for r in rows] == gold_seq, f"{label}: gold misaligned"
        nn = sum(1 for r in rows if r.get("nan"))
        assert nn == 0, f"{label}: {nn} nan rows"
    print(f"[gate] {len(data)} arms x n={n} items, item_ids + gold aligned, 0 nan\n")

    # ---- null 1: best constant letter (dataset-determined, model-free) ----
    gc = Counter(gold_seq)
    marg = {k: gc.get(k, 0) / n for k in LETTERS}
    const_letter, hits = max(gc.items(), key=lambda kv: kv[1])
    const_acc = hits / n
    print("null-1  best-constant letter floor (gold-letter marginals):")
    for k in LETTERS:
        print(f"          always-{k}  {marg[k]:.4f}   ({gc.get(k,0)}/{n})")
    print(f"        => BEST CONSTANT = always-{const_letter} = {const_acc:.6f}\n")

    # ---- null 2: longest option, all conventions (tokenizer-determined) ----
    base_rows = data[arms[0][0]]
    longest = {c: float(longest_option_vector(base_rows, gold_seq, c).mean())
               for c in TIE_CONVS}
    n_tied_long = sum(
        1 for r in base_rows
        if sum(1 for k in LETTERS
               if k in r["content_norm"]["cont_tokens"]
               and r["content_norm"]["cont_tokens"][k]
               == max(r["content_norm"]["cont_tokens"].values())) >= 2)
    print("null-2  longest-option floor (cont_tokens), by tie convention:")
    for c in TIE_CONVS:
        star = "  <-- pre-registered" if c == "split" else ""
        print(f"          {c:8s} {longest[c]:.6f}{star}")
    print(f"        tied-longest items: {n_tied_long}/{n} = {n_tied_long/n:.4f} "
          f"(convention is load-bearing)\n")

    # sanity: the cont_tokens vector must be identical across arms of one family
    ct_ref = [tuple(sorted(r["content_norm"]["cont_tokens"].items()))
              for r in base_rows]
    for label, rows in data.items():
        ct = [tuple(sorted(r["content_norm"]["cont_tokens"].items())) for r in rows]
        assert ct == ct_ref, f"{label}: cont_tokens differ from base (tokenizer drift!)"
    print("[gate] cont_tokens identical across all arms -> one tokenizer, "
          "one longest-option floor for the family\n")

    # ---- per-arm accuracies + tie statistics ----
    out_arms = {}
    hdr = (f"{'arm':24s} {'letter':>7s} {'content':>8s} {'tie2':>7s} "
           f"{'all4':>7s} {'strictAcc':>9s} {'tieLuck':>8s} {'goldInTie':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for label, rows in data.items():
        L = float(np.mean([1.0 if r["letter"]["correct"] else 0.0 for r in rows]))
        C = float(np.mean([1.0 if r["content_norm"]["correct"] else 0.0 for r in rows]))
        ts = tie_stats(rows)
        out_arms[label] = {
            "letter_acc": L, "content_norm_acc": C,
            "letter_vs_const_pp": 100 * (L - const_acc),
            "content_vs_longest_pp": 100 * (C - longest["split"]),
            **ts,
        }
        print(f"{label:24s} {L:7.4f} {C:8.4f} {ts['tie2_rate']:7.4f} "
              f"{ts['all4_tied_rate']:7.4f} "
              f"{(ts['strict_letter_acc'] or 0):9.4f} "
              f"{(ts['tied_item_index_luck_acc'] or 0):8.4f} "
              f"{(ts['gold_in_tieset_rate_among_tied'] or 0):9.4f}")

    print("\nquantisation audit (can 6-dp rounding manufacture ties?):")
    for label, d in out_arms.items():
        print(f"  {label:24s} min positive gap={d['positive_gap_min']:.6f} "
              f"#gap<1e-5={d['positive_gap_lt_1e5']} "
              f"#gap<1e-3={d['positive_gap_lt_1e3']} "
              f"p05={d['positive_gap_p05']:.4f}")

    res = {
        "results_root": a.results_root,
        "n_items": n,
        "gold_letter_marginals": marg,
        "best_constant_letter": const_letter,
        "best_constant_floor": const_acc,
        "longest_option_floor_by_conv": longest,
        "longest_option_tied_items": n_tied_long,
        "arms": out_arms,
    }
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[wrote] {a.out}")


if __name__ == "__main__":
    main()
