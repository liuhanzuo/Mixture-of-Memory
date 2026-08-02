#!/usr/bin/env python3
"""Paper B P0.4 checkpoint-to-checkpoint paired MMLU (trajectory).

Given two per-example MMLU jsonl dumps (from
scripts/eval_olmo2_probe2_downstream.py --save_per_example), each carrying stable
strided item_id across shards, compute the paired McNemar + paired bootstrap for a
checkpoint-to-checkpoint pairing.

Convention: row = the LATER checkpoint (B), col = the EARLIER checkpoint (A).
  Delta_pp = 100 * (B_acc - A_acc)        (later minus earlier)
  b = # items B-correct AND A-wrong      (gains from training more)
  c = # items B-wrong   AND A-correct     (regressions)
  McNemar exact two-sided binomial p on the discordant pairs (b, c),
    n = b + c, using log-space lgamma (scipy not available).
  paired bootstrap 95% CI on Delta_pp over the shared item set,
    numpy.random.RandomState(1234), n_boot=10000.

per-item gold-NLL (if option_scores present): mean over items of
  -option_scores[gold_letter]  (gold sum-logprob; larger NLL = worse). Reported per
  checkpoint over the SHARED aligned item set, plus the paired mean delta (B - A).

No scipy. numpy only.
"""
import argparse
import json
import math

import numpy as np


def load_peritem(path):
    """path -> {item_id: record}. record has correct(bool), gold_letter,
    option_scores(dict letter->float|None), nan(bool)."""
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d[int(r["item_id"])] = r
    return d


def log_binom_coeff(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def mcnemar_exact_two_sided(b, c):
    """Exact two-sided McNemar: binomial test of b successes in n=b+c trials,
    p=0.5, two-sided = min(1, 2 * P(X <= min(b,c))). Log-space via lgamma."""
    n = b + c
    if n == 0:
        return 1.0, 0
    k = min(b, c)
    # P(X <= k) under Binom(n, 0.5)
    log_half_n = n * math.log(0.5)
    # sum_{i=0}^{k} C(n,i) * 0.5^n  ->  logsumexp of log C(n,i) + n*log0.5
    terms = [log_binom_coeff(n, i) + log_half_n for i in range(k + 1)]
    m = max(terms)
    cdf = math.exp(m + math.log(sum(math.exp(t - m) for t in terms)))
    p = min(1.0, 2.0 * cdf)
    return p, n


def paired_bootstrap_ci(a_correct, b_correct, n_boot=10000, seed=1234, alpha=0.05):
    """a_correct, b_correct: aligned 0/1 numpy arrays (same item order).
    Bootstrap Delta_pp = 100*(mean(b)-mean(a)) resampling item indices with
    replacement. RandomState(seed). Returns (lo, hi) percentile CI in pp."""
    rng = np.random.RandomState(seed)
    n = len(a_correct)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[i] = 100.0 * (b_correct[idx].mean() - a_correct[idx].mean())
    lo = np.percentile(diffs, 100 * (alpha / 2))
    hi = np.percentile(diffs, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def gold_nll(rec):
    """-option_scores[gold_letter] if available and finite, else None."""
    os_ = rec.get("option_scores")
    gl = rec.get("gold_letter")
    if not os_ or gl is None or gl not in os_:
        return None
    v = os_[gl]
    if v is None:
        return None
    return -float(v)


def analyze_pair(name, path_a, path_b, label_a, label_b, n_boot, seed):
    """A = earlier ckpt, B = later ckpt (row). Delta = B - A."""
    A = load_peritem(path_a)
    B = load_peritem(path_b)
    shared = sorted(set(A) & set(B))
    only_a = set(A) - set(B)
    only_b = set(B) - set(A)

    a_corr, b_corr = [], []
    b_gain = c_reg = 0  # b: B-correct & A-wrong; c: B-wrong & A-correct
    a_nll_sum = b_nll_sum = 0.0
    nll_n = 0
    for iid in shared:
        ra, rb = A[iid], B[iid]
        ca = 1 if ra["correct"] else 0
        cb = 1 if rb["correct"] else 0
        a_corr.append(ca)
        b_corr.append(cb)
        if cb == 1 and ca == 0:
            b_gain += 1
        elif cb == 0 and ca == 1:
            c_reg += 1
        na, nb = gold_nll(ra), gold_nll(rb)
        if na is not None and nb is not None:
            a_nll_sum += na
            b_nll_sum += nb
            nll_n += 1

    a_corr = np.array(a_corr, dtype=np.float64)
    b_corr = np.array(b_corr, dtype=np.float64)
    n = len(shared)
    a_acc = float(a_corr.mean()) if n else float("nan")
    b_acc = float(b_corr.mean()) if n else float("nan")
    delta_pp = 100.0 * (b_acc - a_acc)
    p, ndisc = mcnemar_exact_two_sided(b_gain, c_reg)
    lo, hi = paired_bootstrap_ci(a_corr, b_corr, n_boot=n_boot, seed=seed)

    res = {
        "pair": name,
        "A_label": label_a, "B_label": label_b,
        "n_shared": n, "n_only_A": len(only_a), "n_only_B": len(only_b),
        "A_acc": a_acc, "B_acc": b_acc,
        "delta_pp_B_minus_A": delta_pp,
        "b_Bcorrect_Awrong": b_gain, "c_Bwrong_Acorrect": c_reg,
        "n_discordant": ndisc,
        "mcnemar_exact_two_sided_p": p,
        "bootstrap_ci95_pp": [lo, hi],
        "bootstrap_seed": seed, "n_boot": n_boot,
    }
    if nll_n:
        res["gold_nll_A_mean"] = a_nll_sum / nll_n
        res["gold_nll_B_mean"] = b_nll_sum / nll_n
        res["gold_nll_delta_B_minus_A"] = (b_nll_sum - a_nll_sum) / nll_n
        res["gold_nll_n"] = nll_n
    return res


def fmt(res):
    L = []
    L.append(f"=== PAIR: {res['pair']} ===")
    L.append(f"  A (earlier) = {res['A_label']}   B (later, row) = {res['B_label']}")
    L.append(f"  n_shared={res['n_shared']}  (only_A={res['n_only_A']} only_B={res['n_only_B']})")
    L.append(f"  A_acc={res['A_acc']:.6f}  B_acc={res['B_acc']:.6f}  "
             f"Delta_pp(B-A)={res['delta_pp_B_minus_A']:+.4f}")
    L.append(f"  b(B-correct,A-wrong)={res['b_Bcorrect_Awrong']}  "
             f"c(B-wrong,A-correct)={res['c_Bwrong_Acorrect']}  "
             f"n_discordant={res['n_discordant']}")
    L.append(f"  McNemar exact two-sided p = {res['mcnemar_exact_two_sided_p']:.6g}")
    lo, hi = res["bootstrap_ci95_pp"]
    L.append(f"  paired bootstrap 95% CI (pp, seed={res['bootstrap_seed']}, "
             f"n_boot={res['n_boot']}) = [{lo:+.4f}, {hi:+.4f}]")
    if "gold_nll_A_mean" in res:
        L.append(f"  mean gold-NLL: A={res['gold_nll_A_mean']:.5f}  "
                 f"B={res['gold_nll_B_mean']:.5f}  "
                 f"delta(B-A)={res['gold_nll_delta_B_minus_A']:+.5f}  "
                 f"(n={res['gold_nll_n']})")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", action="append", nargs=5,
                   metavar=("NAME", "PATH_A_EARLIER", "PATH_B_LATER",
                            "LABEL_A", "LABEL_B"),
                   required=True,
                   help="one pairing: NAME earlierJsonl laterJsonl labelA labelB")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--json_out", type=str, default="")
    args = p.parse_args()

    results = []
    for name, pa, pb, la, lb in args.pair:
        r = analyze_pair(name, pa, pb, la, lb, args.n_boot, args.seed)
        results.append(r)
        print(fmt(r))
        print()

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[wrote] {args.json_out}")


if __name__ == "__main__":
    main()
