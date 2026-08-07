#!/usr/bin/env python3
"""Paired pre->post SFT analysis for the Paper B P2.4 keep8fresh2 arm (.73 / zwfy6).

Pairs the arm's OWN pre-SFT anchor against its post-SFT ckpt, item-by-item
(item_id-keyed), and reports for every metric:
  * pre acc, post acc, delta
  * McNemar EXACT test, TWO-SIDED (exact binomial on the discordant pairs
    b vs c under H0: p=0.5). Two-sided is the honest choice: we did not
    pre-register a direction for the downstream metrics.
  * paired bootstrap 95% CI on the per-item delta (10k resamples, seed 0)

Pre  : 7B_keep8_step121000_v2          (outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt)
Post : 7B_p24_sft_keep8fresh2_final    (outputs/olmo2_p24_sft_keep8fresh2/final.pt)

CAVEAT carried in the output JSON: keep8's pre-SFT anchor is step121000, NOT
200k (keep8 never reached 200k -- status/PAPERB_TABLE4_BUDGET_DEFECT.md). Valid
here because we hold the arm's own pre-SFT ckpt fixed and measure the SFT delta
on it; NOT valid for compute-matched depth comparisons vs keep14 (200k).

Metric conventions (match the paper tables):
  core6      -> acc_norm  (per-item binary field `acc_norm_score`)
  know5      -> acc_norm
  MMLU       -> letter protocol = letter.correct ; content protocol = content_norm.correct
  closedbook -> em and contains (binary per item); f1 is continuous -> bootstrap only

No scipy on .73, so the exact binomial tail is computed with math.comb.
CPU only. Run on .73 after the eval battery finishes.
"""
import json, os, math, argparse
from fractions import Fraction

import numpy as np

ROOT = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
PRE = "7B_keep8_step121000_v2"
POST = "7B_p24_sft_keep8fresh2_final"
CORE6 = ["hellaswag", "arc_challenge", "arc_easy", "piqa", "winogrande", "openbookqa"]
KNOW5 = ["mmlu", "lambada_openai", "boolq", "commonsense_qa", "social_iqa"]
N_BOOT = 10000
SEED = 0


def exact_binom_two_sided(b, c):
    """Exact two-sided McNemar: P(|X - n/2| >= |b - n/2|), X~Bin(n=b+c, 0.5).

    Equivalent to the standard 'exact binomial test at p=0.5' two-sided rule of
    summing all outcomes at least as extreme as the observed one. Computed with
    exact integer/rational arithmetic so it needs no scipy AND does not overflow
    for large n (2**n exceeds float range once n > ~1024, which happens easily:
    MMLU discordant pairs run into the thousands).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided = 2 * lower tail, capped at 1 (symmetric null at p=0.5)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    p = Fraction(2 * tail, 2 ** n)
    if p >= 1:
        return 1.0
    # Fraction -> float can still underflow to 0.0 for astronomically small p;
    # that is the correct IEEE754 representation and matches scipy's behaviour.
    return float(p)


def load_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["item_id"]] = d
    return out


def paired_stats(pre_vals, post_vals, binary=True):
    """pre_vals/post_vals: aligned np arrays over the SAME item ids."""
    n = len(pre_vals)
    res = {
        "n_paired": int(n),
        "pre": float(pre_vals.mean()) if n else 0.0,
        "post": float(post_vals.mean()) if n else 0.0,
    }
    res["delta"] = res["post"] - res["pre"]
    if binary:
        pc = pre_vals.astype(int)
        qc = post_vals.astype(int)
        # b = post correct & pre wrong (SFT gained); c = post wrong & pre correct (SFT lost)
        b = int(np.sum((qc == 1) & (pc == 0)))
        c = int(np.sum((qc == 0) & (pc == 1)))
        res["mcnemar_b_post_gain"] = b
        res["mcnemar_c_post_loss"] = c
        res["mcnemar_exact_p"] = exact_binom_two_sided(b, c)
        res["mcnemar_sidedness"] = "two-sided"
    diffs = post_vals.astype(float) - pre_vals.astype(float)
    if n:
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        bs = diffs[idx].mean(axis=1)
        res["bootstrap_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res["n_boot"] = N_BOOT
    else:
        res["bootstrap_ci95"] = [0.0, 0.0]
    return res


def pair_field(pre_dir, post_dir, task, extract, drop_nan=True):
    """Returns (pre_arr, post_arr, item_ids) aligned on common item_ids."""
    pp = os.path.join(pre_dir, f"per_example_{task}.jsonl")
    qp = os.path.join(post_dir, f"per_example_{task}.jsonl")
    if not (os.path.exists(pp) and os.path.exists(qp)):
        return None
    P, Q = load_jsonl(pp), load_jsonl(qp)
    common = sorted(set(P) & set(Q))
    if drop_nan:
        common = [i for i in common if not P[i].get("nan", False) and not Q[i].get("nan", False)]
    pre = np.array([extract(P[i]) for i in common], dtype=float)
    post = np.array([extract(Q[i]) for i in common], dtype=float)
    return pre, post, common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default=f"{ROOT}/results/paperb_p24_sft_keep8_paired.json")
    args = ap.parse_args()

    DS_PRE = f"{ROOT}/olmo2_downstream_results/{PRE}"
    DS_POST = f"{ROOT}/olmo2_downstream_results/{POST}"
    KN_PRE = f"{ROOT}/olmo2_downstream_results/{PRE}_know"
    KN_POST = f"{ROOT}/olmo2_downstream_results/{POST}_know"
    MM_PRE = f"{ROOT}/olmo2_mmlu_content_results/{PRE}"
    MM_POST = f"{ROOT}/olmo2_mmlu_content_results/{POST}"
    CB_PRE = f"{ROOT}/olmo2_closedbook_results/{PRE}"
    CB_POST = f"{ROOT}/olmo2_closedbook_results/{POST}"

    out = {
        "pre_name": PRE, "post_name": POST,
        "pre_ckpt": "outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt",
        "post_ckpt": "outputs/olmo2_p24_sft_keep8fresh2/final.pt",
        "node": ".73 (H20 cc9.0, zwfy6)",
        "protocol": "chat_template=False, add_bos=0, 8-shard merge (8/8 asserted)",
        "mcnemar": "exact binomial on discordant pairs, TWO-SIDED",
        "caveat": ("keep8 pre-SFT anchor is step121000, NOT 200k (keep8 never reached 200k; "
                   "status/PAPERB_TABLE4_BUDGET_DEFECT.md). Valid for the SFT-delta-on-own-ckpt "
                   "contrast measured here; NOT valid for compute-matched depth comparisons "
                   "against keep14 (200k)."),
        "ppl": {}, "core6": {}, "know5": {}, "mmlu": {}, "closedbook": {},
    }

    # ---------------- PPL (corpus-level, no pairing) ----------------
    try:
        pre_ppl = json.load(open(f"{ROOT}/olmo2_ppl_results/{PRE}/summary.json"))
        post_ppl = json.load(open(f"{ROOT}/olmo2_ppl_results/{POST}/summary.json"))
        p0, p1 = pre_ppl["ppl"], post_ppl["ppl"]
        out["ppl"] = {
            "pre": p0, "post": p1,
            "delta_pct": 100.0 * (p1 - p0) / p0,
            "predicted_delta_pct": 14.0,
            "predicted_post_ppl": 15.20,
            "pre_tokens": pre_ppl["n_tokens"], "post_tokens": post_ppl["n_tokens"],
            "same_val_tokens": pre_ppl["n_tokens"] == post_ppl["n_tokens"],
            "pre_avg_nll": pre_ppl["avg_nll"], "post_avg_nll": post_ppl["avg_nll"],
        }
    except Exception as e:
        out["ppl"] = {"error": str(e)}

    # ---------------- core6 / know5 (acc_norm) ----------------
    for label, tasks, pd_, qd_ in (("core6", CORE6, DS_PRE, DS_POST),
                                   ("know5", KNOW5, KN_PRE, KN_POST)):
        per_task, boot_pool = {}, []
        for t in tasks:
            got = pair_field(pd_, qd_, t, lambda d: d["acc_norm_score"])
            if got is None:
                per_task[t] = {"error": "per-item file missing"}
                continue
            pre, post, _ = got
            per_task[t] = paired_stats(pre, post, binary=True)
            boot_pool.append((pre, post))
        out[label]["tasks"] = per_task
        # macro-average across tasks + paired bootstrap resampling WITHIN each task
        if len(boot_pool) == len(tasks):
            pre_macro = float(np.mean([p.mean() for p, _ in boot_pool]))
            post_macro = float(np.mean([q.mean() for _, q in boot_pool]))
            rng = np.random.default_rng(SEED)
            acc = np.zeros(N_BOOT)
            for pre, post in boot_pool:
                n = len(pre)
                idx = rng.integers(0, n, size=(N_BOOT, n))
                acc += (post - pre)[idx].mean(axis=1)
            acc /= len(boot_pool)
            out[label]["macro"] = {
                "pre": pre_macro, "post": post_macro, "delta": post_macro - pre_macro,
                "metric": "acc_norm, macro-avg over tasks",
                "bootstrap_ci95": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
                "n_boot": N_BOOT,
            }

    # ---------------- MMLU dual protocol ----------------
    for proto, extract in (("letter", lambda d: int(d["letter"]["correct"])),
                           ("content_norm", lambda d: int(d["content_norm"]["correct"])),
                           ("content_raw", lambda d: int(d["content_raw"]["correct"]))):
        got = pair_field(MM_PRE, MM_POST, "mmlu", extract)
        out["mmlu"][proto] = paired_stats(*got[:2], binary=True) if got else {"error": "missing"}

    # ---------------- closed-book QA ----------------
    for t in ("popqa", "triviaqa"):
        d = {}
        for metric, binary in (("em", True), ("contains", True), ("f1", False)):
            got = pair_field(CB_PRE, CB_POST, t, lambda x, m=metric: float(x[m]))
            d[metric] = paired_stats(*got[:2], binary=binary) if got else {"error": "missing"}
        out["closedbook"][t] = d

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n[written] {args.out_json}")


if __name__ == "__main__":
    main()
