#!/usr/bin/env python
"""LoCoMo CoMem - KV-Direct judge-difference bootstrap (Paper A, P0.9).

Two paired bootstraps of the CoMem(+LoRA) vs KV-Direct LoCoMo GPT-4o-judge
difference, over the n=1,540 common judged items (categories 1-4):

  (A) PER-ITEM bootstrap  -- resample the 1,540 questions with replacement.
      This is the interval already reported in the statistics appendix
      (+4.81, 95% CI [2.34, 7.27]); recomputed here as a reproduction GATE.

  (B) CONVERSATION-CLUSTER bootstrap -- resample the 10 LoCoMo conversations
      with replacement, pool the items of the drawn conversations, and compute
      mean(CoMem) - mean(KVD) over that pooled item set. Questions are nested
      within only 10 conversations, so this is the dependence-aware robustness
      check; it does NOT assume the 1,540 questions are independent.

Paired design: both methods are judged on the SAME question set, so per-item
d(id) = judge_CoMem(id) - judge_KVD(id) in {-1,0,+1}, and for any fixed pooled
item set S, mean_S(CoMem) - mean_S(KVD) == mean_S(d) exactly (judge-acc pts x100).

Both bootstraps use B=10,000 resamples, seed=1234, and the 2.5 / 97.5 percentiles.

Sources (chat_template=False, selector=iter_bm25, GPT-4o judge; see
status/PAPER_LOCOMO_ERRATA_20260721.md and status/08_statistics_appendix.tex):
  CoMem+LoRA flagship : locomo_results/qcmem_8b_iter_chatFALSE  (overall judge 38.27)
  KV-Direct           : locomo_results/kvdirect_8b_chatFALSE    (overall judge 34.59)
Per-item judge verdicts come from each dir's judge_cache.jsonl (1,540 cat1-4
API-judged items); category / conversation grouping from the preds_shard*.jsonl.
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The flagship chat=False judged shards live in the staged P3 cluster dir on
# this checkout; fall back to the top-level locomo_results/ layout if present.
_CANDIDATE_ROOTS = [
    os.path.join(PROJECT_ROOT, "status", "p3_locomo_cluster", "locomo_results"),
    os.path.join(PROJECT_ROOT, "locomo_results"),
]

COMEM_NAME = "qcmem_8b_iter_chatFALSE"
KVD_NAME = "kvdirect_8b_chatFALSE"

B = 10000
SEED = 1234


def _resolve(name, roots):
    for r in roots:
        d = os.path.join(r, name)
        if os.path.isdir(d):
            return d
    raise SystemExit(f"[locomo-cluster] could not find run dir {name!r} under "
                     f"any of {roots}")


def load_run(d):
    """Return {id: {'cat': int, 'abst': bool, 'judge': float or None}}.

    Predictions give the (id -> category, is_abstention) map; judge_cache gives
    the GPT-4o CORRECT/WRONG verdict (1.0/0.0) for the cat1-4 answerable items.
    """
    rec = {}
    for f in sorted(glob.glob(os.path.join(d, "preds_shard*.jsonl"))
                    or glob.glob(os.path.join(d, "preds*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                rec[o["id"]] = {"cat": int(o.get("category", -1)),
                                "abst": bool(o.get("is_abstention", False)),
                                "judge": None}
    jc = os.path.join(d, "judge_cache.jsonl")
    with open(jc) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o["id"] in rec:
                rec[o["id"]]["judge"] = float(o["judge"])
    return rec


def conv_of(qid):
    """'conv0_qa48' -> 'conv0' (the LoCoMo conversation the question is nested in)."""
    return qid.split("_", 1)[0]


def percentile_ci(sorted_vals, b):
    """2.5 / 97.5 percentile CI from a sorted bootstrap distribution of length b."""
    lo = sorted_vals[int(0.025 * b)]
    hi = sorted_vals[int(0.975 * b)]
    return lo, hi


def two_sided_p(vals, b):
    """Bootstrap two-sided p that the paired diff is zero."""
    return 2.0 * min(sum(1 for m in vals if m <= 0),
                     sum(1 for m in vals if m >= 0)) / b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None,
                    help="Override dir that holds the two run subdirs "
                         "(default: staged status/p3_locomo_cluster/locomo_results "
                         "or top-level locomo_results).")
    ap.add_argument("--B", type=int, default=B)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    roots = [args.data_root] if args.data_root else _CANDIDATE_ROOTS
    comem_dir = _resolve(COMEM_NAME, roots)
    kvd_dir = _resolve(KVD_NAME, roots)

    comem = load_run(comem_dir)
    kvd = load_run(kvd_dir)

    # matched ids present WITH a judge verdict in BOTH runs
    common = sorted(i for i in comem if i in kvd
                    and comem[i]["judge"] is not None
                    and kvd[i]["judge"] is not None)
    # the appendix's n=1540 headline denominator = categories 1-4
    ids = [i for i in common if comem[i]["cat"] in (1, 2, 3, 4)]

    # paired per-item diffs, grouped by conversation, in a STABLE id order
    by_conv = defaultdict(list)         # conv -> [paired diff]
    c_sum = defaultdict(float)          # conv -> sum CoMem judge
    k_sum = defaultdict(float)          # conv -> sum KVD judge
    diffs = []                          # flat, id-ordered
    for i in ids:
        c = comem[i]["judge"]
        k = kvd[i]["judge"]
        cv = conv_of(i)
        by_conv[cv].append(c - k)
        c_sum[cv] += c
        k_sum[cv] += k
        diffs.append(c - k)
    convs = sorted(by_conv.keys())
    n = len(diffs)

    comem_mean = 100.0 * sum(comem[i]["judge"] for i in ids) / n
    kvd_mean = 100.0 * sum(kvd[i]["judge"] for i in ids) / n
    point = 100.0 * sum(diffs) / n

    print(f"[locomo-cluster] CoMem dir : {comem_dir}")
    print(f"[locomo-cluster] KVD   dir : {kvd_dir}")
    print(f"[locomo-cluster] matched judged (both runs): {len(common)} | "
          f"cat1-4 headline n = {n} across {len(convs)} conversations")
    print(f"[locomo-cluster] CoMem judge = {comem_mean:.2f}   "
          f"KVD judge = {kvd_mean:.2f}   paired point diff = {point:+.2f}")

    # ---- per-conversation observed diffs ----
    n_favor = 0
    per_conv = {}
    print("[locomo-cluster] per-conversation mean diff (CoMem - KVD, judge pts):")
    for cv in convs:
        dl = by_conv[cv]
        cm = 100.0 * c_sum[cv] / len(dl)
        km = 100.0 * k_sum[cv] / len(dl)
        dv = 100.0 * sum(dl) / len(dl)
        if dv > 0:
            n_favor += 1
        per_conv[cv] = {"n": len(dl), "comem": round(cm, 2),
                        "kvd": round(km, 2), "diff": round(dv, 2)}
        print(f"   {cv:>7}: n={len(dl):>4}  CoMem={cm:6.2f}  KVD={km:6.2f}  "
              f"diff={dv:+7.2f}")
    print(f"[locomo-cluster] conversations favoring CoMem: {n_favor}/{len(convs)}")

    # ---- (A) per-item bootstrap (reproduction gate of [2.34, 7.27]) ----
    rng = random.Random(args.seed)
    qmeans = []
    for _ in range(args.B):
        s = rng.choices(diffs, k=n)
        qmeans.append(100.0 * sum(s) / n)
    qmeans.sort()
    q_lo, q_hi = percentile_ci(qmeans, args.B)
    q_p = two_sided_p(qmeans, args.B)

    # ---- (B) conversation-cluster bootstrap (resample the 10 convs) ----
    rng2 = random.Random(args.seed)
    conv_diffs = {cv: by_conv[cv] for cv in convs}
    cmeans = []
    for _ in range(args.B):
        chosen = rng2.choices(convs, k=len(convs))
        tot = 0.0
        cnt = 0
        for cv in chosen:
            lst = conv_diffs[cv]
            tot += sum(lst)
            cnt += len(lst)
        cmeans.append(100.0 * tot / cnt)
    cmeans.sort()
    c_lo, c_hi = percentile_ci(cmeans, args.B)
    c_p = two_sided_p(cmeans, args.B)

    print(f"\n[locomo-cluster] B={args.B}  seed={args.seed}")
    print(f" (A) per-item bootstrap        point={point:+.2f}  "
          f"95% CI = [{q_lo:+.2f}, {q_hi:+.2f}]  p~{q_p:.4f}")
    print(f" (B) conversation-cluster boot  point={point:+.2f}  "
          f"95% CI = [{c_lo:+.2f}, {c_hi:+.2f}]  p~{c_p:.4f}")

    out = {
        "B": args.B, "seed": args.seed,
        "comem_dir": os.path.relpath(comem_dir, PROJECT_ROOT),
        "kvd_dir": os.path.relpath(kvd_dir, PROJECT_ROOT),
        "n_common_judged": len(common),
        "n_cat14": n, "n_conversations": len(convs),
        "comem_judge": round(comem_mean, 2), "kvd_judge": round(kvd_mean, 2),
        "point_diff": round(point, 2),
        "per_item_ci": [round(q_lo, 2), round(q_hi, 2)], "per_item_p": round(q_p, 4),
        "cluster_ci": [round(c_lo, 2), round(c_hi, 2)], "cluster_p": round(c_p, 4),
        "cluster_ci_above_zero": bool(c_lo > 0),
        "n_conv_favor_comem": n_favor,
        "per_conv": per_conv,
    }
    out_path = os.path.join(PROJECT_ROOT, "status", "P0_9_CLUSTER_BOOTSTRAP.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[locomo-cluster] wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
