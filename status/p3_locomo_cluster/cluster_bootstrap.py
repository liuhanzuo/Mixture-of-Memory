#!/usr/bin/env python
"""P3.3 — LoCoMo judge paired-difference bootstrap at the CONVERSATION-CLUSTER level.

Reviewer critique: the stats appendix bootstraps the CoMem-vs-KV-Direct LoCoMo
GPT-4o-judge difference by resampling the 1,540 nested cat1-4 questions as if
they were independent. They are not — questions are nested within 10 LoCoMo
conversations, so the effective sample size is closer to 10 clusters. This
recomputes the paired difference with a CONVERSATION-level cluster bootstrap
(resample the 10 conversations with replacement) and compares it against the
naive question-level bootstrap the paper currently reports.

Paired design: both methods are judged on the SAME question set, so the paired
per-question diff d(id) = judge_CoMem(id) - judge_KVD(id) in {-1,0,+1}, and
mean(d) == mean(CoMem) - mean(KVD) exactly (in judge-accuracy points x100).
"""
import glob
import json
import os
import random
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
COMEM = os.path.join(ROOT, "locomo_results", "qcmem_8b_iter_chatFALSE")   # 38.27
KVD = os.path.join(ROOT, "locomo_results", "kvdirect_8b_chatFALSE")        # 34.59
B = 20000
SEED = 12345


def load_run(d):
    """Return {id: {'cat': int, 'abst': bool, 'judge': float or None}}."""
    rec = {}
    for f in sorted(glob.glob(os.path.join(d, "preds_shard*.jsonl"))):
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
    # 'conv0_qa48' -> 'conv0'
    return qid.split("_", 1)[0]


def main():
    comem = load_run(COMEM)
    kvd = load_run(KVD)

    # matched ids present (with a judge verdict) in BOTH runs
    common = [i for i in comem if i in kvd
              and comem[i]["judge"] is not None and kvd[i]["judge"] is not None]

    # cat1-4 denominator (the paper's n=1540 headline for the +4.81 claim)
    cat14 = [i for i in common if comem[i]["cat"] in (1, 2, 3, 4)]
    allc = list(common)

    def summarize(ids, label):
        by_conv = defaultdict(list)          # conv -> list of paired diffs
        c_sum = defaultdict(float); k_sum = defaultdict(float)
        for i in ids:
            c = comem[i]["judge"]; k = kvd[i]["judge"]
            cv = conv_of(i)
            by_conv[cv].append(c - k)
            c_sum[cv] += c; k_sum[cv] += k
        diffs = [d for lst in by_conv.values() for d in lst]
        n = len(diffs)
        convs = sorted(by_conv.keys())
        point = 100.0 * sum(diffs) / n
        comem_mean = 100.0 * sum(comem[i]["judge"] for i in ids) / n
        kvd_mean = 100.0 * sum(kvd[i]["judge"] for i in ids) / n

        print(f"\n===== {label}  (n={n} questions, {len(convs)} conversations) =====")
        print(f"CoMem judge = {comem_mean:.2f}   KVD judge = {kvd_mean:.2f}   "
              f"paired diff (point) = {point:+.2f}")
        print(" per-conversation diff (CoMem - KVD, judge-acc pts):")
        for cv in convs:
            dl = by_conv[cv]
            print(f"   {cv:>7}: n={len(dl):>4}  CoMem={100*c_sum[cv]/len(dl):6.2f}  "
                  f"KVD={100*k_sum[cv]/len(dl):6.2f}  diff={100*sum(dl)/len(dl):+7.2f}")

        # ---- (A) naive question-level bootstrap (what the paper reports) ----
        rng = random.Random(SEED)
        qmeans = []
        for _ in range(B):
            s = rng.choices(diffs, k=n)
            qmeans.append(100.0 * sum(s) / n)
        qmeans.sort()
        q_lo = qmeans[int(0.025 * B)]; q_hi = qmeans[int(0.975 * B)]
        q_p = 2.0 * min(sum(1 for m in qmeans if m <= 0),
                        sum(1 for m in qmeans if m >= 0)) / B

        # ---- (B) conversation-cluster bootstrap (resample the 10 convs) ----
        rng2 = random.Random(SEED + 1)
        conv_diffs = {cv: by_conv[cv] for cv in convs}
        cmeans = []
        for _ in range(B):
            chosen = rng2.choices(convs, k=len(convs))
            tot = 0.0; cnt = 0
            for cv in chosen:
                lst = conv_diffs[cv]
                tot += sum(lst); cnt += len(lst)
            cmeans.append(100.0 * tot / cnt)
        cmeans.sort()
        c_lo = cmeans[int(0.025 * B)]; c_hi = cmeans[int(0.975 * B)]
        c_p = 2.0 * min(sum(1 for m in cmeans if m <= 0),
                        sum(1 for m in cmeans if m >= 0)) / B

        print(f" (A) question-level bootstrap  95% CI = [{q_lo:+.2f}, {q_hi:+.2f}]  "
              f"p~{q_p:.4f}")
        print(f" (B) conversation-cluster CI  95% CI = [{c_lo:+.2f}, {c_hi:+.2f}]  "
              f"p~{c_p:.4f}")
        return {"label": label, "n": n, "n_conv": len(convs),
                "comem": round(comem_mean, 2), "kvd": round(kvd_mean, 2),
                "point_diff": round(point, 2),
                "question_ci": [round(q_lo, 2), round(q_hi, 2)], "question_p": round(q_p, 4),
                "cluster_ci": [round(c_lo, 2), round(c_hi, 2)], "cluster_p": round(c_p, 4),
                "per_conv": {cv: {"n": len(by_conv[cv]),
                                  "comem": round(100 * c_sum[cv] / len(by_conv[cv]), 2),
                                  "kvd": round(100 * k_sum[cv] / len(by_conv[cv]), 2),
                                  "diff": round(100 * sum(by_conv[cv]) / len(by_conv[cv]), 2)}
                             for cv in convs}}

    print(f"matched judged questions (both runs): {len(common)}  "
          f"| cat1-4: {len(cat14)}")
    res_cat14 = summarize(cat14, "cat1-4 (headline denominator, matches +4.81 claim)")
    res_all = summarize(allc, "all cats (n~1986 denominator)")

    out = {"B": B, "seed": SEED,
           "comem_dir": "qcmem_8b_iter_chatFALSE (judge 38.27)",
           "kvd_dir": "kvdirect_8b_chatFALSE (judge 34.59)",
           "cat1_4": res_cat14, "all": res_all}
    with open(os.path.join(ROOT, "cluster_bootstrap_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n[done] wrote cluster_bootstrap_result.json")


if __name__ == "__main__":
    main()
