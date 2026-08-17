#!/usr/bin/env python
"""Paired significance test for the B01 two executable arms, on LoCoMo.

The point estimates (arm1 F1 9.24 vs arm2 F1 3.85) are not a result until they are
paired and tested: both arms answered the SAME 1986 questions (verified by
b01_collect.py: sample-id sets identical), so the correct test is a PAIRED one, not a
two-sample test that throws the pairing away.

Reports, per metric and per category:
  * both arms' scores recomputed HERE from preds (not read from scores.json), so the
    pairing is over the same rows the test uses
  * the paired difference, a bootstrap CI over question-level paired deltas, and
    (for the binary accuracy metric) an exact McNemar test on discordant pairs

Scoring reuses the driver's OWN metric functions, so numbers reconcile with scores.json
rather than being a second, subtly different implementation.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

R = "/apdcephfs_wzz/share_303419932/pighzliu_code/Mixture-of-Memory"
sys.path.insert(0, R)


def load_arm(root, arm, num_shards):
    rows = {}
    for s in range(num_shards):
        p = os.path.join(root, arm, f"preds_shard{s}of{num_shards}.jsonl")
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arm_a", required=True)
    ap.add_argument("--arm_b", required=True)
    ap.add_argument("--num_shards", type=int, default=4)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json_out", required=True)
    args = ap.parse_args()

    # Use the driver's OWN score_sample(). It is the function run_scoring() calls, and
    # it handles the adversarial/abstention category specially (correct iff refused) --
    # a hand-rolled f1/em/acc would silently disagree with scores.json on 446 of 1986
    # questions. Reusing it is what makes this test reconcile.
    import scripts.eval_qcmem_locomo as drv
    score_sample = drv.score_sample
    print(f"[stats] using driver score_sample={score_sample}", flush=True)

    A = load_arm(args.root, args.arm_a, args.num_shards)
    B = load_arm(args.root, args.arm_b, args.num_shards)
    ids = sorted(set(A) & set(B))
    if len(ids) != len(A) or len(ids) != len(B):
        print(f"[stats] FATAL: id sets differ (A={len(A)} B={len(B)} common={len(ids)})")
        return 1
    print(f"[stats] paired on {len(ids)} questions", flush=True)

    def score(row):
        s = score_sample(row)
        return s["f1"], s["em"], s["acc"]

    per = {}
    for i in ids:
        per[i] = {"a": score(A[i]), "b": score(B[i]), "cat": str(A[i]["category"])}

    rng = random.Random(args.seed)
    metrics = ["f1", "em", "acc"]

    def summarise(subset_ids, label):
        n = len(subset_ids)
        out = {"label": label, "n": n}
        for mi, mname in enumerate(metrics):
            a = [per[i]["a"][mi] for i in subset_ids]
            b = [per[i]["b"][mi] for i in subset_ids]
            d = [x - y for x, y in zip(a, b)]
            ma, mb = sum(a) / n, sum(b) / n
            # paired bootstrap over question-level deltas
            boots = []
            for _ in range(args.n_boot):
                s = 0.0
                for _ in range(n):
                    s += d[rng.randrange(n)]
                boots.append(s / n)
            boots.sort()
            lo = boots[int(0.025 * len(boots))]
            hi = boots[int(0.975 * len(boots)) - 1]
            n_ge = sum(1 for v in boots if v <= 0.0)
            out[mname] = {
                f"{args.arm_a}_mean": ma * 100,
                f"{args.arm_b}_mean": mb * 100,
                "paired_diff_pp": (ma - mb) * 100,
                "boot_ci95_pp": [lo * 100, hi * 100],
                "boot_frac_le_zero": n_ge / len(boots),
                "ci_excludes_zero": (lo > 0) or (hi < 0),
            }
            if mname == "acc":
                # exact McNemar on discordant pairs
                n01 = sum(1 for i in subset_ids
                          if per[i]["a"][mi] < per[i]["b"][mi])
                n10 = sum(1 for i in subset_ids
                          if per[i]["a"][mi] > per[i]["b"][mi])
                nd = n01 + n10
                # two-sided exact binomial p at q=0.5
                from math import comb
                if nd == 0:
                    p = 1.0
                else:
                    k = min(n01, n10)
                    tail = sum(comb(nd, j) for j in range(0, k + 1)) / (2 ** nd)
                    p = min(1.0, 2 * tail)
                out[mname]["mcnemar"] = {
                    f"only_{args.arm_b}_correct": n01,
                    f"only_{args.arm_a}_correct": n10,
                    "n_discordant": nd,
                    "p_exact_two_sided": p,
                }
        return out

    overall = summarise(ids, "OVERALL")
    bycat = {}
    cats = defaultdict(list)
    for i in ids:
        cats[per[i]["cat"]].append(i)
    for c in sorted(cats):
        bycat[c] = summarise(cats[c], f"category_{c}")

    # ---- reconcile against each arm's committed scores.json -------------------
    # If my recomputation disagrees with the file the driver wrote, then one of the two
    # is wrong and neither the point estimates nor the test can be trusted. Checked,
    # not assumed.
    recon = {}
    for arm, key in ((args.arm_a, "a"), (args.arm_b, "b")):
        sp = os.path.join(args.root, arm, "scores.json")
        if not os.path.exists(sp):
            recon[arm] = {"error": "scores.json absent"}
            continue
        sj = json.load(open(sp))
        mi = {"f1": 0, "em": 1, "acc": 2}
        mine = {m: 100.0 * sum(per[i][key][mi[m]] for i in ids) / len(ids)
                for m in mi}
        # the driver writes FLAT keys overall_f1 / overall_em / overall_acc (already
        # in percent). Read exactly those; do NOT guess a nested layout -- an empty
        # dict here would make the comparison vacuously true, which is the fail-open
        # pattern this check exists to avoid.
        theirs = {}
        for m in mi:
            k = f"overall_{m}"
            if k not in sj:
                recon[arm] = {"error": f"expected key {k!r} absent from {sp}; "
                                       f"keys={sorted(sj)[:10]}"}
                break
            theirs[m] = float(sj[k])
        else:
            deltas = {m: (mine[m] - theirs[m]) for m in theirs}
            n_expect = sj.get("n_samples")
            recon[arm] = {
                "recomputed_here": mine,
                "from_scores_json": theirs,
                "abs_delta": {m: abs(d) for m, d in deltas.items()},
                "n_metrics_compared": len(theirs),
                "n_samples_in_scores_json": n_expect,
                "n_paired_here": len(ids),
                "n_samples_agree": (n_expect == len(ids)),
                "reconciles_within_0p01pp": (
                    len(theirs) == len(mi)
                    and n_expect == len(ids)
                    and all(abs(d) < 0.01 for d in deltas.values())),
            }
        if not recon[arm].get("reconciles_within_0p01pp"):
            print(f"[stats] WARNING {arm}: recomputation does NOT match scores.json: "
                  f"{recon[arm]}", flush=True)

    out = {
        "arm_a": args.arm_a, "arm_b": args.arm_b,
        "n_paired": len(ids), "n_boot": args.n_boot, "seed": args.seed,
        "note": ("Scores recomputed here with the DRIVER's own metric functions so they "
                 "reconcile with scores.json. Paired because both arms answered the "
                 "identical question set (verified separately by b01_collect.py). "
                 "Bootstrap resamples question-level paired deltas. numpy is NOT used "
                 "(the repo has three numpy versions across nodes, which makes "
                 "seeded numpy sampling non-reproducible across nodes)."),
        "overall": overall,
        "per_category": bycat,
        "reconciliation_vs_scores_json": recon,
    }
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)

    o = overall
    print(f"\nOVERALL n={o['n']}")
    for m in metrics:
        r = o[m]
        print(f"  {m.upper():4} {args.arm_a}={r[args.arm_a+'_mean']:.2f} "
              f"{args.arm_b}={r[args.arm_b+'_mean']:.2f} "
              f"diff={r['paired_diff_pp']:+.2f}pp "
              f"CI95=[{r['boot_ci95_pp'][0]:+.2f},{r['boot_ci95_pp'][1]:+.2f}] "
              f"excl0={r['ci_excludes_zero']}")
        if "mcnemar" in r:
            mc = r["mcnemar"]
            print(f"       McNemar: discordant={mc['n_discordant']} "
                  f"p={mc['p_exact_two_sided']:.3g}")
    print(f"\nwrote {args.json_out}")
    all_recon = all(v.get("reconciles_within_0p01pp") for v in recon.values())
    print(f"reconciles_vs_scores_json = {all_recon}")
    if not all_recon:
        print("FAIL: recomputed metrics do not reconcile with the driver's scores.json "
              "-- the point estimates and the test are NOT trustworthy")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
