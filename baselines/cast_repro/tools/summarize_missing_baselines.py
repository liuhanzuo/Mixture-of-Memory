#!/usr/bin/env python3
"""Summarize the re-scored SparseForge baseline arms in PLAIN ACC only.

WHY PLAIN ACC ONLY
------------------
Both source papers' headline aggregates (CAST 55.91, AST 58.62/57.94) are means of
``acc``, not ``acc_norm``. A mean that mixes the two -- e.g. lm-eval's own default
"primary metric" convention, acc_norm for hellaswag/arc/obqa and acc for the rest --
is not comparable to any published number. This repository has produced wrong
comparisons that way before, so this tool reports ``acc,none`` and nothing else.

ASSERTIONS (a failure here means the score is not trustworthy, not that the model is bad)
  * per-task ``n_samples`` must equal the known dataset size, so a silently truncated
    or partially-merged run cannot be averaged into a table cell;
  * RTE has n=277, so ``acc * 277`` must be an integer -- any non-integer means the
    accuracy did not come from 277 scored examples.
"""

from __future__ import annotations

import argparse
import json
import os

UNION9 = ("boolq", "rte", "hellaswag", "race", "piqa",
          "winogrande", "arc_easy", "arc_challenge", "openbookqa")
AST7 = ("boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa")
CAST7 = ("hellaswag", "race", "piqa", "winogrande", "arc_easy", "arc_challenge", "openbookqa")

EXPECT_N = {"boolq": 3270, "rte": 277, "hellaswag": 10042, "race": 1045, "piqa": 1838,
            "winogrande": 1267, "arc_easy": 2376, "arc_challenge": 1172, "openbookqa": 500}


def mean(acc, subset):
    return sum(acc[t] for t in subset) / len(subset)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union9-root", required=True)
    ap.add_argument("--gate-root", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = args.union9_root
    arms = sorted(d for d in os.listdir(root)
                  if os.path.isfile(f"{root}/{d}/zeroshot_union9.json"))

    rows = {}
    problems = []
    for arm in arms:
        blob = json.load(open(f"{root}/{arm}/zeroshot_union9.json"))
        pt = blob["per_task"]
        missing = [t for t in UNION9 if t not in pt]
        if missing:
            problems.append(f"{arm}: missing tasks {missing}")
            continue

        bad_n = [(t, pt[t]["n_samples"]) for t in UNION9
                 if pt[t]["n_samples"] != EXPECT_N[t]]
        if bad_n:
            problems.append(f"{arm}: wrong n_samples {bad_n}")
            continue

        acc = {t: pt[t]["acc"] for t in UNION9}
        if any(v is None for v in acc.values()):
            problems.append(f"{arm}: null acc for {[t for t,v in acc.items() if v is None]}")
            continue

        rte_k = acc["rte"] * EXPECT_N["rte"]
        if abs(rte_k - round(rte_k)) > 1e-6:
            problems.append(f"{arm}: RTE acc*277 = {rte_k} is not integral")
            continue

        entry = {
            "plain_acc_per_task": {t: acc[t] * 100 for t in UNION9},
            "rte_k_over_277": round(rte_k),
            "ast7_mean_plain_acc": mean(acc, AST7) * 100,
            "cast7_mean_plain_acc": mean(acc, CAST7) * 100,
            "union9_mean_plain_acc": mean(acc, UNION9) * 100,
            "n_samples_per_task": {t: pt[t]["n_samples"] for t in UNION9},
            "source_results_json": blob.get("source_results_json"),
            "source_model_args": blob.get("source_model_args"),
        }

        if args.gate_root:
            gp = f"{args.gate_root}/tiledir_{arm}.json"
            if os.path.isfile(gp):
                g = json.load(open(gp))
                entry["gate"] = {
                    "zero_frac": g["zero_frac"],
                    "exact_2of4_tile_ratio": g["exact_2of4_tile_ratio"],
                    "tiles_gt2_budget_violations": g["tiles_gt2_TOTAL_budget_violations"],
                    "tiles_lt2_sparser_than_2of4": g["tiles_lt2_TOTAL_sparser_than_2of4"],
                    "strict_gate_would_pass": g["strict_gate_would_pass"],
                    "deployable_2of4": g["deployable_2of4"],
                }
        rows[arm] = entry

    # Seed groups -> mean over seeds, mirroring the paper's "(3 seeds)" rows.
    groups = {}
    for arm in rows:
        if arm.endswith(("_seed0", "_seed1", "_seed2")):
            groups.setdefault(arm.rsplit("_seed", 1)[0], []).append(arm)
    seed_means = {}
    for base, members in sorted(groups.items()):
        members.sort()
        n = len(members)
        agg = {"n_seeds": n, "seeds": members}
        for key in ("ast7_mean_plain_acc", "cast7_mean_plain_acc", "union9_mean_plain_acc"):
            vals = [rows[m][key] for m in members]
            agg[key] = sum(vals) / n
            if n > 1:
                mu = agg[key]
                agg[key + "_sd"] = (sum((v - mu) ** 2 for v in vals) / (n - 1)) ** 0.5
        agg["plain_acc_per_task"] = {
            t: sum(rows[m]["plain_acc_per_task"][t] for m in members) / n for t in UNION9
        }
        seed_means[base] = agg

    out = {
        "metric": "acc,none (plain accuracy) ONLY -- acc_norm deliberately excluded",
        "harness": ("node .21, lm-eval 0.4.8, --model hf, dtype=bfloat16, "
                    "parallelize=True, add_bos_token=False, --batch_size auto, "
                    "--num_fewshot 0, --seed 0, no chat template"),
        "task_sets": {"union9": list(UNION9), "ast7": list(AST7), "cast7": list(CAST7)},
        "per_arm": rows,
        "seed_group_means": seed_means,
        "assertion_failures": problems,
    }
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)

    hdr = f"{'arm':30s} " + " ".join(f"{t[:9]:>9s}" for t in UNION9) + "     AST7    CAST7   UNION9"
    print(hdr)
    print("-" * len(hdr))
    for arm, e in rows.items():
        cells = " ".join(f"{e['plain_acc_per_task'][t]:9.4f}" for t in UNION9)
        print(f"{arm:30s} {cells}  {e['ast7_mean_plain_acc']:8.4f} "
              f"{e['cast7_mean_plain_acc']:8.4f} {e['union9_mean_plain_acc']:8.4f}"
              f"   (RTE k={e['rte_k_over_277']}/277)")
    if seed_means:
        print("\n--- seed-group means (plain acc) ---")
        for base, a in seed_means.items():
            sd = a.get("union9_mean_plain_acc_sd")
            sd_s = f" +/-{sd:.4f}" if sd is not None else ""
            print(f"{base:30s} n={a['n_seeds']}  AST7={a['ast7_mean_plain_acc']:8.4f}"
                  f"  CAST7={a['cast7_mean_plain_acc']:8.4f}"
                  f"  UNION9={a['union9_mean_plain_acc']:8.4f}{sd_s}")
    if problems:
        print("\n!!! ASSERTION FAILURES (these arms were NOT summarized):")
        for p in problems:
            print("   ", p)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
