#!/usr/bin/env python3
"""Analyze non-oracle DreamOn kspan arm vs existing arms."""

import json
from pathlib import Path

def load_rows(rd):
    return json.load(Path(rd, "score.json").open())["rows"]

def task_of(sid):
    return sid.rsplit("/", 1)[0]

def cellmeans(rows, subset=None, key="passed"):
    out = {}
    for k in (1, 2, 3, 4):
        v = [r for r in rows if r["k"] == k and (subset is None or task_of(r["spec_id"]) in subset)]
        out[k] = (sum(r[key] for r in v) / len(v), len(v)) if v else (0.0, 0)
    return out


def main():
    arms = {
        "nonoracle": load_rows("runs/kspan_diffusion_nonoracle"),
        "diffusion_oracle": load_rows("runs/kspan_diffusion"),
        "ar_fim": load_rows("runs/kspan_ar_fim"),
        "ar_fim_fair": load_rows("runs/kspan_ar_fim_fair"),
    }

    print("=" * 80)
    print("FULL-N LADDER (all rows admitted at each k) — pass@1 (n)")
    print("=" * 80)
    hdr = "arm".ljust(20) + "".join(f"{'k='+str(k):>18}" for k in (1, 2, 3, 4))
    print(hdr)
    for arm, rows in arms.items():
        cm = cellmeans(rows)
        line = arm.ljust(20)
        for k in (1, 2, 3, 4):
            p1, n = cm[k]
            line += f"   {p1:.3f} (n={n:>3})"
        print(line)

    print()
    print("FULL-N EM_stripped:")
    print(hdr)
    for arm, rows in arms.items():
        cm = cellmeans(rows, key="em_all_stripped")
        line = arm.ljust(20)
        for k in (1, 2, 3, 4):
            em, n = cm[k]
            line += f"   {em:.3f} (n={n:>3})"
        print(line)

    # Common (n=59) set — from the nonoracle arm
    non_rows = arms["nonoracle"]
    common = None
    for k in (1, 2, 3, 4):
        ids = {task_of(r["spec_id"]) for r in non_rows if r["k"] == k}
        common = ids if common is None else (common & ids)
    common = frozenset(common)

    print()
    print("=" * 80)
    print(f"BALANCED LADDER (n={len(common)} tasks present at every k)")
    print("=" * 80)
    print(hdr)
    for arm, rows in arms.items():
        cm = cellmeans(rows, subset=common)
        line = arm.ljust(20)
        for k in (1, 2, 3, 4):
            p1, n = cm[k]
            line += f"   {p1:.3f} (n={n:>3})"
        print(line)

    print()
    print("BALANCED EM_stripped:")
    print(hdr)
    for arm, rows in arms.items():
        cm = cellmeans(rows, subset=common, key="em_all_stripped")
        line = arm.ljust(20)
        for k in (1, 2, 3, 4):
            em, n = cm[k]
            line += f"   {em:.3f} (n={n:>3})"
        print(line)

    # Naive 4-cell interaction on balanced set
    print()
    print("=" * 80)
    print("NAIVE 4-CELL INTERACTION on balanced set = (diff_k4 - diff_k1) - (AR_k4 - AR_k1)")
    print("=" * 80)
    non_bal = {k: v[0] for k, v in cellmeans(arms["nonoracle"], common).items()}
    oracle_bal = {k: v[0] for k, v in cellmeans(arms["diffusion_oracle"], common).items()}
    ar_bal = {k: v[0] for k, v in cellmeans(arms["ar_fim"], common).items()}
    fair_bal = {k: v[0] for k, v in cellmeans(arms["ar_fim_fair"], common).items()}
    for pair_name, ar in (("ar_fim", ar_bal), ("ar_fim_fair", fair_bal)):
        print(f"vs {pair_name}:")
        for k in (1, 2, 3, 4):
            print(f"  k={k}: nonoracle {non_bal[k]:.3f}  oracle {oracle_bal[k]:.3f}  {pair_name} {ar[k]:.3f}  "
                  f"delta_nonoracle={non_bal[k]-ar[k]:+.3f}  delta_oracle={oracle_bal[k]-ar[k]:+.3f}")
        int_non = (non_bal[4] - non_bal[1]) - (ar[4] - ar[1])
        int_or = (oracle_bal[4] - oracle_bal[1]) - (ar[4] - ar[1])
        print(f"  naive 4-cell interaction: nonoracle={int_non:+.3f}  oracle={int_or:+.3f}")

    # Per-arm within-task-common slope on balanced set (raw pass@1 drops)
    print()
    print("=" * 80)
    print("BALANCED WITHIN-TASK SLOPE (k=1 → k=4 drop)")
    print("=" * 80)
    for arm, rows in arms.items():
        cm = cellmeans(rows, subset=common)
        drop = cm[1][0] - cm[4][0]
        print(f"  {arm:<20}  k=1 → k=4 drop: {cm[1][0]:.3f} → {cm[4][0]:.3f}  = {drop:+.3f}")

    # Truncation / abort disclosure
    print()
    print("=" * 80)
    print("TRUNCATION / ABORT / ERROR — non-oracle arm only (crash-out disclosure)")
    print("=" * 80)
    for k in (1, 2, 3, 4):
        v = [r for r in arms["nonoracle"] if r["k"] == k]
        ntr = sum(1 for r in v if r["truncated_holes"] > 0)
        nab = sum(1 for r in v if r["aborted_holes"] > 0)
        nerr = sum(1 for r in v if r.get("error"))
        npar = sum(1 for r in v if r["parseable"])
        print(f"  k={k}: n={len(v)}  parseable={npar} ({100*npar/len(v):.1f}%)  "
              f"truncated_tasks={ntr}  aborted_tasks={nab}  errors={nerr}")

    print()
    print("=" * 80)
    print("COST comparison (mean per task in tokens_fed / forward_passes)")
    print("=" * 80)
    for arm, rows in arms.items():
        line = arm.ljust(20)
        for k in (1, 2, 3, 4):
            v = [r for r in rows if r["k"] == k]
            tok = sum(r["tokens_fed"] for r in v) / max(1, len(v))
            fwd = sum(r["forward_passes"] for r in v) / max(1, len(v))
            line += f"  k={k}: tok={tok:>7.0f} fwd={fwd:>5.1f}"
        print(line)


if __name__ == "__main__":
    main()
