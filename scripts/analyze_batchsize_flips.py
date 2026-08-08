#!/usr/bin/env python3
"""Per-item flip analysis between two batch_size arms.
Usage:
    python scripts/analyze_batchsize_flips.py \
        --dir_a olmo2_downstream_results/7B_shortgpt16_step200000_bs8 \
        --dir_b olmo2_downstream_results/7B_shortgpt16_step200000_bs16 \
        --label_a bs8 --label_b bs16

Produces:
  - per-task flip table
  - total flip count
  - examples of near-tie items (abs(score_diff) < 0.1 nats)
"""
import argparse
import json
import os
import glob

def load_per_example(results_dir, task):
    path = os.path.join(results_dir, f"per_example_{task}.jsonl")
    if not os.path.exists(path):
        return None
    items = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            items[r["item_id"]] = r
    return items

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir_a", required=True, help="results dir for arm A (e.g. bs8)")
    p.add_argument("--dir_b", required=True, help="results dir for arm B (e.g. bs16)")
    p.add_argument("--label_a", default="arm_a")
    p.add_argument("--label_b", default="arm_b")
    p.add_argument("--show_near_tie_thresh", type=float, default=0.10,
                   help="show flips where max-score gap < thresh nats")
    args = p.parse_args()

    # Load summaries
    def load_summary(d):
        with open(os.path.join(d, "summary.json")) as f:
            return json.load(f)
    sa = load_summary(args.dir_a)
    sb = load_summary(args.dir_b)

    tasks = sorted(set(sa["tasks"].keys()) | set(sb["tasks"].keys()))
    print(f"\n{'Task':<20} {'n':>6} {'n_correct_'+args.label_a:>14} {'n_correct_'+args.label_b:>14} {'diff':>6} {'flips':>6} {'near-tie_flips':>14}")
    print("-" * 90)

    total_flips = 0
    total_near_tie_flips = 0
    all_flip_examples = []

    for task in tasks:
        ta = sa["tasks"].get(task, {})
        tb = sb["tasks"].get(task, {})
        if ta.get("skipped") or tb.get("skipped"):
            print(f"  {task:<20} SKIPPED in one arm")
            continue

        n_a = ta.get("n_correct_acc", 0)
        n_b = tb.get("n_correct_acc", 0)
        n = ta.get("n", tb.get("n", 0))
        diff = n_b - n_a

        # Per-item flip analysis
        items_a = load_per_example(args.dir_a, task)
        items_b = load_per_example(args.dir_b, task)

        if items_a is None or items_b is None:
            print(f"  {task:<20} {n:>6} {n_a:>14} {n_b:>14} {diff:>+6} (no per_example files)")
            continue

        common_ids = sorted(set(items_a.keys()) & set(items_b.keys()))
        flips = [(iid, items_a[iid], items_b[iid]) for iid in common_ids
                 if items_a[iid].get("correct") != items_b[iid].get("correct")]
        task_flips = len(flips)
        total_flips += task_flips

        # Near-tie: max option score gap < threshold
        near_tie_flips = []
        for iid, ra, rb in flips:
            scores_a = list(ra.get("option_scores", {}).values())
            scores_b = list(rb.get("option_scores", {}).values())
            if not scores_a or not scores_b:
                continue
            gap_a = max(scores_a) - sorted(scores_a)[-2] if len(scores_a) >= 2 else float('inf')
            gap_b = max(scores_b) - sorted(scores_b)[-2] if len(scores_b) >= 2 else float('inf')
            min_gap = min(abs(gap_a), abs(gap_b))
            if min_gap < args.show_near_tie_thresh:
                near_tie_flips.append((iid, ra, rb, gap_a, gap_b))
        total_near_tie_flips += len(near_tie_flips)
        all_flip_examples.extend([(task, iid, ra, rb, ga, gb) for (iid, ra, rb, ga, gb) in near_tie_flips[:3]])

        print(f"  {task:<20} {n:>6} {n_a:>14} {n_b:>14} {diff:>+6} {task_flips:>6} {len(near_tie_flips):>14}")

    print("-" * 90)
    print(f"  {'TOTAL':<20} {'':>6} {'':>14} {'':>14} {'':>6} {total_flips:>6} {total_near_tie_flips:>14}")
    print(f"\nTotal flips ({args.label_a} vs {args.label_b}): {total_flips}")
    print(f"Near-tie flips (gap < {args.show_near_tie_thresh} nats): {total_near_tie_flips}")

    if all_flip_examples:
        print(f"\n=== Near-tie flip examples (up to 3 per task, first 6 total) ===")
        shown = 0
        for task, iid, ra, rb, gap_a, gap_b in all_flip_examples[:6]:
            print(f"\n  [{task}] item_id={iid}")
            gold = ra.get("gold_letter", "?")
            print(f"    gold={gold}")
            print(f"    {args.label_a}: pred={ra.get('pred_letter','?')} correct={ra.get('correct')} "
                  f"options={ra.get('option_scores',{})} margin={gap_a:.4f}")
            print(f"    {args.label_b}: pred={rb.get('pred_letter','?')} correct={rb.get('correct')} "
                  f"options={rb.get('option_scores',{})} margin={gap_b:.4f}")
            shown += 1
    else:
        print("\nNo near-tie flips found below threshold.")

    print("\n=== Summary stats ===")
    print(f"  {args.label_a} summary total correct: {sum(sa['tasks'][t]['n_correct_acc'] for t in tasks if not sa['tasks'].get(t,{}).get('skipped'))}")
    print(f"  {args.label_b} summary total correct: {sum(sb['tasks'][t]['n_correct_acc'] for t in tasks if not sb['tasks'].get(t,{}).get('skipped'))}")
    print(f"  n_shards_a={sa.get('n_shards')}, n_shards_b={sb.get('n_shards')}")

if __name__ == "__main__":
    main()
