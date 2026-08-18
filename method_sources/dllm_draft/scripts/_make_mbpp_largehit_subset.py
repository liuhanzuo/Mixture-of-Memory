#!/usr/bin/env python
"""Regenerate the MBPP+ "budget-truncated tasks" subset used by the 8x-budget probe.

`data/` is gitignored, so `data/evalplus/mbpp_plus_largehit35.jsonl` is not in the
repo. This script rederives it deterministically from an existing run's metrics,
so `scripts/_run_scaffold_large_mbpp_budget4096_b200.sh` stays reproducible.

The subset = every task whose termination_reason is `model_call_budget` in the
reference run. Note the reason must be read from `process` OR `failure_process`:
the runtime writes normally-terminating tasks to `process` and budget-truncated
ones to `failure_process`, so looking only at `process` silently drops exactly
the tasks we care about (and they are the most expensive ones).

Usage:
    python scripts/_make_mbpp_largehit_subset.py \
        [runs/scaffold_large_mbppplus] \
        [data/evalplus/mbpp_plus.jsonl] \
        [data/evalplus/mbpp_plus_largehit35.jsonl]
"""
import glob
import json
import os
import sys


def truncated_task_ids(run_dir):
    ids = set()
    for path in sorted(glob.glob(os.path.join(run_dir, "metrics.rank*.jsonl"))):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                proc = row.get("process") or row.get("failure_process") or {}
                if proc.get("termination_reason") == "model_call_budget":
                    ids.add(row["task_id"])
    return ids


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/scaffold_large_mbppplus"
    data_file = sys.argv[2] if len(sys.argv) > 2 else "data/evalplus/mbpp_plus.jsonl"
    out_file = (
        sys.argv[3]
        if len(sys.argv) > 3
        else "data/evalplus/mbpp_plus_largehit35.jsonl"
    )

    ids = truncated_task_ids(run_dir)
    if not ids:
        sys.exit(f"no model_call_budget terminations found in {run_dir}")

    with open(data_file) as handle:
        tasks = [json.loads(l) for l in handle if l.strip()]
    subset = [t for t in tasks if t["task_id"] in ids]

    missing = ids - {t["task_id"] for t in subset}
    if missing:
        sys.exit(f"{len(missing)} truncated ids absent from {data_file}: {sorted(missing)}")

    # Preserve the original file order so shard assignment (index % world_size)
    # stays reproducible across reruns.
    with open(out_file, "w") as handle:
        for task in subset:
            handle.write(json.dumps(task) + "\n")
    print(f"wrote {len(subset)} truncated tasks from {run_dir} -> {out_file}")


if __name__ == "__main__":
    main()
