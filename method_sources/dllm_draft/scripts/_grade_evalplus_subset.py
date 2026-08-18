#!/usr/bin/env python3
"""Grade an EvalPlus *subset* using the official EvalPlus grader.

``python -m evalplus.evaluate`` hard-asserts full dataset coverage
(``assert len(completion_id) == len(problems), "Missing problems in samples"``),
so it cannot score a ``--limit N`` smoke run. This script exists only to lift
that coverage assertion.

It does NOT implement any test execution of its own. It imports and calls
``evalplus.evaluate.check_correctness`` -- the same function
``evalplus.evaluate`` calls -- which in turn calls the sandboxed official
``evalplus.eval.untrusted_check`` for the base and plus test suites. Ground
truth comes from ``evalplus.gen.util.trusted_exec``-backed
``evalplus.evaluate.get_groundtruth``. pass@1 is computed with the same
"plus counts only if base also passes" rule as ``evalplus.evaluate``.

Full-size runs must still go through ``python -m evalplus.evaluate`` directly;
this is a smoke-only convenience.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
)
from evalplus.data.mbpp import mbpp_serialize_inputs
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.evaluate import check_correctness, get_groundtruth
from evalplus.eval import PASS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--parallel", type=int, default=16)
    args = parser.parse_args()

    if args.dataset == "humaneval":
        problems = get_human_eval_plus()
        dataset_hash = get_human_eval_plus_hash()
        expected_output = get_groundtruth(problems, dataset_hash, [])
    else:
        problems = get_mbpp_plus()
        dataset_hash = get_mbpp_plus_hash()
        expected_output = get_groundtruth(
            problems, dataset_hash, MBPP_OUTPUT_NOT_NONE_TASKS
        )

    samples = [
        json.loads(line)
        for line in Path(args.samples).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    eval_results: dict[str, list] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        completion_id: Counter = Counter()
        for index, sample in enumerate(samples):
            task_id = sample["task_id"]
            if task_id not in problems:
                print(f"WARN task {task_id} not in dataset; skipped")
                continue
            futures.append(
                executor.submit(
                    check_correctness,
                    args.dataset,
                    completion_id[task_id],
                    problems[task_id],
                    sample["solution"],
                    expected_output[task_id],
                    False,  # base_only
                    False,  # fast_check (False == test_details on)
                    f"{task_id}:{index}",
                    # remaining args take evalplus defaults
                )
            )
            completion_id[task_id] += 1
        for future in as_completed(futures):
            result = future.result()
            eval_results[result["task_id"]].append(result)

    results = {"hash": dataset_hash, "eval": {}}
    base_pass = 0
    plus_pass = 0
    for task_id, task_results in eval_results.items():
        task_results.sort(key=lambda row: row["completion_id"])
        rows = []
        for res in task_results:
            base_stat = res["base"][0]
            plus_stat = res["plus"][0]
            rows.append(
                {
                    "task_id": task_id,
                    "solution": res["solution"],
                    "base_status": base_stat,
                    "plus_status": plus_stat,
                }
            )
        results["eval"][task_id] = rows
        base_pass += sum(row["base_status"] == PASS for row in rows)
        plus_pass += sum(
            row["base_status"] == row["plus_status"] == PASS for row in rows
        )

    n = sum(len(rows) for rows in results["eval"].values())
    results["pass_at_k"] = {
        "base": {"pass@1": base_pass / n if n else 0.0},
        "plus": {"pass@1": plus_pass / n if n else 0.0},
    }
    results["n_samples"] = n
    results["subset_grader"] = True
    Path(args.output_file).write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "n_samples": n,
                "base_pass": base_pass,
                "plus_pass": plus_pass,
                "base_pass@1": results["pass_at_k"]["base"]["pass@1"],
                "plus_pass@1": results["pass_at_k"]["plus"]["pass@1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
