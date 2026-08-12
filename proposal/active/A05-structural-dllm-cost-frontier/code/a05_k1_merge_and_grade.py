#!/usr/bin/env python3
"""A05 K1 -- merge shards, assert completeness, grade with evalplus, aggregate.

Invariants enforced here (a violation is a hard exit, never a warning):

  * grader is evalplus's own ``untrusted_check``, and every invocation is
    preceded by a SELF-TEST on the same expected-output tables: the canonical
    solution must PASS and an empty stub must FAIL. A hand-rolled verifier that
    discarded return values previously scored empty stubs 7/7 pass in this repo
    (DLLM_RESULTS_20260807.md Retraction 1), so the self-test is not optional.
  * shard completeness: exact item count (HE+ 164 / MBPP+ 378), zero duplicate
    task_id, zero missing task_id, zero nan in any reported statistic.
  * cost is reported as tokens_fed AND NFE, never NFE alone.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from pathlib import Path

EXPECTED = {"humaneval": 164, "mbpp": 378}


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def die(message):
    print(f"FATAL: {message}", file=sys.stderr)
    raise SystemExit(2)


def merge(run_dir: Path, pattern: str, expected: int, label: str):
    shards = sorted(glob.glob(str(run_dir / pattern)))
    if not shards:
        die(f"{label}: no shards matching {pattern} in {run_dir}")
    rows, seen, dups = [], set(), []
    for shard in shards:
        for row in read_jsonl(shard):
            task_id = row["task_id"]
            if task_id in seen:
                dups.append(task_id)
                continue
            seen.add(task_id)
            rows.append(row)
    if dups:
        die(f"{label}: {len(dups)} duplicate task_id, e.g. {dups[:5]}")
    if len(rows) != expected:
        die(f"{label}: got {len(rows)} items, expected exactly {expected} "
            f"(shards found: {len(shards)}) -- refusing to merge a partial run")
    return rows


def grade(dataset: str, solutions: list[dict], workdir: Path) -> dict:
    """Grade with evalplus. Self-tests the grader before scoring anything."""
    os.environ.setdefault("HOME", str(workdir))
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    from evalplus.data import get_human_eval_plus_hash, get_mbpp_plus_hash
    from evalplus.evaluate import get_groundtruth
    from evalplus.eval import untrusted_check

    if dataset == "humaneval":
        problems = get_human_eval_plus()
        dataset_hash = get_human_eval_plus_hash()
        expected_output = get_groundtruth(problems, dataset_hash, [])
    else:
        problems = get_mbpp_plus()
        dataset_hash = get_mbpp_plus_hash()
        from evalplus.data.mbpp import mbpp_serialize_inputs
        expected_output = get_groundtruth(problems, dataset_hash, mbpp_serialize_inputs)

    def check(task_id: str, code: str):
        problem = problems[task_id]
        expected = expected_output[task_id]
        out = {}
        for kind, inputs, ref in (
            ("base", problem["base_input"], expected["base"]),
            ("plus", problem.get("plus_input", []), expected.get("plus", [])),
        ):
            if not inputs:
                out[kind] = "pass"
                continue
            status, _ = untrusted_check(
                dataset,
                code,
                inputs,
                problem["entry_point"],
                ref,
                problem["atol"],
                expected["base_time"] if kind == "base" else expected["plus_time"],
                fast_check=True,
                min_time_limit=1.0,
                gt_time_limit_factor=4.0,
            )
            out[kind] = status
        return out

    # ---- mandatory self-test on the very tables used below ----
    probe_id = "HumanEval/0" if dataset == "humaneval" else "Mbpp/2"
    probe = problems[probe_id]
    canonical = (
        probe["prompt"] + probe["canonical_solution"]
        if dataset == "humaneval"
        else probe["canonical_solution"]
    )
    stub = probe["prompt"] + "    pass\n" if dataset == "humaneval" else "def f():\n    pass\n"
    good = check(probe_id, canonical)
    bad = check(probe_id, stub)
    selftest = {
        "probe_task": probe_id,
        "canonical_base": good["base"], "canonical_plus": good["plus"],
        "stub_base": bad["base"], "stub_plus": bad["plus"],
    }
    if good["base"] != "pass" or good["plus"] != "pass":
        die(f"grader self-test FAILED: canonical solution did not pass -> {selftest}")
    if bad["base"] == "pass":
        die(f"grader self-test FAILED: empty stub scored as pass -> {selftest}")
    print(f"  grader self-test OK: {selftest}", flush=True)

    base_pass, plus_pass, per_item = 0, 0, {}
    for row in solutions:
        task_id = row["task_id"]
        if task_id not in problems:
            die(f"unknown task_id from grader's view: {task_id}")
        result = check(task_id, row["solution"])
        ok_base = result["base"] == "pass"
        ok_plus = ok_base and result["plus"] == "pass"
        base_pass += ok_base
        plus_pass += ok_plus
        per_item[task_id] = {"base": ok_base, "plus": ok_plus}
    n = len(solutions)
    return {
        "n": n,
        "pass_at_1_base": base_pass / n,
        "pass_at_1_plus": plus_pass / n,
        "n_pass_base": base_pass,
        "n_pass_plus": plus_pass,
        "grader_self_test": selftest,
        "per_item": per_item,
    }


def aggregate(metrics_rows: list[dict]) -> dict:
    procs = [r["process"] for r in metrics_rows if r.get("process")]
    if len(procs) != len(metrics_rows):
        die(f"{len(metrics_rows) - len(procs)} items have no process block")
    nfe = [p["nfe"] for p in procs]
    tok_eff = [p["tokens_fed_effective"] for p in procs]
    tok_pad = [p["tokens_fed_padded"] for p in procs]
    gen = [p["generated_tokens"] for p in procs]
    ratios = [p["emitted_gold_ratio"] for p in procs if p.get("emitted_gold_ratio") is not None]
    par = [bool(p["final_parseable"]) for p in procs]
    canvases = sorted({p["initial_masks"] for p in procs})
    clamped = sum(1 for p in procs if p.get("initial_masks_clamped"))
    errors = sum(1 for r in metrics_rows if r.get("error"))
    empty_raw = sum(1 for r in metrics_rows if not r.get("raw_output"))

    long_span = [p for p in procs if p.get("gold_tokens", 0) >= 65]
    ls_ratio = [p["emitted_gold_ratio"] for p in long_span if p.get("emitted_gold_ratio") is not None]
    ls_par = [bool(p["final_parseable"]) for p in long_span]

    out = {
        "n_items": len(procs),
        "nfe_mean": statistics.mean(nfe),
        "nfe_median": statistics.median(nfe),
        "nfe_total": sum(nfe),
        "tokens_fed_effective_mean": statistics.mean(tok_eff),
        "tokens_fed_effective_total": sum(tok_eff),
        "tokens_fed_padded_mean": statistics.mean(tok_pad),
        "generated_tokens_mean": statistics.mean(gen),
        "generated_tokens_median": statistics.median(gen),
        "generated_tokens_max": max(gen),
        "emitted_gold_ratio_mean": statistics.mean(ratios) if ratios else None,
        "emitted_gold_ratio_median": statistics.median(ratios) if ratios else None,
        "parseability": sum(par) / len(par),
        "n_parseable": sum(par),
        "empty_raw_output": empty_raw,
        "generation_errors": errors,
        "initial_masks_observed": canvases,
        "initial_masks_clamped_items": clamped,
        "wall_seconds_sum": sum(r["elapsed_seconds"] for r in metrics_rows),
        "peak_memory_gib_max": max(r["peak_memory_gib"] for r in metrics_rows),
        "long_span_gold_ge65": {
            "n": len(long_span),
            "emitted_gold_ratio_median": statistics.median(ls_ratio) if ls_ratio else None,
            "parseability": (sum(ls_par) / len(ls_par)) if ls_par else None,
        },
    }
    for key, value in out.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            die(f"nan/inf in aggregate field {key}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--workdir", default="/tmp/a05_k1_grade")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    expected = EXPECTED[args.dataset]
    sols = merge(run_dir, "solutions.rank*.jsonl", expected, f"{args.label}/solutions")
    mets = merge(run_dir, "metrics.rank*.jsonl", expected, f"{args.label}/metrics")
    if {r["task_id"] for r in sols} != {r["task_id"] for r in mets}:
        die(f"{args.label}: solutions and metrics disagree on task_id set")

    merged = run_dir / "solutions.jsonl"
    with merged.open("w", encoding="utf-8") as handle:
        for row in sols:
            handle.write(json.dumps({"task_id": row["task_id"], "solution": row["solution"]}) + "\n")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"[{args.label}] merged {len(sols)} items -> grading with evalplus", flush=True)
    scores = grade(args.dataset, sols, workdir)
    stats = aggregate(mets)

    if scores["n"] != expected or stats["n_items"] != expected:
        die(f"{args.label}: post-merge count drift ({scores['n']}, {stats['n_items']}) != {expected}")

    payload = {
        "label": args.label,
        "dataset": args.dataset,
        "run_dir": str(run_dir.resolve()),
        "expected_items": expected,
        "scores": {k: v for k, v in scores.items() if k != "per_item"},
        "cost_and_behaviour": stats,
        "per_item_pass": scores["per_item"],
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({
        "label": args.label,
        "n": scores["n"],
        "pass@1_base": round(scores["pass_at_1_base"], 4),
        "pass@1_plus": round(scores["pass_at_1_plus"], 4),
        "nfe_mean": round(stats["nfe_mean"], 1),
        "tokens_fed_eff_mean": round(stats["tokens_fed_effective_mean"], 1),
        "gen_tok_mean": round(stats["generated_tokens_mean"], 2),
        "ratio_median": stats["emitted_gold_ratio_median"],
        "parseability": round(stats["parseability"], 4),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
