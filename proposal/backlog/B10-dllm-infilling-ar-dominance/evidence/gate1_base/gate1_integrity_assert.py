#!/usr/bin/env python3
"""B10 Gate 1 — data-integrity assertions BEFORE any statistics.

Pre-registered requirement (task spec, 2026-08-15):
  6 arms x 1033 rows; item-id sets identical across arms; no duplicate ids;
  no NaN/missing fields. Any assertion failure => STOP, do not compute p-values.

Run with CUDA_VISIBLE_DEVICES="" (CPU only). Exits non-zero on failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = ["dream_fim", "dreamon_fim", "dreamon_oracle",
        "dream_prefix", "qwen_fim", "qwen_prefix"]


def read_jsonl(p: Path):
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def main() -> int:
    root = Path(sys.argv[1])          # .../outputs/infilling_single_line
    data_file = Path(sys.argv[2])     # HumanEval-SingleLineInfilling.jsonl
    expect_n = int(sys.argv[3]) if len(sys.argv) > 3 else 1033

    report = {"expect_n": expect_n, "arms": {}, "failures": []}

    bench = read_jsonl(data_file)
    bench_ids = [r["task_id"] for r in bench]
    report["bench_n"] = len(bench)
    report["bench_unique_ids"] = len(set(bench_ids))
    if len(bench) != expect_n:
        report["failures"].append(f"bench n={len(bench)} != {expect_n}")
    if len(set(bench_ids)) != len(bench_ids):
        report["failures"].append("duplicate task_id in benchmark data file")
    bench_set = set(bench_ids)

    id_sets = {}
    for arm in ARMS:
        p = root / arm / "solutions.jsonl"
        if not p.exists():
            report["failures"].append(f"{arm}: solutions.jsonl missing at {p}")
            continue
        rows = read_jsonl(p)
        ids = [r["task_id"] for r in rows]
        n_dup = len(ids) - len(set(ids))
        # field-level completeness: 'solution' must be a present, non-None str
        bad_sol = [r["task_id"] for r in rows
                   if not isinstance(r.get("solution"), str)]
        none_mid = [r["task_id"] for r in rows if r.get("middle") is None]
        a = {"path": str(p), "n": len(rows), "n_unique_ids": len(set(ids)),
             "n_duplicate_ids": n_dup,
             "n_solution_not_str": len(bad_sol),
             "n_middle_none": len(none_mid),
             "n_empty_solution": sum(1 for r in rows
                                     if isinstance(r.get("solution"), str)
                                     and r["solution"] == "")}
        report["arms"][arm] = a
        id_sets[arm] = set(ids)
        if len(rows) != expect_n:
            report["failures"].append(f"{arm}: n={len(rows)} != {expect_n}")
        if n_dup:
            report["failures"].append(f"{arm}: {n_dup} duplicate task_ids")
        if bad_sol:
            report["failures"].append(
                f"{arm}: {len(bad_sol)} rows with non-str/missing 'solution'")

    if len(id_sets) == len(ARMS):
        ref = id_sets[ARMS[0]]
        for arm in ARMS[1:]:
            if id_sets[arm] != ref:
                report["failures"].append(
                    f"{arm}: id set differs from {ARMS[0]} "
                    f"(sym-diff {len(id_sets[arm] ^ ref)})")
        report["all_arm_id_sets_identical"] = all(
            id_sets[a] == ref for a in ARMS)
        report["arm_ids_equal_bench_ids"] = (ref == bench_set)
        if ref != bench_set:
            report["failures"].append(
                "arm id set != benchmark id set "
                f"(sym-diff {len(ref ^ bench_set)})")

    report["PASS"] = not report["failures"]
    print(json.dumps(report, indent=2))
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
