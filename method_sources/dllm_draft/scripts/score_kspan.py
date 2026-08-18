#!/usr/bin/env python3
"""Grade k-span infilling runs with the OFFICIAL EvalPlus grader.

GRADER PROVENANCE
=================
Test execution, comparison and timeouts are delegated ENTIRELY to
``evalplus.eval.untrusted_check`` against ``evalplus.evaluate.get_groundtruth``
expected outputs. Nothing is reimplemented. A hand-rolled runner previously used
in this repo discarded return values and gave an EMPTY program a full pass; that
caused a retraction.

``--self-test`` re-verifies on EVERY invocation that
  (a) the gold-refilled program PASSES, and
  (b) a ``pass``-stubbed program FAILS,
so an inflated grader cannot go unnoticed.

Reports, per (arm, k) cell:
  n, pass@1, EM-to-gold (exact and whitespace-stripped), parseable rate,
  and -- SEPARATELY from failures -- truncation and abort rates, plus the
  cost axes (tokens_fed, attended_context_sum, forward_passes) computed over
  ALL tasks, never conditioned on successful termination.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as h:
        for line in h:
            if line.strip():
                yield json.loads(line)


def load_evalplus():
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
    from evalplus.evaluate import get_groundtruth
    he = get_human_eval_plus()
    gt = get_groundtruth(he, get_human_eval_plus_hash(), [])
    return he, gt


def grade_one(task, ref, program, *, which, min_time_limit, factor):
    from evalplus.eval import PASS, untrusted_check
    if which == "base":
        inputs, expected, ref_time = task["base_input"], ref["base"], ref["base_time"]
    else:
        inputs = list(task["base_input"]) + list(task["plus_input"])
        expected = list(ref["base"]) + list(ref["plus"])
        ref_time = list(ref["base_time"]) + list(ref["plus_time"])
    try:
        status, details = untrusted_check(
            "humaneval", program, inputs, task["entry_point"],
            expected=expected, atol=task["atol"], ref_time=ref_time,
            fast_check=False, min_time_limit=min_time_limit,
            gt_time_limit_factor=factor,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"GRADER_ERR: {type(exc).__name__}: {exc}"
    details = list(details) if details is not None else []
    ok = status == PASS and len(details) > 0 and all(bool(d) for d in details)
    return ok, (None if ok else "fail")


def splice(segments, fills):
    out, j = [], 0
    for kind, text in segments:
        if kind == "text":
            out.append(text)
        else:
            out.append(fills[j]); j += 1
    return "".join(out)


def indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--which", default="base", choices=("base", "plus"))
    ap.add_argument("--min-time-limit", type=float, default=1.0)
    ap.add_argument("--factor", type=float, default=4.0)
    ap.add_argument("--self-test", action="store_true", default=True)
    ap.add_argument("--expect-n", default="", help="k=n,k=n asserted cell sizes")
    ap.add_argument("--expect-ranks", type=int, default=0,
                    help="assert exactly this many solutions.rank*.jsonl shards "
                         "are present (0 = no assert). A silent 5/8 merge has "
                         "corrupted a口径 before.")
    ap.add_argument("--expect-rows", type=int, default=0,
                    help="assert total merged solution row count (0 = no assert)")
    ap.add_argument("--out", default="score.json")
    args = ap.parse_args()

    spec = {r["spec_id"]: r for r in read_jsonl(Path(args.spec))}
    he, gt = load_evalplus()

    # ---------------- grader self-test on the frozen spec itself ----------------
    if args.self_test:
        import random
        rng = random.Random(0)
        sample = rng.sample(sorted(spec), min(12, len(spec)))
        bad = []
        for sid in sample:
            r = spec[sid]
            t, ref = he[r["task_id"]], gt[r["task_id"]]
            ok_gold, _ = grade_one(t, ref, splice(r["segments"], r["gold_lines"]),
                                   which=args.which,
                                   min_time_limit=args.min_time_limit, factor=args.factor)
            stub = [indent_of(g) + "pass\n" for g in r["gold_lines"]]
            ok_stub, _ = grade_one(t, ref, splice(r["segments"], stub),
                                   which=args.which,
                                   min_time_limit=args.min_time_limit, factor=args.factor)
            if not ok_gold or ok_stub:
                bad.append((sid, ok_gold, ok_stub))
        if bad:
            print("GRADER SELF-TEST FAILED (gold must pass, stub must fail):")
            for b in bad:
                print("   ", b)
            return 4
        print(f"grader self-test OK on {len(sample)} spec rows "
              f"(gold passes, pass-stub fails)")

    run = Path(args.run_dir)
    sol_shards = sorted(run.glob("solutions.rank*.jsonl"))
    met_shards = sorted(run.glob("metrics.rank*.jsonl"))
    sols = []
    for p in sol_shards:
        sols.extend(read_jsonl(p))
    mets = {}
    for p in met_shards:
        for m in read_jsonl(p):
            mets[m["spec_id"]] = m
    if not sols:
        print(f"FATAL: no solutions in {run}")
        return 2

    # shard completeness: silent partial merges have destroyed a口径 before
    seen = {s["spec_id"] for s in sols}
    dup = len(sols) - len(seen)
    print(f"loaded {len(sols)} solutions ({len(seen)} unique, {dup} duplicate), "
          f"{len(mets)} metric rows "
          f"from {len(sol_shards)} solution shards / {len(met_shards)} metric shards")
    if args.expect_ranks:
        if len(sol_shards) != args.expect_ranks or len(met_shards) != args.expect_ranks:
            print(f"SHARD ASSERT FAIL: expected {args.expect_ranks} shards, got "
                  f"{len(sol_shards)} solutions / {len(met_shards)} metrics")
            return 6
        print(f"shard-count assert OK: {args.expect_ranks}/{args.expect_ranks} ranks present")
    if args.expect_rows:
        if len(sols) != args.expect_rows or len(seen) != args.expect_rows:
            print(f"ROW ASSERT FAIL: expected {args.expect_rows} unique rows, got "
                  f"{len(sols)} ({len(seen)} unique)")
            return 6
        print(f"row-count assert OK: {args.expect_rows} unique solutions, 0 duplicates")
    missing_met = sorted(seen - set(mets))
    if missing_met:
        print(f"METRIC ASSERT FAIL: {len(missing_met)} graded rows have no metric row, "
              f"e.g. {missing_met[:5]}")
        return 6

    rows = []
    for s in sols:
        sid = s["spec_id"]
        r = spec[sid]
        ok, why = grade_one(he[r["task_id"]], gt[r["task_id"]], s["solution"],
                            which=args.which, min_time_limit=args.min_time_limit,
                            factor=args.factor)
        m = mets.get(sid, {})
        c = m.get("cost", {}) or {}
        rows.append({
            "spec_id": sid, "k": r["k"], "passed": bool(ok), "why": why,
            "em_all": bool(m.get("em_all", False)),
            "em_all_stripped": bool(m.get("em_all_stripped", False)),
            "parseable": bool(m.get("parseable", False)),
            "truncated_holes": int(m.get("truncated_holes", 0) or 0),
            "aborted_holes": int(m.get("aborted_holes", 0) or 0),
            "error": m.get("error"),
            "tokens_fed": int(c.get("tokens_fed", 0) or 0),
            "attended": int(c.get("attended_context_sum", 0) or 0),
            "forward_passes": int(c.get("forward_passes", 0) or 0),
        })

    by_k = defaultdict(list)
    for r in rows:
        by_k[r["k"]].append(r)

    summary = {}
    print()
    hdr = (f"{'k':>3} {'n':>5} {'pass@1':>7} {'EM':>7} {'EMstr':>7} {'parse':>7} "
           f"{'trunc%':>7} {'abort%':>7} {'err':>4} {'tok_fed':>9} {'attended':>10} {'fwd':>7}")
    print(hdr); print("-" * len(hdr))
    for k in sorted(by_k):
        v = by_k[k]
        n = len(v)
        def mean(f): return sum(f(x) for x in v) / n
        s = {
            "n": n,
            "pass@1": mean(lambda x: x["passed"]),
            "em_to_gold": mean(lambda x: x["em_all"]),
            "em_to_gold_stripped": mean(lambda x: x["em_all_stripped"]),
            "parseable": mean(lambda x: x["parseable"]),
            "n_pass": sum(x["passed"] for x in v),
            "n_parseable": sum(x["parseable"] for x in v),
            "tasks_with_truncation": sum(1 for x in v if x["truncated_holes"] > 0),
            "tasks_with_abort": sum(1 for x in v if x["aborted_holes"] > 0),
            "n_errors": sum(1 for x in v if x["error"]),
            "tokens_fed_mean": mean(lambda x: x["tokens_fed"]),
            "attended_mean": mean(lambda x: x["attended"]),
            "forward_passes_mean": mean(lambda x: x["forward_passes"]),
            "tokens_fed_median": statistics.median(x["tokens_fed"] for x in v),
            # pass@1 restricted to the NON-memorised subset (EM-to-gold false)
            "n_em_false": sum(1 for x in v if not x["em_all_stripped"]),
            "pass_at_1_given_em_false": (
                sum(x["passed"] for x in v if not x["em_all_stripped"])
                / max(1, sum(1 for x in v if not x["em_all_stripped"]))),
            "n_em_true": sum(1 for x in v if x["em_all_stripped"]),
            "pass_at_1_given_em_true": (
                sum(x["passed"] for x in v if x["em_all_stripped"])
                / max(1, sum(1 for x in v if x["em_all_stripped"]))),
        }
        summary[str(k)] = s
        print(f"{k:>3} {n:>5} {s['pass@1']:>7.3f} {s['em_to_gold']:>7.3f} "
              f"{s['em_to_gold_stripped']:>7.3f} {s['parseable']:>7.3f} "
              f"{100*s['tasks_with_truncation']/n:>7.1f} "
              f"{100*s['tasks_with_abort']/n:>7.1f} {s['n_errors']:>4} "
              f"{s['tokens_fed_mean']:>9.0f} {s['attended_mean']:>10.0f} "
              f"{s['forward_passes_mean']:>7.1f}")

    if args.expect_n:
        want = dict(x.split("=") for x in args.expect_n.split(","))
        for k, n in want.items():
            got = summary.get(k, {}).get("n", 0)
            if got != int(n):
                print(f"CELL-SIZE ASSERT FAIL k={k}: expected n={n}, got {got}")
                return 5
        print("cell-size assert OK:", args.expect_n)

    outp = run / args.out
    outp.write_text(json.dumps({
        "run_dir": str(run), "spec": args.spec,
        "spec_sha256": sha256_of(Path(args.spec)),
        "spec_rows": len(spec),
        "which": args.which,
        "grader": "evalplus.eval.untrusted_check",
        "min_time_limit": args.min_time_limit, "gt_time_limit_factor": args.factor,
        "n_solution_shards": len(sol_shards), "n_metric_shards": len(met_shards),
        "n_solutions": len(sols), "n_unique": len(seen), "n_duplicate": dup,
        "by_k": summary, "rows": rows,
    }, indent=2))
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
