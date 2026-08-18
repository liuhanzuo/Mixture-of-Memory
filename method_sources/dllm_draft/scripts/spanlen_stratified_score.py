#!/usr/bin/env python3
"""Span-length stratified scoring of HumanEval-Infilling arms.

Emits, per length stratum and for BOTH grading axes (base / plus):
  * raw pass@1
  * the GOLD CEILING of that stratum (from spanlen_gold_ceiling.py)
  * ceiling-conditioned pass@1 = pass@1 restricted to items whose own gold
    solution passes. This is the only length comparison that is not confounded
    by a stratum-dependent ceiling.
  * truncation / abort / generation-error counts, reported SEPARATELY from
    grading failures.

Grading is delegated entirely to EvalPlus's official sandbox
(``evalplus.eval.untrusted_check``). Nothing about test execution is
re-implemented. A grader self-test (gold must pass, `pass` stub must fail) runs
on every invocation and the script refuses to emit scores if it fails.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

STRATA = [(0, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 10**9)]


def sname(lo: int, hi: int) -> str:
    return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"


ORDER = [sname(lo, hi) for lo, hi in STRATA]


def stratum_of(n: int) -> str:
    for lo, hi in STRATA:
        if lo <= n <= hi:
            return sname(lo, hi)
    raise AssertionError(n)


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as h:
        for line in h:
            if line.strip():
                yield json.loads(line)


_HE = None
_GT = None


def _load():
    global _HE, _GT
    if _HE is None:
        from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
        from evalplus.evaluate import get_groundtruth

        _HE = get_human_eval_plus()
        _GT = get_groundtruth(_HE, get_human_eval_plus_hash(), [])
    return _HE, _GT


def _grade(bid: str, program: str, which: str):
    from evalplus.eval import PASS, untrusted_check

    he, gt = _load()
    task, ref = he[bid], gt[bid]
    if which == "base":
        inputs, expected, rt = (list(task["base_input"]), list(ref["base"]),
                                list(ref["base_time"]))
    else:
        inputs = list(task["base_input"]) + list(task["plus_input"])
        expected = list(ref["base"]) + list(ref["plus"])
        rt = list(ref["base_time"]) + list(ref["plus_time"])
    try:
        status, details = untrusted_check(
            "humaneval", program, inputs, task["entry_point"],
            expected=expected, atol=task["atol"], ref_time=rt,
            fast_check=False, min_time_limit=1.0, gt_time_limit_factor=4.0,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"GRADER_ERR:{type(exc).__name__}"
    details = list(details) if details is not None else []
    ok = status == PASS and len(details) > 0 and all(bool(d) for d in details)
    return ok, (None if ok else str(status))


def _job(a):
    tid, bid, prog, which = a
    ok, why = _grade(bid, prog, which)
    return tid, which, ok, why


def self_test(data_file, n=12):
    """Refuse to score through an inflated grader."""
    import random

    rows = list(read_jsonl(Path(data_file)))
    random.Random(0).shuffle(rows)
    rows = rows[:n]
    gold_ok = stub_fail = 0
    for r in rows:
        p = r["task_id"].split("/")
        bid = f"{p[1]}/{p[2]}"
        ok, _ = _grade(bid, r["prompt"] + r["canonical_solution"] + r["suffix"], "base")
        gold_ok += ok
        ok2, _ = _grade(bid, r["prompt"] + "    pass\n" + r["suffix"], "base")
        stub_fail += (not ok2)
    return {"n": len(rows), "gold_pass": gold_ok, "stub_fail": stub_fail,
            "trustworthy": gold_ok == len(rows) and stub_fail == len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--ceiling", required=True, help="gold_ceiling_<split>.json")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--split-name", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--jobs", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True,
                                        local_files_only=True)

    st = self_test(args.data_file)
    print("GRADER SELF-TEST:", json.dumps(st), flush=True)
    if not st["trustworthy"]:
        raise SystemExit("grader self-test FAILED -- refusing to emit scores")

    rows = {r["task_id"]: r for r in read_jsonl(Path(args.data_file))}
    sols = list(read_jsonl(Path(args.solutions)))
    mets = {m["task_id"]: m for m in read_jsonl(Path(args.metrics))}
    if len(sols) != len(rows):
        raise SystemExit(f"COVERAGE: {len(sols)} solutions != {len(rows)} data rows")
    if len(mets) != len(rows):
        raise SystemExit(f"COVERAGE: {len(mets)} metrics != {len(rows)} data rows")

    ceil = json.load(open(args.ceiling))
    gold_pass = {r["task_id"]: {"base": r["gold_base_pass"], "plus": r["gold_plus_pass"]}
                 for r in ceil["per_row"]}

    jobs, meta = [], {}
    for s in sols:
        tid = s["task_id"]
        p = tid.split("/")
        bid = s.get("base_task_id") or f"{p[1]}/{p[2]}"
        gt_len = len(tok.encode(rows[tid]["canonical_solution"], add_special_tokens=False))
        meta[tid] = {"base_task_id": bid, "gt_len": gt_len, "stratum": stratum_of(gt_len)}
        jobs.append((tid, bid, s["solution"], "base"))
        jobs.append((tid, bid, s["solution"], "plus"))

    from concurrent.futures import ProcessPoolExecutor

    res = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for tid, which, ok, why in pool.map(_job, jobs, chunksize=4):
            res[tid][which] = {"pass": ok, "why": why}

    agg = defaultdict(lambda: {
        "n": 0, "base_pass": 0, "plus_pass": 0,
        "n_gold_base": 0, "cond_base_pass": 0,
        "n_gold_plus": 0, "cond_plus_pass": 0,
        "trunc": 0, "abort": 0, "gen_err": 0, "not_parseable": 0,
        "middle_tokens_sum": 0, "elapsed_sum": 0.0,
    })
    per_row = []
    for tid, m in meta.items():
        k = m["stratum"]
        a = agg[k]
        met = mets[tid]
        info = met.get("info") or {}
        b = bool(res[tid]["base"]["pass"])
        p_ = bool(res[tid]["plus"]["pass"])
        a["n"] += 1
        a["base_pass"] += b
        a["plus_pass"] += p_
        gb = bool(gold_pass[tid]["base"])
        gp = bool(gold_pass[tid]["plus"])
        if gb:
            a["n_gold_base"] += 1
            a["cond_base_pass"] += b
        if gp:
            a["n_gold_plus"] += 1
            a["cond_plus_pass"] += p_
        # Termination / error accounting, kept strictly separate from grading.
        rec = info.get("recovered")
        if rec == "suffix_not_found":
            a["trunc"] += 1
        if met.get("error"):
            a["abort"] += 1
            a["gen_err"] += 1
        if not met.get("parseable"):
            a["not_parseable"] += 1
        a["middle_tokens_sum"] += int(info.get("middle_tokens") or 0)
        a["elapsed_sum"] += float(met.get("elapsed_seconds") or 0.0)
        per_row.append({"task_id": tid, **m, "base_pass": b, "plus_pass": p_,
                        "gold_base_pass": gb, "gold_plus_pass": gp,
                        "recovered": rec, "error": met.get("error"),
                        "middle_tokens": info.get("middle_tokens")})

    cmap = ceil["by_stratum"]
    out_strata = {}
    for k in ORDER:
        if k not in agg:
            continue
        v = agg[k]
        n = v["n"]
        out_strata[k] = {
            "n": n,
            "pass_at_1_base": v["base_pass"] / n,
            "pass_at_1_plus": v["plus_pass"] / n,
            "gold_ceiling_base": cmap.get(k, {}).get("gold_ceiling_base"),
            "gold_ceiling_plus": cmap.get(k, {}).get("gold_ceiling_plus"),
            "cond_pass_base": (v["cond_base_pass"] / v["n_gold_base"]) if v["n_gold_base"] else None,
            "n_cond_base": v["n_gold_base"],
            "cond_pass_plus": (v["cond_plus_pass"] / v["n_gold_plus"]) if v["n_gold_plus"] else None,
            "n_cond_plus": v["n_gold_plus"],
            "truncated": v["trunc"], "aborts": v["abort"],
            "generation_errors": v["gen_err"], "not_parseable": v["not_parseable"],
            "middle_tokens_mean": v["middle_tokens_sum"] / n,
            "latency_s_mean": v["elapsed_sum"] / n,
        }
    N = len(meta)
    tot = lambda f: sum(agg[k][f] for k in agg)  # noqa: E731
    out = {
        "arm": args.arm, "split": args.split_name, "n": N,
        "grader": "evalplus.eval.untrusted_check (official sandbox)",
        "grader_self_test": st,
        "evalplus_max_memory_bytes": os.environ.get("EVALPLUS_MAX_MEMORY_BYTES", "default(4GiB)"),
        "overall": {
            "pass_at_1_base": tot("base_pass") / N,
            "pass_at_1_plus": tot("plus_pass") / N,
            "gold_ceiling_base": ceil["overall"]["gold_ceiling_base"],
            "gold_ceiling_plus": ceil["overall"]["gold_ceiling_plus"],
            "cond_pass_base": tot("cond_base_pass") / tot("n_gold_base"),
            "cond_pass_plus": tot("cond_plus_pass") / tot("n_gold_plus"),
            "truncated": tot("trunc"), "aborts": tot("abort"),
            "generation_errors": tot("gen_err"), "not_parseable": tot("not_parseable"),
            "latency_s_mean": sum(agg[k]["elapsed_sum"] for k in agg) / N,
        },
        "by_stratum": out_strata,
        "per_row": per_row,
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"arm": args.arm, "split": args.split_name, "n": N,
                      "overall": out["overall"],
                      "by_stratum": out_strata}, indent=2, default=str))


if __name__ == "__main__":
    main()
