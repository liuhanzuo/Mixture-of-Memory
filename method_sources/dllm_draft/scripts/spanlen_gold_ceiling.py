#!/usr/bin/env python3
"""Measure the GOLD CEILING of each HumanEval-Infilling split, per length stratum,
at BOTH grading axes (base / plus).

WHY THIS EXISTS
===============
A stratified pass@1 comparison is meaningless if the *attainable maximum* varies
across strata. So before comparing any model arm, we splice the split's OWN gold
middle back in and grade it:

    program = row['prompt'] + row['canonical_solution'] + row['suffix']

That program is, by construction, the reference answer for that infilling item.
Whatever it scores IS the ceiling for that item.

Grading is delegated entirely to EvalPlus's official sandbox
(``evalplus.eval.untrusted_check``) against EvalPlus ground-truth expected
outputs, exactly as scripts/score_infilling.py does. Nothing about test
execution, comparison, or timeouts is re-implemented here.

The base/plus distinction is load-bearing and is why this script reports both:
HumanEval-Infilling was built on the ORIGINAL HumanEval reference solutions,
while HumanEval+ ships *corrected* canonical solutions and a much larger,
stricter input set. The two are not interchangeable.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

STRATA = [(0, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 100000)]


def stratum_of(n: int) -> str:
    for lo, hi in STRATA:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 100000 else f"{lo}+"
    raise AssertionError(n)


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as h:
        for line in h:
            if line.strip():
                yield json.loads(line)


def base_task_id(task_id: str) -> str:
    p = task_id.split("/")
    return f"{p[1]}/{p[2]}"


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
    """Official EvalPlus sandbox. Returns (ok, n_tests, n_pass)."""
    from evalplus.eval import PASS, untrusted_check

    he, gt = _load()
    task, ref = he[bid], gt[bid]
    if which == "base":
        inputs = list(task["base_input"])
        expected = list(ref["base"])
        ref_time = list(ref["base_time"])
    else:
        inputs = list(task["base_input"]) + list(task["plus_input"])
        expected = list(ref["base"]) + list(ref["plus"])
        ref_time = list(ref["base_time"]) + list(ref["plus_time"])
    try:
        status, details = untrusted_check(
            "humaneval", program, inputs, task["entry_point"],
            expected=expected, atol=task["atol"], ref_time=ref_time,
            fast_check=False, min_time_limit=1.0, gt_time_limit_factor=4.0,
        )
    except Exception as exc:  # noqa: BLE001
        return False, 0, 0, f"GRADER_ERR:{type(exc).__name__}"
    details = list(details) if details is not None else []
    n_pass = int(sum(bool(d) for d in details))
    ok = status == PASS and len(details) > 0 and n_pass == len(details)
    return ok, len(details), n_pass, (None if ok else str(status))


def _job(args):
    tid, bid, program, which = args
    ok, nt, npass, why = _grade(bid, program, which)
    return tid, which, ok, nt, npass, why


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--split-name", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True,
                                        local_files_only=True)
    rows = list(read_jsonl(Path(args.data_file)))
    if args.limit:
        rows = rows[: args.limit]

    jobs = []
    meta = {}
    for r in rows:
        tid = r["task_id"]
        bid = base_task_id(tid)
        gt_len = len(tok.encode(r["canonical_solution"], add_special_tokens=False))
        prog = r["prompt"] + r["canonical_solution"] + r["suffix"]
        meta[tid] = {"base_task_id": bid, "gt_len": gt_len,
                     "stratum": stratum_of(gt_len)}
        jobs.append((tid, bid, prog, "base"))
        jobs.append((tid, bid, prog, "plus"))

    from concurrent.futures import ProcessPoolExecutor

    res = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for tid, which, ok, nt, npass, why in pool.map(_job, jobs, chunksize=4):
            res[tid][which] = {"pass": ok, "n_tests": nt, "n_pass": npass, "why": why}

    # aggregate
    by_stratum = defaultdict(lambda: {"n": 0, "base_pass": 0, "plus_pass": 0})
    per_row = []
    for tid, m in meta.items():
        b = res[tid]["base"]["pass"]
        p = res[tid]["plus"]["pass"]
        s = by_stratum[m["stratum"]]
        s["n"] += 1
        s["base_pass"] += bool(b)
        s["plus_pass"] += bool(p)
        per_row.append({"task_id": tid, **m, "gold_base_pass": bool(b),
                        "gold_plus_pass": bool(p),
                        "base_why": res[tid]["base"]["why"],
                        "plus_why": res[tid]["plus"]["why"]})

    order = [f"{lo}-{hi}" if hi < 100000 else f"{lo}+" for lo, hi in STRATA]
    strata_out = {}
    for k in order:
        if k not in by_stratum:
            continue
        v = by_stratum[k]
        strata_out[k] = {
            "n": v["n"],
            "gold_ceiling_base": v["base_pass"] / v["n"],
            "gold_ceiling_plus": v["plus_pass"] / v["n"],
            "n_base_pass": v["base_pass"], "n_plus_pass": v["plus_pass"],
        }
    n = len(meta)
    out = {
        "split": args.split_name, "data_file": args.data_file, "n_rows": n,
        "grader": "evalplus.eval.untrusted_check (official sandbox)",
        "overall": {
            "gold_ceiling_base": sum(v["base_pass"] for v in by_stratum.values()) / n,
            "gold_ceiling_plus": sum(v["plus_pass"] for v in by_stratum.values()) / n,
        },
        "by_stratum": strata_out,
        "per_row": per_row,
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"split": args.split_name, "n": n,
                      "overall": out["overall"],
                      "by_stratum": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                         for kk, vv in v.items()}
                                     for k, v in strata_out.items()}}, indent=2))


if __name__ == "__main__":
    main()
