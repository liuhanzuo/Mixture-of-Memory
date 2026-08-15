#!/usr/bin/env python3
"""B10 Gate 1 -- re-measure the base-axis gold ceiling on THIS disk's split.

The pre-registered kill condition quotes "gold ceiling 0.9894". That number was
computed on wzc1 (dllm_draft/runs/spanlen/gold_ceiling_SingleLine.json,
gold_ceiling_base = 0.989351403678606). Gate 1 runs on zwfy6, so we re-measure
it here with the same grader the arms are scored with, and also record WHICH
items are gold-infeasible so the contrast can be repeated on the feasible subset.

Splices the benchmark's own canonical_solution back in:
    program = prompt + canonical_solution + suffix
graded with --which base semantics (base_input / ref["base"]).

ZERO GPU.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_infilling import grade_one, load_evalplus, read_jsonl  # noqa: E402

_HE = None
_GT = None


def _job(bid, program, which, mtl, factor):
    global _HE, _GT
    if _HE is None:
        _HE, _GT = load_evalplus()
    return grade_one(_HE[bid], _GT[bid], program, which=which,
                     min_time_limit=mtl, factor=factor)


def main() -> None:
    data_file = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    mtl, factor = 1.0, 4.0

    rows = list(read_jsonl(data_file))
    per_row = {}
    for which in ("base", "plus"):
        specs = [(r["task_id"],
                  f"{r['task_id'].split('/')[1]}/{r['task_id'].split('/')[2]}",
                  r["prompt"] + r["canonical_solution"] + r["suffix"])
                 for r in rows]
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_job, bid, prog, which, mtl, factor): tid
                    for tid, bid, prog in specs}
            for fut, tid in futs.items():
                ok, why, ntests, npass = fut.result()
                per_row.setdefault(tid, {"task_id": tid})
                per_row[tid][f"gold_{which}_pass"] = bool(ok)
                per_row[tid][f"gold_{which}_why"] = why

    n = len(rows)
    nb = sum(per_row[t]["gold_base_pass"] for t in per_row)
    npl = sum(per_row[t]["gold_plus_pass"] for t in per_row)
    out = {
        "split": "SingleLine",
        "data_file": str(data_file),
        "grader": "evalplus.eval.untrusted_check (official sandbox), "
                  "via score_infilling.grade_one",
        "n_rows": n,
        "gold_ceiling_base": nb / n,
        "gold_ceiling_plus": npl / n,
        "n_gold_base_pass": nb,
        "n_gold_plus_pass": npl,
        "per_row": [per_row[r["task_id"]] for r in rows],
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("n_rows", "gold_ceiling_base", "gold_ceiling_plus",
                       "n_gold_base_pass", "n_gold_plus_pass")}, indent=2))


if __name__ == "__main__":
    main()
