#!/usr/bin/env python
"""0-GPU preflight: can this node LOAD all 9 union-9 task datasets, with the
same document counts the completed arms recorded?

WHY THIS EXISTS
    A union-9 gap-fill row costs ~10 min of GPU once it starts, but the failure
    that actually bites is a dataset that will not resolve at all (piqa's
    loading-script removal in datasets 5.0.1) or one that resolves to a
    DIFFERENT number of docs than the arms already in the table. The second is
    the dangerous one: lm_eval will happily score 1838-vs-N docs and emit a
    perfectly-formed results json with n_nan=0. That is the same class of error
    as the 6/8-shard silent merge -- a complete-looking artefact on a changed
    measurement basis.

    So this probe asserts n per task against the counts recorded in the two
    completed token-matched arms' tokenmatched_union9_summary.json, and refuses
    to report OK on a mismatch.

USAGE (no GPU touched -- pure dataset loading)
    export http_proxy=http://hy-proxy.woa.com:3128 https_proxy=$http_proxy
    export HF_HUB_OFFLINE=0 HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=1
    venv_union9/bin/python .../probe_union9_datasets.py

Exit 0 = all 9 load with matching n. Nonzero = do not launch the eval.
"""
from __future__ import annotations

import sys
import traceback

# n_samples as recorded by BOTH completed token-matched arms
# (outputs/cast_eval_spec/sparseforge_tokenmatched_{slorb,noslorb}/
#  tokenmatched_union9_summary.json -> per_task[*].n_samples), and identically by
# every arm in sparseforge_5b/sparseforge_same_harness_table.json.
EXPECT_N = {
    "boolq": 3270,
    "rte": 277,
    "hellaswag": 10042,
    "race": 1045,
    "piqa": 1838,
    "winogrande": 1267,
    "arc_easy": 2376,
    "arc_challenge": 1172,
    "openbookqa": 500,
}


def main() -> int:
    try:
        import importlib.metadata as md

        import lm_eval  # noqa: F401 - imported for the side effect of proving it loads
        from lm_eval.tasks import TaskManager
    except Exception:
        traceback.print_exc()
        print("VERDICT: FAIL -- cannot import lm_eval")
        return 3

    # ⚠️ lm_eval 0.4.8 does NOT define `lm_eval.__version__`; `getattr(..., "unknown")`
    # therefore reports drift on a perfectly correct install. The distribution
    # metadata is the authoritative source, and is what the watcher's own harness
    # assertion uses (scripts/_run_sparseforge_tokenmatched_union9_watcher.sh:253).
    try:
        ver = md.version("lm_eval")
    except Exception as exc:  # noqa: BLE001
        print(f"VERDICT: FAIL -- lm_eval distribution metadata unreadable: {exc}")
        return 4
    print(f"[probe] lm_eval {ver}")
    if ver != "0.4.8":
        print(f"VERDICT: FAIL -- lm_eval {ver} != 0.4.8; harness drift")
        return 4

    tm = TaskManager()
    bad: list[str] = []
    for task, want in EXPECT_N.items():
        try:
            d = tm.load_task_or_group([task])
            obj = d[task]
            got = len(list(obj.eval_docs))
        except Exception as exc:  # noqa: BLE001 - want the class name in the log
            print(f"[probe] {task:14s} FAIL {type(exc).__name__}: {str(exc)[:160]}")
            bad.append(task)
            continue
        ok = got == want
        print(f"[probe] {task:14s} {'OK ' if ok else 'N-MISMATCH'} n={got} expected={want}")
        if not ok:
            bad.append(task)

    if bad:
        print(f"VERDICT: FAIL on {bad} -- do NOT launch the union-9 eval on this node")
        return 5
    print("VERDICT: OK -- all 9 tasks load with n identical to the completed arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
