#!/usr/bin/env python3
"""Preflight + completeness assertions for the Paper B depth-ladder step200000 eval.

CPU-ONLY. Never imports CUDA, never touches a GPU. Called by
`scripts/eval_paperb_ladder_200k.sh`; also usable standalone for audits.

Why this file exists
--------------------
Four separate mechanisms have silently moved Paper B's core6 without touching the
model (`status/PAPERB_FLIP_BOUNDARY_RESOLVED.md`): torch version (~20 flips),
eval batch size (~107 flips), partial shard merge (keep12 arc_easy 6/8 -> +0.19pp),
GPU architecture (7-29 flips). All four are silent by default. Plus the
`--merge` path of `eval_olmo2_probe2_downstream.py` does NOT refuse a partial
merge (it only counts `n_skipped_shards`, which never reaches summary.json), so
"the merge succeeded" is not evidence that all items were scored.

Subcommands
-----------
  ckpt      read {step, keep_front_layers, n_fresh_layers, has_optimizer} out of a
            trainer .pt via mmap (no weight materialisation, ~6 s on a 34 GiB file)
            and assert them against the expected values. This is the assertion
            that stops us evaluating step176500 and labelling it 200k.
  battery   after a merge, assert every task's `n_scored` equals the pinned
            expected count (NOT just `n_nan == 0`), that `n_shards == 8`, and that
            `add_bos` is false. For PPL, assert `n_windows` / `n_tokens`.

Expected counts are pinned from the clean single-protocol `_v2` batteries on
zwfy6/H20 that `status/PAPERB_FLIP_BOUNDARY_RESOLVED.md` designates as "the
version of Table 4 to publish" (read 2026-08-15 from
`zwfy6:olmo2_downstream_results/7B_{keep8_step121000,keep10_step83500,keep12_step124000}_v2{,_know}/summary.json`
and `zwfy6:olmo2_ppl_results/7B_keep8_step121000_v2/summary.json`); all three arms
agree exactly, so the numbers below are the protocol's item inventory, not one
run's accident.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Pinned item inventories. Provenance: see module docstring.
# core6 = the 6 tasks in scripts/eval_olmo2_probe2_downstream.py:79 (ALL_TASKS).
# know5 = the 5 tasks passed by scripts/_run_olmo2_eval_keep14_s200000_b200.sh:63
#         and scripts/_run_olmo2_eval_freezefront_s200000.sh:64. NOTE this is a
#         5-task subset of KNOWLEDGE_TASKS (downstream.py:91), which also lists
#         mmlu_pro; mmlu_pro is NOT part of the Paper B know5 battery.
# ---------------------------------------------------------------------------
CORE6_EXPECTED = {
    "hellaswag": 10042,
    "arc_challenge": 1172,
    "arc_easy": 2376,
    "piqa": 1838,
    "winogrande": 1267,
    "openbookqa": 500,
}
KNOW5_EXPECTED = {
    "mmlu": 14042,
    "lambada_openai": 5153,
    "boolq": 3270,
    "commonsense_qa": 1221,
    "social_iqa": 1954,
}
PPL_EXPECTED = {"n_windows": 4096, "n_tokens": 8384512}

BATTERIES = {"core6": CORE6_EXPECTED, "know5": KNOW5_EXPECTED}

# keep_front per arm, from the live trainer command lines (2026-08-15):
#   keep8  --keep_front_layers 8   (.82 pid 1329294)
#   keep10 --keep_front_layers 10  (LOCAL pid 2937858)
#   keep12 --keep_front_layers 12  (.73 pid 3913545)
ARM_KEEP_FRONT = {"keep8": 8, "keep10": 10, "keep12": 12}


def _fail(msg: str) -> None:
    print(f"ASSERT-FAIL: {msg}", file=sys.stderr)
    sys.exit(2)


def _ok(msg: str) -> None:
    print(f"ASSERT-OK: {msg}")


# ---------------------------------------------------------------------------
def cmd_ckpt(a) -> None:
    if not os.path.isfile(a.path):
        _fail(f"ckpt does not exist: {a.path}")
    size = os.path.getsize(a.path)
    if size < 1_000_000_000:
        _fail(f"ckpt is implausibly small ({size} B): {a.path} "
              "(a partially-written / truncated save?)")

    import torch  # local import: keeps `--help` and path checks torch-free

    try:
        ck = torch.load(a.path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as e:  # noqa: BLE001 - any load failure is fatal here
        _fail(f"torch.load failed on {a.path}: {type(e).__name__}: {e}")
    if not isinstance(ck, dict):
        _fail(f"ckpt is not a dict (got {type(ck)}): {a.path}")
    # Note: 'model_state' is this trainer's key (train_olmo2_arch_probe2.py:528).
    # SparseForge exporters use 'model_state_dict'; do not confuse them.
    if "model_state" not in ck:
        _fail(f"ckpt has no 'model_state' key (keys={sorted(ck)[:8]}): {a.path}")

    got = {
        "path": a.path,
        "size_bytes": size,
        "step": ck.get("step"),
        "keep_front_layers": ck.get("keep_front_layers"),
        "n_fresh_layers": ck.get("n_fresh_layers"),
        "num_hidden_layers": ck.get("num_hidden_layers"),
        "has_optimizer": "optimizer_state" in ck,
        "model_family": ck.get("model_family"),
        "base_model_path": ck.get("base_model_path"),
        "n_model_tensors": len(ck["model_state"]),
    }
    ta = ck.get("train_args") or {}
    if isinstance(ta, dict):
        got["train_batch_size"] = ta.get("batch_size")
        got["train_grad_accum"] = ta.get("grad_accumulation_steps")
        got["train_data_path"] = ta.get("data_path")

    if a.expect_step is not None and int(got["step"] or -1) != int(a.expect_step):
        _fail(f"ckpt step is {got['step']}, expected {a.expect_step}: {a.path}. "
              "Refusing to evaluate a non-endpoint checkpoint under a 200k label.")
    if a.expect_keep is not None and int(got["keep_front_layers"] or -1) != int(a.expect_keep):
        _fail(f"ckpt keep_front_layers is {got['keep_front_layers']}, "
              f"expected {a.expect_keep}: {a.path}")
    if a.expect_fresh is not None and int(got["n_fresh_layers"] or -1) != int(a.expect_fresh):
        _fail(f"ckpt n_fresh_layers is {got['n_fresh_layers']}, "
              f"expected {a.expect_fresh}: {a.path}")

    print(json.dumps(got, indent=2, sort_keys=True))
    _ok(f"ckpt step={got['step']} keep_front={got['keep_front_layers']} "
        f"n_fresh={got['n_fresh_layers']} tensors={got['n_model_tensors']}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(got, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
def cmd_battery(a) -> None:
    d = os.path.join(a.results_root, a.name)
    sp = os.path.join(d, "summary.json")
    if not os.path.isfile(sp):
        _fail(f"no summary.json in {d} (merge did not run, or ran elsewhere)")
    with open(sp) as f:
        s = json.load(f)

    problems: list[str] = []
    n_shards = s.get("n_shards")
    if int(n_shards or 0) != int(a.num_shards):
        problems.append(f"n_shards={n_shards} EXPECT {a.num_shards}")

    if a.kind == "ppl":
        for k, want in PPL_EXPECTED.items():
            got = s.get(k)
            if int(got or -1) != int(want):
                problems.append(f"{k}={got} EXPECT {want}")
        ppl = s.get("ppl")
        if not isinstance(ppl, (int, float)) or not (ppl > 0):
            problems.append(f"ppl={ppl} is not a positive number")
    else:
        expected = BATTERIES[a.kind]
        if s.get("add_bos") is not False:
            problems.append(f"add_bos={s.get('add_bos')} EXPECT False "
                            "(Paper B base protocol: OLMo-2 gets no BOS)")
        tasks = s.get("tasks") or {}
        missing = sorted(set(expected) - set(tasks))
        extra = sorted(set(tasks) - set(expected))
        if missing:
            problems.append(f"tasks missing from summary: {missing}")
        if extra:
            problems.append(f"unexpected extra tasks in summary: {extra}")
        for t, want in expected.items():
            v = tasks.get(t)
            if not isinstance(v, dict):
                continue
            if v.get("skipped"):
                problems.append(f"{t}: SKIPPED ({str(v.get('error'))[:80]})")
                continue
            n_scored = v.get("n_scored")
            n_tot = v.get("n")
            n_nan = v.get("n_nan")
            # n_scored is the load-bearing one: n == expected with n_nan > 0 still
            # means fewer items entered the accuracy denominator.
            if int(n_scored or -1) != int(want):
                frac = (float(n_scored or 0) / want * a.num_shards) if want else 0.0
                problems.append(
                    f"{t}: n_scored={n_scored} EXPECT {want} "
                    f"(={frac:.2f}/{a.num_shards} shards)")
            if int(n_tot or -1) != int(want):
                problems.append(f"{t}: n={n_tot} EXPECT {want}")
            if int(n_nan or 0) != 0:
                problems.append(f"{t}: n_nan={n_nan} EXPECT 0")
            for m in ("acc", "acc_norm"):
                if not isinstance(v.get(m), (int, float)):
                    problems.append(f"{t}: {m}={v.get(m)} is not numeric")

    if problems:
        for p in problems:
            print(f"  DEFECT: {p}", file=sys.stderr)
        _fail(f"{a.kind} battery {a.name} is INCOMPLETE ({len(problems)} defects). "
              "Do NOT use these numbers; re-run the failed shards and re-merge.")
    _ok(f"{a.kind} battery {a.name}: complete "
        f"({'ppl=%.4f' % s['ppl'] if a.kind == 'ppl' else str(len(BATTERIES[a.kind])) + '/' + str(len(BATTERIES[a.kind])) + ' tasks at full n_scored'})")


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("ckpt", help="assert a trainer .pt's step / arch meta")
    c.add_argument("--path", required=True)
    c.add_argument("--expect-step", type=int, default=None)
    c.add_argument("--expect-keep", type=int, default=None)
    c.add_argument("--expect-fresh", type=int, default=None)
    c.add_argument("--out", default="", help="also write the probed meta here")
    c.set_defaults(func=cmd_ckpt)

    b = sub.add_parser("battery", help="assert a merged battery is complete")
    b.add_argument("--results-root", required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--kind", required=True, choices=["ppl", "core6", "know5"])
    b.add_argument("--num-shards", type=int, default=8)
    b.set_defaults(func=cmd_battery)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
