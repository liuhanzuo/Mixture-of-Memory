#!/usr/bin/env python3
"""Canonical per-item shard loaders for the OLMo-2 knowledge-eval harnesses.

RELOCATED 2026-08-11 (no behavioural change) from
`proposal/active/A03-parametric-vs-external-memory/code/recompute_cpt_trajectory_paired.py`
so that archiving the A03 proposal cannot break the analyses that depend on
these loaders. A03 was decided ARCHIVE on 2026-08-11 (`ARM_SET_DECISION.md`),
but A04's Stage-A/Stage-B numbers -- and A03's own seed-45 recompute -- are
produced by `load_cb` / `load_mmlu` / `paired` defined here. Moving the proposal
directory while the loaders lived inside it would have silently repointed or
hard-failed those analyses, so the loaders were lifted to `proposal/shared/`
FIRST and the move made only after byte-for-byte re-verification of the A04
Stage-A/B verdict JSONs.

WHY THIS FILE IS A REAL MODULE
------------------------------
`pilot_one_stage_a_sd_run.py` previously obtained these functions by reading the
A03 script's SOURCE TEXT, truncating it at the first `BASE = ` assignment, and
`exec`-ing the remainder -- because the A03 script has no `if __name__ ==
"__main__"` guard and its whole trajectory driver runs at module scope. That
worked, but it coupled every consumer to (a) A03's directory path and (b) the
exact textual shape of A03's driver boundary. Both are now gone: this file
contains ONLY definitions, so it can be imported normally.

WHAT MUST NOT CHANGE
--------------------
Every assertion below is load-bearing and was written in response to a specific
incident in this repository. Do not "simplify" any of them:

  * 8/8 shard completeness (`load_cb`, `load_mmlu`) -- a silently merged 5-of-8
    shard set has corrupted results here before. A result dir that is ABSENT
    raises `NotRunYet` (a schedule fact); a dir that EXISTS but is partial is a
    hard `SystemExit` (a data-integrity failure). Collapsing those two cases is
    exactly how a partial merge slips through.
  * `N_MMLU = 14042` exact item count, duplicate-`item_id` rejection, and
    `nan:true` rejection (`load_mmlu`) -- paired analysis requires an identical
    valid item set across arms.
  * The nested-key MMLU read, `for iface in ("letter", "content_norm")`. The
    pre-2026-08-10 version guessed FLAT key names (`letter_correct`,
    `content_norm_correct`) that the harness never writes, so every lookup fell
    through to a None default and the whole MMLU axis vanished from 12 trajectory
    cells while four .md files went on asserting "MMLU is flat". Consumers
    positively assert the presence of this exact marker string; keep it.
  * `paired()`'s protocol: n_boot=5000, seed=42, CI95 percentile, SIG iff the CI
    excludes 0. Changing any of these silently invalidates every archived cell.

The bodies below are the 2026-08-10 post-fix versions, relocated VERBATIM.
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path("/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory")
CB = ROOT / "olmo2_closedbook_results"
MM = ROOT / "olmo2_mmlu_content_results"
N_BOOT, SEED, NSHARD = 5000, 42, 8
# cais/mmlu "all" test split, the exact n every A03 MMLU summary.json reports.
N_MMLU = 14042
# Exact merged item counts per closed-book task, DERIVED from disk 2026-08-12:
# each value is constant across six independent 7B arm dirs in
# `olmo2_closedbook_results/` on zwfy6 and equals the merged unique-id count.
# Asserted for the same reason as N_MMLU -- so a truncated shard cannot pair.
# ⚠️ `nq_open` per-example files live in a SEPARATE dir suffixed `_nq`
# (e.g. `A04_1B_stageB_keep12_seed101_step5000_nq`), not alongside the others;
# a caller that globs only the main arm dir will wrongly conclude it is missing.
N_CB = {"triviaqa": 17944, "popqa": 14267, "nq_open": 3610}


class NotRunYet(Exception):
    """Result dir absent entirely -- the arm has not been evaluated yet.

    Kept strictly separate from the partial-shard case below: 'not run' is a
    schedule fact, '5/8 shards' is a data-integrity FAILURE. Collapsing the two
    is how a partial set gets silently merged, which has ruined results here
    before. Only the first is tolerated.
    """


def load_cb(d, task):
    """item_id -> (em, contains, f1); asserts 8/8 shards, exact n, no dup, no nan.

    ⚠️ 2026-08-12 HARDENING. The pre-2026-08-12 body asserted ONLY the shard file
    COUNT and then did a bare `got[r["item_id"]] = ...`, so three failure modes
    were silent:

      * a duplicate `item_id` across shards (overlapping shard ranges) was
        OVERWRITTEN, not detected -- the merged set would look complete while
        double-counting one item and dropping another;
      * a shard that was present but TRUNCATED (writer died mid-flush) passed,
        because 8 files existed; the merged n was simply short;
      * a `nan:true` row was merged as a real score, breaking the identical-
        valid-item-set assumption that `paired()` depends on.

    `load_mmlu` already asserted all three (its own 2026-08-10 fix). This function
    is the one A04's Pilot One decision axes actually go through, so the asymmetry
    was the live risk, not a cosmetic one.

    `N_CB` counts are DERIVED, not assumed: each is constant across six
    independent 7B arm dirs on zwfy6 and equals the merged unique-id count there.
    """
    got = {}
    if not (CB / d).is_dir():
        raise NotRunYet(f"{d} absent")
    if task not in N_CB:
        raise SystemExit(f"FATAL load_cb: unknown task {task!r} -- add its exact "
                         f"item count to N_CB (known: {sorted(N_CB)}) rather than "
                         "letting an unchecked task through")
    files = sorted((CB / d).glob(f"per_example_{task}_shard*of{NSHARD}.jsonl"))
    if len(files) != NSHARD:
        raise SystemExit(f"FATAL {d}/{task}: {len(files)}/{NSHARD} shards -- refusing "
                         "(a silently-merged partial set has ruined results here before)")
    for f in files:
        for ln in f.open():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if "item_id" not in r:
                raise SystemExit(f"FATAL {d}/{task}: record has no 'item_id' "
                                 f"(keys={sorted(r)}) -- schema changed, refusing to guess")
            iid = r["item_id"]
            if iid in got:
                raise SystemExit(f"FATAL {d}/{task}: duplicate item_id {iid} across shards "
                                 "-- overlapping shard ranges would double-count")
            if r.get("nan"):
                raise SystemExit(f"FATAL {d}/{task}: item_id {iid} has nan=true -- "
                                 "paired analysis needs an identical valid item set")
            for k in ("em", "contains", "f1"):
                if k not in r:
                    raise SystemExit(f"FATAL {d}/{task}: item_id {iid} has no {k!r} "
                                     f"(keys={sorted(r)}) -- schema changed")
            got[iid] = (r["em"], r["contains"], r["f1"])
    if len(got) != N_CB[task]:
        raise SystemExit(f"FATAL {d}/{task}: merged {len(got)} items, expected "
                         f"{N_CB[task]} -- truncated shard or wrong eval set")
    return got


def load_mmlu(d):
    """item_id -> (letter_correct, content_norm_correct); asserts 8/8 shards.

    ⚠️ 2026-08-10 BUG FIX -- this function silently returned None-filled data for
    EVERY MMLU cell, so the MMLU axis was missing from all 12 trajectory cells of
    the canonical evidence JSON (arm3/arm4/arm6 x 4 dose points) while four .md
    files went on asserting "MMLU is flat across the trajectory".

    The old body guessed FLAT key names off the record:
        r.get("letter_correct", r.get("correct_letter", r.get("em")))
        r.get("content_norm_correct", r.get("correct_content_norm", ...))
    None of those keys exist. `scripts/eval_olmo2_mmlu_content.py` (see its
    score_examples() docstring and rec construction) writes a NESTED record:
        {item_id, subject, gold, gold_letter, n_opt, nan,
         letter:       {pred, pred_letter, correct, scores},
         content_raw:  {pred, pred_letter, correct, scores},
         content_norm: {pred, pred_letter, correct, scores, cont_tokens}}
    So every `.get` chain fell through to its default of None, `got` filled with
    (None, None), and the caller's `all(mb[i][0] is not None ...)` guard then
    dropped the whole cell WITHOUT writing a "pending"/"error" marker -- the cell
    key simply never appeared. That is why grepping the evidence JSON for MMLU
    found nothing at all rather than an error.

    This version reads the real nested keys and HARD-FAILS on anything it does
    not understand: a missing key, a non-bool `correct`, a duplicate item_id, or
    a `nan:true` row. Silence is what caused the defect, so there is no
    silent-default path left. `analyze_1b_knowledge_floor.py` already read these
    same nested keys correctly (lines 399-400) -- only this script was wrong.
    """
    got = {}
    if not (MM / d).is_dir():
        raise NotRunYet(f"{d} absent")
    files = sorted((MM / d).glob(f"per_example_mmlu_shard*of{NSHARD}.jsonl"))
    if len(files) != NSHARD:
        # Same rule as load_cb: a present-but-partial set is a FAILURE, never a
        # silent skip. Previously this returned None, which made a 5/8 MMLU set
        # indistinguishable from "not evaluated" and simply dropped the cell.
        raise SystemExit(f"FATAL {d}/mmlu: {len(files)}/{NSHARD} shards -- refusing "
                         "(a silently-merged partial set has ruined results here before)")
    for f in files:
        for ln in f.open():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if "item_id" not in r:
                raise SystemExit(f"FATAL {d}/mmlu: record has no 'item_id' "
                                 f"(keys={sorted(r)}) -- schema changed, refusing to guess")
            iid = r["item_id"]
            if iid in got:
                raise SystemExit(f"FATAL {d}/mmlu: duplicate item_id {iid} across shards "
                                 "-- overlapping shard ranges would double-count")
            if r.get("nan"):
                # The harness marks an item nan when ANY candidate scored
                # non-finite, and drops it from every accuracy. A03's arms all
                # report n_nan=0, so encountering one means the pairing
                # assumption (identical valid item set per arm) is broken.
                raise SystemExit(f"FATAL {d}/mmlu: item_id {iid} has nan=true -- "
                                 "paired analysis needs an identical valid item set")
            vals = []
            for iface in ("letter", "content_norm"):
                if iface not in r or not isinstance(r[iface], dict):
                    raise SystemExit(f"FATAL {d}/mmlu: item_id {iid} missing nested "
                                     f"'{iface}' dict (keys={sorted(r)}) -- schema changed")
                if "correct" not in r[iface]:
                    raise SystemExit(f"FATAL {d}/mmlu: item_id {iid} '{iface}' has no "
                                     f"'correct' (keys={sorted(r[iface])})")
                c = r[iface]["correct"]
                if not isinstance(c, bool):
                    raise SystemExit(f"FATAL {d}/mmlu: item_id {iid} '{iface}.correct' is "
                                     f"{c!r} ({type(c).__name__}), expected bool")
                vals.append(1.0 if c else 0.0)
            got[iid] = tuple(vals)
    if len(got) != N_MMLU:
        raise SystemExit(f"FATAL {d}/mmlu: merged {len(got)} items, expected {N_MMLU} "
                         "(cais/mmlu 'all' test split) -- incomplete or wrong dump")
    return got


def paired(base, arm, idx):
    """CI95 of mean(arm-base) in percentage points."""
    b = np.array([base[i] for i in idx], dtype=float)
    a = np.array([arm[i] for i in idx], dtype=float)
    d = a - b
    rng = np.random.default_rng(SEED)
    n = len(d)
    boots = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1) * 100.0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    delta = float(d.mean() * 100.0)
    return {"n": n, "delta_pp": delta, "ci95_pp": [float(lo), float(hi)],
            "verdict": "SIG" if (lo > 0 or hi < 0) else "TIE"}

