#!/usr/bin/env python3
"""proposal/ready_queue.py — make "what can I run right now" a computed property.

Why this exists
---------------
See `proposal/LIFECYCLE_SCHEMA.md`. On 2026-08-14 the scheduler (me, MAIN) caused
five separate stalls by HAND-READING 15 proposal directories and guessing. Every
one of those stalls had its answer already on disk:

  * B02 was called "blocked, nothing to do" — its `next_gate` names a concrete
    1-GPU x 8-config re-run.
  * B11's K1 gate is **0 GPU and blocking**, and its own `gpu_policy` says so.
    It sat untouched because nothing put it in the scheduler's view.
  * B06 carries a confirmed p=2.6e-67 result whose STATUS.json said only
    `backlog_confirmed_seed`.

This file replaces the hand-read. Layer 2 of the schema's three-layer
enforcement: heartbeat reads THIS, not 15 directories.

Design constraint that matters
------------------------------
It is **read-only and additive**. It does not rewrite STATUS.json, because those
files carry irreplaceable prose (A03's ~900-word status paragraph records a
seed45 NOT-CONFIRM that exists nowhere else). It infers `lifecycle` from the
fields that are actually present, and reports what is missing instead of
inventing it.

Field-name reality (censused 2026-08-14 across all 15 STATUS.json)
------------------------------------------------------------------
`status` has 14 distinct values across 15 files and is NOT machine-readable.
The backfill wrote per-proposal variants, not one schema:
  next_gate | next_gate_executable_20260814 | next_gate_gpu |
  next_gate_candidate_not_yet_adopted | next_gate_blocked_by_portfolio_shape
  gpu_cost_estimate (dict with value+basis, OR a bare string)
  novelty_checked (bool) | novelty_status | novelty_status_detail | k1_novelty
  kill_gate (dict OR the sentinel string NO_KILL_GATE_DEFINED) | kill_gates |
  kill_criteria
  gpu_policy | blocking_dependency | blocked_by | required_before_stage0
So this reader accepts ALL of those spellings. Normalising the files instead
would mean rewriting prose-bearing JSON, which is the one thing the schema
forbids.

The load-bearing rule this encodes
----------------------------------
`proposal/README.md`: **Related Work must be written BEFORE new GPU is spent.**
Five backlog proposals (B01/B05/B06/B07/B08) have `novelty_checked: false` and
no `RELATED_WORK.md` on disk. They are therefore NOT `ready_gpu` no matter how
well-specified their gate is — they are `ready_cpu`, and the CPU task is the
related-work write-up. That is the difference between "idle because blocked" and
"idle because nobody dispatched the free work", which is the whole point.

Usage:
  python proposal/ready_queue.py                 # human table
  python proposal/ready_queue.py --json          # machine-readable
  python proposal/ready_queue.py --strict        # exit 1 if any file is unreadable
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# sm_100 = B200 (LOCAL/.212); sm_90 = H20 (.73/.82/.104). Same-harness
# reproduction must stay on one arch or stack drift confounds with hardware
# drift -- 2026-08-14 I "reserved" 16 sm_90 cards for an sm_100-only task.
ARCH_NODES = {"sm_100": ["LOCAL", ".212"], "sm_90": [".73", ".82", ".104"]}

NEXT_GATE_KEYS = [
    "next_gate_executable_20260814",   # backfill's operationalised version wins
    "next_gate",
    "next_gate_gpu",
]
NEXT_GATE_WEAK = [                     # present but explicitly NOT adopted
    "next_gate_candidate_not_yet_adopted",
    "next_gate_blocked_by_portfolio_shape",
]
KILL_KEYS = ["kill_gate", "kill_gates", "kill_criteria", "kill_gate_verbatim"]
NOVELTY_BOOL = ["novelty_checked"]
NOVELTY_OTHER = ["novelty_status", "novelty_status_detail", "k1_novelty",
                 "novelty_check_2026_08_09", "novelty_verdict"]
# Keys whose value is a completed novelty ADJUDICATION, not a narrative note.
# Distinguishing these was a real bug: the first version of this file reported
# A04 / B04 / B10 as "novelty never checked" when all three carry a finished
# verdict. A scheduler that UNDER-reports readiness recreates the stall it
# exists to prevent, so these are read for their verdict rather than ignored.
NOVELTY_VERDICT_KEYS = ["novelty_check_2026_08_09", "novelty_verdict",
                        "k1_novelty", "related_work_status"]
# A finished check can legitimately conclude "not yet promotable". That is a
# CLEARED gate with a hold, not an unrun gate -- the distinction determines
# whether the actionable task is "write related work" or "satisfy the hold".
VERDICT_CLEARED = ("hold_in_backlog", "gate cleared", "clear", "pass",
                   "audited", "no candidate preempts", "not preempted")
VERDICT_PENDING = ("needs_narrowing", "unchecked", "not_checked", "todo")
BLOCK_KEYS = ["blocking_dependency", "blocked_by", "required_before_stage0",
              "gpu_policy", "premise_falsified"]

UNSPEC = ("NOT_SPECIFIED", "UNKNOWN", "NO_KILL_GATE_DEFINED",
          "NO_KILL_GATE_BY_DESIGN")


def _txt(v, n=400):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n - 1] + "…"


def _first(d, keys):
    for k in keys:
        if k in d:
            return k, d[k]
    return None, None


def _is_unspec(v):
    return isinstance(v, str) and any(v.startswith(u) for u in UNSPEC)


def read_one(path):
    """Parse one STATUS.json into a scheduling record. Never raises on content."""
    name = os.path.basename(os.path.dirname(path))
    rec = {"id": name, "path": path, "problems": []}
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        rec["lifecycle"] = "UNREADABLE"
        rec["problems"].append(f"parse failure: {e}")
        return rec

    rec["status_prose"] = _txt(d.get("status", ""), 120)
    rec["n_keys"] = len(d)

    gk, gate = _first(d, NEXT_GATE_KEYS)
    rec["next_gate_key"] = gk
    rec["next_gate"] = _txt(gate) if gate is not None else None
    if gate is None:
        wk, w = _first(d, NEXT_GATE_WEAK)
        if wk:
            rec["problems"].append(f"only a non-adopted gate ({wk})")
    gate_ok = gate is not None and not _is_unspec(gate)
    if gate is not None and _is_unspec(gate):
        rec["problems"].append("next_gate is explicitly NOT_SPECIFIED")
    if gate is None and not rec.get("next_gate_key"):
        rec["problems"].append("no next_gate field at all")

    ck, kill = _first(d, KILL_KEYS)
    rec["kill_gate_key"] = ck
    kill_ok = kill is not None and not _is_unspec(kill)
    if kill is None:
        rec["problems"].append("no kill_gate field (README: kill gate before GPU)")
    elif _is_unspec(kill):
        rec["problems"].append(f"kill gate undefined ({_txt(kill, 60)})")

    nk, nov = _first(d, NOVELTY_BOOL)
    if nk is not None and bool(nov):
        rec["novelty_checked"] = True
        rec["novelty_evidence"] = "novelty_checked=true"
    else:
        # No boolean, or an explicit false. Before concluding "never checked",
        # look for a completed adjudication under any of the spellings actually
        # used in this repo, and read its verdict.
        rec["novelty_checked"] = False
        rec["novelty_evidence"] = "absent"
        for k in NOVELTY_VERDICT_KEYS:
            if k not in d:
                continue
            v = d[k]
            verdict = ""
            if isinstance(v, dict):
                verdict = str(v.get("verdict") or v.get("status") or "")
            elif isinstance(v, str):
                verdict = v
            vl = verdict.lower()
            if any(s in vl for s in VERDICT_PENDING):
                rec["novelty_evidence"] = f"{k}.verdict={_txt(verdict, 80)} (PENDING)"
                break
            if any(s in vl for s in VERDICT_CLEARED):
                rec["novelty_checked"] = True
                rec["novelty_evidence"] = f"{k}.verdict={_txt(verdict, 80)}"
                break
            if verdict:
                rec["novelty_evidence"] = f"{k}.verdict={_txt(verdict, 80)} (UNPARSED)"
                break
    rel = os.path.join(os.path.dirname(path), "RELATED_WORK.md")
    rec["related_work_md"] = os.path.exists(rel)

    ck2, cost = _first(d, ["gpu_cost_estimate", "cost", "cost_to_first_result"])
    if isinstance(cost, dict):
        rec["gpu_cost"] = _txt(cost.get("value", cost), 160)
        rec["gpu_cost_has_basis"] = bool(cost.get("basis") or cost.get("why"))
    else:
        rec["gpu_cost"] = _txt(cost, 160) if cost is not None else None
        rec["gpu_cost_has_basis"] = False
    if cost is None:
        rec["problems"].append("no gpu_cost_estimate")

    bk, blk = _first(d, BLOCK_KEYS)
    rec["blocker_key"] = bk
    rec["blocker"] = _txt(blk, 300) if blk is not None else None

    st = str(d.get("status", "")).lower()
    promoted = ("promoted" in st) or ("promoted_to" in d)
    dead = st.startswith("dead") or "archive" in path

    # ---- an EXPLICIT lifecycle overrides inference ---------------------------
    # LIFECYCLE_SCHEMA.md sec 1 calls `lifecycle` "唯一的机器可读状态", but the
    # first version of this reader never read it: it inferred lifecycle purely
    # from which OTHER fields were present. That is exactly backwards for a
    # terminal state. Measured 2026-08-14: B02, whose own pre-registered kill
    # gate had just FIRED (Delta_excess negative at both lengths, p=0.0008) and
    # which therefore recorded `"lifecycle": "dead"`, was classified
    # **ready_gpu** -- because writing down the kill gate and the novelty
    # verdict satisfied the three presence checks below. Filling in a
    # proposal's paperwork made a killed direction look like the single most
    # dispatchable item in the queue, and it would have been handed 8 idle H20s.
    # A terminal state must be declarable, not only inferable.
    explicit = d.get("lifecycle")
    VALID_LC = ("ready_gpu", "ready_cpu", "needs_prior_gate", "running",
                "promoted", "dead")
    if isinstance(explicit, str) and explicit in VALID_LC:
        rec["lifecycle_declared"] = explicit
        if explicit in ("dead", "promoted", "running"):
            # Terminal/in-flight states are authoritative: no amount of
            # well-formed paperwork should re-open them.
            rec["lifecycle"] = explicit
            rec["lifecycle_reason"] = (
                f"DECLARED lifecycle={explicit} in STATUS.json (authoritative; "
                + _txt(d.get("lifecycle_reason", "no reason field"), 200) + ")")
            if explicit == "running" and not d.get("running_on"):
                rec["problems"].append(
                    "lifecycle=running without running_on (schema sec 1)")
            rec["needs_arch"] = d.get("needs_arch", "UNRECORDED")
            return rec
        if explicit == "ready_cpu":
            # A DECLARED ready_cpu is also authoritative, and in the *safe*
            # direction: it says "my next step needs no card." Inference cannot
            # reach this conclusion, because whether the next gate costs GPU is
            # a property of the gate's TEXT, not of which fields exist. B11 is
            # the case in point: its gates are all specified and its novelty is
            # adjudicated, so inference says ready_gpu -- but its own next_gate
            # is "file the upstream bug report", explicitly 0 GPU, and its
            # measured cost record says a card would buy ~1 GPU-h of provably
            # low-information replication. Down-grading GPU->CPU on the owner's
            # say-so can only ever free a card, never waste one, so it is
            # honoured without further checks.
            rec["lifecycle"] = "ready_cpu"
            rec["lifecycle_reason"] = (
                "DECLARED lifecycle=ready_cpu in STATUS.json (authoritative; "
                + _txt(d.get("lifecycle_reason", "no reason field"), 260) + ")")
            rec["needs_arch"] = d.get("needs_arch", "UNRECORDED")
            return rec
        # A declared ready_gpu is NOT taken on faith: it still has to pass the
        # presence checks below, so that "I promise I'm ready" cannot skip the
        # kill-gate / novelty requirements that the README makes prerequisites
        # for spending GPU. Declaration can downgrade, not upgrade.
    elif explicit is not None:
        rec["problems"].append(
            f"lifecycle={_txt(explicit, 40)!r} is not one of {VALID_LC}")

    # ---- lifecycle inference -------------------------------------------------
    # The novelty gate is satisfied by an ADJUDICATED verdict (in STATUS.json)
    # or novelty_checked=true. A missing RELATED_WORK.md is recorded as a
    # PROMOTION blocker, not automatically a GPU blocker -- except where the
    # proposal's own record says otherwise (B01's novelty_status_detail cites
    # README's gate as "BEFORE new GPU"). Conflating the two is what made the
    # first version of this file report 13/13 as CPU-only, which is as useless
    # to a scheduler as reporting all of them ready.
    rw_ok = rec["novelty_checked"]
    if not rec["related_work_md"]:
        rec["problems"].append(
            "RELATED_WORK.md absent (blocks PROMOTION; 0-GPU task)")
    if promoted:
        lc, why = "promoted", "status/promoted_to says promoted"
    elif dead:
        lc, why = "dead", "archived / status dead"
    elif not rw_ok:
        lc = "ready_cpu"
        why = ("novelty gate not adjudicated (" + rec["novelty_evidence"] +
               ") -> the actionable task is 0 GPU: run it")
    elif not kill_ok:
        lc = "ready_cpu"
        why = "kill gate undefined -> writing it is 0 GPU and blocking"
    elif not gate_ok:
        lc = "ready_cpu"
        why = "next_gate not operationalised -> writing it is 0 GPU and blocking"
    else:
        lc = "ready_gpu"
        why = ("gate + kill gate + adjudicated novelty all present (" +
               rec["novelty_evidence"] + ")")
    rec["lifecycle"] = lc
    rec["lifecycle_reason"] = why

    # needs_arch is not recorded anywhere yet; say so rather than guess.
    rec["needs_arch"] = d.get("needs_arch", "UNRECORDED")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(ROOT, "active", "*", "STATUS.json")) +
                   glob.glob(os.path.join(ROOT, "backlog", "*", "STATUS.json")))
    recs = [read_one(p) for p in paths]

    # A proposal directory with NO STATUS.json is invisible to the scheduler,
    # which is its own failure mode -- report it.
    for d in sorted(glob.glob(os.path.join(ROOT, "active", "*")) +
                    glob.glob(os.path.join(ROOT, "backlog", "*"))):
        if os.path.isdir(d) and not os.path.exists(os.path.join(d, "STATUS.json")):
            recs.append({"id": os.path.basename(d), "path": d,
                         "lifecycle": "NO_STATUS_JSON", "problems":
                         ["no STATUS.json -> invisible to the scheduler"]})

    order = {"ready_gpu": 0, "ready_cpu": 1, "NO_STATUS_JSON": 2,
             "UNREADABLE": 3, "promoted": 4, "dead": 5}
    recs.sort(key=lambda r: (order.get(r["lifecycle"], 9), r["id"]))

    if a.json:
        print(json.dumps({"generated_by": "proposal/ready_queue.py",
                          "n": len(recs), "queue": recs}, indent=1))
    else:
        buckets = {}
        for r in recs:
            buckets.setdefault(r["lifecycle"], []).append(r)
        for lc in ("ready_gpu", "ready_cpu", "NO_STATUS_JSON", "UNREADABLE",
                   "promoted", "dead"):
            if lc not in buckets:
                continue
            print(f"\n=== {lc}  ({len(buckets[lc])}) ===")
            for r in buckets[lc]:
                print(f"  {r['id']}")
                print(f"     why: {r.get('lifecycle_reason', '-')}")
                if r.get("next_gate"):
                    print(f"     gate[{r['next_gate_key']}]: {_txt(r['next_gate'], 220)}")
                if r.get("gpu_cost"):
                    print(f"     cost: {_txt(r['gpu_cost'], 130)}"
                          f"{'' if r.get('gpu_cost_has_basis') else '  (NO BASIS)'}")
                if r.get("blocker"):
                    print(f"     blocker[{r['blocker_key']}]: {_txt(r['blocker'], 180)}")
                for p in r["problems"]:
                    print(f"     ! {p}")
        ng = len(buckets.get("ready_gpu", []))
        nc = len(buckets.get("ready_cpu", []))
        print(f"\nSUMMARY: {ng} ready_gpu, {nc} ready_cpu (0 GPU, dispatchable NOW)")
        if ng == 0 and nc > 0:
            print("  => An idle GPU is NOT a reason to idle: there are "
                  f"{nc} zero-GPU tasks that are blocking their own gates.")

    if a.strict and any(r["lifecycle"] in ("UNREADABLE", "NO_STATUS_JSON")
                        for r in recs):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
