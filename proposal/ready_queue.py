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

Two reader defects that cancelled each other (fixed 2026-08-15)
---------------------------------------------------------------
Both were *under*-reads, and they were only harmless as a pair:

1. **Nested gates were invisible.** `KILL_KEYS` was matched against the top
   level of the document only. A04's kill gate is a fully pre-registered
   three-clause condition with frozen constants, written 2026-08-09 for 0 GPU —
   and it lives at `gate_design.kill_condition_verbatim`. So the reader printed
   "no kill_gate field" and told the next agent the blocking 0-GPU task was to
   *write* a gate that had existed for six days. Fix: `_first_nested`, an
   EXPLICIT one-level container allow-list (`NESTED_GATE_CONTAINERS`), not a
   recursive walk — a recursive `*kill*` search over the live files returns 40+
   paths (`closest_prior_art[0].kills`, `history_20260808.next_gate_then`, …) of
   which exactly one is a gate, and promoting prose to gate-hood is the
   paperwork-counts-as-readiness bug this file exists to stop. The recorded key
   is the dotted path, so the report shows where it was read from.

2. **`BLOCK_KEYS` was display-only.** It was parsed into `rec["blocker"]` and
   then ignored by lifecycle inference. Fixing (1) alone would therefore have
   made A04 the queue's single `ready_gpu` item — while its own
   `blocked_by.still_blocking_before_any_gate_gpu` reads "USER APPROVAL for GPU.
   The full gate is 1,077-4,309 GPU-h; nothing beyond Pilot Zero may be launched
   without it." Fix: `_live_blockers` reads blockers for CONTENT and an
   un-discharged one forces `needs_prior_gate`. Values that state their own
   closure ("CLEARED 2026-08-09 … NO LONGER BLOCKING.") do not count, or the
   reader would hold proposals out citing gates that already passed.

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
KILL_KEYS = ["kill_gate", "kill_gates", "kill_criteria", "kill_gate_verbatim",
             # A04 spells it `kill_condition_verbatim` and files it one level
             # down under `gate_design`. Measured 2026-08-15: the reader
             # therefore printed "no kill_gate field" for the ONE proposal in
             # the repo whose kill gate is a fully pre-registered three-clause
             # condition with frozen constants (T_plateau=2.0%/5k, rho=0.85,
             # delta=0.10*residual(intact), a frozen checkpoint grid), written
             # 2026-08-09 for 0 GPU. It then told the next agent the actionable
             # 0-GPU task was "write the kill gate" -- i.e. rewrite a document
             # that already existed. A reader that cannot see a gate manufactures
             # busywork just as surely as one that hallucinates a gate spends a card.
             "kill_condition_verbatim", "kill_condition"]
# Containers searched ONE level down for the gate key names above.
# Deliberately an EXPLICIT allow-list and NOT a recursive walk of the document:
# a recursive search would promote any prose that happens to contain a key like
# `closest_prior_art[0].kills` or `history_20260808.next_gate_then` into "this
# proposal has a gate", which is the paperwork-counts-as-readiness failure this
# whole file exists to block. Measured on the 15 live STATUS.json: a recursive
# search finds 40+ `*kill*`/`*next_gate*` paths, of which exactly one
# (gate_design.kill_condition_verbatim) is an actual gate.
NESTED_GATE_CONTAINERS = ("gate_design", "gates", "prereg", "preregistration")
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
# A blocker value that says it has been discharged is NOT a blocker. A04's
# `blocked_by.related_work_gate` is literally
#   "CLEARED 2026-08-09 (see related_work_status). NO LONGER BLOCKING."
# so a reader that counts every BLOCK_KEYS hit as live would hold A04 out of the
# queue citing a gate that was cleared six days earlier -- the same
# under-reporting stall, just with a more convincing excuse.
#
# The phrases are deliberately multi-word. A bare "done" was tried and rejected:
# "abandoned" CONTAINS "done", so a blocker reading "that arm was abandoned"
# would have been silently scored as discharged -- a substring match that fires
# inside an unrelated word turns the blocker check into a random amnesty.
BLOCKER_DISCHARGED = ("no longer blocking", "no longer blocks", "not blocking",
                      "not outstanding", "cleared", "discharged", "resolved",
                      "已解除", "已清除")
# "cleared" / "resolved" / "discharged" survive as bare tokens because the repo's
# own usage is the sentence-initial "CLEARED <date>" form. Their negations are
# the obvious hazard ("not cleared" contains "cleared"), so they are enumerated
# here and tested FIRST in _is_discharged.
BLOCKER_NOT_DISCHARGED = ("not cleared", "uncleared", "not yet cleared",
                          "not resolved", "unresolved", "not discharged",
                          "still blocking", "still outstanding")

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


def _first_nested(d, keys, containers=NESTED_GATE_CONTAINERS):
    """Top level first, then ONE level inside an explicit container allow-list.

    Returns (dotted_path, value) so the report shows WHERE the gate was read
    from -- `gate_design.kill_condition_verbatim` is traceable, a bare
    `kill_condition_verbatim` would not be.
    """
    k, v = _first(d, keys)
    if k is not None:
        return k, v
    for c in containers:
        sub = d.get(c)
        if not isinstance(sub, dict):
            continue
        k, v = _first(sub, keys)
        if k is not None:
            return f"{c}.{k}", v
    return None, None


def _is_discharged(v):
    """True if this blocker value states, in its own words, that it is closed.

    Negations are checked FIRST: "not cleared" contains "cleared", and reading
    that as a discharge would let a proposal claim readiness with the exact
    sentence that denies it.
    """
    if not isinstance(v, str):
        return False
    s = v.lower()
    if any(n in s for n in BLOCKER_NOT_DISCHARGED):
        return False
    return any(p in s for p in BLOCKER_DISCHARGED)


def _live_blockers(d):
    """Enumerate blockers that are still blocking, with a citable path each.

    Read for CONTENT, not merely for presence. Before 2026-08-15 BLOCK_KEYS was
    parsed into rec["blocker"] for *display only* and lifecycle inference never
    consulted it, so A04 -- whose own record says
    `still_blocking_before_any_gate_gpu` contains a PROPOSAL.md narrowing, a
    BLOCKING CPU code fix, and "USER APPROVAL for GPU. The full gate is
    1,077-4,309 GPU-h" -- would have been promoted straight to ready_gpu the
    moment its (already-written) kill gate became visible. Making the gate
    visible without making the blocker binding converts one reporting bug into
    a four-thousand-GPU-hour dispatch.
    """
    out = []
    for k in BLOCK_KEYS:
        if k not in d:
            continue
        v = d[k]
        if isinstance(v, dict):
            for sk, sv in v.items():
                if _is_discharged(sv):
                    continue
                if isinstance(sv, list):
                    # Enumerate each clause separately rather than collapsing to
                    # "[+2 more]": A04's third clause is "USER APPROVAL for GPU.
                    # The full gate is 1,077-4,309 GPU-h" -- the single most
                    # important line in the record, and the one a truncated
                    # summary would hide.
                    for i, item in enumerate(sv):
                        if not _is_discharged(item):
                            out.append((f"{k}.{sk}[{i}]", _txt(item, 200)))
                elif isinstance(sv, str) and sv.strip():
                    out.append((f"{k}.{sk}", _txt(sv, 200)))
                elif isinstance(sv, dict) and sv:
                    out.append((f"{k}.{sk}", _txt(sv, 200)))
        elif isinstance(v, list):
            for i, sv in enumerate(v):
                if not _is_discharged(sv):
                    out.append((f"{k}[{i}]", _txt(sv, 200)))
        elif isinstance(v, str) and v.strip() and not _is_discharged(v):
            out.append((k, _txt(v, 200)))
    return out


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

    gk, gate = _first_nested(d, NEXT_GATE_KEYS)
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

    ck, kill = _first_nested(d, KILL_KEYS)
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
    # ...and, separately from that display string, the blockers that are still
    # LIVE. This list is load-bearing for lifecycle inference below.
    live = _live_blockers(d)
    rec["live_blockers"] = [{"path": p, "text": t} for p, t in live]

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
    elif live:
        # ---- the un-discharged-blocker precondition (added 2026-08-15) --------
        # Everything above this line asks "is the paperwork present?". Nothing
        # asked "does this proposal's own record say it may not be launched
        # yet?" -- BLOCK_KEYS was parsed for display and then dropped on the
        # floor. That was survivable only because the nested-kill-gate bug was
        # ALSO present: A04 failed the kill-gate presence check, so it never
        # reached this branch. Fixing the reader's blindness to nested gates
        # WITHOUT this branch would have made A04 the queue's single ready_gpu
        # item while its own blocked_by said "USER APPROVAL for GPU. The full
        # gate is 1,077-4,309 GPU-h; nothing beyond Pilot Zero may be launched
        # without it." Two under-reads that cancelled each other; removing one
        # is strictly worse than removing neither.
        #
        # `needs_prior_gate` and not a new `blocked` value: LIFECYCLE_SCHEMA.md
        # sec 1 bans the word `blocked` outright ("它在 2026-08-14 把三件不同的事
        # 混成了一个字符串"), and sec 1.1 already gives needs_prior_gate the exact
        # semantics wanted here -- something ahead of the gate must close first,
        # and whether that something costs a card is reported separately.
        lc = "needs_prior_gate"
        why = ("gate + kill gate + novelty all OK, but " + str(len(live)) +
               " un-discharged blocker(s) in its own record: " +
               "; ".join(f"{p} = {t}" for p, t in live[:3])[:600] +
               (" ..." if len(live) > 3 else ""))
        for p, t in live:
            rec["problems"].append(f"blocker STILL LIVE [{p}]: {_txt(t, 150)}")
    else:
        lc = "ready_gpu"
        why = ("gate + kill gate + adjudicated novelty all present (" +
               rec["novelty_evidence"] + "), no un-discharged blocker")
        if rec.get("lifecycle_declared") == "needs_prior_gate":
            # Same "declaration can downgrade, not upgrade" rule the ready_cpu
            # branch above already applies. Whether a prior gate is closed is a
            # property of that gate's outcome, which no presence check can see;
            # inferring ready_gpu over the owner's explicit needs_prior_gate
            # would spend a card on the owner's stated say-so that it is too
            # early. Costs at most a delay; the opposite error costs GPU-h.
            pgg = d.get("prior_gate_needs_gpu", "UNRECORDED")
            # LIFECYCLE_SCHEMA.md sec 1.1: a prior gate that costs no card must
            # be surfaced in ready_cpu, not parked. That bool is the direct fix
            # for the B11 stall (a 0-GPU blocking gate nobody dispatched).
            lc = "ready_cpu" if pgg is False else "needs_prior_gate"
            why = ("DECLARED lifecycle=needs_prior_gate (honoured over an "
                   "inferred ready_gpu; prior_gate_needs_gpu=" + repr(pgg) +
                   (" -> schema sec 1.1 folds a 0-GPU prior gate into ready_cpu"
                    if pgg is False else "") + "): " +
                   _txt(d.get("prior_gate", d.get("lifecycle_reason", "-")), 260))
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

    # needs_prior_gate sorts BELOW ready_cpu: schema sec 1.1 already folds the
    # 0-GPU prior gates into ready_cpu, so what is left here genuinely cannot be
    # dispatched yet. It must still be PRINTED -- a bucket that exists in the
    # inference but not in this list is a silently dropped proposal, which is
    # exactly how B06 vanished (schema sec 2: "不许静默略过").
    order = {"ready_gpu": 0, "ready_cpu": 1, "needs_prior_gate": 2,
             "NO_STATUS_JSON": 3, "UNREADABLE": 4, "promoted": 5, "dead": 6}
    recs.sort(key=lambda r: (order.get(r["lifecycle"], 9), r["id"]))

    # Fail loudly if inference can emit a bucket main() would not print.
    emitted = {r["lifecycle"] for r in recs}
    unprintable = emitted - set(order)
    if unprintable:
        print(f"BUG: lifecycle value(s) {sorted(unprintable)} have no print "
              f"bucket -> those proposals would be invisible", file=sys.stderr)

    if a.json:
        print(json.dumps({"generated_by": "proposal/ready_queue.py",
                          "n": len(recs), "queue": recs}, indent=1))
    else:
        buckets = {}
        for r in recs:
            buckets.setdefault(r["lifecycle"], []).append(r)
        for lc in ("ready_gpu", "ready_cpu", "needs_prior_gate",
                   "NO_STATUS_JSON", "UNREADABLE", "promoted", "dead"):
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
        nb = len(buckets.get("needs_prior_gate", []))
        print(f"\nSUMMARY: {ng} ready_gpu, {nc} ready_cpu (0 GPU, dispatchable "
              f"NOW), {nb} needs_prior_gate")
        if ng == 0 and nc > 0:
            print("  => An idle GPU is NOT a reason to idle: there are "
                  f"{nc} zero-GPU tasks that are blocking their own gates.")

    if a.strict and any(r["lifecycle"] in ("UNREADABLE", "NO_STATUS_JSON")
                        for r in recs):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
