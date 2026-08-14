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

Three more, all the same shape (fixed 2026-08-15, later the same day)
--------------------------------------------------------------------
Each is "the answer was on disk and this reader's key list did not reach it",
and each is a direct consequence of append-only (LIFECYCLE_SCHEMA.md sec 0):
a record that must CHANGE can only be corrected by appending a SIBLING key, so
a reader that resolves the OLDEST spelling first is pinned to the WORST version.

3. **`KILL_KEYS` had no dated-priority slot** although `NEXT_GATE_KEYS[0]` did.
   B07 wrote a four-clause pre-registered gate as `kill_gate_executable_20260814`
   and could not overwrite the earlier honest `"NO_KILL_GATE_DEFINED"`, so the
   reader kept reporting B07 as gate-less. **B07's own `_precedence_warning`
   predicted this exact bug**, named the one-line fix, and it sat unapplied.
   Fix: dated keys first, newest first, EXPLICIT (see KILL_KEYS).

4. **A discharged blocker could not be expressed.** Same cause: A04's items
   [0]/[1] were closed on 2026-08-13 in the sibling key
   `blockers_discharged_20260813`, but the original strings may not be edited, so
   the reader reported 3 live blockers where 1 was live — right verdict (A04 may
   not take a card), reason overstated by two, and the next agent would be sent
   to redo a PROPOSAL.md narrowing and a sampler fix that landed as `ce5c298`
   four days before it was requested. Fix: `discharges` POINTERS at exact dotted
   paths (LIFECYCLE_SCHEMA.md sec 2.1), fail-closed by omission.

5. **A DECLARED lifecycle could delete a disk-fact warning.** The missing
   RELATED_WORK.md problem was raised only on the inference path, which the
   early return for a declared lifecycle skips — so adding `"lifecycle":
   "ready_cpu"` to a STATUS.json silently removed a standing promotion blocker
   from the report (measured on B02/B04/B07/B11, all four genuinely lacking the
   file). Paperwork must not be able to retire an unmet requirement by changing
   one field. Fix: raise it where the file is stat()ed.

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
KILL_KEYS = [
             # ---- dated-priority slots: NEWEST FIRST, exactly mirroring
             # NEXT_GATE_KEYS[0]="next_gate_executable_20260814". The asymmetry
             # (next_gate had a dated slot, kill_gate did not) was PREDICTED by
             # the very file it broke: B07's
             # kill_gate_executable_20260814._precedence_warning says
             #   "KILL_KEYS ... has NO dated-priority slot (unlike
             #    NEXT_GATE_KEYS[0]=...). STATUS.json is append-only per
             #    LIFECYCLE_SCHEMA.md section 0, so the older honest
             #    'NO_KILL_GATE_DEFINED' string above CANNOT be overwritten --
             #    therefore this key will NOT be picked up until KILL_KEYS gains
             #    a matching entry."
             # That is the whole mechanism: append-only means a later, BETTER
             # gate can only be added ALONGSIDE the earlier honest sentinel, so
             # a reader that resolves `kill_gate` first is permanently pinned to
             # the worst version of the record. B07 wrote a four-clause
             # pre-registered gate (K1 123.2 ms retention / K2 deployability
             # floor / K3 tiering headroom / K4 edit leg) on 2026-08-14 and the
             # reader kept reporting NO_KILL_GATE_DEFINED.
             #
             # MAINTENANCE RULE: a new dated key goes ABOVE the existing ones,
             # and is added EXPLICITLY. Do NOT replace this list with a
             # glob/regex over `kill_gate_*` -- same reason `_first_nested` is an
             # allow-list and not a recursive walk: `kill_gate_verdict` (B02) is
             # an OUTCOME, `updated_20260814_kill_gate_pass` (B05) is a
             # changelog line, and `original_kill_reassessment` (B10) is prose.
             # A pattern match would promote all three to gate-hood, which is
             # the paperwork-counts-as-readiness bug this file exists to stop.
             "kill_gate_executable_20260814",
             "kill_gate", "kill_gates", "kill_criteria", "kill_gate_verbatim",
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
# ---- cross-reference discharge (LIFECYCLE_SCHEMA.md sec 2.1) -----------------
# The text-based `_is_discharged` above can only close a blocker whose OWN string
# says it is closed. Under append-only (schema sec 0) that is often impossible:
# A04's `blocked_by.still_blocking_before_any_gate_gpu` is a 3-item list whose
# items [0] and [1] were discharged on 2026-08-13 -- item [1] because the
# sampler fix it demands had ALREADY landed four days earlier (ce5c298;
# verified on disk 2026-08-15 at scripts/train_olmo2_arch_probe2.py:869 =
# `DistributedSampler(ds, shuffle=True, seed=args.seed)`) -- and the closure is
# recorded in a SIBLING key, `blockers_discharged_20260813`, because the
# original strings may not be edited. A reader that only looks inside each
# blocker string therefore reports 3 live blockers when 1 is live: the verdict
# (A04 may not take a card) is right, the stated reason overstates by two, and
# the next agent is sent to redo work already done.
#
# So a discharge may be declared by POINTER: any record may carry
#   "discharges": ["blocked_by.still_blocking_before_any_gate_gpu[0]", ...]
# listing EXACT dotted blocker paths -- the same paths this file prints.
#
# Properties that make this safe, and why it is a pointer rather than a
# `blocked_by_v2` restatement:
#   * FAIL-CLOSED BY OMISSION. Only a path explicitly named is discharged. A
#     `blocked_by_v2` list would discharge by SILENCE, so a shorter v2 could
#     retire "USER APPROVAL for GPU. The full gate is 1,077-4,309 GPU-h" by
#     simply not mentioning it -- an unauditable amnesty. Precedence-by-recency
#     is safe for GATES (a newer gate still has to pass _is_unspec) and unsafe
#     for BLOCKERS, where newer-and-shorter means fewer constraints.
#   * SAME NAMESPACE AS THE REPORT, so it is self-checking: you discharge a
#     blocker by copying the path this tool printed, and a typo matches nothing
#     and is reported as dangling instead of silently closing the wrong clause.
#   * CO-LOCATED WITH EVIDENCE. The pointer lives inside the record that
#     documents the discharge, so following it lands on the md5/commit/verbatim
#     proof rather than on a bare assertion in a second list.
DISCHARGE_POINTER_KEY = "discharges"

UNSPEC = ("NOT_SPECIFIED", "UNKNOWN", "NO_KILL_GATE_DEFINED",
          "NO_KILL_GATE_BY_DESIGN")

# Why a DECLARED lifecycle needs its own spelling list. Censused 2026-08-15:
# B02/B04/B05/B11 spell the justification `lifecycle_reason`, but B03 and B07
# use `lifecycle_why_20260815` / `lifecycle_why_20260814` -- so the reader
# printed "DECLARED lifecycle=ready_cpu (authoritative; no reason field)" for
# B07 while a 400-word justification sat in the file naming BOTH of its 0-GPU
# blockers. Same class of defect as the missing dated kill-gate slot: the
# information was on disk and the reader's key list did not reach it, and the
# visible symptom was a proposal that looked emptier than it is. Explicit list,
# newest-dated first, for the reason given at KILL_KEYS.
LIFECYCLE_REASON_KEYS = ["lifecycle_reason",
                         "lifecycle_why_20260815", "lifecycle_why_20260814",
                         "lifecycle_why"]


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


def _discharge_pointers(d):
    """Collect every explicitly-pointed-at blocker path in the document.

    Searched at the top level and ONE level down -- the same depth discipline as
    `_first_nested`, and for the same reason: the discharge record is a sibling
    key of the blocker (A04's `blockers_discharged_20260813`), never buried
    arbitrarily deep, and a recursive walk would start honouring the word
    "discharges" wherever prose happens to use it.

    Returns (set_of_paths, list_of_(container, path)) so the report can name WHO
    discharged WHAT, and so a pointer matching no blocker can be flagged as
    dangling rather than disappearing.
    """
    paths, claims = set(), []

    def _take(container, v):
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return
        for item in v:
            if isinstance(item, str) and item.strip():
                paths.add(item.strip())
                claims.append((container, item.strip()))

    if DISCHARGE_POINTER_KEY in d:
        _take(DISCHARGE_POINTER_KEY, d[DISCHARGE_POINTER_KEY])
    for k, v in d.items():
        if isinstance(v, dict) and DISCHARGE_POINTER_KEY in v:
            _take(k, v[DISCHARGE_POINTER_KEY])
    return paths, claims


def _walk_blockers(d):
    """Yield (dotted_path, raw_value) for every blocker clause in the document.

    ONE traversal with TWO consumers, deliberately: `_live_blockers` filters it
    by discharge state, and the dangling-pointer check needs the SAME namespace
    unfiltered. Deriving the pointer target namespace from a second, separate
    walk is how a pointer ends up "dangling" against a path that does exist.

    Container paths (`blocked_by.still_blocking_before_any_gate_gpu`) are yielded
    alongside their indexed clauses, so a pointer may discharge either one item
    or the whole list.
    """
    for k in BLOCK_KEYS:
        if k not in d:
            continue
        v = d[k]
        if isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, list):
                    yield f"{k}.{sk}", sv          # the container itself
                    for i, item in enumerate(sv):
                        yield f"{k}.{sk}[{i}]", item
                elif isinstance(sv, str) and sv.strip():
                    yield f"{k}.{sk}", sv
                elif isinstance(sv, dict) and sv:
                    yield f"{k}.{sk}", sv
        elif isinstance(v, list):
            yield k, v
            for i, sv in enumerate(v):
                yield f"{k}[{i}]", sv
        elif isinstance(v, str) and v.strip():
            yield k, v


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

    Two independent ways a blocker can be closed:
      1. its own text says so (`_is_discharged`);
      2. some record in the same document POINTS at its exact dotted path
         (`_discharge_pointers`) -- required because append-only forbids editing
         (1) into the original string.
    Discharge by omission is deliberately NOT a third way: see
    DISCHARGE_POINTER_KEY's comment.
    """
    discharged_paths, _ = _discharge_pointers(d)
    out = []
    for path, value in _walk_blockers(d):
        if isinstance(value, list):
            # The container line is a grouping handle for pointers, not a
            # blocker in its own right -- its clauses are enumerated separately
            # and reporting both would double-count every A04 clause.
            continue
        if path in discharged_paths:
            continue
        # A pointer at the CONTAINER discharges all of its clauses at once, so a
        # wholesale closure need not enumerate every index.
        if path.endswith("]") and path.rsplit("[", 1)[0] in discharged_paths:
            continue
        if _is_discharged(value):
            continue
        out.append((path, _txt(value, 200)))
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
    if not rec["related_work_md"]:
        # Raised HERE, not down in the inference block, because it is a fact
        # about the DISK and is true regardless of lifecycle. Measured
        # 2026-08-15: it used to be raised only on the inference path, so the
        # early return for a DECLARED lifecycle skipped it -- meaning the act of
        # declaring `"lifecycle": "ready_cpu"` in B03's STATUS.json DELETED a
        # standing promotion blocker from the report. Paperwork must never be
        # able to retire an unmet requirement by changing one field.
        rec["problems"].append(
            "RELATED_WORK.md absent (blocks PROMOTION; 0-GPU task)")

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
    # A discharge pointer that matches no blocker path is a silent no-op, and
    # that is the worst outcome for an audit trail: it READS as though the
    # blocker was closed. So check every pointer against the SAME namespace
    # `_live_blockers` walks (unfiltered) and report the ones that hit nothing.
    # Reported, never fatal: a stale pointer to a blocker that has since been
    # removed is untidy, not unsafe.
    all_paths = {p for p, _ in _walk_blockers(d)}
    ptrs, claims = _discharge_pointers(d)
    rec["discharged_blockers"] = sorted(ptrs)
    for container, p in claims:
        if p not in all_paths:
            rec["problems"].append(
                f"discharge pointer [{container}.{DISCHARGE_POINTER_KEY}] "
                f"-> {p!r} matches no blocker path (dangling; no effect)")

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
    _, _lc_why = _first(d, LIFECYCLE_REASON_KEYS)
    _lc_why = _lc_why if _lc_why else "no reason field"
    if isinstance(explicit, str) and explicit in VALID_LC:
        rec["lifecycle_declared"] = explicit
        if explicit in ("dead", "promoted", "running"):
            # Terminal/in-flight states are authoritative: no amount of
            # well-formed paperwork should re-open them.
            rec["lifecycle"] = explicit
            rec["lifecycle_reason"] = (
                f"DECLARED lifecycle={explicit} in STATUS.json (authoritative; "
                + _txt(_lc_why, 200) + ")")
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
                + _txt(_lc_why, 260) + ")")
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
                   _txt(d.get("prior_gate", _lc_why), 260))
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
