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
A proposal with no `RELATED_WORK.md` on disk is NOT `ready_gpu` no matter how
well-specified its gate is — it is `ready_cpu`, and the CPU task is the
related-work write-up. That is the difference between "idle because blocked" and
"idle because nobody dispatched the free work", which is the whole point.

**Presence is read from the DISK at each run** (see `related_work_md` below), not
from any list in this file. That distinction is not pedantic — it is the second
defect this section has carried:

> CORRECTED 2026-08-16. Until today this paragraph read "Five backlog proposals
> (B01/B05/B06/B07/B08) have `novelty_checked: false` and no `RELATED_WORK.md` on
> disk." **All five files are present** — 48124 / 10737 / 28761 / 35965 / 59799 B
> respectively. Of the 18 proposal directories on disk, 15 have one. The three
> that do not are `archive/A03`, `archive/A05` — both in `archive/`, which THIS
> READER DOES NOT SCAN (see the globs in `main`: `active/` + `backlog/` only) —
> and the superseded `backlog/B04-eval-fragility` shell, which has no STATUS.json
> either and so is invisible here too. **In other words: among the proposals this
> tool actually reports on, the count of missing RELATED_WORK.md is ZERO.** The
> sentence was true when written and became false as the files were written, and
> because it sits in the section headed "the load-bearing rule this encodes",
> every reader was handed a false premise about which proposals are blocked and why.
>
> It also propagated. `B07-mutable-comem-serving/STATUS.json` `lifecycle_why_20260814`
> states that "ready_queue.py:46-51 hard-codes B07 among B01/B05/B06/B07/B08 whose
> missing RELATED_WORK.md forbids GPU regardless of gate quality". **There is no
> such hard-coding.** Lines 46-51 are inside this module docstring — verified with
> `ast.get_docstring` — and `grep -nE '"(B01|B05|B06|B07|B08)"'` over the file
> returns nothing. A comment was read as a code path, and a proposal wrote a
> blocker against it. The real gate for those proposals is the novelty VERDICT
> (`NOVELTY_VERDICT_KEYS`), which is a separate question from file presence and is
> genuinely unadjudicated for some of them.
>
> Lesson, and the reason this correction is verbose rather than a silent edit: a
> docstring that names specific proposals will rot, and prose in the section a
> reader trusts most is the worst place for a fact with a shelf life. State the
> RULE here; let the code state the FACTS.

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
import re
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
# 2026-08-17: the two lists below still PIN the date 20260814, so a
# next_gate_executable_20260817 or kill_gate_executable_20260817 would be invisible and the
# reader would silently fall back to the older, worse record. Same defect class as the
# lifecycle / lifecycle_why / cost / novelty enumerations resolved earlier today.
#
# The KILL_KEYS comment below states a MAINTENANCE RULE against replacing the list with a
# glob over `kill_gate_*`, and that rule is CORRECT and is kept: `kill_gate_verdict` is an
# OUTCOME, `updated_20260814_kill_gate_pass` is a CHANGELOG, `original_kill_reassessment` is
# PROSE, and promoting any of them to gate-hood is the paperwork-counts-as-readiness bug.
#
# So this is deliberately NOT a wildcard. It is anchored to the two `*_executable` spellings
# followed by an 8-digit date, which is exactly the dated-priority slot the rule describes,
# and nothing else. VERIFIED before use against all 8 keys the rule protects (including the
# three named counterexamples) plus 4 dated forms that must match: 0 failures.
_DATED_EXECUTABLE_RE = re.compile(
    r"^(?:kill_gate_executable|next_gate_executable)_(20\d{6})(?:_[a-z0-9]+)?$")


def dated_executable_keys(doc, base):
    """Dated `<base>_executable_<date>` keys in THIS doc, newest first.

    `base` is "next_gate" or "kill_gate". Recency wins for the reason LIFECYCLE_SCHEMA.md
    sec 2.0 gives: append-only means a later, better gate can only be ADDED alongside the
    earlier honest sentinel, so resolving the undated key first pins the reader permanently
    to the worst version of the record.
    """
    want = base + "_executable_"
    return sorted((k for k in doc
                   if k.startswith(want) and _DATED_EXECUTABLE_RE.match(k)),
                  reverse=True)
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
# Keys whose value states what the NEXT gate costs. A proposal declaring its own
# next step free must be surfaced as ready_cpu, never ready_gpu -- see the
# `_next_gate_is_free` call in the lifecycle inference. Added 2026-08-15 after
# measuring that pure bookkeeping could promote B06 past its own free kill test.
NEXT_GATE_COST_KEYS = (
    # ---- dated-priority slot, NEWEST FIRST. Same mechanism and same reason as
    # NEXT_GATE_KEYS[0] / KILL_KEYS[0] (LIFECYCLE_SCHEMA.md sec 2.0), added
    # 2026-08-16 after MEASURING it on B12.
    #
    # The defect this fixes: a gate that *was* free and has since been
    # DISCHARGED could not be expressed. B12's G0 was genuinely 0 GPU and it
    # genuinely PASSED both legs on 2026-08-16 at zero GPU
    # (g0_result_20260816, with a 224-tensor export self-check and a 30-candidate
    # novelty survey). But its `next_gate_gpu` is byte-frozen at "0 GPU for G0.
    # Both legs of the next gate are CPU-only..." and append-only (sec 0) forbids
    # editing it -- so the reader kept pinning it to ready_cpu, telling the next
    # agent to run a gate that had already been run. That is the B07 kill-gate
    # stall exactly (six days of "NO_KILL_GATE_DEFINED" over a written gate),
    # transplanted from the gate axis to the gate-COST axis.
    #
    # Precedence here is first-PRESENT, not first-MATCH: if a dated slot exists
    # it is authoritative and the older strings are not consulted at all.
    # First-match cannot work, because the older string will always still match
    # (it is frozen), so a dated override would be unreachable. Safe for the
    # same reason sec 2.0 gives for gates and NOT for blockers: a newer gate
    # still has to state its own cost and is auditable against the dated record
    # that discharged it, whereas a newer-and-shorter *blocker* list would
    # loosen constraints by silence.
    "next_gate_gpu_20260816",
    "next_gate_gpu", "next_gate_cost", "gate_gpu",
    "next_gate_gpu_cost")
_DATED_COST_KEYS = ("next_gate_gpu_20260816",)
# 2026-08-17: the tuple above is a PINNED DATE and cannot see next_gate_gpu_20260817 or
# anything later. Measured by an agent that first CLAIMED a dated cost key would guard
# B09's inference path, then tested the claim and found it false. Same defect class as the
# lifecycle / lifecycle_why enumerations fixed earlier today, and as the emitter whose
# hardcoded damaged_rungs silently defined a headline: a hardcoded list standing in for a
# structural query over an append-only file whose whole purpose is new dated keys. Resolve
# by pattern so no future date needs a code edit -- the edit is exactly the step that gets
# forgotten, and the failure is silent (the reader keeps serving a superseded value).
_DATED_COST_RE = re.compile(r"^(?:next_gate_gpu|next_gate_cost|gate_gpu)_(20\d{6})(?:_[a-z0-9]+)?$")


def dated_cost_keys(doc):
    """Dated next-gate-cost keys present in THIS doc with a NON-EMPTY string value, newest first.

    Newest-first is safe here for the same reason it is safe for lifecycle: this value can
    only DOWNGRADE a proposal from ready_gpu to ready_cpu (a gate declared free is surfaced
    as CPU work), never upgrade it past a check. So recency can only free a card.

    The non-empty filter is inherited from the code this replaces and is load-bearing: the
    caller takes only dated[0], so a dated key present but blank would SHADOW a perfectly
    good undated `next_gate_gpu` and silently lose the free-gate signal.
    """
    return sorted((k for k in doc
                   if _DATED_COST_RE.match(k)
                   and isinstance(doc.get(k), str) and doc[k].strip()),
                  reverse=True)
_FREE_MARKERS = ("0 gpu", "zero gpu", "no gpu", "0-gpu", "cpu only", "cpu-only",
                 "0 gpu-h", "0gpu")


def _next_gate_is_free(d):
    """Return a short quote if the doc says its next gate needs no GPU, else ''.

    Read for CONTENT: `next_gate_gpu` is prose, and a value like "The drift leg is
    0 GPU ... Only the replication and second-compressor legs need GPU" means the
    NEXT step is free even though later steps are not. A free first leg is exactly
    the case that must run before any card is spent, so a mixed value counts as
    free. Deliberately conservative in the safe direction: a false "free" costs a
    delay, a false "needs GPU" spends GPU-h on an untested claim.

    A DATED slot short-circuits: it is the only value consulted, including the
    `gpu_cost_estimate` fallback below (whose docstring says it applies when the
    top-level key is absent -- "sometimes carries the per-leg breakdown
    INSTEAD"). Without that, B12's frozen `gpu_cost_estimate.next_gate_cost`
    ("0 GPU -- G0 is CPU-only") would re-pin a discharged gate on its own; it was
    measured doing exactly that.

    ⚠ SUBSTRING HAZARD, measured 2026-08-16 while writing B12's dated value. The
    first draft read "1.46 GPU-h. G0 PASSED both legs 2026-08-16 at 0 GPU" -- and
    the "0 gpu" marker fired on the phrase describing what the DISCHARGED gate
    had cost, re-pinning the proposal it was meant to release. A dated cost value
    must state only what the NEXT step costs; do not narrate the history of a
    free gate inside it.
    """
    dated = dated_cost_keys(d)
    keys = (dated[0],) if dated else NEXT_GATE_COST_KEYS
    for k in keys:
        v = d.get(k)
        if isinstance(v, dict):
            v = " ".join(str(x) for x in v.values())
        if not isinstance(v, str) or not v.strip():
            continue
        vl = v.lower()
        if any(m in vl for m in _FREE_MARKERS):
            return f"{k}: {v.strip()[:120]}"
    if dated:
        return ""       # the dated slot spoke; do not let a frozen sibling veto it
    # gpu_cost_estimate sometimes carries the per-leg breakdown instead.
    gce = d.get("gpu_cost_estimate")
    if isinstance(gce, dict):
        for kk, vv in gce.items():
            if not isinstance(vv, str):
                continue
            vvl = vv.lower()
            if any(m in vvl for m in _FREE_MARKERS) and (
                    "leg" in kk.lower() or "drift" in kk.lower()
                    or "first" in kk.lower() or "next" in kk.lower()):
                return f"gpu_cost_estimate.{kk}: {vv.strip()[:120]}"
    return ""
# Containers one level down that may hold a BLOCK_KEYS clause. Prefix-matched, so
# a dated wrapper like `disposition_2026_08_12` is covered by "disposition".
# Added 2026-08-15 after measuring that A02's nested `gpu_policy` was invisible
# and the proposal therefore reported ready_gpu -- see `_walk_blockers`.
NESTED_BLOCKER_CONTAINERS = ("disposition", "closure", "verdict", "kill_gate_verdict",
                             "postmortem", "decision")
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
# 2026-08-17: the list above has NO dated slot, so a `novelty_verdict_20260817` was
# invisible -- measured on /tmp copies by an agent that wrote one: the dated key alone left
# novelty_checked=False and evidence="absent", i.e. an adjudicated gate still read as unrun.
# Fourth instance of the same defect class today (lifecycle, lifecycle_why, next_gate cost,
# and now this). Resolved by pattern, dated newest-first ahead of the undated spellings.
#
# Deliberately anchored and narrow: it must match `novelty_verdict_<date>` but NOT
# `novelty_status_detail` (prose), `novelty_checked` (the boolean read separately), or
# `novelty_verdict_why_*` (a reason field, not a verdict). This is the same reason
# LIFECYCLE_SCHEMA.md sec 2.0 forbids a wildcard scan over kill_gate_*: `kill_gate_verdict`
# is a RESULT and `updated_..._kill_gate_pass` is a CHANGELOG, and a glob would eat both.
_DATED_NOVELTY_RE = re.compile(r"^(?:novelty_verdict|k1_novelty)_(20\d{6})(?:_[a-z0-9]+)?$")


def novelty_verdict_keys(doc):
    """Dated novelty-verdict keys in THIS doc (newest first), then the undated spellings."""
    dated = sorted((k for k in doc if _DATED_NOVELTY_RE.match(k)), reverse=True)
    return dated + NOVELTY_VERDICT_KEYS
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
#
# 2026-08-17: SAME BUG AS THE LIFECYCLE LIST, TWICE OVER. (a) The dated entries were a
# hardcoded ["..._20260815", "..._20260814"], so a `lifecycle_why_20260817` was invisible.
# (b) Worse, undated `lifecycle_reason` was listed FIRST, so even a legible dated reason
# lost to whatever prose happened to be in the original field. Measured on B12: the queue
# reported the pilot-killed proposal alongside its 08-16 reason string ("Two 0-GPU blockers
# ... NOVELTY IS UNCHECKED"), which had been superseded twice over -- novelty cleared 08-16,
# the pilot ran and failed 08-17. Reason and state must be resolved by the SAME rule, or the
# report pairs a fresh state with a stale justification, which is harder to catch than a
# plainly stale row. Resolve by pattern, newest date first, undated LAST.
_DATED_LIFECYCLE_WHY = re.compile(r"^lifecycle_why_(20\d{6})(?:_[a-z0-9]+)?$")


def lifecycle_reason_keys(doc):
    """Dated lifecycle_why keys in THIS doc, newest first, then the undated fallbacks."""
    dated = sorted((k for k in doc if _DATED_LIFECYCLE_WHY.match(k)), reverse=True)
    return dated + ["lifecycle_reason", "lifecycle_why"]


# Static fallback for any caller without a doc in hand. lifecycle_reason_keys(doc) is
# authoritative -- same relationship as LIFECYCLE_KEYS to lifecycle_keys(doc).
LIFECYCLE_REASON_KEYS = ["lifecycle_reason",
                         "lifecycle_why_20260815", "lifecycle_why_20260814",
                         "lifecycle_why"]
# `lifecycle` itself needs a dated slot for the same append-only reason, and it
# is the LAST of the four to get one (gate 08-14, kill 08-15, reason 08-15,
# lifecycle 08-16). Measured on B12 2026-08-16: `lifecycle` was read at the
# declaration site as a BARE `d.get("lifecycle")`, and because a declared
# ready_cpu is an authoritative EARLY RETURN, a frozen "ready_cpu" could never
# be superseded no matter how much evidence was appended. B12's G0 passed both
# legs at zero GPU and the queue still said "run the free test first".
#
# Same asymmetry as everywhere else in this file: recency is safe for a STATE
# that must still pass the downstream checks (a declared ready_gpu is NOT taken
# on faith -- see the "declaration can downgrade, not upgrade" branch), and
# unsafe only for blockers, which is why blockers use pointers instead.
#
# 2026-08-17: THE ENUMERATION ITSELF WAS THE BUG. This read
# ["lifecycle_20260816", "lifecycle"], so when B12's pilot actually RAN, FAILED its own
# pre-registered gate (clause_4 + clause_5), and I appended
# `lifecycle_20260817 = killed_by_own_gate`, the queue went right on printing `ready_gpu`.
# A hardcoded list of dates cannot track an append-only file whose entire purpose is new
# dated keys: every future date needs a code edit, and the edit is exactly the step that
# gets forgotten -- so the reader silently keeps serving a superseded state. Same class as
# the emitter whose hardcoded damaged_rungs silently defined a headline, and as the
# NEXT_GATE_KEYS list two slots above. Resolve by PATTERN, newest date first.
#
# The optional `_<suffix>` accepts a SECOND record filed on the SAME day, which append-only
# forces the moment a same-day entry needs correcting: sec 0 forbids editing
# `lifecycle_20260817`, and dating the correction 20260818 would be a false timestamp. Plain
# reverse string sort already orders `lifecycle_20260817_corrected` above bare
# `lifecycle_20260817` (verified, not assumed -- the suffix sorts after the bare form because
# it is a strict prefix), so a same-day correction wins without special-casing.
_DATED_LIFECYCLE = re.compile(r"^lifecycle_(20\d{6})(?:_[a-z0-9]+)?$")


def lifecycle_keys(doc):
    """Dated lifecycle keys present in THIS doc, newest first, then the bare key.

    Preferring the newest date is safe for the reason the comment above already gives: a
    declared lifecycle is not taken on faith. A declaration can DOWNGRADE freely, while an
    upgrade still has to survive every downstream check. So recency can only tighten.
    """
    dated = sorted((k for k in doc if _DATED_LIFECYCLE.match(k)), reverse=True)
    return dated + ["lifecycle"]


# Static fallback for any caller without a doc in hand. lifecycle_keys(doc) is authoritative.
LIFECYCLE_KEYS = ["lifecycle_20260816", "lifecycle"]


# Language by which an append-only record RETRACTS something it said earlier in the
# same blob. Anchored on the SUPERSEDING half, never on the claim being retracted:
# "does not exist" is what we are trying to stop believing, so it must not be what
# decides whether we keep reading.
_RETRACTION_RE = re.compile(
    r"SUPERSED|NOW EXISTS|IS NOW DISCHARGED|IT DOES|NO LONGER|CORRECT[ED]?\b|"
    r"APPENDED 20\d{6}|NOT rewritten|kept verbatim|HAS NOW BEEN DONE|RESOLVED|"
    r"WAS WRONG|DISCHARGED", re.I)

# The marker is also the idempotence sentinel (see _txt), so it must be a string that
# cannot plausibly occur in a STATUS.json prose field.
_SPLICE_MARK = " ⟨TAIL RETRACTS, spliced⟩ "


def _txt(v, n=400):
    """Truncate for display WITHOUT severing a retraction from the claim it retracts.

    WHY THIS IS NOT A PLAIN s[:n]
    -----------------------------
    Measured on B03, 2026-08-17. `lifecycle_why_20260815` is 1104 chars and the queue
    printed the first 260 of them:

        "...(2) novelty is unadjudicated and RELATED_WORK.md is absent. NOT cl…"

    cut off mid-word, one clause before its own bracketed correction:

        "[APPENDED 2026-08-15, NOT rewritten: blocker (2) IS NOW DISCHARGED --
         RELATED_WORK.md exists and novelty_checked is true...]"

    RELATED_WORK.md is 34,354 B on disk. So the surface a dispatcher actually reads
    asserted a live blocker whose retraction was sitting just past the cut.

    This is structural, not bad luck. LIFECYCLE_SCHEMA.md sec 0 makes these records
    APPEND-ONLY: a claim is never edited, it is superseded by text APPENDED AFTER it.
    Corrections therefore always live in the TAIL -- exactly what head-truncation
    removes. Head-only truncation is a systematically stale reader of an append-only
    field. Verified across the queue: 9 of 9 lifecycle reasons exceed their limit, so
    every one of them was being shown head-first with its tail dropped.

    B03's own record predicted this failure verbatim, which is why it is worth fixing
    at the display layer rather than per-proposal:

        "This key exists because lifecycle_why_20260815 -- the field ready_queue.py
         prints as the lifecycle justification -- still says 'RELATED_WORK.md is
         absent', so the correction was invisible at the surface a reader actually
         sees."

    BEHAVIOUR: if the hidden tail contains retraction language, keep the head AND
    splice in the retracting tail, marked, instead of dropping it. Cost is a longer
    line; the alternative is dispatching an agent to write a file that exists (which
    has now happened twice -- see memory/append-only-records-outlive-their-own-truth).

    Deliberately NOT done: raising `n` globally. The tail is what matters, not the
    length, and a bigger head still severs a long enough record.
    """
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    if len(s) <= n:
        return s
    # IDEMPOTENCE. Some fields are _txt'd twice -- built at n=400 (rec["next_gate"],
    # :812) and again at print time with a smaller n (:1184). Without this guard the
    # second pass treats the FIRST pass's output as fresh prose: it re-truncates the
    # head, finds "TAIL RETRACTS" (which matches _RETRACTION_RE via "RETRACT") in the
    # remainder, and splices again -- measured on B03's next_gate, which printed the
    # marker twice and spliced the second fragment out of the middle of the first.
    # A double-spliced string is worse than a truncated one, because the reader can no
    # longer tell which retraction attaches to which claim. So: splice at most once,
    # and on an already-spliced string only shorten the head.
    if _SPLICE_MARK in s:
        head, _, spliced = s.partition(_SPLICE_MARK)
        keep = max(n - len(_SPLICE_MARK) - len(spliced), 60)
        return (head if len(head) <= keep else head[:keep - 1] + "…") + \
            _SPLICE_MARK + spliced
    head, tail = s[:n - 1], s[n - 1:]
    m = _RETRACTION_RE.search(tail)
    if not m:
        return head + "…"
    # Splice from the start of the sentence/bracket carrying the retraction, so the
    # correction arrives with its subject attached rather than as a dangling clause.
    start = max(tail.rfind(sep, 0, m.start()) + 1
                for sep in (". ", "[", "-- ", "\n"))
    frag = tail[start:].strip()
    return head + "…" + _SPLICE_MARK + (frag if len(frag) <= 400
                                        else frag[:399] + "…")


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

    ONE level of nesting is searched, via the explicit `NESTED_BLOCKER_CONTAINERS`
    allow-list, for the same reason `_first_nested` exists for gates.
    MEASURED 2026-08-15: A02's disposition verdict is
      disposition_2026_08_12.gpu_policy = "NO further A02 GPU. Resurrection
      requires a NEW MECHANISM, not another read-out of the same ladder."
    Scanning only the top level made that verdict INVISIBLE, and with A02's
    RELATED_WORK.md in place the proposal reported `ready_gpu` -- i.e. the
    scheduler would have offered me a proposal that its own record closes for
    GPU. Reproduced both ways: nested-only -> `1 ready_gpu`; same file with the
    policy also surfaced at top level -> `0 ready_gpu, blocker STILL LIVE`.
    A dated disposition wrapper is exactly where a closing verdict naturally
    gets written, so this is the common case, not an exotic one.
    """
    seen = set()

    def _emit(prefix, v):
        if isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, list):
                    yield f"{prefix}.{sk}", sv          # the container itself
                    for i, item in enumerate(sv):
                        yield f"{prefix}.{sk}[{i}]", item
                elif isinstance(sv, str) and sv.strip():
                    yield f"{prefix}.{sk}", sv
                elif isinstance(sv, dict) and sv:
                    yield f"{prefix}.{sk}", sv
        elif isinstance(v, list):
            yield prefix, v
            for i, sv in enumerate(v):
                yield f"{prefix}[{i}]", sv
        elif isinstance(v, str) and v.strip():
            yield prefix, v

    for k in BLOCK_KEYS:
        if k in d:
            for path, value in _emit(k, d[k]):
                if path not in seen:
                    seen.add(path)
                    yield path, value

    # one level down, allow-listed containers only -- never a blind deep walk,
    # which would re-read prose mentioning a blocker as if it were one.
    for c in NESTED_BLOCKER_CONTAINERS:
        for ck, cv in d.items():
            if not (ck == c or ck.startswith(c)) or not isinstance(cv, dict):
                continue
            for k in BLOCK_KEYS:
                if k in cv:
                    for path, value in _emit(f"{ck}.{k}", cv[k]):
                        if path not in seen:
                            seen.add(path)
                            yield path, value


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

    gk, gate = _first_nested(d, dated_executable_keys(d, "next_gate") + NEXT_GATE_KEYS)
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

    ck, kill = _first_nested(d, dated_executable_keys(d, "kill_gate") + KILL_KEYS)
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
        for k in novelty_verdict_keys(d):
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
    _lck, explicit = _first(d, lifecycle_keys(d))
    VALID_LC = ("ready_gpu", "ready_cpu", "needs_prior_gate", "running",
                "promoted", "dead")
    _, _lc_why = _first(d, lifecycle_reason_keys(d))
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
        # FAIL CLOSED. Measured on B12, 2026-08-17: after the pilot ran and FAILED its own
        # pre-registered gate, I declared `lifecycle_20260817 = "killed_by_own_gate"` -- a
        # word that is not in VALID_LC (the schema's terminal word is `dead`). The reader
        # appended exactly this warning and then FELL THROUGH to inference, which re-derived
        # **ready_gpu** and listed the killed direction as the queue's only dispatchable GPU
        # item. The warning was printed and the wrong answer was served in the same breath.
        #
        # This is the same accident the B02 comment above memorialises -- paperwork
        # out-voting a fired kill gate -- reached through a bad enum value instead of a
        # missing field. So the repair has to be the same shape: an unparseable declaration
        # is NOT an absent declaration. Someone wrote a state down; we could not read it;
        # the one thing we must not do is substitute our own inference and spend a card on
        # it. Park it in needs_prior_gate, where the prior "gate" is a human fixing the word.
        #
        # Deliberately not lenient (no fuzzy match to `dead`, no substring test): guessing
        # what an owner meant by an unknown terminal word is how `blocked` -- the one string
        # LIFECYCLE_SCHEMA.md sec 1 outright bans -- came to mean three different things.
        rec["problems"].append(
            f"lifecycle={_txt(explicit, 40)!r} is not one of {VALID_LC}"
            " -- UNPARSEABLE declaration, parked (fix the word in STATUS.json;"
            " the schema's terminal state is 'dead' with the reason in lifecycle_why_*)")
        rec["lifecycle"] = "needs_prior_gate"
        rec["lifecycle_reason"] = (
            f"DECLARED lifecycle={_txt(explicit, 60)!r} is not in the schema vocabulary, so"
            " it cannot be honoured OR ignored: inference is suppressed and this proposal is"
            " parked until the declaration is legible. Declared reason: " + _txt(_lc_why, 200))
        rec["needs_arch"] = d.get("needs_arch", "UNRECORDED")
        return rec

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
        # A FREE next gate must not be dispatched as a GPU task, even when every
        # paperwork check passes. MEASURED 2026-08-15 on B06: appending
        # `related_work_status: "audited"` -- pure bookkeeping -- flipped it
        # ready_cpu -> ready_gpu, while its own record says
        #   next_gate_gpu = "The drift leg is 0 GPU if the canonical predictions
        #                    are already on disk (rejudge = OpenAI API + CPU)."
        #   kill_gate.condition_1_status = "PARTIALLY TESTABLE FROM DISK NOW, and
        #                                   it is the one at real risk."
        # So the scheduler would have spent cards on a proposal whose free test
        # could KILL the claim first. That is the worst possible ordering, and no
        # amount of novelty adjudication licenses it: a novelty verdict says
        # "not preempted", never "worth a card now".
        _ng_free = _next_gate_is_free(d)
        if _ng_free:
            lc = "ready_cpu"
            why = ("gate + kill gate + novelty all OK, but this proposal's OWN "
                   "next gate costs no GPU (" + _ng_free + ") -- run the free "
                   "test before spending a card; it may settle or kill the claim")
            rec["problems"].append(
                "next gate is 0-GPU by its own record; held in ready_cpu so the "
                "free test runs first (not a paperwork deficiency)")
        elif rec.get("lifecycle_declared") == "needs_prior_gate":
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


def _is_moved_stub(d):
    """True if `d` is a resolved MOVED/merged pointer, not a proposal missing paperwork.

    Deliberately strict on all three counts, because the cost of a false positive
    here is a real proposal going invisible:
      * no STATUS.json (caller already checked),
      * EXACTLY ONE .md in the directory and no subdirectory holding more,
      * that file says MOVED or MERGED within its first 3 lines.
    `B04-eval-fragility/README.md` matches: its title line is
    "# MOVED -- this directory was a bookkeeping split, not a second proposal",
    the merge happened 2026-08-14, and it names the canonical home. Flagging it
    as "invisible to the scheduler" is a false positive that costs the next agent
    a re-investigation of a closed question (it cost MAIN one this session).
    """
    try:
        entries = os.listdir(d)
    except OSError:
        return False
    mds = [f for f in entries if f.endswith(".md")]
    others = [f for f in entries
              if not f.endswith(".md") and not f.startswith(".")]
    if len(mds) != 1 or others:
        return False
    try:
        with open(os.path.join(d, mds[0]), encoding="utf-8") as f:
            head = "".join(f.readline() for _ in range(3)).upper()
    except OSError:
        return False
    return "MOVED" in head or "MERGED" in head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(ROOT, "active", "*", "STATUS.json")) +
                   glob.glob(os.path.join(ROOT, "backlog", "*", "STATUS.json")))
    recs = [read_one(p) for p in paths]

    # A proposal directory with NO STATUS.json is invisible to the scheduler,
    # which is its own failure mode -- report it. EXCEPT a MOVED/merged pointer
    # stub: `B04-eval-fragility/` holds one README.md whose first line is
    # "# MOVED -- this directory was a bookkeeping split, not a second proposal"
    # and which names its canonical home. That is a resolved tombstone, and
    # flagging it as a gap sends the next agent to re-solve a 2026-08-14 merge.
    # Recognised by CONTENT, not by name, so a real gap can never hide behind an
    # empty directory: the stub must be the ONLY .md, carry no STATUS.json, and
    # say MOVED/MERGED near the top.
    for d in sorted(glob.glob(os.path.join(ROOT, "active", "*")) +
                    glob.glob(os.path.join(ROOT, "backlog", "*"))):
        if not os.path.isdir(d) or os.path.exists(os.path.join(d, "STATUS.json")):
            continue
        if _is_moved_stub(d):
            continue
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

        # Stale-absence tripwire, surfaced HERE rather than left opt-in.
        #
        # 2026-08-17: proposal/check_stale_absence_claims.py existed, was correct, and would
        # have caught this exact mistake -- it names B06 and B09 with byte counts, and its own
        # output warns "each row above tells the next agent to produce a file that exists".
        # I dispatched an agent to WRITE two RELATED_WORK.md files that had been on disk since
        # 2026-08-15, and never ran the checker. A tripwire that only fires when someone
        # remembers to pull it is not a tripwire. The queue is the tool that IS run every
        # round, so the warning belongs in its output.
        #
        # Advisory only: it must never change an exit code or a lifecycle. Presence is not
        # sufficiency -- a blocker genuinely about a file's CONTENT stays valid; what is being
        # flagged is only the claim that the file is MISSING.
        try:
            import subprocess
            chk = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "check_stale_absence_claims.py")
            if os.path.exists(chk):
                p = subprocess.run([sys.executable, chk], capture_output=True, text=True,
                                   timeout=60)
                # Parse ONLY the checker's own count line. Do not re-derive the count by
                # filtering its table rows: that is the loose-match failure the checker's
                # own postmortem warns about (a keyword hit inside a banner got counted as
                # a data row). Anchor on the line format, and if the line is absent, say so
                # rather than inferring zero.
                n = None
                for l in p.stdout.splitlines():
                    if l.startswith("stale absence assertions:"):
                        n = int(l.split(":", 1)[1].strip() or 0)
                        break
                if n is None:
                    print("\n  (stale-absence tripwire: could not find the count line in the"
                          " checker's output -- run it by hand rather than assuming zero)")
                elif n:
                    print(f"\n  ⚠ {n} STALE ABSENCE ASSERTION(S): some STATUS.json still says a"
                          " file is missing that is ON DISK.")
                    print("    Before you dispatch an agent to CREATE any file, run:"
                          " python3 proposal/check_stale_absence_claims.py")
                    print("    (measured 2026-08-17: relaying one of these cost an agent most"
                          " of its run disproving the premise)")
        except Exception as e:
            print(f"\n  (stale-absence tripwire could not run: {e}; "
                  "run proposal/check_stale_absence_claims.py by hand)")

    if a.strict and any(r["lifecycle"] in ("UNREADABLE", "NO_STATUS_JSON")
                        for r in recs):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
