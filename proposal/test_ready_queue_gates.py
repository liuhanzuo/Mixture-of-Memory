#!/usr/bin/env python3
"""Negative tests for proposal/ready_queue.py's gate-detection and blocker logic.

0 GPU. stdlib only. Run directly:

    python proposal/test_ready_queue_gates.py

Why these cases and not others
------------------------------
The scheduler's failure mode is asymmetric. A reader that UNDER-reports
readiness manufactures busywork ("go write the kill gate you wrote six days
ago"); a reader that OVER-reports it hands out cards. Both happened, and on
2026-08-15 they happened to the SAME proposal at once and cancelled out:

  * A04's kill gate is real, pre-registered, and quantified -- but nested at
    `gate_design.kill_condition_verbatim`, so `KILL_KEYS`+`_first` (top level
    only) reported "no kill_gate field".
  * `BLOCK_KEYS` was parsed for DISPLAY and never consulted by lifecycle
    inference -- so fixing the nesting bug ALONE would have made A04 the queue's
    single `ready_gpu` item, while its own record says "USER APPROVAL for GPU.
    The full gate is 1,077-4,309 GPU-h; nothing beyond Pilot Zero may be
    launched without it."

So (a) tests the gate is now visible, (c) tests it is still not dispatchable,
and (b)/(d)/(e) pin the three ways an over-eager fix would leak:
  (b) a genuinely gate-less proposal must NOT be waved through by the nested
      lookup broadening what counts as a gate;
  (d) an ALREADY-DISCHARGED blocker must not become a permanent hold (A04's
      `related_work_gate` says "CLEARED 2026-08-09 ... NO LONGER BLOCKING.") --
      otherwise the blocker check is just a second under-report;
  (e) a DECLARED terminal state must survive every new code path, because the
      original bug in this file was a killed direction (B02) being computed as
      the most dispatchable item in the queue.

(f)/(g)/(h), added 2026-08-15, cover the same asymmetry one level up: what
append-only does to a record that must CHANGE. LIFECYCLE_SCHEMA.md sec 0 forbids
editing a string, so every correction arrives as a sibling key -- and a reader
that resolves the OLD key first is permanently pinned to the worst version of
the record:
  (f) a later, better kill gate can only be appended, so KILL_KEYS needs a
      dated-priority slot mirroring NEXT_GATE_KEYS[0]. B07 wrote a four-clause
      pre-registered gate and the reader kept printing NO_KILL_GATE_DEFINED --
      a bug B07's own `_precedence_warning` predicted six days earlier. f3/f4/f5
      pin that precedence must not become a way to SKIP the check, to regress
      the plain spelling, or to become a `kill_gate_*` wildcard.
  (g) a discharged blocker likewise cannot be edited in place, so discharge is
      by POINTER at an exact dotted path (sec 2.1). g3 is the load-bearing one:
      narrowing A04's reason from 3 live blockers to 1 must NOT open the
      1,077-4,309 GPU-h door. g6 pins fail-closed-by-omission, which is the
      whole reason pointers were chosen over a `blocked_by_v2` restatement.
  (h) the justification for a DECLARED lifecycle is spelled `lifecycle_reason`
      in four files and `lifecycle_why_2026081X` in two, and the reader saw only
      the first -- printing "no reason field" for B07 while a 400-word
      justification naming both its 0-GPU blockers sat in the file.

All fixtures are built inline in a tempdir. No real STATUS.json is read or
written -- those files carry irreplaceable prose (LIFECYCLE_SCHEMA.md sec 0) and
a test that mutates its own input is not a test.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ready_queue  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"\n         {detail}" if detail else ""))
    return bool(cond)


def read_fixture(tmp, proposal_id, doc, related_work=False):
    """Write `doc` to <tmp>/<proposal_id>/STATUS.json and parse it."""
    d = os.path.join(tmp, proposal_id)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "STATUS.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    if related_work:
        with open(os.path.join(d, "RELATED_WORK.md"), "w") as f:
            f.write("# fixture\n")
    return ready_queue.read_one(p)


# A minimal document that satisfies EVERY check except the one under test, so a
# failure localises to that check instead of to missing paperwork.
def base_doc(**over):
    doc = {
        "id": "FIXTURE",
        "status": "fixture",
        "next_gate": "Run the one decisive cell: arm A vs arm B, n=100, 1 GPU-h.",
        "kill_gate": "Killed if the paired difference straddles 0 at BH q=.05.",
        "gpu_cost_estimate": {"value": "1.0 GPU-h", "basis": "measured s/step"},
        "novelty_checked": True,
        "needs_arch": "sm_90",
    }
    doc.update(over)
    for k, v in list(doc.items()):
        if v is None:
            del doc[k]
    return doc


def main():
    tmp = tempfile.mkdtemp(prefix="rq_gate_test_")
    try:
        print("\n(a) nested gate_design.kill_condition_verbatim is a kill gate")
        # Shape copied from A04's real STATUS.json (values abbreviated).
        r = read_fixture(tmp, "A04-like", base_doc(
            kill_gate=None,
            gate_design={
                "document": "A04_GATE_DESIGN.md",
                "written": "2026-08-09",
                "gpu_spent_designing": 0,
                "kill_condition_verbatim":
                    "A04 is killed if ANY of three clauses fires. K1 -- no rule "
                    "disagreement ... K2 -- disagreement drowned by seed "
                    "variance ... K3 -- axes at floor ...",
                "frozen_constants": {"T_plateau": "2.0%/5k",
                                     "rho_retention": 0.85},
            }), related_work=True)
        ok_a1 = check(
            "kill_gate_key is the traceable dotted path",
            r["kill_gate_key"] == "gate_design.kill_condition_verbatim",
            f"got {r['kill_gate_key']!r}")
        ok_a2 = check(
            "no 'no kill_gate field' / 'kill gate undefined' problem is raised",
            not any("kill_gate field" in p or "kill gate undefined" in p
                    for p in r["problems"]),
            f"problems={r['problems']}")
        ok_a3 = check(
            "lifecycle_reason does not claim the kill gate must be written",
            "kill gate undefined" not in r["lifecycle_reason"],
            f"reason={r['lifecycle_reason'][:120]!r}")

        print("\n(b) a proposal with NO kill gate anywhere is still ready_cpu")
        r = read_fixture(tmp, "no-kill", base_doc(kill_gate=None),
                         related_work=True)
        ok_b1 = check("lifecycle == ready_cpu", r["lifecycle"] == "ready_cpu",
                      f"got {r['lifecycle']!r}: {r['lifecycle_reason'][:120]}")
        ok_b2 = check("the kill gate is reported as the blocking 0-GPU task",
                      "kill gate undefined" in r["lifecycle_reason"],
                      f"reason={r['lifecycle_reason'][:120]!r}")
        # The nested lookup must not accept prose that merely mentions killing.
        r2 = read_fixture(tmp, "no-kill-prose", base_doc(
            kill_gate=None,
            related_work_status={"closest_prior_art": [
                {"kills": "arXiv:XXXX.YYYYY kills the storage form of the claim"}]},
            history_20260808={"next_gate_then": "an old superseded gate"},
        ), related_work=True)
        ok_b3 = check(
            "prose containing 'kills' / a historical gate is NOT read as a gate",
            r2["kill_gate_key"] is None and r2["lifecycle"] == "ready_cpu",
            f"kill_gate_key={r2['kill_gate_key']!r} lifecycle={r2['lifecycle']!r}")
        # Explicit NO_KILL_GATE_DEFINED must keep failing (B07's real value).
        r3 = read_fixture(tmp, "no-kill-sentinel", base_doc(
            kill_gate="NO_KILL_GATE_DEFINED -- PROPOSAL.md has no Kill section."),
            related_work=True)
        ok_b4 = check("explicit NO_KILL_GATE_DEFINED sentinel still blocks GPU",
                      r3["lifecycle"] == "ready_cpu",
                      f"got {r3['lifecycle']!r}")

        print("\n(c) an un-discharged hard blocker is never ready_gpu")
        r = read_fixture(tmp, "A04-blocked", base_doc(
            gate_design={"kill_condition_verbatim": "K1/K2/K3 as pre-registered."},
            kill_gate=None,
            blocked_by={
                "related_work_gate":
                    "CLEARED 2026-08-09 (see related_work_status). "
                    "NO LONGER BLOCKING.",
                "still_blocking_before_any_gate_gpu": [
                    "PROPOSAL.md rewrite to the narrowed safe_residual_claim.",
                    "CODE (CPU, ~1 line, BLOCKING for K2): sampler seed=.",
                    "USER APPROVAL for GPU. The full gate is 1,077-4,309 "
                    "GPU-h; nothing beyond Pilot Zero may be launched "
                    "without it.",
                ]}), related_work=True)
        ok_c1 = check("lifecycle != ready_gpu", r["lifecycle"] != "ready_gpu",
                      f"got {r['lifecycle']!r}")
        ok_c2 = check("lifecycle == needs_prior_gate",
                      r["lifecycle"] == "needs_prior_gate",
                      f"got {r['lifecycle']!r}")
        ok_c3 = check(
            "the GPU-approval clause is named in the record, not summarised away",
            any("USER APPROVAL" in b["text"]
                for b in r.get("live_blockers", [])),
            f"live_blockers={[b['path'] for b in r.get('live_blockers', [])]}")
        ok_c4 = check(
            "all 3 live clauses are enumerated with citable paths",
            len(r.get("live_blockers", [])) == 3
            and all("still_blocking_before_any_gate_gpu[" in b["path"]
                    for b in r.get("live_blockers", [])),
            f"n={len(r.get('live_blockers', []))} paths="
            f"{[b['path'] for b in r.get('live_blockers', [])]}")
        # gpu_policy (B11's real spelling) is a hard blocker too.
        r2 = read_fixture(tmp, "gpu-policy", base_doc(
            gpu_policy="NO GPU until K1 (novelty, 0 GPU) passes."),
            related_work=True)
        ok_c5 = check("gpu_policy 'NO GPU until ...' is a hard blocker",
                      r2["lifecycle"] == "needs_prior_gate",
                      f"got {r2['lifecycle']!r}: {r2['lifecycle_reason'][:100]}")

        # (c') The counterfactual that makes (c) load-bearing rather than
        # incidental. Pre-fix, the A04 shape above was ALREADY not ready_gpu --
        # but for the wrong reason: its nested kill gate was invisible, so it
        # failed the kill-gate presence check and landed in ready_cpu with the
        # false instruction "write the kill gate". Neutralise ONLY the blocker
        # detector and the same document becomes ready_gpu, i.e. the fix-1-only
        # state would have dispatched a 1,077-4,309 GPU-h gate that its own
        # record says needs user approval. This asserts the blocker branch, not
        # the gate-presence check, is what holds A04 back now.
        real_live = getattr(ready_queue, "_live_blockers", None)
        try:
            ready_queue._live_blockers = lambda d: []
            r3 = read_fixture(tmp, "A04-fix1-only", base_doc(
                gate_design={"kill_condition_verbatim": "K1/K2/K3 as prereg."},
                kill_gate=None,
                blocked_by={"still_blocking_before_any_gate_gpu": [
                    "USER APPROVAL for GPU. The full gate is 1,077-4,309 GPU-h."
                ]}), related_work=True)
        finally:
            if real_live is None:
                del ready_queue._live_blockers
            else:
                ready_queue._live_blockers = real_live
        ok_c6 = check(
            "with the blocker check disabled the SAME doc becomes ready_gpu "
            "(so defect 2 is what holds it back, not defect 1)",
            r3["lifecycle"] == "ready_gpu",
            f"got {r3['lifecycle']!r} -- if this is not ready_gpu the two "
            f"defects are still entangled and (c) proves nothing")

        print("\n(d) an already-CLEARED blocker does not hold anything back")
        r = read_fixture(tmp, "cleared", base_doc(
            blocked_by={"related_work_gate":
                        "CLEARED 2026-08-09 (see related_work_status). "
                        "NO LONGER BLOCKING."}), related_work=True)
        ok_d1 = check("a wholly-discharged blocked_by leaves 0 live blockers",
                      r.get("live_blockers", None) == [],
                      f"live_blockers={r.get('live_blockers')}")
        ok_d2 = check("so the proposal is NOT parked in needs_prior_gate",
                      r["lifecycle"] != "needs_prior_gate",
                      f"got {r['lifecycle']!r}: {r['lifecycle_reason'][:120]}")
        # "not cleared" contains "cleared": the negation must win.
        r2 = read_fixture(tmp, "not-cleared", base_doc(
            blocked_by={"x": "NOT CLEARED: the upstream fix has not landed."}),
            related_work=True)
        ok_d3 = check("'NOT CLEARED' is not mistaken for a discharge",
                      r2["lifecycle"] == "needs_prior_gate",
                      f"got {r2['lifecycle']!r}")
        # "abandoned" contains "done": a bare 'done' token would leak here.
        r3 = read_fixture(tmp, "abandoned-word", base_doc(
            blocked_by={"x": "the second arm was abandoned and must be rebuilt"}),
            related_work=True)
        ok_d4 = check("a blocker containing the word 'abandoned' stays live",
                      r3["lifecycle"] == "needs_prior_gate",
                      f"got {r3['lifecycle']!r}")

        print("\n(e) a DECLARED dead proposal (B02 shape) stays dead")
        r = read_fixture(tmp, "B02-like", base_doc(
            lifecycle="dead",
            lifecycle_reason="Own pre-registered kill gate FIRED on both "
                             "lengths, both clauses.",
            kill_gate={"clause": "fully written"},
            kill_gate_verdict={"verdict": "FIRED"},
            novelty_verdict="gate cleared",
            required_before_stage0=["fix n", "re-run the j-sweep"],
        ), related_work=True)
        ok_e1 = check("lifecycle == dead", r["lifecycle"] == "dead",
                      f"got {r['lifecycle']!r}")
        ok_e2 = check("not resurrected into any ready bucket",
                      r["lifecycle"] not in ("ready_gpu", "ready_cpu",
                                             "needs_prior_gate"),
                      f"got {r['lifecycle']!r}")
        ok_e3 = check("the declaration is cited as authoritative",
                      "DECLARED lifecycle=dead" in r["lifecycle_reason"],
                      f"reason={r['lifecycle_reason'][:100]!r}")

        print("\n(f) a DATED kill gate wins over the older sentinel it may not "
              "overwrite  [B07]")
        # B07's real shape. Append-only (LIFECYCLE_SCHEMA.md sec 0) means the
        # honest early `"kill_gate": "NO_KILL_GATE_DEFINED ..."` CANNOT be
        # edited when a real gate is written later, so the later gate can only
        # arrive as a SIBLING key. B07's own `_precedence_warning` predicted the
        # consequence: KILL_KEYS had no dated slot (unlike
        # NEXT_GATE_KEYS[0]="next_gate_executable_20260814"), so a four-clause
        # pre-registered gate was written on 2026-08-14 and the reader kept
        # reporting NO_KILL_GATE_DEFINED -- telling the next agent to write a
        # document that already existed.
        r = read_fixture(tmp, "B07-like", base_doc(
            kill_gate="NO_KILL_GATE_DEFINED -- PROPOSAL.md has no Kill section.",
            kill_gate_executable_20260814={
                "_precedence_warning": "KILL_KEYS has NO dated-priority slot ...",
                "K1_retention_PRIMARY": "If the paired TTFT p99 advantage at C=8 "
                                        "is < 123.2 ms, or its 95% paired "
                                        "bootstrap CI includes 0, or its sign is "
                                        "negative -> B07 is KILLED.",
            }), related_work=True)
        ok_f1 = check(
            "kill_gate_key is the DATED key, not the older sentinel",
            r["kill_gate_key"] == "kill_gate_executable_20260814",
            f"got {r['kill_gate_key']!r}")
        ok_f2 = check(
            "no 'kill gate undefined' problem survives the newer gate",
            not any("kill gate undefined" in p or "no kill_gate field" in p
                    for p in r["problems"]),
            f"problems={r['problems']}")
        # The precedence fix must not become a way to SKIP checks: the dated slot
        # only changes WHICH string is read, so a dated key that is itself a
        # sentinel must still block.
        r2 = read_fixture(tmp, "B07-dated-sentinel", base_doc(
            kill_gate={"clause": "a real gate, but superseded"},
            kill_gate_executable_20260814="NO_KILL_GATE_DEFINED -- on reflection "
                                          "the earlier gate was not decidable.",
            ), related_work=True)
        ok_f3 = check(
            "a dated key that is itself a sentinel still blocks GPU "
            "(precedence changes which string is read, not whether it is checked)",
            r2["kill_gate_key"] == "kill_gate_executable_20260814"
            and r2["lifecycle"] == "ready_cpu",
            f"key={r2['kill_gate_key']!r} lifecycle={r2['lifecycle']!r}")
        # NO REGRESSION: with no dated key present, the plain `kill_gate` must
        # still be found exactly as before. This is the half of the change that
        # is easy to break by reordering a list.
        r3 = read_fixture(tmp, "kill-gate-plain", base_doc(), related_work=True)
        ok_f4 = check(
            "with no dated key, plain `kill_gate` is still resolved (no "
            "regression) and the doc still reaches ready_gpu",
            r3["kill_gate_key"] == "kill_gate" and r3["lifecycle"] == "ready_gpu",
            f"key={r3['kill_gate_key']!r} lifecycle={r3['lifecycle']!r}")
        # And the dated slot must not be a wildcard over `kill_gate_*`: B02's
        # `kill_gate_verdict` is an OUTCOME, B05's
        # `updated_20260814_kill_gate_pass` is a changelog line. A regex would
        # promote both to gate-hood.
        r4 = read_fixture(tmp, "kill-lookalikes", base_doc(
            kill_gate=None,
            kill_gate_verdict={"verdict": "FIRED"},
            updated_20260814_kill_gate_pass="the gate passed on 08-14",
            original_kill_reassessment="the original kill reason was too strong",
            ), related_work=True)
        ok_f5 = check(
            "`kill_gate_verdict` / `*_kill_gate_pass` / `*_kill_reassessment` "
            "are NOT read as gates (explicit list, not a wildcard)",
            r4["kill_gate_key"] is None and r4["lifecycle"] == "ready_cpu",
            f"key={r4['kill_gate_key']!r} lifecycle={r4['lifecycle']!r}")

        print("\n(g) a discharge POINTER closes the clause it names, and only "
              "that one  [A04]")
        # A04's real shape. Items [0] and [1] were discharged 2026-08-13 -- [1]
        # because the sampler fix it demands landed FOUR DAYS EARLIER as ce5c298
        # (verified on disk 2026-08-15: train_olmo2_arch_probe2.py:869 reads
        # `DistributedSampler(ds, shuffle=True, seed=args.seed)`). Append-only
        # forbids editing those strings, so the closure lives in a SIBLING key
        # and the reader was reporting 3 live blockers where 1 is live: right
        # verdict, reason overstated by two, and the next agent sent to redo
        # finished work. LIFECYCLE_SCHEMA.md sec 2.1.
        a04 = base_doc(
            gate_design={"kill_condition_verbatim": "K1/K2/K3 as pre-registered."},
            kill_gate=None,
            blocked_by={
                "related_work_gate": "CLEARED 2026-08-09. NO LONGER BLOCKING.",
                "still_blocking_before_any_gate_gpu": [
                    "PROPOSAL.md rewrite to the narrowed safe_residual_claim.",
                    "CODE (CPU, ~1 line, BLOCKING for K2): "
                    "train_olmo2_arch_probe2.py:863 is DistributedSampler("
                    "ds, shuffle=True) with NO seed= argument.",
                    "USER APPROVAL for GPU. The full gate is 1,077-4,309 "
                    "GPU-h; nothing beyond Pilot Zero may be launched "
                    "without it.",
                ]},
            blockers_discharged_20260813={
                "item_0_PROPOSAL_rewrite": {"state": "DONE 2026-08-13, 0 GPU."},
                "item_1_sampler_fix": {"state": "NOT OUTSTANDING. ce5c298."},
                "discharges": [
                    "blocked_by.still_blocking_before_any_gate_gpu[0]",
                    "blocked_by.still_blocking_before_any_gate_gpu[1]",
                ]})
        r = read_fixture(tmp, "A04-discharged", a04, related_work=True)
        ok_g1 = check(
            "exactly ONE live blocker remains (was 3 before the pointer)",
            len(r["live_blockers"]) == 1,
            f"n={len(r['live_blockers'])} "
            f"paths={[b['path'] for b in r['live_blockers']]}")
        ok_g2 = check(
            "and it is the GPU-approval clause [2], the only one not pointed at",
            r["live_blockers"][0]["path"].endswith("[2]")
            and "USER APPROVAL" in r["live_blockers"][0]["text"],
            f"got {r['live_blockers'][0] if r['live_blockers'] else None}")
        # THE LOAD-BEARING ASSERTION. The whole point of narrowing the reason is
        # that it must NOT loosen the gate: a mechanism that discharges the
        # 1,077-4,309 GPU-h approval clause as a side effect would be strictly
        # worse than the over-reporting it fixes.
        ok_g3 = check(
            "A04 is STILL needs_prior_gate -- narrowing the reason must not "
            "open the 1,077-4,309 GPU-h door",
            r["lifecycle"] == "needs_prior_gate",
            f"got {r['lifecycle']!r}: {r['lifecycle_reason'][:160]}")
        ok_g4 = check(
            "the reported reason says 1 blocker, not 3",
            "1 un-discharged blocker" in r["lifecycle_reason"],
            f"reason={r['lifecycle_reason'][:160]!r}")
        ok_g5 = check(
            "the discharged paths are recorded for audit",
            r.get("discharged_blockers") == [
                "blocked_by.still_blocking_before_any_gate_gpu[0]",
                "blocked_by.still_blocking_before_any_gate_gpu[1]"],
            f"got {r.get('discharged_blockers')}")

        # (g') Discharge is FAIL-CLOSED BY OMISSION -- the property that decided
        # `discharges` pointers over a `blocked_by_v2` restatement. Pointing at
        # only [0] must leave BOTH [1] and [2] live. Under a v2 list, a shorter
        # rewrite would retire the GPU-approval clause by simply not mentioning
        # it, and nothing would flag that.
        r2 = read_fixture(tmp, "A04-partial", base_doc(
            gate_design={"kill_condition_verbatim": "K1/K2/K3."},
            kill_gate=None,
            blocked_by={"still_blocking_before_any_gate_gpu": [
                "PROPOSAL.md rewrite.",
                "CODE fix, BLOCKING for K2.",
                "USER APPROVAL for GPU. 1,077-4,309 GPU-h.",
            ]},
            rec_20260813={"discharges":
                          ["blocked_by.still_blocking_before_any_gate_gpu[0]"]}),
            related_work=True)
        ok_g6 = check(
            "pointing at [0] alone leaves [1] and [2] LIVE (omission does not "
            "discharge)",
            [b["path"].split("[")[-1] for b in r2["live_blockers"]] == ["1]", "2]"],
            f"paths={[b['path'] for b in r2['live_blockers']]}")
        # A pointer must not be able to close a clause it does not name -- e.g.
        # by prefix. "[1]" is a prefix of nothing here, but "[0]" naming must not
        # sweep "[0]" of a DIFFERENT container either.
        r3 = read_fixture(tmp, "A04-wrong-container", base_doc(
            gate_design={"kill_condition_verbatim": "K1/K2/K3."},
            kill_gate=None,
            blocked_by={"still_blocking_before_any_gate_gpu": [
                "USER APPROVAL for GPU. 1,077-4,309 GPU-h."]},
            required_before_stage0=["a different, still-live prerequisite"],
            rec={"discharges": ["required_before_stage0[0]"]}), related_work=True)
        ok_g7 = check(
            "a pointer at required_before_stage0[0] does NOT touch "
            "blocked_by...[0] (paths are exact, not positional)",
            len(r3["live_blockers"]) == 1
            and "USER APPROVAL" in r3["live_blockers"][0]["text"],
            f"paths={[b['path'] for b in r3['live_blockers']]}")
        # A typo'd pointer must be REPORTED, not silently absolved. A pointer
        # that matches nothing reads, to a human, exactly like a closed blocker.
        r4 = read_fixture(tmp, "A04-dangling", base_doc(
            gate_design={"kill_condition_verbatim": "K1/K2/K3."},
            kill_gate=None,
            blocked_by={"still_blocking_before_any_gate_gpu": [
                "USER APPROVAL for GPU. 1,077-4,309 GPU-h."]},
            rec={"discharges": ["blocked_by.typo_that_does_not_exist[0]"]}),
            related_work=True)
        ok_g8 = check(
            "a dangling pointer is reported and discharges nothing",
            any("dangling" in p for p in r4["problems"])
            and len(r4["live_blockers"]) == 1,
            f"problems={r4['problems']} n_live={len(r4['live_blockers'])}")
        # A pointer at the CONTAINER closes the whole clause list at once.
        r5 = read_fixture(tmp, "A04-container-ptr", base_doc(
            blocked_by={"still_blocking_before_any_gate_gpu": ["a", "b", "c"]},
            rec={"discharges":
                 ["blocked_by.still_blocking_before_any_gate_gpu"]}),
            related_work=True)
        ok_g9 = check(
            "a container-level pointer discharges all of its clauses, and the "
            "doc is then not parked",
            r5["live_blockers"] == [] and r5["lifecycle"] == "ready_gpu",
            f"live={r5['live_blockers']} lifecycle={r5['lifecycle']!r}")

        print("\n(h) a DECLARED lifecycle's justification is read under either "
              "spelling  [B03/B07]")
        # B02/B04/B05/B11 spell it `lifecycle_reason`; B03 and B07 use
        # `lifecycle_why_2026081X`. The reader printed "no reason field" for B07
        # while a 400-word justification naming both its 0-GPU blockers sat in
        # the file -- the same "information on disk, key list does not reach it"
        # defect as the missing dated kill slot.
        r = read_fixture(tmp, "B07-why", base_doc(
            lifecycle="ready_cpu",
            lifecycle_why_20260814="RELATED_WORK.md and the concurrency axis are "
                                   "still missing, and both are 0-GPU."),
            related_work=True)
        ok_h1 = check(
            "lifecycle_why_2026081X is surfaced instead of 'no reason field'",
            "no reason field" not in r["lifecycle_reason"]
            and "concurrency axis" in r["lifecycle_reason"],
            f"reason={r['lifecycle_reason'][:160]!r}")
        r2 = read_fixture(tmp, "plain-why", base_doc(
            lifecycle="dead", lifecycle_reason="kill gate FIRED"), related_work=True)
        ok_h2 = check(
            "the plain `lifecycle_reason` spelling still wins (no regression)",
            "kill gate FIRED" in r2["lifecycle_reason"]
            and r2["lifecycle"] == "dead",
            f"reason={r2['lifecycle_reason'][:120]!r}")

        print("\n(i) DECLARING a lifecycle must not suppress a disk-fact "
              "warning  [B03]")
        # Found while diffing the report before/after appending B03's fields:
        # the missing-RELATED_WORK.md problem was raised only on the INFERENCE
        # path, so the early return for a DECLARED lifecycle skipped it. Adding
        # `"lifecycle": "ready_cpu"` to B03 therefore made a standing promotion
        # blocker disappear from the report -- paperwork retiring an unmet
        # requirement by changing one field, which is the exact failure mode
        # ready_queue.py's header names.
        r = read_fixture(tmp, "declared-no-rw", base_doc(lifecycle="ready_cpu"),
                         related_work=False)
        ok_i1 = check(
            "a DECLARED ready_cpu with no RELATED_WORK.md still reports it",
            any("RELATED_WORK.md absent" in p for p in r["problems"]),
            f"problems={r['problems']}")
        r2 = read_fixture(tmp, "declared-dead-no-rw", base_doc(lifecycle="dead"),
                          related_work=False)
        ok_i2 = check(
            "so does a DECLARED terminal state (the warning is about disk, not "
            "about lifecycle)",
            any("RELATED_WORK.md absent" in p for p in r2["problems"]),
            f"problems={r2['problems']}")
        r3 = read_fixture(tmp, "declared-with-rw", base_doc(lifecycle="ready_cpu"),
                          related_work=True)
        ok_i3 = check(
            "and it is NOT raised when the file exists (not a blanket warning)",
            not any("RELATED_WORK.md absent" in p for p in r3["problems"]),
            f"problems={r3['problems']}")

        # Guard the invariant the whole file exists for: a well-papered document
        # with no blocker SHOULD reach ready_gpu, or the blocker check has just
        # become a blanket refusal and the tests above would pass vacuously.
        print("\n(j) a blocker nested in a dated disposition wrapper is BINDING")
        # Shape copied from A02's real STATUS.json. MEASURED 2026-08-15: with the
        # clause ONLY nested, and RELATED_WORK.md present, ready_queue.py reported
        # `1 ready_gpu` -- i.e. it offered a proposal whose own record says
        # "NO further A02 GPU". `_walk_blockers` scanned top level only.
        r_j = read_fixture(tmp, "A02-like", base_doc(
            disposition_2026_08_12={
                "verdict": "CLOSED. Storage form dead; read-compute form a 1.03-1.37x micro-opt.",
                "gpu_policy": ("NO further A02 GPU. Resurrection requires a NEW MECHANISM, "
                               "not another read-out of the same ladder."),
            },
        ), related_work=True)
        with open(os.path.join(tmp, "A02-like", "STATUS.json"), encoding="utf-8") as f:
            doc_j = json.load(f)
        paths_j = {p for p, _ in ready_queue._walk_blockers(doc_j)}
        ok_j1 = check("nested disposition_*.gpu_policy is seen by _walk_blockers",
                      "disposition_2026_08_12.gpu_policy" in paths_j,
                      f"paths={sorted(paths_j)}")
        ok_j2 = check("a nested closing gpu_policy keeps it OUT of ready_gpu",
                      r_j["lifecycle"] != "ready_gpu",
                      f"lifecycle={r_j['lifecycle']}")

        print("\n(k) the nested blocker lookup must NOT fire on unrelated wrappers")
        # Same clause text under a key that is not allow-listed: it must stay
        # invisible, else the fix degenerates into a blind deep walk that reads
        # prose merely MENTIONING a blocker as a live blocker -- the over-report
        # failure mode, which strands work exactly like the under-report one.
        r_k = read_fixture(tmp, "unrelated-nest", base_doc(
            some_narrative_note={"gpu_policy": "we once considered stopping GPU here"},
        ), related_work=True)
        with open(os.path.join(tmp, "unrelated-nest", "STATUS.json"), encoding="utf-8") as f:
            doc_k = json.load(f)
        paths_k = {p for p, _ in ready_queue._walk_blockers(doc_k)}
        ok_k1 = check("gpu_policy under a non-allow-listed key is NOT a blocker",
                      "some_narrative_note.gpu_policy" not in paths_k,
                      f"paths={sorted(paths_k)}")
        ok_k2 = check("and that proposal is still dispatchable",
                      r_k["lifecycle"] == "ready_gpu",
                      f"lifecycle={r_k['lifecycle']}")

        print("\n(l) a FREE next gate outranks complete paperwork")
        # MEASURED on B06 2026-08-15: appending `related_work_status: "audited"`
        # -- pure bookkeeping -- flipped ready_cpu -> ready_gpu while its own
        # record said the next leg is 0 GPU and its kill condition 1 was
        # "PARTIALLY TESTABLE FROM DISK NOW, and it is the one at real risk".
        r_l = read_fixture(tmp, "free-next-gate", base_doc(
            related_work_status="audited",
            next_gate_gpu=("The drift leg is 0 GPU if the canonical predictions are "
                           "already on disk (rejudge = API + CPU). Only the "
                           "replication and second-compressor legs need GPU."),
        ), related_work=True)
        ok_l1 = check("a 0-GPU next gate is held in ready_cpu, not promoted",
                      r_l["lifecycle"] == "ready_cpu",
                      f"lifecycle={r_l['lifecycle']}")
        ok_l2 = check("and the reason names the free gate, not a paperwork gap",
                      "costs no GPU" in r_l["lifecycle_reason"],
                      f"reason={r_l['lifecycle_reason'][:140]!r}")
        ok_l3 = check("the hold is flagged as NOT a paperwork deficiency",
                      any("not a paperwork deficiency" in p for p in r_l["problems"]),
                      f"problems={r_l['problems']}")

        print("\n(m) a genuinely GPU-costing next gate is still ready_gpu")
        # The mirror case: this fix must not silently park everything in
        # ready_cpu, which would recreate the under-report stall it exists to
        # avoid. Same doc, cost string says the next step needs a card.
        r_m = read_fixture(tmp, "paid-next-gate", base_doc(
            related_work_status="audited",
            next_gate_gpu="1 node x 8 H20 for ~4 GPU-h; no free leg exists.",
        ), related_work=True)
        ok_m1 = check("a paid next gate remains dispatchable as ready_gpu",
                      r_m["lifecycle"] == "ready_gpu",
                      f"lifecycle={r_m['lifecycle']}")
        ok_m2 = check("_next_gate_is_free returns '' for a paid gate",
                      ready_queue._next_gate_is_free(
                          {"next_gate_gpu": "1 node x 8 H20 for ~4 GPU-h"}) == "",
                      "expected empty string")

        print("\n(control) a fully specified, unblocked proposal IS ready_gpu")
        r = read_fixture(tmp, "clean", base_doc(), related_work=True)
        ok_ctl = check("control reaches ready_gpu (checks are not vacuous)",
                       r["lifecycle"] == "ready_gpu",
                       f"got {r['lifecycle']!r}: {r['lifecycle_reason'][:120]}")

        # And that main()'s print/sort tables know every bucket inference emits.
        print("\n(control) every inferable lifecycle has a print bucket")
        src = open(os.path.join(ready_queue.ROOT, "ready_queue.py"),
                   encoding="utf-8").read()
        ok_ctl2 = check(
            "needs_prior_gate appears in both the sort order and the print loop",
            src.count('"needs_prior_gate"') >= 3,
            f"occurrences={src.count('\"needs_prior_gate\"')}")

        del ok_a1, ok_a2, ok_a3, ok_b1, ok_b2, ok_b3, ok_b4
        del ok_c1, ok_c2, ok_c3, ok_c4, ok_c5, ok_c6
        del ok_d1, ok_d2, ok_d3, ok_d4
        del ok_e1, ok_e2, ok_e3, ok_ctl, ok_ctl2
        del ok_f1, ok_f2, ok_f3, ok_f4, ok_f5
        del ok_g1, ok_g2, ok_g3, ok_g4, ok_g5, ok_g6, ok_g7, ok_g8, ok_g9
        del ok_h1, ok_h2
        del ok_i1, ok_i2, ok_i3
        del ok_j1, ok_j2, ok_k1, ok_k2
        del ok_l1, ok_l2, ok_l3, ok_m1, ok_m2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'=' * 66}")
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} assertions PASS")
    if n_fail:
        print("FAILED:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}: {detail}")
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
