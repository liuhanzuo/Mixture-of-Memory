#!/usr/bin/env python3
"""Negative tests for proposal/ready_queue.py's gate-detection and blocker logic.

0 GPU. stdlib only. Run directly:

    python proposal/test_ready_queue_gates.py

Why these five cases and not others
-----------------------------------
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

        # Guard the invariant the whole file exists for: a well-papered document
        # with no blocker SHOULD reach ready_gpu, or the blocker check has just
        # become a blanket refusal and the tests above would pass vacuously.
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
