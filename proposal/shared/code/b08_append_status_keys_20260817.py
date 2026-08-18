#!/usr/bin/env python3
"""Append-only key addition to B08 STATUS.json (2026-08-17, 0 GPU).

Adds THREE new top-level keys, all dated 20260817:

  * ``novelty_verdict_20260817`` -- the DATED superseding novelty verdict. This is
    the only mechanical way the four REQUIRED_NARROWING edits become visible to
    ``ready_queue.py``: it resolves dated keys newest-first via
    ``_DATED_NOVELTY_RE`` (``ready_queue.py:389``), so the bare ``novelty_verdict``
    with its NEEDS_NARROWING string is *left completely untouched* and is simply
    outranked. Editing it would destroy the provenance of when the narrowing was
    genuinely outstanding.
  * ``required_narrowing_applied_20260817`` -- what landed in RELATED_WORK.md,
    site by site, with the two dispatch corrections and the one self-correction.
  * ``judge_adapter_spec_20260817`` -- the field-by-field spec for the REAL
    remaining blocker (5+8). ACC does not exist without it, so ``Delta_aug`` and
    ``Delta_sub`` are not computable even after a card is booked. Writing the spec
    is the 0-GPU half of that work.

VERDICT-STRING ENGINEERING (measured, not guessed)
==================================================
``ready_queue.py:869-873`` tests ``VERDICT_PENDING`` **first** and ``break``s, so a
verdict string containing *any* of ``needs_narrowing / unchecked / not_checked /
todo`` stays PENDING even if it also contains a cleared token. The verdict string
below is therefore checked programmatically for BOTH properties before writing:
it must contain >=1 ``VERDICT_CLEARED`` token and ZERO ``VERDICT_PENDING`` tokens.

Note the trap: substring matching is case-insensitive on a ``.lower()``ed string
and is not word-bounded, so ``"clear"`` also matches inside ``"unclear"`` and
``"todo"`` matches inside e.g. ``"todos"``. The guard below runs the reader's exact
predicate over the whole nested dict, not just the verdict field.

WHAT THIS KEY DOES **NOT** DO
=============================
It does NOT authorise a GPU. Measured, not assumed: ``lifecycle`` stays
``needs_prior_gate`` and ``prior_gate_needs_gpu`` stays ``false``, and
``ready_queue.py:1085-1101`` honours the DECLARED lifecycle over an inferred
``ready_gpu``, folding it to ``ready_cpu``. So B08 remains 0-GPU-held by its own
prior gate. The script asserts this from a /tmp copy AFTER writing, rather than
claiming it.

STRICTLY APPEND-ONLY BY BYTE PREFIX, not by re-serialisation -- a
``json.load``/``json.dump`` round-trip would reformat the whole file and destroy the
byte-prefix audit that the last two appends established. The 28 pre-existing keys
are asserted byte-identical (name, order, canonical value) before and after.

0 GPU, 0 ssh. Not a general tool. Run once.
"""
import hashlib
import json
import os
import sys
from collections import OrderedDict

ROOT = "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
STATUS = os.path.join(ROOT, "proposal/backlog/B08-memory-applications/STATUS.json")

SHA_BEFORE = "bcad641d1a57e4dff56244e61190e83c353fbb6880503460f4476266888d66a9"
BYTES_BEFORE = 76716

EXPECTED_KEYS_BEFORE = [
    "id", "status", "updated", "lifecycle", "prior_gate", "prior_gate_needs_gpu",
    "needs_arch", "portfolio_narrowing_20260814", "next_gate",
    "readout_preregistration", "kill_gate", "gpu_cost_estimate",
    "established_measurements", "novelty_checked", "novelty_status_detail",
    "remaining_blockers_all_CPU", "code_revenant_warning", "priority_note",
    "updated_20260814_gate_written", "history_20260808", "related_work_20260815",
    "leg1_impl_plan_20260815", "blocker_disposition_20260815",
    "ready_queue_visibility_defect_20260815", "novelty_verdict",
    "leg1_code_implemented_20260816", "blocker_1_premise_STALE_20260816",
    "related_work_presence_correction_20260816",
]

# The reader's own token lists (ready_queue.py:399-401), duplicated here ONLY to
# self-check the verdict string before writing. Not imported, because importing
# the scheduler to validate a write to the file the scheduler reads would couple
# the two; a drift in the reader must show up as a queue diff, not be silently
# absorbed here.
VERDICT_CLEARED = ("hold_in_backlog", "gate cleared", "clear", "pass",
                   "audited", "no candidate preempts", "not preempted")
VERDICT_PENDING = ("needs_narrowing", "unchecked", "not_checked", "todo")

NEW_KEYS = OrderedDict()

NEW_KEYS["novelty_verdict_20260817"] = {
    "verdict": (
        "NARROWING APPLIED -- gate cleared, not preempted, HOLD IN BACKLOG. "
        "SUPERSEDES the bare `novelty_verdict` key (2026-08-16), which is left "
        "BYTE-UNTOUCHED as provenance -- see this key's `supersedes` field for "
        "which verdict string it carried, deliberately NOT quoted here (see "
        "`verdict_string_engineering_MEASURED`): its "
        "four REQUIRED_NARROWING_four_edits_0_GPU items were genuinely "
        "outstanding when written and are now applied to RELATED_WORK.md "
        "(section 13, plus in-place markers at 9 live sites). The novelty "
        "question -- 'is leg 1 preempted?' -- is answered NO on the narrowed "
        "claim: nothing found is the same work, all six 2026 hits are "
        "concurrent (2026-05..2026-07 vs the 2026-08-14 prereg) and all are "
        "arXiv-only. The residual claim is now a SINGLE decisive read-out: "
        "Delta_U, the unsupported-claim rate on a notes-only arm, on a "
        "retrieval-closed stratum that is a PRECONDITION rather than a "
        "contribution, with a mandatory single-reader scope clause on any "
        "Delta_aug / Delta_sub number."
    ),
    "when": "2026-08-17",
    "gpu_spent": "ZERO. 0 GPU, 0 ssh, 0 network. No node was touched.",
    "supersedes": (
        "`novelty_verdict` (2026-08-16), whose verdict string opened with the "
        "not-yet-narrowed token (spelled with an underscore; not reproduced here "
        "because ready_queue.py:869 would then read THIS key as pending too, and "
        "proposal/check_stale_absence_claims.py has the same quoting hazard). "
        "NOT EDITED, NOT DELETED. Resolved "
        "newest-dated-first by ready_queue.py:392 novelty_verdict_keys, so this "
        "key outranks it mechanically without touching a byte of it."
    ),
    "what_this_verdict_is_NOT": (
        "It is NOT GPU authorisation, and it is NOT a claim that the gate is "
        "runnable. A novelty verdict answers 'is this preempted?', never 'is it "
        "worth a card now?'. The gate is in fact STILL NOT RUNNABLE, for a "
        "reason unrelated to novelty: ACC does not exist (see "
        "judge_adapter_spec_20260817), so Delta_aug and Delta_sub are not "
        "computable. Delta_U alone IS computable once the arms run."
    ),
    "gpu_authorisation": (
        "NOT AUTHORISED. MEASURED after writing, not assumed: on a /tmp copy of "
        "this file WITH this key present, ready_queue.read_one returns "
        "lifecycle='ready_cpu' (NOT ready_gpu), lifecycle_declared="
        "'needs_prior_gate', novelty_checked=True. ready_queue.py:1085-1101 "
        "honours the DECLARED needs_prior_gate over an inferred ready_gpu and "
        "folds it to ready_cpu only because prior_gate_needs_gpu is false. So "
        "flipping the novelty token CANNOT mis-dispatch a card; B08 stays held "
        "by its own prior gate. `lifecycle` and `novelty_checked` were "
        "deliberately NOT touched by this append."
    ),
    "four_edits_where_they_landed": {
        "1_STRIKE_the_retrieval_pinning_differentiator": (
            "RELATED_WORK.md section 3.3, the arXiv:2607.17545 row: differentiator "
            "(a) is now prefixed with a red STRUCK/FALSE marker and wrapped in "
            "markdown ~~strikethrough~~ IN PLACE, so a reader landing in section 3 "
            "sees the strike. Section 12.1 had already DECLARED it struck on "
            "08-16, but 426 lines below the live sentence, which is why the "
            "sentence kept reading as current."
        ),
        "2_DEMOTE_stratum_to_precondition": (
            "Markers at four sites: section 2's 'Is the measurement prior art?' "
            "cell, section 3.2 (RECOMP row), section 6 ('Leg 1 = WE DID TOO "
            "LITTLE'), section 8's one-line-each paragraph. Section 3.1 already "
            "had the correct framing ('a constraint, not a win'), so edit 2 was "
            "partly a consistency repair between sections."
        ),
        "3_RE_ANCHOR_on_Delta_U": (
            "Marker on section 3.1's PASS disjunction plus a full paragraph under "
            "section 8's blockquote. See dispatch_correction_2 -- the site named "
            "in the dispatch was wrong."
        ),
        "4_single_reader_scope_clause": (
            "New marked paragraph directly under section 8's residual-claim "
            "blockquote, which is where NOVELTY_VERDICT.md 5.1 item 4 says it "
            "must go. Verified genuinely absent beforehand: grep over section 8 "
            "(lines 291-322 pre-edit) for reader|competence|Llama-3-8B|scope|"
            "2606.21807 returned ZERO hits, and arXiv:2606.21807 occurred exactly "
            "once in the whole 59,799-byte file, in section 12.2's table."
        ),
    },
    "dispatch_correction_1_RECOMP_is_not_ROC": (
        "The dispatch that commissioned this round said the falsified sentence is "
        "live 'at line 100 (ROC row) and an identical one at :82 (RECOMP row)' "
        "and should get the same treatment. THEY ARE NOT EQUIVALENT. Line 100 "
        "says arXiv:2607.17545 never pins retrieval, and that is FALSE on ROC's "
        "own quoted Setup. Line 82 says the same of RECOMP (ICLR 2024), and NO "
        "ROUND HAS READ RECOMP'S PROTOCOL. Marking line 82 STRUCK/FALSE would "
        "have manufactured a finding about RECOMP from an unrelated paper's text. "
        "It therefore carries a DEMOTION marker with the non-verification stated "
        "inline. Adjudicating RECOMP's row on the merits is a NEW literature task."
    ),
    "dispatch_correction_2_edit3_site": (
        "The dispatch said edit 3 was unlanded because 'section 8's residual "
        "sentence still gives Delta_aug equal billing'. IT DOES NOT: section 8's "
        "blockquote already reads 'Delta_aug ... does not require notes to beat "
        "raw for the claim to hold', i.e. Delta_U was already sole load-bearing "
        "there. The site that still granted Delta_aug co-decisive standing is "
        "section 3.1's 'PASS on Delta_aug CI > 0 OR Delta_U CI > +5.0 pp' "
        "disjunction, and its source next_gate.decidable_outcome, which is worded "
        "the same way. An agent that 'fixed' section 8 would have rewritten a "
        "correct sentence and left the actual disjunction standing."
    ),
    "new_defect_found_and_then_SELF_CORRECTED": (
        "While placing edit 3 I found the Delta_U PASS threshold written "
        "differently in different places, and first recorded it (section 13.5) as "
        "'three non-equivalent spellings'. I then EXECUTED the sentences as "
        "predicates over (point, lo, hi) on a 108-point grid and my own reading "
        "was wrong twice: (i) kill_gate K2 (hi < 5.0) is a KILL-side clause, not "
        "a PASS spelling -- putting it in a PASS table is a units error; (ii) the "
        "count is TWO, not three, and I had failed to check "
        "B08_LEG1_GATE_PREREG.md 5.6, the pre-registration of record, which "
        "AGREES with next_gate and section 3.1. Corrected in section 13.7; "
        "section 13.5 is marked RETRACTED IN PART and left otherwise unedited. "
        "THE REAL DEFECT, which survives the correction and is smaller and "
        "better localised: prereg 5.6 + next_gate.decidable_outcome + section 3.1 "
        "all say `CI lower bound > +5.0 pp` (3 concordant), while "
        "RELATED_WORK.md section 8's blockquote says `point > +5.0 AND lo > 0` "
        "-- STRICTLY WEAKER and the LONE OUTLIER. Worked, executed: Delta_U=+6.0 "
        "pp CI[+1.0,+11.0] -> section 8 says PASS, prereg/next_gate/3.1 say FAIL, "
        "K2 does not fire. Control: on prereg 5.7's own plausible-PASS figures "
        "(+11.4, [+6.2,+16.9]) all four forms agree PASS, and on its "
        "plausible-KILL figures (+0.7, [-2.6,+4.1]) all agree not-a-pass. The "
        "divergence is confined to lo in (0, 5.0] with point > 5.0. ALSO "
        "RETRACTED from 13.5: I framed the divergent case as 'the proposal has no "
        "verdict' and implied that was itself a defect -- wrong; a two-sided gate "
        "with an inconclusive middle band is normal and is what kill_gate already "
        "specifies. NOTE THE DIRECTION OF BOTH SELF-CORRECTIONS: each one made MY "
        "OWN finding smaller, which is the direction that does not get written "
        "down by default (memory/state-direction-only-for-rows-you-computed.md). "
        "REPAIR IS THE OWNER'S, PRE-DATA: make section 8 quote prereg "
        "5.6 by reference. UNTIL THEN, PREREG 5.6 GOVERNS."
    ),
    "kill_gate_untouched": (
        "kill_gate.conditions_KILL_iff_ALL_THREE (K1+K2+K3) and the disc_U <= "
        "0.0872 evaluability precondition stand exactly as pre-registered. A "
        "literature count MAY NOT fire a kill gate -- only the proposal's own "
        "experiment may (proposal/README.md; "
        "memory/prior-work-differentiate-dont-abandon.md). What changed is only "
        "which clause is DECISIVE (K2 / Delta_U), not any threshold."
    ),
    "still_open_and_NOT_closed_by_this_round": [
        "remaining_blockers_all_CPU[5] + NEW_BLOCKER_8: the judge input adapter. "
        "0 GPU to write, and until it exists ACC does not exist. Spec: "
        "judge_adapter_spec_20260817.",
        "remaining_blockers_all_CPU[6]: the zwfy6 cross-disk asset check. "
        "CANNOT be closed from this node: /apdcephfs_zwfy6 is NOT MOUNTED (ls -d "
        "-> No such file or directory; mount lists only /apdcephfs_wzc1 and "
        "/apdcephfs_wzc1_304376610) and ssh was barred. Every presence claim in "
        "the B08 files remains wzc1-scoped. Needs 1 ssh, 0 GPU.",
        "RECOMP's retrieval protocol has never been read (see "
        "dispatch_correction_1). 0 GPU, network only.",
        "The section-8-vs-prereg-5.6 Delta_U threshold reconciliation is an "
        "OWNER decision and must be made PRE-DATA.",
        "The 2026.findings-acl venue index was truncated at 1,343,488 / 6,675,019 "
        "B on a 120 s timeout in the 08-16 round, so that negative still rests on "
        "DBLP alone (RELATED_WORK.md section 12.6)."
    ],
    "verdict_string_engineering_MEASURED": (
        "ready_queue.py:869-873 tests VERDICT_PENDING BEFORE VERDICT_CLEARED and "
        "breaks, so a verdict string containing the not-yet-narrowed token "
        "anywhere -- even while merely DESCRIBING what it supersedes -- stays "
        "PENDING. THIS IS NOT HYPOTHETICAL: the first draft of this key said "
        "'SUPERSEDES the bare novelty_verdict key (2026-08-16, <that token>)' and "
        "the writer's pre-write guard ABORTED with 'verdict string contains a "
        "PENDING token'. Nothing was written; the string was paraphrased. Without "
        "that guard this key would have been appended and the queue would have "
        "kept printing (PENDING) while this file claimed the gate was cleared. "
        "Same hazard, same shape, as blocker_1_premise_STALE_20260816's finding "
        "that quoting a stale absence sentence verbatim RE-TRIPS "
        "check_stale_absence_claims.py (2 hits when quoted, 0 when paraphrased). "
        "The rule: a superseding record must PARAPHRASE the sentinel it "
        "supersedes, never quote it. The verdict string was checked "
        "programmatically before writing -- >=1 VERDICT_CLEARED token, ZERO "
        "VERDICT_PENDING tokens -- and the whole nested key was probed too, "
        "because the reader's 'only look at the verdict field' behaviour is a "
        "fragile property of today's reader, not a guarantee. That probe reports "
        "PENDING for the whole-key blob (other fields of this key legitimately "
        "discuss the narrowing), so if a future reader ever blobs the whole key, "
        "it will regress -- recorded here rather than left as a landmine."
    ),
}

NEW_KEYS["required_narrowing_applied_20260817"] = {
    "summary": (
        "The four REQUIRED_NARROWING edits are APPLIED to RELATED_WORK.md, both "
        "as an appended section 13 AND as in-place pointer markers at 9 live "
        "sites. The in-place half is the point: on 2026-08-16 section 12.1 "
        "appended a paragraph saying the falsified differentiator 'is hereby "
        "STRUCK' and the sentence stayed live, verbatim and unmarked, 426 lines "
        "above it. A second appended declaration would have looked like progress "
        "and changed nothing a reviewer reads "
        "(memory/agent-output-must-be-persisted-to-the-consumers-file.md)."
    ),
    "gpu_spent": "ZERO. 0 GPU, 0 ssh, 0 network.",
    "writers": [
        "proposal/shared/code/b08_apply_required_narrowing_20260817.py "
        "(8 markers + section 13; 59,799 -> 73,348 B)",
        "proposal/shared/code/b08_fix_struck_dependent_sentence_20260817.py "
        "(1 marker + 13.1 addendum; 73,348 -> 74,838 B)",
        "proposal/shared/code/b08_selfcorrect_threshold_count_20260817.py "
        "(3 retraction markers + section 13.7; 74,838 -> 79,543 B)",
    ],
    "INSERT_ONLY_and_it_is_PROVEN_not_asserted": (
        "Sections 11 and 12 declare the preceding sections byte-stable, but edit 1 "
        "says STRIKE, which is not an append -- the mandate and the convention are "
        "in direct conflict. Resolved by INSERTING ONLY: zero deletions, zero "
        "rewordings, every original byte surviving in original order. Each writer "
        "PROVES this rather than claiming it: it strips exactly the inserted "
        "marker strings from the result and asserts the remainder is "
        "byte-identical to its input. Chain of hashes, all verified: 59,799 B "
        "sha256 83fda7862862e8cd182077e1c042c7c9d201db7b93d5dd671dd7da34e52af5b4 "
        "-> 73,348 B 3219395b2bdcbb80bc20ae90b93d437da68e98407a650f0bc363a2a0737d1e48 "
        "-> 74,838 B 274fa83c9cbfeb2f4781885fdb303a9729628fa5f3ea4f0825e6605dfecf9843 "
        "-> 79,543 B 805ae3604ee4d4d896e80e5b11bab85c9a1e750a508b3dd125bc4210c2b276a7. "
        "So the original adjudication is fully recoverable -- which is what "
        "byte-stability protected -- while section 3 now shows the strike, which "
        "is what the mandate required. Every anchor was asserted to occur EXACTLY "
        "ONCE before any write; a count of 0 or 2 aborts."
    ),
    "consequence_sweep_found_a_9th_site": (
        "Striking a sentence is not finished until the sentences that CITE it are "
        "reconciled (memory/fix-the-class-not-the-instance.md). A regex sweep of "
        "sections 1-11 for `\\(a\\)\\+\\(b\\)|shrinks to|differentiator|"
        "retrieval-closed isolation` returned exactly TWO live hits: section 3.2's "
        "RECOMP row (already marked) and the ROC row's own conclusion, *'Leg 1's "
        "residual claim ... shrinks to (a)+(b)'* -- which still cited the struck "
        "(a). Left alone, a reader trusting the summary line would have restored "
        "the falsified conjunct and OVERSTATED the residual claim. It now carries "
        "a marker: the claim shrinks to (b) ALONE. No third dependent statement "
        "exists in sections 1-11."
    ),
    "what_a_reader_landing_in_section_3_now_sees": (
        "Line 100: 'What it does **not** do: [RED] **(a) IS STRUCK - FALSE. See "
        "13.1: ROC's Setup supplies GOLD EVIDENCE to every arm, i.e. recall 1.000 "
        "by construction. Struck text retained below for provenance:** ~~(a) it "
        "never pins retrieval at a **measured** `any_hit = 1.000`, ... chosen to "
        "remove;~~ (b) ...'. That is the read-out that was missing: the strike is "
        "now visible at the site, not only in an appendix."
    ),
    "supersedes_nothing_it_only_discharges": (
        "This key does not contradict any earlier key. novelty_verdict's "
        "REQUIRED_NARROWING list was correct; this key records that the list is "
        "now discharged. The two dispatch corrections (see "
        "novelty_verdict_20260817) correct the DISPATCH, not any STATUS.json key."
    ),
}

NEW_KEYS["judge_adapter_spec_20260817"] = {
    "why_this_is_the_real_blocker": (
        "With novelty narrowed and the leg-1 code shipped (see "
        "leg1_code_implemented_20260816), the gate's remaining obstruction is "
        "NOT paperwork and NOT novelty: ACC does not exist. Two of the three "
        "pre-registered read-outs (Delta_aug, Delta_sub) are ACC differences, so "
        "they are not computable even after a card is booked. Delta_U is "
        "UNAFFECTED -- longmemeval/faithfulness.py consumes the --context_log "
        "records, which already carry every field it needs -- and Delta_U is now "
        "the DECISIVE clause, so a run would not be worthless without ACC. But "
        "the gate as pre-registered has three clauses and only one would be "
        "scoreable."
    ),
    "gpu_spent": "ZERO. This key is a spec, not a run. Writing the adapter is 0 GPU.",
    "verified_absent_on_wzc1": (
        "No scripts/*b08*, no longmemeval/*judge*, no longmemeval/*adapter*. "
        "ls-checked, not assumed. wzc1-scoped only (zwfy6 is not mounted here)."
    ),
    "the_impedance_mismatch_field_by_field": {
        "producer": (
            "longmemeval/scoring.py:30-43 write_submission emits ONLY "
            "{question_id, hypothesis} per line -- it is deliberately the "
            "upstream LongMemEval GPT-4o evaluate_qa.py format (scoring.py:5-9)."
        ),
        "consumer": (
            "scripts/a02_judge_openweight.py:187-201 load_preds globs "
            "preds*.jsonl and keys each record on item['id']; the judge then "
            "reads item['pred'] (:336), item['question'] + item['answers'] "
            "(:127-128), item['category'] (:354), item['is_abstention'] (:335)."
        ),
        "missing_fields": [
            "id            <- rename of question_id (the judge does NOT read question_id)",
            "pred          <- rename of hypothesis",
            "question      <- re-join from data/longmemeval/longmemeval_s.json",
            "answers       <- re-join; note the source field is the SINGULAR "
            "'answer' (a str, or an int for 2 of the 134 stratum items) while the "
            "judge expects a LIST -> wrap as [str(answer)]",
            "category      <- map from question_type (used only for by-category "
            "reporting, but absent -> everything lands in the '?' bucket)",
            "is_abstention <- DERIVE: LongMemEval marks abstention questions by a "
            "'_abs' SUFFIX ON question_id. MEASURED on disk: 30 of 500 items "
            "repo-wide, and exactly 6 of the 134-item stratum (all 6 in "
            "knowledge-update, all with 2 gold sessions and a gold answer "
            "beginning 'The information provided is not enough.'). This matters "
            "for the read-out, not just for plumbing: the judge SHORT-CIRCUITS "
            "abstention items at :335-338 and scores them by refusal-regex "
            "instead of by the LLM judge. Getting is_abstention wrong silently "
            "moves 6/134 items between two different scoring rules."
        ],
        "filename_convention": (
            "load_preds globs preds*.jsonl, so the adapter's output must be named "
            "preds*.jsonl (e.g. preds_A-notes-only.jsonl) inside a per-arm dir. "
            "A submission written as s.jsonl will raise FileNotFoundError."
        ),
    },
    "per_arm_result_dirs_are_MANDATORY_not_cosmetic": (
        "judge_cache_openweight.jsonl is keyed on 'id' ALONE (:327, :339-340, "
        ":352) with NO arm field. The three arms answer the SAME 134 "
        "question_ids, so a shared cache dir would make arm 2 read arm 1's cached "
        "judgement for every item -- silently producing Delta_aug = 0.000 and "
        "Delta_sub = 0.000 with no error. Each arm MUST get its own --result_dirs "
        "entry AND its own cache dir. This is the highest-risk item in the spec "
        "because its failure mode is a plausible-looking null result, not a crash."
    ),
    "an_adapter_alone_is_not_enough_ACC_needs_a_card": (
        "The adapter itself is 0 GPU. Running the judge is NOT: it loads a "
        "Qwen3-8B judge model. So the honest split is: write + unit-test the "
        "adapter on CPU now (0 GPU, and testable against the existing fixture "
        "tests/fixtures/longmemeval_b08_stratum_fixture.json); the judge pass "
        "itself is part of the ~1.15 GPU-h estimate (gpu_cost_estimate already "
        "budgets 0.045 GPU-h for the judge)."
    ),
    "suggested_selftest_that_would_actually_catch_something": (
        "Per memory/selftest-over-invented-inputs-proves-nothing-about-the-"
        "pipeline.md, do NOT write a selftest over hand-built dicts. Feed it (a) "
        "a real write_submission output, (b) the real longmemeval_s.json join, "
        "and (c) two INJECTED bad-but-plausible cases: a submission whose "
        "question_ids are a subset of the stratum (must FAIL LOUDLY, not "
        "silently score n<134), and two arms pointed at the SAME cache dir (must "
        "be REFUSED). Assert n_scored == 134 per arm, and assert the 6 "
        "'_abs' items took the abstention branch."
    ),
    "does_not_touch_the_prereg": (
        "This is a plumbing spec. It changes no threshold, no arm definition and "
        "no read-out. If writing it reveals that an ACC read-out is infeasible, "
        "that is a finding for the OWNER, not a licence to redefine Delta_aug."
    ),
}


def _reader_predicate(blob_lower):
    """Run ready_queue.py:869-873's exact test. Returns 'PENDING'/'CLEARED'/'UNPARSED'."""
    if any(s in blob_lower for s in VERDICT_PENDING):
        return "PENDING"
    if any(s in blob_lower for s in VERDICT_CLEARED):
        return "CLEARED"
    return "UNPARSED"


def main():
    with open(STATUS, "rb") as f:
        raw_before_b = f.read()
    raw_before = raw_before_b.decode("utf-8")
    sha = hashlib.sha256(raw_before_b).hexdigest()
    print(f"[pre]  {len(raw_before_b)} bytes, sha256 {sha}")
    if len(raw_before_b) != BYTES_BEFORE or sha != SHA_BEFORE:
        sys.exit(f"ABORT: unexpected pre-state.\n  want {BYTES_BEFORE} B / "
                 f"{SHA_BEFORE}\n  got  {len(raw_before_b)} B / {sha}")

    before = json.loads(raw_before, object_pairs_hook=OrderedDict)
    keys_before = list(before.keys())
    if keys_before != EXPECTED_KEYS_BEFORE:
        sys.exit(f"ABORT: unexpected key set.\n got: {keys_before}\n"
                 f"want: {EXPECTED_KEYS_BEFORE}")
    print(f"[pre]  {len(keys_before)} keys, order matches expectation")

    for k in NEW_KEYS:
        if k in before:
            sys.exit(f"ABORT: key already exists, would OVERWRITE: {k}")

    # --- Verdict-string self-check, BEFORE writing. ---
    vstr = NEW_KEYS["novelty_verdict_20260817"]["verdict"]
    vl = vstr.lower()
    print("\n--- verdict string vs the reader's own predicate ---")
    hits_p = [s for s in VERDICT_PENDING if s in vl]
    hits_c = [s for s in VERDICT_CLEARED if s in vl]
    print(f"  VERDICT_PENDING hits: {hits_p or 'NONE (required)'}")
    print(f"  VERDICT_CLEARED hits: {hits_c or 'NONE (this would FAIL)'}")
    verdict_result = _reader_predicate(vl)
    print(f"  reader would classify this as: {verdict_result}")
    if hits_p:
        sys.exit("ABORT: verdict string contains a PENDING token; "
                 "ready_queue tests PENDING first and would keep B08 PENDING.")
    if not hits_c:
        sys.exit("ABORT: verdict string contains no CLEARED token; it would be "
                 "reported UNPARSED.")
    # Fragility probe: the reader only reads the 'verdict' field today. Confirm
    # what WOULD happen if a future reader blobbed the whole key.
    whole = json.dumps(NEW_KEYS["novelty_verdict_20260817"],
                       ensure_ascii=False).lower()
    print(f"  [probe] if a future reader blobbed the WHOLE key it would say: "
          f"{_reader_predicate(whole)}  <- fragile-by-design, recorded in the key")

    frozen = {k: json.dumps(before[k], sort_keys=True, ensure_ascii=False)
              for k in keys_before}

    tail = raw_before.rstrip()
    if not tail.endswith("}"):
        sys.exit("ABORT: file does not end with '}'")
    body = tail[:-1].rstrip()
    if not body.endswith("}"):
        sys.exit("ABORT: unexpected byte before closing brace")

    chunks = []
    for k, v in NEW_KEYS.items():
        blob = json.dumps({k: v}, indent=2, ensure_ascii=False)
        chunks.append(blob[blob.index("\n") + 1:blob.rindex("\n")])
    out = body + ",\n\n" + ",\n\n".join(chunks) + "\n}\n"

    with open(STATUS, "w", encoding="utf-8") as f:
        f.write(out)

    with open(STATUS, "rb") as f:
        raw_after_b = f.read()
    prefix = raw_before_b[:len(body.encode("utf-8"))]
    if not raw_after_b.startswith(prefix):
        sys.exit("ABORT: byte-prefix broken -- restore from git and investigate")
    print(f"\n[post] byte-prefix of {len(prefix)} bytes preserved VERBATIM "
          f"({len(raw_before_b)} -> {len(raw_after_b)} bytes)")

    with open(STATUS, encoding="utf-8") as f:
        rb = json.load(f, object_pairs_hook=OrderedDict)

    ok = True
    want_n = len(keys_before) + len(NEW_KEYS)
    print(f"\n[post] {len(rb)} keys on disk (expect {len(keys_before)}+"
          f"{len(NEW_KEYS)}={want_n})")
    if len(rb) != want_n:
        ok = False
        print("  FAIL: key count")

    print(f"\n--- ASSERT: {len(keys_before)} pre-existing keys unchanged ---")
    rb_keys = list(rb.keys())
    n_bad = 0
    for i, k in enumerate(keys_before):
        name_ok = (i < len(rb_keys) and rb_keys[i] == k)
        val_ok = (k in rb and json.dumps(rb[k], sort_keys=True,
                                         ensure_ascii=False) == frozen[k])
        if not (name_ok and val_ok):
            ok = False
            n_bad += 1
            print(f"  [FAIL] idx {i:2d} name={'same' if name_ok else 'MOVED'} "
                  f"value={'identical' if val_ok else 'CHANGED'}  {k}")
    print(f"  {len(keys_before) - n_bad}/{len(keys_before)} identical in name, "
          f"order and value" + ("" if n_bad == 0 else f"  ({n_bad} FAILED)"))

    print(f"\n--- new keys appended (positions {len(keys_before)}+) ---")
    for i, k in enumerate(NEW_KEYS, start=len(keys_before)):
        print(f"  [OK  ] idx {i:2d} NEW  {k}  ({len(json.dumps(rb[k]))} B)")

    # The old sentinel must still be byte-identical AND still say NEEDS_NARROWING.
    old = rb["novelty_verdict"]["verdict"]
    old_intact = old.startswith("NEEDS_NARROWING")
    print(f"\n[post] bare `novelty_verdict` still says NEEDS_NARROWING "
          f"(provenance preserved): {old_intact}")
    if not old_intact:
        ok = False

    print("\nRESULT:", "PASS - append-only guarantee held" if ok
          else "FAIL - ABORT AND RESTORE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
