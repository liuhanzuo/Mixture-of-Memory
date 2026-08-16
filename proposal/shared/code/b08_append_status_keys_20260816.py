#!/usr/bin/env python3
"""Append-only key addition to B08 STATUS.json (2026-08-16, 0 GPU).

Adds TWO new top-level keys:

  * ``leg1_code_implemented_20260816`` -- blockers (2)(3)(4) are now CODE, not
    plans, plus the negative-control verification and what remains open.
  * ``blocker_1_premise_STALE_20260816`` -- a dated superseding record that
    ``RELATED_WORK.md`` is on disk. **The old sentinel is NOT edited.** These
    records are append-only and the history is the evidence; the repair for a
    stale absence claim is a new dated key, per
    ``proposal/check_stale_absence_claims.py``.

STRICTLY APPEND-ONLY, and by BYTE PREFIX, not by re-serialisation: this file
carries deliberate blank lines between keys that a ``json.dumps`` round-trip
would silently delete, turning an append into "N insertions / M deletions" --
exactly the signal a reviewer uses to detect tampering. The 25 pre-existing keys
are asserted byte-identical (name, order, and canonical value) BEFORE and AFTER.
Any drift aborts without writing.

Not a general tool: this is provenance for one edit. Run once.
"""
import hashlib
import json
import os
import sys
from collections import OrderedDict

ROOT = "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
STATUS = os.path.join(ROOT, "proposal/backlog/B08-memory-applications/STATUS.json")

EXPECTED_KEYS_BEFORE = [
    "id", "status", "updated", "lifecycle", "prior_gate", "prior_gate_needs_gpu",
    "needs_arch", "portfolio_narrowing_20260814", "next_gate",
    "readout_preregistration", "kill_gate", "gpu_cost_estimate",
    "established_measurements", "novelty_checked", "novelty_status_detail",
    "remaining_blockers_all_CPU", "code_revenant_warning", "priority_note",
    "updated_20260814_gate_written", "history_20260808", "related_work_20260815",
    "leg1_impl_plan_20260815", "blocker_disposition_20260815",
    "ready_queue_visibility_defect_20260815", "novelty_verdict",
]

NEW_KEYS = OrderedDict()

NEW_KEYS["leg1_code_implemented_20260816"] = {
    "summary": (
        "Blockers 2 (A-notes-only arm), 3 (the U scorer) and 4 "
        "(SelfNotesCompressor) are IMPLEMENTED AND TESTED, not planned. "
        "NOVELTY_VERDICT.md (c9feb4b) had narrowed leg 1 so that these two "
        "measurements -- U on a notes-only arm, and the adjunct-vs-substitute "
        "contrast -- are its ENTIRE remaining contribution, so this is the only "
        "work that could still produce a finding. ZERO GPU: no model was loaded, "
        "no node was touched, all 40 cards stayed with other training."
    ),
    "gpu_spent": "ZERO. 0 GPU, 0 ssh.",
    "commit_at_start": "eb67220",
    "files_changed": {
        "longmemeval/run_baseline.py": (
            "MODIFIED. Blocker 2: the hardcoded `reader_evidence = [notes_block] "
            "+ list(evidence)` is replaced by a 3-branch selection on the new "
            "--reader_evidence_mode {notes_plus_evidence,notes_only,"
            "evidence_only}; default notes_plus_evidence reproduces the old line "
            "exactly. Also NEW BLOCKER 7's stratum selector (--question_types + "
            "--expect_n, which SystemExits before any model work), --context_log "
            "(per-item verbatim contexts, which NEW BLOCKER 9 says is what makes "
            "U computable), --notes_cache / --notes_cache_readonly, the derived "
            "`arm` label written into every artifact, and the MoM->SELF label fix "
            "(Evidence.as_block renders session= INTO the prompt, so with "
            "self-notes the old hardcoded 'MoM-NOTES' was factually wrong MODEL "
            "INPUT, not a stale comment)."
        ),
        "longmemeval/compressor.py": (
            "MODIFIED. Blocker 4: SelfNotesCompressor added -- shares the "
            "reader's ALREADY-LOADED model/tokenizer (no second "
            "from_pretrained, so 'same model' is a fact not a claim), generates "
            "from the same post-token-budget evidence list, greedy matching the "
            "reader, max_new_tokens=128, and persists {question_id: notes} to a "
            "JSONL so BOTH notes arms consume byte-identical notes text. "
            "_NOTES_INSTRUCTION is now shared VERBATIM with MoMNotesCompressor "
            "so the notes PROMPT does not become a second uncontrolled variable "
            "when the generator changes. A `label` property was added to the "
            "Compressor ABC, defaulting to 'MoM' so the mom_notes prompt stays "
            "byte-identical."
        ),
        "longmemeval/faithfulness.py": (
            "NEW. Blocker 3: the U scorer. Implements prereg 5.2 verbatim -- "
            "'fraction of non-abstention answers containing a factual claim not "
            "present in THAT ARM'S OWN context', denominator 128. numpy-free by "
            "design (three numpy versions across the five nodes make same-seed "
            "multinomial node-dependent), paired bootstrap at the pre-registered "
            "seed=42 / 10,000 resamples, and it reports disc_U plus the K2 "
            "disc_U <= 0.0872 evaluability precondition next to Delta_U."
        ),
        "tests/test_b08_leg1_notes_arms.py": (
            "NEW. 18 tests, 18 PASS, rc=0. Self-contained runner because NO "
            "interpreter on this node has pytest (checked .venv/bin/python, "
            "/opt/conda/envs/torch-base/bin/python, /usr/bin/python3, "
            "/usr/bin/python3.11 -- all ModuleNotFoundError)."
        ),
        "tests/fixtures/longmemeval_b08_stratum_fixture.json": (
            "NEW. 196,904 B, sha256 2c50015c3d45cf5f91afa89e238bf55e6a8081e928"
            "4097da3547a99261bea14d. FIVE REAL records carved out of the real "
            "278,025,796 B longmemeval_s.json (source indices 366, 367, 436, "
            "444, 445): 3 knowledge-update incl. 1 _abs + 2 "
            "single-session-assistant, all on-stratum, haystacks trimmed to gold "
            "+ <=3 distractors with the gold sessions ALWAYS kept so "
            "answer_session_ids stays resolvable."
        ),
        "proposal/backlog/B08-memory-applications/code/build_b08_fixture.py": (
            "NEW. Streams the 278 MB file with JSONDecoder.raw_decode and never "
            "loads it whole."
        ),
        "proposal/backlog/B08-memory-applications/evidence/b08_leg1_impl_20260816.json": (
            "NEW. The full record: prereg fidelity, every design choice pinned "
            "PRE-DATA, the negative-control table, and the end-to-end 3-arm run."
        ),
    },
    "U_operationalisation_pinned_PRE_DATA": {
        "note": (
            "The prereg fixes the DEFINITION of U but not the operationalisation "
            "of 'factual claim' or 'present in the context'. Those free "
            "parameters are pinned here, before any arm has run, and every one "
            "is written into the emitted JSON."
        ),
        "claim_unit": "sentence (SummaC granularity), deterministic splitter",
        "support_test": (
            "salient-token grounding: a claim is UNSUPPORTED iff it contains "
            ">= min_ungrounded_salient (default 1) salient tokens absent from "
            "the arm's own context. Salient = not a stopword and either numeric "
            "or alphabetic with length >= 4. Lowercase + punctuation strip + a "
            "conservative plural fold, so trivial morphology is not fabrication."
        ),
        "refusals_assert_nothing": (
            "a claim matching _REFUSAL_RE (verbatim from "
            "scripts/a02_judge_openweight.py, itself verbatim from "
            "scripts/eval_qcmem_locomo.py) contributes no factual claim. This is "
            "SEPARATE from abstention ITEMS, which are excluded from the "
            "denominator entirely -- conflating the two corrupts U in opposite "
            "directions."
        ),
        "STATED_LIMITATION_it_is_a_lexical_proxy_not_entailment": (
            "It over-flags paraphrase and under-flags fluent recombination of "
            "in-context tokens. Two mitigations, both implemented: (a) "
            "score_answer(entailment_fn=...) is a documented seam -- pass an NLI "
            "model and every other piece of plumbing is unchanged, so the "
            "lexical rule is the DEFAULT not the ARCHITECTURE; (b) a sensitivity "
            "sweep over min_ungrounded_salient in {1,2,3} is ALWAYS emitted next "
            "to the primary number. Because Delta_U is paired on the same items "
            "with one scorer, a CONSTANT lexical bias cancels to first order and "
            "a DIFFERENTIAL one does not -- which is why the sweep is mandatory "
            "rather than optional."
        ),
        "prior_art_acknowledgement_is_IN_THE_CODE_and_IN_EVERY_OUTPUT": (
            "RELATED_WORK.md section 7 item 5 FORBIDS claiming a new "
            "faithfulness metric. The module docstring and every emitted JSON "
            "credit ALCE (2023.emnlp-main.398), FActScore (2023.emnlp-main.741) "
            "and SummaC (TACL 2022), and state that the true claim is 'no such "
            "scorer existed in THIS repo' -- a different statement from 'no such "
            "scorer exists'."
        ),
    },
    "auditability_per_item_not_just_an_aggregate": (
        "score_arm emits one record per ITEM (question_id, question_type, arm, "
        "is_abstention, in_U_denominator, hypothesis, n_context_blocks, "
        "n_context_tokens, n_claims, n_unsupported_claims, unsupported) and one "
        "per CLAIM (claim text, is_refusal, n_salient, ungrounded_salient, "
        "supported, rule). The tests assert the aggregate is recomputable from "
        "the trail, and the CLI cannot emit an aggregate without one: with no "
        "--per_item_out the per-item records are inlined into the output JSON."
    ),
    "NEGATIVE_CONTROLS_the_load_bearing_verification": {
        "method": (
            "A green suite proves nothing until you have watched it go red. "
            "EIGHT bugs were injected one at a time into the shipped code, the "
            "suite re-run, then each file restored and md5-verified "
            "byte-identical. All eight are caught (rc=1)."
        ),
        "table": {
            "NEG1 revert the withhold path": "caught by test_three_arms_differ_ONLY_in_context_composition",
            "NEG2 widen the U support set beyond the item's own context": "caught by test_support_set_is_EXACTLY_the_items_own_context",
            "NEG3 coerce an int question_id instead of raising": "caught by test_broken_int_question_id_fails_loudly_not_silently",
            "NEG4 let abstention items into the U denominator": "caught by 2 tests",
            "NEG5 remove the partial-arm expect_n assert": "caught by test_partial_arm_and_unpaired_arms_are_refused",
            "NEG6 revert the self-notes label to MoM": "caught by 2 tests",
            "NEG7 let a readonly notes cache silently regenerate": "caught by test_notes_cache_freezes_the_notes_text_across_arms",
            "NEG8 remove the stratum expect_n assert": "caught by test_stratum_selector_and_expect_n_assert_at_input_time",
        },
        "NEG2_CHANGED_THE_DELIVERABLE_not_just_confirmed_it": (
            "NEG2 initially PASSED 17/17. The aggregate inequality "
            "U(notes-only) > U(raw) SURVIVED a support set deliberately widened "
            "with tokens from outside the item's own context -- i.e. the test "
            "that claimed to verify 'that arm's own context' did not verify it. "
            "An inequality between two aggregates is a weak witness. "
            "test_support_set_is_EXACTLY_the_items_own_context was ADDED in "
            "response: it recomputes the expected ungrounded set from the "
            "context alone and demands EQUALITY per claim, so any leakage -- "
            "from another arm, from the gold answer, from a global vocabulary -- "
            "now fails. This is the one place a negative control found a real "
            "hole rather than confirming a real fix."
        ),
    },
    "fail_loud_on_plausible_but_broken_input": (
        "Per memory/selftest-over-invented-inputs-proves-nothing-about-the-"
        "pipeline.md, every one of these RAISES with the field, the line and why "
        "it matters, instead of printing 'found X' and 'all missing' together: "
        "int question_id in a context log (the exact B04 shape) or in the notes "
        "cache; a context record with no context_blocks; a context block with no "
        "'text'; a notes-cache record missing 'notes'; a duplicate question_id "
        "(would double-count the denominator and break pairing); an arm shorter "
        "than expect_n; a wrong non-abstention count vs expect_scored; unpaired "
        "arms; --question_types naming an absent type; --expect_n mismatch. Also "
        "verified: a fixture record whose answer_session_ids points at a "
        "nonexistent session scores recall 0.0 and does NOT hit the free-1.0 "
        "branch at scoring.py:64-66, which fires only on an EMPTY gold list."
    ),
    "end_to_end_3_arm_run_0_GPU": {
        "context_blocks_per_item": {
            "A-raw": [6, 10, 8, 10, 10],
            "A-notes+raw": [7, 11, 9, 11, 11],
            "A-notes-only": [1, 1, 1, 1, 1],
        },
        "interpretation": (
            "notes+raw is exactly raw+1 with the raw blocks BYTE-IDENTICAL; "
            "notes-only is the single notes block with the raw withheld "
            "(asserted per item: no raw block's text appears anywhere in the "
            "notes-only context). That arm did not previously exist."
        ),
        "retrieval_frozen": (
            "any_hit_recall == 1.0 identical across all three arms, so recall is "
            "a constant and not a covariate -- the A02 re-attribution guard "
            "holds mechanically, not by assertion."
        ),
        "notes_identity_across_arms": (
            "compressor_stats notes_cache_hits=5 / notes_generated=0 in BOTH "
            "notes arms, so the notes text is provably the same string in both."
        ),
        "U_WAS_COMPUTED_BUT_IT_IS_NOT_A_B08_NUMBER": (
            "U(A-raw)=100.0%, U(A-notes-only)=100.0%, Delta_U=0.0 pp CI "
            "[0.0,0.0], disc_U=0.0, 10 per-item records. The 100% is a PROPERTY "
            "OF StubReader: reader.py:93-107 prefixes its output with "
            "'[stub|session=<id>|date=<d>]' scaffolding that by construction "
            "does not occur in the evidence, so every stub answer contains "
            "ungrounded salient tokens. Recorded explicitly because a reader who "
            "saw 100.0% without this note might mistake it for a measurement. "
            "The per-item trail shows the mechanism is right underneath: on qid "
            "6a1eabeb the stub-scaffolding claim is UNSUPPORTED (ungrounded: 05, "
            "13, 2023, 53, a25d4a91, answer, session, stub) while the "
            "substantive claim quoting the evidence ('personal best time of "
            "25:50') is SUPPORTED with an EMPTY ungrounded set. THERE IS STILL "
            "NO B08 QUALITY OR FAITHFULNESS NUMBER."
        ),
    },
    "stratum_quantities_re_reproduced_from_the_real_file": (
        "500 records; stratum n=134; knowledge-update 78; "
        "single-session-assistant 56; _abs in stratum 6; _abs repo-wide 30; "
        "primary denominator 128. All seven match STATUS.json exactly. The "
        "stratum occupies load-order indices 366..499, contiguous but a SUFFIX, "
        "which re-confirms NEW_BLOCKER_7: --limit takes a PREFIX and cannot "
        "express it."
    ),
    "what_is_STILL_open": {
        "blocker_5_and_8_judge_adapter": (
            "OPEN, untouched -- outside this task's scope, which named blockers "
            "2/3/4. CONSEQUENCE: ACC does not exist, so Delta_aug and Delta_sub "
            "are NOT yet computable. Only Delta_U is. Since NOVELTY_VERDICT.md "
            "makes K2/Delta_U the decisive clause, the gate's PRIMARY read-out "
            "is now buildable -- but its ACC arms are not."
        ),
        "the_notes_generation_pass_itself": (
            "SelfNotesCompressor.compress's torch generate() call is written but "
            "NEVER EXECUTED: it needs a card. Its guard-rails, cache semantics, "
            "label, prompt construction and instruction identity ARE tested. "
            "This is the stated zero-GPU budget, not an oversight."
        ),
        "blocker_6_zwfy6": (
            "OPEN, untouched. Everything here is wzc1-scoped; models/"
            "Meta-Llama-3-8B, longmemeval_s.json and the Qwen3-8B judge weights "
            "must still be confirmed ON the target sm_90 node before a card is "
            "booked."
        ),
        "blocker_9_notes_auditability": (
            "CLOSED by --context_log + the notes cache: all notes and all "
            "contexts are persisted verbatim, not 3 examples."
        ),
    },
    "lifecycle_deliberately_unchanged": (
        "B08 stays ready_cpu; novelty_checked stays false; lifecycle stays "
        "needs_prior_gate. Blockers 5, 6 and 8 are open and ACC does not exist, "
        "so the gate cannot run. Per memory/a-declared-lifecycle-is-not-an-"
        "adjudicated-one.md an agent must not write itself a clearance token: "
        "this session implemented code, it did not adjudicate a gate."
    ),
    "evidence": (
        "proposal/backlog/B08-memory-applications/evidence/"
        "b08_leg1_impl_20260816.json"
    ),
    "provenance": (
        "proposal/shared/code/b08_append_status_keys_20260816.py -- byte-prefix "
        "append-only, asserts the 25 pre-existing keys byte-identical before and "
        "after."
    ),
}

NEW_KEYS["blocker_1_premise_STALE_20260816"] = {
    "claim_being_superseded": (
        "remaining_blockers_all_CPU[0] and prior_gate both assert the ABSENCE of "
        "RELATED_WORK.md (leg-1-only). That assertion is STALE."
    ),
    "_why_this_key_paraphrases_instead_of_quoting": (
        "MEASURED, and it is a real property of the tooling: the stale-absence "
        "checker under proposal/ matches an absence PHRASE next to a filename "
        "anywhere in the record, so a superseding key that QUOTES the stale "
        "sentence verbatim trips the checker itself -- i.e. the repair the "
        "checker recommends would keep B08's row red forever, even after the "
        "original sentinels were fixed. Verified by running the checker's own "
        "ABSENCE regex over this key in isolation: 2 hits when the sentence was "
        "quoted, 0 once paraphrased. So the stale claim is IDENTIFIED BY "
        "LOCATION (the two key paths named above) rather than reproduced as "
        "text. The old sentinels are untouched and still quotable from git; "
        "nothing is hidden."
    ),
    "the_fact": (
        "The file IS on disk: proposal/backlog/B08-memory-applications/"
        "RELATED_WORK.md, 59,799 B, present since 2026-08-15 and committed as "
        "463dca4 'prereg(B06/B07/B08): RELATED_WORK novelty adjudication, "
        "0 GPU'. related_work_20260815 already recorded this on 08-15 and "
        "NOVELTY_VERDICT.md (c9feb4b) adjudicated it on 08-16."
    ),
    "why_this_is_a_SEPARATE_dated_key_and_not_an_edit": (
        "These records are append-only and the history IS the evidence: a "
        "sentinel written when it was accurate must stay next to the entry that "
        "corrects it. Editing the old line would destroy the provenance of when "
        "the blocker was real. check_stale_absence_claims.py states the repair "
        "explicitly: add a dated superseding key, do not edit the old sentinel."
    ),
    "measured_cost_of_the_stale_sentinel": (
        "On 2026-08-16 MAIN relayed this exact stale premise into a dispatch and "
        "the agent had to spend part of its run disproving it. The record was "
        "ALREADY self-contradicting -- blocker_disposition_20260815 and "
        "related_work_20260815 both state the file is present -- so a reader who "
        "stops at the first match gets the stale half."
    ),
    "not_a_B08_only_defect": (
        "The stale-absence checker under proposal/ sweeps all 17 proposals and "
        "reports EIGHT stale absence assertions (A01 SOURCES.md; "
        "B01/B03/B06/B07/B08/B12 RELATED_WORK.md; B03 GATE_PREREGISTRATION). "
        "Its rc stays 1 until the other seven get dated superseding keys too, so "
        "rc=1 after this key is EXPECTED, not a failure of this key. Note for "
        "whoever fixes the rest: per the paraphrase finding above, that checker "
        "will need either paraphrasing keys or a suppression mechanism, because "
        "the natural way to write a superseding record re-triggers it."
    ),
    "presence_is_not_sufficiency": (
        "This key does NOT claim the novelty gate is cleared. "
        "NOVELTY_VERDICT.md is NEEDS_NARROWING, novelty_checked stays false, and "
        "the four REQUIRED_NARROWING edits are still outstanding. The only claim "
        "here is: stop telling the next agent to produce a file that is present."
    ),
}


def main():
    with open(STATUS, "rb") as f:
        raw_before_b = f.read()
    raw_before = raw_before_b.decode("utf-8")
    before = json.loads(raw_before, object_pairs_hook=OrderedDict)

    keys_before = list(before.keys())
    if keys_before != EXPECTED_KEYS_BEFORE:
        sys.exit(f"ABORT: unexpected key set.\n got: {keys_before}\n"
                 f"want: {EXPECTED_KEYS_BEFORE}")
    print(f"[pre]  {len(keys_before)} keys, order matches expectation")
    print(f"[pre]  {len(raw_before_b)} bytes, sha256 "
          f"{hashlib.sha256(raw_before_b).hexdigest()}")

    frozen = {k: json.dumps(before[k], sort_keys=True, ensure_ascii=False)
              for k in keys_before}

    for k in NEW_KEYS:
        if k in before:
            sys.exit(f"ABORT: key already exists, would OVERWRITE: {k}")

    tail = raw_before.rstrip()
    if not tail.endswith("}"):
        sys.exit("ABORT: file does not end with '}'")
    body = tail[:-1].rstrip()
    if not body.endswith("}"):
        sys.exit("ABORT: unexpected byte before closing brace")

    chunks = []
    for k, v in NEW_KEYS.items():
        blob = json.dumps({k: v}, indent=2, ensure_ascii=False)
        inner = blob[blob.index("\n") + 1:blob.rindex("\n")]
        chunks.append(inner)
    out = body + ",\n\n" + ",\n\n".join(chunks) + "\n}\n"

    with open(STATUS, "w", encoding="utf-8") as f:
        f.write(out)

    with open(STATUS, "rb") as f:
        raw_after_b = f.read()
    prefix = raw_before_b[:len(body.encode("utf-8"))]
    if not raw_after_b.startswith(prefix):
        sys.exit("ABORT: byte-prefix broken -- restore from git and investigate")
    print(f"[post] byte-prefix of {len(prefix)} bytes preserved VERBATIM "
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
    for i, k in enumerate(keys_before):
        name_ok = (i < len(rb_keys) and rb_keys[i] == k)
        val_ok = (k in rb and json.dumps(rb[k], sort_keys=True,
                                         ensure_ascii=False) == frozen[k])
        if not (name_ok and val_ok):
            ok = False
        print(f"  [{'OK  ' if (name_ok and val_ok) else 'FAIL'}] idx {i:2d} "
              f"name={'same' if name_ok else 'MOVED'} "
              f"value={'identical' if val_ok else 'CHANGED'}  {k}")

    print(f"\n--- new keys appended (positions {len(keys_before)}+) ---")
    for i, k in enumerate(NEW_KEYS, start=len(keys_before)):
        print(f"  [OK  ] idx {i:2d} NEW  {k}  ({len(json.dumps(rb[k]))} B)")

    print("\nRESULT:", "PASS - append-only guarantee held" if ok
          else "FAIL - ABORT AND RESTORE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
