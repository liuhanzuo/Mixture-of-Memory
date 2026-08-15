#!/usr/bin/env python3
"""Append-only key addition to B08 STATUS.json (2026-08-15, 0 GPU).

Adds three NEW top-level keys recording (a) the RELATED_WORK.md independent
venue re-verification, (b) the leg-1 A-notes-only implementation plan +
mom_notes asset verdict, (c) the disposition of the four CPU blockers.

STRICTLY APPEND-ONLY. The 20 pre-existing keys are asserted byte-identical
(name, order, and json.dumps of value) BEFORE and AFTER. Any drift aborts
without writing.

Not a general tool: this is provenance for one edit. Run once.
"""
import json
import hashlib
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
    "updated_20260814_gate_written", "history_20260808",
]

NEW_KEYS = OrderedDict()

NEW_KEYS["related_work_20260815"] = {
    "file": "proposal/backlog/B08-memory-applications/RELATED_WORK.md",
    "PREMISE_OF_THE_TASK_WAS_STALE": (
        "blocker 1 said RELATED_WORK.md does not exist. IT DOES. Written "
        "2026-08-15 07:14 (39,604 B) and already COMMITTED as 463dca4 "
        "'prereg(B06/B07/B08): RELATED_WORK novelty adjudication, 0 GPU'. "
        "git status for the B08 directory was clean. So this session did NOT "
        "write the adjudication -- it INDEPENDENTLY RE-VERIFIED it and appended "
        "section 11. Recorded because 'the file is absent' was the stated reason "
        "B08 could not be ready by construction, and that reason had already "
        "been discharged by another agent hours earlier."
    ),
    "what_was_added_this_session": (
        "RELATED_WORK.md section 11 (APPENDED; sections 1-10 left byte-stable). "
        "A second agent re-verified every load-bearing citation from a live "
        "network via proxy hy-proxy.woa.com:3128."
    ),
    "verification_result_ZERO_fabrications": (
        "6/6 ACL-family Anthology IDs fetched HTTP 200 with titles matching the "
        "claims verbatim (2024.emnlp-main.813 Chain-of-Note; 2023.emnlp-main.398 "
        "ALCE; 2023.emnlp-main.741 FActScore; 2024.findings-acl.57 LLMLingua-2 "
        "-- the word 'Faithful' IS in the real title; 2024.acl-long.91 "
        "LongLLMLingua; 2023.emnlp-main.825 LLMLingua). 7/7 OpenReview venueids "
        "confirmed (RECOMP ICLR.cc/2024/Conference poster; Provence "
        "ICLR.cc/2025/Conference; xRAG + HippoRAG NeurIPS.cc/2024/Conference; "
        "G-Memory NeurIPS.cc/2025/Conference venue='NeurIPS 2025 spotlight'; "
        "LongMemEval ICLR.cc/2025/Conference; A-Mem NeurIPS.cc/2025/Conference). "
        "11/11 arXiv IDs resolve to the EXACT claimed titles with real "
        "<published> dates. 0 fabricated IDs, 0 title mismatches, 0 venue "
        "overstatements."
    ),
    "one_caveat_RESOLVED_and_it_went_AGAINST_leg_3": (
        "RELATED_WORK.md section 9.2 flagged MM-Mem's 'Accepted by ACL 2026 Main' "
        "as an arXiv COMMENT ONLY because an Anthology search returned nothing. "
        "ACL 2026 proceedings have since been indexed and it is now RESOLVED: "
        "aclanthology.org/events/acl-2026/ returns 573 '2026.acl' links; "
        "volumes 2026.acl-long / acl-short / findings-acl hold 605 / 76 / 662 "
        "papers; the exact title matches in the LONG volume only, at "
        "2026.acl-long.533 (HTTP 200, DOI 10.18653/v1/2026.acl-long.533, title "
        "verbatim 'From Verbatim to Gist: Distilling Pyramidal Multimodal Memory "
        "via Semantic Information Bottleneck for Long-Horizon Video Agents'). "
        "Consequence, exactly as 9.2 predicted: leg 3's foreclosure gets "
        "STRONGER -- the 'pyramidal' name and the verbatim->gist axis are held by "
        "a peer-reviewed ACL 2026 Main long paper, not a preprint. Any B08 .bib "
        "must cite 2026.acl-long.533, NOT arXiv:2603.01455. Cross-checked: NONE "
        "of Retain-or-Consolidate, Ground Truth First, MemConflict, "
        "MemSyco-Bench, LatticeMind, HMO, LongMemEval-V2, Nous, Pancake or "
        "From-Context-to-EDUs appears in any ACL 2026 volume, so leg 1's "
        "residual gap is NOT newly foreclosed."
    ),
    "section_9_3_CONFIRMED_not_pessimistic": (
        "Zep: DBLP publ/api q='Zep temporal knowledge graph memory' returns "
        "hits.@total = 1, 'Zep: A Temporal Knowledge Graph Architecture for "
        "Agent Memory.', CoRR 2025, type 'Informal and Other Publications'. So "
        "the file's self-flagged 'weakest venue attribution for a load-bearing "
        "paper' is ACCURATE. Zep stays arXiv-only. Harmless downstream because "
        "it is load-bearing for leg 2 (section 4.3) and leg 2 is FOLDED, not gated."
    ),
    "verdict_carried_forward": (
        "hold_in_backlog -- novelty gate CLEARED FOR LEG 1 ONLY and only for the "
        "narrowed sentence in RELATED_WORK.md section 8. LEG 2 stays FOLDED "
        "(never separately gated). LEG 3 recommended CUT on literature + our own "
        "adverse A02 measurement (1.03-1.37x fixed-Read advantage, N* 8->25->186) "
        "+ dead code. already_dead_should_archive: NO -- nothing is "
        "完全相同/抄袭, and the three most dangerous hits (2026-03, 2026-07, "
        "2026-07) are CONCURRENT with the 2026-08-14 prereg."
    ),
    "highest_value_remaining_literature_task_0_GPU": (
        "RELATED_WORK.md section 9.6: every 2026 paper was adjudicated from its "
        "ABSTRACT ONLY. The unresolved question that can still collapse leg 1 is "
        "whether 'Retain or Consolidate?' (arXiv:2607.17545) PINS RETRIEVAL. If "
        "it does, leg 1's residual claim -- which is precisely the "
        "retrieval-closed isolation -- goes with it. Read its method + evaluation "
        "sections BEFORE spending the 1.1 GPU-h, not after."
    ),
    "novelty_checked_field_deliberately_NOT_flipped": (
        "novelty_checked stays false. Measured on a /tmp copy: appending "
        "related_work_status='audited' flips ready_queue.py's novelty_checked to "
        "True (ready_queue.py:203-209) but B08 STAYS ready_cpu, held by "
        "prior_gate. Per memory/a-declared-lifecycle-is-not-an-adjudicated-one.md "
        "an agent writing its own clearance field is not the clearance being "
        "reviewed, and two of the remaining blockers ARE the novelty. Flipping it "
        "would buy no scheduling change and would cost an audit trail."
    ),
}

NEW_KEYS["leg1_impl_plan_20260815"] = {
    "file": "proposal/backlog/B08-memory-applications/LEG1_IMPL_PLAN.md",
    "scope": (
        "0 GPU, 0 ssh, PRE-DATA. Decides HOW the A-notes-only arm is built and "
        "whether the missing mem_space asset costs the gate an arm. NO SOURCE "
        "FILE WAS MODIFIED -- run_baseline.py was deliberately left untouched "
        "because it is a git-tracked shared harness file; the diff is specified, "
        "not applied."
    ),
    "blocker_2_verdict_FEW_LINES_NO_RIPPLE": (
        "A few lines, and it does not ripple. MEASURED blast radius of "
        "longmemeval/run_baseline.py = EMPTY: grep for "
        "'longmemeval.run_baseline|from .run_baseline' returns 4 hits, ALL inside "
        "run_baseline.py itself (its own docstring examples at :11,:15,:23 and its "
        "own prog= string at :264). Corroborated: longmemeval/__init__.py:26-38 "
        "re-exports data/backends/reader/scoring and deliberately NOT "
        "run_baseline; _apply_token_budget (:59) has exactly one call site (:142); "
        "tests/ has 19 files and 0 mention longmemeval; the only other "
        "longmemeval consumer scripts/eval_qcmem_longmemeval.py:130 imports "
        "longmemeval.data.load_longmemeval, not run_baseline. It is a CLI LEAF."
    ),
    "blocker_2_minimal_diff": (
        "(1) NEW FLAG after run_baseline.py:334: --reader_evidence_mode "
        "{notes_plus_evidence,notes_only,evidence_only}, default "
        "notes_plus_evidence. (2) REPLACE exactly one line, :162 "
        "'reader_evidence = [notes_block] + list(evidence)', with a 3-branch "
        "if/elif/else using getattr(args,'reader_evidence_mode', "
        "'notes_plus_evidence') so hand-built namespaces and the --self_test path "
        "(:350-356, which mutates args in place) keep working. (3) add "
        "'reader_evidence_mode': args.reader_evidence_mode to the report dict "
        "(:177-193) so the arm is recoverable from the artifact, not from shell "
        "history. TOTAL: +1 flag, +6 lines replacing 1, +1 report line. No "
        "signature change, no new import, and the default reproduces :162 "
        "byte-identically (and with --compressor none the branch is dead anyway "
        "because notes=='' per compressor.py:70-76)."
    ),
    "blocker_4_verdict_NO_ARM_IS_ASSET_LESS_KEEP_THREE_ARMS": (
        "The gate does NOT need to shrink to two arms, and no asset needs to be "
        "built. The third arm was NEVER mom_notes: "
        "next_gate.notes_generator_must_be_the_reader_itself already specifies "
        "SELF-notes from the reader's own Meta-Llama-3-8B, and rejects mem_space "
        "on SCIENTIFIC grounds (a second model would confound 'notes help' with "
        "'the mem_space model is good') independently of availability. So "
        "mom_notes being unrunnable removes an OPTION, not an ARM. All three arms "
        "run on models/Meta-Llama-3-8B (15G, wzc1-verified), which is already the "
        "reader. What is unbuilt is ~80 lines of harness glue, not a checkpoint."
    ),
    "blocker_4_asset_absence_RE_MEASURED": (
        "Confirmed and strengthened on wzc1: 45 adapter_config.json under "
        "outputs/; 0 contain num_slots or slot_dim; peft_type is LORA in 45/45 "
        "(no other value); 'ls -d outputs/*mem_space*' returns 0 directories; "
        "find for *.pt under any *mem_space* path returns 0 files repo-wide. All "
        "45 belong to five qcmem_distill_* LoRA families. CORROBORATION that this "
        "is real absence, not a search artifact: the two mem_space adapter "
        "configs still referenced by hardcoded path in repo scripts -- "
        "outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json "
        "(_launch_stairs_L5.sh:12, _launch_hnstv2_tree.sh:34, "
        "_eval_hnstv2_bundle.sh:51) and "
        "outputs/mem_space_perdoc_chunk128/adapter_config.json "
        "(eval_base_longbench_r33.sh:19) -- BOTH stat as MISSING. Several other "
        "scripts are therefore already broken against wzc1 for the same reason; "
        "B08 is not special-cased, the mem_space asset family is simply gone from "
        "this disk. The only num_slots strings left are inside BABILong RESULT "
        "jsons (tree_step2000/**, model.num_slots=128) -- records of past runs, "
        "not loadable configs."
    ),
    "self_notes_compressor_design_contract": (
        "SelfNotesCompressor in longmemeval/compressor.py (~60 lines) + one "
        "branch in build_compressor (:221-242) + 'self_notes' added to "
        "--compressor choices (run_baseline.py:329-331). Non-negotiables: (a) "
        "SHARE the reader's already-loaded weights (reader.py:206-210) rather "
        "than a second from_pretrained -- pass reader into build_compressor, whose "
        "signature change is safe because it has exactly ONE caller "
        "(run_baseline.py:132); (b) generate from the SAME post-_apply_token_budget "
        "evidence list, which is already where compress() is called (:153); (c) "
        "generate the notes ONCE per question, persist {question_id: notes} to "
        "JSONL, and have BOTH notes arms read that file -- 'same notes text' is "
        "part of the frozen single variable, and regenerating per arm would leak "
        "decoder nondeterminism into the contrast; (d) greedy, matching the reader "
        "(do_sample=False, num_beams=1, reader.py:282-289) with max_new_tokens=128; "
        "(e) reuse MoMNotesCompressor's instruction VERBATIM (compressor.py:181-185) "
        "so the notes prompt is not a new uncontrolled variable."
    ),
    "MoM_label_leaks_into_the_prompt_correctness_not_cosmetics": (
        "run_baseline.py:156-159 hardcodes session_id='MoM-NOTES' and "
        "text=f'MoM NOTES: {notes}'. Evidence.as_block (backends.py:47-53) renders "
        "session= INTO the prompt the model reads, so with a self-notes generator "
        "those strings are factually wrong MODEL INPUT, not a stale comment. They "
        "must become SELF-NOTES / 'SELF NOTES:' under --compressor self_notes or "
        "the arm's context misdescribes its own provenance. Cheapest correct form: "
        "a label property on the Compressor ABC defaulting to 'MoM', so the "
        "mom_notes path stays byte-identical."
    ),
    "measured_0_GPU_stratum_reproduces_exactly": (
        "Real Meta-Llama-3-8B tokenizer, real BM25 top_k=10, real "
        "evidence_token_budget=4000, LocalHFReader._build_prompt (reader.py:229-254) "
        "reimplemented line-for-line, CUDA_VISIBLE_DEVICES=''. stratum n=134 "
        "(knowledge-update 78 + single-session-assistant 56); _abs in stratum = 6 "
        "(6 in KU, 0 in SSA); _abs repo-wide = 30; primary denominator 128. ALL "
        "SEVEN quantities match STATUS.json exactly."
    ),
    "MEASURED_CORRECTION_notes_only_arm_is_14_5x_cheaper": (
        "gpu_cost_estimate.arithmetic books THREE answer arms at 529,620 prefill "
        "tokens each. MEASURED: A-raw and A-notes+raw are 529,620 (mean 3952.4, "
        "max 4609) but A-notes-only is 36,577 (mean 273.0, max 312) -- its context "
        "is a single <=128-token notes block, so it is ~14.5x cheaper. Re-running "
        "the same j0_top12 anchor (0.15180 ms/prefill-tok, 39.0021 ms/decode-tok, "
        "both re-verified against a02_storage_readcompute_verdict.json this "
        "session): A-raw 414.9 s, A-notes+raw 414.9 s, A-notes-only 340.0 s (was "
        "booked 414.7 s), notes-gen 749.4 s => core 1919.2 s = 0.5331 GPU-h, x2 "
        "slack 1.0662, + judge 0.045 = 1.111 GPU-h vs the booked 1.153. THE "
        "HEADLINE SURVIVES (both far under the 2 GPU-h ceiling) and the direction "
        "is favourable, so this is a PRECISION correction, not a re-plan. Recorded "
        "because an unexamined 14.5x error in a per-arm term is harmless at n=134 "
        "and NOT harmless after a K2 escalation to n=500."
    ),
    "MEASURED_GOOD_NEWS_notes_plus_raw_does_not_silently_truncate_raw": (
        "A real hazard, checked rather than assumed: "
        "LocalHFReader._truncate_evidence_block (reader.py:213-227) caps the "
        "evidence section at max_prompt_tokens - reserve, so a prepended notes "
        "block COULD have made the arm labelled 'notes prepended, raw kept' "
        "actually be 'raw partly deleted' -- which would falsify the "
        "single-variable claim. MEASURED: reserve=165, cap=6835, evidence-section "
        "tokens mean 3838.4 / p90 4278 / max 4505; items exceeding cap with RAW "
        "alone = 0/134; items exceeding cap AFTER prepending a ~150-token notes "
        "block = 0/134, with ~2330 tokens of headroom at the observed max. So the "
        "arm name is literally true at this config. WARNING: this is a property of "
        "the CONFIG, not the harness -- re-measure if --top_k, "
        "--evidence_token_budget, max_prompt_tokens or --compressor_max_new_tokens "
        "change, exactly like "
        "established_measurements.closure_RE_VERIFIED_at_the_gates_own_budget_20260814.standing_rule."
    ),
}

NEW_KEYS["blocker_disposition_20260815"] = {
    "summary": (
        "Of the 4 CPU blockers in remaining_blockers_all_CPU: #1 was ALREADY "
        "DISCHARGED before this session (RELATED_WORK.md exists, committed "
        "463dca4) and has now been independently re-verified; #2 and #4 are "
        "DECIDED (minimal diff specified, three-arm design confirmed intact, no "
        "asset needs building) but NOT YET IMPLEMENTED; #3 and #5 are untouched. "
        "THREE NEW blockers were found, and one of them makes the gate UNRUNNABLE "
        "as currently specified. Net: the blocker list did not get shorter by "
        "solving; it got MORE ACCURATE. B08 remains ready_cpu and must NOT be "
        "promoted to ready_gpu."
    ),
    "blocker_1_RELATED_WORK": "DISCHARGED before this session (463dca4); re-verified 2026-08-15, 0 fabrications. See related_work_20260815.",
    "blocker_2_notes_only_arm": "DECIDED, NOT IMPLEMENTED. ~18 lines in one leaf file, zero ripple. Diff in LEG1_IMPL_PLAN.md section 1.3.",
    "blocker_3_U_scorer": "OPEN, unchanged. Still the primary novelty metric and still absent from the tree. Must be built on ALCE / FActScore / SummaC machinery AND SAY SO (RELATED_WORK.md MUST-NOT-CLAIM item 5): the honest statement is 'no such scorer exists in THIS REPO', which is a different claim from 'no such scorer exists'.",
    "blocker_4_self_notes_compressor": "DECIDED, NOT IMPLEMENTED. ~80 lines. mom_notes confirmed dead but IRRELEVANT -- the arm was always self-notes. NO narrowing to two arms.",
    "blocker_5_judge_adapter": "OPEN and BIGGER than its one-line description. See NEW_BLOCKER_8 below.",
    "blocker_6_zwfy6_verification": "OPEN, unchanged, and deliberately LAST. /apdcephfs_zwfy6 is not mounted on this node and ssh was barred, so every asset claim in this file remains wzc1-scoped.",
    "NEW_BLOCKER_7_no_stratum_selector_GATE_IS_UNRUNNABLE_AS_WRITTEN": (
        "run_baseline.py has NO question-type filter (grep question_type -> only "
        "the per_type_recall grouping at :146 and two synthetic fixtures). --limit "
        "(:270-271) takes a PREFIX via load_longmemeval(path, limit=...) "
        "(data.py:138-141). MEASURED: the stratum occupies load-order indices "
        "366..499 -- contiguous, but a SUFFIX, so --limit cannot express it "
        "(--limit 134 selects the FIRST 134, a disjoint set). Running all 500 "
        "instead is NOT a substitute: closure FAILS off-stratum "
        "(single-session-preference any_hit 0.7667 -> 0.7000 at budget=4000, per "
        "this file's own closure re-verification), so an arm difference would "
        "become re-attributable to retrieval -- the exact A02 phase-1 failure mode "
        "the cell was chosen to avoid -- and it would 3.7x the cost (Sigma prefill "
        "2,010,736 vs 529,620) for no added information. FIX (~10 lines, same leaf "
        "file): --question_types knowledge-update,single-session-assistant filtered "
        "right after load_longmemeval (:361), PLUS --expect_n 134 raising SystemExit "
        "on mismatch. The assert is not decoration: it is the mechanical form of "
        "readout_preregistration.why_an_assert_not_a_nan_check applied at INPUT "
        "time, so a silently-wrong stratum cannot reach the scorer. Do NOT rely on "
        "--limit and do NOT hardcode exs[366:500] -- the slice is a property of "
        "today's file ordering, and data.py:163 also accepts JSONL."
    ),
    "NEW_BLOCKER_8_judge_adapter_is_a_field_loss_not_a_rename": (
        "a02_judge_openweight.py:187-201 load_preds() globs preds*.jsonl per "
        "--result_dirs entry and keys on item['id'] (hard KeyError if absent), then "
        "reads item['pred'] (:341,357), item['question'] (:355), "
        "item.get('answers',[]) (:356), item.get('category') (:354) and "
        "item.get('is_abstention',False) (:337). But write_submission "
        "(scoring.py:30-43) emits ONLY question_id + hypothesis into a SINGLE file "
        "and DISCARDS the gold answer, the question text, the question type and any "
        "abstention marker. So the adapter must RE-JOIN the submission against the "
        "source data to recover four fields, not rename two. TWO PRE-COMMITMENTS: "
        "(a) is_abstention MUST be set explicitly -- if it defaults False the 6 "
        "_abs items go to the semantic judge instead of the refusal rule "
        "(:337-340), which abstention_handling says corrupts BOTH ACC and U; they "
        "are identifiable by the _abs question_id suffix, and note they DO carry a "
        "gold answer string plus 2 gold sessions each, so a naive 'no gold answer "
        "=> abstention' heuristic would NOT find them. (b) EACH ARM NEEDS ITS OWN "
        "RESULT DIRECTORY: the judge writes judge_meta.json per dir (:291-293) and "
        "appends to judge_cache_openweight.jsonl (:296,346) keyed on 'id' ALONE "
        "with NO arm field, so a shared directory would make arm 2 silently reuse "
        "arm 1's verdicts -- a same-id-different-pred collision that raises no "
        "error and produces no NaN. THIS IS THE MOST DANGEROUS ITEM FOUND THIS "
        "SESSION. Also: the judge's _JUDGE_TEMPLATE (:48-63) names LoCoMo "
        "explicitly; recommendation is to KEEP IT VERBATIM (protocol identity "
        "beats the benchmark noun, and B08 has no archived judge numbers to be "
        "consistent with) and RECORD the choice in judge_meta.json PRE-DATA."
    ),
    "NEW_BLOCKER_9_minor_notes_are_not_auditable": (
        "run_baseline.py:163-166 caps notes_examples at THREE entries for the "
        "report. U is defined against 'that arm's own context', so scoring it "
        "requires ALL 134 notes strings verbatim. The notes-persistence file in the "
        "self_notes design contract is therefore not an optimisation -- it is what "
        "makes the primary metric computable. The 3-example report field is a "
        "debugging nicety, not provenance."
    ),
    "recommended_build_order_all_CPU": (
        "1) --reader_evidence_mode (~18 lines, ship alone, zero ripple); 2) "
        "SelfNotesCompressor + shared-weights seam + SELF-NOTES label fix (~80 "
        "lines); 3) NEW BLOCKER 7 stratum selector (~10 lines) -- cheaper than #2 "
        "but invisible until this session and the gate cannot run without it; 4) "
        "the U scorer; 5) judge adapter per NEW BLOCKER 8; 6) zwfy6 verification, "
        "LAST and only when a card is actually being booked."
    ),
    "lifecycle_unchanged_DELIBERATELY": (
        "lifecycle stays needs_prior_gate and novelty_checked stays false. This "
        "session cleared 0 of the 4 original blockers by implementation (blocker 1 "
        "was already closed by another agent) and ADDED 3. Promotion would be "
        "paperwork-counts-as-readiness, which is precisely what ready_queue.py "
        "exists to block. The gate also still needs ~1.1 GPU-h and a free card, "
        "and per CLAUDE.md all 40 cards are currently committed to 5 long trainings."
    ),
    "provenance": (
        "proposal/shared/code/b08_append_status_keys_20260815.py -- append-only, "
        "asserts the 20 pre-existing keys byte-identical before and after."
    ),
}

NEW_KEYS["ready_queue_visibility_defect_20260815"] = {
    "finding": (
        "B08's four CPU blockers are INVISIBLE to proposal/ready_queue.py. They "
        "live under the keys `prior_gate` and `remaining_blockers_all_CPU`, and "
        "ready_queue.py:252-253 hard-codes BLOCK_KEYS = ['blocking_dependency', "
        "'blocked_by', 'required_before_stage0', 'gpu_policy', "
        "'premise_falsified'] -- NEITHER of B08's keys is in that list. MEASURED "
        "via ready_queue.read_one() on this file: live_blockers = 0 and problems "
        "= 0, BOTH BEFORE AND AFTER this session's append."
    ),
    "why_this_matters": (
        "The scheduler has never known that B08 has unmet prerequisites. It "
        "classifies B08 ready_cpu purely because novelty_checked is false "
        "('novelty gate not adjudicated (absent) -> the actionable task is 0 GPU: "
        "run it'). That reason is now WRONG IN ITS PARTICULARS -- the "
        "adjudication was done and committed (463dca4) and re-verified here -- yet "
        "it lands on the right ANSWER (ready_cpu, not ready_gpu) for the wrong "
        "reason. If anyone later flips novelty_checked to true, B08 would jump "
        "toward ready_gpu with FIVE open CPU blockers (3, 5, 7, 8, 9) that the "
        "reader cannot see. That is the paperwork-counts-as-readiness failure "
        "ready_queue.py was written to prevent, arriving through a key-name gap "
        "instead of a logic gap."
    ),
    "second_defect_declared_lifecycle_is_ignored": (
        "STATUS.json declares lifecycle 'needs_prior_gate'. ready_queue.py REPORTS "
        "ready_cpu. Only dead / promoted / running / ready_cpu short-circuit as "
        "authoritative in the explicit-lifecycle block; 'needs_prior_gate' falls "
        "through to inference, which overrides it. BENIGN for GPU safety today "
        "(both values are != ready_gpu, so no card can be mis-dispatched) but it "
        "is a declared-vs-inferred mismatch on the field LIFECYCLE_SCHEMA.md calls "
        "'唯一的机器可读状态', and it is the same class of bug as the two "
        "under-reads ready_queue.py already documents fixing at :30-70."
    ),
    "recommended_fix_NOT_APPLIED": (
        "Add 'remaining_blockers_all_CPU' and 'prior_gate' to BLOCK_KEYS, and let "
        "'needs_prior_gate' short-circuit as authoritative like the other terminal "
        "and holding states. DELIBERATELY NOT DONE HERE: ready_queue.py is the "
        "shared scheduler read by every proposal, so changing BLOCK_KEYS would "
        "re-classify all 15 STATUS.json files at once -- that needs its own "
        "before/after diff across the whole queue, not a drive-by edit inside a "
        "B08 task. Per memory/fix-the-class-not-the-instance.md the right move is "
        "to fix the KEY FAMILY for every proposal in one reviewed change, and per "
        "memory/reporting-a-gap-is-not-closing-it.md this is filed as an explicit "
        "task rather than a caveat. Whoever takes it must diff "
        "`python3 proposal/ready_queue.py` before and after and confirm no "
        "proposal moves INTO ready_gpu."
    ),
    "consequence_for_the_commissioning_expectation": (
        "The task that produced this file expected B08 to go from '4 un-cleared "
        "blockers' to fewer in the ready_queue output. It never showed 4: it "
        "showed 0, before and after. So the queue count could not improve, and "
        "the honest read-out is that the blocker list got MORE ACCURATE (4 -> 4 "
        "decided/open + 3 newly found) rather than shorter. B08 correctly remains "
        "ready_cpu and must NOT be promoted."
    ),
}


def main():
    with open(STATUS, "rb") as f:
        raw_before_b = f.read()
    raw_before = raw_before_b.decode("utf-8")
    before = json.loads(raw_before, object_pairs_hook=OrderedDict)

    keys_before = list(before.keys())
    if keys_before != EXPECTED_KEYS_BEFORE:
        sys.exit(f"ABORT: unexpected key set.\n got: {keys_before}\nwant: {EXPECTED_KEYS_BEFORE}")
    print(f"[pre]  {len(keys_before)} keys, order matches expectation")
    print(f"[pre]  {len(raw_before_b)} bytes, sha256 "
          f"{hashlib.sha256(raw_before_b).hexdigest()}")

    frozen = {k: json.dumps(before[k], sort_keys=True, ensure_ascii=False)
              for k in keys_before}

    for k in NEW_KEYS:
        if k in before:
            sys.exit(f"ABORT: key already exists, would OVERWRITE: {k}")

    # ---------------------------------------------------------------------
    # BYTE-PREFIX APPEND, not a re-serialisation.
    #
    # A json.dumps() round-trip is semantically append-only but NOT
    # byte-append-only: this file carries 14 deliberate blank lines that a
    # prior author used for visual grouping, and re-dumping silently deletes
    # them. `git diff --numstat` then reads 39 insertions / 14 DELETIONS on an
    # append-only file -- which is exactly the signal a reviewer uses to detect
    # tampering. So instead of re-dumping, splice the new keys in as text after
    # the final value and before the closing brace, leaving every pre-existing
    # byte untouched. The result: `git diff` shows insertions ONLY.
    # ---------------------------------------------------------------------
    tail = raw_before.rstrip()
    if not tail.endswith("}"):
        sys.exit("ABORT: file does not end with '}'")
    body = tail[:-1].rstrip()          # drop the final closing brace
    if not body.endswith("}"):         # last value is history_20260808, an object
        sys.exit("ABORT: unexpected byte before closing brace")

    chunks = []
    for k, v in NEW_KEYS.items():
        blob = json.dumps({k: v}, indent=2, ensure_ascii=False)
        inner = blob[blob.index("\n") + 1:blob.rindex("\n")]  # strip outer braces
        chunks.append(inner)
    out = body + ",\n\n" + ",\n\n".join(chunks) + "\n}\n"

    with open(STATUS, "w", encoding="utf-8") as f:
        f.write(out)

    # The pre-existing bytes must survive VERBATIM as a prefix.
    with open(STATUS, "rb") as f:
        raw_after_b = f.read()
    prefix = raw_before_b[:len(body.encode("utf-8"))]
    if not raw_after_b.startswith(prefix):
        sys.exit("ABORT: byte-prefix broken -- restore from git and investigate")
    print(f"[post] byte-prefix of {len(prefix)} bytes preserved VERBATIM "
          f"({len(raw_before_b)} -> {len(raw_after_b)} bytes)")

    # ---- read back from DISK and verify ----
    with open(STATUS, encoding="utf-8") as f:
        rb = json.load(f, object_pairs_hook=OrderedDict)

    ok = True
    print(f"\n[post] {len(rb)} keys on disk (expect {len(keys_before)}+{len(NEW_KEYS)}="
          f"{len(keys_before)+len(NEW_KEYS)})")
    if len(rb) != len(keys_before) + len(NEW_KEYS):
        ok = False
        print("  FAIL: key count")

    print("\n--- ASSERT: 20 pre-existing keys unchanged (name, order, value) ---")
    rb_keys = list(rb.keys())
    for i, k in enumerate(keys_before):
        name_ok = (i < len(rb_keys) and rb_keys[i] == k)
        val_ok = (k in rb and json.dumps(rb[k], sort_keys=True, ensure_ascii=False) == frozen[k])
        flag = "OK  " if (name_ok and val_ok) else "FAIL"
        if not (name_ok and val_ok):
            ok = False
        print(f"  [{flag}] idx {i:2d} name={'same' if name_ok else 'MOVED'} "
              f"value={'identical' if val_ok else 'CHANGED'}  {k}")

    print("\n--- new keys appended (positions 20+) ---")
    for i, k in enumerate(NEW_KEYS, start=len(keys_before)):
        print(f"  [OK  ] idx {i:2d} NEW  {k}  ({len(json.dumps(rb[k]))} B)")

    print("\nRESULT:", "PASS - append-only guarantee held" if ok else "FAIL - ABORT AND RESTORE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
