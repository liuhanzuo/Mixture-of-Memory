#!/usr/bin/env python3
"""Append the 2026-08-17 0-GPU closeout keys to B04's STATUY.json, APPEND-ONLY.

Why a script and not a hand edit: STATUS.json is 93 KB of deeply nested prose. A hand edit
risks (a) touching an existing key, which LIFECYCLE_SCHEMA.md sec 0 forbids because the
history IS the evidence, and (b) silently breaking JSON. This script asserts both:

  * every pre-existing top-level key is byte-identical after the write (json.dumps of the
    old value == json.dumps of the new value, key by key);
  * the pre-existing key ORDER is preserved as a prefix, so new keys can only be appended;
  * the result parses.

It refuses to run twice (idempotence guard) rather than double-appending.

0 GPU. Reads/writes one file on wzc1.
"""
import json
import collections
import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
STATUS = HERE.parent / "STATUS.json"

# ---------------------------------------------------------------------------
# The keys to append. Naming is NOT free-form:
#
#  * `related_work_presence_correction_20260817` is chosen so that
#    check_stale_absence_claims.py's SUPERSEDING_KEY regex
#    (presence_correction|related_work_20\d{6}|_correction_20\d{6}|supersed) MATCHES it,
#    it NAMES RELATED_WORK.md, and it ASSERTS existence via "exists": true -- the three
#    conditions _superseded_structurally() requires. Verified by importing the module.
#    A name like `related_work_note_20260817` would satisfy none of them.
#
#  * There is deliberately NO `lifecycle_20260817` key. lifecycle_keys(doc) returns
#    ['lifecycle'] for this document and a dated key is the ONLY way to change what the
#    queue reads -- which is exactly why this agent must not write one. Promotion is MAIN's
#    call with the user, and blocker [0] (independent adversarial re-adjudication) is
#    untouched by this pass.
# ---------------------------------------------------------------------------
NEW_KEYS = collections.OrderedDict()

NEW_KEYS["related_work_presence_correction_20260817"] = {
    "supersedes": "remaining_blockers_after_this_design[1] AND the trailing clause of novelty_verdict",
    "file": "RELATED_WORK.md",
    "exists": True,
    "measured": "ls -la proposal/backlog/B04-eval-fragility-incubator/RELATED_WORK.md -> "
                "-rw-r--r-- 1 root root 46809 Aug 15 06:59. On disk 2 days before this key was written.",
    "the_two_stale_sentinels": [
        "remaining_blockers_after_this_design[1]: 'RELATED_WORK.md still absent -- blocks PROMOTION to paper<X>/'",
        "novelty_verdict, final clause: 'RELATED_WORK.md absence blocks PROMOTION only.'",
    ],
    "already_self_corrected_at": [
        "related_work_status = 'audited'",
        "related_work.doc = proposal/backlog/B04-eval-fragility-incubator/RELATED_WORK.md",
        "related_work.addresses_gap_audit: explicitly says the write 'closes "
        "remaining_blockers_after_this_design[1]'",
    ],
    "why_this_key_is_needed_anyway": "The self-corrections above are real but they are NOT "
        "structured as a superseding key, so neither a human skimming top-to-bottom nor "
        "check_stale_absence_claims.py can act on them. An agent that stops reading at "
        "blocker[1] gets dispatched to write a 46 KB file that already exists -- which has "
        "measurably happened twice this week on other proposals.",
    "GUARD_BLIND_SPOT_MEASURED_NOT_ASSUMED": {
        "claim": "proposal/check_stale_absence_claims.py cannot see EITHER of B04's two sentinels, "
                 "so 'the checker is green for B04' was never evidence of anything.",
        "how_measured": "imported the module and called it directly, 2026-08-17",
        "results": {
            "m.scan(B04/STATUS.json)": "[]  (empty -- zero rows for B04)",
            "m.ABSENCE.findall('RELATED_WORK.md still absent -- ...')": "[]",
            "m.ABSENCE.findall('RELATED_WORK.md absence blocks PROMOTION only.')": "[]",
            "m.ABSENCE.findall('RELATED_WORK.md is absent')  [control]": "['RELATED_WORK.md']",
        },
        "root_cause": "ABSENCE (check_stale_absence_claims.py:92-98) enumerates the predicates "
                      "'does not exist|do not exist|is absent|are absent|is missing|not on disk|"
                      "does NOT exist'. B04 wrote 'still absent' and 'absence blocks', which are "
                      "an ADVERB+adjective and a NOUN respectively -- neither is in the predicate "
                      "list. The whole-repo run returns rc=1 with exactly one row, and that row "
                      "is B09, not B04.",
        "consequence": "ready_queue.py:1232-1256 surfaces this checker's count as its "
                       "stale-absence warning, so the queue reported zero stale absences for B04 "
                       "while B04 carried two.",
        "NOT_fixed_here_and_why": "Widening ABSENCE is a change to a shared repo-wide guard that "
                                  "would re-scan every proposal and could newly flag proposals "
                                  "this pass has not read. It is filed as a separate 0-GPU task "
                                  "rather than done as a side effect of a B04 documentation pass. "
                                  "See suggested_followup_not_done_here.",
    },
}

NEW_KEYS["proposal_md_superseded_20260817"] = {
    "supersedes": "NOTHING is deleted. This RECORDS that next_gate.prereg_G0_first_0_GPU step (c) "
                  "and remaining_blockers_after_this_design[2] are now DISCHARGED.",
    "what_landed": "PROPOSAL.md now opens with a SUPERSEDED banner. The original 2026-08-08 body "
                   "is preserved BYTE-FOR-BYTE below the banner (verified: diff of the post-banner "
                   "region against git HEAD:PROPOSAL.md returned rc=0).",
    "requirement_it_satisfies": "prereg_G0_first_0_GPU (c): \"PROPOSAL.md's table must be marked "
                               "superseded BEFORE a threshold is quoted against it.\" Marked "
                               "2026-08-17, before any G1 fill and before any external write-up.",
    "verification": "grep -nE 'supersed|stale|作废' PROPOSAL.md -> rc=0 (it was rc=1, zero hits, "
                    "immediately before this pass).",
    "TWO_CORRECTIONS_TO_THE_PREREG_S_OWN_DESCRIPTION_OF_THE_DEFECT": {
        "correction_1_there_are_THREE_ladders_not_two": {
            "prereg_said": "prereg_G0_first_0_GPU (c) frames this as PROPOSAL.md vs "
                           "evidence/B04_6rung_bs16_analysis.json, i.e. a two-way disagreement "
                           "where 'the JSON side reproduces'.",
            "measured_2026_08_17": "PROPOSAL.md's row is NOT a miscopy of either JSON. It is a "
                                   "THIRD, EARLIER ladder: evidence/accnorm_margin_verified.md "
                                   "(2026-08-08, task #201). Recomputing Spearman from that file's "
                                   "own table reproduces its numbers EXACTLY, so it is internally "
                                   "consistent -- just a different ladder.",
            "median_margin_base_to_keep8_by_ladder": {
                "accnorm_margin_verified.md (what PROPOSAL.md quotes)": "0.124594 -> 0.075801",
                "B04_6rung_bs16_analysis.json (zwfy6, keep12 rung = step 124000)": "0.131806 -> 0.094933",
                "B04_wzc1_floor_analysis.json (wzc1 sm_100, keep12 rung = step 111500)": "0.131678 -> 0.094779",
            },
            "why_it_matters": "'PROPOSAL.md is wrong' invites someone to 'fix' the numbers in "
                              "place. The correct action is what was done -- mark it superseded and "
                              "NAME ITS LADDER. B04_wzc1_floor_analysis.json.ladder_identity_warning "
                              "already imposes exactly this discipline on Spearman(core6, heal_steps); "
                              "it applies to median_margin too.",
            "which_ladder_is_authoritative": "wzc1 (B04_wzc1_floor_analysis.json). It supplies "
                                             "phi's denominator D=0.021820 and sigma_hat=0.000541.",
        },
        "correction_2_the_p_0_0167_is_a_METRIC_MIX_UP_not_a_stale_value": {
            "prereg_said": "'...and on the p (0.0167 vs 0.0028)', i.e. one p superseding another.",
            "measured_2026_08_17": "Exact two-sided permutation over all 720 orderings at n=6, "
                                   "recomputed independently (hand-rolled tie-aware Spearman; scipy "
                                   "is absent from the conda env). On the 2026-08-08 ladder that "
                                   "PROPOSAL.md quotes: median_margin rho=+1.000000 p=2/720=0.002778; "
                                   "frac<0.005 rho=-0.942857 p=12/720=0.016667; frac<0.010 "
                                   "rho=-1.000000 p=2/720=0.002778.",
            "so": "PROPOSAL.md:14's 'Spearman(core6, frac<.005) = -.9429, p=.0167' is CORRECT FOR "
                  "ITS OWN LADDER. But on BOTH current ladders frac<0.005 reaches rho=-1.000000, "
                  "p=0.002778, and -0.9429/0.0167 there belongs to frac<0.010. The number is not a "
                  "superseded measurement of the same quantity -- on the current ladders it is THE "
                  "WRONG METRIC'S number.",
            "mechanism": "exactly one rung inversion in the 2026-08-08 frac<0.005 column: "
                         "ShortGPT-16 = 3.286% vs keep14 = 3.280%, a 0.006pp gap that the later "
                         "ladders do not reproduce. rho=-0.942857 is the n=6 single-inversion value.",
            "verified_on_current_ladder_too": "B04_6rung_bs16_analysis.json recomputed: "
                                              "median_margin +1.000000 p=0.002778; frac<0.001 "
                                              "-0.985611 p=0.005556; frac<0.005 -1.000000 "
                                              "p=0.002778; frac<0.01 -0.942857 p=0.016667.",
        },
        "additional_hazard_recorded_in_the_banner": "frac(margin<0.005) must not be quoted as a "
            "headline on ANY ladder: kill_gate.primary_metric_choice_is_prereg_not_posthoc DEMOTED "
            "it pre-data for failing its own noise floor (sigma_hat=0.004329, R=3.88, 0/5 adjacent "
            "rung gaps clear 2*sigma_hat) vs median_margin (sigma_hat=0.000541, R=68.26, 4/5 "
            "clear). Two of the four bullets in PROPOSAL.md's 已成立 section rest on the demoted "
            "metric, which the prereg's 'disagrees on every rung' framing does not mention.",
    },
}

NEW_KEYS["novelty_check_regenerated_20260817"] = {
    "discharges": "related_work.actionable_0_gpu_followup (STATUS.json:318) -- IN FULL.",
    "what_that_key_asked_for": "REGENERATE NOVELTY_CHECK.md's top-5 candidate table AND "
                               "differentiation table to include arXiv:2605.07271.",
    "state_found_2026_08_17": "HALF DONE, which that key does not say. The top-5 table WAS "
                              "regenerated on 2026-08-15 (AMENDED banner + row 0 for 2605.07271, "
                              "NOVELTY_CHECK.md:45-59). The differentiation table was NOT: its "
                              "header carried 6 comparator columns (Tropeano / Madaan / ShortGPT / "
                              "Shortened LLaMA / Fluid Bench / B04) and no 2605.07271 column. "
                              "Only the second half was owed; the first was NOT redone.",
    "what_landed_now": [
        "Differentiation table REGENERATED with 2605.07271 as the LEFTMOST comparator, labelled "
        "NEAREST, because row 0 of the top-5 table identifies it as nearer than Tropeano.",
        "Three NEW axes added, because the original 6 axes could not express why 2605.07271 is not "
        "a kill: 'Cross-checkpoint?', 'Noise floor measured?', and an 'Effect on B04's residual' row.",
        "The concluding paragraph's claim that Tropeano is 'the closest but measurement-family "
        "disjoint' is STRUCK and corrected: 2605.07271 is measurement-family IDENTICAL (same "
        "per-item MC margin definition, same layer-pruning damage operation), so differentiation "
        "now rests only on no-heal-ladder / no-cross-checkpoint-rank-statistic / no-measured-floor. "
        "That is a NARROWER and more fragile residual, stated as such.",
        "Model-scope row's 'Qwen replication failed' parenthetical corrected to 'cross-family "
        "UNTESTED', per kill_history[1].",
    ],
    "second_stale_sentence_also_fixed": {
        "sentence": "NOVELTY_CHECK.md:41 'General claim killed by Qwen3-8B replication (rho=+0.43, "
                    "p=0.42; rho=-0.49, p=0.36).'",
        "action": "struck through in place (NOT deleted -- the 2026-08-09 dated record is "
                  "preserved) with a dated CORRECTION block beneath it.",
        "authority": "kill_history[1] downgraded GENERAL_CLAIM_KILLED -> NON_MATCHED_INCONCLUSIVE "
                     "on 2026-08-10. The two rho values are NOT retracted; what is retracted is "
                     "reading them as a refutation.",
        "note": "NOVELTY_CHECK.md's own 2026-08-14 banner flagged BOTH sentences as stale and "
                "actionable, and explicitly said 'Nothing in the body has been edited'. The banner "
                "was added 3 days ago; the sentences were never fixed until now. A banner saying "
                "'the text below is stale' is not a fix -- readers quote the body, not the banner.",
    },
    "verdict_unchanged": "hold_in_backlog. Still in ready_queue.py's VERDICT_CLEARED, so novelty "
                         "remains CLEARED for the queue. This pass makes the SUPPORTING TABLE "
                         "honest; it does not change the verdict, and it does not touch lifecycle.",
    "residual_is_now_visibly_narrower": "Per related_work.what_it_costs_B04, 2605.07271 costs B04 "
        "three must-not-claim items (per-item MC decision margin as a pruning lens; "
        "damage/phase-dependent perturbation sensitivity; recovery bounded by remaining depth). The "
        "regenerated table makes that visible IN the table rather than only in STATUS.json prose.",
}

NEW_KEYS["closeout_20260817_scope_and_what_remains"] = {
    "agent": "0-GPU subagent, dispatched with a hard ZERO-GPU budget",
    "gpu_used": "none. No nvidia-smi, no ssh to any node, no training, no eval. Every number "
                "below was recomputed on wzc1 CPU from JSON already on disk.",
    "DONE_this_pass": [
        "related_work.actionable_0_gpu_followup -- differentiation table regenerated (the owed half).",
        "prereg_G0_first_0_GPU step (c) + remaining_blockers_after_this_design[2] -- PROPOSAL.md "
        "banner landed.",
        "Both stale Qwen sentences in NOVELTY_CHECK.md (:41 and the Model-scope row) corrected.",
        "remaining_blockers_after_this_design[1] + novelty_verdict's trailing clause -- superseded "
        "by related_work_presence_correction_20260817 (RELATED_WORK.md is 46809 B, on disk since "
        "Aug 15 06:59).",
    ],
    "NOT_DONE_and_honestly_why": {
        "remaining_blockers_after_this_design[0] -- the adversarial re-adjudication": {
            "status": "NOT DONE. This is the stated gate to lifecycle=ready_gpu and it is "
                      "UNTOUCHED by this pass.",
            "is_it_0_GPU": "Yes in principle -- it is a review, not a run.",
            "why_not_done_here": "It requires an INDEPENDENT multi-lens pass, and this agent is "
                                 "the one that just edited the documents. STATUS.json is emphatic "
                                 "in four places (revision_2's 'The adversarial verdicts are the "
                                 "authority, not anyone's opinion'; blocker[0]; "
                                 "order_of_operations) that the reviser may not self-adjudicate. "
                                 "Me returning SOUND on my own edits would be worth nothing.",
            "also_verified": "revision 3 landed 2026-08-16 and there is NO adjudication record on "
                             "disk for it: grepped the proposal dir for revision.3 combined with "
                             "adversarial/SOUND/NEEDS_REVISION/lens -> zero hits. So revision 3 "
                             "has never been adjudicated either, not just revision 2.",
        },
        "the_G1_eval_fill": {
            "status": "NOT DONE -- GPU-blocked by construction. This is the headline next step.",
            "command": "setsid nohup bash scripts/_run_b04_readout_evalfill.sh > "
                       "logs/b04_evalfill_<ts>.out 2>&1 &",
            "requires": "8 idle sm_100 cards on ONE node (LOCAL or .21). The driver REFUSES to "
                        "launch otherwise: it exits 5 if GPUs are busy and exits if compute_cap "
                        "!= sm_100.",
            "COST_FIGURE_THE_QUEUE_SHOWS_IS_WRONG_BY_~3x_IN_THE_CHEAP_DIRECTION": {
                "queue_prints": "1.08 GPU-h (gate G1), read from gpu_cost_estimate.value",
                "actual": "~4.83 occupancy GPU-h / ~36 min wall (~1.68 with PREFETCH=1; 0.44 "
                          "compute-only), measured in EVAL_FILL_READY_20260816.md sec 4",
                "why_the_queue_cannot_see_it": "ready_queue.py resolves cost from a fixed key list "
                                               "(next_gate_gpu_20260816 / next_gate_gpu / "
                                               "next_gate_cost / gate_gpu / next_gate_gpu_cost) "
                                               "plus a separate gpu_cost_estimate lookup. The "
                                               "correction lives in a .md the queue never opens.",
                "two_causes_of_the_old_figure": "it was for 4 arms (6 are now required), and it was "
                                                "anchored on a WARM-page-cache 121 s stage; cold "
                                                "ckpt load measures 338-348 s vs 89 s warm.",
                "NOT_written_as_a_dated_cost_key_here": "deliberately. Appending "
                    "next_gate_gpu_20260817 would change what a DISPATCHER reads about a GPU spend, "
                    "which is a scheduling decision, not a documentation fix. Flagged for MAIN.",
            },
            "partial_fill_is_a_TRAP": "The union is 7 steps but neither grid needs all 6 missing "
                "ones (GRID_I needs 153500/175000; GRID_W needs 25000/50000). Filling only some "
                "leaves the COMBINED verdict READOUT_ABSENT while making GRID_I alone computable -- "
                "so a partial run WILL print a number. Adding or dropping points is a "
                "PROTOCOL_VIOLATION and phi_budget() asserts the step set exactly. The bias "
                "direction is documented and favours B04: unused ckpts cluster near 200000, so "
                "extra points shrink the range term, and k=4 vs k=5 costs 11.5% of E[range]/sigma.",
        },
        "the_two_disk_rule_CANNOT_be_discharged_from_LOCAL": {
            "measured": "zwfy6 is NOT MOUNTED on this node. ls /apdcephfs_zwfy6/... -> No such "
                        "file or directory; ls / | grep apdcephfs shows only apdcephfs_wzc1 and "
                        "apdcephfs_wzc1_304376610.",
            "consequence": "READOUT_ABSENT's 'on EITHER disk' clause is not locally decidable. "
                           "analyze_b04_wzc1_floor.py correctly prints UNSEARCHABLE rather than "
                           "treating zwfy6 as empty. prereg_ambiguities item 4 already states this. "
                           "NOBODY MAY REPORT 'checked both disks' FROM LOCAL, and the zwfy6 "
                           "absence currently on record is INHERITED from a 2026-08-16 scan run "
                           "from .73, not re-established.",
        },
    },
    "PHI_IS_UNDEFINED_DO_NOT_QUOTE_ANY_NUMBER_FROM_THIS_FILE_AS_A_RESULT": {
        "warning": "STATUS.json contains dozens of concrete-looking phi values (0.2979 / 0.3896 / "
                   "0.8387 / 0.9395 / 1.7760 / 0.704176 / 0.352088 / 0.300023 ...) and one line "
                   "even reads 'the gate FIRED: phi_GRID_I 0.2979 PASS ... COMBINED NARROWED'. "
                   "EVERY ONE of those is a hand-constructed hypothetical y-vector, a --selftest "
                   "reachability probe, or an explicitly-labelled PLACEHOLDER rehearsal.",
        "the_only_measured_number_in_the_readout": "y[200000] = median_margin 0.108500 (n=17195), "
                                                  "one of five required points.",
        "actual_state": "--readout-only returns rc=3, verdict READOUT_ABSENT. GRID_I is missing "
                        "[100000,128000,153500,175000]; GRID_W is missing "
                        "[25000,50000,100000,128000]. phi is UNDEFINED -- not small, not large.",
        "self_reference": "This proposal's own lesson_for_the_next_agent records the same class of "
                          "error one level up: 'Three separate passes checked whether phi's FORMULA "
                          "was right; none checked whether phi's INPUTS EXISTED.'",
    },
    "mandatory_codisclosure_reminder": "kill_gate.mandatory_disclosure_on_any_report: any quotation "
        "of Spearman(core6, median_margin)=+1.00 must print beside it Spearman(core6, heal_steps) "
        "for the SAME ladder WITH THE LADDER NAMED (+0.6669 wzc1 / +0.8721 zwfy6), sigma_hat and R, "
        "and clause 5's phi (currently UNDEFINED). The regenerated NOVELTY_CHECK.md correction "
        "block now carries this disclosure inline.",
    "lifecycle_DELIBERATELY_NOT_TOUCHED": {
        "value": "unchanged at ready_cpu",
        "why": "Promotion is MAIN's call with the user, and blocker[0] (independent adversarial "
               "re-adjudication) is still open. Note also that lifecycle_keys(STATUS.json) resolves "
               "to ['lifecycle'] only -- B04 has no dated lifecycle key -- so the ONLY way to move "
               "queue state would be to append a key literally named lifecycle_20260817. This pass "
               "intentionally does not, and appending lifecycle_stays / lifecycle_unchanged (B04 "
               "already has three such keys) would change NOTHING the queue reads.",
    },
    "suggested_followup_not_done_here": [
        "0 GPU: widen check_stale_absence_claims.py's ABSENCE predicate list to catch 'still "
        "absent' / 'absence blocks' (adverb+adjective and noun forms). Measured to miss BOTH of "
        "B04's sentinels. Repo-wide guard change -> separate task, since it re-scans every proposal.",
        "0 GPU: independent multi-lens adversarial re-adjudication of revision 3 (blocker[0]). Must "
        "NOT be this agent.",
        "SCHEDULING (MAIN): decide whether to append a dated gpu-cost key so a dispatcher sees "
        "~4.83 occupancy GPU-h instead of the queue's 1.08 GPU-h.",
    ],
}


def main():
    raw_before = STATUS.read_bytes()
    sha_before = hashlib.sha256(raw_before).hexdigest()
    doc = json.loads(raw_before, object_pairs_hook=collections.OrderedDict)
    old_keys = list(doc.keys())

    already = [k for k in NEW_KEYS if k in doc]
    if already:
        print(f"REFUSING: already appended {already}. This script is not idempotent by design; "
              f"re-running would risk overwriting a prior append.")
        return 2

    # Snapshot every pre-existing value so we can prove nothing changed.
    old_blobs = {k: json.dumps(doc[k], ensure_ascii=False, sort_keys=True) for k in old_keys}

    for k, v in NEW_KEYS.items():
        doc[k] = v

    out = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"

    # --- assertions BEFORE writing ---
    reparsed = json.loads(out, object_pairs_hook=collections.OrderedDict)
    new_keys_list = list(reparsed.keys())
    assert new_keys_list[:len(old_keys)] == old_keys, "pre-existing key ORDER changed (not append-only)"
    assert new_keys_list[len(old_keys):] == list(NEW_KEYS.keys()), "appended keys are not the tail"
    for k in old_keys:
        nb = json.dumps(reparsed[k], ensure_ascii=False, sort_keys=True)
        assert nb == old_blobs[k], f"pre-existing key MUTATED: {k}"
    assert reparsed["lifecycle"] == "ready_cpu", "lifecycle must remain ready_cpu"
    assert not any(k.startswith("lifecycle_20") for k in NEW_KEYS), "must not append a dated lifecycle key"

    STATUS.write_text(out, encoding="utf-8")

    raw_after = STATUS.read_bytes()
    json.loads(raw_after)  # final parse check on what is actually on disk
    print(f"OK  sha256 before={sha_before[:16]} after={hashlib.sha256(raw_after).hexdigest()[:16]}")
    print(f"OK  top-level keys {len(old_keys)} -> {len(reparsed)}  (+{len(NEW_KEYS)})")
    print(f"OK  all {len(old_keys)} pre-existing keys byte-identical; order preserved as prefix")
    print(f"OK  lifecycle still {reparsed['lifecycle']!r}; no dated lifecycle key appended")
    for k in NEW_KEYS:
        print(f"    appended: {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
