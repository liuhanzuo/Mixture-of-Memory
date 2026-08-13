#!/usr/bin/env python3
"""Append the `full32_trajectory_ni_20260813` key to A04's STATUS.json.

WHY THIS IS A SCRIPT AND NOT AN INLINE EDIT
-------------------------------------------
`STATUS.json` holds 38 pre-existing keys that are pure provenance; the dispatch
requires that NONE of them be overwritten, reordered or even reformatted. Two
failure modes were hit while doing this by hand and are guarded against here:

 1. `json.dump(..., indent=1)` silently REFORMATTED all 38 existing keys (the
    file is indent=2), producing a 1908-deletion diff that looked like a rewrite.
    Fixed by serialising ONLY the new key and splicing it in as text.
 2. splicing by `str.rindex` + comma juggling produced INVALID JSON. Fixed by
    doing the splice arithmetically on the last `}` and re-parsing before write.

The script is idempotent: it refuses if the key already exists, and it verifies
after writing that the 38 original keys are byte-identical (by `json.dumps`
comparison against the pre-edit parse) and that the file still parses.
"""
import collections
import json
import sys

P = ("/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/"
     "proposal/active/A04-recovery-certification/STATUS.json")
KEY = "full32_trajectory_ni_20260813"

VAL = collections.OrderedDict([
    ("what",
     "ADDITIVE. Closes cheap_next_steps_dominate[1] for the full32_dolmino half "
     "(the keep14 half was 517c8d2): the 4-axis NI DISCRIMINATION CURVE along "
     "the full32_dolmino 7B continued-pretraining trajectory -- step5000 / 10000 "
     "/ 15000 / 20000 plus the already-archived step25000 endpoint. EVAL-ONLY, "
     "zero training. Archived dirs NOT overwritten; the four new ckpts get their "
     "own A04_7B_full32_step* dirs."),
    ("verdict",
     "SOME_CHECKPOINT_MEETS_THE_2OF3_BAR__PREREG_B_ACCEPT_NOT_FROM_CONVERGENCE"),
    ("headline",
     "THE ACCEPT IS NOT THE ENDPOINT. All FOUR earlier ckpts accept on 2 of 3 "
     "decision axes; step25000 -- A04's celebrated 'only NI accept in A04' -- is "
     "the ONLY point on its own trajectory that FAILS the >=2/3 bar (1 of 3). On "
     "triviaqa the accept set is {5000,10000,15000,20000}: present at the FIRST "
     "ckpt measured and LOST between 20000 and 25000. Pre-registered reading (b) "
     "fires in its strongest form."),
    ("document", "A04_FULL32_TRAJECTORY_NI_VERDICT.md"),
    ("prereg_document",
     "A04_FULL32_TRAJECTORY_PREREG.md (commit 537d323, written and committed "
     "BEFORE any of the four new ckpts had a summary.json; the step5000/step15000 "
     "GPU jobs started 11:36:51/11:40:36)"),
    ("evidence", ["evidence/a04_full32_trajectory_ni.json"]),
    ("code", ["code/a04_full32_trajectory_axes_driver.sh",
              "code/a04_full32_trajectory_ni.py",
              "code/a04_full32_stage_parallel.sh",
              "code/a04_status_add_full32_trajectory_key.py"]),
    ("gpu_h_spent", 6.531),
    ("gpu_h_note",
     "wall-clock per-ckpt driver time x 8 GPUs: .73 723s+727s = 3.222; .82 "
     "745s+744s = 3.309. Analysis is CPU-only (59 s). step25000 was NOT "
     "re-scored -- it is the archived endpoint, read from disk, so 0 GPU was "
     "spent re-deriving a number that already existed."),
    ("nodes",
     ".73 (step5000, step10000) + .82 (step15000, step20000), 8xH20 each, "
     "zwfy6, both verified 0 MiB / 0% before launch (the driver refuses to start "
     "if >8000 MiB is held). Analysis PINNED to .73. NOT touched: .104 (paperC "
     "Qwen3 heal), LOCAL/.21 (SparseForge #246 -- wzc1 was read ONLY, as the "
     "ckpt source)."),
    ("margins_pp_split", collections.OrderedDict([
        ("triviaqa", {"step5000": 3.6931564868479714,
                      "step10000": 3.0467008470798036,
                      "step15000": 2.3330918412839945,
                      "step20000": 2.450401248328131,
                      "step25000_archived_endpoint": -0.6035443602318322}),
        ("popqa", {"step5000": -2.0508507056320043,
                   "step10000": -2.1911239817806117,
                   "step15000": -3.074219421487836,
                   "step20000": -3.508778146568043,
                   "step25000_archived_endpoint": -4.539146281628934}),
        ("mmlu_content", {"step5000": 0.5439040022788777,
                          "step10000": 0.37298817832217623,
                          "step15000": 0.9142216208517306,
                          "step20000": 0.6222404215923658,
                          "step25000_archived_endpoint": 1.049529981484119}),
        ("nq_open_DEMOTED", {"step5000": -2.9363, "step10000": -2.9917,
                             "step15000": -2.8532, "step20000": -3.0194,
                             "step25000": -3.6579}),
    ])),
    ("ni_accept_by_step_split", {"step5000": True, "step10000": True,
                                 "step15000": True, "step20000": True,
                                 "step25000": False}),
    ("verdict_per_checkpoint",
     "2 of 3 decision axes accepting at step5000/10000/15000/20000 (threshold 2) "
     "-> NI ACCEPT; 1 of 3 at step25000 -> REJECT. Identical under "
     "split/first/last/wrong. Under credit, D6 retires mmlu_content (5 of 15 "
     "cells) so the family is 2 axes and the pattern becomes 1-of-2 vs 0-of-2 -- "
     "the verdict FLIP is unchanged in EVERY convention."),
    ("answers_to_the_three_prereg_questions", collections.OrderedDict([
        ("Q1_where_does_the_accept_appear",
         "It does not APPEAR anywhere: it is already present at step5000, the "
         "earliest ckpt that exists, and is LOST between 20000 and 25000. The "
         "only boundary on this trajectory is a REJECT boundary in (20000, "
         "25000]. No accept boundary exists in the scanned range. Nothing in "
         "A04's design anticipated that direction."),
        ("Q2_triviaqa_closer_or_further_earlier",
         "NOT close -- already ACCEPTING, by up to +3.6932 pp = 12.39 bootstrap "
         "SE of headroom, i.e. 6.1x further PAST the threshold than the endpoint "
         "is short of it (-0.6035 pp, 1.86 SE). The endpoint is the WORST point "
         "on this axis."),
        ("Q3_ratio_vs_ni",
         "The disagreement is UNIQUE TO THE ENDPOINT (1 of 5 ckpts). RATIO "
         "mean_ratio decays MONOTONICALLY 0.898062 -> 0.893552 -> 0.886364 -> "
         "0.879378 -> 0.851495; the margin over rho=0.85 shrinks +0.048062 -> "
         "+0.001495, i.e. 32x. At the four earlier ckpts BOTH rules accept and "
         "there is no disagreement at all. So the disagreement arises because NI "
         "MOVES while RATIO merely DECAYS -- it is a property of that "
         "checkpoint, not of the arm."),
    ])),
    ("the_load_bearing_finding", collections.OrderedDict([
        ("what",
         "NI DOES discriminate -- it changes state along a trajectory and the "
         "state change is RESOLVED on the item sample. That is the thing that "
         "had never been demonstrated. But the DIRECTION is the reverse of the "
         "design's expectation: NI accepts a model continued-pretrained for 5000 "
         "steps and REVOKES the accept at 25000. The rule is tracking DRIFT AWAY "
         "from the anchor, not APPROACH to it."),
        ("trajectory_is_getting_worse",
         "popqa margin is monotone NON-INCREASING across all five points "
         "(-2.0509 -> -4.5391), 3 of its 4 successive moves RESOLVED. triviaqa "
         "acc falls 61.4021 -> 57.1500. nq_open falls. Only mmlu_content drifts "
         "mildly up (+0.5439 -> +1.0495) and it is NON-monotone (diffs -0.1709, "
         "+0.5412, -0.2920, +0.4273); its 20000->25000 move is NOT resolved (CI "
         "lower bound EXACTLY 0.0000, p=0.0508, criteria_disagree=true, "
         "conservative AND -> not resolved). So continued pretraining made a "
         "HEALTHY model worse on 3 of 4 axes -- measured, not inferred."),
        ("resolved_moves",
         "triviaqa 5000->10000 -0.6409 (p=0.0072), 10000->15000 -0.7078 "
         "(p=0.0038), 20000->25000 -3.0205 (p=0.0001, +771/-1313 of 17944); "
         "popqa 10000->15000 -0.8691 (p=0.0001), 15000->20000 -0.4276 "
         "(p=0.0212), 20000->25000 -1.0023 (p=0.0001); mmlu_content "
         "10000->15000 +0.5056 (p=0.0418)."),
        ("consequence",
         "full32@step25000 should STOP being described as A04's accept. It is "
         "the only ckpt on its own trajectory that FAILS the bar, and it was "
         "selected as the arm only because it was the last save. The accept, and "
         "the 97.7% recovery framing, belong to the ckpts nobody had scored."),
    ])),
    ("endpoint_triviaqa_reject_is_substantially_a_VERBOSITY_artefact",
     collections.OrderedDict([
         ("status",
          "LABELLED DIAGNOSTIC. The decision metric is and remains EM; "
          "`contains` is NEVER substituted and no cell is re-scored."),
         ("measurement",
          "Of the 1313 items that went EM right->wrong over 20000->25000, 622 "
          "(47.4%) STILL CONTAIN the gold answer, and mean prediction length on "
          "exactly those items explodes 10.9 -> 50.4 chars."),
         ("examples",
          "gold 'Rudolf Hess': 'Rudolf Hess' -> 'Rudolf Hess, who was the last "
          "inmate of Spandau prison in Berlin, was found hang...'; gold 'Dark "
          "Blood': 'Dark Blood' -> 'He died during the filming of the movie "
          "\"Dark Blood\" in 1993.'"),
         ("contains_confirms",
          "triviaqa EM falls 61.4021 -> 57.1500 (-4.25 pp) while contains moves "
          "only 69.8729 -> 68.3850 (-1.48 pp); the contains-minus-EM gap WIDENS "
          "8.4708 -> 11.2350 pp, vs 3.8732 pp for the vanilla anchor."),
         ("reading",
          "The axis whose REJECT flips the endpoint's verdict is measuring, to "
          "nearly HALF its movement, a model that STOPPED EMITTING SHORT ANSWERS "
          "-- expected for a base LM continued-pretrained on raw text with no SFT "
          "-- and NOT demonstrated forgetting. The other 52.6% ARE genuine "
          "content substitutions (Richard Noble->Andy Green, David "
          "Hockney->David Bowie), so this is not the whole story either. Both "
          "halves are on the record; neither re-scores any cell."),
         ("not_output_degeneracy",
          "0.000% empty predictions at all five ckpts; top_constant_frac "
          "0.26-0.30% and FALLING at the endpoint (0.2619%); distinct "
          "predictions RISING 7808 -> 8205."),
     ])),
    ("neighbour_precondition_2_0_2_FIRED", collections.OrderedDict([
        ("what",
         "First time A04_GATE_DESIGN.md 2.0.2 has had an actual accept to gate. "
         "9 accepting cells tested against their +/-5000-step neighbours, PER "
         "AXIS as 2.0.2 requires."),
        ("mmlu_content",
         "all 5 accepts SURVIVE their neighbours -- every present neighbour also "
         "accepts. This axis's accept is NOT checkpoint-selection dependent."),
        ("triviaqa",
         "3 of 4 accepts survive; step20000 FAILS -- its upper neighbour "
         "step25000 rejects -> ACCEPT_IS_CHECKPOINT_SELECTION_DEPENDENT."),
        ("endpoints_stated_not_assumed",
         "step5000 (both axes) and step25000 have only ONE neighbour each; "
         "recorded explicitly rather than silently treated as satisfied."),
        ("so",
         "the precondition is doing work, not decorating the design: had this "
         "scan been run at a single hand-picked ckpt, the reported margin would "
         "have been an artefact of that choice."),
    ])),
    ("premises_of_the_dispatch_that_were_WRONG", collections.OrderedDict([
        ("1_the_5p7h_staging_cost",
         "full32_trajectory_staging_remeasured_20260813 priced staging at 5.7 h "
         "from a MEASURED 16.3 MiB/s single-stream rate. The MEASUREMENT "
         "reproduces (17.4 MiB/s, 2 GiB / 118 s) but the INFERENCE from it is "
         "wrong: 16-17 MiB/s is a PER-STREAM ceiling, not the link's capacity. "
         "Measured 8 concurrent streams to ONE node = 130.7 MiB/s (6 GiB/47 s); "
         "4+4 split over two nodes = 134.3 MiB/s; realised per-ckpt 137-138 "
         "MiB/s. All four ckpts staged AND full-file-sha256-verified in ~31 min. "
         "NOTE the deferral's CONCLUSION (stage now) was already correct for a "
         "different reason, so this WIDENS the margin rather than reversing the "
         "call."),
        ("2_earlier_ckpts_should_reject_harder",
         "trajectory_scan_NOT_run.expectation_to_design_for said 'earlier ckpts "
         "are LESS converged so should reject HARDER: the accept boundary is "
         "most likely BEYOND step25000 or off this trajectory'. FALSIFIED -- "
         "earlier ckpts ACCEPT, and the endpoint is the worst point on 3 of 4 "
         "axes."),
        ("3_the_accept_boundary_is_on_that_trajectory",
         "an ACCEPT boundary is not. A REJECT boundary is, in (20000, 25000]."),
        ("4_ckpt_count",
         "confirmed as this key's predecessor already corrected: 5 files exist, "
         "4 are new scan points, step25000 was already scored and was NOT "
         "re-scored (0 wasted GPU)."),
        ("confirmed_as_described",
         "mode=pruned in the archived meta despite zero structural damage; "
         "anchor = vanilla OLMo-2-1124-7B (NOT full32_step25000); all four ckpts "
         "zip-OK with exactly 1435 entries; .73/.82 both fully idle before "
         "launch."),
    ])),
    ("mains_interim_note_corrected", collections.OrderedDict([
        ("addresses",
         "A04_FULL32_READING_B_IS_FIRING.md (MAIN, 2026-08-13 11:58, written "
         "from the first completed ckpts WHILE the scan was still running). MAIN "
         "called reading (b) correctly and asked for exactly the three things "
         "the verdict delivers; its closing instruction was 'Do not let my hand "
         "arithmetic into the record ... Canonical output only.' Enforced."),
        ("mains_hand_arithmetic_was_WRONG",
         "MAIN derived mmlu_content @step15000 margin = +1.2775 pp by "
         "subtracting a RECORDED null instead of importing build_nulls -- the "
         "same shortcut it flagged as having put its keep14 margins ~0.5 pp off. "
         "The canonical value is +0.9142 pp (error +0.36 pp, same direction)."),
        ("consequence",
         "MAIN's stronger claim 'the EARLIER one accepts by more' is FALSE on "
         "mmlu_content: the endpoint IS that axis's best ckpt (+1.0495) and all "
         "five accept, so its accept set IS a suffix. The claim is emphatically "
         "TRUE on triviaqa (+3.6932 @step5000 vs -0.6035 @endpoint), which is "
         "the axis that actually flips the verdict. So reading (b) fires via a "
         "DIFFERENT axis than the interim note inferred, and 'there may be no "
         "boundary on this trajectory' resolves into 'no ACCEPT boundary; a "
         "REJECT boundary in (20000, 25000]'."),
        ("also_answered",
         "mmlu_content is NOT monotone in step (successive diffs -0.1709, "
         "+0.5412, -0.2920, +0.4273); that non-monotonicity is carried in the "
         "verdict string via ACCEPT_NOT_FROM_CONVERGENCE."),
    ])),
    ("integrity", collections.OrderedDict([
        ("archived_endpoint_reproduction",
         "EXACT to 3.6e-07 pp (hard-fail threshold 5e-4): triviaqa -0.603544, "
         "popqa -4.539146, mmlu_content +1.049530 all reproduced; RATIO "
         "mean_ratio 0.8514950516430542 with abs diff EXACTLY 0.0. Proves the "
         "imported guard/anchor/rule are the ones that produced the archive, so "
         "the four new points sit on the endpoint's scale."),
        ("shard_assertions",
         "24 of 24 shard cells: index set EXACTLY {0..7} (not a file count), "
         "merged n exactly EXPECTED_N (triviaqa 17944 / popqa 14267 / nq_open "
         "3610 / mmlu 14042), 0 duplicate item_ids, 0 nan, identical item_id "
         "sequences across all six arms. Per-shard row counts in the JSON."),
        ("guard_D1_D6",
         "all four axes CERTIFIABLE under split/first/last/wrong, 0 of 15 "
         "decision cells retired. Under credit, mmlu_content is NOT_CERTIFIABLE "
         "by D6 -> 5 of 15 retired, exactly as A04_MARGIN_GUARD_PREREG.md 2.2 "
         "predicted for 7B. No verdict depends on it."),
        ("anchor_and_delta",
         "anchor = vanilla ../models/OLMo-2-1124-7B, IMPORTED from "
         "a04_shallow_rung_ni_7b.ANCHOR, never redeclared, never substituted. "
         "Guard G2 honoured: full32_step25000 is an ARM, never the anchor "
         "(it scores BELOW vanilla on all four axes, so substituting it would "
         "shrink every Delta AND lower every target = manufacturing accepts). "
         "Delta never recomputed against a different intact."),
        ("protocol_asserted_TWICE_and_negative_tested",
         "protocol_asserted() runs BEFORE anything is scored and parses "
         "cb_bs/mmlu_bs out of the drivers' OWN echoed lines (summary.json:meta "
         "records NEITHER batch_size NOR chat_template). A SECOND, stricter gate "
         "asserts EVERY DRIVER START line -- not just the first -- plus step "
         "coverage, because the two invocations per node share one append-only "
         "log. NEGATIVE-TESTED twice: a doctored log reading cb_bs=48 -> 'FATAL "
         "protocol deviation'; a missing log -> 'FATAL: driver log ... absent'; "
         "in BOTH cases NO output file was written. cb_bs=32, mmlu_bs=16, "
         "add_bos asserted `is False` (never `is not True`) on all 12 new result "
         "dirs, max_new_tokens=32, chat_template=False established STRUCTURALLY "
         "(neither harness has a chat-template code path)."),
        ("truncation_incident_recorded",
         "logs/a04_full32_traj_15000.out LOST its DRIVER START header: AFTER "
         "step15000 had already completed, a duplicate launch's `>` redirection "
         "re-created the file, and that duplicate driver then exited via its own "
         "gpu_free_or_die guard ('REFUSE: 140116MiB of GPU memory held by "
         "another process'). The imported gate correctly HARD-FAILED on the "
         "truncated file rather than publishing cells whose protocol it could "
         "not establish. Two things saved the run: the driver `tee -a`s every "
         "note to a per-node progress log (append-only, cannot be clobbered), "
         "and the refused duplicate never touched a GPU. The progress logs are "
         "the authoritative protocol record and are what both gates read."),
        ("harness_parity",
         "eval_olmo2_closedbook_qa.py md5 2ed41993241226c795a3ca38375933f7 and "
         "eval_olmo2_mmlu_content.py md5 fe4a62dbdf884a1e2aedc6ed26887b4e "
         "verified IDENTICAL on wzc1, .73 and .82, and identical to the copies "
         "that produced the anchor and the endpoint -> same-CODE comparison, not "
         "code-version drift."),
        ("staging_verification",
         "per ckpt: size == source size; FULL-FILE sha256 equal on BOTH disks "
         "(step5000 a206cb9610d35402169fb78ef0507eabdeca932e4404ea775963964791b832ba, "
         "step10000 244d4db4bd756ae2118c4454b968f368abcee6fff8aef0b73cb20c83dc166143, "
         "step15000 dacf11eacfa9c3d27bde262bc37dde0ac83c36093ee814ef8c2b4a5d09c2b4ed, "
         "step20000 fc28e917d9bbdad24b6086d7f331094108ee87a1423662c006c51484d86b18fb); "
         "zip entry count == 1435 == source on each; plus a torch.load probe "
         "asserting meta step / keep_front 32 / n_fresh 0 / 32 layers / "
         "len(model_state)==355 BEFORE 8 GPUs were spent. A prefix hash was "
         "deliberately NOT used -- the known cluster failure mode is a TRUNCATED "
         "WRITE (shortgpt16/step128000.pt on zwfy6), which a prefix hash cannot "
         "see."),
        ("analysis_node_pinned",
         ".73, numpy 2.5.1 -- the node that produced the archive. "
         "neighbour_variability_20260813.reproducibility_defect_found measured "
         "Generator.multinomial drifting 19/10000 rows vs numpy 2.4.6 on .82, up "
         "to 0.0053 pp, an ORDER OF MAGNITUDE larger than this script's 5e-4 pp "
         "reproduction hard-fail -- so the same code passes on .73 and "
         "assert-fails on .82, and the failure looks like a logic bug. GPU "
         "scoring was split across nodes (it is deterministic); the bootstrap "
         "was NOT split."),
        ("bootstrap_offsets_disjoint",
         "new arms arm_index 500-503 (form 97*arm_index+13*axis); the endpoint "
         "keeps its ARCHIVED offset 203 so the reproduction check is exact and "
         "no archived cell is perturbed. Interval offset SEED+2900. Disjoint "
         "from pilot_zero {0,1}, step100k 100-102, shallow_rung 200-203, keep14 "
         "300-301, neighbour 400-408, and the guard offsets 700/900/1700/1900/2400."),
        ("code_reuse",
         "ni_rule/ratio_rule/build_nulls/load_shards/mmlu_content_norm_vec/"
         "qa_metric_vec/EXPECTED_N/AXES/DEMOTED_AXES/PREREG imported from "
         "pilot_zero_rule_disagreement; paired_bootstrap/TIE_CONVS/N_BOOT/SEED "
         "from A03's analyze_1b_knowledge_floor; ANCHOR/_load_arm/assert_aligned/"
         "SD_RUN_1B_PP from a04_shallow_rung_ni_7b; and shard_integrity_report/"
         "guard_cell/monotone_report/output_shape_and_flips/protocol_asserted "
         "imported from a04_keep14_trajectory_ni, so the two halves of the "
         "dispatch are the SAME CODE by construction, not merely the same "
         "intent. Nothing decision-bearing reimplemented."),
    ])),
    ("licensed",
     "the 7B accuracies, nulls, residuals, Delta, lo95 bounds and margins; the "
     "exact reproduction of the archived endpoint; 'no realisable perturbation "
     "of the ITEM SAMPLE flips these verdicts' (1.54-16.39 bootstrap SE, "
     "measured at 7B); 'the four earlier ckpts accept on 2 of 3 decision axes "
     "and the endpoint does not'; 'the popqa margin degrades monotonically "
     "across the trajectory'; 'the RATIO/NI disagreement occurs at exactly one "
     "of five ckpts'; and '47.4% of the endpoint's triviaqa EM regressions still "
     "contain the gold answer'."),
    ("NOT_licensed", [
        "ANYTHING about recovery from STRUCTURAL DAMAGE. full32 = "
        "keep_front_layers 32 / n_fresh_layers 0: all 32 pretrained layers "
        "present, nothing transplanted, nothing pruned. It is a "
        "CONTINUED-PRETRAINING control, so every statement here is about CPT "
        "DRIFT on the heal corpus. The caveat in "
        "shallow_rung_ni_discrimination_20260812"
        ".the_load_bearing_new_finding.caveat continues to travel with it.",
        "ANY claim that these 7B deficits are large or small 'relative to seed "
        "variance'. sd_run is 1B-ONLY (S=3, keep12@5000). Every 7B rung has "
        "EXACTLY ONE seed and the historical 7B seeds are UNRECORDED (--seed "
        "postdates them; trainer afdfa66 called no seeding function), so no 7B "
        "sd_run is computable or retrospectively reconstructible. The "
        "deficit/sd_run column in the JSON is labelled cross-scale extrapolation "
        "only.",
        "treating the five checkpoints as REPLICATES of each other -- they are "
        "five states of ONE optimisation run, so their spread is training "
        "progress plus data order, not independent-run variance. The triviaqa "
        "collapse may not be attributed to, or excused as, seed variance.",
        "calling any difference here 'harness noise': "
        "full32_rescore_v2_20260812.correction_to_the_jitter_premise established "
        "there is NO measured runtime-jitter floor on this harness (same-code "
        "re-runs are bit-identical). These are five DIFFERENT models so "
        "bit-identity does not apply -- but 'noise' is equally unavailable as an "
        "explanation.",
        "any claim that the mmlu_content 20000->25000 move is real: its CI lower "
        "bound landed EXACTLY at 0.0000 while p=0.0508, so the two criteria "
        "disagree. Recorded as criteria_disagree=true and treated as NOT "
        "resolved (conservative AND of CI-excludes-zero AND p<0.05). Picking the "
        "favourable criterion would turn a tie into a result.",
        "substituting `contains` for EM. The EM-vs-contains diagnostic "
        "characterises WHAT an EM move consists of; it does not re-score any "
        "cell, and every verdict here is computed on EM.",
        "any K1/K2/K3 clause: they are defined over the pre-registered 1B arm "
        "set and a 7B trajectory cannot fire them. STATUS.json:warning's "
        "two-corpora / unequal-steps caveat also still holds.",
    ]),
    ("consequences_for_A04", collections.OrderedDict([
        ("1",
         "full32@step25000 must stop being called A04's accept -- it is the only "
         "ckpt on its own trajectory that FAILS the bar, and it was the arm only "
         "because it was the last save."),
        ("2",
         "the 'RATIO is too permissive' claim weakens considerably: it rested on "
         "a disagreement at ONE ckpt whose RATIO margin was +0.0015, while four "
         "earlier ckpts on the same trajectory show BOTH rules accepting with "
         "RATIO 20-32x further from rho. The endpoint is where RATIO's decay "
         "happens to cross NI's, not where the rules differ in principle."),
        ("3",
         "the 2.0.2 neighbour precondition is vindicated BY USE -- it fired on "
         "triviaqa|step20000, an accept that does not survive its upper "
         "neighbour."),
        ("4",
         "'heal longer' is now falsified at 7B on BOTH a damaged arm "
         "(keep14_trajectory_ni_20260813: 8-244x the whole heal budget away, "
         "negative popqa slope) and an UNDAMAGED one (here: more CPT moved 3 of "
         "4 axes AWAY from the anchor). Any future tranche priced on that "
         "premise is mispriced."),
    ])),
    ("cheapest_next_step",
     "NONE outstanding on these arms. outputs/olmo2_probe2_7B_full32_dolmino/ "
     "holds only these five saves, so the (20000, 25000] reject boundary cannot "
     "be narrowed without new training. cheap_next_steps_dominate[1] is now "
     "COMPLETE on both halves (keep14 in 517c8d2, full32 here)."),
    ("staged_copies_cleanup",
     "the four staged ckpts (outputs/a04_staged/full32_step*_from_wzc1.pt, 326 "
     "GiB) were REMOVED after the evidence JSON was written and copied to wzc1; "
     "the sources remain on wzc1 and the staging script + the recorded sha256s "
     "make them reproducible. outputs/a04_staged/sg16_step128000_from_wzc1.pt "
     "belongs to neighbour_variability_20260813 and was NOT touched."),
])


def main():
    with open(P) as f:
        original_text = f.read()
    before = json.loads(original_text, object_pairs_hook=collections.OrderedDict)
    if KEY in before:
        sys.exit(f"REFUSING: {KEY} already present in STATUS.json")

    # Serialise ONLY the new key, at the file's own indent=2, and splice it in
    # as text so the 38 existing keys are not re-serialised at all.
    frag = json.dumps({KEY: VAL}, indent=2, ensure_ascii=False)
    inner = "\n".join(frag.split("\n")[1:-1])
    stripped = original_text.rstrip()
    assert stripped.endswith("}"), "STATUS.json does not end with '}'"
    new_text = stripped[:-1].rstrip() + ",\n" + inner + "\n}"

    after = json.loads(new_text, object_pairs_hook=collections.OrderedDict)
    assert list(after.keys())[:len(before)] == list(before.keys()), \
        "existing keys reordered"
    assert len(after) == len(before) + 1, "unexpected key count"
    for k in before:
        assert json.dumps(before[k], sort_keys=True) == \
            json.dumps(after[k], sort_keys=True), f"pre-existing key {k} changed"

    with open(P, "w") as f:
        f.write(new_text)
    print(f"[ok] appended {KEY}: {len(before)} -> {len(after)} keys, "
          f"{len(before)} pre-existing keys verified unchanged")


if __name__ == "__main__":
    main()
