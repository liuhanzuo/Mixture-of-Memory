#!/usr/bin/env python3
"""Append `keep10_neighbour_range_20260813` to A04's STATUS.json — APPEND ONLY.

WHY THIS IS A SCRIPT AND NOT A HAND EDIT
----------------------------------------
STATUS.json currently has 39 top-level keys and is the index the next agent reads.
A hand edit (or an LLM rewrite) can silently reorder, reformat or drop one, and the
loss would not be visible in a diff of a 200 KB file. So the mutation is mechanical
and it ASSERTS, before and after writing:

  * every pre-existing key still exists, in the SAME ORDER;
  * every pre-existing key's value is BYTE-IDENTICAL under
    `json.dumps(sort_keys=True)` -- not `==` on the parsed object, which would
    tolerate a float reformat;
  * exactly ONE key was added, and it is the expected name;
  * the file re-parses and round-trips.

Any failure leaves the original file untouched (write is to a temp path, then
os.replace only after all assertions pass).
"""
from __future__ import annotations

import json
import os
import sys

NEW_KEY = "keep10_neighbour_range_20260813"

ENTRY = {
    "what": (
        "SECOND ARM for the neighbour-range test. "
        "neighbour_variability_20260813 measured the NI-margin range across three "
        "500-step-spaced checkpoints of keep8+fresh2 and got SIX ranges, of which "
        "EXACTLY ONE cleared the item-noise gate (triviaqa 1.1202 pp = 1.70x "
        "E[range|noise]). A04_GATE_DESIGN.md 2.0.2 (the neighbour precondition on "
        "any reported accept) and 2.5 (its per-axis tolerance) both rest on that "
        "single cell, and 2.5 says the numbers 'should be widened if a second arm "
        "is ever measured'. This scores keep10+fresh2 step89000/89500/90000 -- a "
        "500-step triple never previously scored on any axis, on a THIRD damage "
        "level (keep_front=10 => 12 layers) -- on all four axes at the frozen "
        "protocol, and gates it with the SAME IMPORTED CODE OBJECT that gated "
        "keep8. Also restates both arms in Heineman et al.'s (arXiv:2508.13144, "
        "NeurIPS 2025 Spotlight) rel.std unit against their published intact "
        "OLMo-2 7B values. EVAL-ONLY, zero training."),
    "verdict": "KEEP10_NEIGHBOUR_RANGE_REPLICATES_MATERIAL_TRIVIAQA",
    "prereg": "A04_KEEP10_NEIGHBOUR_RANGE_PREREG.md",
    "prereg_commit": "0e889cd",
    "prereg_note": (
        "committed as its own commit BEFORE the first checkpoint was scored, so "
        "the readings' git timestamp precedes every number. The pre-registration "
        "explicitly wrote down the outcome that would WEAKEN us ('if keep10's "
        "triviaqa range fails the gate, keep8's 1.1202 pp is an isolated cell and "
        "2.5 loses its claim to generality') and forbade re-describing a "
        "non-replication as 'consistent with noise, so no problem'."),
    "document": "A04_KEEP10_NEIGHBOUR_RANGE_VERDICT.md",
    "evidence": "evidence/a04_keep10_neighbour_range.json",
    "evidence_sha256": (
        "0838a67edda0380e98355b5c723b3be59101d16e3366611dbd383933f8eddc52"),
    "code": [
        "code/a04_keep10_neighbour_range_driver.sh",
        "code/a04_keep10_neighbour_range.py",
    ],
    "gpu_h_spent": 3.4911,
    "gpu_h_note": (
        "driver wall-clock 13:43:16 -> 14:09:27 = 1571 s x 8 GPUs = 3.4911 GPU-h; "
        "per-checkpoint 509 / 522 / 527 s for all four axes. Analysis is "
        "CPU-only."),
    "nodes": (
        ".82 ONLY (8xH20, zwfy6, numpy 2.4.6). Verified 8x0 MiB / 0 % / no compute "
        "processes before launch and 8x0 MiB after. The driver refuses to start if "
        ">8000 MiB is held AND refuses to run on a node whose numpy != 2.4.6 (the "
        "sampler that published the keep8 numbers being replicated). NOT touched: "
        ".73 (keep12 11-ckpt trajectory, running concurrently -- no output_name "
        "collision, prefix A04_7B_keep10f2_NBR_* verified unique before launch), "
        ".104 (paperC Qwen3 heal), LOCAL/.21 (SparseForge #246)."),
    "arm": {
        "dir": "outputs/olmo2_probe2_7B_keep10fresh2",
        "steps": [89000, 89500, 90000],
        "spacing_steps": 500,
        "keep_front": 10, "n_fresh": 2, "num_hidden_layers": 12,
        "n_tensors": 135, "dtype": "fp32", "n_params": 3250786304,
        "resume_seam": False,
        "single_process_provenance": (
            "logs/olmo2_7B_keep10fresh2_resume200k_73.log has EXACTLY ONE "
            "'[resume] loading ckpt ... step86500.pt' banner (03:57:09, epoch=0), "
            "then saved step89000 08:44:52 / step89500 09:42:19 / step90000 "
            "10:39:43, and the process died at 11:15 on a TCPStore error AFTER "
            "all three saves. One process, one loader, continuous data order -- "
            "the keep8-cluster-1 seam trap CHECKED AND CLEARED IN ADVANCE rather "
            "than discovered after scoring."),
        "ckpt_identity_proven": (
            "all three files are exactly 39,009,621,855 B -- byte size is NOT "
            "identity here (keep8's triple also shared one size, and "
            "shortgpt16/step128000.pt was a 7.7 GB truncated write ls -l could "
            "not distinguish from healthy). Distinctness proven by f64 sum of "
            "every parameter (4.3489697564e4 / 4.2742295066e4 / 4.2535548743e4) "
            "and per-tensor sha256 of lm_head / embed_tokens / layers.0.q_proj, "
            "all three distinct on every one."),
        "seed_42_is_not_a_training_seed": (
            "arch_meta.json records seed: 42, but --seed moves only the "
            "fresh-tail init; DistributedSampler(ds, shuffle=True) has no seed=, "
            "so data order is identical across seeds. No sd_run follows."),
    },
    "Q1_per_axis_range_and_gate": {
        "convention": "split",
        "gate": ("range_exceeds_item_noise := range_pp > 1.6926 * mean(per-cell "
                 "bootstrap SE); 1.6926 = 3/sqrt(pi) = E[range of 3 iid N(0,1)], "
                 "exact for the normal. IMPORTED range_report -- the same code "
                 "object that gated keep8."),
        "margins_pp": {
            "triviaqa": [-39.8702, -41.0850, -40.3661],
            "popqa": [-18.4317, -18.7468, -18.7257],
            "mmlu_content": [-11.1282, -10.9431, -11.0356],
            "nq_open_demoted": [-15.8726, -16.2604, -15.5125],
        },
        "range_pp": {"triviaqa": 1.2149, "popqa": 0.3151,
                     "mmlu_content": 0.1852, "nq_open_demoted": 0.7479},
        "expected_range_if_pure_noise_pp": {
            "triviaqa": 0.6595, "popqa": 0.5843,
            "mmlu_content": 0.6351, "nq_open_demoted": 1.1211},
        "range_over_floor": {"triviaqa": 1.8423, "popqa": 0.5392,
                             "mmlu_content": 0.2916, "nq_open_demoted": 0.6671},
        "range_exceeds_item_noise": {
            "triviaqa": True, "popqa": False,
            "mmlu_content": False, "nq_open_demoted": False},
        "n_decision_axes_clearing_gate": 1,
        "three_of_four_ranges_fail_the_gate": (
            "the 0.185-0.748 pp ranges are NOT measured neighbour gaps. A "
            "max-minus-min of 3 noisy cells is biased upward even at zero true "
            "spread. Quoted raw, nq_open's 0.7479 pp would have looked like the "
            "second-largest finding here; it is 0.67x its own noise floor."),
        "range_as_fraction_of_own_delta": (
            "triviaqa 1.2149/6.3291 = 19.2%, but the SAME absolute wobble is "
            "54.1% of popqa's Delta (2.2457) and 65.3% of mmlu_content's "
            "(1.8614) -- i.e. decisive on the axes with tighter Delta."),
        "two_resolved_intervals_in_OPPOSITE_directions": (
            "triviaqa 89000->89500 = -1.2093 pp, CI95 [-1.5270,-0.8972], "
            "p=0.0001, 505 right->wrong vs 288 wrong->right of 17944; then "
            "89500->90000 = +0.7245 pp, CI95 [+0.4235,+1.0310], p=0.0001, 445 vs "
            "315. Also resolved: popqa 89000->89500 (-0.2944 pp, p=0.0014) and "
            "both nq_open intervals. popqa 89500->90000 is an EXACT ZERO with 74 "
            "items flipping each way -- 'no net change' is not 'nothing "
            "happened'."),
        "not_output_degeneracy": (
            "triviaqa across the triple: 0.000% empty predictions throughout, "
            "top_constant_frac FALLING 0.529% -> 0.340% -> 0.334%, distinct "
            "predictions 9208 -> 9599 -> 9358. But only 37.0%/37.8% of triviaqa "
            "prediction STRINGS are identical across the two intervals (popqa "
            "24.3%) -- EM is a coarse read on a model whose outputs are rewritten "
            "wholesale between adjacent saves."),
        "zero_accepts": (
            "0 NI accepts across all 5 tie conventions x 12 cells; RATIO(0.85) "
            "mean ratios 0.3356-0.3474. Recovery fractions at step90000: triviaqa "
            "27.2%, popqa 9.1%, mmlu_content 34.0%. This arm is nowhere near an "
            "accept, which is why the range here is a METHODOLOGICAL quantity and "
            "not an accept being defended."),
        "guard": "all four axes CERTIFIABLE under D1-D6, 0 cells retired (split)",
    },
    "Q2_replication_vs_keep8": {
        "REPLICATES": True,
        "label": "REPLICATES_AND_MATERIAL",
        "headline_axis": "triviaqa",
        "headline_axis_why": (
            "triviaqa is the ONLY axis on which keep8 cleared the gate, so it is "
            "the axis on which replication is defined; the other three keep8 "
            "ranges were sub-noise and there was nothing there to replicate."),
        "range_pp_keep8_vs_keep10": {
            "triviaqa": [1.1202, 1.2149], "popqa": [0.2523, 0.3151],
            "mmlu_content": [0.2208, 0.1852], "nq_open": [0.3324, 0.7479]},
        "keep10_over_keep8": {"triviaqa": 1.08, "popqa": 1.25,
                              "mmlu_content": 0.84, "nq_open": 2.25},
        "ALL_FOUR_GATE_BOOLEANS_AGREE": (
            "stronger than the headline axis alone: not only did the one "
            "supra-noise range reproduce (1.08x), the three sub-noise ones STAYED "
            "sub-noise, on an independently damaged arm."),
        "but_the_SHAPE_does_not_replicate": (
            "keep8's triviaqa was MONOTONE NON-INCREASING (flat, then -1.09 pp: "
            "the LAST checkpoint was the worst). keep10's is a V whose MIDDLE "
            "checkpoint is the minimum (-1.21 then +0.72, both resolved at "
            "p=0.0001). Consequence: the realised best-minus-last advantage is "
            "2.26x SMALLER on keep10 (0.4960 vs 1.1202 pp), and 'average the "
            "final k' (Heineman et al.'s prescription) WOULD have fixed keep10 "
            "but would NOT have fixed keep8's terminal drift."),
        "what_is_arm_independent": (
            "'a single adjacent 500-step checkpoint can misstate the triviaqa "
            "margin by ~1.2 pp' -- the symmetric, direction-free claim."),
        "what_is_NOT_arm_independent": (
            "'later checkpoints are worse' / 'the last checkpoint is the worst'. "
            "That is keep8's shape and it did NOT replicate. Had 2.0.2 been "
            "phrased that way instead of on the RANGE, keep10 would have refuted "
            "it."),
        "absolute_sd_agreement": (
            "keep10 and keep8 triviaqa absolute checkpoint SDs are within 4% of "
            "each other: 0.6086 vs 0.6339 pp. The most reassuring number here."),
    },
    "Q3_heineman_relstd_restatement": {
        "cite": ("Heineman, Hofmann, Magnusson, Gu, Smith, Hajishirzi, Lo, Dodge, "
                 "'Signal and Noise', arXiv:2508.13144"),
        "venue": "NeurIPS 2025 Spotlight",
        "venue_reverified_this_session": (
            "OpenReview note sAFottNlra, venue='NeurIPS 2025 spotlight', "
            "venueid=NeurIPS.cc/2025/Conference, invitation "
            "Submission26329/-/Camera_Ready_Revision present. DBLP has CoRR ONLY "
            "(journals/corr/abs-2508-13144) -> DBLP or S2 alone would misread it "
            "as a preprint."),
        "their_constants_independently_re_extracted": (
            "NOT copied from the repo's prior summary: pulled arXiv:2508.13144v1 "
            "via hy-proxy and ran pdftotext -layout. Table 4 p.22 reads "
            "'TriviaQA 28.15_{0.411/0.015} 47.03_{0.135/0.003} ...' and "
            "'MMLU 14.52_{0.139/0.010} 3.39_{0.078/0.023} ...' -- cells are "
            "SNR_{signal/noise}, so 7B-4T NOISE is 0.003 (TriviaQA) and 0.023 "
            "(MMLU). The bare integers are SNR; misreading that column would "
            "inflate the comparator ~4 orders of magnitude. Their n and spacing "
            "re-verified from the same PDF ('the final 30 intermediate "
            "checkpoints, one checkpoint for every 1000 training steps'), and "
            "their ddof=1. All match the repo's earlier extraction exactly."),
        "our_rel_std": {"keep10_triviaqa": 0.0349, "keep8_triviaqa": 0.0395,
                        "keep10_mmlu_content": 0.0026,
                        "keep8_mmlu_content": 0.0030,
                        "keep10_mmlu_LETTER_interface_matched": 0.0134,
                        "keep10_popqa": 0.0383, "keep10_nq_open": 0.1010},
        "our_absolute_sd_pp": {"triviaqa": 0.6086, "popqa": 0.1700,
                               "mmlu_content": 0.0890, "nq_open": 0.3747,
                               "mmlu_LETTER": 0.3530},
        "their_olmo2_7B_4T_noise": {"triviaqa": 0.003, "mmlu": 0.023,
                                    "popqa": None, "nq_open": None},
        "popqa_and_nqopen_have_NO_comparator": (
            "neither is in their 30-benchmark suite; those cells are LEFT BLANK "
            "and must not be filled with a neighbouring task."),
        "raw_ratio_is_MOSTLY_ARITHMETIC": (
            "raw rel.std ratio on triviaqa is 11.65x, but rel.std = sd/mean and "
            "our arm sits at 17.42% accuracy vs their intact ~0.6-0.7. "
            "Decomposing relstd_ours/relstd_theirs = (sd_ours/sd_theirs) * "
            "(mean_theirs/mean_ours) over a grid for their unpublished mean "
            "(Table 4 gives only Rel.Dispersion and Rel.Std; their mean appears "
            "only in Figure 12, so it is NOT guessed) gives an ABSOLUTE-SD ratio "
            "of 5.07x @0.40, 4.06x @0.50, 3.38x @0.60, 3.12x @0.65, 2.90x @0.70. "
            "The defensible statement is '2.9-5.1x larger absolute triviaqa "
            "checkpoint SD', NOT 11.65x."),
        "MMLU_POINTS_THE_OTHER_WAY": (
            "on mmlu_content we are 9x QUIETER than their MMLU (0.0026 vs 0.023), "
            "and still 1.7x quieter on the interface-matched letter variant "
            "(0.0134 vs 0.023). So 'damage makes everything noisier' is FALSE as "
            "stated: the effect is AXIS-SPECIFIC -- short-form generative EM on a "
            "damaged arm is unstable, multiple-choice-style scoring on the same "
            "arm is MORE stable than the published intact value."),
        "NOT_AN_EQUAL_FOOTING_COMPARISON": (
            "n=3 vs n=30; 500 vs 1000-step spacing; layer-pruned mid-heal 12-layer "
            "arm vs intact 32-layer OLMo-2 7B; STRICT ZERO-SHOT vs FEW-SHOT (their "
            "App. A.1 -- an asymmetry NOT in the repo's prior summary, found by "
            "reading their PDF, and it cuts AGAINST reading a gap as evidence "
            "about injury); this repo's base protocol (cb_bs=32 / add_bos=False / "
            "max_new_tokens=32 / greedy) vs OLMES; content-continuation vs "
            "letter-choice MMLU (the two interfaces disagree on 40.1% of items at "
            "the anchor). MAY NOT BE TABULATED TOGETHER. Any ratio quoted must "
            "carry 'n=3 vs n=30, different protocol' in the same sentence."),
        "n3_estimator_noise": (
            "at n=3 the sample SD's own relative SD is 52.3% (sqrt(1-c4^2)/c4, "
            "c4=0.886227) -- computed and self-tested in the code. A ~2x rel.std "
            "ratio is uninformative BEFORE any protocol difference; only an "
            "order-of-magnitude gap would mean anything, and after decomposition "
            "this one is ~3-5x."),
        "status": (
            "HYPOTHESIS worth a controlled test, not a result. The empirical "
            "foothold survives but narrows to ONE axis, with a decomposed 2.9-5.1x "
            "rather than 11.65x, from n=3, under a different shot setting. It is "
            "NOT strong enough to carry a section alone. The NORMATIVE "
            "contribution -- that in a non-inferiority test neighbour noise is a "
            "ONE-SIDED FREE OPTION rather than a power loss, which Heineman et al. "
            "structurally cannot say because they have no accept -- remains the "
            "load-bearing part."),
    },
    "verification": [
        "STATUS.json had 41 pre-existing keys at append time, NOT the 39 recorded "
        "in the pre-registration: `full32_endpoint_is_not_the_accept_20260813` and "
        "`blockers_discharged_20260813` were added by CONCURRENT A04 work "
        "(commits 7ec11d7 / 6e08c6d) between the prereg commit 0e889cd and this "
        "append. Recorded rather than quietly corrected. The append invariant is "
        "unaffected -- it is 'no pre-existing key is modified', asserted against "
        "the file as read at append time, and 0 of 41 changed -- but the prereg's "
        "'38+/39 keys' figure was stale by two, and the next agent should read the "
        "count from the file rather than from any document.",
        "estimator self-test EXECUTED before publishing: range_report reproduces "
        "E[range of 3] = 3/sqrt(pi) = 1.6925687506 on known inputs and FIRES when "
        "range/SE = 2.0; relstd([1,2,3]) = 0.5; c4(2)=0.7978845608 and "
        "c4(3)=0.8862269255 (textbook). all_ok=True; a failure aborts.",
        "the gate applied to keep10 is the SAME CODE OBJECT that gated keep8: "
        "range_report / adjacent_interval_tests / guard_cell / protocol_asserted / "
        "shard_integrity_report / EXPECTED_RANGE_OVER_SD imported FROM "
        "a04_neighbour_variability; ni_rule / build_nulls / ratio_rule / "
        "EXPECTED_N / AXES / PREREG from pilot_zero; paired_bootstrap / TIE_CONVS / "
        "N_BOOT / SEED from A03; ANCHOR / _load_arm / assert_aligned from "
        "a04_shallow_rung_ni_7b. No metric, null, rule, gate or anchor re-derived, "
        "and NO margin obtained by subtracting a recorded null from a recorded "
        "accuracy.",
        "keep8's numbers READ from its archive, not quoted from prose: "
        "keep8_archive_readback hard-fails if the ranges differ from the "
        "documented 1.1202/0.2523/0.2208/0.3324 by >5e-4 pp or if any gate boolean "
        "moved. It passed, and confirmed from the archive that keep8 was published "
        "on .82/numpy 2.4.6 -- the same node and sampler as this run, so the "
        "comparison is WITHIN one multinomial.",
        "protocol confirmed from the INVOCATION, fail-closed, before anything was "
        "scored: driver echoes 'DRIVER START ... mmlu_bs=16 cb_bs=32' and per-axis "
        "{closedbook:[32], nq_open:[32], mmlu:[16]}; source defaults corroborate. "
        "summary.json:meta records NEITHER batch size NOR chat_template "
        "(A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md). bs is not free: bs32->bs48 "
        "flipped 12/14267 popqa and 10/3610 nq_open.",
        "add_bos is False on all 9 result dirs, asserted with `is False` -- never "
        "`is not True`, which passes silently on None. max_new_tokens==32 on all 6 "
        "generative dirs. chat_template=False established STRUCTURALLY (neither "
        "harness contains a chat-template code path).",
        "shard integrity 16/16 cells clean: index set EXACTLY {0..7} (not a file "
        "count), merged n exactly EXPECTED_N (17944/14267/3610/14042), 0 duplicate "
        "item_ids, 0 nan, identical item_id sequences across all four arms "
        "including the anchor.",
        "bootstrap seeds disjoint, CHECKED AS CODE: arm_index 700-702, guard "
        "SEED+4700, intervals SEED+4900, intersected against every archived "
        "bootstrap_offsets block (full32 203/500-503, keep14 201/300-301, "
        "neighbour_variability 400-408). No archived number can be perturbed.",
        "analysis BYTE-IDENTICAL on re-run on the same node (sha256 0838a67e... "
        "twice), and the evidence JSON is byte-identical on both disks after "
        "scp -O.",
        "nvidia-smi verified 8x0 MiB before launch and 8x0 MiB after; .73's keep12 "
        "job never touched; output prefix A04_7B_keep10f2_NBR_* verified to collide "
        "with nothing on disk before launch.",
    ],
    "tooling_defects_found_and_fixed_here": {
        "1_seed_check_fires_on_its_own_output": (
            "the inherited assert_seeds_disjoint has NO self-exclusion, so the "
            "second run of this script aborted with 'FATAL: arm_index "
            "[700,701,702] already used by a04_keep10_neighbour_range.json' -- its "
            "OWN output from the first run. This is a FALSE POSITIVE that will hit "
            "every idempotent re-run of every A04 analysis carrying this check, "
            "and it arrives as a FATAL that looks like a real collision. Fixed "
            "here by excluding the output path by os.path.realpath (not by name "
            "match, which would let a genuinely different file slip through), with "
            "the exclusion recorded in the JSON so it cannot hide a real clash. "
            "a04_keep12_trajectory_monotonicity.py still has the unfixed version "
            "-- WORTH PORTING."),
        "2_hostname_I_returns_ten_addresses": (
            "on these nodes `hostname -I` returns TEN addresses and 28.82.250.82 "
            "is NOT the first ('28.86.53.217 28.86.81.221 ... 28.82.250.82 ...'). "
            "A node guard written as `hostname -I | awk '{print $1}'` would REFUSE "
            "TO RUN ON THE CORRECT NODE. The guard here matches the whole list AND "
            "requires numpy==2.4.6, which is .82-specific in this cluster and "
            "therefore pins the node of record even if the IP layout changes."),
    },
    "NOT_licensed": [
        "'Later checkpoints are worse' / 'the last checkpoint is the worst.' That "
        "is keep8's SHAPE and it did NOT replicate -- keep10's worst checkpoint is "
        "the MIDDLE one.",
        "Treating the three checkpoints as REPLICATES. Successive states of ONE "
        "optimisation; their spread is heal progress + data order. A "
        "checkpoint-SELECTION quantity, never seed variance. No 7B sd_run exists "
        "or is reconstructible.",
        "Comparing keep10 / keep8 / keep12 / keep14 ABSOLUTE margins as rungs of "
        "one ladder -- four depths, two corpora, unequal steps (see "
        "STATUS.json:warning). Only the RANGES, a within-arm statistic, are "
        "compared across arms.",
        "Reporting the three sub-noise ranges (0.1852 / 0.3151 / 0.7479 pp) as "
        "measured neighbour gaps. They fail range_exceeds_item_noise.",
        "Quoting '11.65x noisier than published OLMo-2' without the decomposition. "
        "The defensible figure is 2.9-5.1x on ABSOLUTE SD, from n=3, and ONLY on "
        "triviaqa -- MMLU points the other way.",
        "Tabulating our rel.std next to Heineman et al.'s as if co-measured.",
        "'Damaged arms are noisier' as a general claim -- true direction on "
        "triviaqa, REVERSED on MMLU (we are 9x quieter on mmlu_content).",
        "Calling any of this 'harness noise' -- same-code re-runs on a fixed "
        "checkpoint are BIT-IDENTICAL "
        "(full32_rescore_v2_20260812.correction_to_the_jitter_premise).",
        "Any K1/K2/K3 clause -- defined over the pre-registered 1B arm set.",
        "Quoting any margin here to better than 0.01 pp ACROSS NODES (numpy "
        "multinomial differs 19/10000 rows between 2.4.6 and 2.5.1; max drift "
        "0.005294 pp, triviaqa only; may NOT explain away any move larger than "
        "~0.006 pp -- every resolved interval here is 55-230x larger).",
    ],
    "consequences_for_the_gate": {
        "1_tolerance_confirmed_not_widened": (
            "GATE_DESIGN 2.5 proposed ~1.2 pp on triviaqa and <=0.35 pp elsewhere "
            "from ONE arm. keep10 independently gives 1.2149 pp on triviaqa and "
            "0.185-0.315 pp on the other decision axes -- within 8% on the axis "
            "that matters. RECOMMEND amending 2.5 to cite TWO arms with the "
            "tolerance UNCHANGED in value."),
        "2_per_axis_phrasing_confirmed_twice": (
            "the same one axis cleared and the same three did not, on two "
            "independently damaged arms. 2.0.2's 'stated PER-AXIS, not blanket' "
            "was correct; blanket distrust of single-checkpoint numbers remains "
            "unsupported."),
        "3_phrase_the_precondition_on_the_RANGE": (
            "2.0.2 already is, and keep10 shows why that matters: a rule phrased "
            "as 'later checkpoints are worse' would have been refuted here."),
        "4_a_third_3point_arm_is_NOT_worth_funding": (
            "two arms now agree on all four gate booleans and on the triviaqa "
            "magnitude to 8%. What is NOT settled is the SHAPE/mechanism, and a "
            "third 3-point cluster cannot settle that -- it needs a DENSER "
            "trajectory on ONE arm, which .73's concurrent keep12 11-checkpoint "
            "scan is already producing. RECOMMEND reading the keep12 trajectory "
            "for the shape question instead of adding a fourth 3-point cluster."),
        "5_heineman_comparison_is_narrow": (
            "axis-specific hypothesis, not a headline. The paper's defensible "
            "novelty remains the equivalence-decision argument."),
    },
}


def detect_format(path):
    """Preserve the file's existing indent width AND its unicode escaping.

    NOT cosmetic, and both halves were found the hard way.

    (1) INDENT. STATUS.json is committed at `indent=2`; writing it back at
        `indent=1` produced a **4517-line diff for a 1-key append** even though 0
        of 41 pre-existing values changed (verified semantically). A reviewer
        cannot see a real modification inside that, and neither can `git log -p`.

    (2) `ensure_ascii`. The file contains **raw UTF-8 CJK** (e.g. the string
        `'提案'` inside `blockers_discharged_20260813`). `json.dump`'s default
        `ensure_ascii=True` rewrites those as `\\u63d0\\u6848`, which is a
        semantically identical but TEXTUALLY different file -- so the append
        silently reflows every line after the first CJK character. The textual
        guard below caught exactly this on the second attempt; without the guard
        it would have shipped as a 4500-line diff and nobody would have looked.

    Both are detected from the file rather than assumed, so this script stays
    correct if STATUS.json is ever reformatted.
    """
    text = open(path, encoding="utf-8").read()
    lines = text.split("\n")
    indent = 1
    if len(lines) > 1:
        n = len(lines[1]) - len(lines[1].lstrip(" "))
        indent = n if n > 0 else 1
    ensure_ascii = text.isascii()
    return indent, ensure_ascii, text


def main():
    path = sys.argv[1]
    indent, ensure_ascii, before_text = detect_format(path)
    blob = json.loads(before_text)
    if not isinstance(blob, dict):
        raise SystemExit("FATAL: STATUS.json is not an object")
    if NEW_KEY in blob:
        raise SystemExit(f"FATAL: key {NEW_KEY} already present -- refusing to "
                         "overwrite. This script is APPEND ONLY.")

    before_keys = list(blob.keys())
    before_vals = {k: json.dumps(blob[k], sort_keys=True) for k in before_keys}
    print(f"pre-existing keys: {len(before_keys)} | indent={indent} "
          f"ensure_ascii={ensure_ascii} (both detected from the file)")

    blob[NEW_KEY] = ENTRY

    tmp = path + ".tmp_append"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=indent, default=float,
                  ensure_ascii=ensure_ascii)

    check = json.load(open(tmp, encoding="utf-8"))
    after_keys = list(check.keys())

    if after_keys[:len(before_keys)] != before_keys:
        os.remove(tmp)
        raise SystemExit("FATAL: pre-existing key ORDER changed -- aborting, "
                         "original untouched")
    if len(after_keys) != len(before_keys) + 1:
        os.remove(tmp)
        raise SystemExit(f"FATAL: key count went {len(before_keys)} -> "
                         f"{len(after_keys)}, expected +1 -- original untouched")
    if after_keys[-1] != NEW_KEY:
        os.remove(tmp)
        raise SystemExit("FATAL: the appended key is not last -- original "
                         "untouched")
    changed = [k for k in before_keys
               if json.dumps(check[k], sort_keys=True) != before_vals[k]]
    if changed:
        os.remove(tmp)
        raise SystemExit(f"FATAL: {len(changed)} pre-existing key(s) MODIFIED: "
                         f"{changed[:8]} -- original untouched")

    # TEXTUAL check on top of the semantic one: the new file must literally begin
    # with the old file's text minus its closing brace. This is what makes the git
    # diff reviewable -- a semantic check alone permits a whole-file reflow, and it
    # is what caught the ensure_ascii bug documented in detect_format().
    stem = before_text.rstrip().rstrip("}").rstrip()
    new_text = open(tmp, encoding="utf-8").read()
    if not new_text.startswith(stem):
        os.remove(tmp)
        raise SystemExit(
            "FATAL: the new file does not textually extend the old one -- the "
            "append reflowed existing lines, which would bury a real change in a "
            "whole-file diff. Original untouched.")

    os.replace(tmp, path)
    print(f"OK appended {NEW_KEY}")
    print(f"keys now: {len(after_keys)} (was {len(before_keys)})")
    print(f"all {len(before_keys)} pre-existing keys byte-identical under "
          "json.dumps(sort_keys=True), order preserved")
    print("existing lines TEXTUALLY unchanged -- the diff is a pure append")
    print(f"gpu_h_spent recorded: {ENTRY['gpu_h_spent']}")


if __name__ == "__main__":
    main()
