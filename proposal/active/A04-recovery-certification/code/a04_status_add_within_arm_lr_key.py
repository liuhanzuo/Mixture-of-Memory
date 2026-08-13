#!/usr/bin/env python3
"""Append ONE key to A04's STATUS.json: `within_arm_lr_refutation_20260813`.

APPEND-ONLY, ENFORCED AT THE TEXT LEVEL, NOT THE SEMANTIC LEVEL.

Why text-level. On 2026-08-13 two STATUS.json edits in this repo passed a
semantic append-only check (old keys present, same values) while `json.dump`
silently REFORMATTED the whole file -- different `indent`, `ensure_ascii`
flipping non-ASCII into `\\uXXXX`, or key order changing. The diff was then
enormous and the provenance of every untouched key became unreviewable.

So the guarantee enforced here is stronger and much simpler:

    THE NEW FILE'S FIRST len(old)-2 BYTES ARE BYTE-IDENTICAL TO THE OLD FILE'S,
    and everything after that is the closing of the old object plus the one new
    key.

Concretely: the old file ends with "\\n  }\\n}" (its last key's closing brace,
then the root's). We keep every byte up to and including the old last key's
closing "  }", then append ",\\n" + the serialised new key + "\\n}". No existing
byte is rewritten, so `indent`, `ensure_ascii`, key order and float formatting of
the 45 existing keys CANNOT change -- they are never re-serialised at all.

Verified afterwards: (1) byte-prefix identity, (2) the file still parses,
(3) key count 45 -> 46, (4) the 45 old keys' values compare equal, (5) the old
key ORDER is preserved and the new key is last.

CPU only. Idempotent-safe: refuses if the key already exists.
"""
from __future__ import annotations

import json
import os
import sys

NEW_KEY = "within_arm_lr_refutation_20260813"
EXPECTED_OLD_N_KEYS = 45


def build_entry(evidence_path, evidence_sha256):
    d = json.load(open(evidence_path))
    C1 = "cluster1_124000_125000"
    C2 = "cluster2_130000_131000"

    def rng(cl, ax):
        return d["per_cluster"][cl]["per_axis"][ax]["margin_range"]

    lrc = d["lr_by_cluster"]["contrast"]
    pw = d["power_statement"]
    ph = d["post_hoc_supplement_resolved_intervals"]["per_axis"]["triviaqa"]
    return {
        "what": (
            "THE WITHIN-ARM TEST OF THE LR HYPOTHESIS. "
            "keep12_trajectory_monotonicity_20260813 5 generated H_LR from n=3 "
            "ARMS: 'checkpoint-to-checkpoint margin scatter is governed by where "
            "you are on the LR schedule' (keep10 LR 1.24e-5 range 1.2149 pp > "
            "keep8 6.80e-6 1.1202 > keep12 3.25e-6 0.1951; depth does NOT order "
            "them). It recorded its own confound -- in these runs 'late' == 'low "
            "LR' -- and proposed a ~3.5 GPU-h probe: an EARLY high-LR 500-step "
            "triple on keep12. MAIN verified that probe needs new TRAINING (no "
            "arm has an early 500-step triple: keep12 starts at 124000, keep8's "
            "only early ckpt 45000 has no neighbour, keep10 starts at 83500) and "
            "identified a zero-GPU substitute: neighbour_variability_20260813 "
            "scored TWO keep8 clusters and only ever reported cluster2. "
            "cluster1 (124000/124500/125000) vs cluster2 (130000/130500/131000) "
            "is the same arm, same depth, same corpus, same protocol -- only the "
            "schedule position differs. This recomputes both clusters' canonical "
            "NI-MARGIN ranges (not accuracy ranges) with each cluster gated by "
            "its OWN sigma, measures the LR from the training logs, and verifies "
            "the seam. EVAL-FREE, ZERO GPU, CPU-only re-analysis of shards "
            "already on disk."),
        "verdict": d["headline_verdict"],
        "verdict_label": d["verdict_label"],
        "document": "A04_WITHIN_ARM_LR_REFUTATION_VERDICT.md",
        "prereg": "A04_WITHIN_ARM_LR_PREREG.md",
        "prereg_commit": "5e9b6fb",
        "prereg_note": (
            "its own commit, made BEFORE the first canonical range for this "
            "comparison was recomputed. It fixed the five verdict labels and "
            "their thresholds (R <= 0.83 refutes, R >= 1.20 supports, either "
            "cluster sub-noise => UNRESOLVED_SUBNOISE), the direction convention "
            "(cluster1 = HIGHER LR, so H_LR predicts R > 1), the +-20% boundary "
            "and its justification, and the banned re-wordings. It ALSO recorded "
            "the 1.11x LR contrast and the resulting underpowering BEFORE any "
            "range was read, i.e. it wrote down in advance that the design "
            "probably could not answer Q2. The label is emitted MECHANICALLY by "
            "the script from those criteria."),
        "evidence": "evidence/a04_within_arm_lr.json",
        "evidence_sha256": evidence_sha256,
        "code": ["code/a04_within_arm_lr.py",
                 "code/a04_status_add_within_arm_lr_key.py"],
        "gpu_h_spent": 0,
        "gpu_h_note": (
            "ZERO. No model was loaded and no GPU was used. Both clusters' "
            "per-example shards were already on disk from "
            "neighbour_variability_20260813, whose 8.6556 GPU-h is counted "
            "THERE. nvidia-smi on .82 read 8 x 0 MiB before and after. NOT "
            "touched: LOCAL/.21 (SparseForge #246, 8 cards each), .104 (paperC "
            "Qwen3 heal), .73 (idle but numpy 2.5.1, wrong version for this job)."),
        "node": (
            ".82 (8xH20, zwfy6, numpy 2.4.6) -- DELIBERATELY the same node and "
            "numpy version that published a04_neighbour_variability.json, so the "
            "documented cross-node multinomial split (19/10000 rows between "
            "2.4.6 and 2.5.1) cannot be operating on the comparison."),
        "answers": {
            "Q1_canonical_margin_ranges": {
                "note": ("NI-MARGIN range (= diff_lower95_one_sided_pp + "
                         "delta_pp), split convention, k=3 so the gate constant "
                         "is 3/sqrt(pi) = 1.6925687506432689 for BOTH clusters; "
                         "sigma is each cluster's own mean bootstrap SE"),
                "cluster1_HIGHER_LR_7p63e6_SEAM": {
                    ax: {"range_pp": rng(C1, ax)["range_pp"],
                         "floor_pp": rng(C1, ax)[
                             "expected_range_if_pure_noise_pp"],
                         "ratio_to_floor": rng(C1, ax)[
                             "range_over_expected_noise_range"],
                         "clears": rng(C1, ax)["range_exceeds_item_noise"]}
                    for ax in ("triviaqa", "popqa", "mmlu_content", "nq_open")},
                "cluster2_LOWER_LR_6p86e6_CLEAN": {
                    ax: {"range_pp": rng(C2, ax)["range_pp"],
                         "floor_pp": rng(C2, ax)[
                             "expected_range_if_pure_noise_pp"],
                         "ratio_to_floor": rng(C2, ax)[
                             "range_over_expected_noise_range"],
                         "clears": rng(C2, ax)["range_exceeds_item_noise"]}
                    for ax in ("triviaqa", "popqa", "mmlu_content", "nq_open")},
                "n_decision_axis_cells_clearing_gate": 1,
                "which_one": "cluster2 / triviaqa, 1.1034 pp at 1.68x its floor",
            },
            "Q2_is_H_LR_refuted": {
                "answer": "NO -- and NOT supported either: UNRESOLVED_SUBNOISE",
                "R_triviaqa_hi_over_lo": d["R_hi_lr_over_lo_lr"],
                "both_clusters_clear_their_own_gate": d[
                    "both_clusters_clear_their_own_gate"],
                "why_undefined": (
                    "cluster1's triviaqa range (0.2842 pp) is INSIDE its own "
                    "item-noise floor (0.6518 pp, 0.44x). A max-minus-min of 3 "
                    "noisy cells is biased upward even at ZERO true spread, so "
                    "0.2842 pp is not an estimate of anything and the ratio of "
                    "the two ranges is undefined. Dividing a measurement by a "
                    "non-measurement is not a refutation. R = 0.2576 IS in "
                    "H_LR's opposite direction and is reported, but per the "
                    "prereg it may not be read either as 'H_LR is dead' or as "
                    "'consistent with noise so H_LR is fine'."),
                "three_independent_blockers": [
                    "SUB-NOISE: cluster1 fails its own gate on all FOUR axes "
                    "(0.17-0.44x floor).",
                    "UNDERPOWERED BY CONSTRUCTION: the two clusters differ in LR "
                    "by only " + f"{lrc['lr_ratio_mean']:.4f}x"
                    " (mean, measured from the trainer's own logged lr= lines) "
                    "against the "
                    f"{pw['crossarm_lr_ratio_keep10_over_keep12_measured']:.4f}x"
                    " cross-arm contrast H_LR was fitted on -- "
                    f"{100 * pw['within_over_crossarm_excess_ratio_scale']:.2f}%"
                    " of it on the excess-ratio scale, "
                    f"{100 * pw['within_over_crossarm_log_scale']:.2f}% on the "
                    "log scale. The clusters are 6000 steps apart on the decayed "
                    "TAIL of a 200000-step cosine, not a high-vs-low pair.",
                    "SEAM: cluster1 straddles a resume seam (124000/124500 from "
                    "the .73 process that died 2026-08-08 20:26 TCPStore; 125000 "
                    "from a DIFFERENT .82 process 2026-08-12 resuming FROM "
                    "step124500.pt), and the trainer rebuilds the loader without "
                    "intra-epoch fast-forward. Worse: cluster1's ONLY resolved "
                    "interval on ANY axis IS the seam-crossing 124500->125000 "
                    "one, so the single datum that could carry a refutation is "
                    "precisely the one the seam invalidates.",
                ],
                "consequence_for_MAINs_premise": (
                    "MAIN's within-arm-control IDEA is sound, but THIS pair of "
                    "clusters cannot execute it. H_LR remains an UNTESTED n=3 "
                    "observation -- neither strengthened nor weakened. Testing it "
                    "properly requires TRAINING a 500-step triple early in some "
                    "arm's schedule (LR >= ~1.5e-5). That is NOT recommended, "
                    "because the Q3 gate wording below is correct whether H_LR is "
                    "true or false."),
                "label_is_seed_invariant": d[
                    "verdict_label_seed_sensitivity"]["label_is_seed_invariant"],
            },
            "Q3_how_2p5_should_read": {
                "the_reconciliation": (
                    "the three candidate phrasings are all TRUE and are not in "
                    "conflict, because 2.5's number is being asked to do TWO "
                    "JOBS and is valid for one only. As a REPORTING TRIGGER / "
                    "upper bound on how much a hand-picked accept could be "
                    "overstated, ~1.2 pp (triviaqa) / <=0.35 pp (popqa, "
                    "mmlu_content) is a solid measured maximum over three arms "
                    "and three clean clusters. As a THRESHOLD an accept must "
                    "CLEAR, it is unavailable, because the statistic is dispersed "
                    "on at least two axes: across arms (triviaqa 0.1951-1.2149 "
                    "pp, 6.23x) and WITHIN one arm across schedule position "
                    "(keep8 triviaqa 0.2842 -> 1.1034 pp, 3.88x, at only a 1.11x "
                    "LR difference). So no single number -- per gate, per arm, or "
                    "per LR -- is defensible as a tolerance to clear."),
                "keep10_candidate": "'~1.2 pp unchanged, now two arms' -- right "
                                    "about the UPPER BOUND and its replication",
                "keep12_candidate": "'conditional, 6.2x cross-arm spread' -- "
                                    "right that it is not a constant",
                "this_run_candidate": "'even within one arm two positions differ "
                                      "3.9x' -- right that it is not even a "
                                      "property of the arm; it is POSITIONAL",
                "proposed_wording": (
                    "see A04_WITHIN_ARM_LR_REFUTATION_VERDICT.md 7 for the "
                    "verbatim replacement text. In brief: (a) the precondition "
                    "is a REPORTING requirement, not a threshold; (b) the pp "
                    "figures are UPPER BOUNDS usable only to decide whether a "
                    "move is worth flagging and to bound overstatement, never to "
                    "license an accept that clears them; (c) every range quoted "
                    "anywhere must carry its own item-noise gate with the "
                    "constant matched to k (k=2 -> 1.128379, k=3 -> 1.692569, "
                    "k=5 -> 2.325929) and sigma from THAT cluster's own SEs; "
                    "(d) NOT conditioned on LR, because H_LR is untested."),
                "gate_census_over_the_three_CLEAN_clusters": (
                    "9 decision-axis range cells (keep8 c2, keep10, keep12 Q4 x "
                    "triviaqa/popqa/mmlu_content): EXACTLY 2 clear their gate, "
                    "and both are triviaqa (keep8 c2 1.1202 pp @1.70x; keep10 "
                    "1.2149 pp @1.84x). Adding keep8's seam cluster c1 makes it "
                    "2 of 12. Axis-concentrated on triviaqa; blanket distrust of "
                    "single-checkpoint numbers remains unsupported."),
                "no_new_gpu_needed": (
                    "the rewording only WEAKENS what the gate may claim (it "
                    "removes a threshold reading that was never measured) and "
                    "keeps every measured number, so it can be adopted as-is."),
            },
        },
        "post_hoc_supplement": {
            "IS_POST_HOC": True,
            "does_not_set_the_verdict": True,
            "what": ("the pre-registered statistic is the RANGE, undefined here. "
                     "The adjacent-interval paired bootstrap is a proper test "
                     "(own CI, own p) and IS defined regardless of the gate, so "
                     "suppressing it because the pre-registered statistic came "
                     "out undefined would itself be a selection effect."),
            "cluster1_largest_resolved_triviaqa_move_pp": ph[C1][
                "largest_resolved_abs_move_pp"],
            "cluster1_interval": ph[C1]["largest_resolved_interval"],
            "cluster2_largest_resolved_triviaqa_move_pp": ph[C2][
                "largest_resolved_abs_move_pp"],
            "cluster2_interval": ph[C2]["largest_resolved_interval"],
            "R": ph["R_hi_over_lo"],
            "inadmissible_because": (
                "post-hoc AND cluster1's only resolved interval IS the "
                "seam-crossing 124500->125000 one. The two defects are not "
                "redundant; either alone blocks the inference."),
            "what_it_DOES_license": (
                "with NO LR interpretation: within one arm, at two positions "
                "6000 steps apart, the largest resolved adjacent-500-step "
                "triviaqa move differs by ~3.9x. That is a statement about the "
                "SPREAD of the statistic across positions, not about which "
                "position is larger, so it survives the seam caveat -- and it is "
                "the empirical basis for Q3."),
        },
        "new_defect_found": {
            "what": ("the NI-margin range is reproducible to ~0.03 pp ACROSS "
                     "BOOTSTRAP SEED CHOICES, not to the 5e-4 pp an existing "
                     "hard-fail assertion demands. A SECOND latent tooling "
                     "hard-fail, independent of the numpy one in "
                     "neighbour_variability_20260813 4.1."),
            "measured_drift_vs_archive_pp": d[
                "reproduction_vs_archive"]["max_abs_margin_drift_pp"],
            "it_is_NOT_the_numpy_split": (
                "both this run and the archive are on .82 / numpy 2.4.6, so the "
                "documented cross-version drift (0.005294 pp max, triviaqa ONLY) "
                "cannot be operating -- and these drifts reach 0.0285 pp and "
                "touch mmlu_content and nq_open."),
            "diagnosis_executed_not_asserted": (
                "the BOOTSTRAP-FREE accuracy range reproduces the archive "
                "EXACTLY on all 8 cells (proving item set / metric / null / "
                "Delta / shards are identical; the script HARD-FAILS if not), "
                "while ni_rule's seed is SEED + 97*arm_index + 13*axis and "
                "assert_seeds_disjoint FORBIDS reusing the archive's arm_index "
                "400-405. Re-running the same 24 cells with the ARCHIVE's own "
                "arm_index reproduces the archive 24 of 24 at max drift 0.0 pp."),
            "consequence": (
                "the seed-disjointness rule and bit-exact margin reproduction "
                "are MUTUALLY EXCLUSIVE for this estimator. The rule wins; the "
                "cost is ~0.03 pp of margin precision. Every gate boolean is "
                "unchanged and the quantities compared here are 10-40x the "
                "drift. But any assertion demanding 5e-4 pp MARGIN reproduction "
                "is valid only at FIXED node AND FIXED arm_index."),
            "may_not_be_used_as_an_excuse_above_pp": 0.03,
            "second_finding_assert_seeds_disjoint_is_weaker_than_it_looks": (
                "it scans ONE evidence/ dir, but zwfy6's copy is MISSING 12 json "
                "archives that wzc1 has -- including a04_sigma_run_postfix.json "
                "(arm_index 900-902) and a04_step100k_plateau_vs_ni.json "
                "(100-102). A single-disk scan would pass on a collision with an "
                "archive that exists only on the other disk. This run added "
                "--extra_evidence_dirs and scanned BOTH disks (the 12 wzc1-only "
                "files staged to .82 with md5 verified). ALWAYS pass both."),
        },
        "verification": {
            "protocol_from_invocation_fail_closed": (
                "mmlu_bs=16 cb_bs=32 parsed from the DRIVER START header of "
                "logs/a04_nbr_keep8_legA.out plus every per-axis 'START ... bs=' "
                "line; driver source defaults as corroboration only. NEVER from "
                "summary.json:meta, which records neither batch_size nor "
                "chat_template. Any deviation => no output file."),
            "add_bos": "asserted `is False` (never `is not True`) on all 18 dirs",
            "max_new_tokens": "32 asserted on all 12 generative dirs",
            "chat_template": (
                "False, established STRUCTURALLY -- neither harness contains a "
                "chat-template code path (only a docstring mentions it). OLMo-2 "
                "is a BASE LM."),
            "shard_integrity": (
                "28 of 28 cells clean: index set EXACTLY {0..7} as a SET (not a "
                "file count), merged n exactly EXPECTED_N (triviaqa 17944 / "
                "popqa 14267 / nq_open 3610 / mmlu 14042), 0 duplicate item_ids, "
                "0 nan, identical item_id sequences across all 7 arms."),
            "everything_imported": (
                "build_nulls, ni_rule, ratio_rule, AXES, DEMOTED_AXES, "
                "EXPECTED_N, PREREG from pilot_zero_rule_disagreement; "
                "paired_bootstrap, TIE_CONVS, N_BOOT, SEED from A03's "
                "analyze_1b_knowledge_floor; ANCHOR, _load_arm, assert_aligned "
                "from a04_shallow_rung_ni_7b; and range_report, guard_cell, "
                "protocol_asserted, shard_integrity_report, "
                "adjacent_interval_tests, output_shape_and_flips, "
                "EXPECTED_RANGE_OVER_SD, LEG_A_CLUSTERS, ARM_ARCH from "
                "a04_neighbour_variability -- the SAME code objects that "
                "produced the archived keep8 numbers. Delta and anchor never "
                "substituted (G0/G2). No metric, null, rule or guard re-derived."),
            "seed_check_unweakened": (
                "the FIXED self-excluding assert_seeds_disjoint from "
                "a04_keep12_trajectory_monotonicity, used as-is. arm_index "
                "1000-1005, guard 6700, interval 6900, disjoint from {0,1}, "
                "100-102, 200-203, 300-301, 400-408, 500-503, 600-610, 700-702, "
                "800-801, 900-902 -- checked mechanically on BOTH disks. It "
                "correctly REFUSED a re-run written to a different filename, "
                "which is exactly the collision it exists for."),
            "gate_constants_re_derived_not_trusted": (
                "closed forms 2/sqrt(pi)=1.1283791670955126 and "
                "3/sqrt(pi)=1.6925687506432689 checked against the table, PLUS "
                "Monte Carlo (600k draws) for k=2/3/5/8 -> 1.1284 / 1.6929 / "
                "2.3266 / 2.8465. k=5 and k=8 are computed ONLY to show 1.6926 "
                "is specific to k=3; neither enters. range_report is shown to "
                "take k from its INPUT LENGTH, and the mistake is quantified: a "
                "range of 1.50 vs mean SE 1.00 gives gate=False under the "
                "correct k=3 floor and gate=True under a wrong k=2 floor -- the "
                "boolean FLIPS."),
            "sigma_is_per_cluster_proven_structurally": (
                "range_report is called ONCE PER CLUSTER with that cluster's own "
                "SE list, so pooling is impossible by construction. Executed "
                "self-test: two triples with IDENTICAL ranges but different SEs "
                "yield different floors and different booleans (True vs False). "
                "cluster1 mean SE 0.385101 -> floor 0.651809; cluster2 0.388489 "
                "-> floor 0.657544."),
            "lr_measured_never_copied": (
                "parsed from the trainer's own `[step N/200000] ... lr=` lines in "
                "BOTH training logs, cross-checked against the trainer's own "
                "cosine (train_semantic_bottleneck_1b.get_lr, base 2e-5, min "
                "2e-6, warmup 150, max 200000 -- confirmed from the run's own "
                "[optim] group banner showing all four groups share those "
                "values) at the logs' 3-sig-fig precision, with the tolerance "
                "DERIVED from the printed precision rather than guessed, and "
                "checked against recorded literals. A step logging two different "
                "LRs in two logs would be fatal. Cross-arm LRs likewise "
                "re-measured from keep10's and keep12's own logs -- keep12 "
                "measures 3.26e-6 at step166000 vs the 3.25e-6 in the keep12 "
                "verdict's table (0.3%, changes no ordering)."),
            "seam_reconstructed_from_logs": (
                "not read from the archive's flag: which process wrote each save "
                "is reconstructed from which log contains the save line, and a "
                "cluster is CLEAN iff all its saves are in ONE log with no "
                "[resume] banner between the first and the last. .73 saves "
                "[121500..124500], .82 saves [125000..131000]; each log has "
                "exactly ONE resume banner. Result asserted to AGREE with the "
                "archive's own flag on both clusters."),
            "archive_read_back_and_asserted": (
                "both clusters' 8 margin ranges and 8 gate booleans checked "
                "against recorded literals (+-5e-4 pp) and "
                "cluster2.resume_seam is False asserted. A drifted archive "
                "raises."),
            "label_seed_invariance": (
                "the verdict is recomputed under BOTH bootstrap seed choices "
                "(this run's mandated 1000-1005 and the archive's 400-405): "
                "R = 0.2576 vs 0.2388, both_clear = False in both, label "
                "IDENTICAL. The script ABORTS if the label is seed-dependent."),
            "within_node_determinism": (
                "re-running to the same output path gives ZERO statistical "
                "differences; the only delta is the self-exclusion bookkeeping "
                "entry naming its own file."),
            "verdict_doc_numbers_machine_checked": (
                "73 of 73 numeric literals in "
                "A04_WITHIN_ARM_LR_REFUTATION_VERDICT.md were verified present "
                "in evidence/a04_within_arm_lr.json by a script, after two "
                "per-step margin rows and one gate-census count (3 of 9 -> the "
                "correct 2 of 9) were found wrong in the draft and fixed."),
        },
        "prereg_own_error_recorded_not_fixed": (
            "A04_WITHIN_ARM_LR_PREREG.md 3 wrote that the within-arm LR contrast "
            "is '~13 %' of the cross-arm one. That matches NONE of the three "
            "defensible scalings (excess-ratio 3.99 %, log 7.93 %, raw-ratio "
            "29.23 %). The prereg's QUALITATIVE point -- underpowered -- is "
            "unaffected and is STRENGTHENED, since the true fraction on both "
            "natural scales is smaller than 13 %. Recorded in "
            "power_statement.prereg_correction rather than edited, because the "
            "prereg is committed and its arithmetic is part of the record."),
        "NOT_licensed": [
            "'H_LR is refuted' -- the verdict is UNRESOLVED_SUBNOISE.",
            "'H_LR is confirmed' or 'supported' -- n=3 arms plus a 1.11x "
            "within-arm contrast cannot confirm a schedule law.",
            "quoting R = 0.2576 (or the post-hoc 0.2564) as an EFFECT SIZE: its "
            "numerator is sub-noise, so the ratio is undefined and is reported "
            "only because the prereg fixed the formula in advance.",
            "treating the two clusters, or the checkpoints within one, as "
            "REPLICATES -- they are successive states of ONE optimisation; the "
            "range is a checkpoint-SELECTION quantity, never seed variance. No "
            "7B sd_run exists or is reconstructible.",
            "promoting cluster1 to a clean 500-step neighbourhood.",
            "using the ACCURACY range in place of the MARGIN range for any "
            "gate-design statement, even though MAIN's accuracy arithmetic all "
            "reproduces to <1e-3 pp -- the failure was the MISSING NOISE GATE, "
            "not the arithmetic.",
            "reporting the 7 sub-noise ranges (0.119-0.305 pp) as measured gaps.",
            "citing the numpy multinomial split to explain the ~0.03 pp drifts -- "
            "both runs are on the same node and numpy; the cause is the seed and "
            "it is proven.",
            "quoting any margin better than 0.01 pp across nodes or ~0.03 pp "
            "across seed choices.",
            "any K1/K2/K3 clause -- defined over the pre-registered 1B arm set.",
            "comparing keep8 / keep10 / keep12 / keep14 / shortgpt16 margins as "
            "rungs of one ladder (different architectures).",
        ],
        "recommendation": (
            "(1) ADOPT the 2.5 rewording in "
            "A04_WITHIN_ARM_LR_REFUTATION_VERDICT.md 7 -- it costs no GPU, "
            "weakens only what was never measured, and resolves the three-way "
            "conflict by separating 'reporting trigger (upper bound)' from "
            "'certification threshold (unavailable)'. (2) RECORD H_LR as "
            "UNTESTED rather than 'supported by 3 arms', and do NOT fund the "
            "training needed to test it -- the gate text is correct either way. "
            "(3) Always pass BOTH disks' evidence dirs to "
            "assert_seeds_disjoint. (4) Any future 5e-4 pp margin-reproduction "
            "assertion must pin node AND arm_index."),
    }


def main():
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} STATUS.json EVIDENCE.json SHA256")
    status_path, ev_path, sha = sys.argv[1], sys.argv[2], sys.argv[3]

    old_bytes = open(status_path, "rb").read()
    old = json.loads(old_bytes.decode("utf-8"))
    old_keys = list(old.keys())
    if len(old_keys) != EXPECTED_OLD_N_KEYS:
        raise SystemExit(
            f"FATAL: STATUS.json has {len(old_keys)} keys, expected "
            f"{EXPECTED_OLD_N_KEYS}. Someone else edited it; resolve first.")
    if NEW_KEY in old:
        raise SystemExit(f"FATAL: key {NEW_KEY} already present; refusing.")

    entry = build_entry(ev_path, sha)

    # --- TEXT-LEVEL APPEND -------------------------------------------------
    # find the last '}' (root close) and, before it, the previous '}' which
    # closes the final existing key. Keep every byte up to and including that
    # inner '}', then append ",\n<newkey>\n}".
    txt = old_bytes.decode("utf-8")
    root_close = txt.rstrip()
    if not root_close.endswith("}"):
        raise SystemExit("FATAL: STATUS.json does not end with '}'")
    i_root = txt.rindex("}")
    i_last_key_close = txt.rindex("}", 0, i_root)
    prefix = txt[:i_last_key_close + 1]

    body = json.dumps({NEW_KEY: entry}, indent=2, ensure_ascii=False,
                      default=float)
    # strip the wrapper braces and re-indent the single key by 2 spaces
    inner = body[body.index("\n") + 1: body.rindex("\n")]
    new_txt = prefix + ",\n" + inner + "\n}"
    new_bytes = new_txt.encode("utf-8")

    # GUARD 1: byte-prefix identity -- no existing byte may be rewritten.
    if new_bytes[:len(prefix.encode("utf-8"))] != prefix.encode("utf-8"):
        raise SystemExit("FATAL: byte prefix changed; refusing to write.")

    tmp = status_path + ".tmp_within_arm_lr"
    with open(tmp, "wb") as f:
        f.write(new_bytes)

    # GUARD 2-5: parses, key count, old values equal, order preserved + new last
    new = json.loads(open(tmp, encoding="utf-8").read())
    if len(new) != EXPECTED_OLD_N_KEYS + 1:
        os.remove(tmp)
        raise SystemExit(f"FATAL: new key count {len(new)} != "
                         f"{EXPECTED_OLD_N_KEYS + 1}")
    for k in old_keys:
        if new[k] != old[k]:
            os.remove(tmp)
            raise SystemExit(f"FATAL: existing key {k} changed value")
    if list(new.keys()) != old_keys + [NEW_KEY]:
        os.remove(tmp)
        raise SystemExit("FATAL: key order changed or new key is not last")

    os.replace(tmp, status_path)
    print(f"[status] appended {NEW_KEY}: {EXPECTED_OLD_N_KEYS} -> {len(new)} keys")
    print(f"[status] byte-prefix identity held over "
          f"{len(prefix.encode('utf-8'))} of {len(old_bytes)} old bytes")
    print(f"[status] gpu_h_spent = {new[NEW_KEY]['gpu_h_spent']}")


if __name__ == "__main__":
    main()
