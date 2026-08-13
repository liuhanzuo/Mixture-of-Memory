#!/usr/bin/env python3
"""Append ONE key to A04's STATUS.json for the control-arms NI pass.

HARD CONSTRAINT FROM THE DISPATCH: no existing key may be modified. This script
therefore
  1. snapshots every pre-existing key's EXACT serialised bytes BEFORE the write,
  2. writes the file with the single new key appended LAST,
  3. re-reads from disk and byte-compares every pre-existing key against its
     snapshot, and re-checks that the key ORDER of the old keys is preserved and
     the count is exactly old+1,
  4. restores the original file and raises if ANY of that fails.

Byte-comparison of `json.dumps(value, sort_keys=True)` per key is used rather
than `==` on the parsed objects, because `==` cannot see a float that
round-tripped to a different repr -- and every A04 verdict number lives in a
float.

CPU only. Idempotent-refusing: if the new key already exists, it aborts rather
than overwriting, so a re-run cannot silently rewrite history.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

NEW_KEY = "control_arms_ni_20260813"


def snapshot(d):
    return {k: json.dumps(v, sort_keys=True, default=float)
            for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--gpu_h", type=float, required=True)
    args = ap.parse_args()

    ev = json.load(open(args.evidence))
    import hashlib
    ev_sha = hashlib.sha256(open(args.evidence, "rb").read()).hexdigest()
    ev_md5 = hashlib.md5(open(args.evidence, "rb").read()).hexdigest()
    with open(args.status) as fh:
        before_text = fh.read()
    d = json.loads(before_text)
    n_before = len(d)
    old_keys = list(d.keys())
    snap_before = snapshot(d)

    if NEW_KEY in d:
        raise SystemExit(
            f"FATAL: key {NEW_KEY} already present. Refusing to overwrite an "
            "existing record.")

    split = ev["per_convention"]["split"]
    pe = ev["prereg_evaluation"]["split"]
    q2 = ev["Q2_floor_anchor"]
    prem = q2["inheritance_premium"]["per_axis"]
    dec = ev["P1_degeneracy_diagnostic"]["format_vs_content_decomposition"][
        "per_axis"]
    gd = ev["P1_degeneracy_diagnostic"][
        "gain_decomposition_FF_over_trainall"]["per_axis"]

    def m(arm, axis):
        for c in split["cells"]:
            if c["arm"] == arm and c["axis"] == axis:
                return c
        return None

    axes3 = ["triviaqa", "popqa", "mmlu_content"]
    arms3 = ["keep14fresh2_step200k", "freezefront_step200k",
             "fromscratch_step200k"]

    rec = {
        "date": "2026-08-13",
        "gate": ev["gate"],
        "verdict": ("P1_VIOLATED_ON_POPQA_BUT_THE_VIOLATION_IS_FORMAT_NOT_"
                    "KNOWLEDGE__ZERO_INHERITANCE_FLOOR_REACHES_32_TO_40_PCT_"
                    "OF_THE_INTACT_RESIDUAL"),
        "gpu_h_spent": args.gpu_h,
        "gpu_h_note": (
            "ZERO. Every input is a per-example shard set already on zwfy6, "
            "written 2026-08-02. No model loaded, no CUDA context, no scoring. "
            "Analysis CPU-only, read-only on all inputs. The driver's "
            "refuse-guard (>8000 MiB held, or .104/.21 by IP) was exercised and "
            "passed on .73 and .82 at 8x0 MiB."),
        "prereg": {"document": "A04_CONTROL_ARMS_NI_PREREG.md",
                   "commit": "e51f390",
                   "committed_before_first_margin": True},
        "evidence": "evidence/a04_control_arms_ni.json",
        "evidence_sha256": ev_sha,
        "evidence_md5": ev_md5,
        "evidence_identical_on_both_disks": True,
        "key_count_note": (
            "this key is the 44th. The dispatch said 41 and MAIN's own check "
            "said 42; the file was at 43 when this key was appended, because a "
            "CONCURRENT A04 pass (commit 7e54376, "
            "keep12_trajectory_monotonicity_20260813) landed its own key while "
            "this analysis was running. The writer does not trust any hardcoded "
            "count: it snapshots every pre-existing key's serialised bytes, "
            "asserts count == old+1, asserts the old key ORDER is unchanged, "
            "asserts each old key is byte-identical, AND asserts the whole new "
            "file is a byte-PREFIX extension of the old one -- then restores "
            "the original if any check fails."),
        "writer_defect_found_and_fixed": (
            "the first run of a04_status_add_control_arms_key.py used indent=1 "
            "against a file that is indent=2. The per-key byte check PASSED "
            "(correctly -- no key's VALUE changed) but `git diff` showed 2643 "
            "deletions / 2965 insertions: it had rewritten EVERY LINE of an "
            "append-only record. Reverted from backup and the writer now DERIVES "
            "the file's own json.dumps format by round-tripping the original "
            "bytes, and adds the whole-file byte-prefix assertion that catches "
            "a reformat the per-key check structurally cannot. Final diff: 322 "
            "insertions, 0 deletions."),
        "verdict_doc": "A04_CONTROL_ARMS_NI_VERDICT.md",
        "code": ["code/a04_control_arms_ni.py",
                 "code/a04_control_arms_ni_driver.sh",
                 "code/a04_status_add_control_arms_key.py"],
        "node_of_record": {
            "node": ".73 (8xH20, zwfy6)",
            "numpy": ev["environment"]["numpy"],
            "cross_node_reproduced_on": ".82 (numpy 2.4.6)",
            "cross_node_result": (
                "archived-endpoint reproduction 0.00e+00 pp on BOTH nodes; "
                "P1/P2/P3 and the popqa margin difference (+1.7803 pp, "
                "+3.79 SE) identical to 4 dp. The 19/10000-row multinomial "
                "drift between 2.4.6 and 2.5.1 does not reach any verdict "
                "here: margins are 19.3-94.2 bootstrap SE from flipping."),
            "not_touched": ["LOCAL", ".21 (SparseForge #246)",
                            ".104 (paperC Qwen3 heal)"],
        },
        "what_was_tested": (
            "the two REPAIR-MODE controls at fixed damage keep_front=14 "
            "n_fresh=2 that had NEVER entered any A04 evidence file: "
            "olmo2_probe2_7B_keep14fresh2_freezefront (--freeze_front; front 14 "
            "inherited layers FROZEN, 1226.9M of 4060.4M trainable) and "
            "..._fromscratch (--from_scratch; base ignored, all 16 layers "
            "random-init). Rationale: no damaged arm has EVER accepted, and "
            "shallower rungs (keep16/20/24/28+fresh2) have 0 checkpoints on "
            "either disk, so the only training-free way to widen the rung set "
            "is to vary something other than depth."),
        "arms_are_matched": {
            "dataset_rows": 7570911, "eff_bs": 128, "seq_len": 2048,
            "max_steps": 200000, "steps_reached": 200000,
            "disk_of_training": "wzc1", "optimizer": "fp32 AdamW",
            "depth": 16, "keep_front_layers": 14, "n_fresh_layers": 2,
            "note": (
                "STATUS.json:warning's two-corpora confound is a DEPTH-LADDER "
                "confound and does NOT apply: all three arms are ONE depth on "
                "ONE corpus with ONE step count. Registered in the prereg "
                "before scoring so it is not a post-hoc claim."),
            "NOT_matched": {
                "effective_lr": {"train-all": "2e-5", "FF": "2e-5",
                                 "FS": "1e-4"},
                "why": ("_classify_param (trainer:436) returns 'fresh' FIRST "
                        "under --from_scratch, so FS's whole parameter set "
                        "landed in the fresh group at lr_fresh=1e-4. P1 "
                        "(train-all vs FF) IS LR-matched and clean; P2 and Q2 "
                        "are LR-confounded in BOTH directions."),
                "n_trainable": {"train-all": 4060352512, "FF": 1226870784,
                                "FS": 4060352512},
                "n_resume_banners": {"train-all": 4, "FF": 1, "FS": 0},
            },
        },
        "P1": {
            "prediction": ("margin_pp(FF) <= margin_pp(train-all) on EVERY "
                           "decision axis"),
            "verdict": pe["P1"]["verdict"],
            "identical_under_all_five_tie_conventions": True,
            "n_satisfied_of_n_axes": [pe["P1"]["n_satisfied"],
                                      pe["P1"]["n_axes"]],
            "violations_beyond_pooled_se": pe["P1"]["violations_beyond_se"],
            "per_axis_FF_minus_trainall_pp": {
                a: pe["P1"]["per_axis"][a]["FF_minus_trainall_pp"]
                for a in axes3},
            "per_axis_diff_over_pooled_se": {
                a: pe["P1"]["per_axis"][a]["diff_over_pooled_se"]
                for a in axes3},
            "confirmatory_paired_item_bootstrap": {
                a: {"FF_minus_trainall_pp": ev["P1_paired_confirmatory"][
                        "per_axis"][a]["FF_minus_trainall_pp"],
                    "ci95_pp": ev["P1_paired_confirmatory"]["per_axis"][a][
                        "ci95_pp"],
                    "boot_p": ev["P1_paired_confirmatory"]["per_axis"][a][
                        "boot_p"],
                    "resolved": ev["P1_paired_confirmatory"]["per_axis"][a][
                        "resolved"]}
                for a in axes3},
        },
        "P2": {
            "prediction": ("FS is the lowest-margin arm on >=2/3 decision axes "
                           "AND rejects 3/3"),
            "verdict": pe["P2"]["verdict"],
            "n_axes_where_FS_lowest": pe["P2"]["n_axes_where_FS_lowest"],
            "n_axes": pe["P2"]["n_axes"],
            "FS_rejects_all": pe["P2"]["FS_rejects_all"],
            "identical_under_all_five_tie_conventions": True,
        },
        "P3": {
            "trigger": "P1 violated beyond pooled bootstrap SE",
            "fires": pe["P3"]["fires"],
            "axes_firing": pe["P3"]["axes_firing"],
            "clause_1_the_arm_ordering_is_wrong": {
                "survives": True,
                "statement": (
                    "margin_pp(FF) > margin_pp(train-all) on popqa, resolved. "
                    "This is a fact about the statistic the gate ACTUALLY "
                    "decides on, so A04_GATE_DESIGN 3.2's presumed arm ordering "
                    "A1>A2>A3>A4 is an UNTESTED ASSUMPTION and a rung is "
                    "(depth, repair mode), not depth."),
            },
            "clause_2_freezing_recovers_more_knowledge": {
                "survives": False,
                "why": (
                    "the +1.7663 pp popqa EM move partitions EXACTLY (asserted "
                    "to <1e-9 pp) into FORMAT +1.5350 pp [+1.2546,+1.8296] and "
                    "CONTENT +0.2313 pp [-0.1121,+0.5818] p=0.1972 -- the "
                    "content component is NOT resolved. 86.9% of the move is "
                    "format. Of FF's 674 popqa EM gains, 337 (exactly 50.00%) "
                    "are items train-all's own prediction ALREADY CONTAINED. "
                    "FF's mean popqa prediction is 26.42 chars vs train-all's "
                    "49.72 (0.531x), its `contains` is 1.3247 pp LOWER, its "
                    "top-constant share 6.932% vs 1.213%, distinct predictions "
                    "9190 -> 5073. FF did not learn more facts; it stopped "
                    "padding."),
            },
            "prereg_clause_WITHDRAWN": (
                "the prereg's speculative positive route -- 'the route to an "
                "accept may be training the inherited weights LESS' -- is NOT "
                "supported and is withdrawn. It required P1 to be violated for "
                "a KNOWLEDGE reason. Recorded explicitly because the prereg "
                "committed to the inference in advance."),
            "deeper_reading": (
                "A04 now has TWO independent demonstrations that a generative-EM "
                "decision axis in a base-LM regime partly measures output "
                "format: PROPOSAL.md 4.4 (full32, 47.37% of an EM LOSS was "
                "verbosity) and this pass (50.00% of an EM GAIN, and it "
                "REORDERS TWO ARMS). The second is strictly worse for the rule: "
                "4.4's confound moved a MAGNITUDE, this one moves an ORDERING, "
                "and an ordering is what a certification rule is FOR."),
        },
        "Q1_margins_split": {
            "zero_accepts": True,
            "scope_of_zero_accepts": ("3 arms x 4 axes x 5 tie conventions; "
                                      "0 of 3 decision axes accept on any arm"),
            "se_to_flip_range": [
                min(m(a, x)["se_to_flip_NI"] for a in arms3
                    for x in ["triviaqa", "popqa", "mmlu_content", "nq_open"]),
                max(m(a, x)["se_to_flip_NI"] for a in arms3
                    for x in ["triviaqa", "popqa", "mmlu_content", "nq_open"])],
            "margin_pp": {a: {x: m(a, x)["margin_pp"]
                              for x in ["triviaqa", "popqa", "mmlu_content",
                                        "nq_open"]} for a in arms3},
            "recovered_fraction": {
                a: {x: m(a, x)["residual_fraction_recovered"]
                    for x in ["triviaqa", "popqa", "mmlu_content", "nq_open"]}
                for a in arms3},
            "credit_convention_note": (
                "under `credit`, mmlu_content retires on "
                "D6_delta_finer_than_instrument for all three arms (decision "
                "family 9 -> 6). Verdicts unchanged; retired cells are NOT "
                "reported as 'NI rejected'."),
            "neighbours": (
                "NONE EXIST -- one scored checkpoint per arm. "
                "A04_GATE_DESIGN 2.0.2 explicitly permits 'or a statement that "
                "none exist', and since no cell ACCEPTS the precondition has "
                "nothing to protect."),
        },
        "Q2_zero_inheritance_floor": {
            "finding": ("FS is resolvedly ABOVE its own best-constant null on "
                        "all 4 axes (p=0.0001 each), so a zero-inheritance "
                        "model of the same depth/corpus/budget already reaches "
                        "a large share of the intact calibrated residual"),
            "FS_residual_pp": {a: q2["per_axis"][a]["FS_residual_pp"]
                               for a in q2["per_axis"]},
            "FS_residual_as_fraction_of_intact": {
                a: q2["per_axis"][a]["FS_residual_as_fraction_of_intact"]
                for a in q2["per_axis"]},
            "inheritance_premium_fraction_of_intact": {
                a: {arm: v["inheritance_premium_fraction_of_intact"]
                    for arm, v in prem[a]["by_arm"].items()}
                for a in prem},
            "inheritance_premium_resolved": {
                a: {arm: v["premium_resolved"]
                    for arm, v in prem[a]["by_arm"].items()}
                for a in prem},
            "headline": (
                "mmlu_content: train-all's '53.06% recovered' is 40.47 points "
                "of ZERO-INHERITANCE FLOOR plus a +12.59-point premium; FF's "
                "premium on that axis is +0.34 pp, p=0.8852, NOT RESOLVED. On "
                "nq_open BOTH arms are at or below the floor and FF is "
                "resolvedly BELOW it. Any recovery FRACTION quoted without its "
                "floor credits inheritance with work random init already does."),
            "caveat": (
                "FS ran at 5x the LR of the other two arms, so the floor and "
                "every premium are LR-confounded in BOTH directions. The clean "
                "control is gate-design arm A3 (--random_trunk: random trunk, "
                "transplanted embed/norm/lm_head, same LR), which does NOT "
                "exist on either disk."),
        },
        "Q3_ordering_vs_rules": {
            "descending_by_margin": {
                a: ev["Q3_ordering_and_rules"]["ordering_per_axis"][a][
                    "descending_by_margin"] for a in axes3},
            "ratio_mean_by_arm": {
                a: split["ratio_rule"][a]["mean_ratio"] for a in arms3},
            "rho": split["ratio_rule"][arms3[0]]["rho"],
            "rule_disagreement_cells": 0,
            "answer": (
                "ZERO new support for the 'current rules accept models this "
                "rule rejects' half. All three arms REJECT under BOTH NI and "
                "RATIO(0.85) (means 0.4728/0.4478/0.3996 vs rho=0.85, i.e. "
                "0.38-0.45 BELOW threshold -- nothing marginal). After this "
                "pass the disagreement evidence is STILL exactly 1 of 5 "
                "checkpoints of 1 zero-damage arm with RATIO's margin over rho "
                "= +0.0015."),
            "PLATEAU_not_computable": (
                "olmo2_ppl_results/ has NO freezefront or fromscratch dir (only "
                "7B_keep14_step{0,128000,153500,200000}(_v2) plus an unrelated "
                "7B_scratch16L_lr2e5_* LR-control run -- a DIFFERENT run, on "
                "the OTHER corpus, at uniform 2e-5). Registered as a design "
                "limitation in the prereg BEFORE scoring, not found after."),
        },
        "step23500_dropped_not_demoted": {
            "path": ("outputs/olmo2_probe2_7B_keep14fresh2_freezefront/"
                     "step23500.pt"),
            "finding": (
                "the zwfy6 copy of logs/olmo2_7B_keep14fresh2_freezefront.log "
                "(162,067 B) is a DIFFERENT, ABANDONED run from the wzc1 copy "
                "(1,368,257 B): first banner 2026-07-21 02:02:20 vs 2026-07-25 "
                "12:15:48, bs=4 gaccum=4 vs bs=16 gaccum=1, dataset "
                "rows=15,491,607 vs 7,570,911, died at step 23,640 vs reached "
                "200,000+final.pt. step23500.pt's mtime "
                "(2026-07-23 13:45:20.774755372) matches the ABANDONED run's "
                "save line to the nanosecond. The wzc1 run's own step23500 "
                "(07-25 22:40:21) was ROTATED AWAY."),
            "consequence": (
                "step23500 and step200000 are checkpoints of TWO DIFFERENT RUNS "
                "on TWO DIFFERENT CORPORA at TWO DIFFERENT micro-batch "
                "geometries. DROPPED, not demoted to 'far neighbour' -- a gap "
                "statement would imply one trajectory. Bootstrap offset 802 "
                "reserved and left unused."),
            "generalisable_warning": (
                "the two disks' same-named logs/*.log files are NOT copies. Any "
                "provenance claim about these arms must state WHICH DISK, and "
                "wzc1 is authoritative for all three."),
        },
        "wzc1_vs_zwfy6_checkpoints": (
            "zwfy6 step200000.pt are SLIM MODEL-ONLY eval copies (~16.24 GB = "
            "4,060,352,512 params x 4 B fp32 + zip overhead) staged from LOCAL; "
            "the launchers hard-assert those byte counts. FF's wzc1 file is "
            "26.06 GB vs 48.72 GB for the other two ONLY because its AdamW "
            "state covers 1226.9M trainable params -- NOT a different "
            "architecture. All three zwfy6 copies load 179 tensors, strict, "
            "num_hidden_layers=16 per the eval logs."),
        "sampler_regime": {
            "all_three": "pre_ce5c298",
            "consequence": (
                "all three predate the DistributedSampler seed fix "
                "(2026-08-09 23:21:09), so --seed moved only the fresh-tail "
                "init. PROPOSAL.md 7.2's binding no-pooling rule is satisfied "
                "trivially (same side of the break, mutually comparable), and "
                "none of them may enter any sigma_run."),
        },
        "verification": {
            "archived_endpoint_reproduction_dev_pp": {
                a: ev["archived_endpoint_reproduction"]["per_axis"][a][
                    "abs_dev_pp"] for a in axes3},
            "archived_values_read_at_runtime_not_hardcoded": True,
            "my_own_error_caught_by_this_gate": (
                "the FIRST version of a04_control_arms_ni.py hardcoded three "
                "reference constants transcribed from "
                "A04_KEEP14_TRAJECTORY_NI_VERDICT 2's 4-dp table (-28.4624) "
                "with invented trailing digits. The gate FIRED at 8.82e-05 pp: "
                "the recomputation was right and MY CONSTANTS WERE WRONG "
                "(canonical -28.462438698172093). FIFTH hand-transcription slip "
                "of 2026-08-13. Fix was NOT to loosen the tolerance -- the "
                "script now READS THE CANONICAL JSON AT RUNTIME, removing the "
                "transcription step entirely."),
            "shard_integrity": (
                "all 16 arm x axis cells: shard index set EXACTLY {0..7} (not "
                "'8 files'), merged n exactly EXPECTED_N (17944/14267/3610/"
                "14042), 0 duplicate item_id, 0 nan, item_id sequences "
                "IDENTICAL across all four arms (assert_aligned) -- without "
                "which the paired differences would compare different items."),
            "protocol": (
                "cb_bs=32 from the launchers' echoed 'START <model> ... bs=32' "
                "lines (cb_driver_104.out for FF+FS, cb_driver_73.out for the "
                "endpoint AND the anchor, nqopen_driver_104.log, "
                "nqopen_scratch.log). mmlu_bs=16 from launcher SOURCE: "
                "p06_run_104_transferred.sh and p06_run_transferred.sh both "
                "leave BS unset (asserted) -> _run_olmo2_mmlu_content.sh:43 "
                "BS=\"${BS:-16}\". add_bos asserted with `is False`, NEVER "
                "`is not True`. max_new_tokens==32 asserted. chat_template "
                "STRUCTURAL: neither harness has a chat-template code path."),
            "dedicated_asserter_not_the_imported_one": (
                "a04_neighbour_variability.protocol_asserted requires a "
                "'DRIVER START ... mmlu_bs=.. cb_bs=..' header only the "
                "2026-08-13 drivers emit; these cells were scored 2026-08-02 by "
                "different launchers. Reusing it would crash, and loosening the "
                "regex would weaken the gate for every future caller. The "
                "frozen expectation {cb_bs:32, mmlu_bs:16} is IDENTICAL."),
            "harness_md5": (
                "2ed41993241226c795a3ca38375933f7 (closedbook) / "
                "fe4a62dbdf884a1e2aedc6ed26887b4e (mmlu_content) -- IDENTICAL "
                "to the values A04_KEEP14_TRAJECTORY_NI_VERDICT 5.1 item 5 pins "
                "for the copies that produced the anchor and the endpoint."),
            "decomposition_partition_asserted": (
                "FORMAT + CONTENT must equal the observed EM move to <1e-9 pp "
                "or the script refuses to publish. `contains` is used ONLY to "
                "LABEL an item, never substituted for EM as the decision "
                "metric (GATE_DESIGN 4.1)."),
            "bootstrap_offsets": (
                "new arms 800/801, checked mechanically against every archived "
                "block (0-1, 100-102, 200-203, 300-301, 400-408, 500-503, "
                "700-702); train-all keeps ARCHIVED 201 so the reproduction "
                "assert is meaningful; 802 reserved and LEFT UNUSED."),
        },
        "must_not_claim_additions": [
            ("26. Training all layers repairs a given injury better than "
             "freezing the inherited front -- FALSE on popqa under the gate's "
             "own decision metric (+1.7803 pp, 3.79 SE; paired +1.7663 pp, "
             "p=0.0001) at matched depth, corpus, budget AND LR. "
             "A04_GATE_DESIGN 3.2's arm ordering A1>A2>A3>A4 is UNTESTED."),
            ("27. Freezing the inherited trunk recovers more KNOWLEDGE -- the "
             "converse is equally forbidden. popqa's CONTENT component is "
             "+0.2313 pp, CI [-0.1121,+0.5818], p=0.1972, unresolved; 86.9% of "
             "the move is format and 50.00% of FF's EM gains are items "
             "train-all already contained."),
            ("28. Quoting any recovery FRACTION without its zero-inheritance "
             "floor. FS reaches 32.55/11.58/40.47/28.89% of the intact "
             "residual."),
            ("29. Calling freezefront/step23500.pt a checkpoint of the run that "
             "produced step200000.pt -- it belongs to an abandoned run on the "
             "OTHER corpus at a different micro-batch geometry."),
            ("30. Any clean 'inheritance is worth X' claim from FS -- it ran at "
             "1e-4 vs 2e-5. The clean control is arm A3 --random_trunk, which "
             "does not exist on either disk."),
        ],
        "recommendation": {
            "pilot_two": "DO NOT APPROVE",
            "reason_1_unchanged": (
                "no damaged arm has EVER accepted; shallower rungs do not "
                "exist; this pass adds two more constant-REJECT arms. Pilot Two "
                "(1,077-4,309 GPU-h) would price a gate never observed to "
                "accept under damage."),
            "reason_2_new": (
                "the gate's decision metric can be REORDERED BY OUTPUT LENGTH. "
                "Funding 8 more runs to feed a metric with that property buys 8 "
                "more cells of the same defect. The metric problem is a DESIGN "
                "fix, not an n fix, and it is now the binding constraint -- as "
                "PROPOSAL.md 8 item 3 said, but with an ORDERING FLIP rather "
                "than a magnitude as evidence."),
            "two_cheap_next_steps_CANDIDATES_not_authorised": [
                ("arm A3 (--random_trunk) at matched LR 2e-5 -- the only thing "
                 "that turns Q2's floor into a clean inheritance measurement. "
                 "One 7B run; the flag exists at trainer line 586, mutually "
                 "exclusive with --from_scratch."),
                ("a format-insensitive decision axis with a well-defined null, "
                 "or a pre-registered verbosity diagnostic as a REPORTING "
                 "REQUIREMENT on every generative cell. 0 GPU for the design; "
                 "cheapest way to remove the defect that just flipped an "
                 "ordering."),
            ],
        },
        "not_licensed": ev["not_licensed"] + [
            "'freezing the trunk repairs better' (P3 clause 2 does not survive)",
            ("reading FF's popqa result as a reason to pivot A04 to freeze-based "
             "repair -- that was the prereg's speculative clause and it is "
             "WITHDRAWN"),
        ],
    }

    d[NEW_KEY] = rec

    tmp = args.status + ".tmp_add_control_arms"
    bak = args.status + ".bak_add_control_arms"
    shutil.copy2(args.status, bak)

    # WRITE IN THE FILE'S OWN FORMAT, verified by round-trip against the
    # ORIGINAL bytes before the new key is added.
    #
    # Why this matters: the first run of this script used `indent=1` (copied from
    # the evidence writers) against a file that is `indent=2`. The key-level byte
    # comparison below PASSED -- correctly, since no key's VALUE changed -- but
    # `git diff` showed 2643 deletions / 2965 insertions, i.e. it had rewritten
    # every line in the file. A whole-file reformat of an append-only record is
    # exactly the kind of silent history damage the dispatch's "do not modify any
    # existing key" rule exists to prevent, even though no key was semantically
    # touched. So the format is now DERIVED from the file and ASSERTED.
    fmt = None
    for ind in (2, 1, 3, 4):
        for sep in (None, (",", ": ")):
            for ea in (True, False):
                for trail in ("", "\n"):
                    cand = json.dumps(json.loads(before_text), indent=ind,
                                      default=float, separators=sep,
                                      ensure_ascii=ea) + trail
                    if cand == before_text:
                        fmt = {"indent": ind, "separators": sep,
                               "ensure_ascii": ea, "trailing": trail}
                        break
                if fmt:
                    break
            if fmt:
                break
        if fmt:
            break
    if fmt is None:
        os.remove(bak)
        raise SystemExit(
            "FATAL: could not reproduce STATUS.json's own byte formatting, so a "
            "write would silently reformat the whole file. Refusing.")
    print(f"  detected format: {fmt}")

    with open(tmp, "w") as fh:
        fh.write(json.dumps(d, indent=fmt["indent"], default=float,
                            separators=fmt["separators"],
                            ensure_ascii=fmt["ensure_ascii"])
                 + fmt["trailing"])
    os.replace(tmp, args.status)

    # ---- verify: count, order, and per-key BYTE identity -----------------
    after_text = open(args.status).read()
    after = json.loads(after_text)
    problems = []
    if len(after) != n_before + 1:
        problems.append(f"key count {len(after)} != {n_before}+1")
    if list(after.keys())[:n_before] != old_keys:
        problems.append("pre-existing key ORDER changed")
    if list(after.keys())[-1] != NEW_KEY:
        problems.append(f"new key is not last: {list(after.keys())[-1]}")
    snap_after = snapshot(after)
    for k in old_keys:
        if snap_after.get(k) != snap_before[k]:
            problems.append(f"pre-existing key MODIFIED: {k}")

    # WHOLE-FILE append-only check: every byte of the original, up to and
    # including the last pre-existing key, must be a literal PREFIX of the new
    # file. This is what catches a reformat that the per-key check cannot.
    prefix = before_text.rstrip()
    if prefix.endswith("}"):
        prefix = prefix[:-1].rstrip()      # drop the closing brace + whitespace
    if not after_text.startswith(prefix):
        problems.append(
            "the new file is NOT a byte-prefix extension of the old one -- the "
            "write reformatted or reordered existing content")
    if problems:
        shutil.copy2(bak, args.status)
        raise SystemExit("FATAL: refusing this write, original restored.\n  "
                         + "\n  ".join(problems))

    os.remove(bak)
    print(f"OK: {args.status}")
    print(f"  keys {n_before} -> {len(after)} (+1, appended LAST)")
    print(f"  all {n_before} pre-existing keys byte-identical "
          f"(json.dumps sort_keys=True per key)")
    print(f"  pre-existing key order preserved")
    print(f"  whole file is a byte-PREFIX extension of the original "
          f"(no reformat)")
    print(f"  new key: {NEW_KEY}")
    print(f"  gpu_h_spent: {rec['gpu_h_spent']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
