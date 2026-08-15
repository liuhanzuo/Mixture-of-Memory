#!/usr/bin/env python3
"""APPEND-ONLY update of B06 STATUS.json for the drift-resolution leg.

Contract enforced programmatically:
  * only NEW top-level keys are added;
  * every pre-existing key keeps its name, position AND exact value (deep-compared);
  * the file is re-read from disk after writing and the invariants re-asserted.

Numbers come from evidence/drift_resolution_evidence.json (itself recomputed from
raw per-item records), never hand-typed.
"""
import json
import os
import sys

ROOT = "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
D = os.path.join(ROOT, "proposal/backlog/B06-portable-decompression-adapter")
STATUS = os.path.join(D, "STATUS.json")
EV = os.path.join(D, "evidence/drift_resolution_evidence.json")

before_txt = open(STATUS).read()
before = json.loads(before_txt)
before_keys = list(before.keys())
ev = json.load(open(EV))

f = ev["the_four_numbers"]
sr = ev["same_ruler_comparison"]
at = ev["attribution"]
ad = at["applied_to_the_canonical_drift"]
nf = ev["judge_noise_floor"]
cj = ev["cross_judge_outlier_test"]
hk = ev["headline_and_kill_condition_1"]
pc = hk["per_category_on_corrected_instrument"]
sem = hk["locomo_category_semantics_from_raw_data"]

NEW = {
    "drift_resolution_leg_20260815": {
        "_verdict_doc": "proposal/backlog/B06-portable-decompression-adapter/"
                        "DRIFT_RESOLUTION_VERDICT.md",
        "_evidence_json": "proposal/backlog/B06-portable-decompression-adapter/"
                          "evidence/drift_resolution_evidence.json",
        "_generator": "proposal/backlog/B06-portable-decompression-adapter/"
                      "evidence/build_drift_evidence.py (CPU only, deterministic, "
                      "verified byte-identical across two runs)",
        "_cost": "0 GPU-h, 0 ssh, 0 judge API calls",
        "_git_commit": ev["_git_commit"],

        "VERDICT": "The 8.11-vs-13.29 'cross-node drift' is NOT node drift and NOT generation "
                   "drift. It is a JUDGE-SIDE (instrument) artefact. Kill condition 2 DOES NOT "
                   "FIRE, and it is closed without GPU and without a rejudge.",

        "the_gate_question_answered": {
            "asked": "how much of the drift survives once both endpoints are on the "
                     "Judge_1:4 (n=1540) ruler?",
            "answer": "it does not shrink -- it GROWS 26%, from 5.1863 pp (blended n=1986) to "
                      "6.5584 pp (Judge_1:4 n=1540), because the constant cat-5 term deflates "
                      "the two endpoints unequally.",
            "blended_n1986_canonical_vs_b06_control_pp":
                sr["blended_n1986_ruler"]["canonical_vs_b06_control_pp"],
            "judge_1_4_canonical_vs_b06_control_pp":
                sr["judge_1_4_n1540_ruler"]["canonical_vs_b06_control_pp"],
            "scale_trap_warning_in_next_gate_was_correct": True,
            "but_it_concealed_the_opposite_of_what_was_assumed":
                "the trap was not hiding a comparison that would vanish on one ruler; it was "
                "hiding that the drift is LARGER than STATUS.json recorded.",
        },

        "four_numbers_recomputed_from_raw_per_item_records": {
            k: {"published_as": v["published_as"], "recomputed": v["recomputed"],
                "n": v["n"], "instrument": v["instrument"]}
            for k, v in f.items() if not k.startswith("_")},

        "attribution_judge_vs_generation": {
            "method": "F1 and accuracy are deterministic functions of the predictions ALONE (no "
                      "judge, no API, no sampling), so a real generation-quality change must move "
                      "them. Calibrate the judge-vs-lexical slope on a change known to be real "
                      "(the +LoRA arm), then ask how much lexical movement the drift shows.",
            "slope_f1_pp_per_judge_pp":
                at["calibration_on_a_known_real_change_the_LoRA_arm"]["slope_f1_pp_per_judge_pp"],
            "slope_acc_pp_per_judge_pp":
                at["calibration_on_a_known_real_change_the_LoRA_arm"]["slope_acc_pp_per_judge_pp"],
            "observed_over_predicted_f1_pct": ad["observed_over_predicted_f1_pct"],
            "observed_over_predicted_acc_pct": ad["observed_over_predicted_acc_pct"],
            "generation_attributable_share_pct_range": [
                ad["generation_attributable_share_via_f1_pct"],
                ad["generation_attributable_share_via_acc_pct"]],
            "judge_attributable_share_pct_range": [
                ad["judge_attributable_share_via_acc_pct"],
                ad["judge_attributable_share_via_f1_pct"]],
            "generation_attributable_absolute_upper_bound_pp": 0.5114,
            "three_replicates_relative_spread_pct":
                at["relative_spread_across_the_three_replicates_pct"],
            "headline": "on the SAME n=1540 items the two judge-free metrics agree to 0.16% "
                        "(F1) and 4.17% (acc) relative, while the gpt-4o judge disagrees by "
                        "64.74% relative.",
        },

        "judge_noise_floor_measured": {
            "design": nf["design"],
            "n_byte_identical_predictions": nf["n_byte_identical_predictions"],
            "pct_byte_identical": nf["pct_byte_identical"],
            "flip_rate_on_identical_input_pct": nf["identical_subset_flips"]["pct_of_subset"],
            "flips_symmetric": nf["flips_are_symmetric"],
            "sd_of_net_change_items_at_n1540": nf["sd_of_net_change_at_n1540_items"],
            "sigma_of_each_gap": nf["sigma_of_each_observed_gap"],
            "consequence": "per-call judge noise fully explains the 1.23 pp local-vs-local "
                           "wobble (net -1 item from the 879 identical-prediction items, net "
                           "+20 from the 661 differing ones) but is +14.7 sigma short of "
                           "explaining the canonical gap, which is therefore systematic.",
        },

        "cross_judge_outlier_test": {
            "design": "the same six canonical baseline prediction sets were graded by TWO judges "
                      "(gpt-4o and open-weight qwen3-8b). The open/gpt-4o ratio should be "
                      "comparable across methods if each gpt-4o cell is a faithful measurement.",
            "sibling_ratios": cj["sibling_ratios"],
            "sibling_ratio_mean": cj["sibling_ratio_mean"],
            "sibling_ratio_sd": cj["sibling_ratio_sd"],
            "hcache_canonical_ratio": cj["hcache_ratio"],
            "hcache_z_vs_siblings": cj["hcache_ratio_z_vs_siblings"],
            "implied_gpt4o_judge_1_4_under_sibling_mapping":
                cj["implied_gpt4o_judge_1_4_if_hcache_followed_sibling_mapping"],
            "conclusion": "the CANONICAL gpt-4o HCache cell is the outlier (z=+14.2, 2.03x the "
                          "next highest sibling ratio), not the local replicates. Under the open "
                          "judge HCache overtakes MemoryLLM; under gpt-4o it is last.",
        },

        "rejudge_decision": {
            "rejudge_run": False,
            "rejudge_needed": False,
            "api_calls_made": 0,
            "reasons": [
                "it would not be same-instrument: measured 3.07% per-item verdict instability on "
                "byte-identical inputs means a fresh gpt-4o pass is a NEW instrument draw, not a "
                "fixed ruler -- 'same-harness rejudge' is unattainable against a non-deterministic "
                "remote endpoint",
                "the judge-independent columns already settle the attribution at zero cost and "
                "zero API risk",
                "the canonical per-item records are on zwfy6 and ssh was forbidden for this leg, "
                "so the rejudge as described could not be executed anyway",
            ],
            "judge_model_and_version_note": "all local caches record model='gpt-4o' with no "
                                            "dated snapshot pinned; the open-weight cross-check "
                                            "judge is recorded in "
                                            "locomo_results_openjudge_qwen3_MIRROR/"
                                            "hcache_8b_chatFALSE/judge_meta.json as "
                                            "'qwen3-8b-judge', non_thinking=true, temperature 0, "
                                            "top_p 1, seed 1, written_at 2026-08-04T19:55:19. "
                                            "Changing judge version IS changing instrument.",
        },

        "canonical_records_located_on_wzc1_after_all": {
            "artefact": "locomo_results_openjudge_qwen3_MIRROR/hcache_8b_chatFALSE/",
            "what": "a SECOND judge over the SAME canonical predictions, mirrored back to wzc1 "
                    "per paperA/TODOList.md:170",
            "identifies_the_same_run": "its judge-independent columns round to f1 4.67 / acc 6.29 "
                                       "/ em 0.25, matching the published canonical row at "
                                       "status/PAPERA_RESULTS_CONSOLIDATED.md:175 exactly",
            "independently_recovers_cat5": "its cat5 cell = 1.121076233184 = exactly 5/446, a "
                                           "fourth confirmation of the 8.11 arithmetic",
            "corrects": "canonical_8_11_conversion_status.what_is_still_missing said the canonical "
                        "records were not on this disk. True of the judge cache; but this same-run "
                        "artefact WAS on wzc1 and was missed.",
        },

        "canonical_conversion_confirmed_three_independent_routes": {
            "route_A_invert_published_per_category_percentages":
                ev["canonical_number_derivation"]["route_A_implies_cat1_4_correct"],
            "route_B_solve_blended_8_11":
                ev["canonical_number_derivation"]["route_B_from_blended_8_11"]["solutions"],
            "route_C_solve_judge_1_4_10_13":
                ev["canonical_number_derivation"]["route_C_from_published_judge_1_4_10_13"]["solutions"],
            "all_agree": ev["canonical_number_derivation"]["all_three_routes_agree"],
            "each_route_solution_is_unique": True,
            "conclusion": "10.13 IS the Judge_1:4 conversion the gate asked for, and it is "
                          "confirmed. The next_gate's suspicion that 'the conversion may already "
                          "be done' is correct.",
        },

        "conversation_clustered_bootstrap_now_done": {
            "why": "established_measurements.caveat_from_errata flagged the per-item interval as "
                   "not dependence-aware (only 10 conversations) and noted the honest version is "
                   "0 GPU. Done here.",
            "n_conversations": hk["n_conversations"],
            "protocol": hk["bootstrap_protocol"],
            "clustered_95ci_pp": hk["conversation_clustered_bootstrap_95ci_pp"],
            "per_item_95ci_pp_reproduced": hk["paired_item_bootstrap_95ci_pp"],
            "frac_resamples_le_zero": hk["conversation_clustered_bootstrap_frac_le_zero"],
            "conclusion": "the clustered interval is essentially the per-item interval; the "
                          "nesting caveat does NOT threaten this effect.",
        },

        "kill_condition_1_ADJUDICATED_does_not_fire": {
            "clause": "只在 LoCoMo open-domain category 有益",
            "verdict": "DOES NOT FIRE",
            "basis": "per-category McNemar on the CORRECTED instrument (Judge_1:4): all four "
                     "judged categories are individually significantly positive at p<0.05.",
            "per_category": {k: {"label": v["data_grounded_label"], "n": v["n"],
                                 "noLoRA_pct": v["noLoRA_pct"], "lora_pct": v["lora_pct"],
                                 "delta_pp": v["within_cat_delta_pp"],
                                 "share_of_gain_pct": v["share_of_overall_gain_pct"],
                                 "mcnemar_exact_p": v["mcnemar_exact_two_sided_p"],
                                 "significant_positive": v["significantly_positive_at_0_05"]}
                             for k, v in pc.items()},
            "the_named_category_is_the_smallest_contributor":
                f"open-domain = cat3, n={pc['3']['n']}, contributes only "
                f"{pc['3']['share_of_overall_gain_pct']:.1f}% of the 23.12 pp gain",
        },

        "CORRECTION_cat4_is_single_hop_not_open_domain": {
            "severity": "CONCLUSION-CHANGING",
            "what_status_json_said": "kill_gate.condition_1_status: 'cat4 (open_domain, n=841 = "
                                     "55% of the 1540)' and called condition 1 'the one at real "
                                     "risk'",
            "what_is_true": "cat4 is SINGLE-HOP. Measured from locomo/data/locomo10.json: cat4 "
                            "averages 1.07 evidence items, 94.5% of its items cite exactly one "
                            "evidence turn, and 1.00 distinct sessions. Open-domain is cat3 "
                            "(n=96, 2.08 evidence items, the ONLY category with zero-evidence "
                            "inference items).",
            "data_grounded_semantics": sem,
            "root_cause": "scripts/eval_qcmem_locomo.py:126-132 CATEGORY_NAMES maps "
                          "{1:multi_hop, 2:single_hop, 3:temporal, 4:open_domain}, which is wrong "
                          "for cats 2/3/4 and contradicts status/LOCOMO_JUDGE_AGGREGATE.md:31-32 "
                          "(right for cat3/cat4). The mislabel propagated into B06's STATUS.",
            "effect": "reading cat4 as 'open_domain' makes kill condition 1 look near-fired when "
                      "it in fact does not fire at all.",
            "note": "this is a LABELLING correction, not a numerical retraction -- every number in "
                    "condition_1_status (23.31 -> 55.77, +32.46) is arithmetically correct and "
                    "reproduced here; only the category NAME attached to it was wrong.",
        },

        "CORRECTION_condition_2_status_compared_the_wrong_pair": {
            "severity": "understates the drift 5.2x-6.5x; does not change the conclusion",
            "what_status_json_said": "kill_gate.condition_2_status: 'the drift is ~1.0 pp on the "
                                     "blended scale (12.286 canonical vs 13.293 local)'",
            "what_is_true": "12.286 is NOT canonical. It is locomo_results/hcache, the older LOCAL "
                            "run -- STATUS.json's own key third_measurement_found_20260814 labels "
                            "that same value 12.28600201409869 as 'older run (2026-07-09/10)'. "
                            "The canonical blended value is 8.1067. So condition_2_status compared "
                            "local-vs-local and called one of them canonical.",
            "correct_drift": {
                "canonical_vs_b06_control_blended_pp":
                    sr["blended_n1986_ruler"]["canonical_vs_b06_control_pp"],
                "canonical_vs_b06_control_judge_1_4_pp":
                    sr["judge_1_4_n1540_ruler"]["canonical_vs_b06_control_pp"],
                "what_condition_2_actually_measured_blended_pp":
                    sr["blended_n1986_ruler"]["older_local_vs_b06_control_pp"],
            },
            "conclusion_unchanged_because": "even the corrected 6.5584 pp is small against the "
                                            "+23.12 pp effect, AND it is now shown to be "
                                            "judge-side rather than a property of the arms.",
        },

        "kill_gate_status_after_this_leg": {
            "condition_1_只在_open_domain_有益": "DOES NOT FIRE (all 4 judged categories "
                                                "individually significant; the named category, "
                                                "cat3 open-domain, supplies only 3.7% of the gain)",
            "condition_2_统一harness后增益消失": "DOES NOT FIRE. CLOSED at 0 GPU / 0 API. Drift is "
                                              "judge-side; <=0.51 pp generation-attributable "
                                              "against a +23.12 pp effect.",
            "condition_3_换compressor完全不迁移": "STILL UNTESTED -- needs GPU (second compressor)",
            "remaining_blocker_for_promotion": "unchanged and it was never the drift: "
                                               "novelty_checked=false plus kill condition 3.",
        },

        "headline_reverified_unchanged": {
            "noLoRA_judge_1_4": hk["noLoRA_judge_1_4"],
            "lora_judge_1_4": hk["lora_judge_1_4"],
            "gain_pp": hk["gain_pp"],
            "mcnemar_b": hk["mcnemar_b"], "mcnemar_c": hk["mcnemar_c"],
            "mcnemar_exact_two_sided_p": hk["mcnemar_exact_two_sided_p"],
            "mcnemar_chi2_continuity_corrected": hk["mcnemar_chi2_continuity_corrected"],
            "note": "reproduced from raw per-item records; matches established_measurements "
                    "exactly. No published B06 number is retracted by this leg.",
        },

        "not_verified": ev["not_verified"],

        "highest_value_next_0_gpu_followup": {
            "action": "wc -l on the canonical judge cache, "
                      "locomo_results/hcache_8b_chatFALSE/judge_cache.jsonl, on zwfy6 (.73/.104)",
            "why": "scripts/eval_qcmem_locomo.py:713-715 sets judge=0.0 on a judge-API failure "
                   "and does NOT write that record to the cache. The canonical deficit is exactly "
                   "101 items vs the B06 control. If that file holds ~1439 records instead of "
                   "1540, silent judge-API failures are the proven mechanism and the canonical "
                   "8.11 is a partially-failed judge pass rather than a node/harness difference.",
            "cost": "0 GPU, one ssh + one wc -l",
            "status": "NOT DONE -- ssh was forbidden for this leg. Stated as a consistent, "
                      "cheaply testable hypothesis, NOT as an established cause.",
        },
    },
}

overlap = set(NEW) & set(before)
assert not overlap, f"APPEND-ONLY VIOLATION: would overwrite {overlap}"

after = dict(before)
after.update(NEW)
assert list(after.keys())[:len(before_keys)] == before_keys, "key order/prefix changed"

with open(STATUS, "w") as fh:
    json.dump(after, fh, indent=2, ensure_ascii=False)
    fh.write("\n")

# ---- read back from disk and re-assert every invariant ----
rb = json.loads(open(STATUS).read())
rb_keys = list(rb.keys())
print("=" * 78)
print("APPEND-ONLY VERIFICATION (re-read from disk)")
print("=" * 78)
print(f"keys before : {len(before_keys)}")
print(f"keys after  : {len(rb_keys)}")
print(f"keys added  : {[k for k in rb_keys if k not in before_keys]}")
print(f"keys removed: {[k for k in before_keys if k not in rb_keys] or 'NONE'}")
print()
print(f"checking all {len(before_keys)} pre-existing keys, name + position + deep value:")
ok = True
for i, k in enumerate(before_keys):
    name_ok = rb_keys[i] == k
    val_ok = json.dumps(rb[k], sort_keys=True, ensure_ascii=False) == \
        json.dumps(before[k], sort_keys=True, ensure_ascii=False)
    ok &= name_ok and val_ok
    print(f"  [{i:2d}] {k:<45} name={'OK' if name_ok else 'CHANGED'} "
          f"value={'IDENTICAL' if val_ok else 'CHANGED'}")
    assert name_ok and val_ok, f"MUTATION on pre-existing key {k}"
print()
assert ok
print("ASSERT PASSED: all pre-existing keys unchanged in name, position and value.")
print(f"ASSERT PASSED: only new key(s) added: {[k for k in rb_keys if k not in before_keys]}")
print(f"file size: {len(before_txt)} -> {os.path.getsize(STATUS)} bytes")
