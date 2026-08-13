#!/usr/bin/env python3
"""Append ONE key to A04's STATUS.json: `shallow_rung_ladder_20260813`.

APPEND-ONLY, ENFORCED AT THE TEXT LEVEL, NOT THE SEMANTIC LEVEL.

WHY TEXT-LEVEL. On 2026-08-13 a STATUS.json edit in this repo passed a full
semantic append-only check (every old key present, every old value equal) while
`json.dump` silently REFORMATTED THE WHOLE FILE: it used `indent=1` against an
`indent=2` file, and `git diff` showed 2,643 deletions / 2,965 insertions. The
per-key byte check PASSED and was CORRECT -- no key's VALUE changed -- but the
provenance of every untouched key had become unreviewable. A per-key check is
STRUCTURALLY INCAPABLE of catching a reformat.

So the guarantee enforced here is stronger and much simpler:

    THE NEW FILE IS A BYTE-PREFIX EXTENSION OF THE OLD FILE.

Concretely: the old file ends with the last key's closing `  }` followed by the
root's `}`. We keep every byte up to and including that inner `}`, then append
`",\\n" + <serialised new key> + "\\n}"`. **No existing byte is ever
re-serialised**, so `indent`, `ensure_ascii`, key order and float formatting of
the existing keys CANNOT change -- the question does not arise.

NO HARDCODED KEY COUNT. The `control_arms` pass was dispatched believing the file
had 41 keys, MAIN's own check said 42, and it was actually at 43 when the key
landed -- because a CONCURRENT A04 pass added one while the analysis ran. So this
writer reads the count at runtime and asserts only the RELATION `new == old + 1`.
The observed count is recorded in the appended key itself.

Verified after writing, with restore-from-backup on any failure:
  1. byte-prefix identity over all len(prefix) old bytes;
  2. the file still parses as JSON;
  3. key count == old + 1;
  4. every old key's value compares equal;
  5. the old key ORDER is unchanged and the new key is LAST;
  6. the new file's first len(old_bytes)-N bytes are literally old bytes.

CPU only. Refuses if the key already exists.

Usage: a04_status_add_shallow_ladder_key.py STATUS.json EVIDENCE.json SHA256
"""
from __future__ import annotations

import json
import os
import shutil
import sys

NEW_KEY = "shallow_rung_ladder_20260813"


def _f(x):
    """json.dumps default: coerce numpy scalars/bools that survived a round-trip."""
    try:
        return float(x)
    except Exception:
        return str(x)


def build_entry(evidence_path, evidence_sha256, old_n_keys):
    d = json.load(open(evidence_path))
    conv = d["mmlu_tie_convention"]
    cells = d["per_convention"][conv]["cells"]
    verd = d["per_arm_verdict"]
    new_arms = [a for a in verd if not a.endswith("_REF")]
    ladder = d["depth_ladder"]["rungs"]

    def per_axis(arm, field):
        return {ax: cells[arm][ax][field] for ax in cells[arm]}

    return {
        "date": "2026-08-13",
        "gate": "A04_shallow_rung_ladder_1B_keep13_keep14_fresh2",
        "verdict": d["headline"],
        "branch": d["BRANCH"],
        "branch_definitions_were_fixed_pre_data":
            d["branch_definitions_were_fixed_pre_data"],
        "gpu_h_spent": d["gpu_h"]["total_gpu_h"],
        "gpu_h_detail": d["gpu_h"],
        "gpu_h_note": (
            "training + eval only; this ANALYSIS is 0 GPU (CPU-only, read-only on every "
            "input: no model load, no CUDA context, no scoring). Training wall time is "
            "MEASURED from each arm's own trainer log as (t_last - t_first) / "
            "(step_last - step_first) over the WHOLE run -- elapsed/iter, NOT an "
            "instantaneous tqdm s/step sample (memory: "
            "one-sample-is-not-a-trend-or-state). Compare Pilot Two at 1,077-4,309 "
            "GPU-h."),
        "prereg": {
            "document": "A04_SHALLOW_RUNG_LADDER_PREREG.md",
            "commit": d["prereg"]["commit"],
            "committed_before_first_margin": True,
            "state_at_commit": d["prereg"]["state_at_commit"],
        },
        "verdict_doc": "A04_SHALLOW_RUNG_LADDER_VERDICT.md",
        "evidence": "evidence/a04_shallow_rung_ladder.json",
        "evidence_sha256": evidence_sha256,
        "code": [
            "code/a04_shallow_rung_ladder_ni.py",
            "code/a04_shallow_ladder_eval_driver.sh",
            "code/a04_shallow_ladder_chain.sh",
            "scripts/_run_a04_shallow_ladder.sh (trainer launcher)",
            "code/a04_status_add_shallow_ladder_key.py",
        ],
        "key_count_note": (
            f"this key is the {old_n_keys + 1}th; the file held {old_n_keys} when it "
            "was appended. NO count is hardcoded: the control_arms pass was dispatched "
            "believing 41, MAIN's check said 42, and the true value was 43 because a "
            "CONCURRENT A04 pass landed a key mid-analysis. This writer asserts only "
            "the RELATION new == old+1, plus byte-prefix identity of the whole old "
            "file -- which is the only check that catches a REFORMAT (an indent=1 "
            "write against this indent=2 file rewrote all 2,643 lines while every "
            "per-key byte check correctly passed)."),
        "what_was_trained": (
            "the TWO LIGHTEST DAMAGED 1B RUNGS THE FAMILY ADMITS, neither of which "
            "existed on either disk before today: keep14+fresh2 (16 layers = base "
            "depth, but base layers 14-15 DISCARDED and random-re-initialised; 2/16 = "
            "12.5% cut) on .73, and keep13+fresh2 (15 layers, 3/16 = 18.75% cut) on "
            ".82. Protocol is Pilot One Stage B VERBATIM -- every hyper-parameter read "
            "out of stageB_seed101/step5000.pt's own train_args dict, with ONLY "
            "keep_front_layers changed. Same seed (101), same corpus, same 5000 steps, "
            "same uniform LR 2e-5, same eff_bs 128, both post-ce5c298."),
        "why_this_was_the_binding_blocker": (
            "STATUS.json:pilot_one.pilot_two_status: 'BLOCKED. 1,077-4,309 GPU-h must "
            "not be committed until a NEW pre-data doc shows a rung exists where NI "
            "can be OBSERVED TO ACCEPT.' Same key: 'it is a rung-selection problem, "
            "not a variance problem.' NI's discrimination curve had an EMPTY gap: "
            "damaged arms cluster at 11-63% recovery and REJECT by tens of SE (1B "
            "keep12 by 27.0-90.4x sd_run at 22-32% recovery), the ONLY accept is "
            "full32_dolmino at ZERO structural damage, keep12 was the lightest damaged "
            "1B rung in existence, and shallower rungs had 0 checkpoints on either "
            "disk. No re-analysis could fill it."),
        "GATE0_no_degeneracy_at_keep_plus_fresh_equals_base_depth": {
            "why_checked": ("keep14+fresh2 gives depth 16 == the base's 16, so a "
                            "special branch in the trainer would have invalidated the "
                            "whole design. Checked BEFORE any 8-GPU commitment: 1 GPU, "
                            "20 steps, /tmp output, 2026-08-13 18:27-18:30."),
            "keep14": {"tensors_copied": 157, "expected_3_plus_11_times_keep": 157,
                       "fresh_layer_ids": [14, 15], "transplant_max_abs_diff": 0.0,
                       "fresh_post_attn_ln_all_ones": True,
                       "fresh_q_norm_all_ones": True, "fresh_q_proj_std": 0.020001403987407684,
                       "reached_step": 20, "exit_code": 0},
            "keep13": {"tensors_copied": 146, "expected_3_plus_11_times_keep": 146,
                       "fresh_layer_ids": [13, 14], "transplant_max_abs_diff": 0.0,
                       "fresh_post_attn_ln_all_ones": True,
                       "fresh_q_norm_all_ones": True, "fresh_q_proj_std": 0.019997162744402885,
                       "reached_step": 20, "exit_code": 0},
            "source_reading": (
                "transplant_front() (trainer:170) selects base keys by "
                "`lid < keep_front_layers` against the BASE state dict, and the "
                "expected fresh set is range(keep, keep+n_fresh) on the NEW cfg. There "
                "is NO branch for keep+fresh == base_layers; the only conditional is "
                "`if n_fresh_layers > 0`, which skips the fresh-init assert for the "
                "n_fresh=0 CPT control. Both arms have n_fresh=2."),
            "optimizer_groups_observed": (
                "keep14 fresh_decay 339.7M / inh_decay 1145.0M / inh_nodecay 0.1M; "
                "keep13 339.7M / 1077.9M / 0.1M -- ALL at 2.00e-05. UNIFORM LR, as in "
                "Stage B. No differential-LR claim is made or licensed."),
        },
        "keep14_is_DAMAGED_not_a_zero_damage_control": (
            "depth 16 == base depth, but n_fresh=2: base layers 14 and 15 are DROPPED "
            "and replaced by random-init Olmo2 layers, so 14 of 16 pretrained layers "
            "are inherited. The zero-damage control is n_fresh_layers=0 CPT "
            "(full32-style) and is a DIFFERENT construction. Reporting keep14+fresh2 "
            "as zero-damage would be a category error, and using a CPT arm as the "
            "ANCHOR is forbidden by guard G2."),
        "positive_preflight_assertions_printed_before_launch": (
            "both progress logs carry, BEFORE the launch line: 'PREFLIGHT-ASSERT "
            "trainer post-ce5c298: 869: sampler = DistributedSampler(ds, shuffle=True, "
            "seed=args.seed)', 'PREFLIGHT-ASSERT trainer md5: "
            "284b286f90b526e4e8ad93a68e2a3b16', 'PREFLIGHT-ASSERT base "
            "num_hidden_layers=16', the exact dolmino byte count 126,907,244,672 "
            "(wzc1's same-named file is a DIFFERENT corpus at 62,020,903,040 B), and "
            "'GPUs clear (0MiB held)'. Both arms are POST-fix, same side of the "
            "ce5c298 break as the Stage B family (PROPOSAL.md 7.2)."),
        "intact_anchor": {
            "chosen": d["intact_anchor"]["dirs"],
            "guards": "G0 (path-pinned) + G2 (VANILLA, executed)",
            "G2_executed_how": d["anchor_is_vanilla_G2"]["per_role"],
            "why_G2_matters": d["anchor_is_vanilla_G2"]["why"],
            "residual_intact_pp": d["intact_anchor"]["residual_intact_pp"],
            "delta_pp": d["intact_anchor"]["delta_pp"],
            "delta_cross_check_vs_canonical_max_abs_diff": max(
                v["abs_diff"] for v in
                d["intact_anchor"]["delta_cross_check_vs_canonical"].values()),
            "delta_never_substituted": True,
        },
        "keep12_reproduction_gate": {
            "why": d["keep12_reproduction_gate"]["why"],
            "max_abs_diff_pp": d["keep12_reproduction_gate"]["max_abs_diff_pp"],
            "tol_pp": d["keep12_reproduction_gate"]["tol_pp"],
            "passed": d["keep12_reproduction_gate"]["passed"],
            "reading": ("this pipeline reproduces keep12 seed101's OWN published "
                        "accuracy on all four axes, so the new rungs are measured on "
                        "the same instrument as the arm they are compared against."),
        },
        "NI_results": {
            arm: {
                "verdict": verd[arm]["verdict"],
                "n_decision_axes_accepting": verd[arm]["n_decision_axes_accepting"],
                "axes_accepting": verd[arm]["axes_accepting"],
                "identical_under_all_five_tie_conventions":
                    verd[arm]["identical_under_all_five_tie_conventions"],
                "margin_pp": verd[arm]["per_axis_margin_pp"],
                "bootstrap_se_pp": verd[arm]["per_axis_bootstrap_se_pp"],
                "se_to_flip": verd[arm]["per_axis_se_to_flip"],
                "recovered_fraction": verd[arm]["per_axis_recovered_fraction"],
            } for arm in verd
        },
        "depth_ladder": {
            "rungs": {k: {"cut_fraction": v["cut_fraction"], "depth": v["depth"],
                          "verdict": v["verdict"],
                          "recovered_fraction": v["recovered_fraction"]}
                      for k, v in ladder.items()},
            "monotonicity_DESCRIPTIVE_ONLY":
                d["depth_ladder"]["monotonicity_DESCRIPTIVE_ONLY"],
            "why_descriptive": d["depth_ladder"]["why_descriptive"],
            "comparability": d["depth_ladder"]["comparability"],
        },
        "adjacent_rung_paired_differences":
            d["adjacent_rung_paired_differences"],
        "ratio_rule": d["ratio_rule"],
        "verification": {
            "shard_integrity": (
                "every (arm x axis) cell: shard index SET exactly {0..7} (not '8 "
                "files'), merged n exactly EXPECTED_N (17944 triviaqa / 14267 popqa / "
                "3610 nq_open / 14042 mmlu), 0 duplicate item_id, 0 nan in the metric "
                "vector, and item_id sequences IDENTICAL across every arm AND the "
                "anchor (assert_aligned) -- without which the paired differences would "
                "compare different items."),
            "protocol": (
                "add_bos asserted `is False` (NEVER `is not True`, so None/missing "
                "FAILS); chat_template asserted `is not False` -> FAIL, plus "
                "STRUCTURALLY: neither eval script contains an apply_chat_template "
                "call site, so no flag can enable one. max_new_tokens == 32. "
                "mmlu_bs=16, cb_bs=32 (harness default) -- the Stage B driver's own "
                "values, so the new cells are protocol-identical to keep12."),
            "arm_architecture_verified_from_eval_meta": (
                "each arm's eval summary.json meta must report keep_front/n_fresh/"
                "num_hidden_layers matching its tag, or the analysis aborts -- an eval "
                "that rebuilt the WRONG shell would otherwise be scored silently."),
            "canonical_code_imported_not_reimplemented": (
                "ni_rule / ratio_rule / load_shards / build_nulls / "
                "mmlu_content_norm_vec / qa_metric_vec / EXPECTED_N / AXES / "
                "DEMOTED_AXES / PREREG from pilot_zero_rule_disagreement; "
                "assert_aligned / d4_interface_degenerate / D4_* / Z95_TWO_SIDED from "
                "a04_shallow_rung_ni_7b; paired_bootstrap / TIE_CONVS / N_BOOT / SEED "
                "from A03's analyze_1b_knowledge_floor. No metric, null, rule, guard "
                "or anchor re-derived. THE NULL IS NEVER HAND-COMPUTED (MAIN's own "
                "subtraction of a recorded null was ~0.5 pp off twice)."),
            "no_constant_hand_transcribed": (
                "keep12's canonical per-axis accuracy is READ AT RUNTIME from "
                "evidence/pilot_one_stage_b_s3_verdict.json, and Delta is BUILT by "
                "calling build_nulls on the pinned anchor then cross-checked at 1e-9. "
                "The control_arms pass caught its own hand-copied constant at 8.82e-05 "
                "pp; the fix was to delete the transcription step, not to loosen a "
                "tolerance."),
            "bootstrap_offsets": (
                "arm_index 1100/1101/1102, guard offset SEED+9700+13*axis. Disjoint "
                "from every archived block (0-1, 100-102, 200-204, 300-301, 400-408, "
                "500-503, 600-610, 700-702, 800-801, 900-902, 1000-1005). The check is "
                "EXECUTED by assert_seeds_disjoint (reads each archive's own recorded "
                "offsets and raises on intersection), not claimed in prose."),
            "zwfy6_evidence_dir_was_incomplete_and_was_repaired": (
                "the first preflight scanned only 7 offset ledgers on zwfy6 vs 8 on "
                "wzc1 -- 14 evidence files existed only on wzc1, so the disjointness "
                "check was running against a PARTIAL archive set and could have missed "
                "a real collision. All 14 were scp -O'd to zwfy6 and md5-verified "
                "(14/14 match) BEFORE the analysis ran. Generalisable: "
                "assert_seeds_disjoint must be pointed at a COMPLETE evidence dir, and "
                "the two disks' proposal trees are not automatically in sync (zwfy6's "
                "is a hand copy, not a git checkout)."),
            "gate_constants_selftested": (
                "E[range of k]/sigma is k-DEPENDENT: k=2 -> 1.1283791670955126, k=3 -> "
                "1.6925687506432689 (both closed form, re-derived not trusted), k=8 -> "
                "~2.847 (Monte Carlo, validated by reproducing the k=3 closed form). "
                "NONE IS USED HERE (one seed per arm, 2 checkpoints per arm), and they "
                "are recorded as DECLARED_UNUSED so nobody can reuse a wrong c_k from "
                "this document. A ratio of two ranges neither of which clears its own "
                "floor is UNDEFINED, not a direction -- the error that voided "
                "within_arm_lr_refutation_20260813."),
            "preflight_validated_the_pipeline_before_the_arms_landed": (
                "--preflight_only ran every guard, the anchor build, the Delta "
                "cross-check and the keep12 reproduction gate WHILE both trainings "
                "were in flight, writing nothing. keep12 seed101 reproduced to "
                "0.000e+00 pp. --preflight_ignore_own_training (needed because our own "
                "training held the cards) is HARD-REFUSED outside preflight mode, so "
                "the GPU refuse-guard can never be bypassed by a run that writes an "
                "evidence file."),
        },
        "node_of_record": {
            "node": d["node"], "node_ips": d.get("node_ips"),
            "numpy": d["numpy_version"], "python": d["python_version"],
            "why_one_node": ("numpy Generator.multinomial differs in 19/10000 rows "
                             "between .82's 2.4.6 and .73's 2.5.1, so every statistic "
                             "comes from ONE node (.73) and the version is pinned with "
                             "--expect_numpy. Training is unaffected."),
            "training_nodes": {"keep14": ".73 (8xH20, zwfy6)",
                               "keep13": ".82 (8xH20, zwfy6)"},
            "not_touched": ["LOCAL and .21 (SparseForge #246 token-matched arms)",
                            ".104 (paperC Qwen3-8B heal, PID 3343471)"],
            "budget_enforced_by_the_launcher": ("scripts/_run_a04_shallow_ladder.sh "
                                                "and both analysis entry points refuse "
                                                "by IP on .104 and .21, plus a >8000 "
                                                "MiB GPU-held guard -- the budget is "
                                                "enforced in code, not by the "
                                                "operator's memory."),
        },
        "not_licensed": d["not_licensed"],
    }


def main():
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} STATUS.json EVIDENCE.json SHA256")
    status_path, ev_path, sha = sys.argv[1], sys.argv[2], sys.argv[3]

    old_bytes = open(status_path, "rb").read()
    old = json.loads(old_bytes.decode("utf-8"))
    old_keys = list(old.keys())
    old_n = len(old_keys)
    if NEW_KEY in old:
        raise SystemExit(f"FATAL: key {NEW_KEY} already present; refusing.")
    print(f"[status] observed {old_n} existing keys (NO count is hardcoded)")

    entry = build_entry(ev_path, sha, old_n)

    # --- TEXT-LEVEL APPEND ------------------------------------------------
    txt = old_bytes.decode("utf-8")
    if not txt.rstrip().endswith("}"):
        raise SystemExit("FATAL: STATUS.json does not end with '}'")
    i_root = txt.rindex("}")
    i_last_key_close = txt.rindex("}", 0, i_root)
    prefix = txt[:i_last_key_close + 1]

    body = json.dumps({NEW_KEY: entry}, indent=2, ensure_ascii=False, default=_f)
    inner = body[body.index("\n") + 1: body.rindex("\n")]
    new_bytes = (prefix + ",\n" + inner + "\n}").encode("utf-8")

    # GUARD 1: byte-prefix identity. No existing byte may be rewritten.
    pb = prefix.encode("utf-8")
    if new_bytes[:len(pb)] != pb:
        raise SystemExit("FATAL: byte prefix changed; refusing to write.")

    bak = status_path + ".bak_shallow_ladder"
    shutil.copy2(status_path, bak)
    tmp = status_path + ".tmp_shallow_ladder"
    with open(tmp, "wb") as f:
        f.write(new_bytes)

    def fail(msg):
        os.remove(tmp)
        shutil.copy2(bak, status_path)
        raise SystemExit(f"FATAL: {msg} -- restored from {bak}")

    new = json.loads(open(tmp, encoding="utf-8").read())
    if len(new) != old_n + 1:
        fail(f"new key count {len(new)} != {old_n + 1}")
    for k in old_keys:
        if new[k] != old[k]:
            fail(f"existing key {k} changed value")
    if list(new.keys()) != old_keys + [NEW_KEY]:
        fail("key order changed or the new key is not last")

    os.replace(tmp, status_path)
    final = open(status_path, "rb").read()
    if final[:len(pb)] != pb:
        shutil.copy2(bak, status_path)
        raise SystemExit("FATAL: post-write byte-prefix check failed; restored.")

    print(f"[status] appended {NEW_KEY}: {old_n} -> {len(new)} keys")
    print(f"[status] byte-prefix identity held over {len(pb)} of {len(old_bytes)} "
          f"old bytes ({100.0*len(pb)/len(old_bytes):.4f}%)")
    print(f"[status] backup at {bak}")
    print(f"[status] gpu_h_spent = {new[NEW_KEY]['gpu_h_spent']}")
    print(f"[status] verdict = {new[NEW_KEY]['verdict']}")


if __name__ == "__main__":
    main()
