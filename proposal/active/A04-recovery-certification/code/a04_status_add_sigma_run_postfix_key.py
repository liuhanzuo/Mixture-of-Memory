#!/usr/bin/env python3
"""Append ONE key to A04's STATUS.json: `sigma_run_postfix_k2_20260813`.

PURE APPEND, enforced at the TEXT level, not merely semantically. Two prior passes
today had `indent` / `ensure_ascii` silently re-serialise the whole file while a
semantic (dict-equality) check still passed, so this script asserts:

  G1  the pre-edit text, minus its final "\n}\n" terminator, is a BYTE-EXACT PREFIX
      of the post-edit text. Nothing before the insertion point may change -- not a
      space, not an escape, not a key order.
  G2  every one of the 44 prior keys re-serialises BYTE-IDENTICALLY (each value
      dumped alone, before and after).
  G3  key count 44 -> 45, and the new key is LAST.
  G4  the new key is not already present.

Writes only if all four hold. CPU only.
"""
import json
import os
import sys
from collections import OrderedDict

STATUS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "STATUS.json")
STATUS = os.path.abspath(STATUS)
NEW_KEY = "sigma_run_postfix_k2_20260813"
EXPECT_BEFORE = 44

PAYLOAD = {
    "scope": (
        "The FIRST sigma_run in this project computed only from runs whose seeds "
        "actually varied the DATA (post-ce5c298), and the K2 re-adjudication it "
        "licenses. keep7+fresh2 @1B, seeds {43,44,45}, step220000, S=3, df=2."),
    "date": "2026-08-13",
    "gpu_h_spent": 0.0,
    "how": (
        "EVAL-ONLY, zero re-scoring: all three arms' per-example shards were already "
        "on zwfy6 from 2026-08-11. Arithmetic on .73 as a CPU host (all 8 cards "
        "verified 0 MiB at dispatch and exit; refuse-guard armed at >8000 MiB)."),
    "prereg": "A04_SIGMA_RUN_POSTFIX_K2_PREREG.md, commit 94839e8, PRE-DATA",
    "verdict_doc": "A04_SIGMA_RUN_POSTFIX_K2_VERDICT.md",
    "evidence": "evidence/a04_sigma_run_postfix.json (md5 5d4f353822a2191ce1d4e1f0dbe00f88)",
    "driver": "code/a04_sigma_run_postfix_k2.py + code/a04_sigma_run_postfix_driver.sh",
    "node": ".73 (zwfy6), numpy 2.5.1, python 3.14.6, scipy ABSENT (chi2 df=2 closed form)",

    "premise_verified_before_any_number": {
        "post_fix": (
            "each arm's own logs/a03_dataorder_seed<S>_progress.log carries a "
            "POSITIVE preflight assertion of the fixed line, printed BEFORE launch: "
            "'trainer post-ce5c298 OK: 869: sampler = DistributedSampler(ds, "
            "shuffle=True, seed=args.seed)'. Launches 08-10 16:55:29 / 16:57:29 / "
            "08-11 12:04:07 = +17.6h / +17.6h / +36.7h after ce5c298 "
            "(2026-08-09 23:21:09 +0800). NOT inferred from ckpt mtime."),
        "different_seeds": (
            "log line 1 is '[seed] set_seed(43|44|45) on all ranks'; arch_meta.json "
            "carries the matching seed."),
        "config_matched": (
            "identical in every logged field except seed: keep_front=7 n_fresh=2 "
            "num_hidden_layers=9, fp32 master, world_size=8 bs=8 gaccum=2 eff_bs=128 "
            "seq_len=2048, max_steps=300000, dataset rows=15491607 "
            "dolmino_now15b.npy, torch AdamW, n_params=1015097344, resume from "
            "olmo2_probe2_1B_keep7fresh2_16card/step200000.pt, resume LR 6.504e-06. "
            "[optim] all FOUR groups at 2.00e-05 => LR UNIFORM, no differential-LR "
            "claim is made."),
        "step_alignment": (
            "all three have step{205000,210000,215000,220000}.pt and they align, but "
            "EVAL SHARDS EXIST AT step220000 ONLY. sigma_run is therefore measured at "
            "exactly ONE step -- correct for a sigma (a level across runs at a common "
            "step, needing no cross-step pairing)."),
    },

    "CRITICAL_this_is_DATA_ORDER_ONLY_variance": (
        "All three arms RESUME FROM ONE COMMON step200000.pt (102 tensors strict, Adam "
        "moments preserved), so fresh-tail INIT variance is identically ZERO in all "
        "three. This is a PURE DATA-ORDER sigma_run, not the full run-to-run sigma a "
        "from-prune multi-seed gate arm would have. The pre-fix families are the exact "
        "mirror image (init only). NEITHER is the full sigma_run. Direction of the "
        "bias: DOWNWARD for full run-to-run variance, i.e. OPTIMISTIC for K2. "
        "Pre-registered as a limitation before any number (PREREG s1.5, s4.2)."),

    "Q1_sigma_run_pp_df2": {
        "triviaqa": 0.2687920190518543,
        "popqa": 0.23109098612117585,
        "mmlu_content": 0.05829216488792428,
        "nq_open": 0.07996529704805887,
        "chi2_95ci_pp": {
            "triviaqa": [0.13993846105449594, 1.6892665382528293],
            "popqa": [0.12030455108371318, 1.4522766155008913],
            "mmlu_content": [0.030350846442700876, 0.36635180436920115],
            "nq_open": [0.04163124905862826, 0.5025629932424913],
        },
        "chi2_width_multiplicative": 12.070747932867992,
        "estimator": ("per-axis ARM MEAN per seed (absolute accuracy), NOT the paired "
                      "delta; s = sample sd ddof=1; df = S-1 = 2"),
        "standing_rule_honoured": ("no sigma_run point estimate is quoted without its "
                                   "d.o.f. AND its chi2 interval (A03 "
                                   "DATAORDER_PREREG.md s4). At df=2 sigma is VERY "
                                   "imprecise: 12.07x multiplicative CI width."),
        "independent_corroboration": ("my independently-loaded arm means reproduce "
                                      "a03_sigma_run_n3.json's recorded means_pct for "
                                      "seeds 43/44/45 to max|diff| = 0.000e+00 on all "
                                      "four axes (byte-exact), and my df=3 chi2 CI on "
                                      "triviaqa reproduces its recorded "
                                      "[0.22877984971402363, 1.505793867745346]."),
    },

    "Q3_K2_VERDICT": {
        "verdict": "K2_DOES_NOT_FIRE",
        "n_decision_axes_exceeding_delta": 0,
        "rule_needs": ">=2 of the 3 decision axes (triviaqa/popqa/mmlu_content); "
                      "nq_open DEMOTED by design s5.2, zero decision weight",
        "tightest_axis": "popqa, bound_3 0.3896 vs Delta 1.3205 = 3.39x headroom",
        "bound_3_pp": {"triviaqa": 0.4531, "popqa": 0.3896,
                       "mmlu_content": 0.0983, "nq_open": 0.1348},
        "delta_pp_used": {"triviaqa": 4.043134195274186,
                          "popqa": 1.3205298941613512,
                          "mmlu_content": 1.0238926078906134,
                          "nq_open": 0.9695290858725762},
        "delta_provenance": ("build_nulls() was IMPORTED AND CALLED on the pinned "
                            "intact anchor (A03_1B_base, rule G0, split tie "
                            "convention); the Delta it produced was CROSS-CHECKED "
                            "against the canonical full-precision constants and "
                            "matched on all 4 axes within 1e-9. No Delta or null is "
                            "copied from prose. Delta NEVER substituted (guard G2)."),
        "ROBUST_ACROSS_ALL_FOUR_ESTIMATORS": (
            "K2 does not fire on the point estimate under ANY of: keep12 S3 df2 "
            "(THE PRE-REGISTERED one, 0/3), keep7 s7.2-clean S3 df2 (0/3), keep7 "
            "contaminated S4 df3 as A03 recorded (0/3), pooled keep7clean+keep12 df4 "
            "(0/3). So the verdict is not an artefact of the contested inclusion "
            "decision, and equally is not rescued by it. The pooled row is a "
            "SENSITIVITY only -- substituting it remains NOT LICENSED."),
        "chi2_upper_reported_NOT_ORed_IN": (
            "at the chi2 upper limit of sigma, popqa alone would exceed Delta (2.4484 "
            "vs 1.3205) = 1 of 3, below the >=2 rule. Same shape on the "
            "pre-registered keep12 family (3.5260 vs 1.3205, 1 of 3), matching "
            "pilot_one.MAIN_correction_20260812_1630. This is NOT a second decision "
            "rule: PREREG s4.3 forecloses OR-ing it in in BOTH directions -- not "
            "'K2 FIRES' because the upper bound would exceed, and not 'K2 cleared' "
            "because the point estimate does not. Honest line that must ship with "
            "any K2 statement: the verdict is FRAGILE on one of three decision axes "
            "(popqa). On the higher-d.o.f. df=3 family even the chi2 upper fires 0/3."),
        "K2_LIMB_1_IS_NOT_EVALUABLE": (
            "K2 has two limbs joined by 'equivalently'. Limb 1 needs 'the smallest "
            "between-arm residual difference the paper WANTS TO CLAIM' -- a quantity "
            "A04 has NEVER DECLARED (the 4-arm gate never ran; no between-arm "
            "difference is claimed anywhere). Limb 1 is therefore not evaluable on "
            "this or any family. Not treated as satisfied, not as failed. Recorded "
            "because no prior pass noticed it."),
        "NOT_A_CLEARANCE": (
            "Pre-committed in PREREG s4.2 BEFORE the data: a large sigma KILLS, a "
            "small sigma does NOT clear. (i) WRONG ARM -- keep7 = 56.2% depth, a "
            "confirmed CONSTANT-REJECT rung; a saturated deficit is highly "
            "reproducible, so low seed variance is EXACTLY WHAT SATURATION LOOKS "
            "LIKE, and K2 is a variance gate STRUCTURALLY BLIND to it. (ii) WRONG "
            "BUDGET -- 20k warm-resume steps. (iii) PARTIALLY STOCHASTIC -- common "
            "init, so sigma is downward-biased. Pilot Two stays BLOCKED: no rung is "
            "known where NI can be OBSERVED TO ACCEPT, which is a RUNG-SELECTION "
            "problem, not a variance problem."),
    },

    "Q2_the_premise_needed_one_CORRECTION": {
        "what_the_task_assumed": ("that STATUS.json:power_analysis's sigma-hat came "
                                 "from PRE-fix arms, so it measured init variance "
                                 "rather than data-order variance."),
        "what_is_actually_true": (
            "A03's keep7 sigma is not a PRE-fix estimate -- it is a CONTAMINATED one: "
            "a03_sigma_run_n3.json families.keep7_20k_cpt.seeds = [0,43,44,45], i.e. "
            "THREE post-fix draws pooled with ONE pre-fix draw. Seed 0 = A03 Arm 3, "
            "logs/a03_arm3_progress.log '[08-09 01:11:43] launched torchrun', "
            "22h09m BEFORE ce5c298; its log line 1 is set_seed(42) and it has NO "
            "post-ce5c298 preflight line."),
        "ratio_clean_over_contaminated": {
            "triviaqa": 0.6656, "popqa": 1.1798,
            "mmlu_content": 1.0509, "nq_open": 1.0660},
        "direction": (
            "MIXED, and on the only axis that moves materially the removal makes "
            "sigma SMALLER (triviaqa 0.67x), because seed 0's triviaqa mean sits "
            "+0.6780 pp = +2.52 clean-sigma above the other three and was carrying "
            "most of the df=3 spread. On the other three axes seed 0 is within "
            "+-1 clean-sigma (+0.45/+0.98/+0.92) and removing it slightly RAISES "
            "sigma, as expected from df 3->2."),
        "THE_TASKS_ACTUAL_QUESTION_IS_NOT_COMPUTABLE": (
            "'is real data-order variance larger or smaller than pre-fix init "
            "variance' needs >=2 PRE-fix SEED REPLICATES of one arm with evals on "
            "these axes. Searched BOTH disks: "
            "outputs/olmo2_probe2_7B_keep14fresh2_seed1234 is the only pre-fix "
            "multi-'seed' object and it is 7B with NO eval shards on either disk; "
            "A03 Arms 3/4/6 are pre-fix but are DIFFERENT LR SCHEDULES "
            "(arm4=peaklr, arm6=lowerband) so their spread is a schedule effect. "
            "Reported as NOT-COMPUTABLE rather than proxied."),
        "consequence_for_power_analysis": (
            "NEITHER optimistic nor pessimistic on this evidence. The power analysis "
            "that actually drives K2 uses the KEEP12 family, which was never "
            "contaminated. The keep7 numbers move 0.67-1.18x in BOTH directions and "
            "K2's verdict is unchanged under every estimator. So the worry 'the "
            "sigma-hat measures the wrong thing, therefore every power calculation "
            "is mis-specified' is NOT SUSTAINED -- the defect is a real 口径 "
            "bookkeeping error in an archived file, not a change to any decision."),
    },

    "DEFECT_a03_keep7_family_violates_PROPOSAL_7_2": {
        "what": ("a03_sigma_run_n3.json families.keep7_20k_cpt pools PRE-fix seed 0 "
                 "with post-fix 43/44/45; families.pooled_df5 (and "
                 "STATUS.json:sigma_run_input_from_A03.pooled_df5) consume it."),
        "rule_violated": ("PROPOSAL.md s7.2: 'A pre-fix seed arm and a post-fix seed "
                          "arm are therefore not draws from the same distribution, "
                          "and must never enter the same sigma_run estimate.'"),
        "scope_of_the_correction": (
            "sampler_fix_and_pilot_one_disposition_20260812's sentence 'Every run A04 "
            "consumes as sigma_run input is POST-fix' is TRUE of the six runs it "
            "enumerates (43/44/45/101/102/103) and FALSE of the keep7 FAMILY as "
            "recorded, which carries a 4th draw it did not enumerate. That entry is "
            "NOT retracted -- its six-run claim stands and its sampler-fix "
            "verification stands."),
        "does_NOT_affect": ("K2's pre-registered estimator = the KEEP12 family "
                            "(101/102/103), all three post-fix with positive "
                            "preflight assertions. K2's arithmetic is untouched."),
        "archive_not_edited": ("a03_sigma_run_n3.json is ARCHIVED provenance and was "
                              "NOT modified. md5 5fb6cd4c3d693831e50d0817bda93ab8 "
                              "re-asserted at runtime."),
    },

    "EXCHANGEABILITY_the_7_2_exclusion_is_CONSERVATIVE_here_not_NECESSARY": {
        "finding": (
            "s7.2 excludes pre-fix arms because pre-fix seeds varied ONLY init. On "
            "THIS arm that premise does not obtain: all four keep7 draws resume from "
            "one common ckpt, so init variance is zero in all four and the ONLY "
            "stochastic input in each is the sampler order. Tested MECHANICALLY with "
            "the real torch DistributedSampler at set_epoch(1): PRE-fix Arm 3's first "
            "12 rank-0 indices are BIT-IDENTICAL to post-fix seed=0; all 4 orders are "
            "distinct; Arm3-vs-post-fix rank-0 20k-slice Jaccard 0.0105/0.0104/0.0101 "
            "is INDISTINGUISHABLE from the post-fix-vs-post-fix 0.0104/0.0106/0.0104. "
            "=> on this arm seed 0 is a LEGITIMATE 4th draw from the SAME data-order "
            "family."),
        "what_was_done_about_it": (
            "The exclusion was STILL APPLIED for the headline sigma: s7.2 is a binding "
            "pre-registered rule and this document does not get to reinterpret it "
            "after seeing that the wider family would be convenient. Both readings are "
            "reported and the K2 verdict is identical under each, so nothing rests on "
            "the choice."),
        "GENERALISATION_PROHIBITED": (
            "This does NOT rehabilitate pre-fix seeds in general. It holds ONLY where "
            "a COMMON RESUME CKPT zeroes the init variance. For any arm pruned fresh "
            "per seed -- the keep12 Stage-B family, and EVERY ARM IN THE GATE DESIGN "
            "-- pre-fix seeds genuinely carry init-only variance and s7.2 applies with "
            "full force."),
    },

    "Q4_1B_sigma_gating_a_7B_experiment": {
        "delta_7B_split_canonical": {"triviaqa": 6.3291350869371374,
                                     "popqa": 2.245741921917712,
                                     "mmlu_content": 1.8613801452784504,
                                     "nq_open": 1.994459833795014},
        "delta_7B_source": ("evidence/a04_keep14_trajectory_ni.json and "
                            "evidence/a04_control_arms_ni.json, "
                            "per_convention.split.delta_pp -- identical in both. "
                            "NOT re-derived."),
        "KEY_STRUCTURAL_FACT": (
            "Every 7B Delta is 1.57-2.06x LARGER than its 1B counterpart, because "
            "Delta = 0.10 x residual(intact) and the 7B intact residual is larger. So "
            "a sigma held constant IN pp is MORE easily accommodated at 7B, not less "
            "-- the K2 test gets EASIER to pass as the anchor's residual grows. That "
            "is a property of a DATA-DEPENDENT MARGIN (must_not_claim[22], "
            "arXiv:2603.16213), not of the model, and it is a second independent "
            "reason not to read a non-firing K2 as reassurance."),
        "BOUND_DIRECTION_CANNOT_BE_SIGNED": (
            "Two unmeasured effects act in OPPOSITE directions: (i) Delta is "
            "1.57-2.06x larger at 7B => the 1B sigma is CONSERVATIVE/pessimistic as a "
            "7B gate input; (ii) this 1B sigma is DOWNWARD-biased (common init) => "
            "OPTIMISTIC. There is no measurement of how sigma_run scales with "
            "parameter count on THIS harness, so the product has unknown sign. "
            "Writing 'upper bound' or 'lower bound' would be a guess, so this "
            "document writes CANNOT BE SIGNED (as PREREG s6 pre-committed)."),
        "how_far_from_firing": (
            "K2 needs >=2 of 3, so the SECOND-easiest axis sets the bar: sigma would "
            "need to be ~8.9x larger to fire against the 1B Delta and ~14.0x against "
            "the 7B Delta. Not marginal -- BUT per NOT_A_CLEARANCE a large distance "
            "from firing is not evidence the gate is safe, since a constant-REJECT "
            "rung is EXPECTED to have small sigma."),
        "external_literature_may_inform_NOT_be_tabulated": (
            "arXiv:2508.13144 (NeurIPS 2025 Spotlight; OpenReview sAFottNlra, "
            "venueid=NeurIPS.cc/2025/Conference, Camera_Ready_Revision present -- "
            "DBLP has it CoRR-only so S2/DBLP alone would misread it) Table 4 gives "
            "OLMo-2 per-task noise at 1.5B/7B/13B/32B, the only published handle on "
            "the sign of the scale effect for this family. But it is INTACT-model "
            "noise, a rel-std over 30 consecutive checkpoints of ONE run (a "
            "checkpoint-SELECTION quantity), on THEIR OLMES protocol -- not a "
            "cross-run sigma on A04's base protocol. Per must_not_claim[20] it may be "
            "DISCUSSED, never TABULATED against these numbers."),
        "no_7B_sigma_exists": ("must_not_claim[23]: one seed per 7B rung, historical "
                              "seeds unrecorded, --seed postdates the trainer "
                              "revision that produced them. No 7B sigma_run is "
                              "computable or reconstructible. This block is a "
                              "SENSITIVITY, never a 7B result."),
    },

    "integrity": {
        "shard_completeness": ("shard INDEX SET exactly {0..7} (not a count), exact "
                              "item counts 17944/14267/3610/14042, 0 duplicate "
                              "item_ids, 0 nan, per cell; MMLU via nested "
                              "content_norm.correct."),
        "chat_template": ("False STRUCTURALLY: grep -c apply_chat_template == 0 in "
                          "BOTH eval_olmo2_closedbook_qa.py and "
                          "eval_olmo2_mmlu_content.py, and neither has a flag to "
                          "enable one. Assertion written `chat_template is not False` "
                          "-> FAIL (so None also fails), NEVER `is not True`. "
                          "add_bos False; greedy do_sample=False num_beams=1; "
                          "max_new_tokens=32; no few-shot."),
        "protocol_source": ("read from the ACTUAL driver "
                           "/tmp/a03_dataorder_ext_driver.sh on .73 and from the eval "
                           "scripts, NOT from summary.json:meta (which records "
                           "neither batch_size nor chat_template)."),
        "canonical_code_identical_across_disks": (
            "pilot_zero_rule_disagreement.py md5 1532257578b1f0855f25696710d488cb on "
            "both disks. analyze_1b_knowledge_floor.py DIFFERS between disks "
            "(4571d76 wzc1 vs 8b454b5 zwfy6) but an AST comparison shows all FOUR "
            "imported functions (best_constant_qa / longest_option_vector / "
            "best_constant_letter / paired_bootstrap) and all three constants "
            "(SEED=0, N_BOOT=10000, TIE_CONVS) are BYTE-IDENTICAL; the diff is "
            "confined to floor_cell()'s residual_fraction reporting, which "
            "build_nulls/ni_rule never call."),
        "seed_disjointness": ("EXECUTED with the self-excluding checker copied "
                             "UNWEAKENED from a04_keep12_trajectory_monotonicity.py, "
                             "against the UNION of both disks' evidence archives (32 "
                             "files). This run claims arm_index 900/901/902 + guard "
                             "SEED+8700, disjoint from 0,1 / 100-102 / 200-204 / "
                             "300,301 / 400-408 / 500-503 / 600-610 / 700-702 / "
                             "800,801. No clash."),
        "no_range_statistic": ("this analysis computes NO range statistic, so "
                              "E[range of k] constants are UNUSED and recorded as "
                              "such (c_3=1.6925687506, c_8=2.8475) so nobody can "
                              "later lift a c_n from this document. sigma is a SAMPLE "
                              "SD (ddof=1)."),
        "cross_node_precision": ("all arithmetic on ONE node (.73, numpy 2.5.1); no "
                                "number quoted finer than 0.01 pp across nodes "
                                "(must_not_claim[24])."),
    },

    "must_not_claim_additions_implied_by_this_entry": [
        "NOT a full run-to-run sigma_run -- data-order only, common init, "
        "downward-biased (see CRITICAL_this_is_DATA_ORDER_ONLY_variance).",
        "NOT a clearance of K2 and NOT an authorisation of Pilot Two.",
        "NOT a pre-vs-post-fix variance contrast -- that is not computable in this "
        "repo (see Q2.THE_TASKS_ACTUAL_QUESTION_IS_NOT_COMPUTABLE).",
        "NOT a trajectory / monotonicity / neighbour statistic, and no c_n is "
        "available from it.",
        "NOT a rehabilitation of pre-fix seeds in general.",
        "NOT a differential-LR claim: the [optim] lines show uniform 2e-5.",
    ],

    "retires_or_amends": {
        "power_analysis.sd_run_is_UNVERIFIED": (
            "already flagged stale by sampler_fix_and_pilot_one_disposition_20260812; "
            "this entry adds a SECOND, independent post-fix sigma_run family (keep7 "
            "data-order, df=2) alongside the keep12 one."),
        "sigma_run_input_from_A03.keep7_20k_cpt_df3": (
            "AMENDED, not deleted: it is s7.2-noncompliant as recorded (pools pre-fix "
            "seed 0). The s7.2-clean keep7 value is Q1_sigma_run_pp_df2 above. Note "
            "EXCHANGEABILITY: the exclusion is conservative on THIS arm, so the df=3 "
            "number is not WRONG as arithmetic -- it is mislabelled as to口径."),
        "sigma_run_input_from_A03.pooled_df5": (
            "inherits the same 口径 defect (it pools the keep7 family). Substituting "
            "any pooled sigma into K2 remains NOT LICENSED per that entry's own "
            "tempting_but_NOT_LICENSED clause."),
    },

    "next": (
        "UNCHANGED by this entry: Pilot Two stays BLOCKED pending a NEW pre-data doc "
        "showing a rung where NI can be OBSERVED TO ACCEPT. What this entry rules out "
        "is 'buy more seeds' as the way forward -- sigma is 3.4-10.4x inside Delta on "
        "every decision axis under every estimator, so d.o.f. on sigma is not the "
        "binding constraint. The binding constraint is RUNG SELECTION."),
}


def dump(o, ensure_ascii):
    return json.dumps(o, indent=2, ensure_ascii=ensure_ascii)


def detect_serialisation(pre_text, pre):
    """Discover the file's OWN indent / ensure_ascii convention instead of assuming
    this script's. This is the whole point of the G1 guard: the first run of this
    script assumed ensure_ascii=False and G1 caught it, because STATUS.json is
    actually written with ensure_ascii=True (its CJK is stored as \\uXXXX escapes).

    Returns (indent, ensure_ascii), or raises if neither combination reproduces the
    existing file byte-for-byte. Refusing on 'no combination reproduces it' is
    correct: if we cannot re-emit the current file exactly, we cannot append to it
    without rewriting bytes we do not own.
    """
    for ea in (True, False):
        for ind in (2, 4):
            if json.dumps(pre, indent=ind, ensure_ascii=ea) + "\n" == pre_text:
                return ind, ea, "exact (incl. trailing newline)"
            if json.dumps(pre, indent=ind, ensure_ascii=ea) == pre_text:
                return ind, ea, "exact (no trailing newline)"
    raise SystemExit(
        "FATAL: cannot reproduce STATUS.json byte-for-byte with any of "
        "indent in {2,4} x ensure_ascii in {True,False}. Refusing to append, "
        "because writing would rewrite bytes this script does not own.")


def main():
    pre_text = open(STATUS, encoding="utf-8").read()
    pre = json.loads(pre_text, object_pairs_hook=OrderedDict)

    indent, ensure_ascii, how = detect_serialisation(pre_text, pre)
    trailing_nl = pre_text.endswith("\n")
    print(f"[serialisation] detected indent={indent} ensure_ascii={ensure_ascii} "
          f"({how}); trailing_newline={trailing_nl}")

    def d(o):
        return dump(o, ensure_ascii)

    # G4 / G3 preconditions
    if NEW_KEY in pre:
        raise SystemExit(f"FATAL G4: key {NEW_KEY!r} already present.")
    if len(pre) != EXPECT_BEFORE:
        raise SystemExit(f"FATAL G3: expected {EXPECT_BEFORE} keys, found {len(pre)}.")
    prior_keys = list(pre.keys())
    prior_dumps = {k: d(pre[k]) for k in prior_keys}

    post = OrderedDict(pre)
    post[NEW_KEY] = PAYLOAD
    post_text = json.dumps(post, indent=indent, ensure_ascii=ensure_ascii)
    if trailing_nl:
        post_text += "\n"

    # ---- G1: TEXT-LEVEL append-only ------------------------------------
    # pre_text ends with the object terminator; strip exactly that and require the
    # remainder to be a byte-exact prefix of post_text.
    stripped = pre_text.rstrip()
    if not stripped.endswith("}"):
        raise SystemExit("FATAL G1: pre-edit file does not end with '}'.")
    prefix = stripped[:-1].rstrip()          # drop final '}' and trailing ws
    if not prefix.endswith("}"):
        raise SystemExit("FATAL G1: unexpected shape before the final '}'.")
    if not post_text.startswith(prefix):
        # locate first divergence for a useful message
        i = next((j for j in range(min(len(prefix), len(post_text)))
                  if prefix[j] != post_text[j]), min(len(prefix), len(post_text)))
        raise SystemExit(
            "FATAL G1: post-edit text is NOT a byte-exact extension of the pre-edit "
            f"text. First divergence at char {i}:\n"
            f"  pre : {prefix[max(0,i-60):i+60]!r}\n"
            f"  post: {post_text[max(0,i-60):i+60]!r}\n"
            "This is exactly the silent indent/ensure_ascii re-serialisation that a "
            "semantic check would have missed. Refusing to write.")

    # ---- G2: every prior key re-serialises byte-identically -------------
    changed = [k for k in prior_keys if d(post[k]) != prior_dumps[k]]
    if changed:
        raise SystemExit(f"FATAL G2: prior keys changed on re-serialise: {changed}")

    # ---- G3: count + position -------------------------------------------
    post_keys = list(post.keys())
    if len(post_keys) != EXPECT_BEFORE + 1:
        raise SystemExit(f"FATAL G3: {len(post_keys)} keys after append.")
    if post_keys[:EXPECT_BEFORE] != prior_keys:
        raise SystemExit("FATAL G3: prior key ORDER changed.")
    if post_keys[-1] != NEW_KEY:
        raise SystemExit("FATAL G3: new key is not last.")

    with open(STATUS, "w", encoding="utf-8") as f:
        f.write(post_text)

    # verify from disk
    new_text = open(STATUS, encoding="utf-8").read()
    rd = json.loads(new_text, object_pairs_hook=OrderedDict)
    assert list(rd.keys()) == post_keys, "post-write key mismatch"
    assert len(rd) == EXPECT_BEFORE + 1
    for k in prior_keys:
        assert d(rd[k]) == prior_dumps[k], f"post-write drift in {k}"
    assert new_text.startswith(prefix), "post-write G1 violated on disk"
    print(f"OK: {EXPECT_BEFORE} -> {len(rd)} keys; appended {NEW_KEY!r}")
    print("G1 text-level append-only: PASS (on-disk file is a byte-exact extension)")
    print(f"G2 all {EXPECT_BEFORE} prior keys byte-identical: PASS")
    print("G3 count + order + new-key-last: PASS")
    print(f"gpu_h_spent recorded: {PAYLOAD['gpu_h_spent']}")


if __name__ == "__main__":
    main()
