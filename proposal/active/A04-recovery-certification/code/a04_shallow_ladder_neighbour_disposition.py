#!/usr/bin/env python3
"""A04 shallow-rung ladder — the §2.0.2 DISPOSITION + the range/noise-floor
disclosures, computed as an evidence file so the verdict .md can be RENDERED and
nothing is hand-transcribed.

WHY THIS IS A SEPARATE FILE AND NOT AN EDIT TO THE ANALYSIS OUTPUT.
`evidence/a04_shallow_rung_ladder.json` is the output of the PRE-REGISTERED
analysis (`a04_shallow_rung_ladder_ni.py`, prereg commit a2e1a95) and its
sha256 is what `STATUS.json` pins. Re-writing it after the fact to bolt on a
disclosure would destroy that pin. So this probe READS it (read-only) and writes
a companion file.

WHAT IT DECIDES, AND FROM WHICH PRE-DATA DOCUMENT.
`A04_SHALLOW_LADDER_NEIGHBOUR_ADMISSIBILITY.md` (commit 46ea84d, PRE-DATA) fixed,
before any margin existed:

  * §2.0.2 DOES bind this ladder, but it gates ACCEPTS ONLY. Its §5 closing
    paragraph fixes the Branch-B reading verbatim: "under Branch B (both arms
    constant-REJECT) §2.0.2 is NOT triggered at all -- it gates accepts only ...
    The precondition is not vacuously satisfied -- it is not triggered."
  * `CERTIFIED` is STRUCTURALLY UNREACHABLE for this ladder whatever the numbers
    are: §2.0.2 conditions it on adjacent saved checkpoints on BOTH sides, and no
    upper neighbour of step5000 can exist (max_steps=5000; `final.pt` is written
    at the SAME step as `step5000.pt`, so it is the same point, not a neighbour).
  * The label rule (§2.3) applies per (arm, decision axis) ONLY to cells where
    `ni_accept == True`. This probe evaluates it mechanically rather than
    asserting the outcome, so a hypothetical accept could not be silently skipped.

RANGE / NOISE-FLOOR DISCLOSURES, WITH THE CONSTANT THAT MATCHES ITS OWN k.
`E[range of k iid N(0,sigma)]/sigma` is k-DEPENDENT. Using k=3's constant at k=8
makes a floor 40.6 % TOO LOW and manufactured a finding once
(A04_KEEP12_TRAJECTORY_MONOTONICITY_VERDICT); using k=3's at k=2 inflates a floor
by 50.0 % and can suppress a real move (admissibility §2.2). So each range here
is gated against the constant for ITS OWN k, and both the k and the constant are
recorded next to the range:

  * k=2 -> 2/sqrt(pi) = 1.1283791670955126   (the adjacent keep13/keep14 pair)
  * k=3 -> 3/sqrt(pi) = 1.6925687506432689   (the 3-rung keep12/13/14 ladder)

Both constants are READ from the analysis JSON's own `gate_constant_selftest`
(which re-derives them from the closed form rather than trusting a literal) and
then re-checked here against the closed form.

sigma is the MEAN OF THE PARTICIPATING CELLS' OWN `bootstrap_se_pp` -- the
per-cell recipe that reproduces `A04_GATE_DESIGN.md` §2.0.2's own worked example
exactly (0.135308 pp / 0.254033 pp = 0.5326 FAIL). The POOLED variant is the one
`PROPOSAL.md` §4.3 retracted as "1.69x off". Do not pool.

NOTHING HERE IS DECISION-BEARING, and the file says so in its own payload.
Prereg §4.2 / §5.5: no range or spread statistic decides anything in this pass.
A range that clears an ITEM-noise floor is not thereby resolved against
run-to-run variance -- ONE seed (101) per arm, so no `sd_run` exists at these
rungs. And no RATIO of two ranges is formed anywhere: a ratio of two ranges
neither of which clears its own floor is UNDEFINED, not a direction (the error
that voided `within_arm_lr_refutation_20260813`).

CPU only, read-only on every input. No GPU, no network, no RNG (every number
below is deterministic float arithmetic over already-computed SEs, so it is
node-independent -- unlike the bootstrap, which is pinned to .73).

Usage: a04_shallow_ladder_neighbour_disposition.py LADDER.json OUT.json PROJECT_ROOT
"""
from __future__ import annotations

import json
import math
import os
import sys

C_CLOSED_FORM = {2: 2.0 / math.sqrt(math.pi), 3: 3.0 / math.sqrt(math.pi)}
SEED_TRAIN = 101
STEP = 5000
SAVE_EVERY = 2500


def ck(k, ladder):
    """The range constant for THIS k, read from the analysis JSON's own selftest
    and re-checked against the closed form. Never a literal typed here."""
    st = ladder["gate_constant_selftest"]
    got = {2: st["c2_closed_form"], 3: st["c3_closed_form"]}[k]
    want = C_CLOSED_FORM[k]
    if abs(got - want) > 1e-15:
        raise SystemExit(f"FATAL: c_{k} from the analysis JSON is {got!r} but the "
                         f"closed form is {want!r}. Refusing to gate on a wrong c_k.")
    return got


def ckpt_facts(root, keep):
    """The neighbour INVENTORY, measured on disk. §2.0.2's escape hatch is
    '(or a statement that none exist)', and a statement covering BOTH sides would
    be FALSE here -- the lower neighbour is real. So both sides are stat'ed."""
    od = os.path.join(root, f"outputs/olmo2_probe2_1B_keep{keep}f2_dolmino_shallow_"
                            f"seed{SEED_TRAIN}")
    def st(name):
        p = os.path.join(od, name)
        return ({"exists": True, "path": p, "size_bytes": os.path.getsize(p),
                 "mtime": int(os.path.getmtime(p))} if os.path.isfile(p)
                else {"exists": False, "path": p})
    lower = st(f"step{STEP - SAVE_EVERY}.pt")
    same = st(f"step{STEP}.pt")
    final = st("final.pt")

    # was the lower neighbour SCORED? (admissibility §6.5: if it was not, the
    # verdict must say the ckpt exists on disk but was not scored -- the weaker,
    # honest disclosure. Which one applies is decided by whether the eval ran,
    # NOT by what the step5000 numbers turned out to be.)
    scored = {}
    for sub, kind in (("olmo2_mmlu_content_results", "mmlu"),
                      ("olmo2_closedbook_results", "cb")):
        tag = f"A04_1B_shallow_keep{keep}_seed{SEED_TRAIN}_step{STEP - SAVE_EVERY}"
        scored[kind] = os.path.isdir(os.path.join(root, sub, tag))
    return {
        "output_dir": od,
        "lower_neighbour_step": STEP - SAVE_EVERY, "lower_neighbour": lower,
        "this_checkpoint_step": STEP, "this_checkpoint": same,
        "final_pt": final,
        "upper_neighbour": {
            "exists": False,
            "why": ("max_steps=5000 and the trainer's save condition is "
                    "`step % save_every == 0 and step > 0` plus a terminal "
                    "_save(..., final=True). `final.pt` is named by "
                    "'final' if final else f'step{step}' at the SAME step value, so it "
                    "is the SAME point as step5000.pt, not a checkpoint beyond it. No "
                    "upper neighbour can exist without new training."),
            "final_pt_is_same_step_not_a_neighbour": True,
        },
        "n_neighbours_present": int(lower["exists"]),
        "lower_neighbour_was_SCORED": any(scored.values()),
        "lower_neighbour_scored_detail": scored,
        "lower_neighbour_disclosure_grammar": (
            "the lower neighbour EXISTS ON DISK but was NOT SCORED (no step2500 eval "
            "dir on either disk, and admissibility §6.5 authorises no GPU for one). "
            "Per §6.5 that is the weaker disclosure and it is the honest one."
            if lower["exists"] and not any(scored.values()) else
            "see lower_neighbour_was_SCORED"),
    }


def range_block(label, k, arms, axis, cells, c, why_not_decision_bearing,
                extra=None):
    """One range, gated against the floor for ITS OWN k.

    The gated quantity is `margin_pp`, whose per-cell SE is `bootstrap_se_pp`:
    margin = diff_lower95 + Delta, and Delta is a per-axis CONSTANT, so the range
    of margins across arms equals the range of the paired-difference bounds, on
    the same scale as the SE. This is the same pairing GATE_DESIGN §2.0.2 uses.
    """
    ms = [cells[a][axis]["margin_pp"] for a in arms]
    ses = [cells[a][axis]["bootstrap_se_pp"] for a in arms]
    rng = max(ms) - min(ms)
    sigma = sum(ses) / len(ses)          # per-cell mean, NEVER pooled cross-axis
    floor = c * sigma
    return {
        "label": label,
        "k": k, "k_matches_n_cells": k == len(arms),
        "c_k": c,
        "c_k_source": ("read from the analysis JSON's gate_constant_selftest "
                       "(closed form, re-derived) and re-checked here"),
        "c_k_closed_form_expr": {2: "2/sqrt(pi)", 3: "3/sqrt(pi)"}[k],
        "arms": list(arms), "axis": axis,
        "margins_pp": ms,
        "per_cell_bootstrap_se_pp": ses,
        "sigma_pp": sigma,
        "sigma_recipe": ("mean of the participating cells' OWN bootstrap_se_pp -- the "
                         "per-cell recipe that reproduces GATE_DESIGN §2.0.2's worked "
                         "example exactly. The POOLED variant is the one PROPOSAL.md "
                         "§4.3 retracted as 1.69x off. Do NOT pool."),
        "range_pp": rng,
        "noise_floor_pp": floor,
        "range_over_floor": (rng / floor) if floor > 0 else None,
        "CLEARS_ITS_OWN_FLOOR": bool(rng > floor),
        "if_wrong_c_k_had_been_used": {
            f"c_{kk}": {"c": C_CLOSED_FORM[kk],
                        "floor_pp": C_CLOSED_FORM[kk] * sigma,
                        "would_have_cleared": bool(rng > C_CLOSED_FORM[kk] * sigma),
                        "floor_error_vs_correct_pct":
                            100.0 * (C_CLOSED_FORM[kk] / c - 1.0)}
            for kk in sorted(C_CLOSED_FORM) if kk != k},
        "NOT_DECISION_BEARING": True,
        "why_not_decision_bearing": why_not_decision_bearing,
        "clearing_an_item_noise_floor_is_NOT_resolution_against_seed_variance": (
            "sigma here is the ITEM-sample bootstrap SE. ONE seed (101) per arm, so no "
            "sd_run exists at these rungs and no range here may be read as resolved "
            "against run-to-run variance."),
        **(extra or {}),
    }


def main():
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} LADDER.json OUT.json PROJECT_ROOT")
    ladder_path, out_path, root = sys.argv[1], sys.argv[2], sys.argv[3]
    L = json.load(open(ladder_path))

    conv = L["mmlu_tie_convention"]
    cells = L["per_convention"][conv]["cells"]
    verd = L["per_arm_verdict"]
    DEC = list(L["decision_axes"])
    AX = DEC + list(L["demoted_axes"])
    branch = L["BRANCH"]
    c2, c3 = ck(2, L), ck(3, L)

    new_arms = sorted([a for a in verd if not a.endswith("_REF")],
                      key=lambda s: int(s.replace("keep", "").split("f2")[0]))
    all_rungs = sorted(verd, key=lambda s: int(s.replace("keep", "").split("f2")[0]))

    # ---- 1. §2.0.2 disposition, evaluated NOT asserted --------------------
    accepting = {a: [ax for ax in DEC if cells[a][ax]["ni_accept"]] for a in verd}
    n_acc_cells = sum(len(v) for v in accepting.values())
    triggered = n_acc_cells > 0

    disp = {
        "precondition": "A04_GATE_DESIGN.md §2.0.2 (neighbour precondition on ACCEPTS)",
        "reconciliation_document": {
            "file": "A04_SHALLOW_LADDER_NEIGHBOUR_ADMISSIBILITY.md",
            "commit": "46ea84d", "written_PRE_DATA": True,
            "pre_data_evidence": ("committed 2026-08-13 14:06:10 UTC with keep14 at "
                                  "step 3740-3800/5000 and keep13 at 3900-3960/5000, "
                                  "no step5000.pt, no A04_1B_shallow_* eval dir and no "
                                  "analysis JSON on either disk"),
        },
        "binds_this_ladder": True,
        "why_it_binds": ("§2.0.2 is scoped to 'any NI(Delta) accept reported by this "
                         "gate', and this ladder's accepts would come from the same "
                         "IMPORTED ni_rule under the same frozen Delta. Prereg §5.5 "
                         "renounces a CLAIM; §2.0.2 imposes a DISCLOSURE duty, and "
                         "renouncing a claim cannot discharge a duty."),
        "gates_accepts_only": True,
        "n_accepting_decision_axis_cells": n_acc_cells,
        "accepting_cells_per_arm": accepting,
        "TRIGGERED": triggered,
        "disposition": (
            "NOT_TRIGGERED -- no cell accepts on any decision axis. Per the "
            "reconciliation document §5 (fixed pre-data, phrasing adopted verbatim from "
            "the full32 pass): 'under Branch B (both arms constant-REJECT) §2.0.2 is "
            "NOT triggered at all -- it gates accepts only ... The precondition is not "
            "vacuously satisfied -- it is not triggered.' Branch B needs nothing from "
            "that document."
            if not triggered else
            "TRIGGERED -- the §2.3 label rule below applies per accepting cell"),
        "CERTIFIED_is_structurally_unreachable": {
            "value": True,
            "why": ("§2.0.2 conditions a certified reading on the immediately adjacent "
                    "saved checkpoints on BOTH sides. No upper neighbour of step5000 can "
                    "exist (max_steps=5000; final.pt is the SAME step). Decided pre-data "
                    "in the admissibility document §2.3 -- no datum can change it."),
            "consequence": ("no cell of this ladder may be labelled CERTIFIED, whatever "
                            "the numbers are. Here it is moot: nothing accepts."),
        },
        "label_rule_evaluated_per_accepting_cell": {},   # empty iff nothing accepts
        "label_rule_source": ("A04_SHALLOW_LADDER_NEIGHBOUR_ADMISSIBILITY.md §2.3, "
                              "three labels with mechanical triggers, no free parameter"),
        "why_the_empty_dict_is_the_correct_output": (
            "the label rule is defined ONLY over cells with ni_accept == True. It is "
            "evaluated here rather than skipped by assertion, so an accept could not "
            "have been silently passed over; the dict is empty because the set of "
            "accepting decision-axis cells is empty."),
        "neighbour_inventory_per_arm": {
            f"keep{k}": ckpt_facts(root, k)
            for k in (int(a.replace("keep", "").split("f2")[0]) for a in new_arms)},
        "step2500_was_NOT_scored_and_no_GPU_was_authorised_for_it": (
            "admissibility §6.5: 'Does not spend GPU. No step2500 eval is requested or "
            "authorised by this file.' Under Branch B the disclosure is not owed at all "
            "(§2.0.2 gates accepts only), so not scoring it costs nothing here."),
        "2500_steps_is_NOT_a_neighbourhood": {
            "verdict": True,
            "why": ("2500 steps is 50 % of this run's 5000-step horizon and spans LR "
                    "1.143706e-05 -> 2.000000e-06, a factor 5.7185 (half a cosine of "
                    "annealed training). Admissibility §3 answers this question against "
                    "us: the k=2 gate is NECESSARY-but-not-sufficient and a move that "
                    "clears it must NOT be read as instability."),
            "forbidden_comparisons": [
                "comparing a 2500-step move to the 500-step keep8/keep10 ranges (7B, "
                "different spacing, different scale, different corpus)",
                "writing 'wider spacing is more convincing' -- wider spacing makes a "
                "neighbour check WEAKER because it admits training progress as an "
                "explanation (banned by name in admissibility §3 consequence 4)",
            ],
        },
        "one_process_provenance_no_resume_seam": (
            "`grep -c resume` on both trainer logs returns 0: each arm is a single "
            "uninterrupted process, so no range here straddles a resume seam "
            "(§2.0.2 trap 2)."),
    }

    # ---- 2. range disclosures, each against ITS OWN k ---------------------
    why_ladder = ("prereg §5.3 / §5.5: the 3-rung spread is across arms of DIFFERENT "
                  "DEPTH -- it is the quantity the design VARIES, not a repeat "
                  "measurement, so it is not a neighbour range and not a sigma_run. "
                  "The verdict is decided by per-cell NI margins (all 12 decision-axis "
                  "cells reject), never by this range.")
    why_pair = ("prereg §5.7 makes the PAIRED item bootstrap the test of record for a "
                "keep13-vs-keep14 difference (see the paired CI table), not a "
                "range-vs-floor heuristic. This k=2 block is a disclosure so that the "
                "floor is stated with the constant that matches k=2, and so that "
                "1.6926 can never be silently reused here.")

    ranges = {
        "ladder_k3_margin_range_keep12_keep13_keep14": {
            ax: range_block(
                "3-rung 1B depth-ladder margin range (keep12/keep13/keep14)",
                3, all_rungs, ax, cells, c3, why_ladder,
                extra={"is_a_neighbour_range": False,
                       "is_a_sigma_run": False}) for ax in AX},
        "adjacent_pair_k2_margin_range_keep13_keep14": {
            ax: range_block(
                "adjacent-rung margin range (keep13 vs keep14)",
                2, new_arms, ax, cells, c2, why_pair,
                extra={"is_a_neighbour_range": False,
                       "is_a_sigma_run": False,
                       "test_of_record_is_the_paired_bootstrap_CI":
                           L["adjacent_rung_paired_differences"]["per_pair"]
                            .get("keep14_minus_keep13", {}).get(ax)}) for ax in AX},
    }

    out = {
        "scope": ("A04 shallow-rung ladder -- the §2.0.2 DISPOSITION and every range's "
                  "own noise floor, as an evidence file so the verdict .md can be "
                  "RENDERED rather than hand-written."),
        "companion_to": {
            "file": os.path.basename(ladder_path),
            "BRANCH": branch, "headline": L["headline"],
            "node_of_record": L["node"], "numpy": L["numpy_version"],
            "note": ("that file is the output of the PRE-REGISTERED analysis and its "
                     "sha256 is pinned in STATUS.json; this probe is read-only on it "
                     "and writes a companion rather than editing it."),
        },
        "this_file_is_deterministic_and_node_independent": (
            "every number here is float arithmetic over SEs already computed by the "
            "pinned analysis -- no RNG, so unlike the bootstrap it does not depend on "
            "the numpy version. Run on the node of record anyway for provenance."),
        "gpu_h": 0.0,
        "section_2_0_2_disposition": disp,
        "range_disclosures": ranges,
        "range_constants_used": {
            "c_2": {"value": c2, "expr": "2/sqrt(pi)", "used_for": "the k=2 adjacent pair"},
            "c_3": {"value": c3, "expr": "3/sqrt(pi)", "used_for": "the k=3 3-rung ladder"},
            "c_8_recorded_but_unused": L["gate_constant_selftest"]["c8_monte_carlo"],
            "why_k_matters": ("E[range of k]/sigma is k-dependent. c_3 at k=8 makes a "
                              "floor 40.6 % TOO LOW and manufactured a finding once; "
                              "c_3 at k=2 inflates a floor by 50.0 % and can suppress a "
                              "real move. Each range above records its own k, its own "
                              "constant, AND what the wrong constant would have done."),
        },
        "no_ratio_of_ranges_is_formed": (
            "no ratio of two ranges appears anywhere in this pass. A ratio of two ranges "
            "neither of which clears its own floor is UNDEFINED, not a direction -- the "
            "error that voided within_arm_lr_refutation_20260813."),
        "nothing_here_is_decision_bearing": (
            "prereg §4.2 / §5.5. The verdict is the per-cell NI margin table; every "
            "range above is a disclosure carrying its own floor."),
    }

    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)
        fh.write("\n")

    print(f"BRANCH {branch}; §2.0.2 TRIGGERED = {triggered} "
          f"({n_acc_cells} accepting decision-axis cells)")
    print(f"  disposition: {disp['disposition'].split('--')[0].strip()}")
    for grp, per_ax in ranges.items():
        k = per_ax[AX[0]]["k"]
        print(f"\n{grp}  (k={k}, c_k={per_ax[AX[0]]['c_k']:.16f})")
        for ax in AX:
            r = per_ax[ax]
            print(f"  {ax:<14} range {r['range_pp']:8.4f} pp  floor "
                  f"{r['noise_floor_pp']:7.4f} pp  ratio "
                  f"{r['range_over_floor']:6.3f}  clears={r['CLEARS_ITS_OWN_FLOOR']}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
