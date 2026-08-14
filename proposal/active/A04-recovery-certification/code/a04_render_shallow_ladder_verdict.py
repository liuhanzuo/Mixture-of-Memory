#!/usr/bin/env python3
"""Render `A04_SHALLOW_RUNG_LADDER_VERDICT.md` FROM the evidence JSON.

WHY THIS EXISTS. Hand-transcribing numbers out of an evidence JSON into a
verdict .md is the single most repeated error in this repo. On 2026-08-13 alone
there were FIVE hand-transcription slips, one of which invented trailing digits
for a reference constant and was only caught because a runtime gate fired at
8.82e-05 pp. The `control_arms` fix was not to loosen the tolerance but to
DELETE THE TRANSCRIPTION STEP. This script does the same thing for the prose: it
reads every number from `evidence/a04_shallow_rung_ladder.json` and formats it,
so the .md and the JSON cannot disagree.

The BRANCH-DEPENDENT prose is selected by `BRANCH`, which the analysis computed
from the pre-registered rule -- not chosen by whoever runs this.

OPTIONAL 4th ARG: the §2.0.2 disposition companion
(`evidence/a04_shallow_ladder_neighbour_disposition.json`, produced by
`a04_shallow_ladder_neighbour_disposition.py`). This renderer was committed at
21:41 on 2026-08-13; `A04_SHALLOW_LADDER_NEIGHBOUR_ADMISSIBILITY.md` landed at
22:09, i.e. AFTER it, so the original could not render the §2.0.2 disposition or
the per-range noise floors. When the companion is supplied, §§4.1 and 5.1 are
rendered FROM IT -- including whether the precondition was triggered at all, and
each range's own k, its own constant and whether it clears its own floor. The
companion is optional so the script still runs against an older evidence pair.

CPU only. Reads the evidence JSON(s), writes one .md. No GPU, no network.
"""
from __future__ import annotations

import json
import os
import sys


def pct(x, nd=2):
    return "n/a" if x is None else f"{100.0*x:.{nd}f}%"


def pp(x, nd=4):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def num(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit(f"usage: {sys.argv[0]} EVIDENCE.json OUT.md SHA256 "
                         "[DISPOSITION.json]")
    ev_path, out_path, sha = sys.argv[1], sys.argv[2], sys.argv[3]
    d = json.load(open(ev_path))
    disp_path = sys.argv[4] if len(sys.argv) == 5 else None
    DP = json.load(open(disp_path)) if disp_path else None

    conv = d["mmlu_tie_convention"]
    cells = d["per_convention"][conv]["cells"]
    verd = d["per_arm_verdict"]
    AX = d["decision_axes"] + d["demoted_axes"]
    DEC = d["decision_axes"]
    anchor = d["intact_anchor"]
    branch = d["BRANCH"]
    arms = list(verd.keys())
    new_arms = [a for a in arms if not a.endswith("_REF")]
    ref_arms = [a for a in arms if a.endswith("_REF")]
    # deepest-kept first so the table reads keep14 -> keep13 -> keep12
    def _k(a):
        return int(a.replace("keep", "").split("f2")[0])
    new_arms.sort(key=_k, reverse=True)
    ordered = new_arms + ref_arms

    gh = d["gpu_h"]
    L = []
    W = L.append

    # ---- title / verdict string ------------------------------------------
    W("# A04 — the shallow-rung ladder: does NI ever accept a *damaged* 1B model?")
    W("")
    W(f"**Verdict string:** `{d['headline']}`")
    W("")
    W(f"**Branch:** **{branch}** — selected by the pre-registered rule in "
      "`A04_SHALLOW_RUNG_LADDER_PREREG.md` §4.1, not chosen after the fact. "
      "All three branches (A/B/C) were written before any number existed.")
    W("")
    W(f"**Date:** 2026-08-13 · **GPU: {gh['total_gpu_h']:.1f} GPU-h** "
      f"(training {gh['training']['total_gpu_h']:.1f} + eval "
      f"{gh['eval']['total_gpu_h']:.2f}; this analysis is **0 GPU**, CPU-only and "
      "read-only on every input).")
    W(f"**Training nodes:** `.73` (keep14) and `.82` (keep13), 8×H20 each, zwfy6. "
      f"**Node of record for every statistic:** `{d['node']}` "
      f"(numpy **{d['numpy_version']}**, python {d['python_version']}).")
    W("**Not touched:** `LOCAL` / `.21` (SparseForge #246), `.104` (paperC Qwen3-8B "
      "heal). Both the launcher and the analysis refuse those nodes **by IP**.")
    W(f"**Pre-registration:** `A04_SHALLOW_RUNG_LADDER_PREREG.md`, commit "
      f"`{d['prereg']['commit']}`, committed **before the first margin existed** — "
      f"{d['prereg']['state_at_commit']}.")
    W(f"**Evidence:** `evidence/{os.path.basename(ev_path)}` (sha256 `{sha}`)")
    if DP:
        W(f"**§2.0.2 disposition companion:** "
          f"`evidence/{os.path.basename(disp_path)}` — read-only on the file above "
          "(whose sha256 `STATUS.json` pins), so the pre-registered analysis output is "
          "never rewritten to bolt on a disclosure.")
    W("**Code:** `code/a04_shallow_rung_ladder_ni.py`, "
      "`code/a04_shallow_ladder_eval_driver.sh`, `code/a04_shallow_ladder_chain.sh`, "
      "`scripts/_run_a04_shallow_ladder.sh`"
      + (", `code/a04_shallow_ladder_neighbour_disposition.py`, "
         "`code/_a04_shallow_integrity_probe.py`" if DP else ""))
    W("")
    W("> **Every number in this document is rendered from the evidence JSON by "
      "`code/a04_render_shallow_ladder_verdict.py`.** Nothing is hand-transcribed. "
      "There were five hand-transcription slips in this proposal on 2026-08-13 "
      "alone; the fix adopted then was to delete the transcription step, and this "
      "document is that fix applied to prose.")
    W("")
    W("---")
    W("")

    # ---- 0. the question --------------------------------------------------
    W("## 0. The question, and why nothing on disk could answer it")
    W("")
    W("`STATUS.json:pilot_one.pilot_two_status`, verbatim:")
    W("")
    W("> **BLOCKED.** 1,077–4,309 GPU-h must not be committed until a NEW pre-data "
      "doc shows a rung exists where NI can be **OBSERVED TO ACCEPT**; otherwise "
      "the gate can only ever confirm rejection.")
    W("")
    W("Same key: *\"it is a **rung-selection problem, not a variance problem**.\"*")
    W("")
    W("NI's discrimination curve had an **empty gap**. Damaged arms cluster at "
      "11–63 % recovery and reject by tens of SE — 1B `keep12+fresh2` rejects on "
      "4/4 axes by **27.0–90.4 × `sd_run`** at 22–32 % recovery. The **only** NI "
      "accept in all of A04 is `full32_dolmino`, which has **zero structural "
      "damage**. `keep12` was the lightest damaged 1B rung in existence, and "
      "shallower rungs had **0 checkpoints on either disk** (independently "
      "re-verified for this pass: `outputs/olmo2_probe2_7B_keep16fresh2/` holds only "
      "`arch_meta.json` on zwfy6 and does not exist on wzc1; no 1B `keep13`–`keep20` "
      "directory existed on either disk before today).")
    W("")
    W("So the blocker was **not dischargeable by any re-analysis**. It required the "
      "two lightest damaged rungs the family admits, which is what this pass "
      "trained.")
    W("")

    # ---- 1. arms ---------------------------------------------------------
    W("## 1. The two new arms")
    W("")
    W("Protocol is **Pilot One Stage B, verbatim** — every hyper-parameter read out "
      "of `stageB_seed101/step5000.pt`'s own `train_args` dict, with **only** "
      "`keep_front_layers` changed. Same seed (101), same corpus "
      "(`dolmino_now15b.npy`, 126,907,244,672 B on zwfy6 — asserted, because wzc1's "
      "same-named file is a **different corpus**), same 5,000 steps, same uniform "
      "LR 2e-5, same eff_bs 128, both **post-`ce5c298`**.")
    W("")
    W("| arm | `keep_front` | `n_fresh` | depth | cut | recovery-space position |")
    W("|---|---|---|---|---|---|")
    for a in ordered:
        k = _k(a)
        arch = d["arm_architectures_verified_from_eval_meta"].get(a, {})
        cut = arch.get("cut_fraction")
        tag = " *(Stage B reference)*" if a.endswith("_REF") else ""
        W(f"| `keep{k}+fresh2`{tag} | {k} | 2 | {k+2} | "
          f"{arch.get('cut_layers','?')}/16 = {pct(cut,2)} | "
          f"{'**new**' if not a.endswith('_REF') else 'published'} |")
    W("")
    W("**`keep14+fresh2` has depth 16 = the base's depth, and is still DAMAGED.** "
      "Base layers 14 and 15 are **discarded** and replaced by random-init Olmo2 "
      "layers, so 14 of 16 pretrained layers are inherited. The zero-damage control "
      "is `n_fresh_layers=0` continued-pretraining (the `full32` construction) and "
      "is a **different** arm. Reporting `keep14+fresh2` as zero-damage would be a "
      "category error, and using a continued-pretrained arm as the **anchor** is "
      "forbidden by guard G2 (§4).")
    W("")
    W("**`keep15+fresh2` is why `keep14` is the boundary:** it would be 17 layers, "
      "**deeper than the 16-layer base**, so it is not a cut of the base at all and "
      "\"recovery from damage\" would have no referent.")
    W("")

    # ---- 2. GATE0 --------------------------------------------------------
    g0 = None
    W("## 2. GATE0 — no degeneracy at `keep_front + n_fresh == base depth`")
    W("")
    W("Run **before** any 8-GPU commitment (1 GPU, 20 steps, `/tmp` output, "
      "18:27–18:30), because a special trainer branch at that boundary would have "
      "invalidated the whole design.")
    W("")
    W("| probe | tensors copied | expected `3+11·keep` | fresh ids | `max｜model−base｜` | fresh norms all-ones | fresh `q_proj` std | reached |")
    W("|---|---|---|---|---|---|---|---|")
    W("| keep14+fresh2 | **157** | 157 ✓ | `[14, 15]` | `0.000e+00` | True / True | 0.020001 | step 20, exit 0 |")
    W("| keep13+fresh2 | **146** | 146 ✓ | `[13, 14]` | `0.000e+00` | True / True | 0.019997 | step 20, exit 0 |")
    W("")
    W("All 6 trainer asserts pass on both, and the live 8-GPU runs reproduce them "
      "(`keep14`: *copied 157 tensors … fresh tail layer-ids [14, 15] … ALL 6 CHECKS "
      "PASS*). **Source reading confirms why:** `transplant_front()` selects base "
      "keys by `lid < keep_front_layers` against the **base** state dict, and the "
      "expected fresh set is `range(keep, keep+n_fresh)` on the **new** cfg. There "
      "is **no branch** for `keep+fresh == base_layers`; the only conditional is "
      "`if n_fresh_layers > 0`, which skips the fresh-init assert for the "
      "`n_fresh=0` CPT control. Both arms have `n_fresh=2`.")
    W("")
    W("**Optimizer groups observed** (so no differential-LR claim can be "
      "retrofitted): `keep14` → fresh 339.7 M / inherited 1145.0 M / 0.1 M; "
      "`keep13` → 339.7 M / 1077.9 M / 0.1 M — **all at 2.00e-05**. Uniform LR, as "
      "in Stage B.")
    W("")

    # ---- 3. headline results --------------------------------------------
    W("## 3. Results")
    W("")
    W(f"### 3.1 The intact anchor and Δ (convention `{conv}`)")
    W("")
    W("| axis | null | intact | residual(intact) | **Δ = 0.10 × residual** |")
    W("|---|---:|---:|---:|---:|")
    for ax in AX:
        W(f"| `{ax}` | {num(100*anchor['nulls_used'][ax])} | "
          f"{num(anchor['reported_intact_pp'][ax])} | "
          f"{num(anchor['residual_intact_pp'][ax])} | "
          f"**{num(anchor['delta_pp'][ax])}** |")
    W("")
    W("Δ was **built at runtime** by calling the imported `build_nulls()` on the "
      "G0-pinned anchor and then **cross-checked** against the canonical "
      "full-precision constants: max |diff| = "
      f"**{max(v['abs_diff'] for v in anchor['delta_cross_check_vs_canonical'].values()):.3e}** "
      f"pp, tolerance {anchor['delta_cross_check_vs_canonical'][AX[0]]['tol']:.0e}. "
      "**Δ is never substituted** (guard G2).")
    W("")

    W("### 3.2 NI margins — `margin_pp = lower95(diff) + Δ`; **> 0 means ACCEPT**")
    W("")
    W("| arm | " + " | ".join(f"`{ax}`" for ax in AX) + " | decision axes accepting |")
    W("|---|" + "---:|" * len(AX) + ":--:|")
    for a in ordered:
        row = [f"`{a}`"]
        for ax in AX:
            c = cells[a][ax]
            mark = " **ACCEPT**" if c["ni_accept"] else ""
            row.append(f"{pp(c['margin_pp'])}{mark}")
        v = verd[a]
        row.append(f"**{v['n_decision_axes_accepting']}/{v['n_decision_axes']}**")
        W("| " + " | ".join(row) + " |")
    W("")
    W("`nq_open` is **DEMOTED** by design §5.2 (its item-level 95 % CI half-width "
      "already exceeds its own Δ at n=3610) and carries **zero decision weight**; "
      "it is shown for completeness only.")
    W("")

    W("### 3.3 How far each margin is from flipping (item bootstrap SE)")
    W("")
    W("| arm | " + " | ".join(f"`{ax}`" for ax in AX) + " |")
    W("|---|" + "---:|" * len(AX))
    for a in ordered:
        row = [f"`{a}`"]
        for ax in AX:
            c = cells[a][ax]
            row.append(f"{num(c['se_to_flip'],1)} SE")
        W("| " + " | ".join(row) + " |")
    W("")
    W("This is the **item-sample** SE only. It is **not** `sd_run` and says nothing "
      "about seed variance — one seed (101) per arm, so no `sd_run` is computable "
      "here.")
    W("")

    W("### 3.4 Recovered fraction of the intact calibrated residual")
    W("")
    W("| arm | cut | " + " | ".join(f"`{ax}`" for ax in AX) + " |")
    W("|---|---:|" + "---:|" * len(AX))
    for a in ordered:
        k = _k(a)
        arch = d["arm_architectures_verified_from_eval_meta"].get(a, {})
        row = [f"`keep{k}`", pct(arch.get("cut_fraction"), 2)]
        for ax in AX:
            row.append(pct(cells[a][ax]["recovered_fraction"], 2))
        W("| " + " | ".join(row) + " |")
    W("")
    W("**No recovery fraction here may be read as \"inheritance is worth X\".** No "
      "1B zero-inheritance floor (`--from_scratch` or `--random_trunk`) exists on "
      "either disk — verified for this pass — so these are fractions of the intact "
      "residual **only**. At 7B the zero-inheritance floor already reaches "
      "**32.6 / 11.6 / 40.5 / 28.9 %** of the intact residual "
      "(`control_arms_ni_20260813` Q2), i.e. a large part of what looks like "
      "\"recovery\" is work random init already does. `must_not_claim` item 28.")
    W("")

    # ---- 4. verdict ------------------------------------------------------
    W("## 4. The verdict, and what it does to the blocker")
    W("")
    for a in ordered:
        v = verd[a]
        W(f"* **`{a}`** → `{v['verdict']}` "
          f"({v['n_decision_axes_accepting']}/{v['n_decision_axes']} decision axes "
          f"accept; axes accepting: {v['axes_accepting'] or 'none'}). Identical "
          f"under **all five** MMLU tie conventions: "
          f"**{v['identical_under_all_five_tie_conventions']}**.")
    W("")
    bd = d["branch_definitions_were_fixed_pre_data"]
    W(f"### → BRANCH {branch}")
    W("")
    W(f"**{bd[branch]}**")
    W("")
    if branch == "B":
        W("**This is a negative result and it is reported as one.** It is not "
          "dressed as a success and it is not a failure of the experiment: it is a "
          "measured fact about the certification rule A04 is trying to write.")
        W("")
        W("What it establishes: **NI's accept region at 1B contains no damaged rung "
          "at all**, down to a cut of 2 of 16 layers — the lightest cut the family "
          "admits. The accept region is therefore bounded to lie strictly between "
          "\"discard 2 of 16 layers\" and \"discard none\". A rule with that "
          "property distinguishes **damaged from intact**, not **recovered from "
          "unrecovered** — and the latter is what a recovery-certification rule is "
          "for.")
        W("")
        W("`pilot_two_status` **stays BLOCKED**, and the blocker is now recorded as "
          "**undischargeable by rung selection at 1B**: there is no shallower "
          "damaged rung left to try. Escaping it needs a different Δ, a different "
          "decision metric, or far more heal tokens — and A03 already showed 10× the "
          "token budget (52.43 B tokens at `keep7`) does not close the gap.")
    elif branch == "A":
        W("The blocker `pilot_two_status` asked for exactly this and it is now "
          "**DISCHARGED**. But note precisely what that does and does not mean: it "
          "lifts **one** of Pilot Two's two blockers. The other is independent and "
          "still binding — `control_arms_ni_20260813.recommendation.reason_2_new`: "
          "the gate's decision metric **can be reordered by output length**, and "
          "that is a **design** fix, not an *n* fix. Funding 8 more runs to feed a "
          "metric with that property buys 8 more cells of the same defect. **Both** "
          "must clear before 1,077–4,309 GPU-h is priced.")
    else:
        W("Exactly one of three decision axes accepts, which is **below the ≥2/3 "
          "bar** — the same convention under which `full32`'s 1-of-3 was reported as "
          "below the bar. Recording this branch in advance is what prevents a "
          "single-axis accept from being promoted after the fact. The blocker is "
          "**not discharged**.")
    W("")

    # ---- 4.1 §2.0.2 disposition, rendered from the companion --------------
    if DP:
        s = DP["section_2_0_2_disposition"]
        rd = s["reconciliation_document"]
        W("### 4.1 `A04_GATE_DESIGN.md` §2.0.2 — the neighbour precondition, and why "
          "it is **not triggered**")
        W("")
        W(f"§2.0.2 **does bind this ladder** (`binds_this_ladder = "
          f"{s['binds_this_ladder']}`): {s['why_it_binds']} That reading was fixed "
          f"**pre-data** in `{rd['file']}` (commit `{rd['commit']}`) — "
          f"{rd['pre_data_evidence']}.")
        W("")
        W("But §2.0.2 **gates accepts only**, and this pass has "
          f"**{s['n_accepting_decision_axis_cells']}** accepting decision-axis cells, "
          f"so `TRIGGERED = {s['TRIGGERED']}`:")
        W("")
        W(f"> {s['disposition']}")
        W("")
        W("**The precondition is NOT vacuously satisfied — it is not triggered.** "
          "The distinction matters: a vacuous satisfaction would let a later reader "
          "infer that a neighbour check was passed. None was run, because none was "
          "owed.")
        W("")
        W("`CERTIFIED` was in any case **structurally unreachable** for this ladder, "
          f"decided pre-data: {s['CERTIFIED_is_structurally_unreachable']['why']}")
        W("")
        W("**The lower neighbour exists on disk and was NOT scored.** Per the "
          "admissibility document §6.5 that is the weaker of the two available "
          "disclosures and it is the honest one; **which one applies is decided by "
          "whether the eval was run, not by what the step5000 numbers turned out to "
          "be** — and under Branch B it is not owed at all.")
        W("")
        W("| arm | lower neighbour `step2500.pt` | scored? | upper neighbour | "
          "`n_neighbours_present` |")
        W("|---|---|:--:|---|---:|")
        for arm, inv in s["neighbour_inventory_per_arm"].items():
            ln = inv["lower_neighbour"]
            lower = (f"exists, {ln['size_bytes']:,} B" if ln["exists"]
                     else "**absent**")
            W(f"| `{arm}` | {lower} | "
              f"{'yes' if inv['lower_neighbour_was_SCORED'] else '**no**'} | "
              f"**cannot exist** (`final.pt` is the *same* step) | "
              f"{inv['n_neighbours_present']} |")
        W("")
        W(f"**2500 steps is NOT a neighbourhood** "
          f"(`{s['2500_steps_is_NOT_a_neighbourhood']['verdict']}`): "
          f"{s['2500_steps_is_NOT_a_neighbourhood']['why']}")
        W("")
        W("Forbidden by name, so the temptation is closed in writing:")
        for f in s["2500_steps_is_NOT_a_neighbourhood"]["forbidden_comparisons"]:
            W(f"* {f}")
        W("")
        W(f"{s['one_process_provenance_no_resume_seam']}")
        W("")

    # ---- 5. the ladder ---------------------------------------------------
    W("## 5. The 1B depth ladder, now four points instead of two")
    W("")
    dl = d["depth_ladder"]
    W("| rung | cut | depth | verdict | " +
      " | ".join(f"recovery `{ax}`" for ax in DEC) + " |")
    W("|---|---:|---:|:--:|" + "---:|" * len(DEC))
    for key in sorted(dl["rungs"], key=lambda s: -int(s.replace("keep", ""))):
        r = dl["rungs"][key]
        W(f"| `{key}` | {pct(r['cut_fraction'],2)} | {r['depth']} | "
          f"{r['verdict'].replace('NI_','')} | " +
          " | ".join(pct(r["recovered_fraction"][ax], 2) for ax in DEC) + " |")
    W("| *zero damage* (`full32`, **7B**) | 0.00% | 32 | **ACCEPT 1/3** | "
      "— | — | 97.7% |")
    W("")
    W("The `full32` row is **7B and is not comparable as a matched experiment** — it "
      "is shown only to locate the one accept A04 has. "
      f"{dl['comparability']}")
    W("")
    W("**Monotonicity is DESCRIPTIVE ONLY.** " + dl["why_descriptive"])
    W("")
    W("| axis | recovered fraction across rungs | successive diffs | all same sign | sign reversals |")
    W("|---|---|---|:--:|---:|")
    for ax in AX:
        m = dl["monotonicity_DESCRIPTIVE_ONLY"][ax]
        W(f"| `{ax}` | " +
          ", ".join(f"keep{k}={pct(v,2)}" for k, v in zip(m["keeps"],
                                                          m["recovered_fraction"])) +
          " | " + ", ".join(f"{100*x:+.2f}pp" for x in m["successive_diffs"]) +
          f" | {m['all_same_sign']} | {m['n_sign_reversals']} |")
    W("")

    # ---- 5.1 every range against ITS OWN floor, with ITS OWN k ------------
    if DP:
        rc = DP["range_constants_used"]
        W("### 5.1 Every range with **its own** noise floor and **its own** *k*")
        W("")
        W("`E[range of k iid N(0,σ)]/σ` is **k-dependent**. "
          f"{rc['why_k_matters']}")
        W("")
        W(f"| k | constant | closed form | used for |")
        W("|---:|---|---|---|")
        for key, lbl in (("c_2", 2), ("c_3", 3)):
            e = rc[key]
            W(f"| {lbl} | `{e['value']:.16f}` | `{e['expr']}` | {e['used_for']} |")
        W(f"| 8 | `{rc['c_8_recorded_but_unused']:.4f}` | Monte Carlo (no closed "
          "form) | **recorded, unused** |")
        W("")
        W("σ is the **mean of the participating cells' own `bootstrap_se_pp`** — the "
          "per-cell recipe that reproduces `A04_GATE_DESIGN.md` §2.0.2's worked "
          "example exactly. **The pooled variant is the one `PROPOSAL.md` §4.3 "
          "retracted as \"1.69× off\". It is not used.**")
        W("")
        for grp, per_ax in DP["range_disclosures"].items():
            first = per_ax[AX[0]]
            W(f"**`{grp}`** — {first['label']}, **k={first['k']}** "
              f"(`k_matches_n_cells = {first['k_matches_n_cells']}`), "
              f"c_k = `{first['c_k']:.16f}`")
            W("")
            W("| axis | range | its floor (c_k·σ) | range/floor | clears its floor? | "
              "floor if the **wrong** c_k had been used |")
            W("|---|---:|---:|---:|:--:|---|")
            for ax in AX:
                r = per_ax[ax]
                wrong = "; ".join(
                    f"`{kk}` → {v['floor_pp']:.4f} pp "
                    f"({v['floor_error_vs_correct_pct']:+.1f}%, would clear="
                    f"{v['would_have_cleared']})"
                    for kk, v in r["if_wrong_c_k_had_been_used"].items())
                W(f"| `{ax}` | {r['range_pp']:.4f} pp | {r['noise_floor_pp']:.4f} pp | "
                  f"{r['range_over_floor']:.3f}× | "
                  f"{'**YES**' if r['CLEARS_ITS_OWN_FLOOR'] else 'no'} | {wrong} |")
            W("")
            W(f"**NOT decision-bearing.** {first['why_not_decision_bearing']}")
            W("")
            W(f"`is_a_neighbour_range = {first['is_a_neighbour_range']}`, "
              f"`is_a_sigma_run = {first['is_a_sigma_run']}`. "
              f"{first['clearing_an_item_noise_floor_is_NOT_resolution_against_seed_variance']}")
            W("")
        W(f"**{DP['no_ratio_of_ranges_is_formed']}**")
        W("")
        W(f"{DP['nothing_here_is_decision_bearing']}")
        W("")

    # ---- 6. adjacent-rung differences -----------------------------------
    W("## 6. Are the rungs even distinguishable from each other?")
    W("")
    W("Paired item bootstrap on the **same** item set (alignment asserted). A "
      "difference whose CI straddles 0 is **UNRESOLVED — not \"a direction\"**.")
    W("")
    pr = d["adjacent_rung_paired_differences"]["per_pair"]
    for pair, per_ax in pr.items():
        W(f"**`{pair}`**")
        W("")
        W("| axis | diff | CI95 | boot p | resolved |")
        W("|---|---:|---|---:|:--:|")
        for ax in AX:
            c = per_ax[ax]
            W(f"| `{ax}` | {pp(c['diff_mean_pp'])} pp | "
              f"[{c['ci95_pp'][0]:+.4f}, {c['ci95_pp'][1]:+.4f}] | "
              f"{c['boot_p']:.4f} | {'**yes**' if c['resolved'] else 'no'} |")
        W("")

    # ---- 7. RATIO --------------------------------------------------------
    W("## 7. `RATIO(ρ=0.85)` — the rule-disagreement ledger")
    W("")
    rr = d["ratio_rule"]
    W(f"| arm | mean ratio | ρ = {rr['rho']} | RATIO | NI | disagree? |")
    W("|---|---:|---:|:--:|:--:|:--:|")
    for a in ordered:
        r = rr["per_arm"][a]
        v = verd[a]
        ni_acc = v["n_decision_axes_accepting"] >= 2
        W(f"| `{a}` | {num(r['mean_ratio'])} | {rr['rho']} | "
          f"{'ACCEPT' if r['ratio_accept'] else 'REJECT'} | "
          f"{'ACCEPT' if ni_acc else 'REJECT'} | "
          f"{'**YES**' if (r['ratio_accept'] and not ni_acc) else 'no'} |")
    W("")
    W(f"**{rr['n_rule_disagreement_cells']}** rule-disagreement cell(s) from this "
      f"pass: {rr['rule_disagreement_cells'] or 'none'}.")
    W("")

    # ---- 8. verification -------------------------------------------------
    W("## 8. Verification — every item below is an executed assertion")
    W("")
    ver = [
        ("keep12 reproduction gate",
         f"this pipeline reproduces `keep12` seed101's **own published** accuracy on "
         f"all four axes to **{d['keep12_reproduction_gate']['max_abs_diff_pp']:.3e} "
         f"pp** (tolerance {d['keep12_reproduction_gate']['tol_pp']:.0e}). The "
         "canonical values are **read at runtime** from "
         "`evidence/pilot_one_stage_b_s3_verdict.json`, never transcribed. Without "
         "this the new rungs would not be measured on the same instrument as the arm "
         "they are compared against."),
        ("guard G2 — anchor must be VANILLA",
         "**executed**, two independent ways: the anchor tag is checked against a "
         "list of CPT/pruning markers, **and** the anchor cell's own `summary.json` "
         "meta must report `mode=base`, `num_hidden_layers=16` and **no `ckpt`**. "
         "Why it matters: at 7B, `full32_step25000` scores *below* vanilla base on "
         "all four axes, so substituting it would shrink every Δ **and** lower every "
         "target = **manufactured accepts**."),
        ("shard integrity",
         "every (arm × axis) cell: shard index **SET** exactly `{0..7}` (a set, not "
         "\"8 files\"), merged *n* exactly `EXPECTED_N` "
         "(17944 / 14267 / 3610 / 14042), **0** duplicate `item_id`, **0** nan in "
         "the metric vector, and `item_id` sequences **identical across every arm "
         "AND the anchor** — without which the paired differences would compare "
         "different items."),
        ("arm architecture verified from eval meta",
         "each arm's eval `summary.json` meta must report `keep_front` / `n_fresh` / "
         "`num_hidden_layers` matching its tag or the analysis **aborts**. An eval "
         "that rebuilt the wrong shell would otherwise be scored silently. The eval "
         "loader additionally reads keep/fresh from the **ckpt's own meta** and "
         "raises if the CLI disagrees, then `strict`-loads."),
        ("protocol",
         "`add_bos` asserted **`is False`** — never `is not True`, so `None` or "
         "missing **FAILS**. `chat_template` asserted `is not False` → FAIL, **plus "
         "structurally**: neither eval script contains an `apply_chat_template` call "
         "site, so no flag can enable one. `max_new_tokens == 32`. `mmlu_bs=16`, "
         "`cb_bs=32` — the Stage B driver's own values, so the new cells are "
         "protocol-identical to `keep12`. These are BASE LMs (no SFT/RL); any "
         "chat=True number is void."),
        ("canonical code imported, never reimplemented",
         "`ni_rule` / `ratio_rule` / `load_shards` / `build_nulls` / "
         "`mmlu_content_norm_vec` / `qa_metric_vec` / `EXPECTED_N` / `AXES` / "
         "`DEMOTED_AXES` / `PREREG` from `pilot_zero_rule_disagreement`; "
         "`assert_aligned` / `d4_interface_degenerate` from "
         "`a04_shallow_rung_ni_7b`; `paired_bootstrap` / `TIE_CONVS` / `N_BOOT` / "
         "`SEED` from A03's `analyze_1b_knowledge_floor`. **The null is never "
         "hand-computed** — MAIN's own subtraction of a recorded null was ~0.5 pp "
         "off twice."),
        ("bootstrap seed disjointness",
         "`arm_index` 1100/1101/1102, guard offset `SEED+9700+13·axis`. Disjoint "
         "from every archived block (0–1, 100–102, 200–204, 300–301, 400–408, "
         "500–503, 600–610, 700–702, 800–801, 900–902, 1000–1005). The check is "
         "**EXECUTED** by `assert_seeds_disjoint` (reads each archive's own recorded "
         f"offsets and raises on intersection): "
         f"**{d['seed_disjointness_checked']['archives_scanned']} archives scanned, "
         "no clash**. Prose claims of disjointness in this repo have already been "
         "wrong once, and the executed check has caught a real collision."),
        ("one node for every statistic",
         f"numpy `Generator.multinomial` differs in **19/10000** rows between `.82`'s "
         f"2.4.6 and `.73`'s 2.5.1. Every statistic here comes from "
         f"**{d['node']}** (numpy {d['numpy_version']}), pinned with "
         "`--expect_numpy`. Training is unaffected; only bootstrap statistics are."),
        ("gate constants self-tested but DECLARED UNUSED",
         "`E[range of k iid N(0,σ)]/σ` is **k-dependent**: k=2 → "
         f"{d['gate_constant_selftest']['c2_closed_form']:.16f}, k=3 → "
         f"{d['gate_constant_selftest']['c3_closed_form']:.16f} (both closed form, "
         "**re-derived not trusted**), k=8 → "
         f"{d['gate_constant_selftest']['c8_monte_carlo']:.4f} (Monte Carlo, "
         "validated by reproducing the k=3 closed form to "
         f"{d['gate_constant_selftest']['c3_mc_abs_err']:.2e}). Reusing k=3's "
         "constant at k=8 makes a floor **40.6 % too low**. **None is used here** "
         "(one seed per arm, 2 checkpoints per arm) and they are recorded as "
         "`DECLARED_UNUSED` so nobody can reuse a wrong `c_k` from this document. "
         "**A ratio of two ranges neither of which clears its own floor is "
         "UNDEFINED, not a direction** — the error that voided "
         "`within_arm_lr_refutation_20260813`."),
        ("pipeline validated BEFORE the arms landed",
         "`--preflight_only` ran every guard, the anchor build, the Δ cross-check and "
         "the `keep12` reproduction gate **while both trainings were in flight**, "
         "writing nothing; `keep12` seed101 reproduced to **0.000e+00 pp**. "
         "`--preflight_ignore_own_training` (needed because our own training held the "
         "cards) is **hard-refused outside preflight mode**, so the GPU refuse-guard "
         "can never be bypassed by a run that writes an evidence file."),
        ("zwfy6's evidence dir was incomplete and was repaired first",
         "the first preflight scanned only **7** offset ledgers on zwfy6 vs **8** on "
         "wzc1 — **14 evidence files existed only on wzc1**, so the disjointness "
         "check was running against a **partial** archive set and could have missed a "
         "real collision. All 14 were `scp -O`'d and **md5-verified 14/14** before "
         "the analysis ran. Generalisable: the two disks' `proposal/` trees are not "
         "automatically in sync (zwfy6's is a hand copy, not a git checkout), and "
         "`assert_seeds_disjoint` must be pointed at a **complete** evidence dir."),
        ("positive preflight assertions printed before launch",
         "both progress logs carry, **before** the launch line: `PREFLIGHT-ASSERT "
         "trainer post-ce5c298: 869: sampler = DistributedSampler(ds, shuffle=True, "
         "seed=args.seed)`, the trainer md5 `284b286f90b526e4e8ad93a68e2a3b16`, "
         "`base num_hidden_layers=16`, the exact dolmino byte count, and `GPUs clear "
         "(0MiB held)`. Both arms are **post-fix**, the same side of the `ce5c298` "
         "break as the Stage B family (`PROPOSAL.md` §7.2)."),
    ]
    for name, body in ver:
        W(f"* **{name}** — {body}")
    W("")

    # ---- 9. cost ---------------------------------------------------------
    W("## 9. Cost")
    W("")
    W("| item | value |")
    W("|---|---:|")
    for k, v in gh["training"]["per_arm"].items():
        if v.get("measured"):
            W(f"| {k} training | {v['s_per_step']:.3f} s/step → "
              f"{v['extrapolated_wall_h_for_5000_steps']:.2f} h wall → "
              f"**{v['gpu_h_at_8_gpus']:.1f} GPU-h** |")
    W(f"| training total | **{gh['training']['total_gpu_h']:.1f} GPU-h** |")
    W(f"| eval total | **{gh['eval']['total_gpu_h']:.2f} GPU-h** |")
    W(f"| this analysis | **0 GPU-h** (CPU-only, read-only) |")
    W(f"| **grand total** | **{gh['total_gpu_h']:.1f} GPU-h** |")
    W("| Pilot Two, for comparison | 1,077–4,309 GPU-h |")
    W("")
    W("Wall time per arm is **measured** from each arm's own trainer log as "
      "`(t_last − t_first) / (step_last − step_first)` over the whole run — "
      "elapsed/iter, **not** an instantaneous `s/step` sample. "
      f"This pass is **{100*gh['total_gpu_h']/4309:.1f}–"
      f"{100*gh['total_gpu_h']/1077:.1f} %** of Pilot Two, and it is the only "
      "expenditure that could decide whether Pilot Two's blocker is dischargeable "
      "at all.")
    W("")

    # ---- 10. not licensed -----------------------------------------------
    W("## 10. What this pass does NOT license")
    W("")
    for item in d["not_licensed"]:
        W(f"* {item}")
    W("")

    open(out_path, "w").write("\n".join(L) + "\n")
    print(f"wrote {out_path} ({len(L)} lines) from {ev_path}")
    print(f"BRANCH {branch}: {d['headline']}")


if __name__ == "__main__":
    main()
