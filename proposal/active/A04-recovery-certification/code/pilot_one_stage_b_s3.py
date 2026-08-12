#!/usr/bin/env python
"""A04 Pilot One, STAGE B — the S=3 `sd_run` at keep12+fresh2 and the K2 decision.

Costs ZERO additional GPU. The 135 GPU-h were spent on 2026-08-11: seeds
{101,102,103} each trained keep12+fresh2 from the OLMo-2-0425-1B base for 5,000
steps on zwfy6, and all three were scored on all four axes. This script HARVESTS
those scores and applies the rule pre-registered in `PILOT_ONE_PREREG.md` §2.2 /
§3, committed as `2ac0b5a` BEFORE any keep12 number existed.

WHY THIS SCRIPT EXISTS WHEN A VERDICT JSON ALREADY DID
------------------------------------------------------
A Stage-B S=3 verdict was written on zwfy6 at 2026-08-11 14:53
(`evidence/stageB_S3_verdict.json`, md5 7145d569f46ec0fa10dd56368071adf2). It was
never harvested to wzc1, was never committed, had no driver in `code/`, and has
three defects this script fixes:

  1. It is MISLABELLED `"gate": "A04 Pilot One Stage A"` with verdict string
     `STAGE_A_DOES_NOT_FIRE`. It is Stage B. A future agent grepping for the
     Stage-B verdict would have found a file claiming to be Stage A.
  2. It records NO integrity counts. `PILOT_ONE_PREREG.md` §4 requires asserting
     `n_shards == 8`, shard indices exactly {0..7}, AND the exact item counts
     (MMLU 14,042 / TriviaQA 17,944 / PopQA 14,267 / NQ-open 3,610) per cell.
     The verdict JSON carries none of these, so nothing in it is auditable.
  3. ★ The exact-item-count assertion was NEVER ACTUALLY RUN on the closed-book
     axes. `canonical_eval_loaders.load_cb` asserts only 8/8 shard FILES; unlike
     `load_mmlu` (which asserts `len(got) == 14042`) it has no count check and no
     duplicate-`item_id` check -- it merges into a dict, so an overlapping shard
     range would be silently absorbed rather than caught. So for triviaqa, popqa
     and nq_open the prereg's headline integrity requirement was, until this
     script, unenforced. This script enforces it OUTSIDE the loader (the loader is
     shared with A03's archived numbers and is deliberately not edited here).

The loaders themselves are IMPORTED, never reimplemented -- same rule, and same
reason, as `pilot_one_stage_a_sd_run.py`: a hand-written `load_mmlu` that guessed
flat key names silently dropped a whole axis from 12 cells on 2026-08-10.

THE ESTIMATOR CHANGES FROM STAGE A; THE RULE DOES NOT
----------------------------------------------------
Stage A had n=2 and used the unbiased-from-a-range estimator
`sd_run = |m_a - m_b| / sqrt(2)`. Stage B has S=3, so the prereg §3 says
"recompute `sd_run` from S=3 (proper `sd`, df=2, `t=2.920`) and apply the §2.2
rule unchanged". Hence:

    sd_run(x) = sample sd over the 3 seed means, ddof=1   (df = 2)
    bound_3(x) = t_{0.05,2} * sd_run(x) / sqrt(3) = 2.920 * sd_run(x) / 1.7321

    K2 FIRES iff bound_3(x) > Delta_x on >= 2 of the 3 DECISION axes.
    Exactly 1 of 3 -> K2_INDETERMINATE (prereg §2.4).
    0 of 3         -> K2 does not fire.

Note the estimator is CONTINUOUS at the n=2 -> n=3 change: for n=2 the sample sd
with ddof=1 *is* |a-b|/sqrt(2). Stage A's formula was the S=2 special case of this
one, not a different estimator.

CAN STAGE B CLEAR K2? -- read `k2_clearance_analysis` in the output
------------------------------------------------------------------
Stage A could NEVER emit K2_CLEARED, for three stated reasons (prereg §2.3):
wrong arm (keep7 not keep12), wrong budget (20k not 5k steps), and n=2 gives a
range not a variance. Stage B repairs the first two by construction -- it IS
keep12+fresh2 at 5,000 steps, the arm and budget the gate would use. The third is
only PARTIALLY repaired: S=3 gives df=2, a real variance but with a chi-square 95%
interval for sigma spanning a multiplicative factor of ~8.7x. This script
therefore reports the point-estimate verdict AND the verdict at the pessimistic
(chi-square 97.5th-percentile) end of each sigma interval, and does not
unilaterally promote "does not fire" into "cleared". See the emitted
`k2_clearance_analysis` block for the two readings stated side by side.

NQ-open is DEMOTED per design §5.2 and carries NO decision weight.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

# --- canonical loaders: a NORMAL import from proposal/shared/code -------------
# Same contract as pilot_one_stage_a_sd_run.py. Failure is LOUD, never a fallback
# to a locally-derived loader -- a silent fallback is the shape the 2026-08-10
# MMLU defect took.
_SHARED_CODE = Path(__file__).resolve().parents[3] / "shared" / "code"
_LOADER_MODULE = _SHARED_CODE / "canonical_eval_loaders.py"
if not _LOADER_MODULE.is_file():
    print(f"[stageB][FATAL] canonical loader module not found: {_LOADER_MODULE}",
          file=sys.stderr)
    sys.exit(2)

sys.path.insert(0, str(_SHARED_CODE))
try:
    from canonical_eval_loaders import (  # noqa: E402
        CB as _CB_ROOT,
        MM as _MM_ROOT,
        ROOT as _RESULTS_ROOT,
        NotRunYet,
        load_cb,
        load_mmlu,
    )
except Exception as _e:
    print(f"[stageB][FATAL] cannot import canonical loaders from "
          f"{_LOADER_MODULE}: {_e!r}", file=sys.stderr)
    sys.exit(2)

# Stale-copy guard, carried over verbatim in intent from Stage A: zwfy6's
# `proposal/` tree is a hand-copied directory, NOT a git checkout
# (`git ls-files proposal/` returns 0 there), so it does not receive commits and a
# stale loader copy on that disk remains possible. Assert the FIX is present
# (positive test) rather than trying to enumerate its absences.
_FIX_MARKER = 'for iface in ("letter", "content_norm")'
if _FIX_MARKER not in _LOADER_MODULE.read_text():
    print("[stageB][FATAL] the canonical loader module on this disk lacks the "
          f"fixed nested-key `load_mmlu` (marker not found: {_FIX_MARKER!r}). "
          "The pre-fix version yields (None, None) for every MMLU item and "
          "silently drops the whole axis. Copy the fixed module from wzc1.",
          file=sys.stderr)
    sys.exit(2)

# --- pre-registered constants. No CLI override, by design. -------------------
# Identical values to pilot_one_stage_a_sd_run.py; verified against
# evidence/pilot_zero_rule_disagreement.json (per_convention.split.cells[*])
# on 2026-08-10, before any keep12 datum existed.
DELTA_PP = {
    "triviaqa": 4.043134195274186,
    "popqa": 1.3205298941613512,
    "mmlu_content": 1.0238926078906136,
    "nq_open": 0.9695290858725762,   # DEMOTED, descriptive only
}
DECISION_AXES = ("triviaqa", "popqa", "mmlu_content")
DEMOTED_AXES = ("nq_open",)

# Exact expected item counts, prereg §4. These are the numbers `load_cb` does NOT
# check; this script does.
EXPECTED_N = {
    "triviaqa": 17944,
    "popqa": 14267,
    "mmlu_content": 14042,
    "nq_open": 3610,
}

T_CRIT_S3_DF2 = 2.920          # t_{0.05, df=2}, one-sided. Design §5.3 table.
S_TARGET = 3
K2_FIRE_MIN_AXES = 2           # ">= 2 of the 3 decision axes"
PREREG_SEEDS = (101, 102, 103)  # pinned in PILOT_ONE_PREREG.md §3, PRE-DATA

# chi-square quantiles at df=2, for the interval on sigma given a sample sd.
# sigma in [ s*sqrt(df/chi2_hi), s*sqrt(df/chi2_lo) ].
CHI2_DF2_P975 = 7.377758908227871
CHI2_DF2_P025 = 0.05063562059591586


def _axis_dirname(seed: int, axis: str, template: str) -> str:
    """On-disk result dirname. The `_nq` suffix matches the harness convention."""
    d = template.format(seed=seed)
    return d + "_nq" if axis == "nq_open" else d


def _shard_integrity(seed: int, axis: str, template: str) -> dict:
    """Assert shard files 0..7 exist for this cell, INDEPENDENT of the loader.

    The loader checks `len(files) == 8` after a glob. That passes on a set of
    eight files whose indices are, say, {0,1,2,3,4,5,6,6} -- a duplicated shard
    with one missing. Check the index SET explicitly.
    """
    d = _axis_dirname(seed, axis, template)
    if axis == "mmlu_content":
        root, stem = _MM_ROOT, "per_example_mmlu"
    else:
        root, stem = _CB_ROOT, f"per_example_{axis}"
    dirp = root / d
    found, missing = [], []
    for i in range(8):
        p = dirp / f"{stem}_shard{i}of8.jsonl"
        (found if p.is_file() else missing).append(i)
    return {
        "dir": str(dirp),
        "shard_indices_present": found,
        "shard_indices_missing": missing,
        "n_shards_present": len(found),
        "shards_complete": (len(found) == 8 and not missing),
    }


def axis_seed_measure(seed: int, axis: str, template: str):
    """(mean_pct, integrity_dict) or (None, integrity_dict_with_error).

    Closed-book axes use `em` (the headline A03/A04 metric). MMLU uses
    `content_norm` -- the letter interface is BANNED as a decision axis by design
    §4.2 and is never read here.
    """
    integ = _shard_integrity(seed, axis, template)
    d = _axis_dirname(seed, axis, template)
    if not integ["shards_complete"]:
        integ["error"] = (f"INTEGRITY_FAILURE: shard set incomplete, missing "
                          f"{integ['shard_indices_missing']} -- refusing to merge")
        return None, integ
    try:
        if axis == "mmlu_content":
            got = load_mmlu(d)
            vals = [float(v[1]) for v in got.values()]
        else:
            got = load_cb(d, axis)
            vals = [float(v[0]) for v in got.values()]
    except NotRunYet as e:
        integ["error"] = f"NOT_RUN_YET: {e}"
        return None, integ
    except SystemExit as e:
        integ["error"] = f"INTEGRITY_FAILURE: {e}"
        return None, integ

    integ["n_scored"] = len(got)
    integ["n_expected"] = EXPECTED_N[axis]
    integ["n_matches_expected"] = (len(got) == EXPECTED_N[axis])
    # NaN accounting. load_mmlu hard-fails on nan:true rows, so a returned MMLU
    # dict is nan-free by construction. The CB harness writes no nan field; a
    # non-finite metric would show up here.
    n_nan = sum(1 for v in vals if not math.isfinite(v))
    integ["n_nan"] = n_nan
    integ["nan_is_zero"] = (n_nan == 0)

    if not integ["n_matches_expected"]:
        integ["error"] = (f"INTEGRITY_FAILURE: merged {len(got)} items, expected "
                          f"{EXPECTED_N[axis]} -- incomplete or wrong dump "
                          f"(load_cb does NOT check this; this script does)")
        return None, integ
    if n_nan:
        integ["error"] = f"INTEGRITY_FAILURE: {n_nan} non-finite metric values"
        return None, integ
    if not vals:
        integ["error"] = "loader returned zero items"
        return None, integ
    return 100.0 * sum(vals) / len(vals), integ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_template",
                    default="A04_1B_stageB_keep12_seed{seed}_step5000",
                    help="result dirname template; {seed} substituted. `_nq` is "
                         "appended automatically for the nq_open axis.")
    ap.add_argument("--out_json", required=True)
    a = ap.parse_args()

    axes = list(DECISION_AXES) + list(DEMOTED_AXES)
    per_axis, problems, integrity = {}, {}, {}

    for axis in axes:
        means, errs = {}, {}
        integrity[axis] = {}
        for seed in PREREG_SEEDS:
            m, integ = axis_seed_measure(seed, axis, a.dir_template)
            integrity[axis][str(seed)] = integ
            if m is None:
                errs[str(seed)] = integ.get("error", "unknown")
            else:
                means[seed] = m

        if len(means) < S_TARGET:
            problems[axis] = errs
            per_axis[axis] = {"status": "PENDING", "errors": errs,
                              "n_seeds_ok": len(means),
                              "decision_weight": axis in DECISION_AXES}
            continue

        vals = [means[s] for s in PREREG_SEEDS]
        # S=3 -> proper sample sd, ddof=1, df=2 (prereg §3). Continuous with
        # Stage A: for n=2 this equals |a-b|/sqrt(2).
        sd_run = statistics.stdev(vals)
        bound_3 = T_CRIT_S3_DF2 * sd_run / math.sqrt(S_TARGET)
        delta = DELTA_PP[axis]

        # chi-square 95% interval on sigma at df=2, and the bound at each end.
        df = S_TARGET - 1
        sigma_lo = sd_run * math.sqrt(df / CHI2_DF2_P975)
        sigma_hi = sd_run * math.sqrt(df / CHI2_DF2_P025)
        bound_lo = T_CRIT_S3_DF2 * sigma_lo / math.sqrt(S_TARGET)
        bound_hi = T_CRIT_S3_DF2 * sigma_hi / math.sqrt(S_TARGET)

        per_axis[axis] = {
            "status": "OK",
            "seed_means_pct": {str(s): means[s] for s in PREREG_SEEDS},
            "mean_of_seed_means_pct": statistics.fmean(vals),
            "range_pp": max(vals) - min(vals),
            "sd_run_pp": sd_run,
            "df": df,
            "bound_S3_pp": bound_3,
            "delta_pp": delta,
            "bound_exceeds_delta": bool(bound_3 > delta),
            "margin_ratio_delta_over_bound": (delta / bound_3) if bound_3 else None,
            "sigma_chi2_95ci_pp": [sigma_lo, sigma_hi],
            "bound_at_sigma_ci_lo_pp": bound_lo,
            "bound_at_sigma_ci_hi_pp": bound_hi,
            "would_fire_at_sigma_ci_hi": bool(bound_hi > delta),
            "decision_weight": axis in DECISION_AXES,
        }

    ready = [x for x in DECISION_AXES if per_axis[x]["status"] == "OK"]
    exceed = [x for x in ready if per_axis[x]["bound_exceeds_delta"]]
    exceed_pess = [x for x in ready if per_axis[x]["would_fire_at_sigma_ci_hi"]]

    if len(ready) < len(DECISION_AXES):
        verdict = "PENDING_INCOMPLETE_AXES"
        missing = [x for x in DECISION_AXES if x not in ready]
        rationale = (f"only {len(ready)}/{len(DECISION_AXES)} decision axes "
                     f"available; the pre-registered rule needs all three. "
                     f"Missing: {missing}")
    elif len(exceed) >= K2_FIRE_MIN_AXES:
        verdict = "K2_FIRES"
        rationale = (f"bound_3 > Delta on {len(exceed)}/{len(DECISION_AXES)} "
                     f"decision axes ({exceed}), meeting the pre-registered "
                     f"threshold of {K2_FIRE_MIN_AXES}. A04 dies here, for the "
                     f"~135 GPU-h Stage B cost instead of the ~2,900 GPU-h full "
                     f"gate. That is the gate design working.")
    elif len(exceed) == 1:
        verdict = "K2_INDETERMINATE_AT_STAGE_B"
        rationale = (f"bound_3 > Delta on exactly 1 axis ({exceed}). "
                     f"Pre-registered (§2.4) as neither a fire nor a pass.")
    else:
        verdict = "K2_DOES_NOT_FIRE_AT_STAGE_B"
        rationale = ("bound_3 <= Delta on all 3 decision axes at the S=3 point "
                     "estimate. See k2_clearance_analysis before reading this as "
                     "a clearance.")

    payload = {
        "gate": "A04_pilot_one_stage_B_S3_sd_run",
        "stage": "B",
        "S": S_TARGET,
        "seeds": list(PREREG_SEEDS),
        "arm": "keep12+fresh2 (--keep_front_layers 12 --n_fresh_layers 2), "
               "OLMo-2-0425-1B base, 5,000 steps, dolmino, 8xH20, zwfy6",
        "gpu_h_spent": 135,
        "gpu_h_additional_for_this_analysis": 0,
        "prereg": {
            "doc": "proposal/active/A04-recovery-certification/PILOT_ONE_PREREG.md",
            "commit": "2ac0b5a",
            "committed_before_data": True,
            "estimator": "sd_run = sample sd over the S=3 seed means (ddof=1, "
                         "df=2). Prereg §3: 'recompute sd_run from S=3 (proper "
                         "sd, df=2, t=2.920)'. Continuous with Stage A's "
                         "|a-b|/sqrt(2), which is the n=2 case of this.",
            "bound": f"t_(0.05,2)={T_CRIT_S3_DF2} * sd_run / sqrt({S_TARGET})",
            "rule": f"K2 FIRES iff bound_3 > Delta on >= {K2_FIRE_MIN_AXES} of "
                    f"{len(DECISION_AXES)} decision axes; exactly 1 -> "
                    f"INDETERMINATE (§2.4)",
            "rule_unchanged_from_stage_a": True,
            "delta_pp": DELTA_PP,
            "decision_axes": list(DECISION_AXES),
            "demoted_axes": list(DEMOTED_AXES),
            "seeds_pinned_pre_data": list(PREREG_SEEDS),
            "no_post_hoc_seed_addition": "prereg §3 forbids S=4 (t=2.353) or S=5 "
                                         "(t=2.132) chosen after seeing S=3.",
        },
        "verdict": verdict,
        "rationale": rationale,
        "n_decision_axes_exceeding": len(exceed),
        "axes_exceeding": exceed,
        "n_decision_axes_exceeding_at_sigma_ci_hi": len(exceed_pess),
        "axes_exceeding_at_sigma_ci_hi": exceed_pess,
        "per_axis": per_axis,
        "integrity": integrity,
        "k2_clearance_analysis": {
            "stage_a_could_never_clear": [
                "wrong arm: keep7+fresh2 (56.2% depth), not keep12+fresh2 (87.5%)",
                "wrong budget: 20,000 steps, not the 5,000 Pilot One would run",
                "n=2 yields a range, not a variance with a usable CI",
            ],
            "which_of_those_stage_b_repairs": {
                "wrong_arm": "REPAIRED -- Stage B IS keep12+fresh2 at 87.5% depth, "
                             "the arm the gate would use.",
                "wrong_budget": "REPAIRED -- Stage B IS 5,000 steps.",
                "n2_is_a_range_not_a_variance": "PARTIALLY repaired only. S=3 "
                    "gives a real sample sd but df=2, whose chi-square 95% "
                    "interval for sigma spans a multiplicative factor of "
                    f"{math.sqrt(CHI2_DF2_P975 / CHI2_DF2_P025):.1f}x. A point "
                    "estimate under Delta is therefore not the same as sigma "
                    "being under Delta.",
            },
            "reading_1_point_estimate": (
                f"K2 does not fire: 0 of 3 decision axes exceed Delta at the S=3 "
                f"point estimate." if not exceed else
                f"K2 fires at the point estimate on {exceed}."),
            "reading_2_pessimistic_end_of_sigma_ci": (
                f"At the chi-square 97.5th-percentile end of each df=2 sigma "
                f"interval, {len(exceed_pess)} of 3 decision axes would exceed "
                f"Delta ({exceed_pess}). The K2 rule needs >= "
                f"{K2_FIRE_MIN_AXES}, so this "
                + ("WOULD ALSO FIRE" if len(exceed_pess) >= K2_FIRE_MIN_AXES
                   else "does NOT reach a fire")
                + "."),
            "honest_summary": "Stage B removes Stage A's arm and budget "
                              "objections but not its d.o.f. objection. The "
                              "prereg does not define a Stage-B 'K2_CLEARED' "
                              "state, so this script does not invent one; the "
                              "terminal value it emits is the rule's own output.",
            "how_to_close_it": "d.o.f. on the KEEP12 family specifically. A03's "
                               "seed-45 keep7 draw does not help: K2's estimator "
                               "is the keep12 family's own sd_run. Substituting a "
                               "pooled keep7+keep12 sigma after seeing which "
                               "answer each gives is a post-hoc estimator change "
                               "and is not licensed.",
        },
        "provenance": {
            "loaders": "load_cb / load_mmlu IMPORTED from "
                       "proposal/shared/code/canonical_eval_loaders.py -- never "
                       "reimplemented.",
            "loader_module": str(_LOADER_MODULE),
            "results_root": str(_RESULTS_ROOT),
            "dir_template": a.dir_template,
            "mmlu_interface": "content_norm (the letter interface is BANNED as a "
                              "decision axis by A04 design §4.2)",
            "integrity_beyond_the_loader": "load_cb asserts only 8/8 shard FILES: "
                "it has NO exact-item-count check and NO duplicate-item_id check "
                "(it merges into a dict). So prereg §4's exact counts "
                "(triviaqa 17,944 / popqa 14,267 / nq_open 3,610) were unenforced "
                "for the closed-book axes until this script. Enforced here, "
                "outside the loader, because the loader is shared with A03's "
                "archived numbers and is not edited by this harvest.",
            "shard_index_set_checked": "explicitly {0..7}, not merely a glob "
                "count of 8 -- a duplicated shard with one missing passes a count "
                "check.",
            "add_bos": "false in every cell's summary.json meta (prereg §4, "
                       "protocol_invariants: BASE LM, no chat template, no BOS)",
            "chat_template": "False -- the closed-book and mmlu_content harnesses "
                             "have no chat path at all; they build raw prompts.",
            "manipulation_check": "training-loss series across the 3 seeds, tail "
                                 "window step>=1000, detrended per-seed by its own "
                                 "rolling median (w=9): r = +0.0089 (101,102), "
                                 "+0.0496 (101,103), +0.0118 (102,103). Compare "
                                 "A03's phase-locked pair +0.99966 and its "
                                 "independent pair -0.0101. Data order genuinely "
                                 "varies; the DistributedSampler(seed=) fix "
                                 "(ce5c298) is active. RAW (untrended) tail r is "
                                 "+0.944 to +0.948, which is the shared 5k-step "
                                 "loss DECAY, not phase-lock -- Stage A's A03 runs "
                                 "were resumed and flat, so raw r was already "
                                 "interpretable there; here it is not.",
            "optim_groups_per_run": "fresh_decay 339.7M @2e-5 / fresh_nodecay "
                                    "0.0M @2e-5 / inh_decay 1010.8M @2e-5 / "
                                    "inh_nodecay 0.1M @2e-5, torch AdamW fp32. "
                                    "All four groups at the SAME 2e-5, so the run "
                                    "is UNIFORM-LR. Differential LR must NOT be "
                                    "claimed. Unlike the distill trainer's "
                                    "_classify_param defect, the fresh groups DO "
                                    "exist here (339.7M is classified), so this is "
                                    "a config choice (--lr 2e-5 == --lr_inherited "
                                    "2e-5), not the silent no-op bug.",
        },
        "problems": problems,
    }

    Path(os.path.dirname(a.out_json) or ".").mkdir(parents=True, exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[stageB] VERDICT: {verdict}")
    print(f"[stageB] {rationale}")
    for axis in axes:
        r = per_axis[axis]
        tag = "" if r.get("decision_weight") else "   (DEMOTED, descriptive)"
        if r["status"] != "OK":
            print(f"  {axis:14s} PENDING {r.get('errors')}{tag}")
            continue
        mark = ">" if r["bound_exceeds_delta"] else "<="
        ms = "/".join(f"{r['seed_means_pct'][str(s)]:.4f}" for s in PREREG_SEEDS)
        print(f"  {axis:14s} means={ms}  sd_run={r['sd_run_pp']:.4f}pp  "
              f"bound_3={r['bound_S3_pp']:.4f}pp {mark} "
              f"Delta={r['delta_pp']:.4f}pp  "
              f"[sigma CI hi -> bound {r['bound_at_sigma_ci_hi_pp']:.4f}pp "
              f"{'FIRES' if r['would_fire_at_sigma_ci_hi'] else 'no fire'}]{tag}")
    print("[stageB] integrity:")
    for axis in axes:
        for seed in PREREG_SEEDS:
            i = integrity[axis][str(seed)]
            ok = i.get("shards_complete") and i.get("n_matches_expected") \
                and i.get("nan_is_zero")
            print(f"  {axis:14s} seed{seed} shards={i.get('n_shards_present')}/8 "
                  f"n_scored={i.get('n_scored')}/{i.get('n_expected')} "
                  f"nan={i.get('n_nan')} -> {'OK' if ok else 'FAIL ' + str(i.get('error'))}")
    print(f"[stageB] wrote {a.out_json}")


if __name__ == "__main__":
    main()
