#!/usr/bin/env python
"""A04 Pilot One, STAGE A — the free `sd_run` bound and the K2 decision.

Costs ZERO GPU. Reads the A03 data-order checkpoints' already-scored per-example
shards (seeds 43 and 44, task #236) and applies the rule pre-registered in
`PILOT_ONE_PREREG.md` §2.2, committed as `2ac0b5a` BEFORE any of those numbers
existed.

WHAT THIS DECIDES
-----------------
K2 ("disagreement drowned by seed variance") is the kill clause A04's own design
calls "the most likely killer". It needs `sd_run`, and no `sd_run` exists anywhere
in this repo -- every pre-`ce5c298` "seed" varied fresh-block INIT only, because
`DistributedSampler(ds, shuffle=True)` was called without `seed=`, so every seed
consumed a byte-identical minibatch sequence. A03's seeds 43/44 are the first true
run-to-run draws (verified pre-data: arm3-vs-seed43 training-loss r = -0.0101 vs
arm3-vs-arm6 +0.99966).

LOADERS ARE IMPORTED, NEVER REIMPLEMENTED
-----------------------------------------
`load_cb` and `load_mmlu` come from A03's `recompute_cpt_trajectory_paired.py`.
This is not merely a style rule here: on 2026-08-10 a hand-written `load_mmlu`
that GUESSED flat key names (`letter_correct`, `content_norm_correct`) against a
NESTED record silently returned None for every MMLU cell, dropping a whole axis
from 12 trajectory cells while four .md files went on asserting "MMLU is flat".
Re-deriving a loader for this script would reopen exactly that defect. The
imported versions also carry the shard-completeness assertions (8/8, exact item
counts, duplicate-item_id and NaN checks) that a silent 5-of-8 merge has
previously defeated.

THE ESTIMATOR AND THE RULE, BOTH FIXED BEFORE THE DATA
------------------------------------------------------
    sd_run(x) = |m_43(x) - m_44(x)| / sqrt(2)            # unbiased sd from n=2
    bound_3(x) = t_{0.05,2} * sd_run(x) / sqrt(3)
               = 2.920 * sd_run(x) / 1.7321

    K2 FIRES iff bound_3(x) > Delta_x on >= 2 of the 3 DECISION axes.
    Exactly 1 of 3  -> K2_INDETERMINATE_AT_STAGE_A (not a fire, not a pass).
    0 of 3          -> Stage A does not fire; proceed to Stage B.

ONE-DIRECTIONAL BY CONSTRUCTION -- the load-bearing caveat
----------------------------------------------------------
A large sd_run KILLS: run-to-run noise here is a property of the shared 1B /
dolmino / 8xH20 apparatus that Pilot One would inherit, so if noise already swamps
Delta there is no case for spending 135 GPU-h to re-measure it.

A small sd_run does NOT clear K2, for three reasons this script refuses to let a
caller forget: (i) wrong arm -- keep7+fresh2 at 56.2% depth, not the recommended
keep12+fresh2 at 87.5%; (ii) wrong budget -- 20,000 steps, not 5,000; (iii) n=2 is
a RANGE, not a variance with a usable CI. The script therefore never emits
"K2_CLEARED"; the best available Stage-A outcome is "STAGE_A_DOES_NOT_FIRE".

NQ-open is DEMOTED per design 5.2 (its own item-level CI half-width 1.459-2.063 pp
at n=3610 exceeds its Delta of 0.970 pp) and carries NO decision weight. It is
computed and reported descriptively only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# --- import A03's canonical loaders by path (its dir is not a package) --------
#
# ⚠️ A03's module has NO `if __name__ == "__main__":` guard -- its whole trajectory
# analysis runs at module scope. A naive exec_module() therefore re-runs that
# entire analysis (printing dozens of cell lines, re-reading every arm, and
# potentially calling sys.exit) just to obtain two functions. Worse, its
# module-level code can raise SystemExit on an unrelated arm's partial shards,
# which would abort THIS script for a reason having nothing to do with Stage A.
#
# So: exec the module source with its trailing module-level driver stripped. We
# keep everything up to the first top-level statement after the function/class
# definitions, which is exactly the loader + helper block. If the file's shape
# ever changes such that the loaders are not found, this fails loudly rather than
# silently importing a partial module.
_A03_CODE = (Path(__file__).resolve().parent.parent.parent
             / "A03-parametric-vs-external-memory" / "code"
             / "recompute_cpt_trajectory_paired.py")
if not _A03_CODE.is_file():
    print(f"[stageA][FATAL] canonical loader module not found: {_A03_CODE}",
          file=sys.stderr)
    sys.exit(2)

_SRC = _A03_CODE.read_text()
# The module-level driver begins at the first assignment to BASE (the arm table).
# Everything before it is imports + constants + class + the loader functions.
_CUT = _SRC.find("\nBASE = ")
if _CUT < 0:
    print("[stageA][FATAL] cannot locate the module-level driver boundary "
          "('BASE = ') in the A03 loader module. Its shape changed; refusing to "
          "exec it wholesale because it has no __main__ guard.", file=sys.stderr)
    sys.exit(2)

_a03_ns: dict = {"__name__": "a03_loaders_only", "__file__": str(_A03_CODE)}
exec(compile(_SRC[:_CUT], str(_A03_CODE), "exec"), _a03_ns)   # noqa: S102

_missing = [n for n in ("load_cb", "load_mmlu", "NotRunYet", "ROOT")
            if n not in _a03_ns]
if _missing:
    print(f"[stageA][FATAL] canonical loaders missing after import: {_missing}",
          file=sys.stderr)
    sys.exit(2)
load_cb, load_mmlu = _a03_ns["load_cb"], _a03_ns["load_mmlu"]
NotRunYet, _A03_ROOT = _a03_ns["NotRunYet"], _a03_ns["ROOT"]

# Guard against the stale-copy defect that cost this script its first run: the
# zwfy6 checkout carried the PRE-FIX `load_mmlu` which guessed FLAT key names
# (`letter_correct` / `content_norm_correct`) against a NESTED record and
# returned (None, None) for all 14,042 items. Detect it by source inspection
# rather than by discovering None mid-analysis.
#
# The check is POSITIVE (require the fixed loader's nested-read loop) rather than
# negative (forbid the string "letter_correct"): the FIXED version quotes the old
# flat key names in its own docstring while explaining the bug, so a
# forbid-the-string test flags the good copy. Assert the fix is present; do not
# try to enumerate the ways it could be absent.
_FIX_MARKER = 'for iface in ("letter", "content_norm")'
if _FIX_MARKER not in _SRC[:_CUT]:
    print("[stageA][FATAL] the A03 loader module on this disk lacks the fixed "
          "nested-key `load_mmlu` (marker not found: "
          f"{_FIX_MARKER!r}). The pre-fix version guesses flat keys against a "
          "nested record and yields (None, None) for every one of the 14,042 "
          "MMLU items, which silently drops the whole axis. Copy the fixed "
          "module from wzc1 before running. See PILOT_ONE_PREREG.md.",
          file=sys.stderr)
    sys.exit(2)

# --- pre-registered constants. No CLI override, by design. -------------------
# Verified against evidence/pilot_zero_rule_disagreement.json
# (per_convention.split.cells[*].delta_pp) on 2026-08-10, before this was written.
DELTA_PP = {
    "triviaqa": 4.043134195274186,
    "popqa": 1.3205298941613512,
    "mmlu_content": 1.0238926078906136,
    "nq_open": 0.9695290858725762,   # DEMOTED, descriptive only
}
DECISION_AXES = ("triviaqa", "popqa", "mmlu_content")
DEMOTED_AXES = ("nq_open",)

T_CRIT_S3_DF2 = 2.920          # t_{0.05, df=2}, one-sided. Design 5.3 table.
S_TARGET = 3
K2_FIRE_MIN_AXES = 2           # ">= 2 of the 3 decision axes"


def axis_mean_pct(dirname: str, axis: str):
    """Mean of the axis's headline metric, in percent. (value, None) or (None, err).

    Closed-book axes use `em` (A03's headline). MMLU uses `content_norm` correct,
    which is the construct-valid interface -- the letter interface is BANNED as a
    decision axis by A04's design 4.2, so it is never read here.
    """
    try:
        if axis == "mmlu_content":
            got = load_mmlu(dirname)
            # load_mmlu returns item_id -> (letter_correct, content_norm_correct)
            vals = [float(v[1]) for v in got.values()]
        else:
            task = "nq_open" if axis == "nq_open" else axis
            got = load_cb(dirname, task)
            # load_cb returns item_id -> (em, contains, f1)
            vals = [float(v[0]) for v in got.values()]
    except NotRunYet as e:
        return None, f"NOT_RUN_YET: {e}"
    except SystemExit as e:
        # the loaders raise SystemExit on partial/corrupt shard sets. Surface it
        # as a problem rather than letting it kill the whole report.
        return None, f"INTEGRITY_FAILURE: {e}"
    if not vals:
        return None, "loader returned zero items"
    return 100.0 * sum(vals) / len(vals), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_a", default="43")
    ap.add_argument("--seed_b", default="44")
    ap.add_argument("--dir_template",
                    default="A03_1B_dataorder_seed{seed}_step220000",
                    help="result dirname template; {seed} substituted. The _nq "
                         "suffix is appended automatically for the nq_open axis, "
                         "matching A03's on-disk convention.")
    ap.add_argument("--out_json", required=True)
    a = ap.parse_args()

    axes = list(DECISION_AXES) + list(DEMOTED_AXES)
    per_axis, problems = {}, {}

    for axis in axes:
        means, errs = {}, {}
        for tag, seed in (("a", a.seed_a), ("b", a.seed_b)):
            dirname = a.dir_template.format(seed=seed)
            if axis == "nq_open":
                dirname += "_nq"
            m, err = axis_mean_pct(dirname, axis)
            if err:
                errs[tag] = f"seed{seed}: {err}"
            else:
                means[tag] = m
        if len(means) < 2:
            problems[axis] = errs
            per_axis[axis] = {"status": "PENDING", "errors": errs,
                              "decision_weight": axis in DECISION_AXES}
            continue
        diff = abs(means["a"] - means["b"])
        sd_run = diff / math.sqrt(2.0)
        bound_3 = T_CRIT_S3_DF2 * sd_run / math.sqrt(S_TARGET)
        delta = DELTA_PP[axis]
        per_axis[axis] = {
            "status": "OK",
            "mean_seed_a_pct": means["a"],
            "mean_seed_b_pct": means["b"],
            "abs_diff_pp": diff,
            "sd_run_pp": sd_run,
            "bound_S3_pp": bound_3,
            "delta_pp": delta,
            "bound_exceeds_delta": bool(bound_3 > delta),
            "decision_weight": axis in DECISION_AXES,
        }

    ready = [x for x in DECISION_AXES if per_axis[x]["status"] == "OK"]
    exceed = [x for x in ready if per_axis[x]["bound_exceeds_delta"]]

    if len(ready) < len(DECISION_AXES):
        verdict = "PENDING_INCOMPLETE_AXES"
        missing = [x for x in DECISION_AXES if x not in ready]
        rationale = (f"only {len(ready)}/{len(DECISION_AXES)} decision axes "
                     f"available; the pre-registered rule needs all three before "
                     f"it can be applied. Missing: {missing}")
    elif len(exceed) >= K2_FIRE_MIN_AXES:
        verdict = "K2_FIRES"
        rationale = (f"bound_3 > Delta on {len(exceed)}/{len(DECISION_AXES)} "
                     f"decision axes ({exceed}), meeting the pre-registered "
                     f"threshold of {K2_FIRE_MIN_AXES}. A04 dies here, for 0 "
                     f"GPU-h beyond what A03 already spent.")
    elif len(exceed) == 1:
        verdict = "K2_INDETERMINATE_AT_STAGE_A"
        rationale = (f"bound_3 > Delta on exactly 1 axis ({exceed}). "
                     f"Pre-registered as neither a fire nor a pass. Stage B "
                     f"proceeds.")
    else:
        verdict = "STAGE_A_DOES_NOT_FIRE"
        rationale = ("bound_3 <= Delta on all 3 decision axes. This is NOT a K2 "
                     "clearance -- see one_directional_caveat. Stage B proceeds.")

    payload = {
        "gate": "A04_pilot_one_stage_A_free_sd_run",
        "prereg": {
            "doc": "proposal/active/A04-recovery-certification/PILOT_ONE_PREREG.md",
            "commit": "2ac0b5a",
            "committed_before_data": True,
            "estimator": "sd_run = |m_a - m_b| / sqrt(2)",
            "bound": f"t_(0.05,2)={T_CRIT_S3_DF2} * sd_run / sqrt({S_TARGET})",
            "rule": f"K2 FIRES iff bound_3 > Delta on >= {K2_FIRE_MIN_AXES} of "
                    f"{len(DECISION_AXES)} decision axes",
            "delta_pp": DELTA_PP,
            "decision_axes": list(DECISION_AXES),
            "demoted_axes": list(DEMOTED_AXES),
        },
        "verdict": verdict,
        "rationale": rationale,
        "n_decision_axes_exceeding": len(exceed),
        "axes_exceeding": exceed,
        "per_axis": per_axis,
        "one_directional_caveat": {
            "large_sd_run": "KILLS -- shared apparatus, Pilot One would inherit it",
            "small_sd_run": "does NOT clear K2",
            "why_not": [
                "wrong arm: keep7+fresh2 (56.2% depth), not keep12+fresh2 (87.5%)",
                "wrong budget: 20,000 steps, not the 5,000 Pilot One would run",
                "n=2 yields a range, not a variance with a usable CI",
            ],
            "consequence": "this script can never emit K2_CLEARED",
        },
        "provenance": {
            "loaders": "load_cb / load_mmlu IMPORTED from A03 (module has no "
                       "__main__ guard, so only its pre-driver prefix is exec'd) "
                       "code/recompute_cpt_trajectory_paired.py -- not "
                       "reimplemented; they carry the 8/8 shard, exact-item-count, "
                       "duplicate-id and NaN assertions",
            "loader_module": str(_A03_CODE),
            "results_root": str(_A03_ROOT),
            "seeds": [a.seed_a, a.seed_b],
            "dir_template": a.dir_template,
            "mmlu_interface": "content_norm (letter interface is BANNED as a "
                              "decision axis by A04 design 4.2)",
            "manipulation_check": "arm3-vs-seed43 training-loss r=-0.0101 vs "
                                  "arm3-vs-arm6 +0.99966 (phase-lock broken)",
        },
        "problems": problems,
    }

    Path(os.path.dirname(a.out_json) or ".").mkdir(parents=True, exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[stageA] VERDICT: {verdict}")
    print(f"[stageA] {rationale}")
    for axis in axes:
        r = per_axis[axis]
        tag = "" if r.get("decision_weight") else "   (DEMOTED, descriptive)"
        if r["status"] != "OK":
            print(f"  {axis:14s} PENDING {r.get('errors')}{tag}")
            continue
        mark = ">" if r["bound_exceeds_delta"] else "<="
        print(f"  {axis:14s} m_a={r['mean_seed_a_pct']:.4f} "
              f"m_b={r['mean_seed_b_pct']:.4f}  "
              f"sd_run={r['sd_run_pp']:.4f}pp  "
              f"bound_3={r['bound_S3_pp']:.4f}pp {mark} "
              f"Delta={r['delta_pp']:.4f}pp{tag}")
    print(f"[stageA] wrote {a.out_json}")


if __name__ == "__main__":
    main()
