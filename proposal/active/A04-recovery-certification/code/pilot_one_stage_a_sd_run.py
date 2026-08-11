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
`load_cb` and `load_mmlu` come from `proposal/shared/code/canonical_eval_loaders.py`.
This is not merely a style rule here: on 2026-08-10 a hand-written `load_mmlu`
that GUESSED flat key names (`letter_correct`, `content_norm_correct`) against a
NESTED record silently returned None for every MMLU cell, dropping a whole axis
from 12 trajectory cells while four .md files went on asserting "MMLU is flat".
Re-deriving a loader for this script would reopen exactly that defect. The
imported versions also carry the shard-completeness assertions (8/8, exact item
counts, duplicate-item_id and NaN checks) that a silent 5-of-8 merge has
previously defeated.

RELOCATED 2026-08-11 -- same loaders, same numbers, no longer inside A03.
They used to live in
`A03-parametric-vs-external-memory/code/recompute_cpt_trajectory_paired.py`, and
this script obtained them by reading that file's SOURCE TEXT, truncating it at the
first `BASE = ` assignment, and `exec`-ing the remainder -- necessary because the
A03 module has no `__main__` guard and runs its whole trajectory driver at module
scope. On 2026-08-11 A03 was decided ARCHIVE (`ARM_SET_DECISION.md`), and that
textual coupling to A03's directory was the one thing physically blocking the
move. The loader bodies were therefore lifted to `proposal/shared/code/`
BYTE-FOR-BYTE -- every assertion and the n_boot=5000 / seed=42 / CI95 protocol
unchanged -- and this script now imports them normally. This script's Stage-A
verdict and the Stage-B S2/S3 verdicts were re-derived after the lift and compared
field-by-field against the archived JSONs BEFORE the move was made.

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

# --- canonical loaders: a NORMAL import from proposal/shared/code -------------
#
# This used to read A03's module source, cut it at the first `BASE = `, and exec
# the prefix (A03's module has no `__main__` guard, so a plain import would have
# re-run its whole trajectory driver and could have raised SystemExit on an
# unrelated arm's partial shards). That textual coupling also hard-wired A03's
# directory path, which is what blocked archiving A03. The loaders now live in
# `proposal/shared/code/canonical_eval_loaders.py` -- definitions only, so a
# normal import is both safe and path-stable.
#
# Failure is LOUD, never a fallback to a locally-derived loader: a silent
# fallback is exactly the shape the 2026-08-10 MMLU defect took.
_SHARED_CODE = Path(__file__).resolve().parents[3] / "shared" / "code"
_LOADER_MODULE = _SHARED_CODE / "canonical_eval_loaders.py"
if not _LOADER_MODULE.is_file():
    print(f"[stageA][FATAL] canonical loader module not found: {_LOADER_MODULE}",
          file=sys.stderr)
    sys.exit(2)

sys.path.insert(0, str(_SHARED_CODE))
try:
    from canonical_eval_loaders import (  # noqa: E402
        ROOT as _A03_ROOT,
        NotRunYet,
        load_cb,
        load_mmlu,
    )
except Exception as _e:      # ImportError, or a numpy/etc. failure inside it
    print(f"[stageA][FATAL] cannot import canonical loaders from "
          f"{_LOADER_MODULE}: {_e!r}", file=sys.stderr)
    sys.exit(2)

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
#
# STILL NECESSARY after the 2026-08-11 relocation: zwfy6's `proposal/` tree is a
# hand-copied directory, NOT a git checkout (`git ls-files proposal/` returns 0
# there), so it does not receive commits automatically and a stale loader copy on
# that disk remains possible. This is the check that catches it.
_FIX_MARKER = 'for iface in ("letter", "content_norm")'
if _FIX_MARKER not in _LOADER_MODULE.read_text():
    print("[stageA][FATAL] the canonical loader module on this disk lacks the "
          "fixed nested-key `load_mmlu` (marker not found: "
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
            "loaders": "load_cb / load_mmlu IMPORTED from "
                       "proposal/shared/code/canonical_eval_loaders.py -- not "
                       "reimplemented; they carry the 8/8 shard, exact-item-count, "
                       "duplicate-id and NaN assertions. Relocated 2026-08-11 "
                       "BYTE-FOR-BYTE out of A03's "
                       "code/recompute_cpt_trajectory_paired.py (which has no "
                       "__main__ guard, so this script used to exec only its "
                       "pre-driver prefix) so that archiving A03 could not break "
                       "these numbers; re-verified field-by-field against the "
                       "pre-move verdict JSONs.",
            "loader_module": str(_LOADER_MODULE),
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
