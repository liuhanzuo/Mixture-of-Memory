#!/usr/bin/env python3
"""B04 G0 — floor-first analysis of the wzc1 sm_100 ladder. ZERO GPU.

Run from repo root:
    python proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py
    ... --selftest        both selftests: arithmetic (revision 2) + on-disk fixture (revision 3)
    ... --readout-only    just clause 5 revision 3, from the banked evidence constants

Exit code: 0 on a computed verdict (KILL/NARROWED/PASS); non-zero on any prereg HARD ABORT
(READOUT_ABSENT 3, PROTOCOL_VIOLATION/FIELD_ASYMMETRY 4, DENOMINATOR_UNRESOLVED/
FLOOR_UNMEASURABLE 5). An abort is a NON-PASS and must never look like success to a shell.

What this does
--------------
1. Computes sigma_hat (the noise floor) from the keep14 seed pair, which holds damage
   depth AND heal step EXACTLY constant and differs only in init seed.
2. Computes the 6-rung wzc1 ladder margin metrics, plus R = range/sigma_hat and the
   count of adjacent (core6-ordered) rung gaps that clear 2*sigma_hat.
3. Computes Spearman(core6, metric) with exact permutation p, AND the mandatory
   co-disclosure Spearman(core6, heal_steps).
4. Emits evidence/B04_wzc1_floor_analysis.json.

Why norm_lens can be transplanted (no GPU, no writes)
-----------------------------------------------------
Six of the eight dirs lack `norm_scores`. But norm_scores[L] = option_scores[L] /
max(norm_lens[L], 1), and `norm_lens` is the raw candidate character count -- a property
of the DATASET, never of the model. So it can be read from the already-enriched donor and
joined by item_id. This is validated at runtime against the donor itself (assert_exact):
recomputing the donor's own metrics through the transplant path must reproduce its native
values bit-for-bit. It does.

The alternative is scripts/enrich_per_example_normscores.py, which WRITES the field in
place. This script deliberately does not write to olmo2_downstream_results/ at all.

Pre-registration
----------------
PRIMARY = median_margin. Fixed 2026-08-14 BEFORE the G1 arms exist, on the noise-floor
argument recorded in GATE_DESIGN.md sec 1. The three frac(margin<t) metrics are reported
and explicitly underpowered.

Decision statistic, REVISION 2 (2026-08-14, still PRE-DATA, 0 GPU)
------------------------------------------------------------------
Revision 1 used  phi = |beta_budget| * 116500 / damaged_range  and was refuted 3/3 by the
adversarial pass. Two defects, both fixed here:

  (D1) decidability. 116500 is the DAMAGED LADDER's heal-step span. The read-out is at
       steps {25000,50000,100000,128000,200000}, whose own span is 175000. Rescaling a
       slope measured over 175000 down to 116500 means the printed number is not the
       measured excursion -- it understates it by 175000/116500 = 1.5021x. The span in the
       statistic must be the READ-OUT's own span. Fixed: READOUT_SPAN = 175000.

  (D2) falsifiability. A slope-only statistic is blind to shape. A non-monotone budget
       response can move median_margin across almost the whole damaged range while its OLS
       slope is ~0. Worked example on this exact x-grid:
           y = [0.1085, 0.0905, 0.0885, 0.0975, 0.1090]
           range = 0.020500 (94.0% of the damaged range 0.021820)
           |beta| * 116500 = 0.003790  ->  revision-1 phi = 0.1895  ->  PASS
       i.e. budget reproducing 94% of the damage range would have been recorded as "budget
       is negligible". The Qwen precedent is itself non-monotone, so this is not a corner
       case. Fixed by a max-guard:

           phi = max( range(median_margin over the 5 read-out points),
                      |beta_budget| * READOUT_SPAN )  /  damaged_range

The range term is shape-agnostic (it assumes nothing about the functional form of the
budget response), and it is k-MATCHED to the denominator: numerator range is over k=5
budget points, denominator range is over k=5 damaged rungs, so E[range of k]/sigma is the
same on both sides (memory/a-range-is-not-a-measurement-until-it-clears-its-floor.md).
Revision 1's numerator was a slope, which had no such k-matching.

Why max(), and not the range alone: on this fixed x-grid the slope term is bounded by
    sup_y  |beta|*S / range(y)  =  S * sum_{i: w_i>0} w_i  =  1.173627,   w_i=(x_i-xbar)/Sxx
attained by a step function. So max() can exceed range-alone by at most 17.4% and NEVER
falls below it: it is strictly the more conservative of the two candidate statistics while
still being shape-agnostic. Dropping the slope term would only ever let a case pass that
max() kills. Both terms are emitted separately so either can be audited.

Decision statistic, REVISION 3 (2026-08-16, still PRE-DATA, 0 GPU) -- and THE READ-OUT PATH
-------------------------------------------------------------------------------------------
Authority: DECIDABILITY_FIX_20260816.md sec 3, which is the verbatim gate text. The code
below IMPLEMENTS it; it does not amend it. Revision 2's grid, span, phi_budget() and
selftest are left in place unmodified for provenance -- revision 2's grid IS revision 3's
secondary grid.

Two things changed:

  (1) THE GRID. Revision 1 multiplied a slope measured on the read-out grid by the
      COMPARATOR's foreign span 116500. Revision 2 fixed the constant to the grid's own
      175000 but left the grid, 2 of whose 5 points (25000, 50000) sit BELOW the comparator
      interval I = [83500, 200000], measuring budget movement that cannot explain any
      damaged rung's position. So neither span was right: the grid had to change.
        GRID_I = {100000,128000,153500,175000,200000}  S_I = 100000  PRIMARY   (5/5 in I,
                 covering 0.8584 of |I|; the uncovered [83500,100000] = 14.16% is DISCLOSED,
                 so phi_I is a LOWER BOUND over I)
        GRID_W = {25000,50000,100000,128000,200000}    S_W = 175000  SECONDARY (revision 2
                 verbatim, retained so revision 2 stays auditable)
      FINAL VERDICT = the MORE SEVERE of verdict_I and verdict_W (KILL > NARROWED > PASS).

  (2) THE INPUTS NOW EXIST AS CODE. Before 2026-08-16 phi_budget() was called from exactly
      three places -- :503, :526, :531 in the pre-2026-08-16 file (git 625053b), i.e. all
      inside selftest_phi(), all on hand-written y-vectors. The analyzer did read
      olmo2_downstream_results/ (for the donor margin and the completeness checks), but
      NOTHING it read from disk ever reached phi. So three adversarial passes checked whether
      phi's FORMULA was right and none checked whether phi's INPUTS EXISTED.
      clause5_revision3() closes that: it loads median_margin at each grid step from the
      on-disk eval dirs, assembles y, computes phi per grid, and combines. Every prereg hard
      abort (READOUT_ABSENT, PROTOCOL_VIOLATION, FIELD_ASYMMETRY, DENOMINATOR_UNRESOLVED,
      FLOOR_UNMEASURABLE) is raised, never swallowed: a missing rung is NEVER interpolated, a
      grid is NEVER shortened, and no NaN is allowed to flow onward.
"""

from __future__ import annotations

import json
import statistics
import sys
from itertools import permutations
from pathlib import Path

ROOT = Path("olmo2_downstream_results")
OUT = Path("proposal/backlog/B04-eval-fragility-incubator/evidence/B04_wzc1_floor_analysis.json")

TASKS = ["hellaswag", "arc_challenge", "arc_easy", "piqa", "winogrande", "openbookqa"]
EXPECTED_N = {"hellaswag": 10042, "arc_challenge": 1172, "arc_easy": 2376,
              "piqa": 1838, "winogrande": 1267, "openbookqa": 500}
EXPECTED_POOLED = sum(EXPECTED_N.values())  # 17195
THRESHOLDS = [0.001, 0.005, 0.010]
METRICS = ["median_margin", "frac_lt_0.001", "frac_lt_0.005", "frac_lt_0.01"]

# donor: the only OLMo dirs on wzc1 that natively carry norm_scores
DONOR = "keep14_s42_step200000_sv181"
SEED_PAIR = [("s42", "keep14_s42_step200000_sv181"),
             ("s1234", "keep14_s1234_step200000_sv181")]

# wzc1 sm_100 ladder. NOTE keep12 is step111500 here, NOT the step124000 of the
# zwfy6 ladder in evidence/B04_6rung_bs16_analysis.json. See GATE_DESIGN.md sec 2.
LADDER = [
    ("base_full",       "7B_full32_base_wzc1_v2",        None, None),
    ("shortgpt16@200k", "7B_shortgpt16_step200000_wzc1",   16, 200000),
    ("keep14@200k",     "7B_keep14_step200000_wzc1_v2",    14, 200000),
    ("keep12@111.5k",   "7B_keep12_step111500_wzc1",       12, 111500),
    ("keep10@83.5k",    "7B_keep10_step83500_wzc1",        10,  83500),
    ("keep8@121k",      "7B_keep8_step121000_wzc1",         8, 121000),
]

# E[range of k] / sigma for a normal sample. Using 1.0 understates the floor and can
# flip the boolean -- see memory/a-range-is-not-a-measurement-until-it-clears-its-floor.md
E_RANGE_OVER_SIGMA = {2: 1.1284, 3: 1.6926, 4: 2.0588, 5: 2.3259, 6: 2.5344,
                      7: 2.7044, 8: 2.8472}

FLOOR_SAFETY_FACTOR = 6  # damaged range must clear 6*sigma_hat to be an admissible denominator

# ---- G1 read-out pre-registration (REVISION 2, 2026-08-14, PRE-DATA) --------------------
# The 5 heal steps of olmo2_probe2_7B_keep14fresh2_seed1234 that constitute the ENTIRE
# read-out. Their own span is the span that appears in phi -- NOT the damaged ladder's
# 116500. Revision 1 used 116500 and the decidability lens correctly refused it: the
# printed number was a rescaling of a different span, understating the measured excursion
# by 175000/116500 = 1.5021x.
G1_READOUT_STEPS = [25000, 50000, 100000, 128000, 200000]
READOUT_SPAN = max(G1_READOUT_STEPS) - min(G1_READOUT_STEPS)   # 175000
# Upper bound of |beta|*READOUT_SPAN / range over this fixed x-grid; attained by a step
# function. Emitted so the max() guard's worst-case inflation over range-alone is auditable.
SLOPE_TERM_SUP_RATIO = 1.173627
PHI_KILL, PHI_PASS = 0.60, 0.30

# =========================================================================================
# ---- REVISION 3 (2026-08-16, PRE-DATA, 0 GPU) -------------------------------------------
# =========================================================================================
# Authority: DECIDABILITY_FIX_20260816.md sec 3 (the verbatim gate block). This section
# IMPLEMENTS that pre-registration; it does not amend it. No threshold, no grid, and no
# abort name below was chosen here -- every one is copied from sec 3.
#
# Everything above (G1_READOUT_STEPS, READOUT_SPAN, SLOPE_TERM_SUP_RATIO, phi_budget) is
# REVISION 2 and is deliberately LEFT IN PLACE, unmodified, for provenance: revision 2's
# grid is revision 3's SECONDARY grid, and its selftest is the record that the max-guard
# and the shape-safe boundaries were checked. GRID_W below is asserted identical to it.
#
# Why revision 3 exists (sec 2): revision 1 multiplied a slope measured on the read-out
# grid by the COMPARATOR's foreign span 116500; revision 2 fixed the constant to the grid's
# own 175000 but left the grid, 2 of whose 5 points (25000, 50000) sit BELOW the comparator
# interval I = [83500, 200000] and so measure budget movement that cannot explain any
# damaged rung's position. Neither span is right; the grid had to change.

# The comparator's heal-step interval. DESCRIPTIVE ONLY. |I| MUST NOT multiply any slope
# (that is the revision-1 defect); it is used ONLY to test whether a grid is on-support.
COMPARATOR_INTERVAL_I = (83500, 200000)

GRID_I = [100000, 128000, 153500, 175000, 200000]   # PRIMARY   -- interval-matched, 5/5 in I
SPAN_I = max(GRID_I) - min(GRID_I)                  # 100000
GRID_W = list(G1_READOUT_STEPS)                     # SECONDARY -- revision 2's grid, verbatim
SPAN_W = max(GRID_W) - min(GRID_W)                  # 175000

# sup_y |beta|*S / range(y) over each FIXED x-grid, attained by a step function:
#   sup = S * sum_{i: w_i > 0} w_i,   w_i = (x_i - xbar) / Sxx
# Recomputed at import (below) and asserted against these pinned literals, so a grid edit
# cannot silently leave a stale sup ratio behind.
SUP_RATIO_I = 1.220390
SUP_RATIO_W = 1.173627

# KILL > NARROWED > PASS. The combined verdict is the MORE SEVERE of the two grids'.
VERDICT_SEVERITY = {"PASS": 0, "NARROWED": 1, "KILL": 2}

READOUT_GRIDS = [
    ("GRID_I", GRID_I, SPAN_I, SUP_RATIO_I, "PRIMARY"),
    ("GRID_W", GRID_W, SPAN_W, SUP_RATIO_W, "SECONDARY"),
]
READOUT_UNION_STEPS = sorted(set(GRID_I) | set(GRID_W))
# = [25000, 50000, 100000, 128000, 153500, 175000, 200000]

# The read-out arm. Damage held EXACTLY at keep_front=14 / n_fresh=2 / seed=1234; ONLY
# ckpt_step varies. A dir is accepted only if its OWN summary.json.meta names this ckpt.
READOUT_CKPT_DIR = "outputs/olmo2_probe2_7B_keep14fresh2_seed1234"
READOUT_CKPT_MARK = "keep14fresh2_seed1234"
READOUT_KEEP_FRONT, READOUT_N_FRESH = 14, 2
READOUT_BASE_MODEL = "../models/OLMo-2-1124-7B"

# ---- what the prereg names but summary.json does NOT record ------------------------------
# The prereg's arm definition (DECIDABILITY_FIX_20260816.md sec 3 READ-OUT; field table in
# EVAL_FILL_READY_20260816.md sec 1) fixes batch_size=8, max_len=1024 (default), the harness
# at git a163a89, and one driver invocation. NONE of those is written into summary.json --
# its only keys are output_name / n_shards / add_bos / meta / tasks, and meta's only keys are
# mode / keep_front_layers / n_fresh_layers / num_hidden_layers / ckpt_step / ckpt /
# base_model / add_bos (verified on the archived rung 2026-08-16). So the read-out path
# CANNOT verify them from disk, and it does not pretend to. It DISCLOSES them instead, in
# every report, under `unverifiable_from_disk`. Enforcement lives in the fill driver
# (scripts/_run_b04_readout_evalfill.sh), which pins the flags and refuses to run against an
# uncommitted harness. Reported as a prereg gap, NOT resolved here.
UNVERIFIABLE_FROM_DISK = {
    "fields": ["batch_size (prereg 8)", "max_len (prereg 1024, harness default)",
               "harness git commit (prereg a163a89)", "one-driver-invocation"],
    "why": "summary.json records none of them; its keys are output_name/n_shards/add_bos/"
           "meta/tasks and meta's are mode/keep_front_layers/n_fresh_layers/"
           "num_hidden_layers/ckpt_step/ckpt/base_model/add_bos.",
    "who_enforces": "scripts/_run_b04_readout_evalfill.sh (pins the flags; refuses an "
                    "uncommitted harness). A rung produced by any OTHER route is NOT "
                    "provably same-harness from its own artifacts.",
    "status": "DISCLOSED PREREG GAP, not resolved by this code. See "
              "READOUT_PATH_20260816.md 'Ambiguity 1'.",
}

# Both physical disks (memory/two-disk-rule-applies-to-main-too.md). A root that is not
# mounted from this node is reported as UNSEARCHABLE -- never silently treated as empty.
SEARCH_ROOTS = [
    ("wzc1", Path("olmo2_downstream_results")),
    ("zwfy6", Path("/apdcephfs_zwfy6/share_304376610/pighzliu_code/"
                   "Mixture-of-Memory/olmo2_downstream_results")),
]


class GateAbort(Exception):
    """A prereg HARD ABORT. Carries the abort's prereg name so callers assert on it.

    Every abort is a NON-PASS and blocks the downstream spend. This is an exception and
    not a sys.exit so the fixture selftest can assert that each one actually fires.
    """

    def __init__(self, code: str, msg: str, detail: dict | None = None):
        super().__init__(f"{code}: {msg}")
        self.code, self.msg, self.detail = code, msg, detail or {}


def _sup_ratio(steps, span):
    n = len(steps)
    mx = sum(steps) / n
    sxx = sum((a - mx) ** 2 for a in steps)
    return span * sum(w for w in ((a - mx) / sxx for a in steps) if w > 0)


# Fail at import if a grid and its pinned sup ratio ever drift apart.
for _nm, _g, _s, _sup, _role in READOUT_GRIDS:
    _got = _sup_ratio(_g, _s)
    if abs(_got - _sup) > 5e-7:
        sys.exit(f"FATAL: {_nm} sup ratio drifted: pinned {_sup}, recomputed {_got:.6f}. "
                 "A grid or span was edited without updating its sup ratio.")
if GRID_W != G1_READOUT_STEPS or SPAN_W != READOUT_SPAN:
    sys.exit("FATAL: GRID_W must be revision 2's grid VERBATIM (auditability requirement, "
             f"DECIDABILITY_FIX_20260816.md sec 3): {GRID_W} vs {G1_READOUT_STEPS}")
del _nm, _g, _s, _sup, _role, _got


def phi_budget_grid(y, damaged_range, grid_name, steps, span, evaluated_steps=None):
    """REVISION-3 per-grid statistic. Identical arithmetic to phi_budget, per grid.

        phi_G = max( max(y_G) - min(y_G), |OLS slope of y_G on heal_step| * S_G ) / D

    `steps` is the grid's PREREG step set; `evaluated_steps` is what was actually found on
    disk. They must be equal -- adding points biases toward PASS (the unused seed1234 ckpts
    cluster near 200000, shrinking the range term); dropping one breaks k=5-vs-k=5 matching
    (E[range]/sigma 2.0588 at k=4 vs 2.3259 at k=5, -11.5%, also toward PASS).

    Raises GateAbort; never returns a NaN and never shortens a grid.
    """
    steps = list(steps)
    if evaluated_steps is not None and list(evaluated_steps) != steps:
        raise GateAbort("PROTOCOL_VIOLATION",
                        f"{grid_name} evaluated step set {list(evaluated_steps)} != prereg "
                        f"{steps}. Extending n until the statistic crosses a threshold is "
                        "the paperC --max_steps error "
                        "(readout_preregistration.not_a_decision_point).",
                        {"grid": grid_name})
    if len(y) != len(steps):
        raise GateAbort("PROTOCOL_VIOLATION",
                        f"{grid_name} read-out has {len(y)} points, prereg has {len(steps)}",
                        {"grid": grid_name})
    if any(v is None or v != v for v in y):          # v != v catches NaN
        raise GateAbort("READOUT_ABSENT",
                        f"{grid_name} y contains a hole/NaN: {y}. phi is UNDEFINED -- not "
                        "small, not large. A NaN must never flow onward.",
                        {"grid": grid_name})
    if damaged_range is None or damaged_range <= 0:
        raise GateAbort("DENOMINATOR_UNRESOLVED",
                        f"damaged_range={damaged_range} <= 0 -> phi is UNDEFINED, not small.",
                        {"grid": grid_name})
    rng = max(y) - min(y)
    slope_term = abs(ols_slope(steps, y)) * span
    phi = max(rng, slope_term) / damaged_range
    v = "KILL" if phi >= PHI_KILL else ("PASS" if phi <= PHI_PASS else "NARROWED")
    return {
        "grid": grid_name, "steps": steps, "span_used": span, "y": list(y),
        "phi": phi, "verdict": v,
        "range_term": rng, "slope_term": slope_term,
        "binding_term": "range" if rng >= slope_term else "slope",
        "damaged_range_D": damaged_range,
        "phi_kill_threshold": PHI_KILL, "phi_pass_threshold": PHI_PASS,
        "sup_ratio_this_grid": _sup_ratio(steps, span),
        "points_inside_comparator_I": sum(
            1 for s in steps if COMPARATOR_INTERVAL_I[0] <= s <= COMPARATOR_INTERVAL_I[1]),
    }


def combine_verdicts(per_grid: dict) -> str:
    """FINAL VERDICT = the MORE SEVERE of verdict_I and verdict_W (KILL > NARROWED > PASS)."""
    if not per_grid:
        raise GateAbort("READOUT_ABSENT", "no grid produced a verdict")
    return max((r["verdict"] for r in per_grid.values()),
               key=lambda v: VERDICT_SEVERITY[v])


# ---- the read-out path: on-disk eval dirs -> y -> phi ------------------------------------

def readout_margins(dirpath: Path) -> list[float]:
    """Pooled |margin| for ONE read-out arm, from its OWN native norm_scores.

    Deliberately does NOT use the norm_lens transplant that margins() falls back to: the
    prereg's FIELD_ASYMMETRY abort exists because an asymmetric-field paired comparison
    already produced a 56x artefact once (status/PAPERF_ACCNORM_VERIFIED.md:43-67). Every
    read-out arm is produced by the a163a89 harness, which writes norm_scores/norm_lens
    natively, so a missing field means the arm is not the protocol of record -- not that it
    needs repairing here.

    margin = |score(gold) - max(other)| on norm_scores  (same definition as margins():172-201)
    """
    out = []
    for t in TASKS:
        p = dirpath / f"per_example_{t}.jsonl"
        if not p.exists():
            raise GateAbort("READOUT_ABSENT",
                            f"{dirpath.name}: no per_example_{t}.jsonl "
                            "(--save_per_example missing?) -> margin not computable")
        n = rows = 0
        for line in open(p):
            if not line.strip():
                continue
            rows += 1
            o = json.loads(line)
            for fld in ("norm_scores", "norm_lens", "gold_letter", "item_id"):
                if fld not in o or o[fld] is None:
                    raise GateAbort("FIELD_ASYMMETRY",
                                    f"{dirpath.name}/{t} row {rows} lacks '{fld}' -> margin "
                                    "not computable on this arm's own fields; the transplant "
                                    "fallback is BARRED for read-out arms")
            sc, g = o["norm_scores"], o["gold_letter"]
            oth = [v for k, v in sc.items() if k != g and v is not None]
            if sc.get(g) is None or not oth:
                continue
            out.append(abs(sc[g] - max(oth)))
            n += 1
        if rows != EXPECTED_N[t]:
            raise GateAbort("PROTOCOL_VIOLATION",
                            f"{dirpath.name}/per_example_{t}.jsonl has {rows} rows, expected "
                            f"{EXPECTED_N[t]}")
        if n != EXPECTED_N[t]:
            raise GateAbort("PROTOCOL_VIOLATION",
                            f"{dirpath.name}/{t} scored {n}, expected {EXPECTED_N[t]}")
    if len(out) != EXPECTED_POOLED:
        raise GateAbort("PROTOCOL_VIOLATION",
                        f"{dirpath.name} pooled {len(out)} != {EXPECTED_POOLED}")
    return out


def inspect_readout_dir(dirpath: Path, want_step: int) -> dict:
    """Is this dir a margin-computable read-out arm for `want_step`? Fail loudly if malformed.

    Returns {"ok": True, "median_margin": ..., ...} or {"ok": False, "why": ...} when the dir
    simply is not this arm (wrong ckpt / wrong task set -> a candidate to skip). Anything
    that IS this arm but is broken raises GateAbort: it must never be silently skipped, or a
    partial merge would read as "absent" and then as "filled" on the next run.
    """
    s = dirpath / "summary.json"
    if not s.exists():
        return {"ok": False, "why": "no summary.json"}
    try:
        j = json.load(open(s))
    except Exception as e:                                    # noqa: BLE001
        return {"ok": False, "why": f"summary.json unreadable: {e}"}
    meta = j.get("meta") or {}
    ck = meta.get("ckpt") or ""
    # --- identity: the dir must belong to the INTENDED checkpoint, not merely be named it.
    if READOUT_CKPT_MARK not in ck:
        return {"ok": False, "why": f"meta.ckpt={ck!r} is not a {READOUT_CKPT_MARK} ckpt"}
    if str(meta.get("ckpt_step")) != str(want_step):
        return {"ok": False, "why": f"meta.ckpt_step={meta.get('ckpt_step')} != {want_step}"}
    tasks = j.get("tasks") or {}
    if sorted(tasks) != sorted(TASKS):
        # e.g. the know5 dir at step200000: right ckpt, DIFFERENT task set. Not a candidate.
        return {"ok": False, "why": f"task set {sorted(tasks)} != core6"}

    # From here on the dir IS the named arm at the named step. Every remaining defect is a
    # hard abort, because a plausible-looking short merge must never reach phi.
    nsh = len(list(dirpath.glob("shard*of8.json")))
    if nsh != 8:
        raise GateAbort("PROTOCOL_VIOLATION",
                        f"{dirpath.name} has {nsh}/8 shard files -- refusing partial merge. "
                        "A silent 5/8 merge has destroyed a measurement in this project.")
    if j.get("n_shards") != 8:
        raise GateAbort("PROTOCOL_VIOLATION",
                        f"{dirpath.name} summary.n_shards={j.get('n_shards')} != 8 -- "
                        "PARTIAL MERGE that looks complete")
    if j.get("add_bos") is not False:
        raise GateAbort("PROTOCOL_VIOLATION",
                        f"{dirpath.name} add_bos={j.get('add_bos')} != False; OLMo-2 "
                        "published numbers are made without BOS")
    if (meta.get("keep_front_layers"), meta.get("n_fresh_layers")) != \
       (READOUT_KEEP_FRONT, READOUT_N_FRESH):
        raise GateAbort("PROTOCOL_VIOLATION",
                        f"{dirpath.name} keep/fresh = {meta.get('keep_front_layers')}/"
                        f"{meta.get('n_fresh_layers')} != {READOUT_KEEP_FRONT}/"
                        f"{READOUT_N_FRESH}: damage is NOT held fixed across the read-out")
    if meta.get("base_model") != READOUT_BASE_MODEL:
        raise GateAbort("PROTOCOL_VIOLATION",
                        f"{dirpath.name} base_model={meta.get('base_model')!r} != "
                        f"{READOUT_BASE_MODEL!r}: wrong base means wrong transplant")
    for t in TASKS:
        e = tasks[t]
        if e.get("skipped"):
            raise GateAbort("PROTOCOL_VIOLATION", f"{dirpath.name}/{t} SKIPPED")
        if e.get("n_scored") != EXPECTED_N[t]:
            raise GateAbort("PROTOCOL_VIOLATION",
                            f"{dirpath.name}/{t} n_scored={e.get('n_scored')} != "
                            f"{EXPECTED_N[t]}")
        if e.get("n_nan", 0) != 0:
            raise GateAbort("PROTOCOL_VIOLATION",
                            f"{dirpath.name}/{t} n_nan={e.get('n_nan')} != 0")
    ms = readout_margins(dirpath)                 # asserts rows AND scored AND pooled AND fields
    return {"ok": True, "dir": str(dirpath), "ckpt": ck, "ckpt_step": meta.get("ckpt_step"),
            "n_pooled": len(ms), "median_margin": statistics.median(ms),
            "base_model": meta.get("base_model"), "num_hidden_layers":
            meta.get("num_hidden_layers")}


def find_readout_arms(steps, roots=None) -> dict:
    """Scan every candidate dir on every mounted root for each read-out step.

    Identification is by the dir's OWN summary.json.meta.ckpt / ckpt_step, never by its
    name -- a dir with the right name and the wrong checkpoint would feed a wrong y into
    phi and nothing downstream would notice.

    A root that is not mounted from this node is reported UNSEARCHABLE, not empty
    (memory/two-disk-rule-applies-to-main-too.md).
    """
    roots = SEARCH_ROOTS if roots is None else roots
    census = {"roots": [], "per_step": {}}
    for label, root in roots:
        if not root.is_dir():
            census["roots"].append({"disk": label, "root": str(root), "searched": False,
                                    "why": "not mounted from this node -> absence on this "
                                           "disk is NOT established here"})
            continue
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
        census["roots"].append({"disk": label, "root": str(root), "searched": True,
                                "n_dirs_scanned": len(dirs)})
        for st in steps:
            for p in dirs:
                r = inspect_readout_dir(p, st)
                if r.get("ok"):
                    census["per_step"].setdefault(str(st), []).append({**r, "disk": label})
    resolved, absent = {}, []
    for st in steps:
        c = census["per_step"].get(str(st), [])
        if not c:
            absent.append(st)
            continue
        if len(c) > 1:
            mm = {round(x["median_margin"], 12) for x in c}
            if len(mm) > 1:
                # Not covered by the prereg. Fail loud rather than pick: see
                # READOUT_PATH_20260816.md "prereg ambiguity 3".
                raise GateAbort("PROTOCOL_VIOLATION",
                                f"step {st} has {len(c)} margin-computable dirs that DISAGREE "
                                f"on median_margin ({sorted(mm)}): "
                                f"{[x['dir'] for x in c]}. The prereg names one arm per step; "
                                "refusing to choose.")
        resolved[st] = c[0]
    # Keyed by int throughout -- callers index it with the grid's own int steps. (An earlier
    # draft re-keyed this to str for the JSON and every int lookup silently missed, so the
    # census found step200000 and `missing` still listed it. Caught by fixture check F1.)
    census["resolved"] = resolved
    census["absent_steps"] = absent
    return census


def clause5_revision3(damaged_range: float, sigma_hat: float, rho_core6_heal: float,
                      roots=None) -> dict:
    """THE GATE. Loads median_margin at the grid steps from disk, computes phi on GRID_I
    (primary) and GRID_W (secondary), and combines by the prereg's severity rule.

    Returns a report dict. `verdict` is one of KILL / NARROWED / PASS / an abort name.
    Never interpolates a missing rung, never shortens a grid, never returns a NaN.
    """
    rep = {
        "revision": "3 (2026-08-16, PRE-DATA). Implements DECIDABILITY_FIX_20260816.md sec 3.",
        "primary_metric": "median_margin",
        "margin_definition": "|score(gold) - max(other)| on norm_scores, pooled over the 6 "
                             "core tasks; same definition as margins() at :172-201",
        "readout_arm": {"ckpt_dir": READOUT_CKPT_DIR, "keep_front_layers": READOUT_KEEP_FRONT,
                        "n_fresh_layers": READOUT_N_FRESH, "seed": 1234,
                        "base_model": READOUT_BASE_MODEL, "only_ckpt_step_varies": True},
        "unverifiable_from_disk": UNVERIFIABLE_FROM_DISK,
        "D_damaged_range": damaged_range, "sigma_hat": sigma_hat,
        "comparator_interval_I": list(COMPARATOR_INTERVAL_I),
        "abs_I": COMPARATOR_INTERVAL_I[1] - COMPARATOR_INTERVAL_I[0],
        "span_116500_status": "DESCRIPTIVE ONLY -- BARRED from multiplying any slope "
                              "(the revision-1 defect); used only to test grid support",
        "grids": {nm: {"steps": g, "span": s, "sup_ratio": sup, "role": role,
                       "points_inside_I": sum(1 for x in g
                                              if COMPARATOR_INTERVAL_I[0] <= x
                                              <= COMPARATOR_INTERVAL_I[1])}
                  for nm, g, s, sup, role in READOUT_GRIDS},
        "combine_rule": "FINAL = the MORE SEVERE of verdict_I and verdict_W "
                        "(KILL > NARROWED > PASS)",
        "mandatory_codisclosure_spearman_core6_heal_steps": rho_core6_heal,
        "union_steps": READOUT_UNION_STEPS,
    }
    if sigma_hat == 0:
        rep["verdict"] = "FLOOR_UNMEASURABLE"
        rep["why"] = "sigma_hat == 0 -> the contrast is not a real nuisance contrast. NOT a pass."
        return rep
    guard = FLOOR_SAFETY_FACTOR * sigma_hat
    rep["denominator_guard_6sigma"] = guard
    if damaged_range is None or damaged_range <= 0 or damaged_range < guard:
        rep["verdict"] = "DENOMINATOR_UNRESOLVED"
        rep["why"] = f"D={damaged_range} below guard {guard}. phi UNDEFINED. NON-PASS."
        return rep

    try:
        census = find_readout_arms(READOUT_UNION_STEPS, roots=roots)
    except GateAbort as e:
        rep["verdict"], rep["why"], rep["abort_detail"] = e.code, e.msg, e.detail
        return rep
    rep["readout_census"] = {
        "roots": census["roots"],
        # str-keyed on purpose: this half is for the JSON/print side. The int-keyed
        # census["resolved"] is what the grid lookups below use.
        "found": {str(k): {"dir": v["dir"], "disk": v["disk"], "ckpt": v["ckpt"],
                           "median_margin": v["median_margin"], "n_pooled": v["n_pooled"]}
                  for k, v in census["resolved"].items()},
        "absent_steps": census["absent_steps"],
    }

    missing = {nm: [s for s in g if s not in census["resolved"]]
               for nm, g, _, _, _ in READOUT_GRIDS}
    rep["missing_per_grid"] = missing
    if any(missing.values()):
        rep["verdict"] = "READOUT_ABSENT"
        rep["why"] = ("a named arm of a grid lacks a margin-computable eval dir. phi is "
                      "UNDEFINED -- not small, not large. NON-PASS: an undefined ratio "
                      "cannot license 244-2560 GPU-h.")
        rep["blocks_spend"] = True
        return rep

    per_grid = {}
    try:
        for nm, g, s, _sup, _role in READOUT_GRIDS:
            y = [census["resolved"][st]["median_margin"] for st in g]
            per_grid[nm] = phi_budget_grid(y, damaged_range, nm, g, s, evaluated_steps=g)
    except GateAbort as e:
        rep["verdict"], rep["why"], rep["abort_detail"] = e.code, e.msg, e.detail
        rep["per_grid"] = per_grid
        return rep
    rep["per_grid"] = per_grid
    rep["verdict"] = combine_verdicts(per_grid)
    rep["blocks_spend"] = rep["verdict"] != "PASS"
    return rep


def print_clause5_revision3(rep: dict) -> None:
    """MANDATORY REPORTING: every phi is printed with its grid, span, binding term, D,
    sigma_hat, and Spearman(core6, heal_steps). A phi without its span is the artefact the
    decidability lens caught (116500 vs 175000 vs 100000 differ by up to 1.75x).
    """
    print("\n=== CLAUSE 5, REVISION 3 (two-grid, interval-matched) ===")
    print(f"  D (damaged median_margin range) = {rep['D_damaged_range']:.6f}   "
          f"sigma_hat = {rep['sigma_hat']:.6f}   "
          f"6*sigma guard = {rep.get('denominator_guard_6sigma', float('nan')):.7f}")
    print(f"  Spearman(core6, heal_steps) = "
          f"{rep['mandatory_codisclosure_spearman_core6_heal_steps']:+.4f} (wzc1 ladder; "
          f"+0.8721 on zwfy6 -- naming the ladder is mandatory)")
    print(f"  comparator interval I = {rep['comparator_interval_I']} |I| = {rep['abs_I']} "
          f"({rep['span_116500_status']})")
    for nm, g in rep["grids"].items():
        print(f"  {nm:7s} {g['role']:9s} steps={g['steps']} span={g['span']} "
              f"sup={g['sup_ratio']:.6f} inside_I={g['points_inside_I']}/{len(g['steps'])}")
    u = rep["unverifiable_from_disk"]
    print(f"  NOT VERIFIABLE FROM DISK (disclosed prereg gap): {', '.join(u['fields'])}")
    print(f"    enforced instead by: {u['who_enforces'].split('.')[0]}")
    for r in rep.get("readout_census", {}).get("roots", []):
        if r.get("searched"):
            print(f"  scanned {r['disk']:6s} {r['root']}  ({r['n_dirs_scanned']} dirs)")
        else:
            print(f"  UNSEARCHABLE {r['disk']:6s} {r['root']}  -> {r['why']}")
    found = rep.get("readout_census", {}).get("found", {})
    for st in [str(s) for s in rep["union_steps"]]:
        if st in found:
            f = found[st]
            print(f"    step{st:<7s} median_margin={f['median_margin']:.6f} "
                  f"n={f['n_pooled']} <- {f['disk']}:{Path(f['dir']).name}")
        else:
            print(f"    step{st:<7s} ABSENT")
    for nm, r in rep.get("per_grid", {}).items():
        print(f"  phi_{nm} = {r['phi']:.4f} -> {r['verdict']}   "
              f"[grid={r['steps']} span={r['span_used']} binding={r['binding_term']} "
              f"range_term={r['range_term']:.6f} slope_term={r['slope_term']:.6f} "
              f"D={r['damaged_range_D']:.6f}]")
    v = rep["verdict"]
    if v in VERDICT_SEVERITY:
        print(f"  COMBINE ({rep['combine_rule']})")
        print(f"  ==> GATE VERDICT = {v}"
              f"{'  [NON-PASS -- blocks the 244-2560 GPU-h ladder]' if v != 'PASS' else ''}")
    else:
        print(f"  ==> GATE VERDICT = {v}   [HARD ABORT, NON-PASS]")
        print(f"      why: {rep.get('why','')}")
        for nm, ms in (rep.get("missing_per_grid") or {}).items():
            if ms:
                print(f"      {nm} missing {len(ms)} of "
                      f"{len(rep['grids'][nm]['steps'])} arms: {ms}")
        print("      phi is UNDEFINED -- not small, not large. No interpolation, no "
              "shortened grid, no NaN.")


ABORT_EXIT_CODES = {"READOUT_ABSENT": 3, "PROTOCOL_VIOLATION": 4,
                    "FIELD_ASYMMETRY": 4, "DENOMINATOR_UNRESOLVED": 5,
                    "FLOOR_UNMEASURABLE": 5, "SHARD_SAMPLES_ARE_NOT_A_READOUT": 4}


def phi_budget(y_readout, damaged_range, span=READOUT_SPAN, steps=None):
    """The REVISION-2 decision statistic. Shape-agnostic, k-matched, max-guarded.

        phi = max( range(y over the read-out points), |OLS slope| * span ) / damaged_range

    y_readout must be median_margin at exactly G1_READOUT_STEPS, in that order.
    Returns the full audit dict -- both terms, which one bound, and the verdict.
    """
    steps = list(G1_READOUT_STEPS if steps is None else steps)
    if len(y_readout) != len(steps):
        sys.exit(f"FATAL: read-out has {len(y_readout)} points, prereg has {len(steps)}")
    if steps != G1_READOUT_STEPS:
        sys.exit(f"PROTOCOL_VIOLATION: read-out steps {steps} != prereg {G1_READOUT_STEPS}. "
                 "Extending n until the statistic crosses a threshold is the paperC "
                 "--max_steps error (readout_preregistration.not_a_decision_point).")
    if damaged_range is None or damaged_range <= 0:
        return {"verdict": "DENOMINATOR_UNRESOLVED",
                "why": "damaged_range <= 0 -> phi is UNDEFINED, not small. Blocks spend."}
    rng = max(y_readout) - min(y_readout)
    slope_term = abs(ols_slope(steps, y_readout)) * span
    num = max(rng, slope_term)
    phi = num / damaged_range
    v = "KILL" if phi >= PHI_KILL else ("PASS" if phi <= PHI_PASS else "NARROWED")
    return {"phi": phi, "verdict": v,
            "range_term": rng, "slope_term": slope_term,
            "binding_term": "range" if rng >= slope_term else "slope",
            "span_used": span, "damaged_range": damaged_range,
            "phi_kill_threshold": PHI_KILL, "phi_pass_threshold": PHI_PASS}


def load_norm_lens(donor: str) -> dict:
    nl = {}
    for t in TASKS:
        p = ROOT / donor / f"per_example_{t}.jsonl"
        d = {}
        for line in open(p):
            o = json.loads(line)
            if "norm_lens" not in o:
                sys.exit(f"FATAL: donor {donor} lacks norm_lens in {t}; cannot transplant")
            d[o["item_id"]] = o["norm_lens"]
        nl[t] = d
    return nl


def margins(dirname: str, norm_lens: dict, force_transplant: bool = False) -> list[float]:
    """Pooled |margin| over the 6 core tasks. Asserts per-task n, not just n_nan."""
    out = []
    for t in TASKS:
        p = ROOT / dirname / f"per_example_{t}.jsonl"
        if not p.exists():
            sys.exit(f"FATAL: missing {p}")
        n = 0
        for line in open(p):
            o = json.loads(line)
            if ("norm_scores" in o and o["norm_scores"]) and not force_transplant:
                sc = o["norm_scores"]
            else:
                nl = norm_lens[t].get(o["item_id"])
                if nl is None:
                    sys.exit(f"FATAL: item_id {o['item_id']} of {t} absent from donor")
                os_ = o["option_scores"]
                # round to 6 dp to match the harness writer exactly
                sc = {k: (round(os_[k] / max(nl[k], 1), 6) if os_.get(k) is not None else None)
                      for k in os_}
            g = o["gold_letter"]
            oth = [v for k, v in sc.items() if k != g and v is not None]
            if sc.get(g) is None or not oth:
                continue
            out.append(abs(sc[g] - max(oth)))
            n += 1
        if n != EXPECTED_N[t]:
            sys.exit(f"PROTOCOL_VIOLATION: {dirname}/{t} scored {n}, expected {EXPECTED_N[t]}")
    if len(out) != EXPECTED_POOLED:
        sys.exit(f"PROTOCOL_VIOLATION: {dirname} pooled {len(out)} != {EXPECTED_POOLED}")
    return out


def metrics_of(ms: list[float]) -> dict:
    d = {"n": len(ms), "median_margin": statistics.median(ms)}
    for th in THRESHOLDS:
        d[f"frac_lt_{th}"] = sum(1 for m in ms if m < th) / len(ms)
    return d


def assert_shards(dirname: str) -> None:
    n = len(list((ROOT / dirname).glob("shard*of8.json")))
    if n != 8:
        sys.exit(f"PROTOCOL_VIOLATION: {dirname} has {n}/8 shards -- refusing partial merge")


def core6(dirname: str) -> float:
    s = json.load(open(ROOT / dirname / "summary.json"))["tasks"]
    for t in TASKS:
        if s[t].get("n_nan", 0) != 0:
            sys.exit(f"PROTOCOL_VIOLATION: {dirname}/{t} n_nan={s[t]['n_nan']}")
        if s[t]["n_scored"] != EXPECTED_N[t]:
            sys.exit(f"PROTOCOL_VIOLATION: {dirname}/{t} n_scored={s[t]['n_scored']}")
    return sum(s[t]["acc_norm"] for t in TASKS) / len(TASKS)


def _rank(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0.0] * len(v)
    i = 0
    while i < len(v):  # average ranks on ties
        j = i
        while j + 1 < len(v) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rk[idx[k]] = avg
        i = j + 1
    return rk


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return 0.0 if dx * dy == 0 else num / (dx * dy)


def exact_p(x, y):
    obs = abs(spearman(x, y))
    hits = tot = 0
    for perm in permutations(y):
        tot += 1
        if abs(spearman(x, list(perm))) >= obs - 1e-12:
            hits += 1
    return hits / tot


def ols_slope(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    den = sum((a - mx) ** 2 for a in x)
    if den == 0:
        sys.exit("FATAL: zero variance in x -- OLS slope undefined")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / den


def main():
    norm_lens = load_norm_lens(DONOR)

    # ---- validation: the transplant path must reproduce the donor's native numbers ----
    nat = metrics_of(margins(DONOR, norm_lens, force_transplant=False))
    tra = metrics_of(margins(DONOR, norm_lens, force_transplant=True))
    for k in nat:
        if nat[k] != tra[k]:
            sys.exit(f"FATAL: transplant != native on {k}: {nat[k]} vs {tra[k]}")
    print(f"[ok] norm_lens transplant validated exactly on donor {DONOR}")

    # ---- sigma_hat from the seed pair (damage AND heal step held constant) ----
    sp = {}
    for tag, d in SEED_PAIR:
        assert_shards(d)
        m = json.load(open(ROOT / d / "summary.json"))["meta"]
        sp[tag] = {"metrics": metrics_of(margins(d, norm_lens)), "ckpt": m["ckpt"],
                   "ckpt_step": m["ckpt_step"], "keep_front_layers": m["keep_front_layers"],
                   "n_fresh_layers": m["n_fresh_layers"], "core6": core6(d)}
    a, b = sp["s42"]["metrics"], sp["s1234"]["metrics"]
    if sp["s42"]["ckpt_step"] != sp["s1234"]["ckpt_step"] or \
       sp["s42"]["keep_front_layers"] != sp["s1234"]["keep_front_layers"]:
        sys.exit("FATAL: seed pair does not hold damage+budget constant; sigma_hat inadmissible")
    k = len(SEED_PAIR)
    div = E_RANGE_OVER_SIGMA[k]
    sigma = {m: abs(a[m] - b[m]) / div for m in METRICS}
    for m, s in sigma.items():
        if s == 0:
            sys.exit(f"FLOOR_UNMEASURABLE on {m}: sigma_hat==0 means the pair is not a real "
                     f"nuisance contrast. This is NOT a pass.")
    print(f"[ok] sigma_hat (k={k}, divisor {div}): " +
          ", ".join(f"{m}={sigma[m]:.6f}" for m in METRICS))

    # ---- ladder ----
    rung, cor = {}, {}
    for label, d, keep, step in LADDER:
        assert_shards(d)
        cor[label] = core6(d)
        rung[label] = metrics_of(margins(d, norm_lens))
    labels = [l for l, _, _, _ in LADDER]
    order = sorted(labels, key=lambda l: cor[l])  # core6-ascending

    per_metric = {}
    for m in METRICS:
        y = [rung[l][m] for l in labels]
        rng = max(y) - min(y)
        gaps = [abs(rung[order[i + 1]][m] - rung[order[i]][m]) for i in range(len(order) - 1)]
        per_metric[m] = {
            "sigma_hat": sigma[m],
            "full_ladder_range": rng,
            "R_range_over_sigma": rng / sigma[m],
            "adjacent_gaps_core6_ordered": gaps,
            "two_sigma": 2 * sigma[m],
            "n_adjacent_gaps_clearing_2sigma": sum(1 for g in gaps if g > 2 * sigma[m]),
            "n_adjacent_gaps": len(gaps),
            "spearman_core6": spearman([cor[l] for l in labels], y),
            "exact_p_two_sided": exact_p([cor[l] for l in labels], y),
        }

    # ---- clause-5 denominator: damaged rungs only ----
    dam = [(l, d, kp, st) for l, d, kp, st in LADDER if kp is not None]
    dl = [l for l, _, _, _ in dam]
    dmed = [rung[l]["median_margin"] for l in dl]
    dstep = [st for _, _, _, st in dam]
    dcore = [cor[l] for l in dl]
    dam_range = max(dmed) - min(dmed)
    guard = FLOOR_SAFETY_FACTOR * sigma["median_margin"]
    denom_ok = dam_range >= guard

    # E[range of k]/sigma at the numerator's own k. The revision-2 numerator is a RANGE over
    # k=5 read-out points and the denominator is a RANGE over k=5 damaged rungs, so the two
    # sides are k-matched and the noise-only expectation of phi is E[range 5]*sigma/dam_range.
    k_readout = len(G1_READOUT_STEPS)
    div_readout = E_RANGE_OVER_SIGMA[k_readout]
    phi_noise_only = div_readout * sigma["median_margin"] / dam_range if dam_range > 0 else None

    clause5 = {
        "revision": "2 (2026-08-14, PRE-DATA). Revision 1 was refuted 3/3 by the adversarial "
                    "pass: it rescaled the slope to the DAMAGED span 116500 instead of the "
                    "READ-OUT's own span 175000 (decidability), and a slope-only statistic "
                    "lets a non-monotone budget response worth 94% of the damaged range pass "
                    "with phi=0.19 (falsifiability/affordability).",
        "primary_metric": "median_margin",
        "damaged_rungs": dl,
        "damaged_range_median_margin": dam_range,
        "damaged_heal_step_span": max(dstep) - min(dstep),
        "G1_readout_steps": G1_READOUT_STEPS,
        "readout_span_USED_IN_PHI": READOUT_SPAN,
        "span_note": "phi uses the READ-OUT's own span 175000. The damaged ladder's 116500 "
                     "is retained above only as a descriptive property of the comparator and "
                     "MUST NOT enter phi (ratio 175000/116500 = 1.5021).",
        "phi_definition": "phi = max( range(median_margin over the 5 prereg read-out steps), "
                          "|OLS slope on heal_step| * 175000 ) / damaged_range",
        "phi_is_shape_agnostic": "the range term assumes nothing about the functional form of "
                                 "the budget response; max() with the slope term can only "
                                 "raise phi, never lower it (sup ratio "
                                 f"{SLOPE_TERM_SUP_RATIO} on this fixed x-grid), so max() is "
                                 "strictly more conservative than range-alone",
        "k_matching": {
            "numerator_k": k_readout, "denominator_k": len(dl),
            "E_range_over_sigma_at_k": div_readout,
            "phi_expected_under_pure_noise": phi_noise_only,
            "note": "numerator and denominator are both RANGES at k=5, so E[range k]/sigma "
                    "cancels between them. phi_expected_under_pure_noise is the value phi "
                    "takes if the budget response is pure run-to-run noise; it must sit far "
                    "below the PASS line or the PASS branch is unfalsifiable.",
        },
        "denominator_guard_threshold": guard,
        "denominator_guard_basis": f"{FLOOR_SAFETY_FACTOR}*sigma_hat(median_margin)",
        "denominator_admissible": denom_ok,
        "denominator_verdict": "OK" if denom_ok else "DENOMINATOR_UNRESOLVED",
        "phi_kill_threshold": PHI_KILL,
        "phi_pass_threshold": PHI_PASS,
        "excursion_kill_absolute": PHI_KILL * dam_range,
        "excursion_pass_absolute": PHI_PASS * dam_range,
        "excursion_kill_in_sigma_hat": PHI_KILL * dam_range / sigma["median_margin"],
        "excursion_pass_in_sigma_hat": PHI_PASS * dam_range / sigma["median_margin"],
        # If the RANGE term happens to be the binding one these are not the operative
        # thresholds; they are the slope-term equivalents at the read-out's own span.
        "beta_budget_kill_per_step_at_readout_span": PHI_KILL * dam_range / READOUT_SPAN,
        "beta_budget_pass_per_step_at_readout_span": PHI_PASS * dam_range / READOUT_SPAN,
        "S_damage_ols": ols_slope(dcore, dmed),
        "spearman_core6_heal_steps_MANDATORY_CODISCLOSURE": spearman(dcore, dstep),
        "spearman_core6_layers_kept": spearman(dcore, [kp for _, _, kp, _ in dam]),
    }

    # ---- adversarial precedent, recomputed under REVISION 2 (0 GPU) ----
    # The only fixed-damage budget ladder that exists anywhere in this project: the Qwen
    # f12k2/14L cell. Its span (198000) is its own, not 175000 -- the whole point of the
    # decidability fix is that a statistic uses the span it was measured over.
    qpath = Path("proposal/backlog/B04-eval-fragility-incubator/evidence/"
                 "B04_Qwen_6rung_bs16_analysis.json")
    if qpath.exists():
        qs = json.load(open(qpath))["fragility_stats"]
        qcell = [("f12k2 @ step2000 (14L)", 2000), ("f12k2 @ step20000 (14L)", 20000),
                 ("f12k2 @ step200000 (14L)", 200000)]
        qx = [s for _, s in qcell]
        qy = [qs[k]["median_margin"] for k, _ in qcell]
        qspan = max(qx) - min(qx)
        qrng = max(qy) - min(qy)
        qslope = abs(ols_slope(qx, qy)) * qspan
        qphi = max(qrng, qslope) / dam_range
        clause5["adversarial_precedent_qwen_f12k2_14L"] = {
            "steps": qx, "median_margin": qy,
            "own_span": qspan, "range_term": qrng, "slope_term": qslope,
            "binding_term": "range" if qrng >= qslope else "slope",
            "phi_revision2": qphi,
            "verdict_revision2": "KILL" if qphi >= PHI_KILL else
                                 ("PASS" if qphi <= PHI_PASS else "NARROWED"),
            "non_monotone": qy[1] < qy[0],
            "note": "This is the single most relevant empirical precedent and revision 2 "
                    "scores it KILL, as revision 1 also did (phi 0.8916 with the wrong span). "
                    "It is non-monotone in budget, which is exactly the shape a slope-only "
                    "statistic under-reads.",
        }
    if not denom_ok:
        print("DENOMINATOR_UNRESOLVED: damaged range below the floor guard; phi is UNDEFINED "
              "(not small, not large). Blocks the family-ladder spend exactly as a KILL would.")

    # ---- REVISION 3: the actual read-out path. Loads y from disk and computes phi. -------
    # This is the part that did not exist before 2026-08-16: revision 2's phi_budget() was
    # only ever called from selftest_phi() on hand-written vectors, so no on-disk read-out
    # ever reached the decision statistic.
    clause5_r3 = clause5_revision3(dam_range, sigma["median_margin"],
                                   clause5["spearman_core6_heal_steps_MANDATORY_CODISCLOSURE"])

    out = {
        "gate": "B04 G0 floor-first (0 GPU)",
        "date": "2026-08-14",
        "gpu_used": "none",
        "arch": "sm_100 (wzc1, LOCAL/.212). Comparator provenance: paperB/SEEDVAR_KEEP14_VERDICT.md",
        "prereg_note": "PRIMARY = median_margin, fixed BEFORE G1 arms exist. See GATE_DESIGN.md sec 1.",
        "ladder_identity_warning": "keep12 rung here is step111500 (wzc1), NOT step124000 "
                                   "(zwfy6, evidence/B04_6rung_bs16_analysis.json). Quoting either "
                                   "Spearman(core6, heal_steps) requires naming its ladder.",
        "sigma_hat_source": {"pair": [d for _, d in SEED_PAIR], "k": k, "divisor": div,
                             "meta": {t: {kk: vv for kk, vv in sp[t].items() if kk != "metrics"}
                                      for t in sp}},
        "seed_pair_metrics": {t: sp[t]["metrics"] for t in sp},
        "core6": cor,
        "fragility_stats": rung,
        "per_metric_floor_analysis": per_metric,
        "clause5_budget_discrimination": clause5,
        # Revision 2's key above is UNCHANGED (provenance). Revision 3 is a NEW key, so the
        # two are auditable side by side and no prior number is overwritten.
        "clause5_budget_discrimination_revision3": clause5_r3,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[ok] wrote {OUT}")

    print(f"\n{'metric':17s} {'sigma_hat':>10s} {'range':>10s} {'R':>7s} {'adj>2sig':>9s} "
          f"{'rho':>8s} {'exact_p':>8s}")
    for m in METRICS:
        d = per_metric[m]
        print(f"{m:17s} {d['sigma_hat']:10.6f} {d['full_ladder_range']:10.6f} "
              f"{d['R_range_over_sigma']:7.2f} "
              f"{d['n_adjacent_gaps_clearing_2sigma']:4d}/{d['n_adjacent_gaps']:<4d} "
              f"{d['spearman_core6']:+8.4f} {d['exact_p_two_sided']:8.4f}")
    print(f"\nclause 5 (REVISION 2) denominator: range={dam_range:.6f} guard={guard:.6f} -> "
          f"{clause5['denominator_verdict']}")
    print(f"  phi = max( range(median_margin over steps {G1_READOUT_STEPS}), "
          f"|beta|*{READOUT_SPAN} ) / {dam_range:.6f}")
    print(f"  KILL if phi >= {PHI_KILL} <=> excursion >= "
          f"{clause5['excursion_kill_absolute']:.6f} "
          f"({clause5['excursion_kill_in_sigma_hat']:.1f} sigma_hat) "
          f"<=> monotone |beta| >= {clause5['beta_budget_kill_per_step_at_readout_span']:.4e} /step")
    print(f"  PASS if phi <= {PHI_PASS} <=> excursion <= "
          f"{clause5['excursion_pass_absolute']:.6f} "
          f"({clause5['excursion_pass_in_sigma_hat']:.1f} sigma_hat) "
          f"<=> monotone |beta| <= {clause5['beta_budget_pass_per_step_at_readout_span']:.4e} /step")
    if phi_noise_only is not None:
        print(f"  phi under pure noise (k={k_readout}) = {phi_noise_only:.4f} -- "
              f"{PHI_PASS / phi_noise_only:.1f}x below the PASS line, so PASS is not free")
    ap = clause5.get("adversarial_precedent_qwen_f12k2_14L")
    if ap:
        print(f"  adversarial precedent (Qwen f12k2/14L, own span {ap['own_span']}): "
              f"phi={ap['phi_revision2']:.4f} -> {ap['verdict_revision2']} "
              f"(binding term: {ap['binding_term']})")
    print(f"  MANDATORY co-disclosure Spearman(core6, heal_steps) = "
          f"{clause5['spearman_core6_heal_steps_MANDATORY_CODISCLOSURE']:+.4f} (wzc1 ladder)")

    # ---- REVISION 3 read-out: report and set the process exit code from the verdict -------
    print_clause5_revision3(clause5_r3)
    v3 = clause5_r3["verdict"]
    if v3 not in VERDICT_SEVERITY:
        # A hard abort must be visible to a shell caller, not only in the JSON. It is a
        # NON-PASS, so it must NOT exit 0.
        sys.exit(ABORT_EXIT_CODES.get(v3, 6))


def selftest_readout_fixture() -> None:
    """SELFTEST 2 (MAIN, 2026-08-16): exercise the REAL disk-reading path end to end.

    The pre-existing selftest_phi() proves the FUNCTION is falsifiable; it says nothing
    about whether the GATE can run, because it feeds hand-written y-vectors. This one
    builds a synthetic fixture tree of fake eval dirs and drives clause5_revision3()
    against it, so that "selftest passes" says something about the pipeline.

    Nothing here points at the live olmo2_downstream_results/ -- the fixture is a freshly
    written tree in a throwaway tmpdir and is removed afterwards. It is NEVER a symlink to
    a live evidence dir: this project's integrity checkers are WRITERS
    (memory/repo-checkers-are-writers-not-probes.md), so a symlinked fixture could be
    mutated in place.
    """
    import shutil
    import tempfile

    D = 0.02181999999999995
    SIG = 0.0005405884438142497
    RHO = 0.6668859288553503

    def write_dir(root: Path, name: str, step: int, mm: float, *, n_shards=8,
                  shard_files=8, keep=14, fresh=2, add_bos=False, ckpt=None,
                  tasks=None, drop_field=None, row_delta=0, n_scored_delta=0, n_nan=0,
                  base_model=None):
        """Write a minimal-but-real eval dir: 6 per_example jsonl + shards + summary.json.

        Rows are constructed so that the pooled MEDIAN of |gold - max(other)| is exactly
        `mm`: every row gets the same margin, so the median is that margin regardless of n.
        """
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        tasks = TASKS if tasks is None else tasks
        for i in range(shard_files):
            (d / f"shard{i}of8.json").write_text("{}\n")
        tj = {}
        for t in tasks:
            n = EXPECTED_N.get(t, 100)
            with open(d / f"per_example_{t}.jsonl", "w") as f:
                for j in range(n + (row_delta if t == tasks[0] else 0)):
                    row = {"item_id": j, "gold_letter": "A",
                           "norm_scores": {"A": round(mm, 6), "B": 0.0},
                           "norm_lens": {"A": 10, "B": 10},
                           "option_scores": {"A": round(mm * 10, 6), "B": 0.0},
                           "nan": False}
                    if drop_field and t == tasks[-1]:
                        row.pop(drop_field, None)
                    f.write(json.dumps(row) + "\n")
            tj[t] = {"n": n, "n_scored": n + (n_scored_delta if t == tasks[0] else 0),
                     "n_nan": n_nan if t == tasks[0] else 0, "acc": 0.3, "acc_norm": 0.4}
        json.dump({"output_name": name, "n_shards": n_shards, "add_bos": add_bos,
                   "meta": {"mode": "pruned", "keep_front_layers": keep,
                            "n_fresh_layers": fresh, "num_hidden_layers": 16,
                            "ckpt_step": step,
                            "ckpt": ckpt or f"{READOUT_CKPT_DIR}/step{step}.pt",
                            "base_model": base_model or READOUT_BASE_MODEL,
                            "add_bos": add_bos},
                   "tasks": tj}, open(d / "summary.json", "w"))
        return d

    tmp = Path(tempfile.mkdtemp(prefix="b04_readout_fixture_"))
    assert not tmp.is_symlink(), "fixture root must be a real dir, never a symlink"
    ok = 0
    try:
        def run(sub, **kw):
            return clause5_revision3(D, SIG, RHO, roots=[("fixture", tmp / sub)], **kw)

        # --- F1: only step200000 present == TODAY's real disk state -> READOUT_ABSENT -----
        r = write_dir(tmp / "f1", "arm_step200000", 200000, 0.108500)
        rep = run("f1")
        assert rep["verdict"] == "READOUT_ABSENT", rep["verdict"]
        assert sorted(rep["missing_per_grid"]["GRID_I"]) == [100000, 128000, 153500, 175000]
        assert sorted(rep["missing_per_grid"]["GRID_W"]) == [25000, 50000, 100000, 128000]
        assert "per_grid" not in rep, "phi must not be computed with a hole"
        assert rep["blocks_spend"] is True
        print(f"  [ok] F1 one-arm-only        -> {rep['verdict']}, GRID_I missing 4, "
              f"GRID_W missing 4, no phi computed"); ok += 1

        # --- F2: all 7 arms, gentle ramp -> phi_I PASS, phi_W NARROWED, combined NARROWED -
        Y2 = {25000: 0.100000, 50000: 0.101000, 100000: 0.102000, 128000: 0.103000,
              153500: 0.104000, 175000: 0.105000, 200000: 0.108500}
        for st, mm in Y2.items():
            write_dir(tmp / "f2", f"arm_step{st}", st, mm)
        rep = run("f2")
        assert rep["verdict"] == "NARROWED", rep
        pi, pw = rep["per_grid"]["GRID_I"], rep["per_grid"]["GRID_W"]
        assert pi["verdict"] == "PASS" and pw["verdict"] == "NARROWED", (pi, pw)
        assert abs(pi["phi"] - 0.2978918423464721) < 1e-12, pi["phi"]
        assert abs(pw["phi"] - 0.38955087076077055) < 1e-12, pw["phi"]
        assert pi["span_used"] == 100000 and pw["span_used"] == 175000
        print(f"  [ok] F2 full 7-arm ramp     -> phi_I={pi['phi']:.4f} {pi['verdict']} @S=100000, "
              f"phi_W={pw['phi']:.4f} {pw['verdict']} @S=175000, combined={rep['verdict']} "
              "(the SECONDARY grid carries it -- the combine rule is load-bearing)"); ok += 1

        # --- F3: steep early compression -> phi_I NARROWED, phi_W KILL, combined KILL -----
        Y3 = {25000: 0.090000, 50000: 0.093000, 100000: 0.096000, 128000: 0.100000,
              153500: 0.103000, 175000: 0.105000, 200000: 0.108500}
        for st, mm in Y3.items():
            write_dir(tmp / "f3", f"arm_step{st}", st, mm)
        rep = run("f3")
        assert rep["verdict"] == "KILL", rep
        assert rep["per_grid"]["GRID_I"]["verdict"] == "NARROWED"
        assert rep["per_grid"]["GRID_W"]["verdict"] == "KILL"
        print(f"  [ok] F3 steep early         -> phi_I={rep['per_grid']['GRID_I']['phi']:.4f} "
              f"NARROWED, phi_W={rep['per_grid']['GRID_W']['phi']:.4f} KILL, "
              f"combined={rep['verdict']}"); ok += 1

        # --- F4: flat read-out -> PASS on both grids --------------------------------------
        Y4 = {s: 0.108500 for s in READOUT_UNION_STEPS}
        Y4[200000] = 0.108900
        for st, mm in Y4.items():
            write_dir(tmp / "f4", f"arm_step{st}", st, mm)
        rep = run("f4")
        assert rep["verdict"] == "PASS", rep
        assert rep["blocks_spend"] is False
        print(f"  [ok] F4 flat read-out       -> phi_I={rep['per_grid']['GRID_I']['phi']:.4f}, "
              f"phi_W={rep['per_grid']['GRID_W']['phi']:.4f}, combined=PASS "
              "(all three verdicts reachable THROUGH DISK, not only in arithmetic)"); ok += 1

        # --- F5: 5 of 8 shard files -- the historical silent-merge disaster ---------------
        for st, mm in Y4.items():
            write_dir(tmp / "f5", f"arm_step{st}", st, mm,
                      shard_files=(5 if st == 153500 else 8))
        rep = run("f5")
        assert rep["verdict"] == "PROTOCOL_VIOLATION" and "5/8 shard" in rep["why"], rep
        print(f"  [ok] F5 5/8 shards          -> {rep['verdict']} ({rep['why'][:58]}...)"); ok += 1

        # --- F6: 8 shard files but summary.n_shards=5 (partial merge that LOOKS complete) -
        for st, mm in Y4.items():
            write_dir(tmp / "f6", f"arm_step{st}", st, mm, n_shards=(5 if st == 175000 else 8))
        rep = run("f6")
        assert rep["verdict"] == "PROTOCOL_VIOLATION" and "PARTIAL MERGE" in rep["why"], rep
        print(f"  [ok] F6 n_shards=5 in summary -> {rep['verdict']} (PARTIAL MERGE)"); ok += 1

        # --- F7: per-task n_scored short by 100 ------------------------------------------
        for st, mm in Y4.items():
            write_dir(tmp / "f7", f"arm_step{st}", st, mm,
                      n_scored_delta=(-100 if st == 128000 else 0))
        rep = run("f7")
        assert rep["verdict"] == "PROTOCOL_VIOLATION" and "n_scored" in rep["why"], rep
        print(f"  [ok] F7 n_scored short 100  -> {rep['verdict']} ({rep['why'][:58]}...)"); ok += 1

        # --- F8: per_example truncated while summary STILL claims complete ----------------
        # The dangerous shape: summary.json passes every check, and only the row count of the
        # file phi is actually computed from is short. n_scored_delta is deliberately 0 here.
        for st, mm in Y4.items():
            write_dir(tmp / "f8", f"arm_step{st}", st, mm,
                      row_delta=(-100 if st == 100000 else 0))
        rep = run("f8")
        assert rep["verdict"] == "PROTOCOL_VIOLATION" and "rows" in rep["why"], rep
        print(f"  [ok] F8 per_example truncated, summary clean -> {rep['verdict']} "
              f"({rep['why'][:52]}...)"); ok += 1

        # --- F9: norm_scores stripped -> FIELD_ASYMMETRY ---------------------------------
        for st, mm in Y4.items():
            write_dir(tmp / "f9", f"arm_step{st}", st, mm,
                      drop_field=("norm_scores" if st == 50000 else None))
        rep = run("f9")
        assert rep["verdict"] == "FIELD_ASYMMETRY", rep
        print(f"  [ok] F9 norm_scores stripped -> {rep['verdict']}"); ok += 1

        # --- F10: right dir NAME, WRONG checkpoint -> the arm reads as ABSENT, never used -
        # This is the failure that would otherwise feed a wrong y into phi unnoticed.
        for st, mm in Y4.items():
            write_dir(tmp / "f10", f"arm_step{st}", st, mm,
                      ckpt=("outputs/olmo2_probe2_7B_keep14fresh2/step153500.pt"
                            if st == 153500 else None))
        rep = run("f10")
        assert rep["verdict"] == "READOUT_ABSENT", rep
        assert rep["missing_per_grid"]["GRID_I"] == [153500], rep["missing_per_grid"]
        print(f"  [ok] F10 wrong ckpt, right name -> {rep['verdict']} (GRID_I missing "
              "[153500]; the dir is NOT silently accepted)"); ok += 1

        # --- F11: ckpt_step disagrees with the dir's own name ----------------------------
        # Identification is by meta.ckpt_step, never by the dir name. The dir NAMED 175000
        # declares ckpt_step 200000, so 175000 must read as ABSENT. Its median_margin is set
        # equal to the true 200000 arm's so the only defect under test is the mislabel (a
        # DISAGREEING duplicate is a different fault, covered by F16).
        for st, mm in Y4.items():
            write_dir(tmp / "f11", f"arm_step{st}", st,
                      (Y4[200000] if st == 175000 else mm))
        p = tmp / "f11" / "arm_step175000" / "summary.json"
        j = json.load(open(p))
        j["meta"]["ckpt_step"] = 200000
        j["meta"]["ckpt"] = f"{READOUT_CKPT_DIR}/step200000.pt"
        json.dump(j, open(p, "w"))
        rep = run("f11")
        assert rep["verdict"] == "READOUT_ABSENT", rep
        assert rep["missing_per_grid"]["GRID_I"] == [175000], rep["missing_per_grid"]
        print(f"  [ok] F11 ckpt_step != dir name -> {rep['verdict']} (GRID_I missing "
              "[175000]; identification is by meta, not by name)"); ok += 1

        # --- F12: damage NOT held fixed (keep/fresh drift) -------------------------------
        for st, mm in Y4.items():
            write_dir(tmp / "f12", f"arm_step{st}", st, mm, keep=(12 if st == 128000 else 14))
        rep = run("f12")
        assert rep["verdict"] == "PROTOCOL_VIOLATION" and "keep/fresh" in rep["why"], rep
        print(f"  [ok] F12 keep/fresh drift   -> {rep['verdict']} (damage not held fixed)"); ok += 1

        # --- F13: add_bos True -----------------------------------------------------------
        for st, mm in Y4.items():
            write_dir(tmp / "f13", f"arm_step{st}", st, mm, add_bos=(st == 25000))
        rep = run("f13")
        assert rep["verdict"] == "PROTOCOL_VIOLATION" and "add_bos" in rep["why"], rep
        print(f"  [ok] F13 add_bos=True       -> {rep['verdict']}"); ok += 1

        # --- F14: n_nan != 0 -------------------------------------------------------------
        for st, mm in Y4.items():
            write_dir(tmp / "f14", f"arm_step{st}", st, mm, n_nan=(3 if st == 200000 else 0))
        rep = run("f14")
        assert rep["verdict"] == "PROTOCOL_VIOLATION" and "n_nan" in rep["why"], rep
        print(f"  [ok] F14 n_nan=3            -> {rep['verdict']}"); ok += 1

        # --- F15: the know5 dir shape (right ckpt, DIFFERENT task set) is not a candidate -
        write_dir(tmp / "f15", "arm_step100000_know", 100000, 0.5,
                  tasks=["mmlu", "boolq", "social_iqa"])
        for st, mm in Y4.items():
            write_dir(tmp / "f15", f"arm_step{st}", st, mm)
        rep = run("f15")
        assert rep["verdict"] == "PASS", rep
        assert Path(rep["readout_census"]["found"]["100000"]["dir"]).name == "arm_step100000"
        print(f"  [ok] F15 know5-shaped decoy -> ignored, core6 dir used, "
              f"combined={rep['verdict']}"); ok += 1

        # --- F16: two core6 dirs for the same step that DISAGREE -> refuse to choose ------
        for st, mm in Y4.items():
            write_dir(tmp / "f16", f"arm_step{st}", st, mm)
        write_dir(tmp / "f16", "arm_step100000_dup", 100000, 0.099999)
        rep = run("f16")
        assert rep["verdict"] == "PROTOCOL_VIOLATION" and "DISAGREE" in rep["why"], rep
        print(f"  [ok] F16 duplicate arms disagree -> {rep['verdict']} (refuses to pick)"); ok += 1

        # --- F17: wrong base_model -> PROTOCOL_VIOLATION ----------------------------------
        for st, mm in Y4.items():
            write_dir(tmp / "f17", f"arm_step{st}", st, mm,
                      base_model=("../models/OLMo-2-1124-13B" if st == 100000 else None))
        rep = run("f17")
        assert rep["verdict"] == "PROTOCOL_VIOLATION" and "base_model" in rep["why"], rep
        print(f"  [ok] F17 wrong base_model   -> {rep['verdict']}"); ok += 1

        # --- F18: an UNMOUNTED root is reported UNSEARCHABLE, never as "empty" ------------
        rep = clause5_revision3(D, SIG, RHO, roots=[
            ("fixture", tmp / "f4"),
            ("nosuchdisk", Path("/apdcephfs_no_such_disk_b04/olmo2_downstream_results"))])
        assert rep["verdict"] == "PASS", rep
        roots = {r["disk"]: r for r in rep["readout_census"]["roots"]}
        assert roots["nosuchdisk"]["searched"] is False
        assert "NOT established" in roots["nosuchdisk"]["why"]
        print(f"  [ok] F18 unmounted root     -> reported searched=False with a reason, "
              "not treated as empty"); ok += 1

        # --- F19: a degenerate denominator aborts before any disk read --------------------
        rep = clause5_revision3(0.0, SIG, RHO, roots=[("fixture", tmp / "f4")])
        assert rep["verdict"] == "DENOMINATOR_UNRESOLVED", rep
        rep = clause5_revision3(0.001, SIG, RHO, roots=[("fixture", tmp / "f4")])
        assert rep["verdict"] == "DENOMINATOR_UNRESOLVED", rep   # 0.001 < 6*sigma = 0.003244
        rep = clause5_revision3(D, 0.0, RHO, roots=[("fixture", tmp / "f4")])
        assert rep["verdict"] == "FLOOR_UNMEASURABLE", rep
        print(f"  [ok] F19 D<=0 / D<6sigma / sigma==0 -> DENOMINATOR_UNRESOLVED x2, "
              "FLOOR_UNMEASURABLE"); ok += 1

        # --- F20: a shortened or extended grid can never reach phi ------------------------
        for bad in ([100000, 128000, 153500, 175000],
                    [100000, 128000, 153500, 175000, 190000, 200000]):
            try:
                phi_budget_grid([0.1] * len(bad), D, "GRID_I", GRID_I, SPAN_I,
                                evaluated_steps=bad)
            except GateAbort as e:
                assert e.code == "PROTOCOL_VIOLATION", e.code
            else:
                sys.exit("FATAL: a mismatched step set did NOT abort")
        try:
            phi_budget_grid([0.1, 0.1, None, 0.1, 0.1], D, "GRID_I", GRID_I, SPAN_I)
        except GateAbort as e:
            assert e.code == "READOUT_ABSENT", e.code
        else:
            sys.exit("FATAL: a None in y did NOT abort")
        try:
            phi_budget_grid([0.1, 0.1, float("nan"), 0.1, 0.1], D, "GRID_I", GRID_I, SPAN_I)
        except GateAbort as e:
            assert e.code == "READOUT_ABSENT", e.code
        else:
            sys.exit("FATAL: a NaN in y did NOT abort")
        print(f"  [ok] F20 k=4 / k=6 / None / NaN -> all abort; no shortened grid, no NaN "
              "flows onward"); ok += 1

        # --- F21: the combine rule is never laxer than either grid alone ------------------
        import random
        rnd = random.Random(20260816)
        worse = 0
        for _ in range(20000):
            y = {s: 0.09 + rnd.random() * 0.03 for s in READOUT_UNION_STEPS}
            ri = phi_budget_grid([y[s] for s in GRID_I], D, "GRID_I", GRID_I, SPAN_I)
            rw = phi_budget_grid([y[s] for s in GRID_W], D, "GRID_W", GRID_W, SPAN_W)
            comb = combine_verdicts({"GRID_I": ri, "GRID_W": rw})
            if VERDICT_SEVERITY[comb] < max(VERDICT_SEVERITY[ri["verdict"]],
                                            VERDICT_SEVERITY[rw["verdict"]]):
                worse += 1
        assert worse == 0, worse
        print(f"  [ok] F21 combine monotone   -> 0/20000 random shapes where combined was "
              "less severe than either grid alone"); ok += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[ok] fixture selftest: {ok}/21 checks passed -- the DISK-READING path is "
          "exercised end to end, not only the arithmetic")


def selftest_phi(dam_range: float = 0.02181999999999995) -> None:
    """0-GPU pre-data check that the revision-2 statistic is falsifiable in BOTH directions.

    Run: python .../analyze_b04_wzc1_floor.py --selftest
    These y-vectors are HYPOTHETICAL, written 2026-08-14 before the 4 arms are evaluated.
    They exist to prove the gate can return each verdict, not to predict which it will.
    """
    cases = [
        ("KILL   monotone compression",      [0.0902, 0.0951, 0.1005, 0.1042, 0.1085]),
        ("KILL   non-monotone V (rev-1 hole)", [0.1085, 0.0905, 0.0885, 0.0975, 0.1090]),
        ("NARROW mid excursion",             [0.1000, 0.1030, 0.1055, 0.1070, 0.1085]),
        ("PASS   early convergence",         [0.1062, 0.1071, 0.1078, 0.1081, 0.1085]),
        ("PASS   pure-noise-scale wobble",   [0.10850, 0.10796, 0.10904, 0.10812, 0.10850]),
    ]
    print(f"phi = max(range, |beta|*{READOUT_SPAN}) / {dam_range:.6f};  "
          f"KILL>={PHI_KILL}  PASS<={PHI_PASS}")
    seen = set()
    for tag, y in cases:
        r = phi_budget(y, dam_range)
        seen.add(r["verdict"])
        print(f"  {tag:34s} range={r['range_term']:.6f} slope_term={r['slope_term']:.6f} "
              f"bind={r['binding_term']:5s} phi={r['phi']:.4f} -> {r['verdict']}")
    for v in ("KILL", "NARROWED", "PASS"):
        if v not in seen:
            sys.exit(f"FATAL: no constructed case reaches {v}; the gate is not falsifiable "
                     f"in that direction")
    print("[ok] all three verdicts are reachable -> the gate is falsifiable both ways")

    # ---- the documented single-number boundaries must be SHAPE-SAFE -------------------
    # Added by MAIN 2026-08-15 (0 GPU, PRE-DATA). The prose PASS boundary was originally
    # derived from the range term alone, but phi = max(range, |beta|*span), so a step-shaped
    # read-out can be slope-dominated and land in NARROWED at a min the range-only arithmetic
    # calls PASS. Two adversarial lenses proposed 0.102922 and 0.102923 for the SAME threshold;
    # the truncated one FAILS (phi=0.300023 > PHI_PASS=0.30). Because the rule has the form
    # `min(y) >= T`, T must be rounded UP -- truncation lands below the exact boundary
    # 0.1029224187071361 and reproduces the very defect this corrects. Pinned so it cannot rot.
    MAX_MEASURED = 0.108500
    PASS_MIN = 0.102923   # ceil6(MAX_MEASURED - PHI_PASS*dam_range/SLOPE_TERM_SUP_RATIO)
    KILL_MIN = 0.095408   # MAX_MEASURED - PHI_KILL*dam_range  (max() >= range => sufficient)
    for mn, want in ((PASS_MIN, "PASS"), (0.102922, "NARROWED"), (0.101954, "NARROWED")):
        # step shape = the shape maximising |slope| at fixed (min, max)
        got = phi_budget([mn, mn, mn, MAX_MEASURED, MAX_MEASURED], dam_range)
        if got["verdict"] != want:
            sys.exit(f"FATAL: single-number PASS boundary is not shape-safe: at min={mn} the "
                     f"worst shape gives phi={got['phi']:.6f} -> {got['verdict']}, expected "
                     f"{want}. Fix the prose in STATUS.json + GATE_DESIGN.md sec 3.2.")
    kr = phi_budget([KILL_MIN, KILL_MIN, KILL_MIN, MAX_MEASURED, MAX_MEASURED], dam_range)
    if kr["verdict"] != "KILL":
        sys.exit(f"FATAL: single-number KILL boundary is not sufficient: phi={kr['phi']:.6f} "
                 f"-> {kr['verdict']}")
    print(f"[ok] single-number boundaries shape-safe: KILL min<={KILL_MIN:.6f} "
          f"(phi={kr['phi']:.6f}), PASS min>={PASS_MIN:.6f}; the range-only 0.101954 and the "
          f"6dp truncation 0.102922 are both correctly rejected as NARROWED")



if __name__ == "__main__":
    if "--selftest" in sys.argv:
        # SELFTEST 1 (arithmetic, revision 2, hand-written y-vectors) -- unchanged.
        selftest_phi()
        # SELFTEST 2 (pipeline, revision 3, synthetic on-disk fixture) -- added 2026-08-16.
        # Both must pass. The first says the function is falsifiable; only the second says
        # anything about whether the gate can RUN.
        print("\n--- fixture selftest: the REAL disk-reading path, on a synthetic tree ---")
        selftest_readout_fixture()
    elif "--readout-only" in sys.argv:
        # Compute ONLY clause 5 revision 3, from the constants already banked in the
        # evidence JSON. Does not rewrite the evidence JSON. Exits non-zero on any abort.
        ev = json.load(open(OUT))
        c5 = ev["clause5_budget_discrimination"]
        rep = clause5_revision3(
            c5["damaged_range_median_margin"],
            ev["per_metric_floor_analysis"]["median_margin"]["sigma_hat"],
            c5["spearman_core6_heal_steps_MANDATORY_CODISCLOSURE"])
        print_clause5_revision3(rep)
        if rep["verdict"] not in VERDICT_SEVERITY:
            sys.exit(ABORT_EXIT_CODES.get(rep["verdict"], 6))
    else:
        main()
