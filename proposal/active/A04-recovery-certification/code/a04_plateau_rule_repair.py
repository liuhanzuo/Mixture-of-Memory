#!/usr/bin/env python3
"""A04 PLATEAU(T) repair — diagnose both readings, then define and exercise a
dimensionally coherent rule on an IRREGULAR checkpoint grid.

THE DEFECT (A04_GATE_DESIGN.md:69-71 defines T as "2.0 % per 5 000 steps";
code/pilot_zero_rule_disagreement.py:168-202 evaluates two readings of it):

  (R1) UNSCALED  -- `accept iff rel_improvement_over_interval < 2.0`
       Compares a quantity measured over d steps against a threshold whose
       stated units are %/5k. Dimensionally wrong: the SAME underlying decay
       rate produces a larger `rel` the wider the interval, so the rule's
       stringency is a function of the grid, not of the training. Measured
       here: on A04's own frozen grid {2500,5000,10000,20000,40000,80000} the
       effective per-5k threshold ranges 3.9600 -> 0.2522 %/5k, a 15.70x
       stringency spread across the six pre-frozen checkpoints. It is 1.98x
       too LENIENT at the 2.5k spacings and 7.93x too STRICT at the 40k one.

  (R2) LINEAR-SCALED -- `accept iff rel < 2.0 * (d/5000)`
       Fixes the units but scales the wrong way: relative improvement
       COMPOUNDS, it does not add. A trajectory decaying at exactly the
       threshold rate 2.0 %/5k yields rel = 100*(1-0.98^(d/5000)), which is
       BELOW 2.0*(d/5000) for every d > 5000 -- so R2 accepts a run that is
       still improving at exactly the rate the pre-registration calls
       "not plateaued", and it does so more readily the wider the interval.
       Measured over-allowance: +1.9228 pp at d=53000, +4.3798 pp at d=80000.
       R2 is additionally NOT composition-consistent: in 174 of 200,000 random
       (d1,d2) checkpoint pairs (seed 0; 161 at seed 1), R2 accepts a merged
       interval while accepting NEITHER half -- the verdict depends on how the
       grid is subdivided.
       R2 also becomes VACUOUS at d >= 250,000 (its threshold reaches 100 %,
       which `rel` can never attain), so it is unfalsifiable on wide grids.
       Both diagnoses are computed below, not asserted.

THE REPAIR (R3), stated so it is checkable:

    Convert the interval improvement to a per-5,000-step GEOMETRIC decay rate
    before comparing:

        rate_5k(c) = 100 * ( 1 - (ppl_c / ppl_prev) ** (5000 / d) )
        PLATEAU(T) accepts at c  iff  rate_5k(c) < T,  T = 2.0 (unchanged)

WHY THIS IS THE RIGHT REPAIR, on principle rather than by fit:
  * It is the unique reading under which "2.0 % per 5 000 steps" names the same
    quantity at every spacing: rate_5k is by construction invariant to how the
    interval is subdivided, so a rule stated in per-5k units is grid-free.
  * At d = 5000 it is EXACTLY the pre-registered arithmetic -- asserted below to
    < 1e-12 on three trajectories. So R3 does not change the pre-registered
    number T = 2.0; it only stops mis-applying it off the 5k grid. The
    threshold is NOT re-tuned, and no new numeric constant is introduced.
  * It is composition-consistent in BOTH directions (0 violations in the same
    200,000-pair test that R2 fails), which is exactly the property an
    irregular grid needs.
  * It is never vacuous: rate_5k has the same range at every d.

PRE-REGISTRATION STATUS, stated plainly:
  * T = 2.0 %/5k is PRE-REGISTERED (git d1ba737, 2026-08-09) and is NOT changed.
  * The choice of R3 over R1/R2 is a POST-HOC repair, decided 2026-08-10, AFTER
    the pilot's PPL trajectory was seen. It is disclosed as such. It is not a
    threshold choice -- it is a units fix, and it is justified above by
    invariance/composition properties that hold for ANY T, not by which arm it
    accepts. R3 is STRICTLY STRICTER than R2 at every d > 5000 (proved by the
    over-allowance column), i.e. the repair makes PLATEAU harder to accept,
    which makes the PLATEAU-vs-NI disagreement A04 needs HARDER to find, not
    easier. A repair that cut the other way would deserve suspicion; this one
    costs A04 rather than paying it.
  * Consequence on the pilot trajectory, reported even though it is unfavourable
    to the pilot's framing: R3 accepts EARLIER than R1 (step 100,000 vs step
    200,000), because at these 47k-53k spacings R1 is the over-strict reading.
    So the repair moves the pilot's PLATEAU-accept checkpoint. See
    §"what changes for Pilot Zero" in the output.

CPU ONLY. Reads one JSON (the PPL trajectory) and no model. No GPU.

Usage:
  python a04_plateau_rule_repair.py --ppl_json <file> --out_json <file>
"""
from __future__ import annotations

import argparse
import json
import os
import random

# pre-registered, frozen by git d1ba737. NOT changed by this script.
T_PCT_PER_5K = 2.0
REF = 5000.0

# A04_GATE_DESIGN.md:155 -- the checkpoint grid frozen by that document.
FROZEN_GATE_GRID = [2500, 5000, 10000, 20000, 40000, 80000]

N_COMPOSITION_TRIALS = 200000
COMPOSITION_SEED_A = 0
COMPOSITION_SEED_B = 1


def rel_improve_pct(ppl_prev, ppl):
    """Relative in-domain val PPL improvement over the interval, in %."""
    return 100.0 * (ppl_prev - ppl) / ppl_prev


def rate_5k_pct(ppl_prev, ppl, d_steps):
    """R3: per-5,000-step GEOMETRIC decay rate, in %. Grid-invariant."""
    return 100.0 * (1.0 - (ppl / ppl_prev) ** (REF / d_steps))


def T_linear(d_steps):
    """R2's threshold: T scaled linearly to the interval."""
    return T_PCT_PER_5K * (d_steps / REF)


def T_compounded(d_steps):
    """The interval improvement a trajectory decaying at EXACTLY T %/5k
    actually produces. R2 allows T_linear >= this for all d > 5000."""
    return 100.0 * (1.0 - (1.0 - T_PCT_PER_5K / 100.0) ** (d_steps / REF))


def unscaled_effective_per5k_threshold(d_steps):
    """R1 compares rel(over d) to 2.0. The per-5k decay rate that sits exactly
    on that boundary -- i.e. what R1's threshold really is, in the units the
    pre-registration claims to use."""
    return 100.0 * (1.0 - (1.0 - T_PCT_PER_5K / 100.0) ** (REF / d_steps))


def composition_check(rule, seed, n=N_COMPOSITION_TRIALS):
    """Two consistency properties a grid-free rule must have.

    forward : accept(d1) and accept(d2)  =>  accept(d1+d2)
    backward: accept(d1+d2)              =>  accept(d1) or accept(d2)

    A rule failing either one gives a verdict that depends on how the operator
    happened to space the checkpoints, which is precisely the defect being
    repaired. Spacings are drawn from the frozen gate grid's own interval
    widths; PPL ratios from a range that brackets the observed trajectory.
    """
    rng = random.Random(seed)
    spacings = [2500, 5000, 10000, 20000, 40000]
    fwd = bwd = 0
    for _ in range(n):
        d1 = rng.choice(spacings)
        d2 = rng.choice(spacings)
        pa = rng.uniform(10.0, 30.0)
        pb = pa * rng.uniform(0.80, 1.0)
        pc = pb * rng.uniform(0.80, 1.0)
        a1 = rule(pa, pb, d1)
        a2 = rule(pb, pc, d2)
        am = rule(pa, pc, d1 + d2)
        if a1 and a2 and not am:
            fwd += 1
        if am and not (a1 or a2):
            bwd += 1
    return {"n_trials": n, "seed": seed,
            "forward_violations": fwd, "backward_violations": bwd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppl_json", required=True,
                    help="[[step, ppl], ...] in-domain val PPL trajectory")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    with open(args.ppl_json) as fh:
        traj = [(int(s), float(p)) for s, p in json.load(fh)]
    traj.sort(key=lambda t: t[0])
    assert len(traj) >= 2, "need >= 2 checkpoints"
    assert len({s for s, _ in traj}) == len(traj), "duplicate step in trajectory"

    # ---- R1/R2/R3 on the actual (irregular) grid --------------------------
    per_ckpt = []
    for i, (step, ppl) in enumerate(traj):
        if i == 0:
            per_ckpt.append({"step": step, "ppl": ppl, "interval_steps": None,
                             "rel_improve_pct": None, "rate_5k_pct": None,
                             "R1_unscaled_accept": None,
                             "R2_linear_accept": None,
                             "R3_repaired_accept": None,
                             "T_linear_pct": None,
                             "R1_effective_per5k_threshold_pct": None})
            continue
        pstep, pppl = traj[i - 1]
        d = step - pstep
        rel = rel_improve_pct(pppl, ppl)
        rate = rate_5k_pct(pppl, ppl, d)
        per_ckpt.append({
            "step": step, "ppl": ppl, "interval_steps": d,
            "rel_improve_pct": rel,
            "rate_5k_pct": rate,
            "R1_unscaled_accept": bool(rel < T_PCT_PER_5K),
            "R2_linear_accept": bool(rel < T_linear(d)),
            "R3_repaired_accept": bool(rate < T_PCT_PER_5K),
            "T_linear_pct": T_linear(d),
            "R1_effective_per5k_threshold_pct":
                unscaled_effective_per5k_threshold(d),
        })

    # ---- R1's stringency spread on the FROZEN gate grid -------------------
    r1_grid = []
    prev = 0
    for c in FROZEN_GATE_GRID:
        d = c - prev
        r1_grid.append({
            "checkpoint": c, "preceding": prev, "interval_steps": d,
            "R1_effective_per5k_threshold_pct":
                unscaled_effective_per5k_threshold(d),
            "distortion_vs_T": unscaled_effective_per5k_threshold(d)
                               / T_PCT_PER_5K,
            "T_linear_pct": T_linear(d),
            "T_compounded_pct": T_compounded(d),
            "R2_over_allowance_pp": T_linear(d) - T_compounded(d),
        })
        prev = c
    effs = [r["R1_effective_per5k_threshold_pct"] for r in r1_grid]
    r1_spread = max(effs) / min(effs)

    # ---- R2's over-allowance across a wide spacing range ------------------
    r2_over = []
    for d in [2500, 5000, 10000, 20000, 40000, 47000, 50000, 53000, 80000,
              100000, 150000, 250000]:
        r2_over.append({
            "interval_steps": d,
            "T_linear_pct": T_linear(d),
            "T_compounded_pct": T_compounded(d),
            "over_allowance_pp": T_linear(d) - T_compounded(d),
            "ratio": T_linear(d) / T_compounded(d),
            "R2_vacuous": bool(T_linear(d) >= 100.0),
        })
    d_vacuous = 100.0 / T_PCT_PER_5K * REF

    # ---- R2 vs R3 at the exact boundary rate ------------------------------
    # A trajectory decaying at EXACTLY T %/5k. The correct rule must sit on the
    # boundary at every d; a rule that accepts is too lenient there.
    boundary = []
    for d in [2500, 5000, 10000, 20000, 40000, 53000, 80000]:
        pp = 20.0
        p = pp * (1.0 - T_PCT_PER_5K / 100.0) ** (d / REF)
        boundary.append({
            "interval_steps": d,
            "rel_improve_pct": rel_improve_pct(pp, p),
            "T_linear_pct": T_linear(d),
            "R2_accepts_a_run_still_at_threshold_rate":
                bool(rel_improve_pct(pp, p) < T_linear(d)),
            "rate_5k_pct": rate_5k_pct(pp, p, d),
            "R3_accepts": bool(rate_5k_pct(pp, p, d) < T_PCT_PER_5K),
        })

    # ---- grid-robustness on the REAL trajectory ---------------------------
    # Same terminal checkpoint, different preceding checkpoint. A grid-free
    # statistic should barely move; a grid-dependent one moves a lot.
    grid_robust = {}
    for term_i in range(1, len(traj)):
        step, ppl = traj[term_i]
        rows = []
        for prev_i in range(term_i):
            pstep, pppl = traj[prev_i]
            d = step - pstep
            rows.append({
                "preceding_step": pstep, "interval_steps": d,
                "rel_improve_pct": rel_improve_pct(pppl, ppl),
                "rate_5k_pct": rate_5k_pct(pppl, ppl, d),
                "R1_accept": bool(rel_improve_pct(pppl, ppl) < T_PCT_PER_5K),
                "R2_accept": bool(rel_improve_pct(pppl, ppl) < T_linear(d)),
                "R3_accept": bool(rate_5k_pct(pppl, ppl, d) < T_PCT_PER_5K),
            })
        rels = [r["rel_improve_pct"] for r in rows]
        rates = [r["rate_5k_pct"] for r in rows]
        entry = {
            "terminal_step": step, "n_subgrids": len(rows), "rows": rows,
            "rel_min": min(rels), "rel_max": max(rels),
            "rel_fold_spread": (max(rels) / min(rels)) if min(rels) > 0 else None,
            "rate_min": min(rates), "rate_max": max(rates),
            "rate_fold_spread": (max(rates) / min(rates)) if min(rates) > 0
                                else None,
            "R1_verdict_unanimous": len({r["R1_accept"] for r in rows}) == 1,
            "R3_verdict_unanimous": len({r["R3_accept"] for r in rows}) == 1,
        }
        grid_robust[f"step{step}"] = entry

    # ---- composition consistency -----------------------------------------
    comp = {
        "R2_linear": composition_check(
            lambda a, b, d: rel_improve_pct(a, b) < T_linear(d),
            COMPOSITION_SEED_A),
        "R3_repaired": composition_check(
            lambda a, b, d: rate_5k_pct(a, b, d) < T_PCT_PER_5K,
            COMPOSITION_SEED_A),
        "R2_linear_alt_seed": composition_check(
            lambda a, b, d: rel_improve_pct(a, b) < T_linear(d),
            COMPOSITION_SEED_B),
        "R3_repaired_alt_seed": composition_check(
            lambda a, b, d: rate_5k_pct(a, b, d) < T_PCT_PER_5K,
            COMPOSITION_SEED_B),
    }

    # ---- HARD ASSERTION: R3 == pre-registered arithmetic at d = 5000 ------
    equiv = []
    for pp, p in [(17.6, 17.2), (10.0, 9.8), (16.1613, 15.9),
                  (17.619441896079884, 16.161295049729876)]:
        a = rel_improve_pct(pp, p)
        b = rate_5k_pct(pp, p, REF)
        assert abs(a - b) < 1e-12, f"R3 != R1 at d=5000 for {pp}->{p}"
        equiv.append({"ppl_prev": pp, "ppl": p, "rel_pct": a,
                      "rate_5k_pct": b, "abs_diff": abs(a - b)})

    # ---- what changes for Pilot Zero -------------------------------------
    def first_accept(key):
        for r in per_ckpt:
            if r.get(key) is True:
                return r["step"]
        return None

    fa_r1, fa_r2, fa_r3 = (first_accept("R1_unscaled_accept"),
                           first_accept("R2_linear_accept"),
                           first_accept("R3_repaired_accept"))
    steps = [s for s, _ in traj]
    ppl_of = dict(traj)
    remaining = None
    if fa_r3 is not None and fa_r3 != steps[-1]:
        remaining = 100.0 * (ppl_of[fa_r3] - ppl_of[steps[-1]]) / ppl_of[fa_r3]

    out = {
        "what": "A04 PLATEAU(T) dimensional repair: diagnose the unscaled and "
                "linear-scaled readings, define a grid-invariant rule, and "
                "exercise all three on the real irregular trajectory.",
        "date": "2026-08-10",
        "gpu_spent": 0,
        "preregistration": {
            "T_pct_per_5k": T_PCT_PER_5K,
            "frozen_by_commit": "d1ba737",
            "T_is_unchanged_by_this_repair": True,
            "what_is_post_hoc": "the CHOICE of reading R3 over R1/R2, decided "
                                "2026-08-10 after the PPL trajectory was seen. "
                                "Disclosed, not concealed. No new numeric "
                                "constant is introduced and T is not retuned.",
            "repair_direction_is_costly_to_A04":
                "R3 is strictly stricter than R2 at every d > 5000, so the "
                "repair makes PLATEAU harder to accept and the "
                "PLATEAU-vs-NI disagreement harder to find.",
        },
        "defect_R1_unscaled": {
            "statement": "compares an interval-length-dependent quantity to a "
                         "per-5k threshold; stringency is set by the grid.",
            "frozen_gate_grid": FROZEN_GATE_GRID,
            "per_checkpoint": r1_grid,
            "effective_threshold_fold_spread_across_frozen_grid": r1_spread,
            "most_lenient_checkpoint": FROZEN_GATE_GRID[
                effs.index(max(effs))],
            "most_strict_checkpoint": FROZEN_GATE_GRID[effs.index(min(effs))],
        },
        "defect_R2_linear": {
            "statement": "correct units, wrong algebra: relative improvement "
                         "compounds, so a linear allowance over-permits, and "
                         "the over-permission grows with the interval.",
            "over_allowance_table": r2_over,
            "boundary_rate_test": boundary,
            "vacuous_at_interval_steps_ge": d_vacuous,
            "composition_consistency": comp["R2_linear"],
            "composition_consistency_alt_seed": comp["R2_linear_alt_seed"],
        },
        "repair_R3": {
            "formula": "rate_5k(c) = 100*(1 - (ppl_c/ppl_prev)**(5000/d)); "
                       "accept iff rate_5k(c) < T with T = 2.0 unchanged",
            "equals_prereg_arithmetic_at_d_5000": equiv,
            "composition_consistency": comp["R3_repaired"],
            "composition_consistency_alt_seed": comp["R3_repaired_alt_seed"],
            "never_vacuous": True,
            "vacuity_note": "rate_5k has the same range at every d, so no "
                            "spacing makes the rule unfalsifiable -- unlike R2, "
                            f"which cannot reject at d >= {d_vacuous:.0f}.",
        },
        "trajectory_per_checkpoint": per_ckpt,
        "grid_robustness_same_terminal_different_preceding": grid_robust,
        "what_changes_for_pilot_zero": {
            "first_accept_step_R1_unscaled": fa_r1,
            "first_accept_step_R2_linear": fa_r2,
            "first_accept_step_R3_repaired": fa_r3,
            "R3_accepts_earlier_than_R1": bool(
                fa_r3 is not None and fa_r1 is not None and fa_r3 < fa_r1),
            "relative_ppl_improvement_remaining_after_R3_first_accept_pct":
                remaining,
            "honest_consequence":
                "under R3 the PLATEAU-accept checkpoint on this trajectory "
                "moves EARLIER (step 100,000, not 200,000). Pilot Zero's "
                "capability axes were only scored at step 200,000, so the "
                "repaired rule's own first-accept checkpoint has NO capability "
                "measurement -- the PLATEAU-vs-NI cell at step 100,000 is "
                "UNMEASURED, not resolved. Step 200,000 still accepts under R3 "
                "(rate 0.1317 %/5k), so the pilot's single measured cell "
                "survives the repair; but a claim about WHERE the earliest "
                "disagreement lies would need step-100,000 capability scoring, "
                "which is GPU work and is not done here.",
        },
        "HEADLINE": {
            "R1_stringency_fold_spread_on_frozen_grid": r1_spread,
            "R2_backward_composition_violations": comp["R2_linear"][
                "backward_violations"],
            "R3_backward_composition_violations": comp["R3_repaired"][
                "backward_violations"],
            "R3_first_accept_step": fa_r3,
            "R1_first_accept_step": fa_r1,
            "step200000_rate_5k_pct": per_ckpt[-1]["rate_5k_pct"],
            "step200000_accepts_under_R3": per_ckpt[-1]["R3_repaired_accept"],
            "gate_is_now_runnable_on_an_irregular_grid": True,
        },
        "provenance": {
            "ppl_json": os.path.abspath(args.ppl_json),
            "trajectory": [list(t) for t in traj],
            "canonical_ppl_source":
                "zwfy6:olmo2_ppl_results/1B_keep7_step{50000,100000,147000,"
                "200000}/summary.json (n_shards=8, n_tokens=8384512, "
                "n_windows=4096) + 1B_base_full for the intact reference; "
                "these four values also appear in "
                "evidence/pilot_zero_rule_disagreement.json:ppl_trajectory.",
            "frozen_grid_source": "A04_GATE_DESIGN.md:155",
            "defect_site": "code/pilot_zero_rule_disagreement.py:168-202 "
                           "(plateau_rule, both readings)",
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=1)

    print("=== A04 PLATEAU(T) repair ===")
    print(f"T = {T_PCT_PER_5K} %/5k (pre-registered, UNCHANGED)")
    print()
    print("R1 (unscaled) effective per-5k threshold on the FROZEN gate grid:")
    for r in r1_grid:
        print(f"  ckpt {r['checkpoint']:>6} d={r['interval_steps']:>6}  "
              f"eff={r['R1_effective_per5k_threshold_pct']:>7.4f} %/5k  "
              f"({r['distortion_vs_T']:.3f}x T)")
    print(f"  -> {r1_spread:.2f}x stringency spread across six checkpoints")
    print()
    print("R2 (linear) over-allowance vs a true threshold-rate trajectory:")
    for r in r2_over:
        if r["interval_steps"] in (5000, 20000, 53000, 80000, 250000):
            print(f"  d={r['interval_steps']:>6}  T_linear="
                  f"{r['T_linear_pct']:>8.4f}%  T_compounded="
                  f"{r['T_compounded_pct']:>8.4f}%  over="
                  f"{r['over_allowance_pp']:>8.4f}pp  vacuous={r['R2_vacuous']}")
    print(f"  -> R2 cannot reject at all for d >= {d_vacuous:.0f} steps")
    print()
    print("composition consistency (200k random checkpoint pairs):")
    for k in ("R2_linear", "R3_repaired"):
        c = comp[k]
        print(f"  {k:<12} forward={c['forward_violations']:>4}  "
              f"backward={c['backward_violations']:>4}")
    print()
    print("real trajectory:")
    for r in per_ckpt:
        if r["interval_steps"] is None:
            print(f"  step {r['step']:>7}  ppl={r['ppl']:.4f}  (baseline)")
            continue
        print(f"  step {r['step']:>7}  ppl={r['ppl']:.4f}  d={r['interval_steps']:>6}"
              f"  rel={r['rel_improve_pct']:>7.4f}%  rate={r['rate_5k_pct']:>7.5f}%/5k"
              f"  R1={r['R1_unscaled_accept']!s:<5} R2={r['R2_linear_accept']!s:<5}"
              f" R3={r['R3_repaired_accept']!s:<5}")
    print()
    print("grid robustness at the terminal checkpoint (vary preceding ckpt):")
    last = grid_robust[f"step{traj[-1][0]}"]
    print(f"  rel  spans {last['rel_min']:.4f} .. {last['rel_max']:.4f} "
          f"({last['rel_fold_spread']:.2f}x); R1 verdict unanimous: "
          f"{last['R1_verdict_unanimous']}")
    print(f"  rate spans {last['rate_min']:.5f} .. {last['rate_max']:.5f} "
          f"({last['rate_fold_spread']:.2f}x); R3 verdict unanimous: "
          f"{last['R3_verdict_unanimous']}")
    print()
    print(f"first accept: R1 @ {fa_r1}, R2 @ {fa_r2}, R3 @ {fa_r3}")
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
