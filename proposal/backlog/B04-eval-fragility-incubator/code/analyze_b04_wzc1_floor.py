#!/usr/bin/env python3
"""B04 G0 — floor-first analysis of the wzc1 sm_100 ladder. ZERO GPU.

Run from repo root:
    python proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py

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
        selftest_phi()
    else:
        main()
