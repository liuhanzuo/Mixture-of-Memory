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
E_RANGE_OVER_SIGMA = {2: 1.1284, 3: 1.6926, 4: 2.0588, 8: 2.8472}

FLOOR_SAFETY_FACTOR = 6  # damaged range must clear 6*sigma_hat to be an admissible denominator


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

    clause5 = {
        "primary_metric": "median_margin",
        "damaged_rungs": dl,
        "damaged_range_median_margin": dam_range,
        "damaged_heal_step_span": max(dstep) - min(dstep),
        "denominator_guard_threshold": guard,
        "denominator_guard_basis": f"{FLOOR_SAFETY_FACTOR}*sigma_hat(median_margin)",
        "denominator_admissible": denom_ok,
        "denominator_verdict": "OK" if denom_ok else "DENOMINATOR_UNRESOLVED",
        "phi_kill_threshold": 0.60,
        "phi_pass_threshold": 0.30,
        "beta_budget_kill_per_step": 0.60 * dam_range / (max(dstep) - min(dstep)),
        "beta_budget_pass_per_step": 0.30 * dam_range / (max(dstep) - min(dstep)),
        "S_damage_ols": ols_slope(dcore, dmed),
        "spearman_core6_heal_steps_MANDATORY_CODISCLOSURE": spearman(dcore, dstep),
        "spearman_core6_layers_kept": spearman(dcore, [kp for _, _, kp, _ in dam]),
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
    print(f"\nclause 5 denominator: range={dam_range:.6f} guard={guard:.6f} -> "
          f"{clause5['denominator_verdict']}")
    print(f"  KILL if |beta_budget| >= {clause5['beta_budget_kill_per_step']:.4e} /step "
          f"(phi>=0.60); PASS if <= {clause5['beta_budget_pass_per_step']:.4e} /step (phi<=0.30)")
    print(f"  MANDATORY co-disclosure Spearman(core6, heal_steps) = "
          f"{clause5['spearman_core6_heal_steps_MANDATORY_CODISCLOSURE']:+.4f} (wzc1 ladder)")


if __name__ == "__main__":
    main()
