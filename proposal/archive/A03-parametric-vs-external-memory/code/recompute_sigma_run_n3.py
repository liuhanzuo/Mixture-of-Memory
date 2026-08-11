#!/usr/bin/env python3
"""Recompute A03's run-to-run spread sigma_run at n=3 draws per family (df=5 pooled).

WHAT CHANGED, AND WHY THIS SCRIPT EXISTS
----------------------------------------
`ARM_SET_DECISION.md` §0/§2 and `A04/STAGE_B_DECISION.md`'s 21:10 addendum both
quote **pooled sigma_run = 0.3620 pp, df = 4, chi2 95 % CI [0.217, 1.040]** on
triviaqa em, and the MDE table derived from it (1.10 pp at S=3, 3.16 pp at the
chi2 upper bound). That pooled figure combined:

    keep7 +fresh2, 20k CPT, sampler seeds {0, 43, 44}     S=3, df=2, s=0.4132
    keep12+fresh2,  5k,           seeds {101, 102, 103}   S=3, df=2, s=0.3023

Sampler seed 45 landed 2026-08-11 23:29 GMT+8 and was scored on the same four
axes. It is the **fourth** draw of the keep7 family, taking that family to S=4 /
df=3 and the pooled estimate to **df=5**. `SEED45_HANDOFF.md` §"How to recompute
sigma_run at n=3" specifies the arithmetic; this script is that arithmetic, run
against the per-item shards rather than against transcribed numbers.

CRITICAL, from `SEED45_HANDOFF.md`: take the per-axis **arm mean** (the absolute
accuracy of each seed's own checkpoint), NOT the paired delta vs the baseline.
The paired delta shares the baseline term across seeds, so its spread is not the
run-to-run spread of a single arm.

CONVENTIONS (reproduced from ARM_SET_DECISION.md §5, not redefined)
------------------------------------------------------------------
    s          = sample sd over the S draws, df = S-1
    chi2 CI    = [ s*sqrt(df/chi2.ppf(.975,df)) , s*sqrt(df/chi2.ppf(.025,df)) ]
    pooled     = sqrt( (df1*s1^2 + df2*s2^2) / (df1+df2) )
    MDE        = ( t_{.975,2S-2} + t_{.80,2S-2} ) * sigma * sqrt(2/S)
    A04 bound  = t_{.05,df} (one-sided) * s / sqrt(S)      -- A04's pre-reg form

STANDING PROHIBITION (prereg §4): never quote a sigma_run point estimate without
its d.o.f. and its chi2 interval.
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared" / "code"))
from canonical_eval_loaders import load_cb, load_mmlu, N_MMLU  # noqa: E402

# --- the two families ------------------------------------------------------
# keep7+fresh2 20k CPT: sampler seed 0 IS the original Arm 3 (DATAORDER_VERDICT.md
# line 20; _run_a03_dataorder_repl.sh is config-identical to _run_a03_arm3_cpt.sh
# apart from --seed; resumed runs, no dropout => --seed's only material channel is
# DistributedSampler(seed=args.seed)).
KEEP7 = {
    0:  "A03_1B_arm3_cpt_step220000",
    43: "A03_1B_dataorder_seed43_step220000",
    44: "A03_1B_dataorder_seed44_step220000",
    45: "A03_1B_dataorder_seed45_step220000",
}
# keep12+fresh2 5k, A04 Pilot One Stage B. Means are re-derived here rather than
# copied from stageB_S3_verdict.json so the pooling uses one estimator end to end.
KEEP12 = {
    101: "A04_1B_stageB_keep12_seed101_step5000",
    102: "A04_1B_stageB_keep12_seed102_step5000",
    103: "A04_1B_stageB_keep12_seed103_step5000",
}
AXES = ("triviaqa", "popqa", "nq_open", "mmlu_content")


def arm_mean(tag, axis):
    """Absolute accuracy of one arm on one axis, in percent. Raises on partial shards."""
    if axis == "mmlu_content":
        d = load_mmlu(tag)
        assert len(d) == N_MMLU, f"{tag}: {len(d)} != {N_MMLU}"
        return float(np.mean([v[1] for v in d.values()]) * 100.0), len(d)
    src = tag + "_nq" if axis == "nq_open" else tag
    d = load_cb(src, axis)
    return float(np.mean([v[0] for v in d.values()]) * 100.0), len(d)   # v[0] = em


def sd_block(vals):
    """s, df, chi2 95 % CI for sigma. ddof=1 sample sd."""
    S = len(vals)
    df = S - 1
    s = float(np.std(vals, ddof=1))
    lo = s * np.sqrt(df / stats.chi2.ppf(0.975, df))
    hi = s * np.sqrt(df / stats.chi2.ppf(0.025, df))
    return {"S": S, "df": df, "s_pp": s, "chi2_ci95_pp": [float(lo), float(hi)],
            "chi2_width_multiplicative": float(hi / lo) if lo > 0 else None,
            "means_pct": [float(v) for v in vals]}


def mde(sigma, S):
    """Two-sample MDE, alpha=.05 two-sided, power .80, t-based (df = 2S-2)."""
    df = 2 * S - 2
    return float((stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df)) * sigma * np.sqrt(2.0 / S))


out = {
    "scope": "A03 sigma_run at n=3 draws per family (pooled df=5) after sampler seed 45 landed",
    "supersedes": "pooled sigma_run = 0.3620 pp at df=4 (keep7 seeds {0,43,44} + keep12 seeds "
                  "{101,102,103}), quoted in ARM_SET_DECISION.md 0/2, POSTMORTEM.md, "
                  "STATUS.json, A04/STAGE_B_DECISION.md addendum item 4, proposal/README.md",
    "estimator": "per-axis ARM MEAN per seed (absolute accuracy, not the paired delta); "
                 "s = sample sd (ddof=1); df = S-1; chi2 95% CI for sigma; "
                 "pooled = sqrt((df1*s1^2+df2*s2^2)/(df1+df2))",
    "standing_prohibition": "prereg s4: never quote a sigma_run point estimate without its "
                            "d.o.f. and its chi2 interval",
    "families": {}, "pooled": {}, "mde": {}, "a04_k2": {},
}

fam = {"keep7_20k_cpt": KEEP7, "keep12_5k": KEEP12}
for fname, tags in fam.items():
    out["families"][fname] = {"seeds": sorted(tags), "dirs": tags, "axes": {}}
    for axis in AXES:
        vals, ns = [], []
        for sd in sorted(tags):
            m, n = arm_mean(tags[sd], axis)
            vals.append(m); ns.append(n)
        if len(set(ns)) != 1:
            raise SystemExit(f"FATAL {fname}/{axis}: item counts differ across seeds {ns} "
                             "-- the arms are not on a common item set")
        blk = sd_block(vals)
        blk["n_items"] = ns[0]
        out["families"][fname]["axes"][axis] = blk

# --- pooled, primary axis (triviaqa em) ------------------------------------
for axis in AXES:
    a = out["families"]["keep7_20k_cpt"]["axes"][axis]
    b = out["families"]["keep12_5k"]["axes"][axis]
    df = a["df"] + b["df"]
    s = float(np.sqrt((a["df"] * a["s_pp"] ** 2 + b["df"] * b["s_pp"] ** 2) / df))
    lo = s * np.sqrt(df / stats.chi2.ppf(0.975, df))
    hi = s * np.sqrt(df / stats.chi2.ppf(0.025, df))
    out["pooled"][axis] = {"df": df, "sigma_pp": s,
                           "chi2_ci95_pp": [float(lo), float(hi)],
                           "chi2_width_multiplicative": float(hi / lo)}

sig = out["pooled"]["triviaqa"]["sigma_pp"]
sig_hi = out["pooled"]["triviaqa"]["chi2_ci95_pp"][1]
out["mde"] = {
    "axis": "triviaqa em", "form": "(t_.975,2S-2 + t_.80,2S-2) * sigma * sqrt(2/S)",
    "at_sigma_hat": {str(S): mde(sig, S) for S in (3, 4, 5, 8)},
    "at_chi2_upper": {str(S): mde(sig_hi, S) for S in (3, 4, 5, 8)},
    "sigma_hat_pp": sig, "chi2_upper_pp": sig_hi,
}

# --- A04 K2: bound_S = t_{.05,df} * s / sqrt(S) vs Delta, at point AND chi2 upper
# The keep12 family is the one A04's K2 adjudicates. Deltas are A04's pre-registered
# values (PILOT_ONE_PREREG.md, reproduced in stageB_S3_verdict.json) -- not recomputed.
DELTA = {"triviaqa": 4.043134195274186, "popqa": 1.3205298941613512,
         "mmlu_content": 1.0238926078906136, "nq_open": 0.9695290858725762}
DECISION_AXES = ("triviaqa", "popqa", "mmlu_content")
for axis, D in DELTA.items():
    b = out["families"]["keep12_5k"]["axes"][axis]
    t1 = float(stats.t.ppf(0.95, b["df"]))          # one-sided t_{.05, df}
    pt = t1 * b["s_pp"] / np.sqrt(b["S"])
    up = t1 * b["chi2_ci95_pp"][1] / np.sqrt(b["S"])
    out["a04_k2"][axis] = {
        "S": b["S"], "df": b["df"], "t_05_onesided": t1, "s_pp": b["s_pp"],
        "delta_pp": D, "bound_at_point_pp": float(pt), "bound_at_chi2_upper_pp": float(up),
        "fires_at_point": bool(pt > D), "fires_at_chi2_upper": bool(up > D),
        "margin_at_point": float(D / pt) if pt > 0 else None,
        "decision_axis": axis in DECISION_AXES,
    }
out["a04_k2"]["_rule"] = ("K2 FIRES iff bound_S > Delta on >= 2 of the 3 decision axes "
                          "(triviaqa, popqa, mmlu_content); nq_open is demoted")
out["a04_k2"]["_n_decision_axes_firing_at_point"] = sum(
    1 for a in DECISION_AXES if out["a04_k2"][a]["fires_at_point"])
out["a04_k2"]["_n_decision_axes_firing_at_chi2_upper"] = sum(
    1 for a in DECISION_AXES if out["a04_k2"][a]["fires_at_chi2_upper"])

dest = sys.argv[1] if len(sys.argv) > 1 else "/tmp/a03_sigma_run_n3.json"
Path(dest).write_text(json.dumps(out, indent=2))
print(f"wrote {dest}\n")
for fname, f in out["families"].items():
    print(f"[{fname}] seeds {f['seeds']}")
    for axis, b in f["axes"].items():
        print(f"   {axis:13s} S={b['S']} df={b['df']} s={b['s_pp']:.4f} pp  "
              f"chi2 95% CI [{b['chi2_ci95_pp'][0]:.3f}, {b['chi2_ci95_pp'][1]:.3f}]  "
              f"(width {b['chi2_width_multiplicative']:.1f}x)  n={b['n_items']}")
        print(f"                 means: {['%.4f' % v for v in b['means_pct']]}")
print("\n[pooled]")
for axis, p in out["pooled"].items():
    print(f"   {axis:13s} df={p['df']} sigma={p['sigma_pp']:.4f} pp  "
          f"chi2 95% CI [{p['chi2_ci95_pp'][0]:.3f}, {p['chi2_ci95_pp'][1]:.3f}]  "
          f"(width {p['chi2_width_multiplicative']:.1f}x)")
print(f"\n[MDE, triviaqa em]  sigma_hat={sig:.4f}  chi2_upper={sig_hi:.4f}")
for S in ("3", "4", "5", "8"):
    print(f"   S={S}: {out['mde']['at_sigma_hat'][S]:.2f} pp at sigma_hat, "
          f"{out['mde']['at_chi2_upper'][S]:.2f} pp at chi2 upper")
print("\n[A04 K2, keep12 family]")
for axis in DELTA:
    k = out["a04_k2"][axis]
    print(f"   {axis:13s} bound={k['bound_at_point_pp']:.3f} vs D={k['delta_pp']:.3f} "
          f"({k['margin_at_point']:.1f}x margin) | at chi2 upper bound="
          f"{k['bound_at_chi2_upper_pp']:.3f} -> fires={k['fires_at_chi2_upper']}"
          f"{'' if k['decision_axis'] else '  [DEMOTED]'}")
print(f"   decision axes firing: {out['a04_k2']['_n_decision_axes_firing_at_point']} at point, "
      f"{out['a04_k2']['_n_decision_axes_firing_at_chi2_upper']} at chi2 upper (K2 needs >=2)")
