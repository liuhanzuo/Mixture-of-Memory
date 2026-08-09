#!/usr/bin/env python3
"""A01 gate-3 residual analysis: the CONTENT interface x the FIVE longest-option
null conventions, on all six OLMo-2-7B dtype arms.

Closes `STATUS.json:gate_results.gate3_fp32_causal_tie_test.remaining_analysis_TODO`:

  "the six summaries also carry content_norm columns and four
   longest_option_floor_by_conv variants (split/first/last/credit/wrong) that
   nobody has tabulated. CPU-only, no GPU."

Why it matters for A01. A01's thesis is that a reported number is meaningless
without a construct-appropriate null. The longest-option null is itself
UNDER-SPECIFIED: "pick the option with the most continuation tokens" does not
say what to do when several options tie on token count. The five conventions
below are all defensible readings of the same English sentence, and they put the
null anywhere from 0.1961 to 0.4537 on MMLU. So the very null A01 recommends has
a convention degree of freedom large enough to flip arm verdicts. That is A01's
own thesis applied to A01's own instrument.

Conventions (verbatim from `a01_gate3_fp32_vs_bf16.py::longest_floor`):
  split  : 1/|W| if gold in W          -- expected accuracy of uniform random
                                          tie-breaking; the CANONICAL choice
  first  : 1 if W[0]  == gold          -- break ties by lowest letter index
                                          (what argmax actually does)
  last   : 1 if W[-1] == gold          -- break ties by highest letter index
  credit : 1 if gold in W              -- OPTIMISTIC / oracle tie-breaking
  wrong  : 1 if |W| == 1 and W[0]==gold-- PESSIMISTIC, any tie scores 0
where W = argmax-set of `cont_tokens` over the available options.

Usage:
  python a01_gate3_content_conventions.py <raw_root> <out_json> [out_csv]

`<raw_root>` holds one subdir per arm, each with
`per_example_dtype_shard{0..7}of8.jsonl` as written by
`a01_gate3_fp32_vs_bf16.py`. Source of truth on disk:
`zwfy6:/apdcephfs_zwfy6/.../results/a01_gate3/dtype_runs/<arm>_dtype/`.

CPU only. No GPU, no model load.
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

LETTERS = ["A", "B", "C", "D"]
CONVS = ["split", "first", "last", "credit", "wrong"]
DTYPES = ["bf16", "fp32"]
N_BOOT = 10000
BOOT_SEED = 7  # matches a01_gate3_fp32_vs_bf16.py's arm-vs-floor test

# Display order = increasing damage, matching STATUS.json's six-arm table.
ARM_ORDER = [
    "7B_base",
    "7B_shortgpt16_step200000",
    "7B_keep14_step200000",
    "7B_keep12_step124000",
    "7B_keep10_step83500",
    "7B_keep8_step121000",
]


def load_arm(d):
    recs = []
    shards = sorted(glob.glob(os.path.join(d, "per_example_dtype_shard*of8.jsonl")))
    assert len(shards) == 8, f"{d}: expected 8 shards, found {len(shards)}"
    for s in shards:
        with open(s) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("nan"):
                    recs.append(r)
    ids = [r["item_id"] for r in recs]
    assert len(set(ids)) == len(ids), f"{d}: duplicate item_id after merge"
    recs.sort(key=lambda r: r["item_id"])
    return recs


def winner_set(rec):
    """Argmax-set of continuation-token counts over the available options."""
    ct = rec["cont_tokens"]
    keys = [k for k in LETTERS if k in ct]
    top = max(ct[k] for k in keys)
    return [k for k in keys if ct[k] == top]


def null_vectors(recs):
    """Per-item score of each longest-option convention. Input-blind: depends
    only on option token counts and the gold label, never on the model."""
    out = {c: np.zeros(len(recs)) for c in CONVS}
    mult = Counter()
    gold_in_win = 0
    for i, r in enumerate(recs):
        W = winner_set(r)
        g = r["gold_letter"]
        mult[len(W)] += 1
        if g in W:
            gold_in_win += 1
        out["split"][i] = (1.0 / len(W)) if g in W else 0.0
        out["first"][i] = 1.0 if W[0] == g else 0.0
        out["last"][i] = 1.0 if W[-1] == g else 0.0
        out["credit"][i] = 1.0 if g in W else 0.0
        out["wrong"][i] = 1.0 if (len(W) == 1 and W[0] == g) else 0.0
    diag = {
        "winner_set_size_hist": {str(k): mult[k] for k in sorted(mult)},
        "frac_items_with_tied_longest": 1.0 - mult[1] / len(recs),
        "frac_items_gold_in_winner_set": gold_in_win / len(recs),
    }
    return out, diag


def argmax_idx(vals):
    """argmax with ties broken by INDEX -- what torch/py max() does, and the
    same operation the model-side readout uses."""
    return max(range(len(vals)), key=lambda k: vals[k])


def model_correct(recs, dt, norm):
    out = np.zeros(len(recs))
    for i, r in enumerate(recs):
        n = r["n_opt"]
        raw = [r[dt]["content_raw"][LETTERS[k]] for k in range(n)]
        if norm:
            v = [raw[k] / max(r["cont_tokens"][LETTERS[k]], 1) for k in range(n)]
        else:
            v = raw
        out[i] = 1.0 if argmax_idx(v) == r["gold"] else 0.0
    return out


def letter_correct(recs, dt):
    out = np.zeros(len(recs))
    for i, r in enumerate(recs):
        v = [r[dt]["letter"][LETTERS[k]] for k in range(r["n_opt"])]
        out[i] = 1.0 if argmax_idx(v) == r["gold"] else 0.0
    return out


def paired_bootstrap(d, n_boot=N_BOOT, seed=BOOT_SEED):
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        bs[i] = d[rng.integers(0, d.size, d.size)].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(max(p, 1.0 / n_boot))


def verdict(mean, p):
    if p >= 0.05:
        return "AT the null (indistinguishable)"
    return "BELOW the null" if mean < 0 else "above the null"


def main():
    raw_root, out_json = sys.argv[1], sys.argv[2]
    out_csv = sys.argv[3] if len(sys.argv) > 3 else None

    arms = [a for a in ARM_ORDER if os.path.isdir(os.path.join(raw_root, a))]
    extra = sorted(set(os.listdir(raw_root)) - set(ARM_ORDER))
    extra = [e for e in extra if os.path.isdir(os.path.join(raw_root, e))]
    assert not extra, f"unexpected arm dirs (add to ARM_ORDER): {extra}"
    assert len(arms) == 6, f"expected 6 arms, found {len(arms)}: {arms}"

    result = {
        "what": "content interface x five longest-option null conventions, "
                "six OLMo-2-7B dtype arms",
        "closes": "STATUS.json gate3_fp32_causal_tie_test.remaining_analysis_TODO",
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "conventions": {
            "split": "1/|W| if gold in W -- uniform random tie-break (CANONICAL)",
            "first": "1 if W[0] == gold -- lowest-index tie-break (what argmax does)",
            "last": "1 if W[-1] == gold -- highest-index tie-break",
            "credit": "1 if gold in W -- OPTIMISTIC / oracle tie-break",
            "wrong": "1 if |W|==1 and W[0]==gold -- PESSIMISTIC, any tie scores 0",
        },
        "arms": {},
    }
    csv_rows = []

    for arm in arms:
        recs = load_arm(os.path.join(raw_root, arm))
        n = len(recs)
        nulls, diag = null_vectors(recs)
        arm_out = {
            "n": n,
            "longest_option_null_diagnostics": diag,
            "longest_option_floor_by_conv": {c: float(nulls[c].mean())
                                             for c in CONVS},
            "letter_acc": {dt: float(letter_correct(recs, dt).mean())
                           for dt in DTYPES},
            "by_dtype": {},
        }
        for dt in DTYPES:
            cn = model_correct(recs, dt, norm=True)
            cr = model_correct(recs, dt, norm=False)
            ent = {"content_norm_acc": float(cn.mean()),
                   "content_raw_acc": float(cr.mean()),
                   "vs_null": {}}
            for readout, vec in (("content_norm", cn), ("content_raw", cr)):
                ent["vs_null"][readout] = {}
                for c in CONVS:
                    m, lo, hi, p = paired_bootstrap(vec - nulls[c])
                    ent["vs_null"][readout][c] = {
                        "null": float(nulls[c].mean()),
                        "delta_pp": 100 * m,
                        "ci95_pp": [100 * lo, 100 * hi],
                        "boot_p": p,
                        "verdict": verdict(m, p),
                        # residual fraction = (reported - null) / reported,
                        # the quadruple A01 asks every construct to report
                        "residual_fraction": (float(vec.mean()) - float(nulls[c].mean()))
                                             / float(vec.mean()),
                    }
                    csv_rows.append({
                        "arm": arm, "n": n, "dtype": dt, "readout": readout,
                        "convention": c,
                        "reported": round(float(vec.mean()), 6),
                        "null": round(float(nulls[c].mean()), 6),
                        "delta_pp": round(100 * m, 4),
                        "ci95_lo_pp": round(100 * lo, 4),
                        "ci95_hi_pp": round(100 * hi, 4),
                        "boot_p": p,
                        "verdict": verdict(m, p),
                        "residual_fraction": round(
                            (float(vec.mean()) - float(nulls[c].mean()))
                            / float(vec.mean()), 6),
                    })
            arm_out["by_dtype"][dt] = ent
        result["arms"][arm] = arm_out
        print(f"[done] {arm} n={n}")

    # ---- cross-arm convention-sensitivity summary ----
    sens = {}
    for c in CONVS:
        verdicts = {arm: result["arms"][arm]["by_dtype"]["bf16"]
                    ["vs_null"]["content_norm"][c]["verdict"] for arm in arms}
        sens[c] = {
            "null": result["arms"][arms[0]]["longest_option_floor_by_conv"][c],
            "n_arms_above": sum(1 for v in verdicts.values() if v.startswith("above")),
            "n_arms_at": sum(1 for v in verdicts.values() if v.startswith("AT")),
            "n_arms_below": sum(1 for v in verdicts.values() if v.startswith("BELOW")),
            "per_arm_verdict_bf16_content_norm": verdicts,
        }
    result["convention_sensitivity_bf16_content_norm"] = sens
    result["null_is_dataset_property_not_arm_property"] = (
        len({round(result["arms"][a]["longest_option_floor_by_conv"]["split"], 9)
             for a in arms}) == 1)

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_json}")
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"wrote {out_csv} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
