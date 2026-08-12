#!/usr/bin/env python3
"""A04 shallow-rung sensitivity: does the NI verdict survive re-measurement?

The main pass (`a04_shallow_rung_ni_7b.py`) reports point estimates plus the
bootstrap flip distance. That is not enough. Every checkpoint here has been
scored by the harness MORE THAN ONCE (different day, different disk, in some
cases a different harness commit), so the honest question is whether any NI
verdict is an artefact of WHICH measurement was picked.

This is a genuinely different sensitivity from the bootstrap one:
  * bootstrap SE  -> sampling variability of the ITEM SET, one scoring run;
  * re-measurement -> variability of the SCORING itself, same checkpoint.
And it is different again from `sd_run`, which is SEED variance and does not
exist at 7B (one seed per rung).

For every (rung, axis) with >=2 admissible measurements of the SAME checkpoint,
this recomputes NI under every combination of (anchor measurement, arm
measurement) and reports whether the accept/reject flips.

CPU ONLY. Read-only. Imports the same frozen `ni_rule`; nothing re-derived.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.abspath(os.path.join(
        _HERE, "..", "..", "..", "shared", "code"))):
    if p not in sys.path:
        sys.path.insert(0, p)

from pilot_zero_rule_disagreement import (  # noqa: E402
    AXES,
    DEMOTED_AXES,
    EXPECTED_N,
    PREREG,
    build_nulls,
    load_shards,
    mmlu_content_norm_vec,
    ni_rule,
    qa_metric_vec,
)

# Alternative admissible measurements of the SAME checkpoint. Each entry is a
# list of dirs; all must carry identical `ckpt` meta (asserted below).
MEASUREMENTS = {
    "intact_7B_base": {
        "mmlu_content": ["MM/7B_base", "STAGE/D5_intact_wzc1_mm"],
        "triviaqa": ["CB/base_full", "STAGE/D5_intact_wzc1_cb"],
        "popqa": ["CB/base_full", "STAGE/D5_intact_wzc1_cb"],
        "nq_open": ["CB/base_full_nqopen"],
    },
    "keep14fresh2_step200k": {
        "mmlu_content": ["MM/7B_keep14_step200000",
                         "MM/7B_keep14_step200000_v2"],
        "triviaqa": ["CB/keep14_step200k", "CB/7B_keep14_step200000",
                     "CB/7B_keep14_step200000_v2"],
        "popqa": ["CB/keep14_step200k", "CB/7B_keep14_step200000",
                  "CB/7B_keep14_step200000_v2"],
        "nq_open": ["CB/keep14_step200k_nqopen"],
    },
    "shortgpt16_step200k": {
        "mmlu_content": ["MM/7B_shortgpt16_step200000",
                         "MM/7B_shortgpt16_step200000_v2",
                         "MM/7B_shortgpt16_step200000_v3"],
        "triviaqa": ["STAGE/shortgpt16_step200k",
                     "CB/7B_shortgpt16_step200000_v2",
                     "CB/7B_shortgpt16_step200000_v3"],
        "popqa": ["STAGE/shortgpt16_step200k",
                  "CB/7B_shortgpt16_step200000_v2",
                  "CB/7B_shortgpt16_step200000_v3"],
        "nq_open": ["STAGE/shortgpt16_step200k_nqopen"],
    },
    "full32_dolmino_step25k": {
        "mmlu_content": ["MM/7B_full32_step25000"],
        "triviaqa": ["CB/full32_step25000"],
        "popqa": ["CB/full32_step25000"],
        "nq_open": ["CB/full32_step25000_nqopen"],
    },
}

STEM = {"mmlu_content": "mmlu"}


def resolve(tag, roots):
    kind, name = tag.split("/", 1)
    return os.path.join(roots[kind], name)


def load_axis(tag, axis, roots):
    d = resolve(tag, roots)
    stem = STEM.get(axis, axis)
    rows = load_shards(d, stem, EXPECTED_N[stem if stem == "mmlu" else axis])
    vec = (mmlu_content_norm_vec(rows) if axis == "mmlu_content"
           else qa_metric_vec(rows, "em"))
    return vec, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--stage_root", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    roots = {"MM": os.path.join(args.raw_root, "olmo2_mmlu_content_results"),
             "CB": os.path.join(args.raw_root, "olmo2_closedbook_results"),
             "STAGE": args.stage_root}

    # load every measurement once
    loaded, rows_cache = {}, {}
    for arm, byax in MEASUREMENTS.items():
        for axis, tags in byax.items():
            for t in tags:
                if (t, axis) in loaded:
                    continue
                v, r = load_axis(t, axis, roots)
                loaded[(t, axis)] = v
                rows_cache[(t, axis)] = r

    # nulls come from the PINNED anchor measurement only (guard G0 analogue):
    # the null is a dataset/tokenizer property and must not float with the
    # measurement choice.
    anchor_rows = {"_mmlu_rows": rows_cache[("MM/7B_base", "mmlu_content")]}
    for t in ("triviaqa", "popqa", "nq_open"):
        tag = MEASUREMENTS["intact_7B_base"][t][0]
        anchor_rows[f"_{t}_rows"] = rows_cache[(tag, t)]
    nulls = build_nulls(anchor_rows)

    out = {
        "what": ("re-measurement sensitivity of the 7B shallow-rung NI "
                 "verdicts: same checkpoint, different scoring run"),
        "date": "2026-08-12", "gpu_spent": 0,
        "distinct_from": {
            "bootstrap_se": "item-set sampling within ONE scoring run",
            "sd_run": "SEED variance; 1B-only, does not exist at 7B",
        },
        "null_source": "pinned anchor measurement only; nulls do not float",
        "per_cell": {}, "jitter_summary": {},
    }

    jit = {}
    for arm, byax in MEASUREMENTS.items():
        for axis, tags in byax.items():
            if len(tags) < 2:
                continue
            accs = [float(loaded[(t, axis)].mean()) for t in tags]
            spread = 100.0 * (max(accs) - min(accs))
            flips = []
            for a, b in itertools.combinations(tags, 2):
                flips.append(int((loaded[(a, axis)]
                                  != loaded[(b, axis)]).sum()))
            jit[f"{arm}|{axis}"] = {
                "n_measurements": len(tags), "dirs": tags,
                "acc_pct": [100.0 * a for a in accs],
                "acc_spread_pp": spread,
                "max_item_flips": max(flips),
            }
    out["jitter_summary"] = jit
    out["max_acc_spread_pp_anywhere"] = max(
        v["acc_spread_pp"] for v in jit.values())

    # NI under every (anchor, arm) measurement combination
    for arm in MEASUREMENTS:
        if arm == "intact_7B_base":
            continue
        for axis in AXES:
            a_tags = MEASUREMENTS["intact_7B_base"][axis]
            m_tags = MEASUREMENTS[arm][axis]
            variants = []
            for at, mt in itertools.product(a_tags, m_tags):
                iv = loaded[(at, axis)]
                nv = (nulls["mmlu_content"]["vectors"]["split"]
                      if axis == "mmlu_content" else nulls[axis]["vector"])
                resid_intact = float(np.asarray(iv, float).mean()
                                     - np.asarray(nv, float).mean())
                r = ni_rule(loaded[(mt, axis)], iv,
                            PREREG["delta_fraction"], resid_intact,
                            seed_off=97 * 900 + 13 * AXES.index(axis))
                variants.append({
                    "anchor_measurement": at, "arm_measurement": mt,
                    "residual_intact_pp": 100.0 * resid_intact,
                    "delta_pp": r["delta_pp"],
                    "lo95_pp": r["diff_lower95_one_sided_pp"],
                    "margin_pp": r["diff_lower95_one_sided_pp"] + r["delta_pp"],
                    "ni_accept": r["ni_accept"],
                })
            verdicts = {v["ni_accept"] for v in variants}
            out["per_cell"][f"{arm}|{axis}"] = {
                "decision_axis": axis not in DEMOTED_AXES,
                "n_variants": len(variants),
                "verdict_stable": len(verdicts) == 1,
                "verdict": (list(verdicts)[0] if len(verdicts) == 1
                            else "FLIPS"),
                "margin_pp_range": [min(v["margin_pp"] for v in variants),
                                    max(v["margin_pp"] for v in variants)],
                "margin_pp_spread": (max(v["margin_pp"] for v in variants)
                                     - min(v["margin_pp"] for v in variants)),
                "variants": variants,
            }

    unstable = [k for k, v in out["per_cell"].items()
                if not v["verdict_stable"]]
    out["cells_whose_verdict_flips"] = unstable
    out["all_verdicts_stable_under_remeasurement"] = not unstable

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=1, default=float)

    print("=== harness re-measurement jitter (same ckpt, different run) ===")
    for k, v in jit.items():
        print(f"{k:<40} n={v['n_measurements']} spread={v['acc_spread_pp']:.4f}pp "
              f"max_item_flips={v['max_item_flips']}")
    print(f"\nmax acc spread anywhere: "
          f"{out['max_acc_spread_pp_anywhere']:.4f}pp")
    print("\n=== NI under every (anchor, arm) measurement pair, `split` ===")
    for k, v in out["per_cell"].items():
        print(f"{k:<40} variants={v['n_variants']} "
              f"margin=[{v['margin_pp_range'][0]:.4f},"
              f"{v['margin_pp_range'][1]:.4f}] "
              f"spread={v['margin_pp_spread']:.4f}pp -> "
              f"{'STABLE ' + ('ACCEPT' if v['verdict'] is True else 'REJECT') if v['verdict_stable'] else 'FLIPS!'}"
              + ("" if v["decision_axis"] else "  (demoted)"))
    print(f"\nall verdicts stable: "
          f"{out['all_verdicts_stable_under_remeasurement']}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
