#!/usr/bin/env python3
"""paperC task #251 follow-up — BEFORE/AFTER audit of the MMLU-Pro truncation fix.

WHAT THIS ANSWERS
-----------------
#251's cross-family run shipped 10 of 15 cells with `n_trunc > 0` (llama2_7b 40,
qwen3_8b_base 20, on every rung; llama3_8b and OLMo-2 0). The re-run raises the
cap to MAXLEN=2048, which clears the global max encoded length (1678) for all
four tokenizers, so `n_trunc == 0` everywhere. Two questions must be answered
with numbers, not assertions:

  Q1. Did any per-cell letter accuracy actually move?
  Q2. Did the "AT the floor" verdict change on any cell?

The a-priori worst case was arithmetically real: 40/12032 = 0.332% of items,
versus damaged-cell effect sizes of 0.1-0.9 pp, so if every truncated item had
flipped it could have moved a cell by ~0.33 pp -- the SAME order as the effect.
This script measures what actually happened instead of bounding it.

It also reports the per-item flip set restricted to the truncation-affected
items, which is the only place a difference is *possible*: outside those items
the two runs see byte-identical inputs, and `use_cache=False` (the other change,
needed to fix the llama2_7b_base OOM) is asserted bit-identical by
`eval_olmo2_mc_letter_content.py --selftest`. So a flip on an UNaffected item
would indicate nondeterminism, not the fix, and is reported separately.

Usage:
  python mmlu_pro_trunc_fix_compare.py <old_root> <new_root> <audit_json> <out_json>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from mmlu_pro_power_nulls import (  # noqa: E402
    LETTERS, TASK, EXPECTED_N, N_BOOT, BOOT_SEED, MMLU_REFERENCE_EFFECT_PP,
    XF_FAMILIES, XF_RUNGS, XF_DAMAGED,
    best_constant_letter, correct_vec, load_records, paired_boot, verdict,
)


def cell_stats(recs):
    """letter acc + floor test, using the SAME estimators as the main table."""
    bc_letter, bc_vec, _diag = best_constant_letter(recs)
    cv = correct_vec(recs, "letter")
    d = cv - bc_vec
    m, lo, hi, p = paired_boot(d)
    modal = None
    preds = [r["letter"].get("pred_letter") for r in recs]
    if preds:
        from collections import Counter
        modal = Counter(preds).most_common(1)[0][1] / len(preds)
    return {
        "letter_acc": float(cv.mean()),
        "floor_letter": bc_letter,
        "floor_acc": float(bc_vec.mean()),
        "delta_pp": 100 * m,
        "ci95_pp": [100 * lo, 100 * hi],
        "ci95_half_width_pp": (100 * hi - 100 * lo) / 2,
        "boot_p": p,
        "verdict": verdict(m, p),
        "modal_letter_share": modal,
        "_cv": cv,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old_root")
    ap.add_argument("new_root")
    ap.add_argument("audit_json")
    ap.add_argument("out_json")
    args = ap.parse_args()

    audit = json.load(open(args.audit_json))
    affected = {f: set(v["affected_item_ids"])
                for f, v in audit["families"].items()}
    union = set(audit["union_affected_item_ids"])

    out = {
        "what": "before/after comparison of the MMLU-Pro cross-family cells "
                "across the #251 truncation fix (MAXLEN 1536 -> 2048) and the "
                "use_cache=False OOM fix",
        "old_root": args.old_root, "new_root": args.new_root,
        "old_max_len": audit["max_len_probed"], "new_max_len": 2048,
        "truncation_audit": {
            "union_affected_item_ids": sorted(union),
            "n_union_affected": len(union),
            "global_max_encoded_tokens": audit["global_max_encoded_tokens"],
            "per_family": {f: {"n_trunc_at_1536": v["n_trunc_at_probe"],
                               "affected_item_ids": v["affected_item_ids"],
                               "max_encoded_tokens": v["max_encoded_tokens"],
                               "vocab_size": v["vocab_size"]}
                           for f, v in audit["families"].items()},
        },
        "estimators": {"n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                       "reference_effect_pp": MMLU_REFERENCE_EFFECT_PP},
        "cells": {},
    }

    n_verdict_changed = 0
    n_cells = 0
    max_abs_dacc_pp = 0.0
    flips_outside_affected_total = 0

    for fam in XF_FAMILIES:
        aff = affected.get(fam, set())
        for rung in XF_RUNGS:
            arm = f"{fam}_{rung}"
            newd = os.path.join(args.new_root, arm)
            if not glob.glob(os.path.join(
                    newd, f"per_example_{TASK}_shard*of8.jsonl")):
                continue
            new_recs, new_integ = load_records(args.new_root, arm)
            ns = cell_stats(new_recs)

            oldd = os.path.join(args.old_root, arm)
            # A BEFORE cell exists only if it is COMPLETE. llama2_7b_base has
            # 3/8 shards in the old root (it OOMed on 5/8), which is exactly why
            # the driver refused to merge it -- so it is a NEW-ONLY cell here,
            # not a comparable one. Counting shards rather than testing for any
            # shard keeps that distinction, and load_records' 8/8 assert stays
            # a real guard instead of something this script trips over.
            old_shards = glob.glob(os.path.join(
                oldd, f"per_example_{TASK}_shard*of8.jsonl"))
            has_old = len(old_shards) == 8
            entry = {
                "family": fam, "rung": rung,
                "damaged": rung in XF_DAMAGED,
                "n_trunc_before": audit["families"].get(fam, {}).get(
                    "n_trunc_at_probe", 0),
                "n_trunc_after": 0,
                "integrity_after": new_integ,
                "after": {k: v for k, v in ns.items() if not k.startswith("_")},
            }
            if has_old:
                old_recs, old_integ = load_records(args.old_root, arm)
                os_ = cell_stats(old_recs)
                entry["integrity_before"] = old_integ
                entry["before"] = {k: v for k, v in os_.items()
                                   if not k.startswith("_")}
                # per-item flips
                oid = {r["item_id"]: r for r in old_recs}
                nid = {r["item_id"]: r for r in new_recs}
                assert set(oid) == set(nid), f"{arm}: item id sets differ"
                flips, flips_aff, flips_out = [], [], []
                for i in sorted(oid):
                    a = oid[i]["letter"].get("pred_letter")
                    b = nid[i]["letter"].get("pred_letter")
                    if a != b:
                        flips.append(i)
                        (flips_aff if i in aff else flips_out).append(i)
                d_acc_pp = 100 * (ns["letter_acc"] - os_["letter_acc"])
                entry["delta"] = {
                    "letter_acc_pp": d_acc_pp,
                    "floor_delta_pp_change": ns["delta_pp"] - os_["delta_pp"],
                    "boot_p_before": os_["boot_p"], "boot_p_after": ns["boot_p"],
                    "verdict_before": os_["verdict"],
                    "verdict_after": ns["verdict"],
                    "verdict_changed": os_["verdict"] != ns["verdict"],
                    "n_argmax_flips_total": len(flips),
                    "n_argmax_flips_on_truncation_affected_items": len(flips_aff),
                    "n_argmax_flips_outside_affected_items": len(flips_out),
                    "flip_item_ids_affected": flips_aff,
                    "flip_item_ids_outside": flips_out[:50],
                    "floor_unchanged": os_["floor_acc"] == ns["floor_acc"],
                }
                n_cells += 1
                n_verdict_changed += int(os_["verdict"] != ns["verdict"])
                max_abs_dacc_pp = max(max_abs_dacc_pp, abs(d_acc_pp))
                flips_outside_affected_total += len(flips_out)
                print(f"[cmp] {arm:22s} letter {os_['letter_acc']:.6f} -> "
                      f"{ns['letter_acc']:.6f} ({d_acc_pp:+.4f} pp)  "
                      f"floor_d {os_['delta_pp']:+.3f} -> {ns['delta_pp']:+.3f} pp  "
                      f"p {os_['boot_p']:.4f} -> {ns['boot_p']:.4f}  "
                      f"flips={len(flips)} (aff {len(flips_aff)} / out "
                      f"{len(flips_out)})  "
                      f"{'VERDICT CHANGED' if os_['verdict'] != ns['verdict'] else 'verdict same'}",
                      flush=True)
            else:
                entry["before"] = None
                entry["delta"] = {
                    "note": "no COMPLETE before cell -- this arm produced no "
                            "scoreable output in the original run",
                    "n_before_shards": len(old_shards),
                    "before_failure_mode": (
                        "CUDA OOM on 5/8 shards. Llama-2 has "
                        "num_key_value_heads=32 (no GQA) x 32 layers = 72.0 GiB "
                        "of fp32 KV at B=48/L=1536, vs 18.0 (llama3) and 20.2 "
                        "(qwen3), both num_kv_heads=8 -- so only the INTACT "
                        "llama2 arm OOMed. Fixed by use_cache=False."),
                }
                print(f"[cmp] {arm:22s} NEW ONLY ({len(old_shards)}/8 before) "
                      f"letter={ns['letter_acc']:.6f} "
                      f"floor_d={ns['delta_pp']:+.3f} pp p={ns['boot_p']:.4f} "
                      f"{ns['verdict']}", flush=True)
            out["cells"][arm] = entry

    out["summary"] = {
        "n_cells_with_before_and_after": n_cells,
        "n_verdict_changed": n_verdict_changed,
        "max_abs_letter_acc_change_pp": max_abs_dacc_pp,
        "n_argmax_flips_outside_affected_items_total": flips_outside_affected_total,
        "interpretation": (
            "flips are only POSSIBLE on the truncation-affected items; a flip "
            "outside them would mean nondeterminism, not the fix"),
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("=" * 72)
    print(f"cells compared      : {n_cells}")
    print(f"verdicts changed    : {n_verdict_changed}")
    print(f"max |d letter acc|  : {max_abs_dacc_pp:.4f} pp")
    print(f"flips outside aff.  : {flips_outside_affected_total}")
    print("=" * 72)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
