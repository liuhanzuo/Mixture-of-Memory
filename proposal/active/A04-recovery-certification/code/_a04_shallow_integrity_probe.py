#!/usr/bin/env python3
"""INDEPENDENT shard-integrity probe for the A04 shallow-rung ladder cells.

Deliberately does NOT import the analysis script's loaders. The point is to be a
second, structurally independent witness to the completeness assertions the
analysis will also make:

  * shard index SET exactly {0..7}   -- a SET, never a count of 8 files
  * merged n exactly EXPECTED_N       -- 17944/14267/3610/14042
  * 0 duplicate item_id
  * 0 nan
  * chat_template is False            -- asserted `is not False` -> FAIL, so
                                         None / missing / True all fail
  * add_bos is False                  -- `is False`, never `is not True`

A silently merged 5-of-8 shard set has corrupted results in this repo before, and
a driver-side check can be bypassed by a later caller, so this check reads the
per-shard files itself. CPU only, read-only.

Usage: python _a04_shallow_integrity_probe.py <RAW_ROOT>
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

EXPECTED_N = {"triviaqa": 17944, "popqa": 14267, "nq_open": 3610, "mmlu": 14042}

CELLS = {
    "intact_anchor":      {"mmlu": "A03_1B_base", "cb": "A03_1B_base", "nq": "A03_1B_base_nq"},
    "keep13f2_step5000":  {"mmlu": "A04_1B_shallow_keep13_seed101_step5000",
                           "cb": "A04_1B_shallow_keep13_seed101_step5000",
                           "nq": "A04_1B_shallow_keep13_seed101_step5000_nq"},
    "keep14f2_step5000":  {"mmlu": "A04_1B_shallow_keep14_seed101_step5000",
                           "cb": "A04_1B_shallow_keep14_seed101_step5000",
                           "nq": "A04_1B_shallow_keep14_seed101_step5000_nq"},
    "keep12f2_step5000_REF": {"mmlu": "A04_1B_stageB_keep12_seed101_step5000",
                              "cb": "A04_1B_stageB_keep12_seed101_step5000",
                              "nq": "A04_1B_stageB_keep12_seed101_step5000_nq"},
}


def read_shards(d, stem):
    files = sorted(glob.glob(os.path.join(d, f"per_example_{stem}_shard*of8.jsonl")))
    idx = sorted(int(os.path.basename(f).split("shard")[-1].split("of")[0]) for f in files)
    rows = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return idx, rows, files


def probe(raw_root):
    out, hard_fail = {}, []
    for cell, spec in CELLS.items():
        out[cell] = {}
        for axis, (root, tag, stem) in {
            "mmlu_content": ("olmo2_mmlu_content_results", spec["mmlu"], "mmlu"),
            "triviaqa":     ("olmo2_closedbook_results",   spec["cb"],   "triviaqa"),
            "popqa":        ("olmo2_closedbook_results",    spec["cb"],   "popqa"),
            "nq_open":      ("olmo2_closedbook_results",    spec["nq"],   "nq_open"),
        }.items():
            d = os.path.join(raw_root, root, tag)
            rec = {"dir": d}
            if not os.path.isdir(d):
                rec["EXISTS"] = False
                hard_fail.append(f"{cell}|{axis}: dir missing {d}")
                out[cell][axis] = rec
                continue
            idx, rows, files = read_shards(d, stem)
            exp = EXPECTED_N[stem]
            ids = [r.get("item_id", r.get("idx")) for r in rows]
            n_dup = len(ids) - len(set(ids))

            # nan: an explicit nan flag on the row, plus a float-nan sweep of the
            # metric field the scorer actually consumes.
            n_nan_flag = sum(1 for r in rows if r.get("nan") is True)
            n_nan_metric = 0
            if stem == "mmlu":
                for r in rows:
                    cn = r.get("content_norm")
                    if not isinstance(cn, dict) or not isinstance(cn.get("correct"), bool):
                        n_nan_metric += 1
            else:
                for r in rows:
                    v = r.get("em")
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        n_nan_metric += 1

            sp = os.path.join(d, "summary.json")
            ct = ab = mnt = "MISSING_SUMMARY"
            meta = {}
            if os.path.isfile(sp):
                blob = json.load(open(sp))
                meta = blob.get("meta", {}) or {}
                ct = blob.get("chat_template", meta.get("chat_template", False))
                ab = blob.get("add_bos", meta.get("add_bos"))
                mnt = meta.get("max_new_tokens")

            rec.update({
                "EXISTS": True,
                "shard_index_set": idx,
                "shard_index_set_is_0_to_7": idx == list(range(8)),
                "n_shard_files": len(files),
                "n_items": len(rows), "n_items_expected": exp,
                "n_items_exact": len(rows) == exp,
                "n_duplicate_item_ids": n_dup,
                "n_nan_flag_rows": n_nan_flag,
                "n_nan_or_malformed_metric": n_nan_metric,
                "chat_template": ct,
                "chat_template_ok_is_not_False_fails": ct is False,
                "add_bos": ab, "add_bos_ok_is_False": ab is False,
                "max_new_tokens": mnt,
                "meta_keep_front_layers": meta.get("keep_front_layers"),
                "meta_n_fresh_layers": meta.get("n_fresh_layers"),
                "meta_num_hidden_layers": meta.get("num_hidden_layers"),
                "meta_mode": meta.get("mode"),
                "meta_ckpt": meta.get("ckpt"),
                "meta_ckpt_step": meta.get("ckpt_step"),
                "meta_base_model": meta.get("base_model"),
            })
            for k, msg in (("shard_index_set_is_0_to_7", "shard set != {0..7}"),
                           ("n_items_exact", "item count != EXPECTED_N"),
                           ("chat_template_ok_is_not_False_fails", "chat_template is not False"),
                           ("add_bos_ok_is_False", "add_bos is not False")):
                if not rec[k]:
                    hard_fail.append(f"{cell}|{axis}: {msg} ({rec.get(k)!r})")
            if n_dup:
                hard_fail.append(f"{cell}|{axis}: {n_dup} duplicate item_id")
            if n_nan_flag or n_nan_metric:
                hard_fail.append(f"{cell}|{axis}: nan flag {n_nan_flag} / "
                                 f"malformed metric {n_nan_metric}")
            out[cell][axis] = rec

    # cross-cell item_id alignment: the paired difference is meaningless if the
    # cells are not the same item set in the same order.
    align = {}
    for axis, (root, key, stem) in {
        "mmlu_content": ("olmo2_mmlu_content_results", "mmlu", "mmlu"),
        "triviaqa":     ("olmo2_closedbook_results",   "cb",   "triviaqa"),
        "popqa":        ("olmo2_closedbook_results",   "cb",   "popqa"),
        "nq_open":      ("olmo2_closedbook_results",   "nq",   "nq_open"),
    }.items():
        seqs = {}
        for cell, spec in CELLS.items():
            d = os.path.join(raw_root, root, spec[key])
            if not os.path.isdir(d):
                continue
            _, rows, _ = read_shards(d, stem)
            seqs[cell] = [r.get("item_id", r.get("idx")) for r in rows]
        ref = CELLS and "intact_anchor"
        base = seqs.get(ref)
        align[axis] = {c: (v == base) for c, v in seqs.items()}
        if not all(align[axis].values()):
            hard_fail.append(f"{axis}: item_id sequences NOT identical across cells "
                             f"{align[axis]}")
    return {"per_cell": out, "item_id_alignment_vs_anchor": align,
            "hard_failures": hard_fail, "ALL_CLEAR": not hard_fail}


if __name__ == "__main__":
    r = probe(sys.argv[1])
    print(json.dumps(r, indent=2, sort_keys=False, default=str))
    print("\nALL_CLEAR =", r["ALL_CLEAR"])
    if not r["ALL_CLEAR"]:
        for f in r["hard_failures"]:
            print("  HARD FAIL:", f)
        sys.exit(1)
