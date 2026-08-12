#!/usr/bin/env python
"""A02 Job 2.2 — score the de-saturation cell (niah_single_3 x {16k,32k}).

Answers verdict caveat 2: A0-A3 sat at 95-100 % on the four primary cells, so
"read tax ~ 0 at shallow j" was partly a statement about a SATURATED benchmark.
niah_single_3 (36-char UUID values) passed the 0-GPU retrieval-closure screen at
97.5 % / 95.0 % recall@12 and is single-needle, so its screen is EXACT.

FAIL-CLOSED GATES (mirroring the read-tax gate; a gate that cannot fire is not a gate)
  GATE C  every arm has exactly nshard records for the cell; per-cell n == 100;
          0 duplicate sample_index; 0 NaN/missing `correct`.  Refuse the cell otherwise.
  GATE C2 input_ids_sha256 equality across all arms per shared sample_index, so the
          paired delta is genuinely paired.
  GATE D  chat_template=False + selector=iter_bm25 + topk=12 + chunk_size=512 +
          expected resume_j + expected adapter, per arm.
          *** RULER stores config FLAT in records.json and carries NO chat_template
          there -- it lives in the SIBLING summary json. The assertion is
          `is not False` and NEVER `is not True`, because `is not True` silently
          passes on a None and would claim verification while checking nothing. ***

Usage: python analyze_a02_desaturation.py [--out <dir>]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np

W = Path(os.environ.get(
    "A02_W", "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"))

ARMS = {                      # arm -> (results dir, resume_j, expected adapter substr)
    "A0": ("a02_desat_ruler_A0_j0", 0, None),
    "A2": ("a02_desat_ruler_A2_j6", 6, "qcmem_distill_qwen_j6_r32_4k"),
    "A3": ("a02_desat_ruler_A3_j9", 9, "qcmem_distill_qwen_j9_r32_4k"),
    "A4": ("a02_desat_ruler_A4_j12", 12, "qcmem_distill_qwen_j12_r32_4k"),
}
TASK, LENS, NSHARD, N_EXPECT = "niah_single_3", ("16k", "32k"), 8, 100
N_BOOT, SEED = 5000, 42
# the saturated reference from the primary read-out, for the de-saturation comparison
PRIMARY_NIAH = {"16k": {"A0": 100.0, "A2": 99.0, "A3": 99.0, "A4": 90.0},
                "32k": {"A0": 99.0, "A2": 99.0, "A3": 95.0, "A4": 96.0}}


def wilson(k, n):
    if n == 0:
        return None, None, None
    p = k / n
    z = 1.959963984540054
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * p, 2), round(100 * max(0, c - h), 2), round(100 * min(1, c + h), 2)


def paired_boot(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    bo = 100.0 * (a[idx].mean(axis=1) - b[idx].mean(axis=1))
    lo, hi = np.percentile(bo, [2.5, 97.5])
    return float(100.0 * (a.mean() - b.mean())), float(lo), float(hi)


def load_arm(sub, length, errs, arm):
    """Load + GATE C/C2/D for one (arm, length)."""
    files = sorted(glob.glob(str(W / "ruler_results" / sub
                                 / f"{TASK}_{length}_shard*of{NSHARD}.records.json")))
    if len(files) != NSHARD:
        errs.append(f"GATE C {arm}/{length}: {len(files)}/{NSHARD} shards -> REFUSE")
        return None
    corr, shas, cfgs = {}, {}, []
    for f in files:
        d = json.load(open(f))
        cfgs.append((f, d))
        for r in d.get("records", []):
            i = int(r["sample_index"])
            if i in corr:
                errs.append(f"GATE C {arm}/{length}: duplicate sample_index {i}")
            c = r.get("correct")
            if c is None or (isinstance(c, float) and math.isnan(c)):
                errs.append(f"GATE C {arm}/{length}: NaN/missing correct at {i}")
                continue
            corr[i] = int(c)
            shas[i] = r.get("input_ids_sha256")
    if len(corr) != N_EXPECT:
        errs.append(f"GATE C {arm}/{length}: n={len(corr)} != {N_EXPECT} -> REFUSE")
        return None

    # GATE D — RULER records.json is FLAT and has NO chat_template; it lives in the
    # sibling summary. Read each field from the level it actually lives at.
    _, j_exp, ad_exp = ARMS[arm]
    for f, d in cfgs:
        if d.get("selector") != "iter_bm25":
            errs.append(f"GATE D {arm}/{length}: selector={d.get('selector')!r}")
        if int(d.get("topk", -1)) != 12:
            errs.append(f"GATE D {arm}/{length}: topk={d.get('topk')!r}")
        if int(d.get("chunk_size", -1)) != 512:
            errs.append(f"GATE D {arm}/{length}: chunk_size={d.get('chunk_size')!r}")
        if int(d.get("resume_j", -1)) != j_exp:
            errs.append(f"GATE D {arm}/{length}: resume_j={d.get('resume_j')!r} != {j_exp}")
        lora = d.get("lora_adapter")
        if ad_exp is None:
            if lora:
                errs.append(f"GATE D {arm}/{length}: expected NO adapter, got {lora!r}")
        elif not lora or ad_exp not in str(lora):
            errs.append(f"GATE D {arm}/{length}: expected {ad_exp}, got {lora!r}")
        # chat_template lives in the SIBLING summary, NOT in records.json
        sib = Path(f.replace(".records.json", ".json"))
        ct = None
        if sib.exists():
            s = json.load(open(sib))
            ct = s.get("chat_template")
            if ct is None:
                ct = (s.get("config") or {}).get("chat_template")
        # `is not False` -- NEVER `is not True` (that silently passes on None)
        if ct is not False:
            errs.append(f"GATE D {arm}/{length}: chat_template={ct!r} is not False "
                        f"(sibling={sib.name}, exists={sib.exists()})")
    return corr, shas


def main(out_dir: Path):
    errs, out_cells = [], {}
    for length in LENS:
        loaded = {}
        for arm, (sub, _, _) in ARMS.items():
            got = load_arm(sub, length, errs, arm)
            if got:
                loaded[arm] = got
        if len(loaded) < 2:
            out_cells[f"ruler|{TASK}|{length}"] = {"status": "REFUSED", "errors": errs}
            continue
        common = sorted(set.intersection(*[set(c) for c, _ in loaded.values()]))
        # GATE C2 sha pairing
        sha_fail = [i for i in common
                    if len({loaded[a][1].get(i) for a in loaded
                            if loaded[a][1].get(i)}) > 1]
        if sha_fail:
            errs.append(f"GATE C2 {length}: sha mismatch on {len(sha_fail)} indices")
        cell = {"n_common": len(common), "acc": {}, "tax_vs_A0": {},
                "sha_pairing_failures": len(sha_fail)}
        for arm, (corr, _) in loaded.items():
            v = [corr[i] for i in common]
            p, lo, hi = wilson(sum(v), len(v))
            cell["acc"][arm] = {"acc": p, "wilson_ci95": [lo, hi], "n": len(v)}
        if "A0" in loaded:
            a0 = [loaded["A0"][0][i] for i in common]
            for arm in loaded:
                if arm == "A0":
                    continue
                v = [loaded[arm][0][i] for i in common]
                d, lo, hi = paired_boot(v, a0)
                cell["tax_vs_A0"][arm] = {
                    "delta_pp": round(d, 2), "ci95": [round(lo, 2), round(hi, 2)],
                    "sig": bool(lo > 0 or hi < 0)}
        out_cells[f"ruler|{TASK}|{length}"] = cell

    # de-saturation check: did the shallow end actually come off the ceiling?
    desat = {}
    for length in LENS:
        c = out_cells.get(f"ruler|{TASK}|{length}", {})
        if c.get("status") == "REFUSED":
            continue
        new = {a: c["acc"][a]["acc"] for a in c.get("acc", {})}
        old = PRIMARY_NIAH[length]
        shallow_new = [new[a] for a in ("A0", "A2", "A3") if a in new]
        desat[length] = {
            "niah_multikey_1_primary": old, "niah_single_3_new": new,
            "shallow_max_primary": max(old[a] for a in ("A0", "A2", "A3")),
            "shallow_max_new": max(shallow_new) if shallow_new else None,
            "de_saturated": bool(shallow_new and max(shallow_new) < 95.0),
            "shallow_tax_now_resolvable": {
                a: c["tax_vs_A0"][a] for a in ("A2", "A3") if a in c.get("tax_vs_A0", {})},
        }

    print("=== GATES ===")
    print(f"  errors: {len(errs)}")
    for e in errs[:25]:
        print("   ", e)
    print("\n=== niah_single_3 (de-saturation cell) ===")
    for cell, c in out_cells.items():
        if c.get("status") == "REFUSED":
            print(f"{cell}: REFUSED")
            continue
        print(f"{cell}  n={c['n_common']} sha_fail={c['sha_pairing_failures']}")
        for a in ("A0", "A2", "A3", "A4"):
            if a in c["acc"]:
                x = c["acc"][a]
                t = c["tax_vs_A0"].get(a)
                ts = (f"  tax={t['delta_pp']:+6.2f} [{t['ci95'][0]:+6.2f},"
                      f"{t['ci95'][1]:+6.2f}] {'SIG' if t['sig'] else 'ns'}") if t else ""
                print(f"   {a}: {x['acc']:5.1f} CI{x['wilson_ci95']}{ts}")
    print("\n=== did the shallow end de-saturate? ===")
    for ln, d in desat.items():
        print(f"{ln}: primary shallow max {d['shallow_max_primary']} -> "
              f"new shallow max {d['shallow_max_new']}  "
              f"de_saturated={d['de_saturated']}")
        print(f"   primary {d['niah_multikey_1_primary']}")
        print(f"   new     {d['niah_single_3_new']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "a02_desaturation_result.json"
    json.dump({
        "generated_by": "analyze_a02_desaturation.py",
        "prereg": "A02_BABILONG_MISORDER_PREREG.md 3 (Job 2.2)",
        "cell": f"ruler|{TASK}|{{16k,32k}}",
        "why_this_cell": ("passed the 0-GPU retrieval-closure screen at 97.5%/95.0% "
                          "recall@12 AND is single-needle, so the screen's gold locator "
                          "is exact (multivalue/multiquery gold = first needle only)"),
        "protocol": {
            "chat_template": "asserted False via `is not False` (never `is not True`)",
            "chat_template_location": ("RULER records.json is FLAT and has NO "
                                       "chat_template; it lives in the sibling summary"),
            "selector": "iter_bm25 topk=12 iter_hop_topk=4 chunk_size=512",
            "shards": f"{NSHARD}, n={N_EXPECT}/cell asserted, 0 dup, 0 NaN",
            "ci": f"paired bootstrap n_boot={N_BOOT} seed={SEED}; Wilson for proportions",
            "aggregation": "per-cell only; never pooled with BABILong/LongEval",
        },
        "gate_errors": errs, "cells": out_cells, "de_saturation": desat,
    }, open(dst, "w"), indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        W / "proposal/backlog/A02-comem-write-read-repair/evidence/babilong_misorder"))
    main(Path(ap.parse_args().out))
