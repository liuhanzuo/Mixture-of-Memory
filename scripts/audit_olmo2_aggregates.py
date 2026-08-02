#!/usr/bin/env python3
"""P0.7 raw-JSON aggregate audit for the OLMo-2 prune/heal (Paper B) arms.

Recompute every reported aggregate straight from each arm's raw ``summary.json``
using the FIXED metric keys mandated by TODOList.md §P0.7, so we stop mixing
``acc`` vs ``acc_norm`` and stop calling a heterogeneous task average
"knowledge recovery".

FIXED OUTPUT SPEC (P0.7):
  core6      = mean( HellaSwag / ARC-Challenge / ARC-Easy / PIQA / OpenBookQA  [acc_norm];
                     WinoGrande [acc] )
  aux4_raw   = mean( LAMBADA(openai) / BoolQ / CommonsenseQA / SocialIQA  [plain acc] )
               -- descriptive mean over heterogeneous tasks only. NOT knowledge.
  MMLU       = reported SEPARATELY (plain acc). Never folded into a "knowledge" aggregate.
  aux5_raw   = mean( aux4 tasks + MMLU  [plain acc] )  -- optional, MUST be named aux5_raw,
               MUST NOT be called knowledge recovery. (== the historical "know5" set.)
  mmlu_recovery = (MMLU - 0.25) / (base_MMLU - 0.25)   -- above-chance recovery vs base full.

Usage:
  python3 scripts/audit_olmo2_aggregates.py                # default manifest, writes MD/CSV/JSON
  python3 scripts/audit_olmo2_aggregates.py --manifest my.json
  python3 scripts/audit_olmo2_aggregates.py --no-write     # print only

All aggregates are arithmetic means of the raw per-task ratios and are cross-checked
against the raw JSON to < 1e-6 by construction (we read the same numbers).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

# ----------------------------------------------------------------------------
# Fixed metric keys (P0.7). Do not edit without updating the docstring + audit MD.
# ----------------------------------------------------------------------------
CORE6_ACC_NORM = ["hellaswag", "arc_challenge", "arc_easy", "piqa", "openbookqa"]
CORE6_ACC = ["winogrande"]
AUX4_ACC = ["lambada_openai", "boolq", "commonsense_qa", "social_iqa"]
MMLU_KEY = "mmlu"
CHANCE = 0.25  # 4-way MC chance for MMLU

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------
# Default manifest: every arm needed for the P0.7 table.
#   core6 -> summary.json holding the 6 reasoning tasks
#   know  -> *_know summary.json holding mmlu + aux4
#   ppl   -> optional held-out PPL summary.json (ppl field)
# All paths are relative to REPO root. The keep10/keep12/shortgpt16 arms were
# pulled from diskB .82 and materialized under paperB/data/raw/... (see audit MD).
# ----------------------------------------------------------------------------
RAW = "paperB/data/raw/olmo2_downstream_results"
RAW_PPL = "paperB/data/raw/olmo2_ppl_results"
TOP = "olmo2_downstream_results"
TOP_PPL = "olmo2_ppl_results"

DEFAULT_MANIFEST: List[Dict[str, Optional[str]]] = [
    {
        "arm": "base_full",
        "label": "base full (OLMo-2-7B, 32L)",
        "is_base": True,
        "core6": f"{RAW}/7B_base_full/summary.json",
        "know": f"{RAW}/7B_base_full_know/summary.json",
        "ppl": f"{RAW_PPL}/7B_base_full/summary.json",
    },
    {
        "arm": "keep8_step121000",
        "label": "keep8 @121k (headline)",
        "core6": f"{TOP}/7B_keep8_step121000/summary.json",
        "know": f"{TOP}/7B_keep8_step121000_know/summary.json",
        "ppl": f"{TOP_PPL}/7B_keep8_step121000/summary.json",
    },
    {
        "arm": "keep8_step110000",
        "label": "keep8 @110k (traj)",
        "core6": f"{TOP}/7B_keep8_step110000/summary.json",
        "know": f"{TOP}/7B_keep8_step110000_know/summary.json",
    },
    {
        "arm": "keep8_step100000",
        "label": "keep8 @100k (traj)",
        "core6": f"{TOP}/7B_keep8_step100000/summary.json",
        "know": f"{TOP}/7B_keep8_step100000_know/summary.json",
    },
    {
        "arm": "keep10_step83500",
        "label": "keep10 @83.5k (headline, plateau=~200k)",
        "core6": f"{RAW}/7B_keep10_step83500/summary.json",
        "know": f"{RAW}/7B_keep10_step83500_know/summary.json",
    },
    {
        "arm": "keep12_step124000",
        "label": "keep12 @124k (headline, plateau=~200k)",
        "core6": f"{RAW}/7B_keep12_step124000/summary.json",
        "know": f"{RAW}/7B_keep12_step124000_know/summary.json",
    },
    {
        "arm": "keep14_step200000",
        "label": "keep14 @200k (headline)",
        "core6": f"{TOP}/7B_keep14_step200000/summary.json",
        "know": f"{TOP}/7B_keep14_step200000_know/summary.json",
        "ppl": f"{RAW_PPL}/7B_keep14_step200000/summary.json",
    },
    {
        "arm": "shortgpt16_step200000",
        "label": "ShortGPT-16 @200k (trained, headline)",
        "core6": f"{RAW}/7B_shortgpt16_step200000/summary.json",
        "know": f"{RAW}/7B_shortgpt16_step200000_know/summary.json",
    },
    {
        "arm": "shortgpt_step0",
        "label": "ShortGPT-16 @0 (no-heal baseline, keep16 fresh0)",
        "core6": f"{RAW}/7B_shortgpt_step0/summary.json",
        "know": f"{RAW}/7B_shortgpt_step0_know/summary.json",
        "ppl": f"{RAW_PPL}/7B_shortgpt_step0/summary.json",
    },
    {
        "arm": "freezefront_step200000",
        "label": "frozen-front @200k",
        "core6": f"{RAW}/7B_freezefront_step200000/summary.json",
        "know": f"{RAW}/7B_freezefront_step200000_know/summary.json",
        "ppl": f"{RAW_PPL}/7B_freezefront_step200000/summary.json",
    },
    {
        "arm": "scratch16L_step200000",
        "label": "random-init (from-scratch 16L) @200k",
        "core6": f"{RAW}/7B_scratch16L_step200000/summary.json",
        "know": f"{RAW}/7B_scratch16L_step200000_know/summary.json",
        "ppl": f"{RAW_PPL}/7B_scratch16L_step200000/summary.json",
    },
]

# Historical published aggregate values that P0.7 must vet. Each entry:
#   arm, old_value, what it was called, which recompute it matches.
OLD_VALUES = {
    "base_full": {"know5_or_aux5": 0.6639},
    "keep10_step83500": {"know5_or_aux5": 0.4491},
    "keep12_step124000": {"know5_or_aux5": 0.4608},
    "keep14_step200000": {"know5_or_aux5": 0.5071},
    "shortgpt16_step200000": {"know5_or_aux5": 0.5596},
}


def load_tasks(path: str) -> Dict[str, dict]:
    full = path if os.path.isabs(path) else os.path.join(REPO, path)
    with open(full) as f:
        return json.load(f)["tasks"]


def get(tasks: Dict[str, dict], name: str, key: str):
    if name not in tasks:
        return None
    return tasks[name].get(key)


def mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def compute_arm(entry: dict) -> dict:
    """Return per-arm breakdown + aggregates, all straight from raw JSON."""
    core_tasks = load_tasks(entry["core6"]) if entry.get("core6") else {}
    know_tasks = load_tasks(entry["know"]) if entry.get("know") else {}

    members = []  # (task, metric_key, n, value)

    core6_vals = []
    for t in CORE6_ACC_NORM:
        v = get(core_tasks, t, "acc_norm")
        n = get(core_tasks, t, "n")
        members.append((t, "acc_norm", n, v))
        core6_vals.append(v)
    for t in CORE6_ACC:
        v = get(core_tasks, t, "acc")
        n = get(core_tasks, t, "n")
        members.append((t, "acc", n, v))
        core6_vals.append(v)
    core6 = mean(core6_vals)

    aux4_vals = []
    for t in AUX4_ACC:
        v = get(know_tasks, t, "acc")
        n = get(know_tasks, t, "n")
        members.append((t, "acc", n, v))
        aux4_vals.append(v)
    aux4_raw = mean(aux4_vals)

    mmlu = get(know_tasks, MMLU_KEY, "acc")
    mmlu_n = get(know_tasks, MMLU_KEY, "n")
    members.append((MMLU_KEY, "acc", mmlu_n, mmlu))

    aux5_vals = list(aux4_vals) + ([mmlu] if mmlu is not None else [])
    aux5_raw = mean(aux5_vals)

    ppl = None
    if entry.get("ppl"):
        full = entry["ppl"] if os.path.isabs(entry["ppl"]) else os.path.join(REPO, entry["ppl"])
        if os.path.exists(full):
            with open(full) as f:
                ppl = json.load(f).get("ppl")

    return {
        "arm": entry["arm"],
        "label": entry.get("label", entry["arm"]),
        "is_base": bool(entry.get("is_base")),
        "members": members,
        "core6": core6,
        "aux4_raw": aux4_raw,
        "mmlu": mmlu,
        "aux5_raw": aux5_raw,
        "ppl": ppl,
        "core6_path": entry.get("core6"),
        "know_path": entry.get("know"),
    }


def fmt(x, nd=4):
    return "" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="JSON list of arm entries; defaults to built-in.")
    ap.add_argument("--out-md", default="paperB/P0_7_AGGREGATE_AUDIT.md")
    ap.add_argument("--out-csv", default="paperB/P0_7_aggregate_audit.csv")
    ap.add_argument("--out-json", default="paperB/P0_7_aggregate_audit.json")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    manifest = DEFAULT_MANIFEST
    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)

    results = []
    for entry in manifest:
        try:
            results.append(compute_arm(entry))
        except FileNotFoundError as e:
            print(f"[MISSING] {entry['arm']}: {e}", file=sys.stderr)
            results.append({"arm": entry["arm"], "label": entry.get("label", entry["arm"]),
                            "missing": str(e)})

    base = next((r for r in results if r.get("is_base")), None)
    base_mmlu = base["mmlu"] if base else None

    # ---- console: per-arm member breakdown -------------------------------
    for r in results:
        if r.get("missing"):
            print(f"\n### {r['arm']}  [MISSING] {r['missing']}")
            continue
        print(f"\n### {r['label']}  ({r['arm']})")
        print(f"  core6 path: {r['core6_path']}")
        print(f"  know  path: {r['know_path']}")
        for task, key, n, v in r["members"]:
            print(f"    {task:16s} {key:9s} n={str(n):6s} {fmt(v,6)}")
        rec = None
        if base_mmlu and r["mmlu"] is not None and (base_mmlu - CHANCE) != 0:
            rec = (r["mmlu"] - CHANCE) / (base_mmlu - CHANCE)
        r["mmlu_recovery"] = rec
        print(f"  => core6={fmt(r['core6'])}  aux4_raw={fmt(r['aux4_raw'])}  "
              f"MMLU={fmt(r['mmlu'])}  aux5_raw={fmt(r['aux5_raw'])}  "
              f"mmlu_recovery={fmt(rec*100,1)+'%' if rec is not None else ''}")

    # ---- old-value discrepancy check -------------------------------------
    disc = []
    for r in results:
        if r.get("missing"):
            continue
        old = OLD_VALUES.get(r["arm"], {}).get("know5_or_aux5")
        if old is None:
            continue
        aux4 = r["aux4_raw"]
        aux5 = r["aux5_raw"]
        # mixed-metric recompute that used acc_norm for boolq/csqa/siqa (the bug):
        kt = load_tasks(next(e["know"] for e in manifest if e["arm"] == r["arm"]))
        mixed = mean([
            get(kt, "mmlu", "acc"), get(kt, "lambada_openai", "acc"),
            get(kt, "boolq", "acc_norm"), get(kt, "commonsense_qa", "acc_norm"),
            get(kt, "social_iqa", "acc_norm"),
        ])
        matches_plain5 = abs(old - aux5) < 5e-4
        matches_mixed = abs(old - mixed) < 5e-4
        diag = ("OK: old == aux5_raw (plain-acc 5-task mean)" if matches_plain5 else
                "BUG: old == acc_norm-CONTAMINATED mean (boolq/csqa/siqa used acc_norm)"
                if matches_mixed else "UNRESOLVED: old matches neither plain nor mixed")
        disc.append({
            "arm": r["arm"], "old": old, "aux5_raw_plain": aux5,
            "mixed_normBCS": mixed, "aux4_raw": aux4, "diag": diag,
        })
        print(f"\n[OLD-CHECK] {r['arm']}: old={old:.4f}  aux5_raw(plain)={aux5:.4f}  "
              f"mixed(normBCS)={mixed:.4f}  -> {diag}")

    if args.no_write:
        return

    # ---- CSV --------------------------------------------------------------
    csv_path = os.path.join(REPO, args.out_csv)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "label", "core6", "aux4_raw", "mmlu", "aux5_raw",
                    "mmlu_above_chance_recovery", "ppl", "core6_path", "know_path"])
        for r in results:
            if r.get("missing"):
                w.writerow([r["arm"], r["label"], "", "", "", "", "", "MISSING", "", ""])
                continue
            rec = r.get("mmlu_recovery")
            w.writerow([r["arm"], r["label"], fmt(r["core6"], 6), fmt(r["aux4_raw"], 6),
                        fmt(r["mmlu"], 6), fmt(r["aux5_raw"], 6),
                        fmt(rec, 6) if rec is not None else "",
                        fmt(r["ppl"], 4) if r["ppl"] is not None else "",
                        r["core6_path"], r["know_path"]])

    # ---- JSON -------------------------------------------------------------
    json_path = os.path.join(REPO, args.out_json)
    payload = {
        "spec": "P0.7 fixed keys: core6(acc_norm x5 + winogrande acc), "
                "aux4_raw(plain acc x4, no mmlu), mmlu separate(plain acc), "
                "aux5_raw(aux4+mmlu plain acc), recovery=(mmlu-0.25)/(base_mmlu-0.25)",
        "base_mmlu": base_mmlu,
        "arms": [
            {k: v for k, v in r.items() if k != "members"} | {
                "members": [
                    {"task": t, "metric": key, "n": n, "value": v}
                    for (t, key, n, v) in r.get("members", [])
                ]
            }
            for r in results
        ],
        "old_value_discrepancies": disc,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {args.out_csv}\nWrote {args.out_json}")
    print("(Markdown report P0_7_AGGREGATE_AUDIT.md is maintained alongside.)")


if __name__ == "__main__":
    main()
