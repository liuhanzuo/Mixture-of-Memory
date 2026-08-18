#!/usr/bin/env python
"""Paper A — #143 CacheBlend-style chunk-KV baseline: 8-shard merge -> aggregate.json.

Mirrors the #144 dense_selector aggregate schema
(all_tasks_reported / missing_required_cells / summary) but adapted to the #143
CacheBlend per-shard record layout, which reuses the QCMem family drivers:

  RULER    (eval_ruler_qcmem.py)   -> <task>_<len>_shardXof8.{json,records.json,csv}
                                       records[].{recall, correct} ; score = mean recall*100
  BABILong (eval_qcmem_babilong.py)-> qa5_<len>_<suffix>_shardXof8.{json,csv}
                                       top-level {n, correct} ; score = correct/n*100
  LoCoMo   (eval_qcmem_locomo.py)  -> preds_shardXof8.jsonl (+ eval_config, efficiency)
                                       scored via qlo.run_scoring (score_sample: F1/EM/substr-acc)

Dirs are named cb_<family>_<task>_<rTOK> where rTOK in {r00,r010,r015,r018}
(LoCoMo: cb_locomo_<rTOK>). Cell keys carry the r token so all recompute ratios
coexist in one summary. RULER end_to_end = mean string_match_all RECALL (the
official RULER metric == per-shard score/100; NOT exact-match, since
variable_tracking recall is fractional). BABILong end_to_end = compare_answers
correct fraction. LoCoMo end_to_end = substr acc (== #144 locomo end_to_end).

LLM-judge is DEFERRED to MAIN (use_llm_judge=False). CPU-only.

Usage:
  PYTHONPATH=<repo>:<repo>/scripts python scripts/aggregate_cacheblend_143.py \
      --cacheblend_dir bench_results/cacheblend
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

ROOT = os.environ.get("CB_REPO_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_qcmem_locomo as qlo  # canonical LoCoMo scoring (score_sample/run_scoring)

R_MAP = {"r00": 0.0, "r010": 0.10, "r015": 0.15, "r018": 0.18}

# expected completeness grid (mirrors #144 REQUIRE + adds per-length/r cells)
RULER_TASKS = ["niah_single_2", "niah_multikey_1", "variable_tracking"]
RULER_LENGTHS = ["4k", "8k", "16k", "32k"]
BABI_LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k"]
R_TOKENS = ["r00", "r010", "r015", "r018"]
LOCOMO_R = ["r00", "r010"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 6) if xs else None


def _round(x, nd=6):
    return round(x, nd) if x is not None else None


def aggregate(cb_dir: Path):
    summary = {"ruler": {}, "babilong": {}, "locomo": {}}
    provenance = {}

    # ---------------- RULER ----------------
    for d in sorted(cb_dir.glob("cb_ruler_*")):
        toks = d.name.split("_")
        rtok = toks[-1]
        task = "_".join(toks[2:-1])
        r = R_MAP.get(rtok)
        cells = collections.defaultdict(list)
        for f in d.glob("*.records.json"):
            base = f.name[:-len(".records.json")]
            base = base[:base.rfind("_shard")]
            cells[base].append(f)
        for cellbase, files in sorted(cells.items()):
            length = cellbase.split("_")[-1]
            recs = {}
            for f in sorted(files):
                obj = json.load(open(f))
                if rtok not in provenance.get("ruler", {}):
                    provenance.setdefault("ruler", {})[rtok] = {
                        k: obj.get(k) for k in
                        ("resume_j", "selector", "topk", "iter_hop_topk",
                         "iter_rounds", "chunk_size", "lora_adapter", "baseline",
                         "seed")}
                for rr in obj.get("records", []):
                    key = rr.get("sample_index", rr.get("input_ids_sha256"))
                    recs[key] = rr
            rows = list(recs.values())
            n = len(rows)
            mean_recall = _mean([rr.get("recall") for rr in rows])
            mean_correct = _mean([rr.get("correct") for rr in rows])
            lat, rl = [], []
            for jf in d.glob(f"{cellbase}_shard*of8.json"):
                s = json.load(open(jf)).get("summary", {})
                lat.append(s.get("avg_prefill_latency_ms"))
                rl.append(s.get("avg_read_len"))
            cellkey = f"{cellbase}_{rtok}"
            summary["ruler"][cellkey] = {
                "n": n,
                "n_expected": 100,
                "complete": n == 100,
                "end_to_end": mean_recall,
                "mean_recall": mean_recall,
                "exact_match": mean_correct,
                "recompute_ratio": r,
                "task": task,
                "length": length,
                "avg_prefill_latency_ms_mean": _mean(lat),
                "read_len_mean": _mean(rl),
                "metric": "string_match_all recall (== per-shard score/100)",
            }

    # ---------------- BABILong ----------------
    for d in sorted(cb_dir.glob("cb_babilong_*")):
        toks = d.name.split("_")
        rtok = toks[-1]
        task = toks[2]  # qa5
        r = R_MAP.get(rtok)
        cells = collections.defaultdict(list)
        for f in d.glob("*.json"):
            base = f.name[:-len(".json")]
            if "_shard" not in base:
                continue
            base = base[:base.rfind("_shard")]
            cells[base].append(f)
        for cellbase, files in sorted(cells.items()):
            length = cellbase.split("_")[1]
            tot_correct, tot_n, lat = 0, 0, []
            for f in sorted(files):
                obj = json.load(open(f))
                tot_correct += obj.get("correct", 0)
                tot_n += obj.get("n", 0)
                cb = obj.get("cacheblend", {})
                lat.append(cb.get("avg_prefill_latency_ms"))
                if rtok not in provenance.get("babilong", {}):
                    provenance.setdefault("babilong", {})[rtok] = {
                        **{k: obj.get("qcmem", {}).get(k) for k in
                           ("resume_j", "selector", "topk", "chunk_size",
                            "lora_adapter")},
                        "baseline": obj.get("baseline"),
                        "recompute_ratio": cb.get("recompute_ratio"),
                    }
            e2e = (tot_correct / tot_n) if tot_n else None
            cellkey = f"{task}_{length}_{rtok}"
            summary["babilong"][cellkey] = {
                "n": tot_n,
                "n_expected": 100,
                "complete": tot_n == 100,
                "end_to_end": _round(e2e),
                "n_correct": tot_correct,
                "recompute_ratio": r,
                "task": task,
                "length": length,
                "avg_prefill_latency_ms_mean": _mean(lat),
                "metric": "babilong compare_answers (correct fraction)",
            }

    # ---------------- LoCoMo ----------------
    for d in sorted(cb_dir.glob("cb_locomo_*")):
        rtok = d.name.split("_")[-1]
        r = R_MAP.get(rtok)
        res = qlo.run_scoring(str(d), use_llm_judge=False)
        if not res:
            continue
        # read a representative eval_config for provenance
        cfgs = sorted(d.glob("eval_config_shard*of8.json"))
        if cfgs:
            c = json.load(open(cfgs[0]))
            provenance.setdefault("locomo", {})[rtok] = {
                k: c.get(k) for k in
                ("resume_j", "selector", "topk", "iter_hop_topk", "iter_rounds",
                 "chunk_size", "lora_adapter", "baseline", "recompute_ratio",
                 "seed")}
        cellkey = f"locomo_all_{rtok}"
        summary["locomo"][cellkey] = {
            "n": res["n_samples"],
            "n_expected": 1986,
            "complete": res["n_samples"] == 1986,
            "end_to_end": _round(res["overall_acc"] / 100.0),  # substr acc
            "substr_acc": _round(res["overall_acc"] / 100.0),
            "f1_mean": _round(res["overall_f1"] / 100.0),
            "em_mean": _round(res["overall_em"] / 100.0),
            "recall_at_k": None,  # not stored per-item for cacheblend LoCoMo
            "recompute_ratio": r,
            "by_category": res.get("by_category", {}),
            "metric": "substr acc / F1 / EM (score_sample; LLM-judge deferred to MAIN)",
        }

    # ---------------- completeness guard ----------------
    missing = []
    incomplete = []
    for fam in ("ruler", "babilong", "locomo"):
        for ck, e in summary[fam].items():
            if not e.get("complete", True):
                incomplete.append(f"{fam}:{ck}:n={e['n']}/{e['n_expected']}")
    for rtok in R_TOKENS:
        for t in RULER_TASKS:
            for ln in RULER_LENGTHS:
                ck = f"{t}_{ln}_{rtok}"
                e = summary["ruler"].get(ck)
                if e is None:
                    missing.append(f"ruler:{ck}:ABSENT")
                elif e["n"] != 100:
                    missing.append(f"ruler:{ck}:n={e['n']}")
        for ln in BABI_LENGTHS:
            ck = f"qa5_{ln}_{rtok}"
            e = summary["babilong"].get(ck)
            if e is None:
                missing.append(f"babilong:{ck}:ABSENT")
            elif e["n"] != 100:
                missing.append(f"babilong:{ck}:n={e['n']}")
    for rtok in LOCOMO_R:
        ck = f"locomo_all_{rtok}"
        e = summary["locomo"].get(ck)
        if e is None:
            missing.append(f"locomo:{ck}:ABSENT")
        elif e["n"] != 1986:
            missing.append(f"locomo:{ck}:n={e['n']}")

    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output_dir": str(cb_dir).replace(ROOT + "/", ""),
        "arm": "cacheblend_chunk_kv (#143)",
        "all_tasks_reported": not missing,
        "missing_required_cells": missing,
        "incomplete_cells": incomplete,
        "recompute_ratios": R_MAP,
        "provenance": provenance,
        "summary": summary,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cacheblend_dir", default="bench_results/cacheblend")
    args = ap.parse_args()
    cb = Path(args.cacheblend_dir)
    if not cb.is_absolute():
        cb = Path(ROOT) / cb
    out = aggregate(cb)
    aggfile = cb / "aggregate.json"
    with open(aggfile, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n[#143] aggregate -> {aggfile}")
    print(f"[#143] all_tasks_reported={out['all_tasks_reported']} "
          f"missing={len(out['missing_required_cells'])}")


if __name__ == "__main__":
    main()
