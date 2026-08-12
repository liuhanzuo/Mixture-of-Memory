#!/usr/bin/env python
"""A02 DEPTH-vs-RETRIEVAL analyzer.  Pure CPU for the recall side; no GPU.

WHAT THIS SETTLES
-----------------
Phase-1's C1-vs-C2 comparison moved FOUR variables at once (verified from the
on-disk eval configs):
    C1: no_retrieval=True, selector=None, topk=None, resume_j=0,  lora=None
    C2: no_retrieval=False, selector=iter_bm25, topk=12, resume_j=12, lora=<flagship>
so its quality losses could be MID-LAYER-READ failures (A02's thesis) or merely
RETRIEVAL-RECALL failures (a property of top-12 iter_bm25, unrelated to CoMem).

The 4-arm chain, each adjacent pair differing in exactly ONE variable:
    c1_pack_all -> j0_top12   : RETRIEVAL   (depth j=0 and no-LoRA held fixed)
    j0_top12    -> j12_frozen : READ DEPTH  (retrieval identical, no-LoRA both)
    j12_frozen  -> c2_comem   : READ-LoRA   (depth + retrieval identical)
The middle two arms are new; c1_pack_all / c2_comem are phase-1's on-disk runs.

This ALSO removes the depth<->LoRA confound the cost gate called irreducible:
by running j=12 WITHOUT the LoRA we attribute depth and adapter separately.

CANONICAL SCORERS ARE IMPORTED, NEVER REIMPLEMENTED. Twice in this project a
subagent reimplemented a metric and produced a significant result where the
canonical one produced a tie. So:
  * BABILong : babilong.metrics.{TASK_LABELS, compare_answers}
  * RULER    : the per-item `correct` the harness already wrote to
               <task>_<length>_shard*of*.records.json
  * selection: scripts.eval_qcmem_babilong.{_iter_bm25_indices,
               _select_context_chunk_indices}   (the flagship selectors)
  * gold span: scripts.eval_qcmem_babilong.harness._locate_needle_chunks
  * CI       : paired-difference bootstrap, n_boot=5000 seed=42, CI95 percentile
               (matches the A03 protocol so numbers are comparable)

RETRIEVAL RECALL is decided INDEPENDENTLY of answer accuracy: a sample is a HIT
iff every gold-support chunk index lands in the top-12 pack. It is never
inferred from whether the model answered correctly.

FAIL-CLOSED GATES
  G1 shard completeness: every cell must have exactly --num_shards shards in
     EVERY arm, else that cell is refused (no silent 5-of-8 merge).
  G2 pairing: BABILong pairs by dataset index reconstructed from shard striding
     (row r of shard s == index s + r*N) and asserts equal (question,target)
     across arms. RULER asserts input_ids_sha256 equality across all arms.
  G3 config identity: every retrieving arm must record selector=iter_bm25,
     topk=12, chunk_size=512, chat_template=False; j0/j12 arms must record
     lora_adapter=None; the comem arm must record the flagship LoRA.
Negative-tested: --selftest_gate deletes a shard from a scratch copy and
confirms G1 fires.

Usage:
  python analyze_a02_depth_vs_retrieval.py --out <evidence_dir>
  python analyze_a02_depth_vs_retrieval.py --selftest_gate
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get(
    "A02_BASE", "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"))
PROJECT_ROOT = str(BASE)
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402  (tensor.split only; no GPU)

import scripts.eval_qcmem_babilong as qcb  # noqa: E402  flagship, unmodified

_iter_bm25_indices = qcb._iter_bm25_indices
_locate_needle_chunks = qcb.harness._locate_needle_chunks

N_BOOT = 5000
SEED = 42
NSHARD = 8
TOPK = 12
HOP = 4
CHUNK = 512

# arm -> (results_subdir, kind)   kind: pack_all | j0 | j12_frozen | comem
BAB_ARMS = {
    "c1_pack_all": "a02_babilong_c1_kvdirect",          # phase-1 C1
    "j0_top12":    "a02_dvr_babilong_j0_top12",         # NEW
    "j12_frozen":  "a02_dvr_babilong_j12_frozen",       # NEW
    "c2_comem":    "a02_babilong_c2_j12_readlora",      # phase-1 C2
}
RUL_ARMS = {
    "c1_pack_all": "a02_ruler_c1_kvdirect",
    "j0_top12":    "a02_dvr_ruler_j0_top12",
    "j12_frozen":  "a02_dvr_ruler_j12_frozen",
    "c2_comem":    "a02_ruler_c2_j12_readlora",
}
ARM_ORDER = ["c1_pack_all", "j0_top12", "j12_frozen", "c2_comem"]
# the single-variable steps
STEPS = [("c1_pack_all", "j0_top12",  "retrieval"),
         ("j0_top12",    "j12_frozen", "read_depth"),
         ("j12_frozen",  "c2_comem",  "read_lora"),
         # DEPLOYED mid-depth read: retrieval held IDENTICAL, j=0 full-depth vs
         # j=12 + the adapter distilled for it. This is the like-for-like number
         # that answers "is C2's quality loss depth or retrieval", because it is
         # the configuration CoMem actually ships, not the unadapted j12_frozen
         # lower bound. It equals read_depth + read_lora.
         ("j0_top12",    "c2_comem",  "read_deployed")]

FLAGSHIP_LORA = "outputs/qcmem_distill_qwen_j12_r32_4k/final"


# ---------------------------------------------------------------- stats ---- #
def bootstrap_diff_ci(a, b, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap CI for mean(b) - mean(a) on per-item paired records."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    assert len(a) == len(b), f"unpaired: {len(a)} vs {len(b)}"
    d = b - a
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), n


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (None, None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(100 * p, 2), round(100 * max(0.0, c - h), 2),
            round(100 * min(1.0, c + h), 2))


def sig(lo, hi):
    return "SIG" if (lo > 0 or hi < 0) else "ns"


# ------------------------------------------------------------- BABILong ---- #
def bab_cell_items(arm_dir: Path, task: str, length: str, nshard=NSHARD):
    """Return {dataset_index: (correct, question, target)} for one cell.

    G1: refuses unless exactly `nshard` shard CSVs are present.
    Pairing: CSV row r of shard s == dataset index s + r*nshard, because the
    eval walks sample_indices = range(n)[shard::nshard] in order.
    Correctness recomputed with the canonical babilong scorer.
    """
    from babilong.metrics import TASK_LABELS, compare_answers
    files = sorted(glob.glob(str(arm_dir / f"{task}_{length}_*shard*of{nshard}.csv")))
    if len(files) != nshard:
        return None, f"G1_SHARD_INCOMPLETE {len(files)}/{nshard}"
    labels = TASK_LABELS[task]
    out = {}
    for fp in files:
        tag = os.path.basename(fp).split("shard")[1].split(".")[0]  # "3of8"
        s, n = (int(x) for x in tag.split("of"))
        if n != nshard:
            return None, f"G1_SHARD_WIDTH {n}!={nshard}"
        with open(fp, newline="") as fh:
            for r, row in enumerate(csv.DictReader(fh)):
                idx = s + r * nshard
                tgt = (row.get("target") or "").strip()
                out_s = (row.get("output") or "").strip()
                q = (row.get("question") or "").strip()
                out[idx] = (1 if compare_answers(tgt, out_s, q, labels) else 0, q, tgt)
    return out, None


def bab_cell_cfg(arm_dir: Path, task: str, length: str):
    fs = sorted(glob.glob(str(arm_dir / f"{task}_{length}_*shard0of{NSHARD}.json")))
    if not fs:
        return {}
    return json.load(open(fs[0]))


def check_cfg(arm: str, cfg: dict, errs: list, where: str):
    """G3 config identity."""
    if not cfg:
        errs.append(f"{where}/{arm}: no cell config json")
        return
    q = cfg.get("qcmem", cfg)
    ct = cfg.get("chat_template")
    if ct is not False:
        errs.append(f"{where}/{arm}: chat_template={ct!r} (must be False)")
    if q.get("chunk_size") != CHUNK:
        errs.append(f"{where}/{arm}: chunk_size={q.get('chunk_size')}")
    lora = q.get("lora_adapter")
    if arm == "c1_pack_all":
        if cfg.get("no_retrieval") is not True:
            errs.append(f"{where}/{arm}: expected no_retrieval=True")
        if q.get("resume_j") != 0:
            errs.append(f"{where}/{arm}: resume_j={q.get('resume_j')} != 0")
        if lora not in (None, ""):
            errs.append(f"{where}/{arm}: unexpected lora {lora!r}")
        return
    # retrieving arms: retrieval MUST be identical
    if cfg.get("no_retrieval") not in (False, None):
        errs.append(f"{where}/{arm}: expected no_retrieval=False")
    if q.get("selector") != "iter_bm25":
        errs.append(f"{where}/{arm}: selector={q.get('selector')!r} != iter_bm25")
    if q.get("topk") != TOPK:
        errs.append(f"{where}/{arm}: topk={q.get('topk')} != {TOPK}")
    want_j = 0 if arm == "j0_top12" else 12
    if q.get("resume_j") != want_j:
        errs.append(f"{where}/{arm}: resume_j={q.get('resume_j')} != {want_j}")
    if arm == "c2_comem":
        if not lora or FLAGSHIP_LORA not in str(lora):
            errs.append(f"{where}/{arm}: expected flagship LoRA, got {lora!r}")
    else:
        if lora not in (None, ""):
            errs.append(f"{where}/{arm}: expected NO LoRA, got {lora!r}")


def babilong_recall(task, length, idx_pool, tokenizer):
    """Retrieval recall of the top-12 iter_bm25 pack, independent of accuracy.

    Reproduces the exact prompt the eval built, runs the flagship selector, and
    asks whether the gold-support chunk(s) are in the pack.
    Gold locator caveat: for qa1 the answer *location string* faithfully marks
    the supporting fact; qa2/qa5 are multi-fact so the answer-string locator
    marks only the ANSWER-bearing fact, not the full support chain -> recall is
    an UPPER BOUND there. Flagged per cell.
    """
    from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
    arrow = _bab_arrow(length, task)
    if not arrow:
        return None, "arrow_not_found"
    data = _read_arrow(arrow)
    cfg = DEFAULT_PROMPTS[task]
    per = []
    for idx in idx_pool:
        if idx >= len(data):
            continue
        s = data[idx]
        text = get_formatted_input(s["input"], s["question"], cfg["examples"],
                                   cfg["instruction"], cfg["post_prompt"],
                                   template=DEFAULT_TEMPLATE)
        ids = tokenizer.encode(text, add_special_tokens=True)
        toks = torch.tensor(ids, dtype=torch.long)
        chunks = list(toks.split(CHUNK))
        ctx = chunks[:-1]
        bq = tokenizer.encode((s["question"] or "").strip(), add_special_tokens=False)
        sel = set(_iter_bm25_indices(ctx, list(bq), topk=TOPK, iter_rounds=0,
                                    iter_hop_topk=HOP))
        gold = _locate_needle_chunks(toks.unsqueeze(0), s["target"], tokenizer, CHUNK)
        gold = sorted(g for g in (gold or []) if g < len(ctx))
        per.append({"sample_index": idx, "gold_chunks": gold, "n_ctx": len(ctx),
                    "hit": bool(gold) and all(g in sel for g in gold),
                    "n_sel": len(sel)})
    n = len(per)
    loc = [p for p in per if p["gold_chunks"]]
    hits = [p for p in loc if p["hit"]]
    return {
        "n": n, "n_gold_locatable": len(loc),
        "recall_at_12": round(100 * len(hits) / len(loc), 2) if loc else None,
        "recall_at_12_ci": wilson_ci(len(hits), len(loc))[1:] if loc else None,
        "mean_n_ctx": round(sum(p["n_ctx"] for p in per) / n, 1) if n else None,
        "gold_locator_note": ("answer-string locator == full support span (qa1)"
                              if task == "qa1" else
                              "multi-fact task: answer-string locator marks only the "
                              "ANSWER-bearing fact, not the whole support chain -> "
                              "recall is an UPPER BOUND"),
        "per_sample": per,
    }, None


def _bab_arrow(length, task):
    hits = glob.glob(os.path.join(PROJECT_ROOT, ".hf_cache", "datasets",
                                  "RMT-team___babilong", length, "*", "*",
                                  f"babilong-{task}.arrow"))
    return hits[0] if hits else None


def _read_arrow(path):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(path, "r") as src:
        try:
            r = ipc.open_stream(src)
        except Exception:
            src.seek(0)
            r = ipc.open_file(src)
        return r.read_all().to_pylist()


# ---------------------------------------------------------------- RULER ---- #
def rul_cell_items(arm_dir: Path, task: str, length: str, nshard=NSHARD):
    """Return ({sample_index: (correct, sha)}, cfg) using the harness's own
    per-item `correct`. G1 refuses unless exactly nshard record files exist."""
    files = sorted(glob.glob(str(arm_dir / f"{task}_{length}_shard*of{nshard}.records.json")))
    if len(files) != nshard:
        return None, None, f"G1_SHARD_INCOMPLETE {len(files)}/{nshard}"
    out, cfg = {}, None
    for fp in files:
        d = json.load(open(fp))
        cfg = d
        for r in d.get("records", []):
            out[int(r["sample_index"])] = (int(r["correct"]),
                                           r.get("input_ids_sha256"))
    return out, cfg, None


def check_rul_cfg(arm, cfg, errs, where):
    if not cfg:
        errs.append(f"{where}/{arm}: no records cfg")
        return
    if cfg.get("chunk_size") != CHUNK:
        errs.append(f"{where}/{arm}: chunk_size={cfg.get('chunk_size')}")
    lora = cfg.get("lora_adapter")
    if arm == "c1_pack_all":
        if cfg.get("baseline") != "kvdirect":
            errs.append(f"{where}/{arm}: baseline={cfg.get('baseline')!r}")
        if cfg.get("resume_j") != 0:
            errs.append(f"{where}/{arm}: resume_j={cfg.get('resume_j')}")
        if lora:
            errs.append(f"{where}/{arm}: unexpected lora {lora!r}")
        return
    if cfg.get("selector") != "iter_bm25":
        errs.append(f"{where}/{arm}: selector={cfg.get('selector')!r} != iter_bm25")
    if cfg.get("topk") != TOPK:
        errs.append(f"{where}/{arm}: topk={cfg.get('topk')}")
    want_j = 0 if arm == "j0_top12" else 12
    if cfg.get("resume_j") != want_j:
        errs.append(f"{where}/{arm}: resume_j={cfg.get('resume_j')} != {want_j}")
    if arm == "c2_comem":
        if not lora or FLAGSHIP_LORA not in str(lora):
            errs.append(f"{where}/{arm}: expected flagship LoRA, got {lora!r}")
    elif lora:
        errs.append(f"{where}/{arm}: expected NO LoRA, got {lora!r}")


def ruler_recall(task, length, idx_pool, tokenizer, shas):
    """Retrieval recall for RULER + fail-closed sha256 pairing verification.

    Regenerates each sample with the eval's crc32 seed and asserts the
    regenerated prompt's sha equals what every arm recorded.
    """
    import scripts.eval_ruler_mem_space as ruler
    if length not in ruler._LENGTH_TOKENS:
        return None, "unknown_length"
    tt = ruler._LENGTH_TOKENS[length]
    base_seed = SEED + (zlib.crc32(f"{task}\x00{length}".encode()) % 100000)
    vt_icl = ruler._make_vt_icl(random.Random(base_seed + 777), 4) \
        if task == "variable_tracking" else None
    per, sha_fail = [], []
    for i in idx_pool:
        rng = random.Random(base_seed * 1000 + i)
        prompt, answers, gold_needle = ruler._build_sample(task, tt, tokenizer,
                                                          rng, vt_icl)
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        toks = torch.tensor(ids, dtype=torch.long)
        regen = hashlib.sha256(
            ",".join(map(str, toks.tolist())).encode()).hexdigest()
        got = {a: s for a, s in shas.get(i, {}).items() if s}
        if got and len(set(list(got.values()) + [regen])) > 1:
            sha_fail.append({"sample_index": i, "regen": regen, **got})
        chunks = list(toks.split(CHUNK))
        ctx = chunks[:-1]
        bq = prompt[prompt.rfind("\n") + 1:].strip()
        bq_ids = tokenizer.encode(bq, add_special_tokens=False)
        sel = set(_iter_bm25_indices(ctx, list(bq_ids), topk=TOPK, iter_rounds=0,
                                    iter_hop_topk=HOP))
        gold = _locate_needle_chunks(toks.unsqueeze(0), gold_needle, tokenizer,
                                     CHUNK) if gold_needle else None
        gold = sorted(g for g in (gold or []) if g < len(ctx))
        per.append({"sample_index": i, "gold_chunks": gold, "n_ctx": len(ctx),
                    "hit": bool(gold) and all(g in sel for g in gold)})
    loc = [p for p in per if p["gold_chunks"]]
    hits = [p for p in loc if p["hit"]]
    return {
        "n": len(per), "n_gold_locatable": len(loc),
        "recall_at_12": round(100 * len(hits) / len(loc), 2) if loc else None,
        "recall_at_12_ci": wilson_ci(len(hits), len(loc))[1:] if loc else None,
        "mean_n_ctx": round(sum(p["n_ctx"] for p in per) / len(per), 1) if per else None,
        "sha_pairing_failures": sha_fail,
        "gold_locator_note": ("NIAH: gold_needle is the queried needle sentence "
                             "(faithful support span)" if task != "variable_tracking"
                             else "VT: gold_needle is the chain; multi-hop -> "
                                  "all-in-pack is strict"),
        "per_sample": per,
    }, None


# ----------------------------------------------------------------- main ---- #
def analyze(out_dir: Path, nshard=NSHARD):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE.parent / "models" / "Qwen--Qwen3-8b"),
                                        trust_remote_code=True, local_files_only=True)
    errs, cells, refused = [], {}, []

    # ---- BABILong ----
    for task in ("qa1", "qa2", "qa5"):
        for length in ("16k", "32k"):
            key = f"babilong|{task}|{length}"
            items, bad = {}, None
            for arm, sub in BAB_ARMS.items():
                d = BASE / "babilong_results" / sub
                it, e = bab_cell_items(d, task, length, nshard)
                if e:
                    bad = f"{arm}: {e}"
                    break
                items[arm] = it
                check_cfg(arm, bab_cell_cfg(d, task, length), errs, key)
            if bad:
                refused.append({"cell": key, "reason": bad})
                continue
            common = sorted(set.intersection(*[set(v) for v in items.values()]))
            if len(common) != 100:
                errs.append(f"{key}: paired n={len(common)} != 100")
            # G2: verify the pairing key agrees across arms
            for i in common:
                qs = {items[a][i][1] for a in items}
                ts = {items[a][i][2] for a in items}
                if len(qs) > 1 or len(ts) > 1:
                    errs.append(f"{key}: G2 pairing mismatch at index {i}")
                    break
            acc = {a: [items[a][i][0] for i in common] for a in ARM_ORDER}
            rec, rerr = babilong_recall(task, length, common, tok)
            cells[key] = _cell_block(acc, common, rec, rerr)

    # ---- RULER ----
    for task in ("niah_multikey_1", "variable_tracking"):
        for length in ("16k", "32k"):
            key = f"ruler|{task}|{length}"
            items, shas, bad = {}, {}, None
            for arm, sub in RUL_ARMS.items():
                d = BASE / "ruler_results" / sub
                it, cfg, e = rul_cell_items(d, task, length, nshard)
                if e:
                    bad = f"{arm}: {e}"
                    break
                items[arm] = {i: v[0] for i, v in it.items()}
                for i, v in it.items():
                    shas.setdefault(i, {})[arm] = v[1]
                check_rul_cfg(arm, cfg, errs, key)
            if bad:
                refused.append({"cell": key, "reason": bad})
                continue
            common = sorted(set.intersection(*[set(v) for v in items.values()]))
            if len(common) != 100:
                errs.append(f"{key}: paired n={len(common)} != 100")
            acc = {a: [items[a][i] for i in common] for a in ARM_ORDER}
            rec, rerr = ruler_recall(task, length, common, tok, shas)
            if rec and rec["sha_pairing_failures"]:
                errs.append(f"{key}: G2 sha pairing FAILED on "
                            f"{len(rec['sha_pairing_failures'])} samples")
            cells[key] = _cell_block(acc, common, rec, rerr)

    verdict = {
        "gate": "A02 depth-vs-retrieval quality gate",
        "generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "node": ".82 (8x H20, zwfy6), bounded 4-GPU pool (GPUs 0-3)",
        "protocol": {
            "ci": f"paired-difference bootstrap n_boot={N_BOOT} seed={SEED}, "
                  "CI95 percentile; SIG = CI excludes 0",
            "chat_template": False,
            "selector": "iter_bm25 topk=12 iter_hop_topk=4 chunk_size=512 "
                        "-- IDENTICAL across every retrieving arm",
            "scorers": "BABILong babilong.metrics.compare_answers (imported); "
                       "RULER harness per-item `correct` (imported, not recomputed)",
            "recall": "HIT iff every gold-support chunk is in the top-12 pack; "
                      "decided independently of answer accuracy",
            "aggregation": "PER-CELL ONLY. No pooled BABILong / LongEval figure is "
                           "computed anywhere -- they average cells of opposite sign.",
        },
        "arms": {
            "c1_pack_all": "phase-1 C1: j=0, no LoRA, packs ALL chunks, no selector",
            "j0_top12":    "NEW: j=0, no LoRA, top-12 iter_bm25 (matched-pack text-RAG)",
            "j12_frozen":  "NEW: j=12, NO LoRA, SAME top-12 pack (isolates depth)",
            "c2_comem":    "phase-1 C2: j=12 + flagship Read-LoRA, same top-12 pack",
        },
        "single_variable_steps": {
            "retrieval":  "c1_pack_all -> j0_top12",
            "read_depth": "j0_top12 -> j12_frozen",
            "read_lora":  "j12_frozen -> c2_comem",
        },
        "confound_note":
            "The cost gate called depth<->LoRA irreducible because it lacked a "
            "j=12 no-LoRA arm. j12_frozen supplies it, so depth and adapter are "
            "attributed SEPARATELY here. The residual confound kept: j12_frozen "
            "runs the flagship j=12 read WITHOUT the adapter that was distilled "
            "for it, so it is a lower bound on what depth-12 can do with training; "
            "the read_lora step measures exactly that recovery.",
        "config_errors": errs,
        "refused_cells": refused,
        "cells": cells,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "a02_depth_vs_retrieval_ci.json").write_text(
        json.dumps({k: v for k, v in verdict.items() if k != "cells"} |
                   {"cells": {k: {kk: vv for kk, vv in v.items()
                                  if kk != "recall_per_sample"}
                              for k, v in cells.items()}}, indent=1))
    (out_dir / "a02_depth_vs_retrieval_per_item.json").write_text(
        json.dumps({"protocol": verdict["protocol"],
                    "cells": {k: {"paired_index": v["paired_index"],
                                  "per_arm_correct": v["per_arm_correct"],
                                  "recall_per_sample": v.get("recall_per_sample")}
                              for k, v in cells.items()}}, indent=1))
    _print(verdict)
    return verdict


def _cell_block(acc, common, rec, rerr):
    blk = {
        "n_paired": len(common),
        "arm_acc_pct": {a: round(100 * float(np.mean(v)), 2) for a, v in acc.items()},
        "steps": {},
        "total_c1_to_c2": None,
        "paired_index": common,
        "per_arm_correct": {a: [int(x) for x in v] for a, v in acc.items()},
    }
    for a, b, name in STEPS:
        pt, lo, hi, n = bootstrap_diff_ci(acc[a], acc[b])
        blk["steps"][name] = {"from": a, "to": b,
                              "diff_pt": round(100 * pt, 2),
                              "ci": [round(100 * lo, 2), round(100 * hi, 2)],
                              "n": n, "sig": sig(lo, hi)}
    pt, lo, hi, n = bootstrap_diff_ci(acc["c1_pack_all"], acc["c2_comem"])
    blk["total_c1_to_c2"] = {"diff_pt": round(100 * pt, 2),
                             "ci": [round(100 * lo, 2), round(100 * hi, 2)],
                             "n": n, "sig": sig(lo, hi)}
    if rerr:
        blk["recall"] = {"error": rerr}
    elif rec:
        blk["recall"] = {k: v for k, v in rec.items() if k != "per_sample"}
        blk["recall_per_sample"] = rec["per_sample"]
    # Attribution of the c1->c2 change. Uses the TWO-WAY decomposition that
    # partitions it without double counting:
    #   c1 -> j0_top12  (retrieval)  +  j0_top12 -> c2_comem (read_deployed)
    # `read_depth` / `read_lora` further split read_deployed and are reported
    # separately; they are NOT summed in here (that would count the read twice).
    s = blk["steps"]
    mags = {"retrieval": abs(s["retrieval"]["diff_pt"]),
            "read_deployed": abs(s["read_deployed"]["diff_pt"])}
    tot = sum(mags.values()) or 1.0
    blk["attribution_share_pct"] = {k: round(100 * v / tot, 1) for k, v in mags.items()}
    blk["dominant_cause"] = max(mags, key=mags.get)
    # only call it dominant if the two are not within noise of each other
    blk["dominant_is_decisive"] = bool(
        max(mags.values()) >= 2 * min(mags.values())) if min(mags.values()) > 0 else True
    return blk


def _print(v):
    print("\n" + "=" * 78)
    print("A02 DEPTH vs RETRIEVAL — per-cell single-variable decomposition")
    print("=" * 78)
    if v["config_errors"]:
        print("\n!! CONFIG/GATE ERRORS:")
        for e in v["config_errors"]:
            print("   -", e)
    if v["refused_cells"]:
        print("\n!! REFUSED CELLS (incomplete shards):")
        for r in v["refused_cells"]:
            print("   -", r["cell"], r["reason"])
    hdr = (f"{'cell':34s} {'c1':>6s} {'j0t12':>6s} {'j12fz':>6s} {'comem':>6s} "
           f"{'RETR':>9s} {'DEPLOY':>9s} {'depth':>9s} {'lora':>9s} "
           f"{'rec@12':>7s} {'dom':>14s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for k, c in v["cells"].items():
        a = c["arm_acc_pct"]
        s = c["steps"]
        def f(x):
            return f"{s[x]['diff_pt']:+.1f}{'*' if s[x]['sig']=='SIG' else ''}"
        r = c.get("recall", {}).get("recall_at_12")
        print(f"{k:34s} {a['c1_pack_all']:6.1f} {a['j0_top12']:6.1f} "
              f"{a['j12_frozen']:6.1f} {a['c2_comem']:6.1f} "
              f"{f('retrieval'):>9s} {f('read_deployed'):>9s} "
              f"{f('read_depth'):>9s} {f('read_lora'):>9s} "
              f"{(f'{r:.1f}' if r is not None else 'n/a'):>7s} "
              f"{c['dominant_cause']:>14s}")
    print("\n* = CI excludes 0. RETR = c1->j0_top12 (retrieval only).")
    print("DEPLOY = j0_top12->c2_comem (mid-depth read AS SHIPPED, retrieval identical)")
    print("   = the like-for-like depth number; it splits into depth + lora.")
    print("depth = j0_top12->j12_frozen (j=12 WITHOUT its adapter: a lower bound).")
    print("lora  = j12_frozen->c2_comem (recovery from the adapter distilled for j=12).")
    print("RETR + DEPLOY partition the total c1->c2 change without double counting.")


def selftest_gate():
    """Negative-test G1: delete a shard in a scratch copy, confirm refusal."""
    src = BASE / "babilong_results" / BAB_ARMS["j0_top12"]
    if not src.exists():
        print("SELFTEST SKIP: arm dir missing"); return 1
    tmp = Path(tempfile.mkdtemp(prefix="a02_g1_"))
    dst = tmp / "arm"
    shutil.copytree(src, dst)
    items, err = bab_cell_items(dst, "qa1", "16k")
    assert err is None and items is not None, f"baseline should pass, got {err}"
    print(f"SELFTEST baseline: PASS ({len(items)} items, no error)")
    victim = sorted(glob.glob(str(dst / "qa1_16k_*shard3of8.csv")))[0]
    os.remove(victim)
    items2, err2 = bab_cell_items(dst, "qa1", "16k")
    shutil.rmtree(tmp)
    assert items2 is None and err2 and err2.startswith("G1_SHARD_INCOMPLETE"), \
        f"G1 DID NOT FIRE: items={items2 is not None} err={err2}"
    print(f"SELFTEST G1 after deleting 1 shard: FIRED correctly -> {err2}")
    print("SELFTEST PASS: shard-completeness gate is real, not decorative.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        Path(__file__).resolve().parent.parent / "evidence"))
    ap.add_argument("--num_shards", type=int, default=NSHARD)
    ap.add_argument("--selftest_gate", action="store_true")
    a = ap.parse_args()
    if a.selftest_gate:
        sys.exit(selftest_gate())
    analyze(Path(a.out), a.num_shards)
