#!/usr/bin/env python
"""P0.19 retrieval-recall vs in-pack-readout decomposition (Paper A).

NEW FILE (2026-08-03). Pure-CPU, ZERO GPU, ZERO training. Does NOT modify any
shared eval module — it only *imports* the unmodified flagship selection / sample
primitives:

  * ``scripts/eval_qcmem_babilong._iter_bm25_indices``   (flagship iter_bm25 pack)
  * ``scripts/eval_qcmem_babilong._select_context_chunk_indices`` (plain bm25 pack)
  * ``scripts/eval_qcmem_babilong.harness._locate_needle_chunks`` (gold-span -> chunk)
  * ``scripts/eval_ruler_mem_space._build_sample`` / ``_make_vt_icl`` (RULER samples)
  * ``scripts/eval_longeval_mem_space.build_lines_prompt`` (LongEval samples)

Thesis (P0.19): the CoMem (``j=12``) vs ``j=0`` (RAG-recompute upper bound) gap
splits into (a) SELECTOR MISS — the flagship iter_bm25 selector did not put the
gold support span into the top-``k`` pack — and (b) CACHED-STATE READOUT FAILURE
— the gold span WAS in the pack but the mid-depth resume could not read it out.

  ** recall is decided INDEPENDENTLY of answer accuracy **: a sample is a
  retrieval HIT iff the gold support span's document-absolute chunk index lands
  in the selected pack. It is NEVER inferred from whether the model answered.

Given a per-sample HIT/MISS label + the already-run j=0 and j=12 per-sample
correctness, we report, per (task,length):

  1. recall@k                              (fraction of samples whose gold in pack)
  2. j=0  accuracy on the recall-HIT subset
  3. j=12 accuracy on the SAME recall-HIT subset       (readout on retrieved)
  4. j=0 / j=12 accuracy on the recall-MISS subset
  + paired 95% CIs (Wilson for a proportion; paired-bootstrap for the j12-j0 gap)
  + raw sample IDs of every subset.

Reproducibility notes (see paperA/P0_19_decomp_NOTES.md for the full audit):
  * LongEval  : samples derive from ``zlib.crc32``-stable seeds -> the existing
    j=0 / j=12(frozen) / j=12+LoRA runs share ONE sample set and are trivially
    paired; per-sample records already store ``sample_index`` + ``correct``.
    -> FULLY CPU-decomposable here. ``--task longeval``.
  * BABILong  : samples come from the fixed HF Arrow cache in dataset row order;
    the eval walks ``range(n)[shard::nshards]`` so CSV row r of shard s == dataset
    index ``s + r*nshards``. Gold SUPPORT (qa1: the last "X moved to <loc>" fact)
    is NOT annotated in the dataset — only the final ``target`` location is. For
    qa1/qa3 the gold-answer *string* IS a faithful support-span locator (the
    answer location token appears in exactly the supporting fact chunk(s)); qa2
    (two-fact) / qa5 (three-arg) need a support-fact locator, flagged below.
    -> qa1 CPU-decomposable (answer-string == support span). ``--task babilong``.
  * RULER     : ``eval_ruler_qcmem`` originally seeded ``base_seed = args.seed +
    hash((task, length)) % 100000`` with Python's ``hash()``. That is per-process
    salted unless ``PYTHONHASHSEED`` is pinned; the EXISTING on-disk runs recorded
    ``pythonhashseed: null`` (NOT pinned) so ``base_seed`` — hence the whole
    sample set — differs run-to-run: those existing j=0 and j=12 RULER dirs are
    NOT paired (verified: same-slot needles differ) and cannot be joined. The
    seed is now ``args.seed + zlib.crc32(f"{task}\x00{length}") % 100000``
    (PYTHONHASHSEED-independent), so a FRESH paired re-run with the fixed code is
    automatically paired across shards/arms without needing PYTHONHASHSEED. This
    script emits the recall side for a *single* run's samples (``--task ruler``);
    once a fresh paired re-run exists it is a pure-CPU join.

Usage (LongEval, all paired arms already on disk):
    python scripts/analyze_p019_recall_readout.py --task longeval \
        --model_path models/Qwen3-8b-local \
        --j0_dir  longeval_results/p0_2_c2_j0_iterbm25_chatFALSE/longeval_8b \
        --j12_dir longeval_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE \
        --flagship_dir longeval_results/qcmem_8b_iter_chatFALSE/longeval_8b \
        --lengths 8k 16k 32k 64k 128k --topk 12 --chunk_size 512 \
        --out paperA/p0_19_longeval_decomp.json

    python scripts/analyze_p019_recall_readout.py --task babilong \
        --model_path models/Qwen3-8b-local \
        --j0_dir  babilong_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE \
        --j12_dir babilong_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE \
        --babilong_tasks qa1 --lengths 4k 8k 16k 32k --topk 12 --chunk_size 512 \
        --out paperA/p0_19_babilong_decomp.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
import zlib
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402  (only tensor.split / tensor build; no GPU)

# flagship, unmodified selection + gold-locator primitives
import scripts.eval_qcmem_babilong as qcb  # noqa: E402

_iter_bm25_indices = qcb._iter_bm25_indices
_select_context_chunk_indices = qcb._select_context_chunk_indices
_locate_needle_chunks = qcb.harness._locate_needle_chunks


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (round(100 * p, 2), round(100 * max(0.0, centre - half), 2),
            round(100 * min(1.0, centre + half), 2))


def paired_bootstrap_gap(pairs, iters: int = 10000, seed: int = 0):
    """Paired bootstrap 95% CI of mean(j12 - j0) over the SAME samples.

    ``pairs`` = list of (j0_correct:int/bool, j12_correct:int/bool)."""
    if not pairs:
        return None
    rng = random.Random(seed)
    n = len(pairs)
    a = [int(x[0]) for x in pairs]
    b = [int(x[1]) for x in pairs]
    obs = (sum(b) - sum(a)) / n
    diffs = []
    for _ in range(iters):
        sa = sb = 0
        for _ in range(n):
            j = rng.randrange(n)
            sa += a[j]
            sb += b[j]
        diffs.append((sb - sa) / n)
    diffs.sort()
    lo = diffs[int(0.025 * iters)]
    hi = diffs[int(0.975 * iters) - 1]
    return {"gap_pp": round(100 * obs, 2),
            "ci95": [round(100 * lo, 2), round(100 * hi, 2)], "n": n}


# --------------------------------------------------------------------------- #
# generic recall bookkeeping
# --------------------------------------------------------------------------- #
def recall_hit(gold_chunks, sel_set) -> bool:
    """A HIT iff EVERY gold-support chunk is in the pack (single-support tasks
    have |gold|==1 so this reduces to 'the gold chunk was selected'). For
    multi-support tasks this is the strict all-in-pack criterion; we also report
    the partial fraction separately."""
    if not gold_chunks:
        return False
    return all(g in sel_set for g in gold_chunks)


def recall_frac(gold_chunks, sel_set) -> float:
    if not gold_chunks:
        return 0.0
    return len([g for g in gold_chunks if g in sel_set]) / len(gold_chunks)


# --------------------------------------------------------------------------- #
# LongEval
# --------------------------------------------------------------------------- #
def analyze_longeval(args, tokenizer):
    import scripts.eval_longeval_mem_space as le
    import scripts.eval_qcmem_longeval as qle
    build_lines_prompt = le.build_lines_prompt
    LENGTH_TOKENS = qle._LENGTH_TOKENS

    def load_records(base_dir):
        """length -> {sample_index: correct(bool)} merged over shards."""
        out = {}
        for length in args.lengths:
            recs = {}
            files = (glob.glob(os.path.join(base_dir, f"longeval_{length}_shard*of*.json"))
                     or glob.glob(os.path.join(base_dir, f"longeval_{length}.json")))
            for fp in files:
                d = json.load(open(fp))
                for r in d.get("records", []):
                    recs[int(r["sample_index"])] = bool(r["correct"])
            if recs:
                out[length] = recs
        return out

    j0 = load_records(args.j0_dir) if args.j0_dir else {}
    j12 = load_records(args.j12_dir) if args.j12_dir else {}
    flag = load_records(args.flagship_dir) if args.flagship_dir else {}

    results = {}
    for length in args.lengths:
        if length not in LENGTH_TOKENS:
            continue
        target_tokens = LENGTH_TOKENS[length]
        length_seed = args.seed + (zlib.crc32(length.encode()) % 100000)
        # sample set = union of indices actually evaluated in the arms we have.
        idx_sets = [set(d.get(length, {}).keys()) for d in (j0, j12, flag) if d.get(length)]
        if not idx_sets:
            continue
        sample_ids = sorted(set.intersection(*idx_sets)) if len(idx_sets) > 1 \
            else sorted(idx_sets[0])

        per_sample = []
        for i in sample_ids:
            rng = random.Random(length_seed * 1000 + i)
            prompt, expected, target_label, n_lines = build_lines_prompt(
                target_tokens, tokenizer, rng)
            # NB: chat=False -> model input == raw prompt (add_special_tokens=True).
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            tokens = torch.tensor(ids, dtype=torch.long)
            chunks = list(tokens.split(args.chunk_size))
            context_chunks = chunks[:-1]
            n_ctx = len(context_chunks)
            # flagship LongEval selector == iter_bm25 over the bare label query.
            bare_q_ids = tokenizer.encode(qle._bm25_query(target_label),
                                          add_special_tokens=False)
            sel = _iter_bm25_indices(context_chunks, list(bare_q_ids),
                                     topk=args.topk, iter_rounds=args.iter_rounds,
                                     iter_hop_topk=args.iter_hop_topk)
            sel_set = set(sel)
            # gold support chunk(s): the single line `line {label}: ... <value>`.
            gold = qle._oracle_needle_chunks(tokens.unsqueeze(0), expected,
                                             target_label, tokenizer, args.chunk_size)
            gold = sorted(gold) if gold else []
            hit = recall_hit(gold, sel_set)
            per_sample.append({
                "sample_index": i, "gold_chunks": gold, "n_ctx": n_ctx,
                "selected": sel, "recall_hit": hit,
                "recall_frac": round(recall_frac(gold, sel_set), 3),
                "j0": j0.get(length, {}).get(i),
                "j12": j12.get(length, {}).get(i),
                "flag": flag.get(length, {}).get(i),
            })
        results[length] = summarize(per_sample, "longeval", length)
    return results


# --------------------------------------------------------------------------- #
# BABILong
# --------------------------------------------------------------------------- #
def _babilong_arrow_path(length, task):
    base = os.path.join(PROJECT_ROOT, ".hf_cache", "datasets", "RMT-team___babilong")
    hits = glob.glob(os.path.join(base, length, "*", "*", f"babilong-{task}.arrow"))
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
        t = r.read_all()
    return t.to_pylist()


def _babilong_csv_map(cell_dir, task, length):
    """Return {dataset_index: correct(bool)} by replaying shard striding.

    CSV row r of shard s (num_shards=N) == dataset index (s + r*N), because the
    eval walks ``sample_indices = range(n)[shard::N]`` in order. Correctness is
    recomputed with the SAME babilong.metrics.compare_answers the eval uses so we
    do not depend on a persisted per-row score column."""
    from babilong.metrics import TASK_LABELS, compare_answers
    files = sorted(glob.glob(os.path.join(
        cell_dir, f"{task}_{length}_*shard*of*.csv")))
    single = glob.glob(os.path.join(cell_dir, f"{task}_{length}_*.csv"))
    if not files and single:
        files = [f for f in single if "shard" not in f]
    out = {}
    for fp in files:
        base = os.path.basename(fp)
        nshard = shard = None
        if "shard" in base:
            tag = base.split("shard")[1].split(".")[0]  # e.g. "2of4"
            shard, nshard = (int(x) for x in tag.split("of"))
        else:
            shard, nshard = 0, 1
        with open(fp) as f:
            rows = list(csv.DictReader(f))
        for r_i, row in enumerate(rows):
            ds_idx = shard + r_i * nshard
            ok = bool(compare_answers(row["target"], row["output"],
                                      row["question"], TASK_LABELS[task]))
            out[ds_idx] = ok
    return out


def analyze_babilong(args, tokenizer):
    from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
    results = {}
    for task in args.babilong_tasks:
        for length in args.lengths:
            arrow = _babilong_arrow_path(length, task)
            if not arrow:
                results[f"{task}|{length}"] = {"error": "arrow_not_found"}
                continue
            data = _read_arrow(arrow)
            prompt_cfg = {
                "instruction": DEFAULT_PROMPTS[task]["instruction"],
                "examples": DEFAULT_PROMPTS[task]["examples"],
                "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"],
                "template": DEFAULT_TEMPLATE,
            }
            j0_cell = _find_cell_dir(args.j0_dir, task, length) if args.j0_dir else None
            j12_cell = _find_cell_dir(args.j12_dir, task, length) if args.j12_dir else None
            j0 = _babilong_csv_map(j0_cell, task, length) if j0_cell else {}
            j12 = _babilong_csv_map(j12_cell, task, length) if j12_cell else {}
            idx_pool = sorted(set(j0) | set(j12)) or list(range(len(data)))

            per_sample = []
            for idx in idx_pool:
                if idx >= len(data):
                    continue
                sample = data[idx]
                target = sample["target"]
                question = sample["question"]
                input_text = get_formatted_input(
                    sample["input"], question, prompt_cfg["examples"],
                    prompt_cfg["instruction"], prompt_cfg["post_prompt"],
                    template=prompt_cfg["template"])
                ids = tokenizer.encode(input_text, add_special_tokens=True)
                tokens = torch.tensor(ids, dtype=torch.long)
                chunks = list(tokens.split(args.chunk_size))
                context_chunks = chunks[:-1]
                n_ctx = len(context_chunks)
                bare_q_ids = tokenizer.encode((question or "").strip(),
                                              add_special_tokens=False)
                sel = _iter_bm25_indices(context_chunks, list(bare_q_ids),
                                         topk=args.topk, iter_rounds=args.iter_rounds,
                                         iter_hop_topk=args.iter_hop_topk)
                sel_set = set(sel)
                # gold-support locator: for qa1/qa3 the answer *location string*
                # marks the supporting fact chunk. This is a SUPPORT-SPAN locator,
                # NOT answer-accuracy: it asks "is the gold-support chunk in the
                # pack", independent of the model output.
                gold = _locate_needle_chunks(tokens.unsqueeze(0), target,
                                             tokenizer, args.chunk_size)
                gold = sorted(gold) if gold else []
                hit = recall_hit(gold, sel_set)
                per_sample.append({
                    "sample_index": idx, "gold_chunks": gold, "n_ctx": n_ctx,
                    "recall_hit": hit,
                    "recall_frac": round(recall_frac(gold, sel_set), 3),
                    "j0": j0.get(idx), "j12": j12.get(idx), "flag": None,
                })
            results[f"{task}|{length}"] = summarize(per_sample, task, length)
    return results


def _find_cell_dir(root, task, length):
    """BABILong nested layout: <root>/<runname>_<length>/ holds the cell CSVs."""
    if not root:
        return None
    hits = glob.glob(os.path.join(root, f"*_{length}"))
    for h in hits:
        if glob.glob(os.path.join(h, f"{task}_{length}_*.csv")):
            return h
    # flat layout fallback
    if glob.glob(os.path.join(root, f"{task}_{length}_*.csv")):
        return root
    return None


# --------------------------------------------------------------------------- #
# RULER (recall side only; needs paired PYTHONHASHSEED-pinned re-run to join)
# --------------------------------------------------------------------------- #
def analyze_ruler(args, tokenizer):
    import scripts.eval_ruler_mem_space as ruler
    results = {}
    for task in args.ruler_tasks:
        for length in args.lengths:
            if length not in ruler._LENGTH_TOKENS:
                results[f"{task}|{length}"] = {"error": "unknown_length"}
                continue
            target_tokens = ruler._LENGTH_TOKENS[length]
            # MUST mirror eval_ruler_qcmem/eval_ruler_mem_space exactly so this
            # regenerates the SAME samples the eval scored. Both now use
            # zlib.crc32 (PYTHONHASHSEED-independent) instead of built-in hash();
            # keep this identical or the recall join would use different needles.
            base_seed = args.seed + (zlib.crc32(f"{task}\x00{length}".encode()) % 100000)
            vt_icl = None
            if task == "variable_tracking":
                vt_icl = ruler._make_vt_icl(random.Random(base_seed + 777), 4)
            sel_task = "iter_bm25" if task == "variable_tracking" else "bm25"
            per_sample = []
            for i in range(args.limit):
                rng = random.Random(base_seed * 1000 + i)
                prompt, answers, gold_needle = ruler._build_sample(
                    task, target_tokens, tokenizer, rng, vt_icl)
                ids = tokenizer.encode(prompt, add_special_tokens=True)
                tokens = torch.tensor(ids, dtype=torch.long)
                chunks = list(tokens.split(args.chunk_size))
                context_chunks = chunks[:-1]
                bare_q = prompt[prompt.rfind("\n") + 1:].strip()
                bare_q_ids = tokenizer.encode(bare_q, add_special_tokens=False)
                if sel_task == "iter_bm25":
                    sel = _iter_bm25_indices(context_chunks, list(bare_q_ids),
                                             topk=args.topk,
                                             iter_rounds=args.iter_rounds,
                                             iter_hop_topk=args.iter_hop_topk)
                else:
                    sel = _select_context_chunk_indices(
                        "bm25", context_chunks, bare_q_ids, args.topk, None)
                sel_set = set(sel)
                # gold-support span: for NIAH the queried needle sentence; for VT
                # the chain sentences (answers are the variable names).
                gold = None
                if gold_needle:
                    gold = _locate_needle_chunks(tokens.unsqueeze(0), gold_needle,
                                                 tokenizer, args.chunk_size)
                gold = sorted(gold) if gold else []
                per_sample.append({
                    "sample_index": i, "gold_chunks": gold,
                    "recall_hit": recall_hit(gold, sel_set),
                    "recall_frac": round(recall_frac(gold, sel_set), 3),
                    "j0": None, "j12": None, "flag": None,
                })
            results[f"{task}|{length}"] = summarize(
                per_sample, task, length,
                note="RULER predictions NOT paired (pythonhashseed unset) — "
                     "recall side only; join needs a PYTHONHASHSEED-pinned "
                     "paired re-run.")
    return results


# --------------------------------------------------------------------------- #
# summarize one (task,length) cell
# --------------------------------------------------------------------------- #
def summarize(per_sample, task, length, note=None):
    n = len(per_sample)
    hits = [s for s in per_sample if s["recall_hit"]]
    misses = [s for s in per_sample if not s["recall_hit"]]

    def acc(subset, arm):
        vals = [int(s[arm]) for s in subset if s.get(arm) is not None]
        if not vals:
            return None
        return {"n": len(vals), **dict(zip(
            ("acc", "lo", "hi"), wilson_ci(sum(vals), len(vals))))}

    def paired(subset):
        pairs = [(s["j0"], s["j12"]) for s in subset
                 if s.get("j0") is not None and s.get("j12") is not None]
        return paired_bootstrap_gap(pairs)

    out = {
        "task": task, "length": length, "n": n,
        "recall_at_k": round(100 * len(hits) / n, 2) if n else None,
        "recall_at_k_ci": wilson_ci(len(hits), n)[1:] if n else None,
        "mean_recall_frac": round(sum(s["recall_frac"] for s in per_sample) / n, 3)
        if n else None,
        "n_gold_locatable": sum(1 for s in per_sample if s["gold_chunks"]),
        "hit": {
            "n": len(hits),
            "j0_acc": acc(hits, "j0"), "j12_acc": acc(hits, "j12"),
            "flag_acc": acc(hits, "flag"),
            "paired_j12_minus_j0": paired(hits),
        },
        "miss": {
            "n": len(misses),
            "j0_acc": acc(misses, "j0"), "j12_acc": acc(misses, "j12"),
            "flag_acc": acc(misses, "flag"),
            "paired_j12_minus_j0": paired(misses),
        },
        "raw_ids": {
            "hit": [s["sample_index"] for s in hits],
            "miss": [s["sample_index"] for s in misses],
        },
        "per_sample": per_sample,
    }
    if note:
        out["note"] = note
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="P0.19 recall vs readout decomposition (CPU)")
    ap.add_argument("--task", choices=["longeval", "babilong", "ruler"], required=True)
    ap.add_argument("--model_path", required=True, help="tokenizer source (local)")
    ap.add_argument("--j0_dir", default="")
    ap.add_argument("--j12_dir", default="")
    ap.add_argument("--flagship_dir", default="")
    ap.add_argument("--lengths", nargs="+", default=["8k", "16k", "32k"])
    ap.add_argument("--babilong_tasks", nargs="+", default=["qa1"])
    ap.add_argument("--ruler_tasks", nargs="+",
                    default=["niah_multikey_1", "variable_tracking"])
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--iter_rounds", type=int, default=0)
    ap.add_argument("--iter_hop_topk", type=int, default=4)
    ap.add_argument("--limit", type=int, default=100, help="RULER samples per cell")
    ap.add_argument("--seed", type=int, default=None,
                    help="LongEval/RULER base seed (default: 1234 longeval, 42 ruler)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.seed is None:
        args.seed = 1234 if args.task == "longeval" else 42

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,
                                              local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.task == "longeval":
        res = analyze_longeval(args, tokenizer)
    elif args.task == "babilong":
        res = analyze_babilong(args, tokenizer)
    else:
        res = analyze_ruler(args, tokenizer)

    payload = {"task": args.task, "topk": args.topk, "chunk_size": args.chunk_size,
               "seed": args.seed, "selector": "iter_bm25(flagship)",
               "cells": res}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    # human-readable table
    print(f"\n=== P0.19 {args.task} decomposition (topk={args.topk}, "
          f"selector=iter_bm25) ===")
    hdr = (f"{'cell':<28}{'n':>4}{'rec@k':>8}{'HIT n':>7}"
           f"{'j0|HIT':>9}{'j12|HIT':>9}{'flag|HIT':>10}"
           f"{'MISS n':>8}{'j0|MISS':>9}{'j12|MISS':>9}")
    print(hdr)
    for cell, c in res.items():
        if "error" in c:
            print(f"{cell:<28} ERROR {c['error']}")
            continue
        def g(d, arm):
            v = d.get(arm)
            return f"{v['acc']:.1f}" if v else "-"
        print(f"{cell:<28}{c['n']:>4}{str(c['recall_at_k']):>8}"
              f"{c['hit']['n']:>7}{g(c['hit'],'j0_acc'):>9}"
              f"{g(c['hit'],'j12_acc'):>9}{g(c['hit'],'flag_acc'):>10}"
              f"{c['miss']['n']:>8}{g(c['miss'],'j0_acc'):>9}"
              f"{g(c['miss'],'j12_acc'):>9}")
    print(f"\n[out] -> {args.out}")


if __name__ == "__main__":
    main()
