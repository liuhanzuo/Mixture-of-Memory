#!/usr/bin/env python
"""A02 Job 2.2 — is there a HARDER RULER cell that is still RETRIEVAL-CLOSED?

Zero GPU: CPU sample generation + BM25 selection only, no model loaded. This is the
pre-GPU screen the PREREG (3) requires before any eval is dispatched.

THE PROBLEM. A0-A3 sit at 95-100 % on the four primary cells, so "read tax ~ 0 at
shallow j" is partly a statement about a SATURATED benchmark. A de-saturating cell must
be harder WHILE STAYING retrieval-closed (recall@12 >= 95 %); otherwise it re-introduces
the exact confound the primary read-out exists to exclude.

WHY LENGTH IS THE WRONG KNOB (pre-registered, not decided after the fact): dvr measured
recall DEGRADING with length (qa2 49.5 % -> 22.9 % from 16k -> 32k). Going to 64k/128k
de-saturates by BREAKING RETRIEVAL, which is disqualifying. So this probe holds length
at 16k/32k and varies TASK DIFFICULTY.

CANDIDATES (all already implemented in scripts/eval_ruler_mem_space.py):
  niah_single_3   essay haystack, 36-char UUID value  (harder value copying)
  niah_multivalue 1 key, 4 values, retrieve ALL 4     (multi-value recall)
  niah_multiquery 4 keys ALL queried, retrieve all 4  (4 needles, scattered)

WHAT IS SCREENED. For each candidate: recall@12 under the SAME selector as the primary
read-out, using the same strict all-in-pack rule. A candidate PASSES the screen iff
recall@12 >= 95 %. Difficulty itself needs a model, so the screen answers only
"would retrieval still be closed?" -- the cheap half, and the half that can disqualify.

Usage: python probe_a02_desaturation_candidates.py [--n 40] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

W = Path(os.environ.get(
    "A02_W", "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"))
sys.path.insert(0, str(W))

import torch  # noqa: E402
import zlib  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import scripts.eval_qcmem_babilong as qcb  # noqa: E402
import scripts.eval_ruler_mem_space as ruler  # noqa: E402

_iter_bm25_indices = qcb._iter_bm25_indices
_locate = qcb.harness._locate_needle_chunks if hasattr(qcb, "harness") else None

SEED, TOPK, HOP, CHUNK = 42, 12, 4, 512
CANDIDATES = ["niah_single_3", "niah_multivalue", "niah_multiquery"]
INCUMBENTS = ["niah_multikey_1"]


def wilson(k, n):
    if n == 0:
        return None, None, None
    p = k / n
    z = 1.959963984540054
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * p, 2), round(100 * max(0, c - h), 2), round(100 * min(1, c + h), 2)


def screen(task, length, tok, n):
    tt = ruler._LENGTH_TOKENS[length]
    base_seed = SEED + (zlib.crc32(f"{task}\x00{length}".encode()) % 100000)
    per = []
    for i in range(n):
        rng = random.Random(base_seed * 1000 + i)
        prompt, answers, gold = ruler._build_sample(task, tt, tok, rng, None)
        ids = tok.encode(prompt, add_special_tokens=True)
        toks = torch.tensor(ids, dtype=torch.long)
        ctx = list(toks.split(CHUNK))[:-1]
        bq = prompt[prompt.rfind("\n") + 1:].strip()
        bq_ids = tok.encode(bq, add_special_tokens=False)
        sel = set(_iter_bm25_indices(ctx, list(bq_ids), topk=TOPK, iter_rounds=0,
                                    iter_hop_topk=HOP))
        # gold: locate EVERY needle sentence the task requires (multivalue/multiquery
        # have several). _build_sample returns the queried needle(s) as a string or list.
        needles = gold if isinstance(gold, (list, tuple)) else ([gold] if gold else [])
        dec = [tok.decode(c) for c in ctx]
        starts, acc = [], 0
        for t in dec:
            starts.append(acc)
            acc += len(t)
        ends = [starts[k] + len(dec[k]) for k in range(len(dec))]
        full = "".join(dec)
        gchunks, unloc = [], 0
        for nd in needles:
            at = full.find(str(nd).strip())
            if at < 0:
                unloc += 1
                continue
            hi = at + len(str(nd).strip())
            gchunks += [k for k in range(len(dec))
                        if not (ends[k] <= at or starts[k] >= hi)]
        g = sorted(set(gchunks))
        per.append({"i": i, "n_needles": len(needles), "n_unlocated": unloc,
                    "gold_chunks": g, "n_ctx": len(ctx),
                    "hit": bool(g) and unloc == 0 and all(c in sel for c in g)})
    loc = [p for p in per if p["gold_chunks"] and p["n_unlocated"] == 0]
    hits = [p for p in loc if p["hit"]]
    p_, lo, hi = wilson(len(hits), len(loc))
    return {"n": len(per), "n_locatable": len(loc), "recall_at_12": p_,
            "ci95": [lo, hi], "mean_n_needles": round(
                sum(p["n_needles"] for p in per) / len(per), 2),
            "mean_n_gold_chunks": round(sum(len(p["gold_chunks"]) for p in per) / len(per), 2),
            "mean_n_ctx": round(sum(p["n_ctx"] for p in per) / len(per), 1),
            "retrieval_closed": (p_ is not None and p_ >= 95.0)}


def main(out_dir: Path, n: int):
    tok = AutoTokenizer.from_pretrained(str(W / "../models/Qwen--Qwen3-8b"),
                                        trust_remote_code=True)
    res = {}
    for task in INCUMBENTS + CANDIDATES:
        for length in ("16k", "32k"):
            cell = f"ruler|{task}|{length}"
            try:
                res[cell] = screen(task, length, tok, n)
            except Exception as ex:
                res[cell] = {"status": f"ERROR {type(ex).__name__}: {ex}"}
            r = res[cell]
            if r.get("status"):
                print(f"{cell:36s} {r['status']}")
            else:
                print(f"{cell:36s} recall@12={r['recall_at_12']}% CI{r['ci95']} "
                      f"locatable={r['n_locatable']}/{r['n']} "
                      f"needles={r['mean_n_needles']} goldchunks={r['mean_n_gold_chunks']} "
                      f"closed={r['retrieval_closed']}")

    passed = [c for c, r in res.items()
              if not r.get("status") and r["retrieval_closed"]
              and not c.startswith("ruler|niah_multikey_1")]
    print(f"\ncandidates passing the retrieval-closed screen: {len(passed)}")
    for c in passed:
        print("  ", c)

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "a02_desaturation_screen.json"
    json.dump({
        "generated_by": "probe_a02_desaturation_candidates.py",
        "prereg": "A02_BABILONG_MISORDER_PREREG.md 3 (Job 2.2)",
        "gpu_spent": "ZERO (CPU generation + BM25 only)",
        "purpose": ("pre-GPU screen: which harder RULER task would STILL be "
                    "retrieval-closed at 16k/32k under the primary selector"),
        "length_knob_rejected": ("dvr measured recall DEGRADING with length "
                                 "(qa2 49.5%->22.9%, 16k->32k), so 64k/128k de-saturates "
                                 "by breaking retrieval -- disqualifying"),
        "selector": "iter_bm25 topk=12 iter_hop_topk=4 chunk_size=512",
        "n_per_cell": n, "criterion": "recall@12 >= 95%",
        "cells": res, "passing_candidates": passed,
    }, open(dst, "w"), indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=str(
        W / "proposal/backlog/A02-comem-write-read-repair/evidence/babilong_misorder"))
    a = ap.parse_args()
    main(Path(a.out), a.n)
