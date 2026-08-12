#!/usr/bin/env python
"""A02 Job 2.1 — measure `variable_tracking` retrieval recall@12 DIRECTLY.

Zero GPU (CPU regeneration + BM25 selection only; no model is loaded).

WHY THIS IS POSSIBLE AT ALL. The dvr gate recorded VT recall as `n/a` with 0
gold-locatable items, and the read-tax verdict carried that forward as caveat 1:
two of the four primary cells are VT, so their "retrieval-closed" status was
*inherited from an accuracy step, not measured*. The reason was NOT that VT is
intrinsically unlocalisable:

  `eval_ruler_mem_space._make_vt()` RETURNS the chain sentences and variable names,
  but `_build_sample()` DISCARDS them for VT -- it returns `gold_needle=None`
  (its docstring: "NIAH tasks only; None for variable_tracking"). The dvr locator
  therefore received None and correctly reported 0 locatable.

The information exists in the generator and was dropped at an interface. This script
recovers it by regenerating each sample with the SAME RNG call sequence the eval used
and re-deriving the chain, then applying the SAME strict all-in-pack rule the dvr used
for NIAH/BABILong.

PRE-REGISTERED DEFINITION (A02_BABILONG_MISORDER_PREREG.md 2):
  a VT sample is a HIT iff EVERY chain sentence lands in the top-12 pack.
  Secondary (lenient): mean fraction of chain sentences in the pack.

FAIL-CLOSED. Every regenerated prompt's sha256 over input_ids must equal the
`input_ids_sha256` that ALL SEVEN arms recorded. Any mismatch => the measurement is
REFUSED for that cell (no approximate recall is reported). This is the same gate the
dvr used to prove pairing, reused here to prove the regeneration is bit-identical.

DECISION RULE (pre-registered):
  recall >= 95 %  -> VT is retrieval-closed DIRECTLY; caveat 1 discharged.
  recall <  95 %  -> primary statistic must be RESTATED as niah_mk1-only, VT demoted.

Usage: python measure_a02_vt_recall.py [--out <dir>]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import sys
import zlib
from pathlib import Path

W = Path(os.environ.get(
    "A02_W", "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"))
sys.path.insert(0, str(W))

import torch  # noqa: E402  (tensor.split only, no GPU)
from transformers import AutoTokenizer  # noqa: E402

import scripts.eval_qcmem_babilong as qcb  # noqa: E402
import scripts.eval_ruler_mem_space as ruler  # noqa: E402

_iter_bm25_indices = qcb._iter_bm25_indices

SEED, TOPK, HOP, CHUNK, NSHARD = 42, 12, 4, 512, 8
# every arm that ran RULER VT -> results dir (for the sha pairing gate)
RULER_ARMS = {
    "A0": "a02_dvr_ruler_j0_top12",
    "A1": "a02_rtax_ruler_A1_j0control",
    "A2": "a02_rtax_ruler_A2_j6",
    "A3": "a02_rtax_ruler_A3_j9",
    "A4": "a02_ruler_c2_j12_readlora",
    "A5": "a02_rtax_ruler_A5_j18",
    "A6": "a02_rtax_ruler_A6_j12_r40",
}


def wilson(k, n):
    if n == 0:
        return None, None, None
    p = k / n
    z = 1.959963984540054
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * p, 2), round(100 * max(0, c - h), 2), round(100 * min(1, c + h), 2)


def recorded_shas(task, length):
    """{sample_index: {arm: input_ids_sha256}} across every arm that ran this cell."""
    out = {}
    for arm, sub in RULER_ARMS.items():
        for f in glob.glob(str(W / "ruler_results" / sub
                               / f"*{task}*{length}*shard*of{NSHARD}.records.json")):
            d = json.load(open(f))
            recs = d.get("records", d if isinstance(d, list) else [])
            for r in recs:
                if r.get("task") not in (None, task):
                    continue
                if r.get("length") not in (None, length):
                    continue
                i = r.get("sample_index")
                s = r.get("input_ids_sha256")
                if i is None or not s:
                    continue
                out.setdefault(int(i), {})[arm] = s
    return out


def measure(task, length, tok):
    """Regenerate the cell, verify sha pairing, and compute strict chain recall."""
    tt = ruler._LENGTH_TOKENS[length]
    base_seed = SEED + (zlib.crc32(f"{task}\x00{length}".encode()) % 100000)
    vt_icl = ruler._make_vt_icl(random.Random(base_seed + 777), 4)
    shas = recorded_shas(task, length)
    idx_pool = sorted(shas)
    per, sha_fail = [], []

    for i in idx_pool:
        rng = random.Random(base_seed * 1000 + i)
        # _build_sample discards the chain for VT, so replicate its RNG sequence and
        # capture the chain by calling the same generator with the same rng object.
        prompt, answers, _ = ruler._build_sample(task, tt, tok, rng, vt_icl)
        ids = tok.encode(prompt, add_special_tokens=True)
        toks = torch.tensor(ids, dtype=torch.long)
        regen = hashlib.sha256(",".join(map(str, toks.tolist())).encode()).hexdigest()
        got = {a: s for a, s in shas.get(i, {}).items() if s}
        if got and len(set(list(got.values()) + [regen])) > 1:
            sha_fail.append({"sample_index": i, "regen": regen[:16],
                             "recorded": {a: s[:16] for a, s in got.items()}})
            continue

        # The VT answer IS the chain's variable names (see _make_vt_icl:
        # answer = " ".join(vars_all)). Reconstruct each chain sentence verbatim from
        # the answer order: "VAR <v0> = <value>", then "VAR <v{k+1}> = VAR <v{k}> ".
        chain_vars = list(answers)
        # Recover the queried value from the QUESTION line via VT_TEMPLATE's wording.
        # NOTE: a naive prompt.rfind("value ") lands in VT_ANSWER_PREFIX ("...assigned
        # the value {query}, they are: ") and yields "12345," WITH A TRAILING COMMA,
        # which made the first chain sentence unfindable in exactly 1 sentence per
        # sample -> n_fully_locatable = 0 while every located chunk was in the pack.
        import re as _re
        mv = _re.findall(r"assigned the value (\S+?) in the text above", prompt)
        if not mv:
            missing_value = True
            value = None
        else:
            missing_value = False
            value = mv[-1].rstrip(".,;:")   # last = the real query, not the ICL example
        if missing_value:
            per.append({"sample_index": i, "status": "VALUE_UNPARSED"})
            continue
        sentences = [f"VAR {chain_vars[0]} = {value}"]
        for k in range(len(chain_vars) - 1):
            sentences.append(f"VAR {chain_vars[k + 1]} = VAR {chain_vars[k]} ")

        chunks = list(toks.split(CHUNK))
        ctx = chunks[:-1]
        bq = prompt[prompt.rfind("\n") + 1:].strip()
        bq_ids = tok.encode(bq, add_special_tokens=False)
        sel = set(_iter_bm25_indices(ctx, list(bq_ids), topk=TOPK, iter_rounds=0,
                                    iter_hop_topk=HOP))
        # Locate each chain sentence via CHAR-OFFSET mapping, not per-chunk substring.
        # A per-chunk substring test silently misses any sentence that STRADDLES a
        # 512-token chunk boundary (measured: it lost ~22-24 % of chain sentences and
        # made every sample look unlocatable). Build the cumulative decoded length of
        # each chunk, find the sentence in the full decoded context, and map its char
        # span back to EVERY chunk it overlaps. A straddling sentence is only fully
        # readable if ALL its chunks are in the pack, so all of them count as gold.
        dec = [tok.decode(ch) for ch in ctx]
        starts, acc = [], 0
        for t in dec:
            starts.append(acc)
            acc += len(t)
        full = "".join(dec)
        ends = [starts[k] + len(dec[k]) for k in range(len(dec))]

        def chunks_for(span_lo, span_hi):
            return [k for k in range(len(dec))
                    if not (ends[k] <= span_lo or starts[k] >= span_hi)]

        located, missing = [], []
        for s in sentences:
            needle = s.strip()
            at = full.find(needle)
            if at < 0:
                # tolerate whitespace normalisation introduced by decode
                import re as _re
                m = _re.search(_re.escape(needle).replace(r"\ ", r"\s+"), full)
                at = m.start() if m else -1
                nlen = (m.end() - m.start()) if m else 0
            else:
                nlen = len(needle)
            if at < 0:
                missing.append(s)
            else:
                located.extend(chunks_for(at, at + nlen))
        gold = sorted(set(located))
        per.append({
            "sample_index": i, "n_chain": len(sentences),
            "gold_chunks": gold, "n_gold_chunks": len(gold),
            "n_unlocated": len(missing), "n_ctx": len(ctx),
            "hit": bool(gold) and not missing and all(c in sel for c in gold),
            # fraction of the DISTINCT gold chunks that are in the pack (a sentence may
            # straddle a boundary and own >1 chunk, so this is per-chunk not per-sentence)
            "frac_gold_chunks_in_pack": (round(sum(1 for c in gold if c in sel) / len(gold), 4)
                                         if gold else None),
        })
    return per, sha_fail


def main(out_dir: Path):
    tok = AutoTokenizer.from_pretrained(str(W / "../models/Qwen--Qwen3-8b"),
                                        trust_remote_code=True)
    res = {}
    for length in ("16k", "32k"):
        task = "variable_tracking"
        cell = f"ruler|{task}|{length}"
        per, sha_fail = measure(task, length, tok)
        if sha_fail:
            res[cell] = {"status": "REFUSED_SHA_MISMATCH", "n_sha_fail": len(sha_fail),
                         "examples": sha_fail[:3],
                         "note": "regeneration not bit-identical -> no recall reported"}
            print(f"{cell}: REFUSED, {len(sha_fail)} sha mismatches")
            continue
        unparsed = [p for p in per if p.get("status") == "VALUE_UNPARSED"]
        if unparsed:
            res[cell] = {"status": "REFUSED_VALUE_UNPARSED", "n_unparsed": len(unparsed),
                         "note": "query value could not be parsed -> no recall reported"}
            print(f"{cell}: REFUSED, {len(unparsed)} samples with unparsed query value")
            continue
        loc = [p for p in per if p["n_unlocated"] == 0]
        hits = [p for p in loc if p["hit"]]
        p_, lo, hi = wilson(len(hits), len(loc))
        res[cell] = {
            "status": "MEASURED", "n": len(per),
            "n_fully_locatable_chain": len(loc),
            "n_partially_unlocatable": sum(1 for p in per if p["n_unlocated"] > 0),
            "recall_at_12_strict_all_chain_in_pack": p_,
            "recall_at_12_ci95_wilson": [lo, hi],
            "mean_frac_gold_chunks_in_pack": round(
                sum(p["frac_gold_chunks_in_pack"] for p in per
                    if p["frac_gold_chunks_in_pack"] is not None) / max(1, len(per)), 4),
            "mean_n_chain": round(sum(p["n_chain"] for p in per) / len(per), 2),
            "mean_n_gold_chunks": round(
                sum(len(p["gold_chunks"]) for p in per) / len(per), 2),
            "mean_n_ctx": round(sum(p["n_ctx"] for p in per) / len(per), 1),
            "sha_pairing": "PASS (regeneration bit-identical to all arms' records)",
            "per_sample": per,
        }
        print(f"{cell}: recall@12 = {p_}% CI{[lo, hi]} "
              f"(n_fully_locatable={len(loc)}/{len(per)}, "
              f"mean frac gold chunks in pack="
              f"{res[cell]['mean_frac_gold_chunks_in_pack']}, "
              f"mean gold chunks={res[cell]['mean_n_gold_chunks']})")

    verdict = {}
    for cell, r in res.items():
        if r.get("status") != "MEASURED":
            verdict[cell] = "REFUSED"
        elif r["recall_at_12_strict_all_chain_in_pack"] is None:
            # locator could not fully resolve the chain -> the measurement does NOT
            # establish retrieval-closure; fail toward restating, never toward claiming
            verdict[cell] = "UNRESOLVED_LOCATOR_RESTATE_AS_NIAH_ONLY"
        else:
            verdict[cell] = ("RETRIEVAL_CLOSED_DIRECTLY"
                             if r["recall_at_12_strict_all_chain_in_pack"] >= 95.0
                             else "NOT_RETRIEVAL_CLOSED_RESTATE_AS_NIAH_ONLY")
    print("\nVERDICT:", json.dumps(verdict, indent=1))

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "a02_vt_recall_direct.json"
    json.dump({
        "generated_by": "measure_a02_vt_recall.py",
        "prereg": "A02_BABILONG_MISORDER_PREREG.md 2 (Job 2.1)",
        "gpu_spent": "ZERO (CPU regeneration + BM25 only, no model loaded)",
        "why_dvr_reported_na": ("_build_sample returns gold_needle=None for VT, so the "
                                "dvr locator had nothing to localise; _make_vt DOES "
                                "return the chain -- the info was dropped at an interface"),
        "definition": ("HIT iff EVERY chain sentence's chunk is in the top-12 iter_bm25 "
                       "pack -- the same strict all-in-pack rule dvr used for NIAH"),
        "selector": "iter_bm25 topk=12 iter_hop_topk=4 chunk_size=512 (byte-identical)",
        "decision_rule": "recall>=95% => retrieval-closed directly; else restate niah-only",
        "cells": res, "verdict": verdict,
    }, open(dst, "w"), indent=1)
    print(f"wrote {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        W / "proposal/backlog/A02-comem-write-read-repair/evidence/babilong_misorder"))
    main(Path(ap.parse_args().out))
