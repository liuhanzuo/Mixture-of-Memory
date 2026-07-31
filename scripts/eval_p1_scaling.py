#!/usr/bin/env python
"""P1.1 store/distractor scaling  +  P1.5 task coverage  for QCMem (Paper A).

NEW FILE (2026-08-01). Does NOT edit any shared eval script — it only *imports*
the unmodified QCMem forward path (``scripts/eval_qcmem_babilong.py``) and the
RULER task primitives (``scripts/eval_ruler_mem_space.py``), so results stay in
the flagship口径.

Thesis under test (P1.1): QCMem's "unbounded context" = a BOUNDED read (fixed
top-k pack, ~6.6k tok) over an EXTENSIBLE store. We grow the store 128k -> 4M
tokens by adding distractors while the *relevant evidence* is fixed, and report,
per store size: recall@k, answer score, retrieval latency, index size, read
tokens. Because the flagship selector is ``iter_bm25`` (pure-CPU lexical BM25 over
token-id chunks) and only the top-k selected chunks are ever encoded to h_j, the
GPU never sees the whole store — decode memory is O(1) in store size; the store
build / lexical retrieval is the only thing that scales. This lets a single H20
address multi-million-token stores.

Store construction is done in TOKEN SPACE with EXACT gold-chunk ground truth:
each context chunk is exactly ``chunk_size`` tokens and is either pure distractor
(PG19 prose / RULER noise) or a needle chunk (a RULER-faithful needle sentence +
distractor pad). The chunk index a needle lands in is recorded, so recall@k is
measured against the true evidence chunk set (not an approximate locator). Chunk
boundaries align bit-for-bit with ``qcmem_generate``'s ``tokens.split(chunk_size)``.

Two passes:
  * ``--mode retrieval``  : model-free. Build store -> run the flagship iter_bm25
    selection -> recall@k, retrieval_ms, index_gb, read_tokens, evidence coverage.
    Cheap; run with more samples for stable numbers.
  * ``--mode full``       : loads Qwen3-8B + LoRA and additionally greedy-decodes
    the packed read -> answer score (RULER string_match_all recall).

Sharding: enumerate a deterministic job grid, keep ``job_idx % num_shards ==
shard_index``; each shard pins one GPU. Merge with ``--merge``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# RULER task primitives (needle sentence format, VT chain, scoring) — verbatim.
import scripts.eval_ruler_mem_space as ruler  # noqa: E402

CHUNK_SIZE_DEFAULT = 512

# ---- flagship needle / value generators (reuse RULER's, RULER-faithful) ----
_rand_word = ruler._rand_word
_rand_number = ruler._rand_number
_rand_uuid = ruler._rand_uuid
NIAH_NEEDLE = ruler.NIAH_NEEDLE
NOISE_HAYSTACK = ruler.NOISE_HAYSTACK
_string_match_all_one = ruler._string_match_all_one


# --------------------------------------------------------------------------- #
# Distractor token pool (PG19 prose), tokenized once + cached.
# --------------------------------------------------------------------------- #
def build_pool(tokenizer, need_tokens: int, cache_path: str,
               source: str = "pg19") -> np.ndarray:
    """Return an int32 array of >= ``need_tokens`` distractor token ids.

    PG19 natural prose (realistic haystack). Tokenized once with the model
    tokenizer and cached to ``cache_path`` (npy) so 8 shards do not re-tokenize.
    """
    if os.path.exists(cache_path):
        arr = np.load(cache_path)
        if arr.shape[0] >= need_tokens:
            return arr
    pg19 = os.path.join(PROJECT_ROOT, "data", "pg19_train.jsonl")
    ids: list[int] = []
    # ~5 chars/token; read generous chunks until we have enough tokens.
    approx_chars = int(need_tokens * 6) + 1_000_000
    with open(pg19, "r", errors="ignore") as f:
        text = f.read(approx_chars)
    import re
    text = re.sub(r"\s+", " ", text).strip()
    # tokenize in ~1MB slices (avoids one giant encode); add_special_tokens=False.
    step = 1_000_000
    for s in range(0, len(text), step):
        piece = text[s:s + step]
        ids.extend(tokenizer.encode(piece, add_special_tokens=False))
        if len(ids) >= need_tokens:
            break
    if len(ids) < need_tokens:
        # loop the corpus if the requested store exceeds what we read.
        base = list(ids)
        while len(ids) < need_tokens:
            ids.extend(base)
    arr = np.asarray(ids[:need_tokens + 4096], dtype=np.int32)
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, arr)
    except Exception as e:  # noqa: BLE001
        print(f"[pool] warn: could not cache pool: {e}", flush=True)
    return arr


# --------------------------------------------------------------------------- #
# Store builders (token space, exact gold-chunk ground truth).
# --------------------------------------------------------------------------- #
def _distractor_block(pool: np.ndarray, offset: int, n: int) -> list[int]:
    L = pool.shape[0]
    o = offset % L
    if o + n <= L:
        return pool[o:o + n].tolist()
    return (pool[o:].tolist() + pool[:n - (L - o)].tolist())


def _pad_needle_block(needle_ids: list[int], pool: np.ndarray, offset: int,
                      chunk_size: int) -> list[int]:
    """A needle chunk = needle sentence tokens + distractor pad to chunk_size.
    If a needle is longer than a chunk it is truncated (never happens for the
    short RULER needle sentences)."""
    if len(needle_ids) >= chunk_size:
        return needle_ids[:chunk_size]
    pad = _distractor_block(pool, offset, chunk_size - len(needle_ids))
    return needle_ids + pad


def _spread_indices(m_chunks: int, count: int) -> list[int]:
    """`count` distinct chunk indices spread evenly across [0, m_chunks)."""
    if count <= 0:
        return []
    if count >= m_chunks:
        return list(range(m_chunks))
    return sorted({int((i + 0.5) * m_chunks / count) for i in range(count)})


def build_niah_store(tokenizer, pool, store_tokens, chunk_size, evidence_count,
                     value_type, rng):
    """Single-key NIAH with ``evidence_count`` needle VALUES (RULER niah_multivalue
    generalisation). evidence_count=1 -> single-evidence NIAH. Returns dict."""
    m = max(evidence_count + 2, store_tokens // chunk_size)
    key = _rand_word(rng)
    plural = "uuids" if value_type == "uuids" else "numbers"
    singular = "uuid" if value_type == "uuids" else "number"
    values = [(_rand_uuid(rng) if value_type == "uuids" else _rand_number(rng))
              for _ in range(evidence_count)]
    needles = [NIAH_NEEDLE.format(type_needle_v=plural, key=key, value=v)
               for v in values]
    gold_idx = _spread_indices(m, evidence_count)
    needle_ids = [tokenizer.encode(s, add_special_tokens=False) for s in needles]

    ctx: list[int] = []
    off = rng.randint(0, pool.shape[0] - 1)
    gi = 0
    for c in range(m):
        if gi < len(gold_idx) and c == gold_idx[gi]:
            block = _pad_needle_block(needle_ids[gi], pool, off + c * chunk_size,
                                      chunk_size)
            gi += 1
        else:
            block = _distractor_block(pool, off + c * chunk_size, chunk_size)
        ctx.extend(block[:chunk_size])

    one = (evidence_count == 1)
    tw = singular if one else plural
    verb = "is" if one else "are"
    instr = (f"Some special magic {tw} are hidden within the following text. "
             f"Make sure to memorize them. I will quiz you about the {tw} "
             f"afterwards.\n")
    question = (f"\nWhat {'is' if one else 'are all'} the special magic {tw} for "
                f"{key} mentioned in the provided text?")
    answer_prefix = (f" The special magic {tw} for {key} mentioned in the "
                     f"provided text {verb}")
    query_ids = tokenizer.encode(instr + question + answer_prefix,
                                 add_special_tokens=False)
    bare_q_ids = tokenizer.encode(question.strip(), add_special_tokens=False)

    input_ids = ctx[:m * chunk_size] + query_ids
    return {
        "input_ids": input_ids, "answers": values, "gold_chunk_idx": gold_idx,
        "bare_q_ids": bare_q_ids, "m_chunks": m, "evidence_count": evidence_count,
        "query_len": len(query_ids),
    }


def build_vt_store(tokenizer, pool, store_tokens, chunk_size, num_hops, rng):
    """variable_tracking: a length-``num_hops`` assignment chain scattered across
    the store (multi-hop / distributed evidence). Evidence chunks = the chain
    sentences. Answer = the chain variable names (RULER口径)."""
    m = max(num_hops + 3, store_tokens // chunk_size)
    vars_all, chain, value = ruler._gen_chain(num_hops, rng)
    n_v = num_hops + 1
    gold_idx = _spread_indices(m, len(chain))
    chain_ids = [tokenizer.encode(s, add_special_tokens=False) for s in chain]

    ctx: list[int] = []
    # VT uses the RULER noise haystack as distractor (not prose); tokenize once.
    noise_ids = tokenizer.encode(NOISE_HAYSTACK + "\n", add_special_tokens=False)
    off = rng.randint(0, pool.shape[0] - 1)
    gi = 0
    for c in range(m):
        if gi < len(gold_idx) and c == gold_idx[gi]:
            # chain sentence + noise pad
            base = list(chain_ids[gi])
            while len(base) < chunk_size:
                base.extend(noise_ids)
            ctx.extend(base[:chunk_size])
            gi += 1
        else:
            base: list[int] = []
            while len(base) < chunk_size:
                base.extend(noise_ids)
            ctx.extend(base[:chunk_size])

    # Flagship口径: RULER prepends a fixed 4-hop in-context worked example (ending
    # in variable NAMES) so the model knows the answer format is names, not the
    # value. We put it in the QUERY region (always read at any store size), so VT
    # answer score is meaningful even at 4M where the ICL chunk would never be
    # BM25-selected. ICL size is constant (~fixed 4-hop) regardless of stress hops.
    vt_icl = ruler._make_vt_icl(rng, 4)
    body = ruler.VT_TEMPLATE.format(context="", query=value)
    prefix = ruler.VT_ANSWER_PREFIX.format(num_v=n_v, query=value)
    query_ids = tokenizer.encode(vt_icl + body + prefix, add_special_tokens=False)
    # BM25 round-1 query = the VT question line (contains the target value number).
    bare_q = ("Question: Find all variables that are assigned the value "
              f"{value} in the text above.")
    bare_q_ids = tokenizer.encode(bare_q, add_special_tokens=False)

    input_ids = ctx[:m * chunk_size] + query_ids
    return {
        "input_ids": input_ids, "answers": list(vars_all),
        "gold_chunk_idx": gold_idx, "bare_q_ids": bare_q_ids, "m_chunks": m,
        "evidence_count": len(chain), "query_len": len(query_ids),
    }


def build_cwe_store(tokenizer, pool, store_tokens, chunk_size, n_common, rng):
    """Common-Word-Extraction (global-statistics / frequency aggregation).

    ``n_common`` distinct made-up words are salted through the store at HIGH
    frequency (freq_common each); a large pool of ``uncommon`` words appear a few
    times each. The answer is the ``n_common`` most-frequent words. This requires
    a GLOBAL count over the whole store — no small evidence chunk set exists — so
    a fixed top-k read fundamentally cannot solve it. We report evidence coverage
    = fraction of common-word occurrences that land in the retrieved chunks."""
    m = max(8, store_tokens // chunk_size)
    freq_common = 30
    freq_uncommon = 3

    def mkword():
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz")
                       for _ in range(rng.randint(5, 8)))

    common = []
    while len(set(common)) < n_common:
        common.append(mkword())
    common = list(dict.fromkeys(common))[:n_common]
    n_uncommon = max(50, m * chunk_size // 40)
    uncommon = []
    while len(set(uncommon)) < n_uncommon:
        uncommon.append(mkword())
    uncommon = list(dict.fromkeys(uncommon))[:n_uncommon]

    tokens_stream: list[str] = []
    for w in common:
        tokens_stream.extend([w] * freq_common)
    for w in uncommon:
        tokens_stream.extend([w] * freq_uncommon)
    rng.shuffle(tokens_stream)
    text = " ".join(tokens_stream)
    word_ids = tokenizer.encode(text, add_special_tokens=False)
    # pad / trim to m*chunk_size with the SAME word soup (keeps freq structure).
    while len(word_ids) < m * chunk_size:
        word_ids.extend(word_ids[:m * chunk_size - len(word_ids)])
    ctx = word_ids[:m * chunk_size]

    # occurrence positions of common words (token-id match) for evidence coverage.
    common_tok = {}
    for w in common:
        wid = tokenizer.encode(" " + w, add_special_tokens=False)
        common_tok[w] = wid[-1] if wid else None

    question = (f"\nQuestion: What are the {n_common} most common words in the "
                f"text above? List only the words.")
    answer_prefix = f" Answer: The {n_common} most common words are:"
    instr = ("Below is a list of words. Some words appear far more often than "
             "others.\n")
    query_ids = tokenizer.encode(instr + question + answer_prefix,
                                 add_special_tokens=False)
    bare_q_ids = tokenizer.encode(question.strip(), add_special_tokens=False)

    input_ids = ctx[:m * chunk_size] + query_ids
    return {
        "input_ids": input_ids, "answers": common, "gold_chunk_idx": None,
        "bare_q_ids": bare_q_ids, "m_chunks": m, "evidence_count": n_common,
        "query_len": len(query_ids), "common_tok": common_tok,
        "freq_common": freq_common,
    }


# --------------------------------------------------------------------------- #
# Job grid (deterministic; both P1.1 and P1.5).
# --------------------------------------------------------------------------- #
KTOK = 1024
STORE_SCALE = [128 * KTOK, 256 * KTOK, 512 * KTOK,
               1024 * KTOK, 2048 * KTOK, 4096 * KTOK]


def build_jobs(limit: int) -> list[dict]:
    jobs: list[dict] = []

    def add(task, store, extra, n):
        for i in range(n):
            j = {"task": task, "store": store, "sample": i}
            j.update(extra)
            jobs.append(j)

    # --- P1.1 store scaling: single-evidence NIAH across 128k..4M ---
    for st in STORE_SCALE:
        add("niah_single", st, {"evidence_count": 1, "value_type": "uuids"}, limit)
    # --- P1.1 multi-hop / distributed evidence (VT chain) across 128k..4M ---
    for st in STORE_SCALE:
        add("vt", st, {"num_hops": 4}, limit)
    # --- P1.1 STRESS: required evidence count EXCEEDS top-12 (fixed-k breaks).
    #     VT chain-length sweep is the cleanest probe for the mandated iter_bm25
    #     selector (it walks a lexical chain), so evidence chunks = num_hops+1
    #     grow past the topk=12 read budget at a FIXED store size. ---
    for hp in [3, 7, 11, 15, 23, 31]:            # evidence = 4,8,12,16,24,32
        add("vt", 128 * KTOK, {"num_hops": hp}, limit)
    # --- P1.5 coverage: cross-chunk aggregation (one key, E scattered values).
    #     Stresses top-k two ways: (i) co-keyed needles are not individually
    #     lexically separable, (ii) evidence count E grows past 12. ---
    for ev in [1, 4, 8, 12, 16, 24, 32]:
        add("niah_multivalue", 128 * KTOK, {"evidence_count": ev,
                                            "value_type": "numbers"}, limit)
    # --- P1.5 coverage: GLOBAL statistics (common-word frequency). No localized
    #     evidence set exists -> fixed top-k fundamentally cannot solve it. ---
    for st in [128 * KTOK, 512 * KTOK]:
        add("cwe", st, {"n_common": 10}, limit)
    return jobs


def cell_key(j: dict) -> str:
    if j["task"] in ("niah_single", "niah_multivalue"):
        return f"{j['task']}|store={j['store']}|E={j['evidence_count']}"
    if j["task"] == "vt":
        return f"vt|store={j['store']}|hops={j['num_hops']}"
    if j["task"] == "cwe":
        return f"cwe|store={j['store']}|ncommon={j['n_common']}"
    return f"{j['task']}|store={j['store']}"


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def make_store(tokenizer, pool, j, chunk_size, rng):
    t = j["task"]
    if t in ("niah_single", "niah_multivalue"):
        return build_niah_store(tokenizer, pool, j["store"], chunk_size,
                                j["evidence_count"], j.get("value_type", "numbers"),
                                rng)
    if t == "vt":
        return build_vt_store(tokenizer, pool, j["store"], chunk_size,
                              j["num_hops"], rng)
    if t == "cwe":
        return build_cwe_store(tokenizer, pool, j["store"], chunk_size,
                               j["n_common"], rng)
    raise ValueError(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "full"], required=True)
    ap.add_argument("--model_path", type=str, default="")
    ap.add_argument("--lora_adapter", type=str, default="")
    ap.add_argument("--resume_j", type=int, default=12)
    ap.add_argument("--selector", type=str, default="iter_bm25")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--iter_rounds", type=int, default=0)
    ap.add_argument("--iter_hop_topk", type=int, default=4)
    ap.add_argument("--sink_tokens", type=str, default="bos")
    ap.add_argument("--chunk_size", type=int, default=CHUNK_SIZE_DEFAULT)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--pool_cache", type=str,
                    default=os.path.join(PROJECT_ROOT, "data",
                                         "p1_qwen3_distractor_pool.npy"))
    ap.add_argument("--hidden_size", type=int, default=4096)
    ap.add_argument("--merge", action="store_true", default=False)
    args = ap.parse_args()

    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        _merge(outdir)
        return

    jobs_all = build_jobs(args.limit)
    jobs = [j for k, j in enumerate(jobs_all) if k % args.num_shards == args.shard_index]
    print(f"[p1] shard {args.shard_index}/{args.num_shards}: {len(jobs)}/{len(jobs_all)} jobs "
          f"mode={args.mode}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,
                                              local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_store = max(j["store"] for j in jobs_all)
    pool = build_pool(tokenizer, max_store + 200_000, args.pool_cache)
    print(f"[p1] distractor pool tokens: {pool.shape[0]}", flush=True)

    qc = None
    torch = None
    if args.mode == "full":
        import torch as _torch
        torch = _torch
        import scripts.eval_qcmem_babilong as qcb
        from transformers import AutoModelForCausalLM
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[args.dtype]
        device = torch.device(args.device)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, attn_implementation=args.attn_impl,
            trust_remote_code=True, local_files_only=True).to(device).eval()
        if args.lora_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.lora_adapter).eval()
            model = model.base_model.model
        qc = qcb.QCMemModel(model, resume_j=args.resume_j)
        globals()["_qcb"] = qcb
        globals()["_device"] = device

    # iter_bm25 selection helper (flagship path), timed.
    import scripts.eval_qcmem_babilong as qcb2
    _iter_bm25 = qcb2._iter_bm25_indices

    out_path = outdir / f"results_shard{args.shard_index}of{args.num_shards}.jsonl"
    fout = open(out_path, "w")
    import torch as _t2  # for tensor building in retrieval mode too
    for n, j in enumerate(jobs):
        seed = args.seed + (hash((cell_key(j), j["sample"])) % 1_000_000)
        rng = random.Random(seed)
        st = make_store(tokenizer, pool, j, args.chunk_size, rng)
        input_ids = st["input_ids"]
        tokens = _t2.tensor(input_ids, dtype=_t2.long)
        chunks = list(tokens.split(args.chunk_size))
        context_chunks = chunks[:-1]
        n_ctx = len(context_chunks)

        # ---- flagship retrieval (iter_bm25), timed ----
        t0 = time.perf_counter()
        sel = _iter_bm25(context_chunks, list(st["bare_q_ids"]), topk=args.topk,
                         iter_rounds=args.iter_rounds, iter_hop_topk=args.iter_hop_topk)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        sel_set = set(sel)
        rec = {}
        gold = st["gold_chunk_idx"]
        if gold is not None:
            hit = len([g for g in gold if g in sel_set])
            rec["recall_at_k"] = hit / len(gold) if gold else 0.0
            rec["n_gold"] = len(gold)
        else:
            # cwe: no small evidence set -> report occurrence coverage.
            ct = st.get("common_tok", {})
            total_occ = 0
            seen_occ = 0
            sel_tok = set()
            for ci in sel:
                sel_tok.update(context_chunks[ci].tolist())
            for w, wid in ct.items():
                if wid is None:
                    continue
                cnt = sum(int((c == wid).sum()) for c in context_chunks)
                seen = sum(int((context_chunks[ci] == wid).sum()) for ci in sel)
                total_occ += cnt
                seen_occ += seen
            rec["evidence_coverage"] = (seen_occ / total_occ) if total_occ else 0.0
            rec["n_gold"] = st["evidence_count"]

        # read tokens (constant-ish): sink(1) + selected chunk lens + query len.
        sel_tok_len = sum(int(context_chunks[ci].shape[0]) for ci in sel)
        sink_len = 1 if args.sink_tokens == "bos" else 0
        read_tokens = sink_len + sel_tok_len + st["query_len"]

        # index sizes.
        index_gb = n_ctx * args.chunk_size * args.hidden_size * 2 / 1e9  # h_j bf16
        bm25_index_mb = (n_ctx * args.chunk_size) * 4 / 1e6  # int32 token ids

        row = {
            "cell": cell_key(j), "task": j["task"], "store": j["store"],
            "sample": j["sample"], "evidence_count": st["evidence_count"],
            "n_context_chunks": n_ctx, "store_tokens_actual": len(input_ids),
            "recall": rec, "n_selected": len(sel),
            "retrieval_ms": round(retrieval_ms, 2),
            "read_tokens": read_tokens, "index_gb": round(index_gb, 4),
            "bm25_index_mb": round(bm25_index_mb, 2),
        }

        if args.mode == "full":
            qcb = globals()["_qcb"]
            device = globals()["_device"]
            input_t = tokens.unsqueeze(0).to(device)
            stats = {}
            td = time.perf_counter()
            output = qcb.qcmem_generate(
                qc=qc, tokenizer=tokenizer, input_ids=input_t,
                chunk_size=args.chunk_size,
                max_new_tokens=(args.max_new_tokens if j["task"] != "vt"
                                else max(args.max_new_tokens, 60)),
                selector=args.selector, topk=args.topk,
                sink_tokens=args.sink_tokens, bare_question_ids=list(st["bare_q_ids"]),
                stats=stats, iter_rounds=args.iter_rounds,
                iter_hop_topk=args.iter_hop_topk)
            read_decode_ms = (time.perf_counter() - td) * 1000.0
            score = _string_match_all_one(output, st["answers"]) * 100.0
            row["score"] = round(score, 2)
            row["read_decode_ms"] = round(read_decode_ms, 1)
            row["read_len_actual"] = stats.get("read_len")
            row["output"] = output[:200]
        fout.write(json.dumps(row) + "\n")
        fout.flush()
        if n % 5 == 0:
            print(f"[p1] {n+1}/{len(jobs)} {row['cell']} "
                  f"recall={row['recall']} ret_ms={row['retrieval_ms']} "
                  f"{'score=' + str(row.get('score')) if args.mode=='full' else ''}",
                  flush=True)
    fout.close()
    print(f"[p1] shard done -> {out_path}", flush=True)


def _merge(outdir: Path):
    import glob
    from collections import defaultdict
    rows = []
    for fp in glob.glob(str(outdir / "results_shard*of*.jsonl")):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    agg = defaultdict(list)
    for r in rows:
        agg[r["cell"]].append(r)
    out = {}
    for cell, rs in sorted(agg.items()):
        n = len(rs)
        def mean(key, sub=None):
            vals = []
            for r in rs:
                v = r.get(key) if sub is None else r.get(sub, {}).get(key)
                if v is not None:
                    vals.append(v)
            return round(sum(vals) / len(vals), 3) if vals else None
        out[cell] = {
            "n": n,
            "task": rs[0]["task"], "store": rs[0]["store"],
            "evidence_count": rs[0]["evidence_count"],
            "n_context_chunks": rs[0]["n_context_chunks"],
            "store_tokens_actual": rs[0]["store_tokens_actual"],
            "recall_at_k": mean("recall_at_k", sub="recall"),
            "evidence_coverage": mean("evidence_coverage", sub="recall"),
            "score": mean("score"),
            "retrieval_ms": mean("retrieval_ms"),
            "read_tokens": mean("read_tokens"),
            "read_decode_ms": mean("read_decode_ms"),
            "index_gb": rs[0]["index_gb"],
            "bm25_index_mb": rs[0]["bm25_index_mb"],
        }
    with open(outdir / "merged_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n[merge] {len(rows)} rows over {len(out)} cells -> "
          f"{outdir / 'merged_summary.json'}")


if __name__ == "__main__":
    main()
