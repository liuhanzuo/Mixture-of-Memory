#!/usr/bin/env python
"""P0.20 Phase B — DENSE-retriever equal-latency frontier: dense-RAG vs CoMem.

NEW FILE (2026-08-03). Phase B is Phase A (``scripts/eval_p0_20_equal_latency.py``)
with EXACTLY ONE change: the raw-text-RAG arm's retrieval selector is swapped from
the lexical flagship ``iter_bm25`` to a FROZEN public DENSE retriever
(BGE-large-en-v1.5, CLS+L2+cosine) — the identical retriever P1.9 uses. Everything
else is held byte-identical to Phase A / config#2, and the CoMem arm is COMPLETELY
UNCHANGED from Phase A (resume_j=12, flagship LoRA, iter_bm25 selector, pre-stored
h12 fetch+H2D) so its TTFT anchor equals Phase A's.

Research question (paperA §P0.20 Phase B): at CoMem's FIXED online latency budget,
how many chunks can a deployment-realistic DENSE RAG read (k_dense*), and does the
dense retriever's better recall let latency-matched dense-RAG close/overturn the
Phase A verdict where BM25-RAG lost?  Dense retrieval is MORE expensive online
(fewer chunks fit in the budget) but higher-recall — the net effect is the result.

This module is a THIN COMPOSITION that IMPORTS BUT NEVER EDITS:
  * ``eval_p0_20_equal_latency`` (Phase A driver)  — all reused helpers: the
    model+LoRA loader with adapter toggle (``_load_with_peft``), per-benchmark
    example providers (``_provider_*`` / ``_get_provider`` — so CoMem sees the
    IDENTICAL Phase A example), the exactly-k calib pack (``_build_calib_pack``),
    the h12 store fetchers (``_make_store_fetchers``), the latency timer
    (``_timeit``), the k-freeze (``_freeze_k``), and the paired-stats primitives.
  * ``eval_p1_9_dense_rag`` — the frozen ``DenseRetriever`` (same weights, same
    sha256 fail-closed gate, same CLS+L2+cosine contract) and its provenance
    constants.  Reusing this class verbatim means the dense ranking is bit-identical
    to P1.9's; we PROVE this with a reproduction gate (top-12 vs P1.9's stored
    ``dense_sel_idx``, matched by ``input_ids_sha256``) where the seed convention
    matches (babilong/longeval/locomo).
  * ``bench_p0_13_quality_latency`` — the pack builder (``_build_pack``), arm
    replica (``_run_arm``), strict shas.  (via the Phase A driver's re-exports.)

The ONLY genuinely new logic here: (a) a dense pack builder mirroring ``_build_pack``
but with BGE-selected chunk indices; (b) dense selection latency timers under TWO
honest cost models — deployment (offline-indexed: query-encode + vector search,
PRIMARY, the fair parallel to CoMem's pre-stored h12) and cold-index (encode all
passages + query + search == P1.9's ``retrieval_latency_ms``, SENSITIVITY); (c) a
dense-vs-CoMem quality/aggregate with k_dense* frozen on the deployment latency.

See paperA/P0_20_PHASEB_NOTES.md for the full design + the two reconciled points
(k-sweep reconstruction; dense latency accounting).  NO fail-closed gate is loosened.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import random
import socket
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- import the Phase A driver verbatim (all reused helpers/providers/stats) ---
import eval_p0_20_equal_latency as p020  # noqa: E402
# ---- import P1.9's frozen dense retriever verbatim ---------------------------
import eval_p1_9_dense_rag as p19  # noqa: E402

# reused Phase A / P0.13 symbols (imported, NOT redefined) ---------------------
_load_with_peft = p020._load_with_peft
_get_provider = p020._get_provider
_n_examples = p020._n_examples
BENCH_MNT = p020.BENCH_MNT
_build_pack = p020._build_pack
_run_arm = p020._run_arm
_eos_ids = p020._eos_ids
_build_calib_pack = p020._build_calib_pack
_make_store_fetchers = p020._make_store_fetchers
_freeze_k = p020._freeze_k
_timeit = p020._timeit
_sync = p020._sync
QCMemModel = p020.QCMemModel
ruler = p020.ruler
_bare_question = p020._bare_question
_backbone_provenance = p020._backbone_provenance
_lora_modules = p020._lora_modules
_versions = p020._versions
_paired_bootstrap_ci = p020._paired_bootstrap_ci
_mcnemar_exact = p020._mcnemar_exact
EXPECTED_LORA_SHA = p020.EXPECTED_LORA_SHA
EXPECTED_LORA_MODULE_COUNT = p020.EXPECTED_LORA_MODULE_COUNT
EXPECTED_BACKBONE_KEY_SHA = p020.EXPECTED_BACKBONE_KEY_SHA
CHUNK_BYTES = p020.CHUNK_BYTES

# reused P1.9 symbols ----------------------------------------------------------
DenseRetriever = p19.DenseRetriever
EXPECTED_BGE_SHA256 = p19.EXPECTED_BGE_SHA256
EXPECTED_BGE_REVISION = p19.EXPECTED_BGE_REVISION
BGE_QUERY_INSTRUCTION = p19.BGE_QUERY_INSTRUCTION
_sha256_str = p19._sha256_str
_sha256_file = p020._sha256_file

_med = lambda t: t["median"]  # noqa: E731


# --------------------------------------------------------------------------- #
# dense pack builder — mirrors p013._build_pack EXACTLY (same dict schema so
# _run_arm consumes it identically) but selects context chunks with the frozen
# BGE retriever instead of iter_bm25.  Selection is over the DECODED chunk texts,
# identical to P1.9 (tokens.split(chunk_size); context=chunks[:-1]; query=last).
# --------------------------------------------------------------------------- #
def _build_dense_pack(input_ids, chunk_size, k, retriever, query_text, tokenizer):
    tokens = input_ids[0]
    chunks = list(tokens.split(chunk_size))
    context_chunks = chunks[:-1]
    query_chunk = chunks[-1]
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = int(tokens[0].item())
    ctx_texts = [tokenizer.decode(c.tolist(), skip_special_tokens=True)
                 for c in context_chunks]
    sel_idx, _scores, sel_lat_ms, index_bytes = retriever.select_topk(
        ctx_texts, query_text, k)  # sel_idx = sorted doc-order top-k (P1.9 contract)
    query_ids = query_chunk.tolist()
    packed_ids = [int(bos_id)]
    for i in sel_idx:
        packed_ids.extend(context_chunks[i].tolist())
    packed_ids.extend(query_ids)
    import hashlib
    pack_sha = hashlib.sha256(
        b",".join(str(t).encode() for t in packed_ids)).hexdigest()
    return {
        "bos_id": int(bos_id),
        "sel_idx": sel_idx,
        "selected_chunk_tensors": [context_chunks[i] for i in sel_idx],
        "query_ids": query_ids,
        "n_ctx_chunks": len(context_chunks),
        "pack_token_count": len(packed_ids),
        "pack_read_len": 1 + sum(int(context_chunks[i].shape[0]) for i in sel_idx)
                         + len(query_ids),
        "packed_ids_sha256": pack_sha,
        "dense_index_bytes": index_bytes,
        "dense_select_latency_ms_coldindex": sel_lat_ms,
        "input_ids_sha256": _sha256_str(",".join(map(str, tokens.tolist()))),
    }


def _load_retriever(args, device, dtype):
    rp = args.retriever_path
    if not os.path.isabs(rp):
        rp = os.path.join(PROJECT_ROOT, rp)
    retr = DenseRetriever(rp, device, dtype,
                          allow_sha_mismatch=args.allow_retriever_sha_mismatch)
    return retr


# --------------------------------------------------------------------------- #
# QUALITY mode — CoMem arm reads the UNCHANGED iter_bm25 pack; dense-RAG arm reads
# the BGE pack.  Both arms per example (paired at the example level).
# --------------------------------------------------------------------------- #
def run_quality(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, peft_model, lora_sha256, lora_layers = _load_with_peft(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p0.20B][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    retriever = _load_retriever(args, device, dtype)
    if not retriever.sha_ok and not args.allow_retriever_sha_mismatch:
        raise SystemExit(
            f"[p0.20B][ABORT] BGE sha {retriever.weight_sha256} != {EXPECTED_BGE_SHA256}")

    L = int(model.config.num_hidden_layers)
    qc0 = QCMemModel(model, resume_j=0)     # dense-RAG (LoRA disabled at read time)
    qc12 = QCMemModel(model, resume_j=12)   # CoMem (LoRA enabled) — UNCHANGED
    eos0 = _eos_ids(qc0, tokenizer)
    eos12 = _eos_ids(qc12, tokenizer)

    gen, score, task = _get_provider(args, tokenizer)
    k = args.k
    mnt = args.max_new_tokens if args.max_new_tokens > 0 else BENCH_MNT[args.benchmark]

    n_uni = _n_examples(args, tokenizer)
    n_eval = min(args.limit, n_uni) if args.limit > 0 else n_uni
    sample_indices = [i for i in range(n_eval)
                      if i % args.num_shards == args.shard_index]
    assert args.calib_offset >= args.limit, \
        f"split overlap: calib_offset={args.calib_offset} < limit={args.limit}"

    shard_tag = (f"_shard{args.shard_index}of{args.num_shards}"
                 if args.num_shards > 1 else "")
    cell = f"{args.benchmark}_{task}_{args.length}_k{k}"
    outdir = Path(args.output_dir) / "quality"
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / f"{cell}{shard_tag}.jsonl"
    fout = open(jsonl_path, "w")
    print(f"[p0.20B][quality] {cell}{shard_tag}: k={k} n={len(sample_indices)}/"
          f"{n_eval} mnt={mnt} denseRAG=j0-noLoRA CoMem=j12-LoRA(bm25)", flush=True)

    records = []
    n_done = 0
    for i in sample_indices:
        sample = gen(i)
        bare_q_ids = tokenizer.encode(sample["bare_q"], add_special_tokens=False)
        ids = tokenizer.encode(sample["prompt"], add_special_tokens=True,
                               return_tensors="pt")
        if isinstance(ids, list):
            ids = torch.tensor([ids], dtype=torch.long)
        input_ids = ids.to(device)
        approx_tokens = int(input_ids.shape[1])

        # CoMem pack — UNCHANGED flagship iter_bm25 (resume_j-independent).
        comem_pack = _build_pack(input_ids, args.chunk_size, "iter_bm25", k,
                                 args.iter_hop_topk, bare_q_ids, tokenizer)
        # dense-RAG pack — frozen BGE selection over the same document/query.
        dense_pack = _build_dense_pack(input_ids, args.chunk_size, k, retriever,
                                       sample["bare_q"], tokenizer)

        oom = False
        try:
            # CoMem arm — LoRA ENABLED, reads the iter_bm25 pack.
            genC, tC, rlC, pkC, finC, lC = _run_arm(
                qc12, tokenizer, comem_pack["bos_id"],
                comem_pack["selected_chunk_tensors"], comem_pack["query_ids"],
                mnt, eos12, capture_first=True)
            # dense-RAG arm — LoRA DISABLED (vanilla Qwen3-8B), reads the BGE pack.
            with peft_model.disable_adapter():
                genR, tR, rlR, pkR, finR, lR = _run_arm(
                    qc0, tokenizer, dense_pack["bos_id"],
                    dense_pack["selected_chunk_tensors"], dense_pack["query_ids"],
                    mnt, eos0, capture_first=True)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            oom = True
            torch.cuda.empty_cache()
            print(f"[p0.20B][OOM] i={i} {cell}: {e}", flush=True)

        if oom:
            rec = {"example_id": i, "cell": cell, "k": k, "oom": True}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            records.append(rec); n_done += 1
            continue

        predR = tokenizer.decode(genR, skip_special_tokens=True).strip()
        predC = tokenizer.decode(genC, skip_special_tokens=True).strip()
        scR = score(predR, sample)
        scC = score(predC, sample)

        # read-len invariant: both arms pack exactly min(k, n_ctx) full 512-tok
        # chunks + same query chunk => equal token count even though the chosen
        # chunk INDICES differ (full-deployment vs full-deployment).
        assert rlR == rlC == comem_pack["pack_read_len"] == dense_pack["pack_read_len"], \
            (f"read_len mismatch i={i}: R={rlR} C={rlC} "
             f"comem={comem_pack['pack_read_len']} dense={dense_pack['pack_read_len']}")

        rec = {
            "example_id": i, "cell": cell, "benchmark": args.benchmark,
            "task": task, "length": args.length, "k": k,
            "approx_tokens": approx_tokens,
            "gold": " | ".join(sample["answers"]),
            "comem_retrieved_chunk_ids": comem_pack["sel_idx"],
            "dense_retrieved_chunk_ids": dense_pack["sel_idx"],
            "n_ctx_chunks": comem_pack["n_ctx_chunks"],
            "n_selected_comem": len(comem_pack["sel_idx"]),
            "n_selected_dense": len(dense_pack["sel_idx"]),
            "pack_read_len": comem_pack["pack_read_len"],
            "comem_packed_ids_sha256": comem_pack["packed_ids_sha256"],
            "dense_packed_ids_sha256": dense_pack["packed_ids_sha256"],
            "input_ids_sha256": dense_pack["input_ids_sha256"],
            "dense_index_bytes": dense_pack["dense_index_bytes"],
            "comem_h12_store_bytes": len(comem_pack["sel_idx"]) * CHUNK_BYTES,
            "lora_sha256": lora_sha256,
            "dense_rag": {"resume_j": 0, "lora": False, "selector": "dense_bge",
                          "prediction": predR, "score": scR["score"],
                          "correct": scR["correct"], "f1": scR.get("f1"),
                          "gen_len": len(genR), "read_len": rlR,
                          "peak_gb": pkR, "finite": finR},
            "comem": {"resume_j": 12, "lora": True, "selector": "iter_bm25",
                      "prediction": predC, "score": scC["score"],
                      "correct": scC["correct"], "f1": scC.get("f1"),
                      "gen_len": len(genC), "read_len": rlC,
                      "peak_gb": pkC, "finite": finC},
            "diff_comem_minus_dense": scC["score"] - scR["score"],
        }
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        records.append(rec); n_done += 1
        torch.cuda.empty_cache()
        if n_done % 10 == 0:
            print(f"[p0.20B][quality] {cell}{shard_tag} {n_done} done "
                  f"(CoMem={scC['score']:.2f} dense={scR['score']:.2f} "
                  f"nselC={len(comem_pack['sel_idx'])} nselD={len(dense_pack['sel_idx'])} "
                  f"readlen={rlC})", flush=True)
    fout.close()

    valid = [r for r in records if not r.get("oom")]

    def _mean(arm):
        xs = [r[arm]["score"] for r in valid]
        return round(sum(xs) / len(xs) * 100.0, 3) if xs else 0.0

    cell_json = {
        "cell": cell, "benchmark": args.benchmark, "task": task,
        "length": args.length, "k": k, "shard": shard_tag,
        "n": len(records), "n_valid": len(valid),
        "oom_count": sum(1 for r in records if r.get("oom")),
        "comem_score": _mean("comem"), "dense_rag_score": _mean("dense_rag"),
        "diff_comem_minus_dense": round(_mean("comem") - _mean("dense_rag"), 3),
        "mean_n_selected_comem": round(
            sum(r["n_selected_comem"] for r in valid) / len(valid), 2) if valid else 0,
        "mean_n_selected_dense": round(
            sum(r["n_selected_dense"] for r in valid) / len(valid), 2) if valid else 0,
        "mean_read_len": round(
            sum(r["pack_read_len"] for r in valid) / len(valid), 1) if valid else 0,
        "comem_selector": "iter_bm25", "dense_selector": "dense_bge",
        "iter_hop_topk": args.iter_hop_topk,
        "chunk_size": args.chunk_size, "max_new_tokens": mnt,
        "lora_sha256": lora_sha256, "bge_sha256": retriever.weight_sha256,
        "num_layers": L,
        "runtime": {"node": socket.gethostname(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                    "seed": args.seed, "dtype": args.dtype,
                    "attn_implementation": args.attn_impl},
        "jsonl": str(jsonl_path),
    }
    with open(outdir / f"{cell}{shard_tag}_cell.json", "w") as f:
        json.dump(cell_json, f, indent=2)
    print(f"[p0.20B][quality] DONE {cell}{shard_tag}: "
          f"CoMem={cell_json['comem_score']} dense={cell_json['dense_rag_score']} "
          f"diff={cell_json['diff_comem_minus_dense']} "
          f"(n_valid={len(valid)})", flush=True)


# --------------------------------------------------------------------------- #
# CALIBRATION LATENCY mode — TTFT breakdown. CoMem side identical to Phase A
# (iter_bm25 selection + h12 fetch + bottom-12 write + layers[12:36] read). The
# dense-RAG side replaces the SELECTION cost with the dense retriever's cost under
# two models: deployment (offline-indexed: query-encode + flat cosine search) and
# cold-index (encode all passages + query + search == P1.9's per-query cost). The
# read/write phases are pack-token-count-only and thus reuse the exactly-k pack.
# --------------------------------------------------------------------------- #
def run_calib_latency(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, peft_model, lora_sha256, lora_layers = _load_with_peft(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p0.20B][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    retriever = _load_retriever(args, device, dtype)
    if not retriever.sha_ok and not args.allow_retriever_sha_mismatch:
        raise SystemExit(
            f"[p0.20B][ABORT] BGE sha {retriever.weight_sha256} != {EXPECTED_BGE_SHA256}")
    qc0 = QCMemModel(model, resume_j=0)
    qc12 = QCMemModel(model, resume_j=12)
    k = args.k

    # RESERVED calibration example (>= calib_offset -> disjoint from quality).
    ct = ruler._LENGTH_TOKENS[args.calib_length]
    task = "niah_multikey_1"
    base_seed = args.seed + (hash((task, args.calib_length)) % 100000)
    ci = args.calib_offset + args.proc_id
    rng = random.Random(base_seed * 1000 + ci)
    prompt, answers, gold_needle = ruler._build_sample(task, ct, tokenizer, rng, None)
    ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    if isinstance(ids, list):
        ids = torch.tensor([ids], dtype=torch.long)
    input_ids = ids.to(device)
    pack = _build_calib_pack(input_ids, args.chunk_size, k, tokenizer)
    n_ctx = int(input_ids.shape[1]) // args.chunk_size
    n_store = max(64, n_ctx, k + 8)

    # dense corpus: decoded context chunk texts + the bare-question query.
    tokens = input_ids[0]
    chunks = list(tokens.split(args.chunk_size))
    context_chunks = chunks[:-1]
    ctx_texts = [tokenizer.decode(c.tolist(), skip_special_tokens=True)
                 for c in context_chunks]
    query_text = _bare_question(prompt)
    # offline-built flat index (encoded ONCE; NOT charged to online TTFT) for the
    # deployment cost model — this is the exact analogue of CoMem pre-storing h12.
    ctx_emb = retriever._encode_texts(ctx_texts, is_query=False)  # [n_ctx, d] cpu f32
    n_dense = len(ctx_texts)

    # ---- precompute the hiddens the READ consumes (offline; NOT online cost) ----
    with torch.no_grad():
        sink_hj0 = qc0.write_chunk([pack["bos_id"]])
        sel_hj0 = qc0.write_chunks(list(pack["selected_chunk_tensors"])) \
            if pack["selected_chunk_tensors"] else []
        q_hj0, _, _ = qc0.write_prefill(pack["query_ids"])
        sink_hj12 = qc12.write_chunk([pack["bos_id"]])
        sel_hj12 = qc12.write_chunks(list(pack["selected_chunk_tensors"])) \
            if pack["selected_chunk_tensors"] else []
        q_hj12, _, _ = qc12.write_prefill(pack["query_ids"])

    gpu_fetch, cpu_gather, cpu_h2d = _make_store_fetchers(
        max(1, pack["k_effective"]), n_store, device)

    W, N = args.warmup, args.n_repeat

    # ---- CoMem selection (UNCHANGED iter_bm25, resume_j-independent) ----------
    @torch.no_grad()
    def _comem_sel():
        bare_q_ids = tokenizer.encode(query_text, add_special_tokens=False)
        _build_pack(input_ids, args.chunk_size, "iter_bm25", k,
                    args.iter_hop_topk, bare_q_ids, tokenizer)
    comem_sel_t = _timeit(_comem_sel, W, N, gpu_sync=True)

    # ---- dense selection, DEPLOYMENT model (offline-indexed) ------------------
    # online cost = encode query only + flat cosine search over pre-built ctx_emb.
    @torch.no_grad()
    def _dense_sel_deploy():
        q_emb = retriever._encode_texts([query_text], is_query=True)  # [1, d]
        sims = (ctx_emb @ q_emb[0]).tolist()
        order = sorted(range(n_dense), key=lambda i: (-sims[i], i))
        _ = sorted(order[:min(k, n_dense)])
    dense_deploy_t = _timeit(_dense_sel_deploy, W, N, gpu_sync=True)

    # ---- dense selection, COLD-INDEX model (no offline index) -----------------
    # online cost = encode ALL context passages + query + search (== P1.9 per query).
    @torch.no_grad()
    def _dense_sel_cold():
        retriever.select_topk(ctx_texts, query_text, k)
    dense_cold_t = _timeit(_dense_sel_cold, W, N, gpu_sync=True)

    # ---- dense-RAG online write (j=0): sink + all k context chunks + query ----
    @torch.no_grad()
    def _rag_write():
        with peft_model.disable_adapter():
            qc0.write_chunk([pack["bos_id"]])
            if pack["selected_chunk_tensors"]:
                qc0.write_chunks(list(pack["selected_chunk_tensors"]))
            qc0.write_prefill(pack["query_ids"])
    rag_write_t = _timeit(_rag_write, W, N, gpu_sync=True)

    @torch.no_grad()
    def _rag_read():
        with peft_model.disable_adapter():
            qc0.read_prefill(sink_hj0, sel_hj0, q_hj0)
    rag_read_t = _timeit(_rag_read, W, N, gpu_sync=True)

    # ---- CoMem online write (j=12): bottom-12 over sink + query ONLY ----------
    @torch.no_grad()
    def _comem_write():
        qc12.write_chunk([pack["bos_id"]])
        qc12.write_prefill(pack["query_ids"])
    comem_write_t = _timeit(_comem_write, W, N, gpu_sync=True)

    @torch.no_grad()
    def _comem_read():
        qc12.read_prefill(sink_hj12, sel_hj12, q_hj12)
    comem_read_t = _timeit(_comem_read, W, N, gpu_sync=True)

    gpu_fetch_t = _timeit(gpu_fetch, W, N, gpu_sync=True)
    cpu_gather_t = _timeit(cpu_gather, W, N, gpu_sync=False)
    cpu_h2d_t = _timeit(cpu_h2d, W, N, gpu_sync=True)

    # ---- assemble TTFT (to first logits; decode EXCLUDED) ---------------------
    # CoMem TTFT — IDENTICAL construction to Phase A (iter_bm25 sel + fetch + write + read).
    comem_ttft_gpu = (_med(comem_sel_t) + _med(gpu_fetch_t)
                      + _med(comem_write_t) + _med(comem_read_t))
    comem_ttft_cpu = (_med(comem_sel_t) + _med(cpu_gather_t) + _med(cpu_h2d_t)
                      + _med(comem_write_t) + _med(comem_read_t))
    # dense-RAG TTFT — two selection cost models; write/read identical to Phase A's j0-RAG.
    densrag_ttft_deploy = _med(dense_deploy_t) + _med(rag_write_t) + _med(rag_read_t)
    densrag_ttft_cold = _med(dense_cold_t) + _med(rag_write_t) + _med(rag_read_t)

    outdir = Path(args.output_dir) / "calib_latency"
    outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": "calib_latency", "phase": "B_dense", "proc_id": args.proc_id, "k": k,
        "k_effective": pack["k_effective"], "calib_index": ci,
        "calib_length": args.calib_length, "task": task,
        "pack_read_len": pack["pack_read_len"], "n_store_rows": n_store,
        "n_dense_ctx_chunks": n_dense,
        "store_bytes_per_chunk": CHUNK_BYTES,
        "pack_h12_bytes": pack["k_effective"] * CHUNK_BYTES,
        "dense_index_bytes": n_dense * retriever.hidden * retriever.dtype_bytes,
        "components_ms": {
            "comem_selection_bm25": {"median": _med(comem_sel_t) * 1e3, "p95": comem_sel_t["p95"] * 1e3},
            "dense_selection_deploy": {"median": _med(dense_deploy_t) * 1e3, "p95": dense_deploy_t["p95"] * 1e3},
            "dense_selection_coldindex": {"median": _med(dense_cold_t) * 1e3, "p95": dense_cold_t["p95"] * 1e3},
            "dense_rag_write_j0": {"median": _med(rag_write_t) * 1e3, "p95": rag_write_t["p95"] * 1e3},
            "dense_rag_read_j0": {"median": _med(rag_read_t) * 1e3, "p95": rag_read_t["p95"] * 1e3},
            "comem_write_j12_sinkquery": {"median": _med(comem_write_t) * 1e3, "p95": comem_write_t["p95"] * 1e3},
            "comem_read_j12": {"median": _med(comem_read_t) * 1e3, "p95": comem_read_t["p95"] * 1e3},
            "comem_fetch_gpu_resident": {"median": _med(gpu_fetch_t) * 1e3, "p95": gpu_fetch_t["p95"] * 1e3},
            "comem_fetch_cpu_gather": {"median": _med(cpu_gather_t) * 1e3, "p95": cpu_gather_t["p95"] * 1e3},
            "comem_h2d_cpu_pinned": {"median": _med(cpu_h2d_t) * 1e3, "p95": cpu_h2d_t["p95"] * 1e3},
        },
        "ttft_ms": {
            "dense_rag_deploy": densrag_ttft_deploy * 1e3,
            "dense_rag_coldindex": densrag_ttft_cold * 1e3,
            "comem_gpu_resident": comem_ttft_gpu * 1e3,
            "comem_cpu_pinned": comem_ttft_cpu * 1e3,
        },
        "model_only_read_ms": {
            "dense_rag_j0": _med(rag_read_t) * 1e3,
            "comem_j12": _med(comem_read_t) * 1e3,
        },
        "config": {"resume_j_rag": 0, "resume_j_comem": 12, "chunk_size": args.chunk_size,
                   "warmup": W, "n_repeat": N, "dtype": args.dtype,
                   "attn_impl": args.attn_impl, "lora_sha256": lora_sha256,
                   "bge_sha256": retriever.weight_sha256,
                   "dense_search": "flat brute-force cosine (CPU, == P1.9)"},
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "python": platform.python_version(), "node": socket.gethostname()},
    }
    with open(outdir / f"calib_k{k}_proc{args.proc_id}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[p0.20B][calib] k={k}(eff={pack['k_effective']}) proc={args.proc_id} "
          f"TTFT ms: dense(deploy)={densrag_ttft_deploy*1e3:.1f} "
          f"dense(cold)={densrag_ttft_cold*1e3:.1f} "
          f"CoMem(gpu)={comem_ttft_gpu*1e3:.1f} CoMem(cpu)={comem_ttft_cpu*1e3:.1f} | "
          f"dense_sel deploy={_med(dense_deploy_t)*1e3:.1f} cold={_med(dense_cold_t)*1e3:.1f}",
          flush=True)


# --------------------------------------------------------------------------- #
# MANIFEST mode — Phase A strict-fix gate PLUS the BGE weight sha gate.
# --------------------------------------------------------------------------- #
def run_manifest(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, peft_model, lora_sha256, lora_layers = _load_with_peft(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    qc0 = QCMemModel(model, resume_j=0)
    prov_backbone = _backbone_provenance(qc0, args.model_path)
    prov_lora = _lora_modules(model)
    prov_versions = _versions(device)
    retriever = _load_retriever(args, device, dtype)

    abort = []
    if lora_sha256 != EXPECTED_LORA_SHA:
        abort.append(f"LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    for kk, vv in EXPECTED_BACKBONE_KEY_SHA.items():
        got = prov_backbone["key_tensor_sha256"].get(kk)
        if got != vv:
            abort.append(f"backbone {kk} sha {got} != expected {vv}")
    if prov_lora["count"] != EXPECTED_LORA_MODULE_COUNT:
        abort.append(f"LoRA module count {prov_lora['count']} != {EXPECTED_LORA_MODULE_COUNT}")
    if sorted(lora_layers or []) != list(range(12, 36)):
        abort.append(f"LoRA layers_to_transform {lora_layers} != [12..35]")
    if retriever.weight_sha256 != EXPECTED_BGE_SHA256:
        abort.append(f"BGE sha {retriever.weight_sha256} != expected {EXPECTED_BGE_SHA256}")
    if retriever.pooling != "cls":
        abort.append(f"BGE pooling {retriever.pooling} != cls")

    manifest = {
        "run": "P0.20_equal_latency_phaseB_dense",
        "arms": {"dense_rag": {"resume_j": 0, "lora": False, "selector": "dense_bge",
                               "note": "vanilla Qwen3-8B full 36-layer recompute over BGE-selected pack"},
                 "comem": {"resume_j": 12, "lora": True, "selector": "iter_bm25",
                           "note": "flagship UNCHANGED: fetch pre-stored h12, resume layers[12:36]"}},
        "strict_fixes": {
            "model_path": args.model_path, "lora_adapter": args.lora_adapter,
            "lora_sha256": lora_sha256, "expected_lora_sha256": EXPECTED_LORA_SHA,
            "lora_sha_match": lora_sha256 == EXPECTED_LORA_SHA,
            "lora_layers_to_transform": lora_layers,
            "lora_module_count": prov_lora["count"],
            "retriever_path": args.retriever_path,
            "bge_sha256": retriever.weight_sha256,
            "expected_bge_sha256": EXPECTED_BGE_SHA256,
            "bge_sha_match": retriever.weight_sha256 == EXPECTED_BGE_SHA256,
            "bge_revision": EXPECTED_BGE_REVISION,
            "bge_pooling": retriever.pooling, "bge_hidden": retriever.hidden,
            "comem_selector": "iter_bm25", "dense_selector": "dense_bge",
            "iter_hop_topk": args.iter_hop_topk,
            "sink_tokens": "bos", "chunk_size": args.chunk_size,
            "chat_template": False, "enable_thinking": False, "add_special_tokens": True,
            "dtype": args.dtype, "attn_impl": args.attn_impl, "seed": args.seed,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "provenance": {"backbone": prov_backbone, "lora": prov_lora,
                       "versions": prov_versions,
                       "retriever": {"model": "BAAI/bge-large-en-v1.5",
                                     "sha256": retriever.weight_sha256,
                                     "revision": EXPECTED_BGE_REVISION,
                                     "pooling": retriever.pooling,
                                     "normalization": "l2",
                                     "metric": "cosine",
                                     "query_instruction": BGE_QUERY_INSTRUCTION}},
        "command": " ".join(sys.argv), "abort_reasons": abort,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    if abort:
        print("[p0.20B][manifest][ABORT] strict-fix mismatch:", flush=True)
        for a in abort:
            print("   - " + a, flush=True)
        sys.exit(3)
    print(f"[p0.20B][manifest] OK — LoRA {lora_sha256[:12]}… layers[12..35] "
          f"{prov_lora['count']} mods; BGE {retriever.weight_sha256[:12]}… "
          f"pooling={retriever.pooling}; torch {prov_versions['torch']} "
          f"git {prov_versions['git_commit_short']}", flush=True)


# --------------------------------------------------------------------------- #
# SANITY mode — Phase A gates + dense determinism + P1.9 reproduction cross-check.
# --------------------------------------------------------------------------- #
def run_sanity(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, peft_model, lora_sha256, lora_layers = _load_with_peft(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    retriever = _load_retriever(args, device, dtype)
    fails = []
    warns = []
    if lora_sha256 != EXPECTED_LORA_SHA:
        fails.append(f"LoRA sha {lora_sha256} != {EXPECTED_LORA_SHA}")
    if retriever.weight_sha256 != EXPECTED_BGE_SHA256 and not args.allow_retriever_sha_mismatch:
        fails.append(f"BGE sha {retriever.weight_sha256} != {EXPECTED_BGE_SHA256}")

    # (1) disable_adapter() structurally toggles the LoRA layers.
    def _n_active_lora():
        n = 0
        for mod in model.modules():
            if hasattr(mod, "active_adapters") and hasattr(mod, "lora_A"):
                if getattr(mod, "_disable_adapters", None) is False:
                    n += 1
        return n
    outside = _n_active_lora()
    with peft_model.disable_adapter():
        inside = _n_active_lora()
    if not (outside > 0 and inside == 0):
        fails.append(f"disable_adapter did not toggle LoRA (out={outside} in={inside})")

    # (2) build both packs on example 0; check read-len arm equality + finite logits.
    gen, score, task = _get_provider(args, tokenizer)
    sample = gen(0)
    bare_q_ids = tokenizer.encode(sample["bare_q"], add_special_tokens=False)
    ids = tokenizer.encode(sample["prompt"], add_special_tokens=True,
                           return_tensors="pt")
    if isinstance(ids, list):
        ids = torch.tensor([ids], dtype=torch.long)
    input_ids = ids.to(device)
    comem_pack = _build_pack(input_ids, args.chunk_size, "iter_bm25", args.k,
                             args.iter_hop_topk, bare_q_ids, tokenizer)
    dense_pack = _build_dense_pack(input_ids, args.chunk_size, args.k, retriever,
                                   sample["bare_q"], tokenizer)
    # (3) dense determinism: recompute selection -> identical top-k.
    dense_pack2 = _build_dense_pack(input_ids, args.chunk_size, args.k, retriever,
                                    sample["bare_q"], tokenizer)
    if dense_pack["sel_idx"] != dense_pack2["sel_idx"]:
        fails.append(f"dense selection non-deterministic: "
                     f"{dense_pack['sel_idx']} != {dense_pack2['sel_idx']}")

    qc0 = QCMemModel(model, resume_j=0)
    qc12 = QCMemModel(model, resume_j=12)
    with peft_model.disable_adapter():
        _, _, rlR, _, finR, _ = _run_arm(
            qc0, tokenizer, dense_pack["bos_id"],
            dense_pack["selected_chunk_tensors"], dense_pack["query_ids"], 4,
            _eos_ids(qc0, tokenizer))
    _, _, rlC, _, finC, _ = _run_arm(
        qc12, tokenizer, comem_pack["bos_id"],
        comem_pack["selected_chunk_tensors"], comem_pack["query_ids"], 4,
        _eos_ids(qc12, tokenizer))
    if not (rlR == rlC == comem_pack["pack_read_len"] == dense_pack["pack_read_len"]):
        fails.append(f"read_len mismatch R={rlR} C={rlC} "
                     f"comem={comem_pack['pack_read_len']} dense={dense_pack['pack_read_len']}")
    if not (finR and finC):
        fails.append(f"non-finite logits R={finR} C={finC}")

    # (4) calib/quality split isolation.
    if not (args.calib_offset >= args.limit):
        fails.append(f"split overlap calib_offset={args.calib_offset} < limit={args.limit}")

    # (5) P1.9 reproduction cross-check (where seed convention matches). Look up the
    #     P1.9 stored dense_sel_idx for this input_ids_sha256 and assert our recompute
    #     of the SAME frozen retriever at topk=12 reproduces it. Skipped-with-warning
    #     if P1.9 data absent on this node or benchmark seed differs (ruler).
    p19_check = {"attempted": False, "matched": None, "note": ""}
    if args.benchmark in ("babilong", "longeval", "locomo"):
        p19_dir = Path(args.p19_output_dir) / args.benchmark
        found = None
        if p19_dir.exists():
            target_sha = dense_pack["input_ids_sha256"]
            for jf in sorted(p19_dir.glob("*.jsonl")):
                with open(jf) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        r = json.loads(line)
                        if r.get("input_ids_sha256") == target_sha:
                            found = r
                            break
                if found:
                    break
        if found is not None:
            p19_check["attempted"] = True
            # recompute at topk=12 (P1.9's TOPK) to compare like-for-like.
            repro = _build_dense_pack(input_ids, args.chunk_size, 12, retriever,
                                      sample["bare_q"], tokenizer)
            p19_sel = list(found.get("dense_sel_idx") or [])
            p19_check["matched"] = (repro["sel_idx"] == p19_sel)
            p19_check["repro_sel"] = repro["sel_idx"]
            p19_check["p19_sel"] = p19_sel
            if not p19_check["matched"]:
                fails.append(f"P1.9 dense reproduction MISMATCH for sha "
                             f"{target_sha[:12]}…: repro={repro['sel_idx']} "
                             f"p19={p19_sel}")
        else:
            p19_check["note"] = ("no P1.9 record with matching input_ids_sha256 on "
                                 "this node (P1.9 not run here / diff shard); cross-"
                                 "check skipped — determinism gate still enforced.")
            warns.append(p19_check["note"])
    else:
        p19_check["note"] = (f"benchmark={args.benchmark}: P1.9 seed convention differs "
                             f"(ruler crc32 vs Phase A hash) -> cross-check N/A; "
                             f"pairing is within Phase B.")
        warns.append(p19_check["note"])

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    res = {"mode": "sanity", "phase": "B_dense", "lora_sha256": lora_sha256,
           "bge_sha256": retriever.weight_sha256,
           "lora_active_outside": outside, "lora_active_inside_disable": inside,
           "example0_read_len": {"dense_rag": rlR, "comem": rlC,
                                 "comem_pack": comem_pack["pack_read_len"],
                                 "dense_pack": dense_pack["pack_read_len"],
                                 "n_selected_comem": len(comem_pack["sel_idx"]),
                                 "n_selected_dense": len(dense_pack["sel_idx"])},
           "comem_packed_ids_sha256": comem_pack["packed_ids_sha256"],
           "dense_packed_ids_sha256": dense_pack["packed_ids_sha256"],
           "dense_deterministic": dense_pack["sel_idx"] == dense_pack2["sel_idx"],
           "p19_reproduction": p19_check,
           "calib_offset": args.calib_offset, "limit": args.limit,
           "split_disjoint": bool(args.calib_offset >= args.limit),
           "warns": warns, "fails": fails}
    with open(outdir / "sanity.json", "w") as f:
        json.dump(res, f, indent=2)
    for w in warns:
        print("[p0.20B][sanity][WARN] " + w, flush=True)
    if fails:
        print("[p0.20B][sanity][FAIL]:", flush=True)
        for a in fails:
            print("   - " + a, flush=True)
        sys.exit(4)
    print(f"[p0.20B][sanity] OK — LoRA toggles ({outside}->{inside}); dense "
          f"deterministic; read_len paired={comem_pack['pack_read_len']} "
          f"(nselC={len(comem_pack['sel_idx'])} nselD={len(dense_pack['sel_idx'])}); "
          f"P1.9 repro={p19_check.get('matched')}; split disjoint.", flush=True)


# --------------------------------------------------------------------------- #
# AGGREGATE mode — freeze k_dense*, build frontier, anchor stats, decision.
# --------------------------------------------------------------------------- #
def run_aggregate(args):
    outdir = Path(args.output_dir)

    # ---- 1) latency: median TTFT vs k per arm (pool procs) ----
    calib = {}
    for lf in sorted(glob.glob(str(outdir / "calib_latency" / "calib_k*_proc*.json"))):
        with open(lf) as f:
            calib.setdefault(json.load(f)["k"], []).append(lf)
    lat_by_k = {}
    for k, files in calib.items():
        dd, dc, cg, cc, rd, cd = [], [], [], [], [], []
        for fp in files:
            with open(fp) as f:
                d = json.load(f)
            dd.append(d["ttft_ms"]["dense_rag_deploy"])
            dc.append(d["ttft_ms"]["dense_rag_coldindex"])
            cg.append(d["ttft_ms"]["comem_gpu_resident"])
            cc.append(d["ttft_ms"]["comem_cpu_pinned"])
            rd.append(d["model_only_read_ms"]["dense_rag_j0"])
            cd.append(d["model_only_read_ms"]["comem_j12"])
        lat_by_k[k] = {
            "n_procs": len(files),
            "dense_rag_deploy_ttft_ms": statistics.median(dd),
            "dense_rag_cold_ttft_ms": statistics.median(dc),
            "comem_gpu_ttft_ms": statistics.median(cg),
            "comem_cpu_ttft_ms": statistics.median(cc),
            "dense_rag_read_ms": statistics.median(rd),
            "comem_read_ms": statistics.median(cd),
        }
    ks_sorted = sorted(lat_by_k.keys())

    frozen = {}
    if 12 in lat_by_k:
        dense_deploy = {k: v["dense_rag_deploy_ttft_ms"] for k, v in lat_by_k.items()}
        dense_cold = {k: v["dense_rag_cold_ttft_ms"] for k, v in lat_by_k.items()}
        # PRIMARY anchor: CoMem(k=12) budget -> largest k_dense* dense-RAG fits.
        frozen["primary_anchor_comem12"] = {
            "gpu_resident_deploy": _freeze_k(lat_by_k[12]["comem_gpu_ttft_ms"], dense_deploy, args.tol),
            "cpu_pinned_deploy": _freeze_k(lat_by_k[12]["comem_cpu_ttft_ms"], dense_deploy, args.tol),
            "gpu_resident_coldindex": _freeze_k(lat_by_k[12]["comem_gpu_ttft_ms"], dense_cold, args.tol),
            "cpu_pinned_coldindex": _freeze_k(lat_by_k[12]["comem_cpu_ttft_ms"], dense_cold, args.tol),
        }

    # ---- 2) quality: merge per-example records, group by (cell,k) ----
    all_recs = []
    for jf in sorted(glob.glob(str(outdir / "quality" / "*.jsonl"))):
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_recs.append(json.loads(line))
    valid = [r for r in all_recs if not r.get("oom")]
    seen = {}
    for r in valid:
        seen[(r["cell"], r["example_id"])] = r
    valid = list(seen.values())

    cells = {}
    for r in valid:
        key = (r["benchmark"], r["task"], r["length"], r["k"])
        cells.setdefault(key, []).append(r)
    per_cell = {}
    for key, recs in sorted(cells.items()):
        c = sum(x["comem"]["score"] for x in recs) / len(recs) * 100.0
        rr = sum(x["dense_rag"]["score"] for x in recs) / len(recs) * 100.0
        cellname = f"{key[0]}/{key[1]}/{key[2]}/k{key[3]}"
        per_cell[cellname] = {
            "n": len(recs), "k": key[3],
            "comem": round(c, 2), "dense_rag": round(rr, 2),
            "diff_comem_minus_dense": round(c - rr, 2),
            "mean_read_len": round(sum(x["pack_read_len"] for x in recs) / len(recs), 1),
            "mean_n_selected_comem": round(sum(x["n_selected_comem"] for x in recs) / len(recs), 2),
            "mean_n_selected_dense": round(sum(x["n_selected_dense"] for x in recs) / len(recs), 2),
        }

    # ---- frontier: per benchmark/length, accuracy vs k for both arms ----
    frontier = {}
    fkey = {}
    for r in valid:
        fkey.setdefault((r["benchmark"], r["task"], r["length"]), {}).setdefault(
            r["k"], []).append(r)
    for grp, byk in sorted(fkey.items()):
        gname = f"{grp[0]}/{grp[1]}/{grp[2]}"
        pts = []
        for k in sorted(byk):
            recs = byk[k]
            c = sum(x["comem"]["score"] for x in recs) / len(recs) * 100.0
            rr = sum(x["dense_rag"]["score"] for x in recs) / len(recs) * 100.0
            pt = {"k": k, "n": len(recs),
                  "comem_acc": round(c, 2), "dense_rag_acc": round(rr, 2),
                  "mean_read_len": round(sum(x["pack_read_len"] for x in recs) / len(recs), 1)}
            if k in lat_by_k:
                pt["dense_rag_deploy_ttft_ms"] = round(lat_by_k[k]["dense_rag_deploy_ttft_ms"], 2)
                pt["dense_rag_cold_ttft_ms"] = round(lat_by_k[k]["dense_rag_cold_ttft_ms"], 2)
                pt["comem_gpu_ttft_ms"] = round(lat_by_k[k]["comem_gpu_ttft_ms"], 2)
                pt["comem_cpu_ttft_ms"] = round(lat_by_k[k]["comem_cpu_ttft_ms"], 2)
                pt["dense_rag_read_ms"] = round(lat_by_k[k]["dense_rag_read_ms"], 2)
                pt["comem_read_ms"] = round(lat_by_k[k]["comem_read_ms"], 2)
            pts.append(pt)
        frontier[gname] = pts

    # ---- 3) anchor quality comparisons w/ paired bootstrap CI + McNemar ----
    def _anchor_stats(comem_k, dense_k):
        """Paired: CoMem@comem_k vs dense-RAG@dense_k, per example (same ids)."""
        out = {"comem_k": comem_k, "dense_k": dense_k, "per_cell": {}}
        macro_c, macro_r, all_diffs = [], [], []
        b = c = both = neither = 0
        for grp, byk in sorted(fkey.items()):
            gname = f"{grp[0]}/{grp[1]}/{grp[2]}"
            if comem_k not in byk or dense_k not in byk:
                continue
            cmap = {x["example_id"]: x for x in byk[comem_k]}
            rmap = {x["example_id"]: x for x in byk[dense_k]}
            ids = sorted(set(cmap) & set(rmap))
            if not ids:
                continue
            cell_diffs = []
            for i in ids:
                cs = cmap[i]["comem"]["score"]
                rs = rmap[i]["dense_rag"]["score"]
                cell_diffs.append((cs - rs) * 100.0)
                cc_ = cmap[i]["comem"]["correct"]
                rc_ = rmap[i]["dense_rag"]["correct"]
                if cc_ and not rc_:
                    b += 1
                elif rc_ and not cc_:
                    c += 1
                elif cc_ and rc_:
                    both += 1
                else:
                    neither += 1
            cm = sum(cmap[i]["comem"]["score"] for i in ids) / len(ids) * 100.0
            rm = sum(rmap[i]["dense_rag"]["score"] for i in ids) / len(ids) * 100.0
            out["per_cell"][gname] = {"n": len(ids), "comem": round(cm, 2),
                                      "dense_rag": round(rm, 2),
                                      "diff": round(cm - rm, 2)}
            macro_c.append(cm); macro_r.append(rm)
            all_diffs.extend(cell_diffs)
        if macro_c:
            mc = sum(macro_c) / len(macro_c)
            mr = sum(macro_r) / len(macro_r)
            mean_d, lo, hi = _paired_bootstrap_ci(all_diffs, n_boot=args.n_boot)
            out.update({
                "macro_comem": round(mc, 2), "macro_dense_rag": round(mr, 2),
                "macro_diff_comem_minus_dense": round(mc - mr, 2),
                "paired_bootstrap_95ci": [round(lo, 3) if lo is not None else None,
                                          round(hi, 3) if hi is not None else None],
                "mcnemar": {"comem_only_b": b, "dense_only_c": c,
                            "both": both, "neither": neither,
                            "exact_two_sided_p": _mcnemar_exact(b, c)},
                "n_examples": len(all_diffs),
            })
        return out

    anchors = {}
    if frozen:
        pa = frozen.get("primary_anchor_comem12", {})
        for store in ("gpu_resident_deploy", "cpu_pinned_deploy",
                      "gpu_resident_coldindex", "cpu_pinned_coldindex"):
            kd = pa.get(store, {}).get("k_star")
            if kd is not None:
                anchors[f"primary_comem12_vs_dense_kstar_{store}"] = _anchor_stats(12, kd)
    anchors["reference_k12_vs_k12"] = _anchor_stats(12, 12)

    # ---- 4) success-criterion decision (report honestly; do not beautify) ----
    decision = {"note": "paperA §P0.20 Phase B — dense-retriever equal-latency."}
    prim = anchors.get("primary_comem12_vs_dense_kstar_gpu_resident_deploy") \
        or anchors.get("primary_comem12_vs_dense_kstar_cpu_pinned_deploy")
    if prim and "macro_diff_comem_minus_dense" in prim:
        d = prim["macro_diff_comem_minus_dense"]
        ci = prim["paired_bootstrap_95ci"]
        ci_pos = ci[0] is not None and ci[0] > 0
        ci_neg = ci[1] is not None and ci[1] < 0
        decision["primary_anchor_diff_comem_minus_dense"] = d
        decision["primary_anchor_95ci"] = ci
        decision["k_dense_star_deploy"] = prim["dense_k"]
        if d >= 0 and ci_pos:
            decision["verdict"] = ("POSITIVE (candidate): at CoMem(k=12)'s latency budget, "
                                   "CoMem quality >= latency-matched DENSE-RAG and the paired "
                                   "CI excludes 0 — dense retrieval's better recall does NOT "
                                   "overturn the frontier; strengthens the Phase A BM25 result.")
        elif d >= 0:
            decision["verdict"] = ("MARGINAL: CoMem >= dense-RAG at matched latency but paired "
                                   "CI includes 0 — not a stable win over dense.")
        elif ci_neg:
            decision["verdict"] = ("NEGATIVE: latency-matched DENSE-RAG beats CoMem (CI<0) — the "
                                   "dense retriever's recall lets it read fewer-but-better chunks "
                                   "and win; report as a genuine limitation, do NOT package the "
                                   "equal-latency frontier as a CoMem win.")
        else:
            decision["verdict"] = ("MIXED: dense-RAG ahead on point estimate but CI includes 0.")
    else:
        decision["verdict"] = ("INCOMPLETE: primary k_dense* not frozen (no dense-deploy "
                               "latency in the CoMem(12) +/-band, or k=12 calib missing). "
                               "See frozen_k_star.bracket for the interpolated crossing.")

    summary = {
        "run": "P0.20_phaseB_dense_equal_latency", "tol": args.tol,
        "n_examples_paired": len(valid),
        "latency_by_k": {str(k): lat_by_k[k] for k in ks_sorted},
        "frozen_k_star": frozen,
        "per_cell_quality": per_cell,
        "boundary_note": ("Phase B (frozen DENSE BGE-large-en-v1.5 retriever). CoMem arm "
                          "UNCHANGED from Phase A (iter_bm25, pre-stored h12). Dense cost "
                          "reported under deployment (offline-indexed: query-encode + flat "
                          "cosine search, PRIMARY) and cold-index (encode all passages, "
                          "SENSITIVITY == P1.9 per-query). Vector search charged on CPU "
                          "flat brute-force (== P1.9), conservative to dense. NVMe/network "
                          "store tiers + ANN indexes out of scope (stated, not fabricated)."),
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(outdir / "frontier.json", "w") as f:
        json.dump(frontier, f, indent=2)
    with open(outdir / "anchors.json", "w") as f:
        json.dump(anchors, f, indent=2)
    with open(outdir / "decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    print("=" * 74)
    print(f"[p0.20B][aggregate] paired examples={len(valid)}  tol=+/-{args.tol*100:.0f}%")
    if frozen:
        p = frozen["primary_anchor_comem12"]
        print(f"  PRIMARY (CoMem@12 budget): k_dense* deploy gpu="
              f"{p['gpu_resident_deploy'].get('k_star')} cpu={p['cpu_pinned_deploy'].get('k_star')}"
              f" | coldindex gpu={p['gpu_resident_coldindex'].get('k_star')} "
              f"cpu={p['cpu_pinned_coldindex'].get('k_star')}")
    for name, a in anchors.items():
        if "macro_diff_comem_minus_dense" in a:
            print(f"  {name}: CoMem={a['macro_comem']} dense={a['macro_dense_rag']} "
                  f"diff={a['macro_diff_comem_minus_dense']} "
                  f"CI={a['paired_bootstrap_95ci']} "
                  f"McNemar p={a['mcnemar']['exact_two_sided_p']:.3g}")
    print(f"  VERDICT: {decision.get('verdict')}")
    print("=" * 74, flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="P0.20 Phase B dense equal-latency frontier")
    ap.add_argument("--mode", required=True,
                    choices=["manifest", "sanity", "calib_latency", "quality", "aggregate"])
    ap.add_argument("--model_path", type=str, default="models/Qwen3-8b-local")
    ap.add_argument("--lora_adapter", type=str,
                    default="outputs/qcmem_distill_qwen_j12_r32_4k/final")
    ap.add_argument("--retriever_path", type=str, default="models/bge-large-en-v1.5")
    ap.add_argument("--benchmark", type=str, default="ruler",
                    choices=["ruler", "babilong", "longeval", "locomo"])
    ap.add_argument("--task", type=str, default="niah_multikey_1")
    ap.add_argument("--length", type=str, default="16k")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--iter_hop_topk", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=0)
    ap.add_argument("--dataset_name", type=str, default="RMT-team/babilong")
    ap.add_argument("--locomo_data", type=str, default="locomo/data/locomo10.json")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default="bench_results/p0_20_phaseB_dense")
    ap.add_argument("--p19_output_dir", type=str, default="bench_results/p1_9_dense_rag",
                    help="P1.9 results dir for the dense reproduction cross-check.")
    ap.add_argument("--allow_retriever_sha_mismatch", action="store_true", default=False)
    # calibration latency
    ap.add_argument("--proc_id", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n_repeat", type=int, default=20)
    ap.add_argument("--calib_offset", type=int, default=900)
    ap.add_argument("--calib_length", type=str, default="32k")
    # freeze / aggregate
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    if args.mode == "aggregate":
        run_aggregate(args)
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print("=" * 78)
    print(f"P0.20-PhaseB :: mode={args.mode} benchmark={args.benchmark} task={args.task} "
          f"length={args.length} k={args.k} shard={args.shard_index}/{args.num_shards}")
    print(f"  model={args.model_path} lora={args.lora_adapter} retriever={args.retriever_path}")
    print("=" * 78, flush=True)

    if args.mode == "manifest":
        run_manifest(args, device, dtype)
    elif args.mode == "sanity":
        run_sanity(args, device, dtype)
    elif args.mode == "calib_latency":
        run_calib_latency(args, device, dtype)
    elif args.mode == "quality":
        run_quality(args, device, dtype)


if __name__ == "__main__":
    main()
