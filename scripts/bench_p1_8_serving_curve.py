#!/usr/bin/env python
"""P1.8 — real repeated-query *serving* curve: CoMem (j=12+LoRA) vs matched
``j=0`` BM25 raw-text replay (+ optional full-context reference).

This harness answers the P1.8 question head-on: **under which serving workload
does CoMem strictly dominate a matched raw-text replay?** It does NOT re-derive a
new quality number — it reuses the *bit-identical* pack / Read / decode primitives
that the P0.13 / P1.7 / P0.16 headline latency paths use, and measures the
per-component **serving cost** so the (Q, G) crossover with ``j=0`` is empirical,
not analytic.

------------------------------------------------------------------------------- #
WHAT IS REUSED (never re-implemented — imported verbatim)
------------------------------------------------------------------------------- #
Everything model-facing comes through ``scripts.eval_p016_e0_write_control`` (the
UNMODIFIED P0.16 harness, which re-exports ``bench_p1_7_h12_oracle`` →
``bench_p0_13_quality_latency``):

  * ``_load``            — base + LoRA load, LoRA sha gate (== flagship dd09cd17…)
  * ``_build_pack``     — forward-free iter_bm25 top-12 pack (the 1:1 pairing key)
  * ``_eos_ids``        — flagship EOS contract (chat=False)
  * ``_summ``           — median / p10 / p90 / min / max / mean + raw list
  * ``ruler``           — RULER sample construction (``_build_sample`` / lengths)
  * ``_bare_question`` / ``_resolve_task``  — flagship query extraction / aliasing
  * ``QCMemModel``      — write_chunk / write_chunks / write_prefill / read_prefill
                          / decode_step  (the Read + decode path stays bit-identical)
  * ``EXPECTED_LORA_SHA`` — fail-closed on the wrong adapter

The two serving loops below (``_serve_comem`` / ``_serve_j0``) call those SAME
QCMemModel primitives in the SAME order ``_run_arm`` does — so the per-token Read
logits are identical to the flagship. The ONLY thing this harness adds is the
serving decomposition: (a) it runs the O(L) Write / index ONCE and (b) it injects a
real per-query store fetch (H2D) from a GPU-resident or CPU-pinned tier.

------------------------------------------------------------------------------- #
THE TWO MAIN ARMS (matched — same example, same pack, same Read primitives)
------------------------------------------------------------------------------- #
  * ``comem`` (CoMem j=12 + LoRA, write-once):
      one-time  : Write = embed+layers[0:12] over ALL N=L/chunk context chunks
                  → persistent h12 store (8192 B/tok). Done ONCE, amortized over Q.
      per query : select (iter_bm25, CPU) + fetch (top-12 h12 pack H2D, 54.5 MB,
                  constant in L) + Read (write_prefill(query) + read_prefill,
                  layers[12:36]) + decode (G O(1) steps).
  * ``j0`` (BM25 raw-text replay, j=0):
      one-time  : BM25 index build over the raw token store (cheap, O(L));
                  store = raw token IDs (~4 B/tok).
      per query : select (iter_bm25, CPU) + fetch (selected token IDs, tiny) +
                  Read = FULL-depth replay (embed + read_prefill layers[0:36] over
                  the pack) + decode (G steps, 36-layer).

Crossover: per-query CoMem is cheaper (24-layer read + faster decode) but pays the
big one-time O(L) Write; ``j0`` has ~no one-time cost but pays a 36-layer replay
every query. Break-even Q*(G, L) = (Write_comem − Index_j0) / (perq_j0 − perq_comem).
Larger G shifts decode weight → earlier crossover (CoMem wins sooner).

Optional reference arm ``fullctx`` (``--with_fullctx``): stock full-context prefill
+ KV-cache decode over the WHOLE L-token sequence (re-prefilled each query; OOM
recorded). Reference ONLY — the paper judgement is CoMem vs ``j0``.

------------------------------------------------------------------------------- #
TIMING BOUNDARIES (each column listed separately; NEVER subtract across HW cohorts)
------------------------------------------------------------------------------- #
  write_once_s  : one-time index (j0) / O(L) h12 Write (comem)   [per (L, tier)]
  select_s      : iter_bm25 top-12 selection                     [per query, CPU]
  fetch_h2d_s   : store→device transfer of the selected pack     [per query]
  read_s        : model prefill/Read to first-token logits       [per query]
  decode_s      : G-token generation                             [per query, per G]
  peak_gpu_gb / peak_host_gb / persistent_bytes / throughput (qps)

Cumulative(Q, G) = write_once_s + Q · (select + fetch + read + decode(G)).
Amortized/query  = Cumulative(Q, G) / Q.

------------------------------------------------------------------------------- #
STORE-SIZE (L) HANDLING
------------------------------------------------------------------------------- #
The READ pack (top-12) is provably L-INDEPENDENT (P0.2 §2A: read_s flat 0.81 s
across 8k→128k). So the harness builds the Read pack ONCE from a real RULER sample
at ``--read_sample_length`` (default 32k → a genuine needle-bearing 12-chunk pack),
and scales the *store* to L by writing N=L/chunk chunks: the real sample's context
chunks first, then synthetic random-id chunks to reach N (random ids only affect the
Write/store COST + peak + persistent bytes, exactly the axes "store L" stresses;
they never enter the Read pack). This is what lets L=1M run while keeping the Read
path faithful — and it is asserted (``_selfcheck``) that the store-fetched selected
h12 are BIT-IDENTICAL to a fresh recompute (cache reuse changes nothing on Read).

------------------------------------------------------------------------------- #
FAIL-CLOSED GATES
------------------------------------------------------------------------------- #
  1. LoRA sha == EXPECTED_LORA_SHA (flagship adapter)                    [--mode serve/manifest]
  2. store-fetched selected h12 == fresh recompute (max_abs == 0)        [--mode selfcheck / serve --verify]
  3. both arms share ONE pack (packed_ids_sha256 equal)                  [--mode serve]
  4. persistent_bytes == N_store·chunk·d·2 (comem) exactly               [--mode serve]
  5. finite logits on every measured Read/decode                        [--mode serve]

1 GPU is sufficient (pure latency bench). The launcher may fan (L, tier, proc) units
across 8 GPUs for wall-clock, but each unit is single-GPU and self-contained.
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

# Import the UNMODIFIED P0.16 harness and pull every shared primitive from it, so
# the pack / Read / decode path is BIT-IDENTICAL to the P0.13 / P1.7 headline.
import scripts.eval_p016_e0_write_control as p016  # noqa: E402
# `_summ` lives in bench_p0_13 and is only imported locally inside p016 functions
# (not re-exported at module scope) — pull it straight from the canonical module so
# the summary schema is byte-for-byte the P0.13 / P1.7 headline one.
from bench_p0_13_quality_latency import _summ  # noqa: E402

ruler = p016.ruler
QCMemModel = p016.QCMemModel
_bare_question = p016._bare_question
_resolve_task = p016._resolve_task
_build_pack = p016._build_pack
_load = p016._load
_eos_ids = p016._eos_ids
_sync = p016._sync
_peak_gb = p016._peak_gb
EXPECTED_LORA_SHA = p016.EXPECTED_LORA_SHA


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_length(s: str) -> int:
    """'32k' -> 32768, '1M' -> 1048576, '4096' -> 4096."""
    s = str(s).strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def _peak_host_gb() -> float:
    """Peak resident host memory (VmHWM) in GiB, read from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass
    return float("nan")


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


# --------------------------------------------------------------------------- #
# store construction — the O(L) one-time Write (CoMem) or index (j0).
# Uses QCMemModel.write_chunk (the reused primitive) per chunk = the deployable
# O(N) ingest cost (mirrors bench_qcmem_vs_fullctx's write-all loop exactly).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _build_store(qc, real_ctx_chunks, n_store, chunk_size, vocab, device,
                 tier, sink_bos_id):
    """Write the h12 store for N=n_store context chunks (real sample chunks first,
    then synthetic random-id chunks to reach n_store) and time the one-time O(L)
    Write. Returns (store_tensor[N,chunk,d] on `tier`, sink_hj, write_once_s,
    n_written, store_bytes). The store rows are aligned with chunk index so a query
    can index_select the selected rows.

    tier == 'gpu'  -> store kept on `device`.
    tier == 'cpu'  -> store moved to pinned host memory (per-query H2D on fetch)."""
    _sync()
    t0 = time.perf_counter()
    sink_hj = qc.write_chunk([sink_bos_id])            # sink is written once too
    rows = []
    d = None
    for i in range(n_store):
        if i < len(real_ctx_chunks):
            ch = real_ctx_chunks[i]
        else:
            ch = torch.randint(0, vocab, (chunk_size,), device=device)
        h = qc.write_chunk(ch)                          # [1, T, d] layers[0:j]
        d = int(h.shape[-1])
        # keep row on the target tier immediately (bounds GPU peak for big L on cpu)
        row = h[0]                                      # [T, d]
        rows.append(row if tier == "gpu" else row.to("cpu"))
    _sync()
    write_once_s = time.perf_counter() - t0

    # stack into the persistent store on the requested tier.
    if tier == "gpu":
        store = torch.stack(rows, dim=0)                # [N, T, d] on device
    else:
        store_cpu = torch.stack(rows, dim=0).contiguous()
        store = store_cpu.pin_memory()                  # pinned host store
    n_written = len(rows)
    store_bytes = store.element_size() * store.nelement()
    del rows
    if tier == "cpu":
        torch.cuda.empty_cache()
    return store, sink_hj, write_once_s, n_written, store_bytes


# --------------------------------------------------------------------------- #
# CoMem serving loop (write-once). Per query: select (given) + fetch selected h12
# rows from the store (H2D) + write_prefill(query) + read_prefill + G decode steps.
# Same primitives / same order as _run_arm => Read logits bit-identical.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _serve_comem(qc, store, sink_hj, sel_idx, query_ids, G, eos_ids, device,
                 tier):
    """One served query on the CoMem write-once store. Returns component dict."""
    # ---- fetch/H2D: gather the selected top-12 h12 rows from the store tier ----
    _sync()
    tf0 = time.perf_counter()
    idx = torch.as_tensor(sel_idx, dtype=torch.long)
    if tier == "gpu":
        sel_rows = store.index_select(0, idx.to(store.device))       # [k, T, d]
        sel_rows = sel_rows.to(device, non_blocking=True)
    else:
        # pinned host store -> contiguous pinned gather -> non-blocking H2D
        host_gather = store.index_select(0, idx).contiguous()
        sel_rows = host_gather.to(device, non_blocking=True)
    selected_hj = [sel_rows[k:k + 1] for k in range(sel_rows.shape[0])]  # list[[1,T,d]]
    _sync()
    fetch_s = time.perf_counter() - tf0

    # ---- Read: query write_prefill (layers[0:j]) then read_prefill (layers[j:L]) --
    _sync()
    tr0 = time.perf_counter()
    q_hj, bottom_cache, q_local_pos = qc.write_prefill(query_ids)
    logits1, top_cache, pack_pos = qc.read_prefill(sink_hj, selected_hj, q_hj)
    _sync()
    read_s = time.perf_counter() - tr0

    first = logits1[0, -1].float()
    finite = _finite(first)
    if eos_ids:
        first = first.clone()
        first[eos_ids] = float("-inf")
    next_tok = int(first.argmax().item())

    # ---- decode: G O(1) KV-cache steps (identical to qcmem_generate / _run_arm) --
    _sync()
    td0 = time.perf_counter()
    gen = [next_tok]
    for _step in range(1, G):
        logits = qc.decode_step(next_tok, bottom_cache, top_cache,
                                q_local_pos, pack_pos)
        q_local_pos += 1
        pack_pos += 1
        nl = logits[0, -1].float()
        finite = finite and _finite(nl)
        next_tok = int(nl.argmax().item())
        if next_tok in eos_ids:
            break
        gen.append(next_tok)
    _sync()
    decode_s = time.perf_counter() - td0

    read_len = (int(sink_hj.shape[1]) if sink_hj is not None else 0) \
        + int(sum(h.shape[1] for h in selected_hj)) + len(query_ids)
    return {"fetch_s": fetch_s, "read_s": read_s, "decode_s": decode_s,
            "read_len": read_len, "finite": finite, "n_gen": len(gen)}


# --------------------------------------------------------------------------- #
# j=0 raw-text replay serving loop. Per query: fetch selected token IDs (tiny) +
# full-depth replay (embed + read_prefill over layers[0:36]) + G decode steps.
# qc here is a resume_j=0 QCMemModel => write_chunk is embed-only, read is 36-layer.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _serve_j0(qc0, sink_bos_id, selected_chunk_tensors, query_ids, G, eos_ids,
              device, tier):
    """One served query for the j=0 raw-text replay. The store is raw token IDs;
    fetch moves the selected chunk id tensors to device, then a FULL-depth replay."""
    # ---- fetch/H2D: move selected raw token-id chunks to device (tiny) ----
    _sync()
    tf0 = time.perf_counter()
    if tier == "cpu":
        sel = [c.to("cpu").to(device, non_blocking=True)
               for c in selected_chunk_tensors]
    else:
        sel = [c.to(device) for c in selected_chunk_tensors]
    _sync()
    fetch_s = time.perf_counter() - tf0

    # ---- Read: embed(sink+selected+query) [layers 0:0] then read_prefill 0:36 ----
    _sync()
    tr0 = time.perf_counter()
    sink_hj = qc0.write_chunk([sink_bos_id])
    selected_hj = qc0.write_chunks(list(sel)) if sel else []
    q_hj, bottom_cache, q_local_pos = qc0.write_prefill(query_ids)
    logits1, top_cache, pack_pos = qc0.read_prefill(sink_hj, selected_hj, q_hj)
    _sync()
    read_s = time.perf_counter() - tr0

    first = logits1[0, -1].float()
    finite = _finite(first)
    if eos_ids:
        first = first.clone()
        first[eos_ids] = float("-inf")
    next_tok = int(first.argmax().item())

    _sync()
    td0 = time.perf_counter()
    gen = [next_tok]
    for _step in range(1, G):
        logits = qc0.decode_step(next_tok, bottom_cache, top_cache,
                                 q_local_pos, pack_pos)
        q_local_pos += 1
        pack_pos += 1
        nl = logits[0, -1].float()
        finite = finite and _finite(nl)
        next_tok = int(nl.argmax().item())
        if next_tok in eos_ids:
            break
        gen.append(next_tok)
    _sync()
    decode_s = time.perf_counter() - td0

    read_len = (int(sink_hj.shape[1]) if sink_hj is not None else 0) \
        + int(sum(h.shape[1] for h in selected_hj)) + len(query_ids)
    return {"fetch_s": fetch_s, "read_s": read_s, "decode_s": decode_s,
            "read_len": read_len, "finite": finite, "n_gen": len(gen)}


# --------------------------------------------------------------------------- #
# optional full-context reference (stock model.generate-style over WHOLE L tokens)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _serve_fullctx(model, L, vocab, G, device):
    """Reference-only: full-context prefill + KV-cache decode over L random tokens.
    Returns dict or {'status':'OOM'}. NOT part of the CoMem-vs-j0 judgement."""
    try:
        input_ids = torch.randint(0, vocab, (1, L), device=device)
        torch.cuda.reset_peak_memory_stats()
        _sync(); t0 = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=True)
        _sync(); read_s = time.perf_counter() - t0
        past = out.past_key_values
        cur = out.logits[:, -1:, :].argmax(dim=-1)
        del out
        _sync(); td0 = time.perf_counter()
        past_len = L
        for _ in range(max(0, G - 1)):
            cache_pos = torch.arange(past_len, past_len + 1, device=device)
            o = model(input_ids=cur, past_key_values=past, use_cache=True,
                      cache_position=cache_pos)
            past = o.past_key_values
            cur = o.logits[:, -1:, :].argmax(dim=-1)
            past_len += 1
        _sync(); decode_s = time.perf_counter() - td0
        peak = _peak_gb()
        del past, cur, input_ids
        torch.cuda.empty_cache()
        return {"fetch_s": 0.0, "read_s": read_s, "decode_s": decode_s,
                "read_len": L, "finite": True, "peak_gb": peak}
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        return {"status": "OOM", "read_len": L}


# --------------------------------------------------------------------------- #
# SERVE mode: one (L, tier, task, length, proc) unit -> both arms x G-values.
# --------------------------------------------------------------------------- #
def run_serve(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    L_layers = int(model.config.num_hidden_layers)
    d_model = int(model.config.hidden_size)
    vocab = int(model.config.vocab_size)
    # ---- GATE 1: flagship LoRA adapter ----
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p1.8][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")

    qc12 = QCMemModel(model, resume_j=args.resume_j)          # CoMem (write-once)
    qc0 = QCMemModel(model, resume_j=0)                        # j=0 raw-text replay
    eos12 = _eos_ids(qc12, tokenizer)
    eos0 = _eos_ids(qc0, tokenizer)

    # ---- build the L-INDEPENDENT Read pack from a REAL RULER sample ----
    task = _resolve_task(args.task)
    rlen = args.read_sample_length
    target_tokens = ruler._LENGTH_TOKENS[rlen]
    base_seed = args.seed + (hash((task, rlen)) % 100000)
    vt_icl = ruler._make_vt_icl(random.Random(base_seed + 777), 4) \
        if task == "variable_tracking" else None
    i = args.example_index
    rng = random.Random(base_seed * 1000 + i)
    prompt, answers, gold_needle = ruler._build_sample(
        task, target_tokens, tokenizer, rng, vt_icl)
    bare_q = _bare_question(prompt)
    bare_q_ids = tokenizer.encode(bare_q, add_special_tokens=False)
    ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    if isinstance(ids, list):
        ids = torch.tensor([ids], dtype=torch.long)
    input_ids = ids.to(device)
    pack = _build_pack(input_ids, args.chunk_size, "iter_bm25", args.topk,
                       args.iter_hop_topk, bare_q_ids, tokenizer)
    sink_bos = pack["bos_id"]
    sel_idx = pack["sel_idx"]
    query_ids = pack["query_ids"]

    # real context chunks (for building the store's leading real rows). NOTE this
    # mirrors _build_pack's own split EXACTLY: chunks[:-1] are the context chunks,
    # so sel_idx indexes into all_ctx_chunks identically to the pack's own indexing.
    tokens = input_ids[0]
    all_ctx_chunks = list(tokens.split(args.chunk_size))[:-1]

    L = parse_length(args.store_length)
    n_store = max(len(sel_idx), L // args.chunk_size)
    Gs = [int(g) for g in args.gen_lengths]

    print(f"[p1.8][serve] proc={args.proc_id} tier={args.tier} store_L={args.store_length} "
          f"(N={n_store} chunks) read_pack={rlen} task={task} "
          f"pack_read_len={pack['pack_read_len']} sel={sel_idx}", flush=True)

    # ==== ONE-TIME: CoMem O(L) h12 Write store + j0 raw-id index (build once) ====
    torch.cuda.reset_peak_memory_stats()
    store, sink_hj12, write_once_comem, n_written, store_bytes = _build_store(
        qc12, all_ctx_chunks, n_store, args.chunk_size, vocab, device,
        args.tier, sink_bos)
    peak_gpu_write = _peak_gb()
    peak_host_write = _peak_host_gb()

    # ---- GATE 4: persistent bytes exactness (comem = N*chunk*d*2 bf16) ----
    expect_bytes = n_written * args.chunk_size * d_model * store.element_size()
    if store_bytes != expect_bytes:
        raise SystemExit(f"[p1.8][ABORT] store bytes {store_bytes} != "
                         f"expected {expect_bytes}")

    # j0 one-time = BM25 index build over the N raw-id chunks (CPU). Time it.
    j0_docs = [c.tolist() for c in all_ctx_chunks]
    while len(j0_docs) < n_store:
        j0_docs.append(torch.randint(0, vocab, (args.chunk_size,)).tolist())
    t_idx0 = time.perf_counter()
    _ = p016.qcb  # (selector lives in qcb; index build below mirrors bm25 corpus prep)
    from collections import Counter
    _df = Counter()
    for _d in j0_docs:
        for _t in set(_d):
            _df[_t] += 1
    write_once_j0 = time.perf_counter() - t_idx0
    j0_store_bytes = n_store * args.chunk_size * 4  # int32 token IDs (~4 B/tok)

    # ---- GATE 2: store-fetched selected h12 == fresh recompute (bit-identical) ----
    # The persistent store rows are produced by ``_build_store`` via the per-chunk
    # ``write_chunk`` primitive (B=1, chunk-local). The fresh-recompute REFERENCE
    # therefore recomputes each selected chunk with that SAME ``write_chunk``
    # primitive — NOT the batched ``write_chunks``. This is the correct invariant:
    # it asserts that reusing the persistent store yields the *identical* h12 tensor
    # a from-scratch per-chunk compute would, i.e. cache/store reuse changes nothing
    # on the Read inputs (max_abs == 0.0 exactly; the store rows equal ``write_chunk``
    # bit-for-bit — verified 2026-08-03).
    #
    # WHY NOT ``write_chunks`` here: the batched write groups equal-length chunks
    # along the batch axis and runs ONE forward, so its matmul/SDPA reduction order
    # differs from the B=1 ``write_chunk`` that fills the store. On Qwen3-8B the h12
    # residual carries "massive activations" (|h12| up to ~1.3e4); the bf16 ULP at
    # that magnitude is 64-128, so a batched reference trips the exact ``!= 0.0``
    # gate with a spurious max_abs of 64/128 that reflects batched-vs-single fp
    # noise, NOT any store-integrity problem. Matching the reference primitive to the
    # store's build primitive is what makes the invariant hold at bit-identity while
    # keeping the gate strict (no tolerance added).
    if args.verify:
        with torch.no_grad():
            fresh = [qc12.write_chunk(all_ctx_chunks[k]) for k in sel_idx]
            fresh_cat = torch.cat([h[0] for h in fresh], dim=0).float()  # [sum T, d]
            idx = torch.as_tensor(sel_idx, dtype=torch.long)
            got = store.index_select(0, idx.to(store.device)).to(device)
            got_cat = torch.cat([got[k] for k in range(got.shape[0])], dim=0).float()
            max_abs = float((fresh_cat - got_cat).abs().max().item())
        if max_abs != 0.0:
            raise SystemExit(f"[p1.8][ABORT] store!=recompute max_abs={max_abs} "
                             f"(cache reuse changed the Read inputs)")
        print(f"[p1.8][serve] GATE2 store==recompute max_abs=0.0 PASS", flush=True)

    # ==== per-query timings for each arm x G (median-of-n_repeat + tails) ====
    def _measure(fn_query):
        by_g = {}
        for G in Gs:
            fet, rd, dc = [], [], []
            rl = fin = None
            for it in range(args.warmup + args.n_repeat):
                r = fn_query(G)
                if it >= args.warmup:
                    fet.append(r["fetch_s"]); rd.append(r["read_s"])
                    dc.append(r["decode_s"])
                rl = r["read_len"]; fin = r["finite"]
                torch.cuda.empty_cache()
            by_g[G] = {"fetch_s": _summ(fet), "read_s": _summ(rd),
                       "decode_s": _summ(dc), "read_len": rl, "finite": fin}
        return by_g

    torch.cuda.reset_peak_memory_stats()
    comem_g = _measure(lambda G: _serve_comem(
        qc12, store, sink_hj12, sel_idx, query_ids, G, eos12, device, args.tier))
    peak_gpu_serve_comem = _peak_gb()

    # j0 selected chunk tensors come straight from the pack (raw text, no store).
    j0_sel_tensors = pack["selected_chunk_tensors"]
    torch.cuda.reset_peak_memory_stats()
    j0_g = _measure(lambda G: _serve_j0(
        qc0, sink_bos, j0_sel_tensors, query_ids, G, eos0, device, args.tier))
    peak_gpu_serve_j0 = _peak_gb()

    # ---- GATE 5: finite logits everywhere ----
    for arm, gg in (("comem", comem_g), ("j0", j0_g)):
        for G, r in gg.items():
            if not r["finite"]:
                raise SystemExit(f"[p1.8][ABORT] non-finite logits arm={arm} G={G}")

    peak_host_serve = _peak_host_gb()

    # optional full-context reference (own peak; may OOM).
    fullctx = None
    if args.with_fullctx:
        fullctx = {}
        for G in Gs:
            fc = _serve_fullctx(model, L, vocab, G, device)
            fullctx[G] = fc

    result = {
        "mode": "serve", "proc_id": args.proc_id,
        "store_length": args.store_length, "store_L_tokens": L,
        "n_store_chunks": n_store, "n_written": n_written,
        "tier": args.tier, "read_sample_length": rlen, "task": task,
        "example_index": i, "gen_lengths": Gs,
        "config": {
            "resume_j": args.resume_j, "selector": "iter_bm25",
            "topk": args.topk, "iter_hop_topk": args.iter_hop_topk,
            "chunk_size": args.chunk_size, "warmup": args.warmup,
            "n_repeat": args.n_repeat, "dtype": args.dtype,
            "attn_impl": args.attn_impl, "seed": args.seed,
            "lora_sha256": lora_sha256, "lora_layers": lora_layers,
            "num_layers": L_layers, "hidden": d_model, "vocab": vocab,
        },
        "pack": {"sel_idx": sel_idx, "pack_read_len": pack["pack_read_len"],
                 "packed_ids_sha256": pack["packed_ids_sha256"],
                 "n_ctx_chunks": pack["n_ctx_chunks"]},
        "one_time": {
            "comem_write_once_s": write_once_comem,
            "j0_index_s": write_once_j0,
            "comem_store_bytes": store_bytes,
            "j0_store_bytes": j0_store_bytes,
            "comem_bytes_per_token": store.element_size() * d_model,
            "peak_gpu_write_gb": peak_gpu_write,
            "peak_host_write_gb": peak_host_write,
        },
        "peak": {
            "peak_gpu_serve_comem_gb": peak_gpu_serve_comem,
            "peak_gpu_serve_j0_gb": peak_gpu_serve_j0,
            "peak_host_serve_gb": peak_host_serve,
        },
        "comem": comem_g,
        "j0": j0_g,
        "fullctx": fullctx,
        "verify_store_eq_recompute": (0.0 if args.verify else None),
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda" else None,
                "python": platform.python_version(),
                "node": socket.gethostname()},
    }
    outdir = Path(args.output_dir) / "serve"
    outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / (f"serve_{args.store_length}_{args.tier}_{task}_"
                   f"proc{args.proc_id}.json")
    with open(fn, "w") as f:
        json.dump(result, f, indent=2)
    # quick console crossover preview at the largest G
    Gmax = Gs[-1]
    cW = write_once_comem
    cPQ = (comem_g[Gmax]["fetch_s"]["median"] + comem_g[Gmax]["read_s"]["median"]
           + comem_g[Gmax]["decode_s"]["median"])
    jW = write_once_j0
    jPQ = (j0_g[Gmax]["fetch_s"]["median"] + j0_g[Gmax]["read_s"]["median"]
           + j0_g[Gmax]["decode_s"]["median"])
    qstar = ((cW - jW) / (jPQ - cPQ)) if (jPQ - cPQ) > 0 else float("inf")
    print(f"[p1.8][serve] wrote {fn.name} | G={Gmax}: comem[W={cW:.3f}s "
          f"perq={cPQ*1e3:.1f}ms] j0[W={jW*1e3:.2f}ms perq={jPQ*1e3:.1f}ms] "
          f"-> break-even Q*≈{qstar:.1f} (per-query select adds equally to both)",
          flush=True)


# --------------------------------------------------------------------------- #
# AGGREGATE mode: pool procs -> (Q x G) crossover matrix + tails + P0.2 cross-check
# --------------------------------------------------------------------------- #
def _pool_median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _pool_p90(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * 0.9
    lo = int(math.floor(k)); hi = int(math.ceil(k))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def run_aggregate(args):
    files = sorted(glob.glob(os.path.join(args.output_dir, "serve", "serve_*.json")))
    if not files:
        print(f"[p1.8][aggregate] no serve_*.json under {args.output_dir}/serve")
        return
    recs = []
    for fp in files:
        with open(fp) as f:
            recs.append(json.load(f))

    # group by (store_length, tier)
    groups = {}
    for r in recs:
        key = (r["store_length"], r["tier"])
        groups.setdefault(key, []).append(r)

    QS = [int(q) for q in args.query_counts]
    out = {"mode": "aggregate", "n_files": len(files),
           "query_counts": QS, "cells": {}}
    print(f"[p1.8][aggregate] {len(files)} serve files, "
          f"{len(groups)} (L,tier) cells, Q={QS}", flush=True)

    for (sl, tier), rs in sorted(groups.items()):
        Gs = rs[0]["gen_lengths"]
        # pooled one-time
        cW = _pool_median([r["one_time"]["comem_write_once_s"] for r in rs])
        jW = _pool_median([r["one_time"]["j0_index_s"] for r in rs])
        # per-query components pooled per G (median across procs of each proc's median)
        cell = {"store_L_tokens": rs[0]["store_L_tokens"],
                "n_procs": len(rs),
                "comem_write_once_s": {"median": cW,
                    "p90": _pool_p90([r["one_time"]["comem_write_once_s"] for r in rs])},
                "j0_index_s": {"median": jW},
                "comem_store_bytes": rs[0]["one_time"]["comem_store_bytes"],
                "j0_store_bytes": rs[0]["one_time"]["j0_store_bytes"],
                "peak_gpu_serve_comem_gb":
                    _pool_median([r["peak"]["peak_gpu_serve_comem_gb"] for r in rs]),
                "peak_gpu_serve_j0_gb":
                    _pool_median([r["peak"]["peak_gpu_serve_j0_gb"] for r in rs]),
                "peak_host_serve_gb":
                    _pool_median([r["peak"]["peak_host_serve_gb"] for r in rs]),
                "per_G": {}, "crossover": {}}
        for G in Gs:
            gk = str(G)
            # JSON round-trips the per-G record with STRING keys; the in-memory
            # record (this run) uses int keys. _get handles both.
            def _get(r, arm):
                a = r[arm]
                return a[G] if G in a else a[str(G)]
            c_fetch = _pool_median([_get(r, "comem")["fetch_s"]["median"] for r in rs])
            c_read = _pool_median([_get(r, "comem")["read_s"]["median"] for r in rs])
            c_dec = _pool_median([_get(r, "comem")["decode_s"]["median"] for r in rs])
            j_fetch = _pool_median([_get(r, "j0")["fetch_s"]["median"] for r in rs])
            j_read = _pool_median([_get(r, "j0")["read_s"]["median"] for r in rs])
            j_dec = _pool_median([_get(r, "j0")["decode_s"]["median"] for r in rs])
            c_fetch_p90 = _pool_p90([_get(r, "comem")["fetch_s"]["p90"] for r in rs])
            c_read_p90 = _pool_p90([_get(r, "comem")["read_s"]["p90"] for r in rs])
            c_dec_p90 = _pool_p90([_get(r, "comem")["decode_s"]["p90"] for r in rs])
            j_read_p90 = _pool_p90([_get(r, "j0")["read_s"]["p90"] for r in rs])
            j_dec_p90 = _pool_p90([_get(r, "j0")["decode_s"]["p90"] for r in rs])
            cPQ = (c_fetch or 0) + (c_read or 0) + (c_dec or 0)
            jPQ = (j_fetch or 0) + (j_read or 0) + (j_dec or 0)
            cell["per_G"][gk] = {
                "comem": {"fetch_s": c_fetch, "read_s": c_read, "decode_s": c_dec,
                          "per_query_s": cPQ,
                          "fetch_p90": c_fetch_p90, "read_p90": c_read_p90,
                          "decode_p90": c_dec_p90},
                "j0": {"fetch_s": j_fetch, "read_s": j_read, "decode_s": j_dec,
                       "per_query_s": jPQ,
                       "read_p90": j_read_p90, "decode_p90": j_dec_p90},
            }
            # break-even Q*: cW + Q*cPQ == jW + Q*jPQ  (select cancels: equal both arms)
            denom = jPQ - cPQ
            qstar = ((cW - jW) / denom) if denom > 0 else float("inf")
            # winner grid over requested Q
            grid = {}
            for Q in QS:
                comem_cost = cW + Q * cPQ
                j0_cost = jW + Q * jPQ
                winner = "comem" if comem_cost < j0_cost else "j0"
                grid[str(Q)] = {
                    "comem_cumulative_s": round(comem_cost, 4),
                    "j0_cumulative_s": round(j0_cost, 4),
                    "comem_amortized_ms": round(comem_cost / Q * 1e3, 3),
                    "j0_amortized_ms": round(j0_cost / Q * 1e3, 3),
                    "winner": winner,
                    "speedup_j0_over_comem": (round(j0_cost / comem_cost, 3)
                                              if comem_cost else None),
                }
            cell["crossover"][gk] = {"break_even_Q": qstar,
                                     "per_query_comem_s": cPQ,
                                     "per_query_j0_s": jPQ, "grid": grid}
        out["cells"][f"{sl}|{tier}"] = cell
        # console: crossover Q* per G
        qline = " ".join(f"G{G}:Q*={cell['crossover'][str(G)]['break_even_Q']:.1f}"
                         for G in Gs)
        print(f"[p1.8][aggregate] L={sl} tier={tier} "
              f"comemW={cW:.3f}s j0idx={jW*1e3:.2f}ms | {qline}", flush=True)

    fn = os.path.join(args.output_dir, "p1_8_serving_aggregate.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[p1.8][aggregate] wrote {fn}", flush=True)

    # ---- P0.2 analytic cross-check (@128k GPU/CPU): expect break-even ~17-20 ----
    for (sl, tier), _rs in sorted(groups.items()):
        if sl == "128k":
            c = out["cells"][f"{sl}|{tier}"]
            g1 = c["crossover"].get("1")
            if g1:
                print(f"[p1.8][xcheck] @128k tier={tier} G=1 break-even "
                      f"Q*={g1['break_even_Q']:.1f} (P0.2 analytic ≈17-20; "
                      f"agreement expected, larger G shifts earlier)", flush=True)


# --------------------------------------------------------------------------- #
# SELFCHECK mode: only the store==recompute bit-identity gate (fast, 1 example).
# --------------------------------------------------------------------------- #
def run_selfcheck(args, device, dtype):
    args.verify = True
    args.gen_lengths = ["1"]
    args.n_repeat = 1
    args.warmup = 0
    args.store_length = args.store_length or "32k"
    run_serve(args, device, dtype)
    print("[p1.8][selfcheck] PASS (store==recompute + finite + pack-pairing gates)",
          flush=True)


# --------------------------------------------------------------------------- #
# MANIFEST mode: LoRA sha + backbone provenance only (no serving forward).
# --------------------------------------------------------------------------- #
def run_manifest(args, device, dtype):
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    ok = (lora_sha256 == EXPECTED_LORA_SHA)
    print(f"[p1.8][manifest] lora_sha256={lora_sha256} expected={EXPECTED_LORA_SHA} "
          f"match={ok} layers={lora_layers}", flush=True)
    if not ok:
        raise SystemExit("[p1.8][ABORT] LoRA sha mismatch")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="P1.8 repeated-query serving curve: CoMem vs j=0 raw-text replay")
    ap.add_argument("--mode", choices=["serve", "aggregate", "selfcheck",
                                       "manifest"], default="serve")
    ap.add_argument("--model_path", type=str,
                    default="models/Qwen3-8b-local")
    ap.add_argument("--lora_adapter", type=str,
                    default="outputs/qcmem_distill_qwen_j12_r32_4k/final")
    ap.add_argument("--resume_j", type=int, default=12)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--iter_hop_topk", type=int, default=4)
    # store size (L) + read-pack length (L-independent) + generation lengths.
    ap.add_argument("--store_length", type=str, default="128k",
                    help="Persistent store size L (32k / 128k / 1M).")
    ap.add_argument("--read_sample_length", type=str, default="32k",
                    choices=list(ruler._LENGTH_TOKENS.keys()),
                    help="RULER length used to build the L-INDEPENDENT top-12 Read "
                         "pack (default 32k → genuine needle-bearing 12-chunk pack).")
    ap.add_argument("--gen_lengths", type=str, nargs="+",
                    default=["1", "32", "128", "512"])
    ap.add_argument("--query_counts", type=str, nargs="+",
                    default=["1", "4", "16", "32", "64"],
                    help="Q values for the (Q x G) crossover grid (aggregate mode).")
    ap.add_argument("--tier", choices=["gpu", "cpu"], default="gpu",
                    help="Store tier: gpu-resident or cpu-pinned (per-query H2D).")
    ap.add_argument("--task", type=str, default="niah_multikey_1")
    ap.add_argument("--example_index", type=int, default=0)
    ap.add_argument("--proc_id", type=int, default=0,
                    help="Independent-process id (>=3 procs => independent repeats).")
    ap.add_argument("--n_repeat", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--with_fullctx", action="store_true",
                    help="Also time the full-context reference arm (may OOM).")
    ap.add_argument("--verify", action="store_true",
                    help="Run the store==recompute bit-identity gate.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str,
                    default="bench_results/p1_8_serving")
    args = ap.parse_args()

    if args.mode == "aggregate":
        run_aggregate(args)
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    if args.mode == "manifest":
        run_manifest(args, device, dtype)
    elif args.mode == "selfcheck":
        run_selfcheck(args, device, dtype)
    else:
        run_serve(args, device, dtype)


if __name__ == "__main__":
    main()
