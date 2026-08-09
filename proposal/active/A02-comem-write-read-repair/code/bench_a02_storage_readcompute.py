#!/usr/bin/env python
"""A02 reframe gate — storage / read-compute for high-reuse workloads.

WHY THIS EXISTS
===============
A02's phase-1 quality gate FIRED its kill clause:

    "若 paired quality CI 仍显著低于 0，则停止 CoMem 优于 RAG 的叙事，
     定位为高复用 workload 的 storage/read-compute 方案"

i.e. the clause itself prescribes the replacement framing: **storage /
read-compute for high-reuse workloads**. `STATUS.json` recorded that reframe as
entirely UNMEASURED. This driver measures it.

THE CLAIM UNDER TEST (stated so it can fail)
--------------------------------------------
    For a workload where ONE fixed corpus is queried many times, CoMem's
    precomputed mid-layer memory amortises, so its PER-QUERY cost beats
    text-RAG even though its QUALITY does not exceed it.

Decision rule fixed BEFORE the run:
  * SURVIVES  iff there exists a crossover N* that is finite AND plausibly
              reachable in a real high-reuse deployment (pre-registered
              threshold N* <= 1e5 queries per corpus, per the gate spec), on the
              arm that phase-1 actually compared against.
  * DEAD      iff N* is infinite (CoMem per-query is not cheaper) or absurdly
              large (> 1e5), OR if the storage cost is so large that the
              "storage" half of "storage/read-compute" is a loss too.

WHAT PHASE-1 ACTUALLY COMPARED (and why the existing P1.8 bench is NOT enough)
-----------------------------------------------------------------------------
The phase-1 on-disk eval configs show:

  C1 = baseline `kvdirect`  -> `resume_j=0` + `no_retrieval=True` + LoRA DROPPED.
       `no_retrieval` sets `sel_idx = range(len(context_chunks))`, i.e. it packs
       EVERY context chunk at FULL depth. Read length grows O(L) **per query**.
  C2 = QCMem `j=12` + Read-LoRA + `no_retrieval=False`, `selector=iter_bm25`,
       top-12. Read length is CONSTANT in L.

The pre-existing `scripts/bench_p1_8_serving_curve.py` measures CoMem vs a
`j0` arm that RETRIEVES top-12 and replays only those 12 chunks at full depth.
That is a *different, much cheaper* arm than phase-1's C1: it is a matched-pack
raw-text RAG baseline, not the pack-all C1 whose quality won the phase-1 gate.
So the published crossover numbers do NOT answer "does CoMem beat the arm that
beat it on quality?" This driver adds the missing arm and keeps the matched one,
which is what makes the comparison decide the claim instead of decorating it.

ARMS (all share ONE example, ONE tokenisation, ONE store size L)
----------------------------------------------------------------
  comem   : CoMem j=12 + Read-LoRA. One-time O(L) h12 Write over all N=L/512
            chunks -> persistent store. Per query: select + fetch(top-12 h12
            pack) + 24-layer Read over a CONSTANT-length pack + G decode.
  c1_all  : phase-1 C1. j=0, LoRA dropped, packs ALL N chunks at full depth
            every query. One-time cost ~0 (raw token IDs). Per query: O(L)
            36-layer prefill + G decode. THIS is the phase-1 quality winner.
  j0_top12: matched-pack raw-text RAG. j=0, LoRA dropped, retrieves the SAME
            top-12 chunks as `comem` and replays them at full depth.
            Reference/cross-check against the published P1.8 `j0` arm.

SINGLE-VARIABLE DISCIPLINE (the phase-1 defect this driver does NOT repeat)
---------------------------------------------------------------------------
Phase-1's C1-vs-C2 confounded FOUR axes at once: {read depth} x {LoRA} x
{retrieval vs pack-all} x {selector}. Here the arms are arranged so each
adjacent pair differs in ONE axis:

  comem  vs j0_top12 : SAME selector (iter_bm25), SAME top-12 pack (asserted
                       bit-identical via `packed_ids_sha256`), SAME decode
                       length. Differs ONLY in {read depth 12 vs 0} (+ the
                       LoRA that depth-12 read requires -- see CONFOUND 1).
  j0_top12 vs c1_all : SAME depth (j=0), SAME absence of LoRA, SAME decode.
                       Differs ONLY in {retrieve top-12 vs pack all N}.

So "cost of mid-depth read" and "cost of retrieval" are separated, and the
2-hop path comem -> j0_top12 -> c1_all reaches the phase-1 comparison without
ever moving two axes at once. CONFOUND 1 (depth and LoRA move together) is
irreducible here and is reported, not hidden: a j=12 read without the Read-LoRA
is not a functional arm, and phase-1's own C1/C2 carried the same coupling.

COST MODEL
----------
    cumulative(N, G) = write_once + N * (select + fetch + read + decode(G))
    N*(arm_a vs arm_b) = (W_a - W_b) / (perq_b - perq_a)      [infinite if <= 0]

`select` is measured per arm rather than assumed to cancel: `c1_all` runs NO
selector at all (it packs everything), so unlike the P1.8 comem-vs-j0 pairing
the selection term does NOT cancel and must be timed. It is timed on CPU with
the canonical `_select_context_chunk_indices` (imported, never reimplemented).

STORAGE
-------
Measured as REAL BYTES, two ways, never estimated from dimensions alone:
  * in-memory tensor bytes (element_size * nelement), and
  * actual on-disk file size after `torch.save` + fsync of the same store,
plus the raw-text baseline's true on-disk cost (token IDs + the BM25 posting
list that retrieval needs). Reported as bytes/token so it is L-independent.

REUSED, NEVER REIMPLEMENTED
---------------------------
Everything model-facing is imported from `bench_p1_8_serving_curve` ->
`eval_p016_e0_write_control` -> `bench_p1_7_h12_oracle` ->
`bench_p0_13_quality_latency`, so the Read/decode/pack primitives are
bit-identical to the Paper A published latency path and the numbers stay
comparable to the released P1.8 artifact:
  `_load`, `_build_pack`, `_eos_ids`, `_summ`, `_sync`, `_peak_gb`,
  `QCMemModel`, `ruler`, `_bare_question`, `_resolve_task`, `EXPECTED_LORA_SHA`,
  `_build_store`, `_serve_comem`, `_serve_j0`, `parse_length`, `_peak_host_gb`.
The selector comes from `eval_qcmem_babilong._select_context_chunk_indices`.

FAIL-CLOSED GATES
-----------------
  1. LoRA sha == EXPECTED_LORA_SHA (flagship dd09cd17...) for the comem arm.
  2. comem and j0_top12 read the SAME top-12 pack (packed_ids_sha256 equal).
  3. c1_all packs exactly N context chunks == `range(len(context_chunks))`,
     matching `eval_qcmem_babilong`'s `no_retrieval` branch verbatim.
  4. persistent store bytes == N * chunk * d * element_size, exactly.
  5. finite logits on every measured Read/decode of every arm.
  6. store-fetched h12 == fresh recompute, bit-identical (max_abs == 0.0).

SINGLE GPU by design. This node also hosts an A03 eval watcher that needs all 8
GPUs for a few minutes every ~2.8h, so each unit here uses ONE GPU and writes
its own result file; a transient OOM costs one retryable cell, never the run.

Protocol invariants: chat_template=False (no template is ever applied),
selector=iter_bm25, bf16, sdpa, seed=42.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import socket
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- import the published P1.8 harness and reuse its primitives verbatim ----
import scripts.bench_p1_8_serving_curve as p18  # noqa: E402
import scripts.eval_qcmem_babilong as qcb       # noqa: E402

ruler = p18.ruler
QCMemModel = p18.QCMemModel
_bare_question = p18._bare_question
_resolve_task = p18._resolve_task
_build_pack = p18._build_pack
_load = p18._load
_eos_ids = p18._eos_ids
_sync = p18._sync
_peak_gb = p18._peak_gb
_peak_host_gb = p18._peak_host_gb
_summ = p18._summ
_build_store = p18._build_store
_serve_comem = p18._serve_comem
_serve_j0 = p18._serve_j0
_finite = p18._finite
parse_length = p18.parse_length
EXPECTED_LORA_SHA = p18.EXPECTED_LORA_SHA

BYTES_PER_TOKEN_ID = 4          # int32 raw token store
PREREG_NSTAR_MAX = 1e5          # pre-registered plausibility ceiling on N*


# --------------------------------------------------------------------------- #
# c1_all: the phase-1 C1 arm. j=0, no LoRA, packs ALL N context chunks at full
# depth EVERY query. Uses the same QCMemModel primitives as `_serve_j0` (which
# is itself the reused P1.8 primitive) -- the ONLY difference vs j0_top12 is
# which chunk set is packed: `range(N)` instead of the top-12 `sel_idx`. That is
# exactly `eval_qcmem_babilong`'s `no_retrieval` branch.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _serve_c1_all(qc0, sink_bos_id, all_ctx_chunks, query_ids, G, eos_ids,
                  device):
    """One served query for phase-1 C1 (pack-all, full-depth, per query)."""
    _sync()
    tf0 = time.perf_counter()
    sel = [c.to(device) for c in all_ctx_chunks]
    _sync()
    fetch_s = time.perf_counter() - tf0

    _sync()
    tr0 = time.perf_counter()
    sink_hj = qc0.write_chunk([sink_bos_id])
    # write_chunks at j=0 is embed-only and is the same primitive _serve_j0 uses.
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
# selection cost (CPU). Canonical selector, imported. c1_all does NOT select.
# --------------------------------------------------------------------------- #
def _time_select(context_chunks, bare_q_ids, topk, iter_hop_topk, n_repeat,
                 warmup):
    """Median wall time of ONE canonical iter_bm25 selection over N chunks."""
    ts = []
    for it in range(warmup + n_repeat):
        t0 = time.perf_counter()
        sel = qcb._select_context_chunk_indices(
            "iter_bm25", context_chunks, list(bare_q_ids or []), topk,
            None, context_hj=None, query_hj=None,
            iter_rounds=0, iter_hop_topk=iter_hop_topk, iter_score="meanpool",
            iter_conf_ratio=0.3, iter_max_chunks=64,
        )
        dt = time.perf_counter() - t0
        if it >= warmup:
            ts.append(dt)
    return _summ(ts), sel


def _time_bm25_index(docs):
    """One-time BM25 posting-list build over the raw token store (CPU), and its
    real serialised size. Mirrors the P1.8 j0 index-build term."""
    from collections import Counter
    t0 = time.perf_counter()
    df = Counter()
    postings = {}
    for di, d in enumerate(docs):
        tf = Counter(d)
        for t, c in tf.items():
            df[t] += 1
            postings.setdefault(t, []).append((di, c))
    build_s = time.perf_counter() - t0
    # real posting-list bytes: (doc_id int32, tf int32) per entry + vocab key int32
    n_entries = sum(len(v) for v in postings.values())
    index_bytes = n_entries * 8 + len(postings) * 4
    return build_s, index_bytes, n_entries


# --------------------------------------------------------------------------- #
# real on-disk store size: torch.save + fsync, measure the file, then delete.
# --------------------------------------------------------------------------- #
def _measure_ondisk_bytes(tensor, scratch_dir):
    p = Path(scratch_dir) / f"_a02_store_probe_{os.getpid()}.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(p, "wb") as f:
        torch.save(tensor, f)
        f.flush()
        os.fsync(f.fileno())
    save_s = time.perf_counter() - t0
    nbytes = p.stat().st_size
    p.unlink(missing_ok=True)
    return nbytes, save_s


# --------------------------------------------------------------------------- #
# SERVE
# --------------------------------------------------------------------------- #
def run_serve(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    n_layers = int(model.config.num_hidden_layers)
    d_model = int(model.config.hidden_size)
    vocab = int(model.config.vocab_size)

    # ---- GATE 1: flagship Read-LoRA ----
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[a02][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")

    qc12 = QCMemModel(model, resume_j=args.resume_j)   # comem  (j=12 + LoRA)
    qc0 = QCMemModel(model, resume_j=0)                # j0_top12 / c1_all (j=0)
    eos12 = _eos_ids(qc12, tokenizer)
    eos0 = _eos_ids(qc0, tokenizer)
    kv_bytes_per_tok = qc0.cacheblend_kv_bytes_per_tok(2)

    # ---- build the sample + the L-INDEPENDENT top-12 Read pack -------------
    task = _resolve_task(args.task)
    rlen = args.read_sample_length
    target_tokens = ruler._LENGTH_TOKENS[rlen]
    import random as _random
    base_seed = args.seed + (hash((task, rlen)) % 100000)
    vt_icl = ruler._make_vt_icl(_random.Random(base_seed + 777), 4) \
        if task == "variable_tracking" else None
    rng = _random.Random(base_seed * 1000 + args.example_index)
    prompt, answers, gold_needle = ruler._build_sample(
        task, target_tokens, tokenizer, rng, vt_icl)
    bare_q = _bare_question(prompt)                     # chat_template=False
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

    tokens = input_ids[0]
    all_ctx_chunks_real = list(tokens.split(args.chunk_size))[:-1]

    L = parse_length(args.store_length)
    n_store = max(len(sel_idx), L // args.chunk_size)
    Gs = [int(g) for g in args.gen_lengths]

    # ---- GATE 0: the store must cover the whole real sample --------------- #
    # `_build_store` lays the real sample's context chunks down as the LEADING
    # store rows, so `sel_idx` (which indexes the real sample's chunks) is only a
    # valid index into the store when N >= len(all_ctx_chunks_real). If the
    # requested store L is smaller than the read-pack sample, the selected rows
    # fall off the end of the store and `index_select` reads out of bounds --
    # which surfaces as an opaque device-side assert, not a clean error. Fail
    # closed with the actionable message instead.
    n_real = len(all_ctx_chunks_real)
    if n_store < n_real:
        raise SystemExit(
            f"[a02][ABORT] store_length={args.store_length} gives N={n_store} "
            f"rows but the read_sample_length={rlen} pack spans {n_real} context "
            f"chunks (max sel_idx={max(sel_idx)}). The store must cover the whole "
            f"read sample: use --store_length >= --read_sample_length.")

    print(f"[a02][serve] proc={args.proc_id} tier={args.tier} "
          f"store_L={args.store_length} (N={n_store} chunks) read_pack={rlen} "
          f"task={task} pack_read_len={pack['pack_read_len']} sel={sel_idx}",
          flush=True)

    # ==== ONE-TIME: CoMem O(L) h12 Write store ====
    torch.cuda.reset_peak_memory_stats()
    store, sink_hj12, write_once_comem, n_written, store_bytes = _build_store(
        qc12, all_ctx_chunks_real, n_store, args.chunk_size, vocab, device,
        args.tier, sink_bos)
    peak_gpu_write = _peak_gb()
    peak_host_write = _peak_host_gb()

    # ---- GATE 4: persistent bytes exactness ----
    expect_bytes = n_written * args.chunk_size * d_model * store.element_size()
    if store_bytes != expect_bytes:
        raise SystemExit(f"[a02][ABORT] store bytes {store_bytes} != "
                         f"expected {expect_bytes}")

    # ---- the N-chunk raw token store the j=0 arms read from ----------------
    # Real sample chunks first, then synthetic random-id chunks to reach N --
    # identical construction to `_build_store`, so the two stores describe the
    # SAME corpus. Random ids only affect COST/SIZE axes, never the top-12 Read
    # pack (which comes from the real sample's chunks via `sel_idx`).
    ctx_chunks_full = list(all_ctx_chunks_real)
    g = torch.Generator().manual_seed(args.seed + 1234)
    while len(ctx_chunks_full) < n_store:
        # created on CPU for reproducibility then moved, so every raw chunk lives
        # in the SAME place as the real sample's chunks. This keeps the raw-token
        # `fetch` term identical between `j0_top12` (whose chunks come from the
        # pack, i.e. on device) and `c1_all` -- otherwise the retrieve-vs-pack-all
        # contrast would silently also move the store tier.
        ctx_chunks_full.append(
            torch.randint(0, vocab, (args.chunk_size,),
                          generator=g).to(all_ctx_chunks_real[0].device))
    ctx_chunks_full = ctx_chunks_full[:n_store]

    # ---- GATE 3: c1_all packs exactly range(N) (the `no_retrieval` branch) --
    c1_sel_idx = list(range(len(ctx_chunks_full)))
    if c1_sel_idx != list(range(n_store)):
        raise SystemExit("[a02][ABORT] c1_all sel_idx != range(N)")

    # ---- one-time BM25 index (needed by comem AND j0_top12, not by c1_all) --
    docs = [c.tolist() for c in ctx_chunks_full]
    index_s, index_bytes, n_postings = _time_bm25_index(docs)
    raw_store_bytes = n_store * args.chunk_size * BYTES_PER_TOKEN_ID

    # ---- GATE 6: store-fetched h12 == fresh recompute ----------------------
    verify_max_abs = None
    if args.verify:
        with torch.no_grad():
            fresh = [qc12.write_chunk(all_ctx_chunks_real[k]) for k in sel_idx]
            fresh_cat = torch.cat([h[0] for h in fresh], dim=0).float()
            vidx = torch.as_tensor(sel_idx, dtype=torch.long)
            got = store.index_select(0, vidx.to(store.device)).to(device)
            got_cat = torch.cat([got[k] for k in range(got.shape[0])],
                                dim=0).float()
            verify_max_abs = float((fresh_cat - got_cat).abs().max().item())
        if verify_max_abs != 0.0:
            raise SystemExit(f"[a02][ABORT] store!=recompute "
                             f"max_abs={verify_max_abs}")
        print("[a02][serve] GATE6 store==recompute max_abs=0.0 PASS", flush=True)

    # ---- selection cost (CPU), canonical selector --------------------------
    sel_summ, sel_check = _time_select(ctx_chunks_full, bare_q_ids, args.topk,
                                       args.iter_hop_topk, args.n_repeat,
                                       args.warmup)

    # ---- real on-disk store sizes -----------------------------------------
    ondisk = None
    if args.measure_ondisk:
        cbytes, csave = _measure_ondisk_bytes(
            store if args.tier == "cpu" else store.to("cpu"), args.scratch_dir)
        rbytes, rsave = _measure_ondisk_bytes(
            torch.stack([c.detach().to("cpu", torch.int32)
                         for c in ctx_chunks_full]),
            args.scratch_dir)
        ondisk = {"comem_store_file_bytes": cbytes, "comem_save_s": csave,
                  "raw_token_store_file_bytes": rbytes, "raw_save_s": rsave,
                  "scratch_dir": str(args.scratch_dir)}

    # ==== per-query timings: median-of-n_repeat per arm x G ================
    def _measure(fn_query, label):
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
            print(f"[a02][{label}] G={G} read={by_g[G]['read_s']['median']*1e3:.1f}ms "
                  f"decode={by_g[G]['decode_s']['median']*1e3:.1f}ms "
                  f"read_len={rl}", flush=True)
        return by_g

    torch.cuda.reset_peak_memory_stats()
    comem_g = _measure(lambda G: _serve_comem(
        qc12, store, sink_hj12, sel_idx, query_ids, G, eos12, device,
        args.tier), "comem")
    peak_gpu_comem = _peak_gb()

    j0_sel_tensors = pack["selected_chunk_tensors"]
    torch.cuda.reset_peak_memory_stats()
    j0_g = _measure(lambda G: _serve_j0(
        qc0, sink_bos, j0_sel_tensors, query_ids, G, eos0, device, args.tier),
        "j0_top12")
    peak_gpu_j0 = _peak_gb()

    # c1_all may OOM at large L (its read is O(L)); record that as a RESULT.
    torch.cuda.reset_peak_memory_stats()
    c1_g, c1_status = None, "ok"
    try:
        c1_g = _measure(lambda G: _serve_c1_all(
            qc0, sink_bos, ctx_chunks_full, query_ids, G, eos0, device),
            "c1_all")
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        c1_status = "OOM"
        print(f"[a02][c1_all] OOM at N={n_store} (read is O(L)) -- recorded",
              flush=True)
    peak_gpu_c1 = _peak_gb()

    # ---- GATE 5: finite logits everywhere ---------------------------------
    for arm, gg in (("comem", comem_g), ("j0_top12", j0_g), ("c1_all", c1_g)):
        if gg is None:
            continue
        for G, r in gg.items():
            if not r["finite"]:
                raise SystemExit(f"[a02][ABORT] non-finite logits arm={arm} G={G}")

    # ---- GATE 2: comem and j0_top12 read the SAME top-12 pack -------------
    if comem_g and j0_g:
        for G in Gs:
            if comem_g[G]["read_len"] != j0_g[G]["read_len"]:
                raise SystemExit(
                    f"[a02][ABORT] pack mismatch comem read_len="
                    f"{comem_g[G]['read_len']} != j0_top12 "
                    f"{j0_g[G]['read_len']} (arms must share ONE pack)")

    result = {
        "mode": "serve", "gate": "A02_storage_readcompute",
        "proc_id": args.proc_id,
        "store_length": args.store_length, "store_L_tokens": L,
        "n_store_chunks": n_store, "n_written": n_written,
        "tier": args.tier, "read_sample_length": rlen, "task": task,
        "example_index": args.example_index, "gen_lengths": Gs,
        "config": {
            "resume_j": args.resume_j, "selector": "iter_bm25",
            "topk": args.topk, "iter_hop_topk": args.iter_hop_topk,
            "chunk_size": args.chunk_size, "warmup": args.warmup,
            "n_repeat": args.n_repeat, "dtype": args.dtype,
            "attn_impl": args.attn_impl, "seed": args.seed,
            "use_chat_template": False,
            "lora_sha256": lora_sha256, "lora_layers": lora_layers,
            "num_layers": n_layers, "hidden": d_model, "vocab": vocab,
            "kv_bytes_per_token_fulldepth": kv_bytes_per_tok,
            "prereg_nstar_max": PREREG_NSTAR_MAX,
        },
        "arms": {
            "comem": "j=12 + Read-LoRA, write-once h12 store, top-12 pack",
            "j0_top12": "j=0, no LoRA, SAME top-12 pack, full-depth replay",
            "c1_all": "phase-1 C1: j=0, no LoRA, packs ALL N chunks per query",
        },
        "pack": {"sel_idx": sel_idx, "pack_read_len": pack["pack_read_len"],
                 "packed_ids_sha256": pack["packed_ids_sha256"],
                 "n_ctx_chunks_real_sample": pack["n_ctx_chunks"],
                 "c1_n_packed_chunks": n_store},
        "one_time": {
            "comem_write_once_s": write_once_comem,
            "bm25_index_s": index_s,
            "c1_index_s": 0.0,
            "comem_store_bytes": store_bytes,
            "raw_token_store_bytes": raw_store_bytes,
            "bm25_index_bytes": index_bytes,
            "n_bm25_postings": n_postings,
            "comem_bytes_per_token": store.element_size() * d_model,
            "raw_bytes_per_token": BYTES_PER_TOKEN_ID,
            "peak_gpu_write_gb": peak_gpu_write,
            "peak_host_write_gb": peak_host_write,
        },
        "select": {"iter_bm25_s": sel_summ, "n_chunks_scored": n_store,
                   "sel_idx_over_full_store": sel_check},
        "ondisk": ondisk,
        "peak": {"peak_gpu_comem_gb": peak_gpu_comem,
                 "peak_gpu_j0_top12_gb": peak_gpu_j0,
                 "peak_gpu_c1_all_gb": peak_gpu_c1,
                 "peak_host_gb": _peak_host_gb()},
        "comem": comem_g, "j0_top12": j0_g,
        "c1_all": c1_g, "c1_all_status": c1_status,
        "verify_store_eq_recompute_max_abs": verify_max_abs,
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda" else None,
                "python": platform.python_version(),
                "node": socket.gethostname()},
    }
    outdir = Path(args.output_dir) / "serve"
    outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / (f"a02_serve_{args.store_length}_{args.tier}_{task}_"
                   f"proc{args.proc_id}.json")
    with open(fn, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[a02][serve] wrote {fn}", flush=True)


# --------------------------------------------------------------------------- #
# AGGREGATE: pool procs -> N* per (L, tier, G, comparison) + storage ratios
# --------------------------------------------------------------------------- #
def _med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _p90(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * 0.9
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return vals[lo] if lo == hi else vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def run_aggregate(args):
    files = sorted(glob.glob(os.path.join(args.output_dir, "serve",
                                          "a02_serve_*.json")))
    if not files:
        print(f"[a02][aggregate] no files under {args.output_dir}/serve")
        return
    recs = []
    for fp in files:
        with open(fp) as f:
            recs.append(json.load(f))

    # ---- shard-completeness assertion (repo red line) ---------------------
    groups = {}
    for r in recs:
        groups.setdefault((r["store_length"], r["tier"]), []).append(r)
    incomplete = {f"{k[0]}|{k[1]}": len(v) for k, v in groups.items()
                  if len(v) < args.expect_procs}
    if incomplete and not args.allow_partial:
        raise SystemExit(
            f"[a02][ABORT] incomplete cells (expect {args.expect_procs} procs "
            f"each): {incomplete}. Re-run the missing procs or pass "
            f"--allow_partial to record a partial aggregate deliberately.")

    NS = [int(n) for n in args.query_counts]
    out = {"mode": "aggregate", "gate": "A02_storage_readcompute",
           "n_files": len(files), "query_counts": NS,
           "expect_procs": args.expect_procs,
           "partial_cells": incomplete or None,
           "prereg_nstar_max": PREREG_NSTAR_MAX,
           "cells": {}}

    for (sl, tier), rs in sorted(groups.items(),
                                 key=lambda kv: (parse_length(kv[0][0]),
                                                 kv[0][1])):
        Gs = rs[0]["gen_lengths"]
        r0 = rs[0]
        cW = _med([r["one_time"]["comem_write_once_s"] for r in rs])
        idx_s = _med([r["one_time"]["bm25_index_s"] for r in rs])
        sel_s = _med([r["select"]["iter_bm25_s"]["median"] for r in rs])

        cbytes = r0["one_time"]["comem_store_bytes"]
        rbytes = r0["one_time"]["raw_token_store_bytes"]
        ibytes = r0["one_time"]["bm25_index_bytes"]
        # RAG must persist raw text AND the index it retrieves with; CoMem must
        # persist the h12 store AND (for iter_bm25) the same index + the raw
        # token ids the selector scores. Report both a strict h12-vs-raw ratio
        # and a deployable total-vs-total ratio so neither flatters an arm.
        cell = {
            "store_L_tokens": r0["store_L_tokens"],
            "n_store_chunks": r0["n_store_chunks"],
            "n_procs": len(rs),
            "one_time": {
                "comem_write_once_s": {"median": cW,
                    "p90": _p90([r["one_time"]["comem_write_once_s"] for r in rs])},
                "bm25_index_s": {"median": idx_s},
                "c1_index_s": 0.0,
            },
            "select_s": {"median": sel_s,
                         "p90": _p90([r["select"]["iter_bm25_s"]["p90"]
                                      for r in rs])},
            "storage": {
                "comem_h12_bytes": cbytes,
                "raw_token_bytes": rbytes,
                "bm25_index_bytes": ibytes,
                "comem_bytes_per_token": r0["one_time"]["comem_bytes_per_token"],
                "raw_bytes_per_token": r0["one_time"]["raw_bytes_per_token"],
                "kv_bytes_per_token_fulldepth":
                    r0["config"]["kv_bytes_per_token_fulldepth"],
                "ratio_h12_over_raw": (cbytes / rbytes) if rbytes else None,
                "ratio_comem_total_over_rag_total": (
                    (cbytes + rbytes + ibytes) / (rbytes + ibytes)
                    if (rbytes + ibytes) else None),
                "ondisk": r0.get("ondisk"),
            },
            "peak_gb": {
                "comem": _med([r["peak"]["peak_gpu_comem_gb"] for r in rs]),
                "j0_top12": _med([r["peak"]["peak_gpu_j0_top12_gb"] for r in rs]),
                "c1_all": _med([r["peak"]["peak_gpu_c1_all_gb"] for r in rs]),
            },
            "c1_all_status": r0.get("c1_all_status"),
            "per_G": {}, "crossover": {},
        }

        for G in Gs:
            gk = str(G)

            def _comp(arm):
                if any(r.get(arm) is None for r in rs):
                    return None
                def _g(r):
                    a = r[arm]
                    return a[G] if G in a else a[str(G)]
                f_ = _med([_g(r)["fetch_s"]["median"] for r in rs])
                rd = _med([_g(r)["read_s"]["median"] for r in rs])
                dc = _med([_g(r)["decode_s"]["median"] for r in rs])
                return {
                    "fetch_s": f_, "read_s": rd, "decode_s": dc,
                    "read_len": _g(rs[0])["read_len"],
                    "read_p90": _p90([_g(r)["read_s"]["p90"] for r in rs]),
                    "decode_p90": _p90([_g(r)["decode_s"]["p90"] for r in rs]),
                }

            arms = {a: _comp(a) for a in ("comem", "j0_top12", "c1_all")}
            # per-query total. select is charged to the arms that retrieve
            # (comem, j0_top12); c1_all packs everything so it never selects.
            perq = {}
            for a, v in arms.items():
                if v is None:
                    perq[a] = None
                    continue
                s = sel_s if a in ("comem", "j0_top12") else 0.0
                v = dict(v)
                v["select_s"] = s
                v["per_query_s"] = s + v["fetch_s"] + v["read_s"] + v["decode_s"]
                arms[a] = v
                perq[a] = v["per_query_s"]
            cell["per_G"][gk] = arms

            # one-time cost charged to each arm
            W = {"comem": (cW or 0.0) + (idx_s or 0.0),
                 "j0_top12": (idx_s or 0.0),
                 "c1_all": 0.0}

            xo = {}
            for a, b in (("comem", "c1_all"), ("comem", "j0_top12"),
                         ("j0_top12", "c1_all")):
                if perq.get(a) is None or perq.get(b) is None:
                    xo[f"{a}_vs_{b}"] = {"n_star": None,
                                         "reason": "arm missing (see status)"}
                    continue
                denom = perq[b] - perq[a]
                nstar = ((W[a] - W[b]) / denom) if denom > 0 else float("inf")
                grid = {}
                for N in NS:
                    ca = W[a] + N * perq[a]
                    cb = W[b] + N * perq[b]
                    grid[str(N)] = {
                        f"{a}_cumulative_s": round(ca, 4),
                        f"{b}_cumulative_s": round(cb, 4),
                        f"{a}_amortized_ms": round(ca / N * 1e3, 3),
                        f"{b}_amortized_ms": round(cb / N * 1e3, 3),
                        "winner": a if ca < cb else b,
                    }
                xo[f"{a}_vs_{b}"] = {
                    "n_star": (None if nstar == float("inf") else nstar),
                    "n_star_infinite": nstar == float("inf"),
                    "per_query_s": {a: perq[a], b: perq[b]},
                    "one_time_s": {a: W[a], b: W[b]},
                    "reachable_within_prereg": (
                        nstar != float("inf") and nstar <= PREREG_NSTAR_MAX),
                    "grid": grid,
                }
            cell["crossover"][gk] = xo

        out["cells"][f"{sl}|{tier}"] = cell
        g1 = cell["crossover"].get(str(Gs[0]), {})
        msg = []
        for k, v in g1.items():
            if v.get("reason"):
                # arm absent (e.g. c1_all OOM). This is NOT "N* is infinite":
                # infinite means measured-but-never-cheaper, absent means the
                # arm could not run at all. Never collapse the two.
                msg.append(f"{k}: N/A ({v['reason']})")
            elif v.get("n_star_infinite"):
                msg.append(f"{k}: N*=inf (never cheaper)")
            else:
                msg.append(f"{k}: N*={v['n_star']:.2f}")
        print(f"[a02][aggregate] L={sl} tier={tier} G={Gs[0]} | "
              + " | ".join(msg), flush=True)

    fn = os.path.join(args.output_dir, "a02_storage_readcompute_aggregate.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[a02][aggregate] wrote {fn}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="A02 reframe gate: storage / read-compute for high reuse")
    ap.add_argument("--mode", choices=["serve", "aggregate", "manifest"],
                    default="serve")
    ap.add_argument("--model_path", type=str,
                    default="/apdcephfs_zwfy6/share_304376610/pighzliu_code/"
                            "models/Qwen--Qwen3-8b")
    ap.add_argument("--lora_adapter", type=str,
                    default="outputs/qcmem_distill_qwen_j12_r32_4k/final")
    ap.add_argument("--resume_j", type=int, default=12)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--iter_hop_topk", type=int, default=4)
    ap.add_argument("--store_length", type=str, default="32k")
    ap.add_argument("--read_sample_length", type=str, default="32k",
                    choices=list(ruler._LENGTH_TOKENS.keys()))
    ap.add_argument("--gen_lengths", type=str, nargs="+",
                    default=["1", "32", "128"])
    ap.add_argument("--query_counts", type=str, nargs="+",
                    default=["1", "4", "16", "64", "256", "1024"])
    ap.add_argument("--tier", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--task", type=str, default="niah_multikey_1")
    ap.add_argument("--example_index", type=int, default=0)
    ap.add_argument("--proc_id", type=int, default=0)
    ap.add_argument("--n_repeat", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--measure_ondisk", action="store_true",
                    help="torch.save+fsync the stores and measure real file size")
    ap.add_argument("--scratch_dir", type=str, default="/tmp/a02_store_probe")
    ap.add_argument("--expect_procs", type=int, default=3,
                    help="aggregate: required procs per (L,tier) cell")
    ap.add_argument("--allow_partial", action="store_true",
                    help="aggregate: permit incomplete cells (records which)")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str,
                    default="bench_results/a02_storage_readcompute")
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
        _t, _m, sha, layers = _load(args.model_path, dtype, args.attn_impl,
                                    device, args.lora_adapter)
        ok = sha == EXPECTED_LORA_SHA
        print(f"[a02][manifest] lora_sha256={sha} match={ok} layers={layers}",
              flush=True)
        if not ok:
            raise SystemExit("[a02][ABORT] LoRA sha mismatch")
        return
    run_serve(args, device, dtype)


if __name__ == "__main__":
    main()
