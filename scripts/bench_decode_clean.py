#!/usr/bin/env python
"""Clean decode-latency benchmark: Dense (full-context) vs CoMem (QCMem).

WHY THIS SCRIPT EXISTS
----------------------
The older ``bench_qcmem_decode.py`` estimates decode cost as ``(t_N - t_1)/(N-1)``
(a noisy 2-point difference) and the older ``bench_qcmem_vs_fullctx.py`` times the
CoMem decode through the SLOW *recompute* path (re-encode + full read every step,
~seconds/step). Neither is the honest deploy number. This script fixes both:

  * BOTH methods decode with a KV cache (O(1)/step):
      - Dense : HF ``model(..., past_key_values=cache, use_cache=True)`` incremental
                decode from a prefilled cache (``logits_to_keep=1`` so prefill only
                computes the last-position logits — the fair "time to first token",
                and the same last-position-only lm_head CoMem's read_prefill does).
      - CoMem : the EXACT resumed-band KV-cache path the real eval uses
                (``eval_qcmem_babilong.qcmem_generate`` with ``use_kv_cache=True``):
                ``write_prefill`` (bottom-band cache) + ``read_prefill`` (top-band
                cache) once, then ``decode_step`` pushes ONE token/step through
                both bands. This is NOT the recompute path.
  * Prefill and decode are timed SEPARATELY.
  * The decode loop is timed DIRECTLY: ``torch.cuda.synchronize()`` before the
    loop -> generate exactly N tokens -> synchronize after; per-token =
    (t_end - t_start)/N. Per-step GPU times are captured with CUDA events (no
    per-step CPU sync stall, captures GPU idle bubbles) and aggregated across all
    repeats for a p95.
  * EOS is DISABLED: every method is forced to emit exactly ``--max_new_tokens``
    tokens so early-stop never pollutes the per-token timing.
  * CORRECTNESS GATE (per length, hard): run CoMem's kv-cache decode vs its
    recompute decode on the same prompt (via ``qcmem_generate`` twice with per-step
    logit capture), assert the generated token ids are token-for-token identical and
    report ``max|logit diff|``. A gate failure RAISES (never silent).

Dense at very long contexts OOMs on one H20 (the KV cache alone is ~1.2 MB/token
for Qwen3-8B => ~155 GB at 128k). That is caught, recorded as ``"OOM"``, and the
run continues — CoMem's read pack is fixed (~topk*chunk) so it never OOMs; the OOM
IS the point (Dense cannot run where CoMem still does).

Run (one length family, one free GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_decode_clean.py \
        --context_lengths 8k 16k 32k 64k 128k \
        --max_new_tokens 128 --warmup 1 --n_repeat 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.memory.qcmem.qcmem_model import QCMemModel  # noqa: E402
import scripts.eval_qcmem_babilong as qcb  # noqa: E402

qcmem_generate = qcb.qcmem_generate
_select_context_chunk_indices = qcb._select_context_chunk_indices

# Candidate Qwen3-8B weight paths, tried in order (first existing dir wins). The
# diskB local mirror is preferred; the wzc1 canonical path is the fallback the
# other bench scripts use.
_MODEL_PATH_CANDIDATES = [
    "/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen3-8b-local",
    "/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b",
    "/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen3-8B-Base",
]

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_length(s: str) -> int:
    """'8k' -> 8192, '128k' -> 131072, '4096' -> 4096."""
    s = str(s).strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_gb(device) -> float:
    return torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


def _percentile(values, q: float) -> float:
    """Linear-interpolated q-quantile (q in [0,1]) of a list of floats."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def resolve_model_path(cli_path: str) -> str:
    """Return the first existing weight dir: CLI override, then the candidates."""
    cands = ([cli_path] if cli_path else []) + _MODEL_PATH_CANDIDATES
    for p in cands:
        if p and os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "no Qwen3-8B weight dir found; tried: " + ", ".join(c for c in cands if c)
    )


def _time_decode_loop(step_fn, n_steps: int, device):
    """Run ``step_fn(i)`` for ``n_steps`` and return (wall_total_s, per_step_s).

    wall_total_s : perf_counter delta with a ``synchronize`` before/after the loop
                   (the honest end-to-end decode wall time; includes any per-step
                   CPU<->GPU sync a decode path incurs, e.g. CoMem's argmax.item()).
    per_step_s   : per-step GPU time from CUDA events recorded around each step
                   (no per-step CPU sync stall; captures GPU idle bubbles). On CPU
                   falls back to per-step perf_counter deltas.
    """
    if device.type == "cuda":
        evs = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps + 1)]
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        evs[0].record()
        for i in range(n_steps):
            step_fn(i)
            evs[i + 1].record()
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - t0
        per_step = [evs[i].elapsed_time(evs[i + 1]) / 1000.0 for i in range(n_steps)]
    else:
        t0 = time.perf_counter()
        per_step = []
        for i in range(n_steps):
            s = time.perf_counter()
            step_fn(i)
            per_step.append(time.perf_counter() - s)
        wall = time.perf_counter() - t0
    return wall, per_step


# --------------------------------------------------------------------------- #
# correctness gate (kv-cache decode == recompute decode), per length
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_correctness_gate(qc, tokenizer, input_ids, *, chunk_size, topk, selector,
                         sink_tokens, gate_new_tokens, tol):
    """CoMem kv-cache vs recompute decode on the SAME prompt (mirrors
    ``bench_qcmem_decode.run_correctness``). Returns dict with max|logit diff| and
    the token-for-token match flag. RAISES on failure (never silent)."""
    bare_q = input_ids[0].tolist()[-8:]  # dummy lexical query (unused by recency)

    def gen(use_kv):
        stats = {"capture_step_logits": True}
        qcmem_generate(
            qc=qc, tokenizer=tokenizer, input_ids=input_ids,
            chunk_size=chunk_size, max_new_tokens=gate_new_tokens,
            selector=selector, topk=topk, sink_tokens=sink_tokens,
            bare_question_ids=bare_q, stats=stats, use_kv_cache=use_kv,
        )
        return stats

    st_base = gen(False)   # recompute path
    st_kv = gen(True)      # resumed-band KV cache path
    lb, lk = st_base["step_logits"], st_kv["step_logits"]
    gb, gk = st_base["generated_ids"], st_kv["generated_ids"]
    n = min(len(lb), len(lk))
    max_diff = 0.0
    argmax_ok = True
    for s in range(n):
        max_diff = max(max_diff, (lb[s] - lk[s]).abs().max().item())
        if int(lb[s].argmax()) != int(lk[s].argmax()):
            argmax_ok = False
    len_ok = (len(lb) == len(lk))
    tokens_match = (gb == gk)
    ok = len_ok and tokens_match and argmax_ok and (max_diff < tol)

    print(f"    [gate] steps recompute={len(lb)} kv={len(lk)} | "
          f"tokens_match={tokens_match} argmax_ok={argmax_ok} | "
          f"max|logit diff|={max_diff:.3e} (tol {tol:.1e}) -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError(
            f"CoMem decode correctness gate FAILED: kv-cache decode != recompute "
            f"decode (tokens_match={tokens_match}, argmax_ok={argmax_ok}, "
            f"len_ok={len_ok}, max_diff={max_diff:.3e}, tol={tol:.1e})"
        )
    return {"max_logit_diff": max_diff, "tokens_match": tokens_match}


# --------------------------------------------------------------------------- #
# Dense (full-context) — both prefill and decode use the HF KV cache
# --------------------------------------------------------------------------- #
def bench_dense(model, input_ids, *, max_new_tokens, warmup, n_repeat, device):
    L = int(input_ids.shape[1])
    prefill_times, per_tok_list, all_step_times = [], [], []
    peak = 0.0
    try:
        for it in range(warmup + n_repeat):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            # ---- prefill: one forward over L tokens -> KV cache + first logits ----
            _sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            first = out.logits[:, -1:, :].argmax(dim=-1)  # [1,1] stays on GPU
            _sync(device)
            t_pref = time.perf_counter() - t0
            p_peak = _peak_gb(device)
            del out

            # ---- decode: exactly max_new_tokens incremental steps from the cache --
            state = {"cur": first, "past": past, "past_len": L}

            def step(_i, _st=state):
                with torch.inference_mode():
                    cache_pos = torch.arange(
                        _st["past_len"], _st["past_len"] + 1, device=device)
                    o = model(input_ids=_st["cur"], past_key_values=_st["past"],
                              use_cache=True, cache_position=cache_pos,
                              logits_to_keep=1)
                _st["past"] = o.past_key_values
                _st["cur"] = o.logits[:, -1:, :].argmax(dim=-1)  # no .item() -> async
                _st["past_len"] += 1

            total, per_step = _time_decode_loop(step, max_new_tokens, device)
            d_peak = _peak_gb(device)

            if it >= warmup:
                prefill_times.append(t_pref)
                per_tok_list.append(total / max_new_tokens)
                all_step_times.extend(per_step)
                peak = max(peak, p_peak, d_peak)

            del past, first, state
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except RuntimeError as e:
        if not _is_oom(e):
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"status": "OOM", "seq_len": L}

    med = statistics.median(per_tok_list)
    return {
        "status": "ok",
        "seq_len": L,
        "prefill_s": statistics.median(prefill_times),
        "ms_per_tok_median": med * 1e3,
        "p95_ms": _percentile(all_step_times, 0.95) * 1e3,
        "tok_s": (1.0 / med) if med > 0 else 0.0,
        "peak_gb": peak,
    }


# --------------------------------------------------------------------------- #
# CoMem (QCMem) — resumed-band KV-cache decode, the EXACT real-eval path
# --------------------------------------------------------------------------- #
def bench_comem(qc, tokenizer, input_ids, *, chunk_size, topk, selector,
                sink_tokens, max_new_tokens, warmup, n_repeat, device):
    tokens = input_ids[0]
    chunks = list(tokens.split(chunk_size))
    if len(chunks) < 2:
        raise ValueError("need at least one context chunk + one query chunk")
    context_chunks = chunks[:-1]
    query_chunk = chunks[-1]
    query_ids = query_chunk.tolist()
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = int(tokens[0].item())

    # Which context chunks get packed into the read. Decode latency is
    # selector-INDEPENDENT (the read pack is always sink + topk*chunk + query);
    # recency is forward-free & deterministic, so it is the default. bare-question
    # ids are only consulted by bm25/reader selectors.
    sel_idx = _select_context_chunk_indices(
        selector, context_chunks, query_ids[-8:], topk, None)
    selected_chunks = [context_chunks[i] for i in sel_idx]

    prefill_times, per_tok_list, all_step_times = [], [], []
    peak = 0.0
    read_len = None

    for it in range(warmup + n_repeat):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # ---- prefill: write sink + selected chunks + query, then read (both bands
        #      cached) -> first-token logits. Mirrors qcmem_generate's can_kv branch.
        _sync(device)
        t0 = time.perf_counter()
        sink_hj = qc.write_chunk([bos_id]) if sink_tokens == "bos" else None
        selected_hj = qc.write_chunks(selected_chunks) if selected_chunks else []
        q_hj, bottom_cache, q_local_pos = qc.write_prefill(query_ids)
        logits1, top_cache, pack_pos = qc.read_prefill(sink_hj, selected_hj, q_hj)
        first_tok = int(logits1[0, -1].float().argmax().item())
        _sync(device)
        t_pref = time.perf_counter() - t0
        p_peak = _peak_gb(device)

        if read_len is None:
            sink_len = int(sink_hj.shape[1]) if sink_hj is not None else 0
            sel_len = int(sum(int(h.shape[1]) for h in selected_hj))
            read_len = sink_len + sel_len + len(query_ids)

        # ---- decode: exactly max_new_tokens O(1) decode_step calls -------------- #
        # decode_step consumes a python int (embeds it), so an argmax().item() sync
        # happens each step — this IS the real eval decode cost, so we keep it and
        # the perf_counter wall time captures it honestly.
        state = {"tok": first_tok, "qlp": q_local_pos, "pp": pack_pos}

        def step(_i, _st=state):
            logits = qc.decode_step(
                _st["tok"], bottom_cache, top_cache, _st["qlp"], _st["pp"])
            _st["qlp"] += 1
            _st["pp"] += 1
            _st["tok"] = int(logits[0, -1].float().argmax().item())

        total, per_step = _time_decode_loop(step, max_new_tokens, device)
        d_peak = _peak_gb(device)

        if it >= warmup:
            prefill_times.append(t_pref)
            per_tok_list.append(total / max_new_tokens)
            all_step_times.extend(per_step)
            peak = max(peak, p_peak, d_peak)

        del sink_hj, selected_hj, q_hj, bottom_cache, top_cache, logits1, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    med = statistics.median(per_tok_list)
    return {
        "status": "ok",
        "read_len": read_len,
        "n_context_chunks": len(context_chunks),
        "n_selected_chunks": len(selected_chunks),
        "prefill_s": statistics.median(prefill_times),
        "ms_per_tok_median": med * 1e3,
        "p95_ms": _percentile(all_step_times, 0.95) * 1e3,
        "tok_s": (1.0 / med) if med > 0 else 0.0,
        "peak_gb": peak,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Clean decode-latency benchmark: Dense vs CoMem (both KV cache)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", type=str, default="",
                    help="Override the Qwen3-8B weight dir (else auto-resolved).")
    ap.add_argument("--resume_j", type=int, default=12)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--selector", type=str, default="recency",
                    help="Chunk selector (decode latency is selector-independent; "
                         "recency is forward-free & deterministic).")
    ap.add_argument("--sink_tokens", type=str, default="bos", choices=["bos", "none"])
    ap.add_argument("--context_lengths", type=str, nargs="+",
                    default=["8k", "16k", "32k", "64k", "128k"])
    ap.add_argument("--max_new_tokens", type=int, default=128,
                    help="Forced decode length (EOS disabled); 128 or 256.")
    ap.add_argument("--warmup", type=int, default=1,
                    help="Un-counted full prefill+decode runs (CUDA graph/autotune).")
    ap.add_argument("--n_repeat", type=int, default=3,
                    help="Timed repeats (>=3); median reported, p95 over all steps.")
    ap.add_argument("--gate_new_tokens", type=int, default=16,
                    help="Decode steps for the per-length correctness gate.")
    ap.add_argument("--tol", type=float, default=-1.0,
                    help="Gate max|logit diff| tolerance (-1 -> 1e-2 bf16/fp16, "
                         "1e-4 fp32).")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_gate", action="store_true", default=False,
                    help="Skip the correctness gate (NOT recommended).")
    ap.add_argument("--output", type=str,
                    default=os.path.join(PROJECT_ROOT, "ruler_results",
                                         "bench_decode_clean.json"))
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[bench] cuda requested but unavailable -> cpu")
        args.device = "cpu"
    device = torch.device(args.device)
    dtype = _DTYPES[args.dtype]
    tol = args.tol if args.tol > 0 else (1e-4 if dtype == torch.float32 else 1e-2)
    torch.manual_seed(args.seed)

    model_path = resolve_model_path(args.model_path)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    print("=" * 80)
    print("Clean decode-latency benchmark  (Dense vs CoMem; BOTH use KV cache)")
    print(f"  model_path = {model_path}")
    print(f"  resume_j={args.resume_j} topk={args.topk} chunk_size={args.chunk_size} "
          f"selector={args.selector} sink={args.sink_tokens}")
    print(f"  dtype={args.dtype} attn={args.attn_impl} device={device} ({gpu_name})")
    print(f"  max_new_tokens={args.max_new_tokens} (EOS disabled) "
          f"warmup={args.warmup} n_repeat={args.n_repeat}")
    print(f"  lengths={args.context_lengths}  gate_new_tokens={args.gate_new_tokens} "
          f"tol={tol:.1e}")
    print("=" * 80, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation=args.attn_impl,
        trust_remote_code=True, local_files_only=True,
    ).to(device).eval()

    n_layers = int(model.config.num_hidden_layers)
    vocab = int(model.config.vocab_size)
    if not (0 <= args.resume_j <= n_layers):
        ap.error(f"--resume_j must be in [0, {n_layers}]; got {args.resume_j}")
    qc = QCMemModel(model, resume_j=args.resume_j)
    print(f"[bench] backbone: {n_layers} layers, hidden={model.config.hidden_size}, "
          f"vocab={vocab}, resume_j={qc.resume_j}", flush=True)

    lengths = [(lab, parse_length(lab)) for lab in args.context_lengths]

    rows = []
    for label, L in lengths:
        print(f"\n[bench] === length {label} (L={L}) ===", flush=True)
        input_ids = torch.randint(1, vocab, (1, L), device=device)

        # --- correctness gate (CoMem kv-cache == recompute), forced-loud on fail ---
        corr = {"max_logit_diff": None, "tokens_match": None}
        if not args.skip_gate:
            corr = run_correctness_gate(
                qc, tokenizer, input_ids, chunk_size=args.chunk_size, topk=args.topk,
                selector=args.selector, sink_tokens=args.sink_tokens,
                gate_new_tokens=args.gate_new_tokens, tol=tol)

        # --- Dense (may OOM at long L on one card) ---
        dense = bench_dense(model, input_ids, max_new_tokens=args.max_new_tokens,
                            warmup=args.warmup, n_repeat=args.n_repeat, device=device)
        if dense["status"] == "OOM":
            print(f"    [dense] OOM at L={L} (KV cache too large for one card)",
                  flush=True)
        else:
            print(f"    [dense] prefill={dense['prefill_s']:.3f}s  "
                  f"{dense['ms_per_tok_median']:.2f} ms/tok  "
                  f"p95={dense['p95_ms']:.2f} ms  {dense['tok_s']:.1f} tok/s  "
                  f"peak={dense['peak_gb']:.1f}GB", flush=True)

        # --- CoMem (constant read pack -> never OOMs) ---
        comem = bench_comem(qc, tokenizer, input_ids, chunk_size=args.chunk_size,
                            topk=args.topk, selector=args.selector,
                            sink_tokens=args.sink_tokens,
                            max_new_tokens=args.max_new_tokens, warmup=args.warmup,
                            n_repeat=args.n_repeat, device=device)
        print(f"    [comem] prefill={comem['prefill_s']:.3f}s  "
              f"{comem['ms_per_tok_median']:.2f} ms/tok  "
              f"p95={comem['p95_ms']:.2f} ms  {comem['tok_s']:.1f} tok/s  "
              f"read_len={comem['read_len']}  peak={comem['peak_gb']:.1f}GB",
              flush=True)

        if dense["status"] == "OOM":
            speedup = None
        else:
            speedup = (dense["ms_per_tok_median"] / comem["ms_per_tok_median"]
                       if comem["ms_per_tok_median"] > 0 else None)
            print(f"    >>> decode speedup (dense/comem) = {speedup:.2f}x", flush=True)

        rows.append({
            "length": label,
            "n_tokens": L,
            "dense_status": dense["status"],
            "dense_ms_per_tok_median": (None if dense["status"] == "OOM"
                                        else dense["ms_per_tok_median"]),
            "comem_ms_per_tok_median": comem["ms_per_tok_median"],
            "dense_p95": None if dense["status"] == "OOM" else dense["p95_ms"],
            "comem_p95": comem["p95_ms"],
            "dense_tok_s": None if dense["status"] == "OOM" else dense["tok_s"],
            "comem_tok_s": comem["tok_s"],
            "decode_speedup": speedup,
            "prefill_dense_s": None if dense["status"] == "OOM" else dense["prefill_s"],
            "prefill_comem_s": comem["prefill_s"],
            "correctness_max_logit_diff": corr["max_logit_diff"],
            "tokens_match": corr["tokens_match"],
            "comem_read_len": comem["read_len"],
            "dense_peak_gb": None if dense["status"] == "OOM" else dense["peak_gb"],
            "comem_peak_gb": comem["peak_gb"],
        })

        del input_ids
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- markdown table ---
    print("\n" + "=" * 80)
    print("DECODE-LATENCY RESULTS  (median ms/tok over "
          f"{args.n_repeat} repeats; both KV cache; EOS disabled)")
    print("=" * 80)
    print("| Length | Dense ms/tok | CoMem ms/tok | Speedup | Correctness(max diff) |")
    print("|--------|--------------|--------------|---------|-----------------------|")
    for r in rows:
        dense_ms = "OOM" if r["dense_status"] == "OOM" else f"{r['dense_ms_per_tok_median']:.2f}"
        comem_ms = f"{r['comem_ms_per_tok_median']:.2f}"
        sp = "-" if r["decode_speedup"] is None else f"{r['decode_speedup']:.2f}x"
        md = "-" if r["correctness_max_logit_diff"] is None else f"{r['correctness_max_logit_diff']:.2e}"
        print(f"| {r['length']:>6} | {dense_ms:>12} | {comem_ms:>12} | "
              f"{sp:>7} | {md:>21} |")
    print("=" * 80)

    result = {
        "config": {
            "model_path": model_path,
            "resume_j": args.resume_j,
            "topk": args.topk,
            "chunk_size": args.chunk_size,
            "selector": args.selector,
            "sink_tokens": args.sink_tokens,
            "dtype": args.dtype,
            "attn_impl": args.attn_impl,
            "max_new_tokens": args.max_new_tokens,
            "warmup": args.warmup,
            "n_repeat": args.n_repeat,
            "gate_new_tokens": args.gate_new_tokens,
            "tol": tol,
            "num_layers": n_layers,
            "vocab_size": vocab,
            "gpu_name": gpu_name,
            "device": str(device),
            "eos_disabled": True,
            "lora": None,
        },
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[bench] wrote JSON -> {out_path}")


if __name__ == "__main__":
    main()
