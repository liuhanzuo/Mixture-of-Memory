#!/usr/bin/env python
"""QCMem vs full-context — end-to-end latency + peak-memory benchmark.

This is the core tool for answering "how much faster is QCMem than a full-context
forward, and by how much does the gap grow with context length?".

------------------------------------------------------------------------------- #
WHAT IS MEASURED (per context length L in {8k, 16k, 32k, 64k, 128k, 256k})
------------------------------------------------------------------------------- #
Two methods, on the SAME synthetic ``[1, L]`` token sequence (random ids — we are
timing compute, not accuracy):

  full-context
    * prefill_s : one forward over the whole L-token sequence (``use_cache=True``)
                  down to the last-position logits — i.e. the time until the model
                  can emit the first answer token. Attention is O(L^2).  OOM at
                  large L is caught and recorded as "OOM".
    * decode_s  : greedy-generate ``n_decode`` tokens from the prefill KV-cache
                  (each step attends over the full growing cache).
    * seq_len   : L  (the sequence the top layers actually attend over).

  QCMem  (mid-depth resume;  resume_j / chunk_size / topk configurable)
    * prefill_s : END-TO-END ingest+answer time, NOT just the read:
                    1. WRITE all N = ceil(L/chunk_size) context chunks — each chunk
                       runs embed + layers[0:j] chunk-locally (this is the O(N)=O(L)
                       linear ingestion cost, the thing that should grow linearly).
                    2. bm25 selector picks the topk context chunks (pure CPU).
                    3. WRITE the query chunk, then READ once: pack
                       [sink ; topk*chunk h_j ; query h_j] and recompute layers[j:]
                       -> logits. The read sequence length is FIXED
                       ~ sink(1) + topk*chunk + query_len, INDEPENDENT of L.
    * decode_s  : greedy-generate ``n_decode`` tokens the way the real QCMem eval
                  does (``scripts/eval_qcmem_babilong.py::qcmem_generate``): each
                  step re-encodes the growing query chunk (write_chunk) and re-reads
                  the fixed pack. No KV-cache in the read path (faithful to eval).
    * read_len  : the fixed packed sequence the read's top layers attend over.

Peak memory = ``torch.cuda.max_memory_allocated`` (includes the resident bf16
weights ~16 GB), reported as the max over the prefill and decode phases.

Each (length, method) is timed ``n_repeat`` times after 1 un-counted warmup; the
MEDIAN is reported. Speedup = full_ctx_prefill_s / qcmem_prefill_s.

------------------------------------------------------------------------------- #
NOTES / CAVEATS
------------------------------------------------------------------------------- #
* Faithful to the eval path we follow ``eval_qcmem_babilong.py`` exactly:
  per-chunk ``qc.write_chunk`` (one forward per chunk) and ``qc.read`` returning
  the FULL ``[1,|H|,V]`` logit tensor. Both are honest deploy costs.
* We WRITE ALL N context chunks (per the benchmark spec) so the write phase is the
  true O(L) ingestion cost. The real bm25 deploy path can select topk from tokens
  FIRST and write only those (constant write) — that would make QCMem prefill
  ~constant; we intentionally show the linear write-all number here.
* This script references ``eval_qcmem_babilong.py`` / ``run_babilong_mem_space.py``
  / ``qcmem_model.py`` but does NOT modify them. The bm25 selector below is
  inlined VERBATIM from ``run_babilong_mem_space.py::_bm25_scores`` (same formula
  + k1/b) so the benchmark stays self-contained and does not drag in the heavy
  babilong / datasets / mem_space import chain (a speed test needs none of it).
* LoRA (``--lora_adapter``) is optional and does NOT change the latency profile
  (LoRA deltas are fused into the same Linear calls); speed runs use the plain
  base by default. If passed, it is loaded onto the frozen backbone.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJECT_ROOT,
          os.path.join(PROJECT_ROOT, "scripts"),
          os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.memory.qcmem import QCMemModel  # noqa: E402


# --------------------------------------------------------------------------- #
# BM25 selector — inlined VERBATIM from
# ``scripts/run_babilong_mem_space.py::_bm25_scores`` (formula + k1/b defaults).
# Pure CPU, no model forward. Kept local so the benchmark does not import the
# heavy babilong/datasets/mem_space chain the harness pulls in.
# --------------------------------------------------------------------------- #
def _bm25_scores(docs, query_ids, k1: float = 1.5, b: float = 0.75):
    """BM25 of ``query_ids`` (list[int]) against each candidate document's token
    IDs. Corpus == the candidate pool ``docs``; IDF is over that pool. Query
    terms are de-duplicated. Returns list[float] of length ``len(docs)``."""
    N = len(docs)
    if N <= 0:
        return None
    df = Counter()
    doc_tf = []
    doc_len = []
    for d in docs:
        c = Counter(d)
        doc_tf.append(c)
        doc_len.append(len(d))
        for t in c:
            df[t] += 1
    avgdl = (sum(doc_len) / N) if N > 0 else 0.0
    idf = {t: math.log((N - dft + 0.5) / (dft + 0.5) + 1.0) for t, dft in df.items()}
    qterms = set(int(t) for t in query_ids)
    scores = []
    for i in range(N):
        tf = doc_tf[i]
        dl = doc_len[i]
        s = 0.0
        for t in qterms:
            f = tf.get(t, 0)
            if f == 0:
                continue
            it = idf.get(t, 0.0)
            if avgdl > 0:
                denom = f + k1 * (1.0 - b + b * dl / avgdl)
            else:
                denom = f + k1
            s += it * (f * (k1 + 1.0)) / denom
        scores.append(s)
    return scores


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


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


# --------------------------------------------------------------------------- #
# full-context method
# --------------------------------------------------------------------------- #
def bench_fullctx(model, input_ids, n_decode, n_repeat, warmup, device):
    """Return dict with prefill_s / decode_s (median) / peak_gb / seq_len, or
    {'status': 'OOM'} if the full-context forward runs out of memory."""
    prefill_times, decode_times = [], []
    peak = 0.0
    L = int(input_ids.shape[1])
    try:
        for it in range(warmup + n_repeat):
            # ---- prefill: one forward over the whole L-token sequence ----
            torch.cuda.reset_peak_memory_stats()
            _sync()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model(input_ids=input_ids, use_cache=True)
            _sync()
            t1 = time.perf_counter()
            past = out.past_key_values
            first = out.logits[:, -1:, :].argmax(dim=-1)  # [1,1]
            p_peak = _peak_gb()
            del out

            # ---- decode: n_decode greedy steps from the prefill KV-cache ----
            torch.cuda.reset_peak_memory_stats()
            _sync()
            t2 = time.perf_counter()
            past_len = L
            cur = first
            with torch.inference_mode():
                for _ in range(n_decode):
                    cache_pos = torch.arange(past_len, past_len + 1, device=device)
                    o = model(input_ids=cur, past_key_values=past,
                              use_cache=True, cache_position=cache_pos)
                    past = o.past_key_values
                    cur = o.logits[:, -1:, :].argmax(dim=-1)
                    past_len += 1
            _sync()
            t3 = time.perf_counter()
            d_peak = _peak_gb()

            if it >= warmup:
                prefill_times.append(t1 - t0)
                decode_times.append(t3 - t2)
                peak = max(peak, p_peak, d_peak)

            del past, first, cur
            torch.cuda.empty_cache()
    except RuntimeError as e:
        if not _is_oom(e):
            raise
        torch.cuda.empty_cache()
        return {"status": "OOM", "seq_len": L}

    return {
        "status": "ok",
        "prefill_s": statistics.median(prefill_times),
        "decode_s": statistics.median(decode_times),
        "peak_gb": peak,
        "seq_len": L,
    }


# --------------------------------------------------------------------------- #
# QCMem method
# --------------------------------------------------------------------------- #
def bench_qcmem(qc, tokenizer, input_ids, chunk_size, topk,
                n_decode, n_repeat, warmup, device):
    """QCMem end-to-end: write all N chunks + bm25 select + read (prefill),
    then n_decode faithful decode steps. Returns the same dict schema."""
    tokens = input_ids[0]
    chunks = list(tokens.split(chunk_size))
    if len(chunks) < 2:
        raise ValueError("need at least one context chunk + one query chunk")
    context_chunks = chunks[:-1]
    query_chunk = chunks[-1]
    n_ctx = len(context_chunks)

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = int(tokens[0].item())
    query_tok_list = query_chunk.tolist()          # bm25 query = the query chunk
    docs = [c.tolist() for c in context_chunks]     # bm25 corpus (built once)

    prefill_times, decode_times = [], []
    peak = 0.0
    read_len = None
    write_calls = None

    for it in range(warmup + n_repeat):
        # ---- prefill: WRITE all N chunks + bm25 select + query write + read ----
        torch.cuda.reset_peak_memory_stats()
        _sync()
        t0 = time.perf_counter()

        sink_hj = qc.write_chunk([bos_id])
        all_hj = [qc.write_chunk(c) for c in context_chunks]   # O(N) = O(L) writes

        scores = _bm25_scores(docs, query_tok_list)
        k = max(0, int(topk))
        if scores:
            order = sorted(range(n_ctx), key=lambda i: scores[i], reverse=True)
            sel_idx = sorted(order[:k])
        else:
            sel_idx = list(range(max(0, n_ctx - k), n_ctx))
        selected_hj = [all_hj[i] for i in sel_idx]

        q_hj = qc.write_chunk(query_tok_list)
        logits = qc.read(sink_hj, selected_hj, q_hj)           # [1,|H|,V] full
        first_tok = int(logits[0, -1].float().argmax().item())
        _sync()
        t1 = time.perf_counter()
        p_peak = _peak_gb()

        read_len = (1 + sum(int(h.shape[1]) for h in selected_hj)
                    + int(q_hj.shape[1]))
        write_calls = 1 + n_ctx + 1  # sink + all context + query

        del logits

        # ---- decode: n_decode faithful steps (re-encode growing query + read) ----
        torch.cuda.reset_peak_memory_stats()
        _sync()
        t2 = time.perf_counter()
        query_ids = query_tok_list + [first_tok]
        for _ in range(n_decode):
            q_hj_d = qc.write_chunk(query_ids)
            logits_d = qc.read(sink_hj, selected_hj, q_hj_d)
            nt = int(logits_d[0, -1].float().argmax().item())
            query_ids.append(nt)
            del q_hj_d, logits_d
        _sync()
        t3 = time.perf_counter()
        d_peak = _peak_gb()

        if it >= warmup:
            prefill_times.append(t1 - t0)
            decode_times.append(t3 - t2)
            peak = max(peak, p_peak, d_peak)

        del sink_hj, all_hj, selected_hj, q_hj
        torch.cuda.empty_cache()

    return {
        "status": "ok",
        "prefill_s": statistics.median(prefill_times),
        "decode_s": statistics.median(decode_times),
        "peak_gb": peak,
        "seq_len": read_len,
        "n_chunks": n_ctx + 1,
        "write_calls": write_calls,
    }


# --------------------------------------------------------------------------- #
# table rendering
# --------------------------------------------------------------------------- #
def _fmt(v, spec="{:.3f}"):
    return "OOM" if v is None else spec.format(v)


def render_table(rows):
    """rows: list of dicts with length_label, and full/qc sub-dicts."""
    header = (
        f"{'len':>6} | {'method':<10} | {'seq_len':>8} | "
        f"{'prefill_s':>10} | {'decode_s':>9} | {'peak_gb':>8} | {'speedup':>13}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        fc = r["full_ctx"]
        qc = r["qcmem"]
        # full-context row
        if fc.get("status") == "OOM":
            lines.append(
                f"{r['length_label']:>6} | {'full-ctx':<10} | {fc['seq_len']:>8d} | "
                f"{'OOM':>10} | {'OOM':>9} | {'OOM':>8} | {'-':>13}"
            )
        else:
            lines.append(
                f"{r['length_label']:>6} | {'full-ctx':<10} | {fc['seq_len']:>8d} | "
                f"{_fmt(fc['prefill_s']):>10} | {_fmt(fc['decode_s']):>9} | "
                f"{_fmt(fc['peak_gb'], '{:.1f}'):>8} | {'1.00x':>13}"
            )
        # QCMem row
        speedup = r.get("speedup")
        sp_str = ("inf(full OOM)" if speedup == "inf"
                  else (f"{speedup:.2f}x" if isinstance(speedup, float) else "-"))
        lines.append(
            f"{r['length_label']:>6} | {'qcmem':<10} | {qc['seq_len']:>8d} | "
            f"{_fmt(qc['prefill_s']):>10} | {_fmt(qc['decode_s']):>9} | "
            f"{_fmt(qc['peak_gb'], '{:.1f}'):>8} | {sp_str:>13}"
        )
        lines.append("-" * len(header))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="QCMem vs full-context latency + peak-memory benchmark")
    ap.add_argument("--model_path", type=str, required=True,
                    help="Absolute path to the (Qwen3/Llama) backbone weights.")
    ap.add_argument("--resume_j", type=int, default=12,
                    help="QCMem mid-depth split j (bottom layers run at write).")
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--topk", type=int, default=12,
                    help="Number of context chunks packed into the read.")
    ap.add_argument("--lengths", type=str, nargs="+",
                    default=["8k", "16k", "32k", "64k", "128k"])
    ap.add_argument("--n_repeat", type=int, default=3,
                    help="Timed repeats (median reported); 1 warmup is un-counted.")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--n_decode", type=int, default=20,
                    help="Number of tokens to greedily decode for decode_s.")
    ap.add_argument("--lora_adapter", type=str, default="",
                    help="Optional LoRA dir (does not change the speed profile).")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default="",
                    help="Path to write the results JSON.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print("=" * 78)
    print("QCMem vs full-context benchmark")
    print(f"  model_path = {args.model_path}")
    print(f"  resume_j={args.resume_j} chunk_size={args.chunk_size} topk={args.topk}")
    print(f"  dtype={dtype} attn_impl={args.attn_impl} device={device}")
    print(f"  n_repeat={args.n_repeat} warmup={args.warmup} n_decode={args.n_decode}")
    print("=" * 78, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()

    if args.lora_adapter:
        from peft import PeftModel
        print(f"[bench] loading LoRA adapter: {args.lora_adapter}")
        peft_model = PeftModel.from_pretrained(model, args.lora_adapter).eval()
        model = peft_model.base_model.model

    L_layers = int(model.config.num_hidden_layers)
    vocab = int(model.config.vocab_size)
    if not (0 <= args.resume_j <= L_layers):
        ap.error(f"--resume_j must be in [0, {L_layers}]; got {args.resume_j}")
    qc = QCMemModel(model, resume_j=args.resume_j)
    print(f"[bench] backbone: {L_layers} layers, hidden={model.config.hidden_size}, "
          f"vocab={vocab}", flush=True)

    lengths = [(lab, parse_length(lab)) for lab in args.lengths]

    rows = []
    for label, L in lengths:
        print(f"\n[bench] === length {label} (L={L}) ===", flush=True)
        input_ids = torch.randint(0, vocab, (1, L), device=device)

        # --- full-context ---
        print(f"[bench]   full-context prefill+decode ...", flush=True)
        fc = bench_fullctx(model, input_ids, args.n_decode,
                           args.n_repeat, args.warmup, device)
        if fc.get("status") == "OOM":
            print(f"[bench]   full-context: OOM at L={L}", flush=True)
        else:
            print(f"[bench]   full-context: prefill={fc['prefill_s']:.3f}s "
                  f"decode={fc['decode_s']:.3f}s peak={fc['peak_gb']:.1f}GB",
                  flush=True)

        # --- QCMem ---
        print(f"[bench]   QCMem write-all+select+read+decode ...", flush=True)
        qcm = bench_qcmem(qc, tokenizer, input_ids, args.chunk_size, args.topk,
                          args.n_decode, args.n_repeat, args.warmup, device)
        print(f"[bench]   QCMem: prefill={qcm['prefill_s']:.3f}s "
              f"decode={qcm['decode_s']:.3f}s peak={qcm['peak_gb']:.1f}GB "
              f"read_len={qcm['seq_len']} (write_calls={qcm['write_calls']})",
              flush=True)

        # --- speedup ---
        if fc.get("status") == "OOM":
            speedup = "inf"   # full-ctx cannot even run; QCMem still does
        else:
            speedup = fc["prefill_s"] / qcm["prefill_s"]
            print(f"[bench]   >>> prefill speedup (full/qcmem) = {speedup:.2f}x",
                  flush=True)

        rows.append({
            "length_label": label,
            "L": L,
            "full_ctx": fc,
            "qcmem": qcm,
            "speedup": speedup,
        })

        del input_ids
        torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print("RESULTS TABLE  (median of {} repeats; speedup = full_ctx_prefill / "
          "qcmem_prefill)".format(args.n_repeat))
    print("=" * 78)
    print(render_table(rows))
    print("\nlegend: full-ctx seq_len = L (O(L^2) attention); "
          "qcmem seq_len = FIXED read pack (sink + topk*chunk + query).")
    print("        QCMem prefill = write ALL N chunks (O(L)) + bm25 select + read; "
          "read cost is L-independent.")

    result = {
        "config": {
            "model_path": args.model_path,
            "resume_j": args.resume_j,
            "chunk_size": args.chunk_size,
            "topk": args.topk,
            "dtype": args.dtype,
            "attn_impl": args.attn_impl,
            "n_repeat": args.n_repeat,
            "warmup": args.warmup,
            "n_decode": args.n_decode,
            "num_layers": L_layers,
            "vocab_size": vocab,
            "lora_adapter": args.lora_adapter or None,
        },
        "rows": rows,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[bench] wrote JSON -> {args.output}")


if __name__ == "__main__":
    main()
