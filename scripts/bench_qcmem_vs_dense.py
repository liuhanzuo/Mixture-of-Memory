#!/usr/bin/env python
"""QCMem (optimised) vs Dense Qwen3-8B — head-to-head speed + accuracy benchmark.

Three things in one command (`--mode`):

1. ``profile`` — QCMem per-phase profile. Times the WRITE phase run SERIALLY
   (per-chunk ``write_chunk`` loop, the pre-optimisation baseline) vs BATCHED
   (``write_chunks``, equal-length chunks stacked into one bottom-band forward),
   plus the READ prefill (``read_prefill``, layers[j:] over the pack once) and the
   per-step DECODE (resumed-band KV cache ``decode_step``). Reports each phase's
   wall time + the WRITE batching speedup. Isolates where QCMem spends time and
   quantifies the Part-A optimisation.

2. ``speed`` — Dense vs QCMem prefill time, decode tok/s, peak GPU memory, at
   each ``--context_lengths`` bucket. Dense = stock ``model.generate`` over the
   FULL context with a standard KV cache; it OOMs / exceeds the RoPE window past
   ~32-64k on an H20 — recorded honestly as ``OOM``. QCMem stays constant (fixed
   ``topk`` read pack) at every length.

3. ``accuracy`` — RULER ``niah_single`` recall (official ``string_match_all``)
   for Dense vs QCMem on ``--n_acc`` samples at each length Dense can run, proving
   QCMem matches Dense inside the window while remaining usable past it.

``both`` / ``all`` run everything.

Examples
--------
# CPU smoke (tiny random Qwen3, no weights) — plumbing correctness:
    python scripts/bench_qcmem_vs_dense.py --mode all --tiny --device cpu

# Full Qwen3-8B head-to-head (speed + accuracy), H20 / L20A:
    python scripts/bench_qcmem_vs_dense.py --mode all \
        --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --resume_j 12 --topk 12 --chunk_size 512 --selector bm25 \
        --context_lengths 8k 16k 32k 64k 128k \
        --acc_lengths 8k 16k 32k --n_acc 30 \
        --max_new_tokens 32 --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.memory.qcmem.qcmem_model import QCMemModel  # noqa: E402
import scripts.eval_qcmem_babilong as qcb  # noqa: E402
from scripts.bench_qcmem_decode import build_tiny_qwen3, _TinyTok  # noqa: E402

qcmem_generate = qcb.qcmem_generate

_LEN_TOKENS = {"1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192,
               "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_gb(device):
    return (torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda" else 0.0)


def _is_oom(err: Exception) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or (
        isinstance(err, RuntimeError) and "out of memory" in str(err).lower())


def _cleanup(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _bare_question(prompt: str) -> str:
    """Trailing question line = the bm25 lexical query (RULER template boundary)."""
    return prompt[prompt.rfind("\n") + 1:].strip()


# RULER task aliases -> canonical ids used by eval_ruler_mem_space._build_sample.
_TASK_ALIASES = {
    "niah_single": "niah_single_2",       # realistic PG19-prose haystack
    "niah_single_noise": "niah_single_1",
    "niah_single_essay": "niah_single_2",
    "niah_multi": "niah_multikey_1",
    "niah_multikey": "niah_multikey_1",
    "vt": "variable_tracking",
}
_CANON_TASKS = {"niah_single_1", "niah_single_2", "niah_multikey_1",
                "variable_tracking"}


def _resolve_task(name: str) -> str:
    if name in _CANON_TASKS:
        return name
    if name in _TASK_ALIASES:
        return _TASK_ALIASES[name]
    raise ValueError(f"unknown RULER task/alias {name!r}; "
                     f"canonical={sorted(_CANON_TASKS)} aliases={sorted(_TASK_ALIASES)}")


# --------------------------------------------------------------------------- #
# Dense: stock model.generate over the full context (standard KV cache decode)
# --------------------------------------------------------------------------- #
def _dense_time(model, tokenizer, input_ids, *, max_new_tokens, device):
    """Prefill time, decode s/step + tok/s, peak mem via the (N vs 1)-step delta.

    ``t_1`` = generate(max_new_tokens=1) = prefill + 1 decode step; ``t_N`` =
    generate(max_new_tokens=N). decode/step = (t_N - t_1)/(N-1); prefill ~= t_1
    minus one decode step. Returns a dict, or ``{"oom": True}`` on CUDA OOM."""
    gen_kwargs = dict(do_sample=False, num_beams=1,
                      pad_token_id=(tokenizer.pad_token_id
                                    if tokenizer.pad_token_id is not None else 0))

    def once(mnt):
        _sync(device)
        t0 = time.perf_counter()
        model.generate(input_ids, max_new_tokens=mnt, min_new_tokens=mnt,
                        **gen_kwargs)
        _sync(device)
        return time.perf_counter() - t0

    try:
        once(1)                       # warmup
        _reset_peak(device)
        t1 = once(1)
        tN = once(max_new_tokens)
    except Exception as e:  # noqa: BLE001
        if _is_oom(e):
            _cleanup(device)
            return {"oom": True}
        raise
    steps = max(1, max_new_tokens - 1)
    dps = (tN - t1) / steps
    prefill = max(0.0, t1 - dps)
    return {"oom": False, "prefill_s": prefill, "decode_s_per_step": dps,
            "decode_tok_s": (1.0 / dps if dps > 0 else 0.0),
            "peak_gb": _peak_gb(device)}


@torch.no_grad()
def _dense_answer(model, tokenizer, input_ids, *, max_new_tokens, device):
    """Greedy dense generation -> decoded string (or None on OOM)."""
    try:
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=1,
            pad_token_id=(tokenizer.pad_token_id
                          if tokenizer.pad_token_id is not None else 0))
    except Exception as e:  # noqa: BLE001
        if _is_oom(e):
            _cleanup(device)
            return None
        raise
    gen = out[0, input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# QCMem: qcmem_generate (batched write + read_prefill + KV-cache decode)
# --------------------------------------------------------------------------- #
def _qcmem_time(qc, tokenizer, input_ids, *, chunk_size, topk, selector,
                sink_tokens, max_new_tokens, device, bare_q_ids):
    def once(mnt, stats=None):
        _sync(device)
        t0 = time.perf_counter()
        qcmem_generate(qc=qc, tokenizer=tokenizer, input_ids=input_ids,
                       chunk_size=chunk_size, max_new_tokens=mnt,
                       selector=selector, topk=topk, sink_tokens=sink_tokens,
                       bare_question_ids=bare_q_ids, use_kv_cache=True, stats=stats)
        _sync(device)
        return time.perf_counter() - t0

    st = {}
    try:
        once(1, stats=st)             # warmup + capture read_len
        _reset_peak(device)
        t1 = once(1)
        tN = once(max_new_tokens)
    except Exception as e:  # noqa: BLE001
        if _is_oom(e):
            _cleanup(device)
            return {"oom": True}
        raise
    steps = max(1, max_new_tokens - 1)
    dps = (tN - t1) / steps
    prefill = max(0.0, t1 - dps)
    return {"oom": False, "prefill_s": prefill, "decode_s_per_step": dps,
            "decode_tok_s": (1.0 / dps if dps > 0 else 0.0),
            "peak_gb": _peak_gb(device), "read_len": st.get("read_len")}


@torch.no_grad()
def _qcmem_answer(qc, tokenizer, input_ids, *, chunk_size, topk, selector,
                  sink_tokens, max_new_tokens, bare_q_ids):
    return qcmem_generate(
        qc=qc, tokenizer=tokenizer, input_ids=input_ids, chunk_size=chunk_size,
        max_new_tokens=max_new_tokens, selector=selector, topk=topk,
        sink_tokens=sink_tokens, bare_question_ids=bare_q_ids, use_kv_cache=True)


# --------------------------------------------------------------------------- #
# 1) profile: QCMem per-phase timing (write serial vs batched, read, decode)
# --------------------------------------------------------------------------- #
def run_profile(qc, tokenizer, *, lengths, chunk_size, topk, device, vocab,
                decode_steps=16):
    print("=" * 78)
    print(f"QCMem per-phase profile  (resume_j={qc.resume_j}, topk={topk}, "
          f"chunk_size={chunk_size})")
    print("=" * 78)
    hdr = (f"{'ctx':>6} | {'#ctx_ch':>7} | {'write_serial':>12} | "
           f"{'write_batch':>11} | {'w_speedup':>9} | {'read_prefill':>12} | "
           f"{'decode/step':>11}")
    print(hdr)
    print("-" * len(hdr))
    g = torch.Generator(device="cpu").manual_seed(0)
    bos = tokenizer.bos_token_id or 1
    for L in lengths:
        n_tok = _LEN_TOKENS.get(L, None) or int(L)
        ids = torch.randint(2, vocab, (1, n_tok), generator=g).to(device)
        chunks = list(ids[0].split(chunk_size))
        context_chunks = chunks[:-1]
        query_chunk = chunks[-1]
        # select topk by recency (forward-free) to isolate write cost of the pack.
        sel = context_chunks[-topk:] if topk < len(context_chunks) else context_chunks

        qc.write_chunk(sel[0])        # warmup (lazy alloc / kernel autotune)
        _sync(device)

        # --- WRITE serial (per-chunk write_chunk loop, pre-optimisation) ---
        _sync(device)
        t0 = time.perf_counter()
        _ = [qc.write_chunk(c) for c in sel]
        _sync(device)
        t_serial = time.perf_counter() - t0

        # --- WRITE batched (write_chunks) ---
        _sync(device)
        t0 = time.perf_counter()
        sel_hj = qc.write_chunks(sel)
        _sync(device)
        t_batch = time.perf_counter() - t0

        sink_hj = qc.write_chunk([bos])
        q_hj = qc.write_chunk(query_chunk)

        # --- READ prefill (layers[j:] over the pack once, KV cached) ---
        _sync(device)
        t0 = time.perf_counter()
        logits1, top_cache, pack_pos = qc.read_prefill(sink_hj, sel_hj, q_hj)
        _sync(device)
        t_read = time.perf_counter() - t0

        # --- bottom-band prefill for decode + per-step DECODE ---
        _, bottom_cache, q_local = qc.write_prefill(query_chunk.tolist())
        tok = int(logits1[0, -1].argmax())
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(decode_steps):
            lg = qc.decode_step(tok, bottom_cache, top_cache, q_local, pack_pos)
            q_local += 1
            pack_pos += 1
            tok = int(lg[0, -1].argmax())
        _sync(device)
        t_dec = (time.perf_counter() - t0) / decode_steps

        sp = t_serial / t_batch if t_batch > 0 else float("nan")
        print(f"{L:>6} | {len(context_chunks):>7} | {t_serial*1e3:>10.1f}ms | "
              f"{t_batch*1e3:>9.1f}ms | {sp:>8.1f}x | {t_read*1e3:>10.1f}ms | "
              f"{t_dec*1e3:>9.1f}ms")
        _cleanup(device)


# --------------------------------------------------------------------------- #
# 2) speed: Dense vs QCMem prefill / decode-tok-s / peak-mem per length
# --------------------------------------------------------------------------- #
def run_speed(model, qc, tokenizer, *, lengths, chunk_size, topk, selector,
              sink_tokens, max_new_tokens, device, vocab):
    print("=" * 78)
    print(f"Dense vs QCMem SPEED  (resume_j={qc.resume_j}, topk={topk}, "
          f"chunk_size={chunk_size}, max_new_tokens={max_new_tokens})")
    print("=" * 78)
    hdr = (f"{'ctx':>6} | {'D_prefill':>10} {'D_tok/s':>8} {'D_mem':>7} | "
           f"{'Q_prefill':>10} {'Q_tok/s':>8} {'Q_mem':>7} {'read':>6} | "
           f"{'decode_x':>8}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for L in lengths:
        n_tok = _LEN_TOKENS.get(L, None) or int(L)
        ids = torch.randint(2, vocab, (1, n_tok), generator=g).to(device)
        bare_q = ids[0].tolist()[-8:]

        d = _dense_time(model, tokenizer, ids, max_new_tokens=max_new_tokens,
                        device=device)
        _cleanup(device)
        q = _qcmem_time(qc, tokenizer, ids, chunk_size=chunk_size, topk=topk,
                        selector=selector, sink_tokens=sink_tokens,
                        max_new_tokens=max_new_tokens, device=device,
                        bare_q_ids=bare_q)
        _cleanup(device)

        if d.get("oom"):
            dstr = f"{'OOM':>10} {'--':>8} {'--':>7}"
            d_tps = None
        else:
            dstr = (f"{d['prefill_s']:>9.2f}s {d['decode_tok_s']:>8.1f} "
                    f"{d['peak_gb']:>6.1f}G")
            d_tps = d["decode_tok_s"]
        if q.get("oom"):
            qstr = f"{'OOM':>10} {'--':>8} {'--':>7} {'--':>6}"
            q_tps = None
        else:
            qstr = (f"{q['prefill_s']:>9.2f}s {q['decode_tok_s']:>8.1f} "
                    f"{q['peak_gb']:>6.1f}G {str(q['read_len']):>6}")
            q_tps = q["decode_tok_s"]
        if d_tps and q_tps:
            spd = f"{q_tps / d_tps:>7.1f}x"
        else:
            spd = f"{'--':>8}"
        print(f"{L:>6} | {dstr} | {qstr} | {spd}")
        rows.append((L, d, q))
    return rows


# --------------------------------------------------------------------------- #
# 3) accuracy: RULER niah_single string_match_all, Dense vs QCMem
# --------------------------------------------------------------------------- #
def run_accuracy(model, qc, tokenizer, *, lengths, chunk_size, topk, selector,
                 sink_tokens, n_acc, max_new_tokens, device, base_seed, task):
    import scripts.eval_ruler_mem_space as ruler  # local import (needs real tok)
    print("=" * 78)
    print(f"Dense vs QCMem ACCURACY  RULER {task}  string_match_all  "
          f"(n={n_acc}, selector={selector})")
    print("=" * 78)
    hdr = f"{'ctx':>6} | {'Dense_acc':>10} | {'QCMem_acc':>10} | {'n':>4}"
    print(hdr)
    print("-" * len(hdr))
    rt = _resolve_task(task)
    for L in lengths:
        target = ruler._LENGTH_TOKENS[L] if L in ruler._LENGTH_TOKENS \
            else _LEN_TOKENS.get(L, int(L))
        d_sum = q_sum = 0.0
        n_done = 0
        d_oom = False
        for k in range(n_acc):
            rng = random.Random(base_seed + k)
            prompt, answers, gold = ruler._build_sample(rt, target, tokenizer,
                                                        rng, None)
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = enc.input_ids.to(device)
            bare_q = tokenizer(_bare_question(prompt),
                               add_special_tokens=False).input_ids
            # Dense
            if not d_oom:
                d_out = _dense_answer(model, tokenizer, input_ids,
                                      max_new_tokens=max_new_tokens, device=device)
                if d_out is None:
                    d_oom = True
                else:
                    d_sum += ruler._string_match_all_one(d_out, answers)
            _cleanup(device)
            # QCMem
            q_out = _qcmem_answer(qc, tokenizer, input_ids, chunk_size=chunk_size,
                                  topk=topk, selector=selector,
                                  sink_tokens=sink_tokens,
                                  max_new_tokens=max_new_tokens, bare_q_ids=bare_q)
            q_sum += ruler._string_match_all_one(q_out, answers)
            n_done += 1
            _cleanup(device)
        d_acc = ("OOM" if d_oom else f"{100.0 * d_sum / n_done:.1f}")
        q_acc = f"{100.0 * q_sum / n_done:.1f}"
        print(f"{L:>6} | {d_acc:>10} | {q_acc:>10} | {n_done:>4}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["profile", "speed", "accuracy", "both",
                                       "all"], default="all")
    ap.add_argument("--model_path", type=str, default="")
    ap.add_argument("--tiny", action="store_true",
                    help="Tiny random Qwen3 (no weights) for CPU plumbing smoke.")
    ap.add_argument("--tiny_layers", type=int, default=6)
    ap.add_argument("--tiny_hidden", type=int, default=64)
    ap.add_argument("--tiny_vocab", type=int, default=256)
    ap.add_argument("--resume_j", type=int, default=12)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--selector", type=str, default="bm25")
    ap.add_argument("--sink_tokens", type=str, default="bos", choices=["bos", "none"])
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--context_lengths", type=str, nargs="+",
                    default=["8k", "16k", "32k", "64k", "128k"])
    ap.add_argument("--acc_lengths", type=str, nargs="+",
                    default=["8k", "16k", "32k"])
    ap.add_argument("--acc_task", type=str, default="niah_single")
    ap.add_argument("--n_acc", type=int, default=30)
    ap.add_argument("--acc_seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    args = ap.parse_args()

    want_cuda = args.device.startswith("cuda")
    if want_cuda and not torch.cuda.is_available():
        print("[bench] cuda requested but unavailable -> cpu")
        args.device = "cpu"
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    if args.tiny or not args.model_path:
        print("[bench] tiny random Qwen3 (fp32) — plumbing smoke only")
        model, cfg = build_tiny_qwen3(args.tiny_layers, args.tiny_hidden,
                                      args.tiny_vocab)
        model = model.to(device=device, dtype=torch.float32).eval()
        vocab = args.tiny_vocab
        tokenizer = _TinyTok(vocab)
        L = args.tiny_layers
        resume_j = max(0, min(args.resume_j if args.resume_j <= L else L // 2, L))
        chunk_size = min(args.chunk_size, 8)
        is_tiny = True
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[bench] loading {args.model_path} dtype={dtype} attn={args.attn_impl}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, attn_implementation=args.attn_impl,
            trust_remote_code=True, local_files_only=True).to(device).eval()
        vocab = int(model.config.vocab_size)
        L = int(model.config.num_hidden_layers)
        resume_j = max(0, min(args.resume_j, L))
        chunk_size = args.chunk_size
        is_tiny = False

    qc = QCMemModel(model, resume_j=resume_j)
    print(f"[bench] backbone L={L} resume_j={resume_j} chunk_size={chunk_size} "
          f"topk={args.topk} device={device}")

    speed_lengths = ["1k", "2k"] if is_tiny else args.context_lengths
    acc_lengths = ["1k"] if is_tiny else args.acc_lengths

    do_profile = args.mode in ("profile", "all")
    do_speed = args.mode in ("speed", "both", "all")
    do_acc = args.mode in ("accuracy", "both", "all")

    if do_profile:
        run_profile(qc, tokenizer, lengths=speed_lengths, chunk_size=chunk_size,
                    topk=args.topk, device=device, vocab=vocab,
                    decode_steps=min(16, args.max_new_tokens))

    if do_speed:
        run_speed(model, qc, tokenizer, lengths=speed_lengths,
                  chunk_size=chunk_size, topk=args.topk, selector=args.selector,
                  sink_tokens=args.sink_tokens, max_new_tokens=args.max_new_tokens,
                  device=device, vocab=vocab)

    if do_acc:
        if is_tiny:
            print("[bench] accuracy skipped for tiny model (needs a real tokenizer "
                  "+ RULER haystack corpus)")
        else:
            run_accuracy(model, qc, tokenizer, lengths=acc_lengths,
                         chunk_size=chunk_size, topk=args.topk,
                         selector=args.selector, sink_tokens=args.sink_tokens,
                         n_acc=args.n_acc, max_new_tokens=args.max_new_tokens,
                         device=device, base_seed=args.acc_seed, task=args.acc_task)


if __name__ == "__main__":
    main()
