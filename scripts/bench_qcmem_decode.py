#!/usr/bin/env python
"""QCMem resumed-band KV-cache decode — benchmark + correctness gate.

Two things in one command:

1. CORRECTNESS (hard gate). Runs ``qcmem_generate`` twice on the SAME prompt —
   once with the baseline recompute decode (``use_kv_cache=False``: every step
   re-runs layers[0:j] over the whole growing query AND layers[j:L] over the full
   ~6.7k read pack) and once with the resumed-band KV cache
   (``use_kv_cache=True``: prefill both bands once, then push ONE token/step) —
   and asserts the per-step next-token logits are identical: top-1 token EXACTLY
   equal at every step and max|logit diff| < tol. Speed must not change the
   output. Runs on a tiny random Qwen3 on CPU (fp32, tol 1e-4) with no weights, so
   it is runnable anywhere; also runnable on the real Qwen3-8B (bf16, tol 1e-2).

2. BENCHMARK. Loads a backbone, builds a synthetic long prompt at each
   ``--context_lengths`` bucket, and measures prefill+decode wall time, per-step
   decode latency / tok-s, and peak GPU memory for BOTH the baseline recompute
   decode and the KV-cache decode -> reports the decode speedup and memory delta.
   Decode cost is isolated from prefill by timing max_new_tokens = 1 vs N and
   taking the (N-1)-step delta.

Examples
--------
# CPU correctness gate (no GPU, no weights) — tiny random Qwen3:
    python scripts/bench_qcmem_decode.py --mode correctness --tiny --device cpu

# Real Qwen3-8B correctness (bf16) on one GPU:
    python scripts/bench_qcmem_decode.py --mode correctness \
        --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --resume_j 12 --topk 12 --chunk_size 512 --device cuda:0

# Full Qwen3-8B benchmark (prefill/decode tok-s + peak mem), 16k & 64k contexts:
    python scripts/bench_qcmem_decode.py --mode both \
        --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --resume_j 12 --topk 12 --chunk_size 512 \
        --context_lengths 16k 64k --max_new_tokens 64 --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.memory.qcmem.qcmem_model import QCMemModel  # noqa: E402
import scripts.eval_qcmem_babilong as qcb  # noqa: E402

qcmem_generate = qcb.qcmem_generate

_LEN_TOKENS = {"1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192,
               "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072}


# --------------------------------------------------------------------------- #
# tiny random Qwen3 (CPU correctness, no weights needed)
# --------------------------------------------------------------------------- #
def build_tiny_qwen3(n_layers=4, hidden=64, vocab=256, seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=8192,
        sliding_window=None,
        use_sliding_window=False,
        attn_implementation="sdpa",
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(cfg).eval()
    return model, cfg


class _TinyTok:
    """Minimal tokenizer stand-in for the tiny-random correctness path."""
    def __init__(self, vocab):
        self.vocab = vocab
        self.bos_token_id = 1
        self.eos_token_id = None      # disable EOS so both paths decode full budget
        self.pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


# --------------------------------------------------------------------------- #
# correctness A/B: KV-cache decode logits == recompute decode logits
# --------------------------------------------------------------------------- #
def run_correctness(qc, tokenizer, input_ids, *, chunk_size, topk, selector,
                    max_new_tokens, tol, sink_tokens="bos"):
    bare_q_ids = input_ids[0].tolist()[-8:]  # dummy lexical query (bm25 unused here)

    def gen(use_kv):
        stats = {"capture_step_logits": True}
        out = qcmem_generate(
            qc=qc, tokenizer=tokenizer, input_ids=input_ids,
            chunk_size=chunk_size, max_new_tokens=max_new_tokens,
            selector=selector, topk=topk, sink_tokens=sink_tokens,
            bare_question_ids=bare_q_ids, stats=stats, use_kv_cache=use_kv,
        )
        return out, stats

    out_base, st_base = gen(False)
    out_kv, st_kv = gen(True)

    lb, lk = st_base["step_logits"], st_kv["step_logits"]
    gb, gk = st_base["generated_ids"], st_kv["generated_ids"]
    n = min(len(lb), len(lk))
    max_diff = 0.0
    argmax_ok = True
    for s in range(n):
        d = (lb[s] - lk[s]).abs().max().item()
        max_diff = max(max_diff, d)
        if int(lb[s].argmax()) != int(lk[s].argmax()):
            argmax_ok = False
    len_ok = (len(lb) == len(lk))
    tok_ok = (gb == gk)

    print("=" * 72)
    print("QCMem decode correctness: KV-cache vs recompute (per-step logits)")
    print("=" * 72)
    print(f"  steps: recompute={len(lb)}  kv={len(lk)}  (len_ok={len_ok})")
    print(f"  generated token ids identical: {tok_ok}")
    print(f"  top-1 argmax identical every step: {argmax_ok}")
    print(f"  max|logit diff| over {n} steps: {max_diff:.3e}  (tol {tol:.1e})")
    print(f"  recompute out: {out_base[:80]!r}")
    print(f"  kv-cache  out: {out_kv[:80]!r}")
    ok = len_ok and tok_ok and argmax_ok and (max_diff < tol)
    print("-" * 72)
    print(f"CORRECTNESS: {'PASS' if ok else 'FAIL'}")
    print("=" * 72)
    return ok


# --------------------------------------------------------------------------- #
# benchmark: prefill/decode wall time + peak mem for both decode paths
# --------------------------------------------------------------------------- #
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_generate(qc, tokenizer, input_ids, *, chunk_size, topk, selector,
                   max_new_tokens, sink_tokens, use_kv, device):
    bare_q_ids = input_ids[0].tolist()[-8:]

    def once(mnt):
        _sync(device)
        t0 = time.perf_counter()
        qcmem_generate(
            qc=qc, tokenizer=tokenizer, input_ids=input_ids,
            chunk_size=chunk_size, max_new_tokens=mnt,
            selector=selector, topk=topk, sink_tokens=sink_tokens,
            bare_question_ids=bare_q_ids, use_kv_cache=use_kv,
        )
        _sync(device)
        return time.perf_counter() - t0

    once(1)  # warmup (also = prefill + 1 step)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t1 = once(1)
    tN = once(max_new_tokens)
    steps = max(1, max_new_tokens - 1)
    decode_per_step = (tN - t1) / steps
    peak = (torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda" else 0.0)
    return {"t_prefill_plus1": t1, "t_full": tN,
            "decode_s_per_step": decode_per_step,
            "decode_tok_s": (1.0 / decode_per_step) if decode_per_step > 0 else 0.0,
            "peak_gb": peak}


def run_benchmark(qc, tokenizer, *, lengths, chunk_size, topk, selector,
                  max_new_tokens, sink_tokens, device, vocab):
    print("=" * 72)
    print(f"QCMem decode benchmark  (resume_j={qc.resume_j}, topk={topk}, "
          f"chunk_size={chunk_size}, max_new_tokens={max_new_tokens})")
    print("=" * 72)
    g = torch.Generator(device="cpu").manual_seed(0)
    for L in lengths:
        n_tok = _LEN_TOKENS[L] if L in _LEN_TOKENS else int(L)
        ids = torch.randint(1, vocab, (1, n_tok), generator=g).to(device)
        read_len = min(topk, max(0, n_tok // chunk_size - 1)) * chunk_size
        print(f"\n[context {L} = {n_tok} tok]  approx read pack ~ "
              f"{read_len + chunk_size} tok")
        base = _time_generate(qc, tokenizer, ids, chunk_size=chunk_size, topk=topk,
                              selector=selector, max_new_tokens=max_new_tokens,
                              sink_tokens=sink_tokens, use_kv=False, device=device)
        kv = _time_generate(qc, tokenizer, ids, chunk_size=chunk_size, topk=topk,
                            selector=selector, max_new_tokens=max_new_tokens,
                            sink_tokens=sink_tokens, use_kv=True, device=device)
        sp = (base["decode_s_per_step"] / kv["decode_s_per_step"]
              if kv["decode_s_per_step"] > 0 else float("nan"))
        print(f"  recompute : {base['decode_s_per_step']*1e3:8.1f} ms/step  "
              f"({base['decode_tok_s']:6.2f} tok/s)  peak {base['peak_gb']:.2f} GB")
        print(f"  kv-cache  : {kv['decode_s_per_step']*1e3:8.1f} ms/step  "
              f"({kv['decode_tok_s']:6.2f} tok/s)  peak {kv['peak_gb']:.2f} GB")
        print(f"  ==> decode speedup {sp:.1f}x   "
              f"mem {kv['peak_gb'] - base['peak_gb']:+.2f} GB")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["correctness", "benchmark", "both"],
                    default="both")
    ap.add_argument("--model_path", type=str, default="")
    ap.add_argument("--tiny", action="store_true",
                    help="Build a tiny random Qwen3 (no weights) for CPU correctness.")
    ap.add_argument("--tiny_layers", type=int, default=4)
    ap.add_argument("--tiny_hidden", type=int, default=64)
    ap.add_argument("--tiny_vocab", type=int, default=256)
    ap.add_argument("--resume_j", type=int, default=12)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--selector", type=str, default="recency",
                    help="Chunk selector (recency is forward-free & deterministic; "
                         "decode cost is selector-independent).")
    ap.add_argument("--sink_tokens", type=str, default="bos", choices=["bos", "none"])
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--context_lengths", type=str, nargs="+", default=["16k", "64k"])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--tol", type=float, default=-1.0,
                    help="Correctness max|logit diff| tolerance (-1 -> 1e-4 for "
                         "tiny/fp32, else 1e-2 for bf16).")
    # correctness-prompt sizing (tiny path)
    ap.add_argument("--corr_context_chunks", type=int, default=5)
    ap.add_argument("--corr_query_len", type=int, default=6)
    args = ap.parse_args()

    want_cuda = args.device.startswith("cuda")
    if want_cuda and not torch.cuda.is_available():
        print("[bench] cuda requested but unavailable -> falling back to cpu")
        args.device = "cpu"
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    # ---- build / load backbone ----
    if args.tiny or not args.model_path:
        if device.type == "cuda":
            print("[bench] tiny random model -> using fp32 for the correctness gate")
        model, cfg = build_tiny_qwen3(args.tiny_layers, args.tiny_hidden,
                                      args.tiny_vocab)
        model = model.to(device=device, dtype=torch.float32).eval()
        vocab = args.tiny_vocab
        tokenizer = _TinyTok(vocab)
        L = args.tiny_layers
        # keep resume_j sane for a tiny model
        resume_j = min(args.resume_j, L // 2) if args.resume_j > L else args.resume_j
        resume_j = max(0, min(resume_j, L))
        chunk_size = min(args.chunk_size, 8)
        tol = args.tol if args.tol > 0 else 1e-4
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
            trust_remote_code=True, local_files_only=True,
        ).to(device).eval()
        vocab = int(model.config.vocab_size)
        L = int(model.config.num_hidden_layers)
        resume_j = max(0, min(args.resume_j, L))
        chunk_size = args.chunk_size
        tol = args.tol if args.tol > 0 else (1e-4 if dtype == torch.float32 else 1e-2)
        is_tiny = False

    qc = QCMemModel(model, resume_j=resume_j)
    print(f"[bench] backbone L={L} resume_j={resume_j} chunk_size={chunk_size} "
          f"device={device} tol={tol:.1e}")

    ok = True
    if args.mode in ("correctness", "both"):
        # Build a deterministic prompt: corr_context_chunks * chunk_size context
        # tokens + a short query chunk.
        cs = chunk_size if is_tiny else args.chunk_size
        n_ctx = args.corr_context_chunks * cs
        n_q = args.corr_query_len
        g = torch.Generator(device="cpu").manual_seed(1234)
        ids = torch.randint(2, vocab, (1, n_ctx + n_q), generator=g).to(device)
        mnt = min(args.max_new_tokens, 24)
        ok = run_correctness(
            qc, tokenizer, ids, chunk_size=cs, topk=args.topk,
            selector=args.selector, max_new_tokens=mnt, tol=tol,
            sink_tokens=args.sink_tokens)

    if args.mode in ("benchmark", "both"):
        if is_tiny:
            # tiny model: still exercise the timing plumbing on short contexts
            run_benchmark(qc, tokenizer, lengths=["1k", "2k"], chunk_size=chunk_size,
                          topk=args.topk, selector=args.selector,
                          max_new_tokens=min(args.max_new_tokens, 24),
                          sink_tokens=args.sink_tokens, device=device, vocab=vocab)
        else:
            run_benchmark(qc, tokenizer, lengths=args.context_lengths,
                          chunk_size=args.chunk_size, topk=args.topk,
                          selector=args.selector, max_new_tokens=args.max_new_tokens,
                          sink_tokens=args.sink_tokens, device=device, vocab=vocab)

    if args.mode in ("correctness", "both") and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
