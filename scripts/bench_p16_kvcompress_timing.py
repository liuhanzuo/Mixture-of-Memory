#!/usr/bin/env python
"""Paper A P1.6 — EFFICIENCY (timing / peak-memory) bench for the standard
prefill-then-compress KV baselines SnapKV / PyramidKV on Qwen3-8B.

WHY THIS SCRIPT
---------------
The P1.6 QUALITY harness (``scripts/eval_p16_kvcompress.py``) already records
per-example timing via ``kvc.generate_kvcompress(stats=...)``, but it does so on
variable-length quality samples with no warmup / no fixed-length control, so the
result table's full-prefill-latency / peak-GB / decode-ms/tok columns are TBD.
This script fills exactly those columns: same model-load + KV-compression logic
(``src.baselines.qwen3_kvcompress`` — the SAME faithful Qwen3 hijack + vendored
GQA-aware SnapKV/PyramidKV clusters the quality eval uses; NO reimplementation),
run on FIXED synthetic ``[1, L]`` sequences with warmup + median-of-N, mirroring
the ``scripts/bench_qcmem_vs_fullctx.py`` timing paradigm (random ids — we time
COMPUTE not accuracy; torch.cuda.synchronize brackets; reset_peak_memory_stats +
max_memory_allocated for peak GB).

TIMING BOUNDARIES (per method, per length L; all measured inside
``kvc.generate_kvcompress``, one call = one prefill + ``n_decode`` greedy steps)
-------------------------------------------------------------------------------
* full-prefill latency (ms)
    Wall time of the SINGLE forward over the ENTIRE, UNCOMPRESSED L-token prompt
    ``model(input_ids=[1,L], past_key_values=DynamicCache, use_cache=True)`` down
    to the last-position logits — i.e. the cost of SEEING THE WHOLE PROMPT. This
    is the defining SnapKV/PyramidKV cost (they must full-prefill before they can
    compress the STORED KV), and the key contrast with CoMem's bounded persistent
    store. Attention here is the exact full O(L^2) attention (identical to a
    full-context prefill); only the KV WRITTEN to the cache is compressed to the
    retained budget. Bracketed by torch.cuda.synchronize().  [prefill_latency_s]
* peak GPU memory (GB)
    torch.cuda.max_memory_allocated() captured right after the prefill forward,
    with reset_peak_memory_stats() called immediately before it. INCLUDES the
    resident bf16 weights (~16 GB) + full-prefill activations/attention — same
    convention as CoMem's tab_eff "Full mem" column, so it is directly
    comparable.  [prefill_peak_gb]
* decode latency (ms/tok)
    Wall time of the greedy decode loop over the (now COMPRESSED) cache, divided
    by the number of decoded tokens. Decode reads only the retained ~budget KV,
    so this is O(budget), context-independent.  [decode_latency_per_tok_ms]

Also reported per cell: compressed retained-KV bytes (MB) and per-layer retained
length (the equal-retained-token audit vs CoMem's 6657 read budget), prompt token
count, and OOM/fallback status.

Each (method, length) is run ``--warmup`` un-counted times then ``--n_repeat``
timed reps; we report the MEDIAN plus the [min, max] spread across reps. EOS is
disabled during decode so every rep decodes exactly ``--n_decode`` tokens (clean,
comparable decode timing).

Protocol matched to the P1.6 quality eval: Qwen3-8B, bf16, SDPA, chat=False
(irrelevant here — synthetic ids), budget=6657, window=32, kernel=5, avgpool,
gqa mean, pyramid beta=20.

128k note: Qwen3-8B native window is 40960, but this is a COMPUTE/MEMORY timing
bench — we prefill the full L tokens (no left-truncation) so the 8k/32k/128k
full-prefill costs line up with CoMem's tab_eff lengths. RoPE extrapolation past
40960 hurts QUALITY (measured elsewhere) but does NOT change tensor shapes /
FLOPs / peak memory, so the reported latency+memory are the faithful full-prefill
cost at that length. (YaRN would only rescale rope frequencies, not the timing.)

Example (single method sanity, 8k):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/bench_p16_kvcompress_timing.py \
    --model_path models/Qwen3-8b-local --methods snapkv --lengths 8k \
    --n_repeat 3 --warmup 1 --output outputs/p16_timing/sanity_8k.json
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# The SAME faithful Qwen3 hijack + vendored SnapKV/PyramidKV clusters the P1.6
# quality harness (scripts/eval_p16_kvcompress.py) uses. No reimplementation.
from src.baselines import qwen3_kvcompress as kvc  # noqa: E402

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def parse_length(s: str) -> int:
    """'8k' -> 8192, '128k' -> 131072, '4096' -> 4096."""
    s = str(s).strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


def _stats(vals):
    """median + [min, max] spread of a list of floats (None-safe)."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {
        "median": round(statistics.median(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "n": len(vals),
    }


def bench_one(model, input_ids, *, n_decode, n_repeat, warmup):
    """Warmup + median-of-N timed reps of a single full-prefill + decode via
    kvc.generate_kvcompress (EOS disabled -> exactly n_decode decode tokens).
    Returns an aggregated dict, or {'status': 'OOM'} on OOM."""
    prefill_ms, peak_gb, decode_ms_tok = [], [], []
    kv_mb, mean_ret, min_ret, max_ret = [], [], [], []
    prompt_tokens = int(input_ids.shape[1])
    decode_tokens = None
    try:
        for it in range(warmup + n_repeat):
            gstats: dict = {}
            _ = kvc.generate_kvcompress(
                model, input_ids, max_new_tokens=n_decode,
                eos_token_ids=[], extra_end_token_ids=[],  # never early-stop
                stats=gstats)
            if it >= warmup:
                prefill_ms.append(gstats["prefill_latency_s"] * 1000.0)
                if gstats.get("prefill_peak_gb") is not None:
                    peak_gb.append(gstats["prefill_peak_gb"])
                if gstats.get("decode_latency_per_tok_ms") is not None:
                    decode_ms_tok.append(gstats["decode_latency_per_tok_ms"])
                if gstats.get("compressed_kv_MB") is not None:
                    kv_mb.append(gstats["compressed_kv_MB"])
                if gstats.get("mean_retained_len") is not None:
                    mean_ret.append(gstats["mean_retained_len"])
                if gstats.get("min_retained_len") is not None:
                    min_ret.append(gstats["min_retained_len"])
                if gstats.get("max_retained_len") is not None:
                    max_ret.append(gstats["max_retained_len"])
                decode_tokens = gstats.get("decode_tokens")
            torch.cuda.empty_cache()
    except RuntimeError as e:
        if not _is_oom(e):
            raise
        torch.cuda.empty_cache()
        return {"status": "OOM", "prompt_tokens": prompt_tokens}

    return {
        "status": "ok",
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        "full_prompt_seen": True,
        "prefill_latency_ms": _stats(prefill_ms),
        "prefill_peak_gb": _stats(peak_gb),
        "decode_latency_per_tok_ms": _stats(decode_ms_tok),
        "compressed_kv_MB": _stats(kv_mb),
        "retained_len": {
            "mean": _stats(mean_ret), "min": _stats(min_ret), "max": _stats(max_ret)},
    }


def main():
    ap = argparse.ArgumentParser(
        description="Paper A P1.6 SnapKV/PyramidKV timing + peak-memory bench")
    ap.add_argument("--model_path", type=str, default="models/Qwen3-8b-local")
    ap.add_argument("--methods", type=str, nargs="+", default=["snapkv", "pyramidkv"],
                    choices=["snapkv", "pyramidkv"])
    ap.add_argument("--lengths", type=str, nargs="+", default=["8k", "32k", "128k"])

    # KV-compression config (identical defaults to the P1.6 quality eval).
    ap.add_argument("--max_capacity_prompt", type=int, default=6657)
    ap.add_argument("--window_size", type=int, default=32)
    ap.add_argument("--kernel_size", type=int, default=5)
    ap.add_argument("--pooling", choices=["avgpool", "maxpool"], default="avgpool")
    ap.add_argument("--gqa_score_agg", choices=["mean", "max", "sum"], default="mean")
    ap.add_argument("--beta", type=int, default=20)

    # protocol
    ap.add_argument("--dtype", choices=list(_DTYPES), default="bfloat16")
    ap.add_argument("--attn_impl", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)

    # timing
    ap.add_argument("--n_repeat", type=int, default=3,
                    help="Timed repeats (median + [min,max] reported).")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--n_decode", type=int, default=64,
                    help="Greedy decode tokens per rep (EOS disabled).")
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        # Pin the default CUDA device so argless max_memory_allocated() /
        # reset_peak_memory_stats() inside generate_kvcompress read THIS device.
        torch.cuda.set_device(device)
    dtype = _DTYPES[args.dtype]

    # normalize model_path relative to PROJECT_ROOT if needed.
    if not os.path.isabs(args.model_path) and not os.path.exists(args.model_path):
        cand = os.path.join(PROJECT_ROOT, args.model_path)
        if os.path.exists(cand):
            args.model_path = cand

    print("=" * 80)
    print("Paper A P1.6 — SnapKV/PyramidKV EFFICIENCY bench (full-prefill-then-compress)")
    print(f"  model_path = {args.model_path}")
    print(f"  methods={args.methods} lengths={args.lengths}")
    print(f"  budget(max_capacity_prompt)={args.max_capacity_prompt} window={args.window_size} "
          f"kernel={args.kernel_size} pool={args.pooling} gqa={args.gqa_score_agg} beta={args.beta}")
    print(f"  dtype={args.dtype} attn={args.attn_impl} device={device}")
    print(f"  n_repeat={args.n_repeat} warmup={args.warmup} n_decode={args.n_decode} seed={args.seed}")
    print(f"  node={socket.gethostname()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print("=" * 80, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, attn_implementation=args.attn_impl,
        trust_remote_code=True, local_files_only=True).to(device).eval()
    vocab = int(model.config.vocab_size)
    L_layers = int(model.config.num_hidden_layers)
    print(f"[bench] backbone: {L_layers} layers, hidden={model.config.hidden_size}, "
          f"vocab={vocab}", flush=True)

    lengths = [(lab, parse_length(lab)) for lab in args.lengths]

    results = {
        "config": {
            "model_path": args.model_path, "methods": args.methods,
            "lengths": args.lengths, "max_capacity_prompt": args.max_capacity_prompt,
            "window_size": args.window_size, "kernel_size": args.kernel_size,
            "pooling": args.pooling, "gqa_score_agg": args.gqa_score_agg,
            "beta": args.beta, "dtype": args.dtype, "attn_impl": args.attn_impl,
            "n_repeat": args.n_repeat, "warmup": args.warmup, "n_decode": args.n_decode,
            "seed": args.seed, "num_layers": L_layers, "vocab_size": vocab,
            "node": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": args.device,
            "timing_boundaries": {
                "prefill_latency_ms": "wall time of the single forward over the FULL "
                "uncompressed L-token prompt (exact O(L^2) attention) to last-pos "
                "logits; sync-bracketed. The cost of seeing the whole prompt.",
                "prefill_peak_gb": "torch.cuda.max_memory_allocated after prefill "
                "(reset just before); includes resident bf16 weights (~16GB); same "
                "convention as CoMem tab_eff 'Full mem'.",
                "decode_latency_per_tok_ms": "greedy decode-loop wall time over the "
                "COMPRESSED cache / decoded tokens; O(retained budget).",
            },
        },
        "methods": {},
    }

    for method in args.methods:
        print(f"\n{'#'*70}\n[bench] METHOD = {method}\n{'#'*70}", flush=True)
        cfg = kvc.install_kv_compression(
            model, method, max_capacity_prompt=args.max_capacity_prompt,
            window_size=args.window_size, kernel_size=args.kernel_size,
            pooling=args.pooling, gqa_score_agg=args.gqa_score_agg, beta=args.beta)
        print(f"[bench] installed: {cfg}", flush=True)
        results["methods"][method] = {}

        for label, L in lengths:
            print(f"\n[bench] --- {method} @ {label} (L={L}) ---", flush=True)
            input_ids = torch.randint(0, vocab, (1, L), device=device)
            r = bench_one(model, input_ids, n_decode=args.n_decode,
                          n_repeat=args.n_repeat, warmup=args.warmup)
            del input_ids
            torch.cuda.empty_cache()
            results["methods"][method][label] = {"L": L, **r}

            if r.get("status") == "OOM":
                print(f"[bench] {method} @ {label}: OOM (L={L})", flush=True)
            else:
                pf = r["prefill_latency_ms"]; pk = r["prefill_peak_gb"]
                dc = r["decode_latency_per_tok_ms"]; kv = r["compressed_kv_MB"]
                ret = r["retained_len"]["mean"]
                print(f"[bench] {method} @ {label}: "
                      f"prefill={pf['median']}ms [{pf['min']},{pf['max']}] | "
                      f"peak={pk['median']}GB | decode={dc['median']}ms/tok | "
                      f"kv={kv['median']}MB | retained~{ret['median']} "
                      f"(prompt={r['prompt_tokens']} tok, decode={r['decode_tokens']} tok)",
                      flush=True)

            # incremental save so partial results survive a late-length OOM/kill.
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)

        kvc.uninstall_kv_compression(model)

    # ---- final summary table ----
    print("\n" + "=" * 80)
    print("P1.6 EFFICIENCY SUMMARY (median-of-{}; full-prefill = SEE WHOLE PROMPT)".format(
        args.n_repeat))
    print("=" * 80)
    hdr = (f"{'method':<11} | {'len':>5} | {'prefill_ms':>12} | {'peak_GB':>8} | "
           f"{'decode_ms/tok':>13} | {'kv_MB':>8} | {'retained':>9}")
    print(hdr); print("-" * len(hdr))
    for method in args.methods:
        for label, _L in lengths:
            r = results["methods"][method][label]
            if r.get("status") == "OOM":
                print(f"{method:<11} | {label:>5} | {'OOM':>12} | {'OOM':>8} | "
                      f"{'OOM':>13} | {'-':>8} | {'-':>9}")
            else:
                pf = r["prefill_latency_ms"]["median"]; pk = r["prefill_peak_gb"]["median"]
                dc = r["decode_latency_per_tok_ms"]["median"]; kv = r["compressed_kv_MB"]["median"]
                ret = r["retained_len"]["mean"]["median"]
                print(f"{method:<11} | {label:>5} | {pf:>12.2f} | {pk:>8.2f} | "
                      f"{dc:>13.3f} | {kv:>8.2f} | {str(ret):>9}")
    print("=" * 80)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[bench] wrote JSON -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
