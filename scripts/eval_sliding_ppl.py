#!/usr/bin/env python3
"""Standard sliding-window long-context PPL eval (PG19 / Proof-Pile / CodeParrot).

This is the StreamingLLM / LM-Infinite "sliding-window PPL" three-set protocol:
build long continuous token sequences and measure token-level next-token loss,
scoring ONLY new (non-overlapping) tokens. Two model paths:

  * base Llama-3-8B  (no --adapter_config): standard HF sliding-window PPL.
    For each eval sequence of length ``--seq_length`` we slide a window of
    ``--window`` tokens with ``--stride`` step; only the last ``stride`` tokens of
    each window contribute to the loss (the first window scores all of its
    tokens). This is exactly the口径 from the HF perplexity doc / LM-Infinite —
    each token is scored once, with as much left-context as the window allows.

  * mem_space (with --adapter_config): the mem_space "sliding window" semantics =
    chunk streaming + persistent memory bank. We reset the bank per eval
    sequence, split the sequence into ``--chunk_size`` chunks, stream them in
    order, and score each chunk's OWN within-chunk next-token loss. Earlier
    chunks reach the current chunk only through the memory bank (exactly the
    train-time TBPTT objective: ``out = model(chunk, labels=chunk)``). Each token
    is scored once. This measures long-range LM fluency under the fixed-size
    memory bottleneck.

Both paths operate on identical ``--seq_length`` sequences carved from the same
token stream, so the numbers are directly comparable.

Usage:
    # base
    python scripts/eval_sliding_ppl.py --data pg19 \
        --data_path data/pg19_chunks_llama3_noeos.npy \
        --model_path models/Meta-Llama-3-8B \
        --seq_length 32768 --window 8192 --stride 4096

    # mem_space
    python scripts/eval_sliding_ppl.py --data pg19 \
        --data_path data/pg19_chunks_llama3_noeos.npy \
        --model_path models/Meta-Llama-3-8B \
        --adapter_config outputs/mem_space_p11_chunk1024_deltarule_normreadout/adapter_config.json \
        --checkpoint outputs/mem_space_p11_chunk1024_deltarule_normreadout/mem_space_adapter.pt \
        --seq_length 32768 --chunk_size 1024
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --------------------------------------------------------------------------- #
# Long token-stream loader
# --------------------------------------------------------------------------- #
def load_token_stream(data_path: str, skip_tokens: int, max_tokens: int) -> np.ndarray:
    """Load a pre-tokenized .npy (any shape) and return a flat 1-D token slice.

    The npy may be (N, L) chunks or 1-D; we flatten to one continuous stream and
    take ``[skip_tokens : skip_tokens + max_tokens]``. The stream is treated as a
    single continuous corpus (documents already concatenated at build time).
    """
    arr = np.load(data_path, mmap_mode="r")
    flat = arr.reshape(-1)
    total = flat.shape[0]
    start = min(skip_tokens, total)
    end = min(start + max_tokens, total) if max_tokens > 0 else total
    out = np.asarray(flat[start:end], dtype=np.int64)
    print(f"[data] {data_path}: total={total} tokens, using [{start}:{end}] = {len(out)} tokens")
    return out


def iter_sequences(stream: np.ndarray, seq_length: int):
    """Yield contiguous non-overlapping sequences of exactly ``seq_length`` tokens."""
    n = (len(stream) // seq_length) * seq_length
    for s in range(0, n, seq_length):
        yield stream[s:s + seq_length]


# --------------------------------------------------------------------------- #
# base Llama-3 sliding-window PPL
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_base_sliding(model, stream, seq_length, window, stride, device, dtype):
    """HF-style sliding-window PPL: score only new (non-overlapping) tokens."""
    total_nll = 0.0
    total_tok = 0
    n_seq = 0
    t0 = time.time()
    for seq in iter_sequences(stream, seq_length):
        ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # [1,S]
        S = ids.shape[1]
        prev_end = 0
        for begin in range(0, S, stride):
            end = min(begin + window, S)
            trg_len = end - prev_end  # number of "new" tokens to score in this window
            input_ids = ids[:, begin:end]
            target = input_ids.clone()
            target[:, :-trg_len] = -100  # only score the last trg_len tokens
            with torch.amp.autocast("cuda", dtype=dtype):
                out = model(input_ids=input_ids, labels=target)
            # HF CE loss is the mean over non-(-100) SHIFTED labels.
            shift = target[:, 1:]
            n_scored = int((shift != -100).sum().item())
            if n_scored > 0 and torch.isfinite(out.loss):
                total_nll += out.loss.item() * n_scored
                total_tok += n_scored
            prev_end = end
            if end == S:
                break
        n_seq += 1
        if n_seq % 5 == 0:
            ppl = math.exp(total_nll / max(1, total_tok))
            print(f"[base] seq {n_seq}: cumul_ppl={ppl:.4f} tok={total_tok} ({time.time()-t0:.0f}s)")
    return total_nll, total_tok, n_seq


# --------------------------------------------------------------------------- #
# mem_space chunk-streaming PPL
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_mem_space_streaming(model, reset_fn, stream, seq_length, chunk_size, device, dtype,
                             skip_first_chunk=False):
    """Stream chunks through mem_space; score each chunk's within-chunk LM loss.

    Memory bank persists across chunks within one sequence and is reset between
    sequences. Mirrors the train-time TBPTT objective (out = model(chunk,
    labels=chunk)) so the PPL reflects exactly what the model was optimized for.
    """
    total_nll = 0.0
    total_tok = 0
    n_seq = 0
    t0 = time.time()
    for seq in iter_sequences(stream, seq_length):
        reset_fn(model)
        ids = torch.tensor(seq, dtype=torch.long, device=device)
        chunks = list(ids.split(chunk_size))
        for ci, chunk in enumerate(chunks):
            if chunk.shape[0] < 2:
                continue
            ctensor = chunk.unsqueeze(0)  # [1, <=chunk_size]
            with torch.amp.autocast("cuda", dtype=dtype):
                out = model(input_ids=ctensor, labels=ctensor, use_cache=False)
            if skip_first_chunk and ci == 0:
                continue  # chunk 0 has no memory; optionally exclude
            n_scored = chunk.shape[0] - 1  # within-chunk next-token predictions
            if n_scored > 0 and torch.isfinite(out.loss):
                total_nll += out.loss.item() * n_scored
                total_tok += n_scored
        n_seq += 1
        if n_seq % 5 == 0:
            ppl = math.exp(total_nll / max(1, total_tok))
            print(f"[mem] seq {n_seq}: cumul_ppl={ppl:.4f} tok={total_tok} ({time.time()-t0:.0f}s)")
    return total_nll, total_tok, n_seq


def main():
    ap = argparse.ArgumentParser(description="Sliding-window long-context PPL eval")
    ap.add_argument("--data", choices=["pg19", "proofpile", "codeparrot"], required=True)
    ap.add_argument("--data_path", required=True, help="Pre-tokenized .npy token stream")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--adapter_config", default=None,
                    help="If set → mem_space chunk-streaming path. Else base sliding-window.")
    ap.add_argument("--checkpoint", default=None, help="mem_space adapter .pt (required with --adapter_config)")
    ap.add_argument("--seq_length", type=int, default=32768)
    ap.add_argument("--window", type=int, default=8192, help="base sliding window size")
    ap.add_argument("--stride", type=int, default=4096, help="base sliding window stride")
    ap.add_argument("--chunk_size", type=int, default=1024, help="mem_space chunk size")
    ap.add_argument("--skip_tokens", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=1_000_000,
                    help="Cap total eval tokens (stream slice). -1 = all.")
    ap.add_argument("--skip_first_chunk", action="store_true",
                    help="mem_space: exclude chunk-0 (no memory yet) from PPL.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    stream = load_token_stream(args.data_path, args.skip_tokens, args.max_tokens)

    is_mem = args.adapter_config is not None
    if is_mem:
        assert args.checkpoint, "--checkpoint required with --adapter_config"
        from scripts.run_babilong_mem_space import (
            build_mem_space_config, load_mem_space_model, _reset_banks, _reset_l2,
        )
        with open(args.adapter_config) as f:
            adapter_cfg = json.load(f)
        mem_config = build_mem_space_config(adapter_cfg)
        mem_config.l3_recon_max_positions = args.chunk_size
        print(f"[mem_space] num_slots={mem_config.num_slots} top_k={mem_config.top_k} "
              f"chunk_size={args.chunk_size}")
        model = load_mem_space_model(
            model_path=args.model_path, checkpoint_path=args.checkpoint,
            mem_config=mem_config, device=device, dtype=dtype, attn_impl=args.attn_impl,
        )

        def reset_fn(m):
            _reset_banks(m)
            _reset_l2(m)

        total_nll, total_tok, n_seq = eval_mem_space_streaming(
            model, reset_fn, stream, args.seq_length, args.chunk_size, device, dtype,
            skip_first_chunk=args.skip_first_chunk,
        )
        mode = "mem_space"
    else:
        from transformers import LlamaForCausalLM
        print(f"[base] loading {args.model_path} ({args.attn_impl})")
        model = LlamaForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, attn_implementation=args.attn_impl,
        ).to(device)
        model.eval()
        total_nll, total_tok, n_seq = eval_base_sliding(
            model, stream, args.seq_length, args.window, args.stride, device, dtype,
        )
        mode = "base"

    ppl = math.exp(total_nll / max(1, total_tok))
    print(f"\n===== RESULT =====")
    print(f"mode={mode} data={args.data} model={args.model_path}")
    print(f"seq_length={args.seq_length} seqs={n_seq} scored_tokens={total_tok}")
    print(f"PPL = {ppl:.4f}  (avg_nll={total_nll/max(1,total_tok):.6f})")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        rec = {
            "mode": mode, "data": args.data, "model": args.model_path,
            "adapter_config": args.adapter_config, "checkpoint": args.checkpoint,
            "seq_length": args.seq_length, "window": args.window, "stride": args.stride,
            "chunk_size": args.chunk_size, "seqs": n_seq, "scored_tokens": total_tok,
            "avg_nll": total_nll / max(1, total_tok), "ppl": ppl,
            "skip_first_chunk": args.skip_first_chunk,
        }
        with open(args.output_json, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[saved] {args.output_json}")


if __name__ == "__main__":
    main()
