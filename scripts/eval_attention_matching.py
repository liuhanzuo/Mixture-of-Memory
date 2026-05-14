#!/usr/bin/env python3
"""Evaluate Attention Matching KV cache compression on PG19.

This is a CPU / single-GPU evaluation script (no torchrun needed).

Usage:
    python scripts/eval_attention_matching.py \
        --model /path/to/Llama3-8b \
        --data /path/to/pg19_chunks_llama3.npy \
        --max_chunks 50 --seq_len 4096 \
        --compression_ratios 2,4,8,16,32 \
        --output_file results/attention_matching_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve(path: str) -> str:
    return str(Path(path).expanduser().resolve())


# ---------------------------------------------------------------------------
# PPL computation helpers
# ---------------------------------------------------------------------------

def compute_full_kv_ppl(model, input_ids: torch.Tensor) -> float:
    """Compute baseline PPL using full (uncompressed) KV cache."""
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits  # [1, T, vocab]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        ppl = math.exp(loss.item())
    return ppl


def compute_compressed_ppl_simple(
    model,
    input_ids: torch.Tensor,
    compressor,
    budget_per_head: int,
) -> float:
    """Compute PPL using compressed KV cache via AttentionMatchingCompressor.

    This does the full pipeline: extract KV, compress, then recompute
    attention with compressed KV and measure perplexity.

    The approach:
    1. Forward pass to capture per-layer KV and hidden states.
    2. Compress each layer's KV.
    3. Second forward pass using compressed KV for attention.
    """
    import torch.nn.functional as F
    from src.memory.mem_space.attention_matching import (
        AttentionMatchingCompressor,
        _to_f32,
        _softmax,
        _apply_rotary_pos_emb,
    )

    device = input_ids.device
    model.eval()
    T = input_ids.shape[1]
    cfg = model.config
    d_head = cfg.hidden_size // cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    n_heads = cfg.num_attention_heads
    n_layers = cfg.num_hidden_layers
    scale = math.sqrt(d_head)
    heads_per_kv = n_heads // n_kv_heads

    with torch.no_grad():
        # === Pass 1: extract all KV and hidden states ===
        hidden = model.model.embed_tokens(input_ids)  # [1, T, d]
        all_K = []
        all_V = []
        all_Q = []
        position_ids = torch.arange(T, device=device).unsqueeze(0)

        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            self_attn = layer.self_attn

            if hasattr(layer, 'input_layernorm'):
                h_ln = layer.input_layernorm(hidden)
            else:
                h_ln = hidden

            K = self_attn.k_proj(h_ln).view(1, T, n_kv_heads, d_head)
            V = self_attn.v_proj(h_ln).view(1, T, n_kv_heads, d_head)
            Q = self_attn.q_proj(h_ln).view(1, T, n_heads, d_head)

            all_K.append(K.squeeze(0))  # [T, n_kv_heads, d_head]
            all_V.append(V.squeeze(0))
            all_Q.append(Q.squeeze(0))

            # Forward through this layer.
            position_embeddings = model.model.rotary_emb(h_ln, position_ids)
            layer_out = layer(
                hidden,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # === Compress each layer ===
        compact_kv = []
        ref_stride = max(1, T // 512)

        for layer_idx in range(n_layers):
            K = all_K[layer_idx]  # [T, n_kv_heads, d_head]
            V = all_V[layer_idx]
            Q_all = all_Q[layer_idx]  # [T, n_heads, d_head]

            Q_ref_all = Q_all[::ref_stride]  # [T_ref, n_heads, d_head]

            layer_Ck, layer_Cv, layer_beta = [], [], []

            for kv_head in range(n_kv_heads):
                q_start = kv_head * heads_per_kv
                q_end = q_start + heads_per_kv
                Q_ref_head = Q_ref_all[:, q_start:q_end, :].mean(dim=1)

                K_head = K[:, kv_head, :]
                V_head = V[:, kv_head, :]

                Ck, Cv, beta = compressor.compress(
                    K_head, V_head, Q_ref_head, budget_per_head
                )
                layer_Ck.append(Ck)
                layer_Cv.append(Cv)
                layer_beta.append(beta)

            compact_kv.append((
                torch.stack(layer_Ck),
                torch.stack(layer_Cv),
                torch.stack(layer_beta),
            ))

        # === Pass 2: compute attention with compressed KV ===
        hidden = model.model.embed_tokens(input_ids)
        total_loss = 0.0
        n_tokens = 0

        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            self_attn = layer.self_attn
            Ck, Cv, beta = compact_kv[layer_idx]
            budget = Ck.shape[1]

            if hasattr(layer, 'input_layernorm'):
                h_ln = layer.input_layernorm(hidden)
            else:
                h_ln = hidden

            Q = self_attn.q_proj(h_ln).view(1, T, n_heads, d_head)

            # Apply RoPE to Q.
            cos, sin = model.model.rotary_emb(h_ln, position_ids)
            Q_rot = Q.clone()
            Q_rot = _apply_rotary_pos_emb(Q_rot, cos, sin)

            # Apply RoPE to compact keys at evenly-spaced positions.
            ck_pos = torch.linspace(0, T - 1, budget).long().to(device)
            ck_cos = cos[:, ck_pos].unsqueeze(2)  # [1, budget, 1, d_head]
            ck_sin = sin[:, ck_pos].unsqueeze(2)
            Ck_for_rope = Ck.permute(1, 0, 2).unsqueeze(0)  # [1, budget, n_kv_heads, d_head]
            Ck_rot = _apply_rotary_pos_emb(Ck_for_rope, ck_cos, ck_sin)

            # Expand for GQA.
            Ck_exp = Ck_rot.repeat_interleave(heads_per_kv, dim=2)
            Cv_exp = Cv.permute(1, 0, 2).unsqueeze(0).repeat_interleave(heads_per_kv, dim=2)
            beta_heads = beta.T.repeat_interleave(heads_per_kv, dim=0).unsqueeze(0)  # [1, n_heads, budget]

            # Attention: [1, n_heads, T, budget]
            Q_t = Q_rot.transpose(1, 2)
            Ck_t = Ck_exp.transpose(1, 2)
            Cv_t = Cv_exp.transpose(1, 2)

            attn_logits = torch.matmul(Q_t, Ck_t.transpose(-2, -1)) / scale
            attn_logits = attn_logits + beta_heads.unsqueeze(2)

            attn_weights = _softmax(attn_logits, dim=-1)
            attn_output = torch.matmul(attn_weights, Cv_t)  # [1, n_heads, T, d_head]

            attn_output = attn_output.transpose(1, 2).contiguous().view(1, T, -1)
            attn_output = self_attn.o_proj(attn_output)

            # Residual connection: hidden = original_hidden + attn_output
            hidden = hidden + attn_output

            # MLP with post-attention layernorm + residual.
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = residual + layer.mlp(hidden)

        # LM head.
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        ppl = math.exp(loss.item())

    return ppl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Attention Matching KV compression"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b",
        help="Path to Llama-3-8B model directory",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy",
        help="Path to pg19 tokenized chunks (.npy)",
    )
    parser.add_argument("--max_chunks", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument(
        "--compression_ratios",
        type=str,
        default="2,4,8,16,32",
        help="Comma-separated compression ratios",
    )
    parser.add_argument("--method", type=str, default="omp", choices=["omp", "highest_attn"])
    parser.add_argument("--ridge_lambda", type=float, default=1e-4)
    parser.add_argument("--beta_clamp", type=float, default=10.0)
    parser.add_argument("--output_file", type=str, default="results/attention_matching_eval.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--start_chunk", type=int, default=0)
    args = parser.parse_args()

    # Device.
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data.
    data = np.load(args.data, mmap_mode="r")
    total_chunks = min(args.max_chunks, data.shape[0] - args.start_chunk)
    print(f"Data shape: {data.shape}, evaluating {total_chunks} chunks (starting at {args.start_chunk})")

    # Load model.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device if device.type == "cuda" else None,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()

    print(f"Model loaded. hidden_size={model.config.hidden_size}, "
          f"n_heads={model.config.num_attention_heads}, "
          f"n_kv_heads={model.config.num_key_value_heads}, "
          f"n_layers={model.config.num_hidden_layers}")

    # Compression ratios.
    ratios = [int(r) for r in args.compression_ratios.split(",")]
    seq_len = args.seq_len
    d_head = model.config.hidden_size // model.config.num_attention_heads

    results: Dict = {
        "model": args.model,
        "data": args.data,
        "seq_len": seq_len,
        "max_chunks": total_chunks,
        "method": args.method,
        "ridge_lambda": args.ridge_lambda,
        "per_ratio": {},
    }

    # === Baseline PPL (full KV) ===
    print("\n=== Computing baseline PPL (full KV) ===")
    baseline_ppls = []
    t0 = time.time()
    for i in range(total_chunks):
        chunk_idx = args.start_chunk + i
        ids = torch.tensor(data[chunk_idx, :seq_len], dtype=torch.long, device=device).unsqueeze(0)
        ppl = compute_full_kv_ppl(model, ids)
        baseline_ppls.append(ppl)
        if (i + 1) % 10 == 0:
            avg = np.mean(baseline_ppls)
            print(f"  [{i+1}/{total_chunks}] avg baseline PPL: {avg:.4f}")
    baseline_avg = float(np.mean(baseline_ppls))
    print(f"Baseline PPL: {baseline_avg:.4f} (took {time.time()-t0:.1f}s)")
    results["baseline_ppl"] = baseline_avg

    # === Compressed PPL at each ratio ===
    from src.memory.mem_space.attention_matching import AttentionMatchingCompressor

    for ratio in ratios:
        budget = seq_len // ratio
        print(f"\n=== Compression ratio {ratio}x (budget={budget} keys per head) ===")

        compressor = AttentionMatchingCompressor(
            compression_ratio=ratio,
            method=args.method,
            ridge_lambda=args.ridge_lambda,
            beta_clamp=args.beta_clamp,
        )

        compressed_ppls = []
        t0 = time.time()
        for i in range(total_chunks):
            chunk_idx = args.start_chunk + i
            ids = torch.tensor(data[chunk_idx, :seq_len], dtype=torch.long, device=device).unsqueeze(0)

            try:
                ppl = compute_compressed_ppl_simple(model, ids, compressor, budget)
                compressed_ppls.append(ppl)
            except Exception as e:
                print(f"  WARNING: chunk {chunk_idx} failed: {e}")
                compressed_ppls.append(float("inf"))

            if (i + 1) % 5 == 0:
                valid = [p for p in compressed_ppls if p < float("inf")]
                if valid:
                    avg = np.mean(valid)
                    print(f"  [{i+1}/{total_chunks}] avg PPL: {avg:.4f}")

        valid_ppls = [p for p in compressed_ppls if p < float("inf")]
        if valid_ppls:
            avg_ppl = float(np.mean(valid_ppls))
            degradation = avg_ppl - baseline_avg
            pct_change = (degradation / baseline_avg) * 100
            print(f"  Result: PPL={avg_ppl:.4f} (degradation: {degradation:+.4f}, {pct_change:+.1f}%)")
        else:
            avg_ppl = float("inf")
            degradation = float("inf")
            pct_change = float("inf")
            print("  Result: ALL FAILED")

        results["per_ratio"][str(ratio)] = {
            "budget_per_head": budget,
            "avg_ppl": avg_ppl,
            "degradation": degradation if degradation != float("inf") else None,
            "pct_change": pct_change if pct_change != float("inf") else None,
            "n_valid": len(valid_ppls),
            "total": total_chunks,
            "time_seconds": time.time() - t0,
        }

    # === Summary ===
    print("\n" + "=" * 60)
    print(f"{'Ratio':>8s} {'Budget':>8s} {'PPL':>10s} {'Degrad.':>10s} {'%Chg':>8s}")
    print("-" * 60)
    print(f"{'Full':>8s} {'N/A':>8s} {baseline_avg:10.4f} {'0.0000':>10s} {'0.0':>8s}")
    for ratio in ratios:
        r = results["per_ratio"][str(ratio)]
        ppl_str = f"{r['avg_ppl']:.4f}" if r["avg_ppl"] != float("inf") else "FAIL"
        deg_str = f"{r['degradation']:+.4f}" if r["degradation"] is not None else "N/A"
        pct_str = f"{r['pct_change']:+.1f}" if r["pct_change"] is not None else "N/A"
        print(f"{f'{ratio}x':>8s} {r['budget_per_head']:>8d} {ppl_str:>10s} {deg_str:>10s} {pct_str:>8s}")
    print("=" * 60)

    # Save results.
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
