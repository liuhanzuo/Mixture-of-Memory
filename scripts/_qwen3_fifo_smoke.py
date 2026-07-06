"""Single-card forward smoke test: Qwen3-8B + mem_space FIFO flat readout.

Zero-training. Loads the base Qwen3-8B backbone, patches every decoder layer
with MemorySpaceLayer using the b64 FIFO flat config, and streams several
chunks through the model to verify:
  1. The patch applies (Qwen3DecoderLayer wrapped, no crash).
  2. Multi-chunk FIFO forward is numerically healthy (no NaN/Inf, finite logits).
  3. QK-norm stays intact (mem_space output must DIFFER from a memory-disabled
     pass only via the FIFO prefix; and a memory-disabled pass must equal the
     raw backbone forward within fp tolerance -> proves the wrapped layer,
     including q_norm/k_norm, is called correctly).

Usage:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/_qwen3_fifo_smoke.py \
        --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --adapter_config outputs/mem_space_pg19base_nctxcurric_b64/adapter_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.memory.mem_space import (  # noqa: E402
    MemorySpaceConfig,
    apply_mem_space_to_model,
    _reset_fifo_memory,
)
# reuse the eval helper that maps adapter_config.json -> MemorySpaceConfig
from scripts.run_babilong_mem_space import build_mem_space_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--adapter_config", required=True)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--n_chunks", type=int, default=6)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    with open(args.adapter_config) as f:
        adapter_cfg = json.load(f)
    mem_cfg = build_mem_space_config(adapter_cfg)
    print(f"[smoke] use_fifo_memory={mem_cfg.use_fifo_memory} "
          f"fifo_buffer_chunks={mem_cfg.fifo_buffer_chunks} "
          f"fifo_detach={mem_cfg.fifo_detach} "
          f"gradient_checkpointing={mem_cfg.gradient_checkpointing}")

    print(f"[smoke] loading base model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, attn_implementation="sdpa",
        local_files_only=True,
    ).to(device)
    print(f"[smoke] model class = {type(model).__name__}; "
          f"decoder layer class = {type(model.model.layers[0]).__name__}")
    # Confirm Qwen3 QK-norm exists on the raw backbone.
    _attn0 = model.model.layers[0].self_attn
    print(f"[smoke] layer0 self_attn has q_norm={hasattr(_attn0, 'q_norm')} "
          f"k_norm={hasattr(_attn0, 'k_norm')} head_dim={getattr(_attn0, 'head_dim', '?')}")

    apply_mem_space_to_model(model, mem_cfg, layer_indices=None)
    model.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    wrapped0 = model.model.layers[0]
    print(f"[smoke] after patch: layer0 class = {type(wrapped0).__name__}; "
          f"wrapped_layer class = {type(wrapped0.wrapped_layer).__name__}")
    # QK-norm is INSIDE wrapped_layer.self_attn -> still present after wrap.
    _wattn0 = wrapped0.wrapped_layer.self_attn
    print(f"[smoke] wrapped layer0 self_attn q_norm={hasattr(_wattn0, 'q_norm')} "
          f"k_norm={hasattr(_wattn0, 'k_norm')}")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    vocab = model.config.vocab_size

    torch.manual_seed(0)
    B = 1
    _reset_fifo_memory(model)

    # ---- Multi-chunk FIFO streaming forward (the SOTA flat path) ----
    all_finite = True
    last_logits = None
    for c in range(args.n_chunks):
        ids = torch.randint(0, vocab, (B, args.chunk_size), device=device)
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        logits = out.logits
        finite = torch.isfinite(logits).all().item()
        all_finite = all_finite and finite
        lo, hi = logits.float().min().item(), logits.float().max().item()
        mean_abs = logits.float().abs().mean().item()
        # count how many FIFO chunks are buffered per layer (should grow then cap)
        nbuf = len(wrapped0._fifo_buf)
        print(f"[smoke] chunk {c}: logits finite={finite} "
              f"min={lo:.2f} max={hi:.2f} mean|.|={mean_abs:.3f} "
              f"fifo_buf_len(layer0)={nbuf}")
        last_logits = logits

    print(f"[smoke] ALL chunks finite: {all_finite}")

    # ---- Parity: memory-disabled pass must equal raw backbone forward ----
    # This proves the wrapped Qwen3 layer (incl. QK-norm) is invoked correctly.
    root = model
    for w in root._mem_space_layers:
        w._memory_disabled = True
    _reset_fifo_memory(model)
    ids = torch.randint(0, vocab, (B, args.chunk_size), device=device)
    with torch.no_grad():
        out_disabled = model(input_ids=ids, use_cache=False).logits
    # raw backbone: temporarily swap layers back to wrapped_layer
    orig_layers = list(model.model.layers)
    try:
        import torch.nn as nn
        model.model.layers = nn.ModuleList([w.wrapped_layer for w in root._mem_space_layers])
        with torch.no_grad():
            out_raw = model(input_ids=ids, use_cache=False).logits
    finally:
        model.model.layers = nn.ModuleList(orig_layers)
    max_diff = (out_disabled.float() - out_raw.float()).abs().max().item()
    print(f"[smoke] memory-disabled vs raw-backbone max|diff| = {max_diff:.6f} "
          f"(should be ~0 -> wrapped Qwen3 layer + QK-norm invoked correctly)")

    # re-enable memory for cleanliness
    for w in root._mem_space_layers:
        w._memory_disabled = False

    ok = all_finite and (max_diff < 1e-2)
    print(f"[smoke] RESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
