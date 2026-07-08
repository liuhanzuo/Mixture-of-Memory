#!/usr/bin/env python
"""Self-test for the QCMem block-diagonal read ablation (2026-07-07).

Two gates on a real backbone (or a tiny random model if no path given):

  (1) SINGLE-CHUNK EQUIVALENCE: with exactly one context chunk, the
      block-diagonal read must be bit-identical to the standard causal read
      (there is no other chunk to hide, and the query already attends to that one
      chunk under both masks). This is the sanity gate demanded by the task.

  (2) MULTI-CHUNK DIVERGENCE: with >=2 context chunks the two reads MUST differ
      (block-diagonal removes cross-chunk attention among context chunks), else
      the mask is a no-op and the ablation is meaningless.

  (3) MASK SHAPE/STRUCTURE: the block-diagonal keep-mask is a strict subset of
      the causal keep-mask (only removes edges, never adds), and its context-chunk
      rows attend only to sink + own block.

Usage:
    python scripts/qcmem_blockdiag_selftest.py \
        --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --resume_j 12
    # or, tiny random model (fast, no weights needed):
    python scripts/qcmem_blockdiag_selftest.py --tiny
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.memory.qcmem import QCMemModel  # noqa: E402


def _build_tiny_model(device, dtype):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128,
        num_hidden_layers=6, num_attention_heads=8, num_key_value_heads=4,
        max_position_embeddings=2048,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(device=device, dtype=dtype).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="")
    ap.add_argument("--tiny", action="store_true", default=False)
    ap.add_argument("--resume_j", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # tight tolerance

    if args.tiny or not args.model_path:
        print("[selftest] building tiny random Llama (6 layers)")
        model = _build_tiny_model(device, dtype)
        resume_j = min(args.resume_j, model.config.num_hidden_layers)
    else:
        from transformers import AutoModelForCausalLM
        print(f"[selftest] loading {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, attn_implementation=args.attn_impl,
            trust_remote_code=True, local_files_only=True,
        ).to(device).eval()
        resume_j = args.resume_j

    V = int(model.config.vocab_size)
    L = int(model.config.num_hidden_layers)
    print(f"[selftest] L={L} resume_j={resume_j} vocab={V} dtype={dtype} device={device}")

    qc_std = QCMemModel(model, resume_j=resume_j, block_diagonal=False)
    qc_bd = QCMemModel(model, resume_j=resume_j, block_diagonal=True)

    bos = 1

    @torch.no_grad()
    def rand_ids(n):
        return torch.randint(2, V, (1, n), device=device)

    def encode(ids_list):
        # write each chunk to depth j
        return [qc_std.write_chunk(x) for x in ids_list]

    torch.manual_seed(123)
    sink_ids = torch.tensor([[bos]], device=device)
    sink_hj = qc_std.write_chunk(sink_ids)

    # ---------------- (1) SINGLE-CHUNK EQUIVALENCE ----------------
    c1 = rand_ids(31)
    q = rand_ids(17)
    ctx_hj = encode([c1])
    q_hj = qc_std.write_chunk(q)

    with torch.no_grad():
        out_std = qc_std.read(sink_hj, ctx_hj, q_hj).float()
        out_bd = qc_bd.read(sink_hj, ctx_hj, q_hj).float()
    diff1 = (out_std - out_bd).abs().max().item()
    tol = 1e-4
    pass1 = diff1 < tol
    print(f"  (1) single-chunk  max|std - blockdiag| = {diff1:.3e}  "
          f"{'PASS (identical)' if pass1 else 'FAIL (should be identical)'}")

    # ---------------- (2) MULTI-CHUNK DIVERGENCE ----------------
    c2, c3, c4 = rand_ids(29), rand_ids(37), rand_ids(23)
    ctx_hj4 = encode([c1, c2, c3, c4])
    with torch.no_grad():
        out_std4 = qc_std.read(sink_hj, ctx_hj4, q_hj).float()
        out_bd4 = qc_bd.read(sink_hj, ctx_hj4, q_hj).float()
    diff2 = (out_std4 - out_bd4).abs().max().item()
    pass2 = diff2 > 1e-2
    print(f"  (2) multi-chunk   max|std - blockdiag| = {diff2:.3e}  "
          f"{'PASS (differ)' if pass2 else 'FAIL (should differ)'}")

    # ---------------- (3) MASK STRUCTURE ----------------
    # Rebuild the packed layout and inspect the raw keep-mask directly.
    pieces = [sink_hj] + ctx_hj4 + [q_hj]
    packed = torch.cat(pieces, dim=1)
    H = packed.shape[1]
    positions = torch.arange(H, device=device).unsqueeze(0)
    seg_lens = [("sink", sink_hj.shape[1])] + \
               [("chunk", h.shape[1]) for h in ctx_hj4] + \
               [("query", q_hj.shape[1])]
    bd_mask, _ = qc_bd._make_block_diagonal_mask_and_rope(packed, positions, seg_lens)
    std_mask = qc_std._make_mask_and_rope(packed, positions)[0]

    def to_keep_bool(m):
        if m is None:
            # pure-causal skip => build causal keep explicitly
            ri = torch.arange(H, device=device).view(H, 1)
            ci = torch.arange(H, device=device).view(1, H)
            return (ci <= ri).view(1, 1, H, H)
        if m.dtype == torch.bool:
            return m
        return m >= (torch.finfo(m.dtype).min / 2)  # additive: keep where ~0

    bd_keep = to_keep_bool(bd_mask)[0, 0]
    std_keep = to_keep_bool(std_mask)[0, 0]
    subset = bool((bd_keep & ~std_keep).sum().item() == 0)  # bd ⊆ std
    removed = int((std_keep & ~bd_keep).sum().item())
    pass3 = subset and removed > 0
    print(f"  (3) mask subset of causal={subset}, edges removed={removed}  "
          f"{'PASS' if pass3 else 'FAIL'}")

    # Extra: verify a context-chunk row attends only to sink + own block.
    # Row inside chunk c2 (block id 1). Layout: sink(1) c1(31) c2(29) ...
    row = 1 + 31 + 3  # a position inside c2
    row_keep = bd_keep[row]
    sink_col = torch.zeros(H, dtype=torch.bool, device=device)
    sink_col[0] = True
    c2_cols = torch.zeros(H, dtype=torch.bool, device=device)
    c2_start = 1 + 31
    c2_cols[c2_start:c2_start + 29] = True
    # allowed = sink + within c2 up to row (causal). No c1/c3/c4/query cols.
    allowed = sink_col | (c2_cols & (torch.arange(H, device=device) <= row))
    ok_row = bool((row_keep == allowed).all().item())
    print(f"  (3b) context-chunk row attends only to sink+own-block(causal): "
          f"{'PASS' if ok_row else 'FAIL'}")

    all_ok = pass1 and pass2 and pass3 and ok_row
    print("-" * 60)
    print(f"BLOCKDIAG SELF-TEST: {'ALL PASS' if all_ok else 'FAILURE'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
