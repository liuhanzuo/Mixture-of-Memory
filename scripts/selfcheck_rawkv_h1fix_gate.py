#!/usr/bin/env python3
"""H1-fix launch-gate self-check: confirm the gist selection is NO LONGER a
no-op under the new config (topk_chunks=2 << n_ctx=16) and that the resulting
col_bias is NON-CONSTANT across columns (i.e. the scorer can express a
discriminative selection — the prerequisite for the 2000-step run to be
meaningful).

Pure CPU, no model load. Builds a GistReadout + a synthetic RawKVReadoutStore
with C chunks of distinct random hidden, calls retrieve(topk_chunks=2), and
asserts:
  (1) keep_all == False  (top-k actually selects; not all chunks kept)
  (2) the kept set has exactly topk_chunks chunks
  (3) col_bias varies across the retrieved columns (std > 0) AND the per-chunk
      softmax weights are non-uniform (the scorer is expressive, not pinned).
Also exercises pool='max' on append to confirm the new code path runs.
"""
from __future__ import annotations

import sys
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.memory.mem_space.rawkv_readout import GistReadout, RawKVReadoutStore  # noqa: E402


def main():
    torch.manual_seed(0)
    d = 256
    C = 16          # n_ctx (matches t2_gap_tokens=8192 / chunk_size=512)
    T = 32          # tokens per chunk (small for the check)
    topk = 2
    B = 1

    gist = GistReadout(d_model=d, gist_dim=64)
    store = RawKVReadoutStore()
    # Distinct chunks: each chunk gets its own random offset so gist keys differ.
    for c in range(C):
        h = torch.randn(B, T, d) + c * 0.5  # per-chunk shift → distinguishable
        store.append_chunk(h, pool="max")   # exercise the new max-pool path
    assert store.n_chunks == C, store.n_chunks

    query = torch.randn(B, 4, d, requires_grad=True)
    ret = gist.retrieve(query, store, topk_chunks=topk, temperature=1.0)
    assert ret is not None, "retrieve returned None"
    ret_h, ret_pos, col_bias = ret

    # (1)+(2): keep_all must be False; kept set size == topk.
    keep_all = topk <= 0 or topk >= C
    R = ret_h.shape[1]
    kept_chunks = R // T   # uniform chunk length → R = kept * T
    print(f"C(n_ctx)={C} topk_chunks={topk} -> keep_all={keep_all}")
    print(f"retrieved R={R} tokens = {kept_chunks} chunks (expect {topk})")

    # (3): col_bias spread + per-chunk weight non-uniformity.
    cb = col_bias.detach()
    cb_std = float(cb.std().item())
    # Recompute the full C-way softmax weights to show non-uniformity.
    gkey = gist.key_proj(store.gist_src)
    gq = gist.query_proj(query)
    score = torch.einsum("bqg,bcg->bqc", gq, gkey) * gist._scale
    w = torch.softmax(score, dim=-1)[0, 0].detach()  # [C]
    uniform = 1.0 / C
    print(f"col_bias std across cols = {cb_std:.4f} (constant would be 0.0)")
    print(f"per-chunk weight vector (C={C}): "
          f"min={float(w.min()):.4f} max={float(w.max()):.4f} "
          f"uniform={uniform:.4f}  max/uniform={float(w.max())/uniform:.2f}x")

    # Gradient flows to BOTH projections through col_bias.
    loss = col_bias.sum()
    loss.backward()
    qg = gist.query_proj.weight.grad
    kg = gist.key_proj.weight.grad
    print(f"grad: query_proj={None if qg is None else float(qg.norm()):.4f} "
          f"key_proj={None if kg is None else float(kg.norm()):.4f}")

    ok1 = keep_all is False
    ok2 = kept_chunks == topk
    ok3 = cb_std > 1e-4 and float(w.max()) > 1.5 * uniform
    ok4 = qg is not None and kg is not None and qg.norm() > 0 and kg.norm() > 0
    print("\n=== LAUNCH GATE ===")
    print(f"(1) keep_all == False           : {ok1}")
    print(f"(2) kept set == topk_chunks({topk})  : {ok2}")
    print(f"(3) col_bias NON-constant+sharp : {ok3}")
    print(f"(4) grad to query+key proj      : {ok4}")
    allok = ok1 and ok2 and ok3 and ok4
    print(f"GATE: {'PASS — selection is real, safe to launch 2000 steps' if allok else 'FAIL — do NOT launch'}")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
