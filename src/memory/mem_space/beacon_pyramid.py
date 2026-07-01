"""Hierarchical Beacon Pyramid — multi-scale compressed FIFO prefix (idea #3, 2026-07-01).

Motivation
----------
Activation Beacon (2401.03462) trains the *reader* to consume a small set of
compressed "beacon" tokens that summarise past context. Idea #3 layers this into
a distance-stratified pyramid:

  * the BEACON tokens are the leaves;
  * near-query context is kept at FINE granularity (raw or many beacons);
  * far context is COARSE (fewer beacons, pooled bottom-up with a fixed
    compression ratio) — a multi-scale "near-fine / far-coarse" prefix.

Crucially — and this is the difference from the falsified *compress-then-inject
into a FROZEN reader* line — the reader is trained JOINTLY to consume these
beacons (the beacon pool + a sparse set of reader layers are unfrozen). The pool
params get a clean gradient path exactly the way ``L3SummaryPool`` /
``L2Compressor`` do in this codebase (a shared singleton applied to the detached
per-chunk hiddens inside the current chunk's forward, so ``.backward()`` reaches
the pool weights).

Design
------
This module wraps two ``L3SummaryPool`` attention pools (num_summary=K), reusing
the tested Q-Former pooling machinery (orthogonal query init, pre-LN cross-attn,
output in the layer-INPUT-hidden space so the wrapped decoder layer scores the
beacons with its native ``k_proj(input_layernorm(·))``, identical to how it
scores raw chunk hiddens in ``_forward_fifo``):

  * ``chunk_pool``: [B, T, d] chunk tokens         -> [B, K, d] beacons (leaf).
  * ``group_pool``: [B, K*g, d] children beacons   -> [B, K, d] beacons (parent).

``build_prefix`` takes the FIFO buffer (list of [B, T_c, d] chunk hiddens, oldest
-> newest) and returns a single [B, P, d] multi-scale prefix ordered oldest ->
newest (so the causal prefix stays in document order), plus P.

Bands, by recency distance d (0 == most-recent buffered chunk):
  * NEAR  (d < fine_chunks)              : RAW chunk hiddens (finest, no pooling).
  * MID   (fine_chunks <= d < fine+mid)  : per-chunk K beacons (compression T/K).
  * FAR   (d >= fine_chunks+mid_chunks)  : groups of ``branch`` chunks pooled to
                                           K beacons each (compression branch*T/K).

Setting mid_chunks huge (>= buffer) with fine_chunks=0 gives a SINGLE-SCALE beacon
ablation (every chunk -> K beacons, no raw, no grouping). Setting fine_chunks >=
buffer recovers the raw FIFO prefix (no pooling) == pure-FIFO baseline. This lets
the pyramid / single-scale / raw arms share one code path.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .l3_summary import L3SummaryPool


class BeaconPyramid(nn.Module):
    """Shared multi-scale beacon pool (one instance used by every FIFO layer).

    A single shared instance is applied at every patched layer to THAT layer's
    own FIFO buffer (each layer's buffer lives in that layer's input-hidden
    space). This mirrors the ``L2Compressor`` "one shared instance across all 32
    layers" pattern already used in this repo — the pool learns a generic
    "summarise a set of d-dim vectors" operator.

    Args:
        d_model: hidden dim (== backbone d_model so the reader's k_proj/v_proj
            score the beacons directly, same space as raw chunk hiddens).
        beacon_k: K — number of beacon tokens produced per pooled group.
        num_heads / ffn_mult / n_layers: Q-Former pool block hyper-params.
    """

    def __init__(
        self,
        d_model: int = 4096,
        beacon_k: int = 8,
        num_heads: int = 8,
        ffn_mult: int = 2,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.beacon_k = int(beacon_k)
        # Leaf pool: chunk tokens -> K beacons. Node pool: children beacons -> K.
        self.chunk_pool = L3SummaryPool(
            d_model=d_model, num_summary=beacon_k, num_heads=num_heads,
            ffn_mult=ffn_mult, n_layers=n_layers, dropout=0.0,
        )
        self.group_pool = L3SummaryPool(
            d_model=d_model, num_summary=beacon_k, num_heads=num_heads,
            ffn_mult=ffn_mult, n_layers=n_layers, dropout=0.0,
        )

    def pool_chunk(self, chunk_hidden: torch.Tensor) -> torch.Tensor:
        """[B, T, d] -> [B, K, d] beacons for one chunk (leaf level)."""
        return self.chunk_pool(chunk_hidden)

    def pool_group(self, children: torch.Tensor) -> torch.Tensor:
        """[B, K*g, d] children beacons -> [B, K, d] parent beacons (coarser)."""
        return self.group_pool(children)

    def build_prefix(
        self,
        buffer_chunks: List[torch.Tensor],
        fine_chunks: int,
        mid_chunks: int,
        branch: int,
    ) -> Tuple[torch.Tensor, int]:
        """Build the multi-scale beacon prefix from the FIFO buffer.

        Args:
            buffer_chunks: list of [B, T_c, d] chunk hiddens, OLDEST -> NEWEST.
            fine_chunks:  # most-recent chunks kept RAW (near-fine band).
            mid_chunks:   # chunks (just before the fine band) pooled per-chunk
                          to K beacons each (mid band).
            branch:       far-band grouping factor (chunks per parent group).

        Returns:
            (prefix, P) where prefix is [B, P, d] ordered OLDEST -> NEWEST so it
            stays a valid causal document-order prefix.
        """
        C = len(buffer_chunks)
        if C == 0:
            raise ValueError("build_prefix called with empty buffer")
        f = max(0, int(fine_chunks))
        m = max(0, int(mid_chunks))
        g = max(2, int(branch))

        # Partition buffer indices [0..C-1] (oldest..newest) by recency distance
        # d = (C-1) - i  (d==0 is the newest). Near = smallest d.
        far_hi = C - f - m          # far band = indices [0, far_hi)
        mid_lo = max(0, far_hi)     # mid band = [mid_lo, C-f)
        near_lo = max(0, C - f)     # near band = [near_lo, C)

        pieces: List[torch.Tensor] = []

        # FAR band (oldest): group consecutive chunks by `branch`, pool the
        # group's raw tokens -> K coarse beacons. We pool the concatenated raw
        # tokens of the group directly (one attention pool over branch*T tokens)
        # rather than pool-then-pool, which is cheaper and keeps a single grad
        # hop to the pool. (This is the "coarser = larger receptive field" scale.)
        if far_hi > 0:
            i = 0
            while i < far_hi:
                grp = buffer_chunks[i:min(far_hi, i + g)]
                cat = torch.cat(grp, dim=1)                 # [B, <=g*T, d]
                pieces.append(self.pool_group(cat))         # [B, K, d]
                i += g

        # MID band: each chunk -> K beacons.
        for i in range(mid_lo, near_lo):
            pieces.append(self.pool_chunk(buffer_chunks[i]))  # [B, K, d]

        # NEAR band (newest): raw hiddens, no pooling.
        for i in range(near_lo, C):
            pieces.append(buffer_chunks[i])                   # [B, T_c, d]

        prefix = torch.cat(pieces, dim=1)                     # [B, P, d]
        return prefix, prefix.shape[1]


__all__ = ["BeaconPyramid"]
