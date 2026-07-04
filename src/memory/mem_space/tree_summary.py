"""HNST v2 — trainable tree-summary pool (2026-06-25).

HNST v1 (``layer.py:_fifo_select_keep_set_tree``) built the navigation tree with
a fixed **max-pool** at every internal node. That killed the needle signal at the
upper levels (a single peak token in one of B*B*... leaves gets max-pooled with
irrelevant peaks and the routing salience of the top node no longer discriminates
which subtree holds the needle). v1 was KILLED: beam-1 recall ~20%, only matched
flat/keep-all.

v2 replaces the fixed aggregation with a **learnable attention pool** (one query,
Q-Former-style cross-attn block, reusing the tested ``L3SummaryPool`` machinery).
The pool is trained jointly with the (unfrozen) reader so that a parent summary
*retains the routing information of whichever child subtree holds the needle*.
The training signal is a per-level "pick the child subtree that contains the
needle" cross-entropy on the reader's own q.k salience (needle position is known
in the synthetic task → free supervision), plus the flat leaf CE.

Two poolers (both ``L3SummaryPool`` with ``num_summary=1``):
  * ``leaf_pool``: pools a chunk's ``[B, T, d]`` token hiddens → one ``[B, d]``
    leaf summary (replaces v1's per-chunk max-pool leaf).
  * ``node_pool``: pools a node's ``[B, n_children, d]`` children summaries → one
    ``[B, d]`` parent summary (replaces v1's max-over-children).

Both output in the layer-L INPUT-HIDDEN space so the reader's native
``k_proj(pre_norm(summary))`` scores them exactly as it scores raw chunk hiddens
(identical to ``_fifo_reader_attn_salience`` / ``_reader_attn_keep_set``). This is
what makes the trained tree navigation directly comparable to (and a drop-in for)
the flat reader-attn selector.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .l3_summary import L3SummaryPool


class TreeSummaryPool(nn.Module):
    """Learnable leaf + internal-node aggregation for the HNST v2 navigation tree.

    Args:
        d_model: hidden dim (must match backbone d_model so the reader's k_proj
            can score the summaries directly).
        num_heads: attention heads in each Q-Former pool block.
        ffn_mult: FFN hidden-dim multiplier inside each pool block.
        n_layers: number of cross-attn blocks per pool (1 ≈ 200M params @ d=4096).
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_heads: int = 8,
        ffn_mult: int = 2,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        # num_summary=1: each pool call collapses its input set to ONE summary vec.
        self.leaf_pool = L3SummaryPool(
            d_model=d_model, num_summary=1, num_heads=num_heads,
            ffn_mult=ffn_mult, n_layers=n_layers, dropout=0.0,
        )
        self.node_pool = L3SummaryPool(
            d_model=d_model, num_summary=1, num_heads=num_heads,
            ffn_mult=ffn_mult, n_layers=n_layers, dropout=0.0,
        )

    def pool_leaf(self, chunk_hidden: torch.Tensor) -> torch.Tensor:
        """chunk_hidden: [B, T, d] → leaf summary [B, d]."""
        s = self.leaf_pool(chunk_hidden)          # [B, 1, d]
        return s[:, 0, :]                          # [B, d]

    def pool_node(self, children: torch.Tensor) -> torch.Tensor:
        """children: [B, n_children, d] → parent summary [B, d]."""
        s = self.node_pool(children)              # [B, 1, d]
        return s[:, 0, :]                          # [B, d]

    def build_levels(self, leaves: torch.Tensor, branch: int) -> list:
        """Build the B-ary tree bottom-up from a stack of leaf summaries.

        Args:
            leaves: [C, B, d] — C leaf summaries (one per chunk), batch B.
            branch: B-ary branching factor.
        Returns:
            levels: list of [n_ℓ, B, d] tensors, level 0 == leaves, last == root.
            Node j at level ℓ covers CONTIGUOUS children [j*branch : (j+1)*branch]
            of level ℓ-1 (same contiguous grouping as v1, so the needle→ancestor
            index map is ``ancestor = leaf // branch**ℓ``).
        """
        levels = [leaves]
        cur = leaves
        while cur.shape[0] > 1:
            n = cur.shape[0]
            groups = (n + branch - 1) // branch
            parts = []
            for g in range(groups):
                s = g * branch
                e = min(n, (g + 1) * branch)
                child_block = cur[s:e].transpose(0, 1)      # [B, n_child, d]
                parts.append(self.pool_node(child_block))   # [B, d]
            cur = torch.stack(parts, dim=0)                 # [groups, B, d]
            levels.append(cur)
        return levels

    def build_readout_prefix(
        self,
        buffer_chunks: List[torch.Tensor],
        branch: int,
        fine_chunks: int = 0,
    ) -> Tuple[torch.Tensor, int]:
        """Aggregate a FIFO buffer into a multi-scale tree READOUT prefix (exp 2).

        This is the readout-side twin of ``build_levels`` (which is used for the
        eval keep-set *navigation*). Instead of choosing WHICH raw chunks to
        keep, we COMPRESS the whole buffer into one condensed prefix the reader
        consumes directly: every chunk collapses to a single learned leaf summary
        (``T`` tokens -> 1), and the internal tree nodes add progressively coarser
        aggregates. Compared with the pure-FIFO ``torch.cat(kept_chunks)`` prefix
        (``C * T`` tokens), this is a ``~C``-token multi-scale prefix (leaves +
        internal nodes) — the "tree aggregation vs naive FIFO concat" comparison.

        Ordering: OLDEST -> NEWEST so the causal prefix stays in document order.
        We emit each level bottom-up but keep each level in oldest->newest chunk
        order, and place COARSER (higher) levels BEFORE finer ones so the finest
        (leaf, most recent) tokens sit closest to the current chunk (near-fine /
        far-coarse, same locality intuition as BeaconPyramid).

        Args:
            buffer_chunks: list of [B, T_c, d] chunk hiddens, OLDEST -> NEWEST.
            branch: B-ary branching factor for the internal-node pooling.
            fine_chunks: # most-recent chunks kept RAW (no leaf pooling) so the
                near-query context keeps full token fidelity. 0 (default) => the
                whole buffer is tree-compressed.

        Returns:
            (prefix, P) where prefix is [B, P, d] ordered OLDEST -> NEWEST.
        """
        C = len(buffer_chunks)
        if C == 0:
            raise ValueError("build_readout_prefix called with empty buffer")
        g = max(2, int(branch))
        f = max(0, min(int(fine_chunks), C))
        B = buffer_chunks[0].shape[0]
        d = self.d_model

        # Split: [0, tree_hi) are tree-compressed; [tree_hi, C) kept RAW (near).
        tree_hi = C - f

        pieces: List[torch.Tensor] = []
        if tree_hi > 0:
            # Leaf summaries for the compressed span (one [B, d] per chunk).
            leaves = torch.stack(
                [self.pool_leaf(buffer_chunks[i]) for i in range(tree_hi)],
                dim=0,
            )                                                  # [tree_hi, B, d]
            levels = self.build_levels(leaves, g)              # list of [n_ℓ, B, d]
            # Emit COARSEST -> FINEST (far-coarse first), each level oldest->newest.
            for lvl in reversed(levels):
                # lvl: [n_ℓ, B, d] -> [B, n_ℓ, d]
                pieces.append(lvl.transpose(0, 1))
        # NEAR band (newest): raw chunk hiddens, no pooling.
        for i in range(tree_hi, C):
            pieces.append(buffer_chunks[i])                    # [B, T_c, d]

        prefix = torch.cat(pieces, dim=1)                      # [B, P, d]
        return prefix, prefix.shape[1]
