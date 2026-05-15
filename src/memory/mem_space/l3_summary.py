"""L3 Summary Pool: produces K dense summary tokens per chunk via Q-Former-style
attention pool over the chunk's top-layer hidden states.

Architecture (per cross-attn block):
    queries Q ∈ [K, d]               (learnable)
    K, V ← chunk_hidden_states ∈ [B, T, d]
    S = LN_q(Q) → cross_attn(S, LN_kv(H), LN_kv(H)) + Q   (pre-LN residual)
    S = S + FFN(LN_ffn(S))                                   (pre-LN residual)

After all blocks:
    S = LN_out(S)

Output: S ∈ [B, K, d]

Design reference: docs/L3_SUMMARY_RESEARCH.md §4-5
Consumed by mem_space.layer.MemorySpaceLayer joint-attention extended sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _L3PoolBlock(nn.Module):
    """One block of (cross-attn + FFN), pre-LN, residual."""

    def __init__(self, d_model: int, num_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_ffn = nn.LayerNorm(d_model)
        ffn_hidden = ffn_mult * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(
        self,
        S: torch.Tensor,
        H: torch.Tensor,
        H_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Pre-LN cross-attention
        Sn = self.ln_q(S)
        Hn = self.ln_kv(H)
        # MultiheadAttention key_padding_mask: True == "ignore this position"
        kpm = ~H_mask.bool() if H_mask is not None else None
        attn_out, _ = self.cross_attn(
            Sn, Hn, Hn, key_padding_mask=kpm, need_weights=False,
        )
        S = S + attn_out

        # Pre-LN FFN
        S = S + self.ffn(self.ln_ffn(S))
        return S


class L3SummaryPool(nn.Module):
    """Attention-pool that compresses a chunk of T hidden states into K summary tokens.

    This module is standalone — no dependencies on mem_space.layer or mem_space.config.

    Args:
        d_model: hidden dimension (must match backbone d_model for direct concat).
        num_summary: K — number of output summary tokens per chunk.
        num_heads: number of attention heads in the cross-attn blocks.
        ffn_mult: FFN hidden-dim multiplier (ffn_hidden = ffn_mult * d_model).
        n_layers: number of cross-attn blocks (1 block ≈ 50-100M params, 2 ≈ 150M).
        dropout: attention dropout (0 for training without stochasticity in attn).
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_summary: int = 64,
        num_heads: int = 8,
        ffn_mult: int = 2,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_summary = num_summary
        self.n_layers = n_layers

        # Learnable query bank
        self.queries = nn.Parameter(torch.empty(num_summary, d_model))
        nn.init.normal_(self.queries, std=0.02)

        self.blocks = nn.ModuleList([
            _L3PoolBlock(d_model, num_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)

        # Xavier init for all linear layers in blocks
        self._init_weights()

    def _init_weights(self):
        for blk in self.blocks:
            for name, module in blk.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        chunk_hidden: torch.Tensor,
        chunk_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce summary tokens from a chunk's top-layer hidden states.

        Args:
            chunk_hidden: [B, T, d_model] — chunk's top-layer hidden states.
            chunk_mask: [B, T] — 1=valid token, 0=padding (optional).

        Returns:
            S: [B, K, d_model] — summary tokens for this chunk.
        """
        B = chunk_hidden.shape[0]
        # Expand learnable queries to batch
        S = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, d]

        for blk in self.blocks:
            S = blk(S, chunk_hidden, chunk_mask)

        S = self.ln_out(S)
        return S
