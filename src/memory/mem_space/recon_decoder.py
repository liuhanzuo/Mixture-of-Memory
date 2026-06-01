"""MemoryReconDecoder (P1, v12, 2026-06-01): summary-reconstruction decoder.

WHY THIS EXISTS
---------------
The 2-chunk passcode toy task (scripts/toy_memory_bootstrap.py, commit e5bb181)
proved the decisive failure mode: routing CAN be fixed (slot_query+temp40 got
top1_sim=0.32, chunk1->chunk2 slot overlap=0.29), but **retrieval_exact_acc
stays 0.0** even when addressing succeeds. The written slot content is simply
not decodable back into the answer. The writer has no near-distance objective
teaching it to "store readable content" — it only gets the very indirect LM
loss signal.

This module supplies that missing signal. Given the slot VALUES written this
chunk (``M_write``), a small 1-layer cross-attention decoder must reconstruct
the chunk's L3 summary tokens. Optimising ``MSE(S_hat, stopgrad(S_L3))`` forces
the write path (``hidden_to_slot`` + the writeback gates) to deposit content
that is recoverable, not a generic drift toward the slot mean.

Design reference: status/MEMORY_PROTOCOL_PLAN.md [P1].

ARCHITECTURE (1 cross-attn block, pre-LN, residual)
----------------------------------------------------
    queries Q ∈ [num_summary, d_model]    (learnable)
    kv      = kv_proj(M_write) ∈ [B, k_write, d_model]
    S = Q (broadcast to B)
    S = S + cross_attn(LN_q(S), LN_kv(kv), LN_kv(kv))
    S = S + FFN(LN_ffn(S))
    S_hat = LN_out(S)                       ∈ [B, num_summary, d_model]

The decoder is intentionally tiny (1 block) — it must not be powerful enough to
reconstruct S_L3 from *generic* slots; the only way to drive the loss down is
for the slots to actually carry the chunk's content.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryReconDecoder(nn.Module):
    """Reconstruct L3 summary tokens from the written slot values.

    Args:
        d_model: backbone hidden dim (== L3 summary token dim, the recon target).
        d_slot: slot value dim (M_write last dim). Projected to d_model for
            cross-attention. Equals d_model when config.slot_dim is None.
        num_summary: number of summary queries / output tokens. MUST equal the
            L3 summary token count (l3_n_summary) so MSE against S_L3 aligns.
        num_heads: cross-attention heads.
        ffn_mult: FFN hidden-dim multiplier.
        dropout: attention dropout (0 for deterministic training).
    """

    def __init__(
        self,
        d_model: int,
        d_slot: int,
        num_summary: int,
        num_heads: int = 8,
        ffn_mult: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_slot = d_slot
        self.num_summary = num_summary

        # Learnable summary queries. Orthogonal init (same rationale as the L3
        # pool's queries, l3_summary.py v9): keeps the queries maximally diverse
        # at init so each one can specialise onto a different part of M_write
        # instead of all collapsing onto the slot mean.
        self.queries = nn.Parameter(torch.empty(num_summary, d_model))
        nn.init.orthogonal_(self.queries)

        # Project slot values into model space for cross-attention K/V.
        self.kv_proj = nn.Linear(d_slot, d_model)

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
        self.ln_out = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.kv_proj.weight)
        if self.kv_proj.bias is not None:
            nn.init.zeros_(self.kv_proj.bias)
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, M_write: torch.Tensor) -> torch.Tensor:
        """Reconstruct summary tokens from written slot values.

        Args:
            M_write: [B, k_write, d_slot] — post-write VALUES of the slots that
                were updated this chunk (gradient-bearing, so gradient flows
                back into the write path).

        Returns:
            S_hat: [B, num_summary, d_model] — reconstructed summary tokens.
        """
        if M_write.dim() != 3:
            raise ValueError(
                f"M_write must be [B, k_write, d_slot], got {tuple(M_write.shape)}"
            )
        if M_write.shape[-1] != self.d_slot:
            raise ValueError(
                f"M_write last-dim {M_write.shape[-1]} != d_slot {self.d_slot}"
            )
        B = M_write.shape[0]

        kv = self.kv_proj(M_write)                          # [B, k_write, d]
        S = self.queries.unsqueeze(0).expand(B, -1, -1)     # [B, num_summary, d]

        Sn = self.ln_q(S)
        kvn = self.ln_kv(kv)
        attn_out, _ = self.cross_attn(Sn, kvn, kvn, need_weights=False)
        S = S + attn_out

        S = S + self.ffn(self.ln_ffn(S))
        S = self.ln_out(S)
        return S

    @staticmethod
    def recon_loss(S_hat: torch.Tensor, S_L3: torch.Tensor) -> torch.Tensor:
        """MSE(S_hat, stopgrad(S_L3)).

        S_L3 is detached so the L3 summary pool is NOT pulled toward whatever
        the slots happen to encode (which would let the pair collapse into a
        trivial mutually-agreed representation). The gradient flows ONLY into
        the recon decoder and, through M_write, into the slot write path.
        """
        return F.mse_loss(S_hat.float(), S_L3.detach().float())
