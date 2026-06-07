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


class L3TokenReconHead(nn.Module):
    """ICAE-style token-level reconstruction head (2026-06-07).

    WHY THIS EXISTS
    ---------------
    ``MemoryReconDecoder`` (above) reconstructs the CONTINUOUS L3 summary hidden
    and ``stopgrad``s the target, so the L3 pool feels no pressure to learn a
    semantic compression — it can store generic token-level hidden and the MSE
    still drops (trivial collapse). The diagnostic root-cause for LongBench
    F1≈3 (vs base 13.9) is exactly this: L3/slots learn token-level hidden, not
    semantic summaries.

    ICAE (Ge et al., ICLR'24, arXiv:2307.06945) fixes this by training memory
    slots to reconstruct the DISCRETE input TEXT via cross-entropy, with the
    gradient flowing back into the compressor. The K-token summary is a real
    information bottleneck, so the only way to drive CE down is to actually
    encode the chunk's content in a recoverable way.

    This head implements that. Given a (grad-bearing) L3 summary of the chunk
    ``S`` ∈ [B, K, d], it cross-attends T learnable positional queries onto S
    and produces per-position decoder hidden ``dec_h`` ∈ [B, T, d]. The caller
    then maps ``dec_h`` through the FROZEN backbone ``lm_head`` to get vocab
    logits [B, T, V] and computes ``CE(logits, chunk_input_ids)``. The target
    is NOT detached: gradient flows summary→l3_pool, teaching L3 to compress
    semantically.

    ARCHITECTURE (n_layers cross-attn blocks, pre-LN, residual)
    -----------------------------------------------------------
        pos_queries Q ∈ [max_positions, d]   (learnable, one per position)
        S = kv_proj(summary) ∈ [B, K, d]
        dec = Q[:T] (broadcast to B)
        for blk in blocks:
            dec = dec + cross_attn(LN_q(dec), LN_kv(S), LN_kv(S))
            dec = dec + FFN(LN_ffn(dec))
        dec_h = LN_out(dec)                    ∈ [B, T, d]

    The head outputs decoder HIDDEN (d_model), not logits — the (frozen, tied)
    ``lm_head`` is applied by the caller so we reuse the backbone's vocab
    projection and never train a fresh [d, V] matrix (V≈128k would dwarf the
    adapter). Output goes straight into the LM cross-entropy.
    """

    def __init__(
        self,
        d_model: int,
        max_positions: int,
        num_heads: int = 8,
        n_layers: int = 1,
        ffn_mult: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_positions = max_positions
        self.n_layers = n_layers

        # One learnable query per decode position. Normal init (these are
        # position embeddings, not a diversity-sensitive query bank).
        self.pos_queries = nn.Parameter(torch.empty(max_positions, d_model))
        nn.init.normal_(self.pos_queries, mean=0.0, std=0.02)

        # Project the L3 summary into the head's working space for K/V. Summary
        # is already d_model (L3SummaryPool output dim == backbone d_model), so
        # this is d_model->d_model — kept so the head can re-map the summary
        # without disturbing the shared L3 representation.
        self.kv_proj = nn.Linear(d_model, d_model)

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(d_model),
                "ln_kv": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                "ln_ffn": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, ffn_mult * d_model),
                    nn.GELU(),
                    nn.Linear(ffn_mult * d_model, d_model),
                ),
            }))
        self.ln_out = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.kv_proj.weight)
        if self.kv_proj.bias is not None:
            nn.init.zeros_(self.kv_proj.bias)
        for blk in self.blocks:
            for module in blk["ffn"]:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, summary: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode per-position hidden from the chunk's L3 summary.

        Args:
            summary: [B, K, d_model] — grad-bearing L3 summary of THIS chunk.
                Gradient flows through this tensor back into the L3 pool.
            seq_len: T — number of token positions to reconstruct (== the
                chunk's token count).

        Returns:
            dec_h: [B, T, d_model] — per-position decoder hidden. Caller maps
                this through the frozen lm_head to logits and computes CE.
        """
        if summary.dim() != 3:
            raise ValueError(
                f"summary must be [B, K, d_model], got {tuple(summary.shape)}"
            )
        if summary.shape[-1] != self.d_model:
            raise ValueError(
                f"summary last-dim {summary.shape[-1]} != d_model {self.d_model}"
            )
        if seq_len > self.max_positions:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_positions {self.max_positions}; "
                "increase config.l3_recon_max_positions (>= chunk_size)."
            )
        B = summary.shape[0]

        kv = self.kv_proj(summary)                              # [B, K, d]
        dec = self.pos_queries[:seq_len].unsqueeze(0).expand(B, -1, -1)  # [B,T,d]

        for blk in self.blocks:
            dn = blk["ln_q"](dec)
            kvn = blk["ln_kv"](kv)
            attn_out, _ = blk["attn"](dn, kvn, kvn, need_weights=False)
            dec = dec + attn_out
            dec = dec + blk["ffn"](blk["ln_ffn"](dec))

        dec_h = self.ln_out(dec)                                # [B, T, d]
        return dec_h
