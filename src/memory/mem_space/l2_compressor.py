"""L2 token-compressed KV memory — NSA / DeepSeek-V4-CSA-style learned attention pool.

For each chunk of T tokens we produce  T/g  latent KV vectors of dim  d_c (+ d_h_R
decoupled-RoPE component). The previous chunk's latents are read by the current
chunk's joint attention (prepended into the wrapped LlamaDecoderLayer's K/V cache
via ``MemorySpaceLayer.forward``).

Reference:
    docs/L2_DEEPSEEK_MLA_RESEARCH.md §5
    docs/L2_IMPLEMENTATION_PLAN_20260516.md §1, §2

Design highlights:
    * Soft attention pooling (V4 CSA / NSA) — wkv + wgate + APE-biased softmax over
      windows of size g. NOT mean-pool (avoids salience dilution).
    * Decoupled-RoPE component (MLA-style) — last d_h_rope dims of the latent are
      computed from the *first* h of each window via a separate linear w_kR.
    * One shared instance is created across all 32 layers (peer to L3SummaryPool).
      We register a post-forward hook on the LAST patched layer to recompute
      ``prev_latents`` at chunk boundary — the next chunk's layers read this state.
    * Cold-start near-zero init on ``kv_b`` (std=l2_init_scale=0.001) so L2's
      contribution to attention starts ≈ 0. Mirrors the Flamingo gate pattern.
    * Cross-chunk state (``prev_latents``) is a non-persistent buffer — does NOT
      appear in state_dict (resets between documents in eval, between optimizer
      steps in training).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Compressor(nn.Module):
    """Single shared L2 token compressor (one instance, used by all 32 layers).

    Args:
        d_model: backbone hidden dim (e.g. 4096 for Llama-3-8B).
        n_heads: backbone attention heads (e.g. 32).
        d_head: backbone per-head dim (e.g. 128). d_model = n_heads * d_head.
        compress_ratio: g — group / window size (e.g. 16 → 4096-token chunk → 256 latents).
        d_c: latent / content dim (default 512, matches V2 MLA).
        d_h_rope: decoupled-RoPE per-latent dim (default 64, matches V2 MLA).
        chunk_size: outer chunk size (kept for clarity; latent count = chunk_size/g).
        n_kv_heads: number of KV heads (for GQA). For Llama-3-8B MHA, equals n_heads=32.
        init_scale: std for kv_b weight init. Near-zero (0.001) so L2 starts ≈ no-op.
    """

    def __init__(
        self,
        d_model: int = 4096,
        n_heads: int = 32,
        d_head: int = 128,
        compress_ratio: int = 16,
        d_c: int = 512,
        d_h_rope: int = 64,
        chunk_size: int = 4096,
        n_kv_heads: int | None = None,
        init_scale: float = 0.001,
    ):
        super().__init__()
        if d_model != n_heads * d_head:
            raise ValueError(
                f"d_model={d_model} must equal n_heads*d_head={n_heads*d_head}"
            )
        if compress_ratio <= 0:
            raise ValueError(f"compress_ratio must be > 0, got {compress_ratio}")
        if d_c <= 0 or d_h_rope < 0:
            raise ValueError(
                f"d_c must be > 0 and d_h_rope >= 0; got d_c={d_c}, d_h_rope={d_h_rope}"
            )
        if n_kv_heads is None:
            n_kv_heads = n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.g = compress_ratio
        self.d_c = d_c
        self.d_h_R = d_h_rope
        self.chunk_size = chunk_size
        self.init_scale = init_scale

        # Content / gate / RoPE projections (no bias — matches V4 CSA & MLA).
        self.w_kv = nn.Linear(d_model, d_c, bias=False)
        self.w_gate = nn.Linear(d_model, d_c, bias=False)
        self.w_kR = nn.Linear(d_model, d_h_rope, bias=False) if d_h_rope > 0 else None

        # Learned absolute-position-in-window bias (APE):  [g, d_c].
        # Init zeros — gate-softmax starts uniform within each window.
        self.ape = nn.Parameter(torch.zeros(compress_ratio, d_c))

        # RMSNorm on the latent (over d_c).
        self.norm = nn.RMSNorm(d_c)

        # Up-projection that reconstructs (K, V) back into model space (single
        # linear, output split into K, V each of shape [n_kv_heads * d_head]).
        self.kv_b = nn.Linear(d_c, 2 * n_kv_heads * d_head, bias=False)

        # Init weights:
        #   * w_kv / w_gate / w_kR: std=0.02 (standard transformer init)
        #   * kv_b: std=init_scale (default 0.001) — near-zero so L2 contribution
        #     to attention starts ≈ 0. Matches Flamingo gating cold-start pattern.
        nn.init.normal_(self.w_kv.weight, std=0.02)
        nn.init.normal_(self.w_gate.weight, std=0.02)
        if self.w_kR is not None:
            nn.init.normal_(self.w_kR.weight, std=0.02)
        nn.init.normal_(self.kv_b.weight, std=init_scale)
        # APE already zeros via Parameter init above.

        # Cross-chunk state: [B, n_latents, d_c + d_h_R] of latents from prior chunk.
        # NOT persistent (does not appear in state_dict). Reset between documents.
        self.register_buffer("prev_latents", torch.empty(0), persistent=False)

    @torch.no_grad()
    def reset(self):
        """Called by training/eval loop at chunk boundary == document boundary.

        Zeros the cross-chunk state so the next chunk starts cold (no L2 read).
        """
        self.prev_latents = self.prev_latents.new_empty(0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass — delegates to compress(). Required for FSDP to unshard params."""
        return self.compress(h)

    def compress(self, h: torch.Tensor) -> torch.Tensor:
        """Compress a chunk of hidden states into per-window latents.

        Args:
            h: [B, T, d_model] — current chunk's hidden states (post the last
                patched layer; passed in detached — see patch.py post-forward hook).

        Returns:
            [B, T_padded/g, d_c + d_h_R] — compressed latents (one per window of
            g tokens). T is right-padded to a multiple of g if needed (padding
            positions contribute zero through the gate softmax).
        """
        B, T, d = h.shape
        if d != self.d_model:
            raise RuntimeError(
                f"L2Compressor.compress: hidden last-dim {d} != d_model {self.d_model}"
            )
        g, d_c, d_R = self.g, self.d_c, self.d_h_R

        # Right-pad to multiple of g if needed (padding contributes 0 via -inf gate).
        pad = 0
        if T % g != 0:
            pad = g - (T % g)
            h = F.pad(h, (0, 0, 0, pad))
            T = T + pad

        # 1. Project to content (kv) and gate scores.
        #    Use the weight's dtype (bf16 in training, fp32 in unit tests) for
        #    the linear ops; promote to fp32 only for the softmax + RMSNorm
        #    where numerical stability matters. This avoids dtype mismatches
        #    when the model is cast to bf16 by the trainer.
        weight_dtype = self.w_kv.weight.dtype
        h_in = h if h.dtype == weight_dtype else h.to(weight_dtype)
        kv = self.w_kv(h_in)             # [B, T, d_c]
        gate = self.w_gate(h_in)         # [B, T, d_c]

        # 2. Unflatten into windows of size g.
        kv_w = kv.unflatten(1, (-1, g))                 # [B, T/g, g, d_c]
        gate_w = gate.unflatten(1, (-1, g))             # [B, T/g, g, d_c]
        # APE broadcasts over (B, T/g):  ape: [g, d_c] → [1, 1, g, d_c]
        gate_w = gate_w + self.ape.unsqueeze(0).unsqueeze(0)

        # Promote to fp32 for the softmax (bf16 softmax can be unstable on
        # large gate magnitudes).
        gate_w_f = gate_w.float()

        # 3. If we padded, mask out the padded positions in the LAST window so
        #    they contribute 0 to the softmax. (Otherwise zero-h tokens still
        #    get a valid gate logit and pollute the pooled latent.)
        if pad > 0:
            n_windows = kv_w.shape[1]
            valid_in_last = g - pad   # number of valid tokens in last window
            neg_inf = torch.finfo(gate_w_f.dtype).min
            gate_w_f[:, n_windows - 1, valid_in_last:, :] = neg_inf

        # 4. Soft-pool: softmax over within-window dim (dim=2).
        weights = gate_w_f.softmax(dim=2).to(kv_w.dtype)     # [B, T/g, g, d_c]
        latent = (kv_w * weights).sum(dim=2)                 # [B, T/g, d_c]
        latent = self.norm(latent)                           # RMSNorm over d_c

        # 5. Decoupled-RoPE component — computed from FIRST h of each window.
        if self.w_kR is not None and d_R > 0:
            h_w_first = h_in.unflatten(1, (-1, g))[:, :, 0, :]   # [B, T/g, d_model]
            kR = self.w_kR(h_w_first)                            # [B, T/g, d_h_R]
            out = torch.cat([latent, kR], dim=-1)                # [B, T/g, d_c + d_h_R]
        else:
            out = latent

        return out.to(h.dtype)


__all__ = ["L2Compressor"]
