"""Slot Memory Compressor — inverse attention with reconstruction loss.

Key difference from RMT: slots attend TO the segment (slots-as-queries),
then the segment reconstructs from updated slots. No cross-attention extractors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SlotMemoryCompressor(nn.Module):
    """Compresses segment hidden states into learnable memory slots.

    Forward:
        1. Slots (queries) attend to segment (keys/values) → updated slots
        2. Segment (queries) attends back to updated slots (keys/values) → reconstruction
        3. Gated reconstruction vs original → MSE loss
        4. Return (updated_slots, recon_loss, gate)

    Args:
        hidden_dim: model hidden dimension (e.g. 4096 for Qwen3-8B)
        num_slots: number of memory slots
        slot_dim: internal slot projection dimension
        top_k: optional sparse routing (reserved, not implemented yet)
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_slots: int = 64,
        slot_dim: int = 256,
        top_k: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Learnable slot embeddings
        self.slot_embeddings = nn.Parameter(
            torch.randn(num_slots, hidden_dim) * 0.02
        )

        # Projection layers (slots ↔ segment via bottleneck)
        self.slot_q_proj = nn.Linear(hidden_dim, slot_dim, bias=False)
        self.seg_k_proj = nn.Linear(hidden_dim, slot_dim, bias=False)
        self.seg_v_proj = nn.Linear(hidden_dim, slot_dim, bias=False)
        self.slot_out_proj = nn.Linear(slot_dim, hidden_dim, bias=False)

        # Reconstruction: segment queries to slot keys/values
        self.recon_q_proj = nn.Linear(hidden_dim, slot_dim, bias=False)
        self.recon_out_proj = nn.Linear(slot_dim, hidden_dim, bias=False)

        # Per-slot learnable gate (initialized near 1 so reconstruction flows)
        self.gate_bias = nn.Parameter(torch.ones(num_slots) * 2.0)

        # Layer norms
        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.recon_norm = nn.LayerNorm(hidden_dim)

        # Scaling
        self.scale = slot_dim ** -0.5

    def _slot_attention(
        self,
        slots: torch.Tensor,       # [B, S, D]
        segment: torch.Tensor,     # [B, T, D]
    ) -> torch.Tensor:
        """Slots attend to segment → updated slots."""
        B, S, D = slots.shape
        T = segment.shape[1]

        q = self.slot_q_proj(slots)       # [B, S, slot_dim]
        k = self.seg_k_proj(segment)      # [B, T, slot_dim]
        v = self.seg_v_proj(segment)      # [B, T, slot_dim]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, S, T]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, S, slot_dim]
        out = self.slot_out_proj(out)  # [B, S, D]
        out = self.slot_norm(slots + out)  # residual + norm
        return out

    def _reconstruct(
        self,
        segment: torch.Tensor,  # [B, T, D]
        slots: torch.Tensor,    # [B, S, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Segment attends to slots → reconstructed hidden states + gate."""
        B, T, D = segment.shape
        S = slots.shape[1]

        q = self.recon_q_proj(segment)   # [B, T, slot_dim]
        k = self.seg_k_proj(slots)       # [B, S, slot_dim]
        v = self.seg_v_proj(slots)       # [B, S, slot_dim]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, T, S]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, T, slot_dim]
        out = self.recon_out_proj(out)  # [B, T, D]
        out = self.recon_norm(out)

        # Per-slot gate applied to output (broadcast over segment)
        gate = torch.sigmoid(self.gate_bias).unsqueeze(0).unsqueeze(0)  # [1, 1, S]
        # Since attn mixes slots, we apply a scalar gate
        scalar_gate = gate.mean(dim=-1, keepdim=True)  # [1, 1, 1] — simple avg gate
        # Actually let's use a per-position gate based on reconstruction confidence
        # For simplicity, use a learned scalar gate:
        scalar_gate = torch.sigmoid(self.gate_bias.mean()).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1]
        reconstruction = scalar_gate * out
        return reconstruction, scalar_gate

    def forward(
        self,
        segment_hidden: torch.Tensor,  # [B, T, D]
        prev_slots: Optional[torch.Tensor] = None,  # [B, S, D] or None
        prev_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress segment into slots with reconstruction loss.

        Args:
            segment_hidden: hidden states of current segment [B, T, D]
            prev_slots: slots from previous segment (for accumulation)
            prev_gate: previous gate value

        Returns:
            updated_slots: [B, S, D]
            recon_loss: scalar MSE reconstruction loss
            gate: scalar gate value
        """
        B = segment_hidden.shape[0]
        device = segment_hidden.device
        dtype = segment_hidden.dtype

        # Initial slots: either previous or learnable embeddings
        if prev_slots is None:
            slots = self.slot_embeddings.unsqueeze(0).expand(B, -1, -1).to(dtype=dtype)
        else:
            slots = prev_slots

        # Step 1: Slots attend to segment → updated slots
        updated_slots = self._slot_attention(slots, segment_hidden)

        # Step 2: Segment reconstructs from updated slots
        reconstruction, gate = self._reconstruct(segment_hidden, updated_slots)

        # Step 3: Reconstruction loss (no detach — full gradient flow through slots)
        # Use mean over sequence dimension
        recon_loss = F.mse_loss(reconstruction, segment_hidden)

        return updated_slots, recon_loss, gate
