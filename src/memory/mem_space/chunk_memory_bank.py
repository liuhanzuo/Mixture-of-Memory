"""Per-layer chunk memory bank for V4: Per-Layer Sparse Memory Bank.

Pure runtime state -- NOT an nn.Module.  DDP ignores it completely.

Phase 1 only: deterministic append until the bank is full.
Phase 2 (future): top-k selection + EMA update of selected slots.

Design reference: versions/v4_chunk_last_hidden_memory.md
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class ChunkMemoryBank:
    """Per-layer, per-sample memory bank.  NOT an nn.Module.

    Each slot stores a detached hidden-state vector from the last token
    of a chunk at the corresponding decoder layer.  The bank persists
    across chunks within a document and is reset at document boundaries.
    """

    def __init__(
        self,
        num_slots: int,
        d_model: int,
        ema_decay: float = 0.9,
    ) -> None:
        self.num_slots = num_slots
        self.d_model = d_model
        self.ema_decay = ema_decay
        self.slots: torch.Tensor | None = None  # [B, N, d], lazy init
        self.num_filled: int = 0

    # ------------------------------------------------------------------ #
    # Phase 1: append-only
    # ------------------------------------------------------------------ #

    def append(self, hidden: torch.Tensor) -> None:
        """Append a new slot.  ``hidden``: [B, d], always detached on write."""
        if self.slots is None:
            B, d = hidden.shape
            self.slots = torch.zeros(
                B, self.num_slots, d,
                dtype=hidden.dtype,
                device=hidden.device,
            )
        if self.num_filled >= self.num_slots:
            raise RuntimeError(
                f"ChunkMemoryBank.append: bank full ({self.num_filled}/{self.num_slots})"
            )
        self.slots[:, self.num_filled, :] = hidden.detach()
        self.num_filled += 1

    def get_all(self) -> torch.Tensor:
        """Return all filled slots, detached.  [B, n_filled, d]."""
        if self.slots is None or self.num_filled == 0:
            raise RuntimeError("ChunkMemoryBank.get_all: no slots filled")
        return self.slots[:, : self.num_filled, :].detach()

    @property
    def is_full(self) -> bool:
        return self.num_filled >= self.num_slots

    # ------------------------------------------------------------------ #
    # Phase 2 (future): top-k selection + EMA update
    # ------------------------------------------------------------------ #

    def top_k(self, query: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Cosine-similarity top-k selection.

        Args:
            query: [B, d] -- chunk mean-pooled hidden state.
            k: number of slots to select.

        Returns:
            (selected_slots [B, k, d], selected_idx [B, k]) -- both detached.
        """
        slots = self.slots.detach()  # [B, N, d]
        q_norm = F.normalize(query, dim=-1).unsqueeze(1)  # [B, 1, d]
        s_norm = F.normalize(slots, dim=-1)  # [B, N, d]
        scores = torch.bmm(q_norm, s_norm.transpose(1, 2)).squeeze(1)  # [B, N]
        _, idx = scores.topk(k, dim=-1)  # [B, k]
        selected = slots.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.d_model))
        return selected, idx

    def update_selected(
        self,
        idx: torch.Tensor,
        new_hidden: torch.Tensor,
    ) -> None:
        """EMA update the selected slots.  ``idx``: [B, k], ``new_hidden``: [B, d]."""
        with torch.no_grad():
            B, k = idx.shape
            new_h = new_hidden.unsqueeze(1).expand(-1, k, -1)  # [B, k, d]
            current = self.slots.gather(
                1, idx.unsqueeze(-1).expand(-1, -1, self.d_model)
            )
            updated = self.ema_decay * current + (1 - self.ema_decay) * new_h
            self.slots.scatter_(
                1, idx.unsqueeze(-1).expand(-1, -1, self.d_model), updated
            )

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset the bank for a new document."""
        self.slots = None
        self.num_filled = 0
