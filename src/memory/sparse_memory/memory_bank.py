"""MemoryBank — EMA memory buffer for sparse memory retrieval.

Per-layer memory buffer (batch_size × N slots × d hidden_dim).
Read: caller detaches memory and applies W_K/W_V projections externally.
Write: update_slots() — only update slots referenced by top-K retrieval indices.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MemoryBank(nn.Module):
    """Fixed-size memory bank with EMA write, batch-aware.

    Args:
        num_slots: Number of memory slots N.
        hidden_dim: Hidden dimension d (full model dim, e.g. 4096).
        ema_alpha: EMA decay rate for writes (default 0.1).
    """

    def __init__(
        self,
        num_slots: int = 128,
        hidden_dim: int = 4096,
        ema_alpha: float = 0.1,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.ema_alpha = ema_alpha
        self._dtype = dtype

        # Memory buffer: [1, N, d] placeholder; expanded to [B, N, d] on reset()
        # Registered as parameter with requires_grad=False so DDP broadcasts it
        # across GPUs (buffers with persistent=False are excluded from DDP sync).
        # The optimizer ignores requires_grad=False parameters.
        self.register_parameter(
            "memory",
            nn.Parameter(torch.zeros(1, num_slots, hidden_dim, dtype=dtype), requires_grad=False),
        )


    def reset(self, batch_size: int = 1) -> None:
        """Zero out memory and reset write pointer for given batch size."""
        device = self.memory.device
        self.memory = nn.Parameter(torch.zeros(batch_size, self.num_slots, self.hidden_dim, device=device, dtype=self._dtype), requires_grad=False)


    @torch.no_grad()
    def update_slots(
        self,
        hidden_states: torch.Tensor,  # [T, D]
        batch_idx: int,
        slot_indices: torch.Tensor,    # [T, K]
    ) -> None:
        """Update only the slots that were retrieved (top-K per token)."""
        alpha = self.ema_alpha
        new_mem = self.memory.clone()

        unique_indices = torch.unique(slot_indices.flatten())

        for idx in unique_indices:
            mask = (slot_indices == idx).any(dim=-1)  # [T]
            if mask.sum() == 0:
                continue
            relevant_hiddens = hidden_states[mask]  # [n, D]
            aggregated = relevant_hiddens.mean(dim=0)  # [D]
            current = self.memory[batch_idx, idx]  # [D]
            updated = (alpha * aggregated + (1.0 - alpha) * current).to(self._dtype)
            new_mem[batch_idx, idx] = updated

        self.memory = nn.Parameter(new_mem, requires_grad=False)
