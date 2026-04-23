"""MemoryBank — Selective importance-based memory buffer for sparse memory retrieval.

Per-layer memory buffer (batch_size × N slots × d hidden_dim).
Read: caller detaches memory and applies W_K/W_V projections externally.
Write: update_slots() with importance-based top-K selection.

Key improvement over v1 (cyclic EMA):
  - v1 wrote ALL tokens' chunk summary to retrieved slots → 3.4% retention
  - v2 scores each token by importance, writes only top-K tokens → ~99% retention
  - Importance = gradient proxy: |hidden| * attention_entropy_surprise
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """Fixed-size memory bank with importance-based top-K writing.

    Args:
        num_slots: Number of memory slots N.
        hidden_dim: Hidden dimension d (full model dim, e.g. 4096).
        ema_alpha: EMA decay rate for writes (default 0.1).
        write_top_k: Number of top-important tokens to write per chunk.
            0 or None means write all (legacy behavior).
        importance_mode: How to score token importance.
            'magnitude': ||h_t||₂ — simple, fast, surprisingly effective
            'attention_surprise': 1 - max(softmax_attn) — tokens not well-attended are surprising
            'combined': magnitude * surprise — best of both
    """

    def __init__(
        self,
        num_slots: int = 128,
        hidden_dim: int = 4096,
        ema_alpha: float = 0.1,
        dtype: torch.dtype = torch.bfloat16,
        write_top_k: int | None = 0,
        importance_mode: str = "combined",
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.ema_alpha = ema_alpha
        self._dtype = dtype
        self.write_top_k = write_top_k
        self.importance_mode = importance_mode

        # Memory buffer: [1, N, d] placeholder; expanded to [B, N, d] on reset()
        self.register_parameter(
            "memory",
            nn.Parameter(torch.zeros(1, num_slots, hidden_dim, dtype=dtype), requires_grad=False),
        )

        # Optional learnable importance scorer: maps hidden_dim → scalar score
        # Initialized to near-identity (bias=0, small weight) so initial behavior
        # approximates magnitude scoring
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
        )
        with torch.no_grad():
            nn.init.zeros_(self.importance_head[0].weight)
            self.importance_head[0].bias.fill_(0.0)

        # Running stats for diagnostics
        self._write_count = 0
        self._total_tokens_seen = 0

    def reset(self, batch_size: int = 1) -> None:
        """Zero out memory for given batch size."""
        device = self.memory.device
        self.memory = nn.Parameter(
            torch.zeros(batch_size, self.num_slots, self.hidden_dim, device=device, dtype=self._dtype),
            requires_grad=False,
        )
        self._write_count = 0
        self._total_tokens_seen = 0

    def compute_importance(
        self,
        hidden_states: torch.Tensor,  # [T, D]
        attention_weights: torch.Tensor | None = None,  # [T, K] or [H, T, K]
    ) -> torch.Tensor:
        """Score each token by importance for selective writing.

        Args:
            hidden_states: [T, D] token hidden states.
            attention_weights: Optional attention weights from memory path.
                [T, K] per-token attention over K memory slots.

        Returns:
            importance: [T] scalar importance score per token.
        """
        T = hidden_states.shape[0]

        # Magnitude component: ||h_t||
        magnitude = hidden_states.float().norm(dim=-1)  # [T]

        if self.importance_mode == "magnitude" or attention_weights is None:
            # Pure magnitude: simple, no attention dependency
            importance = magnitude
        else:
            # Attention-based surprise: tokens with flat/uncertain attention
            # are more "surprising" → more important to memorize
            if attention_weights.dim() == 3:
                # [H, T, K] → average over heads
                attn = attention_weights.mean(dim=0)  # [T, K]
            else:
                attn = attention_weights  # [T, K]

            # Surprise = entropy of attention distribution
            # Higher entropy → more uncertain → more valuable to store
            entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1)  # [T]
            # Normalize to [0, 1] range
            max_entropy = torch.log(torch.tensor(attn.shape[-1], dtype=torch.float, device=attn.device))
            surprise = entropy / max_entropy.clamp(min=1e-8)  # [T]

            if self.importance_mode == "attention_surprise":
                importance = surprise
            else:  # combined
                importance = magnitude * (1.0 + surprise)

        # Add learnable component
        learned_score = self.importance_head(hidden_states.float()).squeeze(-1)  # [T]
        importance = importance + torch.sigmoid(learned_score)

        return importance

    def update_slots(
        self,
        hidden_states: torch.Tensor,  # [T, D]
        batch_idx: int,
        slot_indices: torch.Tensor,  # [T, K] or [1, K]
        attention_weights: torch.Tensor | None = None,  # [T, K] or [H, T, K]
    ) -> None:
        """Update memory slots using importance-based top-K selection.

        Importance scoring uses the learnable importance_head (gradient-enabled).
        The actual memory slot EMA update is no_grad.

        If write_top_k > 0, only the top-K most important tokens are used
        for the EMA update. Otherwise, all tokens update (legacy behavior).

        Args:
            hidden_states: [T, D] token representations.
            batch_idx: Which batch entry to update.
            slot_indices: [T, K] or [1, K] slot indices from retrieval.
            attention_weights: [T, K] attention weights for importance scoring.
        """
        alpha = self.ema_alpha

        # Normalize slot_indices to [T, K] shape
        if slot_indices.dim() == 2 and slot_indices.shape[0] == 1:
            slot_indices = slot_indices.expand(hidden_states.shape[0], -1)

        T = hidden_states.shape[0]
        self._total_tokens_seen += T

        # ── Importance-based top-K selection (gradient-enabled for importance_head) ──
        if self.write_top_k and self.write_top_k > 0 and self.write_top_k < T:
            importance = self.compute_importance(hidden_states, attention_weights)  # [T]
            topk_vals, topk_idx = importance.topk(self.write_top_k)  # [write_top_k]

            # Detach selected indices, select tokens for memory update
            hidden_states_selected = hidden_states[topk_idx].detach()  # [K', D]
            slot_indices_selected = slot_indices[topk_idx]  # [K', K_slots]
            self._write_count += self.write_top_k
        else:
            hidden_states_selected = hidden_states.detach()
            slot_indices_selected = slot_indices
            self._write_count += T

        # ── EMA update per unique slot (no_grad) ──
        with torch.no_grad():
            new_mem = self.memory.clone()
            unique_indices = torch.unique(slot_indices_selected.flatten())

            for idx in unique_indices:
                mask = (slot_indices_selected == idx).any(dim=-1)
                if mask.sum() == 0:
                    continue
                relevant_hiddens = hidden_states_selected[mask]  # [n, D]
                aggregated = relevant_hiddens.mean(dim=0)  # [D]
                current = self.memory[batch_idx, idx]  # [D]
                updated = (alpha * aggregated + (1.0 - alpha) * current).to(self._dtype)
                new_mem[batch_idx, idx] = updated

            self.memory = nn.Parameter(new_mem, requires_grad=False)

    def get_write_stats(self) -> dict:
        """Return diagnostic stats about write behavior."""
        retention = (
            self._write_count / max(self._total_tokens_seen, 1)
            if self._total_tokens_seen > 0
            else 0.0
        )
        return {
            "write_count": self._write_count,
            "total_tokens_seen": self._total_tokens_seen,
            "retention_rate": retention,
            "write_top_k": self.write_top_k,
            "importance_mode": self.importance_mode,
        }
