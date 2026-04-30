"""
SparseMemoryBank — EMA-based memory with FIFO write + top-k read.

Shape: [num_layers, N, head_dim] (N slots per layer).
Write: gated EMA update per token.
Read: dot-product similarity → top-k retrieval → weighted sum.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMemoryBank(nn.Module):
    """Fixed-size memory bank with EMA write and top-k read.

    Args:
        num_layers: Number of transformer layers.
        num_slots: Memory capacity N (default 128).
        hidden_dim: Full hidden dimension d.
        head_dim: Per-head dimension (default hidden_dim).
        ema_alpha: EMA decay rate (default 0.1).
        gate_bias_init: Initial bias for write gate (default 4.0 so σ(4)≈0.98,
            making the gate ~pass-through. Since the gate is frozen (see below),
            a high bias prevents the random Kaiming-init weight from halving
            every EMA update. The old 0.0 default (σ(0)=0.5) caused writes to
            be weighted by random noise — Bug #3, fixed 2026-04-25.
        write_top_k: Number of top-K most-important tokens to write per forward
            call (default 8, matches read top_k). The old 0 default wrote ALL
            T tokens, which with T=4096 and num_slots=128 wrapped the buffer
            ~32× per chunk and wiped all prior memory — Bug #1, fixed 2026-04-25.
    """

    def __init__(
        self,
        num_layers: int = 32,
        num_slots: int = 128,
        hidden_dim: int = 4096,
        head_dim: Optional[int] = None,
        ema_alpha: float = 0.1,
        gate_bias_init: float = 4.0,
        write_top_k: int = 8,
        importance_mode: str = "combined",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim or hidden_dim
        self.ema_alpha = ema_alpha
        self.write_top_k = write_top_k
        self.importance_mode = importance_mode

        # Memory tensor: [num_layers, N, hidden_dim]
        self.memory = nn.Parameter(
            torch.zeros(num_layers, num_slots, hidden_dim),
            requires_grad=False,  # Updated via EMA, not optimizer
        )
        nn.init.normal_(self.memory, std=0.02)

        # Per-layer write gate: linear(hidden_dim, 1) with bias
        # NOTE: These are used only in EMA in-place writes (self.memory.data[...] = ...),
        # so gradients never flow back to them. Mark as requires_grad=False to avoid
        # DDP "unused parameters" deadlocks when ddp_find_unused_parameters=False.
        self.write_gate = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_layers)
        ])
        for gate in self.write_gate:
            nn.init.kaiming_normal_(gate.weight, nonlinearity='linear')
            nn.init.constant_(gate.bias, gate_bias_init)
            gate.weight.requires_grad_(False)
            gate.bias.requires_grad_(False)

        # Write pointer (circular buffer) — not a parameter
        self.register_buffer("_write_ptr", torch.zeros(num_layers, dtype=torch.long))

        # Write-frequency instrumentation (Bug-audit 2026-04-25).
        # _write_tokens[l] counts how many tokens have been EMA-written into
        # layer l's bank across the lifetime of this module. Useful for
        # verifying write_top_k is actually in effect.
        # _write_calls[l] counts how many .write() invocations happened.
        self.register_buffer("_write_tokens", torch.zeros(num_layers, dtype=torch.long))
        self.register_buffer("_write_calls", torch.zeros(num_layers, dtype=torch.long))

    @property
    def write_ptr(self) -> torch.Tensor:
        return self._write_ptr

    # ── Write ──────────────────────────────────────────────────────────
    def write(
        self,
        layer_idx: int,
        values: torch.Tensor,  # [T, hidden_dim]
    ) -> None:
        """EMA-update memory slots for one layer. Handles arbitrary T > 0."""
        T = values.shape[0]
        ptr = self._write_ptr[layer_idx].item()

        # Compute write gate: sigmoid(W @ h + b) → [T, 1]
        g = torch.sigmoid(self.write_gate[layer_idx](values.float())).to(values.dtype)  # [T, 1]

        # Selective writing: only write top-K important tokens
        if self.write_top_k and self.write_top_k > 0 and self.write_top_k < T:
            gate_importance = g.abs().squeeze(-1)  # [T]
            value_importance = values.float().norm(dim=-1)  # [T]
            if self.importance_mode == "magnitude":
                importance = value_importance
            elif self.importance_mode == "attention_surprise":
                importance = gate_importance
            else:  # combined
                importance = gate_importance * value_importance
            topk_vals, topk_idx = importance.topk(self.write_top_k)  # [write_top_k]
            values = values[topk_idx]
            g = g[topk_idx]
            T = self.write_top_k

        # Gather current memory at target slots
        indices = torch.arange(T, device=values.device)
        slot_ids = (ptr + indices) % self.num_slots  # [T]
        current = self.memory.data[layer_idx, slot_ids]  # [T, hidden_dim]

        # EMA update
        alpha = self.ema_alpha
        updated = alpha * g * values + (1.0 - alpha) * current
        self.memory.data[layer_idx, slot_ids] = updated

        # Advance pointer (no gradient)
        self._write_ptr[layer_idx].fill_((ptr + T) % self.num_slots)

        # Instrumentation: track write frequency (Bug-audit 2026-04-25)
        self._write_tokens[layer_idx] += T
        self._write_calls[layer_idx] += 1

    # ── Read ──────────────────────────────────────────────────────────
    def read(
        self,
        layer_idx: int,
        queries: torch.Tensor,  # [T, hidden_dim] — full hidden states
        k: int = 8,
        wk: Optional[nn.Linear] = None,
        wv: Optional[nn.Linear] = None,
        wo: Optional[nn.Linear] = None,
    ) -> torch.Tensor:
        """Top-k retrieval: query against memory keys → weighted sum of values.

        Returns: [T, hidden_dim]
        """
        mem = self.memory[layer_idx]  # [N, hidden_dim]
        dtype = queries.dtype
        mem = mem.to(dtype)

        # For GQA models, wk/wv map to smaller dims causing mismatches.
        # Use raw hidden_dim memory directly to avoid dimension issues.
        # Future: add dedicated projection layers for memory keys/values.
        k_mem = mem  # [N, hidden_dim]
        v_mem = mem  # [N, hidden_dim]
        q_proj = queries  # [T, hidden_dim]

        # Dot-product similarity: [T, N]
        scores = torch.matmul(q_proj, k_mem.transpose(-2, -1)) / (k_mem.shape[-1] ** 0.5)

        # Top-k selection
        topk_scores, topk_idx = scores.topk(k, dim=-1)  # [T, k]

        dim = v_mem.shape[-1]

        # Gather corresponding values and reweight with softmax
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, dim)
        topk_v = v_mem.unsqueeze(0).expand(queries.shape[0], -1, -1).gather(1, idx_exp)  # [T, k, dim]

        # Softmax over top-k
        attn_weights = F.softmax(topk_scores, dim=-1).unsqueeze(-1)  # [T, k, 1]

        # Weighted sum
        out = (attn_weights * topk_v).sum(dim=1)  # [T, hidden_dim]
        return out

    def reset(self) -> None:
        """Reset memory and write pointers."""
        nn.init.normal_(self.memory, std=0.02)
        self._write_ptr.zero_()
        self._write_tokens.zero_()
        self._write_calls.zero_()

    def write_stats(self) -> dict:
        """Return per-layer write-frequency stats for instrumentation.

        Returns dict with keys:
            tokens_per_layer: LongTensor[num_layers]
            calls_per_layer:  LongTensor[num_layers]
            tokens_total:     int
            calls_total:      int
            tokens_per_call:  float (tokens_total / max(calls_total, 1))
        """
        total_tokens = int(self._write_tokens.sum().item())
        total_calls = int(self._write_calls.sum().item())
        return {
            "tokens_per_layer": self._write_tokens.detach().clone(),
            "calls_per_layer": self._write_calls.detach().clone(),
            "tokens_total": total_tokens,
            "calls_total": total_calls,
            "tokens_per_call": total_tokens / max(total_calls, 1),
        }
