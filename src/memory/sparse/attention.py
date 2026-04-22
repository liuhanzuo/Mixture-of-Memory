"""
ConcatFusionAttention — Sliding window local path + sparse memory path.

Fusion: concat([o_local, o_memory]) projected through o_proj,
with a learned bypass gate that can gradually let memory in.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusionAttention(nn.Module):
    """Replaces standard self-attention with local + memory dual paths.

    Uses KV concatenation fusion instead of gated mixing.
    Local heads and memory heads are concatenated along the head dimension,
    then projected back via o_proj. A learned bypass gate (init ≈ 0.12)
    controls how much memory contributes.

    Args:
        original_attn: The model's original attention module (for local path).
        layer_idx: Which transformer layer this belongs to.
        memory_bank: Shared SparseMemoryBank reference.
        window_size: Sliding window size (default 256).
        top_k: Number of memory slots to retrieve (default 8).
        head_dim: Per-head dimension.
        bypass_bias_init: Initial bypass gate bias (default -2.0 → σ(-2)≈0.12).
    """

    def __init__(
        self,
        original_attn: nn.Module,
        layer_idx: int,
        memory_bank: "SparseMemoryBank",  # forward ref
        window_size: int = 256,
        top_k: int = 8,
        head_dim: int = 128,
        bypass_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.original_attn = original_attn
        self.layer_idx = layer_idx
        self.memory_bank = memory_bank
        self.window_size = window_size
        self.top_k = top_k
        self.head_dim = head_dim

        # Bypass gate: hidden_dim → 1, bias=-2.0 so σ(-2)≈0.12
        # Strong local bias at init; gate can gradually increase to let memory in
        num_heads = getattr(original_attn, 'num_heads', original_attn.config.num_attention_heads)
        gate_in_dim = head_dim * num_heads
        self.bypass_gate_proj = nn.Linear(gate_in_dim, 1)
        nn.init.kaiming_normal_(self.bypass_gate_proj.weight, nonlinearity='linear')
        nn.init.constant_(self.bypass_gate_proj.bias, bypass_bias_init)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[Tuple]]:
        """Forward pass with gated two-path attention.

        1. Local path: standard attention (the original module handles this).
        2. Memory path: top-k retrieval from memory bank.
        3. Gated fusion.
        """
        # ── Local path via original attention ──────────────────────────
        # Build a causal sliding window mask if needed
        T = hidden_states.shape[1]
        if attention_mask is None and T > self.window_size:
            attention_mask = self._build_window_mask(T, hidden_states.device, hidden_states.dtype)

        local_out = self.original_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        # Unpack: most HF attn returns (attn_output, ..., present, attn_weights)
        o_local = local_out[0]  # [B, T, D]
        extras = local_out[1:]

        # ── Memory path: top-k retrieval ───────────────────────────────
        attn = self.original_attn
        B, T_seq, D = hidden_states.shape

        wk = getattr(attn, "k_proj", None)
        wv = getattr(attn, "v_proj", None)
        wo = getattr(attn, "o_proj", None)

        # We need to handle memory per layer; memory_bank operates without batch dim
        # Process batch items sequentially (training typically B=1 for long-context)
        memory_outputs = []
        for b in range(B):
            mem_out = self.memory_bank.read(
                layer_idx=self.layer_idx,
                queries=hidden_states[b],  # [T, D]
                k=self.top_k,
                wk=wk,
                wv=wv,
                wo=wo,
            )  # [T, D]
            memory_outputs.append(mem_out)

        # Stack: [B, T, D]
        o_mem = torch.stack(memory_outputs, dim=0)  # [B, T, D]

        # ── Concatenation fusion with bypass gate ──────────────────────
        # Bypass gate: how much of memory output to blend in
        # gate ≈ 0.12 at init (strong local bias), can grow during training
        bypass = torch.sigmoid(self.bypass_gate_proj(o_local.float())).to(o_local.dtype)  # [B, T, 1]

        # Weighted memory contribution + local
        o = o_local + bypass * o_mem

        # ── Memory write (online EMA update) ───────────────────────────
        # Write each batch item's hidden states to memory
        for b in range(B):
            self.memory_bank.write(self.layer_idx, hidden_states[b])  # [T, D]

        return (o, *extras)

    def get_bypass_gate_value(self) -> float:
        """Return mean bypass gate value (for monitoring)."""
        with torch.no_grad():
            w = self.bypass_gate_proj.weight
            b = self.bypass_gate_proj.bias
            return torch.sigmoid(b).item()

    def _build_window_mask(
        self, T: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build a causal sliding window attention mask.

        Returns float mask: 0 for allowed, -inf for blocked.
        Shape: [1, 1, T, T] (broadcastable to [B, H, T, T]).
        """
        mask = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
        w = self.window_size
        for i in range(T):
            lo = max(0, i - w + 1)
            mask[i, lo:i + 1] = 0.0
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
