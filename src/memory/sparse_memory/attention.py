"""SparseMemoryAttention — Local sliding-window + per-chunk memory retrieval + gated fusion.

Wraps LlamaAttention, replacing standard full-sequence attention with:
  1. Local path: flash_attn sliding window (size w)
  2. Memory path: 
     a) Per-chunk retrieval: chunk query (mean pool) selects shared top-K memory slots
     b) Per-token read: each token attends to shared K slots with its own query
  3. Gated fusion: o = g · o_local + (1-g) · o_mem, where g = σ(W_g · h + b_g)
  4. Memory write: per-chunk retrieve-then-update — chunk summary EMA-updates shared slots
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SparseMemoryConfig:
    """Configuration for sparse memory parameters."""
    num_mem_tokens: int = 128
    window_size: int = 256
    top_k: int = 8

try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False

# PyTorch native SDPA (available in PyTorch 2.0+, uses flash/memory-efficient kernels)
HAS_SDPA = hasattr(torch.nn.functional, 'scaled_dot_product_attention')


class SparseMemoryAttention(nn.Module):
    """Dual-path attention replacing a single LlamaAttention layer.

    Args:
        original_attn: The LlamaAttention module being wrapped.
        memory_bank: Per-layer MemoryBank instance.
        window_size: Sliding window size w (default 256).
        top_k: Top-K shared memory slots per chunk (default 8).
    """

    def __init__(
        self,
        original_attn: nn.Module,
        memory_bank: "MemoryBank",
        window_size: int = 256,
        top_k: int = 8,
    ) -> None:
        super().__init__()
        self.original_attn = original_attn
        # Store memory_bank as a plain (non-submodule) attribute to avoid
        # duplicating it in the state_dict.  It is already registered via
        # self.memory_banks (nn.ModuleList) in the top-level model.
        # nn.Module.__setattr__ would register it as a submodule, causing
        # shared-tensor errors when saving checkpoints.
        object.__setattr__(self, '_memory_bank', memory_bank)
        self.window_size = window_size
        self.top_k = top_k

        # Expose config attrs expected by the Llama layer
        self.config = original_attn.config
        # In transformers >=5.x, LlamaAttention doesn't have num_heads as attribute
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = getattr(self.config, "num_key_value_heads", self.num_heads)
        self.head_dim = getattr(original_attn, 'head_dim', self.config.hidden_size // self.num_heads)
        self.hidden_size = self.config.hidden_size
        # Attributes required by transformers 5.x LlamaDecoderLayer / modeling_layers
        self.layer_idx = getattr(original_attn, 'layer_idx', 0)
        self.is_causal = True
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = getattr(self.config, 'attention_dropout', 0.0)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # In transformers >=5.x, rotary_emb is NOT on the attention layer.
        # RoPE is applied via position_embeddings=(cos,sin) passed to forward.
        # No need to steal rotary_emb.

        # Gated two-path fusion: o = g * o_local + (1-g) * o_mem
        # Gate is computed from hidden_states (not from concat)
        fusion_dtype = original_attn.q_proj.weight.dtype
        self.gate_proj = nn.Linear(self.hidden_size, 1, dtype=fusion_dtype)
        with torch.no_grad():
            nn.init.zeros_(self.gate_proj.weight)
            self.gate_proj.bias.fill_(2.0)  # σ(2)≈0.88 → 88% local, 12% memory initially

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = hidden_states.shape
        d_h = self.head_dim

        # ── Q/K/V projections (shared with original attention) ────────
        q = self.original_attn.q_proj(hidden_states)  # [B, T, num_heads * d_h]
        k = self.original_attn.k_proj(hidden_states)  # [B, T, num_kv_heads * d_h]
        v = self.original_attn.v_proj(hidden_states)  # [B, T, num_kv_heads * d_h]

        # Reshape: [B, T, H, d_h] → [B, H, T, d_h]
        q = q.view(B, T, self.num_heads, d_h).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, d_h).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, d_h).transpose(1, 2)

        # ── Apply RoPE via position_embeddings (cos, sin) from LlamaDecoderLayer ─
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ── GQA: expand KV heads to match Q heads for local attention ──
        k_local = k
        v_local = v
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k_local = k.unsqueeze(2).expand(B, self.num_kv_heads, n_rep, T, d_h).reshape(B, self.num_heads, T, d_h)
            v_local = v.unsqueeze(2).expand(B, self.num_kv_heads, n_rep, T, d_h).reshape(B, self.num_heads, T, d_h)

        # ── Local Path: sliding window attention ─────────────────────
        if HAS_FLASH:
            o_local = flash_attn_func(
                q, k_local, v_local,
                window_size=(self.window_size, self.window_size),
                causal=True,
            )
        elif HAS_SDPA:
            # PyTorch native SDPA with explicit causal + sliding window mask
            # NOTE: SDPA is_causal=True does NOT support windowing — we must
            # provide the full mask ourselves.
            w = self.window_size
            row_idx = torch.arange(T, device=q.device).unsqueeze(1)  # [T, 1]
            col_idx = torch.arange(T, device=q.device).unsqueeze(0)  # [1, T]
            # Mask out: future tokens (j > i) AND tokens outside window (j < i - w + 1)
            block_mask = (col_idx > row_idx) | (col_idx < row_idx - w + 1)  # [T, T]
            attn_mask = torch.zeros(T, T, device=q.device, dtype=q.dtype)
            attn_mask.masked_fill_(block_mask, float("-inf"))
            o_local = torch.nn.functional.scaled_dot_product_attention(
                q, k_local, v_local,
                is_causal=False,
                attn_mask=attn_mask,
            )
        else:
            o_local = self._windowed_attention_fallback(q, k_local, v_local, T)

        # [B, H, T, d_h] → [B, T, D]
        o_local = o_local.transpose(1, 2).contiguous().view(B, T, D)
        o_local = self.original_attn.o_proj(o_local)

        # ── Memory Path: per-chunk retrieval + per-token read ──────
        # Memory buffer: [B, N, D], detached and cast to model dtype
        mem = self._memory_bank.memory.detach().clone()  # [B, N, D]
        proj_dtype = self.original_attn.k_proj.weight.dtype
        if mem.dtype != proj_dtype:
            mem = mem.to(proj_dtype)

        # Project memory through W_K, W_V (shared with local attention)
        k_mem = self.original_attn.k_proj(mem)  # [B, N, KV_H*d_h]
        v_mem = self.original_attn.v_proj(mem)  # [B, N, KV_H*d_h]

        # Reshape: [B, N, KV_H, d_h] → [B, KV_H, N, d_h]
        k_mem = k_mem.view(B, self._memory_bank.num_slots, self.num_kv_heads, d_h).permute(0, 2, 1, 3)
        v_mem = v_mem.view(B, self._memory_bank.num_slots, self.num_kv_heads, d_h).permute(0, 2, 1, 3)

        # GQA expand
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k_mem = k_mem.unsqueeze(2).expand(B, self.num_kv_heads, n_rep, -1, d_h).reshape(B, self.num_heads, -1, d_h)
            v_mem = v_mem.unsqueeze(2).expand(B, self.num_kv_heads, n_rep, -1, d_h).reshape(B, self.num_heads, -1, d_h)

        # Now: q [B, H, T, d_h], k_mem [B, H, N, d_h], v_mem [B, H, N, d_h]
        N = k_mem.shape[2]
        effective_top_k = min(self.top_k, N)

        # ── Step 1: Chunk-level retrieval — shared top-K slot indices ──
        # Pool queries across sequence to get chunk representation
        q_chunk = q.mean(dim=2).unsqueeze(2)  # [B, H, 1, d_h] — mean pool over T
        # unsqueeze(2) needed: matmul([B,H,1,d_h], [B,H,d_h,N]) = [B,H,1,N]
        # Without it, matmul([B,H,d_h], [B,H,d_h,N]) misaligns batch dims

        # Similarity: chunk query vs all memory slots
        sim_chunk = (torch.matmul(q_chunk, k_mem.transpose(-2, -1)) / (d_h ** 0.5)).squeeze(2)  # [B, H, N]
        _, shared_idx = sim_chunk.topk(effective_top_k, dim=-1)  # [B, H, K]

        # Gather K_mem and V_mem for shared slots
        shared_idx_clamped = shared_idx.clamp(0, N - 1)
        B_idx = torch.arange(B, device=v_mem.device)[:, None, None]  # [B,1,1]
        H_idx = torch.arange(self.num_heads, device=v_mem.device)[None, :, None]  # [1,H,1]
        k_shared = k_mem[B_idx, H_idx, shared_idx_clamped]  # [B, H, K, d_h]
        v_shared = v_mem[B_idx, H_idx, shared_idx_clamped]  # [B, H, K, d_h]

        # ── Step 2: Per-token read on shared K slots ─────────────────
        # Each token uses its own Q to attend to the shared K slots
        sim_per_token = torch.matmul(q, k_shared.transpose(-2, -1)) / (d_h ** 0.5)  # [B, H, T, K]
        attn_weights = F.softmax(sim_per_token, dim=-1)  # [B, H, T, K]
        o_mem = torch.matmul(attn_weights, v_shared)  # [B, H, T, d_h]

        # Save shared_idx for memory write
        shared_idx_for_write = shared_idx_clamped  # [B, H, K]

        # [B, H, T, d_h] → [B, T, D]
        o_mem = o_mem.transpose(1, 2).contiguous().view(B, T, D)
        # NOTE: no o_proj — memory path already has correct scale from v_proj + attention

        # ── Gated Two-Path Fusion ───────────────────────────────────────
        # g = σ(W_g · h + b_g),  o = g · o_local + (1-g) · o_mem
        # Initial bias=+2.0 → σ(2)≈0.88 → 88% local, 12% memory
        g = torch.sigmoid(self.gate_proj(hidden_states))  # [B, T, 1]
        output = g * o_local + (1.0 - g) * o_mem

        # ── Memory Write: importance-based top-K selective writing ─────
        # Aggregate shared indices across heads
        avg_shared_idx = shared_idx_for_write.float().mean(dim=1).long()  # [B, K]
        avg_shared_idx = avg_shared_idx.clamp(0, self._memory_bank.num_slots - 1)

        # Per-token attention weights for importance scoring
        # attn_weights: [B, H, T, K] → mean over heads → [B, T, K]
        per_token_attn = attn_weights.mean(dim=1)  # [B, T, K]

        for b in range(B):
            self._memory_bank.update_slots(
                hidden_states[b],               # [T, D]
                batch_idx=b,
                slot_indices=avg_shared_idx[b].unsqueeze(0),  # [1, K]
                attention_weights=per_token_attn[b],  # [T, K]
            )
        del shared_idx_for_write

        # LlamaDecoderLayer does `hidden_states, _ = self.self_attn(...)` → expects 2-tuple
        return (output, None)

    def _windowed_attention_fallback(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """Fallback windowed attention when flash_attn is unavailable."""
        B, H, _, d_h = q.shape
        w = self.window_size

        # Build causal window mask: [T, T]
        mask = torch.full((T, T), float("-inf"), device=q.device, dtype=q.dtype)
        for i in range(T):
            lo = max(0, i - w + 1)
            mask[i, lo:i + 1] = 0.0

        scale = d_h ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return out




# ── RoPE helpers (matching HF transformers implementation) ─────────────────

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (HF's standard implementation).

    Args:
        x: [..., d] where d is even.
    Returns:
        [..., d] with halves rotated: cat(-x2, x1)
    """
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to q and k (HF-compatible).

    Args:
        q, k: [B, H, T, d_h]
        cos, sin: [B, 1, T, d_h] (from LlamaRotaryEmbedding)
    Returns:
        (q_embed, k_embed) with rotation applied.
    """
    # Ensure cos/sin have head dim for broadcasting: [B, 1, T, d_h]
    if cos.ndim == 3:
        cos = cos.unsqueeze(1)
    if sin.ndim == 3:
        sin = sin.unsqueeze(1)
    # HF formula: q_embed = q * cos + rotate_half(q) * sin
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed
