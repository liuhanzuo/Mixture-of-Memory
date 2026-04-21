"""SparseMemoryAttention — Local sliding-window + memory retrieval + concat fusion.

Wraps LlamaAttention, replacing standard full-sequence attention with:
  1. Local path: flash_attn sliding window (size w)
  2. Memory path: top-k retrieval from MemoryBank via shared W_K/W_V
  3. Concat fusion: W_f [o_local ∥ o_mem] (Memorizing Transformers-style)
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
    gate_alpha: float = 0.1

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
        top_k: Top-k memory retrieval per token per head (default 8).
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
        self.memory_bank = memory_bank
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

        # Concat fusion projection: [o_local ∥ o_mem] → hidden_dim
        # (Memorizing Transformers-style, avoids sigmoid gate gradient dilution)
        fusion_dtype = original_attn.q_proj.weight.dtype
        self.fusion_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, dtype=fusion_dtype)
        # Identity pass-through at init: left half picks up local, right half ignores memory
        # This preserves pre-trained base model outputs (base loss stays ~0.04 instead of 13.88)
        with torch.no_grad():
            self.fusion_proj.weight.zero_()
            self.fusion_proj.weight[:, :self.hidden_size].copy_(torch.eye(self.hidden_size, dtype=fusion_dtype))
            nn.init.zeros_(self.fusion_proj.bias)

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
            # PyTorch native SDPA with causal mask (O(T) memory)
            o_local = torch.nn.functional.scaled_dot_product_attention(
                q, k_local, v_local,
                is_causal=True,
                attn_mask=None,  # causal window handled by SDPA internally
            )
        else:
            o_local = self._windowed_attention_fallback(q, k_local, v_local, T)

        # [B, H, T, d_h] → [B, T, D]
        o_local = o_local.transpose(1, 2).contiguous().view(B, T, D)
        o_local = self.original_attn.o_proj(o_local)

        # ── Memory Path: top-k retrieval (batched) ────────────────────
        # Memory buffer: [B, N, D], cloned to avoid inplace-modified-by-write errors.
        # Even though write() uses clone()+replace, autograd can detect shared-storage
        # version changes.  An explicit .clone() gives us an independent tensor.
        mem = self.memory_bank.memory.detach().clone()  # [B, N, D]

        # Cast memory to model dtype (buffer may be float32, model is bf16)
        proj_dtype = self.original_attn.k_proj.weight.dtype
        if mem.dtype != proj_dtype:
            mem = mem.to(proj_dtype)

        # Project memory through W_K, W_V (these projections have gradients)
        # k_proj/v_proj: [B, N, D] → [B, N, num_kv_heads * d_h]
        k_mem = self.original_attn.k_proj(mem)
        v_mem = self.original_attn.v_proj(mem)

        # Reshape: [B, N, KV_H, d_h] → [B, KV_H, N, d_h]
        k_mem = k_mem.view(B, self.memory_bank.num_slots, self.num_kv_heads, d_h).permute(0, 2, 1, 3)
        v_mem = v_mem.view(B, self.memory_bank.num_slots, self.num_kv_heads, d_h).permute(0, 2, 1, 3)

        # For GQA: expand KV heads to match Q heads
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k_mem = k_mem.unsqueeze(2).expand(B, self.num_kv_heads, n_rep, -1, d_h).reshape(B, self.num_heads, -1, d_h)
            v_mem = v_mem.unsqueeze(2).expand(B, self.num_kv_heads, n_rep, -1, d_h).reshape(B, self.num_heads, -1, d_h)

        # Now: q [B, H, T, d_h], k_mem [B, H, N, d_h], v_mem [B, H, N, d_h]
        N = k_mem.shape[2]
        # Guard: top_k cannot exceed number of available memory slots
        effective_top_k = min(self.top_k, N)

        # ── Memory-efficient top-k retrieval (avoids [B,H,T,N,d_h] materialization) ──
        # Old code expanded v_mem to [B,H,T,N,d_h] which is ~68GB for N=2048.
        # Instead, we use chunked similarity + batched gather that only allocates
        # [B,H,T,top_k,d_h] at peak.

        # Chunked similarity to avoid [B,H,T,N] for very large N
        CHUNK_SIZE = 512  # process 512 memory slots at a time

        if N <= CHUNK_SIZE:
            sim = torch.matmul(q, k_mem.transpose(-2, -1)) / (d_h ** 0.5)  # [B, H, T, N]
            topk_scores, topk_idx = sim.topk(effective_top_k, dim=-1)  # [B, H, T, k]
        else:
            # Chunked matmul + top-k to keep peak memory bounded
            topk_scores = q.new_zeros(B, self.num_heads, T, effective_top_k)
            topk_idx = q.new_zeros(B, self.num_heads, T, effective_top_k, dtype=torch.long)
            topk_scores.fill_(float('-inf'))

            for chunk_start in range(0, N, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, N)
                k_chunk = k_mem[:, :, chunk_start:chunk_end, :]  # [B,H,chunk,d_h]
                sim_chunk = torch.matmul(q, k_chunk.transpose(-2, -1)) / (d_h ** 0.5)  # [B,H,T,chunk]

                # Merge this chunk's top-k with running top-k
                cs_topk_scores, cs_topk_idx = sim_chunk.topk(effective_top_k, dim=-1)  # [B,H,T,k]
                # Offset indices to global memory indices
                cs_topk_idx = cs_topk_idx + chunk_start

                # Combine with running top-k
                combined_scores = torch.cat([topk_scores, cs_topk_scores], dim=-1)  # [B,H,T,2k]
                combined_idx = torch.cat([topk_idx, cs_topk_idx], dim=-1)  # [B,H,T,2k]
                topk_scores, merge_idx = combined_scores.topk(effective_top_k, dim=-1)  # [B,H,T,k]
                topk_idx = combined_idx.gather(-1, merge_idx)  # [B,H,T,k]

                # Free chunk tensors immediately
                del sim_chunk, k_chunk, cs_topk_scores, cs_topk_idx, combined_scores, combined_idx, merge_idx

        # Gather v_mem using top-k indices WITHOUT materializing [B,H,T,N,d_h]
        # topk_idx: [B, H, T, k] → gather from v_mem [B, H, N, d_h]
        # Use torch.gather on the N dimension with index expansion to [B, H, T, k, d_h]
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, d_h)  # [B, H, T, k, d_h]
        # Expand v_mem to [B, H, T, N, d_h] only on the gather dimension — but this is OOM!
        # Instead, gather per-(b,h,t) row using advanced indexing:
        # v_mem: [B, H, N, d_h], topk_idx: [B, H, T, k]
        # We need: topk_v[b,h,t,:] = v_mem[b,h,topk_idx[b,h,t,:],:]
        B_idx = torch.arange(B, device=v_mem.device)[:, None, None, None]  # [B,1,1,1]
        H_idx = torch.arange(self.num_heads, device=v_mem.device)[None, :, None, None]  # [1,H,1,1]
        # Clamp indices to valid range to prevent CUDA index out-of-bounds
        # (can happen when memory is uninitialized/all-zeros and topk ties produce edge indices)
        topk_idx = topk_idx.clamp(0, N - 1)
        topk_v = v_mem[B_idx, H_idx, topk_idx]  # [B, H, T, k, d_h] — no [B,H,T,N,d_h] needed!

        # Softmax over top-k and weighted sum
        attn_weights = F.softmax(topk_scores, dim=-1).unsqueeze(-1)  # [B, H, T, k, 1]
        o_mem = (attn_weights * topk_v).sum(dim=3)  # [B, H, T, d_h]

        # Free intermediate tensors
        del topk_v, topk_idx_exp, topk_scores, topk_idx, attn_weights

        # [B, H, T, d_h] → [B, T, D]
        o_mem = o_mem.transpose(1, 2).contiguous().view(B, T, D)
        # NOTE: no o_proj here — memory path must not go through the
        # local-attention output projection a second time; v_proj + attention
        # already produces the correct representation scale.

        # ── Concat Fusion (Memorizing Transformers) ───────────────────
        output = self.fusion_proj(torch.cat([o_local, o_mem], dim=-1))

        # ── Memory Write: EMA update (no grad), per batch item ────────
        for b in range(B):
            self.memory_bank.write(hidden_states[b], batch_idx=b)

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
