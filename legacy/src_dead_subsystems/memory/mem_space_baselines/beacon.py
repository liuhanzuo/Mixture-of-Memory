"""Activation Beacon — interleaved compression via beacon tokens.

Reference: arXiv 2401.03462 (Activation Beacon)

Design:
    - Each LlamaDecoderLayer gets a dedicated set of beacon Q/K/V/O projections
      (warm-copied from the base layer weights at init).
    - At training time, long sequences are split into fixed-size intervals.
      At the end of each interval, k beacon tokens compress the interval's KV.
    - Subsequent intervals attend to [historical beacon KV] + [current interval],
      forcing every token's loss to read through the compressed representation.

This file defines:
    BeaconProjection — per-layer beacon Q/K/V/O weight set
    BeaconLayer      — wraps a LlamaDecoderLayer, adds beacon logic
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BeaconProjection(nn.Module):
    """Beacon-specific Q/K/V/O projections for one decoder layer.

    Initialized as a warm copy of the base layer's attention weights so the
    beacon starts with the same representational capacity as normal tokens.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

    def warm_copy_from(self, attn_module: nn.Module) -> None:
        """Copy weights from a HuggingFace LlamaAttention module."""
        with torch.no_grad():
            self.q_proj.weight.copy_(attn_module.q_proj.weight)
            self.k_proj.weight.copy_(attn_module.k_proj.weight)
            self.v_proj.weight.copy_(attn_module.v_proj.weight)
            self.o_proj.weight.copy_(attn_module.o_proj.weight)


class BeaconLayer(nn.Module):
    """Wraps a LlamaDecoderLayer with beacon compression capability.

    The base layer is frozen. Only the beacon projections are trainable.
    During forward, beacon tokens attend to the interval's hidden states
    using the beacon-specific Q/K/V/O projections, producing compressed KV
    that subsequent intervals will attend to.
    """

    def __init__(self, base_layer: nn.Module, layer_idx: int, n_beacon: int = 4):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_beacon = n_beacon

        # Store base layer (frozen, not registered as submodule to avoid
        # double-counting params — it stays in model.model.layers)
        object.__setattr__(self, "base_layer", base_layer)

        # Extract dimensions from the base attention module
        attn = base_layer.self_attn
        self.d_model = attn.q_proj.weight.shape[1]
        self.n_heads = attn.config.num_attention_heads
        self.n_kv_heads = attn.config.num_key_value_heads
        self.d_head = self.d_model // self.n_heads

        # Beacon projections (trainable, warm-copied from base)
        self.beacon_proj = BeaconProjection(
            self.d_model, self.n_heads, self.n_kv_heads, self.d_head
        )
        self.beacon_proj.warm_copy_from(attn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        beacon_embeds: Optional[torch.Tensor] = None,
        beacon_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with beacon compression.

        Args:
            hidden_states: [B, S, D] current interval tokens
            beacon_embeds: [B, n_beacon, D] beacon token embeddings
                (only provided at interval boundaries for compression)
            beacon_kv_cache: (K, V) each [B, n_kv_heads, N_hist, d_head]
                accumulated beacon KV from previous intervals
            position_ids: [B, S] position IDs for RoPE
            attention_mask: optional causal mask

        Returns:
            (output_hidden, new_beacon_kv) where new_beacon_kv is the
            updated cache including this interval's beacon KV (or None
            if no beacon_embeds provided).
        """
        B, S, D = hidden_states.shape
        residual = hidden_states

        # --- Step 1: Run base layer's input_layernorm ---
        normed = self.base_layer.input_layernorm(hidden_states)

        # --- Step 2: Compute base Q/K/V for interval tokens ---
        attn = self.base_layer.self_attn
        q = attn.q_proj(normed)  # [B, S, n_heads * d_head]
        k = attn.k_proj(normed)  # [B, S, n_kv_heads * d_head]
        v = attn.v_proj(normed)  # [B, S, n_kv_heads * d_head]

        # Reshape for multi-head attention
        q = q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to interval tokens
        if hasattr(attn, "rotary_emb"):
            cos, sin = attn.rotary_emb(v, position_ids)
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # --- Step 3: Prepend historical beacon KV if available ---
        if beacon_kv_cache is not None:
            hist_k, hist_v = beacon_kv_cache  # [B, n_kv, N_hist, d_head]
            # GQA repeat for historical beacon KV
            k_full = torch.cat([hist_k, k], dim=2)
            v_full = torch.cat([hist_v, v], dim=2)
        else:
            k_full = k
            v_full = v

        # GQA: repeat KV heads to match Q heads
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k_full = k_full.repeat_interleave(n_rep, dim=1)
            v_full = v_full.repeat_interleave(n_rep, dim=1)

        # --- Step 4: Causal attention (interval tokens attend to
        #     [hist_beacon | current_interval]) ---
        scale = 1.0 / math.sqrt(self.d_head)
        attn_weights = torch.matmul(q, k_full.transpose(-2, -1)) * scale

        # Build causal mask: tokens can attend to all hist beacons +
        # causal within current interval
        N_hist = beacon_kv_cache[0].shape[2] if beacon_kv_cache is not None else 0
        total_kv = N_hist + S
        causal_mask = torch.zeros(
            (1, 1, S, total_kv), device=hidden_states.device, dtype=hidden_states.dtype
        )
        # Causal within current interval (positions N_hist .. N_hist+S-1)
        for i in range(S):
            # Can attend to all hist beacons (0..N_hist-1) and
            # current interval up to position i
            causal_mask[:, :, i, N_hist + i + 1:] = float("-inf")

        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(hidden_states.dtype)

        attn_out = torch.matmul(attn_weights, v_full)  # [B, n_heads, S, d_head]
        attn_out = attn_out.transpose(1, 2).reshape(B, S, -1)
        attn_out = attn.o_proj(attn_out)

        # Residual + post-attention layernorm + MLP (standard transformer)
        hidden_states = residual + attn_out
        residual2 = hidden_states
        hidden_states = self.base_layer.post_attention_layernorm(hidden_states)
        hidden_states = self.base_layer.mlp(hidden_states)
        hidden_states = residual2 + hidden_states

        # --- Step 5: Compute beacon KV if beacon_embeds provided ---
        new_beacon_kv = None
        if beacon_embeds is not None:
            new_beacon_kv = self._compute_beacon_kv(
                beacon_embeds, normed, k, v, position_ids
            )

        return hidden_states, new_beacon_kv

    def _compute_beacon_kv(
        self,
        beacon_embeds: torch.Tensor,
        interval_normed: torch.Tensor,
        interval_k: torch.Tensor,
        interval_v: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute beacon K/V by attending beacon queries to interval tokens.

        Beacon tokens use their own Q/K/V projections (trainable) to read
        from the interval's hidden states, producing compressed KV pairs.

        Args:
            beacon_embeds: [B, n_beacon, D] beacon input embeddings
            interval_normed: [B, S, D] layernorm'd interval hidden states
            interval_k: [B, n_kv, S, d_head] interval K (already RoPE'd)
            interval_v: [B, n_kv, S, d_head] interval V

        Returns:
            (beacon_k, beacon_v) each [B, n_kv_heads, n_beacon, d_head]
        """
        B, nb, D = beacon_embeds.shape
        target_dtype = interval_k.dtype
        beacon_embeds = beacon_embeds.to(target_dtype)

        # Beacon Q from beacon embeddings via beacon-specific projection
        # beacon_proj is float32 (trainable), inputs are bf16 from base model.
        # Cast input to float32 for projection, then cast output to target_dtype.
        beacon_normed = self.base_layer.input_layernorm(beacon_embeds)
        beacon_normed_f32 = beacon_normed.float()
        bq = self.beacon_proj.q_proj(beacon_normed_f32).to(target_dtype)
        bq = bq.view(B, nb, self.n_heads, self.d_head).transpose(1, 2)

        # Beacon K/V (these become the compressed representation stored)
        bk = self.beacon_proj.k_proj(beacon_normed_f32).to(target_dtype)
        bv = self.beacon_proj.v_proj(beacon_normed_f32).to(target_dtype)
        bk = bk.view(B, nb, self.n_kv_heads, self.d_head).transpose(1, 2)
        bv = bv.view(B, nb, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Beacon attends to interval tokens to absorb information
        # GQA expand interval K/V
        n_rep = self.n_heads // self.n_kv_heads
        ik = interval_k
        iv = interval_v
        if n_rep > 1:
            ik = ik.repeat_interleave(n_rep, dim=1)
            iv = iv.repeat_interleave(n_rep, dim=1)

        # beacon_q attends to interval K/V (full, non-causal)
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(bq, ik.transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        weights = weights.to(beacon_embeds.dtype)

        # Aggregate interval values into beacon
        beacon_agg = torch.matmul(weights, iv)  # [B, n_heads, nb, d_head]
        beacon_agg = beacon_agg.transpose(1, 2).reshape(B, nb, -1)
        beacon_agg = self.beacon_proj.o_proj(beacon_agg.float()).to(target_dtype)

        # Add residual + MLP for beacon tokens
        beacon_out = beacon_embeds + beacon_agg
        beacon_res = beacon_out
        beacon_out = self.base_layer.post_attention_layernorm(beacon_out)
        beacon_out = self.base_layer.mlp(beacon_out)
        beacon_out = beacon_res + beacon_out

        # The stored beacon KV is from the beacon projection (not the
        # aggregated output). This is what future intervals will attend to.
        return bk, bv


# --------------------------------------------------------------------------- #
# RoPE helper (matches HuggingFace Llama implementation)
# --------------------------------------------------------------------------- #


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K tensors."""
    # cos/sin shape: [B, S, d_head] or [1, S, d_head] from HF rotary_emb
    # q/k shape: [B, n_heads, S, d_head]
    cos = cos.unsqueeze(1)  # [B, 1, S, d_head]
    sin = sin.unsqueeze(1)  # [B, 1, S, d_head]

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# --------------------------------------------------------------------------- #
# BeaconModel — orchestrates streaming compression across all layers
# --------------------------------------------------------------------------- #


class BeaconModel(nn.Module):
    """Full beacon compression model wrapping a HuggingFace LlamaForCausalLM.

    Manages:
        - Beacon embedding tokens (learnable, shared across intervals)
        - Per-layer BeaconLayer wrappers
        - Streaming forward with interval-based compression
        - Historical beacon KV cache management
    """

    def __init__(
        self,
        base_model: nn.Module,
        n_beacon: int = 4,
        interval_size: int = 512,
    ):
        super().__init__()
        self.n_beacon = n_beacon
        self.interval_size = interval_size

        # Store base model (frozen)
        self.base_model = base_model
        config = base_model.config
        self.d_model = config.hidden_size
        self.n_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size

        # Learnable beacon embeddings [n_beacon, d_model]
        self.beacon_embedding = nn.Parameter(
            torch.randn(n_beacon, self.d_model) * 0.02
        )

        # Create BeaconLayer wrappers for each decoder layer
        layers = base_model.model.layers
        self.beacon_layers = nn.ModuleList()
        for i in range(self.n_layers):
            bl = BeaconLayer(layers[i], layer_idx=i, n_beacon=n_beacon)
            self.beacon_layers.append(bl)

        # Initialize beacon embedding from mean of token embeddings
        with torch.no_grad():
            embed_weight = base_model.model.embed_tokens.weight
            mean_embed = embed_weight.mean(dim=0)
            self.beacon_embedding.data.copy_(
                mean_embed.unsqueeze(0).expand(n_beacon, -1)
                + torch.randn(n_beacon, self.d_model, device=embed_weight.device) * 0.01
            )

    def get_trainable_params(self) -> list:
        """Return only the trainable parameters (beacon projections + embedding)."""
        params = [self.beacon_embedding]
        for bl in self.beacon_layers:
            for p in bl.beacon_proj.parameters():
                params.append(p)
        return params

    def forward_streaming(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        compression_ratio: int = 4,
    ) -> dict:
        """Streaming forward with interval-based beacon compression.

        Splits input_ids into intervals of size `interval_size`. For each
        interval:
            1. Run all layers on interval tokens (attending to hist beacon KV)
            2. At interval boundary, compute beacon KV for this interval
            3. Discard interval's raw KV, keep only beacon KV for next interval

        Args:
            input_ids: [B, total_len] token IDs
            labels: [B, total_len] target IDs for CE loss (shifted internally)
            compression_ratio: how many tokens per beacon token
                (interval_size = n_beacon * compression_ratio)

        Returns:
            dict with 'loss', 'logits', 'n_intervals'
        """
        B, total_len = input_ids.shape
        device = input_ids.device

        # Effective interval size based on compression ratio
        effective_interval = self.n_beacon * compression_ratio
        n_intervals = max(1, total_len // effective_interval)

        # Embed all tokens
        all_embeds = self.base_model.model.embed_tokens(input_ids)

        # Per-layer beacon KV cache: list of (K, V) per layer
        # Each is [B, n_kv_heads, accumulated_beacons, d_head]
        layer_beacon_cache = [None] * self.n_layers

        all_logits = []
        total_loss = torch.zeros((), device=device)
        n_loss_tokens = 0

        for interval_idx in range(n_intervals):
            start = interval_idx * effective_interval
            end = min(start + effective_interval, total_len)
            if start >= total_len:
                break

            interval_embeds = all_embeds[:, start:end, :]  # [B, S_i, D]
            S_i = interval_embeds.shape[1]

            # Position IDs for this interval (absolute positions)
            pos_ids = torch.arange(
                start, start + S_i, device=device
            ).unsqueeze(0).expand(B, -1)

            # Beacon embeddings for this interval
            beacon_emb = self.beacon_embedding.unsqueeze(0).expand(
                B, -1, -1
            )  # [B, n_beacon, D]

            # Forward through all layers
            hidden = interval_embeds
            for layer_idx, bl in enumerate(self.beacon_layers):
                hidden, new_bkv = bl(
                    hidden_states=hidden,
                    beacon_embeds=beacon_emb,
                    beacon_kv_cache=layer_beacon_cache[layer_idx],
                    position_ids=pos_ids,
                )

                # Update beacon KV cache for this layer
                if new_bkv is not None:
                    bk, bv = new_bkv
                    if layer_beacon_cache[layer_idx] is not None:
                        old_k, old_v = layer_beacon_cache[layer_idx]
                        layer_beacon_cache[layer_idx] = (
                            torch.cat([old_k, bk], dim=2),
                            torch.cat([old_v, bv], dim=2),
                        )
                    else:
                        layer_beacon_cache[layer_idx] = (bk, bv)

            # Final layernorm + LM head
            hidden = self.base_model.model.norm(hidden)
            logits = self.base_model.lm_head(hidden)  # [B, S_i, vocab]
            all_logits.append(logits)

            # Compute loss for this interval
            if labels is not None:
                interval_labels = labels[:, start:end]
                # Shift: predict next token
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = interval_labels[:, 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
                interval_loss = loss_fct(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                )
                valid_tokens = (shift_labels != -100).sum().item()
                total_loss = total_loss + interval_loss
                n_loss_tokens += valid_tokens

            # Detach beacon cache to prevent cross-interval BPTT explosion
            # (gradient flows within interval, not across)
            layer_beacon_cache = [
                (kv[0].detach(), kv[1].detach()) if kv is not None else None
                for kv in layer_beacon_cache
            ]

        # Average loss
        if n_loss_tokens > 0:
            avg_loss = total_loss / n_loss_tokens
        else:
            avg_loss = total_loss

        return {
            "loss": avg_loss,
            "logits": torch.cat(all_logits, dim=1) if all_logits else None,
            "n_intervals": n_intervals,
        }

    def reset_cache(self) -> None:
        """Reset beacon KV cache (call at document boundaries)."""
        # Cache is managed per-forward call, this is a no-op placeholder
        pass
