"""DMS Attention Wrapper: modifies attention to apply eviction mask.

During training, the eviction decisions create an additive mask M_α that is
added to the attention scores. During inference, tokens marked for eviction
are removed from the KV cache after the sliding window delay.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dms_decision_head import DMSDecisionHead

logger = logging.getLogger(__name__)


class DMSAttentionWrapper(nn.Module):
    """Wraps a HuggingFace attention layer with DMS eviction logic.

    During training:
        1. Decision head predicts per-token eviction scores α_t ∈ [0, 1]
        2. An additive attention mask M_α is constructed (delayed eviction via sliding window)
        3. Mask is added to attention logits before softmax

    During inference:
        1. Decision head predicts binary eviction decisions
        2. Tokens scheduled for eviction are removed from KV cache after sliding window delay

    Args:
        attention_layer: The original HF attention layer (LlamaAttention, etc.)
        layer_idx: Index of this layer in the model.
        sliding_window: Number of recent tokens protected from eviction.
        tau: Gumbel-Sigmoid temperature for training.
    """

    def __init__(
        self,
        attention_layer: nn.Module,
        layer_idx: int,
        sliding_window: int = 256,
        tau: float = 0.1,
    ):
        super().__init__()
        self.attention_layer = attention_layer
        self.layer_idx = layer_idx
        self.sliding_window = sliding_window
        self.tau = tau

        # Get hidden_dim from the attention layer
        hidden_dim = self._get_hidden_dim()
        self.decision_head = DMSDecisionHead(hidden_dim, bias_init=-5.0, tau=tau)

        # Store eviction decisions for inference
        self.eviction_buffer: dict[str, Any] = {}

    def _get_hidden_dim(self) -> int:
        """Extract hidden dimension from the wrapped attention layer."""
        # Best: q_proj.out_features gives the full hidden dim
        if hasattr(self.attention_layer, "q_proj"):
            return self.attention_layer.q_proj.out_features
        # Try common attribute names
        for attr in ["hidden_size"]:
            if hasattr(self.attention_layer, attr):
                val = getattr(self.attention_layer, attr)
                if isinstance(val, int) and val > 0:
                    return val
        # head_dim * num_heads
        if hasattr(self.attention_layer, "head_dim") and hasattr(self.attention_layer, "num_heads"):
            return self.attention_layer.head_dim * self.attention_layer.num_heads
        raise ValueError("Cannot determine hidden dimension from attention layer")

    def _build_eviction_mask(
        self,
        alpha: torch.Tensor,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build additive attention mask from eviction decisions with sliding window delay.

        Token at position i marked for eviction (α_i = 1) will have its mask entry
        set to -inf only after positions beyond i + sliding_window.

        Args:
            alpha: Eviction decisions, shape ``(B, T)`` or ``(B, num_kv_heads, T)``.
            seq_len: Total sequence length T.
            dtype: Torch dtype for the mask.
            device: Torch device.

        Returns:
            Additive mask, shape ``(B, 1, T, T)`` (broadcastable to attention scores).
        """
        # Ensure alpha is (B, T) — average over kv heads if present (GQA compat)
        if alpha.dim() == 3:
            alpha = alpha.mean(dim=1)  # (B, T)

        B, T = alpha.shape
        w = self.sliding_window

        q_pos = torch.arange(T, device=device, dtype=dtype).view(1, T, 1)
        k_pos = torch.arange(T, device=device, dtype=dtype).view(1, 1, T)

        delay_passed = (q_pos > k_pos + w).float()  # (1, T, T)

        # alpha: (B, T) → (B, 1, T) for broadcasting with (1, T, T)
        mask = alpha.unsqueeze(1) * delay_passed  # (B, T, T)

        # Convert to additive mask: use large negative value instead of -inf
        # to avoid NaN issues with bfloat16 (0 * -inf = NaN)
        NEG_INF = torch.finfo(dtype).min
        mask = torch.where(mask > 0.5, NEG_INF, torch.tensor(0.0, dtype=dtype, device=device))

        # Causal mask: queries cannot attend to future keys
        causal_mask = torch.triu(
            torch.full((T, T), NEG_INF, device=device, dtype=dtype), diagonal=1
        )
        mask = torch.maximum(mask, causal_mask.unsqueeze(0))  # (B, T, T)

        # (B, 1, T, T) — broadcasts to any head count (works with GQA)
        return mask.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple] = None,
        **kwargs,
    ) -> tuple:
        """Forward pass with DMS eviction.

        During training: modifies attention via additive mask.
        During inference: modifies KV cache by removing evicted tokens.
        """
        self.training_mode = self.training

        # Accept both past_key_value (old HF) and past_key_values (new HF like Qwen3)
        kv_cache = past_key_values if past_key_values is not None else past_key_value

        if self.training:
            return self._forward_training(
                hidden_states, attention_mask, position_ids,
                kv_cache, output_attentions, use_cache,
                cache_position, position_embeddings, **kwargs,
            )
        else:
            return self._forward_inference(
                hidden_states, attention_mask, position_ids,
                kv_cache, output_attentions, use_cache,
                cache_position, position_embeddings, **kwargs,
            )

    def _forward_training(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Any],
        output_attentions: bool,
        use_cache: bool,
        cache_position: Optional[torch.Tensor],
        position_embeddings: Optional[tuple],
        **kwargs,
    ) -> tuple:
        """Training forward with additive eviction mask."""
        B, T, D = hidden_states.shape

        # Compute eviction decisions (relaxed, differentiable)
        alpha = self.decision_head(hidden_states)  # (B, T)

        # Store alpha for loss computation
        self._last_alpha = alpha.detach()
        self._last_seq_len = T

        # Build eviction mask
        dms_mask = self._build_eviction_mask(
            alpha, T, hidden_states.dtype, hidden_states.device
        )

        # Merge with existing attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match dms_mask shape
            if attention_mask.dim() == 2:
                # (B, T) → (B, 1, 1, T) for broadcast to (B, 1, T_q, T_k)
                attn_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 4:
                attn_mask = attention_mask
            else:
                attn_mask = attention_mask

            # Convert 0/1 mask to additive: 0 → -inf, 1 → 0
            NEG_INF = torch.finfo(attn_mask.dtype if attn_mask.is_floating_point() else torch.float32).min
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.where(attn_mask, 0.0, NEG_INF)
            elif attn_mask.min() >= 0:
                attn_mask = torch.where(attn_mask.bool(), 0.0, NEG_INF)

            # Combine: take element-wise maximum (most restrictive)
            # dms_mask is (B, 1, T, T), attn_mask broadcasts naturally
            combined_mask = torch.maximum(attn_mask, dms_mask)
        else:
            combined_mask = dms_mask

        # Call the original attention layer with modified mask
        # Use past_key_values (plural) for newer HF models (Qwen3, etc.)
        attn_kwargs = dict(
            attention_mask=combined_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        if position_embeddings is not None:
            attn_kwargs["position_embeddings"] = position_embeddings
        attn_kwargs.update(kwargs)

        return self.attention_layer(hidden_states, **attn_kwargs)

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Any],
        output_attentions: bool,
        use_cache: bool,
        cache_position: Optional[torch.Tensor],
        position_embeddings: Optional[tuple],
        **kwargs,
    ) -> tuple:
        """Inference forward with KV cache eviction."""
        # For inference, use the original attention layer
        # Eviction is handled externally via the DMS press mechanism
        return self.attention_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )


def apply_dms_to_model(
    model: nn.Module,
    sliding_window: int = 256,
    tau: float = 0.1,
    target_layers: Optional[list[int]] = None,
) -> nn.Module:
    """Retrofit a HuggingFace model with DMS decision heads.

    Wraps each decoder layer's self-attention with DMS eviction logic.

    Args:
        model: HuggingFace model (e.g., LlamaForCausalLM).
        sliding_window: Tokens protected from immediate eviction.
        tau: Gumbel-Sigmoid temperature.
        target_layers: Specific layer indices to apply DMS. None = all layers.

    Returns:
        Modified model with DMS wrappers.
    """
    # Find decoder layers
    decoder_layers = None
    for name, module in model.named_modules():
        if "decoder" in name.lower() and "layers" in name.lower():
            decoder_layers = module
            logger.info(f"Found decoder layers at: {name}")
            break

    if decoder_layers is None:
        # Try model.model.layers pattern (Llama, Mistral, etc.)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            decoder_layers = model.model.layers
            logger.info("Found decoder layers at: model.layers")
        else:
            raise ValueError("Cannot find decoder layers in the model")

    num_layers = len(decoder_layers)
    if target_layers is None:
        target_layers = list(range(num_layers))

    for layer_idx in target_layers:
        if layer_idx >= num_layers:
            logger.warning(f"Layer {layer_idx} out of range, skipping")
            continue

        layer = decoder_layers[layer_idx]
        if hasattr(layer, "self_attn"):
            original_attn = layer.self_attn
            wrapper = DMSAttentionWrapper(
                original_attn,
                layer_idx=layer_idx,
                sliding_window=sliding_window,
                tau=tau,
            )
            # Move wrapper to the same device as the attention layer's parameters
            device = next(original_attn.parameters()).device
            wrapper = wrapper.to(device)
            layer.self_attn = wrapper
            logger.info(f"Wrapped layer {layer_idx} self_attn with DMS (device={device})")

    # Count DMS parameters
    dms_params = sum(
        p.numel() for name, p in model.named_parameters()
        if "decision_head" in name
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"DMS parameters: {dms_params:,} / {total_params:,} "
        f"({100 * dms_params / total_params:.4f}%)"
    )

    return model
