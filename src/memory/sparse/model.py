"""
SparseMemoryModel — Wrapper that patches a HuggingFace model with
ConcatFusionAttention + SparseMemoryBank.

Supports Qwen3-8B and Llama2-7B (or any model with compatible attention).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .attention import ConcatFusionAttention
from .memory_bank import SparseMemoryBank


class SparseMemoryModel(nn.Module):
    """Wrap a HF model with sliding window + sparse memory retrieval.

    Args:
        model: A HuggingFace model (e.g. Qwen3-8B, Llama2-7B).
        layers_to_patch: List of layer indices to inject memory into.
            Default: all layers.
        num_slots: Memory bank capacity per layer (default 128).
        window_size: Sliding window size (default 256).
        top_k: Top-k memory retrieval (default 8).
        ema_alpha: EMA write decay rate (default 0.1).
        gate_bias_init: Gate bias for both write and fusion gates (default 0.0 → σ(0)=0.5).
    """

    def __init__(
        self,
        model: nn.Module,
        layers_to_patch: Optional[List[int]] = None,
        num_slots: int = 128,
        window_size: int = 256,
        top_k: int = 8,
        ema_alpha: float = 0.1,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.window_size = window_size
        self.top_k = top_k

        # Auto-detect architecture
        self.arch_type = self._detect_arch(model)

        # Determine config
        config = model.config
        num_layers = config.num_hidden_layers
        hidden_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_dim // num_heads

        if layers_to_patch is None:
            layers_to_patch = list(range(num_layers))
        self.layers_to_patch = layers_to_patch

        # Create independent per-layer memory banks to avoid
        # "shared tensors" RuntimeError when saving checkpoints.
        # Each layer gets its own SparseMemoryBank with num_layers=1.
        self.memory_banks: nn.ModuleList = nn.ModuleList()

        # Patch attention modules
        self._patched_layers: List[int] = []
        self._original_attns: dict = {}
        for layer_idx in layers_to_patch:
            attn = self._get_attention(model, layer_idx)
            if attn is None:
                continue
            self._original_attns[layer_idx] = attn
            self._patched_layers.append(layer_idx)

            # Independent bank per layer (num_layers=1, layer_idx=0)
            bank = SparseMemoryBank(
                num_layers=1,
                num_slots=num_slots,
                hidden_dim=hidden_dim,
                head_dim=head_dim,
                ema_alpha=ema_alpha,
                gate_bias_init=gate_bias_init,
            )
            self.memory_banks.append(bank)

            gated_attn = ConcatFusionAttention(
                original_attn=attn,
                layer_idx=0,  # single-layer bank, always index 0
                memory_bank=bank,
                window_size=window_size,
                top_k=top_k,
                head_dim=head_dim,
                bypass_bias_init=gate_bias_init,
            )
            self._set_attention(model, layer_idx, gated_attn)

    def gradient_checkpointing_enable(self, **kwargs):
        """Pass-through for HF Trainer (transformers 5.x compatibility)."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """Pass-through for HF Trainer compatibility."""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

    @staticmethod
    def _detect_arch(model: nn.Module) -> str:
        cls_name = model.__class__.__name__.lower()
        if "qwen" in cls_name:
            return "qwen"
        if "llama" in cls_name:
            return "llama"
        return "unknown"

    def _get_attention(self, model: nn.Module, layer_idx: int) -> Optional[nn.Module]:
        """Get the self-attention module for a given layer."""
        arch = self.arch_type
        try:
            if arch == "qwen":
                return model.model.layers[layer_idx].self_attn
            elif arch == "llama":
                return model.model.layers[layer_idx].self_attn
            else:
                # Generic fallback: try model.layers[layer_idx].self_attn
                return model.model.layers[layer_idx].self_attn
        except (AttributeError, IndexError):
            return None

    def _set_attention(self, model: nn.Module, layer_idx: int, attn: nn.Module) -> None:
        """Replace the self-attention module for a given layer."""
        arch = self.arch_type
        if arch in ("qwen", "llama", "unknown"):
            model.model.layers[layer_idx].self_attn = attn
        else:
            model.model.layers[layer_idx].self_attn = attn

    @property
    def config(self):
        return self.model.config

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Forward pass through the model with memory integration."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return outputs

    def reset_memory(self) -> None:
        """Reset all per-layer memory banks."""
        for bank in self.memory_banks:
            bank.reset()

    @property
    def gate_values(self) -> torch.Tensor:
        """Access gate values from all patched layers for diagnostics."""
        values = []
        for layer_idx in self._patched_layers:
            attn = self._get_attention(self.model, layer_idx)
            if isinstance(attn, ConcatFusionAttention):
                values.append(attn.bypass_gate_proj)
        return values
