"""SparseMemoryLlamaForCausalLM — Llama2 with per-layer sparse memory.

Inherits LlamaForCausalLM and replaces all self_attn layers with
SparseMemoryAttention (local sliding window + memory retrieval + gated fusion).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig

from .memory_bank import MemoryBank
from .attention import SparseMemoryAttention


class SparseMemoryLlamaForCausalLM(nn.Module):
    """Llama2-7B with sliding window + sparse memory retrieval.

    Args:
        base_model: Path or pretrained LlamaForCausalLM.
        memory_slots: Memory bank capacity N (default 128).
        top_k: Top-k memory retrieval (default 8).
        sliding_window: Sliding window size w (default 256).
        ema_alpha: EMA write decay rate (default 0.1).
    """

    def __init__(
        self,
        base_model: str | nn.Module,
        memory_slots: int = 128,
        top_k: int = 8,
        sliding_window: int = 256,
        ema_alpha: float = 0.1,
        torch_dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = False,
        memory_dtype: torch.dtype | None = None,
        # L0/L1 multi-level memory params
        use_l1: bool = False,
        num_mem_tokens: int = 16,
        l1_num_tokens: int = 16,
        segment_length: int = 1024,
        max_segments: int = 4,
        bptt_depth: int = 2,
        recon_loss_coef: float = 0.1,
        use_importance_routing: bool = False,
        # Selective memory writing params
        write_top_k: int = 0,
        importance_mode: str = "combined",
    ) -> None:
        super().__init__()

        if isinstance(base_model, str):
            self.model = LlamaForCausalLM.from_pretrained(
                base_model, torch_dtype=torch_dtype
            )
        else:
            self.model = base_model

        config: LlamaConfig = self.model.config
        self.config = config

        # Enable gradient checkpointing to reduce activation memory
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Memory dtype for banks (default: match model dtype)
        self._memory_dtype = memory_dtype if memory_dtype is not None else torch_dtype

        # Importance-based selective writing params
        self._write_top_k = write_top_k
        self._importance_mode = importance_mode

        # Create per-layer memory banks
        num_layers = config.num_hidden_layers
        self.memory_banks = nn.ModuleList([
            MemoryBank(
                num_slots=memory_slots,
                hidden_dim=config.hidden_size,
                ema_alpha=ema_alpha,
                dtype=self._memory_dtype,
                write_top_k=self._write_top_k,
                importance_mode=self._importance_mode,
            )
            for _ in range(num_layers)
        ])

        # Replace attention in each layer
        self.sparse_attn_layers: List[SparseMemoryAttention] = []
        for i in range(num_layers):
            original_attn = self.model.model.layers[i].self_attn
            sparse_attn = SparseMemoryAttention(
                original_attn=original_attn,
                memory_bank=self.memory_banks[i],
                window_size=sliding_window,
                top_k=top_k,
            )
            self.model.model.layers[i].self_attn = sparse_attn
            self.sparse_attn_layers.append(sparse_attn)

        # Stash config for easy access
        self._memory_slots = memory_slots
        self._top_k = top_k
        self._sliding_window = sliding_window

        # ── L1 memory layer (stub) ──
        # TODO: Implement full L1 memory compression layer that aggregates
        #   L0 segment-level memories into a coarser L1 representation.
        #   Expected flow:
        #     1. Each segment produces `num_mem_tokens` L0 memory vectors
        #     2. L1 compresses across segments: L0 → l1_num_tokens vectors
        #     3. L1 memories are used as additional context for later segments
        #     4. Reconstruction loss: L1 should reconstruct L0 memories
        #   For now, store the config so training can proceed without crashing.
        self.use_l1 = use_l1
        self.num_mem_tokens = num_mem_tokens
        self.l1_num_tokens = l1_num_tokens
        self.segment_length = segment_length
        self.max_segments = max_segments
        self.bptt_depth = bptt_depth
        self.recon_loss_coef = recon_loss_coef
        self.use_importance_routing = use_importance_routing

        if use_l1:
            # Placeholder: L1 compression projection
            self.l1_compress = nn.Linear(
                config.hidden_size * num_mem_tokens,
                config.hidden_size * l1_num_tokens,
                bias=False,
            )
            self.l1_decompress = nn.Linear(
                config.hidden_size * l1_num_tokens,
                config.hidden_size * num_mem_tokens,
                bias=False,
            )

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def reset_memory(self, batch_size: int = 1) -> None:
        """Reset all memory banks for given batch size."""
        for bank in self.memory_banks:
            bank.reset(batch_size=batch_size)

    def get_fusion_grad_norm(self) -> float:
        """Average gradient norm of fusion_proj weights across layers."""
        total_norm_sq = 0.0
        count = 0
        for attn in self.sparse_attn_layers:
            p = attn.fusion_proj.weight
            if p.grad is not None:
                total_norm_sq += p.grad.data.float().norm().item() ** 2
                count += 1
        return (total_norm_sq / count) ** 0.5 if count > 0 else 0.0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Forward pass with memory reset per sample.

        Memory is reset at the start of each forward call (each sample is independent).
        """
        B = input_ids.shape[0]
        self.reset_memory(batch_size=B)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    @property
    def lm_head(self):
        return self.model.lm_head

    @property
    def model_embed_tokens(self):
        return self.model.model.embed_tokens
