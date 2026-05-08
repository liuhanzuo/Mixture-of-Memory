"""RMT-Slot hybrid: top-k slot retrieval + RMT sandwich injection on Llama-3-8B.

Architecture:
    1. Maintain N=64 persistent memory slots (MemoryBank).
    2. Each segment: retrieve top-k=8 slots via TopKSelector using mean-pooled content query.
    3. Build RMT sandwich: [retrieved_slots | content_tokens | placeholder_mem]
       with continuous position IDs and causal mask (old_mem rows can see everything).
    4. Forward through the full transformer; extract new_mem from placeholder positions.
    5. EMA write-back only to the selected slot indices (gate is learnable sigmoid logit).
    6. Repeat for each segment; loss is computed on content positions only.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.memory.mem_space.selector import TopKSelector
from src.memory.mem_space.memory_bank import MemoryBank


def _inverse_sigmoid(x: float) -> float:
    """Compute logit (inverse sigmoid) of x in (0, 1)."""
    return math.log(x / (1.0 - x))


@dataclass
class RMTSlotConfig:
    """Configuration for RMT-Slot hybrid model."""
    num_slots: int = 64              # N — persistent bank size
    top_k: int = 8                   # retrieved per segment
    segment_length: int = 1024       # tokens per segment
    max_n_segments: int = 4          # max segments during training
    selector_dim: int = 128          # projection dim for TopKSelector
    ema_gate_init: float = 0.3       # sigmoid logit init value for EMA gate
    slot_value_norm_cap: float = 8.0 # prevent bf16 overflow on slots
    bptt_depth: int = -1             # -1 = full BPTT; positive = truncated
    use_importance_routing: bool = False  # start simple
    vary_n_segments: bool = True     # randomize num_segments during training
    gradient_checkpointing: bool = True


class RMTSlotModel(nn.Module):
    """RMT-Slot hybrid: top-k slot retrieval + RMT sandwich injection.

    Wraps a pre-trained LlamaForCausalLM and adds:
    - A persistent MemoryBank of N slots
    - A TopKSelector for content-based slot retrieval
    - Learnable placeholder tokens for the sandwich suffix
    - A learnable EMA gate for write-back
    """

    def __init__(self, base_model: nn.Module, config: RMTSlotConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.hidden_dim = base_model.config.hidden_size  # 4096 for Llama-3-8B

        # Top-k selector (imported from mem_space)
        self.selector = TopKSelector(
            d_model=self.hidden_dim,
            slot_dim=self.hidden_dim,
            selector_dim=config.selector_dim,
            top_k=config.top_k,
            num_slots=config.num_slots,
            temperature=1.0,
        )

        # Memory bank (imported from mem_space)
        self.bank = MemoryBank(
            num_slots=config.num_slots,
            slot_dim=self.hidden_dim,
            slot_init="hidden_pool",
            slot_value_norm_cap=config.slot_value_norm_cap,
        )

        # Learnable placeholder tokens for the sandwich suffix
        embed_std = base_model.model.embed_tokens.weight.std().item()
        self.placeholder = nn.Parameter(
            torch.empty(config.top_k, self.hidden_dim).normal_(std=embed_std)
        )

        # Learnable EMA gate (stored as logit, applied via sigmoid)
        self.gate_logit = nn.Parameter(
            torch.tensor(_inverse_sigmoid(config.ema_gate_init))
        )

        # Enable gradient checkpointing on base model if requested
        if config.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

    def _build_sandwich(
        self,
        content_embeds: torch.Tensor,
        retrieved_slots: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build RMT sandwich: [retrieved_slots | content | placeholder].

        Args:
            content_embeds: [B, S, D] — embedded content tokens
            retrieved_slots: [B, top_k, D] — retrieved slot vectors

        Returns:
            inputs_embeds: [B, T, D] where T = 2*top_k + S
            attn_mask_4d: [B, 1, T, T] additive float mask (-inf where masked, 0 where attended)
            position_ids: [B, T] continuous position IDs
        """
        B, S, D = content_embeds.shape
        K = self.config.top_k
        device = content_embeds.device
        dtype = content_embeds.dtype
        T = 2 * K + S

        # Expand placeholder to batch: [B, K, D]
        placeholder = self.placeholder.unsqueeze(0).expand(B, -1, -1)

        # Concatenate: [retrieved_slots | content | placeholder]
        inputs_embeds = torch.cat([retrieved_slots, content_embeds, placeholder], dim=1)

        # Build causal mask with memory prefix seeing everything
        # Start with lower-triangular (causal)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        # Memory prefix (rows 0:K) can see all positions (bidirectional-ish)
        causal[:K, :] = True

        # Convert to additive float mask: 0.0 where attended, -inf where masked
        neg_inf = torch.finfo(dtype).min
        attn_mask = torch.where(causal, torch.tensor(0.0, device=device, dtype=dtype),
                                torch.tensor(neg_inf, device=device, dtype=dtype))
        # Expand to 4D: [B, 1, T, T]
        attn_mask_4d = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T).contiguous()

        # Continuous position IDs
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        return inputs_embeds, attn_mask_4d, position_ids

    def _forward_backbone(
        self,
        inputs_embeds: torch.Tensor,
        attn_mask_4d: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through the backbone transformer (layer-by-layer).

        Returns:
            last_hidden_state: [B, T, D]
        """
        B, T, D = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        hidden_states = inputs_embeds

        # Get RoPE embeddings
        rotary_emb = self.base_model.model.rotary_emb
        position_embeddings = rotary_emb(hidden_states, position_ids)

        # Forward through decoder layers
        for layer in self.base_model.model.layers:
            layer_out = layer(
                hidden_states,
                attention_mask=attn_mask_4d,
                position_ids=None,
                past_key_value=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # Final layer norm
        hidden_states = self.base_model.model.norm(hidden_states)

        return hidden_states

    def reset_slots(self, batch_size: Optional[int] = None) -> None:
        """Reset the memory bank for a new document."""
        self.bank.reset(batch_size=batch_size)

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward a single chunk through the model with memory.

        Used during evaluation (chunk-by-chunk streaming).

        Args:
            input_ids: [B, S] token IDs for one chunk
            labels: [B, S] (optional) for loss computation

        Returns:
            dict with keys: logits, (loss if labels given)
        """
        B, S = input_ids.shape
        D = self.hidden_dim
        K = self.config.top_k
        device = input_ids.device

        # Embed content tokens
        content_embeds = self.base_model.model.embed_tokens(input_ids)

        # Initialize bank if needed
        if not self.bank.is_initialized(B):
            self.bank.init_from_hidden(content_embeds.detach(), batch_size=B)

        # Get current slots
        slots = self.bank.get()  # [B, N, D]

        # Build query from content (mean-pool)
        pool_q = content_embeds.mean(dim=1)  # [B, D]

        # Select top-k slots
        idx, scores, ste_weights = self.selector(pool_q, slots)  # idx: [B, K]

        # Gather retrieved slots
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)  # [B, K, D]
        retrieved = slots.gather(1, idx_exp)  # [B, K, D]

        # Apply STE weights for gradient flow
        gathered_weights = ste_weights.gather(1, idx)  # [B, K]
        retrieved = retrieved * gathered_weights.unsqueeze(-1)

        # Build sandwich
        inputs_embeds, attn_mask_4d, position_ids = self._build_sandwich(
            content_embeds, retrieved
        )

        # Forward through backbone
        hidden_states = self._forward_backbone(inputs_embeds, attn_mask_4d, position_ids)

        # Extract new memory from placeholder positions
        new_mem = hidden_states[:, -K:, :]  # [B, K, D]

        # Compute logits on content positions only
        content_h = hidden_states[:, K:K+S, :]  # [B, S, D]
        logits = self.base_model.lm_head(content_h)  # [B, S, vocab]

        # Write-back via EMA
        gate = torch.sigmoid(self.gate_logit)
        self.bank.write(idx, new_mem, gate=gate)

        result = {"logits": logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            result["loss"] = loss

        return result

    def forward_vanilla(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward WITHOUT any memory (vanilla model)."""
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, labels=labels)
        result = {"logits": outputs.logits}
        if outputs.loss is not None:
            result["loss"] = outputs.loss
        return result

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Full forward: segment input_ids, loop through segments with memory.

        Args:
            input_ids: [B, L] where L = chunks_per_doc * segment_length
            labels: [B, L] (optional)

        Returns:
            dict: loss, ce_loss, num_segments, aux_metrics
        """
        B, L = input_ids.shape
        S = self.config.segment_length
        K = self.config.top_k
        D = self.hidden_dim
        device = input_ids.device

        # Determine number of segments
        max_segs = min(self.config.max_n_segments, L // S)
        if max_segs < 1:
            max_segs = 1

        if self.training and self.config.vary_n_segments and max_segs > 1:
            num_segments = random.randint(1, max_segs)
        else:
            num_segments = max_segs

        # Truncate input to fit num_segments * S
        effective_len = num_segments * S
        input_ids = input_ids[:, :effective_len]
        if labels is not None:
            labels = labels[:, :effective_len]

        # Reset bank for this document
        self.bank.reset(batch_size=B)

        all_logits = []
        all_labels = []

        for seg_idx in range(num_segments):
            start = seg_idx * S
            end = start + S
            seg_ids = input_ids[:, start:end]  # [B, S]
            seg_labels = labels[:, start:end] if labels is not None else None

            # Embed content tokens
            content_embeds = self.base_model.model.embed_tokens(seg_ids)  # [B, S, D]

            # Initialize bank from first segment's content if needed
            if not self.bank.is_initialized(B):
                self.bank.init_from_hidden(content_embeds.detach(), batch_size=B)

            # Get current slots
            slots = self.bank.get()  # [B, N, D]

            # Build query from content (mean-pool)
            pool_q = content_embeds.mean(dim=1)  # [B, D]

            # Select top-k slots
            idx, scores, ste_weights = self.selector(pool_q, slots)  # idx: [B, K]

            # Gather retrieved slots
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)  # [B, K, D]
            retrieved = slots.gather(1, idx_exp)  # [B, K, D]

            # Apply STE weights for gradient flow
            gathered_weights = ste_weights.gather(1, idx)  # [B, K]
            retrieved = retrieved * gathered_weights.unsqueeze(-1)

            # Build sandwich: [retrieved | content | placeholder]
            inputs_embeds, attn_mask_4d, position_ids = self._build_sandwich(
                content_embeds, retrieved
            )

            # Forward through backbone
            hidden_states = self._forward_backbone(inputs_embeds, attn_mask_4d, position_ids)

            # Extract new memory from placeholder positions
            new_mem = hidden_states[:, -K:, :]  # [B, K, D]

            # Compute logits on content positions only
            content_h = hidden_states[:, K:K+S, :]  # [B, S, D]
            logits = self.base_model.lm_head(content_h)  # [B, S, vocab]

            all_logits.append(logits)
            if seg_labels is not None:
                all_labels.append(seg_labels)

            # Write-back via EMA
            gate = torch.sigmoid(self.gate_logit)
            self.bank.write(idx, new_mem, gate=gate)

            # BPTT truncation: detach slots if beyond bptt_depth
            if (self.config.bptt_depth != -1
                    and seg_idx < num_segments - 1 - self.config.bptt_depth):
                self.bank.detach_()

        # Compute loss over all segments
        result = {"num_segments": num_segments}

        if all_labels:
            cat_logits = torch.cat(all_logits, dim=1)  # [B, num_seg*S, vocab]
            cat_labels = torch.cat(all_labels, dim=1)  # [B, num_seg*S]

            shift_logits = cat_logits[..., :-1, :].contiguous()
            shift_labels = cat_labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
            ce_loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            result["loss"] = ce_loss
            result["ce_loss"] = ce_loss.item()

        # Aux metrics
        result["aux_metrics"] = {
            "gate_value": torch.sigmoid(self.gate_logit).item(),
            "num_segments": num_segments,
        }

        return result
