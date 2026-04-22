"""
Slot Memory Compressor for Qwen3-8B — Memory Slot Attention + Reconstruction Loss.

Why slot attention instead of cross-attention (RMT)?
  RMT's cross-attention had 0% NIH accuracy because the CE loss signal was too
  weak to teach the memory extractor what to store. Slot attention solves this
  by adding an explicit reconstruction objective: the memory slots must preserve
  enough information to reconstruct the original hidden states.

Architecture:
  1. Input hidden states → slot attention → compressed memory slots (num_slots × slot_dim)
  2. Memory slots → reconstruction MLP → reconstructed hidden states
  3. Reconstruction loss = MSE(reconstructed, original) forces slots to retain info
  4. Memory slots are injected as prefix tokens into the next segment

Key design choices:
  - Slot attention uses iterative refinement (default 3 rounds), which lets slots
    compete for "ownership" of different input positions — better than single-pass
    cross-attention for learning distinct information channels.
  - Reconstruction decoder is a lightweight MLP, not a transformer — keeps params low.
  - All slot-memory params are separate from the backbone → LoRA-friendly.

References:
  - Slot Attention (Locatello et al., 2020)
  - Slotted Transformer Memory (inspired by object-centric memory)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SlotAttention(nn.Module):
    """
    Iterative slot attention mechanism.
    
    Given input features [B, N, D], produces slot representations [B, K, slot_dim]
    where K = num_slots. Each slot attends to all input positions and is refined
    over multiple iterations via softmax-weighted averaging.
    
    This is the core mechanism that replaces cross-attention. Unlike cross-attention
    which uses learned queries, slot attention uses competitive attention: slots
    softmax-normalize across the *slot* dimension (not the input dimension), so
    different slots specialize in different parts of the input.
    """

    def __init__(
        self,
        input_dim: int,
        num_slots: int,
        slot_dim: int,
        num_iterations: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations

        # Project input to slot_dim
        self.input_proj = nn.Linear(input_dim, slot_dim)
        # Slot initialization: learned seed + positional encoding
        self.slots_seed = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        # Iterative refinement layers
        self.norm_input = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        # Q/K/V for slot→input attention
        self.q_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.k_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.v_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        # Update MLP (GRU-like refinement)
        self.update_mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = slot_dim ** -0.5

    def forward(
        self, inputs: torch.Tensor, slot_initial: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs: [B, N, input_dim] — hidden states from a segment
            slot_initial: [B, K, slot_dim] — optional previous memory slots to warm-start
        
        Returns:
            slots: [B, K, slot_dim] — refined memory slot representations
        """
        B, N, _ = inputs.shape

        # Project inputs
        inputs = self.input_proj(inputs)  # [B, N, slot_dim]
        inputs = self.norm_input(inputs)
        k = self.k_proj(inputs)  # [B, N, slot_dim]
        v = self.v_proj(inputs)  # [B, N, slot_dim]

        # Initialize slots
        if slot_initial is not None:
            slots = slot_initial  # [B, K, slot_dim]
        else:
            slots = self.slots_seed.expand(B, -1, -1)  # [B, K, slot_dim]

        # Iterative refinement
        for _ in range(self.num_iterations):
            slots = self.norm_slots(slots)
            q = self.q_proj(slots)  # [B, K, slot_dim]

            # Attention: [B, K, N] — slots attend to input positions
            # Note: softmax over SLOTS (dim=1 of query), not positions
            # This is the key difference from standard cross-attention!
            attn_logits = torch.einsum("bkd,bnd->bkn", q, k) * self.scale  # [B, K, N]
            # Normalize across slots for each position → competitive specialization
            attn = F.softmax(attn_logits, dim=1)  # [B, K, N]
            attn = self.dropout(attn)

            # Weighted sum of values
            updates = torch.einsum("bkn,bnd->bkd", attn, v)  # [B, K, slot_dim]

            # Residual update via MLP (no GRU to keep it simple and differentiable)
            slots = slots + self.update_mlp(self.norm_mlp(updates))

        return slots


class ReconstructionDecoder(nn.Module):
    """
    Lightweight decoder that reconstructs original hidden states from memory slots.
    
    This provides the reconstruction loss signal that teaches slots WHAT to store.
    Without this, slots have no supervised signal during early training and collapse.
    """

    def __init__(self, slot_dim: int, hidden_dim: int, num_slots: int, bottleneck: int = 512):
        super().__init__()
        # Expand slots → broadcast, then project to hidden_dim
        self.expand_proj = nn.Sequential(
            nn.Linear(slot_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden_dim),
        )
        self.num_slots = num_slots
        self.slot_dim = slot_dim

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: [B, K, slot_dim]
        Returns:
            reconstructed: [B, K, hidden_dim] — global reconstruction from slots
        """
        return self.expand_proj(slots)  # [B, K, hidden_dim]


class SlotMemoryCompressor(nn.Module):
    """
    Full slot memory compressor: slot attention + reconstruction + injection.

    Usage:
        compressor = SlotMemoryCompressor(hidden_dim=4096, num_slots=16, slot_dim=256)
        
        # During training:
        slots, recon_loss = compressor(segment_hidden, old_slots)
        
        # For injection into next segment:
        mem_tokens = compressor.slots_to_memory_tokens(slots)  # [B, K, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_slots: int = 16,
        slot_dim: int = 256,
        num_iterations: int = 3,
        dropout: float = 0.1,
        num_segments: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Core slot attention
        self.slot_attention = SlotAttention(
            input_dim=hidden_dim,
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_iterations=num_iterations,
            dropout=dropout,
        )

        # Reconstruction decoder
        self.recon_decoder = ReconstructionDecoder(
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
        )

        # Project slots back to hidden_dim for injection as memory tokens
        self.slot_to_hidden = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Learned initial slots for the first segment
        self.initial_slots = nn.Parameter(torch.randn(num_segments, num_slots, slot_dim) * 0.02)

    def get_initial_slots(
        self, segment_idx: int, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get initial memory slots for the first segment (or a given segment index)."""
        seg_idx = min(segment_idx, self.initial_slots.shape[0] - 1)
        return self.initial_slots[seg_idx].unsqueeze(0).expand(batch_size, -1, -1).to(
            device=device, dtype=dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        old_slots: Optional[torch.Tensor] = None,
        compute_recon: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compress segment hidden states into memory slots.

        Args:
            hidden_states: [B, T, hidden_dim] — hidden states from the current segment
            old_slots: [B, K, slot_dim] — slots from previous segment (warm-start)
            compute_recon: whether to compute reconstruction loss

        Returns:
            new_slots: [B, K, slot_dim] — updated memory slots
            recon_loss: scalar or None — reconstruction MSE loss
        """
        # Subsample hidden states if sequence is very long (reduce compute)
        # Take every Nth token + last portion for better coverage
        T = hidden_states.shape[1]
        if T > 512:
            # Keep first 256 evenly-spaced + last 256 contiguous
            indices_front = torch.linspace(0, T - 256, 256, device=hidden_states.device, dtype=torch.long)
            indices_back = torch.arange(T - 256, T, device=hidden_states.device)
            indices = torch.cat([indices_front, indices_back])
            sampled = hidden_states[:, indices, :]
        else:
            sampled = hidden_states

        # Run slot attention
        new_slots = self.slot_attention(sampled, slot_initial=old_slots)  # [B, K, slot_dim]

        # Reconstruction loss: can we recover the hidden states from the slots?
        recon_loss = None
        if compute_recon:
            # Use mean-pooled hidden state as reconstruction target
            target = hidden_states.mean(dim=1)  # [B, hidden_dim]
            # Reconstruct from each slot, then pool
            recon_per_slot = self.recon_decoder(new_slots)  # [B, K, hidden_dim]
            recon = recon_per_slot.mean(dim=1)  # [B, hidden_dim]
            recon_loss = F.mse_loss(recon, target.detach())

        return new_slots, recon_loss

    def slots_to_memory_tokens(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Convert slot representations to memory tokens for injection.
        
        Args:
            slots: [B, K, slot_dim]
        Returns:
            memory_tokens: [B, K, hidden_dim] — ready to prepend to next segment
        """
        return self.slot_to_hidden(slots)


class SlotMemoryWrapper(nn.Module):
    """
    Drop-in wrapper similar to RMTModel.
    Wraps a causal LM with slot memory compression for segment-level processing.
    """

    def __init__(
        self,
        model,  # PeftModel or AutoModelForCausalLM
        compressor: SlotMemoryCompressor,
        segment_length: int = 2048,
    ):
        super().__init__()
        self.model = model
        self.compressor = compressor
        self.segment_length = segment_length
        self.num_memory_tokens = compressor.num_slots

    def _embed_with_memory(self, input_ids, memory_tokens):
        """Prepend memory tokens as input embeddings."""
        inner = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        token_embeds = inner.get_input_embeddings()(input_ids)
        return torch.cat([memory_tokens, token_embeds], dim=1)

    def _build_attention_mask(self, seq_len, num_mem, device):
        """Memory tokens attend bidirectionally, content is causal."""
        total = num_mem + seq_len
        causal = torch.tril(torch.ones(total, total, device=device)).bool()
        causal[:num_mem, :] = True  # memory tokens see everything
        return causal

    def _build_position_ids(self, seq_len, num_mem, device):
        mem_pos = torch.arange(num_mem, device=device)
        seg_pos = torch.arange(seq_len, device=device) + num_mem
        return torch.cat([mem_pos, seg_pos])

    def forward_segment(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
        stage: str = "joint",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward a single segment with memory injection.

        Args:
            input_ids: [B, T] token IDs for this segment
            labels: [B, T] labels (same as input_ids for autoregressive)
            memory_tokens: [B, K, hidden_dim] memory from previous segment
            stage: "recon_only" | "joint" | "ce_only" — controls which losses to use

        Returns:
            ce_loss: CE loss scalar (0 if not computed)
            recon_loss: reconstruction loss scalar (None if not computed)
            segment_hidden: [B, T, hidden_dim] hidden states (for memory extraction)
        """
        B, T = input_ids.shape
        device = input_ids.device
        inner = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        backbone = inner.model
        dtype = next(self.parameters()).dtype

        if memory_tokens is None:
            memory_tokens = torch.zeros(B, self.num_memory_tokens, self.compressor.hidden_dim,
                                        device=device, dtype=dtype)

        K = memory_tokens.shape[1]

        # Build inputs
        inputs_embeds = self._embed_with_memory(input_ids, memory_tokens)
        attn_mask = self._build_attention_mask(T, K, device)
        attn_mask_4d = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        attn_mask_float = torch.zeros_like(attn_mask_4d, dtype=dtype)
        attn_mask_float = attn_mask_float.masked_fill(~attn_mask_4d, float('-inf'))

        position_ids = self._build_position_ids(T, K, device).unsqueeze(0).expand(B, -1)

        # Forward through backbone
        outputs = backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask_float,
            position_ids=position_ids,
        )
        hidden = outputs.last_hidden_state  # [B, K+T, D]
        segment_hidden = hidden[:, K:, :]  # [B, T, D]

        # CE loss
        ce_loss = torch.tensor(0.0, device=device)
        if labels is not None and stage != "recon_only":
            logits = inner.lm_head(hidden)
            mem_labels = torch.full((B, K), -100, device=device, dtype=torch.long)
            full_labels = torch.cat([mem_labels, labels], dim=1)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        # Reconstruction loss
        recon_loss = None
        if stage != "ce_only":
            _, recon_loss = self.compressor(segment_hidden, compute_recon=True)
        elif memory_tokens.requires_grad or True:
            # Always compute recon for slot update, but may not use it
            _, recon_loss = self.compressor(segment_hidden, compute_recon=True)
            if stage == "ce_only":
                recon_loss = None  # Don't use it

        return ce_loss, recon_loss, segment_hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        stage: str = "joint",
        training: bool = True,
    ) -> Tuple[torch.Tensor, float]:
        """
        Full forward pass across all segments.

        Returns:
            total_loss: combined loss
            avg_recon: average reconstruction loss (for logging)
        """
        B, L = input_ids.shape
        device = input_ids.device
        num_segments = L // self.segment_length

        total_ce = torch.tensor(0.0, device=device)
        total_recon = 0.0
        recon_count = 0
        old_slots = None

        for seg_idx in range(num_segments):
            start = seg_idx * self.segment_length
            end = start + self.segment_length
            seg_ids = input_ids[:, start:end]
            seg_labels = labels[:, start:end] if labels is not None else None

            # Get memory tokens for this segment
            if old_slots is None:
                slots = self.compressor.get_initial_slots(seg_idx, B, device, next(self.parameters()).dtype)
            else:
                slots = old_slots
            mem_tokens = self.compressor.slots_to_memory_tokens(slots)

            # Forward segment
            ce_loss, recon_loss, seg_hidden = self.forward_segment(
                seg_ids, seg_labels, mem_tokens, stage=stage
            )

            # Update slots from this segment's hidden states
            new_slots, slot_recon = self.compressor(
                seg_hidden, old_slots=old_slots,
                compute_recon=(stage != "ce_only")
            )
            old_slots = new_slots.detach()  # detach inter-segment to prevent OOM

            # Accumulate losses
            if stage == "recon_only":
                loss = slot_recon if slot_recon is not None else torch.tensor(0.0, device=device)
            elif stage == "joint":
                loss = ce_loss + 0.5 * (slot_recon if slot_recon is not None else torch.tensor(0.0, device=device))
            else:  # ce_only
                loss = ce_loss

            if training:
                loss.backward()

            total_ce = total_ce + ce_loss.detach()
            if slot_recon is not None:
                total_recon += slot_recon.item()
                recon_count += 1

        avg_recon = total_recon / max(recon_count, 1)
        return total_ce / max(num_segments, 1), avg_recon
