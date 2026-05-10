"""
RMT++ v10 — Layered Recurrent Memory Transformer.

Key fixes over v5/v8 (based on official RMT repo analysis):
1. Sandwich injection: [old_mem | content | placeholder_mem], new memory from appended position
2. No custom position offset for memory — let model's native RoPE handle it (continuous 0-based IDs)
3. Full BPTT: accumulate loss across all segments, single backward
4. vary_n_segments: curriculum learning (random 1..max_n_segments)
5. No per-segment backward

Architecture:
- L0: segment-level memory (sandwich, default ON)
- L1: layer-level memory (additive injection at mid-layer, default OFF)
- L2: global memory (additive injection at top-layer, default OFF)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RMTv10Config:
    # L0
    num_mem_tokens: int = 16
    segment_length: int = 1024
    max_n_segments: int = 6
    # L1
    use_l1: bool = False
    l1_num_tokens: int = 8
    l1_update_freq: int = 3
    l1_inject_layer: int = -1  # -1 → auto (layers // 2)
    # L2
    use_l2: bool = False
    l2_num_tokens: int = 4
    l2_update_freq: int = 6
    l2_inject_layer: int = -1  # -1 → auto (layers - 1)
    # Training
    vary_n_segments: bool = True
    bptt_depth: int = -1       # -1 = full BPTT
    recon_loss_coef: float = 0.1
    use_importance_routing: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# L0 Memory — Sandwich injection
# ═══════════════════════════════════════════════════════════════════════════

class L0Memory(nn.Module):
    """
    Segment-level memory using sandwich injection.

    Layout per segment: [old_mem (K) | content (S) | placeholder_mem (K)]
    - old_mem is prepended (read memory)
    - placeholder_mem is appended (write position)
    - New memory = hidden states at appended positions
    - Attention mask: old_mem is bidirectional, content is causal, placeholder attends to everything

    Initialization: scaled by model embedding std (from official RMT MemoryCell.create_memory).
    """

    def __init__(self, num_mem_tokens: int, hidden_dim: int,
                 use_importance_routing: bool = True):
        super().__init__()
        self.num_mem_tokens = num_mem_tokens
        self.hidden_dim = hidden_dim
        self.use_importance_routing = use_importance_routing

        # Learnable memory embedding (initialized later via init_memory)
        self.memory = nn.Parameter(torch.empty(num_mem_tokens, hidden_dim))

        # Importance router
        if use_importance_routing:
            self.importance_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, 1),
            )
            nn.init.zeros_(self.importance_mlp[-1].weight)
            nn.init.zeros_(self.importance_mlp[-1].bias)

    def init_memory(self, embed_std: float):
        """Initialize memory weights scaled by embedding std."""
        nn.init.normal_(self.memory, std=embed_std)

    def get_initial_memory(self, batch_size: int) -> torch.Tensor:
        """Return [B, K, D] initial memory."""
        return self.memory.unsqueeze(0).expand(batch_size, -1, -1)

    def build_sandwich(
        self,
        content_embeds: torch.Tensor,  # [B, S, D]
        memory_state: torch.Tensor,     # [B, K, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build sandwich: [mem | content | placeholder_mem]
        Returns (inputs_embeds [B, K+S+K, D], attention_mask [B, K+S+K])
        """
        B, S, D = content_embeds.shape
        K = self.num_mem_tokens

        # placeholder = copy of old memory (will be overwritten by model output)
        placeholder = memory_state.detach()  # [B, K, D]
        inputs_embeds = torch.cat([memory_state, content_embeds, placeholder], dim=1)

        # Attention mask: 1 = attend, 0 = mask
        #   old_mem (0:K): bidirectional — can attend to all old_mem + content + placeholder
        #   content (K:K+S): causal — can attend to old_mem + self and earlier content
        #   placeholder (K+S:K+S+K): can attend to everything before + self
        total_len = 2 * K + S
        mask = torch.zeros(B, total_len, dtype=torch.int64, device=content_embeds.device)

        # old_mem: attend to all positions (including content and placeholder)
        mask[:, :K] = 1

        # content: causal from start (attend to old_mem + earlier content)
        for i in range(S):
            mask[:, K + i] = 1
            # content[i] can attend to old_mem + content[0..i]
            # already handled by making it 1 for itself; but we need full visibility to old_mem
            # Actually, let's be explicit:
        # Simpler: use a triangular approach
        mask = torch.zeros(B, total_len, total_len, dtype=torch.bool, device=content_embeds.device)

        # old_mem (rows 0:K) → attend to old_mem + content + placeholder (all columns)
        mask[:, :K, :] = True

        # content (rows K:K+S) → causal: attend to old_mem (cols 0:K) + content up to current
        # For content row i: attend to cols 0..K+i
        for i in range(S):
            mask[:, K + i, :K + i + 1] = True

        # placeholder (rows K+S:K+S+K) → attend to everything (old_mem + all content + earlier placeholder)
        for i in range(K):
            mask[:, K + S + i, :K + S + i + 1] = True

        return inputs_embeds, mask

    def build_sandwich_fast(
        self,
        content_embeds: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized version of build_sandwich for efficiency.
        """
        B, S, D = content_embeds.shape
        K = self.num_mem_tokens

        placeholder = memory_state.detach()
        inputs_embeds = torch.cat([memory_state, content_embeds, placeholder], dim=1)
        total_len = 2 * K + S

        # Build 2D causal mask, then fix specific regions
        # Start with full causal (lower triangular)
        causal = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool,
                                       device=content_embeds.device))
        causal = causal.unsqueeze(0).expand(B, -1, -1)  # [B, T, T]

        # old_mem rows: bidirectional (already covered by causal since they're first K rows)
        # But causal only allows old_mem to see old_mem (indices 0..K-1 see 0..K-1)
        # We want old_mem to also see content and placeholder
        causal[:, :K, K:] = True  # old_mem can see everything after

        return inputs_embeds, causal

    def extract_new_memory(
        self,
        hidden_states: torch.Tensor,  # [B, K+S+K, D]
    ) -> torch.Tensor:
        """Extract memory from appended placeholder positions."""
        K = self.num_mem_tokens
        return hidden_states[:, -K:, :]

    def apply_importance_routing(
        self,
        old_memory: torch.Tensor,  # [B, K, D]
        new_memory: torch.Tensor,  # [B, K, D]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply importance-based gating. High importance → keep old, low importance → accept new.
        """
        if not self.use_importance_routing:
            return new_memory, None

        importance = torch.sigmoid(self.importance_mlp(old_memory).squeeze(-1))  # [B, K]
        gate = 1.0 - importance  # high importance → keep old
        updated = gate.unsqueeze(-1) * new_memory + (1.0 - gate.unsqueeze(-1)) * old_memory
        return updated, importance


# ═══════════════════════════════════════════════════════════════════════════
# L1 Memory — Layer-level additive injection
# ═══════════════════════════════════════════════════════════════════════════

class L1Memory(nn.Module):
    """
    Layer-level memory: compressed from L0 memory across all layers.
    Injected additively at a mid-layer. Zero-initialized (safe for additive injection).
    Updated every l1_update_freq segments.
    """

    def __init__(self, num_tokens: int, hidden_dim: int,
                 num_layers: int, update_freq: int = 3):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.update_freq = update_freq

        # Memory state (zero-init, not a learnable parameter — updated dynamically)
        self.register_buffer(
            "memory_state",
            torch.zeros(1, num_tokens, hidden_dim),
            persistent=False,
        )

        # Compressor: all layers' L0 memory → L1 memory
        # Input: [B, num_layers * K, D], Output: [B, num_tokens, D]
        self.compressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_tokens * hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def should_update(self, seg_idx: int) -> bool:
        return (seg_idx + 1) % self.update_freq == 0 and seg_idx > 0

    def update(
        self,
        l0_memory_per_layer: List[torch.Tensor],  # list of [B, K, D] per layer
        batch_size: int,
    ) -> torch.Tensor:
        """
        Compress L0 memories from all layers into L1 memory.
        l0_memory_per_layer: list of [B, K, D] tensors (one per layer).
        """
        # Stack layers: [B, num_layers * K, D]
        stacked = torch.cat(l0_memory_per_layer, dim=1)
        # Compress: [B, num_tokens * D]
        compressed = self.compressor(stacked.mean(dim=1))  # [B, num_tokens * D]
        compressed = compressed.view(-1, self.num_tokens, self.hidden_dim)
        new_mem = self.norm(compressed)
        self.memory_state = new_mem.detach()
        return new_mem

    def get_injection(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return [B, num_tokens, D] to add at injection layer."""
        return self.memory_state.expand(batch_size, -1, -1).to(device)


# ═══════════════════════════════════════════════════════════════════════════
# L2 Memory — Global additive injection
# ═══════════════════════════════════════════════════════════════════════════

class L2Memory(nn.Module):
    """
    Global memory: compressed from L1 memory.
    Injected additively at the top layer. Zero-initialized.
    Updated every l2_update_freq segments.
    """

    def __init__(self, num_tokens: int, hidden_dim: int,
                 update_freq: int = 6):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.update_freq = update_freq

        self.register_buffer(
            "memory_state",
            torch.zeros(1, num_tokens, hidden_dim),
            persistent=False,
        )

        # Compressor: L1 memory → L2 memory
        self.compressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_tokens * hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def should_update(self, seg_idx: int) -> bool:
        return (seg_idx + 1) % self.update_freq == 0 and seg_idx > 0

    def update(self, l1_memory: torch.Tensor) -> torch.Tensor:
        """Compress L1 memory into L2 memory. l1_memory: [B, l1_num_tokens, D]."""
        compressed = self.compressor(l1_memory.mean(dim=1))
        compressed = compressed.view(-1, self.num_tokens, self.hidden_dim)
        new_mem = self.norm(compressed)
        self.memory_state = new_mem.detach()
        return new_mem

    def get_injection(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.memory_state.expand(batch_size, -1, -1).to(device)


# ═══════════════════════════════════════════════════════════════════════════
# Reconstruction Head
# ═══════════════════════════════════════════════════════════════════════════

class ReconstructionHead(nn.Module):
    """
    Predicts the last token(s) of the previous segment from current L0 memory.
    Loss = CE(proj(memory), last_token_ids)
    """

    def __init__(self, hidden_dim: int, vocab_size: int, num_predict: int = 1):
        super().__init__()
        self.num_predict = num_predict
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        memory: torch.Tensor,          # [B, K, D]
        target_token_ids: torch.Tensor, # [B, num_predict]
    ) -> torch.Tensor:
        """
        Returns reconstruction loss (scalar).
        Uses mean-pooled memory projected to vocab.
        """
        pooled = memory.mean(dim=1)  # [B, D]
        logits = self.proj(pooled)   # [B, V]
        loss = F.cross_entropy(logits, target_token_ids)
        return loss


# ═══════════════════════════════════════════════════════════════════════════
# RMTv10Model — Main model wrapper
# ═══════════════════════════════════════════════════════════════════════════

class RMTv10Model(nn.Module):
    """
    Wraps a Qwen3-style causal LM with RMT++ layered memory.

    Forward pass:
    1. Split input_ids into segments
    2. For each segment: sandwich inject L0 memory, forward through model
    3. Extract new L0 memory from placeholder positions
    4. Optionally update L1/L2 memory
    5. Accumulate CE loss, optionally add reconstruction loss
    6. Single backward (full BPTT)
    """

    def __init__(self, base_model: nn.Module, config: RMTv10Config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.hidden_dim = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size

        # Determine number of transformer layers
        num_layers = base_model.config.num_hidden_layers

        # L0 memory
        self.l0 = L0Memory(
            num_mem_tokens=config.num_mem_tokens,
            hidden_dim=self.hidden_dim,
            use_importance_routing=config.use_importance_routing,
        )

        # L1 memory (optional)
        self.use_l1 = config.use_l1
        self.l1_inject_layer = -1
        if self.use_l1:
            self.l1_inject_layer = config.l1_inject_layer if config.l1_inject_layer >= 0 else num_layers // 2
            self.l1 = L1Memory(
                num_tokens=config.l1_num_tokens,
                hidden_dim=self.hidden_dim,
                num_layers=num_layers,
                update_freq=config.l1_update_freq,
            )
            self._register_layer_hook('l1', self.l1_inject_layer)
        else:
            self.l1 = None

        # L2 memory (optional)
        self.use_l2 = config.use_l2
        self.l2_inject_layer = -1
        if self.use_l2:
            self.l2_inject_layer = config.l2_inject_layer if config.l2_inject_layer >= 0 else num_layers - 1
            self.l2 = L2Memory(
                num_tokens=config.l2_num_tokens,
                hidden_dim=self.hidden_dim,
                update_freq=config.l2_update_freq,
            )
            self._register_layer_hook('l2', self.l2_inject_layer)
        else:
            self.l2 = None

        # Reconstruction head (always created)
        self.recon_head = ReconstructionHead(
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size,
        )

        # Initialize L0 memory weights
        embed_std = base_model.get_input_embeddings().weight.data.std().item()
        self.l0.init_memory(embed_std)

        # Register L1/L2 hooks (after recon_head and init are done)
        if self.use_l1:
            self._register_layer_hook('l1', self.l1_inject_layer)
        if self.use_l2:
            self._register_layer_hook('l2', self.l2_inject_layer)

    def _register_layer_hook(self, memory_level: str, layer_idx: int):
        """Register a forward hook on the specified transformer layer for additive memory injection."""
        target_layer = self.base_model.model.layers[layer_idx]

        def hook_fn(module, input, output):
            # output is hidden_states after the layer; inject memory additively
            hidden = output[0] if isinstance(output, tuple) else output
            mem_obj = self.l1 if memory_level == 'l1' else self.l2
            injection = mem_obj.get_injection(hidden.size(0), hidden.device).to(dtype=hidden.dtype)
            # Add to first num_tokens positions only (prepend-like injection)
            num_toks = injection.size(1)
            hidden[:, :num_toks, :] = hidden[:, :num_toks, :] + injection
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        target_layer.register_forward_hook(hook_fn)

    def _make_4d_attn_mask(
        self,
        mask_2d: torch.Tensor,  # [B, T, T] bool
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert 2D bool mask to 4D float mask for Qwen3."""
        mask_4d = torch.zeros(
            mask_2d.size(0), 1, mask_2d.size(1), mask_2d.size(2),
            dtype=dtype, device=mask_2d.device,
        )
        mask_4d.masked_fill_(~mask_2d.unsqueeze(1), float('-inf'))
        return mask_4d

    def _segment_input(
        self,
        input_ids: torch.Tensor,      # [B, L]
        num_segments: int,
    ) -> List[torch.Tensor]:
        """Split input_ids into num_segments chunks. Pad last if needed."""
        B, L = input_ids.shape
        seg_len = self.config.segment_length

        # Truncate or pad to fit num_segments * seg_len
        target_len = num_segments * seg_len
        if L > target_len:
            input_ids = input_ids[:, :target_len]
        elif L < target_len:
            pad_len = target_len - L
            input_ids = F.pad(input_ids, (0, pad_len), value=0)

        segments = list(input_ids.split(seg_len, dim=1))
        return segments

    def forward(
        self,
        input_ids: torch.Tensor,     # [B, L] — full long sequence
        labels: Optional[torch.Tensor] = None,  # [B, L]
    ) -> Dict[str, torch.Tensor]:
        """
        Full RMT++ forward with BPTT.

        Returns dict with 'loss' (scalar) and metadata.
        """
        B, L = input_ids.shape
        device = input_ids.device
        dtype = next(self.base_model.parameters()).dtype
        K = self.config.num_mem_tokens
        seg_len = self.config.segment_length
        cfg = self.config

        # Determine number of segments
        if cfg.vary_n_segments and self.training:
            max_segs = min(cfg.max_n_segments, L // seg_len)
            num_segments = torch.randint(1, max_segs + 1, (1,)).item()
        else:
            num_segments = min(cfg.max_n_segments, L // seg_len)
        num_segments = max(1, num_segments)

        segments = self._segment_input(input_ids, num_segments)
        label_segments = None
        if labels is not None:
            label_segments = self._segment_input(labels, num_segments)

        # Initialize
        memory_state = self.l0.get_initial_memory(B).to(dtype=dtype)
        all_logits = []
        all_labels = []
        recon_losses = []
        prev_last_tokens = None  # for reconstruction

        # L1/L2 state
        l0_memories_per_layer = None
        if self.use_l1:
            l0_memories_per_layer = []

        # Collect segment outputs for BPTT gradient management
        segment_hidden_for_l0 = []  # We'll store hidden states from each segment

        for seg_idx in range(num_segments):
            seg_ids = segments[seg_idx].to(device)  # [B, seg_len]
            seg_labels = label_segments[seg_idx].to(device) if label_segments is not None else None

            # === Reconstruction loss ===
            if prev_last_tokens is not None and cfg.recon_loss_coef > 0:
                recon_loss = self.recon_head(memory_state, prev_last_tokens)
                recon_losses.append(recon_loss)

            # Save last tokens of this segment for next iteration's reconstruction
            # (last non-pad token)
            if seg_labels is not None:
                # Use actual last token from labels (not pad)
                prev_last_tokens = seg_labels[:, -1].clone()

            # === Embed content ===
            content_embeds = self.base_model.get_input_embeddings()(seg_ids)

            # === Build sandwich ===
            inputs_embeds, attn_mask_2d = self.l0.build_sandwich_fast(content_embeds, memory_state)
            attn_mask_4d = self._make_4d_attn_mask(attn_mask_2d, dtype)

            # === Position IDs: continuous 0-based (no offset for memory) ===
            total_len = inputs_embeds.shape[1]  # 2*K + seg_len
            position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(B, -1)

            # === Labels for this segment (shift for LM) ===
            if seg_labels is not None:
                # Sandwich labels: -100 for mem positions, real labels for content, -100 for placeholder
                mem_labels = torch.full((B, K), -100, dtype=torch.long, device=device)
                placeholder_labels = torch.full((B, K), -100, dtype=torch.long, device=device)
                # Content gets actual labels
                full_labels = torch.cat([mem_labels, seg_labels, placeholder_labels], dim=1)
            else:
                full_labels = None

            # === Forward through backbone ===
            # L1/L2 injection is handled by hooks registered in __init__
            # 4D attention mask compatible with both Llama2 and Qwen3
            attn_mask_4d = self._make_4d_attn_mask(attn_mask_2d, dtype)
            outputs = self.base_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states[-1]  # [B, 2K+S, D]

            # === Extract L0 memory from placeholder positions ===
            new_memory = self.l0.extract_new_memory(hidden_states)  # [B, K, D]

            # === Importance routing ===
            memory_state, importance = self.l0.apply_importance_routing(memory_state, new_memory)

            # === L1 update ===
            if self.use_l1 and self.l1 is not None and self.l1.should_update(seg_idx):
                # Collect L0 memories from all layers
                layer_memories = [
                    outputs.hidden_states[l][:, -K:, :]
                    for l in range(len(outputs.hidden_states))
                ]
                self.l1.update(layer_memories, B)

                # L2 update (if enabled)
                if self.use_l2 and self.l2 is not None and self.l2.should_update(seg_idx):
                    l1_mem = self.l1.get_injection(B, device).to(dtype=dtype)
                    self.l2.update(l1_mem)

            # === Compute logits (only content portion) ===
            content_hidden = hidden_states[:, K:K + seg_len, :]  # [B, S, D]
            content_logits = self.base_model.lm_head(content_hidden)  # [B, S, V]
            all_logits.append(content_logits)

            if full_labels is not None:
                # Extract content labels only
                content_labels = full_labels[:, K:K + seg_len]
                all_labels.append(content_labels)

            # === BPTT gradient management ===
            # Detach memory_state to prevent gradient explosion across many segments
            # unless bptt_depth allows it (k2 from official RMT)
            if cfg.bptt_depth != -1:
                # Truncated BPTT: detach memory after bptt_depth segments
                if seg_idx >= (num_segments - 1 - cfg.bptt_depth):
                    pass  # keep gradients for last bptt_depth segments
                else:
                    memory_state = memory_state.detach()

        # === Compute total loss (single backward) ===
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)

        if all_labels:
            # Concatenate all segment logits and labels
            full_logits = torch.cat(all_logits, dim=1)      # [B, total_S, V]
            full_labels = torch.cat(all_labels, dim=1)       # [B, total_S]

            shift_logits = full_logits[..., :-1, :].contiguous()
            shift_labels = full_labels[..., 1:].contiguous()

            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = total_loss + ce_loss

        # Add reconstruction losses
        if recon_losses and cfg.recon_loss_coef > 0:
            recon_total = sum(recon_losses) / len(recon_losses)
            total_loss = total_loss + cfg.recon_loss_coef * recon_total

        return {
            "loss": total_loss,
            "num_segments": num_segments,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,  # [B, L] context
        max_new_tokens: int = 32,
    ) -> torch.Tensor:
        """Generate tokens with RMT memory support (inference mode)."""
        B = input_ids.shape[0]
        device = input_ids.device
        dtype = next(self.base_model.parameters()).dtype
        K = self.config.num_mem_tokens
        cfg = self.config

        num_segments = min(cfg.max_n_segments, input_ids.shape[1] // cfg.segment_length)
        num_segments = max(1, num_segments)

        segments = self._segment_input(input_ids, num_segments)
        memory_state = self.l0.get_initial_memory(B).to(dtype=dtype)

        # Process all context segments
        for seg_idx in range(num_segments):
            seg_ids = segments[seg_idx].to(device)
            content_embeds = self.base_model.get_input_embeddings()(seg_ids)
            inputs_embeds, attn_mask_2d = self.l0.build_sandwich_fast(content_embeds, memory_state)
            attn_mask_4d = self._make_4d_attn_mask(attn_mask_2d, dtype)
            total_len = inputs_embeds.shape[1]
            position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(B, -1)

            # 4D attention mask compatible with both Llama2 and Qwen3
            attn_mask_4d = self._make_4d_attn_mask(attn_mask_2d, dtype)
            outputs = self.base_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                output_hidden_states=True,
            )
            new_memory = self.l0.extract_new_memory(outputs.hidden_states[-1])
            memory_state, _ = self.l0.apply_importance_routing(memory_state, new_memory)

        # Use memory from last segment for generation
        # For simplicity, use the base model's generate with memory prepended
        # (a full KV-cache-aware implementation would be more efficient)
        generated = input_ids[:, -32:]  # last chunk as prompt
        # TODO: integrate with model.generate() using past_key_values
        return generated


# ═══════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════

class RMTv10Memory(nn.Module):
    """Thin wrapper for compatibility — delegates to RMTv10Model."""
    def __init__(self, config: RMTv10Config):
        super().__init__()
        self.config = config

    def wrap(self, base_model: nn.Module) -> RMTv10Model:
        return RMTv10Model(base_model, self.config)
