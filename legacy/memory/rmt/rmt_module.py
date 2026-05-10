"""
RMT Memory Module for Qwen3-8B — V8 Direct Injection.

BABILong-style: memory tokens are the last-K hidden states from the previous
segment, prepended as ordinary tokens to the next segment. No separate
extractor module, no cross-attention, no torch.no_grad(). Natural gradient
flow through memory tokens.

Design:
- Memory = last K hidden-state vectors from previous segment
- Prepended as first K positions of next segment
- Position IDs: memory tokens get 0..K-1, content gets K..K+content_len-1
- Full gradient flow (no detach/detour)
- Simple, minimal parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ─── Legacy extractors kept for backward compat (loading old checkpoints) ───

class MemoryExtractorV2(nn.Module):
    def __init__(self, hidden_dim=4096, num_memory_tokens=8, bottleneck_dim=32):
        super().__init__()
        self.num_memory_tokens = num_memory_tokens
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim), nn.SiLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim), nn.SiLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.query_coeffs = nn.Parameter(torch.randn(num_memory_tokens, 1) * 0.02)
        self.gate_bias = nn.Parameter(torch.zeros(num_memory_tokens))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states, old_memory=None):
        segment_mean = hidden_states.mean(dim=1)
        compressed = self.mlp(segment_mean)
        intermediate = self.norm(segment_mean + compressed)
        new_memory = intermediate.unsqueeze(1) * self.query_coeffs
        if old_memory is not None:
            gate = torch.sigmoid(self.gate_bias).unsqueeze(0).unsqueeze(-1)
            new_memory = gate * new_memory + (1 - gate) * old_memory
        return new_memory


class MemoryExtractorV5(nn.Module):
    def __init__(self, hidden_dim=4096, num_memory_tokens=16, bottleneck_dim=256, n_heads=4):
        super().__init__()
        self.num_memory_tokens = num_memory_tokens
        self.hidden_dim = hidden_dim
        from math import sqrt as _sqrt
        self.cross_attn_extractor = CrossAttentionExtractor(hidden_dim, num_memory_tokens, bottleneck_dim, n_heads)
        self.importance_updater = ImportanceMemoryUpdater(hidden_dim, num_memory_tokens)

    def forward(self, hidden_states, old_memory=None):
        new_memory = self.cross_attn_extractor(hidden_states)
        if old_memory is not None:
            new_memory, _ = self.importance_updater(old_memory, new_memory)
        return new_memory, None


# ─── Legacy helper classes needed by V5 ───

class CrossAttentionExtractor(nn.Module):
    def __init__(self, d_model=4096, n_memory=16, bottleneck_dim=256, n_heads=4):
        super().__init__()
        self.memory_queries = nn.Parameter(torch.randn(n_memory, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, bottleneck_dim, bias=False)
        self.k_proj = nn.Linear(d_model, bottleneck_dim, bias=False)
        self.v_proj = nn.Linear(d_model, bottleneck_dim, bias=False)
        self.out_proj = nn.Linear(bottleneck_dim, d_model, bias=False)
        self.n_heads = n_heads
        self.head_dim = bottleneck_dim // n_heads
        self.norm = nn.LayerNorm(d_model)

    def forward(self, segment_hidden_states):
        B = segment_hidden_states.size(0)
        M = self.memory_queries.size(0)
        q = self.q_proj(self.memory_queries).unsqueeze(0).expand(B, -1, -1)
        k = self.k_proj(segment_hidden_states)
        v = self.v_proj(segment_hidden_states)
        q = q.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, M, -1)
        return self.norm(self.out_proj(attn_output))


class ImportanceMemoryUpdater(nn.Module):
    def __init__(self, d_model=4096, n_memory=16):
        super().__init__()
        self.importance_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.SiLU(), nn.Linear(d_model // 4, 1),
        )
        nn.init.zeros_(self.importance_mlp[-1].weight)
        nn.init.zeros_(self.importance_mlp[-1].bias)
        self.base_gate = 0.1

    def forward(self, old_memory, new_memory):
        importance = torch.sigmoid(self.importance_mlp(old_memory).squeeze(-1))
        gate = self.base_gate + (1.0 - self.base_gate) * (1.0 - importance)
        updated = gate.unsqueeze(-1) * new_memory + (1.0 - gate.unsqueeze(-1)) * old_memory
        return updated, importance


# ─── V8: Direct Injection ──────────────────────────────────────────────────

class DirectMemoryInjector(nn.Module):
    """
    BABILong-style direct injection.
    
    Memory = last K hidden states from the previous segment.
    These are prepended as the first K tokens of the next segment.
    No learnable parameters — pure hidden-state passthrough with full gradients.
    """
    def __init__(self, hidden_dim: int = 4096, num_memory_tokens: int = 16):
        super().__init__()
        self.num_memory_tokens = num_memory_tokens
        self.hidden_dim = hidden_dim
        # Minimal learned gate: should the model trust the injected memory?
        # Sigmoid-initialized near 1.0 so memory flows through at start.
        self.memory_gate = nn.Parameter(torch.ones(num_memory_tokens) * 2.0)

    def get_memory_tokens(
        self,
        prev_hidden_states: torch.Tensor,  # [B, T, D]
        K: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract last K hidden-state vectors as memory tokens."""
        if K is None:
            K = self.num_memory_tokens
        K = min(K, prev_hidden_states.shape[1])
        # Take last K positions — natural gradient flow, no detach()
        mem = prev_hidden_states[:, -K:, :]  # [B, K, D]
        # Apply learned per-token gate
        gate = torch.sigmoid(self.memory_gate[:K]).unsqueeze(0).unsqueeze(-1)  # [1, K, 1]
        return mem * gate


# ─── Sampling helper ────────────────────────────────────────────────────────

def _sample_token(logits, temperature=1.0, top_k=0, top_p=1.0):
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cum_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float('-inf')
        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)
    if temperature <= 0 or (top_k == 0 and top_p >= 1.0):
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ─── RMTMemory (version-aware) ──────────────────────────────────────────────

class RMTMemory(nn.Module):
    """
    RMT Memory module — version-aware.
    v8: DirectMemoryInjector (BABILong-style, no separate extractor)
    v5/v2: legacy extractors for backward compat
    """
    def __init__(
        self,
        hidden_dim: int,
        num_memory_tokens: int = 16,
        num_heads: int = 8,
        max_segments: int = 8,
        bottleneck_dim: int = 32,
        extractor_version: int = 8,
        use_reconstruction: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_memory_tokens = num_memory_tokens
        self.extractor_version = extractor_version

        if extractor_version == 8:
            self.injector = DirectMemoryInjector(hidden_dim, num_memory_tokens)
            # No extractor needed for v8
        elif extractor_version == 5:
            self.extractor = MemoryExtractorV5(
                hidden_dim=hidden_dim,
                num_memory_tokens=num_memory_tokens,
                bottleneck_dim=bottleneck_dim,
            )
        else:
            self.extractor = MemoryExtractorV2(
                hidden_dim=hidden_dim,
                num_memory_tokens=num_memory_tokens,
                bottleneck_dim=bottleneck_dim,
            )

        # Learned initial memory embeddings (used for first segment only)
        self.memory_embeddings = nn.Parameter(
            torch.randn(max_segments, num_memory_tokens, hidden_dim) * 0.02
        )
        self.segment_bias = nn.Embedding(max_segments, num_memory_tokens)
        self.max_segments = max_segments

    def get_initial_memory(self, segment_idx, batch_size, device, dtype):
        mem = self.memory_embeddings[segment_idx].unsqueeze(0).expand(batch_size, -1, -1)
        bias = self.segment_bias(torch.tensor(segment_idx, device=device))
        mem = mem + bias.unsqueeze(0).unsqueeze(-1)
        return mem.to(dtype=dtype)

    def extract_memory(self, hidden_states, old_memory=None, current_K=None):
        """
        Extract memory from segment hidden states.
        
        For v8: returns last K hidden states via DirectMemoryInjector.
        current_K: override num_memory_tokens for curriculum learning.
        """
        if self.extractor_version == 8:
            return self.injector.get_memory_tokens(hidden_states, K=current_K), None
        result = self.extractor(hidden_states, old_memory)
        if isinstance(result, tuple):
            return result
        return result


# ─── Mask / position builders ───────────────────────────────────────────────

def build_rmt_attention_mask(seq_len, num_memory_tokens, device):
    """Build 2D boolean attention mask. Memory tokens attend bidirectionally."""
    total_len = num_memory_tokens + seq_len
    causal = torch.tril(torch.ones(total_len, total_len, device=device)).bool()
    causal[:num_memory_tokens, :] = True  # memory tokens: full bidirectional
    return causal


def build_rmt_position_ids(seq_len, num_memory_tokens, segment_idx, device):
    """
    Position IDs: memory tokens get 0..K-1, content gets K..K+content_len-1.
    In v8 we use local positions (no segment_idx offset) so the model sees
    a consistent positional pattern regardless of segment depth.
    """
    mem_pos = torch.arange(num_memory_tokens, device=device)
    seg_pos = torch.arange(seq_len, device=device) + num_memory_tokens
    return torch.cat([mem_pos, seg_pos])


# ─── RMTModel ───────────────────────────────────────────────────────────────

class RMTModel(nn.Module):
    """Wraps a causal LM with RMT memory injection."""
    def __init__(self, model, rmt_memory, segment_length=2048):
        super().__init__()
        self.model = model
        self.rmt = rmt_memory
        self.segment_length = segment_length
        self.num_memory_tokens = rmt_memory.num_memory_tokens

    def _embed_with_memory(self, input_ids, memory_embeddings):
        token_embeds = self.model.get_input_embeddings()(input_ids)
        return torch.cat([memory_embeddings, token_embeds], dim=1)

    def _forward_single_segment(
        self, input_ids, labels, memory_embeddings, segment_idx,
    ):
        B = input_ids.shape[0]
        device = input_ids.device
        dtype = memory_embeddings.dtype
        actual_seg_len = input_ids.shape[1]

        inputs_embeds = self._embed_with_memory(input_ids, memory_embeddings)

        attn_mask = build_rmt_attention_mask(actual_seg_len, self.num_memory_tokens, device)
        attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)
        position_ids = build_rmt_position_ids(
            actual_seg_len, self.num_memory_tokens, segment_idx, device
        ).unsqueeze(0).expand(B, -1)

        if labels is not None:
            mem_labels = torch.full((B, self.num_memory_tokens), -100, device=device, dtype=labels.dtype)
            full_labels = torch.cat([mem_labels, labels], dim=1)
        else:
            full_labels = None

        bool_mask_4d = attn_mask.unsqueeze(1)
        attn_mask_4d = torch.zeros_like(bool_mask_4d, dtype=dtype)
        attn_mask_4d = attn_mask_4d.masked_fill(~bool_mask_4d, torch.tensor(float('-inf'), dtype=dtype))

        outputs = self.model.model(
            inputs_embeds=inputs_embeds,
            attention_mask={"full_attention": attn_mask_4d},
            position_ids=position_ids,
            output_hidden_states=False,
        )

        hidden = outputs.last_hidden_state
        logits = self.model.lm_head(hidden)

        if full_labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            loss = torch.tensor(0.0, device=device)

        segment_hidden = hidden[:, self.num_memory_tokens:, :]
        return loss, segment_hidden

    def generate_with_memory(
        self, question_ids, memory_embeddings, segment_idx,
        max_new_tokens=20, temperature=1.0, top_k=0, top_p=1.0,
    ):
        B = question_ids.shape[0]
        device = question_ids.device
        dtype = memory_embeddings.dtype
        q_len = question_ids.shape[1]
        total_len = self.num_memory_tokens + q_len

        inputs_embeds = self._embed_with_memory(question_ids, memory_embeddings)

        attn_mask = build_rmt_attention_mask(q_len, self.num_memory_tokens, device)
        attn_mask_3d = attn_mask.unsqueeze(0).expand(B, -1, -1)
        bool_mask_4d = attn_mask_3d.unsqueeze(1)
        attn_mask_4d = torch.zeros_like(bool_mask_4d, dtype=dtype)
        attn_mask_4d = attn_mask_4d.masked_fill(~bool_mask_4d, float('-inf'))

        position_ids = build_rmt_position_ids(
            q_len, self.num_memory_tokens, segment_idx, device
        ).unsqueeze(0).expand(B, -1)

        outputs = self.model.model(
            inputs_embeds=inputs_embeds,
            attention_mask={"full_attention": attn_mask_4d},
            position_ids=position_ids,
            output_hidden_states=False,
            use_cache=True,
        )
        logits = self.model.lm_head(outputs.last_hidden_state)
        past_kv = outputs.past_key_values

        next_token = _sample_token(logits[:, -1, :], temperature, top_k, top_p)
        generated = [next_token]

        for step in range(max_new_tokens - 1):
            token_embeds = self.model.get_input_embeddings()(next_token)
            next_pos = position_ids[:, -1:] + 1 + step
            past_len = total_len + step
            new_attn = torch.zeros(B, 1, 1, past_len + 1, device=device, dtype=dtype)
            outputs = self.model.model(
                inputs_embeds=token_embeds,
                attention_mask={"full_attention": new_attn},
                position_ids=next_pos,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = self.model.lm_head(outputs.last_hidden_state)
            next_token = _sample_token(logits[:, -1, :], temperature, top_k, top_p)
            past_kv = outputs.past_key_values
            generated.append(next_token)

        return torch.cat(generated, dim=1)

    def forward(self, input_ids, labels=None, training=True):
        B, L = input_ids.shape
        device = input_ids.device
        num_segments = L // self.segment_length
        total_loss = torch.tensor(0.0, device=device)
        old_memory = None

        for seg_idx in range(num_segments):
            start = seg_idx * self.segment_length
            end = start + self.segment_length
            seg_ids = input_ids[:, start:end]
            seg_labels = labels[:, start:end] if labels is not None else None

            if old_memory is None:
                mem = self.rmt.get_initial_memory(seg_idx, B, device, torch.bfloat16)
            else:
                mem = old_memory

            loss, seg_hidden = self._forward_single_segment(seg_ids, seg_labels, mem, seg_idx)

            if training:
                loss.backward(retain_graph=False)
            total_loss = total_loss + loss.detach()

            mem_result = self.rmt.extract_memory(seg_hidden, old_memory)
            old_memory = mem_result[0] if isinstance(mem_result, tuple) else mem_result

        return total_loss / max(num_segments, 1)
