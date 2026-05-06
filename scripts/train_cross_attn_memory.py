#!/usr/bin/env python3
"""Scheme A: Cross-Attention Memory with zero-init for continued pretraining.

Based on train_full_finetune.py, replacing ChunkMemoryBank prepend with
independent cross-attention (CrossAttentionMemoryV2).

Key differences from train_full_finetune.py:
- Each layer has its own CrossAttentionMemoryV2 module (nn.Module, trainable)
- Read: tokens cross-attend to slots (Q=hidden, K/V=slots)
- Write: delta-rule update using attention weights from read
- out_proj zero-initialized: model starts equivalent to vanilla
- No LayerNorm, no gate on the read path
- Vanilla self-attention runs FIRST on content tokens only, then cross-attention
  output is ADDED as a residual

Memory budget: Llama3-8B with bf16 + grad_ckpt ~60-70 GiB per GPU on L20A (183 GiB).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, LlamaForCausalLM

from src.memory.mem_space.niah_dataset import NIAHIterableDataset, niah_collate_fn
from src.memory.mem_space.selector import CrossAttentionMemoryV2


def make_swa_mask(seq_len: int, window_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Create a causal sliding window attention mask.

    For position i, can attend to positions [max(0, i-window_size+1), i].
    Returns additive mask: 0 where allowed, -inf where blocked. Shape [1, 1, T, T].
    """
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    if window_size < seq_len:
        window = torch.triu(causal, diagonal=-(window_size - 1))
    else:
        window = causal
    additive = torch.where(window, torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(float('-inf'), device=device, dtype=dtype))
    return additive.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [R%(rank)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dolmino shard dataset
# --------------------------------------------------------------------------- #

class DolminoChunkDataset(Dataset):
    """Loads Dolmino shards, reshapes into fixed-size chunks,
    and groups consecutive chunks into documents.

    Each document = chunks_per_doc consecutive chunks of seq_len tokens.
    Memory banks persist within a document and reset between documents.
    """

    def __init__(
        self,
        shard_dir: str,
        num_shards: int,
        seq_len: int,
        chunks_per_doc: int = 32,
        shard_offset: int = 0,
        seed: int = 42,
    ) -> None:
        self.seq_len = seq_len
        self.chunks_per_doc = chunks_per_doc

        all_chunks = []
        total_tokens = 0
        for i in range(shard_offset, shard_offset + num_shards):
            path = os.path.join(shard_dir, f"shard_{i:04d}.npy")
            if not os.path.exists(path):
                logger.warning("Shard %s not found, skipping", path)
                continue
            data = np.fromfile(path, dtype=np.uint32)
            n_chunks = len(data) // seq_len
            if n_chunks == 0:
                continue
            chunks = data[:n_chunks * seq_len].reshape(n_chunks, seq_len).astype(np.int32)
            all_chunks.append(chunks)
            total_tokens += n_chunks * seq_len

        if all_chunks:
            self.data = np.concatenate(all_chunks, axis=0)
        else:
            self.data = np.zeros((1, seq_len), dtype=np.int32)

        self.n_docs = max(1, len(self.data) // chunks_per_doc)
        logger.info(
            "DolminoChunkDataset: %d shards, %d chunks, %d docs, %.2fB tokens",
            num_shards, len(self.data), self.n_docs, total_tokens / 1e9,
        )

    def __len__(self) -> int:
        return self.n_docs

    def __getitem__(self, idx: int):
        start = idx * self.chunks_per_doc
        end = start + self.chunks_per_doc
        chunks = self.data[start:end]
        input_ids = torch.tensor(chunks, dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


class FlatChunkDataset(Dataset):
    """Flat chunks for eval (1 chunk = 1 sample)."""

    def __init__(self, npy_path: str, seq_len: int, max_chunks: int = 500):
        data = np.load(npy_path, mmap_mode="r")
        self.data = data[:max_chunks].astype(np.int32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t = torch.tensor(self.data[idx], dtype=torch.long)[:self.seq_len]
        return {"input_ids": t, "labels": t.clone()}


def doc_collate_fn(batch):
    return batch[0]


# --------------------------------------------------------------------------- #
# NIAH curriculum helpers
# --------------------------------------------------------------------------- #

@dataclass
class CurriculumParams:
    mix_fraction: float
    max_N: int
    phase: str


def get_curriculum_params(step: int, warmup_steps: int) -> CurriculumParams:
    """Curriculum schedule for NIAH training difficulty."""
    if step < warmup_steps:
        return CurriculumParams(mix_fraction=0.0, max_N=0, phase="warmup")
    elif step < warmup_steps + 3000:
        return CurriculumParams(mix_fraction=0.20, max_N=2, phase="easy")
    elif step < warmup_steps + 8000:
        return CurriculumParams(mix_fraction=0.30, max_N=8, phase="medium")
    else:
        return CurriculumParams(mix_fraction=0.15, max_N=16, phase="full")


def contrastive_retrieval_loss(
    query_attn_logits: torch.Tensor,
    target_slot_idx: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE contrastive loss for retrieval supervision.

    Encourages the query chunk's cross-attention to attend to the slot that
    received the needle information during the write phase.

    Uses pre-softmax logits (not softmax probabilities) to avoid the Rényi
    entropy bug: applying log() then logsumexp() on softmax output is a
    double-softmax that penalises peaked attention when temperature < 1.

    Args:
        query_attn_logits: [B, n_heads, T, num_slots] — PRE-SOFTMAX logits
            from CrossAttentionMemoryV2.read(return_logits=True).
        target_slot_idx: [B] — slot index that received the highest write
            attention from the needle chunk (detached, no gradient).
        temperature: scaling for logits before softmax. Lower = sharper focus.

    Returns:
        scalar loss (mean over batch).
    """
    # Average across heads and time: [B, num_slots]
    avg_logits = query_attn_logits.mean(dim=(1, 2))  # [B, num_slots]
    avg_logits = avg_logits / temperature

    # Gather positive logits (attention to target slot)
    positive_logits = avg_logits.gather(1, target_slot_idx.unsqueeze(1))  # [B, 1]

    # InfoNCE: -log(exp(pos) / sum(exp(all))) = -pos + log(sum(exp(all)))
    loss = -positive_logits + torch.logsumexp(avg_logits, dim=1, keepdim=True)  # [B, 1]
    return loss.mean()


def forward_niah_sample(
    model: CrossAttentionMemoryModel,
    sample: dict,
    device: torch.device,
    seq_len: int,
    lambda_retrieve: float = 1.0,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.1,
):
    """Forward a multi-chunk NIAH sample: stream haystack with no_grad, grad on last chunk.

    When contrastive_weight > 0, also computes InfoNCE contrastive retrieval loss
    that supervises query->slot attention to focus on the needle-containing slot.

    Returns:
        (loss, logits, last_labels, contrastive_loss_value)
        contrastive_loss_value is 0.0 when contrastive_weight == 0.
    """
    input_ids = sample["input_ids"][0]   # shape [total_len]
    labels = sample["labels"][0]         # shape [total_len]

    # Split into chunks of seq_len
    chunks = input_ids.split(seq_len)
    label_chunks = labels.split(seq_len)

    model.reset_slots()

    # Determine which chunk contains the needle (from NIAH dataset metadata)
    niah_N_gap = sample.get("N_gap", None)
    if isinstance(niah_N_gap, (list, tuple)):
        niah_N_gap = niah_N_gap[0] if niah_N_gap else None
    # needle_chunk_pos = N_gap // 2 in the NIAH dataset (midpoint insertion)
    needle_chunk_idx = None
    if niah_N_gap is not None:
        needle_chunk_idx = int(niah_N_gap) // 2

    # Storage for contrastive loss computation
    needle_write_attn_per_layer = None  # dict: layer_idx -> [B, n_heads, T, num_slots]
    query_read_logits_per_layer = None  # dict: layer_idx -> [B, n_heads, T, num_slots] PRE-SOFTMAX

    use_contrastive = contrastive_weight > 0 and needle_chunk_idx is not None

    # Stream haystack chunks with write-path gradient enabled
    for chunk_i, chunk_ids in enumerate(chunks[:-1]):
        chunk_tensor = chunk_ids.unsqueeze(0).to(device)

        if use_contrastive and chunk_i == needle_chunk_idx:
            # Capture write attention from needle chunk.
            # In the delta-rule scheme, the same cross-attention weights are used
            # for both read and write. So "which slot received the needle's write"
            # is determined by the read attention argmax (the slot that most
            # attended to the needle tokens).
            result = model.forward_chunk(
                chunk_tensor, enable_write_grad=True,
                capture_read_attn=True,
            )
            if "read_attn_weights" in result:
                needle_write_attn_per_layer = result["read_attn_weights"]
        else:
            model.forward_chunk(chunk_tensor, enable_write_grad=True)

    # Last chunk (query): compute loss with gradient + capture read logits
    last_ids = chunks[-1].unsqueeze(0).to(device)
    last_labels = label_chunks[-1].unsqueeze(0).to(device)

    if use_contrastive:
        result = model.forward_chunk(
            last_ids, enable_write_grad=True,
            capture_read_attn=True,
        )
        if "read_attn_logits" in result:
            query_read_logits_per_layer = result["read_attn_logits"]
    else:
        result = model.forward_chunk(last_ids, enable_write_grad=True)

    logits = result["logits"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = last_labels[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    # Compute contrastive retrieval loss
    contrastive_loss_val = 0.0
    if (use_contrastive
            and needle_write_attn_per_layer is not None
            and query_read_logits_per_layer is not None):
        # Aggregate across layers: for each layer, find the target slot from
        # needle write attention (softmax), then compute InfoNCE with query
        # read logits (pre-softmax).
        contrastive_losses = []
        for layer_idx in needle_write_attn_per_layer:
            needle_attn = needle_write_attn_per_layer[layer_idx]  # [B, n_heads, T, N]
            query_logits = query_read_logits_per_layer.get(layer_idx)
            if query_logits is None:
                continue

            # Target slot: slot with highest cumulative write attention from needle chunk.
            # Average across heads and time -> [B, num_slots]
            avg_write_attn = needle_attn.mean(dim=(1, 2))  # [B, num_slots]
            target_slot_idx = avg_write_attn.argmax(dim=-1).detach()  # [B], no gradient

            # InfoNCE loss using pre-softmax logits (NOT softmax output)
            cl = contrastive_retrieval_loss(query_logits, target_slot_idx, contrastive_temperature)
            contrastive_losses.append(cl)

        if contrastive_losses:
            contrastive_loss_val = torch.stack(contrastive_losses).mean().item()
            contrastive_loss_tensor = torch.stack(contrastive_losses).mean()
            loss = loss + contrastive_weight * contrastive_loss_tensor

    return loss * lambda_retrieve, logits, last_labels, contrastive_loss_val


def check_niah_accuracy(logits, labels, tokenizer, expected_code):
    """Check if model predicted the correct code at answer positions."""
    answer_mask = (labels[0] != -100)
    if not answer_mask.any():
        return False
    pred = logits[0].argmax(dim=-1)
    ans_positions = answer_mask.nonzero(as_tuple=True)[0]
    # Causal LM: logits[i] predicts token i+1, so to predict token at
    # ans_start we read logits[ans_start - 1]
    pred_start = max(0, ans_positions[0].item() - 1)
    pred_tokens = pred[pred_start : pred_start + 5]
    pred_str = tokenizer.decode(pred_tokens.tolist(), skip_special_tokens=True)
    return expected_code in pred_str


# --------------------------------------------------------------------------- #
# CrossAttentionMemoryModel
# --------------------------------------------------------------------------- #

class CrossAttentionMemoryModel(nn.Module):
    """Full fine-tuning model with per-layer cross-attention memory.

    Architecture per decoder layer:
        1. Vanilla self-attention (content tokens only, no memory prepend)
           -> decoder_output
        2. Cross-attention read: Q=decoder_output, K/V=slots
           -> memory_output (zero-init out_proj, so initially = 0)
        3. hidden_states = decoder_output + memory_output
        4. Delta-rule write: update slots using attention weights from step 2

    This guarantees that at initialization, the model behaves identically
    to the vanilla pretrained model (memory_output = 0).
    """

    def __init__(
        self,
        base_model: LlamaForCausalLM,
        num_slots: int = 64,
        top_k: int = 8,
        full_finetune: bool = True,
        use_memory: bool = True,
        use_cross_attn_memory: bool = True,
        gradient_checkpointing: bool = True,
        cross_attn_dropout: float = 0.0,
        residual_scale: float = 0.01,
        swa_window: int = 0,
        write_lr: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.full_finetune = full_finetune
        self.use_memory = use_memory
        self.use_cross_attn_memory = use_cross_attn_memory
        self.residual_scale = residual_scale
        self._swa_window = swa_window
        self._swa_mask_cache = None

        config = base_model.config
        self.num_layers = config.num_hidden_layers
        self.d_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.d_model // self.num_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)

        self.base_model = base_model

        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

        # All params trainable for full finetune
        for p in self.base_model.parameters():
            p.requires_grad = True

        self._decoder_layers: list[nn.Module] = list(self.base_model.model.layers)

        # Per-layer cross-attention memory modules
        if use_cross_attn_memory and use_memory:
            self.cross_attn_modules = nn.ModuleList([
                CrossAttentionMemoryV2(
                    d_model=self.d_model,
                    n_heads=self.num_heads,
                    n_kv_heads=self.num_kv_heads,
                    num_slots=num_slots,
                    dropout=cross_attn_dropout,
                    write_lr=write_lr,
                )
                for _ in range(self.num_layers)
            ])

            # Memory slot state: per-layer, per-sample.
            # NOT nn.Parameters -- runtime state, detached at chunk boundaries.
            # slot_keys and slot_values are [B, num_slots, d_model]
            self.slot_keys: list[torch.Tensor | None] = [None] * self.num_layers
            self.slot_values: list[torch.Tensor | None] = [None] * self.num_layers
        else:
            self.cross_attn_modules = nn.ModuleList()

    def _get_embed_tokens(self):
        return self.base_model.model.embed_tokens

    def _get_lm_head(self):
        return self.base_model.lm_head

    def _get_norm(self):
        return self.base_model.model.norm

    def _get_rotary_emb(self):
        return self.base_model.model.rotary_emb

    def reset_slots(self) -> None:
        """Reset all memory slots for a new document."""
        for i in range(self.num_layers):
            self.slot_keys[i] = None
            self.slot_values[i] = None

    def _init_slots(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        """Initialize slot keys and values for a layer from hidden states.

        Uses strided token sampling: pick evenly spaced tokens from the
        last chunk as initial slot content. This gives diversity instead
        of copying the same pooled vector.
        """
        B, T, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Initialize from hidden states with noise
        if self.slot_keys[layer_idx] is None or self.slot_keys[layer_idx].shape[0] != B:
            # Strided token sampling for diversity
            stride = max(1, T // self.num_slots)
            indices = torch.arange(0, T, stride)[:self.num_slots]
            if len(indices) < self.num_slots:
                # Pad with copies of the last selected token
                pad_indices = indices[-1:].expand(self.num_slots - len(indices))
                indices = torch.cat([indices, pad_indices])

            # Sample tokens: [B, num_slots, D]
            sampled = hidden_states[:, indices, :].detach()

            # Add small noise for diversity
            noise = torch.randn_like(sampled) * 0.02
            self.slot_keys[layer_idx] = (sampled + noise).clone()
            self.slot_values[layer_idx] = sampled.clone()

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        enable_write_grad: bool = False,
        capture_read_attn: bool = False,
    ) -> dict:
        B, T = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        if not self.use_memory or not self.use_cross_attn_memory:
            return self._forward_no_memory(input_ids, labels)

        # ---- Forward with cross-attention memory ----
        embed_tokens = self._get_embed_tokens()
        hidden_states = embed_tokens(input_ids).to(dtype)

        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        rotary_emb = self._get_rotary_emb()
        position_embeddings = rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self._decoder_layers):
            # Initialize slots on first chunk
            self._init_slots(layer_idx, hidden_states)

            slot_keys = self.slot_keys[layer_idx]    # [B, num_slots, d_model]
            slot_values = self.slot_values[layer_idx]  # [B, num_slots, d_model]

            # Step 1: Vanilla self-attention (content tokens only)
            if self._swa_window > 0:
                if self._swa_mask_cache is None:
                    self._swa_mask_cache = make_swa_mask(T, self._swa_window, hidden_states.dtype, hidden_states.device)
                attn_mask = self._swa_mask_cache
            else:
                attn_mask = None

            layer_out = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            decoder_output = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            # Step 2: Cross-attention read
            cross_attn = self.cross_attn_modules[layer_idx]
            read_result = cross_attn.read(
                decoder_output, slot_keys, slot_values,
                return_logits=capture_read_attn,
            )
            if capture_read_attn:
                memory_output, attn_weights, attn_logits = read_result
            else:
                memory_output, attn_weights = read_result

            # Capture pre-softmax logits and post-softmax weights for contrastive loss
            # (detached to avoid memory leaks)
            if capture_read_attn:
                if not hasattr(self, '_captured_read_logits') or self._captured_read_logits is None:
                    self._captured_read_logits = {}
                if not hasattr(self, '_captured_read_attn') or self._captured_read_attn is None:
                    self._captured_read_attn = {}
                self._captured_read_logits[layer_idx] = attn_logits.detach()
                self._captured_read_attn[layer_idx] = attn_weights.detach()

            # Step 3: Add memory output as residual (scaled to prevent early divergence)
            hidden_states = decoder_output + self.residual_scale * memory_output

            # Step 4: Delta-rule write
            if enable_write_grad:
                new_slot_values = cross_attn.write(
                    decoder_output,
                    slot_values,
                    attn_weights,
                )
            else:
                with torch.no_grad():
                    new_slot_values = cross_attn.write(
                        decoder_output.detach(),
                        slot_values,
                        attn_weights.detach(),
                    )
            # Clamp slot value norms to prevent feedback-loop blowup
            # Use .detach() on scale only to preserve gradient through new_slot_values
            slot_norms = new_slot_values.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            max_norm = 10.0
            scale = torch.where(slot_norms > max_norm, max_norm / slot_norms, torch.ones_like(slot_norms))
            new_slot_values = new_slot_values * scale.detach()
            self.slot_values[layer_idx] = new_slot_values

        norm = self._get_norm()
        lm_head = self._get_lm_head()
        hidden_states = norm(hidden_states)
        logits = lm_head(hidden_states)

        result = {"logits": logits}
        # Include captured read attention weights and logits if requested
        if capture_read_attn and hasattr(self, '_captured_read_attn') and self._captured_read_attn:
            result["read_attn_weights"] = self._captured_read_attn
            self._captured_read_attn = None  # clear to avoid memory leaks
        if capture_read_attn and hasattr(self, '_captured_read_logits') and self._captured_read_logits:
            result["read_attn_logits"] = self._captured_read_logits
            self._captured_read_logits = None  # clear to avoid memory leaks
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

    def _forward_no_memory(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> dict:
        """Standard transformer forward pass -- no memory interference.

        When swa_window > 0, applies sliding window mask to self-attention.
        Otherwise delegates to base_model for full causal attention.
        """
        if self._swa_window <= 0:
            outputs = self.base_model(input_ids=input_ids, labels=labels)
            result = {"logits": outputs.logits}
            if outputs.loss is not None:
                result["loss"] = outputs.loss
            return result

        # SWA path: manual layer-by-layer forward with sliding window mask
        B, T = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        embed_tokens = self._get_embed_tokens()
        hidden_states = embed_tokens(input_ids).to(dtype)
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        rotary_emb = self._get_rotary_emb()
        position_embeddings = rotary_emb(hidden_states, position_ids)

        if self._swa_mask_cache is None:
            self._swa_mask_cache = make_swa_mask(T, self._swa_window, dtype, device)
        attn_mask = self._swa_mask_cache

        for layer in self._decoder_layers:
            layer_out = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        norm = self._get_norm()
        lm_head = self._get_lm_head()
        hidden_states = norm(hidden_states)
        logits = lm_head(hidden_states)

        result = {"logits": logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            result["loss"] = loss
        return result

    def forward(self, input_ids, labels=None, **kwargs):
        return self.forward_chunk(input_ids, labels=labels)

    @torch.no_grad()
    def forward_vanilla(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Forward WITHOUT any memory interference -- vanilla model."""
        result = self.base_model(input_ids=input_ids, labels=labels)
        return {"logits": result.logits, "loss": result.loss}


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #

def init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


# --------------------------------------------------------------------------- #
# Eval helpers
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_vanilla_ppl(model, loader, device, world_size):
    """Compute PPL WITHOUT memory (vanilla forward)."""
    model.eval()
    root = model.module if hasattr(model, "module") else model
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        out = root.forward_vanilla(input_ids, labels=labels)
        loss = out["loss"].detach()
        if not torch.isfinite(loss):
            continue
        n_tok = (labels != -100).sum()
        total_loss += loss.double() * n_tok.double()
        total_tokens += n_tok.double()

    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    tot = int(total_tokens.item())
    if tot == 0:
        return float("inf"), 0
    avg_loss = (total_loss / total_tokens).item()
    return math.exp(avg_loss), tot


@torch.no_grad()
def evaluate_memory_ppl(model, loader, device, world_size):
    """Compute PPL WITH cross-attention memory. Processes chunks sequentially."""
    model.eval()
    root = model.module if hasattr(model, "module") else model

    if not root.use_memory or not root.use_cross_attn_memory:
        return evaluate_vanilla_ppl(model, loader, device, world_size)

    root.reset_slots()

    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        for i in range(input_ids.shape[0]):
            ids = input_ids[i:i+1]
            lab = labels[i:i+1]
            out = root.forward_chunk(ids, labels=lab)
            loss = out["loss"].detach()
            if torch.isfinite(loss):
                n_tok = (lab != -100).sum()
                total_loss += loss.double() * n_tok.double()
                total_tokens += n_tok.double()

    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    tot = int(total_tokens.item())
    if tot == 0:
        return float("inf"), 0
    avg_loss = (total_loss / total_tokens).item()
    return math.exp(avg_loss), tot


# --------------------------------------------------------------------------- #
# Cosine LR schedule with warmup
# --------------------------------------------------------------------------- #

def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scheme A: Cross-Attention Memory with zero-init for continued pretraining"
    )
    # Model
    p.add_argument("--model", type=str, required=True, help="Path to Llama-3-8B weights")
    # Data
    p.add_argument("--shard_dir", type=str, required=True, help="Dir with Dolmino shard_XXXX.npy files")
    p.add_argument("--num_shards", type=int, default=100, help="Number of Dolmino shards to use")
    p.add_argument("--shard_offset", type=int, default=0, help="Start shard index")
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--chunks_per_doc", type=int, default=32)
    # Eval data
    p.add_argument("--wikitext_path", type=str, default=None,
                   help="Path to wikitext_chunks_llama3_4096.npy for OOD eval")
    p.add_argument("--eval_shards", type=int, default=5,
                   help="Number of held-out Dolmino shards for ID eval")
    # Memory bank
    p.add_argument("--num_slots", type=int, default=64)
    p.add_argument("--top_k", type=int, default=8)  # kept for compat, not used in cross-attn
    # Cross-attention memory
    p.add_argument("--use_cross_attn_memory", action="store_true", default=True,
                   help="Enable cross-attention memory (Scheme A, default: True)")
    p.add_argument("--no_cross_attn_memory", action="store_true", default=False,
                   help="Disable cross-attention memory for baseline")
    p.add_argument("--cross_attn_dropout", type=float, default=0.0,
                   help="Dropout for cross-attention")
    p.add_argument("--residual_scale", type=float, default=0.01,
                   help="Scale factor for cross-attn output before adding as residual (default 0.01)")
    p.add_argument("--write_lr", type=float, default=0.1,
                   help="Delta-rule write learning rate (default 0.1). "
                        "write_lr=1.0 breaks gradient chain; 0.1 preserves ~3.4%% gradient over 32 chunks")
    p.add_argument("--swa_window", type=int, default=0,
                   help="Sliding window attention size. 0=full causal attention (default)")
    p.add_argument("--cross_attn_lr_factor", type=float, default=100,
                   help="Divide main lr by this factor to get cross-attn lr (default 100, use 1 for same lr)")
    # Mode
    p.add_argument("--full_finetune", action="store_true", default=True,
                   help="Full fine-tuning (all params trainable)")
    p.add_argument("--use_memory", action="store_true", default=True,
                   help="Enable memory (default: True)")
    p.add_argument("--no_memory", action="store_true", default=False,
                   help="Disable memory for clean baseline")
    # Gradient checkpointing
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_true", default=False)
    # Training
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # Eval / save
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--output_dir", type=str, default="outputs/cross_attn_memory")
    p.add_argument("--resume_checkpoint", type=str, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    # NIAH retrieval training
    p.add_argument("--niah_data", type=str, default=None,
                   help="Path to pg19 tokenized chunks .npy for NIAH haystack")
    p.add_argument("--niah_mix_fraction", type=float, default=0.0,
                   help="Fraction of training steps that use NIAH samples (0.0=off)")
    p.add_argument("--niah_max_N", type=int, default=16,
                   help="Max chunks between needle and query in NIAH samples")
    p.add_argument("--lambda_retrieve", type=float, default=1.0,
                   help="Weight multiplier for NIAH retrieval loss")
    p.add_argument("--niah_warmup_steps", type=int, default=2000,
                   help="Steps of pure LM warmup before NIAH training begins")
    # Contrastive retrieval loss
    p.add_argument("--contrastive_weight", type=float, default=0.0,
                   help="Weight for contrastive retrieval (InfoNCE) loss during NIAH training. "
                        "0.0 = disabled (default). When > 0, supervises query->slot attention "
                        "to attend to the slot that received the needle information.")
    p.add_argument("--contrastive_temperature", type=float, default=0.1,
                   help="Temperature for InfoNCE contrastive loss (default 0.1). "
                        "Lower = sharper focus on target slot.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve mode flags
    if args.no_memory:
        args.use_memory = False
    if args.no_cross_attn_memory:
        args.use_cross_attn_memory = False
    if args.no_gradient_checkpointing:
        args.gradient_checkpointing = False

    rank, world_size, local_rank = init_distributed()
    is_main = rank == 0

    for handler in logging.root.handlers:
        handler.setFormatter(
            logging.Formatter(f"%(asctime)s [R{rank}] %(levelname)s %(message)s")
        )

    device = torch.device(f"cuda:{local_rank}")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    if is_main:
        logger.info("=" * 60)
        mem_str = "CROSS_ATTN_MEMORY" if args.use_cross_attn_memory else "NO_MEMORY"
        logger.info("Scheme A: Cross-Attention Memory / %s", mem_str)
        logger.info("=" * 60)
        logger.info("Args: %s", vars(args))

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if is_main:
        logger.info("Loading base model from %s ...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": device},
    )

    base_model_for_baseline = base_model

    if is_main:
        logger.info(
            "Building CrossAttentionMemoryModel: use_cross_attn=%s, use_memory=%s, grad_ckpt=%s",
            args.use_cross_attn_memory, args.use_memory, args.gradient_checkpointing,
        )

    cm_model = CrossAttentionMemoryModel(
        base_model=base_model,
        num_slots=args.num_slots,
        top_k=args.top_k,
        full_finetune=args.full_finetune,
        use_memory=args.use_memory,
        use_cross_attn_memory=args.use_cross_attn_memory,
        gradient_checkpointing=args.gradient_checkpointing,
        cross_attn_dropout=args.cross_attn_dropout,
        residual_scale=args.residual_scale,
        swa_window=args.swa_window,
        write_lr=args.write_lr,
    ).to(device).to(dtype)

    trainable = sum(p.numel() for p in cm_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in cm_model.parameters())
    if is_main:
        logger.info(
            "Trainable params: %d / %d (%.4f%%)",
            trainable, total, 100.0 * trainable / total,
        )
        logger.info(
            "Config: num_slots=%d, seq_len=%d, chunks_per_doc=%d, write_lr=%.4f",
            args.num_slots, args.seq_len, args.chunks_per_doc, args.write_lr,
        )
        if args.swa_window > 0:
            logger.info("SWA enabled: window_size=%d", args.swa_window)
        # Verify zero-init on out_proj
        for i, ca in enumerate(cm_model.cross_attn_modules):
            w_norm = ca.out_proj.weight.norm().item()
            b_norm = ca.out_proj.bias.norm().item()
            logger.info(
                "Layer %d: out_proj weight_norm=%.6f, bias_norm=%.6f (should be 0)",
                i, w_norm, b_norm,
            )
            break  # only log first layer

    ddp_model = DDP(cm_model, device_ids=[local_rank])
    root_model = ddp_model.module

    # Resume
    start_step = 0
    if args.resume_checkpoint:
        if is_main:
            logger.info("Resuming from checkpoint: %s", args.resume_checkpoint)
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        cm_model.load_state_dict(ckpt['model_state_dict'])
        start_step = ckpt.get('global_step', 0)
        if is_main:
            logger.info("Resumed from step %d", start_step)

    # ------------------------------------------------------------------
    # Optimizer — separate lr for cross-attention params
    # ------------------------------------------------------------------
    base_decay = []
    base_no_decay = []
    cross_decay = []
    cross_no_decay = []
    for name, param in cm_model.named_parameters():
        if not param.requires_grad:
            continue
        is_cross = 'cross_attn_modules' in name
        bucket = (cross_decay if is_cross else base_decay) if param.dim() >= 2 and 'norm' not in name.lower() and 'bias' not in name.lower() else (cross_no_decay if is_cross else base_no_decay)
        bucket.append(param)
    cross_lr = args.lr / args.cross_attn_lr_factor  # smaller lr for new cross-attn params
    optimizer_groups = [
        {"params": base_decay, "weight_decay": args.weight_decay, "lr": args.lr},
        {"params": base_no_decay, "weight_decay": 0.0, "lr": args.lr},
        {"params": cross_decay, "weight_decay": args.weight_decay, "lr": cross_lr},
        {"params": cross_no_decay, "weight_decay": 0.0, "lr": cross_lr},
    ]
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    if is_main:
        n_base = sum(p.numel() for p in base_decay) + sum(p.numel() for p in base_no_decay)
        n_cross = sum(p.numel() for p in cross_decay) + sum(p.numel() for p in cross_no_decay)
        logger.info(
            "Optimizer: AdamW, base_lr=%.2e, cross_attn_lr=%.2e (1/%g), base=%d, cross=%d, residual_scale=%g",
            args.lr, cross_lr, args.cross_attn_lr_factor, n_base, n_cross, args.residual_scale,
        )

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    if is_main:
        logger.info("Loading Dolmino data: %d shards from offset %d ...", args.num_shards, args.shard_offset)

    train_ds = DolminoChunkDataset(
        shard_dir=args.shard_dir,
        num_shards=args.num_shards,
        seq_len=args.seq_len,
        chunks_per_doc=args.chunks_per_doc,
        shard_offset=args.shard_offset,
    )
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        sampler=train_sampler,
        collate_fn=doc_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Eval dataset: flat chunks from held-out Dolmino shards
    eval_shard_offset = args.shard_offset + args.num_shards
    eval_flat_data = []
    for si in range(eval_shard_offset, eval_shard_offset + args.eval_shards):
        epath = os.path.join(args.shard_dir, f"shard_{si:04d}.npy")
        if os.path.exists(epath):
            edata = np.fromfile(epath, dtype=np.uint32)
            en = len(edata) // args.seq_len
            eval_flat_data.append(edata[:en * args.seq_len].reshape(en, args.seq_len).astype(np.int32))
    if eval_flat_data:
        eval_chunks = np.concatenate(eval_flat_data, axis=0)[:2000]
    else:
        eval_chunks = np.zeros((100, args.seq_len), dtype=np.int32)
    if is_main:
        logger.info("Eval chunks: %d from %d held-out shards", len(eval_chunks), args.eval_shards)

    class DolminoFlatEval(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            t = torch.tensor(self.data[idx], dtype=torch.long)
            return {"input_ids": t, "labels": t.clone()}

    eval_ds = DolminoFlatEval(eval_chunks)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=4,
        sampler=DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=False),
        num_workers=0,
        collate_fn=lambda b: {
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "labels": torch.stack([x["labels"] for x in b]),
        },
    )

    # WikiText OOD eval
    wikitext_loader = None
    if args.wikitext_path and os.path.exists(args.wikitext_path):
        wiki_ds = FlatChunkDataset(args.wikitext_path, args.seq_len, max_chunks=500)
        wikitext_loader = DataLoader(
            wiki_ds,
            batch_size=4,
            sampler=DistributedSampler(wiki_ds, num_replicas=world_size, rank=rank, shuffle=False),
            num_workers=0,
            collate_fn=lambda b: {
                "input_ids": torch.stack([x["input_ids"] for x in b]),
                "labels": torch.stack([x["labels"] for x in b]),
            },
        )
        if is_main:
            logger.info("WikiText OOD eval: %d chunks", len(wiki_ds))

    # ------------------------------------------------------------------
    # NIAH data loader (optional, for retrieval training)
    # ------------------------------------------------------------------
    _niah_iter = None
    _mix_rng = None
    niah_loader = None

    if args.niah_data and args.niah_mix_fraction > 0:
        pg19_data_niah = np.load(args.niah_data, mmap_mode="r")
        niah_ds = NIAHIterableDataset(
            pg19_data_niah,
            chunk_size=args.seq_len,
            niah_mix_fraction=1.0,       # NIAH-only; Dolmino comes from train_loader
            niah_max_N=args.niah_max_N,
            tokenizer=tokenizer,
            seed=42 + rank,              # de-correlate across DDP ranks
        )
        niah_loader = DataLoader(
            niah_ds, batch_size=1, num_workers=2,
            collate_fn=niah_collate_fn,
        )
        _niah_iter = iter(niah_loader)
        _mix_rng = random.Random(42)  # same seed across all ranks for consistent NIAH/Dolmino decisions
        if is_main:
            logger.info(
                "NIAH retrieval training enabled: niah_data=%s, mix_fraction=%.2f, "
                "max_N=%d, lambda_retrieve=%.2f, warmup_steps=%d",
                args.niah_data, args.niah_mix_fraction, args.niah_max_N,
                args.lambda_retrieve, args.niah_warmup_steps,
            )

    if is_main:
        logger.info("Data loaded. Train docs: %d, Eval chunks: %d", len(train_ds), len(eval_ds))

    # ------------------------------------------------------------------
    # Measure baseline PPLs before training
    # ------------------------------------------------------------------
    if is_main:
        logger.info("Computing baseline PPLs ...")

    base_vanilla_ppl, base_vanilla_tok = evaluate_vanilla_ppl(
        ddp_model, eval_loader, device, world_size,
    )
    if is_main:
        logger.info("[BASELINE] vanilla PPL: %.4f (%d tokens)", base_vanilla_ppl, base_vanilla_tok)

    init_memory_ppl, init_memory_tok = evaluate_memory_ppl(ddp_model, eval_loader, device, world_size)
    if is_main:
        if args.use_cross_attn_memory:
            logger.info("[BASELINE] cross-attn memory PPL: %.4f (%d tokens)", init_memory_ppl, init_memory_tok)
            # At init, memory PPL should equal vanilla PPL (out_proj = 0)
            if abs(init_memory_ppl - base_vanilla_ppl) / base_vanilla_ppl > 0.01:
                logger.warning(
                    "INIT CHECK: memory PPL (%.4f) differs from vanilla (%.4f) by >1%%! "
                    "Zero-init may not be working correctly.",
                    init_memory_ppl, base_vanilla_ppl,
                )
            else:
                logger.info("INIT CHECK: memory PPL matches vanilla (zero-init confirmed)")

    base_wiki_ppl = None
    if wikitext_loader:
        base_wiki_ppl, _ = evaluate_vanilla_ppl(ddp_model, wikitext_loader, device, world_size)
        if is_main:
            logger.info("[BASELINE] WikiText PPL: %.4f", base_wiki_ppl)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(
            "Starting training: %d steps, lr=%.2e, warmup=%d, grad_accum=%d",
            args.max_steps, args.lr, args.warmup_steps, args.gradient_accumulation_steps,
        )

    global_step = start_step
    epoch = 0
    best_vanilla_ppl = base_vanilla_ppl
    t0 = time.time()
    metrics_history = []

    trainable_params = [p for p in ddp_model.parameters() if p.requires_grad]

    # NIAH tracking counters
    niah_correct = 0
    niah_total = 0
    niah_loss_sum = 0.0
    lm_loss_sum = 0.0
    contrastive_loss_sum = 0.0
    prev_curriculum_phase = None

    while global_step < args.max_steps:
        train_sampler.set_epoch(epoch)
        ddp_model.train()

        for doc_idx, sample in enumerate(train_loader):
            if global_step >= args.max_steps:
                break

            curriculum = get_curriculum_params(global_step, args.niah_warmup_steps)

            # Log curriculum phase changes
            if is_main and prev_curriculum_phase != curriculum.phase:
                logger.info(
                    "[step %d] Curriculum phase changed: %s -> %s (mix=%.2f, max_N=%d)",
                    global_step, prev_curriculum_phase, curriculum.phase,
                    curriculum.mix_fraction, curriculum.max_N,
                )
                prev_curriculum_phase = curriculum.phase

            # Decide: NIAH or Dolmino?
            is_niah = False
            doc_loss = 0.0
            doc_tokens = 0
            chunk_ppls = []

            if (_niah_iter is not None and curriculum.mix_fraction > 0
                    and _mix_rng.random() < curriculum.mix_fraction):
                is_niah = True

            if is_niah:
                # --- NIAH streaming training ---
                try:
                    niah_sample = next(_niah_iter)
                except StopIteration:
                    _niah_iter = iter(niah_loader)
                    niah_sample = next(_niah_iter)

                niah_loss, niah_logits, niah_labels, contrastive_loss_val = forward_niah_sample(
                    root_model, niah_sample, device, args.seq_len, args.lambda_retrieve,
                    contrastive_weight=args.contrastive_weight,
                    contrastive_temperature=args.contrastive_temperature,
                )

                if not torch.isfinite(niah_loss):
                    if is_main:
                        logger.warning("[step %d NIAH] Non-finite loss!", global_step)
                    # Fall through to skip this step entirely
                    continue

                # Track accuracy
                code = niah_sample.get("code", "")
                if isinstance(code, (list, tuple)):
                    code = code[0]
                if check_niah_accuracy(niah_logits, niah_labels, tokenizer, code):
                    niah_correct += 1
                niah_total += 1
                niah_loss_sum += niah_loss.item()
                contrastive_loss_sum += contrastive_loss_val

                # Scale NIAH gradient to compensate for having fewer backward calls than Dolmino
                niah_loss_scaled = niah_loss * args.chunks_per_doc
                (niah_loss_scaled / args.gradient_accumulation_steps).backward()

                # Manual gradient sync since root_model bypasses DDP hooks
                if world_size > 1:
                    for p in trainable_params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            else:
                # --- Standard Dolmino training (unchanged path) ---
                input_ids = sample["input_ids"]  # [chunks_per_doc, seq_len]
                labels = sample["labels"]

                if args.use_memory:
                    root_model.reset_slots()

                n_chunks = input_ids.shape[0]

                for chunk_i in range(n_chunks):
                    chunk_ids = input_ids[chunk_i].unsqueeze(0).to(device)
                    chunk_labels = labels[chunk_i].unsqueeze(0).to(device)

                    result = ddp_model(input_ids=chunk_ids, labels=chunk_labels)
                    loss = result["loss"]

                    if not torch.isfinite(loss):
                        if is_main:
                            logger.warning("[step %d doc %d chunk %d] Non-finite loss!", global_step, doc_idx, chunk_i)
                        continue

                    n_tok = (chunk_labels != -100).sum().item()
                    chunk_ppl = math.exp(min(loss.item(), 20))
                    chunk_ppls.append(chunk_ppl)
                    doc_loss += loss.item() * n_tok
                    doc_tokens += n_tok

                    if ddp_model.training:
                        (loss / args.gradient_accumulation_steps).backward()

                if doc_tokens == 0 or not ddp_model.training:
                    continue

                lm_loss_sum += doc_loss / doc_tokens  # track for logging

            # --- Optimizer step (shared by both NIAH and Dolmino) ---
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            new_lr = get_lr(global_step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
            new_cross_lr = new_lr / args.cross_attn_lr_factor
            # Param groups: 0=base_decay, 1=base_no_decay, 2=cross_decay, 3=cross_no_decay
            for pg_idx, pg in enumerate(optimizer.param_groups):
                pg['lr'] = new_cross_lr if pg_idx >= 2 else new_lr

            doc_ppl = math.exp(min(doc_loss / max(doc_tokens, 1), 20)) if not is_niah and doc_tokens > 0 else 0.0

            if is_main and (global_step % 10 == 0 or global_step <= 5):
                elapsed = time.time() - t0
                # Log cross-attn out_proj norm growth (key diagnostic)
                out_proj_norm = root_model.cross_attn_modules[0].out_proj.weight.norm().item() if len(root_model.cross_attn_modules) > 0 else 0.0
                niah_acc_str = f" niah_acc={niah_correct}/{niah_total}" if niah_total > 0 else ""
                niah_loss_str = f" niah_loss={niah_loss_sum / max(niah_total, 1):.4f}" if niah_total > 0 else ""
                contrastive_str = f" contrastive={contrastive_loss_sum / max(niah_total, 1):.4f}" if niah_total > 0 and args.contrastive_weight > 0 else ""
                logger.info(
                    "[step %d/%d] lr=%.2e doc_ppl=%.4f chunks=[%s] out_proj_norm=%.6f phase=%s%s%s%s %.1fs",
                    global_step, args.max_steps, new_lr, doc_ppl,
                    ",".join(f"{p:.2f}" for p in chunk_ppls[:4]) if not is_niah else "niah",
                    out_proj_norm, curriculum.phase,
                    niah_acc_str, niah_loss_str, contrastive_str, elapsed,
                )

            # Periodic eval
            if global_step % args.eval_interval == 0:
                ddp_model.eval()

                vanilla_ppl, vanilla_tok = evaluate_vanilla_ppl(
                    ddp_model, eval_loader, device, world_size,
                )

                if args.use_cross_attn_memory:
                    memory_ppl, memory_tok = evaluate_memory_ppl(
                        ddp_model, eval_loader, device, world_size,
                    )
                else:
                    memory_ppl = vanilla_ppl

                wiki_ppl = None
                if wikitext_loader:
                    wiki_ppl, _ = evaluate_vanilla_ppl(
                        ddp_model, wikitext_loader, device, world_size,
                    )

                if is_main:
                    ratio = memory_ppl / max(vanilla_ppl, 1e-8)
                    out_proj_norm = root_model.cross_attn_modules[0].out_proj.weight.norm().item() if len(root_model.cross_attn_modules) > 0 else 0.0
                    niah_acc_str = f" niah_acc={niah_correct}/{niah_total}" if niah_total > 0 else ""
                    niah_loss_str = f" niah_loss={niah_loss_sum / max(niah_total, 1):.4f}" if niah_total > 0 else ""
                    contrastive_str = f" contrastive={contrastive_loss_sum / max(niah_total, 1):.4f}" if niah_total > 0 and args.contrastive_weight > 0 else ""
                    logger.info(
                        "[EVAL step=%d] vanilla_ppl=%.4f memory_ppl=%.4f ratio=%.4f | "
                        "base_vanilla=%.4f | wiki_ood=%s | out_proj_norm=%.6f phase=%s%s%s%s",
                        global_step, vanilla_ppl, memory_ppl, ratio,
                        base_vanilla_ppl,
                        f"{wiki_ppl:.4f}" if wiki_ppl else "N/A",
                        out_proj_norm, curriculum.phase,
                        niah_acc_str, niah_loss_str, contrastive_str,
                    )

                    # Abort criteria for contrastive experiment:
                    # - If NIAH accuracy still 0% at step 1000 and niah_loss > 5.5 -> experiment failed
                    # - If PPL ratio > 1.05 -> contrastive loss hurting LM quality
                    if args.contrastive_weight > 0 and niah_total > 0:
                        avg_niah_loss = niah_loss_sum / niah_total
                        niah_acc = niah_correct / niah_total
                        if global_step >= 1000 and niah_acc == 0.0 and avg_niah_loss > 5.5:
                            logger.warning(
                                "ABORT CRITERIA: step=%d, niah_acc=0%%, niah_loss=%.4f > 5.5. "
                                "Contrastive retrieval experiment failed.",
                                global_step, avg_niah_loss,
                            )
                        if ratio > 1.05:
                            logger.warning(
                                "ABORT CRITERIA: PPL ratio=%.4f > 1.05. Contrastive loss hurting LM quality.",
                                ratio,
                            )

                    if vanilla_ppl > base_vanilla_ppl * 1.1:
                        logger.warning(
                            "MONITORING: vanilla_ppl=%.4f > base*1.1=%.4f -- forgetting!",
                            vanilla_ppl, base_vanilla_ppl * 1.1,
                        )

                    metrics = {
                        "step": global_step,
                        "vanilla_ppl": vanilla_ppl,
                        "memory_ppl": memory_ppl,
                        "memory_ratio": ratio,
                        "wiki_ppl": wiki_ppl,
                        "base_vanilla_ppl": base_vanilla_ppl,
                        "lr": new_lr,
                        "train_ppl": doc_ppl,
                        "out_proj_norm": out_proj_norm,
                        "elapsed_s": time.time() - t0,
                        "use_cross_attn_memory": args.use_cross_attn_memory,
                        "niah_correct": niah_correct,
                        "niah_total": niah_total,
                        "niah_avg_loss": niah_loss_sum / max(niah_total, 1) if niah_total > 0 else 0.0,
                        "contrastive_avg_loss": contrastive_loss_sum / max(niah_total, 1) if niah_total > 0 else 0.0,
                        "contrastive_weight": args.contrastive_weight,
                        "curriculum_phase": curriculum.phase,
                    }
                    metrics_history.append(metrics)
                    with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as f:
                        f.write(json.dumps(metrics) + "\n")

                ddp_model.train()

            # Save checkpoint
            if is_main and global_step % args.save_interval == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                ckpt_path = os.path.join(args.output_dir, f"step_{global_step}.pt")
                ckpt_data = {
                    "global_step": global_step,
                    "model_state_dict": cm_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics_history[-1] if metrics_history else {},
                    "use_cross_attn_memory": args.use_cross_attn_memory,
                    "num_slots": args.num_slots,
                    "write_lr": args.write_lr,
                }
                torch.save(ckpt_data, ckpt_path)
                logger.info("Saved checkpoint: %s", ckpt_path)

        epoch += 1

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    if is_main:
        logger.info("Training complete. Running final evaluation ...")

    ddp_model.eval()
    final_vanilla_ppl, _ = evaluate_vanilla_ppl(ddp_model, eval_loader, device, world_size)

    if args.use_cross_attn_memory:
        final_memory_ppl, _ = evaluate_memory_ppl(ddp_model, eval_loader, device, world_size)
    else:
        final_memory_ppl = final_vanilla_ppl

    final_wiki_ppl = None
    if wikitext_loader:
        final_wiki_ppl, _ = evaluate_vanilla_ppl(ddp_model, wikitext_loader, device, world_size)

    if is_main:
        logger.info(
            "[FINAL] vanilla_ppl=%.4f (base=%.4f, delta=%.2f%%) | "
            "memory_ppl=%.4f | wiki_ppl=%s",
            final_vanilla_ppl, base_vanilla_ppl,
            100 * (final_vanilla_ppl / base_vanilla_ppl - 1),
            final_memory_ppl,
            f"{final_wiki_ppl:.4f}" if final_wiki_ppl else "N/A",
        )

        # Save final checkpoint
        os.makedirs(args.output_dir, exist_ok=True)
        final_path = os.path.join(args.output_dir, "final.pt")
        final_ckpt = {
            "global_step": global_step,
            "model_state_dict": cm_model.state_dict(),
            "final_vanilla_ppl": final_vanilla_ppl,
            "final_memory_ppl": final_memory_ppl,
            "final_wiki_ppl": final_wiki_ppl,
            "base_vanilla_ppl": base_vanilla_ppl,
            "use_cross_attn_memory": args.use_cross_attn_memory,
            "num_slots": args.num_slots,
            "write_lr": args.write_lr,
        }
        torch.save(final_ckpt, final_path)
        logger.info("Saved final checkpoint: %s", final_path)

        # Write summary
        summary = {
            "global_step": global_step,
            "total_epochs": epoch,
            "scheme": "A_cross_attention_memory",
            "use_cross_attn_memory": args.use_cross_attn_memory,
            "gradient_checkpointing": args.gradient_checkpointing,
            "base_vanilla_ppl": base_vanilla_ppl,
            "init_memory_ppl": init_memory_ppl,
            "final_vanilla_ppl": final_vanilla_ppl,
            "final_memory_ppl": final_memory_ppl,
            "final_wiki_ppl": final_wiki_ppl,
            "vanilla_delta_pct": 100 * (final_vanilla_ppl / base_vanilla_ppl - 1),
            "memory_ratio": final_memory_ppl / max(final_vanilla_ppl, 1e-8),
            "num_slots": args.num_slots,
            "write_lr": args.write_lr,
            "lr": args.lr,
            "num_shards": args.num_shards,
            "chunks_per_doc": args.chunks_per_doc,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100.0 * trainable / total,
            "total_time_s": time.time() - t0,
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Summary: %s", json.dumps(summary, indent=2))

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
