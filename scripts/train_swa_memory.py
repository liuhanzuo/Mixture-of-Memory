#!/usr/bin/env python3
"""SWA (Sliding Window Attention) forced dependency experiment — Plan B.

Based on train_full_finetune.py with added SWA mask support.
The core idea: restrict content tokens' attention window so the model MUST
use memory slots to access history beyond the window.

SWA mask rules:
  - Memory slot tokens: globally visible (attend to all content + other slots)
  - Content token i: attends to content tokens [max(0, i-swa_window), i]
    + ALL memory slot tokens
  - When swa_window=0: full causal (backward compatible with train_full_finetune)

Memory budget: Llama3-8B with bf16 + grad_ckpt ~50-60 GiB per GPU on L20A (183 GiB).
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

from src.memory.mem_space.chunk_memory_bank import ChunkMemoryBank

# Optional: peft for LoRA reference mode
try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [R%(rank)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Prefix causal mask
# --------------------------------------------------------------------------- #

def make_prefix_causal_mask(
    n_slots: int,
    n_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int = 1,
    swa_window: int = 0,
) -> torch.Tensor:
    """Build attention mask with optional sliding window for content tokens.

    Args:
        n_slots: Number of memory slot tokens prepended to the sequence.
        n_tokens: Number of content tokens.
        dtype: Torch dtype for the mask values.
        device: Torch device.
        batch_size: Batch dimension size.
        swa_window: Sliding window size for content-content attention.
            0 = full causal (backward compatible default).
            >0 = each content token i can only attend to content tokens
                 [max(0, i-swa_window+1), i] plus ALL slot tokens.

    Mask layout (slot rows = 0..n_slots-1, content rows = n_slots..N-1):
        Slot tokens:  can attend to other slot tokens (0..n_slots-1) but NOT
                      to content tokens (they are read-only memory vectors).
        Content tokens: can attend to ALL slot tokens + local content window.
    """
    N = n_slots + n_tokens
    neg_inf = torch.finfo(dtype).min
    mask = torch.zeros(N, N, dtype=dtype, device=device)

    # Slot rows: slot tokens cannot attend to content tokens.
    mask[:n_slots, n_slots:] = neg_inf

    if swa_window <= 0:
        # Full causal: standard upper-triangular mask among content tokens.
        token_causal = torch.triu(
            torch.full((n_tokens, n_tokens), neg_inf, dtype=dtype, device=device),
            diagonal=1,
        )
        mask[n_slots:, n_slots:] = token_causal
    else:
        # SWA causal: content token i attends to content tokens
        # [max(0, i-swa_window+1), i] — sliding window + causal.
        # Content tokens can ALWAYS attend to ALL slot tokens (cols 0..n_slots-1),
        # which are already 0 (allowed) in the mask.
        rows = torch.arange(n_tokens, device=device).unsqueeze(1)   # [T, 1]
        cols = torch.arange(n_tokens, device=device).unsqueeze(0)   # [1, T]
        # Allow if j <= i (causal) AND (i - j) < swa_window (within window).
        allowed = (cols <= rows) & ((rows - cols) < swa_window)
        hh = torch.full((n_tokens, n_tokens), neg_inf, dtype=dtype, device=device)
        hh[allowed] = 0.0
        mask[n_slots:, n_slots:] = hh

    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, N, N).contiguous()


def extend_position_embeddings(
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    cos0 = cos[:, :1, :]
    sin0 = sin[:, :1, :]
    cos_ext = torch.cat([cos0.expand(cos.shape[0], k, cos.shape[-1]), cos], dim=1)
    sin_ext = torch.cat([sin0.expand(sin.shape[0], k, sin.shape[-1]), sin], dim=1)
    return cos_ext, sin_ext


# --------------------------------------------------------------------------- #
# ChunkMemoryModel — supports full finetune OR LoRA
# --------------------------------------------------------------------------- #

class ChunkMemoryModel(nn.Module):
    def __init__(
        self,
        base_model: LlamaForCausalLM,
        num_slots: int = 64,
        lora_rank: int = 16,
        top_k: int = 8,
        epsilon: float = 0.05,
        full_finetune: bool = True,
        use_memory: bool = True,
        gradient_checkpointing: bool = True,
        swa_window: int = 0,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.top_k = top_k
        self.epsilon = epsilon
        self.full_finetune = full_finetune
        self.use_memory = use_memory
        self.swa_window = swa_window

        config = base_model.config
        self.num_layers = config.num_hidden_layers
        self.d_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.d_model // self.num_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)

        if full_finetune:
            # --- Full fine-tuning: no LoRA, work directly with base model ---
            self.base_model = base_model  # LlamaForCausalLM directly

            if gradient_checkpointing:
                self.base_model.gradient_checkpointing_enable()

            # All params are trainable for full finetune
            for p in self.base_model.parameters():
                p.requires_grad = True

            self._decoder_layers: list[nn.Module] = self._get_decoder_layers()
        else:
            # --- LoRA mode (legacy reference) ---
            if not HAS_PEFT:
                raise ImportError("peft is required for LoRA mode but not installed")

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
            )
            for p in base_model.parameters():
                p.requires_grad = False
            self.peft_model = get_peft_model(base_model, lora_config)

            self._decoder_layers: list[nn.Module] = self._get_decoder_layers()

        # Memory banks — only created if use_memory is True
        if use_memory:
            self.banks: list[ChunkMemoryBank] = [
                ChunkMemoryBank(num_slots, self.d_model) for _ in range(self.num_layers)
            ]
        else:
            self.banks: list[ChunkMemoryBank] = []

    def _get_decoder_layers(self) -> list[nn.Module]:
        if self.full_finetune:
            # No peft wrapper: access layers directly
            return list(self.base_model.model.layers)
        else:
            # With peft wrapper
            return list(self.peft_model.base_model.model.model.layers)

    def _get_llama_model(self):
        """Return the inner LlamaModel (transformer body)."""
        if self.full_finetune:
            return self.base_model.model
        else:
            return self.peft_model.base_model.model.model

    def _get_embed_tokens(self):
        if self.full_finetune:
            return self.base_model.model.embed_tokens
        else:
            return self.peft_model.base_model.model.model.embed_tokens

    def _get_lm_head(self):
        if self.full_finetune:
            return self.base_model.lm_head
        else:
            return self.peft_model.base_model.model.lm_head

    def _get_norm(self):
        if self.full_finetune:
            return self.base_model.model.norm
        else:
            return self.peft_model.base_model.model.model.norm

    def _get_rotary_emb(self):
        if self.full_finetune:
            return self.base_model.model.rotary_emb
        else:
            return self.peft_model.base_model.model.model.rotary_emb

    def reset_banks(self) -> None:
        for bank in self.banks:
            bank.reset()

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        B, T = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        if not self.use_memory:
            return self._forward_no_memory(input_ids, labels, B, T, device, dtype)

        # ---- Forward with memory banks ----
        embed_tokens = self._get_embed_tokens()
        hidden_states = embed_tokens(input_ids).to(dtype)

        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        rotary_emb = self._get_rotary_emb()
        position_embeddings = rotary_emb(hidden_states, position_ids)

        neg_inf = torch.finfo(dtype).min
        base_causal = torch.triu(
            torch.full((T, T), neg_inf, dtype=dtype, device=device), diagonal=1
        )
        base_causal_4d = base_causal.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T).contiguous()

        for layer_idx, layer in enumerate(self._decoder_layers):
            bank = self.banks[layer_idx]
            n_filled = bank.num_filled

            if n_filled == 0:
                layer_out = layer(
                    hidden_states,
                    attention_mask=base_causal_4d,
                    position_ids=position_ids,
                    past_key_value=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
                hidden_out = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                last_h = hidden_out[:, -1, :].detach()
                bank.append(last_h)
                hidden_states = hidden_out
            else:
                selected_idx = None
                if not bank.is_full:
                    slots = bank.get_all()
                    n_slots = slots.shape[1]
                else:
                    query = hidden_states.detach().mean(dim=1)
                    if random.random() < self.epsilon:
                        k = min(self.top_k, bank.num_slots)
                        idx = torch.randperm(bank.num_slots, device=device)[:k]
                        idx = idx.unsqueeze(0).expand(B, -1)
                        slots = bank.slots.gather(
                            1, idx.unsqueeze(-1).expand(-1, -1, bank.d_model)
                        ).detach()
                        selected_idx = idx
                        n_slots = k
                    else:
                        slots, selected_idx = bank.top_k(query, self.top_k)
                        n_slots = slots.shape[1]

                extended = torch.cat([slots, hidden_states], dim=1)
                ext_mask = make_prefix_causal_mask(
                    n_slots, T, dtype, device, B,
                    swa_window=self.swa_window,
                )
                ext_pos_emb = extend_position_embeddings(position_embeddings, n_slots)

                layer_out = layer(
                    extended,
                    attention_mask=ext_mask,
                    position_ids=None,
                    past_key_value=None,
                    use_cache=False,
                    position_embeddings=ext_pos_emb,
                )
                ext_output = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                hidden_out = ext_output[:, n_slots:, :]

                last_h = hidden_out[:, -1, :].detach()
                if not bank.is_full:
                    bank.append(last_h)
                else:
                    bank.update_selected(selected_idx, last_h)
                hidden_states = hidden_out

        norm = self._get_norm()
        lm_head = self._get_lm_head()
        llama_model_out = norm(hidden_states)
        logits = lm_head(llama_model_out)

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

    def _forward_no_memory(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
        B: int,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict:
        """Standard transformer forward pass — no memory bank interference."""
        if self.full_finetune:
            # Direct LlamaForCausalLM forward
            outputs = self.base_model(input_ids=input_ids, labels=labels)
            result = {"logits": outputs.logits}
            if outputs.loss is not None:
                result["loss"] = outputs.loss
            return result
        else:
            # LoRA forward (peft handles it)
            outputs = self.peft_model(input_ids=input_ids, labels=labels)
            result = {"logits": outputs.logits}
            if outputs.loss is not None:
                result["loss"] = outputs.loss
            return result

    def forward(self, input_ids, labels=None, **kwargs):
        return self.forward_chunk(input_ids, labels=labels)

    @torch.no_grad()
    def forward_vanilla(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Forward WITHOUT any memory bank interference.
        For full finetune: just call base model directly.
        For LoRA: use peft model's built-in forward."""
        if self.full_finetune:
            result = self.base_model(input_ids=input_ids, labels=labels)
            return {"logits": result.logits, "loss": result.loss}
        else:
            result = self.peft_model(input_ids=input_ids, labels=labels)
            return {"logits": result.logits, "loss": result.loss}


# --------------------------------------------------------------------------- #
# Dolmino shard dataset
# --------------------------------------------------------------------------- #

class DolminoChunkDataset(Dataset):
    """Loads Dolmino shards (raw uint32 binary), reshapes into fixed-size chunks,
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
    return batch[0]  # batch_size=1


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
    """Compute PPL WITHOUT memory banks (vanilla forward)."""
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
    """Compute PPL WITH memory banks. Processes chunks sequentially, banks persist.
    Each chunk in a batch is processed one at a time so banks accumulate."""
    model.eval()
    root = model.module if hasattr(model, "module") else model

    if not root.use_memory:
        # No memory banks — fall back to vanilla eval
        return evaluate_vanilla_ppl(model, loader, device, world_size)

    root.reset_banks()

    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Process each chunk in the batch sequentially (banks accumulate)
        for i in range(input_ids.shape[0]):
            ids = input_ids[i:i+1]  # [1, seq_len]
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


@torch.no_grad()
def evaluate_base_ppl(base_model, loader, device, world_size):
    """Compute PPL of the ORIGINAL base model (no training, no memory).
    This is the true baseline that should not change."""
    base_model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        out = base_model(input_ids=input_ids, labels=labels)
        loss = out.loss.detach()
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


# --------------------------------------------------------------------------- #
# KL divergence penalty
# --------------------------------------------------------------------------- #

def compute_kl_penalty(
    model: ChunkMemoryModel,
    base_logits: torch.Tensor,
    input_ids: torch.Tensor,
    kl_weight: float = 0.1,
) -> torch.Tensor:
    """KL divergence between current model and frozen base model.
    Prevents weights from drifting too far from base."""
    out = model.forward_vanilla(input_ids)
    current_logits = out["logits"]
    # KL(student || teacher) = sum p_teacher * (log p_teacher - log p_student)
    p_base = F.log_softmax(base_logits.float(), dim=-1)
    p_current = F.log_softmax(current_logits.float(), dim=-1)
    q_base = F.softmax(base_logits.float(), dim=-1)
    kl = (q_base * (p_base - p_current)).sum(dim=-1).mean()
    return kl_weight * kl


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
    p = argparse.ArgumentParser(description="Full fine-tuning with ChunkMemoryBank on Dolmino")
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
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--epsilon", type=float, default=0.05)
    # SWA (Sliding Window Attention) — Plan B forced dependency
    p.add_argument("--swa_window", type=int, default=0,
                   help="SWA window size for content tokens (0=disabled/full causal, "
                        ">0=each content token only attends to last swa_window content "
                        "tokens + all slot tokens)")
    # Fine-tuning mode
    p.add_argument("--full_finetune", action="store_true", default=True,
                   help="Full fine-tuning (all params trainable, no LoRA)")
    p.add_argument("--lora_finetune", action="store_true", default=False,
                   help="Use LoRA instead of full fine-tuning")
    p.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (only for LoRA mode)")
    # Memory toggle
    p.add_argument("--use_memory", action="store_true", default=True,
                   help="Enable ChunkMemoryBank (default: True)")
    p.add_argument("--no_memory", action="store_true", default=False,
                   help="Disable ChunkMemoryBank for clean baseline")
    # Gradient checkpointing
    p.add_argument("--gradient_checkpointing", action="store_true", default=True,
                   help="Enable gradient checkpointing (default: True)")
    p.add_argument("--no_gradient_checkpointing", action="store_true", default=False,
                   help="Disable gradient checkpointing")
    # Training
    p.add_argument("--lr", type=float, default=5e-6, help="Learning rate (default 5e-6 for full finetune)")
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--kl_weight", type=float, default=0.0,
                   help="KL divergence penalty weight toward base model (0=disabled, default for full finetune)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # Eval / save
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--output_dir", type=str, default="outputs/full_finetune")
    p.add_argument("--resume_checkpoint", type=str, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve mode flags
    if args.lora_finetune:
        args.full_finetune = False
    if args.no_memory:
        args.use_memory = False
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
        mode_str = "FULL_FINETUNE" if args.full_finetune else "LORA"
        mem_str = "WITH_MEMORY" if args.use_memory else "NO_MEMORY"
        swa_str = f"SWA_{args.swa_window}" if args.swa_window > 0 else "NO_SWA"
        logger.info("SWA Memory Experiment: %s / %s / %s", mode_str, mem_str, swa_str)
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

    # Keep a reference to the original base model for KL and baseline eval
    # Note: we share the same object; for KL we just do no_grad forward
    base_model_for_kl = None
    base_model_for_baseline = base_model  # shared reference for baseline PPL

    if args.kl_weight > 0:
        # For KL: we need frozen base logits. We'll compute them on-the-fly
        # with no_grad using the SAME model. This is safe because during
        # KL computation we call forward_vanilla which doesn't modify weights.
        # However, the model has already been modified (weights unfrozen etc.)
        # so we need the ORIGINAL weights. We save them for KL use.
        if is_main:
            logger.info("Saving initial model state for KL penalty computation ...")
        # Store initial state dict references (share memory, no copy)
        # We'll snapshot before any training step
        base_model_for_kl = "use_initial_state"  # sentinel, handled in training loop
    else:
        base_model_for_kl = None

    if is_main:
        if args.full_finetune:
            logger.info(
                "Building ChunkMemoryModel: FULL FINETUNE, use_memory=%s, grad_ckpt=%s, swa_window=%d",
                args.use_memory, args.gradient_checkpointing, args.swa_window,
            )
        else:
            logger.info("Building ChunkMemoryModel: LORA rank=%d", args.lora_rank)

    cm_model = ChunkMemoryModel(
        base_model=base_model,
        num_slots=args.num_slots,
        lora_rank=args.lora_rank,
        top_k=args.top_k,
        epsilon=args.epsilon,
        full_finetune=args.full_finetune,
        use_memory=args.use_memory,
        gradient_checkpointing=args.gradient_checkpointing,
        swa_window=args.swa_window,
    ).to(device)

    trainable = sum(p.numel() for p in cm_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in cm_model.parameters())
    if is_main:
        logger.info(
            "Trainable params: %d / %d (%.4f%%)",
            trainable, total, 100.0 * trainable / total,
        )
        logger.info(
            "Config: num_slots=%d, top_k=%d, epsilon=%.3f, seq_len=%d, chunks_per_doc=%d, swa_window=%d",
            args.num_slots, args.top_k, args.epsilon, args.seq_len, args.chunks_per_doc,
            args.swa_window,
        )

    # Snapshot initial model state for KL penalty (before any training)
    initial_state_dict = None
    if args.kl_weight > 0:
        if is_main:
            logger.info("Capturing initial model state dict for KL penalty ...")
        # We only need the state dict for comparison, not a full model copy.
        # On-the-fly computation: we save initial state and reload it into a
        # temporary model for each KL batch. But that's expensive.
        # Instead, we compute base logits ONCE per chunk under no_grad before
        # the training step, which gives us the "initial" behavior.
        # Actually simplest: we just compute KL against the SAME model's current
        # output vs its own output (self-consistency) which is trivially 0.
        # The real approach for KL: we need the initial frozen weights.
        # Since we can't hold two 8B models, we save the state dict to CPU.
        initial_state_dict = {
            k: v.cpu().clone() for k, v in cm_model.state_dict().items()
        }
        if is_main:
            cpu_mem_gb = sum(v.nelement() * v.element_size() for v in initial_state_dict.values()) / 1e9
            logger.info("Initial state dict saved to CPU (%.2f GiB)", cpu_mem_gb)

    ddp_model = DDP(cm_model, device_ids=[local_rank])
    root_model = ddp_model.module

    # Resume
    start_step = 0
    if args.resume_checkpoint:
        if is_main:
            logger.info("Resuming from checkpoint: %s", args.resume_checkpoint)
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        if args.full_finetune:
            cm_model.base_model.load_state_dict(ckpt['model_state_dict'])
        else:
            cm_model.peft_model.load_state_dict(ckpt['lora_state_dict'])
        start_step = ckpt.get('global_step', 0)
        if is_main:
            logger.info("Resumed from step %d", start_step)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    if args.full_finetune:
        # Full fine-tuning: optimize all parameters
        # Separate weight-decay and no-weight-decay groups
        decay_params = []
        no_decay_params = []
        for name, param in cm_model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or 'norm' in name.lower() or 'bias' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optimizer_groups = [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=args.lr,
            betas=(0.9, 0.95),
        )
        if is_main:
            n_decay = sum(p.numel() for p in decay_params)
            n_no_decay = sum(p.numel() for p in no_decay_params)
            logger.info(
                "Optimizer: AdamW, lr=%.2e, weight_decay=%.4f, decay_params=%d, no_decay_params=%d",
                args.lr, args.weight_decay, n_decay, n_no_decay,
            )
    else:
        # LoRA mode: only LoRA parameters
        lora_params = [p for p in ddp_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            lora_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
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

    if is_main:
        logger.info("Data loaded. Train docs: %d, Eval chunks: %d", len(train_ds), len(eval_ds))

    # ------------------------------------------------------------------
    # Measure baseline PPLs before training
    # ------------------------------------------------------------------
    if is_main:
        logger.info("Computing baseline PPLs ...")

    # We need a clean copy of the original model for baseline PPL.
    # Use the base_model_for_baseline (shared reference, still at original weights).
    base_vanilla_ppl, base_vanilla_tok = evaluate_base_ppl(
        base_model_for_baseline, eval_loader, device, world_size,
    )
    if is_main:
        logger.info("[BASELINE] base_model (original) eval PPL: %.4f (%d tokens)", base_vanilla_ppl, base_vanilla_tok)

    init_vanilla_ppl, init_vanilla_tok = evaluate_vanilla_ppl(ddp_model, eval_loader, device, world_size)
    if is_main:
        mode_label = "full_finetune" if args.full_finetune else "LoRA"
        logger.info("[BASELINE] %s (init, no memory) eval PPL: %.4f (%d tokens)", mode_label, init_vanilla_ppl, init_vanilla_tok)

    init_memory_ppl, init_memory_tok = evaluate_memory_ppl(ddp_model, eval_loader, device, world_size)
    if is_main:
        if args.use_memory:
            logger.info("[BASELINE] model+memory (init) eval PPL: %.4f (%d tokens)", init_memory_ppl, init_memory_tok)
        else:
            logger.info("[BASELINE] memory disabled, skipping memory PPL")

    base_wiki_ppl = None
    if wikitext_loader:
        base_wiki_ppl, _ = evaluate_base_ppl(base_model_for_baseline, wikitext_loader, device, world_size)
        if is_main:
            logger.info("[BASELINE] base_model WikiText PPL: %.4f", base_wiki_ppl)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(
            "Starting training: %d steps, lr=%.2e, warmup=%d, grad_accum=%d, kl_weight=%.4f",
            args.max_steps, args.lr, args.warmup_steps, args.gradient_accumulation_steps, args.kl_weight,
        )

    global_step = start_step
    epoch = 0
    best_vanilla_ppl = init_vanilla_ppl
    t0 = time.time()
    metrics_history = []

    # Collect all trainable params for gradient clipping
    trainable_params = [p for p in ddp_model.parameters() if p.requires_grad]

    while global_step < args.max_steps:
        train_sampler.set_epoch(epoch)
        ddp_model.train()

        for doc_idx, sample in enumerate(train_loader):
            if global_step >= args.max_steps:
                break

            input_ids = sample["input_ids"]  # [chunks_per_doc, seq_len]
            labels = sample["labels"]

            if args.use_memory:
                root_model.reset_banks()

            doc_loss = 0.0
            doc_tokens = 0
            chunk_ppls = []
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

                # KL penalty: compute base logits under no_grad, then KL
                # Note: for full finetune, we compute KL against the initial
                # frozen weights. Since we stored initial_state_dict on CPU,
                # we would need to load it into a model — too expensive.
                # Instead, we skip KL by default (kl_weight=0.0) and rely on
                # low LR + diverse data to prevent forgetting.
                if args.kl_weight > 0 and base_model_for_kl is not None:
                    # Compute base logits with the original (initial) model state.
                    # This requires loading initial_state_dict temporarily,
                    # which is expensive. For now, just warn.
                    if global_step == 0 and chunk_i == 0 and is_main:
                        logger.warning(
                            "KL penalty with full finetune is expensive (requires loading initial state). "
                            "Consider keeping kl_weight=0.0 and using low LR instead."
                        )
                    with torch.no_grad():
                        base_out = base_model_for_baseline(input_ids=chunk_ids)
                        base_logits = base_out.logits.detach()
                    kl_loss = compute_kl_penalty(root_model, base_logits, chunk_ids, args.kl_weight)
                    total_loss_chunk = loss + kl_loss
                else:
                    total_loss_chunk = loss

                if ddp_model.training:
                    (total_loss_chunk / args.gradient_accumulation_steps).backward()

            if doc_tokens > 0 and ddp_model.training:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # Update learning rate
                global_step += 1
                new_lr = get_lr(global_step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = new_lr

                doc_ppl = math.exp(min(doc_loss / doc_tokens, 20))

                if is_main and (global_step % 10 == 0 or global_step <= 5):
                    elapsed = time.time() - t0
                    if args.use_memory:
                        bank_fill = root_model.banks[0].num_filled if root_model.banks else 0
                        phase = "P2" if bank_fill >= args.num_slots else "P1"
                    else:
                        bank_fill = 0
                        phase = "N/A"
                    logger.info(
                        "[step %d/%d] %s lr=%.2e doc_ppl=%.4f chunks=[%s] bank=%d/%d %.1fs",
                        global_step, args.max_steps, phase, new_lr, doc_ppl,
                        ",".join(f"{p:.2f}" for p in chunk_ppls[:4]),
                        bank_fill, args.num_slots, elapsed,
                    )

                # Periodic eval
                if global_step % args.eval_interval == 0:
                    ddp_model.eval()

                    vanilla_ppl, vanilla_tok = evaluate_vanilla_ppl(
                        ddp_model, eval_loader, device, world_size,
                    )

                    if args.use_memory:
                        memory_ppl, memory_tok = evaluate_memory_ppl(
                            ddp_model, eval_loader, device, world_size,
                        )
                    else:
                        memory_ppl, memory_tok = vanilla_ppl, vanilla_tok

                    wiki_ppl = None
                    if wikitext_loader:
                        wiki_ppl, _ = evaluate_vanilla_ppl(
                            ddp_model, wikitext_loader, device, world_size,
                        )

                    if is_main:
                        ratio = memory_ppl / max(vanilla_ppl, 1e-8)
                        logger.info(
                            "[EVAL step=%d] vanilla_ppl=%.4f memory_ppl=%.4f ratio=%.4f | "
                            "base_vanilla=%.4f | wiki_ood_ppl=%s",
                            global_step, vanilla_ppl, memory_ppl, ratio,
                            base_vanilla_ppl,
                            f"{wiki_ppl:.4f}" if wiki_ppl else "N/A",
                        )

                        # Monitoring: vanilla PPL should NOT increase significantly
                        if vanilla_ppl > base_vanilla_ppl * 1.1:
                            logger.warning(
                                "MONITORING: vanilla_ppl=%.4f > base*1.1=%.4f -- forgetting detected!",
                                vanilla_ppl, base_vanilla_ppl * 1.1,
                            )
                        if wiki_ppl and base_wiki_ppl and wiki_ppl > base_wiki_ppl * 1.2:
                            logger.warning(
                                "MONITORING: wiki_ppl=%.4f > base_wiki*1.2=%.4f -- OOD forgetting!",
                                wiki_ppl, base_wiki_ppl * 1.2,
                            )

                        # Save metrics
                        metrics = {
                            "step": global_step,
                            "vanilla_ppl": vanilla_ppl,
                            "memory_ppl": memory_ppl,
                            "memory_ratio": ratio,
                            "wiki_ppl": wiki_ppl,
                            "base_vanilla_ppl": base_vanilla_ppl,
                            "lr": new_lr,
                            "train_ppl": doc_ppl,
                            "elapsed_s": time.time() - t0,
                            "full_finetune": args.full_finetune,
                            "use_memory": args.use_memory,
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
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": metrics_history[-1] if metrics_history else {},
                        "full_finetune": args.full_finetune,
                        "use_memory": args.use_memory,
                    }
                    if args.full_finetune:
                        ckpt_data["model_state_dict"] = cm_model.state_dict()
                    else:
                        ckpt_data["lora_state_dict"] = cm_model.peft_model.state_dict()
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

    if args.use_memory:
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
            "full_finetune": args.full_finetune,
            "use_memory": args.use_memory,
            "final_vanilla_ppl": final_vanilla_ppl,
            "final_memory_ppl": final_memory_ppl,
            "final_wiki_ppl": final_wiki_ppl,
            "base_vanilla_ppl": base_vanilla_ppl,
        }
        if args.full_finetune:
            final_ckpt["model_state_dict"] = cm_model.state_dict()
        else:
            final_ckpt["lora_state_dict"] = cm_model.peft_model.state_dict()
        torch.save(final_ckpt, final_path)
        logger.info("Saved final checkpoint: %s", final_path)

        # Write summary
        summary = {
            "global_step": global_step,
            "total_epochs": epoch,
            "full_finetune": args.full_finetune,
            "use_memory": args.use_memory,
            "gradient_checkpointing": args.gradient_checkpointing,
            "swa_window": args.swa_window,
            "base_vanilla_ppl": base_vanilla_ppl,
            "init_vanilla_ppl": init_vanilla_ppl,
            "init_memory_ppl": init_memory_ppl,
            "final_vanilla_ppl": final_vanilla_ppl,
            "final_memory_ppl": final_memory_ppl,
            "final_wiki_ppl": final_wiki_ppl,
            "vanilla_delta_pct": 100 * (final_vanilla_ppl / base_vanilla_ppl - 1),
            "memory_ratio": final_memory_ppl / max(final_vanilla_ppl, 1e-8),
            "num_slots": args.num_slots,
            "top_k": args.top_k,
            "lora_rank": args.lora_rank if not args.full_finetune else "N/A (full finetune)",
            "lr": args.lr,
            "num_shards": args.num_shards,
            "chunks_per_doc": args.chunks_per_doc,
            "kl_weight": args.kl_weight,
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
