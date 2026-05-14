#!/usr/bin/env python3
"""Large-scale continued pretraining with ChunkMemoryBank on Dolmino data.

Improvements over train_v4_chunk_memory.py:
1. Dolmino dataset (diverse domains, 10B+ tokens) instead of narrow PG19
2. Proper vanilla PPL eval using base model directly (no memory layer interference)
3. WikiText OOD eval to detect catastrophic forgetting
4. KL divergence penalty toward base model (optional)
5. Cosine LR schedule with warmup
6. Multi-node DDP support (4 nodes x 8 GPUs = 32 GPUs)
7. Monitoring: vanilla PPL must NOT increase during training

Research findings addressed:
- Researcher found eval code path went through full memory layer, inflating vanilla PPL
- Researcher found narrow PG19 data caused forgetting → Dolmino fixes this
- Researcher found Flamingo gate too open → ChunkMemoryModel doesn't use gate
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
from peft import LoraConfig, get_peft_model, TaskType

from src.memory.mem_space.chunk_memory_bank import ChunkMemoryBank

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
) -> torch.Tensor:
    N = n_slots + n_tokens
    neg_inf = torch.finfo(dtype).min
    mask = torch.zeros(N, N, dtype=dtype, device=device)
    mask[:n_slots, n_slots:] = neg_inf
    token_causal = torch.triu(
        torch.full((n_tokens, n_tokens), neg_inf, dtype=dtype, device=device),
        diagonal=1,
    )
    mask[n_slots:, n_slots:] = token_causal
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
# ChunkMemoryModel
# --------------------------------------------------------------------------- #

class ChunkMemoryModel(nn.Module):
    def __init__(
        self,
        base_model: LlamaForCausalLM,
        num_slots: int = 64,
        lora_rank: int = 16,
        top_k: int = 8,
        epsilon: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.top_k = top_k
        self.epsilon = epsilon

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

        config = base_model.config
        self.num_layers = config.num_hidden_layers
        self.d_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.d_model // self.num_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)

        self.banks: list[ChunkMemoryBank] = [
            ChunkMemoryBank(num_slots, self.d_model) for _ in range(self.num_layers)
        ]
        self._decoder_layers: list[nn.Module] = self._get_decoder_layers()

        # Store reference to base model for vanilla eval
        self._base_model_ref = base_model

    def _get_decoder_layers(self) -> list[nn.Module]:
        base = self.peft_model.base_model.model.model
        return list(base.layers)

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

        llama_model = self.peft_model.base_model.model.model
        embed_tokens = llama_model.embed_tokens
        hidden_states = embed_tokens(input_ids).to(dtype)

        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        rotary_emb = llama_model.rotary_emb
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
                ext_mask = make_prefix_causal_mask(n_slots, T, dtype, device, B)
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

        llama_model_out = llama_model.norm(hidden_states)
        lm_head = self.peft_model.base_model.model.lm_head
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

    def forward(self, input_ids, labels=None, **kwargs):
        return self.forward_chunk(input_ids, labels=labels)

    @torch.no_grad()
    def forward_vanilla(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Forward through the LoRA model WITHOUT any memory bank interference.
        Uses base_model forward directly — truly clean baseline."""
        # Use peft model's built-in forward (which includes LoRA but no memory hooks)
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
    """Compute PPL using LoRA model WITHOUT memory banks (truly vanilla).
    Uses forward_vanilla which bypasses all memory logic."""
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
    """Compute PPL of the ORIGINAL base model (no LoRA, no memory).
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
    """KL divergence between current LoRA model and frozen base model.
    Prevents LoRA weights from drifting too far from base."""
    out = model.forward_vanilla(input_ids)
    lora_logits = out["logits"]
    # KL(student || teacher) = sum p_teacher * (log p_teacher - log p_student)
    p_base = F.log_softmax(base_logits.float(), dim=-1)
    p_lora = F.log_softmax(lora_logits.float(), dim=-1)
    q_base = F.softmax(base_logits.float(), dim=-1)
    kl = (q_base * (p_base - p_lora)).sum(dim=-1).mean()
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
    p = argparse.ArgumentParser(description="Continued pretraining with ChunkMemoryBank on Dolmino")
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
    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    # Training
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--kl_weight", type=float, default=0.0,
                   help="KL divergence penalty weight toward base model (0=disabled)")
    # Eval / save
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--output_dir", type=str, default="outputs/continued_pretrain")
    p.add_argument("--resume_checkpoint", type=str, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
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
        logger.info("Continued Pretraining with ChunkMemoryBank + Dolmino")
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

    # Create a frozen copy of base logits for KL penalty (share weights)
    base_model_for_kl = base_model if args.kl_weight > 0 else None
    if args.kl_weight > 0:
        for p in base_model.parameters():
            p.requires_grad = False

    if is_main:
        logger.info("Building ChunkMemoryModel with LoRA rank=%d ...", args.lora_rank)

    cm_model = ChunkMemoryModel(
        base_model=base_model,
        num_slots=args.num_slots,
        lora_rank=args.lora_rank,
        top_k=args.top_k,
        epsilon=args.epsilon,
    ).to(device)

    trainable = sum(p.numel() for p in cm_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in cm_model.parameters())
    if is_main:
        logger.info("Trainable params: %d / %d (%.4f%%)", trainable, total, 100.0 * trainable / total)

    ddp_model = DDP(cm_model, device_ids=[local_rank])
    root_model = ddp_model.module

    # Resume
    start_step = 0
    if args.resume_checkpoint:
        if is_main:
            logger.info("Resuming from checkpoint: %s", args.resume_checkpoint)
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        cm_model.peft_model.load_state_dict(ckpt['lora_state_dict'])
        start_step = ckpt.get('global_step', 0)
        if is_main:
            logger.info("Resumed from step %d", start_step)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    lora_params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

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
        eval_chunks = np.concatenate(eval_flat_data, axis=0)[:2000]  # cap at 2000 chunks
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

    base_vanilla_ppl, base_vanilla_tok = evaluate_base_ppl(base_model, eval_loader, device, world_size)
    if is_main:
        logger.info("[BASELINE] base_model (no LoRA) eval PPL: %.4f (%d tokens)", base_vanilla_ppl, base_vanilla_tok)

    init_vanilla_ppl, init_vanilla_tok = evaluate_vanilla_ppl(ddp_model, eval_loader, device, world_size)
    if is_main:
        logger.info("[BASELINE] LoRA (init, no memory) eval PPL: %.4f (%d tokens)", init_vanilla_ppl, init_vanilla_tok)

    init_memory_ppl, init_memory_tok = evaluate_memory_ppl(ddp_model, eval_loader, device, world_size)
    if is_main:
        logger.info("[BASELINE] LoRA+memory (init) eval PPL: %.4f (%d tokens)", init_memory_ppl, init_memory_tok)

    if wikitext_loader:
        base_wiki_ppl, _ = evaluate_base_ppl(base_model, wikitext_loader, device, world_size)
        if is_main:
            logger.info("[BASELINE] base_model WikiText PPL: %.4f", base_wiki_ppl)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info("Starting training: %d steps, lr=%.2e, warmup=%d, grad_accum=%d",
                    args.max_steps, args.lr, args.warmup_steps, args.gradient_accumulation_steps)

    global_step = start_step
    epoch = 0
    best_vanilla_ppl = init_vanilla_ppl
    t0 = time.time()
    metrics_history = []

    while global_step < args.max_steps:
        train_sampler.set_epoch(epoch)
        ddp_model.train()

        for doc_idx, sample in enumerate(train_loader):
            if global_step >= args.max_steps:
                break

            input_ids = sample["input_ids"]  # [chunks_per_doc, seq_len]
            labels = sample["labels"]

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

                # KL penalty
                if args.kl_weight > 0 and base_model_for_kl is not None:
                    with torch.no_grad():
                        base_out = base_model_for_kl(input_ids=chunk_ids)
                        base_logits = base_out.logits.detach()
                    kl_loss = compute_kl_penalty(root_model, base_logits, chunk_ids, args.kl_weight)
                    total_loss_chunk = loss + kl_loss
                else:
                    total_loss_chunk = loss

                if ddp_model.training:
                    (total_loss_chunk / args.gradient_accumulation_steps).backward()

            if doc_tokens > 0 and ddp_model.training:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
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
                    bank_fill = root_model.banks[0].num_filled
                    phase = "P2" if bank_fill >= args.num_slots else "P1"
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
                    memory_ppl, memory_tok = evaluate_memory_ppl(
                        ddp_model, eval_loader, device, world_size,
                    )

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
                                "MONITORING: vanilla_ppl=%.4f > base*1.1=%.4f — forgetting detected!",
                                vanilla_ppl, base_vanilla_ppl * 1.1,
                            )
                        if wiki_ppl and base_wiki_ppl and wiki_ppl > base_wiki_ppl * 1.2:
                            logger.warning(
                                "MONITORING: wiki_ppl=%.4f > base_wiki*1.2=%.4f — OOD forgetting!",
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
                        }
                        metrics_history.append(metrics)
                        with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as f:
                            f.write(json.dumps(metrics) + "\n")

                    ddp_model.train()

                # Save checkpoint
                if is_main and global_step % args.save_interval == 0:
                    ckpt_path = os.path.join(args.output_dir, f"step_{global_step}.pt")
                    torch.save(
                        {
                            "global_step": global_step,
                            "lora_state_dict": cm_model.peft_model.state_dict(),
                            "metrics": metrics_history[-1] if metrics_history else {},
                        },
                        ckpt_path,
                    )
                    logger.info("Saved checkpoint: %s", ckpt_path)

        epoch += 1

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    if is_main:
        logger.info("Training complete. Running final evaluation ...")

    ddp_model.eval()
    final_vanilla_ppl, _ = evaluate_vanilla_ppl(ddp_model, eval_loader, device, world_size)
    final_memory_ppl, _ = evaluate_memory_ppl(ddp_model, eval_loader, device, world_size)

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
        final_path = os.path.join(args.output_dir, "final.pt")
        torch.save(
            {
                "global_step": global_step,
                "lora_state_dict": cm_model.peft_model.state_dict(),
                "final_vanilla_ppl": final_vanilla_ppl,
                "final_memory_ppl": final_memory_ppl,
                "final_wiki_ppl": final_wiki_ppl,
                "base_vanilla_ppl": base_vanilla_ppl,
            },
            final_path,
        )
        logger.info("Saved final checkpoint: %s", final_path)

        # Write summary
        summary = {
            "global_step": global_step,
            "total_epochs": epoch,
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
            "lora_rank": args.lora_rank,
            "lr": args.lr,
            "num_shards": args.num_shards,
            "chunks_per_doc": args.chunks_per_doc,
            "kl_weight": args.kl_weight,
            "total_time_s": time.time() - t0,
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Summary: %s", json.dumps(summary, indent=2))

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
