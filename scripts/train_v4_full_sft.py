#!/usr/bin/env python3
"""V4 Full SFT -- Full-parameter supervised fine-tuning with per-layer memory banks.

Phase 1 (append-only): slots fill up one by one, model sees all filled slots.
Phase 2 (top-k selection): once bank is full, select top-k slots by cosine
similarity + epsilon-greedy exploration, EMA-update selected slots.

Key difference from train_v4_chunk_memory.py:
- No LoRA. ALL parameters of LlamaForCausalLM are trainable.
- Dual data source: 90% pretrain (forward_plain, no banks) + 10% memory (forward_chunk).
- Gradient checkpointing enabled to reduce activation memory.
- Cosine LR schedule with warmup.

Design reference:  scripts/FULL_SFT_PLAN.md
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, LlamaForCausalLM, get_scheduler

from src.memory.mem_space.chunk_memory_bank import ChunkMemoryBank

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [R%(rank)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Prefix causal mask (from v4 design doc Section 2.3)
# --------------------------------------------------------------------------- #

def make_prefix_causal_mask(
    n_slots: int,
    n_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int = 1,
) -> torch.Tensor:
    """Build [B, 1, n_slots+n_tokens, n_slots+n_tokens] additive mask.

    Pattern:
        slot -> slot  : allowed (0)
        slot -> token : masked  (-inf)  -- slots do NOT see future tokens
        token -> slot : allowed (0)     -- tokens see ALL slots
        token -> token: causal
    """
    N = n_slots + n_tokens
    neg_inf = torch.finfo(dtype).min
    mask = torch.zeros(N, N, dtype=dtype, device=device)

    # slot -> token: masked
    mask[:n_slots, n_slots:] = neg_inf

    # token -> token: causal upper triangle
    token_causal = torch.triu(
        torch.full((n_tokens, n_tokens), neg_inf, dtype=dtype, device=device),
        diagonal=1,
    )
    mask[n_slots:, n_slots:] = token_causal

    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, N, N).contiguous()


# --------------------------------------------------------------------------- #
# Extend position embeddings (same logic as layer.py:_extend_position_embeddings)
# --------------------------------------------------------------------------- #

def extend_position_embeddings(
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepend k position-0 entries to (cos, sin) tables.

    position_id=0 -> cos(0)=1, sin(0)=0 -> no RoPE rotation on slots.
    """
    cos, sin = position_embeddings
    cos0 = cos[:, :1, :]
    sin0 = sin[:, :1, :]
    cos_ext = torch.cat([cos0.expand(cos.shape[0], k, cos.shape[-1]), cos], dim=1)
    sin_ext = torch.cat([sin0.expand(sin.shape[0], k, sin.shape[-1]), sin], dim=1)
    return cos_ext, sin_ext


# --------------------------------------------------------------------------- #
# ChunkMemoryModel -- Full SFT version (no LoRA)
# --------------------------------------------------------------------------- #

class ChunkMemoryModel(nn.Module):
    """Wraps LlamaForCausalLM (fully trainable) with per-layer memory banks.

    Key difference from LoRA version:
    - No PeftModel wrapper. Direct LlamaForCausalLM.
    - ALL parameters are trainable.
    - Memory banks remain pure runtime state (no gradients, NOT nn.Module).
    """

    def __init__(
        self,
        base_model: LlamaForCausalLM,
        num_slots: int = 64,
        top_k: int = 8,
        epsilon: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.top_k = top_k
        self.epsilon = epsilon

        # Store the base model directly -- no LoRA, all params trainable.
        self.model = base_model

        # Derive model metadata.
        config = base_model.config
        self.num_layers = config.num_hidden_layers
        self.d_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.d_model // self.num_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)

        # Per-layer memory banks (plain Python objects, NOT nn.Module).
        self.banks: list[ChunkMemoryBank] = [
            ChunkMemoryBank(num_slots, self.d_model) for _ in range(self.num_layers)
        ]

        # Direct references to decoder layers.
        self._decoder_layers: list[nn.Module] = list(
            self.model.model.layers
        )

        # Enable gradient checkpointing on the base model.
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def reset_banks(self) -> None:
        for bank in self.banks:
            bank.reset()

    def forward_plain(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Standard Llama forward, no memory banks. For pretrain data."""
        outputs = self.model(input_ids=input_ids, labels=labels)
        return {"logits": outputs.logits, "loss": outputs.loss}

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Forward one chunk through all decoder layers with memory-bank prefix.

        Args:
            input_ids: [B, T]
            labels:    [B, T] or None (eval-only mode)

        Returns:
            dict with "loss" (if labels provided) and "logits".
        """
        B, T = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        # Get the internal LlamaModel to compute embeddings + position embeddings.
        llama_model = self.model.model
        embed_tokens = llama_model.embed_tokens
        hidden_states = embed_tokens(input_ids)  # [B, T, d]
        hidden_states = hidden_states.to(dtype)

        # Position ids: [1, T]
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        # Compute rotary embeddings from the model's rotary_emb.
        rotary_emb = llama_model.rotary_emb
        position_embeddings = rotary_emb(hidden_states, position_ids)  # (cos, sin)

        # Build the base causal mask for the tokens-only portion.
        neg_inf = torch.finfo(dtype).min
        base_causal = torch.triu(
            torch.full((T, T), neg_inf, dtype=dtype, device=device), diagonal=1
        )  # [T, T]
        base_causal_4d = base_causal.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T).contiguous()

        # ------------------------------------------------------------------
        # Pass through each decoder layer, injecting bank slots at each layer.
        # ------------------------------------------------------------------
        for layer_idx, layer in enumerate(self._decoder_layers):
            bank = self.banks[layer_idx]
            n_filled = bank.num_filled

            if n_filled == 0:
                # No slots yet -- normal forward.
                layer_out = layer(
                    hidden_states,
                    attention_mask=base_causal_4d,
                    position_ids=position_ids,
                    past_key_value=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
                if isinstance(layer_out, tuple):
                    hidden_out = layer_out[0]
                else:
                    hidden_out = layer_out

                # Update bank with last token hidden (detached).
                last_h = hidden_out[:, -1, :].detach()  # [B, d]
                bank.append(last_h)
                hidden_states = hidden_out
            else:
                # Phase 1 (bank not full) or Phase 2 (bank full).
                selected_idx = None  # track for Phase 2 EMA update

                if not bank.is_full:
                    # Phase 1: use all filled slots.
                    slots = bank.get_all()  # [B, n_filled, d]
                    n_slots = slots.shape[1]
                else:
                    # Phase 2: top-k selection + EMA update.
                    query = hidden_states.detach().mean(dim=1)  # [B, d]

                    # epsilon-greedy exploration.
                    if random.random() < self.epsilon:
                        # Random selection for exploration.
                        k = min(self.top_k, bank.num_slots)
                        idx = torch.randperm(bank.num_slots, device=device)[:k]
                        idx = idx.unsqueeze(0).expand(B, -1)  # [B, k]
                        slots = bank.slots.gather(
                            1, idx.unsqueeze(-1).expand(-1, -1, bank.d_model)
                        ).detach()
                        selected_idx = idx
                        n_slots = k
                    else:
                        slots, selected_idx = bank.top_k(query, self.top_k)
                        n_slots = slots.shape[1]  # = top_k

                # Build extended sequence [slots | tokens].
                extended = torch.cat([slots, hidden_states], dim=1)  # [B, n_slots+T, d]

                # Build prefix causal mask.
                ext_mask = make_prefix_causal_mask(n_slots, T, dtype, device, B)

                # Extend position embeddings: slots get pos=0.
                ext_pos_emb = extend_position_embeddings(position_embeddings, n_slots)

                # Forward through the decoder layer.
                layer_out = layer(
                    extended,
                    attention_mask=ext_mask,
                    position_ids=None,  # RoPE driven by ext_pos_emb
                    past_key_value=None,
                    use_cache=False,
                    position_embeddings=ext_pos_emb,
                )
                if isinstance(layer_out, tuple):
                    ext_output = layer_out[0]
                else:
                    ext_output = layer_out

                # Take only token portion.
                hidden_out = ext_output[:, n_slots:, :]  # [B, T, d]

                # Update bank with last token hidden (detached).
                last_h = hidden_out[:, -1, :].detach()
                if not bank.is_full:
                    # Phase 1: append.
                    bank.append(last_h)
                else:
                    # Phase 2: EMA update selected slots.
                    bank.update_selected(selected_idx, last_h)

                hidden_states = hidden_out

        # ------------------------------------------------------------------
        # Final layernorm + LM head.
        # ------------------------------------------------------------------
        llama_model_out = llama_model.norm(hidden_states)
        lm_head = self.model.lm_head
        logits = lm_head(llama_model_out)  # [B, T, vocab]

        result = {"logits": logits}
        if labels is not None:
            # Shift for NTP loss.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            result["loss"] = loss
        return result

    def forward(self, input_ids, labels=None, mode="memory", **kwargs):
        """Dispatch to forward_plain or forward_chunk based on mode."""
        if mode == "pretrain":
            return self.forward_plain(input_ids, labels=labels)
        else:
            return self.forward_chunk(input_ids, labels=labels)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

class FlatChunkDataset(Dataset):
    """Pretrain data: each chunk is an independent sample.

    No document grouping, no memory banks. Used for forward_plain() calls.
    """

    def __init__(
        self,
        npy_path: str,
        seq_len: int,
        skip: int,
        max_chunks: int,
    ) -> None:
        data = np.load(npy_path, mmap_mode="r")
        self.data = data[skip: skip + max_chunks].astype(np.int32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        t = torch.tensor(self.data[idx], dtype=torch.long)[: self.seq_len]
        return {"input_ids": t, "labels": t.clone()}


class DocumentChunkDataset(Dataset):
    """Groups pg19 chunks into documents of `chunks_per_doc` sequential chunks.

    Each sample is a list of chunk arrays (each chunk = seq_len tokens).
    During training, the model processes chunks sequentially within a document,
    with memory banks persisting across chunks.
    """

    def __init__(
        self,
        npy_path: str,
        seq_length: int,
        skip_chunks: int,
        max_chunks: int,
        chunks_per_doc: int = 8,
    ) -> None:
        data = np.load(npy_path, mmap_mode="r")
        self.data = data[skip_chunks: skip_chunks + max_chunks].astype(np.int32)
        self.seq_length = seq_length
        self.chunks_per_doc = chunks_per_doc

        # Number of complete documents we can form.
        n_chunks = len(self.data)
        self.n_docs = max(1, n_chunks // chunks_per_doc)
        self.n_docs = min(self.n_docs, n_chunks)

        logger.info(
            "Loaded %d chunks -> %d documents (%d chunks/doc) from %s",
            n_chunks, self.n_docs, chunks_per_doc, npy_path,
        )

    def __len__(self) -> int:
        return self.n_docs

    def __getitem__(self, idx: int):
        start = idx * self.chunks_per_doc
        end = start + self.chunks_per_doc
        chunks = []
        for i in range(start, min(end, len(self.data))):
            tokens = torch.tensor(self.data[i], dtype=torch.long)[: self.seq_length]
            chunks.append({"input_ids": tokens, "labels": tokens.clone()})
        # Pad if we ran out of chunks.
        while len(chunks) < self.chunks_per_doc:
            tokens = torch.zeros(self.seq_length, dtype=torch.long)
            chunks.append({"input_ids": tokens, "labels": torch.full_like(tokens, -100)})
        return {"chunks": chunks}


def doc_collate_fn(batch):
    """Collate a batch of documents.  Each doc has chunks_per_doc chunks."""
    # batch: list of dicts with "chunks" key
    # chunks: list of dicts with "input_ids" and "labels"
    return batch[0]["chunks"]  # batch_size=1, return the list of chunks


def cycle_iterator(loader):
    """Infinite iterator that cycles through the DataLoader."""
    while True:
        for batch in loader:
            yield batch


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
def evaluate_vanilla_ppl(model, loader, device, pad_token_id, world_size):
    """Compute PPL WITHOUT memory banks (vanilla baseline).

    Uses forward_plain() so no memory bank slots are injected.
    """
    model.eval()
    root = model.module if hasattr(model, "module") else model
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Use forward_plain -- standard Llama forward, no banks.
        out = root.forward_plain(input_ids, labels=labels)
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
    """Compute PPL WITH memory banks active across sequential chunks."""
    model.eval()
    root = model.module if hasattr(model, "module") else model
    root.reset_banks()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Reset banks so no slots are used.
        root.reset_banks()

        out = root.forward_chunk(input_ids, labels=labels)
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V4 Full SFT -- Full-parameter training with chunk memory banks"
    )
    p.add_argument("--model", type=str, required=True, help="Path to Llama-3-8B weights")
    p.add_argument("--data", type=str, required=True, help="Path to pg19_chunks_llama3.npy")
    p.add_argument("--num_slots", type=int, default=4,
                   help="Number of memory bank slots per layer")
    p.add_argument("--top_k", type=int, default=2,
                   help="Number of slots to select in Phase 2 (top-k)")
    p.add_argument("--epsilon", type=float, default=0.05,
                   help="Epsilon-greedy exploration probability for Phase 2")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Learning rate (conservative for full SFT)")
    p.add_argument("--max_steps", type=int, default=2000,
                   help="Max optimizer steps")
    p.add_argument("--pretrain_max_chunks", type=int, default=4500,
                   help="Max chunks to load for pretrain data")
    p.add_argument("--memory_max_chunks", type=int, default=500,
                   help="Max chunks to load for memory data")
    p.add_argument("--skip_chunks", type=int, default=0,
                   help="Number of initial chunks to skip in npy file")
    p.add_argument("--seq_len", type=int, default=4096,
                   help="Sequence length per chunk")
    p.add_argument("--chunks_per_doc", type=int, default=8,
                   help="Number of chunks per document (banks persist across these)")
    p.add_argument("--pretrain_ratio", type=float, default=0.9,
                   help="Fraction of steps that use pretrain (forward_plain)")
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="Number of LR warmup steps (0 = no warmup, per CPT best practice)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4,
                   help="Gradient accumulation steps before optimizer update")
    p.add_argument("--output_dir", type=str, default="outputs/v4_full_sft",
                   help="Output directory for checkpoints and logs")
    p.add_argument("--eval_interval", type=int, default=100,
                   help="Eval every N steps")
    p.add_argument("--eval_chunks", type=int, default=200,
                   help="Number of chunks for eval PPL computation")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed. Also passed to DistributedSampler(seed=...) -- without that the sampler silently uses its own default 0 and data order is identical across seeds.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    is_main = rank == 0

    # Patch logger with rank info.
    for handler in logging.root.handlers:
        handler.setFormatter(
            logging.Formatter(f"%(asctime)s [R{rank}] %(levelname)s %(message)s")
        )

    device = torch.device(f"cuda:{local_rank}")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    if is_main:
        logger.info("=" * 60)
        logger.info("V4 Full SFT -- Full-parameter training with chunk memory banks")
        logger.info("=" * 60)
        logger.info("Args: %s", vars(args))

    # ------------------------------------------------------------------
    # Load model.
    # ------------------------------------------------------------------
    if is_main:
        logger.info("Loading base model from %s ...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": device},
    )
    if is_main:
        logger.info("Base model loaded.  Building ChunkMemoryModel (full SFT, no LoRA) ...")

    cm_model = ChunkMemoryModel(
        base_model=base_model,
        num_slots=args.num_slots,
        top_k=args.top_k,
        epsilon=args.epsilon,
    ).to(device)

    # Print trainable params.
    trainable = sum(p.numel() for p in cm_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in cm_model.parameters())
    if is_main:
        logger.info("Trainable params: %d / %d (%.4f%%)",
                    trainable, total, 100.0 * trainable / total)
        logger.info("Gradient checkpointing: ENABLED")
        logger.info("Effective batch size: %d GPUs * 1 * %d accum = %d",
                    world_size, args.gradient_accumulation_steps,
                    world_size * args.gradient_accumulation_steps)

    # Wrap in DDP.
    ddp_model = DDP(cm_model, device_ids=[local_rank])

    # ------------------------------------------------------------------
    # Optimizer (ALL parameters).
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        cm_model.parameters(), lr=args.lr, weight_decay=0.01
    )

    # ------------------------------------------------------------------
    # LR Scheduler: cosine with warmup.
    # ------------------------------------------------------------------
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # ------------------------------------------------------------------
    # Data loaders -- dual source.
    # ------------------------------------------------------------------
    # Pretrain data: flat chunks, no document grouping.
    pretrain_ds = FlatChunkDataset(
        npy_path=args.data,
        seq_len=args.seq_len,
        skip=args.skip_chunks,
        max_chunks=args.pretrain_max_chunks,
    )
    # seed=args.seed is LOAD-BEARING: DistributedSampler.__iter__ builds its OWN
    # generator (g.manual_seed(self.seed + self.epoch)) and self.seed defaults to 0,
    # so torch.manual_seed()/set_seed() CANNOT reach it. Without this argument every
    # --seed value gives a BYTE-IDENTICAL data order. Do not delete as redundant.
    pretrain_sampler = DistributedSampler(
        pretrain_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed,
    )
    pretrain_loader = DataLoader(
        pretrain_ds,
        batch_size=1,
        sampler=pretrain_sampler,
        num_workers=0,
        collate_fn=lambda b: {
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "labels": torch.stack([x["labels"] for x in b]),
        },
    )

    # Memory data: grouped into documents.
    memory_ds = DocumentChunkDataset(
        npy_path=args.data,
        seq_length=args.seq_len,
        skip_chunks=args.skip_chunks + args.pretrain_max_chunks,
        max_chunks=args.memory_max_chunks,
        chunks_per_doc=args.chunks_per_doc,
    )
    # seed=args.seed is LOAD-BEARING: DistributedSampler.__iter__ builds its OWN
    # generator (g.manual_seed(self.seed + self.epoch)) and self.seed defaults to 0,
    # so torch.manual_seed()/set_seed() CANNOT reach it. Without this argument every
    # --seed value gives a BYTE-IDENTICAL data order. Do not delete as redundant.
    memory_sampler = DistributedSampler(
        memory_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed,
    )
    memory_loader = DataLoader(
        memory_ds,
        batch_size=1,
        sampler=memory_sampler,
        collate_fn=doc_collate_fn,
        num_workers=0,
    )

    # Eval dataset: flat chunks after training data.
    eval_skip = args.skip_chunks + args.pretrain_max_chunks + args.memory_max_chunks
    eval_ds = FlatChunkDataset(
        npy_path=args.data,
        seq_len=args.seq_len,
        skip=eval_skip,
        max_chunks=args.eval_chunks,
    )
    eval_sampler = DistributedSampler(
        eval_ds, num_replicas=world_size, rank=rank, shuffle=False
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=1, sampler=eval_sampler,
        num_workers=0, collate_fn=lambda b: {
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "labels": torch.stack([x["labels"] for x in b]),
        },
    )

    # Infinite iterators.
    pretrain_iter = cycle_iterator(pretrain_loader)
    memory_iter = cycle_iterator(memory_loader)

    # ------------------------------------------------------------------
    # Training loop -- dual-mode mixed training.
    # ------------------------------------------------------------------
    pad_token_id = tokenizer.pad_token_id or 0
    root_model = ddp_model.module

    if is_main:
        logger.info("Starting training for %d steps ...", args.max_steps)
        logger.info("Pretrain ratio: %.2f (%.0f%% forward_plain, %.0f%% forward_chunk)",
                    args.pretrain_ratio, args.pretrain_ratio * 100,
                    (1 - args.pretrain_ratio) * 100)
        os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    pretrain_steps = 0
    memory_steps = 0
    accum_count = 0
    epoch = 0
    best_vanilla_ppl = float("inf")
    t0 = time.time()

    while global_step < args.max_steps:
        # Set epoch for shuffling.
        pretrain_sampler.set_epoch(epoch)
        memory_sampler.set_epoch(epoch)
        epoch += 1

        ddp_model.train()

        # We track accumulated loss for logging.
        accum_loss = 0.0
        accum_tokens = 0

        for _ in range(len(pretrain_ds) + len(memory_ds)):
            if global_step >= args.max_steps:
                break

            # Choose mode for this micro-step (synchronized across all DDP ranks).
            # Each rank must choose the same mode, otherwise different numbers of
            # backward() calls desynchronize NCCL all-reduce.
            if rank == 0:
                mode_flag = torch.tensor(
                    [1.0 if random.random() < args.pretrain_ratio else 0.0],
                    device=device,
                )
            else:
                mode_flag = torch.tensor([0.0], device=device)
            dist.broadcast(mode_flag, src=0)
            use_pretrain = mode_flag.item() > 0.5

            if use_pretrain:
                # ----------------------------------------------------------
                # PRETRAIN MODE: standard NTP, no memory banks.
                # ----------------------------------------------------------
                batch = next(pretrain_iter)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                # Skip padding-only samples.
                if (labels != -100).sum() == 0:
                    continue

                result = ddp_model(input_ids=input_ids, labels=labels, mode="pretrain")
                loss = result["loss"]

                if not torch.isfinite(loss):
                    if is_main:
                        logger.warning("[step %d] Non-finite pretrain loss!", global_step)
                    continue

                n_tok = (labels != -100).sum().item()
                accum_loss += loss.item() * n_tok
                accum_tokens += n_tok

                # Scale loss by gradient accumulation steps.
                scaled_loss = loss / args.gradient_accumulation_steps
                scaled_loss.backward()

                pretrain_steps += 1
            else:
                # ----------------------------------------------------------
                # MEMORY MODE: sequential chunk processing with banks.
                # ----------------------------------------------------------
                chunks = next(memory_iter)
                root_model.reset_banks()

                doc_loss = 0.0
                doc_tokens = 0
                chunk_ppls = []

                for chunk_i, chunk in enumerate(chunks):
                    input_ids = chunk["input_ids"].unsqueeze(0).to(device)
                    labels = chunk["labels"].unsqueeze(0).to(device)

                    # Skip padding-only chunks.
                    if (labels != -100).sum() == 0:
                        continue

                    result = ddp_model(
                        input_ids=input_ids, labels=labels, mode="memory",
                    )
                    loss = result["loss"]

                    if not torch.isfinite(loss):
                        if is_main:
                            logger.warning(
                                "[step %d chunk %d] Non-finite memory loss!",
                                global_step, chunk_i,
                            )
                        continue

                    n_tok = (labels != -100).sum().item()
                    chunk_ppl = math.exp(min(loss.item(), 20))  # cap for display
                    chunk_ppls.append(chunk_ppl)
                    doc_loss += loss.item() * n_tok
                    doc_tokens += n_tok

                    # Normalize by chunks_per_doc and accumulate gradients.
                    scaled_loss = loss / args.chunks_per_doc / args.gradient_accumulation_steps
                    scaled_loss.backward()

                accum_loss += doc_loss
                accum_tokens += doc_tokens
                memory_steps += 1

            accum_count += 1

            # ----------------------------------------------------------
            # Gradient step when accumulation is complete.
            # ----------------------------------------------------------
            if accum_count >= args.gradient_accumulation_steps:
                # Gradient clipping.
                torch.nn.utils.clip_grad_norm_(cm_model.parameters(), 1.0)

                # Optimizer step.
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                accum_count = 0

                # Compute PPL for this effective step.
                step_ppl = math.exp(min(accum_loss / max(accum_tokens, 1), 20))
                accum_loss = 0.0
                accum_tokens = 0

                current_lr = scheduler.get_last_lr()[0]

                if is_main and (global_step % 10 == 0 or global_step <= 5):
                    elapsed = time.time() - t0
                    bank_fill = root_model.banks[0].num_filled
                    phase_label = "P2" if bank_fill >= args.num_slots else "P1"
                    logger.info(
                        "[step %d] %s ppl=%.4f lr=%.2e pretrain=%d memory=%d "
                        "bank_fill=%d/%d time=%.1fs",
                        global_step, phase_label, step_ppl, current_lr,
                        pretrain_steps, memory_steps,
                        bank_fill, args.num_slots,
                        elapsed,
                    )

                # ----------------------------------------------------------
                # Periodic evaluation.
                # ----------------------------------------------------------
                if global_step % args.eval_interval == 0:
                    ddp_model.eval()

                    # Vanilla PPL: standard forward, no banks.
                    vanilla_ppl, vanilla_tokens = evaluate_vanilla_ppl(
                        ddp_model, eval_loader, device, pad_token_id, world_size,
                    )
                    if is_main:
                        logger.info(
                            "[EVAL step=%d] vanilla_ppl=%.4f (tokens=%d)",
                            global_step, vanilla_ppl, vanilla_tokens,
                        )

                    # Memory PPL: banks active across chunks.
                    root_model.eval()
                    root_model.reset_banks()
                    mem_total_loss = torch.zeros((), device=device, dtype=torch.float64)
                    mem_total_tokens = torch.zeros((), device=device, dtype=torch.float64)

                    for ei, ebatch in enumerate(eval_loader):
                        e_ids = ebatch["input_ids"].to(device)
                        e_labels = ebatch["labels"].to(device)
                        e_result = root_model.forward_chunk(e_ids, labels=e_labels)
                        e_loss = e_result["loss"].detach()
                        if torch.isfinite(e_loss):
                            n_tok = (e_labels != -100).sum()
                            mem_total_loss += e_loss.double() * n_tok.double()
                            mem_total_tokens += n_tok.double()

                    if world_size > 1:
                        dist.all_reduce(mem_total_loss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(mem_total_tokens, op=dist.ReduceOp.SUM)

                    if mem_total_tokens.item() > 0:
                        mem_avg_loss = (mem_total_loss / mem_total_tokens).item()
                        mem_ppl = math.exp(mem_avg_loss)
                    else:
                        mem_ppl = float("inf")

                    if is_main:
                        logger.info(
                            "[EVAL step=%d] memory_ppl=%.4f  vanilla_ppl=%.4f  ratio=%.4f",
                            global_step, mem_ppl, vanilla_ppl,
                            mem_ppl / max(vanilla_ppl, 1e-8),
                        )

                        # Go/No-Go criteria (from researcher recommendation).
                        if vanilla_ppl > 10.0:
                            logger.warning(
                                "GO/NO-GO: vanilla_ppl=%.4f > 10.0 -- KILL, "
                                "base LM severely degraded!",
                                vanilla_ppl,
                            )
                        if global_step >= 200 and vanilla_ppl > 7.5:
                            logger.warning(
                                "GO/NO-GO: vanilla_ppl=%.4f > 7.5 at step %d -- "
                                "WARNING, base LM degrading",
                                vanilla_ppl, global_step,
                            )
                        if global_step >= 500 and vanilla_ppl > 7.2:
                            logger.warning(
                                "GO/NO-GO: vanilla_ppl=%.4f > 7.2 at step %d -- "
                                "INVESTIGATE: LR may be too high",
                                vanilla_ppl, global_step,
                            )
                        if mem_ppl > vanilla_ppl * 1.5:
                            logger.warning(
                                "GO/NO-GO: memory_ppl=%.4f > vanilla*1.5=%.4f -- "
                                "KILL AND DIAGNOSE",
                                mem_ppl, vanilla_ppl * 1.5,
                            )

                        # Track best vanilla PPL.
                        if vanilla_ppl < best_vanilla_ppl:
                            best_vanilla_ppl = vanilla_ppl
                            logger.info(
                                "[EVAL] New best vanilla_ppl=%.4f at step %d",
                                best_vanilla_ppl, global_step,
                            )

                    ddp_model.train()

                # ----------------------------------------------------------
                # Save checkpoint.
                # ----------------------------------------------------------
                if is_main and global_step % 200 == 0:
                    ckpt_path = os.path.join(args.output_dir, f"step_{global_step}.pt")
                    torch.save(
                        {
                            "global_step": global_step,
                            "model_state_dict": root_model.model.state_dict(),
                        },
                        ckpt_path,
                    )
                    logger.info("Saved checkpoint: %s", ckpt_path)

    # ------------------------------------------------------------------
    # Final evaluation.
    # ------------------------------------------------------------------
    if is_main:
        logger.info("Training complete.  Running final evaluation ...")

    ddp_model.eval()
    vanilla_ppl, _ = evaluate_vanilla_ppl(
        ddp_model, eval_loader, device, pad_token_id, world_size,
    )
    if is_main:
        logger.info("[FINAL] vanilla_ppl=%.4f", vanilla_ppl)

    # Memory-augmented final eval.
    root_model.reset_banks()
    mem_total_loss = torch.zeros((), device=device, dtype=torch.float64)
    mem_total_tokens = torch.zeros((), device=device, dtype=torch.float64)
    for ebatch in eval_loader:
        e_ids = ebatch["input_ids"].to(device)
        e_labels = ebatch["labels"].to(device)
        e_result = root_model.forward_chunk(e_ids, labels=e_labels)
        e_loss = e_result["loss"].detach()
        if torch.isfinite(e_loss):
            n_tok = (e_labels != -100).sum()
            mem_total_loss += e_loss.double() * n_tok.double()
            mem_total_tokens += n_tok.double()

    if world_size > 1:
        dist.all_reduce(mem_total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(mem_total_tokens, op=dist.ReduceOp.SUM)

    if mem_total_tokens.item() > 0:
        mem_avg_loss = (mem_total_loss / mem_total_tokens).item()
        mem_ppl = math.exp(mem_avg_loss)
    else:
        mem_ppl = float("inf")

    # Save final checkpoint.
    if is_main:
        final_path = os.path.join(args.output_dir, "final.pt")
        torch.save(
            {
                "global_step": global_step,
                "model_state_dict": root_model.model.state_dict(),
            },
            final_path,
        )
        logger.info("Saved final checkpoint: %s", final_path)

        # Write summary.
        summary = {
            "global_step": global_step,
            "pretrain_steps": pretrain_steps,
            "memory_steps": memory_steps,
            "vanilla_ppl": vanilla_ppl,
            "memory_ppl": mem_ppl,
            "best_vanilla_ppl": best_vanilla_ppl,
            "num_slots": args.num_slots,
            "top_k": args.top_k,
            "epsilon": args.epsilon,
            "lr": args.lr,
            "max_steps": args.max_steps,
            "pretrain_ratio": args.pretrain_ratio,
            "chunks_per_doc": args.chunks_per_doc,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_steps": args.warmup_steps,
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("[FINAL] vanilla_ppl=%.4f  memory_ppl=%.4f  ratio=%.4f",
                    vanilla_ppl, mem_ppl, mem_ppl / max(vanilla_ppl, 1e-8))

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
