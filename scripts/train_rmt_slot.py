#!/usr/bin/env python3
"""RMT-Slot hybrid training: top-k slot retrieval + RMT sandwich on Llama-3-8B.

Dataloader, eval, and distributed setup forked from train_cross_attn_memory.py.
Model construction uses RMTSlotModel instead of CrossAttentionMemoryModel.
"""
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

from transformers import AutoTokenizer, LlamaForCausalLM
from src.memory.rmt_slot import RMTSlotModel, RMTSlotConfig

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [R%(rank)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Datasets (same as train_cross_attn_memory.py)
# --------------------------------------------------------------------------- #

class DolminoChunkDataset(Dataset):
    """Loads Dolmino shards, reshapes into fixed-size chunks, groups into documents."""

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
        # Return flat token sequence: [chunks_per_doc * seq_len]
        input_ids = torch.tensor(chunks.reshape(-1), dtype=torch.long)
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


def flat_collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "labels": labels}


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #

def init_distributed() -> tuple:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


# --------------------------------------------------------------------------- #
# LR schedule
# --------------------------------------------------------------------------- #

def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


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
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            labels = labels.unsqueeze(0)

        out = root.forward_vanilla(input_ids, labels=labels)
        loss = out["loss"]
        if loss is None or not torch.isfinite(loss):
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
def evaluate_memory_ppl(model, loader, device, world_size, segment_length: int):
    """Compute PPL WITH memory (chunk-by-chunk streaming)."""
    model.eval()
    root = model.module if hasattr(model, "module") else model

    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            labels = labels.unsqueeze(0)

        B, L = input_ids.shape
        root.reset_slots(batch_size=B)

        # Stream through in chunks of segment_length
        for start in range(0, L, segment_length):
            end = min(start + segment_length, L)
            chunk_ids = input_ids[:, start:end]
            chunk_labels = labels[:, start:end]

            out = root.forward_chunk(chunk_ids, labels=chunk_labels)
            loss = out.get("loss")
            if loss is not None and torch.isfinite(loss):
                n_tok = (chunk_labels != -100).sum()
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
# MemLong-protocol NIAH evaluation
# --------------------------------------------------------------------------- #

_MEMLONG_NEEDLE_TEMPLATES = [
    "The special code for this document is {code}. Remember this code.",
    "The secret identifier is {code}. You will need this later.",
    "The hidden passcode in this text is {code}.",
    "A unique reference number {code} is embedded in this document.",
    "The magic number buried in this text is {code}.",
]

_MEMLONG_QUESTION_TEMPLATES = [
    "What is the special code? The special code is",
    "What is the secret identifier? The secret identifier is",
    "What is the hidden passcode? The hidden passcode is",
    "What is the unique reference number? The unique reference number is",
    "What is the magic number? The magic number is",
]

_MEMLONG_HAYSTACK_REPEAT = (
    "The grass is green. The sky is blue. The sun is bright. "
    "We live in a world of many wonders. "
)


def _build_memlong_niah_sample(tokenizer, total_length: int, depth_ratio: float, rng: random.Random):
    """Build a single MemLong-protocol NIAH sample."""
    code = f"{rng.randint(100000, 999999)}"
    template_idx = rng.randint(0, len(_MEMLONG_NEEDLE_TEMPLATES) - 1)
    needle_text = _MEMLONG_NEEDLE_TEMPLATES[template_idx].format(code=code)
    question_text = _MEMLONG_QUESTION_TEMPLATES[template_idx]

    needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    code_tokens = tokenizer.encode(code, add_special_tokens=False)

    repeat_tokens = tokenizer.encode(_MEMLONG_HAYSTACK_REPEAT, add_special_tokens=False)
    num_repeats = total_length // max(len(repeat_tokens), 1) + 2
    haystack = (repeat_tokens * num_repeats)[:total_length]

    insert_pos = int(total_length * depth_ratio)
    end_pos = min(insert_pos + len(needle_tokens), total_length)
    actual_needle_len = end_pos - insert_pos
    haystack[insert_pos:end_pos] = needle_tokens[:actual_needle_len]

    question_with_code = question_tokens + code_tokens
    q_start = max(0, total_length - len(question_with_code))
    haystack[q_start:q_start + len(question_with_code)] = question_with_code[:total_length - q_start]

    input_ids = torch.tensor([haystack], dtype=torch.long)
    return input_ids, code, code_tokens


@torch.no_grad()
def evaluate_memlong_niah(
    model,
    tokenizer,
    device,
    *,
    chunk_size: int,
    lengths=(2048, 4096),
    depths=(0.0, 0.25, 0.5, 0.75, 1.0),
    num_trials: int = 3,
    seed: int = 42,
):
    """MemLong-protocol NIAH accuracy for RMT-Slot model."""
    root = model.module if hasattr(model, "module") else model
    was_training = root.training
    root.eval()

    rng = random.Random(seed)
    breakdown = []
    total_correct = 0
    total_trials = 0

    for length in lengths:
        for depth in depths:
            cell_correct = 0
            cell_loss_sum = 0.0
            for _ in range(num_trials):
                input_ids, code_str, code_tokens = _build_memlong_niah_sample(
                    tokenizer, length, depth, rng,
                )
                input_ids = input_ids.to(device)

                if not code_tokens:
                    continue

                root.reset_slots(batch_size=1)

                n_tok = input_ids.shape[1]
                last_logits = None
                for start in range(0, n_tok, chunk_size):
                    end = min(start + chunk_size, n_tok)
                    chunk = input_ids[:, start:end].contiguous()
                    out = root.forward_chunk(chunk)
                    last_logits = out["logits"]

                if last_logits is None:
                    continue

                final_chunk_len = last_logits.shape[1]
                final_chunk_start = n_tok - final_chunk_len
                answer_global_start = n_tok - len(code_tokens)
                ans_local_start = answer_global_start - final_chunk_start
                pred_start = max(0, ans_local_start - 1)
                pred_slice = last_logits[0, pred_start:pred_start + len(code_tokens)]
                if pred_slice.shape[0] < len(code_tokens):
                    continue
                predicted = pred_slice.argmax(dim=-1).tolist()

                match_tokens = predicted == list(code_tokens)
                if not match_tokens:
                    try:
                        pred_text = tokenizer.decode(predicted, skip_special_tokens=True)
                        match_string = code_str in pred_text
                    except Exception:
                        match_string = False
                else:
                    match_string = True

                is_correct = match_tokens or match_string
                if is_correct:
                    cell_correct += 1
                    total_correct += 1

                target = input_ids[0, answer_global_start:answer_global_start + len(code_tokens)]
                try:
                    loss_cell = nn.functional.cross_entropy(
                        pred_slice.float(), target.to(pred_slice.device), reduction="mean",
                    ).item()
                except Exception:
                    loss_cell = float("nan")
                cell_loss_sum += loss_cell
                total_trials += 1

            acc = cell_correct / max(num_trials, 1)
            avg_loss = cell_loss_sum / max(num_trials, 1)
            breakdown.append({
                "length": length, "depth": depth,
                "accuracy": acc, "avg_loss": avg_loss,
                "correct": cell_correct, "total": num_trials,
            })

    if was_training:
        root.train()

    if total_trials == 0:
        return 0.0, breakdown
    return total_correct / total_trials, breakdown


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RMT-Slot hybrid training")
    # Model
    p.add_argument("--model", type=str, required=True, help="Path to Llama-3-8B weights")
    # Data
    p.add_argument("--shard_dir", type=str, required=True, help="Dolmino shard directory")
    p.add_argument("--num_shards", type=int, default=25)
    p.add_argument("--shard_offset", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=4096, help="Total seq per document sample")
    p.add_argument("--chunks_per_doc", type=int, default=32)
    # Eval data
    p.add_argument("--wikitext_path", type=str, default=None)
    p.add_argument("--niah_data", type=str, default=None)
    p.add_argument("--eval_shards", type=int, default=5)
    # RMT-Slot config
    p.add_argument("--num_slots", type=int, default=64)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--segment_length", type=int, default=1024)
    p.add_argument("--max_n_segments", type=int, default=4)
    p.add_argument("--selector_dim", type=int, default=128)
    p.add_argument("--ema_gate_init", type=float, default=0.3)
    p.add_argument("--slot_value_norm_cap", type=float, default=8.0)
    p.add_argument("--bptt_depth", type=int, default=-1)
    p.add_argument("--vary_n_segments", action="store_true", default=True)
    p.add_argument("--no_vary_n_segments", action="store_true", default=False)
    # Training
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_true", default=False)
    # Eval / save
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--output_dir", type=str, default="outputs/rmt_slot_medium")
    # Misc
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--resume_checkpoint", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()

    if args.no_vary_n_segments:
        args.vary_n_segments = False
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

    # Set seed
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    if is_main:
        logger.info("=" * 60)
        logger.info("RMT-Slot Hybrid Training")
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

    # Build RMT-Slot model
    rmt_config = RMTSlotConfig(
        num_slots=args.num_slots,
        top_k=args.top_k,
        segment_length=args.segment_length,
        max_n_segments=args.max_n_segments,
        selector_dim=args.selector_dim,
        ema_gate_init=args.ema_gate_init,
        slot_value_norm_cap=args.slot_value_norm_cap,
        bptt_depth=args.bptt_depth,
        vary_n_segments=args.vary_n_segments,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    model = RMTSlotModel(base_model, rmt_config).to(device).to(dtype)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main:
        logger.info(
            "Trainable params: %d / %d (%.4f%%)",
            trainable, total, 100.0 * trainable / total,
        )
        logger.info(
            "RMT-Slot config: num_slots=%d, top_k=%d, segment_length=%d, max_n_segments=%d, "
            "ema_gate_init=%.2f, bptt_depth=%d",
            args.num_slots, args.top_k, args.segment_length, args.max_n_segments,
            args.ema_gate_init, args.bptt_depth,
        )

    # DDP
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    root_model = ddp_model.module

    # Resume
    start_step = 0
    if args.resume_checkpoint:
        if is_main:
            logger.info("Resuming from checkpoint: %s", args.resume_checkpoint)
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_step = ckpt.get('global_step', 0)
        if is_main:
            logger.info("Resumed from step %d", start_step)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    # Separate groups: base model (decay/no_decay) + RMT-slot new params (higher lr)
    base_decay = []
    base_no_decay = []
    new_params_decay = []
    new_params_no_decay = []

    new_param_names = {'selector', 'placeholder', 'gate_logit'}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_new = any(n in name for n in new_param_names)
        is_nodecay = param.dim() < 2 or 'norm' in name.lower() or 'bias' in name.lower()
        if is_new:
            if is_nodecay:
                new_params_no_decay.append(param)
            else:
                new_params_decay.append(param)
        else:
            if is_nodecay:
                base_no_decay.append(param)
            else:
                base_decay.append(param)

    new_lr = args.lr * 10  # New params get 10x lr
    optimizer = torch.optim.AdamW([
        {"params": base_decay, "weight_decay": args.weight_decay, "lr": args.lr},
        {"params": base_no_decay, "weight_decay": 0.0, "lr": args.lr},
        {"params": new_params_decay, "weight_decay": args.weight_decay, "lr": new_lr},
        {"params": new_params_no_decay, "weight_decay": 0.0, "lr": new_lr},
    ], lr=args.lr, betas=(0.9, 0.95))

    if is_main:
        n_base = sum(p.numel() for p in base_decay) + sum(p.numel() for p in base_no_decay)
        n_new = sum(p.numel() for p in new_params_decay) + sum(p.numel() for p in new_params_no_decay)
        logger.info(
            "Optimizer: AdamW, base_lr=%.2e, new_params_lr=%.2e, base=%d, new=%d",
            args.lr, new_lr, n_base, n_new,
        )

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    if is_main:
        logger.info("Loading Dolmino data: %d shards from offset %d ...", args.num_shards, args.shard_offset)

    train_ds = DolminoChunkDataset(
        shard_dir=args.shard_dir,
        num_shards=args.num_shards,
        seq_len=args.segment_length,  # Note: segment_length for chunk size
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

    # Eval: wikitext
    wiki_eval_loader = None
    if args.wikitext_path and os.path.exists(args.wikitext_path):
        wiki_ds = FlatChunkDataset(args.wikitext_path, args.segment_length, max_chunks=200)
        wiki_eval_loader = DataLoader(
            wiki_ds, batch_size=4, shuffle=False,
            collate_fn=flat_collate_fn, num_workers=0,
        )
        if is_main:
            logger.info("WikiText eval: %d chunks", len(wiki_ds))

    # Eval: held-out dolmino
    eval_shard_offset = args.shard_offset + args.num_shards
    eval_flat_data = []
    for si in range(eval_shard_offset, eval_shard_offset + args.eval_shards):
        epath = os.path.join(args.shard_dir, f"shard_{si:04d}.npy")
        if os.path.exists(epath):
            edata = np.fromfile(epath, dtype=np.uint32)
            en = len(edata) // args.segment_length
            if en > 0:
                eval_flat_data.append(edata[:en * args.segment_length].reshape(en, args.segment_length).astype(np.int32))
    if eval_flat_data:
        eval_chunks = np.concatenate(eval_flat_data, axis=0)[:2000]
    else:
        eval_chunks = np.zeros((100, args.segment_length), dtype=np.int32)

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
        eval_ds, batch_size=4, shuffle=False,
        collate_fn=flat_collate_fn, num_workers=0,
    )
    if is_main:
        logger.info("Eval chunks: %d from %d held-out shards", len(eval_chunks), args.eval_shards)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    global_step = start_step
    accum_loss = 0.0
    accum_steps = 0
    optimizer.zero_grad()
    start_time = time.time()

    ddp_model.train()
    train_iter = iter(train_loader)

    if is_main:
        logger.info("Starting training from step %d to %d", global_step, args.max_steps)

    while global_step < args.max_steps:
        # Get next document
        try:
            batch = next(train_iter)
        except StopIteration:
            train_sampler.set_epoch(global_step)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            labels = labels.unsqueeze(0)

        # Forward
        outputs = ddp_model(input_ids, labels=labels)
        loss = outputs["loss"]

        if loss is None or not torch.isfinite(loss):
            logger.warning("Step %d: non-finite loss, skipping", global_step)
            continue

        # Scale loss for gradient accumulation
        scaled_loss = loss / args.gradient_accumulation_steps
        scaled_loss.backward()

        accum_loss += loss.item()
        accum_steps += 1

        if accum_steps >= args.gradient_accumulation_steps:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # LR schedule
            lr = get_lr(global_step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
            for pg in optimizer.param_groups:
                if pg["lr"] > args.lr:
                    # new params group — scale proportionally
                    pg["lr"] = lr * 10
                else:
                    pg["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()

            avg_loss = accum_loss / accum_steps

            if is_main and global_step % 10 == 0:
                elapsed = time.time() - start_time
                gate_val = torch.sigmoid(root_model.gate_logit).item()
                logger.info(
                    "Step %d | loss=%.4f | lr=%.2e | gate=%.3f | elapsed=%.1fs",
                    global_step, avg_loss, lr, gate_val, elapsed,
                )

            accum_loss = 0.0
            accum_steps = 0
            global_step += 1

            # ----------------------------------------------------------
            # Eval
            # ----------------------------------------------------------
            if global_step % args.eval_interval == 0 and is_main:
                logger.info("=" * 40 + " EVAL step %d " + "=" * 40, global_step)
                elapsed_s = time.time() - start_time

                # Vanilla PPL on held-out dolmino
                vanilla_ppl, _ = evaluate_vanilla_ppl(ddp_model, eval_loader, device, world_size)

                # Memory PPL on held-out dolmino
                memory_ppl, _ = evaluate_memory_ppl(
                    ddp_model, eval_loader, device, world_size,
                    segment_length=args.segment_length,
                )

                # WikiText PPL
                wiki_ppl = float("inf")
                if wiki_eval_loader is not None:
                    wiki_ppl, _ = evaluate_vanilla_ppl(ddp_model, wiki_eval_loader, device, world_size)

                # Memory ratio
                memory_ratio = memory_ppl / max(vanilla_ppl, 1e-6)

                # NIAH eval
                niah_correct = 0
                niah_total = 0
                niah_avg_loss = 0.0
                if args.niah_data and os.path.exists(args.niah_data):
                    niah_acc, niah_breakdown = evaluate_memlong_niah(
                        ddp_model, tokenizer, device,
                        chunk_size=args.segment_length,
                        lengths=(2048, 4096),
                        depths=(0.0, 0.25, 0.5, 0.75),
                        num_trials=2,
                    )
                    for cell in niah_breakdown:
                        niah_correct += cell["correct"]
                        niah_total += cell["total"]
                        niah_avg_loss += cell["avg_loss"]
                    if niah_breakdown:
                        niah_avg_loss /= len(niah_breakdown)

                logger.info(
                    "EVAL | vanilla_ppl=%.2f | memory_ppl=%.2f | ratio=%.4f | "
                    "wiki_ppl=%.2f | niah=%d/%d | gate=%.3f",
                    vanilla_ppl, memory_ppl, memory_ratio,
                    wiki_ppl, niah_correct, niah_total,
                    torch.sigmoid(root_model.gate_logit).item(),
                )

                # Write metrics
                metrics = {
                    "step": global_step,
                    "vanilla_ppl": vanilla_ppl,
                    "memory_ppl": memory_ppl,
                    "memory_ratio": memory_ratio,
                    "wiki_ppl": wiki_ppl,
                    "base_vanilla_ppl": vanilla_ppl,
                    "niah_correct": niah_correct,
                    "niah_total": niah_total,
                    "niah_avg_loss": niah_avg_loss,
                    "curriculum_phase": "full",
                    "elapsed_s": elapsed_s,
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")

                ddp_model.train()

            # ----------------------------------------------------------
            # Save checkpoint
            # ----------------------------------------------------------
            if global_step % args.save_interval == 0 and is_main:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_step{global_step}.pt")
                torch.save({
                    "model_state_dict": root_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "args": vars(args),
                }, ckpt_path)
                logger.info("Saved checkpoint: %s", ckpt_path)

    # Final save
    if is_main:
        final_path = os.path.join(args.output_dir, "final_checkpoint.pt")
        torch.save({
            "model_state_dict": root_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "args": vars(args),
        }, final_path)
        logger.info("Training complete. Final checkpoint: %s", final_path)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
