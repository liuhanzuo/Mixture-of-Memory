#!/usr/bin/env python3
"""Sliding-window-only baseline: standard Llama fine-tuning with sliding window mask.

No SparseMemory wrapper. Vanilla LlamaForCausalLM with default SDPA attention.
Patches LlamaModel.forward to pass and_mask_function to create_causal_mask,
using transformers 5.x native sliding_window_causal_mask_function.

This isolates the effect of window restriction alone (without memory) vs full attention.
"""
import os, sys, math, time, argparse, logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass, field
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WINDOW_SIZE = 256
def patch_llama_model_for_window(window: int):
    """Patch create_causal_mask to add and_mask_function with sliding window.

    Uses the transformers native sliding_window_causal_mask_function.
    This works with any attention backend (SDPA, eager, flex).
    """
    from transformers.masking_utils import sliding_window_causal_mask_function
    window_fn = sliding_window_causal_mask_function(window)

    import transformers.models.llama.modeling_llama as _m
    _orig_create = _m.create_causal_mask

    def _patched_create(*args, **kwargs):
        # Only inject if not already set
        if 'and_mask_function' not in kwargs or kwargs['and_mask_function'] is None:
            kwargs['and_mask_function'] = window_fn
        return _orig_create(*args, **kwargs)

    _m.create_causal_mask = _patched_create
    logger.info(f"Patched create_causal_mask with sliding_window={window}")


# ── Dataset ──────────────────────────────────────────────────────────────────
@dataclass
class PreTokenizedDataset(Dataset):
    path: str
    max_seq_len: int = 4096
    _data: object = field(default=None, repr=False)

    def __post_init__(self):
        import numpy as np
        arr = np.load(self.path)
        self._data = torch.from_numpy(arr).long()
        assert self._data.shape[1] == self.max_seq_len

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        tokens = self._data[idx]
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sliding-window-only baseline")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--sliding_window", type=int, default=256)
    args = parser.parse_args()

    # Patch BEFORE loading model
    patch_llama_model_for_window(args.sliding_window)

    # DDP setup
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if is_main:
        logger.info(f"=== Sliding-Window-Only Baseline (no memory) ===")
        logger.info(f"Window size: {args.sliding_window}")
        logger.info(f"Ranks: {world_size}, Batch: {args.batch_size}, GradAccum: {args.grad_accumulation_steps}")
        logger.info(f"Effective batch size: {args.batch_size * args.grad_accumulation_steps * world_size}")
        os.makedirs(args.output_dir, exist_ok=True)

    # Patch LlamaModel.forward to inject causal+window mask as attention_mask
    # (done via create_causal_mask patch above — no additional LlamaModel patch needed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model — default SDPA attention, mask injected via create_causal_mask patch
    if is_main:
        logger.info(f"Loading base model from {args.base_model} (default attention)...")
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    ).to(device)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model size: {n_params * 2 / 1e9:.2f} GB")

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Dataset
    dataset = PreTokenizedDataset(path=args.data_path, max_seq_len=args.seq_len)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collate_fn, num_workers=2, pin_memory=True, drop_last=True,
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1)

    if is_main:
        logger.info(f"Starting training for {args.max_steps} steps...")

    # Training loop
    model.train()
    step = 0
    micro_step = 0
    accum_loss = 0.0
    accum_count = 0
    optimizer.zero_grad()
    data_iter = iter(dataloader)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            sampler.set_epoch(sampler.epoch + 1)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / args.grad_accumulation_steps

        loss.backward()
        accum_loss += loss.item() * args.grad_accumulation_steps
        accum_count += 1
        micro_step += 1

        if micro_step % args.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if is_main and step % args.log_every == 0:
                avg_loss = accum_loss / accum_count
                lr_val = scheduler.get_last_lr()[0]
                logger.info(f"[step {step:5d}] loss={avg_loss:.4f}  lr={lr_val:.2e}")
                accum_loss = 0.0
                accum_count = 0

            if is_main and step % args.save_every == 0 and step > 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                unwrapped = model.module
                unwrapped.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logger.info(f"Saved checkpoint to {save_path}")

    # Final save
    if is_main:
        save_path = os.path.join(args.output_dir, "final")
        unwrapped = model.module
        unwrapped.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Training complete. Final model saved to {save_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
