#!/usr/bin/env python3
"""Activation Beacon Training — streaming AR with random compression ratio.

Reference: arXiv 2401.03462 (Activation Beacon)

Key design:
    1. Base model fully frozen, only beacon projections + embedding trained
    2. Random compression ratio per step (ratio in {2, 4, 8, 16, 32, 64, 128})
    3. Streaming causal: interval i tokens attend [historical beacon KV] + [interval i causal]
    4. Dense loss: every token contributes next-token CE

Data: Reuses Dolmino pre-tokenized 1024-token chunks, concatenated into long sequences.
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
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, LlamaForCausalLM  # noqa: E402

from src.memory.mem_space.beacon import BeaconModel  # noqa: E402
from src.memory.mem_space.beacon_patch import (  # noqa: E402
    apply_beacon_to_model,
    count_beacon_params,
    freeze_base_unfreeze_beacon,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Cosine LR Schedule with Warmup
# --------------------------------------------------------------------------- #


def cosine_lr_schedule(step: int, total_steps: int, warmup_steps: int,
                       base_lr: float, min_lr_ratio: float = 0.1) -> float:
    """Compute learning rate with linear warmup + cosine decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #


def init_distributed() -> Tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size,
                            timeout=timedelta(minutes=30))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------- #
# Data: streaming long sequences from Dolmino chunks
# --------------------------------------------------------------------------- #


class BeaconStreamingDataset(torch.utils.data.IterableDataset):
    """Concatenates Dolmino 1024-token chunks into long sequences for beacon training.

    Each yielded sample is a dict with:
        input_ids: [seq_len] long tensor (concatenation of multiple chunks)
        labels:    [seq_len] long tensor (same as input_ids for dense CE)
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 8192,
        chunk_size: int = 1024,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = data_path
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

        import datasets
        self._ds = datasets.load_from_disk(data_path)
        self._num_samples = len(self._ds)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        epoch = 0
        while True:
            rng = random.Random(self.seed + epoch * 10007 + self.rank * 1000 + worker_id)
            indices = list(range(self._num_samples))
            rng.shuffle(indices)

            total_consumers = self.world_size * num_workers
            consumer_id = self.rank * num_workers + worker_id
            my_indices = indices[consumer_id::total_consumers]

            # Concatenate chunks into long sequences
            chunks_per_seq = self.seq_len // self.chunk_size
            ptr = 0
            while ptr + chunks_per_seq <= len(my_indices):
                token_ids = []
                for i in range(chunks_per_seq):
                    row = self._ds[my_indices[ptr + i]]
                    token_ids.extend(row["input_ids"][:self.chunk_size])
                ptr += chunks_per_seq

                ids_tensor = torch.tensor(token_ids[:self.seq_len], dtype=torch.long)
                yield {"input_ids": ids_tensor, "labels": ids_tensor.clone()}

            epoch += 1


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Activation Beacon Training")

    # Model
    p.add_argument("--model_path", type=str, default="models/Meta-Llama-3-8B",
                   help="Path to base Llama model directory")
    p.add_argument("--output_dir", type=str, required=True)

    # Data
    p.add_argument("--dolmino_path", type=str,
                   default="MemLong/data/processed/dolmino_0.5B_1024/train",
                   help="Path to pre-tokenised Dolmino Arrow dataset.")
    p.add_argument("--seq_len", type=int, default=8192,
                   help="Total sequence length per sample (concatenated chunks).")
    p.add_argument("--chunk_size", type=int, default=1024,
                   help="Token count per Dolmino chunk.")

    # Beacon config
    p.add_argument("--n_beacon", type=int, default=4,
                   help="Number of beacon tokens per interval boundary.")
    p.add_argument("--compression_ratios", type=str, default="2,4,8,16,32,64,128",
                   help="Comma-separated compression ratios to randomly sample from.")

    # Training
    p.add_argument("--total_steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Peak learning rate for beacon params.")
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Per-rank batch size.")
    p.add_argument("--start_step", type=int, default=0)

    # Logging / saving
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--wandb_project", type=str, default="mixture-of-memory")
    p.add_argument("--wandb_run_name", type=str, default=None)

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #


def main():
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Parse compression ratios
    ratios = [int(r) for r in args.compression_ratios.split(",")]
    if is_main(rank):
        logger.info(f"Compression ratios: {ratios}")
        logger.info(f"N beacon: {args.n_beacon}, Seq len: {args.seq_len}")

    # --- Load base model ---
    if is_main(rank):
        logger.info(f"Loading base model from {args.model_path}")
    base_model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)

    # --- Apply beacon patch ---
    _, beacon_model = apply_beacon_to_model(
        base_model, n_beacon=args.n_beacon, interval_size=args.seq_len // 2
    )
    beacon_model = beacon_model.to(device)

    if is_main(rank):
        param_info = count_beacon_params(beacon_model)
        logger.info(
            f"Beacon params: {param_info['trainable']:,} trainable / "
            f"{param_info['total']:,} total ({param_info['beacon_ratio']:.4%})"
        )

    # --- DDP wrap (only beacon layers, base is frozen) ---
    if world_size > 1:
        beacon_model = DDP(
            beacon_model, device_ids=[local_rank],
            find_unused_parameters=True,
        )

    # --- Optimizer (only beacon params) ---
    raw_model = beacon_model.module if hasattr(beacon_model, "module") else beacon_model
    trainable_params = raw_model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # --- Dataset ---
    dataset = BeaconStreamingDataset(
        data_path=args.dolmino_path,
        seq_len=args.seq_len,
        chunk_size=args.chunk_size,
        rank=rank,
        world_size=world_size,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True,
    )
    data_iter = iter(dataloader)

    # --- Wandb ---
    if is_main(rank) and _WANDB_AVAILABLE:
        run_name = args.wandb_run_name or f"beacon_n{args.n_beacon}_seq{args.seq_len}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # --- Output dir ---
    os.makedirs(args.output_dir, exist_ok=True)
    if is_main(rank):
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # --- Training loop ---
    beacon_model.train()
    optimizer.zero_grad()
    step = args.start_step
    accum_loss = 0.0
    accum_tokens = 0
    t0 = time.time()

    if is_main(rank):
        logger.info(f"Starting training from step {step}, total {args.total_steps}")

    while step < args.total_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)  # [B, seq_len]
        labels = batch["labels"].to(device)

        # Random compression ratio per step
        ratio = random.choice(ratios)

        # Forward
        fwd_model = beacon_model.module if hasattr(beacon_model, "module") else beacon_model
        output = fwd_model.forward_streaming(
            input_ids=input_ids,
            labels=labels,
            compression_ratio=ratio,
        )
        loss = output["loss"] / args.gradient_accumulation_steps
        loss.backward()

        accum_loss += output["loss"].item()
        accum_tokens += input_ids.numel()

        # Gradient accumulation step
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # LR schedule
            lr = cosine_lr_schedule(
                step // args.gradient_accumulation_steps,
                args.total_steps // args.gradient_accumulation_steps,
                args.warmup_steps // args.gradient_accumulation_steps,
                args.lr,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        step += 1

        # Logging
        if is_main(rank) and step % args.log_interval == 0:
            elapsed = time.time() - t0
            avg_loss = accum_loss / args.log_interval
            tps = accum_tokens / elapsed
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"step={step} loss={avg_loss:.4f} ratio={ratio} "
                f"lr={current_lr:.2e} tok/s={tps:.0f} "
                f"intervals={output['n_intervals']}"
            )
            if _WANDB_AVAILABLE:
                wandb.log({
                    "loss": avg_loss,
                    "compression_ratio": ratio,
                    "lr": current_lr,
                    "tokens_per_sec": tps,
                    "n_intervals": output["n_intervals"],
                    "step": step,
                })
            accum_loss = 0.0
            accum_tokens = 0
            t0 = time.time()

        # Save checkpoint
        if is_main(rank) and step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"beacon_step{step}.pt")
            save_model = beacon_model.module if hasattr(beacon_model, "module") else beacon_model
            # Save only beacon params (not the frozen base)
            beacon_state = {
                "beacon_embedding": save_model.beacon_embedding.data,
                "beacon_layers": save_model.beacon_layers.state_dict(),
                "step": step,
                "args": vars(args),
            }
            torch.save(beacon_state, save_path)
            logger.info(f"Saved checkpoint: {save_path}")

    # Final save
    if is_main(rank):
        save_path = os.path.join(args.output_dir, "beacon_final.pt")
        save_model = beacon_model.module if hasattr(beacon_model, "module") else beacon_model
        beacon_state = {
            "beacon_embedding": save_model.beacon_embedding.data,
            "beacon_layers": save_model.beacon_layers.state_dict(),
            "step": step,
            "args": vars(args),
        }
        torch.save(beacon_state, save_path)
        logger.info(f"Training complete. Final checkpoint: {save_path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
