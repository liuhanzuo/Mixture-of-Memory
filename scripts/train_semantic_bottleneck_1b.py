#!/usr/bin/env python3
"""Train a 1B semantic-bottleneck Llama from scratch (feasibility experiment).

Two arms share ``scripts/semantic_bottleneck_model.py``:
  * baseline  : --bottleneck_dim 0            (standard from-scratch 1B Llama)
  * bottleneck: --bottleneck_dim 512 --bottleneck_layer 6

Pure next-token prediction on pre-tokenised chunks (slimpajama / pg19 .npy,
uint32, [N, seq_len], Llama-3 tokenizer). Multi-GPU DDP (torchrun). Checkpoints
are raw ``state_dict`` (+ arch meta json) so the probe script can rebuild the
exact arch and load weights regardless of the custom BottleneckLayer.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from semantic_bottleneck_model import build_bottleneck_model, make_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NpyChunkDataset(Dataset):
    """mmap pre-tokenised [N, seq_len] uint32 chunks; slice to --seq_len."""

    def __init__(self, path: str, seq_len: int):
        self.arr = np.load(path, mmap_mode="r")
        assert self.arr.ndim == 2, self.arr.shape
        self.seq_len = min(seq_len, self.arr.shape[1])

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        row = np.asarray(self.arr[idx, : self.seq_len]).astype(np.int64)
        tokens = torch.from_numpy(row)
        # No pre-shift: LlamaForCausalLM(labels=...) does its own internal shift.
        return {"input_ids": tokens, "labels": tokens.clone()}


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def get_lr(step, warmup, max_steps, base_lr, min_lr):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    prog = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(prog, 1.0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--tokenizer_path", type=str, default="models/Meta-Llama-3-8B")
    p.add_argument("--model_size", type=str, default="1b", choices=["1b", "3b", "7b"],
                   help="Llama shape to build from scratch")
    p.add_argument("--bottleneck_layer", type=int, default=6)
    p.add_argument("--bottleneck_dim", type=int, default=0, help="0 = baseline no bottleneck")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accumulation_steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--max_rows", type=int, default=0, help=">0 to subset dataset (smoke)")
    args = p.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    arm = "baseline" if args.bottleneck_dim <= 0 else f"bottleneck_d{args.bottleneck_dim}@L{args.bottleneck_layer}"
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        eff_bs = args.batch_size * args.grad_accumulation_steps * world_size
        logger.info(f"=== {args.model_size} semantic-bottleneck pretrain [{arm}] ===")
        logger.info(f"world_size={world_size} bs={args.batch_size} gaccum={args.grad_accumulation_steps} "
                    f"eff_bs={eff_bs} seq_len={args.seq_len} lr={args.lr} max_steps={args.max_steps}")

    cfg = make_config(args.model_size, seq_len=args.seq_len)
    model = build_bottleneck_model(
        bottleneck_layer=args.bottleneck_layer,
        bottleneck_dim=args.bottleneck_dim,
        vocab_size=cfg.vocab_size,
        seq_len=args.seq_len,
        dtype=torch.bfloat16,
        size=args.model_size,
    ).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if is_main:
        n = sum(pp.numel() for pp in model.parameters())
        logger.info(f"model params = {n/1e9:.4f}B")
        with open(os.path.join(args.output_dir, "arch_meta.json"), "w") as f:
            json.dump({
                "model_size": args.model_size,
                "bottleneck_layer": args.bottleneck_layer,
                "bottleneck_dim": args.bottleneck_dim,
                "seq_len": args.seq_len,
                "vocab_size": cfg.vocab_size,
                "num_hidden_layers": cfg.num_hidden_layers,
                "hidden_size": cfg.hidden_size,
                "n_params": n,
            }, f, indent=2)

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    ds = NpyChunkDataset(args.data_path, args.seq_len)
    if args.max_rows and args.max_rows > 0:
        ds.arr = ds.arr[: args.max_rows]
    if is_main:
        logger.info(f"dataset rows={len(ds)} seq_len={ds.seq_len} from {args.data_path}")

    if ddp:
        sampler = DistributedSampler(ds, shuffle=True)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True)
    else:
        sampler = None
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True)

    decay, no_decay = [], []
    for nm, pp in model.named_parameters():
        if not pp.requires_grad:
            continue
        (no_decay if pp.ndim < 2 else decay).append(pp)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    model.train()
    step = 0
    micro = 0
    accum_loss = 0.0
    accum_cnt = 0
    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(loader)
    t0 = time.time()
    epoch = 0

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        is_accum_boundary = (micro + 1) % args.grad_accumulation_steps == 0
        sync_ctx = model.no_sync() if (ddp and not is_accum_boundary) else _nullctx()
        with sync_ctx:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss / args.grad_accumulation_steps
            loss.backward()
        accum_loss += loss.item() * args.grad_accumulation_steps
        accum_cnt += 1
        micro += 1

        if is_accum_boundary:
            lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if is_main and step % args.log_every == 0:
                avg = accum_loss / max(accum_cnt, 1)
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 1e9
                logger.info(f"[step {step:5d}/{args.max_steps}] loss={avg:.4f} ppl={math.exp(min(avg,20)):.2f} "
                            f"lr={lr:.2e} gnorm={float(gnorm):.2f} {dt/args.log_every:.2f}s/step "
                            f"maxmem={mem:.1f}GB")
                accum_loss = 0.0
                accum_cnt = 0
                t0 = time.time()

            if is_main and step % args.save_every == 0 and step > 0:
                _save(model, args, step)

    if is_main:
        _save(model, args, step, final=True)
        logger.info(f"DONE [{arm}] at step {step}")
    if ddp:
        dist.destroy_process_group()


class _nullctx:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def _save(model, args, step, final=False):
    root = model.module if hasattr(model, "module") else model
    name = "final" if final else f"step{step}"
    path = os.path.join(args.output_dir, f"{name}.pt")
    torch.save({"model_state": root.state_dict(),
                "step": step,
                "model_size": args.model_size,
                "bottleneck_layer": args.bottleneck_layer,
                "bottleneck_dim": args.bottleneck_dim,
                "seq_len": args.seq_len}, path)
    logger.info(f"saved {path}")


if __name__ == "__main__":
    main()
