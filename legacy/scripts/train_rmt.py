"""
RMT Training Script for Qwen3-8B.

Processes long documents in segments with recurrent memory tokens.
Uses inputs_embeds to inject memory embeddings and custom attention masks
for bidirectional memory attention + causal segment attention.
"""

import os
import sys
import json
import time
import math
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.rmt.rmt_module import RMTMemory, RMTModel


class RMTDataset(Dataset):
    """Dataset for RMT training with long documents."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        segment_length: int = 2048,
        max_segments: int = 6,
    ):
        self.tokenizer = tokenizer
        self.segment_length = segment_length
        self.max_segments = max_segments
        self.max_total_tokens = segment_length * max_segments
        
        print(f"Loading data from {data_path}...")
        self.tokenized_docs = []
        with open(data_path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                tokens = tokenizer.encode(doc['text'], add_special_tokens=False)
                if len(tokens) >= segment_length:
                    # Truncate to max_segments, then pad to fixed length
                    num_segs = min(len(tokens) // segment_length, max_segments)
                    tokens = tokens[:num_segs * segment_length]
                    # Pad to exactly max_segments * segment_length (static graph for DDP)
                    target_len = max_segments * segment_length
                    if len(tokens) < target_len:
                        tokens = tokens + [tokenizer.pad_token_id] * (target_len - len(tokens))
                    self.tokenized_docs.append(tokens)
        
        print(f"Dataset: {len(self.tokenized_docs)} documents")
    
    def __len__(self):
        return len(self.tokenized_docs)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_docs[idx]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        num_segments = len(tokens) // self.segment_length
        return {"input_ids": input_ids, "labels": labels, "num_segments": num_segments}


def collate_fn(batch, segment_length: int, pad_token_id: int):
    max_segs = max(item["num_segments"] for item in batch)
    max_len = max_segs * segment_length
    
    input_ids_list = []
    labels_list = []
    num_segments_list = []
    
    for item in batch:
        ids = item["input_ids"]
        labs = item["labels"]
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            labs = torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)])
        input_ids_list.append(ids)
        labels_list.append(labs)
        num_segments_list.append(item["num_segments"])
    
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "num_segments": num_segments_list,
    }


class _NoSync:
    """No-op context manager for non-DDP mode."""
    def __enter__(self): return self
    def __exit__(self, *a): pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # RMT config
    parser.add_argument("--num_memory_tokens", type=int, default=8)
    parser.add_argument("--segment_length", type=int, default=1024)
    parser.add_argument("--max_segments", type=int, default=6)
    parser.add_argument("--num_memory_heads", type=int, default=8)  # kept for compat, unused in v2
    parser.add_argument("--bottleneck_dim", type=int, default=32)
    
    # Training config
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rmt_lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # System config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    args = parser.parse_args()
    
    # Setup distributed (optional)
    use_ddp = os.environ.get("RANK") is not None
    if use_ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        world_size = 1
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        # Save config
        with open(os.path.join(args.output_dir, "rmt_config.json"), 'w') as f:
            json.dump(vars(args), f, indent=2)
        # Setup heartbeat
        def update_heartbeat(status, progress, **kwargs):
            hb = {
                "timestamp": datetime.now().isoformat(),
                "status": status,
                "progress": progress,
                "metrics": kwargs,
                "extra": {},
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(os.path.join(args.output_dir, "heartbeat.json"), 'w') as f:
                json.dump(hb, f, indent=2)
        print(f"[RMT] world_size={world_size}, Config: {vars(args)}")
    else:
        update_heartbeat = lambda *a, **kw: None
    
    torch.manual_seed(args.seed + local_rank)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    hidden_dim = model.config.hidden_size
    
    # Initialize RMT memory module
    rmt_memory = RMTMemory(
        hidden_dim=hidden_dim,
        num_memory_tokens=args.num_memory_tokens,
        num_heads=args.num_memory_heads,
        max_segments=args.max_segments + 1,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device=device, dtype=torch.bfloat16)
    
    rmt_params = sum(p.numel() for p in rmt_memory.parameters())
    if local_rank == 0:
        print(f"[RMT] Memory module params: {rmt_params:,}")
    
    # Wrap in RMTModel
    rmt_model = RMTModel(model, rmt_memory, segment_length=args.segment_length)
    rmt_model = rmt_model.to(device=device, dtype=torch.bfloat16)
    
    # DDP (only if multi-GPU)
    # static_graph=True is required because forward() calls backward() multiple
    # times (once per segment) — DDP needs to know the graph is deterministic.
    if use_ddp:
        rmt_model = DDP(rmt_model, device_ids=[local_rank], static_graph=True)
        is_ddp = True
    else:
        is_ddp = False

    def get_no_sync():
        """Return no_sync context if DDP, otherwise a no-op."""
        if is_ddp:
            return rmt_model.no_sync()
        return _NoSync()
    
    # Dataset
    dataset = RMTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        segment_length=args.segment_length,
        max_segments=args.max_segments,
    )
    
    if use_ddp:
        sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
    
    def collate_with_config(batch):
        return collate_fn(batch, args.segment_length, tokenizer.pad_token_id)
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collate_with_config, num_workers=0, drop_last=True,
    )
    
    # Optimizer: separate LR for model params and RMT params
    if is_ddp:
        base_model_params = [p for p in rmt_model.module.model.parameters() if p.requires_grad]
        rmt_params_list = list(rmt_model.module.rmt.parameters())
    else:
        base_model_params = [p for p in rmt_model.model.parameters() if p.requires_grad]
        rmt_params_list = list(rmt_model.rmt.parameters())
    optimizer_grouped = [
        {"params": base_model_params, "lr": args.lr},
        {"params": rmt_params_list, "lr": args.rmt_lr},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped, weight_decay=args.weight_decay)
    
    # With DistributedSampler, len(dataloader) is per-GPU batches, not total
    total_steps = len(dataloader) * args.num_epochs // args.grad_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    
    if local_rank == 0:
        print(f"[RMT] Total steps: {total_steps}")
    
    # Training loop
    global_step = 0
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch + args.seed)
        rmt_model.train()
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            B, L = input_ids.shape
            
            # Forward (per-segment backward happens inside)
            # static_graph=True handles multiple backward passes per step.
            # no_sync() for gradient accumulation (skip sync on non-final micro-steps).
            is_accum_step = (batch_idx + 1) % args.grad_accumulation_steps != 0
            sync_ctx = get_no_sync() if is_accum_step else _NoSync()
            with sync_ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = rmt_model(input_ids, labels)
            
            # Gradients already accumulated by per-segment backward inside forward()
            # Scale loss for logging
            loss_val = loss.item()
            
            if (batch_idx + 1) % args.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(rmt_model.parameters()), args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if local_rank == 0 and global_step % args.log_every == 0:
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    progress = (epoch * len(dataloader) + batch_idx + 1) / (args.num_epochs * len(dataloader))
                    print(
                        f"[RMT] Epoch {epoch+1} [{global_step}/{total_steps}] "
                        f"loss={loss_val:.4f} lr={lr:.2e} "
                        f"loss={loss_val:.4f} lr={lr:.2e} "
                        f"mem={args.num_memory_tokens} "
                        f"ETA={elapsed/global_step*(total_steps-global_step)/3600:.1f}h"
                    )
                    update_heartbeat(
                        "running", progress,
                        loss=loss_val, learning_rate=lr,
                        epoch=epoch+1, step=global_step,
                    )
                
                if local_rank == 0 and global_step % args.save_every == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint_{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    _m = rmt_model.module if is_ddp else rmt_model
                    _m.model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    torch.save(_m.rmt.state_dict(),
                               os.path.join(save_path, "rmt_memory.pt"))
        
        if local_rank == 0:
            elapsed = time.time() - start_time
            avg_loss = 0  # simplified
            print(f"[RMT] Epoch {epoch+1} done, elapsed: {elapsed/3600:.1f}h")
    
    # Final save
    if local_rank == 0:
        save_path = os.path.join(args.output_dir, "final")
        os.makedirs(save_path, exist_ok=True)
        _m = rmt_model.module if is_ddp else rmt_model
        _m.model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        torch.save(_m.rmt.state_dict(), os.path.join(save_path, "rmt_memory.pt"))
        
        update_heartbeat(
            "completed", 1.0,
            training_time_s=time.time() - start_time,
        )
        print(f"[RMT] Training complete!")
    
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
