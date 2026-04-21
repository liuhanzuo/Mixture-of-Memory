#!/usr/bin/env python3
"""Training script for SparseMemoryLlamaForCausalLM.

Full fine-tuning of Llama2-7B with sliding window + sparse memory retrieval.
Supports DDP via torchrun, gradient checkpointing, and DeepSpeed ZeRO-3 / FSDP.

Usage:
    torchrun --nnodes=1 --nproc_per_node=8 scripts/train_sparse_memory.py \
        --base_model /path/to/Llama2-7b \
        --data_path /path/to/train.jsonl \
        --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.memory.sparse_memory.model import SparseMemoryLlamaForCausalLM


# ── Dataset ────────────────────────────────────────────────────────────────

@dataclass
class PreTokenizedDataset(Dataset):
    """Load pre-tokenized chunks from numpy file (fast loading, no re-tokenization).

    Expects a numpy array of shape [N, seq_len] with dtype uint16.
    """
    path: str
    max_seq_len: int = 4096
    vocab_size: int = 32000  # clamp OOB tokens to this range
    _data: object = field(default=None, repr=False)

    def __post_init__(self):
        import numpy as np
        arr = np.load(self.path)  # shape [N, seq_len]
        self._data = torch.from_numpy(arr).long()  # [N, seq_len]
        assert self._data.shape[1] == self.max_seq_len, \
            f"Expected seq_len={self.max_seq_len}, got {self._data.shape[1]}"
        # Clamp token IDs to valid vocab range to prevent CUDA OOB in embed_tokens.
        # Can happen when data was tokenized with a different tokenizer (e.g. Qwen vs Llama2).
        # Masked tokens become pad_id=0 (neutral for causal LM).
        vocab_size = getattr(self, 'vocab_size', 32000)
        oob_mask = self._data >= vocab_size
        if oob_mask.any():
            n_oob = oob_mask.sum().item()
            total = self._data.numel()
            print(f"[PreTokenizedDataset] WARNING: {n_oob}/{total} tokens ({100*n_oob/total:.2f}%) "
                  f"exceed vocab_size={vocab_size}, clamping to 0")
            self._data = self._data.clamp(0, vocab_size - 1)

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        tokens = self._data[idx]
        # Shift for causal LM
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


class JsonlDataset(Dataset):
    """Load and tokenize documents from a .jsonl file on-the-fly.

    Each line should be: {"text": "..."}
    Documents are tokenized, truncated/padded to seq_len, and stored.
    Compatible with v5 training scripts.
    """
    def __init__(self, data_path, tokenizer, seq_len=4096):
        self.seq_len = seq_len
        docs = []
        with open(data_path) as f:
            for line in f:
                text = json.loads(line)["text"]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) >= 64:  # skip very short docs
                    docs.append(tokens)
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        tokens = self.docs[idx][:self.seq_len]
        t = torch.tensor(tokens, dtype=torch.long)
        # Shift for causal LM
        return {
            "input_ids": t[:-1],
            "labels": t[1:],
        }


def collate_fn(batch):
    """Stack batch with padding to longest sequence in the batch."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    pad_id = 0
    input_ids = torch.stack([
        torch.cat([item["input_ids"], torch.full([max_len - item["input_ids"].shape[0]], pad_id, dtype=torch.long)])
        for item in batch
    ])
    labels = torch.stack([
        torch.cat([item["labels"], torch.full([max_len - item["labels"].shape[0]], -100, dtype=torch.long)])
        for item in batch
    ])
    return {"input_ids": input_ids, "labels": labels}


# ── Argument Parsing ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train SparseMemoryLlamaForCausalLM")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/sparse_memory")
    parser.add_argument("--data_path", type=str, required=True)

    # Memory hyperparameters
    parser.add_argument("--memory_slots", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--sliding_window", type=int, default=256)
    parser.add_argument("--ema_alpha", type=float, default=0.1)

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Mixed precision
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "none"],
                        help="Mixed precision mode: bf16 (recommended), fp16 (needs GradScaler), none (fp32)")

    # L0/L1 multi-level memory
    parser.add_argument("--use_l1", action="store_true",
                        help="Enable L1 (higher-level) memory layer for multi-level memory compression")
    parser.add_argument("--num_mem_tokens", type=int, default=16,
                        help="Number of L0 memory tokens per segment")
    parser.add_argument("--l1_num_tokens", type=int, default=16,
                        help="Number of L1 memory tokens (compressed from L0)")
    parser.add_argument("--segment_length", type=int, default=1024,
                        help="Tokens per segment before memory compression")
    parser.add_argument("--max_segments", type=int, default=4,
                        help="Maximum segments per document")
    parser.add_argument("--bptt_depth", type=int, default=2,
                        help="How many segments of BPTT backprop to unroll")
    parser.add_argument("--recon_loss_coef", type=float, default=0.1,
                        help="Weight for memory reconstruction loss")
    parser.add_argument("--rmt_lr", type=float, default=None,
                        help="Separate LR for memory/RMT params (overrides per-param-group defaults)")
    parser.add_argument("--use_importance_routing", action="store_true",
                        help="Use importance-based routing for memory writes")
    parser.add_argument("--vary_n_segments", action="store_true",
                        help="Vary number of segments during training for robustness")

    # Backend
    parser.add_argument("--use_fsdp", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # DDP setup
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        print(f"=== SparseMemory Training ===")
        print(f"World size: {world_size}, Global rank: {global_rank}")
        print(f"Memory: slots={args.memory_slots}, top_k={args.top_k}, window={args.sliding_window}")
        print(f"Training: lr={args.lr}, batch={args.batch_size}, accum={args.grad_accumulation_steps}")
        os.makedirs(args.output_dir, exist_ok=True)

    # ── Mixed precision config ──
    mp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": torch.float32}[args.mixed_precision]
    use_grad_scaler = (mp_dtype == torch.float16)  # fp16 needs GradScaler to avoid underflow

    # Build model in target dtype
    model = SparseMemoryLlamaForCausalLM(
        base_model=args.base_model,
        memory_slots=args.memory_slots,
        top_k=args.top_k,
        sliding_window=args.sliding_window,
        ema_alpha=args.ema_alpha,
        torch_dtype=mp_dtype,
        use_l1=args.use_l1,
        num_mem_tokens=args.num_mem_tokens,
        l1_num_tokens=args.l1_num_tokens,
        segment_length=args.segment_length,
        max_segments=args.max_segments,
        bptt_depth=args.bptt_depth,
        recon_loss_coef=args.recon_loss_coef,
        use_importance_routing=args.use_importance_routing,
    )

    if args.gradient_checkpointing:
        if global_rank == 0:
            print("WARNING: gradient_checkpointing disabled — incompatible with memory write side effects (causes NaN)")

    model = model.to(device)

    # GradScaler for fp16 (not needed for bf16)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=use_grad_scaler) if use_grad_scaler else None

    if global_rank == 0:
        print(f"Mixed precision: {args.mixed_precision} (dtype={mp_dtype})")
        print(f"GradScaler: {'enabled (fp16)' if grad_scaler else 'disabled (bf16/fp32)'}")
        # Print memory estimate
        param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        buf_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
        print(f"Model size: {param_bytes / 1e9:.2f} GB params + {buf_bytes / 1e9:.2f} GB buffers")

    # DDP wrapper
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Optimizer: separate param groups for gate (10x LR) vs other params
    gate_params = [p for n, p in model.named_parameters() if p.requires_grad and 'gate_proj' in n]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and 'gate_proj' not in n]
    param_groups = [{'params': other_params, 'lr': args.lr}]
    if gate_params:
        param_groups.append({'params': gate_params, 'lr': args.lr * 10.0})
        if global_rank == 0:
            print(f"Gate LR: {args.lr * 10.0:.2e} ({len(gate_params)} params)")
    if global_rank == 0:
        print(f"Other LR: {args.lr:.2e} ({len(other_params)} params)")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    import math
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Tokenizer
    tokenizer_kwargs = {"use_fast": True}
    if os.environ.get("HF_TOKEN"):
        tokenizer_kwargs["token"] = os.environ["HF_TOKEN"]
    model_kwargs = {}
    if os.environ.get("HF_TOKEN"):
        model_kwargs["token"] = os.environ["HF_TOKEN"]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset — auto-detect format: .jsonl → tokenize on-the-fly, .npy → pre-tokenized
    if args.data_path.endswith(".jsonl"):
        dataset = JsonlDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
        )
    elif args.data_path.endswith(".npy"):
        dataset = PreTokenizedDataset(
            path=args.data_path,
            max_seq_len=args.seq_len,
            vocab_size=model.config.vocab_size,
        )
    else:
        raise ValueError(f"Unsupported data format: {args.data_path}. Use .jsonl or .npy")
    if global_rank == 0:
        print(f"Dataset: {type(dataset).__name__}, {len(dataset)} samples, seq_len={args.seq_len}")
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Training loop
    model.train()
    step = 0
    epoch = 0
    data_iter = iter(dataloader)
    log_history = []

    if global_rank == 0:
        print(f"Starting training for {args.max_steps} steps...")

    while step < args.max_steps:
        sampler.set_epoch(epoch)

        for batch in dataloader:
            if step >= args.max_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            # ── Forward with autocast ──
            with torch.amp.autocast('cuda', dtype=mp_dtype, enabled=(mp_dtype != torch.float32)):
                outputs = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                )
                loss = outputs.loss

            loss_scaled = loss / args.grad_accumulation_steps

            # ── Backward (with GradScaler for fp16) ──
            if grad_scaler is not None:
                grad_scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # ── Optimizer step ──
            if (step + 1) % args.grad_accumulation_steps == 0:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1

            # Logging
            if step % args.log_every == 0 and global_rank == 0:
                fusion_grad = model.module.get_fusion_grad_norm()

                log_entry = {
                    "step": step,
                    "loss": loss_scaled.item() * args.grad_accumulation_steps,
                    "lr": scheduler.get_last_lr()[0],
                    "fusion_grad_norm": fusion_grad,
                    "timestamp": time.time(),
                }
                log_history.append(log_entry)
                print(
                    f"[step {step:>5d}] loss={log_entry['loss']:.4f}  "
                    f"lr={log_entry['lr']:.2e}  "
                    f"fusion_grad={fusion_grad:.4f}"
                )

                # Heartbeat file
                heartbeat_path = os.path.join(args.output_dir, "heartbeat.json")
                with open(heartbeat_path, "w") as f:
                    json.dump(log_entry, f, indent=2)

            # Save checkpoint (save sparse memory weights separately)
            if step % args.save_every == 0 and step > 0 and global_rank == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                os.makedirs(save_path, exist_ok=True)
                model.module.model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                # Save sparse memory state (gate_proj weights)
                sm_state = {
                    f"sparse_attn.layer_{i}.": attn.state_dict()
                    for i, attn in enumerate(model.module.sparse_attn_layers)
                }
                torch.save(sm_state, os.path.join(save_path, "sparse_memory.pt"))
                print(f"Saved checkpoint to {save_path}")

        epoch += 1

    # Final save
    if global_rank == 0:
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        model.module.model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        # Save sparse memory state
        sm_state = {
            f"sparse_attn.layer_{i}.": attn.state_dict()
            for i, attn in enumerate(model.module.sparse_attn_layers)
        }
        torch.save(sm_state, os.path.join(final_path, "sparse_memory.pt"))
        print(f"Training complete. Final model saved to {final_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
