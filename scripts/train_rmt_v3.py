"""RMT v3: LoRA fine-tune base model + memory module.

Key difference from v2: LoRA adapters on base model so it learns to 
utilize memory tokens. Memory module architecture same as v2 (bottleneck).

Usage:
  torchrun --nproc_per_node=8 scripts/train_rmt_v3.py \
    --model_path ../models/Qwen--Qwen3-8b \
    --data_path data/rmt_train_mixed.jsonl \
    --output_dir outputs/rmt_v3_XXXX \
    --num_memory_tokens 8 --segment_length 1024 --max_segments 6 \
    --num_epochs 3 --lr 2e-5 --rmt_lr 5e-5 \
    --lora_r 16 --lora_alpha 32
"""
# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing torch.
# torch caches the GPU device list on first import, so setting this
# after import torch has no effect.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import json, os, sys, time, argparse, math
from contextlib import nullcontext
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from src.memory.rmt.rmt_module import (
    RMTMemory, RMTModel, MemoryExtractorV2,
    build_rmt_attention_mask, build_rmt_position_ids,
)


class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_tokens=6144):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.docs = []
        with open(data_path) as f:
            for line in f:
                doc = json.loads(line)["text"]
                tokens = tokenizer.encode(doc, add_special_tokens=False)
                if len(tokens) >= max_tokens // 2:
                    self.docs.append(tokens)
        print(f"[Data] Loaded {len(self.docs)} docs (min {min(len(d) for d in self.docs)} tokens)")

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        tokens = self.docs[idx]
        tokens = tokens[:self.max_tokens]
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch, seg_len, max_seg):
    # Pad to exact multiple of seg_len * max_seg
    target_len = seg_len * max_seg
    padded = []
    for t in batch:
        if len(t) < target_len:
            pad = torch.full((target_len - len(t),), 0, dtype=torch.long)
            t = torch.cat([t, pad])
        padded.append(t[:target_len])
    return torch.stack(padded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_memory_tokens", type=int, default=8)
    parser.add_argument("--segment_length", type=int, default=1024)
    parser.add_argument("--max_segments", type=int, default=6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--rmt_lr", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint dir to resume from (e.g. outputs/rmt_v3_xxx/checkpoints/epoch_3)")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--bottleneck_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Initialize DDP before any CUDA operations
    # IMPORTANT: CUDA_VISIBLE_DEVICES must be set BEFORE init_process_group
    # because NCCL backend initializes CUDA contexts on all visible GPUs
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        ddp = True
    else:
        rank = 0
        world_size = 1
        ddp = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg = vars(args)
        cfg["timestamp"] = timestamp
        with open(os.path.join(output_dir, "rmt_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    # Load tokenizer & model
    if rank == 0:
        print("[RMT] Loading tokenizer & model...")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=None,
    ).to(device)

    # Apply LoRA (model is already on GPU, LoRA wraps in-place)
    if rank == 0:
        print(f"[RMT] Applying LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create RMT memory module
    hidden_dim = model.config.hidden_size
    rmt_memory = RMTMemory(
        hidden_dim=hidden_dim,
        num_memory_tokens=args.num_memory_tokens,
        num_heads=8,
        max_segments=args.max_segments + 1,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device=device, dtype=torch.bfloat16)

    if rank == 0:
        total_params = sum(p.numel() for p in rmt_memory.parameters())
        lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[RMT] Memory module: {total_params:,} params")
        print(f"[RMT] LoRA params: {lora_params:,}")

    # Wrap with DDP
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        rmt_memory = DDP(rmt_memory, device_ids=[local_rank])

    # Dataset
    max_tokens = args.segment_length * args.max_segments
    dataset = TextDataset(args.data_path, tok, max_tokens)
    sampler = DistributedSampler(dataset, shuffle=True) if ddp else None
    collate = lambda batch: collate_fn(batch, args.segment_length, args.max_segments)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=(sampler is None), collate_fn=collate, num_workers=0,
    )

    # Optimizer: LoRA + RMT memory with different LRs and warmup schedules
    lora_params_list = [p for p in model.parameters() if p.requires_grad]
    rmt_params_list = list(rmt_memory.parameters())
    optimizer = torch.optim.AdamW([
        {"params": lora_params_list, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": rmt_params_list, "lr": args.rmt_lr, "weight_decay": 0.01},
    ])

    total_steps = len(dataloader) * args.num_epochs // args.grad_accumulation_steps
    # Memory modules get 2K warmup, backbone gets proportionally less
    mem_warmup = min(args.warmup_steps, total_steps // 4)
    backbone_warmup = max(mem_warmup // 4, 50)
    scheduler = get_cosine_schedule_with_warmup(optimizer, [backbone_warmup, mem_warmup], total_steps)

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if args.resume_from and os.path.isdir(args.resume_from):
        ckpt_path = os.path.join(args.resume_from, "ckpt.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            # Restore model weights
            base_m = model.module if hasattr(model, "module") else model
            base_m.load_state_dict(ckpt["model_state_dict"], strict=False)
            mem_m = rmt_memory.module if hasattr(rmt_memory, "module") else rmt_memory
            mem_m.load_state_dict(ckpt["rmt_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            global_step = ckpt.get("global_step", 0)
            if rank == 0:
                print(f"[RMT] Resumed from epoch {start_epoch}, step {global_step}")
        elif rank == 0:
            print(f"[RMT] WARNING: resume_from={args.resume_from} but no ckpt.pt found, starting fresh")

    # Training loop
    if rank == 0:
        print(f"[RMT] Starting training: {len(dataloader)} steps/epoch, {args.num_epochs} epochs, {total_steps} total steps")
        print(f"[RMT] LR schedule: backbone warmup={backbone_warmup}, memory warmup={mem_warmup}")
        # Heartbeat
        hb = {"timestamp": datetime.now().isoformat(), "status": "running", "progress": 0.0, "metrics": {}, "extra": {}, "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))

    model.train()
    rmt_memory.train()
    epoch = 0
    step = global_step
    t0 = time.time()
    accum_loss = 0.0
    accum_count = 0
    log_file = open(os.path.join(output_dir, "train.log"), "a") if rank == 0 else open(os.devnull, "w")

    for ep in range(start_epoch, args.num_epochs):
        if sampler:
            sampler.set_epoch(ep)
        epoch = ep + 1

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch.to(device)
            B, L = input_ids.shape
            num_segments = L // args.segment_length

            old_memory = None
            seg_loss_total = torch.tensor(0.0, device=device)

            # Get references once
            base_model = model.module if hasattr(model, "module") else model
            mem_module = rmt_memory.module if hasattr(rmt_memory, "module") else rmt_memory
            inner_model = base_model.get_base_model()
            backbone = inner_model.model

            is_last_accum = (batch_idx + 1) % args.grad_accumulation_steps == 0
            for seg_idx in range(num_segments):
                is_last_seg = (seg_idx == num_segments - 1)
                ddp_context = nullcontext() if (is_last_accum and is_last_seg) else model.no_sync()
                start = seg_idx * args.segment_length
                end = start + args.segment_length
                seg_ids = input_ids[:, start:end]
                seg_labels = seg_ids.clone()

                # Get memory
                if old_memory is None:
                    mem = mem_module.get_initial_memory(seg_idx, B, device, torch.bfloat16)
                else:
                    mem = old_memory

                # Embed with memory
                token_embeds = inner_model.get_input_embeddings()(seg_ids)
                inputs_embeds = torch.cat([mem, token_embeds], dim=1)

                # Attention mask: memory bidirectional, segment causal
                attn_mask = build_rmt_attention_mask(args.segment_length, args.num_memory_tokens, device)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

                # Position IDs
                position_ids = build_rmt_position_ids(
                    args.segment_length, args.num_memory_tokens, seg_idx, device
                ).unsqueeze(0).expand(B, -1)

                # Labels (memory = -100)
                mem_labels = torch.full((B, args.num_memory_tokens), -100, device=device, dtype=torch.long)
                full_labels = torch.cat([mem_labels, seg_labels], dim=1)

                # Per-segment forward + backward to save memory
                with ddp_context:
                    outputs = backbone(
                        inputs_embeds=inputs_embeds,
                        attention_mask={"full_attention": attn_mask},
                        position_ids=position_ids,
                    )
                    hidden = outputs.last_hidden_state

                    logits = inner_model.lm_head(hidden)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = full_labels[..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = loss / num_segments  # Average over segments
                    loss.backward()

                seg_loss_total = seg_loss_total + loss.detach() * num_segments
                accum_loss += loss.item() * num_segments
                accum_count += 1

                # Extract memory for next segment (no grad)
                with torch.no_grad():
                    seg_hidden = hidden[:, args.num_memory_tokens:, :]
                    old_memory = mem_module.extract_memory(seg_hidden.detach(), old_memory.detach() if old_memory is not None else None)

            # Gradient accumulation step
            if (batch_idx + 1) % args.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(rmt_memory.parameters()),
                    args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if rank == 0 and step % args.log_every == 0:
                    avg_loss = accum_loss / accum_count
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    eta = elapsed / step * (total_steps - step) / 3600
                    print(f"[RMT] Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                          f"loss={avg_loss:.4f} lr={lr:.2e} mem={args.num_memory_tokens} ETA={eta:.1f}h",
                          file=log_file)
                    hb = {
                        "timestamp": datetime.now().isoformat(),
                        "status": "running",
                        "progress": step / total_steps,
                        "metrics": {"loss": avg_loss, "learning_rate": lr, "epoch": epoch, "step": step},
                        "extra": {},
                        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))
                    accum_loss = 0.0
                    accum_count = 0

        if rank == 0:
            elapsed = time.time() - t0
            print(f"[RMT] Epoch {epoch} done, elapsed: {elapsed/3600:.1f}h", file=log_file)

            # Save per-epoch checkpoint for resume
            ckpt_dir = os.path.join(output_dir, "checkpoints", f"epoch_{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            base_m = model.module if hasattr(model, "module") else model
            mem_m = rmt_memory.module if hasattr(rmt_memory, "module") else rmt_memory
            torch.save({
                "epoch": epoch,
                "global_step": step,
                "model_state_dict": base_m.state_dict(),
                "rmt_state_dict": mem_m.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }, os.path.join(ckpt_dir, "ckpt.pt"))
            print(f"[RMT] Checkpoint saved: {ckpt_dir}", file=log_file)

            # Gate health monitoring
            _monitor_gates(rmt_memory, output_dir, epoch, step, log_file)

    # Save
    if rank == 0:
        base_model = model.module if hasattr(model, "module") else model
        mem_module = rmt_memory.module if hasattr(rmt_memory, "module") else rmt_memory

        print("[RMT] Saving LoRA adapter + RMT memory...")
        base_model.save_pretrained(os.path.join(output_dir, "final"))  # Saves LoRA adapter
        # Also copy tokenizer from base model path
        tok.save_pretrained(os.path.join(output_dir, "final"))
        torch.save(mem_module.state_dict(), os.path.join(output_dir, "final", "rmt_memory.pt"))

        elapsed = time.time() - t0
        hb = {
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "progress": 1.0,
            "metrics": {"training_time_s": elapsed},
            "extra": {},
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))
        print(f"[RMT] Training complete!", file=log_file)
        log_file.close()


def _monitor_gates(rmt_memory, output_dir, epoch, step, log_file):
    """Check gate values in RMT memory module are in healthy range [0.2, 0.8]."""
    mem = rmt_memory.module if hasattr(rmt_memory, "module") else rmt_memory
    gate_values = []
    for name, param in mem.named_parameters():
        if "gate" in name.lower():
            with torch.no_grad():
                vals = torch.sigmoid(param).flatten() if param.min() < 0 or param.max() > 1 else param.flatten()
                gate_values.append((name, vals))
    if not gate_values:
        return
    alerts = []
    for name, vals in gate_values:
        mean_v = vals.mean().item()
        frac_saturated = ((vals > 0.95) | (vals < 0.05)).float().mean().item()
        if mean_v > 0.8 or mean_v < 0.2:
            alerts.append(f"  WARNING: {name} mean={mean_v:.4f} outside [0.2,0.8]")
        if frac_saturated > 0.5:
            alerts.append(f"  WARNING: {name} {frac_saturated:.0%} saturated (>0.95 or <0.05)")
    if alerts:
        msg = f"[RMT] Gate monitor epoch={epoch} step={step}:\n" + "\n".join(alerts)
        print(msg, file=log_file)
        # Also write to separate monitor file
        with open(os.path.join(output_dir, "gate_monitor.jsonl"), "a") as gf:
            json.dump({"epoch": epoch, "step": step, "alerts": alerts,
                       "timestamp": datetime.now().isoformat()}, gf)
            gf.write("\n")


if __name__ == "__main__":
    main()
