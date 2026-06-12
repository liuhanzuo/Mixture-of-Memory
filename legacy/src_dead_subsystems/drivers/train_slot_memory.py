"""Slot Memory Training Script — Three-stage training.

Stage 1: Reconstruction-only (lambda_recon=1.0, lambda_ce=0.0) — 5 epochs
Stage 2: Joint training (lambda_recon=0.5, lambda_ce=1.0) — 15 epochs
Stage 3: CE-only fine-tuning (lambda_recon=0.0, lambda_ce=1.0) — 10 epochs
"""

import os
import sys
import time
import math
import json
import datetime
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from src.memory.slot.slot_memory_compressor import SlotMemoryCompressor, SlotMemoryWrapper


# ====================== Dataset ======================

class SlotMemoryDataset(Dataset):
    """Each sample is a long document returned as a token tensor."""

    def __init__(self, data_path, tokenizer, max_tokens):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        docs = []
        with open(data_path) as f:
            for line in f:
                text = json.loads(line)["text"]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) >= 512:
                    docs.append(tokens)
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        tokens = self.docs[idx][:self.max_tokens]
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch, segment_length, max_segments):
    """Pad docs to multiples of segment_length * max_segments."""
    max_len = max(len(x) for x in batch)
    padded_len = ((max_len + segment_length - 1) // segment_length) * segment_length
    padded_len = min(padded_len, segment_length * max_segments)

    padded, labels_list = [], []
    for x in batch:
        t = x[:padded_len]
        pad_len = padded_len - len(t)
        if pad_len > 0:
            t = torch.cat([t, torch.zeros(pad_len, dtype=torch.long)])
        padded.append(t)
        labels_list.append(t.clone())

    return torch.stack(padded), torch.stack(labels_list)


# ====================== Stage Config ======================

STAGE_DEFAULTS = {
    1: {"epochs": 5, "lambda_recon": 1.0, "lambda_ce": 0.0,
        "lr": 2e-5, "slots_lr": 2e-4},
    2: {"epochs": 15, "lambda_recon": 0.5, "lambda_ce": 1.0,
        "lr": 2e-5, "slots_lr": 2e-4},
    3: {"epochs": 10, "lambda_recon": 0.0, "lambda_ce": 1.0,
        "lr": 2e-5, "slots_lr": 2e-4},
}


# ====================== Main ======================

def main():
    parser = argparse.ArgumentParser(description="Slot Memory Training")

    # Stage
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Checkpoint dir to resume from (for stage 2/3)")

    # Data
    parser.add_argument("--data", default="data/rmt_train_mixed.jsonl")
    parser.add_argument("--output_dir", default="outputs/slot_memory")

    # Model
    parser.add_argument("--base_model", default="../models/Qwen--Qwen3-8b")

    # Slot memory
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--slot_dim", type=int, default=256)
    parser.add_argument("--segment_length", type=int, default=1024)
    parser.add_argument("--max_segments", type=int, default=4)
    parser.add_argument("--bptt_depth", type=int, default=2)

    # Training overrides
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lambda_recon", type=float, default=None)
    parser.add_argument("--lambda_ce", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--slots_lr_multiplier", type=float, default=10.0)

    # Optimizer
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=200)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddp", action="store_true", help="Enable DDP training")

    args = parser.parse_args()

    # Apply stage defaults
    stage_cfg = STAGE_DEFAULTS[args.stage]
    if args.num_epochs is None:
        args.num_epochs = stage_cfg["epochs"]
    if args.lambda_recon is None:
        args.lambda_recon = stage_cfg["lambda_recon"]
    if args.lambda_ce is None:
        args.lambda_ce = stage_cfg["lambda_ce"]

    torch.manual_seed(args.seed)
    ddp = args.ddp
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir + f"_stage{args.stage}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Save config
    cfg = vars(args)
    cfg["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if rank == 0:
        with open(os.path.join(output_dir, "train_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id

    # Determine base model path: if resuming from a merged checkpoint, use it as base
    effective_base_model = args.base_model
    if args.resume_from:
        _rp = os.path.join(args.resume_from, "final")
        if not os.path.exists(_rp):
            _rp = args.resume_from
        # Check if this is a merged model (has model.safetensors but no adapter)
        has_merged = os.path.exists(os.path.join(_rp, "model.safetensors"))
        has_adapter = os.path.exists(os.path.join(_rp, "adapter_model.safetensors"))
        if has_merged and not has_adapter:
            effective_base_model = _rp
            if rank == 0:
                print(f"[SlotMemory] Using merged checkpoint as base model: {_rp}")

    # Base model + LoRA
    model = AutoModelForCausalLM.from_pretrained(
        effective_base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{rank}"},
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if rank == 0:
        model.print_trainable_parameters()
    model.gradient_checkpointing_enable()

    # Slot memory compressor
    # num_segments = max_segments + 1 so each segment index (0..max_segments-1)
    # has a learned initial slot, plus one extra for safety.
    compressor = SlotMemoryCompressor(
        hidden_dim=4096,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        num_segments=args.max_segments + 1,
    )

    # Slot memory wrapper (old architecture, checkpoint-compatible)
    slot_model = SlotMemoryWrapper(
        model=model,
        compressor=compressor,
        segment_length=args.segment_length,
    )
    slot_model = slot_model.to(device=device, dtype=torch.bfloat16)

    # Resume from checkpoint
    if args.resume_from:
        resume_path = os.path.join(args.resume_from, "final")
        if not os.path.exists(resume_path):
            resume_path = args.resume_from
        if rank == 0:
            print(f"[SlotMemory] Resuming from {resume_path}")
        # Load LoRA adapter if available
        if os.path.exists(os.path.join(resume_path, "adapter_model.safetensors")):
            model.load_adapter(resume_path, adapter_name="default")
        # Load slot weights (saved as slot_memory.pt or slot_weights.pt)
        for slot_fn in ("slot_memory.pt", "slot_weights.pt"):
            slot_weights_path = os.path.join(resume_path, slot_fn)
            if os.path.exists(slot_weights_path):
                slot_state = torch.load(slot_weights_path, map_location=device)
                slot_model.compressor.load_state_dict(slot_state, strict=False)
                # Log any missing/unexpected keys so mismatches are visible
                missing = set(slot_model.compressor.state_dict().keys()) - set(slot_state.keys())
                unexpected = set(slot_state.keys()) - set(slot_model.compressor.state_dict().keys())
                if rank == 0 and (missing or unexpected):
                    print(f"[SlotMemory] load_state_dict: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
                if rank == 0:
                    print(f"[SlotMemory] Loaded slot weights from {slot_fn}")
                break
        else:
            if rank == 0:
                print(f"[SlotMemory] WARNING: no slot weights found in {resume_path}")

    if ddp:
        slot_model = DDP(slot_model, device_ids=[rank], find_unused_parameters=False)
        slot_model._set_static_graph()  # fix: slot memory multi-segment forward shares params across segments

    # Optimizer: separate LR for backbone (LoRA) vs compressor (slots)
    backbone_params = []
    slot_params = []
    for name, param in slot_model.named_parameters():
        if "compressor" in name:
            slot_params.append(param)
        else:
            backbone_params.append(param)

    slots_lr = args.learning_rate * args.slots_lr_multiplier
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.learning_rate},
            {"params": slot_params, "lr": slots_lr},
        ],
        weight_decay=0.01,
    )

    # Dataset
    max_tokens = args.segment_length * args.max_segments
    dataset = SlotMemoryDataset(args.data, tok, max_tokens)

    sampler = None
    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, args.segment_length, args.max_segments),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )

    total_steps = len(dataloader) * args.num_epochs // args.grad_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    # Map stage to old wrapper's stage string
    stage_str = {1: "recon_only", 2: "joint", 3: "ce_only"}[args.stage]

    # Training loop
    if rank == 0:
        print(f"[SlotMemory Stage {args.stage}] Starting: {len(dataloader)} batches/epoch, "
              f"{args.num_epochs} epochs, {total_steps} steps")
        print(f"  lambda_recon={args.lambda_recon}, lambda_ce={args.lambda_ce}")
        print(f"  backbone_lr={args.learning_rate}, slots_lr={slots_lr}")
        hb = {
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "running", "progress": 0.0, "metrics": {},
            "extra": {"stage": args.stage},
            "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))

    slot_model.train()
    epoch = 0
    step = 0
    t0 = time.time()
    accum_loss = 0.0
    accum_ce = 0.0
    accum_recon = 0.0
    accum_count = 0
    log_file = open(os.path.join(output_dir, "train.log"), "a", 1) if rank == 0 else open(os.devnull, "w")

    while epoch < args.num_epochs:
        if ddp:
            sampler.set_epoch(epoch)

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            is_last_accum = (batch_idx + 1) % args.grad_accumulation_steps == 0
            ddp_context = nullcontext()
            if ddp:
                ctx_fn = slot_model.no_sync if not is_last_accum else nullcontext
                ddp_context = ctx_fn()

            # Old SlotMemoryWrapper does backward() internally per segment.
            # Wrap the forward in the DDP no_sync context for gradient accumulation.
            with ddp_context:
                total_ce, avg_recon = slot_model(
                    input_ids=input_ids,
                    labels=labels,
                    stage=stage_str,
                    training=True,
                )
                # Total loss for logging (backward already happened inside wrapper)
                if stage_str == "recon_only":
                    loss = avg_recon
                    ce_loss = 0.0
                    recon_loss = avg_recon
                elif stage_str == "ce_only":
                    loss = total_ce.item()
                    ce_loss = total_ce.item()
                    recon_loss = 0.0
                else:  # joint
                    loss = total_ce.item() + 0.5 * avg_recon
                    ce_loss = total_ce.item()
                    recon_loss = avg_recon

            accum_loss += loss
            accum_ce += ce_loss
            accum_recon += recon_loss
            accum_count += 1

            if is_last_accum:
                torch.nn.utils.clip_grad_norm_(slot_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if rank == 0 and step % args.log_every == 0:
                    avg_loss = accum_loss / accum_count
                    avg_ce = accum_ce / accum_count
                    avg_recon = accum_recon / accum_count
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    eta = elapsed / step * (total_steps - step) / 3600 if step > 0 else 0
                    print(
                        f"[SlotMem S{args.stage}] Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                        f"loss={avg_loss:.4f} ce={avg_ce:.4f} recon={avg_recon:.4f} "
                        f"lr={lr:.2e} ETA={eta:.1f}h",
                        file=log_file,
                    )
                    hb = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "status": "running",
                        "progress": step / total_steps,
                        "metrics": {
                            "loss": avg_loss, "ce_loss": avg_ce, "recon_loss": avg_recon,
                            "learning_rate": lr, "epoch": epoch, "step": step,
                            "stage_str": stage_str,
                        },
                        "extra": {"stage": args.stage},
                        "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))
                    accum_loss = accum_ce = accum_recon = accum_count = 0.0
                    accum_count = 0

            # Save checkpoint
            if rank == 0 and step > 0 and step % args.save_every == 0:
                save_path = os.path.join(output_dir, f"checkpoint_step{step}")
                os.makedirs(save_path, exist_ok=True)
                unwrapped = slot_model.module if hasattr(slot_model, "module") else slot_model
                # Save LoRA adapter
                unwrapped.model.save_pretrained(save_path)
                tok.save_pretrained(save_path)
                # Save slot weights
                torch.save(
                    unwrapped.compressor.state_dict(),
                    os.path.join(save_path, "slot_weights.pt"),
                )
                print(f"[SlotMem] Saved checkpoint at step {step}", file=log_file)

        epoch += 1

        # Save per-epoch checkpoint
        if rank == 0:
            save_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}")
            os.makedirs(save_path, exist_ok=True)
            unwrapped = slot_model.module if hasattr(slot_model, "module") else slot_model
            unwrapped.model.save_pretrained(save_path)
            tok.save_pretrained(save_path)
            torch.save(
                unwrapped.compressor.state_dict(),
                os.path.join(save_path, "slot_weights.pt"),
            )
            print(f"[SlotMem] Saved epoch {epoch} checkpoint", file=log_file)

    # Final save
    if rank == 0:
        unwrapped = slot_model.module if hasattr(slot_model, "module") else slot_model
        save_path = os.path.join(output_dir, "final")
        os.makedirs(save_path, exist_ok=True)

        print("[SlotMem] Merging LoRA adapter...", file=log_file)
        merged = unwrapped.model.merge_and_unload()
        merged.save_pretrained(save_path)
        tok.save_pretrained(save_path)

        torch.save(
            unwrapped.compressor.state_dict(),
            os.path.join(save_path, "slot_weights.pt"),
        )

        hb = {
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "completed", "progress": 1.0,
            "metrics": {"training_time_s": time.time() - t0},
            "extra": {"stage": args.stage},
            "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))
        print(f"[SlotMem Stage {args.stage}] Training complete!", file=log_file)
        log_file.close()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        raise
