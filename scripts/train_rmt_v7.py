"""RMT v7 training — lessons from v5 (flat loss) and v6:
1. CE-only loss (no reconstruction, no z-forcing)
2. Memory tokens 64 (was 16) — compression ratio 16:1
3. LoRA rank 32 (was 16)
4. 20 epochs, proper LR split
5. Memory tokens initialized from backbone embeddings
6. Validation loop + WandB logging
"""

import os
import sys, time, json, datetime, math, argparse
from contextlib import nullcontext
import torch, torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from src.memory.rmt.rmt_module import (
    RMTMemory, build_rmt_attention_mask, build_rmt_position_ids,
)

# ====================== Dataset ======================

class RMTDataset(Dataset):
    def __init__(self, data_path, tokenizer, segment_length, max_segments):
        self.tokenizer = tokenizer
        self.seg_len = segment_length
        self.max_seg = max_segments

        docs = []
        with open(data_path) as f:
            for line in f:
                text = json.loads(line)["text"]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) >= segment_length:
                    docs.append(tokens)
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        tokens = self.docs[idx]
        max_tokens = self.seg_len * self.max_seg
        tokens = tokens[:max_tokens]
        return torch.tensor(tokens, dtype=torch.long)


class RMTValDataset(Dataset):
    """Validation dataset — same structure but smaller subset."""
    def __init__(self, data_path, tokenizer, segment_length, max_segments, max_docs=200):
        self.tokenizer = tokenizer
        self.seg_len = segment_length
        self.max_seg = max_segments

        docs = []
        with open(data_path) as f:
            for line in f:
                text = json.loads(line)["text"]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) >= segment_length:
                    docs.append(tokens)
                if len(docs) >= max_docs:
                    break
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        tokens = self.docs[idx]
        max_tokens = self.seg_len * self.max_seg
        tokens = tokens[:max_tokens]
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch, segment_length):
    """Pad all docs to the same length (multiple of segment_length)."""
    max_len = max([len(x) for x in batch])
    padded_len = ((max_len + segment_length - 1) // segment_length) * segment_length
    padded = []
    attn_masks = []
    for x in batch:
        pad_len = padded_len - len(x)
        padded.append(torch.cat([x, torch.full([pad_len], 0, dtype=torch.long)]))
        attn_masks.append(torch.cat([torch.ones(len(x)), torch.zeros(pad_len)]))
    return torch.stack(padded), torch.stack(attn_masks).bool()


# ====================== Z-forcing schedule ======================

    """Probability schedule for z-forcing (teacher forcing with memory prediction).
    0.5 for epochs 0-1, 0.25 for epochs 2-4, 0.0 after epoch 5.
    """
    if epoch < 2:
        return 0.5
    elif epoch < 5:
        return 0.25
    else:
        return 0.0


# ====================== Validation ======================

@torch.no_grad()
def validate(model, rmt_memory, val_dataloader, args, device, rank):
    """Run a validation pass and return average CE loss."""
    base_model = model.module if hasattr(model, "module") else model
    mem_module = rmt_memory.module if hasattr(rmt_memory, "module") else rmt_memory
    inner_model = base_model.get_base_model()
    backbone = inner_model.model

    total_loss = 0.0
    total_count = 0

    model.eval()
    rmt_memory.eval()

    for input_ids, attn_mask in val_dataloader:
        B = input_ids.shape[0]
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        valid_lens = attn_mask.sum(dim=1).tolist()
        max_len = max(valid_lens)
        input_ids = input_ids[:, :max_len]

        num_segments = (max_len + args.segment_length - 1) // args.segment_length

        old_memory = None
        seg_loss_total = torch.tensor(0.0, device=device)

        for seg_idx in range(num_segments):
            start = seg_idx * args.segment_length
            end = min(start + args.segment_length, max_len)
            seg_ids = input_ids[:, start:end]
            seg_labels = seg_ids.clone()

            if old_memory is None:
                mem = mem_module.get_initial_memory(seg_idx, B, device, torch.bfloat16)
            else:
                mem = old_memory

            token_embeds = inner_model.get_input_embeddings()(seg_ids)
            inputs_embeds = torch.cat([mem, token_embeds], dim=1)

            actual_seg_len = seg_ids.shape[1]
            attn_mask_seg = build_rmt_attention_mask(actual_seg_len, args.num_memory_tokens, device)
            attn_mask_seg = attn_mask_seg.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
            # Convert bool mask to additive format for SDPA:
            # SDPA interprets bool True as BLOCKED, but our convention is True=can attend.
            # So invert: False (can attend) -> 0.0, True (blocked) -> -inf.
            attn_mask_float = torch.zeros_like(attn_mask_seg, dtype=torch.bfloat16)
            attn_mask_float = attn_mask_float.masked_fill(~attn_mask_seg, torch.tensor(float('-inf'), dtype=torch.bfloat16))
            position_ids = build_rmt_position_ids(
                actual_seg_len, args.num_memory_tokens, seg_idx, device
            ).unsqueeze(0).expand(B, -1)

            mem_labels = torch.full((B, args.num_memory_tokens), -100, device=device, dtype=torch.long)
            full_labels = torch.cat([mem_labels, seg_labels], dim=1)

            outputs = backbone(
                inputs_embeds=inputs_embeds,
                attention_mask={"full_attention": attn_mask_float},
                position_ids=position_ids,
            )
            hidden = outputs.last_hidden_state
            logits = inner_model.lm_head(hidden)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss / num_segments

            seg_loss_total = seg_loss_total + loss
            total_loss += loss.item() * num_segments
            total_count += 1

            seg_hidden = hidden[:, args.num_memory_tokens:, :]
            mem_result = mem_module.extract_memory(seg_hidden, old_memory)
            if isinstance(mem_result, tuple):
                new_mem, _ = mem_result
            else:
                new_mem = mem_result
            old_memory = new_mem

    model.train()
    rmt_memory.train()
    return total_loss / max(total_count, 1)


# ====================== Training ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/rmt_train_wiki_zh_10k.jsonl")
    parser.add_argument("--val_data", default="data/rmt_train_wiki_zh_10k.jsonl",
                        help="Validation data (same file is fine; we take first 200 docs)")
    parser.add_argument("--output_dir", default="outputs/rmt_v7_8gpu")
    parser.add_argument("--base_model", default="../models/Qwen--Qwen3-8b")
    parser.add_argument("--num_memory_tokens", type=int, default=64)
    parser.add_argument("--segment_length", type=int, default=1024)
    parser.add_argument("--max_segments", type=int, default=6)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--rmt_lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--bottleneck_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extractor_version", type=int, default=5)
    parser.add_argument("--ddp", action="store_true", help="Enable DDP training")
    parser.add_argument("--wandb_project", type=str, default="rmt-v6")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

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

    output_dir = args.output_dir + f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Save config
    cfg = vars(args)
    cfg["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if rank == 0:
        with open(os.path.join(output_dir, "rmt_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    # WandB logging
    if rank == 0 and args.wandb_project:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"rmt_v7_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=cfg,
            )
        except ImportError:
            print("[W] wandb not installed, skipping wandb logging")
            wandb = None
    else:
        wandb = None

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id

    # Load model + LoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
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
    model.print_trainable_parameters()

    # RMT memory module (V5 architecture)
    hidden_dim = model.config.hidden_size
    rmt_memory = RMTMemory(
        hidden_dim=hidden_dim,
        num_memory_tokens=args.num_memory_tokens,
        num_heads=8,
        max_segments=args.max_segments + 1,
        bottleneck_dim=args.bottleneck_dim,
        extractor_version=args.extractor_version,
        use_reconstruction=False,  # v6: no reconstruction loss
    )

    # Initialize memory tokens from backbone embedding layer (not random)
    with torch.no_grad():
        base_model_ref = model.get_base_model()
        embed_weight = base_model_ref.get_input_embeddings().weight.data  # [vocab_size, hidden_dim]
        # Sample random token indices and use their embeddings as init
        # memory_embeddings shape: [max_segments, num_memory_tokens, hidden_dim]
        init_indices = torch.randint(0, embed_weight.shape[0], (args.num_memory_tokens,))
        rmt_memory.memory_embeddings.data.copy_(
            embed_weight[init_indices].unsqueeze(0).expand_as(rmt_memory.memory_embeddings.data)
        )

    rmt_memory = rmt_memory.to(device=device, dtype=torch.bfloat16)
    if ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        rmt_memory = DDP(rmt_memory, device_ids=[rank])

    # Optimizer: memory LR = 10x base LR
    optimizer = AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": rmt_memory.parameters(), "lr": args.rmt_lr},
        ],
        weight_decay=0.01,
    )

    # Dataset
    dataset = RMTDataset(args.data, tok, args.segment_length, args.max_segments)
    sampler = None
    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, args.segment_length),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )

    # Validation dataset (subset)
    val_dataset = RMTValDataset(args.val_data, tok, args.segment_length, args.max_segments, max_docs=200)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, args.segment_length),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    total_steps = len(dataloader) * args.num_epochs // args.grad_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    # Training loop
    if rank == 0:
        print(f"[RMT v7] Starting training: {len(dataloader)} steps/epoch, {args.num_epochs} epochs, "
              f"{total_steps} total optimizer steps")
        print(f"[RMT v7] Z-forcing schedule: 0.5 (ep 0-1), 0.25 (ep 2-4), 0.0 (ep 5+)")
        print(f"[RMT v7] LR: base={args.lr}, memory={args.rmt_lr}")
        print(f"[RMT v7] Dataset: {len(dataset)} docs, val: {len(val_dataset)} docs")
        hb = {"timestamp": datetime.datetime.now().isoformat(), "status": "running", "progress": 0.0, "metrics": {}, "extra": {}, "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))

    model.train()
    rmt_memory.train()
    epoch = 0
    step = 0
    t0 = time.time()
    accum_loss = 0.0
    accum_count = 0
    log_file = open(os.path.join(output_dir, "train.log"), "a", 1) if rank == 0 else open(os.devnull, "w")

    while epoch < args.num_epochs:
        if ddp:
            sampler.set_epoch(epoch)

        for batch_idx, (input_ids, attn_mask) in enumerate(dataloader):
            B = input_ids.shape[0]
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            valid_lens = attn_mask.sum(dim=1).tolist()
            max_len = max(valid_lens)
            input_ids = input_ids[:, :max_len]

            num_segments = (max_len + args.segment_length - 1) // args.segment_length

            old_memory = None
            prev_memory = None  # for Z-forcing
            seg_loss_total = torch.tensor(0.0, device=device)

            base_model = model.module if hasattr(model, "module") else model
            mem_module = rmt_memory.module if hasattr(rmt_memory, "module") else rmt_memory
            inner_model = base_model.get_base_model()
            backbone = inner_model.model

            for seg_idx in range(num_segments):
                start = seg_idx * args.segment_length
                end = min(start + args.segment_length, max_len)
                seg_ids = input_ids[:, start:end]
                seg_labels = seg_ids.clone()

                if old_memory is None:
                    mem = mem_module.get_initial_memory(seg_idx, B, device, torch.bfloat16)
                else:
                    mem = old_memory

                token_embeds = inner_model.get_input_embeddings()(seg_ids)
                inputs_embeds = torch.cat([mem, token_embeds], dim=1)

                actual_seg_len = seg_ids.shape[1]
                attn_mask_seg = build_rmt_attention_mask(actual_seg_len, args.num_memory_tokens, device)
                attn_mask_seg = attn_mask_seg.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
                # Convert bool mask to additive format for SDPA:
                # SDPA interprets bool True as BLOCKED, but our convention is True=can attend.
                # So invert: False (can attend) -> 0.0, True (blocked) -> -inf.
                attn_mask_float = torch.zeros_like(attn_mask_seg, dtype=torch.bfloat16)
                attn_mask_float = attn_mask_float.masked_fill(~attn_mask_seg, torch.tensor(float('-inf'), dtype=torch.bfloat16))
                position_ids = build_rmt_position_ids(
                    actual_seg_len, args.num_memory_tokens, seg_idx, device
                ).unsqueeze(0).expand(B, -1)

                mem_labels = torch.full((B, args.num_memory_tokens), -100, device=device, dtype=torch.long)
                full_labels = torch.cat([mem_labels, seg_labels], dim=1)

                is_last_seg = (seg_idx == num_segments - 1)
                is_last_accum = (batch_idx + 1) % args.grad_accumulation_steps == 0
                if ddp:
                    ddp_context = nullcontext() if (is_last_accum and is_last_seg) else model.no_sync()
                else:
                    ddp_context = nullcontext()

                with ddp_context:
                    outputs = backbone(
                        inputs_embeds=inputs_embeds,
                        attention_mask={"full_attention": attn_mask_float},
                        position_ids=position_ids,
                    )
                    hidden = outputs.last_hidden_state

                    # Extract memory BEFORE loss.backward() so extractor
                    # receives gradients through the computation graph.
                    seg_hidden = hidden[:, args.num_memory_tokens:, :]
                    mem_result = mem_module.extract_memory(seg_hidden, old_memory)
                    if isinstance(mem_result, tuple):
                        new_mem, _ = mem_result
                    else:
                        new_mem = mem_result
                    prev_memory = old_memory if old_memory is not None else None
                    old_memory = new_mem

                    logits = inner_model.lm_head(hidden)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = full_labels[..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = loss / num_segments
                    loss.backward(retain_graph=(seg_idx < num_segments - 1))

                    # v7: no z-forcing — CE-only training

                seg_loss_total = seg_loss_total + loss.detach() * num_segments
                accum_loss += loss.item() * num_segments
                accum_count += 1

            # Gradient accumulation step
            if (batch_idx + 1) % args.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(rmt_memory.parameters()),
                    1.0
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
                    msg = (f"[RMT v7] Epoch {epoch} step {step} [{batch_idx+1}/{len(dataloader)}] "
                           f"loss={avg_loss:.4f} lr={lr:.2e}  ETA={eta:.1f}h")
                    print(msg, file=log_file)
                    if wandb:
                        wandb.log({"train_loss": avg_loss, "learning_rate": lr,
                                   "epoch": epoch, "step": step})
                    hb = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "status": "running",
                        "progress": step / total_steps,
                        "metrics": {"loss": avg_loss, "learning_rate": lr, "epoch": epoch, "step": step},
                        "extra": {},
                        "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))
                    accum_loss = 0.0
                    accum_count = 0

                # Validation
                if rank == 0 and step % args.val_every == 0:
                    val_loss = validate(model, rmt_memory, val_dataloader, args, device, rank)
                    print(f"[RMT v7] Validation step {step}: val_loss={val_loss:.4f}", file=log_file)
                    if wandb:
                        wandb.log({"val_loss": val_loss, "step": step})

                # Save checkpoint
                if rank == 0 and step % args.save_every == 0:
                    base_model = model.module if hasattr(model, "module") else model
                    mem_module = rmt_memory.module if hasattr(rmt_memory, "module") else rmt_memory
                    ckpt_dir = os.path.join(output_dir, f"checkpoint_step{step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    base_model.save_pretrained(ckpt_dir)
                    tok.save_pretrained(ckpt_dir)
                    torch.save(mem_module.state_dict(), os.path.join(ckpt_dir, "rmt_memory.pt"))
                    print(f"[RMT v7] Saved checkpoint to {ckpt_dir}", file=log_file)

        epoch += 1
        if rank == 0:
            elapsed = time.time() - t0
            print(f"[RMT v7] Epoch {epoch} done, elapsed: {elapsed/3600:.1f}h", file=log_file)

    # Save final model
    if rank == 0:
        base_model = model.module if hasattr(model, "module") else model
        mem_module = rmt_memory.module if hasattr(rmt_memory, "module") else rmt_memory

        print("[RMT v7] Merging LoRA adapter into base model before save...")
        merged_model = base_model.merge_and_unload()
        merged_model.save_pretrained(os.path.join(output_dir, "final"))
        tok.save_pretrained(os.path.join(output_dir, "final"))
        torch.save(mem_module.state_dict(), os.path.join(output_dir, "final", "rmt_memory.pt"))

        elapsed = time.time() - t0
        hb = {
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "completed",
            "progress": 1.0,
            "metrics": {"training_time_s": elapsed},
            "extra": {},
            "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))
        print(f"[RMT v7] Training complete!", file=log_file)
        log_file.close()
        if wandb:
            wandb.finish()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        raise
