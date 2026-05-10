"""RMT v5 training with three architectural improvements over v4:
1. Bottleneck Cross-Attention Extraction (replaces mean pooling + MLP)
2. Importance-Based Selective Update (per-slot gate instead of scalar gate)
3. Z-forcing Memory Prediction Loss (memory predicts future segment summary)
"""

import os
import sys, time, json, datetime, math, argparse
from contextlib import nullcontext
import torch, torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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


# ====================== Training ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/rmt_train_mixed.jsonl")
    parser.add_argument("--output_dir", default="outputs/rmt_v4")
    parser.add_argument("--base_model", default="../models/Qwen--Qwen3-8b")
    parser.add_argument("--num_memory_tokens", type=int, default=16)
    parser.add_argument("--segment_length", type=int, default=1024)
    parser.add_argument("--max_segments", type=int, default=6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--rmt_lr", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--recon_weight", type=float, default=0.1)  # reconstruction loss weight
    parser.add_argument("--lambda_zforce", type=float, default=0.1)  # Z-forcing prediction loss weight
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extractor_version", type=int, default=5)
    parser.add_argument("--ddp", action="store_true", help="Enable DDP training")
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

    # RMT memory module (V3)
    hidden_dim = model.config.hidden_size
    rmt_memory = RMTMemory(
        hidden_dim=hidden_dim,
        num_memory_tokens=args.num_memory_tokens,
        num_heads=8,
        max_segments=args.max_segments + 1,
        bottleneck_dim=args.bottleneck_dim,
        extractor_version=args.extractor_version,
        use_reconstruction=True,
    )
    rmt_memory = rmt_memory.to(device=device, dtype=torch.bfloat16)
    if ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        rmt_memory = DDP(rmt_memory, device_ids=[rank])

    # Optimizer
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

    total_steps = len(dataloader) * args.num_epochs // args.grad_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    # Training loop
    if rank == 0:
        print(f"[RMT v5] Starting training: {len(dataloader)} steps/epoch, {args.num_epochs} epochs")
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

            # Trim valid length based on attention mask
            valid_lens = attn_mask.sum(dim=1).tolist()
            max_len = max(valid_lens)
            input_ids = input_ids[:, :max_len]

            num_segments = (max_len + args.segment_length - 1) // args.segment_length

            old_memory = None
            prev_memory = None  # for Z-forcing
            seg_loss_total = torch.tensor(0.0, device=device)
            recon_loss_total = torch.tensor(0.0, device=device)
            zforce_loss_total = torch.tensor(0.0, device=device)

            base_model = model.module if hasattr(model, "module") else model
            mem_module = rmt_memory.module if hasattr(rmt_memory, "module") else rmt_memory
            inner_model = base_model.get_base_model()
            backbone = inner_model.model

            # DDP gradient sync: only sync on last micro-batch of gradient accumulation
            is_last_accum = (batch_idx + 1) % args.grad_accumulation_steps == 0

            for seg_idx in range(num_segments):
                start = seg_idx * args.segment_length
                end = min(start + args.segment_length, max_len)
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

                # Use actual segment length (not fixed segment_length) for mask and position_ids
                actual_seg_len = seg_ids.shape[1]

                # Attention mask
                attn_mask_seg = build_rmt_attention_mask(actual_seg_len, args.num_memory_tokens, device)
                attn_mask_seg = attn_mask_seg.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

                # Position IDs
                position_ids = build_rmt_position_ids(
                    actual_seg_len, args.num_memory_tokens, seg_idx, device
                ).unsqueeze(0).expand(B, -1)

                # Labels
                mem_labels = torch.full((B, args.num_memory_tokens), -100, device=device, dtype=torch.long)
                full_labels = torch.cat([mem_labels, seg_labels], dim=1)

                # Per-segment forward + backward
                is_last_seg = (seg_idx == num_segments - 1)
                is_last_accum = (batch_idx + 1) % args.grad_accumulation_steps == 0
                if ddp:
                    ddp_context = nullcontext() if (is_last_accum and is_last_seg) else model.no_sync()
                else:
                    ddp_context = nullcontext()
                with ddp_context:
                    outputs = backbone(
                        inputs_embeds=inputs_embeds,
                        attention_mask={"full_attention": attn_mask_seg},
                        position_ids=position_ids,
                    )
                    hidden = outputs.last_hidden_state
                    logits = inner_model.lm_head(hidden)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = full_labels[..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = loss / num_segments

                    # Z-forcing loss: use prev_memory (from segment i-1) to predict this segment's hidden states
                    if prev_memory is not None and args.extractor_version == 5:
                        seg_hidden_for_zf = hidden[:, args.num_memory_tokens:, :]
                        zf_loss, _ = mem_module.memory_predictor(prev_memory.detach(), seg_hidden_for_zf)
                        zf_loss = zf_loss / num_segments
                        zf_loss.backward()
                        zforce_loss_total = zforce_loss_total + zf_loss.detach() * num_segments

                    # Extract memory for next segment (with gradients to extractor)
                    # Done BEFORE loss.backward() so the graph is still alive
                    seg_hidden = hidden[:, args.num_memory_tokens:, :]
                    mem_result = mem_module.extract_memory(seg_hidden, old_memory if old_memory is not None else None)
                    if isinstance(mem_result, tuple):
                        new_mem, recon_loss = mem_result
                        if recon_loss is not None:
                            recon_loss = recon_loss / num_segments
                            recon_loss.backward()
                            recon_loss_total = recon_loss_total + recon_loss.detach() * num_segments
                    else:
                        new_mem = mem_result
                    prev_memory = old_memory if old_memory is not None else None
                    # Detach for cross-segment use (prevents graph accumulation across segments)
                    old_memory = new_mem.detach()

                    # Backward the main CE loss last (graph still alive from extract_memory)
                    loss.backward()

                seg_loss_total = seg_loss_total + loss.detach() * num_segments
                accum_loss += loss.item() * num_segments
                accum_count += 1

            # Total loss (for logging only; actual gradients already applied per-segment)
            extra_loss = torch.tensor(0.0, device=device)
            if recon_loss_total > 0:
                extra_loss = extra_loss + args.recon_weight * recon_loss_total
            if zforce_loss_total > 0:
                extra_loss = extra_loss + args.lambda_zforce * zforce_loss_total
            total_loss = seg_loss_total + extra_loss

            # Gradient accumulation
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
                    print(f"[RMT v4] Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                          f"loss={avg_loss:.4f} lr={lr:.2e} mem={args.num_memory_tokens} ETA={eta:.1f}h",
                          file=log_file)
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

        epoch += 1
        if rank == 0:
            elapsed = time.time() - t0
            print(f"[RMT v4] Epoch {epoch} done, elapsed: {elapsed/3600:.1f}h", file=log_file)

    # Save
    if rank == 0:
        base_model = model.module if hasattr(model, "module") else model
        mem_module = rmt_memory.module if hasattr(rmt_memory, "module") else rmt_memory

        print("[RMT v4] Merging LoRA adapter into base model before save...")
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
        print(f"[RMT v4] Training complete!", file=log_file)
        log_file.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        raise
