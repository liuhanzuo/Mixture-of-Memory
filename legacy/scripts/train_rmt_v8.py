"""
RMT v8 training — BABILong-style direct injection.

Key differences from v5:
- No CrossAttentionExtractor, no ImportanceMemoryUpdater
- Memory = last K hidden states from previous segment (direct passthrough)
- No torch.no_grad() — natural gradient flow
- Curriculum learning: K starts at 4, increases to 16 over training
- Simpler code, fewer parameters, BABILong-proven approach
"""

import os, sys, time, json, datetime, math, argparse
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


def collate_fn(batch, segment_length):
    max_len = max(len(x) for x in batch)
    padded_len = ((max_len + segment_length - 1) // segment_length) * segment_length
    padded = []
    for x in batch:
        pad_len = padded_len - len(x)
        padded.append(torch.cat([x, torch.full([pad_len], 0, dtype=torch.long)]))
    return torch.stack(padded)


def get_current_K(epoch, num_epochs, K_min=4, K_max=16):
    """Curriculum learning: linearly increase memory tokens from K_min to K_max."""
    if num_epochs <= 1:
        return K_max
    progress = min(epoch / max(num_epochs - 1, 1), 1.0)
    K = int(K_min + (K_max - K_min) * progress)
    # Round to nearest power of 2 for cleanliness
    return max(K_min, min(K, K_max))


# ====================== Training ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/rmt_train_mixed.jsonl")
    parser.add_argument("--output_dir", default="outputs/rmt_v8")
    parser.add_argument("--base_model", default="../models/Qwen--Qwen3-8b")
    parser.add_argument("--num_memory_tokens", type=int, default=16)
    parser.add_argument("--K_min", type=int, default=4, help="Starting memory tokens for curriculum")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddp", action="store_true")
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

    cfg = vars(args)
    cfg["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["extractor_version"] = 8
    if rank == 0:
        with open(os.path.join(output_dir, "rmt_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{rank}"},
    )
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    hidden_dim = model.config.hidden_size
    rmt_memory = RMTMemory(
        hidden_dim=hidden_dim,
        num_memory_tokens=args.num_memory_tokens,
        max_segments=args.max_segments + 1,
        extractor_version=8,
    )
    rmt_memory = rmt_memory.to(device=device, dtype=torch.bfloat16)
    if ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        rmt_memory = DDP(rmt_memory, device_ids=[rank])

    optimizer = AdamW([
        {"params": model.parameters(), "lr": args.lr},
        {"params": rmt_memory.parameters(), "lr": args.rmt_lr},
    ], weight_decay=0.01)

    dataset = RMTDataset(args.data, tok, args.segment_length, args.max_segments)
    sampler = None
    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, args.segment_length),
        shuffle=(sampler is None), sampler=sampler, num_workers=0, pin_memory=False,
    )

    total_steps = len(dataloader) * args.num_epochs // args.grad_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    if rank == 0:
        print(f"[RMT v8] Starting: {len(dataloader)} steps/epoch, {args.num_epochs} epochs")

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

        # Curriculum: current K for this epoch
        current_K = get_current_K(epoch, args.num_epochs, args.K_min, args.num_memory_tokens)
        if rank == 0:
            print(f"[RMT v8] Epoch {epoch}: using K={current_K} memory tokens", file=log_file)

        for batch_idx, input_ids in enumerate(dataloader):
            B = input_ids.shape[0]
            input_ids = input_ids.to(device)
            max_len = input_ids.shape[1]
            num_segments = max_len // args.segment_length

            old_memory = None
            seg_loss_total = torch.tensor(0.0, device=device)

            base_model_ref = model.module if hasattr(model, "module") else model
            mem_module = rmt_memory.module if hasattr(rmt_memory, "module") else rmt_memory
            inner_model = base_model_ref.get_base_model()
            backbone = inner_model.model

            for seg_idx in range(num_segments):
                start = seg_idx * args.segment_length
                end = min(start + args.segment_length, max_len)
                seg_ids = input_ids[:, start:end]
                seg_labels = seg_ids.clone()

                # Get memory
                if old_memory is None:
                    mem = mem_module.get_initial_memory(seg_idx, B, device, torch.bfloat16)
                    # Trim to current_K for curriculum
                    mem = mem[:, :current_K, :]
                else:
                    mem = old_memory

                actual_K = mem.shape[1]
                actual_seg_len = seg_ids.shape[1]

                # Embed with memory
                token_embeds = inner_model.get_input_embeddings()(seg_ids)
                inputs_embeds = torch.cat([mem, token_embeds], dim=1)

                # Attention mask
                attn_mask_seg = build_rmt_attention_mask(actual_seg_len, actual_K, device)
                attn_mask_seg = attn_mask_seg.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

                # Position IDs
                position_ids = build_rmt_position_ids(
                    actual_seg_len, actual_K, seg_idx, device
                ).unsqueeze(0).expand(B, -1)

                # Labels
                mem_labels = torch.full((B, actual_K), -100, device=device, dtype=torch.long)
                full_labels = torch.cat([mem_labels, seg_labels], dim=1)

                # DDP sync control
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
                    loss = nn.CrossEntropyLoss()(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                    ) / num_segments

                    # Extract memory BEFORE backward so graph is alive
                    seg_hidden = hidden[:, actual_K:, :]  # skip memory positions
                    mem_result = mem_module.extract_memory(seg_hidden, old_memory, current_K=current_K)
                    new_mem = mem_result[0] if isinstance(mem_result, tuple) else mem_result
                    # For v8: no detach! Gradients flow naturally through memory tokens
                    # But we DO detach for inter-segment to prevent OOM from graph accumulation
                    old_memory = new_mem.detach()

                    loss.backward()

                seg_loss_total = seg_loss_total + loss.detach() * num_segments
                accum_loss += loss.item() * num_segments
                accum_count += 1

            # Gradient accumulation step
            if (batch_idx + 1) % args.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(rmt_memory.parameters()), 1.0
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
                    print(
                        f"[RMT v8] Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                        f"loss={avg_loss:.4f} lr={lr:.2e} K={current_K} ETA={eta:.1f}h",
                        file=log_file,
                    )
                    accum_loss = 0.0
                    accum_count = 0

        epoch += 1
        if rank == 0:
            print(f"[RMT v8] Epoch {epoch} done, elapsed: {(time.time()-t0)/3600:.1f}h", file=log_file)

    # Save
    if rank == 0:
        base_model_ref = model.module if hasattr(model, "module") else model
        mem_module = rmt_memory.module if hasattr(rmt_memory, "module") else rmt_memory

        print("[RMT v8] Merging LoRA and saving...")
        merged = base_model_ref.merge_and_unload()
        merged.save_pretrained(os.path.join(output_dir, "final"))
        tok.save_pretrained(os.path.join(output_dir, "final"))
        torch.save(mem_module.state_dict(), os.path.join(output_dir, "final", "rmt_memory.pt"))

        elapsed = time.time() - t0
        print(f"[RMT v8] Training complete in {elapsed/3600:.1f}h", file=log_file)
        log_file.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        raise
