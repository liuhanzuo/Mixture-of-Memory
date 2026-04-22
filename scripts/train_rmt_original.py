"""Minimal RMT training script using the original Bulatov implementation.

This is the simplest possible reproduction of RMT for causal LM:
- Uses MemoryCell + RecurrentWrapper from booydar/recurrent-memory-transformer
- No LoRA, no custom attention mask tricks, no importance routing
- Full finetuning only
- Designed for Llama2-7B on 8 GPUs via torchrun DDP
"""

import os
import sys
import time
import math
import json
import random
import datetime
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM

# Import original RMT classes
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'recurrent-memory-transformer'))
from modeling_rmt.language_modeling import MemoryCell, RecurrentWrapper


# ====================== Dataset ======================

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_tokens):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.docs = []
        with open(data_path) as f:
            for line in f:
                text = json.loads(line)["text"]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) >= 256:
                    self.docs.append(tokens)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        tokens = self.docs[idx][:self.max_tokens]
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch, segment_size):
    """Pad to multiple of segment_size."""
    max_len = max(len(x) for x in batch)
    padded_len = ((max_len + segment_size - 1) // segment_size) * segment_size

    padded = []
    for x in batch:
        t = x[:padded_len]
        pad_len = padded_len - len(t)
        if pad_len > 0:
            t = torch.cat([t, torch.full((pad_len,), 0, dtype=torch.long)])  # 0 is pad
        padded.append(t)
    return torch.stack(padded)


# ====================== Main ======================

def main():
    parser = argparse.ArgumentParser(description="Original RMT Training")

    # Data
    parser.add_argument("--data", default="data/rmt_train_mixed.jsonl")
    parser.add_argument("--output_dir", default="outputs/rmt_original")

    # Model
    parser.add_argument("--base_model", default="/root/Mixture-of-Memory/models/Llama--Llama2-7b")

    # RMT
    parser.add_argument("--num_mem_tokens", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=1024)  # segment size (excluding memory tokens)
    parser.add_argument("--max_n_segments", type=int, default=4)
    parser.add_argument("--bptt_depth", type=int, default=-1)   # -1 = full BPTT
    parser.add_argument("--vary_n_segments", action="store_true", default=True)
    parser.add_argument("--segment_alignment", type=str, default="right", choices=["left", "right", "center"])

    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--eval_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddp", action="store_true", default=False)

    args = parser.parse_args()

    # DDP setup
    ddp = args.ddp
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir + f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "train_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
        print(f"[RMT-Original] Output: {output_dir}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Base model (full precision loading, then cast)
    print(f"[RMT-Original] Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{rank}"},
    )
    model.gradient_checkpointing_enable()
    print(f"[RMT-Original] Base model loaded", flush=True)

    # RMT MemoryCell + RecurrentWrapper (original Bulatov code)
    memory_cell = MemoryCell(model, num_mem_tokens=args.num_mem_tokens)

    rmt_config = {
        'segment_size': args.input_size,
        'max_n_segments': args.max_n_segments,
        'bptt_depth': args.bptt_depth,
        'segment_alignment': args.segment_alignment,
    }
    rmt_model = RecurrentWrapper(memory_cell, **rmt_config)
    rmt_model = rmt_model.to(device=device, dtype=torch.bfloat16)

    # Monkey-patch process_outputs to avoid DDP tree_unflatten crash.
    # The original adds per-segment keys (logits_0, logits_1, ...)
    # which CausalLMOutputWithCrossAttentions.__init__() rejects.
    # We only need loss + logits for training.
    _orig_process_outputs = RecurrentWrapper.process_outputs
    def _safe_process_outputs(self, cell_outputs, **kwargs):
        from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions as CLO
        from torch.nn import CrossEntropyLoss
        out = CLO()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        labels = kwargs.get('labels')
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            loss_fct = CrossEntropyLoss()
            labels_mask = kwargs.get('labels_mask')
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()
                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]
            out['loss'] = loss_fct(flat_logits, flat_labels)
        else:
            out['loss'] = torch.tensor(0.0, device=full_logits.device)
        out['logits'] = full_logits
        return out
    RecurrentWrapper.process_outputs = _safe_process_outputs

    if ddp:
        rmt_model = DDP(rmt_model, device_ids=[rank], find_unused_parameters=True)

    # Count params
    total_params = sum(p.numel() for p in rmt_model.parameters())
    trainable_params = sum(p.numel() for p in rmt_model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"[RMT-Original] Total params: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")

    # Optimizer
    optimizer = AdamW(
        rmt_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Dataset
    max_tokens = args.input_size * args.max_n_segments
    dataset = TextDataset(args.data, tok, max_tokens)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, args.input_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
    )

    total_steps = len(dataloader) * args.num_epochs // args.grad_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    if rank == 0:
        print(f"[RMT-Original] {len(dataloader)} batches/epoch, {args.num_epochs} epochs, "
              f"{total_steps} total steps, LR={args.lr}")
        hb = {"timestamp": datetime.datetime.now().isoformat(), "status": "running",
              "progress": 0.0, "metrics": {},
              "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))

    # Eval helper
    eval_indices = list(range(min(args.eval_samples, len(dataset))))

    def run_ppl_eval():
        rmt_model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for idx in eval_indices:
                sample = dataset[idx].unsqueeze(0).to(device)
                outputs = rmt_model(input_ids=sample, labels=sample)
                loss = outputs['loss']
                n_tokens = (sample.numel() - 1) * sample.shape[0]  # total prediction targets (shifted by 1)
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
        rmt_model.train()
        avg_loss = total_loss / max(total_tokens, 1)
        return avg_loss, math.exp(min(avg_loss, 20))

    # Training loop
    rmt_model.train()
    epoch = 0
    step = 0
    t0 = time.time()
    accum_loss = 0.0
    accum_count = 0
    log_file = open(os.path.join(output_dir, "train.log"), "a", 1) if rank == 0 else open(os.devnull, "w")

    while epoch < args.num_epochs:
        if ddp:
            sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch.to(device)
            labels = batch.clone()

            # Vary number of segments for curriculum learning
            if args.vary_n_segments and args.max_n_segments > 1:
                actual_segments = random.randint(1, args.max_n_segments)
                # Trim to fit the chosen segment count
                max_tokens_this = args.input_size * actual_segments
                input_ids = input_ids[:, :max_tokens_this]
                labels = labels[:, :max_tokens_this]

            is_last_accum = (batch_idx + 1) % args.grad_accumulation_steps == 0
            ddp_context = nullcontext()
            if ddp:
                ctx_fn = rmt_model.no_sync if not is_last_accum else nullcontext
                ddp_context = ctx_fn()

            with ddp_context:
                outputs = rmt_model(input_ids=input_ids, labels=labels)
                loss = outputs['loss']
                (loss / args.grad_accumulation_steps).backward()

            accum_loss += loss.item()
            accum_count += 1

            if is_last_accum:
                torch.nn.utils.clip_grad_norm_(rmt_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if rank == 0 and step % args.log_every == 0:
                    avg_l = accum_loss / accum_count
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    eta = elapsed / step * (total_steps - step) / 3600
                    print(f"[RMT-Orig] step {step}/{total_steps} loss={avg_l:.4f} lr={lr:.2e} ETA={eta:.1f}h",
                          file=log_file)
                    hb = {"timestamp": datetime.datetime.now().isoformat(), "status": "running",
                          "progress": step / total_steps,
                          "metrics": {"loss": avg_l, "lr": lr, "step": step, "epoch": epoch},
                          "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))
                    accum_loss = 0.0
                    accum_count = 0

            # Save checkpoint
            if rank == 0 and step > 0 and step % args.save_every == 0:
                save_path = os.path.join(output_dir, f"checkpoint_step{step}")
                os.makedirs(save_path, exist_ok=True)
                unwrapped = rmt_model.module if hasattr(rmt_model, "module") else rmt_model
                torch.save(unwrapped.state_dict(), os.path.join(save_path, "rmt_model.pt"))
                print(f"[RMT-Orig] Saved checkpoint step {step}", file=log_file)

            # PPL eval
            if rank == 0 and step > 0 and step % args.eval_every == 0:
                t_eval = time.time()
                eval_loss, ppl = run_ppl_eval()
                print(f"[RMT-Orig] Eval step {step}: loss={eval_loss:.4f} ppl={ppl:.2f} "
                      f"({time.time()-t_eval:.0f}s)", file=log_file)

        epoch += 1

    # Final save
    if rank == 0:
        save_path = os.path.join(output_dir, "final")
        os.makedirs(save_path, exist_ok=True)
        unwrapped = rmt_model.module if hasattr(rmt_model, "module") else rmt_model

        print("[RMT-Original] Saving final model...", file=log_file)
        unwrapped.memory_cell.model.save_pretrained(save_path)
        tok.save_pretrained(save_path)
        torch.save({
            'memory': unwrapped.memory_cell.memory.data.cpu(),
        }, os.path.join(save_path, "rmt_memory.pt"))

        elapsed = time.time() - t0
        hb = {"timestamp": datetime.datetime.now().isoformat(), "status": "completed",
              "progress": 1.0, "metrics": {"training_time_s": elapsed},
              "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))
        print(f"[RMT-Original] Done in {elapsed/3600:.1f}h", file=log_file)
        log_file.close()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    import torch.distributed as dist
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        raise
