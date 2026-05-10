"""RMT training on PG-19 with curriculum learning.

Matches Bulatov AAAI 2024 recipe:
- PG-19 long-text training data
- Curriculum learning: 1 segment → max_n_segments over training
- PPL eval at multiple segment lengths (1, 2, 4)
- Full finetuning, no LoRA
- Uses original MemoryCell + RecurrentWrapper
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

class PG19Dataset(Dataset):
    """Load PG-19 books from HuggingFace datasets library."""

    def __init__(self, split, tokenizer, min_tokens=2048, max_tokens=None, shuffle_seed=42):
        self.tokenizer = tokenizer
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens  # None = no truncation

        from datasets import load_dataset as hf_load
        ds = hf_load("emozilla/pg19", split=split)
        
        self.docs = []
        rng = random.Random(shuffle_seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        
        for i in indices:
            tokens = tokenizer.encode(ds[i]["text"], add_special_tokens=False)
            if len(tokens) >= min_tokens:
                if max_tokens:
                    tokens = tokens[:max_tokens]
                self.docs.append(tokens)
                if len(self.docs) >= 5000:  # cap to avoid OOM on dataset
                    break

        print(f"[PG19Dataset] {split}: loaded {len(self.docs)} docs "
              f"(min_tokens={min_tokens}, max_tokens={max_tokens})")

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        return torch.tensor(self.docs[idx], dtype=torch.long)


def collate_fn(batch, segment_size):
    """Pad to multiple of segment_size."""
    max_len = max(len(x) for x in batch)
    padded_len = ((max_len + segment_size - 1) // segment_size) * segment_size

    padded = []
    for x in batch:
        t = x[:padded_len]
        pad_len = padded_len - len(t)
        if pad_len > 0:
            t = torch.cat([t, torch.full((pad_len,), 0, dtype=torch.long)])
        padded.append(t)
    return torch.stack(padded)


# ====================== Main ======================

def main():
    parser = argparse.ArgumentParser(description="RMT Training on PG-19")

    # Data
    parser.add_argument("--output_dir", default="outputs/rmt_pg19")

    # Model
    parser.add_argument("--base_model", default="/root/Mixture-of-Memory/models/Llama--Llama2-7b")

    # RMT
    parser.add_argument("--num_mem_tokens", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=1024)  # segment size (excluding memory)
    parser.add_argument("--max_n_segments", type=int, default=4)
    parser.add_argument("--bptt_depth", type=int, default=-1)   # -1 = full BPTT
    parser.add_argument("--segment_alignment", type=str, default="right")

    # Curriculum
    parser.add_argument("--curriculum", action="store_true", default=True,
                        help="Gradually increase segments from 1 to max_n_segments")
    parser.add_argument("--curriculum_epochs", type=int, default=3,
                        help="Number of epochs for curriculum ramp-up")

    # Training
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=100)
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
        print(f"[RMT-PG19] Output: {output_dir}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Base model
    print(f"[RMT-PG19] Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{rank}"},
    )
    model.gradient_checkpointing_enable()
    print(f"[RMT-PG19] Base model loaded", flush=True)

    # RMT MemoryCell + RecurrentWrapper
    memory_cell = MemoryCell(model, num_mem_tokens=args.num_mem_tokens)

    rmt_config = {
        'segment_size': args.input_size,
        'max_n_segments': args.max_n_segments,
        'bptt_depth': args.bptt_depth,
        'segment_alignment': args.segment_alignment,
    }
    rmt_model = RecurrentWrapper(memory_cell, **rmt_config)
    rmt_model = rmt_model.to(device=device, dtype=torch.bfloat16)

    # Monkey-patch process_outputs for DDP compatibility
    _orig_process_outputs = RecurrentWrapper.process_outputs
    def _safe_process_outputs(self, cell_outputs, **kwargs):
        from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions as CLO
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
        print(f"[RMT-PG19] Total params: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")

    # PG-19 Dataset
    max_train_tokens = args.input_size * args.max_n_segments
    dataset = PG19Dataset(
        split="train",
        tokenizer=tok,
        min_tokens=args.input_size,
        max_tokens=max_train_tokens * 4,  # allow longer docs for curriculum
        shuffle_seed=args.seed,
    )
    eval_dataset = PG19Dataset(
        split="validation",
        tokenizer=tok,
        min_tokens=args.input_size,
        max_tokens=args.input_size * args.max_n_segments,
        shuffle_seed=args.seed,
    )

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

    # Optimizer & scheduler (must come after dataset to know total_steps)
    optimizer = AdamW(
        rmt_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    if rank == 0:
        print(f"[RMT-PG19] {len(dataloader)} batches/epoch, {args.num_epochs} epochs, "
              f"{total_steps} total steps, LR={args.lr}, curriculum={'ON' if args.curriculum else 'OFF'}")

    # Heartbeat
    def write_heartbeat(status, **kwargs):
        if rank != 0:
            return
        hb = {"timestamp": datetime.datetime.now().isoformat(), "status": status,
              "progress": kwargs.get("progress", 0.0),
              "metrics": kwargs.get("metrics", {}),
              "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))

    write_heartbeat("running", progress=0.0)

    # Eval helper: PPL at specific number of segments
    def run_ppl_eval(n_segments):
        """Compute PPL on eval_dataset using exactly n_segments segments."""
        rmt_model.eval()
        total_loss = 0.0
        total_tokens = 0
        max_tokens = args.input_size * n_segments
        n_eval = min(args.eval_samples, len(eval_dataset))
        
        with torch.no_grad():
            for idx in range(n_eval):
                sample = eval_dataset[idx].unsqueeze(0).to(device)
                # Truncate to exact n_segments
                sample = sample[:, :max_tokens]
                # Pad to multiple of segment_size
                pad_len = max_tokens - sample.shape[1]
                if pad_len > 0:
                    sample = torch.cat([sample, torch.full((1, pad_len), 0, dtype=torch.long, device=device)], dim=1)
                
                outputs = rmt_model(input_ids=sample, labels=sample)
                loss = outputs['loss']
                n_tokens = sample.numel() - 1
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
        
        rmt_model.train()
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        return avg_loss, ppl

    def get_curriculum_segments(epoch, step_in_epoch, steps_per_epoch):
        """Determine max segments for this step based on curriculum schedule."""
        if not args.curriculum or args.max_n_segments <= 1:
            return args.max_n_segments
        
        # Linear ramp from 1 to max_n_segments over curriculum_epochs
        total_progress = epoch + step_in_epoch / max(steps_per_epoch, 1)
        progress = min(total_progress / args.curriculum_epochs, 1.0)
        n_seg = max(1, int(round(1 + progress * (args.max_n_segments - 1))))
        return min(n_seg, args.max_n_segments)

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

            # Curriculum: dynamically adjust number of segments
            curriculum_seg = get_curriculum_segments(epoch, batch_idx, len(dataloader))
            max_tokens_this = args.input_size * curriculum_seg
            input_ids = input_ids[:, :max_tokens_this]
            labels = labels[:, :max_tokens_this]

            # Pad to exact multiple of segment_size
            pad_len = max_tokens_this - input_ids.shape[1]
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.full((input_ids.shape[0], pad_len), 0, dtype=torch.long, device=device)], dim=1)
                labels = torch.cat([labels, torch.full((labels.shape[0], pad_len), 0, dtype=torch.long, device=device)], dim=1)

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
                    print(f"[RMT-PG19] step {step}/{total_steps} epoch={epoch} segs={curriculum_seg} "
                          f"loss={avg_l:.4f} lr={lr:.2e} ETA={eta:.1f}h",
                          file=log_file, flush=True)
                    write_heartbeat("running", progress=step / total_steps,
                                   metrics={"loss": avg_l, "lr": lr, "step": step, 
                                           "epoch": epoch, "curriculum_segs": curriculum_seg})
                    accum_loss = 0.0
                    accum_count = 0

            # Save checkpoint
            if rank == 0 and step > 0 and step % args.save_every == 0:
                save_path = os.path.join(output_dir, f"checkpoint_step{step}")
                os.makedirs(save_path, exist_ok=True)
                unwrapped = rmt_model.module if hasattr(rmt_model, "module") else rmt_model
                torch.save({
                    'memory': unwrapped.memory_cell.memory.data.cpu(),
                    'step': step,
                    'epoch': epoch,
                }, os.path.join(save_path, "rmt_memory.pt"))
                print(f"[RMT-PG19] Saved checkpoint step {step}", file=log_file, flush=True)

            # PPL eval at different segment lengths
            if rank == 0 and step > 0 and step % args.eval_every == 0:
                t_eval = time.time()
                results = {}
                for n_seg in [1, 2, args.max_n_segments]:
                    eval_loss, ppl = run_ppl_eval(n_seg)
                    results[f"ppl_{n_seg}seg"] = round(ppl, 2)
                    print(f"[RMT-PG19] Eval step {step}: {n_seg}seg loss={eval_loss:.4f} ppl={ppl:.2f}",
                          file=log_file, flush=True)
                print(f"[RMT-PG19] Eval done in {time.time()-t_eval:.0f}s, results={results}",
                      file=log_file, flush=True)
                write_heartbeat("running", progress=step / total_steps,
                               metrics={"step": step, "eval": results})

        epoch += 1
        if rank == 0:
            print(f"[RMT-PG19] Epoch {epoch}/{args.num_epochs} done", file=log_file, flush=True)

    # Final save + final eval
    if rank == 0:
        save_path = os.path.join(output_dir, "final")
        os.makedirs(save_path, exist_ok=True)
        unwrapped = rmt_model.module if hasattr(rmt_model, "module") else rmt_model

        print("[RMT-PG19] Saving final model...", file=log_file, flush=True)
        unwrapped.memory_cell.model.save_pretrained(save_path)
        tok.save_pretrained(save_path)
        torch.save({
            'memory': unwrapped.memory_cell.memory.data.cpu(),
            'num_mem_tokens': args.num_mem_tokens,
        }, os.path.join(save_path, "rmt_memory.pt"))

        # Final comprehensive eval
        print("[RMT-PG19] Running final PPL eval...", file=log_file, flush=True)
        final_results = {}
        for n_seg in [1, 2, 4, args.max_n_segments]:
            eval_loss, ppl = run_ppl_eval(n_seg)
            final_results[f"ppl_{n_seg}seg"] = round(ppl, 2)
            print(f"[RMT-PG19] Final {n_seg}seg: loss={eval_loss:.4f} ppl={ppl:.2f}",
                  file=log_file, flush=True)
        
        # Write results
        with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"[RMT-PG19] Final eval results: {final_results}", file=log_file, flush=True)

        elapsed = time.time() - t0
        write_heartbeat("completed", progress=1.0,
                       metrics={"training_time_s": elapsed, "final_eval": final_results})
        print(f"[RMT-PG19] Done in {elapsed/3600:.1f}h", file=log_file, flush=True)
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
