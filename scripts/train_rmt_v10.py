"""RMT++ v10 training script.

Key differences from v5:
- Full BPTT (accumulate all segment losses, single backward)
- vary_n_segments (curriculum learning)
- Sandwich injection (prepended + appended memory)
- L1/L2 optional layered memory
- Reconstruction loss
- Importance routing
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from src.memory.rmt.rmt_v10 import RMTv10Model, RMTv10Config


# ====================== Dataset ======================

class RMTv10Dataset(Dataset):
    """
    Each sample is a long document. We return the raw token tensor.
    Segmentation and sandwich building happen in the training loop.
    """
    def __init__(self, data_path, tokenizer, max_tokens):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        docs = []
        with open(data_path) as f:
            for line in f:
                text = json.loads(line)["text"]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) >= 512:  # at least ~1 segment
                    docs.append(tokens)
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        tokens = self.docs[idx]
        tokens = tokens[:self.max_tokens]
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch, segment_length, max_segments):
    """
    Pad docs to multiples of segment_length.
    Returns (input_ids, labels) where labels = input_ids shifted by 1 for LM.
    """
    max_len = max(len(x) for x in batch)
    padded_len = ((max_len + segment_length - 1) // segment_length) * segment_length
    padded_len = min(padded_len, segment_length * max_segments)

    padded = []
    labels_list = []
    for x in batch:
        t = x[:padded_len]
        pad_len = padded_len - len(t)
        if pad_len > 0:
            t = torch.cat([t, torch.zeros(pad_len, dtype=torch.long)])
        padded.append(t)
        # Labels: identity (shifting happens inside RMTv10Model.forward)
        labels_list.append(t.clone())

    return torch.stack(padded), torch.stack(labels_list)


# ====================== Training ======================

def main():
    parser = argparse.ArgumentParser(description="RMT++ v10 Training")

    # Data
    parser.add_argument("--data", default="data/rmt_train_mixed.jsonl")
    parser.add_argument("--output_dir", default="outputs/rmt_v10")

    # Model
    parser.add_argument("--base_model", default="../models/Qwen--Qwen3-8b")

    # L0 memory
    parser.add_argument("--num_mem_tokens", type=int, default=16)
    parser.add_argument("--segment_length", type=int, default=1024)
    parser.add_argument("--max_segments", type=int, default=6)

    # L1 memory
    parser.add_argument("--use_l1", action="store_true", default=False)
    parser.add_argument("--l1_num_tokens", type=int, default=8)
    parser.add_argument("--l1_update_freq", type=int, default=3)
    parser.add_argument("--l1_inject_layer", type=int, default=-1)

    # L2 memory
    parser.add_argument("--use_l2", action="store_true", default=False)
    parser.add_argument("--l2_num_tokens", type=int, default=4)
    parser.add_argument("--l2_update_freq", type=int, default=6)
    parser.add_argument("--l2_inject_layer", type=int, default=-1)

    # Training
    parser.add_argument("--vary_n_segments", action="store_true", default=True)
    parser.add_argument("--no_vary_n_segments", action="store_false", dest="vary_n_segments")
    parser.add_argument("--bptt_depth", type=int, default=-1, help="-1 = full BPTT")
    parser.add_argument("--recon_loss_coef", type=float, default=0.1)
    parser.add_argument("--use_importance_routing", action="store_true", default=True)
    parser.add_argument("--no_importance_routing", action="store_false", dest="use_importance_routing")

    # Optimizer / schedule
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rmt_lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=100, help="Run lightweight PPL eval every N steps")
    parser.add_argument("--eval_data", type=str, default=None, help="Eval data path (default: use train data)")
    parser.add_argument("--eval_samples", type=int, default=8, help="Number of samples for PPL eval")

    # Full finetune
    parser.add_argument("--full_finetune", action="store_true", default=True)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
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

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id

    # Base model (full finetune)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{rank}"},
    )
    model.gradient_checkpointing_enable()

    # RMT v10
    rmt_config = RMTv10Config(
        num_mem_tokens=args.num_mem_tokens,
        segment_length=args.segment_length,
        max_n_segments=args.max_segments,
        use_l1=args.use_l1,
        l1_num_tokens=args.l1_num_tokens,
        l1_update_freq=args.l1_update_freq,
        l1_inject_layer=args.l1_inject_layer,
        use_l2=args.use_l2,
        l2_num_tokens=args.l2_num_tokens,
        l2_update_freq=args.l2_update_freq,
        l2_inject_layer=args.l2_inject_layer,
        vary_n_segments=args.vary_n_segments,
        bptt_depth=args.bptt_depth,
        recon_loss_coef=args.recon_loss_coef,
        use_importance_routing=args.use_importance_routing,
    )
    rmt_model = RMTv10Model(model, rmt_config)
    rmt_model = rmt_model.to(device=device, dtype=torch.bfloat16)

    if ddp:
        # find_unused_parameters needed because L1/L2 may be disabled
        rmt_model = DDP(rmt_model, device_ids=[rank], find_unused_parameters=True)

    # Optimizer: separate LR for backbone vs RMT params
    backbone_params = []
    rmt_params = []
    for name, param in rmt_model.named_parameters():
        if "l0" in name or "l1" in name or "l2" in name or "recon_head" in name:
            rmt_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": rmt_params, "lr": args.rmt_lr},
        ],
        weight_decay=0.01,
    )

    # Dataset
    max_tokens = args.segment_length * args.max_segments
    dataset = RMTv10Dataset(args.data, tok, max_tokens)

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

    # Training loop
    if rank == 0:
        print(f"[RMT v10] Starting training: {len(dataloader)} batches/epoch, "
              f"{args.num_epochs} epochs, {total_steps} steps")
        hb = {
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "running", "progress": 0.0, "metrics": {},
            "extra": {}, "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))

    # Eval dataset (small subset for lightweight PPL check)
    eval_data_path = args.eval_data or args.data
    eval_dataset = RMTv10Dataset(eval_data_path, tok, max_tokens)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False) if ddp else None
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=1,
        collate_fn=lambda b: collate_fn(b, args.segment_length, args.max_segments),
        shuffle=False, sampler=eval_sampler, num_workers=0,
    )
    eval_samples_list = list(range(min(args.eval_samples, len(eval_dataset))))

    def run_ppl_eval():
        """Lightweight PPL eval — only on rank 0, single sample."""
        rmt_model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for idx in eval_samples_list[:args.eval_samples]:
                sample = eval_dataset[idx]
                input_ids = sample.unsqueeze(0).to(device)
                labels = input_ids.clone()
                outputs = rmt_model(input_ids=input_ids, labels=labels)
                n_tokens = input_ids.numel()
                total_loss += outputs["loss"].item() * n_tokens
                total_tokens += n_tokens
        rmt_model.train()
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        ppl = math.exp(min(avg_loss, 20))  # cap for numerical stability
        return avg_loss, ppl

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

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            is_last_accum = (batch_idx + 1) % args.grad_accumulation_steps == 0
            ddp_context = nullcontext()
            if ddp:
                ctx_fn = rmt_model.no_sync if not is_last_accum else nullcontext
                ddp_context = ctx_fn()

            with ddp_context:
                outputs = rmt_model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]

                # Scale by grad accumulation
                (loss / args.grad_accumulation_steps).backward()

            accum_loss += loss.item()
            accum_count += 1

            if is_last_accum:
                torch.nn.utils.clip_grad_norm_(rmt_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if rank == 0 and step % args.log_every == 0:
                    avg_loss = accum_loss / accum_count
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    eta = elapsed / step * (total_steps - step) / 3600
                    num_segs = outputs.get("num_segments", "?")
                    print(
                        f"[RMT v10] Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                        f"loss={avg_loss:.4f} lr={lr:.2e} segs={num_segs} ETA={eta:.1f}h",
                        file=log_file,
                    )
                    hb = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "status": "running",
                        "progress": step / total_steps,
                        "metrics": {
                            "loss": avg_loss,
                            "learning_rate": lr,
                            "epoch": epoch,
                            "step": step,
                            "num_segments": num_segs,
                        },
                        "extra": {},
                        "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))
                    accum_loss = 0.0
                    accum_count = 0

            # Save checkpoint
            if rank == 0 and step > 0 and step % args.save_every == 0:
                save_path = os.path.join(output_dir, f"checkpoint_step{step}")
                os.makedirs(save_path, exist_ok=True)
                unwrapped = rmt_model.module if hasattr(rmt_model, "module") else rmt_model
                # Save full base model weights
                torch.save(unwrapped.base_model.state_dict(), os.path.join(save_path, "base_model.pt"))
                # Save RMT memory params
                torch.save(
                    {k: v for k, v in unwrapped.state_dict().items()
                     if any(s in k for s in ["l0", "l1", "l2", "recon_head"])},
                    os.path.join(save_path, "rmt_memory.pt"),
                )
                print(f"[RMT v10] Saved checkpoint at step {step}", file=log_file)

            # Lightweight PPL eval
            if rank == 0 and step > 0 and step % args.eval_every == 0:
                t_eval = time.time()
                eval_loss, ppl = run_ppl_eval()
                eval_time = time.time() - t_eval
                print(
                    f"[RMT v10] Eval step {step}: loss={eval_loss:.4f} ppl={ppl:.2f} ({eval_time:.0f}s)",
                    file=log_file,
                )
                hb["metrics"]["eval_ppl"] = ppl
                hb["metrics"]["eval_loss"] = eval_loss
                json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))

        epoch += 1
        if rank == 0:
            elapsed = time.time() - t0
            print(f"[RMT v10] Epoch {epoch} done, elapsed: {elapsed/3600:.1f}h", file=log_file)

    # Final save
    if rank == 0:
        unwrapped = rmt_model.module if hasattr(rmt_model, "module") else rmt_model
        base_model = unwrapped.base_model

        save_path = os.path.join(output_dir, "final")
        os.makedirs(save_path, exist_ok=True)
        base_model.save_pretrained(save_path)
        tok.save_pretrained(save_path)

        torch.save(
            {k: v for k, v in unwrapped.state_dict().items()
             if any(s in k for s in ["l0", "l1", "l2", "recon_head"])},
            os.path.join(save_path, "rmt_memory.pt"),
        )

        hb = {
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "completed",
            "progress": 1.0,
            "metrics": {"training_time_s": time.time() - t0},
            "extra": {},
            "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        json.dump(hb, open(os.path.join(output_dir, "heartbeat.json"), "w"))
        print("[RMT v10] Training complete!", file=log_file)
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
