#!/usr/bin/env python3
"""Paper C P-C1 arm A2: param-matched LoRA finetune of the FULL 32-layer OLMo-2
on the SQuAD SFT packed chunks. Minimal fork of train_olmo2_arch_probe2.py —
does NOT modify that trainer (the freeze-graft/full-FT/from-scratch arms), only
reuses its data utilities (imported from train_semantic_bottleneck_1b).

Construction
------------
* Load the pretrained OLMo-2-7B (all 32 layers) via Olmo2ForCausalLM.from_pretrained
  in bf16 (FROZEN base -> no optimizer state, so bf16 is fine and memory-light).
* Freeze every base param, attach peft LoRA to the 7 linear submodules
  (q/k/v/o_proj + gate/up/down_proj) of ALL 32 layers. LoRA adapter params are the
  ONLY trainables (kept fp32 by peft), optimised with AdamW.
* Param match: A4's fresh-2 grafted layers hold ~405M params. LoRA trainable count
  = r * sum_over_targets(in+out) * n_layers = r * 78080 * 32 = r * 2,498,560.
  r=160 -> ~399.8M ≈ the A4 fresh-cap (the "param-matched" headline). --lora_rank
  overrides (reference r=64 also run). NOTE this matches the fresh *layers*, not
  A4's full trainable set (which also includes lm_head+embed+norm at low LR); the
  report states the match basis.

Optimizer / precision DEVIATION from the fp32-AdamW arms: base is bf16-FROZEN
(no optimizer state) and only the fp32 LoRA adapters are trained with AdamW.
This is the standard LoRA setup and the ONLY way the 7B full-32L base + adapters
fit a single H20; the report notes it.

Eval: at the end we merge_and_unload the LoRA into the base and save_pretrained a
standard OLMo-2 HF dir at <output_dir>/merged, so the existing eval scripts
(eval_olmo2_probe2_downstream.py / eval_olmo2_closedbook_qa.py, base mode
--base_model <merged>) load it with ZERO special-casing. Adapter checkpoints are
also saved at intervals (small) for resume/inspection.

Data / loss: identical packed [N,2048] SQuAD SFT chunks + full-LM loss as the
other arms (NpyChunkDataset), so all four arms are strictly comparable.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import Olmo2ForCausalLM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_semantic_bottleneck_1b import (  # noqa: E402
    NpyChunkDataset,
    collate_fn,
    get_lr,
    _nullctx,
)
from peft import LoraConfig, get_peft_model, TaskType  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# per-layer LoRA "in+out" summed over the 7 default targets (hidden 4096, inter 11008):
#   q/k/v/o_proj 4096+4096=8192 (x4) ; gate/up 4096+11008=15104 (x2) ; down 11008+4096=15104
#   => 4*8192 + 3*15104 = 78080 ; * n_layers => trainable = r * 78080 * n_layers
PER_LAYER_INOUT = 78080


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_path", type=str,
                   default="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B")
    p.add_argument("--lora_rank", type=int, default=160,
                   help="LoRA rank. r=160 ~= 399.8M trainable ~= A4 fresh-2 layers (405M)")
    p.add_argument("--lora_alpha", type=int, default=0,
                   help="0 -> 2*rank (scaling=2, standard). Else the literal alpha.")
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4, help="LoRA adapter LR")
    p.add_argument("--min_lr", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=150)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0
    is_main = rank == 0

    set_seed(args.seed)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        eff_bs = args.batch_size * args.grad_accumulation_steps * world_size
        logger.info(f"=== OLMo-2 LoRA SFT [A2 param-matched] r={args.lora_rank} ===")
        logger.info(f"world_size={world_size} bs={args.batch_size} "
                    f"gaccum={args.grad_accumulation_steps} eff_bs={eff_bs} "
                    f"seq_len={args.seq_len} lr={args.lr} max_steps={args.max_steps}")

    # ---- frozen bf16 base + LoRA adapters ----
    base = Olmo2ForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True)
    L = int(base.config.num_hidden_layers)
    base.config.use_cache = False
    for prm in base.parameters():
        prm.requires_grad = False

    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    alpha = args.lora_alpha if args.lora_alpha > 0 else 2 * args.lora_rank
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank, lora_alpha=alpha, lora_dropout=args.lora_dropout,
        target_modules=targets,
    )
    model = get_peft_model(base, lora_cfg)
    n_train = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    expected = args.lora_rank * PER_LAYER_INOUT * L
    if is_main:
        logger.info(f"[lora] r={args.lora_rank} alpha={alpha} targets={targets} "
                    f"layers=all {L} -> trainable {n_train/1e6:.1f}M "
                    f"(formula r*{PER_LAYER_INOUT}*{L}={expected/1e6:.1f}M) "
                    f"| A4 fresh-2 ref ~405M")
        with open(os.path.join(args.output_dir, "arch_meta.json"), "w") as f:
            json.dump({
                "arm": "A2_lora_parammatched", "model_family": "olmo2",
                "base_model_path": args.model_path, "lora_rank": args.lora_rank,
                "lora_alpha": alpha, "lora_targets": targets, "num_hidden_layers": L,
                "n_trainable": n_train, "seq_len": args.seq_len, "lr": args.lr,
                "optimizer": "adamw_fp32_adapters_bf16_frozen_base", "seed": args.seed,
            }, f, indent=2)

    model = model.to(device)
    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ---- data ----
    ds = NpyChunkDataset(args.data_path, args.seq_len)
    if is_main:
        logger.info(f"dataset rows={len(ds)} seq_len={ds.seq_len} from {args.data_path}")
    if ddp:
        sampler = DistributedSampler(ds, shuffle=True)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True,
                            drop_last=True, multiprocessing_context="fork")
    else:
        sampler = None
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True,
                            drop_last=True, multiprocessing_context="fork")

    trainable = [pp for pp in model.parameters() if pp.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=args.weight_decay)

    model.train()
    step = 0
    micro = 0
    accum_loss = 0.0
    accum_cnt = 0
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    data_iter = iter(loader)
    t0 = time.time()

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        is_accum_boundary = (micro + 1) % args.grad_accumulation_steps == 0
        sync_ctx = model.no_sync() if (ddp and not is_accum_boundary) else _nullctx()
        with sync_ctx:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss / args.grad_accumulation_steps
            loss.backward()
        accum_loss += loss.item() * args.grad_accumulation_steps
        accum_cnt += 1
        micro += 1

        if is_accum_boundary:
            lr_now = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr_now
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if is_main and step % args.log_every == 0:
                avg = accum_loss / max(accum_cnt, 1)
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 1e9
                logger.info(f"[step {step:5d}/{args.max_steps}] loss={avg:.4f} "
                            f"ppl={math.exp(min(avg,20)):.2f} lr={lr_now:.2e} "
                            f"gnorm={float(gnorm):.2f} {dt/args.log_every:.2f}s/step "
                            f"maxmem={mem:.1f}GB")
                accum_loss = 0.0
                accum_cnt = 0
                t0 = time.time()

            if is_main and step % args.save_every == 0 and step > 0:
                adir = os.path.join(args.output_dir, f"adapter_step{step}")
                (model.module if hasattr(model, "module") else model).save_pretrained(adir)
                logger.info(f"saved adapter {adir}")

    # ---- final: save adapter + merged full model for the eval harness ----
    if is_main:
        m = model.module if hasattr(model, "module") else model
        adir = os.path.join(args.output_dir, "adapter_final")
        m.save_pretrained(adir)
        logger.info(f"saved final adapter {adir}")
        merged = m.merge_and_unload()
        mdir = os.path.join(args.output_dir, "merged")
        merged.save_pretrained(mdir, safe_serialization=True)
        # copy tokenizer for a self-contained merged dir
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(args.model_path, local_files_only=True).save_pretrained(mdir)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"tokenizer copy to merged dir skipped: {e}")
        logger.info(f"[A2] merged LoRA -> {mdir} (eval with --base_model {mdir})")
        logger.info(f"DONE [A2 lora r={args.lora_rank}] at step {step}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
