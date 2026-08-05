#!/usr/bin/env python3
"""General instruction SFT driver for the OLMo-2 P2.4 repairability diagnostic.

Fine-tunes ANY of the three P2.4 arms on the SAME general instruction data, token
budget, optimizer, LR and seed:
  * full 32L base       -- vanilla OLMo-2-1124-7B (no --ckpt; --full_base)
  * keep14+fresh2@200k  -- healed pruned ckpt (--ckpt .../keep14fresh2/final.pt)
  * ShortGPT-16@200k    -- healed pruned ckpt (--ckpt .../shortgpt16/final.pt,
                           loaded as keep_front_layers=16 n_fresh_layers=0)
plus an optional keep14 equal-token NTP continuation COMPUTE CONTROL
(--data_mode ntp --data_path <dolmino.npy>): identical token budget, plain
next-token prediction on Dolmino instead of instruction SFT.

NO arch drift: model construction reuses load_pruned_model / load_base_model from
eval_olmo2_probe2_ppl.py (fp32 master weights, bf16-autocast forward, strict load).
Checkpoints are written in the SAME dict contract the prune-heal trainer uses
(model_state + keep_front_layers/n_fresh_layers/... meta) so EVERY Paper B eval
harness (eval_olmo2_probe2_ppl.py / _downstream.py / eval_olmo2_mmlu_content.py /
eval_olmo2_closedbook_qa.py) reloads the SFT'd model verbatim -- including the
full-32L arm, saved with keep_front_layers=32 n_fresh_layers=0 so load_pruned_model
rebuilds the identical 32-layer shell.

Eval AFTER SFT still uses the Paper B base protocol (chat_template=False / no BOS /
LL-based MC) unchanged; the SFT role template is a training-time interface only.

Recipe defaults (SHARED across arms -- do not vary per arm):
  fp32 master weights + bf16 autocast, AdamW betas (0.9,0.95), single uniform LR
  (--lr, default 1e-5), cosine to --min_lr, warmup --warmup_steps, wd 0.1,
  grad_clip 1.0, seq_len 2048, gradient checkpointing, seed 42.
  H20 (97.8GB): BS=4 GA=4 nproc=8 (eff batch 128). B200: BS=16 GA=1.

--max_steps sets the token budget: budget = max_steps * BS * GA * nproc * seq_len.
Pass the SAME --max_steps to all arms. --dry_run_build builds arm 0 + reports the
param/step math on CPU and exits (no GPU, no data, no DDP).
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
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (PROJECT_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ckpt_rotation import (  # noqa: E402
    add_rotation_args,
    rotate_checkpoints,
    rotation_kwargs_from_args,
)


# NO arch drift: same loaders every Paper B eval uses.
from eval_olmo2_probe2_ppl import (  # noqa: E402
    build_pruned_shell,
    load_base_model,
    load_pruned_model,
)
from train_semantic_bottleneck_1b import get_lr  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
class SftNpyDataset(Dataset):
    """Response-only-masked SFT sequences from prepare_olmo2_sft_data.py:
    <tag>_input_ids.npy [N,L] uint32 + <tag>_labels.npy [N,L] int32 (-100 mask)."""

    def __init__(self, ids_path, labels_path, seq_len):
        self.ids = np.load(ids_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        assert self.ids.shape == self.labels.shape, (self.ids.shape, self.labels.shape)
        self.seq_len = min(seq_len, self.ids.shape[1])

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        ids = np.asarray(self.ids[idx, : self.seq_len]).astype(np.int64)
        lab = np.asarray(self.labels[idx, : self.seq_len]).astype(np.int64)
        return {"input_ids": torch.from_numpy(ids), "labels": torch.from_numpy(lab)}


class NtpNpyDataset(Dataset):
    """Plain NTP over [N,L] uint32 tokens (compute control): labels == input_ids
    (OLMo2ForCausalLM shifts + masks internally; no -100 needed)."""

    def __init__(self, path, seq_len):
        self.arr = np.load(path, mmap_mode="r")
        assert self.arr.ndim == 2, self.arr.shape
        self.seq_len = min(seq_len, self.arr.shape[1])

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        row = np.asarray(self.arr[idx, : self.seq_len]).astype(np.int64)
        t = torch.from_numpy(row)
        return {"input_ids": t, "labels": t.clone()}


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def _load_model_for_sft(args, device, is_main):
    """Return (model, keep, fresh, cfg_num_layers). Reuses the eval loaders so the
    arch is byte-identical to every other Paper B eval; then flips to train mode."""
    if args.ckpt:
        model, meta = load_pruned_model(
            args.ckpt, args.base_model, args.keep_front_layers,
            args.n_fresh_layers, device)
        keep = meta["keep_front_layers"]
        fresh = meta["n_fresh_layers"]
        n_layers = meta["num_hidden_layers"]
    else:
        if not args.base_model:
            raise ValueError("no --ckpt -> full-base SFT requires --base_model")
        model, meta = load_base_model(args.base_model, device)
        n_layers = meta["num_hidden_layers"]
        # full model reloads via load_pruned_model with keep=n_layers, fresh=0.
        keep = n_layers
        fresh = 0
    if is_main:
        logger.info(f"[init] loaded {'ckpt='+args.ckpt if args.ckpt else 'FULL BASE '+args.base_model} "
                    f"-> keep={keep} fresh={fresh} layers={n_layers}")
    model = model.to(torch.float32)  # fp32 master weights (loaders already fp32)
    model.train()
    return model, keep, fresh, n_layers


def _save(model, optimizer, args, step, keep, fresh, n_layers, final=False):
    root = model.module if hasattr(model, "module") else model
    name = "final" if final else f"step{step}"
    path = os.path.join(args.output_dir, f"{name}.pt")
    tmp = path + ".tmp"
    rng = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save({
        "model_state": root.state_dict(),
        "step": step,
        "optimizer_state": optimizer.state_dict(),
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "train_args": vars(args),
        "rng_state": rng,
        # arch descriptors -> every Paper B eval rebuilds the identical shell.
        "model_family": "olmo2",
        "base_model_path": args.base_model,
        "keep_front_layers": keep,
        "n_fresh_layers": fresh,
        "num_hidden_layers": n_layers,
        "tie_word_embeddings": False,
        "seq_len": args.seq_len,
        "sft_arm": args.arm_name,
        "data_mode": args.data_mode,
    }, tmp)
    os.replace(tmp, path)
    logger.info(f"[save] {path} (step={step} keep={keep} fresh={fresh})")

    # --- rolling retention (shared policy, see scripts/ckpt_rotation.py) -----
    # Keep the --keep_last_n newest step*.pt + step0 + every --keep_steps entry
    # + the newest --keep_milestones multiples of --milestone_every; delete the
    # rest, so a long run cannot fill the disk. final.pt is NEVER rotated; the
    # just-written path is never removed; a failed/empty save rotates nothing;
    # --keep_last_n 0 disables rotation entirely (dense-save opt-out). Reached
    # only on the saving rank (both call sites are behind `is_main`).
    # This is the ONLY trainer using atomic writes (tmp -> os.replace), so
    # rotation runs strictly AFTER os.replace and also sweeps stale *.pt.tmp
    # left behind by an interrupted save.
    if not final:
        rotate_checkpoints(
            args.output_dir,
            just_written=path,
            log=logger.info,
            sweep_tmp=True,
            **rotation_kwargs_from_args(args),
        )
    return path


def main():
    p = argparse.ArgumentParser()
    # init / arch
    p.add_argument("--base_model", type=str, required=True,
                   help="vanilla OLMo-2 path (cfg source for pruned; full model for base)")
    p.add_argument("--ckpt", type=str, default="",
                   help="healed prune ckpt to SFT from (omit -> full-base SFT)")
    p.add_argument("--keep_front_layers", type=int, default=None)
    p.add_argument("--n_fresh_layers", type=int, default=None)
    p.add_argument("--arm_name", type=str, default="arm")
    # data
    p.add_argument("--data_mode", type=str, default="sft", choices=["sft", "ntp"])
    p.add_argument("--sft_ids", type=str, default="",
                   help="sft mode: <tag>_input_ids.npy")
    p.add_argument("--sft_labels", type=str, default="",
                   help="sft mode: <tag>_labels.npy")
    p.add_argument("--data_path", type=str, default="",
                   help="ntp mode (compute control): dolmino .npy [N,L] uint32")
    # optimisation (SHARED across arms)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accumulation_steps", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=2000,
                   help="token budget = max_steps*BS*GA*nproc*seq_len; SAME for all arms")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=500)
    # checkpoint rotation. Defaults: keep the 3 newest step*.pt, no milestone
    # retention (milestone_every=0), --keep_steps empty. Previously this trainer
    # had NO rotation at all, so these defaults are the first bound on its ckpt
    # volume; pass --keep_last_n 0 to restore the old keep-everything behaviour.
    add_rotation_args(p, default_keep_last_n=3, default_milestone_every=0,
                      default_keep_milestones=0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--dry_run_build", action="store_true",
                   help="CPU: build the model, print param/step/token math, exit "
                        "(no GPU / no data / no DDP)")
    args = p.parse_args()

    # ---- DDP setup ----
    use_cuda = torch.cuda.is_available()
    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank, world, local_rank = 0, 1, 0
        device = torch.device("cuda" if use_cuda else "cpu")
    is_main = rank == 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    eff_batch = args.batch_size * args.grad_accumulation_steps * world
    token_budget = args.max_steps * eff_batch * args.seq_len
    if is_main:
        logger.info(f"[cfg] arm={args.arm_name} mode={args.data_mode} world={world} "
                    f"BS={args.batch_size} GA={args.grad_accumulation_steps} "
                    f"eff_batch={eff_batch} seq_len={args.seq_len} "
                    f"max_steps={args.max_steps} -> token_budget={token_budget:,}")

    # ---- dry build (CPU, no DDP init needed but harmless if launched plain) ----
    if args.dry_run_build:
        if ddp:
            dist.destroy_process_group()
        # build on CPU
        if args.ckpt:
            ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            keep = int(ck.get("keep_front_layers"))
            fresh = int(ck.get("n_fresh_layers", 0))
            m, cfg = build_pruned_shell(args.base_model, keep, fresh, torch.float32)
            n_layers = cfg.num_hidden_layers
        else:
            from transformers import Olmo2Config
            cfg = Olmo2Config.from_pretrained(args.base_model, local_files_only=True)
            n_layers = cfg.num_hidden_layers
            keep, fresh = n_layers, 0
            m = None
        n_params = (sum(pp.numel() for pp in m.parameters()) if m is not None
                    else "(full base -- not instantiated in dry build)")
        logger.info(f"[dry_run_build] arm={args.arm_name} keep={keep} fresh={fresh} "
                    f"layers={n_layers} params={n_params}")
        logger.info(f"[dry_run_build] token_budget={token_budget:,} "
                    f"(max_steps={args.max_steps} * eff_batch={eff_batch} * "
                    f"seq_len={args.seq_len})")
        return

    if not use_cuda:
        raise RuntimeError("CUDA required (use --dry_run_build for CPU math)")

    # ---- model ----
    model, keep, fresh, n_layers = _load_model_for_sft(args, device, is_main)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ---- data ----
    if args.data_mode == "sft":
        if not (args.sft_ids and args.sft_labels):
            raise ValueError("sft mode requires --sft_ids and --sft_labels")
        ds = SftNpyDataset(args.sft_ids, args.sft_labels, args.seq_len)
    else:
        if not args.data_path:
            raise ValueError("ntp mode requires --data_path")
        ds = NtpNpyDataset(args.data_path, args.seq_len)
    if args.max_rows and args.max_rows > 0:
        ds.ids = ds.ids[: args.max_rows] if hasattr(ds, "ids") else ds.arr
    if is_main:
        logger.info(f"[data] {args.data_mode} rows={len(ds)} seq_len={ds.seq_len}")

    if ddp:
        sampler = DistributedSampler(ds, shuffle=True, seed=args.seed)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True,
                            drop_last=True, multiprocessing_context="fork")
    else:
        sampler = None
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True,
                            drop_last=True, multiprocessing_context="fork")

    # ---- optimizer: single uniform LR (SHARED recipe across arms) ----
    decay, nodecay = [], []
    root = model.module if hasattr(model, "module") else model
    for name, pp in root.named_parameters():
        if not pp.requires_grad:
            continue
        (decay if pp.ndim >= 2 else nodecay).append(pp)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    model.train()
    step = 0
    micro = 0
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    t0 = time.time()
    done = False
    while not done:
        if ddp and sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            is_accum_boundary = (micro + 1) % args.grad_accumulation_steps == 0
            sync_ctx = (model.no_sync() if (ddp and not is_accum_boundary)
                        else _nullctx())
            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=input_ids, labels=labels)
                    loss = out.loss / args.grad_accumulation_steps
                loss.backward()
            accum_loss += float(loss.item())
            micro += 1
            if is_accum_boundary:
                lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr,
                            args.min_lr)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(root.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if is_main and step % args.log_every == 0:
                    dt = time.time() - t0
                    logger.info(f"[step {step}/{args.max_steps}] "
                                f"loss={accum_loss:.4f} lr={lr:.2e} "
                                f"{dt/max(step,1):.2f}s/step")
                accum_loss = 0.0
                if is_main and step % args.save_every == 0 and step > 0:
                    _save(model, optimizer, args, step, keep, fresh, n_layers)
                if step >= args.max_steps:
                    done = True
                    break
        epoch += 1

    if is_main:
        _save(model, optimizer, args, step, keep, fresh, n_layers, final=True)
        logger.info(f"[done] arm={args.arm_name} steps={step} "
                    f"time={(time.time()-t0)/60:.1f}min")
    if ddp:
        dist.destroy_process_group()


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
