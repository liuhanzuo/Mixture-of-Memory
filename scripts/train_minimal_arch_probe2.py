#!/usr/bin/env python3
"""Probe #2 (the CORRECT experiment) for the minimal-architecture hypothesis.

Direction 4 / QCMem draft §3.1 "understand-then-generate division of labour":
semantic understanding saturates in the front ~half of the layers, while the
next-token-prediction (NTP / generation) ability only forms in the top layers.
Hypothesis: the top layers may be largely REDUNDANT -- a "front-j layers +
a few fresh NTP layers" model, trained end-to-end, could match the full model.

Probe #1 (``scripts/probe_minimal_arch.py``) only did a TRAINING-FREE layer-skip
on an already-trained model (front-j directly wired into the tail layers). That
is off-manifold and its 6-21x ppl blow-up is an UPPER bound on damage, NOT a test
of the hypothesis. This script runs the real test: it CONSTRUCTS a smaller model
and TRAINS it.

Construction (default keep_front=12, n_fresh=2 -> 14 layers total):
  * load the trained 16-layer 1B baseline ckpt
    (outputs/sembott_1b_base_16k/final.pt),
  * keep its FRONT 12 decoder layers (layers.0..11) + embed_tokens + final norm
    + lm_head (tie_word_embeddings=True -> lm_head is embed_tokens),
  * DROP the top 4 layers (layers.12..15),
  * append 2 FRESH standard-Llama-initialised decoder layers -> a 14-layer model.

Arms (via flags):
  * Arm A --freeze_front : freeze the inherited front 12 layers, train only the
    2 fresh layers + final norm + lm_head/embed. Tests "is the front-12 semantic
    representation already enough, only a new NTP head is missing?".
  * Arm B (default)      : train ALL 14 layers ("healing" -- let the front layers
    adapt to the shortened stack).
  * Control 2 --from_scratch : ignore the ckpt, build a 14-layer model with random
    init and train everything. Isolates "is any gap just because it has fewer
    layers?" (fair same-depth-from-scratch baseline).
Control 1 = the full 16-layer baseline itself (outputs/sembott_1b_base_16k); no
retrain needed -- just eval its ppl as the upper reference.

Shares data loading / optimiser / cosine schedule / DDP loop with
``scripts/train_semantic_bottleneck_1b.py`` (imported, not modified) and the
Llama shape with ``scripts/semantic_bottleneck_model.py`` (make_config, imported).
Checkpoints are raw ``state_dict`` (+ arch meta) so downstream eval can rebuild.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import LlamaForCausalLM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ckpt_rotation import (  # noqa: E402
    add_rotation_args,
    rotate_checkpoints,
    rotation_kwargs_from_args,
)


# Reuse (do NOT modify) the sibling scripts.
from semantic_bottleneck_model import make_config  # noqa: E402
from train_semantic_bottleneck_1b import (  # noqa: E402
    NpyChunkDataset,
    collate_fn,
    get_lr,
    _nullctx,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------
def build_minimal_model(keep_front_layers, n_fresh_layers, model_size, seq_len, dtype):
    """Fresh (random-init) Llama with (keep_front + n_fresh) decoder layers.

    Same shape as make_config(model_size) except num_hidden_layers is shrunk to
    keep_front_layers + n_fresh_layers. Front layers get overwritten later (unless
    --from_scratch); the fresh tail layers keep this random init.
    """
    cfg = make_config(model_size, seq_len=seq_len)
    total_layers = keep_front_layers + n_fresh_layers
    cfg.num_hidden_layers = total_layers
    model = LlamaForCausalLM(cfg).to(dtype)
    return model, cfg


def _copied_keys(state_dict, keep_front_layers):
    """Keys from the base ckpt we transplant: embed / final-norm / lm_head + the
    front decoder layers layers.0..keep-1. (Top layers layers.keep..L-1 dropped.)"""
    keys = []
    for k in state_dict:
        if k.startswith("model.layers."):
            try:
                lid = int(k.split(".")[2])
            except (IndexError, ValueError):
                continue
            if lid < keep_front_layers:
                keys.append(k)
        else:
            # model.embed_tokens.weight, model.norm.weight, lm_head.weight
            keys.append(k)
    return keys


def load_front_weights(model, ckpt_path, keep_front_layers, is_main):
    """Transplant front-j layers + embed + norm + lm_head from the base ckpt into
    ``model`` (leaving the fresh tail layers at their random init). Returns
    (max_abs_diff_over_copied, n_copied_keys). max diff must be 0."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state"]
    L_base = ck.get("num_hidden_layers", None)
    keep_keys = _copied_keys(sd, keep_front_layers)
    filtered = {k: sd[k] for k in keep_keys}

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    # The only allowed "missing" are the FRESH tail layers (>= keep_front_layers).
    bad_missing = []
    for mk in missing:
        if mk.startswith("model.layers."):
            lid = int(mk.split(".")[2])
            if lid < keep_front_layers:
                bad_missing.append(mk)
        else:
            bad_missing.append(mk)
    assert not bad_missing, f"front/embed/norm/head keys unexpectedly missing: {bad_missing[:5]}"
    assert not unexpected, f"unexpected keys when transplanting: {unexpected[:5]}"

    # Sanity: every transplanted param must equal the ckpt tensor elementwise.
    model_sd = model.state_dict()
    max_diff = 0.0
    for k in keep_keys:
        d = (model_sd[k].float() - sd[k].float()).abs().max().item()
        max_diff = max(max_diff, d)
    if is_main:
        logger.info(
            f"[transplant] base_L={L_base} copied {len(keep_keys)} tensors "
            f"(front {keep_front_layers} layers + embed/norm/lm_head); "
            f"fresh-tail keys left random = {len(missing)}"
        )
        logger.info(f"[sanity] max|model_param - ckpt| over transplanted tensors = {max_diff:.3e} "
                    f"({'OK exact match' if max_diff == 0.0 else 'WARN non-zero!'})")
    return max_diff, len(keep_keys)


def apply_freeze_front(model, keep_front_layers, is_main):
    """Arm A: freeze the inherited front layers; keep everything else trainable
    (fresh tail layers + final norm + lm_head/embed). Returns (n_frozen, n_train)."""
    n_frozen = n_train = 0
    for name, p in model.named_parameters():
        freeze = False
        if name.startswith("model.layers."):
            lid = int(name.split(".")[2])
            if lid < keep_front_layers:
                freeze = True
        if freeze:
            p.requires_grad_(False)
            n_frozen += p.numel()
        else:
            n_train += p.numel()
    if is_main:
        logger.info(f"[freeze] front {keep_front_layers} layers frozen: "
                    f"frozen={n_frozen/1e6:.1f}M trainable={n_train/1e6:.1f}M params")
    return n_frozen, n_train


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------
def _save(model, args, step, cfg, final=False):
    root = model.module if hasattr(model, "module") else model
    name = "final" if final else f"step{step}"
    path = os.path.join(args.output_dir, f"{name}.pt")
    torch.save({
        "model_state": root.state_dict(),
        "step": step,
        "model_size": args.model_size,
        # arch descriptors so downstream eval can rebuild an identical model.
        "keep_front_layers": args.keep_front_layers,
        "n_fresh_layers": args.n_fresh_layers,
        "num_hidden_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "freeze_front": bool(args.freeze_front),
        "from_scratch": bool(args.from_scratch),
        "init_ckpt": args.init_ckpt,
        # keep bottleneck_dim=0 tag so probe/eval loaders that expect it work.
        "bottleneck_layer": 0,
        "bottleneck_dim": 0,
        "seq_len": args.seq_len,
    }, path)
    logger.info(f"saved {path}")

    # --- rolling retention (shared policy, see scripts/ckpt_rotation.py) -----
    # Keep the --keep_last_n newest step*.pt + step0 + every --keep_steps entry
    # + the newest --keep_milestones multiples of --milestone_every; delete the
    # rest, so a long run cannot fill the disk. final.pt is NEVER rotated; the
    # just-written path is never removed; a failed/empty save rotates nothing;
    # --keep_last_n 0 disables rotation entirely (dense-save opt-out). Reached
    # only on the saving rank.
    if not final:
        rotate_checkpoints(
            args.output_dir,
            just_written=path,
            log=logger.info,
            **rotation_kwargs_from_args(args),
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--init_ckpt", type=str, default="outputs/sembott_1b_base_16k/final.pt",
                   help="base 16-layer ckpt to transplant the front layers from")
    p.add_argument("--model_size", type=str, default="1b", choices=["1b", "3b", "7b"])
    p.add_argument("--keep_front_layers", type=int, default=12)
    p.add_argument("--n_fresh_layers", type=int, default=2)
    p.add_argument("--freeze_front", action="store_true",
                   help="Arm A: freeze inherited front layers, train fresh+norm+head only")
    p.add_argument("--from_scratch", action="store_true",
                   help="Control 2: ignore init_ckpt, random-init all (keep+fresh) layers, train all")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--grad_accumulation_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=500)
    # checkpoint rotation. Defaults: keep the 3 newest step*.pt, no milestone
    # retention (milestone_every=0), --keep_steps empty. Previously this trainer
    # had NO rotation at all, so these defaults are the first bound on its ckpt
    # volume; pass --keep_last_n 0 to restore the old keep-everything behaviour.
    add_rotation_args(p, default_keep_last_n=3, default_milestone_every=0,
                      default_keep_milestones=0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--max_rows", type=int, default=0, help=">0 to subset dataset (smoke)")
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--device", type=str, default="auto",
                   help="'auto' (cuda if available else cpu), 'cpu', or 'cuda'")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed. Also passed to DistributedSampler(seed=...) -- without that the sampler silently uses its own default 0 and data order is identical across seeds.")
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

    # device / dtype selection (CPU smoke path uses fp32; GPU runs use bf16).
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        device = torch.device("cpu")
        model_dtype = torch.float32
        use_cuda = False
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        model_dtype = torch.bfloat16
        use_cuda = True

    if args.from_scratch:
        arm = f"scratch{args.keep_front_layers + args.n_fresh_layers}L"
    elif args.freeze_front:
        arm = f"frozen_front{args.keep_front_layers}+fresh{args.n_fresh_layers}"
    else:
        arm = f"healing_front{args.keep_front_layers}+fresh{args.n_fresh_layers}"

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        eff_bs = args.batch_size * args.grad_accumulation_steps * world_size
        logger.info(f"=== {args.model_size} minimal-arch probe#2 [{arm}] ===")
        logger.info(f"device={device} dtype={model_dtype} world_size={world_size} "
                    f"bs={args.batch_size} gaccum={args.grad_accumulation_steps} eff_bs={eff_bs} "
                    f"seq_len={args.seq_len} lr={args.lr} max_steps={args.max_steps}")

    # ---- build model ----
    model, cfg = build_minimal_model(
        keep_front_layers=args.keep_front_layers,
        n_fresh_layers=args.n_fresh_layers,
        model_size=args.model_size,
        seq_len=args.seq_len,
        dtype=model_dtype,
    )

    sanity_diff = None
    if args.from_scratch:
        if is_main:
            logger.info(f"[from_scratch] random-init {cfg.num_hidden_layers}-layer model "
                        f"(init_ckpt IGNORED); training all layers")
    else:
        assert args.init_ckpt and os.path.exists(args.init_ckpt), \
            f"init_ckpt not found: {args.init_ckpt}"
        sanity_diff, _ = load_front_weights(model, args.init_ckpt, args.keep_front_layers, is_main)
        if args.freeze_front:
            apply_freeze_front(model, args.keep_front_layers, is_main)

    model = model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if is_main:
        n = sum(pp.numel() for pp in model.parameters())
        n_tr = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
        logger.info(f"model params = {n/1e9:.4f}B (trainable {n_tr/1e9:.4f}B) "
                    f"num_hidden_layers={cfg.num_hidden_layers}")
        with open(os.path.join(args.output_dir, "arch_meta.json"), "w") as f:
            json.dump({
                "arm": arm,
                "model_size": args.model_size,
                "keep_front_layers": args.keep_front_layers,
                "n_fresh_layers": args.n_fresh_layers,
                "num_hidden_layers": cfg.num_hidden_layers,
                "hidden_size": cfg.hidden_size,
                "freeze_front": bool(args.freeze_front),
                "from_scratch": bool(args.from_scratch),
                "init_ckpt": args.init_ckpt,
                "seq_len": args.seq_len,
                "vocab_size": cfg.vocab_size,
                "n_params": n,
                "n_trainable": n_tr,
                "transplant_sanity_max_abs_diff": sanity_diff,
            }, f, indent=2)

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ---- data ----
    ds = NpyChunkDataset(args.data_path, args.seq_len)
    if args.max_rows and args.max_rows > 0:
        ds.arr = ds.arr[: args.max_rows]
    if is_main:
        logger.info(f"dataset rows={len(ds)} seq_len={ds.seq_len} from {args.data_path}")

    if ddp:
        # seed=args.seed is LOAD-BEARING: DistributedSampler.__iter__ builds its OWN
        # generator (g.manual_seed(self.seed + self.epoch)) and self.seed defaults to 0,
        # so torch.manual_seed()/set_seed() CANNOT reach it. Without this argument every
        # --seed value gives a BYTE-IDENTICAL data order. Do not delete as redundant.
        sampler = DistributedSampler(ds, shuffle=True, seed=args.seed)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True)
    else:
        sampler = None
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=0 if not use_cuda else 4,
                            pin_memory=use_cuda, drop_last=True)

    # ---- optimizer (only params that require grad) ----
    decay, no_decay = [], []
    for nm, pp in model.named_parameters():
        if not pp.requires_grad:
            continue
        (no_decay if pp.ndim < 2 else decay).append(pp)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    model.train()
    step = 0
    micro = 0
    accum_loss = 0.0
    accum_cnt = 0
    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(loader)
    t0 = time.time()
    epoch = 0

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=use_cuda)
        labels = batch["labels"].to(device, non_blocking=use_cuda)

        is_accum_boundary = (micro + 1) % args.grad_accumulation_steps == 0
        sync_ctx = model.no_sync() if (ddp and not is_accum_boundary) else _nullctx()
        with sync_ctx:
            if use_cuda:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=input_ids, labels=labels)
                    loss = out.loss / args.grad_accumulation_steps
            else:
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss / args.grad_accumulation_steps
            loss.backward()
        accum_loss += loss.item() * args.grad_accumulation_steps
        accum_cnt += 1
        micro += 1

        if is_accum_boundary:
            lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            gnorm = torch.nn.utils.clip_grad_norm_(
                [pp for pp in model.parameters() if pp.requires_grad], args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if is_main and step % args.log_every == 0:
                avg = accum_loss / max(accum_cnt, 1)
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
                logger.info(f"[step {step:5d}/{args.max_steps}] loss={avg:.4f} "
                            f"ppl={math.exp(min(avg,20)):.2f} lr={lr:.2e} gnorm={float(gnorm):.2f} "
                            f"{dt/args.log_every:.2f}s/step maxmem={mem:.1f}GB")
                accum_loss = 0.0
                accum_cnt = 0
                t0 = time.time()

            if is_main and step % args.save_every == 0 and step > 0:
                _save(model, args, step, cfg)

    if is_main:
        _save(model, args, step, cfg, final=True)
        logger.info(f"DONE [{arm}] at step {step}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
