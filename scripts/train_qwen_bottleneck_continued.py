#!/usr/bin/env python3
"""Continued-pretrain a REAL Qwen3-8B with a mid-layer semantic funnel.

Pretrain-paper upgrade (2026-07-09). Every prior bottleneck arm was a
1B/3B *from-scratch* weak model (ppl 20+, near-random downstream acc), so a
reviewer can always ask "does the cache-friendliness story survive on a strong
model / real tasks?". This script starts from the *pretrained* Qwen3-8B backbone
and does a short **continued pretrain** that inserts a hard low-rank funnel on
the output of decoder layer ``j`` (``--bottleneck_layer``), forcing every bit of
information the upper "generation" stack (layers ``j+1..L``) receives to pass
through a rank-<=``bottleneck_dim`` channel:

    h_j  ->  down(d -> d_bottle)  ->  GELU  ->  up(d_bottle -> d)  ->  h_j'   (NO residual)

This is exactly the quantity QCMem caches at mid depth (``h_j``): making that
representation *compressible* is the point. After this continued pretrain we can
compare, on real RULER / BABILONG, "stock Qwen + QCMem" vs "funnel-Qwen + QCMem"
and show the funnel model is cache-friendlier.

We REUSE:
  * ``semantic_bottleneck_model.BottleneckLayer`` — the exact funnel wrapper
    (down/GELU/up, no residual) used by the from-scratch arms.
  * the ``train_qcmem_distill`` pipeline design — real Qwen3-8B load
    (``AutoModelForCausalLM``, ``local_files_only=True``, bf16), Qwen-tokenizer
    on-the-fly tokenisation of raw ``data/pg19_train.jsonl``, DDP setup, cosine
    LR + warmup, offline-safe wandb.

TRAINING STRATEGY (option (b): freeze bottom, train funnel + upper stack)
------------------------------------------------------------------------
The base's bottom-half semantics are already strong (QCMem shows ``h_12`` caches
rich semantics), so we do NOT relearn them. Instead we FREEZE the embeddings and
``layers[0:unfreeze_from]`` and train ONLY: the funnel (always — it is randomly
initialised and initially DESTROYS the representation, so it MUST adapt), the
layers ``layers[unfreeze_from:]`` (the "generation" stack that must learn to read
from the compressed representation), plus ``model.norm`` and ``lm_head``.
``--unfreeze_from`` defaults to ``--bottleneck_layer`` so the whole upper stack
(the wrapped funnel layer inclusive) co-adapts to the funnel.

Why option (b) not (a) full continued: (1) the readout problem is entirely in
the *upper* stack + funnel — the encoder below ``j`` is already good and we want
to preserve it; (2) freezing the bottom halves the optimiser-state / gradient
memory and lets us use a healthier LR (funnel is random, so this is closer to
"retrain the readout" than "gently finetune everything") — default ``lr=1e-4``,
large enough that bf16 param updates do not underflow, small enough not to wreck
the pretrained upper layers. If you instead want option (a) full continued, set
``--unfreeze_from 0 --lr 1e-5`` (note: pure-bf16 AdamW at lr=1e-5 risks update
underflow; option (b)+lr=1e-4 is the recommended, memory-cheaper default).

Checkpoints save the FULL model ``state_dict`` (funnel weights are in
``model.layers[j].down/up`` and the wrapped layer under
``model.layers[j].inner.*``) + an ``arch_meta.json`` recording
``bottleneck_layer / bottleneck_dim / model_path / unfreeze_from`` so an eval
script can rebuild the exact arch (load stock Qwen from ``model_path``, inject
the funnel, ``load_state_dict``).

Red line: ``eval_interval`` is intentionally absent — inline BABILong eval in a
DDP loop desyncs ranks and triggers the 30-min NCCL watchdog SIGABRT. Eval
checkpoints OFFLINE.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ckpt_rotation import (  # noqa: E402
    add_rotation_args,
    rotate_checkpoints,
    rotation_kwargs_from_args,
)


from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from semantic_bottleneck_model import BottleneckLayer  # noqa: E402


# --------------------------------------------------------------------------- #
# distributed helpers
# --------------------------------------------------------------------------- #
def _dist_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def _is_main(rank):
    return rank == 0


# --------------------------------------------------------------------------- #
# funnel injection + freeze (works on a REAL pretrained Qwen3ForCausalLM)
# --------------------------------------------------------------------------- #
def inject_bottleneck(model, bottleneck_layer: int, bottleneck_dim: int, dtype):
    """Wrap ``model.model.layers[bottleneck_layer]`` with the funnel in-place.

    Reuses the exact ``BottleneckLayer`` (down->GELU->up, NO residual) from the
    from-scratch arms; its ``forward(*args, **kwargs)`` transparently forwards to
    the wrapped Qwen3DecoderLayer (whose transformers>=5 forward returns a plain
    tensor) and funnels the output, so the parent ``Qwen3Model`` layer loop keeps
    calling ``decoder_layer(hidden_states, attention_mask=..., position_embeddings=
    ..., ...)`` unchanged. The wrapped inner layer stays a ``GradientCheckpointing
    Layer`` submodule, so ``gradient_checkpointing_enable()`` still checkpoints it.
    """
    if not (bottleneck_dim and bottleneck_dim > 0):
        return model  # bottleneck_dim<=0 -> no funnel (stock-continued baseline arm)
    L = int(model.config.num_hidden_layers)
    assert 0 <= bottleneck_layer < L, (bottleneck_layer, L)
    inner = model.model.layers[bottleneck_layer]
    device = next(inner.parameters()).device
    wrapped = BottleneckLayer(inner, model.config.hidden_size, bottleneck_dim)
    wrapped = wrapped.to(device=device, dtype=dtype)
    model.model.layers[bottleneck_layer] = wrapped
    return model


def apply_freeze(model, unfreeze_from: int, bottleneck_layer: int):
    """Freeze embeddings + ``layers[0:unfreeze_from]``; train the rest.

    Always unfreeze the funnel (down/up) at ``bottleneck_layer`` regardless of
    ``unfreeze_from`` — it is randomly initialised and MUST adapt. Also unfreezes
    ``model.norm`` and ``lm_head``.
    """
    L = int(model.config.num_hidden_layers)
    for p in model.parameters():
        p.requires_grad = False
    for i in range(unfreeze_from, L):
        for p in model.model.layers[i].parameters():
            p.requires_grad = True
    for p in model.model.norm.parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True
    # funnel is always trainable even if it sits below the unfreeze cut
    wrapped = model.model.layers[bottleneck_layer]
    if isinstance(wrapped, BottleneckLayer):
        for p in wrapped.down.parameters():
            p.requires_grad = True
        for p in wrapped.up.parameters():
            p.requires_grad = True
    return model


# --------------------------------------------------------------------------- #
# PG19 streaming LM dataset — tokenise raw text on the fly, pack seq_len windows
# --------------------------------------------------------------------------- #
class PG19LMStream(IterableDataset):
    """Stream ``pg19_train.jsonl`` (raw wrapped text), Qwen-tokenise on the fly,
    pack into contiguous ``seq_len``-token windows for next-token LM.

    Sharding: window index ``w`` is emitted only when ``w % n_shards == shard_id``
    where ``n_shards = world_size * num_workers`` and ``shard_id = rank *
    num_workers + worker_id`` — so every (rank, dataloader-worker) sees a disjoint
    slice of the packed stream. The corpus is looped indefinitely (training is
    step-bounded).
    """

    def __init__(self, path, tokenizer, seq_len, rank, world_size, num_workers, seed):
        super().__init__()
        self.path = path
        self.tok = tokenizer
        self.seq_len = int(seq_len)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.num_workers = max(1, int(num_workers))
        self.seed = int(seed)

    def __iter__(self) -> Iterator[dict]:
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        n_shards = self.world_size * self.num_workers
        shard_id = self.rank * self.num_workers + worker_id
        buf: list[int] = []
        wcount = 0
        while True:  # loop corpus indefinitely
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    buf.extend(self.tok.encode(line, add_special_tokens=False))
                    while len(buf) >= self.seq_len:
                        w = buf[: self.seq_len]
                        buf = buf[self.seq_len:]
                        if wcount % n_shards == shard_id:
                            toks = torch.tensor(w, dtype=torch.long)
                            # LlamaForCausalLM/Qwen3ForCausalLM shift labels internally
                            yield {"input_ids": toks, "labels": toks.clone()}
                        wcount += 1


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def get_lr(step, warmup, max_steps, base_lr, min_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(prog, 1.0)))


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _save(model, args, step, final=False):
    root = model.module if hasattr(model, "module") else model
    name = "final" if final else f"step{step}"
    path = os.path.join(args.output_dir, f"{name}.pt")
    torch.save(
        {
            "model_state": root.state_dict(),
            "step": step,
            "model_path": args.model_path,
            "bottleneck_layer": args.bottleneck_layer,
            "bottleneck_dim": args.bottleneck_dim,
            "unfreeze_from": args.unfreeze_from,
            "seq_len": args.seq_len,
        },
        path,
    )

    # --- rolling retention (shared policy, see scripts/ckpt_rotation.py) -----
    # Keep the --keep_last_n newest step*.pt + step0 + every --keep_steps entry;
    # delete the rest so a long run cannot fill the disk. final.pt is NEVER
    # rotated; the just-written path is never removed; a failed/empty save
    # rotates nothing; --keep_last_n 0 disables rotation entirely. Both call
    # sites are behind `is_main`, so this is rank-0 only.
    if not final:
        rotate_checkpoints(
            args.output_dir,
            just_written=path,
            log=lambda m: print(f"[qwen-bottleneck] {m}", flush=True),
            **rotation_kwargs_from_args(args),
        )
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Continued-pretrain real Qwen3-8B with a mid-layer funnel")
    p.add_argument("--model_path", type=str,
                   default="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b")
    p.add_argument("--bottleneck_layer", type=int, default=12,
                   help="Inject funnel on the OUTPUT of decoder layer j (QCMem resume_j~12).")
    p.add_argument("--bottleneck_dim", type=int, default=512,
                   help="Funnel channel width d_bottle. <=0 = no funnel (stock-continued baseline).")
    p.add_argument("--unfreeze_from", type=int, default=-1,
                   help="Train layers[unfreeze_from:]+norm+lm_head(+funnel always); freeze below. "
                        "-1 => defaults to bottleneck_layer. 0 => full continued pretrain.")
    # data
    p.add_argument("--data_path", type=str,
                   default=os.path.join(PROJECT_ROOT, "data", "pg19_train.jsonl"))
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--n_ctx", type=int, default=0,
                   help="Unused (kept for CLI compat); plain LM packs contiguous seq_len windows.")
    p.add_argument("--num_workers", type=int, default=2)
    # optim
    p.add_argument("--batch_size", type=int, default=4, help="Per-GPU micro batch.")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=2000)
    # io
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--save_every", type=int, default=500)
    # checkpoint rotation (previously this trainer had none). Defaults keep the
    # 3 newest step*.pt; --keep_last_n 0 restores keep-everything.
    add_rotation_args(p, default_keep_last_n=3, default_milestone_every=0,
                      default_keep_milestones=0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", type=str, default="sdpa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", type=str, default="mixture-of-memory")
    p.add_argument("--wandb_run_name", type=str, default="")
    args = p.parse_args()

    if args.unfreeze_from < 0:
        args.unfreeze_from = args.bottleneck_layer

    rank, world_size, local_rank = _dist_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    is_main = _is_main(rank)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    if is_main:
        eff_bs = args.batch_size * args.grad_accum * world_size
        print(f"[qwen-bottleneck] model={args.model_path} bottleneck_layer={args.bottleneck_layer} "
              f"bottleneck_dim={args.bottleneck_dim} unfreeze_from={args.unfreeze_from}", flush=True)
        print(f"[qwen-bottleneck] world_size={world_size} bs={args.batch_size} "
              f"grad_accum={args.grad_accum} eff_bs={eff_bs} seq_len={args.seq_len} "
              f"lr={args.lr} max_steps={args.max_steps} dtype={dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation=args.attn_impl,
        trust_remote_code=True, local_files_only=True,
    ).to(device)
    model.config.use_cache = False

    inject_bottleneck(model, args.bottleneck_layer, args.bottleneck_dim, dtype)
    apply_freeze(model, args.unfreeze_from, args.bottleneck_layer)

    model.gradient_checkpointing_enable()

    n_total = sum(pp.numel() for pp in model.parameters())
    n_train = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    if is_main:
        print(f"[qwen-bottleneck] params total={n_total/1e9:.3f}B "
              f"trainable={n_train/1e9:.3f}B ({100*n_train/n_total:.1f}%)", flush=True)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "arch_meta.json"), "w") as f:
            json.dump({
                "model_path": args.model_path,
                "bottleneck_layer": args.bottleneck_layer,
                "bottleneck_dim": args.bottleneck_dim,
                "unfreeze_from": args.unfreeze_from,
                "seq_len": args.seq_len,
                "hidden_size": int(model.config.hidden_size),
                "num_hidden_layers": int(model.config.num_hidden_layers),
                "n_params": n_total,
                "n_trainable": n_train,
            }, f, indent=2)
        with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    ds = PG19LMStream(args.data_path, tokenizer, args.seq_len, rank, world_size,
                      args.num_workers, args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=(args.num_workers > 0))

    decay, no_decay = [], []
    for _, pp in (model.module if hasattr(model, "module") else model).named_parameters():
        if not pp.requires_grad:
            continue
        (no_decay if pp.ndim < 2 else decay).append(pp)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    # wandb (main only, offline-safe)
    wb = None
    if is_main and args.wandb_run_name:
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                            config=vars(args))
        except Exception as e:  # pragma: no cover
            print(f"[qwen-bottleneck] wandb init failed ({e}); continuing", flush=True)

    model.train()
    step = 0
    micro = 0
    accum_loss = 0.0
    accum_cnt = 0
    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(loader)
    t0 = time.time()

    while step < args.max_steps:
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        is_boundary = (micro + 1) % args.grad_accum == 0
        sync_ctx = model.no_sync() if (world_size > 1 and not is_boundary) else _nullctx()
        with sync_ctx:
            with torch.amp.autocast("cuda", dtype=dtype):
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss / args.grad_accum
            loss.backward()
        accum_loss += float(loss.item()) * args.grad_accum
        accum_cnt += 1
        micro += 1

        if is_boundary:
            lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in (model.module if hasattr(model, "module") else model).parameters()
                 if p.requires_grad],
                args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if is_main and step % args.log_every == 0:
                avg = accum_loss / max(accum_cnt, 1)
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"[step {step:5d}/{args.max_steps}] loss={avg:.4f} "
                      f"ppl={math.exp(min(avg, 20)):.2f} lr={lr:.2e} "
                      f"gnorm={float(gnorm):.2f} {dt/args.log_every:.2f}s/step "
                      f"maxmem={mem:.1f}GB", flush=True)
                if wb is not None:
                    wb.log({"loss": avg, "ppl": math.exp(min(avg, 20)),
                            "lr": lr, "gnorm": float(gnorm), "step": step})
                accum_loss = 0.0
                accum_cnt = 0
                t0 = time.time()

            if is_main and step % args.save_every == 0 and step > 0:
                path = _save(model, args, step)
                print(f"[qwen-bottleneck] saved {path}", flush=True)

    if is_main:
        path = _save(model, args, step, final=True)
        print(f"[qwen-bottleneck] DONE at step {step}. final -> {path}", flush=True)
        if wb is not None:
            wb.finish()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
