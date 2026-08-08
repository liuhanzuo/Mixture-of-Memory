#!/usr/bin/env python3
"""CAST training for LLaMA-2-7B under plain DDP (arXiv:2509.25996v1).

Paper recipe for LLaMA (Table XI): lr 2e-5, lambda 4e-7, global batch 256,
seqlen 4096, 7500 steps, mask refresh every 10 steps, n=2 scaling groups,
eta=1/3 KL coefficient, Dolmino-Mix-1124.

WHY DDP AND NOT FSDP.  The previous attempt used FSDP FULL_SHARD.  FSDP packs
`weight` and `mask` into a FlatParameter and slices them at *different* global
offsets, so a rank's weight shard and mask shard are not element-aligned (their
numel can even differ).  The old optimizer set `mask = None` in that case and
silently ran vanilla Adam, so the selective L1 decay never happened on most
tensors -- 7.86B tokens burned, Wiki PPL 23.45.  See
Mixture-of-Memory/SparseForge_Data/docs/CAST_REPRODUCTION_AUDIT.md section 4.1.
Under DDP nothing is sharded: `weight` and `mask` are full, same-shape,
same-device tensors, so alignment holds by construction and AdamS asserts it on
every step.

Memory (per rank, LLaMA2-7B, fp32 master + fp32 Adam state):
    params 26.9 GB + grads 26.9 GB + exp_avg 26.9 GB + exp_avg_sq 26.9 GB
    + bool masks 6.5 GB + frozen bf16 teacher 13.5 GB  ~= 128 GB
which fits an L20A/B200-class 183 GB card but NOT an 80/97 GB card.  DDP does
not shard optimizer state, so adding nodes does not reduce per-card memory.

Smoke test (no long run):
    torchrun --nproc_per_node 8 train_cast_llama.py --max-steps 4 --smoke

Full run: see README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cast import (  # noqa: E402
    LLAMA2_7B_CAST_ELEMENTS,
    LLAMA2_7B_CAST_TENSORS,
    AdamS,
    build_param_groups,
    cast_loss,
    cast_scope_stats,
    convert_llama_to_cast,
    finalize_all,
    refresh_all_masks,
)
from cast.diagnostics import magnitude_report  # noqa: E402


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # --- paper hyperparameters (Table XI, LLaMA column) ---
    p.add_argument("--lr", type=float, default=2e-5, help="[paper] Table XI")
    p.add_argument("--l1-decay", type=float, default=4e-7, help="[paper] Table XI decay coefficient")
    p.add_argument("--global-batch", type=int, default=256, help="[paper] Table XI")
    p.add_argument("--seq-len", type=int, default=4096, help="[paper] Table XI")
    p.add_argument("--max-steps", type=int, default=7500, help="[paper] Table XI training steps")
    p.add_argument("--mask-period", type=int, default=10, help="[paper] T1=10, Sec. IV-A")
    p.add_argument("--scale-groups", type=int, default=2, help="[paper] n=2, Sec. VI-A")
    p.add_argument("--eta", type=float, default=1.0 / 3.0, help="[paper] Table XI KL coefficient")
    # --- implementation choices (NOT specified by the paper) ---
    p.add_argument("--kl-temperature", type=float, default=1.0,
                   help="[impl] 1.0 = paper-literal Eq.13; 2.0 = AST-code variant")
    p.add_argument("--min-lr", type=float, default=2e-6, help="[impl] cosine floor (AST alpha_f=0.1)")
    p.add_argument("--warmup", type=int, default=375, help="[impl] 5%% of 7500")
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999), help="[impl] Adam default")
    p.add_argument("--eps", type=float, default=1e-8, help="[impl] Adam default")
    p.add_argument("--grad-clip", type=float, default=1.0, help="[impl] not in paper")
    p.add_argument("--micro-batch", type=int, default=1, help="[impl] memory-driven")
    # --- plumbing ---
    p.add_argument("--model", default="models/Llama--Llama2-7b")
    p.add_argument("--data", default="data/c4_llama",
                   help="dir with train.bin/val.bin; see README for PRIMARY vs FALLBACK")
    p.add_argument("--data-dtype", default="auto", choices=["auto", "uint16", "uint32"],
                   help="auto = read from <data>/metadata.json if present, else fall back to uint16")
    p.add_argument("--out", default="outputs/cast_repro_ddp")
    p.add_argument("--project-root", default="/apdcephfs_wzc1/share_304376610/pighzliu_code")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--diag-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: skip teacher/ckpt, only prove alignment + a few steps")
    p.add_argument("--no-teacher", action="store_true", help="eta=0 ablation (pure CE)")
    return p.parse_args()


def is_master() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(msg: str) -> None:
    if is_master():
        print(f"[cast] {msg}", flush=True)


def lr_at(step: int, args) -> float:
    """Cosine with linear warmup.  [implementation_choice] -- Table XI gives only
    the peak LR; the schedule shape is ours (AST official uses alpha_f=0.1, hence
    min_lr = lr/10)."""
    if step < args.warmup:
        return args.lr * (step + 1) / max(1, args.warmup)
    prog = (step - args.warmup) / max(1, args.max_steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * min(1.0, prog)))


# ---------------------------------------------------------------------------
class BinDataset:
    """Contiguous next-token batches from a flat token .bin memmap."""

    def __init__(self, path: Path, seq_len: int, dtype: str, seed: int, rank: int, world: int):
        self.data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed + rank)
        self.rank, self.world = rank, world
        self.n = len(self.data)
        if self.n < seq_len + 1:
            raise RuntimeError(f"{path} has only {self.n} tokens")

    def batch(self, bs: int, device):
        idx = self.rng.integers(0, self.n - self.seq_len - 1, size=bs)
        x = np.stack([self.data[i : i + self.seq_len].astype(np.int64) for i in idx])
        y = np.stack([self.data[i + 1 : i + 1 + self.seq_len].astype(np.int64) for i in idx])
        return (
            torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True),
        )


# ---------------------------------------------------------------------------
def main():  # noqa: C901
    args = parse_args()
    root = Path(args.project_root)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world > 1
    if ddp:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)

    outdir = root / args.out
    if is_master():
        outdir.mkdir(parents=True, exist_ok=True)

    # ---- gradient accumulation accounting ----
    if args.global_batch % (world * args.micro_batch) != 0:
        raise ValueError(
            f"global_batch {args.global_batch} not divisible by world*micro "
            f"({world}*{args.micro_batch})"
        )
    accum = args.global_batch // (world * args.micro_batch)
    tokens_per_step = args.global_batch * args.seq_len
    log(
        f"world={world} micro={args.micro_batch} accum={accum} "
        f"global_batch={args.global_batch} tokens/step={tokens_per_step:,} "
        f"total_tokens={tokens_per_step * args.max_steps:,}"
    )

    # ---- model ----
    from transformers import LlamaForCausalLM

    model_path = root / args.model
    log(f"loading student from {model_path}")
    # fp32 master weights are REQUIRED: lambda=4e-7 is below bf16 resolution.
    model = LlamaForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.float32, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    converted = convert_llama_to_cast(model, n=2, m=4, scale_groups=args.scale_groups)
    log(f"converted {len(converted)} in-block projections to CastSparseLinear")
    model.to(device)

    # Alg. 2 lines 1-4: initialise the mask BEFORE the first optimizer step.
    n_mod, _ = refresh_all_masks(model)
    stats = cast_scope_stats(model)
    log(f"initial mask: {n_mod} modules, scope={json.dumps(stats)}")

    expected_tensors = LLAMA2_7B_CAST_TENSORS if not args.smoke else None
    expected_elements = LLAMA2_7B_CAST_ELEMENTS if not args.smoke else None
    if expected_elements is not None and stats["cast_elements"] != expected_elements:
        raise RuntimeError(
            f"scope element count {stats['cast_elements']:,} != expected {expected_elements:,}"
        )

    # ---- teacher: the frozen dense model itself (Sec. IV-C self-teacher) ----
    teacher = None
    eta = 0.0 if args.no_teacher else args.eta
    if eta > 0.0:
        log("loading frozen bf16 teacher (dense self-teacher, Sec. IV-C)")
        teacher = LlamaForCausalLM.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        teacher.config.use_cache = False
        teacher.eval().to(device)
        for p in teacher.parameters():
            p.requires_grad_(False)

    if ddp:
        # The mask buffers total 6.5 GB. Broadcasting them every forward would be
        # pure waste AND wrong-headed: every rank recomputes the mask
        # deterministically from its own (all-reduce-synchronised) weights, so
        # they agree by construction. `check_mask_sync` verifies that empirically
        # instead of paying for a broadcast.
        ddp_kwargs = dict(device_ids=[local_rank], gradient_as_bucket_view=True)
        import inspect as _inspect

        if "forward_sync_buffers" in _inspect.signature(DDP.__init__).parameters:
            ddp_kwargs["forward_sync_buffers"] = False  # torch >= 2.13 name
        else:
            ddp_kwargs["broadcast_buffers"] = False
        student = DDP(model, **ddp_kwargs)
    else:
        student = model
    inner = model  # un-wrapped, for mask refresh / diagnostics

    @torch.no_grad()
    def check_mask_sync() -> None:
        """Assert every rank holds the identical mask.

        DDP all-reduces gradients, so weights stay bit-identical across ranks and
        the magnitude mask must too. If they ever diverge, different ranks would
        decay different weights and the run is silently corrupt -- so this is
        checked rather than assumed.
        """
        if not ddp:
            return
        h = torch.zeros(1, dtype=torch.float64, device=device)
        for _, mod in __import__("cast").cast_modules(inner):
            h += mod.mask.sum(dtype=torch.float64)
        mine = h.clone()
        dist.all_reduce(h, op=dist.ReduceOp.MIN)
        if not torch.equal(mine, h):
            raise RuntimeError(
                f"mask diverged across ranks: rank{rank} checksum {mine.item()} != min {h.item()}"
            )

    # ---- optimizer ----
    opt = AdamS(
        build_param_groups(inner, lr=args.lr),
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        total_steps=args.max_steps,
        l1_decay=args.l1_decay,
        expected_scope_elements=expected_elements,
        expected_scope_tensors=expected_tensors,
        require_fp32=True,
    )

    # ---- data ----
    data_dir = root / args.data
    dtype = args.data_dtype
    if dtype == "auto":
        meta_path = data_dir / "metadata.json"
        if meta_path.exists():
            import json as _json
            dtype = _json.loads(meta_path.read_text()).get("dtype", "uint16")
        else:
            dtype = "uint16"
        log(f"data-dtype auto-resolved to {dtype} (from {'metadata.json' if meta_path.exists() else 'fallback default'})")
    train = BinDataset(data_dir / "train.bin", args.seq_len, dtype, args.seed, rank, world)
    log(f"train tokens: {train.n:,} from {data_dir}")

    manifest = {
        "paper": "arXiv:2509.25996v1",
        "parallelism": "plain DDP (NOT FSDP -- see module docstring)",
        "world_size": world,
        "hyperparameters": vars(args),
        "cast_scope": stats,
        "tokens_per_step": tokens_per_step,
        "total_tokens": tokens_per_step * args.max_steps,
        "grad_accum": accum,
    }
    if is_master():
        (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # ---- train ----
    log("starting training")
    t0 = time.time()
    for step in range(args.max_steps):
        # Alg. 1 lines 6-8 / Alg. 2 lines 8-10: refresh the mask at the TOP of
        # step t, BEFORE gradients and BEFORE opt.step().  The old code refreshed
        # after opt.step(), so step 0 ran with an all-ones mask (audit S4.5).
        if step % args.mask_period == 0:
            _, flips = refresh_all_masks(inner)
            if step % (args.mask_period * 50) == 0:
                check_mask_sync()
        else:
            flips = None

        cur_lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        agg = {"loss": 0.0, "ce": 0.0, "kl": 0.0}
        for micro in range(accum):
            x, y = train.batch(args.micro_batch, device)
            sync = (micro == accum - 1)
            ctx = student.no_sync() if (ddp and not sync) else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = student(input_ids=x).logits
                    t_logits = None
                    if teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher(input_ids=x).logits
                    loss, comp = cast_loss(
                        out, t_logits, y, eta=eta, temperature=args.kl_temperature
                    )
                (loss / accum).backward()
            for k in agg:
                agg[k] += comp[k] / accum

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(inner.parameters(), args.grad_clip)
        opt.step()  # raises unless 100% of in-scope weights took the AdamS path

        if step % args.log_every == 0 or step == args.max_steps - 1:
            s = opt.last_stats
            el = time.time() - t0
            log(
                f"step {step}/{args.max_steps} loss={agg['loss']:.4f} ce={agg['ce']:.4f} "
                f"kl={agg['kl']:.4f} lr={cur_lr:.3e} alpha={s['alpha_t']:.4f} "
                f"aligned={s['cast_tensors_aligned']}/{s['cast_tensors']} "
                f"decayed={s['decayed_elements']:,} flips={flips} "
                f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}G {el:.0f}s"
            )

        if args.diag_every and step > 0 and step % args.diag_every == 0 and is_master():
            rep = magnitude_report(inner)
            log(f"DIAG step {step}: {json.dumps(rep['summary'])}")

        if args.save_every and step > 0 and step % args.save_every == 0 and is_master():
            torch.save(
                {"model": inner.state_dict(), "step": step, "args": vars(args)},
                outdir / f"step{step}_prefinal.pt",
            )

    # ---- Alg. 2 lines 19-22: finalise ----
    if is_master():
        pre = magnitude_report(inner)
        log(f"PRE-FINALIZE diagnostics: {json.dumps(pre['summary'])}")
        (outdir / "diag_prefinalize.json").write_text(json.dumps(pre, indent=2))
        torch.save({"model": inner.state_dict(), "step": args.max_steps}, outdir / "prefinal.pt")

    n = finalize_all(inner)
    log(f"finalized {n} modules (pruned with M_T, then folded the scaling module)")

    if is_master():
        viol = sum(m.exact_nm_violations() for _, m in __import__("cast").cast_modules(inner))
        log(f"exact 2:4 violations after finalize: {viol}")
        if viol:
            raise RuntimeError(f"{viol} groups are not exactly 2:4 after finalization")
        torch.save({"model": inner.state_dict(), "final": True}, outdir / "final_sparse.pt")
        log(f"wrote {outdir/'final_sparse.pt'}")

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
