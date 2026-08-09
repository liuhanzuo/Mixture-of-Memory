#!/usr/bin/env python3
"""CAST training for LLaMA-2-7B: plain DDP or ZeRO-1 sharding (arXiv:2509.25996v1).

Paper recipe for LLaMA (Table XI): lr 2e-5, lambda 4e-7, global batch 256,
seqlen 4096, 7500 steps, mask refresh every 10 steps, n=2 scaling groups,
eta=1/3 KL coefficient, Dolmino-Mix-1124.

THE ACTUAL HAZARD IS PARAMETER FLATTENING, NOT SHARDING
-------------------------------------------------------
The previous attempt used FSDP FULL_SHARD and it failed *silently*.  FSDP packs
`weight` and `mask` into a FlatParameter and slices them at *different* global
offsets, so a rank's weight shard and mask shard are not element-aligned (their
numel can even differ).  The old optimizer set `mask = None` in that case and
silently ran vanilla Adam, so the selective L1 decay never happened on most
tensors -- 7.86B tokens burned, Wiki PPL 23.45.  See
Mixture-of-Memory/SparseForge_Data/docs/CAST_REPRODUCTION_AUDIT.md section 4.1.
**FULL_SHARD remains forbidden.**

The refined diagnosis, established empirically by
``tools/fsdp_misalignment_demo.py`` on 8 ranks (torch 2.13.0, L20A): what breaks
CAST is *re-viewing / flattening the Parameter object*, not distributing work.
AdamS needs, at ``step()`` time, the original 2-D ``nn.Parameter`` still carrying
its original ``cast_mask``.  Measured verdicts:

  DDP                                     SAFE, 32/32 aligned, 1.000x state
  FSDP1 FULL_SHARD  use_orig_params=True  UNSAFE - param becomes (262144,) vs mask (512,512)
  FSDP1 SHARD_GRAD_OP use_orig_params=True UNSAFE - *identical* flattening; `use_orig_params`
                                          does NOT prevent it, and ZeRO-2-via-FSDP is
                                          therefore not available to us
  FSDP2 fully_shard (DTensor)             UNSAFE, AND WORSE: the `cast_in_scope` /
                                          `cast_mask` attributes are DROPPED (0/32 tagged),
                                          so AdamS sees no CAST params at all, reports
                                          `aligned=0/0`, and every assertion passes
                                          VACUOUSLY.  This is the same silent-failure
                                          class as the original bug.
  DDP + ZeroRedundancyOptimizer(AdamS)    SAFE, 32/32 aligned, 0.125x state at world=8

`--parallel zero2` therefore uses **DDP + ZeroRedundancyOptimizer**, i.e. ZeRO-1
(optimizer state sharded).  It is safe *by construction*, not incidentally: ZeRO
partitions at whole-tensor granularity (`_partition_parameters` greedily assigns
each entire Parameter to one rank), so a rank either owns a weight completely --
full 2-D shape, original `cast_mask`, alignment exactly as under DDP -- or does
not see it at all.  No tensor is ever split, so no offset mismatch can arise.
Naming note: the flag is `zero2` because grads are also reduced-and-freed by DDP
buckets; the optimizer-state sharding itself is ZeRO-1.

Because each rank now covers only its shard, `AdamS.last_stats` is per-rank and
`expected_scope_tensors`/`expected_scope_elements` (224 / 6.48e9, whole-model
constants) cannot be asserted locally.  They are asserted **globally via
all-reduce every step** instead -- see `assert_full_coverage()`.  Coverage is
never assumed; it is measured on every single step.

Memory per rank (LLaMA2-7B = 6.74e9 params, fp32 master + fp32 Adam state),
MEASURED on 8x L20A (178.35 GiB/card), seq_len 4096, micro_batch 1,
gradient checkpointing + expandable_segments, global_batch 256 (accum 32):

    ddp   : OOM.  Dies on step 1 trying to allocate 172 MiB with 99.75 MiB free;
            174.04 GiB allocated by PyTorch of a 178.35 GiB card.  Step 0
            completes (peak 138.6 GB) and then the optimizer state materialises.
    zero2 : COMPLETES.  Peak allocated 145.6 GB, reserved 147.9 GB, steady from
            step 1 onward -> ~32.7 GiB headroom on the card.
            Adam state on rank0: 7.0 GB over 29 tensors, vs 50.2 GB if
            unsharded => the realised saving is ~43 GB.

HONEST SCOPE OF THE SAVING -- the flag is called `zero2` but what is implemented
is ZeRO-**1**: only optimizer state is sharded.  **Gradients are NOT sharded**;
DDP keeps a full fp32 gradient buffer (~25 GB) on every rank.  Sharding grads too
would need either FSDP (flattens params -> forbidden, see above) or
`overlap_with_ddp=True` (requires a *functional* optimizer; AdamS is not one, and
that path is documented as experimental).  That is why the peak is 145.6 GB rather
than the ~103 GB a true ZeRO-2 would give.  32.7 GiB of headroom is enough for
this recipe but is NOT generous: raising seq_len, micro_batch, or dropping
gradient checkpointing can still OOM.  The next safe lever is CPU-offloading the
Adam state (still whole-tensor, so still alignment-preserving) -- NOT quantising
the master weights.

fp32 IS NON-NEGOTIABLE.  lambda=4e-7 gives a per-step decay ~8e-12, far below
bf16 resolution; bf16 master weights round the entire selective-decay signal to
zero and silently disable CAST (tests/test_cast.py::
test_bf16_swallows_lambda_fp32_does_not).  Never quantise the master weights to
save memory -- shard them or offload them instead.

CHECKPOINT / RESUME -- WHAT "FAITHFUL" MEANS HERE
-------------------------------------------------
A sibling project lost weeks of compute to an *unfaithful* resume: three arms
were restarted from checkpoints, the optimizer param-groups did not match, torch
silently re-initialised the Adam moments, and all three arms became
WARM-RESTARTS rather than continuations.  Nothing crashed; the numbers were just
un-analysable.  Everything below exists so that cannot happen here.

A resumable checkpoint (``ckpt_step<N>.pt``, written every ``--save-every``)
carries the FULL training state:

  model        fp32 master weights + ``cast_scale`` + the 224 ``mask`` buffers
               (``mask`` is a *persistent* buffer, so ``state_dict()`` already
               contains it -- we assert the count on load)
  optim        Adam moments and per-parameter ``step``.  Under zero2 the state
               lives sharded on 8 ranks, so it is gathered with
               ``consolidate_state_dict(to=0)`` and saved in ZeRO's *global*
               (world-size-independent) indexing; ``load_state_dict`` routes each
               entry back to the rank that owns that parameter.
  rng          per-rank torch CPU + torch CUDA + the numpy Generator that drives
               ``BinDataset``.  The data order is the single largest determinism
               lever: without the numpy state a resume reads different tokens and
               the loss trajectory diverges on the very first step.
  args         every trajectory-affecting hyperparameter, re-checked on load.

WHY THE MASK IS SAVED RATHER THAN RECOMPUTED.  The mask is refreshed only every
T1=10 steps, so it is *not* a function of the current weights: at step 503 the
live mask was computed from the weights as they stood at step 500.  Recomputing
it at resume time from the *post-step-500* weights gives a slightly DIFFERENT
mask (a handful of near-threshold entries flip), which perturbs which weights get
decayed.  We therefore persist the mask and restore it verbatim; the semantics
are "continue with exactly the mask that was live", not "close enough".

FAIL LOUD, NEVER SILENTLY.  ``check_resume_args`` raises ``ResumeMismatchError``
if any of ``RESUME_CRITICAL_ARGS`` differs from the checkpoint (a different
``--max-steps`` changes the alpha_t = t/T decay schedule *and* the cosine
horizon; a different ``--l1-decay`` changes the mechanism outright), and
``assert_optimizer_state_restored`` raises if any rank came back without Adam
moments for one of its owned parameters, or with the wrong per-parameter step
counter.  That second guard is the direct antidote to the silent-warm-restart
failure: a re-initialised moment buffer is a crash here, not a footnote.

NCCL TIMEOUT.  An ~87 GB checkpoint at the measured 262 MB/s on this filesystem
takes ~5.6 min, during which the non-zero ranks sit in a barrier.  That is
uncomfortably close to NCCL's 10-minute default watchdog, so the process group
is created with ``--dist-timeout`` (default 3600 s).  Do not lower it.

Smoke test (no long run):
    torchrun --nproc_per_node 8 train_cast_llama.py --max-steps 4 --smoke
    torchrun --nproc_per_node 8 train_cast_llama.py --max-steps 4 --parallel zero2

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
from cast.adams import MaskCoverageError  # noqa: E402
from cast.checkpoint import (  # noqa: E402
    ResumeMismatchError,
    assert_optimizer_state_restored,
    checkpoint_size_bytes,
    find_latest_checkpoint,
    load_training_state,
    prune_old_checkpoints,
    save_training_state,
    summarize_for_log,
)


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
    p.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"],
                   help="[impl] BOTH options are implementation_choice: the paper "
                        "specifies no within-run schedule (grep -ci cosine over the "
                        "fulltext = 0). 'constant' holds peak 2e-5 -- a defensible "
                        "literal reading of a silent paper, but NOT paper_explicit. "
                        "NOTE the 'consistent learning rate' sentence sometimes cited "
                        "for this is fulltext:1646, in Appendix D on the SCALING-LAW "
                        "sweep (hold LR fixed across token-budget points), not the "
                        "within-run schedule. 'cosine' is what SPEC.md S6 prescribes, "
                        "because the residual floor is O(final lr): constant parks "
                        "masked weights at ~2e-5 instead of ~2e-6, giving up ~10x of "
                        "terminal-magnitude margin before the final hard prune.")
    p.add_argument("--min-lr", type=float, default=2e-6,
                   help="[impl] cosine floor (AST alpha_f=0.1); ignored when "
                        "--lr-schedule constant")
    p.add_argument("--warmup", type=int, default=375,
                   help="[impl] 5%% of 7500; ignored when --lr-schedule constant")
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999), help="[impl] Adam default")
    p.add_argument("--eps", type=float, default=1e-8, help="[impl] Adam default")
    p.add_argument("--grad-clip", type=float, default=1.0, help="[impl] not in paper")
    p.add_argument("--micro-batch", type=int, default=1, help="[impl] memory-driven")
    p.add_argument("--parallel", default="ddp", choices=["ddp", "zero2"],
                   help="[impl] ddp = no sharding; MEASURED to OOM on 8x L20A at seq_len 4096 "
                        "(174.04/178.35 GiB). zero2 = DDP + ZeroRedundancyOptimizer(AdamS): "
                        "shards Adam state at WHOLE-TENSOR granularity, so weight<->mask "
                        "alignment is preserved by construction; measured peak 145.6 GB. "
                        "NEVER use FSDP -- it flattens the Parameter and breaks alignment; "
                        "see the module docstring and tools/fsdp_misalignment_demo.py.")
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
    p.add_argument("--save-every", type=int, default=1000,
                   help="write a FULLY RESUMABLE ckpt_step<N>.pt every N steps (model + "
                        "sharded Adam state + masks + per-rank RNG + args). ~87 GB each for "
                        "LLaMA2-7B, so see --keep-last. 0 disables.")
    p.add_argument("--keep-last", type=int, default=2,
                   help="how many ckpt_step*.pt to retain; older ones are deleted after a "
                        "successful newer write. 87 GB each => 2 is ~174 GB.")
    p.add_argument("--resume", default="",
                   help="path to a ckpt_step*.pt, or 'auto' to pick the newest one in --out. "
                        "Refuses (ResumeMismatchError) if any trajectory-affecting arg differs "
                        "from the checkpoint, or if the Adam moments did not survive the round "
                        "trip -- a silent warm restart is not an option here.")
    p.add_argument("--dist-timeout", type=int, default=3600,
                   help="NCCL/process-group timeout in seconds. MUST exceed the checkpoint "
                        "write time: ~87 GB at the measured 262 MB/s on this filesystem is "
                        "~5.6 min, uncomfortably close to the 10 min default, and non-zero "
                        "ranks sit in a barrier for the whole write.")
    p.add_argument("--stop-after", type=int, default=0,
                   help="stop cleanly after this step WITHOUT finalising, leaving the run "
                        "resumable. Distinct from --max-steps, which is the declared training "
                        "horizon and feeds alpha_t = t/T -- lowering --max-steps to stop early "
                        "would silently rescale the whole decay schedule. 0 = run to the end.")
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
    """LR at ``step``.

    ``constant`` (default) is PAPER-LITERAL: Appendix B specifies a "consistent
    learning rate for each model", and Table XI gives the single value 2e-5 for
    LLaMA -- so no warmup and no decay.  This is what a reproduction must use.

    ``cosine`` is a documented DEVIATION (audit S4.4): linear warmup then cosine
    to ``--min-lr``, mirroring AST official's ``alpha_f=0.1``.  Kept available
    because the decay budget is larger under a constant LR (tools/decay_budget.py
    with --min-lr == --lr), but it must be selected explicitly, never silently.
    """
    if args.lr_schedule == "constant":
        return args.lr
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
        # A checkpoint write parks the non-zero ranks in a barrier for minutes
        # (~5.6 min for 87 GB at 262 MB/s here), which the 10-minute NCCL default
        # would eventually turn into a watchdog abort. Raise it explicitly.
        import datetime as _dt

        dist.init_process_group("nccl", timeout=_dt.timedelta(seconds=args.dist_timeout))
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
    # Under zero2 each rank's AdamS only ever sees ITS SHARD of the tensors, so
    # the whole-model constants (224 tensors / 6.48e9 elements) are not locally
    # satisfiable -- they are asserted globally instead, every step, by
    # assert_full_coverage() below.  Passing them to the local AdamS would make
    # it crash on a correct run.
    adams_kwargs = dict(
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        total_steps=args.max_steps,
        l1_decay=args.l1_decay,
        require_fp32=True,
    )
    if args.parallel == "zero2":
        if not ddp:
            raise ValueError("--parallel zero2 requires world_size > 1 (launch with torchrun)")
        from torch.distributed.optim import ZeroRedundancyOptimizer

        # Whole-tensor partitioning => the Parameter a rank owns is the original
        # full 2-D nn.Parameter with its original `cast_mask`.  Verified on 8
        # ranks by tools/fsdp_misalignment_demo.py (32/32 aligned, 0.125x state).
        opt = ZeroRedundancyOptimizer(
            build_param_groups(inner, lr=args.lr),
            optimizer_class=AdamS,
            expected_scope_elements=None,
            expected_scope_tensors=None,
            **adams_kwargs,
        )
        log("optimizer: DDP + ZeroRedundancyOptimizer(AdamS) -- Adam state sharded at "
            "whole-tensor granularity (ZeRO-1); mask alignment preserved by construction")
    else:
        opt = AdamS(
            build_param_groups(inner, lr=args.lr),
            expected_scope_elements=expected_elements,
            expected_scope_tensors=expected_tensors,
            **adams_kwargs,
        )
        log("optimizer: plain AdamS (no sharding), ~131.8 GB/rank static")

    def local_stats() -> dict:
        """AdamS.last_stats for this rank (unwrapping ZeRO's local optimizer)."""
        return getattr(getattr(opt, "optim", opt), "last_stats", {}) or {}

    @torch.no_grad()
    def assert_full_coverage(step: int) -> dict:
        """EVERY step: prove 100% of the CAST scope took the AdamS decay path.

        Under zero2 no single rank can see the whole scope, so the per-rank
        assertions inside AdamS are necessary but NOT sufficient.  We combine the
        per-rank counters and check the totals against the static whole-model
        constants.  This is the guard the previous run lacked: it turns "the
        optimizer thinks it is fine locally" into "the union of all ranks covered
        exactly 224 tensors / 6.48e9 elements, half of them decayed".  A rank that
        silently fell back to vanilla Adam, or lost its cast_mask attribute, shows
        up as a deficit here.  Fault-injection-verified: untagging one rank's
        shard yields "GLOBAL tensors 14 != expected 16" on all ranks.

        The combining rule DIFFERS BY MODE and getting it wrong makes the check
        meaningless:
          zero2 -- the scope is PARTITIONED, so ranks are disjoint => SUM.
          ddp   -- every rank redundantly owns the WHOLE scope => the totals are
                   already per-model; summing would give 8*224=1792.  We instead
                   assert every rank agrees (MIN == MAX == local), which also
                   catches a single rank losing coverage.
        """
        s = local_stats()
        if not s:
            raise MaskCoverageError(
                f"step {step}: AdamS.last_stats empty -- step() did not run the CAST path"
            )
        if s["cast_tensors"] == 0:
            # Defence against the FSDP2 vacuous-pass class: if `cast_scope` were
            # lost from the param groups (e.g. dropped by a state_dict round trip)
            # every check below would pass on all-zero counters while the L1 decay
            # silently never ran. Under zero2 a rank could legitimately own zero
            # in-scope tensors only if the greedy partition gave it none of 224 --
            # impossible at world<=224, so this is always a bug.
            raise MaskCoverageError(
                f"step {step}: rank{rank} sees ZERO in-scope tensors. The `cast_scope` flag "
                "was lost from the optimizer param groups, so AdamS is running vanilla Adam "
                "and the selective L1 decay is not happening at all."
            )
        if s["cast_tensors"] != s["cast_tensors_aligned"]:
            raise MaskCoverageError(
                f"step {step}: rank{rank} local coverage "
                f"{s['cast_tensors_aligned']}/{s['cast_tensors']}"
            )
        keys = ("cast_tensors", "cast_tensors_aligned", "cast_elements", "decayed_elements")
        if not ddp:
            g = {k: s[k] for k in keys}
        elif args.parallel == "zero2":
            # Disjoint shards -> the union is the sum.
            t = torch.tensor([s[k] for k in keys], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            g = dict(zip(keys, (int(v) for v in t.tolist())))
        else:
            # Replicated scope -> totals must be IDENTICAL on every rank.
            mine = torch.tensor([s[k] for k in keys], dtype=torch.float64, device=device)
            lo, hi = mine.clone(), mine.clone()
            dist.all_reduce(lo, op=dist.ReduceOp.MIN)
            dist.all_reduce(hi, op=dist.ReduceOp.MAX)
            if not (torch.equal(lo, mine) and torch.equal(hi, mine)):
                raise MaskCoverageError(
                    f"step {step}: ranks disagree on CAST coverage under ddp: "
                    f"rank{rank}={[int(v) for v in mine.tolist()]} "
                    f"min={[int(v) for v in lo.tolist()]} max={[int(v) for v in hi.tolist()]}"
                )
            g = {k: s[k] for k in keys}
        if g["cast_tensors"] != g["cast_tensors_aligned"]:
            raise MaskCoverageError(
                f"step {step}: GLOBAL coverage {g['cast_tensors_aligned']}/{g['cast_tensors']} "
                "-- some rank ran without an aligned mask"
            )
        exp_t = expected_tensors
        if exp_t is not None and g["cast_tensors"] != exp_t:
            raise MaskCoverageError(
                f"step {step}: GLOBAL in-scope tensors {g['cast_tensors']} != expected {exp_t}. "
                "Under zero2 this means a rank's shard was dropped entirely."
            )
        if expected_elements is not None and g["cast_elements"] != expected_elements:
            raise MaskCoverageError(
                f"step {step}: GLOBAL in-scope elements {g['cast_elements']:,} != expected "
                f"{expected_elements:,}"
            )
        # Exact 2:4 => exactly half the scope is masked, hence decayed.
        if g["cast_elements"] and g["decayed_elements"] != g["cast_elements"] // 2:
            raise MaskCoverageError(
                f"step {step}: GLOBAL decayed {g['decayed_elements']:,} != half of scope "
                f"{g['cast_elements'] // 2:,} -- mask is not a valid 2:4 pattern"
            )
        return g

    # ---- data ----
    data_dir = root / args.data
    dtype = args.data_dtype
    if dtype == "auto":
        # NEVER guess the token width. Reading uint32 data as uint16 reinterprets
        # every token as two, silently doubling the stream and injecting zeros --
        # no exception, no warning, just a corrupt corpus. The previous code
        # defaulted to uint16 when metadata.json was absent, and that default
        # fired for real during development (a mistyped --data resolved to a
        # non-existent dir and still "auto-resolved to uint16"). So: no metadata,
        # no run.
        meta_path = data_dir / "metadata.json"
        if not meta_path.exists():
            raise RuntimeError(
                f"--data-dtype auto but {meta_path} does not exist. Refusing to guess the "
                "token width: reading uint32 tokens as uint16 silently doubles the token "
                "stream and injects zeros, with no error anywhere. Either point --data at a "
                "directory with metadata.json, or pass --data-dtype uint16/uint32 explicitly "
                "if you have verified the width yourself."
            )
        meta = json.loads(meta_path.read_text())
        if "dtype" not in meta:
            raise RuntimeError(f"{meta_path} has no 'dtype' key; refusing to guess")
        dtype = meta["dtype"]
        # Cross-check the byte size against the recorded token count: this catches
        # both a truncated train.bin and a metadata/dtype mismatch.
        nbytes = (data_dir / "train.bin").stat().st_size
        width = np.dtype(dtype).itemsize
        if "total_tokens" in meta and nbytes // width != meta["total_tokens"]:
            raise RuntimeError(
                f"{data_dir}/train.bin is {nbytes:,} bytes = {nbytes // width:,} tokens at "
                f"{width} B/token, but metadata.json says total_tokens={meta['total_tokens']:,}. "
                "Either the file is truncated or the dtype is wrong -- both would silently "
                "corrupt training."
            )
        log(
            f"data-dtype resolved to {dtype} ({width} B/token) from metadata.json; "
            f"dataset={meta.get('dataset')} tokenizer={meta.get('tokenizer')} "
            f"total_tokens={meta.get('total_tokens'):,} (byte size agrees)"
        )
    train = BinDataset(data_dir / "train.bin", args.seq_len, dtype, args.seed, rank, world)
    log(f"train tokens: {train.n:,} from {data_dir}")

    # ---- resume (must come AFTER the optimizer and the dataset exist) -------
    # Ordering is load-bearing: the optimizer must be constructed so ZeRO has
    # already computed its partition (`load_state_dict` routes state by
    # `_param_to_rank`), and the dataset must exist so its numpy Generator is
    # available to restore the token-stream position into.
    is_zero = args.parallel == "zero2"
    start_step = 0
    if args.resume:
        ck = find_latest_checkpoint(outdir) if args.resume == "auto" else Path(args.resume)
        if ck is None:
            raise ResumeMismatchError(
                f"--resume auto found no ckpt_step*.pt in {outdir}. Refusing to silently "
                "start from scratch: if a fresh run is what you want, drop --resume."
            )
        if not ck.exists():
            raise ResumeMismatchError(f"--resume {ck} does not exist")
        log(f"resuming from {ck}")
        meta = load_training_state(
            ck,
            model=inner,
            opt=opt,
            cur_args=vars(args),
            np_generator=train.rng,
            is_zero=is_zero,
            rank=rank,
            world=world,
            device=device,
            expected_mask_buffers=(LLAMA2_7B_CAST_TENSORS if not args.smoke else None),
        )
        # The model was loaded from a CPU blob; re-point every mask tag at the
        # (now restored) buffer and re-assert alignment. `load_state_dict` copies
        # into the existing buffers in place, so identity is preserved -- but
        # asserting is free and this is exactly the invariant whose silent loss
        # caused the original 7.86B-token failure.
        for _, mod in __import__("cast").cast_modules(inner):
            mod.weight.cast_mask = mod.mask
            mod.assert_mask_alignment()
        counts = assert_optimizer_state_restored(
            opt,
            expected_step=meta["step"] + 1,  # state['step'] is 1-based post-increment
            is_zero=is_zero,
            rank=rank,
            world=world,
            device=device,
        )
        start_step = meta["step"] + 1
        log(summarize_for_log(meta, counts))
        if start_step >= args.max_steps:
            log(f"checkpoint is already at/after --max-steps {args.max_steps}; nothing to do")
    elif find_latest_checkpoint(outdir) is not None:
        # Do not let a relaunch silently overwrite an in-progress run's history.
        log(
            f"WARNING: {outdir} already contains ckpt_step*.pt but --resume was NOT passed; "
            "this run starts from step 0 and will overwrite those checkpoints. Pass "
            "--resume auto to continue instead."
        )

    manifest = {
        "paper": "arXiv:2509.25996v1",
        "parallelism": (
            "DDP + ZeroRedundancyOptimizer(AdamS): ZeRO-1, Adam state sharded at "
            "whole-tensor granularity. NOT FSDP -- FSDP flattens the Parameter and "
            "breaks weight<->mask alignment; see module docstring."
            if args.parallel == "zero2"
            else "plain DDP, no sharding (NOT FSDP -- see module docstring)"
        ),
        "lr_schedule": (
            # NOT paper-literal, and NOT Appendix B. The paper never specifies a
            # within-run schedule: `grep -ci cosine` and `grep -ci 'warm.?up'`
            # over the full text are both 0. The one "consistent learning rate"
            # sentence is at line 1646 of the fulltext, inside Appendix D
            # ("Details on Scaling Law Experiments"), and reads "...we maintain a
            # consistent learning rate for each model AND ADJUST THE DECAY FACTOR
            # BASED ON THE TRAINING TOKEN BUDGET" -- i.e. hold LR fixed ACROSS the
            # sweep's token-budget points, not within a run. Both options here are
            # therefore implementation_choice. See SPEC.md S6 for the cost:
            # constant gives up ~10x of terminal-magnitude margin.
            f"constant {args.lr:g} (implementation_choice; the paper specifies no "
            "within-run schedule, so this is a literal-but-OUR reading, NOT "
            "paper_explicit. min_lr/warmup are inert here.)"
            if args.lr_schedule == "constant"
            else f"cosine {args.lr:g}->{args.min_lr:g} warmup {args.warmup} "
                 "(implementation_choice; matches SPEC.md S6, which argues the LR "
                 "must decay so masked weights are not parked at O(lr))"
        ),
        "world_size": world,
        "hyperparameters": vars(args),
        "cast_scope": stats,
        "tokens_per_step": tokens_per_step,
        "total_tokens": tokens_per_step * args.max_steps,
        "grad_accum": accum,
        "resumed_from": str(args.resume) if args.resume else None,
        "start_step": start_step,
        "data_dtype_resolved": dtype,
        "train_tokens": int(train.n),
    }
    if is_master():
        # Never clobber the original manifest on a resume -- it is the provenance
        # record of how the run began.
        name = "run_manifest.json" if start_step == 0 else f"run_manifest_resume{start_step}.json"
        (outdir / name).write_text(json.dumps(manifest, indent=2, default=str))

    # ---- train ----
    log(f"starting training at step {start_step}/{args.max_steps}")
    t0 = time.time()
    last_step = start_step - 1
    for step in range(start_step, args.max_steps):
        last_step = step
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
        # EVERY step, not just step 0: under zero2 the per-rank assertions inside
        # AdamS cannot see the whole scope, so the global totals are the only
        # sufficient check. Cost is one 4-element all-reduce.
        gcov = assert_full_coverage(step)

        if step == start_step:
            # Measure, don't assume, the thing this whole mechanism exists for.
            _o = getattr(opt, "optim", opt)
            _st = sum(
                t.numel() * t.element_size()
                for s_ in _o.state.values()
                for t in (s_.get("exp_avg"), s_.get("exp_avg_sq"))
                if torch.is_tensor(t)
            )
            _pn = sum(p.numel() for p in inner.parameters() if p.requires_grad)
            log(
                f"MEM step0 rank{rank}: adam_state={_st / 2**30:.1f}G over "
                f"{len(_o.state)} tensors (full-model fp32 Adam state would be "
                f"{2 * 4 * _pn / 2**30:.1f}G) peak={torch.cuda.max_memory_allocated() / 2**30:.1f}G "
                f"reserved={torch.cuda.max_memory_reserved() / 2**30:.1f}G"
            )

        if step % args.log_every == 0 or step == args.max_steps - 1:
            s = local_stats()
            el = time.time() - t0
            log(
                f"step {step}/{args.max_steps} loss={agg['loss']:.4f} ce={agg['ce']:.4f} "
                f"kl={agg['kl']:.4f} lr={cur_lr:.3e} alpha={s['alpha_t']:.4f} "
                f"aligned={gcov['cast_tensors_aligned']}/{gcov['cast_tensors']}(global) "
                f"decayed={gcov['decayed_elements']:,} flips={flips} "
                f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}G {el:.0f}s"
            )

        # Per-step loss trace at FULL precision, appended by rank 0. This is what
        # makes the resume faithfulness claim checkable rather than assertable: a
        # resumed run's trace can be diffed against an uninterrupted one
        # step-by-step (tools/resume_faithfulness.py does exactly that). The
        # log line above rounds to 4 decimals, which would hide a 1e-6 drift.
        if is_master():
            with (outdir / "loss_trace.jsonl").open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(agg["loss"]),
                            "ce": float(agg["ce"]),
                            "kl": float(agg["kl"]),
                            "lr": cur_lr,
                            "alpha": local_stats().get("alpha_t"),
                            "decayed": gcov["decayed_elements"],
                            "aligned": gcov["cast_tensors_aligned"],
                        }
                    )
                    + "\n"
                )

        if args.diag_every and step > 0 and step % args.diag_every == 0 and is_master():
            # Pass alpha_t so the verdict is judged against the ramp position.
            # Sec. IV-A / Appendix C targets are END-state; without alpha_t this
            # printed "BAD: ... AdamS is probably not running" on every DIAG of a
            # healthy run until ~step 5700 of 7500.
            rep = magnitude_report(inner, alpha_t=local_stats().get("alpha_t"))
            log(f"DIAG step {step}: {json.dumps(rep['summary'])}")

        # FULLY RESUMABLE checkpoint. Collective (consolidate_state_dict +
        # all_gather_object + barrier), so it must be reached by EVERY rank --
        # never guard this with is_master().
        if args.save_every and step > 0 and step % args.save_every == 0:
            t_save = time.time()
            written = save_training_state(
                outdir / f"ckpt_step{step}",
                step=step,
                model=inner,
                opt=opt,
                args=vars(args),
                np_generator=train.rng,
                is_zero=is_zero,
                rank=rank,
                world=world,
            )
            if is_master():
                gone = prune_old_checkpoints(outdir, args.keep_last)
                sz = checkpoint_size_bytes(written) / 2**30 if written else 0.0
                log(
                    f"saved {written.name}/ ({sz:.1f} GiB, {time.time() - t_save:.0f}s) "
                    f"resumable: model+optim shards+masks+rng; pruned "
                    f"{[p.name for p in gone]}"
                )

        if args.stop_after and step + 1 >= args.stop_after:
            break

    # ---- early stop (interrupted, NOT finished) ----
    # Finalisation is irreversible (Alg. 2 line 20 hard-prunes with M_T), so a run
    # that was merely interrupted must NOT finalise -- otherwise "stop at 4000 and
    # resume" would silently produce a pruned model at 4000 and continue training
    # a already-sparsified network. Save a resumable ckpt and exit instead.
    if args.stop_after and last_step + 1 >= args.stop_after:
        if args.save_every:
            save_training_state(
                outdir / f"ckpt_step{last_step}",
                step=last_step,
                model=inner,
                opt=opt,
                args=vars(args),
                np_generator=train.rng,
                is_zero=is_zero,
                rank=rank,
                world=world,
            )
            log(
                f"--stop-after {args.stop_after}: stopped at step {last_step} of "
                f"{args.max_steps} WITHOUT finalising; wrote ckpt_step{last_step}/. "
                f"Resume with --resume {outdir / f'ckpt_step{last_step}'}"
            )
        else:
            log(
                f"--stop-after {args.stop_after}: stopped at step {last_step} of "
                f"{args.max_steps} WITHOUT finalising. NOTE --save-every 0, so NOTHING "
                "resumable was written and this run cannot be continued."
            )
        if ddp:
            dist.barrier()
            dist.destroy_process_group()
        return

    # ---- Alg. 2 lines 19-22: finalise ----
    if is_master():
        # Ramp is consumed here (alpha_t -> 1), so this is the one place the
        # END-state acceptance targets legitimately apply. Pass alpha_t anyway
        # rather than relying on the None fallback, so the verdict records it.
        pre = magnitude_report(inner, alpha_t=local_stats().get("alpha_t"))
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
