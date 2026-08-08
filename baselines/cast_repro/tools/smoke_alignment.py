#!/usr/bin/env python3
"""Real-scale GPU smoke test for the AdamS alignment assertion. PURE TORCH.

This deliberately avoids `transformers`: the thing under test is the
weight<->mask correspondence and the AdamS scope accounting, which depend only
on the *shapes* of the 224 in-block projections.  So we build those 224 tensors
at exact LLaMA2-7B dimensions and run real AdamS steps on real GPU memory.

What it proves:
  * 224/224 in-scope tensors get an element-aligned mask (the thing that
    silently failed under FSDP);
  * decayed_elements == 3,238,002,688 exactly (= half of the 6,476,005,376
    in-block linear parameters, i.e. an exact 2:4 pattern);
  * AdamS runs at 7B scale on one card without OOM, with fp32 master weights;
  * under DDP the mask buffers stay aligned after wrapping.

Single GPU (scope + assertion):
    python smoke_alignment.py

8-GPU DDP (also checks buffer broadcast):
    torchrun --nproc_per_node 8 smoke_alignment.py --ddp --layers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cast import (  # noqa: E402
    LLAMA2_7B_CAST_ELEMENTS,
    LLAMA2_7B_CAST_TENSORS,
    LLAMA2_7B_DECAYED_ELEMENTS,
    AdamS,
    CastSparseLinear,
    build_param_groups,
    cast_scope_stats,
    magnitude_report,
    refresh_all_masks,
)

HIDDEN = 4096
INTERMEDIATE = 11008
N_LAYERS = 32


class FakeLlamaBlock(nn.Module):
    """The 7 in-block projections of a LLaMA2-7B decoder layer, exact shapes."""

    def __init__(self, scale_groups: int = 2, device=None):
        super().__init__()
        mk = lambda i, o: CastSparseLinear(  # noqa: E731
            i, o, bias=False, scale_groups=scale_groups, device=device, dtype=torch.float32
        )
        self.q_proj = mk(HIDDEN, HIDDEN)
        self.k_proj = mk(HIDDEN, HIDDEN)
        self.v_proj = mk(HIDDEN, HIDDEN)
        self.o_proj = mk(HIDDEN, HIDDEN)
        self.gate_proj = mk(HIDDEN, INTERMEDIATE)
        self.up_proj = mk(HIDDEN, INTERMEDIATE)
        self.down_proj = mk(INTERMEDIATE, HIDDEN)


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--l1-decay", type=float, default=4e-7)
    ap.add_argument("--total-steps", type=int, default=7500)
    ap.add_argument("--mask-period", type=int, default=10)
    ap.add_argument("--ddp", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.ddp and world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    def log(m):
        if rank == 0:
            print(f"[smoke] {m}", flush=True)

    full = args.layers == N_LAYERS
    log(f"building {args.layers} LLaMA2-7B-shaped blocks on {torch.cuda.get_device_name(device)}")
    t0 = time.time()
    model = nn.Sequential(*[FakeLlamaBlock(device=device) for _ in range(args.layers)])
    log(f"built in {time.time()-t0:.1f}s; alloc={torch.cuda.memory_allocated()/2**30:.1f} GiB")

    # Alg. 2 lines 1-4: mask initialised before the first step.
    n_mod, _ = refresh_all_masks(model)
    stats = cast_scope_stats(model)
    log(f"modules={n_mod}  scope={json.dumps(stats)}")

    expected_tensors = LLAMA2_7B_CAST_TENSORS if full else args.layers * 7
    expected_elements = LLAMA2_7B_CAST_ELEMENTS if full else None
    assert n_mod == expected_tensors, f"{n_mod} modules != expected {expected_tensors}"
    if full:
        assert stats["cast_elements"] == LLAMA2_7B_CAST_ELEMENTS, stats
        assert stats["cast_masked_elements"] == LLAMA2_7B_DECAYED_ELEMENTS, stats
        log(
            f"STATIC OK: 224 tensors, {LLAMA2_7B_CAST_ELEMENTS:,} elements, "
            f"{LLAMA2_7B_DECAYED_ELEMENTS:,} masked (exactly half)"
        )

    wrapped = model
    if args.ddp and world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        wrapped = DDP(model, device_ids=[local_rank])
        # DDP broadcasts buffers from rank 0; verify the mask is still the very
        # tensor the Parameter points at (this is what FSDP breaks).
        for _, mod in __import__("cast").cast_modules(model):
            mod.assert_mask_alignment()
        log("DDP wrap OK: every mask still element-aligned with its weight")

    opt = AdamS(
        build_param_groups(model, lr=args.lr),
        lr=args.lr,
        total_steps=args.total_steps,
        l1_decay=args.l1_decay,
        expected_scope_elements=expected_elements,
        expected_scope_tensors=expected_tensors,
        require_fp32=True,
    )

    log(f"running {args.steps} AdamS steps with synthetic gradients")
    results = []
    for step in range(args.steps):
        if step % args.mask_period == 0:
            refresh_all_masks(model)
        # Synthetic gradients: the assertion under test is about mask/weight
        # correspondence and scope accounting, not about the loss value.
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p).mul_(1e-3)
        opt.step()
        s = dict(opt.last_stats)
        results.append(s)
        log(
            f"step {step}: aligned={s['cast_tensors_aligned']}/{s['cast_tensors']} "
            f"coverage={s['coverage']:.0%} decayed={s['decayed_elements']:,} "
            f"alpha={s['alpha_t']:.6f} peak_mem={torch.cuda.max_memory_allocated()/2**30:.1f} GiB"
        )

    last = results[-1]
    assert last["cast_tensors_aligned"] == last["cast_tensors"], last
    assert last["coverage"] == 1.0, last
    if full:
        assert last["decayed_elements"] == LLAMA2_7B_DECAYED_ELEMENTS, (
            f"decayed_elements={last['decayed_elements']:,} != {LLAMA2_7B_DECAYED_ELEMENTS:,}"
        )

    rep = magnitude_report(model, max_modules=8)
    log(f"magnitude sample (8 modules): {json.dumps(rep['summary'])}")

    log("")
    log("=" * 62)
    log(f"PASS  alignment {last['cast_tensors_aligned']}/{last['cast_tensors']} (100%)")
    if full:
        log(f"PASS  decayed_elements = {last['decayed_elements']:,} (expected 3,238,002,688)")
    log(f"PASS  peak memory {torch.cuda.max_memory_allocated()/2**30:.1f} GiB / "
        f"{torch.cuda.get_device_properties(device).total_memory/2**30:.0f} GiB")
    log("=" * 62)

    if args.json_out and rank == 0:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(device),
                    "world_size": world,
                    "layers": args.layers,
                    "scope": stats,
                    "steps": results,
                    "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "magnitude_sample": rep,
                },
                indent=2,
            )
        )

    if args.ddp and world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
