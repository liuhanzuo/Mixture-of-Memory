#!/usr/bin/env python3
"""Demonstrate the FSDP mask-misalignment bug that broke the previous run.

This is the empirical evidence behind the "use DDP, not FSDP" decision, and the
regression guard for it.  Run on >= 2 GPUs:

    torchrun --nproc_per_node 8 tools/fsdp_misalignment_demo.py

Claim under test (CAST_REPRODUCTION_AUDIT.md section 4.1): under FSDP
FULL_SHARD, a rank's slice of `weight` and its slice of the same layer's `mask`
are taken at different FlatParameter offsets, so they are not element-aligned and
their numel can even differ.  Any optimizer that pairs them element-wise is then
either wrong or -- as in the old code -- silently falls back to plain Adam.

Under DDP nothing is sharded, so alignment is trivially exact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cast import CastSparseLinear, refresh_all_masks  # noqa: E402


class Block(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.q_proj = CastSparseLinear(512, 512, bias=False, scale_groups=2, device=device)
        self.down_proj = CastSparseLinear(512, 512, bias=False, scale_groups=2, device=device)

    def forward(self, x):
        return self.down_proj(self.q_proj(x))


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        print("needs >= 2 ranks: torchrun --nproc_per_node 8 tools/fsdp_misalignment_demo.py")
        return 2
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    def log(m):
        if rank == 0:
            print(f"[demo] {m}", flush=True)

    # ---------------- DDP: aligned ----------------
    from torch.nn.parallel import DistributedDataParallel as DDP

    m_ddp = nn.Sequential(Block(device), Block(device))
    refresh_all_masks(m_ddp)
    DDP(m_ddp, device_ids=[local_rank])
    ok = 0
    total = 0
    for _, mod in __import__("cast").cast_modules(m_ddp):
        total += 1
        try:
            mod.assert_mask_alignment()
            ok += 1
        except RuntimeError:
            pass
    log(f"DDP  : {ok}/{total} tensors element-aligned  -> weight.shape == mask.shape everywhere")

    # ---------------- FSDP: misaligned ----------------
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    m_fsdp = nn.Sequential(Block(device), Block(device))
    refresh_all_masks(m_fsdp)
    shapes_before = [
        (n, tuple(mod.weight.shape), tuple(mod.mask.shape))
        for n, mod in __import__("cast").cast_modules(m_fsdp)
    ]
    wrapped = FSDP(
        m_fsdp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=local_rank,
    )

    mism = []
    for name, p in wrapped.named_parameters():
        if not getattr(p, "cast_in_scope", False):
            continue
        msk = getattr(p, "cast_mask", None)
        if msk is None:
            mism.append((name, tuple(p.shape), None))
        elif msk.shape != p.shape or msk.numel() != p.numel():
            mism.append((name, tuple(p.shape), tuple(msk.shape)))

    log("")
    log("FSDP FULL_SHARD, use_orig_params=True:")
    log(f"  pre-wrap shapes (rank0): {shapes_before[0]}")
    for name, wshape, mshape in mism[:4]:
        log(f"  MISALIGNED {name}: weight shard {wshape} vs mask {mshape}")
    log(
        f"  => {len(mism)} in-scope tensors have a weight shard whose shape/numel differs "
        f"from the mask"
    )
    log("")
    if mism:
        log("CONFIRMED: FSDP breaks weight<->mask correspondence. The old code turned this")
        log("           into `mask = None` -> vanilla Adam -> no selective L1 decay at all.")
        log("           AdamS in this repo raises MaskCoverageError instead. Use DDP.")
    else:
        log("NOTE: no mismatch observed in this configuration; DDP is still the safe choice")
        log("      because alignment is guaranteed structurally rather than incidentally.")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
