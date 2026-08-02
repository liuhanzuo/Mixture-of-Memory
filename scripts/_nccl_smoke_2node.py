#!/usr/bin/env python3
"""Minimal 2-node NCCL all-reduce smoke. Confirms inter-node NCCL bring-up
before launching the real DDP training. Each rank contributes rank+1; the
all-reduce sum over W ranks must equal W*(W+1)/2."""
import os
import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    t = torch.full((1024, 1024), float(rank + 1), device=f"cuda:{local_rank}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = world * (world + 1) / 2.0
    got = t[0, 0].item()
    ok = abs(got - expected) < 1e-3
    print(f"[rank {rank}/{world} local{local_rank}] all_reduce got={got} "
          f"expected={expected} OK={ok}", flush=True)
    dist.barrier()
    if rank == 0:
        print(f"[SMOKE] world_size={world} all_reduce {'PASS' if ok else 'FAIL'}",
              flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
