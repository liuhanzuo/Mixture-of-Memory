#!/usr/bin/env python
"""Minimal cross-node NCCL allreduce probe. No model, no data, no training.

Goal: isolate whether the H800 2-node fabric can do a single cross-node
collective at all, and surface the exact NCCL failure/hang reason.

Run identically on both nodes (only NODE_RANK differs), e.g.:
  node0: NNODES=2 NODE_RANK=0 MASTER_ADDR=30.203.138.213 MASTER_PORT=29840 \
         .venv/bin/python -m torch.distributed.run --nnodes=2 --nproc_per_node=8 \
         --node_rank=0 --master_addr=30.203.138.213 --master_port=29840 \
         scripts/nccl_probe.py
  node1: same but --node_rank=1

It prints, per rank: init time, a tiny intra+inter allreduce result, and timing.
A hang will localize to the first cross-node allreduce; the watchdog/timeout
will then dump which ranks were waiting.
"""
import os
import time
import datetime

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])

    t0 = time.time()
    # Short timeout so a hang fails fast with a useful dump instead of hanging 2h.
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world,
        timeout=datetime.timedelta(seconds=120),
    )
    torch.cuda.set_device(local)
    t_init = time.time() - t0
    print(f"[rank {rank}/{world} local{local}] init OK in {t_init:.1f}s", flush=True)

    dev = torch.device(f"cuda:{local}")

    # 1) tiny allreduce (1 element) — first real collective across ALL ranks
    x = torch.tensor([float(rank)], device=dev)
    t1 = time.time()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dt1 = time.time() - t1
    expected = world * (world - 1) / 2.0
    print(f"[rank {rank}] tiny allreduce -> {x.item():.0f} (expect {expected:.0f}) "
          f"in {dt1*1000:.1f}ms", flush=True)

    # 2) bigger allreduce (64 MiB) — exercises real bandwidth / chunked transport
    n = 16 * 1024 * 1024  # 64 MiB fp32
    big = torch.ones(n, device=dev)
    dist.barrier()
    t2 = time.time()
    for _ in range(5):
        dist.all_reduce(big, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dt2 = (time.time() - t2) / 5
    # busbw approx: 2*(N-1)/N * size / time
    size_gb = n * 4 / 1e9
    busbw = 2 * (world - 1) / world * size_gb / dt2
    if rank == 0:
        print(f"[rank 0] 64MiB allreduce avg {dt2*1000:.1f}ms  ~{busbw:.1f} GB/s busbw",
              flush=True)

    dist.barrier()
    if rank == 0:
        print("[rank 0] ALL COLLECTIVES PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
