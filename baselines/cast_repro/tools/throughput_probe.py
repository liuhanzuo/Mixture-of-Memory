#!/usr/bin/env python3
"""Measure real per-step throughput for the CAST recipe. PURE TORCH.

Motivation: the paper's Appendix F quotes 403 s/step for LLaMA3-8B CAST on
32xH800, which is ~40x slower than a FLOPs estimate predicts, so it cannot be
used to project our wall time. This measures the dominant cost directly.

We time forward+backward+AdamS through the 32 LLaMA2-7B-shaped projection
stacks -- i.e. all 6.48B in-block linear parameters, which carry ~96% of the
model FLOPs. Excluded: attention score/AV matmuls (add ~10-15% at 4096 ctx),
embeddings, lm_head (0.5B params, mostly the logit projection), and the
bf16 teacher forward (+~1/3 of student forward FLOPs). The script applies a
correction factor for those and reports both raw and corrected numbers.

    CUDA_VISIBLE_DEVICES=0 python tools/throughput_probe.py --micro-batch 1 --seq-len 4096
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cast import AdamS, build_param_groups, refresh_all_masks  # noqa: E402
from tools.smoke_alignment import FakeLlamaBlock  # noqa: E402

LLAMA2_7B_PARAMS = 6_738_415_616
INBLOCK_PARAMS = 6_476_005_376


class TimedBlock(FakeLlamaBlock):
    """FakeLlamaBlock is shape-only (it exists for the alignment assertion).
    Add the linear-algebra forward so we can time it. Attention score/AV matmuls
    are intentionally omitted and accounted for by the correction factor."""

    def forward(self, x):
        h = self.o_proj(self.v_proj(x) + self.q_proj(x) + self.k_proj(x))
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(h)) * self.up_proj(h)
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--global-batch", type=int, default=256)
    ap.add_argument("--total-steps", type=int, default=7500)
    ap.add_argument("--world", type=int, default=8, help="cards to extrapolate to")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda", 0)
    name = torch.cuda.get_device_name(dev)
    print(f"device: {name}")

    blocks = torch.nn.Sequential(*[TimedBlock(device=dev) for _ in range(args.layers)])
    refresh_all_masks(blocks)
    opt = AdamS(
        build_param_groups(blocks, lr=2e-5), lr=2e-5, total_steps=args.total_steps,
        l1_decay=4e-7, require_fp32=True,
    )
    print(f"weights+state allocated: {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    x = torch.randn(args.micro_batch, args.seq_len, 4096, device=dev, dtype=torch.float32)

    times = []
    for i in range(args.warmup + args.iters):
        torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = blocks(x)
            loss = out.float().pow(2).mean()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        if i >= args.warmup:
            times.append(dt)
        print(f"  iter {i}{' (warmup)' if i < args.warmup else ''}: {dt:.3f}s")

    per_micro = sum(times) / len(times)
    tokens_per_micro = args.micro_batch * args.seq_len

    # FLOPs actually exercised: 6*N*D over the in-block linears.
    flops_measured = 6 * INBLOCK_PARAMS * tokens_per_micro
    tflops = flops_measured / per_micro / 1e12

    # Correction to a real CAST step:
    #   attention score/AV matmuls at 4096 ctx      ~ +12%
    #   embeddings + lm_head                        ~ +4%   (0.26B extra params)
    #   bf16 teacher forward (2*N*D vs 6*N*D)       ~ +33%
    correction = 1.12 * 1.04 * 1.33
    per_micro_real = per_micro * correction

    micros_per_step = args.global_batch // args.world
    sec_per_step = per_micro_real * micros_per_step
    total_h = sec_per_step * args.total_steps / 3600

    print()
    print(f"measured  : {per_micro:.3f}s per micro-batch of {tokens_per_micro:,} tokens")
    print(f"            {tflops:.1f} TFLOP/s achieved on 1 card (in-block linears only)")
    print(f"corrected : x{correction:.2f} for attention+head+teacher -> {per_micro_real:.3f}s")
    print()
    print(f"projection to {args.world} cards, global batch {args.global_batch}:")
    print(f"  {micros_per_step} micro-steps/card/step -> {sec_per_step:.0f}s per optimizer step")
    print(f"  {args.total_steps} steps -> {total_h:.0f} h = {total_h/24:.1f} days")
    print()
    print("NOTE: assumes perfect DDP scaling and no gradient-checkpointing cost;")
    print("      real wall time will be worse. This is a LOWER bound.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "device": name, "per_micro_s": per_micro, "tflops": tflops,
            "correction": correction, "sec_per_step": sec_per_step,
            "total_hours": total_h, "total_days": total_h / 24,
            "micro_batch": args.micro_batch, "seq_len": args.seq_len,
            "world": args.world, "global_batch": args.global_batch,
            "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        }, indent=2))
    print(f"peak memory: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
