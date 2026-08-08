#!/usr/bin/env python3
"""Decay-budget feasibility check for a CAST run.

Why this exists.  A finding from the unit tests (see
tests/test_cast.py::test_terminal_magnitude_is_set_by_final_lr): AdamS is
Adam-normalised, so in the decay-dominated regime the per-step displacement
saturates at ~lr, *independently of lambda*.  Consequences:

  1. The total distance a masked weight can travel toward zero is bounded by
     ``sum_t lr_t * alpha_t`` (alpha_t = t/T ramps the decay in).  If that
     budget is smaller than the typical |W| the weights CANNOT reach zero, no
     matter how correct the implementation is, and the final hard prune will
     destroy the model -- exactly the 23.45-PPL symptom of the failed run.
  2. The residual floor is O(final lr), so the schedule must decay.
  3. lambda controls how quickly decay dominates the gradient, i.e. *when* the
     ramp bites -- not the terminal magnitude.

Run this BEFORE launching a long run.

  python baselines/cast_repro/tools/decay_budget.py
  python baselines/cast_repro/tools/decay_budget.py --steps 7500 --lr 2e-5 --min-lr 2e-6
"""

from __future__ import annotations

import argparse
import math


def cosine_lr(step: int, total: int, lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, prog)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=7500, help="T (Table XI: 7.5k for LLaMA)")
    ap.add_argument("--lr", type=float, default=2e-5, help="peak LR (Table XI: 2e-5)")
    ap.add_argument("--min-lr", type=float, default=2e-6)
    ap.add_argument("--warmup", type=int, default=375)
    ap.add_argument(
        "--typical-magnitude",
        type=float,
        default=0.0067,
        help="mean |masked W| at init; 0.0067 measured on LLaMA2-7B (audit doc S5)",
    )
    ap.add_argument(
        "--saturation",
        type=float,
        default=1.0,
        help="per-step |dw|/lr in the decay-dominated regime; measured 1.0-1.7, 1.0 is conservative",
    )
    args = ap.parse_args()

    budget = 0.0
    plain = 0.0
    for t in range(args.steps):
        lr_t = cosine_lr(t, args.steps, args.lr, args.min_lr, args.warmup)
        alpha_t = t / args.steps
        budget += lr_t * alpha_t * args.saturation
        plain += lr_t
    headroom = budget / args.typical_magnitude
    floor = args.saturation * args.min_lr

    print(f"steps            = {args.steps}")
    print(f"lr               = {args.lr:g} -> {args.min_lr:g} (cosine, {args.warmup} warmup)")
    print(f"sum lr_t         = {plain:.4f}")
    print(f"alpha-weighted   = {budget:.4f}   <- usable decay distance")
    print(f"typical |W|      = {args.typical_magnitude:g}")
    print(f"HEADROOM         = {headroom:.2f}x")
    print(f"residual floor   ~ {floor:.2e}  (should be << 1e-4)")
    print()
    if headroom < 1.5:
        print("VERDICT: INSUFFICIENT. Masked weights cannot reach zero; the final prune will")
        print("         collapse the model. Raise lr, raise steps, or lower min_lr.")
        return 1
    if headroom < 3.0:
        print("VERDICT: MARGINAL. Expect a long tail of insufficiently decayed weights.")
        print("         Watch the masked/kept ratio and p99 |masked W| during training.")
        return 0
    print("VERDICT: OK. Budget is comfortable; monitor masked/kept ratio anyway.")
    if floor > 1e-4:
        print(f"WARNING: residual floor {floor:.2e} > 1e-4; lower --min-lr.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
