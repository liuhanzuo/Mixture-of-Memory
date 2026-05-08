#!/usr/bin/env python3
"""Post-hoc MemLong-protocol NIAH evaluation for H/H2 checkpoints.

Loads a saved checkpoint from scripts/train_cross_attn_memory.py and runs
evaluate_memlong_niah() on it — the same protocol MemLong/eval_niah.py uses,
so numbers are directly comparable across our experiments and the MemLong
baseline.

Usage (single GPU, single process):

    python scripts/eval_memlong_niah_on_checkpoints.py \\
        --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \\
        --checkpoint outputs/experiment_h_middle_layer/step_1000.pt \\
        --output_json outputs/experiment_h_middle_layer/memlong_niah_step1000.json \\
        --middle_layer_memory --memory_write_layer 16 --memory_read_layers 18,22,26,30 \\
        --slot_forward --memory_init strided --num_slots 64 --seq_len 4096 \\
        --lengths 2048,4096 --depths 0.0,0.25,0.5,0.75,1.0 --num_trials 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# Ensure our training script is importable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from scripts.train_cross_attn_memory import (  # noqa: E402
    CrossAttentionMemoryModel,
    evaluate_memlong_niah,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Required
    p.add_argument("--model", type=str, required=True, help="Path to base Llama weights")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to step_N.pt produced by training")
    p.add_argument("--output_json", type=str, required=True, help="Where to write NIAH results JSON")

    # Model architecture flags (must match the training config)
    p.add_argument("--num_slots", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--slot_forward", action="store_true", default=False)
    p.add_argument("--slot_isolated", action="store_true", default=False)
    p.add_argument("--memory_init", type=str, default="strided",
                   choices=["learnable", "mlp", "strided"])
    p.add_argument("--middle_layer_memory", action="store_true", default=False)
    p.add_argument("--memory_write_layer", type=int, default=16)
    p.add_argument("--memory_read_layers", type=str, default="18,22,26,30")
    p.add_argument("--cross_chunk_propagation", action="store_true", default=False)
    p.add_argument("--use_memory", action="store_true", default=True)
    p.add_argument("--use_cross_attn_memory", action="store_true", default=True)

    # NIAH eval grid
    p.add_argument("--lengths", type=str, default="2048,4096",
                   help="Comma-separated sequence lengths to evaluate.")
    p.add_argument("--depths", type=str, default="0.0,0.25,0.5,0.75,1.0",
                   help="Comma-separated needle depth ratios.")
    p.add_argument("--num_trials", type=int, default=3,
                   help="Trials per (length, depth) cell.")
    p.add_argument("--seed", type=int, default=42)

    # Runtime
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"[memlong-niah] loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[memlong-niah] loading base model from {args.model} (dtype={args.dtype})")
    base_model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )

    print("[memlong-niah] constructing CrossAttentionMemoryModel wrapper")
    cm_model = CrossAttentionMemoryModel(
        base_model=base_model,
        num_slots=args.num_slots,
        full_finetune=True,
        use_memory=args.use_memory,
        use_cross_attn_memory=args.use_cross_attn_memory,
        gradient_checkpointing=False,  # no training here
        slot_forward=args.slot_forward,
        slot_isolated=args.slot_isolated,
        memory_init=args.memory_init,
        cross_chunk_propagation=args.cross_chunk_propagation,
        middle_layer_memory=args.middle_layer_memory,
        memory_write_layer=args.memory_write_layer,
        memory_read_layers=args.memory_read_layers,
    ).to(device).to(dtype)
    cm_model.eval()

    print(f"[memlong-niah] loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = cm_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"[memlong-niah] WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[memlong-niah] WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    step = ckpt.get("global_step", -1)
    print(f"[memlong-niah] checkpoint loaded (step={step})")

    lengths = tuple(int(x) for x in args.lengths.split(","))
    depths = tuple(float(x) for x in args.depths.split(","))
    print(f"[memlong-niah] grid: lengths={lengths} depths={depths} trials={args.num_trials}")

    t0 = time.time()
    overall_acc, breakdown = evaluate_memlong_niah(
        cm_model, tokenizer, device,
        chunk_size=args.seq_len,
        lengths=lengths,
        depths=depths,
        num_trials=args.num_trials,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"MemLong-protocol NIAH: overall_acc = {overall_acc:.4f}  (elapsed {elapsed:.1f}s)")
    print(f"{'='*70}")
    for cell in breakdown:
        status = "OK" if cell["accuracy"] > 0.5 else ("PARTIAL" if cell["accuracy"] > 0 else "FAIL")
        print(
            f"  L={cell['length']:>6d} D={cell['depth']:.2f}: "
            f"acc={cell['accuracy']:.2f} ({cell['correct']}/{cell['total']}) "
            f"loss={cell['avg_loss']:.2f} [{status}]"
        )

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    result = {
        "checkpoint": args.checkpoint,
        "step": step,
        "model": args.model,
        "config": {
            "num_slots": args.num_slots,
            "seq_len": args.seq_len,
            "slot_forward": args.slot_forward,
            "slot_isolated": args.slot_isolated,
            "memory_init": args.memory_init,
            "middle_layer_memory": args.middle_layer_memory,
            "memory_write_layer": args.memory_write_layer,
            "memory_read_layers": args.memory_read_layers,
            "cross_chunk_propagation": args.cross_chunk_propagation,
        },
        "grid": {
            "lengths": list(lengths),
            "depths": list(depths),
            "num_trials": args.num_trials,
            "seed": args.seed,
        },
        "overall_accuracy": overall_acc,
        "breakdown": breakdown,
        "elapsed_s": elapsed,
    }
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[memlong-niah] results saved to {args.output_json}")


if __name__ == "__main__":
    main()
