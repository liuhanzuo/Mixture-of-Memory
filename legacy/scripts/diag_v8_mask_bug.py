#!/usr/bin/env env python3
"""
Diagnostic: confirm the inverted attention mask bug in v7/v8 training.

The bug: train_rmt_v7.py converts the bool attention mask incorrectly.
  build_rmt_attention_mask returns True = can attend.
  Training code: masked_fill(mask, -inf)  -> blocks True positions (WRONG)
  Eval code (rmt_module.py): masked_fill(~mask, -inf) -> blocks False positions (CORRECT)

This means during training, memory tokens and causal positions were completely
blocked from attending to the content they should have attended to.
"""

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from src.memory.rmt.rmt_module import build_rmt_attention_mask

print("=" * 60)
print("DIAGNOSTIC: Attention Mask Inversion Bug in v7/v8 Training")
print("=" * 60)

seq_len = 4
num_mem = 2
mask = build_rmt_attention_mask(seq_len, num_mem, torch.device("cpu"))
print(f"\nBool mask from build_rmt_attention_mask (True=can attend):")
print(mask.int())

# Training code (WRONG):
train_mask = torch.zeros_like(mask, dtype=torch.float32)
train_mask_wrong = train_mask.masked_fill(mask, float('-inf'))
print(f"\nTraining conversion (WRONG — masked_fill(mask, -inf)):")
print(train_mask_wrong)

# Eval code (CORRECT):
eval_mask = torch.zeros_like(mask, dtype=torch.float32)
eval_mask_correct = eval_mask.masked_fill(~mask, float('-inf'))
print(f"\nEval conversion (CORRECT — masked_fill(~mask, -inf)):")
print(eval_mask_correct)

# Show the difference
print(f"\nDifference (training blocks positions that should be visible):")
print((train_mask_wrong != eval_mask_correct).int())
num_wrong = (train_mask_wrong != eval_mask_correct).sum().item()
print(f"Positions corrupted: {num_wrong} / {mask.numel()}")

# Check v8 weights for signs of no training
print("\n" + "=" * 60)
print("v8 Memory Weights Analysis")
print("=" * 60)
ckpt = torch.load(
    "outputs/rmt_v8_8gpu_20260418_011145_20260418_011221/final/rmt_memory.pt",
    map_location="cpu",
)
for k, v in ckpt.items():
    norm = v.norm().item()
    std = v.float().std().item()
    trained = "UNTRAINED" if (std < 0.001 or torch.isnan(v).any()) else "possibly trained"
    print(f"  {k}: norm={norm:.2f}, std={std:.6f} -> {trained}")

# Specific check: importance gate
imp_w = ckpt["extractor.importance_updater.importance_mlp.2.weight"]
imp_b = ckpt["extractor.importance_updater.importance_mlp.2.bias"]
print(f"\n  importance_mlp.2.weight all zeros: {(imp_w == 0).all().item()}")
print(f"  importance_mlp.2.bias: {imp_b.item()} (NaN = never trained)")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
BUG CONFIRMED: train_rmt_v7.py has inverted bool attention mask.

Line ~428 in train_rmt_v7.py:
    attn_mask_float = attn_mask_float.masked_fill(attn_mask_seg, float('-inf'))

Should be:
    attn_mask_float = attn_mask_float.masked_fill(~attn_mask_seg, float('-inf'))

During training, this meant:
- Memory tokens could NOT attend to segment content
- Segment tokens could NOT attend to memory tokens or prior context
- The model effectively trained with almost all attention blocked
- Memory extractor received no meaningful hidden states -> weights stayed at init

The eval code (rmt_module.py _forward_single_segment) is CORRECT
and uses ~bool_mask_4d. But the model weights are garbage because
training never allowed proper information flow.

FIX: Change masked_fill(mask, -inf) to masked_fill(~mask, -inf) in
train_rmt_v7.py, then retrain as v9.
""")
