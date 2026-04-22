# Debug Report: RMT v7 Inference Pipeline — 0% NIH Accuracy

**Date:** 2026-04-18
**Status:** ROOT CAUSE IDENTIFIED — Training attention mask is inverted

---

## Executive Summary

The RMT v7 model achieves 0% NIH accuracy because **the training attention mask was inverted**, causing anti-causal attention during training. The model "cheated" by attending to future tokens, making the training loss decrease (3.09→2.11) meaningless. During eval, the correct mask prevents seeing the future, so the model generates degenerate outputs.

**Secondary issue:** The eval script `eval_needle_haystack.py` passes NO attention mask during generation, relying on Qwen3's default causal mask (which blocks memory-to-text bidirectional attention during generation).

---

## Bug #1 (CRITICAL): Inverted Training Attention Mask

### Location
- `scripts/train_rmt_v7.py` lines 157-158 (training loop)
- `scripts/train_rmt_v7.py` lines 417-418 (validation loop)

### The Bug
```python
# build_rmt_attention_mask returns: True = CAN ATTEND (standard causal convention)
attn_mask_seg = build_rmt_attention_mask(actual_seg_len, args.num_memory_tokens, device)

# BUG: fills True (can-attend) positions with -inf (BLOCKED)
attn_mask_float = torch.zeros_like(attn_mask_seg, dtype=torch.float32)
attn_mask_float = attn_mask_float.masked_fill(attn_mask_seg, float('-inf'))  # ← WRONG: missing ~
```

### Correct Code (used in eval `rmt_module.py` line 510)
```python
attn_mask_4d = attn_mask_4d.masked_fill(~bool_mask_4d, float('-inf'))  # ← CORRECT: ~ inverts
```

### Impact
The inverted mask creates **anti-causal** attention:
- **Memory tokens**: can't attend to ANYTHING (all positions get -inf → uniform attention)
- **Text tokens**: can ONLY attend to FUTURE tokens (upper triangle = 0.0, lower triangle = -inf)
- **Result**: The model sees the answer while training, making loss decrease meaningless

### Evidence
1. Training loss went 3.09→2.11 despite broken mask — the model "cheats" by seeing future tokens via anti-causal attention
2. `importance_mlp[-1].weight` is ALL ZEROS — the importance gate never trained because the model never needed memory
3. Memory embeddings for segments 1-6 are IDENTICAL (cosine similarity = 1.0) — no differentiation needed when cheating
4. RMT outputs are all "a 10-digit number..." — degenerate pattern from a model that never learned genuine retrieval

---

## Bug #2 (MODERATE): Missing Attention Mask in Eval Generation

### Location
- `scripts/eval_needle_haystack.py` lines 248-255 (generation loop)

### The Bug
```python
outputs = rmt_model.model.model(
    inputs_embeds=inputs_embeds,
    output_hidden_states=False,
    # ← NO attention_mask! NO position_ids!
)
```

### Impact
Without an explicit attention mask, Qwen3's `forward()` creates a default causal mask via `create_causal_mask()`. This means:
- Memory tokens only see preceding tokens (causal), NOT bidirectional
- Position IDs are auto-generated from `cache_position` starting at 0, not from `build_rmt_position_ids`

This is a separate issue from Bug #1 but compounds the problem.

Note: `eval_nih.py`'s `rmt_inference()` function does NOT have this bug — it properly passes `attention_mask={"full_attention": ...}` and `position_ids`.

---

## Weight Inspection Results

### Memory Module Weights (`rmt_memory.pt`)
| Parameter | Shape | Mean | Std | Status |
|---|---|---|---|---|
| memory_embeddings | [7, 64, 4096] | 0.0001 | 0.023 | Segments 1-6 IDENTICAL |
| extractor.cross_attn_extractor.memory_queries | [64, 4096] | -0.0001 | 0.020 | Looks initialized |
| extractor.importance_updater.importance_mlp.2.weight | [1, 1024] | 0.000 | 0.000 | **ALL ZEROS** (never trained) |
| extractor.importance_updater.importance_mlp.2.bias | [1] | 0.000 | — | **ZERO** (never trained) |
| memory_predictor.* | various | ~0.000 | ~0.009 | Near initialization |
| segment_bias.weight | [7, 64] | 0.062 | 0.999 | Some differentiation |

### Key Observations
1. **Segments 1-6 identical**: Memory embeddings were initialized from backbone token embeddings, all segments got the same initialization, and training gradients (due to inverted mask) couldn't differentiate them
2. **Importance gate dead**: The `importance_mlp` final layer is zero → sigmoid(0) = 0.5 → gate is always exactly `0.1 + 0.9 * 0.5 = 0.55`. It never learned importance-based updating.
3. **Memory predictor untrained**: All weights near initialization values

---

## Environment Notes
- **torch-base env**: transformers 4.40.0 — does NOT support Qwen3 model type
- **mom env**: transformers 5.3.0 — supports Qwen3, handles `{"full_attention": tensor}` dict format correctly
- The training must have been run in the `mom` env (transformers 5.3.0)
- The eval was also run in `mom` env (it loaded Qwen3 successfully)

---

## Recommended Fix

### Step 1: Fix training mask (Bug #1)
In `scripts/train_rmt_v7.py`, change line 158 and line 418:
```python
# BEFORE (broken):
attn_mask_float = attn_mask_float.masked_fill(attn_mask_seg, float('-inf'))

# AFTER (correct):
attn_mask_float = attn_mask_float.masked_fill(~attn_mask_seg, float('-inf'))
```

### Step 2: Fix eval generation mask (Bug #2)  
In `scripts/eval_needle_haystack.py`, add attention_mask and position_ids to the generation loop, similar to `eval_nih.py`'s `rmt_inference()`.

### Step 3: Retrain
The v7 checkpoint is fundamentally broken (trained with anti-causal attention). No fix at eval time can recover useful knowledge. **Must retrain from scratch** with the corrected mask.

---

## Validation Plan
1. Apply mask fix to training script
2. Train for 2-3 epochs on small subset to verify loss dynamics are different
3. Run NIH eval on 5-test subset to check accuracy > 0%
4. If confirmed, full retrain

---

## Files Analyzed
- `src/memory/rmt/rmt_module.py` — RMT module (correct mask in `_forward_single_segment`)
- `scripts/train_rmt_v7.py` — training (INVERTED mask in training + validation loops)
- `scripts/eval_nih.py` — eval NIH (correct mask in `rmt_inference`)
- `scripts/eval_needle_haystack.py` — eval NIH alt (NO mask in generation loop)
- `outputs/rmt_v7_8gpu_20260417_191424_20260417_191503/final/rmt_memory.pt` — weights
- `outputs/rmt_v7_8gpu_20260417_191424_20260417_191503/final/rmt_config.json` — config
- `outputs/nih_eval_v7/nih_results.json` — eval results confirming 0% accuracy
