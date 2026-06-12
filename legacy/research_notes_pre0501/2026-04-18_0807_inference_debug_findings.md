# RMT Inference Debug Findings

**Date:** 2026-04-18  
**Checkpoint:** v8 (`rmt_v8_8gpu_20260418_011145_20260418_011221/final`)  
**Result:** 0/60 accuracy on NIH eval (baseline: 60/60)

## Executive Summary

**The RMT model fails because the Qwen3 attention implementation rejects the float32 additive attention mask when the model runs in bfloat16.** Even when the mask doesn't crash outright (SDPA path sometimes tolerates it), the attention computation is silently corrupted. The model with memory tokens prepended + custom attention mask produces garbage outputs, while the exact same base model without memory works perfectly.

## Root Causes (in order of severity)

### 1. **Dtype Mismatch in Attention Mask (CRITICAL)**

The RMT code builds a **float32** additive attention mask (`attn_mask_4d = torch.zeros_like(..., dtype=torch.float32)`) but the model runs in **bfloat16**. Qwen3's SDPA attention (`scaled_dot_product_attention`) raises:

```
RuntimeError: invalid dtype for bias - should match query's dtype
```

This was confirmed in the ablation test (`debug_rmt_ablation.py`, Test 3). During training, the code may have used a different attention backend (eager vs SDPA) or different PyTorch version that tolerated the mismatch.

**Fix:** Cast the attention mask to the same dtype as the query:
```python
attn_mask_4d = torch.zeros_like(bool_mask_4d, dtype=torch.bfloat16)  # NOT float32
```

### 2. **Memory Tokens Corrupt Generation Even When Mask Works**

Even in cases where the dtype mismatch doesn't crash (e.g., when the attention path falls back or in certain PyTorch versions), prepending 64 memory tokens fundamentally changes the model's behavior:

| Setup | Top-1 Token | Confidence |
|-------|------------|------------|
| Base model (no memory) | `ABC` | 25.88 |
| RMT with zero memory + mask | CRASH (dtype) | - |
| RMT with trained memory, no mask | `a` | 26.25 |
| RMT with trained memory + mask | `the` | 18.75 |

The model was never trained to answer questions from memory-contextualized prompts in this way. During training, memory is only used to carry information between segments of the **same** document. The generation phase prepends memory to the question prompt, which the model has never seen during training.

### 3. **Training vs Inference Mismatch**

**Training forward path:**
1. Process each segment: `[mem | segment_tokens]` through `backbone` (inner model)
2. Extract memory from hidden states
3. Use extracted memory for next segment

**Inference generation path:**
1. Process haystack segments identically ✓
2. **Generation phase:** prepend carried memory to question tokens → `[mem | question_tokens]`
3. Forward through `model.model` with custom attention mask

**Key issue:** The generation phase uses a completely novel input pattern that was never seen during training. During training, memory is always prepended to **continuation** segments of the same document. During eval, memory is prepended to an entirely different **question** prompt.

### 4. **Position ID Reset at Generation Time**

During generation, `build_rmt_position_ids(q_len, n_mem, 0, device)` resets `segment_idx=0`, giving memory tokens positions `0..63` and question tokens positions `0..q_len-1`. This means:
- Memory tokens get the same positions they had in segment 0 of the haystack
- Question tokens also start at position 0, overlapping with memory positions

During training, position IDs advance per segment (`seg_pos = arange(seq_len) + segment_idx * seq_len`), so there's no overlap. The position overlap at inference may confuse the model.

## Evidence

### Debug Script 1 (`debug_rmt_inference.py`)
- Loaded v8 checkpoint, ran NIH test case (needle at position 100 in 1024-token haystack)
- **Baseline answer:** "ABC123..." ✓
- **RMT answer (no mask):** "a sequence of numbers..." ✗
- **RMT answer (with mask):** "the secret code mentioned in the document..." (repeats question) ✗
- Memory was injected: initial norm=496, extracted norm=512

### Debug Script 2 (`debug_rmt_ablation.py`)
- Base model via `input_ids`: top-1 = "ABC" (correct)
- Base model via `inputs_embeds`: top-1 = "ABC" (correct) — rules out embedding pathway issue
- **Zero memory + RMT mask:** CRASHED with dtype error
- This proves the mask dtype is the primary failure mode

### Checkpoint Analysis
- `rmt_memory.pt` weights are non-trivial (not zeroed): `memory_embeddings` mean=0.000085, std=0.023
- `importance_updater.importance_mlp.2.weight` is all zeros, `.2.bias` is zero — the importance gate may be dead
- `segment_bias` has learned non-trivial values (mean=0.062, std=0.999)

## Recommended Fixes (Priority Order)

### P0: Fix dtype mismatch
```python
# In rmt_module.py _forward_single_segment() and eval_needle_haystack.py generate_rmt()
attn_mask_4d = torch.zeros_like(bool_mask_4d, dtype=memory_embeddings.dtype)  # bfloat16, not float32
```

### P1: Fix generation position IDs
Don't reset segment_idx to 0 for generation. Use a position offset that continues from the last segment:
```python
gen_position_ids = build_rmt_position_ids(q_len, n_mem, num_segments, device)
```

### P2: Train with question-answer pairs
The model needs to learn to decode answers from memory-contextualized questions. Add a QA fine-tuning stage where the last segment is a question and the model must generate the answer using information carried in memory from earlier segments.

### P3: Investigate dead importance gate
`importance_updater.importance_mlp.2.weight` is all zeros after training, meaning the importance-based selective update is not functioning. This may be a training bug (vanishing gradients through the gate path).
