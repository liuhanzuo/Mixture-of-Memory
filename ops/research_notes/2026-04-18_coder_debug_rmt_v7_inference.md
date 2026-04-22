# Debug Report: RMT v7 Inference — 0% NIH Accuracy

**Date:** 2026-04-18  
**Status:** ROOT CAUSE IDENTIFIED

## Executive Summary

RMT v7 trained (loss 3.09→2.11, 20 epochs) but NIH eval shows 0% accuracy.  
**Root cause: The memory extractor never receives gradients during training because extraction runs under `torch.no_grad()`.** All inter-segment memory is essentially untrained random output. The model learns to use initial memory tokens but cannot compress/retrieve information across segments.

## Root Cause Detail

### Location: `scripts/train_rmt_v5.py`, lines ~230-240

```python
# Extract memory for next segment
with torch.no_grad():
    seg_hidden = hidden[:, args.num_memory_tokens:, :]
    mem_result = mem_module.extract_memory(seg_hidden.detach(), old_memory.detach() if old_memory is not None else None)
```

The entire memory extraction step is wrapped in `torch.no_grad()` AND inputs are `.detach()`'d. This means:
- `CrossAttentionExtractor` parameters → **zero gradients ever**
- `ImportanceMemoryUpdater` parameters → **zero gradients ever**
- `memory_embeddings` (initial memory) → **does get gradients** (used directly in forward)
- `segment_bias` → **does get gradients** (used directly in forward)
- `memory_predictor` → **does get gradients** (from Z-forcing loss, separate path)
- LoRA weights → **do get gradients** (from LM loss)

### Evidence from checkpoint weights

```
extractor.cross_attn_extractor.norm.weight: mean=1.000000, std=0.000000  ← still at init
extractor.cross_attn_extractor.norm.bias: mean=0.000000, std=0.000000   ← still at init
extractor.importance_updater.importance_mlp.2.weight: all zeros          ← still at init
extractor.importance_updater.importance_mlp.2.bias: all zeros            ← still at init
```

LayerNorm is frozen at (weight=1, bias=0) and the importance MLP last layer is exactly at its zero-initialization. The other weights (q_proj, k_proj, etc.) have values consistent with init (std ~0.009-0.036) — they may have drifted slightly from init due to optimizer momentum but never received meaningful gradients.

## Inference Path Trace (eval_nih.py → rmt_module.py)

1. **`rmt_inference()`** (eval_nih.py ~L174): Processes context in segments
2. For each segment: calls `rmt_model._forward_single_segment()` → returns `seg_hidden`
3. Then: `rmt_model.rmt.extract_memory(seg_hidden, old_memory)` → **uses untrained extractor** → produces garbage memory
4. Final memory is garbage → question segment has no useful context → model generates irrelevant text

The inference code itself is correct — it mirrors training faithfully. The problem is entirely in training.

## Suspected Causes — Ruled Out vs Confirmed

| Suspected Cause | Status | Notes |
|---|---|---|
| Memory tokens not prepended during inference | ❌ Ruled out | `_embed_with_memory()` correctly concatenates |
| Attention mask blocking memory attention | ❌ Ruled out | `build_rmt_attention_mask()` gives memory tokens full bidirectional attention |
| Position IDs wrong | ❌ Ruled out | `build_rmt_position_ids()` is reasonable |
| Training-inference code path mismatch | ❌ Ruled out | Both use same backbone, same mask format |
| `.detach()` blocking gradients | ✅ **CONFIRMED** | Extraction under `torch.no_grad()` + `.detach()` = zero gradient flow |
| Memory weights are garbage | ✅ **CONFIRMED** | Consequence of above — extractor never trains |
| `importance_mlp` all zeros → gate = 0.5 always | ✅ Confirmed | Importance gate is trivial (sigmoid(0)=0.5 for all slots) |

## Additional Issues Found

### Issue 2: `RMTModel.forward()` also has `torch.no_grad()` extraction
In `rmt_module.py` line ~300, the class method `forward()` also extracts memory without gradients, but this method isn't used in training (training uses the inline loop in train_rmt_v5.py).

### Issue 3: Per-segment backward with `retain_graph=False`
Training calls `loss.backward(retain_graph=False)` per segment, then extracts memory. This is fine for memory efficiency but combined with `torch.no_grad()` on extraction, it completely severs the gradient path to the extractor.

## Fix Recommendation

### Option A: Remove `torch.no_grad()` from extraction (simple fix)

In `scripts/train_rmt_v5.py`, change the memory extraction block:

```python
# BEFORE (broken):
with torch.no_grad():
    seg_hidden = hidden[:, args.num_memory_tokens:, :]
    mem_result = mem_module.extract_memory(seg_hidden.detach(), old_memory.detach() if old_memory is not None else None)

# AFTER (fixed):
seg_hidden = hidden[:, args.num_memory_tokens:, :]
mem_result = mem_module.extract_memory(seg_hidden, old_memory)
```

**But this creates OOM risk** — now the computation graph spans two segments (current + memory from previous). With `loss.backward()` already called, the graph from the current segment is freed. The extractor's computation graph for `old_memory` → `new_memory` would need to be retained.

### Option B: Keep no_grad extraction, add explicit extractor loss (safer)

Add a reconstruction/consistency loss that gives the extractor gradients WITHOUT connecting it to the main LM loss:

```python
# After extracting memory under no_grad, compute an auxiliary loss WITH gradients:
with torch.no_grad():
    seg_hidden_detached = hidden[:, args.num_memory_tokens:, :].detach()
    old_mem_detached = old_memory.detach() if old_memory is not None else None

# Extract WITH gradients for aux loss
mem_for_aux = mem_module.extract_memory(seg_hidden_detached, old_mem_detached)
if isinstance(mem_for_aux, tuple):
    new_mem_for_aux, aux_loss = mem_for_aux
else:
    new_mem_for_aux = mem_for_aux
    aux_loss = None

# Use new_mem_for_aux (detached) for next segment's input
old_memory = new_mem_for_aux.detach()

# Backprop aux loss to train extractor
if aux_loss is not None:
    (args.recon_weight * aux_loss).backward()
```

### Option C: Connect extractor to LM loss via straight-through estimator

```python
# Straight-through: use no_grad value but allow gradients to flow
with torch.no_grad():
    mem_result = mem_module.extract_memory(seg_hidden.detach(), ...)
    new_mem = mem_result[0] if isinstance(mem_result, tuple) else mem_result

# Re-compute with gradients (straight-through)
if old_memory is not None and old_memory.requires_grad:
    new_mem_st = mem_module.extract_memory(seg_hidden, old_memory)
    new_mem_st = new_mem_st[0] if isinstance(new_mem_st, tuple) else new_mem_st
    old_memory = (new_mem_st - new_mem.detach() + new_mem).detach()  # STE trick
else:
    old_memory = new_mem.detach()
```

### Recommended: Option B (simplest, lowest OOM risk)

## Validation Steps

After applying the fix:
1. Re-train RMT v8 with same hyperparameters
2. Monitor that extractor weights change from initialization
3. Check `importance_mlp.2.weight` is no longer all zeros
4. Re-run NIH eval

## Files Analyzed

- `src/memory/rmt/rmt_module.py` — RMTModel, RMTMemory, extractors
- `scripts/eval_nih.py` — NIH evaluation pipeline  
- `scripts/train_rmt_v5.py` — training script (where bug lives)
- `outputs/rmt_v7_*/final/rmt_memory.pt` — checkpoint weights (confirmed untrained)
- `outputs/rmt_v7_*/rmt_config.json` — training config (num_mem=64, seg_len=1024, extractor=v5)
