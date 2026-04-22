# RMT Inference Root Cause Analysis

**Date**: 2026-04-19 12:55
**Status**: Bug found — LoRA-merged weights not loaded during evaluation

## Summary

**Root cause**: `eval_needle_haystack.py` loads the **original base model** (Qwen3-8B) instead of the **LoRA-merged model** saved in the checkpoint. The trained LoRA adapter is completely invisible during evaluation. This alone explains the 0% NIH accuracy — the backbone produces different hidden states than what the memory extractor was trained on, making extracted memory meaningless.

## Bug #1 (Critical): LoRA weights silently dropped

**File**: `scripts/eval_needle_haystack.py`, lines 145–166

```python
# eval loads from base_model_path (original Qwen3-8B)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,        # ← ../models/Qwen--Qwen3-8b (UNTRAINED)
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map={"": device},
)

# then looks for adapter in checkpoint_dir (final/)
adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
adapter_bin = os.path.join(checkpoint_dir, "adapter_model.bin")
if os.path.exists(adapter_path) or os.path.exists(adapter_bin):
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model = model.merge_and_unload()
else:
    logger.info("No LoRA adapter found, using base model weights.")  # ← THIS PATH IS TAKEN
```

**Why this happens**: Training saves the LoRA-merged model via `merged.save_pretrained(output_dir/final/)`, which writes full `model.safetensors` (base + LoRA combined) and **no adapter files**. The eval only checks for `adapter_model.safetensors` / `adapter_model.bin` — since those don't exist, it silently falls through to the unmodified base model.

**Evidence**: Verified `final/` directory contains `model.safetensors` (16 GB merged weights) and `rmt_memory.pt`, but no `adapter_model.*` files.

**Impact**: The RMT memory extractor (CrossAttentionExtractor, ImportanceMemoryUpdater) was trained alongside LoRA-modified hidden states. When applied to the original model's hidden states, the extracted memory is meaningless noise. The model cannot recall anything from memory.

### Fix

Change `load_rmt_model()` to prefer loading from `checkpoint_dir` when it contains a full model:

```python
# Option A: Load merged model from checkpoint_dir directly
model_dir = checkpoint_dir if os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")) else base_model_path
model = AutoModelForCausalLM.from_pretrained(
    model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device},
)
# Then only apply LoRA if separate adapter files exist
adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
if os.path.exists(adapter_path) or os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin")):
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model = model.merge_and_unload()
```

## Verified: NOT bugs

1. **Memory initialization** — `get_initial_memory()` is identical in training and eval. Both use `memory_embeddings[seg_idx] + segment_bias[seg_idx]`. ✓

2. **Segment position IDs** — Both training and eval use `build_rmt_position_ids()` with the same `segment_idx * seq_len` offset. ✓

3. **Memory flow between segments** — `old_memory` is properly threaded through segments in both paths. Eval correctly uses `mem_result[0]` for V5 tuples. ✓

4. **V5 extractor code path** — Same `CrossAttentionExtractor` → `ImportanceMemoryUpdater` pipeline in both training and eval. ✓

5. **Checkpoint loading for RMT memory** — `rmt_memory.pt` is loaded correctly with matching architecture (64 tokens, bottleneck=256, extractor_version=5). ✓

## Secondary concern: Position ID drift in eval generation

**File**: `scripts/eval_needle_haystack.py`, `generate_rmt()` function

Training retrieval (`compute_retrieval_loss` in `train_rmt_v10.py`) uses:
```python
combined_len = q_len + a_len  # actual answer length
position_ids = build_rmt_position_ids(combined_len, num_memory_tokens, seg_idx, device)
```

Eval generation uses:
```python
max_combined = q_len + max_new_tokens  # 30
gen_position_ids = build_rmt_position_ids(max_combined, n_mem, last_seg_idx, device)
```

This creates different absolute position values (affects RoPE). Minor issue compared to Bug #1 but should be aligned. The existing `generate_with_memory()` method in `RMTModel` also has similar position drift.

## Language mismatch note

The eval defaults to `--lang en` but training uses Chinese data. The shell script (`run_eval_nih_v10.sh`) doesn't pass `--lang zh`. This is a config issue, not a code bug, but should be noted.

## Priority

1. **Fix Bug #1 first** — this alone should dramatically improve results
2. **Align position IDs** between training retrieval and eval generation
3. **Add `--lang zh`** to eval launcher for Chinese-trained models
