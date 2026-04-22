# Inference Fix: generate_with_memory() for RMT

## Date: 2026-04-18

## Problem
The eval script's generation loop called `self.model.model(inputs_embeds=...)` without attention mask or position_ids. Memory tokens received default causal attention instead of the bidirectional attention they had during training, causing memory to be effectively ignored during answer generation.

## Changes

### 1. `src/memory/rmt/rmt_module.py`
- Added `generate_with_memory()` method to `RMTModel` class
- Builds correct 4D attention mask using `build_rmt_attention_mask()` (same as training)
- Memory tokens get full bidirectional attention; question tokens get causal attention to memory + preceding tokens
- Builds correct position_ids: memory at 0..N-1, question continues from segment's global position
- Uses KV cache for efficient autoregressive decoding (no full recomputation per token)

### 2. `scripts/eval_rmt.py`
- Replaced the broken generation loop (full-sequence recomputation, no mask) with a single call to `self.model.generate_with_memory(question_ids, old_memory, segment_idx=num_segments-1, max_new_tokens=20)`

## Expected Impact
- NIH accuracy should improve significantly — memory tokens now actually participate in generation
- Generation speed should improve due to KV cache (no O(n²) full-sequence recomputation per token)

## Validation
Run: `python scripts/eval_rmt.py --checkpoint_dir <dir> --eval_type nih --nih_num_trials 1`

## Risks
- The `{"full_attention": attn_mask_4d}` format must match what Qwen3's `masking_utils` expects — this is the same format used during training so should be consistent
- KV cache + custom attention mask interaction: during autoregressive decode, only position_ids and past_kv are passed (no attention_mask), which relies on Qwen3's default causal behavior for cached tokens — this is correct since we only need causal for generated tokens
