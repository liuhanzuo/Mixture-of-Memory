# RMT Inference Debug Report

**Date**: 2026-04-18 19:30 GMT+8
**Checkpoint**: `outputs/rmt_v9_8gpu_20260418_083226_20260418_083302/final/`
**Config**: segment_length=1024, num_memory_tokens=64, bottleneck_dim=256, extractor_version=5

## Summary

**The RMT inference pipeline is mechanically correct** — memory tensors flow through the model, attention masks are properly constructed, position IDs are consistent between training and inference. The 0% NIH accuracy is caused by **the memory extractor failing to preserve specific factual information** (the "needle"), not by an implementation bug.

## Detailed Findings

### 1. Memory Weights Are Non-Trivial (✓)

All RMT parameters have reasonable statistics — non-zero means, healthy std, no collapsed weights:
- `memory_embeddings`: mean=0.000082, std=0.023
- `cross_attn_extractor.*`: std ≈ 0.01–0.04 (normal initialization range)
- `importance_updater`: initialized near zero (sigmoid→0.5 gate)
- `memory_predictor`: standard init

### 2. Memory Extraction Produces Non-Zero Values (✓)

After processing a 1024-token segment containing needle "ABC123":
- Initial memory: mean=0.069, std=0.966
- Extracted memory: mean=0.017, std=0.699, norms ≈ 38–51 per token
- Memory diff: mean=-0.052, std=0.780, max_abs=4.375

Memory is actively modified by the segment content. The extractor IS reading the hidden states.

### 3. Memory Influences Model Output (✓)

Comparing logits with real memory vs zero memory:
- Logits diff: mean=0.770, **max=8.06**
- The memory signal reaches the output layer with significant magnitude.

### 4. Attention Mask & Position IDs Are Correct (✓)

- Mask shape: `[1088, 1088]` (= 64 memory + 1024 segment)
- Memory rows: all True (bidirectional) ✓
- Segment rows: causal (tril) + all memory columns True ✓
- Qwen3 dict format `{"full_attention": mask_4d}` works correctly ✓
- Position IDs: memory tokens at 0..63, segment tokens at `segment_idx * 1024 + offset` ✓

### 5. **ROOT CAUSE: Memory Does Not Preserve Needle Information** (✗)

**Critical test** (Debug 6): Segment 0 contains "The secret code is XYZ789." Memory is extracted and injected into segment 1 containing the question. Result:
- **Top-10 tokens: `[' a', ' the', ' an', ' ', ' in', ' to', ' "', ' found', ' made', ' something']`**
- Neither "XYZ" nor "789" appears anywhere in the top-10.
- Generated text: `" 1234567890.\n\nWhat is the secret code mentioned in the document? The secret code is 123"` — complete hallucination.

The memory extractor (V5 CrossAttentionExtractor) compresses 1024 tokens × 4096 dims into 64 × 4096 memory tokens via bottleneck cross-attention (Q/K/V projected to 256 dims). This compression:
- Preserves **general topic/distribution information** (hence memory influences output)
- **Destroys specific factual details** like a 6-character code

This is expected: cross-attention with a bottleneck averages/soft-selects from the full sequence, producing a "fuzzy summary" rather than a precise copy of rare tokens.

## Why Training Loss Drops But NIH Is 0%

Training loss (next-token prediction) decreases because:
1. The memory carries enough **distributional** information to help predict common continuations
2. Chinese wiki text is highly repetitive — general topic information suffices
3. The LM head can use the visible context + memory "gist" to predict next tokens

But NIH requires **exact retrieval of a rare, specific fact** that appears once in the haystack. The memory extractor was never pressured to preserve this level of detail — the LM loss doesn't penalize forgetting the needle because the next token is usually common text, not the needle itself.

## Potential Fixes (Priority Order)

### P0: Add Retrieval-Oriented Auxiliary Loss
- During training, randomly insert "needles" and add a loss that forces the model to reproduce them from memory alone (without the original segment visible)
- This directly trains the memory extractor to preserve specific facts

### P1: Increase Bottleneck Capacity
- Current: 64 memory tokens with 256-dim bottleneck attention
- Try: 128–256 memory tokens, or bottleneck_dim=512–1024
- More capacity = more room to preserve details

### P2: Memory Token Read-Back Mechanism
- Instead of cross-attention pooling (which soft-averages), add a mechanism that can copy exact hidden state vectors into memory slots
- E.g., a "copy" gate: `new_memory[i] = alpha * copied_hidden + (1-alpha) * cross_attn_output`

### P3: Two-Stream Memory
- Split memory into: (a) "gist" stream (current cross-attention) for topic info, (b) "detail" stream with slot attention or sparse retrieval for specific facts

## Inference Pipeline Bugs Found (Minor)

1. **`torch_dtype` deprecation warning** — should use `dtype` kwarg in `AutoModelForCausalLM.from_pretrained`
2. **Eval script generation loop** rebuilds full attention mask + re-encodes all tokens each step (no KV cache). Correct but extremely slow for longer generation.
3. **`generate_rmt` in eval uses `rmt_model.model.model()` + `rmt_model.model.lm_head()`** — bypasses the model's own generate method and any chat template processing. This is intentional but means no `apply_chat_template` or system prompt.

## Files Touched
- Created: `scripts/debug_rmt_inference.py`
- Created: `ops/research_notes/2026-04-18_1930_rmt_inference_debug.md`

## Validation
- Debug script ran successfully on GPU 0 with v9 checkpoint
- All 7 debug checks completed without errors
- Key evidence: logits diff (real vs zero memory) = 8.06 max, but no needle token in top-10
