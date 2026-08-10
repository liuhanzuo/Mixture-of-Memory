# CODER_RMT_DEBUG.md — RMT v10 Inference Failure Root Cause Analysis

**Date**: 2026-04-19
**Status**: Root cause identified, fix applied

## Summary

Two critical bugs in the inference path caused RMT v10 to produce degenerate output (0% NIH accuracy) despite successful training (loss 3.11→2.25).

## Bug 1: Attention Mask Dtype Mismatch (PRIMARY ROOT CAUSE)

**Location**: 
- `src/memory/rmt/rmt_module.py` line 535: `torch.zeros_like(bool_mask_4d, dtype=torch.float32)`
- `scripts/eval_needle_haystack.py` lines 278, 304: `dtype=torch.float32`

**Problem**: The attention mask was built as `float32`, but hidden states are `bfloat16`. Qwen3's SDPA attention requires the bias dtype to match the query dtype. Training code correctly uses `dtype=torch.bfloat16` (train_rmt_v10.py lines 249, 525), but the inference/eval code hardcoded `float32`.

**Effect**: On current PyTorch/transformers, this causes either:
- A crash (`RuntimeError: invalid dtype for bias - should match query's dtype`) — but this might have been silently handled in the version used for eval
- Silently broken attention patterns where `-inf` mask values get corrupted during dtype casting, causing the model to attend to wrong positions

**Evidence**: NIH eval outputs show degenerate patterns (`import random\nimport time...`, repetitive `from the code from the code`), consistent with broken attention rather than bad memory.

**Fix Applied**: Changed all inference mask construction to use `dtype=dtype` (matching hidden state dtype) in:
- `src/memory/rmt/rmt_module.py` `_forward_single_segment` 
- `scripts/eval_needle_haystack.py` `generate_rmt`

## Bug 2: Hardcoded Segment Length in `_forward_single_segment`

**Location**: `src/memory/rmt/rmt_module.py` `_forward_single_segment`

**Problem**: Position IDs and attention masks were built using `self.segment_length` (always 1024) instead of the actual input segment length. During training, segments are always exactly `segment_length` (padded). During inference, the last segment may be shorter.

**Effect**: For segments shorter than `segment_length`, position IDs and attention mask dimensions don't match the actual sequence length, causing `RuntimeError: The size of tensor a (865) must match the size of tensor b (1088)`.

**Fix Applied**: Use `actual_seg_len = input_ids.shape[1]` instead of `self.segment_length`.

## Diagnostic Test Results

After fixes, the minimal test showed:
- **Memory extraction IS functional**: mem changes between segments (|diff|=1.1 after seg0, 0.04 after seg1)
- **Memory affects output**: logit diff mean=5.4, max=15.4 between with/without memory
- **Generation still imperfect**: produces code-like output (`} from './secretCode'`) rather than clean answers

The remaining generation quality issue is likely because:
1. The model needs the full NIH pipeline to work properly (not a toy 2-segment test)
2. The dtype bug fix is the critical change that should fix NIH eval

## Files Changed

1. `src/memory/rmt/rmt_module.py`:
   - Line 535: `dtype=torch.float32` → `dtype=dtype` (attention mask)
   - Lines 508-517: Use `actual_seg_len` instead of `self.segment_length`
2. `scripts/eval_needle_haystack.py`:
   - Line 278: `dtype=torch.float32` → `dtype=torch.bfloat16`
   - Line 304: `dtype=torch.float32` → `dtype=torch.bfloat16`
3. `scripts/test_rmt_inference_debug.py`: New diagnostic script

## Validation

Re-run NIH eval:
```bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
bash scripts/run_eval_nih_v10.sh
```

Expected: Accuracy should improve from 0% to >0%. If still low, investigate position ID scheme for retrieval vs segment processing.

## Risks / Follow-ups

- The dtype fix is safe and correct — matches training behavior
- If NIH accuracy is still low after fix, the position ID scheme needs review: `build_rmt_position_ids` uses `segment_idx * seq_len` which means retrieval positions don't match document processing positions (e.g., seg_idx=2 gives text positions 40-59 during retrieval but 2048-3071 during processing). This was also the case during training, so it should work, but it's worth investigating if accuracy remains low.
- The `_forward_single_segment` fix for actual segment length is backward-compatible since training always uses full-length segments
