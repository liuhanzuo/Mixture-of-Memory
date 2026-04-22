# RMT Inference Debug Report

**Date:** 2026-04-19  
**Checkpoint:** `outputs/rmt_v10_8gpu_20260419_001626_20260419_001703/final/`  
**Status:** 4 bugs found and fixed. Technical inference bugs resolved. Format mismatch remains.

## Summary

RMT inference was producing 0% accuracy across v7/v9/v10 despite training showing near-perfect retrieval loss (~0.008). Root cause: **multiple train/eval mismatches in the eval pipeline**.

## Bugs Found & Fixed

### FIX-002: English question in eval vs Chinese question in training

**Location:** `scripts/eval_needle_haystack.py` — `generate_rmt()` function  
**Was:** `question_text = "What is the secret code mentioned in the document? The secret code is"`  
**Fixed to:** `question_text = "请问 文档中提到的秘密编号是什么？"`  
**Why:** Training retrieval uses `"请问 {name} 的编号是什么？"` (Chinese). The model's retrieval head was never trained on English question patterns. Using a completely different language at eval means the retrieval pattern doesn't match.

### FIX-003: Wrong segment_idx in generation position_ids

**Location:** `scripts/eval_needle_haystack.py` — `generate_rmt()`  
**Was:** `build_rmt_position_ids(q_len, n_mem, 0, device)`  
**Fixed to:** `build_rmt_position_ids(max_combined, n_mem, last_seg_idx, device)`  
**Why:** During training, retrieval happens at `seg_idx = num_segments - 1`. With `segment_length=1024` and 4 segments, training uses `segment_idx=3`, giving RoPE position offsets of `3 * combined_len ≈ 57` for question tokens. Eval used `segment_idx=0`, giving offset 0. This completely changes the rotary positional encoding that the retrieval head relies on.

### FIX-004: Question tokens inside document segments

**Location:** `scripts/eval_needle_haystack.py` — main eval loop  
**Was:** `rmt_tokens = list(full_tokens)` where `full_tokens` includes the English question appended at the end  
**Fixed to:** `rmt_tokens = list(rmt_doc_tokens)` where `rmt_doc_tokens` is document-only (no question)  
**Why:** In training, the question is NEVER part of the segmented document. It's only used during the separate retrieval phase. Including it in the document segments (a) pollutes the memory with question tokens, and (b) doesn't match training where memory only sees document content.

### FIX-005: Position drift during autoregressive generation

**Location:** `scripts/eval_needle_haystack.py` — generation loop  
**Was:** `build_rmt_position_ids(cur_len - n_mem, n_mem, segment_idx, device)` called each step  
**Problem:** `build_rmt_position_ids` computes `arange(seq_len) + segment_idx * seq_len`. When `seq_len` grows during generation, ALL previous positions shift. Token at position 17 becomes 18, etc. This breaks RoPE coherence.  
**Fixed:** Precompute positions once using `max_combined = q_len + max_new_tokens`, then slice `gen_position_ids[:, :cur_len]` for each step. This keeps positions fixed.

## What Changed

- `scripts/eval_needle_haystack.py`: Fixed question language, segment_idx, removed question from segments, fixed position drift

## Validation

After fixes, RMT model now generates **structured Chinese answers** instead of garbage:
- Before: `" a string of 1000000000 from the secret code..."` (incoherent)
- After: `"文档中提到的秘密编号是：**"7741"**"` (structured Chinese, wrong content)

## Remaining Issue: Format Mismatch (NOT a code bug)

The RMT model was trained with:
- Chinese haystack text
- Chinese needle format: `"记住这个信息：阿尔法 的编号是 X7742。"`
- Chinese retrieval question: `"请问 阿尔法 的编号是什么？"`

The eval uses:
- English haystack text  
- English needle format: `"The secret code is XAJI0Y."`
- (Now) Chinese retrieval question (mismatch with English needle content)

The model correctly retrieves in the Chinese pattern it was trained on, but the actual needle information (English code in English text) is not properly captured by a Chinese-trained memory extractor. The `"7741"` answer likely comes from training data interference.

**Recommendation:** Either:
1. Retrain with English haystack+needles (or bilingual data), OR
2. Change eval to use Chinese format matching training, OR
3. Create a bilingual training dataset that covers both languages

## Technical Verification

Memory mechanism confirmed working at inference:
- Memory buffer is non-zero after segment processing
- Memory changes after each segment (information accumulation)
- Logits with vs without memory differ significantly (norm diff ~490)
- Dropout is disabled in eval mode (model.eval() called correctly)
- LoRA correctly merged into checkpoint (no adapter files in final/)
- Cross-attention extractor called during segment processing
- Importance memory updated between segments
- No shape mismatches causing silent failures

## Files Modified

| File | Change |
|------|--------|
| `scripts/eval_needle_haystack.py` | FIX-002, FIX-003, FIX-004, FIX-005 |
