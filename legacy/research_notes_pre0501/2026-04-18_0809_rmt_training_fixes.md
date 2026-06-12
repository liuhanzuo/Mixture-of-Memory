# RMT Training Fixes — 2026-04-18 08:09

## Summary

Applied fixes identified in the RMT training audit.

## Bug 1: Inverted training attention mask (train_rmt_v7.py)

**Status: Already fixed in codebase.** Both occurrences (line ~158 in `validate()` and line ~418 in training loop) already use `~attn_mask_seg` with correct comment explaining the SDPA inversion convention. No change needed.

## Bug 2: Missing attention mask in eval generation (eval_needle_haystack.py)

**Status: Fixed.**

The `generate_rmt()` function was calling `model.model(inputs_embeds=..., output_hidden_states=False)` without `attention_mask` or `position_ids`, causing incorrect attention patterns during answer generation.

**Changes:**
- Added import of `build_rmt_attention_mask` and `build_rmt_position_ids`
- Replaced naive `torch.ones` mask with proper RMT attention mask using `build_rmt_attention_mask`
- Converted to additive float format (`~attn_mask -> -inf`) matching SDPA convention
- Added `position_ids` via `build_rmt_position_ids`
- Added dynamic mask/position rebuild for autoregressive generation steps (token-by-token grows the sequence)
- Pattern matches `eval_nih.py::rmt_inference()` implementation

## Bug 3: torch.no_grad() around memory extraction (train_rmt_v7.py)

**Status: Not applicable.** Memory extraction calls at lines 184 and 456 are not wrapped in `torch.no_grad()`. The only `@torch.no_grad()` is on the `validate()` function (correct) and init block (correct). No change needed.

## Files Touched

- `scripts/eval_needle_haystack.py` — fixed generation mask and added imports

## Validation

- Both scripts compile without syntax errors (`py_compile` passes)
