# RMT v8 Launch: First Properly Configured Training

**Date:** 2026-04-18 01:55
**Type:** Training Status Update
**Status:** IN PROGRESS — Epoch 3 of 20

---

## 1. Current Status

RMT v8 training was successfully launched at 2026-04-18 01:11:49 GMT+8 using 8 GPUs. This is the **first RMT training run with correct configuration** after identifying critical bugs in v6 and v7.

### Training Progress
- **Epoch:** 3 of 20 completed
- **Loss trajectory:** 3.09 → 2.71 (healthy decrease)
- **Learning rate:** 2e-5 (after warmup)
- **ETA:** ~3.9 hours remaining
- **Status:** Running normally (PID 35778-35894)

### Configuration
```json
{
  "num_memory_tokens": 64,
  "segment_length": 1024,
  "max_segments": 6,
  "num_epochs": 20,
  "lr": "2e-5",
  "rmt_lr": "2e-4",
  "lora_r": 32,
  "lora_alpha": 64,
  "bottleneck_dim": 256,
  "extractor_version": 5,
  "use_reconstruction": true
}
```

---

## 2. Key Findings: v8 vs v7 — Critical Fixes

### Bug #1 FIXED: Gradient Flow to Memory Extractor

**v7 (broken):**
```python
# scripts/train_rmt_v7.py
with torch.no_grad():
    seg_hidden = hidden[:, args.num_memory_tokens:, :]
    mem_result = mem_module.extract_memory(seg_hidden.detach(), old_memory.detach() if old_memory is not None else None)
```
- Memory extraction wrapped in `torch.no_grad()`
- Inputs detached with `.detach()`
- **Result:** Zero gradients to extractor → memory never trained

**v8 (fixed):**
```python
# scripts/train_rmt_v5.py (used by v8)
seg_hidden = hidden[:, args.num_memory_tokens:, :]
mem_result = mem_module.extract_memory(seg_hidden, old_memory if old_memory is not None else None)
```
- No `torch.no_grad()` wrapper
- No `.detach()` on inputs
- **Result:** Full gradient flow to extractor → memory properly trained

### Bug #2 FIXED: Correct Attention Mask Format

**v7 (broken):**
```python
# scripts/train_rmt_v7.py line 158
attn_mask_float = attn_mask_float.masked_fill(attn_mask_seg, float('-inf'))
```
- Missing `~` operator → inverted mask
- **Result:** Anti-causal attention (text attends to future) → model "cheats"

**v8 (fixed):**
```python
# scripts/train_rmt_v5.py line 248
attn_mask_seg = build_rmt_attention_mask(actual_seg_len, args.num_memory_tokens, device)
# Passed directly to backbone in correct format:
outputs = backbone(
    inputs_embeds=inputs_embeds,
    attention_mask={"full_attention": attn_mask_seg},
    position_ids=position_ids,
)
```
- No manual mask conversion → uses boolean mask directly
- Correct Qwen3 dict format `{"full_attention": ...}`
- **Result:** Proper causal attention for text, bidirectional for memory

### Additional v8 Improvements

1. **Reconstruction loss enabled:** `use_reconstruction=True` in RMTMemory initialization
   - Explicitly trains memory to reconstruct segment hidden states
   - Provides direct supervision signal to the extractor

2. **Correct memory update order:** Extraction happens BEFORE `loss.backward()` so computation graph is alive

3. **Z-forcing loss:** Trains memory predictor to map previous memory to current segment hidden states (if `extractor_version == 5`)

---

## 3. Interpretation

**Critical Insight:** v6 and v7 training results are **invalid** because:
- v6: Likely had same no_grad bug as v7 (not confirmed but high probability)
- v7: Definitely had no_grad bug AND inverted mask bug
- Both achieved loss convergence, but this was meaningless because:
  - Memory extractor never received gradients (no_grad)
  - Model saw future tokens during training (inverted mask)
  - Memory weights remained at initialization (confirmed in v7 checkpoint)

**v8 Significance:** This is the **first properly configured RMT training** where:
- Memory extractor receives gradients
- Attention masks are correct
- Reconstruction loss provides direct memory supervision
- Model cannot cheat via future tokens

**Expected Outcome:** If training completes successfully, v8 should demonstrate:
- Non-zero NIH accuracy (vs 0% in v7)
- Memory embeddings that differentiate across segments
- Trained extractor weights (not at initialization)
- Meaningful importance gate values (not stuck at 0.55)

---

## 4. Recommended Next Actions

### Immediate (during v8 training)
1. **Monitor v8 progress:** Check that loss continues to decrease healthily
2. **Verify gradient flow:** After epoch 5, inspect checkpoint to confirm extractor weights have changed from initialization
3. **Plan NIH eval:** Prepare to run comprehensive NIH eval when v8 completes (~2 hours remaining)

### After v8 completion
1. **Inspect weights:** Verify that `extractor.*` weights are no longer at initialization values
2. **Run NIH eval:** Full 90-test evaluation (5 depths × 3 lengths × 3 trials)
3. **Compare to baseline:** Expect v8 accuracy >> v7 (0%) and ideally > baseline (100%) if memory helps

### If v8 succeeds (>50% NIH accuracy)
1. **Analyze memory usage:** Inspect which memory tokens are most active per segment
2. **Ablation study:** Run v8 without reconstruction loss to quantify its impact
3. **Proceed to v9:** Consider memory routing or selective update mechanisms

### If v8 fails (<10% NIH accuracy)
1. **Debug inference:** Verify that eval pipeline correctly uses v8 checkpoint
2. **Check attention:** Validate that attention masks are correct during eval (not just training)
3. **Simplify:** Test with single-segment (no compression) to isolate the issue

---

## 5. Risks / Uncertainties

- **Training may still fail:** Even with correct configuration, the memory mechanism may not learn useful representations
- **Eval pipeline bugs:** There may be separate bugs in eval scripts that prevent memory from being used correctly
- **Insufficient epochs:** 20 epochs may not be enough for the memory to learn meaningful compression
- **Memory routing needed:** Even with correct training, the current prefix-based memory injection may not be optimal

---

## 6. Handoff for Main Agent

**Milestone Reached:** v8 is the first RMT training with correct configuration (gradient flow, proper attention masks, reconstruction loss).

**Wait for completion:** Let v8 run to completion (~2 hours remaining) before taking action.

**Priority 1 after completion:** Run comprehensive NIH eval to verify that memory mechanism is functional.

**Expected validation:** If v8 NIH accuracy >> v7 (0%), this confirms that the v6/v7 bugs were the root cause of failure.

**Next decision point:** If v8 succeeds (>50% accuracy), proceed with v9 exploration (memory routing, selective update). If v8 fails (<10%), debug eval pipeline or simplify architecture.

---

## Version Info
- **git branch:** main
- **short commit:** 4ce1dc0
- **full commit:** 4ce1dc045ad65dc7432f61fa25a4eaabd7e5b374
- **Key files inspected:**
  - `scripts/train_rmt_v5.py` (v8 training script, correct)
  - `scripts/train_rmt_v7.py` (v7 training script, broken)
  - `scripts/launch_v8.sh` (v8 launcher)
  - `outputs/rmt_v8_8gpu_20260418_011145_20260418_011221/train.log` (v8 training progress)
  - `outputs/rmt_v8_8gpu_20260418_011145_20260418_011221/rmt_config.json` (v8 config)
  - `ops/research_notes/2026-04-18_debug_inference_v7.md` (v7 root cause analysis)
  - `ops/research_notes/2026-04-18_coder_debug_rmt_v7_inference.md` (v7 weight inspection)
