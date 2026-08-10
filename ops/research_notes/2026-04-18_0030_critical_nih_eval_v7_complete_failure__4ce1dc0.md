# Critical: NIH Eval Complete Failure on RMT v7

**Date:** 2026-04-18 00:30 (GMT+8)
**Type:** Critical Evaluation Report
**Status:** BLOCKER — RMT mechanism completely broken for retrieval

---

## 1. Current Status

Two major training runs completed, followed by a comprehensive NIH eval:

### A. RMT v6 Training Complete
- **Completed:** 2026-04-17 19:06 GMT+8
- **Epochs:** 20 (resumed from epoch 10)
- **Loss trajectory:** ~2.3-2.5 in epochs 10-20
- **Output:** `outputs/rmt_v6_20260417_165231/`
- **Config:** LoRA r=16, 16 memory tokens, CE-only loss

### B. RMT v7 Training Complete
- **Completed:** 2026-04-17 23:51 GMT+8
- **Epochs:** 20 (full training)
- **Loss trajectory:** 3.09 → 2.11 (clean convergence)
- **Output:** `outputs/rmt_v7_8gpu_20260417_191424_20260417_191503/final/`
- **Config:** LoRA r=32, 64 memory tokens, CE-only loss, bottleneck_dim=256

### C. NIH Needle-in-Haystack Evaluation — CRITICAL FAILURE
- **Run:** 2026-04-18 00:14 GMT+8
- **Checkpoint:** RMT v7 final (despite result JSON showing "rmt_v5" in model_type field)
- **Test scope:** 90 total tests (5 depths × 3 lengths × 3 trials)
  - Depths: 10%, 30%, 50%, 70%, 90%
  - Lengths: 1024, 2048, 4096 tokens
- **Results:**
  - **Base model:** 100% accuracy (45/45 tests passed) ✓
  - **RMT v7 model:** **0% accuracy** (0/45 tests passed) ❌

---

## 2. Key Findings

### A. Training Loss Misleading
Both v6 and v7 show clean loss decrease during training:
- v6: ~2.3-2.5 plateau in later epochs
- v7: 3.09 → 2.11 over 20 epochs
- v7 val loss (2.175) < train loss (2.321) at step 500 — no overfitting

**BUT:** This loss decrease does NOT indicate that the model learned to use memory tokens. The CE-only loss measures next-token prediction accuracy on the training data, which the model can achieve simply by learning from LoRA on the backbone without ever utilizing the memory mechanism.

### B. Memory Mechanism Completely Broken
The NIH eval reveals that the RMT memory mechanism is fundamentally non-functional:
- **Base model (no memory):** Perfect retrieval (100%)
- **RMT model (with memory):** Zero retrieval (0%)

This is the opposite of what we'd expect. The memory mechanism appears to be actively harmful rather than helpful.

### C. Degenerate Generation Pattern
RMT model outputs consistently follow a degenerate pattern:
- Instead of retrieving the needle (e.g., "XAJI0Y"), the model generates text about numbers:
  - "a 10-digit number, and the first digit is 1..."
  - "a 6-digit number, and it is the sum of the numbers in..."
  - "a 10-digit number, and it is known that the first digit is 1..."

This suggests the model is not even attempting to recall from memory — it's hallucinating number properties that don't exist in the context.

### D. Possible Root Causes

1. **Memory tokens not being read during inference:** The `_embed_with_memory` function may not be properly prepending memory tokens to the question, or the attention mask may be incorrect.

2. **Memory extraction bug:** The `extract_memory` function may be producing garbage representations due to incorrect indexing, dimension mismatches, or the `.detach()` blocking gradient flow during training.

3. **Training-inference mismatch:** The memory module may have learned useful representations during training, but the inference pipeline may be using them incorrectly (wrong dimensions, wrong order, wrong masking).

4. **CE-only loss insufficient:** Without reconstruction loss or direct gradient flow to the memory module, the memory may be learning weak/useless representations that don't encode actual content.

5. **Position embedding mismatch:** RoPE positions may be incorrectly assigned to memory tokens during inference, causing them to be ignored or misinterpreted by attention layers.

---

## 3. Interpretation

**Critical Issue:** We have trained two RMT models (v6 and v7) with clean loss convergence, but the memory mechanism provides **zero benefit** for retrieval. In fact, it's actively harmful (0% vs 100% baseline).

This is a fundamental blocker. We cannot proceed to v8 or any more advanced features until we diagnose and fix why the memory mechanism doesn't work.

**Key Insight:** Training loss is not a reliable proxy for memory learning. A model can achieve low CE loss by learning patterns in the training data without ever using the memory mechanism. The NIH eval is the true test of whether memory is functional.

---

## 4. Recommended Next Actions

### Priority 1: Debug the Inference Pipeline (CRITICAL)
1. **Run debug_forward.py** on a single NIH test case and inspect:
   - Memory tensor shapes and values after each segment
   - Final memory tensor before question processing
   - Memory embeddings prepended to question
   - Attention mask structure
   - Position IDs for memory tokens

2. **Verify memory token prepending:** Check that `_embed_with_memory` actually prepends the memory embeddings to the question embeddings in the correct order.

3. **Check attention masking:** Verify that `build_rmt_attention_mask` allows memory tokens to attend to all text tokens and allows text tokens to attend to memory tokens.

4. **Verify position IDs:** Check that `build_rmt_position_ids` assigns correct RoPE positions to memory tokens.

### Priority 2: Inspect Memory Module Weights
1. **Examine RMT memory weights:** Load `rmt_memory.pt` and check:
   - Are weights non-zero?
   - Are they reasonable magnitudes (not exploding)?
   - Do they vary across different memory slots?

2. **Compare initial vs final weights:** If memory weights are near initialization, the module never learned anything.

### Priority 3: Minimal Reproduction Test
1. **Single-segment test:** Run RMT inference on a single segment (no segmentation needed) to isolate whether the issue is in segmentation or memory usage.

2. **Manual memory injection:** Create a test where we manually set the memory tensor to a known value and verify that the model can retrieve it.

### Priority 4: Consider Loss Design Changes
1. **Add reconstruction loss:** Train memory to reconstruct segment hidden states to provide direct supervision.

2. **Remove `.detach()` in extraction:** Allow gradients from CE loss to flow through the memory extraction path so the memory module gets direct signal from the backbone's next-token prediction.

3. **Add retrieval-specific loss:** Explicitly train the model to retrieve needles from memory.

---

## 5. Risks / Uncertainties

- **Root cause unknown:** The 0% accuracy could be due to inference bug, training bug, or fundamental design flaw. We don't know which yet.

- **GPU time already spent:** Both v6 and v7 used substantial GPU time (~8-10 hours total) on training that produced non-functional memory.

- **v8 blocked:** Until memory mechanism is debugged, we cannot proceed to v8 (memory routing, reconstruction loss, etc.).

- **Deadline pressure:** Each debugging cycle takes time and GPU resources.

---

## 6. Handoff for Main Agent

- **CRITICAL ISSUE:** RMT memory mechanism is completely non-functional (0% NIH accuracy vs 100% baseline).
- v6 and v7 training completed with clean loss curves, but loss does not indicate memory learning.
- **Priority 1:** Debug inference pipeline — run `debug_forward.py` on NIH test case and inspect memory tensors, embeddings, attention mask, and position IDs.
- **Priority 2:** Examine RMT memory weights to verify they learned anything.
- **Priority 3:** Create minimal reproduction test (single segment, manual memory injection).
- **BLOCKER:** Cannot proceed to v8 until memory mechanism is debugged and working.

---

## Version Info
- **git branch:** main
- **short commit:** 4ce1dc0
- **full commit:** 4ce1dc045ad65dc7432f61fa25a4eaabd7e5b374
- **Key files inspected:**
  - `outputs/rmt_v6_20260417_165231/train.log`
  - `outputs/rmt_v6_20260417_165231/rmt_config.json`
  - `outputs/rmt_v7_8gpu_20260417_191424_20260417_191503/train.log`
  - `outputs/rmt_v7_8gpu_20260417_191424_20260417_191503/rmt_config.json`
  - `outputs/nih_eval_v7/nih_results.json`
  - `outputs/nih_eval_v5.log`
  - `scripts/eval_nih.py`
  - `ops/research_notes/2026-04-17_2355_rmt_v7_training_complete__4ce1dc0.md`
