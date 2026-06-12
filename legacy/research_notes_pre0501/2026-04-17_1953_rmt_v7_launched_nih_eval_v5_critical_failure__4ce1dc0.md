# RMT v7 Launched + NIH Eval v5 Critical Failure

**Date:** 2026-04-17 19:53 (GMT+8)  
**Type:** Research Pre-Research / Training Launch + Eval Alert  
**Status:** RUNNING — RMT v7 training epoch 2, ~4h remaining  

---

## 1. Current Status

- **Model:** Qwen3-8b (8B params), RMT v7 + LoRA (rank=32, alpha=64) — **rank doubled from v6's 16**
- **Other v7 config changes:** num_memory_tokens=64 (was 16), lora_dropout=0.05, rmt_lr=0.0002
- **Training epochs:** 20 (target)
- **Current position:** Epoch 2 step 110 [240/326], ~10% complete
- **Training status:** **RUNNING** (all 8 GPUs active)
- **ETA:** ~4.0h remaining
- **Started:** 2026-04-17 19:14 GMT+8
- **Output dir:** `outputs/rmt_v7_8gpu_20260417_191424_20260417_191503`

### RMT v7 Early Training Metrics

| Epoch | Step | Loss | LR |
|-------|------|------|----|
| 0 | 10 | 3.0868 | 4.00e-06 |
| 0 | 40 | 2.9275 | 1.60e-05 |
| 1 | 50 | 2.7394 | 2.00e-05 |
| 1 | 80 | 2.5766 | 1.99e-05 |
| 2 | 90 | 2.5233 | 1.99e-05 |
| 2 | 110 | 2.3339 | 1.97e-05 |

Loss dropping fast in early epochs — promising start compared to v6's initial trajectory.

---

## 2. Relevant New Findings

### A. RMT v7 Launched (Material Change)
- LoRA rank **16 → 32** (key recommendation from previous brief)
- Memory tokens **16 → 64** (4x increase)
- Training started after v6 epoch 18 completed (v6 was at loss plateau ~2.25-2.47)
- v7 starting loss (3.09) is comparable to v6's initial loss, suggesting fresh training

### B. NIH Eval v5: **CRITICAL — 0% Accuracy**
- Eval completed earlier today (10:39 GMT+8) using RMT v5 checkpoint
- **Base model:** 100% accuracy across all depths (0.1-0.9) and lengths (1024-4096)
- **RMT v5 model:** **0% accuracy** — complete retrieval failure
- RMT model produces nonsensical outputs (e.g., "a 10-digit number") instead of retrieving needles
- This indicates the RMT mechanism is **fundamentally broken** for retrieval tasks
- The model is not even attempting to recall the needle — it generates random number patterns

### C. Debug Forward + NIH Eval Scripts Completed
- Scripts `debug_forward.py` and `eval_nih.py` ran successfully (exec sharp-em, code 0)
- These appear to be the same eval run that produced the nih_eval_v5 results

---

## 3. Interpretation

**Critical Issue:** The NIH eval reveals that RMT (v5 at least) completely fails at needle-in-a-haystack retrieval. The model doesn't just miss the needle — it generates entirely irrelevant content. Possible causes:

1. **RMT memory tokens not being properly read** during inference — the forward pass may not be using the memory mechanism correctly
2. **Training-inference mismatch** — the eval script may not be loading/applying RMT the same way as training
3. **Segmentation bug** — the input may not be properly segmented, causing memory to never be populated
4. **The debug_forward.py script was likely run to diagnose this exact issue**

**Positive:** v7's higher capacity (LoRA rank=32, 64 memory tokens) may help, but if the fundamental RMT mechanism is broken, capacity alone won't fix it.

---

## 4. Recommended Next Actions

1. **URGENT: Review debug_forward.py output** — understand what the forward pass actually does
2. **Verify RMT inference pipeline** — ensure memory tokens are read/written correctly during eval
3. **Check if v5 eval used correct checkpoint/config** — may have been a config mismatch
4. **Re-evaluate NIH after v7 training completes** — but fix inference pipeline first
5. **Consider minimal reproduction test** — single-segment forward pass to verify memory mechanism works
6. **Let v7 training continue** — no reason to stop it, ~4h ETA

---

## 5. Risks / Uncertainties

- **Root cause unknown:** 0% accuracy is extreme — likely a code bug rather than model quality issue
- **v7 may have same bug:** If the eval pipeline is broken, v7 will also show 0% accuracy
- **Training-inference mismatch risk:** Debug script should reveal this
- **GPU utilization:** 8 GPUs still occupied by v7 training

---

## 6. Handoff for Main Agent

- **Priority 1:** Debug the NIH eval pipeline — the 0% accuracy needs explanation before v7 eval
- v7 training is autonomous, ETA ~4h (completion ~23:53 GMT+8)
- Check `scripts/debug_forward.py` output for diagnostic clues
- The eval failure pattern (nonsensical "number" outputs) suggests the model is not receiving/using memory tokens at all during inference

---

## Version Info
- **git branch:** main
- **short commit:** 4ce1dc0
- **full commit:** 4ce1dc045ad65dc7432f61fa25a4eaabd7e5b374
- **Key files inspected:**
  - `outputs/rmt_v7_8gpu_20260417_191424_20260417_191503/train.log`
  - `outputs/rmt_v7_8gpu_20260417_191424_20260417_191503/rmt_config.json`
  - `outputs/nih_eval_v5/nih_results.json`
  - `outputs/v7_launch.log`
  - `ops/research_notes/2026-04-17_1851_training_progress_epochs16_18_complete__4ce1dc0.md`
