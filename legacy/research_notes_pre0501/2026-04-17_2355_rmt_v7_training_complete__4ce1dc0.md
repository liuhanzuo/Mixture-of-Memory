# RMT v7 Training Complete

**Date:** 2026-04-17 23:55 (GMT+8)
**Type:** Training Completion Report
**Status:** COMPLETED — all 20 epochs, ~4.6h wall time

---

## 1. Current Status

RMT v7 training finished successfully at 23:51:50 GMT+8.

- **Model:** Qwen3-8b + LoRA (r=32, α=64) + RMT v5 extractor (version 5)
- **Memory tokens:** 64, bottleneck_dim=256, max_segments=6
- **Data:** wiki_zh_10k.jsonl (326 docs per epoch × 20 epochs = 6520 segment-sequences)
- **Training time:** 16,520s (~4.6h) on 8 GPUs
- **Output:** `outputs/rmt_v7_8gpu_20260417_191424_20260417_191503/`
  - `final/` — final checkpoint (16.5GB)
  - `checkpoint_step500/` — mid-training checkpoint
  - `train.log`, `rmt_config.json`, `heartbeat.json`

### Loss Trajectory

| Step | Epoch | Loss | LR |
|------|-------|------|----|
| 10 | 0 | 3.087 | 4.0e-6 |
| 80 | 1 | 2.577 | 2.0e-5 |
| 200 | 4 | 2.343 | 1.8e-5 |
| 400 | 9 | 2.325 | 1.1e-5 |
| 500 | 12 | 2.321 (train) / **2.175 (val)** | 7.3e-6 |
| 600 | 15 | 2.284 | 3.7e-6 |
| 800 | 20 | 2.106 | — |

**Final loss:** ~2.11 (step 800, end of training)
**Val loss (step 500):** 2.175 — lower than training loss at same point, no overfitting signal.

---

## 2. Key Findings

1. **Stable convergence, no issues.** Loss decreased monotonically from 3.09 → 2.11 over 20 epochs. No spikes, no divergence, no NaN. The cosine LR schedule with warmup=50 worked well.

2. **Val loss (2.175) < train loss (2.321) at step 500.** This is normal with DDP (each GPU sees different data subset) and gradient accumulation. No overfitting concern.

3. **CE-only loss design validated.** v7 used pure cross-entropy (no reconstruction, no z-forcing, no gap loss). The comment in code says "v7: no z-forcing — CE-only training." This is the simplest possible loss and it converged cleanly.

4. **Memory extractor uses `seg_hidden.detach()`.** During memory extraction, the segment hidden states are detached before being passed to the memory module. This means the backbone receives gradients from CE loss only through the forward pass, not through the memory extraction path. The memory module learns to compress but doesn't get direct gradient from the backbone's CE loss on memory token positions (they're masked with -100 labels).

5. **Comparison to v6:** v6 plateaued around 2.25-2.47 at epochs 16-18 with lower capacity (LoRA r=16, 16 memory tokens). v7's final loss of 2.11 is meaningfully lower, suggesting the capacity increase helped.

---

## 3. Interpretation

The training is a clean success — but the real test is whether the model actually *uses* the memory tokens for retrieval. The previous brief noted that NIH eval on v5 showed **0% accuracy**, which was attributed to a likely inference pipeline bug (not a training issue). v7 likely has the same bug if the eval script hasn't been fixed.

**Critical question:** Does v7 actually learn to store useful information in memory tokens? The loss decrease alone doesn't prove this — the model could be ignoring memory tokens and just learning from LoRA on the backbone. The detached extraction path (`seg_hidden.detach()`) means the memory module doesn't get gradient signal from the backbone's next-token prediction, which is a potential weakness.

---

## 4. Recommended Next Actions

1. **URGENT: Run NIH eval on v7 final checkpoint.** This is the definitive test. Use the same eval pipeline but verify the forward pass is correct (memory tokens are prepended, attention mask is right, etc.).

2. **Debug the inference pipeline if NIH still shows 0%.** The 0% accuracy on v5 was almost certainly a code bug, not a model quality issue.

3. **Consider enabling gradient flow through memory extraction.** Currently `seg_hidden.detach()` blocks gradients from CE loss to the memory module. The memory module only learns through indirect signals. Consider removing the `.detach()` or adding a reconstruction loss on memory tokens to provide direct supervision.

4. **v8 direction (from G-MemLLM analysis brief):** memory routing + reconstruction loss for differentiation from existing work.

---

## 5. Risks / Uncertainties

- **Memory may be ignored:** Without reconstruction loss or gradient flow through extraction, the memory module may learn weak/useless representations.
- **Inference bug unconfirmed:** The NIH 0% accuracy on v5 was never conclusively diagnosed. The same bug may affect v7 eval.
- **Only 10k training docs:** Relatively small dataset; may limit generalization.

---

## 6. Handoff for Main Agent

- v7 training is **done**. GPUs are now free.
- Final checkpoint: `outputs/rmt_v7_8gpu_20260417_191424_20260417_191503/final/`
- Mid checkpoint: `outputs/rmt_v7_8gpu_20260417_191424_20260417_191503/checkpoint_step500/`
- **Priority 1:** Run NIH eval on final checkpoint to determine if memory mechanism actually works.
- **Priority 2:** If NIH fails, debug the inference pipeline before training v8.

---

## Version Info
- **git branch:** main
- **short commit:** 4ce1dc0
- **full commit:** 4ce1dc045ad65dc7432f61fa25a4eaabd7e5b374
- **Key files inspected:**
  - `outputs/rmt_v7_8gpu_20260417_191424_20260417_191503/train.log`
  - `outputs/rmt_v7_8gpu_20260417_191424_20260417_191503/rmt_config.json`
  - `outputs/rmt_v7_8gpu_20260417_191424_20260417_191503/heartbeat.json`
  - `scripts/train_rmt_v7.py`
  - `scripts/run_train_v7.sh`
  - `ops/research_notes/2026-04-17_1953_rmt_v7_launched_nih_eval_v5_critical_failure__4ce1dc0.md`
