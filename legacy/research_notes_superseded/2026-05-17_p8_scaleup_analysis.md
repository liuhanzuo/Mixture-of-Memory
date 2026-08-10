# P8 Scale-Up Analysis: Will More Steps Improve BABILong Performance?

**Date**: 2026-05-17
**Confidence**: high (on the negative prediction); medium (on exact degradation trajectory)

## 1. Executive Summary

**Verdict: Scaling P8 beyond 500 steps is very likely to DEGRADE performance, not improve it.**

| Scenario | Predicted 21-cell mean | Confidence |
|----------|----------------------|------------|
| P8 @ 500 steps (current) | 59.29 (measured) | — |
| P8 @ 1000 steps | 50-58 | medium |
| P8 @ 2000 steps | 35-50 | medium |
| P8 @ 5000 steps | 25-35 (P11-like degradation) | high |

The core problem: **memory retrieval (L1 inject) never activates** — `top1_sim_mean` stays at ~0.002 (= 1/512, i.e., uniform/random) across the entire 500 steps and would remain so indefinitely. Meanwhile, the beta writeback gate keeps growing and saturates at 0.15 around step 1000. More training = more random memory content being injected into the backbone = progressive corruption of the LM's output distribution. This is exactly what happened to P11.

---

## 2. P8 Training Curve Analysis (phase8_probe_20260515_2237.log)

### 2.1 Loss Trajectory

| Step | lm_loss | aux | Notes |
|------|---------|-----|-------|
| 50 | 0.0026 | 21.858 | BABILong SFT near-converged |
| 100 | -0.0003 | 21.751 | Fully converged |
| 200 | 0.1087 | 21.594 | — |
| 300 | 0.0125 | 21.591 | aux plateaued |
| 400 | 0.3442 | 21.547 | — |
| 500 | 0.6464 | 21.621 | PG-19 sample (higher is normal) |

**Key observations:**
- **lm_loss converges by step 50-100** — the model already answers BABILong QA tasks correctly very early
- **aux loss plateaus after step ~200** at ~21.5-21.6 and does not improve further
- Both metrics are essentially flat for the last 300 steps

### 2.2 QUERY_DIAG (L1 Retrieval)

| Step | top1_sim_mean | key_max_cos | per_tok_logit_std |
|------|--------------|-------------|-------------------|
| 25 | 0.002228 | 0.9727 | 0.0825 |
| 99 | 0.002197 | 0.9609 | 0.0830 |
| 196 | 0.002167 | 0.9609 | 0.0850 |
| 342 | 0.002197 | 0.9375 | 0.0854 |
| 484 | 0.002197 | 0.9727 | 0.0850 |

**`top1_sim_mean` is CONSTANT at ~0.002 (= 1/num_slots = 1/512) throughout all 500 steps.**

This means: the query-key cosine similarity after softmax is uniform — memory retrieval is selecting slots essentially at random. L1 inject is NOT providing useful information. It was this way at step 1 and still this way at step 500.

### 2.3 WRITEBACK_DIAG (Memory Update)

| Step | gate_val(beta) | alpha(tanh) | slot_delta_abs_mean |
|------|---------------|-------------|---------------------|
| 25 | 0.003754 | 0.462891 | 0.006225 |
| 99 | 0.014893 | 0.462891 | 0.005412 |
| 196 | 0.029419 | 0.462891 | 0.006144 |
| 342 | 0.051270 | 0.462891 | 0.006834 |
| 484 | 0.072754 | 0.462891 | 0.007457 |

**Key findings:**
- **beta gate** grows linearly: 0.0038 → 0.073 over 500 steps. At this rate, it would hit the `writeback_gate_max=0.3` ceiling around step ~2000, but with warmup-based clipping (warmup=1000 steps) it saturates at **0.15 by step ~1000** (exactly as observed in P11)
- **alpha (tanh output gate)** is completely FROZEN at 0.4629 — never changes
- **slot_delta** is tiny (~0.005) and not meaningfully trending

---

## 3. P11 Training Curve Analysis (p11_fsdp_full_20260516_181417.log)

### 3.1 Loss at Key Checkpoints

| Step | lm_loss | aux |
|------|---------|-----|
| 500 | 0.7651 | 21.616 |
| 1000 | 0.0456 | 21.405 |
| 1500 | 0.3929 | 21.318 |
| 2000 | 0.0343 | 21.305 |
| 2500 | 0.0666 | 21.426 |
| 3000 | -0.0024 | 21.372 |
| 4000 | 0.0066 | 21.276 |
| 5000 | 0.0104 | 21.401 |

**aux loss barely moves after step 1000** (21.4 → 21.3 → 21.4 over 4000 more steps). The training loss is near-zero throughout — the model memorizes training examples perfectly but this does NOT help generalization.

### 3.2 Beta Gate Saturation

- Steps 25-993: beta grows linearly from 0.0038 to 0.148
- **Step 1018: beta saturates at 0.150391** (the clipped maximum given warmup schedule)
- Steps 1018-5000: beta stays locked at 0.150391 — **4000 steps of training with a saturated gate**

### 3.3 top1_sim_mean — NEVER moves

| Step | top1_sim_mean |
|------|--------------|
| 25 | 0.002182 |
| 993 | 0.002075 |
| 4500 | 0.002182 |
| 4977 | 0.002029 |

Constant at ~0.002 across all 5000 steps. **Memory retrieval never becomes discriminative.**

---

## 4. P8 vs P11: Same-Step Comparison (Critical Section)

### 4.1 Training-side metrics at step ~500 are IDENTICAL

| Metric | P8 @ step 484 | P11 @ step 484 |
|--------|---------------|----------------|
| top1_sim_mean | 0.002197 | 0.002151 |
| gate_val(beta) | 0.072754 | 0.072754 |
| alpha(tanh) | 0.462891 | 0.462891 |
| slot_delta_abs | 0.007457 | 0.003957 |
| aux loss @ step 500 | 21.621 | 21.616 |

**The two runs are essentially identical in every training metric at step 500.**

### 4.2 Yet evaluation scores differ by 25+ points

| Experiment | 21-cell mean | Short avg (0-4k) | Long avg (8-32k) |
|-----------|-------------|-----------------|-----------------|
| **P8 final (500 steps)** | **59.29** | **65.7** | **50.8** |
| P11 step500 (ckpt from 5000 run) | 33.81 | 45.9 | 17.7 |
| P11 500-step validate (fresh run) | 33.71 | 45.2 | 18.4 |
| P11 temp20 (500 steps) | 35.24 | — | — |
| P11 final (5000 steps) | 26.33 | 43.5 | 3.4 |

### 4.3 Root cause of the 25-point gap

The ONLY differences between P8 and P11 configurations:

1. **P8**: DDP (standard DataParallel), no gradient checkpointing, no FSDP
2. **P11**: FSDP (Option b: mem_space sharded, backbone replicated), gradient_checkpointing=true

The P11 validate run (fresh 500-step on FSDP code) reproduces the ~34 score, confirming this is NOT about training duration — it's about the **code path used at evaluation/inference time**.

**Implication for scale-up**: The P8 baseline of 59.29 is achievable ONLY with the P8 code path (non-FSDP). If you scale P8 with the same code, the question becomes: does more training help?

---

## 5. Why P11 Degrades from Step 500 to Step 5000

| Checkpoint | 21-cell mean | Long context avg (8k-32k) |
|-----------|-------------|--------------------------|
| P11 step 500 | 33.81 | 17.7 |
| P11 step 4500* | 27.43 (user-reported) | ~5 (estimated) |
| P11 step 5000 | 26.33 | 3.4 |

**The degradation is concentrated in long-context performance** (8k/16k/32k → nearly 0%).

**Mechanism**: 
1. Beta gate saturates at 0.15 by step ~1000
2. Memory content (slots) gets increasingly written-to but retrieval NEVER improves (top1_sim stays at 1/512)
3. The model learns to "overwrite" slots with training-set-specific patterns
4. At inference time, the injected memory content is essentially noise relative to the test queries
5. For longer contexts (more chunks → more memory injection per sequence), the corruption accumulates

**This same mechanism would apply to P8 if scaled beyond 500 steps**, because:
- P8 also has top1_sim = 1/512 (retrieval is random)
- P8's beta gate would also saturate at 0.15 by step ~1000
- P8's advantage (higher eval score) comes from inference code path, not from better retrieval

---

## 6. Predicted Scale-Up Trajectory for P8

### 6.1 Quantitative Prediction

| Steps | Expected 21-cell mean | Key driver |
|-------|----------------------|------------|
| 500 | 59.29 (measured) | Baseline, beta=0.07 |
| 750 | 55-59 | Beta ~0.11, slight degradation possible |
| 1000 | 50-58 | Beta saturates at 0.15, degradation onset |
| 2000 | 35-50 | Slot overwriting, long-ctx damage begins |
| 5000 | 25-35 | Full P11-like collapse of long-context |
| 10000 | 20-30 | Possible further slow decline |

**Confidence: medium** — the exact numbers depend on whether P8's non-FSDP inference path buffers against corruption better than FSDP. The qualitative trajectory (decline after step ~1000) is **high confidence**.

### 6.2 Best case (scale IS beneficial)

This would require `top1_sim_mean` to start improving with more training. In 5000 P11 steps it never did. **Evidence for beneficial scaling: NONE.**

### 6.3 Degradation risk factors

The most likely root cause of degradation would be:
1. **Beta gate saturation + random retrieval**: Same mechanism as P11
2. **Slot content drift**: With 512 slots being slowly overwritten by training samples, slot content becomes training-specific noise at test time
3. **Catastrophic forgetting**: The backbone's LM ability may erode as memory injection strengthens (the "PPL > 100 = model polluted" phenomenon at extreme)

---

## 7. The Real Question: Why Is P8 at 59.29 While P11 step500 is at 33.81?

This is the most important finding of this analysis. **The 25-point gap is NOT explained by training duration.** It's explained by code differences:

- P8 uses the original non-FSDP training/eval code path
- P11 uses FSDP with gradient checkpointing

The P11 validate run (identical architecture, fresh 500-step run on FSDP code) gets 33.71, confirming the code path difference.

**If we want to scale P8, we should first understand what the non-FSDP code path does differently at inference time** that gives it +25 points. This is more valuable than brute-force scaling.

---

## 8. Cheapest Next Experiments (Ranked by ROI)

### Experiment 1: Evaluate existing P11 intermediate checkpoints (confidence: high)

**Cost**: ~30 min eval time (no training)
**ROI**: Very high — directly plots the step-vs-score curve

P11 has saved checkpoints at: step500, step1000, step1500, step2000, step2500, step3000, step3500, step4000, step4500.

Run the standard 21-cell BABILong eval on at least: step1000, step2000, step3000, step4500.

This gives us an empirical degradation curve without training anything new.

### Experiment 2: P8 code path @ step 1000 (confidence: medium)

**Cost**: ~25 min training (500 more steps on P8's non-FSDP path) + 30 min eval
**ROI**: High — directly answers "does P8 degrade past 500?"

Extend P8 training to 1000 steps (load step500 checkpoint, run 500 more). Evaluate at step 1000. If score drops below 55, scaling is harmful.

### Experiment 3: FSDP vs non-FSDP inference-time comparison (confidence: high)

**Cost**: ~30 min (two evals of same checkpoint)
**ROI**: High — explains the 25-point gap

Take the P8 step500 adapter checkpoint. Evaluate it:
- (a) using P8's original eval code (non-FSDP)
- (b) using P11's FSDP eval code

If the same weights give different scores, the gap is a code/inference bug, NOT a training effect. This would be the most actionable finding.

---

## 9. Risk / Uncertainty

| Item | Status |
|------|--------|
| P8 vs P11 gap explained by FSDP vs DDP | **Strong evidence** (P11 validate confirms), but not 100% — could also be subtle RNG/data-order effects |
| top1_sim never improves with more training | **Very strong evidence** (0 movement in 5000 P11 steps) |
| Beta gate causes degradation after saturation | **Medium evidence** (correlational — degradation starts ~when beta saturates, but causal mechanism is inferred) |
| Exact degradation magnitude for P8 | **Low-to-medium confidence** — P8's non-FSDP path might buffer against corruption differently |
| Whether memory retrieval CAN be fixed with more steps | **Evidence says no** — structural issue with key-query alignment, not a training-time issue |

---

## 10. Key Takeaway for Decision-Making

**Do NOT scale P8 to 2000/5000/10000 steps expecting improvement.** The expected outcome is degradation.

**Instead, prioritize:**
1. Understanding the FSDP/non-FSDP gap (Experiment 3) — this could recover 25 points for all future runs
2. Evaluating P11 intermediate checkpoints (Experiment 1) — free data, confirms degradation curve
3. Only if Experiments 1-2 show surprising results, consider a longer P8 run

The fundamental limitation is that **memory retrieval (top1_sim) never activates**. Until this is fixed, more training steps only increase the weight of random noise injection into the model's output.
