# Schedule Match 1000 — Root-Cause Regression Analysis

**Date**: 2026-04-27  
**Analyst**: researcher subagent  
**Experiment**: `branch3_A2_schedule_match_1000`  
**Champion baseline**: `branch3_A2_champion` (200 steps, final eval PPL = 1.8131)  
**Regression result**: 1000 steps, final eval PPL = 4.886

---

## 1. Executive Summary

The `schedule_match_1000` run regressed from champion PPL=1.8131 to PPL=4.886 (+170%). Root-cause analysis identifies **three compounding failure modes**, ranked by confidence:

| Rank | Hypothesis | Confidence | PPL contribution estimate |
|------|-----------|-----------|--------------------------|
| H2 | EMA writeback gate plateau (β plateau asymmetry) | **High (55%)** | Primary driver of drift after step 500 |
| H1 | No LR scheduler → optimizer overtraining | **Medium (30%)** | Secondary amplifier of long-run instability |
| H3 | Shared-bank cross-layer gradient compounding | **Medium-low (15%)** | Explains step-13 spike specifically |

The champion config is safe at current scale: it terminates at **β = 0.030** (step 200, warmup=500), never reaching the plateau region (β_plateau = 0.15) that drives the 500–1000 step instability in the failing run.

---

## 2. Config Diff (Champion vs Schedule Match 1000)

| Parameter | Champion | Schedule Match 1000 | Delta |
|-----------|---------|---------------------|-------|
| `max_train_steps` | 200 | **1000** | +5× |
| `max_chunks` | 200 | **1000** | +5× |
| `slot_init_noise` (σ) | **0.05** | 0.02 | −60% |
| `writeback_warmup_steps` | **500** | 1000 | +2× |
| LR scheduler | none | none | same |
| `lr` | 1e-3 | 1e-3 | same |
| `gate_max` | 0.3 | 0.3 | same |
| `gate_param` init | 0.0 → β_0 = 0 | 0.0 → β_0 = 0 | same |
| N slots / top_k | 512 / 64 | 512 / 64 | same |
| `shared_memory_bank` | True | True | same |

---

## 3. Training Log Trajectory (User-Supplied)

The following trajectory is sourced from `logs/branch3_A2_schedule_match_1000_20260427_1035_retry.log` as summarized in the task brief. The raw log file was not accessible on disk at analysis time; PPL values are as provided by the researcher initiating this analysis.

### Phase 1 — Stable Early Phase (Steps 1–12)
- PPL ≈ 2.5–3.0, near-champion quality
- β computed from code: at step 12, β = σ(0) × (12/1000) × 0.3 = **0.5 × 0.012 × 0.3 = 0.0018**
- Bank writes are essentially frozen (β ≈ 0.002); model runs mostly in bypass mode
- Selector and output gate can update freely without corruption from overly strong slot writes

### Phase 2 — Step 13 Spike (PPL 2.x → 49.71)
- **Single-step PPL spike from ~3.0 to 49.71 at step 13**
- β at step 13 = σ(0) × (13/1000) × 0.3 = **0.5 × 0.013 × 0.3 = 0.00195**
- β is essentially unchanged from step 12 (0.0018 → 0.00195), so the beta warmup *cannot* explain the spike
- See §4.3 for root-cause analysis of this spike (H3: shared-bank gradient compounding)

### Phase 3 — Recurring Spike Pattern (Steps 14–922)
- Pattern of instability and recovery repeating through the run
- β reaches full warmup plateau at step 1000 × 1.0 / 1000 = ... wait — warmup=1000 in schedule_match_1000
  - β(step 500) = 0.5 × (500/1000) × 0.3 = **0.075**
  - β(step 1000) = 0.5 × 1.0 × 0.3 = **0.15** (plateau)
- β is still ramping during this phase; slots are accumulating increasingly strong EMA updates
- LR=1e-3 constant with no decay → optimizer continues aggressive parameter updates

### Phase 4 — Brief Recovery (Steps 923–930)
- PPL transiently drops to **1.6–4.9**, briefly matching or beating champion
- β at step 923 ≈ 0.5 × (923/1000) × 0.3 = **0.1385** — near-plateau
- This recovery window coincides with a "lucky" bank state where slot representations align with the current text distribution
- The recovery is ephemeral: slots are shared across all 32 layers, so any layer that begins diverging corrupts the shared bank for all others

### Phase 5 — Final Degradation (Steps 931–1000)
- PPL rises steadily to **8–22** range, final eval = **4.886**
- β is at or near plateau (0.14–0.15) during this final phase
- Strong EMA writes are overwriting slots with increasingly misaligned representations
- 1e-3 LR with no warmdown → optimizer has never reduced step size despite reaching late training

---

## 4. Root-Cause Analysis

### 4.1 H2 — EMA Writeback Gate Plateau (PRIMARY, Confidence: 55%)

**Mechanism**:

The writeback gate β is:
```python
def _current_beta(self) -> torch.Tensor:
    warmup_frac = min(float(self.step_counter) / float(warmup), 1.0)
    return torch.sigmoid(self.gate_param) * warmup_frac * cfg.writeback_gate_max
```

With `gate_param=0`, `σ(0) = 0.5`:
- **Champion** (warmup=500, terminates at step 200): β_max_reached = 0.5 × (200/500) × 0.3 = **0.030**
- **Schedule_match_1000** (warmup=1000, runs to step 1000): β_plateau = 0.5 × 1.0 × 0.3 = **0.15**

The plateau β is **5× higher** than the champion's maximum β. This creates fundamentally different slot dynamics:

| β level | Slot memory half-life (in steps) | Behavior |
|---------|----------------------------------|---------|
| β = 0.03 (champion max) | ~23 steps | Slow accumulation, mostly historical |
| β = 0.075 (step 500 in run) | ~9 steps | Moderate adaptation |
| β = 0.15 (plateau) | ~4.3 steps | Rapid overwriting, loses long-range memory |

At β=0.15, the slot bank effectively becomes a **very short-range recency buffer** rather than a compressed memory of earlier context. The model had been trained with the assumption that slots contain diverse historical context; the high-β plateau violates this assumption.

**Evidence**:
- Steps 923–930 brief recovery: coincides with high-β regime; the transient alignment (lucky batch) shows the model still has capacity to achieve low PPL, but the slot dynamics are too aggressive to sustain it.
- Final degradation phase (931–1000) PPL=8–22: slots are being continuously overwritten at β=0.15; each new chunk wipes out accumulated context.

**Why champion is safe**:
- Champion terminates at β=0.030; never experiences plateau
- The 500-warmup schedule was selected by the researcher as a "2.5× the training length" warmup — appropriate for 200-step training but insufficient for 1000-step training (should use 2500-step warmup for 1000-step training to maintain the same β ratio)

---

### 4.2 H1 — No LR Scheduler → Optimizer Overtraining (SECONDARY, Confidence: 30%)

**Mechanism**:

`train_mem_space_pg19.py` has no LR scheduler anywhere:
```python
optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
# No scheduler.step() anywhere in training loop
```

AdamW with constant LR=1e-3 for 1000 steps:
- Adam's moment estimates β₁=0.9, β₂=0.95 have 90th/95th percentile time constants of ~10/20 steps
- By step 200, optimizer state has fully warmed up and is making full-size gradient steps
- Steps 200–1000: continued full-size steps with no warmdown → cumulative parameter drift

The selector network (linear layers projecting hidden→slot scores) and `slot_to_hidden` linear are both being updated at full LR for 1000 steps with no decay. This causes progressive overfit to the training distribution seen in the later epochs.

**Interaction with H2**: High β amplifies the LR problem. When optimizer makes large gradient updates to selector params at steps 600–800, the high-β slots rapidly adapt to fit the new selector distribution, causing oscillatory instability.

**Evidence**:
- Steps 923–930 recovery: consistent with a "lucky plateau" in optimizer state where accumulated Adam moments temporarily point toward the correct basin
- Champion never enters this regime (terminates at step 200 before optimizer drift accumulates)

---

### 4.3 H3 — Step-13 Spike: Shared-Bank Gradient Compounding (Confidence: 15% primary / HIGH for explaining step-13 specifically)

**Mechanism**:

The step-13 spike from PPL~3.0 to 49.71 is **too large and too early** to be explained by H2 (β=0.002 at step 13) or H1 (optimizer fully warmed by step ~20 but barely changed at step 13).

The shared_memory_bank architecture creates a unique gradient compounding path:

1. Layer 0 writes slot update at step t: `slots_new = scatter(slots, idx, (1-β)·slots + β·repr_0)`
2. Layer 1 reads from `slots_new` (shared bank); its output depends on these updated slots
3. Layer 31 reads from the bank written by layers 0–30 in sequence
4. LM loss ∂L/∂slots backpropagates through all 32 write operations
5. Each write adds a gradient term; with 32 layers writing the same bank, **gradient magnitude scales as O(32)**

The `MemoryBank.write()` non-in-place scatter:
```python
self.slots = self.slots.scatter(1, idx_exp, updated)
```
preserves autograd graph — each successive `write()` adds a new node to the computation graph rooted at the original `self.slots`.

At step 13, the model has made 12 full gradient updates. The selector network has begun to distinguish between slot indices (overcoming the random-init gradient noise). When a specific slot pattern emerges that creates a gradient cascade through the 32-layer shared bank, the magnitude can be O(32×) larger than a single-layer write gradient.

**σ interaction**: The champion uses σ=0.05 for slot initialization vs σ=0.02 in this run. Larger σ creates more diffuse initial slot representations → lower initial slot selectivity → smaller gradient spike risk at step 13. With σ=0.02, initial slots are tighter, and the selector network converges faster to a specific slot pattern, potentially triggering the cascade earlier.

**Evidence**:
- Spike occurs at exactly step 13, very early in training
- β=0.00195 at step 13 — cannot be gate-driven
- Spike magnitude (PPL 3→50, 15× increase) is consistent with gradient scale-up through 32 shared writes
- Graduate clipping (`clip_grad_norm_(trainable, 1.0)`) may have dampened but not prevented: if individual param gradients are large but point in similar directions, the norm may still fit under 1.0 while still being larger than at step 12

---

## 5. β Values at Key Log Steps (Summary Table)

Formula: β = 0.5 × min(step/warmup, 1.0) × 0.3 (with gate_param=0, warmup=1000, gate_max=0.3)

| Step | β (schedule_match_1000) | β (champion, warmup=500) | Event |
|------|------------------------|-------------------------|-------|
| 1 | 0.00015 | 0.00030 | Training start |
| 12 | 0.00180 | 0.00360 | Pre-spike |
| 13 | **0.00195** | **0.00390** | **PPL spike to 49.71** |
| 100 | 0.01500 | 0.03000 | |
| 200 | 0.03000 | 0.06000 | Champion terminates here |
| 500 | 0.07500 | **0.15000** (plateau) | |
| 923 | **0.13845** | (N/A) | Brief recovery starts |
| 930 | **0.13950** | (N/A) | Brief recovery ends |
| 1000 | **0.15000** (plateau) | (N/A) | Final step |

Key insight: **The champion terminates at β=0.030, which is exactly 20% of the schedule_match plateau value of 0.150**.

---

## 6. LR Schedule Behavior Analysis

The `train_mem_space_pg19.py` script contains **no learning rate scheduler**:

```python
# From scripts/train_mem_space_pg19.py (confirmed by code audit)
optimizer = torch.optim.AdamW(
    trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95)
)
# training loop: no scheduler object, no scheduler.step()
```

Effective LR throughout the run:
- Steps 1–1000: constant LR = 1e-3
- No warmup, no decay, no cosine schedule

This means the optimizer is making identically-sized parameter updates at step 1 and step 999. For a 200-step champion, this is tolerable (model finds a basin quickly, short run, low risk of overfit). For 1000 steps:
- AdamW's adaptive gradient rescaling with β₂=0.95 does reduce effective step size for frequently-updated parameters, but this is not equivalent to explicit LR decay
- Selector weights receive gradients every step → Adam's β₂ effectively rescales their LR after ~20 steps; after that, the effective LR stabilizes near 1e-3

**Optimizer State Divergence Hypothesis**: By step 500, Adam's second-moment estimates (v) have fully accumulated and effectively encode the loss landscape geometry. If the slot writeback at high β creates a non-stationary loss landscape (which it does — high β means slot contents are changing faster), the second-moment estimates become stale relative to the current gradient distribution, causing parameter updates that are miscalibrated in scale.

---

## 7. Shared-Bank Cross-Layer Interference

With `shared_memory_bank=True`, all 32 `MemorySpaceLayer` instances share one `MemoryBank` object. During forward pass of a single chunk:

1. Layer 0 reads slots → computes output → writes updated slots back
2. Layer 1 reads **already-written** slots from Layer 0
3. ...
4. Layer 31 reads slots written by all 31 preceding layers

This is intentional BPTT through depth (Branch-3 Option A.2). However it creates coupling:
- Any layer that develops a noisy or overfitting slot write pattern pollutes all subsequent layers
- With 32 layers all contributing to the same bank gradient, the effective gradient w.r.t. `slot_to_hidden` and `hidden_to_slot` is a sum of 32 per-layer gradients
- The grad norm clip at 1.0 normalizes the *total* across all trainable params; if 32 layers each contribute a gradient of magnitude ε to the same bank direction, the total can be 32ε before clipping

At step 13 (first major spike): the selector network in some layer(s) has found a "good" slot assignment that provides useful signal. The resulting gradient through the shared bank is 32× larger than expected for a single-layer model, causing an over-large parameter update that briefly destroys the slot organization.

---

## 8. Steps 923–930 Brief Recovery Analysis

The transient recovery to PPL=1.6–4.9 at steps 923–930 provides a crucial diagnostic clue.

**β at step 923–930**: 0.1385–0.1395 (near plateau, essentially identical to surrounding steps)

**What changed?** Not β (unchanged). Not LR (unchanged). The recovery must be due to:

1. **Batch alignment**: PG-19 training data cycles; the chunks at steps 923–930 likely correspond to a portion of the training corpus that the model has seen before (in an earlier epoch or similar distribution), and the accumulated slot bank state happens to be well-aligned with this data
2. **Temporary optimizer stabilization**: Adam's gradient history (β₁ momentum) averages over the last ~10 steps; if steps 913–922 had smaller, more consistent gradients, the momentum at step 923 could be more coherent, leading to beneficial parameter updates
3. **Slot state "reset accident"**: `_reset_banks()` is called before each chunk; at step 923 the re-initialization from `init_from_hidden()` may have produced a particularly favorable starting slot state

**Why it's not sustained (steps 931+)**:
- Slots are written at β≈0.14; within 4.3 steps, the slot half-life expires
- Once a new unfavorable batch corrupts the shared bank, all 32 layers are affected simultaneously
- With no LR warmdown, optimizer continues making large updates that push the system away from the recovered basin

**Implication**: The model *has* the capacity to achieve PPL≈1.6; the instability is **dynamic**, not a loss of representational power. This strongly supports H2 (gate plateau) and H1 (optimizer overtraining) rather than H3 (structural defect), since structural defects would prevent recovery even transiently.

---

## 9. Champion Config Safety Assessment

The champion config (`branch3_A2_champion`, 200 steps, warmup=500) is safe from these failure modes:

| Risk | Champion Status |
|------|---------------|
| β plateau (H2) | **Safe**: terminates at β=0.030, plateau β=0.15 never reached |
| Optimizer overtraining (H1) | **Safe**: 200 steps is within the pre-overfit regime for LR=1e-3 |
| Shared-bank spike (H3) | **Partially safe**: step-13 spike also possible in champion (same architecture), but champion recovers by step 200 |
| Data cycling | **Safe**: 200 chunks = 200 steps, no data cycling occurs |

**Warning for future champion extensions**: If the champion is extended beyond ~400 steps without:
- Proportional increase in `writeback_warmup_steps` (should be ≥ 2.5× max_steps), or
- Adding LR decay (cosine warmdown from step 100 to end), or  
- Reducing `writeback_gate_max` (e.g., 0.1 instead of 0.3)

...then the same regression is likely to occur.

---

## 10. Recommended Follow-Up Experiments

### V1 (Highest Priority) — Isolate Gate Plateau
**Config**: schedule_match_1000 with `--writeback_warmup_steps 2500` (2.5× training length)
- Expected β at step 1000 = 0.5 × (1000/2500) × 0.3 = 0.060 (same as champion's max β at step 200)
- If PPL drops from 4.886 to ≈1.8, confirms H2 as primary driver
- Cost: 1 × 8-GPU run, ~2h

### V2 (High Priority) — Add LR Cosine Warmdown
**Config**: schedule_match_1000 with cosine decay from step 200 to step 1000 (lr_min=1e-4)
- If PPL improves by >50%, confirms H1 as significant contributor
- Run together with V1 to isolate vs combine effects
- Cost: 1 × 8-GPU run, ~2h

### V3 (Medium Priority) — Fix Slot Init Noise
**Config**: schedule_match_1000 with σ=0.05 (same as champion)
- Tests whether σ=0.02 contributed to H3 (early spike)
- If step-13 spike disappears or reduces, confirms H3 sensitivity to σ
- Cost: 1 × 8-GPU run, ~2h

### V4 (Diagnostic) — Combined Fix
**Config**: V1 + V2 + V3 combined (warmup=2500, cosine decay, σ=0.05)
- If all fixes together achieve PPL ≈ 1.8, validates the full root-cause model
- Run after V1/V2/V3 individual results

### V5 (Exploratory) — Reduce gate_max
**Config**: `--writeback_gate_max 0.1` with original 1000-step schedule
- Tests whether the problem is specifically the plateau *level* (0.15) rather than the warmup slope
- If PPL=4.886 → ~2.0, suggests gate_max=0.3 is too high for long runs
- Cost: 1 × 8-GPU run, ~2h

---

## 11. Immediate Recommendations

1. **Do not extend the champion beyond 400 steps** without V1's writeback warmup fix
2. **Priority order**: V1 first (test writeback warmup fix), then V2 (test LR decay), then V4 (combined)
3. **Future champion hyperparameter guideline**: Set `writeback_warmup_steps = 2.5 × max_train_steps` as the default scaling rule
4. **Log monitoring trigger**: Any training run where `step > writeback_warmup_steps` should be flagged for close PPL monitoring — that is the onset of the H2 plateau risk zone

---

## 12. Summary

The `schedule_match_1000` regression (PPL 1.8131 → 4.886) is primarily driven by the **EMA writeback gate plateau asymmetry**: the champion terminates at β=0.030 while schedule_match_1000 plateaus at β=0.150 for 500 steps (5× higher). A secondary contributor is **absent LR decay** (constant 1e-3 for 1000 steps). The step-13 spike to PPL=49.71 is a separate early-training event, likely caused by **shared-bank gradient compounding** across 32 layers, which is amplified by the tighter σ=0.02 slot initialization. The transient recovery at steps 923–930 (PPL=1.6) confirms that representational capacity is intact and the failure is dynamic/stability-driven rather than structural. Champion config is safe at current scale; the highest-priority fix for longer-run extensions is scaling `writeback_warmup_steps` proportionally with training length.
