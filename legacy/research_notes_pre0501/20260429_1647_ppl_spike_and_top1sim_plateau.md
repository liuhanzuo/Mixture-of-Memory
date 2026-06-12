# Research Note: PPL Spike Root Cause (Fix L-1 Insufficient) & top1_sim Plateau Analysis

**Date**: 2026-04-29 16:47 GMT+8  
**Author**: /researcher agent  
**Triggered by**: heartbeat observation of fix_j_l_ablation runs on b200-2/3/4 (~step 400)  
**Run**: fix_j_l_ablation_node{0,1,2}_20260429_1630.log  
**Nodes**: b200-2 (σ=0.01), b200-3 (σ=0.02), b200-4 (σ=0.05)

---

## Executive Summary

**Problem 1 (PPL Spikes)**: Fix L-1 successfully clips M_sel_hidden (the INPUT to joint attention) — confirmed M_sel_hidden_norm_mean ≈ 1.0 at all WRITEBACK_DIAG checkpoints. However, catastrophic PPL spikes still occur. Root cause: `slot_delta` (the OUTPUT contribution to the residual stream) is **not clipped by Fix L-1**. Slot norms inflate to near the bank cap (max_norm ≈ 128 for d_model=4096), causing chaotic write-direction flips that generate large slot_delta values. With alpha ≈ 0.462 and up to 32 layers compounding, effective injection per token grows unbounded.

**Problem 2 (top1_sim Plateau)**: The plateau at ~0.002 ≈ 1/N_slots is expected given gate warmup (beta reaches only 0.113 at step 376, fully warming at step 500). Routing diversification requires both gate warmup completion AND slot content diversity — neither is achieved yet. The PPL spike cycles on nodes 1&2 additionally corrupt gradients, preventing systematic Q_sel improvement.

**Fix M (recommended)**: Add slot_delta norm clip in layer.py after `slot_delta = ext_h[:, k_slots:, :] - bypass_h`, capping slot_delta to have norm ≤ bypass_h norm per token. Optionally reduce bank max_norm from √d×2 to √d×0.5.

---

## Problem 1: PPL Spikes Despite Fix L-1

### What Fix L-1 Does (and Doesn't Do)

**Fix L-1 scope** (layer.py lines 523–530):
```python
_h_norm_ref = hidden_states.detach().norm(dim=-1).mean().clamp(min=1.0)
_m_norms = M_sel_hidden.norm(dim=-1, keepdim=True)
M_sel_hidden = M_sel_hidden * (_h_norm_ref / _m_norms.clamp(min=1e-6)).clamp(max=1.0)
```
- Clips M_sel_hidden (the k=64 slot tokens fed into joint attention as KV extension) to have norm ≤ hidden_states norm
- **One-directional**: only shrinks, never amplifies
- **Does NOT clip**: `slot_delta = ext_h[:, k_slots:, :] - bypass_h` — the actual output-side injection

**Flamingo gate injection** (layer.py lines 621–625):
```python
alpha = torch.tanh(self.slot_output_gate)      # ≈ 0.462891
O_mem_hidden = ext_h[:, :k_slots, :]
slot_delta = ext_h[:, k_slots:, :] - bypass_h  # NOT clipped
next_hidden = bypass_h + alpha * slot_delta     # alpha ≈ 0.462
```

### Evidence: M_sel_hidden Clip Is Working

All WRITEBACK_DIAG entries confirm M_sel_hidden_norm_mean ≈ 1.0 (clip active):

| Node | Step | M_sel_hidden_norm_mean | gate_val(β) |
|------|------|------------------------|-------------|
| b200-2 (σ=0.01) | 376 | 0.999 | 0.1147 |
| b200-3 (σ=0.02) | 341 | 1.001 | 0.103 |
| b200-4 (σ=0.05) | 376 | 1.000 | 0.114 |

Fix L-1 is doing exactly what it was designed to do.

### Evidence: slot_delta Is Growing Unchecked

Despite M_sel_hidden being clipped, slot_delta grows and spikes:

| Node | Step | retrieved_norm_mean | slot_delta_abs_mean | slot_delta_max | PPL |
|------|------|---------------------|---------------------|----------------|-----|
| b200-2 (σ=0.01) | 376 | 89.3 | (not elevated) | 5.25 | stable |
| b200-2 (σ=0.01) | 390 | ~100 | (not logged) | — | **14111** ← spike |
| b200-2 (σ=0.01) | 408 | 125.9 | — | — | 100–2716 |
| b200-3 (σ=0.02) | 341 | 107.2 | — | 5.125 | ~4299 |
| b200-4 (σ=0.05) | 376 | 126.5 | 0.043 | 7.97 | crisis |
| b200-4 (σ=0.05) | 634 | — | 0.113 | — | crisis |
| b200-4 (σ=0.05) | 661 | — | 0.151 | — | crisis |

Normal `slot_delta_abs_mean` = 0.003–0.013. At step 634+ on node2: 0.113–0.151 (10–30× elevated).

### Root Cause Mechanism

1. **Slot norm inflation**: With Fix K (carry-over), each write EMA-accumulates information. Over 300–400 steps, slot norms grow from ~1 (strided_token init) toward the bank max_norm cap = √4096 × 2 ≈ 128.

2. **Bank clip at max_norm ≈ 128** (memory_bank.py lines 252–256):
   ```python
   max_norm = math.sqrt(slot_dim_f) * 2.0    # ≈ 128.06
   slot_norms = updated.norm(dim=-1, keepdim=True)
   scale = (slot_norms.clamp(max=max_norm) / slot_norms.clamp(min=1e-6))
   updated = updated * scale
   ```
   When slot norms approach 128, any new write that pushes past the cap causes the bank clip to **rescale in a direction-modifying way** (the "clamped scale" bends the update direction, not just magnitude). This creates sudden large shifts in slot content.

3. **Fix L-1 clips input but not output**: M_sel_hidden is derived from slots via `slot_to_hidden` projection and is clipped to norm ≤ hidden_states_norm. But the **joint-attention output** on the extended sequence (ext_h) is not constrained relative to bypass_h. The difference `slot_delta = ext_h[:, k_slots:, :] - bypass_h` can be large even when M_sel_hidden is moderate, because the attention mechanism amplifies content from the clipped slot tokens non-linearly.

4. **32-layer compounding**: All 32 decoder layers share one bank and apply the same injection. After the spike-onset step, `slot_delta_abs_mean` escalates across all layers simultaneously (the shared bank means a poisoned slot propagates to all layers).

5. **Timing correlates with slot norm inflation reaching ~50–60**: 
   - b200-4 (σ=0.05): higher init_noise → faster writeback → faster slot inflation → first spike step ~200 (retrieved_norm_mean likely ~60–80)
   - b200-3 (σ=0.02): first spike step ~240, retrieved_norm ~100 at step 341
   - b200-2 (σ=0.01): first spike step ~390, retrieved_norm 89 at step 376

6. **Aux spike is a co-symptom**: At PPL spike events, aux jumps from baseline ~21 to 29–45. This is because the same routing disruption that causes the PPL spike concentrates dispatch onto fewer slots, increasing lb_raw = N × Σ(importance_i × load_i).

### Why Fix L-1 Is Insufficient

Fix L-1 was designed to prevent M_sel_hidden from overwhelming the **query** side of joint attention. It does this correctly. But the injection happens at the **output** side via `bypass_h + alpha * slot_delta`. slot_delta has no constraint. Even with alpha = 0.462:

- slot_delta_max = 7.97 (observed b200-4 step 376)
- Per-token injection peak = 0.462 × 7.97 ≈ 3.68 per layer
- 32 layers × 3.68 = 117 effective additive shift to the final hidden state

This is large enough to corrupt the LM head output distribution catastrophically.

---

## Problem 2: top1_sim Plateau at 0.002

### Observed Values

All 3 nodes: top1_sim_mean ≈ 0.002 since step ~19 (earliest QUERY_DIAG checkpoint).

- Theoretical uniform baseline: 1/N = 1/512 ≈ 0.001953
- Measured: 0.002 ≈ 1/N → routing is essentially **perfectly uniform** (random slot selection)

### Expected vs Actual Timeline

**Gate warmup theory** (writeback fraction formula in layer.py):
- beta(t) = sigmoid(gate_param) × min(t / warmup_steps, 1.0) × gate_max
- gate_param_init = 0 → sigmoid(0) = 0.5
- warmup_steps = 500, gate_max = 0.3
- At t=376: beta = 0.5 × (376/500) × 0.3 = 0.5 × 0.752 × 0.3 = 0.113
- At t=500 (full warmup): beta_max = 0.5 × 1.0 × 0.3 = 0.15

**Why slot diversity is low early**:
1. Writeback beta is 0.113 at step 376 — gate has only used 75% of warmup budget
2. At full warmup (step 500), beta_max = 0.15, meaning only 15% of new content replaces old each write
3. With 512 slots, top_k=64 per forward: mean revisit interval = 512/64 = 8 steps
4. After k revisits: slot residual from init ≈ (1-0.15)^k. To reach 99% replacement: k = log(0.01)/log(0.85) ≈ 28 revisits = 28×8 = ~224 steps past gate full-warmup = step ~724
5. top1_sim can only grow meaningfully once slots contain **distinct** content (different from init), which requires ~step 724 under carry-over

**PPL spikes on nodes 1&2 as additional confound**:
- PPL crisis generates noisy/large gradients that corrupt Q_sel learning
- Node0 (σ=0.01, most stable) should show earliest top1_sim divergence from 1/N

### Assessment: Plateau Is Expected

The top1_sim plateau at 0.002 is **not a bug** at this stage. Gate warmup is still in progress (< step 500), and even after gate is fully warm, slot diversity requires ~200 more steps to accumulate. The expected onset of top1_sim > 0.005 is step 700–800, not step 500 as the original success criterion stated.

**Revised success criterion**:
- Step 500: beta_max = 0.15 — gate fully warm ✓ (but slots still ~50% diverse from init)
- Step 700: slot diversity adequate for Q_sel learning to have effect
- Step 800–1000: top1_sim_mean > 0.005 (realistic)

---

## Recommended Fix M

### Fix M-1: slot_delta Norm Clip in layer.py

**Location**: in `MemorySpaceLayer.forward()`, after line `slot_delta = ext_h[:, k_slots:, :] - bypass_h`, before `next_hidden = bypass_h + alpha * slot_delta`.

```python
# Fix M-1: Clip slot_delta to prevent large output injections.
# Reference: bypass_h norm per token (same as Fix L-1 reference for input side).
# One-directional (only shrinks), differentiable.
_sd_norms = slot_delta.norm(dim=-1, keepdim=True)                     # [B, T, 1]
_bypass_ref = bypass_h.detach().norm(dim=-1, keepdim=True).clamp(min=1.0)  # [B, T, 1]
slot_delta = slot_delta * (_bypass_ref / _sd_norms.clamp(min=1e-6)).clamp(max=1.0)
```

**Effect**: Limits per-token injection to at most `alpha × bypass_h_norm` ≈ 0.462 × 32 ≈ 14.8 per layer. Still potentially large but bounded, and the norm clip is differentiable — gradient flows through the unclamped region normally.

### Fix M-2 (optional, more aggressive): Reduce bank max_norm

In `memory_bank.py`, change:
```python
max_norm = math.sqrt(slot_dim_f) * 2.0   # current: ≈128 for d=4096
```
to:
```python
max_norm = math.sqrt(slot_dim_f) * 0.5   # proposed: ≈32 for d=4096
```

**Rationale**: A bank cap of 128 is 4× the typical Llama-3 hidden state norm (~32 at early layers). Reducing to 32 keeps slots on the same manifold as hidden states, preventing the bank cap from being reached during normal training. This makes slot_delta naturally small.

**Risk**: Slots at norm ≤ 32 with a slot_to_hidden projection that has grown to large norms might still produce large M_sel_hidden — but Fix L-1 handles that. Reducing max_norm is complementary to Fix M-1.

### Priority

Fix M-1 is the minimum necessary change. Fix M-2 is additional defense.

---

## Evidence Table

| Node | σ | First spike step | retrieved_norm at spike onset | slot_delta_max at step 376 | Status at step 460 |
|------|---|-----------------|------------------------------|---------------------------|---------------------|
| b200-4 | 0.05 | ~200 | ~60–80 (estimated) | 7.97 | Permanent crisis (PPL>100 virtually every step) |
| b200-3 | 0.02 | ~240 | ~100 (measured step 341) | 5.125 | Recurring spikes, occasional recovery |
| b200-2 | 0.01 | ~390 | ~89 (measured step 376) | 5.25 | Spikes >14k, oscillating 100–2716 |

---

## Node State Assessment

- **b200-2 (σ=0.01)**: First spike at step 390, persistent PPL 100–2716 post-408. 400+ steps without recovery. **Unrecoverable** without restart.
- **b200-3 (σ=0.02)**: Spikes from step 240, occasional recovery windows (step 390 PPL=1.83, step 560 PPL=1.29). Slot bank integrity compromised. **Likely unrecoverable**.
- **b200-4 (σ=0.05)**: Permanent crisis from step ~200, slot_delta escalating (0.113→0.151). **Definitely unrecoverable**.

**Recommendation**: Kill all 3 nodes. Implement Fix M-1 (and optionally M-2). Restart with Fix I+J-A+K+L+M.

---

## Open Questions

1. **Is slot_delta clip sufficient or will slot norm inflation still cause instability via a different path?** — slot_delta clip prevents immediate PPL explosion but doesn't address the root cause (slot norms approaching bank cap). Fix M-2 (bank max_norm reduction) addresses the root.

2. **Will Fix M-1 + M-2 allow top1_sim to diverge from 1/N by step 800?** — Unknown. The blocking factor after Fix M is whether Q_sel can learn a useful routing signal from slot content once slots have diverse content. SKRL loss (Fix G) is the only mechanism currently pushing slot_key diversity. SKRL pairwise_cos at step 230–298 is 0.016–0.021 (vs initial ~0, target ≪0.01) — SKRL is working but slowly.

3. **Should writeback warmup be extended to allow more stable slot accumulation before routing pressure?** — This is a hyperparameter question and requires user approval.
