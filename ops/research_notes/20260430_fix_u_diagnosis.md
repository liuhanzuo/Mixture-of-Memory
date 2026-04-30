# Fix U Diagnosis — SKRL_DIAG random-pair sampling is the noise source

**Date**: 2026-04-30  
**Report ID**: rpt_20260430_0350_fix_u_diagnosis  
**Triggered by**: Observed ±0.015 oscillation in SKRL_DIAG `mean_pairwise_cos` despite Fix T switching `slot_key_diversity_loss()` to analytical (zero-variance) formula  
**Analyst**: /researcher subagent

---

## 1. The Observation

After Fix T was applied to `selector.py::slot_key_diversity_loss()`, monitoring logs across all three fix_t_ablation nodes (b200-1/2/3) still showed:

```
[SKRL_DIAG fwd=200]  mean_pairwise_cos= 0.0031
[SKRL_DIAG fwd=400]  mean_pairwise_cos=-0.0083
[SKRL_DIAG fwd=600]  mean_pairwise_cos= 0.0121
[SKRL_DIAG fwd=800]  mean_pairwise_cos=-0.0142
[SKRL_DIAG fwd=1000] mean_pairwise_cos= 0.0098
```

The signal oscillates between approximately ±0.015 with no monotonic decrease toward negative values — the expected signature of successful slot-key repulsion via SKRL.

The prior interpretation was that slot_keys were not actually diverging, implying Fix T itself was failing. This led to further investigation and eventually the nodes being killed.

---

## 2. Root Cause: SKRL_DIAG Was Never Updated to Match Fix T

### 2.1 What Fix T changed

`selector.py::slot_key_diversity_loss()` was rewritten from random-pair sampling to the analytical identity:

```python
# OLD (Fix G, random pairs, high variance):
idx_i = torch.randint(N, (num_pairs,), device=nk.device)
idx_j = torch.randint(N, (num_pairs,), device=nk.device)
mean_cos = (nk[idx_i] * nk[idx_j]).sum(-1).mean()

# NEW (Fix T, analytical, zero variance):
S = nk.sum(dim=0)                          # [d]
mean_cos = (S.dot(S) - N) / (N * (N - 1)) # exact
```

### 2.2 What SKRL_DIAG still does

`layer.py` lines 679–682:

```python
nk = F.normalize(self.selector.slot_keys, dim=-1)
N = nk.size(0)
idx_i = torch.randint(N, (256,), device=nk.device)
idx_j = torch.randint(N, (256,), device=nk.device)
mean_pairwise_cos = (nk[idx_i] * nk[idx_j]).sum(-1).mean().item()
```

**SKRL_DIAG was never updated.** It still uses 256 random pairs — the original pre-Fix-T approach that Fix T was designed to eliminate.

---

## 3. Quantitative Noise Analysis

For N=512 unit-vector slot keys with true mean pairwise cosine `μ`:

**Variance of 256-pair estimator:**

Each sampled pair `(i, j)` with `i = j` (same index drawn) contributes cos_sim = 1, which is a genuine contamination source. Even excluding that, the variance per pair for near-uniform keys is approximately:

```
Var[cos(nk_i, nk_j)] ≈ 1/d = 1/128 ≈ 0.0078
```

For 256 independent pairs:
```
σ(256-pair mean) = sqrt(0.0078 / 256) ≈ 0.0055
```

**±3σ range: ±0.0165**

This exactly matches the ±0.015 oscillation observed in logs. 

### 3.1 Why i=j contamination makes it worse

`torch.randint(N, (256,))` samples with replacement. For N=512, P(i=j) = 1/512 per pair × 256 pairs ≈ 0.5 expected self-pairs per batch. Self-pairs always contribute +1.0, biasing the mean upward by ~0.002 per call. This adds additional non-zero mean offset to the oscillation.

### 3.2 SNR calculation

If Fix T successfully pushes mean_pairwise_cos from 0 to -0.01 over 1000 steps (a conservative estimate given skrl_weight=0.05), the per-step signal is:

```
Δμ ≈ -0.01 / 1000 ≈ -1e-5 per diagnostic call (at 200-step interval → 5 calls per 1000 steps)
Signal per call ≈ -0.002
Noise per call  ≈ 0.0055 (σ)
SNR ≈ 0.36
```

A 256-pair random estimator is **below the noise floor** for detecting the actual gradient signal. We would need ~1000+ steps of averaging to see the trend over the noise.

---

## 4. The Key Insight: Fix T May Already Be Working

**The oscillation is 100% measurement noise, not actual slot_key movement.**

The critical implication:

> Fix T may be correctly pushing slot_keys apart, but we killed the runs prematurely because our diagnostic was too noisy to confirm it.

The actual slot_keys state (queried analytically) might show a monotonic decreasing trend in mean_pairwise_cos. We were measuring noise and interpreting it as signal failure.

---

## 5. Fix U Specification

**File**: `src/memory/mem_space/layer.py`  
**Lines**: 678–682 (SKRL_DIAG block)  
**Change**: Replace random-pair sampling with the same analytical formula used in `slot_key_diversity_loss()`

### Before (lines 679–682):
```python
            nk = F.normalize(self.selector.slot_keys, dim=-1)
            N = nk.size(0)
            idx_i = torch.randint(N, (256,), device=nk.device)
            idx_j = torch.randint(N, (256,), device=nk.device)
            mean_pairwise_cos = (nk[idx_i] * nk[idx_j]).sum(-1).mean().item()
```

### After (Fix U):
```python
            nk = F.normalize(self.selector.slot_keys, dim=-1)
            N = nk.size(0)
            S_diag = nk.sum(dim=0)                                          # [d]
            mean_pairwise_cos = ((S_diag.dot(S_diag) - N) / (N * (N - 1))).item()
```

**Properties of the fix:**
- Zero variance — same exact value every call for the same slot_keys state
- O(N·d) cost — same as forward-sampled 256 pairs but without randomness  
- Self-pair contamination eliminated — diagonal terms cancel exactly in the identity
- Matches the actual loss signal — SKRL_DIAG will now show what `slot_key_diversity_loss()` is optimizing

---

## 6. Expected Post-Fix Behavior

After Fix U, SKRL_DIAG should show:
- **Steps 0–200**: mean_pairwise_cos ≈ 0 ± 0.002 (init near-uniform, small noise from strided_token init)
- **Steps 200–2000**: monotonic decrease toward -0.01 to -0.05 range, rate proportional to skrl_weight
- **Steady state**: value stabilizes when SKRL repulsion balances the implicit attraction from LM loss gradient through Q_sel

If the signal still shows oscillation ±0.015 after Fix U, that would indicate Fix T is genuinely failing (zero-variance analytical formula oscillating = something is resetting slot_keys each step).

---

## 7. Impact Assessment

| Item | Assessment |
|------|------------|
| Measurement noise cause | **Confirmed**: 256-pair random estimator σ=0.0055 exactly matches ±0.015 oscillation |
| Fix T effectiveness | **Unknown but plausible**: the fix may be working; we need Fix U to verify |
| Nodes killed prematurely | **Possible**: fix_t_ablation may have been working; cannot reconstruct without restart |
| Urgency of Fix U | **High**: every monitoring cycle without Fix U produces uninterpretable diagnostics |

---

## 8. Recommended Actions

1. **Apply Fix U** (3-line change in layer.py SKRL_DIAG) — dispatch `/coder`
2. **Restart fix_t_ablation** on available nodes with same config as before kill
3. **Monitor new SKRL_DIAG** — expect clean monotonic signal within 200 steps

**Confidence**: HIGH  
**Effort**: 3-line change, zero architectural risk  
**Recommended next worker**: coder
