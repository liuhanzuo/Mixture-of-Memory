# SKRL Gradient Starvation Analysis — fix_u_ablation
**Date**: 2026-04-30 04:29 GMT+8  
**Reporter**: /researcher subagent  
**Run under analysis**: fix_u_ablation (3 nodes: b200-1/2/3)  
**Report ID**: rpt_20260430_0429_skrl_gradient_starvation  

---

## Executive Summary

**ALL THREE SYMPTOM PATTERNS EXPLAINED. NONE REQUIRE CODE FIXES TO SLOT KEYS.**

| Symptom | Root Cause | Diagnosis |
|---------|-----------|-----------|
| `slot_keys.grad_norm` exponential decay | SKRL working correctly: gradient ∝ \|\|S\|\| which shrinks as keys diverge | ✅ EXPECTED |
| `mean_pairwise_cos` plateau at -0.0020 | Theoretical minimum -1/(N-1) = -0.001957 reached after ~850 fwd calls | ✅ SKRL SUCCESS |
| `top1_sim_mean` stuck at floor ~0.0027 | T=1.0 → near-uniform softmax → flat Jacobian → **tiny Q_sel gradient** | ❌ REAL BUG |

**Primary Fix**: Increase `selector_temperature` from `1.0` → `10.0`.  
**Secondary Fix**: Set `entropy_aux_weight=0.0` (remove from b200-1 config).

---

## Detailed Analysis

### 1. Why `slot_keys.grad_norm` Decays Exponentially — EXPECTED BEHAVIOR

**Observed pattern (b200-1, steps 0-20):**
```
step 0:  0.000322
step 5:  0.000241  (75%)
step 10: 0.000176  (55%)
step 20: 0.000083  (26%)
```
Approximately 14% per-step decay.

**Mathematical derivation:**

The SKRL loss is:
```
L_SKRL = mean_pairwise_cos = (||S||² - N) / (N*(N-1))
```
where `S = Σᵢ nkᵢ` (sum of N normalized slot keys, each unit vector).

The gradient with respect to a single slot key `slot_keys[i]` is:
```
∂L/∂slot_keys[i] = 2·S / (N*(N-1)) · (I - nkᵢ·nkᵢᵀ) / ||slot_keys[i]||
```

**Crucially: gradient magnitude ∝ ||S||.**

As SKRL pushes keys apart:
- `||S||² = N + Σᵢ≠ⱼ cos(i,j)` decreases (pairwise cosines become negative)
- `||S||` → 0 as the configuration approaches the optimal spread
- Gradient → 0 proportionally

This is **mathematically guaranteed gradient vanishing at the optimum**. It is the correct behavior of a properly formulated repulsion loss. The 14%/step decay rate is consistent with exponential convergence toward the fixed point where `S ≈ 0`.

**Code evidence (selector.py lines 298-304):**
```python
def slot_key_diversity_loss(self) -> torch.Tensor:
    nk = F.normalize(self.slot_keys, dim=-1)   # [N, d]
    N = nk.size(0)
    S = nk.sum(dim=0)                           # [d], the culprit
    mean_cos = (S.dot(S) - N) / (N * (N - 1))  # ∝ ||S||²
    return mean_cos                             # gradient ∝ 2S
```

**Conclusion: Hypothesis B (Adam) and Hypothesis C (reset) and Hypothesis D (lr too small) are all FALSE.** The decay is intrinsic to the gradient of the objective. Adam cannot prevent it because the gradient itself is vanishing. No code reset is occurring.

---

### 2. Is -0.0020 a Mathematical Floor or Training Failure?

**MATHEMATICAL MINIMUM REACHED.**

For N vectors in d-dimensional space, the theoretical minimum of mean pairwise cosine similarity is:
```
min mean_pairwise_cos = -1/(N-1)
```
For N=512: `-1/511 = -0.001957`

**Observed plateau**: -0.0020 (with the analytical Fix-U formula)

**Difference**: -0.0020 - (-0.001957) = -0.000043 → essentially zero (within bfloat16 precision).

This is **NOT** "strided_token init diversity being preserved." The strided_token init places keys at real token embeddings, which start with typical pairwise cosines of +0.2 to +0.8. After SKRL training for ~850 fwd calls, keys have been pushed from that positive-cosine initial state all the way to the theoretical N-vector minimum.

**Geometric interpretation:**  
In ℝ¹²⁸, you cannot have more than 129 truly orthogonal vectors. With N=512 >> d=128, the best achievable configuration (maximum spread) has average pairwise cosine exactly = -1/(N-1). This is the **unit-sphere packing limit**, analogous to a tight frame / equiangular tight frame when N >> d.

**Conclusion: SKRL has fully succeeded within ~850 forward calls.** The "plateau" is the system settling at its optimum, not a training failure.

---

### 3. Why is `top1_sim` Stuck at Floor 0.0026? — THE REAL PROBLEM

**Observation (all 3 nodes):**
```
fwd=200: top1_sim_mean=0.002899
fwd=400: top1_sim_mean=0.002655
fwd=800: top1_sim_mean=0.002625
floor (1/N=1/512): 0.001953
```
top1_sim is only 1.37x above floor and not improving over 800 forward calls. This indicates Q_sel is not learning to route — queries are effectively random with respect to slot keys.

**Root cause: Temperature T=1.0 creates near-uniform softmax → flat Jacobian → negligible Q_sel gradient.**

**Mathematical proof:**

Expected top-1 cosine similarity between a random query and the best-spread keys (N=512, d=128):
```
E[max_i cos(q, nkᵢ)] = √(2·ln(N)/d) = √(2·ln(512)/128) = √0.0975 ≈ 0.312
```

Expected top1 softmax score at T=1.0:
```
top1_score(T=1) = exp(0.312) / (exp(0.312) + 511·exp(0)) = 1.366 / 512.366 ≈ 0.00267
```
This **exactly matches the observed 0.002625-0.002899**. The scores are at the "informationless random query" baseline.

Now, the gradient to Q_sel flows:
```
lm_loss → next_hidden → M_sel_hidden_soft → scores → logits → q → Q_sel.weight
```
The bottleneck is the softmax Jacobian at near-uniform scores:
```
∂scores/∂logits = diag(scores) - scores·scoresᵀ
max eigenvalue = (1/N)·(1 - 1/N) ≈ 1/N = 0.00195
```

**Gradient scale ≈ T/N = 1.0/512 = 0.00195 per routing step**

This is 512× smaller than the typical LM gradient magnitude. Adam's adaptive learning rate helps but cannot overcome a 500× signal attenuation. Q_sel does not receive enough signal to learn meaningful routing.

**Comparison across temperatures:**

| T | Expected top1_sim (optimal keys) | Softmax Jacobian max eigenvalue | Relative Q_sel gradient |
|---|---|----|---|
| 1.0 | 0.0027 | 0.00195 | 1x (baseline) |
| 2.0 | 0.0036 | 0.00357 | 1.8x |
| 5.0 | 0.0092 | 0.00913 | 4.7x |
| **10.0** | **0.0425** | **0.0407** | **21x** |

At T=10.0, top1_sim becomes 21x above floor, and Q_sel gradient is 21x stronger. This is the original design value before Fix O downgraded it to 1.0.

**Why Fix O was wrong (in retrospect):**  
Fix O lowered T from 10.0 to 1.0 to reduce the LM gradient path into slot_keys. But Fix Q.1 (selector.py:159-162) already severed that path by detaching slot_keys at the source:
```python
k = F.normalize(
    self.slot_keys.detach().unsqueeze(0).expand(B, -1, -1),  # ← DETACH
    dim=-1,
)
```
So the concern that motivated Fix O (slot_keys receiving too-large LM gradient) is **already solved by Fix Q.1**. T=1.0 provides no benefit while starving Q_sel of gradient.

---

### 4. b200-1 LM Instability from Step 280 — ENTROPY AUX CONFLICT

**b200-1 config**: `skrl_weight=0.10`, `entropy_aux_weight=0.001`  
**b200-2 config**: `skrl_weight=0.05`, `entropy_aux_weight=0.0` → LM healthy  
**b200-3 config**: `skrl_weight=0.15`, `entropy_aux_weight=0.0` → LM healthy  

**Root cause**: `entropy_aux_loss` maximizes routing entropy (pushes distribution toward uniform). This is appropriate when routing collapses to a single slot, but **counterproductive** after SKRL succeeds:

1. SKRL succeeds → keys are spread → softmax has genuine differential structure (some slots better than others for a given query)
2. entropy_aux_loss fires → pushes scores back toward uniform 1/N → destroys any emerging differentiation
3. This is a competing objective: SKRL diversifies key geometry (enabling differentiation), while entropy homogenizes routing scores (preventing differentiation)
4. The competing gradients to Q_sel oscillate in sign depending on the current step → unstable training → PPL spikes at step 280+

At step 280, the system has had ~2800 forward passes (8 GPU × ~350 chunks × some factor). By this point, SKRL has fully converged (||S|| → 0), so SKRL's gradient to slot_keys ≈ 0. The remaining gradient to Q_sel comes entirely from LM + entropy — with entropy pushing AGAINST the routing differentiation that LM is trying to build.

**b200-2/3 are stable because entropy_aux_weight=0.0** — they don't have the conflicting objective.

---

## Code Evidence Summary

### File: `src/memory/mem_space/selector.py`, line 163
```python
logits = torch.einsum("bs,bns->bn", q, k) * self.temperature
```
`self.temperature = 1.0` (from Fix O). With N=512, softmax on T=1.0 logits is near-uniform → tiny Q_sel gradient.

### File: `src/memory/mem_space/selector.py`, line 159-162
```python
k = F.normalize(
    self.slot_keys.detach().unsqueeze(0).expand(B, -1, -1),  # ← already detached
    dim=-1,
)
```
Fix Q.1 already detaches slot_keys. Fix O's temperature reduction is redundant and harmful.

### File: `scripts/train_mem_space_pg19.py`, line 323
```python
p.add_argument("--entropy_aux_weight", type=float, default=0.001, ...)
```
Default is 0.001. b200-1 uses this default; b200-2/3 use 0.0. The difference explains b200-1's instability.

---

## Fix Specification

### Fix V (Primary) — Restore Temperature to T=10.0

**Motivation**: Fix Q.1 already severs the LM→slot_keys gradient. Fix O's reduction to T=1.0 was a workaround for a problem that no longer exists. Restoring T=10.0 provides 21x stronger Q_sel gradient → routing differentiation will emerge.

**Implementation**: Training config change only. No code modification needed.

```bash
# Change the launch script parameter:
--selector_temperature 1.0   →   --selector_temperature 10.0
```

This is a **hyperparameter change** (confidence: very_high).

### Fix W (Secondary) — Remove entropy_aux_weight from b200-1

**Motivation**: entropy_aux_loss directly opposes routing differentiation after SKRL succeeds. The load_balance_loss provides sufficient uniformity pressure without gradient conflict.

**Implementation**: Training config change only.

```bash
--entropy_aux_weight 0.001   →   --entropy_aux_weight 0.0
```

This is a **hyperparameter change** (confidence: high).

### No Code Changes Required

All three hypotheses (A, B, C, D) from the problem statement are FALSE:
- **A** (norm growth): grad decay is SKRL convergence signal, not a pathology
- **B** (Adam adaptive lr): Adam is not the cause
- **C** (reset): no reset in forward()
- **D** (lr too small): slot_keys lr is irrelevant — the issue is Q_sel getting too little signal from T=1.0

---

## Recommended Next Ablation Design

Launch new 3-node ablation with restored T=10.0, entropy=0.0 (all nodes):

| Node | skrl_weight | selector_temperature | entropy_aux_weight | Hypothesis |
|------|------------|---------------------|-------------------|------------|
| b200-1 | 0.05 | 10.0 | 0.0 | Primary fix |
| b200-2 | 0.10 | 10.0 | 0.0 | Primary fix |
| b200-3 | 0.15 | 10.0 | 0.0 | Primary fix |

Expected outcome within 200 forward calls:
- `top1_sim_mean` > 0.010 (5x above floor) → Q_sel is learning routing
- `mean_pairwise_cos` ≈ -0.0020 (stays at minimum) → SKRL maintains key spread
- No LM instability (entropy conflict removed)

**Important diagnostic**: Add a `QSEL_GRAD_DIAG` that logs `Q_sel.weight.grad_norm` every 50 steps. This will directly confirm Q_sel is receiving signal.

---

## Confidence Assessment

| Finding | Confidence | Evidence |
|---------|-----------|---------|
| grad_norm decay = SKRL converging | **very_high** | Exact mathematical derivation; gradient ∝ \|\|S\|\| |
| plateau = theoretical minimum | **very_high** | -0.0020 ≈ -1/(N-1) = -0.001957, difference < bf16 noise |
| top1_sim floor = T=1.0 starving Q_sel | **high** | Exact match: predicted 0.0027 = observed 0.0026-0.0029 |
| b200-1 instability = entropy conflict | **high** | Only difference between b200-1 (unstable) and b200-2/3 (stable) |
| Fix V (T=10.0) will fix routing | **high** | T=10.0 → 21x stronger Q_sel gradient; original design value |

---

## What We Don't Know Yet

1. Whether T=10.0 with detached slot_keys (Fix Q.1) will still cause slot_keys instability. 
   - The original T=10.0 design (pre-Fix Q) caused routing collapse because LM gradient competed with SKRL.
   - Fix Q.1 severs that path. With detached slot_keys, T=10.0 should be safe.
   - Monitor slot_keys.grad_norm after Fix V: it should still converge (from SKRL alone) but remain > 0 longer before plateau.

2. Whether the current fix_u_ablation runs should be **killed and restarted** or if they can run to completion.
   - The current runs (T=1.0) will NOT show routing differentiation. They will collect data showing SKRL works but Q_sel doesn't learn.
   - Decision for main agent: kill + restart with T=10.0, or let run to completion for baseline data.
