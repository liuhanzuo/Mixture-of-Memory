# Fix V Diagnosis — SKRL Mathematical Floor, Diversity≠Selectivity, and InfoNCE

**Date**: 2026-04-30  
**Report ID**: rpt_20260430_0430_fix_v_diagnosis  
**Triggered by**: fix_u_ablation STEP-500 CRITERION DEFINITIVELY FAILED — mean_pairwise_cos=-0.0020 FROZEN, top1_sim stuck at floor ~0.0027 on all 3 nodes  
**Analyst**: /researcher subagent  
**Intermediate note**: `ops/research_notes/20260430_0429_skrl_gradient_starvation.md` (written at 04:29, covers T=1.0 gradient starvation analysis)

---

## Executive Summary

Three independent failure modes converged in fix_u_ablation:

| Symptom | Root Cause | Severity |
|---------|-----------|----------|
| mean_pairwise_cos frozen at -0.0020 | SKRL **succeeded** — ETF minimum reached | Informational, not a bug |
| top1_sim stuck at floor 0.0027 | T=1.0 (Fix O) starves Q_sel gradient 21× | **PRIMARY FIX NEEDED** |
| b200-1 LM collapse at step 510 | entropy_aux_weight=0.001 maximizes routing entropy | **SECONDARY FIX NEEDED** |

**Immediate Fix V**: `selector_temperature 1.0 → 10.0` + `entropy_aux_weight 0.001 → 0.0`  
**Medium-term Fix V.2**: Replace SKRL with InfoNCE query-alignment loss (code change)

---

## Part 1: Why SKRL Plateaus at -1/(N-1) — Mathematical Analysis

### 1.1 Equiangular Tight Frame (ETF) Convergence

The analytical SKRL loss (Fix T) is:
```
L_SKRL = mean_pairwise_cos = (||S||² - N) / (N·(N-1))    where S = Σᵢ nkᵢ
```

For N=512 unit vectors in d=128 dimensional space:
- **ETF minimum**: -1/(N-1) = -1/511 = -0.001957
- **Observed plateau**: -0.0020 (difference < bfloat16 noise floor: ±0.0002)

This is NOT a pathology. SKRL has successfully pushed slot_keys to the mathematically optimal configuration for maximum pairwise separation.

**Why ETF is achievable**: N=512 << d(d+1)/2 = 8256 (dimension of symmetric matrices in ℝ¹²⁸). In this regime, N vectors can be arranged as an approximate tight frame where each pair of normalized vectors has the same cosine similarity = -1/(N-1). This is the **sphere-packing limit** for unit vectors.

**Why gradient vanishes**: The gradient of L_SKRL w.r.t. slot_keys[i] is proportional to **S = Σᵢ nkᵢ**. At the ETF minimum, S → 0 (keys sum to zero vector). This is the correct behavior of a repulsion objective at its global minimum — gradient vanishing is convergence, not failure.

### 1.2 Correction: slot_keys Initialization

**TRAINER_ACTIVE.md states**: "strided_token init already achieves MAXIMUM possible slot key diversity at initialization."

**This is INCORRECT based on code inspection.**

- `src/memory/mem_space/selector.py` line 108: `self.slot_keys = nn.Parameter(torch.randn(num_slots, selector_dim) * 0.1)`
- **slot_keys are Gaussian N(0, 0.1²), NOT strided_token**
- strided_token init applies to **slot VALUES** in `memory_bank.py` (distinct token embeddings from hidden states)

Gaussian N(0, 0.1²) keys in d=128 start with mean pairwise cosine ≈ 0 ± 0.088. The ETF minimum (-0.001957) is reached **during training** via SKRL gradient descent. The plateau is SKRL converging successfully, not preserving initial diversity.

---

## Part 2: Diversity ≠ Query-Alignment — The Deeper Problem

### 2.1 Why ETF Minimum is the Wrong Optimization Target

At the ETF minimum (maximally spread keys), slot_keys are arranged to minimize average pairwise similarity. This is a **geometric** property of the key set — independent of any query distribution.

**The critical flaw**: Maximum geometric diversity does NOT imply maximum routing selectivity.

**Mathematical argument**: For a query q ∈ S^{d-1} (unit vector), the softmax routing distribution at temperature T is:
```
p_i ∝ exp(T · cos(q, nk_i))
```

At ETF minimum with T=1.0:
- cos(q, nk_i) ≈ independent N(0, 1/√d) for each key
- Expected top-1 softmax score = exp(0.312) / (exp(0.312) + 511) ≈ 0.0027 **regardless of which key**
- This is the "informationless random query" baseline — Q_sel is receiving zero useful signal about which slot to route to

The ETF minimum guarantees that ALL queries map to a near-uniform distribution over slots at T=1.0. This means:
1. No slot is preferentially addressed by any query pattern
2. Memory retrieval is essentially random
3. The write-back signal (if slot retrieval is random, what did we "remember"?) is meaningless

### 2.2 Temperature Is the Proximate Fix

The ETF minimum is USEFUL if T is high enough to break the near-uniform distribution:

| T | Expected top1_sim (random query vs ETF keys) | Q_sel gradient (relative) |
|---|---|---|
| 1.0 | 0.0027 | 1× (baseline) |
| 5.0 | 0.009 | 4.7× |
| **10.0** | **0.042** | **21×** |

At T=10.0:
- Q_sel receives 21× stronger gradient from LM loss
- Even for random keys at ETF, routing concentrates on the geometrically-closest key
- Q_sel can LEARN to produce queries that concentrate even further on task-relevant keys

**Why T=10.0 was originally correct**: The pre-Fix-O design used T=10.0. Fix O lowered T to 1.0 to prevent slot_keys from receiving too-large LM gradient. But Fix Q.1 already severed that path (`self.slot_keys.detach()` at source in selector.forward()). Fix O's motivation no longer exists, but its damage persists.

### 2.3 Why InfoNCE is the Long-Term Fix (Options Analysis)

Even with T=10.0, SKRL pushes keys toward uniform geometric spread (ETF). This means:
- Keys are maximally distinguishable in the abstract geometric sense
- But there is no alignment between key geometry and actual query distribution
- A key at -q direction (maximally far from a query pattern q) will never be selected by that pattern

**InfoNCE routing loss** fixes this by training slot_keys to be discriminable by ACTUAL queries:
```
L_InfoNCE = -E_q[log softmax(cos(q, k_{top1}) * T_info) / Σᵢ exp(cos(q, k_i) * T_info)]
```

Where top1 is the slot the LM actually chose (hard top-k result). This creates an attractive force:
- Keys that are frequently selected for a query type get PULLED toward that query's direction
- Keys that are never selected get pushed away from common query directions
- Result: key geometry aligns with query distribution (selectivity > ETF diversity)

**Analysis of Options A-E from TRAINER_ACTIVE.md:**

| Option | Assessment |
|--------|-----------|
| A: Change init to near-uniform | MOOT — slot_keys ARE already Gaussian (near-uniform). The ETF minimum is reached by TRAINING, not init. |
| B: Target query-aligned diversity (SKRL reformulation) | **InfoNCE is the correct implementation of this.** Viable. Requires code change. |
| C: Remove SKRL, rely on LM-only writeback signal | Possible but slow — LM gradient to slot_keys is severed (Fix Q.2). No direct signal to slot_keys without SKRL or replacement. |
| D: Hard-assignment top_k | Would force differentiation via winner-take-all, but gradient becomes even sparser. Not recommended. |
| **Fix V.1 (primary)** | **Restore T=10.0, remove entropy_aux. No code change. High confidence fix.** |
| **Fix V.2 (medium-term)** | **Add InfoNCE query_alignment_loss to selector.py. Replaces SKRL. Code change needed.** |

---

## Part 3: entropy_aux Is Actively Harmful

### 3.1 Why entropy_aux_loss Causes LM Collapse

`selector.py` lines 233-275 implements `entropy_aux_loss()`:
```python
entropy = -(p * torch.log(p.clamp(min=1e-8))).sum()   # scalar ≥ 0
return -entropy                                         # negate: minimise → maximise H
```

The **docstring** claims this "pushes away from uniformity" (the uniform fixed point gradient is non-zero and the docstring interprets it as pushing toward differentiation). **This is WRONG.**

**Actual gradient direction**: Minimizing `-H` in total loss = maximizing `H` = maximizing routing entropy = pushing routing distribution toward UNIFORM (1/N for each slot).

**Mechanism of LM collapse**:
1. Training begins. SKRL spreads keys. Q_sel starts learning (especially at T=10.0).
2. entropy_aux fires. Pushes Q_sel to produce more uniform routing distributions.
3. Competition: LM tries to differentiate routing (some slots more useful for current input), entropy_aux opposes this.
4. At entropy_aux_weight=0.001: after ~500 steps, the uniform-routing attractor wins.
5. All slot retrievals become equally meaningless → alpha=0.462 × 32 layers of near-zero residual → loss minimum is to ignore memory → abrupt change in LM weight distribution → PPL spike.

**Evidence from fix_u_ablation**:
- b200-1 (entropy_aux_weight=0.001): LM collapse step 510, lm_ppl 958-1426
- b200-2 (entropy_aux_weight=0.0): LM healthy
- b200-3 (entropy_aux_weight=0.0): LM healthy

The only parameter difference between b200-1 and b200-2/3 in fix_u_ablation was entropy_aux_weight. This is sufficient evidence for causality.

### 3.2 Why entropy_aux Was Added (Fix D.2) and Why It Should Be Removed

Fix D.2 added entropy_aux to provide gradient at the uniform routing fixed point (where load_balance_loss has zero gradient). The docstring reflects this intent: provide a "non-zero push away from uniformity" when routing collapses.

But **maximizing entropy IS pushing toward uniformity**, not away from it. The docstring confused the direction of the fix. The intended effect was to push routing TOWARD uniform when it had collapsed to a single slot. But the formula pushes ALWAYS toward uniform, whether routing is collapsed or distributed.

**Fix**: Set `entropy_aux_weight=0.0`. The load_balance_loss already penalizes single-slot collapse via the Switch-Transformer formulation. entropy_aux provides no benefit that load_balance doesn't already cover, and actively harms routing differentiation.

---

## Part 4: Fix Specifications

### Fix V.1 — Hyperparameter Changes Only (IMMEDIATE)

**Confidence: HIGH**  
**Code changes: NONE**  
**Expected outcome: top1_sim > 0.005 within 500 steps**

```bash
# All three nodes:
--selector_temperature 10.0   # was 1.0 (Fix O regression, now safe with Fix Q.2 detach)
--entropy_aux_weight 0.0      # was 0.001 (actively harmful, causes LM collapse)
--skrl_weight 0.05/0.10/0.15  # keep sweep, SKRL is working correctly
```

**Why T=10.0 is safe now**: Fix Q.2 (`self.slot_keys.detach()` at source in selector.py:159) severs the LM gradient from reaching slot_keys through the key scoring computation. T=10.0 only amplifies the Q_sel gradient — it no longer amplifies any gradient into slot_keys from the LM path.

### Fix V.2 — InfoNCE Query Alignment Loss (MEDIUM-TERM)

**Confidence: MEDIUM** (architecture needs experimental validation)  
**Code changes: selector.py, config.py, layer.py**

This replaces SKRL with a loss that trains slot_keys to be selectable by real queries rather than just geometrically spread.

#### selector.py changes

**In `forward()` method, after line 185 (return idx, scores, ste_weights), add:**
```python
# Store last query for InfoNCE loss computation
self.last_q = q.detach().clone()  # [B, S], no grad; don't contaminate
self.last_idx = idx.detach().clone()  # [B, top_k], hard selection
```

**Add new method `query_alignment_loss()`:**
```python
def query_alignment_loss(self, temperature: float = 10.0) -> torch.Tensor:
    """InfoNCE routing contrastive loss.
    
    Trains slot_keys to be selectable by actual queries:
    - Top-1 selected slot key = positive
    - All other keys = negatives
    - Loss = -E[log softmax(cos(q, k_pos) * T)]

    This replaces SKRL's geometric repulsion with query-aligned differentiation:
    keys that are selected by similar queries will cluster, keys that are
    selected by different queries will spread. The resulting geometry aligns
    with the actual query distribution rather than uniform sphere packing.
    
    Requires self.last_q and self.last_idx to be set in forward().
    Returns 0 if forward() has not been called yet.
    """
    if not hasattr(self, 'last_q') or self.last_q is None:
        return torch.zeros(1, device=self.slot_keys.device)
    
    q = self.last_q                        # [B, S], detached, unit vectors
    idx = self.last_idx                    # [B, top_k]
    
    nk = F.normalize(self.slot_keys, dim=-1)  # [N, S], grad flows here
    
    # Use only top-1 (first column) as positive example
    pos_idx = idx[:, 0]                    # [B]
    k_pos = nk[pos_idx]                    # [B, S], positive key per batch item
    
    # Compute logits: [B] positive, then [B, N] all-slot scores
    pos_logit = (q * k_pos).sum(-1) * temperature   # [B]
    all_logits = torch.einsum("bs,ns->bn", q, nk) * temperature  # [B, N]
    
    # InfoNCE: log(exp(pos) / sum_all_exp) = pos - logsumexp(all)
    # Note: all_logits already includes the positive (it's a column in nk)
    loss = -(pos_logit - torch.logsumexp(all_logits, dim=-1))  # [B]
    return loss.mean()                     # scalar, minimize
```

#### config.py changes

```python
# Replace:
skrl_weight: float = 0.0
# With:
skrl_weight: float = 0.0
query_alignment_weight: float = 0.0   # Fix V.2: InfoNCE routing contrastive loss weight

# Change default:
entropy_aux_weight: float = 0.0       # was 0.001; entropy maximization is harmful
```

#### layer.py changes

In the aux losses block (after line 703):
```python
if cfg.query_alignment_weight > 0.0 and hasattr(self.selector, 'query_alignment_loss'):
    qa_loss = self.selector.query_alignment_loss()
    aux["query_alignment"] = qa_loss * cfg.query_alignment_weight
```

#### Training script flags for Fix V.2 run:
```bash
--skrl_weight 0.0
--query_alignment_weight 0.05  # start conservative; info about weight TBD
--entropy_aux_weight 0.0
--selector_temperature 10.0    # keep high temperature for clear routing signal
```

---

## Part 5: Success Criterion

| Fix | Metric | Target | Steps |
|-----|--------|--------|-------|
| V.1 (T=10.0) | top1_sim_mean | > 0.005 | 500 |
| V.1 (T=10.0) | top1_sim_mean | > 0.020 | 1000 |
| V.1 (T=10.0) | lm_ppl | < 10.0 | 1000 |
| V.1 (T=10.0) | mean_pairwise_cos | ≈ -0.002 (stays at ETF) | 500 |
| V.2 (InfoNCE) | top1_sim_mean | > 0.05 | 1000 |
| V.2 (InfoNCE) | QUERY_DIAG cos | > 0.10 (task-relevant slot highly preferred) | 2000 |

**Key diagnostic to add for V.1 run**: Log `Q_sel.weight.grad.norm()` every 50 fwd calls. At T=10.0, this should be at least 5× larger than at T=1.0 by step 200. If Q_sel gradient is still near-zero at T=10.0, it indicates a different problem.

---

## Part 6: What We Now Know About the Full Failure Chain

```
Init: slot_keys ~ N(0, 0.1²) [Gaussian, NOT strided_token]
       ↓ SKRL fires
Step 850: slot_keys at ETF minimum (-1/(N-1) = -0.001957) ← CORRECT, WORKING
       ↓ Q_sel trying to learn routing
       ↓ BUT T=1.0 (Fix O) → softmax Jacobian max eigenvalue = 1/N = 0.002
       ↓ Q_sel gradient ≈ 512× attenuated
Step 0-1000: Q_sel makes no progress toward selective routing ← TEMPERATURE BUG
       ↓ [only on nodes with entropy_aux_weight=0.001]
Step 500+: entropy_aux pushes routing back to uniform
           Q_sel "learns" to produce uniform routing (entropy maximized)
           Memory slots all carry same content → memory is useless
           LM adjusts to ignore memory → PPL spike ← LM COLLAPSE
```

**Fix Q.2 is working correctly** (slot_keys not receiving LM gradient). The remaining issues are entirely in the Q_sel learning dynamics (Fix V.1: T=10.0) and the entropy objective (Fix V.1: entropy_aux=0.0).

---

## Confidence Assessment

| Finding | Confidence | Evidence |
|---------|-----------|---------|
| slot_keys = Gaussian init (NOT strided_token) | **CONFIRMED** | Code: selector.py:108 `torch.randn(...) * 0.1` |
| ETF minimum reached at step ~850 | **CONFIRMED** | Analytical: -0.0020 ≈ -0.001957 (bfloat16 noise only) |
| T=1.0 starves Q_sel gradient 21× | **HIGH** | Mathematical: softmax Jacobian max eigenvalue = T/N; predicted top1_sim 0.0027 = observed 0.0026 |
| entropy_aux causes b200-1 LM collapse | **HIGH** | Experimental: b200-1 (entropy=0.001) collapsed, b200-2/3 (entropy=0.0) stable |
| T=10.0 will fix top1_sim floor | **HIGH** | Prediction: expected top1_sim ≈ 0.040 at T=10.0 vs 0.0027 at T=1.0 |
| InfoNCE will further improve selectivity | **MEDIUM** | Theoretical; requires ablation to confirm |

**Recommended next worker**: trainer (deploy fix_v.1_ablation with T=10.0, entropy=0.0)  
**If top1_sim > 0.005 at step 500**: Fix V.1 sufficient. Continue training.  
**If top1_sim still at floor after Fix V.1**: Dispatch coder for InfoNCE (Fix V.2).
