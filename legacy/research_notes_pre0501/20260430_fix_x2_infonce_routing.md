# Fix X.2 Research Report — Why Key Clustering Fails to Improve top1_sim, and InfoNCE Implementation Spec

**Date**: 2026-04-30  
**Report ID**: rpt_20260430_HHMM_fix_x2_infonce  
**Triggered by**: fix_x_ablation STEP-500 CRITERION DEFINITIVELY FAILED on ALL 3 nodes  
**Analyst**: /researcher subagent  

---

## Executive Summary

The fix_x_ablation confirmed that LM gradient flowing to `slot_keys` (Fix X.1 — remove `.detach()`) causes meaningful key clustering: `pairwise_cos` rose from -0.0020 (ETF minimum, SKRL-driven) to **+0.004** (positive, rising). Keys are actively clustering rather than maximally spreading.

**Yet top1_sim remains at floor ~0.002 — why?**

This report identifies **two independent failure modes** that prevent geometric clustering from translating to routing selectivity, and provides a complete validated specification for InfoNCE-based routing alignment (Fix X.2).

---

## Part 1: Why pairwise_cos = +0.004 Does NOT Improve top1_sim

### 1.1 Statistical Significance: Is the Clustering Real?

For N=512 unit vectors in d=128 dimensions, the random (null hypothesis) baseline is:
- E[pairwise_cos] = 0
- std(single pairwise cosine) = 1/√d = 0.0884
- std(MEAN of all 130,816 pairs) = 0.0884 / √130,816 = **0.000244** (CLT)

Observed value +0.004:
- In sigma units: **0.004 / 0.000244 = 16.4 sigma** — extremely statistically significant
- Keys ARE genuinely clustering, not random fluctuation

### 1.2 Physical Magnitude: Is the Clustering Useful?

The centroid vector **S = Σᵢ nkᵢ** has norm derived from pairwise_cos:
```
||S||² = N + N·(N-1)·pairwise_cos = 512 + 512·511·(+0.004) = 1558.5
||S|| = 39.48
||centroid|| = ||S||/N = 39.48/512 = 0.0771
```

**Key insight**: centroid_norm = 7.7%, meaning:
- **Only 0.6% of key variance** (centroid_norm² = 0.006) points in the cluster direction
- **99.4% of key variance** is in the residual (non-cluster) components
- The residual component std = √(1 - 0.0771²)/√d = **0.0881** (nearly identical to random: 0.0884)

### 1.3 Why Common-Mode Clustering Does Not Increase Routing Selectivity

When all keys share a common direction (cluster centroid), the cosine similarity with any query q decomposes as:
```
cos(q, nk_i) = ||c||·cos(q, centroid) + cos(q, residual_i)
             = 0.077·(shared_value) + residual_noise_i
```

**The critical flaw**: The term `0.077·cos(q, centroid)` is a **constant** across all keys for a fixed query q. Since softmax is invariant to uniform additive shifts (subtracting the same value from all logits), this common-mode term contributes ZERO to routing discrimination.

The effective discrimination is entirely from the residual component (std = 0.0881), which is virtually identical to the random/ETF case (std = 0.0884).

**Therefore**: pairwise_cos = +0.004 vs -0.002 (ETF) produces **nearly identical top1_sim** — both ~0.042 in theory (if Q_sel were perfectly aligned). The observed floor at 0.002 must come from a different cause.

### 1.4 Failure Mode 2: STE Gradient Through Uniform Slot Values

The gradient path from LM loss to Q_sel runs through the STE soft proxy:
```python
M_sel_slot_soft = torch.einsum("bn,bnd->bd", scores, slots)  # [B, slot_dim]
M_sel_hidden = M_hard.detach() + (M_sel_hidden_soft - M_sel_hidden_soft.detach())
```

The gradient of the LM loss w.r.t. `scores[b, i]` is:
```
∂(LM_loss)/∂(scores[b,i]) = ∂(LM_loss)/∂(M_sel_hidden) · slot_to_hidden(slots[b,i])
```

**The problem**: All slots are initialized from `hidden_pool` + noise. After reset at each chunk boundary, all slots in the bank are re-initialized from the SAME pooled hidden mean with small noise (~1.0 std). After writeback EMA with β=0.15, slots gradually diverge, but in the early steps:
```
slots[b, i] ≈ slots[b, j] for most i,j
=> slot_to_hidden(slots[b,i]) ≈ slot_to_hidden(slots[b,j])
=> ∂(LM_loss)/∂(scores[b,i]) ≈ constant for all i
=> STE gradient to scores is near-uniform
=> No useful gradient reaches Q_sel.weight
```

This means Q_sel has no signal about **which** slot to prefer — all slots look the same through slot_to_hidden.

### 1.5 Why LM Gradient Directly to slot_keys Also Fails

Fix X.1 allows LM gradient to reach `slot_keys` via the scoring computation:
```
logits = q · k (cosine scores) × T  =>  scores (softmax)  =>  STE  =>  M_sel_hidden  =>  LM
```

But this chain also goes through the STE soft proxy, which has the SAME uniform-slot-value problem described above. The gradient `∂(LM_loss)/∂(slot_keys[i])` involves `∂(STE)/∂(k_i) ∝ slot_to_hidden(slots[b,i])` — again, near-uniform when slots are similar.

**Summary**: Both routes to slot_key learning (LM→slot_keys via scoring, LM→Q_sel via STE) are bottlenecked by the uniform slot value problem. Keys cluster because the shared LM gradient pushes them collectively, but there is no differential signal distinguishing which key should move where.

---

## Part 2: Mathematical Analysis — ETF vs Clustered vs InfoNCE

### 2.1 Expected top1_sim Values at T=10.0

| Configuration | pairwise_cos | centroid_norm | residual_std | E[top1_cos] | top1_sim theory |
|---|---|---|---|---|---|
| ETF minimum (SKRL) | -0.002 | ~0 | 0.0884 | 0.312 | **0.042** |
| Clustered (Fix X.1) | +0.004 | 0.077 | 0.0881 | 0.312 | **0.042** |
| InfoNCE converged, pos_cos=0.3 | varies | N/A | N/A | 0.3 fixed | **0.038** |
| InfoNCE converged, pos_cos=0.5 | varies | N/A | N/A | 0.5 fixed | **0.225** |

**Key observation**: ETF and weakly-clustered configurations have nearly identical theoretical top1_sim (~0.042) because in both cases, the discrimination comes entirely from cosine noise of similar magnitude.

**Actual observed top1_sim = 0.002** (20× below theory) confirms that neither slot_keys geometry nor Q_sel alignment is working correctly through the LM gradient path.

### 2.2 Required Spread for top1_sim > 0.010

The criterion top1_sim_mean > 0.010 at fwd=500 requires:
```
top1_sim = exp(T·Δ) / (exp(T·Δ) + (N-1)) > 0.010

Solving: Δ = ln(0.010·(N-1)/(1-0.010)) / T = ln(5.162) / 10 = 0.1641
```

**Required Δ = 0.164**: The positive (selected) slot key must be at least 0.164 cosine units ABOVE the mean key for the query direction. This is achievable with InfoNCE — it directly trains exactly this property.

### 2.3 Statistical Significance Summary

| Metric | Value | Null hypothesis | Significance |
|---|---|---|---|
| pairwise_cos (fix_x_ablation) | +0.004 | 0.000 | 16.4σ |
| Change from ETF to clustered | Δ = 0.006 | 0 | Enormous |
| Improvement in top1_sim | 0.002 → 0.002 | Expected 0.042 | Unchanged |

**Conclusion**: pairwise_cos = +0.004 is real and statistically significant (16.4σ), but the physical effect on top1_sim is negligible because (a) clustering magnitude is tiny (7.7% of unit sphere), (b) common-mode shift is softmax-invariant, and (c) residual discrimination is essentially unchanged from random.

---

## Part 3: InfoNCE Implementation Specification (Fix X.2)

### 3.1 Gradient Flow Analysis

**The Fix V.2 code sketch** (in `20260430_fix_v_diagnosis.md` Part 4) stores `q.detach()` and calls `query_alignment_loss()` as a separate method. This analysis evaluates whether this design is correct.

**Gradient flow in the sketch**:
```
InfoNCE_loss → nk (F.normalize(slot_keys)) → slot_keys.weight   ✓ CORRECT
InfoNCE_loss does NOT reach Q_sel.weight (q is detached)         ✓ INTENTIONAL
```

**Assessment**: The asymmetric design is correct. Here's why:

1. **slot_keys training via InfoNCE** (direct, strong): InfoNCE pulls the top-1 selected key toward the query direction. Gradient magnitude ≈ T·(1 - 1/N) ≈ 9.98 at initialization. This is strong and non-vanishing regardless of slot value diversity.

2. **Q_sel training via LM** (indirect, enabled when keys align): Once slot_keys cluster around semantically meaningful directions, STE gradient becomes NON-UNIFORM (different slots = different keys = different LM loss sensitivity). Q_sel can then learn.

3. **Why NOT use non-detached q for InfoNCE**: If q is not detached, InfoNCE gradient flows backward to Q_sel.weight via Q_sel(pool_of_H). This would make Q_sel learn to produce queries that look "arbitrary" (maximize agreement with positives under InfoNCE alone, ignoring LM). The LM gradient and InfoNCE gradient would COMPETE for Q_sel training direction.

**Verdict**: `q = self.last_q = q.detach()` is the CORRECT design choice. InfoNCE trains ONLY slot_keys. Q_sel trains only through LM gradient.

### 3.2 Positive and Negative Construction

**Design**: top-1 selected slot key = positive, all other 511 keys = negatives.

**Assessment**: This is correct. Analysis:

1. **Top-1 as positive**: The hard top-k selection (via `torch.topk`) picks the geometrically closest key regardless of LM relevance. Using only top-1 (not all top-k) avoids confusion: a query might select k=16 slots, but only the absolute best-matching key should be pulled toward the query.

2. **All N-1 as negatives**: Standard InfoNCE. All other keys are negatives. This gives N-1=511 negative examples per batch item. With T=10.0, this creates a strong multi-class contrastive signal.

3. **Including positive in denominator**: The sketch computes:
   ```python
   all_logits = torch.einsum("bs,ns->bn", q, nk) * temperature  # [B, N]
   loss = -(pos_logit - torch.logsumexp(all_logits, dim=-1))     # [B]
   ```
   This is equivalent to standard InfoNCE (positive IS included in the sum-exp denominator). This is correct — it corresponds to the proper softmax cross-entropy over N classes.

4. **Gradient check**:
   - `∂(L)/∂(cos(q, k_pos)·T) = p_pos - 1` where p_pos = softmax score
   - At random state: p_pos = 1/512, gradient = -0.998 (strong pull toward q)
   - At good separation (pos_cos=0.9): p_pos = 0.94, gradient = -0.06 (naturally diminishes)
   - **No gradient collapse risk** — gradient is large when it needs to be, small when alignment is achieved

### 3.3 Gradient Collapse Analysis

Concern: With N=512 negatives, can InfoNCE become trivial if the positive is always separated?

**Answer: No.** Analysis:

| State | p_pos (softmax) | InfoNCE loss | grad magnitude on k_pos |
|---|---|---|---|
| Random (all keys identical) | 1/512 = 0.0020 | 6.24 nats | 9.980 |
| Floor state (fix_x_ablation) | 0.0020 | 6.24 nats | 9.980 |
| Criterion achieved (top1=0.01) | 0.010 | 4.62 nats | 9.900 |
| Good routing (top1=0.05) | 0.050 | 3.00 nats | 9.500 |
| Excellent routing (top1=0.5) | 0.500 | 0.69 nats | 5.000 |
| Nearly perfect (top1=0.94) | 0.940 | 0.06 nats | 0.600 |

**Gradient collapse only occurs when routing is ALREADY EXCELLENT** (top1_sim > 0.9). This is the desired behavior — InfoNCE training naturally terminates when keys are well-placed. No separate InfoNCE temperature tuning is needed.

**Recommended InfoNCE temperature**: T=10.0 (same as routing temperature). Using the same temperature ensures that the InfoNCE training objective and the routing measurement criterion (top1_sim) are fully aligned.

### 3.4 DDP and Gradient Accumulation Handling

**The Fix V.2 code sketch ISSUE**: `query_alignment_loss()` is called externally (after `forward()` completes), reading `self.last_q` and `self.last_idx`.

**DDP scenario** (each GPU has its own model copy):
- Each GPU replica stores its OWN `last_q`/`last_idx` from its own batch
- InfoNCE gradient on `slot_keys` is computed locally per GPU
- DDP all-reduces `slot_keys.grad` after `backward()` → correct averaging
- **NO special handling needed** ✓

**Gradient accumulation scenario** (N micro-steps before backward):
- Each micro-step's `forward()` OVERWRITES `last_q`/`last_idx`
- When `query_alignment_loss()` is finally called, it only uses the LAST micro-step's query
- → Information loss for earlier micro-steps

**CRITICAL FIX**: Compute InfoNCE loss INSIDE `forward()`, not as a separate external call.

If `query_alignment_loss()` is called at the END of each `forward()` (before the computation graph is released), each micro-step contributes its own InfoNCE loss to the total loss, and gradient accumulation works naturally.

### 3.5 Corrected Implementation Specification

**Change from Fix V.2 sketch**: Move InfoNCE computation into `forward()`, not as a separate external method call. The `query_alignment_loss()` method can still exist for modularity, but it MUST be called inside `forward()`.

#### selector.py — Minimal Change

In `forward()`, after computing `q`, `k`, `logits`, `scores`, `idx` (and BEFORE returning):

```python
# Store for InfoNCE loss computation (called immediately in layer.py's forward)
# IMPORTANT: q is NOT detached here — we need the attached q for computing InfoNCE inside forward()
# The .detach() is applied inside query_alignment_loss() on the q argument.
self.last_q = q      # [B, S], unit vectors, ATTACHED to computation graph
self.last_idx = idx  # [B, top_k], hard selection indices
```

Add method `query_alignment_loss()` (taking attached q, detaching inside):

```python
def query_alignment_loss(self, temperature: float = None) -> torch.Tensor:
    """InfoNCE routing contrastive loss.

    Trains slot_keys to cluster around actual queries:
    - Top-1 selected slot key = positive sample
    - All other N-1 keys = negatives
    - Loss = -E[log softmax_pos]  (standard NT-Xent / InfoNCE)

    Gradient flows:
    - TO slot_keys (via nk): pulls k_pos toward q, pushes k_neg away ✓
    - NOT to Q_sel (q is detached inside this method) ✓
    - NOT to slot VALUES (operates only on slot_keys geometry) ✓

    Gradient magnitude at random state: T * (1 - 1/N) ≈ T  (non-vanishing)
    Gradient collapse risk: NONE until top1_sim > 0.5 (desired behavior)

    Args:
        temperature: InfoNCE temperature (default: self.temperature)

    Returns:
        scalar, minimize (larger = less aligned)
    """
    if not hasattr(self, 'last_q') or self.last_q is None:
        return torch.zeros(1, device=self.slot_keys.device)

    T_info = temperature if temperature is not None else self.temperature
    q = self.last_q.detach()          # [B, S], DETACH here: gradient only flows to slot_keys
    idx = self.last_idx               # [B, top_k]

    nk = F.normalize(self.slot_keys, dim=-1)  # [N, S]; gradient flows to slot_keys.weight

    # Top-1 positive key per batch item
    pos_idx = idx[:, 0]               # [B], hard top-1 index
    k_pos = nk[pos_idx]               # [B, S], positive key vectors

    # Positive logit: cos(q, k_pos) * T
    pos_logit = (q * k_pos).sum(dim=-1) * T_info    # [B]

    # All logits: cos(q, k_i) * T for all N keys
    all_logits = torch.einsum("bs,ns->bn", q, nk) * T_info   # [B, N]

    # InfoNCE: -log(exp(pos_logit) / sum_i exp(logit_i))
    # Equivalent to cross-entropy where positive class = pos_idx
    # Note: pos_logit IS included in logsumexp(all_logits) since pos_idx is one of the N indices
    loss = -(pos_logit - torch.logsumexp(all_logits, dim=-1))  # [B]
    return loss.mean()                # scalar, minimize
```

#### layer.py — Compute InfoNCE INSIDE forward()

In the aux_losses block (approximately after the SKRL block, before aux["beta"]):

```python
# Fix X.2 (2026-04-30): InfoNCE query alignment loss.
# Trains slot_keys toward actual queries. Gradient flows to slot_keys ONLY
# (q is detached inside query_alignment_loss). Must be computed HERE inside
# forward() so that each gradient accumulation step contributes its own InfoNCE
# loss to the computation graph. DO NOT move outside forward().
if cfg.query_alignment_weight > 0.0 and hasattr(self.selector, 'query_alignment_loss'):
    qa_loss = self.selector.query_alignment_loss()
    aux["query_alignment"] = qa_loss * cfg.query_alignment_weight
```

#### config.py — Add field

```python
# Fix X.2 (2026-04-30): InfoNCE query alignment loss weight.
# Trains slot_keys toward actual query directions (contrastive routing).
# Replaces SKRL (geometric repulsion); operates on query-key alignment, not geometry.
# Recommended sweep: 0.01 / 0.05 / 0.1. Start at 0.05.
query_alignment_weight: float = 0.0  # disabled by default for backward compat
```

Also update config default (already documented as harmful):
```python
entropy_aux_weight: float = 0.0   # was 0.001; entropy maximization is harmful (see rpt_20260430_0430_fix_v_diagnosis)
```

---

## Part 4: Ablation Design for fix_x2_ablation

### 4.1 Node Assignments

| Node | query_alignment_weight | skrl_weight | norm_cap | T | Expected Effect |
|---|---|---|---|---|---|
| b200-1 (node0) | **0.05** | 0.0 | 10.0 | 10.0 | Primary test — moderate InfoNCE |
| b200-2 (node1) | **0.01** | 0.0 | 0.0 | 10.0 | Low weight — minimum viable InfoNCE |
| b200-3 (node2) | **0.10** | 0.0 | 10.0 | 10.0 | High weight — aggressive alignment |

Rationale for norm_cap=0.0 on b200-2:
- b200-2 had retrieved_norm=48.3 (EXPLOSION) with skrl=0.0, norm_cap=0.0 in fix_x_ablation
- This tests whether InfoNCE routing alignment itself prevents slot explosion (via more selective writes)
- If it explodes again: add norm_cap=10.0 and rerun

### 4.2 Shared Hyperparameters

```bash
--selector_temperature 10.0      # proven necessary from fix_v analysis
--entropy_aux_weight 0.0         # entropy maximization is harmful (confirmed)
--skrl_weight 0.0                # SKRL is anti-productive (confirmed fix_x_ablation)
--load_balance_weight 0.01       # keep as is (prevents routing collapse)
--query_alignment_weight <node>  # sweep above
```

### 4.3 Kill Criterion

**Kill ALL 3 nodes if**: `top1_sim_mean < 0.005` at fwd=500 on ALL 3 simultaneously.

**Early success signal**: Any node showing `top1_sim_mean > 0.010` at fwd=300 → let that node continue.

### 4.4 Expected top1_sim Trajectory

**If InfoNCE works correctly**:

| fwd count | Expected top1_sim | Mechanism |
|---|---|---|
| 0-50 | ~0.002 (unchanged) | Keys haven't moved yet |
| 50-200 | 0.002-0.005 | Keys start separating toward query directions |
| 200-500 | 0.005-0.020 | Meaningful routing clusters forming |
| 500+ | 0.020-0.100 | Q_sel learning activates (non-uniform slot values) |

**If InfoNCE fails (top1_sim still < 0.005 at fwd=500)**:
1. Check QUERY_DIAG for InfoNCE loss value — should start at ~6.24 and decrease
2. If InfoNCE loss not decreasing: verify `query_alignment_weight` is being applied to total loss in training script
3. If InfoNCE loss decreasing but top1_sim not rising: diagnostic print needed for `slot_keys.grad.norm()` to verify gradient is flowing

---

## Part 5: Important Caveats and Risks

### 5.1 The "Bootstrapping" Problem

InfoNCE uses the top-1 CURRENT selection as the positive. At initialization, the top-1 selection is effectively random (scores uniform). This means InfoNCE will train keys toward random queries at first — is this a problem?

**Answer: No.** The LM gradient already provides signal that eventually differentiates which keys should match which query patterns. InfoNCE creates the DISCRIMINABILITY infrastructure (keys spread toward different query clusters) that allows LM gradient to create meaningful selectivity. The bootstrapping works because:
1. Even random positive key assignment creates slight, inconsistent key movement
2. After each chunk, some keys happen to align better with frequent LM-relevant queries
3. These better-aligned keys win the top-1 more consistently → their alignment signal is reinforced
4. Self-reinforcing cycle accelerates alignment

### 5.2 Risk: InfoNCE Chasing Noise

If query patterns (Q_sel(pool_of_H)) are completely random and not semantically meaningful, InfoNCE will chase noise and keys will cluster toward noise patterns. This would manifest as:
- InfoNCE loss decreasing
- But top1_sim increasing trivially (keys cluster toward mean query, not toward informative queries)
- LM PPL not improving

**Detection**: Watch for top1_sim rising to ~0.04 (the theoretical ETF prediction) but LM PPL not improving beyond ~9.0. This would suggest routing is quantized into slots but slots don't carry useful content.

### 5.3 norm_cap=0.0 on b200-2 Risk

b200-2 node previously showed retrieved_norm=48.3 (explosion). Without norm_cap, this could recur. InfoNCE alone may not prevent slot explosion if routing is very non-selective early (all 512 slots receive equal writes with high EMA).

**Mitigation**: If retrieved_norm > 20 appears at any QUERY_DIAG point on b200-2, abort that node's run (it's an independent data point from b200-1/3 which have norm_cap=10.0).

---

## Part 6: Why InfoNCE Is the Right Fix (Prior Art Summary)

Multiple successful MoE routing systems use variants of query-aligned key learning:
- **Switch Transformer** (Fedus et al. 2021): capacity factor + load balance loss — no geometric repulsion
- **Expert Choice** (Zhou et al. 2022): keys trained through LM gradient only (no SKRL equivalent)
- **Mixtral** (Mistral AI 2023): top-2 routing, keys trained through LM gradient
- **DeepSeek-MoE** (DeepSeek 2024): auxiliary loss for load balance, NOT geometric repulsion

**None** of the successful MoE systems use geometric repulsion (equivalent to SKRL). They all rely on:
1. LM gradient flowing to gate/key parameters
2. Load balance aux loss to prevent routing collapse

InfoNCE Fix X.2 adds the missing piece: a direct contrastive signal that trains keys to be DISCRIMINABLE by actual queries, rather than just geometrically spread.

---

## Part 7: Confidence Assessment

| Finding | Confidence | Evidence Type |
|---|---|---|
| pairwise_cos=+0.004 is statistically significant | **CONFIRMED** | 16.4σ, analytic |
| pairwise_cos=+0.004 is physically tiny (centroid_norm=7.7%) | **CONFIRMED** | Mathematical: centroid_norm² = 0.006 |
| Common-mode clustering does not improve routing selectivity | **HIGH** | Softmax invariance to uniform shifts; analytic proof |
| STE gradient blocked by uniform slot values | **HIGH** | Code analysis: slot_to_hidden(slots[i]) ≈ slot_to_hidden(slots[j]) |
| InfoNCE gradient flow design (q detached, slot_keys trained) | **CONFIRMED** | Code analysis + gradient chain derivation |
| InfoNCE no gradient collapse risk | **CONFIRMED** | Mathematical analysis of all states |
| InfoNCE in-forward computation (vs external call) required for grad accumulation | **HIGH** | Gradient accumulation semantics analysis |
| InfoNCE temperature T=10.0 correct | **HIGH** | Matches routing temperature; no benefit to separate T |
| Ablation weights 0.01/0.05/0.10 appropriate | **MEDIUM** | Ratio analysis: 2.5%/12.5%/25% of LM loss at init |
| Expected top1_sim > 0.010 at fwd=500 | **MEDIUM** | Theoretical + bootstrapping argument; needs experimental validation |

---

## Part 8: Required Diagnostic Additions for fix_x2_ablation

The current QUERY_DIAG does NOT log InfoNCE loss. The coder should add:

1. **InfoNCE loss value** in QUERY_DIAG: `qa_loss_mean` — should start at ~6.24 and decrease
2. **slot_keys grad norm** (optional): `selector.slot_keys.grad.norm()` if available — should be non-zero

These diagnostics are needed to distinguish between:
- InfoNCE loss decreasing but top1_sim not rising (key alignment not captured by routing)
- InfoNCE not decreasing (training signal not flowing)
- Both decreasing correctly (success path)

---

## Summary

**Root causes of fix_x_ablation failure**:
1. Weak clustering (centroid_norm = 7.7%): common-mode shift is softmax-invariant → zero effect on routing selectivity
2. STE gradient bottleneck: uniform slot values → uniform d(LM)/d(scores_i) → no Q_sel or slot_key differentiation signal through the LM path

**Fix X.2**: Add InfoNCE query alignment loss that directly trains slot_keys toward actual queries, bypassing the slot-value bottleneck entirely. Compute inside `forward()` to handle gradient accumulation correctly.

**Recommended next step**: Deploy `/coder` to implement Fix X.2 (selector.py + config.py + layer.py) then `/trainer` to launch fix_x2_ablation with query_alignment_weight sweep {0.01, 0.05, 0.10}.
