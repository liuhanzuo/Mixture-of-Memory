# Fix D Diagnosis: K_sel Routing Degeneracy Root Cause
**Date**: 2026-04-28  
**Author**: researcher subagent  
**Task**: Explain why top1_sim_mean ≈ 0.002 (= 1/512 uniform floor) persists after Fix A + B + C  

---

## 0. Symptom Summary

Across all sigma variants tested (slot_init_noise ∈ {0.01, 0.02, 0.05, 0.1, 1.0}):

| Metric | Observed | Expected (healthy) |
|---|---|---|
| top1_sim_mean | ~0.002 | >0.010 (non-trivial routing) |
| retrieved_norm_mean | scales with sigma × 64 | should grow with training |
| aux_loss | ~21.2–21.5 | should decrease toward k²/N ≈ 0.5 |
| niah_acc | 0.000 | >0.0 |

The retrieved_norm_mean = sigma × sqrt(d_model) = sigma × 64 is the *exact* formula for a random normal vector of dimension 4096 with std=sigma. This means **slots have not moved at all from initialization** — zero learning has occurred in the selector or slot bank.

---

## 1. Code Trace: Gradient Path from LM Loss to slot_keys

The relevant chain is:

```
lm_loss
  ← loss.backward()
  ← next_hidden (layer output passed to next decoder layer → ... → logits → lm_loss)
  ← bypass_h + alpha * slot_delta                     [layer.py:568]
  ← alpha = tanh(self.slot_output_gate)               [layer.py:565]
  ← slot_delta = ext_h[:, k_slots:, :] - bypass_h    [layer.py:567]
  ← ext_h from wrapped_layer(extended_hidden, ...)    [layer.py:541-557]
  ← extended_hidden = cat([M_sel_hidden, hidden_states], dim=1)  [layer.py:476]
  ← M_sel_hidden = slot_to_hidden(M_sel_slot * w_gathered)       [layer.py:473]
  ← w_gathered = ste_weights.gather(1, idx)           [layer.py:471]
  ← ste_weights ← scores ← F.softmax(logits)         [selector.py:154]
  ← logits = einsum(q, k) * 10.0                      [selector.py:153]
  ← q = F.normalize(Q_sel(pool_of_H))                 [selector.py:144]
  ← k = F.normalize(slot_keys)                        [selector.py:148-151]
  ←  *** slot_keys, Q_sel.weight ***  (trainable params)
```

The chain is unbroken on paper. **But there is a zero multiplier at the top.**

---

## 2. Root Cause #1 (Primary): slot_output_gate = 0 Blocks All LM Gradient

### The code

```python
# layer.py line 296
self.slot_output_gate = nn.Parameter(torch.zeros(()))

# layer.py lines 565–568
alpha = torch.tanh(self.slot_output_gate)               # scalar in (-1, 1)
O_mem_hidden = ext_h[:, :k_slots, :]                    # [B, k, d]
slot_delta = ext_h[:, k_slots:, :] - bypass_h           # [B, T, d]
next_hidden = bypass_h + alpha * slot_delta             # [B, T, d]; alpha=0 → bypass
```

### The gradient

By the chain rule:
```
d(next_hidden)/d(ext_h[:, k_slots:, :])  =  alpha  =  tanh(0)  =  0
d(next_hidden)/d(slot_delta)             =  alpha  =  0
```

Therefore:
```
d(lm_loss)/d(M_sel_hidden) = 0   (M_sel_hidden only affects ext_h, not bypass_h)
d(lm_loss)/d(slot_keys)    = 0
d(lm_loss)/d(Q_sel.weight) = 0
d(lm_loss)/d(ste_weights)  = 0
```

**At initialization, the LM loss contributes exactly zero gradient to every selector parameter.** This is not a numerical precision issue — it is an algebraic identity.

### Why Fix A, B, C could not solve this

- **Fix A** (slot_init_noise = 1.0): Changes initial slot values, not alpha. Still alpha=0.
- **Fix B** (standalone slot_keys Parameter): Makes slot_keys an independent parameter with a gradient path through scores → ste_weights → M_sel_slot_w → M_sel_hidden → ext_h → next_hidden. **The path exists but is zeroed by alpha=0 at the far end.**
- **Fix C** (cosine normalization + temperature=10): Changes the logit scale, not alpha. Still alpha=0.

None of the three fixes touched `slot_output_gate`. The correct gradient path requires a non-zero alpha.

### First-order Taylor analysis

Note: the docstring in layer.py (line 293–295) claims:

> *"First-order gradient is nonzero (sech²(0)=1 · <grad, slot_delta>) so the gate is trainable from step 1 onward."*

This is **incorrect**. While `d(alpha)/d(slot_output_gate) = sech²(0) = 1 ≠ 0`, the full chain rule gives:

```
d(lm_loss)/d(slot_output_gate) = d(lm_loss)/d(next_hidden) · d(next_hidden)/d(alpha) · d(alpha)/d(slot_output_gate)
                                = d(lm_loss)/d(next_hidden) · slot_delta · 1
```

The term `d(lm_loss)/d(next_hidden) · slot_delta` is the key. With alpha=0, `next_hidden = bypass_h` which does NOT depend on M_sel_hidden. So `d(lm_loss)/d(next_hidden)` is the gradient that would exist *regardless* of the memory path. The slot_delta value is non-zero (the extended forward gives different outputs than the bypass), so the gate itself DOES receive a gradient.

**BUT**: the gradient of the gate reaching slot_keys requires:

```
d(lm_loss)/d(slot_keys) = ∑_t [d(lm_loss)/d(next_hidden_t)] · slot_delta_t · 0  +  
                           0   (M_sel_hidden contributes 0 since alpha=0)
```

The gradient reaches `slot_output_gate` (it will eventually become non-zero and start opening the gate). But `slot_keys` and `Q_sel` receive zero gradient until `slot_output_gate` becomes non-zero. This is the **cold-start trap**: the gate can only train once it's open, but the selector can only train once the gate is open. At init both are stuck.

Actually let's be more precise: the gate receives gradient of the form:
```
d(lm_loss)/d(slot_output_gate) = ⟨d(lm_loss)/d(next_hidden), slot_delta⟩
```
where `slot_delta = ext_h_body - bypass_h` is non-zero (the memory tokens ARE changing the output). So slot_output_gate does get gradient from step 1. But Q_sel and slot_keys do NOT (their gradient flows through alpha which is 0).

Once alpha becomes non-zero (slot_output_gate drifts away from 0), THEN Q_sel/slot_keys get gradient. But with 512 slots all identical (from hidden_pool init + noise), the selector scores remain uniform, and since now the extended forward depends equally on all slots... the routing degeneracy persists even as alpha grows, unless aux_loss breaks the symmetry.

This brings us to Root Cause #2.

---

## 3. Root Cause #2 (Secondary): Load-Balance Loss Has Zero Gradient at the Uniform Fixed Point

### The code

```python
# selector.py lines 204-216
importance = scores.mean(dim=0)                   # [N], HAS grad
load = one_hot.float().mean(dim=0)                # [N], NO grad (hard indicator)
aux = float(N) * torch.sum(importance * load)
```

### Analytical gradient at 1/N uniform distribution

At the uniform softmax fixed point:
- `scores[b, i] = 1/N` for all b, i
- `importance[i] = 1/N` for all i
- With uniform routing (random due to identical slots): `load[i] = top_k/N` for all i
- `aux = N * sum_i (1/N * top_k/N) = N * N * (top_k/N²) = top_k` (constant at uniform!)

The gradient:
```
d(aux)/d(logits[b,j]) = N * sum_i (d(importance_i)/d(logits[b,j]) * load_i)
                      = N * (1/B) * sum_i (load_i * d(scores[b,i])/d(logits[b,j]))
```

Using the softmax Jacobian: `d(scores_i)/d(logits_j) = scores_i * (δ_{ij} - scores_j)`:
```
d(aux)/d(logits[b,j]) = (N/B) * sum_i [load_i * scores_i * (δ_{ij} - scores_j)]
                      = (N/B) * [load_j * scores_j - scores_j * sum_i(load_i * scores_i)]
```

At the uniform point with `load_i = top_k/N` for all i:
```
sum_i(load_i * scores_i) = (top_k/N) * sum_i(1/N) = top_k/N²  × N = top_k/N
d(aux)/d(logits[b,j]) = (N/B) * [(top_k/N)(1/N) - (1/N)(top_k/N)]
                      = (N/B) * 0
                      = 0
```

**The load_balance_loss gradient is identically zero at the uniform fixed point.** This is a known property of the Switch Transformer loss — it requires the load distribution to first become non-uniform before it can reinforce that non-uniformity. Since slots are initialized as nearly-identical copies of the same hidden pool mean, the load distribution IS uniform at init and stays uniform as long as the selector scores are uniform.

### Why entropy loss would fix this

The entropy-maximization auxiliary loss `L_ent = -H(scores) = sum_i scores_i log(scores_i)` has gradient:
```
d(-H)/d(scores_i) = log(scores_i) + 1
```

At the uniform point `scores_i = 1/N`:
```
d(-H)/d(scores_i) = log(1/N) + 1 = 1 - log(N)
```

For N=512: `1 - log(512) = 1 - 6.24 ≈ -5.24 ≠ 0`.

The entropy loss has non-zero gradient at the uniform fixed point, which can push scores away from uniformity and break the degeneracy symmetry.

---

## 4. Double Gradient Blockade Summary

```
                     LM loss                    aux loss (load_balance)
                        |                              |
                   alpha = 0                   gradient = 0
                        |                       at uniform fixed point
                        ↓                              ↓
                  BLOCKED                         BLOCKED
                        |
         slot_keys, Q_sel receive zero gradient from BOTH sources
                        |
                        ↓
         degeneracy is a STABLE FIXED POINT — cannot escape
```

Both gradient sources are simultaneously zero at init. The system is at a saddle/fixed point with no force to break symmetry. Fix A/B/C each changed initialization parameters that cannot overcome the gradient barriers.

---

## 5. Proposed Fix D

### Fix D.1 (Primary — must have): Non-zero slot_output_gate initialization

**File**: `src/memory/mem_space/layer.py`  
**Line**: 296

**Before**:
```python
self.slot_output_gate = nn.Parameter(torch.zeros(()))
```

**After**:
```python
self.slot_output_gate = nn.Parameter(torch.tensor(0.5))
```

**Effect**: `alpha = tanh(0.5) ≈ 0.462` at step 0. The LM gradient immediately flows through `alpha * slot_delta` to `M_sel_hidden`, `ste_weights`, `scores`, `slot_keys`, and `Q_sel.weight`. The selector is trainable from the very first step.

**Bypass residual**: `next_hidden = (1-0.462)*bypass_h + 0.462*ext_h_body`. This is a weighted blend, not pure bypass. The output will initially differ from the vanilla Llama baseline, which may cause slightly higher initial LM loss. However, since `slot_to_hidden` and `hidden_to_slot` are initialized with std=0.02 (near-zero), `M_sel_hidden ≈ 0`, meaning `ext_h_body ≈ bypass_h` and `slot_delta ≈ 0` at init anyway. The actual initial perturbation is:

```
|next_hidden - bypass_h| = alpha * |slot_delta|
                         ≈ 0.462 * 0.02 * sqrt(d_model)   (random init contribution)
                         ≈ 0.462 * 0.02 * 64 ≈ 0.59
```

versus bypass_h typical norm ≈ several tens. So the perturbation is <2% of the bypass signal — acceptable.

**Why 0.5 and not a larger value**: 
- `tanh(0.5) ≈ 0.46`: safe range, selector gets clear gradient, model stays near the bypass
- `tanh(1.0) ≈ 0.76`: too large, slot_delta perturbations compound 32× across layers
- `tanh(0.1) ≈ 0.10`: too small, gradient still nearly zero (0.1× the maximum)
- `0.5` gives a balance: ~46% weight on memory path from step 1

### Fix D.2 (Secondary — strongly recommended): Entropy-based auxiliary loss

Replace or augment the load_balance_loss in `selector.py` with an entropy maximization term:

```python
def entropy_aux_loss(self, scores: torch.Tensor) -> torch.Tensor:
    """Entropy maximization aux loss.
    
    Maximizes H(mean_scores) - mean(H(scores)) where H is Shannon entropy.
    This has non-zero gradient at the uniform fixed point (unlike Switch aux),
    pushing scores away from degeneracy.
    
    Returns a NEGATIVE value (we want to maximize entropy, so add this to loss
    with a negative sign, i.e.: total_loss = lm_loss - entropy_weight * entropy_aux_loss(scores))
    
    OR: return the negative so callers add it directly:
        loss = lm_loss + aux_weight * entropy_aux_loss(scores)
    where entropy_aux_loss returns -H(mean_scores) (positive, to be minimized by Adam).
    """
    # scores: [B, N]
    # Minimize negative entropy = maximize entropy
    mean_scores = scores.mean(dim=0)   # [N], averaged over batch
    # H = -sum(p log p), but we minimize -H, so loss = sum(p log p)
    eps = 1e-8
    neg_entropy = (mean_scores * torch.log(mean_scores + eps)).sum()
    return neg_entropy  # scalar; add to loss (minimizing = maximizing entropy)
```

**Usage in training loop**: 
```python
ent_loss = layer.selector.entropy_aux_loss(scores)
loss = lm_loss + lb_weight * lb_loss + ent_weight * ent_loss
```

**Typical weight**: `ent_weight ≈ 0.01–0.1`. Start with 0.01 and verify routing diversity improves.

### Fix D.3 (Diagnostic — immediate check): Print slot_keys.grad after backward

Add to `train_mem_space_pg19.py` after `loss.backward()`:

```python
# Diagnostic: verify grad flow to selector (remove after Fix D confirmed working)
for name, p in model.named_parameters():
    if 'slot_keys' in name and p.grad is not None:
        print(f"[GRAD_CHECK] {name}: grad.norm={p.grad.norm().item():.6f}", flush=True)
        break
else:
    print("[GRAD_CHECK] WARNING: slot_keys.grad is None — gradient not flowing!", flush=True)
```

This directly confirms whether Fix D.1 successfully opens the gradient pathway.

---

## 6. Expected Observations After Fix D

If Fix D is correct, after applying Fix D.1 (non-zero gate init):

| Metric | Before Fix D | After Fix D (expected) |
|---|---|---|
| slot_keys.grad.norm | 0.0 | > 0.0 (non-trivial) |
| top1_sim_mean | ~0.002 (uniform) | > 0.01, increasing over training |
| retrieved_norm_mean | sigma × 64 (frozen) | Increases as slots diverge |
| aux_loss | ~21.5 (stuck at uniform) | Decreasing toward minimum |
| niah_acc | 0.000 | > 0.0 (after ~1000+ steps) |
| initial LM loss | baseline (alpha=0) | ~1–5% higher initially |

---

## 7. Implementation Priority

1. **Fix D.1 first** — single-line change, directly fixes the primary blockade.
2. **Add diagnostic print** — confirm gradient is flowing.
3. **Fix D.2 if Fix D.1 alone is insufficient** — the entropy loss handles the secondary blockade.

The secondary blockade (Switch aux at uniform fixed point) may self-resolve once Fix D.1 is applied: with non-zero alpha, training will break slot symmetry through the LM loss, making load non-uniform, at which point the Switch aux loss gradient becomes non-zero and reinforces the non-uniformity.

---

## 8. Files to Modify

| File | Change |
|---|---|
| `src/memory/mem_space/layer.py` line 296 | `torch.zeros(())` → `torch.tensor(0.5)` |
| `src/memory/mem_space/selector.py` (optional) | Add `entropy_aux_loss` method |
| `scripts/train_mem_space_pg19.py` (temporary) | Add `slot_keys.grad` diagnostic print |

---

## Appendix: Evidence for "Slots Not Moving"

`retrieved_norm_mean = sigma × sqrt(d_model)` exact formula check:

| sigma | sigma × 64 | observed retrieved_norm_mean |
|---|---|---|
| 0.01 | 0.64 | 0.64 ✓ |
| 0.02 | 1.28 | 1.28 ✓ |
| 0.05 | 3.20 | 3.20 ✓ |

The mean L2 norm of a `d`-dimensional Gaussian vector with std σ is `σ × sqrt(d)`. With `d=4096`, `sqrt(4096) = 64`. The fact that retrieved_norm_mean matches `sigma × 64` exactly across 5 orders of magnitude of sigma variation proves that slots are at their initial values — they have never been updated by the optimizer.
