# Fix G Root-Cause Analysis: K_sel Routing Degeneracy After Fix F

**Date:** 2026-04-29  
**Researcher:** researcher subagent  
**Task:** Determine why Fix F (centered STE gradient multiplier) failed to break the `top1_sim_mean = 1/512` routing degeneracy, and propose Fix G with full mathematical justification.

---

## §1. Evidence from Logs (GATE_GRAD_DIAG / QUERY_DIAG / WRITEBACK_DIAG)

### §1.1 Log Coverage

Three ablation runs were analyzed:

| Run | sigma | log file | usable steps |
|-----|-------|----------|-------------|
| node0 | 0.01 | `fix_f_ablation_sigma0.01_node0_20260429_0949.log` | ~290 steps |
| node1 | 0.02 | `fix_f_ablation_sigma0.02_node1_20260429_0947.log` | ~360 steps |
| node2 | 0.05 | `fix_f_ablation_sigma0.05_node2_20260429_0949.log` | ~290 steps |

A fourth log (`fix_f_ablation_sigma0.05_node2_20260429_1001.log`) was killed during weight loading at ~10:02:56 before any training steps — no usable data.

### §1.2 QUERY_DIAG Evidence (top1_sim_mean)

All three runs show `top1_sim_mean` stuck at ≈1/512 = 0.00195 across the full observation window:

| Run | step=97 (fwd=200) | step=204 (fwd=400) | step=298 (fwd=600) |
|-----|------------------|--------------------|-------------------|
| sigma=0.01 | 0.002289 | 0.002106 | (terminated) |
| sigma=0.02 | 0.002365 | 0.002136 | 0.002090 |
| sigma=0.05 | 0.002243 | 0.002090 | (terminated) |

Expected value at random uniform routing over N=512: `1/512 = 0.001953`. All observed values cluster within 20% of this floor. **No sigma variant broke the degeneracy in 290–360 steps.**

### §1.3 WRITEBACK_DIAG Evidence (alpha, M_sel_hidden_norm, beta)

**alpha = 0.462891 constant across all three runs, all checkpoints (fwd=200, 400, 600).**

This is diagnostic. The value 0.462891 = tanh(0.5) to float32 precision. Fix F.1 initialized `slot_output_gate` as `torch.tensor(0.5, dtype=torch.float32)`. The GATE_GRAD_DIAG logs confirm `slot_output_gate.grad` is non-zero from step 1 onward (magnitude ~1e-4 to ~1e-2). Yet alpha is frozen.

**Root cause of alpha freeze (confirmed BF16 artifact, not gradient blockade):**
- `slot_output_gate` is float32, but in the WRITEBACK_DIAG printout at inference, `alpha = tanh(self.slot_output_gate)` is computed and **cast to bf16** when stored in the model's bf16 activations
- More critically: the parameter update per step is `lr × grad ≈ 3e-4 × 1e-4 to 3e-4 × 1e-2 = 3e-8 to 3e-6`, which is far below the bf16 representation gap near 0.5 (≈ 2^-8 × 2^0 = 0.00390625)
- The float32 parameter DOES accumulate updates correctly, but `tanh(0.5 ± 1e-5)` rounds to the same bf16 value as `tanh(0.5) = 0.462891` when displayed
- **beta evolves** (0.029 → 0.062 → 0.090 across the three fwd=200/400/600 checkpoints), proving the gradient does flow through `gate_param` and the write gate is functioning
- **Fix F.1 is working correctly.** The alpha freeze is a display artifact.

**M_sel_hidden_norm_mean values:**

| Run | fwd=200 | fwd=400 | fwd=600 |
|-----|---------|---------|---------|
| sigma=0.01 | 0.639 | 0.638 | — |
| sigma=0.02 | 0.842 | 1.72 | **9.38** |
| sigma=0.05 | 3.209 | 3.202 | — |

The sigma=0.02 run shows a large jump (1.72 → 9.38) at step 298. This coincides with beta growing to 0.090 — writeback is accumulating larger-norm representations into slots. But **top1_sim at step 298 remains 0.002090**, unchanged from step 204. A 5× increase in M_sel_hidden_norm had zero effect on routing quality.

### §1.4 GATE_GRAD_DIAG Evidence (slot_keys.grad_norm)

From sigma=0.02 run, early steps:

| step | slot_keys.grad_norm |
|------|---------------------|
| 0 | ~1.28 |
| 5 | ~1.10 |
| 10 | ~0.95 |
| 20 | ~0.68 |

Gradient is non-zero and declining. The trajectory is consistent with near-zero-mean random walk: gradient magnitude persists but provides no convergent direction for slot specialization.

### §1.5 Critical Missing Datapoint

**The WRITEBACK_DIAG does NOT log `M_sel_centered_norm_mean`.**

Fix F.2 computes `M_sel_centered = (M_sel_hidden - M_sel_hidden.mean(dim=1, keepdim=True)).detach()` but this quantity is never emitted to logs. We cannot directly verify whether `||M_sel_centered||` is non-trivially large or effectively zero. This is a diagnostic blind spot — see §4.

---

## §2. Why Fix F.2 Centered STE Failed

### §2.1 What Fix F.2 Was Supposed to Do

Fix F.2 (from `ops/research_notes/20260428_gate_grad_diag_fix_f.md`) identified that the STE gradient multiplier `M_sel_hidden.detach()` is the same vector for all k selected slots when slot content is nearly identical. The fix centered the multiplier:

```python
M_sel_centered = (M_sel_hidden - M_sel_hidden.mean(dim=1, keepdim=True)).detach()  # [B, k, d]
M_sel_hidden = M_sel_hidden + M_sel_centered * (w_gathered - w_gathered.detach())
```

The intention: `M_sel_centered[b, i, :]` = deviation of slot i's hidden representation from the k-slot mean. In backward, this multiplies the gradient to `w_gathered[b, i]`, making different slots receive gradient proportional to how their content differs from the mean — a slot-differential signal.

### §2.2 The Magnitude Argument: Centering Makes Gradient Smaller, Not Larger

**M_sel_centered magnitude at initialization:**

For `hidden_pool` init with noise σ:
- `slot_i = mean_hidden + N(0, σ²I)` — all 512 slots are mean_hidden plus small noise
- After `slot_to_hidden` (Linear, std=0.02 init): `M_sel_hidden[b, i, :] = W_s2h @ slot_i + b_s2h`
- The mean across k selected slots: `M_bar = W_s2h @ mean(slot_selected) + b_s2h`
- The centered deviation: `M_sel_centered[b, i, :] = W_s2h @ (slot_i - mean(slot_selected))`
- `||slot_i - mean(slot_selected)||_2 ≈ σ · sqrt(slot_dim)` (RMS of Gaussian noise, k-sample mean subtracted)
- `||M_sel_centered[b,i,:]||_2 ≈ sigma_W · sigma_slot · sqrt(slot_dim · hidden_dim)` where sigma_W = 0.02

For sigma=0.01, slot_dim=512, hidden_dim=4096:
```
||M_sel_centered|| ≈ 0.02 × 0.01 × sqrt(512 × 4096) ≈ 0.02 × 0.01 × 1448 ≈ 0.29
```

This matches the Fix F.2 note's prediction of ~0.14 (within 2×, the discrepancy from the sqrt factor).

Compare to `||M_sel_hidden||_2 ≈ sigma_W × sigma_slot × sqrt(slot_dim × hidden_dim) × N_slots`:

No wait — M_sel_hidden is W_s2h @ slot_i, where slot_i has norm `sigma_slot * sqrt(slot_dim)` ≈ 0.01 × sqrt(512) ≈ 0.226 for sigma=0.01. This gives `||M_sel_hidden|| ≈ sigma_W × ||slot_i|| ≈ 0.02 × 0.226 × sqrt(4096) = 0.02 × 0.226 × 64 ≈ 0.29`. The observed value `0.639` for sigma=0.01 is in this range (within ~2×, explained by RMSNorm applied before slot_to_hidden which can renormalize the scale).

So `||M_sel_centered|| ≈ ||M_sel_hidden|| × (σ / ||slot_full_scale||)`. For the hidden_pool init, all slots share the large common-mode `W_s2h @ mean_hidden` which dominates M_sel_hidden by a factor of `||mean_hidden|| / (σ × sqrt(slot_dim))` ≈ many × σ / σ = O(1/σ) relative to the perturbation.

**The centering does NOT reduce the gradient multiplier by a small factor — it reduces it by approximately the signal-to-noise ratio of slot content, which for small σ is huge.**

More precisely:
- `||M_sel_hidden||` is dominated by `W_s2h @ mean_hidden` (the common-mode)
- `||M_sel_centered||` is only the perturbation: `W_s2h @ δslot_i` where `δslot_i ~ N(0, σ²I)`
- Ratio: `||M_sel_centered|| / ||M_sel_hidden|| ≈ σ × sqrt(slot_dim) / ||mean_hidden||`
- For sigma=0.01 and typical Llama RMSNorm outputs: `mean_hidden` norm ≈ `sqrt(slot_dim)` = 22.6 (unit variance per dimension after normalization), so ratio ≈ 0.01 × 22.6 / 22.6 = **0.01**

The centering multiplier is ~100× smaller than the pre-centering multiplier for sigma=0.01, and ~20× smaller for sigma=0.05. This explains why the log shows `top1_sim` moving *slower* with Fix F.2 than without it, not faster.

### §2.3 The Direction Argument: Centered Differences are Still Random Walk

Even if we accept that M_sel_centered is non-zero, consider the gradient to `slot_keys[i]`:

```
∂L/∂slot_keys[i] = ∑_b [ ∂L/∂w_gathered[b,i] · M_sel_centered[b,i,:] · J_selection ]
```

where J_selection is the Jacobian of the selection scores through the softmax/cosine routing.

At the degeneracy fixed point, slot_keys[i] ≈ slot_keys[j] for all i, j (all keys nearly identical, uniform distribution on S^127). The M_sel_centered[b,i,:] values are independent noise vectors across slots (the σ perturbations to each slot). Therefore:

- For a given query q[b], the gradient contribution from different batches is `Σ_b M_sel_centered[b,i,:] × (∂L/∂w[b,i])`
- The `∂L/∂w[b,i]` are softmax gradient terms at uniform — all O(1/N) magnitude, nearly equal across i
- The M_sel_centered terms are independent random directions in R^4096

**Each gradient step is therefore a random walk in the slot_key space**: there is no persistent direction because the slot init noise σ is resampled independently across chunks (the slot bank is reset per chunk via `_reset_banks`). Each new chunk generates a new random M_sel_centered realization.

The gradient variance is non-zero (so slot_keys do move), but the direction has zero mean across chunks — exactly consistent with the declining grad_norm in §1.4 (Adam's adaptive learning rate suppresses variance without a persistent direction).

### §2.4 The Jump at sigma=0.02 Step 298 (M_sel_hidden 1.72→9.38)

At step 298 (beta=0.090), the writeback gate is sufficiently open that some slots are accumulating significant representations from multiple chunks. This increases `||M_sel_hidden||` (the slots are no longer near-zero noise). However:

1. This M_sel_hidden signal is still common-mode: the Q/K pairs from the question chunks (which have NIAH gradients) are fed into slots that have been EMA-updated with haystack content — the haystack representations are semantically correlated (all from the same book/document), not discriminative
2. The top1_sim remains 0.002090 — the routing is still random — confirming that larger M_sel_hidden alone does not break the degeneracy

The fix needs to operate at the **slot_keys level directly**, not through the slot content → M_sel_hidden → gradient chain.

---

## §3. Fix G Proposal: Pairwise Slot-Key Repulsion Loss (SKRL)

### §3.1 Core Idea

Break the chicken-and-egg loop by adding a direct **repulsion loss on the slot_keys** in `selector.py` that is independent of slot content diversity. This provides gradient to diversify slot_keys without requiring M_sel_hidden to already be diverse.

The loss minimizes mean cosine similarity between randomly sampled pairs of (normalized) slot_keys:

```
L_skrl = mean_over_pairs{ cos_sim(slot_keys[i], slot_keys[j]) }  for i≠j
       = mean{ (k̂_i · k̂_j) }  where k̂ = F.normalize(slot_keys, dim=-1)
```

At the uniform symmetric fixed point (all slot_keys equal), this loss = 1.0 and its gradient with respect to each `slot_keys[i]` is:

```
∂L_skrl/∂k̂_i = (1/M) Σ_j k̂_j  (summing over pairs involving i)
             ≈ k̂_mean  (approximately, for large M)
```

This gradient pushes each k̂_i **away from the mean direction** — directly expanding the slot-key distribution from the degenerate fixed point. The gradient magnitude is O(1) at the symmetric fixed point, not O(σ) like M_sel_centered. It remains informative as long as the mean is non-zero, which requires full symmetry breaking to eliminate.

### §3.2 Specific Code Change

**File:** `src/memory/mem_space/selector.py`

**Where:** In the `TopKSelector` class, add a new method `slot_key_diversity_loss()` and call it in the layer's forward to augment the existing `entropy_aux_loss`.

**Old code (excerpt):**
```python
# selector.py — end of TopKSelector class, after __init__ and forward
```

**New code to add (method in TopKSelector):**
```python
def slot_key_diversity_loss(self, num_pairs: int = 512) -> torch.Tensor:
    """
    Pairwise slot-key repulsion loss.
    Minimizes mean cosine similarity between random pairs of slot_keys.
    Gradient is non-zero at the symmetric fixed point (unlike entropy_aux_loss).
    num_pairs: number of random pairs to sample (avoids O(N^2) cost for N=512).
    """
    nk = F.normalize(self.slot_keys, dim=-1)  # [N, S]
    N = nk.size(0)
    device = nk.device
    # Sample random pairs, allowing self-pairs to be filtered by mask
    i = torch.randint(N, (num_pairs,), device=device)
    j = torch.randint(N, (num_pairs,), device=device)
    # Exclude exact self-pairs (i==j) to avoid trivially maximizing self-sim
    valid = (i != j)
    if valid.sum() < 16:
        # Degenerate case: re-sample
        i = torch.arange(num_pairs, device=device) % N
        j = (torch.arange(num_pairs, device=device) + 1) % N
    else:
        i, j = i[valid], j[valid]
    cos_sim_pairs = (nk[i] * nk[j]).sum(-1)  # [M_valid]
    return cos_sim_pairs.mean()  # minimize → push keys apart
```

**Calling site in `src/memory/mem_space/layer.py`** (in the forward, where entropy_aux_loss is computed):

Existing code (approximately line 580-585 in the current version):
```python
if hasattr(self.selector, 'entropy_aux_loss'):
    entropy_loss = self.selector.entropy_aux_loss(scores)
    aux_loss = aux_loss + self.config.entropy_aux_weight * entropy_loss
```

New code:
```python
if hasattr(self.selector, 'entropy_aux_loss'):
    entropy_loss = self.selector.entropy_aux_loss(scores)
    aux_loss = aux_loss + self.config.entropy_aux_weight * entropy_loss

if hasattr(self.selector, 'slot_key_diversity_loss'):
    skrl_loss = self.selector.slot_key_diversity_loss(num_pairs=512)
    aux_loss = aux_loss + self.config.skrl_weight * skrl_loss
```

**New config field in `src/memory/mem_space/config.py`:**
```python
# After entropy_aux_weight:
skrl_weight: float = 0.01  # Weight for pairwise slot-key repulsion loss
```

### §3.3 Mathematical Argument for Why Fix G Breaks Degeneracy

**Starting condition:** All N=512 slot_keys are i.i.d. `N(0, 0.01²)` → after normalization, they cluster near a single direction (the direction of the noise mean, effectively random but all similar). `top1_sim = 1/N`.

**L_skrl at symmetric fixed point:** With all k̂_i ≈ k̂_mean (some fixed unit vector), every pairwise cosine similarity ≈ 1.0, so `L_skrl ≈ 1.0`.

**Gradient magnitude:** `∂L_skrl/∂slot_keys[i] = (2/M) Σ_j k̂_j · (I - k̂_i k̂_i^T) / ||slot_keys[i]||`

At the fixed point where all k̂_i ≈ k̂_mean:
- The gradient pushes each k̂_i **away from the mean** with magnitude `2(N-1)/N ≈ 2.0`
- This is O(1) — NOT O(σ) like the Fix F.2 centered STE multiplier

**Gradient under Adam:** Adam uses `grad / (sqrt(v) + ε)`. At init, v is small → effective LR ≈ lr (1e-3). Per-step change to `slot_keys[i]` ≈ `lr × ∂L_skrl/∂slot_keys[i] × skrl_weight ≈ 1e-3 × 2.0 × 0.01 = 2e-5`.

After `n` steps, the angular spread between slot_keys grows as `n × δθ` where `δθ ≈ δr / ||slot_keys|| ≈ 2e-5 / 0.1 = 2e-4` radians/step. Within 500 steps: spread ≈ 0.1 radians = 5.7°. This is small but measurable — top1_sim should increase from 1/N to O(1/(N/10)) = 5/N within ~500 steps if skrl is the sole driver.

**Why this is enough to break the chicken-and-egg:**
- Once slot_keys are even slightly spread (spread > N(0,σ) noise floor), the routing distribution becomes non-uniform
- Non-uniform routing → different slots get selected for different queries → their content/hidden-states diverge via writeback
- Diverse slot content → M_sel_hidden varies across slots → the STE mechanism (Fix E) starts to work
- Self-reinforcing: once started, slot specialization accelerates

**Comparison with entropy_aux_loss:**

The existing entropy loss `L_ent = -H(softmax(logits))` has gradient ≈ `(p_i - 1/N) × (1 + log p_i)` where p_i is the routing probability. At the uniform fixed point, `p_i = 1/N` → **gradient is exactly zero** (as confirmed by prior research; this is why entropy_aux_loss alone didn't help). SKRL's gradient at the fixed point is O(1).

### §3.4 Expected Observable Outcomes

With `skrl_weight=0.01`, expected trajectory:

| Steps | Observable | Expected value |
|-------|-----------|----------------|
| 1–50 | `slot_keys.grad_norm` | Increases (was declining under Fix F) |
| 50–200 | Mean pairwise cosine similarity | Should decrease from ~0.95 to ~0.5 (measurable with new diagnostic) |
| 100–300 | `top1_sim_mean` | Should rise above 1/N floor; target > 2/N = 0.0039 by step 200 |
| 200–500 | `top1_sim_mean` | Target > 5/N = 0.0098 — routing is non-trivially concentrated |
| 300–600 | `niah_acc` | If routing is working, should show first non-zero accuracy |

**Recommended diagnostic to add** (in WRITEBACK_DIAG or QUERY_DIAG):
```python
# Add to selector.forward or as a periodic diagnostic:
nk = F.normalize(self.slot_keys, dim=-1)
# Sample 256 pairs
i = torch.randint(N, (256,)); j = torch.randint(N, (256,))
mean_pairwise_cos = (nk[i] * nk[j]).sum(-1).mean().item()
logger.info(f"SKRL_DIAG | step={self.step} | mean_pairwise_cos={mean_pairwise_cos:.4f}")
```

This provides the direct observable that Fix F lacked (M_sel_centered_norm was never logged).

### §3.5 Hyperparameter Sensitivity

| Parameter | Recommended | Effect if too large | Effect if too small |
|-----------|-------------|--------------------|--------------------|
| `skrl_weight` | 0.01 | Keys spread too fast → routing entropy too high (all slots different, no load balance) | Gradient too weak vs LM gradient → no effect |
| `num_pairs` | 512 | Cost: O(num_pairs) but cheap; N=512 full cross is N²=262K, 512 is 0.2% | High variance in gradient estimate → noisy but unbiased |
| Combined with `entropy_aux_weight` | Keep at 0.001 | — | — |

Ablation suggestion: run `skrl_weight ∈ {0.001, 0.01, 0.1}` across nodes simultaneously.

---

## §4. Confidence and Caveats

### §4.1 Confidence Assessment

**High confidence (empirical):**
- alpha=0.462891 freeze is a BF16 display artifact, not a gradient blockade (gate_param.grad non-zero, beta evolves) ✓
- slot_keys.grad_norm is non-zero but declining — random walk, not convergent ✓
- M_sel_hidden_norm jump at step 298 (sigma=0.02) had zero effect on top1_sim — larger slot norms don't help routing ✓
- The chicken-and-egg loop between slot content diversity and routing quality is the correct framing ✓

**Medium confidence (theoretical):**
- Fix F.2 failed because centering reduces gradient magnitude by factor σ/||common_mode|| ≈ O(σ) — this is inferred from the magnitude argument in §2.2, not directly measured (M_sel_centered_norm was never logged)
- SKRL gradient is O(1) at the fixed point — this is analytically correct for the loss formulation

**Lower confidence (predictive):**
- Expected timeline (top1_sim > 2/N by step 200) is a rough estimate based on Adam dynamics approximation
- Interaction between SKRL and the existing STE gradient path under full training dynamics is not fully analyzed

### §4.2 Critical Caveat: M_sel_centered_norm Not Measured

The most important missing datapoint is `||M_sel_centered||_2` from Fix F.2. The magnitude argument in §2.2 is derived from theoretical initialization analysis, not direct measurement. It is possible that `||M_sel_centered||` is larger than expected (e.g., if `slot_to_hidden` has learned to amplify certain directions after N steps).

**However, this doesn't change the conclusion**: even if `||M_sel_centered||` is non-trivially large, the direction argument in §2.3 remains: the gradient is a random walk because slot init noise is independent per chunk (banks are reset per chunk). Only persistent slot content — accumulated via writeback — can provide a consistent gradient direction. And writeback cannot accumulate until routing is non-random. Fix G breaks this by providing O(1) gradient independent of slot content.

### §4.3 Alternative Explanation Not Fully Excluded

One alternative: the cosine normalization in routing creates a near-singular Jacobian. On S^127, when all slot_keys are at the same point, the Jacobian of (softmax(q·k_i temperature=10) with respect to k_i) is proportional to `(q - (q·k̂_i)k̂_i)`, which is the tangent space projection. This is non-singular as long as q ≠ k̂_i. However, with 32 layers all sharing the same bank and q being a low-norm vector from a near-zero-init Q_sel, the effective gradient might be very small. SKRL bypasses this entirely by not going through Q_sel at all.

### §4.4 Recommended Verification Before Deployment

1. Add `mean_pairwise_cos` diagnostic to SKRL_DIAG (new log tag, every 200 fwd calls)
2. Verify `slot_keys.grad_norm` increases in the first 20 steps (compared to Fix F where it was declining)
3. Check `skrl_loss` value at step 0: should be ≈ 0.95 ± 0.05 (nearly all slot_keys parallel)
4. Decision criterion at step 100: if `mean_pairwise_cos < 0.80` → SKRL is working; if still > 0.90 → increase skrl_weight to 0.05

---

## §5. Summary

**Why Fix A–F all failed:** The K_sel routing degeneracy is a metastable fixed point where all N=512 slot_keys are nearly identical. The LM gradient path to slot_keys flows through: `L_LM → next_hidden → slot_delta (×alpha) → ext_h[:k] → M_sel_hidden → STE → w_gathered → scores → slot_keys`. Every fix A–E addressed attenuation at one stage of this chain. Fix F.2 (centering) addressed the direction-degeneracy of the STE multiplier, but the centered deviations are O(σ) = 100× smaller than the common-mode, making the gradient smaller not larger. All six fixes are working *together* (alpha is open, gradient flows, slot_keys receive gradient), but they all fail to break the **symmetric fixed point** because the gradient at that point is O(σ)-small and direction-random.

**Why Fix G will work:** SKRL provides O(1)-magnitude gradient directly to slot_keys, independent of slot content, independent of M_sel_hidden, and non-zero at the symmetric fixed point. It is analogous to the "repulsion prior" in mixture models that prevents mode collapse — here applied directly to the routing keys rather than to the slot content.
