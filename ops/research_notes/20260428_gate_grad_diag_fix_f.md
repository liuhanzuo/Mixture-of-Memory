# Fix F: Gradient Symmetry Metastable Fixed Point — Root Cause and Proposed Fix

**Date**: 2026-04-28  
**Author**: researcher subagent (dispatched by main after fix_e confirmed)  
**Status**: diagnosis complete, Fix F proposed — NO code written  
**Input evidence**: GATE_GRAD_DIAG log (fix_e_ablation_sigma0.01_node0_20260428_2329.log), selector.py, layer.py, memory_bank.py, config.py  
**Prior chain**: Fix A (σ=1.0) → Fix B (slot_keys param) → Fix C (F.normalize cosine) → Fix D.1 (gate_init=0.5) → Fix D.2 (entropy_aux) → Fix E (STE scale correction)

---

## §1  Evidence Summary

After Fix D + Fix E, the training diagnostic log shows:

| Metric | Value | Expectation (healthy) |
|--------|-------|----------------------|
| slot_output_gate.grad | ~O(1), all 20 GATE_GRAD_DIAG steps | ≥ 1e-4 ✅ |
| slot_keys.grad_norm | 0.67–1.3 | ≥ 1e-3 ✅ |
| slot_to_hidden.weight.grad_norm | 1.75–14.5 | ≥ 1e-3 ✅ |
| hidden_to_slot.weight.grad_norm | None (frozen) | — (intentional, hidden_to_slot_frozen=True) |
| trainable params with grad | 159/192 | 159/192 (33 frozen = hidden_to_slot × layers) |
| top1_sim_mean at fwd=200 | 0.002090 ≈ 1/512 | >> 0.002 ✗ |
| retrieved_norm_mean at fwd=200 | 0.640491 | >> σ × sqrt(d) |
| alpha (slot_output_gate) | 0.462891, frozen across all diag steps | should evolve ✗ |

**The paradox**: gradient is non-zero and of healthy magnitude, yet routing remains exactly at the 1/N uniform floor and α appears frozen at its initial tanh(0.5) value (BF16 quantization artifact: the actual gate_param oscillates but rounds to the same BF16 representable value near 0.5).

---

## §2  Hypothesis Evaluation

### Hypothesis A — Gradient is zero (falsified)

The GATE_GRAD_DIAG log shows slot_keys.grad_norm = 0.67–1.3 at every step. **This hypothesis is definitively falsified.** The gradient is present.

### Hypothesis B — hidden_to_slot.weight.grad_norm = None is a bug (falsified)

Reading `config.py` line 80 (`hidden_to_slot_frozen: bool = True`) and `layer.py` lines 314–316:

```python
if cfg.hidden_to_slot_frozen:
    for p in self.hidden_to_slot.parameters():
        p.requires_grad = False
```

This is **intentional** — the Stage-2a Tier-3 cure (confirmed PPL=2.1278). The 33 frozen params (hidden_to_slot × layers) perfectly account for the 159/192 = 33 frozen count. **Not a bug.**

### Hypothesis C — STE correction carries no gradient (falsified)

Fix E (`layer.py` lines 478–482) replaced the pre-Fix-E code where `M_sel_hidden` was attenuated by `w_gathered` before projection. The current code:

```python
w_gathered = ste_weights.gather(1, idx).unsqueeze(-1)           # [B, k, 1]
M_sel_hidden = self.slot_to_hidden(M_sel_slot)                   # [B, k, d]; full scale
M_sel_hidden = M_sel_hidden + M_sel_hidden.detach() * (w_gathered - w_gathered.detach())
```

`d/d(w_gathered) = M_sel_hidden.detach()` which is non-zero (previous researcher confirmed M_sel_hidden_norm grew from 0.0016 → 0.82 after Fix E). The STE gradient path is open. **Hypothesis C falsified.**

### Hypothesis D — Load-balance loss has zero gradient at uniform point (confirmed, but secondary)

The Switch-Transformer load_balance_loss (`selector.py` lines 181–217):

```python
importance = scores.mean(dim=0)    # [N]
one_hot = zeros.scatter_(idx, 1.0)
load = one_hot.float().mean(dim=0) # [N]
aux = float(N) * torch.sum(importance * load)
```

`load` uses a hard indicator (non-differentiable). At the uniform fixed point (all `importance[i]` = k/N, all `load[i]` = k/N), `d(aux)/d(importance[i]) = load[i]` — a constant independent of `scores`. The only differentiable path is `d(aux)/d(scores)` through `importance`. At the fixed point, `d(aux)/d(scores_b,i) = load[i]` — a *constant*. This is technically non-zero, but it pushes all slots equally and cannot break slot-symmetry.

Fix D.2's `entropy_aux_loss` (`selector.py` lines 222–265):

```python
p = scores.mean(dim=0)
entropy = -(p * torch.log(p.clamp(min=1e-8))).sum()
return -entropy  # minimise → maximise H
```

At the uniform point: `p[i] = 1/N`, gradient = `-(log(1/N) + 1)/N = (log(N) - 1)/N ≈ 0.010` for N=512. **Non-zero gradient confirmed.** But see §3 for why this doesn't break the symmetry.

### Hypothesis E — Gradient symmetry: gradient is non-zero but carries no slot-specialization signal (confirmed, PRIMARY ROOT CAUSE)

This is the actual explanation. See §3.

---

## §3  Root Cause: Gradient Symmetry Metastable Fixed Point

### 3.1  Setup: near-identical slot content at initialization

`MemoryBank.init_from_hidden` (`memory_bank.py` lines 142–153):

```python
elif self.slot_init == "hidden_pool":
    pooled = H_l.detach().mean(dim=1, keepdim=True)   # [B, 1, d]
    slots = pooled.expand(B, N, d).contiguous().clone()
    if self.init_noise > 0.0:
        slots = slots + torch.randn_like(slots) * self.init_noise
```

With `σ = 0.01` (used in this experiment) and `d = 4096`:
- Mean slot value = `pooled` = same for all N=512 slots
- Slot diversity = `σ = 0.01` per element, relative to post-RMSNorm magnitude ≈ `1/sqrt(d) ≈ 0.016`
- **Slot diversity signal ≈ σ / (1/sqrt(d)) = 0.01 / 0.016 = 0.625 ×** the hidden signal magnitude

While this is not negligible at the element level, the **crucial quantity is slot-to-slot difference** as seen by the gradient:
- `slot_to_hidden(slot_i)` ≈ `slot_to_hidden(slot_j)` because slot_i ≈ slot_j at step 0
- The σ=0.01 noise is 50× smaller than σ=0.5 needed to produce diverse `slot_to_hidden` outputs (std=0.02 projection × σ_input → std output ≈ 0.02 × σ × sqrt(slot_dim))

### 3.2  The STE gradient path

The STE correction is:

```python
M_sel_hidden = M_sel_hidden + M_sel_hidden.detach() * (w_gathered - w_gathered.detach())
```

In backward, the gradient w.r.t. `w_gathered` from this term is:

```
d(loss)/d(w_gathered[b,j]) = sum_d [ d(loss)/d(M_sel_hidden[b,j,d]) * M_sel_hidden[b,j,d].detach() ]
                            = <g_Msel[b,j],  M_sel_hidden[b,j]>_detached
```

where `g_Msel[b,j]` is the gradient of loss w.r.t. M_sel_hidden[b,j] (flowing back through cross-attention).

This scalar then propagates through `ste_weights`:

```
ste_weights = scores + (one_hot_scores - scores).detach()
```

The backward pass of STE is `d(loss)/d(scores[b,i])` — the full softmax Jacobian applied to `d(loss)/d(ste_weights[b,i])`. For selected slot j at index `idx[b,j]`:

```
d(loss)/d(scores[b,idx[b,j]]) 
    = <g_Msel[b,j], M_sel_hidden[b,j]>_detach * softmax_jacobian
```

This gradient then flows to `Q_sel` (via the query) and to `slot_keys[idx[b,j]]` (via the key normalization path):

```
d(loss)/d(slot_keys[i]) ∝ <g_Msel[b,j], M_sel_hidden[b,j]>_detach * q[b] * temperature * normalize_jacobian
```

### 3.3  Why this gradient cannot break symmetry

**The key observation**: when all slots have nearly identical content, `M_sel_hidden[b,j]` ≈ the *same vector* for all j in the selected set {0, 1, ..., k-1}. Specifically:

```
M_sel_hidden[b,j] ≈ μ + ε_j     where μ is the mean output, ||ε_j|| << ||μ||
```

The gradient signal reaching `slot_keys[idx[b,j]]` is approximately:

```
d(loss)/d(slot_keys[idx[b,j]]) ≈ c(b,j) * q[b] * temperature * normalize_jacobian(slot_keys[idx[b,j]])
```

where `c(b,j) = <g_Msel[b,j], μ + ε_j>` ≈ `<g_Msel[b,j], μ>` (since ε_j is small).

**The common-mode problem**: `μ` is approximately the same for all j. Therefore all k selected slot_keys in a given batch item b receive gradient proportional to the same direction `q[b]`. For unselected slots (i ∉ top-k), the gradient arrives through the `scores` path of the STE but is attenuated by the softmax Jacobian at the uniform point (all eigenvalues ≈ 1/N).

The net result over many batch items: each `slot_keys[i]` accumulates gradient from a *random subset* of batch items (those where i was in the top-k under uniform sampling). Since the top-k selection is nearly random (scores all equal), the gradient accumulated by `slot_keys[i]` is approximately:

```
sum_{b: i in top-k(b)} c(b,i) * q[b]
```

This is a random linear combination of query vectors `q[b]` with approximately equal positive coefficients. Its expectation over training steps is:

```
E[grad(slot_keys[i])] ≈ (k/N) * E_b[c(b) * q[b]] ≈ (k/N) * μ_grad * E_b[q[b]]
```

If the training queries are diverse (zero-mean in query space), this expectation is approximately **zero** — a random walk on the hypersphere S^{selector_dim-1}. Under Adam normalization, random-direction gradients perform a correlated random walk with no systematic drift toward slot specialization.

### 3.4  Why entropy_aux also fails to break symmetry

The entropy gradient (§2 Hypothesis D):

```
d(-H)/d(logits[i]) ∝ -(log(p[i]) + 1)
```

At the uniform point, this is identical for ALL i (since all p[i] = 1/N → log(p[i]) = log(1/N)). The entropy gradient pushes all slot_keys to DIVERGE FROM EACH OTHER, but it does so in a completely symmetric way — there is no preferred direction for any specific slot_keys[i] to go. Under Adam with random initialization directions, entropy aux causes a random repulsion that may not converge to a stable diverse allocation.

### 3.5  Why α appears frozen at 0.462891

BF16 has 7 mantissa bits → resolution near 0.5 is ~0.5 × 2^{-7} ≈ 0.0039. The gate_param update at each step (learning rate × gradient × Adam scaling) is of order `1e-4 × 1.0 × 1.0 = 1e-4`, which is 25× smaller than the BF16 representable gap. The gate_param oscillates but rounds to the same BF16 value `0.5` → `tanh(0.5) = 0.462891` every diagnostic step. This is a **BF16 quantization artifact, not a gradient blockade**. The gate IS training, just at sub-BF16 resolution per step.

---

## §4  Fix F Proposal

### 4.1  Centered STE correction

**Location**: `src/memory/mem_space/layer.py`, line 482.

**Current code** (lines 478–482):
```python
w_gathered = ste_weights.gather(1, idx).unsqueeze(-1)          # [B, k, 1]
M_sel_hidden = self.slot_to_hidden(M_sel_slot)                 # [B, k, d]; full scale
# STE correction: zero in forward, non-zero in backward.
# d/d(w_gathered) = M_sel_hidden.detach() → gradient to Q_sel/slot_keys preserved.
M_sel_hidden = M_sel_hidden + M_sel_hidden.detach() * (w_gathered - w_gathered.detach())
```

**Proposed Fix F** (replace line 482 only):
```python
w_gathered = ste_weights.gather(1, idx).unsqueeze(-1)          # [B, k, 1]
M_sel_hidden = self.slot_to_hidden(M_sel_slot)                 # [B, k, d]; full scale
# Fix F (2026-04-28): Center M_sel_hidden across selected slots before using as STE
# gradient multiplier. Centering removes the common-mode component (mean slot content)
# that causes all slot_keys to receive identical gradient direction regardless of
# which slot was selected. Only the DIFFERENTIAL contribution of each slot — how it
# deviates from the mean selected slot — provides gradient signal, creating selection
# pressure toward specialization: slot_keys[i] learns to score high for queries where
# slot i is MORE useful than the average.
# Zero in forward (same as Fix E). Backward: d/d(w_gathered[b,j]) = M_sel_centered[b,j].
M_sel_centered = (M_sel_hidden - M_sel_hidden.mean(dim=1, keepdim=True)).detach()  # [B, k, d]
M_sel_hidden = M_sel_hidden + M_sel_centered * (w_gathered - w_gathered.detach())
```

### 4.2  Mathematical argument for correctness

**Forward pass**: `M_sel_centered * (w_gathered - w_gathered.detach()) = M_sel_centered * 0 = 0`.  
No change to the forward computation. ✅

**Backward pass through `w_gathered`**:
```
d(loss)/d(w_gathered[b,j]) = <g_Msel[b,j], M_sel_centered[b,j]>
                            = <g_Msel[b,j], M_sel_hidden[b,j] - mean_j(M_sel_hidden[b,j])>
```

This is the gradient of loss w.r.t. the *relative* weight of slot j in the current selection, as seen by how much slot j's content DIFFERS from the average selected slot.

**Specialization signal created**: suppose slot i has content that, after `slot_to_hidden`, produces a stronger response to current-layer attention patterns than the average slot. Then:
- `M_sel_hidden[b,j=i] - mean > 0` in the directions that matter
- `g_Msel[b,j=i]` points in those directions (gradient of loss pushes M_sel_hidden to better values)
- Inner product > 0 → positive gradient w.r.t. `w_gathered[b,i]` → positive gradient w.r.t. `scores[b,i]` → positive gradient w.r.t. `slot_keys[i]` in the direction of `q[b]`

This creates the update: "when query q[b] is seen, move slot_keys[i] toward q[b] if slot i was more useful than average." This IS a specialization signal.

**Preservation of entropy_aux effect**: the centering only affects the STE gradient path. The entropy_aux_loss gradient still pushes all slot scores to diverge, providing a secondary diversity signal. The two are now complementary rather than competing.

**Magnitude estimate**: at init with σ=0.01 and N=512 slots, the within-batch variance of `slot_to_hidden(slot_i)` across the k=64 selected slots:

```
E[||M_sel_hidden[b,j] - mean_j||^2] ≈ k/(k-1) * σ² * slot_dim * ||slot_to_hidden.weight||²_F / slot_dim
                                     ≈ σ² * ||slot_to_hidden.weight||²_F
                                     ≈ (0.01)² * (128 * 4096) * (0.02)² 
                                     ≈ 1e-4 * 524288 * 4e-4 ≈ 0.021
```

So `||M_sel_centered[b,j]||` ≈ 0.14 per selected slot at init — non-zero but small. As σ grows with training (slots diversify), this magnitude grows, creating stronger specialization pressure. This is **self-amplifying**: early specialization → slot diversity grows → stronger STE centering signal → more specialization.

### 4.3  Alternative Fix F (secondary recommendation)

Add a **slot-diversity auxiliary loss** to directly push slot content diversity:

```python
# In layer.py, after computing M_sel_hidden:
if cfg.return_aux_losses and self.training:
    # Diversity loss: push selected slots to be different from each other.
    # -||M_sel_hidden - mean||_F^2 / (B * k) — maximise variance
    M_sel_mean = M_sel_hidden.mean(dim=1, keepdim=True).detach()
    slot_diversity_loss = -((M_sel_hidden - M_sel_mean) ** 2).mean()
    # weight: 0.001 (same as entropy_aux)
```

This directly addresses the symptom (identical slot content) rather than just neutralizing its effect on the STE gradient. The two fixes are complementary; the centered STE alone is the minimum viable change.

---

## §5  Code Locations

| Fix | File | Lines | Change |
|-----|------|--------|--------|
| Fix F (centered STE) | `src/memory/mem_space/layer.py` | 481–482 | Replace single line 482 with 2 lines |
| (Optional) slot_diversity aux | `src/memory/mem_space/layer.py` | ~490 | 4-line addition after M_sel_hidden computation |
| (Optional) diversity_aux_weight config | `src/memory/mem_space/config.py` | ~68 | 1-line field addition |

The centered STE fix is **3 lines total** (1 added, 1 replaced). No API changes. No new config fields required (entropy_aux_weight already controls aux magnitude; Fix F only changes the STE gradient multiplier, not the loss).

---

## §6  Predicted Outcome

If Fix F breaks gradient symmetry as argued:

- `top1_sim_mean` should **increase** beyond 1/512 within 100–500 training steps as slot_keys[i] acquire preferred-query-direction associations
- `retrieved_norm_mean` should begin to **diverge** from σ × sqrt(slot_dim) as different slots start receiving different write patterns
- `slot_output_gate` should evolve more freely (less BF16 quantization stall) once slot_delta has spatial structure rather than uniform magnitude
- `entropy_aux_loss` value should decrease (routing becomes more structured; lower entropy)
- `load_balance_loss` should initially increase (routing concentrates), then stabilize as slot_keys specialize

**Diagnostic recommendation**: keep GATE_GRAD_DIAG as-is but add:
- `slot_keys_pairwise_cos_sim_mean` (monitor divergence from 0)
- `M_sel_centered_norm_mean` (monitor growing variance of selected slot content)

---

## §7  Hypotheses Ranked by Confidence

| Hypothesis | Description | Assessment |
|-----------|-------------|------------|
| **E (gradient symmetry)** | Near-identical slots → identical STE multiplier → no specialization signal | **PRIMARY ROOT CAUSE** — explains all observations |
| D (load-balance zero gradient) | Switch-style aux has zero grad at uniform fixed point | **CONFIRMED, secondary** — compounded by E |
| B (hidden_to_slot frozen) | Frozen but intentional | **NOT A BUG** |
| A (gradient zero) | slot_keys.grad_norm non-zero | **FALSIFIED** by log data |
| C (STE carries no gradient) | M_sel_hidden_norm > 0 after Fix E | **FALSIFIED** |

---

## §8  Confidence and Caveats

**Confidence**: HIGH (combined direct evidence from log + mechanistic analysis of code)

**One open caveat**: the analysis assumes the gradient accumulated by `slot_keys[i]` averages to near-zero due to random query directions. If the training data has systematic patterns (e.g., similar long-context queries dominate), it is possible for a few slot_keys to accidentally align with those query patterns and break symmetry without Fix F. The σ=0.01 experiment ran for only 204 steps (fwd=200 at the first QUERY_DIAG). It is *theoretically* possible that symmetry would break spontaneously at longer horizons. However, the fix is low-risk (forward-pass zero, no magnitude change) and definitively addresses the identified mechanism.

**Single-point criticism**: the `alpha=0.462891` being constant across all 20 GATE_GRAD_DIAG steps is consistent with BF16 quantization but also consistent with a gradient of order `slot_output_gate.grad ≈ 1e-4` that is too small for BF16 resolution. The GATE_GRAD_DIAG shows `slot_output_gate.grad` is large (~O(1)) so BF16 quantization of the param update is the correct explanation. If the param update is being quantized, a potential mitigation is `slot_output_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))` (keep in float32 even in bf16 training, as is common for scalar gating parameters).
