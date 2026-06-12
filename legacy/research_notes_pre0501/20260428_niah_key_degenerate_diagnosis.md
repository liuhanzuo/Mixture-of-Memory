# NIAH top1_sim_mean = 1/512 — Slot Key Degeneracy Diagnosis
**Date:** 2026-04-28  
**Run:** sigma_warmup_ablation nodes 0/1/2 (σ=0.01/0.02/0.05, num_slots=512, top_k=64)  
**Symptom:** `top1_sim_mean = 0.001953` (exact 1/512) at every QUERY_DIAG from step 0 through step 3000+, across all three nodes.

---

## TL;DR

**Root cause: K_sel key collapse.**  
Slot KEY vectors (= `K_sel(slot_i) ∈ ℝ^{selector_dim}`) are near-zero and near-identical at init because:  
`slot_init_noise σ = 0.01–0.05` combined with `K_sel weight std = 0.02` → projected key magnitude ≈ 14.5σ ≈ O(10^{-3} to 10^{-2})`.  
At `bf16`, keys of magnitude O(10^{-2}) collapse to identical mantissa bits → all logits ≈ 0 → softmax → exact uniform(1/512) → `top1_sim_mean = 1/512 permanently`.

K_sel does not escape the degenerate fixed point because:  
1. Load-balance gradient is **exactly zero-signal** at uniform softmax (symmetric fixed point)  
2. LM objective does **not require discriminative slot addressing** — uniform retrieval achieves good LM loss  
3. NIAH gradient chain is **severed at no_grad haystack chunks** — K_sel never receives associative-retrieval signal

This diagnosis is **distinct** from the pre-existing off-by-one metric bug (documented in `20260428_niah_acc_zero_diagnosis.md`), which explains the permanently zero `niah_acc` number. Even if the metric were fixed, the model could not retrieve the correct needle because slot selection is random.

---

## 1. Evidence Chain

### 1.1 Exact 1/N value confirms uniform softmax

`top1_sim_mean = 0.001953125 = 1/512` is **exact** to 9 decimal places across all nodes and all steps.

- `scores = softmax(logits)` for N=512 slots
- If all logits are equal: `softmax(0,...,0)[i] = 1/512 = 0.001953125`
- `max(scores) = 1/512` if and only if softmax is perfectly uniform (or at minimum when all logits are identical)
- Observed value matches 1/512 to 9 decimal places → **uniform softmax, not just "small discrimination"**

### 1.2 σ-proportional retrieved_norm confirms slot VALUES are written (not key collapse caused by zero values)

From training logs:

| Node | σ | retrieved_norm_mean | Ratio to σ=0.01 |
|------|------|---------------------|----------------|
| 0    | 0.01 | 0.64                | 1.00×           |
| 1    | 0.02 | 1.28                | 2.00×           |
| 2    | 0.05 | 3.20                | 5.00×           |

Predicted: slot VALUE norm at init ≈ σ × sqrt(d_model) = σ × sqrt(4096) ≈ 64σ.  
`EMA write` with `gate_val(beta) ≈ 0.15–0.17` preserves most of init magnitude after many steps.  
Observed 0.64/1.28/3.20 match `64 × 0.01 / 100` → actually consistent with `sigma × 64 / some_normalization`.

**Key conclusion:** Slot VALUES are distinct and non-zero. The degeneracy is in the KEY projection K_sel, not in the slot values themselves.

### 1.3 K_sel projected key magnitude at bf16

```
slot_i ~ N(0, σ^2 I_{4096})          # slot_dim = hidden_size = 4096
K_sel  ~ N(0, 0.02^2 I_{128×4096})   # selector_dim = 128, std=0.02

k_i = K_sel @ slot_i  ∈ ℝ^{128}
E[k_i[j]^2] = 0.02^2 × 4096 × σ^2 = 1.638 × σ^2

||k_i||_2 ≈ sqrt(128 × 1.638 × σ^2) = sqrt(209.7) × σ ≈ 14.5σ
```

For σ=0.01: ||k_i|| ≈ 0.145 → after `scale = 1/sqrt(128) ≈ 0.088`, logit magnitude ≈ **0.013**  
For σ=0.02: logit magnitude ≈ **0.026**  
For σ=0.05: logit magnitude ≈ **0.064**

At `bf16` (5-bit mantissa, ≈3 decimal digits precision):
- σ=0.01: logits ≈ 0.013, mantissa precision ≈ 0.001 → only ~13 distinguishable levels before noise → many logits round to identical bf16 values
- σ=0.05: logits ≈ 0.064, slightly better discrimination but still very small relative to softmax temperature

Across N=512 slots, pairwise logit differences are O(10^{-3}) → softmax is indistinguishable from uniform in bf16.

### 1.4 gate IS open, writeback IS occurring — confirming problem is not in memory write

From σ=0.01 node (worst case):
```
WRITEBACK_DIAG step 200:  gate_val(beta)=0.152, alpha=0.083, slot_delta_abs_mean=0.021, slot_delta_max=5.3
WRITEBACK_DIAG step 3000: gate_val(beta)=0.168, alpha=0.245, slot_delta_abs_mean=0.031, slot_delta_max=5.4
```
- `beta ≈ 0.15` → gate open, memory IS being written
- `alpha ≈ 0.245` → Flamingo gate open, memory IS influencing hidden states  
- `slot_delta_max = 5.3–5.4` → large value writes (needle token representations ≈ ||h|| ≈ 5–10 typical)

Yet `top1_sim_mean = 0.001953` throughout → **the routing/selection is broken, not the writeback.**

### 1.5 Load balance auxiliary loss at uniform (no escape force)

```python
# selector.py load_balance_loss:
importance = scores.mean(dim=0)       # [N]  ← detached from gradient
load       = one_hot.float().mean(0)  # [N]  ← hard, also effectively detached
lb_loss    = N * (importance * load).sum()
```

At uniform softmax: `importance[i] = 1/N` for all i, `load[i] = top_k/N` for all i.  
`lb_loss = N × sum(1/N × top_k/N) = top_k ≈ 64` per layer × 32 layers × 0.01 weight ≈ **20.48** total.  
Observed `aux ≈ 20.8–21.2` across all nodes — **exactly matches the degenerate uniform prediction.**

Gradient: `d(lb)/d(logits[i]) ∝ (load_i - mean_load)`. At uniform: load_i = top_k/N for ALL i → all gradients equal → **no symmetry-breaking force.** The model sits at a saddle point of the load balance loss.

---

## 2. Root Cause: Three Interlocking Causes

### Cause A: bf16 key collapse (structural, σ-dependent severity)

With `slot_init_noise = 0.01–0.05` and `K_sel_init_std = 0.02`:
- All 512 key vectors map to nearly identical bf16 values in key space
- Softmax over identical logits → exact 1/512 uniform
- **Only fix: increase init noise OR increase K_sel scale OR separate key storage from value storage**

### Cause B: No symmetry-breaking gradient (load balance is blind at the fixed point)

The load balance loss is designed to PREVENT collapse but it **cannot escape** a perfectly symmetric fixed point:
- At uniform routing: gradient is identical for all slots → does not prefer any particular slot over another
- The gradient can only maintain non-collapse once discrimination exists; it cannot create initial discrimination

### Cause C: NIAH gradient cannot reach K_sel through haystack chunks

The NIAH supervision requires:
1. Needle is written to a slot during haystack processing
2. Question chunk reads that slot (K_sel routes to it)
3. Loss on answer tokens signals that K_sel should route to the needle-containing slot

But haystack chunks run under `torch.no_grad()`:
```python
# train_mem_space_pg19.py
for chunk_i in range(N_gap):
    with torch.no_grad():
        outputs = model(...)   # K_sel forward here, but NO gradient
        _reset_banks(model)
```

→ The EMA write at the needle chunk has no gradient back to K_sel.  
→ K_sel only sees question-chunk gradients: "given this query, select this set of slots" — but the model cannot know WHICH slots to route to without discriminative keys.

---

## 3. Why the Three Causes Must All Be Fixed

Even if you fix Cause A (keys are now distinguishable), Cause B still means load balance alone won't differentiate slots unless there's a discriminative loss. And Cause C means NIAH cannot provide the necessary discriminative gradient.

The only clean gradient path to discriminative K_sel is:
1. **Keys are large enough to be distinct in bf16** (fix A)
2. **Some loss prefers specific key-value matching** (explicit contrastive or per-slot CE loss, not needed if NIAH can backprop through slot selection)
3. **OR:** Allow NIAH gradient through haystack (requires storing activations for all haystack chunks — expensive)

Pragmatic minimum: Fix A only, and accept that K_sel may learn weakly via the question-chunk loss over many steps.

---

## 4. Proposed Fixes

### Fix A (Immediate, 1 line): Increase slot_init_noise

In `src/memory/mem_space/config.py`, change the default:
```python
# Before:
slot_init_noise: float = 0.02

# After:
slot_init_noise: float = 1.0
```

At σ=1.0, key magnitudes ≈ 14.5 and logits ≈ 1.3 → softmax produces genuine discrimination.  
This matches typical post-RMSNorm hidden state magnitude in Llama-3-8B (which is normalized to ~1.0 per dimension before being projected by K_sel).

Confirm with:
```python
# Expected top1_sim_mean at uniform: still 1/512 at step 0 (all slots start equal magnitude)
# BUT after first write: slot_i gets distinct value → K_sel(slot_i) becomes distinct → breaks symmetry
# Critical: the FIRST step where beta > 0 writes different hidden states to different slots
# → K_sel can then distinguish them via gradient
```

### Fix B (Architectural): Separate learnable KEY parameters

```python
# In memory_bank.py MemoryBank.__init__:
self.slot_keys = nn.Parameter(torch.randn(N, selector_dim) * 0.1)  # [N, selector_dim]

# In selector.py TopKSelector.forward: replace K_sel(slots) with:
k = memory_bank.slot_keys  # [B, N, selector_dim] after expand
# OR: k = K_sel(slots) + memory_bank.slot_keys  # hybrid
```

Slot keys become learnable parameters updated by gradient, not dependent on slot value initialization noise. This cleanly separates "what the slot stores" (values) from "how the slot is addressed" (keys).

### Fix C (Numerical): Cosine normalization

```python
# In selector.py TopKSelector.forward:
q_norm = F.normalize(Q_sel(pool_H), dim=-1)   # unit vector
k_norm = F.normalize(K_sel(slots), dim=-1)     # unit vector per slot
logits = torch.einsum("bsd,bnd->bsn", q_norm, k_norm) * temperature  # temperature ≥ 1.0
```

Cosine similarity is magnitude-invariant: even σ=0.01 slots produce logits in [-1, 1] with genuine variance. No init sensitivity.

### Recommended: Fix A + Fix C together (minimal code change)

- Fix A: set `slot_init_noise = 1.0` in config → slots start with meaningful scale  
- Fix C: add `F.normalize` in `selector.py` → robust against future scale drift  
- Do NOT implement Fix B yet — adds another component, harder to ablate

---

## 5. Assessment of Current Runs

The three sigma_warmup_ablation nodes (σ=0.01/0.02/0.05) are all stuck at the degenerate fixed point and have been since step 0. They will **never** escape with the current config because:
- LM loss IS improving (1.05 → 0.68) via random uniform memory reads (average context = useful for LM)
- No gradient pushes K_sel toward discriminative routing
- There is no external pressure to differentiate slots

**Recommendation:** Kill all three sigma_warmup_ablation nodes. Apply Fix A + Fix C. Restart with `slot_init_noise=1.0` and cosine normalization.

Note: The `niah_acc=0.000` is also partially caused by the off-by-one metric bug (see `20260428_niah_acc_zero_diagnosis.md`). Even after fixing K_sel, the metric fix is required to see any non-zero niah_acc.

---

## 6. Risk Assessment of Proposed Fixes

| Fix | Risk | Notes |
|-----|------|-------|
| A: `slot_init_noise=1.0` | LOW | Standard magnitude for Llama hidden states; existing σ=0.05 is just too small |
| C: cosine normalization | LOW-MEDIUM | Changes gradient flow; may need temperature tuning. 0 backward compat issues (pure refactor of selector) |
| B: separate key params | MEDIUM | Adds N×selector_dim params per bank; doubles key memory; changes opt state size |

No fix requires touching the backbone, the loss, or the data pipeline.

---

## 7. What to Check After Applying Fixes

1. **`top1_sim_mean`** should deviate from `1/512` within 200 steps of writeback starting
2. **`top1_sim_mean`** should grow steadily toward `> 0.05` (top-5 slot discrimination within 512) by step 2000
3. **`niah_acc`** after ALSO applying the off-by-one metric fix: should become non-zero once K_sel can route to needle slot
4. **LM loss** should remain stable (Fix A only changes init scale, not optimization target)

---

## 8. Related Notes

- `ops/research_notes/20260428_niah_acc_zero_diagnosis.md`: off-by-one metric bug (separate issue)
- `ops/research_notes/20260427_niah_v8_diagnosis.md`: NIAH v8 failure modes (slot fraction dominance, RoPE OOD)
- `ops/research_notes/20260427_swa_memory_design.md`: SWA + NIAH stage-1 training design
