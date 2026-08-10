# Fix H Proposal — Differentiable Soft Routing Proxy

**Date**: 2026-04-29  
**Author**: researcher subagent  
**Context**: Why Fix G (SKRL) failed to fix K_sel routing degeneracy, and what Fix H must do differently.

---

## 1. Experimental Evidence

Three ablation runs (skrl_weight ∈ {0.001, 0.01, 0.1}) all show identical failure:

| Node | skrl_weight | fwd=200 top1_sim | fwd=200 pairwise_cos | fwd=1000 top1_sim | Outcome |
|------|------------|-----------------|---------------------|------------------|---------|
| 0    | 0.001      | 0.002258        | −0.0023             | 0.002045         | NaN ~step 1900 |
| 1    | 0.01       | ~0.00195        | ≈0                  | ~0.00195         | NaN ~step 2000 |
| 2    | 0.1        | ~0.00195        | ≈0                  | ~0.00195         | NaN ~step 2100 |

`top1_sim_mean` floor = 1/N = 1/512 = 0.00195. None of the runs exceed 0.003 before NaN.

---

## 2. Factual Refutation of Fix G's Premise

Fix G was built on the assumption:

> "slot_keys start near-uniform on the hypersphere (S^127). Mean_pairwise_cos ≈ 0.95 at init."

**This is false.** The logs show `mean_pairwise_cos` was already **≈ 0** at the very first diagnostic (fwd=200, step≈97). For node0/skrl=0.001: `−0.0023` at fwd=200.

This is the mathematically expected result: `slot_keys = nn.Parameter(torch.randn(512, 128) * 0.1)`. After `F.normalize(..., dim=-1)`, these are 512 unit vectors sampled roughly uniformly on S^127. For any two independent unit vectors in R^128:

```
E[cos(k_i, k_j)] = 0,  Var[cos(k_i, k_j)] = 1/128 ≈ 0.0078
```

So mean pairwise cosine ≈ 0 ± 0.088 at initialization. SKRL was solving a problem that **did not exist**.

---

## 3. True Root Cause: The Orthogonality Trap + Broken STE

### 3.1 The Orthogonality Trap

With 512 unit vectors uniformly spread on S^127, the routing scores are:

```
q = F.normalize(Q_sel(pool), dim=-1)        # [B, 128]  — small-init (std=0.02)
logits[b,i] = (q[b] · k_i) * 10.0          # cosine × temperature
```

For a random unit vector `q` and any unit vector `k_i` drawn from a near-uniform distribution on S^127:

```
E[q · k_i] ≈ 0  for all i
```

Therefore `logits[b,i] ≈ 0` for all `i` → `softmax(logits) ≈ 1/N` for all entries → `top1_sim ≈ 1/N = 0.00195`.

This is a **structural fixed point**: as long as slot_keys are approximately uniformly spread on S^127 and Q_sel is in its random-init regime, routing is forced to be near-uniform. Diversifying slot_keys (SKRL) cannot break this because they were already maximally diverse.

### 3.2 Broken STE: The M_sel_centered ≈ 0 Bottleneck

The only gradient path from LM loss to `Q_sel`/`slot_keys` is through Fix F's STE correction:

```python
# layer.py lines ~478-488
w_gathered = ste_weights.gather(1, idx).unsqueeze(-1)         # [B, k, 1]
M_sel_hidden = self.slot_to_hidden(M_sel_slot)                # [B, k, d]
M_sel_centered = (M_sel_hidden - M_sel_hidden.mean(dim=1, keepdim=True)).detach()
M_sel_hidden = M_sel_hidden + M_sel_centered * (w_gathered - w_gathered.detach())
```

The backward gradient to `w_gathered` (and thus to `scores`, `logits`, `Q_sel`) is:

```
d(lm_loss)/d(w_gathered) = d(lm_loss)/d(M_sel_hidden) · M_sel_centered
```

But **M_sel_centered ≈ 0** throughout training:

1. All 512 slots are initialized from `hidden_pool_mean + N(0, 1.0)` noise.
2. The `hidden_pool_mean` term is identical for all slots in a given batch element.
3. After `slot_to_hidden` projection: `M_sel_hidden[b, j] ≈ W · (hidden_pool_mean + noise_j)`.
4. The centering operation subtracts the mean across j, which cancels the dominant `W · hidden_pool_mean` term.
5. What remains: `M_sel_centered[b, j] = W · (noise_j - mean_k(noise_k))` — noise centered across the top-k selected slots.

With `slot_init_noise=1.0` and `slot_to_hidden` weights also at std≈0.02, the residual `||M_sel_centered||` is tiny — effectively providing near-zero gradient signal to `Q_sel`.

**This is the chicken-and-egg deadlock:**
- Q_sel cannot learn to route diversely because M_sel_centered ≈ 0 (all slots have similar content).
- Slots cannot become diverse because routing is near-uniform (all slots receive equal update mass).
- SKRL correctly diversifies slot_key geometry, but the bottleneck is slot **content** not slot key **geometry**.

### 3.3 Secondary Problem: NaN Explosion via 32-Layer Compounding Writes

With `shared_memory_bank=True`, all 32 decoder layers write to the same 512-slot bank. With uniform routing selecting the same ~64 slots every step, the effective per-step EMA for frequently-selected slots is:

```
effective_beta ≈ 1 - (1 - beta_per_layer)^32 = 1 - (0.85)^32 ≈ 0.993
```

Each chunk processes ~128 tokens × 32 layers × beta=0.993 compounding → slots saturate to the current hidden-state representation with very high norm within a single chunk. At chunk boundary, `_reset_banks` re-initializes from the current hidden pool (which has already been influenced by the exploded slots) → growing explosion cycle → bf16 overflow → NaN.

---

## 4. Fix H: Differentiable Soft Routing Proxy

### 4.1 Core Idea

Replace the M_sel_centered STE (which provides near-zero gradient) with a **soft routing proxy** that:
1. Has **non-zero gradient** from LM loss to Q_sel regardless of slot content diversity.
2. Does **not change the forward values** (hard-selected slot content is still used).
3. Does not require slot content to be diverse to work.

### 4.2 Implementation in `layer.py`

Replace lines ~478-488 (the Fix F STE block) with:

```python
# Fix H: Differentiable soft routing proxy
# Hard forward: gather the k selected slots (no gradient through selection)
idx_exp = idx.unsqueeze(-1).expand(-1, -1, slot_dim)   # [B, k, slot_dim]
M_sel_slot_hard = slots.gather(1, idx_exp)              # [B, k, slot_dim]
M_sel_hard = self.slot_to_hidden(M_sel_slot_hard)       # [B, k, d]

# Soft proxy: differentiable weighted sum over ALL slots using softmax scores
# scores: [B, N], slots: [B, N, slot_dim]
# slots.detach() prevents gradients from flowing INTO the slot bank through this path
# (we only want gradient to flow TOWARD Q_sel / slot_keys, not change slot values here)
M_sel_slot_soft = torch.einsum(
    "bn,bnd->bd",
    scores,
    slots.detach()
)                                                        # [B, slot_dim]

# Expand soft proxy to match the k-slot shape
M_sel_soft = self.slot_to_hidden(
    M_sel_slot_soft.unsqueeze(1).expand(-1, self.cfg.top_k, -1)
)                                                        # [B, k, d]

# Fix H STE: forward uses hard values (correct slot content)
#            backward gradient flows through soft proxy → scores → Q_sel / slot_keys
# Value: M_sel_hard (uses actual selected slot content)
# Gradient: passes through M_sel_soft (differentiable in scores)
M_sel_hidden = M_sel_hard.detach() + (M_sel_soft - M_sel_soft.detach())
```

**Gradient flow analysis:**

```
d(lm_loss)/d(scores[b,i]) 
    = d(lm_loss)/d(M_sel_soft[b,:]) 
    · d(M_sel_soft[b,:])/d(scores[b,i])
    = d(lm_loss)/d(M_sel_soft) 
    · slot_to_hidden(slots[b,i])          ← O(1), non-zero
```

This path is valid as long as `d(lm_loss)/d(M_sel_soft) ≠ 0`, which holds whenever the model uses the slot information at all — even if all slots have similar content. The gradient magnitude is `||slot_to_hidden(slots[b,i])||` ≈ `||W_s · slots[b,i]||` which is O(1) for reasonable slot norms.

**Why this differs from Fix F:**

Fix F's gradient to `scores` was: `d(lm_loss)/d(M_sel_hidden) · M_sel_centered`. Here `M_sel_centered` is near-zero.

Fix H's gradient to `scores` is: `d(lm_loss)/d(M_sel_soft) · slot_to_hidden(slots[b,i])`. Here `slot_to_hidden(slots[b,i])` is O(1) and non-zero (it's an actual slot representation, not a centered-difference).

### 4.3 Slot Norm Clipping in `memory_bank.py`

Add norm clipping inside `write()` to prevent NaN explosion:

```python
def write(self, idx, new_repr, gate):
    """Write new_repr into slots at idx via EMA gated by gate."""
    B, k, slot_dim = new_repr.shape
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, slot_dim)
    current = self.slots.gather(1, idx_exp)              # [B, k, slot_dim]
    
    if isinstance(gate, torch.Tensor):
        gate_t = gate.view(B, 1, 1)
    else:
        gate_t = float(gate)
    
    updated = (1.0 - gate_t) * current + gate_t * new_repr
    
    # Fix H: Slot norm clipping — prevents 32-layer compounding writes from
    # causing bf16 overflow. Target norm: sqrt(slot_dim) ≈ typical Llama
    # hidden state norm in bf16.
    max_norm = math.sqrt(float(slot_dim)) * 2.0          # e.g. 90.5 for slot_dim=2048
    slot_norms = updated.norm(dim=-1, keepdim=True)       # [B, k, 1]
    scale = (slot_norms.clamp(max=max_norm) / slot_norms.clamp(min=1e-6))
    updated = updated * scale
    
    self.slots = self.slots.scatter(1, idx_exp, updated)
```

### 4.4 Optional: Disable SKRL (Saves ~0.5% Forward Time)

SKRL (Fix G) is now confirmed ineffective. Setting `skrl_weight=0.0` in the config removes the useless computation. The slot_key_diversity_loss can remain in the code for backward compat but will not be called.

---

## 5. Expected Behavior After Fix H

| Metric | Before Fix H | Expected After Fix H |
|--------|-------------|----------------------|
| top1_sim_mean at step 200 | 0.00195 (floor) | > 0.005 within 500 steps |
| top1_sim_mean at step 1000 | 0.00195 (floor) | > 0.01 |
| mean_pairwise_cos | ≈0 (correct, no change) | ≈0 (no change) |
| retrieved_norm_mean | Explodes to NaN | Stays below max_norm |
| Training completes | No (NaN ~step 1900) | Yes |
| PPL improvement from baseline | None | TBD — depends on routing quality |

The key signal to watch is `top1_sim_mean`. If it rises above the 1/N floor and continues increasing, Fix H is working. If it stays at 0.00195, there is a deeper issue.

---

## 6. Files to Modify

1. **`src/memory/mem_space/layer.py`** — Replace Fix F STE block (~lines 478-488) with Fix H soft proxy.
2. **`src/memory/mem_space/memory_bank.py`** — Add slot norm clipping in `write()`.
3. **`src/memory/mem_space/config.py`** — Set `skrl_weight=0.0` default (SKRL disabled).

No changes needed to `selector.py` — the TopKSelector forward is correct as-is.

---

## 7. Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| `slots.detach()` in soft proxy prevents slot content from improving | Low | slots still updated via writeback EMA; we detach only to prevent double-counting in the STE |
| Gradient through `slot_to_hidden` (in soft proxy path) causes instability | Medium | slot_to_hidden is already frozen (`hidden_to_slot_frozen=True`); may need to allow it to train for Fix H to work fully — check config |
| Slot norm clipping interferes with writeback BPTT | Low | clipping is post-write; gradient still flows through the unclipped path |
| Fix H insufficient if slot content truly never diversifies | Medium | Monitor slot content diversity (add `slot_content_var` diagnostic); if diversification requires Fix H + additional measures, implement Fix I |

**Important note on `hidden_to_slot_frozen`**: The soft proxy path routes gradient through `slot_to_hidden` (which is `hidden_to_slot` in the inverse direction). Check whether `hidden_to_slot_frozen=True` also freezes `slot_to_hidden` — if so, Fix H may need to set `hidden_to_slot_frozen=False` to allow the soft proxy gradient to fully propagate.

---

## 8. Summary

**SKRL (Fix G) failed because it solved the wrong problem.** Slot keys were already geometrically diverse at initialization (pairwise cosine ≈ 0). The actual blocker is:

1. **Orthogonality trap**: Uniform spread of 512 slot keys on S^127 means any random query has near-zero dot product with all keys → uniform routing. This is a property of the space, not a fixable geometry issue.

2. **Broken STE**: The only gradient path to Q_sel (Fix F's M_sel_centered) is near-zero because all slots are initialized to similar content. No matter how diverse the keys become, the STE cannot teach Q_sel to distinguish them if the slots themselves carry identical information.

**Fix H replaces the broken STE** with a soft routing proxy that has an O(1) non-zero gradient regardless of slot content diversity, combined with slot norm clipping to prevent the 32-layer compounding write explosion.
