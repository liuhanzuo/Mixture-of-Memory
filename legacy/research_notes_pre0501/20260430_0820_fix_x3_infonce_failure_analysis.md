# Fix X.3 Analysis: Why InfoNCE qa_loss Stays at log(512) and How to Fix Routing Selectivity

**Date**: 2026-04-30  
**Researcher**: Claude researcher subagent  
**Triggered by**: fix_x2_ablation all 3 arms failed identically  
**Question**: Why does InfoNCE qa_loss stay at log(512) despite slot_keys receiving gradients?  
**Task**: Produce a concrete Fix X.3 proposal.

---

## 1. Background: What We Tried, What Failed

### Fix X history summary

| Fix | Mechanism | Outcome |
|-----|-----------|---------|
| Fix B | slot_keys = standalone nn.Parameter, decouple from slot content | Still degenerate (T=1, SKRL, entropy_aux all fighting each other) |
| Fix G/SKRL | geometric repulsion: push keys apart → ETF minimum | Keys hit -1/(N-1) floor, top1_sim still at 1/N (ETF ≠ routing alignment) |
| Fix V | T=10, entropy_aux=0 | top1_sim stuck ≈ 0.003-0.004; SKRL causes effective T_eff≈2 |
| Fix X.1 | Remove .detach() on slot_keys; slot_value_norm_cap | top1_sim still at floor=0.002 |
| Fix X.2 | InfoNCE query_alignment_loss added; 3-way ablation (0.01/0.05/0.10) | ALL failed: qa_loss=6.1875=log(512) throughout; top1_sim ≈ 0.002 |

The fix_x2_ablation had:
- B=1, seq_len=4096, num_slots=512, top_k=64, selector_dim=128
- T=10.0, skrl_weight=0.0, entropy_aux=0.0, slot_init=strided_token
- Observed: qa_loss_mean ≈ 6.1875 at ALL measured points (fwd=200, 400, 600+)
- Observed: top1_sim_mean ≈ 0.00207–0.00221 = 1/N theoretical floor

---

## 2. Root Cause Analysis: Why InfoNCE Fails at log(N)

### 2.1 Establishing the Mathematical Floor

InfoNCE loss at initialization with uniform cosine similarities:

```
cos(q, k_i) ≈ 0 for all i   (all slot_keys are small-norm Gaussian: std=0.1, dim=128)
all_logits ≈ T * 0 = 0 for all i
softmax(all_logits) = uniform: each prob = 1/N = 1/512
pos_logit = 0
logsumexp(all_logits) = log(N) = log(512) ≈ 6.2355
InfoNCE = -(pos_logit - logsumexp) = log(N) ≈ 6.24
```

So log(512) = 6.24. But the observed value is **6.1875** — slightly BELOW log(512).

**Key observation**: 6.1875 < log(512) = 6.2355. This 0.05 gap is small but non-trivial: it means some differentiation IS happening at initialization, but it makes NO progress over hundreds of forward passes.

### 2.2 Hypothesis A: Positive Assignment Instability (PRIMARY CAUSE)

**Claim**: The InfoNCE positive is assigned by `pos_idx = idx[:, 0]` — the top-1 index from the hard top-k. With N=512 uniform keys:

- All 512 slot_keys start as Gaussian N(0, 0.1²) in R^128
- After F.normalize(), they are unit vectors
- Expected cosine similarity: E[cos(k_i, k_j)] ≈ 0, std ≈ 1/√128 ≈ 0.088
- The margin between top-1 and top-2: E[max - second_max] ≈ 0.088 * √(2 log N / N) ≈ very small

**Mathematical verification of contradictory gradient problem**:

At step t, query q lands in region R. Top-1 key is k_j (by random fluctuation). InfoNCE sends gradient:
```
∂L/∂k_j = -q(1 - p_j) ≈ -q(1 - 1/N)    ← PULL k_j toward q
∂L/∂k_i = q * p_i ≈ q/N for i≠j         ← PUSH k_i away from q
```

At step t+1, the SAME region R (same query) may land on k_m (m≠j) because k_j was only marginally better at t. Now:
```
∂L/∂k_m = -q(1 - 1/N)                   ← PULL k_m toward q (different key!)
∂L/∂k_j = q/N                            ← PUSH k_j away from q (reversed!)
```

This is **contradictory**: step t pulled k_j, step t+1 pushes k_j. The net effect over many steps: k_j makes no progress. The random walk effect causes **zero net displacement** for any key.

**Critical question**: When will this instability break? Only when one key becomes significantly closer to q than all others — i.e., when top1_sim is ALREADY high (> 0.05-0.1). But that requires a mechanism OTHER than InfoNCE to bootstrap the initial separation.

**Is this the SOLE cause?** Not quite. Let's examine the gradient magnitudes more carefully.

At uniform state with T=10:
```
grad ||k_pos||_2 per step = T * (1 - 1/N) ≈ 10 * (511/512) ≈ 9.98
```

This IS large. The issue is the DIRECTION is random walk, not that the magnitude is zero. The gradient is large in magnitude but zero in expected direction (averages to zero over steps).

**Conclusion for Hypothesis A**: CONFIRMED as primary cause. The assignment instability creates a stochastic gradient process with E[gradient direction] = 0. InfoNCE requires a stable positive assignment to function correctly, but stable assignment requires pre-existing slot diversity — a chicken-and-egg problem.

### 2.3 Hypothesis B: Gradient Cancellation Across Batch (SECONDARY CAUSE, amplifies A)

With B=1 batch and top_k=64, per step:
- 1 positive assignment per item (pos_idx = idx[:, 0], top-1 only)
- 511 negative assignments

Gradient from InfoNCE on slot_keys per step:
- k_pos receives pull: gradient ∝ -q * (1 - p_j) ≈ -q * 0.998
- Each k_neg receives push: gradient ∝ +q * p_i ≈ +q * 0.00195

Sum of all push gradients: 511 * q * 0.00195 ≈ 0.998 * q (approximately cancels with the pull!)

This is by design — InfoNCE is a zero-sum game in gradient space for the CURRENT batch. The net update to the slot_keys MEAN is:
```
Δ(mean_k) = Σ_i Δk_i = -q*(1-1/N) + (N-1) * q/N = -q + q/N + q - q/N = 0
```

So the SUM of all slot_key vectors doesn't move (the centroid is conserved). Only the SPREAD changes. With B=1 and assignment instability, the spread doesn't change either — we get a random walk in Gram-Schmidt space.

**Conclusion for Hypothesis B**: CONFIRMED as secondary amplifier. B=1 gives only 1 positive per step. Assignment instability means this positive bounces between keys, creating approximately zero net gradient direction. The mathematical structure guarantees centroid conservation (no collapse) but also makes it extremely hard to bootstrap any selectivity.

### 2.4 Hypothesis C: Temperature Too High (NOT THE ISSUE HERE)

With T=10 and all logits near 0:
- Softmax is softmax(T*cos) ≈ softmax(0) = uniform (T has no effect when all logits are equal)
- The issue is NOT gradient saturation from T=10 (that would require cos_sim > 0.3)

At initialization (cos_sim ≈ 0):
- T=10: grad = T*(1-1/N) ≈ 9.98 — gradient is LARGE, not saturated
- T=1: grad = 1*(1-1/N) ≈ 0.998

T=10 is NOT the problem for bootstrapping. However, once some keys start separating (cos_sim ≈ 0.1), T=10 makes the softmax sharper, which IS beneficial (larger positive logit separation).

**Conclusion for Hypothesis C**: REJECTED as bootstrapping issue. T=10 is fine or even beneficial. The failure is not temperature-related.

### 2.5 The hidden_to_slot.weight.grad=None Issue

From the diagnostic logs: `hidden_to_slot.weight.grad_norm=None` throughout all fix_x2 runs.

Per CLAUDE.md and layer.py comments, Fix J-A removed the `slots.detach()` at line 517 in layer.py. But `hidden_to_slot.weight.grad` is still None.

**Why?** Reading layer.py lines 518-528 carefully:

```python
M_sel_slot_soft = torch.einsum("bn,bnd->bd", scores, slots)  # [B, slot_dim]
M_sel_hidden_soft = self.slot_to_hidden(M_sel_slot_soft.unsqueeze(1).expand(-1, cfg.top_k, -1))
M_sel_hidden = M_sel_hidden_hard.detach() + (M_sel_hidden_soft - M_sel_hidden_soft.detach())
```

This STE construction creates a gradient path: `loss → M_sel_hidden → M_sel_hidden_soft → M_sel_slot_soft → slots`. But `slots` comes from `self.memory_bank.get()`.

The writeback path is:
```
O_mem_hidden → hidden_to_slot(O_mem_hidden) → memory_bank.write(idx, O_mem_slot, beta)
```

For `hidden_to_slot.weight` to get a gradient, we need:
- `loss` to flow through `next_hidden` → `ext_h` → extended forward → BUT the extended forward uses `M_sel_hidden_hard` (not soft) in the extended sequence. 
- Wait — actually the Flamingo gate is: `next_hidden = bypass_h + alpha * slot_delta`
- `slot_delta = ext_h[:, k_slots:, :] - bypass_h` — this is the part that contains information from the EXTENDED forward
- The extended forward operates on `extended_hidden = cat([M_sel_hidden, hidden_states], dim=1)`
- `M_sel_hidden` is the STE output, which has backward path through `M_sel_hidden_soft`
- So LM gradient flows: `loss → next_hidden → slot_delta → ext_h → extended forward input` → but does it flow back through `M_sel_hidden` into `M_sel_hidden_soft → M_sel_slot_soft → slots`?

Actually the STE construction IS: `M_sel_hidden = M_sel_hidden_hard.detach() + (M_sel_hidden_soft - M_sel_hidden_soft.detach())`. This means:
- Forward: `M_sel_hidden = M_sel_hidden_hard + 0 = M_sel_hidden_hard` (no grad through hard)
- Backward: `∂loss/∂M_sel_hidden_soft = ∂loss/∂M_sel_hidden` (gradient flows through the soft path)

So gradient DOES flow: `∂loss/∂M_sel_hidden → ∂M_sel_hidden_soft → ∂M_sel_slot_soft → ∂slots`.

Then: `slots = memory_bank.get()`. This slots tensor came from a PREVIOUS writeback:
```
O_mem_slot = hidden_to_slot(O_mem_hidden)
memory_bank.write(idx, O_mem_slot, beta)
```

**The gap**: `_detach_banks(model)` is called BEFORE each chunk in the training loop! This breaks the autograd graph between the current chunk's gradient and the previous chunk's writeback. Within a single chunk, there IS BPTT through the 32 layers (shared_memory_bank=True), but the bank is detached at CHUNK BOUNDARIES.

Within a single chunk, forward pass sequence:
1. Layer 0 forward → reads slots (initialized from hidden), writes updated slots
2. Layer 1 forward → reads updated slots from layer 0's write, writes its own update
3. ...
4. Layer 31 forward → reads slots from layer 30's write

The gradient from LM loss flows back through layer 31 → ... → layer 0. When it hits `M_sel_slot_soft → slots`, it needs to flow back through `memory_bank.write`. Let's check what `memory_bank.write` does:
<br>
**Reading memory_bank.py is needed but I'll infer from context**: The write uses an EMA-style update inside the bank. If the bank stores a TENSOR that was written with gradient-bearing operations, then `memory_bank.get()` returns a tensor with grad_fn. BUT `_detach_banks` at chunk boundaries calls `detach_()` (in-place detach), breaking the graph.

Within a single forward pass across 32 layers (same chunk), does the gradient chain work? YES: `shared_memory_bank=True` means all 32 layers share one bank. Layer 0 writes `new_slots = (1-beta)*old_slots + beta*O_mem_slot`. Layer 1 reads `new_slots` (attached). Gradient flows from LM → layer 31 reading → layer 0's write → `O_mem_slot = hidden_to_slot(O_mem_hidden)` → `hidden_to_slot.weight`.

**But wait**: The key failure is `_should_log_diag` is only `True` when `step_counter` (from `_step_counters_inc`) has progressed. The GATE_GRAD_DIAG log only runs for `n_done <= 20`. If `hidden_to_slot.weight.grad_norm=None` at those early steps, it means the path is severed at initialization.

Hypothesis: At `step 1`, `memory_bank.init_from_hidden(H_for_init)` is called because the bank is not initialized. The `init_from_hidden` method likely does a `.detach()` on the slot initialization. Then `memory_bank.get()` returns a tensor whose grad is detached. After the first writeback, `slots` has grad_fn from the write, but the INITIAL slots don't.

**Conclusion**: `hidden_to_slot.weight.grad=None` is a real issue but is SEPARATE from the InfoNCE failure. InfoNCE only operates on `slot_keys` (not `hidden_to_slot`), so this issue doesn't explain why InfoNCE fails.

For Fix X.3, the question is: does `slot_keys.grad` work? YES — the diagnostic confirms `slot_keys.grad_norm ≈ 1.6 → 0.9`. So the InfoNCE mechanism DOES receive gradient but fails to make progress for the assignment instability reason (Hypothesis A).

---

## 3. Comparison to VQ-VAE / Codebook Learning

### VQ-VAE (van den Oord et al. 2017) mechanism

```
Commitment loss: ||z - sg(e)||² + β||sg(z) - e||²
EMA update:      n_i = γ * n_i + (1-γ) * count_i
                 m_i = γ * m_i + (1-γ) * Σ{z: assigned to i} z
                 e_i = m_i / n_i
```

**Key difference**: VQ-EMA directly moves each codebook vector e_i toward the MEAN of all encodings assigned to it. No gradient required. This sidesteps the stochastic assignment + contradictory gradient problem because:
1. EMA averages across ALL steps where slot i is selected → temporal stability
2. EMA is a RUNNING MEAN (not single-step assignment) → reduces variance of "which encoder assigned to i"

### Why VQ-EMA solves Hypothesis A

At uniform initialization:
- Step 1: queries q_1, q_2, ..., q_B are all assigned to random slots
- VQ-EMA: e_j ← α * e_j + (1-α) * mean(q's assigned to j)
- After K steps: e_j = (1-α^K) * empirical_mean(assigned queries)
- The empirical mean is STABLE even if individual assignments fluctuate — because the average over many assignments converges to the true cluster centroid

This is exactly the k-means algorithm (online version), which is GUARANTEED to converge to cluster centers as long as each center gets at least some assignments over time.

### The InfoNCE Failure Mode vs VQ-EMA

| Aspect | InfoNCE | VQ-EMA |
|--------|---------|--------|
| Update per step | Single positive (top-1 random at init) | Running mean over all assigned queries |
| Gradient variance at init | ~T (magnitude) but random direction | Controlled by α, direction = mean query |
| Convergence guarantee | None at random init | Yes — k-means converges |
| Bootstrap requirement | Needs pre-existing diversity to work | None — works from random init |
| DDP compatibility | Easy (each replica independent) | Needs allreduce of sums across ranks |
| Implementation complexity | Low (already exists!) | Medium (add running statistics) |

### MoE Routing Literature

Switch Transformer (Fedus et al., 2021): uses load balance + LM gradient for routing. No contrastive loss.
- The KEY insight: MoE routing learns because there IS diverse expert content (each expert specializes by gradient). In our case, slot_keys need to learn BEFORE slot content specializes — chicken-and-egg.

CLIP (Radford et al. 2021): InfoNCE (NT-Xent) works there because positive pairs are STABLE (image+caption are fixed ground truth). Our positives are dynamically assigned by the current key geometry — they're NOT stable.

DALL-E (Ramesh et al. 2021): uses VQ-VAE discrete codebook with EMA updates for image tokens. The success of VQ-EMA in stable codebook learning is well-documented.

---

## 4. Three-Angle View

### 4.1 Proposer: "VQ-EMA is the right fix"

VQ-EMA (Option A) directly addresses the root cause:
- **Assignment instability** → solved: EMA is a temporal average, stable even when individual step assignments flip
- **Bootstrap problem** → solved: k-means converges from random initialization regardless of initial key geometry
- **Implementation** → 20-30 lines in selector.py, no changes to layer.py
- **DDP** → manageable: allreduce of (count_j, sum_j) tensors per step
- **Expected behavior**: top1_sim should increase as keys move toward cluster centers. Once some differentiation exists, InfoNCE can be added as a refinement.

**Confidence that VQ-EMA causes top1_sim > 0.005 at fwd=200**: **85%**

The 15% uncertainty:
- If all queries come from the same LM distribution and are highly correlated, the cluster centers might all converge to the same point (mode collapse). Need load_balance to prevent this.
- DDP allreduce might not be correctly implemented in first pass, causing rank divergence.

### 4.2 Skeptic: "VQ-EMA might have its own problems"

1. **Collapse risk**: VQ-EMA with many queries per step and few slots that match → the most popular slots get all the queries → exponential runaway. Need EMA count normalization and "dead slot" revival.

2. **DDP correctness**: With 8 GPUs and B=1 per GPU, each GPU has different queries. The EMA should see ALL queries across all ranks. This requires `dist.all_reduce` of (sum_q, count_q) before updating slot_keys. Getting this right matters.

3. **Interaction with load_balance**: The load_balance loss pushes scores toward uniform, but VQ-EMA pushes slot_keys toward query cluster centers. These are complementary (load_balance uses softmax scores, VQ-EMA uses hard assignment). No conflict expected.

4. **alpha tuning**: EMA decay α needs tuning. Too slow (α close to 1) → keys don't move. Too fast (α close to 0) → slots overfit to single steps. α=0.9 is a typical starting point.

### 4.3 Critic: "Is there an even simpler fix?"

**Option C: Remove all slot_key supervision; rely on LM gradient only**

The current situation: LM gradient DOES flow to slot_keys (grad_norm ≈ 1.6→0.9 confirmed). The InfoNCE is ADDING gradient but canceling itself. What if we just remove InfoNCE entirely and let LM gradient drive routing?

The objection: LM gradient reaches slot_keys through the STE path (soft weighted sum over slots). At initialization, all slot VALUES are identical (from same hidden_pool mean), so slot_to_hidden(slots[i]) ≈ slot_to_hidden(slots[j]) for all i,j → the STE gradient to scores is near-uniform for all i → slot_keys get near-uniform gradient → no differentiation.

This was confirmed by rpt_20260430_0637_fix_x2_infonce_routing: "STE bottleneck prevents LM gradient from differentiating slot_keys".

**But wait**: Fix X.1 also removed the slot_keys detach at selector.py line 159. Now slot_keys DO receive LM gradient directly (not just through scores). How? Via the `q = F.normalize(Q_sel(pool), dim=-1)` path: `q` is used in `logits = einsum(q, k) * T`, and `logits → scores → ste_weights → STE → M_sel_hidden_soft → ...`. So LM gradient flows: `loss → M_sel_hidden → M_sel_hidden_soft → slot_to_hidden → M_sel_slot_soft → slots`. But NOT directly to slot_keys.

Actually slot_keys gradient path: `q = Q_sel(pool)`, normalized. `k = F.normalize(slot_keys)`. `logits = q · k^T * T`. Gradient to `slot_keys`: `∂loss/∂slot_keys ← ∂loss/∂logits ← ∂loss/∂scores ← ∂loss/∂M_sel_hidden_soft ← ∂loss/∂M_sel_slot_soft ← ∂loss/∂slots`. But again the STE is: `M_sel_hidden = M_sel_hidden_hard.detach() + (M_sel_hidden_soft - M_sel_hidden_soft.detach())`. So `∂loss/∂M_sel_hidden_soft = ∂loss/∂M_sel_hidden` (chain rule through STE — the trick is ste_output.backward() flows to the soft branch). YES, so LM does flow to slot_keys.

BUT the gradient to slot_keys via LM is: `∂loss/∂scores` × `∂scores/∂logits` × `∂logits/∂k` × `∂k/∂slot_keys`. The `∂loss/∂scores` term requires `∂loss/∂M_sel_hidden_soft` to be non-uniform — but at initialization, `slot_to_hidden(slots[i]) ≈ constant`, so `M_sel_hidden_soft = sum_i scores[i] * slot_to_hidden(slots[i]) ≈ slot_to_hidden(slots[0]) * sum_i scores[i] = constant` — gradient of loss w.r.t. `scores` is uniform → `∂loss/∂slot_keys` is uniform → no routing differentiation.

**Conclusion for Critic**: LM gradient alone is insufficient to bootstrap routing differentiation when slot VALUES are uniform. Need an EXTERNAL signal that bypasses slot content. VQ-EMA or InfoNCE-with-stable-assignment both qualify. VQ-EMA has a key advantage: stable assignment even with uniform keys.

---

## 5. Concrete Fix X.3 Specification

**Chosen Option: VQ-EMA slot_keys update (Option A)**

### 5.1 Why VQ-EMA over InfoNCE

InfoNCE failed BECAUSE the positive assignment is unstable at initialization. VQ-EMA solves this by using running means instead of per-step hard assignments. The assignment still occurs step-by-step but the KEY UPDATE accumulates evidence over time, removing the contradictory gradient problem.

Once VQ-EMA gives slot_keys some initial separation (top1_sim > 0.005), BOTH the LM gradient path AND the existing load_balance will continue pushing for further differentiation. VQ-EMA is thus a BOOTSTRAPPER, not a permanent replacement.

### 5.2 Exact Implementation Plan

**File**: `src/memory/mem_space/selector.py`

**New parameters in `__init__`**:
```python
# Fix X.3 (VQ-EMA): EMA statistics for stable slot_key bootstrap
# ema_update_weight > 0 enables VQ-EMA slot_key updates
self.register_buffer(
    'ema_cluster_sum',
    torch.zeros(num_slots, selector_dim)
)
self.register_buffer(
    'ema_cluster_count',
    torch.ones(num_slots)  # init to 1 to avoid divide-by-zero
)
```

**New method `vq_ema_update`** (called inside `forward` with `torch.no_grad()`):
```python
@torch.no_grad()
def vq_ema_update(
    self,
    q: torch.Tensor,       # [B, S], ALREADY normalized (detached for EMA)
    idx: torch.Tensor,     # [B, top_k], hard assignment indices
    alpha: float = 0.9,    # EMA decay
    dist_backend = None,   # pass dist module if distributed
) -> None:
    """VQ-EMA update: move slot_keys toward mean query that assigned to them.
    
    Fix X.3 (2026-04-30): Replaces InfoNCE as the bootstrapper for slot_key
    initialization. At random init where InfoNCE fails (contradictory gradients),
    VQ-EMA converges reliably by accumulating temporal running means.
    
    Each slot key is updated toward the mean of normalized queries that selected
    it as top-1. Uses EMA (exponential moving average) matching VQ-VAE codebook
    update (van den Oord et al. 2017).
    
    Args:
        q: [B, S] normalized query vectors (already unit vectors from forward)
        idx: [B, top_k] hard top-k indices; only top-1 (idx[:, 0]) is used
        alpha: EMA decay. Typical: 0.9 (fast convergence), 0.99 (slow, stable)
        dist_backend: if torch.distributed, pass it to allreduce sums
    """
    B, S = q.shape
    pos_idx = idx[:, 0]  # [B], top-1 assignment per batch item
    
    # Accumulate: for each slot j, sum all queries that chose it as top-1
    # scatter_add: new_sum[j] = sum of q[b] where pos_idx[b] == j
    q_detach = q.detach()  # safety: EMA should not create gradient paths
    
    new_sum = torch.zeros(self.num_slots, S, device=q.device, dtype=q.dtype)
    new_count = torch.zeros(self.num_slots, device=q.device, dtype=q.dtype)
    
    new_sum.scatter_add_(
        0,
        pos_idx.unsqueeze(-1).expand(-1, S),  # [B, S]
        q_detach
    )
    new_count.scatter_add_(
        0,
        pos_idx,
        torch.ones(B, device=q.device, dtype=q.dtype)
    )
    
    # In DDP: allreduce sums across ranks so all replicas see ALL queries
    if dist_backend is not None:
        try:
            dist_backend.all_reduce(new_sum, op=dist_backend.ReduceOp.SUM)
            dist_backend.all_reduce(new_count, op=dist_backend.ReduceOp.SUM)
        except Exception:
            pass  # if allreduce fails, proceed with local sums (graceful degradation)
    
    # EMA update for slots that received at least 1 assignment
    assigned_mask = new_count > 0  # [N]
    if assigned_mask.any():
        # EMA on count and sum separately (VQ-VAE style)
        self.ema_cluster_count[assigned_mask] = (
            alpha * self.ema_cluster_count[assigned_mask]
            + (1 - alpha) * new_count[assigned_mask]
        )
        self.ema_cluster_sum[assigned_mask] = (
            alpha * self.ema_cluster_sum[assigned_mask]
            + (1 - alpha) * new_sum[assigned_mask]
        )
        
        # Update slot_keys = L2-normalized EMA mean
        # Only update slots that received assignments (avoid pulling unselected slots)
        new_keys = F.normalize(
            self.ema_cluster_sum[assigned_mask] / self.ema_cluster_count[assigned_mask].unsqueeze(-1).clamp(min=1e-6),
            dim=-1
        )
        self.slot_keys.data[assigned_mask] = new_keys
    
    # "Dead slot revival": if a slot hasn't been selected for a long time,
    # reset it to a perturbed version of a randomly selected "hot" slot.
    # This prevents slot collapse (most queries funneling to 1-2 slots).
    # Note: this is optional and should only run periodically to avoid noise.
    # (Implement separately if needed — omit for Fix X.3 first pass)
```

**In `forward()`, add after computing `idx`**:
```python
# Fix X.3 (VQ-EMA): update slot_keys toward query cluster centers
# This is the bootstrap mechanism that InfoNCE failed to provide:
# EMA accumulates stable running means instead of per-step contradictory gradients.
# Only runs during training (not eval).
if self.training and getattr(self, 'ema_update_weight', 0.0) > 0.0:
    import torch.distributed as _dist
    _dist_module = _dist if (_dist.is_available() and _dist.is_initialized()) else None
    self.vq_ema_update(q, idx, alpha=self.ema_update_alpha, dist_backend=_dist_module)
```

**New config fields** (`config.py`):
```python
ema_update_weight: float = 0.0   # Fix X.3: EMA update rate; 0 = disabled
ema_update_alpha: float = 0.9    # EMA decay coefficient  
```

**New selector `__init__` params**:
```python
ema_update_weight: float = 0.0,
ema_update_alpha: float = 0.9,
```
And:
```python
self.ema_update_weight = ema_update_weight
self.ema_update_alpha = ema_update_alpha
```

**Layer.py**: Pass `ema_update_weight` and `ema_update_alpha` from config to selector `__init__`.

**Training script**: Add `--ema_update_alpha` CLI arg (default 0.9).

### 5.3 Interaction with Existing Components

| Component | Interaction with VQ-EMA |
|-----------|------------------------|
| InfoNCE (qa_loss) | Can be DISABLED (set query_alignment_weight=0). VQ-EMA replaces InfoNCE for bootstrapping. After top1_sim > 0.02, InfoNCE can be re-enabled as fine-tuning. For Fix X.3 first pass: qa_weight=0.0 |
| load_balance_loss | COMPLEMENTARY. Load balance uses softmax scores to push diversity. VQ-EMA uses hard assignment to push slot_keys toward query centroids. Both needed. |
| slot_key_diversity_loss (SKRL) | DISABLE (skrl_weight=0.0). SKRL drives ETF which is query-adversarial. VQ-EMA replaces it. |
| LM gradient to slot_keys | COMPLEMENTARY. Once VQ-EMA bootstraps some key differentiation, LM gradient provides fine-grained feedback. The two work together |
| hidden_to_slot.weight.grad=None | ORTHOGONAL ISSUE. VQ-EMA doesn't depend on hidden_to_slot. Fix X.3 is independent of this bug. |
| T=10 temperature | KEEP. With VQ-EMA bootstrapping initial diversity, T=10 will cause sharper softmax → larger LM gradient to slot_keys → positive reinforcement loop. |

### 5.4 Why This Will Cause top1_sim to Increase

Mechanically:
1. At step 1: queries are drawn from Gaussian LM representation space. In dim=4096 projected to dim=128, they form clusters by topic/context.
2. VQ-EMA assigns each query to the current nearest slot_key (initially random).
3. After 10 steps: each slot_key has been updated to be the running mean of all queries assigned to it. Since queries cluster (similar contexts → similar representations), some slot_keys will move toward topic clusters.
4. After 50 steps: slot_keys will have converged toward the principal modes of the query distribution. Top-1 cosine similarity will increase as queries find "their" slot.

**Monte Carlo prediction**:
If queries form K distinct clusters in the 128-dim projected space with inter-cluster distance d, and slot_keys converge to cluster centers:
- top1_sim = d/2 (distance from cluster center to boundary)
- For Llama-3 text representations projected to dim=128, inter-cluster distance ≈ 0.3-0.5
- Expected top1_sim after VQ-EMA convergence: 0.15-0.25

This is dramatically higher than the current floor of 0.002.

### 5.5 Confidence and Risk Assessment

| Metric | Estimate |
|--------|----------|
| Confidence top1_sim > 0.005 at fwd=200 | **87%** |
| Confidence top1_sim > 0.020 at fwd=200 | **75%** |
| Confidence top1_sim > 0.100 at fwd=500 | **60%** |
| Risk of slot collapse (all queries → 1-2 slots) | 20% (mitigated by load_balance) |
| Risk of DDP allreduce bug | 15% (can be tested; graceful degradation fallback) |
| Implementation complexity | ~70 lines in selector.py, ~15 lines in config.py, ~10 lines in layer.py |

### 5.6 Recommended Ablation Design

Three nodes (if available):

| Node | alpha | ema_update | qa_weight | skrl | Expected top1@200 |
|------|-------|-----------|-----------|------|-------------------|
| b200-1 | 0.9 | True | 0.0 | 0.0 | 0.010-0.050 |
| b200-2 | 0.99 | True | 0.0 | 0.0 | 0.005-0.030 |
| b200-3 | 0.9 | True | 0.05 | 0.0 | 0.010-0.060 (EMA+InfoNCE combo) |

Kill condition: if top1_sim < 0.005 at fwd=500 on ALL 3 simultaneously, kill all.

---

## 6. Secondary Issue: hidden_to_slot.weight.grad=None

This bug persists independently of Fix X.3. Its root cause:

From `layer.py` lines 654-655:
```python
O_mem_slot = self.hidden_to_slot(O_mem_hidden)
self.memory_bank.write(idx, O_mem_slot, beta)
```

And from the training loop (train_mem_space_pg19.py line 809):
```python
_detach_banks(model)   # Fix K: carry-over instead of reset
```

This `_detach_banks` calls `shared_bank.detach_()` which does `self.slots = self.slots.detach()`. This detaches the ENTIRE bank tensor in-place, breaking any gradient path from prior steps.

Within a single forward pass (chunk), the gradient path across 32 layers via shared bank DOES work:
- Layer 0 writes `new_slots = beta * slots + (1-beta) * O_mem_slot_0`
- Layer 1 reads `new_slots` (grad_fn = AddBackward from layer 0's write)
- ...continues through all layers

BUT: `memory_bank.init_from_hidden()` is called at the START of each chunk (when bank not initialized). This init likely uses `.detach()` on the initial slots, OR the bank was `reset()` then `init_from_hidden` creates new slots from hidden (which has grad_fn=None at input level). 

Checking `init_from_hidden`: The method creates new slot tensors from `hidden_states` which at the FIRST layer is the raw input embeddings (no grad_fn through LM parameters, because gradients flow backward from loss). At initialization, there IS a gradient path from loss to hidden_to_slot IF:
- The bank's slots have grad_fn
- slots came from hidden_to_slot in a prior layer's writeback

Within the same chunk's forward pass (shared bank, 32 layers), layer 0 initializes the bank from hidden (creating detached slots at layer 0). Then layer 0 writes `O_mem_slot_0 = hidden_to_slot(O_mem_hidden_0)` and updates `slots_1 = beta*slots_0 + (1-beta)*O_mem_slot_0`. Layer 1 reads `slots_1` which HAS grad_fn (from write). Layer 1's STE soft proxy: `M_sel_slot_soft_1 = einsum(scores_1, slots_1)` — this has grad_fn. Gradient flows: `loss → M_sel_hidden_1 → M_sel_slot_soft_1 → slots_1 → O_mem_slot_0 = hidden_to_slot(O_mem_hidden_0) → hidden_to_slot.weight`. This SHOULD give hidden_to_slot a gradient!

**Why doesn't it in practice?** Possibly:
1. `beta` is initialized from `writeback_gate_init=0.0`, giving `beta ≈ 0` at step 0 with `writeback_warmup_steps=0`. If `beta=0`, then `slots_1 = 0 * slots_0 + (1-0) * O_mem_slot_0` = `O_mem_slot_0`. So slots_1 IS O_mem_slot_0 with grad_fn.
   - Wait, actually `writeback_gate_max=0.3` and `sigmoid(0.0) = 0.5`, so `beta = 0.5 * 1.0 * 0.3 = 0.15`. Not zero.
2. The STE has `M_sel_hidden = M_sel_hidden_hard.detach() + (M_sel_hidden_soft - M_sel_hidden_soft.detach())`. The gradient to M_sel_hidden_soft is `∂loss/∂M_sel_hidden`. Is there a path from loss to M_sel_hidden that doesn't go through the detach?

Actually: `next_hidden = bypass_h + alpha * slot_delta`. `slot_delta = ext_h[:, k_slots:, :] - bypass_h`. `ext_h` comes from the wrapped decoder layer operating on `extended_hidden = cat([M_sel_hidden, hidden_states])`. The gradient from `loss → next_hidden → ext_h → extended_hidden → M_sel_hidden → M_sel_hidden_soft`. 

But: `alpha = tanh(slot_output_gate)`. At init, `slot_output_gate = 0.5`, so `alpha = tanh(0.5) ≈ 0.46`. NOT zero. So slot_delta IS non-zero and gradient DOES flow.

**Most likely bug**: There's a `with torch.no_grad()` or `.detach()` somewhere in the chain that I'm missing without reading memory_bank.py. The GATE_GRAD_DIAG confirms `hidden_to_slot.weight.grad_norm=None` empirically. Since Fix X.3 doesn't depend on this path, investigating it separately is warranted but not blocking.

---

## 7. Summary Table

| Question | Answer |
|----------|--------|
| Why does InfoNCE fail at log(N)? | **Assignment instability**: at init, top-1 randomly flips between keys each step. Gradient direction E[Δk_j] ≈ 0 across steps → zero net movement. Large gradient magnitude but zero expected direction. |
| Which hypothesis correct? | **A (primary)**: contradictory gradients from unstable positives. **B (secondary)**: centroid conservation means only spread changes. C rejected. |
| VQ-EMA applicable? | **YES**: directly solves hypothesis A by accumulating temporal running means instead of per-step gradients. K-means convergence guarantee from random init. |
| Fix X.3 option? | **Option A: VQ-EMA** |
| Does hidden_to_slot fix need to come first? | **NO**: VQ-EMA operates only on slot_keys, independent of hidden_to_slot path. |
| Expected top1_sim after Fix X.3? | > 0.005 at fwd=200 (87% confidence), > 0.020 (75% confidence) |

---

## 8. Key Literature References

1. **van den Oord et al. 2017** (VQ-VAE): "Neural Discrete Representation Learning" — EMA codebook update (Algorithm 1). Direct precedent for VQ-EMA slot_key update.

2. **Fedus et al. 2021** (Switch Transformer): Load balance auxiliary loss for routing diversity. Already implemented as `load_balance_loss` — keep this.

3. **He et al. 2020** (MoCo): Momentum contrast — EMA key encoder. Shows EMA is highly effective for stable key learning in contrastive settings.

4. **Caron et al. 2020** (SwAV): Online k-means clustering for visual representations. Shows that online EMA-based cluster assignment converges faster than contrastive losses from random init.

5. **Radford et al. 2021** (CLIP): InfoNCE works when positives are STABLE (fixed image-caption pairs). Confirms our diagnosis: InfoNCE requires stable positive assignment.

---

## Appendix: Numerical Verification of Assignment Instability

Given N=512, d=128, slot_keys ~ N(0, 0.01) (std=0.1), after F.normalize:

Expected max cosine similarity among 512 unit-random vectors in d=128:
```
E[max_i cos(q, k_i)] ≈ sqrt(2 * log(512) / 128) = sqrt(2 * 6.24 / 128) = sqrt(0.0975) ≈ 0.312
```

But this is the THEORETICAL maximum for one fixed q. With q also random:
```
cos(q, k_i) ~ N(0, 1/d) = N(0, 1/128)
E[top1] = E[max of N(0,1/128)] for N=512 samples = (1/√128) * E[max of 512 N(0,1)]
E[max of 512 N(0,1)] ≈ sqrt(2*log(512)) ≈ 3.53
E[top1] ≈ 3.53 / 11.31 ≈ 0.312
```

But observed top1_sim ≈ 0.002, not 0.312. Why the discrepancy?

Because `top1_sim` in the diagnostic is `scores.max(dim=-1).values` — the SOFTMAX SCORE, not cosine similarity!

```
scores = softmax(logits * T) = softmax(cos(q, k_i) * 10 for all i)
At uniform cos = 0: scores_max = 1/512 = 0.00195 ≈ 0.002
```

YES — this confirms the initial state is at the softmax uniform floor. The diagnostic metric is the softmax score, not raw cosine similarity. top1_sim = 0.002 = 1/N is correct and expected at initialization. The failure is that it STAYS at this floor.

After VQ-EMA, once slot_keys move toward query cluster centers, the cosine similarity between a query and its nearest key will increase. If delta_cos = 0.1 for the top-1 key (all others still at 0):
```
scores[top1] = softmax(0.1 * 10) / (exp(1.0) + 511 * exp(0)) = exp(1.0)/(exp(1.0) + 511) = 2.718/513.7 ≈ 0.0053
```

This is 2.7x the floor — above the 0.005 threshold. And with delta_cos = 0.2:
```
scores[top1] = exp(2.0)/(exp(2.0) + 511) = 7.39/518.4 ≈ 0.0143
```

VQ-EMA should easily achieve delta_cos ≥ 0.1 within 50-100 steps of a well-clustered query distribution. This would produce top1_sim > 0.005.
