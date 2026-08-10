# Research Brief: Fix J — Why `hidden_to_slot.weight.grad = None` even after Fix I

**Date**: 2026-04-29 14:35 GMT+8
**Triggered by**: /trainer killed fix_i_ablation after GATE_GRAD_DIAG confirmed `hidden_to_slot.weight.grad_norm=None` at all steps 0–20 despite `trainable_with_grad=128/224` (param IS in optimizer, but has zero gradient).
**Question**: What is the exact gradient stopper on the `hidden_to_slot` parameter? What is the minimal code change (Fix J) to restore gradient flow?

---

## 1. Background

Recap of the 9-fix saga (A–I):

| Fix | Change | Result | GATE_GRAD_DIAG reading |
|-----|--------|--------|------------------------|
| A   | slot_init_noise ↑ | FAIL | — |
| B   | learnable `slot_keys` Parameter | FAIL | — |
| C   | cosine normalization in selector | FAIL | — |
| D.1 | `slot_output_gate` init = 0.5 | FAIL | — |
| D.2 | entropy aux weight | FAIL | — |
| E   | full-scale `M_sel_hidden` projection | FAIL | — |
| F   | centered STE gradient multiplier | FAIL | — |
| G   | SKRL (slot-key repulsion) | FAIL | — |
| H   | differentiable soft routing proxy + slot norm clip | FAIL | `trainable_with_grad=128/192`; `hidden_to_slot.weight.grad_norm=None` |
| I   | include hidden_to_slot in `_mem_space_params()` | **FAIL** | `trainable_with_grad=128/224`; `hidden_to_slot.weight.grad_norm=None` |

Fix I's denominator change 192→224 (+32 params) confirms `hidden_to_slot.weight` IS now in the optimizer. But its `.grad` is literally `None`, not small — meaning the backward pass never assigns it a gradient tensor. This is a **computation-graph bug**, not a parameter-registration bug.

---

## 2. Findings

### 2.1 Every call site of `hidden_to_slot`

Grep shows exactly TWO call sites in `src/memory/mem_space/layer.py`:

**Call site 1** — lazy init (line 417–420):
```python
if not self.memory_bank.is_initialized(B):
    H_for_init = hidden_states
    if self.slot_dim != self.d_model:
        H_for_init = self.hidden_to_slot(hidden_states)   # ← dead for Llama-3 config
    self.memory_bank.init_from_hidden(H_for_init, batch_size=B)
```

**Config check**: `scripts/train_mem_space_pg19.py:485–500` builds `MemorySpaceConfig` WITHOUT setting `slot_dim`, so it defaults to `None` → `patch.py:104` sets `slot_dim = d_model = 4096`. Therefore `self.slot_dim == self.d_model` is **True**, and the `if` branch at line 418 is **never taken**. **Call site 1 is dead code for the current Llama-3 setup.**

Even if it were taken, `init_from_hidden` immediately calls `slots.detach()` at `memory_bank.py:154` — so no gradient would flow back through `hidden_to_slot` anyway.

**Call site 2** — writeback (line 614–617):
```python
beta_t = self._current_beta()
if cfg.enable_writeback:
    O_mem_slot = self.hidden_to_slot(O_mem_hidden)      # [B, k, slot_dim]
    self.memory_bank.write(idx, O_mem_slot, beta_t)
```

`memory_bank.write()` performs a differentiable EMA (tensor-gate path, `memory_bank.py:227–233`):
```python
updated = (1.0 - gate_t) * current + gate_t * new_contrib
self.slots = self.slots.scatter(1, idx_exp, updated)
```
This IS connected to the autograd graph (uses non-in-place `scatter`, documented at line 247–252). So `hidden_to_slot → O_mem_slot → updated → self.slots` is gradient-bearing in principle.

### 2.2 The smoking gun: `slots.detach()` at layer.py:499

The **next** layer reads `self.memory_bank.slots` at line 422:
```python
slots = self.memory_bank.get()   # [B, N, slot_dim]
```
and uses it in TWO places:

1. **Hard path** (line 468, 490):
   ```python
   M_sel_slot = slots.gather(1, idx_exp)                  # [B, k, slot_dim]
   ...
   M_sel_hidden_hard = self.slot_to_hidden(M_sel_slot)    # [B, k, d]
   ```
2. **Soft proxy** (line 496–500):
   ```python
   M_sel_slot_soft = torch.einsum("bn,bnd->bd", scores, slots.detach())  # ← 🔴 DETACHED
   ```

Then at line 506:
```python
M_sel_hidden = M_sel_hidden_hard.detach() + (M_sel_hidden_soft - M_sel_hidden_soft.detach())
```

**Both paths block gradient back into `slots`:**
- `M_sel_hidden_hard.detach()` — the hard path's gradient into `slots` is killed.
- `slots.detach()` in the soft proxy — the soft path's gradient into `slots` is also killed.

The combined `M_sel_hidden` has **zero gradient** flowing into the `slots` tensor (it has gradient only into `scores` via the `M_sel_hidden_soft` term, which uses `slots.detach()`).

### 2.3 The dev comment that admits the bug (layer.py:299–306)

```python
# Freeze hidden_to_slot — it participates in NO operation whose output
# influences the loss in Tier-3:
#   1. O_mem_slot = hidden_to_slot(O_mem_hidden)
#   2. memory_bank.write(idx, O_mem_slot.detach(), beta) — DETACHED
#   3. _reset_banks discards the bank every chunk
# So its gradient is identically zero. Freezing reclaims ~540M of the
# 1107M trainable params reported in the fix3 smoke and saves ~2.1 GB
# of bf16 Adam optimizer moments.
```

Note: the comment's claim #2 ("memory_bank.write uses `O_mem_slot.detach()`") is **outdated** — current `layer.py:617` passes `O_mem_slot` WITHOUT detach (confirmed by reading the line), and `memory_bank.write()` builds the differentiable EMA. So the write IS gradient-bearing into `self.slots`.

The true remaining blocker is **not** `memory_bank.write` — it is the **subsequent layer's read-side detach** at `layer.py:499` (`slots.detach()` in the soft proxy) and at `layer.py:506` (`M_sel_hidden_hard.detach()`). These two detaches sever the gradient chain `loss → M_sel_hidden → slots → (prev layer's O_mem_slot) → hidden_to_slot.weight`.

### 2.4 Why the current setup makes `hidden_to_slot.grad` identically None

End-to-end backprop trace for `hidden_to_slot.weight.grad`:

```
loss ← next_hidden ← bypass_h + alpha * slot_delta
                      ↑
                     slot_delta uses ext_h (decoder forward over extended_hidden)
                      ↑
                     extended_hidden = cat([M_sel_hidden, hidden_states])
                      ↑
                     M_sel_hidden = M_sel_hidden_hard.detach() + (M_sel_hidden_soft - M_sel_hidden_soft.detach())
                      ↑↑
              [hard part DETACHED]   [soft part has grad only in `scores`, not in `slots` (slots.detach())]

=> gradient into `slots` tensor at the READ side: 0

But `slots` is also WRITTEN to by the previous layer via:
  slots ← scatter(updated, ...) where updated = (1-β) current + β · O_mem_slot
                                                                     ↑
                                                          O_mem_slot = hidden_to_slot(O_mem_hidden)

If the READ side never pulls gradient from `slots`, then the scatter-update's
autograd path does not carry any upstream gradient, so hidden_to_slot.weight.grad = None.
```

This is consistent with the log evidence:
- `trainable_with_grad=128/224`: hidden_to_slot (32 params) has `grad=None`; everything else has grad.
- `top1_sim_mean` stuck at 1/512 — because slot bank content never updates through learning, all slots remain near-identical and the selector falls back to the uniform prior.

### 2.5 Additional complication — `M_sel_hidden_hard.detach()` on the HARD path

Even ignoring the soft-proxy detach, the hard-path `M_sel_hidden_hard = self.slot_to_hidden(M_sel_slot)` where `M_sel_slot = slots.gather(...)` — this IS differentiable in `slots` **until** line 506 applies `.detach()` to it:
```python
M_sel_hidden = M_sel_hidden_hard.detach() + ...
```
So the hard path's gradient into `slots` is also killed.

The `.detach()` at line 506 is **intentional** — it's the STE forward (hard = correct slot content) that is detached so the backward-only uses the soft term. But this detaches both `slot_to_hidden` and `slots` from the gradient graph on the hard path.

---

## 3. Triangulation

### 🟢 Proposer (best case for Fix J — remove the dead path)

Two alternative Fix J implementations, both straightforward:

**Fix J-A (minimal)**: Remove `slots.detach()` at `layer.py:499` (soft proxy uses attached `slots`).

```python
M_sel_slot_soft = torch.einsum("bn,bnd->bd", scores, slots)  # remove .detach()
```

This single-line change opens a gradient path: `loss → M_sel_hidden_soft (non-detached) → slots (via einsum) → previous layer's scatter → O_mem_slot → hidden_to_slot`.

**Fix J-B (explicit)**: Re-architect the read to make the hard path gradient-bearing in `slots` (drop the STE detach, use the soft proxy value AS the forward value, not just for backward).

```python
M_sel_hidden = M_sel_hidden_soft  # forward=backward=soft, drop hard/STE
```
This loses the "exact slot content" property at forward time, reverting to Fix H's soft routing. NOT recommended — it's a step back.

**Recommendation**: Fix J-A (1-line fix). This makes `slots` attached to the graph, which means `hidden_to_slot` now sees gradient from the next layer's read of what the current layer wrote.

### 🔴 Skeptic (risks & counter-evidence)

1. **Fix J-A might cause graph explosion** — the soft proxy with attached `slots` means gradient now flows through the FULL bank `[B, N=512, slot_dim=4096]` instead of just the gathered k=64. For 32 shared_memory_bank layers, this means the backward pass through the EMA chain could be expensive.
   **Mitigation**: bank is detached at chunk boundary (`memory_bank.detach_()` / `reset()`), so the chain is bounded to within-chunk depth = 32 layers. Still O(32× N × slot_dim) extra backward work, but tractable.

2. **Fix J-A doesn't unblock the HARD path** — `M_sel_hidden_hard.detach()` at line 506 remains. The gradient-bearing path is only the soft proxy. But the soft proxy IS differentiable in `slots` and that's enough to give `hidden_to_slot` a non-zero gradient.

3. **SKRL weight=0 means no direct gradient on slot_keys** — not relevant here; slot_keys ARE in optimizer and already get gradient from the main LM loss via Q_sel/scores.

4. **The whole premise might be wrong — what if `scores` drives all routing, and `slots` content is cosmetic?** — This is actually what the architecture became after Fix B (learnable `slot_keys` independent of slot content). Under this view, `hidden_to_slot` is architecturally useless because routing uses `slot_keys`, not `slots`. Fix J-A would still put `hidden_to_slot` in the graph, but it would only learn to make the SLOT CONTENTS match what the decoder wants to read — a reasonable goal.

### 🔵 Critic (blind spots)

1. **Is the failure actually hidden_to_slot being frozen, or is it something else entirely?** Every prior fix A–I targeted a symptom (init noise, key diversity, gate init, STE …). It is possible that the TRUE root cause is that with `slot_init="hidden_pool"` or `"random"` + `slot_init_noise=0.02`, all slots at init are near-identical, and the Q_sel/slot_keys routing has a unique maximum that never moves because the reward signal is flat. In that case, making `hidden_to_slot` trainable changes nothing about the routing, and `top1_sim_mean` stays at 0.00195.
   **Counter-evidence**: Fix H's `trainable_with_grad=128/192` (hidden_to_slot already frozen in Fix H via `hidden_to_slot_frozen=True`) vs. pre-Fix-H runs with different config — symptom identical. The diagnostic shows the issue is structural, not just "slot content is useless." Without `hidden_to_slot` trainable, the bank has no parameterised capacity to shape its contents, so any slot diversity comes only from `init_noise` — severely limited.

2. **Config complexity**: `shared_memory_bank=True` means 32 layers share the bank. Layer 0 writes, layers 1–31 read (and 31 also write). Gradient from layer 31's loss through the shared bank goes through 31 layers of scatter-EMA, which might have vanishing/exploding issues.
   **Mitigation**: the slot-norm clip at `memory_bank.py:240–244` (added in Fix H) bounds slot magnitude, so BPTT through depth should not explode. Vanishing is still possible but not a deal-breaker for Fix I→J (signal just needs to flow, magnitude is secondary).

3. **The dev comment at layer.py:299–306 says "memory_bank.write uses .detach()"** — outdated by Branch-3 (2026-04-26) change, confirmed by reading `layer.py:617` and `memory_bank.py:217–234`. The dev comment is stale documentation. Anyone reading the comment without verifying would miss that the write IS gradient-bearing now — the remaining blocker is on the READ side.

4. **Interaction with Fix H's slot-norm clipping**: Fix J-A will cause gradient to flow through the `slot_norms.clamp(max=max_norm)` op in the write path. `clamp` has zero gradient where saturated. If slot norms are at the cap, gradient to upstream (O_mem_slot → hidden_to_slot) will be zero for those slots. Early steps probably aren't saturated (slots are small at init), so this should not block the initial gradient.

---

## 4. Fix J Proposal

### Minimal change (recommended)

**File**: `src/memory/mem_space/layer.py`
**Location**: Line 499 (inside `forward()`, in the soft-proxy construction)
**Change**: Remove `.detach()` on `slots` in the soft proxy.

**Before (line 496–500)**:
```python
M_sel_slot_soft = torch.einsum(
    "bn,bnd->bd",
    scores,
    slots.detach()
)                                                               # [B, slot_dim]
```

**After**:
```python
M_sel_slot_soft = torch.einsum(
    "bn,bnd->bd",
    scores,
    slots,                      # Fix J (2026-04-29): attach slots so hidden_to_slot gets gradient from next-layer reads
)                                                               # [B, slot_dim]
```

Also update the comment at `layer.py:494–495`:
```python
# Soft proxy: differentiable weighted sum over ALL slots using softmax scores
# scores: [B, N]  (softmax probabilities from selector)
# Fix J (2026-04-29): use attached `slots` so gradient flows back into the EMA
# write chain → hidden_to_slot in previous layers. Prior `.detach()` killed
# the end-to-end signal that hidden_to_slot needs to learn.
```

### Optional: stale comment cleanup

Update the docstring at `layer.py:299–306` to reflect the current Branch-3 state (the "memory_bank.write uses .detach()" bullet is wrong). Non-functional, but prevents future devs from being misled.

### Success criteria for Fix J

Launch `fix_j_ablation` on b200-2/3/4 with:
```
--unfreeze_hidden_to_slot --num_slots 512 --top_k 64 --slot_init_noise 0.02 \
--max_steps 10000 --seq_len 4096 --batch_size 1 --shared_memory_bank \
--skrl_weight 0.0 --slot_init random
```

**Go/No-go at n_done ≤ 20** (GATE_GRAD_DIAG):
- `hidden_to_slot.weight.grad_norm ≠ None` → Fix J works at the autograd level. Continue.
- `hidden_to_slot.weight.grad_norm == None` → Fix J did not attach the graph. Kill and re-audit (possibly the hard-path detach at line 506 is the dominant detach and the soft path's tiny scores×slots inner product doesn't reach hidden_to_slot with measurable norm).

**Intermediate milestones**:
- step 200: `hidden_to_slot.weight.grad_norm > 0` continues
- step 500: `top1_sim_mean > 0.005` (escape from 1/512 floor)
- step 1000: `top1_sim_mean > 0.05` (unblock req_20260427_102400_scale_up_N1024)

### Confidence

**High** that the diagnosed gradient-stopper (`slots.detach()` at line 499 + `M_sel_hidden_hard.detach()` at line 506) is the cause of `hidden_to_slot.weight.grad=None`.

**Medium** that Fix J-A (removing single `.detach()`) will immediately show `grad_norm > 0` at step 0–20 — the gradient path exists but flows only through the soft proxy's `scores → M_sel_hidden_soft → ... → scatter → O_mem_slot`. Magnitude may be small early because slots are tiny (σ=0.02 init) and scores are near-uniform (1/N = 0.00195 routed to each slot).

**Medium** that Fix J-A alone will resolve the routing degeneracy. The autograd fix is necessary but may not be sufficient — if the SIGNAL-TO-NOISE of `hidden_to_slot`'s gradient is too low (see Skeptic #3 on clamp saturation, Critic #2 on BPTT-through-depth vanishing), we may need a secondary fix (e.g. larger `slot_init_noise` to make early slot norms higher, or removing the hard-path detach too).

**Low** that this is the LAST fix needed before seeing `top1_sim_mean > 0.05`. History shows each of A–I exposed a new blocker. Fix J is necessary-but-possibly-insufficient.

---

## 5. Recommended Next Actions

1. **`/coder`**: implement Fix J-A (single-line change at `layer.py:499`, plus comment update). Also update the stale docstring at lines 299–306 to reflect the Branch-3 state.
2. **`/trainer`**: launch `fix_j_ablation` on b200-2/3/4 with the same config as `fix_i_ablation`. Kill criterion: `hidden_to_slot.weight.grad_norm == None` at step 20.
3. If Fix J-A passes the autograd check but `top1_sim_mean` remains stuck: dispatch `/researcher` again to investigate SIGNAL magnitude (gradient norm vs. expected, BPTT-depth vanishing, clamp saturation).
