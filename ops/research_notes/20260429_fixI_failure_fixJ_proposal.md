# Fix I Failure Root Cause + Fix J Proposal

**Date**: 2026-04-29  
**Author**: researcher subagent  
**Triggered by**: GATE_GRAD_DIAG confirmed `hidden_to_slot.weight.grad_norm=None` at ALL steps 0–20 after Fix I (`trainable_with_grad=128/224` — the +32 proves `hidden_to_slot.weight` IS in the optimizer, but grad is literally `None`, not small)

---

## Executive Summary

Fix I placed `hidden_to_slot.weight` in the optimizer but did **not** restore gradient flow. Being in the optimizer is necessary but not sufficient — the parameter also has to participate in the computation graph that produces the loss. A **grad=None** (not grad≈0) means backward never assigned a gradient tensor, i.e., the parameter is not on any path from inputs to loss.

The exact gradient-severing line is:

```python
# src/memory/mem_space/layer.py, line ~499 (pre-Fix-J-A)
M_sel_slot_soft = torch.einsum("bn,bnd->bd", scores, slots.detach())  # 🔴
```

The `.detach()` on `slots` cuts the only viable backward path:

```
loss → M_sel_hidden_soft → slots (einsum) → scatter EMA → O_mem_slot → hidden_to_slot.weight
```

Fix J-A (remove that single `.detach()`) is the minimal correct fix. It is **already implemented** in the current `layer.py` (the `.detach()` was removed and a Fix J-A comment was added at lines 500–514).

---

## 1. The Gradient Chain (end-to-end trace)

### 1.1 Write path (gradient-bearing AFTER Fix I)

```
O_mem_hidden = extended_output[:, :k, :]          # [B, k, d_model]
O_mem_slot   = hidden_to_slot(O_mem_hidden)        # [B, k, slot_dim] ← hidden_to_slot used HERE
self.memory_bank.write(idx, O_mem_slot, beta_t)
  └─ updated = (1-β)*current + β*new_contrib       # EMA, gradient-bearing when β is tensor
  └─ self.slots = self.slots.scatter(1, idx_exp, updated)  # non-in-place scatter preserves graph
```

After Fix I (`hidden_to_slot` added to optimizer + `requires_grad=True` preserved through `_freeze_backbone`), the write path IS connected to autograd. `memory_bank.py:252` uses a **non-in-place** `scatter` which preserves the autograd graph.

### 1.2 Read path (gradient-blocking — the actual bug)

The next layer (or same layer at next chunk) reads `self.memory_bank.get()` → `slots`. Two things happen:

**Hard path (STE forward)**:
```python
M_sel_slot      = slots.gather(1, idx_exp)           # [B, k, slot_dim]
M_sel_hidden_hard = self.slot_to_hidden(M_sel_slot)  # [B, k, d_model]
```

**Soft proxy (STE backward)**:
```python
# PRE-FIX-J-A (the bug):
M_sel_slot_soft = torch.einsum("bn,bnd->bd", scores, slots.detach())  # 🔴 DETACHED
```

**STE recombination**:
```python
M_sel_hidden = M_sel_hidden_hard.detach() + (M_sel_hidden_soft - M_sel_hidden_soft.detach())
```

Both paths sever gradient into `slots`:
1. **Soft proxy**: `slots.detach()` — the gradient from `M_sel_hidden_soft` can never flow into `slots`.
2. **Hard path**: `M_sel_hidden_hard.detach()` — intentional STE design; the hard path gradient into `slots` is also killed.

Because no gradient flows into `slots` on the read side, the EMA write chain (`slots ← scatter ← updated ← O_mem_slot ← hidden_to_slot`) receives zero upstream gradient from the loss. PyTorch never allocates a `.grad` tensor for `hidden_to_slot.weight` → `grad=None`.

### 1.3 Why slot_to_hidden has healthy grad but hidden_to_slot has None

`slot_to_hidden` operates on `M_sel_slot` (gathered slot content) on the **hard path**:
```python
M_sel_hidden_hard = self.slot_to_hidden(M_sel_slot)
```
Even though `M_sel_hidden_hard` is detached in the STE recombination, `slot_to_hidden.weight` **still gets gradient** because the soft proxy also runs through `slot_to_hidden`:
```python
M_sel_hidden_soft = self.slot_to_hidden(M_sel_slot_soft)  # ← NOT detached (pre-detach)
```
The gradient flows: `loss → M_sel_hidden_soft → M_sel_slot_soft → slot_to_hidden.weight`. This path works because `M_sel_slot_soft` is computed from `scores` (which routes through the selector), not from `slots` (which is the memory bank content written by `hidden_to_slot`).

So `slot_to_hidden` is on the `scores → M_sel_slot_soft → slot_to_hidden.weight` path (live), while `hidden_to_slot` is only on the `slots → O_mem_slot → hidden_to_slot.weight` path (dead due to `slots.detach()`).

---

## 2. Exact Gradient-Severing Lines

| Location | Line (pre-Fix-J-A) | Code | Severity |
|---|---|---|---|
| `src/memory/mem_space/layer.py` | ~499 | `torch.einsum("bn,bnd->bd", scores, slots.detach())` | **PRIMARY — severs only viable path to hidden_to_slot** |
| `src/memory/mem_space/layer.py` | ~506 | `M_sel_hidden_hard.detach()` | Secondary — intentional STE; also kills hard-path gradient into slots |

The stale dev comment at `layer.py:299–306` states:
```
# memory_bank.write(idx, O_mem_slot.detach(), beta) — DETACHED
```
This claim is **wrong** as of Branch-3 (2026-04-26): `layer.py:617` does NOT detach `O_mem_slot` before write. The true blocker is on the READ side, not the write side.

---

## 3. Why Fix I Was Insufficient

Fix I patched `_mem_space_params()` in `scripts/train_mem_space_pg19.py` to include `hidden_to_slot.weight` in the optimizer. This fixed the **parameter-registration** bug. Evidence: `trainable_with_grad` denominator went 192→224 (+32 = exactly `hidden_to_slot.weight.numel()`).

But `grad=None` (not `grad≈0`) means the **computation-graph** bug was separate and independent. Putting a parameter in the optimizer does not cause PyTorch to assign it a gradient — the gradient is assigned by backward(), which only traverses actual edges in the autograd graph. If no loss-to-parameter path exists in the graph, grad stays `None`.

Fix I was necessary but not sufficient. Fix J is the companion graph-connectivity fix.

---

## 4. Fix J Options — Evaluation

### Option A — Remove `slots.detach()` from soft proxy (RECOMMENDED)

**File**: `src/memory/mem_space/layer.py`  
**Location**: soft-proxy einsum construction  
**Change**: Remove `.detach()` from `slots`

```python
# Before (bug):
M_sel_slot_soft = torch.einsum("bn,bnd->bd", scores, slots.detach())

# After (Fix J-A):
M_sel_slot_soft = torch.einsum(
    "bn,bnd->bd",
    scores,
    slots,   # Fix J-A: attached so gradient flows to EMA write chain → hidden_to_slot
)
```

**Gradient path restored**:
```
loss
  → M_sel_hidden (STE recombination)
  → M_sel_hidden_soft (non-detached term)
  → M_sel_slot_soft (via slot_to_hidden backward)
  → slots (via einsum with attached slots)
  → self.slots.scatter(1, ..., updated)  [backward through scatter]
  → updated = (1-β)*current + β*O_mem_slot
  → O_mem_slot = hidden_to_slot(O_mem_hidden)
  → hidden_to_slot.weight.grad ≠ None  ✓
```

**Pros**:
- Single-line change
- Restores end-to-end semantics: hidden_to_slot trains to write content that helps future reads
- No new hyperparameters
- Consistent with the Branch-3 design intent (writeback BPTT through shared bank)
- The hard-path `.detach()` at line 506 is unaffected — STE still works correctly for scores/slot_keys

**Cons / Risks**:
- Gradient now flows through the full slot bank `[B, N=512, slot_dim=4096]` (not just k=64 selected). Memory/time cost of backward through einsum is O(B×N×slot_dim) — larger than before, but bounded within-chunk (reset at chunk boundary).
- If slot norms are at the Fix-H clamp cap, `clamp(max=max_norm)` saturates gradient. Early steps with σ=0.02 init should be fine (slots are small).
- BPTT depth through 32 shared-bank layers could cause vanishing — mitigation: signal only needs to flow, not be large.

### Option B — Auxiliary reconstruction loss on slot content

Add `aux_loss_recon = MSE(hidden_to_slot(O_mem_hidden), slots.detach()[gathered])` as a side-channel gradient to `hidden_to_slot`.

**Verdict: INFERIOR**. This gives `hidden_to_slot` gradient from "match current slot content" rather than "write content that helps future reads." It's a self-referential signal that can't improve retrieval quality. It also adds another hyperparameter (reconstruction weight) and does not fix the architectural dead path. It would technically make `grad≠None`, but the learned projection would optimize the wrong objective.

### Option C — Accept frozen write path, redesign slot init

Acknowledge that `hidden_to_slot` training is architecturally unsound for the current setup and use a fixed init (e.g., input-projection codebook or VQ).

**Verdict: WRONG DIRECTION**. This abandons the Branch-3 writeback BPTT design which is architecturally sound once the read-side detach bug is fixed. The write path IS gradient-bearing (confirmed by `memory_bank.py:252`). Option C trades a one-line bug fix for a major architectural regression.

---

## 5. Current Implementation Status

**Fix J-A is already applied in the current codebase.**

`src/memory/mem_space/layer.py` lines 500–514 now read:
```python
# Fix J-A (2026-04-29): REMOVED slots.detach(). The prior detach was from the
# old design when hidden_to_slot was permanently frozen ...
M_sel_slot_soft = torch.einsum(
    "bn,bnd->bd",
    scores,
    slots    # .detach() removed — gradient now flows back through slots
)
```

The stale comment at lines 299–319 has also been updated to reflect the Branch-3 state (the "memory_bank.write uses .detach()" bullet is gone).

---

## 6. Verification Criteria

Launch `fix_j_ablation` on b200-2/3/4 with:
```
--unfreeze_hidden_to_slot --num_slots 512 --top_k 64 --slot_init_noise 0.02 \
--max_steps 10000 --seq_len 4096 --batch_size 1 --shared_memory_bank \
--skrl_weight 0.0 --slot_init random
```

| Milestone | Criterion | Status if Missed |
|---|---|---|
| Step 0–20 (GATE_GRAD_DIAG) | `hidden_to_slot.weight.grad_norm ≠ None` | Fix J-A did not attach graph; re-audit |
| Step 200 | `hidden_to_slot.weight.grad_norm > 0` continues | Vanishing early in training |
| Step 500 | `top1_sim_mean > 0.005` (escape 1/512 floor) | Gradient insufficient to drive routing diversity |
| Step 1000 | `top1_sim_mean > 0.05` | Unblocks req_20260427_102400_scale_up_N1024 |

If Fix J-A passes the autograd check but `top1_sim_mean` remains stuck at step 500, the gradient signal exists but may be too small (small slot norms at init → tiny einsum outputs → small gradient into `hidden_to_slot`). Secondary mitigations:
- Increase `slot_init_noise` to 0.1–0.5
- Also remove the hard-path `.detach()` at line ~506 (makes both STE paths gradient-bearing into `slots`)

---

## 7. Summary of Fixes A–J

| Fix | Root Cause Addressed | Actual Blocker Missed | Result |
|-----|---------------------|----------------------|--------|
| A | slot_init_noise↑ | write path frozen | FAIL |
| B | learnable slot_keys | write path frozen | FAIL |
| C | cosine norm in selector | write path frozen | FAIL |
| D.1/D.2 | slot_output_gate init, entropy aux | write path frozen | FAIL |
| E | full-scale M_sel_hidden projection | write path frozen | FAIL |
| F | centered STE gradient multiplier | all slots identical (slots.detach not yet identified) | FAIL |
| G | SKRL slot-key repulsion | slot_keys already diverse; write path frozen | FAIL |
| H | differentiable soft proxy + slot norm clip | hidden_to_slot excluded from _mem_space_params | FAIL (grad=None, trainable=128/192) |
| I | include hidden_to_slot in _mem_space_params | slots.detach() at read side severs graph | FAIL (grad=None, trainable=128/224) |
| **J-A** | **remove slots.detach() from soft-proxy einsum** | *(pending ablation)* | **PENDING** |

The critical insight across the whole saga: the problem was never hyperparameters (noise, gate init, key diversity) — it was a **structural dead computation path**. Fix J-A is the first fix that directly addresses the graph-connectivity bug.

---

## 8. Files Audited

- `src/memory/mem_space/layer.py` — full read; found both gradient-severing lines
- `src/memory/mem_space/memory_bank.py` — full read; confirmed write path IS gradient-bearing (non-in-place scatter at line 252)
- `scripts/train_mem_space_pg19.py` — lines 130–239 (param registration, _reset_banks) + 640–760 (training loop, GATE_GRAD_DIAG)
- `src/memory/mem_space/config.py` — full read; confirmed slot_dim=None → d_model → lazy-init call site dead for Llama-3
- `src/memory/mem_space/selector.py` — full read; confirmed slot_keys=nn.Parameter, K_sel frozen
- `ops/research_notes/20260429_fix_j_proposal.md` — prior researcher analysis corroborated
- `ops/research_notes/20260429_fixH_failure_analysis.md` — Fix I specification confirmed already applied
