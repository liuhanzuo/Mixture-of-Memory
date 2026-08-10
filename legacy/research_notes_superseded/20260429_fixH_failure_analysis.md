# Fix H Failure Root Cause Analysis + Fix I Proposal
**Date**: 2026-04-29 13:52 CST  
**Author**: researcher subagent (abb2740867882e6e1, completed analysis via main agent synthesis)  
**Triggered by**: Fix H confirmed FAILED 13:13 on all 3 nodes (b200-2/3/4), step 410–427

---

## Executive Summary

Fix H (differentiable soft routing proxy + slot norm clipping) FAILED because it addressed the **wrong** root cause. Fix H fixed the STE gradient path to Q_sel/slot_keys, but the **actual** reason routing is degenerate is that the write path (`hidden_to_slot`) is **permanently frozen** and excluded from the optimizer. Without slot content ever diversifying during training, all downstream gradient signals collapse.

**Fix I**: Modify `_mem_space_params()` in `scripts/train_mem_space_pg19.py` to include `hidden_to_slot` parameters when `wrapper.config.hidden_to_slot_frozen == False`. This makes the existing `--unfreeze_hidden_to_slot` CLI flag actually work.

---

## Root Cause Chain (confirmed, all levels)

### Level 1 — Code Bug (PRIMARY ROOT CAUSE)

**`_mem_space_params()` permanently excludes `hidden_to_slot`** from the optimizer's `trainable` list, with comment:
```python
# hidden_to_slot: EXCLUDED (lines 163-165)
# frozen in __init__ because it participates in no gradient-bearing op.
```

This is architecturally incorrect: `hidden_to_slot` IS in a gradient-bearing op (writeback path). The comment reflects the state when `enable_writeback=False` but was left in when writeback was enabled.

**`_freeze_backbone()` makes `--unfreeze_hidden_to_slot` a no-op**:
```python
def _freeze_backbone(model):
    for p in model.parameters(): p.requires_grad = False  # freeze ALL
    for p in _mem_space_params(model): p.requires_grad = True  # unfreeze only listed
```
- Line 488: `hidden_to_slot_frozen = not args.unfreeze_hidden_to_slot` → sets `requires_grad=True` in `__init__`
- Line 565: `_freeze_backbone()` re-freezes ALL, then re-unfreezes only `_mem_space_params()`
- Since `hidden_to_slot` is never in `_mem_space_params()`, it gets re-frozen regardless of flag

**Evidence**: `hidden_to_slot.weight.grad_norm = None` at ALL GATE_GRAD_DIAG checkpoints (steps 5–20) on all nodes. Write path never receives gradient.

### Level 2 — Architectural Consequence

With `hidden_to_slot` frozen:
- `write()` still calls `self.hidden_to_slot(O_mem_hidden)` — but using fixed (untrained) projection
- Slot content can only change via the initial hidden_pool mean-pool init at each chunk reset
- All 512 slots initialized to `mean(H_l) ± σ=0.02` → near-identical

**Evidence (memory_bank.py lines 143-154)**:
```python
elif self.slot_init == "hidden_pool":
    pooled = H_l.detach().mean(dim=1, keepdim=True)  # [B, 1, d]
    slots = pooled.expand(B, N, d).contiguous().clone()
    if self.init_noise > 0.0:
        slots = slots + torch.randn_like(slots) * self.init_noise
self.slots = slots.detach()
```
All 512 slots start as the same vector + σ=0.02 noise. `slot_dim=128`, so slot diversity = 0.02/||mean|| ≈ tiny.

### Level 3 — Routing Signal Collapse

Identical slots → attention scores uniform → M_sel = mean of identical slots ≈ any slot → `slot_to_hidden(M_sel)` ≈ identical for all selected slots → `M_sel_hidden ≈ ext_h[:, k:, :]` (all slots look the same as bypass_h) → `slot_delta = ext_h[:, k:, :] - bypass_h ≈ 0`.

With `slot_delta ≈ 0`:
```python
next_hidden = bypass_h + alpha * slot_delta   # slot_delta ≈ 0 → no learning signal
```
→ gradient to `slot_output_gate` via `slot_delta` ≈ 0 → alpha barely changes from init.

**This is why alpha = 0.462891 appears "stuck"**: the param DID update (0.5 → ~0.506), but very slowly because the gradient is structurally near-zero, not because the gate is missing from the optimizer.

### Level 4 — Why Fixes A–H All Failed

| Fix | What it addressed | Why insufficient |
|-----|-------------------|-----------------|
| A (slot_init_noise=1.0) | Diversity at init | Init overridden by σ in ablation script (--slot_init_noise 0.01); even if σ=1.0, write path still frozen → slots drift back to mean after first write |
| B (learnable slot_keys) | Key diversity | Keys were already diverse (mean_pairwise_cos≈0 at init = uniform on S^127); not the bottleneck |
| C (cosine norm + T=10) | Sharper routing | Sharper routing of random content = still random top-1 |
| D (entropy_aux) | Force key diversity | Same as B |
| E (various) | Various | Write path still frozen |
| F (centered STE) | Gradient to Q_sel via STE | M_sel_centered≈0 because all slots identical; centering of identical vectors = zero vector |
| G (SKRL) | Key repulsion | Premise wrong: keys already diverse (S^127 init); SKRL on diverse keys gives gradient ≈ 0 |
| H (soft proxy STE) | Gradient to Q_sel | STE path is sufficient but cannot fix degenerate slot content caused by frozen write path |

---

## Fix I Specification

### File: `scripts/train_mem_space_pg19.py`

**Function**: `_mem_space_params()` (lines 137–169)

**Change**: After the `slot_to_hidden` parameter loop, add conditional inclusion of `hidden_to_slot`:

```python
# Fix I (2026-04-29): include hidden_to_slot when explicitly unfrozen via --unfreeze_hidden_to_slot
# Root cause: _freeze_backbone() re-freezes hidden_to_slot even when flag is set,
# because _mem_space_params() never included it. Write path must be trainable
# for slot content to diversify → routing signal to develop.
if not getattr(wrapper.config, 'hidden_to_slot_frozen', True):
    for p in wrapper.hidden_to_slot.parameters():
        if id(p) not in seen:
            params.append(p)
            seen.add(id(p))
```

**This fix**:
1. Makes `--unfreeze_hidden_to_slot` actually work (currently broken no-op)
2. Write path trained → slot content diversifies per batch → routing signal develops
3. Minimal change: 6 lines, no architectural modification

### Verification Criteria (Fix I success)

At step 500 with `--unfreeze_hidden_to_slot`:
- `hidden_to_slot.weight.grad_norm` ≠ None (was None on all prior fixes)
- `top1_sim_mean > 0.005` (was stuck at floor 0.00195 = 1/512)
- `alpha` visibly changing across GATE_GRAD_DIAG steps

At step 1000:
- `top1_sim_mean > 0.05` (unblocks req_20260427_102400_scale_up_N1024)

### Optional Secondary Fix (if Fix I alone insufficient)

If slot content still degenerates despite unfrozen write path (e.g., EMA collapse):
- Increase `init_noise` to 0.1–0.5 (currently 0.02) to give more initial slot diversity
- Consider per-slot distinct token initialization instead of mean-pool broadcast

---

## Files Read During Analysis

- `src/memory/mem_space/layer.py` (lines 290–330, 480–510, 590–620)
- `scripts/train_mem_space_pg19.py` (lines 137–177, 488, 565, 602, 725–747)
- `src/memory/mem_space/memory_bank.py` (lines 100–180)
- `logs/fix_h_ablation_node0_20260429_1257.log` (QUERY_DIAG + GATE_GRAD_DIAG checkpoints)

---

## Confidence Assessment

- Root cause identification: **VERY HIGH** (confirmed by three independent evidence sources: GATE_GRAD_DIAG logs, code audit, mathematical analysis)
- Fix I correctness: **HIGH** (minimal targeted change to fix a clear code bug)
- Fix I timeline to routing recovery: **MEDIUM** (depends on EMA learning rate and slot diversity speed)
