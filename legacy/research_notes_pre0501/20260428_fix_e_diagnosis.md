# Fix E Diagnosis — Slot Token Scale Blockade
**Date:** 2026-04-28 21:51 GMT+8  
**Author:** main agent (independent diagnosis after researcher stall)  
**Status:** COMPLETE — Fix E designed and ready for /coder

---

## Summary

After Fix D.1 (slot_output_gate_init=0.5 → alpha=0.462891), the gradient should flow:

```
LM loss → d/d(next_hidden) → × slot_delta → d/d(alpha) [slot_output_gate]
```

But WRITEBACK_DIAG shows `alpha(tanh_output_gate) = 0.462891` CONSTANT at ALL steps:
- step=97/fwd=200: 0.462891
- step=204/fwd=400: 0.462891
- step=298/fwd=600: 0.462891
- step=427/fwd=800: 0.462891

With Adam lr=3e-4, 430 steps of zero update means the gradient to `slot_output_gate`
is **literally zero**, not just small.

---

## Root Cause: Double Attenuation of Slot Tokens in Cross-Attention

### Layer.py lines 469-474 (current broken code):

```python
w_gathered = ste_weights.gather(1, idx).unsqueeze(-1)   # [B, k, 1]
M_sel_slot_w = M_sel_slot * w_gathered                  # ATTENUATION #1: × 1/512
M_sel_hidden = self.slot_to_hidden(M_sel_slot_w)        # ATTENUATION #2: std=0.02
```

**Attenuation #1: w_gathered ≈ 1/512 at uniform routing**
- At initialization, routing is uniform: all 512 slots get equal scores = 1/512 ≈ 0.00195
- `w_gathered = ste_weights.gather(1, idx)` = 0.00195 for each of the 64 selected slots
- `M_sel_slot_w = M_sel_slot × 0.00195` → 512× attenuation of slot values

**Attenuation #2: slot_to_hidden std=0.02**
- `||M_sel_hidden|| ≈ std × sqrt(d_out) × ||M_sel_slot_w||`
- `= 0.02 × 64 × (sigma × 64 × 0.00195)`
- `= 0.02 × 64 × sigma × 0.125`
- For sigma=0.01: `||M_sel_hidden|| ≈ 0.0016` (per slot token L2 norm)
- For sigma=0.05: `||M_sel_hidden|| ≈ 0.008`

**Combined: M_sel_hidden is 40,000× smaller than Llama hidden states (||H|| ≈ 64)**

| sigma | ratio(M_sel_hidden/H) | attenuation factor |
|---|---|---|
| 0.01 | 2.5e-5 | 40,000× |
| 0.02 | 5.0e-5 | 20,000× |
| 0.05 | 1.25e-4 | 8,000× |

**Consequence:** The 64 slot tokens prepended to the 4096 text tokens are essentially
invisible in cross-attention. Their K/V contributions to the softmax are exp(-∞) × 0.
The LlamaDecoderLayer therefore computes `ext_h[:,k_slots:,:] ≈ bypass_h` (same
text-only attention) → `slot_delta ≈ 0` → `d(loss)/d(slot_output_gate) ≈ 0`.

### Numerical confirmation:

`slot_delta_abs_mean ≈ 0.004-0.006` (observed) vs expected `||H|| per element ≈ 0.5-1.0`.
This 100-250× gap = small but non-zero slot leakage from the phantom exp(0) denominator terms.
The gradient is non-zero in magnitude but has ZERO CONSISTENT DIRECTION (random noise),
so Adam's running mean estimate stays ≈ 0 and the parameter never updates.

---

## Confirmation: NOT Caused by Attention Mask

`_build_extended_attn_mask` (layer.py lines 99-160) was verified:
- Text queries (rows k..k+T-1) have mask=0 for ALL slot columns (0..k-1)
- Text queries CAN attend to slot positions
- The mask is NOT the blockade

---

## Fix E Design

### Minimal fix (layer.py lines 469-474):

**Remove `w_gathered` attenuation from M_sel_hidden.** Apply STE gradient path via
additive correction (zero forward, non-zero backward):

```python
# Fix E (2026-04-28): Do NOT attenuate M_sel_slot by w_gathered before projection.
# Original code: M_sel_slot_w = M_sel_slot * w_gathered (w≈1/512 at uniform routing)
# then slot_to_hidden(M_sel_slot_w) → M_sel_hidden ~40,000× smaller than H → invisible.
# Fix: project at full scale. Preserve STE gradient via additive zero-valued correction.
# Forward: (w_gathered - w_gathered.detach()) = 0 → M_sel_hidden unchanged.
# Backward: d/d(w_gathered) = M_sel_hidden.detach() ≠ 0 → gradient flows to Q_sel/slot_keys.
w_gathered = ste_weights.gather(1, idx).unsqueeze(-1)   # [B, k, 1]; gradient path preserved
M_sel_hidden = self.slot_to_hidden(M_sel_slot)          # [B, k, d]; full scale (no × w_gathered)
M_sel_hidden = M_sel_hidden + M_sel_hidden.detach() * (w_gathered - w_gathered.detach())
```

**Why the STE correction works:**
- `(w_gathered - w_gathered.detach())` = 0 in forward pass (no effect on attention)
- `d/d(w_gathered)` of the correction = `M_sel_hidden.detach()` (non-zero)
- Gradient chain: `loss → slot_delta → cross_attn → M_sel_hidden → STE_correction → w_gathered → ste_weights → scores → Q_sel/slot_keys`

**Expected improvement after Fix E (sigma=0.01):**
- `||M_sel_hidden||` ≈ 0.82 per slot token (512× larger than before: 0.0016 → 0.82)
- ratio(M_sel_hidden/H) ≈ 1.3% (vs 0.00003% before)
- slot_delta should become non-zero and have consistent sign → alpha will decrease from 0.462891
- top1_sim should diverge from 1/512 uniform floor

### No change needed to slot_to_hidden std (0.02 is OK):
The 1.3% slot scale is sufficient to break the zero-gradient dead zone. With slot_output_gate
training, alpha will decrease toward 0, reducing slot influence on LM while the selector learns.

### WRITEBACK_DIAG update:
Add logging of `||M_sel_hidden||_mean` per forward call at the existing diagnostic print.
This confirms Fix E is working if `M_sel_hidden_norm_mean ≈ 0.8-4.0` (vs current ≈ 0.002-0.008).

---

## Files to Modify

1. `src/memory/mem_space/layer.py` — lines 469-474 (M_sel_hidden computation)
   - Remove `M_sel_slot_w = M_sel_slot * w_gathered`
   - Change `M_sel_hidden = self.slot_to_hidden(M_sel_slot_w)` → `slot_to_hidden(M_sel_slot)`
   - Add STE correction line
   - Update WRITEBACK_DIAG to log M_sel_hidden_norm_mean

No changes needed to selector.py, memory_bank.py, or config.py.

---

## Success Criteria for fix_e_ablation

- `top1_sim_mean > 0.005` at step 200 (vs current 0.002029-0.002182)
- `slot_delta_abs_mean > 0.01` at step 200 (vs current 0.004-0.006)
- `M_sel_hidden_norm_mean ≈ 0.8-4.0` at step 1 (confirming fix is active)
- `alpha(tanh_output_gate)` should change from 0.462891 by step 200

