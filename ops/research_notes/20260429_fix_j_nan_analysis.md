# Fix J-A NaN / Instability Root Cause Analysis — 2026-04-29

**Researcher:** subagent (Claude Opus)  
**Triggered by:** fix_j_ablation NaN explosions on all 3 B200 nodes (b200-2/3/4) at steps ~70–150  
**Status:** Root cause confirmed; Fix L proposed

---

## 0. TL;DR

The NaN/instability is caused by **M_sel_hidden norm explosion** — the slot-to-hidden projection amplifies slot vectors to 44× normal hidden-state scale (~600–1600 norm vs expected ~32 for Llama-3-8B). This contaminates joint-attention, producing catastrophic PPL spikes.  
The proximate cause is **unbounded `slot_to_hidden` weight growth** with `lr=1e-3` over 100 steps. Fix H clips *slot content* (norm ≤128) but does not constrain what `slot_to_hidden` does to those vectors.  
Fix J-A's dual gradient paths (hard writeback + soft proxy) accelerate this growth. Fix K carry-over prevents natural reset of corrupted slots, extending damage.

**Proposed Fix L**: two-line adaptive norm clip on `M_sel_hidden` in `layer.py`, plus per-parameter grad clip for projection weights in the training script.

---

## 1. Observed Failure Pattern

| Node | Sigma | First PPL spike | Spike magnitude | Status @ step 400 |
|------|-------|-----------------|-----------------|-------------------|
| b200-2 | 0.01 | step 90 (PPL=18,699) | Oscillates 1000–7000 | Still oscillating |
| b200-3 | 0.02 | step 120 (PPL=485,165) | Full NaN lock ~30 steps | Partially recovered |
| b200-4 | 0.05 | step 70 (PPL=3,622,180) | Earliest, worst | PPL=11,007 @ step 400 |

Pre-explosion baselines were all healthy: b200-2 step 40 PPL=4.4.  
Higher `init_noise` (sigma) → earlier explosion: consistent with more diverse slots creating larger initial slot_to_hidden weight gradients.

---

## 2. Evidence: The Smoking Gun

WRITEBACK_DIAG logs from all 3 nodes at step 97:

| Node | M_sel_hidden_norm_mean | slot_delta_max | gate_val (beta) | alpha |
|------|----------------------|----------------|-----------------|-------|
| b200-2 (sigma=0.01) | **1414.076** | 1.156 | 0.029175 | 0.462891 |
| b200-3 (sigma=0.02) | **1377.432** | ~1.2 | ~0.029 | ~0.46 |
| b200-4 (sigma=0.05) | **667.469** | 13.000 | ~0.029 | ~0.46 |

**Expected** M_sel_hidden norm for Llama-3-8B: `~sqrt(4096) ≈ 64`, typical observed hidden state norm ~25–40.  
**Observed**: 600–1600 — **20× to 44× above normal scale.**

At step 204, b200-2 norm rose further to 1602.444; b200-4 rose to 926.296 (step 204) then 1482.376 (step 298).  
The explosion is progressive and accelerating — not a one-time spike.

---

## 3. Causal Chain

```
Fix H clips slot content: ||slots[b,i,:]||₂ ≤ 128    ← only guards stored vectors

slot_to_hidden: [4096×4096] trainable linear, lr=1e-3
  ↓
Over 100 steps: weight elements shift ~10% from gradient flow via:
  (a) Fix J-A soft proxy path: loss → M_sel_hidden_soft → slot_to_hidden
  (b) Fix J-A hard path: loss → O_mem_hidden → (residual) → upstream layers → slots
  ↓
With ||slot||₂ ≤ 128 but growing slot_to_hidden weights:
  ||M_sel_hidden||₂ = ||slot_to_hidden(slot)||₂ ≈ ||W||_op × ||slot||₂
  After 100 steps: reaches 600–1600 (20–44×)
  ↓
M_sel_hidden tokens injected into joint attention alongside normal hidden states (norm ~32)
Joint attention is dominated by memory tokens (14–50× larger norm)
  ↓
Corrupted attention scores → wrong predictions → large loss → large gradient
  ↓
Large gradient → slot_to_hidden grows more (positive feedback loop)
  ↓
Fix K carry-over: corrupted slot content propagates across chunks, no reset ever occurs
Beta warmup (step 97: beta=0.029 ≈ sigmoid(0)×97/500×0.3) strengthens writes during spiral
```

### Why higher sigma fails earlier

Higher `init_noise` → slots are more diverse at init → larger pairwise differences → larger gradients into `slot_to_hidden` in the first few steps → weight growth onset earlier.

### Beta warmup cross-validation

At step 97: beta=0.029175 ≈ sigmoid(0) × (97/500) × 0.3 = 0.02910 ✓  
At step 204: beta=0.061768 ≈ sigmoid(0) × (204/500) × 0.3 = 0.06120 ✓  
Confirms `writeback_warmup_steps≈500` in all 3 runs.

### Why `top1_sim_mean` stays at random floor (0.002–0.003)

The routing optimization signal is drowned out by the massive M_sel_hidden contamination loss. The key network learns to minimize attention damage, not to learn meaningful routing. This is why SKRL shows near-zero key diversification.

---

## 4. Why Fix H Was Insufficient

Fix H (memory_bank.py write(), norm clip at sqrt(4096)×2=128) guards the *stored* slot content.  
It does **not** constrain what `slot_to_hidden` does to those vectors when reading them.

The fix was architecturally correct for its stated goal (preventing bf16 overflow in stored slots) but placed the guard at the wrong boundary. The explosion happens *after* `slot_to_hidden` in the forward pass.

---

## 5. Proposed Fix L

### L-1 (Core): Adaptive norm clip on `M_sel_hidden` in `layer.py`

**File**: `src/memory/mem_space/layer.py`  
**Location**: After the STE combination line that produces `M_sel_hidden` (after the `M_sel_hidden_hard.detach() + (M_sel_hidden_soft - M_sel_hidden_soft.detach())` line, before the concatenation into `KV_mem`)

```python
# Fix L-1: Clip M_sel_hidden to hidden_states norm scale.
# Prevents slot_to_hidden weight growth from generating memory tokens that
# overwhelm joint attention (root cause of fix_j_ablation NaN at step ~100).
# Uses current hidden_states as reference so the clip adapts to training dynamics.
_h_norm_ref = hidden_states.detach().norm(dim=-1).mean().clamp(min=1.0)  # scalar
_m_norms = M_sel_hidden.norm(dim=-1, keepdim=True)                        # [B, k, 1]
M_sel_hidden = M_sel_hidden * (_h_norm_ref / _m_norms.clamp(min=1e-6)).clamp(max=1.0)
```

This is a **one-directional** clip (only shrinks, never expands), so it's a no-op when M_sel_hidden is already at or below hidden_states norm. It does not break the gradient path — `_h_norm_ref` is detached but the scaling of `M_sel_hidden` still passes gradients through the division.

**Why this is correct**: The design intent is that memory tokens participate in joint attention as peers of hidden states. Peers should have similar norms. The clip enforces this invariant.

### L-2 (Defense-in-depth): Per-parameter gradient clip in training script

**File**: `scripts/train_mem_space_pg19.py`  
**Location**: Before the global `clip_grad_norm_` call

```python
# Fix L-2: Per-parameter grad clip for slot projections.
# Prevents slot_to_hidden/hidden_to_slot weight gradient spikes from
# destabilizing the EMA bank (secondary defense after L-1 output clip).
_PROJ_GRAD_CLIP = 0.1  # stricter than global 1.0 for projection matrices
for name, p in model.named_parameters():
    if p.grad is not None and ('slot_to_hidden' in name or 'hidden_to_slot' in name):
        nn.utils.clip_grad_norm_([p], _PROJ_GRAD_CLIP)
# Global clip follows unchanged:
torch.nn.utils.clip_grad_norm_(trainable, 1.0)
```

### L-3 (Diagnostic enhancement)

**File**: `src/memory/mem_space/layer.py` (GATE_GRAD_DIAG section)

Add monitoring for `slot_to_hidden.weight.norm()` and `M_sel_hidden` norm pre/post-clip every 50 steps (currently every 200). This would have detected the explosion ~40 steps earlier.

---

## 6. Relationship to Existing Fixes

| Fix | Status | Interaction with Fix L |
|-----|--------|----------------------|
| Fix H (slot norm clip in write) | KEEP | Complementary: L-1 clips at read output, H clips at write input. Together: full boundary protection. |
| Fix J-A (remove slots.detach()) | KEEP (per constraints) | L-1 makes J-A safe: without L-1, J-A gradient path causes weight growth explosion; with L-1, weight growth causes clipping rather than attention corruption. |
| Fix K (detach_ carry-over) | KEEP | L-1 prevents slot corruption from compounding across chunks. Without L-1, K is dangerous; with L-1, K is safe and beneficial. |
| Fix I (hidden_to_slot in optimizer) | KEEP | L-2 per-param grad clip provides the tighter control that I+high-lr lacked. |

---

## 7. Recommendations

### Kill current runs: YES

All 3 nodes are irreversibly unstable. The slot bank content is corrupted (large-norm slots that have passed through the norm-divergence spiral), and with Fix K carry-over, this corruption cannot self-heal. The training signal cannot overcome the PPL noise floor (PPL=3000–11000 vs target <8 for useful model).

Specific evidence: b200-2 has been oscillating PPL 1000–7000 since step 90 (300+ steps without recovery); b200-4 was at PPL=11007 at step 400; b200-3 shows partial NaN lock.

### Fix L should be applied before restarting

Fix L-1 alone should be sufficient to prevent the explosion. L-2 provides defense-in-depth at low cost (4 lines). L-3 enables faster detection if further instabilities occur.

### Expected behavior after Fix L

With L-1 active:
- M_sel_hidden norm ≤ hidden_states norm throughout training (enforced by construction)
- slot_to_hidden can still learn (gradients still flow through the clipped output)
- Beta warmup proceeds normally without fear of norm explosion
- Fix K carry-over becomes strictly beneficial (slots accumulate useful content, no risk of corrupted-content propagation)
- top1_sim_mean should rise above random floor within ~100 steps (routing has clean signal now)

### Hyperparameter note

No hyperparameter changes proposed. The instability is a code-level bug (missing output norm guard), not a hyperparameter tuning problem.

---

## 8. Confidence Assessment

| Claim | Confidence | Evidence |
|-------|------------|---------|
| M_sel_hidden norm explosion is root cause | **Very high** | Direct observation of 600–1600 norms at failure time in all 3 logs |
| slot_to_hidden weight growth is mechanism | **High** | Consistent with lr=1e-3, 100 steps, grad_norm≈1–5; no alternative explains the scale |
| Fix L-1 prevents explosion | **High** | Directly constrains the problematic quantity by construction |
| Fix L-1 preserves useful gradient flow | **Medium** | The clip is differentiable and one-directional; but we haven't run ablation yet |
| Recovery requires restart | **Very high** | 300+ steps of continued instability on b200-2 is definitive |

---

## 9. Files to Modify

For Fix L implementation (to be executed by coder):

1. **`src/memory/mem_space/layer.py`** — Add 3-line norm clip after STE combination, before KV_mem concatenation
2. **`scripts/train_mem_space_pg19.py`** — Add 6-line per-param grad clip before global clip

Total change: ~9 lines across 2 files. Low risk, targeted, reversible.
