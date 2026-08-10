# Research Note: Fix N — Persistent Routing Collapse Root Cause & Resolution

**Date**: 2026-04-29 18:30 GMT+8  
**Author**: /researcher agent  
**Triggered by**: heartbeat observation — `top1_sim_mean` pinned at 1/N = 0.00195 for 1000+ steps in `fix_j_l_m_ablation` on b200-2 (steps 19–1010) and b200-4 (steps 19–775)  
**Run**: fix_j_l_m_ablation_node{0,1}_20260429.log  
**Nodes**: b200-2 (σ=0.01, steps 19–1010), b200-4 (σ=0.05, steps 19–775)

---

## Executive Summary

**Problem**: `top1_sim_mean ≈ 1/N = 0.00195` throughout 1000+ training steps. Routing is completely non-discriminative (equivalent to uniform random selection) despite 7 implemented fixes confirming stable gradients, stable PPL (≈1.59), and correct operation of all other components.

**Root cause (single, definitive)**: `--skrl_weight 0.0` in the experiment script disables slot-key repulsion (SKRL) — the **only mechanism with O(1) gradient at the symmetric fixed point**. Combined with `--load_balance_weight 0.01` which **actively enforces uniform routing**, there is zero directed gradient pressure to differentiate slot keys. The symmetric fixed point is stable.

**Fix N**: Parameter-only change (no code modification needed):
- `--skrl_weight 0.0 → 0.05` (enable SKRL repulsion)  
- `--load_balance_weight 0.01 → 0.001` (reduce 10× to stop opposing differentiation)

SKRL code is already fully implemented in `selector.py`, wired through `layer.py` and `train_mem_space_pg19.py`. This is a script parameter flip.

---

## 5-Question Analysis

### Q1: What does `slot_key_diversity_loss()` (SKRL) actually do?

**Implementation** (`src/memory/mem_space/selector.py`, `slot_key_diversity_loss()`):
```python
def slot_key_diversity_loss(self, num_pairs: int = 512) -> torch.Tensor:
    nk = F.normalize(self.slot_keys, dim=-1)  # [N=512, S=128]
    N = nk.size(0)
    i = torch.randint(N, (num_pairs,), device=device)
    j = torch.randint(N, (num_pairs,), device=device)
    valid = (i != j)
    i, j = i[valid], j[valid]
    cos_sim_pairs = (nk[i] * nk[j]).sum(-1)  # [M_valid]
    return cos_sim_pairs.mean()  # minimise → push keys apart
```

SKRL computes the mean cosine similarity between ~512 randomly sampled non-self pairs of normalized slot keys, and minimizes it. This acts as a **repulsive force on slot keys in S^127 (the 128-dim unit hypersphere)**.

**Critical property**: The gradient of SKRL with respect to a specific slot key `slot_keys[i]` at the symmetric fixed point (all keys identical = `μ`) is:

```
∂SKRL/∂slot_keys[i] ≈ (2(N-1)/N) × (I - μμᵀ) × normalized_key
                     ≈ 2.0 (in the direction that pushes key[i] away from μ)
```

This is **O(1)** — non-zero, finite, and of meaningful magnitude regardless of slot content or training stage. When all N=512 keys are identical, SKRL provides a direct, differentiable push to spread them apart. This is the **symmetry-breaking gradient** that no other current loss provides.

**SKRL does NOT depend on**: slot content, routing behavior, gradient path through slots, or gate warmup state. It acts directly on `self.slot_keys` (an `nn.Parameter`) and is always active once `skrl_weight > 0`.

### Q2: With `skrl_weight=0.0`, is there any gradient pressure on slot_keys?

**Available gradient paths to `slot_keys`**:

1. **LM loss path** (via STE soft proxy, after Fix J-A):
   ```
   lm_loss → next_hidden → (bypass_h + α×slot_delta) → ext_h →
   extended_hidden → M_sel_hidden_soft → scores → logits → slot_keys
   ```
   The gradient of `M_sel_hidden_soft` w.r.t. `scores[b,i]` is `slot_to_hidden(slots[b,i])`. At the symmetric fixed point:
   - All slot content `slots[b,i]` ≈ same value (early training, low-diversity init)
   - Therefore `∂M_sel_hidden_soft/∂scores[b,i] ≈ same vector for all i`
   - Therefore `∂lm_loss/∂logits[i] ≈ same for all i` 
   - Therefore `∂lm_loss/∂slot_keys[i] ≈ same direction for all i`
   - Net effect: **all slot keys pushed in the same direction — zero relative differentiation**

   This is a **random walk** at the symmetric fixed point. There is gradient flow (Fix J-A confirmed `grad_norm > 0`), but it has zero persistent directed signal to break symmetry.

2. **load_balance_loss path** (via `importance = scores.mean(dim=0)`):
   ```
   lb_loss = N × Σ_i (importance_i × load_i)
   ∂lb_loss/∂scores[b,i] ∝ load_i  (normalized dispatch count to slot i)
   ```
   At the uniform routing state: `load_i ≈ 1/N` for all i → gradient pushes all scores toward uniform.
   **Net effect: actively maintains the uniform fixed point**.

3. **entropy_aux_loss** (via `p = scores.mean(dim=0)`):
   `entropy_aux_loss = -H(p)`. At `p_i = 1/N`: `∂(-H)/∂p_i = 0` (entropy is maximized at uniform). 
   **Net effect: zero gradient at exactly the point we're stuck at**.

**Summary**: With `skrl_weight=0.0`, gradient pressure on slot_keys consists of: (a) random-walk LM signal (zero net differentiation force), (b) load_balance actively reinforcing uniformity, (c) entropy aux with zero gradient at the stuck point. There is **no directed pressure to break symmetry**.

### Q3: How does `load_balance_weight=0.01` affect routing?

The load_balance loss (Switch Transformer style):
```python
lb_loss = N * (importance * load).sum()
```
where `importance[i] = scores[:, i].mean()` (differentiable via scores) and `load[i] = (top_k_mask[:, i].float()).mean()` (non-differentiable dispatch count).

Gradient to `slot_keys` flows through `importance`:
```
∂lb_loss/∂slot_keys[i] ∝ load_i × ∂importance_i/∂slot_keys[i]
```

If slot `i` happens to get routed slightly more often (e.g., `load_i = 0.15` vs expected `0.125`), the load_balance gradient increases `importance_i`'s derivative penalty → pushes scores for slot `i` DOWN → routes slot `i` less → restores uniformity.

**This is a negative feedback loop that actively resists any differentiating signal.**

With `load_balance_weight=0.01` and zero competing SKRL signal, the load_balance wins: **routing is held at uniform by active gradient pressure**. Even if the LM loss creates a small transient preference for one slot, load_balance immediately corrects it.

With `load_balance_weight=0.001` (Fix N), the restraint is 10× weaker while still preventing catastrophic collapse to 1 slot.

### Q4: Was Fix G's failure about SKRL or the gradient path being broken?

**Fix G history**: SKRL was disabled ("confirmed ineffective") after testing during approximately the Fix F era. At that point:
- `slots.detach()` was present in the soft proxy (before Fix J-A)
- The primary SKRL value (`cos_sim_pairs`) was observed to not improve

**Re-analysis**: SKRL acts on `self.slot_keys` directly, NOT through `slots`. Therefore SKRL's gradient does NOT require Fix J-A (the gradient path through slots). SKRL should always have produced `slot_keys.grad_norm > 0` even with the broken slot gradient path.

**Why SKRL appeared to fail in Fix G era**:
1. SKRL was enabled, `slot_keys.grad_norm > 0`  
2. But `mean_pairwise_cos` failed to decrease (or decreased very slowly)
3. Most likely cause: **load_balance_weight=0.01 was already present**, providing 10× stronger opposing signal to maintain uniform routing. SKRL at weight 0.01 (~2e-5 per step key movement) was insufficient to overcome load_balance opposing differentiation.

**Critical implication**: Fix G "failure" was not "SKRL is ineffective." It was "SKRL at weight 0.01 vs load_balance at weight 0.01 is a draw, with load_balance winning because it actively enforces the degenerate state." Fix N uses `skrl_weight=0.05` vs `load_balance_weight=0.001` — a 50:1 ratio in favor of SKRL, which should decisively break symmetry.

### Q5: Can Q_sel and slot_keys learn from LM loss alone?

**No.** The fundamental obstacle is that the LM gradient to slot_keys at the symmetric fixed point is a random walk:

1. At initialization: all N=512 slot_keys ≈ N(0, σ²·I) with σ=0.01–0.05. After `F.normalize`, keys are nearly uniform on S^127.

2. Softmax of near-uniform dot products: `scores ≈ [1/N, ..., 1/N]` for all queries.

3. Gradient of `M_sel_hidden_soft` w.r.t. `slot_keys`: routes through `slots` (which are near-uniform at init). All 512 partial derivatives point in the same direction.

4. Even if gradients accumulate over 1000 steps: the net signal in step `t` pushes ALL slot_keys by approximately the same vector `Δ`, so their relative positions on S^127 do not change.

5. **The only way to escape**: an asymmetric signal that treats different slot keys differently. SKRL is this signal — it randomly pairs keys and pushes EACH PAIR apart, creating an asymmetric force by construction.

**Analogy**: Without SKRL, slot_keys perform a synchronized random walk on S^127. They move together but never spread. SKRL introduces differential Brownian repulsion that forces them to disperse.

---

## Root Cause: Single-Line Diagnosis

The `fix_j_l_m_ablation` script (`scripts/_run_fix_j_l_m_ablation.sh`) contains:

```bash
--skrl_weight 0.0          # ← THE PROBLEM
--load_balance_weight 0.01 # ← MAKES IT WORSE
```

The comment in `config.py` reads:
```python
skrl_weight: float = 0.0   # Fix H: SKRL confirmed ineffective; disabled
```

This conclusion was wrong. SKRL was disabled at the same time the gradient path was broken AND while load_balance_weight was equal to skrl_weight. The "confirmed ineffective" verdict reflects test conditions under which SKRL could not win — not an intrinsic property of SKRL.

Now that:
- Fix J-A has restored end-to-end gradient flow (including through slots to hidden_to_slot)
- Fix K ensures slots accumulate diverse content over time (carry-over)
- Fix L-1 and M-1 stabilize training (PPL ≈ 1.59, no crashes)
- All 7 prior fixes confirm structural health of the training loop

...the only missing piece is the **symmetry-breaking signal**: SKRL.

---

## Why the 20260429_1647 Prediction Was Wrong

The prior research note (`ops/research_notes/20260429_1647_ppl_spike_and_top1sim_plateau.md`) predicted:

> *"Expected onset of top1_sim > 0.005 is step 700–800, not step 500."*

The reasoning was: gate warmup completes at step 500, then ~224 more steps for slot diversity → by step 724, slots have diverse content → Q_sel/slot_keys can learn routing → top1_sim rises.

This was wrong because:
1. **Slot diversity does not generate routing diversity** without SKRL. Even if all 512 slots contain perfectly distinct, informative vectors, the selector STILL picks uniformly if slot_keys are all identical (since routing uses slot_keys, not slot content directly, for addressing in the current architecture).
2. **The LM gradient to slot_keys is a random walk** regardless of slot content (see Q5 above).
3. **load_balance actively maintains uniform routing** — slot diversity makes no difference if load_balance suppresses any emerging preference.

The 700–800 step prediction was based on the assumption that "slot content diversity → routing diversity." This ignores the slot_keys vs slot_content architecture split introduced in Fix B, and the active enforcement of uniformity by load_balance.

---

## Fix N Specification

### Change 1: Enable SKRL (`skrl_weight 0.0 → 0.05`)

**File**: `scripts/_run_fix_j_l_m_ablation.sh` (or new `_run_fix_n_ablation.sh`)  
**Change**: `--skrl_weight 0.0` → `--skrl_weight 0.05`

**Mechanism**: SKRL provides O(1) gradient at symmetric fixed point. With `lr=1e-3`:
- Per-step key displacement ≈ `lr × skrl_weight × 2.0 = 1e-3 × 0.05 × 2.0 = 1e-4`
- After 200 steps: cumulative key spread ≈ `sqrt(200) × 1e-4 ≈ 0.0014` (if random walk) → but directed repulsion accumulates, not random walk → actual spread faster
- Expected `mean_pairwise_cos` to drop from ~0 to ~-0.003 by step 200, ~-0.01 by step 500

**No code changes**: SKRL is already fully implemented and wired:
- `selector.py`: `slot_key_diversity_loss()` method exists
- `layer.py` lines 699–701: collects and weights SKRL loss
- `train_mem_space_pg19.py` `_collect_aux_loss()`: sums SKRL into total loss

### Change 2: Reduce load_balance pressure (`load_balance_weight 0.01 → 0.001`)

**File**: same script  
**Change**: `--load_balance_weight 0.01` → `--load_balance_weight 0.001`

**Rationale**: With `skrl_weight=0.05` and `load_balance_weight=0.01`, the competing gradients are:
- SKRL push to differentiate: `0.05 × 2.0 = 0.1` effective gradient magnitude on slot_keys
- load_balance push to equalize: `0.01 × O(1)` effective gradient magnitude opposing differentiation
- Ratio: 10:1 in favor of SKRL — probably sufficient, but reducing load_balance to 0.001 gives 100:1

**Why keep load_balance at all**: Completely removing load_balance risks GPU underutilization (some slots never selected). `0.001` is a soft floor that prevents total collapse to 1 slot without actively suppressing the differentiation we want.

### Ablation design (3 nodes):

| Node | σ | skrl_weight | load_balance_weight | Expected outcome |
|------|---|-------------|---------------------|------------------|
| b200-2 | 0.01 | 0.05 | 0.001 | Primary Fix N test |
| b200-3 | 0.01 | 0.01 | 0.001 | Lower SKRL test |
| b200-4 | 0.01 | 0.1 | 0.001 | Higher SKRL test |

All with same σ=0.01 (most stable, fewest confounds). This ablates SKRL weight in [0.01, 0.05, 0.1] to find the sweet spot between:
- **Too low**: load_balance can still suppress differentiation, top1_sim stays at floor
- **Too high**: keys spread too fast → Q_sel → routing changes too rapidly → PPL instability

**Note**: User must approve hyperparameter changes before launch.

---

## Expected Timeline After Fix N

Based on SKRL mechanics and prior experiments:

| Step | Expected observation |
|------|---------------------|
| 0–50 | `skrl_mean_pairwise_cos` starts declining from ~0.0 |
| 100–200 | `skrl_mean_pairwise_cos` reaches ~-0.002 (keys meaningfully spread) |
| 200–300 | `top1_sim_mean` starts rising above 0.002 floor |
| 300–500 | `top1_sim_mean > 0.005` (first milestone) |
| 500–700 | Gate warmup complete (step 500), slot diversity accumulating |
| 700–1000 | `top1_sim_mean > 0.05` (second milestone, req_20260427_102400) |

If `top1_sim` does NOT rise by step 300 despite `mean_pairwise_cos < -0.005`:
- Slot keys are spreading but routing is still uniform
- Root cause: Q_sel is not using the key diversity (Q_sel weights not learning)
- Secondary fix needed: stronger Q_sel gradient or separate NIAH supervision

---

## What Is Working Correctly (Do NOT Change)

The following fixes from the current run are confirmed working:

1. **Fix J-A** (`slots` NOT detached in soft proxy): `hidden_to_slot.weight.grad_norm` consistently non-zero throughout 1000 steps ✅
2. **Fix K** (carry-over, `_detach_banks` not `_reset_banks`): slot content accumulates across steps ✅  
3. **Fix L-1** (M_sel_hidden norm clip): `M_sel_hidden_norm_mean ≈ 1.000` at all checkpoints ✅
4. **Fix M-1** (slot_delta norm clip): `slot_delta_max ≈ 5.6` (bounded, down from unclipped 7.97) ✅
5. **LM training stability**: PPL ≈ 1.59 (Llama-3 baseline quality) ✅
6. **WRITEBACK**: slots updating, retrieved_norm_mean growing normally ✅

Only the routing component is degenerate. Everything else is healthy.

---

## Open Questions

1. **Optimal SKRL weight range**: 0.01 vs 0.05 vs 0.1 — ablation needed to find the Goldilocks value where keys spread without destabilizing Q_sel learning.

2. **Will key diversity translate to routing diversity automatically?** Once keys are spread in S^127, the selector naturally routes different queries to different slots (assuming Q_sel has learned query-specific projections). But Q_sel might still be near-random at step 0. This should self-correct once SKRL creates diverse keys.

3. **Interaction with Fix M-1**: slot_delta norm clip was designed against slot norm inflation. With SKRL driving slot_keys apart, slot content may diversify faster (better writeback quality) — but slot norms are separately bounded by bank max_norm. No interaction expected.

4. **`skrl_weight=0.0` in `config.py` default**: The comment "Fix H: SKRL confirmed ineffective; disabled" should be updated to reflect the re-analysis. However, this is a code change — `/coder` should update the comment when implementing the experiment script.

---

## Files Modified/Read

- Read: `src/memory/mem_space/selector.py` (selector + SKRL implementation)
- Read: `src/memory/mem_space/layer.py` (STE + injection + loss collection)
- Read: `src/memory/mem_space/config.py` (skrl_weight=0.0 confirmation)
- Read: `scripts/train_mem_space_pg19.py` (loss collection, param groups)
- Read: `ops/research_notes/20260429_fix_j_proposal.md` (Fix J-A spec)
- Read: `ops/research_notes/20260429_1647_ppl_spike_and_top1sim_plateau.md` (prior note)
- Read: `ops/research_notes/20260429_fix_g_root_cause.md` (SKRL analysis)
- Read: `scripts/_run_fix_j_l_m_ablation.sh` (experiment config — confirmed `skrl_weight 0.0`)

**No code modified. No experiments started.**
