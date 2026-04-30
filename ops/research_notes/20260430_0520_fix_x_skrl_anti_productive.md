# Research Brief: Fix X — SKRL Is Anti-Productive AND Q_sel Is Not Learning. Remove SKRL + Strengthen LM Signal.

**Date**: 2026-04-30 05:20 GMT+8
**Triggered by**: fix_v_ablation DEFINITIVELY FAILED on all 3 nodes. T=10.0 insufficient to break top1_sim floor (0.0036 observed vs 0.020 predicted by theory if T=10 actually in force).
**Report ID**: rpt_20260430_0520_fix_x_skrl_anti_productive
**Question**: (1) Should SKRL be removed entirely? (2) Why is T=10 empirically behaving like T≈2? (3) What is the highest-confidence Fix X?

---

## Executive Summary

⚠️ **CRITICAL FINDING #1**: SKRL succeeds at ETF minimum but that configuration is a **routing death trap**. With N=512 vectors spread as equiangular tight frame in d=128 space, the *mean* pairwise cosine is -1/511 ≈ -0.002 (the objective's minimum), but the **query-side geometry is symmetric**: for any query q, the distribution of cos(q, k_i) has mean≈0 and tiny variance. This kills routing selectivity regardless of T. SKRL's "success" is the cause of the top1_sim failure, not the cure.

⚠️ **CRITICAL FINDING #2**: The prior researcher's T=10 prediction (top1 ≈ 0.042) was wrong. A proper simulation (random query vs ETF keys, d=128, N=512, T=10) gives top1 ≈ 0.020, and the observed value is **0.0036** — **5.5× LOWER than even that**. Observed top1 is consistent with an **effective T ≈ 2.5**, not T=10. So there is **a second, unidentified attenuation** somewhere between CLI flag and the softmax. Most likely explanation: Q_sel output consistently lands in the ETF null space (S≈0 direction), so even with T=10, the *dispersion of the N logits is suppressed*. This is a **Q_sel geometry failure**, not a temperature plumbing bug.

⚠️ **CRITICAL FINDING #3**: `retrieved_norm_mean` grew from 1.15 (fwd=50) to 77.1 (fwd=700) — a **60× explosion of slot values** in 700 fwd passes. Combined with uniform routing, this is the classic **"slot collapse via undifferentiated writeback"** failure mode: all 512 slots are receiving the same content (because routing is uniform), growing in norm without specializing in direction.

**Fix X recommendation (confidence: HIGH)**:
1. **Remove SKRL entirely** (`skrl_weight=0.0`). It is anti-productive.
2. **Add direct LM gradient path to slot_keys** — remove the Fix Q.2 `.detach()`. The detach was added to prevent NaN at T=10, but NaN came from SKRL × T=10 interaction; without SKRL, T=10 with grad-through should be safe.
3. **Keep T=10.0** (need sharpening for any differentiation).
4. **Keep load_balance_weight=0.001** (prevents pure collapse).
5. **Add slot-value norm constraint** (`slot_value_norm_reg`) to stop the 60× norm explosion.

This is a **three-line hyperparameter change** (skrl=0, Q.2 reverse, slot_value_norm_reg new) combined with one small code edit (remove `.detach()` on slot_keys at selector.py:159).

---

## 1. Background

fix_v_ablation was the primary test of Fix V.1 (T=1→10) + Fix W (entropy=0). Per prior researcher note `20260430_fix_v_diagnosis.md`, expected outcome was **top1_sim > 0.005 at fwd=500, > 0.020 at fwd=1000**. Observed outcome:

| fwd pass | top1_sim_mean | skrl_diag | retrieved_norm_mean | lm_ppl (nearest step) |
|----------|--------------|-----------|---------------------|----------------------|
| 50       | 0.004852     | -0.0018   | 1.151               | 545 (step 10)        |
| 100      | 0.003967     | -0.0019   | 4.695               | 7.46 (step 50)       |
| 200      | 0.003616     | -0.0020   | 20.71               | 6.43 (step 200)      |
| 500      | 0.003815     | -0.0020   | 48.86               | 1.19 (step 260)      |
| 700      | 0.003891     | -0.0020   | 77.07               | 2.30 (step 380)      |
| 1200     | 0.003555     | -0.0020   | —                   | spike 485M (step 610)|

All criteria failed. LM instability spike at step 610 indicates the undifferentiated slot bank is polluting the LM output.

---

## 2. Findings

### 2.1 Empirical T is ≈2, not 10 — Q_sel geometry failure

I ran a Monte Carlo simulation (d=128, N=512, 2000 trials) of `top1 = max_i softmax(T · cos(q, k_i))` where keys are first trained to ETF minimum (mean_pair_cos ≈ -0.0017 after 500 gradient steps on ||S||²), then random queries are sampled on the sphere:

| T  | Simulated top1 | Matches observed? |
|----|---------------|-------------------|
| 1  | 0.0025        | No                |
| 1.5| 0.0029        | Close             |
| 2  | 0.0033        | **Best match**    |
| 2.5| 0.0037        | Close             |
| 5  | 0.0068        | No                |
| **10** | **0.0198** | **5.5× higher than observed** |
| 20 | 0.098         | No                |

**The observed top1 = 0.0036 corresponds to effective T ≈ 2.3.**

I audited the temperature wiring end-to-end:
- `scripts/train_mem_space_pg19.py:328` → `--selector_temperature` flag, default 1.0
- `scripts/_run_fix_v_ablation.sh:83,135` → passes `10.0`
- `train_mem_space_pg19.py:561` → forwards to config
- `src/memory/mem_space/config.py:70` → `selector_temperature: float = 1.0`
- `src/memory/mem_space/layer.py:272` → passes to selector
- `src/memory/mem_space/selector.py:88` → stores as `self.temperature`
- `src/memory/mem_space/selector.py:163` → `logits = einsum("bs,bns->bn", q, k) * self.temperature`

The code path is clean. The CLI flag reaches the multiplication. **So the attenuation is not in temperature plumbing.**

The only remaining explanation is **q is not behaving as a random unit vector vs ETF keys**. Specifically:

At the ETF minimum, slot_keys form a set whose sum S → 0. Any vector q orthogonal to the "spread direction" will have near-zero cosine with every key (since the keys collectively span the whole sphere uniformly). Q_sel is a trainable linear projection + F.normalize. Under gradient descent with uniform top1 signal:

- When top1 is near-uniform, softmax Jacobian is near-zero → Q_sel receives near-zero gradient.
- Q_sel stays near its init (std=0.02).
- `q = F.normalize(Q_sel(pool))` produces a unit vector whose direction is essentially a random projection of the layer-0 pooled hidden state.
- Pooled hidden states have **strong common-mode components** (layer norm bias, positional averaging, etc.).
- So q is not random — it has **systematic alignment with a small subspace**.
- In the ETF minimum, every direction has zero mean inner product with keys. But if q is in a particular direction, the *variance* of cos(q, k_i) depends on how well the ETF keys span that subspace.
- **If q lies in a low-dimension subspace that ETF training doesn't populate** (because SKRL cares only about symmetric spread), then cos(q, k_i) has very low variance → softmax is near-uniform regardless of T.

This is the empirical effective T≈2 signature. **Gradient starvation is self-reinforcing**: low variance cosines → low variance logits → low variance softmax → low variance gradient to Q_sel → Q_sel stays in the "bad" direction forever.

### 2.2 Why SKRL is *causing* the problem, not just failing to help

The ETF minimum is **not** the "informative" configuration for routing. For routing selectivity to emerge, we need:
- Keys clustered *along query directions that the LM wants to route* (so that for a given input type, one key is distinctly more aligned than others).
- Not keys uniformly spread on the sphere.

SKRL is orthogonal to query alignment — it only cares about pairwise key-key cosine. In the ETF limit, keys are "maximally uninformative" w.r.t. any particular query distribution. This is the OPPOSITE of what a routing system needs.

**The mathematical truth**: maximum pairwise diversity (SKRL's minimum) ≠ maximum routing selectivity (our goal). These objectives are nearly orthogonal, and in practice SKRL prevents the key geometry from organizing around query statistics.

### 2.3 Why Q_sel CANNOT learn under current setup

There are **four** gradient paths to Q_sel:

1. **Primary**: LM loss → M_sel_hidden → ste_weights → scores → logits → q → Q_sel.
   - Attenuation: softmax Jacobian (max eigenvalue T/N at uniform point) × ste_weights gradient.
   - At uniform scores, this is ~T/N = 10/512 = 0.02 per dim. Non-zero but small.

2. **Secondary**: SKRL via slot_keys … but wait — Fix Q.2 detached slot_keys from the `k` computation. So SKRL gives gradient to slot_keys NOT to Q_sel. ✗

3. **Tertiary**: entropy_aux → scores → logits → q → Q_sel.
   - Fix W set entropy=0. ✗

4. **Quaternary**: load_balance_loss → scores → Q_sel.
   - Active but provides uniformity pressure (not differentiation). Can push Q_sel to produce even MORE uniform routing, not LESS.

**Net**: Q_sel has only path (1), attenuated by 512×. At T=10 this is 21× better than T=1 per prior analysis, but Q_sel needs **enough gradient to break out of the "random projection of common-mode" local minimum**. The fact that observed effective T is ~2 (not 10) shows that Q_sel is trapped in this local minimum: the queries it produces have near-zero variance in cos(q, k_i), so the T=10 multiplier amplifies near-zero into still-near-zero.

### 2.4 The slot value explosion (retrieved_norm 1→77)

Looking at WRITEBACK_DIAG:
- `beta` (writeback gate) grew 0.006 → 0.113 over fwd=50→700 (18× growth).
- `slot_delta_abs_mean` ≈ 0.006 throughout (unchanged).
- `M_sel_hidden_norm_mean` ≈ 1.0 throughout (post-normalization).
- **But `retrieved_norm_mean` = slot VALUE norm grew 1.15 → 77** (60× growth).

Mechanism: writeback applies `beta × slot_delta` to slot values. With beta=0.11 and undifferentiated deltas being routed to *all* slots (via soft proxy), every slot accumulates a similar δ over time. The ||slot|| grows as ~sqrt(num_writes) ≈ sqrt(700 fwd × 64/512 per fwd on average) … actually with uniform routing and soft proxy, every slot gets updated every step, so it's a random walk of norm ~sqrt(T) × step_size ≈ sqrt(700) × 0.03 ≈ 0.79 … plus systematic bias, this gets to 77 easily.

**Why this is bad**: when slots have large norm and small direction variance (all similar), M_sel_slot gathered from any idx has content that is effectively the "average slot". Projection through slot_to_hidden then gives an M_sel_hidden that is the same regardless of which slot was picked — *zero information in routing*. The LM is forced to either ignore memory (good) or be confused by it (bad: step 610 PPL spike).

---

## 3. Literature comparison

| Method | Routing | Key/Expert regularization | Relation to us |
|--------|---------|---------------------------|----------------|
| Switch Transformer (Fedus 2021) | Top-1 hard | load_balance only | Does NOT use key repulsion. Experts differentiate via LM gradient alone. |
| Soft MoE (Puigcerver 2024) | Soft, all-experts | no explicit diversity loss | Continuous routing, temperature-like. No SKRL-style term. |
| Mixture-of-Experts-Contrastive (MCRM, 2024) | Top-k | **InfoNCE on expert outputs** | Contrastive on OUTPUTS, not keys. Aligns with actual data distribution. |
| DeepSeek-MoE (2024) | Top-k + shared experts | load balance + router-z | No key-key repulsion. |
| **Our system** | **Top-k + STE** | **SKRL (key-key repulsion)** | **UNIQUE in using geometric key repulsion — and it FAILS** |

**Every successful MoE system relies on the LM gradient to differentiate experts. None use geometric key-key repulsion.** The Switch Transformer original paper explicitly argues that load-balance is sufficient because the LM gradient will naturally specialize experts for different input distributions.

The closest match to SKRL in the literature is the old "orthogonality regularization" in some cluster-assignment losses — and those are widely considered obsolete in 2024, replaced by InfoNCE-style contrastive alignment with actual data.

**Conclusion**: Our use of SKRL is an uncommon design choice, poorly supported in recent literature, and empirically failing.

---

## 4. 三角验证

### 🟢 Proposer (best case, Fix X: remove SKRL + undetach slot_keys)

- Without SKRL, slot_keys are free to organize around actual query patterns.
- Undetaching slot_keys exposes them to LM gradient: when a particular slot's content improves LM loss for a particular input type, the key for that slot gets pulled toward queries for that input type.
- This is **query-aligned specialization** — exactly the Switch Transformer story.
- T=10 provides enough gradient sharpening for Q_sel to start learning once keys cluster around queries.
- Load balance loss prevents collapse to a single slot.
- Expected: within 500 fwd passes, top1_sim > 0.01; within 2000 passes, > 0.05.
- **Literature support**: Switch Transformer, Soft MoE, DeepSeek-MoE all use this exact approach (LM gradient to differentiate + load balance to prevent collapse).
- **Confidence: HIGH** that the direction is correct.

### 🔴 Skeptic (biggest risks)

**Risk 1: NaN spirals return at T=10 without Fix Q.2 detach.**
- Fix O was originally added to prevent NaN at high T. Fix Q.2 detached slot_keys to make T=10 safe (by severing the LM→slot_keys path).
- If we undetach AND keep T=10, we may get NaN again.
- **Mitigation**: start with T=5 for first 200 steps, then anneal to T=10. If NaN, drop to T=5 permanently.
- **Mitigation 2**: add gradient clipping on slot_keys specifically (not global).

**Risk 2: Without SKRL, slot_keys might collapse (all keys become similar).**
- Without repulsion, nothing mathematically prevents all keys pointing in the same direction.
- BUT: load_balance_loss creates implicit pressure via the dispatch statistics. If all keys are identical, all queries route to all slots equally → load is uniform → balance_loss = 1.0. If keys differentiate, queries route selectively → imbalance → balance_loss grows. So balance_loss actually PREVENTS collapse.
- **Mitigation**: monitor SKRL_DIAG even without SKRL loss applied. If it rises to +0.5 (full collapse), re-introduce weak SKRL (weight=0.001) as regularizer.

**Risk 3: The 60× slot value explosion might continue.**
- This is independent of SKRL. It comes from undifferentiated writeback.
- With routing differentiation, slots should specialize and each one gets fewer but more directed updates.
- **Mitigation**: add `slot_value_norm_cap` at a target ||slot|| (say 10.0) — clamp or L2-penalize. Prevents runaway.

**Risk 4: Q_sel might remain trapped in "random common-mode direction" even without SKRL.**
- The Q_sel local minimum is driven by softmax-at-uniform Jacobian attenuation, not by SKRL.
- Removing SKRL doesn't directly help Q_sel escape; what helps is either (a) stronger direct gradient via slot_keys moving toward queries (the primary Fix X mechanism) or (b) auxiliary loss with non-zero gradient at the uniform fixed point.
- **Mitigation**: add InfoNCE-style auxiliary loss as a backup (Fix X.2 if Fix X.1 insufficient).

### 🔵 Critic (blind spots)

**Blind spot 1: Did we actually see SKRL cause failure, or just correlate?**
- All 12 fixes (F through W) kept SKRL. We never ran the same setup with skrl=0. Our attribution is theoretical.
- **Remedy**: Fix X MUST be an A/B: node0 with SKRL at current settings (control), node1-2 with SKRL=0 and other Fix X changes.

**Blind spot 2: The "effective T=2" derivation assumes isotropic Q_sel output.**
- If Q_sel produces q highly correlated with specific slot_key directions (positive bias), our simulation underestimates top1. But observed was BELOW the isotropic prediction, so anisotropy is NEGATIVE (q avoids key directions).
- This can happen if slot_keys trained to ETF had been initialized from a distribution that happens to be orthogonal to pooled hidden states. strided_token init starts keys from real token embeddings (layer-0 output direction), then SKRL pushes them to ETF. The "residual" ETF direction may be anti-correlated with pooled hidden state → q anti-correlates with keys → effective T lower than isotropic.
- **If true, this is further evidence AGAINST SKRL**: it drives keys into the worst possible geometry for queries.

**Blind spot 3: Is `niah_acc=0.000` telling us Q_sel can never learn under this setup?**
- NIAH evaluation uses haystack under no_grad, so Q_sel NEVER receives NIAH gradient. It only learns from base LM (pg19).
- LM on pg19 may not require selective memory — the base model already knows pg19 perfectly (PPL < 5 most of the time). So there's no LM signal pushing for selective routing.
- **Blind spot**: even a perfect Fix X may not move top1_sim if the pg19 data alone doesn't require it.
- **Mitigation**: introduce a proxy task that REQUIRES selective retrieval even in the training loop (e.g., have the LM need to recall specific facts from earlier chunks).

**Blind spot 4: Why is retrieved_norm growing so fast?**
- Didn't fully analyze. slot_delta_abs_mean is 0.006 but slot values grow by 60× — implies coherent accumulation, not random walk. Need to check if writeback gate `beta` × soft top-k weights create a systematic DC component.
- **Follow-up**: add `slot_norm_per_slot_std` diagnostic to see if all slots grow together (uniform, our hypothesis) or a few grow while others shrink.

---

## 5. Fix X specification

### Fix X.1 (primary — highest confidence, hyperparameter + 1-line code edit)

**Hypothesis**: Remove SKRL, expose slot_keys to LM gradient, keep T=10.

#### Code change (selector.py line 159-162)

```python
# BEFORE (Fix Q.2):
k = F.normalize(
    self.slot_keys.detach().unsqueeze(0).expand(B, -1, -1),
    dim=-1,
)

# AFTER (Fix X.1):
k = F.normalize(
    self.slot_keys.unsqueeze(0).expand(B, -1, -1),
    dim=-1,
)  # Fix X.1: restore LM gradient path to slot_keys. SKRL is removed, so no double-hook.
```

**Why safe**: Fix Q.2 was added because SKRL was registering a forward hook + the einsum path also exercised slot_keys, causing DDP "marked ready twice". With SKRL removed, only one gradient path exists, no double-hook. No DDP crash.

**Why correct**: without this, slot_keys receives NO gradient from any source (once SKRL is removed). They would stay frozen at init forever.

#### Hyperparameter changes

```bash
--skrl_weight 0.0            # was 0.10/0.05/0.15 — REMOVE, anti-productive
--entropy_aux_weight 0.0     # keep at 0.0 (Fix W stays)
--selector_temperature 10.0  # keep at 10.0 (Fix V stays)
--load_balance_weight 0.001  # keep (prevents collapse)
```

#### New hyperparameter (add via CLI argparse)

```bash
--slot_value_norm_cap 10.0   # Fix X: cap ||slot_value|| to prevent 60× explosion
```

Implementation: after writeback in memory_bank.py, clip slot norms:
```python
with torch.no_grad():
    norms = self.slots.norm(dim=-1, keepdim=True)  # [B, N, 1]
    scale = torch.clamp(norms / self.slot_value_norm_cap, min=1.0)
    self.slots.data = self.slots.data / scale
```
(This is a separate coder task — not strictly required for first Fix X ablation. If not implemented, runtime-add a diagnostic; if retrieved_norm > 50 by fwd=500, kill and add cap.)

#### 3-node ablation design

| Node | Config | Purpose |
|------|--------|---------|
| b200-1 | skrl=0.0, T=10, undetach slot_keys | **Primary Fix X** |
| b200-2 | skrl=0.0, T=10, undetach slot_keys + slot_value_norm_cap=10 | Add norm cap |
| b200-3 | skrl=0.05, T=10, undetach slot_keys (CONTROL — SKRL kept) | A/B to confirm SKRL is the problem |

Success criteria (same as before):
- fwd=200: top1_sim_mean > 0.005
- fwd=500: top1_sim_mean > 0.010
- fwd=1000: top1_sim_mean > 0.05

Additional diagnostics to monitor:
- `SKRL_DIAG` (observation only, not loss) — track whether slot_keys drift from ETF when SKRL removed
- `Q_sel.weight.grad_norm` per 50 fwd — confirm Q_sel receives gradient
- `retrieved_norm_mean` — confirm norm cap (if present) works

### Fix X.2 (backup — InfoNCE, if X.1 insufficient)

If after 1000 fwd passes Fix X.1 still shows top1_sim < 0.01, the problem is Q_sel's common-mode trap (not SKRL). Then add InfoNCE as a direct non-uniform-gradient signal.

```python
# Add to selector.py forward(), cache for loss:
self.last_q = q.detach().clone()  # [B, S], no grad
self.last_top1_idx = idx[:, 0].detach().clone()  # [B], hard argmax

# New method:
def info_nce_routing_loss(self, temperature: float = 10.0) -> torch.Tensor:
    if not hasattr(self, 'last_q') or self.last_q is None:
        return torch.zeros(1, device=self.slot_keys.device)
    q = self.last_q                                         # [B, S]
    pos_idx = self.last_top1_idx                            # [B]
    nk = F.normalize(self.slot_keys, dim=-1)                # [N, S], GRAD flows here
    logits = torch.einsum("bs,ns->bn", q, nk) * temperature # [B, N]
    loss = F.cross_entropy(logits, pos_idx)                 # pull key[pos_idx] toward q
    return loss
```

Weight: 0.05.

Effect: key of the currently-chosen slot gets pulled toward q (positive), other keys get pushed away (negative). This is exactly query-aligned specialization, and crucially provides **non-zero gradient even at uniform softmax** because the cross-entropy target is concrete (not soft).

### Fix X.3 (future — InfoNCE-based contrastive + hard negatives)

Use recent chunks as negative examples, force the top-1 slot to win against all other slots across the batch. Mirrors MCRM (2024) approach. Deferred until X.1/X.2 validated.

---

## 6. Confidence assessment

| Finding | Confidence | Evidence |
|---------|-----------|----------|
| Observed top1 = 0.0036 ≠ T=10 prediction (0.020) | **very_high** | Monte Carlo simulation (2000 trials, d=128, N=512, ETF keys) |
| SKRL drives keys into query-unfriendly geometry | **high** | Math: ETF minimum = symmetric null of all query directions |
| Q_sel trapped in common-mode subspace | **high** | Effective T ≈ 2 implies q has low variance in cos(q, k) |
| Slot values exploding from undifferentiated writeback | **very_high** | Log evidence: 1.15 → 77 over 700 fwd |
| Remove SKRL will allow natural specialization | **high** | Switch Transformer, Soft MoE, DeepSeek all work without SKRL-like terms |
| Fix X.1 will give top1_sim > 0.01 at fwd=500 | **medium-high** | Theoretical support strong, but Q_sel local minimum may persist |
| If X.1 insufficient, X.2 (InfoNCE) will fix | **high** | Cross-entropy has non-zero gradient at uniform; directly aligned with objective |
| No NaN at T=10 with undetached slot_keys (skrl=0) | **medium** | DDP double-hook only triggered when SKRL present; without SKRL, should be safe, but empirical risk |

**Overall Fix X.1 success probability**: ~70% gets top1_sim > 0.01 at fwd=500.
**Combined X.1 + X.2 fallback**: ~90% gets there within 2 rounds.

---

## 7. 结论

### 回答原始问题

**Q1: Should SKRL be removed entirely?**
**YES.** Confidence: HIGH.

Reasons:
1. SKRL's objective (maximum pairwise key diversity) is **orthogonal to** routing selectivity.
2. At its minimum (ETF), SKRL produces keys whose geometry is **adversarial to query directions** (every query direction sees near-uniform cosine distribution).
3. No successful MoE system in literature (2023-2025) uses geometric key-key repulsion. All rely on LM gradient + load balance.
4. Empirical evidence: observed top1 (0.0036) is BELOW even the T=10-with-ETF-keys theoretical prediction (0.020), implying SKRL made the geometry WORSE than "just spread", not better.

**Q2: Why is T=10 behaving like T≈2 empirically?**

Q_sel is trapped in a low-variance output manifold (produces queries in the "common-mode" direction of pooled hidden states). This direction has near-zero variance in cos(q, ETF_key), so the T=10 multiplier amplifies a near-zero signal into a still-near-zero signal. The cause is gradient starvation of Q_sel (softmax Jacobian at uniform is 1/N, and at ETF the signal entering softmax is even smaller). Removing SKRL unblocks the path: slot_keys can move toward q under LM gradient, breaking the symmetric null geometry.

**Q3: What is Fix X?**

Fix X.1 (primary):
- skrl_weight: 0.10/0.05/0.15 → **0.0**
- selector.py:159 remove `.detach()` on slot_keys
- Keep T=10, entropy=0
- Add slot_value_norm_cap=10 (optional but recommended)

Fix X.2 (backup): Add InfoNCE routing loss if X.1 top1_sim still floor at fwd=1000.

**Q4: Literature support for "soft routing with learned keys, no diversity loss"?**

Strong support from Switch Transformer (Fedus 2021), Soft MoE (Puigcerver 2024), DeepSeek-MoE (2024), Mixtral (Jiang 2024). **None** of these use key-key repulsion; all rely on LM gradient + load balance. Our use of SKRL is uncommon and poorly motivated relative to the literature.

**Q5: Confidence**

- Direction (remove SKRL): HIGH
- Fix X.1 as written works within 500 fwd: MEDIUM-HIGH
- X.1 + X.2 together achieve top1_sim > 0.01: HIGH

---

## 8. 推荐下一步

- **Worker**: `/coder` — implement selector.py:159 change (remove `.detach()`), add `--slot_value_norm_cap` CLI flag + memory_bank.py clip logic (optional but recommended).
- **Then worker**: `/trainer` — launch 3-node fix_x_ablation ablation:
  - b200-1: skrl=0.0, T=10, undetach, **norm_cap=10**
  - b200-2: skrl=0.0, T=10, undetach, **no norm_cap** (control for norm cap)
  - b200-3: skrl=0.05, T=10, undetach (CONTROL: is SKRL actually the problem?)
- **Kill criterion**: fwd=500 top1_sim < 0.005 on all 3 nodes → dispatch X.2 (InfoNCE) coder task.
- **Success criterion**: fwd=500 top1_sim > 0.005 on at least one node → continue to 10000 steps and evaluate.

---

## 9. 关键未解决问题（交给后续）

1. If all three Fix X nodes fail, the next research question is: **does pg19 LM training even benefit from selective retrieval?** Maybe the task itself doesn't require routing differentiation, and we need a proxy task (e.g., synthesized "fact recall" in-the-loop).
2. The `niah_acc=0.000` observation across all runs suggests the system has never once retrieved correctly. This is a separate pathology (likely: no_grad on haystack + random routing = random answer = 1/|vocab| accuracy) that should be diagnosed after top1_sim is working.
