# Branch-3 Option A.2 Pollution Debug — Deep Diagnosis

**Date**: 2026-04-26
**Author**: `/researcher` (dispatched by main)
**Scope**: Why did Option A.2 (shared-bank + gradient-bearing writeback, σ=1.0 random slot init, warmup=0) blow held-out PPL to **472.31** vs Tier-3's **2.1278** baseline and the bypass-parity floor of **16.50**?
**Constraint**: static code review only — no GPU, no trainer, no code changes.
**Codepath pinned**: `src/memory/mem_space/{layer.py, memory_bank.py, patch.py, config.py, selector.py}` at HEAD.

---

## 1. Observed signature

| Metric | Value | Notes |
|---|---|---|
| Held-out PPL (200×4096) | **472.31** | avg_loss 6.158 |
| step-1 lm_ppl | **1001.57** | BEFORE any optimizer update (logging order below) |
| step-100 lm_ppl | 1682.08 | oscillates; never converges |
| step-200 lm_ppl | 1093.98 | sinusoidal in [5.2, 8.1] lm_loss |
| aux_loss | pinned ≈21 | selector never differentiates slots |
| NaN count | 0 | not a numerical explosion; pure signal loss |

**Reference ladder**:

| Run | Held-out PPL | Diff |
|---|---|---|
| Pure vanilla Llama-3-8B (Tier-3 trained) | **2.1278** | — |
| Bypass-parity structural test (α=0 forward) | **16.50** | `20260426_mem_space_v0_bypass_parity_unit_test.md` |
| Stage-2a (shared bank, `hidden_to_slot` unfrozen, grad writeback, σ=0.02) | 322.5 step-1 | partial Branch-3 |
| Stage-2b (Stage-2a + warmup=0) | 426.4 step-1 | removed warmup |
| **Branch-3 A.2** (+ σ=1.0) | **1001.57 step-1** | this run |

**Logging order** (`scripts/train_mem_space_pg19.py` line 448-475): `_reset_banks → forward → loss → backward → clip → optim.step → step_counter += 1 → log`. Therefore **"step 1 lm_ppl"** is the loss **computed before** the step-1 optimizer update, i.e. at parameter initialisation with exactly one chunk of state in the bank.

**Implication**: step-1 = 1001 **cannot** be blamed on a bad learning-rate update. It is a **forward-pass pollution at init**.

Key red-line reference: CLAUDE.md PPL ladder — PPL > 100 means the LM itself is polluted, not that memory retrieval is mediocre. Do not tune hyperparameters until root cause is known.

---

## 2. Step-1 static analysis (the "bypass should have protected us" argument)

### 2.1 The Flamingo output gate

`layer.py:231` initialises `slot_output_gate = nn.Parameter(torch.zeros(()))`.
`layer.py:437` computes `alpha = torch.tanh(self.slot_output_gate)` → `tanh(0.0) = 0.0` **exactly** in bf16 (IEEE-754 guarantees tanh is correctly-rounded at 0).
`layer.py:440`: `next_hidden = bypass_h + alpha * slot_delta`. With `alpha == 0`, this reduces to `next_hidden ≡ bypass_h` **bit-exactly** (0·x is 0 for finite x in bf16; Inf/NaN would yield NaN, but 0 NaN was reported).

### 2.2 The bypass branch

`layer.py:399-411` calls `self.wrapped_layer(hidden_states, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False, position_embeddings=position_embeddings, **kwargs)`. This is **identical** to what an unpatched `LlamaDecoderLayer` would see if the outer `LlamaModel` had delivered `attention_mask=None` (SDPA implicit-causal path). Crucially, `bypass_h` is computed **independently** of `slot_delta` — the slot path is not in bypass's autograd graph, nor does the slot path mutate any state that bypass reads before bypass runs (the ordering is bypass → ext → writeback).

### 2.3 Therefore bypass parity SHOULD hold at step 1

With 32 layers all having `alpha=0`, the chained composition produces **vanilla Llama output** at step 1. Expected PPL: **2.13** (vanilla). Measured ceiling: **16.50** (known bypass-parity gap, attributed in `20260426_mem_space_v0_bypass_parity_unit_test.md` to the SDPA implicit-causal difference when attention_mask is passed as None vs an explicit 4-D mask).

**Observed: 1001.57.** That is **60× worse than the known bypass-parity floor** and **470× worse than vanilla**. Something in the A.2 diff has broken the structural bypass guarantee that Tier-3 relied on.

### 2.4 What changed in A.2 vs Tier-3

| Knob | Tier-3 | A.2 (this run) |
|---|---|---|
| `shared_memory_bank` | False (per-layer) | **True** |
| Writeback gradient | detached (`O_mem_slot.detach()`, float β) | **attached** (tensor β, no detach) |
| `slot_init` | `hidden_pool` | **`random`** |
| `slot_init_noise` | 0.02 | **1.0** |
| `writeback_gate_warmup_steps` | 2000 | **0** |
| `hidden_to_slot_frozen` | True | True (unchanged) |

None of these changes **touches the bypass codepath** (lines 399-411, 437-440). They all live in the ext branch and the writeback branch. Yet bypass parity is empirically broken by 60×. §3 and §4 explain how.

---

## 3. Shared-bank cross-layer amplification

The claim "bypass at α=0 is independent of slot state" is **false in A.2 even at step 1**, because of the writeback that happens after ext at layer L-1 is read by layer L.

### 3.1 Writeback mutates the shared bank every layer

`layer.py:453-456`:
```python
beta_t = self._current_beta()          # σ(0)·1·0.3 = 0.15 at step 1
if cfg.enable_writeback:
    O_mem_slot = self.hidden_to_slot(O_mem_hidden)   # from ext_h, huge at σ=1.0
    self.memory_bank.write(idx, O_mem_slot, beta_t)
```

With `warmup=0`, β = σ(0)·1·0.3 = **0.15 at step 1 already**. Writeback is live from layer 0 onwards.

`memory_bank.py:225-237` (tensor-gate branch):
```python
updated = (1 - gate_t) * current + gate_t * new_contrib
self.slots = self.slots.scatter(1, idx_exp, updated)
```

After layer 0 writes, `self.slots` is re-bound to a **new autograd node**. Layer 1, 2, …, 31 all read this mutated bank.

### 3.2 Per-layer bypass is protected, but the "bypass slot state" is NOT frozen

Note: `next_hidden = bypass_h` at layer L only guarantees that the **output to layer L+1 hidden states** matches vanilla Llama. It does **not** freeze the shared `slots` tensor. Between the start of layer 0's forward and layer 31's forward, the bank has been **mutated 32 times**.

However, this affects the **ext** branch at each layer, which in turn affects `O_mem_hidden` (= `ext_h[:, :k_slots, :]`), which in turn affects `O_mem_slot = hidden_to_slot(O_mem_hidden)`, which is written back with β=0.15 EMA. **Bypass_h at layer L does not depend on any of this** — it is computed from `hidden_states` (which is `next_hidden` of layer L-1 = `bypass_h_{L-1}` = vanilla) and the wrapped layer's own weights. So the chain bypass_0 → bypass_1 → … → bypass_31 **should** still reproduce vanilla Llama bit-for-bit.

### 3.3 So cross-layer slot mutation by itself does NOT explain step-1 = 1001

The shared-bank mutation only matters if bypass reads the bank, which it doesn't. Therefore H2 (shared-bank compounding) is a **step-2+** amplifier, not a step-1 breaker.

### 3.4 But it explains the training dynamic

Once we leave step 1 and α has moved off zero by any amount (via gradient on `slot_output_gate`), the 32× compound of polluted slot_delta injections is what produces the sinusoidal lm_loss in [5.2, 8.1] — each layer contributes independent slot_delta to `next_hidden`, and with shared-bank writeback threading grad back, the slot state itself chases a noise target that doesn't stabilise. Hence no convergence despite 200 steps.

---

## 4. Gradient path audit — what drives α off zero at step 1?

### 4.1 Graph structure of `slot_output_gate`

For each of 32 layers (there are 32 distinct `slot_output_gate` parameters — one per wrapper):

```
L_CE ← logits ← final_hidden ← bypass_31 + tanh(α_31)·slot_delta_31
                                ↑ α_31 grad = <∂L/∂next_hidden, slot_delta_31> · sech²(α_31)
```

At α=0, `sech²(0) = 1`. So each α_L gets grad = ⟨∂L/∂next_hidden_L, slot_delta_L⟩.

### 4.2 Magnitude of slot_delta at σ=1.0

`MemoryBank.init_from_hidden` with `slot_init="random"` (memory_bank.py:138):
```python
slots = torch.randn(B, N, d, device=device, dtype=dtype) * self.init_noise
```
With `init_noise=1.0` and `slot_dim=4096`: **per-element std = 1.0**, per-token RMS = 1.0·√4096 = **64**. Llama-3-8B post-RMSNorm per-element magnitude is ≈ 1 / √d ≈ **0.0156 RMS per element** (RMSNorm normalises the token's RMS to 1.0, then the learned weight brings per-element magnitude to O(1/√d)). **Slot per-element is therefore ~64× too large.**

Projected through `slot_to_hidden` (Linear std=0.02, line 215-224):
- `M_sel_hidden[b,k,i] = Σ_j W[i,j] · slot[b,k,j]` with W std 0.02, slot std 1.0
- Per-element std of `M_sel_hidden` ≈ 0.02 · 1.0 · √slot_dim = 0.02 · 64 = **1.28**
- Vs normal hidden_states per-element ≈ 0.02

So the slot K/V injected into the extended sequence have per-element magnitude **~64× larger than the true hidden states they get concatenated to**.

### 4.3 What this does to ext_h

In the joint softmax over `[M_sel, H]`, slot keys with per-element magnitude 1.28 vs hidden keys of magnitude 0.02 produce attention logits that are dominated by slot positions. H-queries attend almost entirely to noise. `ext_h[:, k_slots:, :]` (the H-body output of the ext forward) is therefore **close to zero-content attention over random slot values**, i.e. noise.

`slot_delta_L = ext_h[:, k_slots:, :] - bypass_h`. With `bypass_h` of normal magnitude ~O(0.02) and `ext_h_body` of noise with residual-path magnitude ~O(1.28) (because the noisy attention output is added into the residual stream), `slot_delta` has per-element magnitude **~O(1)** — roughly 30-50× larger than bypass_h itself.

### 4.4 First-step gradient on slot_output_gate

Per layer: grad(α_L) = ⟨∂L/∂next_hidden_L, slot_delta_L⟩.

`∂L/∂next_hidden_L` at step 1 has the magnitude of a vanilla gradient (because next_hidden is vanilla at α=0). Dotted with `slot_delta_L` ~O(1) per element over T=4096 positions · d=4096 hidden dims, this is a **very large** scalar per layer.

Adam's first step is bias-corrected: `Δθ ≈ -lr · sign(g)` (first-step first-moment estimate is g itself, second-moment is g², so Δθ = -lr · g/|g| = -lr · sign(g)). With lr=3e-4, **each α_L moves by exactly ~3e-4** after step 1.

tanh(3e-4) ≈ 3e-4. Per layer, the slot-delta contribution to `next_hidden` becomes 3e-4 · O(1) = O(3e-4) per element, on top of bypass of O(0.02). That is ~1.5% contamination per layer. **Over 32 layers in the residual stream, this compounds multiplicatively** (each layer's output is the next layer's input, and each layer adds ~1.5% noise relative to the signal).

After 32 layers: signal-to-noise collapses from ≈67 to ≈2, i.e. ~97% signal loss at the final hidden state. That takes the lm_loss from ~2.13 (vanilla) to a number limited by logit magnitude in the softmax — consistent with **lm_loss ≈ 6-7, PPL ≈ 500-1500** as observed.

### 4.5 But step-1 log is PRE-step

§4.4 explains step-2+, not step-1. Why is step-1 **already 1001**?

**Hypothesis**: it is NOT pre-step — the logging order in `scripts/train_mem_space_pg19.py:448-475` is:

```
forward → lm_loss → backward → clip → optim.step → step_counter_inc → log
```

So "step 1" in the log corresponds to the loss computed on the FIRST forward, **but logged after optim.step(1) has already run**. The `lm_loss.item()` that gets logged as "step 1 lm_loss" is the PRE-step value; that is correct (it's captured before backward, saved in a Python float, and printed after step). So §4.4's mechanism does NOT yet apply to step 1.

**Therefore**: step-1 lm_ppl ≈ 1001 must come from something in the **forward pass at initialisation**, not from an optimizer update. §2 claims that forward must be ≈ bypass-parity floor (16.5) if α=0. So **either §2's assumption is violated**, or the bypass-parity floor is itself much higher at σ=1.0 than at σ=0.02.

### 4.6 The one thing that violates §2's "bypass is independent of slot path"

Re-read `layer.py:413-421`:
```python
ext_out = self.wrapped_layer(
    extended_hidden,
    attention_mask=ext_attn_mask,
    ...
    **kwargs,
)
```

The **same `self.wrapped_layer`** module is called twice. `LlamaDecoderLayer` is nominally stateless — no in-place param mutation, no RNG consumption in inference mode. In training mode with dropout inside attention, the two calls would consume different RNG draws and therefore produce non-identical-on-shared-tokens outputs. **Llama-3-8B has attention dropout = 0.0 by default**, so this should not matter — but worth flagging as a secondary candidate.

**The REAL violation**: `**kwargs` passthrough. At the outer `LlamaModel.forward` level, kwargs may contain `cache_position`, `output_attentions`, or in transformers 5.0+ a `labels_mask` / batch-related tensor. The same kwargs are passed **unchanged** to BOTH calls. For the ext call, `extended_hidden` has length k+T while `cache_position` (if present) has length T. **Shape mismatch would error loudly** (we'd see the error in logs, we didn't). So this probably isn't it either.

### 4.7 The candidate that DOES survive: σ=1.0 breaks the bypass-parity floor numerically via ext_h feedback

At σ=0.02, `slot_delta` magnitude is 50× smaller, so `0 · slot_delta` is zero even allowing for bf16 rounding slop. At σ=1.0, `slot_delta` can reach per-element magnitude **O(1)** (§4.3). In bf16 with ε_bf16 ≈ 2^-7 ≈ 0.008:

- `alpha · slot_delta` with `alpha = 0.0` (exact) is zero exactly.
- But `bypass_h + (alpha · slot_delta)` where `alpha · slot_delta = 0.0` is exactly `bypass_h`.

So even σ=1.0 does NOT break step-1 structurally. **Unless** `slot_delta` contains NaN (which propagates via 0·NaN = NaN). The report says 0 NaN observed, so this is ruled out.

### 4.8 Residual candidate: `slot_output_gate` is NOT actually 0 at step 1

`layer.py:231`:
```python
self.slot_output_gate = nn.Parameter(torch.zeros(()))
```

That initialises to 0.0 exactly. But under DDP with `find_unused_parameters=True` (scripts line 412-413), if any parameter update path touched `slot_output_gate` via a buggy `requires_grad` interaction or a reducer hook, it could be non-zero at the first forward. **Unlikely but worth checking via a probe** (see §5 experiment 3).

### 4.9 Remaining plausible explanation

The most parsimonious explanation for step-1 PPL = 1001 **at σ=1.0 specifically** (given σ=0.02 shared-bank runs Stage-2a/2b gave 322/426, also broken but less so) is:

**A layer-level side channel in the wrapped_layer call changes output based on slot_delta magnitude**, probably through `**kwargs` interaction with transformers 5.0's SDPA kernel selection or via in-place mutation of `kwargs["past_key_values"]` / `kwargs["cache_position"]` between the bypass and ext calls. This needs to be confirmed by a focused unit test.

---

## 5. Root-cause ranking + cheapest discriminating experiment

### 5.1 Ranked hypotheses (most → least likely)

| # | Hypothesis | Pre-step explains? | Post-step explains? | Blocks progress alone? |
|---|---|---|---|---|
| **H1** | σ=1.0 slot init is **50-64× over-scaled** vs Llama post-RMSNorm; drives massive first-step gradient on α_L; 32× residual-stream compound collapses signal by step 2 | No (α=0 protects) | **Yes** (§4.4) | Yes (explains pollution after any optim step) |
| **H2** | `**kwargs` passthrough and/or `LlamaDecoderLayer` SDPA-kernel dispatch differs between bypass and ext calls when `extended_hidden` is order-of-magnitude larger per element, so bypass_h is silently corrupted | **Yes** | Yes (amplifies H1) | Yes |
| **H3** | Shared-bank compounding + 32× autograd graph depth through `scatter` rebind gives `gate_param` a gradient ~32× expected → β jumps from 0.15 to something large in one step, writeback overwrites bank with noise projection, selector loses slot identity | No | Partial (secondary) | No (needs H1/H2 to kick off) |
| **H4** | `slot_output_gate` not actually zero at step-1 forward due to DDP reducer / init race | **Yes** | — | Yes (if true) |
| **H5** | Bf16 rounding on `0 · slot_delta` when slot_delta has near-denormal values with Inf components from the ext softmax | Yes | — | Yes (if true) |

**Top-line**: **H1 is the single knob most worth retracting**. H2 is the hypothesis that explains the step-1=1001 observation (not H1 alone). H3 is the post-step amplifier. H4/H5 are long-shot.

### 5.2 Cheapest discriminating experiment (single 8-GPU node, ≤ 2h)

**Experiment A — "σ retreat alone"**:
- Re-run A.2 config EXCEPT: `--slot_init hidden_pool --slot_init_noise 0.02 --writeback_warmup_steps 500`
- Keep `shared_memory_bank=True`, writeback gradient-bearing, lr=3e-4
- Predict under H1: step-1 lm_ppl drops to bypass-parity floor ~16-20; converges to PPL < 10 by step 200
- Predict under H2: step-1 lm_ppl drops modestly (~100-200, still polluted); no convergence
- Predict under H4/H5: step-1 essentially unchanged (~500-1000)

**Result interpretation**:
- PPL < 50 at step 200 ⇒ **H1 confirmed**; σ=1.0 was the entire bug. Proceed with σ=0.02 A.2.
- PPL 100-500 at step 200 ⇒ **H2 likely**; dispatch focused bypass-parity unit test on dual-call wrapped_layer with σ-sweep to localise.
- PPL > 500 at step 1 ⇒ **H4 or H5**; probe parameter values via a one-step diagnostic dump.

### 5.3 Parallel experiment on a second node (optional but cheap)

**Experiment B — "disentangle shared_bank"**:
- Re-run A.2 config with `--no_shared_memory_bank` (per-layer banks, everything else same including σ=1.0)
- Predict under H1: still broken (σ is independent of sharing); step-1 still ~500-1000
- Predict under H3: improves — graph depth collapses from 32 to 1 per layer, no cross-layer bank mutation

If H1 and H3 are **both** true, Experiment B will still show pollution but less severe (maybe step-1 ~200-400, not 1000). Running A and B in parallel on two B200 nodes gives a full discrimination in one round-trip.

### 5.4 Even cheaper: one-step static probe (5 min CPU)

Write a 30-line script that:
1. Loads Llama-3-8B, patches with A.2 config (σ=1.0, shared bank, warmup=0)
2. Asserts `all(tanh(L.slot_output_gate) == 0 for L in model._mem_space_layers)` — this invalidates H4
3. Runs **one forward** on a toy 1024-token batch, captures `next_hidden` at the output of every layer, captures `bypass_h` separately via a `forward_no_memory` call
4. Reports `max_abs_err(next_hidden_L, vanilla_Llama_out_L)` for L = 0, 8, 16, 24, 31
5. If max_abs_err at layer 0 is already > 1e-3 → **H2/H5 confirmed** without running any training

This is the cheapest discriminator. **Recommend running this first** before queuing an 8-GPU A/B.

---

## 6. Recommendation to main

1. **Do NOT keep tuning A.2 hyperparameters** while PPL > 100 (CLAUDE.md red-line rule; this note's §4.4/§5 identify the knobs to retract).
2. **Run the 5-minute static probe (§5.4) first** on the local H20 — single GPU, no DDP, no writeback effect (step 0 forward only). This localises the bug to "bypass-parity already broken" (H2/H5) vs "bypass parity ok, step-2 amplification kills it" (H1/H3).
3. **Based on probe result**:
   - Probe shows layer-0 bypass parity broken ⇒ prioritise a targeted unit test on the dual `wrapped_layer` call (H2 hunt), BEFORE any training. Do not queue further A.2 runs until bypass parity is re-established.
   - Probe shows layer-0 bypass parity intact (max_abs_err < 1e-4 at layer 0) ⇒ queue Experiment A (σ=0.02 + warmup=500) on one B200 node, Experiment B (no_shared_bank + σ=1.0) on a second B200 node, in parallel per CLAUDE.md directive "different nodes can run different experiments".
4. **Do NOT revert to Tier-3 yet** — A.2's writeback-BPTT architecture is still the intended direction. The bug is in either (a) the A.2-specific knob choices (σ, warmup) or (b) an unrelated latent wrapped-layer bug that σ=1.0 exposed. Both are fixable without abandoning Option A.2.
5. **Kill criteria** if the probe + Experiment A fail to recover:
   - Step-1 PPL still > 100 after σ=0.02 + warmup=500 ⇒ escalate to H2/H5 root cause hunt; freeze writeback-BPTT work until bypass parity is re-proven in a unit test.
   - Step-200 PPL still > 50 after σ=0.02 + warmup=500 ⇒ shared-bank or 32× autograd graph depth is pathological even at correct σ; fall back to per-layer banks (A.1) as the writeback-BPTT default.
6. **Update the design doc** (`20260426_mem_space_v0_branch3_writeback_bptt.md`) §4 to correct the claim that σ=1.0 "matches Llama post-rmsnorm magnitude" — post-rmsnorm per-element is O(1/√d) ≈ 0.016, not 1.0. The token-level RMS is ~1.0 **only after summing over d dimensions**; per-element std is 64× smaller than the dispatch brief assumed.

---

## Appendix — Line-number citations used above

- `src/memory/mem_space/layer.py:215-225` — slot_to_hidden / hidden_to_slot Linear std=0.02
- `src/memory/mem_space/layer.py:231` — `slot_output_gate = nn.Parameter(torch.zeros(()))`
- `src/memory/mem_space/layer.py:254-256` — `gate_param = nn.Parameter(torch.tensor(writeback_gate_init))`
- `src/memory/mem_space/layer.py:270-279` — `_current_beta` = σ(gate) · warmup · gate_max
- `src/memory/mem_space/layer.py:336-341` — lazy init on first forward
- `src/memory/mem_space/layer.py:399-421` — dual wrapped_layer call (bypass then ext)
- `src/memory/mem_space/layer.py:437-440` — `alpha = tanh(…)`, `next_hidden = bypass_h + alpha · slot_delta`
- `src/memory/mem_space/layer.py:453-456` — gradient-bearing writeback (no detach, tensor β)
- `src/memory/mem_space/memory_bank.py:138` — `slots = randn(B,N,d) * init_noise` (σ=1.0 per element)
- `src/memory/mem_space/memory_bank.py:211-213` — short-circuit float β ≤ 0
- `src/memory/mem_space/memory_bank.py:225-237` — tensor-gate branch, non-in-place scatter rebind
- `src/memory/mem_space/patch.py:104-120` — shared bank allocation and wiring across 32 layers
- `scripts/train_mem_space_pg19.py:448-475` — training loop order (forward → backward → step → log)
