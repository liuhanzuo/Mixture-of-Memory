# Memory-Space v0 — Tier-3 analysis of the fix3 smoke failure

**Date**: 2026-04-26
**Author**: /researcher Tier-3 (subagent)
**Related**:
- `ops/research_notes/20260426_mem_space_v0_tier2_residual_gap.md` (Tier-2 hypothesis → FALSIFIED)
- `ops/research_notes/20260426_mem_space_v0_jointattn_diagnosis.md` (Tier-1 fix1+fix2)
- `src/memory/mem_space/layer.py` (fix3 applied at lines 191-204)

## 1. Problem restate

Fix3 (Tier-2 §3 #1) replaced the `Identity` shortcut with a zero-init
`nn.Linear(d_model, d_model, bias=False)` for `slot_to_hidden` so that at
step 0 the slot K/V path is `W·0 = 0` and the slot contribution to the
residual stream "starts at exactly bypass parity". Prediction: step-0 PPL
∈ [16, 18] (bypass parity), final PPL ≤ 25.

Observed (two bit-identical smoke runs):

| metric | contract | run 1 | run 2 |
|---|---|---|---|
| step-1 lm_ppl | [16, 18] | **62.64** | 62.64 |
| final PPL | ≤ 25 | 2311.26 | 25346.31 |
| n_trainable (M) | ~540 | **1107** | 1107 |
| nan_chunks | 0 | 0 | 0 |

The predicted bypass parity at step 0 **did not hold**. LM loss explodes
from step 4 onward. The hypothesis in Tier-2 §3 #1 is falsified.

## 2. Why step-0 PPL is 62.64, not 16.50 (Q1)

**The RMSNorm-of-zero hypothesis in the brief is false.**  I verified
empirically:

```python
>>> rms = LlamaRMSNorm(4096, eps=1e-6)           # init weight=ones
>>> rms(torch.zeros(1, 4, 4096, dtype=torch.bfloat16)).abs().max()
tensor(0., dtype=torch.float32)
```

because `x * rsqrt(var + eps) = 0 * (1/√eps) = 0` (variance of zeros is
exactly zero, so the output is `weight * 0 = 0` regardless of eps). Llama
has no attention bias (`attention_bias=False`), so
`K_slot = W_k · RMSNorm(0) = W_k · 0 = 0` and likewise for Q, V. Apply-
RoPE on a zero tensor also yields zero. Thus **slot K/V at step 0 are
numerically zero**, as fix3 expected.

**The real mechanism is softmax-denominator pollution by phantom-zero
logits ("attention-sink leak").**  For an H-query at position `t` in the
second half of the chunk (fix2 leaves these slot-visible), the attention
distribution over keys is

    softmax([Q_H · K_slot_1, ..., Q_H · K_slot_k, Q_H · K_H_1, ..., Q_H · K_H_Tcausal])
    = softmax([0, 0, ..., 0, l_{t,1}, ..., l_{t,Tcausal}])

with `Tcausal = t+1`. Because `K_slot = 0`, every slot contributes
`exp(0) = 1` to the softmax *denominator* without adding to the
numerator (since `V_slot = 0` too). The attention output for the H-query
is therefore

    out_H(t) = (Σ_{j≤t} exp(l_{t,j}) · V_H_j) / (k + Σ_{j≤t} exp(l_{t,j}))

vs. the bypass value

    out_bypass(t) = (Σ_{j≤t} exp(l_{t,j}) · V_H_j) / (Σ_{j≤t} exp(l_{t,j}))

i.e. the attention output is **attenuated** by

    α(t) = S_H(t) / (k + S_H(t)),    S_H(t) = Σ_{j≤t} exp(l_{t,j})

Llama attention is sharp — the logit distribution has a few large peaks,
so `S_H(t)` is typically small-O(1)..O(T/head). With `k=64`:

| S_H(t) (effective attention mass on H) | α (per-layer attenuation) |
|---:|---:|
| 0.5 | 0.008 |
| 2 | 0.030 |
| 10 | 0.135 |
| 100 | 0.61 |
| 1000 | 0.94 |

The decoder has **32 such layers** stacked. Even a mild per-layer
α=0.95 compounds to 0.95^32 = 0.19 → the attention stream's signal is
attenuated by ~80 %. For early H-queries (small t), α ≈ 0.01 per layer →
effectively **zeroed-out attention output for 32 layers** → residual
stream degenerates to the embedding+MLP-only path. Hence the LM head
predicts near-uniformly for many positions.

This compounding also explains why both smoke runs are **bit-identical at
step 1**: with slot_to_hidden = 0 the LM head's output depends only on
the frozen backbone and the fixed randomness of the slot init noise; the
selector / hidden_to_slot weights cannot alter out_H(t) at step 0 because
they only enter the writeback path (which Tier-2 §2 showed is inert).

**The 4× PPL (62.6 vs 16.5)** is consistent with the first half of the
chunk being bypass-exact (fix2's `T/2` cutoff masks slots → no phantom
logits → bypass parity) and the second half being severely attenuated:

    avg_loss ≈ 0.5 · log(16.5) + 0.5 · L2  = 4.14  →  L2 ≈ 5.48 → PPL2 ≈ 238

i.e. the second half alone has an effective PPL of ~238. Layer-wise
compounded attenuation easily accounts for this.

### Summary of mechanism enumeration from the brief

| candidate mechanism | verdict |
|---|---|
| softmax mass redistribution over `[k,T]` | **this is the bug** — phantom-logits in denominator |
| fix2 `T/2` asymmetry alone | contributes (makes 1st half bypass-exact, 2nd half catastrophic) — not root cause |
| RoPE rotating zero K_slot | inert (cos·0 + rot(0)·sin = 0) |
| RMSNorm of zero slot hidden | **FALSE** — RMSNorm(0) = 0 exactly (verified numerically) |

## 3. Fix2 mask interaction (Q2)

### Attention pattern at step 0, with fix2 + fix3:

```
              Keys (length k + T = 64 + 4096 = 4160)
              <--k=64 slot-K (val=0)--> <--T=4096 H-K (nonzero)-->
Q slot (64)   [ 0 everywhere ]           [ 0 — no causal on slot-Q]
Q H 0         [ -inf × 64  ]             [ causal(t=0) ]
Q H 1         [ -inf × 64  ]             [ causal(t=1) ]
...                                          (fix2: slots masked)
Q H T/2-1     [ -inf × 64  ]             [ causal(t=2047) ]
Q H T/2       [ 0  × 64    ]             [ causal(t=2048) ]   ← slots visible
Q H T/2+1     [ 0  × 64    ]             [ causal(t=2049) ]
...
Q H T-1       [ 0  × 64    ]             [ causal(t=4095) ]
```

- **First half (rows k..k+T/2-1)**: slot keys are `-inf` → contribute 0
  to softmax → attention pattern is EXACTLY bypass → bypass output.
- **Second half (rows k+T/2..k+T-1)**: slot keys are allowed at logit 0 →
  attention mass stolen by 64 phantom `exp(0)=1` terms → output
  attenuated → compounded 32× → garbage second half.

### If we drop fix2 (set T_half = 0)

Then slots are visible to ALL H-queries → phantom logits pollute all
positions (not just second half). Bypass parity is **not** restored
because the phantom-logit pathology is independent of fix2: it comes
from prepending k zero K-vectors in an unmasked slot block.

Dropping fix2 would give uniformly-attenuated PPL across the chunk
(monotonic in `k`). Predicted PPL: worse than 62.6 — probably in the
several-hundred range for early H-queries where `S_H` is very small.

**Conclusion**: fix2 is neither the villain nor a fix. The villain is
the phantom-logit pollution in the softmax denominator. Removing fix2
**does not** restore bypass parity; it spreads the damage.

## 4. Param-budget audit (Q3)

Confirmed by reading `_mem_space_params` (`scripts/train_mem_space_pg19.py:134-161`):

    wrapper.selector.parameters() : Q_sel 4096·128 + K_sel 4096·128 = 1.05M / layer
    wrapper.gate_param            : 1 scalar / layer
    wrapper.slot_to_hidden        : 4096·4096 = 16.78M / layer
    wrapper.hidden_to_slot        : 4096·4096 = 16.78M / layer
    ---------------------------------------------------------------------
    per layer                     : 34.61M
    × 32 layers                   : 1107.5M  ← matches observed 1107.30M exactly

So Tier-2's "~540M" estimate counted only *one* direction. With fix3 both
`slot_to_hidden` AND `hidden_to_slot` became `nn.Linear` and both are
harvested by `_mem_space_params`.

**`hidden_to_slot` receives no useful gradient**:
1. `O_mem_slot = self.hidden_to_slot(O_mem_hidden)` (layer.py:370)
2. `self.memory_bank.write(idx, O_mem_slot.detach(), beta_val)` — DETACHED.
3. `_reset_banks` discards the bank every chunk (Tier-2 §2).

So `hidden_to_slot.weight` participates in **no** operation whose output
influences the loss. Its gradient is identically zero (or, with
`find_unused_parameters=True`, it is simply not reduced). DDP is tolerant
but the 536.87M parameters waste optimizer memory (≈2.1 GB in bf16 Adam
moments per copy) and bloat checkpoint size.

**Recommendation**: freeze `hidden_to_slot` at init. If/when writeback is
re-activated in Stage-2, re-enable it via config.

## 5. Proposed Tier-3 fix — SINGLE code change

### Root cause recap

Zero-init `slot_to_hidden` + concat-into-softmax → phantom logits of 0 →
k unwanted "attention-sink" terms of `exp(0)=1` in the softmax
denominator → output attenuation per layer → catastrophic compounding
over 32 layers.

The Flamingo discipline ("output of the newly added module starts at
zero") cannot be achieved by input-side zeroing when the added module
fuses into a softmax with the existing path. **Flamingo itself does
this via an OUTPUT-SIDE `tanh(α)` gate**, not an input-side zero.

### Fix: replace the concat-joint-attention with a bypass-plus-gated-delta

`src/memory/mem_space/layer.py:338-364` — the single block between
"Run the wrapped decoder layer" and "Split into memory-head and body":

```python
# === OLD (single concat forward, afflicted by phantom-logit leak) ===
wrapped_out = self.wrapped_layer(
    extended_hidden,
    attention_mask=ext_attn_mask,
    position_ids=None,
    past_key_values=None,
    use_cache=False,
    position_embeddings=ext_pos_emb,
    **kwargs,
)
if isinstance(wrapped_out, tuple):
    ext_out = wrapped_out[0]; extra = wrapped_out[1:]
else:
    ext_out = wrapped_out; extra = ()
O_mem_hidden = ext_out[:, :k_slots, :]
next_hidden  = ext_out[:, k_slots:, :]

# === NEW (Flamingo-style: bypass + gated(extended - bypass)) ===
# 5a. Pure-bypass forward on H alone (exactly reproduces a vanilla Llama
#     decoder step; guaranteed bypass parity regardless of slot state).
bypass_out = self.wrapped_layer(
    hidden_states,
    attention_mask=None,           # let HF install its own causal mask
    position_ids=None,
    past_key_values=None,
    use_cache=False,
    position_embeddings=position_embeddings,
    **kwargs,
)
if isinstance(bypass_out, tuple):
    bypass_h = bypass_out[0]
else:
    bypass_h = bypass_out

# 5b. Extended forward with slots (phantom-logit leak is OK here — its
#     effect is entirely absorbed by the zero-init gate on the delta).
ext_out = self.wrapped_layer(
    extended_hidden,
    attention_mask=ext_attn_mask,
    position_ids=None,
    past_key_values=None,
    use_cache=False,
    position_embeddings=ext_pos_emb,
    **kwargs,
)
if isinstance(ext_out, tuple):
    ext_h = ext_out[0]; extra = ext_out[1:]
else:
    ext_h = ext_out; extra = ()

# 5c. Flamingo-style gated addition, alpha init 0 → next_hidden ≡ bypass_h.
alpha = torch.tanh(self.slot_output_gate)          # scalar in (-1, 1)
O_mem_hidden = ext_h[:, :k_slots, :]
slot_delta   = ext_h[:, k_slots:, :] - bypass_h    # [B, T, d]
next_hidden  = bypass_h + alpha * slot_delta       # at init: alpha=0 → bypass
```

**Supporting edits (still one conceptual change)**:

1. `__init__` — add the output gate and freeze `hidden_to_slot`:
   ```python
   # Flamingo-style output gate: tanh(alpha) on slot_delta; alpha=0 → bypass.
   self.slot_output_gate = nn.Parameter(torch.zeros(()))
   # Stage-1 writeback is inert → no gradient signal for hidden_to_slot.
   for p in self.hidden_to_slot.parameters():
       p.requires_grad = False
   # Revert Tier-2's zero-init on slot_to_hidden: we no longer need input-
   # side zeroing; the tanh(alpha) output gate is what guarantees bypass.
   nn.init.normal_(self.slot_to_hidden.weight, std=0.02)
   ```

2. `_build_extended_attn_mask` — drop fix2 (`T_half = T // 2`). With the
   output gate handling bypass parity, the T/2 cutoff is no longer needed
   and it had asymmetric effects (§3 above).

3. `_mem_space_params` (train script) — harvest `slot_output_gate`; drop
   `hidden_to_slot` params (they're now frozen, so `requires_grad=False`
   filter handles it automatically if we use `p.requires_grad` guard).

### Why this fix is correct at every step

- **Step 0**: `slot_output_gate = 0` → `tanh(0) = 0` → `next_hidden = bypass_h`
  identically, bit-for-bit the vanilla Llama forward. PPL = 16.50 exact.
- **Gradient signal at step 0**:
  `d(loss) / d(slot_output_gate) = sech^2(0) · <grad, slot_delta> = 1.0 · <grad, slot_delta>`
  which is **finite and nonzero** (because `slot_delta ≠ 0` — the extended
  forward has real content). So the gate is trainable from step 1 onward.
- **Step k>0**: gate drifts from 0 as training finds slot content useful.
- **Compute cost**: 2× decoder-layer forwards per training step. Smoke is
  10 steps × 4096 tokens × 32 layers; doubling is tolerable.

## 6. Smoke pass-contract

| metric | PASS | WARN | FAIL |
|---|---|---|---|
| step-0 lm_ppl (pre-training-step, fwd only) | [16.0, 17.0] | 17-20 | > 20 |
| step-1 lm_ppl (after 1 optimizer step) | [16.0, 18.0] | 18-25 | > 25 |
| final PPL (10 steps) | ≤ 20 | 20-30 | > 30 |
| nan_chunks | 0 | — | ≥ 1 |
| monotone-ish descent | lm_loss curve strictly bounded | small bumps | explosions |
| n_trainable (M) | [540, 580] (one-sided projection + gate + selector) | — | ≥ 1000 (hidden_to_slot not frozen) |
| effective lr | **3e-4** (DOWN from 1e-3) | 1e-3 | anything higher |

### Scripted verification

1. `torchrun --nproc_per_node=1 scripts/train_mem_space_pg19.py ... --max_train_steps 0` → step-0 PPL
2. `torchrun --nproc_per_node=1 scripts/train_mem_space_pg19.py ... --max_train_steps 10 --lr 3e-4` → step-1 and final PPL
3. Assert `n_trainable` in the log falls within [540, 580] M.

## 7. Fallback branch — if Tier-3 fails the smoke

### 7a. If step-0 PPL ≠ 16.50 exactly

The two-forward bypass isn't pure bypass → a side-effect is leaking
(e.g., `_reset_banks` is ALSO resetting some selector state, or the
selector's `pool = hidden_states.mean(dim=1)` triggers autograd even
when `slot_output_gate=0`). Diagnose with:
- `torch.testing.assert_close(bypass_h, vanilla_llama_layer(hidden_states, ...))` in a unit test
- log `wrapper.slot_output_gate.grad` at step 0 pre-optimizer-step

### 7b. If step-0 PPL = 16.50 but training explodes (like fix3)

The gate gradient is too large for lr=3e-4. Drop to lr=1e-4 and/or add
gradient clip to 0.1. Also check selector gradients — the selector may
be generating pathologically peaked scores once slot_delta is nonzero.

### 7c. If step-0 passes but gate never moves

The LM-loss gradient on slot_delta at init is truly zero (because the
backbone has learned to be robust to small noise). Warm up slot_delta's
magnitude via an explicit multiplier schedule for the first N steps, or
initialise `slot_output_gate` to a tiny positive value (e.g. 1e-3)
so `tanh ≈ 1e-3` instead of 0.

### 7d. If all the above fail

The Memory-Space v0 architecture is fundamentally incompatible with
post-training a frozen Llama-3-8B using a 10-chunk smoke (signal/noise
too low). Escalate to **Stage-1.5 re-design**:
- Reduce `num_slots` to 64, `top_k` to 8 (slash trainable params 8×).
- Disable `_reset_banks` — let slots persist across chunks within a
  document so writeback actually does something.
- Or fall back to a cross-attention side-channel (out of the decoder
  stack entirely): add one cross-attn adapter block at the top of the
  LM head that reads from slots, leaving all 32 decoder layers fully
  bypassed. This is a much smaller, much more trainable intervention.

## 8. One-paragraph verdict (for RESEARCHER_REPORTS.jsonl)

Tier-2's fix3 (zero-init `slot_to_hidden`) failed because "input-side zero
→ output-side zero" does NOT hold when the added module's K/V are
concatenated into the same softmax as the bypass path: the `k=64` slot
keys contribute `k` phantom `exp(0)=1` terms to the softmax denominator,
attenuating every H-query's attention output by ~`S_H/(k+S_H)` per
layer. Compounded over 32 decoder layers this attenuates the attention
signal by 60-90 %, explaining the step-1 PPL=62.64 (first half bypass-
exact via fix2; second half catastrophic). RMSNorm-of-zero is a red
herring — `RMSNorm(0) = 0` exactly. The correct Tier-3 fix is Flamingo-
style OUTPUT-side gating: run the wrapped layer twice (bypass-only and
extended), combine as `bypass + tanh(alpha) * (extended - bypass)` with
`alpha` init 0. This gives exact bypass parity at step 0, a nonzero
first-order gradient on `alpha`, and a clean path for training to lift
the slot contribution. Also freeze `hidden_to_slot` (inert under
`O_mem_slot.detach()` + `_reset_banks`) to reduce trainable params from
1107M to ≈540M, and drop `lr` from 1e-3 to 3e-4.
