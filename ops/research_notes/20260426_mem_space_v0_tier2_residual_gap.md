# Memory-Space v0 — Tier-2 analysis of the 63.40 residual gap

**Date**: 2026-04-26
**Author**: /researcher (subagent)
**Related**:
- `ops/research_notes/20260426_mem_space_v0_jointattn_diagnosis.md` (Tier-1 diagnosis, fix1+fix2)
- `ops/research_notes/20260426_memory_space_design_direction.md` (original spec)
- `src/memory/mem_space/layer.py` (§ projections, § writeback)
- `scripts/train_mem_space_pg19.py` (§ `_mem_space_params`, `_reset_banks`)

## 1. Problem restate

After fix1 (`slot_init=random, noise=1.0`) + fix2 (`mask[k:k+T/2,:k]=-inf`), the 8-GPU × 200-chunk
full run lands at **PPL = 63.40**, vs bypass-parity **16.50** and predicted healthy band **[15, 30]**.
The oracle-slot-leak pathology from the Tier-1 diagnosis is gone (confirmed by the 407 → 72/63
drop) but a secondary structural cost remains: joint attention still costs 3.8× over the vanilla
backbone and 2.1× over the upper edge of the healthy band.

## 2. Key observation before ranking (this drives the ranking)

Three facts collapse three of the five hypotheses:

1. **`slot_to_hidden` and `hidden_to_slot` are `nn.Identity()` in this regime** (`slot_dim=None → slot_dim = d_model = 4096`, `layer.py:193-195`). With Identity, **no trainable weight sits between `slots` and the decoder's K/V projections** — the selector can pick *which* slot to prepend, but cannot shape *what* the slot looks like as a K/V vector.
2. **`_reset_banks` is called at the top of every chunk in both eval and train** (`train_mem_space_pg19.py:269, 410`). Each chunk therefore re-initialises slots to `randn(B, N, d) * 1.0` before any forward. The EMA writeback mutates a per-chunk bank that is then discarded. → **writeback state has zero effect on any chunk's own PPL**, so the gate / β / warmup never influences measured loss.
3. **`gate_param` receives no gradient**. In `layer.py:364`, `beta_val = float(beta_t.detach().item())` turns β into a Python float *before* it enters `memory_bank.write`, and `O_mem_slot.detach()` detaches the representation. `aux["beta"]` is stashed in `last_aux_losses` but the trainer's `_collect_aux_loss` only sums `load_balance` (`train_mem_space_pg19.py:199-203`). So `gate_param` never moves during training.

Together these mean: **at every forward in every chunk, the joint-attention step is being asked to consume 64 freshly-sampled ~N(0, 1) hidden-dim vectors as extra K/V tokens, with no learnable transformation to adapt their distribution.** The selector's `Q_sel/K_sel` can only change which of the 512 i.i.d. random vectors enters — but since they are i.i.d., this choice is content-neutral. The 63.40 PPL is the intrinsic cost of that arrangement.

## 3. Ranked fix list

### #1 (load-bearing) — Always-on learnable `slot_to_hidden` / `hidden_to_slot` projections, zero-init

**Hypothesis.** The residual gap is structural: there is no parameterised path from slot vectors to the decoder's K/V distribution. A learnable projection, initialised to zero (Flamingo-style), makes the slot contribution to the residual stream start at **exactly bypass parity** (0 × anything = 0) and then *only* add slot content if training finds it reduces loss. This is precisely the "init so added module outputs zero" discipline the design doc §R2 asks for, which the Identity shortcut currently bypasses.

**Concrete code change.** `src/memory/mem_space/layer.py:191-200`:

```python
# REPLACE
if slot_dim == d_model:
    self.slot_to_hidden = nn.Identity()
    self.hidden_to_slot = nn.Identity()
else:
    self.slot_to_hidden = nn.Linear(slot_dim, d_model, bias=False)
    self.hidden_to_slot = nn.Linear(d_model, slot_dim, bias=False)
    nn.init.normal_(self.slot_to_hidden.weight, std=0.02)
    nn.init.normal_(self.hidden_to_slot.weight, std=0.02)
# WITH
self.slot_to_hidden = nn.Linear(slot_dim, d_model, bias=False)
self.hidden_to_slot = nn.Linear(d_model, slot_dim, bias=False)
# Flamingo-style zero init on slot_to_hidden so the memory path starts at
# bypass parity.  hidden_to_slot only affects writeback (discarded between
# chunks under _reset_banks), so a small-random init is harmless.
nn.init.zeros_(self.slot_to_hidden.weight)
nn.init.normal_(self.hidden_to_slot.weight, std=0.02)
```

The trainer already harvests these params (`train_mem_space_pg19.py:157-160`), so no trainer-side change is needed. Adds ~16.8M trainable params per layer × 32 layers ≈ 540M — large enough to matter, but still frozen-backbone regime (~7B backbone stays frozen).

**Expected PPL movement.** At step 0: `slot_to_hidden=0 → M_sel_hidden=0`. K and V for slot positions are `W_kv · 0 = 0`, so slot logits are 0; H-H logits are O(5-10); softmax mass on slots ≈ k·1 / (T + k·1) per H-query, and V_slot = 0 → slot contribution to the residual is 0. **PPL should land ≈ bypass 16.50 at step 0**, then move slowly (up or down) as the projection trains. Target healthy band [15, 25].

**Risk of making things worse.** Low. The zero-init start point guarantees no regression vs bypass; worst case training is noisy and PPL creeps up a few points. The 540M params might be over-parameterised for 200 chunks, but frozen-backbone training is not the bottleneck here — the signal-to-noise of the gradient is.

### #2 (diagnostic) — Wider T_cutoff in the slot-streaming mask

**Hypothesis.** If #1 lands at e.g. 40 rather than <30, the remaining gap is that slots are still harmful (because random). Tighten the visibility window: `T_cutoff = 3*T//4` (block slots from the first three quarters). A fully-blocked `T_cutoff = T` mathematically equals bypass; this knob interpolates.

**Concrete code change.** `layer.py:101-103` — change `T_half = T // 2` to a config-driven fraction, default `3//4`.

**Expected PPL movement.** Monotone with cutoff fraction: more blocking → closer to 16.50. At 3T/4 expect mid-30s → mid-20s. Does not require training.

**Risk.** None for PPL, but it masks memory architecture rather than fixing it — Stage-2 still needs slots to be usable by the earlier half of the chunk.

### #3 (tuning) — slot_dim=128 or 256 with learnable projections

**Hypothesis.** A variant of #1: force projections *and* compress slot rank. Cheaper (0.5M params/layer at slot_dim=128 instead of 16.8M) but strictly less expressive. May help if 540M trainable params at fixed 200 chunks is over-parameterised.

**Concrete code change.** Add `--slot_dim 128` CLI flag (new arg) and surface through `MemorySpaceConfig`. `slot_init=random, noise=1.0` continues to work.

**Expected PPL movement.** Similar ceiling to #1 but slower to train; at 200 chunks probably ~20-30.

**Risk.** Modest: if slot_dim is too small, selector and projection together lose the capacity to encode any useful slot representation once real (non-random) slot content comes in.

### #4 (weak) — Gate re-init / warmup policy

**Hypothesis.** Current β = σ(0) · 1.0 · 0.3 = 0.15 from step 1. Too aggressive before selector converges.

**Why this is weak.** Three independent reasons make this hypothesis near-impossible to matter for the 63.40 measurement:
- `_reset_banks` discards the bank every chunk — writeback cannot contaminate *this* chunk's forward
- `O_mem_slot.detach()` + `beta_val = float(beta_t.detach().item())` makes `gate_param` receive zero gradient, so it never moves regardless of warmup
- Even if it did move, there is no within-chunk feedback loop for the gate to affect

**Expected PPL movement.** None measurable. Skip.

### #5 (weak) — Load-balance weight tuning

**Hypothesis.** Usage concentrated, raise weight to 0.05 / 0.1 to diversify.

**Why this is weak.** With `slot_init=random, noise=1.0` + per-chunk reset, all 512 slots are i.i.d. gaussian noise — index choice is content-neutral. More uniform dispatch changes *which* i.i.d. sample gets picked, not *what* the attention sees. Diagnostic value only: inspect `slot_usage` to see whether selector has collapsed — but doing so does not help PPL.

**Expected PPL movement.** Negligible.

## 4. Recommended next experiment (single-dispatch)

**Dispatch #1 alone, on a SMOKE first, then a FULL if smoke passes.**

### Smoke command (replicates the 71.92 harness, expects ≤ 25)

Edit `src/memory/mem_space/layer.py:191-200` per §3 #1 above, then:

```bash
bash scripts/_run_mem_space_smoke_llama3.sh
```

(no flag changes — the smoke already passes `--slot_init random --slot_init_noise 1.0`).

### Smoke pass-contract

| Metric | PASS | WARN | FAIL |
|---|---|---|---|
| PPL | ≤ 25 | 25-40 | > 40 |
| nan_chunks | 0 | — | ≥1 |
| step-0 PPL (from eval-only) | 16-18 | 18-25 | > 25 |

If PASS: launch the 8-GPU FULL run with the same layer.py change. Expect full PPL in [16, 22].
If WARN: inspect whether training *hurt* (final > step-0). If yes, add `--lr 3e-4` (currently 1e-3) and re-smoke.
If FAIL: the projection zero-init is not reaching the decoder correctly — audit that the trainer actually picked up the new Linear params (log `n_trainable` in `train_mem_space_pg19.py:360-361`; expect a large jump from ~1M to ~540M).

### Diff to apply (exact)

`src/memory/mem_space/layer.py` lines 191-200:

```python
# Learnable slot↔hidden projections. We do NOT take the slot_dim==d_model
# shortcut (Identity) because that path has zero trainable capacity and was
# empirically responsible for the residual-gap pathology after fix1+fix2
# (see ops/research_notes/20260426_mem_space_v0_tier2_residual_gap.md).
self.slot_to_hidden = nn.Linear(slot_dim, d_model, bias=False)
self.hidden_to_slot = nn.Linear(d_model, slot_dim, bias=False)
# Flamingo-style: zero-init the path slots → attention so the memory
# contribution starts at exactly bypass parity and can only be lit up by
# training. Writeback projection (hidden_to_slot) is discarded between
# chunks via _reset_banks so its init is unimportant; keep small-random.
nn.init.zeros_(self.slot_to_hidden.weight)
nn.init.normal_(self.hidden_to_slot.weight, std=0.02)
```

## 5. Caveats / what we still don't know

- **If #1 lands at 40 but not below 30**: evidence that fix2's `T/2` cutoff is also load-bearing (hypothesis #2). Dispatch #2 next with `T_cutoff = 3T/4`.
- **If #1 lands ≤ 25 but does not improve below 17 after 200 train steps**: training signal is too weak — expected, because slots themselves are still per-chunk random noise. Real improvement over bypass requires *persistent* slot content, which needs (a) disabling `_reset_banks` or (b) scheduling resets only across documents, not across chunks of the same doc. Flag this for a Stage-1.5 design change.
- **We have not verified that `gate_param` is actually stuck** — worth a one-line log of `wrapper.gate_param.grad` pre-optimizer-step to confirm the zero-gradient claim in §2. If it turns out β *is* moving, revisit hypothesis #3.
- **`find_unused_parameters=True` is on in DDP** (`train_mem_space_pg19.py:371`). Once #1 removes the Identity branch, all layer params are always used → we can flip that off post-fix (minor perf win, not correctness).
- **The writeback path is currently inert (see §2 point 2)**. This is fine for a Stage-1 stability check but must be re-examined for Stage-2: slots that reset every chunk are not "memory" in the design-doc sense, they are noise tokens with a picked index.

## 6. One-paragraph verdict (for RESEARCHER_REPORTS.jsonl)

The 63.40 residual gap after fix1+fix2 is structural, not a hyper-parameter issue: when `slot_dim == d_model`, `slot_to_hidden` is `nn.Identity()`, which leaves **no trainable tensor between the slot bank and the decoder's K/V projections**, so every forward prepends 64 raw ~N(0, 1) hidden-dim vectors that the pretrained attention cannot adapt to. The load-bearing fix is to replace the Identity shortcut with a zero-initialised `nn.Linear(d_model, d_model, bias=False)` so the memory path starts at bypass parity and can only be lit up by training; the trainer already harvests these params, so this is a single-file code edit. Dispatch a SMOKE first — a PASS is PPL ≤ 25 with nan_chunks=0, and the eval-only step-0 PPL should be in [16, 18] (the zero-init prediction). Hypotheses #3 (gate warmup) and #5 (load-balance weight) cannot affect PPL in the current setup because `_reset_banks` discards per-chunk slot state and `gate_param` receives zero gradient; they should not be dispatched until #1 has been validated.
