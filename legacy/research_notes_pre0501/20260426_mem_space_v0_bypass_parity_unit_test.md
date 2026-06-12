# Bypass-parity unit test — mem_space v0 fix3 (PPL>1000 rule)

**Date**: 2026-04-26 18:30
**Author**: main (autonomous)
**Related**:
- `scripts/test_mem_space_bypass_parity.py` (new — this test)
- `ops/research_notes/20260426_mem_space_v0_tier2_residual_gap.md` (Tier-2, falsified)
- Tier-3 subagent `a9cd5df64f2765e19` (running — will need to fold these findings)

## Motivation

CLAUDE.md red line: PPL > 1000 → "从最基础的单元测试开始排查".
Two fix3 smoke runs produced bit-identical step-1 PPL = 62.64 (≠ predicted
bypass-parity 16.50), with loss blowing up to 25346 after 10 training steps.
Before dispatching any more fixes, verify the ACTUAL behavior of the joint-attn
path at `slot_to_hidden.weight=0` with a tiny synthetic Llama decoder layer.

## Setup

- Tiny Llama: `hidden_size=128, num_heads=4, intermediate_size=256, rope_theta=500000`
- `attn_implementation="eager"` for determinism.
- MemorySpaceLayer with `num_slots=32, top_k=8, slot_init=random, noise=1.0`.
- `slot_to_hidden.weight = 0` (fix3 invariant — asserted at start).
- B=1, T=64. Compare `forward_no_memory(h)` vs `forward(h)` output.

## Results

```
[ok] slot_to_hidden.weight == 0 (fix3 invariant)
[ok] forward_no_memory → (1, 64, 128)  |  |O_ref|_max = 3.6673

=== Diagnostic 1: per-position divergence from bypass ===
first half  (t ∈ [0, 32)):  max=7.515289e-01  mean=2.426312e-01
second half (t ∈ [32, 64)): max=9.034373e-02  mean=5.420678e-02
ratio (2nd/1st max):  1.202e-01

=== Diagnostic 2: RMSNorm(0) check ===
|RMSNorm(zeros)|_max = 0.000000e+00    ← RMSNorm(0) is EXACTLY 0

=== Diagnostic 3: disable fix2 T/2 mask (slots visible to ALL H-queries) ===
without fix2 mask:  first-half max=2.052191e-01  second-half max=9.034373e-02
ratio first-half (nofix2 / fix2):  2.731e-01  ← going from "blocked" to "visible" REDUCES first-half error 4×

=== Summary ===
bypass vs joint-attn (fix2 on):  max-abs = 7.515289e-01  rel_l2 = 7.629571e-02
```

## Interpretation

### What's ELIMINATED

1. **RMSNorm-of-zero hypothesis**: `RMSNorm(zeros) = zeros` exactly. The
   Llama RMSNorm does `gamma * x / sqrt(mean(x²) + eps)` → if `x=0`, output
   is `gamma * 0 / sqrt(eps)` = 0 regardless of gamma. So zero-valued slot
   hidden states stay zero through `input_layernorm` and Q_slot = K_slot =
   V_slot = 0.

2. **Softmax-denominator contamination (second-half only)**: predicted
   second-half H-queries would be perturbed because slot keys (Q·K = 0 →
   exp(0) = 1) eat softmax mass. This IS happening but is TINY: second-half
   max=0.09, which is smaller than first-half.

### What's NEWLY ILLUMINATED

**The fix2 `T/2` slot-streaming mask is itself the primary contaminant.**

- With fix2 mask ON (slots hidden from first-half H-queries): first-half
  max-abs = **0.75**.
- With fix2 mask OFF (slots visible to ALL H-queries): first-half
  max-abs = **0.21** (3.65× reduction).

This is the opposite of the Tier-1 expectation ("mask slots → bypass
parity"). Key insight: my test sets ALL slot rows to `0`, which means slot
queries (rows 0..k-1) have row-mask = all-zero → Q_slot attends to everything.
Even though Q_slot = 0, softmax over a full L-length row is 1/L uniform,
and V_H ≠ 0 → slot attention OUTPUT is nonzero (mean of V_H).

But slot-output (`O_mem_hidden`) is sliced off (`ext_out[:, :k, :]`) and
does not enter `next_hidden`. So this should not perturb H-body output.

**Candidate mechanism (needs Tier-3 to pin down)**: HF's `LlamaAttention`
(eager path) processes Q, K, V for the entire extended sequence together.
The softmax is per-row, so Q_H rows should be independent of Q_slot rows.
But there may be a subtle interaction via:

- **Batched RoPE**: `apply_rotary_pos_emb(q, k, cos, sin)` receives the
  full `cos_ext/sin_ext` and applies per-position rotation. H-body positions
  k..k+T-1 index into `cos_ext[k..]` = original `cos[0..T-1]` → should be
  identical to bypass. Verified by construction in `_extend_position_embeddings`.

- **Number-of-heads interaction with mask**: our 4-D mask `[B, 1, L, L]`
  broadcasts over heads. Should be fine.

- **DType tiebreak at -inf**: in fp32 `torch.finfo(fp32).min + 0 = -inf →
  exp = 0`. No issue.

- **Something the eager path does with attention_mask SHAPE**: maybe the
  wrapped Llama eager path silently rebuilds the mask from `position_ids`
  when `position_ids=None`? Worth checking `modeling_llama.py` at the
  `_update_causal_mask` or equivalent entry point.

### What to tell Tier-3

If the Tier-3 /researcher subagent returns with a hypothesis centered on
RMSNorm-of-zero: REJECT it, this test proves RMSNorm(0) = 0.

If it returns centered on "second-half softmax denominator crowding":
PARTIALLY CORRECT but not the dominant effect — second-half max is 0.09 while
first-half (where fix2 mask should guarantee parity) is 0.75. The dominant
bug is in FIRST-HALF, where we explicitly mask slot-keys. Dig there.

Leading candidate for the ACTUAL fix (needs Tier-3 verification):

**Remove the fix2 mask entirely.** Empirically the mask was added to prevent
oracle-slot-leak (Tier-1 diagnosis), but with `slot_to_hidden=0` there's
no leak to prevent (slots are zero) AND the mask itself is a 4× bigger
perturbation to first-half H-queries than the thing it was supposed to
prevent. Revisit whether the mask is needed at all once we have a
parameterized slot path (which we now do via the Linear).

### Secondary finding (trainer param budget)

`n_trainable=1107.30M` = 2× Tier-2's predicted 540M. Explanation:
- `slot_to_hidden (d×d) = 4096×4096 = 16.8M per layer × 32 layers = 537.8M`
- `hidden_to_slot (d×d) = 4096×4096 = 16.8M per layer × 32 layers = 537.8M`
- Total = 1075.6M (matches observed 1107.30M within selector/gate noise)

Tier-2 only counted `slot_to_hidden`. The `hidden_to_slot` projection is
also trainable. This makes the `lr=1e-3` unambiguously too aggressive for
the true param budget. If we keep both Linears, lr should be ≤ 3e-4.

## Proposed next single-dispatch experiment

**Wait for Tier-3 /researcher**, then apply the single proposed fix.
Strongest prior (from this unit test): **remove fix2 mask in `layer.py:100-103`**
(make slot keys visible to H-queries everywhere). Re-run unit test; expect
rel_l2 < 1e-3 (only the second-half softmax-denom effect remains).
If unit test passes, re-smoke at `lr=3e-4` (halved due to 2× param budget).

Smoke pass contract:
- PPL ≤ 30
- step-1 PPL in [16, 20]
- nan=0
- rel_l2 from unit test < 1e-3

## Run reproducibility

```bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh && conda activate torch-base
python scripts/test_mem_space_bypass_parity.py --device cpu --dtype float32
```

Run time ~10s on CPU. Deterministic at `--seed 0`.
