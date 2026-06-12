# v5 — Cold-Start Alpha Gating (zero_alpha_on_cold_start)

**Date**: 2026-05-17
**Type**: Single-factor ablation over v2 L1+L3 baseline (not a new architecture)

## Architecture

Forward pass (per MemorySpaceLayer, per chunk):

```
1. cold_start_this_call = not memory_bank.is_initialized(B)
2. If cold_start:
     init memory_bank from hidden_states (same as before)
3. selector → top-k slot indices (same as before)
4. extended forward: [L3 | L1_slots | H] through wrapped LlamaDecoderLayer (same)
5. bypass forward: H through wrapped LlamaDecoderLayer (same)
6. alpha = tanh(slot_output_gate)
   IF cold_start_this_call AND cfg.zero_alpha_on_cold_start:
     alpha = 0
7. next_hidden = bypass_h + alpha * clip(slot_delta)
   → on cold start: next_hidden = bypass_h (pure bypass, no slot pollution)
8. writeback: O_mem → dual-gate update into memory bank (STILL EXECUTES on cold start)
```

**Key insight**: the first chunk initialises memory bank from random/noise. Before
any real content is written, slots are uninformative. Allowing `alpha > 0` on this
first call injects noise into hidden states across all 32 layers. By zeroing alpha
only on cold start, we:
- Prevent first-chunk hidden-state pollution (pure bypass on chunk 0)
- Still allow selector + extended forward + writeback to execute, so the first
  chunk's content IS written into the bank for subsequent chunks to read

## Initialization

- `zero_alpha_on_cold_start`: new bool config field, default False
- All other params identical to v2 L1+L3 baseline:
  - num_slots=512, top_k=64, selector_temperature=1.0
  - use_dual_gate=True, forget_bias_init=2.0, input_bias_init=0.0
  - use_l3_summary=True, l3_n_summary=64, l3_n_layers=2, l3_n_heads=8
  - shared_memory_bank=True, slot_init=random, slot_init_noise=0.05
  - slot_output_gate init=0.5 (tanh(0.5)≈0.46 on non-cold-start chunks)

## Relationship to prior work

- **v2 (L1+L3 baseline)**: this is v2 + one flag. No other changes.
- **v3 (short-fix)**: v3 suppresses writeback entirely when n_chunks==1 and
  oversamples 0k context. v5 does NOT touch v3's changes — it operates
  orthogonally by gating the output fusion rather than skipping writeback.
- **MemoryLLM (arXiv:2306.09315)**: uses a similar "memory is cold at start"
  observation but addresses it with a separate pre-fill stage. v5 is simpler:
  just zero the output gate on the first call.
- **Infini-attention (arXiv:2404.07143)**: initialises memory from EMA of past
  KV; on first chunk they use zeros which may cause similar cold-start artifacts.

## Known issues

1. This is a single-factor ablation — if the v2 baseline itself has other
   problems (e.g. selector degeneracy), this fix won't address them.
2. `cold_start_this_call` is per-layer but with `shared_memory_bank=True` the
   flag is True only for the FIRST layer's first call (subsequent layers see
   the bank as already initialized by layer 0). This is correct because layer 0's
   writeback already populates the shared bank; layers 1-31 read useful slots.
3. No warm-up of alpha after cold start — it jumps from 0 to tanh(0.5)≈0.46 on
   the second chunk. If this causes instability, a linear ramp could be added.
