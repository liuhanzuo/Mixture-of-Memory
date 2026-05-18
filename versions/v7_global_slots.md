# v7: Hybrid EMA + Global Always-On Slots

**Date**: 2026-05-18
**Motivation**: Keep retrieval quality of EMA slots while adding EMA-free accumulation registers for counting/tracking

## Architecture

All components identical to v2 except routing and writeback are split:

### Regular slots (indices 0..N-g-1, default 504 of 512)
- Selected by top-k routing score as usual
- Updated with EMA: `slot[idx] = (1-β)·slot[idx] + β·s_new`

### Global always-on slots (indices N-g..N-1, default slots 504..511)
- Appended to `idx` **unconditionally** on every forward call (bypass top-k score)
- Updated with replacement: `slot[glob_idx] = s_new[glob_idx]`  (no EMA)
- These 8 slots see EVERY chunk and accumulate a running state without decay

## How it works in code

1. `layer.py`: after `idx = selector(hidden_states, slots)` returns [B,k], append  
   `global_idx = torch.arange(N-g, N)` → idx becomes [B, k+g]
2. Extended sequence gains g extra slot tokens (8 << 64 top-k, negligible overhead)
3. Writeback splits: first k columns of O_mem_slot → EMA write; last g columns → replace write

## Hypothesis

- Global slots see every chunk → can accumulate `John carried: {milk, football, ...}` state
- Model learns to write running state into global slots via end-to-end gradient signal
- Regular top-k slots remain specialized for content-addressed retrieval (qa1, qa5)
- Expected: qa7/qa8 (counting) improve; qa1/qa5 (retrieval) roughly unchanged

## Config flags

`--num_global_slots 8` in `train_mem_space_babilong.py`  
`MemorySpaceConfig.num_global_slots: int = 8`

## Relationship to prior work

- RMT/ARMT: recurrent memory tokens pass through every chunk (similar to global slots)  
  but use learned read/write projections and can decay over time via gradient updates
- v7 global slots are simpler: unconditional pass-through + hard replacement, no learned gating
- Titans (MAC): persistent memory as context tokens — global slots are a learnable analog

## Ablation directions

- Try `num_global_slots in {4, 8, 16, 32}` — more slots = more capacity but more overhead
- Try `num_global_slots = num_slots` (all slots always-on) → equivalent to v6 replace writeback
- Try giving global slots a slower EMA (β=0.5) instead of hard replacement
