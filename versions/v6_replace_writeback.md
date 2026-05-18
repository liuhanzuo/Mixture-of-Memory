# v6: Replace Writeback (slot ← s_new, no EMA)

**Date**: 2026-05-18
**Motivation**: EMA writeback causes early facts to decay → counting/tracking tasks (qa7) fail at long context

## Architecture

All other components identical to v2 (L1+L3, dual gate, shared bank, Llama-3.2-1B-Instruct backbone).

**Single change**: writeback rule
- **v2 (EMA)**:  `slot[idx] = (1 - β) · slot[idx] + β · s_new`  where β ≈ 0.15
- **v6 (replace)**: `slot[idx] = s_new`  (direct overwrite, no history blending)

`s_new = hidden_to_slot(O_mem_hidden)` — joint-attention output projected to slot_dim.  
Since `O_mem_hidden` is computed from `[slot_tokens | context_tokens]` joint attention,  
`s_new` already contains the current slot content mixed with the new context.

## Hypothesis

The EMA decay rate β≈0.15 means a fact injected at chunk 1 retains only  
0.85^N of its magnitude by chunk N. For qa7 (counting: "how many objects is John holding?")  
all N mentions must be equally remembered. With replacement, each slot always holds  
the **most recent chunk's summary** of the information that routes to it — no decay.

**Expected upside**: qa7/qa8 at long context should improve (no decay)  
**Expected risk**: qa1/qa5 at very long context may degrade (early facts evicted by later writes)

## Config flag

`--use_replace_writeback` in `train_mem_space_babilong.py`  
`MemorySpaceConfig.use_replace_writeback: bool = True`

## Relationship to prior work

- LM2 (dual gate): sets `g_forget ≈ 0` + `g_in ≈ 1` → approaches replacement, but learned per-feature
- v6 is a hard version of that: unconditional full replacement, simpler, easier to diagnose
- Titans (MAC): persistent memory is re-written at test-time without EMA → similar spirit

## Known issues

- If routing is poorly differentiated (top1_sim_mean ≈ 1/N), replacement means all slots converge  
  to the average of the most recent chunk, losing diversity. EMA at least retains some history.
