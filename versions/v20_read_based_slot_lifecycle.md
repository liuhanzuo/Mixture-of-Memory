# v20 — Read-based slot lifecycle (soft-decay A + explicit-eviction B)

## Motivation (2026-06-12, decisive diagnostic)

`dead_slot_read_mass` measurement on N384 (qa1 8k/16k/32k): the read path puts
**93-95% of its attention mass on slots that were NEVER written** (top-k delta-rule
only writes ~24/384 slots over a full 32k sample). Per-slot normalized, read is
**uniform** over written vs never-written slots (ratio 1.00) — the read mechanism
does not care whether a slot was written; it attends by key similarity only.

Two consequences:
1. The trained delta-rule write contributes only ~5% of the read-out signal; the
   bulk of "memory" is the *unmodified strided-token chunk-0 snapshot* being read.
2. The existing dead-slot recycle judge (`_cum_usage==0`, i.e. "never selected for
   WRITE") is the WRONG criterion: read≠write. A slot can carry a long-range fact
   (read-active) while never being written — the window-write judge (R1) flushes
   exactly those, which is why R1 wrecked long-range (8k-32k −3×).

User design constraint (the "password slot" problem): a slot holding a rarely-read
but critical fact (a password) may go many chunks without being read; a naive
window-based eviction flushes it before the late query arrives. **Any eviction has a
trade-off; pick it by experiment, not by prior.** Implement two candidate policies.

## Shared foundation: per-slot cumulative read-mass

New persistent layer-0 accumulator `_cum_read_mass : [B, N]` (float), analogous to
`_cum_usage`. Each chunk, layer-0's `MemoryCrossAttentionRead.read` stashes the
per-slot read mass `read_mass_per_slot : [B, N]` (= `attn_weights[...,:N]` summed
over heads+query-tokens, mean-normalized so it sums to ~1 over slots per token, then
summed over tokens). Layer-0 accumulates it: `_cum_read_mass += read_mass_per_slot`.
Reset on cold-start (new sample), same lifecycle as `_cum_usage`.

This gives, at any chunk, a per-slot "how much has this slot EVER been read" signal
— the correct liveness measure (vs the write-based `_cum_usage`).

## Arm A — soft read-decay (`--slot_read_decay_rate R`, default 0 = off)

Every chunk (or every `interval`), multiply each slot's *value content* by a decay
factor that is a function of its recent read mass: slots read recently keep their
norm; slots not read decay slowly toward zero. NOT a hard flush — a later read can
still pull signal from a partially-decayed slot, and any future write/refresh revives
it. Concretely (no_grad, layer-0 driven, applied to `memory_bank.slots`):

    decay_i = 1 - R * (1 - recent_read_frac_i)     # recent_read_frac in [0,1]
    slots_i *= clamp(decay_i, min_keep, 1.0)

where `recent_read_frac_i` = this slot's read mass over the last `interval` chunks
normalized by the max over slots. `min_keep` (e.g. 0.5) bounds how far one interval
can decay a slot so an unlucky gap can't zero a password in one step. Tunable:
`--slot_read_decay_rate` (R), `--slot_read_decay_interval`, `--slot_read_decay_min_keep`.
Trade-off: gentle, never hard-deletes, but a *very* long unread gap eventually fades
the fact below retrievability.

## Arm B — explicit eviction + read-mass protection (`--slot_evict_mode readmass`, default off)

At each reset boundary (reuse `_dead_slot_reset_interval` plumbing), evict the slots
with the LOWEST cumulative read-mass AND zero recent writes, capped at a max evict
count per boundary, with a protection window: a slot is eligible only if it has been
"cold" (read_mass below a floor) for ≥ `--slot_evict_protect_chunks` consecutive
boundaries. Evicted slots are re-initialized (reuse `recycle_reset` + grace force-write
path) to fresh strided current-chunk content so they become writable空槽. Tunable:
`--slot_evict_mode {off,readmass}`, `--slot_evict_max_frac`, `--slot_evict_floor`,
`--slot_evict_protect_chunks`.
Trade-off: frees space for new info, but if the pool fills and a password slot has
been cold past its protection window, it can still be evicted right before its query.

## Code isolation (both arms独立开关, must not interfere)
- Foundation (`_cum_read_mass` + per-slot read stash): shared, always-on telemetry
  (no behavior change when both arms off). Lives in selector.py read() + layer.py.
- Arm A logic: new block in layer.py guarded by `slot_read_decay_rate > 0`.
- Arm B logic: extends the existing recycle block, guarded by
  `slot_evict_mode == "readmass"`; when off, current `_dead_slot_criterion` behavior
  is byte-for-byte unchanged.
- A and B can be enabled independently or together; default config = both OFF = P11.

## Judge
NIAH "early-plant / late-query" stress (fact in chunk-0, query at 32k) — the
password worst case — plus standard BABILong qa1/qa5 0k-32k. Compare A, B, A+B vs
P11 baseline. Decide policy by data.
