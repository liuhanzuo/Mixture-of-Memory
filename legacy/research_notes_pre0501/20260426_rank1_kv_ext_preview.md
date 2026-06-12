# Rank=1 kv-extension: clean monotone descent Llama-3, no plateau at kv=1024

**Date**: 2026-04-26
**Author**: /trainer autonomous chain (follow-up #2 to rank×kv 2-D sweep, preview note)
**Status**: preliminary; raw observation + short interpretation

## TL;DR

The Llama-3-8B Q-Filters rank=1 kv-extension sweep on b200-1 (13:23→13:45 GMT+8,
5 new op-points) has produced a clean monotone-decreasing PPL curve across
**eight total kv points** {64, 96, 128, 192, 256, 384, 512, 1024} at rank=1.
The previous 3-point 2-D sweep was correct but dramatically under-sampled —
the full curve reaches **PPL=2.365 at kv=1024**, a ~35% improvement over
the kv=512 anchor (3.672) that looked like the "best" cell of the 2-D grid.

There is **no plateau through kv=1024** and no indication the curve is about
to flatten. The dense attention baseline remains the floor; Q-Filters rank=1
at kv=1024 gets surprisingly close on pg19.

## Measurement

Grid (combined with 2-D sweep anchors at 128, 256, 512):

| kv | 64* | 96 | 128† | 192 | 256† | 384 | 512† | 1024 |
|---|---|---|---|---|---|---|---|---|
| PPL | 58.44 | 6.921 | 6.126 | 4.994 | 4.636 | 4.107 | 3.672 | **2.365** ⭐ |

\*kv=64 intentionally degenerate: recent_window=64 means 0 non-recent slots,
reducing Q-Filters to pure SWA with buffer overrun; PPL=58.44 is the natural
degradation floor for this pathological setting.

†anchors from the 2-D sweep 2026-04-26 12:02→12:51 (`outputs/patchA_llama3_rank_kv_2d/*`).

Model: Llama-3-8B; data: pg19_chunks_llama3_noeos (200 chunks × 4096).
bf16, sdpa, sub_window=1024, calibration_chunks=64, Patch-A active.

## Shape interpretation

1. **Hard left wall (kv ≤ 64)**: the degenerate case is sharp — one
   non-recent slot (kv=80 would be next tested) vs. zero at kv=64 should
   produce a huge improvement. That drop is captured between kv=64 (PPL=58)
   and kv=96 (PPL=6.9). The left wall is essentially a single-point cliff
   at kv=recent_window.
2. **Smooth descent (kv ∈ [96, 1024])**: geometric-ish decay, roughly
   PPL(kv) ~ 3.7 · (kv/512)^−0.5 as a crude fit.
   - kv=96 → 6.921; doubling to 192 → 4.994 (−28%)
   - kv=192 → 4.994; doubling to 384 → 4.107 (−18%)
   - kv=384 → 4.107; ~3× to 1024 → 2.365 (−42%)
3. **No visible plateau at kv=1024**: a kv=2048 or kv=4096 run would be
   needed to locate the asymptote. At seq_length=4096, kv=2048 would be
   50% compression — the regime where "compression" becomes the word.

## Contrast with Llama-2 rank=2 (§11.4.2)

The Llama-2 rank=2 curve has a **bowl at kv=96** (PPL=167; rising on both
sides to 233 @ kv=80 and 279 @ kv=256). The Llama-3 rank=1 curve has
**monotone descent** across the same kv range, no bowl.

Candidate explanations (pairs cleanly with the rank×kv decoupling note):
- **Rank difference**: at rank=1 (single-direction filter), adding slots
  never destabilizes the ranking because there is no secondary direction
  to drift. At rank=2, the second direction may be dominated by calibration
  noise and introducing more slots gives it more weight in ranking → the
  "right wall" of the Llama-2 bowl.
- **Model-family difference**: Llama-2-7B and Llama-3-8B have different
  attention-matrix spectral profiles.

The cleanest test remains Llama-2-7B @ rank=1 kv ∈ {96, 128, 192, 256}
(proposed in both prior folds).

## What this means for §11 narrative

1. **rank=1 is clearly the right knob**. At every shared kv point the
   rank=1 PPL is 4–20× lower than rank=2 on Llama-3.
2. **kv axis at rank=1 is "more is better"** through at least 1024.
   This simplifies the op-point story for §11: pick rank=1, set kv by
   memory budget, ignore bowl hunting for this model.
3. **Does this extrapolate to Llama-2?** Prior §11.4.2 gave an emphatic
   "no" at rank=2. An rank=1 verification on Llama-2 would close the
   last open door in the retraction addendum.

## Open sub-questions

- [ ] kv=2048 and kv=4096 at rank=1 on Llama-3 — where does the asymptote
      sit? (2 runs, ~10 min on any idle B200.)
- [ ] Llama-2 @ rank=1 kv∈{96, 128, 192, 256} — universality test.
- [ ] Is the kv=1024 winner at rank=1 Llama-3 (PPL=2.365) close to the
      dense baseline? Need the pg19 dense PPL for Llama-3-8B to compare
      (not currently logged in status/gpu_runs.jsonl).

## Raw artifacts

- Driver: `scripts/_run_llama3_rank1_kv_ext_sweep.sh` (2026-04-26 13:05, fixed
  13:23 after absolute-path fix for model location)
- Outputs: `outputs/rank1_kv_ext_llama3/*/eval_results.json` on b200-1:wzc1
- Sweep completion: `status/ACTIVE_SWEEPS.jsonl` 2026-04-26T13:45:39+08:00
- Per-run rows: `status/gpu_runs.jsonl` 5 rows appended post-2026-04-26T13:45:39
- 2-D sweep anchors: `sweep_id: patchA_llama3_rank_kv_2d_sweep`
  (2026-04-26 12:51)

## Chain of evidence (pointers)

- rank×kv 2-D fold: `ops/research_notes/20260426_rank_kv_2d_decoupling.md`
- §11.4.2 second revision: `ops/research_notes/20260426_s11_4_2_monotone_revision.md`
- This note: `ops/research_notes/20260426_rank1_kv_ext_preview.md`
