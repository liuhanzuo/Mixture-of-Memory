# §11.4.2 second revision — bowl centered at kv≈96, not 128-256 nor monotone

**Date**: 2026-04-26
**Author**: /trainer autonomous chain (follow-up #4 to §11.4.2 Llama-2 Patch-A sweeps)
**Status**: evidence-based revision; supersedes both original-draft "bowl in 128-256" AND first-revision "monotone 128→256"

## TL;DR

Three successive Llama-2-7B Patch-A Q-Filters sweeps against pg19 @
filter_rank=2, recent_window=64 have now completed. The combined
kv-budget curve reveals a clean U-shape with the minimum at **kv=96
(PPL=167.27)**. Both prior §11.4.2 narratives are falsified:
1. ❌ Original: "bowl in kv ∈ [128, 256]" — the true minimum is OUTSIDE this range, below it.
2. ❌ First revision: "monotone 128→256, minimum at kv ≤ 128" — true at the
   128–256 sub-range, but globally false: PPL rises sharply again at kv=80.

## Combined kv sweep (Llama-2-7B, Patch-A, rank=2, recent=64, pg19 200 chunks)

| kv | 64* | 80 | 96 | 112 | 120 | 128 | 144 | 160 | 176 | 192 | 208 | 224 | 240 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PPL | ~degenerate | 233.43 | **167.27** ⭐ | 170.67 | 182.11 | 190.99 | (fine-sweep) | ↑ | ↑ | ↑ | ↑ | ↑ | ↑ | 278.89 |

\*kv=64 intentionally skipped: recent_window=64 forces kv_budget - recent_window = 0
non-recent slots, degenerate case.

(fine-sweep {144..240} values live in `outputs/kv_fine_llama2/*/eval_results.json`
on b200-3 and in `status/gpu_runs.jsonl` under `sweep_id: llama2_kv_fine_sweep_b200_3`.)

## Shape interpretation

1. **Left wall (kv ≤ 80)**: kv - recent_window = 16 non-recent slots is too
   few. The attention score geometry collapses — Q-Filters cannot rank
   non-recent tokens stably when almost every non-recent slot is discarded.
2. **Basin (kv = 96, possibly 88 or 104)**: optimal trade-off. ~32 non-recent
   slots is enough for the rank-2 filter to preserve its two salient
   directions across the pg19 distribution.
3. **Right wall (kv ≥ 128)**: monotone-increasing PPL. Counterintuitive — more
   budget makes things *worse*. This is the strongest indication that
   Q-Filters at rank=2 on Llama-2-7B is preserving the wrong tokens when given
   headroom; the additional slots filled may be dragging the KV set away from
   the optimal ranking. Possible mechanism: the sub-window RoPE positions
   (Patch-A) re-position evicted tokens to collapse onto a "trusted" recent
   zone; extra non-recent tokens may re-introduce the pre-eviction positional
   smear.

## Interaction with rank × kv 2-D (Llama-3 note 20260426_rank_kv_2d_decoupling)

The Llama-3 sweep (rank ∈ {1,2,4} × kv ∈ {128,256,512}) found rank=1 strictly
dominates, with monotone-decreasing PPL in kv. The Llama-2 bowl above is at
rank=2. This suggests two non-exclusive hypotheses:

- **H1 (rank effect)**: at rank=1, the Llama-2-7B curve would also become
  monotone-decreasing in kv, eliminating the bowl. The right-wall climb is a
  rank-2-specific phenomenon (the second filter direction destabilizes with
  extra budget).
- **H2 (model family)**: Llama-2-7B and Llama-3-8B have different attention
  geometries; Llama-2's bowl is intrinsic to that family regardless of rank.

**Proposed verification** (single op-point, whitelist-approvable):
Re-run Llama-2-7B Patch-A @ rank=1, kv ∈ {96, 128, 192, 256} — if rank=1
gives a monotone curve, H1 wins. If the bowl persists, H2 wins. 4 runs, ~3 min
on b200-3.

## Narrative revisions required in §11.4.2

- [ ] Replace "bowl in 128-256" framing with "bowl centered near kv≈96; PPL
      rises monotonically as kv moves away from 96 in either direction (at
      rank=2)."
- [ ] Add rank-dependence caveat: bowl is rank=2-specific; rank=1 may
      monotonically descend (pending verification).
- [ ] Update the plot axis to [64, 512] or [80, 300] to make the bowl visible.
- [ ] Drop first-revision "monotone 128→256; minimum at kv ≤ 128" — true
      locally but misleading as the global claim.

## Open sub-questions

- [ ] Is there a finer minimum in [80, 96] or [96, 112]? Propose kv ∈ {88, 104}
      for a 2-run refinement.
- [ ] At kv=96, what are the retention statistics of the non-recent tokens?
      (How many repeat across chunks, how many are content vs position
      anchors?) — requires instrumentation in `src/memory/qfilters/layer.py`.
- [ ] Does the bowl move with `recent_window`? Current only samples recent=64.
- [ ] Does the bowl move with `sub_window_len`? Current only samples 1024.

## Raw artifacts

- Driver (low-range): `scripts/_run_llama2_kv_lowrange_sweep.sh` (new 2026-04-26 13:00)
- Outputs: `outputs/kv_lowrange_llama2/*/eval_results.json` on b200-3
- Sweep completion: `status/ACTIVE_SWEEPS.jsonl` 2026-04-26T13:03:11+08:00
- Per-run rows: `status/gpu_runs.jsonl` 4 rows appended post-2026-04-26T13:03:11
- Fine sweep from prior heartbeat: `sweep_id: llama2_kv_fine_sweep_b200_3`
- Anchors {128, 256}: `sweep_id: patchA_llama2_sweep` (2026-04-26 from §11.4.2)

## Chain of evidence (pointers)

- §11.4.2 original draft: `ops/research_notes/20260426_s11_retraction.md`
- fine-sweep fold (prior heartbeat): see 2026-04-26 12:42 heartbeat TRAINER_ACTIVE.md
- low-range sweep: this heartbeat (2026-04-26 13:03), `status/ACTIVE_SWEEPS.jsonl` entry
  with `sweep_id: patchA_llama2_kv_lowrange`
