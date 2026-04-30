# Rank × kv 2-D decoupling: rank=1 strictly dominates

**Date**: 2026-04-26
**Author**: /trainer autonomous chain (follow-up #3 to rank×kv 2-D sweep)
**Status**: preliminary observational note; awaits researcher deeper dive

## TL;DR

The Llama-3-8B Q-Filters rank × kv 2-D sweep run on b200-1 (12:02→12:51 GMT+8)
showed that the `filter_rank` and `kv_budget` axes are **not diagonal** — the
joint optimum is not migratory. Instead, `filter_rank=1` strictly dominates
at every kv tested, and within rank=1 the PPL is monotone-decreasing in kv.

This directly contradicts the prior heuristic that "higher rank is fine if
kv is big enough" and the related "diagonal trade-off" narrative that
appeared in earlier §11 drafts.

## Measurement

Grid: rank ∈ {1, 2, 4} × kv ∈ {128, 256, 512} @ recent_window=64, 9 runs total.
Model: Llama-3-8B; data: pg19_chunks_llama3_noeos (200 chunks × 4096).
All sub-window = 1024, bf16, sdpa, Patch-A active. Calibration: 64 chunks.

Llama-3-8B pg19 PPL:

| rank \ kv | 128 | 256 | 512 |
|---|---|---|---|
| 1 | 6.126 | 4.636 | **3.672** ⭐ |
| 2 | 36.32 | 42.29 | 25.79 |
| 4 | 102.04 | 74.56 | 42.44 |

Interpretation (raw):
- **Rank dominance is severe**: rank=1 PPL=6.13 at kv=128 beats rank=2 PPL=25.79
  at *kv=512* by a factor of ~4×, even though rank=2-kv=512 has 4× the memory
  budget. Spectral cutoff matters more than raw budget.
- **Within rank=1**: monotone descent 6.126 → 4.636 → 3.672 as kv grows —
  predictable and clean.
- **Within rank=2**: non-monotone 36.32 → 42.29 → 25.79 — this inversion at
  intermediate kv is surprising and is the strongest remaining source of
  confusion. Hypothesis: at rank=2 the filter has two directions, but with
  kv=256 the eviction policy may be preserving mostly redundant tokens from
  the primary direction, while kv=512 gives enough room for secondary
  direction to contribute. This should be verified with explicit ablation.
- **Within rank=4**: monotone descent but PPL remains much worse than rank≤2.

## Why does rank=1 win?

The Q-Filters method (Godey et al. arXiv:2503.02812) projects the Q matrix
onto a low-rank basis extracted from calibration. In principle a higher-rank
filter captures more variance, which should help. But:

1. **Calibration noise**: rank=1 picks the top singular direction, which is
   stable across calibration chunks. Rank=2 and rank=4 add directions whose
   signal-to-noise ratio drops rapidly.
2. **Recent-window interaction**: recent_window=64 already holds the most
   informative recent tokens. The compressed slots only need to rank the
   non-recent context by a single dominant axis; higher rank may spread the
   saliency score more uniformly and effectively pull noise into the "kept"
   set.
3. **Base-model spectral concentration**: Llama-3-8B's attention matrices
   may be more rank-1-concentrated than the Llama-2/3.1 tested in the
   original Q-Filters paper, where rank-2 was reported as best.

## Universality question (for researcher follow-up)

- Is filter_rank=1 a universal ceiling, or Llama-3-specific?
  - Llama-2-7B Patch-A sweep used rank=2 throughout (§11.4.2). Re-evaluating
    Llama-2-7B at rank=1 would be the cleanest test. Proposed op-point:
    rank=1 @ kv=128 on the same config as §11.4.2, direct comparison to
    kv=128 rank=2 PPL=190.99.
- Does the rank-dominance invert in any regime (e.g., very small kv, very
  large seq_length beyond 4096, different domain)?
- Is there a principled way to predict the optimal rank from a model's
  attention-matrix SVD (so we don't have to sweep)?

## Narrative revisions required

1. §11 (Q-Filters on Llama-3) — any passage claiming "rank and kv trade off
   along a diagonal" or "higher rank compensates for smaller kv" is falsified
   for this regime. The joint minimum is *corner-seeking*: lowest rank, largest
   kv.
2. Earlier researcher-report hypothesis that "rank=2 is the sweet spot" was
   based on kv=256 alone and should be scoped narrowly to that single operating
   point on Llama-3-8B — it does not generalize.

## Raw artifacts

- Driver: `scripts/_run_llama3_rank_kv_2d_sweep.sh`
- Outputs: `outputs/patchA_llama3_rank_kv_2d/*/eval_results.json` (on b200-1)
- Sweep completion logged: `status/ACTIVE_SWEEPS.jsonl` 2026-04-26T12:51:41
- Per-run rows: `status/gpu_runs.jsonl` lines appended this heartbeat (9 rows)

## Open sub-questions

- [ ] Is the rank=2 non-monotone-in-kv inversion reproducible? (repeat kv=256
      with different calibration chunks)
- [ ] Does rank=1 on Llama-2-7B (§11.4.2) beat rank=2 at kv=128? — proposed
      single-op-point run, whitelist-approvable.
- [ ] Does rank=1 on pg19 extend well beyond kv=512 (i.e., keep descending at
      kv=1024, 2048)? — in-progress as rank=1 kv-extension sweep (follow-up
      #2), see note 20260426_rank1_kv_ext_preview.md when complete.
