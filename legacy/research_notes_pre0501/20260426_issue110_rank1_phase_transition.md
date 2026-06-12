# Issue #110 — Llama-2 rank=1 kv≥192 outlier: structural phase-transition diagnosis

**Date**: 2026-04-26 (post-seed-exhaustion)
**Author**: /researcher, thread B
**Status**: diagnosis proposal; prescribes a single disambiguating experiment

## 1. Problem restate

Post exact-SVD fix (`calibration.py` 2026-04-26), Llama-2-7B pg19 200 × 4096,
rank=1, recent_window=64, calibration_chunks=64, sub_window_len=1024, sdpa, bf16.
`mode=qfilters`. Multi-seed characterization on b200-3 finished at 17:xx:

| kv_budget | keep_old = kv − recent | seed 0 | seed 1 | seed 2 | std |
|-----------|------------------------|--------|--------|--------|-----|
|  96 | 32  | 107.006 | — | — | — |
| 128 | 64  | 150.567 | — | — | — |
| **192** | **128** | **479.264** | **479.264** | **479.264** | **0.0** |
| **256** | **192** | **610.872** | **610.872** | **610.872** | **0.0** |

**Seed channel exhausted.** With 3 seeds at each of kv ∈ {192, 256} producing
bit-identical PPL, the 479 / 611 outlier is **not stochastic**. It is a
reproducible *structural* failure of the rank=1 pipeline at kv ≥ 192.
Any remaining explanation must live in the deterministic data-path, not in
calibration stochasticity.

A second suggestive fact: the "acceptable" band (107/151) corresponds exactly
to cells where `keep_old ≤ recent_window = 64`, and the outlier band
(479/611) corresponds to cells where `keep_old > recent_window`. The boundary
is at `kv = 2 · recent_window = 128`.

## 2. Hypothesis evaluation

For each candidate we give (a) the specific single-GPU, 200-chunk,
no-code-change experiment that confirms or rules it out, and (b) the expected
PPL signature.

### H_sink — rank=1 filter mis-scores attention-sink tokens

Attention-sink tokens (positions 0–3 in pg19 chunks, heavily used as bias
reservoirs per Xiao et al. 2023) carry enormous attention mass but sit in
*idiosyncratic* directions unrelated to the leading right-singular vector of
post-RoPE Q. At rank=1 they may receive near-zero filter score and get
dropped. At kv = 96/128, keep_old ∈ {32, 64} is already tiny and dropping
sinks is one-of-many degradations; at kv ≥ 192, keep_old ∈ {128, 192} is
large enough that losing the sink becomes the dominant perturbation.

**Test (code change needed — 2 lines in `compress_kv`)**: pin tokens 0..3 of
the full cache as permanently-kept (`keep_sink=4`). Compare PPL at
kv=256 rank=1 recent=64 with and without sink-pinning.

**Signature**: if H_sink, PPL with sink-pin → [140, 220] band (matches the
pre-fix "lucky-draw" value); without pin → 611 (baseline).

**Caveat**: in a 4096-token chunk with sub_window_len=1024, sink (positions
0–3) is always in the **first** sub-window's cache and becomes "old" the
moment sub-window 2 starts. recent_window=64 never reaches back to the sink.
So H_sink does predict near-universal sink loss — it would need to be
compounded by another mechanism to produce the *phase* pattern at exactly
kv=128→192, unless filter scoring rank=1 is just barely good enough at
kv≤128 to occasionally pick the sink up.

**Posterior**: medium. Predicts qualitative pattern but not the sharp
threshold at `keep_old = recent_window`.

### H_phase — phase transition at `keep_old = recent_window`

The "acceptable / outlier" split lines up exactly with `keep_old ≤ recent_window`.
A candidate mechanism: at rank=1 the filter signal is noisy (filter captures
only ~60–80% of the head's useful Q energy per prior rank×kv 2D sweep on
Llama-3), so the 3000 "old" keys have **most scores near-ties** in abs-cosine
space. With keep_old ≤ 64 we pick from the top tail only (genuine highest
scores), which is reliable; with keep_old ≥ 128 we dip into the middle of
the noise distribution and select ~arbitrarily, **diluting** the genuine
high-score keys that made keep_old=64 work. Recent_window=64 is a hard floor
of "known-useful" keys that gets overwhelmed when keep_old grows past it.

**Test (no code change)**: at kv=256 rank=1, sweep recent_window ∈ {64, 128,
192, 256}. For each cell, `keep_old = 256 − recent_window`. Same filters,
same calibration.

**Signature**:
- `recent=64` (baseline): 611.
- `recent=128` (keep_old=128, at boundary): expect PPL ≈ 150–300 if H_phase.
- `recent=192` (keep_old=64, **below** boundary, matches kv=128 regime):
  expect PPL ≈ 107–150 if H_phase (the genuine filter signal is undiluted).
- `recent=256` (keep_old=0, pure SWA at 256): useful as an absolute ceiling;
  Llama-2 pure SWA at 512 was 1469 PPL in prior §8, so SWA-256 likely
  ≥ 1500 PPL — i.e. filter is net helpful but saturating near keep_old=64.

A monotone drop from 611 → ~150 as recent grows, with the curve hitting floor
at recent=192, **confirms H_phase**.

**Posterior**: high. Cleanly explains the sharp threshold coordinate
`keep_old = recent_window` at the observed data.

### H_per_head — bimodal per-head filter quality

5% of heads in the prior cross-run diagnostic had `|cos(filter_i, filter_j)|
< 0.5` even across runs with nominally identical configs (from
`20260426_issue110_rank1_kv256_ppl752_rootcause.md` §2). Exact-SVD now makes
filters deterministic but does **not** fix "the top-1 singular direction of
this head's Q is ill-defined because its top 2 singular values are near-equal,
so exact SVD still returns an arbitrary-rotating direction within a 2-D
subspace". Those ~5% heads would score their keys essentially randomly.
At kv ≤ 128 the small keep_old hides bad heads under the recency floor; at
kv ≥ 192 the bad heads contaminate attention.

**Test (no code change, offline)**: dump calibration filters, and for each
`(layer, head)` compute the singular-value gap ratio `S[0] / S[1]` from a
second-pass exact SVD on the post-RoPE Q. Count heads with gap < 1.05
(near-degenerate). If H_per_head is right, ≥ 5% of heads have
near-degenerate leading pair, and those heads should correlate with layers
that contribute most to the PPL explosion.

**Signature**: if ≥ 5% heads have S[0]/S[1] < 1.05 and masking-out those
bad heads (set filter to zero → H_per_head's compress_kv treats them as
unfiltered → fall back to keep-last-kv_budget) drops PPL below 300 → H_per_head.

**Posterior**: medium–high. Direct evidence in prior diagnostic, but the
mechanism through which this produces a phase at keep_old=recent is less
direct than H_phase.

### H_numerics — bf16 ties break deterministically-but-arbitrarily

`compression.py` already casts to fp32 before scoring (`f = filters.to(fp32)`,
`k = keys.to(fp32)`), so the **scoring arithmetic** is fp32. But the keys
themselves are stored in bf16 after the original attention forward, so there
is one round of 7-bit-mantissa rounding at store-time that could create
scoring ties between keys that differed in the ignored bits.

**Test (no code change)**: recompute scores in fp32 but also cast **keys to
fp32 before storage** by loading model in fp32 (`--bf16 False` off flag). At
kv=256 rank=1 recent=64, if PPL stays at 611 ± 5%, H_numerics is ruled out;
if PPL drops to < 300, H_numerics is dominant.

**Signature**: order-of-magnitude change between bf16 and fp32 confirms;
< 10% change rules out.

**Posterior**: low. fp32-only scoring is already the status quo, and
bf16-storage ties would have to be systematically biased *toward* anti-sink
directions to produce this pattern. Llama-2-dense PPL in bf16 is fine (300),
so bf16 storage alone is not pathological.

### H_recent_mask — sub_window_len=1024 leaks keys across boundaries

Each 4096-token chunk is split into 4 sub-windows of 1024. `QFiltersCache`
carries across sub-windows within the chunk. The `recent_window=64` applied
during compression is defined *relative to the cache's current T*, not to
the sub-window boundary. After sub-window 1 (T=1024), compression runs;
after sub-window 2 (T=2048 before compress), compression runs again on the
carried cache.

**Audit (already done above)**: `QFiltersCache.update` appends new keys;
`compress_layer` reads `layer.keys.shape[-2]` which is the CURRENT cache
length *after* append. `recent_window=64` cuts the last 64 of **that**
current-length cache. No branch depends on sub-window index; no off-by-one
between physical and logical positions after the 2026-04-25 Patch-A
re-rotation. Nothing in the scoring path references sub_window_len.

**Test**: set `--sub_window_len 4096` (single-forward per chunk) at kv=256
rank=1. If PPL drops dramatically, H_recent_mask; if unchanged (bit-identical
to 611), ruled out.

**Signature**: > 50% PPL change confirms; < 5% rules out.

**Posterior**: very low. Code inspection shows no sub_window-index-dependent
branch in either compression.py or layer.py beyond the Patch-A re-rotation
that already uses **logical** seen_tokens. The issue also reproduces across
3 seeds with identical sub_window_len=1024.

## 3. Ranking

| # | hypothesis | posterior | disambig value | rank score | notes |
|---|-----------|-----------|----------------|------------|-------|
| 1 | **H_phase** | **high** | **high** | **9/10** | matches exact boundary `keep_old = recent_window`; no code change to test |
| 2 | H_per_head | med-high | med  | 6/10 | needs offline filter inspection + optional head-masking |
| 3 | H_sink     | medium  | med  | 5/10 | needs 2-line code change (sink pinning) |
| 4 | H_numerics | low     | med  | 3/10 | fp32-storage rerun is cheap but prior is low |
| 5 | H_recent_mask | v.low | low | 1/10 | no plausible code path, not aligned with seed determinism |

**TOP-1: H_phase.** It is the only candidate that cleanly predicts the *exact*
threshold `kv = 2·recent_window`, is testable **without code change**, runs
in < 1 hr on a single GPU, and its result is diagnostic for all four other
hypotheses as a second-order bonus (H_phase-confirmed makes H_per_head /
H_sink / H_numerics / H_recent_mask unnecessary; H_phase-refuted rules in
H_per_head and promotes the offline filter-geometry audit).

## 4. Proposed single most informative experiment

**Name**: `rank1_kv256_recent_sweep_llama2` (thread B)

**Config (fixed)**: Llama-2-7B, pg19_chunks_llama2_noeos, seq_length=4096,
skip_chunks=40000 (standard eval shard), max_chunks=200, filter_rank=1,
calibration_chunks=64, sub_window_len=1024, bf16, sdpa, mode=qfilters,
seed=0, kv_budget=256. Reuse the cached `filters.pt` from the postfix
sweep (same calibration → identical filters deterministically).

**Sweep axis**: `recent_window ∈ {64, 128, 192, 256}` — 4 cells.

**Wall-time estimate**: ~20 min/cell on 1 × B200 → ~80 min total. Or
~10 min/cell on 8 × B200 DDP → ~40 min. Fits any idle slot; preferred
node b200-3 (same cluster as the seed exhaustion runs, eliminates
hardware-confound).

**Decision rule**:

| outcome | interpretation | next step |
|---------|----------------|-----------|
| recent=192 PPL ≤ 200 | **H_phase confirmed** | declare: rank=1 usable only when `recent_window ≥ kv/2`; update §11.4 op-point table, retire the kv=256 recent=64 row as pathological-by-design |
| recent=192 PPL ∈ (200, 400] | H_phase partial — subspace has some noise but phase also matters | escalate to H_per_head offline audit (#2 in ranking) |
| recent=192 PPL > 400 | **H_phase refuted** | immediately dispatch H_per_head offline: dump per-head S[0]/S[1] + filter-ablation masking |
| recent=256 PPL < 611 | pure-SWA-256 beats qfilters-at-recent=64 → filter is **net harmful** at this cell | retract any "qfilters-positive" claim at kv=256 rank=1; recommend rank≥2 as the publishable op-point |
| recent=256 PPL >> 611 | filter net helpful, SWA saturation floor is high | expected; no action beyond noting |

**No code change required.** Driver: clone `scripts/_run_llama2_rank1_verify_sweep.sh`
as `scripts/_run_llama2_rank1_kv256_recent_sweep.sh`, loop over `--recent_window`,
fix `--kv_budget 256 --filter_rank 1`, reuse `--filters_cache` to skip
re-calibration.

## 5. Artifacts

- This note: `ops/research_notes/20260426_issue110_rank1_phase_transition.md`
- Upstream: `ops/research_notes/20260426_issue110_rank1_kv256_ppl752_rootcause.md` (SVD fix), `ops/research_notes/20260426_h_rank_reg_calib_disambig.md` (rank-reg thread C), `ops/research_notes/20260426_rank1_kv_ext_preview.md` (Llama-3 rank=1 monotone descent contrast)
- Seed-exhaustion data: `outputs/rank1_verify_llama2_postfix110/qf_r1_b{192,256}_rw64_llama2/eval_results.json` on b200-3 (3 seeds each, std=0)
- Proposed driver: `scripts/_run_llama2_rank1_kv256_recent_sweep.sh` (to be created; clone of `_run_llama2_rank1_verify_sweep.sh`)
- Cached filters to reuse: `outputs/rank1_verify_llama2_postfix110/qf_r1_b256_rw64_llama2/filters.pt`

## 6. Reference to external literature

Godey et al., "Q-Filters: Leveraging QK Geometry for Efficient KV Cache
Compression", arXiv:2503.02812 (2025). The reference implementation uses
**signed** projection (`k · f`, sign-preserving) for key scoring, motivated
by the Q-cone geometry: aligned keys → high attention, anti-aligned → low
attention. Our `compression.py:59` uses `cos.abs().sum(dim=-1)` which is
sign-invariant. The asymmetry is not itself evaluated here, but it is a
candidate residual mechanism worth noting for a possible follow-up fold
after H_phase is settled. If H_phase is confirmed and the recent=192 cell
still has PPL ≫ Llama-3 rank=1 analogue (~5), the sign-abs divergence
from Godey's canonical form should be the next target.
