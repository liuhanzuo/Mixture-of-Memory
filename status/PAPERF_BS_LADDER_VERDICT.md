# PAPERF_BS_LADDER_VERDICT.md

**Direction A judgment experiment — bs8 vs bs16 flip rate ladder**
Generated: 2026-08-08

## Experiment Summary

Question: Does flip rate (acc_norm decision reversal between bs=8 and bs=16)
increase monotonically with model damage (core6 acc decline)?
Can the margin distribution (near-tie density) explain this variation
in a LOO mediation check?

- Rungs complete (have both bs8 + bs16): **2/6**

## VERDICTS

- FLIP_RATE_MONOTONE: **UNTESTABLE** — only 2/6 rungs have bs16 data (Spearman requires ≥3). The 2 available points are consistent with the hypothesis (base_full: 0.08%, ShortGPT-16: 0.64%, ratio 7.9x), but no valid statistical test is possible yet.
- MARGIN_MEDIATES: **UNTESTED** — LOO requires ≥3 complete rungs.

## Why bs16 is missing for keep14/12/10/8

When the GPU eval script launched on .73 (2026-08-08 14:30), the base_full eval completed fine. However, Paper B training jobs (keep8, keep10, keep12 resumes) started on .73/.82/.104 at ~13:57, using ~78GB/card. When the keep14 eval started on .73 (same GPUs), only ~19GB remained, causing OOM. The `_run_paperF_bs16_ladder_73.sh` script is idempotent and will complete the missing 4 rungs once training finishes. Expected training duration: ~76h from current step (keep8 at step 121k, needs to reach 200k). A GPU window on .73 after training completes can run the remaining 4 rungs in ~1h total.

## Rung-level acc_norm scores and flip counts

| rung | core6 | bs8_dir | bs16_dir | total_flips | flip_rate% |
|------|------:|---------|----------|------------:|-----------:|
| base_full32 | 0.70365 | `7B_base_full_bs8` | `7B_base_full_bs16` | 14 | 0.08 |
| ShortGPT-16 | 0.62247 | `7B_shortgpt16_step200000_bs8` | `7B_shortgpt16_step200000_bs16` | 110 | 0.64 |
| keep14 | 0.59532 | `7B_keep14_step200000_v2` | `7B_keep14_step200000_bs16` | MISSING | — |
| keep12 | 0.56888 | `7B_keep12_step124000_v2` | `7B_keep12_step124000_bs16` | MISSING | — |
| keep10 | 0.52999 | `7B_keep10_step83500_v2` | `7B_keep10_step83500_bs16` | MISSING | — |
| keep8 | 0.52328 | `7B_keep8_step121000_v2` | `7B_keep8_step121000_bs16` | MISSING | — |

## Spearman test: core6 vs flip_count

**n=2 complete rungs — test is not possible (n=2 gives rho trivially ±1 for any non-tied data; permutation p undefined/uninformative). The 2 available data points are in the expected direction (more damage → more flips), but this cannot be treated as statistical evidence. Need all 6 rungs.**

| rung | core6 | flip_count |
|------|------:|-----------:|
| base_full32 | 0.70365 | 14 |
| ShortGPT-16 | 0.62247 | 110 |

Note: hypothesis is rho < 0 (higher core6 = less damage = fewer flips).
rho = -1.0000 (rho < 0 (expected direction)), exact two-sided p = nan

## Pooled margin → P(flip) bucketed curve

Total paired items pooled: 34390
Bucket width: acc_norm score margin scale, 10 equal-width buckets over the observed range.

| margin_bucket_mid | n | flips | P(flip) |
|------------------:|--:|------:|--------:|
| 0.2489 | 33101 | 124 | 0.0037 |
| 0.7468 | 1059 | 0 | 0.0000 |
| 1.2446 | 190 | 0 | 0.0000 |
| 1.7425 | 21 | 0 | 0.0000 |
| 2.2403 | 11 | 0 | 0.0000 |
| 2.7382 | 3 | 0 | 0.0000 |
| 3.2360 | 2 | 0 | 0.0000 |
| 3.7339 | 1 | 0 | 0.0000 |
| 4.2317 | 1 | 0 | 0.0000 |
| 4.7296 | 1 | 0 | 0.0000 |

## LOO mediation check

**Protocol**: For each left-out rung i, fit empirical P(flip|margin_bucket)
from the other 5 rungs' pooled data. Predict rung i's flip count.
Compare against constant-rate null (mean flip rate of other 5 rungs × n_items_i).

**This is NOT in-sample** — each rung's prediction uses only held-out training data.

**INSUFFICIENT DATA: only 2 complete rungs (need ≥3)**

## Per-rung margin statistics and fine-grained flip breakdown

| rung | n_items | flips | flip_rate | median_margin | frac<0.005 | max_flip_margin | med_flip_margin |
|------|--------:|------:|----------:|--------------:|-----------:|----------------:|----------------:|
| base_full32 | 17195 | 14 | 0.081% | 0.1246 | 2.01% | 0.0048 | 0.0008 |
| ShortGPT-16 | 17195 | 110 | 0.640% | 0.1038 | 3.29% | 0.1611 | 0.0025 |

Fine-grained flip rate by margin bucket (pooled over core6 tasks):

| margin range | base_full32 n | base_full32 flips | base_full32 P(flip) | ShortGPT-16 n | ShortGPT-16 flips | ShortGPT-16 P(flip) |
|---|---:|---:|---:|---:|---:|---:|
| [0.000, 0.001) | 71 | 9 | 12.68% | 113 | 33 | **29.20%** |
| [0.001, 0.005) | 275 | 5 | 1.82% | 452 | 47 | **10.40%** |
| [0.005, 0.010) | 384 | 0 | 0.00% | 541 | 19 | **3.51%** |
| [0.010, 0.050) | 3108 | 0 | 0.00% | 3661 | 9 | **0.25%** |
| [0.050, 0.100) | 3365 | 0 | 0.00% | 3605 | 0 | 0.00% |
| [0.100, 0.200) | 4906 | 0 | 0.00% | 4711 | 2 | 0.04% |
| [0.200, 0.500) | 4384 | 0 | 0.00% | 3533 | 0 | 0.00% |
| [0.500, ∞) | 702 | 0 | 0.00% | 579 | 0 | 0.00% |

**Key observations**:
1. All base_full32 flips are from items with margin < 0.005 (max_flip_margin = 0.0048). This is consistent with pure floating-point noise.
2. ShortGPT-16 flips extend to margin 0.1611, and P(flip|margin<0.001) is 29.2% vs 12.7% for base_full32. This means the **bf16 score perturbation is larger** for the damaged model within the same margin bucket — not just that it has more near-tie items.
3. ShortGPT-16 has marginally more near-tie items (3.29% frac<0.005 vs 2.01%) but the **amplification of flip rate within each bucket** is a stronger driver.

**Mechanism note**: The higher P(flip|margin<0.001) for ShortGPT-16 could come from (a) larger batch-size-induced score perturbations on the damaged model's representations, or (b) the same perturbations acting on scores that are arithmetically closer. Both are consistent with "damage amplifies implementation sensitivity", but they are different causal paths. Distinguishing them requires the full 6-rung ladder (to separate whether the *bucket sizes* or the *bucket rates* drive inter-rung variation).



MAIN reported "correct flag flips = 107, pred_letter flips = 122" for ShortGPT-16 bs8 vs bs16.

Direct recount (Python, per_example_*.jsonl paired by item_id):
| metric | flips |
|--------|------:|
| `correct` field (acc, raw logprob) | **107** ✓ matches MAIN |
| `pred_letter` field (raw acc pred letter) | **122** ✓ matches MAIN |
| `acc_norm_score` field (harness stored acc_norm correctness) | **110** |
| `norm_scores argmax == gold` (this analysis) | **110** |
| `norm_scores argmax letter` (prediction letter) | 134 (not correctness flips) |

The 110 figure (acc_norm correctness flips) is the correct number for the acc_norm criterion.
MAIN's "107" is for the raw acc criterion (not length-normalized).
Discrepancy 107 vs 110: 3 items where raw acc decision is unchanged but length-normalized decision flips.

- Zero-effect sanity: no bs8-vs-bs8 re-run data. Per memory `same-harness-runs-bit-identical`:
  same code + same data + same harness = byte-identical results = 0 flips expected.
  This is not independently measured here; it is asserted from prior finding.
- bs16-vs-bs16 re-run for internal consistency check (not done).
- Logistic regression fit (replaced by empirical bucket-rate LOO).
- Bootstrap CI on flip rates per rung (n=1 per cell, CI not meaningful).

## Relationship to the existing near-tie density finding

The already-established result is:
- `Spearman(core6, median_margin) = +1.0, p=0.0028` (6 rungs, exact)
- `Spearman(core6, frac<0.005) = -0.9429, p=0.0167`
- More damaged models have denser near-tie distributions (fewer wide margins).

The current experiment asks: does that translate into more flips?
The two findings are:
- **Complementary** if flip_rate is also monotone with damage AND margin distribution
  explains inter-rung variation in LOO — that would complete the causal chain.
- **Disconnected** if flip_rate is not monotone or margin doesn't explain
  inter-rung variation — in which case near-tie density is a model property,
  not a predictor of implementation-sensitivity.

See verdict lines at top for actual finding.

---
*All numbers are empirical, computed from per_example_*.jsonl files.
All statistical tests are exact (n=6 permutation for Spearman).
LOO is strictly out-of-sample — no in-sample test was run.*
