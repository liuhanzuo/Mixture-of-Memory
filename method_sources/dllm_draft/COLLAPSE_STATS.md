# Typed-Region Collapse Frequency

The forward doctrine originally collapses a region only when every token in that
region is masked. For a region of length `n`, this introduces probability
approximately `u^n`.

The implementation now supports two modes:

1. `all_mask` — primary symmetry-preserving rule;
2. `coupled` — sample one region-level collapse event with probability
   `u^gamma`, independent of token length.

Validation used 4,541 content-phase states from the normalized eval split.

## Aggregate statement collapse rate by token length

| Length | all-mask | coupled, γ=1 |
|---|---:|---:|
| 1 | 67.8% | 69.0% |
| 2 | 41.2% | 51.5% |
| 3–4 | 33.8% | 52.0% |
| 5–8 | 30.4% | 54.4% |
| 9–16 | 27.1% | 54.9% |
| 17+ | 21.8% | 54.8% |

Aggregate rates include regions whose local clock has already reached one. The
length bias is clearer inside a fixed clock bin.

At local clock `u ∈ [0.5,0.6)`:

| Length | all-mask statement collapse |
|---|---:|
| 2 | 31.4% |
| 3–4 | 13.6% |
| 5–8 | 2.6% |
| 9–16 | 0.6% |
| 17+ | 0% |

At `u ∈ [0.8,0.9)`:

| Length | all-mask statement collapse |
|---|---:|
| 2 | 71.8% |
| 3–4 | 57.8% |
| 5–8 | 38.7% |
| 9–16 | 20.5% |
| 17+ | 3.0% |

## Decision

- First strict/local-body pilot keeps `all_mask`, because it is the literal
  forward/reverse symmetry rule in the design.
- Add a collapse-mode ablation:
  - `all_mask`;
  - `coupled, gamma=1`;
  - optionally `coupled, gamma=2` for a later/steeper collapse hazard.
- Always log collapse frequency by span length and local-clock bin.

The coupled variant is not merely an optimization: it changes the corruption
distribution and must be reported as such.

