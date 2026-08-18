# Independent ARR score history — Paper A

Scores use the ARR 1–5 scale. Each row records an independent review of the
corresponding frozen PDF. Where three independent reviews are available, the
reported aggregate is their arithmetic mean; individual scores are retained.

> **Calibration warning.** v1 and v3 are not a clean longitudinal experiment:
> v1 reviews used heterogeneous, shorter prompts, whereas v3 used the much
> stricter evidence/citation/novelty audit and discovered additional PIC/cache
> prior work. A lower v3 score therefore does not by itself mean the manuscript
> became worse. Starting with the next frozen version, every version uses the
> same 3-strict + 3-normal templates, so only those future means are directly
> comparable.

| Version | Frozen PDF | Reviewer | Soundness | Excitement | Overall | Confidence | Notes |
|---|---|---|---:|---:|---:|---:|---|
| v0 | `v0_original_20260803_185005.pdf` | initial audit | — | — | — | — | Pre-rewrite audit: weak-reject / borderline risk; no numeric ARR score |
| v1 | intermediate rewritten PDF | reviewer A | 3.5 | 3.0 | 3.0 | 4 | Findings; closest-baseline and integrated-serving gaps |
| v1 | intermediate rewritten PDF | reviewer B | — | — | 3.5 | — | Borderline Conference; emphasized strong matched decomposition |
| v1 | intermediate rewritten PDF | reviewer C | — | — | 3.0 | — | Findings; practical operating point not yet dominant |
| **v1 mean** | — | **3-review mean** | — | — | **3.17** | — | Range 3.0–3.5 |
| v2 | `v2_post_feedback_20260803_204059.pdf` | pending rigorous review 1 | — | — | — | — | User-specified evidence-anchored ARR review in progress |
| v2 | `v2_post_feedback_20260803_204059.pdf` | pending rigorous review 2 | — | — | — | — | To be run independently |
| v2 | `v2_post_feedback_20260803_204059.pdf` | pending rigorous review 3 | — | — | — | — | To be run independently |
| v3 | `v3_latest_20260803_204224.pdf` | rigorous review 1 | 3.0 | 3.5 | 2.0 | 4.0 | Resubmit; nearest reusable-cache baselines and integrated serving missing |
| v3 | `v3_latest_20260803_204224.pdf` | rigorous review 2 | 3.0 | 3.0 | 2.5 | 4.0 | Borderline Findings; crossover/factorization/dense details and nearest work missing |
| v3 | `v3_latest_20260803_204224.pdf` | rigorous review 3 | 3.5 | 3.5 | 3.0 | 4.0 | Findings; strongest on matched frontier and honest failure diagnosis |
| **v3 mean** | — | **3-review mean** | **3.17** | **3.33** | **2.50** | **4.00** | Range 2.0–3.0; conservative mean is Borderline Findings |
| v4 | `v4_20260803_224539.pdf` | strict reviewer 1 | 3.5 | 3.5 | 3.0 | 4.5 | Findings; protocol-complete equal-latency evidence and nearest-system comparison remain major gaps |
| v4 | `v4_20260803_224539.pdf` | strict reviewer 2 | 3.5 | 3.0 | 3.0 | 4.5 | Findings; auditability, overlap boundaries, controlled seeds, and main-paper self-containment |
| v4 | `v4_20260803_224539.pdf` | strict reviewer 3 | 3.0 | 3.0 | 2.5 | Borderline Findings; nearest systems, unified frontier, equal-latency protocol, and robustness |
| **v4 strict mean** | — | **3-review strict mean** | **3.33** | **3.17** | **2.83** | **4.33** | Overall range 2.5–3.0 |
| v4 | `v4_20260803_224539.pdf` | normal reviewer 1 | 3.5 | 3.5 | 3.5 | 4.0 | Borderline Conference; strong internal frontier, incomplete practical comparison |
| v4 | `v4_20260803_224539.pdf` | normal reviewer 2 | 4.0 | 3.5 | 3.5 | 4.0 | Borderline Conference; sound matched core, nearest-system and repair-validation gaps |
| v4 | `v4_20260803_224539.pdf` | normal reviewer 3 | 3.5 | 3.5 | 3.5 | 4.0 | Borderline Conference; equal-latency protocol and nearest-system comparison are main gaps |
| **v4 normal mean** | — | **3-review normal mean** | **3.67** | **3.50** | **3.50** | **4.00** | Overall range 3.5–3.5 |
| **v4 all-six mean** | — | **standardized 6-review mean** | **3.50** | **3.33** | **3.17** | **4.17** | Overall range 2.5–3.5; strong Findings / borderline main |
| v5 | `v5_20260804_003238.pdf` | strict reviewer 1 | 3.5 | 3.0 | 3.0 | 4.5 | Findings; protocol auditable, but dependence-aware CI, nearest baseline, and clean seeds remain |
| v5 | `v5_20260804_003238.pdf` | strict reviewer 2 | 3.5 | 2.5 | 3.0 | 4.5 | Findings; nearest baseline, heterogeneous resampling, headline seed replication, local inconsistencies |
| v5 | `v5_20260804_003238.pdf` | strict reviewer 3 | 3.0 | 3.0 | 3.0 | 4.5 | Findings; experiment-level barriers remain despite strong claim calibration |
| **v5 strict mean** | — | **3-review strict mean** | **3.33** | **2.83** | **3.00** | **4.50** | Overall range 3.0–3.0 |
| v5 | `v5_20260804_003238.pdf` | normal reviewer 1 | 4.0 | 3.0 | 3.5 | 4.5 | Borderline Conference; strong soundness, limited competitive/production evidence |
| v5 | `v5_20260804_003238.pdf` | normal reviewer 2 | 3.5 | 3.0 | 3.0 | 4.0 | Findings; closest baseline, resampling level, and audit inconsistencies |
| v5 | `v5_20260804_003238.pdf` | normal reviewer 3 | 3.5 | 3.0 | 3.0 | 4.0 | Findings; nearest comparison, clean run uncertainty, and deployment validation |
| **v5 normal mean** | — | **3-review normal mean** | **3.67** | **3.00** | **3.17** | **4.17** | Overall range 3.0–3.5 |
| **v5 all-six mean** | — | **standardized 6-review mean** | **3.50** | **2.92** | **3.08** | **4.33** | Overall range 3.0–3.5; strict floor improved, strong Findings |
| v6 | `v6_20260804_014520.pdf` | strict reviewer 1 | 3.5 | 3.0 | 3.0 | 4.0 | Findings; nearest baseline, clean run uncertainty, natural validation remain |
| v6 | `v6_20260804_014520.pdf` | strict reviewer 2 | 3.5 | 3.0 | 3.0 | 4.5 | Findings; internal two-point frontier and missing matched Write/nearest system |
| v6 | `v6_20260804_014520.pdf` | strict reviewer 3 | 3.0 | 3.0 | 3.0 | 4.5 | Findings; repaired statistics, experiment-level barriers remain |
| **v6 strict mean** | — | **3-review strict mean** | **3.33** | **3.00** | **3.00** | **4.33** | Overall range 3.0–3.0 |
| v6 | `v6_20260804_014520.pdf` | normal reviewer 1 | 3.5 | 3.0 | 3.0 | 4.5 | Findings; nearest comparison and natural repair validation absent |
| v6 | `v6_20260804_014520.pdf` | normal reviewer 2 | 3.5 | 3.5 | 3.5 | 4.0 | Borderline Conference; careful measurement, practical evidence limited |
| v6 | `v6_20260804_014520.pdf` | normal reviewer 3 | 3.5 | 3.0 | 3.0 | 4.0 | Findings; strong internal endpoint, external/generalization gaps |
| **v6 normal mean** | — | **3-review normal mean** | **3.50** | **3.17** | **3.17** | **4.17** | Overall range 3.0–3.5 |
| **v6 all-six mean** | — | **standardized 6-review mean** | **3.42** | **3.08** | **3.08** | **4.25** | Overall range 3.0–3.5; writing/statistical iteration converged |

## Trend interpretation

- v0 → v1: compliance, claim calibration, matched depth accounting, negative
  equal-latency evidence, overlap-Write analysis, and Figure 1 improved.
- v1 mean is **3.17**, i.e. between Findings and Borderline Conference.
- v2 incorporates reviewer fixes that narrow the multikey attribution, improve
  closest-work positioning, clarify timing cohorts, fix technical notation and
  chunk keys, and strengthen limitations. Its score should be compared only
  after three fresh reviews of the frozen v2 PDF.
- v4 is the first version evaluated under the fixed **3 strict + 3 normal**
  protocol. Its frozen PDF has SHA-256
  `1e593d297f3be6e986a1a002e50085b32fb59a0cc224f595bc2f233ce031f4af`.
  Only v4 and later six-review aggregates should be used for clean trend
  comparisons.
- The v4 standardized aggregate is **3.17 Overall**. Strict reviewers average
  **2.83**, while normal reviewers average **3.50**. The central manuscript is
  now credible and well controlled, but a stable main-conference assessment is
  blocked by the under-specified equal-latency headline, absent matched nearest
  reusable-context comparison, and non-identical multi-seed adapter runs.
