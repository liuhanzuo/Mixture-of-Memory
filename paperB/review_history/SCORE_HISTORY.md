# Independent ARR score history — Paper B

Scores use the ARR 1–5 scale. Each row records an independent review of the
corresponding frozen PDF. Where three independent reviews are available, the
reported aggregate is their arithmetic mean; individual scores are retained.

> **Calibration warning.** v1 and v3 are not a clean longitudinal experiment:
> v1 reviews used heterogeneous, shorter prompts, whereas v3 used the much
> stricter evidence/citation/novelty audit. The apparent drop partly reflects
> reviewer calibration and deeper prior-work/control checks rather than only
> manuscript quality. Starting with the next frozen version, every version uses
> the same 3-strict + 3-normal templates, so only those future means are directly
> comparable.

| Version | Frozen PDF | Reviewer | Soundness | Excitement | Overall | Confidence | Notes |
|---|---|---|---:|---:|---:|---:|---|
| v0 | `v0_original_20260803_185006.pdf` | initial audit | — | — | — | — | Pre-rewrite audit: major consistency/compliance risks; no numeric score |
| v1 | intermediate rewritten PDF | reviewer A | 3.0 | 2.5 | 2.5 | 4 | Borderline Findings; novelty and structural-confound concerns |
| v1 | intermediate rewritten PDF | reviewer B | — | — | 3.0 | — | Findings; useful negative/measurement study |
| v1 | intermediate rewritten PDF | reviewer C | 3.0 | 2.5 | 2.5 | 4 | Borderline Findings; recommended stricter OLMo case-study positioning |
| **v1 mean** | — | **3-review mean** | **3.0** | **2.5** | **2.67** | **4.0** | Range 2.5–3.0 |
| v2 | `v2_post_feedback_20260803_204059.pdf` | pending rigorous review 1 | — | — | — | — | User-specified evidence-anchored ARR review in progress |
| v2 | `v2_post_feedback_20260803_204059.pdf` | pending rigorous review 2 | — | — | — | — | To be run independently |
| v2 | `v2_post_feedback_20260803_204059.pdf` | pending rigorous review 3 | — | — | — | — | To be run independently |
| v3 | `v3_latest_20260803_204224.pdf` | rigorous review 1 | 3.0 | 2.5 | 2.5 | 4.0 | Borderline Findings; seed/full32/structural controls |
| v3 | `v3_latest_20260803_204224.pdf` | rigorous review 2 | 3.0 | 3.0 | 2.5 | 4.0 | Borderline Findings; interface and matched-control gaps |
| v3 | `v3_latest_20260803_204224.pdf` | rigorous review 3 | 3.75* | 2.5* | 2.0 | 4.0 | Reject; source review used a 4-point subscale for Soundness/Excitement |
| **v3 mean** | — | **3-review mean** | **3.25** | **2.67** | **2.33** | **4.00** | Overall between Resubmit and Borderline Findings |
| v4 | `v4_20260803_224552.pdf` | strict reviewer 1 | 3.0 | 2.5 | 2.5 | 4.5 | Borderline Findings; single run, unmatched horizon/controls, reproducibility, limited novelty |
| v4 | `v4_20260803_224552.pdf` | strict reviewer 2 | 3.0 | 2.5 | 2.5 | 4.5 | Borderline Findings; literal trace only, coupled operating points, in-domain measurement, poor artifact closure |
| v4 | `v4_20260803_224552.pdf` | strict reviewer 3 | 2.5 | 2.5 | 2.5 | 4.5 | Borderline Findings; 12 detailed evidence/control/reproducibility weaknesses |
| **v4 strict mean** | — | **3-review strict mean** | **2.83** | **2.50** | **2.50** | **4.50** | Overall range 2.5–2.5 |
| v4 | `v4_20260803_224552.pdf` | normal reviewer 1 | 3.5 | 3.0 | 3.0 | 4.0 | Findings; useful bounded measurement, major seed/control/horizon gaps |
| v4 | `v4_20260803_224552.pdf` | normal reviewer 2 | 3.5 | 3.0 | 3.0 | 4.0 | Findings; careful study, no seed replication and no exact reproduction |
| v4 | `v4_20260803_224552.pdf` | normal reviewer 3 | 3.5 | 3.0 | 3.0 | 4.5 | Findings; bounded proxy-validity evidence with limited controls and efficiency evidence |
| **v4 normal mean** | — | **3-review normal mean** | **3.50** | **3.00** | **3.00** | **4.17** | Overall range 3.0–3.0 |
| **v4 all-six mean** | — | **standardized 6-review mean** | **3.17** | **2.75** | **2.75** | **4.33** | Overall range 2.5–3.0; Borderline Findings / Findings |
| v5 | `v5_20260804_003250.pdf` | strict reviewer 1 | 3.0 | 2.5 | 2.5 | 4.0 | Borderline Findings; seeds, 200k control, novelty, reproduction, scope |
| v5 | `v5_20260804_003250.pdf` | strict reviewer 2 | 3.0 | 2.5 | 3.0 | 4.5 | Findings; bounded claim is credible, but core replication/control gaps remain |
| v5 | `v5_20260804_003250.pdf` | strict reviewer 3 | 3.0 | 2.5 | 3.0 | 4.5 | Findings; single-run case study with limited archival reproducibility |
| **v5 strict mean** | — | **3-review strict mean** | **3.00** | **2.50** | **2.83** | **4.33** | Overall range 2.5–3.0 |
| v5 | `v5_20260804_003250.pdf` | normal reviewer 1 | 3.5 | 3.0 | 3.0 | 4.5 | Findings; careful proxy-validity study, no run replication or long-horizon intact control |
| v5 | `v5_20260804_003250.pdf` | normal reviewer 2 | 3.5 | 3.0 | 3.0 | 4.0 | Findings; load-bearing evidence remains historical and unreproducible |
| v5 | `v5_20260804_003250.pdf` | normal reviewer 3 | 3.5 | 2.5 | 3.0 | 4.0 | Findings; certificate not operationalized and experiment-level gaps remain |
| **v5 normal mean** | — | **3-review normal mean** | **3.50** | **2.83** | **3.00** | **4.17** | Overall range 3.0–3.0 |
| **v5 all-six mean** | — | **standardized 6-review mean** | **3.25** | **2.67** | **2.92** | **4.25** | Overall range 2.5–3.0; improved, stable Findings |
| v6 | `v6_20260804_020805.pdf` | strict reviewer 1 | 3.0 | 2.5 | 3.0 | 4.0 | Findings; literal claim valid, but no seeds/full32 and artifact missing from frozen source |
| v6 | `v6_20260804_020805.pdf` | strict reviewer 2 | 2.5 | 2.5 | 2.5 | 4.0 | Borderline Findings; non-operational premise plus replication/control/artifact gaps |
| v6 | `v6_20260804_020805.pdf` | strict reviewer 3 | 3.0 | 2.5 | 3.0 | 4.0 | Findings; bounded case study, partial artifact verification |
| **v6 strict mean** | — | **3-review strict mean** | **2.83** | **2.50** | **2.83** | **4.00** | Overall range 2.5–3.0 |
| v6 | `v6_20260804_020805.pdf` | normal reviewer 1 | 3.5 | 3.0 | 3.0 | 4.0 | Findings; careful measurement, missing run and long-horizon control |
| v6 | `v6_20260804_020805.pdf` | normal reviewer 2 | 3.0 | 2.5 | 3.0 | 4.0 | Findings; premise narrow and evidence historically unreproducible |
| v6 | `v6_20260804_020805.pdf` | normal reviewer 3 | 3.5 | 3.0 | 3.0 | 4.0 | Findings; narrow claim supported, closed-book artifact/ShortGPT gaps |
| **v6 normal mean** | — | **3-review normal mean** | **3.33** | **2.83** | **3.00** | **4.00** | Overall range 3.0–3.0 |
| **v6 all-six mean** | — | **standardized 6-review mean** | **3.08** | **2.67** | **2.92** | **4.00** | Overall range 2.5–3.0; converged Findings |
| v7 | `v7_20260804_025333.pdf` | strict artifact-verification | 3.0 | 2.5* | 3.0 | 4.0 | Artifact 3.5/5: 38 files present, all payload hashes pass, six-arm MMLU and paired headline reproduce; training/PPL/closed-book derivation still not rerunnable |

\* Review 3 reported Soundness 3/4 and Excitement 2/4. For the trend table,
these are linearly mapped to the ARR 1–5 scale as 3.75 and 2.50; the original
raw scores remain in `v3_review_3_GPT56.md`.

## Trend interpretation

- v0 → v1: result ledgers, endpoint steps, paired statistics, full32 labels,
  anonymous source, Limitations placement, Figure 1, and related work improved.
- v1 mean is **2.67**, between Borderline Findings and Findings.
- v2 further narrows the novelty claim, changes policy claims to construction
  comparisons, removes pending/ongoing language, moves probes to the appendix,
  and adds in-domain-PPL/contamination limitations. It should be rescored by
  three fresh independent reviewers before comparing the trend.
- v4 is the first version evaluated under the fixed **3 strict + 3 normal**
  protocol. Its frozen PDF has SHA-256
  `d2a4e5bb9af96a8d78179ddd66bbc27db8186ed9ef348e33b4d527e7fb3e2d59`.
  Only v4 and later six-review aggregates should be used for clean trend
  comparisons.
- The v4 standardized aggregate is **2.75 Overall**. Strict reviewers are
  unanimous at **2.50**, and normal reviewers are unanimous at **3.00**.
  Writing has successfully bounded the paper as an observational measurement
  study, but stable main-conference evidence still requires training-seed
  replication, matched horizons/controls, and archival reproducibility.
- v5 and v6 both stabilize at **2.92 Overall**. v6 improves claim precision,
  Figure 1 readability, terminology, and local artifact assembly, but cannot
  repair missing training seeds, the 200k intact counterfactual, ShortGPT
  closed-book evaluation, or OOD/contamination evidence.
- v7 changes only the frozen review package. Independent artifact verification
  confirms the 38-file anonymous snapshot is present, anonymous, internally
  hash-consistent, and sufficient to recompute six-arm MMLU summaries and the
  keep14 paired headline. It remains a partial evaluation artifact rather than
  an end-to-end reproduction package; the manuscript stays at Findings level.
