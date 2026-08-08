---
gate: A01 gate-4 (C4 aggregation pre-registration)
date: 2026-08-09
compute: CPU only, zero GPU
verdict: HEADLINE_IS_AGGREGATION_SENSITIVE -- must be reported as a range, not as "10x"
---

# A01 gate-4 — C4 aggregation pre-registration

## 1. What the gate was for

A01's `PROPOSAL.md` lists gate-4 as `C4 aggregation 预注册，不再选择性报告 10×`. The worry
was concrete: the paper's four-construct table reports a residual-fraction **span of ~10×**
(largest construct's residual fraction over smallest), and that span depends on how the C4
construct's layer-knee data is aggregated. If defensible alternative aggregations move the
span below 10, then "10×" is a selected number, not a measured one.

## 2. Result: the worry was justified

Span = C4_residual_fraction / C3_residual_fraction, computed over the archived four-leg
table. C1/C2/C3 come from `evidence/null_calibration_p1_nperm2000.json` and are FIXED:

| construct | residual fraction |
|---|---:|
| C1 (content interface vs longest-option floor) | 0.2094 |
| C2 (SQuAD EM vs majority label) | 0.2436 |
| **C3 (midband z-CKA vs layer-order null)** | **0.0769** ← the span denominator |

C4 varies by aggregation choice:

| C4 variant | C4 frac | span | ≥10×? |
|---|---:|---:|---|
| **V1 Qwen+OLMo, 3-task native, pooled** ← **PRE-REGISTERED PRIMARY** | **0.7724** | **10.04×** | yes |
| V2 Qwen+OLMo, 3-task native, per-model then avg | 0.7677 | 9.98× | **no** |
| V3 All 3 models, 3-task native, per-model avg | 0.7074 | 9.20× | no |
| V4 All 3 models, SST2 only, per-model avg | 0.6852 | 8.91× | no |
| V5 Qwen+OLMo, SST2 only, per-model avg | 0.5278 | 6.86× | no |

C4 fraction range 0.5278–0.7724, a **1.46× spread** in the construct itself.

## 3. The honest reading (and a correction to how this was first reported)

The script prints `Gate passes under ANY defensible C4 variant: YES` because V1 clears 10.04×.
**Reading that as "gate PASSES" would be exactly the selective reporting gate-4 exists to
prevent.** The correct reading:

> Only the pre-registered primary reaches 10×, and it reaches it by 0.04. The nearest
> alternative — pooling vs per-model averaging over the *same* two models and the *same*
> three tasks, a choice with no principled preference — gives 9.98×. Four of five
> defensible variants are below 10.

**Therefore the paper must not print "10×".** It must print, e.g.:

> "residual fractions span 6.9–10.0× across the four constructs depending on the C4
> aggregation convention; under the pre-registered primary (Qwen+OLMo, native 3-task mean,
> pooled) the span is 10.0×."

That sentence is defensible under every variant. "10×" is not.

## 4. The pre-registration itself

**Primary variant: V1** — Qwen+OLMo only, native knee = mean over RTE/SST2/WiC, pooled
across the two models (not per-model-then-averaged).

Rationale, recorded before looking at which variant gave the biggest span:
* **Llama-3 is excluded** because its native knees on RTE (0.9688) and WiC (0.1250) are
  degenerate — WiC at 0.125 and SST2 at 1.0 mean its 3-task native aggregate is not a
  usable null. Including it (V3) drags the aggregate in a way that reflects Llama's broken
  per-task knees, not the construct.
* **Pooled rather than per-model-avg** because with exactly two models the two are nearly
  symmetric (V1 0.7724 vs V2 0.7677, a 0.6% difference) and pooling is what
  `build_null_calibration_table.py` already does — so this is the status-quo convention,
  not a new choice made to favour the number.
* **3-task mean rather than SST2-only** because SST2-only discards two thirds of the
  measurement (and V4/V5 show it moves the answer by 1.5–3pp of C4 fraction).

Per-model inputs (all archived in the JSON):

| model | L | linear-knee frac (3-task mean) | native knee RTE / SST2 / WiC |
|---|---:|---:|---|
| Qwen3-8B | 36 | 0.3926 | 0.9444 / 0.6389 / 0.8889 |
| OLMo-2-1124-7B | 32 | 0.2854 | 1.0 / 0.75 / 0.875 |
| Meta-Llama-3-8B | 32 | 0.2688 | 0.9688 / 1.0 / **0.1250** ← degenerate |

## 5. What this does and does not settle

**Settles:** the selective-reporting objection. Every variant is now enumerated, computed,
and archived, with one pre-registered as primary and the reasons stated. A reviewer can
check any of them.

**Does not settle:** whether the four constructs are comparable enough for a "span" to be
the right summary statistic at all. The span is dominated by C3 (0.0769) being small;
whether the midband z-CKA residual fraction belongs on the same axis as an accuracy
residual fraction is a framing question this gate does not touch.

**Cannot kill A01.** gate-4 was never a kill gate — it removes a reviewer attack surface.
Its value is that it forces the headline from a point estimate to a range.

## 6. Provenance

* Script: `proposal/active/A01-null-calibration-methodology/code/a01_gate4_c4_prereg.py`
* Output: `proposal/active/A01-null-calibration-methodology/evidence/gate4_c4_prereg.json`
* Inputs: `evidence/null_calibration_p1_nperm2000.json` (C1–C3 legs, fixed),
  `proposal/shared/representation/repr_alignment_results.json` (per-model knee data)
* Cost: CPU only, seconds. No GPU touched.
