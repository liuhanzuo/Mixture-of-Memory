# G1 gate result — "sampler dominates method" does NOT generalize to MBPP+

**Date**: 2026-08-08 (CST). **Verified by**: MAIN, re-derived from `eval_results.json` on `.82`, not read off any table.
**GPU cost**: 0 (cells already on disk).

## Verdict: **G1 FAIL — and below the STOP threshold.**

The pre-registered gate (workflow `wf_51cff631`, §5) read:

> **G1 (primary, gates the whole paper).** On the completed MBPP+ plausible grid (excluding `alg=origin`
> and `alg_temp=0.5`), spanning at least T∈{0,0.1,0.2,0.4,0.7}: plus-axis spread must be **≥ 22.6 pt**.
> → If < 15.3 pt (SCOPE+D3IM): **the direction stops.**

Measured, conjunctive plus axis (`base ∧ plus`, official evalplus 0.3.1, n=378):

| benchmark | plausible cells | T coverage | plus-axis spread |
|---|---:|---|---:|
| HumanEval+ | 21 | {0, .1, .2, .4, .7} | **26.8 pt** |
| **MBPP+** | **9** | **{0, .1, .2, .4, .7}** | **9.5 pt** |

- MBPP+ max `0.6905` (`T0.1_p0.95_entropy`), min `0.5952` (`T0.7_p0.95_entropy`).
- Ratio HE+/MBPP+ = **2.82×**.
- 22.6 pt threshold → **FAIL**. 15.3 pt STOP line → **BELOW IT (9.5 pt)**.
- For contrast, including the broken regimes gives 38.4 pt — which is why the `origin`/`alg_temp=0.5`
  exclusion must be stated whenever this number is quoted.

## Caveat that keeps this from being final (stated, not buried)

MBPP+'s plausible grid has **9 cells vs HE+'s 21**. The missing cells are the `T=0.1` top_p sweep
(`p ∈ {0.80, 0.85, 0.90, 0.99, 1.00}`), and **`p=0.80` is exactly where HE+ attains its maximum**
(`0.7439`, i.e. +5.5 pt over `p=0.95`). `mbpp_T0.1_p0.80_entropy_at0` was running on `.82` when this
was written.

Best-case arithmetic, if MBPP+ `p=0.80` gains the same +5.5 pt over `p=0.95` that it did on HE+:
`0.6905 + 0.055 = 0.7455` → spread `0.7455 − 0.5952` = **~15.0 pt**.

**Still below the 15.3 STOP line, and far below the 22.6 G1 line.** To clear STOP, MBPP+ `p=0.80`
would need plus ≥ `0.7482` — a larger top_p gain than HE+ itself showed. So the gate is very likely
to fail even at grid completion, but the completing cells should be scored before the verdict is
called final.

## Consequence for the research plan

The claim *"protocol variance exceeds every published method gain"* is **not supportable as a general
claim**. It is a **HumanEval+ property**. Two honest restatements survive:

1. *Benchmark-scoped*: on HumanEval+, sampler choice spans 26.8 pt, exceeding published method gains —
   **and this does not replicate on MBPP+ (9.5 pt)**, which is itself the interesting finding: the
   benchmark, not the method, decides how much protocol matters.
2. *Instrument-floor*: within-node floor 0.00 pt, cross-architecture floor 2.44 pt (AR and dLLM alike,
   both p>0.3) — a calibration constant for anyone reporting on these benchmarks.

Restatement (1) is a **weaker but more interesting** paper than the original framing, because the
HE+/MBPP+ 2.82× discrepancy is a measured, unexplained benchmark asymmetry. It cannot be the
"sampler dominates method" headline that was planned.

## Provenance

- MBPP+ cells: `.82:/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft_104/runs/sampler_audit_mbpp/*/eval_results.json`
- HE+ cells: `wzc1:dllm_draft/runs/sampler_audit_mirror/summary.json`
- Driver + full planned cell list: `.82:.../dllm_draft_104/scripts/_run_sampler_audit_mbpp_82.sh` (lines 92–128)
- Related corrections found in the same pass: see `DLLM_RESULTS_20260807.md` retractions 8 and 9.
