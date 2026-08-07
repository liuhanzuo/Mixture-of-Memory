# G1 gate result — "sampler dominates method" does NOT generalize to MBPP+

## ★ FINAL (2026-08-08 ~04:0x CST): complete 25-cell MBPP+ grid landed. **G1 FAIL, CONFIRMED.**

The 8h27m MBPP+ audit finished (agent `a06e2c22`, commit `70530326`, 25/25 cells, all
`--expected 378` shard assertions passed). MAIN recomputed every cell conjunctively from
`eval_results.json` rather than trusting the summary. Verdict is **unchanged** from the partial
estimate:

| slice | n | plus-axis spread |
|---|---:|---:|
| HumanEval+ plausible | 21 | **26.80 pt** |
| **MBPP+ plausible** | **21** | **10.58 pt** |
| MBPP+ distinct configs (seed/dup/ref replicates removed) | 13 | **10.05 pt** |

- G1 threshold 22.6 pt → **FAIL** (12 pt short)
- STOP threshold 15.3 pt → **BELOW IT** (4.7 pt short) → **the direction stops as pre-registered**
- MBPP+ max `0.6958` @ `T0.1_p0.99_entropy`; min `0.5899` @ `T0.7_seed3`

My earlier partial figure was 9.5 pt from 9 cells; the full grid gives 10.58 pt. Twelve additional
cells moved it **+1.1 pt** — nowhere near enough. The gate call made ~5 h earlier on incomplete data
was correct.

### An overclaim in the audit report, corrected

`MBPP_SAMPLER_AUDIT.md` states the `top_p` optimum "**flipped**" — HE+ favouring 0.80–0.85 and MBPP+
favouring 0.99 — and treats this as a headline ("the direction of the fix inverts"). **That
overstates what the data supports.** Measured against the audit's *own* T=0.7 seed noise floor of
2.4–2.6 pt:

| benchmark | best top_p | margin over the rival optimum | resolvable? |
|---|---|---:|---|
| HE+ | 0.80 / 0.85 (`.7439`) | **+4.88 pt** over 0.99 | **yes** — well outside the floor |
| MBPP+ | 0.99 (`.6958`) | **+2.12 pt** over 0.80 (8 of 378 items) | **no** — inside the floor |

On MBPP+, `top_p ∈ {0.80, 0.85, 0.95, 0.99}` all sit within 2.1 pt, i.e. **within seed noise**. The
honest statement is not "the optimum flipped" but "**HE+ has a resolvable top_p optimum; MBPP+ does
not.**" The only preference both benchmarks resolve is that `top_p=1.00` is worst, and that one is
comfortably outside the floor on both (HE+ −17.1 pt, MBPP+ −6.9 pt vs their respective best).

This matters because "the optimum inverts across benchmarks" would be a *stronger* protocol-
sensitivity claim than the one the data supports, and it is the kind of claim a reviewer checks.

### What genuinely replicated (and is worth keeping)

- **Noise floor, exactly**: T=0.1 ref × 4 seeds → 0.0 pt on both axes, byte-identical `0.804/0.690`;
  T=0.0 dup × 2 → 0.0 pt, byte-identical. T=0.7 × 4 seeds → 2.4/2.6 pt. Same structure as HE+.
- **The three confidence algs tie byte-identically** (`entropy` = `maskgit_plus` = `topk_margin` =
  `0.6905`), and `origin` collapses to `0.3069`. Exact replication of the HE+ finding that the
  "selection strategy" axis is worth zero once you have *any* confidence ordering.
- **Rank agreement**: ρ=0.864 (base, n=25), 0.785 (plus, n=25), but only **0.637** on the plausible
  plus-side subset — consistent with my earlier n=5 finding of ρ=0.60.

## Original entry (2026-08-08, partial grid) follows below for provenance.

---


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

## Addendum — G2 (rank transfer) and the p-value convention

G2 required Spearman ρ ≥ 0.70 **and** p < 0.05 between HE+ and MBPP+ over distinct plausible cells.
Measured over the 5 distinct plausible points (`entropy`/`maskgit_plus`/`topk_margin` at the
reference cell are byte-identical, so they are one point, not three):

- ρ = **0.6000**
- exact permutation p, **one-sided = 0.1750**; **two-sided = 0.3500** (120 permutations, n=5)

**G2 is NOT met on either convention.** State the sidedness whenever this is quoted — an unlabelled
`0.1750` reads as two-sided to a reviewer, and the two-sided figure is twice that. The same
labelling discipline applies to the McNemar values elsewhere in this direction (those are
two-sided exact binomial: dLLM p=0.3877, AR p=0.3438).
