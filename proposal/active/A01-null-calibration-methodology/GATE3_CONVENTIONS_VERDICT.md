---
gate: A01 gate-3 follow-on — the content interface x five longest-option null conventions
date: 2026-08-09
compute: CPU only, ZERO GPU (re-analysis of per-item records already on disk)
closes: STATUS.json gate_results.gate3_fp32_causal_tie_test.remaining_analysis_TODO
verdict: THE NULL A01 RECOMMENDS HAS ITS OWN UNDECLARED CONVENTION DEGREE OF FREEDOM, AND IT FLIPS 5/6 ARM VERDICTS
---

# A01 gate-3 follow-on — the longest-option null is under-specified

## 0. Why this exists

`STATUS.json`'s gate-3 block carried an open item:

> `"remaining_analysis_TODO": "the six summaries also carry content_norm columns and four
> longest_option_floor_by_conv variants (split/first/last/credit/wrong) that nobody has
> tabulated. CPU-only, no GPU."`

This closes it. No new compute: `code/a01_gate3_content_conventions.py` re-reads the
six per-item dtype record sets and recomputes both sides (model accuracy and null) from
scratch, so it is also a regression test on the archived summaries.

**Regression gate: PASSED.** All five null values and all `letter_acc` / `content_norm_acc`
/ `content_raw_acc` figures reproduce
`evidence/gate3_dtype_runs/*_dtype_summary.json` to **<1e-12** on all six arms and both
dtypes.

## 1. The finding in one line

A01 tells everyone else to report against a construct-appropriate null instead of chance.
**A01's own MC content null is under-specified in the same way the chance line is
wrong-headed**: "the null is the longest option" does not say what to do when several
options tie on token count, and on MMLU **34.22%** of items have such a tie. The five
defensible readings of that one sentence put the null anywhere from **0.1961 to 0.4537**
— a spread of **25.76 pp**, which is larger than the entire distance from chance (0.25) to
the intact base model's content accuracy (0.4706).

## 2. The five conventions

Verbatim from `code/a01_gate3_fp32_vs_bf16.py::longest_floor` (and reimplemented
identically in `code/a01_gate3_content_conventions.py`). `W` = argmax-set of continuation
token counts over the available options; `g` = gold letter.

| convention | rule | reading |
|---|---|---|
| `split` | `1/|W|` if `g ∈ W` | expected accuracy of *uniform random* tie-breaking. **A01's pre-registered canonical choice.** |
| `first` | `1` iff `W[0] == g` | break ties by lowest letter index — what `argmax` actually does |
| `last` | `1` iff `W[-1] == g` | break ties by highest letter index |
| `credit` | `1` iff `g ∈ W` | **optimistic / oracle** tie-breaking |
| `wrong` | `1` iff `|W| == 1` and `W[0] == g` | **pessimistic**: any tie scores 0 |

## 3. The null is a property of MMLU **and of the tokenizer**, not of any arm

Within the six OLMo-2 arms the five null values are identical to 9 decimal places
(asserted in the script, field `null_is_dataset_property_not_arm_property = true`):

| convention | null on MMLU (n = 14,042), OLMo-2 tokenizer |
|---|---:|
| `split` | **0.284450** |
| `first` | 0.281085 |
| `last` | 0.282154 |
| `credit` | **0.453710** |
| `wrong` | **0.196126** |

> ⚠️ **But it is NOT constant across model families**, because "longest option" is measured
> in *continuation tokens* and every family tokenizes differently. From
> `evidence/a01_gate1_third_family.json`, field `longest_option_split_tie_null`:
> **Llama-2-7B `0.2757` / Qwen3-8B `0.2833` / Llama-3-8B `0.2847` / OLMo-2 `0.2845`** —
> a 0.90 pp spread on the *same benchmark items* under the *same* `split` convention.
> So a cross-family content comparison against a single shared "0.2845" is already
> mis-calibrated, independently of the tie-convention issue. This compounds §4: a
> cross-family content claim needs **per-family, per-convention** nulls.

Tie structure of the longest-option winner set (OLMo-2 tokenizer):

| `|W|` | items | share |
|---:|---:|---:|
| 1 | 9,237 | 65.78% |
| 2 | 2,174 | 15.48% |
| 3 | 754 | 5.37% |
| 4 | 1,877 | **13.37%** |

`frac_items_with_tied_longest = 0.342188`; `frac_items_gold_in_winner_set = 0.453710`
(which is by definition the `credit` null). **1,877 items (13.37%) have all four options
tied on token count**, i.e. on those items the longest-option heuristic is not a
heuristic at all — it is a coin flip among all options, and the five conventions score it
1/4, 1 or 0 depending purely on convention.

## 4. The verdict table — `credit` flips 5 of 6 arms

bf16 `content_norm`, paired bootstrap 10,000 resamples, seed 7 (same estimator and seed as
gate-3's own arm-vs-floor test). Δ in pp; every `p` here is 0.0001–0.0010, i.e. at or near
the bootstrap floor of `1/n_boot`, so the flips are not marginal.

| arm | `content_norm` | `split` | `first` | `last` | **`credit`** | `wrong` |
|---|---:|---:|---:|---:|---:|---:|
| `7B_base` | 0.4706 | +18.61 ✅ | +18.95 ✅ | +18.84 ✅ | **+1.69 ✅** | +27.45 ✅ |
| `7B_shortgpt16_step200000` | 0.4012 | +11.67 ✅ | +12.01 ✅ | +11.90 ✅ | **−5.26 ❌** | +20.50 ✅ |
| `7B_keep14_step200000` | 0.3832 | +9.88 ✅ | +10.21 ✅ | +10.11 ✅ | **−7.05 ❌** | +18.71 ✅ |
| `7B_keep12_step124000` | 0.3629 | +7.85 ✅ | +8.18 ✅ | +8.08 ✅ | **−9.08 ❌** | +16.68 ✅ |
| `7B_keep10_step83500` | 0.3445 | +6.00 ✅ | +6.34 ✅ | +6.23 ✅ | **−10.92 ❌** | +14.83 ✅ |
| `7B_keep8_step121000` | 0.3423 | +5.78 ✅ | +6.12 ✅ | +6.01 ✅ | **−11.15 ❌** | +14.61 ✅ |

✅ = "above the null", ❌ = "BELOW the null" (significantly). Verdict counts per
convention, from the script's `convention_sensitivity_bf16_content_norm` block:

| convention | above | at | below |
|---|---:|---:|---:|
| `split` | 6 | 0 | 0 |
| `first` | 6 | 0 | 0 |
| `last` | 6 | 0 | 0 |
| **`credit`** | **1** | **0** | **5** |
| `wrong` | 6 | 0 | 0 |

**Under `credit`, the only arm whose content readout beats its null is the intact base —
and it beats it by 1.69 pp.** Under `split`, all six clear it, the most damaged by 5.78 pp.
The *same measurements* support "the content interface is valid on every arm including
keep8" and "the content interface is invalid on every damaged arm" depending on a tie
convention nobody states.

## 5. Residual fractions — the quantity A01 actually reports

Residual fraction `= (reported − null) / reported`, bf16 `content_norm`:

| arm | `split` | `first` | `last` | `credit` | `wrong` |
|---|---:|---:|---:|---:|---:|
| `7B_base` | 0.3955 | 0.4027 | 0.4004 | **0.0359** | **0.5832** |
| `7B_shortgpt16_step200000` | 0.2909 | 0.2993 | 0.2966 | −0.1310 | 0.5111 |
| `7B_keep14_step200000` | 0.2577 | 0.2665 | 0.2637 | −0.1840 | 0.4882 |
| `7B_keep12_step124000` | 0.2162 | 0.2255 | 0.2225 | −0.2502 | 0.4596 |
| `7B_keep10_step83500` | 0.1742 | 0.1840 | 0.1809 | −0.3171 | 0.4306 |
| `7B_keep8_step121000` | 0.1689 | 0.1787 | 0.1756 | −0.3256 | 0.4270 |

On the intact base arm the two extreme conventions give **0.0359** vs **0.5832** — a
**16.26×** ratio, on one arm, from one undeclared convention. Even discarding `credit`
entirely (as arguably indefensible, since it grants oracle tie-breaking to an input-blind
baseline), `wrong`-vs-`split` still spans **1.47×** on base and **2.53×** on keep8.

**This is directly comparable to gate-4.** Gate-4 (`GATE4_VERDICT.md`) found the paper's
headline span moves 6.86×–10.04× under defensible C4 *aggregation* choices and concluded
the headline must be a range. The same discipline applies here, one level deeper: the
*null itself* has a convention parameter, and its effect (16.26× on one arm's residual
fraction) is larger than the aggregation effect gate-4 was worried about.

For reference, the C1 leg of the four-construct table
(`evidence/null_calibration_p1_nperm2000.json:c1_mc`, reported = `scratch16L @200k`
content_norm = 0.3597778) under each convention:

| convention | null | C1 residual | C1 residual fraction |
|---|---:|---:|---:|
| `split` (as published) | 0.284450 | +0.075328 | **+0.2094** |
| `first` | 0.281085 | +0.078692 | +0.2187 |
| `last` | 0.282154 | +0.077624 | +0.2158 |
| `credit` | 0.453710 | −0.093932 | **−0.2611** |
| `wrong` | 0.196126 | +0.163652 | **+0.4549** |

The published 0.2094 is the `split` value and is correct as published. But the C1 row's
residual fraction can be made to change **sign** by a tie convention.

## 6. `content_raw` and fp32 — no additional story

The full 120-row grid (`evidence/gate3_content_null_conventions.csv`: 6 arms × 2 dtypes ×
2 readouts × 5 conventions) shows the convention effect is the dominant axis:

* **fp32 vs bf16 changes nothing here.** `content_norm` moves ≤0.09 pp between dtypes on
  every arm (largest: `keep10` 0.3445→0.3454). No convention's verdict changes with dtype.
  This is consistent with gate-3's headline that the tie/precision story is a *letter*-side
  artifact, and it adds that the content side is precision-insensitive too.
* `content_raw` (un-normalised sum-LL) is uniformly lower than `content_norm` and is not
  the readout A01 reports; the per-row numbers are archived in the CSV for completeness.

## 7. What this changes in A01's claims

**Adds (new, and not found in the novelty check's candidate set):** the recommendation
"report against a construct-appropriate null" is incomplete. It must be
**"report against a construct-appropriate null *and print the null's convention*"**,
because on MMLU the longest-option null's tie convention alone moves the null 25.76 pp and
reverses 5/6 arm verdicts. A01 is in the strongest possible position to say this, having
just done it to itself twice (C5's headline retraction, and the `GATE1_HEALEDARMS` §7
correction).

**Does not change:** every A01 number published so far used `split`, stated it, and is
correct as published (verified <1e-12 against the archived summaries).

**Retires one item on the banned list, with a correction.**
`STATUS.json:must_not_resurrect` lists `0.2822` as a *wrong* number whose canonical
replacement is `0.2845`. That is right about which number to publish, but the reason
recorded is incomplete: **0.2822 is not an error, it is the `last`-of-maximal convention**
(0.282154, exact). It should be described as "a different, less defensible convention,
superseded by the pre-registered `split`", not as a miscomputation — and A01 should print
it in the convention table above rather than merely banning it. `evidence/C5_self_falsification.md`
§3 already says this ("last-of-maximal (the `.2822` in the initial dossier)"); the
`must_not_resurrect` entry had not caught up.

## 8. Provenance

* Script: `code/a01_gate3_content_conventions.py` (CPU only; asserts 8/8 shards and unique
  `item_id` per arm before merging)
* Outputs: `evidence/gate3_content_null_conventions.json`,
  `evidence/gate3_content_null_conventions.csv` (120 rows)
* Inputs: `results/a01_gate3/dtype_runs/{7B_base,7B_shortgpt16_step200000,
  7B_keep14_step200000,7B_keep12_step124000,7B_keep10_step83500,7B_keep8_step121000}_dtype/
  per_example_dtype_shard{0..7}of8.jsonl` — **on zwfy6 only** (`.73`/`.82`/`.104`); wzc1
  holds only the six `dtype_summary.json` under `evidence/gate3_dtype_runs/`
* Cross-check inputs: `evidence/gate3_dtype_runs/*_dtype_summary.json` (wzc1),
  `evidence/null_calibration_p1_nperm2000.json`
* Estimator: paired item-level bootstrap, `n_boot=10000`, `seed=7`, p floored at
  `1/n_boot = 0.0001` — identical to `a01_gate3_fp32_vs_bf16.py`'s arm-vs-floor test
* Cost: CPU only, ~4 min single process. No GPU touched.
