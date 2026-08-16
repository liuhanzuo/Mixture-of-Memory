# paperC prose compression to the ICLR 2026 9-page main-text limit — 2026-08-17

**Status: DONE. Main text ends on page 9 (was 10). `extent` 8.996 (was 9.480).
`latexmk` rc=0, 0 LaTeX errors, 0 undefined refs, 0 undefined cites. Not one digit changed;
no hedge dropped.** See RESULT below for the three-way verification of the page boundary.

Authoritative page-budget input: `PAGE_BUDGET_STATUS_20260817.md`.
`PAGE_LIMIT_ACTION_20260816.md` is stale by 2 pages and is NOT used here.

## Goal

| | value |
|---|---|
| baseline `main_pages` | 10 |
| baseline `extent` | 9.480 |
| baseline `ref_y` | 395.9 pt on page 10 |
| baseline `pdf_sha256` | `42269aa41ccde747` |
| ICLR 2026 main-text limit | 9 pages |
| required reduction | `extent` < 9.000, i.e. **> 0.480 pages = > 310.4 pt** of body column |

`extent = ((ref_page - 1) * 646.7 + (ref_y - 85.6)) / 646.7`. Body line pitch measured
at 10.9–11.0 pt, so 310 pt ≈ **28 body lines** ≈ a 5% prose reduction. No section has to
be deleted (see the whole-section upper bounds in `PAGE_BUDGET_STATUS_20260817.md`).

## Rules this compression obeys

1. **No hedge, scope statement, or epistemic limitation is dropped.** Wording may tighten;
   the qualification carried must survive. Five sentences were named as must-survive by the
   task; all five are audited at the bottom of this file, quoted as they now read.
2. **No number changes.** Not one digit.
   `code/gate_count_claims.py` must stay `PASS: all 17 registered count claims match` rc=0,
   and `code/check_prose_vs_evidence.py` must not regress from `n_checked=93 n_ok=93
   n_mismatch=0 n_skipped=26 n_uncovered=0` rc=0.
3. **Nothing is moved to the appendix to save pages.** All four former `MAIN_TABLES` are
   already appendix-resident; there is no main-text table left to move.
4. Compression comes only from redundancy: setup repeated between sections, sentences that
   restate the sentence before them, over-long parentheticals, verbose transitions.

## Baseline gate state (measured BEFORE any edit, so regressions are attributable)

```
gate2_crossfamily_nulls.py        rc=2   PRE-EXISTING: argparse, needs xf_root/out_json positionals
gate_bh_27cell_consistency.py    rc=0
gate_build_record_matches_pdf.py rc=2   PRE-EXISTING: gate/build_record.json pins pdf_sha256
                                        ca2339b8..., main.pdf on disk is 42269aa4... The record
                                        describes an earlier build, not this one. Bytes and page
                                        count already match; only the hash disagrees.
gate_cited_paths_in_artifact.py  rc=0
gate_count_claims.py             rc=0   PASS: all 17 registered count claims match their evidence
gate_designated_denominator.py   rc=0
gate_exact_floor_tail.py         rc=0
gate_iclr_scale_legality.py      rc=0
gate_null_expectation_bound.py   rc=0
check_prose_vs_evidence.py       rc=0   n_checked=93 n_ok=93 n_mismatch=0 n_uncovered=0
```

The two rc=2 gates fail identically before and after this work; neither is caused by prose.

---

## RESULT

| | before | after |
|---|---|---|
| **`main_pages`** (page carrying `REFERENCES`) | 10 | **10, but zero main text on it** |
| **`\label{PROBE:mainend}` page** (authoritative) | 10 | **9** |
| **`extent`** | 9.480 | **8.996** |
| `ref_y` on that page | 395.9 | 82.76 (above `BODY_TOP_PT` = 85.6) |
| body blocks above the heading on the refs page | 24 rows | **0** |
| `latexmk` rc | 0 | **0** |
| LaTeX errors / undefined refs / undefined cites | 0/0/0 | **0/0/0** |
| words in §§0–7 | 5298 | 5084 (−214, −4.0%) |
| `main.pdf` | 27 pages | 26 pages |

**Main text ends on page 9.** `extent` 8.996 < 9.000 and the `REFERENCES` heading sits at
y=82.76, i.e. *above* the first body baseline (85.6) — page 10 carries no main text at all.
Verified three independent ways, all agreeing:

1. **`\label{PROBE:mainend}` inserted immediately before `\bibliography{refs}`** on a copy at
   `/tmp/pcprobe`, full `-gg` rebuild, rc=0. `main.aux` line 71:
   `\newlabel{PROBE:mainend}{{7}{9}{Regime, pooling, and literature scope}{section*.24}{}}`
   → **page 9**. This is the boundary measure the task designates as authoritative.
2. **PyMuPDF** (`code/_measure_now.py`): `main_pages=10 ref_y=82.8 extent=8.996`, and a
   separate block scan finds **0** text-column blocks above the heading on page 10.
3. **`main.log`**: `Output written on main.pdf (26 pages, ...)`, 0 lines matching `^! `.

`pdftotext`/`pdfinfo` were not used — they are not installed here, and per
`PAGE_BUDGET_STATUS_20260817.md` a previous attempt silently read a stale `/tmp` file.

## Gate results after the edits (rc captured before anything else on the line)

```
gate_count_claims.py             rc=0   PASS: all 17 registered count claims match their evidence
check_prose_vs_evidence.py       rc=0   n_checked=93 n_ok=93 n_mismatch=0 n_skipped=26 n_uncovered=0
gate_bh_27cell_consistency.py    rc=0
gate_cited_paths_in_artifact.py  rc=0
gate_designated_denominator.py   rc=0
gate_exact_floor_tail.py         rc=0
gate_iclr_scale_legality.py      rc=0
gate_null_expectation_bound.py   rc=0
validate_tex_static.py           rc=0
gate2_crossfamily_nulls.py       rc=2   UNCHANGED PRE-EXISTING (argparse: needs xf_root/out_json)
gate_build_record_matches_pdf.py rc=2   UNCHANGED PRE-EXISTING (build_record.json pins an older pdf_sha256)
```

Both non-zero codes are **byte-for-byte the same failures measured before the first edit**
(see the baseline block above) and neither is caused by prose. `gate_build_record` compares
`gate/build_record.json`'s pinned hash against `main.pdf`; it disagreed at baseline too,
because the record describes an earlier build. Refreshing that record is a separate,
deliberate act (an already-reviewed snapshot hash is provenance) and is **not** done here.

## Not one digit changed

Every numeric literal in all nine edited files is identical before and after, checked as a
sorted multiset per file (`grep -oE '[0-9]+([.,][0-9]+)*'` → md5): **9/9 identical.** The
per-file table is at the bottom. Each individual edit hunk below also carries its own
numeric-identity line.

---

## Edits

### Measurement method used to aim the edits

Word-count trimming alone does not move `extent`: a paragraph only shrinks when a trim
pushes its final line off. I therefore measured, from `main.pdf` itself via PyMuPDF, every
main-text paragraph's line count and **last-line width**. The slack (396.0 pt full column
minus last-line width) is the cost of removing one line from that paragraph. Paragraphs
whose last line is nearly full (slack < 20 pt) cost a whole line's worth of text to shrink;
paragraphs ending in a short widow (slack > 300 pt) shrink for a handful of words.

Two intermediate builds moved 0.000 pages despite real word cuts, because the cuts landed
inside paragraphs with almost no slack. Measured trajectory:

| build | extent | main_pages | words (§§0–7) | note |
|---|---|---|---|---|
| baseline | 9.480 | 10 | 5298 | |
| 1 | 9.455 | 10 | 5255 | first pass, §3 + §5 only |
| 2 | 9.231 | 10 | 5221 | abstract + intro + related + setup + discussion + limitations |
| 3 | 9.231 | 10 | 5210 | **no move** — trims fell in low-slack paragraphs |
| 4 | 9.231 | 10 | 5205 | **no move** — same reason |
| 5 | 9.231 | 10 | 5202 | **no move** |
| 6 | 9.126 | 10 | 5157 | merged the two duplicated prior-art concession paragraphs in §3 |
| 7–11 | 9.126 | 10 | 5119 | **no move** ×4 — page 10 was down to 6 rows, all in one paragraph |
| 12 | 9.091 | 10 | 5108 | 4 rows left on page 10 |
| 13–15 | 9.075 | 10 | 5094 | 3 rows left |
| **16 (final)** | **8.996** | **9 (probe)** | **5084** | **page 10 carries no main text** |

The last 0.13 pages were the hardest: by then the entire overflow was a single limitations
paragraph, and a paragraph cannot shed a fraction of a line. Four builds registered 0.000
change because the cuts, though real, were absorbed as slack. What finally moved it was
cutting the *whole-line* worth of words from that paragraph and its two neighbours at once.


---

## Every edit, verbatim before and after

60 hunks across 9 files. Each carries a mechanical numeric-identity check and a note on
which qualification the hunk had to preserve. `Edit 21` is flagged `MARKER LOST:
materiality bar` by the automatic scan — that is a **false positive** from the hunk
boundary: the phrase moved from line 46 to line 51 inside the same merged paragraph and is
verified present in `03_method.tex:51` (and unchanged in `05_analysis.tex:27`).


## `paperC/sections/00_abstract.tex`

### Edit 1 — lines 2–2  (455→439 words, -16)

**Before (verbatim):**

```latex
Multiple-choice scores are commonly interpreted against chance. We argue that this comparison is often insufficient: before an arm is compared with another arm, its score should be tested against the construct's best constant, input-blind predictor. This floor is a necessary, not sufficient, validity condition. Across ten target constructs and one negative control, the floor is not chance and, for content scoring, is under-specified by the tie convention, length unit, and tokenizer; when option count varies, even ``chance'' is ambiguous. The floor must itself be calibrated: because it is a maximum over $k$ noisy label marginals, it is upward biased even under exactly uniform labels, and the calibrating null must be realisable by the construct --- a test we initially failed on our own largest benchmark. Under a null restricted to each item's legal letters, only two of the eight letter constructs (MMLU, BoolQ, both fixed-$k$) have floors a balanced null could not produce ($p<10^{-5}$); on the remaining six the floor lies inside the estimator's own sampling noise ($p=0.083$--$0.853$), so those constructs establish that chance is the wrong \emph{reference} without establishing that the floor differs from it in magnitude. On MMLU-Pro, all 21 evaluated cells are powered at the scale of the reference effect. Among 17 designated damaged cells in four model families, the null choice alone reverses the reading: damaged arms appear above a chance line while only two clear their best-constant floor, and the honest aggregate is 15/17 at or below the floor. Held to a single evidentiary standard --- a paired bootstrap interval excluding zero on \emph{both} sides, not only the floor side --- the flip is 3/12 above chance versus 1/12 above the floor, about a third of the size the asymmetric point-estimate comparison suggests. Across five smaller benchmarks, the designated set gives 9/85 cells above their floor against 45/85 above chance, the same wrong-null flip, while a mandatory power analysis explains why most per-benchmark significance tests are inconclusive. We then adapt a complementary, arm-conditional permutation null for a different question: whether one arm's predictions contain item-level information. Its statistic is the numerator of Cohen's $\kappa$ evaluated within option-count strata, so it is exactly zero for every pure constant emitter independent of collapse letter; our contribution is the stratification and its use as a pre-comparison gate, not the statistic. We use it to re-judge 27 existing cells. The re-analysis withdraws one above-floor capability label, dissolves both below-floor competence labels, and exposes a weak intact-family anchor. Finally, full-fp32 evaluation removes every bf16 exact tie and changes 18.03\% of letter decisions without improving accuracy, ruling out a numerical-tie explanation for the measurement failure. Together, the results motivate a simple reporting rule: calibrate each reported construct to its explicit input-blind null before interpreting arm differences.
```

**After (verbatim):**

```latex
Multiple-choice scores are commonly interpreted against chance. We argue this is often insufficient: before an arm is compared with another, its score should be tested against the construct's best constant, input-blind predictor --- a necessary, not sufficient, validity condition. Across ten target constructs and one negative control, that floor is not chance and, for content scoring, is under-specified by the tie convention, length unit, and tokenizer; when option count varies, even ``chance'' is ambiguous. The floor must itself be calibrated: being a maximum over $k$ noisy label marginals, it is upward biased even under exactly uniform labels, and the calibrating null must be realisable by the construct --- a test we initially failed on our own largest benchmark. Under a null restricted to each item's legal letters, only two of the eight letter constructs (MMLU, BoolQ, both fixed-$k$) have floors a balanced null could not produce ($p<10^{-5}$); on the remaining six the floor lies inside the estimator's own sampling noise ($p=0.083$--$0.853$), so those constructs establish that chance is the wrong \emph{reference} without establishing that the floor differs from it in magnitude. On MMLU-Pro, all 21 evaluated cells are powered at the scale of the reference effect. Among 17 designated damaged cells in four model families, the null choice alone reverses the reading: damaged arms appear above a chance line while only two clear their best-constant floor, and the honest aggregate is 15/17 at or below the floor. Held to a single evidentiary standard --- a paired bootstrap interval excluding zero on \emph{both} sides, not only the floor side --- the flip is 3/12 above chance versus 1/12 above the floor, about a third of the size the asymmetric point-estimate comparison suggests. Across five smaller benchmarks the designated set gives 9/85 cells above their floor against 45/85 above chance, the same wrong-null flip, while a mandatory power analysis explains why most per-benchmark significance tests are inconclusive. We then adapt a complementary, arm-conditional permutation null for a different question: whether one arm's predictions carry item-level information. Its statistic is the numerator of Cohen's $\kappa$ within option-count strata, hence exactly zero for every pure constant emitter independent of collapse letter; our contribution is the stratification and its use as a pre-comparison gate, not the statistic. Re-judging 27 existing cells with it withdraws one above-floor capability label, dissolves both below-floor competence labels, and exposes a weak intact-family anchor. Finally, full-fp32 evaluation removes every bf16 exact tie and changes 18.03\% of letter decisions without improving accuracy, ruling out a numerical-tie explanation for the failure. Together these motivate a simple reporting rule: calibrate each reported construct to its explicit input-blind null before interpreting arm differences.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** necessary-not-sufficient asymmetry



## `paperC/sections/01_introduction.tex`

### Edit 2 — lines 3–3  (94→93 words, -1)

**Before (verbatim):**

```latex
A recent multilingual evaluation reported an awkward pattern: every tested model was above chance, yet none beat the benchmark's majority-class baseline \citep{arcon2026metalinguistic}. The authors found the flip because they happened to print both reference lines. This is not merely a curiosity about one newly constructed benchmark. A systematic construct-validity review of 445 language-model benchmarks offers 27 actionable checklist items, but none asks authors to report a null, a chance level, or a constant predictor \citep{bean2025measuring}. The missing item is operational: \emph{before comparing model arms, test whether the reported construct clears its own input-blind floor}.
```

**After (verbatim):**

```latex
A recent multilingual evaluation reported an awkward pattern: every tested model was above chance, yet none beat the benchmark's majority-class baseline \citep{arcon2026metalinguistic}. The authors found the flip because they happened to print both reference lines. This is no curiosity of one newly constructed benchmark: a systematic construct-validity review of 445 language-model benchmarks offers 27 actionable checklist items, none of which asks authors to report a null, a chance level, or a constant predictor \citep{bean2025measuring}. The missing item is operational: \emph{before comparing model arms, test whether the reported construct clears its own input-blind floor}.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 3 — lines 5–5  (83→80 words, -3)

**Before (verbatim):**

```latex
For a letter-valued multiple-choice construct, the floor is the most accurate constant letter under the benchmark's empirical gold marginal. For content likelihood, an analogous input-blind heuristic can exploit option length. These references need not equal nominal chance. More importantly, an arm can look ``above chance'' while carrying no more usable signal through the measured interface than a constant predictor. This distinction is especially consequential under structural damage, where option-token selection bias \citep{zheng2024selectors} can dominate the argmax while task-relevant content knowledge remains partly available.
```

**After (verbatim):**

```latex
For a letter-valued multiple-choice construct, the floor is the most accurate constant letter under the benchmark's empirical gold marginal; for content likelihood, an analogous input-blind heuristic can exploit option length. These references need not equal nominal chance, and more importantly an arm can look ``above chance'' while carrying no more usable signal through the measured interface than a constant predictor --- especially under structural damage, where option-token selection bias \citep{zheng2024selectors} can dominate the argmax while content knowledge remains partly available.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 4 — lines 7–7  (73→64 words, -9)

**Before (verbatim):**

```latex
We study null calibration as a \emph{measurement protocol}, not as a new multiple-choice interface or a new debiasing algorithm. Majority-class baselines for MCQA already exist \citep{balepur2024artifacts}; letter-versus-content formulations are established \citep{gu2025olmes,cho2026choices}; and prior work has shown that constant outputs can game automatic judges \citep{zheng2025cheating}. Our contribution is to make the construct-appropriate input-blind null a pre-comparison gate, quantify the degrees of freedom hidden inside that null, and separate two questions that require different references:
```

**After (verbatim):**

```latex
We study null calibration as a \emph{measurement protocol}, not a new multiple-choice interface or debiasing algorithm. Majority-class baselines for MCQA already exist \citep{balepur2024artifacts}; letter-versus-content formulations are established \citep{gu2025olmes,cho2026choices}; constant outputs are known to game automatic judges \citep{zheng2025cheating}. Our contribution is to make the construct-appropriate input-blind null a pre-comparison gate, quantify the degrees of freedom hidden inside it, and separate two questions needing different references:
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 5 — lines 14–14  (52→50 words, -2)

**Before (verbatim):**

```latex
The distinction prevents two opposite mistakes. Chance can credit a literal constant emitter as competent; conversely, the best-constant floor can label two equally empty emitters differently solely because they collapse onto different letters. The first problem motivates our headline floor. The second motivates read-out v2, for which we prove constant-collapse invariance algebraically.
```

**After (verbatim):**

```latex
The distinction prevents two opposite mistakes: chance can credit a literal constant emitter as competent, while the best-constant floor can label two equally empty emitters differently solely for collapsing onto different letters. The first motivates our headline floor; the second motivates read-out v2, for which we prove constant-collapse invariance algebraically.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 6 — lines 16–16  (29→26 words, -3)

**Before (verbatim):**

```latex
Our empirical evidence uses ten target constructs plus Winogrande as a negative control, four base-model families, letter and content likelihood interfaces, and structurally damaged arms. The main findings are:
```

**After (verbatim):**

```latex
Our evidence uses ten target constructs plus Winogrande as a negative control, four base-model families, letter and content likelihood interfaces, and structurally damaged arms. Our findings:
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 7 — lines 19–23  (373→363 words, -10)

**Before (verbatim):**

```latex
    \item The null is not ``chance,'' and the floor is itself an estimate that needs its own calibration against a null the construct can \emph{realise} --- a test we initially failed on our largest benchmark, where uniform-over-ten-letters assigns gold letters 17\% of MMLU-Pro items cannot have. BoolQ's best constant is 0.6217, not 0.50; MMLU-Pro's floor is 0.116606, or 1.1661$\times$ naive ten-way chance. We report both gaps and ratios because either alone distorts comparisons across option counts. Against a null restricted to each item's own legal letters (Table~\ref{tab:nulls}), only MMLU and BoolQ have floors it could not plausibly produce ($p<10^{-5}$), and both have a genuinely fixed option count; the six others, MMLU-Pro included, lie inside the estimator's sampling noise ($p=0.083$--$0.853$). We therefore rest the quantitative claim on MMLU and BoolQ, reading the rest as showing only that chance is the wrong reference to \emph{report}, not that the floor is far from it. MMLU-Pro at $p=0.083$ is unresolved, not settled; Section~\ref{sec:legality-aware-null} says which rows moved and how.
    \item Content floors are not dataset constants unless their tie convention, length unit, and tokenizer are fixed. On MMLU-Pro, one tokenizer and one dataset yield floors from 0.125914 to 0.532164 across tie conventions; the latter exceeds the intact model's content score by 32.5 points.
    \item At MMLU-scale power, 17 designated damaged cells across four families exhibit wrong-null readings: 10/12 non-OLMo cells exceed item-averaged chance (12/12 exceed naive 0.10), yet only 1/12 clears the floor. The honest statement is 15/17 at or below the arm-independent floor, with two exceptions that differ in kind --- \texttt{qwen3\_8b\_base/k14} is statistically real but materially negligible, while OLMo-2 \texttt{shortgpt16} retains 16 of 32 layers and clears its floor by $5.2\times$ its own half-width. Off MMLU the same set gives 9/85 above floor against 45/85 above chance, and all nine are healed arms retaining at least 14 layers; the accompanying power table is essential.
    \item The permutation null makes every pure constant emitter score exactly zero, re-sorts rather than merely shrinks the 27-cell read-out, and reveals that an intact Llama-2-7B is itself a weak anchor on MMLU-Pro.
    \item Full-fp32 evaluation removes 100\% of bf16 exact ties and changes 2,532 letter decisions, but does not recover accuracy. The failure is therefore not repaired by numerical precision.
```

**After (verbatim):**

```latex
    \item The null is not ``chance,'' and the floor is itself an estimate needing its own calibration against a null the construct can \emph{realise} --- a test we initially failed on our largest benchmark, where uniform-over-ten-letters assigns gold letters 17\% of MMLU-Pro items cannot have. BoolQ's best constant is 0.6217, not 0.50; MMLU-Pro's floor is 0.116606, or 1.1661$\times$ naive ten-way chance. We report both gaps and ratios because either alone distorts comparisons across option counts. Against a null restricted to each item's own legal letters (Table~\ref{tab:nulls}), only MMLU and BoolQ have floors it could not plausibly produce ($p<10^{-5}$), and both have a genuinely fixed option count; the six others, MMLU-Pro included, lie inside the estimator's sampling noise ($p=0.083$--$0.853$). We therefore rest the quantitative claim on MMLU and BoolQ, reading the rest as showing only that chance is the wrong reference to \emph{report}, not that the floor is far from it. MMLU-Pro at $p=0.083$ is unresolved, not settled; Section~\ref{sec:legality-aware-null} says which rows moved and how.
    \item Content floors are not dataset constants unless their tie convention, length unit, and tokenizer are fixed: on MMLU-Pro, one tokenizer and one dataset yield floors from 0.125914 to 0.532164 across tie conventions, the latter exceeding the intact model's content score by 32.5 points.
    \item At MMLU-scale power, 17 designated damaged cells across four families exhibit wrong-null readings: 10/12 non-OLMo cells exceed item-averaged chance (12/12 exceed naive 0.10), yet only 1/12 clears the floor. The honest statement is 15/17 at or below the arm-independent floor, the two exceptions differing in kind --- \texttt{qwen3\_8b\_base/k14} is statistically real but materially negligible, while OLMo-2 \texttt{shortgpt16} retains 16 of 32 layers and clears its floor by $5.2\times$ its own half-width. Off MMLU the same set gives 9/85 above floor against 45/85 above chance, all nine healed arms retaining at least 14 layers; the accompanying power table is essential.
    \item The permutation null makes every pure constant emitter score exactly zero, re-sorts rather than shrinks the 27-cell read-out, and reveals intact Llama-2-7B as itself a weak MMLU-Pro anchor.
    \item Full-fp32 evaluation removes 100\% of bf16 exact ties and changes 2,532 letter decisions without recovering accuracy, so the failure is not repaired by numerical precision.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** MMLU-Pro unresolved; reference-not-magnitude scoping

### Edit 8 — lines 26–26  (53→53 words, +0)

**Before (verbatim):**

```latex
The floor test is deliberately one-directional. A failure disqualifies the score as evidence for an arm comparison, but a pass does not certify that the benchmark measures the intended capability. \citet{feng2019misleading} construct a dataset where a partial-input baseline is at chance while artifacts remain exploitable; thus clearing a null is \emph{necessary, not sufficient}.
```

**After (verbatim):**

```latex
The floor test is deliberately one-directional: a failure disqualifies the score as evidence for an arm comparison, but a pass does not certify that the benchmark measures the intended capability. \citet{feng2019misleading} construct a dataset where a partial-input baseline is at chance while artifacts remain exploitable; clearing a null is thus \emph{necessary, not sufficient}.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** necessary-not-sufficient asymmetry; one-directional test



## `paperC/sections/02_related.tex`

### Edit 9 — lines 4–4  (116→107 words, -9)

**Before (verbatim):**

```latex
\citet{balepur2024artifacts} define a majority-class baseline for letter MCQA and recommend stronger-than-chance references; their object is dataset cheatability under a black-box generative setup, whereas ours is a per-arm gate for likelihood-scored constructs. The distinction is operational: their invalid-output analysis imputes either 0.0 or 0.25 on MMLU, while the construct's best-constant letter value is 0.2689. \citet{zheng2025cheating} use the term ``null model'' for crafted constant responses that adversarially exploit an LLM judge. Their null is an attack optimized against a judge; ours is a reference derived from benchmark statistics and requires no attacker. The metalinguistic preprint that motivates our opening reports the above-chance/below-majority pattern descriptively on one new benchmark, but does not turn it into a pre-comparison protocol \citep{arcon2026metalinguistic}.
```

**After (verbatim):**

```latex
\citet{balepur2024artifacts} define a majority-class baseline for letter MCQA and recommend stronger-than-chance references; their object is dataset cheatability under a black-box generative setup, ours a per-arm gate for likelihood-scored constructs. The distinction is operational: their invalid-output analysis imputes either 0.0 or 0.25 on MMLU, while the construct's best-constant letter value is 0.2689. \citet{zheng2025cheating} use ``null model'' for crafted constant responses that adversarially exploit an LLM judge --- an attack optimized against a judge, whereas ours is a reference derived from benchmark statistics, requiring no attacker. The metalinguistic preprint motivating our opening reports the above-chance/below-majority pattern descriptively on one new benchmark, without turning it into a pre-comparison protocol \citep{arcon2026metalinguistic}.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 10 — lines 7–7  (88→88 words, +0)

**Before (verbatim):**

```latex
Letter preference is not a new observation. \citet{zheng2024selectors} identify selection and token bias and propose PriDe, which estimates an option-ID prior from permuted contents. PriDe fixes the predictor; null calibration fixes the reference line. Even after prediction debiasing, reporting only against chance can leave the benchmark's gold-marginal floor unreported. We therefore do not reject permutation controls on cost grounds: they are a useful and potentially inexpensive diagnostic. They answer whether predictions should be corrected, while our arm-independent floor answers whether the resulting interface score is comparable across arms.
```

**After (verbatim):**

```latex
Letter preference is not a new observation. \citet{zheng2024selectors} identify selection and token bias and propose PriDe, which estimates an option-ID prior from permuted contents. PriDe fixes the predictor; null calibration fixes the reference line. Even after prediction debiasing, reporting only against chance can leave the benchmark's gold-marginal floor unreported. We therefore do not reject permutation controls on cost grounds --- they are a useful and potentially inexpensive diagnostic, answering whether predictions should be corrected, while our arm-independent floor answers whether the resulting interface score is comparable across arms.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 11 — lines 9–9  (111→108 words, -3)

**Before (verbatim):**

```latex
OLMES standardizes both multiple-choice formulation (MCF) and cloze formulation (CF), then uses the better score per task and model \citep{gu2025olmes}. This is not a size-keyed rule. Its reference discussion is nevertheless framed around a random baseline rather than a measured label-marginal floor, so the max rule can select an interface without testing whether that interface clears an input-blind reference. Our ARC-Easy result shows a case where CF rescues an at-floor letter read-out; MMLU shows that an interface swap need not do so. \citet{cho2026choices} independently study symbols, cloze, and hybrid formats and propose a new question-sensitive score. We do not claim the interface contrast; our method instead gates whichever construct is reported.
```

**After (verbatim):**

```latex
OLMES standardizes both multiple-choice formulation (MCF) and cloze formulation (CF), then uses the better score per task and model \citep{gu2025olmes}; this is not a size-keyed rule. Its reference discussion is nevertheless framed around a random baseline rather than a measured label-marginal floor, so the max rule can select an interface without testing whether that interface clears an input-blind reference. Our ARC-Easy result shows CF rescuing an at-floor letter read-out; MMLU shows that an interface swap need not do so. \citet{cho2026choices} independently study symbols, cloze, and hybrid formats and propose a new question-sensitive score. We do not claim the interface contrast; our method instead gates whichever construct is reported.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 12 — lines 12–12  (80→76 words, -4)

**Before (verbatim):**

```latex
Length-normalized likelihood can over-correct and its token-based form is tokenizer-dependent \citep{oostermeijer2026length}. We therefore do not claim generic length sensitivity or tokenizer dependence of the scoring rule. We show that the \emph{induced input-blind floor} inherits these choices: its numerical value changes with tie convention, character versus continuation-token length, and tokenizer. OLMES's observation that tokenizer dependence may be irrelevant when ranking options for one fixed model and tokenizer is compatible with our result; our scope is cross-model comparison of the reference itself.
```

**After (verbatim):**

```latex
Length-normalized likelihood can over-correct and its token-based form is tokenizer-dependent \citep{oostermeijer2026length}, so we claim no generic length sensitivity or tokenizer dependence of the scoring rule. We show the \emph{induced input-blind floor} inherits these choices: its value changes with tie convention, character versus continuation-token length, and tokenizer. OLMES's observation that tokenizer dependence may be irrelevant when ranking options for one fixed model and tokenizer is compatible with ours; our scope is cross-model comparison of the reference itself.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 13 — lines 15–15  (57→53 words, -4)

**Before (verbatim):**

```latex
Our arm-conditional read-out v2 is a chance-corrected agreement statistic: $\Delta_{\mathrm{perm}}$ equals the numerator of Cohen's $\kappa$ \citep{cohen1960kappa} computed within option-count strata, and the constant-emitter zero we rely on is that literature's standard property rather than a new result. We claim only the stratification by variable option count and the use of the statistic as a pre-comparison gate.
```

**After (verbatim):**

```latex
Our arm-conditional read-out v2 is a chance-corrected agreement statistic: $\Delta_{\mathrm{perm}}$ equals the numerator of Cohen's $\kappa$ \citep{cohen1960kappa} within option-count strata, and the constant-emitter zero we rely on is that literature's standard property, not a new result. We claim only the stratification by variable option count and the statistic's use as a pre-comparison gate.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 14 — lines 17–17  (73→72 words, -1)

**Before (verbatim):**

```latex
Control tasks and selectivity can reverse conclusions about which representation layer is informative \citep{hewitt2019probes}. That is the closest structural precedent: their null randomizes supervision for a probe, while ours removes input dependence from an MC construct. \citet{ding2021grounding} similarly motivate sensitivity and specificity tests for representation similarity. Finally, \citet{bean2025measuring} provide the construct-validity vocabulary and a broad benchmark checklist. We position null calibration as the missing operational item, not as the invention of construct validity.
```

**After (verbatim):**

```latex
Control tasks and selectivity can reverse conclusions about which representation layer is informative \citep{hewitt2019probes} --- the closest structural precedent, their null randomizing supervision for a probe while ours removes input dependence from an MC construct. \citet{ding2021grounding} similarly motivate sensitivity and specificity tests for representation similarity. Finally, \citet{bean2025measuring} provide the construct-validity vocabulary and a broad benchmark checklist. We position null calibration as the missing operational item, not as the invention of construct validity.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)



## `paperC/sections/03_method.tex`

### Edit 15 — lines 3–3  (20→19 words, -1)

**Before (verbatim):**

```latex
Let a benchmark contain items $i=1,\ldots,n$, gold answers $y_{i}$, and a scored interface producing predictions $\hat{y}_{i}$. Its empirical accuracy is
```

**After (verbatim):**

```latex
Let a benchmark have items $i=1,\ldots,n$, gold answers $y_{i}$, and a scored interface producing predictions $\hat{y}_{i}$, with empirical accuracy
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 16 — lines 14–14  (39→40 words, +1)

**Before (verbatim):**

```latex
The v1 statistic is $\Delta_{\mathrm{floor}}=\mathrm{acc}-f_{\mathrm{const}}$. Because $f_{\mathrm{const}}$ depends only on the evaluated item set, every arm is compared to the same reference. This arm independence is essential: otherwise arm-to-arm comparisons can change because the null changes with the arm.
```

**After (verbatim):**

```latex
The v1 statistic is $\Delta_{\mathrm{floor}}=\mathrm{acc}-f_{\mathrm{const}}$. Because $f_{\mathrm{const}}$ depends only on the evaluated item set, every arm is compared to the same reference --- arm independence is essential, or arm-to-arm comparisons could change merely because the null changed with the arm.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 17 — lines 16–16  (49→44 words, -5)

**Before (verbatim):**

```latex
For content likelihood, we use an input-blind longest-option family. Its prediction is determined solely by option length, but the family is not fully specified until a tie convention and length unit are fixed; token-based variants additionally require a tokenizer. Section~\ref{sec:nulls} treats these choices as part of the construct definition.
```

**After (verbatim):**

```latex
For content likelihood, we use an input-blind longest-option family, whose prediction depends solely on option length but is not fully specified until a tie convention and length unit are fixed; token-based variants also require a tokenizer. Section~\ref{sec:nulls} treats these as part of the construct.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 18 — lines 18–18  (52→38 words, -14)

**Before (verbatim):**

```latex
Our decision rule is intentionally asymmetric. An arm that fails to beat the floor cannot support a claim that the interface expresses arm-specific capability above the strongest constant reference. An arm that clears it has passed only a necessary test. It may still exploit other artifacts that no single null enumerates \citep{feng2019misleading}.
```

**After (verbatim):**

```latex
Our decision rule is intentionally asymmetric: failing the floor disqualifies any claim that the interface expresses arm-specific capability above the strongest constant reference, while clearing it passes only a necessary test, leaving artifacts no single null enumerates \citep{feng2019misleading}.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 19 — lines 23–23  (67→66 words, -1)

**Before (verbatim):**

```latex
The v1 floor asks whether an interface is suitable for mutually comparable arm scores. It is not collapse-invariant as a competence statistic: on the full MMLU-Pro item set, always-A, always-E, and always-J contain the same item-level information---none---yet v1 assigns 0.000, $-2.111$, and $-3.807$ percentage points respectively because the gold marginals are non-uniform. The separate ten-letter implementation self-test below conditions each letter on items where it is legal.
```

**After (verbatim):**

```latex
The v1 floor asks whether an interface is suitable for mutually comparable arm scores; it is not collapse-invariant as a competence statistic. On the full MMLU-Pro item set, always-A, always-E, and always-J contain the same item-level information---none---yet v1 assigns 0.000, $-2.111$, and $-3.807$ percentage points, because the gold marginals are non-uniform. The separate ten-letter implementation self-test below conditions each letter on items where it is legal.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 20 — lines 25–25  (46→43 words, -3)

**Before (verbatim):**

```latex
For the arm-specific question, we preserve the arm's own prediction multiset and remove its alignment with items. MMLU-Pro has variable option count, so permutations are uniform \emph{within} \texttt{n\_opt} strata $s$. If $n_{s}$ is the stratum size and $c^{\mathrm{pred}}_{s,L}$ and $c^{\mathrm{gold}}_{s,L}$ are prediction and gold counts, then
```

**After (verbatim):**

```latex
For the arm-specific question, we preserve the arm's own prediction multiset and remove its alignment with items. MMLU-Pro has variable option count, so permutations are uniform \emph{within} \texttt{n\_opt} strata $s$. With $n_{s}$ the stratum size and $c^{\mathrm{pred}}_{s,L}$, $c^{\mathrm{gold}}_{s,L}$ the prediction and gold counts,
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 21 — lines 36–51  (180→88 words, -92)

**Before (verbatim):**

```latex
\paragraph{The statistic is not new; the stratification and its use are.}
$\Delta_{\mathrm{perm}}$ is the \emph{numerator} of Cohen's $\kappa$ evaluated within
\texttt{n\_opt} strata: writing $p_o=\mathrm{acc}$ and $p_e=\widehat{\mathrm{acc}}$, one has
$\kappa=(p_o-p_e)/(1-p_e)$ and hence $\Delta_{\mathrm{perm}}=\kappa\,(1-p_e)$ identically.
We verified the identity on all 27 cells of Table~\ref{tab:v2-full}; it holds to the printing
precision in every row. We therefore claim neither this statistic nor the collapse identity
below as new: that chance-corrected agreement vanishes for a constant rater is the textbook
property, since $p_o=p_e$ there. Our contribution is (i) stratifying the permutation within
variable option count, so letters that are illegal for an item are never credited, and (ii)
using the quantity as an \emph{arm-conditional pre-comparison gate} with an explicit
materiality bar rather than as an agreement score. We report $\Delta_{\mathrm{perm}}$ rather
than $\kappa$ to keep it on the same percentage-point scale as the v1 floor, which is what
makes the two nulls directly comparable (Table~\ref{tab:two-nulls}).

\paragraph{Option-count-aware chance correction is decades old; only the varying-$k$ stratification is ours.}
That a chance term should depend on how many options an item offers long predates this work
```

**After (verbatim):**

```latex
\paragraph{The statistic is decades old; only the varying-$k$ stratification and its use are ours.}
$\Delta_{\mathrm{perm}}$ is the \emph{numerator} of Cohen's $\kappa$ within \texttt{n\_opt}
strata: with $p_o=\mathrm{acc}$ and $p_e=\widehat{\mathrm{acc}}$, $\kappa=(p_o-p_e)/(1-p_e)$,
hence $\Delta_{\mathrm{perm}}=\kappa\,(1-p_e)$ identically --- verified to printing precision in
all 27 cells of Table~\ref{tab:v2-full}. We claim neither this statistic nor the collapse
identity below as new: that chance-corrected agreement vanishes for a constant rater is the
textbook property, since $p_o=p_e$ there. That a chance term should depend on how many options
an item offers likewise long predates this work
```

- **Numbers in this hunk:** CHANGED ['1', '1', '1', '2', '27'] -> ['1', '1', '2', '27']
- **Qualification preserved:** novelty concession
- **!!! MARKER LOST:** materiality bar — verify by hand

### Edit 22 — lines 53–59  (89→138 words, +49)

**Before (verbatim):**

```latex
devries2008pooledkappa}, and we claim none of it: not correcting for the number of options, not
choosing $p_e$ as a methodological decision, not noticing that $\kappa$-family statistics depend on
$k$. Those works assume a single global $k$; what we are not aware of there is a null in which $k$
\emph{varies item to item} ($\texttt{n\_opt}\in\{3,\dots,10\}$, eight strata here), the permutation
confined within strata so a letter illegal for an item can never be credited to it.
Appendix~\ref{app:priorart} states that boundary claim by claim; its footprint is the $36$ items
below.
```

**After (verbatim):**

```latex
devries2008pooledkappa}, and we claim none of it: neither correcting for option count, nor treating
$p_e$ as a methodological decision, nor noting that $\kappa$-family statistics depend on $k$. Those
works assume a single global $k$; what we are not aware of there, and do claim, is (i) a null in
which $k$ \emph{varies item to item} ($\texttt{n\_opt}\in\{3,\dots,10\}$, eight strata here), the
permutation confined within strata so a letter illegal for an item can never be credited to it, and
(ii) use of the quantity as an \emph{arm-conditional pre-comparison gate} with an explicit
materiality bar rather than as an agreement score. We report $\Delta_{\mathrm{perm}}$ rather than
$\kappa$ to keep it on the v1 floor's percentage-point scale, which is what makes the two nulls
directly comparable (Table~\ref{tab:two-nulls}). Appendix~\ref{app:priorart} states that boundary
claim by claim; its footprint is the $36$ items below.
```

- **Numbers in this hunk:** CHANGED ['10', '2008', '3', '36'] -> ['1', '10', '2008', '3', '36']
- **Qualification preserved:** prior-art concession

### Edit 23 — lines 71–71  (48→48 words, +0)

**Before (verbatim):**

```latex
This is an algebraic identity for every legal collapse letter, not an empirical regularity, and it is the $\kappa=0$ property specialised to our stratified estimator. The implementation asserts zero to below $10^{-12}$ for all ten letters. The loophole ``collapse onto a different letter'' is therefore closed by construction.
```

**After (verbatim):**

```latex
This is an algebraic identity for every legal collapse letter, not an empirical regularity: it is the $\kappa=0$ property specialised to our stratified estimator, and the implementation asserts zero to below $10^{-12}$ for all ten letters. The loophole ``collapse onto a different letter'' is therefore closed by construction.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 24 — lines 73–73  (129→129 words, +0)

**Before (verbatim):**

```latex
The two nulls must not be conflated, and their ordering is weaker than it first appears. Unstratified, $\sum_L p_{L}m_{L}\leq\max_Lm_{L}$ places the v2 null at or below the v1 best-constant floor; our within-stratum permutation only gives $\widehat{\mathrm{acc}}\leq\sum_s w_s\max_L m_{s,L}$, whose right-hand side \emph{dominates} $f_{\mathrm{const}}$ --- on MMLU-Pro by exactly $36$ items, or $0.299$ pp, because the per-stratum argmax is not always-A. The ordering holds empirically in all 27 cells of Table~\ref{tab:v2-full} but becomes a theorem only under a regularity condition we can state and not assume, and an \texttt{n\_opt}-conditional emitter legal on every item attains $f_{\mathrm{const}}+0.299$ pp. Appendix~\ref{app:ordering} gives the per-stratum argmaxes, why the 27-cell count overstates the evidence, the tightest cell's attainable violation ($0.085$ pp), and the condition. \textbf{Passing v2 must never be described as clearing the paper's floor.}
```

**After (verbatim):**

```latex
The two nulls must not be conflated, and their ordering is weaker than it first appears. Unstratified, $\sum_L p_{L}m_{L}\leq\max_Lm_{L}$ places the v2 null at or below the v1 best-constant floor; our within-stratum permutation only gives $\widehat{\mathrm{acc}}\leq\sum_s w_s\max_L m_{s,L}$, whose right-hand side \emph{dominates} $f_{\mathrm{const}}$ --- on MMLU-Pro by exactly $36$ items, or $0.299$ pp, since the per-stratum argmax is not always-A. The ordering holds empirically in all 27 cells of Table~\ref{tab:v2-full} but becomes a theorem only under a regularity condition we can state and not assume, and an \texttt{n\_opt}-conditional emitter legal on every item attains $f_{\mathrm{const}}+0.299$ pp. Appendix~\ref{app:ordering} gives the per-stratum argmaxes, why the 27-cell count overstates the evidence, the tightest cell's attainable violation ($0.085$ pp), and the condition. \textbf{Passing v2 must never be described as clearing the paper's floor.}
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** unproved-ordering caveat

### Edit 25 — lines 75–75  (38→34 words, -4)

**Before (verbatim):**

```latex
We estimate uncertainty with 10,000 paired bootstrap resamples, recomputing $\widehat{\mathrm{acc}}$ inside every resample, and verify the decision with 10,000 within-stratum permutations; both use seed 7 and must agree at $\alpha=0.05$. Significance is not enough at $n=12032$. We define
```

**After (verbatim):**

```latex
Uncertainty uses 10,000 paired bootstrap resamples with $\widehat{\mathrm{acc}}$ recomputed inside each, cross-checked against 10,000 within-stratum permutations; both use seed 7 and must agree at $\alpha=0.05$. Significance is not enough at $n=12032$, so we define
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 26 — lines 79–79  (38→37 words, -1)

**Before (verbatim):**

```latex
where $\Delta_{\max}$ is the best alignment achievable by reassigning the observed prediction multiset, and require at least 0.10 times the same-family intact anchor for a material signal. If the intact anchor itself has \texttt{recovery\_fraction}$<0.10$, relative claims are blocked.
```

**After (verbatim):**

```latex
with $\Delta_{\max}$ the best alignment achievable by reassigning the observed prediction multiset, and require at least 0.10 times the same-family intact anchor for a material signal. If the intact anchor itself has \texttt{recovery\_fraction}$<0.10$, relative claims are blocked.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** relative-claim block

### Edit 27 — lines 82–82  (69→67 words, -2)

**Before (verbatim):**

```latex
The v2 rule, gates, stratification, and materiality constant were committed before re-judging the existing cells and before future unscored milestones. The 27 numbers reported here are nevertheless post-hoc by construction because those cells had already been scored, and the designer had already seen the shape of the collapse defect. We therefore present v2 as a pre-registered rule for future read-outs and a transparent post-hoc re-analysis of the existing cells.
```

**After (verbatim):**

```latex
The v2 rule, gates, stratification, and materiality constant were committed before re-judging the existing cells and before future unscored milestones. The 27 numbers here are nevertheless post-hoc by construction: those cells had already been scored, and the designer had already seen the shape of the collapse defect. We therefore present v2 as a pre-registered rule for future read-outs and a transparent post-hoc re-analysis of the existing cells.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** post-hoc disclosure



## `paperC/sections/03b_nulls_summary.tex`

### Edit 28 — lines 4–4  (245→239 words, -6)

**Before (verbatim):**

```latex
A content floor is not a dataset constant. Four choices usually left implicit each move it by more than the effects under study, so all four are part of the measured construct and must be printed with the score. \emph{(i) The tie convention:} on MMLU-Pro, one dataset and one tokenizer give 0.125914 under \texttt{wrong} and 0.532164 under \texttt{credit}, a 40.6-point span whose upper end exceeds the intact base model's own \texttt{content\_norm} score by 32.5 points, and on ARC-Challenge \texttt{credit} alone moves five of six OLMo-2 arms from significantly above the floor to significantly below it. \emph{(ii) The length unit}, characters or continuation tokens: OpenBookQA \texttt{credit} is 0.416 versus 0.644, ARC-Challenge 18.60\% tied-longest under characters against 50.85\% under tokens. \emph{(iii) The tokenizer}, which moves \texttt{credit} by up to 10.6 points on four-way tasks and 9.26 on MMLU-Pro, non-monotonically in vocabulary size \citep{oostermeijer2026length}. \emph{(iv) The meaning of ``chance''} when \texttt{n\_opt} runs from three to ten: naive $1/10$ gives 0.100000 and $\mathrm{mean}(1/\texttt{n\_opt})$ gives 0.110877, so the gap to the 0.116606 always-A floor is either $+1.661$ points and $1.1661\times$ or $+0.573$ points and $1.0517\times$; both must be reported. The four are independent, so a token content floor is fixed only once all four are --- a property of $(\text{dataset},\text{convention},\text{unit},\text{tokenizer})$, not of the dataset. The letter floor is exempt from all four: a pure property of the item set's gold labels, invariant across all 15 cross-family arms and bit-identical across all 21 MMLU-Pro cells. Appendix~\ref{app:nulls} develops each with Tables~\ref{tab:nulls} and~\ref{tab:conventions}.
```

**After (verbatim):**

```latex
A content floor is not a dataset constant. Four choices usually left implicit each move it by more than the effects under study, so all four are part of the measured construct and must be printed with the score. \emph{(i) The tie convention:} on MMLU-Pro, one dataset and one tokenizer give 0.125914 under \texttt{wrong} and 0.532164 under \texttt{credit}, a 40.6-point span whose upper end exceeds the intact base model's own \texttt{content\_norm} score by 32.5 points; on ARC-Challenge \texttt{credit} alone moves five of six OLMo-2 arms from significantly above the floor to significantly below. \emph{(ii) The length unit}, characters or continuation tokens: OpenBookQA \texttt{credit} is 0.416 versus 0.644, ARC-Challenge 18.60\% tied-longest under characters against 50.85\% under tokens. \emph{(iii) The tokenizer}, moving \texttt{credit} by up to 10.6 points on four-way tasks and 9.26 on MMLU-Pro, non-monotonically in vocabulary size \citep{oostermeijer2026length}. \emph{(iv) The meaning of ``chance''} when \texttt{n\_opt} runs from three to ten: naive $1/10$ gives 0.100000 and $\mathrm{mean}(1/\texttt{n\_opt})$ gives 0.110877, so the gap to the 0.116606 always-A floor is either $+1.661$ points and $1.1661\times$ or $+0.573$ points and $1.0517\times$; both must be reported. Being independent, the four fix a token content floor only jointly --- it is a property of $(\text{dataset},\text{convention},\text{unit},\text{tokenizer})$, not of the dataset. The letter floor is exempt from all four: a pure property of the item set's gold labels, invariant across all 15 cross-family arms and bit-identical across all 21 MMLU-Pro cells. Appendix~\ref{app:nulls} develops each with Tables~\ref{tab:nulls} and~\ref{tab:conventions}.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)



## `paperC/sections/04_experiments.tex`

### Edit 29 — lines 5–5  (69→65 words, -4)

**Before (verbatim):**

```latex
We evaluate letter constructs on MMLU, ARC-Challenge, ARC-Easy, OpenBookQA, CommonsenseQA, PIQA, MMLU-Pro, and Winogrande as a negative control; BoolQ supplies an additional binary label floor. Content-side longest-option constructs are evaluated on MMLU, MMLU-Pro, and the five non-MMLU evidence tasks, with Winogrande retained only as a control. The target inventory comprises ten constructs plus the control; OpenBookQA content is reported under both character and token units rather than counted twice.
```

**After (verbatim):**

```latex
We evaluate letter constructs on MMLU, ARC-Challenge, ARC-Easy, OpenBookQA, CommonsenseQA, PIQA, MMLU-Pro, and Winogrande as a negative control; BoolQ supplies an additional binary label floor. Content-side longest-option constructs use MMLU, MMLU-Pro, and the five non-MMLU evidence tasks, Winogrande again only as a control. The inventory is ten constructs plus the control; OpenBookQA content is reported under both character and token units rather than counted twice.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 30 — lines 8–8  (89→81 words, -8)

**Before (verbatim):**

```latex
The four families are OLMo-2-7B, Llama-2-7B, Llama-3-8B, and Qwen3-8B-Base. OLMo-2 arms are prune-then-heal checkpoints, including \texttt{keep8}, \texttt{keep10}, \texttt{keep12}, \texttt{keep14}, and \texttt{shortgpt16}. Non-OLMo damage is evaluation-time front-$N$ truncation: the intact base model is loaded and its layer list is replaced by the first $N$ blocks, with no fresh block, optimizer, gradient, healing, or damaged checkpoint. This is a regime confound, not a clean family contrast, and every joint table labels it. Qwen3 has 36 layers whereas the other families have 32; consequently \texttt{k8} retains 22.2\% versus 25.0\% of the stack.
```

**After (verbatim):**

```latex
The four families are OLMo-2-7B, Llama-2-7B, Llama-3-8B, and Qwen3-8B-Base. OLMo-2 arms are prune-then-heal checkpoints, including \texttt{keep8}, \texttt{keep10}, \texttt{keep12}, \texttt{keep14}, and \texttt{shortgpt16}. Non-OLMo damage is evaluation-time front-$N$ truncation: the intact base is loaded and its layer list replaced by the first $N$ blocks, with no fresh block, optimizer, gradient, healing, or damaged checkpoint --- a regime confound, not a clean family contrast, and every joint table labels it. Qwen3's 36 layers against the others' 32 make \texttt{k8} retain 22.2\% rather than 25.0\%.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** confound disclosure

### Edit 31 — lines 11–11  (67→65 words, -2)

**Before (verbatim):**

```latex
The letter prompt is the content prompt with the labelled option body inserted before \texttt{Answer:}; the letter interface scores the candidate continuations `` A'', `` B'', and so forth. The content interface omits the labelled body and scores option text. Scoring is likelihood-based throughout: summed token log-probability over each candidate continuation followed by argmax. There is no sampling, no decoding, and no regular expression over generated text.
```

**After (verbatim):**

```latex
The letter prompt is the content prompt with the labelled option body inserted before \texttt{Answer:}; the letter interface scores the continuations `` A'', `` B'', and so forth, while the content interface omits the labelled body and scores option text. Scoring is likelihood-based throughout: summed token log-probability over each candidate continuation, then argmax. There is no sampling, no decoding, and no regex over generated text.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 32 — lines 13–13  (68→68 words, +0)

**Before (verbatim):**

```latex
All results use \texttt{chat\_template = False}. The evaluated systems are base language models with no supervised fine-tuning and no reinforcement learning from human feedback, so applying a chat template would be an unfair and non-comparable measurement. We also use \texttt{add\_bos = 0}, \texttt{desc\_style = none}, fp32 master weights with bf16-autocast forward, and batch size 48. MMLU-Pro cross-family runs use maximum length 2048; every final cell has zero truncation.
```

**After (verbatim):**

```latex
All results use \texttt{chat\_template = False}: the evaluated systems are base language models with no supervised fine-tuning and no reinforcement learning from human feedback, so applying a chat template would be an unfair and non-comparable measurement. We also use \texttt{add\_bos = 0}, \texttt{desc\_style = none}, fp32 master weights with bf16-autocast forward, and batch size 48. MMLU-Pro cross-family runs use maximum length 2048; every final cell has zero truncation.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 33 — lines 16–16  (52→52 words, +0)

**Before (verbatim):**

```latex
For deterministic per-item null predictions we report 10,000-resample paired-bootstrap intervals and two-sided mid-$p$ values with seed 7, plus exact McNemar tests. V2 additionally uses 10,000 within-stratum permutations. Every cell requires shard indices exactly $\{0,\ldots,7\}$, expected cardinality, zero duplicate IDs, zero NaNs, zero truncations, and \texttt{chat\_template is False}; partial merges raise an error.
```

**After (verbatim):**

```latex
For deterministic per-item null predictions we report 10,000-resample paired-bootstrap intervals and two-sided mid-$p$ values with seed 7, plus exact McNemar tests; v2 additionally uses 10,000 within-stratum permutations. Every cell requires shard indices exactly $\{0,\ldots,7\}$, expected cardinality, zero duplicate IDs, zero NaNs, zero truncations, and \texttt{chat\_template is False}; partial merges raise an error.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 34 — lines 19–20  (50→47 words, -3)

**Before (verbatim):**

```latex
The five smaller evidence benchmarks are unable to resolve every effect of interest, so null results are never interpreted without the power table (Table~\ref{tab:power}, Appendix~\ref{app:designated}).
MMLU-Pro supplies $n=12032$ and makes all 21 letter-floor cells capable of detecting MMLU's $-1.389$-point reference effect; their 95\% half-widths range from 0.083 to 0.968 points.
```

**After (verbatim):**

```latex
The five smaller evidence benchmarks cannot resolve every effect of interest, so null results are never interpreted without the power table (Table~\ref{tab:power}, Appendix~\ref{app:designated}).
MMLU-Pro supplies $n=12032$ and makes all 21 letter-floor cells capable of detecting MMLU's $-1.389$-point reference effect, with 95\% half-widths from 0.083 to 0.968 points.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 35 — lines 23–23  (109→107 words, -2)

**Before (verbatim):**

```latex
Every denominator below is over \emph{designated damaged} cells: all five OLMo-2 prune-then-heal arms and all four truncation rungs of each non-OLMo family, giving 17 MMLU-Pro cells and 85 off MMLU, with Winogrande the sole excluded control. The designation is fixed by construction, never by a measured score, and the set is identical in every denominator; Appendix~\ref{app:designated} records that an earlier version narrowed it on MMLU-Pro without saying so. We report per-cell $\alpha=0.05$ decisions with no family-wise correction, so headline counts are counts of per-cell decisions rather than simultaneously valid claims; the conclusions rest on the aggregate's direction and on the regime-and-depth pattern organising its exceptions, not on one cell.
```

**After (verbatim):**

```latex
Every denominator below is over \emph{designated damaged} cells: all five OLMo-2 prune-then-heal arms and all four truncation rungs of each non-OLMo family, giving 17 MMLU-Pro cells and 85 off MMLU, with Winogrande the sole excluded control. The designation is fixed by construction, never by a measured score, and is identical in every denominator; Appendix~\ref{app:designated} records that an earlier version narrowed it on MMLU-Pro without saying so. We report per-cell $\alpha=0.05$ decisions with no family-wise correction, so headline counts are counts of per-cell decisions rather than simultaneously valid claims; the conclusions rest on the aggregate's direction and on the regime-and-depth pattern organising its exceptions, not on one cell.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** designation fixed by construction



## `paperC/sections/05_analysis.tex`

### Edit 36 — lines 7–7  (69→69 words, +0)

**Before (verbatim):**

```latex
Under $\mathrm{mean}(1/\texttt{n\_opt})$, 10/12 damaged non-OLMo cells read above chance; under naive 0.10, 12/12 do. Yet only 1/12 clears the best-constant floor. Four of the five designated damaged OLMo-2 cells are likewise above either chance line but not above the floor; the fifth, \texttt{shortgpt16}, is above both. Across these 17 cells, the reference changes the interpretation; the exact chance-line counts are reported rather than compressed into a universal categorical flip.
```

**After (verbatim):**

```latex
Under $\mathrm{mean}(1/\texttt{n\_opt})$, 10/12 damaged non-OLMo cells read above chance; under naive 0.10, 12/12 do; yet only 1/12 clears the best-constant floor. Four of the five designated damaged OLMo-2 cells are likewise above either chance line but not the floor; the fifth, \texttt{shortgpt16}, is above both. Across these 17 cells the reference changes the interpretation; we report the exact chance-line counts rather than compressing them into a universal categorical flip.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 37 — lines 10–10  (79→77 words, -2)

**Before (verbatim):**

```latex
The counts above are not symmetric in their evidentiary standard: the floor comparison carries a bootstrap CI, a bootstrap $p$, and an exact McNemar test, whereas ``above chance'' was a bare point comparison. Since $\mathrm{mean}(1/\texttt{n\_opt})$ is, like the floor, a deterministic function of the evaluated item set, the matching test needs no new measurement --- the paired item bootstrap with the reference recomputed inside each resample. Applying it (10{,}000 resamples, seed 7) to the same 12 damaged non-OLMo cells:
```

**After (verbatim):**

```latex
Those counts are not symmetric in evidentiary standard: the floor comparison carries a bootstrap CI, a bootstrap $p$, and an exact McNemar test, whereas ``above chance'' was a bare point comparison. Since $\mathrm{mean}(1/\texttt{n\_opt})$ is, like the floor, a deterministic function of the evaluated item set, the matching test needs no new measurement --- the paired item bootstrap with the reference recomputed inside each resample, applied below (10{,}000 resamples, seed 7) to the same 12 damaged non-OLMo cells:
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 38 — lines 23–23  (110→94 words, -16)

**Before (verbatim):**

```latex
So the flip survives a symmetric test but is about a third of its advertised size: 3/12 versus 1/12 rather than 10/12 versus 1/12. Seven of the ten cells that read ``above chance'' were never above chance by any tested standard --- their intervals cover zero. This does not weaken the paper's rule, which is that the \emph{reference} must be stated and calibrated; it removes an asymmetry that inflated the apparent effect of choosing the wrong one. We report both columns throughout and hold the two references to the same standard: the criterion is the two-sided 95\% interval on \emph{both} sides, matching the verdict rule already used for the floor.
```

**After (verbatim):**

```latex
So the flip survives a symmetric test at about a third of its advertised size: 3/12 versus 1/12, not 10/12 versus 1/12. Seven of the ten cells reading ``above chance'' were never above chance by any tested standard --- their intervals cover zero. This does not weaken the paper's rule, that the \emph{reference} must be stated and calibrated; it removes an asymmetry that inflated the apparent effect of choosing the wrong one. We report both columns throughout, held to one standard: the two-sided 95\% interval on \emph{both} sides, as already used for the floor.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 39 — lines 25–25  (102→99 words, -3)

**Before (verbatim):**

```latex
Two qualifications belong with this two-reference comparison, both carried in Appendix~\ref{app:analysis-mult}. The exact McNemar instrument does not transfer to the chance reference, which is not a $0/1$ predictor; and under Benjamini--Hochberg at $q=0.05$ --- or Bonferroni --- \emph{neither} reference retains a single cell (0/12 both sides), so the corrected comparison is undefined rather than 3/12 versus 1/12. What survives correction is the count itself: observing 3 or more rejections out of 12 has binomial probability $0.0196$ under the global null. We therefore read this comparison as evidence that the chance side is not uniformly null, not as three simultaneously valid per-cell claims.
```

**After (verbatim):**

```latex
Two qualifications belong with this comparison, both in Appendix~\ref{app:analysis-mult}. The exact McNemar instrument does not transfer to the chance reference, which is not a $0/1$ predictor; and under Benjamini--Hochberg at $q=0.05$ --- or Bonferroni --- \emph{neither} reference retains a single cell (0/12 both sides), so the corrected comparison is undefined rather than 3/12 versus 1/12. What survives correction is the count itself: 3 or more rejections out of 12 has binomial probability $0.0196$ under the global null. We therefore read this comparison as evidence that the chance side is not uniformly null, not as three simultaneously valid per-cell claims.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no simultaneous per-cell claims

### Edit 40 — lines 27–27  (133→130 words, -3)

**Before (verbatim):**

```latex
The result is strong but not universal, and the two exceptions differ in kind. Fifteen of the 17 cells are at or below the floor. \texttt{qwen3\_8b\_base/k14} is $+0.233$ points above it ($p=0.0192$, half-width 0.191). V2 confirms a small alignment effect, $+0.267$ points at $p=0.0066$, but assigns \texttt{recovery\_fraction}=0.049, only 9.1\% of the intact-family anchor and below the 10\% materiality bar. It is a real but immaterial exception, not evidence that \texttt{k14} retains MMLU-Pro competence. OLMo-2 \texttt{shortgpt16} is not borderline: $+3.674$ points at $p=0.0001$, $5.2\times$ its own half-width. It retains 16 of 32 layers, more than any \texttt{keepN} rung, and we read it as a positive result --- the arm retaining most of the stack is the one whose read-out survives, so the floor test tracks capability rather than failing everything put to it (Appendix~\ref{app:designated}).
```

**After (verbatim):**

```latex
The result is strong but not universal: fifteen of the 17 cells are at or below the floor, and the two exceptions differ in kind. \texttt{qwen3\_8b\_base/k14} is $+0.233$ points above it ($p=0.0192$, half-width 0.191); v2 confirms a small alignment effect, $+0.267$ points at $p=0.0066$, but assigns \texttt{recovery\_fraction}=0.049, only 9.1\% of the intact-family anchor and below the 10\% materiality bar --- a real but immaterial exception, not evidence that \texttt{k14} retains MMLU-Pro competence. OLMo-2 \texttt{shortgpt16} is not borderline: $+3.674$ points at $p=0.0001$, $5.2\times$ its own half-width. Retaining 16 of 32 layers, more than any \texttt{keepN} rung, we read it as a positive result --- the arm retaining most of the stack is the one whose read-out survives, so the floor test tracks capability rather than failing everything put to it (Appendix~\ref{app:designated}).
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** materiality bar; non-universality

### Edit 41 — lines 29–29  (244→235 words, -9)

**Before (verbatim):**

```latex
The most vivid small-benchmark cases are literal constant emitters: nine damaged cells in the designated set have accuracy equal to the marginal of the emitted letter to machine precision (sixteen if the negative control is included, which enters no denominator), and two OpenBookQA cells emit A on every item, score exactly 0.276000, and land on the optimal constant with $\Delta=0.000$ points and CI95 $[0,0]$ --- yet read 2.6 points above chance 0.25. Off MMLU the designated set gives 9/85 above floor and 45/85 above chance, and those nine above-floor cells are the shallowest-pruned healed arms: all retain at least 14 of 32 layers, while 0/15 OLMo-2 cells at 12 layers or fewer and 0/60 truncate-only cells at any depth clear a floor. Retained depth alone does not carry this: \emph{at the same} retained depth of 14, the healed arm clears 4 of its 5 floors while the fifteen truncate-only cells at that depth clear none, so the threshold is depth \emph{and} prune-then-heal rather than depth by itself --- and since OLMo-2 is our only healed family, regime and family stay confounded, which is why we state the threshold as a property of that ladder rather than of depth in general. Only 7/60 are significantly below because 52/60 are underpowered for the MMLU reference effect. That is a power limit, not a small effect: ARC-Challenge's median damaged effect, $-3.840$ points, is larger than MMLU's $-3.603$, but its half-width is 3.92 rather than 1.18 points (Appendix~\ref{app:offmmlu}).
```

**After (verbatim):**

```latex
The most vivid small-benchmark cases are literal constant emitters: nine damaged cells in the designated set have accuracy equal to the marginal of the emitted letter to machine precision (sixteen if the negative control is included, which enters no denominator), and two OpenBookQA cells emit A on every item, scoring exactly 0.276000 --- the optimal constant, $\Delta=0.000$ points, CI95 $[0,0]$ --- yet read 2.6 points above chance 0.25. Off MMLU the designated set gives 9/85 above floor and 45/85 above chance, and those nine are the shallowest-pruned healed arms: all retain at least 14 of 32 layers, while 0/15 OLMo-2 cells at 12 layers or fewer and 0/60 truncate-only cells at any depth clear a floor. Retained depth alone does not carry this: \emph{at the same} retained depth of 14, the healed arm clears 4 of its 5 floors while the fifteen truncate-only cells at that depth clear none, so the threshold is depth \emph{and} prune-then-heal rather than depth by itself --- and since OLMo-2 is our only healed family, regime and family stay confounded, which is why we state the threshold as a property of that ladder rather than of depth in general. Only 7/60 are significantly below, because 52/60 are underpowered for the MMLU reference effect. That is a power limit, not a small effect: ARC-Challenge's median damaged effect, $-3.840$ points, exceeds MMLU's $-3.603$, yet its half-width is 3.92 rather than 1.18 (Appendix~\ref{app:offmmlu}).
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** power limit vs small effect; regime/family confound; power caveat; confound disclosure

### Edit 42 — lines 33–33  (73→69 words, -4)

**Before (verbatim):**

```latex
A floor-level letter score does not imply that all task-relevant competence is gone. On ARC-Easy, OLMo-2 \texttt{keep8} scores 0.2584 on letters, statistically at its 0.266414 letter floor, while \texttt{content\_norm} reaches 0.6460. The paired gap is $+38.76$ points with McNemar $p=9.8\times10^{-148}$. The arm can rank answer contents but cannot express that knowledge through the damaged letter interface. Conversely, healthy models often favor letters, so ``content is the fair interface'' is not a general conclusion.
```

**After (verbatim):**

```latex
A floor-level letter score does not imply all task-relevant competence is gone. On ARC-Easy, OLMo-2 \texttt{keep8} scores 0.2584 on letters, statistically at its 0.266414 letter floor, while \texttt{content\_norm} reaches 0.6460 --- a paired gap of $+38.76$ points, McNemar $p=9.8\times10^{-148}$. The arm can rank contents but cannot express that knowledge through the damaged letter interface. Conversely, healthy models often favor letters, so ``content is the fair interface'' generalises no better.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 43 — lines 35–35  (58→56 words, -2)

**Before (verbatim):**

```latex
The residual-fraction effect is also construct-specific. On OpenBookQA \texttt{content\_norm}, using chance rather than the 0.3680 token-longest floor inflates the base arm's residual fraction by 2.11$\times$. PIQA and ARC-Easy move in the opposite direction, to 0.90$\times$ and 0.98$\times$, because their token-longest floors are below chance. Null calibration can increase or decrease the residual; it is not a one-way correction.
```

**After (verbatim):**

```latex
The residual-fraction effect is also construct-specific. On OpenBookQA \texttt{content\_norm}, using chance rather than the 0.3680 token-longest floor inflates the base arm's residual fraction by 2.11$\times$; PIQA and ARC-Easy move the opposite way, to 0.90$\times$ and 0.98$\times$, their token-longest floors being below chance. Null calibration can increase or decrease the residual; it is not a one-way correction.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** null calibration is not one-way

### Edit 44 — lines 39–39  (65→65 words, +0)

**Before (verbatim):**

```latex
Under v1, two MMLU-Pro cells are significantly below the best-constant floor. Under v2, the below-null count is zero. \texttt{qwen3/k8} moves from $-0.881$ points ($p=0.0362$) to $-0.139$ ($p=0.0964$), and \texttt{llama2/k8} from $-0.914$ to $-0.416$ ($p=0.1002$). Their v1 verdicts remain valid statements about an arm-independent interface floor, but they are not evidence that either arm is worse than its own input-blind prediction marginal. Both competence claims dissolve.
```

**After (verbatim):**

```latex
Under v1, two MMLU-Pro cells are significantly below the best-constant floor; under v2 the below-null count is zero. \texttt{qwen3/k8} moves from $-0.881$ points ($p=0.0362$) to $-0.139$ ($p=0.0964$), and \texttt{llama2/k8} from $-0.914$ to $-0.416$ ($p=0.1002$). Their v1 verdicts remain valid statements about an arm-independent interface floor, but they are not evidence that either arm is worse than its own input-blind prediction marginal. Both competence claims dissolve.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 45 — lines 41–41  (190→181 words, -9)

**Before (verbatim):**

```latex
V2 is not uniformly conservative (Table~\ref{tab:v2-resort}, Appendix~\ref{app:mmlupro-and-resort}). It withdraws the published above-floor capability label for \texttt{qwen3/k14}, replacing it with \texttt{TRACE\_SIGNAL}, while \texttt{olmo2/keep14} moves from at-floor to \texttt{TRACE\_SIGNAL}. The two moves are not equally defensible and we separate them rather than reading them as one bidirectional effect. Table~\ref{tab:v2-full} carries a Benjamini--Hochberg $q$-value over exactly its own 27 cells, which is the one place in this paper where the family definition is not in dispute: one table, one null, one statistic, one item set. The downgrade survives correction --- \texttt{qwen3/k14} has $q_{\mathrm{BH}}=0.0297$ --- but the upgrade does not: \texttt{olmo2/keep14} at $p=0.0172$ sits at rank 7, where the step-up threshold is $7\times0.05/27=0.0130$, giving $q_{\mathrm{BH}}=0.0663$. Under Bonferroni ($\alpha/27=0.001852$) neither trace cell clears and only the five $p=0.0001$ anchors do. Since \texttt{olmo2/keep14} is the \emph{only} cell moving upward, the corrected read-out re-sorts in one direction rather than two. We therefore claim only what survives: v2 does not merely rescale v1, because it reverses one verdict that holds up under multiplicity correction over its own family. The upward move is reported as a single uncorrected per-cell observation and no conclusion in this paper rests on it.
```

**After (verbatim):**

```latex
V2 is not uniformly conservative (Table~\ref{tab:v2-resort}, Appendix~\ref{app:mmlupro-and-resort}). It withdraws the published above-floor capability label for \texttt{qwen3/k14}, replacing it with \texttt{TRACE\_SIGNAL}, while \texttt{olmo2/keep14} moves from at-floor to \texttt{TRACE\_SIGNAL}. The two are not equally defensible, so we separate them rather than reading them as one bidirectional effect. Table~\ref{tab:v2-full} carries a Benjamini--Hochberg $q$-value over exactly its own 27 cells --- the one place here where the family definition is not in dispute: one table, one null, one statistic, one item set. The downgrade survives correction (\texttt{qwen3/k14}, $q_{\mathrm{BH}}=0.0297$); the upgrade does not, since \texttt{olmo2/keep14} at $p=0.0172$ sits at rank 7, where the step-up threshold is $7\times0.05/27=0.0130$, giving $q_{\mathrm{BH}}=0.0663$. Under Bonferroni ($\alpha/27=0.001852$) neither trace cell clears; only the five $p=0.0001$ anchors do. Since \texttt{olmo2/keep14} is the \emph{only} cell moving upward, the corrected read-out re-sorts in one direction rather than two, so we claim only what survives: v2 does not merely rescale v1, because it reverses one verdict holding up under multiplicity correction over its own family. The upward move is reported as a single uncorrected per-cell observation and no conclusion in this paper rests on it.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** single-observation caveat; asymmetric defensibility

### Edit 46 — lines 43–43  (42→41 words, -1)

**Before (verbatim):**

```latex
It also exposes a benchmark-level limitation invisible to v1. Intact Llama-2-7B has \texttt{recovery\_fraction}=0.0545 on MMLU-Pro, even though v1 calls it comfortably above floor by $+1.538$ points. Because the intact anchor itself is below 0.10, relative-recovery claims for the Llama-2 family are blocked.
```

**After (verbatim):**

```latex
It also exposes a benchmark-level limitation invisible to v1: intact Llama-2-7B has \texttt{recovery\_fraction}=0.0545 on MMLU-Pro even though v1 calls it comfortably above floor by $+1.538$ points. The intact anchor being itself below 0.10, relative-recovery claims for the Llama-2 family are blocked.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** relative-claim block

### Edit 47 — lines 45–45  (31→31 words, +0)

**Before (verbatim):**

```latex
Significance alone would re-import the defect. At $n=12032$, \texttt{qwen3/k14} reaches $p=0.0066$ at only $+0.267$ points while A wins the likelihood argmax on 94.6\% of items. The same data reject simpler diagnostics
```

**After (verbatim):**

```latex
Significance alone would re-import the defect: at $n=12032$, \texttt{qwen3/k14} reaches $p=0.0066$ at only $+0.267$ points while A wins the likelihood argmax on 94.6\% of items. The same data reject simpler diagnostics
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 48 — lines 47–47  (60→59 words, -1)

**Before (verbatim):**

```latex
the A01 own-modal null over-credits P1 by 1.37 points and flips its sign; for Qwen3 heal@7000 and unhealed \texttt{k8}, content is $-3.610$ and $-1.076$ points under the same permutation null, compared with letter at $-0.022$ and $-0.139$; and \texttt{llama3/k12} has modal share 0.339 and normalized entropy 0.614 yet $\Delta_{\mathrm{perm}}=+0.002$ points ($p=0.997$). Entropy and modal share are descriptive, not competence metrics.
```

**After (verbatim):**

```latex
the A01 own-modal null over-credits P1 by 1.37 points and flips its sign; for Qwen3 heal@7000 and unhealed \texttt{k8}, content is $-3.610$ and $-1.076$ points under the same permutation null, against letter at $-0.022$ and $-0.139$; and \texttt{llama3/k12} has modal share 0.339 and normalized entropy 0.614 yet $\Delta_{\mathrm{perm}}=+0.002$ points ($p=0.997$). Entropy and modal share are descriptive, not competence measures.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** descriptive-not-competence

### Edit 49 — lines 51–51  (67→69 words, +2)

**Before (verbatim):**

```latex
For OLMo-2 \texttt{keep8} on MMLU, fp32 removes all 4,303 bf16 exact top-two ties and changes 2,532 of 14,042 letter argmax decisions, or 18.03\%. Letter accuracy nevertheless changes by only $-0.0015$, CI95 $[-0.0064,+0.0033]$, exact McNemar $p=0.5702$. The arm is more significantly below its floor in fp32: $-1.538$ points, $p=0.0060$, versus $-1.389$, $p=0.0190$, in bf16. Higher precision can reshuffle ambiguous decisions; it cannot create item-level information that is absent.
```

**After (verbatim):**

```latex
For OLMo-2 \texttt{keep8} on MMLU, fp32 removes all 4,303 bf16 exact top-two ties and changes 2,532 of 14,042 letter argmax decisions, or 18.03\%, yet letter accuracy changes by only $-0.0015$, CI95 $[-0.0064,+0.0033]$, exact McNemar $p=0.5702$. The arm is in fact more significantly below its floor in fp32: $-1.538$ points, $p=0.0060$, versus $-1.389$, $p=0.0190$, in bf16. Higher precision can reshuffle ambiguous decisions; it cannot create item-level information that is absent.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 50 — lines 54–54  (91→89 words, -2)

**Before (verbatim):**

```latex
The analysis uncovered and repaired three defects without changing the substantive verdicts: a doubled-tail bootstrap formula that produced an illegal $p=1.042$ (re-emission changed 0/24 verdicts and moved 0/30 $p$-values across 0.05), a sequence cap validated on the wrong tokenizer that silently left-truncated 10/15 MMLU-Pro cells (0/14 verdicts changed, at most 0.0083 points, but benignity was unknowable in advance), and an out-of-memory failure on 5/8 intact-Llama-2 shards that the merge guard correctly refused. All reported cross-family MMLU-Pro results use the corrected launch. Table~\ref{tab:integrity} and Appendix~\ref{app:audit} give each defect with its measured impact.
```

**After (verbatim):**

```latex
The analysis uncovered and repaired three defects without changing the substantive verdicts: a doubled-tail bootstrap formula producing an illegal $p=1.042$ (re-emission changed 0/24 verdicts and moved 0/30 $p$-values across 0.05); a sequence cap validated on the wrong tokenizer that silently left-truncated 10/15 MMLU-Pro cells (0/14 verdicts changed, at most 0.0083 points, but benignity was unknowable in advance); and an out-of-memory failure on 5/8 intact-Llama-2 shards that the merge guard correctly refused. All reported cross-family MMLU-Pro results use the corrected launch; Table~\ref{tab:integrity} and Appendix~\ref{app:audit} give each with its measured impact.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** benignity unknowable



## `paperC/sections/06_discussion.tex`

### Edit 51 — lines 4–4  (65→65 words, +0)

**Before (verbatim):**

```latex
The practical recommendation is small: define the strongest construct-appropriate constant or input-blind predictor, report its value and specification, test the arm against it, and only then compare arms. The rule does not say every damaged model becomes constant, every MC interface is invalid, or every chance line inflates a claim. It says that an arm score without its appropriate floor can support the wrong verdict.
```

**After (verbatim):**

```latex
The practical recommendation is small: define the strongest construct-appropriate constant or input-blind predictor, report its value and specification, test the arm against it, and only then compare arms. The rule does not say every damaged model becomes constant, every MC interface is invalid, or every chance line inflates a claim --- only that an arm score without its appropriate floor can support the wrong verdict.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 52 — lines 6–6  (62→62 words, +0)

**Before (verbatim):**

```latex
The letter-floor and permutation read-outs partition scope. V1 must be arm-independent because it certifies mutual comparability of an interface. V2 must be arm-conditional because it asks whether one prediction vector aligns with its items beyond its own output marginal. Substituting one for the other creates the defects we observe: chance credits constants, while a best-constant competence test confounds knowledge with collapse-letter identity.
```

**After (verbatim):**

```latex
The letter-floor and permutation read-outs partition scope. V1 must be arm-independent because it certifies mutual comparability of an interface; v2 must be arm-conditional because it asks whether one prediction vector aligns with its items beyond its own output marginal. Substituting one for the other creates the defects we observe: chance credits constants, while a best-constant competence test confounds knowledge with collapse-letter identity.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** confound disclosure

### Edit 53 — lines 9–9  (54→56 words, +2)

**Before (verbatim):**

```latex
PriDe-style option permutations estimate and remove a model's selection bias \citep{zheng2024selectors}. This is complementary, not redundant. A corrected predictor can still be evaluated against the wrong reference; an explicit floor remains necessary. Conversely, a floor test does not repair a biased predictor. A complete evaluation can report both predictor-side debiasing and reference-side null calibration.
```

**After (verbatim):**

```latex
PriDe-style option permutations estimate and remove a model's selection bias \citep{zheng2024selectors}. This is complementary, not redundant: a corrected predictor can still be evaluated against the wrong reference, so an explicit floor remains necessary, while conversely a floor test does not repair a biased predictor. A complete evaluation can report both predictor-side debiasing and reference-side null calibration.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** reference-not-magnitude scoping

### Edit 54 — lines 12–12  (68→66 words, -2)

**Before (verbatim):**

```latex
Small benchmarks produce apparently reassuring at-floor results even when observed effects are large. Any interpretation of a null result must include the achieved interval. MMLU-Pro changes the evidential status: OLMo-2 \texttt{keep8}'s interval excludes an MMLU-sized below-floor effect, so this is a powered non-replication, not an absence of evidence. The broader ladder remains a cliff rather than a gradient; no depth curve should be fitted to the sparse rungs.
```

**After (verbatim):**

```latex
Small benchmarks produce apparently reassuring at-floor results even when observed effects are large, so any interpretation of a null result must include the achieved interval. MMLU-Pro changes the evidential status: OLMo-2 \texttt{keep8}'s interval excludes an MMLU-sized below-floor effect, making this a powered non-replication, not an absence of evidence. The ladder remains a cliff, not a gradient; no depth curve should be fitted to the sparse rungs.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no curve fitting; cliff not gradient

### Edit 55 — lines 17–17  (45→42 words, -3)

**Before (verbatim):**

```latex
A healed non-OLMo arm is in training, but it is not part of this paper's evidence. Its remaining question is only whether it develops material item-level signal. The original \texttt{H\_heal}/\texttt{H\_family} dichotomy is unavailable because its unhealed comparator does not fall below the collapse-proof permutation null.
```

**After (verbatim):**

```latex
A healed non-OLMo arm is in training but is not part of this paper's evidence; its only remaining question is whether it develops material item-level signal. The original \texttt{H\_heal}/\texttt{H\_family} dichotomy is unavailable: its unhealed comparator does not fall below the collapse-proof null.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)



## `paperC/sections/07_limitations.tex`

### Edit 56 — lines 4–4  (96→93 words, -3)

**Before (verbatim):**

```latex
The OpenBookQA character floor was added after the original audit identified an evidence hole. It is now recomputed from raw parquet and passes a 12/12 character/token self-test, so we do not use the earlier ``unrecomputable'' status. The remaining limitation is narrower: the character values are reconstructed from raw option text rather than retained as per-item fields in the scored records. Item matching to the token leg therefore follows the shared loader and self-test rather than a stored character-count column. The convention of excluding the continuation's leading space was explicit; including it yielded the identical 0.363500 value.
```

**After (verbatim):**

```latex
The OpenBookQA character floor was added after the original audit identified an evidence hole. It is now recomputed from raw parquet and passes a 12/12 character/token self-test, so we drop the earlier ``unrecomputable'' status. The remaining limitation is narrower: the character values are reconstructed from raw option text rather than retained as per-item fields in the scored records, so item matching to the token leg follows the shared loader and self-test, not a stored character-count column. The convention of excluding the continuation's leading space was explicit; including it yielded the identical 0.363500 value.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 57 — lines 7–7  (35→35 words, +0)

**Before (verbatim):**

```latex
The prospective healed contrast contains one non-OLMo family at one absolute depth. Llama-2, Llama-3, and the \texttt{k10}/\texttt{k12}/\texttt{k14} rungs remain confounded. The observed ladder is a cliff, not a gradient, and we fit no depth curve.
```

**After (verbatim):**

```latex
The prospective healed contrast has one non-OLMo family at one absolute depth, leaving Llama-2, Llama-3, and the \texttt{k10}/\texttt{k12}/\texttt{k14} rungs confounded. The observed ladder is a cliff, not a gradient, and we fit no depth curve.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** confound disclosure; no curve fitting; cliff not gradient

### Edit 58 — lines 10–10  (52→49 words, -3)

**Before (verbatim):**

```latex
The Qwen3 arm heals on 5.72 epochs of 5.541B SlimPajama tokens, whereas OLMo-2 used 1.0 epoch of 31.7B Dolmino tokens. Qwen3 cannot consume OLMo-2-token Dolmino and raw Dolmino text is on neither disk. Relative depth is also untested: Qwen3 has 36 layers and OLMo-2 has 32, so \texttt{keep8} retains 22.2\% versus 25.0\%.
```

**After (verbatim):**

```latex
The Qwen3 arm heals on 5.72 epochs of 5.541B SlimPajama tokens, OLMo-2 on 1.0 epoch of 31.7B Dolmino tokens; Qwen3 cannot consume OLMo-2-token Dolmino and raw Dolmino text is on neither disk. Relative depth is untested too: Qwen3's 36 layers to OLMo-2's 32 make \texttt{keep8} retain 22.2\% versus 25.0\%.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 59 — lines 13–13  (64→64 words, +0)

**Before (verbatim):**

```latex
\texttt{H\_heal} is not merely unmeasured. Its antecedent required the unhealed Qwen3 \texttt{k8} comparator to sit below an arm-conditional null; it does not ($-0.139$ points, $p=0.0964$). No amount of further training repairs a comparator that fails the criterion. \texttt{H\_family} is equally unavailable because 0/27 cells are significantly below the permutation null. A future read-out can ask only whether the healed arm has material item-level signal.
```

**After (verbatim):**

```latex
\texttt{H\_heal} is not merely unmeasured: its antecedent required the unhealed Qwen3 \texttt{k8} comparator to sit below an arm-conditional null, and it does not ($-0.139$ points, $p=0.0964$). No amount of further training repairs a comparator failing the criterion. \texttt{H\_family} is equally unavailable because 0/27 cells are significantly below the permutation null. A future read-out can ask only whether the healed arm has material item-level signal.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** no scoping/hedging marker in this hunk (pure wording tightening)

### Edit 60 — lines 16–16  (69→62 words, -7)

**Before (verbatim):**

```latex
Non-OLMo cross-family arms are truncate-only, whereas OLMo-2 arms are prune-then-heal; tables that include both cannot identify a family effect. Pooled results average per-item deviations over disjoint benchmarks and must not be read as per-benchmark verdicts. Finally, the Cho et al. overlap audit used arXiv v4 dated January 12, 2026; the January 26, 2026 camera-ready could not be diffed because the OpenReview PDF endpoint was inaccessible from the audit network.
```

**After (verbatim):**

```latex
Tables mixing truncate-only non-OLMo arms with prune-then-heal OLMo-2 arms cannot identify a family effect. Pooled results average per-item deviations over disjoint benchmarks and must not be read as per-benchmark verdicts. Finally, the Cho et al. overlap audit used arXiv v4 (January 12, 2026); the January 26, 2026 camera-ready could not be diffed, the OpenReview PDF endpoint being inaccessible from the audit network.
```

- **Numbers in this hunk:** IDENTICAL
- **Qualification preserved:** non-identifiability; pooling caveat



**TOTAL: 60 edit hunks.**


---

## Per-file word counts and number-identity

| file | words before | words after | Δ | numeric literals (sorted md5) |
|---|---|---|---|---|
| `00_abstract.tex` | 421 | 407 | -14 | `e76c1c5537` = `e76c1c5537` **identical** |
| `01_introduction.tex` | 647 | 621 | -26 | `b3846dffbe` = `b3846dffbe` **identical** |
| `02_related.tex` | 542 | 521 | -21 | `429526de68` = `429526de68` **identical** |
| `03_method.tex` | 949 | 876 | -73 | `b02d1f985e` = `b02d1f985e` **identical** |
| `03b_nulls_summary.tex` | 117 | 115 | -2 | `ed7b146cca` = `ed7b146cca` **identical** |
| `04_experiments.tex` | 516 | 500 | -16 | `269a4785bc` = `269a4785bc` **identical** |
| `05_analysis.tex` | 1344 | 1298 | -46 | `5ecc3a143b` = `5ecc3a143b` **identical** |
| `06_discussion.tex` | 414 | 411 | -3 | `c7e22b3b67` = `c7e22b3b67` **identical** |
| `07_limitations.tex` | 348 | 335 | -13 | `8ac5445513` = `8ac5445513` **identical** |
| **total** | **5298** | **5084** | **-214** | all nine identical |


---

## Sentence-level audit: the five must-survive sentences

All five are **present**, and four of the five are **byte-identical** to their original
wording. Quoted below exactly as they now read on disk.

### 1. The regime/family confound disclosure

**PRESENT — byte-identical.** `sections/05_analysis.tex:29`:

> since OLMo-2 is our only healed family, regime and family stay confounded, which is why we
> state the threshold as a property of that ladder rather than of depth in general

Its host sentence was tightened around it (`does not carry this` … `while the fifteen
truncate-only cells at that depth clear none`), but this clause is untouched. It is also
guarded by two registered count claims (`depth-14 healed above-floor` = 4,
`depth-14 truncate-only cells` = 15), both still `ok`.

### 2. The power-limit-not-small-effect distinction

**PRESENT — byte-identical.** `sections/05_analysis.tex:29`:

> That is a power limit, not a small effect

Its evidence clause was tightened from `is larger than MMLU's $-3.603$, but its half-width is
3.92 rather than 1.18 points` to `exceeds MMLU's $-3.603$, yet its half-width is 3.92 rather
than 1.18` — same two numbers, same comparison, same direction. Dropping the repeated unit
word `points` (already established twice in the same sentence) is the only change.

### 3. Null calibration is not a one-way correction

**PRESENT — byte-identical.** `sections/05_analysis.tex:35`:

> Null calibration can increase or decrease the residual; it is not a one-way correction.

Only the preceding clause changed (`because their token-longest floors are below chance` →
`their token-longest floors being below chance`). The bidirectionality claim, and both
directions of evidence (2.11× up; 0.90× and 0.98× down), are intact.

### 4. The upward move is a single uncorrected observation

**PRESENT — byte-identical.** `sections/05_analysis.tex:41`:

> The upward move is reported as a single uncorrected per-cell observation and no conclusion
> in this paper rests on it.

The surrounding paragraph kept every multiplicity number ($q_{\mathrm{BH}}=0.0297$,
$p=0.0172$, rank 7, $7\times0.05/27=0.0130$, $q_{\mathrm{BH}}=0.0663$,
$\alpha/27=0.001852$) and the asymmetric-defensibility framing (`The two are not equally
defensible, so we separate them rather than reading them as one bidirectional effect`).

### 5. Not three simultaneously valid per-cell claims

**PRESENT — byte-identical.** `sections/05_analysis.tex:25`:

> We therefore read this comparison as evidence that the chance side is not uniformly null,
> not as three simultaneously valid per-cell claims.

Two words were cut earlier in that paragraph (`both carried in Appendix` → `both in
Appendix`; `observing 3 or more rejections` → `3 or more rejections`). The BH/Bonferroni
`0/12 both sides`, the `undefined rather than 3/12 versus 1/12` verdict, and the binomial
$0.0196$ all remain.

### Wider hedge sweep

Beyond those five, I counted 29 scoping/hedging markers across all nine files before and
after. **28 are unchanged in count.** The one apparent drop is `not new`, which fell from 1 to
0 — because the §3 paragraph heading was rewritten from

> The statistic is not new; the stratification and its use are.

to

> The statistic is decades old; only the varying-$k$ stratification and its use are ours.

This **strengthens** the concession rather than removing it (`decades old` is a stronger
admission of prior art than `not new`), and the body still says `We claim neither this
statistic nor the collapse identity below as new` plus `we claim none of it`. All five
prior-art citations (`bennett1954communications`, `brennan1981kappa`, `frary1988formula`,
`brenner1996weightedkappa`, `devries2008pooledkappa`) are retained.

## The one structural change, and why it is redundancy rather than content

`03_method.tex` carried **two consecutive paragraphs making the same concession**: "the
statistic is not ours" and "option-count-aware chance correction is not ours". They shared a
clause almost verbatim — the first said *"so letters that are illegal for an item are never
credited"*, the second *"so a letter illegal for an item can never be credited to it"*. I
merged them into one paragraph (180 → 88 words, the single largest saving in this pass) that
still states, in order:

1. the $\kappa$-numerator identity and its verification on all 27 cells;
2. that neither the statistic nor the collapse identity is claimed as new;
3. that option-count-aware correction predates this work, with all five citations;
4. the three things explicitly *not* claimed (correcting for option count, choosing $p_e$,
   $\kappa$ depending on $k$);
5. both things that *are* claimed — (i) varying-$k$ within-stratum permutation, (ii) use as an
   arm-conditional pre-comparison gate with a materiality bar;
6. why $\Delta_{\mathrm{perm}}$ is reported rather than $\kappa$;
7. the pointer to `app:priorart` and the 36-item footprint.

Nothing was moved to the appendix, and no claim boundary was widened.
