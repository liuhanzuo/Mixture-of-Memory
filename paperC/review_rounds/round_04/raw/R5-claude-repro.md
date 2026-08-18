# Review — R5-claude-repro (reproducibility & provenance lens)

- Round: 4
- Snapshot sha256: `7fcb9ccc55c5c1d6ad1de868215b1d9253d0695a2f996340226aa6a6fa10db5a`
- Overall: **5 / 10** (weak reject / negative borderline)
- Confidence: **4**
- Ceiling without new experiments: **6.5**

## 1. What the paper claims

A measurement-protocol paper. Before comparing model arms on a multiple-choice construct, test the
arm's score against the construct's best *input-blind* predictor (v1: the best constant letter under
the empirical gold marginal; for content-likelihood, a longest-option heuristic), not against nominal
chance. Four claims carry the paper:

- **H1** the floor is not chance, and for content it is under-specified by (tie convention, length
  unit, tokenizer), plus a fourth degree of freedom when option count varies;
- **H2** the floor estimator is itself upward biased (a max over k noisy marginals) and must be
  calibrated — after calibration only MMLU-Pro, MMLU, BoolQ survive at p<1e-5;
- **H3** the reference choice flips published readings: 14/15 designated damaged cells sit at or
  below the floor while reading above chance; held to one symmetric standard the flip is 3/12 vs 1/12;
- **H4** a stratified arm-conditional permutation null (v2), exactly zero for any constant emitter,
  re-sorts 27 previously scored cells.

The self-discipline on display is genuinely unusual: a 16-row retraction ledger, an explicit
"prohibited numbers" row, a bootstrap-p bug disclosed with its measured impact (0/24 verdicts), a
silent-truncation bug disclosed with the honest note that its benignity was unknowable in advance,
and a paragraph that voluntarily shrinks the paper's own headline from 10/12-vs-1/12 to 3/12-vs-1/12.
I want to be clear that I am scoring a paper that is trying hard to be honest. The problems below are
not bad faith; they are places where the honesty machinery did not actually run on the number it was
pointed at.

## 2. Strongest verified contribution

The **internal arithmetic consistency of the measurement tables is real and I verified it
end-to-end.** All 27 rows of Table 7 (`tab_v2_full`) satisfy Δ_perm = 100·(acc − acc-hat) to printing
precision (0/27 mismatches). All 11 MMLU-Pro cells in Table 4 (`tab_mmlupro`) satisfy Δ =
100·(acc − 0.116606) against the *same* always-A floor, and the accuracies in Table 4 agree with the
independent Table 7 listing of the same cells. Every floor in Table 1 is an exact integer label count
over n (1403/12032, 3776/14042, 138/500, 633/2376, 311/1172, 255/1221, 928/1838, 2033/3270 — all
integers to <7e-3 of a count), which is what a genuine label-marginal floor must look like and is
strong evidence these were computed from real per-item records rather than transcribed. Gaps and
ratios in Table 1 reproduce to the printed digit in 9/9 rows. The `+38.76` pp ARC-Easy content–letter
gap, the `18.03%` fp32 flip rate (2532/14042 = 18.0316%), the `0.299` pp / 36-item stratification
slack (36/12032), the `0.0078` pp Llama-2 k12 margin, the `94.8%` A-argmax weight (11409/12032), and
the `0.0196` binomial P(X≥3 | 12, 0.05) all reproduce exactly. That is a high standard of internal
bookkeeping and the paper deserves credit for it.

## 3. The decisive problem: the calibration that the paper leans on hardest is the one null in the
paper that is misspecified, and it is misspecified in the direction that flatters the claim

This is my fatal issue (**R5-01**). Table 1 / Table 10 calibrate each floor against "an exactly
balanced multinomial null at that construct's own (n, k)". For MMLU-Pro the paper simultaneously
tells us, in its own §A.1.4, that `n_opt` ranges from 3 to 10 and that mean(1/n_opt) = 0.110877 —
i.e. **the letters are not equiprobable across the item set, because most letters are illegal on some
items.** But the calibration uses k = 10 (Table 10 prints `k = 10` explicitly for both MMLU-Pro rows)
and therefore draws gold labels uniform over ten letters on every item.

I reproduced the paper's numbers under its own stated recipe and they match to 6 decimals
(Multinomial(12032, uniform over 10), 2e5 draws: E[f-hat] = 0.104463, q95 = 0.107048, p = 0.0 —
paper prints 0.104457 / 0.107048 / <1e-5). So the paper's script does what the caption says. The
problem is that the recipe is the wrong null for this construct, and there is a **deterministic**
proof, no simulation required:

> Letter A is legal on every MMLU-Pro item. Under any null in which each item's gold label is uniform
> over that item's *legal* letters, E[m-hat_A] = mean_i(1/n_opt_i) = **0.110877** — which is the
> paper's own "Chance" entry in that very table row. Since f-hat = max_L m-hat_L ≥ m-hat_A, we get
> **E[f-hat] ≥ 0.110877**. Table 1's item-average row prints **E[f-hat] = 0.104460, which is below
> its own Chance column.** An unbiased-upward maximum cannot have expectation below the expectation
> of one of the terms it maximises over. The row is internally impossible.

Quantitatively: sd(m-hat_A) = sqrt(p(1−p)/n) = 0.002862, so the observed floor 0.116606 is only
**2.0 sd** above E[m-hat_A] *before* accounting for the max — and the max pushes the null mean up
further. Simulating the correct item-wise-legal null over admissible `n_opt` histograms (I searched
histograms consistent with all three of the paper's own constraints: `n_opt ∈ {3..10}`, 623 items in
strata 5–8, mean(1/n_opt) = 0.110877) gives, across every admissible histogram I found:

| null | E[f-hat] | q95 | p(f-hat ≥ 0.116606) |
|---|---|---|---|
| paper's, uniform over k=10 | 0.104460 | 0.107048 | **< 1e-5** |
| item-wise-legal, hist A | 0.113932 | 0.117188 | **0.081** |
| item-wise-legal, hist B | 0.113463 | 0.116938 | **0.064** |
| item-wise-legal, 6 further admissible histograms | 0.1135–0.1138 | 0.1170–0.1174 | **0.064–0.087** |

The observed floor is **inside** the correctly specified estimator's noise, and in several histograms
below q95. This is not a rounding quibble. **MMLU-Pro is the paper's flagship construct** — it is the
only one with MMLU-scale n, it supplies the 14/15 headline, the 3/12-vs-1/12 symmetric comparison, all
27 v2 cells, and the abstract's "only three of the eight letter constructs (MMLU-Pro, MMLU, BoolQ)
have floors a balanced null could not produce (p<1e-5)". Under the correct null, MMLU-Pro plausibly
moves into the same "inside estimator noise" bucket as the five constructs the paper already
demoted, and the sentence becomes "two of eight". The paper invented exactly the right diagnostic —
winner's-curse calibration of a max — and then ran it against a null that ignores the varying option
count it spends a whole appendix subsection establishing. Note this cuts *against* the paper on the
`k=10` row and would also affect ARC-Easy (chance printed 0.250161 ≠ 0.25) and ARC-Challenge
(0.250156), whose non-round chance lines prove `n_opt` varies there too while their calibration rows
print `k = 4`.

I want to be fair about scope: this does **not** touch H1 (the floor differs from *nominal* chance,
which for BoolQ 0.6217 vs 0.50 is a 12.17 pp gap that no plausible null explains), does not touch H3
(the flip counts are ordinal comparisons against a fixed floor and do not use the calibration), and
does not touch H4. It kills one specific quantitative sentence in the abstract, intro bullet 1,
Table 1, Table 10, and the Reproducibility Statement's "load-bearing property #1". But that sentence
is currently presented as the paper's methodological centrepiece, and the manuscript twice instructs
the reader that reproducing it is mandatory.

## 4. Provenance: the artifact map resolves to files that are not in the submission

**R5-02 (major).** Appendix A.14 Table 12 is an evidence-identifier→path map, and the Reproducibility
Statement says every claim is "bound to a machine-readable record". I checked. The snapshot contains
**exactly two** evidence files: `evidence/build_record.json` and `evidence/claim_evidence_map.tsv`.
Every single artifact named in Table 12 is absent: `floor_winners_curse_calibration.json` (E-CAL, the
"single machine-readable source" for the nine calibration rows and the file whose sha256 prefix
`275112623d05` is printed in Table 10's caption), `heal_readout_v2_permutation_null.json` (E-D, all
27 v2 cells), `construct_nulls_length_unit.json` (E-H), `R7_BOOTSTRAP_P_FIX.md` (E-E),
`SECOND_MC_BENCHMARK_VERDICT.md`, `POWER_WALL_VERDICT.md`, the whole `EVIDENCE_PACK.md` that E-A/B/I/K
point into, and both named scripts `code/emit_tab_construct_nulls.py` and
`code/check_prose_vs_evidence.py`. The shipped `claim_evidence_map.tsv` additionally cites
`evidence/s2_02_stratified_ordering.json` and `evidence/s2_03_symmetric_inference.json`, also absent.

Consequence for my lens: the Reproducibility Statement's headline verification — "a checker walks all
numerals in the source and requires each to be an exact match to a value in the evidence set … it
reports 610 numerals with none unresolved" — is **unfalsifiable from the submission**. I cannot run
the checker, cannot see the evidence set it checks against, and cannot confirm that 610 is the right
count. This is precisely the "artifact described but not shown" failure my role is asked to detect. It
is also *why* R5-01 survived: a checker that verifies "printed number equals stored number" cannot
catch "stored number came from the wrong null", and there is no independent recomputation of the
calibration in the package. Note that the two shipped files are *good* artifacts — the claim map is a
genuinely useful adjudication ledger — which makes their loneliness more conspicuous, not less.

## 5. The build record certifies a different PDF than the one shipped

**R5-03 (major, and it is the reason §6's errors are still in the paper).** `evidence/build_record.json`
declares `build_gate_pass: true`, `pdf_sha256: 56a376e128c3…`, `pdf_bytes: 355196`, `pdf_pages: 22`.
The shipped `manuscript/main.pdf` has `sha256 1fbaaf9983220f…`, **366583 bytes, and 24 pages** (I
counted three ways: object-stream inflation of `/Type /Page`, the `/Count 24` node, and pymupdf's
`page_count`). File mtimes agree with the direction of the discrepancy: build record 06:48, PDF
10:13, same day. So the build record was produced against an *earlier* compile and then shipped
alongside a later, differently-paginated PDF. Every diagnostic in that record — `n_overfull_hbox: 0`,
`n_undefined_references: 0`, `n_latex_errors: 0`, `build_gate_pass: true` — therefore certifies an
artifact the reviewer does not have.

To be even-handed I re-derived what I could from the shipped PDF directly: it does compile cleanly
(no `??` markers, all 12 tables present with captions, no text block exceeding the ICLR 504 pt right
text edge on any of 24 pages). And the record's own `pdf_visually_inspected: false` note is an
admirable disclosure. But a provenance record that does not hash the shipped artifact is not a
provenance record, and the 22→24 page drift means the page-budget engineering the source comments
describe (main text ends where? — in the shipped PDF `REFERENCES` begins on page 9, Ethics page 10,
Reproducibility page 11, appendix page 11 onward) was validated on a different object than the one
submitted.

## 6. Three prose numbers that do not match the tables they cite

These are the spot-checks my lens owes the panel. I checked well more than three; most passed (§2).
These three failed.

**R5-04 (major) — Appendix A.1.1: "The `credit` value is 4.6× the `wrong` value."** From Table 2,
`credit` = 0.532164 and `wrong` = 0.125914, so the ratio is **4.2264**, not 4.6. This is not a
rounding error; it is 0.37 wide. What makes it more than cosmetic is where 4.6 *does* come from:
credit/floor = 4.5638 ≈ 4.6. And credit/item-avg-chance = **4.7996 ≈ 4.80** — and the paper's own
ledger row 16 prohibits resurrecting "4.8×". So a sentence in the appendix appears to carry a ratio
computed against the wrong denominator, in a paper whose ledger explicitly polices that family of
ratios. The same sentence's other two numbers (40.6 pp span, 32.5 pp over intact 0.207613) both check
out exactly, which suggests a stale edit rather than a systematic error.

**R5-05 (minor→major) — Table 1 caption: "the +0.43–+2.60 pp gaps there".** "There" is the five
inside-noise constructs. Their Table 1 gaps are OpenBookQA 2.600, ARC-Easy 1.625, ARC-Challenge 1.520,
CommonsenseQA 0.885, PIQA **0.490**. The range is +0.49–+2.60; **no row has a 0.43 gap.** The upper
endpoint is right, the lower is not. In a paper that scopes a claim by exactly this range, a range
endpoint that matches no row in its own table is a provenance failure, not a typo — and it is the kind
of thing the missing prose-checker was supposed to catch.

**R5-06 (minor) — Appendix A.1.2: "`split` changes by as much as 2.02 points across the studied
tasks."** The largest char-vs-token `split` movement available in Table 2 is ARC-Challenge
|0.283902 − 0.274104| = **0.980 pp**; OpenBookQA 0.450, Winogrande 0.947. 2.02 is not attainable from
any Table-2 pair. The accompanying `credit` range in the same sentence, 14.53–35.20, is also not
reproducible from Table 2 (the three Table-2 credit movements are 22.800, 19.624, 35.201): the upper
endpoint matches Winogrande exactly, the lower matches nothing. Both numbers presumably come from a
wider task set than Table 2 shows — which is a legitimate possibility, but then the sentence cites a
table that cannot support it and the wider set is not shipped.

## 7. Two further statistical-integrity items

**R5-07 (major) — `recovery_fraction` is 0/0 exactly where it is used as a gate.** §3.2 defines
`recovery_fraction = Δ_perm / Δ_max` and makes "≥ 0.10 × the same-family intact anchor" a hard
materiality bar; §5.1 uses it to demote the paper's one above-floor exception. But the paper also
proves (correctly, and I verified the algebra numerically on a two-stratum toy) that a pure constant
emitter has Δ_perm = 0 — and for that same emitter Δ_max − acc-hat = 0 as well, since permuting a
constant multiset cannot improve alignment. So on the exact class of arm the paper cares most about,
the gate statistic is **0/0**, undefined rather than zero. For the near-constant regime it actually
reports (Qwen3 k14 emits A on 94.6% of items) the denominator is small and the ratio is a quotient of
two near-zero percentage-point quantities, reported as `0.049` with **no interval**, then compared to
an anchor `0.0545` also reported with no interval. Two point estimates in the third decimal place of a
ratio of small differences decide a published label. Separately, the text says 0.049 is "only 9.1% of
the intact-family anchor" — 0.049/0.0545 = **89.9%**, so either the ratio or the anchor in that
sentence is wrong; if the intended anchor is a different quantity it is not identified in the text.
(0.049/0.0545 = 0.899 and the text's "9.1%" would need an anchor near 0.538. I could not resolve which
is meant, and the E-D file that would settle it is not shipped.)

**R5-08 (minor) — "all 21 evaluated cells are powered", 21 cells never enumerated.** The abstract and
§4 both assert 21 MMLU-Pro letter-floor cells with half-widths 0.083–0.968, and §A.1.3 asserts the
floor is "bit-identical across all 21 MMLU-Pro cells". Table 4 shows 12 (with Llama-3 collapsed into
one range row) and Table 7 shows 27 (a different, overlapping set). The 21-cell set is never listed,
and 0.968 appears nowhere else in the manuscript. Similarly the 0/60, 25/60, 52/60, 7/60 off-MMLU
counts have no per-cell table. Appendix A.4 promises that "any damaged arm excluded from that
denominator is named at the point of exclusion together with its own floor delta" — but `shortgpt16`,
a designated damaged OLMo-2 arm per §4, appears in Table 7 with a large `+4.054` pp signal and is
**not** in the 15-cell denominator, and I found no sentence naming it as excluded with its floor
delta. The self-imposed disclosure rule is violated by the one arm whose inclusion would most change
the reading.

## 8. Citation integrity

Better than typical, with one localised overreach. I verified venues on OpenReview (positive control
passed; `api2.openreview.net` returned `venueid` for known-good queries): `cho2026choices` = ICLR 2026
Poster ✓, `oostermeijer2026length` = ICML 2026 ✓, `bean2025measuring` = NeurIPS 2025 D&B ✓,
`zheng2025cheating` = ICLR 2025 Oral ✓, `arcon2026metalinguistic` = arXiv 2602.02182 ✓. I fetched two
primary sources and checked local support:

- `arcon2026metalinguistic` **supports** the opening sentence precisely — its abstract states "all
  models perform above chance, they fail to outperform the majority-class baseline". Good citation.
- `bean2025measuring` (arXiv 2511.04703, 29 pp) **partially contradicts** its local sentence. The
  intro says it "offers 27 actionable checklist items". The paper's Appendix A checklist has **28**
  box items (3+3+3+4+3+4+4+4 across eight recommendation groups); 27 is the count of box glyphs in the
  *main-text* §5 rendering, so the number is defensible but the item count is version-dependent and
  the paper cites it as if canonical. The load-bearing half of the sentence — "none asks authors to
  report a null, a chance level, or a constant predictor" — I checked directly: the strings "null",
  "majority", and "constant" appear **zero times** in that paper, and "chance" only in reuse and
  inter-rater-agreement contexts. So the substantive claim holds. Fix the count, keep the claim.

## 9. What I could not verify

Anything requiring the unshipped evidence: the 610-numeral checker result; the actual `n_opt`
histogram of MMLU-Pro (I bounded it instead, which is why R5-01 rests on a deterministic inequality
rather than on the true histogram); the 21-cell power set; the off-MMLU 60-cell counts; the ARC-Challenge
"five of six OLMo-2 arms" convention flip; the 12/12 char/token self-test; whether E-CAL's sha256
prefix `275112623d05` matches anything; the Cho et al. camera-ready diff the limitations honestly flag
as blocked. I did not run any author script (per instructions).

## 10. Score and what would move it

Overall **5**. The paper is a good idea, competently and unusually honestly executed, with verified
internal arithmetic — and its single most emphasised methodological contribution rests on a
misspecified null that its own appendix contains the material to refute, while the evidence package
that would have caught this is not in the submission. Under this rubric that is "at least one
material issue prevents a positive decision in the current form."

Ceiling without new experiments: **6.5**. The fix is pure re-analysis on data the authors already
have — recompute the calibration with gold drawn uniform over each item's legal letters, restate
which constructs survive, correct 4.2264/0.49/0.980, ship E-CAL + E-D + the two scripts, re-emit
`build_record.json` against the shipped PDF, enumerate the 21 cells, name `shortgpt16` at its
exclusion, and either give `recovery_fraction` an interval or replace the gate with something defined
at Δ_max = 0. Because a corrected calibration will most likely *narrow* H2 (three survivors → two),
predicted post-revision score is **6.0**: the paper becomes trustworthy but its headline gets smaller,
and this is exactly the trade the paper's own ledger says it prefers.

Would lower it: if MMLU-Pro's floor also fails the corrected null and the authors keep the abstract's
current framing; if the missing artifacts do not exist rather than merely being unshipped; if the 21
vs 12 vs 27 cell sets turn out to be different measurement protocols pooled under one label.
