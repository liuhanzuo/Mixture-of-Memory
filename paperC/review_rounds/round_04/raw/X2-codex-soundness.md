```json
{
  "reviewer_id": "X2-codex-soundness",
  "round": 4,
  "snapshot_sha256": "7fcb9ccc55c5c1d6ad1de868215b1d9253d0695a2f996340226aa6a6fa10db5a",
  "role": "technical_soundness",
  "overall_score": 4,
  "confidence": 4,
  "recommendation": "reject",
  "dimension_scores": {
    "novelty": 3,
    "significance": 3,
    "technical_soundness": 2,
    "experimental_rigor": 2,
    "clarity": 3,
    "reproducibility": 1,
    "citation_integrity": 3,
    "limitations_responsible_claims": 4
  },
  "paper_summary": "The paper proposes calibrating multiple-choice evaluation against construct-specific input-blind references rather than nominal chance. It distinguishes an arm-independent best-constant floor, intended to contextualize cross-arm scores, from an arm-conditional within-option-count permutation statistic intended to measure alignment beyond an arm's output marginal. It then applies these read-outs to structurally damaged model arms, content-versus-letter interfaces, variable-option-count benchmarks, and one bf16-versus-fp32 comparison.",
  "strongest_verified_contribution": "The cleanest verified contribution is the algebraic and arithmetic treatment of the stratified permutation statistic. Pure within-stratum constant emitters have zero Delta_perm, all 27 printed Delta_perm values equal 100 times (accuracy minus permutation expectation) within 0.0005 percentage points, and the stated v1/v2 ordering slack is exactly 36/12032 = 0.2992 percentage points.",
  "strengths": [
    "The paper makes a useful distinction between an arm-independent reference and an arm-conditional independence null, and it appropriately avoids claiming Cohen's kappa or constant-emitter invariance as new.",
    "Several central printed quantities re-derive correctly: the MMLU-Pro floor is 1403/12032 = 0.116606; its gaps are 1.6606 and 0.5729 percentage points against the two printed chance lines; the point-estimate counts are 10/12 non-OLMo arms above item-averaged chance and 3/12 above the floor; and 2532/14042 = 18.0316 percent.",
    "The v1/v2 ordering discussion correctly recognizes that stratification breaks the simple unstratified bound and supplies a valid counterexample and regularity condition.",
    "The manuscript is unusually candid about post-hoc analysis, multiplicity, regime confounding, earlier implementation defects, retracted claims, and the necessary-not-sufficient nature of input-blind controls.",
    "The paper is generally well organized, and the current PDF is readable with clearly labeled main and appendix material."
  ],
  "issues": [
    {
      "id": "TS-VALIDITY-GATE",
      "severity": "major",
      "location": "Abstract; Introduction, the two-question list and final paragraph; Section 3.1 decision rule; Discussion paragraphs 'A reporting rule' and 'The letter-floor and permutation read-outs partition scope'; Appendix reporting checklist",
      "dimension": "technical_soundness, limitations_responsible_claims",
      "description": "The paper alternates between the defensible statement that a below-floor arm has not demonstrated accuracy above the best constant and the much stronger claim that the floor is a necessary validity condition that certifies mutual comparability or must be passed before arms can be compared. The latter does not follow from acc <= f_const. Two arms can have a real and reproducible accuracy difference while both lie below the floor, and a systematically anti-aligned predictor can contain substantial item information despite very low accuracy. The paper's own v2 analysis demonstrates that below-floor status is not equivalent to absence of item dependence. Thus the floor is useful absolute context, but the manuscript has not established it as a validity certificate for relative arm comparisons.",
      "proposed_fix": "Narrow the rule to the claim actually implied by the statistic: a below-floor score cannot be described as performance above the strongest specified constant reference. Remove language that it certifies or is necessary for mutual comparability unless a formal measurement criterion or empirical invariance argument is supplied.",
      "verification_test": "The revised manuscript should contain no inference from Delta_floor <= 0 alone to invalidity or incomparability of arm differences. Test the wording against counterexamples containing two distinct below-floor predictors and a perfectly anti-aligned predictor."
    },
    {
      "id": "TS-CONTENT-DEFINITION",
      "severity": "major",
      "location": "Section 3.1 definition of the content-side longest-option family; Section 4 paragraph 'Interfaces and scoring protocol'; Section 5.2 claims using content_norm; Table 2 and its caption",
      "dimension": "technical_soundness, clarity, reproducibility",
      "description": "The content construct is not consistently defined. Section 4 says scoring is summed token log-probability followed by argmax, while Sections 2 and 5 use the name content_norm and discuss length-normalized likelihood, and the content null is a longest-option predictor. No equation maps content_norm to either summed or per-token-normalized likelihood. Consequently it is impossible to determine which quantity produced the ARC-Easy 38.76-point gap, residual-fraction claims, or longest-option floors.",
      "proposed_fix": "Provide explicit equations and field mappings for every content score. State whether scores are sums, means, or another normalization, and define the corresponding input-blind family under exactly that convention. Recompute all content-side claims if the current prose does not match the actual records.",
      "verification_test": "From shipped per-item candidate log-probabilities and token lengths, an independent implementation should reproduce each content prediction, content accuracy, longest-option null, ARC-Easy paired gap, and residual fraction under the printed equation."
    },
    {
      "id": "STAT-CONSTRUCT-NULL",
      "severity": "major",
      "location": "Table 1 MMLU-Pro rows and caption; generated Table 10 rows and caption; Section 3.3 and Appendix A.1.4; evidence identifier E-CAL",
      "dimension": "technical_soundness, experimental_rigor",
      "description": "The item-averaged MMLU-Pro chance row is calibrated using a null incompatible with its legal-option structure. The table's uniform-ten-label simulation gives E[max marginal] about 0.10446. Under the construct-respecting null y_i uniform over the legal labels of item i, however, E[m_A] already equals mean(1/n_opt) = 0.110877, so E[max_L m_L] must be at least 0.110877. The reported calibration therefore cannot justify the duplicated p < 1e-5 for the item-averaged-chance interpretation. The same concern applies to rows whose chance differs from 1/k. There is also a local tail inconsistency: PIQA's printed floor is 928/1838, for which the exact balanced binary probability P(max count >= 928) is 0.691725, not the reported 0.658.",
      "proposed_fix": "Calibrate variable-option constructs by sampling each item's gold label uniformly from its actual legal option set, or clearly state that the uniform-k balance test answers a different question and do not use it to calibrate item-averaged chance. Regenerate the PIQA tail using the stated greater-than-or-equal threshold.",
      "verification_test": "Using the frozen n_opt vector, independently regenerate E[f_hat], q95, and tail probabilities under the legal-option null. The MMLU-Pro expected maximum must be at least 0.110877. An exact binary calculation must reproduce the corrected PIQA p-value."
    },
    {
      "id": "STAT-DEPENDENT-COUNT",
      "severity": "major",
      "location": "Section 5.1 final multiplicity paragraph; Appendix A.4 multiplicity paragraph; Appendix A.5 'Multiplicity in the two-reference comparison'",
      "dimension": "technical_soundness, experimental_rigor",
      "description": "The reported 0.0196 probability for observing at least 3 rejections among 12 tests is arithmetically correct for Binomial(12, 0.05), but that model assumes independent Bernoulli tests. The paper explicitly states that these cells share items, nest arms, share a null, and are neither independent nor exchangeable. The binomial calculation therefore does not validly rescue an aggregate significance claim after BH and Bonferroni yield 0/12 discoveries.",
      "proposed_fix": "Either report 3/12 descriptively without an aggregate p-value, or construct a joint global-null test that preserves cross-arm dependence by applying the same item resampling or null randomization to all 12 cells.",
      "verification_test": "Under simulated or permuted joint global-null data preserving the observed cross-arm dependence, the count-level test should have empirical type-I error at most 0.05."
    },
    {
      "id": "DESIGN-DESIGNATED-DENOMINATOR",
      "severity": "major",
      "location": "Section 4 paragraphs 'Models and structural damage' and 'Designated damaged cells'; Section 5.1 headline 15-cell count; Appendix A.4; Tables 5, 6, and 7",
      "dimension": "experimental_rigor, technical_soundness",
      "description": "The written designation rule does not reproduce the headline denominator. Section 4 lists OLMo-2 keep8, keep10, keep12, keep14, and shortgpt16 as structurally damaged arms, but the 15-cell headline includes only three OLMo-2 arms plus 12 non-OLMo arms. Appendix A.4 promises that every excluded damaged arm is named at the point of exclusion with its floor delta, but keep14 and shortgpt16 are not so identified. This is outcome-relevant: shortgpt16 has printed accuracy 0.153341, which is 3.674 points above the 0.116606 floor, and has an item-level-signal v2 verdict.",
      "proposed_fix": "Publish an exhaustive, outcome-independent designation table covering every evaluated arm, with a mechanically checkable inclusion rule and explicit rationale for exclusions. Recompute all 14/15, 10/15, 0/60, and 25/60 aggregates using the operational rule and provide sensitivity counts including every structurally damaged arm.",
      "verification_test": "An independent script using only arm metadata should reproduce every denominator before loading scores. No arm satisfying the printed rule may be excluded, and all sensitivity counts must be reported."
    },
    {
      "id": "PROV-MISSING-EVIDENCE",
      "severity": "major",
      "location": "MANIFEST.json; evidence/claim_evidence_map.tsv; Reproducibility Statement; Appendix A.15 and Table 12; evidence/build_record.json",
      "dimension": "reproducibility, experimental_rigor",
      "description": "The snapshot ships only build_record.json and claim_evidence_map.tsv as evidence, while the claim map and captions rely on absent records including floor_winners_curse_calibration.json, s2_03_symmetric_inference.json, s2_02_stratified_ordering.json, E-B/E-D/E-F per-item records, and emitters. Thus the central bootstrap intervals, permutation p-values, 3/12 count, 0/60 aggregate, 52/60 power classification, recovery fractions, and integrity assertions cannot be independently checked. The shipped build record is also stale: it records a 22-page, 355196-byte PDF with hash 56a376..., while the frozen PDF is 24 pages, 366583 bytes, with manifest hash 1fbaaf....",
      "proposed_fix": "Include every evidence file and emitter referenced by the manuscript, preferably with per-item gold labels, predictions, legal-option counts, candidate scores, and immutable configurations. Regenerate the build record against the exact frozen PDF.",
      "verification_test": "In a clean directory containing only manifest-listed files, one command should regenerate every table, aggregate, interval, p-value, verdict, and the exact PDF hash without accessing excluded repository paths."
    },
    {
      "id": "STAT-POWER-NOMENCLATURE",
      "severity": "major",
      "location": "Section 4 paragraph 'Power design'; Table 4; Section 5.1 off-MMLU interpretation; Discussion paragraph 'Power is part of construct validity'",
      "dimension": "experimental_rigor, clarity",
      "description": "The reported 'power analysis' appears to classify cells by whether an observed 95 percent interval half-width is smaller than a 1.389-point target effect. That is an achieved-resolution criterion, not statistical power. No target power, alternative distribution, discordance model, or rejection-probability calculation is specified. For example, ARC-Easy's 1.305-point half-width corresponds to only about 55 percent two-sided normal-approximation power for a true 1.389-point effect, despite being labeled 'yes, borderline'.",
      "proposed_fix": "Rename the analysis as achieved precision or effect-resolution throughout, or compute power using an explicit paired-outcome model, alpha level, target effect, and target such as 80 percent.",
      "verification_test": "For every cell labeled powered, simulate paired outcomes under the stated alternative and show that empirical rejection probability meets the declared target; otherwise use only interval-exclusion language."
    },
    {
      "id": "DEF-RESAMPLING-MATERIALITY",
      "severity": "major",
      "location": "Section 3.1 Equation 2; Section 3.2 Equations 3-6; Section 4 statistics paragraph; Section 5.1 Qwen3 k14 materiality claim; Section 5.3 Llama-2 anchor claim",
      "dimension": "technical_soundness, reproducibility",
      "description": "Two inferential definitions are incomplete. First, v1 is defined using a maximum over labels, but the bootstrap is described as comparison with a deterministic per-item null predictor; the paper does not state whether the maximizing label is reselected in every resample. Holding the original label fixed estimates a different statistic from bootstrapping the maximum. Second, Delta_max, which controls recovery_fraction, 'trace signal', 'immaterial', and 'anchor blocked', is only described as the best reassignment of the prediction multiset; its within-stratum and legal-label constraints are not defined.",
      "proposed_fix": "Specify whether v1 reselects the maximizing constant inside every resample and justify that target. Give a closed-form or algorithmic definition of Delta_max, including option-count strata, label legality, tie handling, and the exact intact-anchor normalization. Print Delta_max and recovery_fraction for every relevant row.",
      "verification_test": "Independently regenerate fixed-label and reselected-maximum v1 intervals and report their differences. From published stratum-level counts, independently reproduce all recovery fractions and the Qwen3 k14 and Llama-2 materiality verdicts."
    },
    {
      "id": "CLAIM-FP32-SCOPE",
      "severity": "minor",
      "location": "Abstract final empirical sentence; Section 5.4 'Full precision falsifies the numerical-tie mechanism'",
      "dimension": "limitations_responsible_claims, technical_soundness",
      "description": "One OLMo-2 keep8 result on MMLU shows that removing exact bf16 top-two ties does not improve that cell's accuracy. It does not broadly rule out numerical ties or other finite-precision effects as explanations across the measurement failures studied. The interval supports a cell-specific bound, not a family-general mechanism claim.",
      "proposed_fix": "Narrow the conclusion to this model, arm, benchmark, and exact-tie intervention, or define an equivalence margin and test a representative set of affected cells.",
      "verification_test": "The revised conclusion should be logically implied by the printed interval, for example that this intervention improves this cell by less than approximately 0.33 percentage points at the stated confidence level."
    }
  ],
  "score_ceiling_under_current_evidence": 6,
  "predicted_score_after_required_changes": 6,
  "evidence_that_would_raise_score": [
    "A complete manifest-contained artifact that regenerates all tables, intervals, p-values, denominators, and verdicts from per-item records.",
    "A construct-respecting legal-option calibration showing that the MMLU-Pro floor remains above its appropriate variable-option null with a correctly calibrated uncertainty statement.",
    "A joint dependence-preserving analysis showing that the chance-versus-floor comparison remains non-null after replacing the invalid binomial count calculation.",
    "An exhaustive designation table showing that the main prevalence claims are robust to inclusion of every arm satisfying the written damage criterion.",
    "A precise content-scoring equation and regeneration demonstrating that the ARC-Easy gap, content floors, and residual fractions use the same scoring convention.",
    "Claim narrowing that treats the best-constant floor as an absolute reference rather than an unsupported certificate of arm comparability."
  ],
  "evidence_that_would_lower_score": [
    "The legal-option null places the MMLU-Pro floor inside ordinary selection noise or materially weakens its tail probability.",
    "Including omitted damaged arms substantially reduces the 14/15, 10/15, or 0/60 prevalence claims.",
    "A clean regeneration disagrees with the printed bootstrap intervals, permutation p-values, recovery fractions, or aggregate counts.",
    "The actual content computation uses a different normalization from the manuscript and correcting it reverses the ARC-Easy or residual-fraction conclusions.",
    "A dependence-preserving global-null analysis shows that observing 3/12 chance-side rejections is unsurprising.",
    "The absent evidence reveals outcome-dependent designation or resampling choices."
  ],
  "review_limitations": [
    "I read only the frozen submission snapshot and the required rubric and protocol; I did not inspect any excluded paperC path, history, prior review, or author-side note.",
    "The per-item records, legal-option vector, calibration JSON, permutation records, bootstrap records, and emitters referenced by the manuscript are absent from the snapshot, so their associated results are unverified rather than refuted.",
    "I did not run the authors' checker scripts, as explicitly prohibited.",
    "The exact tail probability under the construct-respecting MMLU-Pro legal-option null cannot be computed without the missing n_opt vector, although the contradiction E[max marginal] >= 0.110877 > 0.104460 is algebraically sufficient to establish that the printed simulation is a different null.",
    "External cited papers were not available under the review allowlist, so load-bearing local claims about the Bean checklist, Cho overlap, and the absence of variable-k stratified nulls in prior kappa literature remain unverified.",
    "All 34 manifest-listed files matched their individual recorded sizes and SHA-256 hashes; the stale build record concerns its mismatch with the current manifest-listed PDF, not corruption of those files."
  ]
}
```

## PROSE REVIEW

### Summary and overall assessment

This paper argues that multiple-choice evaluation should report a construct-specific input-blind reference rather than relying only on nominal chance. It separates:

1. **v1:** an arm-independent best-constant floor; and  
2. **v2:** an arm-conditional permutation expectation preserving prediction marginals within option-count strata.

This is a useful measurement distinction, and the paper is commendably candid about confounds, post-hoc analysis, multiplicity, retracted claims, and implementation failures. However, the current submission has several decision-relevant soundness problems. Most importantly, its strongest interpretation of v1 as a validity gate is not logically established; its MMLU-Pro item-averaged calibration uses a null incompatible with variable legal option counts; its aggregate \(p=0.0196\) assumes independence that the paper explicitly denies; its designated-cell denominator is not reproduced by its written rule; and nearly all load-bearing evidence files are absent from the frozen artifact.

I therefore recommend **reject, score 4/10**, with confidence 4. The paper could plausibly reach a weak-accept level using existing data—without new model experiments—if the analyses are corrected, the claims narrowed, and the complete evidence is shipped.

### Quantities I independently re-derived

I checked more than the requested two central quantities:

- The MMLU-Pro floor is
  \[
  1403/12032=0.1166057,
  \]
  matching 0.116606. Its gaps are \(1.6606\) points against 0.10 and \(0.5729\) points against 0.110877.

- Using the per-cell accuracies in Tables 5 and 7, I recover the point-estimate counts of **10/12** non-OLMo arms above item-averaged chance, **12/12** above naive 0.10, and **3/12** above the best-constant floor.

- For every one of the 27 rows in Table 7,
  \[
  100(\mathrm{acc}-\widehat{\mathrm{acc}})
  \]
  matches the printed \(\Delta_{\mathrm{perm}}\) within 0.0005 percentage points, i.e. the expected rounding error.

- The v1/v2 ordering slack is exactly
  \[
  100(1439-1403)/12032=0.299202
  \]
  percentage points.

- The fp32 decision-change rate is
  \[
  2532/14042=18.0316\%,
  \]
  matching 18.03%.

- The paper’s binomial arithmetic is also correct:
  \[
  \Pr\{\mathrm{Binomial}(12,0.05)\ge3\}=0.019568.
  \]
  The problem is not arithmetic but the invalid independence model.

- I also checked the PIQA calibration. Its floor is \(928/1838=0.5048966\). Under an exactly balanced binary null, the exact probability that the maximum count is at least 928 is **0.691725**, not the printed 0.658.

### Strengths

1. **Useful separation of two questions.** The distinction between a common arm-independent reference and an arm-conditional independence null is conceptually helpful.

2. **Correct v2 algebra.** The constant-collapse identity and the relationship to the numerator of Cohen’s \(\kappa\) are correct. The paper also appropriately disclaims novelty for these identities.

3. **Careful treatment of stratification.** The paper correctly observes that within-\(n_{\mathrm{opt}}\) stratification invalidates the simple unstratified ordering bound. The 36-item counterexample and regularity condition are useful.

4. **Responsible disclosure.** The discussion of post-hoc status, regime confounding, multiplicity, underpowered small benchmarks, and prior implementation defects is substantially better than typical.

5. **Readable presentation.** Despite its density, the argument is organized and the current PDF is legible.

### Major issues

#### TS-VALIDITY-GATE — Major — Technical soundness

**Location:** Abstract; Introduction’s two-question list; Section 3.1; Discussion’s first two paragraphs; reporting checklist.

The paper’s narrower statement is correct: an arm below \(f_{\mathrm{const}}\) has not demonstrated accuracy above the strongest specified constant predictor. The paper repeatedly upgrades this into a stronger statement that v1 is a necessary “validity condition,” must be passed before arms are compared, or “certifies mutual comparability.”

That stronger conclusion does not follow. Two predictors may have a real, stable difference while both are below the best constant. A perfectly anti-aligned predictor can also contain maximal item information despite very low accuracy. Indeed, the paper’s own v2 analysis shows that below-floor status cannot be interpreted as absence of item dependence.

**Fix:** Present the floor as an absolute interpretive reference: below-floor performance cannot be called above-best-constant capability. Remove claims that it certifies or is necessary for comparability unless a formal measurement argument is supplied.

**Verification:** The revised rule should behave correctly on two distinct below-floor predictors and on a perfectly anti-aligned predictor; it must not declare their scores intrinsically incomparable solely from v1.

---

#### TS-CONTENT-DEFINITION — Major — Technical soundness and reproducibility

**Location:** Section 3.1; Section 4 “Interfaces and scoring protocol”; Section 5.2; Table 2.

The content score is not consistently defined. Section 4 says all candidates are scored with **summed token log-probability**, but the results use `content_norm`, the related-work discussion concerns length-normalized likelihood, and the null is a longest-option heuristic. No equation defines `content_norm` or connects it to the stated summed score.

This prevents evaluation of the ARC-Easy \(+38.76\)-point claim, residual-fraction calculations, and longest-option floors.

**Fix:** Give equations for summed and normalized content scores, identify the exact field used in each result, and ensure that the input-blind null matches that scoring convention.

**Verification:** Starting from per-item candidate log-probabilities and lengths, an independent script should reproduce all content predictions, floors, paired gaps, and residual fractions.

---

#### STAT-CONSTRUCT-NULL — Major — Technical soundness and experimental rigor

**Location:** Table 1; generated Table 10; Section 3.3; Appendix A.1.4; E-CAL.

The item-averaged MMLU-Pro row uses the wrong calibration null. Its simulation samples ten labels uniformly, yielding
\[
E[\widehat f]\approx0.10446.
\]
But under the legal-option null corresponding to item-averaged chance,
\[
Y_i\sim\mathrm{Uniform}\{A,\ldots,n_{\mathrm{opt},i}\},
\]
the expected A marginal alone is
\[
E[m_A]=\mathrm{mean}(1/n_{\mathrm{opt}})=0.110877.
\]
Therefore
\[
E[\max_L m_L]\ge 0.110877,
\]
so it cannot equal 0.104460. The duplicated \(p<10^{-5}\) does not calibrate the item-averaged-chance interpretation.

There is also a confirmed local threshold problem in PIQA: the exact tail is 0.691725, not 0.658, under the stated greater-than-or-equal definition.

**Fix:** Simulate each item using its actual legal label set. If uniformity over ten nominal labels is intended as a separate hypothesis, label it as such and do not use it to calibrate item-averaged chance.

**Verification:** Regenerate \(E[\widehat f]\), \(q_{95}\), and \(p\) from the actual \(n_{\mathrm{opt}}\) vector. The expected MMLU-Pro maximum must be at least 0.110877, and the exact PIQA calculation must match.

---

#### STAT-DEPENDENT-COUNT — Major — Technical soundness

**Location:** Section 5.1 and Appendix A.5.

The value 0.0196 is calculated from a Binomial\((12,0.05)\) model. Yet Appendix A.4 explicitly says the tests share items, nest arms, share a null, and are neither independent nor exchangeable. The assumptions needed for that binomial distribution are therefore contradicted by the manuscript itself.

The paper correctly reports that BH and Bonferroni retain no cells. The unsupported binomial calculation cannot then be used to rescue aggregate significance.

**Fix:** Report 3/12 descriptively, or construct a joint null using the same resampled/permuted item vector across all arms.

**Verification:** Demonstrate correct type-I error under a global null that preserves the observed cross-arm dependence.

---

#### DESIGN-DESIGNATED-DENOMINATOR — Major — Experimental rigor

**Location:** Section 4 model and designation paragraphs; Section 5.1; Appendix A.4; Tables 5–7.

The written designation rule does not reproduce the 15-cell denominator. Section 4 lists OLMo-2 `keep8`, `keep10`, `keep12`, `keep14`, and `shortgpt16` as structurally damaged arms. The headline includes only three OLMo-2 arms. Contrary to Appendix A.4, `keep14` and `shortgpt16` are not named at the point of exclusion with their floor deltas.

This omission is outcome-relevant. `shortgpt16` has accuracy 0.153341, or approximately \(+3.674\) points over the MMLU-Pro floor, and Table 7 labels it as carrying item-level signal.

**Fix:** Provide an exhaustive designation ledger fixed without looking at outcomes. Recompute the headline counts under that rule and give sensitivity analyses including every structurally damaged arm.

**Verification:** A script operating only on arm metadata—not scores—must reproduce each denominator.

---

#### PROV-MISSING-EVIDENCE — Major — Reproducibility

**Location:** MANIFEST.json; claim-evidence map; Reproducibility Statement; Appendix A.15/Table 12; build record.

The artifact claims unusually strong provenance, but the frozen snapshot contains only two evidence files: the build record and claim-evidence map. The central referenced records—E-CAL, E-B, E-D, E-F, symmetric-inference JSON, ordering JSON, per-item predictions, and emitters—are absent.

Consequently I could not verify:

- the 3/12 chance-side interval count;
- bootstrap or permutation \(p\)-values;
- 0/60 and 25/60 aggregates;
- 52/60 power classification;
- recovery fractions;
- bootstrap/permutation agreement;
- truncation and shard-integrity assertions.

These results are **unverified, not refuted**.

The build record also does not describe the frozen PDF. It records 22 pages, 355196 bytes, and hash `56a376...`; the manifest-listed PDF has 24 pages, 366583 bytes, and hash `1fbaaf...`.

**Fix:** Ship all cited records and emitters, with per-item predictions and metadata, and regenerate the build record for the exact snapshot.

**Verification:** A clean, offline regeneration using only manifest-listed files must reproduce the tables and exact PDF hash.

---

#### STAT-POWER-NOMENCLATURE — Major — Experimental rigor

**Location:** Section 4 “Power design”; Table 4; Section 5.1; Discussion.

The reported “power” analysis appears to classify a cell as capable of detecting an effect whenever its observed 95% interval half-width is smaller than 1.389 points. This is an achieved-resolution criterion, not power. No target rejection probability, alternative distribution, or paired discordance model is given.

For example, treating the ARC-Easy half-width of 1.305 points with a normal approximation gives only about 55% two-sided power for a true 1.389-point effect—not a conventional powered design.

**Fix:** Rename the analysis “achieved precision” or “effect resolution,” or conduct an actual paired power analysis with a declared target such as 80%.

**Verification:** For each “yes” row, simulation under the stated alternative should attain the declared rejection probability.

---

#### DEF-RESAMPLING-MATERIALITY — Major — Technical soundness

**Location:** Equations 2–6; Section 4 statistics paragraph; Qwen3 `k14` and Llama-2 anchor claims.

Two load-bearing choices are underdefined.

First, v1 is a maximum over labels, but the bootstrap is described using a deterministic per-item null predictor. It is not stated whether the maximizing label is reselected in each resample. A fixed-\(L^\star\) bootstrap and a bootstrap of \(\max_L m_L\) estimate different quantities.

Second, \(\Delta_{\max}\), which determines `recovery_fraction`, “trace signal,” “immaterial,” and “anchor blocked,” is not formally defined. In particular, its option-count strata and legal-label constraints are unspecified.

**Fix:** State whether \(L^\star\) is reselected per bootstrap sample. Define \(\Delta_{\max}\) algorithmically, including legality, stratification, and ties, and print its value for each relevant row.

**Verification:** Independently regenerate every recovery fraction and compare fixed-label versus reselected-maximum v1 intervals.

### Minor issue

#### CLAIM-FP32-SCOPE — Minor — Responsible claims

**Location:** Abstract and Section 5.4.

One OLMo-2 `keep8` MMLU result establishes that eliminating exact bf16 top-two ties does not recover accuracy for that cell. It does not generally rule out numerical ties or other finite-precision mechanisms across all studied failures.

**Fix:** Narrow the conclusion to this cell and intervention, or evaluate representative cells under a predefined equivalence margin.

**Verification:** The conclusion should follow directly from the reported interval—for example, an upper bound of approximately \(+0.33\) percentage points on this cell’s improvement.

### Decision rationale

The paper’s **strongest verified result** is the technically careful v2 algebra and stratification analysis. The **weakest evidence-to-importance ratio** is the prevalence/significance package around 14/15, 0/60, and 3/12: the denominator is not operationally reproduced, the aggregate significance calculation uses an invalid independence assumption, and the underlying records are absent.

The paper’s **ceiling without new experiments is approximately 6/10**. Existing per-item data should be sufficient to:

- correct the legal-option calibration;
- run a dependence-preserving joint null;
- define exhaustive denominators;
- formalize content scoring and \(\Delta_{\max}\);
- regenerate the artifact; and
- narrow the validity claim.

The single change that would most improve confidence is a **clean end-to-end regeneration from shipped per-item records using construct-respecting nulls and mechanically fixed denominators**.