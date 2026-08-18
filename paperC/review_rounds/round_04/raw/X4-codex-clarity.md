```json
{
  "reviewer_id": "X4-codex-clarity",
  "round": 4,
  "snapshot_sha256": "7fcb9ccc55c5c1d6ad1de868215b1d9253d0695a2f996340226aa6a6fa10db5a",
  "role": "clarity_presentation",
  "overall_score": 4,
  "confidence": 4,
  "recommendation": "reject",
  "dimension_scores": {
    "novelty": 3,
    "significance": 3,
    "technical_soundness": 2,
    "experimental_rigor": 2,
    "clarity": 3,
    "reproducibility": 2,
    "citation_integrity": 3,
    "limitations_responsible_claims": 3
  },
  "paper_summary": "The paper argues that multiple-choice results should be calibrated against an explicit input-blind null rather than only nominal chance. It distinguishes an arm-independent best-constant floor (v1) from an arm-conditional, option-count-stratified permutation statistic (v2), applies both to structurally damaged model arms, examines length-based content nulls, and reports a precision-control experiment. The paper is unusually candid about post-hoc analysis, confounds, multiplicity, and repaired integrity defects, but several central statistical and conceptual claims remain unsupported or unreconstructable from the submitted artifact.",
  "strongest_verified_contribution": "The clearest verified contribution is the conceptual separation between an arm-independent constant reference and an arm-conditional prediction-marginal null, together with the correct algebraic observation that the stratified permutation statistic is exactly zero for every pure legal constant emitter. The paper also draws unusually explicit novelty boundaries around this statistic.",
  "strengths": [
    "The paper identifies a practically important reporting failure: a constant or nearly constant predictor can exceed a conventional chance line.",
    "The distinction between absolute comparison to an arm-independent floor and item-level alignment relative to an arm's own output marginal is useful.",
    "The derivation of constant-collapse invariance for the stratified permutation statistic is concise and correct.",
    "The discussion of tie conventions, length units, tokenizers, and varying option counts surfaces measurement choices that are often omitted.",
    "The manuscript is commendably candid about post-hoc status, family/regime confounding, multiple testing, withdrawn claims, and repaired analysis defects.",
    "The full-precision control is well motivated and its stated interpretation is appropriately narrower than a causal account of all failures."
  ],
  "issues": [
    {
      "id": "X4-01",
      "severity": "major",
      "location": "Abstract opening and closing claims; Introduction paragraphs 1-4 and numbered question 1; Section 3.1, especially the claim that v1 tests whether an interface is valid for comparing arms; Discussion paragraphs 3-6",
      "dimension": "technical_soundness; significance; limitations_responsible_claims",
      "description": "The evidence supports a warning about absolute capability interpretations, but not the stronger claim that clearing the floor is a necessary gate for pairwise arm comparison. Because the same arm-independent floor is subtracted from every arm, (acc_A - f) - (acc_B - f) equals acc_A - acc_B exactly; v1 cannot alter an arm difference or ordering. A below-floor arm may still participate in a meaningful relative comparison. The manuscript repeatedly moves from 'above chance can be misleading' to 'failure disqualifies arm comparison' without supplying a measurement-theoretic argument or an empirical pairwise comparison that becomes invalid.",
      "proposed_fix": "Narrow the main rule to absolute claims such as 'do not interpret above-chance performance as above-null capability without reporting the floor.' Explicitly state that a common floor leaves pairwise differences invariant. If the pre-comparison-gate claim is retained, define a separate validity criterion and demonstrate why failing it invalidates a concrete arm-comparison claim.",
      "verification_test": "Check that the revised abstract, Introduction, Section 3.1, and Discussion distinguish absolute null-relative interpretation from pairwise comparison. Every retained claim that an arm comparison is invalid should identify a formal criterion or a pairwise conclusion that fails under that criterion."
    },
    {
      "id": "X4-02",
      "severity": "major",
      "location": "Abstract claim that MMLU-Pro has p<10^-5; Introduction main finding 1; Table 1 MMLU-Pro rows and caption; Appendix Table 10; Reproducibility Statement paragraph 3",
      "dimension": "technical_soundness; experimental_rigor; limitations_responsible_claims",
      "description": "The winner's-curse calibration for MMLU-Pro uses a balanced multinomial with nominal k=10 even though the paper emphasizes that item option counts vary from 3 to 10. Under a random legal-answer null, A is legal for every item and has expected frequency mean(1/n_opt)=0.110877, not 0.10. The identical expected maxima and p-values printed for the naive and item-averaged MMLU-Pro rows show that the simulation does not preserve item-specific legal label sets. It can assign labels that are illegal for low-option items. Therefore the reported p<10^-5 evidence that the 0.116606 floor is unusual under the construct-appropriate null is not established.",
      "proposed_fix": "Recompute the floor-estimator distribution by drawing each item's gold label uniformly from that item's legal options, preserving the observed n_opt profile, and recomputing the maximum letter marginal on every draw. Report its mean, quantiles, and p-value separately from the naive ten-way thought experiment, then revise all headline claims accordingly.",
      "verification_test": "The null generator must never produce an illegal label, its expected always-A frequency must reproduce 0.110877, and independent Monte Carlo runs or an exact calculation must agree on the resulting floor distribution. The abstract and tables must reflect whether 0.116606 actually exceeds the corrected 95th percentile."
    },
    {
      "id": "X4-03",
      "severity": "major",
      "location": "Section 5.1 paragraph beginning 'Two qualifications'; Appendix A.5, especially the claim that observing at least 3 rejections among 12 has probability 0.0196",
      "dimension": "technical_soundness; experimental_rigor; limitations_responsible_claims",
      "description": "The binomial calculation treats the twelve rejection indicators as independent Bernoulli(0.05) variables, while the paper elsewhere correctly states that these cells share items, nest model arms, and are neither independent nor exchangeable. The 0.0196 number is arithmetically correct only under the independence model and is not a valid global-null p-value here. This matters because BH and Bonferroni retain zero cells on both sides, so the unsupported binomial count is presented as the inferential result that survives correction.",
      "proposed_fix": "Either describe 3/12 versus 1/12 as a descriptive count only, or construct a joint item-level randomization/resampling test that preserves dependence across all twelve arms. Define the tested family and global null before reporting an aggregate p-value.",
      "verification_test": "A dependence-preserving simulation should demonstrate calibrated type-I error for the rejection-count statistic under the stated joint null. If this cannot be shown, the 0.0196 claim and all language that the count 'survives correction' should be removed."
    },
    {
      "id": "X4-04",
      "severity": "major",
      "location": "Abstract statement about mandatory power analysis; Section 4 'Power design'; Table 4; Section 5.1 claim that 52/60 cells are underpowered; Discussion 'Power is part of construct validity'",
      "dimension": "experimental_rigor; clarity",
      "description": "The manuscript labels comparisons between observed CI half-widths and a 1.389-point reference effect as a power analysis, but it provides no power definition, target power, alternative distribution, or rejection-probability calculation. Observed interval width is useful precision information, but it is not itself statistical power. Consequently the yes/no 'Detect -1.389 pp?' column and claims such as 'all 21 cells are powered' are not reconstructable from a conventional power criterion.",
      "proposed_fix": "Either rename this material as an achieved-precision or resolution analysis and avoid power terminology, or compute power under a stated effect, alpha, dependence structure, and target such as 80%, preferably using the same item-level procedure as the actual test.",
      "verification_test": "Each yes/no classification should be generated by a documented rule. If called power, the table should report numerical power estimates and a threshold; an independent implementation should reproduce every classification."
    },
    {
      "id": "X4-05",
      "severity": "major",
      "location": "Section 4 'Models and structural damage' and 'Designated damaged cells'; Section 5.1 headline 14/15 result; Table 5; Appendix A.4; complete v2 Table 7",
      "dimension": "clarity; experimental_rigor; reproducibility",
      "description": "The headline denominator cannot be reconstructed unambiguously. Section 4 lists OLMo-2 keep8, keep10, keep12, keep14, and shortgpt16 as damaged arms, but the 15-cell v1 headline includes only three OLMo-2 cells plus twelve non-OLMo cells. The exclusion of keep14 and shortgpt16 is not stated at the first denominator, despite Appendix A.4 promising that every excluded damaged arm will be named there. Table 5 also compresses four Llama-3 cells into one range row, so it is not actually a complete per-cell table. Because keep14 and shortgpt16 appear in the v2 table, a reader cannot verify that the cohort was fixed independently of outcomes or reproduce 14/15.",
      "proposed_fix": "Add one arm-inventory table listing every evaluated arm, construction, designated-damaged status, inclusion or exclusion rationale fixed without reference to score, v1 result, and v2 result. Print all four Llama-3 rows separately and recompute headline fractions if the explicit rule changes the cohort.",
      "verification_test": "An independent reader should be able to assign every arm mentioned in Section 4 and Table 7 to exactly one cohort status, obtain a denominator of 15 from the stated rule, and reproduce 14/15 and the other headline counts without guessing."
    },
    {
      "id": "X4-06",
      "severity": "major",
      "location": "Section 3.2 equations 3-6 and materiality paragraph; Appendix A.9; Tables 6-8",
      "dimension": "clarity; technical_soundness; reproducibility",
      "description": "The v2 verdict algorithm is incomplete in the text. Delta_max is described but not operationally defined; G1 and G2 are mentioned only in Appendix A.9 and never defined; the order of bootstrap, permutation, integrity, significance, anchor, and materiality gates is absent; and the single p column does not identify which test it reports. Tables omit recovery_fraction and most intact-anchor values even though those quantities determine 'trace signal', 'item-level signal', and 'anchor blocked'. A competent reader cannot reproduce the 27 labels from the manuscript alone.",
      "proposed_fix": "Provide pseudocode or a formal decision table defining Delta_max, G1/G2, both p-values, gate order, handling of negative deltas, the exact comparison to the family anchor, and every verdict label. Add recovery_fraction, anchor value, bootstrap p, and permutation p to the complete table.",
      "verification_test": "Give the method and printed table to an independent implementer with the per-item predictions. Their implementation should reproduce all 27 verdicts and both test decisions exactly without consulting hidden code or notes."
    },
    {
      "id": "X4-07",
      "severity": "major",
      "location": "Reproducibility Statement; Appendix A.15 and Table 12; submission MANIFEST.json; evidence/build_record.json; evidence/claim_evidence_map.tsv",
      "dimension": "reproducibility; experimental_rigor",
      "description": "The frozen snapshot does not ship the records needed to verify the quantitative claims. It contains only build_record.json and claim_evidence_map.tsv as evidence, while Table 12 and the map point to E-A through E-K, E-CAL, per-cell records, and scripts outside the allowed snapshot. Thus the assertions that every numeral is mechanically bound and that tables are emitted from per-item records cannot be checked. The build record also does not describe the shipped PDF: it records a 22-page PDF with SHA-256 56a376..., whereas the manifest contains a 24-page PDF with SHA-256 1fbaaf... and a different byte count.",
      "proposed_fix": "Ship immutable, snapshot-relative copies of every machine-readable record used by the paper, together with read-only emitters or sufficient derived per-item summaries. Regenerate the build record after the final compile and make Table 12 resolve only to files present in the submission artifact.",
      "verification_test": "A manifest resolver should confirm that every evidence identifier exists and matches its declared hash. The build record's PDF hash, size, and page count must match the frozen PDF. Randomly selected headline values should be reproducible from the shipped records without accessing external directories or running scripts that modify evidence."
    },
    {
      "id": "X4-08",
      "severity": "minor",
      "location": "427-word Abstract; Section 3.3 single-paragraph summary; Sections 5.1 and 5.3 appendix pointers; PDF Tables 5-12 and pages 12, 16-24",
      "dimension": "clarity",
      "description": "The presentation is over-compressed in the main text and inefficient in the appendix. The abstract introduces many denominators, tests, verdict labels, and implementation details before defining 'cell', 'arm', v1, v2, or recovery_fraction. Section 3.3 compresses four distinct measurement choices and numerous numbers into one paragraph. Meanwhile, most decision-critical tables are appendix-only, several use scriptsize text, and float placement leaves large portions of many appendix pages blank. The high-level message is understandable, but the main-text-only argument is not independently auditable.",
      "proposed_fix": "Shorten the abstract to the problem, two null questions, one primary calibrated result, and the scope limitation. Move a compact cohort/result table into the main text, define verdict terminology before use, split Section 3.3 into a short conceptual paragraph plus table, and reflow appendix floats.",
      "verification_test": "A domain reader using only the main nine pages should be able to identify the primary claim, reconstruct every headline denominator, distinguish point and significance counts, and explain all verdict labels. Tables should remain legible at normal page scale without excessive blank float pages."
    }
  ],
  "score_ceiling_under_current_evidence": 6,
  "predicted_score_after_required_changes": 5.5,
  "evidence_that_would_raise_score": [
    "A construct-valid MMLU-Pro floor calibration that preserves each item's legal option set and still supports the stated conclusion.",
    "A dependence-aware global analysis of the twelve-cell two-reference comparison, or a responsible reclassification of it as descriptive.",
    "A complete, outcome-independent arm inventory that reproduces all headline denominators.",
    "A formal v2 decision algorithm and complete table from which all 27 labels can be independently reproduced.",
    "A self-contained evidence package whose hashes and final-PDF build record match the frozen submission.",
    "A revised main text that narrows the claim from pairwise-comparison validity to the absolute capability interpretation actually tested."
  ],
  "evidence_that_would_lower_score": [
    "The corrected varying-option-count null places the MMLU-Pro floor inside ordinary estimator noise.",
    "A joint dependence-preserving test removes the claimed aggregate evidence behind 3/12 versus 1/12.",
    "The explicit cohort rule reveals outcome-dependent exclusions or materially changes 14/15.",
    "The underlying records disagree with any printed central value, verdict, or integrity claim.",
    "External checking shows that the cited historical literature already contains the claimed varying-k stratification or pre-comparison use."
  ],
  "review_limitations": [
    "I followed the blindness restriction and did not inspect any artifact path outside round_04/submission, including the paths named in Table 12.",
    "The underlying per-item predictions and central evidence records were absent from the snapshot, so most empirical values could be checked only for internal arithmetic and consistency, not confirmed or refuted from source data.",
    "I did not run the authors' checker or emitter scripts, as explicitly prohibited.",
    "I visually inspected all 24 pages of the frozen PDF read-only.",
    "External papers were not available within the permitted snapshot, so load-bearing claims about citation content, historical priority, and literature coverage could not be independently verified."
  ]
}
```

## PROSE REVIEW

### Summary and recommendation

This paper argues that multiple-choice evaluation should report an explicit input-blind null rather than relying solely on nominal chance. It proposes two read-outs:

1. **v1:** an arm-independent best-constant floor intended to gate interpretation of an interface.
2. **v2:** an arm-conditional permutation null, stratified by option count, intended to detect item-level alignment beyond the arm’s own output marginal.

The paper applies these read-outs to damaged model arms, analyzes length-based content nulls, and performs a full-precision control. The topic is important, the paper is unusually candid about failed hypotheses and analysis repairs, and the distinction between the two null questions is useful.

However, I recommend **reject (4/10)** in the current form. The main concerns are not cosmetic. The MMLU-Pro calibration appears to use the wrong null for a variable-option-count construct; the only aggregate significance claim surviving multiplicity discussion assumes independence that the paper itself denies; the central pre-comparison-gate interpretation does not follow from subtracting a common constant; and the frozen artifact lacks the evidence required to verify most quantitative claims. Several headline cohorts and v2 verdicts are also not reconstructable from the manuscript.

### Strongest verified contribution

The strongest contribution I could verify is the separation of two questions that should not be conflated:

- comparison with an arm-independent constant reference; and
- item-level alignment relative to an arm’s own prediction marginal.

The derivation that the stratified permutation statistic is exactly zero for every pure legal constant emitter is correct. I also appreciate the unusually explicit statement that the statistic itself is not new and that the claimed contribution is limited to varying-option-count stratification and its use as a gate.

### Major issues

#### 1. The claimed pairwise-comparison gate is not established

**Location:** Abstract; Introduction’s first numbered question; Section 3.1; Discussion.

**Severity:** Major.  
**Dimensions:** Technical soundness, significance, responsible claims.

The empirical results show that “above chance” can be a misleading **absolute** description. They do not establish that a below-floor arm cannot participate in a meaningful relative comparison.

Because v1 uses the same floor for every arm,

\[
(\mathrm{acc}_A-f)-(\mathrm{acc}_B-f)=\mathrm{acc}_A-\mathrm{acc}_B.
\]

Thus v1 cannot alter a pairwise difference or ranking. Treating floor passage as a necessary validity gate for arm comparison is a separate normative claim that needs an argument beyond the presented experiments.

**Fix:** Narrow the recommendation to absolute capability interpretation, explicitly acknowledge pairwise invariance, or define and test a separate criterion under which an actual arm comparison becomes invalid.

**Verification:** Every retained pairwise-validity claim should identify a formal criterion or a concrete pairwise conclusion that fails under it.

#### 2. The MMLU-Pro winner’s-curse null does not preserve legal option sets

**Location:** Abstract; Introduction finding 1; Table 1; Appendix Table 10.

**Severity:** Major.  
**Dimensions:** Technical soundness, experimental rigor.

The paper correctly emphasizes that MMLU-Pro has between three and ten options per item. Nevertheless, the floor-estimator calibration treats it as a balanced \(k=10\) multinomial. Under a random legal-answer null, A is legal on every item and has expected frequency

\[
\operatorname{mean}(1/n_{\mathrm{opt}})=0.110877,
\]

not 0.10. The simulation should preserve each item’s legal label set. The identical null-distribution columns for the naive and item-averaged MMLU-Pro rows make clear that it does not.

This means the reported \(p<10^{-5}\) for the 0.116606 floor is currently unverified.

**Fix:** Resimulate by drawing each item’s label uniformly from its legal options and recomputing the maximum letter marginal on every draw.

**Verification:** The generator must produce no illegal labels, reproduce expected A frequency 0.110877, and yield stable quantiles and p-values across independent runs or an exact calculation.

#### 3. The binomial aggregate test assumes independence contradicted by the paper

**Location:** Section 5.1 and Appendix A.5.

**Severity:** Major.  
**Dimensions:** Technical soundness, experimental rigor.

The reported probability \(P(X\ge3)=0.0196\) is correct for \(X\sim\mathrm{Binomial}(12,0.05)\). But these twelve tests share benchmark items and include related, nested model arms. The manuscript itself says the cells are neither independent nor exchangeable.

This is especially consequential because BH and Bonferroni leave 0/12 cells on both sides. The binomial count is then described as what “survives correction,” but it is not a calibrated correction for this dependent family.

**Fix:** Use a joint item-level randomization or resampling procedure that preserves dependence across arms, or report 3/12 versus 1/12 descriptively without an aggregate p-value.

**Verification:** Demonstrate calibrated type-I error for the rejection-count statistic under the specified joint null.

#### 4. The reported “power analysis” is an achieved-precision comparison

**Location:** Section 4, Table 4, Section 5.1, and Discussion.

**Severity:** Major.  
**Dimensions:** Experimental rigor, clarity.

Table 4 compares observed CI half-widths with a 1.389-point reference effect. That is useful precision information, but the paper does not define a target power, an alternative distribution, or the probability of rejection under that alternative. Therefore labels such as “Detect \(-1.389\) pp?” and statements that all 21 cells are “powered” are not reproducible as conventional power claims.

**Fix:** Rename this an achieved-precision or resolution analysis, or compute numerical power using the same test and dependence structure as the experiment.

**Verification:** Report a documented threshold and numerical power for each classification if the term “power” is retained.

#### 5. The 15-cell headline cohort cannot be reconstructed

**Location:** Section 4; Section 5.1; Table 5; Appendix A.4; Table 7.

**Severity:** Major.  
**Dimensions:** Clarity, experimental rigor, reproducibility.

Section 4 lists five damaged OLMo-2 arms: keep8, keep10, keep12, keep14, and shortgpt16. The 15-cell headline appears to include only three OLMo-2 arms and twelve non-OLMo arms. The first occurrence of the denominator does not explain why keep14 and shortgpt16 are excluded, despite Appendix A.4 promising that every such exclusion will be named.

Table 5 also compresses four Llama-3 cells into a range row, so the claimed per-cell table is not actually per-cell.

**Fix:** Add a complete arm-inventory table with construction, designated status, inclusion/exclusion rationale, and v1/v2 results.

**Verification:** A reader should be able to map every evaluated arm to one cohort status and reproduce 14/15 without inference.

#### 6. The v2 decision procedure is incompletely specified

**Location:** Section 3.2; Appendix A.9; Tables 6–8.

**Severity:** Major.  
**Dimensions:** Clarity, technical soundness, reproducibility.

The paper does not operationally define \(\Delta_{\max}\), G1, or G2. It does not give the gate order or identify whether the table’s single p-value is from the bootstrap or permutation test. The complete table omits `recovery_fraction` and most anchor values, even though these determine the distinction among “trace signal,” “item-level signal,” and “anchor blocked.”

Consequently, a competent reader cannot reproduce the 27 verdicts from the manuscript alone.

**Fix:** Add pseudocode or a formal decision table and print all decision-relevant quantities.

**Verification:** An independent implementation using only the written procedure and prediction records should reproduce all 27 verdicts.

#### 7. The evidence package is incomplete, and the build record describes a different PDF

**Location:** Reproducibility Statement; Appendix Table 12; `MANIFEST.json`; `evidence/build_record.json`; `evidence/claim_evidence_map.tsv`.

**Severity:** Major.  
**Dimensions:** Reproducibility, experimental rigor.

The snapshot ships only two evidence files. The cited E-A through E-K and E-CAL records, per-cell results, and emitters are absent and point outside the only directory reviewers are permitted to inspect. Therefore claims that every numeral is mechanically bound and that tables are emitted from per-item records cannot be verified.

There is also a concrete provenance mismatch:

- the build record reports a 22-page PDF with hash beginning `56a376...`;
- the frozen snapshot contains a 24-page PDF with hash beginning `1fbaaf...`.

**Fix:** Include immutable snapshot-relative copies of every evidence record and regenerate the build record after the final compile.

**Verification:** Every evidence identifier should resolve inside the snapshot, and the build record’s hash, byte count, and page count should match the frozen PDF.

### Presentation issue

The paper is understandable at a high level, but overly compressed. The 427-word abstract contains many denominators, verdicts, statistical qualifications, and implementation details before key terms are defined. Section 3.3 compresses four distinct measurement questions and numerous numerical examples into one paragraph. Core tables are appendix-only, often in very small type, while float placement leaves large blank regions across the appendix.

I recommend moving a compact cohort/result table into the main text and substantially shortening the abstract. A main-text-only reader should be able to reconstruct the primary denominator, distinguish point comparisons from significant comparisons, and understand each verdict label.

### Positive aspects

The manuscript deserves credit for several practices:

- It clearly states that passing a floor is not sufficient to establish construct validity.
- It acknowledges that v2 is post-hoc on the existing cells.
- It reports regime and family confounding rather than implying a clean family comparison.
- It discloses multiple-testing problems rather than hiding them.
- It reports repaired bootstrap, truncation, and OOM defects.
- It retracts unsupported mechanisms and universal claims.
- The full-precision result is interpreted narrowly rather than used to claim a universal mechanism.

These strengths make the work promising, but they do not resolve the current central statistical and provenance problems.

### Score rationale

- **Overall:** 4/10, reject.
- **Confidence:** 4/5.
- **Current-evidence ceiling:** 6/10.

A weak-accept-level paper may be achievable without new model experiments if the authors correct the MMLU-Pro null, replace or remove the dependent binomial inference, narrow the arm-comparison claim, make cohorts and v2 labels fully reconstructable, and ship the claimed evidence. The principal uncertainty is whether the corrected construct-aware null and dependence-aware analysis still support the headline empirical conclusion.