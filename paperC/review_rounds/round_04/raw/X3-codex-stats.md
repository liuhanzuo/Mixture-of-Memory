```json
{
  "reviewer_id": "X3-codex-stats",
  "round": 4,
  "snapshot_sha256": "7fcb9ccc55c5c1d6ad1de868215b1d9253d0695a2f996340226aa6a6fa10db5a",
  "role": "experimental_rigor",
  "overall_score": 4,
  "confidence": 4,
  "recommendation": "reject",
  "dimension_scores": {
    "novelty": 3,
    "significance": 4,
    "technical_soundness": 2,
    "experimental_rigor": 2,
    "clarity": 4,
    "reproducibility": 2,
    "citation_integrity": 3,
    "limitations_responsible_claims": 3
  },
  "paper_summary": "The paper proposes null calibration for likelihood-scored multiple-choice evaluation: an arm-independent best-constant floor for interface comparability and a within-option-count, arm-conditional permutation statistic for item-level information. It applies these read-outs to structurally damaged language-model arms and argues that conventional chance lines can credit degenerate behavior. The measurement question is important and the manuscript is unusually candid about several negative findings, but a load-bearing MMLU-Pro calibration null is incompatible with the benchmark's variable legal option counts, the aggregate multiplicity argument assumes independence the paper explicitly denies, and most empirical evidence cited by the manuscript is absent from the frozen evidence snapshot.",
  "strongest_verified_contribution": "The cleanest verified contribution is the conceptual separation between an arm-independent interface floor and an arm-conditional alignment null. From the definitions in Section 3.2, Delta_perm is the stratified Cohen-kappa numerator and is exactly zero for every legal pure constant emitter; the paper correctly labels this as an algebraic identity rather than empirical evidence.",
  "strengths": [
    "The paper identifies a practically important evaluation failure: nominal chance can make a literal constant emitter look above baseline when the empirical label marginal is imbalanced.",
    "The v1/v2 distinction is conceptually useful, and the derivation of the stratified permutation expectation is clear; the paper also correctly warns that v1/v2 ordering is not an identity.",
    "The manuscript is unusually transparent about post-hoc status, family/regime confounding, failed mechanisms, integrity defects, and the fact that passing a null is necessary rather than sufficient.",
    "For the fixed-k constructs, the reported winner's-curse means and quantiles are arithmetically plausible under the stated balanced multinomial simulation; several displayed deltas and the 18.03% precision-change calculation are internally consistent.",
    "The distinction between point-above-reference counts and interval-excluding-zero counts on MMLU-Pro is a valuable correction to an initially asymmetric comparison."
  ],
  "issues": [
    {
      "id": "TS-01",
      "severity": "major",
      "location": "Abstract; Introduction lines 9-14; Section 3.1 lines 14-18; Table 3 ('Is this interface valid for comparing arms?')",
      "dimension": "technical_soundness; limitations_responsible_claims",
      "description": "The paper's evidence establishes an absolute-reference problem, but not the stronger claim that failing the best-constant floor invalidates arm comparison. Because v1 subtracts the same arm-independent constant from every arm, Delta_floor(A)-Delta_floor(B)=acc(A)-acc(B): neither ranking nor pairwise effect changes. A below-floor arm may still carry item-level information, as the paper's own v2 distinction permits. Thus the floor can diagnose that an absolute score does not beat a trivial predictor, but it is not by itself a necessary condition for a statistically meaningful arm-to-arm contrast.",
      "proposed_fix": "Narrow the central rule from an 'interface-validity/pre-comparison gate' to an absolute-score reporting requirement, or add a pre-specified validation showing that below-floor pairwise differences fail an independent criterion such as replicated item-level alignment or downstream retained-knowledge measurement.",
      "verification_test": "After revision, no claim should say that a floor failure alone invalidates a pairwise arm difference unless a stated theorem or empirical validation supports that implication; pairwise conclusions should be tested directly with paired arm contrasts and an independent construct-validity criterion."
    },
    {
      "id": "ER-01",
      "severity": "major",
      "location": "Section 3.3(iv), Table 1 rows 'MMLU-Pro, naive/item-avg.' (tab_nulls.tex lines 9-10 and 20-29), and Table 10",
      "dimension": "technical_soundness; experimental_rigor",
      "description": "The winner's-curse calibration for MMLU-Pro uses a ten-category balanced multinomial that does not preserve the item-specific legal option sets. The paper states that n_opt varies from 3 to 10 and that mean(1/n_opt)=0.110877. If gold labels are drawn uniformly from each item's legal options, A is legal on every item, so E[m_A]=0.110877 and therefore E[max_L m_L] must be at least 0.110877. Table 1 instead reports E[f_hat]=0.104460 for the item-average row, which is mathematically impossible for the construct-aware null. Thus the p<1e-5 MMLU-Pro calibration is not evidence against the relevant balanced variable-k null; this directly affects one of the three constructs carrying the abstract's quantitative floor-above-chance claim.",
      "proposed_fix": "Recompute the calibration by drawing one gold label uniformly from each item's actual legal label set, preserving the full n_opt vector, and taking the maximum empirical label marginal on every draw. Report the revised expectation, quantiles, tail probability, and resulting scope of the MMLU-Pro claim.",
      "verification_test": "In the corrected simulation, the empirical mean of the always-A marginal must equal 0.110877 within Monte Carlo error and the mean simulated maximum must be at least 0.110877. The published MMLU-Pro p-value and verdict must reproduce from shipped item-level option-count/legal-label data."
    },
    {
      "id": "ER-02",
      "severity": "major",
      "location": "Section 5.1 paragraph 'Both sides of the flip' (05_analysis.tex line 25) and Appendix A.5 (09a_relocated.tex line 37)",
      "dimension": "experimental_rigor; technical_soundness",
      "description": "After BH and Bonferroni leave 0/12 cells on both sides, the paper claims that the count survives because Pr[Binomial(12,0.05)>=3]=0.0196. That calculation assumes independent Bernoulli rejections with common size 0.05. The manuscript itself says these tests share items and a null, nest arms, and are neither independent nor exchangeable. The binomial null is therefore unjustified and cannot rescue the 3/12 versus 1/12 inference.",
      "proposed_fix": "Treat 3/12 versus 1/12 as descriptive, or define a joint aggregate statistic and obtain its null distribution by synchronized item-level permutation/bootstrap across all 12 arms, preserving cross-arm dependence.",
      "verification_test": "A simulation under the complete joint null must show type-I error at or below 0.05 for the full aggregate procedure; report the dependence-preserving joint p-value and Monte Carlo uncertainty."
    },
    {
      "id": "ER-03",
      "severity": "major",
      "location": "Section 4 'Power design', Table 4 (tab_power.tex), Section 5.1 off-MMLU claims, and Section 6 'Power is part of construct validity'",
      "dimension": "experimental_rigor; limitations_responsible_claims",
      "description": "The displayed 'power' analysis classifies detection from whether an observed 95% CI half-width is smaller than 1.389 percentage points. This is a precision/MDE diagnostic, not achieved power at a declared level. Under a normal approximation, ARC-Easy's half-width 1.305 corresponds to only about 55% two-sided power for a 1.389-point effect, despite being labeled 'yes, borderline'; Winogrande is only about 63%. Consequently, '52/60 underpowered' and related small-benchmark power labels are not established by the table as written.",
      "proposed_fix": "Either relabel the table as observed precision and avoid power language, or calculate prospective/achieved rejection probability under a specified paired alternative using the item-level discordance structure, with a declared target such as 80% or 90%.",
      "verification_test": "For every cell called powered, a pre-specified item-level simulation or exact paired calculation under the -1.389-point alternative must attain the declared power threshold."
    },
    {
      "id": "ER-04",
      "severity": "major",
      "location": "Abstract; Section 5.1; Table 5 v1 verdicts; Table 7 v2 verdicts",
      "dimension": "experimental_rigor; limitations_responsible_claims",
      "description": "The paper repeatedly converts failure to reject into equality/absence language: '14/15 at or below the floor', v1 labels of 'at floor', and v2 labels of 'no signal'. Positive estimates such as Llama-2 k14 (+0.017 pp) and Qwen3 k10 (+0.158 pp) merely fail to demonstrate superiority; p>0.05 does not establish that the population effect is zero or below zero. Likewise, 'no signal' requires an equivalence or materiality analysis, not a conventional null test.",
      "proposed_fix": "Use 'did not significantly clear the floor/null' for ordinary non-rejections. If literal equivalence or no material signal is claimed, pre-specify a negligible-effect margin and use two one-sided tests or confidence-bound equivalence analysis.",
      "verification_test": "Every cell labeled 'at floor' or 'no material signal' must have its full confidence interval contained within a pre-specified equivalence region; otherwise it must be labeled inconclusive/non-significant."
    },
    {
      "id": "ER-05",
      "severity": "major",
      "location": "Abstract and Introduction off-MMLU 10/15 claim; Section 5.1 line 29; Appendix A.6 lines 42-44",
      "dimension": "experimental_rigor; clarity",
      "description": "The paper correctly applies a symmetric CI standard on MMLU-Pro, reducing the apparent flip from 10/12 versus 1/12 to 3/12 versus 1/12, but then headlines 10/15 and 25/60 off-MMLU 'above chance' point counts without the corresponding chance-side interval-exclusion counts. The claimed 'same wrong-null flip' therefore restores the exact evidentiary asymmetry that the MMLU-Pro analysis diagnoses.",
      "proposed_fix": "For every off-MMLU cell, recompute the chance reference inside the same paired item bootstrap and report both point-above and CI-excludes-zero counts for chance and floor. Reword the headline to descriptive point comparisons unless the symmetric inferential flip survives.",
      "verification_test": "A per-cell table must apply an identical CI decision rule to both references and reproduce any retained aggregate count from those symmetric decisions."
    },
    {
      "id": "ER-06",
      "severity": "major",
      "location": "Section 3.2 lines 75-82 and Section 5.1/5.3 claims that Qwen3 k14 is 'immaterial' and that Llama-2's anchor is blocked",
      "dimension": "experimental_rigor; limitations_responsible_claims",
      "description": "The 10% relative-recovery materiality rule is applied to point ratios without uncertainty. Qwen3 k14 is called immaterial at 9.1% of its intact anchor, narrowly below the 10% cutoff, although both numerator and anchor are estimated on shared items and the current 27-cell analysis is explicitly post-hoc. This supports, at most, a point-estimate classification.",
      "proposed_fix": "Justify the 10% threshold independently and jointly bootstrap damaged and intact recovery fractions with synchronized item resampling. Use a one-sided confidence bound for materiality decisions.",
      "verification_test": "Call an arm immaterial only if the one-sided 95% upper confidence bound for its relative recovery is below 0.10; otherwise report materiality as indeterminate."
    },
    {
      "id": "RP-01",
      "severity": "major",
      "location": "Reproducibility Statement lines 3-10; Appendix A.15/Table 12; claim_evidence_map.tsv H-01 through H-04; snapshot MANIFEST",
      "dimension": "reproducibility; experimental_rigor",
      "description": "The manuscript says every quantitative claim is bound to machine-readable records and that a reader can rerun emitters. The frozen snapshot ships only build_record.json and claim_evidence_map.tsv; the cited calibration JSON, per-item prediction records, permutation/bootstrap outputs, power records, emitters, and checkers are absent. Table 12 resolves most evidence IDs to paths outside the snapshot, including excluded tcodex_out. Thus almost all headline empirical results can be checked only for displayed arithmetic, not from primary evidence.",
      "proposed_fix": "Ship a self-contained artifact containing every cited evidence record, the central per-item labels/predictions and legal option metadata, exact analysis configurations, and emitter/checker source, with internal immutable paths and hashes.",
      "verification_test": "In a clean environment using only the frozen artifact, every Table 12 path resolves and an independent implementation reproduces all headline floors, intervals, p-values, counts, and v2 classifications within stated Monte Carlo tolerance."
    },
    {
      "id": "RP-02",
      "severity": "major",
      "location": "evidence/build_record.json lines 53-60 versus MANIFEST entry for manuscript/main.pdf",
      "dimension": "reproducibility",
      "description": "The build record authenticates a different PDF: it records 22 pages, 355196 bytes, and SHA-256 56a376..., while the shipped PDF has 24 pages, 366583 bytes, and manifest SHA-256 1fbaaf.... Therefore build_gate_pass=true does not attest to the reviewed artifact.",
      "proposed_fix": "Regenerate the build record after freezing the final sources/PDF and bind it to the manifest or source-tree hash.",
      "verification_test": "The regenerated record's PDF hash, byte size, and page count must exactly equal the manifest and an independent PDF parser's output."
    },
    {
      "id": "NR-01",
      "severity": "minor",
      "location": "Appendix A.1.1 line 12; Tables 1 and 10 and their captions",
      "dimension": "reproducibility; clarity",
      "description": "Several arithmetic inconsistencies undermine the claim that all numerals are mechanically checked: 0.532164/0.125914 is 4.226, not 4.6; the exact CommonsenseQA gap rounds to 0.885 pp in Table 1 but 0.884 in generated Table 10; the stated small-construct gap range starts at +0.43 pp although the displayed minimum is about +0.490 pp. In addition, the PIQA balanced-null p-value 0.658 corresponds to thresholding at 929 successes after rounding the floor, whereas the exact floor is 928/1838 and gives about 0.692 under the stated binary null. These do not change the qualitative row verdicts but reveal semantic and rounding gaps in the checker.",
      "proposed_fix": "Carry exact integer counts/rationals through all tests, round only for display, correct the ratio/range, and extend the checker to validate derived ratios, extrema, and test thresholds rather than merely matching numerals.",
      "verification_test": "Tables 1 and 10 must agree under one documented rounding rule; the PIQA p-value must be recomputed from the exact count; injected errors in a ratio, range endpoint, or rounded test threshold must make the checker fail."
    }
  ],
  "score_ceiling_under_current_evidence": 6,
  "predicted_score_after_required_changes": 6,
  "evidence_that_would_raise_score": [
    "A defensible narrowing of the headline from 'pre-comparison validity gate' to absolute-reference calibration, or direct evidence that floor failure predicts invalid pairwise arm comparisons.",
    "A corrected MMLU-Pro variable-option calibration preserving each item's legal labels, with the revised result still supporting a practically meaningful floor/chance distinction.",
    "A dependence-preserving joint analysis of the 12-cell symmetric flip and symmetric chance-versus-floor inference for the off-MMLU cells.",
    "A genuine power/equivalence analysis replacing CI-width labels and non-rejection-as-equality language.",
    "A self-contained frozen evidence artifact that reproduces all headline tables and p-values from per-item records, plus a build record matching the submitted PDF.",
    "Uncertainty for relative-recovery/materiality decisions, especially the Qwen3 k14 boundary case."
  ],
  "evidence_that_would_lower_score": [
    "If the corrected variable-k MMLU-Pro null places the observed 0.116606 floor inside ordinary selection noise, the abstract's main three-construct calibration claim would shrink materially.",
    "If a joint dependence-preserving null shows that the 3/12 rejection count is unremarkable, the inferential wrong-null flip would reduce to descriptive examples.",
    "If the unshipped primary records fail to reproduce the printed per-cell intervals, p-values, designated-cell counts, or integrity-repair claims.",
    "If full-text citation checking shows that the claimed novelty boundary for heterogeneous option-count stratification is already established."
  ],
  "review_limitations": [
    "I obeyed the blind scope and did not inspect any paperC path outside round_04/submission, any repository history, or any unshipped evidence path named by the manuscript.",
    "I did not run the authors' checker or emitter scripts. Most primary empirical records were not present in the snapshot, so those claims are marked unverified rather than refuted.",
    "Citation full texts were not shipped; I could verify key existence and broad topical alignment from the bibliography, but not all precise local attributions or the novelty boundary.",
    "I verified all manifest-listed payload hashes individually and inspected the submitted PDF read-only; the manifest does not specify a procedure for independently reconstructing the aggregate snapshot hash."
  ]
}
```

## PROSE REVIEW

### Summary and overall assessment

This paper argues that multiple-choice results should be calibrated against an explicit input-blind null rather than nominal chance alone. It distinguishes:

1. **V1**, an arm-independent best-constant reference intended to assess whether an interface supports arm comparison; and
2. **V2**, an arm-conditional, option-count-stratified permutation reference intended to detect item-level alignment beyond the arm’s output marginal.

The underlying evaluation problem is important. In particular, the observation that a literal constant emitter can exceed nominal chance on an imbalanced benchmark is a useful warning. The V1/V2 distinction is also conceptually helpful, and the algebra showing that a legal constant emitter has exactly zero V2 statistic is correct.

However, I find several decision-relevant statistical problems. Most seriously, the MMLU-Pro winner’s-curse calibration does not preserve its variable legal option sets and is internally incompatible with the reported item-averaged chance. The multiplicity argument then invokes an independent binomial model after explicitly acknowledging that the tests are dependent. The paper additionally uses CI width as “power,” interprets non-rejection as equality or absence of signal, and applies an uncertain 10% materiality boundary without uncertainty on the ratio. Finally, almost all primary evidence referenced by the paper is absent from the frozen snapshot, and the supplied build record describes a different PDF.

I therefore assign **4/10, reject**, with confidence **4/5**. The core idea is promising, but the current statistical evidence does not support the breadth or certainty of the headline claims.

### Strengths

- The paper addresses a real and consequential evaluation pathology. The OpenBookQA constant-emitter example, if reproduced from the missing records, would be an especially intuitive demonstration.
- Section 3 clearly separates an absolute, arm-independent reference from an arm-conditional independence reference.
- The paper correctly identifies  
  \[
  \Delta_{\mathrm{perm}}=p_o-p_e=\kappa(1-p_e)
  \]
  as an identity rather than a new statistic. It similarly avoids presenting constant-emitter zero as an empirical discovery.
- The discussion of why V1/V2 ordering is not guaranteed is careful. The manuscript supplies a counter-construction instead of treating the observed 27/27 ordering as a theorem.
- The authors disclose post-hoc analysis, regime/family confounding, prior numerical and launch defects, underpowered benchmarks, and retracted hypotheses. This is substantially more responsible than hiding contradictory results.
- On MMLU-Pro, the move from asymmetric point-count comparisons to a symmetric interval criterion is directionally correct and materially changes the apparent result from 10/12 versus 1/12 to 3/12 versus 1/12.

### Major concerns

#### 1. The V1 floor does not by itself establish whether arms are comparable  
**Severity:** Major  
**Location:** Abstract; Introduction, items 1–2; Section 3.1; Table 3  
**Dimensions:** Technical soundness; responsible claims

The paper’s examples establish that an absolute score can fail to beat a trivial constant predictor. They do not establish the stronger headline proposition that floor failure makes an arm-to-arm comparison invalid.

Since the same floor is subtracted from each arm,

\[
\Delta_{\mathrm{floor}}(A)-\Delta_{\mathrm{floor}}(B)
=\mathrm{acc}(A)-\mathrm{acc}(B).
\]

Thus V1 changes neither arm rankings nor pairwise effects. Two below-floor arms can still differ reliably, and a below-floor arm can still carry item-level information under the paper’s own V2 definition. The floor is consequently a useful absolute-reference diagnostic, but it is not automatically a necessary condition for every meaningful paired comparison.

**Required fix:** Narrow the recommendation to an absolute-score reporting requirement, or provide independent validation that below-floor pairwise differences systematically fail a separate construct-validity criterion.

**Verification:** The revised paper should not infer invalidity of a pairwise contrast solely from V1 failure. Pairwise conclusions should be tested directly and linked to an independently stated validity criterion.

#### 2. The MMLU-Pro winner’s-curse null is incompatible with variable legal option counts  
**Severity:** Major  
**Location:** Section 3.3(iv); Table 1 MMLU-Pro rows; Table 10  
**Dimensions:** Technical soundness; experimental rigor

MMLU-Pro has between three and ten legal options per item, and the paper reports

\[
\frac{1}{n}\sum_i\frac{1}{k_i}=0.110877.
\]

Under a balanced null that draws uniformly from each item’s legal options, A is legal for every item. Therefore

\[
E[\hat m_A]=0.110877
\quad\text{and}\quad
E[\max_L\hat m_L]\ge 0.110877.
\]

Table 1 instead reports \(E[\hat f]=0.104460\). This is impossible under the construct-aware variable-option null. It is consistent with a ten-way multinomial that permits labels which are illegal for shorter-option items.

The reported \(p<10^{-5}\) therefore does not test the relevant balanced MMLU-Pro null. This is load-bearing because MMLU-Pro is one of only three constructs supporting the abstract’s calibrated floor-above-chance claim.

**Required fix:** Simulate each item’s label uniformly from its actual legal label set and recompute the maximum marginal in every draw.

**Verification:** The simulated always-A mean must reproduce 0.110877, and the simulated maximum must have mean at least 0.110877. The revised tail probability must be reproducible from shipped legal-option metadata.

#### 3. The rejection-count binomial assumes independence that the paper explicitly denies  
**Severity:** Major  
**Location:** Section 5.1, final multiplicity paragraph; Appendix A.5  
**Dimensions:** Experimental rigor; technical soundness

The value 0.0196 is arithmetically

\[
P\{\operatorname{Binomial}(12,0.05)\ge3\}.
\]

That distribution requires independent, equally sized Bernoulli tests. Appendix A.4 states that the arms share items and a null, nest within families, and are “neither independent nor exchangeable.” Consequently, the binomial calculation cannot be used to claim that the 3/12 count “survives correction,” especially after both BH and Bonferroni produce 0/12 on both sides.

**Required fix:** Present 3/12 versus 1/12 as descriptive, or construct a joint null by synchronously permuting/resampling items across all arms.

**Verification:** Simulations under the complete joint null must demonstrate type-I error control for the aggregate procedure. The reported count-level \(p\)-value must come from that dependence-preserving distribution.

#### 4. Table 4 reports precision, not conventional power  
**Severity:** Major  
**Location:** Section 4 “Power design”; Table 4; Section 5.1; Section 6  
**Dimensions:** Experimental rigor; responsible claims

The criterion appears to be whether the observed 95% half-width is smaller than 1.389 points. This says that a hypothetical interval centered exactly at the target effect would exclude zero. It does not establish a conventional power level.

For example, a half-width of 1.305 points implies only approximately 55% two-sided power for a 1.389-point effect under a normal approximation, yet ARC-Easy is labeled “yes, borderline.” Winogrande’s corresponding figure is approximately 63%. The classification “52/60 underpowered” is therefore not supported by a stated power calculation.

This does not invalidate a direct statement that a particular observed interval excludes a specified effect. It does invalidate treating all half-width classifications as achieved power.

**Required fix:** Rename the table as a precision or minimum-detectable-effect diagnostic, or compute actual rejection probabilities under a specified paired alternative and declared target such as 80%.

**Verification:** Every cell called powered must meet the declared target in an item-level simulation or exact paired calculation.

#### 5. Non-rejection is interpreted as equality or absence  
**Severity:** Major  
**Location:** Abstract; Section 5.1; Tables 5 and 7  
**Dimensions:** Experimental rigor; responsible claims

“Fourteen of 15 at or below the floor,” “at floor,” and “no signal” are stronger than the reported tests permit. Several cells have positive point estimates but fail to reject zero—for example Llama-2 \(k14\) at \(+0.017\) points and Qwen3 \(k10\) at \(+0.158\) points. These results mean that superiority was not demonstrated, not that the true effect is equal to or below zero.

Similarly, failure to reject the V2 null does not establish absence of signal unless an equivalence or negligibility region is tested.

**Required fix:** Replace these labels with “did not significantly clear the floor/null.” If equality or no material signal is intended, define an equivalence margin and test it.

**Verification:** A cell may be labeled equivalent or materially null only when its entire confidence interval lies inside the pre-specified negligible-effect interval.

#### 6. The off-MMLU headline repeats the asymmetric comparison  
**Severity:** Major  
**Location:** Abstract; Introduction; Section 5.1; Appendix A.6  
**Dimensions:** Experimental rigor; clarity

The paper shows that applying a symmetric interval criterion changes the MMLU-Pro comparison from 10/12 versus 1/12 to 3/12 versus 1/12. However, it continues to headline 10/15 and 25/60 off-MMLU “above chance” point counts without reporting the corresponding chance-side interval-exclusion counts.

It has therefore not demonstrated that the smaller benchmarks exhibit the same inferential flip. At present, these are descriptive reference-crossing counts.

**Required fix:** Apply the same paired-bootstrap decision criterion to chance and floor for every off-MMLU cell.

**Verification:** A revised per-cell table should report point estimates and interval decisions against both references, and any retained aggregate must be computed from the symmetric decisions.

#### 7. The 10% materiality verdict lacks uncertainty  
**Severity:** Major  
**Location:** Section 3.2; Sections 5.1 and 5.3  
**Dimensions:** Experimental rigor; responsible claims

Qwen3 \(k14\) is called immaterial because its point estimate is 9.1% of the intact-family anchor, narrowly below a 10% threshold. No uncertainty is reported for that relative-recovery ratio. Both the damaged and intact quantities are estimated on shared items, and the paper acknowledges that the present 27-cell re-analysis is post-hoc.

**Required fix:** Independently justify the 10% threshold and jointly bootstrap the damaged and intact recovery fractions using synchronized item resampling.

**Verification:** “Immaterial” should require a one-sided 95% upper confidence bound below 0.10. Otherwise the materiality classification is indeterminate.

#### 8. The primary evidence needed to verify the empirical claims is absent  
**Severity:** Major  
**Location:** Reproducibility Statement; Appendix A.15/Table 12; claim-evidence map; snapshot manifest  
**Dimensions:** Reproducibility; experimental rigor

The manuscript says every quantitative claim is tied to machine-readable evidence and that a reader can regenerate the tables. The snapshot contains only `build_record.json` and `claim_evidence_map.tsv` under `evidence/`. The actual calibration record, prediction records, permutation outputs, power records, emitters, and checkers cited by Tables 1–12 are not shipped. Many references point to `tcodex_out` or other paths outside the frozen submission.

I can verify selected arithmetic, but not the central empirical counts, resampling implementation, confidence intervals, integrity repairs, or designated-cell membership.

**Required fix:** Supply a self-contained artifact with all central per-item records, legal-option metadata, configurations, analysis source, and immutable hashes.

**Verification:** From a clean copy of the frozen artifact, every Table 12 path should resolve and an independent implementation should reproduce all headline values within declared Monte Carlo tolerances.

#### 9. The build record is for a different PDF  
**Severity:** Major  
**Location:** `evidence/build_record.json`, lines 53–60; `MANIFEST.json` PDF entry  
**Dimension:** Reproducibility

The build record reports a 22-page, 355,196-byte PDF with SHA-256 beginning `56a376`. The shipped submission is a 24-page, 366,583-byte PDF with SHA-256 beginning `1fbaaf`. Thus `build_gate_pass: true` does not authenticate the artifact under review.

**Required fix:** Regenerate the build record after freezing the final source and PDF.

**Verification:** Hash, byte size, and page count must exactly match the manifest and an independent PDF parser.

### Minor numerical issue

**NR-01 — Minor; Appendix A.1.1 and Tables 1/10; reproducibility and clarity.**

Several calculations conflict with the claim that all numerals are mechanically checked:

- \(0.532164/0.125914=4.226\), not \(4.6\).
- The CommonsenseQA gap rounds to 0.885 points in Table 1 but 0.884 in generated Table 10.
- The displayed minimum among the five small-construct gaps is approximately 0.490 points, not the caption’s 0.43.
- PIQA’s exact floor is \(928/1838\). Under the stated binary null, thresholding at that exact count gives approximately \(p=0.692\); the reported 0.658 corresponds to thresholding at 929 after rounding the displayed floor upward.

These do not change the row-level qualitative verdicts, but they show that the checker does not reliably validate semantic arithmetic or exact testing thresholds.

**Fix:** Retain exact counts/rationals through testing and expand the checker to validate ratios, extrema, and derived thresholds.

**Verification:** Tables 1 and 10 should agree under one rounding rule, and deliberately perturbing any derived calculation should make the checker fail.

### Score rationale and evidence ceiling

The paper has a worthwhile core observation and unusually responsible discussion, but the current evidence is not decision-ready. The most severe empirical issue is the variable-option MMLU-Pro calibration, while the weakest evidence-to-importance ratio belongs to the claim that V1 failure invalidates arm comparisons.

I set the **current score at 4**. The **ceiling without new experiments is 6**: a weak-accept-level paper may be recoverable through corrected analysis of existing records, narrower claims, proper joint inference, equivalence-aware wording, and a complete artifact. If the corrected MMLU-Pro null places the observed floor inside ordinary selection noise, or if joint inference makes the 3/12 count unremarkable, the paper’s empirical headline would need substantial narrowing.

### Review limitations

I inspected only the permitted frozen snapshot and the supplied rubric/protocol. I did not inspect any excluded repository path, history, prior review, or unshipped evidence file, and I did not run the authors’ scripts. Citation full texts were not included, so precise load-bearing citation claims and the stated novelty boundary remain unverified rather than refuted.