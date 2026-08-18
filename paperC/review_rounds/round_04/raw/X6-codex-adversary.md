```json
{
  "reviewer_id": "X6-codex-adversary",
  "round": 4,
  "snapshot_sha256": "7fcb9ccc55c5c1d6ad1de868215b1d9253d0695a2f996340226aa6a6fa10db5a",
  "role": "technical_soundness",
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
  "paper_summary": "The paper advocates replacing conventional chance lines in multiple-choice evaluation with explicit input-blind references. Its v1 read-out compares accuracy with an arm-independent best-constant or length-based floor, while v2 subtracts the expected accuracy obtained by permuting an arm's prediction multiset within option-count strata. The empirical study examines damaged base language models across several benchmarks, argues that conventional chance can credit degenerate outputs, analyzes reference under-specification, and proposes null calibration as a pre-comparison reporting rule.",
  "strongest_verified_contribution": "The clearest technically verified contribution is the separation of two distinct null questions. The v2 statistic is algebraically p_o minus p_e within option-count strata and is exactly zero for any legal pure constant emitter; the manuscript also correctly recognizes that v1-versus-v2 ordering is not an identity when option count varies. This is a useful clarification even though the underlying chance-corrected statistic is not new.",
  "strengths": [
    "The paper addresses an important evaluation failure mode: a conventional chance line can make a constant or nearly constant prediction interface appear informative.",
    "The distinction between an arm-independent absolute reference and an arm-conditional prediction-marginal reference is conceptually useful and presented with unusually explicit algebra.",
    "The authors disclose substantial negative findings, post-hoc status, regime confounds, power limitations, multiplicity concerns, and repaired integrity defects rather than hiding them.",
    "The treatment of content-side floors usefully shows that tie convention, length unit, and tokenizer are part of the measurement specification.",
    "The manuscript and rendered PDF are generally clear, well organized, and internally cross-referenced."
  ],
  "issues": [
    {
      "id": "TS-1",
      "severity": "major",
      "location": "Table 1 in sections/tab_nulls.tex, MMLU-Pro rows and caption; generated Table in sections/tab_construct_nulls.tex; Section 3.2 variable-option-count formulation",
      "dimension": "technical_soundness, experimental_rigor, limitations_responsible_claims",
      "description": "The flagship MMLU-Pro winner's-curse calibration uses an incompatible null for the item-averaged chance comparison. The paper defines item-averaged chance as mean(1/n_opt)=0.110877. Since option labels are A through the item's legal maximum, A is legal on every item; under uniform random choice among legal options, the expected always-A accuracy is therefore exactly 0.110877, and the expected best-constant floor must be at least 0.110877. Table 1 instead reports E[f_hat]=0.104460 from a nominal ten-category balanced multinomial. That simulation permits labels that are illegal on low-option-count items and cannot represent the legal-choice null underlying the item-averaged chance line. Consequently, the reported p<1e-5 does not establish that the MMLU-Pro floor exceeds the appropriate variable-k chance distribution.",
      "proposed_fix": "Recompute the calibration by preserving every item's observed n_opt and drawing its null label uniformly from its legal options. Report this support-respecting calibration separately from any deliberately naive ten-way reference, and revise all MMLU-Pro significance and headline statements accordingly.",
      "verification_test": "Under the corrected simulation, E[f_hat] must be at least 0.110877. Ship the item-level n_opt vector or stratum counts and the resulting null record, and independently reproduce E[f_hat], its 95th percentile, and the tail probability for the observed 0.116606 floor."
    },
    {
      "id": "TS-2",
      "severity": "major",
      "location": "Introduction paragraphs 1 and 3 and lines 9-12 and 26; Section 3.1 lines 14 and 18; Table 3 in sections/tab_two_nulls.tex; Discussion paragraph 'A reporting rule' and paragraph defining v1/v2 scope",
      "dimension": "technical_soundness, significance, limitations_responsible_claims",
      "description": "The central claim that clearing v1 is a necessary condition for arm-to-arm comparability is not established and is generally false as stated. Because every arm is shifted by the same floor, Delta_floor(a)-Delta_floor(b)=accuracy(a)-accuracy(b). The floor therefore cannot mathematically determine whether a pairwise difference can be estimated or tested. Failing it supports the narrower statement that an arm has not demonstrated absolute performance above the selected constant family; it does not by itself disqualify all evidence about differences between arms. The paper's own v2 analysis further admits item-level alignment in cases not certified by v1.",
      "proposed_fix": "Narrow the recommendation throughout to an absolute above-constant capability gate, rather than a necessary gate for all arm comparison. Alternatively, define a precise non-arithmetic notion of interface comparability and prove why floor failure is necessary under that definition.",
      "verification_test": "After revision, no abstract, introduction, table, checklist, or discussion sentence should declare arms incomparable solely because a common floor was not cleared, unless accompanied by a formal definition and proof that excludes counterexamples where pairwise accuracy differences remain meaningful."
    },
    {
      "id": "TS-3",
      "severity": "major",
      "location": "Section 5.1 paragraph 'Both sides of the flip' and its multiplicity paragraph; Appendix A.4 and A.5 in sections/09a_relocated.tex",
      "dimension": "experimental_rigor, technical_soundness",
      "description": "The claim that the 3/12 count 'survives correction' uses P(Binomial(12,0.05)>=3)=0.0196 even though the manuscript explicitly states that the twelve tests share items and nulls, contain nested arms, and are neither independent nor exchangeable. The arithmetic is correct only for independent, identically calibrated rejection indicators. It therefore cannot rescue the result after both Benjamini-Hochberg and Bonferroni leave 0/12 discoveries. Moreover, testing three chance-side rejections does not directly test the claimed contrast between 3/12 chance rejections and 1/12 floor rejection.",
      "proposed_fix": "Use a joint item-level resampling or randomization procedure that preserves cross-arm dependence and evaluates a predeclared aggregate statistic, such as the difference in rejection counts or a pooled paired effect. If that cannot be justified, present 3/12 versus 1/12 as descriptive only and remove the p=0.0196 inferential rescue.",
      "verification_test": "Provide an empirical joint-null distribution generated with all twelve arms resampled together, and show a valid tail probability for the exact aggregate claim. The null generator must reproduce the observed dependence induced by shared items and nested arms."
    },
    {
      "id": "TS-4",
      "severity": "major",
      "location": "Reproducibility Statement; Appendix subsection 'Evidence provenance' and Table 12 artifact map; submission MANIFEST.json and evidence/claim_evidence_map.tsv",
      "dimension": "reproducibility, experimental_rigor",
      "description": "The frozen submission does not ship the machine-readable records that the manuscript identifies as the sources of its central results. The snapshot contains only build_record.json and claim_evidence_map.tsv under evidence, while the map references absent files such as floor_winners_curse_calibration.json, s2_03_symmetric_inference.json, s2_02_stratified_ordering.json, and the v2 permutation record. Table 12 additionally points to tcodex_out and other paths outside the permitted snapshot. Thus claims that every quantitative result is mechanically bound to evidence cannot be verified from the submitted artifact.",
      "proposed_fix": "Include every central evidence record and the minimal non-writing emitter or analysis code inside the submission artifact, with paths matching the claim map. At minimum this must cover the construct-null calibration, all per-cell v1/v2 summaries, the symmetric inference analysis, power calculations, precision comparison, and integrity repairs.",
      "verification_test": "Starting only from the frozen artifact, resolve every evidence identifier to an existing hash-checked file and independently regenerate all headline table rows and counts. The verification must not depend on excluded directories or mutable external paths."
    },
    {
      "id": "TS-5",
      "severity": "major",
      "location": "Section 3.2 definition of recovery_fraction and materiality; Section 5.1 Qwen3 k14 discussion; Tables 8 and 10 in sections/tab_v2_full.tex and sections/tab_v2_resort.tex",
      "dimension": "experimental_rigor, limitations_responsible_claims",
      "description": "The Qwen3 k14 result is called 'real but immaterial' because its point recovery fraction is 0.049, reported as 9.1% of the intact anchor and just below a 10% threshold. No uncertainty interval is reported for the recovery fraction, its attainable-alignment denominator, or the intact-relative ratio. The reported Delta_perm half-width is 0.188 percentage points, while the distance implied by the reported point quantities to the materiality cutoff is only about 0.026 points. The available uncertainty therefore does not establish that the effect is below the materiality boundary.",
      "proposed_fix": "Bootstrap the entire derived statistic, including Delta_max and the same-family intact anchor. Reserve 'immaterial' for an equivalence-style result whose upper confidence bound is below the threshold; reserve 'material' for a lower bound above it; otherwise call the result indeterminate or trace-level.",
      "verification_test": "Report a confidence interval for damaged recovery divided by intact recovery. The upper 95% bound must be below 0.10 to verify immateriality; if the interval crosses 0.10, all categorical immateriality claims must be withdrawn."
    },
    {
      "id": "TS-6",
      "severity": "minor",
      "location": "Abstract final precision claim; Introduction main finding on fp32; Section 5.4",
      "dimension": "limitations_responsible_claims, clarity",
      "description": "The full-fp32 experiment covers OLMo-2 keep8 on MMLU, but the abstract says it rules out a numerical-tie explanation for 'the measurement failure' without immediately restricting that conclusion to this cell. It rules out exact bf16 ties as the repair mechanism for the tested arm and benchmark, not as a general mechanism across all four families, damage regimes, or MMLU-Pro.",
      "proposed_fix": "Scope the abstract and conclusion explicitly to OLMo-2 keep8 on MMLU, or add representative precision controls from the other regimes before making a general mechanism claim.",
      "verification_test": "The no-new-experiment verification is a text audit showing that every causal precision statement names the tested arm and benchmark. A broader statement requires full-precision results on representative non-OLMo and MMLU-Pro cells."
    }
  ],
  "score_ceiling_under_current_evidence": 6.5,
  "predicted_score_after_required_changes": 6,
  "evidence_that_would_raise_score": [
    "A support-respecting variable-option-count calibration that preserves each MMLU-Pro item's legal answer set and shows what remains of the floor-versus-chance result.",
    "A principled narrowing of v1 from universal comparability gate to absolute above-constant claim, or a formal definition and proof supporting the stronger necessity statement.",
    "A dependence-preserving joint analysis of the symmetric flip counts, with an aggregate statistic specified and calibrated under the joint null.",
    "Confidence intervals or equivalence tests for the recovery-based materiality decisions.",
    "A self-contained artifact containing all evidence records and enough non-mutating code to reproduce every headline table and count.",
    "Robustness of v2 to subject- or category-stratified permutation and an explicitly justified resampling unit."
  ],
  "evidence_that_would_lower_score": [
    "The corrected legal-option MMLU-Pro null places 0.116606 inside ordinary selection noise and removes the flagship calibrated floor-versus-chance result.",
    "A joint dependence-preserving null shows that 3/12 chance-side rejections are unexceptional or that the difference between the two reference counts is unsupported.",
    "The per-item records fail to reproduce the printed cell values, repaired-run provenance, or integrity assertions.",
    "Recovery-ratio uncertainty shows that the supposedly immaterial Qwen3 k14 exception is compatible with material signal.",
    "Subject- or construction-aware permutations eliminate v2 signals attributed to item-level information."
  ],
  "review_limitations": [
    "The allowed snapshot omits the named per-item records, calibration files, permutation summaries, and symmetric-inference record, so most empirical p-values and counts could not be independently recomputed.",
    "I verified all 34 included files against the manifest hashes and inspected both manuscript sources and the rendered PDF, but did not run the authors' checker scripts as prohibited.",
    "External literature access was prohibited by the blind-review instructions, so load-bearing citation claims and the novelty claim for varying-k stratification could not be independently confirmed.",
    "The exchangeability of items within n_opt strata and the appropriate bootstrap clustering unit cannot be assessed without benchmark metadata and per-item records; this remains an unresolved assumption rather than a confirmed error."
  ]
}
```

## PROSE REVIEW

### Summary and assessment

This paper argues that multiple-choice evaluation should report an explicit input-blind null before interpreting model-arm differences. It separates an arm-independent best-constant floor (v1) from an arm-conditional, option-count-stratified permutation null (v2), then applies these references to damaged language-model interfaces across several benchmarks.

The problem is important, and the paper contains a useful distinction between absolute performance relative to a constant baseline and item-level alignment beyond an arm’s own prediction marginal. It is also unusually candid about negative results, confounds, post-hoc analysis, multiplicity, and repaired experimental defects.

However, I find multiple decision-relevant technical problems. Most importantly, the paper’s recommended v1 gate is stronger than its mathematics supports, and the flagship MMLU-Pro calibration uses a null incompatible with the benchmark’s variable legal answer set. The symmetric flip analysis then attempts to recover significance using a binomial calculation whose independence assumption directly contradicts the manuscript’s own dependence disclosure. These are major rather than fatal issues: the narrower reporting recommendation remains potentially useful, but the present central claims are not technically secure.

### Strongest verified contribution

The strongest part is the algebraic separation of the two null questions:

- v1 provides a common arm-independent absolute reference.
- v2 asks whether an arm’s predictions align with items beyond the alignment expected from its own prediction marginal.

The v2 quantity is the numerator of Cohen’s kappa within option-count strata, and its zero value for a legal pure constant emitter follows directly. The manuscript also correctly notices that the ordering between v1 and stratified v2 is not generally a theorem when option count varies. These are useful clarifications, even though the underlying chance-corrected statistic is established prior art.

### Major concerns

#### 1. Major — the MMLU-Pro balanced-null calibration is incompatible with variable option count

**Location:** Table 1 (`tab_nulls.tex`), its two MMLU-Pro rows and caption; generated construct-null table; Section 3.2.

The item-averaged chance line is reported as

\[
\frac{1}{n}\sum_i \frac{1}{k_i}=0.110877.
\]

Because MMLU-Pro uses labels A through the item’s legal maximum, A is legal on every item. Under random choice uniformly among each item’s legal options, expected always-A accuracy is therefore exactly 0.110877. Consequently, the expected maximum label marginal must be at least 0.110877.

The table instead reports an expected maximum of 0.104460. This reveals that the calibration used a nominal ten-category uniform multinomial, which can assign gold labels that are illegal on low-option-count items. That may calibrate an artificial global ten-letter balance hypothesis, but it cannot calibrate the legal-choice null associated with the item-averaged chance line. Therefore the reported \(p<10^{-5}\) does not establish the claimed MMLU-Pro result against the appropriate variable-\(k\) null.

**Fix:** preserve every item’s observed `n_opt`, sample uniformly over its legal labels, and recompute the distribution of the maximum marginal.

**Verification:** the corrected expected maximum must be at least 0.110877; the artifact should permit independent reproduction of its mean, 95th percentile, and tail probability.

This is the strongest statistical objection because MMLU-Pro supplies the paper’s main powered example and its calibrated “floor differs from chance” claim.

#### 2. Major — v1 is not a necessary condition for arm-to-arm comparability

**Location:** Introduction; Section 3.1; Table 3; Discussion.

The manuscript repeatedly states that an arm failing the best-constant floor cannot support an arm comparison or that v1 “certifies mutual comparability.” But for two arms \(a\) and \(b\),

\[
\Delta_{\mathrm{floor},a}-\Delta_{\mathrm{floor},b}
=\mathrm{acc}_a-\mathrm{acc}_b,
\]

because the same floor is subtracted from both. Thus floor clearance cannot be a mathematical prerequisite for estimating a pairwise difference.

A floor failure supports a narrower conclusion: the arm has not demonstrated absolute performance above the selected constant family. That can be an important warning, but it does not make every arm-to-arm comparison invalid. Indeed, the paper’s own v2 analysis recognizes that an arm may have item-level alignment not captured by its v1 status.

**Fix:** recast v1 as a gate for an absolute “above the best constant” capability claim, not as a universal comparability gate. Alternatively, define a more specialized notion of construct comparability and prove the claimed necessity.

**Verification:** no revised sentence should declare two arms incomparable solely because the common floor was not cleared, absent such a formal definition and proof.

This conceptual overreach affects the abstract, introduction, method, checklist, and conclusion, so it is not merely terminological.

#### 3. Major — the binomial multiplicity rescue assumes away the acknowledged dependence

**Location:** Section 5.1 and Appendix A.5.

The manuscript correctly reports that Benjamini–Hochberg and Bonferroni leave zero discoveries on both sides of the 3/12-versus-1/12 comparison. It then states that “what survives correction is the count itself,” because observing at least three rejections under a \(\mathrm{Binomial}(12,0.05)\) model has probability 0.0196.

The arithmetic is correct, but the model is not justified. Elsewhere the paper explicitly states that these tests share items and nulls, contain nested arms, and are neither independent nor exchangeable. The binomial tail assumes independent, identically calibrated rejection indicators. Under strong positive dependence, for example, the probability of at least three rejections can be 0.05 rather than 0.0196.

Furthermore, testing whether the chance side contains at least three rejections is not a direct test of the claimed difference between three chance-side and one floor-side rejection.

**Fix:** jointly resample all twelve arms and both references while preserving shared-item dependence, then calibrate a predeclared aggregate statistic. Otherwise, the counts should be explicitly descriptive.

**Verification:** provide the empirical joint-null distribution and tail probability for the actual aggregate claim.

Until then, the symmetric result is best described as a descriptive 3/12 versus 1/12 pattern with no multiplicity-corrected cell and no valid aggregate \(p\)-value.

#### 4. Major — the promised evidence records are absent from the frozen artifact

**Location:** Reproducibility Statement; Appendix evidence-provenance subsection and artifact map; snapshot manifest.

The paper states that every numerical claim is mechanically bound to a machine-readable record. However, the frozen snapshot contains only `build_record.json` and `claim_evidence_map.tsv` in its evidence directory. The claim map references absent files including:

- the winner’s-curse calibration record,
- the symmetric-inference record,
- the stratified-ordering record,
- the v2 permutation record.

The appendix also resolves evidence identifiers to paths under `tcodex_out` and other locations outside the permitted snapshot. I therefore could not verify the central bootstrap values, permutation values, cell counts, repaired-run provenance, or generated-table claims.

This is not evidence that those numbers are false, but it is a confirmed artifact failure relative to the paper’s reproducibility claims.

**Fix:** include all central records and sufficient read-only analysis code in the frozen artifact.

**Verification:** every evidence identifier must resolve within the artifact, and an independent user must be able to regenerate all headline tables and counts without external mutable paths.

#### 5. Major — “real but immaterial” is not supported by uncertainty around the materiality threshold

**Location:** Section 3.2; Qwen3 `k14` discussion in Section 5.1; v2 full and re-sort tables.

Qwen3 `k14` is called a “real but immaterial” exception because its recovery point estimate is 0.049, or 9.1% of the intact anchor, just below the 10% threshold. But the paper reports no interval for the recovery fraction, its attainable-alignment denominator, or its ratio to the intact anchor.

Using the reported quantities, the point estimate is only about 0.026 percentage points below the implied materiality boundary, while the reported half-width for \(\Delta_{\mathrm{perm}}\) is 0.188 points. An effect can be statistically nonzero without being demonstrated to lie below a materiality threshold. Failure of a point estimate to exceed the threshold is not an equivalence test.

**Fix:** bootstrap the entire damaged-to-intact recovery ratio, including both denominators.

**Verification:** “immaterial” requires the upper confidence bound to be below 0.10; “material” requires the lower bound to exceed 0.10. Otherwise the result should be called indeterminate or trace-level.

### Minor concern

The fp32 experiment is limited to OLMo-2 `keep8` on MMLU. It convincingly shows that exact bf16 ties do not explain or repair that particular failure. The abstract’s broader wording risks sounding as though numerical ties were excluded as a mechanism across all four families and benchmarks. This can be fixed by explicitly naming the tested cell whenever drawing the causal conclusion.

### Strengths

1. The motivating failure mode is real and practically important: conventional chance can credit an empty constant-output interface.
2. The distinction between arm-independent and arm-conditional nulls is valuable.
3. The content-floor discussion makes useful measurement choices explicit, particularly tie handling, length units, and tokenizer dependence.
4. The paper is unusually transparent about retracted hypotheses, post-hoc status, underpowered benchmarks, regime confounds, and integrity repairs.
5. The manuscript is dense but generally clear, with coherent cross-referencing and readable tables.

### Recommendation and score rationale

I assign **4/10 (reject)** with **confidence 4/5**.

I do not view the paper as fundamentally without value. The narrower recommendation—report an explicit constant or input-blind reference before making an absolute capability claim—is useful. But the present paper elevates that recommendation into a necessary gate for all arm comparison without a valid argument, uses the wrong support structure for a flagship variable-\(k\) calibration, and makes an invalid dependence-blind multiplicity argument. The absence of the promised evidence records prevents independent adjudication of most remaining empirical claims.

The score ceiling without new model experiments is approximately **6.5**. Existing predictions should suffice to perform the corrected legal-option calibration, a joint dependence-preserving analysis, uncertainty-aware materiality tests, and artifact packaging. With those changes and appropriate claim narrowing, I would expect a score around **6**, subject to what the corrected analyses show.

### Review limitations

I confined review to the mandated snapshot and verified all 34 included files against their manifest hashes. I inspected the source and rendered PDF but did not run any author checker. The central machine-readable records were absent, so I could not independently reproduce most empirical statistics. External literature access was prohibited, so I also could not verify the load-bearing novelty claim for varying-option-count stratification or the local support of recent citations.