```json
{
  "reviewer_id": "X1-codex-novelty",
  "round": 4,
  "snapshot_sha256": "7fcb9ccc55c5c1d6ad1de868215b1d9253d0695a2f996340226aa6a6fa10db5a",
  "role": "novelty_positioning",
  "overall_score": 5,
  "confidence": 4,
  "recommendation": "weak_reject",
  "dimension_scores": {
    "novelty": 3,
    "significance": 4,
    "technical_soundness": 3,
    "experimental_rigor": 3,
    "clarity": 4,
    "reproducibility": 2,
    "citation_integrity": 2,
    "limitations_responsible_claims": 4
  },
  "paper_summary": "The paper proposes null-calibrated multiple-choice evaluation: an arm-independent best-constant or other input-blind floor is used as a necessary gate before comparing model arms, and a separate arm-conditional within-option-count permutation statistic asks whether a prediction vector contains item-level alignment beyond its output marginal. It re-analyzes structurally damaged language models, studies content-floor sensitivity to tie convention, length unit, and tokenizer, and reports that chance-versus-floor choice can alter interpretations.",
  "strongest_verified_contribution": "The clearest contribution is the operational synthesis: distinguish an arm-independent interface-comparability floor from an arm-conditional item-information null, and specify the former for content scoring by tie convention, length unit, and tokenizer. The algebraic definitions and the constant-emitter property are internally clear, and the manuscript responsibly disclaims novelty for Cohen's kappa numerator and for generic length/tokenizer effects.",
  "strengths": [
    "The paper draws a useful conceptual distinction between interface comparability and arm-specific item-level information, rather than forcing one null to answer both questions.",
    "Novelty disclaimers are unusually explicit for several ingredients: majority baselines, the kappa numerator, the constant-rater-zero property, option-count-aware chance correction, and generic length/tokenizer dependence.",
    "The prose is careful about necessary-versus-sufficient validity, post-hoc v2 analysis, regime confounding, underpowered small benchmarks, and multiplicity.",
    "The exact arm-independent MCQA gate and exact legality-aware blocked-permutation construction were not found to be fully preempted in the literature checked.",
    "The paper reports self-audits and narrowed claims rather than hiding adverse or null findings."
  ],
  "issues": [
    {
      "id": "N1",
      "severity": "major",
      "location": "Introduction contribution framing (Section 1, paragraphs 3-4); Related Work, 'Selection bias and MC interfaces' and 'Nulls as interpretive controls'; Method Section 3.2",
      "dimension": "novelty, citation_integrity",
      "description": "Cho et al. (ICLR 2026) is materially under-positioned. That work does more than compare symbols, cloze, and hybrid formats: it removes the question, computes a choice-driven baseline, separates question- and choice-driven components, and constructs NPSQ to eliminate choice-only influence. It is therefore the closest MCQA-specific structural neighbor to v2, whereas the paper calls Hewitt and Liang the closest structural precedent. Cho does not preempt v1 or the prediction-marginal permutation, but it preempts the broader idea of removing question/input dependence to assess whether an MCQA score reflects item-level question information.",
      "proposed_fix": "Add a formula-level comparison table for Cho/NPSQ, v1, and v2 covering inputs, null distribution, preserved quantities, statistic, computational cost, and inferential question. Replace the 'closest structural precedent' sentence with a bounded claim that recognizes Cho as the closest MCQA-specific precedent.",
      "verification_test": "A reader can identify a mathematical, not merely verbal, distinction for every Cho-v1-v2 pair and can see that the residual novelty is the benchmark-derived arm-independent gate and the output-marginal blocked randomization, not generic removal of question dependence."
    },
    {
      "id": "N2",
      "severity": "major",
      "location": "Method Section 3.2, paragraphs 'The statistic is not new' and 'only the varying-k stratification is ours'; Appendix A.2 'Prior art on option-count-aware chance correction'",
      "dimension": "novelty, citation_integrity, technical_soundness",
      "description": "The statement that only the varying-option-count stratification is new is not adequately established. Stratified-kappa methods predate this paper, including Barlow, Lai, and Azen (1991), 'A comparison of methods for calculating a stratified kappa' (DOI 10.1002/sim.4780100913), while the manuscript already cites pooled-kappa work. I could not confirm that these methods implement the exact item-dependent legal-label blocked permutation, so this is not a confirmed preemption; however, the current blanket claim is unsupported and the novelty boundary remains decision-relevant uncertainty.",
      "proposed_fix": "Cite and derive the closest stratified-kappa estimator alongside the proposed expected-agreement term. If equivalent, retract stratification novelty. If different, narrow the claim to the exact contribution, such as blocked permutation under item-dependent legal-label structural zeros, applied as an MCQA gate.",
      "verification_test": "On the same variable-option-count contingency tables, give both estimators algebraically and provide the smallest explicit example where they differ; identify the exact legality constraint responsible."
    },
    {
      "id": "N3",
      "severity": "major",
      "location": "Related Work, especially 'Input-blind and constant baselines' and 'Selection bias and MC interfaces'",
      "dimension": "novelty, citation_integrity",
      "description": "The paper omits the broad ACL 2025 synthesis 'Which of These Best Describes Multiple Choice Evaluation with LLMs?' (DOI 10.18653/v1/2025.acl-long.169), which reviews choices-only/partial-input shortcuts and includes 'Calibrated Scoring Can Deter Guessing.' Its calibration means confidence/probability scoring, negative marking, or elimination rather than reference-line calibration, so it does not preempt this paper; nevertheless, omitting the closest recent field synthesis makes the positioning incomplete. Balepur et al. (ACL 2024) is also described too narrowly: it explicitly recommends choices-only baselines as stronger alternatives to majority baselines.",
      "proposed_fix": "Cite the ACL 2025 synthesis, distinguish the two meanings of calibration, and state the full overlap with Balepur et al. 2024 before identifying the residual novelty: a cheap benchmark-derived constant/input-blind reference used as an arm-comparison gate.",
      "verification_test": "The revised related work explicitly separates predictor/interface diagnostics, confidence or guessing calibration, choices-only artifact baselines, and reference-line null calibration, with one precise residual claim for this paper."
    },
    {
      "id": "N4",
      "severity": "major",
      "location": "Introduction, contribution paragraph beginning 'We study null calibration'; Related Work, 'Input-blind and constant baselines'",
      "dimension": "novelty, citation_integrity",
      "description": "The generic premise that a model should clear an input-independent baseline as a necessary sanity check is not new beyond MCQA. Pries et al. (2023), 'The optimal input-independent baseline for binary classification: The Dutch Draw' (DOI 10.1111/stan.12297), explicitly studies the best feature-independent baseline and argues that a developed model should at least beat such baselines. Its formal results are binary-classification-specific and do not preempt the paper's exact multiclass MCQA floor or content construction, but omitting it overstates the conceptual novelty of turning an input-blind reference into a gate.",
      "proposed_fix": "Cite the general input-independent-baseline literature and reframe the contribution as an MCQA-specific operationalization, specification audit, and empirical demonstration rather than the generic invention of an input-blind sanity gate.",
      "verification_test": "The introduction states a novelty claim that remains true after substituting the prior work's generic rule; the related-work comparison explicitly separates binary general theory from this paper's multiclass, variable-legality, and content-scoring cases."
    },
    {
      "id": "R1",
      "severity": "major",
      "location": "Reproducibility Statement; Appendix A.15 'Evidence provenance' and Table 12; all result tables citing E-A through E-I and E-CAL; frozen snapshot evidence directory",
      "dimension": "reproducibility, experimental_rigor, technical_soundness",
      "description": "The frozen submission does not ship the evidence it says binds the quantitative claims. Only build_record.json and claim_evidence_map.tsv are present, while Table 12 points to EVIDENCE_PACK.md, per-cell records, permutation-null JSON, floor calibration JSON, power records, and emitters/checkers that are absent. The supplied claim map also references forbidden/non-shipped tcodex_out paths and does not contain the underlying per-item data. Thus I could verify arithmetic visible in the manuscript, but not the headline cell counts, bootstrap/permutation p-values, power claims, fp32 comparison, or table provenance.",
      "proposed_fix": "Ship a self-contained, anonymized evidence bundle containing every record referenced by E-A through E-I and E-CAL, per-item sufficient statistics or records for the central tables, the exact configs/seeds, and read-only verification scripts. Make all artifact paths resolve inside the frozen submission.",
      "verification_test": "Starting only from the frozen snapshot on a clean machine, a reviewer can regenerate Tables 1-10 and every headline count/p-value, and a manifest check reports zero missing evidence identifiers or external repository paths."
    },
    {
      "id": "R2",
      "severity": "minor",
      "location": "evidence/build_record.json versus manuscript/main.pdf and MANIFEST.json",
      "dimension": "reproducibility, clarity",
      "description": "The build record is stale: it records a 22-page PDF of 355196 bytes with SHA-256 56a376..., whereas the shipped PDF is 24 pages, 366583 bytes, SHA-256 1fbaaf.... It also says visual inspection was not done. The manifest hashes the shipped PDF correctly, so the snapshot itself is identifiable, but the claimed build provenance does not correspond to it.",
      "proposed_fix": "Regenerate the build record from the exact frozen source/PDF and include a completed visual-inspection record or remove the implication that the current build gate covers the shipped PDF.",
      "verification_test": "The build record's page count, byte count, and SHA-256 exactly equal the shipped main.pdf, and a clean rebuild produces that same hash or a documented reproducible-equivalence record."
    },
    {
      "id": "S1",
      "severity": "major",
      "location": "Results Section 5.1 and Appendix A.5, claim that observing at least 3 rejections among 12 has binomial probability 0.0196 under the global null",
      "dimension": "experimental_rigor, technical_soundness, limitations_responsible_claims",
      "description": "The 0.0196 calculation is the Binomial(12, 0.05) independent-test tail, but the paper elsewhere emphasizes that the tests share items, nested arms, and a common null and are neither independent nor exchangeable. Therefore the binomial tail is not justified by the stated dependence structure and cannot be presented as what 'survives correction.' This matters because no individual cell survives BH or Bonferroni, leaving this count-level claim as the attempted multiplicity-resilient evidence.",
      "proposed_fix": "Remove the binomial inference or replace it with a valid joint test that preserves cross-cell dependence, for example a synchronized item-level permutation/bootstrap recomputing all 12 statistics and the rejection-count/max statistic. Alternatively present 3/12 descriptively only.",
      "verification_test": "Under a simulated global null with the observed shared-item/nested-arm dependence preserved, the proposed joint test controls type-I error at 0.05; the reported count-level p-value is then produced by that test rather than an independence formula."
    },
    {
      "id": "C1",
      "severity": "minor",
      "location": "Introduction opening paragraph, claim that Bean et al. provide 27 actionable checklist items",
      "dimension": "citation_integrity, clarity",
      "description": "The published Bean et al. checklist contains 28 active checkbox items, not 27. The local substantive point survives: none explicitly asks for a null, chance level, or constant predictor.",
      "proposed_fix": "Change 27 to 28 and retain the narrower statement about the missing checklist item.",
      "verification_test": "Count the active checkbox entries in the published checklist and ensure the manuscript count matches."
    },
    {
      "id": "C2",
      "severity": "minor",
      "location": "Related Work, MC-interface discussion",
      "dimension": "citation_integrity, novelty",
      "description": "The related-work taxonomy omits Molfese et al., Findings of ACL 2025, 'Right Answer, Wrong Score' (DOI 10.18653/v1/2025.findings-acl.950), which studies how prompt format, label binding, log-probability scoring, and answer extraction can yield misleading MCQA comparisons. It does not preempt null calibration but is directly adjacent and helps distinguish interface/extraction failures from reference-line misspecification.",
      "proposed_fix": "Add the work and organize related work into predictor debiasing, interface/extraction reliability, input-deprived baselines, and reference/null specification.",
      "verification_test": "The related-work section names each neighboring failure mode and states why the proposed null gate is complementary rather than redundant."
    }
  ],
  "score_ceiling_under_current_evidence": 7,
  "predicted_score_after_required_changes": 7,
  "evidence_that_would_raise_score": [
    "A self-contained artifact that regenerates all headline tables, p-values, power analyses, and integrity claims from shipped per-item records.",
    "A formal comparison with stratified-kappa prior art showing that the legality-aware blocked estimator is genuinely distinct, or an honest narrowing if it is not.",
    "A formula-level Cho/NPSQ-versus-v1/v2 comparison and complete positioning against the 2025 MCQA evaluation literature.",
    "A valid dependence-aware joint multiplicity analysis replacing the independent Binomial(12, 0.05) count argument.",
    "A corrected build record tied to the exact 24-page frozen PDF."
  ],
  "evidence_that_would_lower_score": [
    "A derivation showing that prior stratified-kappa methods are algebraically identical to v2's claimed novel stratification.",
    "Regenerated evidence that changes the 14/15, 3/12 versus 1/12, 0/60, or v2 re-sorting conclusions.",
    "Discovery that the absent evidence files cannot reproduce the printed tables or that the stale build record reflects source/PDF divergence beyond metadata.",
    "A dependence-aware global-null simulation showing the headline rejection count is common under the actual correlation structure."
  ],
  "review_limitations": [
    "The frozen snapshot did not contain the central machine-readable evidence records or per-item predictions named in Appendix Table 12, so central empirical claims could not be independently confirmed or refuted.",
    "I verified closest-work content using official ACL Anthology/proceedings where available, official ICLR/ICML/NeurIPS venue pages or identifiers, arXiv papers, and DOI metadata; OpenReview itself intermittently required browser verification.",
    "I found directly relevant stratified-kappa prior art but could not obtain the full paywalled 1991 article, so equivalence to the paper's exact legality-aware blocked permutation remains unresolved rather than refuted.",
    "I did not inspect any prohibited review history, author notes, state, git history, or non-snapshot paper files, and I did not run author checker scripts."
  ]
}
```

## PROSE REVIEW

### Summary and overall assessment

This paper argues that multiple-choice model scores should not be interpreted against nominal chance alone. It proposes two distinct read-outs:

1. **v1:** an arm-independent, input-blind floor—typically the best constant label—for determining whether an interface supports meaningful arm comparisons; and  
2. **v2:** an arm-conditional permutation null preserving the arm’s prediction marginal within option-count strata, intended to test for item-level information.

The distinction is useful, and the manuscript is unusually responsible about several boundaries: passing the floor is necessary rather than sufficient; v2 is post-hoc on the reported cells; the kappa numerator and constant-rater-zero identity are not new; generic length bias and tokenizer dependence are not claimed; family and damage regimes are confounded; and small-benchmark nulls are often underpowered.

I did **not** find a confirmed fatal duplication of the exact arm-independent MCQA gate or the exact legality-aware blocked permutation. However, the novelty is narrower than presented, its boundary against the closest work is incomplete, and the frozen artifact does not contain the evidence records needed to verify the headline empirical claims. I therefore recommend **weak reject (5)**, with a plausible path to a 7 through positioning, analysis, and artifact corrections that do not inherently require new model experiments.

### Strongest verified contribution

The strongest contribution is the **operational separation of two inferential questions**:

- whether an interface clears a common arm-independent reference sufficiently to support comparisons; and
- whether a particular arm’s prediction vector contains item-level alignment beyond its own output marginal.

This is clearer than applying either chance or the best constant indiscriminately. The paper also makes a useful reproducibility point that a content-side longest-option floor depends jointly on tie convention, length unit, and tokenizer.

### Strengths

- The conceptual v1/v2 distinction is useful and clearly explained.
- The mathematical definitions are understandable, and the constant-emitter calculation is correct as an algebraic consequence of the chosen expected-agreement term.
- The manuscript avoids treating a floor pass as full construct validation.
- It reports negative and contradictory findings, including retracted mechanisms and failed causal contrasts.
- It is transparent about post-hoc analysis, multiplicity, power limitations, and regime confounding.
- Several important prior-art boundaries are stated honestly rather than claimed as new.

### Major issues

#### N1 — Cho et al. is the closest MCQA-specific precedent but is under-positioned

**Location:** Section 1 contribution framing; Section 2 paragraphs on MC interfaces and interpretive controls; Section 3.2.  
**Severity:** Major.  
**Dimensions:** Novelty and citation integrity.

Cho et al., *Choices Speak Louder than Questions* (ICLR 2026), does substantially more than compare symbols, cloze, and hybrid formats. It removes the question, computes a choice-driven component, separates question- and choice-driven influence, and defines NPSQ to remove choice-only influence. This is structurally much closer to v2 than the probing work identified as the “closest structural precedent.” ([proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7ae2b0eedc560e3afbffd68445b8a220-Abstract-Conference.html))

Cho does not preempt v1, nor does it preserve the observed prediction marginal through a blocked randomization. It does, however, preempt the broader contribution framing around removing question/input dependence to determine whether MCQA behavior reflects item-level question information.

**Required fix:** Add a formula-level Cho/NPSQ–v1–v2 comparison covering the available inputs, preserved quantities, null distribution, statistic, cost, and inferential question. Recognize Cho as the closest MCQA-specific precedent.

**Verification:** A reader should be able to identify a mathematical distinction for every pair; the residual novelty should not rely on different terminology for closely related operations.

#### N2 — The claimed novelty of varying-option-count stratification is unresolved

**Location:** Section 3.2, especially “only the varying-\(k\) stratification is ours”; Appendix A.2.  
**Severity:** Major.  
**Dimensions:** Novelty, citation integrity, and technical soundness.

The paper claims that prior kappa-family work assumes one global category count and that stratification for item-varying option count is new. However, stratified-kappa methods long predate this work. Barlow, Lai, and Azen explicitly study a stratified kappa built from stratum-level kappas, and the manuscript itself cites pooled-kappa literature. ([europepmc.org](https://europepmc.org/article/MED/1925174))

I could not verify that the 1991 construction is equivalent to this paper’s exact blocked permutation with item-dependent legal labels. Thus, this is **not a confirmed preemption**. But the present blanket novelty claim is insufficiently supported.

**Required fix:** Derive the closest stratified-kappa expected-agreement term beside the proposed estimator. If they coincide, retract the stratification novelty. If they differ, narrow the contribution to the exact differentiator—for example, blocked randomization under item-dependent structural zeros in the legal label set.

**Verification:** Provide the smallest contingency-table example on which the estimators differ and identify exactly which legality constraint produces the difference.

#### N3 — The closest recent MCQA evaluation synthesis is omitted

**Location:** Section 2, input-blind baselines and MC-interface paragraphs.  
**Severity:** Major.  
**Dimensions:** Novelty and citation integrity.

The related work omits the ACL 2025 synthesis *Which of These Best Describes Multiple Choice Evaluation with LLMs?* That paper reviews choices-only and partial-input shortcuts and discusses “calibrated scoring,” although its meaning is confidence/probability scoring, negative marking, and elimination rather than reference-line calibration. This does not preempt the present proposal, but the distinction needs to be made explicitly. ([aclanthology.org](https://aclanthology.org/2025.acl-long.169/))

Balepur et al. 2024 is also described somewhat narrowly. Besides studying dataset cheatability, it explicitly advocates stronger MCQA baselines and choices-only evaluation, so the overlap is more operational than the current paragraph suggests. ([aclanthology.org](https://aclanthology.org/2024.acl-long.555/))

**Required fix:** Cite the ACL 2025 synthesis, distinguish the two meanings of calibration, and state the strongest overlap with Balepur et al. before defining the residual contribution.

**Verification:** The revised taxonomy should separately cover choices-only artifact baselines, predictor debiasing, confidence/guessing calibration, interface reliability, and reference-line null calibration.

#### N4 — The general input-independent-baseline gate is established outside MCQA

**Location:** Section 1 contribution paragraph; Section 2 input-blind baseline paragraph.  
**Severity:** Major.  
**Dimensions:** Novelty and citation integrity.

Pries et al., *The Optimal Input-Independent Baseline for Binary Classification: The Dutch Draw*, explicitly argues that a developed model should at least beat feature-independent baselines and studies which such baseline is optimal. Its formal result is for binary classification and therefore does not preempt the paper’s exact multiclass MCQA floor, content-side construction, or variable legal-label setting. It nevertheless preempts the broad conceptual framing that using an input-independent reference as a necessary sanity gate is itself new. ([onlinelibrary.wiley.com](https://onlinelibrary.wiley.com/doi/pdf/10.1111/stan.12297))

**Required fix:** Position the work as an MCQA-specific operationalization, specification audit, and empirical demonstration of a more general evaluation principle.

**Verification:** The revised contribution statement must remain true after replacing the claimed generic principle with the prior binary-classification result.

#### R1 — The evidence package described by the paper is not in the frozen submission

**Location:** Reproducibility Statement; Appendix A.15 and Table 12; all captions citing E-A through E-I and E-CAL.  
**Severity:** Major.  
**Dimensions:** Reproducibility, experimental rigor, and technical soundness.

The snapshot contains only `build_record.json` and `claim_evidence_map.tsv` under `evidence/`. The paper’s provenance table instead points to an evidence pack, per-cell records, a permutation-null JSON file, a winner’s-curse calibration JSON file, power records, and emitters/checkers that are absent. Some listed paths point outside the shipped snapshot.

Consequently, I could inspect manuscript arithmetic, but I could not independently confirm or refute:

- the 14/15 and 0/60 aggregates;
- the 3/12 versus 1/12 bootstrap result;
- permutation and bootstrap p-values;
- the power claims;
- the fp32 comparison; or
- the claimed table-to-record bindings.

**Required fix:** Ship a self-contained, anonymized evidence bundle with every referenced record, sufficient per-item data, exact configurations and seeds, and read-only reproduction scripts.

**Verification:** On a clean machine using only the frozen artifact, Tables 1–10 and all headline counts and p-values should regenerate, with zero unresolved evidence identifiers.

#### S1 — The binomial rejection-count argument assumes independence that the paper denies

**Location:** Section 5.1 and Appendix A.5.  
**Severity:** Major.  
**Dimensions:** Experimental rigor, technical soundness, and responsible claims.

The reported value \(0.0196\) is the probability that a \(\mathrm{Binomial}(12,0.05)\) variable is at least three. Arithmetically, that value is correct. Statistically, however, the binomial model assumes independent Bernoulli tests. The paper explicitly states that the twelve tests share items, nested arms, and a common null and are neither independent nor exchangeable.

The issue is decision-relevant because no cell survives BH or Bonferroni, and the binomial count is then presented as the evidence that “survives correction.” Under the stated dependence structure, that conclusion does not follow.

**Required fix:** Either report 3/12 descriptively or use a synchronized item-level permutation/bootstrap that recomputes all twelve statistics and calibrates a rejection-count or max statistic under the joint null.

**Verification:** A simulation preserving the observed cross-cell dependence should demonstrate type-I error control at 0.05 for the replacement global test.

### Minor issues

#### R2 — Stale build provenance

**Location:** `evidence/build_record.json`.  
**Severity:** Minor.  
**Dimensions:** Reproducibility and clarity.

The build record describes a 22-page, 355,196-byte PDF with SHA-256 beginning `56a376`, while the frozen PDF has 24 pages, 366,583 bytes, and SHA-256 beginning `1fbaaf`. The manifest correctly identifies the shipped PDF, but the build record does not correspond to it.

**Fix:** Regenerate the record from the exact frozen build and document visual inspection.

**Verification:** Page count, size, and hash must match the shipped PDF exactly.

#### C1 — Incorrect Bean et al. checklist count

**Location:** Opening paragraph of Section 1.  
**Severity:** Minor.  
**Dimensions:** Citation integrity and clarity.

The published checklist contains 28 active entries, not 27. The substantive statement that none explicitly demands a null, chance level, or constant predictor remains supportable. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1967e0fc3aa6cbbace562f5cb8e3954e-Abstract-Datasets_and_Benchmarks_Track.html))

**Fix:** Change 27 to 28.

**Verification:** Count the active checklist entries in the published source.

#### C2 — Missing adjacent work on MCQA extraction and scoring reliability

**Location:** Section 2 MC-interface discussion.  
**Severity:** Minor.  
**Dimensions:** Citation integrity and novelty.

Molfese et al., *Right Answer, Wrong Score* (Findings of ACL 2025), directly studies misleading MCQA comparisons arising from prompting, answer constraints, and extraction strategies. It does not preempt null calibration, but it is relevant for distinguishing interface/extraction failures from reference-line misspecification. ([aclanthology.org](https://aclanthology.org/2025.findings-acl.950/))

**Fix:** Add it and organize neighboring work by failure mode.

**Verification:** The related-work taxonomy should distinguish predictor bias, interface/extraction inconsistency, input-deprived baselines, and null/reference misspecification.

### Score rationale

The paper addresses an important evaluation problem and contains a useful operational synthesis. I found no confirmed fatal novelty duplication. However:

- the closest MCQA-specific work is under-positioned;
- the varying-\(k\) novelty claim remains unresolved against stratified-kappa literature;
- the broad input-independent-gate principle has prior general treatment;
- relevant recent MCQA evaluation work is omitted;
- the central empirical evidence is not contained in the frozen artifact; and
- one key multiplicity-resilient statistical argument uses an unjustified independence model.

These issues prevent a positive recommendation in the current form. The paper’s **ceiling without new model experiments is approximately 7**: complete artifact release, corrected joint inference, and precise novelty narrowing could make this a solid and useful evaluation-methodology contribution.