```json
{
  "reviewer_id": "X5-codex-repro",
  "round": 4,
  "snapshot_sha256": "7fcb9ccc55c5c1d6ad1de868215b1d9253d0695a2f996340226aa6a6fa10db5a",
  "role": "reproducibility_provenance",
  "overall_score": 4,
  "confidence": 4,
  "recommendation": "reject",
  "dimension_scores": {
    "novelty": 3,
    "significance": 4,
    "technical_soundness": 2,
    "experimental_rigor": 2,
    "clarity": 4,
    "reproducibility": 1,
    "citation_integrity": 2,
    "limitations_responsible_claims": 4
  },
  "paper_summary": "The paper argues that multiple-choice evaluations should be calibrated against construct-appropriate input-blind nulls rather than nominal chance. It distinguishes an arm-independent best-constant floor for interface comparability from an arm-conditional, option-count-stratified permutation null for item-level information, then reanalyzes damaged-model evaluations across several benchmarks and model families.",
  "strongest_verified_contribution": "The strongest self-contained contribution is the clear separation of the arm-independent v1 question from the arm-conditional v2 question. The derivation that every legal pure constant emitter has Delta_perm = 0 is algebraically valid, and the paper correctly avoids claiming the underlying kappa numerator as a new statistic.",
  "strengths": [
    "The paper identifies a practically important evaluation failure: nominal chance can credit an input-blind constant predictor.",
    "The distinction between interface comparability and arm-specific item alignment is conceptually useful and clearly explained.",
    "The manuscript is unusually candid about post-hoc analysis, regime confounds, withdrawn claims, prior integrity defects, and the fact that passing a floor is necessary rather than sufficient.",
    "All 34 payload files listed in MANIFEST.json match their listed byte sizes and SHA-256 hashes.",
    "Several internal numerical checks succeed, including 36/12032 = 0.2992 percentage points, 2532/14042 = 18.03%, the 40.6-point tie-convention span, and the 32.5-point excess over the intact content score."
  ],
  "issues": [
    {
      "id": "R1",
      "severity": "major",
      "location": "Reproducibility Statement, manuscript/sections/10_reproducibility.tex:3-10; Evidence provenance, manuscript/sections/09_appendix.tex:96-126; evidence/claim_evidence_map.tsv:2-5; MANIFEST.json",
      "dimension": "reproducibility",
      "description": "The paper says every quantitative claim is bound to a machine-readable record and that readers can rerun emitters, but nearly all named records and all emitters are absent from the frozen snapshot. Missing load-bearing files include floor_winners_curse_calibration.json, s2_03_symmetric_inference.json, s2_02_stratified_ordering.json, the v2 permutation record, per-item predictions, power records, and the prose checker. The shipped claim map is an assertion ledger rather than underlying evidence, so the central empirical numbers cannot be independently traced or reproduced.",
      "proposed_fix": "Ship every artifact referenced by E-A through E-CAL, anonymized per-item records or a hash-addressed archive sufficient to regenerate them, all analysis emitters and checkers, and manifest entries for every file. Otherwise narrow the reproducibility statement to the material actually included.",
      "verification_test": "In a clean unpacked snapshot, every path in the artifact map and claim_evidence_map.tsv must resolve, and running the shipped code must reproduce every table and headline count from per-item records with a zero diff or a declared numerical tolerance."
    },
    {
      "id": "R2",
      "severity": "major",
      "location": "Abstract, manuscript/sections/00_abstract.tex:2; Introduction, manuscript/sections/01_introduction.tex:19; manuscript/sections/tab_nulls.tex:9-10 and 24-29; manuscript/sections/tab_construct_nulls.tex:13-14 and 24; compare manuscript/sections/03b_nulls.tex:28",
      "dimension": "technical_soundness, experimental_rigor",
      "description": "The MMLU-Pro winner's-curse calibration uses an iid ten-class multinomial null even though MMLU-Pro has three to ten legal options per item. Under construct-respecting uniform guessing, A, B, and C are legal on every item and each has expected marginal mean(1/n_opt) = 0.110877. Therefore the expected maximum must be at least 0.110877, contradicting the reported null mean 0.104460. The headline MMLU-Pro p < 1e-5 consequently calibrates against an infeasible global ten-way label distribution rather than the item-conditional chance structure discussed by the paper.",
      "proposed_fix": "Recompute the calibration by sampling each item's null label uniformly from that item's legal options while preserving the full option-count and legality vector. Revise the abstract and introduction if MMLU-Pro no longer rejects.",
      "verification_test": "Report the corrected null mean, 95th percentile, Monte Carlo exceedance count, correction convention, and p-value for the observed count 1403/12032. Confirm that every simulated answer is legal for its item."
    },
    {
      "id": "R3",
      "severity": "major",
      "location": "Results, manuscript/sections/05_analysis.tex:25; multiplicity discussion, manuscript/sections/09a_relocated.tex:26 and 34-37",
      "dimension": "experimental_rigor, limitations_responsible_claims",
      "description": "The reported binomial probability P(Binomial(12, 0.05) >= 3) = 0.0196 is arithmetically correct, but its interpretation is not justified. The manuscript itself states that the 12 tests share items, contain nested arms, and are neither independent nor exchangeable. The binomial calculation nevertheless assumes independent size-0.05 Bernoulli rejections, so the statements that the count survives correction and establishes that the chance side is not uniformly null do not follow.",
      "proposed_fix": "Use a synchronized global-null resampling procedure that preserves shared items and cross-arm dependence, recording the rejection count in each joint replicate. Alternatively, retain 3/12 only as a descriptive count and remove the binomial significance claim.",
      "verification_test": "Demonstrate calibrated family-level Type-I error under joint null simulations and report the empirical probability that at least three of the 12 dependent tests reject."
    },
    {
      "id": "R4",
      "severity": "major",
      "location": "evidence/build_record.json fields pdf_bytes, pdf_sha256, and pdf_pages; MANIFEST.json entry for manuscript/main.pdf",
      "dimension": "reproducibility",
      "description": "The build record authenticates a different PDF. It reports 355196 bytes, SHA-256 56a376e128c358e9f09f2daeee7f543a3717c5cab318e113d28c4102a56e2485, and 22 pages. The shipped PDF is 366583 bytes, has SHA-256 1fbaaf9983220fd83e10ff772fbea15ea0f6131e9f913a20b1d69911fd863acf, and contains 24 pages. The PDF matches MANIFEST.json, so the build record is stale and its build_gate_pass does not authenticate the submission.",
      "proposed_fix": "Generate the build record after the final PDF and freeze the sources, PDF, record, and manifest atomically.",
      "verification_test": "After unpacking, independently recompute the PDF byte size, SHA-256, and page count and require exact agreement among the PDF, build record, and manifest."
    },
    {
      "id": "R5",
      "severity": "major",
      "location": "Experimental Setup, manuscript/sections/04_experiments.tex:4-16; Reproducibility Statement, manuscript/sections/10_reproducibility.tex:12-18; manuscript/refs.bib",
      "dimension": "reproducibility, citation_integrity",
      "description": "The paper gives several useful settings but omits immutable model and tokenizer revisions, dataset versions and checksums, complete prompts, software versions, experimental hardware, checkpoint hashes, and full OLMo pruning/healing configurations and seeds. The central benchmarks and model checkpoints also lack canonical citations. Family names and arm labels are insufficient to reconstruct the evaluated objects or exact item inventory.",
      "proposed_fix": "Provide a machine-readable configuration for every reported cell containing canonical dataset and model citations, repository identifiers and revisions, split and preprocessing provenance, prompt bytes, tokenizer revision, damage transformation, checkpoint hash, software lockfile or container digest, seeds, and hardware.",
      "verification_test": "An independent group using only the recorded identifiers must recover the stated item counts and option strata and reproduce one intact, one truncate-only, and one prune-then-heal cell within a declared tolerance."
    },
    {
      "id": "R6",
      "severity": "major",
      "location": "manuscript/sections/03_method.tex:75; manuscript/sections/04_experiments.tex:15-16; manuscript/sections/05_analysis.tex:10-23; manuscript/sections/09a_relocated.tex:49-50",
      "dimension": "experimental_rigor, reproducibility",
      "description": "The resampling unit is identified as the item, but important inferential details remain unspecified: bootstrap interval type, null centering, exact two-sided mid-p formula, zero-atom treatment, whether the v1 maximizing constant is reselected within every resample, and the permutation exceedance and plus-one convention. Repairing an illegal p = 1.042 proves boundedness, not Type-I calibration.",
      "proposed_fix": "Add explicit formulas or executable pseudocode for every interval and p-value, including tie handling, reference recomputation, threshold comparisons, and Monte Carlo corrections.",
      "verification_test": "Run null simulations, including distributions with a large zero atom, and demonstrate approximately 95% interval coverage and 5% rejection for each implemented procedure."
    },
    {
      "id": "R7",
      "severity": "minor",
      "location": "manuscript/sections/03b_nulls.tex:12; manuscript/sections/tab_conventions.tex:9; manuscript/sections/tab_nulls.tex:15-16 and 26; manuscript/sections/tab_construct_nulls.tex:19-20 and 24; manuscript/sections/10_reproducibility.tex:3-7",
      "dimension": "technical_soundness, reproducibility",
      "description": "The claimed mechanical numeral guarantee is contradicted by several exact inconsistencies. First, 0.532164/0.125914 = 4.2264, not 4.6. Second, CommonsenseQA's shared gap is printed as +0.885 in one table and +0.884 in the generated table. Third, PIQA defines p = P(max >= observed floor) but reports 0.658; with floor 928/1838 the exact inclusive tail is 0.691725, while 0.657647 is the strict tail. These do not change the qualitative conclusions but show that the checker did not validate all derived expressions and thresholds.",
      "proposed_fix": "Correct the ratio, generate shared table fields from one full-precision source and rounding rule, compare Monte Carlo statistics using integer counts or full-precision values, and extend the checker to validate arithmetic expressions rather than merely matching component numerals.",
      "verification_test": "Unit tests must reject 4.6 for the stated ratio, require identical CommonsenseQA formatting across tables, and make the PIQA simulation agree with exact enumeration of the stated inclusive event."
    },
    {
      "id": "R8",
      "severity": "major",
      "location": "manuscript/sections/01_introduction.tex:3 and 7; manuscript/sections/02_related.tex:9 and 17; manuscript/sections/03_method.tex:50-58; manuscript/sections/07_limitations.tex:15-16; manuscript/refs.bib:33-45",
      "dimension": "novelty, citation_integrity",
      "description": "Claims such as the missing operational item, closest structural precedent, and only varying-k stratification is ours are broader than the documented citation audit supports. The closest contemporaneous Cho comparison was performed on arXiv v4 dated January 12, 2026, while the bibliography cites the ICLR 2026 poster and the manuscript states that the January 26 camera-ready was not compared. This is a support and versioning gap, not evidence that the novelty claim is false.",
      "proposed_fix": "Scope novelty statements to the explicitly audited corpus, document a reproducible literature search and claim-by-claim comparison, and cite the exact version inspected. Alternatively, inspect the camera-ready and update the overlap analysis.",
      "verification_test": "Every first, only, closest, missing, or not-aware statement must either be limited to a documented corpus or supported by a reproducible search; the cited Cho record must identify the exact audited version or a diff must establish that the final version does not alter the novelty boundary."
    }
  ],
  "score_ceiling_under_current_evidence": 6,
  "predicted_score_after_required_changes": 6,
  "evidence_that_would_raise_score": [
    "A complete immutable artifact that regenerates all tables and prose counts from shipped per-item records.",
    "A legality-aware MMLU-Pro winner's-curse calibration that still supports the stated conclusion.",
    "A dependence-preserving joint-null analysis showing that the 3/12 count remains unlikely.",
    "Fully specified and empirically calibrated bootstrap and permutation procedures.",
    "Immutable benchmark, model, tokenizer, checkpoint, software, and hardware provenance sufficient for independent reproduction.",
    "A scoped and version-specific prior-work comparison supporting the novelty boundary."
  ],
  "evidence_that_would_lower_score": [
    "The legality-aware MMLU-Pro calibration no longer rejects the balanced null.",
    "Joint-null resampling shows that three rejections are common under the dependence structure.",
    "The missing artifact, once supplied, fails to regenerate the reported tables or reveals additional threshold and rounding errors.",
    "Exact checkpoint or dataset reconstruction yields a different item inventory, option structure, or prediction record.",
    "The final version of the closest prior work already contains the claimed variable-option stratification or pre-comparison gate."
  ],
  "review_limitations": [
    "The named raw evidence records and analysis scripts are absent, so most empirical claims can neither be independently confirmed nor refuted from the snapshot.",
    "External lookup was prohibited; citation contents and the completeness of the novelty search could therefore be assessed only from local metadata and manuscript disclosures.",
    "Author checker scripts were not run, as required by the review instructions; all checks were read-only hash, PDF, table, and arithmetic inspections.",
    "The declared aggregate snapshot hash was accepted as the review identifier because its aggregation procedure is not specified, although every individually listed payload hash and size was verified."
  ]
}
```

## PROSE REVIEW

### Summary and recommendation

This paper proposes a useful evaluation rule: before interpreting multiple-choice arm differences, compare the reported construct against an explicit input-blind null rather than nominal chance. It separates two questions:

1. **V1:** Is the interface score above an arm-independent best-constant floor and therefore usable for comparing arms?
2. **V2:** Does one arm’s prediction vector align with items beyond what its own output marginal would produce?

The conceptual distinction is valuable, and the constant-emitter identity for the stratified v2 statistic is self-contained and correct. The manuscript is also unusually candid about post-hoc analysis, regime confounds, failed hypotheses, prior integrity defects, and the necessary-not-sufficient status of the floor test.

However, I recommend **reject (4/10)** in the current form. The main reason is not presentation: it is that the frozen artifact does not contain the evidence it repeatedly claims to publish, and two decision-relevant statistical arguments are invalid as written. Most headline empirical results are therefore untraceable, while the MMLU-Pro winner’s-curse calibration appears to use a null incompatible with the benchmark’s variable legal option sets.

### Strongest verified contribution

The strongest verified contribution is the separation between arm-independent interface comparability and arm-conditional item-level information. Equations in Section 3 establish that a pure constant emitter has `Delta_perm = 0`, independent of its collapse letter. The manuscript also correctly acknowledges that this is a standard property of the numerator of Cohen’s kappa and limits its novelty claim to stratification and use as a gate.

### Strengths

- The motivating measurement problem is important and easy to encounter in practice.
- The paper clearly explains why nominal chance can credit a literal constant predictor.
- V1 and v2 answer meaningfully different questions, and the paper warns against conflating them.
- The limitations and retraction ledger are more responsible than is typical.
- Every listed snapshot payload matches its manifest hash and size.
- Several arithmetic claims are internally correct, as detailed below.

### Major issues

**R1 — Major, reproducibility.**  
**Location:** Reproducibility Statement, `10_reproducibility.tex:3-10`; Evidence provenance, `09_appendix.tex:96-126`; `claim_evidence_map.tsv:2-5`; `MANIFEST.json`.

The paper says that every quantitative claim is bound to a machine-readable record and that readers can rerun emitters. The frozen snapshot, however, contains only `build_record.json` and `claim_evidence_map.tsv` under `evidence/`. The named calibration, symmetric-inference, stratified-ordering, permutation, power, prediction, emitter, and checker artifacts are absent. Several paths instead point to an absent `tcodex_out/EVIDENCE_PACK.md`.

This prevents an independent reviewer from tracing the headline numbers to per-item records. The claim map only repeats the conclusions; it does not contain the underlying observations.

**Fix:** Ship all E-A–E-CAL records, sufficient per-item data, and the exact emitters/checkers, each included in the manifest.  
**Verification:** A clean checkout must resolve every advertised path and regenerate every table and headline count from the shipped records.

---

**R2 — Major, technical soundness and experimental rigor.**  
**Location:** Abstract; Introduction line 19; Table 1; generated construct-null table; compare the variable-option discussion in `03b_nulls.tex:28`.

The MMLU-Pro winner’s-curse analysis samples an iid ten-class multinomial null. Yet MMLU-Pro has between three and ten legal options per item. Under item-conditional uniform guessing, A, B, and C are legal for every item and therefore each has expected marginal

`mean(1/n_opt) = 0.110877`.

The expected maximum cannot be below the expected marginal of A, but the paper reports an expected maximum of only `0.104460`. Thus the reported MMLU-Pro `p < 1e-5` is calibrated against global ten-way uniformity, not a null respecting the benchmark’s legal answer sets.

This directly affects a headline statement in the abstract. It does not invalidate the MMLU or BoolQ rows, but MMLU-Pro is the paper’s main high-power benchmark.

**Fix:** Generate each null answer uniformly from the options legal for that item and then recompute the maximum empirical letter marginal.  
**Verification:** Report the corrected mean, 95th percentile, exceedance count, Monte Carlo convention, and p-value for the observed `1403/12032` floor.

---

**R3 — Major, experimental rigor and responsible claims.**  
**Location:** `05_analysis.tex:25`; `09a_relocated.tex:26,37`.

The calculation

`P(Binomial(12, 0.05) >= 3) = 0.0195683`

is arithmetically correct. Its use is not. The manuscript explicitly says the tests share items, involve nested arms, and are neither independent nor exchangeable. The binomial model nevertheless assumes independent size-0.05 rejection indicators.

Consequently, the statement that “what survives correction is the count itself” is unsupported. The count `3/12` can still be reported descriptively, but the binomial global-null interpretation cannot be used without modeling the dependence.

**Fix:** Use synchronized joint-null resampling over items and arms, or remove the global-null p-value.  
**Verification:** Under the joint null, measure the empirical probability of at least three rejections and demonstrate calibrated family-level error.

---

**R4 — Major, reproducibility.**  
**Location:** `evidence/build_record.json`; `MANIFEST.json` entry for `main.pdf`.

The build record describes a different PDF:

- Build record: 355,196 bytes, 22 pages, SHA-256 `56a376...2485`.
- Shipped PDF: 366,583 bytes, 24 pages, SHA-256 `1fbaaf...3acf`.

The shipped PDF does match the manifest. Therefore the manifest authenticates the actual payload, but `build_gate_pass: true` applies to a stale build and does not authenticate the submitted PDF.

**Fix:** Generate the build record after the final PDF and freeze all build outputs atomically.  
**Verification:** Page count, byte size, and SHA-256 must agree across the PDF, manifest, and build record.

---

**R5 — Major, reproducibility and citation integrity.**  
**Location:** Experimental Setup, `04_experiments.tex:4-16`; Reproducibility Statement, `10_reproducibility.tex:12-18`; bibliography.

The paper gives batch size, precision, sequence length, and several prompt switches, but omits exact model/tokenizer revisions, dataset versions, split checksums, complete prompt serialization, experimental software versions, hardware, checkpoint hashes, and full OLMo pruning/healing configurations and seeds. The benchmarks and model checkpoints also lack canonical citations.

These omissions are load-bearing because the conclusions depend on exact item inventories, variable option counts, tokenizer-dependent lengths, model layer counts, and particular healed checkpoints.

**Fix:** Supply immutable model, tokenizer, dataset, checkpoint, environment, prompt, and transformation identifiers per cell.  
**Verification:** An independent group should be able to reconstruct one intact, one truncate-only, and one prune-then-heal cell without making undocumented choices.

---

**R6 — Major, experimental rigor and reproducibility.**  
**Location:** `03_method.tex:75`; `04_experiments.tex:15-16`; `05_analysis.tex:10-23`; `09a_relocated.tex:49-50`.

The paper states that bootstrap resampling occurs over items and that some references are recomputed per resample. It does not define the interval construction, null centering, exact mid-p formula, treatment of zero atoms and ties, whether v1 reselects the maximizing constant within each resample, or the permutation plus-one convention.

The prior `p = 1.042` defect makes these details particularly important. Splitting the zero atom so that p-values remain bounded does not establish calibrated Type-I error.

**Fix:** Provide formulas or executable pseudocode for every interval and p-value.  
**Verification:** Null simulations, including statistics with substantial probability at zero, should show approximately 95% coverage and 5% rejection.

---

**R8 — Major, novelty and citation integrity.**  
**Location:** Introduction lines 3 and 7; Related Work lines 9 and 17; Method lines 50-58; Limitations lines 15-16.

Statements such as “the missing operational item,” “the closest structural precedent,” and “only the varying-k stratification is ours” exceed what the documented citation audit establishes. In particular, the comparison with Cho et al. was performed against arXiv v4 dated **January 12, 2026**, while the bibliography cites the ICLR 2026 poster and the paper states that the **January 26, 2026** camera-ready was not compared.

I do not conclude that the novelty claim is false. Rather, the current sources do not establish it at the breadth claimed.

**Fix:** Scope these statements to a documented corpus, cite the exact versions inspected, and provide a claim-level comparison with the nearest methods.  
**Verification:** Every “first,” “only,” “closest,” or “missing” claim should either be explicitly scoped or backed by a reproducible literature audit.

### Numerical and provenance spot checks

Because the snapshot omits the named raw evidence records, the following are checks against the shipped claim ledger, manuscript tables, and arithmetic—not independent reproduction from per-item evidence.

**Matches:**

1. `claim_evidence_map.tsv` H-02 says the symmetric comparison is **3/12 versus 1/12**, matching `05_analysis.tex:17-18`.
2. H-04 reports a stratification slack of **0.2992 pp**; independently,  
   `100 × 36 / 12032 = 0.299202 pp`.
3. C-13 reports **14/15** at or below the floor, consistent with the single above-floor designated exception in the MMLU-Pro table.
4. The tie-convention endpoints `0.125914` and `0.532164` imply  
   `100 × (0.532164 − 0.125914) = 40.625 pp`, matching **40.6 pp**.
5. `0.532164 − 0.207613 = 0.324551`, matching the claimed **32.5 pp** excess.
6. `2532 / 14042 = 18.0316%`, matching **18.03%**.

**Mismatches:**

1. `03b_nulls.tex:12` says `0.532164` is **4.6×** `0.125914`. The actual ratio is  
   `0.532164 / 0.125914 = 4.2264085`, i.e. **4.23×**.
2. CommonsenseQA’s gap is printed as **+0.885** in `tab_nulls.tex` and **+0.884** in the supposedly synchronized generated table.
3. PIQA defines its calibration as `P(max >= observed floor)` and reports `p = 0.658`. For floor `928/1838`, the exact inclusive two-sided tail is **0.691725**. The reported value matches the strict tail, **0.657647**, suggesting comparison against a rounded decimal or use of `>` rather than the stated `>=`.

These errors do not reverse the affected qualitative verdicts, but they refute the Reproducibility Statement’s claim that the mechanical checker has resolved every derived numeral.

### Score rationale and ceiling

My dimension scores are:

- Novelty: **3/5**
- Significance: **4/5**
- Technical soundness: **2/5**
- Experimental rigor: **2/5**
- Clarity: **4/5**
- Reproducibility: **1/5**
- Citation integrity: **2/5**
- Limitations and responsible claims: **4/5**

The paper’s conceptual contribution is promising, but the current submission is not decision-ready. The most severe issue is the absence of the evidence needed to audit the empirical claims. The claim with the weakest evidence-to-importance ratio is the MMLU-Pro winner’s-curse conclusion, because it appears in the abstract yet uses a null that does not preserve item-level answer legality.

The **score ceiling without new model experiments is 6/10**. The authors may be able to reach that ceiling by releasing existing records, correcting and rerunning the statistical analyses on existing data, fully specifying provenance, and narrowing any claims that do not survive. A new model evaluation is not necessarily required, but the present manuscript cannot receive credit for evidence that is described but not shipped.