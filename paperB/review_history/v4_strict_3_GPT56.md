```yaml
review_mode: strict
soundness: 2.5
excitement: 2.5
overall: 2.5
confidence: 4.5
reproducibility: 2.0
```

# Paper Summary

This paper presents an observational case study of continued pretraining after depth pruning of OLMo-2-7B. Its principal construction keeps the first 14 pretrained blocks, appends two freshly initialized blocks, and trains the resulting 16-layer model for 200k optimizer steps. It measures in-domain held-out perplexity, answer-letter and complete-option MMLU, three no-retrieval closed-book QA tasks, and a broader zero-shot likelihood suite. Auxiliary observations include an intact 32-layer branch available only through 25k steps, frozen-prefix and fully random 16-layer operating points, a non-contiguous ShortGPT-16 construction, shallower prefix arms stopped at unequal checkpoints, a qualitative OLMo-2-1B trajectory, and one unmatched Qwen3-8B endpoint.

The central measurement is real and clearly bounded: keep14 improves to PPL 10.561 at 200k but remains substantially below the intact base (PPL 7.398; MMLU .319 versus .605). Complete-option MMLU raises keep14 to .383, yet random initialization reaches .360 under that interface while staying at chance on answer letters, indicating a large fluency/interface floor. ShortGPT-16 reaches PPL 9.780 and answer-letter MMLU .474, showing that nominal 16-layer depth does not specify a unique endpoint. The manuscript repeatedly and correctly disclaims causal localization, factor isolation, seed stability, and universal recovery laws.

My assessment is that the paper is unusually candid and well documented for a case study, but the evidence is still too under-controlled and unreplicated to make the proposed reporting prescription an ACL-main contribution. At present I place it below Findings: most comparisons are single runs; the crucial 25k intact control cannot support long-horizon interpretation; ShortGPT differs in four coupled construction dimensions; random/frozen controls are not treatment-matched; uncertainty is predominantly over evaluation items rather than training realizations; and all PPL evidence is in-domain. These are not presentation defects—the paper itself acknowledges them—but they sharply limit what is learned beyond “different, confounded operating points produce different observed metrics.”

# Claims and Evidence Map

## C1. Likelihood recovery and target-capability recovery remain separated in the principal observed run

- **Claim anchor:** Abstract; §5.2; Table 2; Figure 1; Conclusion.
- **Exact evidence:** keep14 at 200k has PPL 10.561, answer-letter MMLU .3191, content MMLU .3832, PopQA .1415, TriviaQA .2940, and NQ-open .0598, versus base 7.398/.6053/.4706/.2571/.6355/.2050.
- **Minimum sufficient experiment:** one correctly implemented trajectory with fixed validation and target evaluations can establish this strictly descriptive, run-conditional fact.
- **Assessment:** **Supported only as a measurement claim for this realized checkpoint path.** It does not establish a typical pruning effect, a seed-stable dissociation, or eventual non-recovery.

## C2. The available full32 branch argues against short-horizon corpus shift

- **Claim anchor:** Abstract; §5.2 and §5.3; Table 2.
- **Exact evidence:** full32@25k is near the base on PPL and downstream metrics (e.g., PPL 7.670 versus 7.398; MMLU .588 versus .605).
- **Minimum sufficient experiment:** an intact branch on the identical token stream and schedule through the same horizon as the claim. For the paper's explicitly short-horizon claim, 25k is sufficient; for interpreting the 200k keep14 endpoint, full32 must run through 200k with matched data order/resume behavior.
- **Assessment:** **Supported only through 25k.** It is not a 200k counterfactual, and the manuscript generally says so.

## C3. MMLU scoring interface materially changes measured recovery, but complete-option scoring is not a clean knowledge readout

- **Claim anchor:** §3.4, §5.2, §5.3; Tables 2 and 16.
- **Exact evidence:** keep14 rises from .3184 letter accuracy to .3832 normalized-content accuracy; random rises from .2470 to .3598; letter/content protocols change prompt, candidate, tokenization, and normalization together.
- **Minimum sufficient experiment:** paired per-item predictions under controlled transformations that separately manipulate answer-symbol mapping, prompt, candidate string, and normalization, including random/frozen controls.
- **Assessment:** **Supported as interface sensitivity and as evidence against treating content accuracy as recovered knowledge.** The evidence does not isolate a readout or symbol-mapping mechanism.

## C4. The residual gap is not only an answer-letter artifact

- **Claim anchor:** §5.2; Table 2; Table 17.
- **Exact evidence:** keep14 remains below base on PopQA, TriviaQA, and NQ-open under one shared generation protocol.
- **Minimum sufficient experiment:** independent non-letter tasks under a fixed protocol; ideally paired uncertainty and multiple decoding/prompt normalizations.
- **Assessment:** **Directionally supported for the evaluated protocol.** It does not quantify seed uncertainty or rule out all interface/normalization effects.

## C5. Nominal depth alone does not determine a unique observed endpoint

- **Claim anchor:** §6.3; Table 3; Discussion.
- **Exact evidence:** two 16-layer, 200k constructions end at PPL/MMLU 10.561/.3191 (keep14) and 9.780/.4739 (ShortGPT).
- **Minimum sufficient experiment:** two correctly reconstructed same-depth models are sufficient to falsify a literal “all 16-layer constructions have one endpoint” statement.
- **Assessment:** **Supported as an existential/descriptive measurement claim.** No causal conclusion about layer choice, final-layer retention, inherited count, or fresh tails follows.

## C6. The same-shape inherited, frozen-front, and random operating points have discordant PPL/MMLU ordering

- **Claim anchor:** §5.2–5.3; Figure 2; Tables 2 and 15.
- **Exact evidence:** random has better PPL than frozen (11.498 versus 12.797) but lower answer-letter MMLU (.247 versus .262); keep14 is stronger than both.
- **Minimum sufficient experiment:** matched shape plus reliable evaluation establishes the ordering of these observed points; causal attribution requires matched learning rate, trainable modules, initialization scope, seeds, and token order.
- **Assessment:** **Supported descriptively; not a clean initialization/adaptation ablation.**

## C7. Late keep14 healing continues but does not produce broad catch-up over 128k–200k

- **Claim anchor:** Figure 1; §6.2; Table 13.
- **Exact evidence:** PPL 10.826→10.561 and headline MMLU .3012→.3191; common rerun estimates +1.68 points with item-paired CI [1.08, 2.29], while most other task aggregates move little or with mixed signs after 153.5k.
- **Minimum sufficient experiment:** a predeclared checkpoint series with the same evaluator and paired items supports the realized-trajectory statement; replicated runs are required for a characteristic training-dynamics claim.
- **Assessment:** **Supported conditionally on this run.** The item CI does not establish run-to-run stability, and continued eventual recovery remains unresolved.

## C8. Broad MMLU recovery differs across domains

- **Claim anchor:** §6.1; Figure 3; Table 18.
- **Exact evidence:** keep14 chance-adjusted recovery ranges from roughly 15.6% in STEM to 29.1% in Other; ShortGPT is higher in each group.
- **Minimum sufficient experiment:** subject-stratified item-level analysis with uncertainty and a formal interaction/heterogeneity test if inferential language is used.
- **Assessment:** **Descriptive heterogeneity is supported.** The paper correctly avoids knowledge-type localization, but no formal domain-difference inference is provided.

## C9. A multi-axis reporting bundle is preferable to reporting PPL alone

- **Claim anchor:** Contributions; §7; Conclusion.
- **Exact evidence:** the study gives several examples where interface, construction, budget, or target metric changes interpretation.
- **Minimum sufficient experiment:** validation across multiple pruning/recovery methods, model families, corpora, and replicated runs, or a systematic survey showing that the bundle prevents erroneous conclusions.
- **Assessment:** **Plausible recommendation, not validated as a general protocol.** The manuscript admits this, which reduces overclaiming but also limits contribution strength.

## C10. Layer-wise appendix readouts do not localize stored knowledge or explain keep14/ShortGPT

- **Claim anchor:** Appendix C; Figure 6; Tables 20–21.
- **Exact evidence:** intact-model logit/tuned/linear-probe thresholds are measured with different probes and definitions; no causal intervention connects them to pruning outcomes.
- **Minimum sufficient experiment for localization:** matched causal interventions or structure-isolating ablations on the pruned models, with replicated effects.
- **Assessment:** **The non-causal disclaimer is correct.** These plots are background only and must not be read as evidence that removed upper layers “contain” missing knowledge.

# Strengths

## S1. The scope discipline is excellent

- **Anchor:** Abstract lines 19–22; §3.3; §4; §6.3; Limitations; Appendix C.
- The paper repeatedly distinguishes observed operating points from causal ablations, item uncertainty from seed uncertainty, short-horizon from long-horizon controls, and readout thresholds from knowledge localization. This directly addresses the most dangerous interpretation errors in this topic.

## S2. The paper exposes rather than hides important confounds

- **Anchor:** §3.2–3.4; Tables 2–3, 15–16; Limitations.
- It explicitly reports the 25k/200k mismatch, metric-based unequal stopping, different random-init learning rate, frozen trainable set, ShortGPT's inherited-count/final-layer/fresh-tail differences, and the fact that headline and common-rerun MMLU aggregates differ slightly.

## S3. Evaluation is substantially broader than a PPL-only pruning report

- **Anchor:** §3.4; Tables 2, 10, 16–19.
- The combination of answer-letter MMLU, complete-option MMLU, three closed-book generation tasks, and nine additional likelihood-style tasks makes the central measurement more informative than a single aggregate benchmark.

## S4. Reproducibility reporting is candid and technically detailed

- **Anchor:** Appendix B.1–B.2, PDF pp. 13–15.
- The paper gives batch size, sequence length, LR schedules, AdamW parameters, precision, gradient clipping, reconstruction equality checks, parameter count, validation size, PPL merge formula, prompts, decoding, normalization, sample counts, and checkpoint provenance. It also discloses unset training seeds and the resumed-loader defect rather than implying exact reproducibility.

## S5. Item-level comparisons are handled more carefully than simple marginal bars

- **Anchor:** Table 15 and Table 13.
- Exact McNemar tests and paired bootstrap intervals on common MMLU items are appropriate for conditional checkpoint comparisons. The manuscript correctly says these do not measure training-seed variation.

## S6. The rendered paper is readable and all figures/tables are interpretable

- **Anchor:** PDF pp. 3, 5, 7, 12–18.
- I inspected all 6 figures and 22 tables. Captions usually state the comparison limits, axes and legends are legible, and missing cells/checkpoint mismatches are visibly disclosed.

# Weaknesses

## W1. The central training conclusions have no training-seed replication

- **Location:** Limitations, PDF p. 8, lines 582–586; Appendix B.1, PDF p. 14, lines 1020–1023.
- **Quote (9 words):** “keep14, ShortGPT, and the same-shape points are single runs”
- **Problem:** Every central endpoint and trajectory is one realization, with no explicit training seed. Item-level McNemar/bootstrap intervals condition on fixed checkpoints and cannot estimate variation due to fresh initialization, shuffle, optimizer noise, or selected blocks.
- **Affected claim/norm:** C1, C5–C9 and the empirical-contribution norm for an ACL paper. A one-run case can document what happened, but it cannot support language suggesting a characteristic “recovery path,” stable construction ordering, or a generally useful reporting prescription.
- **Why it matters:** The observed MMLU differences, especially smaller trajectory/domain effects, may be materially seed-dependent. Even the large ShortGPT gap has unknown run-level variance.
- **Sufficient remedy:** Run at least 3 independent seeds for keep14, ShortGPT-16, frozen-front, and random-init under fixed token exposure and report mean/SD or hierarchical intervals over seeds. Replicate at least the principal 128k/153.5k/200k trajectory evaluations. If compute forbids this, narrow the paper to a data/report and remove inferential/general recommendation language.
- **Severity:** **Major**
- **Mechanical verification:** Exact quoted text appears in `sections/06_limitations.tex`; source also states “Runs do not set an explicit random seed” in `sections/08_appendix.tex`.

## W2. The 25k intact branch does not control the principal 200k recovery horizon

- **Location:** Limitations, PDF p. 8, lines 587–591; §5.3, PDF p. 6, lines 385–394.
- **Quote (4 words):** “full32 stops at 25k”
- **Problem:** The control can reject only an early catastrophic corpus/schedule drift. It cannot determine how 200k of continued pretraining on the same in-domain array changes intact-model PPL, MMLU, or closed-book QA, particularly given nominal rather than exact unique-token accounting after resume.
- **Affected claim/norm:** C2 and interpretation of C1. Any comparison of the keep14 200k endpoint to the untouched base mixes pruning/reconstruction effects with 200k continuation effects.
- **Why it matters:** The paper's strongest explanatory control ends after one eighth of the principal optimization horizon. The authors avoid calling it exact, but the absence still leaves a central alternative unresolved.
- **Sufficient remedy:** Continue full32 through 200k with the same optimizer-step schedule, token stream/data order, resume semantics, and evaluations at the keep14 checkpoints. Report both delta-from-base and keep14-minus-full32 trajectories with seed replication.
- **Severity:** **Major**
- **Mechanical verification:** Exact quote appears in `sections/06_limitations.tex`; Table 2 and Table 4 mark all full32 results as 25k only.

## W3. The ShortGPT comparison cannot explain the large construction gap

- **Location:** Limitations, PDF p. 8, lines 592–596; §6.3, PDF p. 6, lines 446–465.
- **Quote (11 words):** “ShortGPT changes inherited count, selected layers, final-block retention, and fresh-tail use.”
- **Problem:** ShortGPT inherits 16 pretrained blocks including block 31, whereas keep14 inherits 14 contiguous prefix blocks and appends two fresh blocks. Four factors change together. The result only establishes two different operating points, not what makes one recover better.
- **Affected claim/norm:** C5 and novelty/value. The existential statement “nominal depth alone is insufficient” is valid but weak; any stronger practical implication about selection, preserving late layers, or avoiding fresh blocks is unsupported.
- **Why it matters:** The .155 MMLU gap is the most striking result, but the design cannot turn it into actionable or mechanistic knowledge.
- **Sufficient remedy:** Add a factorial minimum set at matched 200k budget and seeds: (i) keep14+fresh2; (ii) same 14 prefix blocks plus two inherited late blocks; (iii) ShortGPT-selected 14+fresh2; (iv) 16 contiguous inherited blocks/no fresh tail; and, ideally, a variant toggling final-block retention while holding inherited count fixed.
- **Severity:** **Major**
- **Mechanical verification:** Exact quote appears in `sections/06_limitations.tex`; coupled dimensions are enumerated independently in `sections/05_analysis.tex`.

## W4. The random/frozen comparisons are confounded and therefore weak as explanations

- **Location:** §3.3, PDF pp. 3–4, lines 257–266; Limitations, PDF p. 8, lines 592–596.
- **Quote (11 words):** “Random-init uses a higher learning rate, frozen-front changes the trainable set”
- **Problem:** Random-init changes all lexical and decoder initialization and uses \(10^{-4}\) rather than \(2\times10^{-5}\); frozen-front changes which parameters can adapt. Shape and step count alone do not isolate inherited initialization or adaptation.
- **Affected claim/norm:** C3 and C6; baseline validity. The random content score is useful as an observed floor, and the ranking reversal is descriptive, but neither control estimates a treatment effect.
- **Why it matters:** Much of the paper's “control bundle” value rests on comparisons that are deliberately not matched. This limits the ability to adjudicate competing explanations rather than merely list them.
- **Sufficient remedy:** For random initialization, match optimizer/LR schedule and separately randomize decoder blocks versus embeddings/head. For frozen-front, add parameter-count/update-budget-matched alternatives (e.g., train an equally sized module set in keep14) and multiple seeds. Report token- and FLOP-matched results.
- **Severity:** **Major**
- **Mechanical verification:** Exact quote appears in `sections/06_limitations.tex`; the higher random LR is specified in Appendix B.1.

## W5. Checkpoint selection and compute are unequal across the depth ladder

- **Location:** §3.2, PDF p. 3, lines 242–250; Table 3; Limitations.
- **Quote (7 words):** “there was no registered common stopping rule”
- **Problem:** keep8/10/12 stop at 121k/83.5k/124k after knowledge-sensitive metrics “appeared stable,” while PPL was still decreasing. This is post hoc metric-informed selection, not a common-budget depth experiment.
- **Affected claim/norm:** C7–C9 and any depth-related reading of Figures 2–4/Table 3. The paper says the ladder is descriptive, but visual ordering and terms such as “depth sensitive” invite comparison.
- **Why it matters:** Unequal exposure, model FLOPs, and stopping criteria make depth, time, and selection inseparable. A shallow arm may simply be undertrained relative to its recovery timescale.
- **Sufficient remedy:** Evaluate all depths at a prespecified common grid through 200k, and additionally compare at matched tokens and recovery FLOPs. Predefine the primary endpoint and stopping rule; report all seeds and failed runs.
- **Severity:** **Major**
- **Mechanical verification:** Exact quote appears in `sections/03_method.tex` and `sections/06_limitations.tex`; literal unequal steps are listed in Tables 3–4.

## W6. The PPL evidence is entirely in-domain and cannot support a broad language-model recovery interpretation

- **Location:** Limitations, PDF p. 9, lines 603–606.
- **Quote (8 words):** “no contamination audit or out-of-domain PPL is reported”
- **Problem:** Training and validation are both from the DCLM portion of Dolmino, with only shard disjointness reported. Falling PPL may measure adaptation/memorization to that mixture rather than broad distributional recovery; no contamination analysis connects the training mixture to MMLU/QA.
- **Affected claim/norm:** C1 and C9, especially phrases such as “aggregate language-model quality,” “distributional modeling,” and the recommendation to report likelihood as a recovery axis.
- **Why it matters:** The paper correctly calls PPL in-domain in the conclusion/limitations, but several broader formulations can still be read as general LM recovery. This also complicates interpretation of PPL–capability separation.
- **Sufficient remedy:** Add multiple out-of-domain likelihood corpora with no overlap to the continuation mixture, document deduplication/contamination checks for target benchmarks, and distinguish in-domain adaptation from general LM quality throughout.
- **Severity:** **Major**
- **Mechanical verification:** Exact quote appears in `sections/06_limitations.tex`; the training and validation source is specified in `sections/03_method.tex` and Appendix B.1.

## W7. Reproduction of the principal run is not currently possible

- **Location:** Limitations, PDF p. 9, lines 606–610; Appendix B.1, PDF p. 14, lines 1020–1034.
- **Quote (21 words):** “Exact reproduction is limited by unset training seeds, an unrecorded resumed data-loader offset, incomplete compute accounting, and no frozen runnable artifact.”
- **Problem:** Fresh initialization and initial shuffle are unspecified; the resumed epoch's loader offset was not stored; token counts are nominal; exact code/environment/checkpoint artifacts are absent; project-wide compute is unknown.
- **Affected claim/norm:** Reproducibility and auditability of all empirical claims.
- **Why it matters:** Detailed prose is valuable but cannot reconstruct the actual 200k path or verify that comparisons saw equivalent examples. The resumed data behavior may itself alter the effective curriculum.
- **Sufficient remedy:** Release frozen code/config/environment, checkpoint hashes, exact selected ShortGPT layers and selection inputs, per-item predictions, RNG states, data indices/loader offsets, and complete per-run GPU-hours/FLOPs. Re-run the principal experiment with explicit seeds and recorded data order if exact recovery of the old run is impossible.
- **Severity:** **Major**
- **Mechanical verification:** The full exact quote appears in `sections/06_limitations.tex`; the resume behavior is detailed in `sections/08_appendix.tex`.

## W8. The proposed reporting protocol is not empirically validated across methods or families

- **Location:** Discussion, PDF p. 8, lines 557–562.
- **Quote (13 words):** “This proposal does not replace efficiency or endpoint-quality comparisons and is not validated”
- **Problem:** The reporting bundle is motivated by one principal OLMo recipe. The 1B result is same-family and more compressed; the Qwen result changes family, corpus, depth fraction, and available evaluations; neither is a replication of the full diagnostic.
- **Affected claim/norm:** C9 and excitement/novelty. A generally useful reporting recommendation should be shown to alter conclusions across multiple independent settings, not only asserted from one case.
- **Why it matters:** Without broader validation, the recommendation is sensible best practice rather than a demonstrated research contribution.
- **Sufficient remedy:** Apply the same predeclared bundle to at least two additional pruning/recovery methods and a second family with matched depth fraction, budgets, seeds, content/letter MMLU, closed-book QA, and in-/out-of-domain PPL. Show concrete cases where single-axis reporting would reverse or materially distort a conclusion.
- **Severity:** **Major**
- **Mechanical verification:** The quoted clause appears in `sections/04b_discussion.tex`; the limited 1B/Qwen scope is stated in `sections/05_analysis.tex`.

## W9. Closed-book evidence is missing for the strongest compressed comparator

- **Location:** Table 2 caption, PDF p. 5.
- **Quote (7 words):** “missing ShortGPT closed-book cells were not evaluated”
- **Problem:** The paper uses PopQA/TriviaQA/NQ-open to argue that keep14's deficit is not answer-letter-only, but the strongest half-depth construction has no corresponding generation results.
- **Affected claim/norm:** C4–C5 and completeness of the construction comparison.
- **Why it matters:** It remains unknown whether ShortGPT's large MMLU advantage transfers to factual generation or is partly specific to likelihood interfaces.
- **Sufficient remedy:** Evaluate ShortGPT and all same-shape controls on the identical closed-book prompts, aliases, decoding, and normalization; provide item-paired uncertainty and seed-level uncertainty.
- **Severity:** **Minor**
- **Mechanical verification:** Exact quote appears in `sections/tab_main_results.tex`.

## W10. The interface comparison changes several variables simultaneously and lacks paired uncertainty

- **Location:** §3.4, PDF p. 4, lines 273–282; Table 16.
- **Quote (13 words):** “Letter and content scoring change the prompt, candidate string, tokenization, and normalization together.”
- **Problem:** The comparison cannot identify whether gains arise from avoiding letter mapping, a different prompt, option-text priors, token length, or normalization. No paired uncertainty accompanies the nine-row interface comparison.
- **Affected claim/norm:** C3. The paper appropriately says “consistent with” rather than causal, but phrases such as “answer-symbol component” remain only hypotheses.
- **Why it matters:** This is a central diagnostic and one of the claimed contributions; its current design primarily establishes protocol sensitivity.
- **Sufficient remedy:** Add controlled variants changing one factor at a time, include option-order randomization and calibrated scoring, and report paired per-item bootstrap/McNemar-style contrasts with correction for the planned family of comparisons.
- **Severity:** **Minor**
- **Mechanical verification:** Exact quote appears in `sections/03_method.tex`; Table 16 explicitly states that no paired uncertainty analysis is provided.

## W11. Domain and late-trajectory inferences rely on item uncertainty, not training uncertainty, with limited multiplicity treatment

- **Location:** §6.1–6.2; Tables 12–13, 18, and 22.
- **Quote (8 words):** “These intervals are conditional on the realized checkpoints”
- **Problem:** Wald/item-bootstrap intervals treat the trained model as fixed. The paper examines many tasks, checkpoints, groups, and 57 subjects without a predeclared inferential family or multiplicity adjustment.
- **Affected claim/norm:** C7–C8 and statistical reliability. Descriptive reporting is acceptable, but terms such as “detectable,” “heterogeneous,” and subject examples should not be interpreted as replicated effects.
- **Why it matters:** Item-level precision can be extremely high at \(n=14{,}042\) while uncertainty over training is completely unknown, creating false confidence.
- **Sufficient remedy:** With replicated seeds, use a hierarchical analysis over seeds and items; predeclare primary contrasts; report corrected or explicitly exploratory secondary analyses. Add uncertainty to Figure 3 group differences and avoid cherry-picking subject examples without a selection rule.
- **Severity:** **Minor**
- **Mechanical verification:** Exact quote appears in `sections/05_analysis.tex`; Tables 12 and 15 identify their intervals as marginal/item-paired.

## W12. The supplementary readout section is visually suggestive despite being non-causal and non-comparable

- **Location:** Appendix C, PDF pp. 15 and 18; Figure 6; Tables 20–21.
- **Quote (8 words):** “These are output-head readouts, not causal storage localizations.”
- **Problem:** Figure 6 overlays semantic-probe, MMLU-logit-lens, and next-token thresholds derived from different datasets, probes, targets, and definitions, alongside the pruning-cut band. This invites anatomical interpretation that the text explicitly rejects.
- **Affected claim/norm:** Claim-scope discipline concerning knowledge localization.
- **Why it matters:** The section adds no evidence for the main recovery claims and risks encouraging exactly the causal/localization expansion the paper warns against.
- **Sufficient remedy:** Remove the section from the paper, or place it in clearly separated supplementary material without the pruning-cut overlay. If retained as analysis, run matched causal interventions on the actual pruned models before relating thresholds to recovery.
- **Severity:** **Minor**
- **Mechanical verification:** Exact quote appears in `sections/app_tab_logitlens_full.tex`; Appendix C states that the section is not evidence for the recovery-path claims.

# Questions That Could Change the Score

1. **Are there unreported independent seeds** for keep14, ShortGPT, frozen-front, random-init, or full32? If yes, reporting them with run-level variance could materially improve soundness.
2. **Can full32 be extended to 200k** on the same data order/schedule and evaluated at 128k, 153.5k, and 200k? This is the most important missing control.
3. **Can the ShortGPT gap be decomposed** with even a small matched construction matrix that independently toggles inherited count, non-contiguity, final block, and fresh tail?
4. **Can out-of-domain PPL and contamination checks be added?** This would determine whether the likelihood axis reflects general recovery or only in-mixture adaptation.
5. **Were keep8/10/12 stopping decisions made before seeing the reported target metrics?** If not, please state the exact decision history and avoid any inferential depth comparison.
6. **Can ShortGPT be evaluated on PopQA/TriviaQA/NQ-open** and the interface comparison be rerun with one-factor-at-a-time protocol changes?
7. **What are the exact per-run GPU-hours/FLOPs and hardware mappings?** H20 versus B200 and depth-dependent FLOPs make step counts an incomplete compute comparison.

# Non-Scoring Suggestions and Typos

1. In §3.2, replace “selected after knowledge-sensitive metrics appeared stable” with an exact, auditable checkpoint-retention rule or a factual decision log.
2. Avoid “fluency floor” unless operationally defined; “high random-init option-text baseline” is more literal.
3. Define `core6` directly in Table 3 rather than only as an average whose exact constituent tasks must be inferred from Table 10.
4. State whether ShortGPT's 128-window layer-selection set overlaps the fixed validation shard or any evaluation text.
5. Clarify whether the 4096×2048 validation windows contribute 8,388,608 tokens or 8,384,512 **target** tokens because of one-token shifts; Table 4 gives the latter, which is consistent with 4096×2047 but deserves one sentence.
6. Table 19 says all listed tasks use no free-form generation, while PopQA/TriviaQA/NQ-open are documented elsewhere; make the table's scope (“broad likelihood suite only”) explicit.
7. Report exact software versions, evaluator/harness commit, tokenizer hash, checkpoint hashes, and licenses in one compact reproducibility table.
8. Consider omitting Appendix C. It is not used by the main claims and increases the chance of causal overreading.

# Numerical and Internal-Consistency Audit

I checked the abstract's quantitative statements against the rendered tables/source:

1. **10.561 keep14 PPL at 200k:** matches Tables 2, 3, 4, 13 and Figure 1.
2. **\(1.428\times\) base PPL:** \(10.561/7.398 \approx 1.428\); matches Table 4.
3. **Base PPL 7.398:** matches Tables 2–4.
4. **Answer-letter MMLU .319 versus .605:** matches rounded Table 2; detailed values are .3191 and .6053.
5. **Full32 available through 25k:** matches Tables 2 and 4 and the method text.
6. **keep14 complete-option MMLU .383:** matches normalized-content .3832 in Table 16.
7. **Random content score nearly the same as frozen-front:** .3598 versus .3604 in Table 16; both round to .360 in Table 2.
8. **ShortGPT MMLU .474 at 200k:** matches .4739/.4742 in Tables 3, 10, and 16.
9. **Closed-book keep14/base values:** .142/.257 PopQA, .294/.636 TriviaQA, .060/.205 NQ-open match Tables 2 and 17 after rounding.
10. **Late keep14 change:** headline .3012→.3191 is +1.79 pp; the common rerun's +1.68 pp is separately and correctly disclosed.

Formula/boundary checks:

- \(d_{\text{cut}}=k/32\) and \(d_{\text{model}}=(k+2)/32\) are consistent for prefix+fresh2 arms.
- Chance-adjusted recovery \(100(x-c)/(b-c)\) reproduces the reported keep14 MMLU value: \(100(.3191-.25)/(.6053-.25)\approx19.4\%\).
- PPL merging by exponentiating global token-average NLL is correct; the paper explicitly avoids averaging shard PPLs.
- MMLU Wald intervals are numerically plausible for \(n=14{,}042\), but they are item-level, fixed-model intervals.
- The use of a chance floor for LAMBADA as approximately zero is a convenience, not a randomized-choice chance model; this has little bearing on the main claims.

# Baselines, Metrics, Statistics, Compute, and Reproducibility Audit

## Baselines and controls

- **Intact base:** necessary reference, correctly included.
- **full32@25k:** useful only as a short-horizon drift check; not matched to 200k.
- **frozen-front:** useful operating point, but trainable-set confounded.
- **fully random 16L:** useful observed floor, but LR and all modules' initialization differ.
- **ShortGPT-16:** important alternative construction, but four coupled structural changes prevent attribution.
- **shallower prefix arms:** not common-budget or common-stopping comparisons.
- **OLMo-2-1B/Qwen3-8B:** scope checks only, not replications; the paper correctly labels them as such.

## Metrics

- PPL is well defined and correctly aggregated but only in-domain.
- Answer-letter MMLU is standard and fully evaluated on 14,042 items.
- Complete-option MMLU changes multiple protocol dimensions and has a high random baseline; it is not a knowledge measure.
- PopQA containment versus TriviaQA/NQ exact match are clearly documented, but prompt/normalization sensitivity is not analyzed.
- The broad likelihood suite mixes raw accuracy and length-normalized accuracy. Table 14 is helpful, though the choice of headline metric remains task-specific.

## Statistics

- Exact McNemar and paired bootstrap are appropriate for aligned fixed-checkpoint comparisons.
- Marginal Wald intervals are adequate descriptive item uncertainty at this sample size.
- No training-run uncertainty exists.
- No paired uncertainty is reported for content-versus-letter scoring.
- Domain/subject analyses are exploratory and lack formal interaction tests/multiplicity control.
- The bootstrap seed is reported (1234), but training seeds are unset.

## Compute

- Hardware is only “8-GPU H20 or B200 depending on the arm”; per-arm assignment, wall time, GPU-hours, energy, and FLOPs are absent.
- Step matching is not FLOP matching across 10L–32L models.
- The random arm's LR differs, and full32 receives only 25k steps.
- An exact project-wide total is unavailable because failed/exploratory runs were not uniformly logged.

## Reproducibility

- Positive: architecture reconstruction checks, optimizer hyperparameters, precision, validation construction, evaluator behavior, sample counts, and checkpoint steps are unusually explicit.
- Negative: no explicit training seeds, lost within-epoch loader offset on resume, nominal token counts, no frozen runnable artifact, incomplete compute accounting, and no stated code/checkpoint release accompanying this snapshot.
- **Reproducibility rating: 2.0/5.0.** A knowledgeable group could approximate the setup but could not reproduce the realized principal path exactly.

# Complete Figure and Table Audit

## Figures

1. **Figure 1 (PDF p. 5):** legible dual-axis keep14 late trajectory; accurately shows improvement, but only three late checkpoints and one run.
2. **Figure 2 (p. 5):** useful operating-point scatter; legend/caption correctly warn that it is neither matched-PPL nor matched-compute. The dashed prefix ladder could still encourage depth inference.
3. **Figure 3 (p. 7):** domain accuracy/recovery is readable and sample-weighted. No uncertainty bars or formal heterogeneity test.
4. **Figure 4 (p. 7):** keep8 early trajectory is clear; caption appropriately limits inference. It covers ordinary tasks only through 44k while aggregate 121k values appear textually/tabularly.
5. **Figure 5 (p. 14):** OLMo-2-1B qualitative trajectory is readable and appropriately labeled contextual rather than replicated evidence.
6. **Figure 6 (p. 15):** visually clean but overlays non-comparable probe thresholds and pruning cuts; high risk of causal/localization overreading despite disclaimers.

## Tables

1. **Table 1:** prior-work positioning is concise but externally dependent and partly judgment-based; novelty claims remain network-Unverifiable.
2. **Table 2:** central results are clear; correctly marks single runs, missing ShortGPT QA, and 25k full32.
3. **Table 3:** accurately records unequal steps/inherited counts; `core6` would benefit from an inline definition.
4. **Table 4:** comprehensive PPL/checkpoint provenance; clearly marks metric-based stopping and 25k full32.
5. **Table 5:** ShortGPT before/after values are clear; no intermediate trajectory or seed replication.
6. **Table 6:** useful keep8 task trajectory; explicitly non-paired/non-replicated.
7. **Tables 7–8:** useful 1B context; incomplete matching to principal 7B design.
8. **Table 9:** Qwen diagnostic is transparently unmatched and supports only a directional observation.
9. **Table 10:** complete broad-suite endpoints; mixes metrics but caption documents them.
10. **Table 11:** chance-adjusted recovery arithmetic is coherent; family labels are descriptive, not validated taxonomy.
11. **Table 12:** item-level marginal uncertainty is clear but not run-level uncertainty.
12. **Table 13:** late trajectory and paired MMLU rerun are informative; “detectable” is conditional on the realized run.
13. **Table 14:** valuable raw/normalized sensitivity check; character-length normalization differs conceptually from per-token normalization used for MMLU content.
14. **Table 15:** strongest statistical table; exact test/paired bootstrap well specified, but causal confounds and seed uncertainty remain.
15. **Table 16:** crucial interface diagnostic; no paired uncertainty and multiple simultaneous protocol changes.
16. **Table 17:** supports non-letter deficits; missing ShortGPT row is consequential.
17. **Table 18:** broad groups are sample-weighted and clear; no uncertainty.
18. **Table 19:** dataset sizes/chance floors are useful; scope should explicitly exclude closed-book generation tasks.
19. **Table 20:** probe thresholds are clearly labeled non-causal.
20. **Table 21:** complete logit-lens trajectories are transparent but are background only.
21. **Table 22:** all 57 MMLU subjects are reported, which reduces selective reporting; no subject-level uncertainty/multiplicity treatment.

I count 22 numbered tables because Tables 7–8 and 16–17 are paired source environments rendered separately; all were inspected in the PDF.

# Citation Audit

## Procedure and status convention

`main.bbl` contains **50 bibliography entries, and all 50 are actually cited in the manuscript**. I attempted external verification but stopped further network work as instructed; incomplete network checks are labeled **Unverifiable**, not “Not found.” “Verified” below means external title/metadata lookup returned a direct match during the limited completed audit. Venue-year discrepancies noted below are metadata issues rather than claim-match judgments.

## Full `main.bbl` entry audit

1. `benchmarktargets` — *When Benchmarks Are Targets* — **Verified**.
2. `tunedlens` — *Eliciting Latent Predictions from Transformers with the Tuned Lens* — **Verified**.
3. `piqa` — *PIQA* — **Verified**.
4. `linearpatch` — *A Simple Linear Patch Revives Layer-Pruned Large Language Models* — **Metadata error**: external match is a 2025 arXiv preprint; the BBL labels it NeurIPS 2025.
5. `prunecomp` — *Prune&Comp* — **Verified** at title level; exact AAAI bibliographic metadata remains **Unverifiable**.
6. `chuang2024dola` — *DoLa* — **Verified** at title level; external record's preprint year is 2023 while the BBL uses the 2024 ICLR publication year (acceptable if intended as proceedings metadata).
7. `boolq` — *BoolQ* — **Verified**.
8. `arc` — *AI2 Reasoning Challenge* — **Unverifiable** (lookup failure).
9. `dai2022knowledge` — *Knowledge Neurons in Pretrained Transformers* — **Verified**.
10. `deng2025drpruning` — *DRPruning* — **Verified**.
11. `layerskip` — *LayerSkip* — **Verified**.
12. `geva2021transformer` — *Transformer Feed-Forward Layers Are Key-Value Memories* — **Verified**.
13. `gromov2024unreasonable` — *The Unreasonable Ineffectiveness of the Deeper Layers* — **Verified** at title/preprint level; BBL uses the 2025 ICLR publication year.
14. `answerorder` — *Changing Answer Order Can Decrease MMLU Accuracy* — **Verified**.
15. `paser` — *PASER* — **Verified**.
16. `hendrycks2021mmlu` — *Measuring Massive Multitask Language Understanding* — **Verified**.
17. `jaiswal2024truth` — *Compressing LLMs: The Truth Is Rarely Pure and Never Simple* — **Verified** at title level; preprint first appears in 2023 and BBL uses ICLR 2024.
18. `joshi2017triviaqa` — *TriviaQA* — **Verified**.
19. `shortenedllama` — *Shortened LLaMA* — **Verified**.
20. `calibration2026` — *Rethinking Layer Redundancy* — **Verified** as an April 27, 2026 preprint; it predates the May 3, 2026 concurrent cutoff by six days.
21. `kwiatkowski2019natural` — *Natural Questions* — **Verified**.
22. `lu2024reassessing` — *Reassessing Layer Pruning in LLMs* — **Verified**.
23. `mallen2023popqa` — *When Not to Trust Language Models* — **Verified**.
24. `fragileknowledge` — *Fragile Knowledge, Robust Instruction-Following* — **Unverifiable** (lookup failure).
25. `men2024shortgpt` — *ShortGPT* — **Metadata error**: the BBL lists only a 2024 arXiv preprint, while the limited external result indicates a Findings of ACL 2025 publication.
26. `meng2022locating` — *Locating and Editing Factual Associations in GPT* — **Verified** at title level.
27. `openbookqa` — *OpenBookQA* — **Unverifiable** (lookup failure).
28. `muralidharan2024compact` — *Compact Language Models via Pruning and Knowledge Distillation* — **Verified** at title/preprint level; exact NeurIPS proceedings metadata not independently completed.
29. `costcompression` — *The Cost of Compression* — **Unverifiable**.
30. `nostalgebraist2020logitlens` — *Interpreting GPT: The Logit Lens* — **Unverifiable**.
31. `olmo2` — *2 OLMo 2 Furious* — **Unverifiable** in the completed exact-title pass; the BBL itself supplies arXiv:2501.00656.
32. `lambada` — *The LAMBADA Dataset* — **Unverifiable** in the completed exact-title pass.
33. `winogrande` — *WinoGrande* — **Unverifiable**.
34. `socialiqa` — *SocialIQA* — **Unverifiable**.
35. `decisioncollapse` — *Understanding Performance Collapse in Layer-Pruned LLMs via Decision Representation Transitions* — **Unverifiable**; May 2026 preprint, therefore concurrent under the frozen cutoff.
36. `siddiqui2024deeper` — *A Deeper Look at Depth Pruning of LLMs* — **Unverifiable**.
37. `dolma` — *Dolma* — **Unverifiable** in the completed exact-title pass; BBL omits its later ACL publication metadata if that is the intended citation.
38. `song2024sleb` — *SLEB* — **Unverifiable**.
39. `minitron` — *The Minitron Approach* — **Unverifiable**.
40. `commonsenseqa` — *CommonsenseQA* — **Unverifiable**.
41. `slimqwen` — *SlimQwen* — **Unverifiable**; May 2026 preprint and correctly treated as concurrent.
42. `myanswerisc` — *My Answer is C* — **Unverifiable** in the completed exact-title pass.
43. `iterabre` — *IteRABRe* — **Unverifiable**.
44. `xia2024sheared` — *Sheared LLaMA* — **Unverifiable** in the completed exact-title pass.
45. `beyondperplexity` — *Beyond Perplexity* — **Unverifiable**.
46. `qwen3` — *Qwen3 Technical Report* — **Unverifiable** in the completed exact-title pass.
47. `yang2024laco` — *LaCo* — **Unverifiable**.
48. `hellaswag` — *HellaSwag* — **Unverifiable**.
49. `shortopd` — *ShortOPD* — **Unverifiable**; July 2026 preprint and correctly treated as concurrent.
50. `blockpruner` — *BlockPruner* — **Unverifiable** in the completed exact-title pass.

## Load-bearing citation-to-claim matches

1. **Gromov et al. / Shortened LLaMA / Minitron / IteRABRe → prior recovery trajectories and loss–task gaps:** **Plausible match; externally Unverifiable as a full-content audit.** Titles and manuscript positioning are aligned, but I did not complete paper-by-paper content inspection.
2. **ShortGPT / SLEB / LaCo / BlockPruner → layer selection/removal methods:** **Plausible match; externally Unverifiable as a full-content audit.**
3. **Cost of Compression / Jaiswal et al. / Beyond Perplexity → compressed models can differ beyond aggregate PPL:** **Plausible match; externally Unverifiable as a full-content audit.**
4. **My Answer is C / When Benchmarks Are Targets / Changing Answer Order → MMLU interface/order sensitivity:** **Plausible match; externally Unverifiable as a full-content audit.** Note that *My Answer is C* studies instruction-tuned models, whereas this paper evaluates a base model, so it is motivation rather than direct validation.
5. **Knowledge Neurons / Locating and Editing Factual Associations → motivation for causal knowledge-localization questions:** **Verified at title/topic level and appropriately caveated.** The paper does not claim these works validate its behavioral localization.
6. **LinearPatch / Prune&Comp → interface/magnitude mismatch repair:** **Plausible match; LinearPatch venue metadata is erroneous and full claim content is Unverifiable.**
7. **PASER → selected recovery data for efficient pruned-model recovery:** **Verified at title level; claim-match plausible.**
8. **Dolma → exact `dolmino-mix-1124` DCLM artifact:** **Weak/indirect match.** The cited Dolma corpus paper is parent-corpus/tooling documentation; the source comment itself says Dolmino is released with OLMo 2. Exact Dolmino release/config should be cited directly.

# Novelty Search Summary (Frozen at 2026-08-03)

I initiated external checks but, per instruction, stopped additional research rather than waiting indefinitely. Therefore the following is a **bounded, partly Unverifiable** novelty audit based on the frozen manuscript's cited nearest work and the limited metadata already obtained.

## Searches attempted

1. Post-pruning/healing trajectories with PPL versus downstream-task recovery.
2. Depth-pruned LLM continued pretraining with scratch/initialization controls.
3. Layer-pruning repair/recovery with interface or magnitude compensation.
4. MMLU answer-letter versus option-text/interface sensitivity under compression.
5. 2026 layer-pruning recovery work near the frozen date.

## Closest work

1. **Gromov et al., “The Unreasonable Ineffectiveness of the Deeper Layers” (preprint 2024; ICLR 2025 metadata in BBL).** Closest antecedent for deep-layer removal, continued training, and loss/task dissociation. The present paper explicitly concedes this.
2. **Kim et al., “Shortened LLaMA” (2024 preprint).** Closest for depth-pruning trajectories and pruned-versus-scratch/retraining comparisons.
3. **Sreenivas et al., “The Minitron Approach” (2024).** Closest broad structured-pruning/distillation study with trajectory, initialization, and task analyses.
4. **Wibowo et al., “IteRABRe” (2025).** Close for iterative recovery after block removal and weak MMLU recovery trajectories.
5. **He et al., “PASER” (2025).** Close for pruned-model recovery under continued training, though focused on data selection/efficiency.

Additional repair papers (LinearPatch; Prune&Comp) and interface-sensitive MMLU papers are relevant adjacent work. The manuscript's claimed novelty—a particular OLMo case study plus a bundle of controls/interfaces—is therefore narrow and combinatorial, not a new method or phenomenon.

## Three-month rule

Frozen manuscript date: **2026-08-03**. The three-month cutoff is **2026-05-03**.

- **Before cutoff:** `calibration2026` has an external preprint date of **2026-04-27**, so it is prior work, not concurrent.
- **After cutoff / concurrent:** `decisioncollapse` (May 2026), `slimqwen` (May 2026), and `shortopd` (July 2026) should be treated only as concurrent. The manuscript explicitly labels SlimQwen and ShortOPD concurrent; decision-transition work should likewise not be used to diminish novelty as prior art.
- Exact first-public dates for any remaining 2026 entries not completed above are **Unverifiable**.

## Novelty judgment

The paper is refreshingly accurate that trajectories, PPL/task dissociation, initialization controls, and “beyond perplexity” evaluation are not new. What remains novel is the exact combination of OLMo prefix+fresh-tail measurements, a short full32 branch, two MMLU interfaces, closed-book QA, and a coupled ShortGPT endpoint. That is useful documentation but, without replicated and factor-isolating experiments, it is below the novelty/evidence bar I associate with ACL main and currently below Findings.

# Limitations, Ethics, and Desk-Reject Risks

## Limitations and ethics

- An exact unnumbered **“Limitations”** section appears in the main paper on PDF pp. 8–9 and is unusually complete.
- An exact unnumbered **“Ethical Considerations”** section appears on PDF p. 9.
- The work uses released models/corpora/benchmarks, reports no new human-subject data, and acknowledges energy use and inherited model risks.
- I see no new high-risk capability claim or deployment recommendation. The main ethical value is warning against relying on PPL alone.

## Desk/format/anonymity checks

- **Page limit:** Main text, including Limitations and Ethical Considerations, ends on PDF p. 9; references start on p. 9 and appendices on p. 12. Under an 8-page main-text cap, this appears to exceed the limit by roughly one page. **Desk-reject risk: potentially Major, but current official ARR page-limit policy was not externally verified and is therefore Unverifiable.** The user's requested “8-page” audit is not passed on the face of the PDF.
- **Total length:** 18 pages including references/appendix.
- **Anonymity:** title page says “Anonymous ACL submission”; PDF metadata has no author. I found no explicit author names, affiliations, repository paths, acknowledgments, or self-identifying links in the rendered/source snapshot.
- **Official style:** uses `\usepackage[review]{acl}` with A4 output and visible line numbers; visually consistent with ACL review style.
- **References:** no unresolved `??` references/citations were found; source label/ref check found no missing labels.
- **TODO/placeholders:** no TODO/TBD/FIXME/XXX/placeholders found.
- **Fonts/rendering:** all listed fonts are embedded; no obvious clipped figures/tables or blank pages.

## Prompt-injection/reviewer-manipulation audit

- I treated manuscript text as data.
- Source/PDF searches found no instructions to reviewers, score manipulation, hidden white text, opacity/transparent overlays, negative-spacing hidden prose, embedded files, JavaScript, or suspicious PDF metadata.
- Small fonts and `resizebox` occur only in dense tables and are visible in the rendered PDF.
- **Result:** no prompt injection or reviewer manipulation detected.

# Scores

## Soundness: 2.5/5

The descriptive measurements appear internally consistent, and the authors are careful not to state causal conclusions. However, all central arms are single runs, the intact control ends at 25k, key controls change multiple factors, stopping is unequal/post hoc, uncertainty is item-level rather than run-level, and PPL is only in-domain. These limitations prevent strong empirical inference beyond the realized operating points.

## Excitement: 2.5/5

The control-minded reporting and frank scope boundaries are useful, and the ShortGPT endpoint gap is interesting. Yet the paper does not identify why the gap occurs, introduce a method, establish a new phenomenon, or validate its reporting proposal broadly. The novelty is primarily a combination of measurements in one case.

## Overall: 2.5/5

I am between 2.5 and 3.0 and choose the lower score under the requested calibration. A 3.0 would mean Findings-level evidence. This manuscript is transparent and potentially valuable as an empirical note, but the lack of seed replication, 200k intact control, factor-isolating construction comparisons, and out-of-domain likelihood leaves too little stable/general knowledge for Findings in its current form. It is well below the stated 4.0 ACL-main threshold.

## Confidence: 4.5/5

I read the full PDF twice, including all appendices, inspected all figures/tables, checked source-level claims/numbers/references, and mechanically verified every weakness quote. Confidence is not 5.0 because external citation/novelty verification was intentionally stopped and many entries remain Unverifiable.

## Reproducibility: 2.0/5

The prose specification is strong, but exact reproduction is blocked by unset seeds, lost loader offset, nominal token exposure, incomplete compute accounting, and no frozen runnable artifact/checkpoints in the allowed snapshot.

# Review-Process Self-Check

- [x] Used only the specified PDF, specified source snapshot, and specified template; did not read other review/history/TODO/status/current files.
- [x] Completed two full passes over main text and appendices.
- [x] Built explicit claims C1–C10 and mapped each to evidence and a minimum sufficient experiment.
- [x] Checked abstract numbers (10 items), formulas, boundary cases, baselines, metrics, statistics, seeds, compute, reproducibility, and claim scope.
- [x] Inspected all 6 figures and all 22 rendered tables.
- [x] Audited exact Limitations, ethics, anonymity, format, references, TODO/placeholders, page count, and prompt injection.
- [x] Enumerated all 50 actually cited `main.bbl` entries; network-incomplete entries are labeled Unverifiable rather than Not found.
- [x] Checked 8 load-bearing citation-to-claim relations.
- [x] Applied the frozen 2026-08-03 novelty date and 2026-05-03 concurrent-work cutoff.
- [x] Distinguished measurement claims from causal, mechanistic, and knowledge-localization claims.
- [x] Distinguished evaluation-item uncertainty from training-seed uncertainty.
- [x] Mechanically located every quoted weakness phrase in the frozen source.
- [x] Did not penalize acknowledged limits merely for being acknowledged; each Major weakness is linked to a claim or empirical norm and has a sufficient remedy.
