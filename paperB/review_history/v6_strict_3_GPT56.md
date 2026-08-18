review_mode: strict
soundness: 3.0
excitement: 2.5
overall: 3.0
confidence: 4.0
reproducibility: 2.0

## Paper Summary

This paper studies a deliberately narrow proxy-validity question after depth pruning: on literal OLMo-2-7B continued-pretraining paths, does an improvement in same-source held-out perplexity imply recovery to the intact model on measured knowledge-sensitive evaluations? The principal keep14+fresh2 run improves from PPL 10.826 to 10.561 between 128k and 200k steps, while answer-letter MMLU rises only from .3012 to .3191 versus .6053 for the intact base; the final keep14 endpoint is also substantially below the base on PopQA, TriviaQA, and NQ-open (Table 2; §5.1). The paper adds a 25k intact-CPT point, frozen and fully-random 16-layer operating points, paired letter/content MMLU scoring, and a coupled ShortGPT-16 endpoint. It explicitly disclaims causal localization, a universal recovery law, a prospective PPL threshold, and clean one-factor attribution.

The central descriptive conclusion is supported **as phrased locally**: a lower in-domain PPL alone was insufficient to establish intact-base recovery on the reported keep14 path and measured tasks. The work is careful about several otherwise serious confounds. However, it remains a single historical training realization with no replayable seed/loader state, no 200k intact control, no ShortGPT closed-book results, and no out-of-domain PPL/contamination audit. These limitations materially reduce reproducibility and the force of several auxiliary “bounds alternatives” interpretations, but they do not overturn the narrow observed-path claim. I therefore view this as a careful Findings-level measurement case study rather than a main-conference contribution.

## Claims and evidence audit

- **C1 (principal, supported at observed-path scope):** PPL improvement alone does not imply target recovery on the reported keep14 path and measured evaluations. Evidence: §4 defines recovery as closing the large intact-base gap; §5.1/Table 2 show PPL decreasing to 10.561 while MMLU remains .319 versus .605, and all three closed-book endpoints remain below base. Figure 1 shows the 128k–200k joint trajectory.
- **C2 (supported but only conditionally):** The late keep14 MMLU gain is nonzero at the fixed realized checkpoints. Evidence: §5.1 and Appendix Table 12 report +1.68 points with item-bootstrap CI [1.08, 2.29]. This is item uncertainty, not training-run uncertainty.
- **C3 (partly supported):** The MMLU conclusion is not solely an answer-letter artifact. Evidence: Table 15 exposes multi-factor letter/content sensitivity and Table 16 shows a large base–keep14 gap on three generative QA sets. The latter lacks aligned per-item predictions/intervals, so it supports a large descriptive gap, not calibrated comparative inference.
- **C4 (supported only as construction dependence):** The keep14 endpoint is not shared by every tested nominal 16-layer construction. Evidence: Table 2 gives ShortGPT .474 MMLU and 9.780 PPL at 200k. The paper correctly says this does not identify the responsible factor because block selection, inherited count, final-layer retention, and fresh-tail use all change.
- **C5 (not established beyond a weak boundary):** The operating points “bound three complete explanations” (§6.4). The 25k full32, multi-factor interface comparison, and confounded nulls rule out only limited versions of those explanations; they do not support a long-horizon corpus-shift control or causal explanations.

### Minimum sufficient experiments

For C1, one internally consistent path with at least two post-intervention checkpoints and target evaluations is sufficient to falsify the universal-looking operational implication **for that path**, and the paper has this. For an operational recommendation that PPL should not be used as recovery evidence more broadly, the minimum stronger design would be multiple training seeds plus a matched intact 200k branch and at least one shifted/out-of-domain likelihood evaluation. For C3, aligned predictions (or paired intervals) for the three QA tasks and the ShortGPT endpoint would be sufficient to substantiate cross-interface/generalization claims. For causal construction claims, a one-factor-matched block-selection/fresh-tail/inherited-count ablation is required; the manuscript appropriately does not make that claim.

## Strengths

- **S1 — unusually precise scope discipline.** Section 4 (PDF lines 287–305) formalizes the tested implication, defines “target recovery” relative to the intact base, and says the study is not a prospective threshold test. The conclusion and Figure 1 caption repeat the observed-path, non-causal boundary. This is substantially better calibrated than the title alone suggests.
- **S2 — direct, informative principal evidence.** Table 2 and §5.1 (PDF lines 319–338) jointly report PPL, MMLU, and three closed-book endpoints, while retaining the late keep14 trajectory rather than inferring dynamics from only an endpoint. The item-paired MMLU analysis is correctly labeled conditional on realized checkpoints.
- **S3 — confounds are visible rather than relabeled as controls.** Table 2 and §5.3 state that full32 stops at 25k, Random changes LR and lexical modules, Frozen changes the trainable set, and ShortGPT changes four construction factors. This supports a descriptive operating-point comparison without overstating causal attribution.
- **S4 — MMLU interface diagnosis is valuable.** §3.3, §5.2, and Table 15 clearly distinguish letter versus complete-option scoring, document that several factors change together, and show that random initialization attains a high content-score floor while remaining letter-chance. The artifact includes six headline-arm prompt-free MMLU per-item records and paired keep14 trajectory records, sufficient to inspect those reported MMLU aggregates.
- **S5 — Figure 1 communicates the headline result effectively.** The rendered figure is legible at normal PDF zoom, its visual hierarchy is strong, and it places the main caveats (one run, 25k-only full32, coupled ShortGPT, random content floor) adjacent to the plots. It is among the manuscript’s clearest components.

## Weaknesses

### W1 — No seed replication and an unreplayable principal history limit the reliability of the central empirical generalization. **Major**

- **Location:** Limitations, PDF lines 500–510; Appendix B, PDF lines 813–829.
- **Exact quote:** “seeds were not explicitly set”.
- **Problem:** The principal keep14 result, all same-shape points, and ShortGPT are each one run. Moreover, the historical seed is unavailable and the keep14 resume lacks the within-epoch loader offset. MMLU bootstraps/McNemar quantify only evaluation-item variability conditional on selected checkpoints, not training stochasticity.
- **Affected claim/norm:** C1 remains valid as a literal observation, but the paper’s reporting recommendation and its broader operational framing need evidence that the PPL–target dissociation is stable across realizations rather than a trajectory-specific outcome. ARR empirical standards require run-level uncertainty when a conclusion rests on training dynamics.
- **Sufficient remedy:** Re-run the principal keep14 and a matched intact continuation for at least 3 independently seeded runs from a fully specified data-order/offset state; report across-run distributions for PPL and each target metric. If historical replay is impossible, narrow the title/abstract/recommendation further to an archival single-run case report and make the non-reproducibility a front-page limitation.

### W2 — The intact continued-pretraining comparator does not test the 200k counterfactual, so it cannot strongly rule out long-horizon corpus/training effects. **Major**

- **Location:** §5.2, PDF lines 341–350; Table 2; Limitations, PDF lines 505–507.
- **Exact quote:** “no full32 result is available after 25k”.
- **Problem:** The full32 observation is 25k/6.6B nominal presentations, while keep14 ends at 200k/52.4B. A small 25k full32 deficit makes an *early catastrophic* corpus-shift story incomplete, but it does not identify what intact continuation would do after 200k under the same history and evaluation schedule.
- **Affected claim/norm:** This does not damage the within-keep14 C1 observation, but weakens C5’s claim that operating points bound corpus-shift alternatives and weakens any interpretation of the base-to-keep14 endpoint gap as structural rather than training-history dependent.
- **Sufficient remedy:** Train/evaluate full32 to 200k with matched data pipeline and checkpoints (ideally across seeds). Pending that, replace “bound”/“constrain” language with “the available 25k point rules out only a large early degradation,” consistently in abstract, introduction, §5.2, §6.4, and Figure 1.

### W3 — The artifact permits partial verification but not end-to-end reproduction or audit of the central closed-book evidence. **Major**

- **Location:** Appendix B, PDF lines 857–868; Limitations, PDF lines 511–521; anonymous artifact README.
- **Exact quote:** “Not included: ... checkpoints/model weights, training arrays”.
- **Problem:** The snapshot contains evaluator scripts, configs, aggregate closed-book summaries, MMLU per-item score records, and paired MMLU files, but excludes checkpoints, training arrays, weights, seed/offset history, benchmark text, and closed-book generations/per-item predictions. Its evaluator provenance is a source-file snapshot of local-only commits, not recoverable public history. Consequently, readers can recompute some reported MMLU analyses from released outputs, but cannot rerun models, verify closed-book per-example scoring, reproduce training, or check data contamination.
- **Affected claim/norm:** The paper is transparent about this, but reproducibility is substantially below the norm for an empirical training-dynamics paper. The closed-book recurrence is load-bearing for C3 yet has only aggregate files and no paired uncertainty.
- **Sufficient remedy:** Release public/reconstructable model checkpoints or a feasible access path; versioned training/evaluation code and environment; data construction manifests; seed/offset metadata for new reruns; and sanitized closed-book per-item predictions/generations or at minimum item IDs, scores, and scorer outputs. If redistribution is impossible, provide executable scripts that fetch every public dependency and release reproducible reruns.

### W4 — Missing ShortGPT closed-book evaluation leaves the strongest alternative construction only partially assessed. **Minor**

- **Location:** Table 2, PDF line 397; Limitations, PDF lines 516–518.
- **Exact quote:** “no ShortGPT closed-book evaluation is available.”
- **Problem:** ShortGPT is the paper’s strongest counterexample to treating nominal 16-layer depth as decisive, but it is compared only on PPL/MMLU. The paper cannot determine whether its stronger MMLU endpoint also extends to the generative knowledge-sensitive tasks used to strengthen C1/C3.
- **Affected claim/norm:** It does not invalidate C4’s MMLU construction-dependence observation. It limits claims that the full diagnostic package or “target recovery” comparison generalizes across constructions.
- **Sufficient remedy:** Run the same PopQA/TriviaQA/NQ-open protocol for ShortGPT and release the aligned predictions; otherwise state in §5.3 and §6.4 that construction dependence is shown only for PPL and MMLU.

### W5 — The main Figure 1 is readable but over-dense and risks conflating an observed endpoint gap with the formal implication. **Minor**

- **Location:** Figure 1, PDF page 2 / lines 72–84.
- **Exact quote:** “supports proxy insufficiency of improvement alone”.
- **Problem:** The left panel is the actual implication test (within-path PPL decrease and persistent base gap), whereas the right panel combines heterogeneous endpoints with two scales and multiple caveats. Its small labels and colored callouts are legible, but close to the lower practical limit in the two-column rendering; the strong banner can be read more broadly than the caption’s careful “these observed paths” qualification.
- **Affected claim/norm:** Presentation clarity for C1/C5, not numerical validity.
- **Sufficient remedy:** Make the banner say “Observed keep14 path: lower PPL did not establish intact-base recovery”; move the endpoint/null material to a separate figure or simplify it; enlarge axis/callout text and label all endpoint budget differences directly in the plot subtitle.

## Questions that could change the score

1. Can the authors provide 3+ fully specified reruns of keep14 and 200k full32, or explain why a new reproducible experiment is infeasible? This would most affect soundness and reproducibility.
2. Are sanitized closed-book per-item predictions/scorer outputs available for all reported arms, and can ShortGPT be evaluated on the same three tasks? This would determine how strongly the non-letter recurrence and construction conclusions can be audited.
3. Can the authors make the intended logical target explicit in the title/abstract: a descriptive counterexample to using **improvement alone** as a recovery certificate, rather than a claim that PPL has no diagnostic value or that any PPL decrease necessarily fails to predict later recovery?

## Non-scoring suggestions and mechanical issues

- The manuscript’s improvement-only definition is mostly precise and should be retained: §4 explicitly rejects an absolute threshold, base-relative tolerance, plateau rule, and construction-specific calibration. I recommend moving one sentence of that definition into the abstract, because “PPL improvement alone does not imply target recovery” is otherwise easily misread as a universal predictive claim.
- State in the abstract that the decisive evidence is one keep14 run and that full32 is available only to 25k; these caveats currently appear later and are important for interpreting the operational claim.
- Correct the duplicated `keep10 endpoint` row in Appendix Table 15 (PDF page 16). It appears twice with identical values.
- Appendix Table 4 says `core6` is an average of six tasks but lists only five named tasks plus WinoGrande; this is understandable but should be typeset as an explicit six-item list for auditability.
- Table 1 is useful but its study-level binary coding is fragile and not independently substantiated in the paper. Add citations in the table or move detailed comparative assertions to prose/appendix with exact evidence.
- The manuscript visually inspected well: Figure 1 is readable; Figure 2 and tables are generally legible, though the two-column appendix tables are dense. I found no unresolved references, TODO/placeholders, prompt-injection/reviewer-manipulation text, or anonymization breach in the frozen source/PDF. The paper is 17 PDF pages including appendix; main text, limitations, ethics, references, and appendix are present.

## Soundness, Excitement, Overall, Confidence, Reproducibility

- **Soundness: 3.0/5.0.** The central observed-path result follows from reported measurements and is cautiously framed. Lack of seeds, replayability, 200k full32, OOD PPL, and per-item closed-book artifacts precludes stronger operational or causal interpretation.
- **Excitement: 2.5/5.0.** The question is practically relevant, and the interface/null diagnostics are useful. Novelty is primarily a carefully documented diagnostic combination and reporting recommendation, not a new method, general law, or broadly validated empirical finding.
- **Overall: 3.0/5.0 — Findings.** A transparent, focused measurement case study whose narrow conclusion is credible, but whose evidence/reproducibility gaps make it short of main-conference level (4.0 under the requested calibration).
- **Confidence: 4.0/5.0.** The frozen PDF, complete source/appendices, artifact manifest, scripts, and all cited bibliography entries were audited. The main uncertainty is about unreleased historical training and prediction artifacts, not about what the submission claims to contain.
- **Reproducibility: 2.0/5.0.** Evaluation-side MMLU auditing is meaningfully supported, but training replay and independent end-to-end reproduction are blocked by missing checkpoints/arrays, unavailable seed and loader offset, local-only history, and missing closed-book per-item outputs.

## Limitations/Ethics and desk-reject risks

The paper has explicit `Limitations` and `Ethical Considerations` sections and appropriately notes English-only scope, one family/mixture/recipe, absent compute accounting, in-domain PPL, no contamination audit, and no claim of knowledge deletion/localization. I found no evident desk-reject issue: the submission is anonymous, uses ACL review style, contains the required sections, has no unresolved citation keys, and does not appear to contain hidden text or reviewer-directed instructions. The principal risk is not a formatting violation but that the contribution could be viewed as too narrow/single-run for the venue unless positioned explicitly as Findings-level measurement evidence.

## Citation audit

All 33 bibliography entries in `main.bbl` are cited by the frozen source. Twenty entries carry a direct DOI/arXiv link in the frozen bibliography and those landing pages resolved during this audit (**Verified**). Thirteen entries have no persistent link in `main.bbl`; because I did not rely on a secondary authoritative metadata record for those entries, they are marked **Unverifiable**, not “Not found.” This does not mean that every comparative characterization in Table 1 has been independently reproduced from each paper.

| Key | Status | Citation-claim check |
|---|---|---|
| Alzahrani et al. (2024), `benchmarktargets` | Verified | Supports leaderboard sensitivity framing. |
| Chen et al. (2025), `linearpatch` | Verified | Supports repair/patching discussion. |
| Chen et al. (2026), `prunecomp` | Verified | Supports pruning-compensation discussion. |
| Deng et al. (2025), `deng2025drpruning` | Unverifiable | Supports robust pruning context. |
| Gromov et al. (2025), `gromov2024unreasonable` | Verified | Load-bearing closest antecedent for deep-layer removal/recovery-gap framing. |
| Gupta et al. (2024), `answerorder` | Verified | Supports answer-order/interface sensitivity. |
| He et al. (2025), `paser` | Verified | Supports post-training data selection comparison. |
| Hendrycks et al. (2021), `hendrycks2021mmlu` | Unverifiable | Supports MMLU definition. |
| Jaiswal et al. (2024), `jaiswal2024truth` | Unverifiable | Load-bearing “low PPL can coexist with knowledge deficits” context. |
| Joshi et al. (2017), `joshi2017triviaqa` | Unverifiable | Supports TriviaQA evaluation. |
| Kim et al. (2024), `shortenedllama` | Verified | Load-bearing retraining/init comparator antecedent. |
| Kim et al. (2026), `calibration2026` | Verified | Supports calibration/selection context. |
| Kwiatkowski et al. (2019), `kwiatkowski2019natural` | Unverifiable | Supports NQ-open evaluation. |
| Lu et al. (2024), `lu2024reassessing` | Unverifiable | Supports task/calibration sensitivity discussion. |
| Mallen et al. (2023), `mallen2023popqa` | Unverifiable | Supports PopQA evaluation. |
| Martra (2025), `fragileknowledge` | Verified | Supports adjacent width-pruning/interface context. |
| Men et al. (2025), `men2024shortgpt` | Verified | Load-bearing for ShortGPT method and nearest-work positioning. |
| Muralidharan et al. (2024), `muralidharan2024compact` | Unverifiable | Supports pruning/distillation context. |
| Namburi et al. (2023), `costcompression` | Verified | Load-bearing beyond-perplexity/parametric-knowledge motivation. |
| OLMo Team (2025), `olmo2` | Verified | Supports base-model provenance. |
| Shi et al. (2026), `decisioncollapse` | Verified | Supports decision-transition related-work discussion. |
| Siddiqui et al. (2024), `siddiqui2024deeper` | Unverifiable | Supports depth-pruning sensitivity context. |
| Song et al. (2024), `song2024sleb` | Unverifiable | Supports layer-selection related work. |
| Sreenivas et al. (2024), `minitron` | Unverifiable | Load-bearing structured pruning/distillation antecedent. |
| Tang et al. (2026), `slimqwen` | Verified | Concurrent-work characterization; treated as contemporaneous in manuscript. |
| Wang et al. (2024), `myanswerisc` | Verified | Supports first-token/answer-interface sensitivity. |
| Wibowo et al. (2025), `iterabre` | Verified | Load-bearing iterative recovery antecedent. |
| Xia et al. (2024), `xia2024sheared` | Unverifiable | Supports pruning/data-allocation context. |
| Xu et al. (2024), `beyondperplexity` | Verified | Supports multi-dimensional compression evaluation motivation. |
| Yang et al. (2025), `qwen3` | Unverifiable | Supports Qwen model provenance. |
| Yang et al. (2024), `yang2024laco` | Verified | Supports layer-collapse related work. |
| Zhang et al. (2026), `shortopd` | Verified | Concurrent-work characterization; treated as contemporaneous in manuscript. |
| Zhong et al. (2025), `blockpruner` | Verified | Supports block-granularity pruning context. |

### Load-bearing citation–claim matches

1. Gromov et al. is appropriately presented as the closest antecedent for loss/task dissociation after removing deep blocks, not as a novelty-free proof of this exact OLMo diagnostic.
2. Shortened LLaMA and Minitron support the claim that retraining trajectories and initialization choices predate this paper; the manuscript appropriately limits its novelty to the combination of diagnostics.
3. Namburi, Jaiswal, and Xu support the high-level premise that aggregate likelihood/compression metrics can miss other behavioral dimensions; they do **not** by themselves establish this manuscript’s observed-path implication, which relies on the new measurements.
4. Wang, Alzahrani, and Gupta support concern that multiple-choice scores can be interface/evaluation sensitive; the manuscript correctly avoids attributing its own letter/content difference to a single answer-symbol mechanism.
5. SlimQwen and ShortOPD are clearly labeled concurrent work under the stated three-month rule; they should not be used to diminish novelty retroactively beyond that limited positioning.

## Novelty-search summary

I searched the closest literature around depth pruning, recovery/retraining, loss–task dissociation, and evaluation-interface sensitivity. The closest papers are: (1) **Gromov et al. (2025), The Unreasonable Ineffectiveness of the Deeper Layers**; (2) **Kim et al. (2024), Shortened LLaMA**; (3) **Sreenivas et al. (2024), Minitron**; (4) **Wibowo et al. (2025), IteRABRe**; and (5) **Men et al. (2025), ShortGPT**. The manuscript’s novelty claim is defensible only as the particular OLMo observed-path measurement package: late checkpoints, an available short intact branch, MMLU-interface comparison, closed-book QA, and visibly confounded operating points. It should not claim novelty for recovery trajectories, evaluating beyond PPL, initialization comparisons, or depth-selection methods. SlimQwen (May 2026) and ShortOPD (July 2026) are plausibly within the manuscript’s declared contemporaneous-work window and should remain treated as such.

## Review-process self-check

- Reviewed the frozen v6 PDF twice, including all appendix pages, and checked frozen submission source sections/tables against quoted weaknesses.
- Examined the anonymous artifact snapshot contents, README, manifest, configs, score files, and evaluator scripts list; did not claim it contains unavailable weights/training arrays/predictions.
- Mechanically checked all citations: 33 cited keys, 33 `main.bbl` entries, no missing or uncited keys; resolved every directly embedded DOI/arXiv URL; entries without a direct persistent bibliography link are explicitly marked Unverifiable.
- Inspected rendered Figure 1 directly and enumerated all rendered figures/tables; checked PDF/source for unresolved references, placeholders, hidden-text/reviewer manipulation patterns, and anonymity/style issues.
- Did not inspect any other Paper B review/history/TODO/status/current/calibration material. Every weakness above is tied to quoted frozen-source text and distinguishes a missing experiment/artifact only where it affects a stated claim or reproducibility norm.
