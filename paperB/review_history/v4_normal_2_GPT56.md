```yaml
review_mode: normal
soundness: 3.5
excitement: 3.0
overall: 3.0
confidence: 4.0
reproducibility: 2.0
```

# Summary and recommendation

This paper is an explicitly bounded observational study of continued pretraining after OLMo-2 depth pruning. Its principal construction keeps the first 14 of 32 pretrained blocks, appends two fresh blocks, and trains for 200k optimizer steps. Rather than proposing a pruning algorithm, the paper asks whether likelihood recovery, target-capability recovery, evaluation interface, construction, and recovery budget tell the same story. The answer in the observed runs is “no”: keep14 improves to PPL 10.561 but remains at .319 answer-letter MMLU, versus 7.398 and .605 for the intact base; complete-option MMLU is .383, but the fully random 16-layer arm is already .360 on that metric. PopQA/TriviaQA/NQ-open are .142/.294/.060 for keep14 versus .257/.636/.205 for the base. A coupled ShortGPT-16 construction reaches PPL 9.780 and MMLU .474.

I find the descriptive measurements useful and unusually candid about what they do **not** identify. In particular, the paper does not claim that PPL and capability are globally uncorrelated, does not turn the ShortGPT comparison into a selection-only ablation, and does not mistake item-level intervals for training-run uncertainty. However, the main empirical conclusions rest on single training realizations, unmatched controls, and an incompletely reproducible training history. The paper therefore supports a careful **case-study warning and reporting proposal**, but not a stable empirical characterization of post-pruning recovery. My recommendation is **Findings-level (overall 3.0)** rather than main-conference level: soundness is above the overall score because the claim scope is restrained, while excitement is moderate because closely related work already establishes recovery trajectories and capability-specific failure.

# Claims map

| ID | Paper claim | Main evidence | Assessment |
|---|---|---|---|
| C1 | In the observed keep14 run, likelihood recovers much more than answer-letter MMLU and closed-book QA. | Table 2; Fig. 1; Appendix Tables 4, 10, 13, 17. PPL 10.826→10.561 from 128k→200k; MMLU .3012→.3191; base .6053. | Supported for this one realized run and observed horizon. |
| C2 | A short-horizon intact-model branch weakens a simple corpus-shift explanation. | Full32@25k: PPL 7.670, MMLU .588, PopQA .228, TriviaQA .572, NQ .158, all near the base relative to keep14. | Supported only through 25k; not a 200k counterfactual. |
| C3 | MMLU scoring interface materially changes apparent recovery, but complete-option scoring has a fluency floor. | Table 16: keep14 letter/content-norm .3184/.3832; random .2470/.3598; ShortGPT .4742/.4012. | Descriptively supported; protocols change several factors jointly and lack paired uncertainty. |
| C4 | The deficit is not merely an answer-letter formatting failure. | Closed-book QA gaps in Table 17 and main Table 2. | Supported directionally, though generation-set sizes and uncertainty are not reported. |
| C5 | Nominal 16-layer depth does not determine a unique observed endpoint. | ShortGPT-16 vs keep14: .474 vs .319 MMLU and 9.780 vs 10.561 PPL after 200k. | Supported as a construction-level observation, not a causal attribution. |
| C6 | Same-shape operating points show inherited/trained structure matters for answer-letter MMLU. | Table 15 paired item tests: keep14−random +7.11 pp, keep14−frozen +5.50 pp. | Supported conditionally on checkpoints; LR and trainable modules are unmatched, and seed variance is unknown. |
| C7 | Recovery differs across MMLU domains and prefix-arm trajectories. | Fig. 3/Table 18; keep8 and keep14 trajectories; 128k→200k paired keep14 gain 1.68 pp, CI [1.08, 2.29]. | Descriptive only; no multiplicity/domain tests and no replicated trajectories. |
| C8 | Recovery studies should separately report likelihood, target capability, interface, exact construction, and budget. | Synthesis of C1–C7. | Reasonable reporting proposal; not validated as a universal standard. |
| C9 | Supplementary readouts order semantic, MMLU, and next-token thresholds. | Appendix C, Tables 20–21 and Fig. 6. | Exploratory background only; correctly excluded from causal recovery claims. |

# Strengths

1. **Excellent claim bounding and control bookkeeping.** The abstract already says the study “does not establish seed-stable dynamics or a general law,” and the body repeatedly labels full32 as short-horizon, random/frozen as operating points, and ShortGPT as a coupled construction comparison. This is materially better than overclaiming from the available data.

2. **The control bundle is diagnostically useful despite imperfect matching.** The combination of an intact CPT branch, same-shape inherited/frozen/random points, a different 16-layer construction, two MMLU interfaces, and three closed-book tasks makes several simplistic interpretations untenable. The random arm’s PPL 11.498 versus frozen 12.797, but letter-MMLU .247 versus .263, is a concrete example of why one scalar is insufficient within these operating points.

3. **The paper reports enough numbers to inspect the story rather than relying on prose.** Examples include the 1.428× PPL tax, the .605→.319 MMLU gap, the .383 versus .360 normalized-content scores, the +7.11 pp paired keep14–random difference, and the late +1.68 pp MMLU change with CI [1.08, 2.29]. All figures and 22 tables were inspected; captions generally state matching limitations and provenance.

4. **Statistical interpretation is mostly careful.** The paper explicitly distinguishes marginal item uncertainty, paired item tests, and absent training-seed uncertainty. It does not claim that extremely small McNemar p-values establish run-to-run stability.

5. **The bounded measurement contribution is legitimate even without a new algorithm.** The OLMo-specific recovery path and evaluation-interface/control bundle can be useful as a negative/diagnostic result, and I do not penalize the paper merely for not proposing a new pruning method.

# Weaknesses

## W1 — Major: no training-seed replication for the central recovery and construction claims

- **Location and exact quote (Limitations, p. 8, lines 582–585; 18 words):** “keep14, ShortGPT, and the same-shape points are single runs; item-level intervals do not measure variation from initialization”.
- **Problem:** C1, C5, C6, and C7 depend on one realization per arm. The item-level McNemar/bootstrap analyses condition on fixed checkpoints and cannot quantify variation from fresh-block initialization, data order, optimization, or ShortGPT selection. This is especially consequential because the main scientific object is a *trajectory*, not merely a deterministic evaluation of a released checkpoint.
- **Affected claim/norm:** Empirical robustness and statistical support for the claimed recovery separation and construction dependence. ARR norms expect uncertainty at the level of the stochastic experimental unit when feasible.
- **Why it matters:** The paper may have accurately observed a large gap, but it cannot tell whether the size, late slope, or relative ordering is seed-stable. The manuscript itself says this limitation prevents a general law; it also prevents elevating the evidence beyond a case study.
- **Sufficient remedy:** Run at least 3 seeds for keep14 and one most informative comparator under identical data order/LR/schedule (preferably a matched 16-layer inherited or scratch arm), report mean/dispersion for PPL and MMLU trajectories at preregistered checkpoints, and retain item-paired analyses within each seed. If this is computationally infeasible, release all checkpoints/predictions and make “single-run case study” even more prominent in the title/claims; that mitigates presentation but does not fully resolve the evidence gap.

## W2 — Major: the minimal sufficient controls are unmatched in horizon or coupled in several causal factors

- **Location and exact quote (Limitations, p. 8, lines 589–591; 22 words):** “neither the depth ladder nor full32 is a step-, token-, or FLOP-matched 200k counterfactual.”
- **Problem:** The intact branch ends at 25k, the shallow prefix arms end at unequal metric-selected steps, random init uses a 5× higher LR, frozen-front changes the trainable parameter set, and ShortGPT changes inherited count, contiguity, final-block retention, and fresh-tail use simultaneously. These controls are useful for falsifying simple stories, but they do not isolate corpus drift, initialization, adaptation, block choice, or fresh-tail damage.
- **Affected claim/norm:** C2, C5, C6, and any inference about what CPT “restores.” A minimally sufficient experiment should match the intervention of interest while varying one factor at a time or explicitly frame results as heterogeneous endpoints.
- **Why it matters:** The largest comparative result (.474 versus .319 MMLU) cannot identify why ShortGPT is better; the 25k full32 result cannot exclude long-horizon corpus forgetting; and random-versus-inherited comparisons mix initialization with optimization.
- **Sufficient remedy:** Add (i) full32 to 200k on the same token order/schedule; (ii) random-init 16L at the inherited-arm LR and, ideally, inherited keep14 at the random-arm LR; and (iii) a small factorial construction set at 16 layers—e.g., inherited prefix16/no fresh tail, keep14+fresh2, and a non-contiguous 14+fresh2 variant while controlling final-block retention. At minimum, one matched control for each headline causal alternative is needed.

## W3 — Major: exact reproduction is presently impossible and compute reporting is incomplete

- **Location and exact quote (Limitations, p. 9, lines 606–609; 18 words):** “Exact reproduction is limited by unset training seeds, an unrecorded resumed data-loader offset, incomplete compute accounting”.
- **Problem:** Appendix B provides many hyperparameters, but the runs have unset seeds; keep14 resumed with an unrecorded loader offset; the manuscript states that there is no frozen runnable artifact and does not provide executable configs, an environment lock, checkpoint hashes, or per-item files; and hardware is only “H20 or B200 depending on the arm.” Unique-token exposure is therefore unknown, and arm-level GPU-hours/FLOPs are absent.
- **Affected claim/norm:** Reproducibility, compute transparency, and exact verification of all central measurements.
- **Why it matters:** This is not just an artifact-release nicety: the unknown resume behavior directly affects the training trajectory, and mixed hardware plus no arm mapping prevents efficiency/compute interpretation. A measurement paper should make the measurement path reconstructable.
- **Sufficient remedy:** Release anonymized runnable training/evaluation code, exact per-arm configs, package/container lock, explicit seeds, checkpoint and prediction hashes, exact data-array indices including resume offsets, per-arm hardware/node mapping, wall-clock/GPU-hours and estimated training FLOPs/tokens. For the historical keep14 run, document the best reconstructable data-order trace and label exact rerun equivalence as impossible.

## W4 — Minor: closed-book evidence lacks sample counts and uncertainty, and task normalizations are not fully specified

- **Location and exact quote (Appendix B.2, p. 14, lines 148–155; 23 words):** “PopQA's headline is normalized-answer containment; TriviaQA and NQ-open use exact match. We additionally retain exact match, containment, and token F1”.
- **Problem:** The closed-book tables do not state evaluated split sizes, confidence intervals, or paired tests, and the exact alias/normalization implementation is described only at a high level. In contrast, the likelihood suite has an explicit sample-count table. The apparent random>keep14 NQ difference (.0632 vs .0598) illustrates why uncertainty matters.
- **Affected claim/norm:** C4 and the strength of the “not answer-letter-only” conclusion; complete metric reporting.
- **Why it matters:** The large base-to-keep14 gaps are likely robust, but without n and uncertainty the reader cannot assess smaller control differences or reproduce normalization exactly.
- **Sufficient remedy:** Add dataset split and n for PopQA/TriviaQA/NQ-open, exact normalization/alias rules or code version, paired bootstrap confidence intervals for base/full32/keep14/frozen/random, and sensitivity results for EM/containment/F1 already said to be retained.

# Questions for the authors

1. Why was full32 stopped at 25k? Is a 200k checkpoint genuinely unavailable, or was it terminated because the short-horizon metrics appeared stable? Please report its loss/MMLU trajectory at all available intermediate checkpoints.
2. What exact learning rate and trainable modules were used for ShortGPT-16? The text says “in inherited runs” LR is 2e-5, but a per-arm configuration table would remove ambiguity.
3. Were all arms trained on the same ordered corpus array before shuffling/resume effects? Which arms used H20 versus B200, and did implementation or numerical settings differ across node types?
4. How were the shallow “metric-based stopping” decisions made, by whom, and using which metrics/checkpoint look frequency? Were these decisions made before viewing the final comparisons?
5. Can the authors provide sample counts and paired uncertainty for all three closed-book datasets, especially for comparisons among keep14, frozen, and random?
6. Is there an existing matched prefix16/no-fresh-tail or keep14+copied-final-block run? Either would substantially sharpen interpretation of the ShortGPT gap.

# Suggestions

- Add a compact per-arm design matrix listing architecture, inherited layers, fresh layers, copied lexical modules, trainable modules, LR, steps, nominal tokens, resume events, seed, hardware, and checkpoint-selection rule.
- Replace “fluency floor” with a more operational phrase such as “high random-init baseline under normalized option-text scoring,” unless fluency itself is independently measured.
- Report correlation only if defined over a matched set; the current manuscript correctly avoids global correlation claims, so keep that restraint.
- Consider moving the exploratory layer-wise readout section to supplementary material outside the core paper. It is carefully caveated but not evidence for the main claims and consumes substantial space.
- If page pressure requires prioritization, keep the main trajectory, main operating-point table, paired tests, protocol-control table, and reproducibility details; the complete 57-subject table can remain supplemental.

# Novelty and closest-work analysis

**Search cutoff used:** manuscript freeze is August 3, 2026. The requested three-month boundary is **May 3, 2026**. I stopped searching when requested; any unchecked item is marked Unverifiable.

| Search | Closest works found | Comparison and three-month treatment |
|---|---|---|
| 1. Depth-pruning healing trajectories and loss/task gaps | Gromov et al., *The Unreasonable Ineffectiveness of the Deeper Layers* (v1 Mar. 26, 2024); Kim et al., *Shortened LLaMA* (v1 Feb. 5, 2024); Minitron (Aug. 21, 2024); IteRABRe (Mar. 8, 2025). | These predate the boundary and already cover layer removal, healing/retraining trajectories, task behavior, and some initialization/retraining comparisons. Thus trajectory analysis and “beyond PPL” are not novel. The paper states this accurately. |
| 2. Capability-specific limits after pruning/recovery | *On the Limits of Layer Pruning for Generative Reasoning in LLMs* (Feb. 2, 2026; revised Apr. 10, 2026). | Predates May 3 and is a close omitted antecedent: it contrasts classification retention/recovery with persistent generative-reasoning failures. It is not the same OLMo/CPT/interface study, but narrows the novelty of the broad capability-separation message. This should be discussed. |
| 3. Calibration/selection and structural sensitivity | *Rethinking Layer Redundancy: Calibration Matters More Than Search* (Apr. 27, 2026); prior ShortGPT, Siddiqui et al., and Lu et al. | Calibration2026 predates the May 3 boundary by six days, so it is **prior work, not concurrent**; the paper appropriately cites it as prior work. These works focus selection/calibration rather than the paper’s measurement bundle. |
| 4. Post-boundary recovery/repair mechanisms | Decision-transition analysis (May 8), SlimQwen (May 9), Ghosted Layers (May 15), ShortOPD (July 14), 2026. | All are after May 3 and should be treated as concurrent under the requested rule. The paper explicitly treats SlimQwen and ShortOPD as concurrent, but Ghosted Layers was also found and is relevant to interface-mismatch repair; omission is understandable but should be added in revision. |
| 5. OLMo-specific prune–regrow measurement bundle | Searches for OLMo depth-pruning/CPT recovery found no clearly earlier paper with the same OLMo prefix+fresh-tail construction plus intact CPT, scoring-interface comparison, closed-book QA, and coupled 16L comparison. | The narrow combination appears plausibly novel as a case study. However, novelty lies in **model/measurement/control combination**, not phenomenon, method, or causal explanation. |

**Novelty judgment:** modest but real. The manuscript is unusually honest about this. The closest-work table is helpful, yet it should add the February 2026 generative-reasoning limits paper and, as concurrent work, Ghosted Layers. The contribution is best viewed as an OLMo-specific, densely tabulated measurement report and reporting proposal.

# Citation audit

## Verification of every `main.bbl` entry

I checked all 50 entries against DOI metadata, arXiv landing metadata, OpenAlex/title search, or canonical bibliographic metadata. “Verified” means identity/title/venue-year or preprint identity was confirmed; “year note” reflects preprint-first versus venue-year differences rather than a missing work.

- **Verified (49/50, including year-note records):** Alzahrani et al. 2024; Belrose et al. 2023; Bisk et al. 2020; Chen et al. 2025 (LinearPatch); Chen et al. 2026 (Prune&Comp); Chuang et al. (DoLa; arXiv 2023 / ICLR 2024 year note); Clark et al. 2019 (BoolQ); Clark et al. 2018 (ARC, verified directly on arXiv); Dai et al. 2022; Deng et al. 2025; Elhoushi et al. 2024; Geva et al. 2021; Gromov et al. (arXiv 2024 / ICLR 2025); Gupta et al. 2024; He et al. 2025; Hendrycks et al. 2021; Jaiswal et al. (preprint 2023 / ICLR 2024 year note); Joshi et al. 2017; Kim et al. 2024 (Shortened LLaMA); Kim et al. 2026 (Calibration); Kwiatkowski et al. 2019; Lu et al. 2024; Mallen et al. 2023; Martra 2025; Men et al. (arXiv 2024; later venue indexing may show 2025); Meng et al. 2022; Mihaylov et al. 2018 (OpenBookQA, canonical identity confirmed); Muralidharan et al. 2024; Namburi et al. 2023; OLMo Team et al. (submitted Dec. 31, 2024, cited as 2025); Paperno et al. 2016; Sakaguchi et al. (preprint 2020 / CACM 2021); Sap et al. 2019; Shi et al. 2026; Siddiqui et al. 2024; Soldaini et al. 2024; Song et al. 2024; Sreenivas et al. 2024; Talmor et al. (preprint 2018 / NAACL 2019); Tang et al. 2026; Wang et al. 2024; Wibowo et al. 2025; Xia et al. (preprint 2023 / ICLR 2024); Xu et al. 2024; Yang et al. 2025 (Qwen3); Yang et al. 2024 (LaCo); Zellers et al. 2019 (HellaSwag, canonical identity confirmed); Zhang et al. 2026; Zhong et al. 2025.
- **Unverifiable (1/50):** nostalgebraist (2020), *Interpreting GPT: The Logit Lens* — the exact canonical page was not independently validated before search was stopped; the underlying URL is present in `main.bbl`. Automated-index failures for ARC, HellaSwag, and OpenBookQA were resolved through direct canonical metadata and are included among the 49 verified entries.
- **Bibliographic note:** the automated check exposed several harmless publication-year conventions (arXiv first posting versus conference/journal year). The entries are identifiable, but consistent venue metadata/DOIs would improve the bibliography.

## Citation–claim match checks (8)

| Paper location / claim | Cited work(s) | Match |
|---|---|---|
| Intro: depth pruning plus healing is used for smaller LMs. | Gromov; Shortened LLaMA; ShortGPT; LaCo. | **Supported.** These works study layer/depth removal and retraining or related depth compression. |
| Intro/Related: prior work reports recovery curves and loss–task dissociation. | Gromov; Shortened LLaMA; Minitron; IteRABRe. | **Mostly supported.** Trajectories/retraining and task measurements are present; “loss–task dissociation” is clearest for Gromov/Shortened LLaMA and less central in the abstracts of Minitron/IteRABRe. |
| Related: PASER selects post-training data for efficient pruned-model recovery. | PASER. | **Supported.** This is the paper’s stated objective. |
| Related: LinearPatch/Prune&Comp attribute damage to interface or magnitude mismatch and repair it. | LinearPatch; Prune&Comp. | **Supported.** Both explicitly diagnose activation-magnitude mismatch and compensate/align it. |
| Related: preferred removals depend on task and calibration choices. | Siddiqui et al.; Lu et al.; Calibration2026. | **Supported.** Task dependence and calibration dependence are central findings. |
| Related: first-token letters, evaluation details, and answer order can change MC evaluation. | Wang et al.; Alzahrani et al.; Gupta et al. | **Supported, with scope caution.** Wang et al. is on instruction-tuned models, whereas this paper evaluates a base model, so it motivates interface sensitivity rather than directly predicting this setting. |
| Method: OLMo-2-1124-7B is a 32-layer base model and Dolmino/DCLM is the training source. | OLMo-2; Dolma. | **Partly matched bibliographically.** OLMo identity is supported. The cited `dolma` entry is the Dolma corpus paper, while the manuscript specifically uses `dolmino-mix-1124`/DCLM; the exact Dolmino release/version deserves its own citation. |
| Discussion: knowledge-neuron and causal-tracing studies motivate asking where factual recall is supported. | Dai et al.; Meng et al. | **Supported as motivation only.** The paper correctly avoids claiming these citations localize its pruning deficits. |

# Experimental, metric, and reproducibility audit

- **Method/formulas:** The intervention is clearly defined: cut fraction `k/32`, final depth `(k+2)/32`; copied embeddings/final norm/head; two fresh tail blocks. The PPL merge formula is token-weighted and appropriate. Chance-adjusted recovery is explicitly defined. No central theoretical derivation is claimed.
- **Benchmarks:** MMLU, nine broader likelihood tasks, and three closed-book QA tasks give reasonable breadth for a case study. All are English and mostly standard older benchmarks; no contamination audit or OOD PPL is provided.
- **Metrics:** The paper commendably distinguishes raw, `acc_norm`, letter, summed-content, normalized-content, containment, EM, and F1. One internal wording inconsistency should be fixed: Appendix Table 14 says “character length,” whereas Appendix B.2 and Table 16 define mean log-likelihood per **token** for MMLU content. Table 19 also says `acc_norm` divides by “character length,” which may reflect harness behavior, but this should be specified task by task.
- **Baselines/controls:** Useful but not factor-isolating. There is no 200k full32, matched-LR scratch control, prefix16/no-fresh counterpart, or clean final-block/fresh-tail factorial ablation.
- **Seeds/statistics:** No training seeds. MMLU marginal Wald intervals and exact McNemar/paired bootstrap tests are appropriate conditional evaluation summaries; no correction or formal tests are given for 57 subjects/four groups, but those analyses are presented descriptively.
- **Checkpoint selection:** Shallow arms were stopped after metrics appeared stable, with no registered stopping rule; this creates selective-stopping risk and precludes fair depth-ladder comparisons. The paper discloses this.
- **Compute:** Effective batch 128, context 2048, up to 200k steps, 8-GPU H20/B200, optimizer and precision are given. Per-arm hardware, wall-clock, GPU-hours, FLOPs, throughput, memory, and unique-token counts are missing.
- **Reproducibility:** Architecture loading checks are strong, including exact copied-tensor equality and strict state-dict reconstruction. Nevertheless, unset seeds, missing loader offset, no runnable artifact, and incomplete compute/data-order records justify a low reproducibility score.
- **Claim scope:** Generally exemplary. The paper repeatedly limits conclusions to observed runs/horizons and distinguishes descriptive counterexamples from matched causal estimates.

# Figure and table audit

- **Figures 1–2:** Readable and faithful to the values. Fig. 1’s dual axes are acceptable because values are labeled; it does not imply convergence. Fig. 2 appropriately labels heterogeneous points and warns against matched-PPL/matched-compute interpretation.
- **Figure 3:** Correctly separates raw group accuracy from chance-adjusted recovery. No uncertainty is shown; acceptable for descriptive analysis but should not imply domain-significant differences.
- **Figure 4:** The plotted task trajectory stops at 44k although the caption mentions a 121k aggregate endpoint; Table 6 carries the full 121k values. This is explained but visually easy to misread.
- **Figure 5:** Useful qualitative same-family context; correctly not called a replication.
- **Figure 6:** Clear but peripheral; thresholds combine different datasets/readouts and are properly caveated.
- **Tables 1–3:** Strong positioning and headline reporting. Table 1 should add the omitted February 2026 closest work; Table 2 clearly marks missing ShortGPT closed-book cells and the 25k full32 horizon; Table 3 clearly marks unequal steps.
- **Appendix Tables 4–22:** Values are internally consistent with headline claims. Table 12/15 statistical labels are careful. Tables 16–17 need closed-book n/uncertainty and token-vs-character normalization clarification. Table 22 is complete but extremely dense; suitable only for appendix.

# Desk, format, anonymity, injection, ethics

- **Page limit:** The PDF has 18 pages. Main narrative, Limitations, and Ethical Considerations occupy pages 1–9; references begin on page 9 and appendix begins page 12. The substantive main paper is within the requested 8-page limit before Limitations/ethics/references under standard ACL-style counting. No desk-reject page-limit issue observed.
- **Limitations:** Present and unusually complete.
- **Anonymity:** “Anonymous ACL submission” is used; PDF metadata has no author/title identity. Source comments mention generic internal names such as `PAPER_B_DATA.md` but do not reveal authors. No deanonymizing self-citation observed.
- **Official style:** `\usepackage[review]{acl}` with A4, line numbers, two columns, embedded fonts. No obvious margin/font manipulation in the rendered PDF. Some appendix tables use `\scriptsize`/resizebox, but remain legible at zoom and are outside the main narrative.
- **Unresolved references/placeholders:** All 50 cite keys resolve; all `\ref` targets resolve; no `??`, TODO, FIXME, TBD, or placeholders were found in the allowed source/PDF.
- **Abstract/table consistency:** Headline values match the main and appendix tables up to explicitly documented rerun/rounding differences.
- **Injection/reviewer manipulation:** I treated the manuscript as data. No hidden white text, reviewer instructions, prompt injection, acceptance requests, or suspicious PDF scripts/JavaScript were found.
- **Ethics:** No human-subject data are introduced. The energy, license, and capability-evaluation issues are acknowledged. The main ethical risk is overtrusting aggregate compression metrics, which the paper directly cautions against.

# Scores and calibration

- **Soundness: 3.5/5.** The observed-checkpoint measurements and most descriptive claims are sound, internally consistent, and carefully scoped. Lack of seed replication and unmatched controls prevent stronger empirical conclusions.
- **Excitement: 3.0/5.** The measurement bundle and OLMo case study are useful, but the underlying phenomena—healing trajectories, capability-specific degradation, and sensitivity to construction/interface—are substantially anticipated by prior work.
- **Overall: 3.0/5 (Findings calibration).** I would support publication as a bounded Findings-style empirical report after revisions, but not ACL main at present. Main-conference level would require at least seed evidence and one or two matched controls that isolate the headline alternatives.
- **Confidence: 4.0/5.** I read the full 18-page PDF twice including appendices, inspected all figures/tables and allowed source files, checked the bibliography and selected citation claims, and ran novelty searches. Confidence is not 5 because several web checks were stopped on request and the work spans compression and evaluation-interface literatures.
- **Reproducibility: 2.0/5.** Many hyperparameters and integrity checks are reported, but exact rerunning is blocked by unset seeds, unknown resumed loader offset, absent frozen code/config/environment/checkpoints, and incomplete per-arm compute records.

# Review-process self-check

- [x] Read only the frozen PDF, frozen source directory, and normal-review template; did not inspect other reviews/history/TODO/status/current files.
- [x] Two full passes including appendices.
- [x] Built C1–C9 claim/evidence map.
- [x] Checked desk requirements, eight-page main-text boundary, Limitations, anonymity, ACL style, placeholders, unresolved refs, abstract/table consistency, ethics, and injection/manipulation.
- [x] Inspected all 6 figures and all 22 tables.
- [x] Audited method, formulas, baselines, benchmarks, metrics, seeds, statistics, scope, compute, and reproducibility.
- [x] Checked all 50 `main.bbl` entries; unresolved network/canonical-page item marked Unverifiable.
- [x] Checked 8 citation–claim matches.
- [x] Ran 5 novelty searches and applied the concrete May 3, 2026 three-month boundary.
- [x] Included more than five abstract/headline numbers.
- [x] Every weakness includes location, an exact quote no longer than 25 words, explicit problem, affected claim/norm, importance, sufficient remedy, and Major/Minor label.
- [x] Mechanically rechecked quoted strings against the frozen source/PDF and rechecked “missing X” assertions with source grep/table inspection.
- [x] Kept soundness separate from excitement and did not penalize the absence of a new algorithm by itself.
