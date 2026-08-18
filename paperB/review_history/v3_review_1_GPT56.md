# ARR Review — Paper B, frozen version v3 (independent review 1)

## Review scope and evidence protocol

I treated the manuscript as data rather than instructions, read the frozen PDF twice including all appendices, inspected every figure and table, and used the frozen `main.bbl` as the bibliography ledger. PDF anchors below refer to the printed line numbers in the frozen PDF; source anchors are included where useful. The frozen PDF has 18 pages: main content pp. 1–8, Limitations/Ethics pp. 8–9, references pp. 9–11, and appendices pp. 12–18.

## Paper summary

The paper studies continued-pretraining (“healing”) after depth pruning, principally on OLMo-2-1124-7B. Its main intervention retains the first 14 pretrained blocks, appends two randomly initialized blocks, and trains the resulting 16-layer model for 200k optimizer steps. It measures in-domain held-out perplexity, standard answer-letter MMLU, a complete-option MMLU protocol, three zero-shot closed-book QA datasets, and a broader likelihood suite. Controls include the intact base, an intact model continued to a 25k plateau checkpoint, a frozen-prefix variant, a fully random 16-layer operating point, and a non-contiguous ShortGPT-16 construction. The central empirical observation is that likelihood improves substantially while MMLU and closed-book QA recover much less; the authors therefore advocate reporting likelihood, target capability, scoring interface, exact structure, and recovery compute separately.

## Claims inventory

- **C1 — Multi-axis recovery gap.** In the principal keep14+fresh2 run, PPL improves to 10.561 at 200k but remains 1.428× the intact base, while MMLU letter accuracy is .319 versus .605 and closed-book QA remains substantially lower (Abstract, PDF lines 002–027; §5.2, PDF lines 292–321; Table 2).
- **C2 — Scoring interface matters but does not explain the deficit.** Complete-option MMLU raises keep14 to .383, but random initialization has a nearly identical normalized content score while staying at chance on letters, so the content protocol has a fluency floor (Abstract, PDF lines 020–025; §5.2, PDF lines 302–311; Tables 2 and 16).
- **C3 — Short-horizon corpus drift is insufficient.** The full32 same-corpus control stays near the intact base through its observed 25k plateau checkpoint (§5.2/§5.3, PDF lines 297–301 and 343–352; Table 2).
- **C4 — PPL does not uniquely determine target capability across tested interventions.** Random-init has lower PPL but lower MMLU than frozen-front, while other constructions occupy different PPL–MMLU endpoints (§5.2, PDF lines 322–330; Figure 3).
- **C5 — Recovery trajectories differ with inherited prefix depth over the observed budget.** keep8 shows falling PPL but no detectable late MMLU change, whereas keep14 gains 1.68 MMLU points from 128k to 200k (§6.2, PDF lines 385–398; Figure 5; Appendix Tables 6 and 13).
- **C6 — Nominal depth does not determine a unique endpoint.** ShortGPT-16 is much stronger than keep14 at the same total depth and 200k target budget, but four coupled structural differences preclude single-factor attribution (§6.3, PDF lines 399–418; Table 3).
- **C7 — Limited qualitative generality.** An OLMo-2-1B trajectory and a more-compressed Qwen3-8B endpoint show a similar directional gap, but are explicitly not matched replications (§6.4, PDF lines 419–435; Appendix Figure 6 and Table 9).
- **C8 — Methodological recommendation.** Prune-then-heal work should report likelihood, target capability, scoring/generation interface, exact structure, and healing budget as separate axes (§7, PDF lines 498–510; §8, PDF lines 511–527).
- **C9 — Layer-wise readout depths are descriptive only.** Semantic probes, MMLU logit-lens readout, and next-token agreement become readable at different depths, but do not localize causal storage (§4, PDF lines 273–280; Appendix C, PDF lines 976–1001; Figure 7/Table 20).

## Desk-rejection / compliance checklist

- **Length and order:** appears compliant as a long paper: eight pages of content through the conclusion, followed by Limitations/Ethics, references, then appendix. The appendix is double-column.
- **Required Limitations section:** present before references (PDF lines 528–575).
- **Anonymity:** title page is anonymous; I found no obvious author-identifying text or repository link in the frozen source/PDF.
- **Prompt-injection/manipulative text:** none found in the frozen source/PDF.
- **Citations:** all 39 citation keys used by the source occur in frozen `main.bbl`; no missing or unused `main.bbl` entries were found.
- **Responsible NLP checklist:** not present in the PDF, but ARR normally collects it in the submission form; therefore not assessable from the allowed artifacts.
- **Prior-version/resubmission disclosure, concurrent-submission overlap, review-duty registration:** not assessable from the allowed artifacts.

**Desk recommendation:** no desk-rejection trigger is visible in the frozen source/PDF.

## Summary of strengths

1. **The principal observation is clearly bounded and triangulated across genuinely different evaluation interfaces.** The authors do not rely only on answer-letter MMLU: they add complete-option scoring and PopQA/TriviaQA/NQ-open generation, and explicitly avoid declaring either MMLU interface a gold standard. This materially supports C1/C2. Evidence: §5.2, Table 2, and Appendix Tables 16–17 (PDF lines 292–321, 951–959; PDF p. 17). Short anchor: “**The separation is not specific to multiple choice.**” (PDF lines 312–318).

2. **The paper is unusually careful about causal scope.** It repeatedly distinguishes operating-point comparisons from clean ablations, notes the unequal learning rate for random-init, the changed trainable set for frozen-front, and the four coupled differences between keep14 and ShortGPT. Evidence: §3.3 (PDF lines 212–225), §6.3 (PDF lines 399–418), and Limitations (PDF lines 540–549). Short anchor: “**does not isolate its coupled structural differences**” (PDF lines 223–225).

3. **The evaluation bookkeeping and internal integrity checks are strong.** The appendix gives sample counts and chance floors, exact PPL aggregation, strict architecture reconstruction, zero-NaN/full-count checks, and independent recomputation of the reported keep14 cells. Evidence: Appendix B, Tables 15–19 (PDF lines 932–975; PDF pp. 16–17). Short anchor: “**reproduces every PPL and downstream cell to 10−9**” (PDF lines 964–967).

4. **The manuscript is transparent about important limitations rather than hiding them.** It states that central runs are single seeds, depth/full32 comparisons are not long-budget matched, the random/frozen controls are not causal ablations, PPL is in-domain, and no contamination audit was performed. Evidence: Limitations (PDF lines 528–575). This transparency makes the supported observational claims easier to trust.

5. **The figures/tables are generally legible, internally consistent, and appropriately qualified.** I checked Figures 1–7 and Tables 1–22. Figure 3 explicitly says it is not evidence of global uncorrelatedness; Figure 5 distinguishes the plotted early 44k trajectory from a separately cited later paired comparison, though that later statistic is under-documented (W6); Figure 7 labels probes as non-causal. The central numeric ratios and chance-adjusted recoveries are arithmetically consistent with the tables.

## Summary of weaknesses

### W1. [Major] The central trajectory-level evidence has no training-seed replication, and the exact run is not command-line reproducible.

- **Location:** Limitations, PDF lines 531–534; Appendix B.1, PDF lines 921–931; source `sections/06_limitations.tex:3–9` and `sections/08_appendix.tex:126–132`.
- **Short excerpts:** “**same-shape controls are single training runs**”; “**Runs do not set an explicit random seed**.”
- **Weakens:** C1, C5, and C6 as claims about recovery dynamics rather than one realized trajectory; ARR soundness and reproducibility norms. Item-level McNemar/bootstrap intervals quantify evaluation uncertainty conditional on fixed checkpoints, not optimization, fresh-tail initialization, data-order, or block-selection variability.
- **Remedy:** rerun at least keep14 and one key control/alternative construction with 3 independent seeds (ideally keep14, random-init/frozen-front, and ShortGPT), report mean/dispersion for the trajectory and endpoint deltas, set and publish explicit seeds, and preserve/release exact data-order and resume state. If compute prevents this, narrow the claims throughout to “in this run” and avoid treating small late gains as stable properties.

### W2. [Major] The main controls do not identify the mechanisms emphasized in the interpretation, and the ideal matched counterfactuals are absent.

- **Location:** §3.3, PDF lines 212–225; §5.3, PDF lines 331–352; §6.3, PDF lines 399–418; Limitations, PDF lines 540–549.
- **Short excerpt:** “**It shows only that the two coupled constructions yield different endpoints.**”
- **Weakens:** C3 and especially C6, plus the contribution claim that controls “separate interpretations.” The full32 branch ends at 25k, not 200k; random-init uses a different LR; frozen-front changes the trainable module set; ShortGPT differs in inherited count, selected indices, last-block retention, and fresh-tail presence. Thus corpus drift at the long horizon, initialization, adaptation, and structural policy remain only partially separated.
- **Remedy:** add (i) a 200k/token-matched full32 control; (ii) an LR-matched random-init run; (iii) trainable-parameter-matched freeze controls; and (iv) a factorial structural ablation at 16 layers: contiguous 16 inherited/no fresh tail, 14 inherited+2 fresh with and without block 31, non-contiguous 14+2 fresh, etc. These would make the interpretation materially stronger.

### W3. [Major] Novelty positioning omits a close prior trajectory/recovery study and understates how much the “beyond perplexity” conclusion is already established.

- **Location:** Related Work, PDF lines 115–172; especially Table 1 and the claim “**Our narrower increment is to treat the recovery path itself as the analysis object**” (PDF lines 131–137).
- **Weakens:** novelty/excitement for C8 and the first contribution. IteRABRe (Wibowo et al., arXiv:2503.06291, March 8, 2025) explicitly plots iterative pruning/recovery trajectories, separates PPL/language/reasoning/knowledge behavior, and reports that recovery helps many tasks but MMLU remains weak. Gromov et al. and Shortened LLaMA also show training progress or recovery curves. Separately, Jaiswal et al., Namburi et al., and the cited 2026 calibration paper already establish that perplexity can misalign with knowledge/downstream behavior.
- **Remedy:** cite and directly compare with IteRABRe and other trajectory/repair studies; revise Table 1’s “Trajectory?” characterization; define the narrower novelty as the particular combination of same-corpus full-model control, MMLU interface decomposition, closed-book QA, and construction comparison. A concise side-by-side table of controls, trajectories, metrics, and model families would make the incremental contribution credible.

### W4. [Major] Generality is too limited for the broad reporting prescription, and the benchmark package is vulnerable to contamination/domain confounds.

- **Location:** §6.4, PDF lines 419–435; Limitations, PDF lines 563–575; Appendix Table 9.
- **Short excerpts:** “**directional evidence, not a matched replication**”; “**We do not report out-of-domain PPL or a benchmark-contamination audit**.”
- **Weakens:** C7 and the field-level scope of C8. The central evidence is one English OLMo-2-7B recipe and one training realization; the 1B run is more compressed and same-family; Qwen differs in architecture, corpus, and compression fraction and lacks the interface/closed-book controls. Because healing data are web-scale and PPL validation is in-domain, benchmark behavior can reflect exposure/contamination and domain fit rather than recovered knowledge.
- **Remedy:** add one matched second-family replication with the same retained fraction, structure, budget, MMLU interfaces, and closed-book suite; report at least one out-of-domain PPL corpus and a benchmark-contamination/overlap analysis. Until then, state the reporting protocol as a proposal motivated by these case studies, not as an empirically generalized requirement.

### W5. [Minor] The “all nine arms” interface claim is imprecise, and the complete-option protocol is an exploratory diagnostic rather than a clean interface ablation.

- **Location:** §5.3, PDF lines 353–359; Appendix Table 16, PDF p. 17; source `sections/04_experiments.tex:84–89`.
- **Short excerpt:** “**Across all nine evaluated arms**.”
- **Weakens:** C2’s quantitative scope and clarity. Table 16 lists nine rows, but this is eight model arms plus three different keep14 checkpoints; shallower arms are at unequal checkpoint budgets. Moreover, letter versus content scoring changes prompt, candidate string, tokenization, and normalization simultaneously, as the paper itself concedes.
- **Remedy:** say “across nine evaluated checkpoints/operating points,” report within-arm paired deltas with uncertainty, and add a cleaner answer-symbol mapping control (e.g., permuted labels, direct answer-text generation, or a shared prompt with only the readout varied).

### W6. [Minor] The keep8 late-comparison statistic is not documented to the same standard as the keep14 statistic.

- **Location:** §6.2, PDF lines 387–398; Figure 5 caption; Appendix Table 6.
- **Short excerpt:** “**a paired 45k–121k comparison finds no significant MMLU change**.”
- **Weakens:** C5 and reproducibility. The available table reports 44k and 121k aggregate values (.2463 and .2535), while the text cites a separate 45k–121k paired change of +0.24 points and CI [−0.51, 0.98] without a corresponding table, exact checkpoint provenance, test, p-value, or retained per-item ledger.
- **Remedy:** add the 45k checkpoint and paired-test details to the appendix, explain why it differs from the tabulated 44k checkpoint, and provide the same McNemar/bootstrap specification used for keep14.

### W7. [Minor] Artifact availability is insufficiently specified for independent reproduction.

- **Location:** Appendix B.1–B.2, PDF lines 885–975; frozen source tree contains paper source/figures but no training or evaluation code.
- **Short excerpt:** “**exact fresh initialization and initial shuffle are not reproducible from the command line alone**.”
- **Weakens:** reproducibility. Hyperparameters and evaluation descriptions are detailed, but exact data-array construction/order, code versions, launch commands, plateau preregistration criterion, per-arm hardware mapping, checkpoints, and evaluation scripts are not supplied in the frozen artifacts.
- **Remedy:** provide an anonymous artifact with code, environment lockfile/commit hashes, exact configs and commands, data indices/order, plateau rule, per-example predictions, and the evaluated checkpoints; explicitly state what will be released.

## Citation verification and citation–claim matching

### Bibliographic authenticity

Using frozen `main.bbl` as the ledger, I verified that all 39 cited keys have corresponding bibliography entries and spot-checked the entries against arXiv, Crossref, ACL Anthology/DOI metadata, or the original project page. I found **no fabricated citation**. The main entries relevant to the paper’s positioning—including Gromov et al. (arXiv:2403.17887), Shortened LLaMA (arXiv:2402.02834), ShortGPT (arXiv:2403.03853 / Findings ACL 2025), SLEB (arXiv:2402.09025 / ICML 2024), Sheared LLaMA (arXiv:2310.06694 / ICLR 2024), BlockPruner (arXiv:2406.10594 / Findings ACL 2025), DRPruning (arXiv:2411.14055 / ACL 2025), OLMo 2 (arXiv:2501.00656), Qwen3 (arXiv:2505.09388), and the compression-evaluation papers are real.

Minor metadata caveat: some entries cite an arXiv version while naming a later venue/year in the rendered reference (e.g., Gromov; ShortGPT). This is not fabrication, but final bibliography cleanup should prefer the canonical proceedings record where available.

### Citation–claim checks (8 sampled, load-bearing matches)

1. **Depth pruning + healing is an established compression route** (§1, PDF lines 039–043): supported by Gromov, Shortened LLaMA, ShortGPT, and LaCo. **Match: good.**
2. **Compression metrics can miss parametric knowledge/downstream/safety changes** (§1, PDF lines 049–056): supported by Namburi et al., Jaiswal et al., and Xu et al. **Match: good.**
3. **ShortGPT/SLEB rank and remove blocks; LaCo merges layers; BlockPruner works at finer block granularity** (§2, PDF lines 120–125): supported by the cited methods. **Match: good.**
4. **Preferred removals depend on task/calibration choices** (§2, PDF lines 126–130): supported by Siddiqui et al., Lu et al., and Kim et al. 2026. **Match: good.**
5. **Gromov shows post-healing autoregressive loss and downstream tasks can behave differently** (§2, PDF lines 131–137): the paper explicitly contrasts QA benchmarks with autoregressive loss before/after healing. **Match: good, but the current paper should more clearly distinguish its trajectory contribution.**
6. **Shortened LLaMA compares CPT and LoRA after depth pruning** (§2, PDF lines 138–140): directly supported. **Match: good.**
7. **Readout/knowledge-mechanism citations** (§2, PDF lines 161–172): logit/tuned lens, FFN memory, knowledge neurons, causal tracing, DoLa, and LayerSkip are used accurately as motivation; the manuscript correctly avoids causal overclaiming. **Match: good.**
8. **Dataset citations and task descriptions** (§3.4, PDF lines 230–248): MMLU, PopQA, TriviaQA, NQ, HellaSwag, ARC, PIQA, WinoGrande, LAMBADA, BoolQ, OpenBookQA, CommonsenseQA, and SocialIQA correspond to the cited datasets. **Match: good.**

The main citation problem is therefore **omission/positioning**, not falsity.

## Novelty search

Search date: **August 3, 2026**. I searched arXiv/OpenAlex/Crossref/ACL metadata. No discovered paper falls within three months of the frozen paper date, so the three-month protection rule does not remove any of the close works below from novelty consideration.

### Q1. Has prior work already shown that post-pruning healing can improve PPL differently from downstream/knowledge tasks?

- **Gromov et al., “The Unreasonable Ineffectiveness of the Deeper Layers”** (March 26, 2024) compares healed autoregressive loss with MMLU/BoolQ and explicitly asks why healing affects loss and QA differently.
- **Shortened LLaMA** (February 5, 2024) compares CPT and LoRA, reports training progress/learning curves, and evaluates PPL plus downstream tasks.
- **IteRABRe** (March 8, 2025) plots pruning/recovery phases and reports that recovery helps language/reasoning more than MMLU/knowledge.

**Assessment:** C1’s phenomenon and trajectory framing are not wholly new. The paper’s novelty is the richer control/evaluation package in one OLMo case study.

### Q2. Has prior work already argued that perplexity is an inadequate certificate for compressed-model capability?

- **Jaiswal et al., “Compressing LLMs: The Truth is Rarely Pure and Never Simple”** (October 2, 2023) explicitly introduces a knowledge-intensive benchmark because perplexity misses capability changes.
- **Namburi et al., “The Cost of Compression”** (December 1, 2023) focuses on parametric knowledge beyond general metrics.
- **Kim et al., “Rethinking Layer Redundancy”** (April 27, 2026) reports perplexity–downstream misalignment under depth pruning.

**Assessment:** the broad “perplexity is not sufficient” conclusion is established. C8 is best viewed as a concrete reporting checklist and controlled case study, not a new conceptual discovery.

### Q3. Has prior work studied iterative/trajectory-level recovery after block pruning?

- **IteRABRe** is the closest omitted work: iterative layer removal and recovery, per-iteration curves, task-family analysis, and explicit weak MMLU recovery.
- **Shortened LLaMA** includes CPT learning curves at several compression ratios.
- Gromov includes before/after healing across pruning fractions and seed/LoRA-rank ablations.

**Assessment:** Table 1’s positioning is incomplete; the recovery path is not unique to this study.

### Q4. Have recent works offered alternative explanations/repairs for pruning collapse that bear on the ShortGPT comparison?

- **LinearPatch** (May 30, 2025) attributes much pruning degradation to activation-magnitude mismatch at the pruning interface.
- **Prune&Comp** (July 24, 2025) similarly repairs magnitude gaps and improves PPL/QA.
- **Shi et al., “Understanding Performance Collapse…”** (May 8, 2026) studies layer-wise decision transitions and bounded recovery.

**Assessment:** these do not invalidate C6, but they show that structural-interface repair and transition dynamics are important nearby explanations. The current paper should engage them, especially when interpreting prefix+fresh-tail damage.

### Q5. What is the nearest overall work?

**Nearest overall:** IteRABRe for recovery trajectories and task-dependent recovery; Gromov for healed loss-vs-QA dissociation; Jaiswal/Namburi for beyond-perplexity evaluation. The present paper is distinguishable by its same-corpus intact control, answer-letter/content decomposition, closed-book QA, random/frozen operating points, and OLMo-2-focused long trajectory. This is a meaningful but incremental empirical contribution.

## Per-claim technical and experimental audit

### C1: Multi-axis recovery gap

- **Technical support:** strong for the observed checkpoints; Table 2 and Appendix Tables 10/13/17 agree.
- **Ideal experiment:** multi-seed keep14 trajectories plus a matched 200k full32 run and out-of-domain PPL.
- **Baselines:** intact base, full32@25k, random, frozen, ShortGPT are useful; long-horizon full32 is missing.
- **Benchmarks/statistics:** MMLU has item-level uncertainty; closed-book tasks lack CIs/significance and prompt sensitivity tests.
- **Reproducibility:** detailed optimizer/eval description, but no seed/code/data-order artifact.
- **Verdict:** supported as an observational single-run case study; not yet a stable estimate of a general recovery law.

### C2: Interface sensitivity and fluency floor

- **Technical support:** Table 16 strongly shows letter/content differences and the random-init content floor.
- **Ideal experiment:** controlled answer-label permutation/direct text generation under a shared prompt, with paired uncertainty.
- **Baseline adequacy:** random-init is informative, though LR differs and content scoring changes multiple factors.
- **Verdict:** directionally supported; “answer-symbol/readout component” remains a hypothesis, not isolated causally.

### C3: Corpus drift insufficient

- **Technical support:** full32@25k stays near base, so short-horizon drift is unlikely to explain the large keep14 gap.
- **Ideal experiment:** full32 at the same 200k steps/tokens/FLOPs and repeated seed.
- **Verdict:** supported only for **short-horizon** drift, as the paper usually states; cannot exclude long-horizon corpus effects.

### C4: PPL does not uniquely determine MMLU

- **Technical support:** the random-vs-frozen rank reversal is a valid counterexample within tested interventions.
- **Ideal experiment:** larger matched intervention grid and correlation analysis with uncertainty.
- **Verdict:** supported in the finite tested set; the manuscript correctly avoids a global independence claim.

### C5: Prefix depth interacts with observed healing

- **Technical support:** keep8 and keep14 trajectories differ, but they are different arms, endpoints, and likely seeds; the 45k statistic is under-documented.
- **Ideal experiment:** common-step, common-token trajectories for keep8/10/12/14 with repeated seeds and identical evaluation checkpoints.
- **Verdict:** suggestive, not sufficient for a robust depth-by-recovery interaction claim.

### C6: Nominal depth does not determine a unique endpoint

- **Technical support:** logically supported by two 16-layer constructions with different endpoints.
- **Ideal experiment:** matched factorial structural ablations.
- **Verdict:** the weak existential claim is supported; any mechanism attribution is not, and the authors mostly respect this boundary.

### C7: Generality

- **Technical support:** 1B and Qwen are directional only.
- **Ideal experiment:** matched second-family replication.
- **Verdict:** insufficient for generality beyond a case-study claim; paper appropriately labels this limited, but title/conclusion/recommendation still read broadly.

### C8: Reporting protocol

- **Technical support:** sensible and useful, but largely synthesizes established lessons plus this case study.
- **Ideal validation:** survey/benchmark multiple pruning methods, model families, data mixtures, and recovery budgets to show which checklist elements change conclusions.
- **Verdict:** useful recommendation, incremental novelty.

### C9: Readout-depth ordering

- **Technical support:** definitions and complete trajectories are provided; caveats are strong.
- **Ideal experiment:** probe-training details/splits/seeds, causal interventions, and held-out probe uncertainty.
- **Verdict:** acceptable as descriptive supplementary context, not a mechanism result.

## Figure and table audit

- **Figure 1:** visually clear and numerically consistent. Probe markers are explicitly labeled non-causal.
- **Figure 2:** correctly shows a small but nonzero late keep14 gain; three checkpoints are too few for a strong curve-shape claim.
- **Figure 3:** rank-reversal counterexample is useful; mixed checkpoint budgets and the connecting prefix line require the caption caveat, which is present.
- **Figure 4:** group aggregation is transparent; no uncertainty bars, so domain differences should remain descriptive.
- **Figure 5:** visually clear, but the plotted early trajectory and the separately cited 45k–121k paired statistic are not fully reconciled.
- **Figure 6:** appropriately labeled qualitative; not an independent replication.
- **Figure 7:** clear and heavily caveated; semantic-probe training details are too sparse for mechanistic weight.
- **Tables 1–3:** central positioning/results are readable. Table 1 omits IteRABRe and overstates distinctiveness. Table 3 clearly marks unmatched checkpoints.
- **Tables 4–22:** extensive and mostly internally consistent. Table 12 uses Wald intervals; acceptable at n=14,042, though Wilson intervals would be preferable near chance. Table 15 is well specified. Table 16’s “nine arms” wording should be “nine checkpoints/operating points.” Table 22 is complete but very small; machine-readable release would help.
- **No obvious numerical contradiction** was found among the main tables, captions, and appendix beyond expected rounding/rerun differences that the paper itself documents.

## Comments / suggestions / typos

1. Replace “all nine evaluated arms” with “nine evaluated checkpoints/operating points.”
2. Clarify “45k” versus the tabulated “44k” keep8 checkpoint and include the paired-test ledger.
3. In the bibliography, prefer canonical venue entries where available (e.g., ShortGPT Findings ACL 2025) rather than mixing arXiv links with venue-year text.
4. The distinction between DCLM/dolmino-mix-1124 and the citation to the Dolma paper should be stated more explicitly; cite the exact Dolmino/DCLM release artifact if a canonical citation exists.
5. Add prompt text and answer-normalization rules for PopQA/TriviaQA/NQ-open, or point to an artifact containing them; “Question:/Answer:” alone is not enough for exact replication.
6. Report confidence intervals for closed-book QA and broad-group differences, not only MMLU.
7. State the plateau stopping rule operationally (metric, window, threshold, preregistration timestamp/protocol) rather than only calling it “pre-registered.”

## Limitations and societal impact

The limitations discussion is unusually candid and covers single-run uncertainty, unmatched controls, coupled constructions, interface dependence, English/model/corpus scope, in-domain PPL, and missing contamination audit. The ethics section appropriately notes that compression does not remove hallucination, bias, or unsafe-completion risks and warns against deployment decisions based on PPL alone. I would add one explicit point: capability degradation may be uneven across domains or user groups, so compression audits should include safety/fairness regressions relevant to the intended deployment, not only aggregate capability preservation.

## Ethical concerns

No specific ethics violation is apparent from the frozen artifacts. The work uses released models/corpora/benchmarks and no new human-subject data. The primary concern is downstream misuse through overtrust in compressed models, which the paper itself acknowledges.

**Needs ethics review:** No.

## ARR ratings

- **Soundness: 3.0 / 5 (Acceptable).** The main observational result is well documented and cautiously interpreted, but central runs are single-seed, key controls are unmatched, and one trajectory statistic is under-documented. I considered 3.5, but ARR guidance says to take the lower score when uncertain.
- **Excitement: 2.5 / 5 (between Potentially Interesting and Interesting).** The control package and reporting checklist are useful, but the core PPL/capability dissociation and recovery-trajectory framing have close prior art, including an omitted trajectory study.
- **Overall assessment: 2.5 / 5 (Borderline Findings).** The paper is a careful and potentially useful measurement case study, but I would not yet recommend Conference acceptance. Findings is plausible after stronger novelty positioning and clearer narrowing; a stronger revision would add seed replication and matched controls.
- **Reviewer confidence: 4 / 5.** I read the full paper and appendix twice, checked every figure/table, verified the bibliography ledger and sampled claim matches, and searched close work. Some implementation facts cannot be independently checked without code/checkpoints.
- **Reproducibility: 3 / 5.** A knowledgeable reader could recreate an approximate pipeline with difficulty, but exact reproduction is blocked by unset seeds, missing data-loader offset/order, absent code/configs/checkpoints in the frozen artifacts, and incomplete release commitments.
- **Datasets: 1 / 5 (No usable datasets submitted).** No new dataset or dataset artifact is claimed.
- **Software: 1 / 5 (No usable software released).** No usable software release is stated in the frozen paper.
- **Knowledge/educated guess at author identity:** 1 / 5 — I do not have an educated guess from the submission.

## Final recommendation

**Borderline Findings (2.5).** The paper’s strongest value is not a new compression method or a new discovery that perplexity can mislead, but a careful OLMo-2 case study combining trajectory checkpoints, interface diagnostics, closed-book QA, and several interpretation controls. That contribution is useful. The present version, however, overstates its trajectory novelty, lacks training-seed evidence and matched long-horizon/structural counterfactuals, and has only directional cross-model generality. A revision that narrows the novelty claim, cites the closest trajectory literature, and strengthens replication/control evidence would be substantially more compelling.
