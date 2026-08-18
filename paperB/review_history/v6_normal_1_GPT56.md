review_mode: normal
soundness: 3.5
excitement: 3.0
overall: 3.0
confidence: 4.0
reproducibility: 2.5

## Summary

This paper is a careful measurement case study of post-pruning continued pretraining. Its principal OLMo-2-7B construction retains blocks 0--13, adds two fresh blocks, and is trained for 200k steps. Along its three late checkpoints, in-domain held-out PPL decreases (10.826 to 10.561) and answer-letter MMLU improves modestly (.3012 to .3191), yet the final checkpoint remains far below the intact base (.605), as do three zero-shot closed-book QA scores. The authors frame the supported conclusion narrowly: on these literal paths, PPL improvement alone is insufficient evidence of recovery to intact-base performance on the measured targets.

The paper also reports a 25k intact-CPT point, same-shape frozen and random operating points, a non-contiguous ShortGPT-16 point, two MMLU scoring interfaces, and detailed appendix tables. It is explicitly not a new pruning algorithm, causal layer-localization study, or claim of a universal recovery law.

## Strengths

1. **The central claim is appropriately narrow and directly supported by the principal trajectory.** The keep14 path improves in PPL by 0.265 from 128k to 200k while final answer-letter MMLU is still .319 versus .605 for the intact base; PopQA/TriviaQA/NQ-open are .142/.294/.060 versus .257/.636/.205. This is a meaningful, practically relevant reminder that an in-domain optimization proxy is not a target-capability certificate after structural intervention.

2. **The manuscript distinguishes measurement evidence from causal attribution unusually well.** Table 2 and Sections 3, 5.3, and 6 repeatedly label random, frozen, and ShortGPT comparisons as operating points rather than clean ablations. In particular, the authors disclose that random initialization also changes lexical modules and learning rate, frozen changes the trainable set, and ShortGPT simultaneously changes inherited-block count, selection/contiguity, final-block retention, and fresh-tail use.

3. **The interface and null-baseline analysis is useful.** On the common 14,042-item MMLU rerun, complete-option normalized scoring raises keep14 to .383, but random initialization reaches .360 while remaining .247 under answer-letter scoring. This makes the conclusion about interface sensitivity concrete rather than treating one MMLU protocol as definitive. The paper correctly says the protocols differ in several factors at once and does not attribute the effect to answer symbols alone.

4. **The evidence and reporting are unusually transparent for a retrospective measurement study.** The appendix gives training hyperparameters, sample counts, prompts/normalization, checkpoint grids, item-level MMLU procedures, paired bootstrap intervals, exact McNemar testing, artifact contents, and the important missing provenance. It clearly separates the 1.68-point 128k-to-200k MMLU paired-item interval ([1.08, 2.29]) from training-run uncertainty.

5. **Presentation is strong.** Figure 1 succinctly communicates the main trajectory and the limits of the control points; Table 2 makes construction, budget, metric, and missing-evaluation differences easy to inspect. The appendix is extensive and readable, including the 1B qualitative trace and full MMLU subject table.

## Weaknesses

### Major weakness 1: Single-run evidence cannot establish the stability of the headline dissociation.

- **Issue:** The load-bearing keep14 result, all same-shape operating points, and ShortGPT endpoint each come from one historical training realization. Training seeds were not explicitly set, and the resumed keep14 run lacks the within-epoch loader offset. Item-level CIs and McNemar tests quantify only evaluation-item variation conditional on fixed checkpoints.
- **Why it matters:** The central observation is credible for these realized checkpoints, but the paper's practical reporting recommendation would be considerably stronger if the PPL--target gap were shown to survive variation in data order, initialization, and optimization. The current evidence cannot tell whether the magnitude of the gap or the modest late MMLU gain is typical.
- **Concrete remedy:** At minimum, rerun the principal keep14 construction with several recorded seeds and a fixed data-order protocol, report mean/dispersion for PPL and all headline targets at predeclared checkpoints, and test whether the intact-base deficit persists across runs. If a full rerun is infeasible, elevate this limitation earlier in the abstract/conclusion and make the claim consistently checkpoint-realization-specific.
- **Expected impact:** This would materially increase soundness and reproducibility; without it, the paper is best read as a well-documented case study rather than robust empirical characterization.

### Major weakness 2: The study does not provide a long-horizon intact control, so it cannot separate compression-specific degradation from long-horizon CPT effects at the endpoint.

- **Issue:** full32 is observed only at 25k (6.6B nominal presentations), whereas keep14 reaches 200k (52.4B). The 25k full32 result usefully argues against an immediate catastrophic corpus-shift account, but it is not a matched 200k counterfactual.
- **Why it matters:** The main implication itself does not require a full32-200k branch, but claims about the size and source of the final intact-base gap, and the interpretation of recovery under continued pretraining, remain undercontrolled at the headline horizon.
- **Concrete remedy:** Continue intact full32 on the same corpus/evaluation schedule through 200k, or provide a clearly documented matched surrogate with identical data order and optimizer schedule. Plot both paths at shared checkpoints and report PPL plus the same MMLU and closed-book metrics.
- **Expected impact:** A matched branch would substantially sharpen the causal interpretation; if it narrows the gap, the paper should revise its discussion of compression-specific effects, while if it does not, it would strengthen the case study.

### Major weakness 3: Reproducibility is partial despite an unusually candid artifact description.

- **Issue:** The artifact snapshots evaluators, sanitized manifests, aggregate results, six content-MMLU per-item files, and paired outputs, but not model weights, original training seed, historical loader offset, public evaluator commit ancestry, closed-book per-item predictions, or GPU/wall-time records. The manuscript states exact reproduction of keep14 is blocked by the lost loader offset.
- **Why it matters:** A reader can audit some headline MMLU computations but cannot fully recreate the principal training path or independently recompute paired uncertainty for the three closed-book findings.
- **Concrete remedy:** Release, where licenses permit, checkpoint identifiers/weights or deterministic reconstruction scripts; provide the complete retained training-state metadata and data-order reconstruction; and add closed-book per-item predictions (or sufficient hashed/aligned correctness records) for the reported arms. If historical reconstruction is impossible, provide a new fully reproducible small-scale rerun with recorded seeds, offsets, environment lockfile, and compute log.
- **Expected impact:** This would turn the paper from auditable retrospective reporting into a reproducible measurement contribution. The current candid disclosure mitigates, but does not remove, this limitation.

### Minor weakness 4: The scope checks are informative but add limited independent evidence.

- **Issue:** The 1B trajectory and Qwen3 endpoint both point in the same direction, but the authors correctly note changes in model scale, retained fraction, architecture, corpus, and available evaluations. The Qwen arm is only an endpoint and lacks the content-MMLU and closed-book diagnostics.
- **Why it matters:** Readers may visually treat these additions as cross-family replication although they do not isolate a common intervention/evaluation protocol.
- **Concrete remedy:** Either move them to a compact “context only” appendix paragraph/figure or add one preregistered, matched cross-family experiment using the same PPL split definition, checkpoints, MMLU interfaces, and closed-book suite.
- **Expected impact:** This would improve the paper’s evidential focus and prevent overreading; it is not essential to the core case-study claim.

## Questions for the authors

1. Is a 200k full32 continuation feasible from existing checkpoints? If not, can the authors quantify how closely the 25k full32 data order, optimizer state, and evaluation implementation match the keep14 path before its 34.5k resume?
2. The base PPL is 7.398 while keep14's best PPL is still 10.561 (1.428x base). Do the authors have evidence for the more decision-relevant regime in which PPL approaches the intact model more closely but target recovery still fails? This would help distinguish proxy insufficiency from the simpler fact that the compressed model remains substantially worse on the proxy itself.
3. Can the authors release aligned closed-book correctness/prediction artifacts, even if benchmark text is excluded, so the reported base--keep14 gaps can receive paired uncertainty intervals and independent audit?

## Suggestions

- Put a one-sentence single-run/no-seed warning in the abstract near the main result, not only in the final limitations section.
- Add a compact trajectory figure/table for the intact base, keep14, and any future replicated seeds with a common horizontal axis (steps, tokens, and ideally estimated FLOPs).
- Consider reporting an explicit effect-size view in addition to absolute scores: e.g., base-gap closure and chance-adjusted recovery, with the caveats already made in Appendix Table 10.
- Preserve the disciplined language around “operating points”; it is a strength and should also be applied to any wording that could imply that the available full32 point rules out all corpus/training explanations.

## Claim--evidence audit

| Claim | Evidence in manuscript | Assessment |
|---|---|---|
| C1: Along the observed keep14 path, PPL improvement alone does not imply recovery on the measured targets. | PPL 10.826 -> 10.561 from 128k to 200k; MMLU .3012 -> .3191 but final .319 vs base .605; closed-book gaps in Table 2/16. | **Supported for this single literal path and measured tasks.** It should not be read as a calibrated threshold failure or universal statement. |
| C2: The result is not merely an answer-letter artifact. | Complete-option MMLU changes the score but random-init has a high content floor; all three closed-book generation evaluations retain a large base--keep14 gap. | **Partly supported.** The closed-book recurrence is persuasive descriptively, but there are no aligned per-item artifacts/CIs for those tasks and interfaces are multi-factor. |
| C3: Nominal 16-layer depth alone does not determine the endpoint. | ShortGPT-16 reaches PPL 9.780 and MMLU .474 vs keep14 10.561/.319 at 200k. | **Supported as construction dependence.** Not evidence for which structural feature causes the difference. |
| C4: The operating points bound short-horizon corpus shift. | full32 at 25k is close to base relative to the keep14 endpoint. | **Supported only against an immediate/short-horizon complete explanation.** It does not control long-horizon intact CPT. |
| C5: The paper motivates reporting likelihood, targets, interfaces, construction, budget, and run uncertainty separately. | The observed metric/interface/construction divergences plus detailed limitations. | **Reasonable case-study recommendation**, not a validated universal protocol. |

## Citation and novelty audit

**Citation coverage.** I checked all 33 rendered bibliography entries in `main.bbl`; citations are present and the visible bibliography is consistent with the manuscript's related-work narrative. Five representative claim--citation matches are sound:

| Manuscript claim | Citation(s) | Match |
|---|---|---|
| Deeper-layer removal/CPT recovery is established prior work. | Gromov et al. (2025) | Appropriate closest antecedent for deep-layer pruning and recovery behavior. |
| Recovery trajectories and retraining/initialization comparisons predate this study. | Shortened LLaMA; Minitron; IteRABRe | Appropriate; the paper does not claim these ideas as new. |
| Compression can yield loss/task or knowledge-sensitive evaluation gaps. | Namburi et al. (2023); Jaiswal et al. (2024); Xu et al. (2024) | Appropriate motivation for multi-dimensional evaluation. |
| MMLU outcomes depend on scoring/evaluation details. | Wang et al. (2024); Alzahrani et al. (2024); Gupta et al. (2024) | Appropriate support for treating the protocol contrast cautiously. |
| ShortGPT is a relevant selected-layer comparator. | Men et al. (2025) | Appropriate; the manuscript accurately describes its own comparison as coupled rather than a reproduction of a selection-only ablation. |

**Novelty assessment.** The manuscript correctly positions novelty as modest. (1) The closest technical antecedents are depth-pruning/recovery works such as Gromov et al., Shortened LLaMA, Minitron, and IteRABRe; these already cover trajectories, loss/task gaps, and some initialization or retraining controls. (2) The closest evaluation antecedents already argue that compression quality cannot be summarized by perplexity alone. (3) The incremental contribution is the particular, transparent diagnostic package in an OLMo prefix-plus-fresh-tail case: late-path tracking, a short intact branch, null operating points, two MMLU interfaces, and three closed-book evaluations. I found no basis to treat this as an algorithmic pruning novelty, and the paper does not do so. The discussion of very recent concurrent 2026 pruning/recovery work appropriately further limits the novelty claim; under the manuscript's stated three-month rule, those works should remain explicitly labeled concurrent rather than used to claim precedence.

## Desk, artifact, ethics, and presentation checks

- **Desk checks:** The frozen PDF is anonymous, uses ACL review style, has 17 pages including references/appendix, contains a Limitations section and Ethical Considerations, and has no visible unresolved citations/references, placeholders, or reviewer-directed/instructional text. The main-body result numbers and Table 2 agree to stated rounding; the paper explains why common-rerun aggregates differ slightly from headline cells.
- **Methods/metrics/statistics:** Metrics, sample counts, prompts, normalizations, checkpoint budgets, and the distinction between item-level and seed-level uncertainty are mostly explicit. The statistical analysis is appropriate for fixed-checkpoint MMLU comparisons, but it is necessarily insufficient for run-level generalization. Closed-book results lack paired uncertainty artifacts.
- **Artifact:** The anonymized package is valuable for audit but incomplete for reproduction, for the reasons above. The paper honestly documents those gaps rather than overstating availability.
- **Figures/tables:** Figure 1 is clear and self-contained; its caption states the single-run and short-horizon-control caveats. Tables 2, 14--16, and the appendix trajectory tables are informative. The quantity of appendix detail is a strength, though a reader would benefit from one consolidated common-checkpoint plot.
- **Ethics:** The ethics discussion is adequate for this measurement study: it notes inherited model/data risks, energy-use uncertainty, and redistribution constraints. No new human-subject data are claimed.

## Overall assessment

This is a useful, carefully bounded Findings-level measurement paper. Its main empirical point is credible and important for practitioners who might equate improving continuation loss with recovery after structural compression. The manuscript’s unusually frank treatment of confounds, missing provenance, and what its controls do *not* establish substantially improves its credibility. However, the absence of training replicates, a long-horizon intact control, and fully reproducible principal-run/closed-book artifacts prevents a main-conference-level evidential claim. I therefore recommend **3.0 (Findings)** rather than rejection: the core observation is worth disseminating if it remains explicitly framed as a single-run observed-path case study.

## Review-process self-check

I read the main paper and appendix in two passes; inspected the reported figures and tables; audited claim scope, controls, metrics, seeds/statistics, reproducibility, and ethics; checked rendered bibliography coverage and representative claim--citation matches; and evaluated novelty against the closest cited pruning/recovery and compression-evaluation literature. I did not treat the paper as proposing a new algorithm, and I did not penalize it merely for being a measurement study.
