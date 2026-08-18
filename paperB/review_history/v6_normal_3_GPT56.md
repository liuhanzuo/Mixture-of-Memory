review_mode: normal
soundness: 3.5
excitement: 3.0
overall: 3.0
confidence: 4.0
reproducibility: 2.5

## Summary and outline of claims

This paper is a deliberately narrow measurement case study of post-pruning continued pretraining. It studies OLMo-2-7B after retaining blocks 0--13, appending two fresh blocks, and training for 200k steps (`keep14+fresh2`). The main claim is not that perplexity is unhelpful, nor that a capability has been deleted or localized. Instead, it is that, on the reported literal path, improvement in same-source held-out perplexity alone does **not** imply recovery to intact-base performance on the measured knowledge-sensitive evaluations.

The paper supports this with (i) a late within-run keep14 trajectory, (ii) answer-letter and complete-option MMLU interfaces, (iii) three zero-shot closed-book QA evaluations, (iv) a 25k intact full32 continuation, (v) random-init and frozen-front same-shape operating points, and (vi) a coupled non-contiguous ShortGPT-16 construction. The manuscript is unusually explicit that these are not clean causal ablations and that each trained construction is one run.

### Claim--evidence map

- **C1: PPL improvement alone is insufficient evidence of target recovery on the observed keep14 path.** From 128k to 200k, PPL falls from 10.826 to 10.561 while answer-letter MMLU rises only from .3012 to .3191 and remains 28.6 points below the .605 intact base; at 200k, the base--keep14 gaps recur on PopQA (.257 vs .142), TriviaQA (.636 vs .294), and NQ-open (.205 vs .060) (Fig. 1; Table 2; Appendix Tables 10 and 15). This is well aligned with the carefully limited wording.
- **C2: The late MMLU gain is real for the two fixed checkpoints but does not establish a training dynamic.** The common-item rerun reports +1.68 points from 128k to 200k with paired bootstrap CI [1.08, 2.29] and exact McNemar p-value \(4.12\times10^{-8}\) (Appendix Table 10). The paper correctly says that this is item uncertainty conditional on realized checkpoints, not seed uncertainty.
- **C3: The conclusion is interface-sensitive, and complete-option MMLU has a substantial non-target floor under this recipe.** Keep14's content-normalized score is .3832, but random-init reaches .3598 while being .2470 on answer letters (Appendix Table 15). This motivates caution about treating the content score as recovered knowledge, though the two interfaces change several factors simultaneously.
- **C4: The endpoint depends on construction, not merely nominal 16-layer depth.** At equal nominal step budget, ShortGPT-16 reaches PPL 9.780 and answer-letter MMLU .474 versus 10.561/.319 for keep14 (Table 2). The paper correctly restricts this to construction dependence because block selection, inherited count, final-block retention, and fresh-tail use all change together.
- **C5: Several simple explanations are bounded but not eliminated.** The 25k full32 point is closer to base than keep14 on the reported metrics, and the closed-book results are not an answer-letter-only artifact. However, full32 is not available at 200k, so it cannot establish a matched long-horizon counterfactual.

## Strengths

1. **The paper's scope discipline is excellent.** It repeatedly distinguishes the supported implication from much stronger claims. For example, the abstract says, “We do not claim knowledge deletion or localization, causal factor attribution, universal recovery dynamics, or failure beyond the measured budgets.” This restraint is matched by the design and substantially improves credibility.

2. **The central descriptive result is clear and well triangulated.** Figure 1 is effective: it displays the within-run PPL/MMLU path, the base gap, the short full32 horizon, the random content floor, and the stronger but coupled ShortGPT point. The main observation is also repeated across independently motivated closed-book generation metrics rather than resting solely on MMLU.

3. **The manuscript handles confounding and uncertainty more responsibly than is typical for a pruning case study.** It explicitly labels random/frozen/ShortGPT as operating points, documents LR and trainable-set differences, gives sample counts and scoring procedures, and carefully distinguishes item-level bootstrap/McNemar inference from run-level uncertainty. The limitations section is concrete about unavailable seed, loader offset, GPU-hours, and closed-book per-item outputs.

4. **The interface analysis is useful.** The paper does not oversell complete-option scoring. The comparison with a fully random model is a valuable warning that a higher content-normalized MMLU score need not mean target recovery in this setting.

5. **Presentation and compliance are strong.** The paper is anonymous, uses the review style, has a Limitations and Ethical Considerations section, has no unresolved TeX references/placeholders in the supplied source, and the 17-page PDF rendered cleanly. Tables and figures are legible; the main figure and the 1B trajectory were inspected directly.

## Weaknesses and required revisions

### Major weakness 1: The central result is based on one unreplicated optimization trajectory.

- **Location / short quote:** Introduction, “Every trained construction is a single run”; Limitations, “Training seeds were not explicitly set in the historical runs.”
- **Problem:** The conclusion is appropriately phrased as a statement about literal observed paths, so the single run does not invalidate C1. But it materially limits the scientific value of the proposed reporting discipline and prevents separating a persistent proxy failure from a particular optimization/data-order realization. The retained checkpoints were also selected after target metrics were inspected, with no registered stopping rule.
- **Impact:** This is the main reason I do not view the evidence as ACL-main-conference strength. The paper establishes an informative case study, not robust evidence about post-pruning recovery behavior.
- **Concrete remedy:** Run at least 3 independent keep14+fresh2 seeds with a predeclared 128k/200k (or fixed-token) evaluation plan, report mean/dispersion for PPL and each target metric, and state whether each seed exhibits the same “PPL improves while a large base gap remains” pattern. A smaller but still valuable alternative is two additional seeds at the final 200k point plus one intermediate checkpoint, explicitly framed as a replication of C1 rather than an exhaustive scaling study.
- **Severity:** **Major.**

### Major weakness 2: The intact-control evidence cannot adjudicate the 200k comparison.

- **Location / short quote:** Table 2 caption: “full32 ends at 25k and only bounds short-horizon corpus shift; it is not a 200k endpoint control.”
- **Problem:** This caveat is correct, but it leaves a key alternative insufficiently tested: an intact model may also drift over 200k under the same continuation setup. The 25k result rules out only catastrophic *early* corpus/recipe shift, not a long-horizon effect. The same limitation applies to interpreting performance gaps as attributable to pruning rather than the long continuation path.
- **Impact:** The main claim “PPL improvement alone does not imply target recovery” still holds descriptively for keep14 relative to the original base, but claims that the operating points “bound” corpus-shift explanations should be weakened, and the paper cannot characterize a pruning-specific 200k deficit relative to intact CPT.
- **Concrete remedy:** The most decisive experiment is a full32 continuation to 200k using the same data stream, token accounting, optimizer schedule, and evaluation protocol. If that run is unavailable, revise the prose to consistently say that the full32 branch only checks early drift, and foreground it as an unresolved control rather than a bounded explanation.
- **Severity:** **Major.**

### Major weakness 3: Closed-book evidence lacks released prediction-level artifacts and paired uncertainty.

- **Location / short quote:** Results: “the saved paper bundle does not contain the aligned per-item artifacts needed to recompute it”; Appendix B.3 says the source package includes `anonymous_artifact/`.
- **Problem:** The frozen source directory supplied for review does not actually contain `anonymous_artifact/`, despite the paper asserting that the source package includes it. Thus neither the claimed scripts/manifests nor the available per-item MMLU files can be audited from this submission bundle. Separately, the closed-book results are headline evidence for C1 but have only aggregate values, no aligned predictions, and no paired intervals.
- **Impact:** This lowers reproducibility and makes it difficult to audit normalization, answer matching, or the scale of uncertainty for the QA gaps. The very large base--keep14 differences are plausibly robust, but this should not rely on trust alone.
- **Concrete remedy:** Include the stated anonymous artifact directory in the submission or provide an anonymous archival link, with checksums and a one-command evaluation/analysis entry point. At minimum, release aligned prediction IDs and normalized outputs for base, keep14, full32-25k, random, and frozen for all three QA tasks, plus scripts that reproduce aggregate scores and paired bootstrap intervals.
- **Severity:** **Major for reproducibility; minor-to-moderate for the descriptive conclusion.**

### Minor weakness 4: The novelty framing risks being more package-level than research-level.

- **Location / short quote:** Related Work: “Our narrower increment is the joint diagnostic package in one OLMo prune--regrow setting.”
- **Problem:** The paper itself documents close antecedents on loss--task dissociation, recovery curves, initialization comparisons, and interface sensitivity. The novelty is therefore the combination of controls and unusually careful interpretation, not a new method, measurement principle, or broadly established empirical law.
- **Impact:** This limits excitement, especially for a main-conference paper, though it is compatible with a solid Findings contribution.
- **Concrete remedy:** Tighten the title/abstract/contributions around a reproducible OLMo case study and reporting checklist, reduce any implication that the observation is newly discovered, and make the nearest-work comparison more evidence-based (e.g., exact settings/results rather than binary table codes).
- **Severity:** **Minor.**

## Questions for the authors

1. Can the authors provide (or confirm that they will provide upon publication) the missing `anonymous_artifact/` directory described in Appendix B.3, including the named checksum manifest and MMLU per-item files?
2. Is a 200k full32 continuation feasible? If not, what evidence supports treating the original base rather than a long-horizon intact branch as the relevant recovery reference for this continuation recipe?
3. For the 128k-to-200k keep14 trajectory, were these exact checkpoints selected before looking at MMLU/QA results, or were they selected after inspection along with the shallow endpoints? Please give the checkpoint-retention timeline.
4. Why is the random-init operating point trained with a 5x higher peak LR? It is fairly presented as a non-clean ablation, but an LR-matched random control would sharpen the interpretation of the content-score floor.

## Suggestions

- Add a compact “what would falsify this case-study conclusion?” paragraph: e.g., a replicated keep14 seed that both improves PPL and closes the predeclared base-relative target gap, or an intact 200k run showing comparable degradation.
- Report effect sizes for the large base--keep14 closed-book gaps with binomial/paired intervals once predictions are released; do not imply that lack of paired artifacts makes uncertainty impossible in principle.
- Consider adding out-of-domain likelihood or a contamination/overlap audit. The paper is transparent that PPL is same-source and in-domain, but a second likelihood domain would make the proxy discussion more informative.
- Move the exact artifact availability statement into the main reproducibility paragraph and ensure it matches the actual submitted package.

## Technical and statistical assessment

The formula-level methodology is simple and largely appropriate. PPL aggregation as \(\exp(\sum_g \mathrm{NLL}_g/\sum_g n_g)\) is correctly preferable to averaging shard PPLs. The paper reports sample counts, chance levels, prompts, decoding, normalization, effective batch size, optimizer hyperparameters, and nominal token presentations. The MMLU paired bootstrap and exact McNemar tests are correctly described as conditional on fixed checkpoints. I found no unsupported conversion of item-level CIs into seed-level claims.

However, the statistical evidence addresses only a small portion of the paper's inferential burden. The QA results have no prediction-level artifacts or intervals; all arm comparisons rely on one training realization; the shallow depth rows are selected at unequal, outcome-informed stopping points; and neither steps nor tokens nor FLOPs are matched across all comparisons. The paper acknowledges these limitations, but acknowledgement does not substitute for the missing controls when interpreting construction or corpus effects.

## Novelty analysis

I checked the cited closest work and recent related-work descriptions against the manuscript's positioning. Gromov et al. already study deep-layer removal with post-healing loss/task dissociation; Shortened LLaMA and Minitron study recovery/retraining and initialization-related comparisons; IteRABRe studies iterative recovery trajectories; and the cited compression/interface literature already motivates evaluation beyond perplexity. The manuscript's “within three months” concurrent-work cutoff appears to be handled reasonably for SlimQwen (May 2026) and ShortOPD (July 2026) relative to this August 4, 2026 frozen version.

Accordingly, I find the paper's *narrow* novelty claim credible: it combines an OLMo prefix-plus-fresh-tail trajectory, a short intact branch, same-shape operating points, two MMLU interfaces, and closed-book QA, while being unusually explicit about confounds. I do **not** find support for novelty as a new general discovery that perplexity and target recovery can diverge, nor for a new pruning or recovery technique. This novelty profile supports a Findings-level empirical/measurement paper if the artifact discrepancy is fixed.

## Citation and desk-check audit

- **Bibliography:** `main.bbl` contains 33 entries, matching the 33 unique citation keys used in the TeX source; there are no missing cited keys or duplicate BibTeX keys. The `.bib` file contains 17 uncited background entries, which is harmless but should be pruned for cleanliness.
- **Citation--claim spot checks:** (1) OLMo-2 base model citation for the 7B backbone; (2) MMLU, PopQA, TriviaQA, and Natural Questions benchmark citations; (3) Gromov et al. for post-healing loss/task dissociation; (4) Shortened LLaMA/Minitron/IteRABRe for trajectories/retraining; (5) interface sensitivity citations for MMLU scoring/order; (6) SlimQwen/ShortOPD as concurrent work. The claims are generally conservative and aligned with the cited works as characterized.
- **Formatting/anonymity:** The supplied PDF is 17 A4 pages including references and appendix, is anonymous, uses `\usepackage[review]{acl}`, includes Limitations and Ethical Considerations, and has no visible reviewer-directed or instruction-like text. I found no unresolved source references, TODO/TBD/FIXME/placeholder markers, or obvious table/abstract number contradictions.
- **Figures/tables:** Both figures were inspected; the main figure is clear and quantitatively consistent with the tables, and the 1B trajectory is readable. The remaining tables are dense but legible in the PDF. The source uses vector PDF figures (no raster images embedded in the compiled PDF).
- **Compute/reproducibility:** The paper candidly states that seeds, loader offset, historical GPU-hours, and some prediction artifacts are unavailable. This honesty is a strength, but it warrants the low reproducibility score, particularly because the described anonymous artifact directory is absent from the frozen source bundle reviewed here.
- **Ethics:** The discussion is appropriate for a measurement/compression study; no new human data are collected, and it notes inherited model/data risks and energy use without inventing unavailable compute totals.

## Overall assessment

This is a careful, clearly written, and commendably scoped empirical note. The main descriptive claim is supported: on this one keep14 OLMo path, lower in-domain PPL did not imply recovery to intact-base performance on the measured MMLU and closed-book QA outcomes. Its strongest contribution is methodological honesty about what the available controls do and do not identify.

The evidence is nevertheless too narrow and incomplete for me to recommend main-conference acceptance: one training run per arm, no 200k intact control, and an artifact/package mismatch constrain both generality and auditability. With the artifact fixed and the contribution framed as a measurement case study, I would view it as a reasonable **Findings** paper.

## Scores

- **Soundness: 3.5 / 5.0.** The literal-path conclusion is well supported and the caveats are technically sound; missing replication and long-horizon intact control limit broader interpretation.
- **Excitement: 3.0 / 5.0.** Useful diagnostic packaging and unusually responsible analysis, but limited methodological novelty and one setting.
- **Overall: 3.0 / 5.0.** Findings level: worthwhile if the artifact discrepancy is corrected, but below ACL main-conference strength in its current form.
- **Confidence: 4.0 / 5.0.** The paper's claims and limitations are unusually explicit, and the source/PDF checks were straightforward; uncertainty remains about unavailable artifacts and external verification of all experimental results.
- **Reproducibility: 2.5 / 5.0.** Good written protocol detail and stated checksums, but missing submitted artifact directory, unavailable seeds/loader offset, unavailable GPU records, and missing aligned closed-book predictions prevent reliable end-to-end reproduction or audit.

## Review-process self-check

I evaluated only the frozen v6 manuscript PDF/source and the NORMAL template, independently of prior reviews. I read the main paper and appendix in two passes; inspected the supplied figures and all tables; checked claimed limitations, controls, metrics, uncertainty, compute, artifact statements, bibliography/citation-key consistency, anonymity/style, and unresolved-reference/placeholder markers. Each weakness above includes a verified location/short quote, the problem, its impact, a concrete remedy, and severity. I did not interpret item-level statistical tests as independent training-seed evidence, and I did not treat confounded operating points as causal ablations.
