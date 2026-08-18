---
review_mode: normal
soundness: 3.5
excitement: 3.0
overall: 3.0
confidence: 4.5
reproducibility: 2.0
---

## Summary and outline of the approach

This paper is an observational measurement study of post-depth-pruning recovery in OLMo-2. The principal construction keeps the first 14 of 32 pretrained decoder blocks, appends two fresh blocks, and continues pretraining (CPT) for 200k optimizer steps. The study asks whether improving held-out next-token likelihood is sufficient evidence that target capabilities have returned.

The paper evaluates several complementary but mostly unmatched operating points: the intact base; an intact full32 CPT branch available only at 25k; keep14+fresh2; shallower keep8/10/12 prefix arms with unequal stopping points; frozen-front and fully random 16-layer points; and a 16-layer non-contiguous ShortGPT construction. Measurements include in-domain perplexity, answer-letter MMLU, complete-option MMLU, three no-retrieval closed-book QA tasks, a broader zero-shot likelihood suite, group/subject breakdowns, and exploratory intact-model layer-wise readouts.

The central empirical result is a substantial likelihood/capability separation in the realized keep14 run. At 200k, keep14 reaches PPL 10.561 versus 7.398 for the intact base (1.428x), but MMLU letter accuracy is .319 versus .605. Complete-option scoring increases keep14 to .383, but a fully random 16-layer model obtains .360 under that protocol while remaining at chance on letters, revealing a large fluency/interface floor. Closed-book scores also remain far below the base: PopQA .142 versus .257, TriviaQA .294 versus .636, and NQ-open .060 versus .205. A coupled ShortGPT-16 construction is much stronger (PPL 9.780; MMLU .474), showing that one nominal 16-layer depth does not determine a unique endpoint, while not isolating the responsible architectural factor.

## Claims and evidence map

| ID | Claim | Main evidence | Assessment |
|---|---|---|---|
| C1 | In the observed keep14 run, likelihood recovers substantially faster/more completely than answer-letter MMLU and closed-book recall. | Table 2; Figures 1–2; Tables 4, 10, 13, 17. | Directly supported for this single realization and observed budget. |
| C2 | The MMLU scoring interface materially changes apparent recovery, but complete-option accuracy is not a clean knowledge measure. | Table 16: keep14 .3184 letter/.3832 normalized content; random .2470/.3598. | Supported as an exploratory protocol sensitivity result; not a readout or knowledge ablation. |
| C3 | Short-horizon corpus shift alone is insufficient to explain the large keep14 gap. | Full32@25k remains near base: PPL 7.670, letter MMLU .588, content .466, plus closed-book retention. | Supports only a 25k bound, not the 200k counterfactual. The paper states this correctly. |
| C4 | Nominal depth alone does not fix a unique recovery endpoint. | keep14 versus ShortGPT-16 at 200k: .319 versus .474 MMLU and 10.561 versus 9.780 PPL. | Descriptively supported; causal source is not isolated. |
| C5 | The same-shape 200k operating points differ reliably on realized MMLU items. | Table 15 exact McNemar and paired bootstrap; keep14-random +7.11 pp, CI [6.14, 8.09]. | Supports checkpoint-conditional item differences, not seed stability. |
| C6 | Late keep14 recovery continues but does not broadly catch up within 200k. | 128k→200k PPL 10.826→10.561; aggregate MMLU .3012→.3191; paired rerun +1.68 pp, CI [1.08, 2.29]. | Supported for the realized trajectory; “eventual plateau/convergence” remains untested. |
| C7 | Recovery is heterogeneous across MMLU groups/subjects. | Figure 3; Tables 18 and 22. | Descriptive evidence is adequate; no domain-level causal or multiplicity-controlled claim should be made. |
| C8 | The paper’s reporting proposal—separate likelihood, capability, interface, construction, and budget—is motivated by this case. | The combined control/evaluation bundle. | Reasonable methodological recommendation, but not validated as a universal protocol. |

## Strengths

1. **Careful claim bounding.** The manuscript repeatedly distinguishes observations from causal conclusions. It explicitly states that item-level intervals are not training-seed uncertainty, unmatched controls are operating points rather than factor-isolating ablations, and layer-wise readouts are not knowledge localization.

2. **A useful negative/measurement result.** The paper documents a practically important failure mode: a large improvement in in-domain likelihood can coexist with large deficits in answer selection and closed-book recall after structural intervention. This is valuable even without a new pruning algorithm.

3. **Multiple interfaces and capability probes.** The combination of answer-letter MMLU, complete-option MMLU, and PopQA/TriviaQA/NQ-open prevents the argument from resting on one brittle interface. The random-init content floor is especially informative.

4. **Transparent control limitations.** The full32 25k horizon, random-init learning rate mismatch, frozen trainable set, and four coupled ShortGPT differences are all disclosed rather than presented as clean causal ablations.

5. **Good appendix and numerical bookkeeping.** The paper reports all 14,042 MMLU items, 57 subject results, checkpoint provenance, paired tests, raw/normalized metric sensitivity, architecture reconstruction checks, optimizer details, and exact sample counts. At least the following headline numbers are internally consistent across abstract, text, and tables: 7.398 base PPL; 10.561 keep14 PPL; 1.428x PPL tax; .605/.319 base/keep14 letter MMLU; .383 keep14 content MMLU; .474 ShortGPT MMLU; and the .142/.294/.060 keep14 closed-book scores.

6. **Readable presentation.** All six figures and all 22 tables are legible and captioned with important qualifications. The visualizations do not overstate matched-PPL, matched-compute, causal, or mechanistic interpretations.

## Major weaknesses

### W1. The central training-dynamics evidence has no seed replication

- **Location and quote:** Limitations, p. 8: “keep14, ShortGPT, and the same-shape points are single runs”.
- **Problem:** Every central endpoint and trajectory is one realized training run. The item-level CIs and McNemar tests quantify finite evaluation-item uncertainty conditional on fixed checkpoints; they cannot establish that the keep14 trajectory, late gain, or keep14–ShortGPT gap is stable across fresh-block initialization, data order, optimization noise, or block selection.
- **Affected claim/norm:** C1, C4, C5, and C6; empirical claims about training dynamics normally require independent training replications or substantially narrower wording.
- **Why important:** The paper’s contribution is specifically a recovery-path measurement. Without seed variation, the field cannot tell whether the observed path is representative or idiosyncratic. The authors correctly bound this, which preserves basic soundness, but it limits the result to a case study and caps its evidential strength.
- **Sufficient remedy:** Replicate at minimum keep14 and ShortGPT-16 with three independent seeds using recorded initialization/data-order seeds, reporting mean/range or seed-level uncertainty for PPL, MMLU, and closed-book endpoints. Ideally replicate the 128k/153.5k/200k keep14 checkpoints as well.
- **Severity:** Major.

### W2. The available controls do not identify why recovery differs

- **Location and quote:** Analysis §6.3, p. 6: “The comparison does not show that block 31 is uniquely responsible”.
- **Problem:** ShortGPT versus keep14 changes four factors simultaneously: 16 versus 14 inherited blocks, non-contiguous versus prefix selection, retention of the original final block, and absence versus presence of fresh tails. The random-init arm uses a different learning rate; frozen-front changes the trainable set. Consequently, the strongest contrast cannot attribute the endpoint gap to selection, inherited computation, final-layer retention, fresh-layer damage, or adaptation.
- **Affected claim/norm:** C4 and any implication about construction choice; causal conclusions require matched ablations.
- **Why important:** The observed .155 MMLU gap is the paper’s clearest evidence that “exact construction” matters, but it provides little actionable guidance about which construction property matters. The current paper generally avoids causal language, yet novelty and practical insight remain limited without isolation.
- **Sufficient remedy:** Add a minimal matched 16-layer factorial set under the same schedule/LR: contiguous keep16; keep14+copied original final two; ShortGPT-selected 14+fresh2; and/or keep14 with alternative copied/fresh tails. Even two carefully selected matched arms that isolate inherited-block count and final-block/fresh-tail use would materially improve attribution.
- **Severity:** Major.

### W3. Training horizon and stopping are unmatched, weakening trajectory/depth conclusions

- **Location and quote:** Method §3.2, p. 3: “selected after knowledge-sensitive metrics appeared stable while PPL was still decreasing”.
- **Problem:** keep8/10/12 stop at 121k/83.5k/124k, keep14 and ShortGPT at 200k, and full32 at 25k, with no registered common stopping rule. The shallower endpoints are partly selected after observing target metrics. These data cannot support a compute-matched depth ladder, a long-horizon corpus control, or robust statements about depth-dependent recovery speed.
- **Affected claim/norm:** C3 and any cross-depth interpretation around Figures 2/4 and Table 3; fair comparison requires matched token/FLOP budgets and predeclared checkpoints.
- **Why important:** The paper explicitly frames these as unequal observed operating points, so this is not a fatal validity error. However, the unmatched horizon leaves the key alternative—continued later capability recovery or long-horizon full-model drift—open.
- **Sufficient remedy:** Evaluate all prefix arms and full32 at a shared set of token checkpoints through 200k (or clearly compute-matched FLOPs), with a stopping schedule fixed before evaluating MMLU. A cheaper minimum is full32@200k plus literal 200k evaluations for keep8/10/12.
- **Severity:** Major.

### W4. Reproducibility is materially compromised by unrecorded randomness and resume state

- **Location and quote:** Limitations, p. 8: “Exact reproduction is limited by unset training seeds, an unrecorded resumed data-loader offset”.
- **Problem:** Training seeds are unset, the resumed keep14 run does not preserve the within-epoch data-loader offset, exact project compute is unavailable, and there is no frozen runnable artifact. The allowed source contains tables and prose but no training/evaluation implementation or configs that can independently reproduce the results.
- **Affected claim/norm:** Reproducibility of all empirical claims; ARR expects sufficient details/artifacts for independent replication.
- **Why important:** This is unusually consequential in a single-run paper, because the missing seed and exact data order are precisely the sources of variation that were not estimated.
- **Sufficient remedy:** Release anonymized runnable code, exact configs/environment lock, checkpoint hashes, data-array/shard identifiers, all seeds, restored sampler/data-loader state, and per-item predictions. Re-run at least the principal arm with a fully recorded state if exact reconstruction of the reported run is impossible.
- **Severity:** Major.

## Minor weaknesses

### W5. Evaluation remains narrow and partly in-domain

- **Location and quote:** Limitations, p. 8: “no contamination audit or out-of-domain PPL is reported.”
- **Problem:** Likelihood is measured only on an in-domain Dolmino shard, the study is English/base-model/mostly 7B, and benchmark contamination is not assessed. The Qwen and 1B results are deliberately only directional context.
- **Affected claim/norm:** C1 and C8; claim scale/generalization should match model, corpus, language, and distribution coverage.
- **Why important:** A reporting recommendation aimed at post-pruning recovery would be more convincing if the likelihood–capability separation held under out-of-domain likelihood and at another matched family/scale.
- **Sufficient remedy:** Add at least one out-of-domain held-out corpus and one matched second model family or size; report a basic contamination/overlap audit for the knowledge benchmarks.
- **Severity:** Minor (because the manuscript consistently labels itself a narrow case study).

### W6. Closed-book coverage is incomplete for the strongest alternative construction

- **Location and quote:** Table 2 caption, p. 5: “missing ShortGPT closed-book cells were not evaluated.”
- **Problem:** ShortGPT is the strongest 16-layer construction and central to the construction argument, but it lacks PopQA/TriviaQA/NQ-open results. Thus the paper cannot show whether its large MMLU advantage extends to the independent generation interface used to reject an answer-letter-only account.
- **Affected claim/norm:** C4 and the breadth of the construction comparison; central baselines should be evaluated on the same headline metrics.
- **Why important:** This is a small, direct experiment and could change the interpretation of whether ShortGPT preserves knowledge/recall or mainly the MMLU interface.
- **Sufficient remedy:** Run the already-defined closed-book protocol on ShortGPT-16 and add the three cells, ideally with per-item prediction files and the retained EM/containment/F1 sensitivity metrics.
- **Severity:** Minor.

### W7. The content-MMLU comparison bundles several protocol changes

- **Location and quote:** Appendix Table 16 caption, p. 17: “change prompt, candidate string, tokenization, and normalization together”.
- **Problem:** Letter versus complete-option scoring is not a single-variable interface ablation, and there is no paired uncertainty analysis for the protocol differences. The random-init floor usefully falsifies a clean knowledge interpretation, but the remaining gap cannot be assigned specifically to answer-symbol mapping/readout.
- **Affected claim/norm:** C2; interface or readout claims require matched prompts/candidates or explicit decomposition.
- **Why important:** The paper already calls this exploratory, so the issue is bounded; however, phrases such as “consistent with a content-to-symbol or readout contribution” remain underdetermined.
- **Sufficient remedy:** Factor the protocol into matched variants: same prompt with letter versus option-text continuation, summed versus token-normalized scoring, and answer-order permutations; report paired item-level differences.
- **Severity:** Minor.

### W8. Efficiency/compute evidence is insufficient for practical pruning conclusions

- **Location and quote:** Limitations, p. 8: “We also do not report latency, throughput, memory, or recovery FLOPs.”
- **Problem:** The manuscript motivates smaller language models but reports neither inference efficiency nor recovery cost, and hardware differs across H20/B200 nodes. A step budget is not a compute-normalized comparison across 10–32 layers.
- **Affected claim/norm:** Practical relevance and any implicit efficiency motivation; compression work should quantify realized inference benefit and training/recovery cost.
- **Why important:** This does not undermine the measurement result, but it prevents assessing the practical tradeoff between the stronger ShortGPT endpoint and the recovery compute spent.
- **Sufficient remedy:** Report parameter count for every arm, tokens and approximate FLOPs, GPU-hours by run, peak memory, and standardized latency/throughput at representative batch/sequence lengths.
- **Severity:** Minor.

## Questions for the authors

1. Were keep14 and ShortGPT trained on exactly the same token array/order through 200k, apart from the documented keep14 resume discontinuity? If not, how much of the construction comparison could reflect exposure/order differences?
2. Why was the full32 branch stopped at 25k? Is a later checkpoint unavailable because of cost, failure, or deletion, and can at least a 200k MMLU/PPL control be produced?
3. Can the authors evaluate ShortGPT on PopQA, TriviaQA, and NQ-open using the existing protocol before final submission?
4. For the complete-option MMLU protocol, can the authors report paired per-item transitions between letter, content-raw, and content-normalized predictions and answer-order sensitivity?
5. The keep14 run resumed at 34.5k without the data-loader offset. Approximately how many windows may have been repeated, and is the full32/ShortGPT data stream affected by analogous resumes?
6. Are there failed or exploratory keep14-like runs whose outcomes could inform selection bias, even if they cannot be aggregated as formal seeds?

## Suggestions

- Preserve the paper’s current bounded language; it is one of its strongest features.
- Prioritize two additions over a broader benchmark sweep: (i) seed replication of keep14/ShortGPT and (ii) one or two matched construction ablations.
- Add ShortGPT closed-book results and full32@200k if computationally feasible; these are the highest-value minimal experiments.
- Separate checkpoint-conditional evaluation uncertainty from training-run uncertainty in every table heading, not only captions/limitations.
- Move a compact “what is and is not identified” table into the main paper, listing each control, matched dimensions, unmatched dimensions, and permissible inference.
- If the appendix readout section remains, keep it explicitly quarantined as background. It is correctly described as a readout rather than a causal storage localization.

## Citation verification and related-work audit

### `main.bbl` integrity

Mechanical audit found **50 bibliography entries, 50 unique cited keys, no uncited entries, no cited-but-missing entries, and no duplicate keys**. I inspected every `main.bbl` entry for author/year/title/venue plausibility, key resolution, duplication, and relevance to its citation context. DOI/arXiv landing-page verification was completed for the load-bearing/newer entries listed below; remaining entries are marked **Unverifiable** at the external-metadata level because the user requested that further network research stop. “Unverifiable” does not mean the entry is false.

| Group / entries | Result |
|---|---|
| Verified arXiv identity/date/title: Gromov et al. (2403.17887), Shortened LLaMA (2402.02834), Minitron (2408.11796), IteRABRe (2503.06291), PASER (2502.12594), LinearPatch (2505.24680), Prune&Comp (2507.18212), calibration study (2604.24938), decision-transition study (2605.07271), SlimQwen (2605.08738), ShortOPD (2607.13124), Fragile Knowledge (2512.22671), answer-order work (2406.19470), Lu et al. (2411.15558), Siddiqui et al. (2407.16286). | Verified against arXiv landing metadata/abstracts. |
| Verified DOI identity: Wang et al. 2024 “My Answer is C”; Alzahrani et al. 2024 “When Benchmarks are Targets.” | DOI metadata matched title/year/authors. |
| Remaining classic datasets/methods and older entries in `main.bbl`. | Bibliographic strings internally checked; external metadata **Unverifiable** under the stop instruction. |

A minor bibliography-quality note: several arXiv-only entries omit URLs in rendered form, and venue/page metadata are sparse for some older entries, but I found no citation-resolution error or false duplicate.

### Citation–claim matches (8 sampled load-bearing claims)

| Paper claim | Cited work(s) | Verification |
|---|---|---|
| Prior depth-pruning work reports loss/task dissociation after healing. | Gromov et al. | **Supported.** The paper explicitly contrasts smooth/continuous healed next-token loss with sharp QA transitions and discusses decoupling loss from MMLU/BoolQ. |
| Shortened LLaMA compares retraining methods and finds CPT preferable at severe pruning. | Kim et al. 2024 | **Supported** by the arXiv abstract, which contrasts CPT and LoRA-based tuning. |
| Minitron studies depth/width pruning and distillation/CPT-style recovery in a practical compression pipeline. | Sreenivas et al. 2024 | **Broadly supported** by title/abstract; the exact Table 1 checkmarks for trajectory and scratch/init were not exhaustively inspected: **partly Unverifiable**. |
| IteRABRe alternates pruning/recovery and analyzes recovery behavior. | Wibowo et al. 2025 | **Supported** at method/analysis level by abstract; the specific phrase “weak MMLU recovery” is **Unverifiable** without deeper paper inspection. |
| PASER selects post-training data for efficient recovery rather than serving primarily as a diagnostic measurement study. | He et al. 2025 | **Supported** by title/abstract. |
| LinearPatch and Prune&Comp diagnose interface/activation-magnitude mismatch and add lightweight compensation. | Chen et al. 2025/2026 | **Supported** by both abstracts. |
| MMLU is sensitive to answer order and first-token answer interfaces. | Gupta et al.; Wang et al. | **Supported** by arXiv/DOI titles and abstracts/metadata. |
| SlimQwen and ShortOPD cover matched scratch/pruned training or recognition/generation recovery issues. | Tang et al.; Zhang et al. | **Supported in broad direction.** SlimQwen compares pruned initialization with scratch under matched budget; ShortOPD explicitly contrasts multiple-choice recognition with free-form generation collapse. |

### Novelty and closest-work analysis (freeze date 2026-08-03; cutoff 2026-05-03)

I used the paper’s freeze date **August 3, 2026** and a three-month cutoff of **May 3, 2026**. Work first posted after May 3 is treated as concurrent rather than prior art.

| Closest work | First public date | Before cutoff? | Overlap and remaining distinction |
|---|---:|---:|---|
| Gromov et al., *The Unreasonable Ineffectiveness of the Deeper Layers* | 2024-03-26 | Yes | Already studies deep-layer removal, healing, QA curves, and loss–task dissociation. The present paper’s distinction is OLMo prefix+fresh-tail recovery with a denser control/interface bundle, not the basic phenomenon. |
| Kim et al., *Shortened LLaMA* | 2024-02-05 | Yes | Already compares depth pruning and retraining methods, including CPT versus LoRA. Present novelty is diagnostic bookkeeping and OLMo-specific trajectories rather than a new pruning/retraining method. |
| Sreenivas et al., *Minitron* | 2024-08-21 | Yes | Broader practical pruning/distillation study with task evaluation. Present work is narrower, with intact-CPT, random/frozen operating points, two MMLU interfaces, and closed-book QA in one case study. |
| Wibowo et al., *IteRABRe* | 2025-03-08 | Yes | Iterative removal/recovery and trajectory analysis. Present distinction is the exact OLMo prefix+fresh-tail construction and control bundle; novelty is incremental. |
| He et al., *PASER* | 2025-02-18 | Yes | Optimizes recovery-data selection rather than primarily measuring what recovered. Complementary rather than directly superseding. |
| Kim et al., calibration study | 2026-04-27 | Yes, narrowly | Shows calibration configuration can dominate search choice in depth pruning. It further weakens broad selection/generalization claims and should remain in closest-work discussion. |
| Shi et al., decision-transition analysis | 2026-05-08 | No—concurrent | Mechanistic/decision-stage account of pruning collapse; more explanatory, but posted five days after cutoff. Correctly treated as concurrent. |
| Tang et al., SlimQwen | 2026-05-09 | No—concurrent | Matched scratch/pruned initialization and progressive large-scale recovery, but MoE/pretraining setting differs. Correctly concurrent. |
| Zhang et al., ShortOPD | 2026-07-14 | No—concurrent | Recognition/generation recovery gap and on-policy distillation. Correctly concurrent and especially relevant to the paper’s interface argument. |

**Novelty judgment:** the manuscript’s claimed novelty is appropriately narrow. Recovery trajectories, perplexity/task dissociation, scratch/init comparisons, and beyond-perplexity evaluation all predate this paper. The defensible increment is the combination of an OLMo prefix+fresh-tail case, a short-horizon same-corpus intact branch, same-shape operating points, two MMLU interfaces, closed-book QA, and an explicitly confounded 16-layer construction comparison. This is useful but incremental; it supports Findings-level excitement more readily than ACL-main novelty.

Search coverage was limited to the closest works above because the user requested termination of further investigation. Any unsearched additional 2026 work is **Unverifiable**.

## Technical, experimental, and statistical audit

- **Method/formulas:** The prune–regrow construction is clearly defined: keep first `k` blocks, append two fresh blocks, copy embeddings/final norm/output head, and define cut/model depth. The chance-adjusted recovery formula and PPL merge formula are correct as stated. No unsupported mechanistic formula is central.
- **Minimal experiments:** The paper has enough evidence for a bounded descriptive case study, but the minimal experiments for stronger claims are missing: seed replication, matched structural ablations, full32@200k, and ShortGPT closed-book evaluation.
- **Baselines/controls:** Intact base, short-horizon intact CPT, frozen-front, random-init, prefix ladder, and ShortGPT are relevant. None except some shape/step matches are fully controlled; the paper mostly labels this correctly. A contiguous keep16 and matched inherited/fresh-tail variants are the most important missing baselines.
- **Metrics:** PPL, MMLU letter/content, closed-book generation, and broad likelihood tasks are complementary. Metric definitions/sample sizes are reported. BoolQ/CSQA/SIQA raw versus normalized sensitivity is exposed. Content MMLU bundles prompt/tokenization/normalization changes and should not be treated as knowledge recovery.
- **Statistics:** Exact McNemar tests and 10,000-sample paired bootstrap intervals are appropriate for aligned item outcomes. Wald marginal CIs are acceptable descriptive intervals with n=14,042, but both interval types are conditional on fixed checkpoints. There is no seed-level inference, no domain multiplicity correction, and no paired uncertainty for the interface comparison.
- **Seeds:** Bootstrap seed 1234 is reported; training seeds are not set. Therefore item resampling is reproducible but training is not.
- **Claim scale:** The paper generally keeps claims at single-run/case-study scale and explicitly rejects a universal recovery law. This restraint is a major positive.
- **Compute:** Hardware type, steps, batch size, sequence length, optimizer, and LR are reported. Exact GPU-hours, FLOPs, latency, throughput, and project-wide compute are absent; mixed H20/B200 hardware complicates comparison.
- **Reproducibility:** Architecture reconstruction and tensor-equality checks are strong, but exact data order, seed, resume offset, runnable artifact, and code/config release are missing. Reproducibility is therefore low despite good prose-level documentation.

## Figure and table audit

- **Figures 1–2:** Correctly visualize late keep14 recovery and heterogeneous operating points. Captions explicitly reject numerical constancy, matched-PPL, matched-compute, and correlation claims.
- **Figure 3:** Group accuracies/recovery appear consistent with Table 18 and the stated chance-adjusted formula. It is descriptive, not causal.
- **Figure 4:** Supports improvement in PPL/ordinary tasks for keep8 while aggregate MMLU remains near chance. The title says “through 44k” although the caption discusses retained 44k and 121k aggregate checkpoints; this is slightly awkward but not materially misleading.
- **Figure 5:** The 1B trajectory is appropriately labeled qualitative context, not replication.
- **Figure 6:** Readout thresholds are clearly quarantined from the recovery claims and explicitly not treated as causal localization.
- **Tables 1–3:** Prior-art positioning and headline results are clear. Table 1’s exact checkmarks for some prior work are only partly externally verified; no decisive contradiction found.
- **Tables 4–10:** Checkpoint provenance and unequal stopping are disclosed. The ShortGPT step-0 PPL 401.124 contextualizes immediate damage. All main/downstream headline cells checked are consistent.
- **Tables 11–17:** Recovery, CIs, late trajectory, metric sensitivity, paired tests, protocol controls, and closed-book results are well documented. Crucially, Table 15 is item-level rather than seed-level evidence.
- **Tables 18–22:** Group/subject and readout details are complete. Subject-level patterns are descriptive and not overinterpreted. Table 22 omits ShortGPT subject-level columns, but Table 18 supplies its broad-group results; this is not central.

## Desk, formatting, anonymity, ethics, and manipulation checks

- **Page limit:** The PDF has 18 pages. Main text, including Limitations and Ethical Considerations, ends on p. 8; references begin on p. 9 and appendices on p. 12. This appears compatible with an 8-page ACL/ARR main-text limit.
- **Limitations:** Present and unusually candid. It covers single runs, unequal horizons, unmatched controls, interface confounds, in-domain evaluation, missing compute, unset seeds, resume offset, and non-mechanistic readouts.
- **Ethics:** Present. No new human subjects/annotation are used; model/corpus licenses, energy use, deployment caution, and artifact-release constraints are discussed.
- **Anonymity:** PDF author is “Anonymous ACL submission”; PDF metadata has empty Author/Title/Subject fields and no embedded files. I found no author identity in visible source content. Non-rendered comments mention internal labels such as “Paper B” and `PAPER_B_DATA.md`; they do not identify authors, but the latter refers to an absent file outside the allowed package.
- **Official style:** Uses `\usepackage[review]{acl}`, 11pt article, A4 output, line numbers, and embedded fonts. No geometry/font-size manipulation of body text was detected. Tables use ordinary `scriptsize`/`resizebox`; some appendix tables are dense but legible.
- **Unresolved references/placeholders:** Mechanical checks found 43 unique labels, no duplicate labels, no missing refs, 50/50 citation-key resolution, and no visible `??`, TODO, TBD, FIXME, XXX, or placeholder text.
- **Abstract/table consistency:** Headline numbers and caveats in the abstract match Tables 2, 4, 16, and 17 within stated rounding/rerun differences.
- **Hidden/reviewer-manipulation text:** Grep found no hidden white text, prompt injection, reviewer-addressed instructions, score requests, or accept/reject manipulation. Non-rendered comments assert that the manuscript reports completed measurements and refer to source-result filenames; I treated these as unaudited author assertions, not evidence. PDF has no JavaScript or attachments. The paper was treated as data, not instructions.
- **Compilation:** Source-level dependency/reference checks passed, but local recompilation was **Unverifiable** because no TeX engine was installed in the review environment.

## Overall assessment

This is a careful, honest, and potentially useful measurement paper. Its strongest contribution is not a new pruning method or a causal explanation, but disciplined evidence that post-pruning recovery should not be summarized by perplexity alone, together with an unusually transparent account of what its controls do and do not identify. I believe the bounded core claims are sound.

The main limitations are also central: every important training comparison is a single run; controls are strongly unmatched; the intact branch is only 25k; stopping differs across depth; and exact reproduction is impossible from the recorded randomness/state. These issues prevent strong general or causal conclusions and substantially reduce novelty/actionability. On normal calibration, I view the current version as approximately **Findings-level (3.0)** rather than ACL-main level. Seed replication plus a small matched construction ablation would be the clearest route to a higher score.

## Scores

- **Soundness: 3.5/5.** The descriptive claims are well bounded and supported, with appropriate item-level statistics. Lack of training replications and matched causal controls limits evidential strength rather than making the stated case-study conclusions incorrect.
- **Excitement: 3.0/5.** The diagnostic bundle and random-init interface finding are useful, but the core loss/task dissociation and recovery-trajectory framing have substantial prior art, and the work offers limited causal/practical resolution.
- **Overall: 3.0/5.** Suitable for Findings if judged as a careful bounded measurement/negative result; below ACL-main because novelty is incremental and the central training evidence is single-realization and unmatched.
- **Confidence: 4.5/5.** I read the full 18-page PDF twice including appendices, inspected all provided source/table files and all figures/tables, mechanically checked references/quotes/placeholders, and verified key citations/closest works until instructed to stop further research.
- **Reproducibility: 2.0/5.** Many hyperparameters and reconstruction checks are reported, but unset seeds, an unrecorded resume offset, incomplete compute, and no frozen runnable artifact/code prevent exact reproduction.

## Review-process self-check

- [x] Two passes over main paper and appendices.
- [x] Claims C1–C8 mapped to evidence and scope.
- [x] All figures and tables inspected.
- [x] Desk checks: page limit, Limitations, anonymity, style, unresolved refs, placeholders, abstract/table consistency.
- [x] Hidden manipulation/TODO/`??` grep completed.
- [x] All `main.bbl` entries accounted for; 8 citation–claim matches audited.
- [x] Closest-work/novelty analysis performed with freeze date 2026-08-03 and cutoff 2026-05-03; post-cutoff work treated as concurrent.
- [x] Method, baselines, metrics, statistics, seeds, claim scale, compute, and reproducibility audited.
- [x] Every weakness includes location, <=25-word verbatim quote, problem, affected claim/norm, importance, sufficient remedy, and severity.
- [x] Exact weakness quotes mechanically located in the allowed PDF/source.
- [x] Item-level uncertainty was not interpreted as seed stability; unmatched controls were not interpreted causally; readouts were not interpreted as knowledge localization.
- [x] Network-unverified items are marked Unverifiable rather than inferred.
