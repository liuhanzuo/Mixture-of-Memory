```yaml
review_mode: normal
soundness: 3.5
excitement: 3.0
overall: 3.0
confidence: 4.5
reproducibility: 2.5
```

# Summary and scope

This paper presents a measurement study of continued pretraining after depth pruning, centered on a single OLMo-2-1124-7B `keep14+fresh2` path. It asks a deliberately narrow question: whether improving **same-source in-domain perplexity** is sufficient to certify recovery on several knowledge-sensitive evaluations. The paper compares the intact base, an intact full32 continuation available only at 25k steps, the 200k keep14 path, a 200k non-contiguous ShortGPT-16 construction, and 200k frozen-front and fully random operating points. It evaluates answer-letter and complete-option MMLU plus zero-shot closed-book PopQA, TriviaQA, and NQ-open, and reports additional downstream trajectories and a qualitative 1B/Qwen scope check in the appendix.

The main observation is supported for the literal path studied: keep14 PPL improves from **10.826 to 10.561** between 128k and 200k while answer-letter MMLU rises only from **.3012 to .3191**, remaining far below the intact base (**.6053**); the 200k keep14 endpoint also remains below the base on PopQA (**.1415 vs .2571**), TriviaQA (**.2940 vs .6355**), and NQ-open (**.0598 vs .2050**). The paper is unusually explicit that this is not a causal, localization, deletion, convergence, or universal-recovery claim.

I view the work as a careful and useful **Findings-level measurement paper** rather than a new pruning method. Lack of a new algorithm is not itself a negative here. The main limitation is that the paper's strongest practical framing still rests on one principal training realization, without a 200k intact control or run-level uncertainty; this sharply limits how broadly the “certificate” lesson can be interpreted.

# Claims and evidence map

| ID | Claim as assessed | Main evidence | Assessment |
|---|---|---|---|
| C1 | In-domain PPL is not a sufficient certificate for the measured evaluations on the literal keep14 path. | Fig. 1; §5.1; Tables 2, 12, 16: PPL 10.826→10.561 while MMLU remains .3012→.3191 vs base .6053; all three closed-book endpoints remain far below base. | **Supported for this fixed observed path and budget.** |
| C2 | Short-horizon corpus shift is not a complete explanation for the large keep14 deficit. | §5.2/Table 2: full32 at 25k stays near base (PPL 7.670 vs 7.398; MMLU-L .588 vs .605; MMLU-C .466 vs .471; QA gaps comparatively small). | **Supported only at the available 25k horizon; not a 200k control.** |
| C3 | The letter/content MMLU contrast is interface-sensitive and content scoring has a high non-inherited baseline. | §5.2; Table 15: keep14 .3184 letter/.3832 content-norm; random .2470/.3598; paired interface differences with item bootstrap CIs. | **Supported as protocol sensitivity, not as a one-factor mechanism.** |
| C4 | The base–keep14 gap is not merely an answer-letter artifact. | Tables 2 and 16: the gap recurs on PopQA, TriviaQA, and NQ-open generation. | **Supported descriptively; no aligned per-item uncertainty and no ShortGPT QA.** |
| C5 | Endpoint quality is construction-dependent, not determined by nominal 16-layer depth alone. | Table 2: ShortGPT-16 at 200k has PPL 9.780/MMLU .474 vs keep14 10.561/.319; same nominal depth and step/token presentations. | **Supported as an operating-point contrast; causal factor attribution is not supported.** |
| C6 | Shallow-prefix observations do not establish a depth law or convergence ordering. | §6.2; Tables 3, 5, 9: unequal 83.5k/121k/124k checkpoints selected after metric inspection, PPL still decreasing, unequal FLOPs. | **Correctly bounded; no depth-law claim should be drawn.** |
| C7 | The 1B and Qwen observations provide directional context but not independent replication or universality. | Fig. 2; Tables 6–8; explicit changes in scale, retained fraction, architecture, corpus, and evaluation coverage. | **Appropriately scoped.** |
| C8 | Likelihood, target evaluation, interface, construction, budget/compute, and run-level uncertainty should be reported separately. | The failure modes exposed by Tables 2, 12, 15, 16 and the stated design confounds. | **Reasonable recommendation motivated by the case study, not itself universally validated.** |

# Strengths

1. **The central claim is narrow, falsifiable, and evidence-aligned.** The manuscript repeatedly says “on the literal observed paths” and explicitly rejects knowledge-deletion, localization, causal-factor, universal-dynamics, and beyond-budget interpretations (Abstract; §§4, 6.4, 8). This restraint materially improves soundness.

2. **The operating-point bookkeeping is excellent.** Table 2 exposes inherited/fresh blocks, trainable modules, LR, steps, nominal token presentations, evaluation interface, and missing cells in one place. The text does not disguise Random, Frozen, ShortGPT, or the 25k full32 branch as clean ablations.

3. **The evaluation-interface analysis is valuable.** The paired letter/content MMLU comparison over all **14,042** items and the fully random baseline prevent an easy but misleading interpretation of content-MMLU gains. The paper also correctly notes that prompt, candidate text, tokenization, and normalization all change together.

4. **Statistics are used with unusually clear conditioning statements.** The paper reports a paired 128k→200k MMLU gain of **1.68 pp**, 95% CI **[1.08, 2.29]**, and exact McNemar **p = 4.12×10⁻⁸**, while repeatedly distinguishing item-level uncertainty from training-run uncertainty. Table 14 likewise reports exact paired tests for the three 200k same-shape operating points.

5. **The appendix is comprehensive and mostly easy to audit.** It reports all consolidated PPL checkpoints, the 11-task keep8 trajectory, 1B trajectories, chance-adjusted recovery, marginal and paired MMLU uncertainty, scoring sensitivity, closed-book protocols, sample counts, broad groups, and all 57 MMLU subjects. Both figures and all 19 tables were inspected; captions generally state the relevant scope restrictions.

6. **The manuscript is candid about irreparable historical gaps.** It discloses unavailable seeds, the lost within-epoch loader offset after the 34.5k resume, unavailable GPU-hours, local-only evaluator commits, absent aligned closed-book predictions, and no contamination/OOD-PPL audit rather than silently imputing them.

# Major weaknesses

## M1. No run-level replication for the load-bearing path

- **Issue:** The main inference is based on one keep14 training trajectory, while every other trained construction is also one realization.
- **Why it matters:** Item-level CIs can establish that two fixed checkpoints differ on sampled evaluation items, but they cannot show that the PPL–target gap, its magnitude, or the late trajectory is robust to initialization, data order, optimizer noise, or block-selection variability. This is the main reason I would not place the current evidence at main-conference level.
- **Location + exact quote:** p. 2, lines 88–89: **“Every trained construction is a single run.”** p. 8, lines 509–515 further states that the principal path and operating points are single runs and that training seeds were not explicitly set.
- **Minimal remedy:** Replicate at least the principal keep14 run with 2–3 independently seeded runs (ideally also one matched comparator), predefine checkpoints/stopping, and report run-level dispersion for PPL, answer-letter MMLU, and at least one closed-book metric. If new training is impossible, make the headline explicitly a single-realization case report throughout and avoid any deployment-general wording.
- **Severity:** **Major.**

## M2. The intact control does not match the 200k horizon

- **Issue:** full32 is available only at 25k/6.6B presentations, whereas keep14 and the main operating points are evaluated at 200k/52.4B.
- **Why it matters:** The paper can reject a *short-horizon* corpus-shift-only account, but cannot determine whether 200k of intact continuation changes the base itself, nor cleanly separate intervention-specific failure from long-horizon data/optimization effects. This weakens C2 and any interpretation comparing 200k compressed endpoints to an uncontinued base.
- **Location + exact quote:** p. 4, lines 241–243: **“keep14, ShortGPT, frozen-front, and random-init are evaluated at 200k. full32 has an available 25k checkpoint only.”**
- **Minimal remedy:** Add a full32 checkpoint at 200k under the same data order/schedule, or at least several later intact checkpoints. Short of that, keep C2 explicitly limited to 25k and treat intact-200k behavior as unresolved in the abstract, figure, discussion, and conclusion.
- **Severity:** **Major.**

## M3. The study does not directly test a defined stopping/certification rule

- **Issue:** “certificate” is framed conceptually as sufficient PPL improvement, but no operational threshold, stopping decision, or prospectively specified proxy rule is evaluated. Moreover, the shallow endpoints were retained after inspecting target metrics.
- **Why it matters:** The experiment clearly demonstrates dissociation from base-level target performance, but it does not estimate false-certification rates or show that a real PPL-based stopping policy would have declared recovery. Without an operational rule, the practical certification framing is less sharp than the descriptive result.
- **Location + exact quote:** p. 4, lines 288–296 defines the certificate as “sufficient improvement,” while p. 2, lines 91–94 says checkpoints were selected after target metrics were inspected and **“there was no registered common stopping rule.”**
- **Minimal remedy:** Define one or more plausible rules before analysis (e.g., absolute PPL threshold, fraction of intact PPL gap recovered, or plateau criterion), then report whether/when each rule fires and the corresponding target deficits. A small retrospective sensitivity table would be a meaningful minimal experiment if clearly labeled post hoc.
- **Severity:** **Major.**

## M4. The claim's domain is narrower than “perplexity” in general

- **Issue:** The only likelihood proxy is a disjoint shard from the same Dolmino/DCLM source; there is no out-of-domain likelihood or contamination audit.
- **Why it matters:** The result establishes insufficiency of **same-source in-domain PPL**, which is useful, but cannot tell whether a broader or target-adjacent likelihood suite would be a better certificate. The title is broad, although the abstract and body usually restore the qualifier.
- **Location + exact quote:** p. 8, lines 533–534: **“PPL is in-domain, with no contamination audit or out-of-domain likelihood.”**
- **Minimal remedy:** Add at least one out-of-domain perplexity set and, if feasible, a target-adjacent likelihood measure; otherwise consistently retain “in-domain” in every headline formulation and explicitly state that the paper does not compare candidate likelihood proxies.
- **Severity:** **Major-to-moderate.**

# Minor weaknesses

## m1. Closed-book support is incomplete across operating points

- **Issue:** ShortGPT has no PopQA/TriviaQA/NQ-open cells, and aligned prediction files are unavailable for new paired intervals.
- **Why it matters:** The strongest alternative 16-layer construction cannot be compared on the generation evaluations used to rule out an answer-letter artifact; uncertainty is also weaker for those tasks.
- **Location + exact quote:** Table 2 caption, p. 6: **“Missing ShortGPT closed-book cells were not evaluated.”** Table 16 states that aligned per-item files were not consolidated.
- **Minimal remedy:** Run the three closed-book evaluations for ShortGPT and archive per-item predictions for all arms, then report paired differences/CIs.
- **Severity:** **Minor-to-moderate.**

## m2. Reproducibility is limited by unavailable seeds, compute, and public code

- **Issue:** Historical seeds and GPU-hours are unavailable; the exact keep14 loader position was lost; several evaluator commits are local-only.
- **Why it matters:** Readers can understand the protocol, but cannot exactly rerun the principal path or independently retrieve all code used for headline evaluation.
- **Location + exact quotes:** p. 12: **“Historical training seeds are unavailable.”** and **“Per-run wall time/GPU-hours and aggregate project compute are unavailable.”** p. 13: task-specific harness commits are local-only.
- **Minimal remedy:** Release the exact evaluator snapshot, configs, manifests, per-item artifacts, and a best-effort reconstruction script; provide hardware estimates where exact totals are unavailable.
- **Severity:** **Minor-to-moderate** for the scientific claim, but it drives the reproducibility score down.

## m3. Metric-normalization terminology is inconsistent

- **Issue:** Table 13 defines its normalized score using continuation **character** length, whereas Tables 15 and 18 define content/`acc_norm` normalization by continuation **token** count.
- **Why it matters:** These may intentionally be different evaluators, but the shared phrase “continuation-length-normalized” can lead readers to assume the same denominator across tasks.
- **Location + exact quote:** Table 13 caption, p. 15: **“divides candidate log-likelihood by continuation character length”**; Table 18 caption, p. 16 says **“continuation token count.”**
- **Minimal remedy:** Name the metrics explicitly as character-normalized versus token-normalized everywhere and explain why each denominator is used.
- **Severity:** **Minor.**

## m4. Presentation is clear but somewhat repetitive and space-inefficient

- **Issue:** The same caveats (single run, no causal attribution, 25k-only full32, coupled ShortGPT) recur in the abstract, introduction, figure, results, discussion, conclusion, and limitations; p. 7 is also largely blank after forced float/page breaks.
- **Why it matters:** Repetition protects against overclaiming but reduces the space available for a sharper operational analysis of certification rules.
- **Location:** Abstract; Fig. 1; §§5–8; p. 7 layout.
- **Minimal remedy:** Consolidate caveats into one protocol-boundaries box/table and use the recovered space for the operational proxy analysis requested in M3.
- **Severity:** **Minor.**

# Questions for the authors

1. What concrete PPL-based rule would have declared the keep14 model “recovered,” and at which checkpoint would it fire? Can you provide a sensitivity analysis over plausible absolute/relative/plateau thresholds?
2. Were the 128k, 153.5k, and 200k keep14 checkpoints chosen independently of target evaluations, or were all analyzed after inspecting them? Please distinguish training-time checkpoint retention from analysis-time selection.
3. Can a 200k full32 run be produced, or is there a principled reason it is unavailable? Even a smaller number of later intact checkpoints would materially sharpen the corpus-shift claim.
4. Why is character-length normalization used for BoolQ/CSQA/SIQA while token-length normalization is used elsewhere? Are all headline cells reproducible under one clearly specified evaluator version?
5. Can the ShortGPT closed-book evaluations and all per-item closed-book outputs be released/run before final publication?
6. What artifacts will actually accompany an anonymous submission? The PDF refers to an “accompanying anonymous checksum manifest,” but several critical code commits are described as local-only.

# Novelty analysis (cutoff: 2026-05-04)

I conducted four targeted searches (depth-pruning recovery/perplexity; PPL–task dissociation after healing; scratch-vs-pruned initialization; OLMo depth pruning/MMLU) and compared the manuscript to the closest cited work. Search coverage was imperfect; where external metadata or full text could not be reliably checked, I mark the item **Unverifiable** rather than infer.

1. **Gromov et al., “The Unreasonable Ineffectiveness of the Deeper Layers” (arXiv:2403.17887; ICLR 2025).** Closest for deep-layer removal, healing, and loss/task dissociation. The present paper is not novel in showing that loss recovery need not imply task recovery. Its incremental novelty is the OLMo prefix+fresh-tail case with the specific control/interface package.
2. **Kim et al., “Shortened LLaMA” (arXiv:2402.02834, 2024).** Closest for CPT trajectories and pruned-vs-scratch comparisons. Again, trajectories and initialization comparisons are prior art; this paper adds same-source intact short-horizon continuation, paired MMLU interfaces, and closed-book QA in one measurement audit.
3. **Sreenivas et al., Minitron (arXiv:2408.11796, 2024) and Wibowo et al., IteRABRe (arXiv:2503.06291, 2025).** These already study iterative recovery trajectories and downstream behavior. The manuscript's novelty is a diagnostic/reporting synthesis, not a recovery algorithm.
4. **Namburi et al. (2023), Jaiswal et al. (ICLR 2024), and Xu et al. (Findings EMNLP 2024).** These establish that compressed-model perplexity/aggregate metrics can miss knowledge, downstream, subgroup, or safety degradation. Therefore “beyond perplexity” is not novel. The narrower contribution is the literal-path bookkeeping and interface/null-baseline analysis.
5. **Kim et al., “Calibration Matters More Than Search” (arXiv:2604.24938, submitted 2026-04-27, before the cutoff).** This reinforces that pruning outcomes depend on calibration/configuration and weakens any novelty based on selection alone; it does not appear to duplicate the paper's PPL-certificate case study.

**Post-cutoff/concurrent context (not novelty-destroying under the stated rule):** Shi et al. (arXiv:2605.07271, May 8), SlimQwen (arXiv:2605.08738, May 9), Ghosted Layers (arXiv:2605.15491, May 15), Small LLMs: Pruning vs. Training from Scratch (arXiv:2606.14150, June 12), and ShortOPD (arXiv:2607.13124, July 14) appeared after 2026-05-04. They broaden mechanism, matched-token scratch comparisons, and generation recovery. Full-text comparison for all post-cutoff papers is **Unverifiable** within this review's network/time budget.

**Novelty conclusion:** modest but legitimate for a measurement study. The paper's novelty lies in the *combination and discipline of controls*, not in the underlying phenomenon or a new method. This supports moderate excitement, not automatic rejection.

# Citation audit

## Verification of all `main.bbl` entries

`main.bbl` contains **33 entries**, and all 33 are cited in the manuscript. I checked title/year/venue or arXiv identifier against DOI/Crossref/arXiv metadata where reachable.

- **Verified (metadata matched):** Alzahrani et al. 2024; Chen et al. 2025 LinearPatch (arXiv id/title); Chen et al. 2026 Prune&Comp; Deng et al. 2025 DRPruning; Gromov et al. 2025; Gupta et al. 2024; He et al. 2025 PASER; Hendrycks et al. 2021; Jaiswal et al. 2024; Joshi et al. 2017; Kim et al. 2024 Shortened LLaMA; Kim et al. 2026 calibration; Kwiatkowski et al. 2019; Lu et al. 2024; Mallen et al. 2023; Martra 2025; Men et al. 2025; Muralidharan et al. 2024; Namburi et al. 2023; OLMo Team et al. 2025; Shi et al. 2026; Siddiqui et al. 2024; Song et al. 2024; Sreenivas et al. 2024; Tang et al. 2026; Wang et al. 2024; Wibowo et al. 2025; Xia et al. 2024; Xu et al. 2024; Yang et al. 2025 Qwen3; Yang et al. 2024 LaCo; Zhang et al. 2026 ShortOPD; Zhong et al. 2025.
- **Bibliographic caveat:** LinearPatch's arXiv record was verified, but the `main.bbl` venue string “Advances in Neural Information Processing Systems” was **Unverifiable** from the metadata retrieved in this audit. The paper should ensure the final venue/year is correct.
- **Remaining author-list/page-level details:** **Unverifiable** for records without DOI-backed metadata in the completed audit; no title/arXiv-id contradiction was found.

## Citation–claim matching (7 load-bearing checks)

| Manuscript claim | Cited source(s) | Match |
|---|---|---|
| Deep-layer pruning/healing can show loss–task dissociation. | Gromov et al. | **Match.** This is central to the cited work and appropriately acknowledged as prior art. |
| Shortened LLaMA reports CPT/LoRA trajectories and pruned-vs-scratch initialization. | Kim et al. 2024 | **Match.** |
| Minitron studies pruning/distillation trajectories and initialization choices. | Sreenivas et al. 2024 | **Match**, though the paper is a broader pipeline than this sentence alone suggests. |
| IteRABRe alternates block removal and recovery and reports weak MMLU recovery. | Wibowo et al. 2025 | **Match** based on title/abstract-level metadata; full figure-level verification **Unverifiable**. |
| Compression metrics/perplexity can miss parametric-knowledge or downstream degradation. | Namburi et al. 2023; Jaiswal et al. 2024 | **Match.** |
| Compression can have divergent safety effects beyond perplexity. | Xu et al. 2024 | **Match.** |
| Multiple-choice/MMLU results are sensitive to first-token interfaces, evaluation details, or answer order. | Wang et al. 2024; Alzahrani et al. 2024; Gupta et al. 2024 | **Match.** |

No citation was found to be clearly irrelevant to the associated claim. The related-work section is more candid than average about antecedents.

# Method, experimental, and reproducibility audit

- **Method/constructions:** Exact retained blocks, copied modules, trainable sets, parameter count (keep14: **4.060B**), ShortGPT indices `[0–12,16,17,31]`, and LRs are reported. Random/Frozen/ShortGPT confounds are explicitly listed.
- **Minimal experiment:** The minimum convincing experiment for the narrow descriptive claim is a within-run PPL+target trajectory plus an intact/reference endpoint and non-letter target evaluation; the paper supplies this. The minimum experiment for a broader certification claim would additionally require replicated runs, an operational stopping rule, and a long-horizon intact control; these are missing (M1–M3).
- **Operating points:** Base, full32-25k, keep14-200k, ShortGPT-200k, Frozen-200k, Random-200k are clear. Shallow keep8/10/12 are explicitly not matched. Same nominal step/token budget does not imply equal FLOPs, which is acknowledged.
- **Metrics/protocols:** MMLU-L, MMLU-C, PopQA containment, TriviaQA/NQ exact match, PPL aggregation, decoding, normalization, prompts, sample counts, and no-BOS/no-chat-template choices are largely specified. The character-vs-token normalization naming issue remains.
- **Seeds/statistics:** Training seeds are unavailable. Item-level Wald CIs, exact McNemar tests, and paired bootstrap with **10,000** resamples/seed **1234** are reported. No run-level uncertainty is available.
- **Scope:** English, mainly one 7B OLMo family/mixture/recipe; 1B and Qwen are directional only. No latency/throughput/memory/energy frontier.
- **Compute:** Steps and nominal presentations are given (**200k = 52.4B**, **25k = 6.6B**), but realized FLOPs, wall time, GPU type/count/hours, and total project compute are absent.
- **Reproducibility:** Optimizer schedule and data sizes/checksum fragments are detailed, but exact keep14 reproduction is blocked by the lost loader offset; critical evaluator commits are not public; some prediction files are absent. Hence reproducibility is below average despite strong documentation.

# Figure and table audit

- **Figure 1:** Clear and faithful to Table 2/12. It visibly labels the single-run status, 25k full32 limit, and coupled ShortGPT comparison. No unsupported causal visual encoding was found.
- **Figure 2:** Correctly presented as qualitative context, not replication. The plotted values agree with Tables 6–7.
- **Tables 1–2:** Useful nearest-work and protocol matrices. Table 1's row-level judgments for all prior work are partly **Unverifiable** without full-text reinspection of every source, but no obvious contradiction was found.
- **Tables 3–10:** Numerically coherent with the claims; depth/budget caveats are stated. Above-chance recovery calculations spot-check correctly (e.g., keep14 MMLU ≈ **19.4%**).
- **Tables 11–16:** Statistical conditioning and interface distinctions are unusually explicit. Table 13 terminology should be disambiguated as noted.
- **Tables 17–19:** Group/sample totals and all 57 subjects are supplied; the paper appropriately avoids multiplicity-adjusted domain claims.
- **Mechanical numeric spot checks:** 200k nominal tokens = 200,000×128×2,048 = **52.4288B**; 25k = **6.5536B**. Recovery and reported endpoint differences are arithmetically consistent up to rounding.

# Desk, style, anonymity, ethics, and integrity checks

- **Page/format:** 17-page A4 PDF in ACL review style. Main text ends on p. 7; Limitations/Ethics occupy p. 8; references pp. 9–10; appendix pp. 11–17. This appears consistent with an 8-page main-text limit plus unlimited references/appendix, but compliance with the exact target ARR cycle rules is **Unverifiable** because no venue policy file was permitted/read.
- **Limitations:** Present and substantive.
- **Ethical considerations:** Present; no human-subject data or annotators; model/data risks, energy, and licensing are discussed.
- **Anonymity:** PDF metadata has an empty Author field and the title page says “Anonymous ACL submission.” No author affiliation or obvious identity leak was found. Local absolute environment paths and “origin/main” are poor artifact style but do not obviously deanonymize the authors.
- **Official style:** `acl` review mode is used; fonts are embedded; no overfull boxes or unresolved references/citations were found in the frozen build. Numerous underfull-box warnings and a sparse p. 7 are cosmetic.
- **Abstract consistency:** The ~202-word abstract's numerical and scope statements match the body/tables.
- **Placeholders/TODO/unresolved refs:** None found in the frozen manuscript/source used for review.
- **Hidden/injection/reviewer manipulation:** I searched extracted PDF text, source, PDF metadata/strings, hyperlinks, and rendered pages. No hidden white text, prompt injection, reviewer-directed scoring instruction, or acceptance solicitation was found. The paper was treated as data, not instructions.

# Scores and recommendation

- **Soundness: 3.5/5.0.** The narrow fixed-path observation is well supported and carefully scoped, with strong protocol disclosure and appropriate item-level statistics. Lack of replicated runs, a 200k intact control, and an operational certification rule prevents a higher score.
- **Excitement: 3.0/5.0.** The study is useful and timely, especially as a warning about proxy reporting, but the phenomenon is substantially anticipated by prior compression work and the novelty is mainly a well-designed combination of controls/interfaces.
- **Overall: 3.0/5.0.** **Findings-level.** I would support acceptance to Findings after the claims remain narrow and the authors clarify the operational proxy framing and reproducibility package. I would not currently recommend main-conference acceptance because the load-bearing training evidence is one realization with no long-horizon intact control.
- **Confidence: 4.5/5.0.** I read the full paper and appendix twice, inspected all figures/tables, checked the source and build diagnostics, audited all 33 references, and performed targeted novelty searches. A few external full-text/venue details remain Unverifiable.
- **Reproducibility: 2.5/5.0.** Documentation is strong, but exact rerunning is materially blocked by unavailable seeds/loader offset/compute records, local-only evaluator commits, and missing per-item closed-book artifacts.

# Suggested revision priorities

1. Replicate keep14 and add a 200k full32 control if at all feasible.
2. Operationalize “certificate” with explicit proxy thresholds/stopping rules and report false-certification behavior.
3. Run ShortGPT closed-book QA and archive all per-item predictions.
4. Release the exact evaluator snapshot/configs/manifests and clearly distinguish token- from character-normalized metrics.
5. Shorten repeated caveats and use the space for the operational analysis.

# Review-process self-check

- Two complete passes, including all appendices: **done**.
- Claims C1–C8 mapped to evidence and bounded: **done**.
- Desk/page/Limitations/anonymity/style/injection/TODO checks: **done**; exact venue page-policy status marked **Unverifiable**.
- 5+ key numbers checked: **done** (including 10.826, 10.561, .3191, .6053, 9.780, .4739, 52.4288B, 6.5536B, 1.68 pp).
- Every `main.bbl` entry checked and citedness verified; 7 citation–claim matches audited: **done**, with unresolved metadata explicitly marked **Unverifiable**.
- 3–5 novelty searches and cutoff 2026-05-04 applied: **done**.
- Method/minimal experiment/operating points/metrics/seeds/stats/scope/compute/reproducibility and all figures/tables audited: **done**.
- Weaknesses include issue, importance, exact location/quote, minimal remedy, and severity: **done**.
- Mechanical quote verification: all quoted manuscript strings above were matched against normalized frozen source/PDF text before saving. “Missing X” assertions were checked against the methods, tables, limitations, appendix, and build log.
