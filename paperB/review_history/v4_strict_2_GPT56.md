---
review_mode: strict
soundness: 3.0
excitement: 2.5
overall: 2.5
confidence: 4.5
reproducibility: 2.0
---

# Paper Summary

This paper presents an observational OLMo-2 depth-pruning case study rather than a new pruning algorithm. Its principal intervention keeps the first 14 of 32 pretrained OLMo-2-7B blocks, appends two freshly initialized blocks, and continues pretraining for 200k optimizer steps. It evaluates held-out in-domain perplexity, zero-shot answer-letter MMLU, a coupled complete-option MMLU interface, three no-retrieval closed-book QA tasks, and a broader likelihood suite. Controls include the intact base, one intact full32 checkpoint continued to 25k, a frozen-prefix operating point, a fully random same-shape operating point, and a ShortGPT-16 construction that retains 16 non-contiguous pretrained blocks.

The main empirical record is useful and unusually candid about its limits: keep14 reaches PPL 10.561 versus 7.398 for the intact base, but only .319 answer-letter MMLU versus .605; the complete-option score is higher (.383), yet random initialization reaches .360, exposing a large protocol floor; and ShortGPT-16 reaches PPL 9.780 and MMLU .474, showing that the tested prefix-plus-fresh-tail endpoint is not shared by every 16-layer construction. The paper repeatedly states that these are single realizations, that full32 is only a 25k short-horizon control, and that ShortGPT/random/frozen comparisons are confounded.

My assessment is that the paper contains a careful descriptive case study and a good reporting lesson, but not yet a sufficiently identified or statistically replicated result for ACL main. The central path statement is literal and defensible for the realized run; any stronger interpretation of a recovery dynamic is limited by one training realization, sparse late checkpoints, no matched 200k intact control, unset seeds, and an unrecorded resume offset. The scientific increment over prior loss--task, trajectory, initialization, and construction studies is also narrow. I therefore place it below Findings level despite valuing the transparency and negative evidence.

# Claims and Evidence Map

For each claim I state the minimum sufficient evidence first, then the paper's actual evidence.

- **C1 — In the realized keep14 run, likelihood improves substantially while target capabilities remain far below the intact model.**
  - *Minimum sufficient evidence:* a correctly reconstructed model; fixed held-out likelihood data; identical evaluation interfaces across checkpoints/base; at least multiple within-run checkpoints; all task counts and metrics disclosed.
  - *Actual evidence:* PDF pp. 4--5, Table 2 and Figure 1; Appendix Tables 4, 10, 13, 16--17 and evaluation-integrity text on pp. 13--14. The numerical descriptive claim is supported for this run.

- **C2 — A 25k intact full32 branch makes short-horizon corpus shift insufficient to explain the large keep14 gap.**
  - *Minimum sufficient evidence:* same corpus order, optimizer/schedule, and token exposure through the horizon being excluded; ideally replicated branches.
  - *Actual evidence:* full32@25k is near base on PPL/MMLU/closed-book (Table 2; Appendix Tables 4, 16--18), while keep14's large gap is reported at 200k. This supports only a 25k short-horizon bound, not exclusion of corpus/order/optimization effects accumulated over 200k.

- **C3 — MMLU scoring interface materially changes the observed score, but complete-option accuracy is not a direct knowledge measure.**
  - *Minimum sufficient evidence:* paired scoring of the same items with one factor changed at a time, item-level uncertainty, and negative controls that diagnose generic language-model preference.
  - *Actual evidence:* all 14,042 items are scored under letter and content protocols (Appendix Table 16), and random-init content-normalized accuracy is .3598 versus .3832 for keep14. However, prompt, candidate, tokenization, and normalization all change together and there is no paired interface uncertainty. Supported as an exploratory protocol diagnostic, not as a readout localization.

- **C4 — Closed-book deficits argue against an answer-letter-only account.**
  - *Minimum sufficient evidence:* the same generative protocol across arms, adequate sample sizes, uncertainty/significance, and preferably several training seeds.
  - *Actual evidence:* PopQA/TriviaQA/NQ-open gaps are large under one shared zero-shot protocol (Table 2; Appendix Table 17), but no confidence intervals or paired tests are reported and all arms are single training runs. Directionally persuasive, not seed-stable.

- **C5 — PPL is not a sufficient one-number summary across the available interventions.**
  - *Minimum sufficient evidence:* either matched-PPL comparisons or a deliberately narrow existential counterexample with confounds stated.
  - *Actual evidence:* random-init has lower PPL than frozen-front (11.498 vs. 12.797) but lower answer-letter MMLU (.247 vs. .262), while ShortGPT dominates keep14 on both. Figure 2 explicitly labels the points heterogeneous and not matched. This supports a descriptive non-sufficiency claim for the observed operating points, not global decorrelation or a causal dissociation.

- **C6 — Nominal 16-layer depth does not uniquely determine the observed endpoint.**
  - *Minimum sufficient evidence:* at least two same-depth constructions trained/evaluated under the same budget; to identify a responsible factor, a factorial set matching inherited count, selected blocks, final-block retention, and fresh tails.
  - *Actual evidence:* keep14 and ShortGPT both complete 200k and differ greatly (Table 3; Appendix Tables 4--5, 10, 16), but four structural factors are coupled. The existential endpoint claim is supported; no factor attribution is supported.

- **C7 — The two observed prefix trajectories differ over their observed intervals.**
  - *Minimum sufficient evidence:* dense checkpoints and replicated seeds if the claim is about a stochastic dynamic; one run suffices only for a literal realized-path description.
  - *Actual evidence:* keep8 and keep14 checkpoints plus item-paired keep14 128k--200k estimates (Figure 4; Appendix Tables 6 and 13). This establishes different recorded paths, but not a depth effect, typical trajectory, plateau, or convergence law.

- **C8 — Recovery is heterogeneous across MMLU groups.**
  - *Minimum sufficient evidence:* prespecified groups, group-wise uncertainty or interaction tests, and training-seed replication for a general heterogeneity claim.
  - *Actual evidence:* sample-weighted group accuracies and chance-adjusted ratios (Figure 3; Appendix Table 18) with no group-level CIs or formal interaction tests. The table is descriptive only.

- **C9 — Layer-wise readout depths are background and do not locate stored knowledge or explain keep14/ShortGPT.**
  - *Minimum sufficient evidence for localization:* causal interventions on matched states/layers, multiple seeds/models, and readout controls.
  - *Actual evidence:* descriptive logit/probe thresholds on intact models (Appendix pp. 15, 17--18), explicitly disclaimed as causal evidence. The paper's bounded non-causal wording is appropriate.

# Strengths

## S1. Exceptional scope discipline and disclosure

The manuscript repeatedly distinguishes observation from identification. Examples include the full32 horizon caveat (PDF p. 4, lines 335--340), the confounded ShortGPT comparison (p. 6, lines 448--466), and the single-run/statistics caveat in Limitations (p. 8, lines 582--591). This is materially better than presenting the same data as causal ablations.

## S2. Broad control and interface bundle for one frozen case study

Table 2 combines PPL, two MMLU interfaces, and three closed-book tasks; Appendix Tables 10--18 expose the broader suite, exact item counts, paired MMLU comparisons, metric sensitivity, and subject/group detail. The random-init content result is especially valuable because it prevents an otherwise tempting but unsupported “knowledge recovered under content scoring” conclusion.

## S3. Numerical reporting is internally coherent

I mechanically checked headline arithmetic and table consistency. Examples: 10.561/7.398 = 1.42755 (reported 1.428x); keep14 above-chance MMLU recovery is approximately 19.4%; ShortGPT recovery is approximately 63.0%; and the group recovery values in Figure 3 agree with Appendix Table 18 up to rounding. The PPL merge formula is correct, and the paper states that all task cells have full counts and no NaNs (Appendix p. 14, lines 1073--1086).

## S4. The paper correctly separates item uncertainty from training-run uncertainty

Appendix Table 15 gives exact McNemar tests and paired-bootstrap intervals on aligned items, and the text explicitly says these are conditional evaluation-item intervals rather than seed variance (Appendix p. 12, lines 960--968; Limitations p. 8). This distinction is often missed.

## S5. The knowledge/readout/causal boundary is mostly handled responsibly

The main text calls the MMLU content protocol exploratory and confounded, and Appendix C says its readouts are not causal storage localizations. The discussion explicitly requires structure-isolation and causal intervention before anatomical claims (PDF p. 8, lines 539--547).

## S6. Every rendered figure/table was legible and substantively captioned

I inspected all 18 PDF pages, including six figures and Tables 1--22. Captions generally state the relevant matching limitations, units, and intended scope; no clipped figure, unresolved reference, or unreadable main-text graphic was found.

# Weaknesses

## W1. The recovery-path evidence is one stochastic realization, so it cannot support a stable dynamic beyond a literal trace — **Major**

- **Location:** Abstract; Section 6.2; Limitations; Appendix B.1 (PDF p. 1, lines 14--18; p. 6, lines 432--445; p. 8, lines 582--591; p. 14, lines 1022--1029).
- **Exact quote (9 words):** “The supported claim is only that the two observed trajectories differ”
- **Problem:** Even this deliberately narrow statement is vulnerable to being read as a phenomenon rather than a record: keep14, keep8, ShortGPT, frozen, and random are each one training realization; seeds were unset; keep14 resumed with an unrecorded data-loader offset; and the late keep14 path has only three checkpoints (128k, 153.5k, 200k). Item-level CIs only condition on those realized checkpoints.
- **Affected claim/norm and why it matters:** C1/C7 and the title/contribution “Observed Recovery Paths.” A single run is sufficient to document *what happened in that run*, but not to infer a characteristic recovery shape, delayed-vs-absent capability recovery, depth-dependent trajectory, or seed-stable separation. The closest Gromov antecedent reports a three-seed fine-tuning ablation, so this is not merely a theoretical concern.
- **Sufficient remedy:** Repeat at least the principal keep14 run for 3 independent seeds with fixed, recorded seeds/data order; evaluate a prespecified checkpoint grid; plot between-run intervals for PPL and target tasks. If this is infeasible, retitle/reframe throughout as an execution trace and remove language such as “recovery varies” or “trajectories differ” wherever it implies a stochastic regularity.

## W2. full32@25k excludes only early drift; it cannot adjudicate the 200k keep14 endpoint — **Major**

- **Location:** Abstract; Sections 3.3 and 5.3; Limitations (PDF p. 1, lines 18--21; p. 3, lines 252--258; p. 6, lines 385--394; p. 8, lines 586--591).
- **Exact quote (13 words):** “it does not identify what would happen to the intact model through 200k steps.”
- **Problem:** The main target observation is keep14@200k, whereas the intact branch stops at 25k. It therefore cannot rule out long-horizon corpus adaptation, forgetting, schedule effects, or interactions with the resumed data order. It also lacks a same-depth unpruned counterfactual because such a model does not exist by construction.
- **Affected claim/norm and why it matters:** C2 and any interpretation that attributes the 200k gap specifically to the structural intervention rather than structure-plus-long-horizon training. The current control supports the exact narrow exclusion “large damage is not already induced in intact OLMo by 25k on this corpus,” and no more.
- **Sufficient remedy:** Continue full32 to 200k under matched token order/schedule, or at minimum to the keep14 checkpoint grid; report deltas from base with seed replication. Until then, confine the claim to the explicit 25k horizon in every summary and avoid using full32 as evidence about the late endpoint.

## W3. The paper's main baseline contrasts are operating points, not clean ablations; the responsible mechanism remains unidentified — **Major**

- **Location:** Sections 3.3, 5.3, 6.3; Limitations (PDF p. 4, lines 259--266; p. 5, lines 372--384; p. 6, lines 448--466; p. 8, lines 592--599).
- **Exact quote (10 words):** “These are operating points, not factor-isolating ablations.”
- **Problem:** Random-init changes initialization of all modules and uses a 5x higher peak LR; frozen-front changes the trainable parameter set; ShortGPT changes inherited count (16 vs. 14), contiguity, retention of original block 31, and the presence of fresh tails. Thus the impressive numerical gaps do not isolate initialization, inheritance, adaptation, final-block retention, or selection policy.
- **Affected claim/norm and why it matters:** C5/C6 and the scientific value beyond “different recipes yield different outcomes.” The existential claim is valid, but the paper cannot explain why ShortGPT is better or what pretrained inheritance contributes relative to scratch. Because the proposed contribution is a “control bundle,” confounded controls sharply limit its diagnostic payoff.
- **Sufficient remedy:** Add a minimal matched factorial set at 16 layers: (i) same 14 prefix blocks plus two inherited original tail blocks; (ii) ShortGPT-selected 14 plus two fresh blocks; (iii) contiguous 16 inherited blocks; (iv) same-shape random with the inherited-run LR and, separately, a learning-rate sweep; (v) train-all vs. frozen with matched optimizer groups. Three seeds for the decisive contrasts would permit causal interpretation.

## W4. The PPL--capability conclusion is descriptive over heterogeneous, in-domain measurements and is not a matched dissociation — **Major**

- **Location:** Section 5.2 and Discussion; Limitations (PDF p. 5, lines 361--371; p. 7, lines 484--502; p. 9, lines 602--606).
- **Exact quote (14 words):** “The plot is neither a matched-PPL nor a matched-compute analysis.”
- **Problem:** PPL is measured on a held-out shard from the same Dolmino/DCLM source used for continued pretraining, while the compared points vary in architecture, LR, trainable modules, and checkpoint budget. No compressed arm reaches base PPL. The ordering reversal between random and frozen is real, but it does not quantify how much capability differs at matched distributional fit or out of domain.
- **Affected claim/norm and why it matters:** C5 and the central reporting proposal. The evidence rules out PPL as a complete label for these *particular interventions*, but it cannot establish a recovery-law dissociation, global weak correlation, or capability deficit conditional on equal language-model quality.
- **Sufficient remedy:** Evaluate out-of-domain PPL and contamination-controlled likelihood; compare checkpoints/arms at interpolated matched PPL or matched tokens/FLOPs; fit within-arm capability-vs.-loss curves with replicated seeds. Keep the present reversal as a descriptive counterexample, not the primary scientific identification.

## W5. Knowledge and readout interpretations remain underidentified despite careful caveats — **Major**

- **Location:** Sections 3.4, 5.2, 7; Appendix C (PDF p. 4, lines 268--282 and 341--350; p. 8, lines 503--518 and 534--547; pp. 15, 17--18).
- **Exact quote (17 words):** “they do not isolate mapping loss.”
- **Problem:** Letter versus content MMLU simultaneously changes prompt, candidate string, tokenization, and normalization. The “fluency floor” is inferred from one random operating point whose LR and all lexical modules differ. Closed-book QA adds generation and normalization effects. Appendix logit-lens/probe results use different tasks and intact models. These measurements cannot separate stored subject knowledge, option plausibility, answer-symbol mapping, calibration, or causal location.
- **Affected claim/norm and why it matters:** C3/C4/C9 and phrases such as “knowledge-sensitive,” “content-to-symbol or readout contribution,” and “factual recall.” The paper is largely appropriately cautious, but the conceptual vocabulary can still encourage a knowledge-loss interpretation that the designs do not identify.
- **Sufficient remedy:** Use a crossed interface experiment on identical stems/options: letter vs. full option under the same prompt, length-normalization crossed independently, answer-order randomization, and calibrated PMI/null-prompt scoring; add paired item analysis. For causal localization, intervene on matched hidden states/layers or omit Appendix C from this paper. Label the current tasks behavioral proxies, not knowledge measures.

## W6. Statistical treatment is strong for selected MMLU item contrasts but incomplete for other headline claims — **Minor**

- **Location:** Table 2, Figure 3, Appendix Tables 13--18, Limitations (PDF pp. 5--7 and 16--17).
- **Exact quote (7 words):** “Only selected answer-letter contrasts have paired intervals.”
- **Problem:** No uncertainty is reported for PopQA/TriviaQA/NQ-open, content-MMLU protocol differences, broad-group heterogeneity, PPL differences, or the ShortGPT--keep14 gap. Wald intervals around marginal MMLU accuracy do not address paired comparisons or seed variance; multiple subject examples are selected descriptively without multiplicity control.
- **Affected claim/norm and why it matters:** C3/C4/C6/C8. Several qualitative conclusions are likely directionally robust because gaps are large, but readers cannot assess evaluation sampling error, paired dependence, or whether domain differences exceed noise.
- **Sufficient remedy:** Provide paired bootstrap CIs for all same-item task/protocol contrasts, bootstrap group recovery and interaction contrasts, token/block bootstrap uncertainty for PPL, and clearly separate these from between-seed intervals.

## W7. Reproducibility is materially below archival standard — **Major**

- **Location:** Appendix B.1 and Limitations (PDF p. 9, lines 602--611; pp. 13--14, lines 985--1042).
- **Exact quote (10 words):** “Runs do not set an explicit random seed”
- **Problem:** The paper explicitly reports no frozen runnable artifact, no exact project-wide compute total, unset training seeds, and an unrecoverable resume data offset. It lists code/configs/environment locks/checkpoint hashes/prediction files as material an anonymous artifact *should* include rather than as material currently supplied.
- **Affected claim/norm and why it matters:** Reproducibility of all central training results and auditability of a single-run case study. When the contribution is a particular observed path, inability to reproduce that path is especially consequential.
- **Sufficient remedy:** Release anonymized code/configs, dependency/container lock, data manifests/order semantics, explicit seeds, checkpoint hashes, per-item predictions, and the exact resume protocol. Report per-run wall time/GPU-hours and estimated FLOPs even if failed exploratory runs remain unaccounted.

## W8. Bibliography metadata contains several concrete archival errors/omissions — **Minor**

- **Location:** References, PDF pp. 9--11; `main.bbl`.
- **Exact quote (12 words):** “ShortGPT: Layers in large language models are more redundant than you expect.”
- **Problem:** The cited ShortGPT entry is left as a 2024 arXiv preprint and omits the ACL Findings 2025 archival version and its expanded author list; WinoGrande is listed as 2021 *Communications of the ACM* rather than its 2020 AAAI paper; Dolma is left as an arXiv preprint despite ACL 2024 publication; multiple archival records lack DOI/pages. LinearPatch is labeled NeurIPS although my metadata checks found only the arXiv record, so that venue is **Unverifiable** from the completed checks.
- **Affected claim/norm and why it matters:** Citation accuracy and the required complete bibliography audit. These do not overturn the experiments but weaken scholarly reliability.
- **Sufficient remedy:** Regenerate the bibliography against ACL Anthology/Crossref/arXiv canonical records; update archival versions, years, author lists, venues, DOI, and pages; mark genuinely preprint-only work as such.

# Questions That Could Change the Score

1. Do independent keep14 seeds reproduce (a) the slow late MMLU increase, (b) the large final base gap, and (c) the random/frozen ordering reversal? Three fixed-seed runs with checkpoint curves could raise soundness materially.
2. Is there a full32 checkpoint beyond 25k, or can one be run to 200k with the same token order and schedule? A matched long-horizon control would sharply improve attribution.
3. Can the authors provide one matched construction that isolates fresh-tail versus inherited-tail effects, and one random-init run at the inherited-run learning rate? These are the minimum controls needed to interpret the ShortGPT and random comparisons.
4. Under a fully crossed, same-prompt MMLU interface design, how much of the letter/content gap remains after independently controlling answer representation, candidate length normalization, and answer order?
5. Can all headline task gaps receive paired item bootstrap intervals, and can between-seed variation be reported separately?

# Non-scoring Suggestions and Typos

- Rename “Recovery varies across knowledge domains” to “Descriptive MMLU group differences” unless group interaction tests and seed replication are added.
- Avoid “surface,” “reasoning,” and “in-context” taxonomies in Appendix Table 11 unless these labels are justified; they mix task format and hypothesized competence.
- The Appendix Table 19 caption says all listed tasks are likelihood scored, but LAMBADA is described as greedy final-word exact accuracy; clarify the distinction.
- Report total training tokens as well as nominal steps, and explicitly quantify possible duplicate exposure caused by the resumed iterator.
- State the precise ShortGPT block-influence formula and whether selection used any evaluation-related data; “cosine block influence on 128 Dolmino windows” is not enough for exact reproduction.
- Consider moving Appendix C to supplementary material for another paper: it is carefully disclaimed and does not support the central recovery claims, while adding conceptual distraction.
- Add accessible vector legends/markers for grayscale printing; Figure 3 relies substantially on color, although labels and captions make it recoverable.

# Score Reasons

## Soundness: 3.0 / 5

The numerical descriptive claims for the frozen runs are generally sound, internally consistent, and transparently bounded. However, the central evidence is one unseeded realization, the intact control is horizon-mismatched, the main controls are confounded operating points, and most non-MMLU claims lack uncertainty. This is reliable as a case record, not as an identified recovery phenomenon.

## Excitement: 2.5 / 5

The random-init content floor, closed-book bundle, and unusually explicit bookkeeping are useful. Yet prior work already contains loss--task dissociation, retraining trajectories, random/scratch comparisons, and construction sensitivity; the paper itself correctly concedes this. The remaining novelty is the particular OLMo/control/interface combination and reporting proposal, which is informative but modest.

## Overall: 2.5 / 5

I calibrate 4.0 as ACL main and 3.0 as Findings. The paper is below Findings because its core scientific increment is narrow and the single-run/unmatched/confounded design prevents stronger conclusions. I considered 3.0 because the manuscript is exceptionally honest and the descriptive artifact is useful, but strict calibration requires the lower bin: the missing seed replication and matched long-horizon/factor-isolating controls are claim-linked, not cosmetic.

## Confidence: 4.5 / 5

I read the full 18-page PDF twice, inspected every rendered figure/table, checked source line anchors and arithmetic, audited all 50 cited `main.bbl` entries with available metadata, and examined the closest related/concurrent papers available before the instructed research stop. Residual uncertainty concerns external metadata/venue status for entries marked Unverifiable and the absence of runnable artifacts.

## Reproducibility: 2.0 / 5

The paper gives substantial architecture, optimizer, dataset-size, metric, and reconstruction detail, but exact reruns are prevented by unset seeds and an unrecorded resume offset; the paper reports no frozen runnable artifact or exact project-wide compute total and does not claim that the recommended code/config/environment/hash/prediction bundle is currently supplied.

# Limitations, Ethics, and Desk-Reject Risks

## Limitations and ethics

The exact unnumbered **Limitations** section is present and unusually complete. It covers single runs, item-vs-seed uncertainty, unequal checkpoints, full32@25k, confounded controls, in-domain PPL, no contamination audit, missing efficiency/compute, unset seeds, resume offset, and non-causal readouts. **Ethical Considerations** is present and addresses capability regression, energy use, licenses, no new human subjects, and artifact-release responsibilities. I found no unaddressed acute human-subject or privacy issue.

## Desk/page/style/anonymity/formats

- Rendered PDF: 18 pages, A4, ACL review style, anonymous author line.
- Main text ends on PDF p. 8; Limitations/Ethics occupy pp. 8--9; references pp. 9--11; appendix begins p. 12. This appears within the standard long-paper body allocation, subject to the venue's exact treatment of mandatory Limitations/Ethics.
- No author/institution identity was found in the manuscript source or PDF metadata.
- No unresolved `??`, TODO, TBD, FIXME, undefined reference, missing citation marker, or clipped object was found.
- All fonts are embedded; PDF has no JavaScript/forms.
- Potential desk risk: none obvious. The appendix is long but begins after references; A4 and the bundled ACL review style are consistent.

## Prompt injection / reviewer manipulation audit

I searched the allowed source and rendered text for hidden/white/tiny text, negative spacing, reviewer instructions, score requests, prompt-injection phrases, and suspicious PDF objects. No reviewer manipulation or hidden instruction was found. The source comments contain ordinary build notes only. PDF line numbers and colored hyperlinks are style-generated, not hidden content.

# Citation Audit

## Audit method and status legend

I audited every one of the 50 actually cited `main.bbl` entries against the available canonical metadata from Crossref, OpenAlex, arXiv API, ACL/venue DOI records, or the cited webpage. **Verified** means identity and core metadata were confirmed; **Metadata error** means the work exists but the BBL has a concrete year/venue/author/version error or important archival omission; **Unverifiable** means network/source checks did not establish the claimed metadata. Network failure was not converted to “Not found.”

## Per-entry audit (50/50)

1. Alzahrani et al. 2024, *When Benchmarks are Targets* — **Verified** (ACL 2024 DOI matched).
2. Belrose et al. 2023, *Tuned Lens* — **Verified** (arXiv 2303.08112).
3. Bisk et al. 2020, PIQA — **Verified** (AAAI 2020 record).
4. Chen et al. 2025, LinearPatch — **Metadata error / Unverifiable venue**: title/authors/arXiv 2505.24680 verified; claimed NeurIPS venue not established by completed checks.
5. Chen et al. 2026, Prune&Comp — **Verified with archival omission** (AAAI 2026 DOI exists; BBL links the 2025 arXiv version rather than DOI/pages).
6. Chuang et al. 2024, DoLa — **Verified** as ICLR 2024; arXiv first posted 2023.
7. Clark et al. 2019, BoolQ — **Verified**.
8. Clark et al. 2018, ARC — **Verified** (arXiv identity); archival venue metadata absent in BBL.
9. Dai et al. 2022, Knowledge Neurons — **Verified** (ACL 2022 DOI).
10. Deng et al. 2025, DRPruning — **Verified** (ACL 2025 DOI); BBL omits DOI/pages.
11. Elhoushi et al. 2024, LayerSkip — **Verified** (ACL 2024 DOI).
12. Geva et al. 2021, FFN key-value memories — **Verified** (EMNLP 2021 DOI).
13. Gromov et al., *Unreasonable Ineffectiveness* — **Verified** as ICLR 2025; cite key says 2024 but rendered year 2025 is appropriate.
14. Gupta et al. 2024, answer order — **Verified** (arXiv 2406.19470).
15. He et al. 2025, PASER — **Verified** (arXiv 2502.12594).
16. Hendrycks et al. 2021, MMLU — **Verified** (ICLR 2021).
17. Jaiswal et al. 2024, *Truth is Rarely Pure* — **Verified** as ICLR 2024; arXiv first posted 2023.
18. Joshi et al. 2017, TriviaQA — **Verified** (ACL 2017 DOI).
19. Kim et al. 2024, Shortened LLaMA — **Verified** (arXiv 2402.02834); later title/version details should be normalized.
20. Kim et al. 2026, calibration matters — **Verified** (arXiv 2604.24938; posted 2026-04-27).
21. Kwiatkowski et al. 2019, Natural Questions — **Verified** (TACL DOI).
22. Lu et al. 2024, Reassessing Layer Pruning — **Verified** (arXiv 2411.15558).
23. Mallen et al. 2023, PopQA paper — **Verified** (ACL 2023 DOI).
24. Martra 2025, *Fragile Knowledge* — **Verified as preprint** (arXiv 2512.22671 / TechRxiv); preprint-only.
25. Men et al., ShortGPT — **Metadata error**: BBL gives 2024 arXiv metadata and eight authors; archival ACL Findings 2025 record has nine authors, including Qianhao Yuan, and DOI 10.18653/v1/2025.findings-acl.1035.
26. Meng et al. 2022, causal tracing/ROME — **Verified** (NeurIPS 2022 identity).
27. Mihaylov et al. 2018, OpenBookQA — **Verified** (EMNLP 2018 DOI); DOI/pages omitted.
28. Muralidharan et al. 2024, compact models — **Verified** (NeurIPS 2024 identity).
29. Namburi et al. 2023, Cost of Compression — **Verified** (Findings EMNLP DOI).
30. nostalgebraist 2020, Logit Lens — **Verified at cited webpage identity; Unverifiable access metadata** because the page returned rate limiting during direct check.
31. OLMo Team et al., *2 OLMo 2 Furious* — **Minor metadata ambiguity**: arXiv ID 2501.00656 was first posted 2024-12-31; BBL uses 2025. Work identity verified.
32. Paperno et al. 2016, LAMBADA — **Verified** (ACL 2016 DOI).
33. Sakaguchi et al., WinoGrande — **Metadata error**: BBL says 2021 *Communications of the ACM*; canonical benchmark paper is AAAI 2020, DOI 10.1609/aaai.v34i05.6399.
34. Sap et al. 2019, SocialIQA — **Verified** (EMNLP-IJCNLP 2019 DOI).
35. Shi et al. 2026, decision transitions — **Verified as preprint** (arXiv 2605.07271; 2026-05-08); concurrent by the stated cutoff.
36. Siddiqui et al. 2024, deeper look — **Verified** (arXiv 2407.16286).
37. Soldaini et al., Dolma — **Metadata error/archival omission**: BBL lists only arXiv 2402.00159, while ACL 2024 archival DOI 10.18653/v1/2024.acl-long.840 exists.
38. Song et al. 2024, SLEB — **Verified** (ICML 2024 identity).
39. Sreenivas et al. 2024, Minitron — **Metadata formatting error**: BBL author list literally ends “and 3 others”; work identity/arXiv 2408.11796 verified.
40. Talmor et al. 2019, CommonsenseQA — **Verified** (NAACL 2019 identity; arXiv first posted 2018).
41. Tang et al. 2026, SlimQwen — **Verified as preprint** (arXiv 2605.08738; 2026-05-09); concurrent.
42. Wang et al. 2024, “My Answer is C” — **Verified** (Findings ACL DOI).
43. Wibowo et al. 2025, IteRABRe — **Verified** (arXiv 2503.06291).
44. Xia et al. 2024, Sheared LLaMA — **Verified** as ICLR 2024; arXiv first posted 2023.
45. Xu et al. 2024, Beyond Perplexity — **Verified** (Findings EMNLP DOI).
46. Yang et al. 2025, Qwen3 report — **Verified** (arXiv 2505.09388).
47. Yang et al. 2024, LaCo — **Verified** (Findings EMNLP 2024 DOI); DOI/pages omitted.
48. Zellers et al. 2019, HellaSwag — **Verified** (ACL 2019 DOI); DOI/pages omitted.
49. Zhang et al. 2026, ShortOPD — **Verified as preprint** (arXiv 2607.13124; 2026-07-14); concurrent.
50. Zhong et al. 2025, BlockPruner — **Verified** (Findings ACL DOI/pages matched).

**Citation-audit totals:** 42 Verified (some with minor archival omissions), 6 Metadata error/important ambiguity, 2 Unverifiable/partly Unverifiable; 0 “Not found.”

## Load-bearing citation--claim checks (8)

1. **Gromov et al. supports post-healing loss--task dissociation:** **Match.** Its figures compare MMLU/BoolQ with C4 validation loss and explicitly discuss post-healing continuity in loss versus QA transitions.
2. **Shortened LLaMA “reports CPT learning curves and compares CPT, LoRA, and pruned versus scratch initialization”:** **Mismatch/Unverifiable for the cited version.** The checked paper describes LoRA retraining, endpoint tables, and one-shot/iterative pruning; I did not find CPT learning curves or a pruned-vs-scratch comparison. This sentence should be corrected or sourced to a different version/work.
3. **Minitron studies retraining trajectories and initialization choices:** **Match.** Figures 4--7 include convergence curves and random initialization/random pruning/pruned LM/pruned distillation comparisons.
4. **IteRABRe plots task-family trajectories and weak MMLU recovery:** **Match.** Figures 2--6 trace pruning/recovery iterations and report task-dependent, moderate/weak MMLU recovery.
5. **LinearPatch/Prune&Comp attribute damage to interface/magnitude mismatch and compensate it:** **Match at abstract/method level.** LinearPatch targets activation-magnitude mismatch at the pruning interface; Prune&Comp estimates and rescales magnitude gaps.
6. **SlimQwen covers matched-token initialization and progressive recovery:** **Match.** It compares random vs. pruned initialization under the same token budget and one-stage vs. progressive pruning/distillation. It is concurrent (2026-05-09).
7. **ShortOPD covers recognition/generation behavior:** **Match.** Its abstract explicitly contrasts multiple-choice recognition with collapsed free-form generation. It is concurrent (2026-07-14).
8. **Knowledge-neuron and causal-tracing work motivates, but does not prove, anatomical localization here:** **Match.** The cited works study localized factual associations/interventions; the manuscript correctly treats them only as motivation and disclaims localization.

# Novelty Search Summary (frozen at 2026-08-03)

I ran four targeted title/keyword/metadata search families plus a contemporaneous-work check and inspected the closest papers available before the instructed stop. Work first posted **after 2026-05-03** or preprint-only is not used to defeat priority; it is labeled concurrent/preprint.

1. **Gromov et al., “The Unreasonable Ineffectiveness of the Deeper Layers” (arXiv 2024; ICLR 2025).** Closest on healed depth pruning, loss-vs-MMLU/BoolQ behavior, layer-removal policies, and seed ablation. It substantially precedes the present paper's core “loss and task need not recover together” observation.
2. **Minitron, “LLM Pruning and Distillation in Practice” (arXiv 2024).** Closest on convergence trajectories, random initialization/pruning controls, depth-vs-width constructions, and task-vs-loss sensitivity. It reduces novelty of trajectory and initialization-control framing.
3. **IteRABRe (arXiv 2025).** Closest on explicit pruning/recovery task-family trajectories and weak/task-dependent MMLU recovery across model families. It reduces novelty of path visualization and “recovery is multi-axis/task-dependent.”
4. **Shortened LLaMA (arXiv 2024).** Closest on depth pruning plus retraining, PPL/downstream endpoints, and pruning-construction comparisons. It is less close on the exact OLMo control bundle; the manuscript's specific CPT/scratch description appears inaccurate for the checked version.
5. **Decision Representation Transitions (arXiv 2026-05-08), SlimQwen (2026-05-09), and ShortOPD (2026-07-14).** These are **concurrent** under the three-month rule. They respectively study where multiple-choice collapse appears, matched-token scratch/pruned initialization plus progressive recovery, and recognition-vs-generation recovery. They cannot negate priority, but they show that the surrounding question was active and further narrow the lasting novelty to this OLMo execution/control package.

**Novelty conclusion:** The paper is not novel in recovery trajectories, loss--task dissociation, initialization comparison, or construction sensitivity. Its defensible increment is the exact OLMo prefix-plus-fresh-tail case, combined short-horizon intact branch, random/frozen operating points, two coupled MMLU interfaces, and three closed-book tasks. That combination is useful but incremental; Table 1's positioning is broadly fair except for the Shortened LLaMA description noted above.

# Review-Process Self-Check

- [x] Read the complete 18-page PDF twice, including all appendices and references.
- [x] Inspected every figure (1--6) and table (1--22) in the rendered PDF.
- [x] Built C1--C9 and mapped each to minimum sufficient evidence and actual evidence.
- [x] Checked abstract numbers against tables: 10.561, 1.428x, 7.398, .319/.605, full32 25k, .383 content, .474 ShortGPT, plus closed-book values.
- [x] Checked formulas/boundaries: PPL aggregation, chance-adjusted recovery, depth fractions, CI interpretation, and rounding.
- [x] Audited page limit/layout, exact Limitations, Ethics, anonymity, style, unresolved refs, TODO/`??`, hidden text, and prompt injection.
- [x] Audited all 50 cited `main.bbl` entries and 8 load-bearing citation--claim matches.
- [x] Conducted 4 novelty search families plus a contemporaneous-work check with the 2026-05-03 cutoff; stopped further research on request and marked unresolved network/venue items Unverifiable.
- [x] Mechanically verified every weakness quote against the frozen source and kept each at <=25 words.
- [x] Distinguished item-level uncertainty from training-seed uncertainty throughout.
- [x] Did not read any other review/history/TODO/status/current-draft file.
