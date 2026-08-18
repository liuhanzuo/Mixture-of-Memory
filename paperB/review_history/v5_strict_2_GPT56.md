```yaml
review_mode: strict
soundness: 3.0
excitement: 2.5
overall: 3.0
confidence: 4.5
reproducibility: 2.0
```

# Paper Summary

This paper studies whether decreasing **in-domain** perplexity can certify recovery of selected knowledge-sensitive evaluations after depth pruning. Its principal observed path retains OLMo-2-7B blocks 0--13, appends two fresh blocks, and continues pretraining for 200k optimizer steps. The paper compares the intact base, a full32 continued-pretraining branch observed only to 25k, frozen-front and fully random same-shape operating points, and a coupled ShortGPT-16 construction. Evaluation includes answer-letter and complete-option MMLU plus zero-shot closed-book PopQA, TriviaQA, and NQ-open.

The narrowly supported result is descriptive and useful: on the one keep14 run, PPL improves from 10.826 to 10.561 between 128k and 200k while answer-letter MMLU rises only from .3012 to .3191 and remains far below the intact .6053; the base--keep14 gap also occurs on all three closed-book tasks. A random-init content-MMLU floor and the stronger ShortGPT endpoint demonstrate interface and construction dependence. The manuscript is unusually candid about what is not identified. However, the claim is intentionally narrow, the principal path has no independent training replication, the intact control is not available at the claim horizon, and exact/artifact-level reproduction is currently blocked. I therefore place it at **Findings level (3.0)** rather than ACL-main level.

# Claim--Evidence Audit and Minimum Sufficient Experiments

## C1. In-domain PPL is not a sufficient certificate for the measured knowledge-sensitive evaluations on the literal observed keep14 path.

- **Minimum sufficient experiment:** At least one predeclared pruned+CPT path with repeated target and in-domain-PPL measurements; show PPL improvement while the target remains materially below a clearly defined recovery criterion, using the same fixed evaluation protocol. Independent training reruns are required only for a stochastic/general recovery-dynamics claim, not for the paper's explicitly literal-path claim.
- **Actual evidence:** §5.1, lines 312--331; Figure 1; Table 2; Appendix Tables 12, 15, and 16. PPL 10.826→10.561 and MMLU-L .3012→.3191 from 128k→200k; intact base .6053. The endpoint gaps recur on PopQA, TriviaQA, and NQ-open.
- **Judgment:** **Supported for the one realized path and the named evaluations.** Not evidence of eventual non-recovery, causal damage, or a population-level dynamic.

## C2. Short-horizon corpus shift is not a complete explanation for the keep14 deficit.

- **Minimum sufficient experiment:** An intact model continued on the identical data/order/schedule through the same horizon, ideally across seeds, with the same evaluations.
- **Actual evidence:** §5.2, lines 334--341; Table 2. full32 is close to base at 25k, but has no 200k checkpoint.
- **Judgment:** **Supported only at short horizon.** It excludes a rapid universal corpus-shift account, not an intact-model effect at 200k.

## C3. The MMLU conclusion is interface-sensitive, and complete-option scoring has a substantial non-target floor.

- **Minimum sufficient experiment:** Paired item-level interface scores plus a relevant null model; ideally vary one interface factor at a time.
- **Actual evidence:** §3.3, §5.2, Appendix Table 15. keep14 content-normalized .3832 vs letter .3184; random .3598 vs .2470. The paper explicitly states that prompt, candidate, tokenization, and normalization all change.
- **Judgment:** **Supported as protocol sensitivity and a null floor, not as isolation of answer-symbol mapping.**

## C4. The keep14 endpoint depends on construction rather than nominal 16-layer depth alone.

- **Minimum sufficient experiment:** Two same-depth constructions at matched training budget, with exact structural differences disclosed; factor attribution would require matched one-factor ablations.
- **Actual evidence:** §5.3 and Table 2: ShortGPT-16 has PPL 9.780 and MMLU-L .474 at 200k versus keep14 10.561/.319. Four structural factors differ.
- **Judgment:** **Supported as construction dependence; not as a ShortGPT-selection, block-31, inherited-count, or fresh-tail causal effect.**

## C5. The proposed reporting axes are warranted by this case.

- **Minimum sufficient experiment:** Demonstrate at least one consequential disagreement across likelihood, target metric, interface, construction, and uncertainty/budget reporting.
- **Actual evidence:** Figure 1, Table 2, §7, and Limitations expose precisely these disagreements and confounds.
- **Judgment:** **Reasonable case-study recommendation, not a universally validated standard.**

# Strengths

**S1. The central claim is sharply bounded and matched to the evidence.** The abstract, §4, §6.4, §7, and conclusion repeatedly restrict the contribution to “literal observed paths” and reject causal, localization, deletion, universal-dynamics, and post-budget interpretations (e.g., lines 288--308 and 435--446). This is materially better calibrated than the stronger claims often made from a single pruning endpoint.

**S2. The paper makes important confounds visible instead of relabeling them as clean ablations.** Table 2 states inherited/fresh block counts, trainable sets, learning rates, budgets, and missing cells; §3.1--3.2 explicitly says that Random, Frozen, and ShortGPT are operating points. The ShortGPT comparison is appropriately limited to construction dependence.

**S3. Evaluation interfaces are handled thoughtfully.** The same 14,042 MMLU items are scored under answer-letter and content interfaces; Appendix Table 15 reports raw and token-normalized content scores, and random initialization supplies a useful null. Closed-book PopQA/TriviaQA/NQ-open reduces dependence on the answer-letter interface.

**S4. Item-level uncertainty is correctly separated from training-run uncertainty.** §5.1, Appendix Tables 11--15, and Appendix B.3 state that bootstrap CIs and McNemar tests condition on fixed checkpoints. The paper reports the common-item rerun discrepancy rather than silently merging rounded aggregates.

**S5. Reproducibility disclosure is unusually honest.** Appendix B provides batch size, schedule, optimizer, precision, clipping, sample counts, prompts, normalization, checksums, environment, commit hashes, and the missing seed/loader-offset/compute fields. The disclosure does not make the work reproducible, but it lets a reviewer distinguish known from unknown provenance.

**S6. The rendered paper is readable and self-contained at the claim level.** I inspected Figures 1--2 and Tables 1--19. Captions generally state scope and confounds; the main seven content pages contain the central protocol and result rather than relying on an unseen supplement.

# Weaknesses (ordered by severity)

## W1. No independent training replication for the load-bearing path — **Major**

- **Location:** Limitations, PDF lines 508--515; also Figure 1 caption and Table 2.
- **Exact quote:** “Training seeds were not explicitly set in the historical runs.”
- **Problem:** keep14, ShortGPT, Frozen, Random, and the principal trajectory are single realized training runs. The 95% CIs quantify item variation conditional on fixed checkpoints, not uncertainty from initialization, data order, or optimization.
- **Affected claim/norm:** C1 remains a valid existence claim for this path, but the paper's practical recommendation and scientific interest depend on the phenomenon not being a rare realization. C4's endpoint ordering is likewise unreplicated. This blocks an ACL-main level conclusion about recovery behavior and limits the evidence to a careful case study.
- **Sufficient remedy:** Repeat at least the principal keep14 path and one relevant comparator with ≥3 independently seeded runs using fixed data-order/stopping rules; report mean/dispersion for PPL and target metrics and whether every run exhibits the claimed separation. If compute prohibits this, keep the literal-path framing and position the work explicitly as a Findings case study.

## W2. The intact same-corpus control does not reach the 200k claim horizon — **Major**

- **Location:** Limitations, PDF lines 516--524; §5.2, lines 334--341.
- **Exact quote:** “full32 ends at 25k and cannot control the keep14 200k endpoint.”
- **Problem:** The central path is interpreted at 200k/52.4B nominal presentations, but the intact branch stops at 25k/6.6B. Thus corpus continuation, schedule effects, or long-horizon base drift are not measured at the endpoint where the paper makes its main comparison.
- **Affected claim/norm:** C2 is supported only as a short-horizon bound. It cannot serve as a matched endpoint counterfactual for C1, and any wording that readers interpret as fully controlling corpus shift would be too strong.
- **Sufficient remedy:** Continue full32 to 200k under the identical data order and schedule (preferably replicated), or remove it from the set of endpoint controls and describe it solely as a 25k diagnostic. The current manuscript mostly adopts the latter wording, so this is an evidential limitation rather than a contradiction.

## W3. Scope remains one principal family/recipe with weak external validity — **Major**

- **Location:** Limitations, PDF lines 535--544; §6.3 and Appendix Tables 6--8.
- **Exact quote:** “The study is English, mainly 7B, one model family, one mixture, and one prefix+fresh-tail recipe.”
- **Problem:** The 1B and Qwen rows change scale, retained fraction, corpus, architecture, or available evaluations and are correctly labeled directional context. They do not replicate the full proxy-validity package or the OLMo-7B claim.
- **Affected claim/norm:** C1 is sound for the stated OLMo path, but excitement/general scientific reach is limited. The recommendation may be sensible, yet the evidence does not establish how common the failure is across model families, pruning rules, data mixtures, or instruction-tuned models.
- **Sufficient remedy:** Run the same predeclared trajectory, intact control, MMLU interfaces, and closed-book suite on at least one second architecture/family at a comparable retained fraction and budget, or further foreground that the contribution is a single-setting measurement audit.

## W4. Exact reproduction and independent artifact verification are currently blocked — **Major**

- **Location:** Limitations, PDF lines 537--550; Appendix B.1--B.3, lines 857--930.
- **Exact quote:** “Exact reproduction of keep14 is further blocked by an unrecorded within-epoch loader offset after a 34.5k resume.”
- **Problem:** Historical training seeds, the resumed loader position, per-run GPU hours, and exact realized compute are unavailable. Moreover, the task-specific evaluator revisions are identified only by local commit hashes rather than released code. The source archive supplied with the frozen paper contains manuscript source, not the training/evaluation artifacts claimed by the appendix.
- **Affected claim/norm:** This does not falsify C1--C4, but it prevents exact reproduction and limits independent verification of the reported evaluations. It materially lowers reproducibility.
- **Sufficient remedy:** Release a runnable anonymized artifact with the exact checkpoint files/checksums, arrays/manifests, evaluator commits, environment lockfile, commands, and aligned prediction files. Exact historical rerun remains impossible without the loader offset; therefore also provide a clean seeded replication protocol and clearly distinguish reproduction of reported numbers from replication of the phenomenon.

## W5. Missing closed-book evaluation for the strongest alternative construction — **Minor**

- **Location:** Table 2 caption, PDF page 6; Limitations, lines 529--533.
- **Exact quote:** “no ShortGPT closed-book evaluation is available.”
- **Problem:** ShortGPT is the key evidence that nominal 16-layer depth is insufficient to determine the endpoint, but its superiority is demonstrated only for PPL/MMLU and the broader likelihood suite, not the three closed-book tasks used to show that keep14's deficit is not an answer-letter artifact.
- **Affected claim/norm:** C4 is supported for PPL and MMLU, but construction dependence is not established for closed-book recall. Readers cannot tell whether ShortGPT's advantage transfers to the paper's other load-bearing interface.
- **Sufficient remedy:** Evaluate ShortGPT on PopQA, TriviaQA, and NQ-open with the identical prompt/normalization and report paired or bootstrap uncertainty where prediction alignment is available.

## W6. Figure 1 uses text below the ACL readability guideline — **Minor / desk-risk**

- **Location:** Figure 1, PDF page 2.
- **Exact quote:** “Single run; no seed-level uncertainty.”
- **Problem:** Mechanical PDF inspection finds extensive 6--7 pt text in Figure 1 (axis labels, annotations, operating-point metadata), below the document's 11 pt body and the ACL guidance that figure/table text should be body-sized whenever possible and clearly readable when printed. The figure is legible when zoomed, but dense at A4 print scale.
- **Affected claim/norm:** Formatting/readability norm, not scientific validity. Because ACL warns that abuse of figure/table font size can be desk-rejectable, this should be corrected before submission.
- **Sufficient remedy:** Split the figure or simplify annotations and render all essential labels at a clearly printable size (approximately caption/body scale); move secondary construction metadata into the caption or Table 2.

# Questions That Could Change the Score

1. Can the authors provide independent seeded reruns of keep14 (and ideally ShortGPT or full32) showing that the PPL--target separation and endpoint ordering persist? A consistent replicated result could raise soundness/overall by 0.5.
2. Is there a 200k full32 checkpoint or a defensible reason it cannot be run? A matched long-horizon intact control would substantially strengthen interpretation.
3. Can the exact evaluator commits, aligned predictions, checksum manifest, and checkpoints be made available anonymously during review? If yes, reproducibility could rise materially.
4. What are ShortGPT's PopQA/TriviaQA/NQ-open results under the same protocol? If its advantage recurs, the construction-dependence claim becomes broader and more compelling.
5. Was the submission accompanied by the Responsible NLP checklist and anonymous checksum/artifact manifest referenced in Appendix B? These were not part of the frozen PDF/source set I was permitted to inspect, so their compliance is **Unverifiable** here.

# Non-scoring Suggestions / Typos

- Define an operational “recovery” criterion even if the paper ultimately rejects certification by PPL; currently “base-level” is intuitive but not formalized as a tolerance or non-inferiority margin.
- In Table 2 and nearby text, keep `MMLU-L`/`MMLU-C` naming fully consistent with “letter/content-normalized” to reduce switching costs.
- The bibliography entry for OLMo 2 says 2025, while arXiv records the first submission on 2024-12-31; this is defensible if citing the 2025 version, but adding the exact arXiv/version date would remove ambiguity.
- If archival versions exist, prefer them consistently over generic venue labels (e.g., include DOI/pages for DRPruning and other ACL/NeurIPS entries).
- Page 7 has substantial unused whitespace; Figure 1 could potentially be simplified/rebalanced without increasing the content-page count.

# Detailed Scores

## Soundness: 3.0 / 5.0

The core literal-path claim is supported by correctly reported measurements and restrained interpretation. The strongest deductions—interface sensitivity and construction dependence—are also appropriately bounded. Soundness is held at 3.0 because all trained constructions are single runs, the intact comparator is not horizon-matched, and several controls are intentionally confounded operating points. These weaknesses do not make the paper wrong; they prevent stronger population/causal conclusions.

## Excitement: 2.5 / 5.0

The paper is useful as an evidence-discipline case study and offers unusually transparent bookkeeping. However, “perplexity is not enough,” loss/task dissociation after compression, recovery trajectories, and interface sensitivity are substantially anticipated by prior work. The novel increment is a specific OLMo control/interface package rather than a new method, theory, benchmark, or broadly replicated empirical law.

## Overall: 3.0 / 5.0

**Findings.** This is a careful, credible, narrow paper with real diagnostic value. It is below ACL-main threshold because its main scientific object is one unreplicated path, external validity is limited, the endpoint lacks a matched intact control, and exact reproduction is blocked. I was between 3.0 and 3.5; following the requested lower-bin rule, I choose 3.0 because the missing training-level uncertainty is directly claim-linked.

## Confidence: 4.5 / 5.0

I read the complete 17-page PDF twice, including all appendices; inspected every figure/table; checked the frozen source corresponding to the PDF; mechanically verified all weakness quotes and “lacks X” assertions; audited all 33 rendered bibliography entries; and checked novelty/cutoff dates. Remaining uncertainty concerns unavailable external artifacts and a few metadata records, which are marked Unverifiable rather than treated as absent.

## Reproducibility: 2.0 / 5.0

The paper documents many details, but exact keep14 reproduction is explicitly impossible from recorded state, training seeds/loader offset and compute are missing, task-specific evaluator commits are not released, and the frozen source bundle does not contain the claimed training/evaluation artifacts. A clean future replication is feasible from the described protocol, but reproduction of the reported run is not.

# Limitations, Ethics, and Desk-Reject Risks

- **Page limit:** Main content ends on page 7; Limitations/Ethical Considerations are on page 8; references start page 9; appendices start page 11. This fits the 8-page long-paper limit.
- **Limitations:** Exact section titled `Limitations` is present after Conclusion and before references. It contains limitation disclosure rather than new experiments/figures.
- **Ethics:** `Ethical Considerations` is present and discusses deployment caution, inherited model/data risks, licensing, and energy. No new human-subject data or annotators are used.
- **Anonymity:** Author is “Anonymous ACL submission”; PDF Author/Subject/Keywords metadata are empty; no acknowledgments or identifying repository URLs were found. Local commit hashes and `/opt/conda/...` environment text are not by themselves identifying. No clear anonymity violation found.
- **Official style:** A4, two-column ACL review style, embedded fonts, and line rulers are present. Appendices remain two-column. Figure 1's 6--7 pt annotation text is the main style/readability concern.
- **Unresolved references/TODOs:** No `??`, unresolved citation markers, TODO/TBD/FIXME placeholders, or leftover revision prose found in the rendered PDF/actually included source.
- **Prompt injection/manipulation:** I searched extracted text, included source, colors, font sizes, and page bounds. No reviewer-directed instruction, hidden/white/off-page text, or prompt injection was found. Non-black/tiny text corresponds to line rulers, hyperlinks, and visible plot annotations.
- **Responsible NLP checklist:** **Unverifiable** from the permitted frozen PDF/source; no checklist form was available in that set.
- **Supplementary artifact anonymity/completeness:** **Unverifiable** beyond the manuscript source archive. The paper references an accompanying checksum manifest and prediction artifacts that were not in the permitted set.

# Abstract-number Verification

Five representative abstract numbers/quantitative statements were checked against the main/appendix tables:

1. keep14 PPL **10.561 at 200k** — matches Figure 1, Table 2, and Appendix Table 3.
2. keep14 MMLU-L **.319 vs intact .605** — matches Table 2 and Appendix Tables 9/11 (rounding from .3191/.6053).
3. full32 observed only through **25k** and near base — matches Table 2 and Appendix Tables 3, 15, and 16.
4. random 16-layer content score similar to keep14 while letter score is chance — Table 15: random .3598 content-normalized/.2470 letter; keep14 .3832/.3184.
5. ShortGPT-16 **.474 MMLU at 200k** — matches Table 2 and Appendix Tables 4/9/11/15 (.4739/.4742 under different common reruns).

No abstract-number contradiction found.

# Complete `main.bbl` Citation Audit

Status uses **Verified** when the title/authors/year/venue or arXiv identifier matched an authoritative DOI/Crossref, ACL Anthology metadata, arXiv record, or well-established primary record; **Metadata error** for a substantive mismatch; **Unverifiable** when the network/available record did not support a reliable full check. Network failure is not treated as “Not found.”

1. Alzahrani et al. 2024, *When Benchmarks are Targets* — **Verified** (ACL DOI metadata).
2. Chen et al. 2025, *A Simple Linear Patch Revives Layer-Pruned Large Language Models* — **Verified** (arXiv 2505.24680; first posted 2025-05-30).
3. Chen et al. 2026, *Prune&Comp* — **Verified** title/authors via arXiv 2507.18212; venue-year metadata **Unverifiable** beyond the paper record.
4. Deng et al. 2025, *DRPruning* — **Verified** (arXiv 2411.14055; ACL 2025 DOI 10.18653/v1/2025.acl-long.1414).
5. Gromov et al. 2025, *The Unreasonable Ineffectiveness of the Deeper Layers* — **Verified** (arXiv 2403.17887; ICLR 2025 metadata consistent).
6. Gupta et al. 2024, *Changing Answer Order Can Decrease MMLU Accuracy* — **Verified** (arXiv 2406.19470).
7. He et al. 2025, *PASER* — **Verified** (arXiv 2502.12594).
8. Hendrycks et al. 2021, *Measuring Massive Multitask Language Understanding* — **Verified** (arXiv 2009.03300; ICLR 2021).
9. Jaiswal et al. 2024, *Compressing LLMs: The Truth Is Rarely Pure and Never Simple* — **Verified** (arXiv 2310.01382; ICLR 2024).
10. Joshi et al. 2017, *TriviaQA* — **Verified** (ACL DOI 10.18653/v1/P17-1147; arXiv 1705.03551).
11. Kim et al. 2024, *Shortened LLaMA* — **Verified** (arXiv 2402.02834).
12. Kim et al. 2026, *Rethinking Layer Redundancy* — **Verified** (arXiv 2604.24938; first posted 2026-04-27, before cutoff).
13. Kwiatkowski et al. 2019, *Natural Questions* — **Verified** (TACL DOI 10.1162/tacl_a_00276).
14. Lu et al. 2024, *Reassessing Layer Pruning in LLMs* — **Verified** (arXiv 2411.15558).
15. Mallen et al. 2023, *When Not to Trust Language Models* — **Verified** (ACL DOI 10.18653/v1/2023.acl-long.546; arXiv 2212.10511).
16. Martra 2025, *Fragile Knowledge, Robust Instruction-Following* — **Verified** (arXiv 2512.22671; first posted 2025-12-27).
17. Men et al. 2025, *ShortGPT* — **Verified** (ACL Findings DOI 10.18653/v1/2025.findings-acl.1035).
18. Muralidharan et al. 2024, *Compact Language Models via Pruning and Knowledge Distillation* — **Verified** (arXiv 2407.14679; NeurIPS 2024 record).
19. Namburi et al. 2023, *The Cost of Compression* — **Verified** (ACL DOI 10.18653/v1/2023.findings-emnlp.349).
20. OLMo Team et al. 2025, *2 OLMo 2 Furious* — **Verified** (arXiv 2501.00656; first posted 2024-12-31, later 2025 versions). Minor year-version ambiguity only.
21. Shi et al. 2026, *Understanding Performance Collapse...* — **Verified** (arXiv 2605.07271; first posted 2026-05-08). This is post-cutoff contemporaneous work.
22. Siddiqui et al. 2024, *A Deeper Look at Depth Pruning of LLMs* — **Verified** (arXiv 2407.16286).
23. Song et al. 2024, *SLEB* — **Verified** (arXiv 2402.09025; ICML 2024).
24. Sreenivas et al. 2024, *The Minitron Approach* — **Verified** (arXiv 2408.11796; NeurIPS 2024).
25. Tang et al. 2026, *SlimQwen* — **Verified** (arXiv 2605.08738; first posted 2026-05-09). Post-cutoff contemporaneous work.
26. Wang et al. 2024, *My Answer Is C* — **Verified** (ACL Findings DOI 10.18653/v1/2024.findings-acl.441).
27. Wibowo et al. 2025, *IteRABRe* — **Verified** (arXiv 2503.06291).
28. Xia et al. 2024, *Sheared LLaMA* — **Verified** (arXiv 2310.06694; ICLR 2024).
29. Xu et al. 2024, *Beyond Perplexity* — **Verified** (ACL Findings DOI 10.18653/v1/2024.findings-emnlp.901).
30. Yang et al. 2025, *Qwen3 Technical Report* — **Verified** (arXiv 2505.09388; first posted 2025-05-14).
31. Yang et al. 2024, *LaCo* — **Verified** (ACL Findings DOI 10.18653/v1/2024.findings-emnlp.372).
32. Zhang et al. 2026, *ShortOPD* — **Verified** (arXiv 2607.13124; first posted 2026-07-14). Post-cutoff contemporaneous work.
33. Zhong et al. 2025, *BlockPruner* — **Verified** (ACL Findings DOI 10.18653/v1/2025.findings-acl.262).

**Totals:** 33/33 entries located at title/record level; 32 **Verified**, 1 **Verified with venue-year Unverifiable** (Prune&Comp); 0 Not found. Minor metadata improvements: add archival DOI/pages where available; clarify OLMo arXiv version/year.

# Load-bearing Citation--Claim Matches (7 checks)

1. **Gromov et al.** cited for deeper-layer removal, healing, and loss/task dissociation — **Broadly matched.** The primary abstract confirms layer pruning, healing via finetuning, and QA performance; detailed “loss--task dissociation” strength is consistent with the paper but the full PDF fetch was incomplete, so that narrow subclaim is **Unverifiable** here rather than rejected.
2. **Shortened LLaMA** cited for CPT vs LoRA and recovery curves — **Matched.** Its primary abstract explicitly says CPT markedly outperforms LoRA after severe depth pruning.
3. **Minitron** cited for structured pruning/distillation and task behavior — **Matched at method scope.** Primary abstract confirms depth/width pruning, distillation, and LM-evaluation benchmarks; the manuscript's more specific “trajectory/initialization choices” characterization is **Unverifiable** from the accessible primary abstract.
4. **IteRABRe** cited for iterative block removal/recovery and capability trajectories — **Matched in broad scope** (iterative recovery-aided block reduction and capability preservation); exact plot-level characterization is **Unverifiable** from the accessible abstract.
5. **Namburi/Jaiswal/Xu** cited for compression effects missed by aggregate LM metrics across knowledge/downstream/safety — **Matched.** Titles/abstract records directly concern parametric knowledge, multi-benchmark compressed-LLM evaluation, and multidimensional safety beyond perplexity.
6. **Wang/Alzahrani/Gupta** cited for multiple-choice interface sensitivity — **Matched.** Primary records concern first-token-vs-text mismatch, leaderboard/evaluation sensitivity, and MMLU answer-order sensitivity.
7. **Kim et al. 2026 calibration paper** cited for dependence of pruning choices on calibration — **Matched.** The primary abstract explicitly finds calibration configuration dominates search choice in pruning patterns/perplexity and materially affects downstream accuracy.

# Novelty Search Summary (cutoff: 2026-05-04)

I ran four targeted primary-record searches: (i) depth pruning + perplexity/task recovery; (ii) post-pruning recovery trajectories; (iii) pruning initialization/scratch/CPT; and (iv) interface or recognition/generation mismatch after compression. Closest pre-cutoff papers:

1. **Gromov et al., 2025, *The Unreasonable Ineffectiveness of the Deeper Layers*** — contiguous/deeper-layer removal followed by healing; directly adjacent on post-pruning task recovery and the strongest antecedent to the paper's motivating gap.
2. **Kim et al., 2024, *Shortened LLaMA*** — depth pruning with CPT-vs-LoRA retraining and recovery measurements; close on the recovery-path object.
3. **Sreenivas et al., 2024, *The Minitron Approach*** — depth/structured pruning plus distillation/CPT and benchmark recovery, with practical retraining analysis.
4. **Wibowo et al., 2025, *IteRABRe*** — iterative block pruning/recovery and task-family behavior.
5. **Kim et al., 2026, *Rethinking Layer Redundancy: Calibration Matters More Than Search*** — posted 2026-04-27, before cutoff; directly shows perplexity/downstream misalignment under controlled depth pruning and reduces novelty of the broad proxy critique.

Additional adjacent pre-cutoff work includes Jaiswal et al. (2024), Namburi et al. (2023), and Xu et al. (2024) on multidimensional evaluation of compressed models, plus interface-sensitivity work on MMLU.

**Novelty judgment:** The paper correctly does **not** claim discovery of trajectories or “beyond perplexity.” Its defensible increment is the **combination** of one densely measured OLMo prefix+fresh-tail path, a 25k intact branch, same-shape null operating points, paired MMLU interfaces, closed-book QA, and an explicitly confounded ShortGPT comparison. That is a useful empirical package, but incremental.

**Three-month rule:** Decision-transition (2026-05-08), SlimQwen (2026-05-09), and ShortOPD (2026-07-14) are after the 2026-05-04 cutoff and are appropriately treated as contemporaneous/post-cutoff rather than novelty-destroying prior art. Note that ShortOPD is more than three months after the cutoff but still post-cutoff; it should not be used against priority.

# Review-Process Self-Check

- Read the 17-page frozen PDF twice, including pages 11--17 appendices.
- Built claims C1--C5 and compared each with a minimum sufficient experiment before judging actual evidence.
- Inspected Figures 1--2 and Tables 1--19 in the rendered PDF; verified captions, missing cells, and major numbers.
- Checked formulas/boundaries: PPL aggregation is token-weighted; 200k×128×2048 = 52.4B nominal presentations; chance-adjusted recovery values are consistent; confidence intervals are correctly labeled item-level rather than seed-level.
- Audited page limit, Limitations placement/content, ethics, anonymity, official style, unresolved refs/TODOs, injection/hidden text, and abstract numbers.
- Audited every rendered `main.bbl` entry and seven load-bearing claim matches; network-incomplete checks were marked **Unverifiable**, never Not found.
- Ran four novelty searches and enforced the 2026-05-04 cutoff/contemporaneous-work rule.
- Mechanically found each weakness quote in the frozen PDF/source and verified every “paper lacks X” assertion against the main text and appendices.
- Scoring calibration applied: 4.0 = ACL main; 3.0 = Findings; uncertainty between bins resolved downward only for claim-linked reasons.
