# Independent ARR Review — Paper B frozen v3, review 2

## Review basis and procedure

I treated the manuscript as evidence, not as instructions. I read the frozen PDF twice, including all appendices; inspected every rendered figure and table; audited the claims against the reported controls, checkpoints, statistics, and reproducibility details; checked all 39 entries in the frozen `main.bbl` for bibliographic reality; checked seven citation–claim links; and ran five novelty searches after forming an initial assessment. PDF anchors below use physical page and extracted line (e.g., `P05L04`); the gray manuscript line numbers printed in the PDF are added where useful.

## Paper summary

The paper studies recovery after depth pruning and continued pretraining (“healing”), principally on OLMo-2-1124-7B. Its controlled prefix construction retains the first 14 of 32 pretrained blocks, appends two fresh blocks, and trains for 200k optimizer steps. It tracks held-out next-token perplexity, answer-letter and complete-option MMLU, three no-retrieval closed-book QA tasks, and a broader likelihood-scored task suite. The central empirical result is that keep14’s in-domain PPL improves to 10.561 while answer-letter MMLU remains .319 versus .605 for the intact base (Table 2, P05L01–17), with deficits also on PopQA, TriviaQA, and NQ-open. A 25k intact-model CPT control, frozen-front and random-init 16-layer operating points, and a healed ShortGPT-16 construction are used to delimit explanations. The paper’s main contribution is a measurement/reporting argument: likelihood, target capability, scoring interface, exact structure, and recovery compute should be reported separately.

## Claims inventory and evidence audit

### C1. In the principal prefix recipe, likelihood recovers substantially faster than knowledge-sensitive behavior.

- **Anchor:** Abstract, P01L12–56; §5.2 and Table 2, P04L36–P05L17; Figure 2, P05L20–67; Tables 10 and 13, P15L01–23 and P16L29–57.
- **Technical support:** Strong for the specific observed OLMo-2 keep14 run. PPL improves to 10.561, while MMLU and three closed-book QA metrics remain far below the intact base. The late trajectory shows a statistically detectable but small MMLU gain, rather than falsely claiming numerical flatness.
- **Ideal experiment:** multiple seeded keep14 trajectories; matched long-budget full32; out-of-domain likelihood; checkpointed closed-book/interface evaluations over the full trajectory.
- **Baseline/benchmark/statistics:** benchmark breadth and per-item MMLU tests are good; training-run uncertainty is unquantified, and no long-budget full32 counterfactual is reported.
- **Verdict:** **Supported, narrowly scoped to the observed run and in-domain likelihood.**

### C2. Short-horizon corpus drift is insufficient to explain the large keep14 gap.

- **Anchor:** §5.2, P04L47–53 and P04L67–73; §5.3, P06L01–10; Table 2, P05L01–17.
- **Technical support:** The intact 32-layer model at 25k remains close to base on PPL, MMLU, and closed-book QA. This is useful evidence against an immediate corpus-shift account.
- **Ideal experiment:** continue full32 for the same 200k steps/tokens and report the same checkpoint grid and learning-rate schedule.
- **Verdict:** **Supported only for short-horizon drift; not a long-budget causal control.** The paper acknowledges this.

### C3. MMLU scoring interface contributes to the observed deficit, but does not explain it away.

- **Anchor:** §3.4, P04L15–27; §5.2, P04L57–64; Discussion, P08 manuscript lines 454–470; Table 16, P17L58–81; Table 17, P17L01–25.
- **Technical support:** Complete-option scoring narrows the gap, while a random-init model attains a similar normalized-content score and closed-book QA remains weak. This successfully demonstrates protocol sensitivity plus a fluency floor.
- **Ideal experiment:** a factorial interface study that holds prompt wording/context fixed while varying only answer representation and normalization; answer-order permutations; free-generation answer extraction.
- **Verdict:** **The descriptive interface-sensitivity claim is supported; an “answer-symbol/readout component” is not isolated causally.**

### C4. Nominal 16-layer depth does not uniquely determine the recovered endpoint.

- **Anchor:** §6.3, P06L11–29; Table 3, P07L76–93; Discussion, P08 manuscript lines 471–483.
- **Technical support:** ShortGPT-16 reaches lower PPL and much higher MMLU than keep14 at 200k.
- **Ideal experiment:** matched 14-inherited+2-fresh variants that separately manipulate final-block retention, inherited count, contiguity, and fresh-tail use, with equal tokens/FLOPs and seeds.
- **Verdict:** **Supported as an existence/comparison claim; no single structural cause is identified.**

### C5. Observed recovery rate depends on inherited depth within the prefix recipe.

- **Anchor:** §6.2, P06L69–75 and continuation at manuscript lines 392–398; Figure 5 and Table 3, P07L36–93; Table 6, P13; Table 13, P16L29–57.
- **Technical support:** keep8 shows no detectable late MMLU change while keep14 gains 1.68 points from 128k to 200k.
- **Ideal experiment:** all depths trained to a common token/FLOP budget with identical checkpoint grids and multiple seeds; estimate depth × time interaction directly.
- **Verdict:** **Suggestive, not cleanly identified.** The arms differ in checkpoints, inherited depth, and potentially optimization regime; the current evidence establishes differing observed trajectories, not an isolated depth effect.

### C6. The qualitative dissociation extends beyond the principal OLMo-2-7B arm.

- **Anchor:** §6.4, P06 manuscript lines 419–435; Figure 6 and Table 9, P14L01–60; Tables 7–8, P13.
- **Technical support:** OLMo-2-1B shows falling PPL with near-chance MMLU, and a Qwen3-8B endpoint is directionally consistent.
- **Ideal experiment:** matched multi-family, multi-scale replications using the same compression fraction, corpus, budget, interfaces, and controls.
- **Verdict:** **Directional only, as the paper states.**

### C7. Layer-wise probe thresholds are descriptive correlates, not storage localizations.

- **Anchor:** §4, P04L54–73; Appendix C, P14 manuscript lines 976–990 and P15 manuscript lines 991–1001; Figure 7, P15L25–45; Tables 20–21, P17L69–81 and P18L01–26.
- **Technical support:** The readouts are clearly defined and the paper repeatedly avoids causal localization claims.
- **Ideal experiment:** causal ablation/patching or matched interventions at probe-transition depths.
- **Verdict:** **Supported as descriptive analysis; appropriately caveated.**

## Summary of strengths

1. **[Major] The paper asks a useful, well-scoped measurement question and supports its central specific observation with multiple behavioral interfaces.** Table 2 (P05L01–17) puts PPL, two MMLU interfaces, and three closed-book QA tasks side by side; Tables 10, 16, and 17 expand the evidence. This makes the central keep14 dissociation substantially more credible than an MMLU-only result.

2. **[Major] The control design is unusually interpretation-aware for a compression study.** The intact same-corpus control, frozen-front and random-init operating points, interface comparison, and ShortGPT construction comparison each address a distinct alternative explanation (§3.3–3.4, P03L62–P04L34; §5.3, P05L73–P06L18). The manuscript is also disciplined about what these controls do *not* identify.

3. **[Major] The paper reports trajectories rather than only endpoints and uses appropriate item-level statistics where predictions are available.** Figure 2 (P05L20–67), Table 13 (P16L29–57), and Table 15 (P17L32–49) show late recovery and paired uncertainty. The wording explicitly distinguishes “small but detectable” recovery from no change.

4. **[Minor] Evaluation and integrity details are strong.** The appendix provides sample counts, chance floors, scoring conventions, shard-weighted PPL aggregation, architecture reconstruction checks, NaN/truncation checks, and per-subject MMLU results (P14 manuscript lines 943–975; Tables 19–22, P17–18).

5. **[Minor] The paper is commendably candid.** The Limitations section explicitly acknowledges single runs, unequal budgets, control confounds, interface non-equivalence, probe dependence, same-source PPL, and no contamination audit (P09 manuscript lines 531–575). This improves interpretability and trust.

6. **[Minor] Figures and tables are generally legible and accurately captioned.** I found no material arithmetic inconsistency in the headline recovery ratios, PPL taxes, or group-level recovery values. Figure 3 is also carefully framed as motivating two-axis reporting rather than proving global decorrelation.

## Summary of weaknesses

### W1. **[Major] The central training claims rest on single, non-reproducible runs, so training variance is unknown.**

- **Location:** Limitations, P09L01–09 (manuscript lines 531–539); Appendix B.1, P14L63–74 (manuscript lines 921–930).
- **Short quote:** “same-shape controls are single training runs”
- **Weakens:** C1, C5, and partly C4; ARR soundness and reproducibility. Item-level intervals quantify benchmark sampling uncertainty, not variability from fresh initialization, data order, or optimization.
- **Remedy:** run at least 3 seeds for keep14 and the key 16-layer controls; report mean/SD or hierarchical intervals for PPL and headline capabilities. At minimum, add one independent keep14 replication and seeded launch/data-order details.

### W2. **[Major] The intact corpus control does not match the 200k healing exposure, so it cannot exclude long-horizon corpus/schedule effects.**

- **Location:** §5.3, P06L01–10 (manuscript lines 343–352); Limitations, P09L05–09.
- **Short quote:** “not step- or token-exposure-matched”
- **Weakens:** C2 and the interpretation of C1. The 25k control supports only the manuscript’s “short-horizon” wording; it is not the relevant 200k counterfactual.
- **Remedy:** train full32 to 200k or at least evaluate 50k/100k/200k checkpoints under the same data array and schedule. If computationally infeasible, narrow all causal language to the observed 25k horizon and quantify plausible drift by extrapolation/sensitivity analysis.

### W3. **[Major] The “inherited depth affects recovery rate” conclusion is not identified by a compute-matched depth experiment.**

- **Location:** §6.2, P06L69–75 and manuscript lines 392–398; Table 3 caption, P07L84–93.
- **Short quote:** “PPL was still decreasing”
- **Weakens:** C5. keep8 and keep14 are observed over different checkpoint windows and budgets; the shallow ladder stops at heterogeneous “plateau” checkpoints, while PPL is still improving. Comparing a null 45k–121k MMLU interval to a positive 128k–200k interval is not a direct depth × healing-rate test.
- **Remedy:** train keep8/10/12/14 with a common token/FLOP budget and checkpoint grid, then test the interaction between depth and step on aligned per-item outcomes. Otherwise rephrase C5 as “the observed trajectories differ across the two arms.”

### W4. **[Major] The ShortGPT comparison is useful but heavily confounded and is not a sufficient structural baseline set.**

- **Location:** §6.3, P06L11–29; Limitations, P09L13–19.
- **Short quote:** “differ along four coupled dimensions”
- **Weakens:** C4’s practical implications and any suggestion that upper-block preservation is responsible. ShortGPT changes inherited count (16 vs 14), contiguity, final-block retention, and fresh-tail construction simultaneously. Its closed-book QA cells are unreported in Table 2, unlike keep14.
- **Remedy:** add matched variants: contiguous 16 inherited/no fresh tail; 14 selected inherited+2 fresh; prefix14 plus original final block under fixed total depth; and ShortGPT with closed-book QA/content-MMLU. Equalize tokens and, preferably, FLOPs.

### W5. **[Major] The interface analysis changes several factors at once and omits close prior work on MMLU/answer-selection sensitivity.**

- **Location:** §3.4, P04L15–27; Discussion, P08 manuscript lines 454–470; Limitations, P09L20–29.
- **Short quote:** “different prompts and candidate scoring”
- **Weakens:** C3 and the novelty/positioning of the scoring-interface contribution. Letter vs complete-option scoring changes prompt, continuation tokens, tokenization, and length normalization, so the “answer-symbol/readout component” remains only a consistency statement. Relevant prior work not cited includes Wang et al. (2024), *“My Answer is C”*; Alzahrani et al. (2024), *When Benchmarks are Targets*; and Gupta et al. (2024), *Changing Answer Order Can Decrease MMLU Accuracy*.
- **Remedy:** add these references and a factorial control that holds context/prompt fixed while varying answer labels versus answer strings, raw versus normalized scoring, and answer order. Free-generation answer extraction would further test whether letter scoring reflects usable behavior.

### W6. **[Minor] The manuscript overstates the scope of the paired-control appendix and contains a concrete internal inconsistency.**

- **Location:** Appendix A.3, P12L74–81 (manuscript lines 858–864) versus Table 15, P17L32–49.
- **Short quote:** “All five differences favor keep14”
- **Weakens:** reporting accuracy/statistical clarity. Table 15 contains three comparisons, including frozen-front, while the prose says five differences and then says frozen-front is not included because predictions were not retained. The displayed table explicitly reports keep14–frozen and frozen–random.
- **Remedy:** reconcile the prose with the actual retained predictions/table, state exactly which rerun produced each row, and explain whether the “five” refers to tasks or comparisons. This should be corrected before publication.

### W7. **[Minor] The likelihood axis is only in-domain, and contamination/out-of-domain validity is untested.**

- **Location:** Limitations, P09L43–50 (manuscript lines 568–573).
- **Short quote:** “We do not report out-of-domain PPL”
- **Weakens:** C1’s broader interpretation. The PPL shard is disjoint but from the same Dolmino/DCLM source as healing; it measures in-domain fit and may improve without broader language-model recovery.
- **Remedy:** add at least one out-of-domain PPL/cross-entropy corpus and a benchmark-overlap/contamination audit or explicit overlap screening. Keep the present in-domain interpretation if these cannot be added.

### W8. **[Minor] Exact artifact reproduction and compute accounting are incomplete.**

- **Location:** Appendix B.1, P13 manuscript lines 891–920 and P14L63–74; Ethics, P09 manuscript lines 589–599.
- **Short quote:** “cannot provide an exact project-wide GPU-hour total”
- **Weakens:** reproducibility and practical cost interpretation. Hardware and optimizer settings are given, but no released code/config/checkpoint manifest is stated, no exact seed is given, the data-loader resume is not exact, and total training compute/energy is unavailable.
- **Remedy:** release anonymized code/configs/evaluation scripts plus checkpoint hashes and exact command lines; provide per-arm wall-clock/GPU-hours/tokens/FLOPs and the actual data-order/resume behavior.

## Novelty and closest-work audit

I used the following search questions after reading the paper:

1. **Has prior depth-pruning work already shown post-healing loss/PPL and downstream QA dissociation?**  
   Closest: Gromov et al., *The Unreasonable Ineffectiveness of the Deeper Layers* (first public March 26, 2024). It explicitly reports that healing restores next-token loss near unpruned levels while QA curves can show different behavior. This substantially overlaps the broad headline that likelihood is not a sufficient capability certificate.

2. **Has prior work compared continued pretraining/retraining choices after depth pruning?**  
   Closest: Kim et al., *Shortened LLaMA* (February 5, 2024), which finds continued pretraining superior to LoRA at severe pruning; also Sheared LLaMA/Minitron-style work studies pruning followed by substantial retraining/distillation. The current paper’s novelty is therefore not “healing pruned models,” but the trajectory/control/interface measurement package.

3. **Has prior compression work argued that perplexity misses knowledge or other behavior?**  
   Closest: Namburi et al. (2023), Jaiswal et al. (October 2, 2023), and Xu et al. (2024). These precede the paper by far more than three months and already establish the general beyond-perplexity motivation.

4. **Has prior work studied MMLU answer-label/prompt/scoring sensitivity?**  
   Closest: Wang et al. (2024), Alzahrani et al. (2024), and Gupta et al. (June 27, 2024). The paper does not cite these, which makes the interface contribution less novel than the Related Work section suggests, although the application to prune-then-heal remains useful.

5. **Are there recent trajectory/initialization studies close enough to affect novelty under the three-month rule?**  
   Closest found: Kim et al., *Rethinking Layer Redundancy* (April 27, 2026) and Tang et al., *SlimQwen* (May 9, 2026). Relative to the frozen PDF date of August 3, 2026, both are within roughly three months. I therefore treat them as concurrent/recent context, not as novelty-defeating omissions. SlimQwen is nevertheless relevant because it compares pruned initialization with scratch and optimization trajectories under continued training, albeit for MoE pretraining rather than this OLMo measurement setting.

**Novelty conclusion:** The broad thesis—perplexity can fail to track post-compression capabilities—is not new. The paper’s credible novelty is narrower: a trajectory-centered OLMo case study combining an intact same-corpus control, same-shape operating points, an MMLU interface comparison, closed-book QA, and a construction comparison. That package is useful, but the empirical scope and confounds make it an incremental measurement contribution rather than a new general principle.

## Citation audit

### Bibliographic reality

- Frozen `main.bbl`: **39/39 entries matched to real works** via DOI/ACL Anthology, arXiv metadata, OpenAlex/Crossref, or (for the 2020 Logit Lens post) the stated LessWrong page.
- **Per-entry result (all verified):** tunedlens; PIQA; DoLa; BoolQ; ARC; Knowledge Neurons; DRPruning; LayerSkip; Geva et al.; Gromov et al.; MMLU; Jaiswal et al.; TriviaQA; Shortened LLaMA; Calibration Matters; Natural Questions; Reassessing Layer Pruning; PopQA; ShortGPT; ROME; OpenBookQA; Compact Language Models; Cost of Compression; Logit Lens; OLMo-2; LAMBADA; WinoGrande; SocialIQA; A Deeper Look; Dolma; SLEB; Minitron; CommonsenseQA; Sheared LLaMA; Beyond Perplexity; Qwen3; LaCo; HellaSwag; BlockPruner.
- I found **no hallucinated reference** and every cited key appears in `main.bbl`.
- Minor metadata caveat: some `main.bbl` years reflect conference publication rather than first public posting (e.g., Gromov/ShortGPT/OLMo-2); this is not a hallucination.

### Citation–claim spot checks (7)

1. **Gromov et al.** for post-healing loss/downstream divergence: **match**, and it is the closest overlapping prior result.
2. **Shortened LLaMA** for CPT versus LoRA after depth pruning: **match**.
3. **Namburi et al.** for compression effects on parametric knowledge: **match**.
4. **Jaiswal et al.** for knowledge-intensive degradation despite low/nearly preserved PPL: **match**.
5. **Xu et al.** for divergent safety effects under compression: **match**.
6. **ShortGPT/SLEB/LaCo/BlockPruner** for their respective structural selection/removal methods: **match**.
7. **Logit Lens/Tuned Lens, DoLa/LayerSkip, and knowledge-neuron/causal-tracing citations** for layer-wise readout or factual-computation context: **broadly match**, and the manuscript appropriately avoids turning these into causal storage claims.

The main citation issue is therefore **omission**, not fabrication: close work on MMLU scoring/prompt/answer-label sensitivity should be discussed.

## Desk-review checklist

- **In ARR/NLP scope:** Yes.
- **Originality/salience:** The specific control-and-trajectory package is original enough to review, but the broad beyond-perplexity thesis is prior art; see novelty audit.
- **Readable and self-contained:** Yes; essential method and headline evidence are in the main paper, with details in the appendix.
- **Long-paper length:** Main content ends on physical page 8; Limitations begins there and continues on page 9. No obvious page-limit violation.
- **ACL review style/anonymity:** Review mode and anonymous author are present; no acknowledgments or identity-revealing repository links in the rendered paper.
- **Limitations section:** Present and substantive.
- **Ethical considerations:** Present; no human-subject data or annotators.
- **Hallucinated citations:** None found (39/39 verified).
- **Formatting/readability:** All figures/tables rendered; no fatal overflow or unreadable text found. Page 7 is sparse, but this is not a desk issue.
- **Responsible-NLP submission-form checklist:** Not available in the frozen source/PDF, so cannot be assessed.
- **Dual submission/resubmission metadata:** Not available and not assessed.
- **Desk-reject recommendation:** **No.**
- **Ethics review:** **No.** The main risks are ordinary deployment overconfidence and compute/energy, already acknowledged; no special ethics-panel issue is apparent.

## All figure/table inspection

- **Figures 1–7:** inspected. Captions match plotted values and stated scope. Figure 3 appropriately labels heterogeneous checkpoints; Figure 5’s graphical trajectory ends at 44k while the separate 45k–121k inference is only in text, which is clear but easy to miss. Figure 7 is descriptive and not overclaimed.
- **Tables 1–22:** inspected. Headline cells are arithmetically consistent with reported taxes/recovery. The material issue is the Appendix A.3 prose/Table 15 inconsistency in W6. Table 2 also leaves ShortGPT closed-book cells unreported, limiting that policy comparison rather than invalidating existing values.

## Limitations and societal impact

The paper adequately discusses most scientific limitations and the practical risk of relying on aggregate LM metrics. It could improve by reporting per-arm compute/energy, stating the intended artifact-release plan, and discussing that MMLU/closed-book benchmark behavior is not equivalent to user-facing capability or safety. No special ethics review is needed.

## Questions for the authors

1. What exactly does “pre-registered 25k likelihood plateau” mean—where and when was it registered, what stopping rule was fixed, and was any later full32 checkpoint run but omitted?
2. Can the authors reconcile Appendix A.3’s “five differences”/“frozen-front is not included” statement with Table 15, which contains three comparisons including frozen-front?
3. Are there independent seeds or duplicate keep14/ShortGPT runs not reported? If not, how sensitive are the main trajectories to fresh-layer initialization and data order?
4. Can a matched 16-layer ablation isolate inherited count, final-block retention, contiguity, and fresh-tail effects?
5. Will code, exact configs, checkpoint hashes, per-item predictions, and the data-order/resume logic be released?

## Scores

**Best paper justification:** N/A (overall score below 4.5).

- **Soundness: 3 / 5 (Acceptable).** The main narrow claim is supported, and the paper is unusually careful, but several secondary causal/trajectory interpretations rely on single runs and unmatched controls.
- **Excitement: 3 / 5 (Interesting).** The measurement protocol and controls are useful; the broad beyond-perplexity conclusion is already established, so the novelty is incremental.
- **Overall Assessment: 2.5 / 5 (Borderline Findings).** I lean slightly below Findings because the strongest new claims would benefit from seeded replication, a 200k full32 control, and matched structural/interface ablations. A revision could plausibly reach Findings.
- **Reviewer Confidence: 4 / 5.** I read the full paper and appendix twice, checked all figures/tables and references, and searched the closest literature. Some uncertainty remains because raw code/checkpoints and submission metadata were unavailable.
- **Reproducibility: 3 / 5.** Many implementation/evaluation details are documented, but exact reproduction is impeded by unspecified seeds, non-exact data-loader resumption, no stated artifact release, and incomplete compute accounting.
- **Datasets:** **N/A.** The paper does not claim a new dataset contribution or release.
- **Software:** **N/A / no release stated.** The frozen paper does not promise a software artifact; this also contributes to the reproducibility concern above.

## Final recommendation

**Borderline Findings / resubmission-worthy.** I believe the paper contains a useful and mostly sound measurement result, but in a selective setting I would prioritize it after the authors (i) establish training robustness, (ii) add or sharply delimit the long-budget full-model counterfactual, (iii) isolate at least one structural/interface factor, and (iv) correct the appendix inconsistency and related-work omissions.
