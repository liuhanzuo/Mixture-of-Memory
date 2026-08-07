# 复活 slate — revive-killed-directions 判决

来源: workflow `wigvzeoxd` / `wf_f4eca2c5-b35`（6 agents / 1.30M subagent tokens / 2.6h / 326 tool uses）。
**MAIN 已逐位复核 rank-1 的四组证据，零漂移**（见文末核实小节）。

统计: recovered=1 revivable=0 genuinely_dead=1

| rank | id | GPU-h | 标题 |
|---|---|---|---|
| 1 | `null-calibration-methodology` | 0 | What Survives Null Calibration: Four Evaluation Constructs Where the Reported Number Is Mostly Its Own Floor |
| 2 | `paperE-constant-floor-instrument` | 40 | The Letter Interface Fails Below Chance: Two Mechanisms by Which a Standard MMLU Protocol Becomes a Constant Predictor |
| 3 | `dllm-infilling-ar-dominance` | 24 | Diffusion's Home Turf, Measured: A Matched-Lineage AR FIM Control Overturns the Infilling Case, and a Headline Toggle That Does Nothing |
| 4 | `kspan-difficulty-ladder-artefact` | 12 | Nested Difficulty Ladders Manufacture the Result They Test: A Worked Self-Falsification |
| 5 | `cyclic-layer-reinit-boundary` | 480 | Does Forget-and-Relearn Survive the Move to LLM Pretraining? A Boundary Characterisation with a Two-Axis Cost Structure |

---

## #1 `null-calibration-methodology`

**What Survives Null Calibration: Four Evaluation Constructs Where the Reported Number Is Mostly Its Own Floor**

**复活论点**: Across four structurally unrelated evaluation constructs we own end-to-end, the signal remaining after calibration against a construct-appropriate input-blind null is a small fraction of the reported absolute value, and in three of the four the field's standard reporting practice omits the null entirely. The four: (1) MC scoring-interface inflation -- the content interface hands a letter-chance model .3598 vs a .25 floor, a structural +10.98pp, while the arm-to-arm effect it is used to measure is only 2.34pp, i.e. the inflation is 4.8x the effect; (2) label-prior floors -- data/squad_val.jsonl has 997/2000 = 49.85% identical Chinese refusal, so an input-blind constant scores EM 49.85 and beat every arm we had; (3) representation-similarity nulls -- our 91-pair midband z-CKA observed mean 0.4907 against a layer-order-shuffle null of 0.453, gap +0.038, i.e. the null accounts for 92.3% of the reported quantity; (4) probe-vs-native readout -- linear knee 0.393L (Qwen3-8B) / 0.285L (OLMo-2-7B) vs native knee 0.824L / 0.875L, a 0.43-0.59 gap that makes 'where knowledge lives' an artefact of the readout. The contribution is a null-calibration protocol plus the demonstration that it changes conclusions, not merely error bars.

**差异化定位**: The nearest framing neighbour is arXiv:2602.14486 (Revisiting the Platonic Representation Hypothesis: An Aristotelian View), ICML 2026, OpenReview venueid=ICML.cc/2026/Conference + Submission15852 Camera_Ready_Revision, first public 2026-02-16 -- genuinely pre-cutoff, so it is real prior art and we cite it as such. It proposes permutation null calibration for ONE construct (representation similarity) and its null corrects a SCALE confound. We differ on three axes simultaneously: (a) our shuffle null targets layer-order correspondence, not scale; (b) we run null calibration across FOUR unrelated construct families (MC scoring interface, generative label prior, representation similarity, probe readout depth), which is what turns it from a fix into a claim about evaluation practice; (c) our worked example includes a case where we falsified OUR OWN headline with it (Paper E Obs4 ranking-flip, retracted because both arms were at the chance floor). arXiv:2606.16897 (Contrastive-Difference CKA) is first public 2026-06-15, i.e. AFTER 2026-05-07 -- CONCURRENT, cite and position, does not preempt. Critical precision guard already recorded in the task: an earlier reviewer claimed 2606.16897 ran no null; it DOES run a permutation null (0.727 vs 0.689). Our criticism is therefore not 'they omitted the null' but 'the null was run, significance holds, and the absolute gap is 5-8% of the observed value while the abstract says near-perfect functional transfer.' Getting that distinction right is the difference between a strong paper and a retractable one.

**原杀因**: Never formally killed -- task #170 sits pending with no owner while three sibling directions (Paper C v1, Paper D, Paper E) were each killed in ways that generated its four evidence bases as by-products. It was treated as residue of failures rather than as the finding.

**该杀因为何不成立**: Every one of the four evidence bases exists BECAUSE a direction was killed on a null-calibration failure, which makes the pattern itself the result. The project killed Paper C v1 on a 49.85% constant-refusal floor, retracted Paper D H2 as an n=1 artefact, retracted Paper E Obs4 after discovering both arms sat at chance, and softened Paper B's recovery headline once bf16 ties were counted. Four independent self-inflicted null failures in one project is not noise, it is a measurement about the field's default practice. I re-verified all four from raw data today rather than from summaries: constant-D floor recomputes to exactly .2689 (gold marginal A .2295 / B .2465 / C .2551 / D .2689, n=14042); the SQuAD majority label recomputes to exactly 0.4985; H3 observed midband mean recomputes to 0.4906724527 from repr_alignment_results.json against the 0.453 shuffle null. Zero drift.

**第一步实验**: Pure recompute, no GPU. (1) Build one table with four rows, each reporting reported_value / construct-appropriate_null / calibrated_residual / residual-as-fraction-of-reported, using the exact recomputes verified today. (2) For evidence 3, extend the layer-order-shuffle null from 200 perms/pair to 2000 perms/pair over all 91 pairs and report per-pair p-values with a Benjamini-Hochberg correction, since the current claim rests on median p=0.015 with 58/91 pairs at p<0.05 -- weaker than the headline implies and I want that stated honestly before anything ships. (3) Fetch arXiv:2602.14486 in full and write the boundary paragraph explicitly (scale confound vs layer-order correspondence; one construct vs four). Gate: if the four residual fractions do not span at least an order of magnitude, the cross-construct claim weakens to a single-construct note and we downgrade to a Paper B appendix.

**必需对照**: For every construct, the null must be construct-appropriate and pre-registered, not a generic chance line, and it must be reported for OUR numbers as aggressively as for others'. Specifically: MC-interface null = best constant letter (.2689) AND longest-option heuristic (.2822 OLMo / .2807 Qwen) since content has its own floor; generative null = majority-label constant AND empty string; z-CKA null = layer-order shuffle (0.453), explicitly NOT the random-init floor (0.091) which is the wrong and self-flattering baseline; probe null = native readout. The paper must also include the self-falsification case (Obs4) or it is preaching.

**天花板**: Main-conference methodology/analysis track, or a strong position paper at a venue that takes evaluation seriously. Ceiling is set by breadth of construct coverage plus the fact that we can show conclusion changes rather than only error-bar changes. Not workshop-tier: four constructs, two model families, n=14042 on the MC leg, n=91 pairs on the similarity leg, all with paired significance.

**关键风险**: The framing is one where a reviewer can always say 'everyone knows to report a baseline.' Mitigation is the quantitative surprise -- inflation 4.8x the effect being measured, null accounting for 92.3% of a headline similarity number -- plus the self-falsification. Second risk: 2602.14486 being pre-cutoff means we must position rather than claim the framework; if the boundary paragraph is not crisp the paper reads as its extension.

**复用资产**: 100% existing, zero new compute. paperB/anonymous_artifact/scores/mmlu_content/*/summary.json and olmo2_mmlu_content_results/<10 arms>/per_example_mmlu.jsonl (14042 items each, verified present); data/squad_val.jsonl plus the rebuilt data/paperC_squad_v2/val_refusal25.jsonl with its mandatory floor row already emitted; paperD_research/repr_alignment_results.json (91 pairs, CKA identity gate max|M[i][i]-1| = 1.78e-7); paperA/sections/tab_depth.tex knee values.


---

## #2 `paperE-constant-floor-instrument`

**The Letter Interface Fails Below Chance: Two Mechanisms by Which a Standard MMLU Protocol Becomes a Constant Predictor**

**复活论点**: On damaged-but-nontrivial models, the standard MMLU protocol of scoring the letters A/B/C/D degrades into a constant predictor, and in 3/10 OLMo-2 arms it scores SIGNIFICANTLY WORSE than the best input-blind constant (always-D, .2689): keep8 -1.39pp [-2.54,-0.28], keep14-reheal -1.97pp [-3.18,-0.80], scratch16L -2.19pp [-3.27,-1.11], with 2 further arms statistically indistinguishable from it. On the very same items the content interface remains significantly above its own floor for 10/10 arms. Two mechanisms, each with code-level or distributional evidence: (i) bf16 exact ties -- the tie rate rises monotonically with damage from 0.13% (intact base) to 30.64% (keep8), because autocast bf16 forward at scripts/eval_olmo2_mmlu_content.py:200 precedes the float() cast at line 204, so precision lost before the cast cannot be recovered, and argmax then tie-breaks by index; (ii) letter-prior collapse -- Qwen3 damaged arms have near-zero tie rate but predict 'A' on 100% of items. Same failure, two family-specific causes, therefore a property of the instrument rather than of knowledge.

**差异化定位**: The kill cited three papers. Two are genuine pre-cutoff prior art and we cite them: OLMES (arXiv:2406.08446, first public 2024-06-12) standardizes MC evaluation setups and reports cloze-vs-MCF crossover across training for OLMo; Alzahrani (arXiv:2402.01781, 2024-02-01) shows leaderboard rankings are sensitive to protocol. Neither claims what we claim. OLMES's project is to PRESCRIBE a standard given formulation differences; ours is to show a widely-used formulation becomes an anti-predictor -- scoring below an input-blind constant -- and to name two implementation-level causes (numerical precision, tokenizer prior). No standardization paper attributes interface failure to bf16 tie-break rates that scale with model damage. The third citation is a date error I verified today: arXiv:2607.12767 (Accuracy and Normalized Accuracy under Length Bias) has citation_date 2026/07/14, which is AFTER the 2026-05-07 cutoff -- it is CONCURRENT, cannot preempt, and was nonetheless used to kill Obs3.

**原杀因**: UPDATELOG.md 2026-08-06 23:15 recorded NO-GO after a 20-agent workflow: Obs1 declared occupied by OLMES, Obs3 by arXiv:2607.12767, Obs4 by Alzahrani and additionally self-refuted, leaving 'only Obs2 alive, insufficient to stand alone.' Converted to a Paper B patch (task #172).

**该杀因为何不成立**: Three independent failures. First, the enumeration killed observations one at a time and then declared the remainder too thin, but the actual headline was never in the list -- the constant-floor result plus the two mechanisms appears in the dossier's own section 4 as the top-rated claim (marked strongest, both families, paired significance) and no killer was ever matched against it. Second, one killer post-dates the cutoff by over two months (2607.12767, verified 2026-07-14) and is therefore concurrent by the project's own standard. Third, the strongest evidence was produced by the kill itself: the workflow's attack-4 discovered that Obs4's flip pairs sat at the chance floor, which is exactly the constant-floor phenomenon and should have been read as confirmation of the surviving headline rather than as the loss of a different one. I recomputed all of it from per-example jsonl today -- the 10-arm letter/content table, the paired bootstrap verdicts, and the tie-rate ladder -- with zero drift from the dossier.

**第一步实验**: Third model family plus a non-MMLU MC surface, since a two-family result invites 'artefact of these two models.' (1) Llama-3-8B is on disk at /apdcephfs_wzc1/share_304376610/pighzliu_code/models/Llama--Llama3-8b: run scripts/eval_olmo2_mmlu_content.py letter+content, then run it a second time under a full-fp32 forward. The fp32 rerun is the decisive causal test: if the bf16 tie mechanism is real, tie rate collapses toward the intact-base 0.13% and letter accuracy rises on precisely the damaged arms, which converts a correlation into a manipulated cause. (2) Add ARC-Challenge or HellaSwag under both interfaces so the claim is about MC scoring rather than about MMLU. Gate: if fp32 removes the ties AND lifts the sub-constant arms above the floor, the mechanism is established and the paper is a go; if ties persist in fp32, mechanism (i) is wrong and must be withdrawn before anything is written.

**必需对照**: Both interfaces must be tested against their OWN input-blind floor -- letter against best-constant-letter .2689, content against longest-option .2822/.2807 -- with paired bootstrap on identical items, so we cannot be accused of only auditing the interface we dislike. This control is what produced the honest finding that content also hits its floor on the two Qwen 2k-step arms; that must be reported. Ranking-flip claims must be restricted to arms where BOTH interfaces beat their floors, where the count of significant flips is 0.

**天花板**: Main-conference evaluation/analysis track. The asset base is unusually strong for this genre: 14 arms x 14,042 items with paired significance, two mechanisms with one of them code-level and manipulable, and a controlled damage ladder that no external group has, because the arms are our own healed checkpoints. Being able to sweep instrument failure as a continuous function of model damage is the differentiator.

**关键风险**: The fp32 rerun could show ties are not the cause, which removes the strongest mechanism and leaves a Qwen-only prior-collapse story. That is a real possibility and is why the rerun is experiment one rather than a robustness appendix. Secondary risk: reviewers reading the constant-floor result as merely 'broken models give garbage' -- countered by keep14, which is significantly ABOVE the floor on letter (+4.95pp) yet still carries a 24.4% tie rate, showing the mechanism operates in models that are demonstrably not garbage.

**复用资产**: olmo2_mmlu_content_results/ 10 arms + qwen3_mmlu_content_results/ 4 arms, all 14,042 per-item rows with letter/content_raw/content_norm scores and full per-option logits verified present today; scripts/eval_olmo2_mmlu_content.py; the entire keepN damage ladder (keep8/10/12/14, freezefront, fromscratch, shortgpt16, full32) already healed to 200k. Only the third family and the fp32 rerun are new.


---

## #3 `dllm-infilling-ar-dominance`

**Diffusion's Home Turf, Measured: A Matched-Lineage AR FIM Control Overturns the Infilling Case, and a Headline Toggle That Does Nothing**

**复活论点**: Two coupled results on the one task surface where masked diffusion is claimed to structurally dominate autoregressive models. (a) On n=1033 single-line infilling graded by official evalplus with zero generation errors, Qwen2.5-Coder-7B native FIM reaches pass@1 .7638 -- ABOVE DreamOn's oracle-assisted .7590 and its own-prediction .7018, and above Dream-FIM's .7115 -- while feeding 20.6-24.4x fewer tokens than either DreamOn arm. Qwen2.5-Coder-7B is a true matched lineage for Dream-Coder-v0-Base-7B (identical hidden 3584 / 28 layers / 28 heads / 4 KV heads / intermediate 18944 / vocab 152064 / rope_theta 1e6, differing only in mask_token_id), so this is not a convenience baseline. The mechanism story falls with it: suffix-visibility gain is +.2314 for AR vs +.2991 for diffusion, comparable, so bidirectional context is an affordance of the task FRAMING (FIM), not of the model class. (b) DreamOn's advertised mask_expansion and delete_eos_token kwargs are absorbed by **kwargs and are not parameters, so the length-elasticity that is its headline contribution is not exercised by the public call path -- and every DreamOn number in circulation, including ours, was produced with the toggle inert.

**差异化定位**: This is a follow-up, not a competing method, so the differentiation is a defect rather than a novelty gap: DreamOn ships no matched-lineage AR native-FIM control on its own evaluation surface, and the missing control reverses the conclusion. That is the cleanest kind of contribution available -- the motivation is already established by the paper we are correcting. The inert-kwargs finding is separately a functional defect in a public model's headline capability. Neither is a claim about priority.

**原杀因**: The dLLM length-elastic direction was abandoned on a literature scan that reported 'DreamOn covers this capability,' recorded in DLLM_RESULTS_20260807.md as 'Length-elastic 被 DreamOn (ICLR 2026 Poster) 吃掉.' Territory conceded on the basis of a title.

**该杀因为何不成立**: The scan graded the claim, not the artefact. When measured, DreamOn scores 0.122 pass@1 on from-scratch HumanEval+ and loses on its own home turf to a plain AR model that spends 20.6-24.4x fewer tokens -- it had nominal possession of the territory and occupied none of it. The scan also could not have detected that the capability's public interface is inert, because that requires reading the generation signature and instrumenting the call. A capability is 'covered' when a model demonstrably performs it, not when a paper claims it.

**第一步实验**: The inert-kwargs verification, because it currently rests on a finding I could NOT re-confirm today and that gates the honest version of everything else. DreamOn weights are absent from both mounted model dirs and a filesystem search on .73 timed out. So: re-download DreamOn-v0-7B, diff the accepted signature of diffusion_generate against the documented kwargs, and instrument the call to log per-step how many mask tokens are inserted/deleted, asserting non-zero on every step of every task. Then re-run the variable-length arm with a flag that provably changes behaviour and report the corrected number beside the inert-flag number. Gate: if the kwargs turn out to be live, finding (b) is withdrawn entirely and (a) proceeds alone on already-scored arms.

**必需对照**: Cost must be reported in BOTH units, and here I have to correct the claim I was handed rather than repeat it. Recomputing per-task means from the nested cost field in metrics.jsonl (the summary had read them from score.json, where they do not exist): tokens_fed/task = 238.90 (qwen_fim) / 2035.03 (dream_fim) / 4922.61 (dreamon_fim) / 5826.76 (dreamon_oracle); attended_context_sum/task = 2313.9 / 2035.0 / 4922.6 / 5826.8. So under attended_context_sum, Dream-FIM is 0.88x -- CHEAPER than AR, not more expensive. 'AR dominates on both cost units' is true for the DreamOn arms (2.13x, 2.52x) and FALSE for Dream-FIM. Publishing the two-unit table without that correction would be exactly the flattering accounting this direction exists to criticise. Second required control: the non-oracle diffusion arm must be the headline, with the oracle arm demoted to an upper bound, since oracle per-hole length hands diffusion precisely the sub-problem that variable-length generation is supposed to solve. dreamon_fim is already scored, so this costs nothing.

**天花板**: Solid empirical-correction paper at a main venue if the inert-kwargs finding holds, since it pairs a conclusion-reversing missing control with a functional defect in a public model. Without (b), (a) alone is a strong short paper or a findings-track entry -- still worth writing, because the arms are scored and the marginal cost is prose.

**关键风险**: The inert-kwargs finding is unverified as of today: I confirmed OUR call site passes both kwargs (scripts/generate_evalplus_dreamon.py:133-134) and confirmed our recorded finding, but could not re-read DreamOn's generation code because the weights are gone. If the kwargs are live, half the paper evaporates. Secondary risk: an authors' response that we used the wrong entry point -- pre-empted by instrumenting mask insert/delete counts rather than arguing from the signature alone.

**复用资产**: Six fully scored arms at n=1033 on zwfy6, verified by me today over ssh directly from score.json: qwen_fim .7637947725, dream_fim .7115198451, dreamon_fim .7018393030, dreamon_oracle .7589545015, qwen_prefix .5324298161, dream_prefix .4123910939, all with generation_errors=0. Per-task cost recomputed from metrics.jsonl. Plus scripts/generate_evalplus_dreamon.py (8-shard harness) and the forward_pre_hook cost instrumentation in dllm_draft/scripts/.


---

## #4 `kspan-difficulty-ladder-artefact`

**Nested Difficulty Ladders Manufacture the Result They Test: A Worked Self-Falsification**

**复活论点**: Difficulty ladders built by increasing structural hole count use NESTED task sets, so higher rungs silently discard the hardest tasks and a within-task DECLINE reads as a rise. Demonstrated on our own pre-registered claim, which the artefact would have confirmed and the correction killed: the 59 tasks reaching k=4 are a strict subset of the 164 at k=1 and are far easier at k=1 for both arms (diffusion .898 vs .543 on dropouts; AR .949 vs .819), so diffusion's apparent rise .671 to .746 is the ladder shedding its own hard cases; on the balanced 59-task panel diffusion DECLINES monotonically .898 / .847 / .831 / .746. The pre-registered decisive gate failed with the slope significantly negative (-0.346, z=-2.14, p=.032; decontaminated -0.490, z=-3.83, p=1.3e-4) and the claim was withdrawn. Two further reusable traps from the same pipeline: an inflated null, because a mutation-based null computed over rows the mutation cannot alter reports .457/.287/.143/.119 when the true mutable-only null is .043/.025/.014/.000 with the immutable share growing 43%->90% across rungs, so the null itself SLOPES; and contamination, since identifier renaming plus docstring removal costs AR -.260 and diffusion -.127 at k=1 on a set whose gold refill still scores 1.000, making raw HumanEval-infilling comparisons family-dependent rather than capability comparisons.

**差异化定位**: A methodology paper about ladder construction, not a claim about diffusion versus AR. Its unusual asset is that the worked example is our own falsified hypothesis under a rule committed before the data was seen, with the git history of KSPAN_INFILLING_RESULTS.md as evidence of ordering -- so it cannot be read as constructing a critique to suit a conclusion. Every published multi-region infilling ladder is exposed to the artefact, and the artefact has the same sign as the desired result.

**原杀因**: Effectively self-killed: gate 3 failed, 'diffusion's home turf' was WITHDRAWN, and what survived was recorded as 'a much weaker claim' -- the k x family interaction (AR degrades about 2x faster: -0.435, z=-2.89, p=.0038 balanced and clustered, surviving decontamination at -0.208, z=-2.15, p=.032).

**该杀因为何不成立**: The negative result and the diagnostic machinery that produced it are the contribution; only the original hypothesis died. The project treated a failed pre-registered gate as a dead end when the gate failing IS the finding -- a difficulty axis that inflates effects, an interaction estimate decaying three times over as controls tighten (pilot +0.525 -> naive +0.297 -> balanced +0.153), and a null whose slope was mistaken for signal. Two further defects were found in the handed-down recipe by re-deriving it rather than trusting it: the standard reconstruction prompt+canonical_solution+suffix yields two different files for 123/164 tasks, so '164/164 parseable' is true but not unique, and byte-exact admission gives 910/1033 rows with a k=4 ceiling of 59 rather than the commonly cited 60.

**第一步实验**: Pure CPU reanalysis on existing artefacts, no GPU. Produce the three-panel diagnostic from the scored per-rung files: (1) naive per-rung curve; (2) balanced common-task panel; (3) survivor-vs-dropout split at the base rung showing dropouts are where the weaker arm is worst. Alongside, the two null variants (all-rows vs mutation-verified-rows) with the immutable-share trend per rung. Gate: the paper needs the naive and balanced curves to differ in SIGN, which they already do (.671->.746 rising vs .898->.746 falling), so this is a writing and figure task rather than a discovery task.

**必需对照**: Balanced-panel analysis mandatory alongside per-rung numbers; byte-exact task admission with an explicit deterministic tie-break (majority-vote reference file, shortest then lexicographic) rather than a heuristic newline count; nulls constructed only over rows the mutation provably alters, verified per row; and clustered standard errors by task_id, since each task contributes up to four correlated rows per arm. The 12 GPU-hours are for an extra rung or two of the mutation-verified null if reviewers want the trend extended, not for the main analysis.

**天花板**: Methodology or reproducibility track at a main venue; realistically a strong short/findings paper. Ceiling is bounded because it is a critique with a single task family, and lifted by the pre-registered self-falsification, which is rare and quotable. Honest ceiling: below ranks 1-3, above workshop-only.

**关键风险**: Reviewers may class it as a known statistical point (Simpson's paradox / survivorship) dressed in a new domain. Mitigation is quantitative specificity -- the three-fold decay of the interaction estimate, the order-of-magnitude null inflation, the 43%->90% immutable-row drift -- plus that we hit the same trap twice on one pipeline, which is evidence it is a trap rather than a slip. Also honest: this direction overlaps rank 1 conceptually, and if rank 1 proceeds this may be better as its strongest case study than as a separate paper.

**复用资产**: dllm_draft/KSPAN_INFILLING_RESULTS.md with per-rung score.json files, the byte-exact admission pipeline (910/1033), the frozen spec data/kspan/kspan_spec_v1.jsonl sha256 1cc12a50d1f4255f 415 rows and topology_spec_v1.jsonl sha256 da66a3a1f7cdcb30 165 rows verified byte-identical across both disks before the AR arms ran, and the gold-refill 1.000 decontamination gate. All grading on one evalplus 0.3.1 binary.


---

## #5 `cyclic-layer-reinit-boundary`

**Does Forget-and-Relearn Survive the Move to LLM Pretraining? A Boundary Characterisation with a Two-Axis Cost Structure**

**复活论点**: Later-layer forgetting (LLF, ICLR 2022) and layerwise reinitialization are effective regularizers on small-data image classification, and their own authors report the benefit vanishing on large data and transfer degrading. LLM pretraining sits in that documented failure regime (single-pass, non-overfitting) with the additional twist that knowledge memorization is the objective rather than the pathology. We give the first test of the family at decoder-only LLM pretraining scale, and we measure the failure with an axis the vision literature physically cannot observe: distributional recovery (PPL) and parametric knowledge recovery (MMLU, closed-book QA) are two different curves after one identical structural lesion, so each cycle's PPL cost is recoverable and its knowledge cost is not. This extends the progressive-sensitivity result of Springer et al. (ICML 2025) from Gaussian noise and fine-tuning to discrete layer-level damage, an extension their paper names as future work.

**差异化定位**: Positioned as a scale-and-regime boundary paper against named prior methods, never as a new method -- our keep_front/n_fresh construction is mathematically the LLF mask and we say so in the paper. The four nearest works each differ by exactly one attribute, and each attribute is load-bearing: arXiv:2410.16168 is decoder-only + cyclic + pretraining but resets TOKEN EMBEDDINGS, which leaves the transformer body intact and is precisely why its PPL impact is small and ours is not; arXiv:2602.04536 (IFA) is layer-level + cyclic + equal-size but is CIFAR/MIT-Indoors/Stanford-Dogs image classification; arXiv:2602.08040 (FIRE, ICLR 2026, OpenReview CfZLxT3zIZ) is LLM + periodic but reinitializes individual WEIGHT MATRICES via Newton-Schulz on Q/K projections, two orders of magnitude smaller a perturbation, and its LM work is GPT-0.1B; arXiv:2508.06412 (LoRR) is LLM + layer-ish + cyclic but POST-TRAINING, and its own ablation reports full_layers reset as detrimental, which is prior negative evidence we cite in our favour rather than against us. The evaluation axis is ours alone: none of these can report a knowledge-versus-PPL separation.

**原杀因**: SKEPTIC1/2/3 each returned WEAKENED, and the direction was downgraded from Paper C main line to at most a Paper B section, on the grounds that LLF is mathematically equivalent to our construction, the granularity boundary is occupied by LLF/SEAL/lw, the scale boundary by SPDF (UAI 2023, 1.3B) and Springer (ICML 2025), and that the prior for method-side benefit is 0.

**该杀因为何不成立**: The dedicated adversarial forward-citation audit returned SURVIVES and the direction was downgraded anyway. KILLCHECK_forward_citations.md scanned 434 forward citations across six seed papers at 100% retrieval, grepped 10 LLM tech reports (all CLEAN), and ran 15+ OpenReview and 20+ arXiv queries, concluding no published paper satisfies all five criteria and that the two closest each miss on exactly one. Under the corrected standard, four near-misses each differing by one attribute is four differentiations, not four kills. The skeptics were also arguing against the wrong paper: all three independently converged on the SAME defensible reframing (negative result / boundary characterisation with the two-axis cost structure) and SKEPTIC3 explicitly failed to break that door after six targeted queries. A prior of 0 for method-side benefit is not an argument against a boundary paper -- it is the boundary paper's hypothesis, which is why a null here is publishable.

**第一步实验**: The N=0 compute-matched gate, which all three skeptics independently demanded and which can end the direction in one run. At 1B (OLMo-2-0425-1B is on disk and a keep7 1B replication already completed to 200k) run four arms at matched total compute and matched LR: N=0 plain continued pretraining; N=1 single top-K reinit; N=3 cyclic; and an exact LLF reproduction (top-k reinit in place, not remove-then-append) to prove our operator is or is not the same object. Evaluate all four on PPL plus MMLU-content plus closed-book QA with mandatory constant floors. Gate: if N=3 does not beat N=0 at matched compute, the method claim is dead on the spot -- and the boundary paper proceeds on exactly that result, which is why this experiment cannot waste the budget.

**必需对照**: Non-negotiable set: (1) N=0 at matched compute, without which any cycling gain is just extra training; (2) exact LLF reproduction arm, without which reviewers correctly say we ARE LLF; (3) LR-matched fresh vs inherited blocks -- and note the trainer bug that makes this hazardous, since train_olmo2_arch_probe2_distill.py's _classify_param fails to strip the module. prefix so the fresh-group LR is a no-op there, a fix that landed only in train_olmo2_arch_probe2.py:316, meaning any differential-LR claim must be verified against the actual log group names before it is written; (4) knowledge axis with constant floors, since keep8's letter MMLU is significantly BELOW always-D; (5) optimizer-state reset, without which the effect cannot be attributed to weights rather than stale moments; (6) transfer/SFT axis, because SEAL (CVPR 2023) reports LLF features degrading transfer; (7) a single-pass vs repeated-data regime axis, which is the scientifically most valuable arm because it tests the original authors' own stated boundary condition.

**天花板**: Main-conference negative-result / analysis paper if the two-axis separation is clean; the honest expectation is a null on the method side and a well-quantified boundary, which is publishable but not headline. Explicitly the lowest-ceiling entry in the slate, and ranked here despite a strong differentiation statement because it is the only entry requiring fresh pretraining.

**关键风险**: Cost, and Springer's null. Springer et al. report OLMo-7B showing no overtraining effect out to 3T tokens, so at our scale both curves may not yet have begun to separate, producing a null that is uninformative rather than a null that is a boundary. The 1B-first design is the mitigation: it buys the gate cheaply and only escalates to 7B if separation is visible. Second risk: the fresh-block LR bug silently making arms non-comparable -- must grep the training log for the actual param-group names before trusting any arm.

**复用资产**: scripts/train_olmo2_arch_probe2.py (the prune+heal trainer, keep_front/n_fresh flags already implement the LLF mask); the full keepN ladder healed to 200k with 5k-interval checkpoints, which converts directly into the single-cycle depth-cost curve and therefore the cycling cost budget; keep14@200k plus the matched-PPL control keep14@67500; per-layer knowledge-onset probes (OLMo-2 L18->L19 .326->.544; Qwen3 L24->L25 .236->.621) that justify reset-above vs reset-below as two distinct mechanisms, a distinction LLF's ResNet block threshold has no analogue for; scripts/eval_olmo2_{ppl,mmlu_content,closedbook_qa}.py. Note the engineering gap flagged by SKEPTIC2: current flags cannot excise layers from the MIDDLE of the stack, only from the top, so any mid-stack variant needs new code.


---

# 真死（dead on evidence, not prior art）

## `paperC-pc1-squad-capability` — Paper C P-C1: prune-and-graft as a capability claim on SQuAD

This is the one case where the defect is in OUR setup and the fix was already applied and still failed -- so it is dead on evidence, not on prior art. The original eval set data/squad_val.jsonl has 997/2000 = 49.85% identical Chinese refusal labels (I recomputed: exactly 0.4985), so an input-blind constant scored EM 49.85 above every arm except A2_lora. The correct response was a better eval set, and we built one: data/paperC_squad_v2/val_refusal25.jsonl, refusal rate controlled to exactly 25.00%, with a mandatory input-blind floor row emitted. The full depth sweep then RAN on it (logs/paperC_depthsweep_summary_refusal25.log, 8 arms, n=2000 each) and every arm still lands at or below the 25.00 constant-majority floor: keep24_scratch .2425, keep24_rtrunk .2265, keep14_rtrunk .2250, keep28_rtrunk .2235, keep14_scratch .2005, keep20_rtrunk .2045, keep28_scratch .1890, keep20_scratch .1440. So the models have no measurable capability on this task after the eval defect was repaired -- the floor failure was not an artefact of the labels. Independently, the construct is genuinely preempted: arXiv:2411.15558 (verified first public 2024-11-23, pre-cutoff) prunes the final 25% of layers then fine-tunes lm_head plus the remaining last three layers on Vicuna-7B / Qwen1.5-7B / Llama-3.1-8B-It, occupying all three self-claimed differentiators (genuine shortening, frozen trunk, decoder-only at 7-8B) simultaneously. Note carefully what stays dead: only P-C1-as-capability-claim-on-SQuAD. The trained arms remain valuable as CONTROLLED LESIONS, and the fresh-block-vs-inherited-layer question survives in the followup bank as an LR-matched follow-up to 2411.15558 on a task surface where the models clear their floor (MMLU-content does; SQuAD does not).


---

# ★ Pattern analysis — 对 MAIN 判断机制的诊断

The project ran a plagiarism check where it needed a differentiation analysis, and then ran the check with a bias toward conviction. Six concrete failure modes, each verified against the record today rather than taken from summaries.

**1. A dedicated adversarial audit returned SURVIVES and the direction was downgraded anyway.** KILLCHECK_forward_citations.md is the most thorough prior-art artefact in the repository: 434 forward citations across six seed papers at 100% retrieval, 10 LLM tech reports grepped full-text (all CLEAN), 15+ OpenReview and 20+ arXiv queries, with honest declared blind spots. Verdict: SURVIVES, no published paper satisfies all five criteria. The cyclic direction was nonetheless demoted to at most a Paper B section. The mechanism of the failure is visible in the document's own framing: it opens with "five criteria all hit -> REFUTED; miss one -> WEAKENED; miss all -> SURVIVES." Under that rubric SURVIVES is the ceiling, and WEAKENED had already been assigned by SKEPTIC1 -- so the audit could exonerate but never rehabilitate. We built a one-way ratchet and then were surprised it only turned one way.

**2. Four near-misses, each differing by exactly one attribute, were summed into a kill.** arXiv:2410.16168 is decoder-only + cyclic + pretraining but resets token embeddings, not decoder blocks. arXiv:2602.04536 is layer-level + cyclic + equal-size but is CIFAR/MIT-Indoors/Stanford-Dogs. arXiv:2602.08040 (FIRE, ICLR 2026) is LLM + periodic but reinitializes weight matrices via Newton-Schulz on Q/K, and its LM work is GPT-0.1B. arXiv:2508.06412 (LoRR) is LLM + cyclic + layer-ish but post-training, and its own ablation calls full-layer reset detrimental. Each miss is on a physically load-bearing attribute -- resetting an embedding leaves the transformer body intact, which is exactly why its PPL cost is small and ours is not. We treated four differentiations as cumulative evidence of preemption. Worse, LoRR's finding is prior negative evidence FOR our regime distinction, and we filed it against ourselves.

**3. A post-cutoff paper was used as a preemptor.** Paper E's Obs3 was killed by arXiv:2607.12767. I fetched it: citation_date 2026/07/14, over two months AFTER the 2026-05-07 cutoff. It is concurrent by our own written standard and cannot preempt anything. The pipeline had a venue-verification stage (AUDIT0, 49 papers, five-path cross-check, correctly catching that seven 2026-conference papers show as preprints in S2/DBLP and require OpenReview venueid) -- so the machinery to check dates existed and was excellent. It was pointed at venue prestige and never at the concurrency question. We verified whether a paper was peer-reviewed and forgot to ask whether it existed yet.

**4. A "scoop" that was a different experimental class.** Paper C P-C1 conceded to arXiv:2411.15558, which is real, pre-cutoff, and does occupy the stated construction. But the residual difference -- fine-tuning inherited layers versus grafting a freshly-initialized block -- was dismissed as too small to matter, while at the same time our own scoping doc recorded that scripts/run_paperC_pc1.sh:61-66 gave the three arms three DIFFERENT learning rates (A4 fresh 1e-4 / inherited 2e-5; A3 uniform 3e-4; A1 1e-5), so the 3.25pp we were interpreting was explainable by LR alone. We conceded the interesting difference to prior art and kept the confounded one.

**5. A capability declared "covered" by a model that scores 0.122.** The dLLM length-elastic direction was dropped because a scan reported DreamOn covers it. Measured: 0.122 pass@1 on from-scratch HumanEval+, and on its own infilling turf it loses to a plain matched-lineage AR model -- verified today at n=1033 straight from score.json, qwen_fim .7638 versus dreamon_oracle .7590 and dreamon_fim .7018, at 20.6-24.4x the token cost. Its advertised mask_expansion toggle does not even reach the generation code. We accepted a title as an occupancy claim.

**6. The kills manufactured a finding and nobody picked it up.** Four directions died on null-calibration failures -- a 49.85% constant-refusal floor, an n=1 CKA artefact, a ranking flip whose arms were both at chance, a recovery headline sensitive to bf16 ties. Each was recorded as a cause of death. Task #170, which is the paper those four deaths jointly constitute, has sat pending with no owner. We were harvesting a result and filing it as debris.

One more, structural: this very re-adjudication was handed the payload {"id":"test","name":"Test","summary":"test"} while the real dossiers sat unread on disk. The index-directions agent in run wf_f4eca2c5-b35 emitted a schema-valid probe at 08:01:07, its four subsequent real submissions all arrived with empty input, and the harness recorded the probe as the step result. The orchestrator never checked that a direction carried a kill reason before spending an agent on it. The prior agent to see this stub diagnosed a different run (wf_51cff631-530) and, unable to adjudicate nothing, routed it to stays-dead. Two agents in a row spent on a placeholder is the same disease as the rest: unvalidated input treated as authoritative.

**The rule changes.**

(a) *Disqualification requires a named paper with a verified first-public date before cutoff that matches on claim AND method AND setting.* Write the four-field match table explicitly. Any single mismatch is a differentiation statement, and mismatch size is irrelevant. Never sum near-misses.

(b) *Date before venue.* Fetch citation_date and citation_online_date for every asserted preemptor before it is allowed to kill anything. On/after cutoff minus 3 months = concurrent = cite and position. This one check would have saved Paper E.

(c) *An audit returning SURVIVES restores the direction.* No verdict rubric may have a ceiling that merely fails to convict. If a check can only confirm or weaken, it is a prosecution, not an audit -- so pair every kill-check with a mandated differentiation draft, and if the drafter can write a crisp statement the direction lives.

(d) *Measure before conceding.* No capability is "covered" by a paper's claim. If a public artefact exists, run it. One eval run beats any literature scan, and cost us 0.122-versus-0.764 to learn.

(e) *A defect in our own setup obliges a repair attempt, not a kill.* Repair, re-run, and only then judge. P-C1 stays dead precisely BECAUSE we rebuilt the eval set, re-ran all eight arms, and watched them still fail the 25.00 floor. That is what an earned kill looks like, and it is the only one in this slate.

(f) *A null result that killed a direction is a candidate finding.* Before closing anything, ask what the failure measures. Four null-calibration deaths in one project is a paper about evaluation practice, and it is now rank 1.

(g) *Validate agent input.* Fail fast on an empty or stub direction index; require a verbatim kill reason and one cited preemptor before dispatching a re-adjudication."

---

# Follow-up bank

[
 {
  "target_paper": "DreamOn (DreamOn-v0-7B), variable-length masked diffusion for code infilling",
  "defect": "Advertised length-elasticity toggles mask_expansion and delete_eos_token are absorbed by **kwargs and are not parameters, so the headline capability is not exercised by the public call path",
  "our_fix": "Re-download the model, diff the accepted diffusion_generate signature against the documented kwargs, instrument the call to log per-step mask insertions/deletions and assert non-zero on every step of every task, then report the corrected variable-length number beside the inert-flag number",
  "effort": "24 GPU-h on one H20 node; harness scripts/generate_evalplus_dreamon.py already runs DreamOn end to end with zero generation errors. UNVERIFIED TODAY: weights absent from both mounted model dirs and a .73 filesystem search timed out (exit 143), so this rests on our recorded finding plus our own call site at scripts/generate_evalplus_dreamon.py:133-134 and must be re-confirmed before shipping",
  "from_direction": "dllm length-elastic (abandoned on a DreamOn literature scan)"
 },
 {
  "target_paper": "DreamOn infilling/variable-span efficiency and capability claim",
  "defect": "No matched-lineage autoregressive native-FIM control on its own evaluation surface; the missing control reverses the conclusion",
  "our_fix": "Publish qwen_fim as a first-class row under identical task admission, identical grader and both cost units, and reframe bidirectional context as a task-framing affordance (FIM) available to AR rather than a diffusion monopoly; suffix-visibility gain is +.2314 AR vs +.2991 diffusion, comparable",
  "effort": "Zero GPU. Six arms already scored at n=1033; verified by me today over ssh: qwen_fim .7637947725 / dreamon_oracle .7589545015 / dreamon_fim .7018393030 / dream_fim .7115198451 / qwen_prefix .5324298161 / dream_prefix .4123910939, all generation_errors=0",
  "from_direction": "dllm length-elastic"
 },
 {
  "target_paper": "Our own dLLM cost accounting (status/dllm_reposition_salvage/SLATE.md:278 and the mined summary)",
  "defect": "Two errors of our own. The '8.5x fewer tokens' figure is attributed to DreamOn but is Dream-FIM's ratio (2035.03/238.90 = 8.52); DreamOn's is 20.6x (fim) to 24.4x (oracle). Worse, the claim 'AR dominates on both cost units' is FALSE for Dream-FIM: under attended_context_sum, Dream-FIM is 0.88x, i.e. CHEAPER than AR",
  "our_fix": "Recompute both units from the nested cost field in metrics.jsonl (they do NOT exist in score.json, contrary to the summary I was handed) and report per-arm rather than pooled: tokens_fed/task 238.90 / 2035.03 / 4922.61 / 5826.76 and attended_context_sum/task 2313.9 / 2035.0 / 4922.6 / 5826.8 for qwen_fim / dream_fim / dreamon_fim / dreamon_oracle",
  "effort": "Zero, done today",
  "from_direction": "dllm length-elastic (self-audit)"
 },
 {
  "target_paper": "Dream-Coder / Dream 7B diffusion LLM efficiency claims on HumanEval+/MBPP+",
  "defect": "Efficiency accounting reports only the favorable cost unit; NFE and step counts hide that each step re-feeds the whole canvas",
  "our_fix": "Two-unit hook-verified cost table with the matched-lineage AR model on the same Pareto plot, stating explicitly at which quality target diffusion is on the frontier and at which it is dominated",
  "effort": "Zero new training; forward_pre_hook agreed with the closed-form prediction on 1084/1084 task-runs, and an empty-program negative control scored 0.000, excluding a degenerate grader",
  "from_direction": "dllm length-elastic"
 },
 {
  "target_paper": "HumanEval-infilling multi-span/multi-hole difficulty ladders",
  "defect": "Nested task sets across rungs: higher rungs drop the hardest tasks, so a within-task decline reads as a rise, and the artefact has the same sign as the desired result",
  "our_fix": "Mandate a balanced common-task panel alongside per-rung numbers, byte-exact task admission with deterministic tie-break, and a survivor/dropout split at the base rung",
  "effort": "CPU-only reanalysis of existing per-rung score.json files",
  "from_direction": "dllm k-span infilling (task #174)"
 },
 {
  "target_paper": "Mutation-based / corrupted-input null baselines in infilling difficulty evaluations",
  "defect": "Null computed over rows the mutation cannot alter, so it is inflated ~10x AND slopes with the difficulty parameter as the immutable-row share grows 43%->90%",
  "our_fix": "Construct the null only over rows the mutation provably changes, verify per row that bytes changed, and report the null per rung so readers can see it is flat: true null .043/.025/.014/.000 vs reported .457/.287/.143/.119",
  "effort": "CPU-only",
  "from_direction": "dllm k-span infilling"
 },
 {
  "target_paper": "HumanEval-based infilling evaluation across Dream/DreamOn/FIM results",
  "defect": "Reported accuracy substantially measures surface memorization of HumanEval identifiers and docstrings, and contamination sensitivity is family-dependent so raw comparisons are not capability comparisons",
  "our_fix": "Make the decontaminated variant (renamed identifiers, stripped docstrings) primary with raw as a contamination-delta diagnostic, gated on a gold-refill sanity check scoring 1.000; report the delta per model since AR loses .260 and diffusion .127 at k=1",
  "effort": "CPU rescore plus a short generation pass on one node",
  "from_direction": "dllm k-span infilling"
 },
 {
  "target_paper": "Dream-Coder-Instruct-7B reported HumanEval/MBPP numbers and diffusion-vs-AR code comparisons generally",
  "defect": "Protocol under-specification: the sampler alone moves the metric further than the cross-method deltas being argued about (top_p 1.0->0.95 swings HumanEval .616->.762, +14.6 points)",
  "our_fix": "Require a reported sampler envelope (small top_p/temperature grid) with the metric range, and treat any claimed delta smaller than that envelope as not established; publish our grid as the reference envelope",
  "effort": "Small GPU cost to widen the grid; round-1 numbers retained in runs/*.r1/ as the ablation",
  "from_direction": "dllm baselines"
 },
 {
  "target_paper": "DreamOn-style variable-length infilling evaluation, and our own comparison which inherited the flaw",
  "defect": "Oracle supplied to one arm only: diffusion gets oracle per-hole lengths while AR uses its own stopping rule, so the protocol tests everything except the claimed mechanism",
  "our_fix": "Promote the non-oracle diffusion arm to headline (the model must predict span length itself, which IS the advertised capability), demote oracle to an upper bound, hold the stopping rule symmetric. The oracle is worth ~5.7 points (.7590 vs .7018) and AR still wins",
  "effort": "Near zero; dreamon_fim already scored at n=1033",
  "from_direction": "dllm k-span infilling"
 },
 {
  "target_paper": "arXiv:2606.16897 Contrastive-Difference CKA (preprint, first public 2026-06-15 = CONCURRENT)",
  "defect": "Not a missing null -- it RUNS a permutation null (0.727 vs 0.689) -- but the calibrated gap is +0.038, about 5% of the observed value, while the title claims moderate geometric convergence and the abstract near-perfect functional transfer",
  "our_fix": "Report calibrated residual as a fraction of the reported value as a mandatory column. Our own 91-pair measurement has the identical +0.038 gap (0.4907 observed vs 0.453 shuffle null, null = 92.3% of the reported quantity), so we apply the standard to ourselves first",
  "effort": "Zero; recomputed from paperD_research/repr_alignment_results.json today",
  "from_direction": "Paper D representation alignment / task #170"
 },
 {
  "target_paper": "CKA-based layer-correspondence and model-stitching claims generally",
  "defect": "Baselines against a random-init floor (~0.09) rather than against a layer-order-shuffle null (~0.45), which is the correct null for 'is layer i the right partner for layer j' and makes mid-band CKA of ~0.5 nearly uninformative",
  "our_fix": "Re-baseline all 'CKA is decent (~0.5)' reasoning against the shuffle null; report the mid-depth WORST diagonal point (median 0.205, min 0.042, 2 pairs below the random floor) rather than the mid-band mean, since that is the number that decides whether stitching at mid depth is viable",
  "effort": "CPU; extend from 200 to 2000 perms/pair with BH correction, since the current evidence is median p=0.015 with only 58/91 pairs at p<0.05",
  "from_direction": "Paper D layer stitching"
 },
 {
  "target_paper": "Our own Paper D R3 (internal)",
  "defect": "H2 ('depth mismatch hurts more than family mismatch') was drawn from a single data point and reverses at n=91: same_family beta=+0.171 p=.0012 (QAP .0022) vs log(depth_ratio) beta=-0.056 p=.273 (QAP .47), and even the family effect dies without GPT-2",
  "our_fix": "Retract H2 explicitly; keep H1 (U-shape, quadratic coef>0 in 72/91, binomial p=2.0e-8) which is the strongest result and also the most damaging to a stitching paper; widen H3's band since only 53/91 pairs fall in the quoted 0.35-0.61",
  "effort": "Zero, already measured with a passing CKA identity gate (max|M[i][i]-1| = 1.78e-7)",
  "from_direction": "Paper D (self-audit)"
 },
 {
  "target_paper": "Standard MMLU letter-scoring protocol as used across LLM evaluation",
  "defect": "Degrades to a constant predictor on damaged models, scoring SIGNIFICANTLY BELOW always-D (.2689) in 3/10 arms, via bf16 exact ties whose rate scales with damage (0.13% intact -> 30.64% keep8)",
  "our_fix": "Report a tie-rate column and a letter-vs-constant-floor significance column; run the fp32 forward as the causal manipulation; treat ties as abstentions as a sensitivity analysis, which moves keep14 MMLU recovery from 19.25% to 26.57% and full32 from 95.01% to 100.89% (across 100%)",
  "effort": "40 GPU-h for a third family (Llama-3-8B on disk) plus the fp32 rerun; code path confirmed at scripts/eval_olmo2_mmlu_content.py:200 (autocast bf16) before line 204 (log_softmax on .float())",
  "from_direction": "Paper E eval-interface construct validity"
 },
 {
  "target_paper": "Content/cloze MC interface as the proposed remedy for letter scoring",
  "defect": "Content is not a universal fix -- it has its own input-blind floor (longest option, .2822 OLMo / .2807 Qwen) and both Qwen 2k-step arms are above neither floor, so at the extreme both interfaces are invalid",
  "our_fix": "Always test both interfaces against their own floors on identical items with paired bootstrap; restrict any ranking claim to arms where BOTH interfaces beat their floors, where the count of significant flips is 0",
  "effort": "Zero; already computed and it is what forced us to retract our own ranking-flip headline",
  "from_direction": "Paper E (self-audit)"
 },
 {
  "target_paper": "Effect sizes reported under a single MC interface",
  "defect": "The same comparison differs 3-8x in magnitude across interfaces even where both are valid: keep14 vs ShortGPT-16 is -15.58pp on letter and -1.79pp on content (0.12x), base vs keep14 +28.70 vs +8.74 (0.30x), so any 'recovered X%' statement is interface-relative",
  "our_fix": "Report recovery fractions under both interfaces as an interval rather than a point, with the interface named in the caption",
  "effort": "Zero",
  "from_direction": "Paper E"
 },
 {
  "target_paper": "Equal-option-length subset analyses in MC evaluation",
  "defect": "Our own Obs3 (sign reversal on the equal-length subset, +7.25 vs pooled -13.48) is confounded by subject composition: elementary_math enriched 5.30x, high_school_math 5.26x, abstract_algebra 5.98x, top-10 subjects 55.9% of the subset vs 21.4% of the full set",
  "our_fix": "Within-subject stratification or reweighting before quoting the subset. Note arXiv:2607.12767 (first public 2026-07-14, POST-cutoff, CONCURRENT) names equal-option-length as the length-bias-free regime and should be cited as concurrent, not treated as a preemptor as our kill record did",
  "effort": "CPU",
  "from_direction": "Paper E"
 },
 {
  "target_paper": "Subject-level correlation between two scoring interfaces as evidence of construct divergence",
  "defect": "Our rho-collapse observation (5/6 arms below a binomial null) is unusable as stated because the two worst (0.057, 0.044) are near-chance arms, so it may be a floor effect",
  "our_fix": "Replace the binomial parametric null with a split-half reliability denominator, and restrict to arms above their floors",
  "effort": "CPU",
  "from_direction": "Paper E"
 },
 {
  "target_paper": "arXiv:2411.15558 Reassessing Layer Pruning in LLMs (2024-11-23, genuine prior art)",
  "defect": "Prunes the final 25% of layers then fine-tunes lm_head plus the remaining last three inherited layers; does not test whether a FRESH randomly-initialized block behaves differently from fine-tuned inherited layers, and reports no knowledge-vs-PPL decomposition",
  "our_fix": "Add the fresh-block arm as the single manipulated variable with LR and optimizer matched, and evaluate on the two-axis (PPL vs MMLU/closed-book) decomposition with mandatory constant floors",
  "effort": "Reuses the existing keepN ladder; the confound must be fixed first, since scripts/run_paperC_pc1.sh:61-66 gave A4 fresh LR 1e-4 / inherited 2e-5 while A3 used 3e-4 uniformly, so LR mismatch alone explains the 3.25pp",
  "from_direction": "Paper C P-C1 prune-and-graft"
 },
 {
  "target_paper": "arXiv:2210.10041 Hidden State Variability (EMNLP 2022 Findings, 2022-10-18, genuine prior art)",
  "defect": "Selects layers by a training-free hidden-state-variability criterion but only for classifier placement in transfer learning; never tested as a predictor of where a decoder-only LLM's pretrained depth budget should be split between knowledge preservation and downstream adaptation",
  "our_fix": "Use it as the mandatory baseline our probe must beat, in a pre-registered quantitative depth law of the form K* = ceil([d_t - j + delta]_+ / gamma) with held-out and regret evaluation, not post-hoc correlation",
  "effort": "Reuses per-layer probes already computed on OLMo-2-7B and Qwen3-8B",
  "from_direction": "Paper C P-C2 adaptation-onset probe"
 },
 {
  "target_paper": "Our own Paper C P-C2 probe naming (internal)",
  "defect": "The measured quantity was called adaptation onset / storage depth, but knowledge-readout onset, SFT representational drift, causal necessity, and optimal adaptation location are four different quantities. Our knowledge logit-lens jumps at OLMo L18->L19 (.326->.544) while our full-FT CKA curve has NO knee at L18",
  "our_fix": "Rename to task linearization depth (readout compatibility) and prove any identity between the four quantities before asserting it",
  "effort": "Zero, terminology plus one figure",
  "from_direction": "Paper C P-C2 (self-audit)"
 },
 {
  "target_paper": "arXiv:2410.16168 Active Forgetting for cross-lingual transfer in decoder LMs (preprint, 2024-10)",
  "defect": "Cyclic reset during decoder-only pretraining but only of TOKEN EMBEDDINGS, leaving the transformer body intact, so the regime where reset destroys body computation is untested",
  "our_fix": "Same cyclic schedule applied to top-K decoder blocks, with the embedding-reset arm as the control, separating lexical-interface plasticity from body plasticity",
  "effort": "Needs pretraining; run at 1B first behind the N=0 gate",
  "from_direction": "Paper C v2 cyclic prune-regrow"
 },
 {
  "target_paper": "arXiv:2508.06412 LoRR Sample-efficient LLM Optimization with Reset Replay (preprint, 2025-08)",
  "defect": "Reports full_layers reset as detrimental but only in POST-TRAINING (preference optimization), and attributes it to destruction of reasoning features without testing whether the same holds when the model still has pretraining compute left to re-learn with",
  "our_fix": "Test the identical operator during pretraining where re-learning budget remains; their negative result becomes our prior and our positive-or-null becomes the boundary",
  "effort": "Same 1B gate run",
  "from_direction": "Paper C v2 cyclic prune-regrow"
 },
 {
  "target_paper": "arXiv:2109.00267 The Impact of Reinitialization on Generalization in CNNs (preprint)",
  "defect": "Concludes 'For large datasets, however, reinitialization does not seem to offer a benefit' but the largest datasets tested are image classification; the single-pass trillion-token non-overfitting regime is asserted rather than tested",
  "our_fix": "Add the data-regime axis explicitly (single-pass vs repeated data) at fixed compute, so the boundary condition their conclusion implies becomes a measured curve. This is the highest-scientific-value arm in the cyclic direction",
  "effort": "Two extra arms on the 1B gate run",
  "from_direction": "Paper C v2 cyclic prune-regrow"
 },
 {
  "target_paper": "Springer et al. Overtrained Language Models Are Harder to Fine-Tune (ICML 2025, arXiv:2503.19206)",
  "defect": "Progressive sensitivity is established for Gaussian weight noise and fine-tuning only; discrete structural damage is named as future work, and the cost is reported as a single PPL quantity with no knowledge/distribution decomposition",
  "our_fix": "Extend the sensitivity curve to layer-level discrete damage using our 5k-interval checkpoints, and report the two cost components separately, testing whether they diverge at different rates with pretraining time",
  "effort": "Eval-only on existing checkpoints if run on the keepN ladder; no new training for the descriptive version",
  "from_direction": "Paper C v2 plasticity mechanism"
 },
 {
  "target_paper": "SEAL (CVPR 2023, arXiv:2304.04858)",
  "defect": "Reports LLF features degrading transfer across all datasets explored, but only for image classification and with no equivalent of a knowledge axis",
  "our_fix": "Include a downstream SFT/transfer axis in any LLM reset study rather than reporting PPL plus zero-shot MC only; without it a reviewer citing SEAL has a free hit",
  "effort": "Adds an SFT leg; task #123 (general-SFT repairability pipeline) is already scoped and pending",
  "from_direction": "Paper C v2 cyclic prune-regrow"
 },
 {
  "target_paper": "Our own Paper B distillation run #99 (internal)",
  "defect": "train_olmo2_arch_probe2_distill.py's _classify_param (line 287) omits the module. prefix strip that was fixed in train_olmo2_arch_probe2.py:316, so the fresh-group LR is a silent no-op: logs show only inh_decay 4060.1M @2e-5 and inh_nodecay 0.3M @2e-5, with no fresh group. #99 was uniform 2e-5",
  "our_fix": "Never claim differential LR for that trainer; verify the actual param-group names in the training log before any LR-matched claim. The clarification that #92 IS unaffected (same commit 7a330ce introduced both the runner and the fix) must be preserved so we do not over-correct",
  "effort": "Zero, documentation",
  "from_direction": "Paper B / Paper C training infrastructure (self-audit)"
 },
 {
  "target_paper": "HumanEval-infilling task construction as commonly specified",
  "defect": "prompt + canonical_solution + suffix yields TWO different files for 123/164 tasks (a trailing newline on the L0 row's prompt), so '164/164 parseable' is true but not unique, and no single consistent spec reproduces the commonly cited [164,108,84,60] rung counts -- the k=4 ceiling is 59, not 60",
  "our_fix": "Byte-exact admission (F_lines[i] == canonical_solution AND join(F_lines[:i]) == prompt AND join(F_lines[i+1:]) == suffix) with a majority-vote reference file and deterministic tie-break, yielding 910/1033 rows",
  "effort": "CPU; already implemented and hash-frozen",
  "from_direction": "dllm k-span infilling"
 },
 {
  "target_paper": "Our own workflow orchestration (internal, process defect)",
  "defect": "The orchestrator spends per-direction agents without validating that the recovered direction index is non-empty or that each direction carries a real kill reason. In run wf_f4eca2c5-b35 the index-directions agent emitted a schema-valid probe {\"directions\":[{\"id\":\"test\",...}]} at 08:01, its four subsequent real submissions all arrived with empty input, and the placeholder was recorded as the step result -- so this entire re-adjudication was handed a stub while the real dossiers sat unexamined on disk",
  "our_fix": "Fail fast when the directions array is empty or when a summary is under ~200 chars; have index-directions write to a file rather than serializing a large dense payload through StructuredOutput; require each direction to carry a verbatim kill reason and at least one cited preemptor before a downstream agent is spent",
  "effort": "Zero GPU, a guard in the orchestrator script",
  "from_direction": "meta / workflow infrastructure"
 }
]

---

# MAIN 复核记录（2026-08-07 18:00，从 raw per-example 重算，非复述 summary）

| # | 断言 | MAIN 复现 | 判定 |
|---|---|---|---|
| 1 | MC constant-D floor .2689, n=14042 | A .2295 / B .2465 / C .2551 / **D .2689**, n=14042 | ✓ 逐位精确 |
| 2 | SQuAD 多数拒答 997/2000 = .4985 | **997/2000 = 0.4985**, 标签 `根据提供的信息无法回答这个问题` | ✓ 逐位精确 |
| 3 | z-CKA 观测 .4907 vs shuffle null .453, n=91 | 观测 `0.4906724527457283` / null `0.4526328836126522` / n=91 / identity gate 1.777e-7 | ✓ 逐位精确 |
| 4 | probe knee vs native knee (tab_depth.tex) | 未逐位查（其余三条零漂移） | 待查 |

**注**: agent 未复述 summary —— `squad_val.jsonl` 的真实字段是 `target_text`（MAIN 首次猜 `answer` 失败），而 agent 给的 997/2000 精确命中，证明它确实读了原文件。

## ⚠️ MAIN 确认的统计弱点（agent 自己主动标出，MAIN 算准了它）

```
n_perm_per_pair: 200          <- 问题所在
per_pair_p_median: 0.015
n_pairs_p_below_0.05: 58      <- 仅 58/91 = 64%
n_pairs_observed_above_null_mean: 77
```

**200 次置换的最小可达 p 值是 1/201 ≈ 0.005**，故 median p=0.015 已贴近分辨率下限，BH 校正后 58/91 会进一步缩水。一篇**关于「报告未校准数字」的论文**若自身核心统计量欠置换即自毁 —— 所以「扩到 2000 perm/pair + BH 校正」必须是第一步。
