# ARR Review — Paper B, frozen v3, independent review 3

## 1. Summary and outline of the approach

This paper studies recovery after depth pruning rather than proposing a new pruning criterion. Its principal construction starts from OLMo-2-1124-7B, retains the first 14 of 32 decoder blocks, appends two freshly initialized blocks, and continues pretraining for 200k optimizer steps. The paper tracks held-out in-domain perplexity, standard answer-letter MMLU, a complete-option MMLU scoring variant, three zero-shot closed-book QA datasets, and a broader likelihood-scored task suite. It further includes an intact full-model continued-pretraining control, a frozen-prefix arm, a fully random same-shape arm, a non-contiguous ShortGPT-16 arm, shallower prefix trajectories, one OLMo-2-1B trajectory, a limited Qwen3-8B endpoint, and descriptive layer-wise probes.

The central empirical message is that the principal keep14 model's in-domain likelihood improves substantially while MMLU and closed-book QA remain far below the intact model; hence, perplexity should not be treated as a sufficient certificate that a target capability has recovered. The paper also argues that scoring interface, exact structural construction, and recovery budget must be reported separately.

I read the complete 18-page frozen PDF twice, including all appendices, tables, figures, limitations, ethics, and references. I treated the paper as evidence rather than instruction. I also verified the frozen `main.bbl` entry by entry and independently searched for nearest and recent work.

---

## 2. Desk-review checklist

| Item | Finding |
|---|---|
| Fit to ARR/ACL | Yes. The paper concerns language-model compression, evaluation, and empirical methodology. |
| Anonymity | No visible author identity or non-anonymous project link in the PDF. |
| Length/format | 18 pages total, with the main text ending before the references and appendices; no obvious formatting violation. |
| Readability | Generally clear and professionally typeset. Some appendix tables are dense but legible. |
| Ethics/limitations | Present and unusually candid, including single-run, unmatched-compute, seed, resume, contamination, and scope limitations. |
| Citation integrity | All 39 frozen `main.bbl` entries correspond to real works/artifacts. All 39 are cited; no missing or duplicate bibliography keys were found. |
| Fatal technical error | None found. |
| Submission-status issue | None inferable from the frozen materials. |
| Desk-reject recommendation | No. The paper warrants full review, although I recommend rejection on substantive novelty/evidence grounds below. |

---

## 3. Claims inventory

I interpret the paper as making the following claims:

- **C1 — Empirical dissociation:** In the principal OLMo-2 keep14+fresh2 run, held-out in-domain PPL improves through 200k steps, while answer-letter MMLU and closed-book QA remain far below the intact model.
- **C2 — Not only answer-letter formatting:** Complete-option MMLU narrows the gap, but the random-init content-score floor and the PopQA/TriviaQA/NQ-open deficits show that the main deficit is not merely answer-letter formatting.
- **C3 — Short-horizon corpus drift is insufficient:** The intact full32 control remains near the base at its observed step-25k checkpoint, so short-horizon continued-training drift does not explain the half-depth gap.
- **C4 — Construction matters:** A 16-layer ShortGPT construction reaches a substantially stronger endpoint than keep14 at 200k, so nominal depth alone does not determine the endpoint.
- **C5 — Recovery rate depends on inherited prefix depth within the observed recipe:** keep8 shows little MMLU change over its later observed interval, whereas keep14 shows a small but statistically detectable late MMLU gain.
- **C6 — Domain heterogeneity:** MMLU recovery differs across broad subject groups and individual subjects.
- **C7 — Cross-setting directionality:** OLMo-2-1B and a Qwen3-8B endpoint provide qualitative or directional support for the main pattern.
- **C8 — Descriptive readout ordering:** Semantic probes become readable earlier than MMLU letter readout, while exact next-token agreement emerges latest; this is explicitly non-causal.
- **C9 — Methodological recommendation:** Prune-then-heal studies should separately report likelihood, target capability, interface, exact structure, and recovery compute, ideally with an intact same-corpus control.
- **C10 — Novelty/positioning:** The recovery trajectory plus the collection of same-corpus, same-shape, interface, construction, and closed-book controls is presented as the paper's distinguishing contribution.

---

## 4. Strengths

### S1. The paper is unusually disciplined about the scope of causal interpretation.

The authors repeatedly distinguish observations from causal claims. For example, the ShortGPT comparison is explicitly described as differing in four coupled dimensions, and the paper states that further matched structures are needed before attributing the gap to block 31, non-contiguous selection, inherited count, or fresh tails (Section 6.3, PDF p.6, lines 399–418). The same caution appears for the probe analysis (Appendix C, Figure 7/Table 20, PDF p.15, lines 991–999; PDF p.17, lines 69–81). This is excellent scientific hygiene.

### S2. The main empirical phenomenon is clearly exposed by complementary evaluations.

Table 2 places PPL, two MMLU interfaces, and three closed-book tasks side by side (Section 5.2, PDF p.5, table lines 1–17). This makes the central result easy to assess: keep14 reaches PPL 10.561 but remains at .319 MMLU letter, .142 PopQA, .294 TriviaQA, and .060 NQ-open, all well below the base. The use of both likelihood-scored and generated-answer tasks is materially stronger than relying on one MMLU protocol.

### S3. The interface analysis is thoughtful and avoids overclaiming.

The paper does not equate complete-option scoring with recovered knowledge. It explicitly uses the random-init arm to reveal a fluency floor (Section 5.2, PDF p.4, lines 302–311; Table 16, PDF p.17, lines 55–81) and notes that the two MMLU protocols also change prompt, continuation, tokenization, and normalization (Discussion, PDF p.8, lines 454–470). This is a useful caution for benchmark interpretation.

### S4. The appendices contain substantial provenance and integrity information.

The paper reports optimization hyperparameters, precision, batch size, architecture transplant checks, exact sample counts, truncations, PPL aggregation, and strict checkpoint reconstruction (Appendix B, PDF p.13, lines 882–920; PDF p.14, lines 921–975). The claim that PPL is token-weighted rather than averaged across shard perplexities is particularly welcome. The authors also disclose the missing seed and data-loader offset.

### S5. The statistical treatment of selected MMLU comparisons is better than is typical for large-model compression papers.

The paper gives marginal uncertainty for the full MMLU frontier (Table 12, PDF p.16, lines 30–51), exact McNemar tests and paired-bootstrap intervals for the three 200k same-shape operating points (Table 15, PDF p.17, lines 32–49), and a paired late-trajectory comparison for keep14 (Table 13, PDF p.16, lines 46–54). These analyses appropriately use item-level pairing where predictions are available.

### S6. Limitations are transparent rather than hidden.

The limitations section clearly acknowledges single training runs, unmatched full32 and shallow-arm budgets, the higher learning rate for random initialization, the changed trainable set for frozen-front, coupled ShortGPT differences, in-domain PPL, no contamination audit, and limited model/corpus coverage (PDF p.9, lines 531–575). This materially improves trust in the reported observations.

### S7. The writing and figures are generally effective.

Figure 1 communicates the setup and endpoint gap cleanly (PDF p.2, lines 36–40); Figures 2–5 accurately qualify their conclusions; Figure 3 explicitly avoids claiming global non-correlation (Section 5.2, PDF p.5, lines 322–330). I found no visually misleading figure. All 7 figures and 22 tables were inspected; axes, legends, captions, and table values are mostly consistent and readable.

---

## 5. Weaknesses

### W1. **Major — The novelty claim is materially overstated relative to the closest prior work.**

- **Location:** Contributions and Related Work/Table 1; PDF p.2, lines 98–112 and 115–141; PDF p.3, Table 1 lines 1–11.
- **Short quote:** “Its distinguishing object is the recovery trajectory”
- **Weakens:** **C10**, and therefore Excitement/novelty.
- **Why this matters:** The closest literature already contains important parts of the claimed increment:
  - **Gromov et al. (first posted March 26, 2024)** explicitly show a post-healing decoupling between next-token cross-entropy and MMLU/BoolQ and describe this as a “miscalibration” between continuous LM loss and downstream tasks.
  - **Shortened LLaMA (February 5, 2024)** reports zero-shot performance over the course of CPT, including learning curves across pruning ratios and a pruned-vs-random-initialization training comparison.
  - **Minitron / Compact Language Models (July 19, 2024)** reports retraining loss curves, a ranking flip during retraining, random-init versus pruned training, MMLU/HellaSwag outcomes, and one-shot versus iterative prune/retrain studies.
  - **SlimQwen (May 9, 2026)** studies pruned versus random initialization under matched training tokens, training-loss trajectories, depth/width/expert compression, and progressive recovery.
  - **ShortOPD (July 14, 2026; inside the three-month rule)** explicitly frames recognition and generation as partly independent recovery axes and supplies a trajectory-level recovery analysis.

  The current Table 1 labels prior trajectory coverage as “no” or “limited,” but this is too coarse and makes the paper appear more distinct than it is. The paper's strongest novelty is narrower: a carefully controlled OLMo-2 case study combining in-domain PPL, answer-letter/content MMLU, closed-book QA, an intact same-corpus control, and a policy comparison. That package may still be useful, but it is not convincingly established as a major new methodological contribution.
- **Remedy:** Rewrite the novelty claim and Table 1 around the *specific combination of controls not jointly present in prior work*, rather than trajectory analysis per se. Add direct comparisons to Gromov, Shortened LLaMA, Minitron, SlimQwen, and ShortOPD, with a column-level accounting of which axes each already studies. If a stronger novelty claim is desired, add a genuinely new matched experiment or analysis that those papers do not contain.

### W2. **Major — The central “PPL versus capability recovery” conclusion is not identified at matched quality or matched compute.**

- **Location:** Main result and Discussion; PDF p.4, lines 292–318; PDF p.5, lines 322–342; PDF p.8, lines 437–453.
- **Short quote:** “PPL does not uniquely determine answer-letter MMLU”
- **Weakens:** **C1, C4, and C9** as general conclusions rather than descriptive facts.
- **Why this matters:** Every principal compressed endpoint still has substantially worse PPL than the intact base (best half-depth PPL 9.780 versus 7.398), so the paper does not show a model whose likelihood has actually recovered to the intact level while capability remains impaired. Nor does it compare models at equal PPL along their trajectories. Across the displayed endpoints, PPL and MMLU mostly co-vary, as the paper itself concedes. The clearest ordering reversal is random-init versus frozen-front, but that comparison changes both initialization and learning rate/trainable modules. Therefore, the evidence supports the modest statement that PPL alone is not a sufficient scalar summary across heterogeneous interventions; it does not cleanly characterize recovery rates conditional on matched distributional fit or matched optimization.
- **Remedy:** Add one or more of:
  1. matched-PPL checkpoint comparisons across arms using interpolation or denser saved checkpoints;
  2. matched-token/FLOP trajectories for keep14, ShortGPT, random-init, and frozen-front;
  3. regression or partial-rank analyses separating PPL, structure, and compute;
  4. an arm healed until it reaches near-base PPL, followed by capability evaluation.
  
  Without this, narrow the title/abstract and recommendations to “PPL is not a sufficient cross-intervention summary in these observed runs.”

### W3. **Major — Training-run variance is absent for every key model comparison.**

- **Location:** Limitations and Appendix B; PDF p.9, lines 531–539; PDF p.14, lines 921–931.
- **Short quote:** “same-shape controls are single training runs”
- **Weakens:** **C1–C7**, especially claims about recovery rate, domain heterogeneity, and construction effects.
- **Why this matters:** Item-level confidence intervals quantify evaluation uncertainty for a fixed checkpoint, not uncertainty from fresh-tail initialization, data order, optimizer stochasticity, or the resumed iterator. This is particularly important because:
  - fresh blocks are randomly initialized;
  - no explicit seed was set;
  - the principal keep14 run resumed with an altered within-epoch data position;
  - the late MMLU gain is only 1.68 points;
  - comparisons such as keep14 versus ShortGPT or frozen-front combine training and construction effects.
  
  The current evidence cannot establish that the observed trajectory shapes or endpoint gaps are stable across training runs.
- **Remedy:** Repeat at least the principal keep14, ShortGPT, frozen-front, and random-init arms for multiple seeds, or provide a less expensive replicated subset/checkpoint study. Report between-run intervals for PPL and headline capabilities. At minimum, replicate the 128k→200k keep14 segment from multiple saved states or initializations.

### W4. **Major — Several controls do not isolate the interpretations assigned to them, and the intact control is too short for the main horizon.**

- **Location:** Control branches and controlled comparisons; PDF p.3, lines 212–220; PDF p.5, lines 331–342; PDF p.6, lines 343–359.
- **Short quote:** “uses a higher learning rate”
- **Weakens:** **C2, C3, and the control-related part of C9**.
- **Why this matters:** The random-init arm uses a five-times higher peak LR and randomizes embeddings, final norm, and output head, so it is not an initialization-only control and may not provide a clean “fluency floor” for the inherited model. Frozen-front changes the trainable parameter set and effective optimization problem. The full32 control ends at 25k while keep14 is interpreted at 200k; thus it only rules out short-horizon drift, not long-budget data/corpus effects. The paper states these caveats, but the contribution still advertises controls that “separate interpretations” more strongly than the experiments actually do.
- **Remedy:** Add:
  - random-init with the inherited-arm LR and, ideally, a factorial control that preserves lexical modules but randomizes decoder blocks;
  - a trainable-parameter- or update-matched frozen/adaptation control;
  - full32 CPT through 200k or token/FLOP matching;
  - explicit preregistration/stopping criteria for the 25k plateau.
  
  Otherwise, rephrase these as informative operating points rather than separating controls throughout the abstract/contributions.

### W5. **Minor — The statistical coverage is selective and does not support all cross-interface and closed-book claims.**

- **Location:** Table 2, Section 5.2, and Appendix Tables 12–17; PDF p.5, lines 1–17 and 319–342; PDF pp.16–17.
- **Short quote:** “Together with the two MMLU protocols, they rule out an answer-letter-only explanation.”
- **Weakens:** **C2, C6**, and the strength of the word “rule out.”
- **Why this matters:** MMLU letter comparisons receive item-paired tests, but complete-option MMLU, PopQA, TriviaQA, NQ-open, domain-group comparisons, and most broad-suite differences do not receive paired uncertainty estimates. Closed-book metrics also use different normalizations (containment versus exact match), and the generation details required to reproduce those outcomes are not fully specified in the paper. The qualitative direction is persuasive, but “rule out” is stronger than the reported inferential support.
- **Remedy:** Report paired bootstrap intervals/tests for content MMLU, each closed-book task, and key domain-group differences; define correction policy for multiple comparisons; replace “rule out” with “argue against” unless the expanded analyses support the stronger wording.

### W6. **Minor — Reproducibility remains low because neither the training randomness nor a runnable artifact is recoverable from the frozen paper.**

- **Location:** Appendix B; PDF p.13, lines 885–920; PDF p.14, lines 921–941.
- **Short quote:** “not reproducible from the command line alone”
- **Weakens:** Reproducibility of **C1–C8**.
- **Why this matters:** The paper provides many hyperparameters, but the random seed was unspecified, the data-loader offset was not saved, failed/exploratory runs were not uniformly logged, exact GPU hours are unavailable, and the frozen submission does not provide code, configs, prediction files, or checkpoints. The closed-book decoding parameters and the WiC/SST-2/RTE probe-training protocol are also under-specified.
- **Remedy:** Release anonymized code/configs, exact corpus-array construction, checkpoint metadata, per-item predictions, closed-book decoding and normalization code, probe train/dev/test splits and regularization, and deterministic seed/data-loader handling. Archive the precise ShortGPT selection/calibration script and layer list.

### W7. **Minor — There is an internal inconsistency in the description of paired predictions/tests.**

- **Location:** Main controlled-comparison text and Appendix A.3 versus Table 15; PDF p.5, lines 331–342; PDF p.12, lines 839–864; PDF p.17, lines 32–49.
- **Short quote:** “frozen-front is not included”
- **Weakens:** Trust in the statistical provenance.
- **Why this matters:** The main text says paired tests confirm keep14 over “each control.” Table 15 indeed reports keep14–random, keep14–frozen, and frozen–random. However, Appendix A.3 says “All five differences favor keep14” and then says frozen-front is not included because its per-example predictions were not retained. This directly contradicts Table 15's frozen-front paired results and the stated three comparisons.
- **Remedy:** Correct the appendix narrative and explain which per-item rerun produced Table 15. Harmonize the exact aggregate values across the canonical table, rerun table, and paired tests.

### W8. **Minor — The paper uses “pre-registered,” “pre-specified,” and “plateau” terminology without enough procedural evidence.**

- **Location:** Introduction, method, and Appendix A/B; PDF p.1, lines 76–80; PDF p.3, lines 205–211; PDF p.12, lines 809–827; PDF p.13, lines 891–900.
- **Short quote:** “pre-registered 25k plateau checkpoint”
- **Weakens:** **C3** and confidence in stopping-rule neutrality.
- **Why this matters:** The paper provides no registry or timestamped protocol and does not define the plateau, evaluation schedule, or stopping threshold. Shallower arms are stopped when knowledge-sensitive metrics had stabilized even though PPL was still falling, which introduces metric-dependent stopping and complicates the depth-ladder interpretation.
- **Remedy:** State the exact preregistration artifact or remove “pre-registered.” Define plateau mathematically, list checkpoint-evaluation cadence, and report all available checkpoints under a common stopping rule.

### W9. **Minor — The mechanistic probe section is insufficiently motivated and incompletely specified.**

- **Location:** Section 4 and Appendix C; PDF p.4, lines 273–280; PDF p.14, lines 976–990; PDF p.15, lines 991–999.
- **Short quote:** “semantic < MMLU < next-token”
- **Weakens:** **C8** and paper focus.
- **Why this matters:** The ordering combines three different quantities, thresholds, datasets, and readout procedures. The paper appropriately calls it descriptive, but the semantic classifiers' training splits, regularization, hyperparameter selection, seed, and uncertainty are not reported. The analysis does not directly test the healed models or explain the endpoint gap, so it currently reads as an intriguing but weakly connected appendix.
- **Remedy:** Either remove/de-emphasize this section or fully specify and replicate the probes, report uncertainty, and connect them to matched interventions in the healed models.

### W10. **Minor — The paper does not demonstrate an efficiency benefit of its principal pruning recipe.**

- **Location:** Method and Discussion; PDF p.3, lines 174–211; PDF p.8, lines 498–510.
- **Short quote:** “does not replace efficiency or endpoint-quality comparisons”
- **Weakens:** Practical significance/Excitement, though not the core measurement claim.
- **Why this matters:** Parameter count is reported, but latency, throughput, memory, FLOPs, and end-to-end recovery cost are not. Since the setup is motivated by smaller language models, readers cannot assess whether the studied construction is practically attractive or whether ShortGPT's stronger endpoint has a different serving profile.
- **Remedy:** Report parameter/FLOP reduction, measured latency and throughput on the stated hardware, memory use, and recovery compute for all principal constructions.

---

## 6. Citation and related-work audit

### 6.1 Frozen `main.bbl` authenticity

I checked all **39** entries in the frozen `main.bbl`. Every entry corresponds to a real paper, proceedings article, technical report, dataset paper, or—only for the original logit-lens item—a real LessWrong post. The three entries not reliably returned by title-only bibliographic search were independently verified by their arXiv identifiers: ARC (1803.05457), A Deeper Look at Depth Pruning (2407.16286), and HellaSwag (1905.07830). I found no fabricated title, author list, or identifier. Minor bibliographic simplifications exist (e.g., arXiv versus later venue year), but no material authenticity issue.

### 6.2 Citation-to-claim matching checks

I checked the following citation clusters against the cited works:

1. **Depth pruning plus healing/CPT** — Gromov, Shortened LLaMA, ShortGPT, and LaCo support the introductory statement that block removal followed by retraining/healing is an established compression route.
2. **Aggregate LM metrics can miss other behavior** — The Cost of Compression supports parametric-knowledge effects; Jaiswal et al. support knowledge-intensive benchmark degradation; Beyond Perplexity supports divergent safety effects. The citation cluster matches the sentence.
3. **Selection methods** — ShortGPT/SLEB rank blocks, LaCo merges layers, and BlockPruner searches finer attention/MLP units. This is accurately described.
4. **Calibration/task dependence** — Siddiqui, Lu et al., and Kim et al. (2026) support the claim that preferred removals depend on task/calibration/search setup.
5. **Gromov's PPL/downstream mismatch** — Strong match. Gromov explicitly reports post-healing continuity in C4 loss across pruning fractions where MMLU/BoolQ show sharp transitions.
6. **Shortened LLaMA's CPT versus LoRA comparison** — Strong match. The paper directly compares retraining methods and reports CPT learning curves.
7. **Sheared/Minitron/compact/DRPruning with data allocation or teacher supervision** — Broadly accurate, although Minitron's trajectory and random-init analyses deserve fuller discussion because they are close to this submission's positioning.
8. **Logit lens/tuned lens and factual-localization work** — The citations appropriately support the descriptive background; the submission correctly avoids treating probe thresholds as causal storage locations.

The main citation problem is therefore **omission/under-positioning**, not false citation. In particular, PASER (February 18, 2025), SlimQwen (May 9, 2026), and ShortOPD (July 14, 2026) are relevant to recovery controls/trajectories and should be discussed.

---

## 7. Novelty search and three-month rule

I posed five independent search questions:

1. **Has prior work already shown post-pruning LM-loss/perplexity and downstream-task dissociation?**  
   Yes. Gromov et al. (March 26, 2024) is the closest direct antecedent.

2. **Has prior work tracked capability during CPT/healing rather than only endpoints?**  
   Yes. Shortened LLaMA (February 5, 2024) includes training-progress and multi-ratio CPT learning curves. Minitron (July 19, 2024) includes retraining loss curves and task trajectories/iterative depth experiments.

3. **Has prior work compared pruned initialization against random initialization under recovery training?**  
   Yes. Shortened LLaMA and Minitron do so; SlimQwen (May 9, 2026) performs a broader matched-token study.

4. **Has prior work emphasized that distinct capabilities/interfaces recover differently after compression?**  
   Yes. The Cost of Compression, Jaiswal et al., Beyond Perplexity, and the December 27, 2025 “Fragile Knowledge, Robust Instruction-Following” paper all support capability-selective effects. The latter is width-pruning rather than depth-pruning, so it is adjacent rather than a direct duplicate.

5. **Are there papers within three months of the frozen version that materially affect novelty?**  
   Yes:
   - **SlimQwen**, first posted **May 9, 2026**: pruned versus scratch initialization, training-loss trajectories, progressive pruning, and knowledge-sensitive benchmarks.
   - **ShortOPD**, first posted **July 14, 2026**: recovery trajectory, multiple-choice versus free-generation dissociation, and recovery-method controls.
   
   Under the ARR three-month rule, I do **not** treat these as prior-art grounds for rejecting the paper's novelty. They do, however, show that the claimed methodological territory is currently crowded and should be acknowledged as concurrent work. My novelty concern is independently supported by older work outside that window, especially Gromov, Shortened LLaMA, and Minitron.

**Nearest work:**  
For the *phenomenon*, Gromov et al. is nearest. For *CPT trajectories and pruned-vs-scratch recovery*, Shortened LLaMA is nearest. For *multi-axis compression/retraining analyses including MMLU and random-init controls*, Minitron is nearest. The present paper's distinctive element is the exact bundle of OLMo-2 controls and interfaces, not the underlying observation that recovery is multi-dimensional.

---

## 8. Claim-by-claim technical and experimental assessment

| Claim | Evidence in paper | Ideal experiment / baseline | Assessment |
|---|---|---|---|
| C1 | keep14 trajectory, Table 2, Figure 2, Table 13 | Replicated seeds; near-base-PPL endpoint; matched-PPL analysis | **Supported descriptively for this run**, not as a seed-stable or matched-quality law. |
| C2 | Content MMLU, random-init floor, closed-book QA | Paired CIs for all interfaces; exact generation protocol; lexical-module-matched random control | **Directionally supported**, but “rules out” is too strong. |
| C3 | full32@25k | full32@200k or matched tokens/FLOPs; explicit stopping rule | **Supported only for short-horizon drift**, exactly as the limitations concede. |
| C4 | keep14 versus ShortGPT at 200k | Factorial matched structures isolating inherited count, final block, contiguity, and fresh tail | **Endpoint difference is supported; explanation is not identified.** |
| C5 | keep8 45k→121k and keep14 128k→200k | Same step interval, same starting quality, multiple seeds, formal interaction test | **Suggestive but not a clean inherited-depth interaction.** |
| C6 | group and subject tables | Group-level paired bootstrap, multiplicity control, replicated runs | **Descriptively supported; inferential generality is limited.** |
| C7 | one OLMo-1B arm and one unmatched Qwen endpoint | Matched recipes across several families/scales and controls | **Only qualitative/directional**, appropriately labeled. |
| C8 | logit lens and semantic probes | Full probe protocol, uncertainty, healed-model probes, causal interventions | **Descriptive only; weak connection to core claims.** |
| C9 | Synthesis of controls and observed gaps | Demonstration across multiple papers/models; ablation showing each checklist item changes conclusions | **Reasonable advice, but not yet established as a validated protocol.** |
| C10 | Table 1 and related-work narrative | Fine-grained nearest-work matrix and concurrent-work discussion | **Overstated in current form.** |

---

## 9. Questions for the authors

1. What exact artifact and date justify the term “pre-registered” for the full32 plateau checkpoint and “pre-specified” for the content metric?
2. Were per-item frozen-front predictions retained or not? How should Table 15 be reconciled with Appendix A.3?
3. What were the exact decoding parameters and answer-normalization rules for PopQA, TriviaQA, and NQ-open?
4. Can the authors provide matched-PPL or matched-token comparisons among keep14, ShortGPT, frozen-front, and random-init?
5. How variable are the keep14 late gain and ShortGPT endpoint across training seeds?
6. Why are Gromov's loss/task decoupling, Shortened LLaMA's CPT curves, and Minitron's retraining trajectories not treated as closer antecedents to the “trajectory-level” contribution?

---

## 10. Overall assessment

This is a careful, transparent, and useful empirical case study. I believe the central *descriptive* observation for the reported OLMo-2 runs: in-domain likelihood, MMLU interfaces, and closed-book QA do not recover at the same observed rate, and exact construction matters. The paper's strongest qualities are its caution about causality, its broad appendix, and its refusal to equate complete-option scores with knowledge.

However, the paper currently sells a broader methodological novelty than the literature supports. The closest older work already studies healing-induced loss/task decoupling, CPT trajectories, random-init baselines, and recovery dynamics. Moreover, the main empirical conclusion is not tested at matched PPL, matched long-horizon compute, or across training seeds. The controls are informative but not cleanly isolating, and the full32 control covers only 25k of the 200k horizon. These limitations leave the paper as a strong diagnostic report rather than a sufficiently novel and conclusive ARR contribution.

I would be open to a substantially revised version that (i) sharply narrows and accurately positions the novelty, (ii) adds matched-PPL/compute analyses, and (iii) includes at least limited training-seed replication.

---

## 11. ARR scores

### Soundness: **3 / 4 — Good**

The reported numbers, evaluation integrity checks, and narrow descriptive conclusions appear credible. I found no fatal technical flaw. The score is not 4 because key comparisons lack training-seed variance, several controls are confounded, and the main relation is not tested at matched PPL/compute.

### Excitement: **2 / 4 — Mediocre**

The control bundle and OLMo-2 evidence are useful, but the high-level phenomenon and much of the trajectory/control framing overlap substantially with prior work. Practical impact is also hard to judge without efficiency measurements.

### Overall recommendation: **2 / 5 — Reject**

The paper contains solid observations and commendable reporting, but the novelty is overstated and the evidence does not yet support the breadth of the methodological claim. A more narrowly framed paper with matched-quality/compute analysis and seed replication could become competitive.

### Confidence: **4 / 5 — High**

I read the full paper and appendices twice, inspected every figure and table, audited all bibliography entries, checked multiple citation-to-claim pairs, and searched both older and three-month-window related work. My remaining uncertainty concerns unavailable training artifacts and variance, not paper comprehension.

### Reproducibility: **2 / 4 — Low**

The paper gives substantial implementation detail, but exact reproduction is prevented by absent seeds, the altered resumed data-loader position, no frozen runnable artifact/checkpoints/predictions, incomplete closed-book/probe details, and no exact project compute accounting.

---

## 12. Self-check

- Every listed strength and weakness is anchored to a section/table/figure and PDF page/line range.
- Every weakness includes a location, a short quote of at most 25 words, the affected claim/norm, a remedy, and Major/Minor severity.
- I checked all short quotes against the frozen PDF text.
- I avoided unsupported generic absence claims; each absence is tied to a specific claim and concrete consequence.
- I did not use or inspect any pre-existing review, report, score-history, or review-history markdown file.
