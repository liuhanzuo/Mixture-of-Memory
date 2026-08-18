```yaml
review_mode: strict
soundness: 3.0
excitement: 2.5
overall: 2.5
confidence: 4.5
reproducibility: 2.0
```

# Paper Summary

This paper is an observational study of post-pruning recovery in OLMo-2. The principal construction keeps the first 14 of 32 pretrained blocks, appends two fresh blocks, and continues pretraining for 200k optimizer steps. The paper measures held-out in-domain perplexity, answer-letter and complete-option MMLU, three closed-book QA tasks, and a broader zero-shot likelihood suite. Its central empirical observation is deliberately narrow: in one keep14 run, likelihood improves substantially but the tested knowledge-sensitive behaviors remain far below the intact model. A short-horizon unpruned control, a frozen-front arm, a fully random same-shape arm, and a non-contiguous ShortGPT-16 endpoint are used as complementary—but mostly unmatched or confounded—operating points. The paper proposes that likelihood, target capability, evaluation interface, exact construction, and recovery budget be reported separately.

The strongest numerical observations are: (i) keep14 reaches PPL 10.561 versus 7.398 for the intact base (1.428×); (ii) answer-letter MMLU is .319 versus .605; (iii) complete-option MMLU is .383, but random-init is .360; (iv) PopQA/TriviaQA/NQ-open are .142/.294/.060 versus .257/.636/.205; and (v) ShortGPT-16 reaches PPL 9.780 and MMLU .474 at 200k. These numbers are internally consistent with the reported tables up to documented rerun/rounding differences.

# Claims and Evidence Map

- **C1 — Measurement claim:** In the realized keep14 run, in-domain PPL improves substantially while answer-letter MMLU and closed-book QA remain far below the intact base. **Minimum sufficient evidence:** a within-run trajectory plus identical evaluation of keep14 and intact base. **Actual evidence:** Fig. 1, Table 2, Tables 4/10/13/17. This is supported as a descriptive, observed-budget claim.
- **C2 — Measurement claim:** The MMLU scoring interface materially changes apparent recovery, but complete-option scoring has a high non-knowledge floor. **Minimum sufficient evidence:** both interfaces on the same items and controls that can reveal generic fluency. **Actual evidence:** Table 16, especially keep14 .3832, frozen .3604, random .3598. Supported descriptively, but the protocols change several factors jointly and have no paired uncertainty analysis.
- **C3 — Bounded corpus-drift claim:** Short-horizon corpus shift alone is insufficient to explain the observed half-depth deficit. **Minimum sufficient evidence:** intact CPT on the same corpus/schedule over the relevant horizon. **Actual evidence:** full32 only through 25k (Table 2/17), whereas keep14 is evaluated at 200k. Supported only for the explicitly stated short horizon; it is not a 200k counterfactual.
- **C4 — Construction claim:** Nominal 16-layer depth does not determine a unique observed endpoint. **Minimum sufficient evidence:** at least two 16-layer constructions trained/evaluated under a common budget. **Actual evidence:** keep14 and ShortGPT both reach 200k, with .319 versus .474 MMLU and 10.561 versus 9.780 PPL (Table 3). Supported for these two single realizations only; no individual structural factor is isolated.
- **C5 — Training-dynamics claim:** Late keep14 healing continues but does not show broad catch-up over the observed interval. **Minimum sufficient evidence:** repeated checkpoints on the same run with item-paired downstream comparisons. **Actual evidence:** 128k/153.5k/200k trajectory and paired MMLU rerun (Table 13). Supported conditionally on this training run. Importantly, the 95% CI is **item-level uncertainty**, not **training-seed uncertainty**.
- **C6 — Domain claim:** Recovery varies across broad MMLU groups. **Minimum sufficient evidence:** group scores plus uncertainty or formal heterogeneity testing if inferential. **Actual evidence:** Fig. 3/Table 18 provide descriptive group differences only. Supported as description, not as a tested domain-specific effect.
- **C7 — Reporting proposal:** Recovery should be reported as multiple axes rather than as PPL alone. **Minimum sufficient evidence:** either broad validation across methods/models or a clearly framed case-study recommendation. **Actual evidence:** one principal OLMo run plus limited 1B/Qwen scope checks. Reasonable as a case-study suggestion, not validated as a universal protocol.
- **C8 — Causal/knowledge-localization claim:** The paper does **not** claim that removed blocks store a specific knowledge type or that block 31/fresh tails cause the endpoint gap. The manuscript consistently labels its layerwise analyses as readouts rather than causal localizations (Sec. 6.1, Sec. 7, App. C). I agree with this bounded scope.

# Strengths

## S1. Unusually disciplined claim bounding
The paper repeatedly distinguishes observations from causal claims: Sec. 3.3 calls the controls “operating points”; Sec. 6.3 lists the four coupled ShortGPT differences; Sec. 7 says localization requires structure-isolating interventions; and App. C explicitly excludes the readout plots from the recovery evidence. This is much better calibrated than many pruning papers.

## S2. The control bundle exposes genuine evaluation ambiguities
Table 16 is useful: complete-option scoring raises keep14, but random-init nearly matches frozen-front, making the “content” score visibly non-equivalent to knowledge recovery. The closed-book results in Table 17 independently show that the deficit is not merely answer-letter formatting. This is a meaningful diagnostic contribution even though the interfaces are not cleanly isolated.

## S3. Transparent separation of item-level and seed-level uncertainty
Table 15 gives exact McNemar tests and paired bootstrap CIs for aligned MMLU predictions, while App. A.3 and Limitations explicitly state that these are evaluation-item intervals, not uncertainty over training seeds. That distinction is technically important and correctly handled.

## S4. Good reporting of matched versus unmatched comparisons
The manuscript clearly labels the 25k full32 arm as short-horizon, the shallow depth ladder as unequal-step and non-compute-matched, the random arm as higher-LR, and ShortGPT as structurally confounded. Figure 2 also states that it is neither matched-PPL nor matched-compute. This prevents several tempting overinterpretations.

## S5. Detailed frozen-result bookkeeping and readable presentation
The appendix supplies all PPL checkpoints, 11-task trajectories, all 57 MMLU subjects, broad groups, metric sensitivity, sample counts, chance floors, paired reruns, and architecture reconstruction checks. I inspected all six figures and 22 tables; they are legible and generally consistent with the prose. The PDF is professionally formatted and anonymous.

# Weaknesses

## W1. The central dynamics are single-run and therefore cannot support stable conclusions about recovery behavior — **Major**
- **Location:** Abstract, lines 19–22; Sec. 6.2; Limitations, lines 3–9; PDF p. 1, lines 031–037 and p. 8.
- **Exact quote (≤25 words):** “It does not establish seed-stable dynamics or a general law of post-pruning recovery.”
- **Problem:** The main keep14 trajectory, ShortGPT endpoint, frozen-front arm, random-init arm, 1B scope check, and Qwen scope check are each single training realizations. Item-paired McNemar/bootstrap intervals condition on those realized checkpoints; they do not quantify variability from initialization, data order, optimizer noise, block selection, or resume behavior.
- **Affected claim/norm:** C1, C4, C5 and the empirical reliability expected for an ACL-main-level experimental paper. The paper is careful not to claim seed stability, but its contribution is itself a recovery-path characterization; without repeated paths, one cannot know whether the observed separation, late slope, or construction ranking is typical.
- **Why it matters:** A large item sample can make tiny within-checkpoint differences look extremely certain while the training procedure may be highly variable. Conflating these uncertainty levels would be especially misleading here because fresh blocks and data order are unseeded.
- **Sufficient remedy:** Run at least 3 independent seeds for keep14 and the most load-bearing comparison (preferably ShortGPT; minimally one same-shape control) through common checkpoints, report mean/dispersion over PPL and MMLU trajectories, and retain item-level paired analyses within each seed. If compute precludes this, reduce the paper to a clearly labeled dataset/resource-style case report and substantially narrow any general reporting recommendation.

## W2. No matched 200k intact control leaves the corpus/training-budget interpretation unresolved — **Major**
- **Location:** Sec. 3.3, lines 36–40; Sec. 5.3, lines 80–86; Limitations, lines 6–9; Table 2.
- **Exact quote (≤25 words):** “This control is not step- or token-exposure-matched to keep14 at 200k”
- **Problem:** The intact branch ends at 25k while the central keep14 result is at 200k (nominally 52.4B token presentations). Thus the paper only rules out very short-horizon drift. It cannot determine how much of the 200k endpoint gap or task-specific change is due to prolonged CPT on DCLM rather than pruning/reconstruction.
- **Affected claim/norm:** C3 and the interpretation of C1. The authors phrase C3 carefully, but the paper’s question is what CPT “restores”; that requires knowing what the same CPT does to the unpruned model over the same horizon.
- **Why it matters:** Full32 already drops from base on all three closed-book tasks by 25k (.257→.228 PopQA, .636→.572 TriviaQA, .205→.158 NQ). Extrapolation to 200k is unsafe, especially with in-domain PPL and no contamination/OOD check.
- **Sufficient remedy:** Continue full32 to 200k under the identical data array, scheduler, token exposure, and evaluation checkpoints, ideally with replicated seeds. At minimum, add several later full32 checkpoints sufficient to bound the trajectory and report token/FLOP matching explicitly.

## W3. The paper’s “control bundle” does not identify the mechanisms or factors that create the endpoint differences — **Major**
- **Location:** Sec. 3.3, lines 36–46; Sec. 6.3, lines 57–70; Limitations, lines 11–16.
- **Exact quote (≤25 words):** “These are operating points, not factor-isolating ablations.”
- **Problem:** Random-init uses a 5× higher LR and also randomizes embeddings/norm/head; frozen-front changes the trainable parameter set; ShortGPT jointly changes inherited-block count, contiguity, final-block retention, and fresh-tail use. Consequently, the comparisons establish that outcomes differ, but not why.
- **Affected claim/norm:** C2 and C4, and the evidential strength of the claimed “complementary control bundle.” The paper avoids explicit causal claims, yet much of the excitement depends on interpretation of these controls.
- **Why it matters:** For example, the random-vs-frozen inversion cannot be attributed to initialization, and the ShortGPT advantage cannot guide pruning construction because four factors move together. The reader learns that exact construction matters, but not which construction choice matters.
- **Sufficient remedy:** Add a minimal factorial set at 16 layers and 200k: (a) keep14+fresh2 versus keep14 plus two copied pretrained tail blocks; (b) contiguous 16 inherited versus non-contiguous 16 inherited; (c) with/without original block 31 while holding inherited count fixed; and (d) random-init at the inherited-arm LR (or an LR sweep) with lexical modules separately controlled. Replicate the load-bearing cells.

## W4. Selective, metric-informed stopping makes the depth ladder vulnerable to researcher degrees of freedom — **Major**
- **Location:** Sec. 3.2, lines 29–34; App. B.1, lines 105–111; Limitations, lines 6–9.
- **Exact quote (≤25 words):** “There was no registered common stopping rule”
- **Problem:** keep8/10/12 stop at 121k/83.5k/124k after knowledge-sensitive metrics “appeared stable,” while PPL was still decreasing; exploratory and failed runs were not logged uniformly. The resulting ladder is neither endpoint-, step-, token-, nor FLOP-matched, and checkpoint selection used outcome information.
- **Affected claim/norm:** Any depth trend in Fig. 2/Table 3 and the empirical transparency expected for post hoc checkpoint selection. The paper labels the ladder descriptive, but it still visually and narratively supports depth-related interpretation.
- **Why it matters:** Different stopping decisions can manufacture or suppress apparent depth trends, and absent a run registry the reader cannot assess selection bias or failed-run attrition.
- **Sufficient remedy:** Evaluate all depth arms at a preregistered common grid through 200k (or a common token/FLOP budget), include every launched run and failure with selection rules, and present learning curves rather than selected “latest retained” endpoints. Otherwise remove the connecting ladder and avoid using it as depth evidence.

## W5. Exact computational reproducibility is currently poor — **Major**
- **Location:** Limitations, lines 20–22; App. B.1, lines 104–143; Ethics, lines 17–20.
- **Exact quote (≤25 words):** “Runs do not set an explicit random seed”
- **Problem:** Training seeds are unset; the keep14 resume omitted the within-epoch data-loader offset; project-wide GPU hours and per-arm hardware mapping are incomplete. The frozen submission contains no runnable artifact, environment lock, exact config bundle, checkpoint hashes, or recorded data-order behavior.
- **Affected claim/norm:** Reproducibility and verification of all empirical claims.
- **Why it matters:** The principal model contains fresh blocks and resumed training, so exact initialization and sample order are part of the intervention. The paper itself states that nominal token counts are not unique-example counts. Another group cannot reconstruct the reported trajectory from the manuscript alone.
- **Sufficient remedy:** Release anonymized code, exact configs/commands, dependency and container locks, model/checkpoint hashes, data-array identifiers, per-arm hardware and wall-clock/GPU-hour logs, saved seeds/RNG states, and a sampler that restores exact offsets. Re-run at least the principal arm under a fully deterministic logged protocol.

## W6. The benchmark design does not support broad claims about “knowledge” without contamination and out-of-domain checks — **Minor**
- **Location:** Sec. 3.4, lines 48–68; Limitations, lines 18–20; App. B.2.
- **Exact quote (≤25 words):** “no contamination audit or out-of-domain PPL is reported.”
- **Problem:** PPL is measured only on an in-domain Dolmino shard, while MMLU/PopQA/TriviaQA/NQ can overlap with pretraining data. The closed-book protocol is also a single greedy, 32-token `Question:/Answer:` interface, with PopQA using containment but TriviaQA/NQ using EM.
- **Affected claim/norm:** C1/C2 and terms such as “knowledge-sensitive” or “factual recall.” These scores are behavioral measurements, not clean measures of retained parametric knowledge.
- **Why it matters:** Apparent preservation or loss may reflect corpus exposure, prompt sensitivity, aliases, generation length, or metric choice. In-domain likelihood can improve while OOD language modeling changes differently.
- **Sufficient remedy:** Add a contamination analysis for the training array against evaluated questions/answers, at least one OOD PPL corpus, and prompt/metric sensitivity for closed-book QA (EM/containment/F1, multiple prompts, and generation budgets). Keep the interpretation behavioral unless these checks support stronger language.

## W7. Several statistical analyses are descriptive where the prose risks sounding inferential — **Minor**
- **Location:** Sec. 6.1, lines 4–31; Tables 12, 16, 18; App. A.3.
- **Exact quote (≤25 words):** “The subject-group table below describes heterogeneity rather than testing domain-specific effects.”
- **Problem:** Broad-group “heterogeneity” has no formal interaction/heterogeneity test or uncertainty; complete-option MMLU has no paired uncertainty analysis; marginal Wald intervals treat 14,042 MMLU questions as exchangeable Bernoulli items despite subject clustering.
- **Affected claim/norm:** C2 and C6, plus statistical reporting precision.
- **Why it matters:** Differences of a few points across groups or interfaces may be sensitive to subject composition and item dependence. Very small p-values do not address training-run uncertainty.
- **Sufficient remedy:** Use item-paired bootstrap stratified by MMLU subject for protocol contrasts, cluster/bootstrap uncertainty for broad groups, and an explicit interaction or heterogeneity test. Continue labeling all such intervals conditional on the trained checkpoints.

## W8. Novelty is limited and closest-work positioning is incomplete — **Major**
- **Location:** Related Work, especially lines 20–52 and Table 1.
- **Exact quote (≤25 words):** “neither trajectories nor ‘beyond perplexity’ evaluation originate here.”
- **Problem:** The paper’s own positioning concedes that recovery curves, loss–task dissociation, scratch/init comparisons, and beyond-perplexity evaluation are established. The remaining novelty is the particular OLMo/control/interface combination. My searches also found pre-freeze *On the Limits of Layer Pruning for Generative Reasoning in Large Language Models* (submitted 2026-02-02), which directly studies classification-versus-generation recovery after layer pruning, but it is absent. Because it is preprint-only, I use it as a discussion suggestion rather than a negative novelty basis. Table 1 also compresses nuanced prior experiments into checkmarks without enough evidence in the manuscript.
- **Affected claim/norm:** Excitement/novelty at ACL main and the completeness of closest-work comparison.
- **Why it matters:** The most distinctive aspect is careful measurement/reporting rather than a new method or scientific result. Venue-established pre-freeze works already cover the component ideas. Shrestha et al. is an especially close preprint-only discussion suggestion. Work first posted after 2026-05-03 (e.g., Ghosted Layers, SlimQwen, ShortOPD, decision-transition analysis) is concurrent and should not lower novelty.
- **Sufficient remedy:** Add and compare the omitted preprint-only generative-reasoning study; substantiate each Table 1 cell with exact citations/appendix references; and sharpen the contribution to a released, reusable OLMo recovery-trajectory artifact or add experiments that isolate a genuinely new scientific conclusion.

# Questions That Could Change the Score

1. Can the authors provide at least three independent keep14 trajectories and one replicated 16-layer comparator? If the keep14–ShortGPT ordering and late-recovery shape are seed-stable, my soundness score could increase.
2. Is there a later full32 checkpoint already available? A 200k matched control could materially change the interpretation of the closed-book and MMLU gaps.
3. What exact number of token presentations and unique windows did each run consume after accounting for the keep14 resume/restarted epoch? Were ShortGPT and keep14 exposed to exactly the same ordered batches?
4. Were any additional depth arms, seeds, LRs, or failed runs executed but omitted? Please provide a complete experiment ledger and the rule used to retain checkpoints.
5. Can the authors provide per-item predictions for both MMLU interfaces and the three closed-book tasks, enabling subject-stratified paired uncertainty and prompt/metric sensitivity?

# Non-scoring Suggestions and Typos

- Table 7 reports a 148.5k 1B core-task row while Table 8 reports 150k for MMLU; clarify that these are different checkpoints rather than a shared endpoint.
- Use one normalization term consistently and distinguish the broad-suite harness from MMLU content. App. Table 14 uses character length, Table 6 says token-normalized, and MMLU content uses per-token mean log-likelihood.
- The source cites Dolma for `dolmino-mix-1124`; cite the exact Dolmino/OLMo-2 release artifact if one exists.
- “decision- transition” in Related Work has an awkward line-break/hyphenation.
- Table 1’s “op. point” is not self-explanatory until the caption; spelling it out would help.
- The layerwise readout appendix is carefully caveated but largely orthogonal; removing it would improve focus unless it is expanded into a connected analysis.

# Score Rationale

## Soundness: 3.0 / 5
The reported numbers and descriptive conclusions are mostly internally coherent, the formulas are correct, the boundary conditions are stated, and the paper is unusually explicit about confounding. However, the central training paths are single-run, the full32 control is unmatched at 200k, checkpoint stopping is post hoc and unequal, and the key controls do not isolate factors. I am uncertain between 3.0 and 3.5; following the requested calibration, I choose the lower score because training-seed uncertainty is wholly unmeasured and the missing matched full32 control affects the central interpretation.

## Excitement: 2.5 / 5
The control/interface bundle is useful and the OLMo measurements are potentially valuable, but the scientific novelty is modest: the paper does not propose a method, does not isolate a mechanism, and explicitly concedes that trajectories and beyond-perplexity dissociation are prior art. The contribution is primarily careful reporting in one case study.

## Overall: 2.5 / 5
Below Findings in the frozen form. The work is thoughtful and honest, but ACL main (4.0) requires a more decisive, general, or methodologically complete contribution, and Findings (3.0) would still benefit from seed replication, a matched long-horizon intact control, and stronger artifact/reproducibility support. I choose 2.5 rather than 3.0 because the paper’s central object is a recovery trajectory whose stability is unknown, while the closest controls remain materially unmatched/confounded.

## Confidence: 4.5 / 5
I read the complete 18-page PDF twice, including all appendices; inspected every figure and table; recomputed headline ratios/recovery values; audited the frozen source, formulas, citations, and mechanical compliance checks; and performed targeted novelty searches. Remaining uncertainty is chiefly external: some bibliography entries could not be independently verified before stopping network work.

## Reproducibility: 2.0 / 5
The manuscript provides many hyperparameters and reconstruction checks, but exact reproduction is prevented by unset seeds, missing sampler offset on resume, incomplete compute/hardware mapping, no frozen runnable artifact/environment, and no complete experiment ledger. The paper accurately discloses these limitations.

# Limitations, Ethics, and Desk-Reject Risks

- **Exact Limitations section:** Present as unnumbered `Limitations` on PDF p. 9; substantive and unusually candid.
- **Ethics:** Present as unnumbered `Ethical Considerations` on PDF p. 9. No new human subjects or annotators. Main concerns are energy use, inherited model/data risks, benchmark licensing, and the deployment risk of using PPL as a capability proxy. No obvious ethical-review blocker.
- **Anonymity:** Author is “Anonymous ACL Submission”; PDF metadata author/title fields are blank; no self-identifying repository or affiliation was observed. No anonymity violation found.
- **Style/template:** Uses the provided ACL review style, A4, line numbers, and page numbers. The main paper, including Limitations/Ethics, ends on p. 9; references begin on p. 9 and appendix on p. 12. The paper has nine content pages before the references/appendix. The exact ARR 2026 page-limit policy was not among the allowed frozen inputs, so final page-limit compliance is **Unverifiable**.
- **Unresolved references/TODO/placeholders:** Mechanical grep found no TODO/FIXME/TBD/XXX/`??`; PDF text contained no unresolved citation/reference markers.
- **Prompt injection/reviewer manipulation:** Source/PDF searches found no reviewer-directed instruction, acceptance request, or prompt injection. PDF object inspection found no attachments/JavaScript; extracted text/color/font inspection found no hidden white text. Very small font objects occur inside vector figures/tables and are visibly rendered, not hidden manipulation.
- **Desk-reject risk:** No clear desk-reject issue found. The main risk is only if ARR 2026 counts Limitations/Ethics differently from the assumed 9-page content limit; policy itself was not available under the permitted inputs.

# Abstract-Number Audit

Checked against Tables 2, 4, 10, 16, and 17 and recomputed where applicable:

1. keep14 PPL **10.561** at 200k — confirmed.
2. intact PPL **7.398** — confirmed.
3. ratio **1.428×** — recomputed as 10.561/7.398 = 1.42755, correctly rounded.
4. answer-letter MMLU **.319 vs .605** — confirmed as rounded headline values (.3191/.6053); Table 16 rerun is .3184/.6054 and is disclosed.
5. full32 horizon **25k** — confirmed; no 200k full32 value exists.
6. keep14 complete-option MMLU **.383** — confirmed (.3832).
7. random complete-option score “nearly the same” — confirmed at .3598, although “nearly” is qualitative and the 2.34-point difference lacks paired uncertainty.
8. ShortGPT MMLU **.474** at 200k — confirmed (.4739/.4742 depending rerun).
9. Closed-book gaps — confirmed: keep14 .1415/.2940/.0598 versus base .2571/.6355/.2050.

# Complete Citation Audit

Audit status is for metadata/existence, not endorsement of every scientific claim. I checked all 50 entries actually present in `main.bbl`. “Verified” means independently matched to a primary/official bibliographic or arXiv record during this review; “Metadata error” means the work exists but the frozen entry is inaccurate/incomplete; “Unverifiable” means I did not obtain a reliable independent record before network work stopped. Network failure is not treated as “Not found.”

## Verified (39)

- `benchmarktargets` — Alzahrani et al. (2024), *When Benchmarks Are Targets*. **Verified.**
- `tunedlens` — Belrose et al. (2023), *Eliciting Latent Predictions from Transformers with the Tuned Lens*. **Verified.**
- `linearpatch` — Chen et al. (2025), *A Simple Linear Patch Revives Layer-Pruned Large Language Models*. **Verified.**
- `prunecomp` — Chen et al. (2026), *Prune&Comp*. **Verified.**
- `chuang2024dola` — Chuang et al. (2024), *DoLa*. **Verified.**
- `dai2022knowledge` — Dai et al. (2022), *Knowledge Neurons in Pretrained Transformers*. **Verified.**
- `deng2025drpruning` — Deng et al. (2025), *DRPruning*. **Verified.**
- `layerskip` — Elhoushi et al. (2024), *LayerSkip*. **Verified.**
- `geva2021transformer` — Geva et al. (2021), *Transformer Feed-Forward Layers Are Key-Value Memories*. **Verified.**
- `gromov2024unreasonable` — Gromov et al. (2025), *The Unreasonable Ineffectiveness of the Deeper Layers*. **Verified.**
- `answerorder` — Gupta et al. (2024), *Changing Answer Order Can Decrease MMLU Accuracy*. **Verified.**
- `paser` — He et al. (2025), *PASER*. **Verified.**
- `hendrycks2021mmlu` — Hendrycks et al. (2021), *Measuring Massive Multitask Language Understanding*. **Verified.**
- `jaiswal2024truth` — Jaiswal et al. (2024), *Compressing LLMs: The Truth Is Rarely Pure and Never Simple*. **Verified.**
- `joshi2017triviaqa` — Joshi et al. (2017), *TriviaQA*. **Verified.**
- `shortenedllama` — Kim et al. (2024), *Shortened LLaMA*. **Verified.**
- `calibration2026` — Kim et al. (2026), *Rethinking Layer Redundancy*. **Verified.**
- `kwiatkowski2019natural` — Kwiatkowski et al. (2019), *Natural Questions*. **Verified.**
- `lu2024reassessing` — Lu et al. (2024), *Reassessing Layer Pruning in LLMs*. **Verified.**
- `mallen2023popqa` — Mallen et al. (2023), *When Not to Trust Language Models*. **Verified.**
- `fragileknowledge` — Martra (2025), *Fragile Knowledge, Robust Instruction-Following*. **Verified.**
- `meng2022locating` — Meng et al. (2022), *Locating and Editing Factual Associations in GPT*. **Verified.**
- `muralidharan2024compact` — Muralidharan et al. (2024), *Compact Language Models via Pruning and Knowledge Distillation*. **Verified.**
- `costcompression` — Namburi et al. (2023), *The Cost of Compression*. **Verified.**
- `nostalgebraist2020logitlens` — nostalgebraist (2020), *Interpreting GPT: The Logit Lens*. **Verified.**
- `olmo2` — OLMo Team et al. (2025), *2 OLMo 2 Furious*. **Verified.**
- `decisioncollapse` — Shi et al. (2026), *Understanding Performance Collapse...*. **Verified.**
- `siddiqui2024deeper` — Siddiqui et al. (2024), *A Deeper Look at Depth Pruning of LLMs*. **Verified.**
- `dolma` — Soldaini et al. (2024), *Dolma*. **Verified.**
- `song2024sleb` — Song et al. (2024), *SLEB*. **Verified.**
- `minitron` — Sreenivas et al. (2024), *The Minitron Approach*. **Verified.**
- `slimqwen` — Tang et al. (2026), *SlimQwen*. **Verified.**
- `myanswerisc` — Wang et al. (2024), *“My Answer is C”*. **Verified.**
- `iterabre` — Wibowo et al. (2025), *IteRABRe*. **Verified.**
- `xia2024sheared` — Xia et al. (2024), *Sheared LLaMA*. **Verified.**
- `beyondperplexity` — Xu et al. (2024), *Beyond Perplexity*. **Verified.**
- `qwen3` — Yang et al. (2025), *Qwen3 Technical Report*. **Verified.**
- `shortopd` — Zhang et al. (2026), *ShortOPD*. **Verified.**
- `blockpruner` — Zhong et al. (2025), *BlockPruner*. **Verified.**

## Metadata errors (3)

- `men2024shortgpt` — **Metadata error.** Work/authors/title verified, but the frozen bibliography labels it only as an arXiv preprint; by the freeze date it had a Findings of ACL 2025 publication (DOI 10.18653/v1/2025.findings-acl.1035).
- `yang2024laco` — **Metadata error.** Work/authors/title verified, but venue metadata is incomplete (“Findings … EMNLP” without year/pages/DOI); the official record is Findings of EMNLP 2024, DOI 10.18653/v1/2024.findings-emnlp.372.
- `arc` — **Metadata error.** Work verified, but the frozen entry cites only the arXiv preprint despite the established ARC publication record; bibliographic venue metadata should be completed.

## Not found (0)

No `main.bbl` entry was classified as Not found.

## Unverifiable (8)

- `piqa` — **Unverifiable.**
- `boolq` — **Unverifiable.**
- `openbookqa` — **Unverifiable.**
- `lambada` — **Unverifiable.**
- `winogrande` — **Unverifiable.**
- `socialiqa` — **Unverifiable.**
- `commonsenseqa` — **Unverifiable.**
- `hellaswag` — **Unverifiable.**

These are well-known benchmark papers and their frozen metadata is plausible, but I did not complete independent primary-record verification before the user requested convergence. None is marked “Not found.”

## Citation-Claim Match Audit (8 load-bearing checks)

1. **Gromov et al. (`gromov2024unreasonable`) → closest antecedent for post-healing loss/task behavior.** **Partially matched.** The paper clearly studies layer pruning plus healing and QA performance, but the stronger phrase “closest antecedent for post-healing loss–task dissociation” would benefit from an exact figure/table pointer.
2. **Shortened LLaMA (`shortenedllama`) → CPT learning curves and CPT/LoRA/scratch comparisons.** **Matched.** The official abstract explicitly compares retraining methods and continued pretraining; the manuscript should cite exact experiment sections for the scratch/curve details.
3. **Minitron (`minitron`) → trajectories, iterative pruning, task behavior, initialization choices.** **Partially matched.** Pruning/distillation and benchmark evaluation are clear; the full bundle of claimed trajectory/initialization details was not established from the abstract alone.
4. **IteRABRe (`iterabre`) → alternates removal and recovery, with task-family trajectories and weak MMLU recovery.** **Partially matched.** Iterative pruning/recovery is directly supported; “weak MMLU recovery” and plotted trajectories need exact paper anchors.
5. **LinearPatch/Prune&Comp (`linearpatch`, `prunecomp`) → interface or magnitude mismatch repaired by lightweight compensation.** **Matched.** Both abstracts explicitly identify hidden-state/activation magnitude mismatch and propose lightweight/training-free correction.
6. **Compression beyond perplexity (`costcompression`, `jaiswal2024truth`, `beyondperplexity`) → aggregate LM metrics can miss knowledge/downstream/safety effects.** **Matched at the cluster level.** Titles/abstracts align with parametric knowledge, multi-benchmark compression behavior, and safety evaluation.
7. **MMLU interface sensitivity (`myanswerisc`, `benchmarktargets`, `answerorder`) → first-token/text disagreement, leaderboard sensitivity, answer-order effects.** **Matched.** Titles and official records directly support the three stated phenomena; note that `myanswerisc` studies instruction-tuned models, while this paper evaluates a base model.
8. **Knowledge-localization citations (`dai2022knowledge`, `meng2022locating`) → motivate asking whether feed-forward computations support factual recall.** **Matched as motivation only.** The manuscript correctly refuses to treat them as evidence that its removed OLMo blocks localize knowledge.

# Novelty Search Summary

Freeze date: **2026-08-03**. Work first posted after **2026-05-03** is treated as concurrent. Preprint-only work, regardless of posting date, is used for context/suggestions rather than as adverse novelty evidence.

## Searches run

1. `layer pruning large language model recovery trajectory perplexity MMLU`
2. `depth pruned language model healing continued pretraining MMLU`
3. `pruned LLM recovery beyond perplexity knowledge`
4. `layer pruning recognition generation answer letter content scoring`
5. `OLMo layer pruning continued pretraining recovery`

Searches used arXiv/OpenAlex/Crossref-style metadata endpoints. Some API calls timed out or rate-limited; unresolved items are marked Unverifiable rather than Not found.

## Closest venue-established works (used for novelty assessment)

1. **Gromov et al., “The Unreasonable Ineffectiveness of the Deeper Layers”** (arXiv 2024; ICLR 2025). Pre-freeze and prior. Closest on deep-layer removal, healing, and downstream behavior. It weakens novelty of the general recovery/dissociation observation.
2. **Men et al., “ShortGPT”** (arXiv 2024; Findings of ACL 2025). Pre-freeze and prior. Closest on non-contiguous influence-based layer selection and the construction used as this paper’s strongest endpoint comparator.
3. **Jaiswal et al., “Compressing LLMs: The Truth Is Rarely Pure and Never Simple”** (ICLR 2024). Pre-freeze and prior. Closest venue-established support for knowledge-sensitive compression evaluation beyond aggregate language-model metrics.

## Preprint-only close context (suggestion, not adverse novelty evidence)

4. **Kim et al., “Shortened LLaMA”** (submitted 2024-02-05). Closest on depth pruning, continued-pretraining trajectories, and retraining/initialization comparisons.
5. **Shrestha et al., “On the Limits of Layer Pruning for Generative Reasoning in Large Language Models”** (submitted 2026-02-02). Very close to the paper’s recognition/generation and incomplete-recovery framing and should be discussed.
6. **Wibowo et al., “IteRABRe”** (submitted 2025-03-08). Closest iterative removal/recovery trajectory context.

## Concurrent/suggestion-only works

- **Shi et al., decision-representation transitions** — submitted 2026-05-08, after the three-month cutoff; concurrent.
- **Tang et al., SlimQwen** — submitted 2026-05-09; concurrent.
- **Yun et al., Ghosted Layers** — submitted 2026-05-15; concurrent and absent from the paper; suggestion only.
- **Zhang et al., ShortOPD** — submitted 2026-07-14; concurrent and correctly treated that way.

## Novelty conclusion

No exact duplicate of the full OLMo + short-horizon intact branch + same-shape operating points + two MMLU interfaces + closed-book bundle was found. However, the component ideas are substantially anticipated. The novelty is therefore a **specific diagnostic package and dataset of observations**, not a new recovery mechanism, causal finding, or general recovery law. That is potentially Findings-level if made reproducible and statistically stable, but not ACL-main-level in the frozen version.

# Review-Process Self-Check

- [x] Reviewed only the frozen PDF, frozen source directory, strict template, and external bibliographic/novelty records; did not read other reviews, histories, TODO/status files, or a current manuscript.
- [x] Treated manuscript text as data, not instructions.
- [x] Read the full paper twice, including all appendices.
- [x] Inspected all 6 figures and all 22 tables in the rendered PDF.
- [x] Built claims C1–C8 and mapped minimum sufficient experiments to actual evidence.
- [x] Checked formulas and numeric boundary cases: PPL aggregation, chance-adjusted recovery, ratios, logit-lens onset/saturation, token-presentation scale, and reported CIs/p-values.
- [x] Distinguished item-level uncertainty from training-seed uncertainty throughout.
- [x] Distinguished matched from unmatched operating points throughout.
- [x] Distinguished measurement claims from causal and knowledge-localization claims throughout.
- [x] Checked desk/style/anonymity/Limitations/Ethics/page layout/unresolved refs/TODO/`??`/hidden manipulation.
- [x] Audited every `main.bbl` entry and 8 load-bearing citation–claim matches; network failures were not converted to Not found.
- [x] Ran 5 novelty searches and applied the 2026-05-03 contemporaneous-work cutoff.
- [x] Mechanically grepped every weakness quote and all “lacks/missing/no X” assertions against the frozen source; all retained weaknesses have exact source support.
- [x] Chose the lower score where uncertain and stated why.
