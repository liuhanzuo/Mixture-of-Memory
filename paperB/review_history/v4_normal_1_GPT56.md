```yaml
review_mode: normal
soundness: 3.5
excitement: 3.0
overall: 3.0
confidence: 4.0
reproducibility: 2.5
```

# Summary

This paper presents an observational measurement study of post-pruning recovery in OLMo-2. Its main intervention keeps the first 14 of 32 pretrained OLMo-2-7B blocks, appends two fresh blocks, and continues pretraining for 200k optimizer steps. Rather than proposing a new pruning or recovery algorithm, the paper asks whether improvements in held-out likelihood track restoration of target behaviors, and it advocates reporting recovery along five axes: likelihood, target capability, evaluation interface, exact construction, and training budget.

The central empirical picture is clear. The keep14+fresh2 run reaches held-out PPL 10.561 versus 7.398 for the intact base (a 1.428x tax), while answer-letter MMLU is .319 versus .605. Complete-option MMLU raises keep14 to .383, but a random-init 16-layer operating point obtains .360 under that interface while remaining near chance on answer letters (.247), indicating a substantial non-knowledge/fluency floor. Closed-book results tell a similar story: keep14 scores .142/.294/.060 on PopQA/TriviaQA/NQ-open versus .257/.636/.205 for the intact model. A non-contiguous ShortGPT-16 construction reaches PPL 9.780 and MMLU .474 at 200k, but differs in inherited-block count, layer selection, final-block retention, and use of fresh layers. The 25k full32 continued-pretraining branch remains relatively close to the base but is not a long-horizon matched control.

I view this as a careful and useful, but deliberately narrow, measurement paper. Its strongest feature is unusually explicit separation of observation from causation: the manuscript repeatedly labels single runs, unmatched budgets, and coupled controls. Its principal limitation is that nearly all claims of practical interest remain conditional on one training realization and one main model/recipe, with no 200k intact control and no factor-isolating construction ablations. This supports a Findings-level recommendation rather than ACL-main level.

# Claims and evidence map

- **C1 — Lower post-pruning PPL does not certify recovery of target behavior in the observed OLMo run.** Evidence: Table 2 and Figure 1; keep14 improves to PPL 10.561 but remains at .319 MMLU and far below base on three closed-book tasks. Table 13 shows continued late PPL and MMLU improvement without broad late catch-up. This is well supported as a descriptive, within-run claim, not as a general law.
- **C2 — MMLU recovery is materially interface-sensitive.** Evidence: Table 16; keep14 is .3184 on answer letters, .3548 on raw complete-option scoring, and .3832 on token-normalized complete-option scoring. This is directly supported, although the protocols change several factors jointly.
- **C3 — Complete-option MMLU has a substantial fluency/plausibility floor and cannot be read as a clean knowledge measure here.** Evidence: random init obtains .3598 normalized content MMLU versus .3832 for keep14 while its letter score is .2470. This is persuasive as an operating-point diagnostic, but “fluency floor” is an interpretation rather than an isolated mechanism.
- **C4 — Short-horizon corpus shift alone is insufficient to explain the half-depth deficit.** Evidence: full32@25k has PPL 7.670, MMLU .588, and closed-book .228/.572/.158, substantially nearer the intact base than keep14@200k. This is appropriately limited to the observed 25k horizon; it does not establish a 200k counterfactual.
- **C5 — Nominal depth does not determine a unique recovery endpoint.** Evidence: at 16 layers and 200k steps, ShortGPT-16 reaches PPL 9.780/MMLU .474, versus 10.561/.319 for keep14. Supported for these two constructions, but no component-level cause is identified.
- **C6 — Late healing continues, but slowly and unevenly.** Evidence: keep14 moves from PPL 10.826/MMLU .3012 at 128k to 10.561/.3191 at 200k; the paired MMLU rerun gives +1.68 points, 95% CI [1.08, 2.29], exact McNemar p=4.12e-8. Other tasks change by mostly less than one point from 153.5k to 200k. The item-level uncertainty is valid conditionally on checkpoints, not across training seeds.
- **C7 — Recovery differs across broad MMLU domains and tasks.** Evidence: Figure 3/Table 18 report keep14 chance-adjusted recovery of 15.6% STEM, 16.2% Humanities, 18.6% Social Science, and 29.1% Other; Table 10/11 show much higher recovery on some continuation-style tasks than on answer-letter MMLU. This is descriptive only; the paper correctly avoids anatomical or causal claims.
- **C8 — A five-axis reporting bundle is useful for post-pruning recovery studies.** Evidence: the paper’s own controls repeatedly change the interpretation. This is a reasonable proposal motivated by the case study, but not externally validated as a standard.

# Strengths

1. **Strong claim discipline and honest scope.** The abstract, main text, captions, Limitations, and Discussion consistently distinguish single-run evidence, unmatched budgets, coupled constructions, and causal non-identifiability. This is especially important for a measurement paper and substantially increases trust.
2. **A genuinely informative control bundle.** The intact base, 25k full32 branch, frozen-front arm, fully random same-shape operating point, ShortGPT construction, two MMLU interfaces, and closed-book QA each rule out or weaken a different simplistic interpretation. No individual control is clean, but together they make the descriptive result considerably more useful than a PPL/MMLU curve alone.
3. **Broad, transparent evaluation.** The paper reports PPL, all 14,042 MMLU items, three closed-book tasks, ten additional likelihood tasks, MMLU groups and all 57 subjects, metric sensitivity, sample counts, chance floors, prompts/scoring descriptions, and checkpoint provenance. At least six useful headline quantities are easy to recover: 10.561 PPL, 1.428x PPL tax, .319 letter MMLU, .383 content MMLU, .474 ShortGPT MMLU, and +1.68 paired late-healing points.
4. **Appropriate statistical treatment where item-level alignment exists.** Table 15 uses exact McNemar tests and 10,000 paired bootstrap resamples; Table 13 reports a paired temporal comparison. The authors explicitly state that these are evaluation-item intervals, not training-run uncertainty.
5. **Good negative/measurement-paper contribution.** The paper does not need a new algorithm to be useful. The most valuable result is not “PPL and capability differ,” which is known, but the concrete demonstration that interpretation changes with scoring interface, intact-CPT horizon, initialization/adaptation operating point, and exact 16-layer construction in one open model family.
6. **Readable presentation and complete appendices.** I inspected all six figures and Tables 1–22. Figures 1–5 accurately reflect their tabulated values; Figure 2 is explicitly labeled as neither matched-PPL nor matched-compute; Figure 6 and Tables 20–21 are clearly quarantined as non-causal background. Tables are dense but legible, and the main text points to the necessary caveats.

# Weaknesses

## Major W1 — Training-run uncertainty is unmeasured

- **Issue:** Every central training comparison is a single realization, and training seeds were not set. Item-level tests cannot establish seed-stable recovery dynamics or stable rankings between constructions.
- **Why it matters:** C1, C5, C6, and the practical reporting proposal are potentially sensitive to fresh-layer initialization, data order, optimization noise, and ShortGPT selection/calibration. This is the main reason I cannot treat the study as ACL-main-level evidence.
- **Location / exact quote:** Limitations, p.8, lines 582–586; Appendix A discussion, p.14. “All pairwise intervals exclude zero, but these are evaluation-item intervals rather than training-seed uncertainty” (14 words).
- **Required remedy:** Replicate at minimum keep14 and ShortGPT-16 with 3 seeds, report mean/dispersion for PPL, letter/content MMLU, and closed-book QA, and use seed as the experimental unit. **Major.**

## Major W2 — The strongest controls do not isolate the causal alternatives

- **Issue:** Random init uses a higher learning rate and randomizes lexical modules; frozen-front changes trainable parameters; ShortGPT changes four structural factors together.
- **Why it matters:** The results establish distinct operating points, but cannot attribute the keep14 gap to initialization, adaptation, final-layer retention, inherited-block count, non-contiguity, or fresh-tail damage. Readers interested in pruning design receive a diagnosis without a minimal sufficient causal ablation.
- **Location / exact quote:** Experiments §5.3, p.5, lines 376–383; Analysis §6.3, pp.6–7, lines 448–464. “learning rate and trainable-module differences prevent an initialization-only or adaptation-only causal claim.” (12 words).
- **Required remedy:** Add matched-LR/trainable-set initialization controls and a small factorial construction study: keep14+fresh2, keep16 prefix, ShortGPT-selected14+fresh2, and selected16 with/without original final block. **Major.**

## Major W3 — Recovery horizon and compute matching remain incomplete

- **Issue:** The only intact-CPT control stops at 25k, the shallow depth ladder stops at 83.5k–124k based on observed metrics, and FLOPs/tokens are not matched or fully accounted.
- **Why it matters:** Corpus drift is bounded only at short horizon, and cross-depth endpoint comparisons confound depth with exposure and stopping. The late keep14 curve is still improving, so eventual gap size is unresolved. Equal optimizer steps also do not imply equal compute across 16- and 32-layer models.
- **Location / exact quote:** Method §3.2–3.3, pp.3–4, lines 242–267; Limitations, p.8, lines 587–590. “there was no registered common stopping rule.” (7 words).
- **Required remedy:** Extend full32 to 200k or match unique tokens; evaluate all depth arms on a common token/FLOP grid; report realized token counts after resume and per-arm training FLOPs/GPU-hours. **Major.**

## Major W4 — External validity and novelty are narrower than the paper’s length suggests

- **Issue:** The principal evidence is one OLMo-2-7B prefix+fresh-tail recipe on one in-domain corpus. The 1B arm is same-family and more compressed; the Qwen arm changes model, corpus, and compression ratio and lacks the key interface/closed-book controls.
- **Why it matters:** Prior work already establishes recovery curves and loss–task dissociation. The incremental value is the control combination, but the proposed five-axis reporting practice is not tested across methods or families.
- **Location / exact quote:** Discussion, p.8, lines 557–562; Limitations, p.9, lines 602–610. “This proposal does not replace efficiency or endpoint-quality comparisons and is not validated across methods or model families.” (18 words).
- **Required remedy:** Apply the full diagnostic bundle to one additional architecture or one additional pruning/recovery method at a roughly matched retained-depth fraction. **Major.**

## Minor W5 — Reproducibility is materially incomplete

- **Issue:** The paper gives many hyperparameters and strict reconstruction checks, but lacks explicit training seeds, the resumed data-loader offset, an exact aggregate compute account, a frozen runnable artifact, evaluator/harness version or commit, environment lock, and checkpoint hashes.
- **Why it matters:** Exact reproduction of fresh initialization, data order, and some benchmark details is impossible from the manuscript alone; the unusual discrepancies between rounded headline and common-rerun MMLU values also make provenance especially important.
- **Location / exact quote:** Limitations, p.9, lines 606–610; Appendix B.1, p.14. “Exact reproduction is limited by unset training seeds, an unrecorded resumed data-loader offset, incomplete compute accounting, and no frozen runnable artifact.” (21 words).
- **Required remedy:** Release code/configs, environment lock, data-array and checkpoint hashes, evaluator version/commit, per-item predictions, seeds, and resume/data-order metadata. **Minor.**

## Minor W6 — The interface diagnostic is useful but bundles multiple changes and lacks paired uncertainty

- **Issue:** Letter versus content MMLU jointly changes prompt, candidate representation, tokenization, and normalization; no item-paired confidence interval is reported for the interface difference.
- **Why it matters:** The observed gap cannot be assigned specifically to answer-symbol readout, and the “fluency floor” could combine option-length and prompt effects.
- **Location / exact quote:** Method §3.4, p.4, lines 274–283; Table 16 caption, p.17. “Letter and content scoring change the prompt, candidate string, tokenization, and normalization together.” (13 words).
- **Required remedy:** Add paired item-level comparisons and controlled variants that change one factor at a time (same prompt with letter/text targets; summed versus token-normalized text). **Minor.**

## Minor W7 — Efficiency evidence is absent

- **Issue:** No latency, throughput, memory, inference FLOPs, recovery FLOPs, or quality–compute frontier is reported.
- **Why it matters:** The motivation is model compression, but the reader cannot quantify the practical benefit or recovery cost of the 16-layer endpoints, especially ShortGPT versus keep14.
- **Location / exact quote:** Limitations, p.9, lines 602–606. “We also do not report latency, throughput, memory, or recovery FLOPs.” (11 words).
- **Required remedy:** Report parameter count for every arm, inference FLOPs/latency/memory, training FLOPs/tokens, and a capability-versus-total-compute plot. **Minor.**

# Questions for the authors

1. Were keep14 and ShortGPT trained over exactly the same ordered corpus array after accounting for the keep14 resume, or only the same nominal schedule/data source? How many unique and total tokens did each actually process?
2. Why was the full32 branch stopped at 25k? Is a longer checkpoint unavailable because of cost, failure, or early stopping, and can any intermediate full32 trend justify extrapolation?
3. For the fully random arm, what was the rationale for the 5x higher peak LR? Did a matched-LR random run fail, and if so can that result be reported?
4. Can the authors report item-paired differences between letter and content protocols, including how often each protocol corrects or breaks an item, rather than only marginal accuracies?
5. How sensitive is ShortGPT’s selected layer set [0–12,16,17,31] to the 128-window calibration sample? Was selection itself seeded or repeated?
6. Are PopQA/TriviaQA/NQ-open closed-book results available for ShortGPT? These would test whether its MMLU advantage extends to generation-based factual recall.
7. Please reconcile token versus character normalization language: Appendix B.2 says token mean for MMLU content, while Table 14 describes character-length normalization for other harness tasks. A compact per-task evaluator specification would help.

# Suggestions

- Make the paper’s evidence hierarchy explicit in one compact table: cleanly matched, partially matched, and descriptive-only comparisons.
- Promote the strongest statistical caveat—items are not seeds—to the first mention of p-values, not only captions/appendix prose.
- Add a minimal matched-factor ablation before broadening the benchmark suite further; this would improve scientific value more than additional descriptive tasks.
- Report both observed-step and observed-token/FLOP axes, especially because the resumed iterator restarted an epoch shuffle.
- Include ShortGPT closed-book QA and the same MMLU content variants; currently the best construction is missing one major evaluation family.
- Shorten or remove the supplementary layer-wise readout section unless it is tied to a preregistered hypothesis; it is carefully caveated but not evidence for the paper’s claims.
- Clarify Table 1’s coding criteria (“partial,” trajectory checkmark, construction) and provide citations in the caption or cells, since the novelty argument depends on this matrix.

# Score rationale

- **Soundness: 3.5/5.** The descriptive measurements, formulas, tables, and item-level statistics are internally coherent, and causal scope is unusually well bounded. Soundness is reduced by single training runs, unset seeds, unmatched/coupled controls, and incomplete long-horizon/compute counterfactuals.
- **Excitement: 3.0/5.** The combination of controls and interfaces is useful and likely to influence better reporting, but the high-level PPL–capability dissociation and recovery trajectories are already known, and no new algorithm or causal mechanism is established.
- **Overall: 3.0/5 (Findings).** I would support publication as a careful Findings-level measurement case study. For ACL main (4.0), I would want seed replication plus either a matched factor-isolation experiment or a second full diagnostic study across a model/method.
- **Confidence: 4.0/5.** I read the PDF twice, including appendices, inspected every figure and table, checked source claims/labels/numbers, and audited the bibliography mechanically. Some external citation and novelty checks were Unverifiable because network verification was incomplete by instruction.
- **Reproducibility: 2.5/5.** Hyperparameters, architecture reconstruction, sample counts, prompts at a high level, and consistency checks are unusually detailed. However, unset seeds, missing resume offset, incomplete compute accounting, absent frozen artifact/environment/version identifiers, and missing checkpoint hashes prevent exact reruns.

# Desk, style, anonymity, and ethics audit

- **Page/format:** PDF has 18 A4 pages. Main text through Conclusion occupies pp.1–8; Limitations begins on p.8, Ethics and References on p.9, appendices on pp.12–18. The apparent main-content length is within the usual 8-page ACL/ARR body allowance, with limitations/ethics/references/appendix outside it. No page-limit desk issue found.
- **Official style:** Uses the supplied ACL style with `\usepackage[review]{acl}`, 11pt article, line numbers, A4 layout, and anonymous header. No obvious formatting manipulation found. Tables use `\scriptsize`/resize for density but remained legible in the rendered PDF.
- **Anonymity:** Author is “Anonymous ACL Submission”; PDF metadata has no author. No institution, repository, self-identifying URL, acknowledgment, or deanonymizing self-citation found.
- **Limitations:** Present and substantive; explicitly covers single runs, unmatched checkpoints, coupled controls, in-domain PPL, contamination, compute, seeds, resume behavior, artifact absence, and mechanism scope.
- **Ethics:** Present. No new human-subject data or annotation. The paper identifies deployment risk from relying on aggregate PPL, licensing obligations, and energy use. No obvious unaddressed high-risk experimentation issue.
- **Style/readability:** Generally professional and precise. It is somewhat repetitive—the same caveats recur in abstract, captions, Discussion, Limitations, and appendix—but this repetition reduces overclaiming. Appendix tables are very dense, especially Tables 21–22.
- **Unresolved markers:** Mechanical grep found no `??`, TODO, FIXME, TBD, XXX, placeholder, or unresolved references/citations in rendered text. All 27 cross-reference targets used in source have labels. One source comment mentions `PAPER_B_DATA.md`, but it is not rendered and is not a reviewer instruction.
- **Prompt injection/manipulation:** I inspected source text/comments, rendered PDF text, PDF strings, font usage, and suspicious TeX constructs. No hidden white text, reviewer-directed instruction, score request, prompt injection, or concealed manipulation was found. The paper was treated solely as data.
- **Numerical consistency:** Spot checks pass: 10.561/7.398=1.42755 (reported 1.428x); chance-adjusted MMLU recovery is 19.4% for keep14 and 63.0% for ShortGPT; table/figure values align. The paper transparently notes small differences between separate aggregates and common-interface reruns.

# Citation audit

## `main.bbl` entry-by-entry verification

I audited all 50 entries for internal completeness (authors, year, title, venue/source), source use, and identifier availability. “Verified” below means that the identifier resolved during the completed mechanical check; “plausible/Unverifiable” means the entry is internally plausible but external resolution was not completed before the instructed stop.

| # | Key / work | Audit result |
|---:|---|---|
| 1 | `benchmarktargets` | DOI resolved; metadata/use plausible. |
| 2 | `tunedlens` | arXiv identifier present; external resolution timed out — **Unverifiable**. |
| 3 | `piqa` | Canonical benchmark citation appears plausible; no identifier in `main.bbl` — **Unverifiable**. |
| 4 | `linearpatch` | arXiv identifier present; resolution timed out — **Unverifiable**. |
| 5 | `prunecomp` | arXiv identifier present; resolution timed out — **Unverifiable**. |
| 6 | `chuang2024dola` | Plausible ICLR citation; no identifier in `main.bbl` — **Unverifiable**. |
| 7 | `boolq` | Plausible NAACL citation; no identifier — **Unverifiable**. |
| 8 | `arc` | arXiv identifier present; resolution timed out — **Unverifiable**. |
| 9 | `dai2022knowledge` | Plausible ACL citation; no identifier — **Unverifiable**. |
| 10 | `deng2025drpruning` | Plausible ACL citation; no identifier — **Unverifiable**. |
| 11 | `layerskip` | DOI resolved; metadata/use plausible. |
| 12 | `geva2021transformer` | Plausible EMNLP citation; no identifier — **Unverifiable**. |
| 13 | `gromov2024unreasonable` | arXiv identifier resolved; title/year/use plausible. |
| 14 | `answerorder` | arXiv identifier present; resolution timed out — **Unverifiable**. |
| 15 | `paser` | arXiv identifier resolved; title/year/use plausible. |
| 16 | `hendrycks2021mmlu` | Canonical MMLU citation appears correct; no identifier — **Unverifiable**. |
| 17 | `jaiswal2024truth` | Plausible ICLR citation; no identifier — **Unverifiable**. |
| 18 | `joshi2017triviaqa` | Canonical TriviaQA citation appears correct; no identifier — **Unverifiable**. |
| 19 | `shortenedllama` | arXiv identifier resolved; title/year/use plausible. |
| 20 | `calibration2026` | arXiv identifier resolved; title/date plausible. |
| 21 | `kwiatkowski2019natural` | Canonical TACL citation appears correct; no identifier — **Unverifiable**. |
| 22 | `lu2024reassessing` | arXiv identifier resolved; title/year/use plausible. |
| 23 | `mallen2023popqa` | Canonical PopQA citation appears plausible; no identifier — **Unverifiable**. |
| 24 | `fragileknowledge` | arXiv identifier resolved; title/date plausible. |
| 25 | `men2024shortgpt` | arXiv identifier resolved; title/year/use plausible. |
| 26 | `meng2022locating` | Plausible NeurIPS citation; no identifier — **Unverifiable**. |
| 27 | `openbookqa` | Canonical EMNLP citation appears correct; no identifier — **Unverifiable**. |
| 28 | `muralidharan2024compact` | Plausible NeurIPS citation; no identifier — **Unverifiable**. |
| 29 | `costcompression` | DOI resolved; metadata/use plausible. |
| 30 | `nostalgebraist2020logitlens` | URL returned rate limit (429) — **Unverifiable**. |
| 31 | `olmo2` | arXiv identifier resolved; title/year/use plausible. |
| 32 | `lambada` | Canonical ACL citation appears correct; no identifier — **Unverifiable**. |
| 33 | `winogrande` | Plausible CACM citation; no identifier — **Unverifiable**. |
| 34 | `socialiqa` | Canonical EMNLP citation appears correct; no identifier — **Unverifiable**. |
| 35 | `decisioncollapse` | arXiv identifier resolved; title/date plausible. |
| 36 | `siddiqui2024deeper` | arXiv identifier resolved; title/year/use plausible. |
| 37 | `dolma` | arXiv identifier resolved. Citation is to Dolma although the experiment uses the Dolmino mix; the text explains the relationship, but a direct Dolmino/OLMo-2 data citation would be cleaner. |
| 38 | `song2024sleb` | Plausible ICML citation; no identifier — **Unverifiable**. |
| 39 | `minitron` | arXiv identifier resolved; title/year/use plausible. |
| 40 | `commonsenseqa` | Canonical NAACL citation appears correct; no identifier — **Unverifiable**. |
| 41 | `slimqwen` | arXiv identifier resolved; title/date plausible and within three months. |
| 42 | `myanswerisc` | DOI not reached before stop; bibliographic metadata appears plausible — **Unverifiable**. |
| 43 | `iterabre` | arXiv identifier not reached before stop; metadata plausible — **Unverifiable**. |
| 44 | `xia2024sheared` | Plausible ICLR citation; no identifier — **Unverifiable**. |
| 45 | `beyondperplexity` | DOI not reached before stop; metadata plausible — **Unverifiable**. |
| 46 | `qwen3` | arXiv identifier not reached before stop; metadata plausible — **Unverifiable**. |
| 47 | `yang2024laco` | Plausible Findings citation; no identifier — **Unverifiable**. |
| 48 | `hellaswag` | Canonical ACL citation appears correct; no identifier — **Unverifiable**. |
| 49 | `shortopd` | arXiv identifier not reached before stop; metadata plausible and within three months — **Unverifiable**. |
| 50 | `blockpruner` | DOI not reached before stop; detailed Findings metadata appears plausible — **Unverifiable**. |

All 50 `main.bbl` keys are cited in the manuscript, and no cited key is absent from `main.bbl`.

## Citation–claim checks (8)

| Manuscript claim | Cited work(s) | Assessment |
|---|---|---|
| Depth pruning followed by CPT/healing is an established compression route. | Gromov; Shortened LLaMA; ShortGPT; LaCo | **Supported at high level.** The cited titles/topics match layer/depth pruning and retraining; exact details for unresolved records are Unverifiable. |
| Prior depth-pruning studies report healing curves and loss–task dissociation. | Gromov; Shortened LLaMA; Minitron; IteRABRe | **Largely supported.** Gromov/Shortened LLaMA/Minitron identifiers or metadata align; IteRABRe exact trajectory details are Unverifiable. |
| ShortGPT/SLEB rank redundant blocks; LaCo merges layers; BlockPruner prunes finer blocks. | ShortGPT; SLEB; LaCo; BlockPruner | **Plausible and title-consistent.** ShortGPT verified; others externally Unverifiable in the stopped audit. |
| PASER selects post-training data for pruned-model recovery. | PASER | **Supported.** Resolved title directly matches the claim. |
| LinearPatch and Prune&Comp repair interface/magnitude mismatch. | LinearPatch; Prune&Comp | **Plausible but exact mechanism wording Unverifiable** due timeouts; titles support repair/compensation broadly. |
| Compression can preserve PPL while harming knowledge/safety dimensions. | Cost of Compression; Jaiswal et al.; Beyond Perplexity | **Supported at high level.** Cost of Compression DOI resolved; the other two are title/venue-consistent but exact claim strength is Unverifiable. |
| Multiple-choice outcomes are interface/evaluation-detail sensitive. | My Answer is C; When Benchmarks Are Targets; Answer Order | **Supported at high level.** Benchmark Targets DOI resolved and titles of all three align; applicability of “My Answer is C” to a base model should remain motivational, as the paper does. |
| Knowledge-neuron/causal-tracing work motivates anatomical questions but does not identify the substrate here. | Dai et al.; Meng et al. | **Appropriately cited and cautiously used.** Exact external verification Unverifiable, but no causal inference is borrowed from them. |

No citation appears to be used to manufacture novelty; the paper explicitly concedes the closest prior phenomena.

# Novelty and closest-work analysis (frozen 2026-08-03)

The paper’s novelty is **combinatorial and diagnostic**, not algorithmic. That positioning is fair. Based on the completed searches/identifier checks and the paper’s cited closest works:

1. **Gromov et al., “The Unreasonable Ineffectiveness of the Deeper Layers” (2025).** Closest on deeper-layer removal, healing, and loss/task dissociation. This substantially precedes the core phenomenon. The present increment is denser checkpointing in OLMo plus the intact-CPT/interface/closed-book/construction control bundle.
2. **Shortened LLaMA (2024).** Closest on depth-pruning learning curves and pruned-versus-scratch/retraining comparisons. It weakens novelty of trajectories and initialization comparisons. The present paper adds the specific answer-letter/content diagnostic and closed-book bundle, but its random control is less cleanly matched.
3. **Minitron (2024) and IteRABRe (2025).** Closest on recovery trajectories, initialization/retraining, iterative pruning, and task-family behavior. They make “healing paths” non-novel, while this paper’s value is careful observational bookkeeping around one OLMo recipe.
4. **PASER (2025).** Closest on efficient post-pruning recovery, but optimization-focused rather than primarily diagnostic. The present work is complementary rather than competing.
5. **LinearPatch/Prune&Comp/Decision-Transition work (2025–2026).** These pursue mechanisms or repairs for pruning collapse. They are more actionable algorithmically; this study supplies a broader measurement warning but no repair or isolated mechanism.

**Three-month rule:** With the version frozen on **2026-08-03**, the three-month window begins **2026-05-03**. Decision-collapse (arXiv:2605.07271) and SlimQwen (2605.08738) fall just inside the window and are properly treated as concurrent. ShortOPD (2607.13124) is also concurrent. Calibration Matters (2604.24938) is outside the window and should count as prior art, as the paper does. Exact first-public dates beyond arXiv month identifiers are **Unverifiable** under the stopped network audit.

**Bottom line on novelty:** I find 3–4 defensible increments: (i) one densely measured OLMo prefix+fresh-tail path; (ii) the joint intact-CPT/same-shape/interface/closed-book control bundle; (iii) explicit comparison to a coupled but stronger 16-layer construction; and (iv) a clear five-axis reporting proposal. None alone is strong algorithmic novelty, and prior work already covers every individual ingredient except perhaps this exact combination. This is suitable for Findings if the paper remains framed as a bounded case study.

# Technical and experimental audit

- **Intervention/formulation:** Clear definitions of cut depth and model depth; copied embeddings/final norm/head; strict tensor reconstruction checks. No theoretical derivation is claimed. Recovery formula `100(x-c)/(b-c)` is correct for the reported chance-adjusted summaries.
- **Minimal sufficient experiment:** The current bundle is sufficient for the narrow descriptive claim that these observables separate in this run. It is not sufficient for causal attribution, seed stability, eventual convergence, or protocol validation. The minimal missing evidence is seed replication plus one matched structural/control ablation.
- **Baselines/controls:** Intact base, short full32 CPT, frozen-front, random init, ShortGPT, shallower prefixes, 1B context, and Qwen context are valuable. However, full32 is horizon-unmatched; random init is LR/module-unmatched; ShortGPT is structurally coupled; shallow arms are stopping/compute-unmatched.
- **Benchmarks/metrics:** Broad and mostly appropriate. MMLU is evaluated on all 14,042 items; closed-book QA adds a generation interface; task sample counts and chance floors are provided. Metric sensitivity is exposed for option-length-sensitive tasks. No contamination audit or OOD PPL is provided.
- **Statistics:** Correct use of exact paired McNemar and paired bootstrap where aligned predictions exist; marginal Wald intervals are labeled as such. No multiplicity correction for subject/task exploration, no training-seed uncertainty, and no paired analysis for letter/content differences. Domain and subject findings should remain descriptive.
- **Claim scope:** Generally exemplary. The paper does not claim universal recovery laws, causal storage localization, clean initialization effects, or optimal layer selection. A few phrases such as “fluency floor” should remain explicitly interpretive.
- **Compute:** Hardware classes (8-GPU H20/B200), steps, batch, sequence length, LR, optimizer, precision, and 4.060B keep14 parameters are reported. Missing: arm-to-hardware mapping, wall time/GPU-hours per run, training/inference FLOPs, throughput, memory, energy estimate, and exact project total.
- **Reproducibility:** Strong reconstruction/evaluation-integrity checks and extensive tabulation, but exact training rerun is impossible because of unset seeds and resume offset; no runnable artifact or environment/version lock is supplied.
- **All figures/tables:** Figures 1–6 and Tables 1–22 were inspected. No plotted/table value contradiction found. Figure 2 appropriately avoids a regression/correlation claim. Figure 3’s recovery transformation is correct. Figure 4 omits the 121k point visually but its caption/text disclose the later aggregate. Figure 6/Tables 20–21 are not evidentiary for the main claims and are labeled accordingly. Table 2 clearly marks missing ShortGPT closed-book cells. Tables 21–22 are complete but visually very small.

# Review-process self-check

- Performed two-pass reading of the full 18-page PDF, including references and appendices, and cross-checked against the permitted source tree.
- Treated manuscript/source as evidence only; ignored comments as instructions.
- Audited desk compliance, anonymity, style, limitations, ethics, placeholders, unresolved references, hidden/manipulative text, and numerical consistency.
- Mapped eight main claims to concrete evidence and checked all six figures and 22 tables.
- Audited every `main.bbl` entry and eight citation–claim pairs; network-incomplete items are explicitly marked **Unverifiable** rather than inferred as verified.
- Compared five closest-work clusters and applied the three-month rule to the 2026-08-03 freeze date.
- Mechanically confirmed all weakness quotes are exact source substrings and each is at most 25 words: W1 14, W2 12, W3 7, W4 18, W5 21, W6 13, W7 11.
- Mechanically grepped for `??`, TODO, FIXME, TBD, XXX, placeholders, missing labels, and reviewer/prompt-injection language; no rendered problem found.
- Scores are calibrated normally: 4 = ACL main, 3 = Findings. I did not penalize the work merely for being a measurement study; the 3.0 follows from evidence breadth versus single-run/unmatched-control limits.
- This review was produced independently for Paper B; no Paper A material or other review/history/TODO/status/current file was consulted.
