---
review_mode: strict
soundness: 3.0
excitement: 2.5
overall: 3.0
confidence: 4.5
reproducibility: 2.0
---

## Paper Summary

This paper is a deliberately narrow, descriptive case study of continued pretraining after depth pruning/regrowth. Its principal arm keeps OLMo-2-7B blocks 0--13, appends two fresh blocks, and trains to 200k optimizer steps. The paper asks whether improvement in same-source held-out perplexity is a sufficient certificate that selected knowledge-sensitive evaluations have recovered. It evaluates a single keep14 trajectory, a 25k intact full32 continuation, frozen-front and fully random same-shape operating points, and a coupled ShortGPT-16 construction. The principal result is that keep14 perplexity improves from 10.826 to 10.561 between 128k and 200k while answer-letter MMLU remains .319 versus .605 for the intact base; PopQA/TriviaQA/NQ-open also remain substantially lower. Complete-option MMLU raises compressed-model scores but has a high fully-random floor. ShortGPT is stronger, showing construction dependence without identifying the causal factor.

I find the central *observed-path* claim adequately supported and unusually careful about scope. However, the evidence cannot estimate training-run uncertainty, does not include a long-horizon intact counterfactual, omits the strongest construction from closed-book evaluation, and is not exactly reproducible. The scientific increment is primarily a well-audited diagnostic package and reporting recommendation; prior work already establishes both recovery trajectories and perplexity--capability dissociation. On the stated calibration, this is Findings-level rather than ACL-main-level evidence.

## Claims and Minimum-Sufficient-Evidence Audit

I reconstructed the following main claims before judging the evidence:

| Claim | Minimum sufficient experiment | Actual evidence and judgment |
|---|---|---|
| **C1.** In-domain PPL is not a sufficient certificate for the measured evaluations on the literal keep14 path. | A predeclared within-run path showing meaningful PPL improvement while target performance remains materially below a relevant reference, with identical evaluation protocol. | Fig. 1; §5.1; Tables 2, 12, 16. Supported descriptively for this realized path and observed budget. The paper correctly avoids a universal or beyond-200k claim. |
| **C2.** Short-horizon corpus shift is not a complete explanation. | Intact-model CPT on the same corpus, with the same evaluation suite, through at least the horizon being discussed. | full32 is near base at 25k (Table 2/16), so the 25k-only claim is supported; it does not address 200k, which the paper states. |
| **C3.** The answer-letter interface alone does not explain the base--keep14 deficit. | An independently motivated non-letter target evaluation with identical arms and enough examples. | Three zero-shot closed-book QA datasets (Table 16) reproduce a large base--keep14 gap. Supported for those tasks, though no run-level uncertainty is available. |
| **C4.** Complete-option MMLU has a substantial non-target/random floor. | Paired letter/content scoring plus a null model that lacks inherited target knowledge, using the same items. | Table 15: random .247 letter / .360 content-norm; keep14 .318/.383. Supported as protocol sensitivity, not as isolation of one interface factor. |
| **C5.** The keep14 endpoint is construction-dependent, not a property of nominal 16-layer depth. | A second 16-layer construction trained/evaluated at the same nominal budget. | ShortGPT at 200k reaches PPL 9.780/MMLU .474 versus keep14 10.561/.319 (Tables 2, 4, 9). Supported; causal factor attribution is not supported and is not claimed. |
| **C6.** The shallow ladder does not establish a depth law. | Step/token/FLOP-matched depths, common stopping, and replicated runs would be required for a depth law. | Those conditions are absent; §6.2 explicitly rejects such a law. Appropriate non-claim. |
| **C7.** 1B and Qwen arms provide only directional scope checks. | Matched replications across scale/family for cross-family generalization. | Fig. 2 and Table 8 change scale, retained fraction, architecture/corpus, and available metrics. Directional context only, as stated. |
| **C8.** The recommended reporting axes follow from the case. | Evidence that omitted axes change interpretation; not necessarily a universal intervention study. | The null-interface, construction, budget, and uncertainty examples motivate the checklist. It remains a case-study recommendation, not a validated standard, as §7 says. |

Formula/boundary audit: the chance-adjusted recovery formula in Table 10, `100(x-c)/(b-c)`, reproduces the reported keep14 MMLU recovery to rounding; negative recovery below chance is correctly defined. PPL aggregation in Appendix B.2 uses token-weighted NLL before exponentiation, which is the appropriate merge rather than averaging shard PPLs. The main inferential procedures are item-level Wald intervals, paired bootstrap, and exact McNemar tests; the manuscript repeatedly and correctly says these condition on fixed checkpoints and are not training-run uncertainty.

## Strengths

### S1. Strong claim discipline and unusually explicit boundaries

The title, abstract, framing (§4), results (§5), boundaries (§6), conclusion, and Limitations consistently restrict the result to literal observed paths, measured interfaces, and observed budgets. Exact anchors include PDF lines 288--299, 319--331, 436--446, and 494--507. The paper expressly disclaims causal localization, knowledge deletion, universal recovery dynamics, and beyond-budget failure. This substantially reduces the risk of a seductive but unsupported pruning narrative.

### S2. The diagnostic controls materially improve interpretation

The design does more than compare one compressed endpoint with the base. The 25k full32 arm bounds short-horizon same-corpus shift; random initialization exposes the content-MMLU floor; frozen-front supplies another same-shape operating point; ShortGPT falsifies a nominal-depth-only interpretation. Table 2 is especially good at exposing inherited/fresh layers, trainable set, LR, budget, all headline metrics, and the confounds directly in its caption.

### S3. Interface sensitivity is measured rather than merely mentioned

Table 15 evaluates all 14,042 MMLU items under letter and complete-option protocols and includes a fully random null. The manuscript appropriately says the protocols jointly change prompt, candidate string, tokenization, and normalization, so the result is protocol sensitivity rather than an answer-symbol mechanism (§3.3, PDF lines 263--275; §5.2, lines 342--353).

### S4. Appendices are substantive and audit-friendly

The appendix reports every held-out-PPL checkpoint (Table 3), the complete keep8 and 1B trajectories (Tables 5--7), all 57 MMLU subjects (Table 19), sample counts/metrics (Table 18), raw versus normalized variants (Table 13), item-paired MMLU comparisons (Table 14), and training/evaluator provenance (Appendix B). This is considerably more transparent than an endpoint-only compression paper.

### S5. Numerical reporting is internally coherent

I checked five abstract/load-bearing numbers against the body: keep14 PPL 10.826→10.561 and MMLU .301→.319 match Fig. 1/Table 12; full32's 25k availability matches Tables 2/3/16; random letter/content values approximately .247/.360 match Table 15; ShortGPT MMLU .474 matches Tables 2/4/9; and the three base→keep14 QA gaps match Tables 2/16. The reported 128k→200k MMLU gain of 1.68 points and CI [1.08, 2.29] are consistently stated in §5.1/Table 12.

## Weaknesses (ordered by severity)

### W1 — No training-run replication or run-level uncertainty (**Major**)

- **Location:** Fig. 1 caption/annotation (PDF p.2); §3.2; §5.1; Limitations (p.8, lines 509--515).
- **Exact quote (7 words):** “Every trained construction is a single run.”
- **Problem:** Every headline comparison and trajectory is one stochastic realization; historical training seeds were not explicitly set. Item bootstrap and McNemar quantify evaluation-item variation only.
- **Affected claim/norm and impact:** C1, C3, C4, and C5 are credible descriptions of fixed checkpoints, but the paper cannot establish that the observed gap, ranking, or late slope is stable to initialization, data order, optimizer noise, or ShortGPT selection. This is the largest gap between a careful case study and ACL-main-level empirical evidence.
- **Sufficient remedy:** Run at least three independently seeded keep14 trainings and, for the construction claim, at least three ShortGPT trainings under the reported recipe; report per-run PPL/target trajectories and run-level intervals. If compute prevents all arms, replicate keep14 plus the one load-bearing comparator.

### W2 — The intact control does not reach the load-bearing 200k horizon (**Major**)

- **Location:** §3.2 (p.4, lines 242--250); §5.2 (p.5, lines 334--341); Table 2 caption; Limitations.
- **Exact quote (7 words):** “full32 has an available 25k checkpoint only.”
- **Problem:** The same-corpus intact branch ends at 25k while the central keep14 endpoint is 200k/52.4B nominal token presentations.
- **Affected claim/norm and impact:** The paper carefully limits C2 to short-horizon corpus shift, so this does not invalidate C1. It nevertheless leaves a major alternative unresolved at the endpoint: long-horizon corpus/optimization effects in the intact model are unmeasured, weakening interpretation of the 200k base--keep14 gap and preventing a matched recovery counterfactual.
- **Sufficient remedy:** Continue full32 to 200k under the same data schedule and evaluate the identical checkpoints/tasks. A cheaper sufficient alternative for the paper's narrow claim is a preregistered matched-token subset with repeated intact and keep14 runs, but the definitive remedy is the 200k full32 trajectory.

### W3 — Closed-book evidence omits the strongest alternative construction (**Major**)

- **Location:** Table 2 caption (p.6); Limitations (p.8, lines 529--533); Table 16.
- **Exact quote (7 words):** “Missing ShortGPT closed-book cells were not evaluated.”
- **Problem:** ShortGPT is the only substantially stronger 16-layer construction, yet PopQA/TriviaQA/NQ-open are absent for it.
- **Affected claim/norm and impact:** C3 establishes that keep14's MMLU letter deficit is not purely a letter-interface artifact, and C5 establishes construction dependence only for PPL/MMLU. Without ShortGPT QA, the paper cannot tell whether construction dependence generalizes to the independent generative evaluations or whether ShortGPT's gain is partly MMLU-interface-specific. This limits the proposed “joint diagnostic package.”
- **Sufficient remedy:** Evaluate the existing ShortGPT checkpoint on the exact Table 16 harness and report all three scores, ideally with aligned per-item predictions and paired intervals against keep14/base.

### W4 — Same-source PPL is narrow and unchecked for contamination (**Major**)

- **Location:** Protocol boundary (p.2, lines 98--100); §3.2; Limitations (p.8, lines 533--534).
- **Exact quote (10 words):** “PPL is in-domain, with no contamination audit or out-of-domain likelihood.”
- **Problem:** The proxy variable is a single disjoint shard from the same Dolmino/DCLM source as CPT. Disjointness does not establish absence of near-duplicates or benchmark contamination, and no out-of-domain likelihood tests whether the dissociation is specific to the training distribution.
- **Affected claim/norm and impact:** C1 is explicitly about *in-domain* PPL and remains logically valid, but its practical reach as a warning about “perplexity recovery” is narrow. A same-source metric may be unusually easy to improve after 52.4B nominal presentations and may not represent commonly reported held-out/generalization likelihood.
- **Sufficient remedy:** Add deduplication/overlap checks for the validation shard and target benchmarks, plus at least one genuinely out-of-domain corpus PPL evaluated at the same checkpoints. Reframe any broader recommendation according to whether the dissociation persists.

### W5 — Exact reproduction of the principal run is impossible from the recorded state (**Major**)

- **Location:** Appendix B.1 (pp.12--13, lines 857--867); Limitations (p.8, lines 541--548).
- **Exact quote (17 words):** “Exact reproduction of keep14 is further blocked by an unrecorded within-epoch loader offset after a 34.5k resume.”
- **Problem:** The load order after resume cannot be reconstructed, historical seeds are unavailable, GPU-hours are unavailable, and key evaluator commits are local-only rather than released.
- **Affected claim/norm and impact:** This does not make the displayed fixed-checkpoint measurements internally unsound, but it prevents independent rerunning of the central training path and makes exact provenance dependent on inaccessible local state. Reproducibility is therefore materially below normal ARR expectations.
- **Sufficient remedy:** Release all evaluator commits/scripts and prediction artifacts now; for the training result, perform and publish a fresh seeded reproduction with a resumable sampler/loader state, full config, environment lockfile, checkpoint hashes, and measured compute. The historical trajectory itself cannot be made exactly reproducible retroactively.

### W6 — Novelty is limited to a combination of controls and bookkeeping (**Minor**, but score-relevant for excitement)

- **Location:** §2 opening and Table 1; §2 “Recovery paths and retraining controls.”
- **Exact quote (14 words):** “We do not propose a pruning criterion or claim recovery-path analysis itself as new.”
- **Problem:** The paper's own closest-work analysis concedes that recovery curves, loss--task gaps, initialization comparisons, iterative recovery, and beyond-perplexity evaluation all pre-exist. The increment is one OLMo construction plus a particular bundle of controls/interfaces.
- **Affected claim/norm and impact:** Soundness is not harmed, but the conceptual/methodological advance is modest. This is a useful negative measurement case and reporting lesson rather than a new method or broad empirical law, which caps excitement and main-conference impact.
- **Sufficient remedy:** Either add a genuinely general empirical result (replicated, matched designs across multiple families/constructions) or sharpen the deliverable into a reusable proxy-validity benchmark/protocol with released artifacts and evidence that it changes conclusions across prior pruning methods.

## Questions That Could Change the Score

1. Are there independent reruns of keep14 or ShortGPT not included in the paper? Run-level trajectories showing the same qualitative separation would most directly improve soundness.
2. Can full32 be continued to 200k, or is there another matched long-horizon intact checkpoint? A negative answer preserves the narrow claim but keeps the evidence below main level.
3. What are ShortGPT's PopQA, TriviaQA, and NQ-open scores under the exact Table 16 harness? If it substantially closes those gaps too, the construction-dependence result becomes much stronger.
4. Will the local-only evaluator commits, full prediction files, and anonymous checksum/config bundle be public at review time? This directly affects reproducibility.
5. Was any exact/near-duplicate audit performed between DCLM/Dolmino training/validation data and MMLU/QA items? If so, please report method and counts.

## Non-scoring Suggestions / Typos

- Table 13 says normalization divides by continuation **character** length, whereas §3.3, Table 15, and Table 18 describe per-token normalization. This should be reconciled; it appears to be a wording error or a metric inconsistency.
- The title asks about “perplexity” generally, while the evidence is specifically same-source in-domain PPL. Consider putting “same-source in-domain” in the title or subtitle.
- Report model parameter counts for all headline arms, not only keep14, and add measured inference latency/memory if deployment motivation remains in the introduction.
- In Table 1, define precisely how “trajectory,” “partial,” and checkmarks were assigned; several cells require reading the caption to understand the taxonomy.
- The sparse layout on pp.7, 10, 16, and 17 is legal but could be compressed to improve readability or add protocol details.

## Citation Audit

### Completeness and metadata procedure

I audited all 33 entries actually present in `main.bbl`. Source citekeys and `main.bbl` matched exactly: 33 cited keys, 33 bibliography entries, no uncited entry and no missing key. Verification used DOI/ACL records when available, arXiv API records for arXiv-indexed work, and title/author/venue matching. A network timeout was not converted to “Not found”; the affected entry was checked via the arXiv API. “Verified” below means the cited identity and core metadata were found; it does not independently validate every result attributed to the work.

| `main.bbl` key / work | Status | Audit note |
|---|---|---|
| `benchmarktargets` — Alzahrani et al. 2024 | Verified | ACL DOI/title/authors/year matched. |
| `linearpatch` — Chen et al. 2025 | Verified | arXiv 2505.24680 title/authors/date matched. |
| `prunecomp` — Chen et al. 2026 | Verified | arXiv 2507.18212 title/authors matched; 2026 refers to venue year while preprint appeared 2025. |
| `deng2025drpruning` | Verified | ACL DOI/title/authors/year matched. |
| `gromov2024unreasonable` | Verified | arXiv 2403.17887 and ICLR 2025 identity matched. |
| `answerorder` | Verified | arXiv 2406.19470 title/authors/year matched. |
| `paser` | Verified | arXiv 2502.12594 title/authors/year matched. |
| `hendrycks2021mmlu` | Verified | ICLR/OpenReview title/authors/year matched. |
| `jaiswal2024truth` | Verified | arXiv 2310.01382 / ICLR 2024 identity matched. |
| `joshi2017triviaqa` | Verified | ACL DOI/title/authors/year matched. |
| `shortenedllama` | Verified | Initial direct-page request timed out; arXiv API 2402.02834 matched title/authors/year. |
| `calibration2026` | Verified | arXiv 2604.24938, submitted 2026-04-27, title/authors matched. |
| `kwiatkowski2019natural` | Verified | TACL DOI/title/authors/volume/pages matched. |
| `lu2024reassessing` | Verified | arXiv 2411.15558 title/authors/year matched. |
| `mallen2023popqa` | Verified | ACL DOI/title/authors/year matched. |
| `fragileknowledge` | Verified | arXiv 2512.22671 title/author/date matched. |
| `men2024shortgpt` | Verified | ACL Findings DOI/title/authors/pages matched. |
| `muralidharan2024compact` | Verified | NeurIPS proceedings/title/authors/year matched. |
| `costcompression` | Verified | EMNLP Findings DOI/title/authors/year matched. |
| `olmo2` | Verified | arXiv 2501.00656 title/team/date matched. |
| `decisioncollapse` | Verified | arXiv 2605.07271 title/authors matched; first submission 2026-05-08. |
| `siddiqui2024deeper` | Verified | arXiv 2407.16286 title/authors/year matched. |
| `song2024sleb` | Verified | ICML/PMLR title/authors/year matched. |
| `minitron` | Verified | arXiv 2408.11796 title/authors/year matched. |
| `slimqwen` | Verified | arXiv 2605.08738 title/authors matched; first submission 2026-05-09. |
| `myanswerisc` | Verified | ACL Findings DOI/title/authors/year matched. |
| `iterabre` | Verified | arXiv 2503.06291 title/authors/year matched. |
| `xia2024sheared` | Verified | ICLR/OpenReview title/authors/year matched. |
| `beyondperplexity` | Verified | EMNLP Findings DOI/title/authors/year matched. |
| `qwen3` | Verified | arXiv 2505.09388 title/date matched. |
| `yang2024laco` | Verified | EMNLP Findings DOI/title/authors/pages matched. |
| `shortopd` | Verified | arXiv 2607.13124 title/authors matched; first submission 2026-07-14. |
| `blockpruner` | Verified | ACL Findings DOI/title/authors/pages matched. |

No entry was classified “Not found.” No remaining entry required “Unverifiable” at the identity/metadata level.

### Load-bearing citation--claim matches

1. **Gromov et al. as the closest antecedent for post-healing loss--task dissociation:** **Match.** The paper studies layer removal/healing and downstream QA behavior; this is directly relevant.
2. **Shortened LLaMA reports CPT versus LoRA and scratch/pruned recovery comparisons:** **Match.** Its abstract and paper scope explicitly compare retraining methods and continued pretraining.
3. **Minitron studies pruning/distillation trajectories and initialization/retraining choices:** **Broadly matched.** It is a comprehensive pruning/distillation study, though Table 1's compact taxonomy necessarily compresses details.
4. **Jaiswal et al. show that low perplexity can coexist with deficits on knowledge-intensive compressed-LM benchmarks:** **Match.** This is a central motivation of LLM-KICK.
5. **Wang/Alzahrani/Gupta support multiple-choice interface sensitivity:** **Match.** They respectively address first-token versus text answers, leaderboard/evaluation sensitivity, and answer-order sensitivity.
6. **LinearPatch/Prune&Comp attribute damage to interface or magnitude mismatch and repair it:** **Match.** Both works explicitly diagnose activation/magnitude discontinuities and propose compensation/patching.
7. **Siddiqui/Lu/Kim support task/calibration dependence of selected layers:** **Match.** The works examine task tradeoffs, pruning choices, and calibration sensitivity.
8. **SlimQwen/ShortOPD “further cover matched-token initialization, progressive recovery, and recognition/generation behavior”:** **Partially verified / Unverifiable in full.** Identity and contemporaneous dates were verified; within the time budget I did not independently inspect both full papers deeply enough to assign each subclaim to a specific source. This point is not load-bearing for the paper's empirical conclusion.

## Novelty Search Summary

**Cutoff applied:** 2026-05-04. Work first public after that date is treated as contemporaneous/post-cutoff rather than novelty-destroying prior art. I ran five searches around (i) depth/layer pruning + recovery, (ii) continued pretraining after pruning, (iii) perplexity versus downstream/knowledge evaluation, (iv) layer-pruned repair/recovery, and (v) calibration/selection effects.

Closest pre-cutoff work:

1. **Gromov et al., “The Unreasonable Ineffectiveness of the Deeper Layers” (2024 preprint; ICLR 2025):** closest conceptual antecedent for layer removal, healing, loss/task behavior, and claims about knowledge retention.
2. **Kim et al., “Shortened LLaMA” (2024):** closest for depth pruning plus continued pretraining trajectories and retraining/scratch comparisons.
3. **Sreenivas et al., “The Minitron Approach” (2024):** broad structured pruning/distillation study with downstream evaluation and practical recovery trajectories.
4. **Jaiswal et al., “Compressing LLMs: The Truth Is Rarely Pure and Never Simple” (2023/ICLR 2024):** closest to the high-level “perplexity is not enough for compressed LMs” message.
5. **Wibowo et al., “IteRABRe” (2025):** iterative block removal/recovery with task-family trajectories and limited MMLU recovery.

Also close: PASER (recovery-data selection), LinearPatch and Prune&Comp (repair mechanisms), and the 2026-04-27 calibration paper. The search supports the manuscript's own modest novelty statement: the increment is the **combination** of an OLMo prefix+fresh-tail path, a short intact branch, same-shape null operating points, paired MMLU interfaces, closed-book QA, and explicit confound bookkeeping. I did not find a pre-cutoff paper with that exact package, but most components and the main caution already exist separately.

Post-cutoff/contemporaneous handling:

- Decision-transition paper: first arXiv submission 2026-05-08 — 4 days after cutoff; contemporaneous.
- SlimQwen: 2026-05-09 — contemporaneous.
- ShortOPD: 2026-07-14 — contemporaneous.

Thus these do not reduce novelty under the requested cutoff, though they narrow the current landscape. No unresolved search failure was interpreted as absence; any uninspected full-text subclaim is marked Unverifiable above.

## Limitations, Ethics, and Desk-Reject Risks

- **Page/style:** 17 A4 pages: 8 pages through Limitations/Ethical Considerations, 2 reference pages, 7 appendix pages. The main content concludes on p.7 and the exact `Limitations` section begins p.8. The PDF uses the official ACL review style with line numbers and “Anonymous ACL submission.” I found no obvious long-paper page-limit violation under the usual separation of references/appendices, but the final administrative determination belongs to ARR.
- **Required sections:** Exact heading `Limitations` is present. `Ethical Considerations` is present and addresses inherited model/data risks, no new human subjects, energy, incomplete compute records, and licensing.
- **Anonymity:** No author names, affiliations, emails, acknowledgments, or identifying public URLs appear in the rendered paper. Local commit hashes are described as provenance and not linked to identities.
- **References/placeholders:** No unresolved `??`, TODO, FIXME, placeholder, or undefined citation/reference was found. Build log contains underfull-box warnings but no unresolved-reference warning.
- **Prompt injection/manipulation:** I inspected extracted text, PDF objects/strings, font sizes, and all rendered pages. I found no reviewer-directed instruction, score manipulation, hidden off-page text, suspicious white/tiny prose, or embedded JavaScript. Small text is confined to figures/tables and line numbers and is visibly rendered.
- **Figures/tables:** I visually inspected Figures 1--2 and Tables 1--19. They are readable at zoom, captions expose most protocol caveats, and no obvious graphical contradiction was found. Tables 18--19 are small; Table 13 has the token/character-normalization wording issue noted above.
- **Desk risk:** Low on formatting/anonymity/required sections; nonzero on artifact/reproducibility expectations because core evaluator commits are not public and the principal run cannot be exactly reproduced. This is more a score issue than an obvious desk rejection from the PDF alone.

## Scores

### Soundness: 3.0 / 5.0

The narrow fixed-checkpoint conclusion is supported, numerically coherent, and carefully scoped. I do not score higher because all trained arms are single runs, the intact control ends at 25k, the strongest comparator lacks closed-book evaluation, and the proxy is only same-source PPL. These prevent robust generalization even within the chosen recipe.

### Excitement: 2.5 / 5.0

The paper is useful and unusually honest, but the scientific novelty is a combination of controls and reporting discipline rather than a new method, causal mechanism, or replicated general result. Prior work already establishes both depth-pruning recovery behavior and failures of perplexity as a capability proxy.

### Overall: 3.0 / 5.0

**Findings-level.** The central message is reliable enough in its narrow form and the evidence presentation is strong. It does not meet my ACL-main threshold because the load-bearing experiments lack run-level replication and a long-horizon intact control, while the novelty is deliberately modest. I considered 3.5 because the paper handles confounds better than many empirical submissions, but under the instruction to choose the lower bin when uncertain, the unreplicated training evidence and incomplete counterfactual keep it at 3.0.

### Confidence: 4.5 / 5.0

I read the entire 17-page PDF twice, including appendices; checked every figure/table; reconstructed claims and minimum experiments; checked key arithmetic, all 33 bibliography entries, citation matching, desk risks, and novelty relative to the requested cutoff. Remaining uncertainty concerns venue policy and a few full-text attribution details explicitly marked Unverifiable.

### Reproducibility: 2.0 / 5.0

The manuscript reports many useful details, hashes, configs, sample counts, and conditional statistical procedures. However, exact principal-run reproduction is explicitly blocked by the missing loader offset and seeds; compute records are incomplete; task-specific evaluator commits are local-only; and aligned closed-book predictions are absent. A fresh, seeded released reproduction could materially raise this score.

## Review-Process Self-Check

- Independently reviewed frozen `v5_20260804_003250` from zero; did not use other reviews/history/TODO/status/current/calibration files.
- Completed two full passes, including appendices, references, figures, and tables.
- Built C1--C8 and specified the minimum sufficient experiment before comparing actual evidence.
- Audited abstract numbers, formulas, controls, metrics, sample counts, seeds/statistics, scope, compute, artifacts, desk risks, and ethics.
- Audited all 33 `main.bbl` entries and 8 load-bearing citation--claim matches.
- Applied novelty cutoff 2026-05-04 and treated 2026-05-08/05-09/07-14 works as contemporaneous.
- Mechanically checked every weakness quote against normalized frozen-PDF text; each is ≤25 words.
- Each weakness contains location, exact quote, problem, affected claim/norm and impact, sufficient remedy, and severity.
- Did not infer “paper lacks X” where the appendices provide X; incomplete external verification is labeled Unverifiable rather than Not found.
