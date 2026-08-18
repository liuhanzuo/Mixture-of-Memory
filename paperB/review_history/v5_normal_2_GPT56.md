---
review_mode: normal
soundness: 3.5
excitement: 3.0
overall: 3.0
confidence: 4.0
reproducibility: 2.5
---

# ARR Review

## Paper summary

This paper presents a deliberately narrow measurement case study of continued
pretraining after depth pruning. Its principal OLMo-2-7B construction retains
the first 14 of 32 pretrained decoder blocks, appends two fresh blocks, and is
continued-pretrained for 200k optimizer steps. The central question is not
whether perplexity is useful, but whether improving *in-domain* perplexity is a
sufficient certificate that selected knowledge-sensitive evaluations have
recovered after this intervention.

The principal observed path reduces held-out PPL from 10.826 at 128k to 10.561
at 200k while answer-letter MMLU rises from .3012 to .3191, still far below the
intact base at .6053. At the endpoint, PopQA, TriviaQA, and NQ-open are likewise
below the intact base. The paper supplements this observation with: an intact
full32 continuation available only through 25k; frozen-front and fully random
same-shape operating points; a non-contiguous ShortGPT-16 construction; paired
answer-letter versus complete-option MMLU interfaces; item-level conditional
uncertainty on MMLU; and qualitative 1B and Qwen3 scope checks.

The supported conclusion is appropriately restricted: on these literal
observed paths and measured interfaces, in-domain PPL alone does not certify
base-level performance on the tested knowledge-sensitive evaluations. The
paper explicitly does **not** claim knowledge deletion/localization, causal
factor attribution, a universal recovery law, or failure beyond the measured
budgets.

## Claim–evidence map

| ID | Claim | Main evidence | Assessment |
|---|---|---|---|
| C1 | In-domain PPL is not a sufficient certificate for the measured target evaluations on the principal observed path. | Sec. 5.1, Fig. 1, Table 2: keep14 PPL improves along 128k→153.5k→200k, but final MMLU-L is .319 versus base .605; PopQA/TriviaQA/NQ-open are .142/.294/.060 versus .257/.636/.205. | Supported for this fixed run, horizon, domain, and evaluation set. |
| C2 | The late keep14 path is not numerically flat, but does not exhibit broad catch-up. | Sec. 5.1 and App. Table 12: paired MMLU rises 1.68 pp from 128k to 200k, CI [1.08, 2.29], while most other reported metrics change by under one point from 153.5k to 200k. | Supported conditionally on the realized checkpoints; not seed-level evidence. |
| C3 | Short-horizon corpus shift is not a complete explanation. | Sec. 5.2, Tables 2 and 16: full32 at 25k remains near base across PPL, both MMLU interfaces, and closed-book QA. | Supported only for the first 25k steps; no inference about 200k is licensed. |
| C4 | The answer-letter/content contrast is interface-sensitive and content scoring has a high non-target floor. | Sec. 5.2 and App. Table 15: keep14 content-normalized score .3832 versus letter .3184, but random reaches .3598 content while at .2470 letter. | Supported as a multi-factor interface comparison, not as isolation of symbol binding or “knowledge.” |
| C5 | The base–keep14 gap is not solely an answer-letter artifact. | Sec. 5.2 and App. Table 16: the gap recurs on three zero-shot closed-book generation datasets. | Descriptively supported; paired artifacts/intervals and a ShortGPT generation row are absent. |
| C6 | Endpoint quality depends on exact 16-layer construction, not nominal depth alone. | Sec. 5.3, Tables 2, 9, and 15: ShortGPT-16 reaches PPL 9.780 and MMLU-L .474 versus keep14 10.561/.319 at the same nominal step and token-presentation budget. | Supported as construction dependence; the four coupled structural differences prevent factor attribution. |
| C7 | Frozen and random rows bound alternatives but are not clean ablations. | Secs. 3.1–3.2 and 5.3: frozen changes trainable modules; random changes all initialization, lexical modules, and LR; paired MMLU differences are reported. | Correctly characterized. |
| C8 | The shallow prefix ladder does not establish a depth law. | Sec. 6.2 and App. Tables 3, 5, 9: keep8/10/12 use unequal, metric-informed retained checkpoints while PPL was still falling. | Supported; these rows should remain descriptive only. |
| C9 | 1B and Qwen3 observations provide only directional context. | Sec. 6.3, Fig. 2, and App. Tables 6–8. | Appropriately scoped; not replications. |
| C10 | Recovery papers should separately report likelihood, target evaluation, interface, construction, budget/compute, and run-level uncertainty. | Synthesis in Sec. 7 from the observed discrepancies and confounds. | Reasonable recommendation, but not itself experimentally validated as a universal standard. |

## Strengths

1. **Exceptionally disciplined scope.** The manuscript repeatedly states the
   inferential boundary. For example, Sec. 4 says that an observed path “cannot
   identify where any capability resides,” and Sec. 8 disclaims causal,
   localization, deletion, and universal-dynamics conclusions. This materially
   improves the credibility of a retrospective, single-run study.

2. **The central descriptive result is clear and decision-relevant.** Fig. 1
   and Table 2 make the proxy question immediately legible: PPL can continue to
   improve while the target evaluations remain far from the intact model. This
   is useful guidance for practitioners who might otherwise stop recovery based
   on training-domain likelihood alone.

3. **Controls are presented honestly rather than relabeled as ablations.**
   The paper exposes the 25k/200k horizon mismatch, random-arm LR difference,
   frozen trainable-set difference, ShortGPT construction bundle, and
   post-inspection stopping of shallow runs at first use. This is better
   scientific bookkeeping than is typical for compression comparisons.

4. **The evaluation-interface analysis is valuable.** The paired MMLU table
   shows that conclusions can move substantially when the candidate and scoring
   interface changes, while the random-init null prevents interpreting the
   entire content-score increase as recovered target capability.

5. **Statistical reporting distinguishes item variation from run variation.**
   Exact McNemar tests and paired bootstrap intervals are appropriate for
   aligned fixed-checkpoint predictions, and the paper explicitly says that
   these are not training-seed intervals. The arithmetic and the stated
   52.4B/6.6B nominal token-presentation budgets are internally consistent.

6. **The appendix is unusually complete for the available evidence.** It gives
   all consolidated PPL checkpoints, the keep8 trajectory, full task tables,
   raw/normalized metric sensitivity, paired MMLU comparisons, interface
   controls, closed-book results, group and 57-subject MMLU tables, sample
   counts, prompts, optimizer settings, precision, and provenance limitations.

7. **Presentation and figures are strong.** Both figures are readable and
   appropriately caveated. The main figure integrates the trajectory, null
   operating points, budgets, and inferential boundary without visually hiding
   the unmatched controls.

## Major weaknesses

### W1. The paper’s load-bearing evidence is one historical training realization

- **Location / short quote:** Abstract, lines 6–8: “In the single principal
  keep14+fresh2 run”; Limitations, lines 3–7: “single training runs” and
  “Training seeds were not explicitly set.”
- **Problem:** C1 is a valid counterexample for the realized path, but the paper
  cannot estimate how often or how strongly the PPL/task separation occurs
  under ordinary variation in initialization, data order, block selection, or
  optimization. The significant item-level intervals do not address this
  uncertainty.
- **Why it matters:** The practical recommendation is broader than the
  evidential unit. A single realized path can refute a universal certificate,
  but it gives weak evidence about prevalence, robustness, and expected
  magnitude—information needed to turn the case study into main-conference
  empirical guidance.
- **Remedy / severity:** **Major, experiment-required.** The minimum useful
  addition is 3 independent seeds for keep14, each evaluated at a predeclared
  common checkpoint set (at least 128k, 153.5k, and 200k), reporting per-run
  PPL and target scores plus mean, SD, and a run-level CI. If full repetition is
  unaffordable, two additional shorter replicas with a prospectively fixed
  horizon would still materially improve the paper. Writing alone cannot repair
  this.
- **Mechanical verification:** The source contains no independent keep14
  training seed, and the limitations explicitly state that historical seeds
  were not set.

### W2. There is no matched long-horizon intact control, so the strongest corpus/time explanation remains open at 200k

- **Location / short quote:** Sec. 5.2: “no full32 result is available after
  25k”; Limitations: “full32 ends at 25k.”
- **Problem:** The full32 arm rules out a catastrophic same-corpus effect over
  the first 25k steps, but it does not show what 52.4B nominal token
  presentations do to the intact model or how much of the final gap should be
  attributed to pruning/regrowth rather than long-horizon continuation,
  forgetting, or the missing loader-offset bookkeeping.
- **Why it matters:** The claim is carefully worded as proxy insufficiency, so
  this does not invalidate C1. It does, however, limit interpretation of the
  endpoint and weakens C3 as a control of the principal horizon.
- **Remedy / severity:** **Major, experiment-required.** Run full32 to the same
  200k-step data schedule, or preferably compare at both equal nominal tokens
  and measured FLOPs. Evaluate the identical PPL, both MMLU interfaces, and all
  three closed-book tasks. A smaller minimum is a dense checkpoint grid past
  25k sufficient to show whether the intact branch remains near base. Writing
  alone cannot repair this.
- **Mechanical verification:** Table 2 gives full32 budget “25k / 6.6B” versus
  keep14 “200k / 52.4B”; no later full32 row appears in the complete appendix.

### W3. The intervention controls do not identify even a minimal causal contrast

- **Location / short quote:** Sec. 5.3: ShortGPT “inherits two additional
  pretrained blocks, selects non-contiguously, retains block 31, and appends no
  fresh tail”; Sec. 3.2 says random-init uses a different LR.
- **Problem:** ShortGPT, keep14, frozen, and random are informative operating
  points, but every comparison changes several factors. Consequently, the study
  cannot tell whether the observed deficit is driven mainly by retained-block
  count, prefix selection, loss of the original final block, fresh-tail
  initialization, lexical-module initialization, trainability, or LR.
- **Why it matters:** The paper itself does not overclaim causality, but this
  leaves the empirical contribution mostly at “one recipe failed while a
  bundled alternative did better.” That is useful diagnosis, yet scientifically
  less informative and less actionable than a minimal controlled contrast.
- **Remedy / severity:** **Major for main-conference impact; moderate for
  Findings.** Add at least one 2×2 or one-factor control at 16 layers. The most
  diagnostic minimum would hold inherited count and LR fixed while comparing
  (i) prefix14+two fresh blocks, (ii) prefix14+two inherited later blocks, and
  (iii) a selected 14+two-fresh construction; alternatively isolate retention
  of block 31. Use the same data order, checkpoint grid, and evaluation suite.
- **Mechanical verification:** Table 2 and Secs. 3.1–3.2 explicitly list these
  coupled factors; no one-factor row exists.

### W4. Reproducibility of the principal result is materially blocked

- **Location / short quote:** Limitations: “an unrecorded within-epoch loader
  offset”; Appendix B.3: relevant commits “are not ancestors of the public
  origin/main.”
- **Problem:** The frozen source describes the recipe well, but exact keep14
  replay is impossible from the reported information, the historical training
  seeds are unavailable, task-specific evaluator revisions are local-only, and
  aligned closed-book prediction files were not consolidated. Hardware,
  wall-time, GPU-hours, and measured FLOPs are also absent.
- **Why it matters:** Readers cannot independently reproduce the exact training
  trajectory or fully audit every load-bearing evaluation. This is especially
  important because small implementation/interface choices are themselves a
  central message of the paper.
- **Remedy / severity:** **Major reproducibility weakness, partly
  artifact-fixable.** Publicly release the exact evaluator commits or a
  self-contained archive, configs, environment lockfile, checkpoint metadata,
  per-item predictions for all headline tasks, and a deterministic data-order
  prescription for new runs. The original keep14 path will remain exactly
  unreproducible without the loader offset; that fact should remain explicit.
- **Mechanical verification:** Appendix B.1/B.3 explicitly records each missing
  element. No claim is made that the local hashes are publicly retrievable.

## Minor weaknesses

### W5. “Knowledge-sensitive” is heterogeneous and the scope lacks modern generative/functional checks

- **Location / short quote:** Sec. 5.1: “These evaluations do not exhaust
  capability.”
- **Problem:** MMLU, PopQA, TriviaQA, and NQ-open are reasonable, but they cover
  a limited slice of factual/academic behavior. The remaining harness tasks mix
  language modeling, commonsense, cloze, and passage-grounded behavior, while no
  instruction-following, math/code generation, long-context, calibration, or
  safety behavior is tested.
- **Why it matters:** The paper scopes its conclusion correctly, but readers
  should not infer a general capability-recovery diagnostic from this suite.
- **Remedy / severity:** **Minor-to-moderate.** Add one independently motivated
  free-form functional suite and one out-of-domain likelihood set, or sharpen
  the title/abstract terminology to the exact evaluated families.
- **Mechanical verification:** Table 18 lists the full suite; no such additional
  evaluation is present.

### W6. The content-MMLU contrast changes too many interface factors at once

- **Location / short quote:** Sec. 3.3: the protocols “simultaneously change
  prompt, candidate string, tokenization, and normalization.”
- **Problem:** This proves sensitivity but does not reveal whether the shift
  comes from answer-symbol mapping, option visibility, sequence length,
  normalization, or prompt format.
- **Why it matters:** Interface sensitivity is one of the paper’s distinctive
  diagnostics, so a more controlled decomposition would substantially improve
  interpretability.
- **Remedy / severity:** **Minor.** Add a factorial evaluation-only ablation:
  visible versus hidden options, letter versus option-text candidates, summed
  versus normalized likelihood, with the same prompt wherever possible. No new
  training is required.
- **Mechanical verification:** Table 15 reports only the bundled letter,
  content-raw, and content-normalized protocols.

### W7. Closed-book uncertainty and the strongest construction comparison are incomplete

- **Location / short quote:** Sec. 5.2: “the saved paper bundle does not contain
  the aligned per-item artifacts”; Table 2 has missing ShortGPT closed-book
  cells.
- **Problem:** The recurrence across three datasets is persuasive at the
  base–keep14 scale, but there are no paired intervals, and the strongest
  alternative construction is not evaluated on these generative tasks.
- **Why it matters:** This prevents testing whether ShortGPT’s MMLU advantage
  generalizes beyond recognition and makes smaller operating-point differences
  difficult to interpret.
- **Remedy / severity:** **Minor, evaluation-only.** Re-run all headline
  checkpoints including ShortGPT, retain aligned predictions, and report paired
  bootstrap CIs/effect sizes for predeclared comparisons.
- **Mechanical verification:** The absent cells and artifacts are stated
  explicitly in Table 2, Table 16, and Limitations.

### W8. No contamination audit or out-of-domain likelihood is provided

- **Location / short quote:** Limitations: “no contamination audit or
  out-of-domain likelihood.”
- **Problem:** Same-source PPL is intentionally in-domain, but benchmark overlap
  with continued-pretraining data is not assessed and there is no likelihood
  check outside Dolmino/DCLM.
- **Why it matters:** This does not erase the observed divergence, but it limits
  interpretation of both the PPL axis and closed-book scores under continued
  pretraining.
- **Remedy / severity:** **Minor.** Add deduplicated n-gram/minhash overlap
  checks for target evaluations and at least one out-of-domain PPL set.
- **Mechanical verification:** No contamination or OOD-PPL result appears in
  the complete appendix.

## Minimal experiment package that would most change my score

1. **Three keep14 seeds**, prospectively fixed, at common checkpoints, with
   run-level mean/SD/CI.
2. **One full32 200k control** on the identical data schedule and full target
   suite.
3. **One matched one-factor 16-layer control** holding LR, inherited count,
   data order, and training budget fixed.
4. **Evaluation-only completion:** ShortGPT on PopQA/TriviaQA/NQ-open, paired
   per-item files and CIs, plus a decomposed MMLU-interface factorial.
5. Report both nominal tokens and realized FLOPs/accelerator-hours; predeclare
   stopping and primary comparisons; distinguish run-level from item-level
   statistics.

The first two items are the minimum needed for me to consider a main-conference
score. Items 3–4 would make the paper considerably more explanatory and novel.

## Questions for the authors

1. What exact proposition is being falsified by “certificate”: any monotone PPL
   improvement, recovery to a threshold, or recovery relative to an intact
   continuation? Please state it operationally enough that a future study could
   pre-register a pass/fail test.
2. Why were 128k, 153.5k, and 200k chosen as the load-bearing keep14 checkpoints,
   and were all target evaluations at those checkpoints selected before seeing
   results?
3. Can full32 be extended to 200k, even for only the primary metrics? If not,
   what concrete resource or checkpoint limitation prevents it?
4. Can the authors release or reconstruct the local evaluator commits and the
   aligned closed-book predictions before publication?
5. Is the large content-MMLU floor of the random arm driven mainly by option
   length/normalization, lexical likelihood, or prompt removal? A small
   factorial evaluation would answer this.
6. For ShortGPT selection, were the 128 Dolmino windows disjoint from both
   training and the PPL validation shard, and was the layer-selection code fixed
   before inspecting target evaluations?
7. Are the base and full32 PPL values directly comparable to compressed-arm PPL
   under identical tokenizer, BOS, packing, and shard-merging code paths?
8. Which claims, if any, would the authors expect to change if the target were
   recovery relative to a size-matched model trained from scratch rather than
   the intact 7B base?

## Suggestions

- Put a one-sentence formal operational definition of “certificate” immediately
  before Sec. 4’s notation, including the reference level of “recovery.”
- Preserve the current “operating point” terminology; it is accurate and
  prevents readers from mistaking confounded rows for ablations.
- Move a compact statement of the unavailable seed/loader-offset/public-code
  limitations into the reproducibility paragraph of the main text, not only the
  Limitations/appendix.
- Add a panel plotting target score against PPL for every checkpoint available
  within each construction, but avoid fitting a cross-arm regression because
  the constructions and budgets are unmatched.
- Report absolute differences and base-relative recovery next to each headline
  target metric; the current chance-adjusted recovery table is helpful but
  applies only to part of the suite.
- If no new training can be done, position the paper even more explicitly as a
  documented counterexample/data note and prioritize release of the exact
  artifacts.

## Novelty analysis (cutoff: 2026-05-04)

I used the requested three-month rule relative to the 2026-05-04 novelty
cutoff: work first public after 2026-02-04 is treated as concurrent rather than
novelty-destroying. Searches were performed for combinations of layer/depth
pruning, continued pretraining/healing, perplexity–task gaps, recovery
trajectories, MMLU interfaces, and knowledge-sensitive evaluation.

### Closest pre-cutoff work

| Work | Overlap with this paper | Remaining increment here |
|---|---|---|
| Gromov et al., *The Unreasonable Ineffectiveness of the Deeper Layers* | Layer removal, healing, QA/task evaluation, and interpretation of what pruning preserves. | This paper is more conservative about localization and focuses on an OLMo prefix+fresh-tail observed path, but the broad pruning/healing–task-gap idea is prior. |
| Kim et al., *Shortened LLaMA* | Depth pruning, continued pretraining curves, recovery comparisons, scratch/retraining baselines. | The paired MMLU interfaces, same-source intact 25k branch, closed-book trio, and explicit proxy-certificate framing are the main additions. |
| Sreenivas et al., *Minitron* | Structured pruning/distillation trajectories, initialization and task behavior. | The present work contributes a narrower diagnostic package rather than a competitive compression method. |
| Wibowo et al., *IteRABRe* | Iterative pruning/recovery and weak MMLU recovery trajectories. | OLMo-specific fixed-path measurement, interface null, and closed-book checks. |
| Jaiswal et al., *Compressing LLMs: The Truth Is Rarely Pure and Never Simple* | Directly establishes that perplexity can miss knowledge-intensive deficits in compressed LMs. | The present paper narrows this to post-depth-pruning recovery along observed checkpoints and adds construction/interface bookkeeping. |
| Namburi et al., *The Cost of Compression*; Xu et al., *Beyond Perplexity* | Multi-dimensional compression evaluation beyond aggregate likelihood. | Same as above: a specific recovery-path case and control combination. |
| Wang et al., *My Answer is C*; Alzahrani et al.; Gupta et al. | Multiple-choice results are sensitive to answer interface, implementation, and option order. | The paper applies an interface null specifically to depth-pruned recovery, but does not introduce the general phenomenon. |

### Concurrent work under the three-month rule

- Kim et al., *Rethinking Layer Redundancy* was first submitted 2026-04-27 and
  is therefore concurrent. It emphasizes calibration/search sensitivity rather
  than this paper’s proxy-validity package.
- Shi et al., *Understanding Performance Collapse...* (2026-05-08), SlimQwen
  (2026-05-09), and later recovery work such as ShortOPD (2026-07-14) are
  post-cutoff/concurrent. They should be discussed for context but should not be
  used to erase novelty under the requested rule.

### Novelty conclusion

The paper does **not** introduce the facts that (i) pruning can hurt downstream
capabilities despite acceptable perplexity, (ii) depth-pruned models can have
different recovery trajectories, or (iii) multiple-choice scoring is
interface-sensitive. Its novelty is the *combination* of a documented OLMo
prefix+fresh-tail path, a short intact continuation, same-shape null operating
points, paired MMLU interfaces, three closed-book evaluations, and unusually
explicit inferential bookkeeping. I regard this as a useful but incremental
empirical package, more naturally calibrated to Findings without the minimal
replication/matched-control experiments above.

Search completeness beyond the returned arXiv/Crossref records is
**Unverifiable** because network retrieval was incomplete at review time.

## Citation audit

### Bibliography integrity

- `main.bbl` contains 33 entries; all 33 are cited in the manuscript, and all
  manuscript citation keys resolve to a `main.bbl` entry.
- DOI/arXiv identity was mechanically verified for 24 entries.
- The following nine entries had no DOI/arXiv identifier in `main.bbl`; their
  complete external metadata is therefore **Unverifiable** in this bounded
  audit: DRPruning, MMLU, Jaiswal et al., TriviaQA, Natural Questions, PopQA,
  Compact Language Models, SLEB, and Sheared LLaMA. Several are standard and
  appear bibliographically plausible, but I do not mark unverified metadata as
  verified.
- Exact venue-status wording for LinearPatch and Prune&Comp is
  **Unverifiable** from the retrieved arXiv records alone.

### Citation–claim spot checks

| Manuscript claim | Cited work(s) | Result |
|---|---|---|
| Depth pruning plus continued pretraining/healing is an established compression route. | Gromov; Shortened LLaMA; ShortGPT; LaCo | Supported at the level claimed. |
| Aggregate/perplexity-style metrics can miss knowledge, downstream, or safety effects of compression. | Cost of Compression; Jaiswal et al.; Beyond Perplexity | Supported; Jaiswal is especially close to the paper’s motivating claim. |
| Prior depth-pruning work already reports recovery curves and loss–task gaps. | Gromov; Shortened LLaMA; Minitron; IteRABRe | Broadly supported. |
| ShortGPT/SLEB rank or remove redundant blocks; LaCo merges adjacent layers; BlockPruner searches finer block units. | ShortGPT; SLEB; LaCo; BlockPruner | Supported by titles/method descriptions; SLEB full metadata **Unverifiable** in the bounded network audit. |
| Calibration/task choices affect preferred layer removals. | Siddiqui; Lu et al.; Kim et al. 2026 | Supported in scope; the Kim paper is concurrent under the cutoff rule. |
| Multiple-choice outcomes are interface/evaluation sensitive. | *My Answer is C*; *When Benchmarks Are Targets*; answer-order paper | Supported, though none exactly equals the paper’s bundled content protocol. |
| PopQA, TriviaQA, and Natural Questions are appropriate benchmark citations for the named datasets. | Mallen; Joshi; Kwiatkowski | Bibliographically plausible; full external verification for these no-ID `bbl` entries is **Unverifiable** here. |
| OLMo-2-1124-7B is the underlying model family. | OLMo Team | Verified by arXiv identity. |

I found no citation whose retrieved title directly contradicted the associated
claim. Some entries omit DOI/arXiv data despite having standard published
versions; adding persistent identifiers would improve auditability.

## Desk, formatting, and integrity checks

- **Length/style:** 17-page review PDF: 8 numbered main-text pages, 2 reference
  pages, and 7 appendix pages. It uses the supplied ACL review style, A4 paper,
  anonymous author rendering, line numbers, and embedded fonts. Whether this
  exact main-text allowance matches the active ARR venue cycle is
  **Unverifiable** from the frozen materials alone; no obvious overlength issue
  is visible under the common 8-page long-paper format.
- **Required sections:** Explicit Limitations and Ethical Considerations are
  present.
- **Anonymity:** No author names or affiliations are printed; PDF metadata has
  no author. Local commit hashes and internal provenance do not by themselves
  reveal identity in the frozen manuscript.
- **References/placeholders:** No unresolved refs, missing inputs, missing
  figures, duplicate labels, `??`, TODO/TBD/XXX placeholders, or uncited
  `main.bbl` entries were found.
- **Abstract/table consistency:** Headline values and budgets in the abstract,
  Fig. 1, Table 2, and appendix tables are consistent to the stated rounding.
  Nominal token arithmetic checks: 200k×128×2048 = 52.4288B and
  25k×128×2048 = 6.5536B.
- **Hidden/reviewer-manipulation text:** Source/PDF string scans found no
  reviewer-directed instructions, white-text commands, PDF annotations, or
  embedded raster images outside the two declared vector figures. No suspicious
  content was observed visually.
- **Figures/tables:** All 2 figures and 19 tables were inspected. They are
  legible, captions expose key caveats, and no obvious label/value contradiction
  was found.
- **Ethics:** Risks, energy use, licensing, and absence of new human-subject data
  are addressed. No direct ethical blocker is evident.

## Reproducibility assessment

The paper reports many useful details: exact constructions, parameter count,
data source and windowing, validation size, effective batch, optimizer,
warmup/decay, weight decay, precision, gradient clipping/checkpointing, LRs,
checkpoint budgets, prompts, decoding, normalization, sample counts, PPL merge
formula, item-bootstrap seed, and checksums/prefixes for selected artifacts.

However, independent reproduction is substantially limited by absent historical
training seeds, the missing loader offset after the keep14 resume, unavailable
hardware/time/FLOP records, local-only evaluator commits, absent aligned
closed-book predictions, and no public artifact in the frozen source. Thus my
score is 2.5: partial procedural reproducibility and good provenance disclosure,
but not exact replay of the central run.

## Scores

- **Soundness: 3.5/5.0.** The narrow descriptive claim is supported, numbers
  and statistics are internally coherent, and limitations are unusually well
  handled. Soundness is capped by single-run evidence, the unmatched 200k
  control, confounded operating points, and incomplete artifact availability.
- **Excitement: 3.0/5.0.** The measurement lesson is practical and the control
  package is useful, but the core beyond-perplexity and recovery-gap phenomena
  are established; the new contribution is primarily a careful case study.
- **Overall: 3.0/5.0 (Findings).** I would support publication as a Findings
  paper in its current, tightly scoped form. I do not currently see sufficient
  replicated or controlled evidence for ACL main-conference level. A replicated
  keep14 path plus a matched full32 200k control could move me toward 4.0.
- **Confidence: 4.0/5.0.** I read the frozen PDF/source twice, including all
  appendices, checked figures/tables, claims, arithmetic, references, and the
  requested novelty cutoff. Confidence is not 5 because the external search was
  incomplete and parts of the artifact are unavailable.
- **Reproducibility: 2.5/5.0.** See assessment above.

## Review-process self-check

- Read only the specified v5 frozen PDF/source snapshot and the NORMAL template;
  no prohibited review, history, planning, project-state, or calibration
  content was used.
- Completed two full passes including appendices, all claims, tables, figures,
  limitations, ethics, and reproducibility material.
- Re-checked every quoted phrase against the permitted source.
- Re-checked each absence assertion against the complete permitted source file
  list and appendix.
- Mechanically verified citation-key closure, labels/refs/inputs/figures,
  headline arithmetic, token counts, and selected confidence intervals.
- Applied the 2026-05-04 novelty cutoff and three-month rule.
- Network-incomplete facts are explicitly marked **Unverifiable** rather than
  inferred.
