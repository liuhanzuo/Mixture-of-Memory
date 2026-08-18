```yaml
review_mode: strict
soundness: 3.0
excitement: 2.5
overall: 2.5
confidence: 4.0
reproducibility: 2.0
```

# Paper Summary

This paper presents a deliberately narrow empirical case study of continued
pretraining after depth pruning. Its principal construction keeps OLMo-2-7B
blocks 0--13, appends two randomly initialized blocks, and trains the resulting
16-block, 4.060B-parameter model for 200k optimizer steps (52.4B nominal token
presentations). The central question is whether improving held-out,
same-source, in-domain perplexity can *certify* recovery on MMLU and three
closed-book QA evaluations.

The principal observed path does show the claimed separation: from 128k to
200k, PPL improves from 10.826 to 10.561 and answer-letter MMLU from .3012 to
.3191, but the final model remains far below the intact checkpoint (.6053
MMLU; .2571/.6355/.2050 versus .1415/.2940/.0598 on
PopQA/TriviaQA/NQ-open). The paper adds a 25k intact-CPT point, a frozen-front
point, a fully random same-shape point, a coupled ShortGPT-16 point, a second
MMLU interface, and qualitative 1B/Qwen scope checks. It is unusually explicit
that these are operating points rather than causal ablations.

My assessment is that the paper is careful, readable, and mostly sound for a
*descriptive statement about the realized checkpoints*. However, the empirical
design is too incomplete to support a strong recovery/certification study:
there is one run per construction, no 200k intact control, no seed-level
uncertainty, post-hoc stopping for the depth ladder, no matched causal control
for the much stronger ShortGPT point, and no released runnable artifact for the
reported custom evaluations. Moreover, prior work already establishes the
broader loss/perplexity--task dissociation after compression; the remaining
novelty is the specific OLMo bookkeeping and diagnostic combination. I
therefore place this below Findings rather than at ARR Findings level.

# Claim–Evidence Audit

| ID | Paper claim | Minimum sufficient experiment | Evidence actually supplied | Assessment |
|---|---|---|---|---|
| C1 | On the literal keep14 path, improved in-domain PPL does not certify base-level measured performance. | A predeclared within-arm trajectory, target evaluations at the same checkpoints, and enough independent runs to show the separation is not a run accident. | One keep14 run at 128k/153.5k/200k; Table 12 and Figure 1 show PPL 10.826→10.561 and MMLU .3012→.3191, with final MMLU/QA far below base. | **Descriptively supported for this run; inferential strength limited by W1.** |
| C2 | Short-horizon corpus shift is not a complete explanation. | Intact-model CPT on the same corpus, ideally through the same 200k horizon and with matched evaluation checkpoints. | full32 is near base at 25k/6.6B, but no later intact checkpoint exists. | **Supported only for the short horizon explicitly claimed.** |
| C3 | The content-MMLU interface has a high random-init floor and interface choice matters. | Same items across interfaces, an appropriate null, and paired uncertainty. | Table 15 uses all 14,042 items; random is .247 letter versus .3598 content-normalized; paired interface CIs are reported. | **Supported as protocol sensitivity, not as a one-factor answer-symbol mechanism.** |
| C4 | The base–keep14 gap is not merely an answer-letter artifact. | Independently motivated non-letter evaluations using the same checkpoints and disclosed protocols. | Three zero-shot closed-book tasks all show large base–keep14 gaps; prompts, splits, normalization, and sample sizes are disclosed. | **Supported for these three tasks, though no paired files/intervals are available.** |
| C5 | The keep14 endpoint is construction-dependent rather than a property of nominal 16-layer depth. | A different 16-layer construction at a comparable budget; causal attribution would additionally require factor-matched arms. | ShortGPT-16 at 200k obtains PPL 9.780 and MMLU .4739 versus keep14 10.561/.3191, while four construction factors differ. | **Supports construction dependence, not which factor matters.** |
| C6 | The shallow checkpoints do not establish a depth law. | Common stopping, matched tokens/FLOPs, and replicated runs across depths. | Those conditions are absent and the paper explicitly declines the depth-law claim. | **Correctly bounded negative claim.** |
| C7 | The 1B and Qwen observations are directional context, not replications. | Matched model-family/corpus/protocol replications for universality. | One unmatched 1B path and one unmatched Qwen endpoint. | **Correctly scoped.** |
| C8 | The evidence motivates separate reporting of likelihood, target evaluation, interface, construction, budget/compute, and run uncertainty. | Broad multi-study validation for a universal standard, or a clearly labeled case-study recommendation. | A single case study with multiple diagnostic operating points; the paper calls this a case-study recommendation. | **Reasonable non-universal recommendation.** |

# Strengths

## S1. The paper is commendably disciplined about claim scope

The most valuable aspect is not a new pruning algorithm but the consistent
separation of description from causal inference. Examples include the explicit
statement that random/frozen are operating points, the four-factor accounting
for ShortGPT, the refusal to infer a depth law, and the statement that 1B/Qwen
are not replications. These boundaries appear early (PDF lines 88--100), in
the method (§3.2), in Table 2's caption, in §6, and again in Limitations. This
substantially reduces the risk of readers interpreting a heterogeneous set of
historical runs as controlled ablations.

## S2. New Figure 1 and the main table make the operating points legible

Figure 1 is effective. The left panel shows the literal late keep14 trajectory
without falsely depicting a plateau; the right panel makes the 25k full32
horizon, chance-level random answer-letter score, and stronger but confounded
ShortGPT endpoint immediately visible. Table 2 is also unusually honest for a
headline table: it includes inherited/fresh counts, trainable set, LR, budget,
metrics, missing cells, and explicit caveats. I checked the plotted and tabular
values against Tables 3, 9, 12, 15, and 16; the displayed numbers are
consistent up to the paper's stated independent-rerun/rounding differences.

## S3. The interface audit is stronger than a single MMLU number

The paper evaluates all 14,042 MMLU items using letter and complete-option
interfaces, discloses that prompt/candidate/tokenization/normalization all
change, reports both summed and normalized option scores, and includes a
random-init null. Table 15's .2470→.3598 random shift is an important warning
against reading the keep14 content score in isolation. The paper appropriately
calls this protocol sensitivity, not a clean mechanism test.

## S4. Evaluation bookkeeping and fixed-checkpoint statistics are detailed

The appendix gives task splits, sample sizes, chance levels, metrics, exact
closed-book prompt/decoding/normalization, the PPL merge formula, optimizer
hyperparameters, trainable sets, LRs, and checkpoint-specific item-level
statistics. The paired 128k–200k MMLU change (1.68 points, 95% CI
[1.08, 2.29]) and the same-shape comparisons in Table 14 are correctly labeled
as item uncertainty conditional on realized checkpoints rather than
training-run uncertainty. The binomial SEs in Table 11 and the
chance-adjusted-recovery calculations in Table 10 are arithmetically
consistent.

## S5. The paper reports adverse historical details rather than hiding them

The missing loader offset, unavailable training seeds/GPU-hours, local-only
evaluation commits, metric-informed stopping, missing ShortGPT closed-book
cells, and absent closed-book paired artifacts are all disclosed. These facts
hurt the reproducibility and score, but their disclosure materially improves
reviewability and scientific honesty.

# Weaknesses

## W1 — The central trajectory has no run-level uncertainty

- **Location:** Abstract; §3.2; §5.1; Limitations; PDF lines 2--18, 241--259,
  312--331, 509--515.
- **Exact quote (9 words):** “Every trained construction is a single run.”
- **Problem:** The paper's load-bearing C1 is established on one keep14
  realization, while historical seeds were not explicitly set. The paired
  bootstrap only quantifies item variation at fixed checkpoints, not variation
  due to initialization, data order, optimization, pruning choice, or the
  interrupted loader state.
- **Affected claim/norm and why it matters:** This directly affects C1 and the
  norm that a training-dynamics conclusion distinguish a repeatable effect from
  one stochastic trajectory. The data prove what happened in this run, but do
  not estimate how often or how strongly the claimed separation recurs.
- **Sufficient remedy:** Run at least three independent keep14 trainings with
  recorded seeds and intact data-loader state; evaluate the same predeclared
  checkpoints; report per-run PPL/target trajectories and a hierarchical or
  run-level summary. If that is infeasible, narrow the contribution throughout
  to an auditable case report rather than a test with inferential force.
- **Severity:** **Major**

## W2 — The missing 200k intact-CPT control leaves a major alternative open

- **Location:** Abstract; §3.2; §5.2; Table 2; Limitations; PDF lines 15--18,
  241--250, 334--341, 516--520.
- **Exact quote (12 words):** “full32 ends at 25k and cannot control the keep14 200k endpoint.”
- **Problem:** The intact branch sees only 6.6B nominal token presentations,
  whereas keep14 sees 52.4B. Thus the paper cannot measure the full effect of
  long-horizon continuation on the same corpus, evaluator drift, or later
  checkpoint behavior in an unpruned model.
- **Affected claim/norm and why it matters:** C2 is valid only at short horizon,
  but C1 compares a 200k pruned endpoint to the original base and repeatedly
  frames recovery under continued pretraining. A same-budget intact path is the
  minimum control needed to anchor the 200k operating point and distinguish
  pruning-specific failure from long-horizon training effects.
- **Sufficient remedy:** Continue full32 to the same predeclared
  128k/153.5k/200k checkpoints, using the same corpus order and evaluation
  harness; report PPL, both MMLU interfaces, closed-book tasks, realized
  compute, and run-level replication. At minimum, add a matched-token intact
  checkpoint.
- **Severity:** **Major**

## W3 — The experiment set cannot explain the strongest practical result

- **Location:** §3.1; §5.3; Table 2; Discussion; PDF lines 225--230, 367--387,
  469--479.
- **Exact quote (8 words):** “ShortGPT therefore changes inherited count, contiguity/selection, final-block retention, and fresh-tail use together.”
- **Problem:** ShortGPT is dramatically better than keep14 at the same nominal
  step budget, but it changes four structural factors at once. Random also
  changes all initialization and LR, while Frozen changes the trainable set.
  Consequently, the study diagnoses heterogeneous operating points without
  determining which design choice drives the result.
- **Affected claim/norm and why it matters:** C5's narrow construction-
  dependence claim is supported, but the paper's practical value is limited:
  readers cannot tell whether the proxy gap is due to prefix selection, two
  fewer inherited blocks, loss of the original final block, fresh-tail
  insertion, or their interaction. This is especially important because the
  stronger ShortGPT point weakens any intuition that nominal depth or PPL tax
  alone explains the poor keep14 endpoint.
- **Sufficient remedy:** Add a minimal factorized 16-layer matrix: contiguous
  16 inherited blocks; selected 14 inherited plus two fresh; prefix14 plus two
  inherited original blocks; and variants with/without original block 31 and
  with matched LR/trainable modules. Replicate the most informative contrasts.
- **Severity:** **Major**

## W4 — The novelty is narrow relative to the closest literature

- **Location:** Contributions; Related Work; Table 1; PDF lines 101--124,
  137--209.
- **Exact quote (10 words):** “We do not claim either phenomenon as new.”
- **Problem:** Prior work already reports post-pruning loss/task dissociation,
  recovery trajectories, scratch/init comparisons, iterative recovery, and
  multidimensional evaluation beyond perplexity. The remaining increment is a
  particular OLMo prefix+fresh-tail case plus an incomplete intact branch,
  heterogeneous operating points, two MMLU interfaces, and closed-book QA.
- **Affected claim/norm and why it matters:** This affects excitement and the
  ARR novelty threshold rather than falsifying C1. A main-conference or strong
  Findings contribution normally needs either a new general empirical finding,
  a decisive control, broader replication, or a method. Here the paper itself
  correctly concedes that it supplies a measurement case study and reporting
  discipline.
- **Sufficient remedy:** Either (i) establish generality with preregistered,
  replicated, matched-budget paths across several model families and pruning
  constructions, or (ii) turn the work into a sharper methodological paper by
  proposing and validating a proxy-certification test with operating
  characteristics across many interventions.
- **Severity:** **Major**

## W5 — Exact reproduction and independent result audit are currently blocked

- **Location:** Appendix B.1--B.3; Limitations; PDF lines 541--550, 829--868,
  903--931.
- **Exact quote (10 words):** “those commits are not on the public remote as of this manuscript version.”
- **Problem:** The frozen submission contains LaTeX and figures but not the
  stated anonymous checksum manifest, training/evaluation code, configurations,
  per-item files, or checkpoints. The custom evaluator revisions are
  local-only; the keep14 loader offset is unrecoverable; seeds and hardware
  records are missing.
- **Affected claim/norm and why it matters:** This affects reproducibility and
  independent verification of C1--C5. Hash prefixes and local commit IDs provide
  provenance only to the authors; reviewers cannot execute or inspect the
  harness that generated the content-MMLU, pairing, and closed-book numbers.
- **Sufficient remedy:** Release an anonymous, immutable artifact containing
  all evaluator/training code, exact configs, environment lockfile, complete
  manifests, per-item predictions used in every reported statistic, and
  reconstructable checkpoints or lawful download scripts. Rerun the principal
  experiment with fully recorded seed/loader state.
- **Severity:** **Major**

## W6 — The target scope is too narrow for capability-level rhetoric

- **Location:** Abstract; §3.3; §5.1; §6.3; Limitations; PDF lines 5--9,
  263--285, 319--322, 425--434, 533--538.
- **Exact quote (6 words):** “These evaluations do not exhaust capability”
- **Problem:** The principal evidence is English, mainly one 7B OLMo family,
  one same-source PPL shard, MMLU, and three factual closed-book tasks. There is
  no out-of-domain likelihood, contamination analysis, generative reasoning,
  code/math, instruction following, safety, or deployment efficiency frontier.
  The Qwen result lacks the principal diagnostics.
- **Affected claim/norm and why it matters:** The paper usually says
  “measured knowledge-sensitive evaluations,” which is appropriate, but the
  title/abstract language of “capability recovery” can still invite broader
  interpretation. The evidence cannot establish a general relationship between
  PPL and capability recovery.
- **Sufficient remedy:** Add preregistered out-of-domain PPL and at least one
  generative reasoning/code family, one instruction-following family, and a
  matched second model-family replication; otherwise consistently replace
  capability-level language with the exact evaluated task set.
- **Severity:** **Minor**

## W7 — Historical stopping and compute accounting weaken the depth-side evidence

- **Location:** §3.2; §6.2; Table 3; Limitations; PDF lines 244--250,
  404--424, 517--520.
- **Exact quote (8 words):** “there was no registered common stopping rule.”
- **Problem:** keep8/10/12 stop at different metric-informed checkpoints while
  PPL is still decreasing; equal steps would not equal FLOPs, and realized
  recovery FLOPs/GPU-hours are unavailable.
- **Affected claim/norm and why it matters:** The paper correctly declines a
  depth law, so this does not invalidate C1. It does, however, prevent the
  appendix ladder from answering whether recovery rate or endpoint changes
  systematically with retained depth, despite occupying substantial empirical
  space.
- **Sufficient remedy:** Predeclare a common token/FLOP budget and stopping
  schedule, record realized compute, evaluate all depths at common checkpoints,
  and replicate enough runs to estimate depth-by-budget effects.
- **Severity:** **Minor**

## W8 — One metric description is internally inconsistent

- **Location:** Appendix Table 13 versus Appendix B.2/Table 18.
- **Exact quote (8 words):** “divides candidate log-likelihood by continuation character length”
- **Problem:** Table 13 says normalization uses continuation **character**
  length, whereas §B.2 and Table 18 say continuation **token** count. These are
  different scoring rules and can materially change rankings on unequal-length
  options.
- **Affected claim/norm and why it matters:** This affects reproducibility and
  interpretation of the BoolQ/CSQA/SIQA sensitivity table, though not the
  headline MMLU result.
- **Sufficient remedy:** State the implemented denominator consistently,
  identify the exact code path, and regenerate Table 13 if the caption is not
  merely a typo.
- **Severity:** **Minor**

# Questions That Could Change the Score

1. Do independent keep14 reruns with recorded seeds reproduce a large final
   base gap while PPL improves, and what is the between-run distribution?
2. What happens to full32 at matched 128k/153.5k/200k or matched token/FLOP
   budgets? A stable intact trajectory would materially strengthen the causal
   interpretation of the case.
3. Which one or two factors explain the ShortGPT advantage? Even a small,
   carefully chosen matched-control matrix could substantially increase the
   paper's scientific value.
4. Can the authors provide an anonymous executable artifact containing the
   local evaluator commits, manifests, per-item predictions, and exact configs?
5. Is Table 13 normalized by characters or tokenizer tokens?
6. Was benchmark contamination checked for the DCLM training portion, especially
   for MMLU and the three closed-book datasets?

# Non-Scoring Suggestions and Typos

- Table 13 should use “token length” if that is the implementation.
- The main paper would benefit from one sentence explaining why the principal
  keep14 architecture appends two fresh blocks rather than retaining 16
  pretrained blocks; this is currently historical construction rather than a
  motivated design choice.
- Consider reporting relative as well as absolute PPL change, since the
  128k→200k improvement is small (0.265 absolute, about 2.45% relative).
- “core6” in Table 4 is not defined in that table.
- Table 1's checkmark coding is necessarily coarse; footnotes with exact
  checkpoints/evaluations for the closest two works would be more informative.
- The very large blank area on PDF page 7 is stylistically awkward but not a
  substantive violation.

# Quantitative and Formula Checks

- **Abstract/main-text numbers checked (more than five):**
  10.826→10.561 PPL; .3012→.3191 MMLU; base .6053; PopQA
  .1415/.2571; TriviaQA .2940/.6355; NQ-open .0598/.2050; full32
  25k/6.6B; principal 200k/52.4B; random .2470 letter/.3598 content;
  ShortGPT 9.780/.4739. All match the rendered tables after stated rounding.
- **Token budget:** \(200{,}000\times128\times2048=52.4288\)B and
  \(25{,}000\times128\times2048=6.5536\)B, consistent with 52.4B/6.6B.
- **PPL merge:** \(\exp(\sum_g \mathrm{NLL}_g/\sum_g n_g)\) is the correct
  token-weighted aggregation, unlike averaging shard perplexities.
- **Chance-adjusted recovery:** keep14 MMLU is
  \(100(.3191-.25)/(.6053-.25)=19.45\%\); ShortGPT is 63.02%; random is
  -1.10%. These match Tables 9--10 after rounding.
- **Marginal MMLU SE:** \(\sqrt{p(1-p)/14042}\) reproduces Table 11.
- **Boundary cases:** The paper explicitly handles negative chance-adjusted
  recovery, missing ShortGPT QA cells, differing rerun aggregates, incomplete
  horizons, and fixed-checkpoint versus run-level uncertainty.

# All Figures and Tables Audit

- **Figure 1:** Values and annotations match Tables 2/12/15; communicates the
  main result and confounds clearly. No causal overclaim in caption.
- **Figure 2:** Matches 1B Tables 6--7; correctly labeled qualitative context.
- **Tables 1--2:** Positioning is broadly fair; Table 2 is especially strong on
  protocol disclosure. Table 1 is necessarily coarse and was checked against
  the closest papers where accessible.
- **Tables 3--8:** Checkpoint inventory, ShortGPT selection, keep8 trajectory,
  1B trajectory, and Qwen endpoint are internally consistent. Historical
  stopping limits interpretation.
- **Tables 9--10:** Downstream and chance-adjusted summaries are numerically
  consistent; descriptive family labels should not be read as validated
  taxonomies.
- **Tables 11--15:** Fixed-item uncertainty and interface controls are useful
  and correctly caveated. Table 13 has the character/token inconsistency noted
  in W8.
- **Tables 16--19:** Closed-book, broad-group, evaluation-suite, and complete
  57-subject tables are legible and internally coherent. Table 19 is descriptive
  only; the paper correctly avoids multiplicity-adjusted subject claims.

# Citation Audit

I audited all **33/33 actually cited entries in `main.bbl`**. “Verified” means
that title/authors/year and a primary identifier or official venue record were
matched. “Metadata error” means the work exists but the bibliography metadata
is materially inaccurate or incomplete. Network failure is never treated as
“Not found.”

| Key | Status | Audit note |
|---|---|---|
| benchmarktargets | Verified | ACL Anthology DOI/title/year matched. |
| linearpatch | Verified | arXiv 2505.24680 title/authors/date matched; NeurIPS venue metadata not independently checked. |
| prunecomp | Verified | arXiv 2507.18212 and AAAI 2026 work matched. |
| deng2025drpruning | Verified | ACL 2025 DOI 10.18653/v1/2025.acl-long.1414 matched. |
| gromov2024unreasonable | Verified | arXiv 2403.17887 and ICLR 2025 identity matched. |
| answerorder | Verified | arXiv 2406.19470 title/authors matched. |
| paser | Verified | arXiv 2502.12594 title/authors/date matched. |
| hendrycks2021mmlu | Verified | ICLR 2021 work matched. |
| jaiswal2024truth | Verified | arXiv 2310.01382/ICLR identity matched. |
| joshi2017triviaqa | Verified | ACL DOI 10.18653/v1/P17-1147 matched. |
| shortenedllama | Verified | arXiv 2402.02834 matched. |
| calibration2026 | Verified | arXiv 2604.24938, first posted 2026-04-27, matched. |
| kwiatkowski2019natural | Verified | TACL DOI 10.1162/tacl_a_00276 matched. |
| lu2024reassessing | Verified | arXiv 2411.15558 matched. |
| mallen2023popqa | Verified | ACL DOI 10.18653/v1/2023.acl-long.546 matched. |
| fragileknowledge | Verified | arXiv 2512.22671/title/author/year matched; alternate TechRxiv indexing does not alter the cited work identity. |
| men2024shortgpt | Verified | Findings ACL 2025 DOI/title/pages matched. |
| muralidharan2024compact | Verified | arXiv 2407.14679 and NeurIPS work matched. |
| costcompression | Verified | Findings EMNLP DOI matched. |
| olmo2 | Verified | arXiv 2501.00656/title/authors matched; the record uses the 2025 identifier/citation year. |
| decisioncollapse | Verified | arXiv 2605.07271, first posted 2026-05-08, matched. |
| siddiqui2024deeper | Verified | arXiv 2407.16286 matched. |
| song2024sleb | Verified | arXiv 2402.09025/ICML identity matched. |
| minitron | Verified | arXiv 2408.11796/title/year/authors matched; `main.bbl` uses bibliography-style author-list truncation. |
| slimqwen | Verified | arXiv 2605.08738, first posted 2026-05-09, matched. |
| myanswerisc | Verified | Findings ACL DOI matched. |
| iterabre | Verified | arXiv 2503.06291 matched. |
| xia2024sheared | Verified | arXiv 2310.06694/ICLR 2024 work matched. |
| beyondperplexity | Verified | Findings EMNLP DOI matched. |
| qwen3 | Verified | arXiv 2505.09388 matched. |
| yang2024laco | Verified | Findings EMNLP DOI/pages matched. |
| shortopd | Verified | arXiv 2607.13124, first posted 2026-07-14, matched as contemporaneous work. |
| blockpruner | Verified | Findings ACL 2025 DOI/pages matched. |

## Load-Bearing Citation–Claim Matches

1. **Gromov et al. → post-healing loss/task dissociation:** **Matched.**
   The paper explicitly contrasts post-healing QA collapse with smoother
   autoregressive loss and reports MMLU/BoolQ versus loss behavior.
2. **Shortened LLaMA → CPT curves and scratch/init comparison:** **Matched.**
   The source compares CPT/LoRA and pruned initialization versus training the
   same architecture from scratch.
3. **Minitron → trajectories, task behavior, and initialization choices:**
   **Matched.** Its report includes loss curves, downstream tasks, and random
   initialization/pruning/distillation comparisons.
4. **IteRABRe → iterative recovery with weak/task-dependent MMLU recovery:**
   **Matched.** Its iterative plots report MMLU and show that recovery varies by
   task/model.
5. **Jaiswal et al. → low/nearly preserved perplexity can hide
   knowledge-intensive deficits:** **Matched.** This is a central motivation of
   LLM-KICK.
6. **Beyond Perplexity → compression has divergent safety/downstream effects:**
   **Matched.**
7. **LinearPatch/Prune&Comp → interface/magnitude mismatch and lightweight
   repair:** **Matched** to their abstracts and methods.
8. **SlimQwen/ShortOPD concurrent-work characterization:** **Partially
   matched.** SlimQwen does cover matched-budget scratch comparison and
   progressive training; ShortOPD covers recognition/generation and
   short-to-long recovery. Full-paper detail beyond accessible primary
   metadata is **Unverifiable**.

# Novelty Search Summary

**Cutoff used: 2026-05-04.** Searches covered depth-pruning healing/recovery,
perplexity–task gaps, scratch initialization, MMLU trajectories, OLMo
prefix/fresh-tail constructions, and compression evaluation beyond
perplexity. Work first posted after the cutoff was treated as contemporaneous,
not prior art.

## Closest prior work

1. **Gromov et al., “The Unreasonable Ineffectiveness of the Deeper
   Layers” (2024 preprint/ICLR 2025).** Closest to the loss–task dissociation
   after pruning and healing; already compares QA behavior with autoregressive
   loss.
2. **Shortened LLaMA (2024).** Closest on depth pruning, CPT learning curves,
   retraining alternatives, and pruned-versus-scratch initialization.
3. **IteRABRe (2025).** Closest on iterative removal/recovery trajectories and
   task-dependent, weak MMLU recovery.
4. **Jaiswal et al., “Compressing LLMs: The Truth Is Rarely Pure and Never
   Simple” (2023/ICLR 2024).** Closest on the broader conclusion that
   perplexity does not adequately capture knowledge-intensive compressed-model
   behavior.
5. **“On the Limits of Layer Pruning for Generative Reasoning in Large
   Language Models” (2026-02-02).** Additional pre-cutoff close work not cited
   in the manuscript; it studies post-pruning recovery limits, including very
   long training budgets, and finds substantially weaker recovery for
   generative reasoning than classification.

**Novelty conclusion:** The exact OLMo prefix14+fresh2 path, paired MMLU
interfaces, closed-book recurrence, and explicit operating-point bookkeeping
appear distinctive in combination. The underlying phenomena—PPL/task
dissociation, recovery trajectories, initialization effects, and the need for
multidimensional evaluation—are not new. The novelty is therefore
**incremental and case-study-level**.

**Contemporaneous work:** decision-collapse (2026-05-08) and SlimQwen
(2026-05-09) fall within days after the cutoff; ShortOPD (2026-07-14) is also
later. They should not be used to deny novelty. Full exhaustive web search was
interrupted; any undiscovered or inaccessible item is **Unverifiable**, not
“not found.”

# Limitations, Ethics, and Desk-Reject Risks

- **Page/format:** The PDF has 17 A4 pages: 7 numbered main-content pages,
  followed by one page containing the exact unnumbered **Limitations** and
  Ethical Considerations sections, two reference pages, and seven appendix
  pages. The main empirical narrative fits within 7 pages. I found no clear
  page-limit violation.
- **Exact Limitations section:** Present as `\section*{Limitations}` and
  substantive.
- **Anonymity:** Title page says “Anonymous ACL submission”; author and PDF
  author metadata are empty. No author identity was found. Local commit hashes
  and `origin/main` are awkward but not by themselves identifying.
- **Style:** Uses supplied `acl.sty` with `[review]`, line/page numbers, A4
  geometry, embedded fonts, and ACL bibliography style. No clear style
  violation was found.
- **References/TODOs:** No unresolved `?` references, duplicate labels, missing
  bibliography entries, TODO/TBD/FIXME/placeholder text, attachments,
  JavaScript, or forms were found.
- **Prompt injection/reviewer manipulation:** I scanned source, extracted PDF
  text, figure text, color/font-size commands, and PDF metadata. No reviewer
  instructions, score manipulation, hidden white text, or suspicious
  zero-scale overlays were found. Small Figure 1 labels are visually present,
  not hidden.
- **Ethics:** The section appropriately covers inherited model/data risks,
  deployment caution, energy use, licenses, and lack of new human-subject data.
  No obvious ethics desk-reject issue.
- **Potential desk risk:** The frozen source supplied for review does not
  contain the “accompanying anonymous checksum manifest” referenced in Appendix
  B.1. If ARR requires all claimed supplementary artifacts to be uploaded, this
  should be checked administratively. I do not treat it as an automatic desk
  reject from the PDF alone.

# Scores

## Soundness: 3.0 / 5

The descriptive claims about the realized checkpoints are well supported and
carefully bounded. The score is limited by the lack of independent training
runs, the missing matched-horizon intact control, heterogeneous confounded
operating points, and inaccessible custom-evaluator artifacts. This is not
unsound, but it is not a strong inferential experiment.

## Excitement: 2.5 / 5

The paper is useful as a cautionary, well-documented case, and Figure 1/Table 2
are strong scientific communication. However, the main phenomenon and most
ingredients are established in prior work; the new contribution is a narrow
combination of controls and interfaces on one historical OLMo path.

## Overall: 2.5 / 5

Under the requested calibration, 3.0 corresponds to Findings. I am uncertain
between 2.5 and 3.0, and therefore choose the lower score. The paper is careful
and potentially publishable after substantial empirical strengthening, but the
current single-run, unmatched-control case study does not yet clear my Findings
bar.

## Confidence: 4.0 / 5

I completed two passes of all 17 PDF pages, inspected both figures and all 19
tables, audited the source and all 33 references, checked key arithmetic, and
performed primary-source novelty/citation searches. Confidence is not 4.5/5
because some very recent full papers and venue metadata were only partially
accessible; those items are marked Unverifiable or partial rather than inferred.

## Reproducibility: 2.0 / 5

The written protocol is detailed, and the new appendix substantially improves
paper-level reproducibility. Exact reproduction nevertheless remains blocked
by unavailable seeds, loader offset, compute records, local-only code commits,
and absence of an executable artifact/checkpoints/per-item bundle in the frozen
source.

# Review-Process Self-Check

- [x] Evaluated v5 from scratch; did not use or inherit any earlier review.
- [x] Read the complete main paper and appendix twice.
- [x] Built and audited claims C1--C8 against exact evidence and designed the
      minimum sufficient experiment for each.
- [x] Checked more than five abstract/headline numbers against tables.
- [x] Inspected every rendered figure and table.
- [x] Checked formulas, boundary cases, metrics, sample sizes, seeds/statistics,
      scope, compute, and reproducibility.
- [x] Checked page/Limitations/anonymity/style/references/TODO/injection risks.
- [x] Audited all 33 actually cited `main.bbl` entries and eight load-bearing
      citation–claim matches.
- [x] Ran multiple novelty searches with the 2026-05-04 cutoff and treated
      post-cutoff papers as contemporaneous.
- [x] Marked incomplete network/full-text checks as **Unverifiable**, never
      “Not found.”
- [x] Mechanically grep-checked every weakness quote against the frozen source.
- [x] No weakness asserts absence of an item that appears elsewhere in the
      frozen main paper or appendix.
