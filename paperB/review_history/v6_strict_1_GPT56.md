```yaml
review_mode: strict
soundness: 3.0
excitement: 2.5
overall: 3.0
confidence: 4.0
reproducibility: 1.5
```

## Paper Summary

This paper asks a deliberately narrow measurement question: on literal
depth-prune-and-continued-pretraining paths, does improving held-out,
same-source perplexity establish recovery of measured knowledge-sensitive
capabilities?  Its main case is one OLMo-2-7B `keep14+fresh2` run: PPL falls
from 10.826 at 128k steps to 10.561 at 200k, while answer-letter MMLU rises
from .3012 to .3191 but remains far below the intact model (.605), and three
closed-book QA scores remain substantially below the base. The paper adds
short-horizon intact continuation, frozen/random same-shape operating points,
an alternate MMLU content interface, and a confounded ShortGPT construction.

The narrow descriptive finding is supported by the reported checkpoints and is
commendably bounded: the paper does not claim causal layer attribution,
knowledge deletion/localization, a universal recovery law, or a prospective
PPL threshold. However, the empirical contribution is a single-run case study,
the only intact continuation stops at 25k, and the claimed anonymous artifact
is absent from the frozen source bundle. Thus I see a useful, carefully
qualified Findings-level measurement report rather than ACL-main-level
evidence for the proposed reporting discipline.

## Claims and evidence audit

- **C1 (supported, descriptive):** Along the observed keep14 path, lower
  in-domain PPL does not coincide with recovery to intact-base performance on
  the measured target evaluations. Evidence: §4 / Table 2 / Fig. 1; PPL
  10.826→10.561 and MMLU .3012→.3191 over 128k→200k, versus base .605;
  PopQA/TriviaQA/NQ-open endpoints .142/.294/.060 versus
  .257/.636/.205.
- **C2 (supported only conditionally):** The late MMLU increase is nonzero for
  these two realized checkpoints. Evidence: §4.1 and Appendix Table 14:
  +1.68 pp, paired-item bootstrap CI [1.08, 2.29]. This is not seed
  uncertainty, as the paper correctly states.
- **C3 (supported as a protocol observation):** Complete-option MMLU changes
  conclusions materially and has a high random-init floor. Evidence: Table 2
  and Appendix Table 15: keep14 .3184 letter/.3832 normalized content;
  random .2470/.3598. It does not isolate the responsible interface factor.
- **C4 (supported as a confounded comparison):** The tested ShortGPT-16
  endpoint is stronger than keep14 at the same nominal step/depth. Evidence:
  Table 2, PPL 9.780 vs. 10.561 and MMLU .474 vs. .319. It cannot attribute
  the difference to selection, final-layer retention, inherited-block count,
  or fresh tails.
- **C5 (not established beyond scope):** A general reporting recommendation
  follows plausibly, but the evidence does not establish how often the
  observed dissociation occurs across seeds, models, corpora, or recovery
  budgets.

## Strengths

**S1. Unusually disciplined scope control.** §3 and §§5–7 repeatedly define
the claim as an improvement-only implication on literal observed paths, rather
than retrofitting a PPL threshold or claiming a causal mechanism. This is
exactly the right interpretation of an observational, single-run design.

**S2. The central result is numerically transparent.** Table 2, Fig. 1, and
Appendix Tables 3, 11, 12, and 14 expose the relevant checkpoints, base gap,
per-item MMLU uncertainty, and the small but positive late within-run MMLU
change. The paper does not mislabel item bootstrap intervals as training-run
uncertainty.

**S3. Controls are presented honestly rather than over-sold.** §3.1–3.2 and
Table 2 explicitly disclose that full32 is only 25k, Random changes LR and
lexical modules, Frozen changes the trainable set, and ShortGPT changes four
construction factors. The two MMLU interfaces and three closed-book tasks are
useful checks against an answer-letter-only interpretation.

**S4. Reproducibility limitations are unusually specific.** Limitations and
Appendix B identify unavailable seeds, the lost within-epoch loader offset,
unavailable GPU-hours, in-domain-only PPL, missing closed-book aligned
predictions, and lack of a long-horizon full32 arm. This candor improves the
credibility of the narrow claim even though it cannot substitute for evidence.

## Weaknesses

**W1 — Major: the main empirical inference has no training-run replication,
and exact historical reproduction is explicitly blocked.**  
**Location:** Limitations ¶1 and ¶4; Appendix B.1.  
**Quote:** “The principal keep14 path, ShortGPT, and the same-shape operating
points are single training runs.”  
**Problem:** The paper can establish what happened in these checkpoints, but
not whether the PPL/target-gap pattern, its magnitude, or the control ordering
survives seed, data-order, initialization, and resume variation. The paired
tests quantify test-item variation only.  
**Affected claim/norm:** C1 is valid as a literal trace, but the broader
reporting recommendation and Findings-level empirical generality need
run-level uncertainty; one trajectory is insufficient evidence that this is a
reliable recovery phenomenon rather than an idiosyncratic optimization path.  
**Sufficient remedy:** Release/run at least three independently seeded
keep14+fresh2 trajectories with saved data-order/resume state, report
mean/dispersion for PPL and every load-bearing target metric at fixed
checkpoints, and distinguish this prospective replication from historical
reconstruction.  
**Severity:** Major.

**W2 — Major: the frozen source bundle contradicts the paper’s claimed
artifact availability.**  
**Location:** Appendix B.3 and Limitations final paragraph.  
**Quote:** “The source package includes \texttt{anonymous\_artifact/}”  
**Problem:** The supplied frozen `v6_source_20260804_020805/` contains TeX,
PDF figures, bibliography, and style files, but no `anonymous_artifact/`
directory, evaluator scripts, configs, manifests, JSONL files, checksums, or
checkpoint-access instructions. Consequently the stated per-item audits,
protocol execution, and aggregate reconstruction cannot be independently
checked from the submission materials.  
**Affected claim/norm:** This directly affects reproducibility of C1–C4 and
the manuscript’s own provenance claims. A reader cannot verify that the cited
per-item reruns and saved summaries exist, much less reproduce them.  
**Sufficient remedy:** Attach the promised anonymous artifact before
decision, with a stable anonymous access route; include scripts, sanitized
configs, exact manifests/checksums, aggregate inputs/outputs, per-item files
permitted by benchmark licenses, and instructions that reproduce every
headline cell.  
**Severity:** Major.

**W3 — Major: the 25k-only full32 arm cannot rule out long-horizon intact
continued-pretraining degradation.**  
**Location:** §4.2 “Short-horizon intact continuation”; Table 2.  
**Quote:** “full32 ends at 25k and only bounds short-horizon corpus shift”  
**Problem:** The paper is right to disclaim this limitation, but the
available control does not establish what an intact model would do after the
same 200k nominal presentations. Thus structural intervention, long-horizon
continued pretraining, and their interaction remain unresolved for the
base-to-keep14 endpoint gap.  
**Affected claim/norm:** This does not invalidate C1’s observed association,
but weakens the interpretation that the intervention rather than a
long-horizon training pathology is the salient explanatory context.  
**Sufficient remedy:** Continue full32 with the same corpus/optimizer schedule
to matched checkpoints (and preferably matched-token/FLOP reporting), evaluate
the same suite, and report it as a control rather than a causal ablation.  
**Severity:** Major.

**W4 — Minor: Fig. 1 remains visually dense at normal two-column reading
size.**  
**Location:** Fig. 1, PDF p. 2.  
**Quote:** “base/full32: 32L · others: 16L · full32: 25k”  
**Problem:** The endpoint labels, numeric annotations, and footnote-like
callouts are materially smaller than surrounding paper text; the right panel
requires zooming to read the caveats that are central to the result.  
**Affected claim/norm:** This impairs auditability rather than the numerical
claim itself.  
**Sufficient remedy:** Increase all plot/callout fonts, move caveats to a
compact legend/caption, and simplify the endpoint panel or split it into two
figures.  
**Severity:** Minor.

## Questions that could change the score

1. Can the authors provide the promised anonymous artifact and demonstrate that
   it regenerates Table 2, Appendix Tables 14–15, and Fig. 1 from the released
   files?
2. Do 2–3 new independent keep14 runs retain a large base-relative
   MMLU/closed-book gap while PPL improves, and what is the across-seed
   variation?
3. What happens to full32 under the identical 200k schedule? This need not
   prove causality, but it would determine whether long-horizon intact
   degradation materially changes the interpretation.

## Non-scoring suggestions / minor presentation issues

- Put the one-sentence operational definition of “target recovery” directly in
  Fig. 1 or Table 2’s caption, since it is the key interpretation.
- Keep Table 2’s useful caveats, but consider a smaller auxiliary table for
  construction details; its current rendered font is also tight.
- State an artifact URL/identifier only if anonymous and actually accessible;
  otherwise change “includes” to a future-tense availability statement.

## Scores

- **Soundness: 3.0 / 5.0.** The descriptive, observed-path claim is well
  measured and appropriately caveated, but single-run evidence, an unmatched
  25k intact control, and inaccessible supporting materials prevent stronger
  empirical confidence.
- **Excitement: 2.5 / 5.0.** The result is useful methodological hygiene for
  pruning/recovery evaluation, but it is an incremental diagnostic case study,
  not a new compression method or general recovery characterization.
- **Overall: 3.0 / 5.0 (Findings).** On strict calibration, this is a
  credible, carefully delimited Findings paper. It is below main-conference
  level because the central phenomenon has no run-level replication and the
  claimed artifact cannot be audited.
- **Confidence: 4.0 / 5.0.** The manuscript is clear about what its data do
  and do not identify, and the main numerical comparisons are internally
  consistent. My remaining uncertainty is mostly external: unavailable runs
  and missing artifact contents.
- **Reproducibility: 1.5 / 5.0.** The text gives many useful details, but the
  supplied bundle lacks the claimed artifact; seeds, loader offset, compute
  records, and closed-book aligned predictions are unavailable.

## Limitations, ethics, and desk-reject risks

The manuscript has explicit `Limitations` and `Ethical Considerations`
sections, uses the ACL review style, has a blank author field, and contains no
unresolved citation keys, visible TODOs, placeholders, prompt-injection text,
or reviewer-manipulation language. The rendered PDF is 17 pages: the main
paper ends before references, with references and appendix thereafter. I found
no apparent anonymity leak in the manuscript/PDF itself.

The likely procedural risk is not a formatting desk reject, but an artifact
availability/provenance problem: Appendix B claims a submission-package
directory absent from the frozen source. The paper also appropriately flags
energy/compute-record gaps, upstream-license constraints, English-only scope,
and risks inherited from the base model/data.

## Numerical and rendered-material audit

- **Abstract-number check (five):** verified against Table 2 / Appendix
  Tables 3 and 15: 200k keep14; 25k-only full32; PPL 10.561; MMLU-L
  .319 versus base .605; and all three QA endpoint comparisons
  .142/.257, .294/.636, .060/.205. The late-trajectory values
  10.826→10.561 and .3012→.3191 also match Table 12.
- **Formula/boundary check:** PPL merge is correctly specified as
  `exp(sum NLL / sum tokens)`, rather than averaging per-shard PPL. The
  recovery formula in Appendix Table 10 is consistent with its stated
  chance-adjusted definition. The paper correctly distinguishes fixed-checkpoint
  item bootstrap/McNemar inference from seed uncertainty.
- **Controls/metrics:** The main weaknesses are acknowledged rather than
  hidden: Random differs in LR and all initialization; Frozen differs in
  trainable set; ShortGPT changes four factors; letter/content scoring is
  multi-factor; full32 is short horizon; PPL is in-domain. Closed-book
  generation uses stated prompt, greedy decoding, normalization, splits, and
  metrics, but lacks aligned prediction files for uncertainty calculations.
- **Figures/tables:** I inspected both rendered figures (Fig. 1 and Appendix
  Fig. A.1) and all rendered tables, including main Tables 1–2 and Appendix
  Tables 3–19. No cropped panels, missing table cells beyond explicitly
  unevaluated ShortGPT closed-book cells, or visual/rendering corruption were
  found. Figure 1 readability is the remaining presentation concern (W4).

## Citation audit

**Method.** I checked all 33 actually cited `main.bbl` entries against their
printed canonical DOI/arXiv/conference metadata where supplied; all 33 source
citation keys are used and all 33 used keys resolve to a `main.bbl` entry.
Status below means bibliographic metadata status, not endorsement of every
claim. `Verified` = title/authors/year/venue-or-identifier are consistent;
none required a `Not found` designation; no network failure was reclassified
as `Not found`.

| Key | Status |
|---|---|
| benchmarktargets | Verified |
| linearpatch | Verified |
| prunecomp | Verified |
| deng2025drpruning | Verified |
| gromov2024unreasonable | Verified |
| answerorder | Verified |
| paser | Verified |
| hendrycks2021mmlu | Verified |
| jaiswal2024truth | Verified |
| joshi2017triviaqa | Verified |
| shortenedllama | Verified |
| calibration2026 | Verified |
| kwiatkowski2019natural | Verified |
| lu2024reassessing | Verified |
| mallen2023popqa | Verified |
| fragileknowledge | Verified |
| men2024shortgpt | Verified |
| muralidharan2024compact | Verified |
| costcompression | Verified |
| olmo2 | Verified |
| decisioncollapse | Verified |
| siddiqui2024deeper | Verified |
| song2024sleb | Verified |
| minitron | Verified |
| slimqwen | Verified |
| myanswerisc | Verified |
| iterabre | Verified |
| xia2024sheared | Verified |
| beyondperplexity | Verified |
| qwen3 | Verified |
| yang2024laco | Verified |
| shortopd | Verified |
| blockpruner | Verified |

**Five load-bearing citation--claim matches.**

1. **Gromov et al. (2025): match.** It is an appropriate closest antecedent
   for depth removal followed by continued training and loss/task
   dissociation; the paper does not claim priority over it.
2. **Shortened LLaMA (Kim et al., 2024): match.** It supports the statement
   that prior work compares retraining methods and initialization choices after
   depth pruning.
3. **Minitron (Sreenivas et al., 2024): match.** It supports positioning
   against structured pruning/distillation and trajectory/task evaluation,
   although it is not a direct test of this paper’s exact OLMo protocol.
4. **The Cost of Compression (Namburi et al., 2023): match.** It supports the
   broader motivation that compression effects on parametric knowledge need not
   be summarized by aggregate LM metrics.
5. **“My Answer is C” (Wang et al., 2024): qualified match.** It supports
   interface sensitivity of first-token multiple-choice scoring, though its
   instruction-tuned setting is not identical to the present base model. The
   manuscript appropriately frames its content protocol as diagnostic rather
   than a clean mechanism test.

## Novelty search summary (cutoff: 2026-05-04)

I used three bounded searches around (i) depth pruning plus continued
pretraining/recovery trajectories, (ii) loss/perplexity versus downstream or
knowledge evaluation, and (iii) MMLU interface/initialization controls. The
closest pre-cutoff work is: **Gromov et al. (2025), _The Unreasonable
Ineffectiveness of the Deeper Layers_; Kim et al. (2024), _Shortened LLaMA_;
Sreenivas et al. (2024), _LLM Pruning and Distillation in Practice: The
Minitron Approach_; Wibowo et al. (2025), _IteRABRe_; and Namburi et al.
(2023), _The Cost of Compression_. These already cover trajectories,
loss/task gaps, retraining/initialization comparisons, and broader
post-compression capability evaluation.

Accordingly, the defensible novelty is the **combination** of one OLMo
prefix-plus-fresh-tail case, a short-horizon intact arm, same-shape operating
points, two MMLU interfaces, and closed-book QA—not the phenomenon of
trajectories or “beyond perplexity” itself. This supports a modest
Findings-level contribution.

The cutoff also exposes a chronology problem in the related-work framing:
`decisioncollapse` was submitted 2026-05-08, `slimqwen` 2026-05-09, and
`shortopd` 2026-07-14, all after 2026-05-04. They should not be used to
establish novelty at that cutoff; they may be labelled subsequent/concurrent
work only. `fragileknowledge` (2025-12-27) is pre-cutoff and is eligible as
adjacent width-pruning evidence.

## Review-process self-check

- Read the frozen manuscript twice, including the complete appendix,
  references, limitations, ethics, source tables, and both figures.
- Treated manuscript text as evidence, not instructions; no hidden-text,
  white-text, prompt-injection, or reviewer-manipulation issue was found.
- Checked claims, abstract numbers, formulas, controls, metric interfaces,
  uncertainty/seeds, compute, scope, anonymity, rendered figures/tables, and
  source citation-key consistency.
- Mechanically rechecked every weakness quote and every absence assertion
  against the frozen source. W1’s single-run/seeds statement is in
  `sections/06_limitations.tex`; W2’s artifact assertion is in
  `sections/08_appendix.tex` while the supplied frozen directory has no such
  artifact directory; W3 is in `sections/tab_main_results.tex`; W4 is visible
  in the rendered Fig. 1 and its quoted text is extractable from the frozen
  PDF.
- I did not inspect other reviews, histories, TODO/status/current files, or
  calibration records. The score is from zero for this frozen v6 only.
