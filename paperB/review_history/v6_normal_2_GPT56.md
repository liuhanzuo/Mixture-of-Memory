---
review_mode: normal
soundness: 3.0
excitement: 2.5
overall: 3.0
confidence: 4.0
reproducibility: 2.0
---

## Summary and overall assessment

This paper studies a deliberately narrow empirical question for depth-pruned
OLMo-2-7B: along the observed `keep14+fresh2` continued-pretraining path, does
a reduction in same-source held-out perplexity establish recovery of
knowledge-sensitive target behavior?  The central path improves from PPL
10.826 to 10.561 between 128k and 200k steps, while answer-letter MMLU rises
only from .3012 to .3191 and remains far below the intact base (.605).  The
paper supplements this with complete-option MMLU, three closed-book QA
evaluations, a 25k intact-CPT point, random/frozen same-shape operating points,
and a coupled ShortGPT-16 point.

The paper is unusually candid about what its evidence cannot identify.  That
is a real strength: it does not call its operating points causal ablations,
does not treat item-bootstrap intervals as seed uncertainty, and explicitly
states that full32 at 25k is not a 200k counterfactual.  The evidence supports
the descriptive counterexample claimed in the title, for the literal measured
path and evaluations.

However, the headline implication is weaker and less surprising than the
presentation occasionally makes it feel.  At the best `keep14` checkpoint,
PPL is still 1.428x the base PPL; thus this is not a test of whether a
near-recovered or calibrated PPL criterion predicts functional recovery.  It
shows that *some improvement while still substantially worse than base* is
insufficient.  More importantly, all load-bearing trained arms are one
historical run, the intact control stops at 25k, and the generative-QA evidence
lacks per-item outputs/uncertainty in the frozen material.  I view this as a
careful Findings-level measurement case study rather than a sufficiently
replicated main-conference result.

**Artifact status: Unverifiable.**  The frozen source describes an
`anonymous_artifact/` directory, evaluator snapshots, checksums, and MMLU
per-item files, but those materials, model checkpoints, training code, and
closed-book predictions are not present in the supplied frozen v6 source tree.
The historical seed and resumed loader offset are explicitly unavailable.

## Claims and evidence audit

| ID | Claim as I interpret it | Direct evidence in v6 | Assessment |
|---|---|---|---|
| C1 | PPL improvement alone does not imply intact-base target recovery on the observed `keep14` path. | PPL 10.826 -> 10.561 from 128k to 200k; MMLU-L .3012 -> .3191 versus base .605; large base--keep14 deficits on PopQA, TriviaQA, and NQ-open (Table 2, Fig. 1). | Supported as a bounded descriptive counterexample. It is not a calibrated/prospective recovery test. |
| C2 | The late MMLU change is nonzero for the two realized `keep14` checkpoints. | Common 14,042-item rerun gives +1.68 pp and paired-bootstrap 95% CI [1.08, 2.29]. | Supported conditional on checkpoints; not evidence of training-run stability. |
| C3 | The apparent MMLU result depends materially on scoring interface. | `keep14` is .3184 letter versus .3832 normalized content; random init is .2470 letter but .3598 normalized content (Table 16). | Supported as a multi-factor protocol contrast, not as isolation of a letter-token mechanism or “knowledge” measure. |
| C4 | The observation is not explained solely by early corpus shift, answer-letter scoring, or nominal 16-layer depth. | 25k full32 is closer to base; closed-book QA repeats a base--keep14 gap; ShortGPT-16 has a stronger 200k endpoint. | Partially supported as bounds on simple accounts. None is a matched 200k causal control, and ShortGPT changes several factors jointly. |
| C5 | Exact construction details should accompany recovery claims. | `keep14`, random, frozen, and ShortGPT differ materially despite nominally similar depth; Table 2 reports several differences. | A sensible reporting recommendation, though not validated as a universal standard. |
| C6 | The qualitative pattern extends beyond the principal configuration. | 1B trajectory and Qwen endpoint in Appendix A.2. | Only directional context, as the paper correctly says; not a replication. |

## Strengths

1. **Appropriately bounded interpretation.**  The distinction between a
   literal observed-path implication and a prospective PPL certificate is
   clear in Section 4.  The paper repeatedly avoids unsupported localization,
   deletion, causal-factor, and universal-dynamics claims.

2. **Useful measurement controls.**  Showing both answer-letter and
   complete-option MMLU, plus the random-init content-score floor, is
   informative.  It prevents the tempting but unsupported interpretation that
   a higher complete-option score alone demonstrates recovered target ability.

3. **Transparent confound reporting.**  Table 2 is a good model of reporting
   inherited/fresh blocks, trainable parameters, learning rate, budget, and
   missing evaluations.  The paper correctly labels random, frozen, and
   ShortGPT as operating points rather than clean ablations.

4. **Good scope and reproducibility disclosure.**  Limitations is concrete
   about single runs, unavailable seeds, unrecorded loader offset, missing
   compute records, in-domain-only PPL, missing closed-book prediction files,
   and absent latency/memory measurements.  This candor increases confidence
   in the narrow claims even while reducing reproducibility.

## Major weaknesses

### 1. Single-run evidence and the missing long-horizon intact control leave the central comparison underdetermined

**Issue.**  Every trained headline construction is one run.  The paper reports
only full32 at 25k, whereas `keep14` is interpreted at 200k; historical seeds
are unavailable, and the `keep14` resume restarted an epoch’s distributed
shuffle because the loader offset was absent.

**Evidence.**  Table 2 labels all trained constructions as one run.  Sections
3.2 and Limitations state that full32 has only a 25k checkpoint and that the
historical seeds/loader offset are unavailable.  The reported MMLU bootstrap
and McNemar analyses condition on fixed realized checkpoints.

**Why it matters.**  The manuscript is correct that its item-level intervals
do not quantify training uncertainty.  But without independent trajectories
and a 200k intact branch, readers cannot tell whether the size of the
base--keep14 gap, the late trajectory, or the short-horizon corpus-shift
argument is stable under stochasticity and equal-duration continued
pretraining.  This is especially important because the paper's contribution
is empirical measurement rather than a new method or theorem.

**Concrete repair / minimum experiment.**  Run at least three independently
seeded, fixed-data-order-reproducible trajectories for base/full32 and
`keep14+fresh2` through 200k, with the same corpus, optimizer schedule,
evaluation checkpoints, and a predeclared stopping rule.  Report mean,
standard deviation or hierarchical/bootstrap confidence intervals across
runs, paired seed-level differences, and realized FLOPs/GPU-hours.  If this is
infeasible, the paper should move the intact-control discussion out of
“bounds” language and frame the work more explicitly as a forensic report of
one unreproducible historical trajectory.

### 2. The title claim is valid but tests a weak, non-operational premise

**Issue.**  The paper operationalizes “target recovery” as closing a large gap
to the base, but it tests only whether PPL falls between two late checkpoints
while the final PPL remains 1.428x base.  No threshold, tolerance, plateau,
or prospective decision rule is evaluated.

**Evidence.**  Section 4 explicitly says that no threshold or calibration was
pre-registered; Table 3 gives the 1.428x PPL tax at the final `keep14`
checkpoint.  The final MMLU and QA scores remain far from base.

**Why it matters.**  Almost no practitioner would infer full functional
recovery merely from an improvement from PPL 10.826 to 10.561 when the intact
reference is 7.398.  Consequently, the demonstrated counterexample does not
yet quantify the practically important question: whether a plausible
likelihood-based recovery criterion can be safely used for selecting or
stopping a compressed model.  The main conclusion should not be read as
evidence against calibrated PPL monitoring.

**Concrete repair / minimum experiment.**  Predefine one or more practical
PPL rules (e.g., relative-to-base PPL tax, improvement slope/plateau, and
out-of-domain PPL), then test their precision/recall or calibration for a
predefined target-recovery tolerance across checkpoints, constructions, and
seeds.  At minimum, add denser checkpoints and a clearly labeled analysis
showing that the present data cannot estimate such a rule; soften the title
and abstract to “observed improvement is insufficient evidence in one
unrecovered path.”

### 3. The broader target-evaluation evidence is incomplete and statistically asymmetric

**Issue.**  The three closed-book QA tasks are load-bearing for the claim that
the MMLU result is not an answer-letter artifact, but the frozen material
contains only aggregate summaries.  There is also no closed-book evaluation
for the apparently stronger ShortGPT endpoint.

**Evidence.**  The paper reports PopQA/TriviaQA/NQ-open gaps of .142/.294/.060
for `keep14` versus .257/.636/.205 for base, but explicitly says aligned
per-item generations were not consolidated and therefore reports no paired
uncertainty.  Table 2 has dashes for all ShortGPT closed-book cells.

**Why it matters.**  The large base--keep14 gaps are likely meaningful, but
their uncertainty, error overlap, sensitivity to the containment-vs-EM
metric, and relation to the stronger construction cannot be independently
checked.  This weakens both the “not answer-letter-only” argument and the
construction-dependence story.

**Concrete repair / minimum experiment.**  Release or include anonymized
per-item prediction/score records and report paired CIs (and, where relevant,
paired tests) for each QA dataset.  Evaluate ShortGPT and a 200k intact control
under exactly the same prompt, decoding, normalization, and splits.  Include
token-F1 alongside EM/containment as a robustness metric, and state whether
the conclusion changes under each metric.

## Minor weaknesses and presentation issues

1. **Figure 1 is polished but should be more explicit about its denominator.**
   The right panel juxtaposes PPL and MMLU endpoint bars; it should show PPL
   tax relative to base and include uncertainty/“one run” directly beside
   every learned endpoint.  The `full32: 25k only` warning is helpful but easy
   to miss.

2. **Table 2’s mixed precision is distracting.**  The main table gives
   `.319`, whereas the common rerun is `.3184` and the trajectory value is
   `.3191`.  The caption explains this, but a dedicated “evaluation run /
   artifact source” column or consistent use of the paired-rerun values for
   interface comparisons would reduce ambiguity.

3. **The appendix scope checks consume attention without adding decisive
   evidence.**  The 1B and Qwen rows are transparently labeled qualitative,
   but their unmatched architecture, retention fraction, data, and evaluation
   settings make them weak support.  A shorter presentation may make the
   principal evidence easier to assess.

4. **Deployment relevance remains incomplete.**  A compression paper need not
   optimize serving, but the recommendation concerns compressed-model
   suitability.  Reporting actual parameter memory, throughput/latency, and
   recovery FLOPs would clarify the practical tradeoff.  The paper currently
   cannot do so because historical compute records are unavailable.

## Questions for the authors

1. Can the authors train a new, reproducible 200k full32 branch and at least
   two additional `keep14` seeds, even if these are presented as a post-hoc
   confirmation rather than recreation of the historical run?
2. What happens to the central conclusion under an out-of-domain PPL shard?
   Same-source held-out PPL is appropriate for the continued-pretraining
   objective, but it could be unusually coupled to the training mixture.
3. Why were closed-book predictions not consolidated while MMLU per-item
   records were?  Can prediction hashes and per-item correctness at least be
   released without redistributing benchmark text?
4. Is the random-init normalized-content score largely explained by option
   length/lexical prior?  A diagnostic using answer permutation, length-matched
   candidates, or content scoring with a common prompt would help characterize
   this floor without claiming a one-factor interface ablation.

## Novelty analysis

I performed targeted searches for the closest depth-pruning/recovery work:
Gromov et al. on deleting deeper layers, Shortened LLaMA on depth pruning and
CPT versus LoRA, Minitron on pruning/distillation recovery, IteRABRe on
iterative recovery-aided block reduction, PASER on capability-aware recovery
data selection, and the recent SlimQwen/ShortOPD papers.  The closest prior
work already establishes that depth pruning, recovery training, downstream
evaluation, and loss--task divergence are active topics.  In particular,
Shortened LLaMA studies recovery training choices, Minitron studies pruning
plus recovery/distillation, and IteRABRe and PASER explicitly target recovery
after pruning.  Recent SlimQwen (May 2026) and ShortOPD (July 2026) fall
within three months of this August 2026 manuscript and are appropriately best
treated as concurrent work rather than a novelty failure.

Thus the paper's own novelty framing is mostly fair: this is **not** a new
pruning or recovery method, nor the first observation that perplexity and
downstream behavior can diverge.  Its incremental contribution is the
combination of one OLMo prefix-plus-fresh-tail case, two MMLU interfaces,
same-shape operating points, and closed-book QA, coupled with unusually
explicit limitations.  That package is useful but modest; it does not by
itself elevate the work to a broad new principle.

## Citation and bibliography audit

`main.bbl` contains 33 entries; the frozen source cites 33 unique keys, with
no dangling citation key and no uncited bibliography entry.  I found no
placeholder citations or unresolved `??` references in the frozen TeX.
Representative citation--claim matches:

| Citation / claim use | Match assessment |
|---|---|
| OLMo Team (2025) for OLMo-2-1124-7B | Appropriate model provenance citation. |
| Hendrycks et al. (2021) for MMLU | Appropriate benchmark citation. |
| Mallen et al. (2023), Joshi et al. (2017), Kwiatkowski et al. (2019) for PopQA, TriviaQA, NQ-open | Appropriate dataset provenance citations. |
| Gromov et al. (2025) for depth deletion and downstream evaluation context | Appropriate close antecedent; it supports the broader depth-pruning motivation, not this paper's specific proxy claim. |
| Kim et al. (2024, Shortened LLaMA) for CPT/retraining after depth pruning | Appropriate. |
| Sreenivas et al. (2024, Minitron) and Wibowo et al. (2025, IteRABRe) for recovery/pruning trajectories | Appropriate high-level related-work uses, though their recovery pipelines differ substantially from the OLMo case. |
| Namburi et al. (2023), Jaiswal et al. (2024), Xu et al. (2024) for compression metrics diverging from capability/safety | Appropriate motivation; the manuscript does not overclaim that these papers establish its exact setting. |
| Wang et al. (2024), Alzahrani et al. (2024), Gupta et al. (2024) for evaluation-interface/order sensitivity | Appropriate motivation for reporting interface details. |

The temporal treatment is also acceptable: SlimQwen and ShortOPD are called
concurrent rather than used to claim priority.  I did not find evidence of a
missing seminal citation that would change the novelty assessment.

## Desk, ethics, and process checks

- **Desk/style:** The frozen PDF is anonymous, uses ACL styling, has a
  limitations section and ethical-considerations section, and renders without
  unresolved references/placeholders.  It is 17 PDF pages: the main narrative
  reaches page 8, references begin afterward, and the remaining pages are
  appendix material.  Tables are readable at PDF zoom, although Table 2 and
  the final 57-subject table are dense.
- **Numbers:** The central arithmetic is internally consistent: 200k x 128 x
  2,048 = 52.4288B nominal token presentations, matching the stated 52.4B;
  25k gives 6.5536B, matching 6.6B.  The reported 1.428x PPL tax is consistent
  with 10.561 / 7.398.  The stated 28.6-point MMLU deficit is consistent with
  .605 - .319.
- **Hidden-instruction check:** I inspected the frozen TeX/source text for
  reviewer-directed or instruction-like hidden content and found none.
- **Ethics:** The ethics section is proportionate.  It addresses inherited
  model/data harms, energy use, and benchmark redistribution.  The key
  reproducibility/compute shortcomings are disclosed rather than concealed.
- **Review self-check:** My recommendation relies only on claims visible in
  the frozen v6 PDF/source and on the stated availability boundaries.  I have
  not treated missing historical information as proof of a result; rather, it
  limits the strength and reproducibility of the empirical inference.

## Recommendation

**Overall: 3.0 / 5 (Findings-level).**  The paper is careful, readable, and
honest about a useful measurement caution.  Its direct descriptive claim is
supported, but the evidence is single-run, lacks a long-horizon intact control,
does not evaluate an operational PPL-recovery rule, and leaves key generative
evaluation artifacts unverifiable.  Replicated matched trajectories plus
paired closed-book analyses would materially strengthen it.
