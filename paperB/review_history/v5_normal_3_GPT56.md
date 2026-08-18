review_mode: normal
soundness: 3.5
excitement: 2.5
overall: 3.0
confidence: 4.0
reproducibility: 2.0

# Summary and recommendation

This paper presents a deliberately bounded measurement case study of continued
pretraining after depth pruning. Its principal construction keeps OLMo-2-7B
blocks 0--13, adds two fresh blocks, and trains the resulting 16-layer model for
200k optimizer steps. The paper tracks same-source held-out perplexity, two
MMLU scoring interfaces, and three zero-shot closed-book QA evaluations. It
also reports an intact full32 branch available only at 25k steps, a frozen-front
operating point, a fully random 16-layer operating point, a non-contiguous
ShortGPT-16 operating point, shallower prefix runs, one OLMo-2-1B trajectory,
and one Qwen3-8B endpoint.

The central empirical observation is real and clearly reported: along the
single keep14 path, PPL decreases from 10.826 at 128k to 10.561 at 200k and
answer-letter MMLU increases from .3012 to .3191, but MMLU remains far below
the intact base (.6053), as do PopQA, TriviaQA, and NQ-open. The paper is
unusually careful not to turn its unmatched operating points into causal
ablations, not to interpret item-bootstrap intervals as training-run
uncertainty, and not to claim knowledge localization, deletion, or universal
recovery dynamics.

My recommendation is **Findings-level (Overall 3.0), not main-conference level**.
The bounded observed-path result is useful and mostly sound, but (i) the term
"certificate" is not operationalized as a prospective threshold or decision
rule, and the principal model never recovers to base-level PPL; (ii) the
evidence rests on one historical principal run without a matched 200k intact
branch or training-seed uncertainty; (iii) exact reproduction is blocked by
missing seeds, a missing loader offset, unavailable public evaluator commits,
and incomplete run/compute records; and (iv) the novelty is primarily the
combination and documentation of controls rather than a new method or a new
general empirical phenomenon. These limitations are compatible with a careful
Findings paper if the claims remain strictly literal and the "certificate"
rhetoric is narrowed.

# Claim--evidence map

| ID | Claim | Main evidence | Assessment |
|---|---|---|---|
| C1 | Along the observed keep14 path, improving in-domain PPL does not imply recovery to intact-base performance on the measured knowledge-sensitive evaluations. | Figure 1; Table 2; Appendix Tables 12, 15, and 16. PPL 10.826 -> 10.561 while MMLU .3012 -> .3191 versus base .6053; final PopQA/TriviaQA/NQ-open .1415/.2940/.0598 versus .2571/.6355/.2050. | **Supported as a literal, single-run descriptive claim.** It does not by itself test every possible calibrated PPL certificate. |
| C2 | Short-horizon corpus shift is not a complete explanation for the keep14 deficit. | full32 at 25k remains much closer to base than keep14 on PPL, both MMLU interfaces, and closed-book QA (Table 2/15/16). | **Reasonably supported only for the observed 25k horizon.** The paper correctly states that full32 is not a 200k control. |
| C3 | Complete-option MMLU is interface-sensitive and has a high non-inherited baseline. | Table 15: keep14 .3184 letter versus .3832 content-normalized; random .2470 versus .3598. | **Supported for these fixed checkpoints and this multi-factor interface change.** It does not isolate answer symbols, prompt format, tokenization, or normalization. |
| C4 | The principal deficit is not solely an answer-letter artifact. | Table 16 reproduces a large base--keep14 gap on PopQA, TriviaQA, and NQ-open generation. | **Supported descriptively.** No aligned closed-book predictions, paired intervals, or ShortGPT closed-book scores are available. |
| C5 | The keep14 endpoint is construction-dependent rather than a property of nominal 16-layer depth. | Table 2/4: ShortGPT-16 reaches PPL 9.780 and MMLU .4739 at 200k, versus keep14 10.561/.3191. | **Supported as an existence comparison.** The four coupled construction changes prevent factor attribution. |
| C6 | Frozen-front and random-init operating points help bound simple explanations but are not clean causal controls. | Table 2 and Table 14; the methods explicitly expose the different LR and trainable set. | **Supported and appropriately qualified.** |
| C7 | Shallow OLMo-2-7B, OLMo-2-1B, and Qwen3 observations provide directional context, not a depth law or cross-family replication. | Appendix Tables 3--8 and Figure 2. | **Appropriately scoped.** Their unequal checkpoints, compression ratios, corpora, and available metrics preclude stronger inference. |
| C8 | Likelihood, target evaluations, interface, construction, budget/compute, and run-level uncertainty should be reported separately. | Discussion and the study's own bookkeeping failures/sensitivities. | **A reasonable reporting recommendation**, but not empirically validated as a universal standard. |

# Strengths

1. **The paper is admirably explicit about inferential boundaries.** For
   example, the introduction states that "Every trained construction is a
   single run," that full32 is available only through 25k, that shallow
   checkpoints were selected after target metrics were inspected, and that
   random/frozen are operating points rather than clean ablations. This is much
   better than presenting unmatched historical runs as causal controls.

2. **The evaluation triangulation is useful.** Answer-letter MMLU,
   complete-option MMLU, and three closed-book generation datasets expose
   different failure modes. The fully random content-MMLU score is especially
   informative because it prevents over-reading keep14's higher content score
   as preserved inherited knowledge.

3. **The paper separates item uncertainty from run uncertainty.** The paired
   bootstrap and exact McNemar analyses are explicitly described as
   conditional on fixed checkpoints. The manuscript repeatedly warns that
   these are not seed-level intervals. This is statistically responsible.

4. **The appendix is unusually complete for a negative/measurement paper.**
   It includes all retained PPL checkpoints, the keep8 trajectory, the 1B
   trajectory, the Qwen endpoint, broad downstream tasks, raw versus normalized
   scoring, paired comparisons, interface controls, closed-book results, MMLU
   groups, and all 57 MMLU subjects.

5. **The main numerical claims are internally consistent.** The abstract,
   Figure 1, Table 2, and the detailed appendix agree up to explicitly disclosed
   differences between separately stored aggregates and common per-item reruns.
   The paper does not hide the unfavorable ShortGPT comparison or the weakness
   of the full32 control.

6. **The contribution is potentially useful despite being modest.** The work
   gives a concrete example of why a pruning paper should not collapse
   likelihood, target-task performance, construction, and compute into one
   "recovery" statement.

# Major weaknesses

## M1. The tested "certificate" is not specified as a prospective decision rule

**Location.** Title and Abstract, especially lines 002--029; Section 4,
lines 286--299; Section 5.1, lines 312--322; Figure 1 and Table 2.

**Exact quote.** Section 4 states that the examined certificate is "the
stronger implication that sufficient improvement in \(P_t\) certifies recovery
of \(E_t\)." The abstract concludes that "in-domain perplexity is not a
sufficient certificate for the measured knowledge-sensitive evaluations on
these observed paths."

**Severity.** **Major conceptual/soundness limitation.** "Sufficient
improvement" is never converted into a threshold, calibration curve, stopping
rule, or other falsifiable prospective certificate. More importantly, the
keep14 endpoint PPL is 10.561 versus 7.398 for the intact base, i.e. about
1.428x the base PPL. Thus, the study does not show a model whose likelihood has
recovered to the intact reference while its target behavior has not. It shows
that PPL can improve monotonically over a late interval while target behavior
remains far below base. That refutes the rule "any continued PPL improvement
certifies base-level capability recovery," but it does not refute every
possible PPL-based certificate, such as reaching a predeclared absolute
threshold, reaching a percentage of the intact model's PPL, or using a
construction-specific calibration.

The paper often uses careful phrases such as "on the literal observed paths,"
but the title and repeated unqualified noun phrase "PPL is not a certificate"
are broader than the operational test.

**Mechanical verification and required fix.** I verified the base and keep14
PPL values in Figure 1, Table 2, and Appendix Table 3, and checked Section 4
and the appendix for a numerical certificate threshold or registered stopping
criterion; none is provided. At minimum, the title, abstract, contribution
statement, and conclusion should be narrowed to **"PPL improvement alone does
not imply target recovery along these observed paths."** A stronger paper would
predeclare one or more practically plausible rules (for example, PPL within
5--10% of the intact reference, a plateau rule, or a calibrated
construction-specific predictor), then report sensitivity/specificity or
failure cases across checkpoints and runs.

## M2. One historical principal run and no matched long-horizon intact control

**Location.** Introduction lines 088--100; Section 3.2 lines 231--262;
Section 5.2 lines 334--341; Limitations lines 508--524; Tables 2--4 and 12.

**Exact quote.** "Every trained construction is a single run." "full32 has an
available 25k checkpoint only." The Limitations section also states that
"full32 ends at 25k and cannot control the keep14 200k endpoint" and that
"Training seeds were not explicitly set in the historical runs."

**Severity.** **Major evidence limitation.** A single trajectory is enough to
document that this particular checkpoint path exists, so this weakness does
not erase C1's literal observation. It does, however, prevent estimating
whether the dissociation is stable to initialization/data order and leaves the
200k comparison without an intact same-corpus counterfactual. The shallow
checkpoints were additionally retained after target metrics were inspected and
without a common stopping rule. Therefore the study cannot estimate a
run-level effect, a recovery-rate difference, or a long-horizon
pruning-specific gap. The paper acknowledges all of this, but these are still
the experiments needed to make the result more than a historical case.

**Mechanical verification and minimum experiment.** I checked Table 2's
budget column, Appendix Table 3's checkpoint inventory, and the training and
limitations text. There is one 200k keep14 run, one 25k full32 run, and no
reported training seed. The minimum strengthening experiment is:

1. extend full32 to the same 200k/52.4B nominal presentations;
2. run at least two additional independently seeded keep14 trajectories
   (three total), with checkpoints fixed in advance;
3. report per-seed curves and mean/range or a hierarchical analysis that keeps
   training-run and item uncertainty separate; and
4. report realized FLOPs, or at least a defensible FLOP estimate, alongside
   steps and nominal token presentations.

If compute permits only one priority, the matched 200k full32 branch plus two
additional keep14 seeds would be more informative than adding further
unmatched pruning constructions.

## M3. Exact reproduction and independent artifact verification are presently blocked

**Location.** Limitations lines 535--550; Appendix B.1 lines 829--868;
Appendix B.3 lines 903--931.

**Exact quote.** "Historical training seeds are unavailable." "the checkpoint
omitted the within-epoch loader offset." "Per-run wall time/GPU-hours and
aggregate project compute are unavailable." The evaluator commits "are not
ancestors of the public origin/main" and are "not publicly downloadable
artifacts."

**Severity.** **Major reproducibility weakness.** The paper gives many useful
hyperparameters, sample counts, prompts, and hashes, but an independent group
cannot exactly replay the principal training order or retrieve the exact
task-specific evaluator revisions from the public repository as described.
The paper itself says exact keep14 reproduction is blocked. In addition, the
closed-book aligned prediction files are absent, preventing independent paired
uncertainty analysis for those results. This does not automatically imply the
reported aggregate numbers are wrong, but it materially lowers confidence in
reproducibility and auditability.

**Mechanical verification and required fix.** I checked all reproducibility
subsections and the reviewed v5 source bundle. The manuscript names local
commit hashes and checksum prefixes, but also explicitly states that the
relevant commits are not public and that the historical loader offset is
missing. Before publication, the authors should release an anonymous,
self-contained evaluator snapshot or patch, environment lockfile/container,
exact commands/configurations, complete metric definitions, all legally
shareable per-item predictions, the full checksum manifest, and a script that
reconstructs every table from those artifacts. Because the historical loader
offset cannot be recovered, the authors should clearly distinguish
"reproduction of the exact historical trajectory" from "a clean replication
under a fully specified new seed," and ideally provide the latter.

## M4. The novelty is a control/interface combination rather than a new phenomenon or method

**Location.** Section 2, especially lines 137--183 and Table 1.

**Exact quote.** The paper states that "neither trajectories nor 'beyond
perplexity' evaluation originate here" and that it contributes "a measurement
case study and control combination, not a new recovery mechanism."

**Severity.** **Major for excitement and main-conference placement, but
compatible with Findings.** The cited pre-cutoff literature already covers
depth-pruning recovery curves, loss--task gaps, scratch/initialization
comparisons, iterative recovery, and evaluation beyond perplexity. The new
increment is the specific OLMo prefix+fresh-tail case together with an
available short-horizon intact branch, same-shape operating points, two MMLU
interfaces, closed-book QA, and explicit disclosure of confounds. That is
careful and useful, but it does not yet yield a general measurement framework,
predictive diagnostic, new benchmark, or causal insight.

**Mechanical verification and possible strengthening.** I compared the
paper's own nearest-work table and related-work descriptions across Gromov et
al., Shortened LLaMA, Minitron, IteRABRe, PASER, and the beyond-perplexity
compression papers. External novelty search was not performed and is marked
**Unverifiable** below. To reach main-conference novelty, the paper would need
to go beyond one case: e.g., define and evaluate a family of prospective
certificates across multiple architectures/constructions, quantify when
likelihood predicts target recovery, or release a reusable recovery-curve
benchmark with replicated runs and compute-normalized controls.

# Minor weaknesses

## m1. Inconsistent description of length normalization

**Location.** Appendix Table 13 on page 15 versus Appendix Table 18 on page 16.

**Exact quote.** Table 13 says: "The normalized score divides candidate
log-likelihood by continuation **character length** before selecting an
answer." Table 18 says: "acc_norm divides summed candidate log-likelihood by
continuation **token count** in the Paper B evaluator."

**Severity.** **Minor-to-moderate technical clarity issue.** These are not
equivalent normalizations and can change rankings, especially for BoolQ,
CommonsenseQA, and SocialIQA, which the paper explicitly identifies as
normalization-sensitive. The central answer-letter MMLU and closed-book claims
are unaffected, but secondary task comparisons are not exactly reproducible
until this is resolved.

**Mechanical verification and fix.** The two quoted captions use different
units. The authors should identify the exact denominator separately for every
reported `acc_norm` family, correct the inconsistent caption, and regenerate
affected values if one description is wrong.

## m2. `core6` in the ShortGPT table is undefined

**Location.** Appendix Table 4 on page 12.

**Exact quote.** The table header reports "PPL core6 MMLU" and gives .6215 for
the ShortGPT 200k endpoint.

**Severity.** **Minor presentation/reproducibility issue.** The table and nearby
text do not define which six tasks enter `core6`, their metrics, or whether the
number is a macro-average.

**Mechanical verification and fix.** I checked the Table 4 caption, methods,
and surrounding appendix text and found no definition there. Add the six task
names, aggregation formula, and rationale, or remove the column because it is
not load-bearing.

## m3. "Near the base" overstates some full32 closed-book values

**Location.** Section 5.2, lines 334--341.

**Exact quote.** "full32 remains near the base at 25k on PPL, both MMLU
interfaces, and all three closed-book tasks."

**Severity.** **Minor wording issue.** full32 is clearly much closer to base
than keep14, which is sufficient for the intended bounded argument, but its
NQ-open score is .1582 versus .2050 (about 23% relative lower) and TriviaQA is
.5715 versus .6355. "Near" should be quantified rather than asserted.

**Mechanical verification and fix.** These values are in Tables 2 and 16.
Replace "near" with absolute/relative differences and say that full32 retains a
substantially larger fraction of base performance than keep14 at 25k.

## m4. No out-of-domain likelihood or contamination audit

**Location.** Section 3.2; Limitations lines 529--534.

**Exact quote.** "PPL is in-domain, with no contamination audit or
out-of-domain likelihood."

**Severity.** **Minor under the paper's explicitly in-domain claim, but
important for practical interpretation.** Same-source held-out PPL may be a
particularly weak proxy for broad target tasks, and benchmark overlap cannot
be assessed.

**Mechanical verification and minimal fix.** The only PPL set is the disjoint
same-source 4096x2048 Dolmino shard. Add at least one genuinely out-of-domain
likelihood set and a documented overlap/contamination check for the target
benchmarks. If this cannot be run, keep "in-domain" in every headline claim and
avoid generalizing to likelihood metrics broadly.

## m5. The nearest-work matrix uses subjective binary cells without criteria

**Location.** Table 1.

**Exact quote.** Columns such as "Trajectory," "Loss+task,"
"Scratch/init," and "Construction" are populated with checkmarks, dashes, or
"partial."

**Severity.** **Minor literature-positioning issue.** "Partial" is explained,
but the inclusion thresholds for several columns remain subjective, and online
verification of every cell was unavailable.

**Mechanical verification and fix.** Add a short operational definition for
each column and page/figure pointers into each compared paper. This would make
the modest novelty claim easier to audit.

## m6. Two bibliography entries are rendered non-standardly

**Location.** References, especially the Minitron and OLMo entries.

**Exact quote.** Minitron's author list ends with "and 3 others"; OLMo is
rendered with a literal abbreviated team list.

**Severity.** **Minor bibliographic quality issue.**

**Mechanical verification and fix.** The strings appear in `main.bbl` and the
rendered references. Use valid BibTeX author lists or a properly formatted
collaboration author so ACL's bibliography style handles abbreviation
consistently.

# Questions for the authors

1. What exact prospective rule is meant by a "PPL certificate"? Would a model
   have to reach the intact model's PPL, a percentage of it, a plateau, or only
   improve relative to its post-pruning value?
2. What is the post-pruning/step-zero PPL and target score for keep14? Showing
   the full path from the intervention, rather than only 128k--200k in the
   headline panel, would make "recovery" easier to interpret.
3. Can full32 be extended to 200k, or can the authors explain why only the 25k
   checkpoint exists and whether later checkpoints were ever produced?
4. Can at least two new keep14 replications be run with explicit seeds and
   fixed checkpoint schedules?
5. Is `acc_norm` based on character length or token count for each task family?
6. Which tasks and aggregation define `core6`?
7. Can the exact local evaluator commits be released as an anonymous patch or
   source archive, together with the per-item content-MMLU and closed-book
   predictions?
8. Were the 128k, 153.5k, and 200k keep14 checkpoints selected before inspecting
   MMLU, or were all retained/selected after target evaluation?
9. Are there any available hardware logs from which approximate model FLOPs or
   energy use can be reconstructed, even if exact GPU-hours remain unavailable?
10. Why is the paper's concurrency discussion explicit for SlimQwen and
    ShortOPD but not for the May-2026 decision-transition paper? Exact public
    posting dates should be provided relative to the May 4, 2026 cutoff.

# Concrete revision and minimum-experiment priorities

## Required writing/analysis changes

1. Narrow "PPL is not a certificate" to "PPL improvement alone does not imply
   recovery on these observed paths," unless a prospective certificate is
   formally specified and tested.
2. Put the PPL ratio to base (1.428x for keep14) next to the target gaps so
   readers do not interpret the endpoint as likelihood recovery to intact
   quality.
3. Resolve character-versus-token normalization and define `core6`.
4. Quantify rather than verbally describe full32 as "near" base.
5. Include an explicit availability table for code, configs, checkpoints,
   predictions, hashes, and known unavailable fields.

## Minimum new experiments, in priority order

1. **Matched horizon control:** full32 through 200k/52.4B nominal
   presentations, with the same evaluation checkpoints.
2. **Run-level uncertainty:** two additional explicitly seeded keep14 runs
   (three total). Report each run, not only a mean.
3. **Prospective certificate:** predefine at least one PPL threshold/plateau
   rule and evaluate whether it predicts target recovery across all available
   checkpoints and runs.
4. **Likelihood scope:** one out-of-domain PPL dataset plus a documented
   contamination/overlap audit.
5. **Reproducibility replication:** one clean rerun using released code,
   explicit seeds, preserved loader state, environment lock, and recorded
   hardware/wall time.

The random-init and frozen-front controls need not be expanded unless the paper
wants causal initialization/adaptation claims. For the paper's current bounded
claim, matched full32 and replicated keep14 trajectories are the highest-value
additions.

# Technical audit

## Constructions and controls

- **Base:** intact 32-layer OLMo-2-1124-7B, no continued pretraining.
- **full32:** intact continued-pretraining control, but available only at 25k.
- **keep14:** principal 14 inherited + 2 fresh construction, all parameters
  trained through 200k.
- **Frozen:** same nominal 14+2 shape but inherited blocks frozen; non-block and
  fresh parameters train. This changes the trainable set.
- **Random:** same 16-layer shape, all modules random, different LR. This is a
  null operating point, not an initialization-only ablation.
- **ShortGPT:** 16 selected inherited blocks [0--12,16,17,31], no fresh tail.
  It changes inherited count, contiguity, final-layer retention, and fresh-tail
  use simultaneously.
- **Shallow prefix rows:** useful inventory, but unequal metric-informed
  stopping points and not step/token/FLOP matched.
- **1B/Qwen:** scope checks only; not matched replications.

This control taxonomy is correctly described in the paper. The missing
load-bearing control is an intact 200k trajectory.

## Metrics

- PPL uses a fixed disjoint same-source Dolmino shard and token-weighted
  aggregation over eight shards.
- MMLU letter scoring uses A--D continuations.
- MMLU content scoring changes prompt, candidate text, tokenization, and
  normalization simultaneously; it is therefore an interface comparison, not
  a one-factor ablation.
- PopQA uses normalized-answer containment; TriviaQA and NQ-open use exact
  match under greedy zero-shot no-retrieval generation.
- Broad downstream metrics are mostly clear, except the character/token
  normalization contradiction noted above.

## Seeds and statistics

- Every trained construction is one run; historical training seeds are
  unavailable.
- The paired MMLU analyses use 10,000 item bootstrap resamples with seed 1234
  and exact two-sided McNemar tests.
- The manuscript correctly labels these as item-level conditional uncertainty.
- Marginal MMLU intervals use approximate Wald intervals; with n=14,042 this is
  numerically acceptable for the reported aggregate rates, although it is not
  a substitute for run uncertainty.
- No correction is applied for the 57 subject-level views, but the paper does
  not present them as confirmatory tests.
- No statistical uncertainty is reported for closed-book QA because aligned
  per-item artifacts were not retained in the reviewed bundle.

## Scope

The load-bearing evidence is English, one 7B model family, one continued-
pretraining mixture, one prefix+fresh-tail recipe, and one historical principal
run. The paper states this accurately. The 1B and Qwen observations should
remain contextual.

## Compute

The paper reports optimizer steps and nominal token presentations:
200k x 128 x 2048 = approximately 52.4B presentations and 25k approximately
6.6B. It does not report realized FLOPs, unique-token counts, latency,
throughput, memory, per-run GPU-hours, project compute, or reliable hardware
records. Equal steps also imply unequal FLOPs across depths.

## Reproducibility assessment

Positive: model block indices, trainable sets, principal parameter count,
dataset/window sizes, validation size, batch size, schedule, optimizer
hyperparameters, precision, clipping, prompts, decoding, sample counts,
bootstrap seed, and several hashes are reported.

Negative: training seeds, loader offset, exact historical data order,
hardware/wall time, public task-specific evaluator commits, aligned
closed-book predictions, and an exactly replayable end-to-end package are
unavailable. The paper itself says exact reproduction of keep14 is blocked.
This supports a reproducibility score of **2.0/5.0**.

# Figure and table audit

## Figures

- **Figure 1:** The plotted keep14 values and endpoint bars agree with the
  tables. It clearly marks chance, the 25k-only full32 branch, single-run
  status, nominal tokens, and the non-causal scope. The figure is dense but
  legible. Its wording should be narrowed in line with M1.
- **Figure 2:** The 1B PPL trajectory and near-chance MMLU values agree with
  Tables 6--7. The caption appropriately calls this qualitative context rather
  than replication.

## Tables

- **Table 1:** Useful nearest-work map, but externally unverifiable here and
  based on partly subjective binary labels.
- **Table 2:** Main values, construction factors, LR, trainable set, and budget
  are exposed. This is the strongest presentation choice in the paper.
- **Table 3:** Clearly separates retained checkpoints from true/matched
  endpoints and labels full32's 25k limitation.
- **Table 4:** ShortGPT selection and endpoint are informative; `core6` is
  undefined.
- **Table 5:** Supports a descriptive keep8 within-run trajectory; correctly
  says it is not paired or replicated.
- **Tables 6--7:** Internally consistent 1B context.
- **Table 8:** Qwen endpoint is appropriately labeled directional only.
- **Table 9:** Broad downstream inventory is useful; conclusions should not be
  collapsed into a single average.
- **Table 10:** Chance-adjusted recovery formula is explicit. Negative random
  MMLU recovery is correctly described as chance-level rather than meaningful
  negative capability.
- **Table 11:** Marginal fixed-checkpoint uncertainty is clearly distinguished
  from seed uncertainty.
- **Table 12:** Late keep14 changes are internally consistent and appropriately
  described as small/mixed.
- **Table 13:** Helpful sensitivity analysis, but its character-length
  definition conflicts with Table 18's token-count definition.
- **Table 14:** Paired comparisons and slight aggregate discrepancies are
  transparently documented.
- **Table 15:** Strong interface control; correctly states that multiple
  factors change together.
- **Table 16:** Supports the large base--keep14 closed-book gap. Lack of aligned
  predictions and ShortGPT scores is disclosed.
- **Table 17:** Broad-group values match the textual range and are presented
  descriptively.
- **Table 18:** Sample counts and metrics are valuable; normalization wording
  needs correction.
- **Table 19:** Complete 57-subject reporting is useful and avoids selective
  subject presentation.

# Full `main.bbl` audit

The frozen `main.bbl` contains **33 entries**. All 33 have corresponding
resolved citation keys in the compiled auxiliary file, and no undefined
citation/reference warning appears in the build log. No bibliography entry is
orphaned. External metadata, venue, date, and URL verification is
**Unverifiable** because no network lookup was used.

| # | `main.bbl` entry | In-paper use | Internal audit | External status |
|---:|---|---|---|---|
| 1 | Alzahrani et al. (2024), *When benchmarks are targets* | Leaderboard/evaluation sensitivity | Cited and rendered; claim is plausible from title | Unverifiable |
| 2 | Chen et al. (2025), *A simple linear patch revives layer-pruned language models* | Cross-layer repair | Cited and rendered | Unverifiable |
| 3 | Chen et al. (2026), *Prune&Comp* | Magnitude compensation/iterative pruning | Cited and rendered | Unverifiable |
| 4 | Deng et al. (2025), *DRPruning* | Robust pruning/training context | Cited and rendered | Unverifiable |
| 5 | Gromov et al. (2025), *The unreasonable ineffectiveness of the deeper layers* | Deep-layer removal and loss--task gap | Cited and rendered; closest antecedent claimed | Unverifiable |
| 6 | Gupta et al. (2024), *Changing answer order can decrease MMLU accuracy* | MMLU interface sensitivity | Cited and rendered | Unverifiable |
| 7 | He et al. (2025), *PASER* | Recovery-oriented data selection | Cited and rendered | Unverifiable |
| 8 | Hendrycks et al. (2021), MMLU | Dataset definition | Cited and rendered | Unverifiable |
| 9 | Jaiswal et al. (2024), *Compressing LLMs* | Low PPL versus knowledge-intensive deficits | Cited and rendered | Unverifiable |
| 10 | Joshi et al. (2017), TriviaQA | Dataset definition | Cited and rendered | Unverifiable |
| 11 | Kim et al. (2024), *Shortened LLaMA* | Depth-pruning retraining trajectories and scratch comparison | Cited and rendered | Unverifiable |
| 12 | Kim et al. (2026), *Rethinking layer redundancy* | Calibration/task dependence of pruning choices | Cited and rendered; pre-cutoff by bibliography's April-2026 identifier | Exact date Unverifiable |
| 13 | Kwiatkowski et al. (2019), Natural Questions | Dataset definition | Cited and rendered | Unverifiable |
| 14 | Lu et al. (2024), *Reassessing layer pruning in LLMs* | Task/calibration dependence | Cited and rendered | Unverifiable |
| 15 | Mallen et al. (2023), PopQA | Dataset definition | Cited and rendered | Unverifiable |
| 16 | Martra (2025), *Fragile knowledge, robust instruction-following* | Selective capability effects under width pruning | Cited and rendered | Unverifiable |
| 17 | Men et al. (2025), ShortGPT | Block selection/removal and the ShortGPT construction | Cited and rendered | Unverifiable |
| 18 | Muralidharan et al. (2024), compact models via pruning/distillation | Pruning/distillation context | Cited; author list renders "and 3 others" | Unverifiable |
| 19 | Namburi et al. (2023), *The cost of compression* | Compression effects on parametric knowledge | Cited and rendered | Unverifiable |
| 20 | OLMo Team et al. (2025), *2 OLMo 2 Furious* | Base model definition | Cited and rendered | Unverifiable |
| 21 | Shi et al. (2026), decision representation transitions | Post-pruning collapse analysis | Cited and rendered | Exact public date Unverifiable |
| 22 | Siddiqui et al. (2024), *A deeper look at depth pruning of LLMs* | Task/calibration dependence | Cited and rendered | Unverifiable |
| 23 | Song et al. (2024), SLEB | Redundant-block removal | Cited and rendered | Unverifiable |
| 24 | Sreenivas et al. (2024), Minitron | Structured pruning/distillation and trajectories | Cited; non-standard truncated author rendering | Unverifiable |
| 25 | Tang et al. (2026), SlimQwen | Concurrent pruning/distillation work | Cited and rendered | Exact public date Unverifiable |
| 26 | Wang et al. (2024), *My Answer is C* | First-token versus generated-answer mismatch | Cited and rendered | Unverifiable |
| 27 | Wibowo et al. (2025), IteRABRe | Iterative block removal/recovery trajectories | Cited and rendered | Unverifiable |
| 28 | Xia et al. (2024), Sheared LLaMA | Structured pruning/data allocation | Cited and rendered | Unverifiable |
| 29 | Xu et al. (2024), *Beyond perplexity* | Multidimensional safety evaluation of compression | Cited and rendered | Unverifiable |
| 30 | Yang et al. (2025), Qwen3 technical report | Qwen scope-check model | Cited twice and rendered | Unverifiable |
| 31 | Yang et al. (2024), LaCo | Layer collapse/depth pruning | Cited and rendered | Unverifiable |
| 32 | Zhang et al. (2026), ShortOPD | Concurrent pruned-model recovery | Cited and rendered | Exact public date Unverifiable |
| 33 | Zhong et al. (2025), BlockPruner | Fine-grained attention/MLP block pruning | Cited and rendered | Unverifiable |

# Citation--claim match audit

The following semantic matches were checked against the frozen paper's
bibliographic titles and the way each work is described. Full-text external
verification remains **Unverifiable**.

| Citation | Paper's associated claim | Match assessment |
|---|---|---|
| Gromov et al. (2025) | Deep-layer removal; recovery/loss--task dissociation | **Plausible strong match** and identified as the closest antecedent. |
| Kim et al. (2024), Shortened LLaMA | Depth pruning with CPT/LoRA and scratch/init comparisons | **Plausible strong match** from title and description. |
| Sreenivas et al. (2024), Minitron | Structured pruning/distillation, trajectories, and initialization choices | **Plausible match**, but exact matrix cells are externally unverified. |
| Wibowo et al. (2025), IteRABRe | Iterative removal/recovery and weak MMLU recovery | **Plausible strong match** from title and description. |
| Namburi et al. (2023); Jaiswal et al. (2024); Xu et al. (2024) | Compression evaluation beyond aggregate PPL | **Plausible strong thematic match**. |
| Wang et al. (2024); Alzahrani et al. (2024); Gupta et al. (2024) | Multiple-choice/interface sensitivity | **Plausible strong match** to titles. |
| Men et al. (2025), ShortGPT | Layer redundancy/block selection and the selected 16-layer operating point | **Plausible strong match**; the paper correctly avoids calling its comparison selection-only. |
| OLMo Team et al. (2025) | OLMo-2-1124-7B model identity | **Direct model-source match**. |

No obvious citation is attached to a claim that contradicts its title. The
main unresolved issue is not a clear mismatch but the inability to externally
verify fine-grained Table 1 cells.

# Novelty analysis with cutoff May 4, 2026

## Search status

Network search was not used. Therefore all online novelty-search results,
first-public dates, and closest-paper full-text comparisons are
**Unverifiable**.

The five search formulations that would be required for a complete external
novelty audit are:

1. `"depth pruning" continued pretraining perplexity MMLU recovery curve`
2. `"layer-pruned" language model loss task gap healing`
3. `compressed LLM low perplexity knowledge benchmark`
4. `OLMo depth pruning fresh blocks continued pretraining`
5. `MMLU answer-letter complete-option scoring pruning`

## Closest pre-cutoff work represented in the frozen bibliography

- **Gromov et al.** already covers deeper-layer removal and post-healing
  loss--task dissociation.
- **Shortened LLaMA** already covers depth-pruning retraining curves and
  scratch/init comparisons.
- **Minitron** already covers structured pruning, retraining/distillation
  trajectories, and initialization choices.
- **IteRABRe** already covers iterative removal and recovery with task-family
  trajectories.
- **Namburi et al., Jaiswal et al., and Xu et al.** already motivate
  evaluating compressed models beyond PPL.
- **Kim et al. (arXiv identifier 2604.24938)** appears to precede the
  May 4, 2026 cutoff and therefore should count as prior work, subject to exact
  public-date verification.

Against this set, the credible novelty is the **combination** of an OLMo
prefix+fresh-tail path, a short-horizon intact branch, same-shape null operating
points, answer-letter/content MMLU, closed-book generation, and explicit
construction bookkeeping. This is nonzero but modest novelty.

## Three-month/concurrent-work handling

The bibliography includes Shi et al. (`2605.07271`), SlimQwen
(`2605.08738`), and ShortOPD (`2607.13124`). Their identifiers suggest May/July
2026 postings, likely after the May 4, 2026 novelty cutoff and within the
three-month window before this August 4, 2026 manuscript. Exact first-public
dates are **Unverifiable** without network access. They should therefore not be
used to erase novelty if confirmed post-cutoff, but they are relevant evidence
that the space was rapidly converging on matched initialization, progressive
recovery, and recognition/generation diagnostics.

# Desk, formatting, anonymity, and ethics audit

## Desk checks

- **Page count:** 17 PDF pages: 8 pages of main text including Limitations and
  Ethical Considerations, 2 pages of references, and 7 appendix pages. This
  appears compatible with an 8-page main-text limit.
- **Limitations:** Present and substantive on page 8.
- **Ethical Considerations:** Present on page 8.
- **Anonymity:** The PDF says "Anonymous ACL submission"; author metadata is
  blank. No author identity is visible in the reviewed PDF/source.
- **Style:** Uses `\usepackage[review]{acl}`, A4 pages, line numbers, and the
  expected ACL review layout.
- **References:** 33 resolved entries; no unresolved citations or references
  detected. Build warnings are underfull boxes, not unresolved references.
- **Placeholders:** No visible TODO, XX, unresolved `??`, or missing figure/table
  placeholder in the compiled paper.
- **Abstract/table consistency:** The headline numbers match Table 2 and the
  detailed appendix, with rerun differences explicitly disclosed.
- **Figures/tables:** All referenced figures and Tables 1--19 are present.
- **PDF integrity:** 17 pages, unencrypted, no embedded files, no JavaScript,
  no forms, and no suspicious non-link annotations found.
- **Reviewer manipulation/hidden instructions:** None detected in the PDF text
  or included source. The paper was treated as data, not as instructions.

I see **no clear desk-reject condition** in the reviewed frozen manuscript.

## Ethics

The work uses released models, corpora, and benchmarks and reports no new human
subjects or annotators. The ethical discussion appropriately mentions inherited
bias, hallucination, unsafe completions, deployment risk from misleading proxy
metrics, redistribution constraints, and energy use. Missing compute records
limit environmental accounting, but the paper discloses this rather than
inventing a total. I see no new high-severity ethical concern beyond ordinary
large-model training/evaluation concerns.

# Score rationale

- **Soundness 3.5/5.0.** The literal observed-path measurements are mostly
  sound and unusually well bounded. The score is below 4 because the
  "certificate" construct is not prospectively defined, the principal path is
  one run, and the matched long-horizon control is absent.
- **Excitement 2.5/5.0.** The negative result and reporting discipline are
  useful, but the paper itself correctly acknowledges that recovery curves,
  loss--task gaps, and beyond-PPL compression evaluation are established. The
  new contribution is primarily a careful combination.
- **Overall 3.0/5.0.** This is a reasonable Findings paper if the title and
  claims are narrowed to PPL improvement on literal observed paths. It is not
  yet main-conference level because novelty and experimental generality are
  limited.
- **Confidence 4.0/5.0.** I read the full 17-page paper twice, including both
  appendix sections, inspected all figures and tables, checked the frozen
  source/build artifacts and all 33 bibliography entries, and mechanically
  checked the quoted weaknesses. Confidence is not 5 because external
  citation/novelty verification is Unverifiable.
- **Reproducibility 2.0/5.0.** Many protocol details are present, but exact
  historical replay is blocked and key evaluator revisions/predictions are not
  publicly available as described.

# Review-process self-check

- Independent review from the v5 frozen manuscript; no prior review,
  score-history, TODO, status, current-version, or calibration file was used.
- Completed two readings of all 17 pages, including Appendices A and B.
- Enumerated the main claims and linked each to evidence and scope.
- Inspected Figures 1--2 and Tables 1--19.
- Checked page limit, Limitations, ethics, anonymity, ACL review style,
  placeholders, unresolved references, and abstract/table numerical
  consistency.
- Audited all 33 `main.bbl` entries for internal citation resolution.
- Audited eight load-bearing citation--claim matches from the frozen source.
- Applied the explicit novelty cutoff of May 4, 2026; external search and exact
  dates are marked Unverifiable.
- Distinguished item-level statistics from missing training-run uncertainty.
- Audited controls, metrics, seeds, statistics, scope, compute, and
  reproducibility.
- Each listed weakness includes a location, exact quote, severity, and a
  mechanically checkable verification/fix.
- No network-derived assertion is presented as verified.
