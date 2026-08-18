# Six-review synthesis

## Scores

| Group | Soundness | Excitement | Overall | Confidence | Reproducibility |
|---|---:|---:|---:|---:|---:|
| Strict | 2.83 [2.5, 3.0] | 2.50 [2.5, 2.5] | 2.50 [2.5, 2.5] | 4.50 [4.5, 4.5] | 2.00 [2.0, 2.0] |
| Normal | 3.50 [3.5, 3.5] | 3.00 [3.0, 3.0] | 3.00 [3.0, 3.0] | 4.17 [4.0, 4.5] | 2.17 [2.0, 2.5] |
| All six | 3.17 [2.5, 3.5] | 2.75 [2.5, 3.0] | 2.75 [2.5, 3.0] | 4.33 [4.0, 4.5] | 2.08 [2.0, 2.5] |

## Individual reviews

- `v4_strict_1_GPT56.md` (strict): S=3.0, E=2.5, O=2.5, C=4.5, R=2.0
- `v4_strict_2_GPT56.md` (strict): S=3.0, E=2.5, O=2.5, C=4.5, R=2.0
- `v4_strict_3_GPT56.md` (strict): S=2.5, E=2.5, O=2.5, C=4.5, R=2.0
- `v4_normal_1_GPT56.md` (normal): S=3.5, E=3.0, O=3.0, C=4.0, R=2.5
- `v4_normal_2_GPT56.md` (normal): S=3.5, E=3.0, O=3.0, C=4.0, R=2.0
- `v4_normal_3_GPT56.md` (normal): S=3.5, E=3.0, O=3.0, C=4.5, R=2.0

## Critique extraction note

Consensus critique must be synthesized manually from the six evidence-anchored reports. Treat issues raised by >=3 reviewers as consensus; retain one-review concerns as outliers rather than deleting them.

## Consensus critique (manual synthesis)

### Consensus major issues

1. **No training-seed replication for the central paths and rankings (6/6).**
   Every reviewer distinguished item-level uncertainty from training-run
   uncertainty. The paper can report literal observed traces, but cannot support
   seed-stable recovery dynamics, construction rankings, or a generally reliable
   reporting protocol. The decisive experiment is at least 3 matched seeds for
   keep14 and the strongest comparator, with seed-level PPL, MMLU interfaces, and
   closed-book metrics.
2. **The available controls are unmatched or coupled and therefore do not
   identify causes (6/6).** full32 stops at 25k rather than 200k; random changes
   LR and lexical modules; frozen changes the trainable set; ShortGPT changes
   retained count, selection, final-block retention, and fresh-tail design.
   Reviewers accept these as operating points, not clean ablations. Either add
   matched controls or narrow all causal language to bounded alternatives.
3. **Recovery horizon, stopping, and compute are not matched (5/6).** Shallow
   arms stop at different observed checkpoints, the intact control covers only
   the early horizon, and equal optimizer steps are not equal FLOPs across
   depths. Cross-depth trajectory claims require a common token/FLOP/checkpoint
   grid and explicit realized compute.
4. **Archival reproducibility is poor (6/6).** Unset training seeds, unrecorded
   resumed data-loader offset, incomplete compute accounting, absent frozen
   runnable artifact/environment/evaluator commits/checkpoint hashes, and
   incomplete prediction provenance prevent exact reproduction. This explains
   the uniformly low reproducibility scores (mean 2.08).

### Consensus secondary issues

5. **External validity and novelty are bounded (5/6).** The core is one
   OLMo-2-7B recipe on one in-domain stream; 1B/Qwen evidence is not a matched
   replication. Prior work already establishes pruning-recovery curves and
   loss/task dissociations, so the incremental contribution is the control
   bundle and reporting discipline.
6. **PPL/knowledge conclusions need tighter construct language (4/6).** PPL is
   entirely in-domain; MMLU/closed-book tasks are imperfect knowledge proxies;
   no contamination audit or out-of-domain PPL closes the interpretation.
   Preserve “knowledge-sensitive evaluations” and “observed path,” not knowledge
   storage/loss or universal recovery dynamics.
7. **Interface and closed-book evidence is incomplete (4/6).** Letter versus
   content scoring changes several protocol variables simultaneously and lacks
   paired uncertainty for the interface shift. Closed-book sample/uncertainty
   reporting is weaker and the strongest ShortGPT comparator lacks the same
   closed-book evaluation.
8. **Practical compression evidence is absent (3/6).** Several reviewers noted
   the lack of latency, throughput, memory, inference/recovery FLOPs, or a
   quality--total-compute frontier. This is secondary if the paper remains a
   measurement study, but blocks practical pruning claims.

### Strength consensus

- The manuscript is candid that it is an observational case study rather than a
  new pruning algorithm or universal law.
- It reports negative/null controls, two MMLU interfaces, closed-book tasks, and
  paired item-level MMLU statistics rather than relying on one endpoint metric.
- The distinction between likelihood, target behavior, interface, construction,
  and budget is useful and actionable as a reporting checklist.
- Limitations are unusually explicit; this prevents several confounded controls
  from becoming invalid causal claims.

## Outlier or low-frequency concerns

- Selective, metric-informed stopping was framed by two reviewers as a
  researcher-degrees-of-freedom risk rather than only unmatched compute.
- One reviewer emphasized the visually suggestive but non-causal readout/probe
  appendix.
- One strict reviewer requested multiplicity treatment for domain and late-path
  item analyses.
- Bibliographic archival metadata errors were a minor issue in one report.

## Decision and next iteration target

The standardized v4 aggregate is **Overall 2.75**: strict reviewers unanimously
score **2.50**, while normal reviewers unanimously score **3.00**. This is a
borderline-Findings / Findings-level bounded measurement study, not yet an ACL
main-conference submission. Writing alone can improve construct validity and
presentation, but a stable main-conference score likely requires new matched
experiments.

Priorities:

1. freeze the contribution as proxy insufficiency over literal observed paths;
2. make every operating-point confound visible in the main table and first use;
3. remove any implication that full32@25k controls the 200k endpoint;
4. expose missing seed/resume/compute/artifact metadata and release what exists;
5. restore the AutoFigure-Edit Figure 1 in the paper, but redesign it around the
   proxy-validity question and move causal-looking probe markers out;
6. if experiments are authorized, prioritize matched seeds and a 200k or
   token/FLOP-matched full32 control before expanding benchmarks.
