# Six-review synthesis

## Scores

| Group | Soundness | Excitement | Overall | Confidence | Reproducibility |
|---|---:|---:|---:|---:|---:|
| Strict | 3.33 [3.0, 3.5] | 3.00 [3.0, 3.0] | 3.00 [3.0, 3.0] | 4.33 [4.0, 4.5] | 3.17 [3.0, 3.5] |
| Normal | 3.50 [3.5, 3.5] | 3.17 [3.0, 3.5] | 3.17 [3.0, 3.5] | 4.17 [4.0, 4.5] | 3.67 [3.5, 4.0] |
| All six | 3.42 [3.0, 3.5] | 3.08 [3.0, 3.5] | 3.08 [3.0, 3.5] | 4.25 [4.0, 4.5] | 3.42 [3.0, 4.0] |

## Individual reviews

- `v6_strict_1_GPT56.md` (strict): S=3.5, E=3.0, O=3.0, C=4.0, R=3.0
- `v6_strict_2_GPT56.md` (strict): S=3.5, E=3.0, O=3.0, C=4.5, R=3.5
- `v6_strict_3_GPT56.md` (strict): S=3.0, E=3.0, O=3.0, C=4.5, R=3.0
- `v6_normal_1_GPT56.md` (normal): S=3.5, E=3.0, O=3.0, C=4.5, R=4.0
- `v6_normal_2_GPT56.md` (normal): S=3.5, E=3.5, O=3.5, C=4.0, R=3.5
- `v6_normal_3_GPT56.md` (normal): S=3.5, E=3.0, O=3.0, C=4.0, R=3.5

## Critique extraction note

Consensus critique must be synthesized manually from the six evidence-anchored reports. Treat issues raised by >=3 reviewers as consensus; retain one-review concerns as outliers rather than deleting them.

## Consensus critique (manual synthesis)

### Consensus major issues

1. **No matched nearest reusable-context baseline (6/6).** This remains the
   dominant ACL-main barrier. The paper establishes an internal endpoint, not
   competitive utility against PIC/chunk-KV repair/modular-KV on the same model,
   pack, hardware, storage tier, and timing boundary.
2. **No clean run-level uncertainty for the exact headline adapter (6/6).** The
   flagship remains one run; extra adapters confound seed and effective batch and
   miss the exact headline cohorts.
3. **The overlap/context repair is not validated on natural or end-to-end claims
   (5/6).** Reviewers regard it as a useful synthetic diagnosis, not a repaired
   deployment frontier.
4. **Natural-task generalization remains contamination/judge limited (4/6).**
   Overlap audits are incomplete and LoCoMo uses a mutable judge, although the
   paper now labels these scope checks.

### What v6 fixed

- The dependence-aware equal-latency reanalysis is reproducible and numerically
  consistent. Reviewers no longer criticize the pooled-IID CI as the sole
  inferential analysis.
- Local v5 inconsistencies (break-even wording, LongEval/LongBench audit trail,
  unsupported prompt assertion, TurboRAG pages) were repaired.
- The BM25 conclusion remains robust under hierarchical/cell analyses; the BGE
  result remains unresolved, as it should.

### Remaining secondary concerns

- The nine-cell deployment estimand is still hand-selected, selector-asymmetric,
  and includes a one-conversation LoCoMo slice; better statistics do not make it
  representative.
- The fully matched depth frontier is primarily `j=0→12`; separately trained
  depths are not one controlled multi-depth frontier, and matched `j=12` Write
  cost was not retained.
- Teacher top-64 retained mass is unlogged.
- Frozen review source documents protocols well but does not itself prove a
  fully runnable end-to-end artifact.

## Score trend and stopping assessment

v6 all-six Overall is **3.08**, identical to v5 **3.08**. Strict reviewers are
again unanimous at **3.0**; normal mean is **3.17**. Soundness remains strong
(**3.42**) but excitement/practical impact is capped by missing nearest-system
and natural repair evidence. The writing/statistical iteration has converged:
further prose-only cycles are unlikely to raise the score.

Paper A is a **stable strong-Findings submission**. Reaching stable ACL-main
requires new experiments, principally a matched nearest reusable-cache baseline
and clean same-batch replications. The next action should therefore be to pause
writing-only iteration and obtain author authorization/resources for those
experiments, or submit as a carefully scoped Findings-level paper.
