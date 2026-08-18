# Six-review synthesis

## Scores

| Group | Soundness | Excitement | Overall | Confidence | Reproducibility |
|---|---:|---:|---:|---:|---:|
| Strict | 3.33 [3.0, 3.5] | 3.17 [3.0, 3.5] | 2.83 [2.5, 3.0] | 4.33 [4.0, 4.5] | 3.50 [3.5, 3.5] |
| Normal | 3.67 [3.5, 4.0] | 3.50 [3.5, 3.5] | 3.50 [3.5, 3.5] | 4.00 [4.0, 4.0] | 3.67 [3.5, 4.0] |
| All six | 3.50 [3.0, 4.0] | 3.33 [3.0, 3.5] | 3.17 [2.5, 3.5] | 4.17 [4.0, 4.5] | 3.58 [3.5, 4.0] |

## Individual reviews

- `v4_strict_1_GPT56.md` (strict): S=3.5, E=3.5, O=3.0, C=4.5, R=3.5
- `v4_strict_2_GPT56.md` (strict): S=3.5, E=3.0, O=3.0, C=4.5, R=3.5
- `v4_strict_3_GPT56.md` (strict): S=3.0, E=3.0, O=2.5, C=4.0, R=3.5
- `v4_normal_1_GPT56.md` (normal): S=3.5, E=3.5, O=3.5, C=4.0, R=3.5
- `v4_normal_2_GPT56.md` (normal): S=4.0, E=3.5, O=3.5, C=4.0, R=4.0
- `v4_normal_3_GPT56.md` (normal): S=3.5, E=3.5, O=3.5, C=4.0, R=3.5

## Critique extraction note

Consensus critique must be synthesized manually from the six evidence-anchored reports. Treat issues raised by >=3 reviewers as consensus; retain one-review concerns as outliers rather than deleting them.

## Consensus critique (manual synthesis)

The counts below are based on six independent reports. An item is consensus
when at least three reviewers raised the same underlying issue, even if its
severity differed.

### Consensus major issues

1. **The equal-latency negative result is not protocol-complete (6/6).** All
   reviewers found that the abstract/Table 4 result (64.78 versus 53.22) lacks a
   self-contained definition of the mixed cohort, sample/task weights,
   calibration support and selection rule, absolute latency distributions,
   generation limits, and/or bootstrap unit. This is the clearest revision that
   can be completed without new experiments: add the raw protocol and result
   table, or remove the result from the abstract/headline.
2. **No matched nearest reusable-context baseline (5/6).** Reviewers agreed that
   the internal `j=0`/`j=12` causal frontier is strong, but it does not establish
   practical competitiveness against PIC/chunk-KV repair/learned modular-cache
   systems on the same backbone, evidence pack, hardware, storage tier, and
   timing boundary. This is the main experiment-level barrier to a stable ACL
   main-conference score.
3. **Adapter robustness is not a controlled multi-seed estimate (5/6).** The two
   extra runs change effective batch size and cover only selected cells. The
   manuscript should call this batch-plus-seed robustness, report seed-level
   headline aggregates, and avoid implying a clean training-seed variance
   estimate; a decisive fix would use identical training configurations.

### Consensus secondary issues

4. **Natural-task reproducibility/generalization remains partly open (4/6).** A
   mutable GPT-4o judge prevents exact LoCoMo replay, and several reviewers also
   noted unresolved PG-19/benchmark-overlap boundaries. Archive item-level judge
   outputs and dates/snapshots, report a full fixed-judge audit if possible, and
   label contamination-unresolved scope checks accordingly.
5. **Some claims/tables need stronger self-contained specification (3/6).** This
   includes the multi-depth meaning of “frontier,” cross-task numbers stated in
   prose without a protocol-complete table, the token/support definition of the
   distillation objective, and sparse 128k break-even support. These do not
   overturn the matched core but reduce auditability.

### Strength consensus

- The `j=0` versus `j=12` same-pack/same-adapter comparison is unusually well
  controlled and cleanly separates bounded selection from depth reuse.
- Negative results, storage costs, timing exclusions, and deployment limits are
  disclosed rather than hidden.
- Statistical and appendix reporting are substantially stronger than typical
  for an inference-systems paper.
- The paper now makes a credible bounded claim: an internal
  quality--latency--storage frontier, not competitive superiority.

## Outlier or low-frequency concerns

- One strict reviewer requested a more unified multi-depth frontier rather than
  a mainly endpoint-focused central table.
- One reviewer found the position boundary under-specified outside the Qwen
  implementation.
- One reviewer singled out the self-distillation support/renormalization
  definition.
- One reviewer treated the 128k break-even table as too sparse for the prose
  summary.
- Bibliographic metadata staleness was raised by two reviewers; it is a real but
  minor editorial fix rather than a scientific-score driver.

## Decision and next iteration target

The standardized v4 aggregate is **Overall 3.17**, with a sharp calibration
split: **strict 2.83** versus **normal 3.50**. Thus v4 is a strong Findings /
borderline-main manuscript, not yet a stable ARR/ACL-main submission. The next
iteration should prioritize, in order:

1. make the equal-latency result fully auditable or demote it;
2. add one matched nearest-system comparison if existing artifacts permit it;
3. accurately relabel and tabulate adapter robustness;
4. close judge/contamination reproducibility boundaries;
5. update stale bibliography metadata and tighten under-specified tables/formulas.
