# Schedule-Only Pivot Statistics

Validation source: 1,000 normalized eval rows × five stochastic epochs  
Artifact: `ops/schedule_only_stats.json`

The schedule-only arm keeps ordinary Python tokens and applies depth-banded
masking without meta-token targets, region collapse, or deterministic template
expansion during training.

## State sizes

| Metric | Mean | p50 | p90 | p99 | Max |
|---|---:|---:|---:|---:|---:|
| canvas length | 149.8 | 135 | 235 | 434 | 503 |
| supervised masks | 61.4 | 48 | 126 | 259 | 399 |
| normalized loss mass | 1.92 | 1.07 | 3.79 | 14.07 | 20.0 |

For comparison:

| Training state | Mean canvas | Mean supervised masks |
|---|---:|---:|
| Plain uniform masking | 137.6 | 42.1 |
| Meta-token Scaffold | 110.8 | 16.3 |
| Schedule-only | 149.8 | 61.4 |

Schedule-only is therefore not a canvas-efficiency method. It deliberately
trades more token supervision and compute for preserving the pretrained
ordinary-code vocabulary.

## Supervision roles

Across 307,062 supervised positions:

```text
TOKEN_STMT  166,737
TOKEN_HDR    97,183
RULE         43,142
```

Structural/rule positions are 14.05% of supervised targets. The remaining
85.95% retain ordinary header and statement modeling, avoiding the semantic
signal dilution observed in the meta-token checkpoint.

## Mask count by global t

Mean masks rise from 4.4 in `t ∈ [0.0,0.1)` to 99.0 in
`t ∈ [0.9,1.0]`. Content masking dominates early intervals; rule/structural
tokens enter through their later depth bands. Every sampled item has positive
supervised mass.

## GPU smoke

The 8-H20 smoke completed with:

```text
step 0 train loss  3.926
step 1 train loss  3.394
validation loss    3.554
steady step time   0.414 s at global batch 8
reserved memory    31.9 GiB/GPU
```

The five-epoch global-batch-128 run began on 2026-07-24 at 12:38 +08:00.

## Launch-time provenance

The registered launch history and remote Git reflog recover the exact
launch-time commit as:

```text
1825ded0a607844a30a45ca56a0ff98402fce55f
```

The effective Hydra config parsed from the live training log is:

```text
per-rank train batch       16
micro-batch per GPU         8
length bucketing        false
checkpoint interval      2000 steps
```

Later repository patches enabled length bucketing and 1,000-step checkpoints,
but the already-running process did not consume those edits. Final experiment
manifests recover the launch commit, hash the launcher at that commit, and
store this resolved log config rather than attributing the current launcher to
the historical run.

## Completed Stage-1 run

The five-epoch run completed on 2026-07-24 at 17:45:50 +08:00:

```text
final checkpoint       global_step_4465
attempts                1
active wall time        5.116 h
allocated GPU-hours    40.93
```

Final loss diagnostics:

| Metric | Value |
|---|---:|
| first 100-step mean | 1.5751 |
| last 100-step mean | 0.5034 |
| last / first | 0.3196 |
| last-500 slope per 100 steps | -0.00314 |
| final validation loss | 0.567 |

Final steady telemetry over 445 post-warmup profile records:

| Metric | Mean | Median | Max |
|---|---:|---:|---:|
| step seconds | 3.917 | 3.818 | 7.291 |
| examples/s | 33.94 | 33.53 | 49.95 |
| non-padding tokens/s | 5,061.7 | 5,032.2 | 7,512.5 |
| reserved GiB/GPU | 72.80 | 75.71 | 84.41 |

The registered 128-NFE, 16-task decode gate loaded the final checkpoint with
zero generation failures and 16/16 nonempty outputs, but only 3/16 were
parseable. The infrastructure gate therefore passed, while the low-NFE quality
signal is weak. Full 512-NFE EvalPlus generation began immediately afterward.

## HumanEval 512-NFE result

Full HumanEval generation completed with zero generation exceptions:

| Metric | Result |
|---|---:|
| HumanEval pass@1 | 3.66% |
| HumanEval+ pass@1 | 3.66% |
| nonempty | 164 / 164 |
| parseable | 96 / 164 (58.54%) |
| mean NFE | 512 |
| mean seconds/sample | 50.23 |

Failure attribution:

```text
syntax error            68 / 164  (41.46%)
base semantic failure   90 / 164  (54.88%)
HumanEval+ pass          6 / 164   (3.66%)
```

Corrected HumanEval+ depth slices (Plus pass requires both base and additional
tests):

| Compound depth | Tasks | Pass@1 |
|---|---:|---:|
| 0–1 | 97 | 6.19% |
| 2 | 49 | 0.00% |
| 3+ | 18 | 0.00% |

This rules out a positive schedule-only HumanEval claim for the current
configuration. It is substantially below both the meta-token Scaffold
checkpoint (18.29% at about 59 NFE) and Dream-Coder controls. MBPP and the
64/128-NFE arms remain necessary for the matched Plain-vs-Schedule attribution.

## MBPP 512-NFE result

| Metric | Result |
|---|---:|
| MBPP pass@1 | 12.96% |
| MBPP+ pass@1 | 11.38% |
| nonempty | 378 / 378 |
| parseable | 249 / 378 (65.87%) |
| generation errors | 0 |
| mean NFE | 512 |
| mean seconds/sample | 43.92 |

Failure attribution:

```text
syntax error            129 / 378  (34.13%)
base semantic failure   200 / 378  (52.91%)
Plus-only failure         6 / 378   (1.59%)
MBPP+ pass               43 / 378  (11.38%)
```

MBPP+ by canonical compound depth:

| Depth | Tasks | Pass@1 |
|---|---:|---:|
| 0–1 | 331 | 12.99% |
| 2 | 27 | 0.00% |
| 3+ | 20 | 0.00% |

Schedule-only is also substantially below the meta-token Scaffold checkpoint
on MBPP+ (11.38% versus 32.01%), despite using 512 rather than about 48 NFE.
The critical remaining attribution is Schedule versus matched Plain SFT.

## HumanEval NFE curve

| Model | NFE | HumanEval+ | Parseable | Mean seconds/sample |
|---|---:|---:|---:|---:|
| Schedule-only | 512 | 3.66% | 58.54% | 50.23 |
| Schedule-only | 128 | 0.00% | 11.59% | 12.59 |
| Schedule-only | 64 | 0.00% | 3.05% | 6.32 |
| Dream-Coder | 128 | 41.46% | 79.88% | 12.59 |

All three Schedule arms had zero generation exceptions and 164 nonempty
outputs. Lowering NFE sharply reduces syntactic completion, and neither
128 nor 64 NFE solves any HumanEval+ task. The normalized log2-NFE AUC for
Schedule-only is approximately 1.22%, compared with approximately 41.8% for
Dream-Coder across its 64/128/512 points. The intended "degrades more slowly at
low NFE" hypothesis is decisively unsupported for this checkpoint.
