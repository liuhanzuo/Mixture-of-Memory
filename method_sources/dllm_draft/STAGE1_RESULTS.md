# Stage-1 Scaffold-Coder Results

Date: 2026-07-24  
Checkpoint:
`outputs/scaffold_sft_stage1/global_step_4465`

## Training configuration

- Dream-Coder-v0-Base-7B initialization;
- strict/local-body Scaffold corruption;
- `sigma_d = 0`;
- five epochs over 114,363 normalized examples;
- 8-way FSDP FULL_SHARD;
- global batch 128, micro-batch 8/GPU;
- maximum sequence length 1,024;
- C0 monotone decoding for the primary evaluation.

The run completed from 2026-07-23 14:19 to 18:39 +08:00. All eight H20s
remained useful, with typical steady-state steps around 2.5–3.5 seconds and
roughly 34–39 GiB reserved per GPU. Registered active wall time was 4.324
hours, corresponding to **34.59 allocated GPU-hours** on eight H20s. This is
allocation accounting, not an electrical-energy estimate.

## Sampler issues discovered after training

The first HumanEval smoke exposed a deterministic two-state loop:

```text
four header masks --[expand]--> five masks --[delete]--> four masks
```

Fifteen of sixteen tasks exhausted 512 model calls. Two inference-only fixes
were added:

1. detect a repeated canvas reached through an elastic edit and suppress
   `[expand]`/`[delete]` on that repeated state;
2. remove edits from vocabulary support when they are illegal, including
   deleting the final required token or expanding past a budget.

These changes require no retraining. The repaired 16-task quality smoke had
zero generation exceptions and 14/16 parseable outputs.

## Full EvalPlus outcome

| Dataset | Generation errors | Parseable successful outputs | Base pass@1 | Plus pass@1 |
|---|---:|---:|---:|---:|
| HumanEval | 15 / 164 | 134 / 149 (89.9%) | 19.51% | 18.29% |
| MBPP | 35 / 378 | 325 / 343 (94.8%) | 38.36% | 32.01% |

All generation errors were:

```text
BudgetExceededError: generation exceeded 512 model calls
```

Process metrics over completed generations:

| Metric | HumanEval | MBPP |
|---|---:|---:|
| mean NFE | 58.79 | 47.77 |
| median NFE | 53 | 41 |
| p90 NFE | 96.0 | 83.6 |
| mean cumulative model tokens | 12,514 | 5,888 |
| mean maximum canvas tokens | 214.1 | 126.2 |
| mean expansions | 17.45 | 14.67 |
| mean edit-cycle breaks | 3.57 | 2.38 |

## Low-NFE smoke

On the same first 16 HumanEval tasks:

| Setting | Parseable | Mean NFE | Median NFE | Mean cumulative tokens |
|---|---:|---:|---:|---:|
| 1 commit/call | 14 / 16 | 68.19 | 62.5 | 12,373 |
| 4 commits/call | 4 / 16 | 21.56 | 17.0 | 3,670 |

The current sampler therefore degrades sharply at four commits per call. The
low-NFE hypothesis is not supported by this first checkpoint.

## Interpretation and G1 decision

The engineering path is valid: training, checkpoint/resume, neural structured
decoding, deterministic expansion, and full EvalPlus all run end to end.
However, the primary checkpoint does **not** yet support a quality-improvement
claim:

- functional scores are low in absolute terms;
- 9% of tasks still hit the NFE budget;
- lexical header/statement syntax is not guaranteed;
- aggressive parallel commits substantially reduce parseability.

## Same-harness Dream-Coder comparison

Dream-Coder-v0-Instruct-7B was evaluated with the same prompts, EvalPlus
versions, output extraction, and 8-way task sharding.

| Model / decode | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---:|---:|---:|---:|
| Dream-Coder, 512 NFE | 53.05% | 50.00% | 74.60% | 65.08% |
| Scaffold Stage-1, mean NFE 59/48 | 19.51% | 18.29% | 38.36% | 32.01% |
| Delta | -33.54 | -31.71 | -36.24 | -33.07 |

At approximately matched HumanEval NFE:

| Model | NFE | HumanEval | HumanEval+ | parseable |
|---|---:|---:|---:|---:|
| Dream-Coder | 64 fixed | 27.44% | 26.22% | 83 / 164 (50.6%) |
| Scaffold | 58.8 mean | 19.51% | 18.29% | 134 / 164 (81.7%) |

This is the first clear signal:

- explicit structure improves low-NFE parseability by roughly 31 percentage
  points;
- it still loses roughly eight HumanEval+ points at matched NFE;
- the current method trades semantic quality for structural completion.

The matched-NFE comparison is paired by HumanEval task:

| Outcome | Tasks |
|---|---:|
| both pass HumanEval+ | 16 |
| Scaffold only | 14 |
| Dream-Coder 64 only | 27 |
| neither | 107 |

Scaffold minus Dream-Coder-64 HumanEval+ is **−7.93 percentage points**. A
deterministic paired bootstrap gives a 95% interval of
`[−15.24, −0.61]` points; the exact two-sided McNemar test over the 41
discordant tasks gives `p = 0.0596`. These are descriptive single-run
statistics without multiple-comparison correction, but they confirm that the
quality gap is task-level rather than an artifact of comparing only aggregate
rates.

### Failure attribution

The automatic failure taxonomy separates generation-budget failures, Python
syntax failures, base-test semantic failures, and failures that pass the base
tests but fail the stronger Plus tests.

HumanEval:

| Model / decode | Generation | Syntax | Base semantic | Plus-only | Plus pass |
|---|---:|---:|---:|---:|---:|
| Scaffold (~59 NFE) | 15 (9.15%) | 15 (9.15%) | 102 (62.20%) | 2 (1.22%) | 30 (18.29%) |
| Dream-Coder, 512 NFE | 0 | 10 (6.10%) | 67 (40.85%) | 5 (3.05%) | 82 (50.00%) |
| Dream-Coder, 64 NFE | 0 | 81 (49.39%) | 38 (23.17%) | 2 (1.22%) | 43 (26.22%) |

MBPP:

| Model / decode | Generation | Syntax | Base semantic | Plus-only | Plus timeout | Plus pass |
|---|---:|---:|---:|---:|---:|---:|
| Scaffold (~48 NFE) | 35 (9.26%) | 18 (4.76%) | 180 (47.62%) | 24 (6.35%) | 0 | 121 (32.01%) |
| Dream-Coder, 512 NFE | 0 | 11 (2.91%) | 85 (22.49%) | 32 (8.47%) | 4 (1.06%) | 246 (65.08%) |

This sharpens the low-NFE result. Compared with Dream-Coder at 64 NFE,
Scaffold reduces HumanEval syntax failures from 49.4% to 9.1%, even after
counting a separate 9.1% generation-budget failure rate. The remaining gap is
predominantly semantic: 62.2% of HumanEval tasks fail base tests after
successful generation and parsing. The current runtime therefore improves
structural completion, but the checkpoint has not learned correspondingly
strong program semantics.

### HumanEval+ by canonical nesting depth

| Canonical compound depth | Tasks | Dream 512 | Dream 64 | Scaffold (~59 NFE) |
|---|---:|---:|---:|---:|
| 0–1 | 97 | 53.61% | 29.90% | 25.77% |
| 2 | 49 | 44.90% | 22.45% | 8.16% |
| 3+ | 18 | 44.44% | 16.67% | 5.56% |

The meta-token checkpoint does not improve the intended deep-nesting slice.
Its semantic regression is substantially larger at depth two and above. This
rules out a claim that the current structural templates already solve the hard
nested-control-flow regime.

DreamOn-v0 was also probed as body infilling from four initial masks. It often
deleted the entire body, so that setup is not a valid full-solution baseline;
it should be reported only as an infilling control.

## Final G1 attribution

The complete matched matrix is:

| Method | HumanEval+ | MBPP+ | HumanEval parse | MBPP parse | NFE |
|---|---:|---:|---:|---:|---:|
| Dream-Coder | 50.00% | 65.08% | 93.90% | 97.09% | 512 |
| Plain SFT | 21.95% | 24.34% | 79.88% | 75.40% | 512 |
| Schedule-only | 3.66% | 11.38% | 58.54% | 65.87% | 512 |
| Scaffold meta-token | 18.29% | 32.01% | 81.71% | 85.98% | 58.8 / 47.8 mean |

Conclusions:

1. **Drop the current depth-banded schedule.** Matched Plain exceeds Schedule
   by 18.29 HumanEval+ points and 12.96 MBPP+ points. Paired bootstrap
   intervals exclude zero, and exact McNemar tests remain significant after
   Holm correction.
2. **Use Plain SFT as the ordinary-quality primary checkpoint.** It is also
   2.7× faster in non-padding training throughput and consumes 15.31 rather
   than 40.93 allocated GPU-hours.
3. **Retain meta-token Scaffold only as a separately costed structural /
   low-NFE arm.** It beats Plain on MBPP+ and low-NFE parseability, but not
   consistently on functional quality or deep HumanEval nesting.
4. **Do not claim the original headline hypotheses.** Neither explicit
   structure nor schedule-only improves functional quality, deep nesting, or
   low-NFE degradation in the current checkpoints.
5. **Correction remains C0.** The tested C1/C2/C3 thresholds produced zero
   actions, so the calibration grid was inactive. Adaptive thresholds require
   confidence-distribution instrumentation before another correction claim.

The seeded-function-header ablation reduced mean NFE on the 16-task smoke from
68.2 to 52.3, but retained 14/16 parseability and introduced one budget
failure, so signature preservation alone does not repair the semantic gap.
