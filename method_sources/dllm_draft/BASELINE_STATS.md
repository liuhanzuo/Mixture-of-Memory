# Matched Plain-SFT Control

The plain baseline uses:

- the same Dream-Coder Base checkpoint;
- the same normalized train/eval split;
- the same chat formatting;
- response-only corruption;
- uniform `t ~ U(0,1)`;
- the same `min(20, 1/t)` per-sample weight;
- no scaffold tokens, deterministic templates, holes, or expand/delete edits.

Implementation:

- `PlainMaskedSFTDataset`
- `scaffold.mode=plain`
- the same weighted-loss/FSDP trainer
- queued smoke: `PLAIN-SFT-8GPU-SMOKE-001`

## State statistics

Validation: 1,000 eval rows × five epochs = 5,000 states.

| Metric | Plain SFT | Hierarchical global-band sampler |
|---|---:|---:|
| mean canvas length | 137.6 | 110.8 |
| p90 canvas length | 215 | 190 |
| p99 canvas length | 386 | 331 |
| mean supervised masks | 42.1 | 16.3 |
| median supervised masks | 31 | 2 |
| p99 supervised masks | 190 | 130 |

The hierarchical sampler has a 19.5% shorter mean model canvas and substantially
fewer simultaneously supervised positions because high-level states compress
whole subtrees into line labels or holes.

This does not by itself prove better wall-clock efficiency: attention cost must
be measured on GPU, and some hierarchical runs add extra denoising/model calls.
The matched GPU smoke and later NFE curves will report both wall time and
cumulative processed tokens.

## Bucketed 8-GPU launch selection

Before the full control, two full-global-batch probes compared micro-batch 16
and 8 using the actual length-bucketed Plain dataset:

| Micro/GPU | Median non-padding tokens/s | Median padding | Peak reserved | Headroom |
|---:|---:|---:|---:|---:|
| 16 | 14,050.5 | 0.241% | 30.53 GiB | 65.04 GiB |
| 8 | 11,914.0 | 0.241% | 31.04 GiB | 64.53 GiB |

The preregistered selector chose **micro-batch 16** by non-padding token
throughput while preserving far more than the required 5 GiB/GPU headroom.
The probes contain only three post-warmup records each, so the full-run
telemetry remains the authoritative efficiency measurement.

Plain Stage-1 launched on 2026-07-24 at 19:11 +08:00 with:

```text
launch commit           9104ba8419282c7110454bd66ef93675d244f280
world size              8
global batch            128
per-rank/local batch    16
micro-batch/GPU         16
gradient accumulation   1
length bucketing        true
checkpoint interval     1000 steps
```

The effective config is frozen in
`outputs/plain_sft_stage1/resolved_training_config.json`.

## Completed Stage-1 run

Plain Stage-1 completed on 2026-07-24 at 21:06:09 +08:00:

```text
final checkpoint       global_step_4465
attempts                1
active wall time        1.914 h
allocated GPU-hours    15.31
final validation loss   0.131
```

Final telemetry over 445 post-warmup profiles:

| Metric | Mean | Median | Max |
|---|---:|---:|---:|
| step seconds | 1.315 | 1.160 | 6.585 |
| examples/s | 114.84 | 110.32 | 207.27 |
| non-padding tokens/s | 13,736.3 | 14,152.0 | 16,036.7 |
| padding fraction | 0.352% | 0.189% | 11.664% |
| max sequence length | 147.9 | 128 | 894 |
| reserved GiB/GPU | 80.86 | 82.52 | 82.95 |

Compared with Schedule-only, Plain used 1.914 versus 5.116 active hours and
15.31 versus 40.93 allocated GPU-hours. Mean non-padding throughput was
13,736 versus 5,062 tokens/s, a 2.71× increase. This validates length
bucketing plus micro-batch 16 as a major efficiency improvement.

Loss diagnostics:

| Metric | Value |
|---|---:|
| first 100-step mean | 0.1880 |
| last 100-step mean | 0.1135 |
| last / first | 0.6035 |
| last-500 slope per 100 steps | -0.00144 |
| final validation loss | 0.131 |

Absolute loss values are not compared to Schedule-only because the corruption
and weighting objectives differ.

The registered 128-NFE, 16-task decode gate loaded the final checkpoint with
zero generation failures and 16/16 nonempty outputs, but only 3/16 parseable.
As with Schedule-only, the infrastructure is sound while the low-NFE quality
signal is weak. Full 512-NFE EvalPlus generation started immediately.

## HumanEval 512-NFE result

| Metric | Result |
|---|---:|
| HumanEval pass@1 | 28.05% |
| HumanEval+ pass@1 | 21.95% |
| nonempty | 163 / 164 |
| parseable | 131 / 164 (79.88%) |
| generation errors | 0 |
| mean NFE | 512 |

Failure attribution:

```text
syntax error             32 / 164  (19.51%)
empty output              1 / 164   (0.61%)
base semantic failure    85 / 164  (51.83%)
Plus-only failure         8 / 164   (4.88%)
Plus timeout              2 / 164   (1.22%)
HumanEval+ pass          36 / 164  (21.95%)
```

HumanEval+ by canonical compound depth:

| Depth | Tasks | Plain pass@1 | Schedule pass@1 | Scaffold pass@1 |
|---|---:|---:|---:|---:|
| 0–1 | 97 | 20.62% | 6.19% | 25.77% |
| 2 | 49 | 26.53% | 0.00% | 8.16% |
| 3+ | 18 | 16.67% | 0.00% | 5.56% |

Plain SFT is far stronger than Schedule-only (21.95% versus 3.66%) and also
exceeds the meta-token Scaffold checkpoint overall and at depth two and above,
although Scaffold remains better on the shallow slice while using far fewer
model calls. The matched result attributes the Schedule regression to the
depth-banded training distribution rather than generic SFT alone.

## MBPP 512-NFE result

| Metric | Result |
|---|---:|
| MBPP pass@1 | 30.42% |
| MBPP+ pass@1 | 24.34% |
| nonempty | 378 / 378 |
| parseable | 285 / 378 (75.40%) |
| generation errors | 0 |

Failure attribution:

```text
syntax error             93 / 378  (24.60%)
base semantic failure   170 / 378  (44.97%)
Plus-only failure        22 / 378   (5.82%)
Plus timeout              1 / 378   (0.26%)
MBPP+ pass               92 / 378  (24.34%)
```

MBPP+ depth slices:

| Depth | Tasks | Plain pass@1 | Schedule pass@1 |
|---|---:|---:|---:|
| 0–1 | 331 | 26.28% | 12.99% |
| 2 | 27 | 7.41% | 0.00% |
| 3+ | 20 | 15.00% | 0.00% |

Plain again dominates Schedule-only, proving that the structured depth schedule
rather than generic SFT caused most of the regression. Meta-token Scaffold
remains stronger on MBPP+ overall (32.01% versus 24.34%) while using about
48 rather than 512 mean NFE, so the two benchmark families favor different
trade-offs.

## HumanEval NFE curve

| NFE | HumanEval+ | Parseable |
|---:|---:|---:|
| 512 | 21.95% | 79.88% |
| 128 | 4.88% | 14.02% |
| 64 | 0.00% | 3.05% |

Plain's normalized log2-NFE AUC is 9.76%, well above Schedule-only's 1.22% but
far below Dream-Coder's 41.77%. Generic SFT recovers much of the 512-NFE
quality, but neither matched SFT control is robust at 64 NFE.
