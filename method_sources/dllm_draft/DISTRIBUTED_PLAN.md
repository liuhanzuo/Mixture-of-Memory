# 8×H20 Distributed Execution Plan

Hardware:

- 8 × NVIDIA H20, approximately 95 GiB usable memory each;
- full `NV18` connectivity among all GPUs;
- 2 × 96-core AMD EPYC sockets, 384 logical CPUs;
- GPU 0–3 on NUMA 0, GPU 4–7 on NUMA 1.

## Primary topology

For full-parameter 7B SFT, use one process per GPU with FSDP FULL_SHARD:

```text
8 ranks
data-parallel/FSDP world size = 8
tensor parallel size = 1
sequence parallel size = 1 initially
BF16 parameters/compute
FP32 reductions and optimizer states as in upstream
gradient checkpointing enabled
```

This matches the released DreamOn trainer and avoids unnecessary tensor-parallel
communication. Full DDP is not the primary full-finetuning topology because
parameters, gradients, Adam states, activations, and temporary buffers are
likely to exceed a 96GB GPU at useful sequence/batch sizes.

TP is considered only if:

- a future model no longer fits under FSDP;
- profiling shows FSDP all-gather is the limiting factor;
- or a longer-context configuration benefits from tested TP kernels.

## Efficiency tuning order

1. Reproduce the upstream 8-way FSDP command unchanged.
2. Confirm all ranks make progress and checkpoint/resume works.
3. Sweep micro batch size upward until memory headroom is approximately
   5–10 GiB/GPU under steady-state training.
4. Tune gradient accumulation to preserve the intended global batch.
5. Measure data-loader/collator time. Precompute AST/IR and keep only stochastic
   noising online if CPU preprocessing is a bottleneck.
6. Enable remove-padding/FlashAttention only after correctness is reproduced.
7. Consider Ulysses sequence parallelism only for longer contexts where the
   measured gain exceeds communication overhead.

## Metrics required for every topology test

- step time after warmup;
- non-padding tokens/s;
- effective examples/s;
- peak allocated/reserved memory per rank;
- GPU utilization and power;
- dataloader wait fraction;
- FSDP communication fraction where profiling is available;
- checkpoint time and size;
- loss equality/near-equality across topology changes.

“All eight GPUs are used” is accepted only when every rank performs useful
synchronized work. Eight allocated but data-starved or communication-bound
processes do not satisfy the efficiency goal.

## Implemented telemetry and micro-batch selection

The Scaffold trainer now has opt-in synchronized telemetry:

- `trainer.profile_every_steps=N`;
- `trainer.metrics_jsonl=/path/to/metrics.jsonl`.

On profiled steps, every rank synchronizes around the optimizer step. Rank 0
records the maximum rank wall time, globally summed examples/non-padding
tokens/supervised tokens, and maximum allocated/reserved GPU memory. This
measures model-step throughput after the batch has been delivered; dataloader
wait time is intentionally not included. Normal stage-1 profiling runs every
ten steps so synchronization overhead remains small.

Before stage 1, `run_scaffold_throughput_sweep_8gpu.sh` probes per-GPU
micro-batches `16, 8, 4, 2, 1` at the same global batch 128. Each candidate:

- uses the exact strict Scaffold dataset/model/FSDP path;
- runs four optimizer steps;
- suppresses checkpoints;
- records one JSONL profile per step;
- survives an individual OOM/failure and continues to smaller candidates.

The summarizer discards the first warmup record, compares median non-padding
tokens/s, and selects the fastest candidate with at least 5 GiB of reserved
memory headroom. If none reaches that margin, it selects the fastest successful
candidate and marks that the headroom constraint could not be satisfied.

The fixed artifact is:

```text
ops/artifacts/scaffold_throughput_sweep.json
```

`run_scaffold_sft_stage1_8gpu.sh` reads its recommended
`micro_batch_size_per_gpu` automatically, with an explicit environment
override still available. The stage-1 run writes ongoing telemetry to its
output directory.

For ordinary-token schedule-only and plain controls, the trainer additionally
supports distributed length bucketing. It shuffles large length-sorted global
buckets, forms global batches, then assigns one local slice to each rank.
Simulation over all 114,363 training examples at global batch 128 reduced mean
per-rank padding waste from 2,550 to 16.5 token slots per local batch
(99.35%), while preserving deterministic epoch shuffling and disjoint rank
coverage. The currently running schedule-only job predates this switch; future
resume launches and the plain control enable it.

Profile JSONL now records padded tokens/s, padding fraction, and maximum
sequence length in addition to non-padding tokens/s. This makes the realized
GPU benefit auditable rather than relying only on the offline simulation.

## CPU and storage

- Keep source, data manifests, and checkpoints on the shared project filesystem.
- Stage reconstructible token/IR batches to `/dev/shm` when they fit.
- Use pinned-memory loaders.
- Start with 4–8 data-loader workers per rank and profile rather than allocating
  all 384 logical CPUs immediately.
- Respect NUMA placement during final throughput tuning:
  ranks 0–3 prefer CPU NUMA 0, ranks 4–7 prefer CPU NUMA 1.

## Inference/evaluation

- Single-sample dynamic-tree decoding starts on one GPU because each example has
  a different canvas length.
- Use all eight GPUs through data parallel evaluation: one independent sample
  stream per GPU, followed by result aggregation.
- Length-bucket samples before attempting within-rank batching.
- Report NFE, wall time, peak canvas length, and cumulative processed tokens.
