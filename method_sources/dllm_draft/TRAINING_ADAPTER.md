# Scaffold-Coder FSDP Training Adapter

## Components

- `scaffold_coder/sft_dataset.py`
  - reads cached IR parquet;
  - samples one global `t`;
  - builds a stochastic hierarchical canvas;
  - dynamically pads each batch;
  - returns role IDs, explicit target/loss masks, local clocks, and weights.
- `scaffold_coder/loss.py`
  - applies the Dream shift-op;
  - computes weighted masked CE;
  - normalizes by explicit weight mass rather than raw mask count.
- `scaffold_coder/training/scaffold_sft_trainer.py`
  - subclasses the released DreamOn FSDP trainer;
  - initializes new input/LM-head rows before FSDP synchronization;
  - uses FSDP FULL_SHARD across eight GPUs;
  - optionally records synchronized throughput and peak-memory JSONL;
  - supports checkpoint-free benchmark mode;
  - saves the tokenizer and `scaffold_tokens.json` with every checkpoint;
  - loads model weights from the actual resume checkpoint.
- `scaffold_coder/training/config/scaffold_sft.yaml`
  - primary five-epoch stage-1 configuration.

## CPU validation evidence

Remote environment:

```text
Python 3.11.6
torch 2.5.1+cu124
transformers 4.46.2
flash-attn 2.7.3
verl 0.3.0.post1
```

An eight-example validation batch produced:

- dynamic shape: `[8, 178]`;
- sample lengths: 46–178;
- supervised masks: 1–64/sample;
- finite explicit sample weights;
- maximum target ID: 151,685;
- configured model vocabulary: 152,064.

Hydra resolves the complete trainer configuration without launching distributed
workers. `TRAIN-PROFILE-CPU-001` rendered the profiling fields in the pinned
remote Python 3.11 environment and all 61 integrated tests pass locally and
remotely.

A tiny real Dream architecture (2 layers, hidden size 64, full 152,064-token
vocabulary) completed a CPU forward/backward/AdamW step with:

- finite loss: 11.98;
- nonzero `[FUNC]` LM-head gradient;
- nonzero mask input-embedding gradient;
- finite global gradient norm.

This verifies that structural targets, reserved-row initialization, the Dream
shift-op, explicit weights, and optimizer flow are connected end to end before
the full 7B GPU gate.

## Queued GPU gates

The heartbeat queue will run these only after all GPUs are idle:

1. `DREAM-CODER-GPU-SMOKE-001`
   - one-GPU Dream-Coder generation;
   - records load/generation time and peak memory.
2. `DREAMON-GPU-SMOKE-001`
   - one-GPU variable-length infilling;
   - validates released expand/EOS-delete behavior.
3. `SCAFFOLD-SFT-8GPU-SMOKE-001`
   - eight-way FSDP;
   - global batch 8, micro batch 1/GPU;
   - two optimization steps;
   - saves `global_step_2`.
4. `SCAFFOLD-SFT-RESUME-8GPU-SMOKE-001`
   - loads model, optimizer, scheduler, and tokenizer from step 2;
   - runs to step 3;
   - saves `global_step_3`.
5. `SCAFFOLD-THROUGHPUT-SWEEP-8GPU-001`
   - fixed global batch 128;
   - probes micro-batch 16, 8, 4, 2, and 1/GPU;
   - runs four optimizer steps per candidate without checkpoints;
   - excludes the first profile as warmup;
   - recommends the fastest setting with at least 5 GiB reserved-memory
     headroom when possible.
6. `SCAFFOLD-SFT-STAGE1-8GPU-001`
   - reads the recommendation artifact automatically;
   - profiles one step in ten during the five-epoch run.

Telemetry reports globally reduced model-step time, examples/s, non-padding
tokens/s, supervised tokens/s, and maximum allocated/reserved memory. It starts
after dataloader delivery, so separate profiling is still required if CPU input
wait becomes material.

## Unverified GPU risks

CPU tests cannot prove:

- H20 FlashAttention kernel execution;
- rank-0 initialized reserved rows synchronize correctly through FSDP;
- peak memory at optimizer initialization;
- dynamic sequence shapes interact correctly with FSDP collectives;
- full-state checkpoint and optimizer scattering work on this image.

The queued gates are deliberately ordered so a failure blocks later jobs and is
marked `NEEDS_DEBUG` rather than consuming all GPUs with a known-bad launch.
