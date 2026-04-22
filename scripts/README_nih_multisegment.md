# Multi-Segment NIH Evaluation Script

## Purpose

`eval_nih_multisegment.py` tests whether slot memory preserves information across segment boundaries. This is the critical validation that determines if the architecture actually works for long-context tasks.

## Key Features

1. **Multi-segment processing**: Documents are split into 2-4 segments with slot memory injected at each boundary
2. **Context lengths tested**: 2048, 4096, 6144 tokens
3. **Stage comparison**: Evaluates both Stage 2 (joint recon+CE) and Stage 3 (CE-only) checkpoints
4. **Retention calculation**: (multi-segment accuracy) / (single-segment accuracy) × 100%

## Success Criteria

| Retention | Interpretation | Action |
|-----------|---------------|--------|
| <10% | ⚠️ Slot attention collapse | Investigate implementation bug or fundamental architecture flaw |
| 10-50% | ⚡ Partial functionality | Consider improving training or increasing memory capacity |
| >50% | ✅ Architecture validated | Proceed to full RULER benchmark |

## Usage

### Basic Example

```bash
python scripts/eval_nih_multisegment.py \
    --model_path ../models/Qwen--Qwen3-8b/ \
    --stage2_path outputs/slot_memory_8gpu_stage2_20260420_122501_stage2_20260420_122538/final \
    --stage3_path outputs/slot_memory_8gpu_stage3_20260420_164731_stage3_20260420_164811/final \
    --output_dir outputs/nih_multisegment/ \
    --device cuda:0 \
    --num_trials 3
```

### Evaluate Only Stage 2

```bash
python scripts/eval_nih_multisegment.py \
    --model_path ../models/Qwen--Qwen3-8b/ \
    --stage2_path outputs/slot_memory_8gpu_stage2_.../final \
    --output_dir outputs/nih_multisegment/ \
    --skip_baseline
```

### Evaluate Only Stage 3

```bash
python scripts/eval_nih_multisegment.py \
    --model_path ../models/Qwen--Qwen3-8b/ \
    --stage3_path outputs/slot_memory_8gpu_stage3_.../final \
    --output_dir outputs/nih_multisegment/ \
    --skip_baseline
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | (required) | Path to base Qwen3-8B model |
| `--stage2_path` | "" | Path to Stage 2 slot memory checkpoint |
| `--stage3_path` | "" | Path to Stage 3 slot memory checkpoint |
| `--output_dir` | "outputs/nih_multisegment/" | Output directory for results |
| `--device` | "cuda:0" | GPU device to use |
| `--dtype` | "bfloat16" | Data type (float32/float16/bfloat16) |
| `--num_trials` | 3 | Number of trials per configuration |
| `--seed` | 42 | Random seed |
| `--num_slots` | 16 | Number of memory slots |
| `--slot_dim` | 256 | Slot dimension |
| `--skip_baseline` | False | Skip single-segment baseline (uses 98% from previous run) |

## Test Configurations

The script evaluates the following configurations:

- **Context lengths**: 2048, 4096, 6144 tokens
- **Needle depths**: 10%, 50%, 90%
- **Segment configurations**:
  - 2048 tokens: 2 segments × 1024 tokens
  - 4096 tokens: 2 segments × 2048 tokens OR 4 segments × 1024 tokens
  - 6144 tokens: 3 segments × 2048 tokens

Total configurations: 9 (3 lengths × 3 depths)
Total tests: 9 configs × 3 trials = 27 per checkpoint

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "timestamp": "2026-04-21T04:55:00",
  "model_path": "/path/to/Qwen3-8B",
  "baseline": {
    "overall_accuracy": 0.98,
    "per_config": {
      "single_seg_2048_d10%": {"accuracy": 1.0, "correct": 3, "total": 3},
      ...
    }
  },
  "stage2": {
    "overall_accuracy": 0.65,
    "retention_pct": 66.3,
    "interpretation": "✅ ARCHITECTURE VALIDATED",
    "per_config": {...}
  },
  "stage3": {
    "overall_accuracy": 0.68,
    "retention_pct": 69.4,
    "interpretation": "✅ ARCHITECTURE VALIDATED",
    "per_config": {...}
  },
  "comparison": {
    "stage2_retention_pct": 66.3,
    "stage3_retention_pct": 69.4,
    "stage2_interpretation": "✅ ARCHITECTURE VALIDATED",
    "stage3_interpretation": "✅ ARCHITECTURE VALIDATED"
  }
}
```

## Critical Finding Flagging

The script automatically flags critical findings:

- **Slot attention collapse**: Both Stage 2 and Stage 3 <10% retention
  - This matches RMT V7-V10 failure pattern
  - Indicates fundamental implementation or architecture issue

- **Architecture validated**: Either checkpoint >50% retention
  - Slot memory successfully preserves cross-segment information
  - Proceed to full RULER benchmark

- **Partial functionality**: 10-50% retention
  - Slot memory partially works but has limitations
  - Consider investigating training quality or memory capacity

## Expected Runtime

- Baseline (single-seg): ~5-10 minutes
- Stage 2 evaluation: ~10-15 minutes
- Stage 3 evaluation: ~10-15 minutes
- Total: ~25-40 minutes (with `--skip_baseline`: ~20-30 minutes)

## Dependencies

The script depends on:

- `src.memory.slot_memory.SlotMemoryCompressor`
- `src.memory.slot_memory.SlotMemoryModel`
- `transformers.AutoModelForCausalLM`
- `transformers.AutoTokenizer`

Ensure these modules are available in the Python path.

## Troubleshooting

### "No checkpoint file found" warning

If the script warns about missing checkpoint files, verify:

1. The checkpoint path is correct
2. Checkpoint contains `model.safetensors` or `pytorch_model.bin`
3. Compressor weights are in the checkpoint with `compressor.` prefix

### CUDA out of memory

Reduce memory usage by:

1. Using `--dtype float16` instead of `bfloat16`
2. Reducing `--num_slots` (e.g., to 8)
3. Using a larger GPU or switching to a different GPU with `--device`

### Low single-segment baseline accuracy

If baseline accuracy is significantly below 98%, the test cases may be too difficult. Consider:

1. Increasing `--num_trials` to get more stable measurements
2. Checking that the needle insertion is working correctly
3. Verifying tokenizer configuration

## References

- Original NIH-Extended: `scripts/eval_nih_extended.py`
- Slot memory model: `src/memory/slot_memory/slot_model.py`
- Slot compressor: `src/memory/slot_memory/slot_compressor.py`
