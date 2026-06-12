# ⚠️ CRITICAL: RMT v10 Complete Failure on NIH Evaluation

## Status
**CRITICAL ISSUE** - RMT v10 achieved 0% accuracy on Needle-in-Haystack (NiH) evaluation across all configurations, indicating complete failure of the memory mechanism.

## Key Findings

### 1. RMT v10 Complete Failure
- **Base model accuracy**: 91.1% (82/90 correct)
- **RMT v10 accuracy**: 0% (0/90 correct)
- All configurations (depths 0.1-0.9, lengths 1024-4096) failed equally

### 2. Symptom Analysis
RMT v10 outputs show clear degradation patterns:
- **Hallucinated code snippets**: "import { create } from 'zustand'", "import React from 'react';"
- **Repeated text loops**: "from 'from from from from" repeated 20+ times
- **Random imports**: Python/JS imports unrelated to the task
- **No needle retrieval**: Model completely ignores embedded information

### 3. Training Success vs. Inference Failure
Training logs indicate healthy convergence:
- Loss decreased from 3.11 → 2.25 over 20 epochs
- Retrieve_loss decreased from 0.014 → 0.008
- Training completed successfully (10.5h)
- Checkpoint files present and valid:
  - model.safetensors (16.3GB)
  - rmt_memory.pt (180MB)
  - Config files intact

### 4. Model Configuration
```
extractor_version: 5
num_memory_tokens: 64
segment_length: 1024
max_segments: 6
bottleneck_dim: 256
base_model: Qwen3-8B
lambda_retrieve: 0.5
```

## Interpretation

### Root Cause Hypothesis
**Memory extraction is completely non-functional during inference**, despite training success. Possible causes:

1. **Memory initialization mismatch**: Inference code may use different memory initialization than training
2. **Segment index position drift**: Position IDs for memory tokens may be incorrect during generation
3. **Extractor version mismatch**: V5 extractor may have inference code path issues not present in training
4. **Memory flow break**: Old memory may not be properly passed between segments in evaluation script
5. **Embedding layer hook failure**: Memory embeddings may not be inserted correctly at inference time

### Evidence Supporting Hypothesis
- Training worked (loss decreased, model converged)
- Evaluation script uses same `RMTModel._forward_single_segment` as training
- Base model works fine (91% accuracy), so tokenizer and data are correct
- Degenerate outputs (repeated imports, code snippets) suggest embedding-level corruption

## Recommended Next Actions

### Immediate (Priority 1)
1. **Debug memory extraction** - Add logging to track:
   - Memory values after each segment
   - Old memory → new memory transition
   - Memory embeddings at input layer
   - Extractor outputs at each step

2. **Verify checkpoint loading** - Confirm:
   - rmt_memory.pt loaded correctly
   - Memory embeddings initialized properly
   - Extractor weights match training

3. **Minimal reproduction test** - Create a simple 2-segment test with known content:
   ```
   Segment 1: "The code is 123456."
   Segment 2: "What is the code?"
   ```
   Log memory states at each step.

### Short-term (Priority 2)
4. **Compare v5 vs v2/v3** - Test if earlier extractor versions work:
   - Re-train with extractor_version=2
   - Re-train with extractor_version=3
   - Compare NIH results

5. **Position ID audit** - Verify position IDs are consistent:
   - Training: segment_idx * segment_length
   - Inference: same calculation?
   - Memory tokens: always 0..num_memory_tokens-1

6. **Memory initialization check** - Compare training vs inference:
   - Training: `self.get_initial_memory(seg_idx, B, device, dtype)`
   - Inference: same call?
   - Check segment_idx values

### Long-term (Priority 3)
7. **Add validation checkpoints** - During training, run NIH on checkpoint:
   - Step 500, 1000, final
   - Catch degradation early
   - Save training+inference state at each checkpoint

8. **Extractor V5 audit** - Review V5 implementation for:
   - CrossAttentionExtractor correctness
   - ImportanceMemoryUpdater logic
   - Tuple return handling (memory, aux_loss)

9. **Evaluation script robustness** - Add sanity checks:
   - Memory value variance (should not be all zeros)
   - Embedding shape checks
   - Position ID validation
   - Segment count verification

## Risks & Uncertainties

### Risks
- **High**: RMT v10 memory mechanism is completely broken in inference
- **Medium**: May require significant debugging time to root cause
- **Medium**: May need to abandon V5 extractor if fundamental design issue

### Uncertainties
- **Unknown**: Why training succeeded but inference failed
- **Unknown**: Whether V2/V3 extractors would work better
- **Unknown**: If this is a code bug or fundamental V5 design issue

## Handoff for Main Agent

**⚠️ CRITICAL**: Do not proceed with RMT v10 experiments until NIH evaluation passes at least 50% accuracy.

Priority order:
1. Debug memory extraction with logging
2. Verify checkpoint loading
3. Create minimal reproduction test
4. Consider reverting to V2/V3 extractor if V5 cannot be fixed quickly

**Context**: This is the third consecutive NIH failure (v7, v8, v10). Base model (91%) works fine, so data/evaluation pipeline is correct. Issue is specifically in RMT memory mechanism.

## Metadata

- **Git branch**: main
- **Short commit**: 4ce1dc0
- **Full commit**: 4ce1dc045ad65dc7432f61fa25a4eaabd7e5b374
- **Key files inspected**:
  - `outputs/nih_eval_v10_zh/nih_results.json`
  - `outputs/rmt_v10_8gpu_20260419_001626_20260419_001703/train.log`
  - `outputs/rmt_v10_8gpu_20260419_001626_20260419_001703/final/rmt_config.json`
  - `scripts/eval_needle_haystack.py`
  - `src/memory/rmt/rmt_module.py`
- **Timestamp**: 2026-04-19T12:18:00+08:00
- **Evaluation command**: `scripts/run_eval_nih_v10.sh`
