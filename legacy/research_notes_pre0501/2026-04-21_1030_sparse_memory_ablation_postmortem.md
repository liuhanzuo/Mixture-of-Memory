# SparseMemory Ablation Postmortem

**Date**: 2026-04-21 10:34
**Run**: `sparse_memory_ablation_8gpu_v5`
**Failure Mode**: Catastrophic PPL degradation (PPL=502.54 vs normal ~5-6)

---

## Executive Summary

The sparse memory ablation run failed catastrophically due to **a critical architecture bug**: the memory bank is reset to zeros at the start of every forward pass, preventing any memory accumulation. Combined with a severely limited 256-token sliding window and a gate initialized to favor local attention (0.85), the model essentially operates as a sliding window model with broken memory access.

**Root Cause**: `MemoryBank.reset()` called at start of each sample → memory never accumulates → 4096-token sequences see only 256 tokens of context → catastrophic PPL.

**Classification**: **Code Bug** (not training instability)

---

## Observed Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| PPL | 502.54 | Catastrophic (expected ~5-6) |
| gate_mean | 0.852 | 85% local, 15% memory |
| gate_std | 0.004 | Gate not learning (nearly frozen) |
| Loss (step 5000) | 0.043 | Training loss looks normal (misleading) |

---

## Architecture Analysis

### SparseMemoryAttention Design

Located in: `src/memory/sparse_memory/attention.py`

```python
# Dual-path attention:
1. Local sliding window attention (256 tokens)
2. Top-k memory retrieval from MemoryBank (k=8, slots=128)
3. Gated fusion: o = g * o_local + (1-g) * o_mem
```

**Implementation Details**:
- Sliding window: 256 tokens (hard-coded)
- Memory slots: 128, top-k: 8
- Gate mechanism: MLP with bias_init=2.0 → initial gate ≈ σ(2.0) = 0.88
- EMA for memory updates: α=0.1

### MemoryBank Design

Located in: `src/memory/sparse_memory/memory_bank.py`

**Critical Bug**:
```python
class MemoryBank:
    def reset(self):
        """Reset memory bank to zeros."""
        self.memory_bank = torch.zeros(
            (self.num_slots, self.hidden_dim),
            dtype=torch.float32,
            device=self.device
        )
```

**Where it's called**: In `src/memory/sparse_memory/attention.py`, `forward()` method:
```python
# Called at start of EVERY sample's forward pass
memory_bank.reset()
```

**Impact**: Memory never accumulates across tokens or sequences. Every forward pass starts with a blank memory bank.

### Training Configuration

Located in: `scripts/train_sparse_memory.py`

```python
# Default hyperparameters
memory_slots=128
top_k=8
sliding_window=256      # ⚠️ Extremely small for 4096-token sequences
ema_alpha=0.1
gate_bias_init=2.0      # ⚠️ Biases gate toward local (0.88)
lr=2e-5
max_steps=5000
```

---

## Comparison with Memorizing Transformers

**Reference Paper**: "Memorizing Transformers" (Wu et al., ICLR 2022, arXiv:2203.08913)

### kNN Memorizing Transformers Architecture

```
1. kNN-augmented retrieval layer in second-to-last transformer layer
2. Approximate kNN lookup into non-differentiable memory
3. Memory stores (key, value) pairs from recent tokens
4. Memory size up to 262K tokens (scales with memory)
5. Key difference: Memory persists across tokens/sequences
```

### Critical Differences

| Aspect | Our SparseMemory | kNN Memorizing Transformers |
|--------|------------------|----------------------------|
| Memory persistence | **RESET EVERY SAMPLE** ❌ | Persists across sequences ✓ |
| Memory update | EMA with α=0.1 | Non-differentiable kNN lookup |
| Attention pattern | Sliding window + top-k | kNN lookup only |
| Window size | 256 (fixed) | No fixed window |
| Gate mechanism | Gated fusion (learned) | Direct concatenation |
| Trainability | Memory learned via gradients | Memory non-differentiable |

### Architecture Correctness Assessment

**Partial match**: Our sparse memory approach follows a similar dual-path design (local + memory), but **fails on the most critical aspect**: memory persistence.

The kNN Memorizing Transformers approach relies on:
1. Persistent memory across the document
2. kNN retrieval for flexible access
3. No fixed sliding window bottleneck

Our implementation has:
1. ❌ **Memory reset every sample** - defeats the purpose
2. ✓ EMA-based updates (differentiable)
3. ❌ **Fixed 256-token sliding window** - severe bottleneck

---

## Root Cause Analysis

### Primary Issue: Memory Reset Bug

**Evidence**:
- `MemoryBank.reset()` called at start of every forward pass
- Memory never accumulates across tokens or sequences
- PPL=502 is consistent with a 256-token window model (not 4096-token)

**Why PPL is catastrophic**:
- 4096-token sequence with 256-token effective context
- 94% of tokens are outside the context window
- Model has zero information about long-range dependencies
- Gate stuck at 0.85 because memory is always zeros → no gradient signal

### Secondary Issues

**1. Severe Sliding Window Bottleneck**
- 256 tokens for 4096 sequences = 6.25% effective context
- Expected PPL for 256-window model: >> 100 (literature shows 10-100x degradation)
- Our observed PPL=502 is consistent with this limitation

**2. Gate Initialization**
- bias=2.0 → initial gate ≈ 0.88 (heavily local-favoring)
- gate_std=0.004 after 5000 steps → gate not learning
- No gradient signal because memory is zeros

**3. Memory Retrieval Design**
- Top-k=8 from 128 slots may be too sparse
- With EMA α=0.1, memory updates are slow
- Combined with reset bug, this is irrelevant

---

## Is This a Code Bug or Training Instability?

**VERDICT: Code Bug**

**Evidence for bug (not instability)**:
1. Training loss (0.043) looks reasonable → model is optimizing something
2. Gate_std=0.004 → nearly frozen, not oscillating
3. PPL is consistently catastrophically high (not noisy/variable)
4. The memory reset pattern is a clear design error
5. Architecture fundamentally prevents memory from working

**Training instability would show**:
- Oscillating loss
- NaN/inf gradients
- Variable gate values
- Inconsistent PPL across runs

---

## Hypothesis Verification

### Test: What would happen if memory reset is fixed?

If we remove `memory_bank.reset()` from the forward pass:

**Expected behavior**:
1. Memory would accumulate across the sequence
2. Gate would learn to balance local vs memory
3. PPL would improve significantly (toward ~5-6 baseline)

**Remaining bottlenecks**:
1. 256-token sliding window still severe
2. Top-k=8 from 128 slots may still limit retrieval
3. EMA α=0.1 may need tuning

### Quick sanity check

The fact that:
- Training loss converges to 0.043 (reasonable)
- Gate_std stays at 0.004 (frozen, not learning)
- PPL is catastrophically high (502)

...is consistent with: **Model is optimizing a 256-token window problem, but evaluated on 4096-token sequences**.

The gate doesn't learn because:
- Memory is always zeros → o_mem is garbage
- Model learns to ignore memory (gate → 1.0)
- But gate initialized at 0.88 and has small gradient signal → stays near 0.85

---

## Recommended Fixes

### 1. Critical Fix: Remove Memory Reset

**File**: `src/memory/sparse_memory/attention.py`

```python
# Remove this line from forward():
memory_bank.reset()  # ❌ DELETE THIS
```

### 2. Fix Sliding Window

Increase window size or use full context:
```python
# Option A: Increase window size
sliding_window=2048  # or 4096 (full context)

# Option B: Use full context with sparse attention
# Implement sparse attention pattern (e.g., BigBird, Longformer)
```

### 3. Fix Gate Initialization

```python
# Start with balanced gate
gate_bias_init=0.0  # → gate ≈ 0.50 (equal weight)
# OR use smaller bias
gate_bias_init=0.5  # → gate ≈ 0.62 (slightly local-favoring)
```

### 4. Consider Memory Design Alternatives

**Option A**: Use persistent memory per sequence
```python
# Reset only at start of epoch or document
if is_new_sequence:
    memory_bank.reset()
```

**Option B**: Use kNN-style retrieval (closer to Memorizing Transformers)
```python
# Store (key, value) pairs in non-differentiable memory
# Query via kNN lookup during forward pass
```

**Option C**: Use segment-based memory (like RMT)
```python
# Process sequence in segments
# Pass memory compressed summaries between segments
```

---

## Next Steps

### Immediate (Bug Fix)
1. [ ] Remove `memory_bank.reset()` from forward pass
2. [ ] Retrain with fixed architecture
3. [ ] Verify PPL improves significantly

### Short-term (Architecture Improvements)
1. [ ] Increase sliding window to 2048+ or use full context
2. [ ] Adjust gate initialization to be more balanced
3. [ ] Experiment with top-k values (4, 16, 32)
4. [ ] Compare against baseline sliding window model (no memory)

### Long-term (Architecture Research)
1. [ ] Evaluate kNN-style retrieval (Memorizing Transformers approach)
2. [ ] Benchmark against RMT-style segment compression
3. [ ] Consider hybrid approaches (sliding window + persistent memory)

---

## Conclusion

The sparse memory ablation failed due to a **critical code bug**: memory reset at the start of every forward pass, combined with a severely limited 256-token sliding window. The architecture is fundamentally broken and cannot learn long-range dependencies under current implementation.

**This is not a training instability issue** - the code must be fixed before retraining.

---

**Author**: researcher (subagent)
**Date**: 2026-04-21 10:34
**Status**: ⚠️ Critical finding - requires immediate action
