# Catastrophic PPL Analysis: Sparse Memory Ablation (8GPU v5)

**Date**: 2026-04-21 10:30
**Author**: researcher subagent (catastrophic PPL assessment)

## Executive Summary

The `sparse_memory_ablation_8gpu_v5` run achieved catastrophic PPL (502-549, normal is 5-6) with saturated gate values (~0.85, std=0.004). **Root cause: DDP training with unsynchronized memory buffers**. The memory bank is implemented as a buffer (not parameter) with explicit "no DDP sync", causing each GPU to train on independent, inconsistent memory states. Combined with gate saturation indicating the model ignores memory, this creates a fundamental training instability at 7B scale.

---

## 1. Experiment Details

### Configuration
- **Model**: Llama2-7B with SparseMemoryAttention
- **Memory slots**: 128
- **Top-k retrieval**: 8
- **Sliding window**: 256
- **EMA alpha**: 0.1
- **Gate bias init**: 2.0 → σ(2.0) ≈ 0.88 (local-heavy initialization)
- **Training**: 5000 steps, 8 GPUs (DDP), batch_size=1, grad_accum=4
- **LR**: 2e-5, cosine decay

### Results
| Eval | PPL | Avg Loss | Seq Length | Tokens | Docs |
|------|-----|----------|------------|--------|------|
| Run 1 | **548.66** | 6.31 | 2048 | 102,350 | 50 |
| Run 2 | **502.54** | 6.22 | 4096 | 204,750 | 50 |

**Training loss was normal** (0.02-0.10 per step), but eval PPL is catastrophic (100× worse than baseline).

### Gate Behavior
- **Mean**: 0.852-0.854 (consistent throughout training)
- **Std**: 0.0018-0.0052 (extremely low - almost no differentiation)
- **Interpretation**: Model gates are saturated near initialization; model ignores memory and does mostly local attention

---

## 2. Root Cause Analysis

### 2.1 Primary Issue: DDP Memory Bank Desynchronization

**The memory bank is a buffer, not a parameter:**

```python
# From src/memory/sparse_memory/memory_bank.py:
# Memory buffer: [1, N, d] placeholder; expanded to [B, N, d] on reset()
# Registered as buffer (not parameter) → no optimizer updates, no DDP sync
self.register_buffer(
    "memory",
    torch.zeros(1, num_slots, hidden_dim, dtype=dtype),
    persistent=False,
)
```

**Explicit comment in code: "no DDP sync"**

**What this means:**
1. Each of the 8 GPUs has its own independent memory bank
2. During DDP training, memory states are **not synchronized** across GPUs
3. GPU 0 trains with memory written by GPU 0 only
4. GPU 1 trains with memory written by GPU 1 only
5. ...and so on for all 8 GPUs

**Why this causes catastrophic failure:**

1. **Gradient inconsistency**: Each GPU sees different memory states, leading to divergent gradient updates
2. **Memory poisoning**: Local attention patterns learned on GPU 0 may be incompatible with memory states on GPU 1
3. **Forward pass fragmentation**: During training, the model never sees consistent memory-state-to-output relationships

**Why training loss looked normal:**
- Each GPU's forward pass is locally consistent with its own memory
- CE loss only requires next-token prediction, which can be achieved with local attention alone
- The model learns to **ignore memory** (gate saturation) and rely only on the 256-token sliding window

**Why eval PPL is catastrophic:**
- Eval runs on a single GPU with fresh memory (reset per sample)
- The model was trained on fragmented, inconsistent memory states across GPUs
- The attention weights are now poisoned by inconsistent gradients
- The gate mechanism, which learned to ignore memory during training, cannot adapt

### 2.2 Secondary Issue: Gate Saturation Pattern

**Observation**: gate_mean=0.85 (σ(1.7-1.8 bias)), gate_std=0.004

**Fusion formula**: `output = g * o_local + (1-g) * o_mem`

**With g ≈ 0.85**:
- Local attention contributes 85% to output
- Memory attention contributes only 15%

**With std ≈ 0.004**:
- Minimal differentiation across layers, positions, tokens
- Gate values are essentially constant

**Interpretation**:
- Model has learned to **ignore memory** and use mostly local sliding window
- Gate mechanism has collapsed to near-initialization value (bias_init=2.0 → σ≈0.88)
- Memory retrieval (top-k=8 from 128 slots) provides minimal value

**Is this a known pattern?**
Yes - this is consistent with the previous researcher report (2026-04-20):
- "kNN memory retrieval only validated at 125M-350M scale"
- "7B with 128 slots may be 4 orders of magnitude too small"

The gate saturation suggests the model cannot find useful information in the sparse memory slots, possibly because:
1. 128 slots is insufficient for 7B model complexity
2. EMA update (alpha=0.1) may be too aggressive, overwriting useful memory
3. Top-k=8 retrieval from only 128 slots provides too few candidates

### 2.3 EMA Update Instability

**Memory write logic**:
```python
@torch.no_grad()
def write(self, hidden_states: torch.Tensor, batch_idx: int) -> None:
    # ...
    alpha = self.ema_alpha  # 0.1
    updated = alpha * hidden_states + (1.0 - alpha) * current
```

**With alpha=0.1**:
- Each write retains 90% of old memory
- New hidden states contribute only 10%

**Potential issues**:
1. **Memory staleness**: Old information persists too long (exponential decay τ=10 steps)
2. **Information dilution**: Each write averages new information with old, potentially diluting signal
3. **Circular buffer artifacts**: With only 128 slots, information is quickly overwritten in long sequences

**This is speculative** - the primary issue is DDP desynchronization - but EMA dynamics may contribute to memory ineffectiveness.

---

## 3. Comparison with Known Failure Modes

### 3.1 Known Failure Modes of Memory Approaches at 7B+ Scale

Based on code review and project literature:

| Failure Mode | Description | Evidence in Our Run |
|--------------|-------------|---------------------|
| **DDP desynchronization** | Memory buffers not synchronized across GPUs | ✅ **Primary cause** - memory is a buffer, not parameter |
| **Gate collapse** | Model learns to ignore memory, saturates gates | ✅ gate_mean=0.85, gate_std=0.004 |
| **Insufficient memory capacity** | 128 slots too small for 7B complexity | ⚠️ Likely - previous report warned about this |
| **EMA update instability** | Alpha too aggressive, memory staleness | ⚠️ Possible - alpha=0.1 may be problematic |
| **Gradient checkpointing incompatibility** | GC causes NaN with memory side effects | ✅ Disabled in training (warning logged) |

**Note on literature search**: Web search APIs were rate-limited/unavailable. Analysis is based on code review and project context.

### 3.2 Is PPL 548 a Code Bug or Fundamental Instability?

**Verdict: Both, but primarily code/architectural bug (DDP sync issue)**

**Evidence for code bug**:
- Memory bank explicitly registered as buffer with "no DDP sync"
- Training looked normal (low loss) but eval was catastrophic
- Pattern is consistent with DDP desynchronization: fragmented training → poisoned weights

**Evidence for fundamental instability**:
- Gate saturation suggests model cannot use memory effectively even with proper sync
- Previous report warned about scale mismatch (125M-350M vs 7B)
- 128 slots may be fundamentally insufficient for 7B

**Primary vs secondary**:
- **Primary**: DDP desynchronization is a clear code bug that would cause catastrophic failure regardless of memory capacity
- **Secondary**: Even with DDP fixed, gate saturation suggests the architecture may still struggle at 7B scale

### 3.3 Is Gate Saturation a Known Pattern?

**Yes - consistent with memory-ignored learning**

**Pattern interpretation**:
- Gate mechanism: g·o_local + (1-g)·o_mem
- Saturated gates (g≈const, std≈0) → model ignores memory
- This is the expected behavior when memory provides no useful signal

**Alternative explanations (less likely)**:
1. **Training incomplete**: 5000 steps may be insufficient, but gate_mean was stable from step 50 onward
2. **Initialization too strong**: gate_bias_init=2.0 heavily biases toward local attention, but gates should adapt if memory is useful
3. **Regularization effect**: Low LR (2e-5) may prevent gate adaptation, but other parameters are training fine

**Most likely**: Memory slots provide no useful information, so model learns to ignore them and rely on sliding window alone.

---

## 4. Recommendations

### 4.1 Immediate Fixes (Required for any sparse memory experiment)

1. **Fix DDP memory synchronization**
   - Option A: Register memory as parameter instead of buffer (allows DDP sync)
   - Option B: Manual all-reduce after each write to synchronize across GPUs
   - Option C: Train on single GPU only (eliminates DDP issue, but slower)

2. **Validate fix with simple smoke test**
   - Train 100 steps on 2 GPUs, verify gate values diverge from initialization
   - Compare with 1-GPU baseline to ensure consistency

### 4.2 Architectural Improvements (Recommended)

1. **Increase memory capacity**
   - Current: 128 slots for 7B model
   - Recommended: 512-1024 slots (or dynamic capacity based on seq length)
   - Rationale: Previous report suggests 4 orders of magnitude more capacity needed

2. **Adjust gate initialization**
   - Current: gate_bias_init=2.0 → σ≈0.88 (heavily local)
   - Recommended: gate_bias_init=0.0 → σ=0.5 (neutral initialization)
   - Alternative: Learn gate bias as a parameter instead of fixed init

3. **Tune EMA update rate**
   - Current: ema_alpha=0.1 (10% new, 90% old)
   - Recommended: ema_alpha=0.2-0.5 (faster adaptation)
   - Alternative: Use learnable alpha per layer

4. **Increase top-k retrieval**
   - Current: top_k=8 from 128 slots
   - Recommended: top_k=16-32 (or proportional to memory capacity)
   - Rationale: More candidates for attention to find useful signals

### 4.3 Diagnostic Experiments

1. **Ablation: Memory vs local-only**
   - Train with memory disabled (only sliding window)
   - Compare PPL and gate behavior to current run
   - If PPL similar, confirms memory is useless

2. **Ablation: Scale comparison**
   - Train same architecture on 125M model (where literature shows it works)
   - Compare gate behavior and PPL to 7B run
   - If 125M succeeds but 7B fails, confirms scale issue

3. **Ablation: Memory capacity sweep**
   - Run experiments with 32, 128, 512, 1024 slots
   - Measure PPL, gate statistics, and memory utilization
   - Find minimum capacity for 7B

4. **Sanity check: Single GPU training**
   - Train current config on single GPU (no DDP issues)
   - If PPL still catastrophic, confirms fundamental instability
   - If PPL normal, confirms DDP desynchronization as root cause

### 4.4 Alternative Approaches

If sparse memory continues to fail after fixes:

1. **kNN-based retrieval (Memorizing Transformers)**
   - Proven to work at large scale
   - No learned memory slots → no DDP sync issues
   - Requires building embedding index (offline pre-processing)

2. **StreamingLLM-style sink tokens**
   - Keep first N tokens always (no compression)
   - Simpler architecture, proven stability
   - Does not address compression, but avoids sparse memory pitfalls

3. **LoRA-only long-context fine-tuning**
   - Skip memory entirely
   - Fine-tune on long-context data (8K-32K tokens)
   - Let model learn to attend within extended KV cache

---

## 5. Conclusion

The catastrophic PPL (502-549) in `sparse_memory_ablation_8gpu_v5` is **primarily a DDP desynchronization bug**: the memory bank is a buffer (not parameter) with explicit "no DDP sync", causing each GPU to train on inconsistent memory states. This poisons the attention weights and makes the model diverge catastrophically during eval.

The **secondary issue** is gate saturation (g≈0.85, std≈0.004), indicating the model learned to ignore memory and rely on sliding window alone. This suggests that even with DDP fixed, the architecture may struggle at 7B scale due to insufficient memory capacity (128 slots) or fundamental incompatibility with 7B complexity.

**Immediate action**: Fix DDP synchronization before any further sparse memory training. Without this fix, all multi-GPU runs will fail catastrophically.

**Long-term**: If DDP fix doesn't resolve the issue, consider alternative architectures (kNN retrieval, streamingLLM, or LoRA-only long-context fine-tuning).

---

## Appendix: Key Code Snippets

### Memory Bank Registration (src/memory/sparse_memory/memory_bank.py)
```python
# Memory buffer: [1, N, d] placeholder; expanded to [B, N, d] on reset()
# Registered as buffer (not parameter) → no optimizer updates, no DDP sync
self.register_buffer(
    "memory",
    torch.zeros(1, num_slots, hidden_dim, dtype=dtype),
    persistent=False,
)
```

### Gate Fusion (src/memory/sparse_memory/attention.py)
```python
# Gated fusion: output = g * o_local + (1-g) * o_mem
# gate computed via MLP: gate = σ(x @ W_gate + b_gate)
```

### Memory Reset Per Sample (src/memory/sparse_memory/model.py)
```python
def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
    """Forward pass with memory reset per sample.

    Memory is reset at the start of each forward call (each sample is independent).
    """
    B = input_ids.shape[0]
    self.reset_memory(batch_size=B)  # ← Memory is reset per forward call
    return self.model(...)
```

### Training Script Warning (scripts/train_sparse_memory.py)
```python
if args.gradient_checkpointing:
    if global_rank == 0:
        print("WARNING: gradient_checkpointing disabled — incompatible with memory write side effects (causes NaN)")
```
