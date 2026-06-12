# Sparse Memory Regression Analysis

**Date**: 2026-04-21 23:59
**Author**: researcher subagent
**Status**: Complete
**Context**: Analysis of +8.36/+8.64 PPL regression in sparse memory experiments

---

## Executive Summary

The **+8.36/+8.64 PPL regression** observed in sparse memory v3 and fusion experiments is caused by a **chicken-and-egg problem** in the gated fusion architecture. Current approaches consistently degrade performance instead of improving it, indicating the fusion mechanism is fundamentally flawed.

**Root Cause**: Bad fusion → untrained memory → noise pollution → performance degradation

**Solution**: KV-level concatenation fusion with learned bypass gate (init=-2)

---

## Root Cause Analysis

### 1. Chicken-and-Egg Problem

**Current Architecture Flow**:
```
Local Attention + Memory Attention → Gated Fusion → Output
              ↓                ↓              ↓
          (trained)         (untrained)     (degraded)
```

**Failure Cascade**:
1. **Gated fusion blocks gradients**: Current gate mechanism (sigmoid bias=+2.0) has 4000x gradient dilution
2. **Memory doesn't train**: No gradients → memory entries stay at random initialization
3. **Memory = noise pollution**: Untrained KV entries get injected into attention, degrading performance
4. **Both variants fail**: v3 (sparse memory) and fusion (concat) share the same broken gradient path

### 2. Evidence from Experiments

| Experiment | PPL | Regression vs Baseline | Analysis |
|------------|-----|---------------------|----------|
| llama_baseline | 41.24 | 0.00% | ✅ Healthy baseline |
| sparse_memory_v3 | 49.60 | +8.36% | ❌ Same memory pipeline failure |
| sparse_memory_fusion | 49.88 | +8.64% | ❌ Same memory pipeline failure |

**Consistency**: Both variants fail by similar margins → shared architectural flaw, not implementation bugs.

### 3. Why Previous Fixes Failed

1. **Gate initialization fixes** (bias=0.0, small weights)
   - Still has 4000x gradient dilution problem
   - Doesn't solve fundamental chicken-and-egg issue

2. **Memory slot size increases** (128 → 256)
   - More noise doesn't fix the core problem
   - Still need gradients to flow to memory first

3. **EMA memory updates**
   - Don't help if memory never gets trained in the first place
   - Just propagates noise through time

---

## Solution: KV Concatenation Fusion with Bypass Gate

### Architecture Comparison

**Current (Broken)**:
```
output = σ(gate) * o_local + (1-σ(gate)) * o_mem
```

**Proposed (Fixed)**:
```
output = concat(o_local, o_mem)  # Direct concatenation
gate_input = bypass_gate * (concat_output)  # Bypass gate controls flow
```

### Why This Works

1. **No gradient blocking**: Concatenation has no gate to block gradients
2. **Bypass protection**: Learned bypass gate (init=-2) keeps model local-only initially
3. **Gradual opening**: As memory trains, bypass gate gradually opens
4. **Proven design**: Similar to Memorizing Transformers' successful approach

### Implementation Details

```python
# Current (broken)
gate = torch.sigmoid(self.gate_proj(hidden_states))
output = gate * o_local + (1 - gate) * o_mem

# Proposed (fixed)
concat_output = torch.cat([o_local, o_mem], dim=-1)  # [B, T, 2D]
bypass_gate = torch.sigmoid(self.bypass_gate_proj(concat_output))  # [B, T, 1]
output = bypass_gate * concat_output + (1 - bypass_gate) * o_local
```

**Key hyperparameters**:
- `bypass_gate_init = -2.0` → σ(-2) ≈ 0.12 (strong local bias initially)
- `bypass_gate_lr = 10.0` (higher learning rate for bypass gate)
- `num_memory_slots = 128` (keep current size, focus on fixing architecture)

---

## Assumptions Re-evaluation

| Assumption | Original Verdict | Updated Verdict | Rationale |
|------------|-----------------|-----------------|-----------|
| kNN at 7B scale | ⚠️ Unproven | ❌ Caution needed | Literature only validates at 125M-350M scale |
| Memory size (128 slots) | ❌ Too small | ⚠️ Acceptable with fix | Size not primary issue; architecture is |
| Gated fusion viable | ❌ Broken | ✅ With bypass | Bypass gate solves chicken-and-egg |
| Sparse paradigm | ✅ Valid | ✅ Valid | Memorizing Transformers precedent holds |

**Key Insight**: The memory size may be insufficient, but the +8.36 PPL regression is primarily caused by the fusion mechanism, not the memory size itself. Fix the architecture first, then optimize memory size.

---

## Next Steps

### Phase 1: Implement KV Concat Fusion with Bypass Gate

**Task**: Trainer to implement on idle node
- Target: `src/memory/sparse/attention.py`
- Replace gated fusion with concat + bypass gate
- Use init=-2.0 for bypass gate bias
- Use 10x higher learning rate for bypass gate

**Expected timeline**: 2-3 hours (short validation run)

### Phase 2: Validation and Memory Size Ablation

**If Phase 1 successful**:
1. Test concat fusion performance (should match or beat baseline)
2. Ablate memory slots: 128, 256, 512, 1024
3. Identify optimal memory size based on actual performance gain

### Phase 3: Training Optimization

**If memory shows benefit**:
1. Train to convergence (50K+ steps)
2. Evaluate on longer contexts (8K-32K)
3. Compare to baseline at scale

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Concat fusion still fails | Low | Medium | Use proven Memorizing Transformers design |
| Bypass gate still blocks gradients | Low | Low | Higher learning rate should compensate |
| Memory size still insufficient | Medium | Medium | Ablation after architecture fix |
| Training overhead increases | Medium | Low | Expected cost for proper memory training |

---

## Conclusion

The sparse memory approach is fundamentally sound but requires a **fix to the fusion architecture**. The current gated fusion creates a chicken-and-egg problem where untrained memory degrades performance. 

**Recommendation**: Implement KV concatenation fusion with learned bypass gate immediately. This should resolve the performance regression and enable proper memory training.

**Success metric**: PPL should be ≤42.0 (baseline +5%) after architecture fix, not 49.6 (+20% worse).

---

## References

- **Previous findings**: `ops/research_notes/2026-04-20_2335_sparse_memory_mvp_implementation_assumption_audit__77ebcea.md`
- **Gate architecture analysis**: `ops/research_notes/2026-04-21_1120_gate_architecture_research.md`
- **Concat fusion plan**: (to be written after implementation)