# RESEARCH PLAN REVISION - 2026-04-21 05:17 UTC

## 🚨 CRITICAL CHANGE: Slot Memory Architecture FAILED

**Original Plan (pre-failure)**: 3-experiment sequence (RULER → sparse scale → comparison)
**Reality Now**: Slot memory proven dead (0% retention across segments)

## Revised Research Path

### **Experiment 1: RULER CWE Benchmark Migration** (Priority 0 - UNCHANGED)
- Implement RULER CWE evaluation at 64K for Qwen3-8B  
- Establish baseline with 20-40% diagnostic headroom
- **Status**: Hard benchmarks running, results pending
- **Critical**: Still required for any memory evaluation

### **Experiment 2: Sparse Memory Scale Ablation** (Priority 1 - CHANGED)  
- Test N ∈ {512, 1024, 2048, 8192} with full fine-tuning
- 50K+ steps, BPTT depth=2 (literature-validated protocol)
- **Duration**: 1-2 weeks, 4-8 GPUs
- **Pivot criterion**: If N=2048 + full FT fails → ABANDON kNN → ARMT
- **NEW**: Compare against streaming alternatives (ARMT/R3Mem/CMT)

### **Experiment 3: Multi-Method Comparison** (Priority 2 - COMPLETELY REVISED)
- **OLD**: Compare baseline, slot memory, sparse memory
- **NEW**: Compare baseline, sparse memory, ARMT, R3Mem, CMT
- **Duration**: 2-3 weeks, 8-16 GPUs (parallel evaluation)
- **Focus**: Validate which approach actually works at 7B scale

## Key Changes
1. **Remove slot memory** - proven fundamentally broken
2. **Add streaming alternatives** - ARMT/R3Mem/CMT ready for implementation  
3. **Parallel evaluation** - test multiple approaches simultaneously
4. **Earlier pivot point** - decide after scale ablation, not after comparison

## Timeline: 3-4 weeks total (same duration, different content)

## Implementation Priority
1. Complete hard benchmarks (establish evaluation framework)
2. Start sparse scale ablation + ARMT implementation in parallel
3. Evaluate all working approaches on diagnostic benchmarks

## Risk Mitigation
- Multiple approaches tested in parallel
- Early failure detection for kNN vs streaming approaches
- Clear pivot criteria based on empirical results