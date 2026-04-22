# Inflection Point Assessment: What Worked, What Didn't, and What's Next

**Date**: 2026-04-21 02:18 GMT+8
**Author**: researcher subagent
**Context**: Slot Memory Stage 3 complete, Sparse Memory MVP scored 0% at 32K, all compute idle

---

## 1. What We Learned

### 1.1 Slot Memory Stage 3: Trained Successfully, recon=0.0 by Design

**Training details:**
- 10 epochs, 452 steps/epoch, ~4,520 total steps
- Loss: 2.23 → 2.08 (steady decline, healthy convergence)
- `recon=0.0000` on every log line — **this is correct**: Stage 3 is CE-only (`lambda_recon=0.0`)
- This was intentional: 3-stage curriculum (recon → joint → CE-only)
- Checkpoint at `outputs/slot_memory_8gpu_stage3_.../final/`

**Previous eval (from post-analysis note):** 91.7% NIH accuracy vs 100% base model — 8.3% degradation, no improvement anywhere. One config (ctx2048_d0.25) scored 0%, likely an eval bug.

### 1.2 Sparse Memory MVP: 0% at 32K — Untrained, Frozen Base

- Used untrained sparse memory on frozen Qwen3-8B base model
- 0/9 on NIH-Extended at 32K context (base model also fails at 32K)
- Memory slots cannot encode useful information without training
- 4 bugs were found and fixed (RoPE, flash_attn padding, batch conflicts, lr_lambda leak)
- Training was being set up (PG-19 data rsynced to remote node) but hasn't started

### 1.3 RMT v10 Remote Cluster: Completed but Results Unknown Locally

- All 4 nodes report `status: completed` (l0, l0l1, l0l2, l0l1l2)
- Results are on the remote nodes at `/root/Mixture-of-Memory/outputs/rmt_v10_*`
- Previous local RMT v10 eval already showed 0% NIH — remote variants (L1/L2 injection layers) are architectural variants, not fundamentally different
- **Given prior RMT v10 = 0% locally, remote results are unlikely to change the picture.**

### 1.4 NIH Benchmark Situation

- Original NIH (1K-4K): Base model = 100%, too easy to measure any benefit
- NIH-Extended (32K): Base model also fails, but sparse memory untrained MVP also fails
- NIH-Hard (64K, 15 needles, 98% depth): Created but base model eval results unknown
- **Core problem persists**: We lack a benchmark where (a) base model fails AND (b) a trained memory model can demonstrate improvement

---

## 2. Direction Assessment

| Direction | Status | Verdict | Rationale |
|-----------|--------|---------|-----------|
| **RMT v1-v10** (cross-attention memory tokens) | Complete failure | ❌ **Cut** | 0% NIH across all versions. Training converges but inference produces garbage. Memory tokens treated as noise. |
| **RMT v10 remote variants** (L1/L2 injection) | Completed, unknown results | ❌ **Cut** | Likely similar failure — same fundamental issue (memory tokens disrupt causal attention). |
| **Slot Memory** (3-stage training) | Trained, -8.3% vs base | ⚠️ **Deprioritize** | Less destructive than RMT but no benefit. Only 4,520 steps vs literature recommending 50K+. May work with much longer training. |
| **Sparse Memory** (kNN bank + gated fusion) | MVP at 0% (untrained) | ⚠️ **Most Promising** | 0% is expected for untrained model. Architecture is sound (literature-backed at small scale). Needs training + evaluation. |
| **Long-context LoRA fine-tuning** (no memory) | Not attempted | ✅ **Strong baseline** | Proven in literature. Skip memory entirely, just extend context via LoRA. Quick win for comparison. |

---

## 3. Recommendation: Concrete Next Steps

### Priority 1: Get the benchmark right (1-2 hours)

1. **Check NIH-Hard base model results** — the eval was launched but results may be ready
2. If NIH-Hard works (base model fails at 64K), use it as the evaluation benchmark
3. If not working, fix and run it — this is the critical blocker

### Priority 2: Train Sparse Memory on remote cluster (immediate)

Sparse Memory is the **only remaining promising direction** because:
- It doesn't insert foreign tokens into the attention stream (avoids RMT's failure mode)
- It's inference-only modification (memory bank is external), not training-only
- Literature (Memorizing Transformers) validates the approach at smaller scale
- The architecture is simple and debuggable

**Action**: Launch sparse memory training on the 4 idle remote L20A nodes:
- Data: PG-19 (already rsynced)
- Run 5K-10K step sanity check first
- Use NIH-Hard for periodic evaluation during training

### Priority 3: Quick baseline — Long-context LoRA (same day)

Before investing more in memory mechanisms, establish a **simple baseline**:
- Fine-tune Qwen3-8B with LoRA on 8K-32K context data
- No memory mechanism at all
- If this alone solves the long-context problem, memory compression is unnecessary
- If it doesn't, we have a proper comparison point

### Priority 4: Fetch remote RMT results (low effort)

SSH into one node, check if L1/L2 variants show any improvement over L0 baseline. Expected: no. But worth 5 minutes to confirm.

### Priority 5: Evaluate Slot Memory checkpoint on NIH-Hard (optional)

The slot memory checkpoint exists and might perform better at 64K context where base model fails. Worth running if NIH-Hard benchmark is ready. But don't invest more compute in slot memory training until we see if it helps where it matters.

---

## 4. What Evaluation to Run on Slot Memory Checkpoint

**Only if NIH-Hard benchmark is ready.** The standard NIH eval already showed -8.3% degradation and is uninformative.

Specific eval:
```
python scripts/eval_nih_hard.py \
  --model outputs/slot_memory_8gpu_stage3_.../final/ \
  --context_lengths 32768 65536 \
  --num_needles 15 \
  --depth 0.98
```

Compare against base model on same benchmark. If slot memory ≥ base model at 64K, it's worth further investigation. If still worse, cut.

---

## 5. Key Risk: Training Scale

All our memory experiments have been **1-2 orders of magnitude under-trained** compared to literature recommendations:

| Experiment | Our Steps | Literature Recommendation |
|------------|-----------|--------------------------|
| RMT v1-v10 | 84-1,356 | 50K-200K |
| Slot Memory | ~4,500 | Unknown (novel architecture) |
| Sparse Memory | 0 (not trained yet) | N/A |

**If we proceed with sparse memory, we must commit to 10K+ steps minimum**, not the 500-step sanity checks we've been running. Under-training has been our #1 systematic error.

---

## Summary

| Question | Answer |
|----------|--------|
| Did slot memory work? | No measurable benefit. Trained fine but doesn't help. |
| Did sparse memory work? | Can't tell — untrained model scored 0% at 32K (expected). |
| Should we cut RMT? | Yes. 10+ versions, all 0%. |
| Should we cut slot memory? | Deprioritize. Run NIH-Hard eval only, no more training. |
| What should we do next? | (1) Fix/validate NIH-Hard benchmark, (2) Train sparse memory on remote cluster, (3) Run long-context LoRA baseline. |
| What's the biggest mistake so far? | Consistently under-training (100-1000x fewer steps than literature). |
