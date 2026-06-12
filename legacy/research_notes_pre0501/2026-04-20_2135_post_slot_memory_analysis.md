# Post Slot Memory Analysis: Is This Direction Working?

**Date**: 2026-04-20 21:35
**Author**: researcher subagent

## Executive Summary

After RMT v1-v10 all achieved 0% NIH accuracy, we attempted SlotMemoryCompressor with 3-stage training (reconstruction → joint → CE-only). The slot memory approach completed successfully but provides **no measurable benefit over base model** on NIH needle-in-haystack task. The remote RMT v10 cluster experiments also failed catastrophically. **NIH benchmark is too easy** (base model achieves 100%), making it impossible to properly evaluate memory compression effectiveness.

**Key Finding**: Neither RMT-style memory tokens nor slot memory compression provides demonstrable benefit on current evaluation benchmarks. We need either (a) harder benchmarks or (b) a fundamentally different approach.

---

## 1. RMT v10 Cluster Results: Complete Failure

### 1.1 Configuration
- **Memory tokens**: 64 (vs v1-v9's 8-32)
- **Segment length**: 1024 tokens
- **Max segments**: 6 (max context: 6K tokens)
- **Bottleneck dim**: 256 (reduced from 512)
- **Training**: 20 epochs, 8 GPUs, 10.5 hours
- **Loss trajectory**: 3.109 → 2.254 (decreased steadily)
- **Data**: 10K Chinese Wikipedia docs

### 1.2 Training Behavior
- **Loss decreased steadily** from 3.11 (epoch 0) to 2.25 (epoch 20)
- **Retrieve loss** (auxiliary supervision): 0.014 → 0.008 (also decreased)
- **Training converged**: No NaN, no divergence, clean loss curve

**Interpretation**: Training was successful by standard metrics. The model learned something.

### 1.3 NIH Evaluation: 0% Accuracy (Catastrophic)

| Config | Accuracy | Total Tests | Pass/Fail |
|--------|----------|-------------|-----------|
| **Base model** | **100%** (45/45) | 45 | ✅ |
| **RMT v10** | **0%** (0/45) | 45 | ❌ |

**All RMT v10 test cases failed across all configurations**:
- Context lengths: 1024, 2048, 4096
- Needle depths: 10%, 30%, 50%, 70%, 90%
- All 3 trials per configuration

**What does RMT v10 output look like?**
```json
{
  "needle": "The secret code is XAJI0Y.",
  "expected": "XAJI0Y",
  "answer": " a string of 1000000000 from the secret code from the document\nfrom the document\nfrom the document\n",
  "is_correct": false
}
```

The model hallucinates repetitive nonsense ("from the document") instead of retrieving the code. This suggests memory tokens are being treated as noise rather than useful compressed context.

### 1.4 Root Cause Analysis

**Why did RMT v10 fail despite successful training?**

1. **Base model doesn't know how to use memory tokens**
   - RMT inserts new token types into the attention sequence
   - Pre-trained attention patterns assume standard input tokens
   - Without explicit supervision on "how to attend memory," model treats them as noise

2. **NIH is too easy to expose RMT's value**
   - Base model: 100% NIH accuracy (6K context fits in KV cache)
   - RMT: 0% NIH accuracy (memory tokens degrade performance)
   - If both models should theoretically perform perfectly, RMT's failure is about **architecture disruption**, not compression quality

3. **Memory compression quality may be poor**
   - Mean pooling + MLP bottleneck (256 dim → 64 tokens)
   - Aggressive 16:1 compression (1024 → 64 tokens)
   - Loss decreased but doesn't prove compression is semantically useful

**Verdict**: RMT v10 training succeeded, but NIH evaluation reveals fundamental issues with the approach.

---

## 2. Slot Memory Results: Better Than RMT, But No Benefit

### 2.1 Configuration
- **Architecture**: SlotMemoryCompressor (16 slots, 256-dim)
- **Training**: 3-stage (reconstruction → joint → CE-only)
  - Stage 1: Reconstruction loss only
  - Stage 2: Joint reconstruction + CE
  - Stage 3: CE-only (final stage)
- **Segment length**: 1024, **max segments**: 4
- **Training time**: Stage 3 alone: 2.6 hours, 10 epochs

### 2.2 Training Behavior (Stage 3, CE-only)

| Epoch | Loss | Trend |
|-------|------|-------|
| 0 | 2.23 | Initial |
| 3 | 2.16 | Small improvement |
| 6 | 2.12 | Continued improvement |
| 9 | 2.08 | Final loss |

Loss decreased modestly (2.23 → 2.08), suggesting learning occurred.

### 2.3 NIH Evaluation: 91.7% Overall (No Benefit Over Base)

| Model | Accuracy | vs Base |
|-------|----------|---------|
| **Base model** | **100%** (~1080/1080) | baseline |
| **Slot memory** | **91.7%** (990/1080) | -8.3% ❌ |

**Breakdown by configuration**:
- **11/12 configs**: 100% accuracy
- **1/12 configs** (ctx2048_d0.25): 0% accuracy ⚠️ (suspicious, possible bug)
- **No configuration showed >100% (improvement over base)**

**Key observation**: Slot memory performs **worse** than base model overall (91.7% vs 100%). This is consistent with RMT: memory mechanisms add noise rather than providing benefit.

### 2.4 The Suspicious 0% Cell (ctx2048_d0.25)

One configuration failed completely:
- Context length: 2048
- Depth: 25%
- All trials: 0% accuracy

**Possible explanations**:
1. **Bug in evaluation pipeline** for this specific config
2. **BPTT depth issue** (bptt_depth=2 in training, may not generalize to longer ctx)
3. **Segment boundary artifacts** at specific context lengths

**Recommendation**: Debug this cell to understand if it's a systemic issue or an eval bug.

### 2.5 Comparison: Slot Memory vs RMT v10

| Metric | RMT v10 | Slot Memory | Winner |
|--------|---------|-------------|--------|
| NIH accuracy | 0% (0/45) | 91.7% (990/1080) | **Slot memory** ✅ |
| Training success | ✅ (loss ↓) | ✅ (loss ↓) | Tie |
| vs base | -100% | -8.3% | **Slot memory** (less bad) |

**Verdict**: Slot memory is **less destructive** than RMT, but still provides **no benefit** over base model.

---

## 3. Core Problem: NIH Benchmark Is Too Easy

### 3.1 Base Model Performance
- **NIH accuracy**: 100% (1080/1080 tests)
- **Context capability**: Handles 4096+ tokens without memory compression
- **Interpretation**: Base model already exceeds what NIH tests

### 3.2 What This Means

**If base model already achieves 100% NIH, then:**

1. **Any memory mechanism that deviates from 100% is actively harmful**
   - RMT v10: 0% → catastrophic
   - Slot memory: 91.7% → harmful (degrades performance)

2. **Memory compression cannot demonstrate benefit on NIH**
   - The benchmark doesn't need memory compression to pass
   - We cannot measure "how much better" compression helps, because base is already perfect

3. **We need harder benchmarks** where base model fails

### 3.3 Alternative Interpretation: Memory Mechanisms Add Noise

**Alternative theory**: Both RMT and slot memory add noise to the base model's forward pass, degrading performance on a task where the base model already excels.

**Evidence**:
- RMT inserts bidirectional memory tokens into causal forward pass
- Slot memory modifies attention dynamics
- Both approaches deviate from pure causal LM
- NIH is sensitive to attention dynamics (needle retrieval requires precise attention)

**Counter-evidence**: If memory compression were truly useful, it should help on tasks where base model fails. We haven't tested those yet.

---

## 4. Literature-Based Reality Check

From RESEARCH_LITERATURE.md analysis:

### 4.1 Original RMT (Bulatov et al. 2023)
- **Successful configurations**:
  - Segment length: 2048
  - Memory tokens: 64-128
  - Compression ratio: 16:1-32:1
  - Training: 10-20 epochs, 50K-200K steps

- **Our RMT v10**:
  - ✅ Memory tokens: 64 (aligned)
  - ✅ Epochs: 20 (aligned)
  - ❌ Segments: 6 max (should be 8+)
  - ❌ Total steps: ~800 (vs 50K-200K recommended)
  - ❌ Compression ratio: 16:1 (aggressive, but okay)

**Verdict**: Our RMT v10 is under-trained (800 steps vs 50K-200K recommended). This may explain poor compression quality.

### 4.2 Training Duration Insights
Literature suggests **tens to hundreds of thousands of steps** for learning memory compression. Our experiments:

| Experiment | Steps | Verdict |
|------------|-------|---------|
| RMT v1 | 84 | ❌ 1-2 orders too small |
| RMT v2 | 282 | ❌ 1-2 orders too small |
| RMT v3 | 1356 | ❌ 1-2 orders too small |
| RMT v10 | ~800 | ❌ 1-2 orders too small |

**Key insight**: We're training for **100-1000x fewer steps** than successful baselines.

### 4.3 Memory Token Count
Original RMT finds optimal memory tokens at **5-10% of segment length**:

| Segment length | Optimal memory tokens | Our config |
|----------------|----------------------|-------------|
| 1024 | 64-128 | 16 (v4), 64 (v10) |
| 2048 | 128-256 | N/A |

RMT v10's 64 tokens for 1024 segments is **aligned with literature**.

---

## 5. Diagnosis and Recommendations

### 5.1 Why Did Our Experiments Fail?

**Primary reasons**:

1. **Training duration insufficient**
   - 800 steps (RMT v10) vs 50K-200K (literature)
   - Memory compression is a complex skill requiring many gradient updates

2. **Evaluation mismatch**
   - NIH is too easy (base model: 100%)
   - Cannot measure compression benefit when baseline is perfect

3. **Architecture disruption**
   - RMT inserts bidirectional memory into causal forward
   - Slot memory modifies attention patterns
   - Base model doesn't know how to use new token types

**Secondary reasons**:

4. **Data quality/quantity**
   - 10K docs may be insufficient for learning robust compression
   - Need diverse data: technical, narrative, dialogue, code

5. **Compression ratio**
   - 16:1 (1024 → 64 tokens) is aggressive
   - May be losing critical information

### 5.2 What Should We Do Next?

#### Option A: Continue Improving Slot Memory

**Pros**:
- Less destructive than RMT (91.7% vs 0% NIH)
- 3-stage training worked (completed successfully)
- Slot attention is more interpretable than RMT tokens

**Cons**:
- No benefit over base model
- NIH cannot measure improvement
- Training is expensive (2.6h per 10-epoch stage)

**Required improvements**:
1. **Train longer**: 50K-100K steps (not 800)
2. **Increase memory slots**: 16 → 64-128 (match literature)
3. **Harder benchmarks**: See Option C below
4. **Debug 0% cell**: Fix ctx2048_d0.25 issue

**Effort**: High (requires massive compute + new benchmarks)

---

#### Option B: Try a Fundamentally Different Approach

**Alternative directions to explore**:

1. **StreamingLLM-style sink tokens**
   - Instead of compressing all context, keep first few tokens always
   - Simpler architecture, less disruption to base model
   - Proven to work for long-context inference

2. **kNN-based memory (Memorizing Transformers)**
   - Don't learn compression; retrieve from pre-built index
   - Offline: Build embedding index on large corpus
   - Online: kNN lookup + attention injection
   - Proven strong performance

3. **Hierarchical compression (Multi-level RMT)**
   - L0 (recent): Full attention
   - L1 (medium): RMT compression
   - L2 (long): Coarse summary
   - More complex but matches project's original vision

4. **Pure LoRA long-context fine-tuning**
   - Skip memory tokens entirely
   - Fine-tune LoRA on long-context data (4K-16K tokens)
   - Let model learn to attend within extended context
   - Proven approach (long LoRA works well)

**Pros of Option B**:
- Addresses architecture disruption issue
- Leverages proven methods from literature
- May be faster to implement than fixing current approach

**Cons of Option B**:
- Loses progress on current direction
- Requires implementing new architectures
- Unknown which alternative will work best

**Effort**: Medium (depends on alternative chosen)

---

#### Option C: Develop Harder Benchmarks First

**Critical insight**: We cannot evaluate memory compression without benchmarks where base model fails.

**Benchmark requirements**:
1. **Long context** (8K-32K tokens) where base model degrades
2. **Need retrieval from early context** (like NIH, but harder)
3. **Multi-hop reasoning** (need to connect information across segments)
4. **Quantitative metric** (not just pass/fail)

**Candidate benchmarks**:

| Benchmark | Context Length | Base Performance | Notes |
|-----------|----------------|------------------|-------|
| **Needle-in-Haystack** | 4K | 100% | ✅ Too easy |
| **NIH-Extended** | 8K-32K | Unknown | ⭐ Try this first |
| **Passkey Retrieval** | 10K+ | Unknown | Similar to NIH |
| **NarrativeQA** | Long stories | ~50-70% | Requires comprehension |
| **HotpotQA (multi-hop)** | 2-4K | ~40-60% | Requires connecting facts |
| **PG19 books QA** | Book-length | Unknown | Long document understanding |
| **Custom synthetic tasks**: Generate long documents with structured questions that require early context recall |

**Implementation plan**:

1. **NIH-Extended (immediate)**
   - Reuse NIH infrastructure
   - Extend context to 8K, 16K, 32K tokens
   - Add multiple needles (not just one)
   - Add distractor content that mimics needle

2. **Custom long-context QA (1-2 days)**
   - Generate synthetic documents (10K-50K tokens)
   - Embed "secret facts" at random positions
   - Questions require retrieving those facts
   - Test at 1K, 4K, 8K, 16K, 32K context lengths

3. **NarrativeQA / HotpotQA (2-3 days)**
   - Download existing benchmark datasets
   - Adapt to our evaluation pipeline
   - Compare base vs memory-augmented models

**Pros of Option C**:
- Essential regardless of direction
- Low effort to start (NIH-Extension is trivial)
- Provides proper evaluation for all future experiments

**Cons of Option C**:
- Delays "real" progress
- Benchmark development takes time
- May reveal that even harder benchmarks are easy for base model

**Effort**: Low (NIH-Extended) to Medium (custom benchmarks)

---

### 5.3 Recommended Action Plan

**Immediate (next 1-2 days)**:

1. ✅ **Implement NIH-Extended** (Option C, first priority)
   - Context lengths: 8K, 16K, 32K (add to existing 1K, 2K, 4K)
   - Multi-needle variants (2-5 needles per document)
   - Run base model evaluation first (establish baseline degradation curve)

2. ✅ **Debug slot memory 0% cell**
   - Investigate ctx2048_d0.25 failure
   - Fix eval bug or identify training issue

3. ✅ **Survey kNN memory approaches** (Option B exploration)
   - Read Memorizing Transformers paper (detailed implementation notes)
   - Assess feasibility: Can we build embedding index on our corpus?

**Short-term (next week)**:

4. If NIH-Extended reveals base model degrades at 16K+:
   - **Re-train slot memory** with:
     - 50K+ steps (not 800)
     - 64-128 slots (not 16)
     - Target: Improve on 16K+ context where base fails
   - **Or switch to kNN memory** if it's easier to implement

5. If NIH-Extended is still too easy (base >90% even at 32K):
   - **Focus on custom multi-hop benchmarks** (Option C)
   - Tasks requiring connecting facts across 10K+ tokens
   - Only after benchmarks expose base model limits should we train memory models

**Long-term (2-4 weeks)**:

6. **Multi-level compression (L0→L1→L2)** if single-level memory proves insufficient
7. **Architecture ablation study**: Compare RMT, slot memory, streamingLLM, kNN on same benchmarks
8. **Production evaluation**: Test on real long-context tasks (document analysis, long-form QA)

---

## 6. Final Verdict

### 6.1 Current Direction Assessment

| Direction | Status | Verdict |
|-----------|--------|---------|
| **RMT v1-v10** | ❌ 0% NIH | Failed completely |
| **Slot memory** | ⚠️ 91.7% NIH | No benefit over base |
| **Base model** | ✅ 100% NIH | Exceeds NIH requirements |

**Key conclusion**: Current memory compression approaches do not provide measurable benefit on existing benchmarks. This is partially because benchmarks are too easy, but also because training duration is insufficient (1-2 orders of magnitude).

### 6.2 Critical Path Forward

**Do not** continue training memory models without harder benchmarks.

**Do**:
1. Implement NIH-Extended (8K-32K context, multi-needle)
2. Establish where base model actually degrades
3. Only then re-approach memory compression (either improved slot/RMT or alternative like kNN)

**Why this order?**
- Without knowing where base model fails, we cannot measure memory compression benefit
- Training memory models blindly on "easy" benchmarks wastes compute
- Harder benchmarks are orthogonal to architecture choice

### 6.3 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| NIH-Extended is still too easy | Medium | High wasted effort | Add custom synthetic benchmarks |
| kNN memory requires large compute for index | High | Medium | Use subset of corpus first |
| Slot memory requires 50K+ steps to converge | High | High compute | Train incrementally, evaluate at 10K, 20K, 50K |
| Multi-level compression too complex | Medium | High development time | Start with single-level, add L2 later |

---

## 7. References

### 7.1 Experiment Outputs
- Slot memory NIH: `outputs/slot_memory_nih_eval_20260420_205759/nih_results.json`
- Slot memory training: `outputs/slot_memory_8gpu_stage3_20260420_164731_*/`
- RMT v10 NIH: `outputs/nih_eval_v10/nih_results.json`
- RMT v10 training: `outputs/rmt_v10_8gpu_20260419_001626_*/`

### 7.2 Research Notes
- 2026-04-19 direction survey: `ops/research_notes/2026-04-19_1407_direction_survey_post_rmt_failure.md`
- 2026-04-17 G-MemLLM analysis: `ops/research_notes/2026-04-17_2140_gmemllm_gating_analysis__4ce1dc0.md`
- RESEARCH_LITERATURE.md (extensive literature review)

### 7.3 Literature
- Bulatov et al. (2023). Recurrent Memory Transformer (RMT)
- Wu et al. (2022). Memorizing Transformers (kNN-based memory)
- Rae et al. (2019). Compressive Transformer
- Dai et al. (2019). Transformer-XL
- Xiao et al. (2023). StreamingLLM (sink tokens)
- Liu et al. (2024). MemWalker (memory selection)
