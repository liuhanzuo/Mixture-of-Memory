# Debug Report: RMT v7 Inference Pipeline — NIH 0% Accuracy

**Date:** 2026-04-18  
**Author:** coder subagent  
**Trigger:** NIH needle-in-haystack eval: RMT 0% vs base model 100%

---

## Executive Summary

**The inference pipeline code is correct.** No bugs found in attention mask construction, position IDs, memory injection, or mask format conversion. The 0% accuracy is caused by a **training methodology defect**: CE-only loss does not train the memory mechanism to store or retrieve specific facts.

The model can minimize training loss through the LoRA-adapted backbone alone, without utilizing memory tokens for information transfer between segments.

---

## Priority 1: Inference Pipeline Debug

### Test Setup
- Single NIH test case: needle "The secret code for project abcdef is 739184" at position 1285 of 1501 tokens
- 2 segments (padded to 2048), needle in segment 1
- Question: "What is the secret code for project abcdef?"
- Expected: "739184"

### Results

| Check | Result | Status |
|-------|--------|--------|
| Memory tensor extraction | Non-zero, std~1.0, no NaN/Inf | ✅ OK |
| Memory injection into next segment | Correctly concatenated as prefix | ✅ OK |
| Attention mask: text→memory | All True (bidirectional) | ✅ OK |
| Attention mask: memory→all | All True | ✅ OK |
| Attention mask: text causal | Matches `torch.tril` | ✅ OK |
| Position IDs (seg 0) | [0..63] for memory, [0..1023] for text | ✅ OK |
| Position IDs (seg 1) | [0..63] for memory, [1024..2047] for text | ✅ OK |
| Mask format for Qwen3 | Additive (0.0 / -inf), passed via dict | ✅ OK |
| Question segment: memory prepended | Verified via `torch.allclose` | ✅ OK |

### Base Model vs RMT Comparison

| Mode | Input Length | Output | Correct? |
|------|-------------|--------|----------|
| Base model (full context) | 1528 tokens | "739184\n\n..." | ✅ |
| Base model (short context) | ~50 tokens | "739184\n\n..." | ✅ |
| RMT (2 segments + question) | 91 tokens (64 mem + 27 question) | "01\nAnswer:01..." | ❌ |

### Step-by-step RMT Output

```
Step 0: token=220 (' '), top5=[220, 20678, 3555, 715, 35127]
Step 1: token=15 ('0'), top5=[15, 220, 16, 18, 17]
Step 2: token=16 ('1'), top5=[16, 18, 15, 17, 22]
Step 3: token=198 ('\n'), top5=[198, 271, 12, 15, 7948]
Step 4: token=16141 ('Answer'), top5=[16141, 15, 220, 16, 21806]
```

The model has no idea what the answer is. It outputs a random digit ("01") then enters a degenerate loop repeating "Answer:01".

---

## Priority 2: Memory Module Weights

### Weight Analysis

| Parameter | Shape | Mean | Std | All Zero? | Concern |
|-----------|-------|------|-----|-----------|---------|
| `memory_embeddings` | [7, 64, 4096] | 0.0001 | 0.023 | No | Normal init range |
| `extractor.memory_queries` | [64, 4096] | -0.0001 | 0.020 | No | Near init |
| `extractor.q/k/v_proj.weight` | [256, 4096] | ~0 | 0.009 | No | Normal init |
| `extractor.out_proj.weight` | [4096, 256] | 0.000 | 0.036 | No | Normal init |
| `extractor.norm.weight` | [4096] | 1.0 | 0.0 | No (all 1.0) | LayerNorm unchanged |
| **`importance_mlp.2.weight`** | **[1, 1024]** | **0.0** | **0.0** | **⚠️ YES** | Last layer never learned |
| **`importance_mlp.2.bias`** | **[1]** | **0.0** | — | **⚠️ YES** | Never learned |
| `memory_predictor.in_proj_weight` | [12288, 4096] | ~0 | 0.011 | No | Near init |
| `memory_predictor.in_proj_bias` | [12288] | 0.0 | 0.0 | ⚠️ YES | Zero init (normal) |
| `memory_predictor.out_proj.bias` | [4096] | 0.0 | 0.0 | ⚠️ YES | Zero init (normal) |
| `segment_bias.weight` | [7, 64] | 0.062 | 0.999 | No | Has learned variation |

### Key Weight Observations

1. **`importance_mlp` last layer is completely zero** — importance scores are stuck at sigmoid(0)=0.5 for all slots. The gate formula: `gate = 0.1 + 0.9 * (1 - sigmoid(importance))` → gate = 0.55 for all slots. This means uniform 55%/45% new/old blending, no selectivity.

2. **`memory_predictor` weights barely moved from init** — std ~0.009-0.011 is consistent with initial random weights for a 4096-dim linear layer (expected init std ≈ 1/√4096 ≈ 0.016). The Z-forcing predictor was never actually trained (v7 used CE-only loss).

3. **`extractor.norm` weight is exactly 1.0** — LayerNorm hasn't been updated, suggesting the extractor's output path has minimal gradient flow.

4. **`memory_embeddings` has low std (0.023)** — these are the initial memory vectors. They're close to their random init value, meaning initial memory doesn't carry much information.

---

## Root Cause Analysis

### The Core Problem: CE-Only Loss ≠ Memory Utilization

**The training loss (CE on next-token prediction) can be minimized entirely by the LoRA-adapted backbone, without using memory tokens at all.**

Here's why:

1. **Within each segment**, the backbone processes `[memory_tokens | segment_tokens]` and predicts next tokens for `segment_tokens`. The CE loss only measures prediction quality of segment tokens.

2. **The backbone has full causal access to all segment tokens** (1024 tokens of context). A Qwen3-8B model with LoRA can easily predict next tokens within a 1024-token window — it doesn't need memory to do this.

3. **Memory tokens sit at positions 0-63**, attended to bidirectionally. But since the backbone already has 1024 tokens of causal context, it can ignore memory without hurting CE loss.

4. **Memory extraction runs after the forward pass**, producing compressed representations. But nothing in the loss function measures whether these representations are useful for retrieval.

### Why Training Loss Decreased (3.09 → 2.11)

The loss decrease came from:
- LoRA adapting the backbone to the training data distribution (Chinese Wikipedia)
- The model learning to predict next tokens better within 1024-token windows
- **Not from the memory mechanism becoming functional**

### Why 0% NIH Accuracy

1. Needle information is in segment 1's 1024 tokens
2. After processing, 64 memory tokens try to "summarize" segment 1
3. But training never required memory to preserve specific facts
4. When the question arrives, the 64 memory vectors don't contain "739184"
5. The model guesses randomly → 0% accuracy

---

## Recommended Fixes

### Fix 1: Add Memory Retrieval Loss (Critical)
Add a loss term that directly measures whether memory preserves retrievable information:
```python
# After extracting memory from a segment containing a known fact:
# Use a small readout head on memory to predict the fact
memory_readout_loss = predict_fact_from_memory(memory, known_facts)
```

### Fix 2: Add Reconstruction Loss
Require memory to reconstruct key segment statistics:
```python
# Memory should be able to reconstruct segment hidden states
recon = decoder(memory)  # [B, M, D] -> [B, T_reduced, D]
recon_loss = MSE(recon, segment_hidden[:, ::stride, :])
```

### Fix 3: Enable Z-forcing (Already Implemented but Unused)
The `MemoryPredictor` module exists but is never called in v7 training. Enable it:
- Memory should predict next-segment summaries
- Creates gradient pressure for memory to carry forward useful information

### Fix 4: Importance Gate Training
The `importance_mlp` last layer is all zeros. Either:
- Initialize with small non-zero values (e.g., `nn.init.normal_(w, 0, 0.01)`)
- Add a loss term that differentiates importance across slots
- Or remove importance gating entirely (simpler is better at this stage)

### Fix 5: Consider Removing Memory Tokens from LM Forward
Current approach inserts memory tokens into the backbone's attention window, wasting 64 positions of context. Alternative: compute memory *after* backbone forward (pure compression, no backbone modification needed). This would also eliminate the need for custom attention masks.

---

## Inference Pipeline Code Quality Assessment

The inference code is **correct and well-structured**:
- `build_rmt_attention_mask()`: Correct boolean mask → additive conversion
- `build_rmt_position_ids()`: Correct global positions across segments
- `RMTModel._forward_single_segment()`: Correct Qwen3 dict-mask format
- `eval_nih.py`: Correct segment processing + greedy generation
- Training script: Correct segment iteration, memory flow, mask construction

**No code bugs found. The issue is purely in training methodology.**

---

## Files Examined
- `src/memory/rmt/rmt_module.py` — RMT module, correct
- `scripts/eval_nih.py` — NIH eval script, correct
- `scripts/train_rmt_v7.py` — Training script, correct but missing memory-specific loss
- `scripts/debug_rmt_inference.py` — Debug script created for this analysis
- `outputs/rmt_v7_*/final/rmt_memory.pt` — Memory weights, loaded and analyzed
- `outputs/rmt_v7_*/final/config.json` — Model config
- Qwen3 modeling code (transformers) — Verified mask handling
