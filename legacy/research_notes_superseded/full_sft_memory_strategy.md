# Full-Parameter SFT Training Strategy for V4 ChunkMemory

**Date**: 2026-05-03
**Author**: /researcher agent
**Status**: ACTIONABLE -- training should begin within hours
**Target model**: Llama-3-8B (all parameters, no LoRA)
**Hardware**: 8x L20A (183 GiB each) on a single B200 node

---

## 0. Executive Summary

The core finding from our LoRA experiments is devastating: LoRA rank-16 Q/V only training degraded base PPL from 6.70 to 9.08 (+35.6%), and memory bank generation produces garbled text ("`:// Angeles`"). This confirms the prior literature survey conclusion that **LoRA + memory injection has no success precedent at 7B+ scale** (see `memory_injection_lora_survey.md`).

The solution is full-parameter continued pretraining (CPT) with 90% normal pretrain data + 10% memory-bank data. The model's existing weights already know how to do language modeling; they just need to learn to attend to slot prefixes without disrupting that knowledge.

**Recommended action**: Single-phase mixed training, ~2000 steps, learning rate 1e-5, cosine decay, DeepSpeed ZeRO-2 on 8 GPUs.

---

## 1. Problem Diagnosis: Why LoRA Failed

### 1.1 Empirical Evidence

From `outputs/v4_capability_eval/capability_results.json`:
- Base PPL: **6.70** (vanilla Llama-3-8B, no modifications)
- LoRA-only PPL: **9.08** (+35.6% degradation)
- LoRA+Memory PPL: **8.87** (slightly better than LoRA-only, but still terrible)

The garbled generation from memory mode (`"The history of artificial intelligence began in the :// Angeles"`) shows the model's output distribution has been corrupted.

### 1.2 Root Cause

LoRA rank-16 on Q/V only:
1. **Low-rank constraint cannot reshape global attention patterns.** "Attend to memory slot prefixes" requires high-rank changes to attention weight matrices. Rank-16 is ~0.05% of a 4096x4096 weight matrix.
2. **LoRA creates a distribution shift in attention outputs** that propagates through all 32 layers, corrupting the base model's representations.
3. **The model finds a shortcut:** instead of learning to use memory slots, the LoRA adapters shift the attention distribution enough to degrade general NTP quality. Memory slots become noise.

### 1.3 Literature Support

This is consistent with:
- **RMT (Bulatov et al., 2022, 2024)**: All published results use full fine-tuning. Appendix mentions parameter-efficient methods but shows no strong results with them.
- **Block Recurrent Transformer (Hutchins et al., 2022)**: Full fine-tuning from scratch on PG19. No LoRA experiments reported.
- **MemoryLLM (Wang et al., 2024)**: Freezes backbone, only trains memory pool parameters -- a fundamentally different approach that avoids corrupting the base model.

---

## 2. Recommended Training Strategy

### 2.1 Core Principle: Continued Pretraining, Not Fine-Tuning

We frame this as **continued pretraining (CPT)** rather than SFT. The model is already trained; we are adding one capability (reading memory slot prefixes) without changing anything else.

Reference: NVIDIA "Reuse, Don't Retrain" (Satheesh et al., 2024) provides the definitive CPT recipe for LLMs.

### 2.2 Architecture Changes from LoRA Version

The `ChunkMemoryModel` class needs minimal changes:
1. **Remove LoRA wrapping.** All parameters are trainable.
2. **Memory bank operations remain detached** (no gradient through slots).
3. **The forward pass is identical** -- same prefix causal mask, same position embedding extension, same per-layer bank injection.

The only difference: instead of `get_peft_model(base_model, lora_config)`, we keep `base_model` as-is with all parameters unfrozen.

### 2.3 Training Objective

**Pure NTP loss on both data types.** No auxiliary losses.

Rationale:
- Memory bank slots are detached (non-differentiable). The model's only job is to attend to them correctly.
- Adding auxiliary losses (retrieval quality, slot diversity) would complicate the training and risks introducing new failure modes.
- Block Recurrent Transformer and RMT both use pure NTP loss for their memory mechanisms.
- The user's insight is correct: "memory bank doesn't need special data -- we just need the LM to accept memory outputs."

### 2.4 Memory Bank Operations During Training

**Keep slots detached. Do NOT backprop through slot selection.**

Rationale:
1. Slots are running state (like Mamba's hidden state), not parameters.
2. Top-k selection is non-differentiable (argmax). Straight-through estimation would introduce gradient noise.
3. The v4 design doc correctly identifies that "the model needs to learn how to process slot prefixes, not how to generate good slots."
4. EMA update rules are fine-tuned manually (ema_decay=0.9, epsilon=0.05); these do not need gradient optimization.

---

## 3. Data Mixing Strategy

### 3.1 Recommended Approach: Per-Sample Within-Batch Mixing

**90% of samples are pure pretrain data (normal NTP, no memory banks). 10% of samples use memory banks.**

Implementation:
```python
# Per batch:
for sample_idx in range(batch_size):
    if random.random() < 0.1:
        # Memory-bank mode: process document with chunked memory
        process_with_memory_banks(sample)
    else:
        # Normal mode: standard causal LM forward, no memory banks
        process_vanilla_ntp(sample)
```

**Confidence: HIGH**

### 3.2 Why Per-Sample (Not Per-Batch)

- Per-batch mixing (entire batches are either pure or memory) would cause gradient variance spikes. In a memory-bank-only batch, ALL gradients push the model toward "attend to slots," which conflicts with the "normal LM" behavior.
- Per-sample mixing ensures every gradient update contains a weighted combination of "normal LM" and "memory LM" signals, keeping the model anchored to its base capabilities.
- NVIDIA CPT paper shows that mixing data within each batch is critical for preventing distribution shift.

### 3.3 Data Sources

| Source | Path | Shape | Use |
|--------|------|-------|-----|
| SlimPajama (general pretrain) | `data/slimpajama_chunks_4096.npy` | (1,566,247, 4096) | 90% of training data |
| PG-19 (memory-bank documents) | `data/pg19_chunks_llama3.npy` | (5,916, 4096) | Memory-bank samples |

SlimPajama provides ~6.4B tokens of diverse web text (code, papers, books, etc.), which is far more than we need for CPT.

PG-19 books are ideal for memory-bank training because:
- They are long documents that benefit from cross-chunk memory
- We have them pre-tokenized for Llama-3 at 4096 tokens/chunk
- 5,916 chunks = ~740 documents at 8 chunks/doc = ~24M tokens of memory-bank data

### 3.4 Curriculum (Optional, Low Priority)

If time permits, use a 2-phase approach:
1. **Phase A (first 500 steps):** 95% pretrain + 5% memory (gentle introduction)
2. **Phase B (remaining 1500 steps):** 90% pretrain + 10% memory (full mixing)

**Confidence: MEDIUM** for curriculum vs. flat mixing. Start with flat 90/10; if vanilla PPL degrades, add curriculum.

---

## 4. Hyperparameters

### 4.1 Learning Rate Schedule

| Parameter | Value | Rationale | Confidence |
|-----------|-------|-----------|------------|
| Peak LR | **1e-5** | CPT best practice: start at or below pretrain's min LR. Llama-3's min LR was ~1e-5 (standard for 8B). NVIDIA CPT paper starts CPT at eta_min. | HIGH |
| Min LR | **1e-6** | Decay to 10% of peak. NVIDIA paper finds eta_max_ct/100 is optimal for CPT decay floor. | HIGH |
| LR Schedule | **Cosine decay** | NVIDIA CPT paper: cosine decay is strictly better than WSD for CPT. Matches pretrain schedule. | HIGH |
| Warmup | **0 steps** | NVIDIA CPT paper: warmup in CPT causes accuracy regression. Start directly from peak LR. | HIGH |
| Weight decay | **0.1** | Standard for Llama-3 CPT. | HIGH |

### 4.2 Training Duration

| Parameter | Value | Rationale | Confidence |
|-----------|-------|-----------|------------|
| Total steps | **2000** | ~26M tokens with memory + ~234M tokens pretrain = ~260M total tokens. Sufficient for CPT of one capability. | MEDIUM |
| Effective batch size | **128** (16 per GPU x 8 GPUs) | 128 x 4096 = 524K tokens/step. 2000 steps = ~1B tokens total. Reasonable for CPT. | MEDIUM |

### 4.3 Regularization

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Gradient clipping | **1.0** | Standard for full-parameter training. |
| Dropout | **0.0** | Llama-3 uses no dropout in pretrain. Keep same. |
| EMA decay (memory bank) | **0.9** | Same as current v4 config. |

---

## 5. Distributed Training Configuration

### 5.1 DeepSpeed ZeRO Stage

**Recommendation: ZeRO Stage 2**

Memory analysis for Llama-3-8B full-parameter training:

| Component | Size (bf16) |
|-----------|------------|
| Model parameters | ~16 GB |
| Gradients | ~16 GB |
| Optimizer states (AdamW, fp32 copies) | ~64 GB |
| Activations (seq_len=4096, batch=2) | ~8-12 GB per layer |

Per-GPU memory with ZeRO Stage 2 on 8 GPUs:
- Model params: ~16 GB (replicated)
- Gradients: ~2 GB (sharded across 8 GPUs)
- Optimizer states: ~8 GB (sharded across 8 GPUs)
- Activations: ~8-12 GB (with gradient checkpointing)
- NCCL buffers: ~2 GB
- **Total: ~36-40 GB per GPU** (well within 183 GiB L20A)

With ZeRO Stage 3:
- Model params also sharded: ~2 GB more saved
- But adds communication overhead for parameter gathering
- Not necessary given our headroom

**Confidence: HIGH** that ZeRO Stage 2 is sufficient.

### 5.2 Batch Size Calculation

With gradient checkpointing and ZeRO-2:
- Per-GPU micro-batch size: **2** (2 x 4096 = 8192 tokens)
- Gradient accumulation steps: **8**
- Effective batch size: 2 x 8 x 8 GPUs = **128 samples** = 524,288 tokens

This should use ~40-50 GB per GPU. If OOM, reduce micro-batch to 1 and increase gradient accumulation to 16.

### 5.3 Gradient Checkpointing

**Required.** Full backprop through 32 layers with seq_len=4096 would use excessive activation memory without checkpointing.

Enable via: `model.gradient_checkpointing_enable()`

---

## 6. Code Changes Required

### 6.1 Minimal Changes from Current v4

The current `ChunkMemoryModel` (in `scripts/train_v4_chunk_memory.py`) wraps a PeftModel. The full-SFT version needs:

1. **Remove LoRA wrapping:**
   ```python
   # OLD:
   lora_config = LoraConfig(...)
   self.peft_model = get_peft_model(base_model, lora_config)

   # NEW:
   self.model = base_model  # All params trainable
   ```

2. **Update forward pass references:**
   - `self.peft_model.base_model.model.model` -> `self.model.model`
   - `self.peft_model.base_model.model.lm_head` -> `self.model.lm_head`

3. **Add vanilla NTP forward path:**
   ```python
   def forward_vanilla(self, input_ids, labels):
       """Standard causal LM forward, no memory banks."""
       return self.model(input_ids=input_ids, labels=labels)
   ```

4. **Add mixed training loop:**
   ```python
   for batch in dataloader:
       if should_use_memory(batch_idx):  # 10% chance
           reset_banks()
           for chunk in batch.chunks:
               loss = forward_chunk(chunk.input_ids, chunk.labels)
               loss.backward()
       else:
           loss = forward_vanilla(batch.input_ids, batch.labels)
           loss.backward()

       optimizer.step()
       optimizer.zero_grad()
   ```

5. **Use DeepSpeed instead of raw DDP:**
   ```bash
   deepspeed --num_gpus=8 train_v4_full_sft.py --deepspeed ds_config.json
   ```

### 6.2 DeepSpeed Config

```json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "gradient_accumulation_steps": 8
    },
    "gradient_clipping": 1.0,
    "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 2,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1
        }
    },
    "scheduler": {
        "type": "CosineDecay",
        "params": {
            "total_num_steps": 2000,
            "warmup_num_steps": 0,
            "min_lr": 1e-6
        }
    }
}
```

---

## 7. Risk Analysis and Mitigation

### 7.1 Risk: Catastrophic Forgetting Despite Data Mixing

**Severity: MEDIUM**
**Mitigation:**
- 90% pretrain data mixing is the strongest defense (NVIDIA CPT paper confirms this)
- Low learning rate (1e-5) limits weight drift
- Cosine decay to 1e-6 ensures late-training stability
- Monitor vanilla PPL every 100 steps; kill if PPL > 7.5 (base is 6.70)

### 7.2 Risk: Model Learns to Ignore Memory Slots

**Severity: MEDIUM-HIGH**
**Mitigation:**
- This is the expected behavior initially; slots are detached hidden states that look like noise to the untrained model
- The model has no incentive to attend to slots unless they reduce NTP loss
- If slots carry useful information (from earlier chunks in the same document), the model should learn to use them
- **Go/No-Go check:** After 500 steps, memory-bank PPL should be <= vanilla PPL. If memory PPL > vanilla PPL after the model has converged on normal NTP, it means slots are providing no useful signal.

### 7.3 Risk: Memory-Bank Samples Disrupt Batch Normalization

**Severity: LOW**
Llama-3 uses pre-norm (RMSNorm), not batch norm. No batch statistics are affected by mixing sample types.

### 7.4 Risk: Gradient Conflict Between Memory and Vanilla Paths

**Severity: MEDIUM**
**Mitigation:**
- Per-sample mixing ensures gradients from both paths are averaged
- The memory path gradients and vanilla path gradients should be compatible because:
  - Memory path: "Attend to [slots | tokens], predict next token"
  - Vanilla path: "Attend to [tokens], predict next token"
  - The shared "attend to tokens" component dominates
- Gradient clipping (1.0) prevents extreme updates from either path

### 7.5 Risk: Training Takes Too Long

**Severity: LOW**
- 2000 steps at ~30s/step (estimated) = ~17 hours
- This is well within a single weekend run

---

## 8. Evaluation Protocol

### 8.1 During Training (Every 100 Steps)

1. **Vanilla PPL** on WikiText: Must stay < 7.5 (base is 6.70)
2. **Memory-bank PPL** on PG-19 (8-chunk documents): Track convergence
3. **Memory vs Vanilla ratio**: Target < 1.0 (memory should help, not hurt)

### 8.2 After Training

1. **WikiText PPL**: Must be < 7.0 (+5% tolerance from base 6.70)
2. **PG-19 long-document PPL**: Should improve over vanilla on 8+ chunk documents
3. **Generation quality**: No garbled text, coherent completions
4. **NIAH (Needle in a Haystack)**: Test if memory banks enable cross-chunk retrieval

### 8.3 Go/No-Go Criteria

| Checkpoint | Metric | Threshold | Action if Fail |
|------------|--------|-----------|----------------|
| Step 200 | Vanilla PPL | < 7.5 | Kill, investigate |
| Step 500 | Vanilla PPL | < 7.2 | Kill, reduce LR to 5e-6 |
| Step 500 | Memory PPL | < Vanilla PPL | Continue; otherwise diagnose |
| Step 1000 | Vanilla PPL | < 7.0 | Good; continue |
| Step 2000 | Vanilla PPL | < 7.0 | Success |
| Step 2000 | Generation quality | No garbled text | Must pass |

---

## 9. Concrete Next Steps (Ordered by Priority)

1. **Create `train_v4_full_sft.py`** by modifying `train_v4_chunk_memory.py`:
   - Remove LoRA, make all params trainable
   - Add vanilla NTP forward path
   - Add per-sample 90/10 mixing logic
   - Integrate DeepSpeed ZeRO-2 config

2. **Prepare DeepSpeed config** (`configs/ds_zero2_v4.json`)

3. **Create launch script** (`scripts/launch_v4_full_sft.sh`):
   ```bash
   deepspeed --num_gpus=8 train_v4_full_sft.py \
       --model /apdcephfs/.../models/Llama--Llama3-8b \
       --pretrain_data data/slimpajama_chunks_4096.npy \
       --memory_data data/pg19_chunks_llama3.npy \
       --seq_len 4096 \
       --num_slots 8 \
       --top_k 4 \
       --output_dir outputs/v4_full_sft \
       --deepspeed configs/ds_zero2_v4.json
   ```

4. **Launch on a B200 node** (183 GiB x 8 GPUs is plenty for ZeRO-2 + grad checkpointing)

5. **Monitor** every 100 steps. Kill immediately if vanilla PPL > 7.5.

---

## 10. Confidence Levels Summary

| Recommendation | Confidence | Rationale |
|----------------|------------|-----------|
| Full-parameter (no LoRA) | **VERY HIGH** | LoRA results are catastrophic. All literature uses full fine-tuning for memory mechanisms. |
| 90/10 data mixing | **HIGH** | NVIDIA CPT paper confirms data mixing prevents catastrophic forgetting. User's insight is sound. |
| Per-sample mixing (not per-batch) | **HIGH** | Reduces gradient variance; ensures every update balances both objectives. |
| Pure NTP loss (no auxiliary) | **HIGH** | RMT, Block Recurrent Transformer all use pure NTP. Slots are detached; auxiliary losses add complexity without benefit. |
| LR = 1e-5, cosine to 1e-6 | **HIGH** | Standard CPT configuration from NVIDIA paper. Matches Llama-3's original min LR. |
| No warmup | **HIGH** | NVIDIA CPT paper: warmup in CPT causes regression. Start from peak LR directly. |
| ZeRO Stage 2 (not 3) | **HIGH** | Memory analysis shows 36-40 GB per GPU usage, well within 183 GiB. Stage 3 adds unnecessary overhead. |
| 2000 training steps | **MEDIUM** | ~1B tokens is reasonable for one capability. May need adjustment based on early results. |
| Memory bank slots remain detached | **HIGH** | v4 design doc rationale is sound. Slots are running state, not parameters. |
| Batch size = 128 effective | **MEDIUM** | Depends on actual activation memory. May need to reduce to 64 if OOM. |

---

## 11. Literature References

1. **NVIDIA CPT Recipe**: Satheesh et al., "Reuse, Don't Retrain: A Recipe for Continued Pretraining of Language Models", arXiv 2407.07263 (2024). Key findings: start CPT at min LR, no warmup, cosine decay, mix data within batch.

2. **Block Recurrent Transformer**: Hutchins et al., "Block-Recurrent Transformers", NeurIPS 2022 (arXiv 2203.07852). Full fine-tuning on PG19, LSTM-style gates, linear complexity. Uses Adafactor with inverse sqrt decay, 1000 warmup steps.

3. **RMT**: Bulatov et al., "Recurrent Memory Transformer", NeurIPS 2022 (arXiv 2207.06881). Segment-level recurrence with memory tokens. All published results use full fine-tuning. Uses BPTT for multi-segment training.

4. **RMT Scaling**: Bulatov et al., "Beyond Attention: Breaking the Limits of Transformer Context Length with Recurrent Memory", AAAI 2024. Extended RMT to 2M tokens. Full fine-tuning on BERT and GPT-Neo scales.

5. **MemoryLLM**: Wang et al., "MEMORYLLM: Towards Self-Updatable Large Language Models", ICML 2024 (arXiv 2402.04624). Fixed-size memory pool in latent space. Freezes backbone, only updates memory pool parameters. Avoids catastrophic forgetting by design.

6. **Catastrophic Forgetting in CPT**: Ibrahim et al., "Simple and Scalable Strategies to Continually Pre-train Large Language Models", arXiv 2403.08763 (2024). Data replay and LR schedule are key to preventing forgetting.

---

## 12. Comparison: Why This Will Work Better Than LoRA

| Factor | LoRA (Failed) | Full SFT (Proposed) |
|--------|--------------|---------------------|
| Trainable params | ~0.05% (LoRA rank-16) | 100% |
| Attention change capacity | Low-rank only | Full-rank |
| Base PPL after training | 9.08 (+35.6%) | Expected < 7.0 (+5%) |
| Memory slot learning | Cannot reshape attention globally | Can learn any attention pattern |
| Data mixing | Not used (100% memory data) | 90% pretrain + 10% memory |
| LR | 1e-4 (too high for body) | 1e-5 (CPT-appropriate) |
| Forgetting prevention | None (no pretrain data) | 90% pretrain data mixing |

The key insight: we are not trying to teach the model a new task. We are teaching it to tolerate an additional input type (slot prefixes) that it already has the capacity to attend to. Full-parameter training with heavy pretrain data mixing is the right tool for this.
