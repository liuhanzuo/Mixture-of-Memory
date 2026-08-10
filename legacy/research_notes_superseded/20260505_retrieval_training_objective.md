# Retrieval Training Objectives for Cross-Attention Memory Modules
**Date:** 2026-05-05
**Scope:** How to make cross-attention memory capable of explicit fact retrieval, not just PPL improvement
**Context:** CrossAttentionMemoryV2 achieves PPL ratio 0.9947 (0.53% improvement) but 0% NIAH retrieval accuracy

---

## Executive Summary

**Core finding:** PPL-only training (next-token cross-entropy on PG-19) is fundamentally insufficient for training a memory module to perform explicit fact retrieval. Every successful memory-augmented model in the literature that achieves retrieval uses one or more of: (1) synthetic retrieval tasks as training signal, (2) specialized gating/mechanisms that create explicit key-value addressing, (3) curriculum learning over progressively harder retrieval tasks, (4) BPTT through segment boundaries.

**Recommendation:** Add an auxiliary retrieval loss alongside PPL training, using NIAH-style synthetic data where the model must retrieve a specific key-value pair from earlier in a streamed sequence. This is the minimal viable intervention and matches the approach that worked for ARMT, RMT, and Infini-attention.

---

## 1. The PPL vs. Retrieval Gap: Why It Exists

### 1.1 What PPL training actually teaches the memory

When training CrossAttentionMemoryV2 with next-token cross-entropy on PG-19:

1. **The gradient signal is diffuse.** The loss is averaged over all T tokens in the chunk. The gradient from any single token's prediction is distributed across all slots proportionally to the softmax weights. No single slot receives a strong "store this exact fact" signal.

2. **The optimal strategy for PPL is statistical smoothing.** To minimize perplexity on generic text, the best strategy is to make the memory carry aggregate statistics about the distribution (topic, style, common collocations) rather than storing any specific fact. This is because most next-token predictions benefit from context-level statistics, not from retrieving one specific earlier token.

3. **Slot addressing degenerates under PPL.** Our own diagnosis (20260427_niah_v8_diagnosis.md) confirmed: "The training loss (PG-19 next-token CE) doesn't reward needle-style addressing. There is no gradient signal in training that would teach the selector to exactly address the slot holding the needle."

4. **The write mechanism learns to average, not to store.** The delta-rule write `slot_values += write_lr * (avg_weights^T @ hidden_states)` averages hidden states weighted by attention. Under PPL loss, this converges to storing the "average context" per slot, not a specific (key, value) pair.

### 1.2 What retrieval requires that PPL does not provide

Explicit fact retrieval requires:
- **Precise addressing:** The query "The secret number for agent X is" must address the exact slot that contains the needle, not just any semantically similar slot.
- **Content-addressable storage:** The write mechanism must create a binding between the key ("agent X") and the value ("12345") in the same slot or tightly coupled slots.
- **Gradient signal for exact match:** The model must receive gradient when it retrieves the wrong value, not just when the next-token distribution is suboptimal.

---

## 2. Survey of Successful Approaches

### 2.1 ARMT (Associative Recurrent Memory Transformer)

**Paper:** "Associative Recurrent Memory Transformer" (Bulatov et al., 2023)

**Architecture:**
- Segment-level recurrence with special memory tokens at segment boundaries
- Associative memory mechanism using linear attention with DPFP-3 non-linearity: `sigma(x) = (1 + x / ||x||) ^ p`
- Memory matrix `M` updated via delta-rule with gamma-correction: `M_new = gamma * M + sigma(K)^T * (V - sigma(K) @ M / sigma(K) @ z)`
- Explicit key-value separation: K and V projections are separate from the backbone's Q/K/V

**Training approach (critical):**
1. **Pre-training on synthetic associative retrieval tasks** before any LM training:
   - Tasks: (a) key-value pair memorization and recall, (b) associative retrieval from a list of pairs
   - Data format: `[key1, val1, key2, val2, ..., query_key, ?]` → model must output `query_val`
   - Curriculum: start with 2-4 pairs, increase to hundreds
2. **Then** fine-tune on downstream tasks (LM, QA) with the memory mechanism active
3. BPTT through segment boundaries (not truncated)

**Results:** 79.9% accuracy at 50M token context length on passkey retrieval; strong performance on LongBench.

**Key insight:** The memory was specifically trained to do associative recall. The DPFP-3 non-linearity creates sharper attention than softmax (more peaked addressing), and the pre-training task explicitly teaches the model "store this pair, retrieve it later."

### 2.2 LM2 (Large Memory Models)

**Paper:** "LM2: Large Memory Models" (Sun et al., 2025)

**Architecture:**
- Cross-attention memory with LSTM-style gating:
  - Input gate `i = sigma(W_i * [x; h_prev])`: controls what to write
  - Forget gate `f = sigma(W_f * [x; h_prev])`: controls what to erase
  - Output gate `o = sigma(W_o * [x; h_prev])`: controls what to read
- 2048 memory slots per layer
- Pre-trained from scratch (not adapted from existing model)

**Training approach:**
- Standard next-token prediction on large corpus
- The LSTM-style gating naturally creates content-addressable memory:
  - Forget gate learns to clear irrelevant slots
  - Input gate learns to write to specific slots based on content
  - Output gate learns to read from relevant slots

**Results:** 6.7% improvement on MMLU over baseline; outperforms RMT by 37.1% on long-context tasks.

**Key insight:** The LSTM gating mechanism creates a differentiable, content-addressable memory without requiring auxiliary retrieval tasks. The forget/input/output gates provide explicit write/read/erase control that softmax cross-attention lacks. However, this requires pre-training from scratch.

### 2.3 RMT (Recurrent Memory Transformer)

**Paper:** "Recurrent Memory Transformer" (Bulatov et al., 2022-2023)

**Architecture:**
- Special [MEM] tokens inserted at segment boundaries
- Segment-level recurrence via BPTT through [MEM] tokens
- No explicit memory mechanism: the [MEM] tokens simply carry a compressed representation

**Training approach (critical):**
1. **Curriculum learning on passkey retrieval:**
   - Stage 1: 2 segments, find a random "passkey" number embedded in text
   - Stage 2: 4 segments, same task
   - Stage 3: 8, 16, 32 segments progressively
   - Each stage: model must output the passkey at the end
2. Loss: cross-entropy only on the passkey tokens (not the whole sequence)
3. After curriculum, fine-tune on downstream tasks

**Results:** Length generalization up to 50M tokens (2Mx extrapolation from training length of 2048).

**Key insight:** The retrieval task must be the primary training signal, not auxiliary. The model learns to store and retrieve because the loss explicitly requires it. Curriculum learning is essential: the model cannot learn 32-segment retrieval directly.

### 2.4 Memorizing Transformer

**Paper:** "Memorizing Transformer" (Wu et al., 2022)

**Architecture:**
- kNN-augmented attention: at each layer, attend to both local context AND a kNN index of past key-value pairs
- Differentiable retrieval: the kNN index supports gradient flow
- Learned gating: `output = alpha * local_attn + (1-alpha) * knn_attn`

**Training approach:**
- Standard next-token prediction
- No auxiliary retrieval task needed because kNN retrieval is inherently content-addressable

**Results:** Consistent PPL improvement on PG-19, WikiText-103; scales with memory size.

**Key insight:** kNN is the "gold standard" for retrieval because it guarantees exact nearest-neighbor lookup. The price is O(N) memory and O(log N) retrieval time (with FAISS), which is expensive. The differentiable gating allows the model to learn when to use memory vs. local context.

### 2.5 Infini-attention

**Paper:** "Infini-attention" (Munkhdalai et al., 2024)

**Architecture:**
- Linear attention memory: `M = sigma(K)^T @ V` where `sigma(x) = ELU(x) + 1`
- Delta-rule update: `M_new = M + sigma(K)^T @ (V - sigma(K) @ M / sigma(K) @ z)`
- Compressive: memory size is O(d^2) per head, independent of sequence length
- Learned beta gate: `output = sigmoid(beta) * mem_output + (1-sigmoid(beta)) * local_output`

**Training approach:**
- Primarily evaluated on passkey retrieval and book summarization
- Passkey retrieval trained with explicit supervision on the passkey tokens
- Curriculum learning: start with short sequences, increase length

**Results:** Passkey retrieval at 1M context length; PPL improvement on long documents.

**Key insight:** Linear attention memory is inherently associative (stores key-value bindings via outer product). The delta-rule provides self-correction. But like ARMT, it needs explicit retrieval training signal.

### 2.6 Titans

**Paper:** "Titans: Learning to Memorize at Test Time" (Behrouz et al., 2024)

**Architecture:**
- Neural long-term memory: a deep MLP that acts as the memory module
- Surprise-based update: `M.update(x)` is weighted by the "surprise" (gradient magnitude) of x
- Three variants: MAC (Memory as Context), MAG (Memory as Gate), MAL (Memory as Layer)

**Training approach:**
- Next-token prediction, but the memory module is a learned neural network
- The surprise-based update mechanism naturally focuses memory writes on unexpected/informative tokens
- The MLP memory is more expressive than associative matrix memory

**Results:** Competitive with Transformers on language modeling; strong on associative recall benchmarks.

**Key insight:** Making the memory itself a neural network (not just a matrix) gives it the capacity to learn complex retrieval patterns. Surprise-based updates naturally prioritize rare/informative content.

### 2.7 D-RAG (Differentiable Retrieval-Augmented Generation)

**Paper:** "D-RAG: End-to-End Differentiable Retrieval-Augmented Generation" (2024)

**Architecture:**
- Differentiable retrieval index: document embeddings are learned, retrieval scores are soft
- `p(doc|query) = softmax(sim(query_embed, doc_embed) / tau)` with temperature tau
- End-to-end training: gradients flow from generation loss through retrieval to document encoder

**Training approach:**
- Standard generation loss, but the retrieval is differentiable
- The model learns to retrieve relevant documents and to generate from retrieved content
- No auxiliary retrieval loss needed because the retrieval is part of the forward pass

**Key insight:** Making retrieval differentiable and trainable allows the model to learn what to retrieve. The soft retrieval (softmax over all documents) provides gradient signal to the document encoder, teaching it to create retrievable representations.

---

## 3. Comparison Table

| Method | Memory Mechanism | Training Signal | Retrieval Task | BPTT | Pre-train from Scratch | Retrieval Accuracy | Scale |
|--------|-----------------|-----------------|----------------|------|----------------------|-------------------|-------|
| **ARMT** | Linear attn + DPFP-3, delta-rule | Synthetic associative retrieval, then LM | Explicit (key-value recall) | Yes | No (adapts existing) | 79.9% at 50M ctx | 1.3B |
| **LM2** | Cross-attn + LSTM gates | Next-token CE only | Implicit (via gates) | No | Yes (from scratch) | +37.1% over RMT | 1.7B |
| **RMT** | Segment recurrence via [MEM] tokens | Curriculum passkey retrieval | Explicit (passkey) | Yes | No (adapts existing) | 100% at 2K, extends to 50M | 350M-1.3B |
| **Memorizing TF** | kNN index + learned gating | Next-token CE only | Implicit (kNN) | No | No (adapts existing) | Consistent PPL gain | 252M-1.3B |
| **Infini-attention** | Linear attn matrix + delta-rule | Passkey retrieval + LM | Explicit (passkey) | No | No (adapts existing) | 100% at 1M ctx | 1B |
| **Titans** | Deep MLP (neural memory) | Next-token CE + surprise update | Implicit (surprise) | No | No (adapts existing) | Competitive with TF | 1.3B |
| **D-RAG** | Differentiable soft retrieval | Generation loss through retrieval | Implicit (differentiable) | No | No (adapts existing) | Strong on RAG tasks | 7B |
| **Our CrossAttnV2** | Cross-attn Q=hidden K/V=slots, delta-rule | Next-token CE only (PG-19) | None | No | No (adapts Llama-3) | 0% NIAH | 8B |

### Key patterns from the table:

1. **Every method with explicit retrieval task achieves non-trivial retrieval.** ARMT, RMT, and Infini-attention all use synthetic retrieval as primary or co-training signal.
2. **Methods without explicit retrieval task rely on specialized mechanisms.** Memorizing TF uses kNN (inherently exact retrieval); LM2 uses LSTM gates (inherently content-addressable); Titans uses neural memory (inherently expressive).
3. **Our CrossAttentionMemoryV2 has neither.** Standard cross-attention + delta-rule + PPL-only training = no retrieval capability.

---

## 4. Recommended Approach (Ranked by Feasibility)

### Tier 1: Minimal viable intervention (recommended first)

#### 4A. Add NIAH-style retrieval loss alongside PPL training

**What:** During training, mix in NIAH-style samples where a (key, value) pair appears early in the stream and the model must retrieve the value at the end. Add a retrieval-specific loss that supervises the answer tokens.

**Implementation:**
```python
# Data: NIAH samples mixed with PG-19 at ratio ~10-20%
# Each NIAH sample: stream of text chunks + needle "The secret code for agent X is 12345"
# Last chunk: question "What is the secret code for agent X?" -> answer "12345"

# Loss:
total_loss = lm_loss + lambda_retrieve * retrieval_loss

# lm_loss: standard CE on all tokens (existing)
# retrieval_loss: CE on answer tokens only, weighted more heavily
#   - Only computed on NIAH samples
#   - Uses the SAME positions that the NIAH accuracy metric checks
#   - Forces the memory to store and retrieve the needle

# lambda_retrieve: start at 1.0, tune
```

**Why this works:**
- Creates explicit gradient signal for "store this fact, retrieve it later"
- Does not require architecture changes
- Compatible with existing training loop (just add data + loss)
- Matches the approach used by ARMT and RMT (synthetic retrieval as co-training)

**Estimated effort:** 2-3 days (data pipeline + loss implementation + integration)

**Risk:** Low. Even if retrieval accuracy is still low, this cannot hurt PPL because PPL loss is retained.

#### 4B. Curriculum learning on retrieval task difficulty

**What:** After 4A is implemented, add curriculum learning:
- Stage 1: Needle in the last chunk (trivial: memory not needed, just local attention)
- Stage 2: Needle in the second-to-last chunk (1-step retrieval)
- Stage 3: Needle 2-4 chunks back
- Stage 4: Needle at arbitrary depth (full NIAH)

**Why:** Directly following RMT's curriculum approach. Prevents the model from being overwhelmed by the full NIAH task.

**Estimated effort:** 1 day (on top of 4A)

### Tier 2: Architecture modifications (if Tier 1 insufficient)

#### 4C. Add explicit key-value addressing to CrossAttentionMemoryV2

**What:** Modify the write mechanism to create explicit (key, value) bindings:
- During write, compute a "key" for each hidden state (via a learned projection)
- Compute slot-key similarity to determine which slot to write to
- Write the hidden state only to the top-1 or top-3 most similar slots
- This makes retrieval content-addressable: query key matches write key

**Implementation sketch:**
```python
# In CrossAttentionMemoryV2.write():
# Instead of delta-rule (avg attention weights):
write_keys = self.write_key_proj(hidden_states)  # [B, T, d_key]
slot_keys_expanded = slot_keys  # [B, N, d_key]

# Compute which slot each hidden state should write to
write_sim = torch.bmm(write_keys, slot_keys_expanded.transpose(1, 2))  # [B, T, N]
write_weights = F.softmax(write_sim / tau, dim=-1)  # sharp addressing

# Write: only update the most similar slot(s)
# (hard top-1 with STE, or soft top-k)
delta = torch.bmm(write_weights.transpose(1, 2), hidden_states)  # [B, N, d_model]
updated_slots = slot_values + write_lr * delta
```

**Why this might help:** The current write mechanism (delta-rule from read attention weights) is "passive" -- it writes based on where the READ attention happens to land. Explicit key-addressed writing creates a content-addressable store where the write is deterministic based on the content.

**Estimated effort:** 3-5 days

**Risk:** Medium. Could interact badly with PPL loss if the key projection learns degenerate patterns.

#### 4D. Replace softmax attention with linear attention (ARMT-style)

**What:** Replace the softmax in CrossAttentionMemoryV2.read() with a DPFP-3 kernel or ELU+1 kernel (Infini-attention style). This makes the memory associative: `M = sigma(K)^T @ V`, and retrieval becomes `sigma(Q) @ M`.

**Why:** Softmax attention normalizes across slots, which creates competition between slots. Linear attention accumulates evidence without competition, which is better for associative recall.

**Estimated effort:** 5-7 days (significant rewrite of the read/write mechanism)

**Risk:** High. Changing the attention mechanism fundamentally changes how the memory interacts with the backbone.

### Tier 3: Major architectural changes (if Tiers 1-2 insufficient)

#### 4E. Replace cross-attention with LSTM-gated memory (LM2-style)

**What:** Replace CrossAttentionMemoryV2 with an LSTM-gated memory module:
- Input gate, forget gate, output gate
- Each gate is conditioned on [hidden_state, current_memory]
- Input gate controls writing, forget gate controls erasure, output gate controls reading

**Why:** LSTM gates provide explicit, differentiable control over what to store, what to erase, and what to retrieve. This is more expressive than cross-attention for retrieval.

**Estimated effort:** 1-2 weeks (new module, re-training)

**Risk:** High. Requires re-training from scratch or careful adaptation. LSTM gates may not scale well to 8B parameters.

#### 4F. kNN-augmented attention (Memorizing Transformer style)

**What:** Add a kNN index on top of the memory slots. During read, query the kNN index to find the top-k most similar slots, then attend only to those.

**Why:** kNN provides exact nearest-neighbor retrieval (no softmax blurring). Combined with a learned gating mechanism, this guarantees that the model CAN retrieve exact facts.

**Estimated effort:** 1-2 weeks (FAISS integration, gating mechanism, training)

**Risk:** Medium-High. kNN adds latency and complexity. May not integrate cleanly with the streaming/chunked training setup.

---

## 5. Concrete Implementation Plan for Tier 1 (4A + 4B)

### 5.1 Data pipeline

**Existing infrastructure:** The project already has `NIAHIterableDataset` in `scripts/niah_dataset.py` that generates NIAH-style samples with:
- Haystack text (PG-19 chunks)
- Needle ("The secret code for agent <name> is <code>")
- Question at the end
- Labels for the answer tokens

**Modifications needed:**
1. In `train_mem_space_pg19.py`, increase `niah_mix_fraction` from 0.10 to 0.30-0.50 for the retrieval training run
2. Ensure NIAH samples have sufficient depth variation (needle at different positions in the stream, not always near the end)
3. Add curriculum: start with `niah_mix_fraction=0.10` and needle in the last 2 chunks, gradually increase distance

### 5.2 Loss function

**Current loss:** `loss = out.loss` (standard CE over all tokens, handled by HuggingFace)

**Proposed loss:**
```python
# For NIAH samples:
# 1. Standard CE loss (existing) -- trains the LM backbone
# 2. Retrieval-specific loss: CE on answer tokens only, with higher weight

if is_niah:
    # out.loss already includes the answer token supervision
    # But we want to UPWEIGHT the answer tokens specifically
    # Option A: Modify labels so non-answer positions are -100 (only supervise answer)
    #           This is too aggressive -- loses LM signal for NIAH chunks

    # Option B: Compute a separate loss on answer tokens and add it
    answer_mask = (labels != -100)  # [B, T] bool, True at answer positions
    if answer_mask.any():
        # Get logits shifted by 1 (causal LM convention)
        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_answer_mask = answer_mask[..., 1:].contiguous()

        # CE only on answer positions
        answer_logits = shift_logits[shift_answer_mask]  # [n_answer, V]
        answer_labels = shift_labels[shift_answer_mask]  # [n_answer]

        retrieval_loss = F.cross_entropy(answer_logits, answer_labels)
        total_loss = out.loss + lambda_retrieve * retrieval_loss
    else:
        total_loss = out.loss
else:
    total_loss = out.loss
```

**lambda_retrieve:** Start at 5.0-10.0 (heavily weight the retrieval signal). Tune based on whether PPL degrades.

### 5.3 Training schedule

**Phase 1 (steps 0-2000):** Warmup with PPL-only on PG-19
- Memory module learns basic representation (as in current training)
- `niah_mix_fraction=0.0`

**Phase 2 (steps 2000-5000):** Add NIAH with curriculum
- `niah_mix_fraction=0.20`
- Needle always in the last 2 chunks (easy retrieval)
- `lambda_retrieve=10.0`

**Phase 3 (steps 5000-10000):** Increase difficulty
- `niah_mix_fraction=0.30`
- Needle at any depth (full NIAH)
- `lambda_retrieve=5.0` (reduce to balance with PPL)

**Phase 4 (steps 10000+):** Joint training
- `niah_mix_fraction=0.15`
- Full difficulty
- `lambda_retrieve=2.0`

### 5.4 Diagnostics to add

1. **Separate NIAH loss logging:** Log `lm_loss` and `retrieval_loss` separately
2. **Per-step NIAH accuracy:** Using the fixed metric (after the off-by-one fix from 20260428)
3. **Attention weight diagnostics:** Log the attention entropy during NIAH samples (should decrease over training, indicating the model learns to focus on specific slots)
4. **Slot write magnitude:** Track how much each slot changes per write (should become more peaked/focused during NIAH training)

### 5.5 Expected outcomes

**Optimistic:** NIAH accuracy reaches 30-50% within 5000 steps of Phase 2. This would indicate the memory module is learning to store and retrieve specific facts.

**Realistic:** NIAH accuracy reaches 5-15%. Some retrieval capability emerges, but the delta-rule write mechanism may not be precise enough for high accuracy. This would motivate Tier 2 modifications.

**Pessimistic:** NIAH accuracy stays at 0%. The cross-attention + delta-rule mechanism fundamentally cannot support precise retrieval even with explicit supervision. This would indicate the need for Tier 3 (architectural replacement).

---

## 6. Theoretical Analysis: Why Tier 1 Should Work

### 6.1 Gradient flow analysis

Currently, the gradient for the memory module comes from:
```
d(loss)/d(memory_output) = d(CE_all_tokens)/d(out_proj) * d(out_proj)/d(attn_output)
```

This gradient is the SUM of gradients from ALL T tokens' next-token predictions. The gradient from any single token is small relative to the total. The answer token's gradient is 1/T of the total, diluted by T-1 irrelevant tokens.

With the retrieval loss:
```
d(total_loss)/d(memory_output) = d(CE_all)/d(memory) + lambda * d(CE_answer_only)/d(memory)
```

The second term provides a MUCH stronger gradient signal specifically targeting the memory's ability to retrieve the answer. The gradient from `CE_answer_only` backpropagates through:
1. `out_proj` -> activates the answer token logit
2. `attn_weights` -> forces the query to attend to the correct slot
3. `q_proj` -> shapes the query to be content-addressable
4. Via the write path (if write is not detached): shapes the slot content

### 6.2 Alignment with successful precedents

ARMT pre-trains on associative retrieval BEFORE any LM training. Our approach adds retrieval loss DURING LM training. This is weaker (the retrieval signal competes with PPL signal) but more practical (no need to change the training pipeline drastically).

RMT uses curriculum learning on passkey retrieval as the ONLY training task. Our approach uses it as an auxiliary task. Again weaker, but avoids the need to re-train the entire model.

The question is whether the auxiliary retrieval signal is strong enough to override the PPL-trained "statistical smoothing" tendency. The lambda_retrieve parameter controls this tradeoff.

### 6.3 Potential failure modes for Tier 1

1. **Catastrophic interference:** The retrieval loss could disrupt PPL optimization, causing PPL to degrade significantly. Mitigation: start with low lambda_retrieve, monitor PPL closely.

2. **Slot collision:** Multiple (key, value) pairs get written to the same slot. With 128 slots and typical NIAH settings (1 needle in ~20K tokens), this is unlikely for a single needle, but could be an issue for multi-needle variants.

3. **Write-path detachment:** Currently, the write path uses DETACHED hidden_states (`no_grad`). This means the retrieval loss cannot directly shape what gets written to slots. It can only shape the READ path (q_proj, k_proj, v_proj, out_proj). The write path shapes itself only through the PPL loss on subsequent chunks. This is a significant limitation and may be the reason Tier 1 fails.

**Critical fix needed:** Enable gradient flow through the write path, at least for NIAH samples. This means NOT detaching hidden_states in the write call for NIAH chunks. This creates BPTT-like gradient flow: retrieval loss -> read attention -> slot values -> write delta -> hidden_states at write time.

### 6.4 The write-path gradient issue (important)

Looking at the current code in `CrossAttentionMemoryV2.write()`:
```python
# Uses DETACHED hidden_states for write-back (no gradient through write path).
```

This is a major obstacle. For retrieval training to work:
1. The READ path must learn to address the correct slot (q_proj, k_proj)
2. The WRITE path must learn to store the needle in a findable slot

If the write path has no gradient, only (1) gets trained. The write path continues to do "statistical averaging" regardless of the retrieval loss. The retrieval loss can only train the read path to find the best match among whatever averaged representations happen to be in the slots.

**Recommendation:** For NIAH samples, do NOT detach hidden_states during write. Accept the higher memory cost (BPTT through the write path) for the NIAH chunks. This is a small fraction of total training steps (~20%), so the memory overhead is manageable.

---

## 7. Summary of Recommendations

| Priority | Action | Effort | Expected Impact | Risk |
|----------|--------|--------|----------------|------|
| **1** | Fix NIAH metric off-by-one bug (already diagnosed) | 0.5 day | Correct visibility | None |
| **2** | Add retrieval loss (Section 5.2) | 1-2 days | Essential signal | Low |
| **3** | Enable write-path gradient for NIAH samples | 0.5 day | Critical for retrieval | Low (memory overhead) |
| **4** | Curriculum learning (Section 5.3) | 1 day | Stabilizes training | None |
| **5** | Diagnostic logging (Section 5.4) | 0.5 day | Observability | None |
| **6** | If Tier 1 fails: explicit key-addressed write (4C) | 3-5 days | Major | Medium |
| **7** | If Tier 1+2 fails: architectural replacement (4E/4F) | 1-2 weeks | Transformative | High |

**Total effort for Tier 1 (items 1-5): 3-4 days**

---

## 8. Key Literature References

1. **ARMT:** Bulatov et al., "Associative Recurrent Memory Transformer" (2023). Core idea: associative memory + synthetic retrieval pre-training.
2. **LM2:** Sun et al., "LM2: Large Memory Models" (2025). Core idea: LSTM-gated cross-attention memory, pre-trained from scratch.
3. **RMT:** Bulatov et al., "Recurrent Memory Transformer" (2022-2023). Core idea: segment recurrence + curriculum passkey retrieval.
4. **Memorizing Transformer:** Wu et al., "Memorizing Transformer" (2022). Core idea: kNN-augmented attention with learned gating.
5. **Infini-attention:** Munkhdalai et al., "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention" (2024). Core idea: linear attention memory with delta-rule.
6. **Titans:** Behrouz et al., "Titans: Learning to Memorize at Test Time" (2024). Core idea: neural long-term memory with surprise-based updates.
7. **D-RAG:** "End-to-End Differentiable Retrieval-Augmented Generation" (2024). Core idea: differentiable soft retrieval.
8. **Block Recurrent Transformer:** Hutchins et al. (2022). Core idea: block-level recurrence with cross-attention memory.
9. **Perceiver IO:** Jaegle et al. (2021). Core idea: cross-attention as a general-purpose read/write mechanism.
10. **Delta-rule:** Munkhdalai & Yu (2017), Schlag et al. (2020). Core idea: self-correcting memory update that stores only the residual.
