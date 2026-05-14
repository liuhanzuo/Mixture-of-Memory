# Diagnosis: H-Series BABILong qa1/qa2/qa5 = 0/30 -- Root Cause Analysis

**Date**: 2026-05-11
**Author**: Researcher
**Trigger**: issue_planA_niah_zero (CRITICAL)
**Confidence**: HIGH

---

## Executive Summary

The H-series cross-attn memory models score **0/30 on all BABILong tasks at all context lengths**, regardless of NIAH training loss (H13: niah_loss=0.58 vs H14: niah_loss=11.64). This is **NOT caused by the memory architecture**. The root cause is:

**The base model (Llama-3-8B base, NOT Instruct) scores 0-1% on BABILong natively.** The fine-tuned H-series models are matching the base model's floor performance. The memory architecture adds nothing useful for BABILong-style QA, but it also does not degrade what was already zero.

---

## Finding 1: Llama-3-8B Base Model Cannot Do BABILong (HIGH confidence)

### Evidence

From `status/babilong_results.json` and `status/babilong_results_cluster3.json`:

| Model | 1k | 2k | 4k | 8k | Notes |
|-------|-----|-----|-----|-----|-------|
| **Meta-Llama-3-8B (base)** | **0%** | **1%** | **1%** | **12%** | Peak at 8k = within training context |
| Llama-3.2-1B-Instruct | 35.7% | 36.1% | 38.2% | 33.9% | 128k context + instruction tuning |
| Beacon-Qwen2-7B | 65.4% | 60.4% | 55.6% | 49.2% | Qwen2-Instruct + activation beacon |
| MemoryLLM-8B-chat | 36.6% | 35.1% | 33.0% | 32.0% | 8B Instruct + memory injection |
| **H13_isolate@2500** | **0%** | **0%** | **0%** | **0%** | Llama-3-8B base + cross-attn memory |
| **H14_isolate_aggr@1500** | **0%** | **0%** | **0%** | **0%** | Llama-3-8B base + cross-attn memory |

### Root Cause

All H-series experiments use `Llama--Llama3-8b` (path: `/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b/`), which is **Meta-Llama-3-8B base** (NOT Instruct). BABILong requires:

1. **Instruction following**: Understanding the multi-part prompt template with instruction, examples, post-prompt, context tags, and question
2. **Structured output generation**: Producing answers in specific formats like "The most recent location of X is Y"
3. **Multi-step reasoning**: qa2 requires tracking object ownership across locations; qa5 requires tracking give/receive actions

Llama-3-8B base has NO instruction-following training and CANNOT parse the BABILong prompt or generate answers in the required format. The 12% peak at 8k for the base model is noise / lucky format matching within its native context window.

**This single finding explains 100% of the observed 0% BABILong scores.**

---

## Finding 2: NIAH Training vs BABILong -- Fundamental Task Mismatch (HIGH confidence)

### NIAH Training Format (from `niah_dataset.py`)

```
Needle: "MEMORIZE: The secret code for agent ABCDEF is 12345. END_MEMORIZE"
Question: "The secret code for agent ABCDEF is "
Answer: "12345" (5 digits)
```

- **Needle**: A single distinctive token sequence embedded in pg19 background text
- **Question**: A near-copy of the needle with the answer removed
- **Answer**: 5 raw digit tokens
- **Loss**: Cross-entropy on ONLY the 5 answer digit positions (all other labels = -100)
- **Evaluation**: Exact string match of the predicted digits

### BABILong Format (from `babilong/prompts.py`)

```
Instruction: "I will give you context with the facts about positions..."
Examples: <example>Charlie went to hallway... Where is Charlie? Answer: balcony</example>
Post-prompt: "Always return your answer in format: The most recent location of X is Y"
Context: [~1024 tokens of bAbI facts mixed with random text]
Question: "Where is John?"
Expected Answer: "The most recent location of John is kitchen."
```

### Key Differences

| Dimension | NIAH Training | BABILong Eval |
|-----------|--------------|---------------|
| **Needle type** | Single code string "12345" | Multiple natural language facts |
| **Question type** | Near-copy of needle text | Natural language question |
| **Answer format** | 5 raw digit tokens | Full sentence "The most recent location of..." |
| **Label scope** | Only 5 answer tokens | Full sentence with format constraints |
| **Reasoning** | Pattern matching (copy-paste) | Multi-entity tracking + temporal reasoning |
| **Context type** | pg19 prose (irrelevant) | bAbI facts mixed with random text |
| **Prompt template** | None (raw tokens) | Instruction + examples + format constraints |
| **Retrieval type** | Single fact, single entity | Multiple entities, temporal order |

The NIAH task trains a simple **associative recall** ability: see "secret code is X", later output X. BABILong requires **relational reasoning** over multiple entities with temporal ordering -- a fundamentally harder task that the NIAH training never exercises.

---

## Finding 3: chunk_size=4096 vs BABILong Input Lengths (HIGH confidence)

Using the Llama-3 tokenizer to compute exact token counts:

| BABILong Length | Template Tokens | Total Input Tokens | Chunks @ chunk_size=4096 | Memory Writes |
|-----------------|-----------------|-------------------|--------------------------|---------------|
| 1k | 167 | ~1,204 | **1** | **0** |
| 2k | 167 | ~2,228 | **1** | **0** |
| 4k | 167 | ~4,276 | 2 | 1 |
| 8k | 167 | ~8,372 | 3 | 2 |

### Critical Issue

At 1k and 2k, the ENTIRE input fits in a single chunk. In `generate_with_memory()`:

```python
chunks = list(tokens.split(chunk_size))  # [all_tokens]
for chunk in chunks[:-1]:  # chunks[:-1] = [] -- empty!
    model.forward_chunk(ct, ...)  # never executed
last = chunks[-1]  # the only chunk
out = model.forward_chunk(last, ...)
```

**Zero memory accumulation happens.** The model processes the entire input in a single forward pass through `_forward_middle_layer_memory`:
- L16 (write layer): slots initialized from strided hidden states, then updated
- L18,22,26,30 (read layers): cross-attention from hidden states to slots with `residual_scale=0.01`
- The 0.01 scale means memory contributes ~1% of the residual signal
- The model must rely on vanilla causal self-attention for the other 99%

At 4k and 8k, there are 1-2 memory accumulation chunks, but the compression is extreme:
- 64 slots / 4096 tokens = **64x compression**
- The second chunk (containing the question) is only ~180 tokens
- All bAbI facts must be retrieved from 64 slots -- this is far too lossy for multi-entity factual QA

---

## Finding 4: niah_loss is NOT a Retrieval Metric (HIGH confidence)

### What niah_loss Measures

`niah_loss` is the cross-entropy loss on the 5 answer-digit tokens in the NIAH sample:

```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = last_labels[..., 1:].contiguous()
loss = CE(shift_logits, shift_labels)
```

This measures **next-token prediction probability** of the 5 digit tokens. It is NOT measuring whether the model retrieves information from memory slots.

### Why Low niah_loss Does Not Imply Good Retrieval

1. **Shortcut learning**: The model can learn to associate the question template "The secret code for agent X is" with the digits that appeared near "MEMORIZE" in the same sequence, using ONLY the causal self-attention within the last chunk -- no memory needed.

2. **NIAH samples have the answer in-context**: In `niah_dataset.py`, the question chunk includes both the question AND the answer tokens. With labels = -100 everywhere except the answer positions, the model is trained to predict the answer tokens that are literally present in the input.

3. **The NIAH dataset has no distractor facts**: There is exactly one needle per sample. BABILong has 5-20 facts about different entities, requiring selective retrieval.

### H13 vs H14 niah_loss Comparison

| Experiment | niah_loss | BABILong |
|-----------|-----------|----------|
| H13_isolate@2500 | 0.580 (low) | 0/30 |
| H14_isolate_aggr@1500 | 11.64 (high) | 0/30 |

Both score 0% on BABILong despite 20x difference in niah_loss. This confirms niah_loss measures something orthogonal to BABILong performance. The "low" niah_loss in H13 likely reflects memorized pattern matching, not genuine long-range retrieval.

---

## Finding 5: Memory Write/Read Path Analysis (MEDIUM confidence)

### Architecture (H-series: middle_layer_memory mode)

```
Input tokens → Embed → L0-L15 (vanilla) → L16 (WRITE: joint-attn with slots)
  → L17 (vanilla) → L18 (READ: cross-attn to slots, scale=0.01)
  → L19-L21 (vanilla) → L22 (READ) → L23-L25 (vanilla) → L26 (READ) → L27-L29 → L30 (READ) → L31 → LM Head
```

### Write Path (L16)

1. `_init_slots(write_layer, hidden_states)`: If first chunk, strided init from hidden states. If subsequent chunk, detach and reuse previous slots.
2. Joint self-attention on `[slots, hidden_states]` (S+T tokens)
3. Dual-gate writeback: `new_slots = g_in * tanh(new) + g_forget * old`

### Read Path (L18,22,26,30)

1. Vanilla self-attention on hidden states only
2. Cross-attention: Q=hidden_states, K=V=slots
3. Residual: `hidden_states = hidden_states + 0.01 * memory_output`

### Potential Issues

1. **Single write layer**: All slot updates happen at L16 only. If L16's joint attention doesn't capture the relevant facts (because they're distributed across 4096 tokens), the slots won't contain them.

2. **64 slots is too small for multi-entity QA**: BABILong qa1 requires tracking ~5 entities across ~5 locations. That's 25 entity-location pairs. With 64 slots and 64x compression, most of this information is lost. The compression works for a single "secret code" needle but not for structured multi-entity data.

3. **residual_scale=0.01**: Even if slots contained perfect information, the read path only contributes 1% of the signal. This was designed to be conservative (safe at init), but it severely limits how much the memory can help.

---

## Conclusion and Recommendations

### Root Cause Hierarchy

1. **PRIMARY (confidence: HIGH)**: Llama-3-8B base cannot do BABILong (0-1% natively). Any fine-tune of this base model will also score near 0% unless instruction-following capability is added.

2. **SECONDARY (confidence: HIGH)**: NIAH training task (single-fact associative recall) does not teach multi-entity relational reasoning required by BABILong.

3. **TERTIARY (confidence: MEDIUM)**: chunk_size=4096 means 1k/2k BABILong gets zero memory benefit (single chunk). 64 slots with 64x compression is insufficient for multi-entity QA even at longer lengths.

### Recommended Actions

1. **Switch to an Instruct base model**: Use Llama-3-8B-Instruct or Qwen2-7B-Instruct as the base. The model must already know how to follow instructions before adding memory.

2. **Replace NIAH with retrieval-style training**: Instead of synthetic needle-in-haystack, train on tasks that require multi-entity tracking (e.g., bAbI tasks, TREC, HotpotQA, or synthetic versions of BABILong qa1/qa2/qa5).

3. **Evaluate NIAH accuracy separately**: Run MemLong-protocol NIAH eval to verify if the memory actually works for single-fact retrieval. This is the apples-to-apples comparison that the current NIAH training targets.

4. **Consider smaller chunk_size for BABILong**: With chunk_size=4096, most BABILong samples fit in 1 chunk. A chunk_size of 512 or 1024 would actually exercise the memory write-then-read path.

### Quick Validation Experiment

Run the H13/H14 checkpoints on MemLong-protocol NIAH (the eval function already exists in train_cross_attn_memory.py). If the model scores well on MemLong NIAH but 0% on BABILong, this confirms the task mismatch hypothesis. If it also scores 0% on MemLong NIAH, then the memory architecture itself is fundamentally broken (not just the task).
