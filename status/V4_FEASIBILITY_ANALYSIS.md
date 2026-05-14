# V4 Feasibility Analysis: Per-Layer Sparse Memory Bank

**Date**: 2026-05-01
**Reviewer**: Researcher Agent
**Design doc**: `versions/v4_chunk_last_hidden_memory.md`
**Status**: PROCEED WITH CAUTION -- Phase 1 (append-only) is viable; Phase 2 (top-k) has one blocking issue

---

## Executive Summary

V4 is the cleanest design so far in this project. By making the memory bank a detached runtime state (not a differentiable module), it sidesteps every gradient-path failure that plagued v0-v3 (routing degeneracy, gate saturation, VQ collapse, slot norm explosion). The per-layer bank architecture naturally avoids the OOD injection that killed RMT v3-v10.

**Overall confidence: 7/10 for Phase 1, 5/10 for full Phase 1+2.**

The single biggest risk is the attention mask compatibility with Flash Attention / SDPA in HuggingFace transformers 5.5.4. The existing codebase already has a working `_build_extended_attn_mask` that does exactly what v4 needs, so this is not a new problem.

---

## 1. Comparison with Existing Codebase

### What already exists and can be reused

| Component | Existing | V4 Needs | Reusable? |
|-----------|----------|----------|-----------|
| `MemoryBank` | `memory_bank.py`: nn.Module with slots [B,N,d], EMA write, lazy init, norm cap | Plain Python class with slots [B,N,d], append/get_all/top_k/update/reset | **Partially** -- The EMA write and scatter logic can be copied. But the new bank is NOT an nn.Module (no parameters, no init_from_hidden). Need a new class. |
| `TopKSelector` | `selector.py`: Q_sel/K_sel projections, learnable slot_keys, STE, load-balance loss | Cosine similarity top-k with NO learnable parameters | **No** -- V4 uses raw cosine sim, not learned projections. Simpler but different. |
| `_build_extended_attn_mask` | `layer.py`: builds [B,1,k+T,k+T] prefix causal mask with SWA support | `make_prefix_causal_mask`: same shape, same semantics (slot rows see everything, token rows see slots + causal among tokens) | **Yes** -- Functionally identical. The existing implementation already handles the v4 mask pattern exactly. |
| `_extend_position_embeddings` | `layer.py`: prepends k position-0 entries to (cos, sin) | Same: slot positions all get pos=0 so cos=1, sin=0 (no rotation) | **Yes** -- Direct reuse. |
| `patch.py` | Swaps `LlamaDecoderLayer` -> `MemorySpaceLayer` | Same pattern but for `ChunkMemoryLayer` | **Yes** -- Same monkey-patching approach. |
| Training loop | `train_mem_space_pg19.py`: chunk-based forward, _reset_banks, _detach_banks, DDP, PPL eval | Same infrastructure but banks are persistent across chunks within a document, reset between documents | **Mostly** -- Major change: banks persist across chunks (v0 resets every chunk). Need a "document boundary" signal. |

### What must be written from scratch

1. **`ChunkMemoryBank`** (~50 lines): Plain Python class (not nn.Module). `append`, `get_all`, `top_k`, `update_selected`, `reset`. No projections, no noise init.
2. **`ChunkMemoryLayer`** (~100 lines): Wrapper around LlamaDecoderLayer. Handles Phase 1 (append) and Phase 2 (top-k) forwarding. No selector, no gate_param, no slot_output_gate.
3. **`patch_v4`** (~30 lines): Simpler version of `patch.py`. Creates per-layer `ChunkMemoryBank` instances and wraps each decoder layer.
4. **Training loop modifications** (~50 lines): Document-boundary-aware bank management. Banks persist across chunks within a document but reset at boundaries.

**Total new code: ~230 lines** (within the ~200 line estimate from the design doc).

---

## 2. Design Decision Assessment

### Decision A: No gradient on bank (detached runtime state)

**Confidence: 9/10 -- This is the strongest design decision in v4.**

**Feasibility analysis:**

- **DDP compatibility**: Since `ChunkMemoryBank` is not an `nn.Module` and holds no `nn.Parameter`, DDP will not try to synchronize its state. The bank is purely per-GPU per-sample runtime state. This is identical to how a running mean/variance works in batch norm during inference -- no DDP issues.
- **LoRA training**: The design doc says "model trains via LoRA to learn how to use slots." This is compatible: LoRA adapters are applied to the frozen backbone's attention/MLP weights. The memory bank prefix is just additional input tokens. The LoRA weights will learn to attend to slot tokens. No conflict.
- **Why this works**: The key insight is that "learning to read memory" and "learning to write memory" are separate problems. V4 only needs the model to learn reading (via LoRA fine-tune of the attention weights). Writing is a fixed rule (last_hidden detach, append/EMA). This eliminates the entire gradient-through-routing failure mode.

**Potential issue**: Without gradient signal, the bank's content quality depends entirely on the heuristic (last hidden state). If last hidden is a poor chunk summary, the model cannot learn to produce better summaries. The design doc acknowledges this (Section 3.5) and defers attention pooling to a future version.

### Decision B: Per-layer bank

**Confidence: 8/10 -- Sound architectural choice.**

**Feasibility analysis:**

- **Implementation complexity**: Minimal. Each `ChunkMemoryLayer` holds its own `ChunkMemoryBank`. The bank dimension is `d_model` (hidden_size), which varies by model but is constant across layers for Llama.
- **Semantic alignment**: Each layer's bank stores hidden states from THAT layer. When layer k reads its bank, the slots are in the same representational space as the current hidden states. No projection needed. This is fundamentally better than RMT's approach of injecting layer-32 representations into layer-1.
- **Memory cost**: 32 layers x 64 slots x 4096 dim x 2 bytes (bf16) = 16 MB per sample. Negligible.

**Potential issue**: During training, the bank content for layer k depends on the forward pass through layers 0..k-1. If the model weights change (LoRA updates), the "old" slots from earlier chunks were produced by slightly different weights. This is a form of staleness. The design doc's EMA decay partially mitigates this -- newer chunks' representations dominate. For Phase 1 testing (few training steps), staleness is negligible.

### Decision C: Two-phase strategy

**Confidence for Phase 1 (append): 9/10. Confidence for Phase 2 (top-k): 6/10.**

**Phase 1 feasibility:**

Phase 1 is deterministic: slots are appended in order until the bank is full. No routing, no selection. The model sees [slot_0 | tokens], then [slot_0, slot_1 | tokens], etc. This is almost exactly TransformerXL's memory mechanism with a fixed cache size.

The critical test for Phase 1 is whether the model can learn to extract useful information from detached slot tokens via LoRA fine-tuning. If PPL <= vanilla * 1.05 at step 50 (the go/no-go criterion), this confirms the model can attend to slot prefixes without disruption.

**Phase 2 feasibility:**

Phase 2 introduces top-k selection. The cosine similarity computation is:
```python
scores = F.normalize(query, dim=-1) @ F.normalize(slots, dim=-1).T  # [B, N]
```

**Risk**: If all slots have very similar norms/directions (which happens when chunks come from the same document and share vocabulary), cosine similarity becomes near-uniform, and top-k is essentially random. The design doc's epsilon-greedy (5% random exploration) is a band-aid but not a fix.

However, this is mitigated by Decision A: since routing doesn't affect gradients, random routing just means "slightly less optimal slot selection" rather than "training collapse." The PPL impact should be small.

**Recommendation**: Implement Phase 1 first. Only add Phase 2 if Phase 1 passes the go/no-go criterion and PPL does NOT decrease with more chunks.

### Decision D: No RoPE on slots (position_ids = 0)

**Confidence: 7/10 -- Works but has a subtle interaction with BOS tokens.**

**Feasibility analysis:**

In HuggingFace transformers 5.5.4, `LlamaModel.forward` computes:
```python
position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
```
This produces `(cos, sin)` tensors of shape `[B, T, head_dim]`. The `_extend_position_embeddings` function in the existing codebase already prepends k copies of `cos[:, :1, :]` and `sin[:, :1, :]` -- which corresponds to position_id=0.

When `position_id=0`: `cos(0)=1, sin(0)=0`, so `q_rotated = q * 1 + q_perp * 0 = q` (identity rotation). This means slot Q/K vectors are NOT rotated by RoPE. Slot-token attention becomes pure semantic similarity (no positional bias).

**The BOS conflict**: In Llama-3, the BOS token occupies position 0 in real sequences. When we set slot position_ids to 0, slots share the same rotary embedding as the BOS token. During pretraining, the model learned that "position 0 = beginning of sequence." Slots at position 0 might inherit this "I am at the beginning" attention pattern.

**Mitigation options** (in order of preference):
1. **Use a large fixed position** (e.g., `position_id = max_position_embeddings - 1`). This gives a unique, consistent rotary value that doesn't conflict with any real token position. The cos/sin values will be some fixed rotation that the model has never seen in pretraining, but since we're LoRA-fine-tuning, it will adapt.
2. **Use the existing approach** (position_id=0). Since Phase 1 only tests 2-8 chunks, the BOS conflict is unlikely to cause measurable PPL degradation. Evaluate first, fix if needed.
3. **ALiBi-style** (no RoPE for slots). Requires modifying the SDPA call to use a custom attention mask with position biases. More complex but conceptually cleanest.

**Recommendation**: Start with option 2 (position_id=0, existing implementation) for Phase 1. If PPL regression is observed, switch to option 1.

**SDPA compatibility**: The existing `_build_extended_attn_mask` already produces a float additive mask `[B, 1, k+T, k+T]` with `0 = allowed, -inf = masked`. HuggingFace's SDPA path accepts this via the `attn_mask` parameter. Flash Attention 2 also accepts float masks. Both paths work with the v4 design. No blocking issue here.

### Decision E: Sparse update (EMA on selected slots, untouched on unselected)

**Confidence: 8/10 -- Correct and well-motivated.**

**Feasibility analysis:**

The EMA formula:
```python
updated = ema_decay * current + (1 - ema_decay) * new_hidden
```
where `ema_decay = 0.9` means each update blends 10% new content with 90% old content.

This is mathematically sound. After k updates, a slot contains:
```
slot = 0.9^k * original + (1 - 0.9^k) * weighted_average_of_updates
```

With `ema_decay=0.9`:
- After 1 update: 90% original + 10% new
- After 5 updates: 59% original + 41% blended new
- After 10 updates: 35% original + 65% blended new

A slot that is NEVER selected retains 100% of its original content indefinitely. This is the information retention mechanism described in Section 3.1.

**Implementation correctness**: The `scatter` operation (already implemented in the existing `MemoryBank.write`) correctly writes to selected positions only. The `gather` + EMA + `scatter` pattern is standard and well-tested in the existing codebase.

**One concern**: All k selected slots are updated with the SAME `new_hidden` (the last token of the current chunk). This means if k=8 slots are selected, all 8 get an EMA update with identical content. After several chunks, the selected slots will converge toward the same value. This reduces slot diversity.

**Mitigation**: Use per-slot updates from different hidden positions (e.g., strided sampling from the chunk). But this is an optimization, not a blocker. Phase 1 doesn't use top-k at all.

---

## 3. Implementation Challenges

### Challenge 1: Patching LlamaDecoderLayer.forward

**Severity: LOW -- Already solved in the existing codebase.**

The existing `MemorySpaceLayer` wraps `LlamaDecoderLayer` and calls it twice per forward (bypass + extended). V4 needs only ONE call (extended) or one call (bypass, when bank is empty). This is simpler.

The wrapper intercepts the forward at the decoder-layer level:
```python
# In LlamaModel.forward:
for decoder_layer in self.layers:
    layer_outputs = decoder_layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        ...
        position_embeddings=position_embeddings,
    )
```

V4's `ChunkMemoryLayer` replaces the decoder layer in `self.layers` (same as `patch.py` does). When the bank has slots, it:
1. Concatenates slots to hidden_states: `[B, n_slots+T, d]`
2. Extends position_embeddings: prepend k copies of pos-0 (cos, sin)
3. Builds the prefix causal mask
4. Calls `wrapped_layer(extended_input, extended_pos_emb, extended_mask)`
5. Slices output: `out[:, n_slots:, :]`

This works because `LlamaDecoderLayer` processes whatever input it receives. It does not validate sequence length or position_ids against any internal state.

### Challenge 2: DDP with per-sample state across chunks

**Severity: MEDIUM -- Requires careful training loop design.**

The bank is per-sample state that persists across chunks within a document. In DDP:

1. **Bank is not an nn.Module**: DDP ignores it. No gradient synchronization needed (bank is detached).
2. **Bank consistency across GPUs**: Each GPU processes different samples (DistributedSampler). Each GPU maintains its own bank for its own samples. No cross-GPU bank synchronization needed.
3. **Document boundaries**: Banks must reset when a new document starts. The training loop must detect document boundaries and call `bank.reset()`.

**The key constraint**: The same sample must be processed on the same GPU across all its chunks. Standard `DistributedSampler` with `shuffle=False` ensures this if the dataset is organized as contiguous chunks from the same document.

**Current training loop issue**: `train_mem_space_pg19.py` treats each chunk as an independent sample. It calls `_reset_banks(model)` or `_detach_banks(model)` per step. For v4, we need:
- Group chunks by document
- Stream all chunks of a document sequentially on the same GPU
- Reset banks only at document boundaries

This is a significant change to the data loading logic but not to the model architecture. The NIAH streaming code in `train_mem_space_pg19.py` (lines 862-894) already implements this pattern (stream chunks with `torch.no_grad()`, then gradient on the last chunk). V4 would use the same pattern for ALL chunks, not just NIAH.

### Challenge 3: Attention mask and Flash Attention / SDPA compatibility

**Severity: LOW -- Already solved.**

The existing `_build_extended_attn_mask` in `layer.py` produces exactly the mask v4 needs:
```python
# Slot rows: see everything (all-zeros)
# Token rows: see all slots + causal among tokens
mask[:k, :] = 0          # slots see everything
mask[k:, :k] = 0         # tokens see all slots
mask[k:, k:] = causal    # tokens see tokens causally
```

The existing code uses `torch.finfo(dtype).min` (not `-inf`) as the mask value, which is compatible with both SDPA and Flash Attention 2 backends.

For the v4 "slot rows see everything" pattern: the design doc says `slot -> token: invisible (slot does not attend to future tokens)`. This is DIFFERENT from the existing code where slot rows see everything. The v4 mask should be:
```python
# Slot rows: see other slots (left portion = 0), but NOT tokens (right portion = -inf)
mask[:n_slots, :n_slots] = 0           # slots see slots
mask[:n_slots, n_slots:] = -inf        # slots do NOT see tokens
# Token rows: see all slots + causal among tokens
mask[n_slots:, :n_slots] = 0           # tokens see all slots
mask[n_slots:, n_slots:] = causal      # tokens see tokens causally
```

Wait -- the design doc's mask (Section 2.3) shows exactly this pattern. But there is a subtlety: if slot queries cannot see token keys, then the slot positions in the output will not benefit from attending to the current chunk. The slot outputs (`O_mem`) will be pure slot-to-slot attention, which is not useful.

Actually, re-reading the design doc more carefully: the slot outputs are DISCARDED in v4. Only the token outputs (`out[:, n_slots:, :]`) are used. The bank update uses `out[:, -1, :]` from the TOKEN portion, not the slot portion. So it doesn't matter what the slot queries attend to -- their outputs are ignored.

However, this means the "double forward" optimization from the existing code (bypass + extended) is unnecessary for v4. We can do a SINGLE forward with the extended sequence and just take the token portion of the output.

### Challenge 4: The "same new_hidden for all k slots" problem

**Severity: LOW for Phase 1 (no top-k), MEDIUM for Phase 2.**

In Phase 2, when k=8 slots are selected and updated with the same `last_hidden`, they will converge. After 20 chunks with ema_decay=0.9, selected slots contain:
```
slot_i ≈ 0.9^20 * original + 0.88 * blended_last_hiddens
```
If the same 8 slots are always selected, they become near-identical. This defeats the purpose of having 64 diverse slots.

**Mitigation for Phase 2 implementation**:
1. Use the last-k token hidden states (not just the last token) for the k selected slots
2. Add epsilon-greedy exploration (already in the design doc)
3. Use mean-pooled query instead of last-token query for top-k selection

---

## 4. Concrete Step 1 Implementation Plan

### Files to create/modify

| File | Action | Lines | Description |
|------|--------|-------|-------------|
| `src/memory/mem_space/chunk_memory.py` | **CREATE** | ~80 | `ChunkMemoryBank` class + `make_prefix_causal_mask` |
| `src/memory/mem_space/chunk_layer.py` | **CREATE** | ~100 | `ChunkMemoryLayer` wrapper |
| `src/memory/mem_space/patch_v4.py` | **CREATE** | ~40 | `apply_chunk_memory_to_model` function |
| `scripts/train_v4_chunk_memory.py` | **CREATE** | ~200 | Training script with document-boundary-aware bank management |
| `src/memory/mem_space/__init__.py` | **MODIFY** | +3 | Export new classes |

### Classes/functions to implement

**`ChunkMemoryBank`** (~50 lines):
```python
class ChunkMemoryBank:
    """Per-layer, per-sample memory bank. NOT an nn.Module."""
    def __init__(self, num_slots, d_model, ema_decay=0.9):
        self.num_slots = num_slots
        self.d_model = d_model
        self.ema_decay = ema_decay
        self.slots = None        # [B, N, d], lazy init
        self.num_filled = 0

    def append(self, hidden):     # hidden: [B, d]
    def get_all(self):            # -> [B, n_filled, d]
    def top_k(self, query, k):   # query: [B, d] -> (slots [B,k,d], idx [B,k])
    def update_selected(self, idx, new_hidden):  # idx: [B,k], new_hidden: [B,d]
    def reset(self):
```

**`ChunkMemoryLayer`** (~80 lines):
```python
class ChunkMemoryLayer(nn.Module):
    """Wraps LlamaDecoderLayer with chunk memory bank prefix."""
    def __init__(self, base_layer, num_slots=64, k=8):
        self.base_layer = base_layer
        self.num_slots = num_slots
        self.k = k

    def forward(self, hidden_states, attention_mask, position_ids,
                position_embeddings, **kwargs):
        # Phase 1 (bank empty): normal forward, append last_hidden to bank
        # Phase 1 (bank not full): prepend all slots, forward, append
        # Phase 2 (bank full): top-k select, prepend, forward, EMA update
```

**`make_prefix_causal_mask`** (~20 lines): Copied from `layer.py:_build_extended_attn_mask` with simplified semantics (no SWA).

**`apply_chunk_memory_to_model`** (~30 lines): Same pattern as `patch.py:apply_mem_space_to_model` but creates `ChunkMemoryBank` per layer and `ChunkMemoryLayer` wrappers.

### Training script structure

```python
# scripts/train_v4_chunk_memory.py
# Key differences from train_mem_space_pg19.py:
# 1. Data organized as documents (multiple chunks per document)
# 2. Banks persist across chunks within a document
# 3. Only LoRA params are trainable (or no training at all for Phase 0)
# 4. No aux losses (no selector, no load balance, no key repulsion)
```

### Test plan

**Phase 0 (cold start verification, 50 steps)**:
- Config: num_slots=64, k=8, chunk_size=4096, 2 chunks per sample
- No training (forward-only, LoRA not applied)
- Expectation: PPL <= vanilla * 1.05
- Why this should pass: Slots are detached hidden states. The model sees them as additional prefix tokens. With no gradient flow to the bank, the model's existing weights process slots as "extra context." If the attention mask is correct, slots should not disrupt the model's predictions.

**Phase 1 (LoRA fine-tune, 500 steps)**:
- Config: Same + LoRA rank=16, lr=1e-4
- Data: pg19 documents, 4-8 chunks each
- Expectation: PPL at chunk_4 < PPL at chunk_1 (model learns to use memory)
- Go/No-go: chunk_4 PPL < chunk_1 PPL

### Go/No-Go criteria

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Phase 0: No PPL regression | PPL <= vanilla * 1.05 | Eval 50 chunks with empty+1-slot banks |
| Phase 1: Memory helps | chunk_4 PPL < chunk_1 PPL | Per-chunk PPL tracking |
| Phase 1: No NaN | 0 non-finite losses | Standard NaN check |
| Phase 2: Top-k selection | slot_usage > 1 unique slot per step | Diagnostic log |

---

## 5. Risk Assessment

### High-probability risks

1. **Bank persistence across chunks causes PPL regression** (Probability: 20%)
   - Cause: Stale slot content from earlier chunks confuses the model
   - Detection: PPL at chunk_2 > PPL at chunk_1 (memory hurts rather than helps)
   - Mitigation: Start with append-only (Phase 1), measure per-chunk PPL curve

2. **Position_id=0 interacts badly with pretrained attention patterns** (Probability: 15%)
   - Cause: Pretrained model expects position 0 to be BOS, slots at position 0 confuse the attention
   - Detection: Phase 0 PPL > vanilla * 1.05
   - Mitigation: Switch to position_id = max_pos - 1

### Low-probability risks

3. **DDP hangs due to bank state inconsistency** (Probability: 5%)
   - Cause: Bank is not an nn.Module, so DDP doesn't know about it. If bank somehow affects gradient computation (it shouldn't since everything is detached), DDP may deadlock.
   - Detection: Training hangs at backward()
   - Mitigation: Verify all bank tensors are `.detach()` in the forward path

4. **SDPA numerical differences between extended and vanilla sequences** (Probability: 10%)
   - Cause: Different sequence lengths trigger different SDPA kernels
   - Detection: Phase 0 PPL != vanilla exactly
   - Mitigation: Already handled by the existing `_build_causal_attn_mask` / `_build_extended_attn_mask` approach

---

## 6. Comparison with Failed Approaches

| Failure Mode | Why v0-v3 Failed | Why V4 Avoids It |
|---|---|---|
| Routing degeneracy (top1_sim -> 1/N) | Selector gradient collapsed because STE had zero gradient at uniform fixed point | No selector. Top-k is cosine sim on detached slots, no gradient, no degeneracy |
| VQ codebook collapse | Learned slot_keys converged to same direction | No learnable keys. Slots are raw hidden states |
| Gate saturation (output_gate=0) | tanh(0)=0, zeroing all slot contribution. Gradient to gate was also 0 | No output gate. Slots are prepended directly. Model's native attention weights decide how much to attend |
| Frozen o_proj bottleneck (Infini v3-v5) | Linear attention retrieval signal suppressed to 1.5% of hidden norm | No separate retrieval path. Slots go through the standard softmax attention |
| OOD injection (RMT v3-v10) | Layer-32 hidden injected into layer-1, causing repetitive generation | Per-layer bank. Layer k's slots are in layer k's semantic space |
| Unbounded M accumulation (Infini) | Associative matrix grows without bound, PPL degrades over long sequences | Fixed N slots. No accumulation, just EMA update of existing slots |

---

## 7. Recommendation

**PROCEED with Step 1 implementation.**

The v4 design is the first approach in this project that:
1. Has no learnable routing parameters (eliminates routing degeneracy)
2. Uses no output gate (eliminates gate saturation)
3. Has no cross-layer state injection (eliminates OOD)
4. Has bounded memory (fixed N slots, eliminates unbounded accumulation)

The implementation is simpler than the existing `MemorySpaceLayer` (no selector, no gate, no STE, no aux losses). The existing codebase provides working implementations for the two hardest parts (extended attention mask and position embedding extension).

**Concrete next steps:**
1. Create `chunk_memory.py` (ChunkMemoryBank) -- 50 lines
2. Create `chunk_layer.py` (ChunkMemoryLayer) -- 100 lines
3. Create `patch_v4.py` -- 30 lines
4. Create `train_v4_chunk_memory.py` -- 200 lines
5. Run Phase 0 test: 2-chunk forward-only on pg19, verify PPL <= vanilla * 1.05
6. If Phase 0 passes, run Phase 1 with LoRA fine-tuning
7. If Phase 1 passes, add Phase 2 (top-k selection)

**Time estimate**: 1-2 hours for implementation, 30 minutes for Phase 0 test.

**Critical path**: If Phase 0 fails (PPL > vanilla * 1.05), the issue is almost certainly in the attention mask or RoPE handling, not in the memory bank logic. Debug by comparing the extended forward output against the vanilla forward output token-by-token for a single sequence.
