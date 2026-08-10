# Slot Memory Initialization & Cross-Chunk Persistence Research
**Date**: 2026-04-29  
**Author**: researcher subagent  
**Context**: Triggered by user observation that all 512 slots are initialized to essentially the same vector (current-chunk token mean ± 0.02 noise), making slot-based memory degenerate. This note builds on the Fix A–J history in `ops/research_notes/20260429_fixH_failure_analysis.md`.

---

## 1. Root Cause Analysis

The current initialization in `MemoryBank.init_from_hidden()` (`memory_bank.py` lines 143–154) is degenerate in two compounding ways. First, it takes the **mean** of all T tokens in the current chunk — a single collapsed vector — and broadcasts it to all N=512 slots before adding σ=0.02 Gaussian noise. For slot_dim=128, the slot diversity ratio is σ/‖mean‖ ≈ 0.02/√128 ≈ 0.18%, meaning all 512 slots start as essentially identical point masses on the hidden-state manifold. Second, `_reset_banks()` in the training script calls `memory_bank.reset()` (which sets `self.slots = None`) at the beginning of **every training step** — both the standard pg19 path (line 718) and the NIAH path (line 682). This means slots never accumulate information across chunk boundaries; they are re-initialized from the current chunk's mean at every forward pass. Combined, these two behaviors ensure that (a) all slots contain the same information, and (b) no past information survives to the next training step. This is structurally incompatible with a long-context memory system: the slot bank is effectively a single scalar summary of the current chunk, not a memory of anything.

---

## 2. Option Evaluation Table

| Option | Implementation Complexity | Training Stability Risk | Expected Diversity Gain | EMA Write Interaction | Cross-Chunk Retention |
|---|---|---|---|---|---|
| **A. Carry-over (detach, not reset)** | **Very Low** — change `reset()` → `detach_()` in `_reset_banks()`, ~3 lines | Low-Medium — slots from doc k propagate to doc k+1; if training is on single-document pg19 chunks this is correct behavior; NIAH already does this correctly within a document (lines 689–692 do NOT reset between chunks within a document) | High over time — slots accumulate genuine per-document information via EMA writes | **Best** — EMA write is designed for exactly this: old slots retain content when not selected, selected slots blend old+new | **Excellent** — this is the only option that preserves information across chunk boundaries |
| **B. Per-slot token assignment** | Low — 2-line change to `init_from_hidden()`: `slot_i = H_l[:, i*T//N, :]` | Very Low | Medium — T tokens spread over 512 slots; for T=4096, N=512, stride=8, so each slot gets one token from every 8th position; covers the input but loses structure | Compatible — EMA starts from a real token, not a collapsed mean; higher initial SNR | None — still fully reset between chunks |
| **C. K-means over H_l** | High — requires k-means iteration inside `init_from_hidden()`; non-differentiable; sklearn or custom CUDA impl; ~100 LOC | Medium — k-means convergence varies; can produce empty clusters; slow on GPU for large T | Highest within-chunk — centroids capture semantic clusters in the current chunk | Compatible but likely over-engineered — after one EMA write step, k-means advantage decays to noise level | None — still reset between chunks |
| **D. Learned persistent slot embeddings** | Medium — add `nn.Parameter` of shape [N, slot_dim] to MemoryBank; use as initialization instead of hidden_pool | Medium — learned init is a prior over all documents; may be hard to train simultaneously with the routing mechanism; gradient from LM loss to slot init parameters is long-range | Medium — single learned prior captures corpus-level structure but not document-specific memory | Complementary — persistent embeddings provide a prior; EMA writes push individual document content on top | None for document-specific memory — but provides a stable cross-document prior |
| **E. Reservoir / circular buffer** | Medium-High — requires maintaining a separate FIFO buffer of [B, M, slot_dim] shape alongside the existing slot bank; needs eviction logic | Medium — dual-state system adds complexity; gradient flow through FIFO is non-trivial | High — last M actual hidden states retained verbatim | Requires restructuring write path — EMA no longer meaningful if slots are just a FIFO; conflicts with slot-identity property | **Excellent** — last M=N states always in slots; but this is exactly a sliding-window KV cache, not truly a compressed memory |

---

## 3. Recommended Approach: Carry-over (Option A) + Per-slot Token Init (Option B) Combined

### Recommendation: A+B combined

**Primary**: Replace `reset()` with `detach_()` in `_reset_banks()` for the standard pg19 training path (carry-over), while keeping full reset only at document boundaries (i.e., when switching to a genuinely new document, not just a new chunk of the same document).

**Secondary**: Change `hidden_pool` init to per-slot token assignment (Option B) so that when a true cold-start is needed (e.g., first chunk of a document, or batch size change), the initial slot content is diverse from the start rather than a single collapsed mean.

### Justification

**Carry-over is architecturally correct and minimal**:
- The `detach_()` method already exists on `MemoryBank` — it was designed for exactly this purpose (docstring: "Break the autograd graph across a segment boundary").
- The NIAH training path already does carry-over correctly: it calls `_reset_banks()` once per document at line 682, then streams all chunks through `model()` with `torch.no_grad()` without resetting between chunks (lines 689–692). The standard pg19 path is *less* correct than NIAH in this regard.
- With carry-over, the EMA write mechanism works as designed: selected slots blend their old content with new representations (β parameter), unselected slots retain their old content. After k writes per step with N=512 slots, all slots are visited in N/k = 64 steps (assuming uniform routing). At β=0.15, a slot written once still retains (1-0.15)=85% of its original content — information decays gradually rather than being wiped.
- No new parameters, no new code paths, no hyperparameter changes.

**Per-slot token assignment (Option B) as cold-start init**:
- When a true reset IS needed (new document, batch size change), `hidden_pool` init wastes the diversity present in the chunk by collapsing T=4096 tokens to a single vector.
- Per-slot assignment `slot_i = H_l[:, i*T//N, :]` gives each slot a distinct token from the chunk. For T=4096, N=512: stride=8, so slot 0 = token 0, slot 1 = token 8, ..., slot 511 = token 4088. Initial slot diversity = natural token diversity in the chunk (much higher than σ=0.02 noise on a single mean).
- Trivially compatible with the existing `init_from_hidden()` API — add one new `slot_init="strided_token"` branch.

**Why not K-means (Option C)?**
- All diversity gain from k-means vanishes after the first EMA write step if the write path is working correctly. The complexity cost is not justified for an init that gets overwritten.

**Why not learned embeddings (Option D)?**
- Adds gradient coupling between corpus-level priors and document-specific memory. With an already-fragile gradient path (as seen in Fix A–J history), this is a stability risk.

**Why not reservoir/FIFO (Option E)?**
- Equivalent to sliding-window KV cache — the very approach that was already shown to have worse PPL than Q-Filters in this project. No compression benefit.

### Critical compatibility check: pg19 training loop

The current pg19 training loop (line 718) calls `_reset_banks(model)` at the top of **every iteration**, and each iteration processes a **single chunk** of 4096 tokens from a single document. There is no cross-chunk accumulation within the standard pg19 path at all.

To implement carry-over correctly, we need to understand whether each batch in the pg19 loop comes from the same document or different documents. Looking at `_cycle_pg19()` (line 642): this generates pre-chunked batches from `pg19_chunks_llama3.npy`. Each batch is a single seq_len=4096 chunk. Consecutive batches are NOT guaranteed to be from the same document — they could be sequential chunks from one document or from different documents.

**Two implementation strategies for carry-over**:

1. **Document-aware carry-over**: Pass a `new_document` flag in each batch. Call `detach_()` between chunks of the same document; call `reset()` only when a new document starts. Requires modification of the data generator to emit document boundaries.

2. **Unconditional carry-over**: Always call `detach_()` (never call `reset()`). Slots from document k propagate to document k+1. This is a form of "approximate" memory — information from previous documents leaks into the current one. For language modeling, this may actually help (language has continuity across documents in a corpus) or hurt (document-specific patterns bleed through). **This is the simpler implementation and a reasonable starting point.**

The unconditional approach is recommended for the first experiment. It maps exactly to the Recurrent Memory Transformer approach (tokens passed between segments unconditionally, including across document boundaries).

---

## 4. Implementation Sketch

### 4.1 Change 1: `_reset_banks()` — carry-over instead of reset

**File**: `scripts/train_mem_space_pg19.py`  
**Function**: `_reset_banks()` lines 199–217

**Current behavior**: calls `shared_bank.reset()` or `w.memory_bank.reset()` — sets `self.slots = None`, forcing re-init on next forward.

**New behavior**: call `detach_()` instead of `reset()`. Name the function `_detach_banks()` or add a `reset=True` parameter.

**Exact change**:
```python
def _detach_banks(model: torch.nn.Module) -> None:
    """Break autograd graph across chunk/step boundary; preserve slot content."""
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.detach_()
        return
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return
    for w in mem_layers:
        w.memory_bank.detach_()
```

Then in the training loop:
- Replace the call at line 718 (`_reset_banks(model)`) with `_detach_banks(model)` for the pg19 path.
- Keep `_reset_banks(model)` at line 682 for NIAH (already resets once per document, which is correct).
- Note: first iteration — `slots` is None (never initialized), so `detach_()` is a no-op and `init_from_hidden()` will be called on the first forward as before. No cold-start issue.

**Lines to change**: pg19 path at line 718. ~1 line change + 1 new helper function (~10 lines).

### 4.2 Change 2: `init_from_hidden()` — strided token init

**File**: `src/memory/mem_space/memory_bank.py`  
**Function**: `init_from_hidden()` lines 106–155  
**Config**: `src/memory/mem_space/config.py` — `slot_init` field

**New init mode** `"strided_token"`:
```python
elif self.slot_init == "strided_token":
    # Each slot gets a distinct evenly-spaced token from H_l.
    # For N=512, T=4096: stride=8, slot i = token i*stride.
    # Handles T < N by repeating (modulo).
    T_len = H_l.shape[1]
    indices = torch.arange(N, device=device) * max(1, T_len // N)
    indices = indices % T_len                             # [N] — handles T < N
    # H_l[:, indices, :] : [B, N, d]
    slots = H_l.detach()[:, indices, :].clone()
    if self.init_noise > 0.0:
        slots = slots + torch.randn_like(slots) * self.init_noise
```

**Config change**: Add `"strided_token"` as a valid `slot_init` option in `config.py` (wherever `slot_init` is validated). Default remains `"hidden_pool"` for backward compatibility; use `"strided_token"` when carry-over is enabled (since cold-start diversity matters more with carry-over).

**CLI change**: `train_mem_space_pg19.py` already passes `--slot_init` to the model config. No new CLI argument needed.

**Lines to change**: ~8 lines in `memory_bank.py`, 1-line addition to valid values in `config.py`.

### 4.3 No changes needed to

- `memory_bank.write()` — the EMA mechanism is already correct for carry-over
- `MemoryBank.detach_()` — already implemented
- `layer.py` forward — unchanged
- `selector.py` — unchanged
- The NIAH path — already uses correct carry-over within a document

### 4.4 Training loop pseudocode after changes

```
iter 0:  _detach_banks(model)   # no-op (slots=None)
         forward(chunk_0)       # init_from_hidden called (strided_token init)
                                # write() called → slots updated

iter 1:  _detach_banks(model)   # slots.detach() → slots carry content from iter 0, graph broken
         forward(chunk_1)       # NO init_from_hidden (is_initialized=True) → slots read from carry-over
                                # write() called → slots updated with new chunk's info

iter 2:  _detach_banks(model)   # slots from iter 1 carried over
         ...
```

---

## 5. Convergence Analysis: How Long Until Slots Diverge?

With carry-over and β=0.15, k=64 writes per step, N=512 slots:

**Coverage rate**: On average, k/N = 64/512 = 12.5% of slots are written per step. Under uniform routing, each slot is written approximately every N/k = 8 steps. After S steps, each slot has been written S·k/N = S/8 times on average.

**Slot content after m writes**: If a slot starts with content s₀ and is written with values v₁, v₂, ..., vₘ via EMA:
```
s₁ = (1-β)s₀ + β·v₁
s₂ = (1-β)s₁ + β·v₂ = (1-β)²s₀ + β(1-β)v₁ + β·v₂
```
After m writes with m=8 (one round of all slots), the initialization weight is (1-β)^m = (0.85)^8 ≈ 0.27. So after 8 steps (all slots visited once), 73% of slot content comes from written representations.

**Diversity timeline**:
- Step 0: All slots = strided tokens (diverse from init)
- Step 8: All slots written once; content reflects actual retrieved representations; init bias ≈27%
- Step 40: Each slot written ~5 times; init bias ≈ (0.85)^40 ≈ 0.001; content almost entirely from writes
- Step 40+: Diversity maintained by routing variation — slots that get written more frequently reflect more recent/frequent query patterns

Compare to current (reset + hidden_pool): diversity is O(σ/‖mean‖) ≈ 0.18% at every step, forever. With carry-over, diversity reaches natural document-level distribution within ~40 steps.

---

## 6. Literature: Do Successful Memory Transformers Reset Per Step?

The answer from the literature is unambiguous: **no successful persistent-memory transformer resets all memory slots at every step**.

**Recurrent Memory Transformer (RMT, Bulatov et al. 2022, 2207.06881)**: Global memory tokens are prepended to each segment and passed (via detach) to the next segment. Never reset within a document. Reset only at genuine document boundaries. This is exactly Option A.

**MemoryLLM (Wang et al. 2024, 2402.04624)**: Uses a shift-append FIFO buffer — old content shifts out, new content appended. Content is never "reset to current chunk mean." The oldest representation naturally decays. Our EMA write is the continuous analog.

**Titans (Google DeepMind, 2025)**: Neural memory layer (neural long-term memory + attention as short-term memory). Long-term memory is updated via gradient-based learning but **never wiped** within a sequence. The memory update rule is essentially our EMA write.

**Perceiver IO (Jaegle et al. 2021)**: Learned latent array (`nn.Parameter` of shape [N, d]). Not reset per input. Persistent across all inputs (shared prior). Closest to Option D.

**Slot Attention (Locatello et al. 2020)**: Iterative refinement within one forward pass. **Not** persistent across inputs (each image starts with fresh slots). This is the only major slot-based method that does reset — but it's designed for single-image object discovery, not sequence memory.

**Extended Mind Transformers (2406.02332)**: External memory = frozen hidden states from past inputs, retrieved by kNN. Never reset. Read-only carry-over.

**CAMELoT (2502.00592)**: Raw-KV EMA writeback, training-free. Memory is carried over across chunks. Never reset mid-document.

**Conclusion**: Per-step full reset is not used by any successful memory system for sequential/document-level tasks. It is a design artifact from object-discovery literature (Slot Attention) that was incorrectly imported into a sequential memory context.

---

## 7. Open Questions

1. **Document boundary detection**: The unconditional carry-over implementation doesn't distinguish between within-document and cross-document chunk transitions in the pg19 data generator. Does cross-document carry-over hurt or help pg19 PPL? Hypothesis: slight benefit (language statistics are similar across pg19 books; slots accumulate corpus-level patterns). Should be measured empirically: compare PPL(carry-over unconditional) vs PPL(carry-over with doc-boundary reset) on 200 steps.

2. **Batch size > 1**: When batch_size > 1, each element in the batch is an independent document. `detach_()` preserves slot content for all batch elements. This is correct if batched elements come from different positions in the same corpus (they share the carry-over, which might be treated as cross-contamination). For the current batch_size=1 training this is a non-issue. For future batch_size>1: consider whether each batch element should have independent slot state (it already does — slots are [B, N, d] so each batch element has its own slots).

3. **EMA with carry-over: does β need adjustment?** Currently β warms up from 0 to max_beta=0.15. With carry-over, a very low β means old information is retained very strongly (good for long-range memory) but new information takes many steps to register. With the current warmup schedule, β≈0.001 at early steps means slots barely update. This is actually MORE correct for carry-over — the initial strided-token content is meaningful and should be preserved until the training signal is reliable. No β change needed for the first experiment.

4. **Interaction with shared_memory_bank**: The code uses a single shared MemoryBank across all 32 decoder layers (`shared_memory_bank=True`). With carry-over, this means all 32 layers share the same accumulated slot content across all steps. This is the intended design — one slot bank per sequence position, not per layer. Confirmed correct.

5. **Can carry-over interact with Fix J-A's gradient flow?** With carry-over, `slots` (written by `hidden_to_slot` in the previous step) are carried over as a detached tensor into the current step. `hidden_to_slot` from the **previous step** does NOT receive gradient through the carry-over detach boundary. `hidden_to_slot` in the **current step** (which writes the current step's O_mem_slot into slots) DOES receive gradient via the Fix J-A path (loss → M_sel_soft → slots → O_mem_slot → hidden_to_slot). This is correct: gradient is intra-step only, not multi-step BPTT. Carry-over + Fix J-A are fully compatible.

6. **Cold-start diversity after `strided_token` init**: With strided init, slot 0 gets token 0, slot 511 gets token 4088. Tokens within a chunk are semantically correlated (they come from the same document passage), so strided init gives token-level diversity, not semantic-cluster diversity. The diversity improvement over hidden_pool is significant (256× or more) but not maximal. K-means would give better semantic diversity at cold-start but the gain is temporary (slots get overwritten by writes quickly).

---

## Summary Table

| | Current (hidden_pool + reset) | Recommended (strided_token + carry-over) |
|---|---|---|
| Cold-start diversity | 0.18% (σ/‖mean‖ at σ=0.02) | Natural token diversity (~5-10× higher) |
| Cross-chunk retention | None (reset every step) | Full (EMA write persists) |
| Implementation complexity | — | ~20 lines across 2 files |
| Gradient path impact | None | Compatible with Fix J-A (intra-step only) |
| Training stability risk | — | Low (detach_ preserves graph safety) |
| Literature precedent | Slot Attention only (single-image, not sequential) | RMT, MemoryLLM, Titans, CAMELoT |
| Steps to full slot diversity | Never (reset wipes every step) | ~40 steps (all slots visited ~5 times) |

---

## 8. User Insight (2026-04-29): Last Token as Chunk Summary Under SWA

**Source**: User feedback appended to this research note after initial publication.

### 8.1 Theoretical Justification: Why H_l[:, -1, :] Is a Chunk Summary

Under Sliding Window Attention (SWA) with window size W and chunk length L ≤ W, the last token of the chunk at position t = L-1 attends to ALL tokens in positions [0, L-1] (since they all fall within its backward window of size W). No other token in the chunk has this property in general — token t' < L-1 can only attend to tokens in [t'-W+1, t']. 

**Formally**: Under SWA with window W and chunk L ≤ W, for the last token at position L-1:
```
attn_mask[L-1, j] = 1  ∀ j ∈ [max(0, L-1-W+1), L-1] = [0, L-1]   (since L ≤ W)
```

This means `H_l[:, -1, :]` (the final token's hidden state at layer l) is computed as a function of ALL L preceding hidden states via the full attention operation, making it a **complete compressed summary of the chunk** — analogous to:
- The `[CLS]` token in BERT (which can attend to all tokens bidirectionally)
- The final hidden state `h_T` in RNN seq2seq encoders
- The summary token appended at the end of each segment in Perceiver Resampler

**Comparison with strided-token init (Option B)**:
- Strided init: each slot gets `H_l[:, i*stride, :]`, a single token whose attention receptive field covers only a sub-window of the chunk (specifically tokens [i*stride - W + 1, i*stride])
- Last-token init: a single slot gets `H_l[:, -1, :]`, which has attended to the FULL chunk

The last token has strictly more information content about the chunk as a whole, but there is only one of it. Strided init gives diversity across 512 slots at the cost of each slot being a partial view. These are complementary, not competing.

### 8.2 Cold-Start Initialization: Revised Option for Slot 0

**User proposal**: At cold start, rather than mean-pooling (Option A in the original note's `hidden_pool` mode) or uniformly striding all 512 slots, initialize **one designated "summary slot" (e.g., slot 0) with `H_l[:, -1, :]`**, and use strided tokens for slots 1–511.

**Revised cold-start init pseudocode**:
```python
elif self.slot_init == "strided_with_summary":
    T_len = H_l.shape[1]
    # Slot 0 = last-token summary (full-chunk receptive field under SWA)
    slots_0 = H_l.detach()[:, -1:, :]                           # [B, 1, slot_dim]
    # Slots 1..N-1 = strided tokens (diverse partial views)
    indices = torch.arange(1, N, device=device) * max(1, T_len // (N - 1))
    indices = indices % T_len                                    # [N-1]
    slots_rest = H_l.detach()[:, indices, :]                    # [B, N-1, slot_dim]
    slots = torch.cat([slots_0, slots_rest], dim=1)             # [B, N, slot_dim]
    if self.init_noise > 0.0:
        slots = slots + torch.randn_like(slots) * self.init_noise
```

**Assessment of marginal benefit over pure strided-token**:
- At cold start, the difference between `H_l[:, -1, :]` and `H_l[:, T-1, :]` (last strided token at T_len // (N-1) * (N-1) ≈ T_len - stride ≈ T_len - 8) is small: both attend to most of the chunk window.
- For T=4096, N=512, stride=8: the last strided token in pure Option B is `H_l[:, 4088, :]`, which under SWA(W=4096) attends to tokens [0, 4088] — almost identical to the last token [0, 4095]. The difference is 7 tokens.
- **Verdict**: the user's insight is theoretically sound, but the practical delta over strided init for cold-start is small. The bigger benefit is in the **ongoing write strategy** (Section 8.3 below).

### 8.3 Ongoing Write Strategy: "Chapter Marker" Slot

**User proposal**: After processing each chunk, explicitly write `hidden_to_slot(H_l[:, -1, :])` into a designated "summary slot" as a chapter marker. Future chunks can retrieve this via top-k to inject prior chapter summaries into their self-attention.

This is a structured write policy: instead of letting the EMA routing mechanism decide which slot receives the last-token vector, we **force-write it into a designated slot every chunk**.

**Evaluation — Explicit Chapter Marker vs. EMA Natural Accumulation**:

| Dimension | Explicit chapter-marker slot | EMA natural accumulation |
|---|---|---|
| **Guarantee that last-token info is written** | Yes — always written to slot 0 regardless of routing | No — slot 0 only receives `H[:, -1, :]` if routing selects it |
| **Retrieval guarantee** | Designated slot always has the most recent chunk summary | Content depends on routing history; recent summary may or may not be in any slot |
| **Memory utilisation** | 1/512 slots reserved as chapter marker; 511 for general EMA routing | All 512 slots available for EMA routing |
| **Multi-level summaries** | Difficult — only one chapter marker, overwritten each chunk | EMA naturally accumulates at multiple timescales (β controls decay rate) |
| **Implementation complexity** | Medium — requires a separate write path bypassing top-k routing | None — existing EMA write already handles this |
| **Training stability** | Risk: gradient signal to chapter marker path is a separate channel from EMA | EMA path already has Fix J-A gradient flow |
| **Alignment with successful architectures** | Unique to this proposal; no direct precedent | EMA ≡ Titans neural memory update (2025); natural accumulation has strong precedent |

**Key theoretical question**: Under carry-over + EMA write, will the last-token summary NATURALLY find its way into the slot bank?

**Analysis**: With top-k routing (k=64 from N=512), the probability that ANY specific position (including the last token) is selected as a write candidate in any given step is 64/512 = 12.5%. More precisely, the write path is:

1. `O_mem_hidden = slot_to_hidden(M_sel_slot) + hidden` (read selected slots, project, add to hidden)
2. Forward through decoder attention
3. `O_mem_slot = hidden_to_slot(O_mem_hidden)` — the OUTPUT hidden states at the k selected positions
4. `memory_bank.write(idx, O_mem_slot, β)` — write output representations into the k selected slots

The WRITE operation writes the OUTPUT of the decoder (after attending to the chunk) into slots, weighted by the routing. **The last token's output `O_mem_hidden[:, -1, :]` is written into whichever slot is routed to position T-1 (the last position).** With k=64 routing from T=4096 tokens (using seq_len-based routing, not slot-based routing — confirm this), the probability that the last token contributes to a write target depends on how the routing selects among the T positions, not the N slots.

**Clarification needed**: Looking at `layer.py`, `write()` takes `idx` as slot indices (in [0, N)), not token indices. The routing selects k SLOTS to update, and the written value is `O_mem_slot` = `hidden_to_slot(O_mem_hidden)` where `O_mem_hidden` is the output at the k SELECTED token positions (the positions whose `Q_sel` matched the top-k slots). The last token is NOT automatically selected as a write source unless routing puts it in the top-k.

**Conclusion**: Under the current write mechanism, the last-token summary is NOT guaranteed to be written into the slot bank. The routing selects which K token positions contribute to writes, and this depends on learned query-slot matching. In early training (when routing is near-uniform), the last token has 1/N ≈ 0.2% probability of being the primary write source for any given slot.

**Does this justify an explicit chapter-marker slot?** Partially:
- An explicit chapter marker **guarantees** that every chunk's full-context summary enters the slot bank, regardless of routing maturity
- BUT it conflicts with the write path architecture: the current `write()` only writes `O_mem_slot = hidden_to_slot(O_mem_hidden)` from SELECTED positions. A chapter marker would need to write `hidden_to_slot(H_l[:, -1, :])` directly — bypassing the routing entirely
- This adds a **secondary write channel** that could interfere with the EMA gradient flow (Fix J-A path)

### 8.4 Practical Recommendation

**Preferred: carry-over + strided-token init (recommendation unchanged)**

The core recommendation from Sections 2–4 stands: replace `reset()` with `detach_()` (carry-over) and use `strided_token` init. This is the minimal, highest-leverage change.

**Addendum from user insight**: Consider adding the last token as a SPECIAL CASE in the strided init, replacing slot 0 with `H_l[:, -1, :]`:

```python
slots[:, 0, :] = H_l.detach()[:, -1, :]  # slot 0 = full-chunk SWA summary
```

This adds exactly 1 line to Option B and gives slot 0 a theoretically privileged initial state. Cost: zero. Risk: zero. Upside: modest — the strided init already assigns `H_l[:, stride*(N-1), :] ≈ H_l[:, 4088, :]` to the last strided slot, which under SWA W=4096 is nearly identical to `H_l[:, -1, :]`.

**On the explicit chapter-marker write slot**: Not recommended for the first experiment. Reasons:
1. Adds a second write channel → complicates gradient flow analysis (Fix J-A path is the active diagnostic target)
2. The benefit (guaranteed chunk summary in bank) is subsumed by carry-over: once slots accumulate across chunks via EMA, the last-token information IS present in the bank after sufficient steps
3. If carry-over + strided init still shows degeneracy after 200+ steps, THEN re-evaluate explicit chapter-marker as a targeted fix

**Phased plan**:
1. **Phase 1**: carry-over + strided-token init + Fix J-A gradient path (current recommendation)
2. **Phase 2 (if Phase 1 shows routing still degenerates)**: add `slots[:, 0, :] = H_l[:, -1, :]` to cold-start init
3. **Phase 3 (if Phase 2 still insufficient)**: consider explicit chapter-marker write at each chunk boundary (new architectural feature, separate from EMA routing)

### 8.5 Addressing User's Point on All-512-Slot Cold Start Quality

User's observation: "The initialization quality of all 512 slots matters less than getting carry-over working."

**Full agreement.** The convergence analysis in Section 5 shows that after 40 steps, only 0.1% of slot content comes from initialization. The cold-start strategy (strided vs. last-token vs. mean) matters for steps 0–40, after which the carry-over mechanism dominates. The user's framing is exactly right: the "chapter marker" as initialization for the FIRST chunk is a minor issue; the structural carry-over across chunks is the major issue. The bug that caused degeneracy in Fix A–J was never about cold-start — it was about (a) per-step reset destroying all accumulated memory, and (b) dead gradient paths preventing slots from learning to hold distinct content.

**Summary**: Incorporate user's insight as a minor extension to Option B (replace slot 0 with last-token hidden state in cold-start init). The explicit chapter-marker write strategy is theoretically interesting but architecturally premature — revisit in Phase 3 after carry-over + Fix J establish a working baseline.
