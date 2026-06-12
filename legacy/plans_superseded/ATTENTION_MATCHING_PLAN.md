# Attention Matching + Memory Slots: Implementation Plan

Date: 2026-05-01
Status: PLANNING (no code yet)

---

## 1. Paper Summary

**Attention Matching** (arXiv:2602.16284) is a **training-free** KV cache compression method. Given a full KV cache from a long context, it constructs a compact representation (Ck, Cv, beta) that preserves both attention mass and attention output at the per-KV-head level.

### Core Algorithm (3 Steps)

Given keys K in `[T, d]`, values V in `[T, d]`, and a set of reference queries Q_ref in `[T_ref, d]`:

**Step 1: Key Selection** -- Choose `m` representative keys from the original `T` keys.
- **OMP method**: Greedy orthogonal matching pursuit on K^T K. At each iteration, pick the key most correlated with the residual of the unselected keys.
- **Highest-attention method**: Score each key by `sum_i exp(q_i . k_j / sqrt(d))` using reference queries, pick top-m.

**Step 2: Beta Fitting (Mass Matching)** -- Fit per-token scalar biases via NNLS.
- Goal: make the compact attention mass match the original attention mass for every reference query.
- Feature matrix: `Phi[i,j] = exp(q_i . Ck[j]^T / sqrt(d))` in `[T_ref, m]`
- Target: `a[i] = sum_j exp(q_i . k_j^T / sqrt(d))` for all original keys j, in `[T_ref]`
- Solve: `min || Phi * exp(beta) - a ||^2` s.t. `beta >= 0` (NNLS)
- The bias `exp(beta_j)` lets each retained key absorb the mass of the keys it represents.

**Step 3: Value Fitting (Output Matching)** -- Fit compact values via ordinary least squares.
- Goal: make the compact attention output match the original attention output for every reference query.
- Weighted design matrix: `X[i, j*d : (j+1)*d] = softmax_i[j] * Ck[j]` where softmax uses fitted beta
- Target: `Y[i] = sum_j softmax_original[i,j] * v[j]` (original attention output)
- Solve: `Cv = (X^T X + lambda*I)^{-1} X^T Y` (ridge regression, per KV head)

### Key Properties
- **Training-free**: no gradient updates, works with any pretrained model
- **50x compression** in seconds: 8192 tokens -> 164 keys (2% budget)
- **Near-lossless**: PPL degradation < 0.1 at 8x compression on Llama-3-8B
- **Per-KV-head**: each GQA group compressed independently
- **On-policy queries**: compact layers sequentially so later layers see already-compacted KV

### Reference Queries Generation
1. **Repeat-prefill**: `"{context} Repeat the previous context. {context}"` -> extract queries from the repeat portion
2. **Self-study**: `"{context} Based on the above, what comes next?"` -> extract queries
3. **Context-prefill**: `"{context} Please summarize the above passage."` -> extract queries

### Chunked Compaction for Long Contexts
- KV-based: prefill full context to get all KV, then slice into chunks and compact each chunk independently
- Text-based: split text into chunks, prefill+compact each chunk sequentially

### Nonuniform Head Budgets
- Precompute per-head sensitivity curves: for each head, sweep compression ratio and measure PPL impact
- Greedy allocation: give more budget to sensitive heads, less to robust ones

---

## 2. Proposed Architecture: Attention Matching + Memory Slots

### 2.1 Core Innovation: Trainable Attention Matching with Memory-Slot-Guided Selection

The paper's method is training-free. Our innovation is to make it **trainable** and integrate it with our existing memory slot infrastructure:

1. **Trainable beta (attention bias)** instead of NNLS-fitted: The per-key scalar bias becomes a learnable parameter that the model trains end-to-end, rather than being analytically solved per-inference. This allows the model to learn *what* attention mass pattern is most useful for downstream prediction, not just what best reproduces the original mass.

2. **Memory-slot-guided key selection**: Instead of OMP or highest-attention heuristics, use our existing `MemoryBank` slots as **learned compressed key-value prototypes**. The top-k selector (already implemented) scores slots against the current hidden state, selecting which memory prototypes to attend to. This combines the training-free compression insight (a small set of representatives can reconstruct the full attention) with learned content-addressed retrieval.

3. **Hybrid read**: At inference, the model reads from BOTH (a) the compact KV cache compressed via Attention Matching AND (b) the learned memory slots. The slots capture patterns that AM's analytical compression misses, while AM provides a strong baseline for KV reconstruction.

### 2.2 Architecture Diagram (Forward Pass)

```
Input hidden states H in [B, T, d]
         |
         v
    +----+----+
    | Llama   |
    | Decoder |  (standard self-attention with full KV)
    | Layer L  |
    +----+----+
         |
         v
  H' in [B, T, d]  (output of self-attention, input to memory module)
         |
         +----------------------------+
         |                            |
         v                            v
  +------+-------+          +--------+--------+
  | AM Compress  |          | Memory Slot     |
  | (training-   |          | Retrieval       |
  |  free path)  |          | (top-k selector)|
  +------+-------+          +--------+--------+
         |                            |
         v                            v
  KV_compact:                    M_sel in [B, k, d]
    Ck in [m, d]                 (selected slot content)
    Cv in [m, d]
    beta in [m]
         |                            |
         +----------------------------+
         |                            |
         v                            v
  +------+----------------------------+------+
  | Joint Attention (cross-attend to          |
  | compact KV + memory slots simultaneously) |
  +-------------------------------------------+
         |
         v
  next_hidden in [B, T, d]
         |
         v
  +------+-------+
  | Memory Slot  |
  | Writeback    |
  | (EMA update) |
  +------+-------+
         |
         v
  Updated slots for next layer
```

### 2.3 Two Operating Modes

**Mode A: Training (with learned memory slots)**
- Standard Llama forward with full KV (uncompressed)
- Memory slots are read (top-k) and written (EMA) as in the existing pipeline
- The AM compression module is **not used** during training -- slots learn to compress on their own
- Training teaches the slots what information to retain

**Mode B: Inference (AM + slots hybrid)**
- Prefill with full KV, then **compress KV via AM** to produce Ck, Cv, beta
- Simultaneously retrieve from **trained memory slots** via top-k
- Cross-attend to both compressed KV and slots
- This gives AM's near-lossless KV compression PLUS the slot-based learned retrieval

**Mode C: Pure AM (ablation baseline)**
- AM compression only, no memory slots
- Reproduce the paper's results as a baseline

### 2.4 Implementation Strategy

Rather than implementing all three modes at once, we take a staged approach:

**Stage 1: Pure AM reproduction** (validate the method works on our setup)
**Stage 2: AM + slots hybrid** (add the innovation)

---

## 3. Stage 1: Pure Attention Matching Implementation

### 3.1 New File: `src/memory/mem_space/attention_matching.py`

This file implements the core AM algorithm as a standalone module.

#### Class: `AttentionMatchingCompressor`

```python
class AttentionMatchingCompressor:
    """Training-free KV cache compression via Attention Matching.

    Implements the three-step pipeline from arXiv:2602.16284:
    1. Key selection (OMP or highest-attention)
    2. Beta fitting (NNLS for mass matching)
    3. Value fitting (least squares for output matching)

    This is NOT an nn.Module -- it has no trainable parameters.
    It operates on extracted KV caches from a pretrained model.
    """

    def __init__(self, model, compression_ratio=8, method='omp',
                 ref_query_mode='repeat_prefill', ridge_lambda=1e-4):
        ...

    def select_keys_omp(self, K, budget):
        """Orthogonal Matching Pursuit key selection.

        Args:
            K: [T, d] original keys for one KV head
            budget: number of keys to select (m)

        Returns:
            indices: [m] selected key indices
        """
        # Greedy: pick key with highest correlation to residual
        # residual = K - K[selected] @ K[selected]^T @ K  (projection residual)
        ...

    def select_keys_highest_attn(self, K, Q_ref, budget):
        """Highest-attention-score key selection.

        Args:
            K: [T, d] original keys
            Q_ref: [T_ref, d] reference queries
            budget: number of keys to select

        Returns:
            indices: [m] selected key indices
        """
        # Score: for each key j, sum_i exp(q_i . k_j / sqrt(d))
        ...

    def fit_beta_nnls(self, Q_ref, Ck, a_target):
        """Fit per-key bias beta via Nonnegative Least Squares.

        Args:
            Q_ref: [T_ref, d] reference queries
            Ck: [m, d] selected compact keys
            a_target: [T_ref] target attention mass per query

        Returns:
            beta: [m] per-key log-bias values
        """
        # Feature matrix Phi[i,j] = exp(q_i . Ck[j]^T / sqrt(d))
        # Solve min ||Phi @ exp(beta) - a_target||^2 s.t. beta >= 0
        # Use scipy.optimize.nnls or torch-based projected gradient descent
        ...

    def fit_values_lstsq(self, Q_ref, K, V, Ck, beta):
        """Fit compact values via ridge regression.

        Args:
            Q_ref: [T_ref, d] reference queries
            K: [T, d] original keys
            V: [T, d] original values
            Ck: [m, d] compact keys
            beta: [m] fitted biases

        Returns:
            Cv: [m, d] compact values
        """
        # Compute compact softmax weights with bias
        # attn_weights[i,j] = exp(q_i . Ck[j]^T / sqrt(d) + beta[j])
        # Normalize per query
        # X[i, j*d:(j+1)*d] = attn_weights[i,j] * Ck[j]
        # Y[i] = sum_j softmax_orig[i,j] * V[j]
        # Solve Cv = (X^T X + lambda*I)^-1 X^T Y
        ...

    def compact_layer(self, K, V, Q_ref, budget):
        """Full compaction pipeline for one KV head of one layer.

        Args:
            K: [T, d_head] original keys
            V: [T, d_head] original values
            Q_ref: [T_ref, d_head] reference queries
            budget: target compact size (m)

        Returns:
            Ck: [m, d_head] compact keys
            Cv: [m, d_head] compact values
            beta: [m] per-key biases
        """
        # Step 1: key selection
        if self.method == 'omp':
            indices = self.select_keys_omp(K, budget)
        else:
            indices = self.select_keys_highest_attn(K, Q_ref, budget)

        Ck = K[indices]

        # Step 2: beta fitting
        a_target = torch.exp(Q_ref @ K.T / math.sqrt(K.shape[-1])).sum(dim=-1)
        beta = self.fit_beta_nnls(Q_ref, Ck, a_target)

        # Step 3: value fitting
        Cv = self.fit_values_lstsq(Q_ref, K, V, Ck, beta)

        return Ck, Cv, beta

    def compact_model(self, input_ids, compression_ratio):
        """Compact all layers of the model sequentially.

        Args:
            input_ids: [1, T] input token IDs
            compression_ratio: target compression (e.g., 8 = keep 1/8 keys)

        Returns:
            compact_kv: list of (Ck, Cv, beta) per layer
        """
        # On-policy compaction:
        # For layer 0: use reference queries from input
        # For layer l > 0: run forward through layers 0..l-1 (already compacted)
        #   to extract queries at layer l
        # This ensures later layers see compacted KV from earlier layers
        ...
```

#### Reference Query Generation

```python
def generate_reference_queries(model, input_ids, mode='repeat_prefill'):
    """Generate reference queries for compaction.

    Args:
        model: pretrained LlamaModel
        input_ids: [1, T] original input
        mode: 'repeat_prefill', 'self_study', or 'context_prefill'

    Returns:
        ref_queries_per_layer: list of [T_ref, d] tensors, one per layer
    """
    if mode == 'repeat_prefill':
        # Construct: "{input_ids} Repeat the previous context. {input_ids}"
        # Extract hidden states from the second copy's positions
        repeat_prompt = tokenizer.encode("Repeat the previous context.")
        extended_ids = torch.cat([
            input_ids,
            torch.tensor([repeat_prompt]),
            input_ids,
        ], dim=-1)
        # Forward, extract per-layer hidden states at the second copy positions
        ...
    elif mode == 'self_study':
        prompt = tokenizer.encode("Based on the above, what comes next?")
        extended_ids = torch.cat([input_ids, torch.tensor([prompt])], dim=-1)
        # Extract hidden states at prompt positions
        ...
```

### 3.2 Evaluation Script: `scripts/eval_attention_matching.py`

```python
"""Evaluate Attention Matching compression on pg19 / WikiText-2 / NIAH.

Compaction happens at inference time (training-free).
Reports: PPL at various compression ratios (4x, 8x, 16x, 32x, 50x)
"""
```

### 3.3 Integration Point: Modify `MemorySpaceConfig`

Add to `src/memory/mem_space/config.py`:

```python
# Attention Matching mode.
# When True: compaction is performed at inference time via the AM algorithm.
# The model uses compressed (Ck, Cv, beta) instead of full KV for attention.
use_attention_matching: bool = False

# Compression ratio: keep 1/am_compaction_ratio of original KV tokens.
# E.g., 8 means keep 12.5% of keys (8192 -> 1024).
am_compaction_ratio: int = 8

# Key selection method: 'omp' or 'highest_attn'
am_key_selection: str = 'omp'

# Reference query generation: 'repeat_prefill', 'self_study', 'context_prefill'
am_ref_query_mode: str = 'repeat_prefill'

# Ridge regression lambda for value fitting
am_ridge_lambda: float = 1e-4

# Chunk size for chunked compaction (0 = compress entire context at once)
am_chunk_size: int = 0

# Beta stability clamp: clamp exp(beta) to [e^{-clamp}, e^{clamp}]
am_beta_clamp: float = 3.0

# Prune keys with beta < prune_threshold (OMP method)
am_beta_prune_threshold: float = -7.0
```

---

## 4. Stage 2: AM + Memory Slots Hybrid

### 4.1 New Class: `AttentionMatchingMemory` in `selector.py`

This class bridges AM compression with the existing memory slot infrastructure.

```python
class AttentionMatchingMemory(nn.Module):
    """Hybrid Attention Matching + Memory Slots.

    Combines training-free AM KV compression with trainable memory slots.

    At training time:
        - Standard Llama forward with full KV
        - Memory slots are read (top-k) and written (EMA) as usual
        - AM module is inactive (no compression during training)

    At inference time:
        - AM compresses the KV cache (training-free)
        - Memory slots are read via top-k (trained during training)
        - Model attends to both compressed KV and memory slots
        - Slots capture patterns AM misses; AM provides near-lossless KV baseline

    Trainable params:
        - Top-k selector (Q_sel, K_sel): learned from training
        - Memory slots (initialized from training, frozen at inference)
        - slot_to_hidden, hidden_to_slot projections: learned from training
        - slot_output_gate: learned Flamingo-style gate
    """

    def __init__(self, d_model, n_heads, n_kv_heads, n_slots=128,
                 top_k=16, selector_dim=128, slot_dim=None,
                 compression_ratio=8, key_selection='omp'):
        super().__init__()
        ...

        # Reuse existing TopKSelector infrastructure for slot retrieval
        self.selector = TopKSelector(d_model, slot_dim, selector_dim, top_k, n_slots)

        # Slot projections (same as existing TopKSelector path)
        self.slot_to_hidden = nn.Linear(slot_dim, d_model, bias=False)
        self.hidden_to_slot = nn.Linear(d_model, slot_dim, bias=False)

        # Flamingo-style output gate
        self.slot_output_gate = nn.Parameter(torch.tensor(0.0))

        # AM compressor (not nn.Module, no trainable params)
        self.am_compressor = AttentionMatchingCompressor(
            compression_ratio=compression_ratio,
            method=key_selection,
        )
```

### 4.2 Forward Path: `_forward_attention_matching` in `layer.py`

```python
def _forward_attention_matching(self, hidden_states, position_embeddings, **kwargs):
    """Forward pass with AM + memory slots hybrid.

    Training: identical to existing top-k forward (AM not used).
    Inference: AM-compressed KV + memory slot retrieval.
    """
    cfg = self.cfg
    B, T, D = hidden_states.shape
    k_slots = cfg.top_k

    if self.training:
        # During training: use standard top-k slot injection
        # (same as existing _forward_top_k path)
        # This trains the selector, slots, and gate
        return self._forward_top_k(hidden_states, position_embeddings, **kwargs)

    # ---- Inference: AM + slots hybrid ----

    # 1. Standard decoder forward to get KV cache
    bypass_out = self.wrapped_layer(
        hidden_states, attention_mask=None,
        position_embeddings=position_embeddings, **kwargs)
    bypass_h = bypass_out[0] if isinstance(bypass_out, tuple) else bypass_out

    # 2. Extract K, V from the decoder layer's self-attention
    # (we need to hook into the attention module to capture K, V)
    K_full = ...  # [B, n_kv_heads, T, d_head]
    V_full = ...  # [B, n_kv_heads, T, d_head]

    # 3. Compress KV via AM
    Ck, Cv, beta = self.am_compressor.compact_layer(K_full, V_full, Q_ref, budget)

    # 4. Retrieve memory slots via top-k selector
    slots = self.memory_bank.slots  # [B, N, slot_dim]
    pool_of_H = hidden_states.mean(dim=1)  # [B, d]
    scores, idx = self.selector(pool_of_H, slots)  # [B, N], [B, k]
    M_sel = self.slot_to_hidden(slots.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)))

    # 5. Build extended sequence: [M_sel, hidden_states]
    #    Plus inject compressed KV into the attention layer's KV cache
    #    so the decoder attends to both slot tokens and compressed KV

    # Option A: Replace the layer's KV cache with (Ck, Cv, beta)
    #   The decoder's self-attention sees compressed KV instead of full KV
    #   Plus k slot tokens prepended

    # Option B: Run cross-attention to slots, then replace KV with compressed
    #   for subsequent self-attention

    # Option A is cleaner: single forward pass with compressed KV + slots
    ...
```

### 4.3 Key Design Decision: How Slots Interact with AM

There are three viable options for combining AM and slots:

**Option A: Sequential (AM first, slots add information)**
1. AM compresses the full KV cache -> Ck, Cv, beta
2. Slots retrieve additional learned prototypes via top-k
3. Decoder attends to: [slot_tokens, content_tokens] with KV=[Ck, Cv, slot_KV]
4. Pro: clean separation, easy to ablate
5. Con: slots don't benefit from AM's analytical optimization

**Option B: Slot-Guided Selection (slots tell AM what's important)**
1. Top-k selector scores all positions in the full KV cache
2. Highest-scored positions become the key selection for AM (replacing OMP/highest-attn)
3. AM then fits beta and Cv for these slot-selected keys
4. Pro: learned content-addressing replaces heuristic selection
5. Con: selector was trained on slots, not on KV positions; domain mismatch

**Option C: AM for long-range, slots for short-range**
1. Split the KV cache: recent window (last W tokens) stays uncompressed
2. Remaining (older) KV compressed via AM
3. Memory slots capture the "gist" of the entire context
4. Decoder attends to: [slot_tokens, recent_window, compressed_old_kv]
5. Pro: best of all worlds; AM handles bulk compression, slots handle semantic compression
6. Con: more complex pipeline

**Recommended: Start with Option A (simplest), then explore Option C if results are promising.**

---

## 5. Implementation Steps

### Step 1: Create `src/memory/mem_space/attention_matching.py`
- Implement `AttentionMatchingCompressor` class
- OMP key selection (greedy, per KV head)
- Highest-attention key selection
- Beta fitting via NNLS (use `scipy.optimize.nnls` or implement projected gradient descent in torch)
- Value fitting via ridge regression (`torch.linalg.lstsq`)
- Chunked compaction support
- On-policy sequential layer compaction

### Step 2: Create `scripts/eval_attention_matching.py`
- Load Llama-3-8B
- Run AM compaction on pg19 / WikiText-2 validation chunks
- Report PPL at compression ratios: 4x, 8x, 16x, 32x
- Compare with paper's reported numbers to validate reproduction

### Step 3: Add config fields to `src/memory/mem_space/config.py`
- `use_attention_matching`, `am_compaction_ratio`, `am_key_selection`, etc.
- Validation in `__post_init__`

### Step 4: Add `AttentionMatchingMemory` class to `src/memory/mem_space/selector.py`
- Combines existing TopKSelector with AMCompressor
- Training path delegates to TopKSelector
- Inference path uses AM compression + slot retrieval

### Step 5: Add `_forward_attention_matching` to `src/memory/mem_space/layer.py`
- Dispatch from `MemorySpaceLayer.forward()` based on `cfg.use_attention_matching`
- Training: identical to top-k path
- Inference: AM-compressed KV + slot retrieval hybrid

### Step 6: Update `scripts/train_mem_space_pg19.py`
- Add CLI args for AM mode: `--use_attention_matching`, `--am_compaction_ratio`, etc.
- Training phase: identical to existing slot training (AM not used during training)
- Evaluation phase: apply AM compression at inference time, measure PPL

### Step 7: Update `src/memory/mem_space/patch.py`
- Support `use_attention_matching` config flag
- Create `AttentionMatchingMemory` instead of `TopKSelector` when enabled

---

## 6. File Change Summary

| File | Action | Lines Changed (est.) |
|------|--------|---------------------|
| `src/memory/mem_space/attention_matching.py` | **NEW** | ~400 |
| `src/memory/mem_space/selector.py` | ADD class `AttentionMatchingMemory` | ~150 |
| `src/memory/mem_space/layer.py` | ADD method `_forward_attention_matching` | ~100 |
| `src/memory/mem_space/config.py` | ADD AM config fields | ~30 |
| `src/memory/mem_space/patch.py` | ADD AM dispatch | ~20 |
| `scripts/eval_attention_matching.py` | **NEW** | ~200 |
| `scripts/train_mem_space_pg19.py` | ADD AM CLI args + eval path | ~50 |
| **Total** | | **~950** |

---

## 7. Expected Parameters and Compute

### Trainable Parameters (Training Phase)
When training with AM + slots (same as existing slot training):
- TopKSelector: Q_sel (4096 x 128) + K_sel (4096 x 128) = ~1M params
- slot_to_hidden: (4096 x 4096) = ~16M params
- hidden_to_slot: (4096 x 4096) = ~16M params (if not frozen)
- slot_output_gate: 1 param
- Per layer total: ~33M trainable params
- 32 layers: ~1.06B trainable params
- Backbone: frozen Llama-3-8B (~8B params, not trained)

### AM Compression Compute (Inference Phase)
- OMP key selection: O(T * m * d) per head -- greedy with projections
- Beta fitting (NNLS): O(T_ref * m * iterations) -- typically converges in ~50 iterations
- Value fitting (ridge): O(T_ref * m * d^2) -- dominated by the matrix inverse
- For T=8192, m=1024 (8x), d=128, T_ref=512: ~2-5 seconds per layer on A100
- Full model (32 layers): ~1-3 minutes total compaction time

### Memory Requirements
- Training: same as current slot training (~40GB for Llama-3-8B + optimizer states on 8 GPUs)
- Inference (with AM): needs to hold full KV cache during compaction, then discard
  - Full KV for 8192 tokens: 8192 * 128 * 32 layers * 2 (K+V) * 2 bytes (bf16) = ~128 MB
  - Plus compressed KV: negligible (~2 MB at 8x)
  - Total: ~130 MB KV cache during compaction, ~2 MB after

---

## 8. Experimental Plan

### Experiment 1: Pure AM Reproduction (Validate)
- Run AM compression on Llama-3-8B with pg19 validation
- Compression ratios: 4x, 8x, 16x, 32x, 50x
- Key selection: OMP vs highest-attn
- Reference queries: repeat-prefill vs self-study
- Target: reproduce paper's PPL numbers (within 0.1 PPL)

### Experiment 2: AM + Slots (Training then Inference)
- Train memory slots on pg19 chunks (same setup as existing slot training)
- At inference: AM-compressed KV + trained slot retrieval
- Compare: AM-only vs AM+slots vs slots-only
- Metrics: PPL, NIAH accuracy

### Experiment 3: Ablations
- Slot count: 32, 64, 128, 256 slots
- Top-k: 4, 8, 16, 32
- AM budget interaction: fixed total budget, split between AM and slots
- Option A vs Option C architecture comparison

---

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| NNLS too slow for real-time inference | Medium | Pre-compute at chunk boundaries; use torch-based projected GD instead of scipy |
| OMP degenerates with GQA (shared K/V heads) | Low | Apply OMP per KV head independently (paper does this) |
| Slot training doesn't generalize to AM inference | Medium | Option A (slots independent of AM) avoids this coupling |
| Ridge regression unstable for small budgets | Low | Clamp beta, regularize with lambda=1e-4, skip heads where cond(X) > 1e10 |
| Memory bank lifecycle conflicts with AM | Medium | Keep AM stateless (recompute each inference); slots are separate persistent state |

---

## 10. Success Criteria

1. **Stage 1 complete**: Pure AM reproduces paper PPL within 0.1 on Llama-3-8B with pg19
2. **Stage 2 complete**: AM+slots achieves >= 0.5 PPL improvement over AM-only at same compression ratio
3. **End-to-end**: Full pipeline (train slots, then inference with AM+slots) runs on pg19 + NIAH without NaN or divergence
