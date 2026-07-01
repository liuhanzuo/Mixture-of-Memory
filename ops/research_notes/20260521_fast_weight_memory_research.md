# Fast Weight / Fast Memory Module Research Report

**Date**: 2026-05-21  
**Purpose**: Literature review + architecture design for a continuous fast-weight memory to complement mem_space's discrete slot routing

---

## 1. Literature Summary

### 1.1 Fast Weight Programmers (Schmidhuber 1991; Schlag et al. 2021)

**Paper**: "Linear Transformers Are Secretly Fast Weight Programmers" ([arXiv:2102.11174](https://arxiv.org/abs/2102.11174))

**Core Insight**: Linear attention is mathematically equivalent to a Fast Weight Programmer (FWP) from the 1990s. A "slow" network programs a "fast" weight matrix through outer products:

```
A_t = A_{t-1} + v_t ⊗ φ(k_t)         # additive outer-product update
y_t = A_t · φ(q_t)                     # retrieval via matrix-vector product
```

**Pros**: Simple, O(1) update per token, O(d²) memory, clean theoretical framework  
**Cons**: Capacity limited by interference (superposition catastrophe); no forgetting mechanism → memory saturates

**Delta Rule Extension** (error-correcting):
```
A_t = A_{t-1} + β_t · (v_t - A_{t-1} · φ(k_t)) ⊗ φ(k_t)
```
This selectively corrects existing associations rather than blindly accumulating, dramatically improving capacity.

---

### 1.2 DeltaNet (Yang et al. 2024)

**Paper**: "Parallelizing Linear Transformers with the Delta Rule over Sequence Length" ([arXiv:2406.06484](https://arxiv.org/html/2406.06484v6))

**Key Contributions**:
- Efficient parallel algorithm for the delta rule via **generalized Householder transformations** + WY representation
- Scales to 1.3B parameters trained on 100B tokens
- **Outperforms Mamba and GLA** in perplexity and zero-shot tasks
- Hybrid DeltaNet + sliding-window attention achieves best-of-both

**Update Rule (RNN form)**:
```
S_t = S_{t-1} + β_t · (v_t - S_{t-1} · k_t) · k_t^T    # delta rule
o_t = S_t · q_t                                           # retrieval
```

**Relevance**: DeltaNet's delta rule is the strongest known update rule for linear-complexity associative memory. It's the natural choice for our fast weight module.

---

### 1.3 Gated Linear Attention (GLA) — Yang et al. 2024

**Paper**: [arXiv:2312.06635](https://arxiv.org/abs/2312.06635)

**Innovation**: Data-dependent gating on the recurrent state:
```
S_t = G_t ⊙ S_{t-1} + k_t ⊗ v_t
o_t = S_t · q_t
```
where `G_t` is input-conditioned (learned per-token forget gate).

**Chunk-wise parallelism**: O(LCd + Ld²) — inter-chunk recurrence + intra-chunk parallelism.

**Performance**: Competitive with LLaMA/Mamba at 1.3B scale. Length extrapolation from 2K→20K+.

---

### 1.4 Gated Slot Attention (GSA) — 2024

**Paper**: [arXiv:2409.07146](https://arxiv.org/abs/2409.07146)

**Key Insight**: Combines bounded-memory control (ABC) with GLA-style gating. Uses a **two-layer GLA linked via softmax** for context-aware memory reading + adaptive forgetting.

**Extremely Relevant**: GSA is literally "gated slot attention" — it merges the slot/bounded-memory concept with linear attention recurrence. This is the closest existing work to what we want to build.

---

### 1.5 Titans (Google 2024/2025)

**Paper**: "Learning to Memorize at Test Time"

**Architecture**: Three memory types:
1. **Short-term**: Standard attention (local window)
2. **Long-term (LMM)**: Neural memory module updated via surprise-based gradient descent
3. **Persistent**: Static learned parameters

**LMM Update Rule (surprise-based)**:
```
M_t = (1 - α_t) · M_{t-1} - θ_t · ∇ℓ(M_{t-1}; x_t)
```
where `θ_t` is a data-dependent learning rate and `α_t` is a forget gate.

**Integration Variants**:
- **MAC** (Memory as Context): Concatenate LMM output with input before attention
- **MAG** (Memory as Gate): Gated blend of attention + memory outputs
- **MAL** (Memory as Layer): Standalone compression layer

**Relevance**: The MAC variant (prepend memory to context) is exactly our current approach with L1 slots. The MAG variant (gated blend) is closer to what we'd want for the fast weight module — an additive/gated contribution that doesn't require KV-prepend overhead.

---

### 1.6 Infini-attention (Google 2024)

**Paper**: [arXiv:2404.07143](https://arxiv.org/pdf/2404.07143.pdf)

**Design**: Combines local dot-product attention with a compressive memory bank:
```
# After computing local attention within a segment:
M_s = M_{s-1} + σ(K_s)^T · V_s                # update compressive memory
z_s = z_{s-1} + Σ σ(K_s)                      # normalization term
A_mem = (σ(Q_s) · M_s) / (σ(Q_s) · z_s)      # memory retrieval
output = sigmoid(β) · A_mem + (1 - sigmoid(β)) · A_dot  # gated blend
```

**Pros**: Elegant, minimal extra parameters (just β gate per head), reuses existing KV  
**Cons**: Pure linear attention accumulation (no delta rule) → capacity degrades with length; no per-key forgetting

**Relevance**: Very close to what we want! The gated blend of local attention + compressive memory is the exact pattern. But we can do better with a delta-rule update instead of pure accumulation.

---

### 1.7 xLSTM — mLSTM (Hochreiter et al. 2024)

**Core Innovation**: **Matrix memory** `C_t ∈ ℝ^(d×d)` with covariance-style update:
```
C_t = f_t · C_{t-1} + i_t · v_t · k_t^T      # exponential gating
h_t = C_t · q_t / n_t                          # retrieval + normalization
```

**Exponential gating**: `f_t = exp(w_f · x_t)`, `i_t = exp(w_i · x_t)` — more decisive than sigmoid, allows values >1 for stronger memory overwrites.

**Relevance**: The mLSTM matrix memory IS a fast weight matrix with exponential gating. The stabilizer mechanism prevents overflow. This is one of the best-performing continuous memory designs.

---

### 1.8 RWKV-7 "Goose" (2025)

**Key Mechanism**: Generalized Delta Rule with vectorized gating:
```
state_t = (1 - δ_t) · state_{t-1} + η_t · v_t
```
- Separated delete & add mechanisms
- 30% lower perplexity than Mamba on PG19
- Only 2 layers needed for complex state tracking

---

### 1.9 Test-Time Training (TTT) Layers (Sun et al. 2024)

**Paper**: [arXiv:2407.04620](https://arxiv.org/abs/2407.04620) — ICML 2025 Spotlight

**Radical Approach**: Hidden state IS a machine learning model (linear or MLP), updated via self-supervised gradient descent at inference time.

**Update**: `W_t = W_{t-1} - η · ∇L_ssl(W_{t-1}, x_t)` where L_ssl is a reconstruction loss.

**Connection to Fast Weights**: TTT generalizes the linear attention = fast weight equivalence. When W is linear and L is MSE, the gradient update reduces to a delta-rule-like update.

**Performance**: Linear complexity, surpasses Transformers at 8K context, matches Mamba's speed.

---

### 1.10 RetNet (Microsoft 2023)

**Paper**: [arXiv:2307.08621](https://arxiv.org/pdf/2307.08621.pdf)

**Retention with exponential decay**:
```
S_t = γ · S_{t-1} + K_t^T · V_t
o_t = Q_t · S_t
```

Three computation modes: parallel (training), recurrent (inference), chunk-wise (hybrid).

**Efficiency**: 3.4× lower memory, 8.4× higher throughput, 15.6× lower latency vs. Transformers.

---

### 1.11 Fast-weight Product Key Memory (FwPKM) — Sakana AI 2025

**Paper**: [arXiv:2601.00671](https://arxiv.org/abs/2601.00671)

**Innovation**: Makes Product Key Memory dynamic via chunk-level gradient descent. Complements static semantic PKM with episodic fast-weight memory.

**Performance**: Trained on 4K, generalizes to 128K contexts. Iterative retrieval (second pass) boosts accuracy from <10% to >70%.

**Key Design**: Inverse Distance Weighting for scoring, gated importance for updates.

---

### 1.12 Trellis (Google 2024)

**Paper**: [arXiv:2512.23852](https://arxiv.org/abs/2512.23852v1)

**Approach**: Fixed-size memory slots compressed via online gradient descent + learnable forget gate. Two-pass recurrent compression.

**Relevance**: Shows gradient-based online compression into fixed slots can outperform heuristic eviction.

---

### 1.13 Block-Recurrent Transformer (NeurIPS 2022) + Block-State Transformer (DeepMind 2023)

**BRT**: Cross-attention between current block and recurrent states. LSTM-style gating for state updates.

**BST** ([arXiv:2306.09539](https://arxiv.org/html/2306.09539v1)): Replaces BRT's recurrent units with State Space Models. 10× speedup, linear complexity.

---

## 2. Design Space Analysis

### 2.1 Update Rules (ranked by capacity/quality)

| Update Rule | Formula | Capacity | Parallelizable | Forgetting |
|---|---|---|---|---|
| **Additive (linear attn)** | S += v⊗k | Low (interference) | Yes (cumsum) | No |
| **Exponential decay** | S = γ·S + v⊗k | Medium | Yes (RetNet) | Global fixed-rate |
| **Data-dependent gate (GLA)** | S = G⊙S + v⊗k | High | Yes (chunk-wise) | Per-feature, adaptive |
| **Delta rule** | S += β·(v - S·k)⊗k | Very High | Yes (WY repr) | Implicit (correction) |
| **Gated Delta** | S = G⊙S + β·(v-S·k)⊗k | Highest | Yes (Gated DeltaNet) | Both explicit + implicit |
| **Gradient descent (TTT)** | S -= η·∇L(S,x) | Highest (nonlinear) | Harder | Via decay term |

**Recommendation**: **Gated Delta Rule** — best capacity with efficient parallelization. The gate provides explicit forgetting (essential for long sequences), the delta rule provides error-correcting capacity.

### 2.2 Retrieval Mechanisms

| Method | Formula | Integration Effort |
|---|---|---|
| **Linear query** | o = S·q | Minimal — pure matrix-vector product |
| **Additive bias** | h' = h + gate·(S·q) | Low — add to hidden states |
| **Multiplicative gate (Infini)** | h' = β·(S·q) + (1-β)·h_attn | Medium — learnable per-head gate |
| **Cross-attention** | h' = softmax(Q·M^T)·M | High — extra attention op |
| **KV-prepend** | [M_tokens; H] → attention | High — doubles sequence length |

**Recommendation**: **Additive bias with learned gate** (Infini-attention style). This is the simplest, adds O(d²) compute per token, and can be fused into the existing attention output.

### 2.3 Placement in Transformer

| Position | Pros | Cons |
|---|---|---|
| **Before attention (input)** | Enriches Q/K/V with memory context | May interfere with attention patterns |
| **After attention (output)** | Clean separation of roles | Memory can't influence attention routing |
| **Parallel to attention** | Independent processing, gated merge | Best for our case — doesn't change attention |
| **Replace attention** | Full linear-time model | Incompatible with pretrained LLM |

**Recommendation**: **Parallel to attention with gated merge** — exactly the MAG (Memory as Gate) variant from Titans. The fast weight operates in parallel, producing a "memory-informed" representation that's gated-blended with the standard attention output. This:
- Doesn't change the base LLM's attention at all
- Can be initialized to zero contribution (safe for continued pretraining)
- Works naturally with our existing bypass + slot_delta architecture

### 2.4 Per-Layer vs. Shared

| Design | Pros | Cons |
|---|---|---|
| **Per-layer** | Each layer has its own memory context | 32× memory cost, no cross-layer sharing |
| **Shared (like our L1 bank)** | Single state, BPTT through depth | State gets overwritten 32× per chunk |
| **Hybrid: shared state, per-layer projections** | Best of both | Moderate complexity |

**Recommendation**: **Per-layer fast weight state with shared projections for K/V generation**. Unlike discrete slots (where sharing makes sense because slots are content-addressed), the fast weight matrix accumulates a running summary — sharing it across layers would mean each layer overwrites the previous layer's contribution. Per-layer states with shared projection weights keeps parameter count low while allowing layer-specific memory.

### 2.5 Dimensionality / Capacity

The fast weight matrix S ∈ ℝ^(d_k × d_v). For a model with hidden_size=4096:
- Full d×d = 4096×4096 = 64MB per layer per sample in fp16 → **too expensive for 32 layers**
- Compressed: d_k=256, d_v=256 → 256KB per layer → very manageable
- Middle ground: d_k=512, d_v=512 → 1MB per layer → good capacity/cost trade-off

With multi-head decomposition (like GLA): H heads × (d_k/H × d_v/H) states. For H=16, d_k=d_v=256: each head has a 16×16 matrix → very low capacity. Better to use fewer, larger heads for the fast weight (e.g., H=4, each 64×64).

**Recommendation**: Use **d_k = d_v = d_model // num_heads_fw** where num_heads_fw=4-8. This gives 512×512 or 256×256 per head — reasonable capacity without blowing up memory.

---

## 3. Recommended Architecture: "FastMem" Module

### 3.1 Overview

A **per-layer Gated DeltaNet module** that runs in parallel with standard attention, capturing a continuous running summary of ALL tokens. Its output is gated-blended with the attention output, complementing the discrete slot memory.

### 3.2 Architecture Pseudocode

```python
class FastMemModule(nn.Module):
    """Per-layer fast weight memory using Gated Delta Rule.
    
    Captures a continuous summary of ALL tokens (complementing the
    discrete top-k slot routing which only stores 12.5% of tokens).
    
    Memory state: S ∈ [B, H_fw, d_k, d_v] per layer
    where H_fw = num_fast_heads (e.g., 4-8)
    """
    
    def __init__(self, d_model, num_fast_heads=4, d_state=128):
        self.d_model = d_model
        self.H = num_fast_heads
        self.d_k = d_state       # key dim per head
        self.d_v = d_state       # value dim per head
        
        # Projections: hidden → (key, value, query, gate)
        self.W_k = nn.Linear(d_model, num_fast_heads * d_state, bias=False)
        self.W_v = nn.Linear(d_model, num_fast_heads * d_state, bias=False)
        self.W_q = nn.Linear(d_model, num_fast_heads * d_state, bias=False)
        
        # Data-dependent forget gate (GLA-style)
        self.W_gate = nn.Linear(d_model, num_fast_heads * d_state, bias=False)
        
        # Delta rule learning rate (data-dependent)
        self.W_beta = nn.Linear(d_model, num_fast_heads, bias=True)
        
        # Output projection: concat all heads → d_model
        self.W_o = nn.Linear(num_fast_heads * d_v, d_model, bias=False)
        
        # Fusion gate: controls how much fast-mem output contributes
        # Init to small negative → sigmoid ≈ 0.1 at start (safe for cont. pretrain)
        self.fusion_gate = nn.Parameter(torch.full((d_model,), -2.0))
        
        # Init projections small for safe continued pretraining
        nn.init.normal_(self.W_k.weight, std=0.01)
        nn.init.normal_(self.W_v.weight, std=0.01)
        nn.init.normal_(self.W_q.weight, std=0.01)
        nn.init.normal_(self.W_o.weight, std=0.01)
    
    def forward(self, hidden_states, memory_state=None):
        """
        Args:
            hidden_states: [B, T, d_model] — current chunk's hidden states
            memory_state:  [B, H, d_k, d_v] or None (cold start)
        
        Returns:
            output: [B, T, d_model] — fast-mem contribution (to be added to attn output)
            new_state: [B, H, d_k, d_v] — updated memory state for next chunk
        """
        B, T, d = hidden_states.shape
        H, d_k, d_v = self.H, self.d_k, self.d_v
        
        # Project to multi-head format
        k = self.W_k(hidden_states).view(B, T, H, d_k)  # [B, T, H, d_k]
        v = self.W_v(hidden_states).view(B, T, H, d_v)  # [B, T, H, d_v]
        q = self.W_q(hidden_states).view(B, T, H, d_k)  # [B, T, H, d_k]
        
        # Normalize keys (critical for stable delta rule)
        k = F.normalize(k, dim=-1)
        
        # Data-dependent forget gate: per-head, per-feature
        # gate ∈ (0, 1) — how much of old state to retain
        gate = torch.sigmoid(
            self.W_gate(hidden_states).view(B, T, H, d_k)
        )  # [B, T, H, d_k] — applied to rows of S
        
        # Data-dependent learning rate for delta rule
        beta = torch.sigmoid(self.W_beta(hidden_states))  # [B, T, H]
        
        # Initialize state if cold start
        if memory_state is None:
            S = torch.zeros(B, H, d_k, d_v, 
                          device=hidden_states.device, 
                          dtype=hidden_states.dtype)
        else:
            S = memory_state  # [B, H, d_k, d_v]
        
        # === Chunk-wise recurrence (can be parallelized within chunk) ===
        outputs = []
        for t in range(T):
            k_t = k[:, t]           # [B, H, d_k]
            v_t = v[:, t]           # [B, H, d_v]
            q_t = q[:, t]           # [B, H, d_k]
            gate_t = gate[:, t]     # [B, H, d_k]
            beta_t = beta[:, t]     # [B, H]
            
            # 1. Apply forget gate (element-wise on key dimension of S)
            #    S = diag(gate_t) @ S — retains gate fraction of each row
            S = gate_t.unsqueeze(-1) * S  # [B, H, d_k, d_v]
            
            # 2. Delta rule update:
            #    error = v_t - S^T @ k_t  (what the memory SHOULD output vs what it does)
            #    S += beta * error ⊗ k_t
            retrieved = torch.einsum('bhkv,bhk->bhv', S, k_t)  # [B, H, d_v]
            error = v_t - retrieved                              # [B, H, d_v]
            # Outer product update, scaled by beta
            delta = torch.einsum('bhv,bhk->bhkv', error, k_t)  # [B, H, d_k, d_v]
            S = S + beta_t.unsqueeze(-1).unsqueeze(-1) * delta
            
            # 3. Retrieve: o_t = S^T @ q_t
            o_t = torch.einsum('bhkv,bhk->bhv', S, q_t)  # [B, H, d_v]
            outputs.append(o_t)
        
        # Stack outputs: [B, T, H, d_v]
        output = torch.stack(outputs, dim=1)
        
        # Reshape and project to d_model
        output = output.reshape(B, T, H * d_v)
        output = self.W_o(output)  # [B, T, d_model]
        
        # Apply fusion gate (sigmoid per-feature)
        fusion = torch.sigmoid(self.fusion_gate)  # [d_model]
        output = fusion * output
        
        return output, S.detach()  # detach state for inter-chunk break
```

### 3.3 Integration with mem_space (in MemorySpaceLayer.forward)

```python
# Current flow:
#   bypass_h = wrapped_layer(hidden_states)
#   ext_h = wrapped_layer(extended_hidden)  # with L1/L2/L3 prepend
#   slot_delta = ext_h[:, prefix:, :] - bypass_h
#   next_hidden = bypass_h + alpha * slot_delta
#
# NEW flow with FastMem:
#   bypass_h = wrapped_layer(hidden_states)
#   ext_h = wrapped_layer(extended_hidden)
#   slot_delta = ext_h[:, prefix:, :] - bypass_h
#   
#   # FastMem runs on the PRE-attention hidden states (or post-attention — ablate both)
#   fast_mem_output, new_fast_state = self.fast_mem(hidden_states, self._fast_mem_state)
#   self._fast_mem_state = new_fast_state
#   
#   # Combine: bypass + discrete_slot_contribution + continuous_fast_mem
#   next_hidden = bypass_h + alpha * slot_delta + fast_mem_output
#
# The fusion_gate inside FastMem starts near 0, so at init:
#   next_hidden ≈ bypass_h + alpha * slot_delta  (same as before)
# As training progresses, the fast_mem learns to contribute useful information.
```

### 3.4 Chunk-Parallel Optimization (Production Version)

The sequential loop above is for clarity. In practice, use **chunk-wise parallel recurrence** (GLA/DeltaNet style):

```python
def forward_chunkwise(self, hidden_states, memory_state, chunk_size=64):
    """Chunk-parallel version: O(TC·d + T/C·d²) instead of O(T·d²).
    
    Within each mini-chunk of size C:
      - Compute intra-chunk attention in parallel (quadratic within chunk, but C is small)
      - Update state sequentially across chunks
    """
    B, T, d = hidden_states.shape
    C = chunk_size
    n_chunks = T // C
    
    # ... (use flash-linear-attention library for efficient implementation)
    # Key insight: the delta rule can be parallelized via the WY representation
    # See: github.com/sustcsonglin/flash-linear-attention
```

### 3.5 Dimensional Recommendations

For Llama-3-8B (d_model=4096, 32 layers):

| Config | H_fw | d_k=d_v | State Size/layer | Total (32L) | Notes |
|---|---|---|---|---|---|
| **Minimal** | 4 | 64 | 4×64×64×2B = 32KB | 1MB | Good for ablation |
| **Balanced** | 4 | 128 | 4×128×128×2B = 128KB | 4MB | Recommended starting point |
| **Large** | 8 | 128 | 8×128×128×2B = 256KB | 8MB | If capacity is insufficient |
| **XL** | 4 | 256 | 4×256×256×2B = 512KB | 16MB | Maximum reasonable |

Extra parameters per layer (projections):
- W_k, W_v, W_q: 3 × (4096 × H×d_k) = 3 × (4096 × 512) = 6.3M params
- W_gate: 4096 × 512 = 2.1M params
- W_beta: 4096 × H = 16K params
- W_o: 512 × 4096 = 2.1M params
- **Total per layer: ~10.5M params** → 32 layers = **336M extra params** (4% of 8B)

For the **Balanced** config with Llama-3-1B (d_model=2048):
- Per layer: ~2.6M params → 16 layers = **42M extra params** (4% of 1B)

---

## 4. Relationship to Prior Work

### 4.1 What's Novel About Our Combination

| Component | Prior Art | Our Innovation |
|---|---|---|
| Discrete slot memory (L1) | MemoryLLM, Slot Attention | Per-token routing (not per-segment), dual-gate writeback |
| L3 summary tokens | Q-Former, CLS tokens | Cross-chunk persistent summaries |
| **FastMem (proposed)** | Infini-attention, GLA, DeltaNet | **Gated delta rule as complement to discrete routing** |

**The key novelty**: No prior work combines discrete top-k slot routing (which provides precise, content-addressed recall for important tokens) with a continuous fast-weight memory (which provides a lossy but complete summary of ALL tokens). This is analogous to:
- **Computer architecture**: L1 cache (fast, small, selective) + DRAM (slow, large, complete)
- **Neuroscience**: Hippocampal episodic memory (discrete, content-addressed) + Neocortical semantic memory (distributed, continuous)
- **The "Two Complementary Learning Systems" theory** (McClelland et al. 1995)

### 4.2 Why This Design Is Better Than Alternatives

1. **vs. Pure Infini-attention**: We use the delta rule instead of pure accumulation → much higher associative capacity before interference degrades retrieval. Infini-attention's additive M += K^T·V has O(d) effective capacity; our delta rule has O(d²).

2. **vs. Pure slot memory (current system)**: 87.5% of tokens are completely lost. The fast weight captures a continuous summary of everything, acting as a "background context" that the LLM can query.

3. **vs. Larger slot bank**: More slots = more memory but doesn't solve the fundamental problem that routing must discard most tokens. Even with 1024 slots and top-k=128, 87.5% is still lost.

4. **vs. Full linear attention replacement**: We keep the pretrained softmax attention intact and add the fast weight as a parallel, gated module. This is critical for continued pretraining — the base LLM's capabilities are preserved.

5. **vs. Titans' surprise-based update**: Surprise-based gradient descent requires computing a loss per token, which is expensive. Our gated delta rule achieves similar selective memorization through the data-dependent gate + beta, but with O(d²) compute per token instead of O(d³) for gradient computation.

### 4.3 Expected Benefits

1. **Background context**: Even when a token doesn't match any slot's routing, its information is compressed into the fast weight. Later queries can retrieve a lossy version of it.

2. **Temporal smoothing**: The fast weight naturally provides a decaying summary of recent context, useful for tracking state (counting, entity tracking, dialogue state).

3. **Complementary failure modes**:
   - Slots fail when: relevant token wasn't routed (87.5% of tokens); slot was overwritten by a newer token
   - FastMem fails when: too many similar keys cause interference; old information decays
   - These failure modes are largely **orthogonal** → combining them should be strictly better

4. **Training signal**: The fast weight provides gradient signal for ALL tokens (not just the top-k routed ones), potentially helping the base model learn better representations overall.

---

## 5. Training Considerations

### 5.1 Initialization Strategy (for continued pretraining)

- **fusion_gate init = -2.0** → sigmoid(-2) ≈ 0.12 → fast_mem contributes ~12% at start
  - Alternative: init to -4.0 for even safer start (sigmoid(-4) ≈ 0.018)
- **W_o init = small** (std=0.01) → even with non-zero fusion_gate, output magnitude is tiny
- **Combined effect**: At step 0, fast_mem_output ≈ 0 → no PPL regression from adding the module

### 5.2 Learning Rate

- Fast weight projections (W_k, W_v, W_q, W_gate, W_beta, W_o): **2-5× base LR**
  - These are new parameters that need to learn from scratch
  - The base model's LR is typically low for continued pretraining (1e-5 to 5e-5)
  - FastMem projections should use 1e-4 to 5e-4
- fusion_gate: **0.5× base LR** — should ramp up slowly

### 5.3 Gradient Flow

```
loss → next_hidden → fast_mem_output → W_o → output (per-token)
                                      → S_t → delta update → W_k, W_v, W_beta
                                            → gate → W_gate
                   → slot_delta → (existing L1 path)
                   → bypass_h → (base LLM)
```

The fast weight module has a clean, direct gradient path from loss to all its parameters. No STE tricks needed (unlike the discrete slot routing).

### 5.4 Memory State Management

- **Intra-chunk**: State flows sequentially (token by token within chunk)
- **Inter-chunk**: State is **detached** at chunk boundary (same as our L1 bank reset pattern)
  - Why: prevents BPTT through unbounded time → gradient explosion
  - The state VALUE carries forward; the gradient graph does not
- **Sample boundary**: State is reset to zeros (new document = new context)

### 5.5 Curriculum

Phase 1 (steps 0-500): Warm up fusion_gate from -4.0 to -1.0 linearly  
Phase 2 (steps 500+): Let fusion_gate train freely via gradient  

This ensures the fast weight doesn't disrupt early training but ramps up contribution as it learns useful representations.

---

## 6. Implementation Plan

### Phase 1: Minimal Viable FastMem
1. Implement `FastMemModule` class with sequential loop (clarity first)
2. Integrate into `MemorySpaceLayer.forward` as additive term after slot_delta
3. Test on single-GPU with Llama-3-1B + Dolmino
4. Verify: PPL at step 0 matches baseline (safe init confirmed)
5. Train 2000 steps, check if fusion_gate increases and PPL improves

### Phase 2: Chunk-Parallel Optimization
1. Replace sequential loop with chunk-wise parallel algorithm
2. Use flash-linear-attention Triton kernels if available
3. Benchmark: should be <10% overhead vs. base forward (for Balanced config)

### Phase 3: Ablation Study
1. FastMem only (no L1 slots) vs. L1 slots only vs. Both
2. Delta rule vs. pure accumulation vs. gated (no delta)
3. Placement: parallel to attention vs. after attention
4. Dimensionality sweep: d_state ∈ {64, 128, 256}

### Phase 4: BABILong Evaluation
1. Train with curriculum (4k → 8k → 16k → 32k context)
2. Evaluate on BABILong qa1-qa10 at 0k-32k lengths
3. Compare with current L1+L3 results

---

## 7. Summary of Recommendations

| Design Choice | Recommendation | Rationale |
|---|---|---|
| **Update rule** | Gated Delta Rule | Best capacity; parallelizable; explicit forgetting |
| **Retrieval** | Linear query (o = S·q) | Minimal compute; no softmax needed |
| **Integration** | Additive with learned per-feature gate | Safe init; clean gradient; no attention modification |
| **Placement** | Parallel to attention (Titans MAG style) | Independent; doesn't change base LLM |
| **Scope** | Per-layer state | Each layer needs its own context summary |
| **Dimensions** | H=4, d_k=d_v=128 (Balanced) | 4% param overhead; good capacity |
| **Init** | Near-zero output at start | Critical for continued pretraining |
| **State management** | Detach at chunk boundary | Prevents gradient explosion |

---

## References

- Schlag et al. 2021: [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174)
- Yang et al. 2024: [Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/html/2406.06484v6)
- Yang et al. 2024: [Gated Linear Attention](https://arxiv.org/abs/2312.06635)
- GSA 2024: [Gated Slot Attention](https://arxiv.org/abs/2409.07146)
- Titans 2024: [Learning to Memorize at Test Time](https://community.datascience.hp.com/ideas/titans-learning-to-memorize-at-test-time-280)
- Infini-attention 2024: [Leave No Context Behind](https://arxiv.org/pdf/2404.07143.pdf)
- xLSTM 2024: [Extended Long Short-Term Memory](https://deep-paper.org/paper/2405.04517)
- RWKV-7 2025: [Goose](https://ghost.oxen.ai/how-rwkv-7-goose-works-notes-from-the-author/)
- TTT 2024: [Learning to Learn at Test Time](https://arxiv.org/abs/2407.04620)
- RetNet 2023: [Retentive Network](https://arxiv.org/pdf/2307.08621.pdf)
- FwPKM 2025: [Fast-weight Product Key Memory](https://arxiv.org/abs/2601.00671)
- Trellis 2024: [Learning to Compress KV Memory](https://arxiv.org/abs/2512.23852v1)
- Mamba-2 2024: [State Space Duality](https://arxiv.org/pdf/2405.21060)
- Block-Recurrent Transformer 2022: [arXiv:2203.07852](https://www.sciencestack.ai/paper/2203.07852v3)
- DeltaNet blog: [sustcsonglin.github.io](https://sustcsonglin.github.io/blog/2024/deltanet-1)
- Flash Linear Attention library: [github.com/sustcsonglin/flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)
- MeSH 2025: [Memory-as-State-Highways](https://arxiv.org/abs/2510.07739)
- MemoryLLM 2024: [Self-Updatable LLMs](https://scholar.googleusercontent.com/scholar?q=cache:qEfzqkAalOoJ:scholar.google.com/)
- Based 2024: [Simple Linear Attention Language Models](https://arxiv.org/abs/2402.18668)
