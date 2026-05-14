# Version 3: Infini-Attention Memory (2026-05-01)

## Architecture

Replace cross-attention (v2) with Infini-attention (Munkhdalai et al., 2024, arXiv:2404.07143).
Uses **linear attention** (no softmax) over a compressive associative matrix M,
completely avoiding the routing degeneracy and uniform-attention problems of v1/v2.

### Forward pass (per layer)

```
# 1. Run decoder layer normally (UNCHANGED — no modification to internal attention)
decoder_out = wrapped_decoder_layer(hidden_states)  # standard LlamaDecoderLayer

# 2. Memory retrieval (linear attention)
Q = q_proj(hidden_states)                  # [B, T, d_model], frozen backbone weights
Q_heads = reshape(Q, [B, n_heads, T, d_head])
sigma_Q = ELU(Q_heads) + 1                # linear attention kernel

A_mem = sigma_Q @ M / (sigma_Q @ z + eps)  # [B, n_heads, T, d_head]
A_mem_flat = reshape(A_mem, [B, T, d_model])
mem_output = o_proj(A_mem_flat)            # frozen backbone weight

# 3. Gated residual (per-head gate)
gate = sigmoid(beta)                       # [n_heads], init=-5 → sigmoid≈0.007
augmented = decoder_out + (gate * A_mem projected through o_proj)

# 4. Memory update (delta rule — self-correcting)
K = k_proj(hidden_states)                  # frozen backbone weights
V = v_proj(hidden_states)                  # frozen backbone weights
sigma_K = ELU(K_heads) + 1

existing_V = sigma_K @ M / (sigma_K @ z + eps)
delta_V = V_heads - existing_V             # residual: only new information

M += sigma_K^T @ delta_V                   # outer product accumulation
z += sigma_K.sum(dim=time)                 # normalizer update
```

### Key differences from v2 (cross-attention)

| Aspect | V2 (cross-attention) | V3 (Infini-attention) |
|--------|---------------------|----------------------|
| Attention type | Softmax cross-attn | Linear attn (ELU+1 kernel) |
| New attention params | Yes (Q/K/V/out_proj) | None (reuse pretrained) |
| Memory representation | 128 slot vectors | d×d associative matrix per head |
| Routing/selectivity | Top-k or softmax uniform | No routing (linear attn is global) |
| Initialization problem | Bootstrap paradox (symmetry) | M=0, z=0 → zero output at start |
| Write mechanism | EMA writeback | Delta rule (self-correcting) |
| Trainable params | ~4M (cross-attn weights) | ~1K (beta scalars only) |
| PPL at step 0 | 1.5-1M (depending on gate) | Vanilla (beta gate ≈ 0) |

### Initialization

- `beta`: nn.Parameter(full([n_heads], -5.0)) → sigmoid(-5) ≈ 0.007
  - Near-zero memory contribution at start → model starts as vanilla Llama
  - Gradient to beta: d(loss)/d(beta) = (A_mem - A_local) × sigmoid'(beta)
  - Non-zero from step 1 because A_local = 0 (M=0 → A_mem=0)
  - Wait, that means d(loss)/d(beta) = (0 - A_local) × sigmoid'(beta) = -A_local × sigmoid'(beta)
  - This pushes beta toward more memory usage if A_local helps reduce loss
  - Actually, since A_mem = 0 initially, the gate output is 0, so augmented = decoder_out
  - The gradient to beta is: d(loss)/d(gate * A_mem) × A_mem = 0 (because A_mem = 0)
  - Hmm, this means beta gets ZERO gradient initially... but that's OK!
  - beta doesn't NEED to change initially. M and z get non-zero updates from K, V.
  - After one chunk: M ≠ 0, z ≠ 0 → next chunk has non-zero A_mem → beta gets gradient.
- `M`: torch.zeros(n_heads, d_head, d_head) → zero memory at start
- `z`: torch.zeros(n_heads, d_head) → zero normalizer at start
- Q/K/V/o projections: frozen pretrained Llama-3-8B weights

### Memory management

- M, z are per-sample state (like MemoryBank slots)
- Reset at document boundaries (same as _reset_banks)
- Detach across chunk boundaries (same as _detach_banks)
- Memory footprint: n_heads × d_head × d_head × 2 bytes × 32 layers = 32 × 32 × 128 × 128 × 2 = 33 MB

## Relationship to prior work

### Infini-attention (Munkhdalai et al., 2024, arXiv:2404.07143)

Our v3 is a direct application of Infini-attention, adapted for our wrapper-based architecture.
Key differences:
- Infini-attention replaces the internal attention computation
- Our wrapper adds memory retrieval as a POST-attention residual
- This avoids modifying HF's attention implementation (SDPA/eager/FlashAttention)

### Why v2 failed but v3 should work (per Opus researcher analysis)

1. **Symmetric cold-start**: All 128 slots identical → softmax uniform → no gradient
   - v3: No softmax. Linear attention doesn't normalize, so no uniformity trap.
2. **Bootstrap paradox**: Zero-init out_proj → zero gradient to attention weights
   - v3: Reuses pretrained QKV. Gradient to M flows from loss through frozen projections.
3. **Writeback pollution**: EMA writeback from corrupted hidden states → positive feedback
   - v3: Delta rule is self-correcting (stores only residual).

## Known unknowns

1. **RoPE interference**: We use Q/K WITHOUT RoPE for memory. K with RoPE is position-dependent,
   which might cause issues when querying across chunks. Need to test both with/without RoPE.
2. **Linear attention quality**: ELU+1 kernel is known to be less expressive than softmax.
   May need to use the "delta rule" variant for better performance.
3. **Batch dimension**: M, z need to be per-sample. Need careful handling with DDP.
4. **Wrapper vs internal**: Adding memory as a post-attention residual may be less effective
   than modifying the attention computation directly. If wrapper approach fails, try internal.
