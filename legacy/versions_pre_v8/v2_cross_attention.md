# Version 2: Cross-Attention Memory (2026-04-30)

## Architecture

Cross-attention replaces the top-k slot routing that suffered from routing degeneracy
(chunk-level cosine similarity always collapsed to uniform 1/N).

### Forward pass (per layer)

```
Read phase:
    slots = memory_bank.get()                          # [B, N=128, d]
    attn_out, attn_weights = CrossAttn(Q=hidden, K=slots, V=slots)  # [B, T, d]
    gated = hidden + tanh(output_gate) * LayerNorm(attn_out)
    decoder_out = wrapped_decoder_layer(gated)         # standard LlamaDecoderLayer

Write phase:
    new_slot_content = CrossAttn(Q=slots, K=decoder_out, V=decoder_out)  # [B, N, d]
    slots = (1 - beta) * slots + beta * LayerNorm(new_slot_content)
    memory_bank.write(slots)
```

### Key differences from top-k routing (Version 1)

| Aspect | V1 (top-k routing) | V2 (cross-attention) |
|--------|-------------------|---------------------|
| Slot selection | Top-k cosine similarity | All slots via attention weights |
| Sequence | Extended [slots, tokens] | Tokens only (slots via residual) |
| Gradient flow | STE estimator (brittle) | Direct (through attention weights) |
| Dual forward | Yes (bypass + extended) | No (single forward) |
| Attention mask | Complex extended causal | Standard causal |
| Aux losses | load_balance, key_repulsion, peak_routing | None needed |

### Initialization

- `output_gate`: nn.Parameter(0.01) → tanh(0.01) ≈ 0.01 (small but non-zero)
  - Too large (0.5): random cross-attn output destroys LM → PPL explodes to 1M
  - Too small (0.0): zero gradient → dead path
- `write_gate`: nn.Parameter(0.0) → sigmoid(0) = 0.5 (moderate writeback)
- Cross-attention weights: PyTorch default (Xavier)
- Slot params: randn * 0.02 (small random)

### Known issues (2026-05-01)

All 5 initialization attempts (v1-v5) resulted in PPL > 100 ("model polluted" per CLAUDE.md rules).

| Attempt | Gate | LayerNorm | Writeback | out_proj zero-init | Result |
|---------|------|-----------|-----------|-------------------|--------|
| v1: gate=0 | tanh(0)=0 | Yes | Immediate | No | PPL oscillates 2-14, gate dead |
| v2: gate=0.5 | tanh(0.5)=0.46 | Yes | Immediate | No | PPL explodes to 1M |
| v3: gate=0.01 | tanh(0.01)=0.01 | Yes | Immediate | No | PPL ~1400 stable |
| v4: gate=1.0 | tanh(1.0)=0.76 | Yes | Delayed 500 | Yes (read+write) | PPL ~1500, slots decay |
| v5: no gate | None | No | Delayed 500 | Yes (read+write) | PPL ~1200-2000 |

Root causes identified:
1. **32-layer compounding**: cross-attn noise at every layer compounds 32×
2. **LayerNorm amplification**: normalizes small out_proj output to O(1) scale
3. **Writeback slot pollution**: once writeback activates, slot norms explode
4. **Attention remains uniform**: entropy=4.84/4.85 throughout — cross-attn never learns peaked patterns

**Conclusion**: Direct residual addition of cross-attention output to hidden_states at every layer is too disruptive for pretrained LLM. Need architectural redesign:
- Option A: Only insert at every 4th layer (8/32 layers)
- Option B: Use adapter-style bottleneck (down-project → cross-attn → up-project)
- Option C: Replace self-attention heads (not add to them)

## Relationship to prior work

### NOT MemoryLLM (Wang et al., 2024, arXiv:2402.04624)

MemoryLLM uses KV-prepend: memory vectors are concatenated to the input KV in self-attention,
sharing the same Q/K/V projection weights. Our approach uses **separate cross-attention modules**
with dedicated weights. This is a fundamental architectural difference:
- MemoryLLM: `Attn(Q=X W_Q, K=[M; X] W_K, V=[M; X] W_V)` — fused, shared weights
- Ours: `CrossAttn(Q=X, K=M, V=M) + SelfAttn(X)` — factored, separate weights

### IS Block Recurrent Transformer (Hutchins et al., 2022, arXiv:2203.07852)

Our read/write cross-attention pattern is architecturally identical to Block Recurrent Transformer:
- Read: `CrossAttn(Q=input, K=state, V=state)` → identical
- Write: `CrossAttn(Q=state, K=input, V=input)` → identical

Key difference: Block Recurrent was trained from scratch on PG-19. We retrofit onto pretrained Llama-3-8B.

### Related but different

- **Infini-attention** (Munkhdalai et al., 2024): Uses linear attention for write (cheaper, less expressive)
- **Perceiver IO** (Jaegle et al., 2022): Same cross-attn pattern but for multimodal, not recurrence
- **Compressive Transformer** (Rae et al., 2020): Compresses old activations, no cross-attention
