# v3 — RMT-Slot Hybrid: Top-K Slot Retrieval + RMT Sandwich Injection

## Architecture

```
Forward pass (per segment):
1. content_embeds = embed(input_ids)           # [B, S, D]
2. slots = bank.get()                          # [B, N=64, D]
3. pool_q = mean_pool(content_embeds)          # [B, D]
4. idx, scores, ste_weights = TopKSelector(pool_q, slots)  # idx: [B, K=8]
5. retrieved = gather(slots, idx) * ste_weights  # [B, K, D] with STE gradient
6. sandwich = [retrieved | content | placeholder]  # [B, 2K+S, D]
7. mask: causal + memory prefix rows see all positions
8. hidden = transformer_layers(sandwich, mask, continuous_pos_ids)
9. new_mem = hidden[:, -K:, :]                 # extract from placeholder positions
10. logits = lm_head(hidden[:, K:K+S, :])      # content positions only
11. bank.write(idx, new_mem, gate=sigmoid(gate_logit))  # EMA write-back to selected slots
```

## Initialization

| Parameter | Init | Rationale |
|-----------|------|-----------|
| `placeholder` | `N(0, embed_std)` | Match embedding scale for stable first forward |
| `gate_logit` | `logit(0.3) = -0.847` | Conservative EMA: 30% new, 70% old |
| `selector.slot_keys` | `N(0, 0.02)` | Small init for stable routing |
| `selector.Q_sel` | Xavier uniform | Standard linear init |
| `bank` slots | `hidden_pool` mode | Mean-pool first segment's embeddings + noise |

## Relationship to prior work

- **RMT (Bulatov et al., 2022)**: We adopt the sandwich injection format `[mem | content | mem]` but replace fixed read/write tokens with content-routed retrieval from a persistent bank.
- **MemoryLLM (Wang et al., 2024)**: Similar persistent memory concept but they use full-bank injection; we use sparse top-k retrieval (8 of 64) for efficiency.
- **Infini-attention (Munkhdalai et al., 2024)**: They compress into a single matrix via delta rule; we keep discrete slots with EMA write-back for interpretability.
- **Block Recurrent Transformer (Hutchins et al., 2022)**: Cross-attention memory read; we use prefix-injection which requires no new attention heads.

## Known issues

1. **Position aliasing**: Retrieved slots share position IDs 0..K-1 regardless of original content distance — may limit temporal reasoning.
2. **Mean-pool query**: Single vector query may not capture multi-topic segments well; future work could use multi-head queries.
3. **No slot eviction**: Bank size is fixed at N=64; very long documents may saturate capacity.
4. **BPTT cost**: Full backprop through all segments is O(n_segments) in memory; `bptt_depth` truncation trades accuracy for memory.
