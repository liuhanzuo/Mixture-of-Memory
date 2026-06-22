# v23 — Per-slot raw-KV cache (unlimited upper-bound)

## Architecture

Goal: test whether the existing slot selector can point to useful long-range locations when the selected slots carry lossless raw KV beneath them.

Forward pass, for the single owner layer `slot_kv_cache_layer`:

```python
# Per chunk, before wrapped decoder forward
slots = memory_bank.get()
idx, scores, _ = selector(hidden_states, slots)      # [B, top_k]

# READ path: use the current selector result, not a new scorer
cached_h, cached_pos = memory_bank.retrieve_slot_kv_cache(idx)
if cached_h is not None:
    K_raw, V_raw = build_retrieved_kv(
        self_attn,
        cached_h,             # all raw hidden cached under selected slots
        cached_pos,           # source within-chunk RoPE offsets
        position_embeddings,
        pre_norm=input_layernorm,
    )
    self_attn._inattn_kv = (K_raw, V_raw)

bypass_h = wrapped_layer(hidden_states, native_plus_cached_kv)
ext_h = wrapped_layer([L3, L2, L1, evidence, hidden_states], native_plus_cached_kv)
next_hidden = bypass_h + gate * (ext_h_body - bypass_h) + memory_xattn_out

# Normal mem_space writeback updates selected compressed slots
O_mem_slot = hidden_to_slot(O_mem_hidden)
memory_bank.write(idx, O_mem_slot, ...)

# WRITE path: after the read/write, bind this chunk's raw hidden to selected slots
memory_bank.append_slot_kv_cache(
    slot_idx=idx,
    token_hidden=hidden_states.detach(),
    token_pos=torch.arange(T),
    token_mask=active_token_mask,
)
```

The cache is flat but semantically per-slot:

```python
slot_kv_hidden: [B, M, d_model]
slot_kv_slot:   [B, M]
slot_kv_pos:    [B, M]
```

For every chunk, every selected slot receives every real token in that chunk, so `M += top_k * real_tokens`. Retrieval returns all entries whose `slot_kv_slot` is in the current selected slot set.

## Initialization

No trainable parameters are added.

Runtime state starts empty on `MemoryBank.reset()`:

```python
slot_kv_hidden = None
slot_kv_slot = None
slot_kv_pos = None
```

The first routed chunk materializes the flat cache. Cached hidden states are detached bf16/fp tensors matching the layer input dtype; slot ids and source positions are `long` tensors.

## Relationship to prior work / Method A

This is deliberately not `rawkv_readout.py` Method A.

- Method A stores raw KV per chunk and retrieves chunks with an independent trainable gist scorer.
- v23 stores raw KV under the slot ids selected by the existing mem_space router.
- Method A tests whether a new chunk scorer can learn retrieval.
- v23 tests whether the existing slot router is already sufficient if selected slots expose lossless raw content.
- Both reuse the same native in-attention KV concat mechanism (`build_retrieved_kv` + `_inattn_kv`) so the reader sees raw K/V in the self-attention softmax rather than as OOD prefix tokens.

This mirrors the SWA insight: raw KV improves readout, but v23 sources raw KV from slot retrieval across the whole stream instead of the last W chunks.

## Known issues

- No capacity limit by design. Memory grows as `chunks * top_k * chunk_size`, so nctx63 can be very large and may OOM; this is the intended upper-bound probe.
- Batch rows with different valid-token counts or retrieved sizes are skipped rather than padded, because fake padded KV columns would perturb attention. The prepared launch uses batch size 1.
- Slot recycle / eviction does not clear old per-slot raw KV. Planned nctx63 script does not enable recycle; if future runs combine them, stale cache invalidation should be added.
- If combined with Method A or `use_inattn_kv` on the same layer, the last writer to `_inattn_kv` wins. v23 should be run as the only raw-KV injection channel.
- Because raw hidden is detached, gradients train the reader/injection path and mem_space parameters, not the cached context chunks themselves.
