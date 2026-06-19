# v22 — Raw-KV Readout (Method A): per-chunk raw-KV + emergent trainable gist-key soft attention

**Date.** 2026-06-19
**Author intent.** Implement `docs/RAWKV_READOUT_PROPOSAL.md §2 Method A` as a
self-contained, default-off code path that does NOT touch any existing readout
(byte-identical when `use_rawkv_readout=False`). This is the architectural
prototype for the only direction that differs *structurally* from every
exhausted negative in `docs/32K_WALL_FINDINGS.md`.

**Master switch.** `MemorySpaceConfig.use_rawkv_readout` (default `False`).
CLI: `--use_rawkv_readout`. New modules / state only materialise when on.

---

## 1. Architecture (forward pseudocode)

```
# Files: src/memory/mem_space/rawkv_readout.py  (store + trainable scorer)
#        src/memory/mem_space/inattn_kv.py       (reused in-attn concat + new bias)
#        src/memory/mem_space/layer.py           (write + read wiring)

# ---- WRITE (per chunk c, NO compression) ----
# Owned by ONE layer (smallest index of rawkv_readout_layers). hidden_states is
# the pre-LN layer input — exactly what build_retrieved_kv re-projects.
store.append_chunk(token_hidden = hidden_states.detach(),   # [B,T,d] raw tokens
                   token_pos    = 0..T-1)                    # source RoPE phase
#   internally also derives the per-chunk GIST SOURCE = mean-pool(chunk tokens)
#   → store.gist_src[:, c, :]  (detached; indexed, never stores content)

# ---- READ / INJECT (current chunk query, at each readout layer ℓ) ----
# Query = hidden_states (pre-LN layer input, GRAD-BEARING).
gkey   = key_proj(store.gist_src.detach())            # [B,C,gist]  TRAINABLE proj
gq     = query_proj(query_hidden)                     # [B,Tq,gist] TRAINABLE proj
score  = (gq · gkeyᵀ) * gist_dim**-0.5 / temperature  # [B,Tq,C]
kept   = top-k chunks by per-query-max score          # soft-top-k (set), or all
w      = softmax(score[:,:,kept], dim=chunks)         # DIFFERENTIABLE weights
col_bias = log(w)  gathered per retrieved token       # [B,Tq,R]  grad-bearing

K_raw, V_raw = build_retrieved_kv(self_attn,          # native k/v_proj + RoPE
                                  retrieved_hidden=kept chunks' raw tokens,
                                  retrieved_pos, position_embeddings, pre_norm)
self_attn._inattn_kv = (K_raw, V_raw, col_bias)       # 3-tuple stash

# wrapped self-attn (inattn_kv.make_inattn_attention_forward):
attn over [native_KV ; K_raw]                          # ONE softmax (landmark §4b)
#   the retrieved columns get an additive bias = col_bias (= log gist weight),
#   so cross-token weight = native_softmax_logit  +  log(gist_weight)
#   i.e. effective weight ∝ token-score × gist-score  (Landmark's score×landmark)
```

**Gradient path (the whole point).** `loss → native attention softmax →
col_bias → query_proj / key_proj`. The gist scorer is IN the loss graph and
gets gradient every time the read fires. The retrieved *content* is detached
(raw KV is data, not learned), but `K_raw/V_raw` carry gradient into the
reader's native `k_proj/v_proj/o_proj` when the reader is unfrozen — so the
reader is **trained on the raw-KV-concat path** (Method A's novelty over the
frozen-reader in-attn oracle negative, `76efbd4`, 21.0 ≈ OFF 22.0).

---

## 2. Initialization

- `GistReadout.query_proj`, `key_proj`: `nn.Linear(d_model, gist_dim, bias=False)`,
  `normal_(std=0.02)`. Small-random → the selection is non-uniform and the
  scorer is in the loss graph from step 0, but not large enough to swamp the
  native softmax before it learns. Scale = `gist_dim ** -0.5`.
- `rawkv_gist_dim=128`, `rawkv_readout_topk_chunks=8`, `rawkv_readout_temp=1.0`,
  `rawkv_readout_layer=16` / `rawkv_readout_layers=None`.
- The gist scorer is a **shared singleton** registered on the model root
  (`root.gist_readout`, peer to `l3_pool`) → params appear once in state_dict +
  are collected by `_mem_space_params` → enter the optimizer. The per-sequence
  raw-KV store is pure runtime state on the shared `MemoryBank`
  (`_rawkv_readout_store`), reset at the document/rollout boundary by
  `MemoryBank.reset()`.

---

## 3. Relationship to prior work

- **vs the eval-time `use_inattn_kv` probe (v17-era / `inattn_kv.py`):** that
  proved the in-attention raw-KV concat mechanism is in-graph + multi-layer, but
  its RETRIEVAL is `no_grad` hard-top-k keyed by the **TopKSelector** routing-q —
  the gist scorer would never get gradient (the dead-retriever trap). Method A
  **deletes the TopKSelector from this path** and replaces it with a trainable,
  differentiable gist-key soft attention. The `inattn_kv` wrapper is reused
  wholesale; the only new mechanism is the differentiable retrieval + the
  per-column gist log-bias (3-tuple stash, backward-compatible with the 2-tuple).
- **vs Landmark Attention:** this IS the Landmark mechanism — raw past-block KV
  addressed by a trained gist/landmark score, reader trained on the
  grouped/in-attention path. The gist weight as an additive log-bias on the
  retrieved columns mirrors Landmark §4b (cross-block weight = token-score ×
  landmark-score), expressed in additive log space inside the one softmax.
- **vs mem_space slots:** removes BOTH bottlenecks named in
  `32K_WALL_FINDINGS.md §1.3` — (1) lossy compression (raw KV is byte-
  recoverable), (2) dead-slot pollution (soft attention over per-chunk gist keys,
  no fixed slot bank, no broken selector).

---

## 4. Known issues / what is NOT yet done (architecture validation only)

1. **Reader unfreeze is mandatory and NOT auto-enforced.** Per proposal §4, a
   frozen reader reproduces the in-attn oracle negative. Method A must be run
   with `--unfreeze_backbone --unfreeze_layers_from 16` and the readout layers
   inside the unfrozen range. The code is compatible (the readout layers' native
   k/v/o_proj receive gradient when unfrozen, verified in the smoke), but nothing
   *forces* the unfreeze — the launcher must set it.
2. **B>1 shared kept-set simplification.** Soft-top-k chooses ONE kept chunk set
   for the batch (via batch-mean salience) assuming a shared per-sequence store
   layout + uniform chunk length. Correct for B=1 and for uniform-length chunked
   training; for ragged B>1 it is an MVP approximation. Per-batch independent
   kept sets (ragged R) are future work.
3. **Memory cost of raw KV is O(total tokens).** The store grows by chunk_len
   each chunk; `rawkv_readout_topk_chunks` bounds R at read but the *store* is
   full. At 32k this is the same O(n) KV-cache budget the proposal flags
   (Method B §displacement) — fine at a few injection layers, heavy at all 32.
4. **Cross-chunk read fires (verified), but only when the store is non-empty.**
   The FIRST chunk has an empty store → no read (correct). In the
   `dolmino_train_step` regime the context chunks stream under `no_grad` then
   the bank is detached; the store content is detached anyway, so the gist
   scorer's gradient comes from the *target* chunk's read over the
   already-written store (the target query is grad-bearing, the projections are
   trainable). The smoke runs the multi-chunk forward WITH grad and confirms the
   scorer + reader both receive non-zero gradient through the cross-chunk read.
   ⚠️ Under `--gradient_checkpointing` + the streamed-context `no_grad` regime,
   confirm on the real model that the target-chunk read still back-props to the
   gist scorer (the proposal's named max risk); the CPU smoke validates the
   mechanism but not the full FSDP/grad-ckpt training stack.

---

## 5. Verification (CPU, tiny random Llama — `tests/test_rawkv_readout_smoke.py`)

- OFF → byte-identical to a no-readout build; no gist scorer created.
- ON, multi-chunk fwd+bwd: loss finite; `query_proj`/`key_proj` grads non-zero
  (scorer in graph); reader `o_proj` grad non-zero (reader trained on path);
  read path fires on both readout layers (R=64); store grows across chunks
  (cross-chunk read fires).
