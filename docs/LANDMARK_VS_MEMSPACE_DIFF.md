# Landmark Attention vs. our mem_space — authoritative diff map

**Purpose.** Phase-1 anchor reproduction of Landmark passkey is confirmed GO. This
document is the authoritative map for **Phase-3 diff-based migration**: starting
from the *known-working* Landmark setup and changing **exactly one dimension at a
time** toward our `mem_space`, so that the migration step where passkey accuracy
collapses pinpoints which difference kills long-range recall.

**Sources.**
- Paper: Mohtashami & Jaggi, *Landmark Attention*, arXiv:2305.16300 (NeurIPS 2023).
- Official repo: `epfml/landmark-attention` — `llama/train.py`, `llama/run_test.py`,
  `llama/llama_mem.py`, `llama/requirements.txt`.
- Our side: `src/memory/mem_space/{memory_bank.py,selector.py,inattn_kv.py,layer.py,l2_compressor.py,l3_summary.py}`,
  `scripts/train_mem_space_dolmino_cpt.py`, `scripts/launch_sft_unfreeze_inattn.sh`.

**Gating eval for every migration step.**
- **Primary anchor = passkey retrieval** (`llama/run_test.py`): garbage filler
  `n ∈ {0,100,500,1000,5000,8000,10000,12000,14000,18000,20000,25000,38000}` chars
  (≈ up to 32k+ tok), `top_k=5`, `num_tests=50`/length, passkey `randint(1,50000)`,
  fixed question `"What is the pass key? The pass key is"`, `max_new_tokens=10`.
- **Cross-check = our BABILong qa1** (single-fact NIAH, same semantics as passkey),
  lengths 0k–32k, n=100, `babilong.metrics`. qa2/qa5 (multi supporting-fact) are
  advanced observation only — Landmark has no multi-fact baseline.
- **PPL is NOT a migration gate**: the Landmark *fine-tune* line (§4.2) reports
  only passkey; PG19/arXiv PPL (§4.1, Table 1) is the *from-scratch GPT-2* line and
  is off-scope.

---

## Summary table

| # | Dimension | Landmark (working) | Our mem_space | Migration cost | Expected long-range impact | Migration order |
|---|-----------|--------------------|---------------|----------------|----------------------------|-----------------|
| 1 | **Unfreeze scope** | full-FT entire model | v1 full / v2 partial L16–31 | Low | High | **S-anchor / kept fixed** |
| 2 | **Base model** | LLaMA-2-7B | Llama-3-8B | Low (swap ckpt) | Low | **S1** |
| 3 | **Data volume / source** | RedPajama-1T-Sample, ~0.98B tok, 7 sources, 1 epoch | dolmino per_doc, ~32.8M fwd / 8.2M loss tok, single source, 1k steps | Medium | High | **S2** |
| 4 | **ctx / block structure** | ctx 512, block 50, `<landmark>` every 50 tok | chunk 512, n_ctx=3 (eff ctx 2048), 128 slots | Medium | Medium | **S3** |
| 5 | **Retrieval** | train: soft, all blocks visible → infer: grouped-softmax top-5, **real raw KV blocks** | TopKSelector routing-q on pooled hidden vs slot keys (measured **0% needle precision**) | Medium | **High** | **S4** |
| 6 | **Readout / injection** | grouped-softmax in **every layer** (intra-block denom isolation + landmark gating) | in-attn raw-KV concat at **single layer 16** (or prefix) | High | **High** | **S5** |
| 7 | **Memory unit** | in-context `<landmark>` token (one per block, ephemeral, trainable summary via embedding) | 128 fixed learned slots + adapter compression (MemoryBank) | High | **High** | **S6** |

---

## Dimension 1 — Unfreeze scope

- **Landmark.** Full fine-tune, no freezing, no LoRA. `llama/train.py` uses
  `--fsdp "full_shard auto_wrap"` over all `LlamaDecoderLayer`s; every param
  (embed + attn + FFN + the new `<landmark>` token embedding + lm_head) is trained.
  Only a full weight-diff is released (`weight_diff.py`) — they never tried
  attention-only or single-layer unfreezing.
- **Ours.** `scripts/launch_sft_unfreeze_inattn.sh` v1 = `--unfreeze_backbone`
  (full 8B) + memory adapter, lr 2e-5. v2 (planned) = partial unfreeze L16–31.
- **Cost.** Low — a CLI flag (`--unfreeze_backbone` / a layer-range filter).
- **Long-range impact.** High in principle: Landmark needs the *whole* model to
  recalibrate to the attended distribution. Our v1 evidence: full unfreeze for only
  1000 steps on short dolmino *damaged* base NIAH (OFF 22→12), i.e. too-few-steps
  full FT hurts before it helps (damage-investigator).
- **Migration plan.** Keep **full unfreeze fixed during S0–S5** (it is part of the
  working anchor), so it is *not* a variable. Revisit partial unfreeze only after
  the rest of the diff is understood.

## Dimension 2 — Base model

- **Landmark.** LLaMA-2-7B (released tuned weight-diff `epfml/landmark-attention-llama7b-wdiff`).
- **Ours.** `models/Meta-Llama-3-8B`.
- **Cost.** Low — swap checkpoint + tokenizer; re-train (or short-train) with the
  same `train.py`. Llama-3 has a different tokenizer/vocab + GQA; the `<landmark>`
  special-token resize logic (`smart_tokenizer_and_embedding_resize`) carries over.
- **Long-range impact.** Low — the mechanism is base-agnostic; this only rules out
  "Llama-3 itself breaks it".
- **Migration plan.** **S1**: re-run Landmark mechanism on Llama-3-8B (owned by
  landmark-repro in parallel). Gate: passkey must still hit ~32k≈high. Done first
  because it is low-risk and removes base-model as a confound for all later steps.

## Dimension 3 — Data volume / source

- **Landmark.** `load_dataset("togethercomputer/RedPajama-Data-1T-Sample")`
  (`train.py`) — ~0.98B tokens (15000 steps × eff-batch 128 × 512), exactly 1
  epoch, **7 sources** (CommonCrawl, C4, GitHub, Books, ArXiv, Wikipedia,
  StackExchange). LM loss over **all** tokens (`DataCollatorForLanguageModeling`).
- **Ours.** dolmino per_doc single source; v1 = ~32.8M forward / 8.2M loss tokens
  (`last_chunk_loss_only`), 1000 steps. → **~30× fewer forward / ~120× fewer loss
  tokens**, single source vs 7.
- **Cost.** Medium — point the loader at RedPajama-Sample (already wired in
  Landmark's `train.py`); for our trainer, add a RedPajama path + drop
  `last_chunk_loss_only` to match full-token LM loss.
- **Long-range impact.** High — short single-source dolmino (n_ctx=3 ⇒ only 2048
  eff ctx) never exposes true long-range dependencies; our ~30 experiments
  repeatedly show "real long data > training-side knobs".
- **Migration plan.** **S2**: from the S1 Llama-3 anchor, swap RedPajama → dolmino
  (keep ctx512/block fixed). Gate passkey: if it drops here, single-source/short
  data is a primary killer (confirms the v1 suspicion).

## Dimension 4 — Retrieval

- **Landmark.** During training every block is softly visible; the grouped-softmax
  *trains* the model to gate blocks via landmark tokens. At inference (`run_test.py`,
  `cache_top_k=5`) it retrieves the **actual raw K/V** of the top-5 blocks selected
  by landmark scores — i.e. retrieval returns true source tokens, and the selection
  signal is the same attention the model trained on.
- **Ours.** `TopKSelector` (`selector.py`): scores N slots against a *pooled*
  hidden summary via small `Q_sel`/`K_sel` projections (+ learnable per-slot keys),
  straight-through hard top-k, MoE load-balance aux. Measured **0% needle
  precision** — routing-q on a pooled summary does not localize the needle.
- **Cost.** Medium — retrieval is a self-contained module; can swap the scoring
  source (pooled-hidden routing-q → landmark/block-score top-k over real KV).
- **Long-range impact.** High — if the retriever never returns the needle block,
  no downstream readout can recover it.
- **Migration plan.** **S4** (after data/ctx settled): replace grouped-softmax soft
  selection with our top-k selector routing. Gate: a passkey collapse here isolates
  **selector 0% precision** as the culprit, independent of the readout mechanism.

## Dimension 5 — Readout / injection

- **Landmark.** Grouped-softmax operates in **every attention layer**: a token's
  own block tokens *and* every other block's landmark token share one softmax
  group, forcing a local-vs-retrieved trade-off; cross-block weight = token-score ×
  landmark-score (gated). One normalized softmax per layer; retrieved KV carry real
  source RoPE positions. The reader is *trained on this exact path*.
- **Ours.** `inattn_kv.py`: retrieved raw K/V are projected through the layer's
  native `k_proj`/`v_proj` and concatenated onto native K/V at a **single layer
  (16)** in **one** softmax `[native_KV ; retrieved_KV]` with real source RoPE
  (the architecturally-correct variant); prior probes injected as a prefix block
  and failed. Still single-layer, eval-time wrapper.
- **Cost.** High — moving from single-layer concat to all-layer grouped-softmax is
  a mechanism rewrite touching `layer.py` + the attention wrapper across the stack.
- **Long-range impact.** High — single-layer injection gives the model one shot to
  consume retrieved KV; Landmark distributes the readout across all layers and
  *trains* it. This is the most mechanism-essential difference besides the memory
  unit.
- **Migration plan.** **S5** (high risk, late): single-layer concat → all-layer
  grouped-softmax-style gating. Gate passkey + qa1.

## Dimension 6 — Memory unit (most essential difference)

- **Landmark.** No fixed memory bank. A `<landmark>` token is inserted every block
  (`add_mem_tokens`, `mem_freq` default 63 in repo / 50 in paper); its representation
  is computed *in context* by the model itself and is **ephemeral** (recomputed each
  forward). It is "trainable" only through the shared token embedding + the
  end-to-end LM objective — there is no separate compression module. Memory =
  the KV cache of past blocks, addressable via landmark scores.
- **Ours.** `MemoryBank` (`memory_bank.py`): 128 **fixed learned slots**
  `[B, N, slot_dim]`, written by an adapter that *compresses* chunk hidden states
  into slots (write/read, dual-gate, L2 compressor, L3 summary). Memory is a
  learned lossy compression, not raw past KV.
- **Cost.** High — this is the architectural identity of mem_space; replacing
  in-context landmark tokens with the slot bank + adapter is the final, largest
  jump.
- **Long-range impact.** High — fixed-size lossy slots may simply not preserve a
  specific needle the way Landmark's raw-block KV does. This is the hypothesis the
  whole migration is built to test.
- **Migration plan.** **S6** (final): in-context landmark tokens → 128 fixed slots
  + adapter. Reaching full mem_space. Gate passkey + qa1; a collapse here vs S5
  isolates "lossy slot compression" as the long-range bottleneck.

---

## Phase-3 migration order (one diff at a time)

| Step | Change from → to | Gate | Expected reading if it collapses |
|------|------------------|------|----------------------------------|
| **S0** | reproduce Landmark anchor (LLaMA-2-7B, wdiff or 15k train) | passkey 32k≈98% | infra/eval口径 broken — fix before proceeding |
| **S1** | base LLaMA-2-7B → Llama-3-8B | passkey | Llama-3 base incompatible with mechanism |
| **S2** | data RedPajama → dolmino (single source/short) | passkey | single-source/short data is a killer |
| **S3** | ctx/block: landmark-every-50 → chunk512/n_ctx3 | passkey + qa1 | our chunking structure hurts |
| **S4** | retrieval: grouped-softmax soft → top-k selector routing | passkey + qa1 | **selector 0% precision** is the culprit |
| **S5** | readout: all-layer grouped-softmax → single-layer in-attn concat | passkey + qa1 | single-layer / concat readout insufficient |
| **S6** | memory unit: in-context landmark token → 128 fixed slots + adapter | passkey + qa1 | **lossy slot compression** loses the needle |

**Most-suspect ordering (prior belief):** S4 (retrieval precision) > S5 (single- vs
all-layer readout) ≈ S6 (lossy slots) > S2 (data). Unfreeze (full) is held fixed as
part of the anchor and is not a migration variable.

**Recommended first coder actions:** S0 + S1 (landmark-repro is already scoping S1
feasibility on Llama-3) to establish the "our infra reproduces Landmark's wall-break"
anchor, then migrate S2→S6 with passkey + qa1 gating at every step.
