# Pyramid v1 — MemoryLLM (mid-layer) + QCMem (raw-hidden base) merge

Date: 2026-07-07
Backbone: **Llama-3-8B** (L=32, d=4096, 32 heads / 8 KV, vocab 128256, rope_theta 5e5).
Status:阶段-1 validated (MemoryLLM port smoke PASS) + 阶段-2 design/skeleton.

This document specifies how to merge two of the three pyramid memory layers:

* **Base layer = raw hidden (QCMem).** No compression. `write_chunk` runs the
  bottom `j` layers over one chunk chunk-local, caches the depth-`j` hidden
  `h_j`; `read` packs `[sink; selected chunk h_j; query h_j]` and resumes
  `layers[j:]`. Already the project SOTA on BABILong (qa1/16k=57 vs MemoryLLM 9).
  Precise, recent context. Impl: `src/memory/qcmem/qcmem_model.py`.
* **Mid layer = MemoryLLM pool.** Medium compression. A fixed per-layer memory
  pool `memory[L, num_blocks*num_tokens, d] = [32, 12800, 4096]` (1.68B params,
  trained). `inject_memory(ctx)` pushes new context into the pool (FIFO drop +
  cross-attention write); at read the pool is a per-layer KV prefix. Distant,
  compressed context. Impl (ported): `src/memory/memoryllm_ported/`.
* Top layer = slot bank (highest compression) — **out of scope for v1**.

阶段-1 result — MemoryLLM port smoke (`scripts/smoke_memoryllm_port.py`, GPU
cuda:0, bf16, L20A): load OK; memory pool `[32, 12800, 4096]` finite (1.6777 B);
`inject_memory(45-tok ctx)` -> `delta_memory [1, 32, 256, 4096]` finite; forward
after inject gives finite non-degenerate logits, argmax next-token after "What
fruits does David like?" = `'app'` (p=0.245) and a manual greedy decode yields
`'applesbananas...'` — **the injected fact (apples + bananas) is recalled**, so
the memory read path works end-to-end on the local .venv.

---

## The merge insight (why these two compose)

MemoryLLM's `memory[idx]` is **not** an opaque table — in `cat_memory_and_hiddens`
it is *concatenated to `hidden_states` at the input of decoder layer `idx`* and
then processed by that layer. So `memory[idx]` lives in **layer-`idx`'s residual
stream** — it is a depth-`idx` hidden representation, the **exact same currency**
as QCMem's cached `h_j` (also a depth-`j` residual-stream hidden). Two consequences:

1. At a shared split depth `a` (`= resume_j`), a slice of the MemoryLLM pool
   `memory[a]` (compressed far context) and QCMem's `h_a` chunk caches (raw near
   context) are **mutually concatenable** into one packed sequence and resumed
   through `layers[a:]` together. No projection / adapter needed to bridge them.
2. Their *injection cadence* differs and this is the one real design choice:
   * QCMem near chunks enter **once** at layer `a` as ordinary causal tokens and
     flow up through every `layers[a:]` (they persist in the residual stream).
   * MemoryLLM far memory is re-prefixed at **every** layer `idx` and sliced off
     after that layer (`prefix_token_length`); it never persists across layers.

v1 keeps both cadences intact (Variant P2 below) so each sub-mechanism runs the
way it was trained/validated, and the two only meet through attention.

---

## Architecture (forward pass)

Notation: `a = resume_j` (split depth), `L = 32`. Long context is partitioned
into a **far** part (old) compressed into the MemoryLLM pool, and a **near** part
(recent `K` chunks) kept as QCMem raw depth-`a` hiddens. `d=4096`.

### WRITE (build both memories from a long context)

```
# Split the document into chunk_size chunks in reading order.
far_chunks  = chunks[: -K-1]        # distant context -> mid layer (compressed)
near_chunks = chunks[-K-1 : -1]     # recent K chunks -> base layer (raw)
query_chunk = chunks[-1]

# ---- MID (MemoryLLM): FIFO-inject the far context into the pool ----
#   Each inject runs the FULL ported MemoryLLM (all 32 layers) on one segment,
#   producing delta_memory[1, L, num_tokens, d]; update_memory=True does the
#   trained drop(1/num_blocks) + append into memory[L, 12800, d].
mllm.reset_memory()                            # start from the trained init pool
for seg in far_chunks.grouped_into(num_tokens):   # >16 tokens per README
    mllm.inject_memory(seg_ids, update_memory=True)

# ---- BASE (QCMem raw): chunk-local bottom-a encode of the near chunks ----
#   Uses the SAME Llama-3-8B decoder layers[0:a] (MemoryLLM's `mllm.model.layers`),
#   run chunk-local (no memory prefix, is_injection=False path with empty pool
#   contribution) -> depth-a hidden h_a per near chunk.
sink_ha  = qc_write_chunk(BOS_id)              # attention-sink anchor
near_ha  = [qc_write_chunk(c) for c in near_chunks]   # [1, T_c, d] each
```

### READ (Variant P2 — faithful dual-cadence, recommended)

```
# Resume layers[a:L]. At the split we inject BOTH memory sources with their
# native cadence; the query attends to both through the causal mask.
#
#   near_pack (QCMem raw) : [sink_ha ; near_ha... ; query_ha]     enters ONCE @ layer a
#   far_prefix (MemoryLLM): memory[idx]                            re-prefixed EVERY layer

query_ha = qc_write_chunk(query_chunk)                 # depth-a hidden of the query
h = cat([sink_ha, *near_ha, query_ha], dim=1)          # [1, N, d]  (N near-tokens)
N = h.shape[1]

for idx in range(a, L):
    mem_idx = mllm.memory[idx]                         # [12800, d]  far, per-layer
    #   (optionally subsample memory[idx] -> M_far tokens for speed; M_far<=12800)
    h_in = cat([mem_idx[None].expand(B, -1, -1), h], dim=1)   # [1, M_far + N, d]
    pref = M_far                                       # MemoryLLM prefix length
    # Custom (ported) layer API: prefix_token_length slices the prefix off the
    # OUTPUT (memory positions produce no query output), so h keeps N rows.
    h = mllm.model.layers[idx](
            h_in,
            attention_mask = pyramid_mask,   # near tokens causal among themselves;
                                             # all near tokens attend to far prefix
            position_ids   = pos_ids,        # far prefix + contiguous near positions
            prefix_token_length = pref,
            use_cache = False,
        )[0]                                 # -> [1, N, d]
# add MemoryLLM's bos_embedding contribution exactly as its own forward does when
# add_bos_embedding=True (prepend bos_embedding[idx] to the far prefix each layer).

logits = mllm.lm_head(mllm.model.norm(h))              # [1, N, V]
next_token = argmax(logits[0, -1])                     # greedy decode from query tail
```

* `layers[0:a]` of the far context were already consumed during `inject_memory`
  (they shaped the pool), so resuming from `a` loses nothing extra there — the
  far context reaches the resumed band only through `memory[a:L]`, which is
  exactly MemoryLLM's design. The near context's `layers[0:a]` were paid by the
  QCMem chunk-local writes. So the split is coherent for both sources.
* `a` is the single knob trading compute vs fidelity, shared by both layers:
  `a=0` = RAG upper bound for near + full MemoryLLM depth; larger `a` saves the
  bottom-`a` recompute of the near pack per decode step.

### READ (Variant P1 — single-injection MVP, simpler, approximate)

Prepend one depth-`a` slice of the pool as ordinary tokens and resume once:
```
far_ha = compress(mllm.memory[a])          # [1, M_far, d]  (subsample/mean-pool pool@a)
h = cat([sink_ha, far_ha, near_ha..., query_ha])        # single packed seq
logits = qc_resume(layers[a:L])(h)          # identical to QCMem.read_core, +far_ha block
```
Cheaper (one forward, reuses QCMem `read_core` verbatim) but **discards
MemoryLLM's per-layer memory** (`memory[a+1:L]` unused) -> off-distribution for the
trained pool; expected to need light LoRA finetuning on `layers[a:]` to recover.
Kept as the fast bring-up path / ablation floor.

---

## Initialization

| component | init | reason |
|---|---|---|
| backbone | Llama-3-8B, frozen | shared by both layers; MemoryLLM pool was trained on this exact base |
| MemoryLLM pool `memory` | trained checkpoint `[32,12800,4096]` | 1.68B trained params — the whole point of the mid layer; never re-init |
| `bos_embedding`, `new_memory_positional_emb` | trained ckpt | required by MemoryLLM read/inject; keep as loaded |
| `resume_j = a` | 6 (bring-up), sweep {0,4,6,9,12} | QCMem Llama sweet spot region; a=0 self-tests to full forward |
| `K` (near chunks) | 4 | matches QCMem `topk` default; recent precise window |
| `M_far` | 12800 (P2 full) or 256-1024 (subsampled) | full pool is faithful; subsample trades recall for speed |
| chunk_size | 512 | matches QCMem/BABILong driver default |
| sink | BOS depth-`a` hidden | attention-sink anchor at packed pos 0 (QCMem-proven) |

QCMem side is **training-free**; MemoryLLM pool is **pretrained-and-frozen**. v1
requires **no new training** in Variant P2 (both sub-mechanisms run natively). If
BABILong shows the far/near attention interaction is off-distribution, add a
LoRA on `layers[a:]` distilled toward a full-context teacher (reuse
`scripts/train_qcmem_distill.py` recipe).

## Relationship to prior work

* **MemoryLLM (arXiv:2402.04624)** — we reuse its trained pool + inject/read
  *verbatim* for the far context, but only consume `memory[a:L]` at read (resume
  from `a`) instead of all 32 layers. Novel: pairing it with a raw-hidden near
  cache at a shared split depth.
* **QCMem (this project)** — base layer verbatim (`write_chunk` / `read_core`),
  but the read pack gains a second prefix source (the MemoryLLM pool) with a
  different (per-layer) injection cadence. QCMem alone keeps *raw* recent context
  but has no compressed far memory; Pyramid adds the compressed far tier.
* **Block-Recurrent / Infini-attention / RMT** — those compress *everything* into
  one recurrent state. Pyramid keeps a two-tier split (raw-recent + compressed-far)
  and never compresses the recent window, which is why QCMem beats MemoryLLM on
  precise localisation (qa1) — Pyramid inherits that and adds far coverage.
* **Backbone-consistency decision (方案 A vs B): recommend 方案 A — unify on
  Llama-3-8B.** MemoryLLM's 1.68B pool + LoRA are the *output of a full
  continual-pretraining run on Llama-3-8B*; porting the mechanism to Qwen3-8B
  (方案 B) means re-running that training (weeks of compute) to regenerate a pool.
  QCMem, by contrast, is training-free and backbone-agnostic (it only slices
  layers), so a Llama-3-8B QCMem base works out of the box — we already have the
  Llama backbone loaded here; the only thing "missing" vs the Qwen QCMem line is
  self-distillation, which is cheap to redo on Llama. Net: 方案 A moves the cheap
  component (QCMem) to Llama and keeps the expensive component (MemoryLLM pool)
  where it was trained. 方案 B throws away the trained pool. Choose A.

## Known issues

1. **Two different Llama implementations.** QCMemModel wraps **stock HF Llama**
   (model-level `rotary_emb`, decoder layer takes `position_embeddings`). The
   ported MemoryLLM is a **custom tf-4.43-era Llama** (per-attention `rotary_emb`,
   decoder layer takes `cache_position` + `prefix_token_length`, no
   `position_embeddings`). They cannot share the same `_run_layers`. PyramidMemory
   therefore targets the **MemoryLLM layer API** for both the QCMem-style near
   writes and the resumed read (so a single backbone drives both). QCMemModel is
   used as a *reference*, not imported into the P2 read path.
2. **Custom read mask (P2).** The resumed band needs a bespoke 4D mask: far
   prefix (`M_far` cols) fully visible to all near rows; near rows causal among
   themselves; query tail last. The ported layer builds a mask internally only
   for the `prefix_token_length` fast-path (all-visible prefix + causal tail) —
   which matches near-attends-to-far, but the near block also contains sink+ctx+
   query that must be causal among themselves. Must verify the ported
   `prefix_token_length` mask == the intended pyramid mask, else pass an explicit
   4D `attention_mask`. (Self-test: P2 with `M_far=0` must equal QCMem read.)
3. **Position ids for the mixed prefix.** MemoryLLM assigns memory prefix
   positions `0..M_far-1` then continues; the near pack must get contiguous
   positions after the prefix. RoPE consistency across the two cadences is
   untested — needs a diff-vs-reference gate like QCMem's `--self_test`.
4. **`inject_memory` mutates `self.memory` in place** (FIFO drop+append). Pyramid
   must snapshot/reset the pool per document (`reset_memory()` to the trained
   init) or far context bleeds across eval samples. Not yet implemented.
5. **`model.generate()` is broken on the port under tf-5.5.4** (its `_prefill`
   passes `cache_position=None` to the tf-4.43 `prepare_inputs_for_generation`).
   Pyramid decodes with a manual greedy loop of full reads (as the smoke test
   does), not `.generate()`.
6. **P2 cost.** Re-prefixing the full 12800-token pool at every layer for every
   decode step is expensive (12800+N attention per layer × (L-a) layers × new
   tokens). Subsample `M_far` or cache the far-prefix KV across decode steps
   (memory is static during decode) — an optimisation, not v1-blocking.
