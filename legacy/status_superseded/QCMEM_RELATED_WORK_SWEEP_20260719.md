# QCMem / CoMem — Related-Work Proximity Sweep (2026-07-19)

> Purpose: user feedback "现有 related work 感觉不够，还有什么接近我们方法的？". This sweep
> hunts for **mechanistically closest** prior work **not yet in** `paper/sections/02_related.tex`
> / `paper/qcmem.bib`, and sharpens the exact novelty delta.
> Method: every arXiv id below was verified by `curl` against `arxiv.org/abs/<id>` (title + authors +
> date + abstract), through hy-proxy. **No id is from memory.** Nothing here is fabricated; anything
> uncertain is marked "待核实".

## QCMem mechanism (the yardstick for "closeness")
Qwen3-8B backbone. (1) split context into 512-tok chunks; (2) **offline pre-compute + cache each chunk's
hidden state at a single mid layer j (≈0.33L, e.g. j=12/36)** — not KV, not compressed to fewer tokens,
the *raw layer-j residual hidden*; (3) at query time **BM25 / iter_bm25 retrieve top-k chunks**, assemble
`[sink ; cached h_j of selected chunks ; query]` into a **constant-length read pack (~6657 tok)**, and
**resume the forward pass from layer j** (skip layers [0:j] for those chunks); (4) train with
**self-distillation** (bidirectional KL, teacher = full-context j=0 forward, student = resume-from-j
forward, on the teacher's top-64 support, LoRA r32/α64 on layers[j:]). Selling point: **prefill
time/memory constant** in context length (≈50-100× prefill speedup, ~18 GB flat at 128k), read budget
constant, 128k still scores high.

The mechanism decomposes into **four primitives**. Use these to judge every candidate:
- **P1 — cache a partial forward state, recompute the rest** (cache-then-reuse of an intermediate state).
- **P2 — the cached state is a *single mid-layer hidden*, and recompute is *layer-partial* (only [j:L])**
  — the depth-axis partition. This is the one primitive almost nobody else has.
- **P3 — retrieval selects which cached units enter a *constant-length* working set** (context-length-
  independent read).
- **P4 — learning (self-distillation LoRA) makes the reused/resumed forward match the full forward**,
  i.e. train the backbone to tolerate reading from the cache.

---

## Already covered (for reference — do not re-add)
`02_related.tex` + bib already discuss: HCache, KV-Direct/ResidualStream, KV-CAT, CompressKV, YOCO,
InfLLM, StreamingLLM, H2O, SnapKV, PyramidKV, LCKV, Activation Beacon, Landmark, FoT/LongLLaMA,
CacheBlend, RAGCache, MemoryLLM, RecursiveSummarizing, StagesOfInference. The sweep below is **only new
candidates** the user flagged or that I found, plus how each maps onto P1–P4.

---

## Tier S — Closest NEW work; **related work MUST discuss** (novelty defense)

### 1. LLoCO — Learning Long Contexts Offline (Tan, Li, Patil, Wu, Zhang, Keutzer; 2024-04; arXiv:2404.07979)
- **Mechanism**: process long context **offline** → compress it (context compressor, AutoCompressor-style
  summary embeddings) → **in-domain parameter-efficient finetune with LoRA** so the model reads the
  compressed representation → at query time **retrieve** the relevant compressed context and answer.
  Extends 4k LLaMA2-7B to 128k, 30× fewer inference tokens, 7.6× speedup.
- **Same as us**: this is the single closest **pipeline-level** match — **offline pre-compute + LoRA
  adaptation + retrieval** = QCMem's exact top-level recipe (P3 + P4, and P1 in spirit). Both "learn
  contexts offline" and add a LoRA so the frozen backbone tolerates a cheap surrogate of the full
  context.
- **Our novelty vs it (delta)**: LLoCO's stored unit is a **compressed summary-token embedding**
  (lossy, token-count compression, à la AutoCompressor) that is **prepended as soft prompt**; QCMem
  stores the **raw layer-j hidden of the original chunk (lossless at that depth)** and **resumes the
  forward from layer j** (P2). LLoCO has **no depth-axis partition and no layer-partial recompute** —
  it never re-runs the backbone over the original tokens; it conditions on compressed vectors. QCMem's
  "compression" is purely *which layers you skip*, not *how many tokens you keep*. **This is the paper's
  most important new comparison: reviewers will say "isn't this LLoCO?" — the answer is depth-partition
  + resume-from-j + self-distillation vs. token-count compression + soft-prompt injection.**
- **Priority**: **MUST ADD to 02_related.tex**, ideally in the "closest" paragraph next to HCache/KV-Direct.

### 2. Block-Attention for Efficient Prefilling (Ma et al.; 2024-09; arXiv:2409.15355)
- **Mechanism**: in RAG, split retrieved docs into **blocks**; each block computes its **KV
  independently** (block-diagonal attention, only the final/query block attends across); **re-encode
  positions** and **fine-tune the LLM to adapt to block-attention**; then **reuse previously-seen
  blocks' KV**. TTFT ↓98.7%, FLOPs ↓99.8% at 32k; matches full-attention after block finetuning; can
  switch between block/full modes losslessly.
- **Same as us**: (i) **cache-then-reuse of per-chunk state + position re-encoding** (P1/P3); (ii)
  **fine-tune the backbone so it tolerates the block-wise reused cache** — this is *exactly* QCMem's P4
  motivation (train the model to read from a non-full-attention cache); (iii) Block-Attention's
  block-diagonal vs full attention is **precisely the axis of QCMem's own "full-attention recompute vs
  `--reuse_kv_blockdiag`" ablation** (draft §6). This is the closest prior work to that ablation and
  should be cited there.
- **Our novelty vs it (delta)**: Block-Attention caches **full-depth KV at all layers (layer-0
  onward)**; QCMem caches **a single mid-layer hidden and recomputes upper layers** (P2 — depth-axis,
  which Block-Attention has none of). Block-Attention's "reuse" is *avoiding recompute entirely* (store
  KV); QCMem *trades storage for a partial recompute*. Block-Attention's selection is external RAG
  retrieval over full KV; QCMem retrieves chunks whose only stored trace is one layer's hidden.
- **Priority**: **MUST ADD** — both in the retrieval/KV-reuse paragraph AND cited at the block-diag
  ablation.

### 3. CEPE — Context Expansion with Parallel Encoding (Yen, Gao, Chen; 2024-02; arXiv:2402.16617)
- **Mechanism**: a **small encoder processes long input chunk-by-chunk in parallel**; the **frozen
  decoder cross-attends** to the encoded chunk representations. Trained on 8k docs, extends LLaMA-2 to
  128k, 10× throughput, 1/6 memory; explicitly strong in **retrieval-augmented** settings where dense
  long-context models degrade.
- **Same as us**: **parallel per-chunk encoding + a bounded working set + retrieval-friendliness**
  (P1/P3). CEPE's "encode chunks offline, let the main model consume them cheaply" is the same shape as
  QCMem's write/read split.
- **Our novelty vs it (delta)**: CEPE uses a **separate small encoder + cross-attention injection**
  into a frozen decoder (a compress-then-inject variant); QCMem uses **the same backbone**, caches its
  own **mid-layer hidden**, and **resumes the forward** — no auxiliary encoder, no cross-attention
  prefix. QCMem has no separate encoder to train and its cached state is *on the backbone's own
  residual stream* at depth j. QCMem also selects a **constant-k** set via BM25; CEPE cross-attends to
  all encoded chunks.
- **Priority**: **ADD** to the retrieval/parallel-encoding paragraph.

---

## Tier A — Serving KV-reuse (same family as CacheBlend/RAGCache already cited; add 1-line each)

### 4. Prompt Cache — Modular Attention Reuse (Gim et al.; 2023-11; arXiv:2311.04934)
- **Mechanism**: precompute + store **attention (KV) states of frequently-recurring text segments
  ("prompt modules")**; a schema guarantees **positional accuracy** on reuse; splice cached states into
  new prompts. 8-60× TTFT speedup, no weight change.
- **Delta vs us**: reuses **full-depth KV of exact-match segments** (P1, training-free, no retrieval
  ranking, no depth-partition). QCMem reuses **one layer's hidden of retrieved chunks + resume**. Cite
  as prior art for "position-correct reuse of a cached forward state," which QCMem does via chunk-local
  RoPE + fresh RoPE at read.
- **Priority**: mention (group with CacheBlend/RAGCache).

### 5. CacheGen — KV Cache Compression and Streaming (Liu et al.; 2023-10; arXiv:2310.07240)
- **Mechanism**: **compress the KV cache into compact bitstreams** exploiting KV distributional
  structure, and **stream** it with bandwidth-aware adaptation for fast context loading in serving.
- **Delta vs us**: compresses/streams **full KV** for transport (systems/IO objective); QCMem stores a
  **single layer's hidden** and recomputes. Orthogonal storage-encoding trick; cite as "another axis of
  making the cached forward state cheap (compress KV bytes) vs. ours (store fewer layers, recompute)."
- **Priority**: optional mention.

---

## Tier B — Learned compression into fewer tokens/vectors (same family as Activation Beacon; group them)

These are the **compress-then-inject** family: encode context into a few soft tokens/vectors and
inject them. QCMem's whole thesis is the *opposite* of this family (it keeps the raw hidden and
recomputes, never compresses token count), so a **single grouped paragraph** contrasting the family is
the right move. All verified:

| id | name | one-line mechanism | delta vs QCMem |
|---|---|---|---|
| **2305.14788** | AutoCompressor (Chevalier, Wettig, Ajith, Chen; 2023-05) | compress segments into **summary vectors** used as soft prompts; recurrent, unsupervised | QCMem stores raw layer-j hidden, no summary vectors, resume not inject; **LLoCO's compressor is built on this** |
| **2307.06945** | ICAE — In-Context Autoencoder (Ge et al.; 2023-07) | autoencode context into **compact memory slots** (4× compression) directly conditioned on | token-count compression + inject; QCMem = depth-partition, no autoencoder |
| **2304.08467** | Gist Tokens (Mu, Li, Goodman; 2023-04) | train **gist tokens** that compress a prompt so it can be cached/reused | compresses *prompt* into few tokens; QCMem keeps all tokens, drops *layers* |
| **2405.13792** | xRAG (Cheng, Wang, Zhang; 2024-05) | compress a retrieved doc to **one token** via a modality projector for RAG | extreme token-count compression + inject; QCMem stores per-chunk layer-j hidden, retrieves + resumes |

- **Priority**: add ONE grouped sentence ("A compress-then-inject family — Gist, AutoCompressor, ICAE,
  xRAG, and Activation Beacon — encodes context into a few soft tokens/vectors; CoMem instead keeps the
  raw hidden and partitions on depth"). Activation Beacon is already cited; fold the rest in beside it.

---

## Tier C — Retrieval over cached hidden / KV (the *ancestors* of "retrieve over stored representations")

These matter because QCMem's P3 = "retrieve over cached representations." These are the classic
instances — but crucially they retrieve to **feed attention**, not to **resume a forward**.

### 6. Unlimiformer (Bertsch, Alon, Neubig, Gormley; 2023-05; arXiv:2305.01625)
- **Mechanism**: wrap a pretrained **encoder-decoder**; offload cross-attention to a **single kNN index
  over the encoder's top-layer hidden states**; each decoder head retrieves its top-k keys instead of
  attending to all. Handles 500k-token inputs, no new weights.
- **Same as us**: **retrieval over cached *hidden states* (not KV)** — the nearest ancestor to
  "retrieve over stored hidden." (P3.)
- **Delta vs us**: Unlimiformer retrieves hidden to **reconstruct cross-attention scores** inside a
  frozen encoder-decoder; it **does not resume a forward pass** and has **no depth-partition** (P2
  absent). Decoder-only backbone + resume-from-j is what makes QCMem different. Good "we are not the
  first to retrieve over stored hidden, but we are the first to *resume the forward* from them" cite.
- **Priority**: **ADD** (strong for honest positioning).

### 7. Memorizing Transformers (Wu, Rabe, Hutchins, Szegedy; 2022-03; arXiv:2203.08913)
- **Mechanism**: cache **(key,value) pairs at one layer** in a non-differentiable memory; **approximate
  kNN lookup** into it augments that layer's attention; scales to 262k memory.
- **Delta vs us**: caches **KV at a single layer and attends** (not hidden, not resume). Same "one-layer
  memory + retrieval" skeleton but the stored object is KV and it feeds attention, not a resumed
  forward. Classic anchor for "single-layer retrieval memory."
- **Priority**: mention alongside Landmark/FoT (retrieval memory lineage).

---

## Tier D — Depth-semantics / early-exit motivation (supports the "why layer j" rationale, alongside StagesOfInference)

QCMem's claim "semantics saturate at a mid layer, so caching h_j loses little" needs motivation
citations. StagesOfInference (2406.19384) is already cited; add the two canonical **early-exit** works,
noting they exit/skip the **upper** layers (opposite direction) — which is a *useful contrast*: they
argue upper layers are sometimes skippable for easy tokens; QCMem argues the **lower** layers are
recomputable-only-once (cache their output at j) and the **upper** layers must be recomputed with the
query. Same "layers have distinct roles" premise, orthogonal cut.

- **2404.16710** — **LayerSkip** (Elhoushi et al.; 2024-04): train with layer dropout + early-exit loss
  so the model can **exit at earlier layers**; self-speculative decoding verifies with remaining layers.
- **2207.07061** — **CALM / Confident Adaptive LM** (Schuster, Fisch, Gupta et al.; 2022-07): per-token
  **early exit** from upper layers based on confidence; skipped upper layers approximated.
- **Priority**: 1 sentence in the depth-rationale spot (§3 / related), grouped with StagesOfInference.

---

## The exact-combination novelty check (the load-bearing question)

**Has anyone published "cache a single mid-layer hidden + retrieve top-k chunks + resume the forward
from that layer"?** Based on this sweep and the existing file's sweep: **No.** The four primitives each
have precedent, but the *combination* — specifically **P2 (layer-partial recompute of only [j:L] from a
cached single mid-layer hidden)** combined with **P3 (external BM25 retrieval into a constant-length
read)** and **P4 (self-distillation to align resume-from-j with full forward)** — is not present in any
verified prior work:

- **P1 (cache partial state, recompute rest)**: HCache, KV-Direct, Prompt Cache, Block-Attention. But
  they recompute **full-depth KV** (KV-Direct, Prompt Cache) or **upper layers with no retrieval**
  (HCache), or store **full KV** (Block-Attention).
- **P2 (single mid-layer hidden + layer-partial recompute)**: **HCache is the only one that recomputes
  upper layers from a mid activation** — but it is a *serving cache restore*, no retrieval, no training,
  no depth-as-knob framing. **Nobody combines P2 with retrieval.**
- **P3 (retrieve into a constant working set)**: InfLLM, Landmark, FoT, Unlimiformer, Memorizing
  Transformers, LLoCO — but all retrieve **KV or hidden to feed attention**, or **compressed summary
  tokens to inject**; none **resume a forward**.
- **P4 (train backbone to tolerate the reused cache)**: Block-Attention (finetune for block-attn),
  LLoCO (in-domain LoRA on compressed context), Activation Beacon (compression-based training). QCMem's
  twist is **self-distillation from the *full-context* teacher onto the *resume-from-j* student on the
  teacher's top-64 support** — a distillation target that is specific to the depth-partition and, as far
  as this sweep found, unique.

**Verdict**: the novelty claim ("mid-layer-hidden cache + retrieval + resume-from-layer") **holds**. The
danger is not that someone did the exact thing; it is that a reviewer collapses QCMem into "LLoCO
(offline+LoRA+retrieval) with a different compressor" or "HCache + retrieval" or "Block-Attention with
mid-layer instead of layer-0." The defense is to make **P2 (depth-axis partition + layer-partial
recompute)** the crisp, singular contribution and show the others each lack it.

---

## ★ Proximity ranking — Top-8 closest (by mechanism)

| # | Work (arXiv) | Shares | Our unique point vs it |
|---|---|---|---|
| 1 | **HCache** (2410.05004, already cited) | cache **mid-layer hidden + recompute upper layers** (P1+P2) | + retrieval + constant read + self-distillation training + depth-as-knob; HCache is post-hoc serving, no retrieval |
| 2 | **KV-Direct / ResidualStream** (2603.19664, cited) | cache a residual + recompute on demand (P1) | ours is **layer-partial** ([j:L]) not full-depth KV; + retrieval + constant memory (KV-Direct keeps all tokens) |
| 3 | **LLoCO** (2404.07979, **NEW—add**) | **offline compute + LoRA + retrieval** (P3+P4, pipeline twin) | ours stores **raw layer-j hidden + resumes**; LLoCO compresses to summary tokens + injects (no depth-partition, no resume) |
| 4 | **Block-Attention** (2409.15355, **NEW—add**) | cache per-chunk state + position re-encode + **finetune to tolerate reuse** (P1+P3+P4); = our block-diag ablation | ours caches **mid-layer hidden** not full-depth KV; layer-partial recompute vs no-recompute |
| 5 | **CEPE** (2402.16617, **NEW—add**) | parallel per-chunk encode + bounded working set + retrieval-friendly (P1+P3) | ours uses **same backbone + resume-from-j**, not a separate encoder + cross-attention inject |
| 6 | **InfLLM** (2402.04617, cited) | training-free retrieval + super-long extrapolation (P3) | ours stores **one layer's hidden** vs InfLLM's block full-KV; resume vs attention lookup |
| 7 | **Unlimiformer** (2305.01625, **NEW—add**) | **retrieval over cached *hidden states*** (P3) | ours **resumes a forward** from retrieved hidden; Unlimiformer only reconstructs cross-attn scores, enc-dec, no depth-partition |
| 8 | **Activation Beacon** (2401.03462, cited) | **trained** fixed-budget long-context compression (P4) | ours keeps raw hidden + depth-partition; Beacon compresses per-layer activations into beacon tokens (token-count compression) |

(Just outside Top-8: Memorizing Transformers, Landmark, FoT, MemoryLLM, Prompt Cache, StreamingLLM,
AutoCompressor/ICAE/Gist/xRAG.)

---

## Honest novelty positioning (paper-ready paragraph seed)

CoMem's four primitives are individually well-precedented; the contribution is a **specific
intersection plus one primitive nobody else has**. Being candid about which is which:

- **Genuinely new: depth-axis partition with layer-partial recompute (P2).** The only work that caches a
  mid-layer activation and recomputes the layers above it is **HCache**, and it does so as a training-
  free serving-cache *restore* with no retrieval and no notion of depth-as-a-RAG-knob. **No retrieval-
  based long-context method partitions on depth and recomputes only [j:L].** This is the real novelty and
  should be stated as such.

- **New combination, not new ingredients: retrieval into a constant-length resumed forward (P2+P3).**
  Retrieval over cached representations exists (InfLLM, Unlimiformer, Memorizing Transformers, LLoCO),
  and cache-then-recompute exists (HCache, KV-Direct, Block-Attention, Prompt Cache) — but **retrieving
  chunks whose only stored trace is one mid layer, then resuming the forward from that layer** is the
  novel fusion. Frame as "we are the first to make retrieval feed a *layer-partial resume* rather than
  an attention lookup or a soft-prompt injection."

- **Combination creativity, honestly incremental: offline + LoRA + retrieval (P4).** This top-level
  recipe is **shared with LLoCO** and the "finetune-to-tolerate-reuse" idea is **shared with
  Block-Attention**. We should *not* claim "offline compression + LoRA + retrieval" as novel per se —
  LLoCO owns that framing. Our differentiator inside P4 is the **self-distillation target** (full-context
  teacher → resume-from-j student on the teacher's top-64 support), which is specific to the depth
  partition and, per this sweep, not used elsewhere.

**The 1-2 nearest rivals and the crisp cut:**
- **vs LLoCO** (nearest *pipeline*): both are offline + LoRA + retrieval. Cut = **what is stored and how
  it is read**: LLoCO stores a *lossy compressed summary* and *injects it as a soft prompt* (never
  re-runs the backbone over original content); CoMem stores the *raw layer-j hidden* and *resumes the
  forward through the upper layers with the query present* (lossless at depth j, query-conditioned upper-
  layer reasoning). CoMem has a **depth knob j**; LLoCO has a compression ratio.
- **vs HCache** (nearest *inference form*): both cache a mid-layer activation and recompute upper layers.
  Cut = **retrieval + constant read + training**: HCache restores *all* evicted tokens (read cost grows
  with context) with no learning; CoMem retrieves top-k into a *constant-length* read and self-distills
  the backbone to read from depth j — turning a serving-cache trick into a long-context *method* with a
  RAG↔closed-book knob.
- **vs Block-Attention** (nearest *ablation twin*): our full-attention-recompute vs block-diagonal
  ablation is exactly Block-Attention's regime, but on **mid-layer hidden** rather than **layer-0 full
  KV** — cite it at the ablation and claim the depth-axis move as the delta.

**One-sentence positioning**: *CoMem is the first long-context method to partition the transformer on
the **depth** axis — caching a single mid-layer hidden per chunk and recomputing only the upper layers —
driven by external retrieval into a constant-length resumed forward and stabilized by self-distillation;
its nearest neighbors either recompute without retrieval (HCache, KV-Direct, Block-Attention) or retrieve
without a layer-partial resume (LLoCO, InfLLM, Unlimiformer).*

---

## Recommended concrete edits to `paper/sections/02_related.tex` + `paper/qcmem.bib`

**MUST add (novelty defense):**
1. `lloco` = arXiv:2404.07979 — into the "closest" paragraph (offline+LoRA+retrieval twin).
2. `blockattn` = arXiv:2409.15355 — into retrieval/KV-reuse paragraph AND cited at the block-diag ablation in §6.
3. `cepe` = arXiv:2402.16617 — retrieval/parallel-encoding paragraph.
4. `unlimiformer` = arXiv:2305.01625 — "retrieve over cached hidden" ancestor, honest positioning.

**SHOULD add (1 grouped sentence each):**
5. `memtransformers` = arXiv:2203.08913 — single-layer retrieval memory lineage (beside Landmark/FoT).
6. Compress-then-inject group: `gist` 2304.08467, `autocompressor` 2305.14788, `icae` 2307.06945,
   `xrag` 2405.13792 — one sentence beside Activation Beacon.
7. Depth-rationale group: `layerskip` 2404.16710, `calm` 2207.07061 — one sentence beside StagesOfInference.

**OPTIONAL (serving KV-reuse, group with CacheBlend/RAGCache):**
8. `promptcache` 2311.04934, `cachegen` 2310.07240.

**Verified arXiv ids (all curl-checked 2026-07-19):**
- ✅ 2404.07979 LLoCO — Learning Long Contexts Offline (Tan et al., 2024-04)
- ✅ 2409.15355 Block-Attention for Efficient Prefilling (Ma et al., 2024-09)
- ✅ 2402.16617 CEPE — Long-Context LM with Parallel Context Encoding (Yen, Gao, Chen, 2024-02)
- ✅ 2311.04934 Prompt Cache — Modular Attention Reuse (Gim et al., 2023-11)
- ✅ 2310.07240 CacheGen — KV Cache Compression and Streaming (Liu et al., 2023-10)
- ✅ 2305.14788 AutoCompressor — Adapting LMs to Compress Contexts (Chevalier et al., 2023-05)
- ✅ 2307.06945 ICAE — In-context Autoencoder (Ge et al., 2023-07)
- ✅ 2304.08467 Gist Tokens (Mu, Li, Goodman, 2023-04)
- ✅ 2405.13792 xRAG — Extreme Context Compression w/ One Token (Cheng et al., 2024-05)
- ✅ 2203.08913 Memorizing Transformers (Wu et al., 2022-03)
- ✅ 2305.01625 Unlimiformer (Bertsch et al., 2023-05)
- ✅ 2404.16710 LayerSkip (Elhoushi et al., 2024-04)
- ✅ 2207.07061 CALM — Confident Adaptive LM (Schuster et al., 2022-07)

> 待核实: Block-Attention author list — arXiv meta returned "Ma, Dongyang; Wang, Yan; ..." plausibly with
> additional authors (Tingchen Fu et al.); cite as "Ma et al., 2024" until the full list is confirmed.
