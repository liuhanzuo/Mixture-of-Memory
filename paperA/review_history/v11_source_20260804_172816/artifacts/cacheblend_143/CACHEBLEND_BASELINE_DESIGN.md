# Paper A baseline design — CacheBlend chunk-KV (★1, #143) + CoMem dense-selector swap (★2, #144)

> **Scratch design record only.** MAIN owns `TODOList.md` / `status/`; this file is a
> read-only scout's implementation plan, not a task board. Every code claim is
> anchored to a `path:line`. Nothing here has been executed; no `.tex` / status /
> versions file was touched.

The two baselines share ONE governing principle already used by every other Paper A
baseline (`kvdirect` / `hcache`, `eval_ruler_qcmem.py:356-395`): a baseline is a
**re-parameterisation of the CoMem eval harness where exactly ONE variable changes**,
so the comparison is single-variable and same-backbone / same-sample / same-scoring.

- **★1 CacheBlend** changes the *cache object*: full 36-layer per-chunk KV (144 KiB/tok)
  instead of one depth-12 residual `h_j` (8 KiB/tok). Everything else — chunk=512,
  iter_bm25 top-12, sink=BOS, pack order — is held identical to flagship CoMem.
- **★2 dense-selector swap** changes the *selector*: BGE-large dense top-12 instead of
  iter_bm25 top-12. Everything else — resume_j=12, flagship LoRA, chunk=512, sink=BOS —
  is held identical to flagship CoMem.

---

## 0. Shared verified facts (Qwen3-8B geometry + storage math)

From `models/Qwen3-8b-local/config.json` (identical in `models/Qwen3-8B-Base/config.json`):
`num_hidden_layers=36` (:18), `num_attention_heads=32` (:17), `num_key_value_heads=8`
GQA-4 (:19), `head_dim=128` (:9), `hidden_size=4096` (:11), `rope_theta=1e6` (:22),
`max_position_embeddings=40960` (:14), bf16 (:25).

**KV bytes/token (GQA-correct):**
`2(K+V) × 36 layers × 8 kv_heads × 128 head_dim × 2 B = 147,456 B = 144.0 KiB/token`
(147456 / 1024 = 144 exactly; 4 KiB per layer/token).

> ⚠️ The often-feared "n_kv=8 makes it tiny" is a red herring for *this* model: GQA
> already gives the 4× reduction vs MHA and the product still lands at exactly 144 KiB.
> The number that is WRONG is using the 32 *query* heads: `2×36×32×128×2 = 576 KiB/tok`
> (4× too large). Cite **144 KiB** (GQA), never 576 KiB. Matches the paper text:
> `paperA/sections/04_methodology.tex:26` "8 KiB rather than 144 KiB/token in bf16".

**CoMem depth-j residual:** `d × 2 B = 4096 × 2 = 8,192 B = 8 KiB/token`
(`paperA/sections/05_experiments.tex:41`; `04_methodology.tex` closed form
`|h_j|/|KV| = n_q/(2·L·n_kv) = 32/576 = 1/18`).

**Ratio 18×.** At a fixed **1 GiB** store budget:
- CoMem: `2^30 / 8192 = 131,072 tokens = 128k` (exactly 1 GiB at 128k, matches
  `04_methodology.tex:26`).
- CacheBlend: `2^30 / 147456 = 7,281 tokens ≈ 7.1k`.
So under equal storage, CacheBlend reaches ~18× fewer context tokens than CoMem — the
crux of the "storage-limited winner" question.

**Venue for the citation:** CacheBlend = Yao et al., **EuroSys 2025**, **arXiv:2405.16444**
(`paperA/qcmem.bib:379-388`). NOT "ACL 2024" — the task prompt's venue is wrong; the bib
entry is already correct.

---

## 1. Verified substrate (what already exists, reused unchanged)

### 1.1 Selector + pack (identical to CoMem, reused by BOTH baselines)
- `_select_context_chunk_indices(selector, context_chunks, query_ids, topk, needle_chunk_set, …)`
  — the single top-k choke point at `scripts/eval_qcmem_babilong.py:464-576`. iter_bm25
  branch `:530-539` (→ `_iter_bm25_indices`, hop=4, pure CPU); **oracle branch `:508-518`
  packs EXACTLY the supplied doc-absolute `needle_chunk_set`** — this is the no-edit
  injection point for a pre-computed dense selection.
- Chunking convention (every driver): `tokens = input_ids[0]; chunks = tokens.split(512);
  context_chunks = chunks[:-1]; query_chunk = chunks[-1]` (`bench_p0_13_quality_latency.py:231-234`).
- Pack order `[sink=BOS ; selected ctx (doc order) ; query]`, fresh contiguous RoPE `0:H`,
  causal mask (`_build_pack` `bench_p0_13:227-262`; `QCMemModel.read_core:515-551`).

### 1.2 CoMem read/write primitives (`src/memory/qcmem/qcmem_model.py`)
- Accessors exposed: `embed_tokens`(:94), `layers`(:95), `norm`(:96), `rotary_emb`(:97),
  `lm_head`(:98), `num_layers`(:100), `config`(:99).
- `_run_layers(hidden, layer_slice, mask, positions, position_embeddings, past_key_values=None, use_cache=False)`
  (:306-367) — runs ANY layer slice, optional `DynamicCache` keyed on each layer's real
  `self_attn.layer_idx` (:322-327); separate bands use separate caches so indices never collide.
- `write_chunk`(:372-392) / `write_chunks`(:394-464) — embed + `layers[0:resume_j]`
  chunk-local → depth-j hidden `h_j` `[1,T,d]`; bottom-j KV **discarded**.
- `write_prefill`(:640-663) — same bottom band **with** a `DynamicCache` (use_cache=True).
  **This is the exact template for CacheBlend's full-depth chunk prefill** (generalise
  `slice(0, resume_j)` → `slice(0, L)`).
- `read_core`/`read`(:469-571) — cat per-chunk `h_j` (:527) + fresh RoPE `0:H` (:529) +
  resume `layers[resume_j:L]` (:545). Concats HIDDEN STATES, **not raw K/V**.
- `read_prefill`(:665-702) / `decode_step`(:704-743) — top-band KV-cache decode.

### 1.3 Frozen dense retriever (reused unchanged by ★2)
- `DenseRetriever` (`scripts/eval_p1_9_dense_rag.py:125-220`): BGE-large-en-v1.5, CLS
  pooling + L2-norm + cosine. Constructor fail-closed sha256 gate (:141-149,
  `EXPECTED_BGE_SHA256` :102-103, `EXPECTED_BGE_REVISION` :104) + CLS-pooling gate
  (:151-160). `select_topk(context_texts, query_text, topk) -> (sel_idx sorted doc-order,
  scores dict, latency_ms, index_bytes)` (:198-220). Query gets `BGE_QUERY_INSTRUCTION`
  (:105); deterministic stable tie-break `(-score, idx)` (:214-216).
- **Unified dense-RAG harness `eval_p1_9_dense_rag.py` already covers all target
  benchmarks**: `_FAMILY_ITER = {babilong, longeval, locomo, ruler}` (:389-392),
  `_FAMILY_SCORE` (:399-432), imports `qcb/qle/qlo/qru/ruler` (:89-93), and its runner
  does dense-select → `qcmem_generate(selector="oracle", needle_chunk_set=set(sel_idx))`
  (:577-609). **But it HARD-GUARDS `resume_j==0` and empty LoRA** (two
  `raise SystemExit("[p1.9][ABORT] …")` guards at :464-467 for `resume_j!=0` and :468-470
  for a non-empty `--lora_adapter`) — so it cannot be flag-flipped to the CoMem arm; ★2
  must be a NEW file that reuses its parts (see §3).

### 1.4 Byte-accounting helpers (reused by ★1 for storage columns)
- SnapKV/PyramidKV per-layer GQA KV byte accounting `compressed_kv_bytes`(:239) /
  `retained_kv_stats`(:213) in `src/baselines/qwen3_kvcompress.py` — reuse the counting
  idiom for CacheBlend's store-size columns.

### 1.5 Flagship config anchors (both baselines must match)
- LoRA adapter default `outputs/qcmem_distill_qwen_j12_r32_4k/final`
  (`bench_p0_13:814`, `eval_p0_20_equal_latency.py:1094`); sha-gated by `EXPECTED_LORA_SHA`
  (`bench_p0_13:96`).
- RULER "Cohort A" = **NIAH + variable_tracking** (`eval_ruler_qcmem.py:6`, task aliases
  `:105-114`); default lengths `[4k,8k,16k,32k]` (:337-338). LoCoMo resume_j=12 default
  (:670), topk 12 (:722). Protocol: **chat_template=False, enable_thinking=False,
  iter_bm25, chunk=512, sink=bos** (project memory; config #2 anchor
  `eval_p1_9_dense_rag.py:9-27`).

---

## 2. ★1 — CacheBlend-style chunk-KV baseline (#143)

### 2.1 New file
`scripts/eval_cacheblend_qcmem.py` — a standalone driver. It reuses P1.9's family
iterators + scorers for RULER/LoCoMo/BABILong (import, do not edit), and CoMem's
selector + chunking, changing ONLY the cache object.

### 2.2 Pipeline (faithful CacheBlend, NOT naive concat)
1. **Retrieve** top-12 with the SAME flagship selector as CoMem:
   `sel_idx = qcb._select_context_chunk_indices("iter_bm25", context_chunks, bare_q_ids, 12, None, iter_hop_topk=4)`
   (`eval_qcmem_babilong.py:530-539`). Identical chunking (`tokens.split(512)`, ctx=chunks[:-1]).
2. **Offline full-depth chunk-KV precompute** — NET-NEW helper `prefill_chunk_full(qc, ids)`
   in the new file: mirror `write_prefill` (`qcmem_model.py:640-663`) but with
   `slice(0, L)` and chunk-local RoPE `0:T` + per-chunk causal mask + `use_cache=True`,
   using the exposed accessors `qc.embed_tokens`, `qc._make_mask_and_rope`, `qc._run_layers`,
   `DynamicCache`. Returns one `DynamicCache` (all 36 layers' K/V) per chunk. ~15 lines,
   **no edit to `qcmem_model.py`** (uses only public accessors). Do the same for the BOS
   sink (1-token cache) and the query chunk.
3. **Concat KV in CoMem pack order with RoPE re-index** — NET-NEW `concat_kv_reindex(...)`.
   Merge per-layer K/V of `[sink ; ctx_{sel_idx} ; query]` into one `DynamicCache` in pack
   order. Each chunk's cached K was rotated at LOCAL positions `0:T_c`; repair to GLOBAL
   pack positions. **RoPE is a rotation, so `K_global = R(Δ)·K_local` with `Δ = global_offset_c`
   computed from `qc.rotary_emb` cos/sin** — exact, no need to store pre-RoPE K. This is the
   classic PIC/TurboRAG failure point: getting it wrong silently corrupts positions.
   ⚠️ Verify against the Qwen3 HF attention (whether cache stores pre- or post-rotary K)
   before implementing — see risks. ~40-60 lines.
4. **Selective boundary recompute (HKVD) — the load-bearing step, NET-NEW.** At an early
   resumed layer recompute fresh KV for all packed tokens (query attends to the full
   preceding blended KV at global positions); measure per-token deviation
   `δ_t = ||KV_fresh(t) − KV_cached(t)||`; select top-`r%` HKVD tokens; for the remaining
   layers recompute ONLY those HKVD tokens' KV (sparse-update the merged cache), keep
   cached KV for the rest. Requires a custom decoder loop mixing reused + freshly-computed
   per-layer K/V (stock `layer()` with a single mask cannot do a token subset). ~80-120 lines.
   - **Knob:** recompute ratio `r ∈ {0.0 (naive-concat floor), 0.10, 0.15, 0.18, 1.0
     (full-recompute ceiling)}`. `r=0` = the strawman lower bound; `r=1` = full-context
     prefill upper bound.
   - **Faithful-minimal 1–2 day version:** seed the HKVD set once at an early layer and
     freeze it across layers (skip per-few-layers re-ranking). **Non-negotiable for
     faithfulness:** step 3 (global-RoPE repair) + step 4 (selective recompute). Dropping
     step 4 collapses it to naive KV-concat = a strawman that misrepresents CacheBlend.
5. **Decode** over the blended full-KV cache: plain single-band incremental decode (all 36
   layers, no resume split) — embed new token, `_run_layers(slice(0,L), past=merged_cache,
   use_cache=True)`, norm + lm_head. Mirrors `decode_step` shape but over the merged cache.

### 2.3 What is reused vs newly written
| Component | Source | Status |
|---|---|---|
| chunking + iter_bm25 top-12 | `eval_qcmem_babilong.py:464-539` | reuse |
| family iterators + scorers (RULER/LoCoMo/BABILong) | `eval_p1_9_dense_rag.py:263-432` | import unchanged |
| model accessors `_run_layers`/`embed_tokens`/`rotary_emb`/`norm`/`lm_head`/`DynamicCache` | `qcmem_model.py:94-367,640-663` | reuse (public) |
| KV byte-accounting idiom | `qwen3_kvcompress.py:213-254` | reuse |
| `prefill_chunk_full` (slice(0,L) + use_cache) | NEW in `eval_cacheblend_qcmem.py` | **new (~15 L)** |
| `concat_kv_reindex` (merge + delta-RoPE) | NEW | **new (~40-60 L)** |
| HKVD selective-recompute pass + `r` sweep | NEW | **new (~80-120 L)** |
| full-KV decode loop | NEW | **new (~20 L)** |

### 2.4 Eval cells + protocol
RULER Cohort A (NIAH + variable_tracking, lengths 4k/8k/16k/32k) + LoCoMo + BABILong qa5.
`chat_template=False`, `enable_thinking=False`, selector=iter_bm25 (matched to CoMem),
chunk=512, topk=12, sink=bos, Qwen3-8B `models/Qwen3-8b-local`, bf16/SDPA, one H20 — the
config #2 protocol, single-variable vs flagship CoMem.

### 2.5 Storage accounting to report
Per-token: CacheBlend **144 KiB/tok (full 36-layer KV, GQA-correct)** vs CoMem **8 KiB/tok**
(18×). Total store for a 1 GiB budget: CacheBlend **7,281 tok ≈ 7.1k**, CoMem **131,072 tok
= 128k**. **State explicitly that CacheBlend does NOT compress storage** — it caches the same
bytes as a full KV cache and wins only on prefill/TTFT; the fair table must show the 144
KiB/tok tier next to any prefill-latency win and never file it as storage-saving.

### 2.6 The three questions ★1 answers
1. **Residual 8 KB vs full KV 144 KB:** same retrieval + same pack, only the cache object
   differs — does keeping full 36-layer KV (18× the bytes) beat keeping one depth-12
   residual on quality?
2. **Decompose CoMem's RULER gap: "no KV" vs "depth partition".** CacheBlend keeps all-layer
   KV but still assembles chunks independently (cross-chunk fusion only via selective
   recompute). If CacheBlend closes CoMem's gap to full-context → the gap was information
   thrown away by storing only `h_j`. If it does not → the gap is intrinsic to
   chunk-independent assembly / the depth partition, not the cache object.
3. **Storage-limited winner:** at a fixed store budget (1 GiB, or equal-storage 8 KiB/tok)
   CacheBlend caches ~18× fewer tokens (~7k vs 128k) → CoMem should dominate long context.
   Headline claim of the efficiency table.

---

## 3. ★2 — CoMem dense-selector swap (#144)

### 3.1 New file
`scripts/eval_comem_dense_selector.py` — a thin driver that **imports** P1.9's parts
unchanged and only changes the reader depth + LoRA + the fact that the dense selection
now fills CoMem's `h12`.

Why a new file and not a P1.9 flag: `eval_p1_9_dense_rag.py:464-470` HARD-GUARDS
`resume_j==0` and empty LoRA via two `raise SystemExit("[p1.9][ABORT] …")` guards
(:464-467 and :468-470), precisely to keep P1.9 the pure RAG reference. So ★2 copies
P1.9's ~40-line runner loop with the guards removed and `resume_j=12` + flagship LoRA
loaded (via `_load_with_peft`, `eval_p0_20_equal_latency.py:178`).

### 3.2 Pipeline (single variable vs flagship CoMem)
1. `retriever = DenseRetriever(retriever_path, device, dtype)` — imported from
   `eval_p1_9_dense_rag.py:125` **unchanged** (sha + CLS gates fire on construction).
2. `qc = QCMemModel(model, resume_j=12)` with the flagship LoRA
   (`outputs/qcmem_distill_qwen_j12_r32_4k/final`, sha-gated) — the CoMem reader, NOT the
   j=0 RAG reader.
3. Per sample, reuse P1.9's `_FAMILY_ITER[family]` to build the exact config-#2 sample;
   chunk `tokens.split(512)`; `ctx_texts = [tokenizer.decode(c, skip_special_tokens=True) …]`;
   `sel_idx, _, _, _ = retriever.select_topk(ctx_texts, query_text, 12)`.
4. `output = qcb.qcmem_generate(qc, tokenizer, input_ids, chunk_size=512, …,
   selector="oracle", needle_chunk_set=set(sel_idx), topk=12, bare_question_ids=…)`
   — the **oracle branch** (`eval_qcmem_babilong.py:508-518`) packs EXACTLY those
   doc-absolute indices, and `qc.write_chunk/write_chunks` writes them to depth-12 → the
   dense-selected chunks become CoMem's `h12`.
5. Score with P1.9's `_FAMILY_SCORE[family]` (import unchanged).

The ONLY variable vs flagship CoMem(iter_bm25) is the selector (`iter_bm25 → dense_bge`);
resume_j=12, LoRA, chunk=512, topk=12, sink=bos are all held.

### 3.3 Reconciliation vs #140 (P1.9) and #141 (P0.20 PhaseB) — NOT a duplicate
- **#140 P1.9** = dense selection feeding the **resume_j=0, LoRA-OFF** full-recompute RAG
  reader (`eval_p1_9_dense_rag.py:14,464-470,498`). Dense never touches CoMem's h12.
- **#141 P0.20 PhaseB** swaps only the **text-RAG arm's** selector to dense; the CoMem arm
  is byte-identical to Phase A (resume_j=12, iter_bm25, pre-stored h12 —
  `eval_p0_20_phaseB_dense.py:5-11,217,226-235`). Dense never touches CoMem's h12.
- **#144 (this)** feeds dense selection into the **resume_j=12 + LoRA** CoMem reader — i.e.
  dense chooses which chunks land in `h12`. No existing script does this (P1.9 fail-closes
  on resume_j≠0). It answers a distinct question: **does CoMem's own quality improve when
  its slots are filled by dense retrieval instead of BM25?** — orthogonal to "does a
  stronger RAG baseline overturn CoMem?" (#141).

### 3.4 Reused vs new
| Component | Source | Status |
|---|---|---|
| `DenseRetriever` (BGE, sha-gated) | `eval_p1_9_dense_rag.py:125-220` | import unchanged |
| family iterators + scorers | `eval_p1_9_dense_rag.py:263-432` | import unchanged |
| `qcmem_generate` oracle branch | `eval_qcmem_babilong.py:508-518,583+` | reuse (no edit) |
| QCMemModel resume_j=12 + LoRA loader | `eval_p0_20_equal_latency._load_with_peft` | reuse |
| runner loop (resume_j=12, guard removed) | NEW in `eval_comem_dense_selector.py` | **new (~40 L)** |

### 3.5 Eval cells + protocol
Identical to §2.4 but selector=dense_bge (top-12): RULER Cohort A + LoCoMo + BABILong qa5,
chat_template=False, enable_thinking=False, chunk=512, topk=12, sink=bos, flagship LoRA,
resume_j=12.

---

## 4. new_files / reused_files / risks

**new_files:** `scripts/eval_cacheblend_qcmem.py`, `scripts/eval_comem_dense_selector.py`.

**reused_files (imported UNCHANGED):** `scripts/eval_p1_9_dense_rag.py` (DenseRetriever +
`_FAMILY_ITER` + `_FAMILY_SCORE`), `scripts/eval_qcmem_babilong.py`
(`_select_context_chunk_indices`, `qcmem_generate`, `QCMemModel`, `harness`),
`src/memory/qcmem/qcmem_model.py` (public accessors), `scripts/eval_qcmem_locomo.py` +
`scripts/eval_ruler_qcmem.py` (sample construction, via P1.9),
`scripts/eval_p0_20_equal_latency.py` (`_load_with_peft`, `EXPECTED_LORA_SHA`),
`src/baselines/qwen3_kvcompress.py` (byte-accounting idiom for ★1).

**risks:**
- **[★1, high] RoPE re-index correctness.** Merging independently-prefilled chunk KV needs
  exact global-position repair; Qwen3 cache stores post-rotary K, so verify whether to
  delta-rotate cached K or capture pre-RoPE K in the precompute forward before coding.
  Mishandling silently corrupts positions (TurboRAG/PIC failure mode).
- **[★1, high] Selective-recompute is net-new custom attention.** Must mix reused +
  freshly-computed per-layer K/V for a token subset; stock `layer()` cannot. Omitting it
  reduces the baseline to naive KV-concat (strawman) — must not ship that as "CacheBlend".
- **[★1, med] Exact HKVD selection heuristic / recompute-ratio band unverified against
  arXiv:2405.16444** (WebFetch was blocked; ~15%, cross-layer stability recalled not read).
  Confirm Algorithm 1 before quoting any speedup number; the `r`-sweep design is robust to
  the exact default.
- **[★1, med] Full-KV memory at long context** (144 KiB/tok → 19.3 GB at 128k). RULER 32k
  is fine on one H20; do not push CacheBlend to 128k without checking peak mem.
- **[★2, low] BGE bf16 GPU matmul cross-node determinism** not guaranteed bit-identical
  (`eval_p0_20_phaseB_dense.py:696-736` only cross-checks via input_ids sha). Fix device +
  dtype; report the retriever revision.
- **[★2, low] `needle_chunk_set` semantics.** oracle branch expects DOC-ABSOLUTE indices;
  `DenseRetriever.select_topk` returns doc-absolute sorted indices (`eval_p1_9_dense_rag.py:198-216`)
  → compatible, but assert `0<=c<n_ctx` (oracle already filters at `:512`).
- **[both, low] "Cohort A" / config #2 exact task+length list** taken from
  `eval_ruler_qcmem.py:6,337-338` defaults; pull the frozen per-benchmark k/length list
  from the flagship eval config before launch so cells match CoMem 1:1.

**duplicates_done_work:** No. ★1 (#143) has NO in-repo implementation (grep for
cacheblend/hkvd/kv_deviation returned only bib citations); the closest substrate (HCache =
layer-band recompute over all tokens) is a DIFFERENT mechanism from CacheBlend's
token-selective recompute across all layers. ★2 (#144) is NOT done by #140 (dense→j0 RAG
reader) or #141 (dense→text-RAG arm, CoMem untouched); neither ever feeds a dense selection
into the resume_j=12 + LoRA CoMem reader, and P1.9 hard-guards against it (:464-470), so a
new file is required. Both are genuinely new single-variable arms.
