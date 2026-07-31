# P1.1 store/distractor scaling + P1.5 task coverage — QCMem (Paper A)

**Date:** 2026-08-01  ·  **Node:** `.82` (28.82.250.82, diskB `zwfy6/share_304376610`, port 36000), 8× H20
**Owner item:** Paper A **P1.1** (store scaling) + **P1.5** (task coverage).
**Per user instruction: this record does NOT edit any `.tex` file and does NOT touch `paperA/TODOList.md`.** Numbers + verdict are handed to `main`.

---

## Thesis under test

**P1.1** — QCMem's "unbounded context" = a **bounded read** (fixed top-k pack, ~6.5k tok) over an
**extensible store**. Fix the relevant evidence + read budget (top-12); grow the store 128k → 4M
tokens by adding distractors. Because the flagship selector is `iter_bm25` (pure-CPU lexical BM25
over token-id chunks) and only the top-12 selected chunks are ever GPU-encoded to `h_j`, the GPU
never sees the whole store → decode is **O(1)** in store size; only store build / lexical retrieval
scales. This lets a single H20 address multi-million-token stores.

**P1.5** — Which tasks most stress a *fixed top-k* retrieval read (not just single-needle)? Report
evidence recall, final-answer quality, read budget, failure cases.

---

## Configuration (flagship口径, identical to `comem_lora_A` / P0.3)

- Entry: `scripts/eval_p1_scaling.py` (NEW; only *imports* the unmodified QCMem forward path
  `scripts/eval_qcmem_babilong.py` + RULER task primitives `scripts/eval_ruler_mem_space.py`).
- Model `models/Qwen3-8b-local` (Qwen3-8B, 36 layers, hidden 4096), LoRA
  `outputs/qcmem_distill_qwen_j12_r32_4k/final`, PY `/opt/conda/envs/torch-base/bin/python`.
- `--resume_j 12 --selector iter_bm25 --topk 12 --iter_rounds 0 --iter_hop_topk 4`
- `--sink_tokens bos --chunk_size 512 --dtype bfloat16 --attn_impl sdpa --seed 42`
- **chat_template = FALSE** (paper mandate). `PYTHONHASHSEED=0`, seed per-cell deterministic.
- Two passes: **retrieval** (model-free, n=20, gives recall/latency/index/read-tokens/coverage) and
  **full** (loads Qwen3-8B+LoRA, greedy-decodes the packed read → answer score, n=10). 8-way GPU/CPU
  shard, `job_idx % 8 == shard_index`; merge = per-cell mean.
- Exact launch (per shard `k`):
  ```
  # retrieval (CPU, no GPU contention):
  CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python scripts/eval_p1_scaling.py --mode retrieval \
    --model_path models/Qwen3-8b-local --resume_j 12 --selector iter_bm25 --topk 12 \
    --iter_rounds 0 --iter_hop_topk 4 --sink_tokens bos --chunk_size 512 --seed 42 --limit 20 \
    --num_shards 8 --shard_index $k --results_dir ruler_results/p1_scaling_retrieval \
    --pool_cache data/p1_qwen3_distractor_pool.npy
  # full (GPU k, answer score):
  CUDA_VISIBLE_DEVICES=$k PYTHONHASHSEED=0 python scripts/eval_p1_scaling.py --mode full \
    --model_path models/Qwen3-8b-local --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
    --resume_j 12 --selector iter_bm25 --topk 12 --iter_rounds 0 --iter_hop_topk 4 \
    --sink_tokens bos --chunk_size 512 --dtype bfloat16 --attn_impl sdpa --device cuda:0 --seed 42 \
    --limit 10 --num_shards 8 --shard_index $k --results_dir ruler_results/p1_scaling_full \
    --pool_cache data/p1_qwen3_distractor_pool.npy
  ```

### Store construction (token-space, EXACT gold-chunk ground truth)

Stores are built in **token space** so recall@k is measured against the *true* evidence chunk set,
not an approximate locator. Each context chunk is exactly 512 tokens and is either pure distractor
(PG19 prose for NIAH; RULER noise haystack for VT) or a needle chunk (RULER-faithful needle sentence
+ distractor pad). Chunk boundaries align bit-for-bit with `qcmem_generate`'s
`tokens.split(512)`. Distractor pool = PG19 natural prose, tokenized once + cached
(`data/p1_qwen3_distractor_pool.npy`, 4.40M tokens). Needle/value/VT-chain generators are RULER's
verbatim (`NIAH_NEEDLE`, `_gen_chain`, `_make_vt_icl`). VT prepends the fixed 4-hop RULER in-context
worked example to the query region so the answer format (variable NAMES) is available at any store
size.

### Task selection rationale (chosen BEFORE seeing results)

Tasks were fixed in `build_jobs()` before any run. The design targets the three ways a *fixed top-k*
read can fail, distinct from single-needle NIAH:
1. **Single-evidence NIAH** (E=1) — control: does one needle stay retrievable as distractors grow to 4M?
2. **Multi-hop / distributed evidence (VT chain)** — evidence is a chain scattered across the store;
   ≥2 chunks must co-occur in the read.
3. **Evidence count EXCEEDS the read budget** (stress) — VT chain-length sweep at fixed 128k so the
   required evidence chunks grow from 4 → 32, crossing top-12.
4. **Cross-chunk aggregation** (niah_multivalue) — one key, E scattered values, must retrieve ALL.
5. **Global statistics** (cwe common-word frequency) — no localized evidence set exists; a fixed
   top-k read fundamentally cannot cover the whole store.

---

## P1.1 — Store scaling  (fixed evidence, fixed top-12; grow store 128k → 4M)

Format: `| Store | Evidence count | Recall@k | Score | Retrieval ms | Index GB | Read tokens | Raw |`.
Recall@k / Retrieval ms from the model-free retrieval pass (n=20); Score from the full pass (n=10).
Index GB = write-once `h_j` store (`n_chunks·512·4096·2 B`, bf16); the lexical BM25 index is
0.5 MB (128k) → 16.8 MB (4M), negligible.

### (a) Single-evidence NIAH (single-key UUID needle), E=1

| Store | Evidence | Recall@k | Score | Retrieval ms | Index GB | Read tokens | Raw |
|-------|----------|----------|-------|--------------|----------|-------------|-----|
| 128k | 1 | **1.000** | 100.0 | 80    | 1.07  | 6211 | `p1_scaling_*` |
| 256k | 1 | **1.000** | 100.0 | 164   | 2.15  | 6209 | ″ |
| 512k | 1 | **1.000** | 100.0 | 328   | 4.29  | 6209 | ″ |
| 1M   | 1 | **1.000** | 100.0 | 697   | 8.59  | 6209 | ″ |
| 2M   | 1 | **1.000** | 90.0  | 1407  | 17.18 | 6211 | ″ |
| 4M   | 1 | **1.000** | 100.0 | 2852  | 34.36 | 6208 | ″ |

→ **Recall and answer score are INVARIANT to store size** (1.000 / ~100 from 128k to 4M). **Read
tokens are constant ~6210** — the read pack does not grow with the store (O(1) read). Retrieval
latency scales ≈linearly with `n_chunks` (BM25 cost), 80 ms → 2.85 s at 4M; the `h_j` index grows
1.07 → 34.4 GB but is never resident on GPU during decode. (2M score 90 = 1/10 sampling miss at n=10;
recall there is still 1.000.)

### (b) Multi-hop / distributed evidence (VT chain, 5 links), E=5

| Store | Evidence | Recall@k | Score | Retrieval ms | Index GB | Read tokens | Raw |
|-------|----------|----------|-------|--------------|----------|-------------|-----|
| 128k | 5 | **1.000** | 18.0 | 20  | 1.07  | 6463 | `p1_scaling_*` |
| 256k | 5 | **1.000** | 34.0 | 39  | 2.15  | 6463 | ″ |
| 512k | 5 | **1.000** | 40.0 | 81  | 4.29  | 6464 | ″ |
| 1M   | 5 | **1.000** | 54.0 | 170 | 8.59  | 6463 | ″ |
| 2M   | 5 | **1.000** | 38.0 | 357 | 17.18 | 6463 | ″ |
| 4M   | 5 | **1.000** | 34.0 | 725 | 34.36 | 6463 | ″ |

→ **Evidence recall for the scattered 5-chunk chain is 1.000 at every store size up to 4M** — the
iter_bm25 BFS walks the lexical chain and pulls all links into the bounded read regardless of store.
Retrieval invariance holds; read tokens constant ~6463 (includes the fixed ICL block). The **answer
score is low and noisy (18–54)**: retrieval succeeds but multi-hop chain-*reasoning* over the
depth-12 `h_j` read is the bottleneck, not retrieval (see failure modes).

### (c) STRESS — required evidence count EXCEEDS top-12 (VT chain-length sweep @128k)

| Store | Evidence | Recall@k | Score | Retrieval ms | Index GB | Read tokens | Raw |
|-------|----------|----------|-------|--------------|----------|-------------|-----|
| 128k | 4  | **1.000** | 62.5 | 19 | 1.07 | 6463 | `p1_scaling_*` |
| 128k | 5  | **1.000** | 18.0 | 20 | 1.07 | 6463 | ″ |
| 128k | 8  | **1.000** | 47.5 | 20 | 1.07 | 6463 | ″ |
| 128k | 12 | **1.000** | 11.7 | 20 | 1.07 | 6464 | ″ |
| 128k | 16 | **0.750** | 7.5  | 20 | 1.07 | 6465 | ″ |
| 128k | 24 | **0.500** | 3.8  | 20 | 1.07 | 6466 | ″ |
| 128k | 32 | **0.375** | 2.2  | 20 | 1.07 | 6465 | ″ |

→ **Clean law: recall@k = min(1, top-k / E).** For E ≤ 12 the bounded read fetches all evidence
(recall 1.000); once E > 12 the fixed read *physically* caps at 12 chunks, so recall = 12/16 = 0.750,
12/24 = 0.500, 12/32 = 0.375 — exactly. This is the boundary of "bounded read over extensible store":
the store can be unbounded, but the answer is only complete while the **required** evidence fits the
read budget. (Read tokens stay ~6465 = the top-12 cap, confirming the read never expands to chase
extra evidence.)

---

## P1.5 — Task coverage (which tasks stress fixed top-k, beyond single needle)

### (d) Cross-chunk aggregation — one key, E scattered VALUES (niah_multivalue @128k)

| Store | Evidence | Recall@k | Score | Retrieval ms | Index GB | Read tokens | Raw |
|-------|----------|----------|-------|--------------|----------|-------------|-----|
| 128k | 1  | **1.000** | 100.0 | 84 | 1.07 | 6210 | `p1_scaling_*` |
| 128k | 4  | **1.000** | 95.0  | 79 | 1.07 | 6211 | ″ |
| 128k | 8  | **0.619** | 56.2  | 79 | 1.07 | 6212 | ″ |
| 128k | 12 | **0.400** | 35.8  | 76 | 1.07 | 6211 | ″ |
| 128k | 16 | **0.309** | 29.4  | 80 | 1.07 | 6211 | ″ |
| 128k | 24 | **0.233** | 20.4  | 82 | 1.07 | 6210 | ″ |
| 128k | 32 | **0.172** | 14.4  | 80 | 1.07 | 6210 | ″ |

→ **Answer quality tracks retrieval recall** (score ≈ recall) — the model reports the values it can
see and misses the rest. Two failures stack: (i) **evidence-count > budget** (E ≥ 16), same
min(1, 12/E) cap; and (ii) **lexical non-separability** — all E values share ONE key, so BM25 cannot
individually rank co-keyed needles and recall drops **even below the budget** (E=8 → 0.62, vs the VT
chain's 1.000 at E=8, because chain links carry distinct linking tokens). Cross-chunk aggregation
over co-referent evidence is the hardest case for a lexical fixed-k read.

### (e) Global statistics — common-word frequency (cwe)

| Store | Metric | Value | Score | Retrieval ms | Index GB | Read tokens | Raw |
|-------|--------|-------|-------|--------------|----------|-------------|-----|
| 128k | evidence coverage | **0.045** | 0.0 | 91  | 1.07 | 6193 | `p1_scaling_*` |
| 512k | evidence coverage | **0.012** | 0.0 | 393 | 4.29 | 6193 | ″ |

→ No localized evidence set exists: the answer (10 most-frequent words) depends on a **global count**
over the whole store. A top-12 read sees only **4.5 % of occurrences at 128k, 1.2 % at 512k**
(coverage → 0 as the store grows), so score = 0. This is the qualitative failure mode of a
fixed-k / retrieval-augmented read: it **cannot** answer questions requiring global aggregation over
the full context — a genuine ceiling that no read budget short of the whole store fixes.

---

## Failure modes (summary)

1. **Evidence count > read budget** — hard cap recall = min(1, top-k/E) (VT stress: E=16/24/32 →
   0.75/0.50/0.375). Mitigation is a larger top-k or multi-round `iter_rounds`, but never
   store-size-dependent.
2. **Lexically co-referent evidence** (niah_multivalue) — co-keyed needles are not individually
   rankable by BM25; recall degrades even for E < budget (E=8 → 0.62). This is a *selector* limit,
   not a capacity limit (a semantic/learned selector would help).
3. **Global-aggregation tasks** (cwe) — fundamentally unanswerable by any bounded top-k read;
   coverage → 0 as the store grows. Retrieval-based memory is the wrong tool here.
4. **Multi-hop chain reasoning** (VT) — retrieval is solved (recall 1.000 to 4M) but the *answer* is
   weak (18–54): the depth-12 `h_j` read carries the chain links but the model inconsistently
   composes the transitive chain and drifts into noise-repetition. Bottleneck is reasoning over the
   bounded read, not retrieval — an important nuance for the "bounded read" claim.

## Verdict for `main`

- **P1.1 thesis confirmed.** Single-evidence recall + answer score are invariant to store size
  (1.000 / ~100 from 128k → **4M** tokens), read tokens constant ~6210, decode cost O(1); only store
  build (index 1.07 → 34.4 GB) and lexical retrieval (80 ms → 2.85 s) scale — and the lexical index
  is 0.5–16.8 MB. A single H20 addressed a **4M-token store** (co-residing with a sibling job).
- The stress test cleanly exhibits the **recall = min(1, top-k/E)** boundary — the honest limit of
  "unbounded context = bounded read over extensible store."
- **P1.5** identifies three failure classes beyond single-needle: evidence-count overflow,
  lexical co-reference (multivalue), and global aggregation (cwe, hard 0), plus the multi-hop
  reasoning-vs-retrieval nuance.

---

## Result artifacts (on `.82` / diskB `zwfy6/share_304376610`)

- `ruler_results/p1_scaling_retrieval/` — model-free pass, n=20: `results_shard{0..7}of8.jsonl` (540 rows, 27 cells) + `merged_summary.json`
- `ruler_results/p1_scaling_full/`      — model pass, n=10: `results_shard{0..7}of8.jsonl` (270 rows, 27 cells) + `merged_summary.json`
- Logs: `logs/p1_retr_shard{0..7}.log`, `logs/p1_full_shard{0..7}.log`
- Distractor pool cache: `data/p1_qwen3_distractor_pool.npy` (PG19, 4.40M tokens, int32)
- Harness: `scripts/eval_p1_scaling.py` (+ `scripts/_p1_smoke.py` GPU sanity)
