# Paper A — P1.9 Dense-retriever + native-prompting standard RAG reference

**Status:** harness built + syntax/import/provenance-gate verified (2026-08-03). NOT
yet run on GPU (DRY-by-default; MAIN launches with `RUN=1` on a free diskB node).

## What P1.9 is (and is not)

A **zero-training** deployment-realistic standard-RAG reference. It swaps **only the
selector** of the config#2 `j=0` RAG-recompute reader:

- lexical `iter_bm25`  ->  **frozen public DENSE retriever** (BGE-large-en-v1.5).

Everything else is held byte-identical, so the comparison isolates lexical-vs-dense
retrieval on the *same* reader and *same* examples:

| held fixed                | value |
|---------------------------|-------|
| reader model             | `models/Qwen3-8b-local`, **NO LoRA** |
| reader depth             | `resume_j=0` (full 36-layer recompute over the selected pack) |
| chunking                 | `chunk_size=512`, `sink=bos` |
| numerics                 | bf16 + sdpa, `seed=42`, greedy |
| examples                 | each family's **own unmodified** sample builder + seed/shard convention |

**It does NOT replace** the matched-BM25 `j=0` reference (config #2, the RAG-recompute
upper bound), and is **not conflated with MemoryLLM**. It is an *additional*
deployment-realistic point beyond matched BM25.

## Mechanism (how the reuse stays bit-exact)

The dense retriever ranks the sample's `context_chunks` (decoded from the reader's
own `input_ids.split(chunk_size)[:-1]`) against a family-specific query, returns the
top-k **document-absolute** chunk indices, and feeds them as `needle_chunk_set` into
the **unmodified** `eval_qcmem_babilong.qcmem_generate` with `selector="oracle"`. The
oracle branch packs *exactly* those indices — so the read pack differs from config#2
`j0-RAG` only in *which* chunks the selector picked. No shared eval module is edited.

## Retriever provenance (fail-closed)

- model: `BAAI/bge-large-en-v1.5` at `models/bge-large-en-v1.5`
- revision: `d4aa6901d3a41ba39fb536a557fa166f842b0e09`
- weight sha256: `45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7` (verified present + matching)
- pooling **CLS** (asserted off `1_Pooling/config.json`), **L2-normalized**, **cosine** (== dot)
- query instruction: `Represent this sentence for searching relevant passages:`; passages raw
- truncation: 512-token position budget; encode dtype bf16
- index: flat brute-force exact cosine, rebuilt per query; size reported = `n_ctx_chunks x 1024 x dtype_bytes`

The driver aborts (`--mode provenance` exit 6; `DenseRetriever` raises) if the weight
sha256 does not match, unless `--allow_retriever_sha_mismatch` (audit only).

## Report decomposition (`--mode aggregate`)

Per (family, task, length) cell:
- **recall@k** — gold *support-span* chunk (family oracle locator) in the dense top-k
  pack, decided **independently of the answer**; unlocatable gold excluded from denom; Wilson 95% CI.
- reader accuracy **conditional-on-hit** and **conditional-on-miss**
- **end-to-end** quality (BABILong/LongEval/RULER: accuracy; LoCoMo: F1 + substr-acc; judge fields emitted for offline GPT-4o)
- **retrieval latency** (ms/query) and **index size** (bytes)
- **all-tasks fail-closed guard**: `--require_family` asserts every requested task is
  present (exit 5) — forbidden to report only dense-wins tasks.

## Reader-prompt variants (both zero-shot, greedy)

- `plain` (default) = unified no-chat main protocol (chat_template OFF, add_special_tokens=True) — the config#2 j0-RAG口径.
- `native` = reader native-prompt / template-sensitivity variant (chat_template ON, no-think generation boundary).

## Files

- `scripts/eval_p1_9_dense_rag.py` — driver (run / aggregate / provenance modes).
- `scripts/_run_p1_9_dense_rag_8gpu.sh` — DRY-by-default 8-GPU flock task-pool launcher.
- `bench_results/p1_9_dense_rag/<family>/` — per-cell jsonl + `.config.json` (full provenance).
- `retrieval_results/p1_9_dense/<family>/` — per-cell index-size manifest.

## Launch (MAIN, on a free diskB H20 node — e.g. .104)

```bash
PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
RUN=1 setsid nohup bash scripts/_run_p1_9_dense_rag_8gpu.sh \
  >logs/p1_9_dense_rag.out 2>&1 &
```

Cohort (default): babilong qa1/qa2 x {4k,8k,16k} + longeval {8k,16k} + locomo + ruler
niah_multikey_1 x {8k,16k}, n=100/cell, 4 shards/cell → 44 jobs. Set
`READER_PROMPTS="plain native"` to also run the template-sensitivity variant.

## Pairing note

Examples pair 1:1 with BM25 `j=0` (config #2) and CoMem because each family's own
sample builder + seed/shard is reused verbatim: BABILong HF Arrow row order with
`range(n)[shard::nshards]`; LongEval per-length `crc32` seed; RULER per-(task,length)
`crc32` seed; LoCoMo `samples[shard::nshards]`. Each record additionally stores
`input_ids_sha256` and `pack_sel_sha256` so pairing can be re-verified downstream.
