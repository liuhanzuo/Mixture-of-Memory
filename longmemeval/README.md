# LongMemEval system-memory baseline (Track B)

A modular, official-paper-compatible RAG **memory baseline** for the
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark
(Wu et al. 2024, [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)).

This is the **baseline plus a lightweight MoM-assisted extension seam**. The
current MoM Track B variant is API-free/GPU-free: cheap rerankers and a
`MoMSlotReranker` stub that preserves the future slot/query-similarity adapter
surface (see [MoM extension points](#mom-extension-points)).

## What LongMemEval is

A long-term *interactive* memory benchmark: each question comes with a haystack
of timestamped user/assistant **sessions** plus a question asked on a given
date. 500 questions across 5 abilities: information extraction, multi-session
reasoning, knowledge updates, temporal reasoning, abstention. The `S` variant
(`longmemeval_s`) is ~115k tokens / 30–40 sessions per question.

Data fields per question: `question_id`, `question_type`, `question`, `answer`,
`question_date`, `haystack_sessions`, `haystack_dates`, `answer_session_ids`.
Submission is JSONL of `{question_id, hypothesis}`; the official eval uses a
GPT-4o auto-evaluator plus turn/session recall.

## Data download (not bundled)

No data is committed to this repo. Download the official files and point
`--data` at them:

```bash
# Option A: HuggingFace dataset (recommended)
#   https://huggingface.co/datasets/xiaowu0162/longmemeval
# Option B: the GitHub repo release
#   https://github.com/xiaowu0162/LongMemEval
```

You get three JSON files: `longmemeval_s.json` (≈115k tok, main),
`longmemeval_m.json` (longer), `longmemeval_oracle.json` (gold-session only).
Suggested local placement: `data/longmemeval/longmemeval_s.json`.

If the file is missing, the loader raises a `FileNotFoundError` pointing back
here — nothing is auto-downloaded.

## Dependencies (venv status as of build)

Checked against `.venv/bin/python`:

| package | status | used for |
|---|---|---|
| `transformers` | **present** (5.5.4) | HF embedding encoder (e.g. `models/bge-m3`) |
| `torch` | **present** (2.10.0+cu128) | embedding encoder |
| `numpy` | **present** | vector ops |
| `scikit-learn` | present | (not required by baseline) |
| `rank_bm25` | **MISSING** | not needed — BM25 is implemented in pure Python |
| `sentence_transformers` | **MISSING** | optional; falls back to raw `transformers` encoder |
| `openai` | **MISSING** | only needed for `--reader openai`; `--reader stub` needs nothing |

**Graceful degradation:**
- BM25 has **no external dependency** (pure-Python Okapi BM25).
- `embedding` / `union` retrievers try `sentence-transformers`, then a raw
  `transformers` encoder at `--embed_model` (default `models/bge-m3`, which is
  present locally). If neither loads, the retriever **transparently degrades to
  BM25-only** and reports `degraded: true` in the metrics.
- `--reader openai` requires the `openai` package + an API key env var. Without
  them, use `--reader stub` (no LLM, recall diagnostic only).

## How to run

### 0. Synthetic smoke (no data, no API, no extra deps)

```bash
.venv/bin/python -m longmemeval.run_baseline --self_test
```

### 1. BM25 retriever + stub reader (retrieval recall diagnostic)

```bash
.venv/bin/python -m longmemeval.run_baseline \
    --data data/longmemeval/longmemeval_s.json \
    --retriever bm25 --top_k 10 --reader stub \
    --out outputs/longmemeval/bm25_stub.jsonl \
    --report outputs/longmemeval/bm25_stub.report.json
```

### 2. BM25 + lightweight reranker recall sweeps (no API/GPU)

Use `--reader stub` to measure retrieval recall only. Reranking happens over a
larger first-stage pool (`--candidate_multiplier * --top_k`) before final
`top_k` evidence is budgeted.

```bash
for reranker in none keyword temporal mom_stub; do
  .venv/bin/python -m longmemeval.run_baseline \
      --data data/longmemeval/longmemeval_s.json \
      --retriever bm25 --reranker ${reranker} \
      --top_k 10 --candidate_multiplier 8 --reader stub \
      --out outputs/longmemeval/bm25_${reranker}_stub.jsonl \
      --report outputs/longmemeval/bm25_${reranker}_stub.report.json
done
```

`mom_stub` is intentionally cheap: it uses the `MoMSlotReranker` interface but
falls back to keyword-overlap scoring until real MoM slot/query similarities are
plugged in.

### 3. Embedding retriever (local bge-m3) + stub reader

```bash
.venv/bin/python -m longmemeval.run_baseline \
    --data data/longmemeval/longmemeval_s.json \
    --retriever embedding --embed_model models/bge-m3 \
    --top_k 10 --reader stub \
    --out outputs/longmemeval/embed_stub.jsonl
```

### 4. Union (BM25 ⊕ embedding, reciprocal-rank fusion) + GPT-4o reader

```bash
export LONGMEMEVAL_READER_API_KEY=...        # never hardcode
# optional: export LONGMEMEVAL_READER_BASE_URL=...   (for vLLM/compatible endpoints)
.venv/bin/python -m longmemeval.run_baseline \
    --data data/longmemeval/longmemeval_s.json \
    --retriever union --embed_model models/bge-m3 \
    --top_k 10 --evidence_token_budget 4000 \
    --reader openai --reader_model gpt-4o \
    --out outputs/longmemeval/union_gpt4o.jsonl
```

### Reader API env vars

| var | default | meaning |
|---|---|---|
| `LONGMEMEVAL_READER_API_KEY` (or `OPENAI_API_KEY`) | — | API key (required for `--reader openai`) |
| `LONGMEMEVAL_READER_BASE_URL` (or `OPENAI_BASE_URL`) | OpenAI default | OpenAI-compatible endpoint (vLLM, etc) |
| `LONGMEMEVAL_READER_MODEL` | `gpt-4o` | reader model id |

## Output

1. **Submission JSONL** (`--out`): one `{"question_id", "hypothesis"}` per line —
   the exact format the upstream `evaluate_qa.py` GPT-4o judge consumes. This
   harness does **not** bundle the GPT-4o judge (it needs your API key); it
   produces the file the judge expects.
2. **Metrics report** (printed; `--report` to save): API-free **session-level
   retrieval recall@k** vs `answer_session_ids` — `any_hit_recall` (≥1 gold
   session in top-k) and `mean_covered` (fraction of gold sessions covered),
   overall and per `question_type`. Use this to tune retriever / `top_k`
   without spending judge API calls.

## Pipeline structure

```
longmemeval/
  data.py         load_longmemeval(), iter_rounds()  -> ROUND-level units
  backends.py     MemoryBackend(ABC), Reranker(ABC), RoundFlatRetriever
                  (pure-python BM25 + optional dense embeddings, RRF union)
  reader.py       Reader(ABC), StubReader, OpenAIChatReader (Chain-of-Note prompt)
  scoring.py      write_submission(), recall_at_k(), aggregate_recall()
  run_baseline.py CLI entrypoint
```

Design follows the paper's working baseline choices: **round-level** memory
granularity (beats session-level), structured evidence blocks carrying session
ids + dates (for temporal reasoning / knowledge updates), and a
Chain-of-Note / structured-JSON reader prompt.

## MoM extension points

The interfaces deliberately leave clean seams for Mixture-of-Memory:

- **MoM-reranker** — subclass `backends.Reranker` to reorder
  `RoundFlatRetriever` candidates using the MoM memory representation; pass via
  `RoundFlatRetriever(reranker=...)` for an **MoM-hybrid** (sparse/dense recall
  + MoM precision). Current CLI choices are `none`, `keyword`, `temporal`, and
  `mom_stub`; `mom_stub` is the no-GPU `MoMSlotReranker` adapter placeholder.
- **MoM-summary-key** — subclass `backends.MemoryBackend` so per-round keys are
  MoM gist summaries instead of raw-text BM25/embedding keys.
- **MoM-compressor** — subclass `backends.MemoryBackend` to compress sessions
  into a fixed-size MoM buffer and `query` that buffer directly.
- **MoM-hybrid** — RoundFlatRetriever recall feeding an MoM-reranker.

To add a variant: implement the relevant ABC, then register it in
`run_baseline._build_reranker` / `_build_retriever` / `build_reader` behind a
new `--reranker` / `--retriever` / `--reader` choice. The recall@k and
submission machinery are reused unchanged.
