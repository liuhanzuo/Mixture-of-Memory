# P0.20 Stage A — BM25 equal-latency frontier: raw-text RAG vs CoMem

**Status:** harness BUILT + py_compile OK (local `.venv` and .104 `torch-base`) +
CPU sanity PASSED + rsync'd to .104 diskB. NOT yet launched (MAIN owns the GPU
launch to avoid colliding with the live P1.9 dense-RAG job on .104).

## Question
At a FIXED online latency budget, can CoMem spend the compute it SAVES (skipping
lower `j=12` layers, resuming from a cached depth-12 residual) to READ MORE
evidence chunks and match/beat raw-text RAG on quality? More direct value test
than the fixed-topk=12 "1.4× model-only read speedup" of P0.13.

## Two paths (share the EXACT SAME pack per example)
The `iter_bm25` selector is forward-free → selected chunk indices/order/packed ids
are `resume_j`-independent → both arms get a **bit-identical pack** (P0.13/P1.7
pairing guarantee, re-asserted per example via `read_len == pack_read_len`).
- **TEXT-RAG (Config #2):** `QCMemModel(model, resume_j=0)` with the LoRA
  **disabled** (`peft_model.disable_adapter()`) == vanilla Qwen3-8B full 36-layer
  recompute over `[sink; k·512; query]`. Online store = raw chunk token ids.
- **CoMem (flagship):** `QCMemModel(model, resume_j=12)` LoRA **enabled** (layers
  12..35). In deployment the k context chunks' depth-12 residuals `h12` are
  PRE-STORED offline, so the online context write is REPLACED by a persistent-store
  fetch + H2D of k·4 MiB; only sink+query are written online (bottom-12), then
  layers[12:36] resume over the pack.

## Design decisions (mandated protocol)
- **Reuse, don't modify:** the driver `scripts/eval_p0_20_equal_latency.py` is a
  thin composition — it imports `bench_p0_13_quality_latency`'s `_load` ordering,
  `_build_pack`, `_run_arm`, `_eos_ids`, `_summ`, `_sha256_file`,
  `_paired_bootstrap_ci`, `_mcnemar_exact`, provenance (`bench_p0_12_acceptance`),
  and `EXPECTED_LORA_SHA`/module-count/backbone-key-sha VERBATIM. The only new
  handle is `_load_with_peft` (mirrors `p013._load` but keeps the `peft_model` so
  we can toggle the adapter on/off between arms).
- **Latency口径:** same node/GPU/process TTFT = query encode/selection + store
  fetch + H2D + online write (sink+query) + model prefill/Read to first logits.
  "model-only Read" listed SEPARATELY. GPU-resident vs CPU-pinned store reported
  SEPARATELY (`_make_store_fetchers` reuses `bench_persistent_store_io`'s
  index_select / pinned-gather + non_blocking H2D pattern, k-parameterized).
  warmup 5 + ≥20 timed reps × ≥3 independent procs; median + p95. Numbers NEVER
  subtracted across harnesses/hardware. NVMe/network tiers = out of scope (stated,
  not fabricated).
- **Calibration freeze:** k_RAG*/k_CoMem* selected ONLY on a reserved calibration
  split (indices ≥ `--calib_offset` (900) ≥ `--limit` (100) → disjoint from
  quality, asserted fail-closed) by LATENCY alone, never after seeing quality. If
  no integer k lands in the ±5% band, the two bracketing points are reported and
  latency (only) is interpolated.
- **Anchors:** PRIMARY = fixed CoMem(k=12) budget → largest k_RAG* text-RAG fits;
  SECONDARY = fixed text-RAG(k=12) budget → largest k_CoMem* CoMem fits. Plus a
  reference k12-vs-k12 head-to-head.
- **Quality tasks:** primary = BABILong qa1/qa2 (4k,16k), LongEval (8k,16k), LoCoMo;
  secondary = RULER niah_multikey_1 (8k,16k). Same sample IDs/query/prefix/gen for
  both arms. Paired bootstrap 95% CI + exact McNemar + per-example JSONL.
- **Success criterion:** at PRIMARY anchor, CoMem quality ≥ latency-matched
  text-RAG AND ≥1 non-lexical benchmark stably wins (CI excludes 0) → POSITIVE. If
  the advantage only exists model-only and vanishes after residual fetch → limit
  to compute-side. If CoMem reads more but is still worse → bottleneck = cached-
  state readout, redirect to P0.17/P0.18/P1.10 (NOT packaged as a positive Pareto
  result). Reported honestly regardless of sign.
- **Fail-closed gates (STEP 0/1):** LoRA sha == dd09cd17… (168 modules, layers
  [12..35]) + backbone key-tensor shas; `disable_adapter()` structurally toggles
  the LoRA layers (active outside > 0, inside == 0); pack read_len equality between
  arms; calibration/quality split disjoint; finite logits. Any failure aborts
  (exit 3 manifest / exit 4 sanity).

Dense retriever = Stage B / P1.9 (uses the SAME BGE ordering as the running P1.9
job — reuse `retrieval_results/p1_9_dense`, do NOT re-embed), NOT this run.

## Modes
`manifest` | `sanity` | `calib_latency` (per (k,proc)) | `quality` (per
(benchmark,task,length,k[,shard])) | `aggregate` (CPU-only: freeze k*, frontier,
anchor stats, verdict).

## Files
- `scripts/eval_p0_20_equal_latency.py` (driver; new sibling — no existing harness
  touched)
- `scripts/_run_p0_20_8gpu.sh` (DRY-by-default flock 8-GPU task-pool; RUN=1 to
  execute; parallels `_run_p017_e2_8gpu.sh`)
- outputs → `bench_results/p0_20_eqlat/` (`quality/*.jsonl` per-example +
  `*_cell.json`, `calib_latency/calib_k*_proc*.json`, and aggregate
  `summary.json` / `frontier.json` / `anchors.json` / `decision.json`)

## Exact RUN=1 launch command (for MAIN, on FREE .104 diskB)
```bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
RUN=1 setsid nohup bash scripts/_run_p0_20_8gpu.sh \
  >/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/logs/p0_20_eqlat.out 2>&1 &
```
(.104 `.venv` is broken py3.14 → MUST use `torch-base`. Optional faster first pass:
prepend `QKS="2 6 12 16 20 24"` to run the frontier on fewer k, but then k* may
land on a k without a quality cell — full-sweep default is safest.)
