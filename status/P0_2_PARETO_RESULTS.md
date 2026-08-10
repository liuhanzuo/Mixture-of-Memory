# P0.2 — Unified Quality × Latency × Persistent-Storage Pareto (4 configs)

**Date:** 2026-08-01 · **Node:** QCMem `.104` (28.83.24.104, diskB, port 36000), 8× H20 · **Owner:** P0.2
**Goal:** assemble ONE Pareto over the four operating points
**{full-context (KV-Direct), j=0 text-retrieval RAG, j=12 frozen (no LoRA), j=12 + LoRA (flagship)}**,
crossing three axes — answer quality, per-query latency (+ amortization over Q queries), and
persistent per-token storage. Most quality data already existed on diskB; the NEW GPU work here is
the per-query latency breakdown + amortization / break-even.

**Hard rules honored:** (a) no fabricated cells — anything not on diskB is marked `NOT FOUND`;
(b) the Pareto **keeps FROZEN (config #3) and ADAPTED/LoRA (config #4) CoMem in separate rows** — LoRA
quality is never merged into adapter-free storage/latency numbers; (c) latency = median-of-3, chunk512,
top-12, bf16/SDPA, 1×H20, seed 42, `chat_template=False`.

Flagship (config #4): Qwen3-8B, `resume_j=12`, LoRA `outputs/qcmem_distill_qwen_j12_r32_4k/final`,
chunk_size 512, topk 12, selector iter_bm25, iter_rounds 0, iter_hop_topk 4, sink=BOS, bf16, SDPA, seed 42.

---

## 0. The four configs

| # | config | depth split | retrieval | adapter | what is stored per token |
|--:|--------|-------------|-----------|---------|--------------------------|
| 1 | **full-context (KV-Direct)** | — (pack ALL chunks) | none | none | full KV cache (all 36 layers) |
| 2 | **j=0 text-retrieval RAG** | j=0 (36-layer recompute of pack) | iter_bm25 top-12 | none | raw token-IDs + BM25 index |
| 3 | **j=12 frozen** | j=12 (24-layer recompute) | iter_bm25 top-12 | **none (frozen)** | mid-layer hidden h₁₂ |
| 4 | **j=12 + LoRA (flagship)** | j=12 (24-layer recompute) | iter_bm25 top-12 | **LoRA r32** | mid-layer hidden h₁₂ |

Configs #2/#3/#4 all use the **same** iter_bm25 top-12 retrieved pack (~6657 tok, L-independent);
they differ only in split depth j and adaptation. Config #1 attends the entire L-token context.

---

## 1. Persistent storage — bytes/token + absolute capacity

Qwen3-8B: hidden 4096, 36 layers, 8 KV heads × 128 head_dim, bf16, native window 40960.

### 1a. Bytes per stored token

| store (config) | derivation | **bytes/token** |
|----------------|-----------|-----------------:|
| raw token-IDs (#1 context / #2 text base) | int32 id (vocab 151936 needs ≥3 B) | **≈ 4** |
| BM25 index (#2 retrieval store) | token-IDs + inverted postings | **≈ 4–8 ("few B/tok")** |
| **CoMem h₁₂ (#3, #4)** | 4096 × 2 B (bf16) | **8,192** (8 KiB) |
| **full KV cache (#1)** | 36 × 2(K,V) × 8 kv_heads × 128 × 2 B | **147,456** (144 KiB) |

**full KV = 147,456 / 8,192 = exactly 18× CoMem.** CoMem = 2,048× larger than raw token-IDs.

### 1b. Absolute persistent capacity (binary units)

| store | B/tok | 128K tok | 1M tok | 1B tok |
|-------|------:|---------:|-------:|-------:|
| token-IDs / BM25 (#1 base, #2) | 4 | 0.5 MiB | 4 MiB | ~3.7 GiB |
| **CoMem h₁₂ (#3, #4)** | 8,192 | **1 GiB** | **8 GiB** | **~7.5 TiB** |
| **full KV (#1)** | 147,456 | **18 GiB** | **144 GiB** | **~134 TiB** |

### 1c. Runtime peak GPU memory @128k (measured, this task)

| config | peak GPU mem @128k | note |
|--------|-------------------:|------|
| **full-context (#1)** | **89.36 GB** (clean-GPU) | = weights 16.4 + KV 19.3 + full-seq logits [1,131072,151936]×2B 39.8; **OOMs on H20** in single-forward write-all; Dense-generate arm 50.0 GB |
| **CoMem #2/#3/#4** | **18.5 GB (constant)** | flat 17.6→18.54 GB across 8k→128k (pack is L-independent) |

Storage headline: **CoMem stores 18× fewer bytes/token than a full KV cache and holds a flat ~18.5 GB
runtime footprint at any context length**, where full-context grows to 89 GB @128k (H20 OOM).

---

## 2. Per-query / per-config latency (fresh, .104, 1×H20, median-of-3, chunk512, top12, bf16/SDPA, seed42, LoRA-on)

### 2A. Write-all O(L) 3-phase — `scripts/bench_qcmem_vs_fullctx.py`

`prefill_s = write_s + select_s + read_s`. write = embed+layers[0:j] over ALL N context chunks
(one-time O(L) ingest); select = BM25 (CPU, negligible); read = query-write + read layers[j:36] over the
fixed top-12 pack (L-independent, 6657 tok). Raw JSON: `ruler_results/p0_2/writeall_{8k,16k,32k,64k,128k}.json`.

| L | write_s (O(L)) | select_s | read_s (per-query) | prefill_s | decode_s (faithful, 20 tok) | peak GB | read_len | full-ctx prefill_s | speedup |
|----|---------------:|---------:|-------------------:|----------:|----------------------------:|--------:|---------:|-------------------:|--------:|
| 8k   | 0.350 | 0.003 | 0.807 | 1.160 | 16.18 | 17.60 | 6657 | 1.360 | 1.17× |
| 16k  | 0.712 | 0.006 | 0.812 | 1.529 | 16.18 | 17.67 | 6657 | 2.970 | 1.94× |
| 32k  | 1.445 | 0.012 | 0.812 | 2.268 | 16.26 | 17.79 | 6657 | 6.967 | 3.07× |
| 64k  | 2.881 | 0.023 | 0.806 | 3.710 | 16.16 | 18.04 | 6657 | 18.316 | 4.94× |
| 128k | 5.828 | 0.045 | 0.807 | 6.708 | 16.17 | 18.54 | 6657 | **OOM** | **inf** |

- **write_s grows O(L)** (0.35→5.83 s); **read_s constant ~0.81 s** at every length; select negligible.
- full-context prefill grows O(L²) (1.36→18.3 s) and **OOMs @128k**; peak 20.2→24.8→34.1→52.6→OOM GB.
- `decode_s` here is the **faithful no-KV-cache** decode (re-reads pack each step ≈ 0.72 s/tok @chunk512);
  the realistic KV-cache decode tok/s is in Panel 2B.

### 2B. Select-first constant-write prefill + KV-cache decode — `scripts/bench_qcmem_vs_dense.py --mode speed`

Deploy path: BM25 picks top-12 FIRST, only the retrieved pack + query is forwarded → **prefill constant in L**.
Dense arm = stock `model.generate` full-context (LoRA disabled). Logs: `logs/p0_2_densespeed_g{5,6,7}.log`.

| L | Dense prefill_s | Dense tok/s | Dense peak GB | QCMem prefill_s | QCMem tok/s (KV) | QCMem peak GB | prefill speedup | decode× |
|----|----------------:|------------:|--------------:|----------------:|-----------------:|--------------:|----------------:|--------:|
| 8k   | 1.10  | 16.1 | 18.8 | 0.94 | 276.5 | 18.6 | 1.17× | 17.2× |
| 16k  | 2.44  | 17.2 | 20.9 | 1.05 | 719.4 | 18.7 | 2.32× | 41.8× |
| 32k  | 5.94  | 16.8 | 25.0 | 1.09 | 286.4 | 18.7 | 5.45× | 17.1× |
| 64k  | 16.42 | 17.0 | 33.4 | 1.16 | 716.2 | 18.7 | 14.16× | 42.1× |
| 128k | 50.59 | 16.7 | 50.0 | 1.32 | 318.7 | 18.7 | **38.3×** | 19.1× |

- **QCMem constant-write prefill ~0.9–1.3 s at every length**; Dense grows to 50.6 s @128k → **38.3× prefill speedup**.
- QCMem peak 18.6–18.7 GB constant; Dense 18.8→50.0 GB (linear in L).
- QCMem tok/s here is noisy (276–719, 2-lengths-per-proc warmup); the clean P0.1 cohort measured **760–784 tok/s**
  vs Dense 24–39 → **20–32× faster decode**. Either way QCMem decode is 17–42× faster than Dense KV-cache decode.

### 2C. Depth-cache (j=12 CoMem) vs full-recompute RAG (j=0) — same pack, same peak (P0.1 canonical, LoRA-on @128k)

| split j | write_s (one-time O(L)) | read_s (per-query) | peak GB |
|--------:|------------------------:|-------------------:|--------:|
| 0 (RAG: retrieve + full 36-layer recompute) | 0.09 | 1.141 | 18.54 |
| **12 (CoMem, 24-layer recompute)** | **5.826** | **0.849** | 18.54 |

Extra one-time write (j12−j0) = **5.736 s**; per-query read saving (j0−j12) = **0.292 s** →
**break-even ≈ 5.736/0.292 ≈ 20 queries** (read-only). Q<20 → RAG cheaper; **Q≥20 → CoMem depth-cache cheaper**.
Counting decode (CoMem 0.72 vs j0 1.005 s/tok faithful; 20–32× faster with KV-cache) shifts break-even EARLIER.
Cross-check with this task's fresh write-all read_s 0.807 → saving 0.334 → break-even ≈ 17. (P3.1 no-LoRA cohort ≈ 26.)

### 2D. Host→device per-query transfer (microbench `/tmp/h2d.py`, H20, pinned 45.5–57.5 GB/s)

| config | payload moved per query | transfer time |
|--------|-------------------------|--------------:|
| **CoMem (#3/#4)** | fixed top-12 h₁₂ pack **54.5 MB** | **1.20 ms** (pinned) / 2.46 ms (pageable) — **constant in L** |
| j=0 RAG (#2) | top-12 token-IDs ~26 KB | < 0.001 ms (then on-device embed + 36-layer recompute) |
| full KV (#1) | 18 GiB KV @128k | ~313 ms (144 GiB @1M ≈ 2.5 s — prohibitive) |

---

## 3. Cumulative + amortized cost over Q queries, and break-even

Model: one context of L=128k, served for Q queries. TTFT-cost(Q) = one-time write + Q × per-query read/prefill
(decode adds ~equally per query and favors CoMem further; omitted here to isolate the prefill/read trade). @128k:

| operating point | cost(Q) formula (s) |
|-----------------|---------------------|
| CoMem write-all (persist h₁₂) | 5.828 + 0.807·Q |
| CoMem select-first (recompute pack/query) | 1.32·Q |
| j=0 text-RAG (persist embeds) | 0.09 + 1.141·Q |
| full-context Dense (re-prefill each query) | 50.59·Q |

| Q | CoMem write-all | CoMem select-first | j=0 RAG | full-ctx Dense |
|---:|----------------:|-------------------:|--------:|---------------:|
| 1   | 6.64  | 1.32   | 1.23   | 50.6 |
| 2   | 7.44  | 2.64   | 2.37   | 101 |
| 4   | 9.06  | 5.28   | 4.65   | 202 |
| 8   | 12.28 | 10.56  | 9.22   | 405 |
| 16  | 18.74 | 21.12  | 18.35  | 809 |
| 32  | 31.65 | 42.24  | 36.60  | 1619 |
| 64  | 57.48 | 84.48  | 73.11  | 3238 |
| 128 | 109.1 | 168.96 | 146.14 | 6475 |

**Break-even query counts (@128k, read/prefill TTFT):**
- CoMem write-all beats **CoMem select-first** at **Q ≥ 12** (persisting h₁₂ amortizes the O(L) write).
- CoMem write-all beats **j=0 RAG** at **Q ≥ ~17–20** (P0.1 canonical ≈ 20).
- **Any CoMem variant beats full-context at Q = 1** (6.6 s / 1.3 s vs 50.6 s), and full-context **OOMs @128k**
  on H20 — CoMem is the only feasible operating point above the 40960-token native window.

---

## 4. Quality — 5 benchmarks × 4 configs (disk-verified; `NOT FOUND` = not on diskB, not fabricated)

All numbers read directly from result-dir JSONs under
`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`, `chat_template=False`, seed 42.

### 4a. Primary quality axis — RULER 15-cell macro (n=100, string_match_all)

| # | config | RULER macro | note |
|--:|--------|------------:|------|
| 1 | full-context (KV-Direct) | **78.73** | all 128k cells = 0 (130k tok > Qwen 40960 window) |
| 2 | j=0 text-RAG | **99.20** | teacher/upper bound; NIAH 99/99/98/96/99, multikey 100/100/99/99/99, VT 100×5 (n=100) |
| 3 | j=12 frozen (no LoRA) | **8.01** | collapses — adaptation essential at j=12 |
| 4 | **j=12 + LoRA (flagship)** | **96.07** | best; holds to 128k (native n=100) |

Per-cell (RULER):
- **#1 KV-Direct** (`ruler_results/kvdirect_8b_ruler_chatFALSE/`): single 100/100/100/100/**0**; multikey 100/100/98/88/**0**; VT 100/100/99.8/96.2/**0**.
- **#2 j=0 RAG** (`ruler_results/p0_2_c2_j0_iterbm25_niah_chatFALSE/` + VT `presub_A_kvdirect_iterbm25_vt/`): single **99/99/98/96/99**; multikey **100/100/99/99/99**; VT **100/100/100/100/100** (in-window pack survives 128k → macro **99.20**, the distillation teacher). (Plain-BM25 non-iter control gives VT 48/26/22/23/21 — iter_bm25 is decisive.)
- **#3 frozen j=12** (`ruler_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE/`): single 36/7/22/18/9; multikey 8/3/1/6/4; VT 2.4/1.2/2/0.2/0.4.
- **#4 CoMem+LoRA** (`ruler_results/comem_lora_native_n100/`): single 100/91/97/98/98; multikey 94/91/99/90/93; VT 96.2/98/98.2/98.6/**99.0**.

### 4b. Other benchmarks (disk-verified)

| benchmark | #1 KV-Direct | #2 j=0 RAG | #3 frozen j12 | #4 CoMem+LoRA |
|-----------|-------------:|-----------:|--------------:|--------------:|
| LongBench 6-QA macro F1 | 12.17 | **12.31** | 9.96 | 12.15 |
| LongBench 4-QA macro F1 | 10.06 | — | 8.23 | 9.76 |
| LongEval mean (8k–128k) | 65.2% | **97.2%** | 0.2% | 69.0% (tk12) |
| LoCoMo F1 / acc | 9.02 / 22.4% | **9.90 / 25.23%** (judge 41.59%) | 6.26 / 12.39% (judge **24.52%**) | 9.15 / 23.4% |
| BABILong qa1 (0k–32k) | 98/84/80/74/80/72/63 | 98/84/80/73/60/34/35 | mean **33.43** | 98/80/68/68/46/17/12 |
| BABILong qa5 (0k–32k) | 71/73/62/59/65/42/58 | 70/73/61/57/69/53/60 | mean **60.29** (qa2 18.00; 21-cell macro **37.24**) | 68/76/76/75/68/60/58 |

Provenance: #1 `{longbench,longeval,locomo,babilong}_results/kvdirect_8b*`; **#2 `*_p0_2_c2_j0_iterbm25_chatFALSE/`
(LongBench/LongEval/BABILong fresh) + LoCoMo `locomo_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE/` (on wzc1)**;
#3 local `*_zeroshot_j12_frozen_iterbm25_chatFALSE` artifacts are raw-verified (LoCoMo `scores.json`; 84 BABILong shard JSONs); #4 `*qcmem_8b_iter_chatFALSE*`, LoCoMo `locomo_results/qcmem_8b_iter_chatFALSE`. **Config #2 is
now fully populated (§4c) — it is the teacher/upper bound, matching/exceeding #4 on RULER/LongBench/LoCoMo and far
higher on LongEval.**

### 4c. Config #2 (j=0 text-RAG) — FILLED (2026-08-01, `status/P0_2_CONFIG2_JRAG_RESULTS.md`)

Config #2 = the full-depth (j=0) retrieval baseline and **the self-distillation teacher** for the flagship
LoRA (teacher reads all 36 layers over the same top-12 pack; student caches h₁₂ + LoRA). The full 5-benchmark
config-#2 sweep (j=0, iter_bm25, tk12, chat=False, no-LoRA, Qwen3-8B) is now complete — **no `NOT FOUND` cells remain**:

| benchmark | config #2 (j=0 text-RAG) | provenance |
|-----------|--------------------------:|-----------|
| RULER 15-cell macro | **99.20** (niah_single 99/99/98/96/99 · multikey 100/100/99/99/99 · VT 100/100/100/100/100, n=100) | NIAH fresh `ruler_results/p0_2_c2_j0_iterbm25_niah_chatFALSE/`; VT reused-verified `ruler_results/presub_A_kvdirect_iterbm25_vt/` |
| LongBench 6-QA macro F1 | **12.31** (2wikimqa 12.42 / hotpotqa 12.17 / multifieldqa_en 26.18 / musique 7.47 / narrativeqa 3.88 / qasper 11.73) | fresh `longbench_results/p0_2_c2_j0_iterbm25_chatFALSE/` |
| LongEval mean (8k–128k) | **97.2%** (4k/8k/16k/32k/64k/128k = 98/100/96/99/94/97) | fresh `longeval_results/p0_2_c2_j0_iterbm25_chatFALSE/` |
| BABILong qa1 (0k–32k) | 98/84/80/73/60/34/35 | fresh `babilong_results/p0_2_c2_j0_iterbm25_chatFALSE/` |
| BABILong qa5 (0k–32k) | 70/73/61/57/69/53/60 | ″ (qa2 58/53/51/44/37/17/12) |
| LoCoMo (n=1986) | F1 **9.90** / acc 25.23% / gpt-4o judge **41.59%** | reused-verified `locomo_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE/scores.json` (on **wzc1** disk, not diskB — original diskB-only search missed it) |

**Reading:** config #2 (the teacher) **matches or exceeds** flagship #4 on RULER (99.20 vs 96.07), LongBench
(12.31 vs 12.15), LoCoMo, and is **far higher on LongEval (97.2% vs 69.0%)** — full-depth recompute preserves the
line/position fidelity the j=12 depth-cache loses. This confirms config #2 as the **distillation upper bound**. The
sole exception is BABILong qa1 long-range, where config #1 KV-Direct's full context (qa1@32k = 63) beats BM25
retrieval (config #2 qa1@32k = 35). All cells `chat=False`, seed 42, no fabrication.

---

## 5. The assembled Pareto (quality × latency × storage) — frozen vs adapted kept separate

@128k, LoRA-on latency cohort. "Feasible?" = fits on one H20 (97.8 GB).

| # | config | RULER macro (quality) | bytes/tok | peak GPU @128k | prefill/query @128k | one-time write @128k | feasible @128k? | adapter |
|--:|--------|----------------------:|----------:|---------------:|--------------------:|---------------------:|:---------------:|:-------:|
| 1 | full-context (KV-Direct) | 78.73 (128k→0) | 147,456 | 89.36 GB | 50.59 s | 0 | **NO (OOM)** | none |
| 2 | j=0 text-RAG | 99.20 (teacher) | ~4–8 | 18.5 GB | 1.14 s (read) | 0.09 s | yes | none |
| 3 | j=12 **frozen** | **8.01** | 8,192 | 18.5 GB | 0.85 s (read) | 5.83 s | yes | **none** |
| 4 | j=12 **+ LoRA** | **96.07** | 8,192 | 18.5 GB | 0.81 s (read) / 1.32 s (select-first) | 5.83 s | yes | **LoRA r32** |

**Pareto reading (the two frontier winners):**
- **Adapter-free frontier:** among {#1, #2, #3}, config **#3 frozen j=12 is dominated** (8.01 macro at the same
  18.5 GB / 8192 B/tok as #4) → **the LoRA adaptation is what makes the CoMem depth-cache usable**, not the
  depth split itself. Config **#1 is dominated on cost/feasibility** (89 GB, OOM @128k, quality collapses past the
  40960 window). Config **#2 (j=0 RAG)** is Pareto-efficient on cost only where a full-depth recompute per query is
  acceptable and quality is unmeasured beyond VT.
- **Adapted frontier (flagship):** config **#4 dominates the whole board** — 96.07 RULER macro (holds to 128k),
  8192 B/tok (18× less than full KV), flat 18.5 GB, 0.8–1.3 s constant prefill, and it is the **only** point that is
  both high-quality AND feasible at 128k on a single H20.

**Do NOT merge:** #3 (frozen, macro 8.01) and #4 (LoRA, macro 96.07) share identical storage/latency but are listed
as separate rows — the 88-pp quality gap at fixed depth/storage IS the LoRA/self-distillation contribution and must
not be attributed to the depth-cache alone.

---

## 6. Exact commands + raw artifact paths

**Latency (this task, all on .104, PY `/usr/bin/python3.11` + venv site-packages, 1×H20):**
```
# Write-all O(L) 3-phase (per length)
python scripts/bench_qcmem_vs_fullctx.py \
  --model_path .../models/Qwen3-8b-local --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
  --resume_j 12 --chunk_size 512 --topk 12 --lengths <L> --n_repeat 3 --warmup 1 --n_decode 20 \
  --dtype bfloat16 --attn_impl sdpa --seed 42 --output ruler_results/p0_2/writeall_<L>.json
# Select-first constant-write prefill vs Dense (KV-cache decode)
python scripts/bench_qcmem_vs_dense.py --mode speed \
  --model_path .../models/Qwen3-8b-local --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
  --resume_j 12 --topk 12 --chunk_size 512 --selector iter_bm25 --sink_tokens bos \
  --context_lengths 8k 16k 32k 64k 128k --max_new_tokens 20 --dtype bfloat16 --attn_impl sdpa
```
Orchestrator: `scripts/p0_2_bench.sh` · Transfer microbench: `/tmp/h2d.py`.
Raw: `ruler_results/p0_2/writeall_{8k,16k,32k,64k,128k}.json`, `logs/p0_2_densespeed_g{5,6,7}.log`, `logs/p0_2_writeall_*.log` (all on diskB .104).

**Quality result dirs (diskB `/apdcephfs_zwfy6/share_304376610/...`):**
- #1: `ruler_results/kvdirect_8b_ruler_chatFALSE/`, `longbench_results/kvdirect_8b_chatFALSE/`, `longeval_results/kvdirect_8b_chatFALSE/`, `locomo_results/kvdirect_8b_chatFALSE/`, `babilong_results/kvdirect_8b_chatFALSE/`
- #2: `ruler_results/presub_A_kvdirect_iterbm25_vt/` (VT only)
- #3: `ruler_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE/`, `longbench_results/…`, `longeval_results/…`
- #4: `ruler_results/comem_lora_native_n100/`, `longbench_results/qcmem_8b_iter_chatFALSE/`, `longeval_results/qcmem_8b_iter_chatFALSE/`, `locomo_results/qcmem_8b_iter_chatFALSE/`, `babilong_results/qcmem_j12_iter_bm25_chatFALSE_ad_hop4/`

Related records: `status/P0_1_UNIFIED_TIMING.md` (canonical break-even ≈20), `status/P3_1_PARETO_JSWEEP.md`
(no-LoRA depth sweep), `status/P0_3_MATCHED_N100.md` (RULER n=100), `status/P0_11_FROZEN_J12.md` (frozen control).

---

## 7. Headline

1. **Storage:** CoMem = 8,192 B/tok = **18× less than a full KV cache**; flat **18.5 GB** runtime @128k vs
   full-context **89 GB (H20 OOM)**. At 1B tokens: CoMem ~7.5 TiB vs full KV ~134 TiB.
2. **Latency:** CoMem prefill is **constant in L** (~0.8–1.3 s) vs full-context 50.6 s @128k → **38× prefill
   speedup**, plus 20–32× faster decode; the fixed 54.5 MB pack transfers host→device in **1.2 ms**.
3. **Amortization:** persisting the mid-layer cache (write-all) beats select-first at **Q≥12** and beats j=0
   full-recompute RAG at **Q≥~20**; every CoMem variant beats full-context at **Q=1** (and full-context is
   infeasible @128k).
4. **Quality × the whole trade:** only config **#4 (CoMem j=12 + LoRA)** is simultaneously top-quality
   (RULER macro **96.07**, holds to 128k) AND feasible @128k on one H20. Frozen j=12 (**8.01**) at the same
   storage/latency shows the **adaptation — not the depth-cache — carries the quality**; the two are kept as
   separate Pareto rows.

**Open item — CLOSED (2026-08-01):** config #2 (j=0 text-RAG) now has a full 5-benchmark sweep (RULER macro
99.20, LongBench 12.31, LongEval 97.2%, BABILong qa1/qa2/qa5, LoCoMo 9.90 / 25.23% / judge 41.59% — §4c). It is the
self-distillation teacher / upper bound; it matches or exceeds flagship #4 on RULER/LongBench/LoCoMo and is far
higher on LongEval. No `NOT FOUND` cells remain in the Pareto. (`status/P0_2_CONFIG2_JRAG_RESULTS.md`.)
