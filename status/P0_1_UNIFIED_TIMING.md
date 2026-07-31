# P0.1 — Unified Same-Harness Timing Table

**Created**: 2026-07-31
**Node**: 28.85.35.73 (`.73`, 8× H20 97.8 GiB, diskB), 1×H20 per bench, `/opt/conda`→`/usr/bin/python3.11`
**Config**: flagship Qwen3-8B, `resume_j=12` **+ LoRA** (`outputs/qcmem_distill_qwen_j12_r32_4k/final`), topk12, chunk512, bf16, SDPA, median-of-3 (1 warmup), lengths 8k/32k/128k.
**Scripts** (single model load each): `scripts/bench_qcmem_vs_fullctx.py` (write-all O(L)) + `scripts/bench_qcmem_vs_dense.py --mode speed` (select-first constant-write). Drivers `scripts/_p01_unified_timing_73.sh`, raw JSON `ruler_results/p01_unified_fullctx_seed42.json`, logs `logs/p01_unified_timing_73.log`.

> Purpose: reconcile the two write conventions across the two bench scripts in ONE table on ONE model load, and recompute the depth-cache vs RAG break-even from fresh numbers. This is the LoRA-on cohort (matches the paper's same-platform LoRA-on efficiency control), so numbers are directly comparable to `tab_eff_lora` / abstract's `2.74×`, `18.54 vs 89.39 GB`.

---

## A. `bench_qcmem_vs_fullctx.py` — write-all O(L) decomposition (per length)

`prefill_s = write_s + select_s + read_s`; write = embed+layers[0:12] over ALL N context chunks (one-time O(L) ingest); select = bm25 (CPU); read = query-write + read layers[12:] over the fixed pack. Read pack is L-independent (6657 tok).

| Length | write_s (O(L) ingest) | select_s | read_s (per-query) | prefill_s (sum) | read_len | peak GB | write_calls | full-ctx prefill_s | speedup |
|--------|----------------------:|---------:|-------------------:|----------------:|---------:|--------:|------------:|-------------------:|--------:|
| 8k   | 0.355 | 0.0035 | 0.849 | 1.208 | 6657 | 17.60 | 17  | 1.444 | 1.20× |
| 32k  | 1.444 | 0.0139 | 0.848 | 2.306 | 6657 | 17.79 | 65  | 8.277 | 3.59× |
| 128k | 5.826 | 0.0521 | 0.849 | 6.727 | 6657 | 18.54 | 257 | **OOM** | inf (full OOM) |

- **write_s grows O(L)** (0.355 → 1.444 → 5.826 s across 8k→32k→128k); **read_s constant ~0.849 s** at every length (fixed 6657-tok pack); **select negligible** (bm25 CPU).
- **Peak GB near-constant** (17.6 → 18.54 GB); full-context OOMs at 128k on H20 in this single-forward script (matches the paper's Dense-OOM claim).
- NOTE: this script's `decode_s` is the faithful **no-KV-cache** decode (each step re-reads the pack) → ~17 s/20 tok; the realistic KV-cache decode tok/s is in Panel B.

## B. `bench_qcmem_vs_dense.py --mode speed` — constant-write prefill vs Dense (per length)

Select-first deploy path: bm25 picks top-12 FIRST, only the retrieved pack + query is forwarded → prefill constant in L. Dense = stock `model.generate` full-context (LoRA disabled for the Dense arm). KV-cache decode.

| Length | Dense prefill_s | Dense tok/s | Dense peak GB | QCMem prefill_s | QCMem tok/s | QCMem peak GB | read_len | prefill speedup | decode× |
|--------|----------------:|------------:|--------------:|----------------:|------------:|--------------:|---------:|----------------:|--------:|
| 8k   | 1.20  | 37.6 | 18.8 | 1.05 | 760.2 | 18.7 | 6657 | 1.14× | 20.2× |
| 32k  | 7.33  | 38.9 | 25.0 | 1.06 | 776.0 | 18.7 | 6657 | 6.92× | 20.0× |
| 128k | 71.37 | 24.4 | 50.0 | 1.10 | 783.6 | 18.7 | 6657 | **64.9×** | 32.1× |

- **QCMem constant-write prefill ~1.05–1.10 s at every length**; Dense prefill grows to 71.37 s @128k → **64.9× prefill speedup @128k** (LoRA-on).
- **QCMem peak 18.7 GB constant**; Dense 18.8 → 25.0 → 50.0 GB (linear in L).
- **decode 20–32× faster** (KV-cache resumed-band decode: 760–784 tok/s vs Dense 24–39 tok/s).

## C. Break-even: depth-cache (j=12) vs RAG (j=0), fresh @128k, LoRA-on

From `bench_qcmem_vs_fullctx.py` write-all decomposition, both **LoRA-on**, 128k, same harness/node:

| Split j | write_s (one-time O(L)) | read_s (per-query) | peak GB |
|--------:|------------------------:|-------------------:|--------:|
| 0 (RAG: retrieve + full recompute) | 0.09 | 1.141 | 18.54 |
| **12 (CoMem)** | **5.826** | **0.849** | 18.54 |

- extra one-time write (j12 − j0) = 5.826 − 0.09 = **5.736 s**
- per-query read saving (j0 − j12) = 1.141 − 0.849 = **0.292 s**
- **break-even Q = 5.736 / 0.292 ≈ 19.6 → ~20 queries** (read-only accounting).
- Interpretation: **Q<20 → j=0 (RAG) cheaper; Q≥20 → CoMem depth-cache cheaper** (before counting decode savings, which shift it earlier).

---

## Reconciliation with `paperA/benchmark.md` §1c / P0.1 TODOList

| Quantity | Paper value | Fresh here | Match? |
|---|---|---|---|
| QCMem read pack | 6657 tok | 6657 tok | ✅ exact |
| 128k QCMem peak (LoRA-on) | 18.54 GB | 18.54 GB | ✅ exact |
| 128k write-all write_s (j12) | 7.79 s (zero-shot) | 5.826 s (LoRA-on) | ⚠ cohort differs (see below) |
| 128k read_s (j12) | 0.722 s (zero-shot) | 0.849 s (LoRA-on) | ⚠ cohort differs |
| constant-write prefill @128k | 1.917 s (zero-shot j12) / 1.92 s (§1c) | 1.10 s (LoRA-on) | same regime (constant, L-indep) |
| break-even | ≈26 queries (zero-shot fullctx) | ≈20 queries (LoRA-on fullctx) | same order, cohort/node-driven |

**Discrepancy flag (not clearly wrong → no paper edit made):** the paper's write-all sweep (7.79 s write, 0.722 s read, break-even 26) is the **zero-shot / no-LoRA** cohort measured in an earlier run; this fresh run is the **LoRA-on flagship** cohort on `.73`. The break-even is set by the ratio (extra write)/(read saving), which both cohorts place in the ~20–26 range. The 8k/32k write_s (0.355/1.444) and constant read_s (~0.849) reproduce the O(L)-write + L-independent-read structure exactly. The select-first constant-write prefill (~1.10 s) reproduces the §1c "constant ~1.9 s" regime (faster here: fewer decode-warmup artifacts, H20, LoRA path). No paper table edited — differences are cohort (LoRA vs zero-shot) + node, not a data error.
