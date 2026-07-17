# Repository Data Inventory — QCMem/CoMem benchmark audit

Generated read-only on git commit `9c258a7` (2026-07-17). No GPU / training / eval / download / API was used. All figures below come from `find`/JSON parsing of existing artifacts on two shared filesystems:

- **wzc1** (local, canonical): `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory`
- **diskB** (read over SSH, node 28.85.35.73): `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory`

> ⚠️ There is **no file literally named `benchmark.md`** in the repo. The "benchmark doc" the task refers to is the trio `status/QCMEM_BENCHMARK_PLAN.md` (primary), `status/RUN_REGISTRY.md` (191 KB ledger), `status/BENCHMARK_RESULTS.md`. `benchmark_clean.md` (this audit's clean successor) is written fresh and does **not** overwrite any of them.

## 1. Run-directory counts

| benchmark dir | wzc1 dirs | diskB dirs | with config/json | with predictions | with `scores.json` |
|---|---:|---:|---:|---:|---:|
| `ruler_results/`     |  99 | 227 | 93/99 | 84/99 | 0 (RULER stores score in per-cell json, not scores.json) |
| `babilong_results/`  | 351 | 386 | 327/351 | 337/351 | **0 (no score persisted anywhere)** |
| `longbench_results/` |  19 |  30 | 18/19 | 13/19 | 12/19 |
| `locomo_results/`    |  13 |  10 | 13/13 | 11/13 | 6/13 (**diskB: 0/10 have scores.json**) |
| `longeval_results/`  |  41 |  34 | 41/41 | 0/41 | 0/41 (longeval stores summary json only) |

Total distinct run directories ≈ **1,210** across both disks (8B runs are largely duplicated on both; the 0.6B/1.7B/4B/14B/32B/30B-A3B scale sweep lives **only on diskB**).

## 2. Atomic cells extracted → `results.csv` (9,783 rows)

| benchmark | cells | provenance A | B | C | U | complete | needs_rescore | unverifiable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RULER    | 1,245 | 1,100 | 0 | 22 | 9 | 1,214 | 0 | 31 |
| BABILong | 8,311 | 0 | 8,311 | 0 | 0 | 0 | **8,311** | 0 |
| LongBench|   192 | ~180 | 0 | ~12 | 0 | 192 | 0 | 0 |
| LoCoMo   |    35 | ~25 | 0 | ~10 | 0 | ~25 | 0 | ~10 |
| **total**| **9,783** | **1,100** | **8,652** | **22** | **9** | **1,441** | **8,311** | **31** |

Provenance levels: **A** = config + predictions + machine score all present; **B** = config + predictions but **no machine score on disk** (all BABILong); **C** = summary/manual only; **U** = source unverifiable.

## 3. What is machine-verifiable, by model

Machine-scored QCMem cells (RULER per-cell json score, LongBench/LoCoMo `scores.json`) exist for:

| model | RULER | LongBench | LoCoMo | BABILong | primary disk |
|---|---|---|---|---|---|
| **Qwen3-8B** | ✅ rich (j∈{0,2,3,4,6,9,10,12,13,14,16,18,24,36}) | ✅ (adapter + stock) | ✅ **only model with scores.json** | ⚠️ preds only, no score | wzc1 + diskB |
| Qwen3-0.6B | ✅ (zs j2/3/5/7/9/11/13; ad j9/j13) | ✅ | ⚠️ preds only | ⚠️ preds only | diskB |
| Qwen3-1.7B | ✅ (zs; ad j9/j13) | ✅ | ⚠️ preds only | ⚠️ preds only | diskB |
| Qwen3-4B | ✅ (zs j3/6/9/12; ad j12/j16) | ✅ | ⚠️ preds only | ⚠️ preds only | diskB |
| Qwen3-14B | ✅ (zs j13; ad j13/j18) | ✅ | ⚠️ preds only | ⚠️ preds only | diskB |
| Qwen3-32B | ✅ (zs j3/12/13/16/21/24/27) | ✅ | ⚠️ preds only | ⚠️ preds only | diskB |
| Qwen3-30B-A3B | ✅ (zs j12/j20) | ✅ | ⚠️ preds only | ⚠️ preds only | diskB |
| Llama-3-8B | ✅ (legacy, j12) | legacy | — | ⚠️ preds only | wzc1 |
| Hunyuan-A13B | ✅ (j32) | — | — | — | wzc1 |

**Headline conclusion:** every scale except Qwen3-8B is machine-verifiable **only on diskB**; on the canonical wzc1 disk, only Qwen3-8B (+ legacy Llama/Hunyuan) is present. **LongBench** numbers cross-check cleanly against the manual ledger (provenance A). **LoCoMo** is machine-scored only for 8B; scale-model LoCoMo numbers in the ledger are hand-recorded (diskB has predictions but 0 scores.json). **BABILong has NO score persisted for any model/method** — every BABILong number in the paper docs is manual and requires a rescore of existing predictions.

## 4. Adapter checkpoints (verified by presence + hash)

Only these LoRA adapters exist on wzc1 `outputs/*/final`:

| adapter dir | model / j | on wzc1 | notes |
|---|---|---|---|
| `qcmem_distill_qwen_j12_r32_4k/final` | Qwen3-8B j12 | ✅ | the ONE fully verifiable adapter (hash in results.csv) |
| `qcmem_distill_qwen_j12/final` | Qwen3-8B j12 (older) | ✅ | earlier variant |
| `qcmem_distill_qwen_j9b0_pg19_nctx7/final` | Qwen3-8B j9 | ✅ | |
| `qcmem_distill_llama3_j12_r32_4k/final` | Llama-3-8B j12 | ✅ | legacy |
| `qcmem_distill_hy3_j32_r32/final` | Hunyuan j32 | ✅ | |

**NOT on wzc1** (cited by main table, presumably on diskB — unverified in this pass): `qcmem_distill_14b_j13_r32`, and all content-j adapters (0.6B/1.7B j13, 4B/8B j16, 14B j18). RULER cells for those adapters exist on diskB (so the adapters were used), but the checkpoint files themselves were not hashed here.

## 5. Files that are NOT machine results (excluded from results.csv)

- `longeval_results/*` — summary-json only, no per-sample predictions parsed (LongEval numbers 92/71/74/65 etc. are provenance-C ledger entries).
- `status/bench_qcmem_vs_dense_result.txt` — the ONLY source of speed + vs-Dense-accuracy numbers; it is a **log file**, provenance C, not a result dir.
- The non-QCMem legacy directions (mem_space, funnel, beacon, L2/L3, DMS, RMT, landmark) populate ~341 "unknown-method" RULER cells and ~2,900 Llama BABILong cells — out of scope for the QCMem paper.
