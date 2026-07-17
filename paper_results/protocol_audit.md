# Protocol Audit — QCMem/CoMem

The single biggest risk to the paper is that `QCMEM_BENCHMARK_PLAN.md` §1/§1a present numbers **stitched from runs that used different `j`, different selectors, and different `n`**. This file enumerates the distinct protocols actually found on disk (from each run's own config, not its dir name) and assigns every `results.csv` row a `protocol_group`.

## A. Protocol groups (auto-assigned from config in `build_results.py`)

| protocol_group | definition (from config) | intended use |
|---|---|---|
| `qcmem_zs_recalloptimal_n*` | zero-shot, j/L ≤ 0.15 (shallow j2/j3/j4) | recall **upper bound**; mk/vt strongest here |
| `qcmem_zs_readoutsafe_n*` | zero-shot, 0.15 < j/L ≤ 0.35 (~0.25–0.33L) | the "readout-safe" zero-shot main-table row |
| `qcmem_zs_deepj_n*` | zero-shot, j/L ≥ 0.40 (content-depth probes) | mostly **collapse probes** — near-0 single; NOT reportable as results |
| `qcmem_adapter033L_n*` | +adapter, j/L < 0.40 | hard-task **sweet spot** (8B j12, 14B j13) |
| `qcmem_contentj_n*` | +adapter, j/L ≥ 0.40 (0.45L semantic peak) | single recovers to ~100 but mk/vt weaker than 0.33L |
| `baseline_kvdirect` / `_hcache` / `_dense` / `_unknown` | resume_j=0 / mid-layer no-retrieval / full-ctx | H2H baselines |
| `*_n500` vs `*_n100` vs `*_n50` | by `n_valid` bucket | sample-size cohorts (do NOT mix in one table) |

The `n` suffix is derived from **actual valid sample count**, not the planned n.

## B. The ten protocol axes the plan mixes (task §three)

1. **n = 50 vs 100 vs 500 (vs 300/600/1000/1500/1800/2000/2500/3000)** — RULER/BABILong appear at *all* of these. The plan's fixed protocol says n=500 for RULER/BABILong, but the headline main table (§1) is n=100, and vs-Dense/selector tables are n=30–50. `results.csv` shows RULER n_valid values of 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1500, 1800, 2000, 2500, 3000 — the larger numbers are **sums across duplicated runs of the same (model,j,task,length)**, i.e. the same cell was run many times and the ledger picked one.
2. **zero-shot readout-safe j** (per-model: 0.6B j2 / 1.7B j3 / 4B j9 / 8B j9 / 14B j13 / 32B j27).
3. **content-j adapter** (~0.45L: 0.6B/1.7B j13, 4B/8B j16, 14B j18).
4. **legacy fixed ~0.33L** (8B j12, 32B j21).
5. **recall-optimal shallow j3/j9** (the original main-table basis; mk/vt strongest).
6. **selector**: bm25 / iter_bm25 / iter_bm25_adaptive / reader_attn / oracle / recency — see §C.
7. **adapter checkpoint**: only 8B j12 is on wzc1; 14B/content-j adapters unverified.
8. **scorer/metric**: RULER `string_match`; BABILong `TASK_LABELS`+`compare_answers`; LongBench `qa_f1`; LoCoMo `token_f1` vs `substring_acc` (mixed — see anomalies A11).
9. **8B single-point speed** (`resume_j=12`, `bench_qcmem_vs_dense_result.txt`) vs later all-scale sweep — different dates, same read-only definition.
10. **LongBench/LoCoMo full test set** — confirmed full (LongBench n=200/dataset; LoCoMo n=1986) EXCEPT 0.6B/1.7B/4B LongBench are **single-shard `_0`** partial merges (registry note L1679).

## C. Selector is NOT consistent within the "main table"

`build_results.py` flags every RULER qcmem cell whose selector ≠ the protocol selector for its task family (single/mkey → `bm25`, vt → `iter_bm25`). Excluded counts:

- 52 cells: **vt run with `bm25`** (single-pass) instead of iter_bm25 — vt bm25 ≈ 23–48 vs iter_bm25 ≈ 97.
- 27 + 25 cells: **`reader_attn`** used where bm25/iter_bm25 expected.
- 17 + 9 cells: **`oracle`** (cheating upper bound) where bm25/iter_bm25 expected.
- 14 + 9 cells: **`recency`** (StreamingLLM-style).
- 8 cells: **iter_bm25 where bm25 expected** (single/mkey run with iter).
- 6 cells: **iter_bm25_adaptive** (ρ=0.3 early-stop, project-flagged as broken).

**Consequence:** the RUN_REGISTRY-designated "canonical n=500" dirs are selector-inconsistent — e.g. `qcmem_8b_zeroshot_j9_n500` is an **oracle, vt-only** run; `qcmem_8b_n500` (adapter) uses **reader_attn** (→ single 86/81/81, mk 80/72/60, vt 55/44/41, far below the headline 100/100/100, 91/91/92, 97/97/98). The headline 8B numbers actually come from **n=100 bm25/iter_bm25** runs, not the "n=500" dirs. See anomalies A02.

## D. Recommended reporting protocol (one coherent slice)

For a fair paper table, fix per row: **one model, one adapter state, one j, bm25 for single/mkey + iter_bm25 for vt, n≥100, a single run_id per cell.** `usable_now.csv` is exactly this cohort (588 cells). Everything else → `excluded_results.csv` with a reason.
