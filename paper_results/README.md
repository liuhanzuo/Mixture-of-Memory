# paper_results/ — QCMem/CoMem benchmark audit

Read-only audit of all existing QCMem/CoMem benchmark data, produced to make the numbers safe for paper writing. **No model / GPU / training / eval / download / API was used.** No original result file was moved, deleted, or overwritten. Machine numbers come from parsing existing artifacts; the ledgers (`status/QCMEM_BENCHMARK_PLAN.md`, `status/RUN_REGISTRY.md`) were read but not modified.

Generated on git commit `9c258a7`, 2026-07-17.

## Files

| file | what it is |
|---|---|
| `repository_inventory.md` | counts of run dirs / configs / predictions / scorer-outputs across wzc1 + diskB; what is machine-verifiable per model |
| `results.csv` | **9,783 atomic cells** (one per model×method×adapter×j×benchmark×task×length), with provenance_level, protocol_group, status, config/prediction/scorer paths, source_disk |
| `usable_now.csv` | 588 cells safe to use now (RULER qcmem with protocol selector + n≥100; LongBench/LoCoMo machine-scored) |
| `excluded_results.csv` | 9,195 excluded cells, each with `exclude_reason` (needs_rescore / scorer_mismatch / n_mismatch / baseline / no_machine_score) |
| `manual_claims.csv` | headline numbers that live only in the ledger (provenance C), each cross-checked against machine data where possible |
| `protocol_audit.md` | the distinct protocols on disk (n, j, selector, adapter, scorer) and how the main table mixes them |
| `anomalies.md` | 16 anomalies with evidence paths, cause, paper impact, no-rerun handling, must-rerun? |
| `paper_tables.md` | candidate tables, each cell cited to one run_id; empty frames where no coherent data exists |
| `baseline_completeness.csv` | which baselines (dense/kvdirect/hcache/streamingllm/memoryllm) are complete/partial/missing per benchmark |
| `minimal_rerun_plan.md` | P0–P3 rerun priorities; the no-GPU fixes that resolve most gaps |
| `build_results.py` | the reproducible aggregator (below) |
| `_warnings.log`, `_warnings_diskB.log` | non-fatal parse warnings (never silent) |
| `_results_diskB_raw.csv` | diskB-parsed CSV (input to the merge; the scale sweep lives on diskB) |

## Reproduce

```bash
# 1. parse local (wzc1) results
python3 paper_results/build_results.py

# 2. (already done) parse the diskB mirror that holds the 0.6B-32B scale sweep,
#    pull its CSV to paper_results/_results_diskB_raw.csv, and re-run step 1 to merge:
#    - copy build_results.py to a diskB node via `ssh 'cat > /tmp/x.py'`
#    - AUDIT_ROOT=<diskB repo> AUDIT_OUT=/tmp/out python3 /tmp/x.py
#    - pull /tmp/out/results.csv -> paper_results/_results_diskB_raw.csv
#    - re-run `python3 paper_results/build_results.py`  (auto-merges diskB-only run_ids)
```

`build_results.py` is read-only, CPU-only, no network. It aggregates already-computed scores (it does **not** rescore or regenerate predictions). It emits `results.csv`, `usable_now.csv`, `excluded_results.csv`, warns (never silently skips) on unparseable files, and sorts output stably.

## The five audit questions, answered

1. **What truly completed?** Machine-scored: RULER (all scales, on diskB; 8B on wzc1), LongBench (all scales, provenance A, matches ledger), LoCoMo (8B only). **BABILong: predictions exist but NO score persisted for any model** — nothing "completed" in a citable sense (A06).
2. **Where does each number come from?** `results.csv` gives run_id + config_path + predictions_path + scorer_output_path + source_disk for every machine cell; `manual_claims.csv` marks ledger-only numbers.
3. **What is coherent now?** `usable_now.csv` (588 cells): the RULER adapter on/off ablations on 8B/14B (Table 2), the full LongBench table (Table 5), the 8B LoCoMo token-F1 (Table 6). These need no rerun.
4. **What is legacy/pilot/partial/unverifiable?** BABILong (all, needs rescore), scale LoCoMo (needs rescore), the "n=500" RULER dirs (selector-inconsistent), deep-j collapse probes, speed/vs-Dense (log-only, read-prefill, context_overflow) — see `excluded_results.csv` + `anomalies.md`.
5. **Fair tables without rerun?** Table 2 (adapter ablation), Table 5 (LongBench), Table 3 (128k beyond-window, relabeled), Table 6 (8B LoCoMo F1). Full zero-shot scaling table and BABILong require the no-GPU fixes (N1–N7) first.

## Top-3 protocol conflicts

1. **A02 — the "n=500 main table" dirs used the wrong selector** (8B zs n500 = oracle/vt-only; 8B adapter n500 = reader_attn → 86/81/81 not 100/100/100). Headline numbers actually come from n=100 bm25/iter_bm25.
2. **A04/A05 — every model reported at a different j; 32B has 5 coexisting j** (3/16/21/24/27) with no single blessed choice.
3. **A06 — BABILong has no machine score anywhere**; all BABILong paper numbers are manual + thinking-contaminated (A07).

## No-GPU path clears P0

If the seven no-GPU fixes in `minimal_rerun_plan.md` (rescore BABILong + scale LoCoMo, hash adapters, relabel speed/overflow, decide j, rebuild RULER table from correct-selector n100) are done, the **P0 rerun set is empty**. The realistic minimal GPU rerun to *strengthen* (not merely defend) is ~20 RULER cells (P1-A) + 1 vs-Dense cell (P1-C), optionally 8B clean BABILong (P1-B).
