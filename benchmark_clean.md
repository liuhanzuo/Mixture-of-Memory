# benchmark_clean.md — QCMem/CoMem (audited, paper-safe)

> **Status:** clean successor to `status/QCMEM_BENCHMARK_PLAN.md` / `status/RUN_REGISTRY.md`, produced by the read-only audit in `paper_results/` (git `9c258a7`, 2026-07-17). The originals are **kept unchanged** as historical record. This file lists **only numbers traceable to a run_id in `paper_results/results.csv`**, or clearly marks a number as `[LEDGER-ONLY]` / `[NEEDS-RESCORE]` / `[LOG-ONLY]`. It removes contradictory "completed" claims.
>
> Legend: **[A]** machine-scored, single run_id · **[C]** ledger/manual only · **[RESCORE]** predictions exist, no score on disk · **[LOG]** from `bench_qcmem_vs_dense_result.txt` only · **[OVERFLOW]** Dense beyond native window.

## 0. Fixed protocol (as reported here)
RULER `string_match`, chunk 512, topk 12; selector = bm25 (single/multikey), iter_bm25 (variable-tracking). LongBench `qa_f1` full test set. LoCoMo SQuAD token-F1 (substring-acc is an internal proxy, **not** the official LLM judge). BABILong `TASK_LABELS`+`compare_answers`. Scores 0–100. Every model's split-depth **j is stated explicitly** — the numbers below are NOT at a single shared j.

## 1. What is paper-ready NOW (no rerun)

### 1a. RULER adapter on/off ablation — same model, same j [A]
The cleanest, fully machine-backed result.

| model | j | metric | zero-shot | +adapter |
|---|---|---|---|---|
| Qwen3-8B | 12 | single 8k/16k/32k | 35/5/17 | 100/100/100 |
| Qwen3-8B | 12 | multikey | 4/2/1 | 94/91/92 |
| Qwen3-8B | 12 | vt | 2/1/1 | 96/94/93 |
| Qwen3-14B | 13 | single | 99/89/98 | 100/100/100 |
| Qwen3-14B | 13 | multikey | 50/44/11 | 100/98/99 |
| Qwen3-14B | 13 | vt | 16/13/14 | 100/100/100 |
| Qwen3-0.6B | 13 (content-j) | multikey | 6/5/3 | 24/24/22 (tiny-model adapter fails) |

Runs: `qcmem_8b_zeroshot_j12_n100`, `qcmem_n100`/`_local`, `qcmem_iterbm25_ext`, `qcmem_14b_zs_j13_n500`, `qcmem_14b_ad_j13_n500`, `qcmem_0p6b_adapter_contentj13_n100`. **Finding:** at j12/j13 the self-distill adapter is a decisive lever for hard tasks (8B/14B); at content-j on 0.6B it fails → adapter value shrinks with scale.

### 1b. LongBench qa_f1 — full test set [A] (matches ledger)
| model | j | adapter | AVERAGE f1 |
|---|---|---|---|
| Qwen3-32B | 3 | zs | 12.37 |
| Qwen3-14B | 3 | zs | 9.63 |
| Qwen3-8B | 12 | +adapter | 9.58 |
| Qwen3-4B | 9 | zs | 8.51 ⚠️single-shard |
| Qwen3-8B | 12 | stock | 7.17 |
| Qwen3-30B-A3B | 12 | zs | 6.61 |
| Qwen3-1.7B | 4 | zs | 6.07 ⚠️single-shard |
| Qwen3-0.6B | 3 | zs | 4.51 ⚠️single-shard |

### 1c. LoCoMo — Qwen3-8B only [A]
Qwen3-8B +adapter j12: **token-F1 = 9.59** (topk4) / 9.05 (topk8), n=1986 full set. (substring-acc 25/24 = internal proxy, not reported as accuracy.)

### 1d. 128k beyond-window survival [LOG][OVERFLOW]
Beyond the 40960-tok native window, Dense is unusable (context_overflow at 8B/14B; OOM at 32B/30B-A3B) while QCMem's constant 6657-tok read-pack runs at ≈100 (single) / 93–100 (multikey). Correct framing: "Dense cannot run beyond its window; QCMem's memory is context-length-independent." Not "Dense answers wrong."

### 1e. Constant-memory / read-prefill speed [LOG]
QCMem peak memory is constant across 8k/32k/128k (8B ≈ 17.9 G) because the read-pack is fixed; Dense memory grows with length and OOMs at 128k on 32B/30B-A3B. Read-prefill at 128k is ≈50–100× cheaper than full-context prefill. **Do not claim end-to-end or a numeric decode speedup** (decode timing is measurement noise).

## 2. What is PENDING (predictions exist, needs rescore — no GPU)

- **BABILong — ALL models [RESCORE].** No score is persisted on disk for any BABILong run. The ledger overall-means (8B-ad 55.5, 4B 49.3, 14B-ad 46.6, 32B 41.7, 8B-zs 39.2, 1.7B 34.2, 14B-zs 32.7, 30B-A3B 32.3, 0.6B 11.0) are **[C] ledger-only** and additionally **thinking-contaminated at long lengths** (8B qa1 32k=21 is a deflated artifact; true est. 35–50). → rescore existing CSVs (strict + lenient) before use.
- **LoCoMo — scale models (14B/32B/30B-A3B) [RESCORE].** diskB has predictions but 0 scores.json. Ledger token-F1 (32B 4.12, 30B-A3B 5.02, 14B 2.17) are [C].
- **MemoryLLM baseline [RESCORE].** Old numbers are known-wrong; re-measured CSV exists.

## 3. What is UNRESOLVED (decide, then no rerun)

- **Reporting j per model.** zero-shot readout-safe (0.6B j2 / 1.7B j3 / 4B j9 / 8B j9 / 14B j13 / 32B j27) vs recall-optimal shallow j3 give very different multikey/vt. Pick one definition. **32B has 5 coexisting j (3/16/21/24/27)** — choose the paper j.
- **Zero-shot scaling table.** No single coherent full-scale zero-shot RULER table exists (mixed j/selector/n). Either fix a protocol and rebuild from `usable_now.csv`, or restrict to the unambiguous models (0.6B, 1.7B, 14B, 32B).

## 4. What NOT to claim (unless the supporting data is fixed first)
- ~~"all-scale, all-benchmark complete"~~ — BABILong unscored, scale LoCoMo unscored.
- ~~"end-to-end 50–100× speedup"~~ — it is read-prefill-only; decode numbers are noise.
- ~~"in-window matches dense" as a scaling law~~ — only cleanly shown for 8B/14B RULER; BABILong pending.
- ~~"adapter benefit is largest for small models"~~ as proven — the gap-vs-scale story is supported by RULER content-j (0.6B adapter fails), but BABILong (which the ranking cites) is unscored.
- Dense@128k "0 accuracy" — say context_overflow / OOM.

## 5. Baselines actually completed (Qwen3-8B)
KV-Direct (RULER/LongBench/LoCoMo [A]; BABILong [RESCORE]); HCache (RULER/LoCoMo [A]); Dense/stock (LongBench [A]; RULER fullctx dirs need relabel; LoCoMo missing). StreamingLLM = recency-selector ablation only. MemoryLLM = BABILong CSV only, needs rescore.

---
*Provenance for every number: `paper_results/results.csv` (machine) / `manual_claims.csv` (ledger). Anomalies + evidence paths: `paper_results/anomalies.md`. Rerun priorities: `paper_results/minimal_rerun_plan.md`.*
