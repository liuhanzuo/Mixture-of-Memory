# Anomalies — QCMem/CoMem benchmark audit

Each anomaly lists the observed phenomenon, affected runs, the raw evidence path, most-likely cause, whether it affects a paper conclusion, how to handle it **without re-running**, and whether a re-run is ultimately unavoidable. Where evidence is insufficient the entry says so rather than concluding for the project.

---

### A01 — n=100 main table vs n=500 stated protocol
- **Phenomenon:** `QCMEM_BENCHMARK_PLAN.md` §0 mandates n=500 for RULER/BABILong; the §1 main table and §1a are n=100; vs-Dense/selector tables are n=30–50.
- **Runs:** all RULER/BABILong main-table cells.
- **Evidence:** `results.csv` `n_valid` column; `RUN_REGISTRY.md` L1492–1514 ("n=500 全 cell 与 n=100 一致（噪声内）").
- **Cause:** n=500 firm-up was done for a subset of cells; the headline table was never re-typeset to n=500.
- **Affects conclusion?** Weakly — registry claims n500≈n100 within noise, and machine n=500 dirs exist for most cells. But a table labeled "n=500" that is actually n=100 is a factual error.
- **No-rerun handling:** report the actual n per cell (from `results.csv`); use the n=500 machine dirs where they exist and are selector-correct; otherwise label n=100 honestly.
- **Must rerun?** No.

### A02 — "n=500 main-table" dirs used the wrong selector / are task-partial ★ most important
- **Phenomenon:** the specific n=500 dirs the ledger maps to the main table do NOT reproduce the headline numbers.
  - `qcmem_8b_zeroshot_j9_n500` → selector=**oracle**, contains **vt only** (27/24/22). It is NOT the source of 8B zs single/mk/vt (100/97/99, 42/36/31, 46/42/39).
  - `qcmem_8b_n500` (8B +adapter) → selector=**reader_attn** → single 86/81/81, mk 80/72/60, vt 55/44/41 — **far below** the headline 100/100/100, 91/91/92, 97/97/98.
  - `qcmem_4b_j9_n500` → **vt only** (single/mk absent); 4B main-table single/mk come from an n=100 dir.
  - `qcmem_0p6b_balancej2_n500` → selector=**iter_bm25 for all tasks** (protocol says bm25 for single/mkey).
- **Runs:** the canonical n500 set in `RUN_REGISTRY.md` L1498–1512.
- **Evidence:** `results.csv` (filter benchmark=RULER, run_id contains `_n500`); reproduced in audit shell logs.
- **Cause:** dir naming ("_n500") does not encode selector/tasks; the ledger assumed 1 dir = 1 clean cell.
- **Affects conclusion?** **Yes** — the headline RULER numbers are provably from n=100 bm25/iter_bm25 runs, not these n=500 dirs. The n=500 firm-up for the *reported selector* is incomplete.
- **No-rerun handling:** rebuild the table from the correct-selector n=100 runs (`usable_now.csv`); demote the n=500 claim.
- **Must rerun?** Only if a true n=500 bm25/iter_bm25 table is required (P1, not P0).

### A03 — Same (model, j, task, length) has many runs with different scores
- **Phenomenon:** e.g. 8B +adapter j12 niah_single 8k appears in ≥5 dirs; naive aggregation gives n_valid=2500/3000/3500 and scores that vary by run.
- **Evidence:** `results.csv` (group by model/adapter/j/task/length → count>1); audit pivot.
- **Cause:** repeated re-runs during development; no single canonical run_id was blessed.
- **Affects conclusion?** Yes if a table cherry-picks the best run.
- **No-rerun handling:** in `paper_tables.md`, cite an explicit run_id per cell (largest-n correct-selector run); never average across runs.
- **Must rerun?** No.

### A04 — Multiple `resume_j` per model (up to 5 for 32B)
- **Phenomenon:** 32B appears at j3 / j12 / j16 / j21 / j24 / j27; 8B at j9 / j12 / j16; every model has ≥3 candidate j.
- **Evidence:** `results.csv` distinct resume_j; `QCMEM_J_DETERMINATION.md`; RUN_REGISTRY §j.
- **Cause:** the "reporting j" definition changed 4+ times (0.33L → semantic 0.45L → readout-safe → recall-optimal), each leaving runs behind.
- **Affects conclusion?** Yes — the paper must state ONE zero-shot j (readout-safe) and ONE adapter j per model and stick to it.
- **No-rerun handling:** adopt the documented dual-j (zs readout-safe / adapter content-or-0.33L); tag every cell's j; drop deep-j collapse probes from results.
- **Must rerun?** No — machine runs exist at the chosen j for all models.

### A05 — 32B split-j is genuinely unresolved (we j27 vs collaborator j16 vs benchmarks j3)
- **Phenomenon:** three coexisting 32B stories: semantic-report j27, collaborator sanity-verdict j16, actual benchmark runs j3.
- **Evidence:** `QCMEM_J_DETERMINATION.md` L49; RUN_REGISTRY L1456–1460, L1596, L1611.
- **Cause:** two methodologies (our truncation-probe vs collaborator intrinsic PPL/KL) + separate benchmark runs.
- **Affects conclusion?** Yes — 32B is the "readout reaches content peak, adapter unneeded" claim; it hinges on which j.
- **No-rerun handling:** pick j27 for the semantic-depth narrative OR j3 for the benchmark table, and say which; do not blend. All three j have machine RULER runs (j3, j16, j21, j24, j27 all present in results.csv).
- **Must rerun?** No.

### A06 — BABILong has NO score persisted anywhere ★
- **Phenomenon:** 8,311 BABILong cells have config + predictions (CSV) but no score file; per-cell json carries only config.
- **Evidence:** inventory §1 (0/351 wzc1, 0/386 diskB with scores.json); every BABILong row in `results.csv` is status=needs_rescore, score empty.
- **Cause:** the BABILong scorer (`compare_answers`+`TASK_LABELS`) was run ad-hoc and results hand-copied to RUN_REGISTRY.
- **Affects conclusion?** Yes — **every BABILong number in the paper (overall means 55.5/49.3/46.6/…) is currently provenance-C manual.**
- **No-rerun handling:** **rescore the existing CSVs** with the local scorer (allowed; CPU-only; NOT regeneration). This converts all BABILong to provenance-A without any GPU.
- **Must rerun (GPU)?** No — rescore only.

### A07 — BABILong Qwen3 predictions are thinking-contaminated
- **Phenomenon:** raw CSV outputs contain thinking preamble ("Answer: … Okay, let's tackle this question.") — the answer is present but followed by ramble; at 32k the answer migrates out of the first sentence.
- **Runs:** all Qwen3 BABILong CSVs dated ≤ 2026-07-15 (pre-fix `30bb2ab`, 2026-07-16).
- **Evidence:** `babilong_results/best_topk12_full/_shard0of8/qa1_*` CSV (seen directly); PLAN §1a caveat; predictions `chat_template=false` recorded in per-cell json.
- **Cause:** `enable_thinking` not suppressed; official first-sentence scorer deflates long-context.
- **Affects conclusion?** Yes at 32k specifically (8B qa1 32k=21 is a deflated artifact; project's own estimate 35–50).
- **No-rerun handling:** a **lenient rescore** (search whole output, not first sentence) on existing CSVs gives a fairer number AND quantifies the contamination — no GPU. Report both strict/lenient.
- **Must rerun (GPU)?** For a *clean* number, yes (re-generate with `--enable_thinking False`) — but this is P1; the lenient rescore is the no-GPU stopgap.

### A08 — Speed numbers are read-prefill-only, not end-to-end
- **Phenomenon:** the per-phase profile lists write_serial ≈ 586 ms, read_prefill ≈ 1.3 s, decode ≈ 35 ms/step **separately**; the published "prefill speedup" uses only Q_prefill (≈1.9 s) and excludes write + index + retrieval.
- **Evidence:** `status/bench_qcmem_vs_dense_result.txt` L12–28.
- **Cause:** speed table reports the read phase; the constant 6657-tok read pack is a read-side quantity.
- **Affects conclusion?** Yes if the paper says "end-to-end 50–100×". It is a **prefill-phase** speedup with constant memory.
- **No-rerun handling:** relabel as "read-prefill speedup / constant read-pack memory"; add write+index once as a fixed one-time cost; do not claim end-to-end.
- **Must rerun?** No (data exists); a true end-to-end timing would be P2.

### A09 — decode speedup (32.4×, 68.7×) is measurement noise
- **Phenomenon:** Q_tok/s reported as 15.4 / 891.7 / 486.1 / 750.2 / 721.3 across 8k/16k/32k/64k/128k — physically implausible for a constant 6657-tok pack; the published row keeps 3 points (0.9/32.4/68.7) and drops 16k=55.8× and 64k=57.2×.
- **Evidence:** `bench_qcmem_vs_dense_result.txt` L24–28; L113 self-flags "iter_bm25 decode异常需查".
- **Cause:** warmup / short-run timing noise on decode.
- **Affects conclusion?** Yes — decode-speedup magnitude is not trustworthy.
- **No-rerun handling:** **do not report decode speedup numerically**; state decode is O(1) in context (constant pack) qualitatively.
- **Must rerun?** For a numeric decode claim, yes (P2, GPU) — else drop it.

### A10 — Dense "accuracy = 0" at 128k is context overflow, not wrong answers
- **Phenomenon:** vs-Dense 128k shows Dense=0 / QCMem=100.
- **Evidence:** `bench_qcmem_vs_dense_result.txt` L7 ("exceeded predefined maximum length (40960)"), L45–56.
- **Cause:** Dense window = 40960 tok ≪ 128k → the model cannot attend; 0 is overflow.
- **Affects conclusion?** It IS a legitimate selling point ("beyond-window, Dense unusable, QCMem constant"), but must be labeled **context_overflow / OOM**, not "Dense answers wrong."
- **No-rerun handling:** relabel the cell status as context_overflow; keep the QCMem-survives claim.
- **Must rerun?** No.

### A11 — LoCoMo mixes token-F1, substring-acc, and (absent) LLM-judge
- **Phenomenon:** ledger reports LoCoMo "acc" (e.g. 30B-A3B acc 7.4) and F1 (5.0); the eval script's `overall_acc` is a **substring proxy**, self-documented as a judge stand-in, NOT the official GPT-judge.
- **Evidence:** `locomo_results/*/scores.json` (overall_f1 / overall_acc / overall_em); PLAN §1a note; `eval_qcmem_locomo.py`.
- **Cause:** no API judge available; substring proxy used.
- **Affects conclusion?** Yes if "acc" is presented as official LoCoMo accuracy.
- **No-rerun handling:** report **token-F1 only** (SQuAD-comparable) and label substring-acc explicitly as an internal proxy; scale models need a rescore (diskB has 0 scores.json for LoCoMo).
- **Must rerun (GPU)?** No for 8B (scores.json exists); scale models = rescore preds (P1). Official LLM-judge = needs API (out of scope).

### A12 — 14B readout-safe j13 has single@16k = 89 (< the ≥90 rule)
- **Phenomenon:** the readout-safe rule is "single ≥90 deepest j", but 14B j13 single 16k = 89.
- **Evidence:** `results.csv` run `qcmem_14b_zs_j13_n500` single 16k = 88.6; PLAN §1 shows 99/89/98.
- **Cause:** j13 is marginally past the ≥90 knee at 16k.
- **Affects conclusion?** Minor — the "single recall near-saturated" claim holds at 8k/32k, dips to 89 at 16k.
- **No-rerun handling:** report the exact 99/89/98 with the caveat; do not round 89→90.
- **Must rerun?** No.

### A13 — 0.6B / 1.7B content-j adapter near-collapse on hard tasks IS a real run
- **Phenomenon:** 0.6B +adapter@j13 multikey 24/24/22, vt 0/4.8/0; worse than its own shallow-j zero-shot.
- **Evidence:** `results.csv` run `qcmem_0p6b_adapter_contentj13_n100` (diskB); `QCMEM_J_DETERMINATION.md` L108–116.
- **Cause:** tiny model cannot distill deep-cache compositional readout (gap≈0.39L).
- **Affects conclusion?** No — it *supports* "adapter value shrinks with scale, tiny models fail at content-j." Verified real, not a bug.
- **No-rerun handling:** report as a positive finding (gap-vs-scale).
- **Must rerun?** No.

### A14 — 14B and content-j adapter checkpoints not on wzc1
- **Phenomenon:** `qcmem_distill_14b_j13_r32` and all content-j adapters (0.6B/1.7B j13, 4B/8B j16, 14B j18) are absent from local `outputs/`.
- **Evidence:** `outputs/*/final` listing (§4 inventory); RULER cells for these adapters exist on diskB.
- **Cause:** adapters trained/stored on diskB, not synced to wzc1.
- **Affects conclusion?** Provenance only — the adapter *was* used (eval cells exist), but the checkpoint is not hashed/verified here.
- **No-rerun handling:** hash the adapters on diskB (read-only) to complete provenance; until then mark `adapter_hash=MISSING`.
- **Must rerun?** No.

### A15 — Overall ranking depends on incomparable j across models (BABILong)
- **Phenomenon:** the "overall mean acc" ranking (8B-ad 55.5 > 4B 49.3 > 14B-ad 46.6 > …) compares models at **different j and different adapter states** (8B j12+ad, 4B j9 zs, 14B j13+ad, 32B j3 zs, 30B-A3B j12 zs).
- **Evidence:** RUN_REGISTRY L1664; `manual_claims.csv` MC001–009.
- **Cause:** each model reported at its own best config.
- **Affects conclusion?** Yes — a scale-trend claim across models needs a fixed protocol; this ranking is best-config-per-model, not a controlled scaling curve.
- **No-rerun handling:** present as "best achievable per scale" with the j/adapter noted per bar; do not claim a monotone scaling law.
- **Must rerun?** No (but a fixed-j scaling row would be P2).

### A16 — MemoryLLM baseline numbers were wrong once already
- **Phenomenon:** the widely-cited MemoryLLM qa1 (53/42/32/23/14/9/7) was re-measured and found WRONG; corrected qa1 ≈ (–)/50/49/25/30/20/12, changing the 16k QCMem-vs-MemoryLLM ratio from 6.3× to 2.85×.
- **Evidence:** RUN_REGISTRY L1350–1358; `babilong_results/MemoryLLM-8B-chat` (diskB).
- **Cause:** earlier hand-transcription error.
- **Affects conclusion?** Yes if any MemoryLLM comparison is made.
- **No-rerun handling:** use ONLY the re-measured CSV (rescore it); drop the old numbers.
- **Must rerun?** No — rescore existing CSV.
