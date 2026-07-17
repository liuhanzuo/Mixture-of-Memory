# Minimal Re-run Plan — QCMem/CoMem

Principle: **尽量不补跑.** Most gaps are fixable with **no GPU** by (a) rescoring existing predictions, (b) recovering config from logs, (c) relabeling, (d) shrinking the claim. This file lists only what to do; it executes nothing. Priorities: **P0** = paper numbers untraceable / core comparison unfair (must fix); **P1** = usable now, but a small rerun materially strengthens; **P2** = completeness, non-blocking; **P3** = do not rerun.

## No-GPU fixes first (resolve most of the audit before any rerun)

| id | action | benchmark | GPU? | resolves |
|---|---|---|---|---|
| N1 | **Rescore all BABILong predictions** with the local scorer (`TASK_LABELS`+`compare_answers`), writing a `scores.json` per run. Report strict (first-sentence) AND lenient (whole-output) to quantify thinking contamination. | BABILong | **No** (CPU) | A06, A07, MC001–010 — converts all BABILong from C→A |
| N2 | **Rescore scale-model LoCoMo predictions** on diskB (token-F1) → write scores.json. | LoCoMo | No | A11, MC031–033 |
| N3 | **Re-measure MemoryLLM BABILong** from its CSV (drop the old wrong numbers). | BABILong | No | A16 |
| N4 | **Hash the 14B / content-j adapters on diskB** (read-only) to complete provenance. | — | No | A14 |
| N5 | **Relabel** Dense@128k cells as `context_overflow`/`OOM` (not 0-accuracy); relabel speed as read-prefill. | vs-Dense/speed | No | A08, A10 |
| N6 | **Decide + document one j** per model (zs readout-safe / adapter), drop deep-j collapse probes from results. Pick 32B j. | RULER | No | A04, A05, A15 |
| N7 | **Rebuild RULER main table from the correct-selector n=100 runs** (`usable_now.csv`), demote the "n=500" label where the n500 dir used oracle/reader_attn. | RULER | No | A02, A03 |

## P0 — must rerun (only if the no-GPU fixes cannot cover a load-bearing claim)

- **None strictly required for a defensible paper**, *provided* N1 (BABILong rescore) succeeds and BABILong is reported with the thinking caveat. If reviewers demand a *clean* (non-thinking) BABILong headline, that single item escalates to P1-B below.

## P1 — usable now, rerun materially strengthens

- **P1-A — n=500 RULER at the reported selector (bm25 single/mkey, iter_bm25 vt).** Current n=500 dirs are selector-inconsistent (A02). Minimal set: only the cells whose n=500 dir used oracle/reader_attn/bm25-vt — i.e. **8B zs (single/mkey/vt), 8B +adapter (all), 4B zs (single/mkey)**. ~ (3 tasks × 3 lengths × 2 configs) ≈ 18 cells × 500 = one node-few-hours. GPU: yes. Can be scoped to same-j only (no full grid).
- **P1-B — Clean (thinking-suppressed) BABILong for the headline models** (8B ±adapter). Only if N1's lenient rescore is deemed insufficient. Re-generate with `--enable_thinking False`, qa1/qa2/qa5 × 0k–32k, n=100. GPU: yes. Not full scale — just 8B.
- **P1-C — 30B-A3B vs-Dense multikey @128k** (1 cell, incomplete run). GPU: yes, ~1 cell.

## P2 — completeness, non-blocking

- **P2-A — Fixed-j scaling row** (one j across all models) to support any across-scale trend claim (current ranking mixes j — A15). GPU: yes; or drop the scaling-law framing (no rerun).
- **P2-B — End-to-end + reliable decode timing** (write+index+retrieval+prefill+decode, warmed up) if a numeric speed claim beyond "read-prefill 50–100×" is wanted (A08/A09). GPU: yes.
- **P2-C — Full 4-shard LongBench merge for 0.6B/1.7B/4B** (currently single-shard `_0`). Likely just re-aggregate existing shards (check diskB for shards 1–3) → possibly **no rerun**, only rescore/merge.

## P3 — do not rerun

- Deep-j (≥0.45L) zero-shot RULER collapse probes — already served their purpose (readout-cliff mapping); keep as appendix, do not report as results.
- oracle/recency/adaptive selector runs — keep as ablations, not baselines.
- Legacy non-QCMem directions (mem_space/funnel/beacon/L2-L3/DMS/RMT).

## Bottom line

The **P0 set is empty** if N1–N7 (all no-GPU) are done. The realistic minimal GPU rerun to make the paper strong rather than merely defensible is **P1-A + P1-C ≈ ~20 RULER cells + 1 vs-Dense cell**, plus optionally P1-B (8B clean BABILong). Everything else is rescore/relabel/decide.
