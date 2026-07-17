# Paper Candidate Tables — QCMem/CoMem

Every number below cites a **single** machine-scored run_id from `results.csv`/`usable_now.csv`. No cross-run averaging, no interpolation. Cells that cannot be filled from one coherent run are left `—` with a reason. **Protocol conventions:** RULER `string_match`; single/multikey selector = `bm25`, variable-tracking (vt) selector = `iter_bm25`; chunk 512; topk 12; scores 0–100.

> ⚠️ Read `anomalies.md` first. In particular: the ledger's "n=500 main table" is selector-inconsistent (A02); every model is reported at a **different j** (A04, A15); BABILong has **no machine score** (A06); speed is **read-prefill-only** (A08–A09); Dense@128k=0 is **context_overflow** (A10).

---

## Table 1 — RULER zero-shot, per-model readout-safe j (bm25 single/mkey, iter_bm25 vt)

n and run_id per row; scores 8k/16k/32k. Readout-safe j = deepest j with single-recall ≈ saturated (per `QCMEM_J_DETERMINATION.md`).

| model | j (/L) | n | niah_single | niah_multikey | vt | run_id |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 2 (0.07) | 500 | 100/100/100 | 85/84/82 | 58/60/82 | `qcmem_0p6b_balancej2_n500` |
| Qwen3-1.7B | 3 (0.11) | 500 | 99/99/99 | 54/42/41 | 56/56/52 | `qcmem_1p7b_balancej3_n500` |
| Qwen3-4B | 3 (0.08)* | 500 | 100/100/100 | 91/94/93 | 91/87/93 | `qcmem_4b_balancej3_n500` |
| Qwen3-8B | 3 (0.08)* | 300/100 | —/—/— † | 90/89/86 | 82/75/71 | `qcmem_balancej_n100` |
| Qwen3-8B | 9 (0.25) | 100 | 100/99/— | — | 27/24/22 | `qcmem_8b_j9_control_n100` / `qcmem_0p6b_j9…` |
| Qwen3-14B | 13 (0.325) | 500 | 99/89/98 | 50/44/11 | 16/13/14 | `qcmem_14b_zs_j13_n500` |
| Qwen3-32B | 27 (0.42) | 500 | 100/99.8/100 | 96/95/91 | 40/44/40 | `qcmem_32b_zs_j27_n500` |
| Qwen3-30B-A3B | 12 (0.25) | 200 | 81/66/74 | 66/52/45 | 87/77/74 | `qcmem_30ba3b_n100` |

\* **Conflict, must resolve:** for 4B and 8B the ledger's "readout-safe j" is documented as **j9** (0.25L), but the strongest coherent zero-shot cell is at **j3** (recall-optimal). j3 and j9 give very different mkey/vt (8B j3: mkey 90/89/86, vt 82/75/71; 8B j9: mkey collapses, vt 27/24/22). The paper must pick ONE definition. `results.csv` has both.
† 8B zero-shot single at j3 was not found as a clean 3-length run in the usable cohort (only mkey/vt at j3 in `qcmem_balancej_n100`); single is 100 at j9 (`qcmem_8b_j9_control_n100`).
> **This is not a single coherent table** — it stitches per-model j and even per-model n. Present it only with the j column explicit, or restrict to the models where zs is unambiguous (0.6B, 1.7B, 14B, 32B).

---

## Table 2 — Adapter on/off ablation, SAME model + SAME j (the fair ablation)

The only fully-controlled adapter ablations (same model, same j, same selector, both cells machine-scored):

| model | j | metric | zero-shot (run) | +adapter (run) | Δ |
|---|---|---|---|---|---|
| Qwen3-8B | 12 | single 8k/16k/32k | 35/5/17 (`qcmem_8b_zeroshot_j12_n100`) | 100/100/100 (`qcmem_n100`, n500) | +65/+95/+83 |
| Qwen3-8B | 12 | multikey | 4/2/1 (`…zeroshot_j12`) | 94/91/92 (`qcmem_n100_local`) | huge |
| Qwen3-8B | 12 | vt | 2/1.2/0.8 (`…zeroshot_j12`) | 96/94/93 (`qcmem_iterbm25_ext`) | huge |
| Qwen3-14B | 13 | single | 99/89/98 (`qcmem_14b_zs_j13_n500`) | 100/100/100 (`qcmem_14b_ad_j13_n500`) | +1/+11/+2 |
| Qwen3-14B | 13 | multikey | 50/44/11 (`…zs_j13_n500`) | 100/98/99 (`…ad_j13_n500`) | +50/+54/+88 |
| Qwen3-14B | 13 | vt | 16/13/14 (`…zs_j13_n500`) | 100/100/100 (`…ad_j13_n500`) | huge |
| Qwen3-0.6B | 13 (content-j) | multikey | 6/5/3 (`qcmem_0.6b_j13_semantic_n100`) | **24/24/22** (`qcmem_0p6b_adapter_contentj13_n100`) | small — tiny-model failure (A13) |
| Qwen3-0.6B | 13 (content-j) | vt | 1.8/0.6/0.2 | **0/4.8/0** | adapter does NOT help (A13) |

**Clean finding:** at j12/j13 the adapter is a decisive lever for hard tasks on 8B/14B; at content-j (0.45L) on tiny 0.6B it fails — supports the gap-vs-scale story. This is the strongest, cleanest ablation available and is fully machine-backed.

> Note the 8B +adapter j12 numbers come from n=100/300/500 **bm25** dirs (`qcmem_n100`, `qcmem_n100_local`, `qcmem_h2h_n100`), NOT the reader_attn `qcmem_8b_n500` dir (see A02).

---

## Table 3 — 128k beyond-window: QCMem vs Dense (context_overflow)

Source: `status/bench_qcmem_vs_dense_result.txt` (provenance C, log-only, n=30–100, +adapter, PG19-prose haystack). Dense window = 40960 tok.

| model | task | Dense @128k | QCMem @128k | Dense status |
|---|---|---|---|---|
| Qwen3-8B | niah_single | 0 | 100 | **context_overflow** (128k ≫ 40960) |
| Qwen3-8B | niah_multikey | 0 | 93→100(n100) | context_overflow |
| Qwen3-14B | niah_single | 11 | 100 | overflow/degraded |
| Qwen3-14B | niah_multikey | 5 | 98 | overflow |
| Qwen3-32B | niah_single | OOM | 100 | **OOM** at 128k |
| Qwen3-32B | niah_multikey | OOM | 98 | OOM |
| Qwen3-30B-A3B | niah_single | OOM | 100 | OOM |
| Qwen3-30B-A3B | niah_multikey | — | **incomplete** (A15/MC051) | node repurposed |

**Legit headline** (with correct label): beyond the native window Dense is unusable (context_overflow / OOM) while QCMem's constant 6657-tok read-pack keeps it running at ≈100. Do **not** phrase Dense's 0 as "answers wrong."

---

## Table 4 — BABILong (⚠️ NO machine score on disk — provenance C, needs rescore)

**No BABILong number can currently be cited to a machine score** (A06). The overall means below are hand-recorded in `RUN_REGISTRY.md` and are additionally **thinking-contaminated at long lengths** (A07). Present as an empty framework until a rescore of existing predictions is run.

| model | j | overall-mean (ledger, provenance C) | machine-backed? |
|---|---|---|---|
| Qwen3-8B +adapter | 12 | 55.5 | ✗ needs rescore + thinking-fix |
| Qwen3-4B | 9 | 49.3 | ✗ |
| Qwen3-14B +adapter | 13 | 46.6 | ✗ (+adapter ckpt not on wzc1) |
| Qwen3-32B | 3 | 41.7 | ✗ |
| Qwen3-8B zs | 9 | 39.2 | ✗ |
| Qwen3-1.7B | 4 | 34.2 | ✗ |
| Qwen3-14B zs | 3 | 32.7 | ✗ |
| Qwen3-30B-A3B | 12 | 32.3 | ✗ |
| Qwen3-0.6B | 3 | 11.0 | ✗ |

---

## Table 5 — LongBench qa_f1 (✅ machine-backed, provenance A)

Full test set (n=200/dataset). zero-shot at recall-optimal j except 8B (+adapter). **Cross-checks cleanly against the ledger.**

| model | j | adapter | AVERAGE f1 | run_id |
|---|---|---|---|---|
| Qwen3-32B | 3 | zs | **12.37** | `qcmem_32b` |
| Qwen3-8B | 12 | +adapter | 9.58 | `qcmem_j12` (ledger says 9.76 — use machine 9.58) |
| Qwen3-14B | 3 | zs | 9.63 | `qcmem_14b` |
| Qwen3-4B | 9 | zs | 8.51 | `qcmem_4b_j9` (single-shard `_0`, partial merge) |
| Qwen3-8B | 12 | zs/stock | 7.17 | `stock_noLoRA` |
| Qwen3-30B-A3B | 12 | zs | 6.61 | `qcmem_30ba3b` |
| Qwen3-1.7B | 4 | zs | 6.07 | `qcmem_1p7b_j4` (single-shard) |
| Qwen3-0.6B | 3 | zs | 4.51 | `qcmem_0p6b_j3` (single-shard) |

⚠️ 0.6B/1.7B/4B are single-shard `_0` (not full 4-shard merge) per ledger — verify shard-completeness before final.

---

## Table 6 — LoCoMo (⚠️ report token-F1; substring-acc is a proxy, not official judge — A11)

Only Qwen3-8B is machine-scored (scores.json). Scale models: diskB has predictions but **0 scores.json** → provenance C.

| model | j | token_f1 | substring_acc (proxy, NOT official) | machine? | run_id |
|---|---|---|---|---|---|
| Qwen3-8B +adapter | 12 | **9.59** (tk4) / 9.05 (tk8) | 25.2 / 24.1 | ✅ A | `qcmem_tk4` / `qcmem_j12` |
| Qwen3-30B-A3B | 12 | 5.02 (ledger) | 7.40 (proxy) | ✗ C | diskB, no scores.json |
| Qwen3-32B | 3 | 4.12 (ledger) | 6.55 (proxy) | ✗ C | diskB, no scores.json |
| Qwen3-14B | 3 | 2.17 (ledger) | 1.41 (proxy) | ✗ C | diskB, no scores.json |

Official LoCoMo uses an LLM judge → out of scope (needs API). Report SQuAD token-F1 only.

---

## Table 7 — Speed / memory (⚠️ read-prefill-only + constant read-pack; NOT end-to-end — A08/A09)

Source: `status/bench_qcmem_vs_dense_result.txt` (provenance C, log). Values are the **read-prefill** phase; write (≈0.59 s) + index + retrieval are excluded. Decode-speedup is measurement noise (A09) — reported qualitatively only.

| model | j | prefill× @128k (read-phase) | Dense mem @128k | QCMem mem (all lengths) | decode |
|---|---|---|---|---|---|
| Qwen3-8B | 12 | ≈57× (110.4 s → 1.92 s) | 49.8 G | 17.9 G (constant) | O(1) in ctx (numeric ✗ noisy) |
| others (0.6B–32B) | per-model | 50–103× (PLAN §1c) | grows w/ length | constant read-pack | qualitative only |

**Safe claim:** "constant read-pack ⇒ prefill cost and peak memory are independent of context length; at 128k the read-prefill is 50–100× cheaper than full-context prefill, which OOMs on 32B/30B-A3B." **Unsafe claim:** "50–100× end-to-end" or a numeric decode speedup.

---

## Table 8 — Baseline completeness matrix

See `baseline_completeness.csv`. Summary for the primary model (Qwen3-8B):

| baseline | RULER | LongBench | LoCoMo | BABILong |
|---|---|---|---|---|
| Dense/full-ctx | partial (fullctx dirs, relabel) | ✅ stock_noLoRA | ✗ missing | needs rescore |
| KV-Direct (j=0) | ✅ complete | ✅ | ✅ | needs rescore |
| HCache | ✅ complete | — | ✅ | needs rescore |
| StreamingLLM (recency) | selector-ablation only | — | — | — |
| MemoryLLM | ✗ (BABILong CSV only, old numbers wrong) | — | — | needs rescore (re-measured CSV) |

---

## Table 9 — Missing / partial / unverifiable matrix

| item | status | why | fix |
|---|---|---|---|
| All BABILong scores | needs_rescore | no score persisted (A06) | rescore existing CSVs (no GPU) |
| Scale LoCoMo (14B/32B/30B-A3B) | provenance C | diskB 0 scores.json (A11) | rescore preds (no GPU) |
| 14B / content-j adapters | ckpt unverified on wzc1 (A14) | not synced from diskB | hash on diskB |
| 30B-A3B vs-Dense multikey 128k | incomplete (MC051) | node repurposed | rerun 1 cell (GPU, P1) |
| Clean 32B reporting j | unresolved (A05) | j3/16/21/24/27 coexist | decide, no rerun |
| 8B decode speedup | suspect (A09) | noisy tok/s | drop or re-time (P2) |
| n=500 bm25/iter_bm25 full table | partial (A02) | n500 dirs used oracle/reader_attn | rebuild from n100 or rerun (P1) |
