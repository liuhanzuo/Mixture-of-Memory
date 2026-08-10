# P0.2 Pareto — Config #2 (j=0 text-RAG) quality, 5 benchmarks

**Date:** 2026-08-01 · **Node:** .104 (28.83.24.104, port 36000, diskB) · 8× H20, 8-way sample sharding
**Config #2 recipe (EXACT, all cells):** Qwen3-8B `models/Qwen3-8b-local`, `--resume_j 0` (full 36-layer
recompute of the retrieved pack — this is the self-distillation **teacher** / full-depth retrieval baseline),
`--selector iter_bm25 --topk 12 --iter_rounds 0 --iter_hop_topk 4 --chunk_size 512 --sink_tokens bos`,
`--dtype bfloat16 --attn_impl sdpa --seed 42`, **NO LoRA adapter**, **chat_template=FALSE, enable_thinking=FALSE**.

All numbers below are **disk-verified** (n=100 per cell for RULER/BABILong/LongEval; n=1986 for LoCoMo).
Nothing is fabricated; any absent cell is marked `NOT FOUND`. This file reports **config #2 ONLY** — it does not
restate the #1/#3/#4 numbers already in `status/P0_2_PARETO_RESULTS.md` except in the explicit comparison note.

---

## 1. RULER (n=100, string_match recall × 100, chat=False)

| task | 8k | 16k | 32k | 64k | 128k | provenance | src |
|------|----|----|----|----|----|------------|-----|
| niah_single_3   |  99 |  99 |  98 |  96 |  99 | `ruler_results/p0_2_c2_j0_iterbm25_niah_chatFALSE/` | FRESH |
| niah_multikey_1 | 100 | 100 |  99 |  99 |  99 | `ruler_results/p0_2_c2_j0_iterbm25_niah_chatFALSE/` | FRESH |
| variable_tracking | 100 | 100 | 100 | 100 | 100 | `ruler_results/presub_A_kvdirect_iterbm25_vt/` | REUSED |

- **RULER 15-cell macro = 99.20** (single_3 491 + multikey_1 497 + VT 500 = 1488 / 15).
- VT dir is named `...kvdirect...` but its `eval_config` is verified config #2
  (`resume_j=0, selector=iter_bm25, topk=12, sink_tokens=bos, lora_adapter=null, chunk_size=512,
  reuse_kv_blockdiag=false, top_prepay_b=0`) — the dir name is misleading, the recorded args are j=0 RAG.
- NIAH 8k/16k/32k were the RULER-CORE phase; 64k/128k the RULER-EXT phase; both wrote to the same
  `p0_2_c2_j0_iterbm25_niah_chatFALSE/` dir. All cells n=100 (8 shards × ~13 each, summed).

## 2. LongBench (6-QA, SQuAD token-F1, macro; chat=False)

| dataset | F1 | n |
|---------|----|----|
| 2wikimqa        | 12.42 | 200 |
| hotpotqa        | 12.17 | 200 |
| multifieldqa_en | 26.18 | 150 |
| musique         |  7.47 | 200 |
| narrativeqa     |  3.88 | 200 |
| qasper          | 11.73 | 200 |
| **6-QA macro**  | **12.31** | — |

- Provenance: `longbench_results/p0_2_c2_j0_iterbm25_chatFALSE/` (FRESH, 48 `_metrics.json` = 6 ds × 8 shards).
- **6-QA macro is authoritative.** The "4-QA" subset used in `P0_2_PARETO_RESULTS.md` §4b is **not formally
  defined in this harness** (`lb.DEFAULT_DATASETS` = the 6 above), so no 4-QA macro is asserted here — it is
  derivable from the per-dataset F1 above if a subset is fixed.
- Aggregator口径 validated: same script reproduces flagship #4 = **12.15** exactly (matches doc §4b).

## 3. LongEval (line-key retrieval accuracy, n=100/length, chat=False)

| 4k | 8k | 16k | 32k | 64k | 128k |
|----|----|----|----|----|----|
| 98.0 | 100.0 | 96.0 | 99.0 | 94.0 | 97.0 |

- **Mean (8k–128k) = 97.2%** (doc口径, 5 lengths) · Mean (4k–128k) = 97.3% (6 lengths).
- Provenance: `longeval_results/p0_2_c2_j0_iterbm25_chatFALSE/longeval_8b/` (FRESH, `_summary_merged.json`).
- read_len ~6.3–6.5k across all lengths → L-independent retrieved pack, as expected for RAG.

## 4. BABILong (n=100, TASK_LABELS + compare_answers, chat=False)

| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|------|----|----|----|----|----|----|----|
| qa1 | 98 | 84 | 80 | 73 | 60 | 34 | 35 |
| qa2 | 58 | 53 | 51 | 44 | 37 | 17 | 12 |
| qa5 | 70 | 73 | 61 | 57 | 69 | 53 | 60 |

- Provenance: `babilong_results/p0_2_c2_j0_iterbm25_chatFALSE/` (FRESH, flat layout, 168 CSVs = 3 tasks × 7 len × 8 shards; scored by summing shard CSVs, all cells n=100).
- qa1/qa5 requested; qa2 included for completeness. 21-cell mean = 56.1.

## 5. LoCoMo (full, n=1986, chat=False)

| F1 | acc | EM | GPT-4o judge |
|----|-----|----|--------------|
| 9.90 | 25.23% | 0.81% | 41.59% |

- Provenance: `locomo_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE/scores.json` — **on the wzc1 (local) disk**,
  NOT diskB (that is why a diskB-only search returned NOT FOUND). `eval_config_shard0of8.json` verifies exact
  config #2 (`resume_j=0, selector=iter_bm25, topk=12, iter_hop_topk=4, sink_tokens=bos, chunk_size=512,
  use_chat_template=false, enable_thinking=false, Qwen3-8b-local`). REUSED — judge already computed (gpt-4o),
  no re-run needed.
- per-category judge: cat1 31.2 / cat2 23.1 / cat3 40.6 / cat4 72.4 / cat5 3.6.

---

## 6. vs flagship #4 (j=12 + LoRA) / #1 (KV-Direct) — comparison note

| benchmark | **#2 j=0 RAG (this file)** | #4 LoRA | #1 KV-Direct |
|-----------|---------------------------|---------|--------------|
| RULER 15-cell macro | **99.20** | 96.07 | (see doc) |
| LongBench 6-QA macro F1 | **12.31** | 12.15 | 12.17 |
| LongEval mean (8k–128k) | **97.2%** | 69.0% | 65.2% |
| LoCoMo F1 / acc / judge | **9.90 / 25.23% / 41.59%** | 9.15 / 23.4% / — | 9.02 / 22.4% / — |
| BABILong qa1 (0k–32k) | **98/84/80/73/60/34/35** | 98/80/68/68/46/17/12 | 98/84/80/74/80/72/63 |
| BABILong qa5 (0k–32k) | **70/73/61/57/69/53/60** | 68/76/76/75/68/60/58 | 71/73/62/59/65/42/58 |

- Config #2 is the **full-depth (j=0) retrieval teacher**: it matches/exceeds flagship #4 on RULER (99.2 vs 96.1),
  LongBench (12.31 vs 12.15), and LoCoMo, and is **far higher on LongEval (97.2% vs 69.0%)** — full recompute
  preserves the line/position fidelity that the j=12 mid-layer cache loses. This confirms #2 as the upper-bound
  teacher the flagship distills toward.
- On **BABILong qa1 at long lengths** #2 (34/35 @16k/32k) beats #4 (17/12) but trails #1 KV-Direct (72/63):
  BM25 top-12 retrieval can miss the required fact that full-context KV-Direct always keeps. qa5 comparable across all three.

## 7. Exact commands (per benchmark; COMMON args as in §recipe)

```
COMMON="--model_path models/Qwen3-8b-local --resume_j 0 --selector iter_bm25 --topk 12 \
        --iter_rounds 0 --iter_hop_topk 4 --chunk_size 512 --sink_tokens bos \
        --dtype bfloat16 --attn_impl sdpa"   # NO --use_chat_template, NO --enable_thinking, NO --lora_adapter
PY=/opt/conda/envs/torch-base/bin/python   # on .104 (diskB)

# RULER (8-way shard i in 0..7; --seed 42)
$PY scripts/eval_ruler_qcmem.py $COMMON --seed 42 --ruler_tasks niah_single_3 niah_multikey_1 \
    --lengths 8k 16k 32k --limit 100 --num_shards 8 --shard_index $i \
    --results_folder ruler_results --output_name p0_2_c2_j0_iterbm25_niah_chatFALSE   # + --lengths 64k 128k (EXT)
# BABILong
$PY scripts/eval_qcmem_babilong.py $COMMON --tasks qa1 qa2 qa5 --lengths 0k 1k 2k 4k 8k 16k 32k \
    --limit 100 --num_shards 8 --shard_index $i --output_name p0_2_c2_j0_iterbm25_chatFALSE
# LongEval
$PY scripts/eval_qcmem_longeval.py $COMMON --lengths 4k 8k 16k 32k 64k 128k --num_samples 100 \
    --num_shards 8 --shard_index $i --results_folder longeval_results \
    --output_name longeval_8b   # (dir p0_2_c2_j0_iterbm25_chatFALSE)
# LongBench
$PY scripts/eval_qcmem_longbench.py $COMMON --max_samples -1 --num_shards 8 --shard_index $i \
    --output_dir longbench_results/p0_2_c2_j0_iterbm25_chatFALSE

# Scoring
$PY scripts/score_nested_babilong.py <flat variant used>   # BABILong (flat: glob task_len_*.csv, sum shards)
$PY scripts/eval_qcmem_longeval.py --score_only --results_folder longeval_results/p0_2_c2_j0_iterbm25_chatFALSE \
    --output_name longeval_8b --lengths 4k 8k 16k 32k 64k 128k
# RULER: sum per-shard recall column; LongBench: per-ds weighted-f1 by n, macro over 6.
```

## 8. Summary (config #2, j=0 text-RAG)

- RULER macro **99.20** · LongBench 6-QA macro **12.31** · LongEval mean(8k–128k) **97.2%** ·
  LoCoMo F1 **9.90** / acc **25.23%** / gpt-4o judge **41.59%** · BABILong qa1 98/84/80/73/60/34/35, qa5 70/73/61/57/69/53/60.
- All cells disk-verified, n=100 (RULER/BABILong/LongEval per cell) / n=1986 (LoCoMo). No NOT-FOUND / NOT-RUN cells.
