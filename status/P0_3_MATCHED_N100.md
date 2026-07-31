# P0.3 — Matched-n=100 CoMem YaRN control (was n=50)

**Date:** 2026-07-31  ·  **Node:** QCMem `.104` (28.83.24.104, diskB, port 36000), 8× H20, GPUs 0-7
**Goal:** Close the P0.3 "mixed-n" fairness caveat in Paper A. The over-window YaRN control
reported KV-Direct (KVD) YaRN/unextended cells at n=100 but the CoMem (± self-distill LoRA)
YaRN/unextended cells at n=50. This run re-evaluates the exact CoMem cells appearing in the
scaling tables at **n=100**, into NEW dirs suffixed `_n100` (n=50 originals never touched), so
every compared cell is matched-n=100.

**Per user instruction (2026-07-31): this record does NOT edit any `.tex` file.** Numbers +
provenance verdict are handed to `main`, who folds them into `paperA/TODOList.md` and (if
desired) the tex tables.

---

## Configuration (identical to `comem_lora_A`; only limit 50→100 + 8-way shard)

- Entry: `scripts/eval_ruler_qcmem.py`  ·  PY: `/opt/conda/envs/torch-base/bin/python`
- `--resume_j 12 --selector iter_bm25 --topk 12 --iter_rounds 0 --iter_hop_topk 4`
- `--sink_tokens bos --chunk_size 512 --dtype bfloat16 --attn_impl sdpa --seed 42`
- LoRA adapter: `outputs/qcmem_distill_qwen_j12_r32_4k/final`
- **chat_template = FALSE**, enable_thinking = FALSE (paper mandate)
- `--limit 100 --num_shards 8 --shard_index {0..7}` (samples `[i::8]`), `PYTHONHASHSEED=0`
- Tasks: `niah_single_3 niah_multikey_1 variable_tracking`  ·  Lengths: `8k 16k 32k 64k 128k`
- niah_single_**3** used (matched to KVD's needle variant) for self-consistency across the table.
- Scoring: `scripts.eval_ruler_mem_space._string_match_all_one`; merge = mean of per-sample
  `recall` over all 8 shards (= n-weighted mean of shard scores).

**Two arms:**
- **Arm A — CoMem+LoRA (native):** model `models/Qwen--Qwen3-8b`  → dir `ruler_results/comem_lora_native_n100/`
- **Arm B — CoMem+LoRA+YaRN:** model `models/Qwen--Qwen3-8b-yarn` (factor-4, eff. window 163,840) → dir `ruler_results/comem_lora_yarn_n100/`

Launch driver: `scripts/_p03_matched_n100_run.sh` (on .104). Merge: `scripts/_p03_merge.py`.
Per-cell exact command:
```
CUDA_VISIBLE_DEVICES=$k python scripts/eval_ruler_qcmem.py \
  --model_path <MODEL> --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
  --resume_j 12 --selector iter_bm25 --topk 12 --iter_rounds 0 --iter_hop_topk 4 \
  --sink_tokens bos --chunk_size 512 --dtype bfloat16 --attn_impl sdpa --device cuda:0 \
  --seed 42 --ruler_tasks niah_single_3 niah_multikey_1 variable_tracking \
  --lengths 8k 16k 32k 64k 128k --limit 100 --num_shards 8 --shard_index $k \
  --output_name <OUT> --results_folder ruler_results
```

**Sanity gate:** ALL 30 cells (2 arms × 15) passed — n=100, 0 empty predictions, 0 OOM.

---

## Arm A — CoMem+LoRA (native Qwen3-8B), n=100

| Task | 8k | 16k | 32k | 64k | 128k |
|------|----|-----|-----|-----|------|
| niah_single_3     | 100.0 | 91.0 | 97.0 | 98.0 | 98.0 |
| niah_multikey_1   | 94.0  | 91.0 | 99.0 | 90.0 | 93.0 |
| variable_tracking | 96.2  | 98.0 | 98.2 | 98.6 | **99.0** |

## Arm B — CoMem+LoRA+YaRN (Qwen3-8B-YaRN factor-4), n=100

| Task | 8k | 16k | 32k | 64k | 128k |
|------|----|-----|-----|-----|------|
| niah_single_3     | 100.0 | 90.0 | 97.0 | 97.0 | 98.0 |
| niah_multikey_1   | 82.0  | 85.0 | 94.0 | 88.0 | 93.0 |
| variable_tracking | 88.6  | 90.0 | 89.4 | 95.0 | **93.6** |

---

## Old n=50 (paper/sections/tab_scaling.tex) → new n=100  (per cell)

**CoMem+LoRA (native):**
| Task | 8k | 16k | 32k | 64k | 128k |
|------|----|-----|-----|-----|------|
| niah_single | 100→100 | 86→91 | 100→97 | 100→98 | 98→98 |
| niah_multikey | 94→94 | 96→91 | 100→99 | 98→90 | 96→93 |
| var-track | 98.0→96.2 | 98.4→98.0 | 98.8→98.2 | 97.6→98.6 | **98.4→99.0** |

**CoMem+LoRA+YaRN:**
| Task | 8k | 16k | 32k | 64k | 128k |
|------|----|-----|-----|-----|------|
| niah_single | 100→100 | 86→90 | 92→97 | 90→97 | 96→98 |
| niah_multikey | 88→82 | 90→85 | 96→94 | 82→88 | 90→93 |
| var-track | 81.2→88.6 | 86.0→90.0 | 90.4→89.4 | 96.8→95.0 | **87.6→93.6** |

Differences are within expected n=50→n=100 sampling variance; no cell moved qualitatively.

---

## Macro averages (mean over 15 cells)

| Arm | niah_single | niah_multikey | var-track | **Macro (15)** |
|-----|-------------|---------------|-----------|----------------|
| CoMem+LoRA (native)   | 96.8 | 93.4 | 98.0  | **96.07** |
| CoMem+LoRA+YaRN       | 96.4 | 88.4 | 91.32 | **92.04** |

**YaRN in-window tax on CoMem ≈ −4.0 pp macro** (single −0.4, multikey −5.0, var-track −6.7).

---

## Recomputed headline — 128k variable_tracking (all n=100)

| Method | vt@128k (n=100) |
|--------|-----------------|
| CoMem+LoRA (native)     | **99.0** |
| CoMem+LoRA+YaRN         | 93.6 |
| KVD+YaRN (unchanged)    | 57.8 |
| unextended full-ctx     | 0 |

- Ranking **CoMem+LoRA > CoMem+LoRA+YaRN > KVD+YaRN > full-ctx** is PRESERVED.
- **CoMem advantage over length-extended reference = 99.0 − 57.8 = +41.2 pp**
  (was +40.6 pp using n=50's 98.4). **Headline HOLDS and slightly strengthens.**
- CoMem+LoRA+YaRN − KVD+YaRN = 93.6 − 57.8 = **+35.8 pp**.

---

## Provenance verdict (resolves the "95.8 == native" anomaly)

- **YaRN is NOT a no-op for CoMem.** Although the CoMem read pack (~6.6k tok) is always
  in-window, YaRN rescales RoPE for *all* positions, so it imposes an in-window quality tax
  (macro −4 pp; multikey@8k −12, var-track@8k −7.6). This matches the *degraded*
  `CoMem+LoRA+YaRN` row in `paper/sections/tab_scaling.tex` (n=50: vt 81.2/86.0/90.4/96.8/87.6).
- The row labeled "CoMem+LoRA+YaRN" in `paperA/sections/tab_scaling.tex` (vt 99.2/99.2/99.2/99.2/95.8)
  is actually populated with **native** n=100 numbers (from `ruler_results/qcmem_8b_iter_chatFALSE_ad`,
  which uses niah_single_2) — i.e. it is mislabeled; those are native, not YaRN, values.
- The `comem_yarn_128k` dir (vt@128k = 0.8) is a **broken/different config** — disregarded; it is
  NOT the paper recipe. Arm B here (Qwen3-8b-yarn + LoRA + iter_bm25) is the correct +YaRN recipe.

**Recommendation for `main`:** in the scaling table, the CoMem+LoRA+YaRN row should carry the
true YaRN numbers (Arm B above: vt 88.6/90.0/89.4/95.0/93.6), and both CoMem+LoRA rows are now
matched-n=100 with the KVD rows — closing the mixed-n caveat. Native vt@128k = 99.0 is the
headline number.

---

## Result artifacts (on .104 / diskB)

- `ruler_results/comem_lora_native_n100/` — Arm A, 120 shard CSVs+JSONs (15 cells × 8 shards)
- `ruler_results/comem_lora_yarn_n100/`   — Arm B, 120 shard CSVs+JSONs
- Logs: `logs/comem_lora_native_n100_shard{0..7}.log`, `logs/comem_lora_yarn_n100_shard{0..7}.log`
- Completion sentinels: `logs/comem_lora_{native,yarn}_n100_ALLDONE`, `logs/p03_matched_n100_ALLDONE`
