# P0.11 — Frozen j=12 (NO LoRA) control eval

**Purpose:** Same-depth (j=12) control isolating the LoRA/distillation adaptation gain.
Compare against the flagship **CoMem + LoRA (j=12)** row and the **CoMem frozen (j=9)** row in `paperA/sections/tab_overview.tex`.

## Fixed config (all cells)

- Model: **Qwen3-8B** (`models/Qwen3-8b-local` → `/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b`)
- `resume_j = 12`
- **NO LoRA adapter (frozen backbone)** — no `--lora_adapter`
- `selector = iter_bm25`, `topk = 12`, `iter_hop_topk = 4`, `rounds = 0`
- `chunk_size = 512`, sink = **BOS**
- **`chat_template = False`** (paper-wide mandate; model is continue-train BASE LM, no SFT/RL)
- seed = 42
- Node: diskB **.73** (`28.48.7.53`), PYBIN `/opt/conda/envs/torch-base/bin/python`
- Scheduler: 8-GPU task-pool (flock queue), 8-shard per (task,length), samples `[i::8]`
- Launcher: `scripts/_p0_11_frozen_j12_chatFALSE_taskpool.sh`
- All 80 jobs completed, **0 failures**, `SCHED_DONE` present.

---

## 1. RULER — 15-cell macro

**Macro (15-cell equal-weight mean) = 120.2 / 15 = 8.01** (string_match_all, n=100/cell, all cells Iron-Law-2 OK: 8 shards, empty=0, recompute-vs-ondisk mismatch=0)

| task | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|
| niah_single_2 | 36.0 | 7.0 | 22.0 | 18.0 | 9.0 |
| niah_multikey_1 | 8.0 | 3.0 | 1.0 | 6.0 | 4.0 |
| variable_tracking | 2.4 | 1.2 | 2.0 | 0.2 | 0.4 |

- n = 100 per cell. Scoring kernel: `scripts/score_ruler_taskbreadth.py` (official `_string_match_all_one`, case-insensitive substring recall).
- Driver: `scripts/eval_ruler_qcmem.py` (all 5 lengths incl. 64k/128k, resume_j=12), max_new=48.
- Result dir: `ruler_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE/` (diskB .73)
- Aliases: niah_single→niah_single_2, niah_multi→niah_multikey_1, vt→variable_tracking.

## 2. LongEval — per length

**mean(8k–128k) = 0.2%** (n=100/length)

| 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|
| 0.0% | 0.0% | 1.0% | 0.0% | 0.0% |

- Driver: `scripts/eval_qcmem_longeval.py` (iter_bm25/iter_hop_topk), max_new=64, 8-shard.
- Result dir: `longeval_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE/_summary_merged.json` (diskB .73)

## 3. LongBench — 6-QA macro F1

**AVERAGE macro F1 = 9.96** (mean of 6 dataset F1s)

| narrativeqa | qasper | hotpotqa | 2wikimqa | multifieldqa_en | musique |
|---|---|---|---|---|---|
| 3.85 | 10.67 | 8.85 | 9.55 | 20.76 | 6.06 |

- Driver: `scripts/eval_qcmem_longbench.py` / `scripts/eval_longbench_mem_space.py` (DEFAULT 6 QA sets), 8-shard.
- Result dir: `longbench_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE/scores.json` (diskB .73)

---

## Known (NOT rerun for this control)

- BABILong = 24.52
- LoCoMo = 24.52

## Provenance / logs

- Summary: `logs/qcmem_frozen_j12_chatFALSE/SUMMARY.txt` (diskB .73)
- Sched log: `logs/qcmem_frozen_j12_chatFALSE/sched.out` (diskB .73)
- Launch record: `status/TRAINER_ACTIVITY.jsonl` (exp `qcmem_frozen_j12_chatFALSE`, 2026-07-31T17:14:00+08:00, 80 jobs)

## Interpretation

Frozen j=12 (no LoRA) collapses vs the flagship **CoMem + LoRA (j=12)** RULER 97.05 (this control: 8.01) and even vs **CoMem frozen (j=9)** RULER 59.41. At the deeper j=12 split the frozen backbone cannot read the compressed memory buffer without distillation/LoRA adaptation — i.e. the adaptation gain is essential at j=12, not a free property of the depth. This isolates the distillation/LoRA contribution at fixed depth.
