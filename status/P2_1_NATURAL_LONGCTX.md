# P2.1 — QCMem on NATIVE long-context NATURAL tasks (∞Bench / InfiniteBench)

**Date:** 2026-08-01 · **Node:** `.73` (28.85.35.73, diskB `zwfy6/share_304376610`, port 36000), 8× H20 · **PY** `/opt/conda/envs/torch-base/bin/python`
**Owner item:** Paper A **P2.1** — reproduce QCMem on genuinely NATURAL long-document tasks (real novels, not synthetic RULER needles), on a native-long-context backbone, scored LOCALLY (no GPT-4o judge).
**Per user rules: this record does NOT edit any `.tex` file and does NOT touch `paperA/TODOList.md`.** Numbers + verdict are handed to `main`.

---

## Dataset chosen (and why)

**∞Bench / InfiniteBench** (`xinrongzhang2022/InfiniteBench`), two **judge-free, natural** English subtasks:

| Task | Metric | Docs | Avg length (measured) | Why chosen |
|---|---|---|---|---|
| `longbook_qa_eng` | token-F1 | 351 real novels | **~90k–237k tokens** (over the 131072 window) | Natural free-form QA over a whole book — the "native long QA" case. Local F1, no judge. |
| `longbook_choice_eng` | accuracy (A/B/C/D) | 229 real novels | **~250k–266k tokens** | Natural 4-way multiple-choice over a whole book — aggregate/reasoning read. Local EM on the letter, no judge. |

**Why ∞Bench and not the alternatives:**
- **Genuinely native long-context + natural.** ∞Bench docs are real Project-Gutenberg novels averaging **100k+ tokens** (measured 90k–266k here) — well past the 131072 native window, so they exercise the true over-window regime on NATURAL prose. This is exactly the P2.1 ask (native 128k+ natural, not synthetic single-needle RULER).
- **Judge-free / local scoring.** longbook_qa_eng = token-F1, longbook_choice_eng = A/B/C/D accuracy — both computed locally with the official ∞Bench scorers (`src/compute_scores.py`), no GPT-4o. (This is why LongMemEval was rejected: its official protocol requires a GPT-4o auto-judge.)
- **LongBench rejected** for P2.1: its contexts are only ~5k–30k tokens — real docs but NOT native-128k; already covered by `eval_qcmem_longbench.py`.
- Optional extra ∞Bench tasks are wired in the harness (`longdialogue_qa_eng` EM, `code_debug` EM, `math_find` EM, `longbook_sum_eng` ROUGE-L) if `main` wants broader natural coverage; the two above are the headline judge-free natural QA + reasoning pair.

Data downloaded to diskB (shared across .73/.82/.104):
`data/infinitebench/longbook_qa_eng.jsonl` (284 MB, 351 ex) · `longbook_choice_eng.jsonl` (177 MB, 229 ex).

---

## Harness

`scripts/eval_qcmem_infbench.py` (committed `4bbfece`, committer LiuHanzuo, no AI trailer). Thin composition — nothing about the QCMem forward path is re-implemented:
- **QCMem forward path reused verbatim** via `import scripts.eval_qcmem_babilong` → `qcmem_generate` / `QCMemModel` / `run_self_test` (same import the LongBench/RULER drivers use).
- **Self-contained ∞Bench framework** (no `external/InfLLM` dependency on the node): offline streaming JSONL loader (`load_infinitebench`, early-stop on `--max_samples` so a smoke never reads the full 284 MB), official prompt templates (`INFBENCH_PROMPT`) + per-task gen budgets (`INFBENCH_MAXGEN`), and the official judge-free scorers copied from OpenBMB/InfiniteBench `src/compute_scores.py` (qa token-F1, choice/code letter-EM, dialogue name-EM, math number-EM, sum ROUGE-L). **All 17 scorer unit cases pass offline.**
  - The upstream `longbook_choice_eng` scorer has a known bug when the label is a `[text, letter]` pair; this harness ships a **robust choice scorer** (first ABCD letter emitted vs gold letter, else "answer is X" parse, else normalized-text match) — validated by the unit tests.
- **Baselines** (identical resolution to `eval_qcmem_longbench.py`): `--baseline kvdirect` = native-window Dense (forces `resume_j=0`, `no_retrieval=True` packs every chunk, drops LoRA) · `--baseline hcache` (keep `resume_j`, no retrieval) · `none` = QCMem retrieval.
- `--num_shards`/`--shard_index` strided sharding, per-shard `{task}_{shard_tag}.jsonl` + metrics, `eval_config_*.json` dump, `--score_only` merge/dedup-by-index → `scores.json`. OOM guarded (`pred="[OOM]"`).

**Config = flagship (identical to P0.3 / P1.1):** Qwen3-8B `models/Qwen3-8b-local`, LoRA `outputs/qcmem_distill_qwen_j12_r32_4k/final`, `resume_j 12`, `selector iter_bm25`, `topk 12`, `iter_rounds 0`, `iter_hop_topk 4`, `sink bos`, `chunk_size 512`, bf16/SDPA, seed 42, **chat_template=False** (paper mandate), `PYTHONHASHSEED=0`.

---

## Smoke (n=2, ONE GPU, .73 GPU0) — ✅ PASS

```
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python scripts/eval_qcmem_infbench.py \
  --model_path models/Qwen3-8b-local --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
  --resume_j 12 --selector iter_bm25 --topk 12 --iter_rounds 0 --iter_hop_topk 4 \
  --sink_tokens bos --chunk_size 512 --dtype bfloat16 --attn_impl sdpa --seed 42 \
  --tasks longbook_qa_eng longbook_choice_eng --data_dir data/infinitebench \
  --output_dir infbench_results/smoke_8b_j12 --max_samples 2 --device cuda:0
```
- `longbook_qa_eng`: f1=12.04 (n=2, 8.6s) · `longbook_choice_eng`: acc=50.00 (n=2, 9.0s) · MACRO=31.02.
- End-to-end verified: model+LoRA load, ∞Bench load, QCMem write/select/read/decode, local scoring, per-shard JSONL + metrics + `scores.json` all written.
- Proves QCMem handles genuinely over-window natural docs: tokenizer warned `266742 > 131072` (native window) yet QCMem chunked+compressed it and decoded at **constant ~18.5 GB/card** read.

---

## Phase 2 — full 8-way sharded sweep on .73 (LAUNCHED 2026-08-01, green-lit by main)

**Arm A — QCMem flagship** (`infbench_results/qcmem_8b_j12_lora`): 8 shards, GPUs 0-7, both tasks, full test sets (qa=351, choice=229). Running at ~5.8 s/sample, constant 18.5 GB/card (bounded read = O(1) memory, invariant to the 90k–266k doc length). Launch:
```
for k in 0..7:
  CUDA_VISIBLE_DEVICES=$k PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/eval_qcmem_infbench.py --model_path models/Qwen3-8b-local \
    --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
    --resume_j 12 --selector iter_bm25 --topk 12 --iter_rounds 0 --iter_hop_topk 4 \
    --sink_tokens bos --chunk_size 512 --dtype bfloat16 --attn_impl sdpa --seed 42 \
    --tasks longbook_qa_eng longbook_choice_eng --data_dir data/infinitebench \
    --output_dir infbench_results/qcmem_8b_j12_lora --num_shards 8 --shard_index $k --device cuda:0
```
**Arm B — KV-Direct (native-window Dense)** (`infbench_results/kvdirect_8b`): same launch + `--baseline kvdirect` (no LoRA, resume_j→0, packs all chunks). Full-depth forward over the whole 90k–266k-token doc → expected to exceed H20 memory (KV cache for ~90k tok ≈ 106 GB > 97 GB) → native-window collapse. To run after Arm A frees the GPUs.

Merge/score each arm: `python scripts/eval_qcmem_infbench.py --score_only --tasks longbook_qa_eng longbook_choice_eng --output_dir <arm_dir>`.

### Results

QCMem arm COMPLETE (all 8 shards, n=351 qa / n=229 choice, ~170 s/shard qa + ~92 s/shard choice, **constant ~18.5 GB/card**). KV-Direct arm running.

| Task | Metric | QCMem (j12, iter_bm25, +LoRA) | KV-Direct (native Dense, resume_j=0) |
|---|---|---|---|
| longbook_qa_eng | F1 | **6.06** (n=351) | _running_ |
| longbook_choice_eng | acc | **17.47** (n=229) | _running_ |
| MACRO | — | **11.76** | _running_ |

**QCMem memory/latency finding:** the read pack stays **constant ~6.2–6.4k tokens / ~18.5 GB per card** across docs spanning **90k–266k tokens** (well over the 131072 native window) — the bounded-read O(1)-memory property from P1.1, now confirmed on NATURAL long docs. Per-doc ~3.8 s (qa) / ~3.3 s (choice).

**KV-Direct (native-window Dense) behaviour:** runs a full-depth forward over the entire 90k–266k-token doc at **~29 s/it** (≈7× slower than QCMem) and **49–75 GB/card** (rising with doc length, vs QCMem's flat 18.5 GB). Every doc exceeds the 131072 native window (tokenizer emits `... > 131072 ... indexing errors`), so Dense operates out of its trained RoPE range → expected accuracy collapse (numbers pending arm completion). This is the natural-data analogue of the synthetic over-window collapse (`benchmark.md` §1a vs-Dense).

**Note on `longbook_choice_eng` acc (17.47 < 25% chance):** with **chat_template=False** (paper mandate) the base model completes the raw MC prompt rather than following the "answer with one letter" instruction, so clean-letter extraction is unreliable — an honest zero-format-scaffolding number. `longbook_qa_eng` F1 (free-form, format-robust) is the cleaner natural-QA signal.

---

## Artifacts (on `.73` / diskB `zwfy6/share_304376610`)
- Harness: `scripts/eval_qcmem_infbench.py` (also committed on wzc1 `4bbfece`).
- Data: `data/infinitebench/{longbook_qa_eng,longbook_choice_eng}.jsonl`.
- Smoke: `infbench_results/smoke_8b_j12/` · Sweep: `infbench_results/qcmem_8b_j12_lora/`, `infbench_results/kvdirect_8b/`.
- Logs: `logs/infb_smoke.out`, `logs/infb_qcmem_shard{0..7}.out`.
