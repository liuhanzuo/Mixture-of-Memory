# P1.3 — MemoryLLM LoCoMo under its OFFICIAL/native chat template (GPT-4o judge)

**Date:** 2026-08-01 · **Node:** `.82` = `28.82.250.82:36000`, diskB shared FS · **Owner:** P1.3 eval worker
**Deliverable:** close P1.3's last missing cell = MemoryLLM's LoCoMo score under its **official/native
chat template**, scored by the GPT-4o judge (the ONE prompt/chat-template-sensitivity diagnostic; the rest
of the paper is `chat_template=False`). This is **distinct from** the existing OOD no-chat MemoryLLM
diagnostic already in the appendix (`locomo_results/memoryllm_chatFALSE`, judge 16.11).

## Headline number

| MemoryLLM template variant | LoCoMo judge (gpt-4o, over 1986) |
|----------------------------|:--------------------------------:|
| **official / native chat template (this task)** | **14.75** |
| no-chat OOD diagnostic (already in appendix) | 16.11 |

**Finding:** MemoryLLM under its own official Llama-3 chat template scores **14.75** overall-judge on LoCoMo —
**slightly LOWER** than the no-chat OOD feed (16.11). i.e. wrapping the (already very long, injected-context)
LoCoMo prompt in the model's native chat template does not help MemoryLLM here; the sensitivity is small
(≈ −1.4 pp overall) and does not change the qualitative story (MemoryLLM's fixed-pool memory is far below
CoMem/KV-Direct on LoCoMo regardless of template).

## Pre-existing vs generated

**Predictions were PRE-EXISTING (zero new GPU work).** The official-template predictions already existed on
.82 diskB as `locomo_results/memoryllm_chatnothink/` (8 shards, all 1986 items), generated with
`use_chat_template=true` (`no_chat_template=false`). Verified in `scripts/eval_memoryllm_common.py::generate_answer`:
`use_chat_template=True` applies `tokenizer.apply_chat_template(...)` = MemoryLLM's **native Llama-3 chat
template**, with the README-recommended leading-BOS drop (`inputs[:, 1:]`, since MemoryLLM adds its own learned
bos-embedding at every layer). That IS the official template. (The `chatnothink` naming carries the QCMem
`enable_thinking=False` convention over from Qwen3; MemoryLLM/Llama has no think mode, so it is a plain native
chat-template run.)

Only the **GPT-4o judge** had never been run on it (its `scores.json` had F1/EM/acc but no `judge`).
So this task = judge-only, NO GPU. To keep a cleanly-labeled, self-contained P1.3 artifact I copied the
byte-identical preds + shard configs into a NEW dir `locomo_results/memoryllm_officialtmpl_locomo/` (the
no-chat diagnostic dir `memoryllm_chatFALSE` was NOT touched) and ran the judge there.

## Config

- Model: `baselines/memoryllm-8b-chat-hf` (MemoryLLM-8B), bf16, sdpa, chunk_size 1024, max_new_tokens 48, greedy.
- Template: **official native Llama-3 chat template** (`use_chat_template=true`, `enable_thinking` N/A), BOS dropped.
- Generation harness (that produced the preds): `scripts/eval_memoryllm_locomo.py`, 8-shard strided (`[i::8]`).
- Judge: `scripts/eval_qcmem_locomo.py --score_only --use_llm_judge` — the **exact same** shared judge
  (`llm_judge_preds` / `_JUDGE_TEMPLATE` / `run_scoring`) used for the CoMem/KV-Direct LoCoMo headline and the
  MemoryLLM no-chat diagnostic (`memoryllm_chatFALSE`, model=gpt-4o, confirmed via its `judge_cache.jsonl`).
- Judge protocol (matches `paperA/sections/08_statistics_appendix.tex`): endpoint
  `https://maas-openapi.wanjiedata.com/api/v1/chat/completions` (via hy-proxy), model `gpt-4o`, `seed=1`,
  no client temperature/top_p, 4 retries w/ exponential backoff (2s→4s→8s…), unparsable/failed → scored WRONG.
- Denominator = all **1986** items; GPT-4o judges the **1540** answerable cat1–4 items; **cat5 = 446**
  adversarial-abstention items scored **locally** (correct iff pred empty or matches the refusal regex).
- **0 judge-API failures** (1540/1540 cat1–4 verdicts cached in `judge_cache.jsonl`).

## Scores (gpt-4o judge)

| category | n | judge |
|----------|---:|------:|
| cat1 (single-hop) | 282 | 13.12 |
| cat2 (multi-hop) | 321 | 9.03 |
| cat3 (temporal) | 96 | 13.54 |
| cat4 (open-domain) | 841 | 25.09 |
| **cat1–4 answerable (judged)** | **1540** | (per-cat above) |
| cat5 (adversarial abstention, local) | 446 | 0.67 |
| **OVERALL (over 1986)** | **1986** | **14.75** |

For reference (auxiliary metrics, unchanged from the pre-existing run): overall F1 = 9.93, EM = 0.96, acc = 9.72.

## Raw artifact paths (all on .82 diskB `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`)

- Scored (this task): `locomo_results/memoryllm_officialtmpl_locomo/scores.json`,
  `.../judge_cache.jsonl` (1540 gpt-4o verdicts), `.../preds_shard{0..7}of8.jsonl`,
  `.../eval_config_shard{0..7}of8.json`
- Source predictions (pre-existing, official template): `locomo_results/memoryllm_chatnothink/preds_shard*.jsonl`
- No-chat diagnostic (untouched, for contrast): `locomo_results/memoryllm_chatFALSE/scores.json` (judge 16.11)
- Judge log: `logs/p1_3_memoryllm_officialtmpl_judge.log`

## Exact commands

```bash
# on .82, diskB workdir, ZERO GPU (CUDA hidden so dllm's 8-GPU SFT job is never touched):
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/

# 1) copy pre-existing official-template preds into a clean P1.3 artifact dir
mkdir -p locomo_results/memoryllm_officialtmpl_locomo
cp -n locomo_results/memoryllm_chatnothink/preds_shard*.jsonl        locomo_results/memoryllm_officialtmpl_locomo/
cp -n locomo_results/memoryllm_chatnothink/eval_config_shard*.json   locomo_results/memoryllm_officialtmpl_locomo/

# 2) GPT-4o judge (endpoint + key + proxy read from ./.env by _load_dotenv)
CUDA_VISIBLE_DEVICES='' /opt/conda/envs/torch-base/bin/python scripts/eval_qcmem_locomo.py \
  --score_only \
  --output_dir locomo_results/memoryllm_officialtmpl_locomo \
  --use_llm_judge --judge_model gpt-4o --judge_workers 8
```

## Safety / cleanliness

- Judge is CPU/network only; ran with `CUDA_VISIBLE_DEVICES=''` → **never allocated any GPU**.
- Verified before + after: dllm's `dllm_draft` SFT job intact on all 8 GPUs (~17.3 GB/card, 98–99% util,
  unchanged). No orphan processes of mine remain.
- No `.tex` / `paperA/TODOList.md` edited (per task rules). Main folds the number in.
