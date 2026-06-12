# Eval Audit

Generated: 2026-05-12

This file is an artifact-based audit of evals already present in the repo. I did **not** trust sweep names like `full` or status labels like `running` by themselves; status below is assigned from actual result files, logs, and status records.

## Audit rules

- **completed**: the expected artifact matrix for that eval family/scope exists
- **partial**: some artifacts exist, but the matrix is incomplete, or only smoke/dryrun scope exists
- **stale status**: status files still say `running`, but artifacts show the eval later finished
- **not found**: no eval artifact found in the audited roots

Audited roots:

- `Mixture-of-Memory/babilong_results/`
- `Mixture-of-Memory/outputs/niah*/niah_results.json`
- `Mixture-of-Memory/status/ACTIVE_SWEEPS.jsonl`
- `Mixture-of-Memory/status/gpu_runs.jsonl`
- `Mixture-of-Memory/status/RESEARCHER_REPORTS.jsonl`
- `Mixture-of-Memory/logs/*babilong*.log`
- `Mixture-of-Memory/logs/*niah*.log`

`proxy_match_acc` below is a **rough audit metric** computed from CSV outputs by checking whether the gold target string appears in the model output. It is useful for rough ranking and for spotting obvious failures, but it is **not** an official paper metric.

---

## 1) Baselines already present

| Family | Benchmark | Status | Proven scope | proxy_match_acc | Evidence |
|---|---|---:|---|---:|---|
| Meta-Llama-3-8B | BABILong | completed | `qa1..qa5 × {0k,1k,2k,4k,8k,16k,32k}` = 35 csv + 35 json | 0.0309 | `status/ACTIVE_SWEEPS.jsonl:172-173`; example artifact: `babilong_results/Meta-Llama-3-8B/qa5_32k_instruction_yes_examples_yes_post_prompt_yes_chat_template_no_system_prompt_no.csv` |
| Meta-Llama-3.2-1B | BABILong | completed | `qa1..qa5 × {0k,1k,2k,4k,8k,16k,32k}` = 35 csv + 35 json | 0.2486 | artifact-complete; example: `babilong_results/Meta-Llama-3.2-1B/qa5_32k_instruction_yes_examples_yes_post_prompt_yes_chat_template_no_system_prompt_no.csv` |
| Llama-3.2-1B-Instruct | BABILong | completed outputs / metadata partial | csv scope is complete for `qa1..qa10 × {1k,2k,4k,8k,16k,32k}` = 60 csv; json sidecars missing for `qa5..qa10 @ 32k` (54 json total) | 0.3915 | `status/ACTIVE_SWEEPS.jsonl:178` still says rerun `running`, but artifacts now exist; example: `babilong_results/Llama-3.2-1B-Instruct/qa10_32k_instruction_yes_examples_yes_post_prompt_yes_chat_template_yes_system_prompt_no.csv` |
| Beacon-Qwen2-7B | BABILong | completed | `qa1..qa10 × {1k,2k,4k,8k,16k,32k}` = 60 csv | 0.5551 | artifact-complete; example: `babilong_results/Beacon-Qwen2-7B/qa10_32k_instruction_yes_examples_yes_post_prompt_yes_chat_template_yes_system_prompt_no.csv` |
| Beacon-Qwen2-7B-full-repro | BABILong | completed (narrow repro scope) | `qa1..qa5 × {0k,1k,2k,4k,8k}` = 25 csv | 0.6092 | `logs/beacon_babilong_full_b2004.log:220-4746`, especially `:4746` DONE; example artifact: `babilong_results/Beacon-Qwen2-7B-full-repro/qa5_8k_instruction_yes_examples_yes_post_prompt_yes_chat_template_yes_system_prompt_no.csv` |
| MemoryLLM-8B-chat | BABILong | partial | 55/60 csv present for `qa1..qa10 × {1k,2k,4k,8k,16k,32k}`; missing `qa3@32k`, `qa4@16k`, `qa4@32k`, `qa5@16k`, `qa5@32k` | 0.3620 | artifact-only; example: `babilong_results/MemoryLLM-8B-chat/qa10_8k_instruction_yes_examples_yes_post_prompt_yes_chat_template_yes_system_prompt_no.csv` |
| MemoryLLM-8B-chat-repro | BABILong | completed (narrow repro scope) | `qa1..qa5 × {0k,1k,2k,4k}` = 20 csv | 0.3625 | `logs/memoryllm_babilong_full_b2003.log:234-4160`, especially `:4160` DONE |
| LM2 iter8000 | BABILong | completed (status stale) | `qa1..qa5 × {0k,1k,2k,4k,8k,16k,32k}` = 35 csv + 35 json | 0.0163 | artifacts complete; multi-ckpt launch still recorded as running at `status/ACTIVE_SWEEPS.jsonl:175` |
| LM2 iter10000 | BABILong | completed (status stale) | same full 35 csv + 35 json matrix | 0.0086 | artifacts complete; `status/ACTIVE_SWEEPS.jsonl:175` |
| LM2 iter12000 | BABILong | completed (status stale) | same full 35 csv + 35 json matrix | 0.0131 | `status/ACTIVE_SWEEPS.jsonl:174` still says running; artifacts complete now |
| LM2 iter14000 | BABILong | completed (status stale) | same full 35 csv + 35 json matrix | 0.0077 | artifacts complete; `status/ACTIVE_SWEEPS.jsonl:175` |
| LM2 iter16000 | BABILong | completed (status stale) | same full 35 csv + 35 json matrix | 0.0069 | artifacts complete; `status/ACTIVE_SWEEPS.jsonl:175` |
| MemLong | BABILong / NIAH | not found | no eval artifact found in audited roots | — | `status/gpu_runs.jsonl:461-464` shows training status only (stage1 completed, stage2 running), not eval output |

### Baseline notes

1. `Beacon-Qwen2-7B-full-repro` is **not** the same scope as the broader `Beacon-Qwen2-7B` baseline. The log-backed repro is only `qa1..qa5 × 0k..8k`, even though the H20 launcher was broader.
2. `MemoryLLM-8B-chat-repro` is also a narrower repro than the broader `MemoryLLM-8B-chat` directory.
3. LM2 status files are stale: `status/ACTIVE_SWEEPS.jsonl:174-175` still say `running`, but every checkpoint directory now has the full 35 csv + 35 json BABILong matrix.

---

## 2) Our BABILong evals already present

### H-series / project checkpoints

All of the following have a **full BABILong artifact matrix** of `qa1..qa5 × {0k,1k,2k,4k,8k,16k,32k}` = 35 csv + 35 json, unless noted otherwise.

Research note `status/RESEARCHER_REPORTS.jsonl:74` explains the high-level result: these H-series runs are effectively near-zero on BABILong because they inherit the weak BABILong behavior of the Llama-3-8B base model.

| Checkpoint | Status | Proven scope | proxy_match_acc | Evidence |
|---|---:|---|---:|---|
| H-step5000 | completed | full 35 csv + 35 json | 0.0026 | example: `babilong_results/H-step5000/qa5_32k_instruction_yes_examples_yes_post_prompt_yes_chat_template_no_system_prompt_no.csv` |
| H2-step5000 | completed | full 35 csv + 35 json | 0.0029 | example artifact in `babilong_results/H2-step5000/` |
| H3-step5000 | completed | full 35 csv + 35 json | 0.0029 | example artifact in `babilong_results/H3-step5000/` |
| H4-step3000 | completed | full 35 csv + 35 json | 0.0020 | example artifact in `babilong_results/H4-step3000/` |
| H5-step2000 | completed | full 35 csv + 35 json | 0.0017 | example artifact in `babilong_results/H5-step2000/` |
| H5b-step2000 | completed | full 35 csv + 35 json | 0.0020 | example artifact in `babilong_results/H5b-step2000/` |
| H6-step1000 | completed (status stale) | full 35 csv + 35 json | 0.0020 | `status/ACTIVE_SWEEPS.jsonl:176` still says running; artifacts are complete |
| H6b-step1000 | completed (status stale) | full 35 csv + 35 json | 0.0011 | `status/ACTIVE_SWEEPS.jsonl:177` still says running; artifacts are complete |
| H6-step5000 | completed | full 35 csv + 35 json | 0.0031 | example artifact in `babilong_results/H6-step5000/` |
| H6b-step5000 | completed | full 35 csv + 35 json | 0.0023 | example artifact in `babilong_results/H6b-step5000/` |
| H7-step500 | completed | full 35 csv + 35 json | 0.0017 | example artifact in `babilong_results/H7-step500/` |
| H7-step1000 | completed | full 35 csv + 35 json | 0.0023 | example artifact in `babilong_results/H7-step1000/` |
| H7-step1500 | completed | full 35 csv + 35 json | 0.0026 | example artifact in `babilong_results/H7-step1500/` |
| H7-step2000 | completed | full 35 csv + 35 json | 0.0034 | example artifact in `babilong_results/H7-step2000/` |
| H7-step2500 | completed | full 35 csv + 35 json | 0.0034 | example artifact in `babilong_results/H7-step2500/` |
| H7-step3000 | completed | full 35 csv + 35 json | 0.0031 | example artifact in `babilong_results/H7-step3000/` |
| H6-step1000-dryrun | partial / dryrun | only `qa1@0k` = 1 csv + 1 json | — | `babilong_results/H6-step1000-dryrun/` |

### MPlus / M+ smoke BABILong runs

| Family | Status | Proven scope | proxy_match_acc | Evidence |
|---|---:|---|---:|---|
| MPlus-8B-smoke | completed smoke only | single example only: `qa1@0k` (1 csv row) | 0.0000 | `logs/mplus_babilong_smoke_postfix_20260511.log:57`; csv output is literal `!!!!!!!!!!!!!!!!!!!!` |
| MPlus-8B-smoke-plainprompt | completed smoke only | single example only: `qa1@0k` (1 csv row) | 0.0000 | `logs/mplus_babilong_smoke_plainprompt_20260511.log:57`; csv output is literal `!!!!!!!!!!` |
| MPlus-8B-smoke-readme-prompt | completed smoke only | single example only: `qa1@0k` (1 csv row) | 0.0000 | `logs/mplus_babilong_smoke_readme_prompt_20260511.log:56`; csv output is literal `!!!!!!!!!!!!!!!!!!!!` |

Conclusion: only smoke artifacts exist for MPlus right now; I do **not** see a full MPlus BABILong eval yet.

---

## 3) NIAH / NIH-style evals already present

Common scope for the `niah_results.json` files below: `context_lengths = {8192,16384,32768}`, `depths = {0.1,0.3,0.5,0.75}`, `num_samples = 5`, so **60 total samples** per eval.

### 3.1 Earlier runs with explicit status explanations

| Eval | Status | Result | Evidence |
|---|---:|---|---|
| niah_mem_space_champion | completed but failed | `0/60 = 0.0%` | `logs/niah_with_memory_20260427_1353.log:149,158,161`; `outputs/niah_mem_space_champion/niah_results.json:103-110` |
| niah_bypass | completed but failed | `0/60 = 0.0%` | `logs/niah_bypass_20260427_1354.log:160`; `outputs/niah_bypass/niah_results.json:103-110` |
| niah_mem_space_v2 | completed but **invalid / discard** | `0/60 = 0.0%` | `status/ACTIVE_SWEEPS.jsonl:155-160`; `logs/niah_mem_space_v2_20260427_142316.log:161`; `outputs/niah_mem_space_v2/niah_results.json:103-110` |
| niah_bypass_v2 | completed but **invalid / discard** | `0/60 = 0.0%` | `status/ACTIVE_SWEEPS.jsonl:153,157,159`; `logs/niah_bypass_v2_20260427_141803.log:161`; `outputs/niah_bypass_v2/niah_results.json:103-110` |
| niah_mem_space_v3 | completed but **invalid / discard** | `0/60 = 0.0%` | `status/ACTIVE_SWEEPS.jsonl:161,163`; reason given there is GPU-distribution + generation bug |
| niah_bypass_v3 | completed but **invalid / discard** | `0/60 = 0.0%` | `status/ACTIVE_SWEEPS.jsonl:162,164`; reason given there is generation bug |
| niah_mem_space_v4 | completed but **invalid / discard** | `0/60 = 0.0%` | `status/ACTIVE_SWEEPS.jsonl:167`; `outputs/niah_mem_space_v4/niah_results.json` |
| niah_bypass_v4 | completed but **invalid / discard** | `0/60 = 0.0%` | `status/ACTIVE_SWEEPS.jsonl:168`; `status/gpu_runs.jsonl:299`; `outputs/niah_bypass_v4/niah_results.json` |

### 3.2 Later output-backed reruns (artifact exists even when status is stale/missing)

| Eval | Status | Result | Evidence |
|---|---:|---|---|
| niah_mem_space_v5 | completed output | `0/60 = 0.0%` | `outputs/niah_mem_space_v5/niah_results.json` |
| niah_bypass_v5 | completed output | `0/60 = 0.0%` | `status/ACTIVE_SWEEPS.jsonl:170` still says pending, but `outputs/niah_bypass_v5/niah_results.json` exists |
| niah_mem_space_v6 | completed output | `0/60 = 0.0%` | `outputs/niah_mem_space_v6/niah_results.json` |
| niah_bypass_v6 | completed output | `0/60 = 0.0%` | `outputs/niah_bypass_v6/niah_results.json` |
| niah_mem_space_v7 | completed output | `0/60 = 0.0%` | `outputs/niah_mem_space_v7/niah_results.json` |
| niah_bypass_v7 | completed output (status stale) | `20/60 = 33.33%`; `8192 -> 1.0`, `16384 -> 0.0`, `32768 -> 0.0` | `status/gpu_runs.jsonl:303-305` still says running; `outputs/niah_bypass_v7/niah_results.json:103-110` shows the completed result |
| niah_mem_space_v8 | completed output | `0/60 = 0.0%` | `outputs/niah_mem_space_v8/niah_results.json:103-110` |

### NIAH note

The NIAH tree contains several generations of runs. For v2/v3/v4, the status files explicitly say the 0% results are invalid and should be discarded. For v5/v6/v7/v8, output JSONs exist, but status bookkeeping is incomplete/stale. The only clearly non-zero later NIAH result I found is **`niah_bypass_v7 = 20/60 = 33.33%`, and that only at 8k context**.

---

## 4) Things that were mentioned before but are still not actually present as eval artifacts here

These are the items I do **not** see as finished eval artifacts in the audited roots:

- **MemLong BABILong/NIAH eval**: training exists, but no BABILong or NIAH eval artifact found here
- **ARMT eval**: no BABILong/NIAH artifact found here
- **HMT eval**: no BABILong/NIAH artifact found here
- **full MPlus BABILong eval**: not found; only the 1-example smoke runs exist

---

## 5) Short takeaways

1. The clearly finished baseline BABILong artifacts are: **Meta-Llama-3-8B**, **Meta-Llama-3.2-1B**, **Beacon-Qwen2-7B**, **Beacon-Qwen2-7B-full-repro**, **MemoryLLM-8B-chat-repro**, and **all LM2 checkpoints 8k/10k/12k/14k/16k**.
2. The clearly incomplete baseline artifact is: **MemoryLLM-8B-chat** (55/60 csv, five combinations missing).
3. The strongest baseline in the current BABILong artifact tree is still **Beacon-Qwen2-7B** / **Beacon-Qwen2-7B-full-repro** by rough CSV inspection.
4. Our H-series BABILong runs are artifact-complete for many checkpoints, but all are effectively near-zero.
5. For NIAH, the only later non-zero result I found is **`niah_bypass_v7 = 33.33%`**, and only at 8k context; all memory versions in this tree are still at 0%.
6. Some status files are stale. In particular, **LM2** and at least **H6/H6b step1000** still show `running` in status logs even though their result directories are complete.
