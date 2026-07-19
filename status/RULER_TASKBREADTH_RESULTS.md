# RULER Task-Breadth: CoMem vs InfLLM (Qwen3-8B, apples-to-apples)

**Run date (real, on .73):** Sun Jul 19 11:20:09 PM CST 2026
**Node:** .73 = `28.85.35.73` (port 36000), 8× H20, **EVAL-ONLY** (no training placed on node)
**Backbone:** Qwen3-8B (36 layers), `/apdcephfs_zwfy6/.../Qwen--Qwen3-8b`
**Purpose:** Paper A's RULER comparison covered only 3 task types (niah_single_2,
niah_multikey_1, variable_tracking). This adds **4 NEW storyful RULER task types**
for **CoMem (ours)** and the **strongest baseline InfLLM** on identical
tasks/lengths/samples — apples-to-apples.

## Protocol (identical for both methods)

- **Length grid:** {16k, 64k, 128k}. **n = 100/cell.**
- **Sharding:** 8-shard, one shard per GPU (`--num_shards 8 --shard_index k`,
  `CUDA_VISIBLE_DEVICES=k`), samples `[k::8]` → per-shard counts
  `[13,13,13,13,12,12,12,12]` (sum 100).
- **`PYTHONHASHSEED=0`** → `hash((task,length))` deterministic → **both methods and
  all shards share the identical sample set** per cell.
- **Decoding:** greedy, `max_new_tokens=128` (raised from 48 — 48 truncated UUIDs and
  multi-value answers mid-string, a false-low artifact; 128 fully contains every
  answer type: a UUID is ~36 chars ≈ 18 tokens).
- **Scorer:** official RULER `string_match_all` recall
  (`sum(1 for r in refs if r.lower() in pred.lower())/len(refs)`). **Never `re.search`.**

**CoMem config (user's standing directive — enforced):** `--resume_j 12
--lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final --selector iter_bm25
--topk 12 --sink_tokens bos --chunk_size 512 --use_chat_template` (chat_template=ON,
no-think). Constant read budget ≈ 6657 tokens.

**InfLLM config (paper-faithful):** `block_size=128, n_init=128, n_local=4096,
topk=16, repr_topk=4, chunk_size=8192, base=1e6`, eager attention.

## New task types (4)

| task | haystack | # keys | # values/key | # queries | value type |
|------|----------|--------|--------------|-----------|-----------|
| niah_single_1  | noise (rep. sentence) | 1 | 1 | 1 | 7-digit number |
| niah_single_3  | essay (Paul Graham)   | 1 | 1 | 1 | **UUID (36-char)** |
| niah_multivalue| essay                 | 1 | **4** | 1 | 7-digit number |
| niah_multiquery| essay                 | **4** | 1 | **4** | 7-digit number |

(niah_single_2 + niah_multikey_1 already exist in the framework / Paper A; not re-run.)

## Results — recall %, n=100/cell (independently recomputed)

| task | len | **CoMem** | **InfLLM** | Δ (CoMem−InfLLM) |
|------|-----|----------:|-----------:|-----------------:|
| niah_single_1  | 16k  | 100.00 | 100.00 |  0.00 |
| niah_single_1  | 64k  |  99.00 | 100.00 | −1.00 |
| niah_single_1  | 128k |  99.00 | 100.00 | −1.00 |
| niah_single_3 (uuid) | 16k  |  90.00 |  95.00 | −5.00 |
| niah_single_3 (uuid) | 64k  |  98.00 |  25.00 | **+73.00** |
| niah_single_3 (uuid) | 128k |  97.00 |   5.00 | **+92.00** |
| niah_multivalue | 16k  |  94.50 |  81.25 | +13.25 |
| niah_multivalue | 64k  |  92.50 |  36.25 | **+56.25** |
| niah_multivalue | 128k |  95.25 |  23.00 | **+72.25** |
| niah_multiquery | 16k  |  97.50 |  87.50 | +10.00 |
| niah_multiquery | 64k  |  94.75 |  35.00 | **+59.75** |
| niah_multiquery | 128k |  97.00 |  18.75 | **+78.25** |

### Story
- On the **easy** task (niah_single_1, plain 7-digit number in noise) both methods are
  saturated: InfLLM 100/100/100, CoMem 100/99/99. The needle is short and any
  block-memory that lands on the right block copies it verbatim.
- On the **harder** tasks — **UUID needle** (hard to copy), **multi-value** (4 answers
  per key), **multi-query** (4 keys/answers) — **InfLLM collapses as context grows**
  (128k: 5 / 23 / 18.75), while **CoMem stays 90–99% flat across all lengths**.
- **Mechanism (verified by eyeballing preds):** InfLLM's misses are *genuine*, not
  artifacts — outputs are well-formed full sentences (mean 172 chars, never
  empty/truncated) that retrieve the *start* of the UUID but hallucinate/corrupt the
  middle (e.g. target `46a45115-784e-4e88-95bd-…` → output
  `46a45115-784e-4e88-910a-…`). Its top-k block selection surfaces the needle's
  neighbourhood but the full 36-char string isn't cleanly inside the retained KV, so
  it drifts. CoMem's write/read resume reconstructs the full needle faithfully under a
  constant ~6.6k read budget.
- **Takeaway for the paper:** the 3-task RULER slice understated CoMem's advantage. On
  copy-hard / multi-answer needles at ≥64k, CoMem beats the strongest training-free
  baseline by **+56 to +92 recall points**, while remaining within ~1 pt on the trivial
  case.

## 铁律2 verification — ALL 24 CELLS PASS

Each cell independently recomputed after 8-shard merge:

- **n = 100** for every cell (12 CoMem + 12 InfLLM).
- **Per-shard counts = `[13,13,13,13,12,12,12,12]`** for every cell → **no dup / no miss**
  (shards partition `range(100)[k::8]`, union = all 100).
- **empty_output = 0**, **[OOM] = 0** for every cell.
- **Recomputed recall == driver-stored recall** exactly for every cell (independent
  re-score with official `string_match_all` agrees with the driver → scorer integrity).
- **uniq_outputs = 100** for every cell → non-degenerate (no collapsed/repeated preds);
  low-recall cells confirmed to be genuine wrong answers, not empty/looping.

## Skipped tasks (noted, not deadlocked)

Standard RULER has 13 task types. This run covers 4 new + 3 existing (7 total). The
remaining 6 were skipped:

| task | reason skipped |
|------|----------------|
| niah_multikey_2, niah_multikey_3 | Need a distinct "needle"-style haystack builder (gold + hard distractor key/value strings interleaved into the haystack); `_make_niah` only builds the essay/noise + injected-KV variants. Would require a separate generator — out of scope for this eval-only breadth pass. |
| cwe (common word extraction), fwe (frequent-word extraction) | Different template (word-frequency counting, not needle-in-haystack); driver's NIAH branch does not cover them. |
| qa_1 (SQuAD), qa_2 (HotpotQA) | Require external QA datasets (SQuAD / HotpotQA) not present in the tree. |

## Raw output paths (on .73, diskB `share_304376610`)

- CoMem:  `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/ruler_results/qcmem_8b_taskbreadth_iter_ad/qcmem_8b_taskbreadth_iter_ad/`
- InfLLM: `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/ruler_results/infllm_8b_taskbreadth/infllm_8b_taskbreadth/`
- Per-cell files: `{task}_{length}_shard{k}of8.{csv,json}` (96 CSV each method).
- Logs: `logs/tb_qcmem_shard{k}.log`, `logs/tb_infllm_shard{k}.log`,
  `logs/tb_taskbreadth_master.log`; DONE markers `logs/tb_{qcmem,infllm,ALL}_DONE`.

## GPU release

All 8 GPUs on .73 released after ALL_DONE: `0 MiB / 0 % / 0 compute-apps` per GPU.
