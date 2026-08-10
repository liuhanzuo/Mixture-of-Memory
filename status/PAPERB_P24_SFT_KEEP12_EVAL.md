# Paper B P2.4 — keep12 POST-SFT Eval Battery on .104

**Date:** 2026-08-08
**Node:** .104 8×H20 (zwfy6 disk)
**Arm:** OLMo-2 7B keep=12 fresh=2 (14L pruned) SFT-continued
**Status:** PPL ✅ DONE @ 07:03:50; downstream RUNNING (core6 in flight, then know5/MMLU-dual/closedbook)

## ★ Headline PPL result

- Pre-SFT PPL = **11.4425** (anchor `7B_keep12_step124000_v2`)
- Post-SFT PPL = **12.4801** (merged 8/8 shards, n_tokens=8384512, n_windows=4096)
- **ΔPPL% observed = +9.07%**
- Prediction was **+10.98%** ⇒ **miss = −1.91 pp** (same direction as keep8/keep10 misses; magnitude smaller)

## Prediction audit — full n=5 table (all misses same direction)

| arm | pre PPL | post PPL | ΔPPL% obs | ΔPPL% pred (n=3 fit) | miss |
|---|---|---|---|---|---|
| keep8 | 13.3329 | 14.6857 | **+10.15%** | +14.0% | −3.85 pp |
| keep10 | 12.8159 | 13.9221 | **+8.63%** | +13.18% | −4.55 pp |
| keep12 | 11.4425 | 12.4801 | **+9.07%** | +10.98% | −1.91 pp |

- **Non-monotone**: keep8 (10.15) > keep12 (9.07) > keep10 (8.63). Damage-sensitivity is NOT monotone in keep-depth over this ladder.
- **All three misses are in the same direction** (predictions over-estimate ΔPPL%). The n=3 linear fit was systematically biased upward.
- **Conclusion**: the linear damage-sensitivity fit is dead (already declared dead after keep10). keep12 completes the n=5 Table 4 SFT sweep; it does NOT resurrect the linear claim.

## Provenance

- **Pre-SFT anchor**: `olmo2_ppl_results/7B_keep12_step124000_v2` on zwfy6
  - PPL = **11.4425** (verified from `summary.json`: sum_nll=20435867.05, n_tokens=8384512, n_windows=4096)
  - core6 = **0.56888** (hellaswag 0.4572 / arc_challenge 0.3780 / arc_easy 0.7239 / piqa 0.7323 / winogrande 0.6054 / openbookqa 0.2800)
  - ⚠️ **Uses `_v2` not the non-v2 keep12 dir** — the non-v2 (`7B_keep12_step124000`) has a 6/8-shard partial merge on arc_easy (n_scored=1782 vs expected 2376; see `status/PAPERB_TABLE4_KEEP12_PARTIAL_MERGE.md`). v2 is shard-integrity-clean (all core6 n_scored match expected).
  - per-item preds verified present: `per_example_hellaswag.jsonl`, `per_example_mmlu.jsonl`, `per_example_popqa.jsonl`, `per_example_triviaqa.jsonl`.
- **Post-SFT ckpt**: `outputs/olmo2_p24_sft_keep12fresh2/final.pt` (43.87 GB, step=842, no NaN, final training step loss=1.2483, finished 2026-08-08 06:32:28 CST)
- **Output name**: `7B_p24_sft_keep12fresh2_final` (+`_know`)

## Launch verification

- Wrapper PID `3291603` (bash) alive at 06:58:27; 8 shard workers `3291613-3291620` alive on GPU 0-7.
- Log `logs/p24_eval_sft_keep12_104.log` shows anchor gate passed and PPL section started.
- Shard 0 log confirms tensor shape: `val=data/dolmino_now_val.npy shape=(4096, 2048) | shard 0/8 -> 512 windows (seq_len=2048) batch_size=4` — matches keep8/keep10 batteries.
- `chat_template=False`, `--add_bos 0`, `LOCAL_RANK=0 RANK=$g` per shard (byte-identical to keep10 driver).
- `assert_8shards` gate enforced before every merge across all 5 harnesses (this exact check caught the keep12 v1 partial-merge; do not skip).

## Prediction and audit posture

- **n=3 linear fit prediction**: ΔPPL% ≈ **+10.98%** ⇒ post-PPL ~12.70.
- **This prediction will almost certainly miss** — keep8 predicted +14.0 came in at +10.15 (miss −3.85pp), keep10 predicted +13.2 came in at +8.63 (miss −4.55pp, also broke monotonicity). The linear damage-sensitivity fit is already falsified twice tonight.
- **Reporting purpose has shifted**: keep12 completes the n=5 Table 4 SFT sweep; it does NOT resurrect the linear claim. Report against +10.98% as-observed.

## Battery contents (5 harnesses × 8 GPU shards each)

| harness | output dir | per-example |
|---|---|---|
| PPL (dolmino_now_val) | `olmo2_ppl_results/7B_p24_sft_keep12fresh2_final` | — |
| downstream core6 | `olmo2_downstream_results/7B_p24_sft_keep12fresh2_final` | ✅ per-item |
| downstream know5 | `olmo2_downstream_results/7B_p24_sft_keep12fresh2_final_know` | ✅ per-item |
| MMLU letter+content | `olmo2_mmlu_content_results/7B_p24_sft_keep12fresh2_final` | ✅ per-item |
| closedbook PopQA+TriviaQA | `olmo2_closedbook_results/7B_p24_sft_keep12fresh2_final` | ✅ per-item |

## Ledger entries

- `status/gpu_runs.jsonl` — appended `{"node":".104","exp":"paperB_p24_sft_keep12_eval","commit_hash":"e575f14","status":"running",...}`
- `status/GPU_STATUS.md` — .104 block updated to reflect keep12 SFT DONE + eval battery RUNNING with PID `3291603`.

## Post-completion action items (do NOT run auto)

1. Fill in observed ΔPPL% vs +10.98% prediction.
2. Item-level paired analysis (McNemar + 10k bootstrap) against pre anchor `7B_keep12_step124000_v2` per-item preds — script pattern in `scripts/paired_analysis_p24_sft_keep8.py` (mirror it as `paired_analysis_p24_sft_keep12.py`).
3. Cross-check whether keep12 breaks monotonicity too (keep10 did). n=5 sweep complete after this.
