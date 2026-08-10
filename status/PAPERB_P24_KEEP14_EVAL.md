# Paper B P2.4 — keep14fresh2 pre/post-SFT eval battery (.73)

> Started 2026-08-08 01:26:36 +08:00 on `.73` (8×H20, zwfy6 disk).
> Launcher: `scripts/_run_olmo2_p24_eval_keep14_73.sh` (md5 wzc1 == zwfy6 == `7bc763ef38d591bbca289ff49564bc2a`).
> Log: `logs/p24_eval_keep14_73.log`.
> Report generated at 01:32 (after PART A PPL merge; downstream shards actively spinning).

## What this covers

**Only the keep14fresh2 arm's pre/post-SFT eval battery on `.73`.** The other two P2.4 arms (`full32`, `shortgpt16`) get analogous evals on `.252` in a separate dispatch. This script does NOT touch them.

## Ckpts (verified present on zwfy6)

| leg | ckpt | size | source |
|---|---|---|---|
| pre-SFT | `outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt` | 16.2 GB | healed pruned base (paper Table 4 anchor) |
| post-SFT | `outputs/olmo2_p24_sft_keep14fresh2/final.pt` | 48.7 GB | SFT trained 01:07 CST 2026-08-08, step=842, no NaN, loss@840=1.19 |

## Loud finding — pre-SFT anchor split across disks

The pre-SFT anchor `7B_keep14_step200000{,_know}` summary.json for **PPL + core6 + know5** **only exists on wzc1** (not zwfy6). Wzc1 numbers match paper Table 4 exactly:

- **PPL = 10.561295** (paper Table 4: 10.561 ✓)
- **core6 avg = 0.5938** (paper Table 4: .5938 ✓)
  - hs 0.6446 / arcc 0.4377 / arce 0.7050 / piqa 0.7454 / obqa 0.4040 / wg 0.6259
- **MMLU = 0.3191** (paper Table 4: .3191 ✓)

Pre-SFT `mmlu_content_results/7B_keep14_step200000/` and `closedbook_results/keep14_step200k/` **do exist on zwfy6**.

**Consequence**: for same-disk paired pre/post analysis (McNemar/paired bootstrap require per-item file colocation on one filesystem), PART A of the script re-runs PPL + core6 + know5 on zwfy6. Confirmed the harness path has not drifted: PART A PPL rerun already merged (2:08 wall) and reported **PPL = 10.5611512**, matching wzc1 anchor 10.5612951 to the 4th decimal (diff = 1.4e-4, within expected shard-order/reduction float noise).

MMLU-content and closedbook `pre-SFT` are auto-skipped (summary.json guard) since they exist on zwfy6.

## Harness invariants (all enforced)

- **8-shard rule**: `CUDA_VISIBLE_DEVICES=$g` + `LOCAL_RANK=0 RANK=$g` for every shard.
- **Merge gate**: `assert_8shards <root> <name> shard{i}of8.json` runs before every merge. If any shard missing → abort merge, log the miss, fail loud. **No partial merge = no silent 5/8 contamination.**
- **`chat_template=False`** throughout (project paper hard rule; OLMo-2 base has no SFT/RL). All eval scripts default to `--add_bos 0` (add_special_tokens=False), preserving the base-protocol tokenisation used by the Table 4 anchor.
- **Per-item predictions retained** for pre/post pairing:
  - downstream MC: `--save_per_example` → `per_example_<task>.jsonl` after merge.
  - MMLU content: `per_example_mmlu.jsonl` (default in harness).
  - closed-book: `per_example_<task>.jsonl` (default in harness).
- **Arch meta from ckpt**: `load_pruned_model` reads `keep_front_layers=14, n_fresh_layers=2` directly from `final.pt` meta (the SFT trainer at line 172 of `train_olmo2_sft.py` writes these); CLI-passed values (none passed) would be checked for mismatch.
- **Model construction shared**: PPL, downstream MC, MMLU-content, closed-book all import `load_pruned_model` from `eval_olmo2_probe2_ppl.py` → NO arch drift across the four harnesses.

## Output layout

| harness | pre-SFT (`7B_keep14_step200000`) | post-SFT (`7B_p24_sft_keep14fresh2_final`) |
|---|---|---|
| PPL (held-out Dolmino) | `olmo2_ppl_results/<name>/summary.json` | same tree |
| core6 (hs/arcc/arce/piqa/wg/obqa, acc_norm & acc) | `olmo2_downstream_results/<name>/summary.json` | same |
| know5 (mmlu/lambada/boolq/csqa/siqa) | `olmo2_downstream_results/<name>_know/summary.json` | same |
| MMLU dual (letter + content_norm/raw, McNemar, paired bootstrap on the diff) | `olmo2_mmlu_content_results/<name>/summary.json` | same |
| PopQA + TriviaQA closed-book (EM/contains/F1) | `olmo2_closedbook_results/<name>/summary.json` | same |

Naming matches paper Table 4 anchor convention. **NQ-open** is deliberately skipped (harness supports it via `--tasks nq_open` but spec marked optional). To add later: rerun closed-book with `--tasks nq_open --output_name <name>_nqopen`.

## Chat template & SFT contamination caveat

- `chat_template=False` — every harness call verified in the script; matches memory `paper-eval-chat-false-mandatory` and paper hard rule.
- SFT-data contamination audit (sibling agent): **Tulu-3 general-clean vs MMLU test = 45 hits**; vs PopQA/TriviaQA/NQ-open = 0. **The MMLU post-SFT number reported in the paper must be recomputed on the clean subset** (drop those 45 items, rescore). This is CPU-only post-processing that runs off the `per_example_mmlu.jsonl` files (already being retained). Closed-book numbers are contamination-free.

## Timing

- PART A pre-SFT PPL: **2:08 wall** (already DONE).
- PART A pre-SFT downstream core6/know5: ~15-25 min each (currently running, 8/8 GPUs @ 60-80% util, 24 GiB each).
- PART A pre-SFT MMLU-content, closedbook: auto-skipped (already on zwfy6).
- PART B post-SFT (5 harnesses × ~15-25 min): **~1.5-2 h**.
- **Overall ETA: 2026-08-08 03:00-03:30 +08:00**.

## Reproducibility

- Launcher tracked in git (working tree; separate commit will follow if needed).
- Commit hash at launch: `c345a00`.
- gpu_runs.jsonl entry appended.
- GPU_STATUS.md .73 block updated.
- Per-item jsonl written per shard, merged & sorted by `item_id` on `--merge` (see downstream/mmlu_content/closedbook merge sections).

## No touching (per dispatch spec)

- `.tex` files: not touched.
- `versions/*.md`: not touched.
- `paperB/TODOList.md`: not touched.
- Sibling `PAPERB_P24_*_ARM.md`: not touched (`.252` agents' territory).
