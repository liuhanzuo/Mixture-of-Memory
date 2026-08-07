# Paper B P2.4 — full32 + shortgpt16 pre/post-SFT eval battery (.252, wzc1)

> Started 2026-08-08 02:02:07 +08:00 on `.252` (8×L20A cc10.0, wzc1 disk).
> Launcher: `scripts/_run_olmo2_p24_eval_full32_shortgpt_252.sh` (commit `0b2f707`).
> Log: `logs/p24_eval_full32_shortgpt_252.log`.
> PID: `3032751` (setsid nohup, detached).

## What this covers

**The two wzc1-side arms — `full32` and `shortgpt16` — pre-SFT AND post-SFT eval batteries.** The third P2.4 arm (`keep14fresh2`) is handled by the sibling dispatch on `.73` (`scripts/_run_olmo2_p24_eval_keep14_73.sh` / agent aefe8b20 / report `status/PAPERB_P24_KEEP14_EVAL.md`). This driver does NOT touch that arm's outputs.

## Why same-arch pairing is mandatory (spec context)

Earlier tonight MAIN discovered that **L20A cc10.0 and H20 cc9.0 give different core6 numbers for bit-identical checkpoints** (28-item flip across ~10k, +0.156 pp on the average; signs differ per task; symmetric bf16 kernel noise, not a directional hardware advantage; `n_nan=0` both sides; model_state SHA-256 identical). Full write-up: `status/PAPERB_CORE6_CROSSARCH_FLOOR.md`.

Consequence: **P2.4's per-item McNemar / paired bootstrap must be computed within a single architecture.** Pairing a wzc1 post-SFT against a zwfy6 pre-SFT would inject the ~0.156 pp hardware artifact into the reported SFT effect. That is why this driver runs BOTH pre-SFT and post-SFT on `.252`.

## Ckpts (verified present on wzc1, metadata inspected)

| leg | ckpt | size | source | key meta |
|---|---|---|---|---|
| full32 pre-SFT   | `../models/OLMo-2-1124-7B/` (HF safetensors) | 6-file base | vanilla base, same `--base_model` all three P2.4 arms trained from | 32L, `chat_template=False` protocol |
| full32 post-SFT  | `outputs/olmo2_p24_sft_full32/final.pt` | 87.6 GB | SFT 2026-08-08 00:38, step=842, from vanilla base | `keep_front_layers=32 n_fresh_layers=0 num_hidden_layers=32` |
| shortgpt16 pre-SFT | `outputs/olmo2_probe2_7B_shortgpt16/step200000.pt` | 48.7 GB | healed ShortGPT-16 (paper Table 4 anchor) | `keep_front_layers=16 n_fresh_layers=0 keep_layer_indices=[0..12,16,17,31] arm=shortgpt num_hidden_layers=16` |
| shortgpt16 post-SFT | `outputs/olmo2_p24_sft_shortgpt16/final.pt` | 48.7 GB | SFT 2026-08-08 01:33, step=842, from shortgpt16@200k | `keep_front_layers=16 n_fresh_layers=0 num_hidden_layers=16` |

`load_pruned_model` reads `keep_front_layers` and `n_fresh_layers` directly from the ckpt meta for legs 2/3/4 → no CLI overrides required. Leg 1 (vanilla base) takes the `load_base_model` path (no `--ckpt`).

## Output layout (paper Table 4 anchor convention + `_wzc1` suffix on pre-SFT anchors)

| harness | full32 pre (`7B_full32_base_wzc1`) | full32 post (`7B_p24_sft_full32_final`) | shortgpt16 pre (`7B_shortgpt16_step200000_wzc1`) | shortgpt16 post (`7B_p24_sft_shortgpt16_final`) |
|---|---|---|---|---|
| PPL (held-out Dolmino) | `olmo2_ppl_results/<name>/summary.json` | same | same | same |
| core6 (hs/arcc/arce/piqa/wg/obqa) | `olmo2_downstream_results/<name>/summary.json` | same | same | same |
| know5 (mmlu/lambada/boolq/csqa/siqa) | `olmo2_downstream_results/<name>_know/summary.json` | same | same | same |
| MMLU dual (letter + content_norm/raw) | `olmo2_mmlu_content_results/<name>/summary.json` | same | same | same |
| PopQA + TriviaQA closed-book | `olmo2_closedbook_results/<name>/summary.json` | same | same | same |

The `_wzc1` suffix on the two pre-SFT anchors keeps provenance explicit. In particular:
- `olmo2_downstream_results/7B_shortgpt_step200000{,_know}/` already exists on wzc1 from a prior run **without `--save_per_example`** → not reusable for McNemar. We rerun into the `_wzc1`-suffixed dir with per-example retained.
- `olmo2_mmlu_content_results/7B_shortgpt16_step200000/` exists on wzc1 with `per_example_mmlu.jsonl` — but for provenance uniformity we still write to the `_wzc1`-suffixed dir. (Skip-guard `summary.json` check is per-name, so no accidental reuse.)
- Neither `full32_base` (vanilla) nor the closed-book slot for either arm has any prior wzc1 eval — those are fresh.

## Harness invariants (all enforced)

- **8-shard rule**: `CUDA_VISIBLE_DEVICES=$g` + `LOCAL_RANK=0 RANK=$g` for every shard.
- **Merge gate**: `assert_8shards <root> <name> shard{i}of8.json` runs before every merge. If any shard missing → abort merge, log the miss, fail loud. **No partial merge = no silent 5/8 contamination** (memory: `kill-remote-gpu-job-by-pid-not-pkill`).
- **`chat_template=False`** throughout (project paper hard rule; OLMo-2 base has no SFT/RL). All eval scripts default to `--add_bos 0` (add_special_tokens=False), preserving the base-protocol tokenisation used by Table 4 anchors.
- **Per-item predictions retained** for pre/post pairing:
  - downstream MC: `--save_per_example` → `per_example_<task>.jsonl` after merge.
  - MMLU content: `per_example_mmlu.jsonl` (default in harness).
  - closed-book: `per_example_<task>.jsonl` (default in harness).
- **Arch meta from ckpt**: `load_pruned_model` reads `keep_front_layers` / `n_fresh_layers` directly from each `.pt` meta; CLI-passed values (none passed) would be checked for mismatch.
- **Model construction shared**: PPL, downstream MC, MMLU-content, closed-book all import `load_pruned_model` / `load_base_model` from `eval_olmo2_probe2_ppl.py` → NO arch drift across the four harnesses.

## Health check at t+0:30

- `nvidia-smi` on `.252` at 02:02:37: **8/8 GPUs @ 99-100% util, 64.6 GiB/183 GiB per card** (bf16 32L base model + KV cache).
- 8 PPL shards live on GPU 0-7 loading vanilla 32L base successfully (`num_hidden_layers=32 vocab=100352`, `torch_dtype` warning is a benign transformers version quirk).
- Log confirms base-mode path taken correctly (no `--ckpt` argument on leg 1).

## Loud-report anchors

Per spec, the following pre-SFT numbers are load-bearing. If any deviates substantially I MUST report loudly (a further cross-arch/harness drift, or a base-model corruption):

| anchor | expected (paper Table 4) | tolerance for loud report |
|---|---|---|
| full32 pre-SFT PPL (Dolmino) | **7.398** | > 0.2 |
| full32 pre-SFT MMLU (letter) | **0.6053** | > 0.5 pp |
| shortgpt16 pre-SFT PPL       | **9.7803** | > 0.2 |
| shortgpt16 pre-SFT MMLU      | **0.4739** | > 0.5 pp |

**These will be filled in as they land (see "Results" below).**

## Sanity cross-checks vs sibling status doc

Reviewed `status/PAPERB_P24_KEEP14_EVAL.md` (sibling agent aefe8b20):
- Their pre-SFT PPL rerun 10.5611512 vs wzc1 anchor 10.5612951 (diff 1.4e-4) — cleanly within float-reduction noise; harness path is stable across disks. Consistent with the "core6 diverges but PPL is float-tight" story from `PAPERB_CORE6_CROSSARCH_FLOOR.md` (PPL reductions are sum-of-logs; core6 acc is a hard argmax that flips under bf16 noise).
- No inconsistencies flagged in the sibling doc that need correction here.

## Timing

- Each battery (5 harnesses × 8 shards): **~60-90 min on 8×L20A** (L20A ≈ 2× H20 throughput, but per-harness fixed overhead + prepare_data cache warm-up dominates the short runs).
- Four batteries serial: **~4-6 h total → 2026-08-08 06:00-08:00 +08:00**.
- Serial by design: 8 shards already saturate the 8 GPUs; oversubscribing would cause CUDA context contention and slow both.

## Reproducibility

- Committer `LiuHanzuo <lhz24@mails.tsinghua.edu.cn>`; no `Co-Authored-By` / no AI trailer.
- Commit hash at launch: `0b2f707` — `experiment(paperB): P2.4 pre/post-SFT eval battery for full32+shortgpt16 on .252 (wzc1, same-arch pairing)`.
- `status/gpu_runs.jsonl` entry appended.
- `status/GPU_STATUS.md` `.252` block updated (SFT full32 + shortgpt16 → DONE, eval battery → running).

## SFT contamination caveat (paper-side, not this eval)

Sibling audit (per spec): **MMLU: 45 hits** in the Tulu-3 general-clean SFT data (PopQA/TriviaQA/NQ-open: 0). Any post-SFT MMLU number will need clean-subset recompute (CPU post-process, off the retained `per_example_mmlu.jsonl`). Closed-book numbers are contamination-free.

## No touching (per dispatch spec)

Did not touch: `.tex`, `versions/*.md`, `paperB/TODOList.md`, `status/PAPERB_P24_*_ARM.md`, `status/PAPERB_P24_KEEP14_EVAL.md`, or any other sibling dispatch artifacts.

## Results

Filled as batteries land. Empty placeholders now; the driver's shard invariant ensures the numbers are only visible after 8/8 shards merge.

### Leg 1 — full32 pre-SFT (`7B_full32_base_wzc1`) — PPL + core6 DONE @ 02:04

- [x] **PPL = 7.398071** (paper Table 4: 7.398 ✓ **exact match, 6-decimal**) — cross-disk harness is stable; `n_windows=4096 n_tokens=8384512`.
- [x] **core6** (`acc_norm` for hs/arcc/arce/piqa/obqa, `acc` for wg):
  - hellaswag: acc_norm=**0.8048** (acc=0.6035, n=10042)
  - arc_challenge: acc_norm=**0.5725** (acc=0.5410, n=1172)
  - arc_easy: acc_norm=**0.8283** (acc=0.8237, n=2376)
  - piqa: acc_norm=**0.8107** (acc=0.8090, n=1838)
  - winogrande: acc=**0.7459** (n=1267)
  - openbookqa: acc_norm=**0.4620** (acc=0.3700, n=500)
  - **core6 avg ≈ 0.7040**
- [ ] know5:         `olmo2_downstream_results/7B_full32_base_wzc1_know/summary.json`  (MMLU expect ≈ 0.6053) — running now
- [ ] MMLU dual:     `olmo2_mmlu_content_results/7B_full32_base_wzc1/summary.json`
- [ ] closedbook:    `olmo2_closedbook_results/7B_full32_base_wzc1/summary.json`

### Leg 2 — full32 post-SFT (`7B_p24_sft_full32_final`)

- [ ] PPL, core6, know5, MMLU dual, closedbook — same tree.

### Leg 3 — shortgpt16 pre-SFT (`7B_shortgpt16_step200000_wzc1`)

- [ ] PPL:           expect ≈ 9.7803
- [ ] know5 (MMLU):  expect ≈ 0.4739
- [ ] core6, MMLU dual, closedbook: same tree.

### Leg 4 — shortgpt16 post-SFT (`7B_p24_sft_shortgpt16_final`)

- [ ] Full 5-harness battery.
