# OLMo-2 Prune-then-Heal — Held-out PPL Ledger (Paper B)

Held-out next-token-prediction perplexity of the **prune-then-heal** OLMo-2
checkpoints vs the **full-depth pretrained base** (the denominator). This is the
early-stop signal + prune-baseline PPL requested in pending #1 (analogous to the
historical armB-vs-full-Qwen3-8B story).

## Setup (single source of truth)
- Driver: `scripts/eval_olmo2_probe2_ppl.py` (self-contained; pruned-arch construction copied verbatim from `scripts/train_olmo2_arch_probe2.py`).
- Launcher: `scripts/_run_olmo2_probe2_ppl_8gpu.sh` (8-GPU sharded, `[g::8]` window striding, token-weighted merge).
- Node: **.73** (`28.85.35.73`, diskB, EVAL-ONLY, forward-only PPL). Env `/opt/conda/envs/torch-base/bin/python` (torch2.13 / tf5.5.4). Ran full 8-GPU, all cards 100%.
- Val: `data/dolmino_now_val.npy` — OLMo-2 tokenizer, uint32, shape **[4096, 2048]**. Full val used (no `--limit`), 8 shards × 512 windows.
- Scoring: teacher-forced NTP CE per 2048-token window, fp32 `reduction='sum'` over shifted targets, bf16-autocast forward on fp32 weights (matches training). Merge = `exp(sum_nll / sum_tokens)` (token-weighted, **not** a mean of per-shard PPL).
- Per config: `n_windows=4096`, `n_tokens=8,384,512`. All PPL finite.
- Run finished **2026-07-19 13:23 CST**.

## Results

| model | keep / full | layers | step | held-out PPL | n_windows | gap vs full base |
|-------|-------------|-------:|-----:|-------------:|----------:|-----------------:|
| OLMo-2-1B | **full base** | 16 | — | **10.642** | 4096 | 1.00× (denom) |
| OLMo-2-1B | keep7+fresh2 | 9 | 50000 | 17.619 | 4096 | 1.656× |
| OLMo-2-1B | keep7+fresh2 | 9 | 100000 | 16.161 | 4096 | 1.519× |
| OLMo-2-1B | keep7+fresh2 | 9 | **147000** (latest) | **15.628** | 4096 | **1.469×** |
| OLMo-2-7B | **full base** | 32 | — | **7.398** | 4096 | 1.00× (denom) |
| OLMo-2-7B | keep10+fresh2 | 12 | **10000** (early) | **17.239** | 4096 | **2.330×** |

## Story

- **1B keep7+fresh2 (9L vs full 16L):** latest step147000 = **15.63 PPL**, gap **1.47×** the full-depth 1B base (10.64). The heal curve is **monotone decreasing and NOT yet plateaued** — 17.62 (50k) → 16.16 (100k) → 15.63 (147k), still ~0.53 PPL dropped over the last 47k steps. **Early-stop signal: keep training** (no convergence flat yet); gap should continue closing.
- **7B keep10+fresh2 (12L vs full 32L):** only step10000 (very early in healing), PPL **17.24**, gap **2.33×** the full 7B base (7.40). This ~2.3× early-stage gap closely reproduces the historical **armB ≈ 2× Qwen3-8B** signature; expected to shrink substantially with more heal steps (cf. the 1B curve).

## Raw JSON (diskB `.../Mixture-of-Memory/`)
- `olmo2_ppl_results/{1B_base_full,1B_keep7_step50000,1B_keep7_step100000,1B_keep7_step147000,7B_base_full,7B_keep10_step10000}/summary.json` (+ per-shard `shard{0..7}of8.json`).
- Scheduler log: `logs/olmo2_ppl_sched.out`; DONE marker: `logs/olmo2_ppl_DONE`.
