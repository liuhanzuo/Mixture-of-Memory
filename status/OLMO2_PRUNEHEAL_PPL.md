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
| OLMo-2-7B | keep14+fresh2 | 16 | **128000** (apex, converged) | **10.827** | 4096 | **1.463×** |
| OLMo-2-7B | keep14+fresh2 | 16 | **153500** (post-apex, +25.5k) | **10.693** | 4096 | **1.446×** |
| OLMo-2-7B | **from-scratch** 16L (control 2) | 16 | **200000** | **11.498** | 4096 | **1.554×** |

## Story

- **1B keep7+fresh2 (9L vs full 16L):** latest step147000 = **15.63 PPL**, gap **1.47×** the full-depth 1B base (10.64). The heal curve is **monotone decreasing and NOT yet plateaued** — 17.62 (50k) → 16.16 (100k) → 15.63 (147k), still ~0.53 PPL dropped over the last 47k steps. **Early-stop signal: keep training** (no convergence flat yet); gap should continue closing.
- **7B keep10+fresh2 (12L vs full 32L):** only step10000 (very early in healing), PPL **17.24**, gap **2.33×** the full 7B base (7.40). This ~2.3× early-stage gap closely reproduces the historical **armB ≈ 2× Qwen3-8B** signature; expected to shrink substantially with more heal steps (cf. the 1B curve).
- **7B keep14+fresh2 APEX (16L/32 = 50%, step128000, fully-converged heal):** held-out PPL **10.827**, gap **1.463×** the full 32L base (7.398) — i.e. the pruned-to-half-depth healed model recovers language modelling to within a ~46% LM-loss penalty, on par with the near-converged 1B keep7 point (1.47×). Big drop from the 12L early point (17.24/2.33× → 10.83/1.46×) confirms both **more kept layers (12→16) and more heal (10k→128k)** close the PPL gap. **This step128000 ppl is PAIRED with the downstream MC apex** (`OLMO2_PRUNEHEAL_DOWNSTREAM.md` §APEX, same ckpt step128000) — same-ckpt PPL-vs-capability read: PPL recovered to 1.46× while MMLU still only recovered ~14% of the base's above-chance signal, i.e. **PPL is largely blind to the residual knowledge gap.** Base 口径 (raw 2048-tok NTP windows, no chat_template / no generation / no BOS). Run **2026-07-20 03:06 CST** on .73 (8×H20, all 8 shards n_tokens=1,048,064, none empty; recompute exp(Σnll/Σtok)=summary to 1e-9).
- **7B keep14+fresh2 POST-APEX (step153500, +25.5k steps past the step128000 apex):** held-out PPL **10.693**, gap **1.446×** the full 32L base (7.398) — a **slight further improvement** over the step128000 apex (10.827 / 1.463×), ~0.13 PPL (~1.2%) lower. PPL is **not** regressing/overfitting with extra heal; it is still creeping down but has largely plateaued (the last 25.5k steps bought only ~0.13 PPL vs the huge 12L→16L / 10k→128k drop). **Same-ckpt PPL-vs-capability pair** with `OLMO2_PRUNEHEAL_DOWNSTREAM.md` §POST-APEX (step153500): MC/knowledge also only nudged (MMLU .301→.312) — confirms **"PPL plateaued early, knowledge still recovering but only marginally per extra step."** Base 口径 (no chat/gen/BOS). Run **2026-07-20 04:06 CST** on .73 (8×H20, all 8 shards n_tokens=1,048,064, none empty; recompute exp(Σnll/Σtok)=summary **10.693429** to 1e-9). Raw: `olmo2_ppl_results/7B_keep14_step153500/summary.json`; log `logs/olmo2_ppl_keep14_s153500_DONE`.
- **★ 7B from-scratch 16L (CONTROL 2, random-init keep14/fresh2 shell, step200000):** held-out PPL **11.498**, gap **1.554×** the full 32L base (7.398). **This is the decisive Paper-B control**: same 16-layer architecture as prune-then-heal keep14, but **all weights random-init** (no inherited pretrained front layers). Trained to **step200000 — MORE steps than the healed keep14 (128k apex / 153.5k post-apex)** — yet still **+0.80 PPL worse (11.498 vs 10.693)**. → **Inheriting the pretrained front 14 layers is a real, non-trivial advantage** over training the same small model from scratch, AND it is more sample-efficient (heal reaches 10.69 by 153k; from-scratch only 11.50 at 200k). Note the PPL gap is **modest (1.554× vs 1.446×, ~7.5% relative)** — the more decisive control read is expected in downstream MC/knowledge (from-scratch never saw OLMo-2's pretraining corpus → should collapse on MMLU/ARC vs the healed model that inherited pretrained knowledge; see `OLMO2_PRUNEHEAL_DOWNSTREAM.md` §CONTROL2). Base 口径 (no chat/gen/BOS). PPL run **2026-07-25** on LOCAL (8×B200, .venv, all 8 shards loaded ckpt step=200000 keep_front=14 n_fresh=2 num_hidden_layers=16 strict, n_windows=4096 n_tokens=8,384,512). Raw: `olmo2_ppl_results/7B_scratch16L_step200000/summary.json` (PPL 11.4983 avg_nll 2.4422). Training ckpt `outputs/olmo2_probe2_7B_keep14fresh2_fromscratch/final.pt` (48.7GB, arch_meta from_scratch=true n_trainable=4.06B).

## Raw JSON (diskB `.../Mixture-of-Memory/`)
- `olmo2_ppl_results/{1B_base_full,1B_keep7_step50000,1B_keep7_step100000,1B_keep7_step147000,7B_base_full,7B_keep10_step10000}/summary.json` (+ per-shard `shard{0..7}of8.json`).
- **keep14 apex (2026-07-20):** `olmo2_ppl_results/7B_keep14_step128000/summary.json` (PPL 10.8265, n_tokens 8,384,512, meta keep_front=14/n_fresh=2/num_hidden_layers=16/ckpt_step=128000) + per-shard `shard{0..7}of8.json`. Launched inline on .73 (base 口径, `--keep_front_layers 14 --n_fresh_layers 2` guard matches ckpt meta); scheduler log `logs/olmo2_ppl_keep14_sched.out`, DONE marker `logs/olmo2_ppl_keep14_DONE`.
- Scheduler log: `logs/olmo2_ppl_sched.out`; DONE marker: `logs/olmo2_ppl_DONE`.
