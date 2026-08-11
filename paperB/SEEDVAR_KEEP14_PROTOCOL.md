# Paper B — keep14+fresh2 seed variance (task #181): PINNED EVAL PROTOCOL

**Status**: protocol pinned 2026-08-12, before any GPU spend.
**Purpose**: the seed-1234 keep14 run's entire value is a *same-protocol* delta against the
existing seed-42 keep14@200000 arm. If harness / axes / batch size / node differ, the measured
quantity is code-version + hardware drift, not seed variance. This file is the audit trail for
what the seed-42 baseline actually is, per axis.

## Both training arms (config diff)

| field | seed 42 | seed 1234 |
|---|---|---|
| dir | `outputs/olmo2_probe2_7B_keep14fresh2/` | `outputs/olmo2_probe2_7B_keep14fresh2_seed1234/` |
| arm | `healing_front14+fresh2` | `healing_front14+fresh2` (identical) |
| keep_front / n_fresh / total | 14 / 2 / 16 | 14 / 2 / 16 (identical) |
| n_params | 4,060,352,512 | 4,060,352,512 (identical) |
| seq_len | 2048 | 2048 |
| `lr_inherited` | 2e-5 | 2e-5 |
| `lr_fresh` in `arch_meta.json` | **1e-4** | **2e-5** |
| `seed` field | absent (default 42) | 1234 |
| endpoint ckpt | `step200000.pt` (48,724,473,850 B) | `step200000.pt` (48,724,474,298 B) |
| final train loss/ppl | — | `loss=2.3286 ppl=10.26` @200000 |

> ⚠️ **The `lr_fresh` field differs (1e-4 vs 2e-5) but this is a metadata artefact, not a real
> config difference.** CLAUDE.md records that `_classify_param` did not strip the `module.`
> prefix in the affected trainer revisions, so the `fresh` param group was never populated and
> **both arms trained at a uniform 2e-5**. Paper B's `tab_main_results.tex` already reports
> keep14 as "Peak LR 2e-5, trainable set = all", consistent with the uniform-LR reading.
> Do **not** describe this pair as a differential-LR contrast. Both arms are the same recipe
> at two data-order/init seeds, which is exactly what a seed-variance contrast needs.

## Seed-42 baseline per axis (what exists, where, on which disk)

`n_scored`/`n_nan` are as recorded in each archived `summary.json`.

| # | axis | harness script | driver that produced seed-42 | seed-42 result path | disk | n / n_nan | published in Paper B as |
|---|---|---|---|---|---|---|---|
| 1 | in-domain held-out NTP PPL (Dolmino) | `scripts/eval_olmo2_probe2_ppl.py` | `scripts/_run_olmo2_eval_keep14_s200000_b200.sh` (node **.252**, `$WD/.venv/bin/python`, 2026-07-28) | `olmo2_ppl_results/7B_keep14_step200000/` | **wzc1** | 4096 windows / 8,384,512 tok | `10.561` (tab_main_results, tab_policy_endpoint, tab_depth_ppl, app_tab_ppl, fig1) |
| 2 | core6 downstream MC | `scripts/eval_olmo2_probe2_downstream.py` | same `_b200.sh` driver, `--batch_size 8` | `olmo2_downstream_results/7B_keep14_step200000/` | **wzc1** | 10042/1172/2376/1838/1267/500, nan=0 | core6 `.594`, HellaSwag `.645`, ARC-C `.438` (tab_policy_endpoint) |
| 2b | core6 **per-item** replica | same | `scripts/_run_olmo2_keep14_wzc1_v2_252.sh` (node .252, **conda** `torch-base`, 2026-08-08) | `olmo2_downstream_results/7B_keep14_step200000_wzc1_v2/` + `..._perex/` | **wzc1** | identical to row 2 | (per-item source for pairing) |
| 3 | know5 downstream MC | `scripts/eval_olmo2_probe2_downstream.py` | same `_b200.sh` driver (`_know` output name) | `olmo2_downstream_results/7B_keep14_step200000_know/` | **wzc1** | mmlu 14042, lambada 5153, boolq 3270, csqa 1221, siqa 1954, nan=0 | MMLU-L `.319` (tab_main_results); `aux5_raw` in RUN_REGISTRY |
| 3b | know5 **per-item** replica | same | (Jul-29 perex run) | `olmo2_downstream_results/7B_keep14_step200000_perex_know/` | **wzc1** | identical to row 3 | (per-item source for pairing) |
| 4 | MMLU letter+content dual | `scripts/eval_olmo2_mmlu_content.py` | `scripts/_run_olmo2_mmlu_content.sh` (2026-08-02) | `olmo2_mmlu_content_results/7B_keep14_step200000/` | **wzc1** | 14042 / n_nan=0 | MMLU-L `.3184` / MMLU-C `.383` (tab_main_results, tab_control, rebuttal tab_letter_headroom) |
| 5 | OOD PPL WikiText-103 | `scripts/eval_olmo2_probe2_ppl.py` | `scripts/_run_ood_ppl_pool.sh` (2026-08-04, `num_shards=1 bs=8`) | `ood_ppl_results/keep14_step200000_wikitext103/` | **wzc1** | 141 windows / 288,627 tok | `11.56` (tab_ood_audit) |
| 6 | OOD PPL PG-19 | same | same pool driver | `ood_ppl_results/keep14_step200000_pg19/` | **wzc1** | 1200 windows / 2,456,400 tok | `15.43` (tab_ood_audit) |

### Axes where the seed-42 baseline does NOT exist on wzc1 → excluded from the seed contrast

| axis | why excluded |
|---|---|
| **closed-book PopQA / TriviaQA** (`eval_olmo2_closedbook_qa.py`) | Paper B publishes PopQA `.142` / TriviaQA `.294` for keep14, but the seed-42 result exists **only on zwfy6** (`olmo2_closedbook_results/7B_keep14_step200000{,_v2}`, confirmed via .104). wzc1 has closed-book dirs for base/keep8/keep10/keep12/shortgpt16 but **not keep14**. Also the generation path needs HF PopQA/TriviaQA/NQ-open; wzc1's `data/hf_datasets_cache` has `akariasai___pop_qa` and `mandarjoshi___trivia_qa` but **no nq_open**. Running this axis on wzc1 would put the seed-42 side on H20 and the seed-1234 side on L20A → the delta would confound seed with architecture. **Excluded; cannot be used for seed variance without first re-running seed-42 closed-book on wzc1.** |
| **NQ-open** | Paper B publishes `.060` for keep14, but there is **no wzc1 seed-42 dir at all** and the dataset is not in the wzc1 HF cache. Excluded, same reason. |
| **`olmo2_mc_letter_content_results/7B_keep14_step200000`** (6 non-MMLU MC, letter-vs-content) | This exists on wzc1 (2026-08-11) but it is a **paperG gate-2** artefact (`scripts/_run_olmo2_mc_letter_content.sh`, run on **.73/H20** and copied), not a Paper B seed axis, and Paper B does not publish it as a keep14 row. Excluded to keep the contrast to Paper B's own axes. |
| `olmo2_probe2_downstream_results` / `olmo2_probe2_ppl_results` (dir names in task #93's description) | **These directories do not exist on either disk.** The real dirs are `olmo2_downstream_results` / `olmo2_ppl_results` (no `probe2_` infix); the `probe2` lives in the *script* name. Task #93's "PPL + core6 + know5" axes therefore DO exist — rows 1-3 above — just under different directory names. |

## Harness-version decision: re-run BOTH seeds on LOCAL under one driver

Two facts force this:

1. The archived seed-42 rows were produced on **node .252 (retired)** with **`$WD/.venv/bin/python`**.
   LOCAL's `.venv` **no longer has torch** (`ModuleNotFoundError: No module named 'torch'`,
   verified 2026-08-12; matches the CLAUDE.md 2026-08-04 note). Only
   `/opt/conda/envs/torch-base/bin/python` (torch 2.13.0, 8 GPUs visible) is usable.
   So seed-1234 *cannot* be run on the archive's exact interpreter.
2. `status/PAPERB_WITHIN_DISK_FLOOR_V3.md` establishes that under a **single driver revision**
   same-disk same-arch eval is **bit-deterministic (0 flips)**, and that the only non-zero
   within-disk comparison in the whole ledger **crossed a driver boundary**. Driver drift is a
   *systematic* bias, not zero-mean noise, so it cannot be averaged away.

**Chosen path: option (b) — recompute BOTH sides with the new toolchain, in one driver,
back-to-back on the same 8 L20A GPUs** (`scripts/_run_paperB_keep14_seedvar_local.sh`).
The alternative (seed-1234-on-conda vs seed-42-archived-on-.252/.venv) would measure
seed + node + interpreter jointly and is rejected.

Supporting evidence that this is safe and that the archive is portable: on wzc1 the Jul-28
`.venv`/.252 per-item dumps and the Aug-8 **conda** `_wzc1_v2` per-item dumps are
**byte-identical for all six core6 tasks** (`cmp` clean), with summary accs equal to six
decimals. The re-run of seed-42 in this battery is therefore also a live provenance check: if
it reproduces `10.561295 / .644593 / .318402`, the archived numbers are confirmed portable
across node+interpreter; if it does not, the discrepancy is quantified and the seed contrast
is still valid because **both** arms in the contrast were produced by this one driver.

### Protocol constants held fixed across both arms

- `chat_template=False` (OLMo-2-1124-7B is a BASE LM, no SFT/RL) — enforced by these harnesses
  having no chat path at all; `add_bos=0` recorded in every `summary.json`.
- fp32 weights + bf16 autocast; `--keep_front_layers 14 --n_fresh_layers 2`; strict state-dict load.
- 8 shards, stride `[i::8]`; batch sizes verbatim from the seed-42 drivers:
  **ppl 4**, **downstream 8**, **mmlu 16**, **OOD ppl 1 shard / bs 8**.
- Merges: `assert_shards` refuses <8/8; the ppl harness itself raises on an incomplete set
  (`d380bbc` guard); the MMLU merge asserts the exact 14042 count.
- `--save_per_example` on both downstream legs (per-item needed for McNemar pairing).
- Output names namespaced `keep14_{s42,s1234}_step200000_sv181*` so **nothing archived is
  overwritten**; the driver hard-fails if a target dir already exists.

## Statistics plan

- Product = per-axis **seed42 − seed1234** delta, with **exact McNemar** (paired, per-item) and
  **paired bootstrap** `n_boot=10000` on the accuracy axes; PPL axes report Δppl / Δavg_nll
  (token-weighted, no per-item test).
- ⚠️ **n=2 → df=1.** Two draws do not give a reportable σ_run. The deliverable is the delta plus
  an explicit statement of how wide a df=1 χ² interval on σ is.
- ⚠️ Do **not** pool with the A03 σ numbers (df=5, 0.3666 pp, χ² CI [0.229, 0.899]) — that is
  **1B keep7/keep12 on TriviaQA**, a different model, scale, and harness.
