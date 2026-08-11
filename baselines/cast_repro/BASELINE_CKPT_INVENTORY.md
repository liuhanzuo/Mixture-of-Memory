# SparseForge main-table baselines — checkpoint inventory & re-scorability

Scope: the **7 measured rows of the main table**
(`SparseForge_NIPS_2026/sections/experiments.tex`, `\label{tab:llama2_compare}`)
that the reproduction harness `outputs/cast_eval_spec/` did **not** cover.
The harness already covered 4 rows: `dense_ref`, `wanda`, `ast_official`, `sparseforge_5b`.

Reproduction harness definition (must be matched byte-for-byte, see
`scripts/_union9_eval_spec_21.sh`): node **.21**, lm-eval **0.4.8**,
`--model hf`, `dtype=bfloat16`, `parallelize=True`, `trust_remote_code=True`,
`add_bos_token=False`, `--batch_size auto`, `--num_fewshot 0`, `--seed 0`,
`--log_samples`, no chat template. Union-9 = CAST-7 ∪ AST-7 in **one** invocation.

## Why re-scoring is necessary (not cosmetic)

`ast_official` is the only arm measured under **both** the old (`SparseForge_Data`)
and the new (reproduction) harness. Its **plain-acc AST-7 mean** differs:

| | AST-7 plain acc |
|---|---|
| old harness (as reported in main table) | 57.9436 |
| reproduction harness (`outputs/cast_eval_spec/ast_official/zeroshot_metrics.json`) | **57.5976** |

Δ = **−0.3460 pp**, with max single-task delta **1.59 pp** (BoolQ). So old-CSV
numbers and reproduction numbers are **different measurements** and must not be
pooled into one column. Every row still reachable on disk has to be re-measured.

## Two-disk search performed

Both physical disks were searched (a wzc1-only or zwfy6-only search is not
sufficient to claim "not on disk"):

- **wzc1** = LOCAL + `.21` → `/apdcephfs_wzc1/share_304376610/pighzliu_code/`
- **zwfy6** = `.73`/`.82`/`.104` → `/apdcephfs_zwfy6/share_304376610/pighzliu_code/`

Primary ledgers read first (faster than blind search, and they proved accurate):
`SparseForge_Data/tables/checkpoints.tsv`, `unified_baselines.csv`,
`out_llama_ast7_checkpoint_inventory.csv`, `common_recovery.csv`,
`trainingfree_lowtraining_detailed_ast7.csv`.

⚠️ `checkpoints.tsv` records the baseline/ELSA rows as living on `.73`/`.82`
("diskB") with **wzc1-looking path strings**. Those strings are misleading:
on `.73` the path `/apdcephfs_wzc1` is a **symlink into zwfy6**. The
authoritative finding below is from direct `ls` on both disks, not from the TSV.

---

## Row-by-row inventory

### 1. SparseGPT (3 seeds) — paper 51.87

| | |
|---|---|
| Materialized HF (2:4 applied) | **zwfy6**: `outputs/paper_v2/trainingfree_3seed/trainingfree_3seed_20260730_v1/models/sparsegpt_seed{0,1,2}/` — 13 GB each, HF safetensors, 3-shard |
| Also on wzc1 | `outputs/paper_v2/materialized_baselines/sparsegpt_seed0/` (seed0 only, 13 GB) |
| Masks | zwfy6 `outputs/paper_v2/sparsegpt/{llama2_wandb_sf_alps_v1_sparsegpt_2of4_seed0,…seed1,llama2_baselines_sparsegpt_2of4_seed2}/mask.pt` (6.6 GB each); wzc1 `outputs/paper_v2/baseline_masks/sparsegpt_seed0_mask.pt` |
| Exists | **YES**, all 3 seeds |
| Re-scorable | ⚠️ **BLOCKED BY 2:4 GATE** — see below |

**Gate result (seed0, wzc1 copy):** `zero_frac 0.500000071`, **bad_tiles 462**,
`exact_2of4_tile_ratio 0.999999715` → `VERDICT: FAIL`.
Per the task's hard gate ("verify unless `zero_frac 0.500000` +
`exact_2of4_tile_ratio 1.0`; if it fails, do not score, report it"), **not scored**.
See `outputs/cast_eval_spec_gate/gate_sparsegpt_seed0.log` and
`tiledir_sparsegpt_seed0.json` for the direction breakdown (whether the 462 tiles
are *sparser* than 2:4, which would be deployable, or *denser*, which would be a
genuine budget violation).

### 2. ALPS (3 seeds) — paper 52.12

| | |
|---|---|
| HF export | **wzc1**: `outputs/paper_v2/staged_diskb_models/outputs/paper_v2/alps/llama2_wandb_sf_alps_v1_alps_seed{0,1,2}/hf_model/` — 13 GB each |
| Mirror | **zwfy6**: `outputs/paper_v2/alps/llama2_wandb_sf_alps_v1_alps_seed{0,1,2}/hf_model/` — byte-identical file sizes |
| Exists | **YES**, all 3 seeds, already on wzc1 → **no cross-disk staging needed** |
| 2:4 gate | ✅ **PASS** ×3 (`zero_frac 0.500000000`, `bad_tiles 0`, `exact_2of4_frac 1.000000000`, 224 in-scope tensors) |
| Re-scorable | **YES — re-scored** |

### 3. ELSA (4,096 steps; 3 seeds) — paper 54.59

| | |
|---|---|
| HF export | **wzc1**: `outputs/paper_v2/staged_diskb_models/outputs/paper_v2/elsa/{paper_v2_overnight_20260725_v1_elsa_full4096/Llama--Llama2-7b_pruned0.5_admm_lr0.0002_20260725_0136, paper_v2_node82_elsa_3seed_20260725_full4096_seed1/…_1050, …_seed2/…_2124}/` — 26 GB each (fp32, 6 shards) |
| Mirror | **zwfy6**: `outputs/paper_v2/elsa/…` same three |
| Exists | **YES**, all 3 seeds, already on wzc1 |
| Note | `config.json` declares `architectures: ["FSDPLlamaForCausalLM"]` and `torch_dtype: float32`. Verified on .21 that `AutoConfig` → `LlamaConfig` → `LlamaForCausalLM`, so it loads through the standard harness with `dtype=bfloat16` like every other arm. No custom code path. |
| 2:4 gate | ✅ **PASS** ×3 (`bad_tiles 0`, `exact_2of4_frac 1.000000000`) |
| Re-scorable | **YES — re-scored** |

### 4. ProxSparse (official checkpoint) — paper 51.92

| | |
|---|---|
| HF ckpt | **wzc1**: `outputs/paper_v2/staged_diskb_models/proxsparse_models/Llama-2-7b-hf-en_sft_final_400_len4096_batch1_lambda0.25/` — 13 GB |
| Mirror | **zwfy6**: `proxsparse_models/Llama-2-7b-hf-en_sft_final_400_len4096_batch1_lambda0.25/` (13 GB, byte-identical sizes). Note there is **no** `proxsparse_models/` at the wzc1 project root — only the staged copy above. |
| Exists | **YES** |
| Re-scorable | ⚠️ **BLOCKED BY 2:4 GATE** |

**Gate result:** `zero_frac 0.500000011`, **bad_tiles 68**,
`exact_2of4_tile_ratio 0.999999958` → `VERDICT: FAIL`. **Not scored.**
This is *consistent with a pre-existing note*: `checkpoints.tsv` labels the
derived mask `ProxSparse-top2-2of4` as `usable_68_groups_projected` — i.e. the 68
non-conforming tiles were already known when the mask was built for the recovery
runs, and were projected away at that point. The official checkpoint itself is
therefore not bit-exact 2:4. Direction breakdown in
`outputs/cast_eval_spec_gate/tiledir_proxsparse_official.json`.

### 5–7. ALPS / ELSA / ProxSparse mask + recovery (625M) — paper 55.82 / 55.94 / 55.91

These are **training products of the SparseForge trainer**, not HF exports.

| row | checkpoint | disk | size |
|---|---|---|---|
| ProxSparse + recovery | `outputs/paper_v2/fixed_mask_full_healing/fixed_mask_full_healing_proxsparse_625m_safe_20260729_v1_full625m/_apdcephfs…_20260729_120004/model.pt` | **wzc1** | 28.1 GB |
| ALPS + recovery | `outputs/paper_v2/fixed_mask_full_healing/fixed_mask_full_healing_alps_625m_safe_20260728_v1_full625m/_apdcephfs…_20260728_232212/model.pt` | **zwfy6 only** | 28.1 GB |
| ELSA + recovery | `outputs/paper_v2/fixed_mask_full_healing/fixed_mask_full_healing_elsa_625m_safe_20260728_v1_full625m/_apdcephfs…_20260728_232221/model.pt` | **zwfy6 only** | 28.1 GB |

wzc1's `outputs/paper_v2/staged_full_healing_checkpoints/{alps,elsa}/` exist but
are **empty directories** — a staging attempt that never ran. ALPS/ELSA recovery
must be `scp -O`'d from zwfy6 (measured link ≈ 150 MB/s on a small file; 28 GB ≈
several minutes each, so this is cheap — it is *not* the blocker).

**Exists:** YES, all 3. **Wiki PPL matches the paper's column** (7.1006 / 7.1121 /
7.1284 in `common_recovery.csv` = the paper's Wiki PPL for these rows), so these
are unambiguously the checkpoints behind rows 5–7.

**Re-scorable in the reproduction harness: NO, not without an export decision.**
The blocker is structural, not availability. `model.pt` is a trainer checkpoint
(`{model_state_dict, iter_num, args, finalization_done, finalization_iter}`,
1859 tensors) whose per-projection state is:

- `.weight` — **dense** (4096×11008 etc.)
- `.mask` — **continuous soft** mask, same shape
- `.SLoRB_Weight` (out×256 / out×688) + `.x_proj` (256×in / 688×in) — a **dense,
  trained, low-rank side branch**, `SLoRB_k=16`, `args.SLoRB=True`
- `.hessian_diag`, `.scaler_row`, `.grad_ema`, `.frozen_mask_flags` — optimizer aux

`args.hard_mask_type = nm_2_4`, so hardening must use the **`nm_2_4` branch**
(`sparse_modeling.py:596-621`, exact per-group `topk(2)+scatter_`), **not**
threshold-0.5. That part is settled and tooled
(`baselines/cast_repro/tools/probe_sparseforge_ckpt.py` ports it verbatim).

The unsettled part is **SLoRB**. `sparse_modeling.py:819-822` computes
`out = masked_linear(x, W, mask) + (x @ x_proj.T) @ SLoRB_Weight.T`, i.e. the
evaluated model is 2:4 `W` **plus** a dense low-rank term. Therefore:
- **Fold SLoRB into W** → writes into pruned positions → **destroys 2:4** → the
  arm fails the same hard gate that just excluded SparseGPT/ProxSparse.
- **Drop SLoRB** (what `deploy_sparse_24/convert.py:189` does, treating it as a
  training auxiliary) → genuinely 2:4, but a **different model** from the one whose
  accuracy the paper reports.

This is exactly the fork another agent is currently resolving for the
`sparseforge_5b` row (`outputs/cast_eval_spec/sparseforge_5b/{slorb_probe.json,
mask_binariness.json}`, and live `hard_drop` vs `soft_fold` arms). The measured
SLoRB magnitude there is **not** negligible — `ratio_SLoRB_over_Wmasked ≈ 0.366–0.376`
and `frac_SLoRB_energy_on_pruned_positions ≈ 0.49` — so the choice materially
changes the number and cannot be made silently.

**Decision: rows 5–7 are deferred**, to be exported under whichever convention the
`sparseforge_5b` fork settles on, so that all four SLoRB-bearing rows (5, 6, 7 and
the SparseForge headline) use **one** convention. Exporting them now under a guessed
convention would produce four rows that are not mutually comparable. Cost once
settled: 3 × (export ≈ 41 GB read + materialize) + 3 × union-9 (≈ 5 min/arm on 4
GPUs) + 3 × PPL — well under one GPU-hour on .21, plus ~1 h of cross-disk `scp -O`
for the two zwfy6-only ckpts. **No retraining needed.**

---

## Summary

| Main-table row | ckpt on disk | disk | 2:4 gate | re-scored here |
|---|---|---|---|---|
| SparseGPT (3 seeds) | yes (3 seeds) | zwfy6 (+seed0 on wzc1) | **FAIL** (462 bad tiles) | no — gate |
| ALPS (3 seeds) | yes (3 seeds) | **wzc1** + zwfy6 | PASS | **yes** |
| ELSA (4,096 steps; 3 seeds) | yes (3 seeds) | **wzc1** + zwfy6 | PASS | **yes** |
| ProxSparse (official) | yes | **wzc1** + zwfy6 | **FAIL** (68 bad tiles) | no — gate |
| ALPS mask + recovery (625M) | yes | zwfy6 only | n/a (soft mask) | no — SLoRB export fork |
| ELSA mask + recovery (625M) | yes | zwfy6 only | n/a (soft mask) | no — SLoRB export fork |
| ProxSparse mask + recovery (625M) | yes | **wzc1** | n/a (soft mask) | no — SLoRB export fork |

**Nothing is lost.** All 7 rows have their checkpoints on disk; none requires
retraining. 2 are re-scored, 2 are gate-blocked (a finding about the checkpoints,
not a missing file), 3 await one export-convention decision shared with the
SparseForge headline row.

## Provenance of numbers quoted from elsewhere

- Paper column values (51.87 / 52.12 / 54.59 / 51.92 / 55.82 / 55.94 / 55.91):
  **old harness**, from `SparseForge_Data/tables/trainingfree_lowtraining_detailed_ast7.csv`
  and `unified_baselines.csv`. Not measured by me.
- Old-harness raw eval trees: zwfy6 `outputs/paper_v2/unified_eval/{alps_seed*,
  elsa_full4096_seed*,proxsparse_official,dense_llama2}/` and
  `outputs/paper_v2/trainingfree_3seed/trainingfree_3seed_20260730_v1/eval/`.
- Recovery-row Wiki PPL (7.1006 / 7.1121 / 7.1284): `common_recovery.csv` +
  each run's own `eval.json`. Not measured by me.
- Everything under `outputs/cast_eval_spec/{alps_seed*,elsa_seed*}/` and
  `outputs/cast_eval_spec_union9/{alps_seed*,elsa_seed*}/` **was measured by me**
  on .21 under the invocation at the top of this file.
