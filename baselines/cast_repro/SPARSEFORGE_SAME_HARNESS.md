# SparseForge 5B on the CAST-repro harness — one harness, seven arms

**Task**: #244. **Date**: 2026-08-11. **Node**: `.21` (8×L20A, wzc1).
**Driver**: `scripts/_sparseforge_same_harness_21.sh` + `scripts/_sparseforge_ppl_grid_21.sh`
**Provenance**: `outputs/cast_eval_spec/sparseforge_5b/` (+ `outputs/cast_eval_spec_ppl4096_sf/`)
**Table**: `outputs/cast_eval_spec/sparseforge_5b/sparseforge_same_harness_table.json`

## Why this was run

The SparseForge main table came from `SparseForge_Data/tables/cast9_dense_ast_current_harness.csv`
(old harness); our CAST reproduction lives in `outputs/cast_eval_spec_union9/` (lm-eval 0.4.8,
git `b86c479`). AST-official is the only arm in both, and its plain-acc AST-7 differs by
**−0.3460 pp** across the two (57.9436 → 57.5976). That offset is 0.65× SparseForge's stated
+0.53 pp margin, so no cross-harness comparison to SparseForge is admissible.

## ⛔ The checkpoint is not a 2:4 model — two blockers the brief did not anticipate

`out_llama/models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260413_201320/model_best_lm_eval.pt`
(`iter_num=17900`, `finalization_done=True`):

1. **Weights are dense, mask is continuous.** `zero_frac = 0.000000000`; `.mask` ∈ [7.96e-11, 1.0].
   The `nm_2_4` projection (`sparse_modeling.py:594-621`, per-group `topk(2)`+`scatter_`) gives
   **exact 2:4** (0 bad tiles of 1,619,001,344 over the full 224 tensors); threshold-0.5 does
   **not** — 4,300 bad tiles on a 6-layer sample alone (the 26,726 full-model figure is MAIN's
   earlier measurement, not re-verified here).
   Hardening is **not** numerically free: `‖W·soft − W·hard‖/‖W·hard‖` = 1e-3 … 8.5e-3 per layer
   (`tools/probe_mask_binariness.py`).

2. **★ There is an active dense low-rank branch (SLoRB).** `sparse_modeling.py:819-822` adds
   `(x @ x_proj.T) @ SLoRB_Weight.T` to *every* in-scope projection.
   - `SLoRB_Weight` fully nonzero; `x_proj` **trained away** from its fixed block-sum init
     (`trainable_projection: true`), so it is not a reconstructible constant.
   - Adds **848,429,056** live params (404,750,336 + 443,678,720) = **+26.2 %** over the
     3.238 G surviving weights. Frobenius norm **0.204–0.469 ×** the masked weight, with
     **~50 % of its energy on positions the 2:4 mask prunes**.
   - `deploy_sparse_24/convert.py:189` silently drops both tensors as "training auxiliaries".

So "SparseForge's 5B checkpoint" is ambiguous, and the ambiguity is worth ~4.9 pp. Three
variants were exported and all three scored (`tools/export_sparseforge_to_hf.py`).

## Faithfulness control — the pipeline is validated

`hard_fold` = hard mask + SLoRB active is **the published protocol**, per
`outputs/paper_v2/ast7_eval/sparseforge_5b_table2/eval.log`:
`Set hardening_x=0 for all SparseLinear layers (using hard masks)` +
`Keeping sparse_forward mode (SLoRB enabled, mask converted to hard mask)`.

| | CAST-7 plain acc |
|---|---:|
| checkpoint's own `best_lm_eval.json` anchor | 57.2672 |
| our `hard_fold` on this harness | **57.2750** |
| **delta** | **+0.0078 pp** |

Reproducing the anchor to 0.008 pp means the export (prefix stripping, `nm_2_4`, SLoRB fold)
and the scoring path are correct. The `hard_drop` deficit below is therefore a **real property
of the model**, not an export bug.

## The table (plain acc, `acc,none`, lm-eval 0.4.8, batch 64, seed 0, identical invocation)

| arm | boolq | rte | hellaswag | race | piqa | winogrande | arc_e | arc_c | obqa | **AST-7** | CAST-7 | UNION-9 | PPL@2048 | PPL@4096 | 2:4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| dense LLaMA-2-7B | 77.7676 | 63.1769 | 57.1301 | 39.7129 | 77.8564 | 69.2976 | 76.3468 | 43.1741 | 31.6000 | **59.7847** | 56.4454 | 59.5625 | 5.5637 | 5.2004 | no |
| SparseForge `hard_fold` (published protocol) | 77.7982 | 50.1805 | 54.4911 | 41.3397 | 76.8770 | 69.4554 | 78.1987 | 45.5631 | 35.0000 | **58.6696** | 57.2750 | 58.7671 | 6.6510 | 6.2115 | **no** |
| SparseForge `soft_fold` | 77.7370 | 50.1805 | 54.4513 | 41.6268 | 76.9314 | 69.9290 | 78.1145 | 45.3925 | 35.2000 | 58.7150 | 57.3779 | 58.8403 | 6.6486 | 6.2091 | **no** |
| CAST-repro @7500 | 69.2355 | 74.7292 | 53.9235 | 39.5215 | 76.3330 | 66.2983 | 74.0741 | 41.6382 | 28.8000 | **58.3856** | 54.3698 | 58.2837 | 6.5268 | 6.1372 | YES |
| AST-official | 72.9052 | 66.4260 | 54.5111 | 39.6172 | 76.9859 | 67.3244 | 73.4848 | 39.9317 | 28.6000 | **57.5976** | 54.3507 | 57.7540 | 6.3430 | 5.9125 | YES |
| **SparseForge `hard_drop` (true 2:4)** | 69.3272 | 57.0397 | 49.4722 | 36.9378 | 73.1230 | 62.9834 | 72.0960 | 36.7747 | 28.6000 | **53.7562** | 51.4267 | 54.0393 | 9.3770 | 8.8290 | **YES** |
| Wanda 2:4 | 68.1957 | 53.4296 | 41.2966 | 35.4067 | 70.2938 | 62.9834 | 61.0690 | 29.4369 | 23.0000 | 48.4873 | 46.2123 | 49.4569 | 12.4749 | 11.7733 | YES |

RTE is integral on n=277 for every arm: dense 175, `hard_fold`/`soft_fold` **139**,
AST 184, CAST-repro 207, `hard_drop` 158, Wanda 148. (139/277 = 50.1805 corroborates the
already-established 49.82 ≈ 138/277 truth and re-confirms the paper's 69.82 is a transcription error.)

## ★ Verdict — SparseForge does NOT beat the baselines under matched conditions

**AST-7 plain-acc gap vs baselines, same harness:**

| | vs dense | vs CAST-repro | vs AST-official | vs Wanda |
|---|---:|---:|---:|---:|
| `hard_drop` (**true 2:4, apples-to-apples**) | **−6.03** | **−4.63** | **−3.84** | +5.27 |
| `hard_fold` (published protocol, **not 2:4**) | −1.12 | +0.28 | +1.07 | +10.18 |

1. **As a genuine 2:4 model, SparseForge is the worst non-Wanda arm.** Strip the dense
   low-rank branch and it loses **3.84 pp AST-7 to AST-official** and **4.63 pp to our own
   CAST reproduction**, with PPL@2048 **9.3770 vs 6.3430/6.5268** — a 48 % PPL regression.
   The 2:4 support that SparseForge's mask search produces is *worse* than AST's.
2. **The published +0.53 pp margin is bought entirely by the SLoRB branch**, which adds
   26 % extra dense parameters and puts half its energy exactly where the mask claims zeros.
   Against AST-official the SLoRB-enabled margin is +1.07 pp here, but that compares a
   dense-augmented model to a strictly 2:4 one.
3. **Even with SLoRB, it does not clearly beat our CAST reproduction**: +0.28 pp AST-7,
   which is *below* the −0.346 pp cross-harness offset measured on the AST arm — i.e. within
   protocol noise, and it loses CAST-7 head-to-head PPL (6.6510 vs 6.5268 @2048).

## ★ Second finding — SparseForge's headline PPL 6.2179 is a **seqlen-4096** number sitting in a 2048 column

`outputs/paper_v2/ast7_eval/sparseforge_5b_table2/eval.log` shows `[eval_ppl]: 100%|…| 82/82`,
and its sibling `ast7_eval.json` records `"block_size": 4096`. 82 × 4096 = 335,872 target
tokens (at 2048 the same corpus yields 164 sequences). But the AST row in the *same* CSV
(6.3430) comes from `rebuttal_artifacts/2026-07-27/ast_official/ppl_metrics.json`, whose own
field says `"seqlen": 2048`.

**So `cast9_dense_ast_current_harness.csv` mixes 4096 and 2048 in one PPL column.** Our
`hard_fold` reproduces 6.2115 @4096 (vs published 6.2179, Δ 0.006) but reads 6.6510 @2048 —
confirming the published value is the 4096 one.

⚠️ **This invalidates SPEC.md:213's normalisation decision**, which assumed "SparseForge's
entire PPL column is 2048" and normalised everything *to* 2048 on that basis. At matched 4096:

    dense 5.2004 < AST-official 5.9125 < CAST-repro 6.1372 < SparseForge hard_fold 6.2115 < hard_drop 8.8290 < Wanda 11.7733

SparseForge is **last among the non-Wanda arms**, and AST-official beats it by **0.299**. The
claimed PPL win (6.2179 < 6.3430) was a 4096-vs-2048 artifact. SPEC.md needs updating.

## Honest caveats

- `hard_drop` is my construction, not something SparseForge published; it answers "what is this
  checkpoint worth as a real 2:4 model". Its 2:4 status is verified (`zero_frac 0.500000000`,
  0 bad tiles, 224 tensors, 6,476,005,376 elems, PRE- and POST-inference).
- `soft_fold`/`hard_fold` are **not 2:4** (verify FAILs by design, `exact_2of4_frac = 0`) and
  must never be placed in a 2:4 column.
- The dense-arm 2048 PPL (5.5637) is quoted from SPEC.md:204, not re-measured here; every
  other cell in the table was measured in this task or in task #243 on this harness.
- Not investigated: whether AST-official/CAST-repro would also gain from an added SLoRB-style
  branch. The fair statement is "SparseForge's *mask* is worse; its *added capacity* is what helps."

## Raw 2:4 gate output (`hard_drop`, PRE-inference)

```
[verify_2of4] in-scope tensors: 224
[verify_2of4] global: elems=6,476,005,376 zeros=3,238,002,688 zero_frac=0.500000000
              tiles=1,619,001,344 bad_tiles=0 exact_2of4_frac=1.000000000
[verify_2of4] VERDICT: PASS
```
POST-inference re-verify: identical, `VERDICT: PASS` (`verify_2of4_hard_drop_post.log`).
