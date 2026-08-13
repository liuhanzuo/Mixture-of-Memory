---
scope: which on-disk SparseForge run corresponds to the paper's headline numbers
date: 2026-08-13 GMT+8
status: RESOLVED — the checkpoint used IS the paper's. My own "identity unproven" alarm was WRONG and is retracted below.
gpu_cost: 0
trigger: user 2026-08-13「而且你找的ckpt也不对啊」
---

# The checkpoint IS the paper's — settled from the paper's own eval log

## ✅ Resolution (one grep; should have been step 1)

`outputs/paper_v2/ast7_eval/sparseforge_5b_table2/eval.log` is the artefact the paper's Table 2
was rendered from. Line 16 reads:

    [INFO] Loading checkpoint from: /apdcephfs_wzc1/share_304376610/pighzliu_code/out_llama/
           models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260413_201320/model_best_lm_eval.pt

`grep -cE 'out_llama/models_'` on that log returns **1** — exactly one checkpoint is loaded, so
there is no ambiguity. **`20260413_201320` is the paper's headline run**, and
`SPARSEFORGE_SAME_HARNESS.md` (#244) scored the correct file.

## ⛔ Retraction of my own alarm (written ~20 min earlier the same day)

I first wrote that identity was "UNRESOLVED — three candidates fit the headline PPL". That was
**wrong, and wrong in the same way as the error it was trying to fix**: I searched every run for
a `wiki_ppl` near 6.2179, found a collision, and reported ambiguity — instead of reading the
paper's own rendering log, which names the checkpoint outright.

The PPL collision is real but **irrelevant to identity**:

| run | file | wiki_ppl | Δ vs 6.2179 | dataset |
|---|---|---:|---:|---|
| `20260404_110624` | `eval.json` | 6.219058 | 0.0012 | `dolmino-mix-1124-raw` |
| `20260413_201320` ← **the paper's** | `best_lm_eval.json` | 6.215503 | 0.0024 | `qa_format_sft_llama` |
| `20260327_114738` | `eval.json` | 6.210766 | 0.0071 | `c4_llama` |

Three runs land within 0.02 of each other on three different corpora. The transferable lesson is
narrower than "identity unknown": **a `wiki_ppl` value is not a run fingerprint in this repo.**
Never attribute a number to a run by PPL matching; use the rendering log.

The 39 LLaMA-2-7b run directories, and the four dated later than `20260413_201320`
(`20260416_170622`, `20260416_175757`, `20260420_200348`, `20260427_000532`, newest 2026-04-30),
are **later experiments, not later headline candidates**. Newer on disk means nothing.

## ⚠️ What IS wrong: the "5B" label (this part stands)

`args.json` for `20260413_201320`: `global_batch_size = 256`; checkpoint at `iter_num = 17900`.
At `block_size = 4096`:

    17,900 × 256 × 4096 = 18,769,510,400 ≈ 18.77 B token

not 5 B — **off by ~3.75×**. The `5B` string is in `SPARSEFORGE_SAME_HARNESS.md`'s title and in
the output path `outputs/cast_eval_spec/sparseforge_5b/`.
`AST_VS_SPARSEFORGE_DATA_CONFOUND.md` already computed 18.77 B for this same run — so the two
documents contradicted each other and nobody noticed.

Caveat on the caveat: `args.json` records `block_size: None`, so 4096 is the *runtime default*.
The token count is therefore derived, not read. Pin it from the run log before printing 18.77 B
in the paper.

## Verified run configuration (for the record)

From `out_llama/models_..._20260413_201320/args.json`:

| field | value |
|---|---|
| `dataset` | **`qa_format_sft_llama`** |
| `max_iters` | 17000 (checkpoint at `iter_num = 17900`) |
| `global_batch_size` | 256 |
| `block_size` | `None` (⇒ runtime default 4096) |
| `SLoRB` | **True** |
| `SLoRB_k` | 16 |
| `trainable_projection` | **True** |
| `learning_rate` | 1e-4 |
| `mask_hardening_start` / `_duration` | 12000 / 5000 |

## The real problem with this checkpoint is not identity — it is the corpus

`dataset = qa_format_sft_llama` is the training-data confound documented in
`AST_VS_SPARSEFORGE_DATA_CONFOUND.md`: benchmark-derived multiple-choice QA/SFT assembled from 8
benchmark *train* splits, 129,752,281 tokens on disk ⇒ **≈144.7 epochs**, while AST trained on C4
web text. RACE is both a training source and a CAST-7 eval task.

**That** is the load-bearing defect for this checkpoint — and confirming the identity makes it a
*stronger* statement than before, because the confound is now known to apply to the paper's
actual headline run rather than to some unidentified sibling.

## Related

- `SPARSEFORGE_SAME_HARNESS.md` CORRECTION block (2026-08-13): `hard_drop` is a post-hoc SLoRB
  amputation, not a trained 2:4 arm. **Unaffected by this resolution — that defect stands.**
- `AST_VS_SPARSEFORGE_DATA_CONFOUND.md`: the corpus confound, now confirmed on the paper's run.
