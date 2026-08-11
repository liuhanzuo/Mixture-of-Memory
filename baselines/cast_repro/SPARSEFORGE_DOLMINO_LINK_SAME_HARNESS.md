# SparseForge's dolmino-only link, scored on the CAST-repro harness

**Date**: 2026-08-11. **Node**: `.21` (8×L20A, wzc1). **Harness**: lm-eval `0.4.8`, repo git `b86c479`.
**Drivers**: `scripts/_sparseforge_dolmino_link_21.sh` (+ `scripts/_sf_l2_softfold_21.sh` for the
faithfulness control).
**Provenance**: `outputs/cast_eval_spec/sparseforge_dolmino_link2/` (per-arm `zeroshot_union9.json`,
`ppl4096/ppl_metrics.json`, `ppl2048/ppl_metrics.json`, `lm_eval.log`, `link2_summary.json`);
exports in `outputs/sparseforge_dolmino_link2_hf/{hard_fold,soft_fold,hard_drop}/`.

## Why this run exists

Commit `e44c742` established that the published SparseForge-5B arm trained on
`data/qa_format_sft_llama` — 8 multiple-choice-QA benchmark **TRAIN** splits, 129,752,281 tokens,
`repeat=3`, whose own `metadata.json` claims "no benchmark overlap" while listing `race_middle` +
`race_high`, and `race` **is** a CAST-7 eval task. So the published SparseForge-vs-CAST/AST
comparison is confounded by training data, not only by the SLoRB branch.

But that arm is the **last** of a resume chain, and the link before it trained only on dolmino —
the same general-corpus family CAST trained on. If that link's weights are competitive, the
algorithmic claim can be assessed on general data **without** a ~4-GPU-day data-matched retrain.
That is what this run did, for about one GPU-hour.

## 1. Verified lineage (from on-disk `args.json`, not from the task brief)

`resume_dir` is an **explicit absolute-ish path** in every `args.json` — the trainer does not have to
resolve it by convention. `main_llama.py:1443-1456`: `resume_ckpt_dir = args.resume_dir`, and only
falls back to `out_dir/last` / `last_dir.txt` when `resume_dir is None`. It is not None in any link
here, so the chain is unambiguous.

All paths below are under `/apdcephfs_wzc1/share_304376610/pighzliu_code/out_llama/`, prefix
`models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_`.

| link | dir | `dataset` | `resume` | `resume_dir` → | `resume_optimizer` | best iter | own AST-7 anchor | own wiki_ppl |
|---|---|---|---|---|---|---|---|---|
| 1 | `20260331_150310` | `dolmino-mix-1124-raw` | `False` | `None` | `True` (moot, no resume) | *(no `best_lm_eval.json`)* | — | — |
| 2 | `20260401_124938` | `dolmino-mix-1124-raw` | `True` | link 1 | **`False`** | 17600 | **59.2153** | **6.0970** |
| 3 | `20260413_201320` | **`qa_format_sft_llama`** | `True` | link 2 | **`False`** | 17900 | 57.2673 (CAST-7 task set) | 6.2155 |

**Confirmed**: links 1 and 2 used `dataset=dolmino-mix-1124-raw`; only link 3 used
`qa_format_sft_llama`. Link 2's weights therefore never saw MC-QA data. Both corpora are the LLaMA-2
tokenizer; `data/dolmino-mix-1124-raw/data/` is the raw HF dataset (`dclm`, `flan`, `math`, `pes2o`,
`stackexchange`, `wiki`, 1.6 TB). CAST-repro's `run_manifest.json` records
`data = Mixture-of-Memory/data/dolmino-mix-1124-llama2`, whose `metadata.json` says
`dataset: allenai/dolmino-mix-1124`, same tokenizer, `total_tokens 77,721,665,859`. **Same source
corpus, different tokenized snapshot** — see the limits section.

### ⚠️ Correction to the task brief: the two anchors are NOT the same task set

The brief compared link 2's 59.2153 against link 3's 57.2673 and concluded "link 2 is +1.948 pp
better on AST-7". **That comparison is invalid**: `args.json` shows

* link 2 `lm_eval_tasks = boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa` (**AST-7**)
* link 3 `lm_eval_tasks = hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,piqa,race` (**CAST-7**)

Two different 7-task means. The real, same-task-set comparison is measured below (and link 2 does
still win AST-7, by +0.40 pp — not +1.95).

### ⚠️ Second correction: link 3 resumed from `model.pt`, not the checkpoint scored here

Link 2's directory holds **two different models**:

| file | `iter_num` | extras | bytes |
|---|---|---|---|
| `model_best_lm_eval.pt` | **17600** | `best_lm_eval_mean=59.2153`, `finalization_done=True` | 41,078,444,091 |
| `model.pt` | **20000** | `optimizer_state_dict`, `slorb_state_dict`, `best_wiki_ppl=5.1614` | 45,772,988,056 |

Current `main_llama.py:1461-1476` prefers `model_final.pt` → `model.pt` → `model_best_lm_eval.pt`.
The **contemporaneous** April version hardcoded `model.pt`
(`SparseForge_worktree` `git show a7fd22f:legacy/main_llama.py`, line 970:
`model_pt_path = os.path.join(resume_ckpt_dir, "model.pt")`). Either way `model.pt` wins, so
**link 3 continued from link 2 @ iter 20000**, while the checkpoint scored in this report is
link 2 @ iter 17600. They are 2,400 iterations apart on the same dolmino trajectory.

I scored `model_best_lm_eval.pt` because it is the direct analogue of what was scored for link 3
(also `model_best_lm_eval.pt`) and it is the one carrying a published-style anchor. This is a
**sibling-of-the-parent** comparison, not literally "the weights link 3 started from".

### ★ Bonus finding: link 2 is a shared parent of SIX sibling branches

Surveying every April LLaMA-2-7B run with `resume_dir` → link 2:

| dir | dataset | own AST-7 anchor | wiki_ppl |
|---|---|---:|---:|
| `20260412_133441` | `dolmino-flan-heavy` | 58.3530 | 6.0507 |
| `20260412_190642` | `fineweb-edu-llama` | 59.0696 | 6.0935 |
| **`20260413_114730`** | **`benchmark_sft_llama`** | **65.7256** | 6.9024 |
| `20260413_170333` | `qa_format_sft_llama` | 58.4567 | 6.1016 |
| `20260413_190247` | `qa_dolmino_mix_llama` | 57.4684 | 6.0738 |
| `20260413_201320` | `qa_format_sft_llama` | 57.2673 (CAST-7) | 6.2155 |

The published arm is **one of six data-recipe branches off a common dolmino parent** — i.e. the final
stage was a data-selection sweep. The `benchmark_sft_llama` sibling reaches **65.73 AST-7** with
BoolQ 84.25 / RTE 79.42 / WinoGrande 75.69, which is *far* outside the range of every other arm and is
direct evidence that training on benchmark train splits inflates these exact metrics. That sibling was
not published; it calibrates how much of the published arm's per-task profile is data-driven.

## 2. Export

`tools/export_sparseforge_to_hf.py`, unmodified. Its real CLI is
`--ckpt --output --mask {hard,soft} --slorb {drop,fold} --model --project-root --dtype`; the four
variants are `--mask`/`--slorb` combinations, exactly as the docstring says. Link 2's state dict is
structurally identical to link 3's, so the tool's hard invariants passed untouched: **224** in-scope
tensors, **6,476,005,376** elements (`EXPECTED_SCOPE_TENSORS`/`EXPECTED_SCOPE_ELEMENTS`),
`SLoRB_Weight` 404,750,336 + `x_proj` 443,678,720 = **848,429,056**, weights dense
(`zero_frac 0.000000000`), mask continuous in `[2.76e-11, 1.0]` with ~1000 unique values per tensor.

Three variants exported: `hard_fold` (published deploy protocol), `hard_drop` (the amputation), and
`soft_fold` (faithfulness control = the model the trainer's own forward evaluated).

## 3. Results — plain acc (`acc,none`) only

Identical invocation for every arm, only `pretrained` differs:

```
lm_eval --model hf --model_args pretrained=<M>,dtype=bfloat16,parallelize=True,\
trust_remote_code=True,add_bos_token=False \
  --tasks boolq,rte,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa \
  --batch_size auto --num_fewshot 0 --seed 0 --trust_remote_code --log_samples
```

### Per task (plain acc)

| task | L2 `hard_fold` | L2 `soft_fold` | L2 `hard_drop` |
|---|---:|---:|---:|
| boolq | 72.4159 | 72.4159 | 69.1437 |
| rte | 69.6751 | 69.6751 | 56.3177 |
| hellaswag | 53.5252 | 53.6148 | 49.5220 |
| race | 40.3828 | 40.1914 | 36.9378 |
| piqa | 76.7682 | 76.6594 | 73.1774 |
| winogrande | 67.4822 | 67.4822 | 63.2991 |
| arc_easy | 75.5471 | 75.7997 | 71.9276 |
| arc_challenge | 41.6382 | 41.3823 | 36.3481 |
| openbookqa | 33.2000 | 33.4000 | 28.2000 |

### ★ Main table

`live params` = surviving in-scope weights (+ SLoRB where the branch is present). PPL is WikiText-2
via `baselines/eval_hf_sparse_model.py`; **the headline column is seqlen 4096, stated explicitly**,
because commit `501dafb` had to retract a conclusion drawn from a column that silently mixed 2048
and 4096. `2:4 status` gates on `tiles_gt2 == 0`.

| arm | data | AST-7 | CAST-7 | union-9 | wiki-ppl@4096 | (ppl@2048) | 2:4 status | live params |
|---|---|---:|---:|---:|---:|---:|:--:|---:|
| dense LLaMA-2-7B | — | **59.7847** | 56.4454 | 59.5625 | **5.2004** | 5.5637 | no (dense) | 6,476,005,376 |
| **L2 `hard_fold`** (dolmino) | dolmino-only | **59.0691** | 55.5063 | **58.9594** | **6.0985** | 6.5292 | **NO** (dense fold) | 4,086,431,744 |
| **L2 `soft_fold`** (dolmino, faithfulness ctrl) | dolmino-only | **59.1100** | 55.5043 | 58.9579 | 6.0980 | 6.5287 | **NO** (dense fold) | 4,086,431,744 |
| L3 `hard_fold` (published arm) | **MC-QA SFT** | 58.6696 | **57.2750** | 58.7671 | 6.2115 | 6.6510 | **NO** (dense fold) | 4,086,431,744 |
| CAST-repro @7500 | dolmino | 58.3856 | 54.3698 | 58.2837 | 6.1372 | 6.5268 | **YES** | 3,238,002,688 |
| AST-official | (AST's own) | 57.5976 | 54.3507 | 57.7540 | **5.9125** | 6.3430 | **YES** | 3,238,002,688 |
| **L2 `hard_drop`** (amputated) | dolmino-only | 53.5369 | 51.3446 | 53.8748 | 8.8329 | 9.3810 | YES | 3,238,002,688 |
| L3 `hard_drop` (amputated) | MC-QA SFT | 53.7562 | 51.4267 | 54.0393 | 8.8290 | 9.3770 | YES | 3,238,002,688 |
| Wanda 2:4 | — | 48.4873 | 46.2123 | 49.4569 | 11.7733 | 12.4749 | YES | 3,238,002,688 |

Rows for dense / L3 / CAST-repro / AST-official / Wanda are the previously measured values from
`outputs/cast_eval_spec/sparseforge_5b/sparseforge_same_harness_table.json`, same harness, same box.
The dense @2048 cell (5.5637) is quoted from SPEC.md:204, not re-measured.

⚠️ `hard_drop` is reported **strictly as amputation damage**, never as "SparseForge's 2:4 result".
The weights were *trained with* the SLoRB branch (`trainable_projection: true`), so removing it
deletes a live component of the trained function. Per the user's explicit instruction:
「我们sparseforge就是带SLoRB的啊，你不能直接简单把SLoRB删了然后就说我们效果差。训练的时候就带了这个。」

### Faithfulness control — the pipeline is validated on this checkpoint too

| | AST-7 | wiki_ppl@4096 |
|---|---:|---:|
| link 2's own `best_lm_eval.json` anchor | 59.2153 | 6.09699821 |
| our `soft_fold` (the variant the trainer's forward evaluates) | 59.1100 | 6.09797 |
| our `hard_fold` | 59.0691 | 6.09855 |
| **delta (`soft_fold` − anchor)** | **−0.1053** | **+0.0010** |

PPL reproduces the anchor to **0.001**. The whole −0.1053 pp accuracy delta is **one task, two
samples**: RTE anchor 70.3971 = 195/277 vs ours 193/277 = 69.6751 (Δ = 2 samples = 0.7220 pp / 7 =
0.1031 pp of the mean). Every other AST-7 task matches to ≤0.20 pp, with boolq and arc_challenge
**bit-exact** (72.4159, 41.6382). Export + scoring path are correct.

## 4. Sanity checks — all pass

* **RTE integrality** (n = 277, `acc` must be k/277): `hard_fold`/`soft_fold` **k = 193** exactly,
  `hard_drop` **k = 156** exactly. Extended the check to **all 9 tasks × all 3 arms**: every `acc`
  is integral in k. No transcription/merge error.
* **2:4 gate** (`tools/verify_2of4_hf_export.py`, `tiles_gt2 == 0`):
  ```
  hard_drop: elems=6,476,005,376 zeros=3,238,002,688 zero_frac=0.500000000
             tiles=1,619,001,344 tiles_gt2=0 tiles_lt2=0 exact_2of4_frac=1.000000000
             VERDICT: PASS      (PRE- and POST-inference, both rc=0)
  hard_fold: zeros=8 zero_frac=0.000000001 tiles_gt2=1,619,001,344   VERDICT: FAIL (by design)
  ```
* **Sparsity verified from actual weight tensors, not trainer counters.** Per the warning that
  `nm_magnitude_mask` writes exactly 2 True per group of 4 unconditionally (so `mask.sum()` is
  numel/2 even for all-NaN weights), I re-read the exported **safetensors directly**:
  `hard_drop` zero_frac `0.500000000`, `tiles_gt2 = 0`, `tiles_lt2 = 0`; `hard_fold` zeros = 8 of
  6.476 G, `tiles_gt2 = 1,619,001,344`. **NaN = 0, Inf = 0** in all 224 in-scope tensors of both.
  No trainer `aligned=`/`decayed=` counter was used anywhere in this report.
* **Both PPL seqlens asserted** in `ppl_metrics.json` (`"seqlen": 4096` / `"seqlen": 2048`), and the
  4096 runs consumed 335,872 target tokens = 82 × 4096.
* **Aggregator hard-fails on a missing task**; all 9 present for all 3 arms, and its slice means
  reproduce my independent recomputation to 4 dp.

## 5. ★ The honest read

**Does the CAST-7 margin survive on a dolmino-only trajectory? No — it inverts. But the general-data
picture is better for SparseForge than the contamination finding suggested.**

Same harness, plain acc, `hard_fold` (deploy protocol) throughout:

| slice | L2 (dolmino) | L3 (MC-QA) | L3 − L2 | L2 − CAST-repro | L2 − AST-official |
|---|---:|---:|---:|---:|---:|
| **AST-7** | **59.0691** | 58.6696 | −0.3995 | **+0.6835** | **+1.4715** |
| **CAST-7** | 55.5063 | **57.2750** | **+1.7687** | **+1.1365** | **+1.1556** |
| union-9 | 58.9594 | 58.7671 | −0.1923 | +0.6757 | +1.2054 |
| ppl@4096 (lower better) | **6.0985** | 6.2115 | +0.1130 | **−0.0387 (L2 better)** | +0.1861 (L2 worse) |

Three things follow, and they cut in different directions.

1. **The MC-QA stage bought CAST-7 and cost AST-7 and PPL.** Going from link 2 to link 3 gains
   **+1.7687 pp CAST-7** while *losing* **0.3995 pp AST-7** and *worsening* PPL by **0.1130**. A
   genuine capability gain should not be slice-selective in exactly the slice whose eval tasks
   overlap the training corpus. Per task, L3 − L2 is arc_c **+3.92**, arc_e **+2.65**,
   winogrande **+1.97**, obqa **+1.80**, hellaswag **+0.97**, race **+0.96**, piqa **+0.11**,
   boolq **+5.38** — but RTE **−19.49**. Every CAST-7 task rises while the one AST-only task that is
   not multiple-choice-QA (RTE) collapses by 19.5 pp. This is the signature of a narrow MC-QA
   specialisation, not a broad improvement, and it corroborates commit `e44c742` from a new direction.
2. **On general data SparseForge still beats both 2:4 baselines on accuracy** — +0.68 pp AST-7 and
   +1.14 pp CAST-7 over CAST-repro, +1.47 / +1.16 over AST-official, and it does so *without any
   benchmark-train exposure*. **The algorithm claim is in better shape than the contamination
   finding alone implied.** The dolmino-only arm is also the best non-dense arm on AST-7 in the whole
   table (59.0691 vs dense 59.7847 — only 0.72 pp of dense).
3. **But the accuracy win is still not a like-for-like win.** `hard_fold` is
   **not a 2:4 model** (`tiles_gt2` = all tiles) and spends **+26.20 %** live in-scope params
   (848,429,056 on top of 3,238,002,688 → 4,086,431,744) that CAST and AST do not have. On PPL@4096
   it **beats CAST-repro by 0.0387** (6.0985 vs 6.1372) but **loses to AST-official by 0.1861**
   (vs 5.9125). So the dolmino link partly overturns the earlier "SparseForge is last among the
   non-Wanda arms on PPL" reading — at matched seqlen 4096 the ordering becomes
   **dense 5.2004 < AST-official 5.9125 < L2 6.0985 < CAST-repro 6.1372 < L3 6.2115 ≪ Wanda 11.7733**.
   AST-official still wins PPL outright while using 26 % fewer live params.

**Net**: the contamination finding stands (the *published* arm's CAST-7 margin is data-driven and its
AST-7/PPL got *worse* on the way there), and simultaneously the *method* looks better than that
finding implied, because its clean-data ancestor beats both 2:4 baselines on accuracy. What is not
yet established is whether it beats them **at matched live-parameter count** — every accuracy win
above is a 4.09 G-live model against 3.24 G-live models.

### The RTE row is unusable in the published table (independent of everything above)

Provenance for the paper's SparseForge row is
`outputs/paper_v2/ast7_eval/sparseforge_5b_table2/ast7_eval.json`, which records
**`rte: 49.81949458483754`** (= 138/277), and
`SparseForge_Data/tables/cast9_dense_ast_current_harness.csv` carries the same 49.8195.
But `SparseForge_NIPS_2026/sections/experiments.tex:33` prints **69.82**:

```
SparseForge (Dolmino, 5B) & 78.47 & 69.82 & 54.65 & 69.85 & 76.39 & 43.86 & 35.20 & 41.63 & 76.33 & 60.69 & 6.2179
```

69.82 × 277 = **193.4014** — not an integer, so it cannot be an RTE accuracy at n = 277 at all. And
the printed AVG **60.69** is the mean *of those 9 printed cells including the impossible 69.82*
(recomputed: 60.6889); with the on-disk 49.8195 the AVG is **58.4666**, matching the CSV's own row
mean 58.4663. **So the +margin in the paper's AVG column is ~2.22 pp of pure transcription error**,
independent of SLoRB and of contamination.

Curiously, our measured L2 `hard_fold` RTE is **193/277 = 69.6751** — i.e. 193, the very numerator
that 69.82 rounds from. Substituting 69.6751 into the CSV row gives AVG 60.6724 ≈ the printed 60.69.
This is consistent with the published row having had **one cell spliced in from a different
checkpoint on this chain** (the table's own note admits "BoolQ/RTE come from a separate unified
invocation"), but I did not find a file that states that, so I am flagging the arithmetic coincidence
and **not** asserting the mechanism.

## 6. Explicit limits — what this does NOT establish

* **n = 1.** One checkpoint, one seed, no variance estimate. Every margin above (+0.68, +1.14 pp) is
  smaller than the **−0.3460 pp** cross-harness offset already measured on the AST arm, and no
  seed-level dispersion exists for L2 at all.
* **Still a 2-link resume with `resume_optimizer=False`.** Adam moments were discarded at the link
  1→2 boundary, so the trajectory is not a clean single training run.
* **Still +26.2 % live params via SLoRB** (848,429,056; 4,086,431,744 total live vs 3,238,002,688 for
  CAST/AST/Wanda). The accuracy comparison is **not** at matched capacity. Not tested: whether
  CAST-repro or AST-official would gain similarly from an added rank-16 SLoRB branch — the fair
  statement remains "the added capacity is doing work, and we have not isolated how much".
* **Token budget does not match CAST.** CAST-repro = 7500 × 256 × 4096 = **7,864,320,000** tokens.
  Link 2 @17600 = 17600 × 256 × 4096 = **18,454,937,600** nominal (and it inherits link 1's first
  4400 iters, so its own window is 13,200 iters ≈ 13.84 B). That is **~2.35× CAST's budget**. So
  link 2 is *not* the data-matched arm — it is data-*family*-matched and **budget-advantaged**. The
  brief's hope that "link 2 may already BE the data-matched arm" is **only half true**: the
  contamination is gone, the budget confound is not.
* **Different tokenized snapshot.** Link 2 read `data/dolmino-mix-1124-raw`; CAST read
  `Mixture-of-Memory/data/dolmino-mix-1124-llama2`. Same upstream `allenai/dolmino-mix-1124` and same
  LLaMA-2 tokenizer, but different shard/mix realisation — not verified to be the same token stream.
* **Also unmatched vs CAST**: `lr 1e-4` vs CAST's `2e-5`, FSDP hybrid-sharded vs CAST's DDP+ZeRO-1,
  distillation from the dense teacher (`distill_model: true`, `hardness_kldiv 1.0`) — CAST-repro's
  manifest is a different recipe. This run isolates **data content**, nothing else.
* **Not the exact parent of link 3**: link 3 resumed from `model.pt` @ iter 20000; this is
  `model_best_lm_eval.pt` @ iter 17600 (see §1).

### What is still needed to make the claim clean

1. **Matched-budget arm**: SparseForge on dolmino truncated to 7500 × 256 × 4096 = 7.864 B tokens,
   so the token count matches CAST exactly.
2. **Matched-capacity arm**: either SparseForge with SLoRB removed *during training* (not amputated
   post hoc), or CAST/AST *with* a rank-16 SLoRB branch added, to separate mask quality from
   added capacity.
3. **≥2 more seeds** on whichever arm becomes the headline, since the margins are ≈ the harness
   offset.
4. **Fix `experiments.tex:33`**: RTE 69.82 → 49.82 and AVG 60.69 → 58.47, or state on the record which
   checkpoint the BoolQ/RTE cells came from.
5. Optionally score link 2's `model.pt` @20000 (the literal parent) and the `fineweb-edu` sibling
   (59.0696 anchor, also clean data) to see whether "clean data ≈ 59" is stable across corpora.
