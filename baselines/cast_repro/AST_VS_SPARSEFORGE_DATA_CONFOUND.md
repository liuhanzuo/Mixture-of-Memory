---
scope: the training-data confound between SparseForge-5B and AST, found while scoping the AST+SLoRB reproduction
date: 2026-08-11 16:0x GMT+8
status: FINDING — decision-relevant for how AST+SLoRB must be run, and for what the existing CAST-7 margin means
severity: high — it plausibly explains the entire reported margin without any algorithmic contribution
---

# The finding

SparseForge-5B and AST are **not trained on the same kind of data**, and the difference lines up
exactly with where SparseForge's reported gains appear.

| | SparseForge-5B | AST official |
|---|---|---|
| dataset flag | `dataset = qa_format_sft_llama` | `dataset = 'c4_dataset'` (`main.py:97`) |
| what it is | multiple-choice QA/SFT, 8 benchmark train splits | C4 English web text (README §"Data Preparation") |
| train tokens on disk | 129,752,281 | C4-scale |
| tokens consumed | 17,900 × 256 × 4096 = 18,769,510,400 | 40,000 × 128 × block |
| **epochs over its own data** | **≈144.7×** | ≪1× |

Provenance: `data/qa_format_sft_llama/metadata.json` (wzc1, `pighzliu_code/data/`), and
`out_llama/models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260413_201320/args.json`.

## 1. `metadata.json` says "no benchmark overlap" — but RACE is a CAST-7 eval task

The file's own description field reads:

    "description": "QA format learning dataset for final finetune (no benchmark overlap)"

Its `benchmarks` list is:

    commonsenseqa, social_iqa, cosmosqa, sciq, race_middle, race_high, qasc, dream

**`race_middle` and `race_high` are the two constituent splits of RACE, and `race` is one of the
seven CAST-7 evaluation tasks.** The "no benchmark overlap" claim is false as written — it is
presumably true of the *AST-7* suite (boolq/rte/hellaswag/winogrande/arc_e/arc_c/obqa, no RACE),
but CAST-7 replaces boolq+rte with race+piqa, and that is the suite the SparseForge↔AST comparison
is reported on.

Note also `"repeat": 3` — each benchmark's train split is triplicated inside the 129.75M tokens,
*before* the ≈144.7× epoch wrap. RACE train material is therefore seen on the order of hundreds of
times.

Quantified on `SparseForge_Data/tables/cast7_ast_official.csv`:

    race: AST 38.8517 -> SparseForge 41.6268   (+2.7751 pp)
    that single cell supplies +0.3964 pp of the +2.3288 pp CAST-7 margin = 17.0 % of it

Dropping race gives a CAST-6 margin of **+2.2544 pp**, so race alone does *not* explain the result.
That is the narrow reading, and on its own it would be a minor errata. **The broader pattern is the
real problem.**

## 2. The per-task pattern is the pattern of MC-QA format adaptation, not of better sparsity

| task | AST | SparseForge-5B | Δ | format |
|---|---:|---:|---:|---|
| openbookqa | 30.00 | 35.20 | **+5.20** | MC-QA |
| arc_challenge | 39.51 | 43.86 | **+4.35** | MC-QA |
| arc_easy | 73.36 | 76.39 | **+3.03** | MC-QA |
| race | 38.85 | 41.63 | **+2.78** | MC-QA — **in the training set** |
| winogrande | 68.19 | 69.85 | +1.66 | cloze |
| piqa | 76.99 | 76.33 | −0.65 | cloze-ish |
| hellaswag | 54.71 | 54.65 | −0.06 | cloze/continuation |

The four largest gains are the four MC-QA tasks. The three flat-or-negative cells are the
continuation/cloze tasks. SparseForge trained ~144 epochs on eight MC-QA datasets in exactly that
answer format; AST trained on web text. **An arm that was format-adapted to multiple-choice QA
beating a web-text arm on multiple-choice QA is the expected result of the data difference alone,
with no sparsity-algorithm explanation required.**

This is consistent with, and independent of, the PPL direction recorded in `SPEC.md` (commit
`501dafb`): at matched seqlen 4096 SparseForge's WikiText-2 PPL **6.2179 is worse** than
AST-official's **5.9125**. Language-modelling quality goes the other way from the MC-QA accuracy.
A 144-epoch MC-QA finetune improving MC-QA accuracy while degrading held-out LM perplexity is the
signature of format/task adaptation, not of a better 2:4 mask.

## 3. Why this changes how AST+SLoRB must be run

The whole point of the AST+SLoRB reproduction is to isolate SparseForge's mask-learning machinery
(the 46 CLI flags AST lacks) from its shared infrastructure (SLoRB). If AST+SLoRB is trained on C4
and compared to SparseForge on `qa_format_sft_llama`, the comparison measures **data**, and the
mask machinery is unidentifiable. Concretely:

* **AST+SLoRB must be trained on `qa_format_sft_llama`**, same 17,000+3,000 iters, same
  `global_batch_size 256`, same `block_size 4096`, so that data, budget, and SLoRB are all held
  fixed and only the mask-learning differs. Reproducing "what AST's paper would have produced"
  (C4, 40k iters, `SLoRB_k=64`, `wanda`, `srste_decay 1e-4`) answers a different and much weaker
  question, and would hand us a win we did not earn.
* Because AST-official's published numbers are C4-trained, **the existing AST column cannot serve
  as the data-matched control at all.** That is an argument for running AST+SLoRB, not against it.
* Whatever margin survives a data-matched AST+SLoRB is the first number in this comparison that
  can be attributed to the algorithm.

## 4. Independent problems this does NOT fix

* **RACE must be reported as contaminated** wherever CAST-7 appears, or CAST-7 must be replaced by
  CAST-6. Training on `race_middle`+`race_high` and evaluating on `race` is not defensible with a
  footnote about "no benchmark overlap" that contradicts the manifest.
* The other seven training benchmarks (commonsenseqa, social_iqa, cosmosqa, sciq, qasc, dream) are
  not in CAST-7 or AST-7, but they are **near-neighbours** of arc/obqa in format and often in
  source material. A same-task overlap audit (n-gram or exact-question match between those train
  splits and the arc_easy / arc_challenge / openbookqa test sets) is cheap, CPU-only, and should be
  run before the +5.20 obqa and +4.35 arc_c cells are used as evidence.
* The "5B tokens" label on this run is separately wrong: nominal consumption is **18.77 B**.

## 5. Feasibility facts found in the same pass (relevant to launching AST+SLoRB)

Read from `pighzliu_code/baselines/ast_official_clean/` (note: **not** under
`Mixture-of-Memory/baselines/`, which contains only `cast_repro`):

* **AST has plain DDP only** — `main.py:27` imports `DistributedDataParallel`; there is no FSDP and
  no deepspeed anywhere in the repo. This is exactly why SparseForge *added* `use_fsdp` /
  `use_deepspeed` as new flags.
* **Our own 7B run required FSDP**: `args.json` has `use_fsdp=True`, `fsdp_mode=hybrid_sharded`,
  `fsdp_mixed_precision=True`, `gradient_checkpointing=True`.
* Plain-DDP per-rank static memory for fp32 AdamW at 7B + SLoRB(k=16) ≈ **113 GiB**
  (28.3 params + 28.3 grads + 28.3 m + 28.3 v), before activations and before the **second resident
  7B teacher** that `distill_model=True` implies (+~12.6 GiB bf16). Against **H20 = 97.8 GiB** this
  does not fit; against **L20A = 183 GiB** it is tight but plausible only with bf16 params and
  gradient checkpointing.
* `main.py:266` has `model.enable_gradient_checkpointing(...)` **commented out**, so the
  `--gradient_checkpointing` flag it parses at line 72 may be inert — must be verified before
  relying on it.
* AST's loader is `np.memmap(data_dir/'train.bin', dtype=np.uint16)` (`main.py:178`). Our
  `qa_format_sft_llama/{train,val}.bin` are uint16 (`dtype.txt`), so the data is **directly
  loadable** by AST's code with only a `--dataset` path change. This is the one thing that is easy.
* `data/c4_dataset/` in the AST checkout contains **only `prepare.py`** — no `.bin`. AST's own data
  would have to be built from scratch (another reason to run the data-matched design instead).

**Consequence: AST+SLoRB at 7B cannot be launched from the AST repo as-shipped.** It needs either
a sharding backend added, or L20A (`.21`) with bf16 + working gradient checkpointing, and the
checkpointing path must be repaired first. This is a real engineering cost that must be priced
before committing GPU.

## 6. What to do

1. Run AST+SLoRB **data-matched** on `qa_format_sft_llama`, not C4. (Design detail in
   `AST_SLORB_REPRO_PREREG.md`.)
2. Fix `metadata.json`'s "no benchmark overlap" description, which is false for CAST-7.
3. Report CAST-6 (race excluded) alongside CAST-7, with the contamination stated.
4. Run the CPU-only overlap audit of the 8 training benchmarks against arc_easy / arc_challenge /
   openbookqa test sets.
5. Do not describe the CAST-7 margin as an algorithmic result until 1–4 are done. At present the
   per-task pattern is equally consistent with pure MC-QA format adaptation, and the PPL direction
   actively favours that reading.
