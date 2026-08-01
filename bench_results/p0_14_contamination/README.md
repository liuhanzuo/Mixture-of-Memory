# P0.14 — InfiniteBench / PG-19 train-eval contamination audit

**Date:** 2026-08-02 · **Node:** wzc1 local (diskB `share_304376610`), **CPU-only, NO GPU touched** · **Owner:** Paper A P0.14

This directory contains the required contamination audit for the InfiniteBench
natural long-document quality numbers in `paperA/sections/tab_infbench.tex`
(Book-QA F1 **6.06**, Book-choice acc **17.47**), which the flagship LoRA
(`outputs/qcmem_distill_qwen_j12_r32_4k/final`) produced. That adapter is
distilled **only** on PG-19 train text (`data/pg19_train.jsonl`, verified via its
sibling `distill_args.json`). PG-19 and InfiniteBench `longbook_*` both draw on
public-domain Project-Gutenberg books, so we must rule out train-on-test overlap.

## Bottom line

- **Eval docs are heavily contaminated by the PG-19 training corpus.** Of **86
  unique books** behind the **580 eval records** (351 QA + 229 choice), the
  13-gram containment distribution is sharply **bimodal**: **31 books < 0.10**
  (30 essentially 0.000, i.e. absent from training) vs **54 books ≥ 0.60**
  (present in PG-19 train). The band 0.18–0.60 is **empty**.
- **Headline (conservative, ≥0.80 cut):** 24/86 books, **163/580 records =
  28.1%** contaminated.
- **Honest level (any cut in 0.20–0.60, identical result):** **54/86 books =
  62.8%**, **387/580 records = 66.7%** contaminated. The ≥0.80 cut *undercounts*
  because InfiniteBench **anonymizes character names** (e.g. QA #0 = Woolf,
  *To the Lighthouse*, "Mrs Ramsay"→"Mrs Bronwyn"), which breaks a fraction of
  13-grams even in books that are otherwise verbatim in training — dragging
  genuinely-in-training books down to 0.60–0.92 instead of 1.0.
- **Remediation is NOT possible on this node.** The per-example predictions live
  on the off-limits GPU node `.73` (zwfy6), not on wzc1. Clean-subset F1/acc
  therefore **cannot** be recomputed here without a model run or touching that
  node. → **Recommend WITHDRAW or RELABEL the InfiniteBench quality comparison**
  (`tab:infbench`): keep it only as the bounded-read **coverage / memory stress
  test** it already is described as in `07_limitations.tex`, and drop the
  QA-F1 / choice-acc quality claim (or recompute on the clean subset once the
  predictions are transferred — the helper is ready, see below).

## Methods (three, per the task spec)

| Method | Result | Note |
|---|---|---|
| (a) title / author / PG-ID intersection | **NOT COMPUTABLE** | Neither side carries this metadata (eval records = `{id,context,input,answer,options}`; training loader concatenates raw lines with no doc boundary). |
| (b) exact document hash | **0 matches** (structural) | Eval contexts are anonymized whole books; training corpus has no doc boundary → a book-level exact match cannot exist by construction. |
| (c) **13-gram containment** (bottom-hash/MinHash sketch, 1/32 downsample, xxh64 seed 0) | **decisive** | Fraction of a book's unique sampled 13-gram hashes that appear anywhere in the PG-19 training sketch (59.1M unique hashes). |

**Critical correctness fix (documented):** the training side is reflowed into a
**continuous** token stream to match the eval side's whitespace collapse. PG-19's
raw text is hard-wrapped at ~13 words/line, so tokenizing per-line would form
almost no 13-grams (and none across a wrap) and would report ~0 containment even
for an identical book — a false-CLEAN bug. Caught in the first run and fixed;
the reflow only *raises* training coverage (the safe direction for an audit).

## Verification

Manual spot-check confirms the audit is correct in both directions:
- A **0.999-containment** book's mid-document 14-word run (name-free narration)
  is found **verbatim** in `pg19_train.jsonl`.
- A **0.000-containment** book's equivalent run is **absent**.
(See `verification.txt`.)

## Artifacts

| File | Contents |
|---|---|
| `audit_summary.json` | top-line counts, ratios, thresholds |
| `match_list.json` | per-unique-book: containment, n-gram counts, verdict, the (task,id) records it maps to |
| `per_record_verdict.jsonl` | one row per eval record (task,id) → book verdict |
| `clean_subset_ids.json` | eval (task,id) records whose book is CLEAN (containment < 0.10): QA **113**, choice **76** at the ≥0.80/<0.10 bands |
| `threshold_sensitivity.json` | contamination vs cut ∈ [0.05,0.90] — shows the bimodal robustness |
| `thresholds.json` | exact params/definitions |
| `data_manifest.json` | every data path, size, hash, role, metadata availability |
| `clean_subset_recomputed.json` | output of the recompute helper (currently: no predictions on this node) |
| `verification.txt` | the high/low phrase spot-check result |

## Scripts (in `scripts/`)

- `scripts/audit_p0_14_contamination.py` — builds the PG-19 13-gram sketch and
  scores every unique eval book. Reproduce:
  ```
  python scripts/audit_p0_14_contamination.py \
    --train_corpus data/pg19_train.jsonl --eval_dir <infbench jsonl dir> \
    --out_dir bench_results/p0_14_contamination \
    --n 13 --downsample 32 --contam_high 0.80 --contam_low 0.10 --workers 96
  ```
  Eval JSONLs are the public `xinrongzhang2022/InfiniteBench`
  `longbook_qa_eng.jsonl` + `longbook_choice_eng.jsonl` (CPU HF download).
- `scripts/recompute_p0_14_clean_subset.py` — **remediation helper**: given the
  existing per-example predictions and `clean_subset_ids.json`, recomputes the
  official InfiniteBench metric on the CLEAN subset only (reuses the exact
  scorers from `eval_qcmem_infbench.py`; no model run). A **no-op on wzc1** since
  the predictions are on the GPU node; run it there / after transfer to get the
  clean numbers.
