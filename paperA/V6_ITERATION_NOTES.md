# Paper A v6 iteration notes

Date: 2026-08-04 (working manuscript; not frozen).

## Scope

This pass edited only the current `paperA/` manuscript and Paper A-local
analysis artifacts. It did not modify `review_history/`, review scores, Paper B,
or any training/evaluation state. No model evaluation, teacher forward, or
training was run.

## Saved-score reanalysis

Script:

`analysis/equal_latency/reanalyze_equal_latency.py`

Result:

`analysis/equal_latency/equal_latency_dependence_results.json`

The source is the completed saved equal-latency anchor:
CoMem `k=12` versus latency-selected replay `k=10`, for BM25 and frozen BGE.
Remote saved shards were reduced to Paper A-local score-only exports containing
cell, example ID, arm scores, and paired difference. They contain no benchmark
text, gold answers, prompts, or predictions. Aggregate run records were also
copied locally. Source and output hashes are in:

* `analysis/equal_latency/source/SHA256SUMS.txt`
* `analysis/equal_latency/SHA256SUMS.txt`

Definitions:

1. **Stratified fixed-cell paired bootstrap:** keep all nine cells; within every
   cell resample exactly its fixed `n=100` paired examples; equally average the
   nine resampled cell means.
2. **Hierarchical bootstrap:** resample nine cell labels with replacement, then
   independently resample `n=100` paired examples within every selected cell
   occurrence; equally average the nine cell means.
3. **Leave one cell out:** omit each cell and equally average the other eight.
4. **Pooled IID:** pool 900 paired differences and resample 900; retained only
   as the original-analysis sensitivity.

All bootstraps use 100,000 replicates and deterministic base seed 20260804.

Results, CoMem minus replay in percentage points:

| Selector | Point | Stratified 95% CI | Hierarchical 95% CI | LOCO range | Pooled-IID sensitivity |
|---|---:|---:|---:|---:|---:|
| BM25 | -11.56 | [-14.33, -8.78] | [-18.67, -5.11] | [-13.13, -9.50] | [-14.44, -8.67] |
| BGE | -1.00 | [-4.56, 2.56] | [-10.67, 8.33] | [-3.50, 1.75] | [-4.67, 2.67] |

The first 100 LoCoMo items are all conversation 0. There is one observed
conversation cluster, so conversation-cluster resampling is not identifiable
and was not fabricated.

## Review issues handled

* Abstract 32k crossover now explicitly reports CPU-pinned `8.9--10.9`;
  body/appendix also state the full GPU/CPU `5.5--10.9` grid.
* LongEval/LongBench matched `j=0` prose now points to the actual matched rows in
  Table `tab:overview`; task tables are described as `j=12`/external breakdowns.
* Unsupported MemoryLLM prompt-sensitivity assertions were removed.
* TurboRAG pages corrected to 6599--6612.
* Equal-latency main table, protocol table, abstract, body, conclusion,
  limitations, and statistics appendix now use dependence-aware inference and
  retain pooled IID only as sensitivity.
* Abstract mechanically counts to 197 words under the same simple stripped-TeX
  counting approach used in the v5 review.

## Still blocked without new experiments

* No matched nearest reusable-KV/PIC/modular-cache baseline.
* No clean same-effective-batch replications of the exact headline adapter.
* No natural-task validation of overlap/contextual Write or concurrent p95
  serving.
* Teacher top-64 retained mass cannot be recovered from saved logs; the
  limitation remains and no teacher run was launched.

## Build

The local environment lacked a `latexmk` executable, so compilation used the
already installed Tectonic 0.17 XeTeX-compatible engine and cached bundle.
Output `main.pdf`:

* A4, 24 pages;
* main text ends on page 8; appendix begins page 11;
* zero overfull boxes, zero undefined references/citations, zero LaTeX/package
  warnings in the final TeX log;
* underfull box diagnostics remain (also present in v5);
* all fonts embedded; no Type 3 fonts.

`PDF_SHA256.txt` records the current working PDF hash. The manuscript was not
frozen into `review_history/`.
