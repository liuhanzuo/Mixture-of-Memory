# Equal-latency dependence-aware reanalysis

This directory contains a **statistics-only** reanalysis of the saved Paper A
equal-latency experiments. No model evaluation, teacher forward, or training is
performed by the script.

## Reproduce

From the repository root:

```bash
python paperA/analysis/equal_latency/reanalyze_equal_latency.py \
  --n-boot 100000
```

The command rewrites:

`paperA/analysis/equal_latency/equal_latency_dependence_results.json`.

Python's standard-library `random.Random` is used with base seed `20260804`.
The BGE streams use a deterministic seed offset of 1,000,000. Percentile
intervals use linear interpolation at 2.5% and 97.5%.

## Estimand and bootstrap definitions

Every effect is the percentage-point paired score difference
`CoMem(k=12) - replay(k=10)`.

* **Nine-cell stratified paired bootstrap:** retain all nine observed cells;
  within each cell resample exactly its fixed `n=100` paired examples; take the
  equal-weight mean of the nine resampled cell means.
* **Hierarchical bootstrap:** first resample nine cell labels with replacement;
  for every selected occurrence independently resample `n=100` paired examples
  within that cell; equal-weight the nine resulting cell means.
* **Leave one cell out:** omit each observed cell in turn and equal-weight the
  remaining eight cell means.
* **Pooled IID sensitivity:** pool all 900 paired differences and resample 900
  pairs, matching the original v5 analysis. It is retained only as sensitivity.

The first 100 LoCoMo items are all from conversation 0. Thus the retained slice
has only one observed conversation cluster, so a conversation-cluster bootstrap
is not identifiable and is deliberately not reported.

## Source and privacy

`source/bm25` and `source/bge` contain:

* copied aggregate JSON records (`decision`, `summary`, `anchors`, `frontier`,
  `manifest`, and `sanity`) from the completed saved runs; and
* publication-safe paired score exports with cell, example ID, arm scores, and
  paired difference only.

The score exports contain **no benchmark text, gold answers, or predictions**.
They replace remote/internal absolute paths with Paper A-local relative paths.
Hashes are listed in `source/SHA256SUMS.txt`; script/result hashes are in the
parent `SHA256SUMS.txt`.

Saved-run source directories at extraction time:

* BM25: `bench_results/p0_20_eqlat`
* BGE: `bench_results/p0_20_phaseB_dense`

Only the frozen latency-selected anchor was exported:
CoMem budget 12 versus replay budget 10.
