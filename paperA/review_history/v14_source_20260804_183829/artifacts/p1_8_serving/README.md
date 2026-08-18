# P1.8 repeated-query serving artifact

This directory is a read-only copy of the completed 2026-08-03 P1.8 serving
run from the H20 evaluation node. It contains the aggregate and 18 per-process
records for the full grid:

- store length: 32k, 128k, 1M tokens;
- placement: GPU-resident, CPU-pinned;
- generated tokens: 1, 32, 128, 512;
- three independent processes per (length, placement).

`p1_8_serving_aggregate.json` is the authoritative source for Paper A Table 3.
The break-even value is computed from measured component medians as
`(CoMem Write - j0 index)/(j0 per-query - CoMem per-query)`. An infinite value
means CoMem's measured per-query time was not lower, so no finite reuse count
amortizes the one-time Write under that cell.

The files contain timing/configuration data, not benchmark text or predictions.
