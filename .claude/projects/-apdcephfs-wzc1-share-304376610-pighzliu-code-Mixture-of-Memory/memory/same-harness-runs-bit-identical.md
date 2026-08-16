---
name: same-harness-runs-bit-identical
description: "Same-harness re-runs are byte-identical, not noisy — treat \"±0.2 pp floor\" as a harness-version boundary suspect until proven otherwise"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

Under the OLMo-2 downstream eval harness, **same-arch same-disk re-runs of the same checkpoint
under the same harness version are BYTE-IDENTICAL** — 0 flips across all core6 tasks. Verified
2026-08-08 on keep14 v1-vs-v2, keep8 v2-vs-v3, shortgpt16 v2-vs-v3.

**Why**: I spent hours tonight writing up a 15-20 flip "within-disk noise floor" (see
`status/PAPERB_WITHIN_DISK_FLOOR.md`) that seemed to disqualify a cross-architecture damage-
scaling claim. When I finally ran the actual within-cell control (same-harness same-arch pair
on the same disk), the floor turned out to be 0. The 18-flip observation that motivated the
retraction was between old-harness v1 and new-harness v2, i.e. a **code-version boundary**,
not runtime jitter. The `assert_8shards` guard was added between v1 and v2 and probably fixed
a silent partial-merge bug (see `PAPERB_TABLE4_KEEP12_PARTIAL_MERGE.md`), which is a plausible
mechanism for the "floor".

**How to apply**: If someone (including me) claims that eval numbers are "irreproducible to
about ±X pp" without a same-code re-run pair on file, treat the number as a harness-version
suspect until they've verified. The control that pins it down is **same-arch same-disk
same-harness N≥2 re-runs of the same ckpt**; run that first, before treating floors as
physical. Also see [[cluster-two-disks-not-shared]] since floors reported "cross-disk" are
often cross-harness too — the two disks were checked out at different code versions.

Corollary: **downstream aggregators must assert `n_scored == expected_task_size` per task on
merge**, not just check `n_nan == 0`. The keep12 arc_easy partial-merge (`.5669` published,
`.5689` corrected) survived only because the aggregator did not check this. If a re-run
returns +0.19 pp on a rung expected to be reproducible to sub-permille, the first hypothesis
is a partial-merge silent bug, not a physical effect.
