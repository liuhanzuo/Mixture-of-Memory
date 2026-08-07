# Table 4's keep12 row includes a partial-merge arc_easy — 6/8 shards, not 8/8

**Date**: 2026-08-08 ~06:5x CST. **Verified by**: MAIN, by comparing v1 and v2 batteries of the same
ckpt on the same disk on the same architecture. **GPU cost**: 0 — both batteries already on disk.

## Discovery path

I was testing whether the within-disk core6 floor (keep10: 18 flips) generalizes across rungs. Ran a
same-disk-same-arch flip count for shortgpt16, keep10, keep12 (zwfy6 v1 vs `_v2`). Got:

| rung | Σ\|net flips\| | Δ core6 pp |
|---|---:|---:|
| keep10 | 18 | −0.031 |
| shortgpt16 | 20 | +0.098 |
| **keep12** | **437** | **+0.194** |

437 is not a bf16 kernel effect — it's an order of magnitude out of family. Investigated the
per-task counts:

| task | v1 `n_scored` | v2 `n_scored` |
|---|---:|---:|
| hellaswag | 10042 | 10042 |
| arc_challenge | 1172 | 1172 |
| **arc_easy** | **1782** | **2376** |
| piqa | 1838 | 1838 |
| openbookqa | 500 | 500 |
| winogrande | 1267 | 1267 |

`1782 = 2376 × 0.75` exactly — **v1 merged only 6/8 shards on arc_easy**, silently, and averaged
that partial result into its core6. Neither eval's `summary.json` flagged it (`n_nan` is 0 on both;
the harness's own "n_scored" is honest — 1782 — but the aggregation didn't check `n_scored ==
task_full_size`).

Same ckpt path in the meta of both evals, same `ckpt_step=124000`, and the base file mtime is
Aug 2 10:08 (untouched since). So this is a merge-side bug, not a weights difference.

## Impact on the paper's Table 4

The paper's headline **keep12 core6 = `.5669`** matches v1's partial-arc_easy value `.56694`. The
correct full-shard v2 value is `.56888` → rounds to **`.5689`**, a **+0.194 pp** correction —
significantly above the within-disk floor of ~0.03–0.10 pp seen on other rungs, and comparable to
the cross-architecture core6 span itself.

So Table 4's keep12 row is **not just measured on a different architecture than the base row**
(finding earlier tonight in `PAPERB_TABLE4_ARCH_AUDIT.md`) — it is also **measured on a partial
eval** for one of its six component tasks. Both defects sit on the same row.

**Consequence for the paper**: the keep12 row must be quoted as `.5689` (the v2 full-shard value)
or as `.5669*` with an explicit note (I would not recommend the latter — a `*` inviting explanation
is a footnote a reviewer will spend time on). Adjacent ladder rungs differ 2.7–3.7 pp, so the
correction does not flip the ordering, but the number that goes in the table must be the corrected
one.

## Why this survived until now

The `assert_8shards` guard that flags partial merges was added tonight (referenced in memory
`kill-remote-gpu-job-by-pid-not-pkill`). Every v2 battery run tonight enforces it and would have
caught this at merge time. The v1 eval predates that guard — the merge summed 6 of the 8 shard
files that happened to be on disk, wrote a `summary.json` with correct-looking per-task keys but
`n_scored=1782`, and the aggregator averaged it as if complete.

## Should I check the other four Table 4 rungs?

Yes, and I already have the data. keep10 and shortgpt16 v1-vs-v2 flip counts are 18 and 20 —
consistent with a real ~15-25-flip floor and no per-task `n_scored` mismatches. So the same-disk
sanity check that surfaced this can be run on the two remaining rungs I have not yet audited (keep8
v2 does not exist on zwfy6 per my earlier check; base full32 was measured once). This is CPU-only
and I will add it if any further Table 4 row shows a `n_scored` mismatch.

## Broader lesson

Every downstream/PPL/MMLU summary in Paper B's results tree should carry an explicit
"total_task_size" field that the aggregator asserts against `n_scored`, and every legacy summary
should be scanned once for `n_scored < expected` on any task. That is the one-time-cost fix that
prevents a class of silent Table-4-defects across the whole paper. **Task added.**

Also: **`v1` here is Table 4's authoritative source**. The `_v2` batteries I ran tonight for
per-item-preds purposes now double as **byte-for-byte shard-integrity audits** of the paper's
figures. That is unexpected additional value out of what I dispatched for a different reason.

## Provenance

- v1 (Table 4 source): `zwfy6:olmo2_downstream_results/7B_keep12_step124000/summary.json` — meta `ckpt=outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt` `ckpt_step=124000` — arc_easy `n_scored=1782 (=6/8 shards)`
- v2 (this tonight): `zwfy6:olmo2_downstream_results/7B_keep12_step124000_v2/summary.json` — same ckpt path — arc_easy `n_scored=2376` (full)
- ckpt file: `43,867,047,810 B, Aug 2 10:08, untouched`
- Related: `PAPERB_TABLE4_ARCH_AUDIT.md`, `PAPERB_TABLE4_BUDGET_DEFECT.md`, `PAPERB_WITHIN_DISK_FLOOR.md`
