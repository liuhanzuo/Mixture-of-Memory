# Protocol-provenance gap in the keep14 trajectory scan (MAIN, 2026-08-13 05:35 GMT+8)

Written by MAIN *while* the scan was running on `.73` (step128000) + `.82` (step153500),
so it is a pre-result note, not a post-hoc rationalisation.

## What I independently verified from the output dirs

Read from `zwfy6:olmo2_closedbook_results/*128000*/summary.json`:

| field | value | status |
|---|---|---|
| `base_model` | `../models/OLMo-2-1124-7B` | correct vanilla anchor (guard G2 satisfied) |
| `ckpt` | `outputs/olmo2_probe2_7B_keep14fresh2/step128000.pt` | correct |
| `mode` | `pruned` | correct |
| `keep_front_layers` | 14 | correct |
| `add_bos` | `false` | matches frozen protocol |
| `max_new_tokens` | 32 | matches frozen protocol |
| `n_shards` | 8 | matches frozen protocol |
| nq_open `n` | 3610 | == `EXPECTED_N` exactly |

## The gap

**`summary.json` records neither `batch_size` nor `chat_template`.** Its `meta` block is
exactly:

```
mode / keep_front_layers / n_fresh_layers / num_hidden_layers /
ckpt_step / ckpt / base_model / add_bos / max_new_tokens
```

So the two most decision-critical fields **cannot be reconstructed from the artefacts
after the fact**. They are only knowable from the invocation.

Why each matters:

1. **`batch_size` must be 32.** A04's own `STATUS.json:full32_rescore_v2_20260812.sensitivity_bs48_probe`
   established that batch size *really moves items* on this harness — bs48 flipped
   12/14267 on popqa and 10/3610 on nq_open (bf16 numerics depend on left-pad width).
   If these two ckpts were scored at any other batch size, they are **not
   protocol-identical** to `keep14fresh2_step200k` or to the anchor, the NI margins are
   not comparable, and the "trajectory" is an artefact of a protocol change rather than
   of heal progress.
2. **`chat_template` must be False.** Repo-wide invariant; these are BASE LMs with no
   SFT/RL. Assert with `is not False` — **never** `is not True`, which passes silently
   on `None`.

## Required of the scan's own write-up

- Confirm both values **from the actual invocation** (driver/launch args), not by
  inference from `summary.json`.
- Record them, and *how* they were confirmed, in a `protocol_asserted` field of
  `evidence/a04_keep14_trajectory_ni.json` and in the verdict doc.
- If either differs from the frozen value: **do not publish the cells as comparable
  trajectory points.** Report protocol deviation and re-run at bs=32. A re-run is far
  cheaper than a non-comparable margin entering the record.

## Suggested durable fix (beyond this task)

The scoring harness should write `batch_size` and `chat_template` into `summary.json`'s
`meta`. Every future protocol audit on these dirs hits this same wall otherwise. Filed
here rather than patching the harness mid-run, because editing a running script is
prohibited.
