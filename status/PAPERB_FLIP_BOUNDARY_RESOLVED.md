# RESOLVED: the flip boundary was a torch version change — and Table 4 has a separate batch-size defect

**Date**: 2026-08-08 ~09:3x CST. **Found by**: agent `a71d3994` (torch version); MAIN verified
independently. **Supersedes the "cause unidentified" verdict** in
`PAPERB_HARNESS_DRIFT_REVISION.md` and `PAPERB_HARNESS_BOUNDARY_BISECT.md`.

## The cause: torch 2.7.0 vs 2.13.0

The v1 launcher for the ShortGPT-16 rung is `scripts/_run_shortgpt_downstream_only.sh`
(mtime **Aug 2 04:59**, three minutes before v1's summary at 05:02). It sets:

```
PY=$WD/olmo2_venv/bin/python      # torch 2.7.0+cu126
```

Every v2/v3 battery uses `/opt/conda/envs/torch-base/bin/python` = **torch 2.13.0**. MAIN verified
both interpreters on .73 directly. **That is the boundary variable** — not the driver source (whose
diff is purely additive), not the datasets (Arrow MD5s identical), not batch size (the v1 shortgpt
launcher also used `--batch_size 8`).

So the Aug 2 20:12 driver-scp was **coincident with, not causal of**, the boundary. The real
change was which Python environment the eval ran under. Both my "driver drift" framing and my
"batch size" hypothesis were wrong about *this* boundary.

## What batch size actually showed (a real, separate finding)

Agent `a71d3994` ran a 4-arm batch-size sweep on ShortGPT-16 (same ckpt, same node, same driver,
same torch), measuring exact per-item flips:

| comparison | total flips | of which near-tie (<0.1 nats) |
|---|---:|---:|
| bs4 vs bs8 | 90 | 63 |
| bs8 vs bs16 | 107 | 84 |
| bs4 vs bs16 | 119 | 88 |
| bs4 vs bs32 | 117 | 92 |
| bs8 vs bs32 | 123 | 100 |
| bs16 vs bs32 | 70 | 59 |
| **v3 vs bs8 (both bs8, same torch)** | **0** | 0 |

**Batch size changes ~90-120 per-item outcomes**, ~70-80% of them on options separated by <0.1 nats.
Mechanism: batch membership sets each batch's pad-to-max length, which perturbs the bf16-autocast
log-softmax, which flips near-ties. The `v3 vs bs8 = 0` row is the control that makes this clean —
hold torch *and* batch size fixed and the harness is bit-deterministic, as established in
`PAPERB_HARNESS_DRIFT_REVISION.md`.

Note the aggregate hides this: bs8 vs bs16 is **107 exact flips but only 16 net** and +0.078 pp on
core6. Per-item churn is ~7× the net movement. Any paper claim resting on per-item pairing
(McNemar, paired bootstrap) is far more exposed to this than the headline accuracies are.

## The actionable defect: Table 4's base row is off-protocol

`7B_base_full` — the published base row — came from `_run_olmo2_probe2_downstream_8gpu.sh` (Jul 19)
which sets **`BS=16`**. Every pruned rung ran at **`BS=8`**. So base and pruned are measured under
different eval protocols. Same launcher also produced `1B_base_full`, `1B_keep7_step*`, and
`7B_keep10_step10000` at BS=16.

**Re-eval dispatched** (`scripts/_run_olmo2_base_bs8_73.sh`, .73, 09:31): vanilla base at BS=8,
conda torch 2.13.0, `chat_template=False`, `--save_per_example`, `assert_8shards`, output
`7B_base_full_bs8` so it cannot clobber the published number.

## The clean Table 4 is already on disk

zwfy6 `_v2` batteries — **all BS=8, all conda torch 2.13.0, all full `n_scored`, all at the paper's
own steps** — verified by MAIN:

| rung | dir | core6 |
|---|---|---:|
| ShortGPT-16 | `7B_shortgpt16_step200000_v2` | .62247 |
| keep14@200000 | `7B_keep14_step200000_v2` | .59532 |
| keep12@124000 | `7B_keep12_step124000_v2` | .56888 |
| keep10@83500 | `7B_keep10_step83500_v2` | .52999 |
| keep8@121000 | `7B_keep8_step121000_v2` | .52328 |
| base full32 | *pending* `7B_base_full_bs8` | (BS=16 value was .70368) |

Five of six rungs need no new compute. One 6-minute job closes the sixth. **This is the version of
Table 4 to publish**: single disk, single architecture (H20 cc9.0), single torch, single batch size,
full shards, paper's own steps.

## Protocol note Paper B must carry

The appendix needs to pin, and the harness should assert: **torch version, eval batch size,
`num_shards`, `chat_template`, `add_bos`, and per-task `n_scored == expected`.** Tonight produced
four separate ways to move core6 without touching the model — torch version (~20 flips), batch size
(~107 flips), partial shard merge (keep12 arc_easy, +0.19 pp), GPU architecture (7-29 flips) — and
only the last is physically interesting. All four are silent by default.

## Correction record

Three of my own attributions for this boundary were wrong, in order: "harness nondeterminism /
runtime jitter" → "driver drift" → "batch size". The agent's torch-version finding is the one that
holds up, and it was reached by checking the v1 launcher's `PY=` line, which I read past twice.
Also, my "Aug 4 HF cache lock" clue was a false lead: both launcher generations set
`HF_DATASETS_CACHE` to the project directory, so `~/.cache/huggingface` was never in the read path.

## Provenance

- `zwfy6:scripts/_run_shortgpt_downstream_only.sh` mtime Aug 2 04:59, `PY=$WD/olmo2_venv/bin/python`
- `zwfy6:olmo2_venv/bin/python` → torch `2.7.0+cu126`; `/opt/conda/envs/torch-base/bin/python` → torch `2.13.0`
- agent report `status/PAPERB_BATCHSIZE_FLIP_CAUSE.md`, commit `66535d4`; scripts
  `_batchsize_flip_exp_82.sh`, `_batchsize_flip_exp_104.sh`, `analyze_batchsize_flips.py`
- superseded: `PAPERB_HARNESS_BOUNDARY_BISECT.md` (`29b18ca`) "cause cannot be identified";
  `PAPERB_HARNESS_DRIFT_REVISION.md` (`c6bdc05`, `cb26f17`)
