# Union-9 harness rebuild — same-arm reproduction control

**Verdict: PASS.** The rebuilt pinned harness reproduces an archived union-9 arm
**exactly** — 0 metric flips, 0 doc/prompt/target hash mismatches, all 9
`task_hashes` identical, and **all 73,840 raw continuation logprobs
bit-identical (max |Δ| = 0.000e+00)**. The rebuilt stack is admissible for
scoring the two SparseForge ±SLoRB token-matched arms.

Date: 2026-08-14. Control run on **LOCAL** (= `28.89.19.21`, 8×B200 sm_100),
GPUs 0-3, alongside the live `noslorb` training arm. Cost **≈1.1 GPU-h**
(4 GPUs × 17.8 min wall: lm_eval 13:55:57 → 14:13:20).

---

## 1. Why a control was required

The 2026-08-13 22:2x node restart destroyed the pinned union-9 harness
(`lm_eval 0.4.8` + `transformers 4.57.6`) on every node. It was rebuilt as an
isolated venv at `/apdcephfs_wzc1/share_304376610/pighzliu_code/venv_union9`
(deliberately *not* by touching the conda env, which now has
`transformers 5.15.0` and is the env the two live training arms run in).

Matching version *strings* are not evidence of a matching harness. This project
already retracted a headline conclusion built on a cross-harness comparison
whose AST-7 offset was **−0.346 pp**
(`baselines/cast_repro/SPARSEFORGE_SAME_HARNESS.md`, CORRECTION block). So the
rebuild had to be shown to reproduce an already-archived arm before being used
to add a row.

**Bar = 0 flips**, not "within noise", per memory
`[[same-harness-runs-bit-identical]]`: same-arch / same-disk / same-harness
re-runs were previously measured byte-identical.

## 2. Arm chosen: `dense_ref`

`dense_ref` = `models/Llama--Llama2-7b`, unmodified Llama-2-7B. Chosen because
it is the one arm whose weights are *provably* unchanged, so a per-task delta
could not be blamed on the checkpoint. sha256 verified identical on **both**
physical disks:

| file | sha256 | wzc1 | zwfy6 (`.73`) |
|---|---|---|---|
| `model-00001-of-00002.safetensors` | `4ec71fd53e99766de38f24753b30c9e8942630e9e576a1ba27b0ec531e87be41` | ✓ | ✓ |
| `model-00002-of-00002.safetensors` | `41780b5dac322ac35598737e99208d90bdc632a1ba3389ebedbb46a1d8385a7f` | ✓ | ✓ |
| `tokenizer.model` | `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347` | ✓ | ✓ |

A pruned arm (`wanda` / `sparsegpt`) would have confounded stack drift with
export drift.

## 3. Provenance

| | path |
|---|---|
| **archive** | `/apdcephfs_wzc1/share_304376610/pighzliu_code/outputs/cast_eval_spec_union9/dense_ref/lm_eval_out/__apdcephfs_wzc1__share_304376610__pighzliu_code__models__Llama--Llama2-7b/results_2026-08-11T11-58-45.812255.json` |
| **rerun** | `/apdcephfs_wzc1/share_304376610/pighzliu_code/outputs/union9_harness_rebuild_control/dense_ref/lm_eval_out/__apdcephfs_wzc1__share_304376610__pighzliu_code__models__Llama--Llama2-7b/results_2026-08-14T14-13-20.326263.json` |
| **machine-readable verdict** | `/apdcephfs_wzc1/share_304376610/pighzliu_code/outputs/union9_harness_rebuild_control/dense_ref/harness_rebuild_control.json` |
| **driver** | `scripts/_union9_harness_rebuild_control.sh` |
| **comparator** | `baselines/cast_repro/tools/compare_union9_rerun_vs_archive.py` |
| **run log** | `logs/union9_harness_rebuild_control.log` |

Env: archive `tf=4.57.6 git=b86c479`; rerun `tf=4.57.6 git=4532077` (the
`git_hash` field is lm_eval's record of *this repo's* HEAD, not the harness — the
harness identity is carried by `transformers_version` plus the assertion below).

## 4. Result — per-task, archive vs rerun

```
task           n             archive_acc      rerun_acc     d_acc_pp     archive_accn     rerun_accn    d_accn_pp
boolq          3270         0.7776758410   0.7776758410    +0.000000                -              -            -
rte            277          0.6317689531   0.6317689531    +0.000000                -              -            -
hellaswag      10042        0.5713005377   0.5713005377    +0.000000     0.7597092213   0.7597092213    +0.000000
race           1045         0.3971291866   0.3971291866    +0.000000                -              -            -
piqa           1838         0.7785636561   0.7785636561    +0.000000     0.7889009793   0.7889009793    +0.000000
winogrande     1267         0.6929755328   0.6929755328    +0.000000                -              -            -
arc_easy       2376         0.7634680135   0.7634680135    +0.000000     0.7466329966   0.7466329966    +0.000000
arc_challenge  1172         0.4317406143   0.4317406143    +0.000000     0.4624573379   0.4624573379    +0.000000
openbookqa     500          0.3160000000   0.3160000000    +0.000000     0.4420000000   0.4420000000    +0.000000
```

**flips = 0 / 14 metrics.** All 9 tasks present at protocol `n`; RTE n=277.

Three independent levels were checked, because equal accuracy alone can hide a
permuted split or changed prompt (and the piqa cell here comes from a
*substituted* source, so "same number" would be the easiest way to fool
ourselves):

1. **per-task acc/acc_norm** — 0 flips (above).
2. **per-doc `doc_hash`/`prompt_hash`/`target_hash`** — 0 mismatches on all
   21,787 docs across all 9 tasks; no doc present in only one run.
3. **`task_hashes`** — lm_eval's own cumulative per-task digest
   (`loggers/evaluation_tracker.py:219`) identical for all 9, e.g. piqa
   `74d5a816572c396a…` in both.

Diagnostic level (not a gate): **raw logprob agreement 100.00 % bit-identical
on every task**, 73,840/73,840 continuations, `max|Δ| = 0.000e+00`. So this is
not "agrees after rounding" — the rebuilt stack is numerically identical to the
archive on this hardware.

---

## 5. Two real problems found and fixed en route

### 5.1 ⚠️ piqa could not load AT ALL — the archive's load path is extinct

**MAIN's pre-brief said "`TaskManager()` confirms 9/9 tasks resolve". That is not
what `TaskManager()` checks** — it indexes task *YAML*, and never touches a
dataset. When the datasets are actually loaded, **piqa fails on every node**:

```
RuntimeError: Dataset scripts are no longer supported, but found piqa.py
```

Upstream `piqa.yaml` uses `dataset_path: piqa`, a loading **script**, and
`datasets 5.0.1` hard-refuses scripts. Verified failing on LOCAL and on `.73`,
both online and with `HF_DATASETS_OFFLINE=1`. The other 8 tasks load fine.

The archive did not hit this because on 2026-08-11 its hub lookup *failed* and
`datasets` silently fell back to a pre-existing script-built cache — visible
verbatim in every archived `lm_eval.log`:

```
Using the latest cached version of the dataset since ybisk/piqa couldn't be found on the Hugging Face Hub
Found the latest cached dataset configuration 'plain_text' at
/root/.cache/huggingface/datasets/ybisk___piqa/plain_text/1.1.0/6c611c1a9bf220943c4174e117d3b660859665baf1d43156230116185312d011
```

That cache is gone everywhere: LOCAL and `.212` have **no
`/root/.cache/huggingface/datasets/` at all** (wiped by the restart); `.73` has
only a *different*, parquet-built `default/0.0.0/142c5123…`; `.82`/`.104` have no
piqa. So the archive's exact load path is **not reconstructible**.

**Recovery, proven rather than assumed.** The archived
`samples_piqa_*.jsonl` retains all 1838 validation docs verbatim, which makes
the substitution *checkable*. A hub parquet snapshot found on the project disk
was compared field-by-field against them:

- `baselines/cast_repro/tools/check_piqa_source_matches_archive.py` →
  n=1838, **field diffs 0** (`goal`/`sol1`/`sol2`/`label`, every index, in order).
- `baselines/cast_repro/tools/verify_piqa_override_hashes.py` drives it through
  lm_eval's own pipeline and recomputes lm_eval's hashes →
  **doc_hash 0, prompt_hash 0, target_hash 0 mismatches / 1838**.

The parquet was copied out of the evictable HF blob cache to a stable path
(`data/union9_piqa_parquet/`, md5 `82cbf1cde7abde1d10a666023013861a` train /
`bb470a523f55e14d25ef4a7529dabcc1` validation) and mirrored to zwfy6 with
matching md5, so a future cache wipe cannot repeat this. The override YAML is
`baselines/cast_repro/union9_taskoverride/piqa.yaml`; its only semantic delta
from upstream is `dataset_path` + `dataset_kwargs.data_files`. That the
substitution is faithful is confirmed independently by the run itself:
`task_hashes["piqa"]` came out identical to the archive's.

Two documented dead ends (both tried, both failed) are recorded in the driver's
header so nobody repeats them: `--include_path` loses to the built-in registry
(`lm_eval/tasks/__init__.py:83` merges `{**tasks, **task_index}`), and a
`--tasks <path>` entry raises `ValueError: Tasks not found` because
`__main__.py:353` tests the raw string against a list that now holds dicts.

### 5.2 ⚠️ Watcher bug: `TRAIN_NODE` was hard-assigned, and the topology inverted

`_run_sparseforge_tokenmatched_union9_watcher.sh:161,168` hard-assigned
`TRAIN_NODE=local` / `=remote` (no `${VAR:-}`), so a caller override was
**silently ignored**. Since the restart the mapping is exactly backwards:
`noslorb` now trains on the scoring box (LOCAL) while `slorb` trains on `.212`.
Left as-is, `ARM=slorb` would have pgrep'd locally for a trainer living on
another host, found nothing, declared training finished, and **scored a
mid-training checkpoint ~7 h early**. Both are now `${TRAIN_NODE:-…}` /
`${TRAIN_LOG:-…}` with the historical values kept as defaults.

---

## 6. Note on hardware: the control had to run on B200

All **37** `results_*.json` in this repo were scored on `NVIDIA L20A` (= B200
sm_100, driver 580.105.08, torch 2.13.0, Python 3.14.6). **There is no
H20-scored union-9 precedent.** `.73`/`.82` are sm_90 on driver 535.247.01, so
scoring the control there would have confounded "did the software rebuild change
the numbers" with "does sm_90 vs sm_100 change the numbers" — and a FAIL would
have been uninterpretable. The driver therefore asserts `sm_100` and exits 22
otherwise. GPU pressure was handled by waiting for stable headroom on 4 cards
(sampled 3×, per `[[one-sample-is-not-a-trend-or-state]]`), never by moving to
the wrong hardware.

Consequence: a pinned venv *was* built on `.73`
(`/apdcephfs_zwfy6/…/venv_union9`, `lm_eval 0.4.8` + `transformers 4.57.6`) and
is left in place, but it was **not** used for the control and must not be used
for a union-9 row without its own sm_90-vs-sm_100 control.

## 7. One deviation from the archive's invocation, deliberate

`--batch_size 64` (pinned) instead of `--batch_size auto`. The archive ran `auto`
on an idle box and auto *resolved* to 64 (every archived
`results_*.json`: `batch_size="auto"`, `batch_sizes=[64]`). Here a trainer
already holds ~111-120 GB/card, and `auto` re-probes by allocating until OOM
(`huggingface.py:745`, `find_executable_batch_size(starting_batch_size=64)`),
which could either resolve *lower* than 64 — silently making the control a
different invocation — or OOM the neighbouring trainer. Pinning 64 reproduces the
archive's *effective* batch while removing the probe. The 100 % bit-identical
logprobs confirm this was the right call: the effective computation was identical.
