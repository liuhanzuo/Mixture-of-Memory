# Paper B P2.4 / Task #189 — wzc1/L20A ladder pre-SFT eval (keep8, keep12) on .252

**Owner**: subagent dispatched by MAIN 2026-08-08 03:58 CST.
**Purpose**: Complete the 5-rung cross-architecture (L20A cc10.0 vs H20 cc9.0)
audit for Paper B Table 4. Sibling `_v2` H20 batteries for the four ladder
rungs (keep8/keep10/keep12/shortgpt16) were produced overnight on zwfy6 by
agent `a50df6cd` via `_run_olmo2_p24_eval_ladder_prev2_73.sh`. This dispatch
gives us the wzc1/L20A twin for the two rungs whose ckpts exist on wzc1
(keep8, keep12), so `core6_wzc1(X) − core6_zwfy6_v2(X)` can be computed
per-rung and tested against the "flip count scales with pruning damage"
hypothesis (n=2 so far: full base 10 flips; keep14 28 flips).

## Load-bearing signal: keep12 step mismatch

wzc1 has **only one `step*.pt`** in `outputs/olmo2_probe2_7B_keep12fresh2/`:

    -rw-r--r-- 1 root root 41G Jul 24 11:53 outputs/olmo2_probe2_7B_keep12fresh2/step111500.pt

**That is NOT the Table 4 headline step**. Table 4's keep12 row was measured
from `step124000.pt` (as per `paperB/P0_7_AGGREGATE_AUDIT.md` §2 and the
zwfy6 `_v2` sibling), which does **not** exist on wzc1. Options considered
and rejected:

  * `scp -O` step124000 from zwfy6 to wzc1: 48.7 GB @ ~12 MB/s = ~70 min
    just to line up steps — MAIN's dispatch explicitly told us not to block on
    cross-disk shuffling for this task.
  * Skip keep12 entirely: defeats the point (keep12 is one of the two rungs
    MAIN explicitly asked for).

**Decision**: run keep12 on wzc1 with `step111500.pt` and label its
output `7B_keep12_step111500_wzc1` (the step number is embedded so
provenance is explicit). The direct wzc1↔zwfy6 same-ckpt delta only holds
for **keep8** (`step121000` on both disks); the **keep12** delta will
compare **wzc1 step111500 vs zwfy6 step124000**, i.e. a cross-arch AND
cross-step delta. This must be called out when interpreting the numbers —
a non-zero core6 delta at keep12 could be arch, step, or both.

The right forward action (MAIN's call, out of scope for this dispatch) is
either (a) scp step124000 to wzc1 and rerun the two `7B_keep12_step124000_wzc1`
harnesses for a clean arch-only delta, or (b) find keep12 step111500 on
zwfy6 (if it exists there too) and rerun the sibling `_v2` at 111500 for
the same reason.

## Legs

Two batteries, serial:

| leg | ckpt | output_name |
|---|---|---|
| keep8_wzc1  | `outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt` (32 G on wzc1)  | `7B_keep8_step121000_wzc1`  |
| keep12_wzc1 | `outputs/olmo2_probe2_7B_keep12fresh2/step111500.pt` (41 G on wzc1) | `7B_keep12_step111500_wzc1` |

Note the size difference (32 G vs 41 G): the keep8 ckpt is weights-only, the
keep12 ckpt carries optimizer state. `load_pruned_model` reads only
`model_state` so this does not affect eval semantics.

Each battery = 5 harnesses:
  1. held-out Dolmino PPL (`eval_olmo2_probe2_ppl.py`)
  2. core6 downstream — hellaswag, arc_challenge, arc_easy, piqa,
     winogrande, openbookqa (`eval_olmo2_probe2_downstream.py --tasks CORE`)
  3. know5 downstream — mmlu, lambada_openai, boolq, commonsense_qa,
     social_iqa (same script, `--tasks KNOW`)
  4. MMLU dual protocol — letter + content
     (`eval_olmo2_mmlu_content.py`)
  5. closed-book — PopQA, TriviaQA (`eval_olmo2_closedbook_qa.py`)

## Node / launch

  * **Node**: `.252` (`28.89.19.252`), 8×L20A cc10.0 wzc1
  * **Verified idle** at 03:51 CST (0/8 procs, 0 MiB used); prior
    `p24_eval_full32_shortgpt_252` had finished (log ended with the "DONE"
    output-listing block)
  * **Python**: `/opt/conda/envs/torch-base/bin/python` torch 2.13.0 py 3.14
  * **Driver**: `scripts/_run_olmo2_p24_eval_ladder_wzc1_252.sh`
    (committed as `73d2e5c`, byte-identical to
    `_run_olmo2_p24_eval_ladder_prev2_73.sh` sibling except: 2 rungs, ROOT
    is wzc1, output-name suffix is `_wzc1` not `_v2`)
  * **Launched**: 2026-08-08 **03:58:36** CST
    (`setsid nohup bash scripts/_run_olmo2_p24_eval_ladder_wzc1_252.sh > logs/p24_eval_ladder_wzc1_252.log 2>&1 &`)
  * **Parent bash PID**: `3115943`
  * **Log**: `logs/p24_eval_ladder_wzc1_252.log`
  * **Post-launch verification** at 03:59 CST: 8 python shard processes
    running (`3115954..3115961`) — PPL of keep8 in-flight
  * **ETA**: ~2 h → **~2026-08-08 06:00 CST** (2 rungs × 5 harnesses = 10
    harness invocations; each ~10-15 min on L20A at 8-shard parallelism)

## Invariants preserved

  * **`chat_template=False`, `--add_bos 0`** everywhere (memory
    `paper-eval-chat-false-mandatory`; OLMo-2 base is not SFT'd/RL'd).
  * **8/8 shard invariant**: driver runs `assert_8shards` before every
    merge; missing any shard aborts the merge to prevent silent
    contamination from partial merges (memory
    `kill-remote-gpu-job-by-pid-not-pkill`).
  * **`--save_per_example`** on all downstream evals; MMLU-content and
    closed-book harnesses write per-item files by default. Per-item
    predictions are required for the cross-arch McNemar / paired-bootstrap
    downstream analysis.
  * **`_wzc1` output-name suffix** — verified no collisions before launch
    against `olmo2_ppl_results/`, `olmo2_downstream_results/`,
    `olmo2_mmlu_content_results/`, `olmo2_closedbook_results/` for either
    `7B_keep8_step121000_wzc1` or `7B_keep12_step111500_wzc1` (both bare
    and `_know` variants).

## Cross-architecture delta tripwire (post-eval)

Once both batteries finish, compute per rung:

    Δcore6(X) = core6_wzc1(X) − core6_zwfy6_v2(X)

Expected pattern per MAIN's n=2 findings:
  * **full base intact**: Δ ≈ +0.034 pp (10 net flips), L20A slightly higher
  * **keep14 pruned**:    Δ ≈ +0.156 pp (28 net flips), L20A slightly higher

For keep8 (same-ckpt, same-step): if Δ falls between +0.03 and +0.30 pp
with L20A ≥ H20 on core6, that extends the flip-count-scales-with-damage
hypothesis to n=3. **Negative Δ** or **|net flips| > 50** across all 6
tasks is a signal not noise — report loudly.

For keep12 (**different-step** — see load-bearing signal above): the delta
is confounded by 12500 training steps. If it looks anomalously large or
oppositely-signed, that is either (a) the arch flip magnifying at
higher damage, (b) real step-125000-vs-step-111500 skill drift, or (c)
both. Cannot disambiguate without also running one of the two ckpts on the
other disk.

## Files created

  * `scripts/_run_olmo2_p24_eval_ladder_wzc1_252.sh` (this driver)
  * `status/PAPERB_P24_LADDER_WZC1_EVAL.md` (this report)
  * `logs/p24_eval_ladder_wzc1_252.log` (live)
  * eval outputs (on wzc1, when done):
      - `olmo2_ppl_results/7B_keep{8,12}_step*_wzc1/summary.json`
      - `olmo2_downstream_results/7B_keep{8,12}_step*_wzc1{,_know}/summary.json`
        + `per_example_*.jsonl` (per-item preds)
      - `olmo2_mmlu_content_results/7B_keep{8,12}_step*_wzc1/summary.json`
      - `olmo2_closedbook_results/7B_keep{8,12}_step*_wzc1/summary.json`

## Cross-arch delta table (fill in when batteries complete)

| rung | ckpt (wzc1) | ckpt (zwfy6 `_v2`) | same-step? | core6 wzc1 | core6 zwfy6 v2 | Δ core6 (pp) | net flips | status |
|---|---|---|---|---:|---:|---:|---:|---|
| full-base | (n/a — vanilla base) | (n/a) | — | .59532 (10 flips) | see PAPERB_CORE6_CROSSARCH_FLOOR | +0.034 | 10 | (MAIN n=2 anchor) |
| keep14  | `.../keep14fresh2/step200000.pt` | same, on zwfy6 | ✅ | .59376 | .59532 | +0.156 | 28 | (MAIN n=2 anchor) |
| keep8   | `.../keep8fresh2/step121000.pt`  | same, on zwfy6 | ✅ | pending | pending | pending | pending | eval running |
| keep10  | (not on wzc1; max step 83500 zwfy6-only) | `step83500` | n/a | skipped | (baseline exists) | n/a | n/a | skipped (see §keep10) |
| keep12  | `.../keep12fresh2/step111500.pt` | `step124000` (zwfy6) | **NO — Δstep=12500** | pending | pending | pending (confounded) | pending | eval running |

## Ledger entry (gpu_runs.jsonl)

```json
{"ts":"2026-08-07T20:01:04Z","node":".252","exp":"paperB_p24_ladder_wzc1_252","commit_hash":"73d2e5c","legs":["7B_keep8_step121000_wzc1","7B_keep12_step111500_wzc1"],"parent_pid":3115943,"log":"logs/p24_eval_ladder_wzc1_252.log","status":"running","note":"keep12 wzc1 uses step111500 (Table 4 headline is step124000 - only on zwfy6); keep8 uses step121000 (matches Table 4 headline). 2 rungs x 5 harnesses = 10 evals, ~2h ETA."}
```
