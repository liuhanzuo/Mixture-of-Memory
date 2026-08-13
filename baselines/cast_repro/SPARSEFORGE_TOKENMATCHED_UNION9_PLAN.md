# Task #246 closeout — scoring the token-matched ±SLoRB arms on the union-9 harness

**Status:** watchers armed 2026-08-13, both on node `.21`, waiting for training to end.
**Driver:** `scripts/_run_sparseforge_tokenmatched_union9_watcher.sh`
**Outputs:** `outputs/cast_eval_spec/sparseforge_tokenmatched_{noslorb,slorb}/`

---

## 1. The gap this closes

`scripts/_run_sparseforge_tokenmatched.sh:294` configures the in-run eval as, verbatim:

```
  --lm_eval_tasks "hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,piqa,race" \
```

That is **seven** tasks. The union-9 table's protocol is **nine**
(`scripts/_sparseforge_same_harness_21.sh:53`, byte-identical in
`scripts/_missing_baselines_union9_21.sh:58`):

```
TASKS=boolq,rte,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa
```

`boolq` and `rte` are missing. A 7-task mean cannot be placed beside the existing arms'
9-task mean, and `aggregate_zeroshot_union9.py:103` refuses to try
(`missing task results: [...] -- this arm's row is INVALID`).

Confirmed present in both live runs' own `args.json`:
`lm_eval_tasks = hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,piqa,race`.

## 2. The gap is bigger than "two missing tasks" — there will be **zero** in-run numbers

`--finalize_lm_eval True` is set on both arms, but it is a **dead flag here**. Traced in
`main_llama.py`:

| line | code | consequence |
|---|---|---|
| 2102 | `if iter_num % args.eval_interval == 0:` | the eval block |
| 2248 | `if finalization_done and args.finalize_lm_eval:` | lm_eval gate — needs `finalization_done` |
| 3052 | `if iter_num > max_iters:` | termination |
| 3053 | `if not finalization_done:` | → finalize |
| 3215 | `finalization_done = True` | the **only** place it is set |
| 3436 | `extra = int(args.final_finetune_iters)` | `= 0` for both arms (`FINAL_FT=0`) |
| 3437 | `if extra > 0: ... 3466 continue` | not taken |
| 3467 | `else: ... 3470 break` | **leaves `while True:`** |

`finalization_done` becomes `True` only *after* `iter_num > max_iters`, and with
`final_finetune_iters = 0` the loop immediately `break`s. Control never returns to L2102,
so the L2248 gate is never evaluated. **No lm_eval runs at all — not even the seven.**

`FINAL_FT=0` is deliberate and correct: the published run's 3000-iter final stage is exactly
where the `qa_format_sft_llama` MC-QA contamination entered. It should not be "fixed".

Empirical corroboration in `out_llama/`:

| run | `final_finetune_iters` | `best_lm_eval.json`? |
|---|---|---|
| `..._20260413_201320` (published headline) | 3000 | **yes** (iter 17900, mean 57.2672) |
| `..._20260404_110624` | 2000 | yes |
| `..._20260304_140825`, `..._20260306_211245` | 0 | **no** |

**So this script is not "topping up boolq+rte". It is the only source of zero-shot numbers
for these two arms.** Everything — 9 tasks, both PPL columns, both 2:4 verifies — is measured
offline here.

## 3. Export variant: why `fold` for `slorb` and why `drop` is safe for `noslorb`

`--slorb drop` discards the trained low-rank branch. On a model trained *with* it, that is a
post-hoc **amputation**; its cost (≈4.9 pp AST-7) measures amputation damage, not the method.
This is Defect 1 of the CORRECTION block at the top of `SPARSEFORGE_SAME_HARNESS.md`, which
retracted a headline conclusion built on exactly that confusion. The entire point of these two
arms is that ±SLoRB is decided **at training time**.

| arm | ckpt contains | export flags | rationale |
|---|---|---|---|
| `slorb` | `SLoRB_Weight`, `x_proj` **present** | `--mask hard --slorb fold` | fold is exact linear algebra; keeps the branch the model learned. `drop` here would re-create the amputation confound. |
| `noslorb` | `SLoRB_Weight`, `x_proj` **absent** | `--mask hard --slorb drop` | not an amputation — there is no branch to remove, so `drop` is a strict no-op. |

Both verified 2026-08-13 by probing the live checkpoints' pickles: `noslorb` has
`SLoRB_Weight=False x_proj=False`; `slorb` has both `True`. Their `args.json` agree
(`SLoRB=False` / `SLoRB=True`).

`--slorb fold` on the `noslorb` ckpt is not merely wrong, it is **impossible**: the exporter
hard-exits at `export_sparseforge_to_hf.py:181`
(`--slorb fold requested but ...SLoRB_Weight/...x_proj missing`). The script derives the flag
from `ARM` and additionally asserts the ckpt's tensor inventory matches, so a mismatched
checkpoint is refused rather than scored.

## 4. Both arms are scored on `.21`, not on the node that trained them

Measured 2026-08-13:

| node | lm_eval | transformers |
|---|---|---|
| `.21` `/opt/conda/envs/torch-base` | **0.4.8** | **4.57.6** |
| LOCAL `/opt/conda/envs/torch-base` | **absent** | 5.13.1 |
| LOCAL `Mixture-of-Memory/.venv` | absent | (no torch) |

Every existing union-9 arm records `"transformers_version": "4.57.6"` and
`"git_hash": "b86c479"` in its `results_*.json` (checked for `dense_ref`, `cast_7500`,
`ast_official`, `wanda`). Scoring the `noslorb` arm on LOCAL would silently introduce a
different stack — the same class of error as the −0.346 pp cross-harness AST-7 offset that
already invalidated one comparison. The watcher's preflight therefore **hard-refuses** on a
stack mismatch (exit 21) instead of producing a non-comparable row.

`noslorb` trains on LOCAL but is scored on `.21`. Both are on the wzc1 disk, so no copy is
needed — verified `.21` can `stat` LOCAL's `out_llama_tokenmatched_noslorb/` and read its log
(identical md5 for the script from both nodes).

Because the trainer for `noslorb` is invisible to `pgrep` on `.21`, its liveness is judged by
the **shared training log's mtime** (a tqdm line lands every 45-55 s; idle ≥ `LOG_STALE_S`
= 1800 s means it stopped). The `slorb` arm trains on `.21` itself, so `pgrep` is authoritative.
Neither test uses instantaneous `nvidia-smi` utilisation.

## 5. Harness invocation — byte-identical to the existing arms

```
lm_eval --model hf \
  --model_args "pretrained=<export>,dtype=bfloat16,parallelize=True,trust_remote_code=True,add_bos_token=False" \
  --tasks boolq,rte,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa \
  --batch_size auto --num_fewshot 0 --output_path <out> --seed 0 --trust_remote_code --log_samples
```

`--batch_size auto` is deliberate, **not** `--batch_size 64`. The four arms already in the
table record `batch_size: "auto"` with resolved `batch_sizes: [64]` in their `results_*.json`
— `auto` is the invocation, `64` is what it resolved to. Hard-coding `64` would be a
different invocation string. Each arm gets 4 GPUs with `parallelize=True`, matching the
existing rows (their `lm_eval.log` shows `max memory per GPU` over devices 0-3).

GPU split: `noslorb` → 0,1,2,3 and `slorb` → 4,5,6,7, mirroring
`_sparseforge_same_harness_21.sh`'s `run_one()` topology, so both watchers can run
concurrently on `.21` without contending.

`boolq`/`rte` resolve through hub parquet redirects, so the `hy-proxy.woa.com:3128` exports
are **required**, not decorative. Verified reachable (HTTP 200 on `aps/super_glue`) and the
datasets are already in `.21`'s HF cache.

## 6. PPL at both seqlens, each labelled

`SPEC.md:213`'s "the whole PPL column is 2048" assumption was falsified — the SparseForge
headline 6.2179 is a seqlen-4096 number that sat in a 2048 column (commit 501dafb). Both are
measured, into separate directories (`ppl2048/`, `ppl4096/`), and the summary **asserts** each
`ppl_metrics.json`'s own `seqlen` field matches the directory it sits in. Same layout as the
`sparseforge_dolmino_link2` precedent.

## 7. 2:4 verification — two distinct FAIL modes

`verify_2of4_hf_export.py` gates on `tiles_gt2 == 0 && zero_frac >= 0.5-1e-4 &&
len(scope) == 224` (its lines 118-130), run PRE- and POST-inference, raw output kept
(`verify_2of4_<variant>_{pre,post}.log`).

| arm | expected | on FAIL |
|---|---|---|
| `noslorb` / `hard_drop` | **PASS** — `drop` removed nothing, must be exact 2:4 | **REAL failure** → abort, score nothing |
| `slorb` / `hard_fold` | **FAIL by design** — the fold writes into pruned positions, `zero_frac → ~0`, `exact_2of4_frac = 0` | expected; scored and reported, but `2of4_eligible=false` and **barred from any 2:4 column** |

If `hard_fold` unexpectedly *passes*, that is flagged `suspicious` (it would imply the branch
is a no-op).

## 8. RTE integral count

RTE n=277. Existing arms: dense 175, `hard_fold`/`soft_fold` 139, AST 184, CAST-repro 207,
`hard_drop` 158, Wanda 148. The published 69.82 was caught as a transcription error precisely
by reconstructing k = acc × 277 = 139. The summary emits k for these arms and **asserts**
`|k − round(k)| < 0.01` and `n == 277`, so a metric/split mismatch cannot pass silently.

## 9. ⚠️ What this table can and cannot answer — declared in advance

**CAN answer:** whether **training-time** SLoRB helps, at matched corpus, matched token budget
(7,864,320,000 = 7500 × 256 × 4096), matched seq-len, matched LR schedule, and matched eval
harness. `slorb` vs `noslorb` is a clean ±SLoRB contrast, and each vs CAST-repro @7500 isolates
mask machinery ± the branch.

**CANNOT answer:**

1. **Whether SparseForge's mask search is good.** Both arms train on
   `dolmino-mix-1124-llama2`, *not* SparseForge's own `qa_format_sft_llama`. This deliberately
   removes the MC-QA contamination (129,752,281 tokens from 8 benchmark *train* splits,
   ≈144.7 epochs, with RACE both a training source and a CAST-7 eval task —
   `AST_VS_SPARSEFORGE_DATA_CONFOUND.md`), but it also means these arms are **not** a
   reproduction of the published SparseForge result. They cannot confirm or refute it.
2. **Anything LR-matched to SparseForge's published recipe.** These arms use CAST-repro's
   absolute LR schedule (2e-5 / 2e-6 / warmup 375 / decay 7125) so that "same config as CAST"
   is literally true. SparseForge's own 1e-4 **diverged** at iter ~860 (ppl 2230). Any claim
   about SparseForge "at its own LR, on this corpus" needs a third pair of runs.
3. **The published "5B" arm's numbers.** Those are a 3-link resume chain totalling ≈18.77 B
   tokens; these are single 7.86 B-token runs from the base model.
4. **A 2:4 verdict on the `slorb` arm.** Its export is dense by construction (SLoRB folded).
   Only `noslorb`/`hard_drop` is 2:4-eligible.

## 10. Verified before arming (2026-08-13)

- `bash -n` clean, on LOCAL and on `.21` (same md5 `b736824c…`, shared wzc1 disk).
- Bad/absent `ARM` rejected with rc=2.
- Preflight on `.21`: tooling OK, harness assertion `lm_eval 0.4.8` + `transformers 4.57.6`
  → **MATCH**; proxy → HTTP 200.
- Preflight on LOCAL: correctly **refuses** (no lm_eval; transformers 5.13.1).
- Liveness, `slorb`: detects its 9 live trainer PIDs on `.21`.
- Liveness, `noslorb`: reads LOCAL's log via the shared disk, "advanced 44s ago → still running".
- All assets exist: exporter, verifier, aggregator, PPL harness, `wiki.test.raw`,
  `models/Llama--Llama2-7b/{model.safetensors.index.json,tokenizer.model}`.
- Completeness guard exercised against synthetic fixtures (then deleted): valid ckpt → proceeds;
  wrong-arm ckpt (SLoRB present under `ARM=noslorb`) → refused; **truncated** ckpt → refused;
  GPUs busy → refused, never competes; `DIVERGED_*` dir → skipped in favour of the good sibling.
- Summary block reproduces `158/277` for RTE, and its assertions fire on a wrong `n`, on a
  seqlen mislabel, and emit the 2:4-barred notice when `eligible=false`.

Bugs found and fixed during that dry-run (all were in this new script):

1. `readlink -f` on `<out_dir>/last` returned **empty** — the trainer stores that symlink
   relative to `$ROOT`, not to its own directory. Resolved against `$ROOT` with a
   newest-non-`DIVERGED_` fallback.
2. The eval-exclusion filter `*lm_eval*` matched the **trainer's own** `--lm_eval_tasks`
   / `--lm_eval_batch_size` flags and discarded all 9 genuine trainer PIDs, making the watcher
   believe training had finished. Now keys on invoked program names.
3. `if ! "$PY" … | tee` tested `tee`'s status, so a **failing probe was ignored** — a
   deliberately wrong-arm checkpoint was declared COMPLETE. Now uses `PIPESTATUS[0]`.
4. The PPL stage discarded its return code; a failed PPL cell would have produced a row with a
   missing column. Now fatal.

**Only knowable after training ends:** the actual export success, 2:4 verdicts, PPL values,
the 9 task accuracies, and RTE's integral k.

## 11. Operating notes

```bash
# both on .21, from the project root
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
setsid nohup env ARM=noslorb bash scripts/_run_sparseforge_tokenmatched_union9_watcher.sh \
  > logs/sparseforge_tm_union9_noslorb_watch.out 2>&1 &
setsid nohup env ARM=slorb   bash scripts/_run_sparseforge_tokenmatched_union9_watcher.sh \
  > logs/sparseforge_tm_union9_slorb_watch.out 2>&1 &
```

Progress: `logs/sparseforge_tm_union9_<arm>_progress.log`.
Stop a watcher with `kill -9 <PID>` on its own PID — **never `pkill -f`**
(memory: `kill-remote-gpu-job-by-pid-not-pkill`). The watcher never kills the trainer; it only
waits, polling every 300 s with a 48 h budget, and exits 9 rather than scoring a checkpoint it
could not verify.
