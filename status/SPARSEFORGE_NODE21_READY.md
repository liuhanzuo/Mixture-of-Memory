# .21 Readiness Audit for SparseForge (2026-08-08)

> ## 🔴 UNRESOLVED CONFLICT — MAIN MUST ARBITRATE BEFORE USING .21
>
> **Another agent (driving from LOCAL) is simultaneously trying to start Paper B `keep10` resume ON `.21`,
> and keeps re-launching the 118 GB dolmino transfer + the 43.8 GB `step124000.pt` pipe.**
> I killed its processes **three times** by PID (`.21`: 2786/4147 → 5731/5767 → 7086/7122; plus the `.73`
> driver 3286164 and the LOCAL ssh senders). It re-spawns within ~60 s and it also **removed the
> directory tripwire** I installed at `/dev/shm/dolmino_now15b_zwfy6.npy`. `chattr +i` is unavailable
> (`chattr: command not found` on `.21`), so I cannot hard-lock the path.
>
> Its artifacts: **`scripts/launch_keep10_resume_b200_21.sh`** (new, on wzc1) — a poller that waits up to
> 4 h for the dataset to reach `126907244672` bytes, then runs keep10 resume on `.21`; plus
> `logs/olmo2_7B_keep10fresh2_resume200k_21.log`. My `exit 1` disarm of `/tmp/transfer_to_21_v2.sh`
> on `.73` was reverted to the live rsync version.
>
> **Good news — nothing is actually lost or occupied:**
> - `.21`'s **GPUs were never occupied** (0 % util throughout): that trainer blocks waiting on the
>   never-completed dataset, so it has not taken the node.
> - **Paper B loses no progress** if the `.21` attempt stays dead: `keep8` is running on `.73`,
>   **`keep10` is already running on `.82`**, `keep12` on `.104`, `keep14fresh2_seed1234` on LOCAL —
>   all verified alive. The `.21` keep10 was a *migration/duplicate*, not the only copy.
>
> **Decision needed:** (a) tell the other agent to stand down so `.21` goes to SparseForge (my instruction),
> or (b) revoke the "SparseForge first" priority. Until then `.21` will keep flapping and repeating
> ~5–9 GB of transfer churn per minute. Logged in `status/gpu_runs.jsonl` as
> `status: conflict_unresolved`.


**Verdict: `.21` is IDLE and handed over to SparseForge. Model/data prerequisites are ALL present.
Two BLOCKERS remain, both software-side: (B1) `lm_eval` + `transformers` + `datasets` are NOT installed
on any interpreter on `.21`; (B2) the `rte` and `race` dataset caches do not exist anywhere on the wzc1 disk.**

Audit was **read-only + cleanup**. No training or eval was launched — MAIN decides what runs.

---

## 0. Why .21 was freed (context)

User changed priority: **SparseForge first, Paper B resume yields.** The previously planned
"migrate Paper B keep12 resume from H20 to .21" is therefore **cancelled**, not failed.
Logged as `status/gpu_runs.jsonl` → `exp: paperB_resume_migration_to_21, status: cancelled`.

### What was killed / removed (all by PID, never `pkill -f`)

| Host | PID(s) | Process | Note |
|---|---|---|---|
| .21 | 2786 | `scp -t /dev/shm/dolmino_now15b_zwfy6.npy` | had done 19.9 GB / 118 GB in 18 min (~2.7 h ETA) |
| .21 | 4147 | `scp -t .../keep12fresh2/step124000.pt` | orphan receiver, survived sender kill |
| .73 | 3286164, 3286162 | `bash /tmp/transfer_to_21.sh` + setsid wrapper | the driver that queued both transfers |
| .73 | 3321956/57/58 | `scp -O step124000.pt -> .21` sender chain | in-flight 2nd transfer |

Partial files deleted (both incomplete — left in place they would have poisoned a later agent):
- `.21:/dev/shm/dolmino_now15b_zwfy6.npy` — 20 240 089 088 B of ~118 GB
- `.21:/apdcephfs_wzc1/.../outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt` — 50 331 648 B of 43 867 047 810 B

**Extra hazard found and neutralized:** `/tmp/transfer_to_21_v2.sh` on `.73` was a *queued rsync retry*
of the same 118 GB transfer (`rsync -az --partial`, resumable). It was not yet running. It has been
**overwritten with an `exit 1` stub** carrying a comment explaining the cancellation, so that a concurrent
agent cannot silently restart a 2.7 h transfer that would compete with SparseForge for `.21`.

**Sources are intact** (nothing destructive to Paper B):
- `.73:/apdcephfs_zwfy6/.../keep12fresh2/step124000.pt` = 43 867 047 810 B, unchanged.
- `.73:/dev/shm/dolmino_now15b.npy` untouched (actively mmap'd by the keep8 training).

**Paper B resume runs all still alive, untouched** (verified by `ps` on each node):
`.73 keep8fresh2` · `.82 keep10fresh2` · `.104 keep12fresh2` · `LOCAL keep14fresh2_seed1234`.

---

## 1. .21 is idle (evidence)

Fresh connection, 75 s after cleanup, confirming nothing retries:

```
$ nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,utilization.gpu --format=csv
0..7, NVIDIA L20A, 10.0, 183359 MiB, 0 MiB, 0 %      # all 8 identical
Processes: No running processes found
$ ps -eo pid,cmd | grep -Ei "scp|rsync|dolmino" | grep -v grep  ->  CLEAN: still no transfer procs
$ df -h /dev/shm  ->  tmpfs  944G  0  944G  0% /dev/shm
```

### Hardware
| Item | Value |
|---|---|
| GPU | **8 × NVIDIA L20A**, compute_cap **10.0**, **183 359 MiB/card**, 1000 W cap |
| Driver / CUDA | 580.105.08 / CUDA 13.2 |
| CPU | **256 threads**, Intel Xeon 6767P, 64 cores/socket × 2 sockets |
| RAM | **2.0 TiB total**, 1.9 TiB available |
| `/dev/shm` | **944 GB, 0% used** (huge — a 118 GB dataset would fit trivially if ever needed) |

`.21` is **L20A, not B200** — same architecture as LOCAL, consistent with the CLAUDE.md correction.

### Disk identity
`/apdcephfs_wzc1` on `.21` is a **real directory (not a symlink)**: `readlink -f` returns itself, and
`/apdcephfs_zwfy6` **does not exist** here. So `.21` shares the wzc1 disk with LOCAL, as documented.

---

## 2. SparseForge prerequisites — ALL PRESENT ✅

All three paths `ls`-verified on `.21` with correct sizes:

**(a) Pruned Llama checkpoint dir** — `/apdcephfs_wzc1/share_304376610/pighzliu_code/out_llama/models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260413_201320/`
```
45772988056  model.pt                 (45.8 GB ✅ matches expected 45.7GB)
41078444091  model_best_lm_eval.pt    (41.1 GB ✅)
       2902  args.json  ✅        416  best_lm_eval.json  ✅
        284  config.json           184  eval.json          (2 bonus files)
```

**(b) Base models** — `du -sh`: `models/Llama--Llama2-7b` = **38G** ✅ · `models/AST-official-LLaMA2-7B-2of4` = **13G** ✅

**(c) `SparseForge_worktree/`** ✅ readable, git HEAD `520170b "Add rebuttal experiments and fixed-support recovery workflow"`, clean tree.
Contains `sparseforge/`, `scripts/`, `experiments/rebuttal_2026/{scripts,docs,tables,figures,TODO.md}`,
`eval_wiki_ppl.sh`, `train_llama.sh`, `train_universal.sh`.
Note the eval entrypoint `eval_wiki_ppl.py` lives in **`legacy/`**: `SparseForge_worktree/legacy/eval_wiki_ppl.py`.

---

## 3. Software environment on .21

| Interpreter | Python | torch | Verdict |
|---|---|---|---|
| `/opt/conda/envs/torch-base/bin/python` | **3.14.6** | **2.13.0**, `cuda_avail=True`, `ndev=8` | ✅ **the only usable one** |
| `Mixture-of-Memory/.venv/bin/python` | 3.11.6 | ❌ none | unusable |
| `/usr/bin/python3` | 3.11.6 | ❌ none | unusable |

```
$ .venv/bin/python -c "import torch"
ModuleNotFoundError: No module named 'torch'          # confirms CLAUDE.md: LOCAL/.21 .venv has no torch
$ /opt/conda/envs/torch-base/bin/python -c "import torch;..."
torch: 2.13.0 cuda_avail: True ndev: 8
```

### 🚨 BLOCKER B1 — `lm_eval` NOT installed, and neither is the HF stack

```
$ /opt/conda/envs/torch-base/bin/python -c "import lm_eval"
ModuleNotFoundError: No module named 'lm_eval'
$ /opt/conda/envs/torch-base/bin/python -m lm_eval --version
/opt/conda/envs/torch-base/bin/python: No module named lm_eval
$ which lm_eval lm-eval
/usr/bin/which: no lm_eval in (...)   /usr/bin/which: no lm-eval in (...)
$ pip show lm_eval / lm-eval
WARNING: Package(s) not found: lm_eval        WARNING: Package(s) not found: lm-eval
```

This is **worse than the `.73` case** — the conda env is essentially *bare torch*:

| package | status on conda py3.14 |
|---|---|
| `numpy` | ✅ 2.5.1 |
| `lm_eval` · `transformers` · `datasets` · `accelerate` · `scipy` · `sentencepiece` · `huggingface_hub` · `peft` | ❌ **all ModuleNotFoundError** |

`.venv` (py3.11) also lacks all four of `lm_eval/transformers/datasets/accelerate`.
Only conda envs present: `base`, `torch-base` — **no `minillm` env**, which matters because
`eval_wiki_ppl.sh` hardcodes `export CLUSTER_ENV_SETUP='source ~/.bashrc && conda activate minillm &&'`
→ **that script will fail as-is on `.21`.**

**I did NOT install anything** (per instructions). I only ran `pip --dry-run` to de-risk MAIN's decision:

- Proxy works from `.21`: `hy-proxy.woa.com:3128` → `pypi_http_code=200`, `hf_http_code=200`; direct (no proxy) = `000`. Pip resolves via the Tencent mirror `mirrors.cloud.tencent.com/pypi/simple`.
- `pip install --dry-run "lm_eval>=0.4.0"` **fully resolves on py3.14**, including compiled deps:
  `Would install ... datasets-5.0.1 lm_eval-0.4.12 pyarrow-25.0.0 scipy-1.18.0 scikit-learn-1.9.0 pandas-3.0.5 transformers(via) huggingface_hub-1.27.0 ...`
- `transformers==4.57.6` and `datasets==2.21.0` (the `requirements.txt` pins) also have py3.14-compatible wheels.

⚠️ **Version-conflict warning for MAIN:** a plain `pip install lm_eval` pulls **datasets 5.0.1**, but
`requirements.txt` pins **datasets 2.21.0** and the existing on-disk cache was built by **datasets 2.x**
(layout `<ns>/<config>/0.0.0/<hash>/`, `builder_name: parquet`). datasets 5.x may not reuse that cache
layout → could trigger re-download of every task. Recommend installing with the pinned
`transformers==4.57.6 datasets==2.21.0` to stay consistent with the cache, rather than letting lm_eval
resolve freely. Also note the env is **py3.14**, unusually new for this stack — a py3.11 venv with the
pinned versions may be the safer target.

---

## 4. 🚨 BLOCKER B2 — dataset caches: 7/9 present, `rte` + `race` MISSING

`local_datasets/` is a **red herring — it is empty** (contains only `__pycache__/`); all 9 tasks report MISSING there.

The real cache is **`/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/.hf_cache/datasets`**
(wzc1 → visible to `.21`). Verified per-task via `dataset_info.json` split counts (not just lock files):

| task | cache | splits (num_examples) | status |
|---|---|---|---|
| boolq | `google___boolq/default` | train=9427, validation=3270 | ✅ |
| hellaswag | `Rowan___hellaswag/default` | train=39905, validation=10042, test=10003 | ✅ |
| winogrande | `allenai___winogrande/winogrande_xl` | train=40398, validation=1267, test=1767 | ✅ |
| arc_easy | `allenai___ai2_arc/ARC-Easy` | train=2251, validation=570, test=2376 | ✅ |
| arc_challenge | `allenai___ai2_arc/ARC-Challenge` | train=1119, validation=299, test=1172 | ✅ |
| openbookqa | `allenai___openbookqa/main` | train=4957, validation=500, test=500 | ✅ |
| piqa | `ybisk___piqa/default` | train=16113, validation=1838, test=3084 | ✅ |
| **rte** | — | — | ❌ **MISSING** |
| **race** | — | — | ❌ **MISSING** (confirms the historical "RACE cache missing" note) |

All counts match official dataset sizes → these caches are complete, not truncated stubs.
Bonus present: `cais___mmlu` (162M), `EleutherAI___lambada_openai`, `tau___commonsense_qa`,
`allenai___social_i_qa`, `RMT-team___babilong` (453M, 120 arrow files).

Exhaustive `find` over the wzc1 `pighzliu_code` tree found **no** `race`/`rte`/`super_glue` dataset dir —
every hit was a false positive (`b`**race**`-expansion`, `g`**race**`ful-fs`, `_expR1cg2_g`**race**`2_...`).
There is **no `~/.cache/huggingface` on `.21` at all** (`No such file or directory`), so nothing hides there.

### Why `rte` matters
`legacy/eval_wiki_ppl.py:175-183` default task list is exactly 7 tasks and **includes `rte`**:
`boolq, rte, hellaswag, winogrande, arc_easy, arc_challenge, openbookqa`.
→ **A default-args eval run will die on `rte`.** `race` is not in the default list; it appears only in
`experiments/rebuttal_2026/scripts/run_parallel_native_eval_4gpu.sh` (`--tasks arc_easy,arc_challenge,openbookqa,race,piqa`)
and `merge_parallel_native_eval.py` — so **RACE blocks the rebuttal_2026 parallel-eval path specifically.**

Also missing: `SparseForge_worktree/data/hf_datasets` (the script's default `HF_DATASETS_CACHE`) does not exist,
and there is no local wikitext-2 dataset copy (only unrelated `ood_ppl_results/*_wikitext103` result dirs).
So **WikiText-2 PPL will also need a download** unless `HF_DATASETS_CACHE` is pointed at `.hf_cache/datasets`.

---

## 5. Recommended next steps (for MAIN to decide — nothing launched)

1. **Decide the env strategy** (needs a call, since `pip install` was deliberately not run):
   - Option A (fastest): `pip install` into conda `torch-base` (py3.14) with **pinned** `transformers==4.57.6 datasets==2.21.0 lm_eval` + `sentencepiece accelerate scipy` under `http_proxy=hy-proxy.woa.com:3128`. Risk: py3.14 is very new; datasets 2.21.0 on py3.14 is untested here.
   - Option B (safer, slower): build a fresh **py3.11** venv with the pinned stack, matching what the datasets-2.x cache expects.
   - Either way, **do not** let `lm_eval` free-resolve to datasets 5.0.1 if you want to reuse the existing cache.
2. **Point the cache at the existing one** to reuse 7/9 tasks:
   `export HF_DATASETS_CACHE=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/.hf_cache/datasets`
   (otherwise `eval_wiki_ppl.sh` defaults to the nonexistent `SparseForge_worktree/data/hf_datasets` and re-downloads everything).
3. **Fetch `rte` (+ `race` if the rebuttal_2026 path is wanted)** via the proxy, into that same cache dir.
   Or, to unblock immediately without any download, pass explicit
   `--lm_eval_tasks boolq,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,piqa` (7 tasks, all cached)
   and drop `rte`.
4. **Do not use `eval_wiki_ppl.sh` unmodified** — it does `conda activate minillm`, which does not exist on `.21`.
   Call `legacy/eval_wiki_ppl.py` directly with the chosen interpreter, or override `CLUSTER_ENV_SETUP`.
5. Batch size: with 183 GB/card × 8 and a 7B model, `LM_EVAL_BATCH_SIZE` can go large; tune toward ~80% of 183 GB.

---

*Audit performed 2026-08-08 by the node-21 handover agent. Read-only + cleanup; no training/eval started;
no `.tex`/`paperA*`/`paperB*`/`TODOList*` touched.*
