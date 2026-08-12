# DISK_CLEANUP_AUDIT_20260812.md — two-disk deletion audit

> Scope: full audit of both filesystems for safely-deletable folders / checkpoints / markdown.
> **Default verdict is KEEP.** Every `DELETE`/`PRUNE` row below carries the evidence for all five
> mandatory safety checks. Anything short of certain is filed `NEEDS-DECISION`, not deleted.
> Method note: on this `dop-fuse` mount `ls -ld <dir>` reports the **recursive byte count**
> (verified equal to `du -sb`), so sizes below are exact-at-audit-time, not samples.

## 0. Measured `df` (≥3 samples each, spaced ~8 s — dop-fuse lags)

| Disk | Before | After | Δ |
|---|---|---|---|
| **wzc1** (LOCAL + .21) | 120T / **110T used** / 11T free / **92%** | 120T / **109T used** / 11T free / **91%** | −1T used, 92%→91% |
| **zwfy6** (.73/.82/.104) | 689T / **652T used** / 38T free / **95%** | 689T / **650T used** / **39T** free / 95% | −2T used, +1T free |

Total measured reclaim this pass: **~3 TB** (1T wzc1 + 2T zwfy6). This is on top of the
1.27 TB the user had already reclaimed from `outputs/olmo2_keep14_densesave_reheal/`
(not redone; `step67500.pt` confirmed still present).

## 1. Live writers at audit time (check #1 baseline for every row)

| Node | Disk | Process | Writes to |
|---|---|---|---|
| LOCAL | wzc1 | `main_llama.py` ×10 (SparseForge) | `out_llama_tokenmatched_noslorb/` |
| .21 | wzc1 | `main_llama.py` ×10 (SparseForge) | `out_llama_tokenmatched_slorb/` |
| .73 | zwfy6 | `eval_olmo2_closedbook_qa.py` ×8 | **reads** `outputs/olmo2_probe2_7B_full32_dolmino/step25000.pt` |
| .82 | zwfy6 | `train_qcmem_distill.py` ×8 (A02) | `outputs/a02_j12_capmatch_r40/` |
| .104 | zwfy6 | `train_qwen3_arch_probe2.py` ×8 (paperC) | `outputs/paperC_qwen3base_heal_k8f2/` |

All five were re-confirmed healthy **after** the deletions. Nothing above was touched.

---

## 2. EXECUTED — `PRUNE`: byte-identical `final.pt` ↔ `step{max}.pt` hardlink dedup

**The single largest zero-risk win, and it destroys no provenance at all.**

`scripts/train_olmo2_arch_probe2.py:1073-1077` writes the terminal step twice: the
`step % save_every == 0` branch writes `step{N}.pt`, then the loop exits and `_save(..., final=True)`
writes `final.pt` from the *same* in-memory state, seconds later. So the two files are the same
checkpoint under two names.

Verified on `outputs/olmo2_probe2_7B_keep14fresh2` tensor-by-tensor:

- `model_state`: **0 of 179 tensors differ**
- `optimizer_state`: param_groups equal; **0 of 179** params differ in `exp_avg`/`exp_avg_sq`
- `rng_state`: torch RNG equal, all **8** CUDA device RNG states equal
- `train_args`, `step`, `epoch`, `max_steps`, all arch descriptors: equal
- the on-disk size delta (6023 B) is **purely the zip entry-name length** (`final/...` vs
  `step200000/...`); entry count 731 = 731 and summed entry sizes are identical

Cheap identity oracle used for the rest: compare the zip central directory
(`(entry_name_without_root, file_size, CRC32)`), which reads only metadata. It agreed exactly with
the tensor-level result on the pair above, and it **correctly rejected** 6 pairs that merely had
similar sizes (`olmo2_p24_sft_*`, `paperC_depthsweep_keep24_scratch_refusal25`, the 4 zwfy6
`olmo2_p24_sft_*`) — those are genuinely different checkpoints and were left alone.

**Action taken:** `cp -l final.pt .tmp.link && mv -f .tmp.link step{N}.pt` — atomic, and
**both filenames still exist and still resolve**. Any table, script, or `SOURCES.md` citing
either name keeps working. Re-verified post-hoc that both names load with correct
`step`/tensor-count/optimizer state.

| Disk | Dir | Deduped name | Size |
|---|---|---|---|
| wzc1 | `outputs/olmo2_probe2_7B_keep14fresh2` | `step200000.pt` | 46G |
| wzc1 | `outputs/olmo2_probe2_7B_keep14fresh2_freezefront` | `step200000.pt` | 25G |
| wzc1 | `outputs/olmo2_probe2_7B_keep14fresh2_fromscratch` | `step200000.pt` | 46G |
| wzc1 | `outputs/olmo2_probe2_7B_keep14fresh2_seed1234` | `step200000.pt` | 46G |
| wzc1 | `outputs/olmo2_probe2_7B_shortgpt16` | `step200000.pt` | 46G |
| wzc1 | `outputs/paperC_depthsweep_keep14_rtrunk_refusal25` | `step1000.pt` | 46G |
| wzc1 | `outputs/paperC_depthsweep_keep14_scratch_refusal25` | `step1000.pt` | 46G |
| wzc1 | `outputs/paperC_depthsweep_keep20_rtrunk_refusal25` | `step1000.pt` | 59G |
| wzc1 | `outputs/paperC_depthsweep_keep20_scratch_refusal25` | `step1000.pt` | 59G |
| wzc1 | `outputs/paperC_depthsweep_keep24_rtrunk_refusal25` | `step1000.pt` | 68G |
| wzc1 | `outputs/paperC_depthsweep_keep28_rtrunk_refusal25` | `step1000.pt` | 78G |
| wzc1 | `outputs/paperC_pc1_squad_A3_keep20fresh2` | `step1000.pt` | 59G |
| wzc1 | `outputs/paperC_pc1_squad_A3_keep24fresh2` | `step1000.pt` | 68G |
| wzc1 | `outputs/paperC_pc1_squad_A3_keep28fresh2` | `step1000.pt` | 78G |
| zwfy6 | `outputs/olmo2_probe2_1B_keep12f2_dolmino_stageB_seed{101,102,103}` | `step5000.pt` | 16G ×3 |
| zwfy6 | `outputs/olmo2_probe2_1B_keep7fresh2_16card` | `step200000.pt` | 12G |
| zwfy6 | `outputs/paperC_depthsweep_keep{14,20,28}_graft_refusal25` | `step1000.pt` | 25/29/35G |
| zwfy6 | `outputs/paperC_pc1_squad_A4_keep{20,24,28}fresh4` | `step1000.pt` | 34/37/40G |

wzc1 ~719 G + zwfy6 ~254 G of duplicated bytes collapsed. Note this touches
`paperC_pc1_squad_A4_*` and the `stageB_seed10*` dirs that A04/paperC cite — **safe precisely
because both names survive**; had I deleted one name this would have been forbidden.

## 3. EXECUTED — `PRUNE`: `watchdog_ckpts/` (zwfy6, was 962 G → now 231 G)

- **#1 no writer**: `babilong_ckpt_watchdog.py` not running on .73/.82/.104; `fuser` clean on every
  `.pt`. The daemon was killed 2026-05-11 (`UPDATELOG.md:4750`, "Killed stale watchdog").
- **#2 refs**: only `scripts/babilong_ckpt_watchdog.py:44` (the dead daemon's own
  `LOCAL_CKPT_DIR`) and `.watchdog_state.json` (its state file). Zero refs from
  `paper*/`, `proposal/`, `code/`.
- **#3 purpose discharged**: this dir is a **staging mirror** the daemon rsync'd in purely to feed
  BABILong eval. The resulting numbers live in `status/babilong_realtime.jsonl` (12 rows) and the
  verdict is in `UPDATELOG.md` 2026-05-11: *"PPL/NIAH trade-off 假说被实证否决 … 整个 H 系列
  cross-attn memory 在 bAbI 风格 retrieval 任务上无法 generalize"* — every recorded cell is
  **0.0** (qa1/qa2/qa5 across 1k-8k). Training curves survive in
  `logs/experiment_h1*.log` (metrics JSON at tail).
- **#4 not parked work**: H-series is a closed direction; `scripts/train_cross_attn_memory.py` last
  touched 2026-05-10. No `PENDING_TASKS.md` / `proposal/` entry asks for H-series eval.
- **#5 trajectory**: not a published trajectory — no paper cites any H step. I still kept **one
  checkpoint per arm** so any arm remains re-evaluable.

Deleted: `H10/step_{500..3000}` (6), `H11_v2/step_{500..4000}` (8), and
`H12/step_1000.pt`. Kept: `H10/step_3500`, `H11_v2/step_4500`, `H12/step_500`,
`H13_isolate/step_2500`, `H14_isolate_aggr/step_1500`.

> **Bonus finding — a corrupt file:** `watchdog_ckpts/H12/step_1000.pt` was **19,610,468,352 B vs
> the 49.6 GB of every sibling, mode `600`, and not a valid zip** (`BadZipFile: File is not a zip
> file`). It is a **truncated partial rsync** from the 2026-05-11 daemon kill — unloadable, so it
> was never evidence for anything. All 5 retained files verified as openable zips (1262 entries).

## 4. EXECUTED — `DELETE`: `distill_cache/pg19_512_nctx{15,63}` (zwfy6, 394 G + 86 G = 480 G)

- **#1 no writer**: 0 live `train_mem_space*` / `build_distill_cache` processes on any zwfy6 node.
- **#2 refs**: every reference is a **launcher, and all are ≥6 weeks stale** — 17 in
  `legacy/launchers/`, 3 in `legacy/eval_sched_dead/`, and the `scripts/` ones last committed
  **2026-06-20…2026-07-01**. No `paper*/`, `proposal/`, `SOURCES.md` reference at all.
- **#3 purpose discharged**: this is the **v21 self-study distillation teacher cache** (top-64
  logits + hidden states dumped from a frozen Llama-3-8B). Its scientific output is written up in
  `status/RUN_REGISTRY.md` §"mem_space self-study 蒸馏" (the step-ladder table, the SWA contrast
  qa1 32k 8→15, and the authoritative "蒸馏退化机制总账" section) plus
  `versions/v21_selfstudy_distillation.md`. **This is the dead mem_space v21 direction, NOT the
  live QCMem self-distillation** (which is `train_qcmem_distill.py` → `outputs/qcmem_distill_*`
  and was left completely untouched).
- **Decisive point — it is a derived cache, fully regenerable.** `scripts/build_distill_cache.py`
  is present, and all three inputs are still on zwfy6: teacher `../models/Llama--Llama3-8b`,
  `data/pg19_train.jsonl` (11.4 GB), and the dolmino corpora. The builder is explicitly designed
  to be reproducible — its docstring guarantees the cache key `(doc_idx, group_pos)` is
  "INVARIANT to the per-epoch shuffle / DDP sharding", walking docs in original order with no
  shuffle. Regenerating costs GPU time; it loses no information.

`distill_cache/512` (946 G, the dolmino cache) was **kept** — see NEEDS-DECISION.

## 5. EXECUTED — `PRUNE`: discharged peak-memory probes (zwfy6, 133 G)

- **#2 refs**: `peakmem_smoke`, `peakmem_smoke8`, `peakmem_smoke8e`, `pm8_c256`,
  `peakmem_smoke8sd1024` have **zero references anywhere** in the repo (`*.py/*.sh/*.md/*.tex/*.json`).
- **#3 purpose discharged, and pointable**: these were VRAM-ceiling probes. Their entire
  deliverable is the batch-size table now in `status/RUN_REGISTRY.md` — "推荐物理 bs … H20 (97 GiB):
  chunk128 → bs=16 … chunk512 → bs=6", with the stated purpose *"落账目的：未来 launch 直接查表设
  `--batch_size`，无需重新 probe"*. A peak-memory measurement never needs its weights.
- Deleted **only** `full_model.pt` from each; kept `adapter_config.json` + the wandb run logs.
- `outputs/dms_8x_smoke/checkpoint-2` (16 G) deleted: a **2-step** smoke run
  (`UPDATELOG.md:425`), and its `final/` is retained.
  ⚠️ **`outputs/dms_8x/` was NOT touched** — `CODE_CLEANUP_SUGGESTIONS.md` marks it
  「待评估，**绝不能动**」. `dms_8x_smoke` is a different directory.

---

## 6. KEEP — leads that did NOT pan out (reported plainly)

### 6.1 `outputs/mem_space*` (zwfy6, 39 dirs / 1.42 TB) — **KEEP. The lead was wrong.**

The hypothesis was "if the BABILong result is recorded in `RUN_REGISTRY.md`, the weights are
prunable". Checking every dir individually, the premise fails in both directions:

- **Only 3 of 39 dirs are named anywhere in `RUN_REGISTRY.md`** (`p11_chunk512_deltarule_normreadout`,
  `p11_chunk1024_deltarule_normreadout`, `p11_chunk512_INSTRUCT`). The 5 big FIFO dirs
  (141.9 G each = 710 G, the bulk of the 1.42 TB) have **0 mentions** — the registry records
  results under *arm* names (`b25/c512`), never the directory. So "is it a row in RUN_REGISTRY"
  cannot be answered per-directory, and absence of a row is **not** evidence of expendability.
- **Worse, check #4 actively fails.** `status/PENDING_TASKS.md` carries live
  `auto_launch: true` tasks that name these exact intermediate checkpoints:
  *"★ b25/c512 中间 ckpt 早评（step500/1000/1500/2000/2500）— auto_launch: true"* citing
  `outputs/mem_space_fifo_b25_chunk512/full_model_step00{500,1000,1500,2000,2500}.pt`, plus two
  follow-on `auto_launch: true` tasks (real-long-document benchmark; b50/b100 cross-arm ladder).
  These are parked, not cancelled — exactly the situation check #4 forbids touching.
- And the science is explicitly **unfinished**: RUN_REGISTRY's own FIFO section says the headline
  was retracted for BABILong contamination and demands two follow-ups — *"(1) memory-disabled
  对照隔离 8k-32k 真实性；(2) 用 held-out … 重测"* — with *"在此之前 b25 不能算 SOTA"*.
  Re-scoring needs the weights.
- `scripts/launch_b25_ckpt_eval_196.sh:10-11` hardcodes `full_model_step000500.pt` /
  `full_model_step001000.pt`; `report/figs/fifo_hidden_recall_qa1_8k.json` and
  `scripts/e2_fullquery_probe.py` cite `..._noleak/full_model.pt`.

**Verdict: nothing in `mem_space*` is safe to delete.** A correct "no" here.

### 6.2 `outputs/olmo2_p13_scratch16_lr2e5_uniform` (zwfy6, 244 G) — **KEEP**

Task #127 is `[completed]`, but the intermediate checkpoints are explicitly designated resume
material, and `paperB` names them three times:

- `paperB/TODOList.md:54` — *"保留 step5000…25000 checkpoint"*, and §P1.3: *"checkpoint 每 5000 存
  (step0/5000/…/25000.pt 在 H20 FS)，**若 reviewer 强烈要求可 resume 续训到 200k**"*
- `paperB/FINDING3_LR_CONTROL_DESIGN.md:119` names `--resume_from …/step25000.pt` as the launch spec
- `paperB/audit_20260805/crossing_meaning_audit.json:146` cites the surviving checkpoint set as
  evidence for the arm's stopping point

This is a `[STOPPED EARLY]` arm kept deliberately resumable for a reviewer request, i.e. check #4.
It also never got held-out PPL/MMLU (*"未跑 held-out PPL/MMLU/closed-book"*), so its number does
**not** yet live anywhere else. **KEEP all 5 checkpoints.**

### 6.3 `paperC_squad_results`, `paperF_evalfragility`, `evidence_squad_label_prior`, `evidence_evalfragility_code` — **FORBIDDEN, re-verified**

`evidence_squad_label_prior/` is cited by `proposal/active/A01-null-calibration-methodology/SOURCES.md`;
`evidence_evalfragility_code/` by B04. Both look like dead-direction leftovers and are in fact live
proposal evidence sources. Not touched.

### 6.4 `paperD_research/align_acts` (zwfy6, 19.1 G) — **KEEP (cheap, and provenance-adjacent)**

paperD is archived (`proposal/README.md:267`, "跨家族 layer stitching 方法已死亡"), and the derived
CKA matrices **are** committed (109 files under `proposal/shared/representation/cka_matrices/`), so
the numbers survive without the raw activations. But `proposal/shared/representation/README.md`
declares this directory the store of *"the 91-pair CKA evidence **used by the null-calibration
proposal**"* — i.e. it feeds **A01, which is active**. 19 G is not worth the risk of severing a live
proposal's evidence chain. Left in place.

### 6.5 `outputs/hyv3_probe2_keep36_fresh2/step200.pt` (wzc1, 285 G) — **KEEP (deliberate survivor)**

Tempting: a single step-200 MoE toy from an archived direction, 285 G, whose run ended in SIGKILL.
But `status/SESSION_HANDOFF.md:302` shows this file is the **deliberately chosen survivor** of an
earlier cleanup: *"ckpt 285GB(含fp32 optim)已 rm 省磁盘（⚠️CEPH 92%），**仅留 keep36/step200.pt**"*.
It is the last remaining artifact of the prune-heal frontier (keep36→ppl12) whose 4 points are cited
in `status/PENDING_TASKS.md:101`. Deleting the one file a previous audit chose to keep needs an
explicit decision → NEEDS-DECISION.

### 6.6 Other checks that came back clean

- **Stray `.pt`/`.bin` outside `outputs/`**: all legitimate — `data/dolmino-mix-1124-llama2/train.bin`
  (311 G, SparseForge's live training corpus), `external/landmark_ckpts/` (reproduction baseline),
  `models/Meta-Llama-3-8B` (base weights). Nothing orphaned.
- **`*_20260731.zip` / `*_20260801.zip` archives**: searched both disks — **none exist**. That lead
  was empty.
- **`results/` / `logs/` / `*_results/`**: no orphaned multi-GB junk. Largest are
  `olmo2_mmlu_content_results` (1.09 G wzc1 / 1.66 G zwfy6) and `logs/` (407 M / 1.0 G) — small, and
  they are the per-item provenance for Paper B / B04 tables.
- **`final.pt`-only dirs** (`sembott_*`, `qwenbott_*`, `minarch_1b_*`, `qwen3_minarch_*`): no
  duplicate to collapse, and `qwen3_minarch_armB_f12k2_200k/final.pt` is the ckpt task #129 scored.

---

## 7. NEEDS-DECISION (not touched)

| Path | Disk | Size | Why I stopped | What would settle it |
|---|---|---|---|---|
| `pighzliu_code/out_llama/` (96 dirs w/ `.pt`) | wzc1 | **4.7 TB** | **Biggest single opportunity on wzc1 by far**, and it is *outside* the repo. But it is SparseForge's historical checkpoint pool and **actively load-bearing**: task #244 (2026-08-11) re-scored the paper's 5B headline directly from `…_20260413_201320/model_best_lm_eval.pt`, and `SparseForge_Data/tables/{checkpoints.tsv,out_llama_ast7_checkpoint_inventory.csv}` name 4 dirs as `usable`/`available` table candidates. Only **7** of 96 dirs are cited by name → the other ~89 are *probably* dead sweep runs, but SparseForge is mid-rebuttal and re-scoring baselines this week. | Owner confirms which `out_llama/*` runs the NeurIPS table + rebuttal still need. Then delete the complement — likely **multiple TB**. Each dir holds `model.pt` + `model_best_lm_eval.pt` (80-170 G/dir) with `args.json`/`best_lm_eval.json` metadata that could be retained as the record. |
| `distill_cache/512` | zwfy6 | **946 G** | Same dead v21 direction and same regenerable-derived-cache argument as the two pg19 caches I did delete, and it has the same stale-launcher-only reference profile (19 refs). I stopped because it is the **dolmino** cache — the corpus behind the *primary* self-study result in RUN_REGISTRY — and 74,032 `.npz` files is a much larger regeneration bill than the pg19 pair. | Confirm the mem_space/v21 self-study line will not be re-scored. Then delete for ~946 G. |
| `outputs/hyv3_probe2_keep36_fresh2/step200.pt` | wzc1 | **285 G** | A prior cleanup explicitly kept *only* this file when deleting its siblings (§6.5). Overriding that needs a decision, not an inference. | Confirm the Hy-MT2 prune-heal frontier (keep36→ppl12) will never be re-measured; its loss curves already survive in `logs/hyv3_probe2_keep*_fresh2.log`. |
| `MemLong/` | wzc1 31 G / zwfy6 300 G | 331 G | Third-party baseline checkout. Not audited in depth this pass. | Confirm no paper still cites a MemLong baseline number. |
| `watchdog_ckpts/` remaining 5 files | zwfy6 | 231 G | Kept one per arm so each stays re-evaluable, but the H-series verdict is **already definitive and entirely 0.0** (§3). If nobody will ever re-run it, all 5 can go. | One-line confirmation that the cross-attn H-series is permanently closed → +231 G. |

---

## 8. What was NOT touched (protected, verified intact after the fact)

- `outputs/olmo2_probe2_7B_keep8fresh2/step131000.pt` — PRESENT (34,152,196,306 B)
- `outputs/olmo2_probe2_7B_keep10fresh2/step90000.pt` — PRESENT (39,009,621,855 B)
- `outputs/olmo2_probe2_7B_keep12fresh2/step166000.pt` — PRESENT (43,867,049,986 B)
  → the three deprioritised-but-not-cancelled Paper B resumes. **Do not delete; they do not expire.**
- Trajectory arms `olmo2_probe2_7B_keep14fresh2_seed1234` / `_shortgpt16` / `_full32_dolmino`:
  **no intermediate step deleted.** The only change is the terminal-step hardlink dedup, which
  keeps both filenames resolvable.
- `outputs/olmo2_keep14_densesave_reheal/step67500.pt` — the user's retained crossing point, present.
- `outputs/dms_8x/`, `paperC_squad_results`, `paperF_evalfragility`,
  `evidence_squad_label_prior`, `evidence_evalfragility_code` — untouched.
- All 5 live jobs re-confirmed running after deletions (LOCAL 10 procs, .21 10 procs,
  .104 writing `step1500.pt` at 16:51).

## 9. Note for future cleanups

The `final.pt` == `step{max}.pt` duplication in §2 is **structural**, not a one-off: every run of
`train_olmo2_arch_probe2.py` (and the qwen3 sibling) that reaches a `save_every` multiple at its
last step produces it, because `final.pt` is exempt from rotation by design
(`"final.pt is never rotated"`). Two consequences:

1. Currently-running trainings will create fresh duplicates when they finish — worth re-running the
   dedup periodically.
2. A cheap permanent fix would be for `_save(..., final=True)` to hardlink when an identical
   `step{N}.pt` already exists. **Not implemented here** — that is a trainer code change and three
   trainings are mid-flight.
