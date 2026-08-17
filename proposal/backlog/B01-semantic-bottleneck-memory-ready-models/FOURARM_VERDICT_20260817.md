# B01 — four-arm gate (`next_gate_executable_20260814`): VERDICT

**Date** 2026-08-17 · **Node** `.25` (`28.197.251.25`, 8× B200 sm_100, disk **wzz**)
**GPU spent** **1.74 GPU-h** of the 25–40 GPU-h budget (**4.3 % of the ceiling**)
**Outcome** **PARTIAL — 2 of 4 arms executed at full n; arms 3/4 NOT RUN, and the
reason is a measured defect in the assets, not a shortage of compute.**

Kill-gate verdict: **NOT KILLED, NOT PASSED — INSUFFICIENT.** None of the three
`kill_gate.conditions` is adjudicable from what this gate could legally run (§6).

---

## 1. What the gate asked for, and what actually happened

The gate (quoted verbatim in `STATUS.json`):

> Four-arm comparison at ONE scale, with the bottleneck latent **ACTUALLY PERSISTED**:
> (1) stock + Read-LoRA, (2) bottleneck only, (3) bottleneck + Read-LoRA,
> (4) bottleneck + Read-LoRA + Write-LoRA. Mandatory reported quantity:
> **bytes/token of what is written to the store (not of the restored hidden)**.

| arm | status | why |
|---|---|---|
| 1 · stock + Read-LoRA | ✅ **RUN**, n=1986 | assets present |
| 2 · bottleneck only, persisted | ✅ **RUN**, n=1986 | assets present; persist path works |
| 3 · bottleneck + Read-LoRA | ❌ **NOT RUN** | **base mismatch, measured** (§3) |
| 4 · + Write-LoRA | ❌ **NOT RUN** | §3 **and** cost 5.9× the whole budget (§4) |

I did **not** run arms 3/4 by lifting the argparse mutex that currently blocks them.
Lifting it is trivial and I verified PEFT would attach cleanly (§3.3) — which is
exactly why it would have been the wrong thing to do: it produces a number that
*looks* like "bottleneck + Read-LoRA" but is dominated by an artefact.

## 2. Results — the two arms that are legitimate today

Full LoCoMo, **n = 1986 per arm**, `selector=iter_bm25 --iter_rounds 2`,
`chat_template=False` (both repo-mandated), 4 strided shards × 1 GPU, merged.

| | arm 1 · stock + Read-LoRA | arm 2 · bottleneck only, **persisted** |
|---|---|---|
| `resume_j` | 12 | 13 ( = `bottleneck_layer`+1, enforced) |
| **bytes/token WRITTEN** | **8192** | **1024** |
| overall F1 | **9.24** | **3.85** |
| overall acc | **24.17** | **5.59** |
| overall EM | 0.40 | 0.05 |
| n | **1986** | **1986** |

Per category (n identical across arms — 282 / 321 / 96 / 841 / 446):

| cat | | arm 1 F1 / acc | arm 2 F1 / acc |
|---|---|---|---|
| 1 | multi_hop | 9.71 / 13.83 | 5.30 / 7.45 |
| 2 | single_hop | 6.03 / 7.79 | 1.51 / 1.25 |
| 3 | temporal | 7.60 / 19.79 | 5.38 / 11.46 |
| 4 | open_domain | 14.44 / 46.25 | 6.01 / 8.80 |
| 5 | adversarial | 1.79 / 1.79 | 0.22 / 0.22 |

**Paired test** (both arms answered the *identical* question set, so the pairing is
real, not assumed — id-set equality asserted, §5):

| metric | paired diff (arm1 − arm2) | bootstrap 95 % CI | excludes 0 |
|---|---|---|---|
| F1 | **+5.39 pp** | [+4.91, +5.88] | yes |
| acc | **+18.58 pp** | [+16.77, +20.39] | yes |
| EM | +0.35 pp | [+0.10, +0.65] | yes |

McNemar on accuracy: 411 discordant pairs, **p = 3.6e-89**.

### bytes/token — MEASURED off the write tensor, not derived

The gate insists on *"bytes/token of what is written to the store (not of the restored
hidden)"*. Measured as `numel * element_size() / T` on the tensor `write_chunk`
**returns**:

| arm | write tensor | dtype | **bytes/token** |
|---|---|---|---|
| 1 (stock) | `[1, 512, 4096]` | bf16 | **8192** |
| 2 (persist) | `[1, 512, 512]` | bf16 | **1024** |

**Ratio 8.0×.** `store_bytes_per_token()` returns 8192 / 1024 and was cross-checked
*against* the measurement rather than reported in its place. All four arm-2 shards
independently printed `store_bytes_per_token=1024` into their logs.

### The honest reading of these two arms

Arm 2 is **8× cheaper in storage and much worse in quality** (−5.39 pp F1,
−18.58 pp acc, both far outside CI). This is a **storage-vs-quality trade-off
measured at one point, not a frontier**, and the comparison is **confounded**: arm 1
carries a trained Read-LoRA and arm 2 carries none. So the 18.58 pp is **an
upper bound on the bottleneck's cost that also contains the entire Read-LoRA
benefit** — it is *not* "the price of the bottleneck". Deconfounding it is precisely
what arms 3/4 exist for, and that is what §3 shows cannot currently be done.

## 3. Why arms 3/4 were not run — a measured defect, not an excuse

Arms 3/4 need a Read-LoRA on top of the **funnel CPT** endpoint. The only Read-LoRA
on disk (`outputs/qcmem_distill_qwen_j12_r32_4k/final`, `distill_args.json`
`model_path=.../Qwen--Qwen3-8b`) was distilled against the **STOCK** upper stack.
The funnel endpoint (`unfreeze_from=12`) **retrained layers 12–35** — the exact band
the adapter spans (`layers_to_transform=[12..35]`, verified: 24 layers, 168 modules).

### 3.1 How far the CPT moved the read band

`proposal/shared/code/b01_cpt_drift_probe.py`, per-tensor
`‖W_ckpt − W_stock‖_F / ‖W_stock‖_F`, fp32:

| endpoint | frozen band 0–11 | trained band 12–35 |
|---|---|---|
| funnel `L12_d512` | **bit-identical, 24/24** (control) | rel_fro **0.0503 – 0.0926**, mean 0.0643 |
| stock-continued `L12` | **bit-identical, 24/24** (control) | rel_fro 0.0488 – 0.0645, mean 0.0568 |

The frozen-band control is what makes this attributable: `unfreeze_from=12` predicts
layers 0–11 are untouched, and they are **exactly** untouched (0 differing elements).
Had that control failed, my reading of the trainer would have been wrong and the
drift numbers would mean nothing.

### 3.2 The comparison that decides it

`proposal/shared/code/b01_lora_magnitude.py` measures the size of the correction the
adapter actually applies, `‖(α/r)·B A‖_F / ‖W_stock‖_F` (α/r = 2.0, 168/168 modules
scored, no failures). Compared **on the same 48 tensors** the drift probe measured:

| | value |
|---|---|
| Read-LoRA delta, rel_fro | 0.0059 – 0.0588 (mean 0.0132) |
| CPT drift, rel_fro | 0.0503 – 0.0926 (mean 0.0643) |
| **ratio drift / adapter-delta** | **min 1.57, median 6.06, max 8.62** |
| **tensors where drift > the adapter's own correction** | **48 / 48** |

**On every single matched tensor, the base moved further than the adapter's
correction — typically ~6× further.** Grafting this adapter onto the funnel would
measure a base mismatch and report it as an arm effect. That is why arms 3/4 are
`NOT RUN` rather than run-with-a-caveat.

### 3.3 The mutex is only argparse-deep — so the guard is *not* what stopped me

I checked whether the blocker is merely cosmetic, because "just remove the mutex" is
the obvious next move and I wanted to know what it would actually do:

- every driver refuses `--bottleneck_ckpt` + `--lora_adapter` (rc=**2**, verified);
- but PEFT targets the funnel model **correctly**: after `inject_bottleneck` renames
  layer 12's submodules to `...layers.12.inner.*`, `get_peft_model` still adapts
  exactly layers **12–35** with **7** LoRA modules at layer 12 — *identical* to the
  stock control (`b01_peft_funnel_probe.py`).

So the mechanism would work; **the science is what fails.** The mutex is arguably
right for the wrong reason, and a future agent who deletes it will get a clean-looking
run and a wrong conclusion.

### 3.4 What arms 3/4 actually require

Re-distil a Read-LoRA **against the funnel endpoint** (that is the missing asset), then
re-run. Cost by the repo's own measured Read-LoRA rate: 4000 steps @ 0.326 s/step
(ckpt mtimes, 3 intervals all 163 s; log's 25.3 samp/s independently gives 0.316 s/step)
× world 8 = **2.9 GPU-h per adapter**. Two adapters (arm 3, arm 4-read) ≈ **5.8 GPU-h**.
Affordable — it is simply **not an asset that exists today**.

## 4. `gpu_cost_estimate` is wrong for arm 4 by ~6× the entire budget

The estimate says *"Write-LoRA has NO measured rate in this repo, so the upper end of
the range is an extrapolation from the Read-LoRA rate."* **That extrapolation is
unsound, and the rate does exist** — on zwfy6, from `.82`:

`outputs/qcmem_writepath_distill_qwen_j12_r32/step{500..2500}`, 4 ckpt intervals:
13173 / 13170 / 13172 / 13167 s → **26.341 s/step**, spread 6 s over 4 intervals.
Cross-checked against the log's own `7.3 win/s` at batch 24 × world 8 = 192 win/step
→ **26.30 s/step**. Two independent sources, 0.16 % apart.

| | Read-LoRA | Write-LoRA |
|---|---|---|
| s/step (world 8) | 0.326 | **26.341** |
| data per step | 8 samples | **192 windows (24×)** |
| 4000 steps | 2.9 GPU-h | **234 GPU-h** |

**The write leg alone is ~234 GPU-h = 5.9× the 40 GPU-h ceiling** (even the existing
2500-step partial cost ~146 GPU-h). The estimate's error is treating "steps" as
comparable between the two trainers when the write trainer consumes **24× the data per
step**. Also: the on-disk Write-LoRA stops at **step2500/4000 with no `final/`**, so
arm 4 has no completed adapter either.

**Per the brief's instruction to stop rather than exceed the budget, I stopped.**

## 5. n-integrity — asserted, not eyeballed

The brief names a real accident: 8 structurally complete shards where every task was
`skipped:true / n=0` while `n_nan=0`, which a NaN-only aggregator passed.
`b01_fourarm_collect.py` therefore asserts, and **`verdict: PASS`, `problems: []`**:

- per-arm total **n == 1986** (expected) ✔
- per-category n **identical across arms** (282/321/96/841/446) ✔
- **sample-id *sets* identical** across arms — 0 exclusive either way. Equal *counts*
  over *different* samples is the failure a count-only check waves through ✔
- no shard contributed 0 rows; shard counts **497+497+496+496 = 1986** ✔
- 0 duplicate ids; **0 empty predictions** ✔

The paired statistics are recomputed from `preds` using the driver's **own**
`score_sample()` (which handles the 446 adversarial/abstention rows specially) and then
**reconciled against each committed `scores.json`**: agreement to <0.01 pp on all 3
metrics for both arms (F1 9.238951 vs 9.238951).

**The reconciliation check was verified to be able to fail.** My first version read a
nested key that does not exist, got `{}`, and reported `reconciles: true` — vacuously.
That is the fail-open pattern, and it passed. Fixed to read the driver's real flat
keys (`overall_f1`), to require all 3 metrics **and** `n_samples` agreement, and to
`return 1`. Negative control: `overall_f1 += 1.0` on a **copy** → `rc=1`,
`abs_delta.f1 = 1.0000`, explicit FAIL. A guard that has never failed is not known to
work.

## 6. Kill-gate adjudication — by `kill_gate`'s own three conditions

| condition (verbatim) | verdict |
|---|---|
| 低秩 latent 在 **RULER/LongEval** 上不保留精确 evidence | **NOT TESTED.** Only LoCoMo ran. `--persist_bottleneck_latent` exists **only** in `eval_qcmem_locomo.py` (11 hits); RULER/LongEval/BABILong/LongBench drivers have **0** — the flag is not plumbed, so this condition could not be evaluated at any cost. |
| fixed LM tax 在强模型上**扩大**而非缩小 | **NOT ADJUDICABLE.** Needs an LM-tax (PPL) measurement at 8B vs the 1B/3B ladder. This gate measured downstream QA, not PPL. The two 8B endpoints remain **never PPL-evaluated**. |
| full-depth RAG 在同存储预算下**严格支配** | **NOT TESTED.** No RAG / `kvdirect` / `hcache` arm was in this gate's four arms. |

**So: NOT KILLED and NOT PASSED.** No condition fired; none was answerable. Reporting
this as a pass on the strength of two arms would be the error the brief warns about —
the arms that ran are not the arms the kill conditions ask about.

## 7. What I did NOT do

- Did **not** lift the mutex to force arms 3/4 (§3.3 — mechanism fine, science not).
- Did **not** train any adapter (arm 3/4 read legs 5.8 GPU-h — affordable, but that is
  a **new asset**, i.e. a decision beyond this gate's scope, and arm 4's write leg
  blows the budget by 5.9×).
- Did **not** touch LOCAL / `.212` / `.73` / `.82` / `.104`. Only `.25`, verified 0
  compute apps immediately before launch (the launcher itself refuses to start
  otherwise). `.82` was read **read-only** to inspect the Write-LoRA.
- Did **not** run RULER/LongEval — the persist flag is not plumbed into those drivers.
- Did **not** use an LLM judge (LoCoMo's `--use_llm_judge` needs an API key); F1/EM/acc
  are the repo's default primary metrics.

## 8. Provenance

Assets on `.25` at `/root/b01_assets/` (local overlay — wzz ceph is 96 % full):
`base/` = Qwen3-8B (**`eos_token_id=151645` → this is the Instruct checkpoint**, which
is what **both** CPT endpoints were trained from per their `train_args.json`, so the
arms are internally consistent), `funnel/` (`bottleneck_dim=512, layer=12, step=2000`),
`baseline/` (`bottleneck_dim=0`), `read_lora/final` (md5
`d0a180a094bdf942b290bf0d6a667eb5`, verified post-transfer).

Eval deps were missing on `.25` and were added **outside** the shipped code tree
(`/root/b01_deps`) so `ship_code_to_node.sh` parity stays clean: `babilong` package
(the driver imports `babilong.metrics`) and `locomo10.json`. RULER's haystack loader
reads `f.read(8_000_000)` characters of `data/pg19_train.jsonl` (11.4 GB); I shipped a
24 MB prefix and **proved** it yields a byte-identical word list
(n_words 1429398, md5 `cb0f152eaa89aa85fc92e60fbdd2785a`, matching the full file).

Evidence (`evidence/`): `fourarm_{collect,bytes_per_token,paired_stats,arm1_scores,arm2_scores,cptdrift_funnel,cptdrift_baseline,readlora_magnitude,peft_funnel_targeting}_20260817.json`.
Generators: `proposal/shared/code/b01_{cpt_drift_probe,lora_magnitude,peft_funnel_probe,fourarm_collect,fourarm_bytes_measure,fourarm_paired_stats}.py`, `b01_run_fourarm.sh`.

## 9. Recommended next step

1. **Plumb `--persist_bottleneck_latent` into the RULER driver** (0 GPU). Kill
   condition #1 names RULER explicitly and is currently *unreachable*, not *unmeasured*.
   RULER self-generates from PG19, so no dataset work is needed.
2. **PPL-evaluate the two existing 8B endpoints** (funnel vs stock-continued). This is
   kill condition #2, it is the cheapest of the three, and both ckpts are already on
   `.25` — they remain never-evaluated.
3. **Re-distil a Read-LoRA against the funnel endpoint** (~2.9 GPU-h) to make arm 3
   legitimate, which also **deconfounds** the −18.58 pp reported above.
4. **Re-cost arm 4 before committing to it** — 234 GPU-h at the measured write rate, or
   redesign it (fewer steps / smaller batch); do not inherit the old estimate.
