# A02 — Matched-Quality Depth Control: PRE-REGISTRATION

**Written**: 2026-08-12, BEFORE any training GPU was spent (one 1-GPU × 20-step
probe excepted; it is GATE 0 below and its result is what shaped this design).
**Node**: `.82` (8× H20, zwfy6 disk). Verified idle: 8/8 cards 0 MiB / 0 %, no
stray `train_*` / `eval_*` / watcher processes, 3.2 T free.
**Target defect**: `STATUS.json → next_gate[3]`:

> "NOT tested anywhere yet: a matched-quality depth control (a LoRA distilled for
> j=0) — without it, depth cannot be isolated from adapter quality. This is the
> only remaining way to make a clean depth claim, and it requires TRAINING, not
> eval."

and its source, `A02_DEPTH_VS_RETRIEVAL_VERDICT.md §7.1`:

> "there is NO arm here that isolates depth with a matched-quality adapter …
> Removing this confound needs a j=0 control LoRA = TRAINING, out of scope for an
> eval gate."

---

## 0. What this is NOT

A02's original thesis is **DEAD** and this document does not revive it. Storage
form: h12 is **2048× raw text** (8192 vs 4 B/token). Read-compute form: a
**1.03–1.37×** per-query micro-optimisation, repaid after 8–226 queries vs a
matched-pack text-RAG control. Status stays
`needs_revision_original_thesis_dead_diagnostic_result_is_the_asset`.

This gate makes **one diagnostic claim clean**. Every outcome below is a *tax*
(an accuracy cost of the depth knob), so **no outcome of this gate can produce a
quality win**, and none is a thesis.

**Is this in the "would benefit plain text-RAG identically" bucket?** (STATUS
flags that trap for the two other candidate directions.) **No** — plain text-RAG
has no depth knob; it always reads at full depth. The depth axis is specific to
mid-depth resume. But that only makes the result *CoMem-specific*, not *good*:
it prices a knob, it does not find a win.

---

## 1. GATE 0 — the requested control is DEGENERATE BY CONSTRUCTION

Before designing a match, I checked whether the literal experiment is
well-posed. **It is not**, and this is the central finding.

`scripts/train_qcmem_distill.py` defines (lines 540–575):

* **TEACHER** = QCMem read at **j = 0**, `peft_model.disable_adapter()`, `no_grad`.
* **STUDENT** = QCMem read at **j = `--resume_j`**, adapters ON, LoRA on `layers[resume_j:]`.
* **LOSS** = bidirectional top-64 KL between them on the query-segment tokens.

At `--resume_j 0` the student and the teacher are **the same computational path**
— same pack, same depth, same RoPE, same mask — differing *only* by the LoRA
delta. PEFT zero-inits `lora_B`, so at step 0 the delta is exactly 0 and
**teacher ≡ student ⇒ loss ≡ 0**. The trainer's own docstring says so:

> "at `--resume_j 0` teacher==student by construction (both are the full forward,
> adapters are zero-init so make no difference at step 0)"

**Measured, not argued.** I ran the flagship recipe verbatim with *only*
`resume_j` changed 12 → 0, using the same trainer that built the on-disk ladder
(1 GPU, 20 steps, `logs/a02_gate0_j0_vacuity.log` on `.82`):

| | flagship j=12 (on disk) | **j=0 probe (this gate)** |
|---|---|---|
| `LoRA on layers[...]` | `[12:36]`, 58.20 M | `[0:36]`, **87.29 M** |
| loss @ step 1 | — | **0.0000** |
| loss @ step 10 | 0.2991 | **0.0021** |
| loss @ step 4000 | 0.0555 | (n/a, 20-step probe) |

The j=0 objective is **already globally minimised at initialisation**. Therefore:

> **A "LoRA distilled for j=0" under this recipe is not a matched-quality j=0
> adapter. It is a null adapter: its optimum is the identity, i.e. the base
> model.**

### 1.1 Consequence — the defect does not get *repaired*, it *dissolves*

"j = 0 with its own optimally-distilled adapter" **≡ j = 0 base model**, which is
**already on disk** as the dvr arm `j0_top12`
(`ruler_results/a02_dvr_ruler_j0_top12`, verified config `resume_j=0,
selector=iter_bm25, topk=12, iter_hop_topk=4, lora_adapter=None, baseline=none,
seed=42`).

So the dvr verdict's `read_deployed` step (`j0_top12 → c2_comem`, **−3 to −12 pp**
on RULER) **already is the matched-quality depth contrast**: each depth is being
run with the adapter its own distillation objective optimally supplies (identity
for j=0, the flagship 58.2 M adapter for j=12). §7.1's "no arm isolates depth
with a matched-quality adapter" is **over-pessimistic**.

The only way to give j=0 a *helpful* trained adapter is to change the objective
(task data, or a stronger-than-j0 teacher) — which would break the "only j
varies" match and answer a different question.

### 1.2 But there is a second-order effect worth measuring, and it is not free

At q ≡ p the KL and its gradient are both exactly zero, so in exact arithmetic
the adapter would never move. In **bf16** the two paths are *numerically*
distinguishable (the student path executes an extra matmul-and-add of zeros), so
gradients are tiny-but-nonzero. **AdamW normalises by √v and is therefore
scale-invariant**: arbitrarily small noise gradients still produce **full
lr-sized parameter updates**. Over 4000 steps at lr 8e-5 (cosine) the j=0 control
adapter is predicted to become an **Adam-amplified random walk on bf16 noise** —
plausibly *worse* than base, certainly not better.

That is a sharper, testable statement than "it is vacuous", so I will **run the
requested experiment to completion** rather than assert vacuity from a 20-step
probe.

---

## 2. Design — what is MATCHED, what is VARIED

### 2.1 Derived from (reported per instruction)

| artefact | role |
|---|---|
| `scripts/_launch_p2_4_depthcurve.sh` (on `.82`, 6011 B, Aug 2 15:04) | **the launcher I derived from.** It already reproduces the flagship recipe verbatim, changing only `--resume_j`. My driver is this file with the depth list and two extra knobs. |
| `scripts/train_qcmem_distill.py` (zwfy6 md5 `a8c56b100432bb10293820b0936a6874`) | the trainer. **No new trainer written.** |
| `outputs/qcmem_distill_qwen_j12_r32_4k/distill_args.json` | ground-truth flagship recipe |
| `proposal/active/A02-comem-write-read-repair/code/run_a02_depth_vs_retrieval.sh` | eval-arm dispatch + GATE 0 sha gate + bounded pool, reused verbatim in spirit |

⚠️ **Version skew found and handled**: the wzc1 copy of the trainer (md5
`8e2c94bb…`) has `--keep_last_n/--keep_steps` rotation args; the **zwfy6 copy
does not**, and the zwfy6 copy is the one that built the on-disk ladder. My first
probe died with `error: unrecognized arguments: --keep_last_n 3`. The driver
therefore passes **no rotation flags** — matching the ladder exactly. Adapters are
~223 MB each; 8 saves/run × 2 runs ≈ 3.6 GB, fine against 3.2 T free.

### 2.2 The comparison it is compared against, config-confirmed

The j=12 reference is the flagship, confirmed from **on-disk artefacts**, not prose:

* recipe: `outputs/qcmem_distill_qwen_j12_r32_4k/distill_args.json` →
  `resume_j=12, lora_rank=32, lora_alpha=64, lora_dropout=0.0, targets=q,k,v,o,gate,up,down,
  chunk=512, n_ctx=3, query_loss_tokens=0, teacher_topk=64, distill_lambda=0.6,
  ce_weight=0.0, total_steps=4000, lr=8e-05, warmup=100, wd=0.0, grad_accum=1,
  grad_clip=1.0, gradient_checkpointing=false, dtype=bf16, attn=sdpa, seed=42`
* adapter: `final/adapter_config.json` → `r=32, alpha=64, layers_to_transform=12..35`
* identity: `sha256(final/adapter_model.safetensors)` = `dd09cd17457c63578c0f38da`
  on zwfy6 == the flagship sha the dvr driver fail-closes on. **Re-asserted as
  GATE A in my driver.**
* eval arm: `ruler_results/a02_ruler_c2_j12_readlora` → `resume_j=12,
  selector=iter_bm25, topk=12, iter_hop_topk=4, chunk_size=512,
  lora_adapter=outputs/qcmem_distill_qwen_j12_r32_4k/final, baseline=none, seed=42`

The **existing ladder** `j∈{6,9,18}` (`outputs/qcmem_distill_qwen_j{6,9,18}_r32_4k`)
was verified field-by-field against the flagship `distill_args.json`: `lora_rank
32, lora_alpha 64, lora_dropout 0.0, n_ctx 3, chunk 512, total_steps 4000, lr
8e-05, warmup 100, wd 0.0, grad_accum 1, grad_clip 1.0, teacher_topk 64,
distill_lambda 0.6, ce_weight 0.0, gradient_checkpointing False, seed 42,
top_prepay_b 0, query_loss_tokens 0, targets identical, dtype bf16, attn sdpa` —
**all identical; only `resume_j` differs.** So the match is real, not nominal.

**Data identity** (the match's other load-bearing leg): `data/pg19_train.jsonl`
is **11450766349 B on both disks**, `md5(first 100 MB)` = `ed2868281b1dc815dc96e2f3592cccbe`
and `md5(last 100 MB)` = `bec4b5f244b85f1dbd66cd9054058630` **identical wzc1 vs
zwfy6**. With `seed=42`, `world_size=8`, `n_ctx=3` fixed, `PG19Packer` shards
windows `wcount % world_size == rank` over a deterministic single-pass stream ⇒
**every arm sees the same windows in the same order**. Training tokens are
therefore matched exactly: 4000 steps × 8 ranks × 2048 tok = **65.5 M tokens/arm**.

### 2.3 Matched vs varied

| axis | value | matched? |
|---|---|---|
| backbone | `models/Qwen--Qwen3-8b` (36 L). **Not `Qwen3-8B-Base`** — the flagship adapter's `base_model_name_or_path` and every phase-1/dvr arm used `Qwen--Qwen3-8b`; switching would break the match. Verified present on zwfy6. | **MATCHED** |
| teacher | QCMem read at j=0, adapters off, no grad | **MATCHED** |
| data + order + tokens | PG19, seed 42, ws 8, n_ctx 3 → 65.5 M tok | **MATCHED (byte-identical file)** |
| LR schedule | 8e-5, warmup 100, cosine to 0, AdamW β=(0.9,0.95), wd 0, clip 1.0 | **MATCHED** |
| seq_len / eff batch | 2048-tok window, 1 window/rank/step, grad_accum 1 → 8 windows/step | **MATCHED** |
| loss | bidir top-64 KL, λ=0.6, ce 0 | **MATCHED** |
| dtype / attn / gc | bf16 / sdpa / **off** | **MATCHED** |
| retrieval at eval | `--selector iter_bm25 --topk 12 --iter_hop_topk 4 --sink_tokens bos --chunk_size 512`, `chat_template=False` | **MATCHED (byte-identical)** |
| **resume depth j** | **0 / 6 / 9 / 12 / 18** | ***VARIED — the only thing*** |
| adapter capacity | r=32/α=64 ⇒ `75776 × r × (36−j)` params: **87.29 / 72.74 / 65.47 / 58.20 / 43.65 M** | **NOT matched — inherent to `layers[j:]`; addressed by Arm 2** |

### 2.4 Arm 2 — the capacity control that removes the one unmatched axis

Capacity is *structurally* tied to j (the span is `layers[j:36]`), so it cannot
be matched by holding r fixed. But it can be matched **exactly** by moving r.
LoRA params per layer per unit r for Qwen3-8B = `r·Σ(fi+fo)` over the 7 targets
= `r × 75776` (q 8192 + k 5120 + v 5120 + o 8192 + gate 16384 + up 16384 + down
16384), which reproduces the observed 58.20 M / 65.47 M / 72.74 M / 43.65 M exactly.

> **`r=40` at `j=12` (24 layers) = 75776 × 40 × 24 = 72,744,960**
> **`r=32` at `j=6`  (30 layers) = 75776 × 32 × 30 = 72,744,960**  → **EXACT match.**

So Arm 2 = **j=12, r=40, α=80** (α/r = 2.0 preserved, as flagship 64/32).
It answers: *is the j=6-vs-j=12 difference about depth, or about capacity?*

### 2.5 Arms

| # | arm | j | adapter | source | status |
|---|---|---|---|---|---|
| A0 | `j0_top12` | 0 | none (= the optimal j=0 adapter, per GATE 0) | on disk | **reuse** |
| **A1** | `j0_control_lora` | 0 | **NEW** r32/α64, `layers[0:36]`, 87.29 M | **TRAIN** | the literal requested control |
| A2 | `j6_lora` | 6 | r32/α64 72.74 M | on disk | **reuse adapter, RULER never run** |
| A3 | `j9_lora` | 9 | r32/α64 65.47 M | on disk | **reuse adapter, RULER never run** |
| A4 | `c2_comem` | 12 | flagship r32/α64 58.20 M, sha `dd09cd17…` | on disk | **reuse** |
| A5 | `j18_lora` | 18 | r32/α64 43.65 M | on disk | **reuse adapter, RULER never run** |
| **A6** | `j12_r40_capmatch` | 12 | **NEW** r40/α80, 72.74 M | **TRAIN** | capacity control == A2 |

Only **two** trainings are needed; the depth ladder already exists and has
**never been evaluated on RULER** (`ruler_results/*p2_4*` and `*depth*` are both
empty; only LoCoMo/quality evals exist for it). That is the cheap, real gap.

### 2.6 Read-out

**Primary — RULER, where retrieval is CLOSED** (dvr: recall@12 = 99–100 %, so
retrieval cannot confound): `niah_multikey_1`, `variable_tracking` × {16k, 32k},
n=100/cell, 8 shards, `limit 100`. Statistic: per-cell accuracy, and the paired
per-cell delta vs **A0** (the j=0 full-depth anchor) = *the read tax at depth j*.

**Secondary — BABILong qa1/qa2/qa5 × {16k, 32k}**, run for contrast only. dvr
established these cells are **retrieval-dominated** (54.9–78.6 %, recall@12
22.9–63.2 %), so they **cannot** support depth inference; they are reported to
show the curve is only interpretable where retrieval is closed.

**Aggregation hygiene**: per-cell only. **No pooled BABILong / LongEval figure**
will be computed or quoted (the banned −17.89 pp / +2.00 pp).

### 2.7 Pre-registered predictions (so the result can falsify me)

1. **A1 ≈ A0, or A1 < A0.** If the j=0 control instead comes out *significantly
   better* than base, GATE 0's vacuity argument is wrong and I must retract it.
2. **A1's training loss stays ~1e-3 for all 4000 steps** (vs flagship 0.2991 →
   0.0555). This is the machine-checkable form of "no learning signal".
3. **Read tax is monotone-ish in j** across A0/A2/A3/A4/A5 on RULER.
4. **A6 ≈ A4.** If A6 ≫ A4, the depth curve is partly a capacity artefact and
   every depth claim must be re-stated at matched capacity.

### 2.8 Integrity gates (fail-closed, mirroring the dvr gate)

* **GATE A**: `sha256(flagship final)` first 20 hex == `dd09cd17457c63578c0f` else abort.
* **GATE B**: assert each reused ladder adapter's `adapter_config.json` has
  `r=32, alpha=64, layers_to_transform == [j..35]`, and its `distill_args.json`
  matches the flagship on all 20 recipe fields except `resume_j`.
* **GATE C**: shard completeness — RULER `tasks×lens×8` records, BABILong 8 CSVs
  per cell; **refuse to score a partial cell** (the dvr precedent; silent 7/8
  merges have destroyed a口径 in this project before).
* **GATE D**: `chat_template=False` asserted in every emitted cell config.
* **GATE E**: canonical scorers **imported, never reimplemented** (RULER harness
  per-item `correct`; BABILong `babilong.metrics.compare_answers`). This project
  has twice had a reimplemented metric produce a spurious significant result.

---

## 3. Per-rank memory and batch size

**The trainer is a full replica per rank — not FSDP, and not even stock DDP.**
`train_qcmem_distill.py` (lines ~470–500) deliberately avoids `DistributedDataParallel`
because QCMem calls `causal_lm.model.layers[i](...)` directly rather than
`module.forward()`, so DDP's hooks would never fire; it instead does an explicit
`dist.all_reduce(grad)/world_size`. Per CLAUDE.md's rule, **adding cards does not
reduce per-card memory**.

Per-rank static budget (Qwen3-8B = 8.19 B params, bf16):

| item | bytes | note |
|---|---|---|
| frozen backbone | 8.19 B × 2 = **16.4 GB** | full replica, every rank |
| LoRA params (worst arm, A1 87.29 M) | × 2 = 0.17 GB | bf16 |
| LoRA grads | 0.17 GB | |
| AdamW `exp_avg` + `exp_avg_sq` | 0.35 GB | LoRA only; backbone is frozen |
| **static subtotal** | **≈ 17.1 GB** | |
| activations | small | 2048-tok pack, batch 1, grad only on `layers[j:]` |

⇒ ~20–35 GB of 97.8 GB, i.e. **~20–36 %**, well under the ≥80 % target.

**Batch size chosen: unchanged, and it is not tunable here.** Two independent
reasons, stated plainly rather than papered over:

1. **The trainer has no batch dimension.** There is no `--batch_size` flag
   (`grep batch_size scripts/train_qcmem_distill.py` → nothing). `PG19Packer`
   yields exactly one window per step and `write_chunk` consumes a single
   `[1,T]` chunk. The only volume knobs are `n_ctx` (window length) and
   `grad_accum`.
2. **The match forbids raising them.** `n_ctx=3` and `grad_accum=1` are part of
   the flagship recipe. Changing either changes the tokens/step and the data
   partition, destroying the very "matched training tokens / matched eff batch"
   property this gate exists to establish. **Filling the card would invalidate
   the experiment.**

So per CLAUDE.md's sanctioned escape hatch, recorded explicitly: **batch size is
fixed by the recipe-match requirement, not by memory; there is no adjustment
room.** I keep `gradient_checkpointing=false` (flagship) since 17 GB static
leaves ample headroom; `GRAD_CKPT=1` remains available as a gradient-identical
OOM safety net.

**`--eval_interval`: this trainer HAS NO SUCH FLAG.** Verified —
`grep -n 'eval_interval|quick_eval|babilong' scripts/train_qcmem_distill.py`
returns only a docstring mention of `eval_qcmem_babilong.run_self_test`. There is
**no inline eval inside the step loop**, so the NCCL-watchdog SIGABRT failure
mode cannot occur, and passing `--eval_interval 0` would be a bogus argument that
argparse would reject and that would kill the launch (exactly how my first probe
died on `--keep_last_n`). All eval is offline, after training, on saved adapters.

---

## 4. Cost, from MEASURED rates on this hardware

Ladder timings on **`.82` (H20)**, from `logs/p2_4_depthcurve_master.log` — 4000
steps each: **j=6 (30 grad layers) 67.9 min · j=9 (27) 65.4 min · j=18 (18) 57.5 min**.
Reported `samp/s` = `steps × world_size / dt`, so `steps/s = samp/s ÷ 8`
(j=6: 8.0 ÷ 8 = 1.0 steps/s ⇒ 4000 s ⇒ 66.7 min ✓ consistent with the wall clock).

⚠️ The flagship's own 24.5 samp/s is **not** usable as an ETA here: its
`distill_args.json` `model_path` is the **wzc1** path, i.e. it ran on **L20A**,
not H20. Mixing the two would understate ETA ~3×.

Linear fit on grad-bearing layer count (18→57.5, 27→65.4, 30→67.9 min):

| arm | grad layers | predicted | 
|---|---|---|
| **A1** j=0 | 36 | **≈ 82–95 min** |
| **A6** j=12 r40 | 24 | **≈ 62–68 min** |

Evals (from the dvr progress log on a **4**-GPU pool): RULER 2 tasks × 2 lens ×
8 shards = 13.4 min/arm → ~7–8 min on 8 GPUs; BABILong ~1.2 min/cell → ~9 min per
6-cell arm. 5 arms to evaluate ⇒ ≈ 85 min.

**Total ≈ 4–4.5 h on 8 idle H20s, serial.** I will re-report s/step from a real
window of my own log (≥ 30 steps, `elapsed/iter`), never from the first one or two
lines.

---

## 5. What this WILL and WILL NOT license

**WILL license** (if it runs clean):

* The retirement of `next_gate[3]`: the matched-quality j=0 depth control is
  **degenerate under this recipe**, demonstrated by running it, not asserted.
* Restating `A02_DEPTH_VS_RETRIEVAL_VERDICT.md §7.1`: `read_deployed` (−3 to
  −12 pp on RULER) **already is** the matched-quality depth contrast, because the
  optimal j=0 adapter is the identity.
* A **5-point depth-tax curve** (j = 0/6/9/12/18) with retrieval held
  byte-identical and the distillation recipe held fixed — the first such curve on
  a retrieval-closed benchmark.
* Whether that curve survives **exact capacity matching** (A6 vs A4).

**WILL NOT license**:

* Any revival of A02's thesis. Storage stays dead (2048×); read-compute stays
  1.03–1.37×.
* Any claim that CoMem beats RAG on quality. Every number here is a **tax**.
* Any claim about `j12_frozen`'s −97 pp being a "depth effect" — that remains
  "untrained depth-12 resume is non-functional".
* Any BABILong-based depth claim (retrieval-dominated there, by dvr).
* Any cross-model / cross-family claim: Qwen3-8B only, one seed (42), H20/bf16/sdpa.
* Any claim of differential LR or capacity matching **across** the ladder — the
  ladder is capacity-*un*matched by construction (§2.3); only A6-vs-A4 is matched.

---

## 6. Failure modes I will report as results, not hide

* If A1 (j=0 control) trains to a **non-trivial loss**, GATE 0 is wrong → retract §1.
* If A1 evaluates **better** than A0, "optimal j=0 adapter == identity" is wrong.
* If A6 ≠ A4 materially, the depth curve is capacity-contaminated → every depth
  number must be re-stated at matched capacity, including the dvr's.
* If any cell is shard-incomplete, GATE C refuses it; the cell is reported
  **absent**, never merged partial.
