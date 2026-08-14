# ALPS+SLoRB GATE0 — PASSED (224/224), and the earlier FAILED verdict is retracted

**Recorded 2026-08-15. Cost: 1.12 GPU-h total (two 20-step probes on 4 B200 cards).**
Supersedes `status/ALPS_SLORB_GATE0_FAILED.md`, which adjudicated run A as FAILED at 02:30.
Both of that document's two failure signals were **artifacts of the probe's own configuration**,
not properties of the arm. Each is disproved below by a measurement, not by an argument.

GPUs 0-3 only. **GPUs 4-7 were never touched** — watcher PID 176751 (`ARM=slorb`,
`GPUS=4,5,6,7`) stayed alive throughout, so the `.212` +/-SLoRB comparison is intact.

---

## STEP 1 — the mask. PASS, and it cost 0 GPU-h

**No mask needed to be generated.** A real ALPS 2:4 mask already existed:

```
outputs/paper_v2/alps/llama2_wandb_sf_alps_v1_alps_seed0/mask.pt
6,607,138,721 bytes, dated 2026-07-31
```

The brief said `SparseForge_Data/results/alps_mask_dense_init/` held only a 615-byte
`cast9_summary.json` and that "the first step is to generate the mask". That directory is
indeed nearly empty, but the mask lives elsewhere: it is the artifact
`scripts/h20/run_alps_slorb_rebuttal_queue.sh:37` already points `MASK=` at. Running
`run_alps_llama2_nm.py` would have re-derived a file that was already on disk.

All five required checks, from `SparseForge_Data/results/alps_mask_seed0/mask_validation.json`
(validator: `Mixture-of-Memory/scripts/validate_alps_mask_artifact.py`, CPU-only):

| # | check | measured |
|---|---|---|
| a | `format` | `"sparseforge-mask-v1"` ✅ |
| b | `pattern` | `"2:4"` ✅ |
| c | nnz-per-group-of-4 histogram | **`{2: 1619001344}`** — a single bucket; no group has 0/1/3/4 ✅ |
| d | in-scope module count | **224** ✅ |
| e | global zero fraction | **0.500000000** (nnz 3,238,002,688 / elems 6,476,005,376) ✅ |

Two details worth keeping:

- The artifact holds **225** 2-D entries, not 224. The extra one is `lm_head`, which
  `save_mask_artifact()` exports because it is an `nn.Linear`, but which is **not** a
  `SparseLinear` in the trainer and is therefore never loaded. 224 is the in-scope count and it
  matches `verify_2of4_pre.log`'s `in-scope tensors: 224` for the same model.
- `elems` equals `6,476,005,376` exactly, the same number in the ALPS eval-arm verify logs.

---

## STEP 2 — GATE0. PASS: `aligned=224/224`, twice, both `rc=0`

I added the counter the hazard note asked for (`main_llama.py`, in
`_load_external_fixed_masks`). It compares `module.mask.shape` against
`module.weight.shape` — the pairing the optimizer relies on — and **raises** unless the count is
224. Both runs printed:

```
[Common Recovery] GATE0 mask/param alignment: aligned=224/224 misaligned=0
[Common Recovery] Loaded 224 fixed masks (pattern=2:4) from .../mask.pt
[Common Recovery] SLoRB-only isolation: trainable=848,429,056, frozen=13,214,421,888, trainable_tensors=448
```

| run | log | steps | s/it (Δelapsed/Δiter, iters 2→20) | rc |
|---|---|---|---|---|
| A | `logs/alps_slorb_gate0_20260815_021407.log` | 20/20 | **25.056** | **0** |
| B | `logs/alps_slorb_gate0_20260815_023200.log` | 20/20 | **25.556** | **0** |

Peak memory **~86.8 GB / 183 GB per card (47%)** at `micro=8, block=4096` — headroom for
`micro=16`, though see the cost section: throughput is not the binding constraint.

`srste_decay == 0.0` confirmed, and now **enforced**: loading a fixed external mask with
`--srste_decay != 0` raises. The argparse default is **`6e-5`, not 0** — so a caller who simply
omitted the flag would have silently armed the exact buggy branch
`SRSTE_SILENT_DEGRADATION_HAZARD.md` documents. That is now impossible on this path.

### Retraction 1 — "only 96 of 224 masks are visible" is a debug-print artifact

The line
`[nm_2_4_tile_stats DEBUG] sparse_linear_count: 224, processed: 96 ... skipped 128`
does **not** mean 128 masks were lost. `processed`/`skipped` are **rank-0-local counters** in
`utils.py:1040-1046`, incremented before the FSDP all_reduce. Under `hybrid_sharded` across 4
ranks, each rank holds a slice of the flat parameter buffer, so 128 of the 224 masks have
`numel()==0` *on rank 0* and are skipped *by rank 0*.

The decisive check is the reduced total on the very same line:

```
total_tiles: 1619001344.0        (after dist.all_reduce, utils.py:1150)
elems/4    = 6476005376/4
           = 1619001344          (STEP 1, independently, on CPU, from the artifact)
```

**Exact match.** Every one of the 1,619,001,344 tiles was counted across ranks; nothing was
dropped. `96 + 128 = 224` is arithmetic about one rank's shard, not about coverage.

So the training-time count and the finalization count do **not** disagree. The authoritative
per-module check is the 224/224 assertion, which runs **before** FSDP wrapping, where masks are
whole.

### Retraction 2 — the frozen `loss=27.9` was my own logging bug, not a stalled model

I passed `--output_flip_every 1000000`. The tqdm postfix is refreshed **only** inside
`if iter_num % args.output_flip_every == 0` (`main_llama.py:2917`), so in run A it was written
once at iter 0 and then never again. The identical `27.9` across all 24 readings was **one
stale value redrawn 24 times** — not 24 measurements agreeing.

Run B, identical except `--output_flip_every 1`, gives a moving loss:

```
27.9  26.8  27.1  26.9  26.6  25.7  25.2  24.5  25.0  25.2
25.1  24.3  24.7  25.2  24.6  26.3  27.3  26.0  24.0  25.9  24.0
```

Descending with noise, as a 20-step window at `lr` ramping from 0 (warmup 375) should be.

### The level (~25, not ~1.7) is also expected, and is not a perplexity

`exp(27.9)` is meaningless here: the logged number is
`hardness_task*task_loss + hardness_kldiv*kl_loss`, and `kl_loss` carries a `temperature**2 = 4`
factor (`sparse_modeling.py:1883`). Decomposition:

- The +SLoRB reference arm logs `1.7148` at iter 0 because it starts from **dense** weights with
  an all-ones mask, so student == teacher and `kl_loss ≈ 0`; the total is just `task_loss`.
- This arm starts from an **ALPS 2:4-pruned** student against a **dense** teacher. Measured
  ALPS-pruned `wiki_ppl = 10.288` (`tables/alps_slorb_accuracy_curve.csv`) implies
  `task_loss ≈ ln(10.288) = 2.33`, leaving ~25.6 for `4*KL(teacher||student)` — a large but
  finite divergence, which is precisely the quantity SLoRB is being asked to close.

The two numbers are not comparable, and `CLAUDE.md`'s `PPL > 1000` rule does not apply to a
KL-inclusive distillation objective.

### `flip_ratio = 0` is correct here, not a symptom

`flip_ratio` counts mask flips. This arm sets `change_mask=False`, and `main_llama.py:912-915`
sets `_hardening_finalized=True` at load time. **A flip would be a bug.** Constant 0 is the
intended behaviour of a fixed-support run.

---

## Two real bugs found and fixed (both were blocking)

1. **`--eval_interval 0` crashed the trainer.** `main_llama.py:2151` computed
   `iter_num % args.eval_interval` unguarded → `ZeroDivisionError` on the first step, on all
   ranks. `CLAUDE.md` mandates `--eval_interval 0` for training (inline BABILong eval in a DDP
   loop desyncs ranks → NCCL watchdog SIGABRT), so **the documented-safe setting was the one
   setting that could not run.** Guarded at three sites (2157, 2657, and the `save_interval`
   derivation at 2499). Note the tempting workaround — "pass a number larger than max_iters" —
   does **not** work: `iter_num == 0` is divisible by every interval, so eval still fires once.
2. **`SLoRB_init_type` needs care, and `sum` is right here for a non-obvious reason.**
   `init_SLoRB()` runs **twice**: once from `initialize_model()` (`utils.py:81`) while masks are
   still all-ones, and again from `main_llama.py:921` after the ALPS mask is installed. In the
   first context `W*(1-mask) ≡ 0`, so `sum` and `zero` are numerically identical — this is the
   degeneracy the AST recon measured. In the second it is **not** degenerate. Measured on
   `layers.0.self_attn.q_proj` with the real ALPS mask (`/tmp/slorb_init_probe.py`, CPU):

   | quantity | all-ones mask (ctx 1) | real ALPS 2:4 mask (ctx 2) |
   |---|---|---|
   | `max|W*(1-mask)|` | **0.0** | **1.240e-01** |
   | `max|SLoRB_Weight|` after `sum` | **0.0** | **2.666e-01** |
   | `rms|SLoRB_Weight|` | 0.0 | 1.432e-02 |

   So SLoRB **is** non-degenerate at init on this mask, and the concern that
   `--freeze_non_slorb` leaves nothing trainable is disproved: 848,429,056 params are trainable
   and the branch starts from the pruned mass rather than from zero.

---

## STEP 3 — cost. **211 GPU-h. Over the 12 GPU-h threshold by ~17x. NOT LAUNCHED.**

### The 5294 knee does not apply to this arm — do not segment by it

The brief asked for a two-segment estimate around `mask_hardening_start=5294`. **That knee is a
property of the +/-SLoRB arms, not of this one**, and applying it here would be wrong:

- In those arms (`change_mask=True`), once `iter >= 5294` the per-module `hardening_x` drops
  below 1, so `sparse_modeling.py:797` leaves the `hx >= 1.0` fast path and calls
  `_hard_mask_from_soft()` on every forward, while `calculate_model_mask()` keeps running EMA
  mask updates. Hence 15.5-19.8 s/it before, 50-57 s/it after.
- This arm sets `change_mask=False`, `mask_hardening_start=0`, `mask_lr=0`, and
  `main_llama.py:912-915` pins `hardening_x=1.0` with `_hardening_finalized=True` at load time.
  The forward takes `effective_mask = self.mask` unconditionally for all 7500 iters.
  **There is no second regime; a single measured rate applies end-to-end.**

### The numbers

Rate is `Δelapsed/Δiter` over iters 2→20 (dropping iter ≤1, which includes model build and
first-step warmup). Both probes started at iter 0, so tqdm's elapsed and iter share an origin
and the 9x resume-origin trap does not apply.

| config | s/it | wall | GPU-h |
|---|---|---|---|
| 4 cards, measured | **25.31** | 52.7 h | **211** |
| 8 cards @100% scaling (bound) | 12.65 | 26.4 h | 211 |
| 8 cards @85% scaling | 14.89 | 31.0 h | 248 |
| 8 cards @70% scaling | 18.08 | 37.7 h | 301 |

Adding cards cannot help: `global_batch_size=256` is fixed, so 8 cards halve the grad-accum
micro-steps per card and GPU-h is flat at best, worse with imperfect scaling.

**12 GPU-h buys 427 iters = 5.7% of the horizon.** Per instructions: reporting, not launching.

### Before anyone authorises 211 GPU-h, two prior questions

1. **An ALPS+SLoRB arm may already exist.** `SparseForge_Data/tables/alps_slorb_accuracy_curve.csv`
   has two `ALPS_SLoRB_only` rows at 33,554,432 and 624,951,296 recovery tokens
   (avg9 53.19 and 53.78, vs `ALPS_native` 51.24). Its `args.json`
   (`rebuttal_artifacts/2026-07-27/common_recovery/alps_slorb_rebuttal_20260725_v5_*/`) confirms
   the intended treatment: same `initial_mask_path`, `SLoRB=True`, `SLoRB_k=16`,
   `freeze_non_slorb=True`, `change_mask=False`, `srste_decay=0.0`. **The 7500-iter config
   would be 7.86B tokens = 13x the largest existing point.** Whether the reviewer needs a
   token-matched arm or whether the 625M point already answers them is a scoping decision I am
   not making unilaterally.
2. **That existing run is not drop-in comparable, and its differences must be priced.** Its
   `args.json` shows `SLoRB_init_type='zero'` (not `'sum'`), `dataset='c4_llama'` (not
   `dolmino-mix-1124-llama2`), `block_size=2048` (not 4096), `learning_rate=1e-6` (not 2e-5).
   Its run directory is also named `m-magnitude`, i.e. `mask_metric='magnitude'` — inert under
   `change_mask=False`, but it means **the directory name does not identify the mask actually
   used**; only `initial_mask_path` in `args.json` does.

---

## Not changed, flagged only

- **`nm_2_4_tile_stats`' `processed`/`skipped` debug counters are rank-local and read as a
  coverage failure.** They cost one FAILED adjudication today. A one-line all_reduce of
  `processed_count` would make them mean what they appear to mean. Left alone: it is shared
  code on the live +/-SLoRB path and this is not the moment to perturb it.
- **`adamw.py:102-112` SR-STE `else: mask = None` is still silent.** Now unreachable *on this
  path* by assertion, but the asymmetry with the CAST branch's `raise` (`:173`) remains for any
  future `--srste_decay > 0` arm.
- **`alps_slorb_accuracy_curve.csv`'s two rows have *different* provenance, and only one is
  traceable.** The `33554432` row's `wiki_ppl 7.951222005396828` matches
  `medium256/*/eval.json`'s `7.951221942901611` (in-run eval). The `624951296` row's
  `7.619548918314024` does **not** match `full625m/*/eval.json`'s `7.623181343078613`; it comes
  from the separate `alps_slorb_full_independent_eval.log`, which prints
  `[RESULT] Wiki PPL: 7.6195`. So the two CSV rows mix an in-run number with an
  independent-rerun number. Worse, that same log then shows a traceback in
  `eval_wiki_ppl.py:304 run_lm_eval_benchmarks` immediately after the PPL line, so **the 9 task
  accuracies in that row are not derivable from it** and their source is unlocated. Not
  touched — flagged for whoever cites that row. Its `lm_eval` task list is also the **7**-task
  set, not the union-9 set.
