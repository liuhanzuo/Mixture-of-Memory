---
scope: A03 data-order replication (seed 43 / 44 / 45) — PRE-REGISTRATION
date: 2026-08-10 19:20 GMT+8
status: PRE-DATA. Written while both runs are at step ~204100 of 220000; no
        step220000 checkpoint exists on either disk, no eval has been run, and
        the three `dataorder_seed*` cells in the recompute script all read
        `PENDING (… absent)`. Verified at time of writing.
primary_endpoint: triviaqa `em`, step220000, paired vs `A03_1B_keep7_step200k`
protocol: per-item paired difference bootstrap, n_boot=5000, seed=42, CI95
          percentile; SIG = CI excludes 0. UNCHANGED from Arms 3/4/6.
evidence_for_priors: evidence/arm3_arm4_arm6_cpt_trajectory_paired_full.json
          (md5 28584639f120aaff07bd1a52120f983e, canonical on wzc1)
---

# Does Arm 3's `triviaqa em +0.479pp SIG` survive a change in data order?

## 1. The question

`ARM6_FINAL_VERDICT.md` establishes that two independent LR bands land on the
same value at step220000. Read from the evidence JSON, not from prose:

| arm | LR band (peak) | step205000 | step210000 | step215000 | **step220000** |
|---|---|---|---|---|---|
| Arm 3 (cosine tail) | [0.325x, 0.249x] | −0.4848 SIG | +0.3734 SIG | −0.0167 TIE | **+0.4793 SIG** |
| Arm 6 (lower band) | [0.499x, 0.425x] | −0.7300 SIG | +0.7746 SIG | −0.0557 TIE | **+0.5016 SIG** |
| Arm 4 (peak-anchored) | [0.998x, 0.559x] | −1.4044 SIG | +1.2595 SIG | −0.1170 TIE | **−0.9307 SIG** |

All n = 17944 items. Arm 3 step220000 CI95 = [+0.2675, +0.6910]
(half-width **0.2118 pp**); Arm 6 = [+0.2564, +0.7356] (half-width 0.2396 pp).

`ARM6_FINAL_VERDICT.md` then shows the replication is **not independent**: the
arms trace the same curve (Pearson r Arm3–Arm6 over all 4 dose points =
**0.9642**), because `ce5c298` ("pass seed=args.seed to every shuffling
DistributedSampler") existed only on wzc1 and was never copied to zwfy6, so all
three arms ran with `DistributedSampler(ds, shuffle=True)` — sampler seed 0 —
and consumed the **identical minibatch sequence** (training-loss correlation
Arm3–Arm6 = **0.99982**, Arm3–Arm4 to step215 = 0.99187; STATUS.json
`phase_locked_defect` / `distributedsampler_fix_not_on_disk`).

So "Arm 6 replicates Arm 3" currently reduces to "the same data prefix produces
the same effect at the same step", which it must. **Zero data-order replication
exists.** This experiment supplies it.

## 2. The manipulation actually worked (pre-data check, already done)

Before committing to a decision rule it is worth confirming the seed argument
now reaches the sampler — otherwise the experiment is vacuous. Computed by MAIN
at 19:05 GMT+8 from the four training logs on zwfy6, over the common step window
**200020–202880 (144 logged loss points)**:

| pair | r (loss series) |
|---|---|
| Arm 3 vs Arm 6 (both sampler seed 0) | **+0.99966** |
| Arm 3 vs seed 43 | **−0.0101** |
| Arm 3 vs seed 44 | **+0.0087** |
| Arm 6 vs seed 43 | −0.0102 |
| Arm 6 vs seed 44 | +0.0087 |
| seed 43 vs seed 44 | **+0.0041** |

The phase-lock is gone: the new runs are mutually uncorrelated and uncorrelated
with the old arms, while the two old arms remain at r ≈ 1.0. The data order
genuinely differs. (`+0.99966` here vs the `0.99982` quoted in
`ARM6_FINAL_VERDICT.md` is a different window, not a discrepancy — mine is the
first 144 points only.)

Both new runs are otherwise **Arm 3's exact config**: `max_steps=300000`,
`warmup_steps=150`, `lr 2e-5 / min_lr 2e-6`, resumed from the same
`outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt`. Confirmed identical
LR at the resume point — `lr_fresh(now)=6.504e-06` appears verbatim in Arm 3's
log and in both new logs. Arm 6's is `9.984e-06` (its `max_steps=373000`), which
is why Arm 6 is not the comparison config here.

## 3. Decision rule — fixed now, before any number exists

The prose rule in `ARM6_FINAL_VERDICT.md` §"The one experiment that would settle
it" is:

> If `triviaqa em ≈ +0.48 SIG` at all 3 data orders → the CPT recovery effect is
> real and A03 has a publishable narrow claim. If it scatters across
> [−0.9, +1.3] → the effect is a data-order artifact and the retraction returns,
> this time correctly grounded.

That is too loose to adjudicate on: "≈ +0.48" has no width, and "[−0.9, +1.3]"
is Arm 4's **across-step** span (its step220000 = −0.9307 and step210000 =
+1.2595), which is a within-arm oscillation, not an across-seed scatter. The
following makes it numeric. **It is binding.**

### 3.1 Definitions

Let a seed *land* iff its `step220000.pt` passes the watcher's v3 write guard and
all four axes score **8/8 shards** (the recompute script hard-fails otherwise).
For each landed seed *i*, let θ_i = triviaqa `em` delta in pp vs
`A03_1B_keep7_step200k`, with its own bootstrap CI_i.

Reference value: the mean of the two phase-locked arms,
θ_ref = (0.4793 + 0.5016)/2 = **+0.4905 pp**.

### 3.2 The replication band — [+0.20, +0.80] pp

σ_item = half-width / 1.96 = 0.2118/1.96 = **0.108 pp** (Arm 3); 0.2396/1.96 =
0.122 pp (Arm 6). Take σ_item ≈ 0.11 pp.

If the effect is real at θ_ref and run-to-run variance is **no larger than** the
item-sampling variance already measured (σ_run ≤ σ_item), then a new seed's point
estimate has SE = √(σ_item² + σ_run²) ≤ 0.11·√2 = 0.156 pp, so a 95% predictive
interval for a new data order is θ_ref ± 1.96·0.156 = **±0.305 pp**.

> **Replication band := [+0.20, +0.80] pp** (= +0.4905 ± 0.305 → [+0.186,
> +0.795], each endpoint rounded to one decimal place).

This is the interval a genuinely-real +0.49 effect should reproduce into, under
the most generous variance assumption the existing data can support. Its
half-width (0.30 pp) is **1.42×** the observed bootstrap half-width on this axis
(0.2118 pp), so it is not a disguised demand for bit-reproducibility.

### 3.3 Per-seed classification (binary, exhaustive)

* **CONFIRM_i** ⟺ CI_i excludes 0, θ_i > 0, **and** θ_i ∈ [+0.20, +0.80].
* **NOT-CONFIRM_i** ⟺ everything else. This deliberately includes all three
  failure modes without further judgement: SIG-negative, TIE/null, and
  SIG-positive-but-out-of-band (θ_i > +0.80).

Note that both seeds being in-band forces their span ≤ 0.60 pp, so no separate
"scatter span" criterion is needed; the band subsumes it.

### 3.4 Aggregate verdict (n = number of landed seeds, n ≥ 2 required)

| outcome | condition | disposition |
|---|---|---|
| **REPLICATES** | every landed seed is CONFIRM | Effect is real under data-order variation. `ARM6_FINAL_VERDICT.md`'s positive reading **stands**. A03 keeps a **narrow** claim, stated with its scope: this base ckpt, this corpus slice, Arm 3's schedule, step220000, n data orders. |
| **ARTIFACT** | **zero** landed seeds are CONFIRM | Effect is a data-order artifact. `ARM6_FINAL_VERDICT.md`'s positive reading is **RETRACTED**, this time grounded. A03 retains only its Gate-1 pilot. |
| **MIXED** (the ambiguous middle) | ≥1 CONFIRM and ≥1 NOT-CONFIRM | **Pre-declared: counts as failure of the claim under test.** See 3.5. |

### 3.5 The ambiguous middle — disposition declared in advance

If the outcome is MIXED:

1. The claim under test is *"the effect survives a change in data order."* One
   non-confirming data order is a **direct counterexample**. Replication logic
   here is **conjunctive, not majority-vote** — a 1-of-2 or 2-of-3 result does
   not establish it.
2. Therefore: the positive reading of `ARM6_FINAL_VERDICT.md` is **retracted as
   a general claim**, and the headline may **not** be the confirming seed(s).
3. A03's disposition becomes `hold_in_backlog` with a written requirement of
   **n ≥ 5 data orders** before any positive claim is reconsidered.
4. **No tie-breaking seed may be added to resolve a MIXED result.** Adding runs
   after seeing the split is optional stopping and is forbidden here. Seeds
   43/44/45 are the whole pre-registered set; seed 45 counts only because it is
   declared *now* (§5), not after the fact.
5. The interim public statement is the negative one.

### 3.6 Primary vs secondary endpoints

* **triviaqa `em` is the sole primary endpoint.** It is the metric that generated
  the claim. The verdict in 3.4 is decided on it alone.
* popqa, nq_open and MMLU (letter + content_norm), and triviaqa `contains`/`f1`,
  are **secondary**: reported for completeness, and explicitly **may not be used
  to rescue a failed primary**.
* `contains`/`f1` agreeing with `em` is **not** extra replication. Per
  `evidence/TCODEX_AUDIT_20260810.md`, item-level delta correlations on Arm 3
  step220000 are corr(em,f1)=0.781, corr(contains,f1)=0.497,
  corr(em,contains)=0.380, giving an effective **1.8–1.9** independent tests,
  not 3.
* No multiplicity correction is applied to the primary decision, because there
  is exactly one primary test per seed.

## 4. What n = 2 can and cannot conclude

Seed 45 is queued for .104 but blocked on the keep12 7B run, which at 19:16
GMT+8 was still training with all 8 GPUs at 96421 MiB and an elapsed time of
2 d 05 h. **Plan for n = 2 and treat n = 3 as upside.**

**n = 2 CAN:**
* **Falsify** data-order independence. One NOT-CONFIRM seed is a counterexample,
  and falsification does not need n = 3. The ARTIFACT and MIXED branches are
  therefore both fully decidable at n = 2.
* Show the manipulation had an effect at all (already shown, §2).

**n = 2 CANNOT:**
* Estimate σ_run usefully. With n = 2 the sample SD has one degree of freedom;
  E[s] = σ·√(2/π) ≈ 0.80σ with an enormous spread. It cannot distinguish
  σ_run ≈ 0 from σ_run ≈ 0.3 pp.
* Support any claim of the form "the effect is +0.5 ± Y pp across data orders".
* Establish REPLICATES as a promotable finding — two agreeing draws are
  consistent with a real effect but do not bound the variance.

**Disposition if exactly 2 land and both CONFIRM:** record as *"consistent with a
real effect; NOT established"*. A03 **stays at proposal stage** — this does not
satisfy the promotion gate in CLAUDE.md, and the n = 2 caveat must appear in the
STATUS.json verdict string and in any downstream write-up.

**Disposition if exactly 2 land and ≥1 is NOT-CONFIRM:** the ARTIFACT or MIXED
branch fires immediately. No waiting for seed 45.

**Disposition if fewer than 2 land:** no verdict. `INCONCLUSIVE_INSUFFICIENT_N`;
the pre-registration stays open and the failed run is re-run per §5.2.

## 5. Runs, immunity to Arm 4's defect, and the one live risk

### 5.1 The runs

| seed | node | disk path | started | step220000 ETA |
|---|---|---|---|---|
| 43 | .82 | `outputs/olmo2_probe2_1B_keep7f2_dolmino_dataorder_seed43/` | 16:55:29 | ~04:19 GMT+8 |
| 44 | .73 | `outputs/olmo2_probe2_1B_keep7f2_dolmino_dataorder_seed44/` | 16:57:29 | ~04:21 GMT+8 |
| 45 | .104 | queued, blocked on keep12 7B | — | — |

Driver `scripts/_run_a03_dataorder_repl.sh` (md5
`8f0ca66d3453beb4b9d2345a0453980d`, identical on both disks). Eval automation:
`code/a03_dataorder_trajectory_watcher.sh` + `code/a03_dataorder_ext_driver.sh`,
one watcher per node, each pinned to its own seed. Watchers running detached
(ppid 1) since 19:14 GMT+8: .82 pid 2792301, .73 pid 1175671.

### 5.2 Immunity to Arm 4's dataloader-offset defect — VERIFIED, not assumed

Arm 4's defect (STATUS.json `arm4_dataloader_offset`): its truncated step220000
was redone by resuming from step215000, and the checkpoint stores `epoch` but
**not** the within-epoch batch offset, so its last 5k steps replayed the epoch
opening. Original-vs-redo loss correlation at global steps 215020–220000 =
**−0.0667**. Its step220000 is therefore not a matched-20k-exposure endpoint.

Verified from the zwfy6 logs that the new runs do not share this:

* `grep -c "resume] loading ckpt"` = **1** for both seed 43 and seed 44 — a
  single resume, the intended step200000 load. No mid-run restart has occurred.
* `[resume] sampler.set_epoch(1)` appears exactly once in each; identical to
  Arm 3 and Arm 6. All four runs therefore start from **epoch-1 batch 0**, and
  differ only in the shuffle permutation. That is precisely the intended
  one-variable manipulation.
* `grep -c "RNG restore skipped"` = **0** in both.
* Both are running straight through 200000 → 220000 in a single process
  (wrapper pids .82/2686795 and .73/1072179, live since launch).

**Conditional on no restart, the new runs are immune.** The immunity is a
property of "no resume", so it is void the moment a restart happens — hence 5.3.

### 5.3 ⚠️ Live risk: the running driver carries the v1 truncation race

`scripts/_run_a03_dataorder_repl.sh` — the script running **right now** for both
seeds — stops the trainer with the **v1** guard:

```
if [ -f "$OUT/step220000.pt" ]; then
  kill -TERM "$TRAIN_PID"; sleep 20; ... kill -9 ...
```

This is the exact bare-`[ -f ]` race that truncated Arm 4's step220000.pt at
5,956,287,104 B of 12,181,311,650 B (49%): `torch.save` creates the file
immediately and then streams ~12.18 GB, so `-f` is true from the first byte. It
was fixed in `_run_a03_arm4_peaklr.sh` and `_run_a03_arm6_lowerband.sh` (v3
settled-size guard) but **never back-ported to the data-order script**.

Measured save window on this filesystem, from Arm 6's log: the step220000 line is
`16:20:04` and `saved …step220000.pt` is `16:20:17` — **13 s**. The v1 watcher
polls every 60 s, so it fires inside that window with probability roughly 13/60 ≈
**1 in 5 per run**.

The runs are **not** being patched: bash reads a running script incrementally, so
editing it mid-flight is itself a corruption risk, and the two runs are 9 h from
completion. Instead the new watcher **detects** the outcome and refuses to score
a truncated checkpoint, writing `TRUNCATED_step220000.ALARM` into the run dir
after 10 consecutive size refusals.

**Pre-registered remedy if a checkpoint is truncated:** re-run the **full 20k
from step200000** with the v3 settled-size stop guard. **Do NOT resume from
step215000** — that reproduces Arm 4's defect exactly and voids the
matched-exposure premise. A seed whose ckpt is truncated and not re-run simply
does not land (§4, `INCONCLUSIVE_INSUFFICIENT_N`).

## 6. Analysis is fixed too

Cells are produced only by
`code/recompute_cpt_trajectory_paired.py`, arms `dataorder_seed43` /
`dataorder_seed44` / `dataorder_seed45`, each holding **step220000 only**.
Protocol constants are untouched (`N_BOOT=5000`, `SEED=42`, CI95 percentile,
`BASE = A03_1B_keep7_step200k`). Verified by regression: re-running the extended
script reproduces the Arm 3 / Arm 4 / Arm 6 subset **byte-identically** against
the canonical wzc1 evidence JSON (0 diffs across all 12 arm-step cells).

Two integrity rules, both load-bearing:
* result dir **absent** → `{"pending": …}`, no number invented;
* result dir **present but < 8 shards** → hard `SystemExit` for closedbook
  **and** MMLU. (MMLU previously returned `None` on a short shard set, which made
  a 5/8 MMLU set indistinguishable from "not evaluated" and silently dropped the
  cell. Now it fails loudly.)

Only step220000 is evaluated. The 5k intermediates are deliberately **not**
scored: they would add nine arm-axis cells whose only use is post-hoc re-reading
of an oscillation already characterised in `ARM6_FINAL_VERDICT.md`
(median swing / bootstrap half-width = 2.4×, worst 10.0×).
