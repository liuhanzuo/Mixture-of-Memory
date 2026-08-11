---
scope: A03 seed 45 — operational handoff after the A03 directory was archived. What to run when step220000 lands, and how to recompute σ_run at n=3.
date: 2026-08-11 21:45 GMT+8
status: HANDOFF (written while seed 45 was still training at step ~217,220 on `.82`)
prereg: DATAORDER_PREREG.md (`a25d780`) — seed 45 is inside the pre-registered set {43,44,45}
predeclaration: SEED45_PREDECLARATION.md — scope and analysis rules locked before launch
---

# Why this doc exists

A03 was decided **ARCHIVE** on 2026-08-11 and the directory was physically moved
from `proposal/active/` to `proposal/archive/`. **Seed 45 is not voided by that
move.** `ARM_SET_DECISION.md` §4 and `A04/STAGE_B_DECISION.md`'s addendum both
rest on the pooled run-to-run spread; seed 45 adds the third draw to the keep7-20k
family, taking that family from **df = 2 to df = 3** and the pooled estimate from
**df = 4 to df = 5**. The interval it tightens is precisely the one that justifies
(a) archiving A03 and (b) the MDE threshold A04 must clear before Pilot Two.

So: **do not kill it, do not change its `output_dir`, do not change its result
namespace.**

# ★ THE OPERATIONAL PROBLEM — the eval will NOT fire by itself

Measured read-only on `.82` at 2026-08-11 21:30 GMT+8:

| check | result |
|---|---|
| trainer running | ✅ 24 ranks of `train_olmo2_arch_probe2.py ... --seed 45`, PPID `3456324` = `bash scripts/_run_a03_dataorder_repl.sh` |
| progress | step ~217,220 / 220,000 (`logs/a03_dataorder_seed45.log`) |
| ckpts on disk | `step205000.pt`, `step210000.pt`, `step215000.pt`, each 12,181,311,650 B |
| **eval watcher process** | ❌ **NONE.** `pgrep -af watcher` returns nothing |
| **watcher log** | ❌ `logs/a03_dataorder_seed45_eval_progress.log` **does not exist** |

Seeds 43 and 44 were evaluated by a `a03_dataorder_trajectory_watcher.sh` instance
per node. **No such instance was ever started for seed 45.** `_run_a03_dataorder_repl.sh`
does not spawn one — its own `while` loop is only the *trainer stop* watcher (it
kills training once `step220000.pt` appears) and its last line merely *prints*
"next: eval step220000...". Therefore:

> **When step220000 lands, training will stop and then NOTHING will happen.**
> Someone must launch the eval by hand.

At the 2.02 s/step median observed on seeds 43/44, step 220,000 was due roughly
**95 minutes** after this doc was written (~23:20 GMT+8 on 2026-08-11).

# What to run when step220000 lands

Requires **8 free GPUs on `.82`** (the ext driver refuses if >8000 MiB is held; it
checks per seed, because the trainer for this seed may still be shutting down).
`.82` is a zwfy6 node; run everything from the zwfy6 root.

```bash
# on .82  (sshpass -f configs/password_h20_82250.txt ssh ... root@28.82.250.82)
W=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd $W

# 0) confirm the ckpt is complete, NOT truncated. This is not optional:
#    _run_a03_dataorder_repl.sh ships the *v1* bare-[-f] trainer stop watcher
#    (kill -TERM; sleep 20; kill -9), which is the exact race that truncated
#    Arm 4's step220000.pt to 49%. Siblings are all 12,181,311,650 B.
stat -c '%s %n' outputs/olmo2_probe2_1B_keep7f2_dolmino_dataorder_seed45/step2[01]*.pt
#    step220000.pt must be within 64 KiB of 12181311650. If it is NOT:
#    STOP. The pre-registered remedy (DATAORDER_PREREG.md) is to re-run the full
#    20k from step200000 with the v3 settled-size stop guard.
#    NEVER resume from step215000 -- that reproduces Arm 4's dataloader-offset
#    defect (original-vs-redo loss r = -0.0667) and voids the matched-20k premise.

# 1) score all four axes (8-way sharded, ~8 GPUs, writes 3 result dirs)
SEEDS=45 bash $W/proposal/archive/A03-parametric-vs-external-memory/code/a03_dataorder_ext_driver.sh
```

The ext driver is idempotent (it skips any axis whose `summary.json` already
exists) and self-shards across GPUs 0-7. It writes exactly the pre-registered
namespace — **do not rename these**:

```text
olmo2_mmlu_content_results/A03_1B_dataorder_seed45_step220000
olmo2_closedbook_results/  A03_1B_dataorder_seed45_step220000      (popqa, triviaqa)
olmo2_closedbook_results/  A03_1B_dataorder_seed45_step220000_nq   (nq_open)
```

Then recompute the paired cell. The baseline is unchanged: `A03_1B_keep7_step200k`,
n_boot=5000, seed=42, CI95 percentile.

```bash
# 2) paired diff vs the same baseline as arms 3/4/6 (CPU only)
/opt/conda/envs/torch-base/bin/python \
  $W/proposal/archive/A03-parametric-vs-external-memory/code/recompute_cpt_trajectory_paired.py \
  /tmp/a03_cpt_trajectory_paired_with_seed45.json
```

**Deposit the result under `archive/`, not `active/`** (the directory moved):

```text
proposal/archive/A03-parametric-vs-external-memory/evidence/
    a03_cpt_trajectory_paired_full_with_seed45.json
```

⚠️ **zwfy6's `proposal/` is a hand-copied tree, not a git checkout**
(`git ls-files proposal/` returns 0 there), so the `git mv` did **not** propagate.
On `.82`/`.104` the A03 code may still be at `proposal/active/...`; use whichever
path exists, and `scp -O` the resulting evidence JSON back to wzc1 to commit it.
The loaders themselves are now at `proposal/shared/code/canonical_eval_loaders.py`
on **both** disks (copied and md5-verified 2026-08-11:
`2ccce419839b17f0d8f29233b4b569ff`).

# The pre-declared verdict rule (locked before the run — do not re-argue it)

From `SEED45_PREDECLARATION.md` §"Pre-declared analysis rules":

* Primary axis is **triviaqa `em`**, 17,944 items, same baseline, same bootstrap.
* `CONFIRM_45` ⟺ CI excludes 0 **and** θ > 0 **and** θ ∈ [+0.20, +0.80] pp.
* Aggregate verdict is **mechanical**:
  * 0/3 CONFIRM → **ARTIFACT** (status quo, verdict unchanged);
  * 1/3 CONFIRM → **MIXED**, which §3.5 of the prereg **pre-declares a FAILURE of
    the claim** (the positive reading is retracted; the headline may not be the
    confirming seed).
* **Both reachable branches retract A-2.** REPLICATES is unreachable at n=3
  because seeds 43 and 44 are already NOT-CONFIRM and on disk.
* The 3 secondary axes (popqa / nq_open / mmlu_content) are barred by §3.6 from
  rescuing *or* strengthening the primary.
* **Seed 45 is the last run under this prereg. No seed 46.**

# How to recompute σ_run at n=3 (and what not to say about it)

The keep7-20k family is **sampler seeds {0, 43, 44, 45}** where seed 0 is the
original Arm 3 (`DATAORDER_VERDICT.md` line 20 labels it the sampler-seed-0 draw;
`_run_a03_dataorder_repl.sh` is config-identical to `_run_a03_arm3_cpt.sh` apart
from `--seed`, these are *resumed* runs, and the trainer has no dropout, so
`--seed`'s only material channel is `DistributedSampler(seed=args.seed)`).

Take the per-axis **arm mean** (not the paired delta) for each seed, then:

```text
s   = sample sd over the S seeds,  df = S - 1
chi-square 95% CI for sigma:
    lower = s * sqrt(df / chi2.ppf(0.975, df))
    upper = s * sqrt(df / chi2.ppf(0.025, df))
MDE for a two-arm comparison at S seeds/arm:
    MDE = t_{0.05, 2(S-1)} * sigma * sqrt(2/S)
```

Current state before seed 45 lands (from `ARM_SET_DECISION.md` §2):

| family | S | df | s (triviaqa) | χ² 95% CI for σ |
|---|---|---|---|---|
| keep7 20k (seeds 0/43/44) | 3 | 2 | 0.4132 pp | [0.215, 2.597] |
| keep12 5k (seeds 101/102/103) | 3 | 2 | 0.3023 pp | — |
| **pooled** | — | **4** | **0.3620 pp** | **[0.217, 1.040]** |

After seed 45: keep7 goes to **S=4, df=3**, and the pooled df goes to **4 → 5**.
Recompute both the pooled σ and the resulting MDE, and update
`ARM_SET_DECISION.md` §2's table plus `A04/STAGE_B_DECISION.md`'s addendum item 2.

**Reporting discipline, from the predeclaration:**

* **Always report σ_run with its d.o.f. and its χ² interval, never as a bare point
  estimate.** The original n=2 mistake was quoting 0.3231 pp as "the apparatus
  noise floor"; the fix is not a bigger n, it is always reporting the interval.
* Do **not** quote the keep7 popqa/mmlu/nq_open pairwise values
  (0.2726 / 0.0252 / 0.0000 pp) as σ — those are df=1 ranges.
* **Spread is not monotone in damage** — keep12 is *larger* than keep7 on
  popqa/mmlu_content/nq_open (mmlu 3.1×). Any seed budget premised on "less damage
  ⇒ less variance" is mis-budgeted.
* A tighter σ does **not** revive A03. Archiving was decided on *effect size vs
  spread*, and the parametric leg's measured increment is **+0.0818 pp, CI95
  [−0.945, +1.108]** — a CI containing zero, 0.26 % of the 31.10 pp deficit. Seed
  45 changes the width of the σ interval, not the location of that effect.

# Consumers to update once the number exists

1. `archive/A03-parametric-vs-external-memory/DATAORDER_VERDICT.md` — append the
   seed-45 cell and the mechanical aggregate verdict (ARTIFACT or MIXED).
2. `archive/A03-parametric-vs-external-memory/ARM_SET_DECISION.md` §2 — the σ / χ² /
   MDE table at df=5.
3. `archive/A03-parametric-vs-external-memory/STATUS.json` —
   `arm_set_decision.seed45_note` and `the_decisive_arithmetic`.
4. `active/A04-recovery-certification/STAGE_B_DECISION.md` addendum items 2 and 4,
   and `A04/STATUS.json:next_gate[4]` — the Pilot Two MDE threshold is stated at
   σ̂ = 0.362 pp (df=4); it must be restated at df=5.
5. `proposal/README.md` — the A03 archive bullet quotes the df=4 numbers.
