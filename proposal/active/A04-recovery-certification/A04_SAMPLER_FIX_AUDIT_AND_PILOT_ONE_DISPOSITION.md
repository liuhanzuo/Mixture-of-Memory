# A04 — the "blocking one-line fix" is not outstanding, and Pilot One already ran

**Date**: 2026-08-12. **GPU spent: ZERO.** `.73` was used as a CPU host only; `nvidia-smi`
reported 0 % / 0 MiB on all 8 cards at dispatch and at exit. No model was loaded, no training
launched, no eval re-run.

**Two questions, two answers.**

1. Has the `DistributedSampler(seed=)` fix landed, on **both** disks — verified by execution?
   **Yes.** Landed `ce5c298`, 2026-08-09 23:21:09 +0800. Byte-identical on wzc1 and zwfy6.
2. Does A04's σ_run therefore measure the wrong thing? **No — the premise is rejected.** All six
   runs feeding σ_run launched *after* the fix reached their disk, with a positive per-run preflight
   assertion in every log. And Pilot One, whose "entire purpose" was to produce this σ_run, **already
   ran on 2026-08-11**.

Consequence: no GPU was spent, because the thing to be bought was already on the shelf, and the
budget could not have bought it anyway (§4).

---

## 0. Separating RAN from READ

| | what |
|---|---|
| **RAN** (this pass) | (a) `code/a04_sampler_seed_probe.py` — CPU execution probe of the real `torch` sampler, both ways, 6 seeds. (b) `code/a04_sigma_run_independent_recompute.py` — σ_run recomputed from **raw per-example shards** on `.73`, reading **no** verdict JSON. (c) `md5sum` / `git cat-file` / `stat` / preflight-log greps on both disks. |
| **READ** (pre-existing) | `STATUS.json` (`next_gate`, `blocked_by`, `unverified`, `power_analysis`, `cost`, `sigma_run_input_from_A03`, `pilot_one`), `STAGE_B_VERDICT.md`, `PILOT_ONE_PREREG.md` §2.2/§3, A03's `DATAORDER_PREREG.md` §2 and `DATAORDER_VERDICT.md`, `A04_SHALLOW_RUNG_NI_DISCRIMINATION_VERDICT.md`, `A04_STEP150K_PPL_CLOSES_PLATEAU_GRID.md`. |
| **NOT RUN** | Pilot One (§4), Pilot Two, any training, any eval. |

**Falsification condition, pre-registered before measuring.** The premise (σ_run is init-variance
only) would be **confirmed** if *either* (a) zwfy6's live trainer lacked `seed=`, *or* (b) any of the
six σ_run runs launched before its disk received the fix. Both are decidable from file mtimes and
preflight logs *independently of which answer they give*. Neither held.

---

## 1. The fix, per disk — READ

`next_gate` names **line 863**. That was the **pre-fix** line number; the fix added six comment
lines above the call, so the live call is at **869** on both disks. The old 863 now exists only
inside zwfy6's backup file.

| disk | line 869 | md5 of `scripts/train_olmo2_arch_probe2.py` |
|---|---|---|
| wzc1 (LOCAL, `.21`) | `sampler = DistributedSampler(ds, shuffle=True, seed=args.seed)` | `284b286f90b526e4e8ad93a68e2a3b16` |
| zwfy6 (`.73`, `.82`, `.104`) | **identical** | **`284b286f90b526e4e8ad93a68e2a3b16`** |

The call now carries a load-bearing comment against re-deletion:

```
# seed=args.seed is LOAD-BEARING -- do not delete as "redundant with
# set_seed()/torch.manual_seed() above". DistributedSampler.__iter__ builds
# its OWN generator (`g = torch.Generator(); g.manual_seed(self.seed + self.epoch)`)
# and `self.seed` defaults to 0, so the global torch RNG cannot reach it.
# Without this argument every --seed value yields a BYTE-IDENTICAL data order,
# and "seed variance" collapses to fresh-block-init variance only.
```

`ce5c298` fixed **20 sampler sites** repo-wide, not just this one; `--seed`'s help text in the
sibling trainers now documents the trap.

### 1.1 How it got to zwfy6 — by copy, not by git ⚠️

zwfy6's `HEAD` is still `2d98c5a`; `git cat-file -t ce5c298` there returns **`Not a valid object
name`**. The fixed file was **copied**, and the pre-fix version is preserved as
`scripts/train_olmo2_arch_probe2.py.PRE_CE5C298_BAK` (md5 `879541f001568ceea16528e2e5d8035f`), whose
line 863 is the old `DistributedSampler(ds, shuffle=True)`.

**Ops risk found, not previously recorded.** Because the fixed file is **untracked** on zwfy6,
`git clean -nd scripts/` there reports:

```
Would remove scripts/train_olmo2_arch_probe2.py
Would remove scripts/train_olmo2_arch_probe2.py.PRE_CE5C298_BAK
Would remove scripts/train_olmo2_arch_probe2_distill.py
Would remove scripts/train_qwen3_arch_probe2.py
```

A routine `git clean` on zwfy6 deletes the fixed trainer. **Do not run it in zwfy6 `scripts/`.**
Since git on zwfy6 cannot vouch for this file, the per-run preflight assertion (§3) is the only
defence and must be kept in every future launcher.

---

## 2. Verified by execution, not by reading — RAN

Reading shows the argument is *present*; only execution shows it *changes* something.
`code/a04_sampler_seed_probe.py` instantiates the real `torch.utils.data.distributed.DistributedSampler`
both ways over a **15,491,607**-row index (the `dolmino_now15b.npy` row count), `num_replicas=8`,
`rank=0`, `set_epoch(0)`. No GPU, no model, no data file.

**Pre-fix — `DistributedSampler(ds, shuffle=True)`** (the defect, reproduced):

| `--seed` | `sampler.seed` | first 12 indices |
|---|---|---|
| 42 | **0** | `2411780, 7066422, 2645275, 7554317, …` |
| 43 | **0** | `2411780, 7066422, 2645275, 7554317, …` |
| 44 | **0** | `2411780, 7066422, 2645275, 7554317, …` |
| 101 / 102 / 103 | **0** | *same again* |

**Post-fix — `DistributedSampler(ds, shuffle=True, seed=args.seed)`**:

| `--seed` | `sampler.seed` | first 12 indices |
|---|---|---|
| 42 | 42 | `13002021, 3956201, 10165225, …` |
| 43 | 43 | `13915771, 10807635, 680699, …` |
| 44 | 44 | `7058515, 9559998, 5699832, …` |
| 101 | 101 | `2615430, 15330020, 689108, …` |
| 102 | 102 | `10907693, 7010929, 3636818, …` |
| 103 | 103 | `12402262, 12242335, 2286225, …` |

```
PRE-FIX  distinct orders across 6 seeds: 1
POST-FIX distinct orders across 6 seeds: 6
```

**On the slice actually consumed.** A 20 000-step run at eff_bs 128 sees 2 560 000 sequences =
**16.53 %** of one epoch, so the seed selects *which subset*, not merely the order:

| pair | rank-0 slice Jaccard |
|---|---|
| post-fix seed 43 vs 44 | **0.0102** (near-disjoint) |
| pre-fix seed 43 vs 44 | **1.0000** (identical) |

This independently confirms `DATAORDER_VERDICT.md`'s corrected characterisation — **"sampler-seed /
data-subset variation"**, not "data-order only". Downstream prose should keep that wording.

---

## 3. Provenance: every σ_run input is POST-fix — READ

Each launcher writes a **positive** assertion of the live line into `logs/*_progress.log` *before*
launching. All six checked:

| run | family | launched (GMT+8) | preflight line |
|---|---|---|---|
| `dataorder_seed43` | keep7 | 2026-08-10 16:55:29 | `trainer post-ce5c298 OK: 869: … seed=args.seed` |
| `dataorder_seed44` | keep7 | 2026-08-10 16:57:29 | same |
| `dataorder_seed45` | keep7 | 2026-08-11 12:04:07 | same |
| `stageB_keep12_seed101` | keep12 | 2026-08-11 05:53:47 | same |
| `stageB_keep12_seed102` | keep12 | 2026-08-11 05:53:57 | same |
| `stageB_keep12_seed103` | keep12 | 2026-08-11 10:22:27 | same |

zwfy6's trainer mtime is **2026-08-10 16:54:16** — **73 seconds** before the first seeded run. The
fix landed on that disk immediately before the σ_run campaign began; **nothing in the campaign
predates it.**

**Independent cross-check from training dynamics** (A03 `DATAORDER_PREREG.md` §2, loss correlation
over the common window 200 020–202 880): Arm 3 vs Arm 6, both **pre-fix** at sampler seed 0,
**r = +0.99966**; seed 43 vs 44 **r = +0.0041**; seed 43 vs Arm 3 **r = −0.0101**. The phase-lock is
present in the pre-fix arms and **absent in exactly the runs used for σ_run**.

### 3.1 The scientific consequence — the premise is rejected

The worry was: *if the fix was not in place, A03's seeds 43/44/45 are init-variance only, so A04's
σ_run measures the wrong thing and every power calculation resting on it is mis-specified.*

**That is false.** All three, **and** the three `keep12` runs that actually feed K2, ran post-fix
with verified-varying data subsets. σ_run measures run-to-run variance as intended, and **no power
calculation is mis-specified on this account.**

What *was* pre-fix is A03's **Arms 3/4/6** — and that is already on the books as the
`phase_locked_defect` behind the **ARTIFACT** verdict. Those arms are not σ_run inputs. The two
families also differ in construction, which is worth stating precisely:

* **keep7** seeds 43/44/45 **resume** from `keep7fresh2_16card/step200000.pt` (`has_optimizer=True`),
  so there is **no fresh-block init draw at all** — the seed moves *only* the data subset. Pre-fix,
  these runs would have had **zero** variance of any kind.
* **keep12** seeds 101/102/103 are **fresh prunes** from the 1B base (`copied 135 tensors`, no resume
  line), so the seed moves **both** the fresh-tail init and the data subset. `set_seed(args.seed)` is
  called before model construction, so both components are genuinely seed-driven.

---

## 4. Pilot One already ran — and could not have been re-run inside the budget

### 4.1 `unverified[0]` is stale

> *"sd_run for any 1B recovery arm — no multi-seed 1B run exists on either disk. Pilot One's entire
> purpose."*

Pilot One as designed = **one arm at the chosen j, S=3 seeds, 5 000 steps**. What is on disk:
`keep12+fresh2`, seeds `{101,102,103}`, **5 000 steps each**, trained on zwfy6 on 2026-08-11 and
scored on all four axes the same day. **That is the design, executed.** `STAGE_B_VERDICT.md` is its
harvest. `power_analysis.sd_run_is_UNVERIFIED` is stale for the same reason.

### 4.2 Independent corroboration from raw shards — RAN

To avoid taking the committed verdict on trust, σ_run was recomputed from the **raw per-example
shards**, reading **no** verdict JSON → `evidence/a04_sigma_run_independent_recompute_20260812.json`
(md5 `920c710c9612d0ccfea6c1723393b460`).

| axis | seed 101 | seed 102 | seed 103 | σ_run (pp) | bound₃ (pp) | Δ (pp) | exceeds Δ? |
|---|---:|---:|---:|---:|---:|---:|:--:|
| `triviaqa` | 9.2566 | 9.6634 | 9.0727 | **0.30229** | 0.5096 | 4.0431 | no (7.93×) |
| `popqa` | 5.5793 | 4.9415 | 5.0957 | **0.33279** | 0.5610 | 1.3205 | no (2.35×) |
| `mmlu_content` | 31.6194 | 31.6978 | 31.7761 | **0.07834** | 0.1321 | 1.0239 | no (7.75×) |
| `nq_open` *(demoted)* | 2.7147 | 2.2992 | 2.4654 | **0.20914** | 0.3526 | 0.9695 | no (2.75×) |

Agreement with `evidence/pilot_one_stage_b_s3_verdict.json` is **≤ 1.9 × 10⁻¹⁶ on all four axes** —
the arithmetic is corroborated by a second, independent implementation.

Integrity was re-asserted *outside* the shared loader, because `load_cb` checks only that 8 shard
*files* exist: **shard index set exactly `{0..7}`** (a count of 8 would accept `{0,…,5,6,6}`),
exact item counts **17 944 / 14 267 / 3 610 / 14 042**, **0** duplicate `item_id`, **0** nan, MMLU read
through the nested `content_norm.correct` key (the flat-key bug is what silently voided the MMLU axis
before).

**χ² interval, as required** (df = 2; computed closed-form since scipy is absent on `.73` — at df=2
the χ² CDF is `1−exp(−x/2)`, so `ppf(p) = −2·ln(1−p)`):

| axis | σ_run point (pp) | χ² 95 % CI (pp) | bound₃ at CI upper | Δ | would exceed? |
|---|---:|---|---:|---:|:--:|
| `triviaqa` | 0.30229 | [0.1574, 1.8998] | 3.2028 | 4.0431 | no |
| `popqa` | 0.33279 | [0.1733, 2.0915] | **3.5260** | 1.3205 | **yes** |
| `mmlu_content` | 0.07834 | [0.0408, 0.4923] | 0.8302 | 1.0239 | no |

The interval is **6.1× wide multiplicatively** — df=2 is very imprecise, exactly as the standing rule
warns. **K2 does not fire** (0 of 3 decision axes at the point estimate; 1 of 3 at the pessimistic
end, and the rule needs ≥2) — **but the verdict is fragile on `popqa`** and must always be reported
with that sentence attached.

Δ was **not** substituted or retuned: the pre-registered values were read from the prereg block and
used verbatim, the fixed fraction stays **0.10**, the anchor was untouched. **Guard G2 intact.**

### 4.3 Why no GPU was spent — the arithmetic

| basis | per seed | S=2 | S=3 |
|---|---:|---:|---:|
| `cost.measured_anchors` 2.02 s/step × 5 000 | 2.81 h wall → **22.4 GPU-h** | 44.9 GPU-h | 67.3 GPU-h |
| **measured Stage-B walls, 2026-08-11** | **4.15 h wall → 33.2 GPU-h** | **66.4 GPU-h** | **99.6 GPU-h** |

Measured walls: seed 101 `05:53:54 → 10:03:06` = 4.153 h; seed 102 4.148 h; seed 103
`10:22:37 → 14:31:36` = 4.150 h. These exceed the 2.02 s/step anchor because that anchor is
steady-state step time and omits prune + transplant + checkpoint I/O — **use the measured wall, not
the anchor, when pricing a 5 000-step run.**

At measured cost the **50 GPU-h ceiling on one node admits neither S=3 (99.6) nor S=2 (66.4)**. Only
S=1 fits (33.2 GPU-h) — and S=1 has no variance at all. So:

* A **budget-respecting re-run was arithmetically impossible**, and
* it was **unnecessary**, since the S=3 result already exists at better precision.

**S=2 is not offered even as a variance measurement.** With df=1, `t = 6.314` puts the bound above Δ
on 3 of 4 axes (`power_analysis.S2_is_unusable`); it could not certify, and it would only
approximate a measurement already on disk. Presenting one would have been a weakened version of the
designed gate dressed up as the gate — the thing the budget explicitly forbids.

### 4.4 The free win, and what is still genuinely unmeasured

**Free win, 0 GPU-h:** `sigma_run_input_from_A03` already carries the **keep7** family at **S=4 /
df=3** (seeds `{0,43,44,45}`): triviaqa 0.4039 pp (χ² 95 % CI [0.229, 1.506]), popqa 0.1959,
nq_open 0.0750, mmlu_content 0.0555 pp; pooled with keep12 → **df=5**, triviaqa 0.3666 pp, CI
[0.229, 0.899]. So **two** 1B σ_run families exist, not zero.

⚠️ Pooling is **not licensed for K2**: K2's estimator is the **keep12 family's own** df=2 σ_run, and
substituting the pooled df=5 value makes nothing fire (popqa 0.740 vs 1.321). Choosing the estimator
after seeing which answer it gives is exactly what `K2_STATUS_UNCHANGED_BY_SEED45.tempting_but_NOT_LICENSED`
prohibits. **The df=2 fragility stands.**

**Still genuinely unmeasured — and it is not a variance problem:** σ_run for a 1B arm at a rung where
`NI(Δ)` can be *observed to accept*. `keep7` and `keep12` are both **constant-REJECT** (keep12: NI
rejects 4/4 axes at 27–90× σ_run; keep7: 4/4 axes at all four checkpoints, 0 accepts in 16 cells).
This is why Stage B "passed its kill gate and still failed its purpose". **More seeds on either
family cannot fix it** — the binding constraint is **rung selection**, which is what
`pilot_two_status: BLOCKED` already demands a new pre-data document for.

---

## 5. What changes, and what does not

**Changes (bookkeeping only — no verdict moves):** four stale flags are retired — `next_gate` item 2
(the "CPU CODE FIX … BLOCKING for K2"), `blocked_by.still_blocking_before_any_gate_gpu[1]`,
`unverified[0]`, and `power_analysis.sd_run_is_UNVERIFIED`. A previously unrecorded ops risk is
added (§1.1).

**Does not change:** K1 INDETERMINATE. K2 does not fire, still necessary-not-sufficient, still
fragile on `popqa` at the χ² upper bound. `keep7` and `keep12` remain constant-REJECT rungs. Pilot
Two remains **BLOCKED** pending a new pre-data document identifying a rung where NI can accept — and
requires explicit user approval regardless. Every number in `STAGE_B_VERDICT.md` stands, now with
independent corroboration.

**The remaining blockers before any gate GPU are the two non-code ones**: the `PROPOSAL.md` rewrite
to the narrowed `safe_residual_claim`, and user approval for the 1 077–4 309 GPU-h tranche. The code
blocker is gone, and was gone before this pass began.
