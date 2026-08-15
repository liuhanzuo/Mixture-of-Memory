# Faithful-resume defect: data position is NOT restored on resume

**Found by MAIN 2026-08-15 while checking whether keep10's post-resume loss rise was divergence.**
It was not divergence — it is the *recovery* from an artificially low start, and the cause is a real
defect in `scripts/train_olmo2_arch_probe2.py`.

## What the code does

```python
# line 892-895: resume restores optimizer state + global_step + epoch + RNG
epoch = int(resume_ckpt.get("epoch", 0))
...
# line 1013-1018: "data loader position"
if sampler is not None and epoch > 0:
    sampler.set_epoch(epoch)
    logger.info(f"[resume] sampler.set_epoch({epoch}) (deterministic reshuffle for this epoch)")
data_iter = iter(loader)
```

`global_step` is restored, but the **within-epoch position is not**. `iter(loader)` restarts the
epoch at batch 0. And because keep10 resumed at `epoch=0`, the `epoch > 0` guard meant
`set_epoch` was **never called at all**.

## Measured consequence (keep10, resumed at step 90000)

| window | loss | vs pre-kill |
|---|---|---|
| pre-kill @ step 90200-90300 (`.82` log, 6 lines) | **2.5549** ± 0.0174 | — |
| post-resume first 60 log-lines (90020-90800) | **2.4826** ± 0.0185 | **−0.0723** |
| post-resume last 60 log-lines (95620-96400) | **2.5605** ± 0.0171 | +0.0056 |

Block means show a step change, not a drift: 2.4835 / 2.4826 / 2.4863 / 2.4923 / **2.5371** (sd
doubles to 0.0411 — the transition) / 2.5647 / 2.5615 / 2.5601. The rise is z = 23.96 against
within-window noise, i.e. unambiguously real, and it lands **exactly on the arm's own pre-kill
level**. So the low reading is the artefact and the current value is correct.

Not an epoch wrap: epoch 1 begins at step 121028 (15,491,607 rows / eff_batch 128), and the jump is
at ~0.77 epoch. Not warm-up either: the resume banner says `warmup=150`, and the jump is ~3200
steps after resume.

## Why it matters beyond monitoring

1. **~3200 steps of duplicate data** per resume. The model re-trains on batches it already saw,
   which is a mild repeated-data contamination the run does not record anywhere.
2. **Loss curves across a resume boundary are not comparable.** Anyone reading a per-step loss
   trace will see a dip-then-rise that looks like instability and is purely bookkeeping.
3. **All three resumed arms are affected** (keep8, keep10, keep12) — every one of them resumed with
   `--resume_from`, so each has a duplicate-data window after its own restart point.

## What is NOT claimed

- No effect on the final checkpoint's validity has been demonstrated. The duplicate window is
  ~3200 of 200000 steps (1.6%) and the loss returns to trajectory.
- Whether this measurably shifts downstream eval numbers is **untested**.
- This is not the `save_every` / checkpoint-flush issue from the same day (see
  `memory/ckpt-interval-rate-is-not-compute-rate.md`); that one was a measurement artefact, this
  one is in the trainer.

## Correct fix (not applied — would require touching a running trainer)

Persist the sampler/loader position (or a batch counter) in the checkpoint and fast-forward the
iterator on resume; at minimum call `sampler.set_epoch(epoch)` unconditionally so the reshuffle is
at least deterministic. **Do not edit the script while the three arms are running.**

Relates to open task #199 ("ckpt save/resume 机制深调 + 忠实 resume 方案") — this is a concrete,
measured instance of exactly what that task was opened to investigate.

---

## 10. Follow-up: the monotone-trend concern, raised and then resolved (2026-08-15 18:25)

At heartbeat 17:45 I flagged a problem with my own reporting. The six checks to that
point were `-0.0009, +0.0006, +0.0009, +0.0017, +0.0019, +0.0021` — every magnitude
tiny, but **all six same-signed and monotonically increasing**. I had been describing
that as "the account keeps passing," which was true of the magnitudes and quietly
ignored the direction. Six same-signed increments are not six independent
confirmations, and a slow monotone climb is also what a real divergence looks like
early on.

So I pre-registered a discriminating test rather than continuing to report
confirmations: re-evaluate around check #8-10, and **if the trend continues past
~0.5 σ, treat it as a live signal** rather than noise around a fixed level.

**Check #7 settled it in the opposite direction, one round later.**

| check | diff from pre-kill 2.5549 | in σ (σ = 0.0174) |
|---|---|---|
| 1 | −0.0009 | −0.05 |
| 2 | +0.0006 | +0.03 |
| 3 | +0.0009 | +0.05 |
| 4 | +0.0017 | +0.10 |
| 5 | +0.0019 | +0.11 |
| 6 | +0.0021 | +0.12 |
| **7** | **−0.0081** | **−0.46** |

The monotone run is broken, the sign flipped, and the excursion went *below* the
pre-kill level instead of continuing up. Over all seven checks: **mean −0.00026
(−0.015 σ)**, sd of the checks 0.00334. That is scatter about a **fixed** level with
both signs represented — precisely what the replayed-data account predicts, and
something a genuine divergence cannot produce.

**Status: the loss-recovery half of this finding is now settled.** The defect itself
(unrestored data position, `epoch > 0` guard skipping `set_epoch` at `epoch=0`) is
unchanged and still unfixed; what is settled is that keep10's post-resume loss
returns to and stays on its pre-kill trajectory rather than drifting off it.

Two things this does **not** establish, unchanged from §"What is NOT claimed":
final-checkpoint validity, and whether the ~3200-step duplicate window measurably
shifts downstream eval numbers. Both remain untested.

---

## 11. Check #11 crosses my threshold's magnitude — and why that does not reopen the finding (2026-08-15 21:45)

Check #11 came in at **−0.0088 = −0.506 σ**, which exceeds the "~0.5 σ" magnitude I
pre-registered in §10. Taken literally my rule does not fire, because what I wrote was
*"if the trend continues past ~0.5 σ"* and the trend I was worried about was **upward**
while this excursion is **downward**. Dismissing it on that wording would be the wrong
move — the threshold was under-specified, and I should not get to benefit from my own
imprecision.

So I applied the test the rule was *for*, on all eleven checks:

| # | diff | σ | | # | diff | σ |
|---|---|---|---|---|---|---|
| 1 | −0.0009 | −0.052 | | 7 | −0.0081 | −0.466 |
| 2 | +0.0006 | +0.034 | | 8 | −0.0046 | −0.264 |
| 3 | +0.0009 | +0.052 | | 9 | −0.0044 | −0.253 |
| 4 | +0.0017 | +0.098 | | 10 | −0.0012 | −0.069 |
| 5 | +0.0019 | +0.109 | | 11 | **−0.0088** | **−0.506** |
| 6 | +0.0021 | +0.121 | | | | |

- **one-sample t vs the pre-kill level: t = −1.575, df = 10.** |t| > 2.228 is needed for
  p < 0.05 two-sided, so the eleven checks are **not distinguishable** from 2.5549.
- mean **−0.109 σ**, sd of checks **0.218 σ**, signs **+5/−6**.
- the negative run #7→#11 is **not monotone**: #10 (−0.069 σ) is *less* negative than
  #9 (−0.253 σ). So there is no downward trend either, just a wider spread than the
  first six checks suggested.
- max excursion over eleven checks is **0.506 σ** — a half-sigma wander in a 30-line
  window mean is unremarkable for a loss series with σ = 0.0174 per line.

**Verdict: the finding stands, and the correct restatement is about dispersion, not
drift.** My §10 summary said "scatter about a fixed level"; that remains right, but I
under-stated the width. The honest version: keep10's post-replay loss sits at the
pre-kill level with excursions up to ~0.5 σ, and no trend in either direction survives
eleven samples.

**Threshold, corrected for any future use:** the discriminating test is the
*t-statistic over all checks*, not the magnitude of the latest one. A single check
crossing 0.5 σ is expected; what would matter is |t| > 2.228, or a monotone run long
enough to be improbable. Neither has happened.

---

## §12 (2026-08-15 19:0x) — check #12 crossed the t-threshold's neighbourhood, and
## the test turned out to be **structurally invalid**. Retracting the check series.

Check #12 came in at diff **−0.0126 = −0.726 σ**, the largest excursion yet, and it
pushed the running t to **−1.969** (df = 11, crit 2.201). That is close enough to my own
§11 threshold that I stopped and asked what would happen if it crossed — and found the
test cannot support the conclusion either way.

### The defect in my own test

Every check compared a 30-line window mean against the **fixed** pre-kill reference
2.5549 ± 0.0174, measured at **step 90000**. But keep10 is *training*. The reference is a
constant while the quantity being compared to it is on a descending learning curve. So
the test accumulates bias in proportion to distance from step 90000:

| quantity | value |
|---|---|
| clean-tail fit (steps 93540-112400, n = 946) | loss = 2.6553 **−0.00098 / 1000 steps** |
| steps elapsed since the reference | 22400 |
| **bias from learning progress alone** | **−0.0222 = −1.27 σ** |
| observed diff at check #12 | −0.0116 = −0.66 σ |
| **residual (observed − expected)** | **+0.0106 = +0.61 σ** |

The drift I was tracking as a possible signal is **smaller than the bias built into the
test**. Worse, once the bias is removed the residual is *positive* — loss slightly above
the curve, i.e. the **opposite** sign to what the defect predicts. Checks 1-12 measured
"how far has training progressed since step 90000", with the defect's effect buried
inside it. **The series is retracted, not merely re-thresholded.**

This is the same error as §11 one level up: there I fixed an under-specified *threshold*
while leaving the *statistic* unexamined. A threshold cannot rescue a biased estimator.

### The test the defect actually predicts

The defect is that `iter(loader)` restarts at batch 0, so the first ~3200 steps after a
resume **re-consume data the model has already seen**. The prediction is therefore local
and specific: the **replay window** should read *low* against the model's own learning
curve, and the tail should not. Fitting the curve on the clean tail (which excludes the
replay window by construction) and evaluating the replay window against it:

| window | steps | n | mean residual | in resid-σ | t |
|---|---|---|---|---|---|
| clean tail | 93540-112400 | 946 | 0 (by construction) | — | — |
| **replay** | **90020-93520** | **176** | **−0.0778** | **−4.15 σ** | **−48.7** (df 175) |

resid σ = 0.0188. **Direction as predicted, magnitude 4.15 σ, t = −48.7.**

### Two confounds, both killed at source rather than argued away

1. **Curve convexity.** A straight line fitted to the tail and extrapolated *backwards*
   under a convex-decreasing loss curve under-predicts the earlier loss, i.e. biases the
   replay residual **positive**. Convexity therefore works *against* the finding; the
   −4.15 σ survives it rather than being produced by it.
2. **LR re-warm.** A lower LR during replay would also lower loss. Testable, because the
   trainer logs `lr` per line — not inferable, and I did not infer it:
   - lr is **monotone non-increasing across the entire post-resume span** → no re-warm.
     (Consistent with `train_olmo2_arch_probe2.py:983`, "LR resumes on cosine curve at
     ckpt step".)
   - replay window lr = **1.190e-05 … 1.240e-05**; tail lr = **9.260e-06 … 1.190e-05**.
   - the replay window holds the **highest** LR in the span. A high LR *raises* loss, so
     this confound also pushes against the finding.

### Standing corrections

- **Do not resume the fixed-reference check series.** It is a progress meter, not a
  defect probe. If it appears in any writeup, it must be cited as retracted.
- The defect's evidence is the **replay-window residual against a clean-tail fit**,
  re-derived from the log each time (the fit's intercept moves as the tail grows).
- Generalisation: *comparing a moving quantity to a fixed historical reference measures
  the motion, not the effect.* Before pre-registering a threshold, check whether the
  statistic is unbiased under the null — a threshold on a biased estimator is decoration.
- What this does **not** establish: any downstream harm. ~3200 duplicated steps out of
  110000 is 2.9% of the resume segment; the tail returns to its own curve with no level
  shift. The writeup consequence remains the one already recorded — the three arms had
  warm restarts at different progress fractions, so "all arms at 200k" is not a matched
  budget claim.

---

## §13 (2026-08-15 19:3x) — RETRACTING my own "corpus confound only half-fixed" flag.
## No OLMo-2 ladder arm ever trained on the 7,570,911-row prefix.

For several heartbeats I carried a writeup consequence worded roughly as: *"the corpus
confound is only half-fixed -- keep8/10/12 share rows=15491607 but keep14 / ShortGPT /
freeze_front remain on the 7,570,911-row prefix (3.38 vs 1.65 epochs)."* I went to close
it and **it does not survive its own evidence check.**

### What is actually on disk

| file | disk | rows | size |
|---|---|---|---|
| `data/dolmino_now15b.npy` | wzc1 | **7,570,911** | 57.8 GiB |
| `/dev/shm/dolmino_now15b_wzc1.npy` | wzc1 (shm) | **15,491,607** | 118.2 GiB |
| `/dev/shm/dolmino_now15b.npy` | zwfy6 (shm) | **15,491,607** | 118.2 GiB |

The 7.57M file is real. My error was inferring **from its existence** that some arm had
consumed it. Existence of a truncated artifact is not evidence that anything read it.

### What every arm actually trained on

Method: `grep -l 'rows='` over **all** logs on **both** disks, then list every log whose
banner value is NOT 15491607. (The banner is emitted by the trainer at load time, so it
reports the array actually opened -- not a source default.)

- **zwfy6: 38 logs at rows=15491607.** Non-canonical hits: `llama2_rank1_*` /
  `llama2_rank2_*` at rows=32 -- a different model family, 2026-04 calibration sweeps.
- **wzc1: 1 log at rows=15491607.** Non-canonical hits: Hunyuan A13B (rows=1016774),
  hyv3 probes (55419), `dolmino_contam_audit` (8000), llama2/llama3 calib (32 / 8).
  **All different model families or tiny audits.**
- The three live arms self-report canonical: keep10 `rows=15491607` (its own log banner),
  keep12 and keep8 likewise.
- **`freeze_front` trained at rows=15491607** (`olmo2_7B_keep14fresh2_freezefront.log`),
  which is the specific arm my flag named as being on the prefix. **Directly contradicted.**

**Conclusion: zero OLMo-2 ladder arms on the 7.57M prefix. The "3.38 vs 1.65 epochs"
mismatch I was going to caveat in the writeup does not exist. Retracted.**

### Residual uncertainty, stated rather than buried

I did **not** find the original `keep14fresh2` 200k *training* log on either disk in this
pass -- the keyword matches were SFT (`p24_sft_*`, rows=107740, 842 steps) and eval shards.
So keep14's corpus is **canonical-by-inference** (its sibling `freeze_front`, forked from
the same launch generation, is confirmed canonical) rather than canonical-by-banner. That is
weaker than the other four arms and is the one thing here worth one more check before the
writeup asserts uniformity. It does **not** revive the flag: the flag claimed the prefix WAS
used, and there is no positive evidence of that anywhere on either disk.

### Why I got it wrong

Same shape as §12, one level up in the stack. In §12 I built a statistic on a reference that
looked authoritative and wasn't; here I built a caveat on an artifact that looked consumed
and wasn't. Both times the missing step was the same: **I never asked what evidence would
show the thing I was asserting, and then went and looked for it.** A 57.8 GiB file with a
plausible name sitting next to a 118.2 GiB one is suggestive, not probative -- and
`/proc/<pid>/cmdline` plus the trainer's own `rows=` banner were available the whole time.

**Standing rule:** before carrying a confound into a writeup, produce the log line that
proves the bad path was taken. "The bad artifact exists" is not that line.

---

## §14 (2026-08-15 20:0x) — the §13 residual uncertainty is now CLOSED, with a
## stronger evidence class than §13 itself used.

§13 retracted the corpus-confound flag but left one gap open and said so: keep14's original
200k *training* log was not on either disk, so its corpus was **canonical-by-inference** (via
its sibling `freeze_front`) rather than canonical-by-banner. That gap is now closed by a
better source than the one §13 relied on.

### The source: `train_args` inside the checkpoint

`scripts/train_olmo2_arch_probe2.py:535` writes `"train_args": vars(args)` into every
checkpoint. This is **strictly stronger than the log banner**:

- it is written by the trainer at save time, so it cannot drift from what actually ran;
- it **survives log deletion / rotation** -- which is exactly the failure mode that left §13
  with a gap (the log was gone; the checkpoint was not);
- it records the *resolved* `--data_path`, not a source default.

`arch_meta.json` was checked first and does **not** carry it (only `base_model_path`), so the
checkpoint is the only in-band record.

### Result: four more arms, all canonical

Read with `torch.load(..., mmap=True, weights_only=False)` on wzc1:

| arm | `train_args.data_path` |
|---|---|
| `keep14fresh2` | `/dev/shm/dolmino_now15b.npy` |
| `keep14fresh2_freezefront` | `/dev/shm/dolmino_now15b.npy` |
| `keep14fresh2_fromscratch` | `/dev/shm/dolmino_now15b.npy` |
| `7B_shortgpt16` | `/dev/shm/dolmino_now15b.npy` |

That file was measured at **15,491,607 rows / 118.2 GiB** (§13, on zwfy6). None of the four
points at `data/dolmino_now15b.npy` (7,570,911 rows).

**keep14 is therefore canonical-by-checkpoint-args, not canonical-by-inference. The §13
retraction is confirmed on an independent and stronger evidence class, and every ladder arm
now has direct provenance rather than sibling inference.**

### What this changes about how to check provenance here

For any future "which corpus did arm X train on?" question the order is:
1. **`train_args` in the checkpoint** -- authoritative, log-independent;
2. the trainer's `rows=` banner in the log -- authoritative but perishable;
3. `/proc/<pid>/cmdline` -- authoritative only while the process lives;
4. filename / directory listing -- **not evidence at all** (that was the §13 error).

I reached for (2) and (3) first and only found (1) when I went looking for a way to close the
gap I had honestly flagged. **The flagged gap is what led to the better method** -- which is
the argument for writing residual uncertainty down instead of rounding it off.
