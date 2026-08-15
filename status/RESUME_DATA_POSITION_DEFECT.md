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
