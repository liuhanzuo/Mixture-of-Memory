# Pre-registered reading (b) is firing — MAIN, 2026-08-13 11:58 GMT+8

Written **while the scan is still running**, from the first two completed checkpoints, so it
is a pre-result note on the *interpretation*, not a post-hoc reframe. The canonical numbers
supersede everything here.

## The dispatch pre-registered two readings

- **(a)** accept only at step25000, earlier all reject → **boundary located** in 20000–25000;
  A04's first reportable discrimination point.
- **(b)** earlier steps also accept, possibly with a *larger* margin → step25000's accept is
  **not a convergence product**; it may be a non-monotone CPT-drift side effect, which
  **weakens** reading it as "recovery".

**(b) is the worse outcome for A04 and it is the one that appears to be happening.**

## What the driver log already shows

`logs/a04_full32_traj_15000.out` (08-13 11:53:01, `DRIVER END rc=0`, all 4 axes in 745 s,
mmlu shards 8/8, `n=14042 valid=14042 nan=0`):

```
step=15000  letter=0.587238  content_norm=0.464749
```

Against A04's own recorded anchors — intact vanilla `content_norm = 0.470588`, 7B MMLU split
null `0.284450`, intact residual `18.6138 pp`, `Delta = 1.8614`, NI target `16.7524 pp`:

| ckpt | content_norm | residual | margin | verdict | recovery |
|---|---|---|---|---|---|
| **step15000** | 0.464749 | 18.0299 pp | **+1.2775 pp** | **ACCEPT** | **96.86 %** |
| step25000 (archived) | — | — | +1.0495 pp | ACCEPT | 97.7 % |

**step15000's margin is LARGER than step25000's** (+1.2775 vs +1.0495), 10 000 steps earlier.

## Why this matters more than "one more accept"

1. **There may be no boundary on this trajectory.** The dispatch's scientific goal was to
   locate where accept begins, because A04 called that "what 'the gate discriminates' would
   actually mean". If 15000 and 25000 both accept and the *earlier* one accepts by more, the
   boundary is not in 20000–25000 — it is earlier still, or absent from the scanned range.
2. **It attacks the accept's interpretation, not its validity.** A margin that does not
   improve with training is hard to call *convergence to recovery*. Combined with
   `neighbour_variability_20260813` (a single 500-step step can move a margin 1.12 pp) and
   `keep14_trajectory_ni_20260813` (popqa got resolvedly WORSE over 25 500 steps), the
   emerging picture is that **margins on this harness wander non-monotonically at a scale
   comparable to Delta itself**.
3. **The zero-damage caveat still binds and now bites harder.** full32 has no structural
   injury, so every accept here is about **continued-pretraining drift**. If the accept is
   also non-monotone in training time, "97.7 % recovered" is even less supportable as a
   recovery claim than `shallow_rung_ni_discrimination_20260812` already warned.

## Required of the scan's write-up

- Report (b) **as the finding**, not as an anomaly to be explained away. It was pre-committed.
- Apply **§2.0.2** per-axis: for each accept, give the same axis's margin at the adjacent
  saved checkpoints (here they are 5 000 steps apart, so the neighbours are each other).
- State explicitly whether the mmlu_content margin is **monotone in step** across
  5000/10000/15000/20000/25000. If it is not, say so in the verdict string.
- Do **not** let my hand arithmetic into the record. I derived the residual by subtracting a
  recorded null instead of importing `build_nulls`, which is exactly the shortcut that made my
  keep14 margins ~0.5 pp off. **Canonical output only.**
