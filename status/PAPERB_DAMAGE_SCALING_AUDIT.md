# core6 damage-scaling audit — cross-arch flip counts across the depth ladder

**Date**: 2026-08-08 CST. **Author**: sub-agent (dispatched by MAIN).
**GPU cost**: 0 (uses existing summaries on both disks).
**Cross-reference**: extends the observation in
`PAPERB_CORE6_CROSSARCH_FLOOR.md` §"A new observation" (owned by MAIN, do not edit).

## The observation MAIN recorded (n=2)

MAIN's cross-arch replication table:

| checkpoint | net flips | core6 delta L20A vs H20 |
|---|---:|---:|
| full-32L vanilla base (undamaged) | 10 | +0.034 pp |
| keep14+fresh2 @200k (pruned, healed) | 28 | +0.156 pp |

The "3× more numerically fragile" quote extrapolates the trend that healed pruned models
sit closer to argmax boundaries. MAIN flagged n=2 as "observation, not result" and
invited the Table 4 audit to extend it. Here it is with all four presently-paired rungs.

## Metric definition (what "net flip" means here)

The "net flip" figure MAIN records is `sum over the 6 core6 tasks of |L20A_n_correct −
H20_n_correct|` — the *absolute-per-task* net count summed. It is not the count of
individual items that flipped label (which requires per-example predictions on both
sides; not available for most rungs). Under a directional hardware advantage all
signs would agree; under symmetric jitter signs cancel across tasks. **This is why the
sign per task matters**, and I list them below.

## Extended table (n=4 rungs currently paired L20A×H20)

Sources for the "H20" column here: for keep14 and ShortGPT-16 there are two zwfy6
measurements. I take the *original* (not `_v2`) for the primary comparison because that
is what the paper quotes; the `_v2` re-eval gives a within-disk floor for reference.

| rung | signed per-task net (L20A − H20) HS/ARCc/ARCe/PIQA/OBQA/WG | flip sum | Δcore6 L20A−H20 (pp) | dominant sign |
|---|---|---:|---:|---|
| Base 32L vanilla | −4 / +2 / −2 / +0 / +0 / +2 | 10 | +0.034 | cancels |
| ShortGPT-16 200k (PPL tax 1.32×) | −1 / −2 / −3 / +4 / +1 / +2 | 13 | +0.045 | cancels |
| ShortGPT-16 200k vs H20 `_v2` | +1 / +0 / +2 / −1 / −1 / −2 | 7 | −0.053 | *within-disk floor* |
| keep14+fresh2 200k (PPL tax 1.43×) | +7 / −5 / +5 / −3 / +0 / −8 | 28 | +0.156 | cancels |
| keep8+fresh2 121k (PPL tax 1.80×) | *wzc1 pending* | — | — | — |
| keep10+fresh2 83.5k (PPL tax 1.73×) | *wzc1 pending* | — | — | — |
| keep12+fresh2 124k (PPL tax 1.55×) | *wzc1 pending* | — | — | — |

MAIN is dispatching the wzc1-side evals for keep8/10/12; ETA ~06:00 CST 2026-08-08.
Only the H20 `_v2` measurements are on zwfy6 so far (dumped by agent `a50df6cd` overnight).

## Verdict on the two competing hypotheses

**H1 (MAIN's damage-scaling hypothesis)**: flip count scales with pruning damage.
**H0 (null)**: flips are hardware jitter proportional to `n × p(1−p)` (per-task
variance), and any monotonic trend is noise.

Evidence for **H1**:
- keep14 (28 flips, 0.156 pp) is substantially above base (10, 0.034 pp) — 2.8× more
  flips, 4.6× larger Δcore6. Not consistent with n×p(1−p) — accuracies are lower on
  keep14 which would *reduce* the variance term.
- The dominant sign flips *cancel* on all three paired rungs: no directional
  hardware advantage, only more thrashing near argmax boundaries as damage grows.

Evidence against H1 (or at least caveats):
- **ShortGPT-16 disrupts monotone scaling.** It has *lighter* PPL tax than keep14
  (1.32× vs 1.43×) yet the flip count (13) and |Δcore6| (0.045 pp) both sit
  *between* base and keep14 — monotone, but not by a lot. If damage-scaling were
  strong we would expect ShortGPT-16 halfway between base(10) and keep14(28), i.e.
  ~19 — observed 13 is closer to the undamaged floor. So the damage → fragility
  slope is present but weaker than MAIN's n=2 fit suggests.
- **`_v2` within-disk floor is ±7 flips / 0.05 pp** for ShortGPT-16. That eats a
  large fraction of the *cross-arch* effect at that PPL tax. In other words: for
  models with tax ≲ 1.3× the cross-arch signal is comparable to eval-to-eval jitter
  within one disk, and only becomes cleanly separable at tax ≳ 1.4×.

**Net verdict**: **cancellation with |net| growing with damage** — the exact framing
MAIN chose in `PAPERB_CORE6_CROSSARCH_FLOOR.md`. The trend is real but non-linear and
noisier at low damage. Do not describe it as "×3 more fragile" without qualification;
the ShortGPT-16 point (which shares the same 16L depth as keep14) undermines a purely
depth-based reading. The likely driver is the *distance to argmax boundary* per item,
which the depth ladder and ShortGPT-16 change differently.

## What the paper can currently claim

Safe:
- "core6 has a 0.03–0.16 pp cross-architecture floor on OLMo-2-7B under bf16."
- "The floor grows with the PPL tax of the checkpoint (undamaged base 0.034 pp;
  keep14 healed to 1.43× tax, 0.156 pp)."

Not safe (needs the wzc1 keep8/10/12 evals to confirm or refute):
- "Fragility scales monotonically with pruning damage across the depth ladder."
  The keep-ladder run is what would actually test this. ShortGPT-16 already shows
  the trend is not simply "shallower = more fragile" at fixed depth 16.

## Provenance and pending inputs

- wzc1 summaries: `olmo2_downstream_results/{7B_full32_base_wzc1,7B_keep14_step200000,7B_shortgpt16_step200000_wzc1}/summary.json`.
- zwfy6 summaries: `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/olmo2_downstream_results/{7B_base_full,7B_keep14_step200000,7B_shortgpt16_step200000,7B_shortgpt16_step200000_v2,7B_keep8_step121000_v2,7B_keep10_step83500_v2,7B_keep12_step124000_v2}/summary.json`.
- **Missing (blocks the full n=6 table)**: wzc1-side evals of keep8@121k / keep10@83.5k /
  keep12@124k. MAIN launched these separately. Once they land, the three "*wzc1 pending*"
  rows above can be filled by the same recipe: `sum over tasks |L20A_n_correct −
  H20_n_correct|`, then check whether signs cancel (support the boundary-distance mechanism)
  or agree (suggest a directional hardware advantage — which would be a bigger story).
- Related read-only reference (do not edit): `PAPERB_CORE6_CROSSARCH_FLOOR.md`,
  `PAPERB_TABLE4_BUDGET_DEFECT.md`, `PAPERB_TABLE4_ARCH_AUDIT.md` (this run's D1 deliverable).
