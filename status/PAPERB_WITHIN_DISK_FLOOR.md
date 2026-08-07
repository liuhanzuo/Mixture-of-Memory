# ⚠️ The cross-architecture "effect" is mostly harness nondeterminism — within-disk floor ≈ cross-arch effect

**Date**: 2026-08-08 ~06:0x CST. **Found by**: MAIN, computing the within-disk control that had been
sitting on disk unused. **GPU cost**: 0.

## The finding

keep10 is the first rung where **three** measurements of the *same weights* exist:

| measurement | disk / arch | core6 | source |
|---|---|---:|---|
| L20A | wzc1 | `.53217` | `wzc1:7B_keep10_step83500_wzc1` |
| H20 (Table 4 source) | zwfy6 | `.53029` | `zwfy6:7B_keep10_step83500` |
| H20 (v2 re-run, **same disk, same arch**) | zwfy6 | `.52999` | `zwfy6:7B_keep10_step83500_v2` |

Per-task correct counts:

| task | L20A | H20-T4 | H20-v2 | cross-**arch** | **within-disk** |
|---|---:|---:|---:|---:|---:|
| hellaswag | 5490 | 5491 | 5490 | −1 | −1 |
| arc_challenge | 430 | 429 | 426 | +1 | **−3** |
| arc_easy | 1534 | 1531 | 1540 | +3 | **+9** |
| piqa | 1336 | 1335 | 1334 | +1 | −1 |
| openbookqa | 178 | 178 | 176 | 0 | **−2** |
| winogrande | 698 | 687 | 689 | **+11** | +2 |
| **Σ\|net\|** | | | | **17** | **18** |

- cross-**architecture** delta: **+0.188 pp**, Σ|net flips| = **17**
- **within-disk, within-architecture** delta: **−0.030 pp**, Σ|net flips| = **18**

**The within-disk flip count (18) is slightly LARGER than the cross-architecture flip count (17).**
Two runs of the same harness on the same GPU architecture, same disk, same bit-identical weights,
disagree on 18 items — as much as swapping cc9.0 for cc10.0 does.

## What this invalidates

Earlier tonight I wrote up cross-arch flip counts as if they measured a *hardware* effect, and
proposed that the count "scales with pruning damage" (`PAPERB_CORE6_CROSSARCH_FLOOR.md`, n=2 → n=4).
**That framing does not survive this control.** The right decomposition is:

- There is a **harness nondeterminism floor** of order ~15-20 flips / ~0.03-0.05 pp on core6, present
  even with architecture and disk held fixed. Likely sources: nondeterministic reduction order in
  batched MC scoring, non-fixed shard→item assignment, or unseeded tie-breaking on near-equal option
  likelihoods.
- The cross-architecture delta is **of the same order** as that floor, not clearly above it.
- Therefore the honest statement is: **core6 is reproducible only to ~±0.2 pp under *any* re-run**, and
  attributing that specifically to GPU architecture is not supported by the data I have.

The single genuinely arch-suspicious signal is **winogrande +11** (cross-arch) against **+2**
(within-disk) — one task, one rung. Not enough for a claim.

## Consequences

1. **The damage-scaling hypothesis is not currently testable with these numbers.** The base(10) /
   shortgpt16(13) / keep10(17) / keep14(28) progression I was fitting is comparable in magnitude to a
   floor I had not measured. The keep14 value (28) is the only one clearly above ~18, so at most there
   is a hint at the most-damaged end. `status/PAPERB_DAMAGE_SCALING_AUDIT.md` must be read with this
   caveat; **do not put damage-scaling in a paper on this evidence.**
2. **Paper B's actionable requirement is unchanged and if anything stronger**: core6 must not be
   reported to 4 decimals, and Table 4's mixed-architecture provenance
   (`status/PAPERB_TABLE4_ARCH_AUDIT.md`) still needs fixing — not because architecture is a large
   effect, but because **the table's rows are not reproducible to the precision printed**, for any
   reason. Adjacent ladder rungs differ 2.7-3.7 pp, so ordering is safe.
3. **PPL remains genuinely robust**: keep10 L20A `12.8158` vs H20 `12.816` — identical to 4 s.f.
   Summed NLL averages the jitter instead of thresholding it through an argmax. So PPL-based claims
   are unaffected by all of this.
4. **The P2.4 SFT deltas are unaffected**, because those are multi-point-percent effects
   (+4.5% to +9.4% PPL) measured within a single node, far above any of these floors.

## What would actually establish an architecture effect

A proper variance decomposition, not more rungs:
- **≥3 repeat runs per (ckpt, architecture)** cell, same node, to estimate the within-cell variance.
- Then test whether the between-architecture variance exceeds it (a simple nested ANOVA / ICC).
- Fix the harness nondeterminism first if possible: seed the shard assignment, force deterministic
  reductions, and check whether the within-disk floor collapses to 0. If it does, the cross-arch
  number becomes interpretable; if it doesn't, that's the real story and it is a harness bug worth
  reporting.

Until then, treat every core6 difference below ~0.2 pp as unresolvable.

## Provenance

- `wzc1:olmo2_downstream_results/7B_keep10_step83500_wzc1/summary.json`
- `zwfy6:olmo2_downstream_results/7B_keep10_step83500/summary.json` (Table 4 source)
- `zwfy6:olmo2_downstream_results/7B_keep10_step83500_v2/summary.json` (same-disk re-run)
- ckpt md5 `8bf07fa0d08ddfdf66bd80fbc6721b33` verified identical on both disks before eval
- Related: `PAPERB_CORE6_CROSSARCH_FLOOR.md`, `PAPERB_DAMAGE_SCALING_AUDIT.md`, `PAPERB_TABLE4_ARCH_AUDIT.md`
