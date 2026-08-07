# The SFT "damage-sensitivity" fit is confounded — pre-SFT PPL is not a damage axis

**Date**: 2026-08-08 ~06:3x CST. **Origin**: raised as a second-order caveat by agent `a6b017e7`
(keep8 post-SFT eval); MAIN judged it first-order and verified. **GPU cost**: 0.

## The prediction failed, as pre-registered

I fit a line to n=3 arms relating pre-SFT PPL to SFT-induced ΔPPL%, got r=0.998, and pre-registered
keep8's value as a falsification test. Result:

| arm | pre-PPL | predicted ΔPPL% | observed | residual |
|---|---:|---:|---:|---:|
| keep8 | 13.333 | **+14.01%** | **+10.15%** | **−3.86 pp** |

The agent's shard-level bootstrap gives ΔPPL% 95% CI **[+10.05, +10.23]** — the prediction is far
outside it, so this is not noise. Refit statistics, which I reproduced exactly:

| | slope | intercept | r | r² |
|---|---:|---:|---:|---:|
| n=3 | 1.6015 | −7.3412 | 0.9979 | 0.9959 |
| n=4 | 0.9424 | −1.5385 | 0.9070 | 0.8226 |

Adding one point cut the slope by **41%** and r² from .996 to .823. **Three points were collinear by
chance.** Monotonicity in pre-PPL survives; linearity does not.

## But the deeper problem makes even the saturating version uninterpretable

The agent flagged, and I confirmed, that **pre-SFT PPL is not a clean measure of pruning damage** —
it is damage *minus* however much healing that arm received, and the healing budgets are wildly
unequal (`status/PAPERB_TABLE4_BUDGET_DEFECT.md`):

| arm | shape | healing steps | pre-PPL |
|---|---|---:|---:|
| full32 | 32L intact | (base) | 7.398 |
| ShortGPT-16 | 16L pruned | 200,000 | 9.780 |
| keep14 | 16L pruned | 200,000 | 10.561 |
| keep12 | 14L pruned | 124,000 | 11.443 |
| keep10 | 12L pruned | **83,500** | 12.816 |
| keep8 | 10L pruned | 121,000 | 13.333 |

The decisive comparison: **keep8 is structurally shallower than keep10 (10L vs 12L) yet received 45%
MORE healing (121k vs 83.5k steps).** Their pre-PPLs (13.333, 12.816) therefore order by *neither*
depth nor budget cleanly — they order by the two effects tangled together.

So a regression of ΔPPL% on pre-SFT PPL cannot distinguish:
- "more structurally damaged models respond more to SFT", from
- "less-healed models respond more to SFT", from
- "models further from convergence have more headroom to move in any direction".

**This is not fixable by adding rungs.** keep10 and keep12 will land shortly and take it to n=6, but
every additional point carries the same tangle. n=6 with a confounded x-axis is not better evidence
than n=4 with a confounded x-axis.

## What is actually supportable

1. **SFT costs held-out PPL on every arm measured** — +4.46% (full32), +8.51% (ShortGPT-16),
   +9.43% (keep14), +10.15% (keep8). Monotone in pre-PPL, and each individual number is solid
   (tight bootstrap CIs, single node, no NaN, byte-identical recipe).
2. **The intact base is the smallest mover (+4.46%) and all four pruned arms move roughly 2×.** That
   contrast is between-groups (intact vs pruned), does not require the confounded within-pruned
   ordering, and is the honest headline.
3. **Do NOT publish a slope, an r², or a "scales with damage" claim.** Report the four numbers and the
   intact-vs-pruned contrast. If a mechanism claim is wanted, it needs a design where damage varies
   and budget is held fixed.

## The clean experiment, if this direction is worth pursuing

Hold healing budget **constant** and vary only depth: eval SFT-response at a common step (e.g. 83,500,
the largest budget all four pruned rungs actually reached) for keep8/keep10/keep12/keep14. Every arm
has checkpoints at or near that step. Then pre-PPL differences are attributable to depth alone, and a
slope becomes interpretable. Cost: 4 SFT arms (~40 min each) + 4 eval batteries (~16 min each) on the
common-step checkpoints — under 4 GPU-hours, cheaper than what was already spent on the confounded
version.

## Secondary findings from the same eval worth keeping

Both from the keep8 battery, and both the kind of thing that changes an interpretation:

- **`arc_easy` alone supplies −0.43 pp of the −0.90 pp core6 drop** (p=3.6e−12 two-sided); the other
  five tasks are individually non-significant. So "core6 declines after SFT" is largely one task.
- **PopQA exact-match falls (−0.88 pp, p=1.2e−12) while `contains` and F1 stay flat** (CIs straddle
  zero) — the model still surfaces the right string but stops exact-matching, i.e. a
  formatting/verbosity shift rather than knowledge loss. TriviaQA degrades on `contains` and F1 too
  (−4.30 pp EM, p=6.8e−106), so there the loss is real. **These two must not be aggregated into one
  "closed-book QA declines" sentence.**

## Provenance

- keep8: `zwfy6:olmo2_ppl_results/7B_keep8_step121000_v2/` (pre) → `7B_p24_sft_keep8fresh2_final/` (post)
- agent report `status/PAPERB_P24_SFT_KEEP8_EVAL.md`, commits `fd1633c`, `2959b12`
- healing budgets: `status/PAPERB_TABLE4_BUDGET_DEFECT.md` (commit `c037cba`)
- prior (now-retracted) framing: `status/PAPERB_DAMAGE_SCALING_AUDIT.md`, `status/PAPERB_WITHIN_DISK_FLOOR.md`
