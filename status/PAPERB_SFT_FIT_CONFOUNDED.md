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

---

# UPDATE (MAIN, ~06:5x CST): keep10 lands — **monotonicity is dead too**, and the agent's reading of it is backwards

keep10 post-SFT eval finished (agent `a1a428bf`, 19 min, 5/5 shard assertions passed).
**Observed ΔPPL% = +8.63%** against a predicted +13.18% — a **−4.55 pp** miss.

## Sorted by the claimed "damage" axis, the sequence is NOT monotone

| arm | pre-SFT PPL | ΔPPL% | |
|---|---:|---:|---|
| full32 | 7.398 | +4.46 | |
| ShortGPT-16 | 9.780 | +8.51 | ok |
| keep14 | 10.561 | +9.43 | ok |
| keep12 | 11.443 | *pending* | |
| **keep10** | **12.816** | **+8.63** | **← DROPS from +9.43** |
| keep8 | 13.333 | +10.15 | ok |

The keep8 result (`PAPERB_SFT_FIT_CONFOUNDED.md`, above) killed *linearity* but left the note
"monotonicity in pre-PPL survives". **It no longer does.** keep10 sits at higher pre-PPL than keep14
yet responds *less* to SFT (+8.63 vs +9.43). So the surviving claim from the keep8 analysis —
"monotone but sub-linear/saturating" — is now also unsupported.

## The agent's directional claim is inverted

Its report states: *"Both misses same sign, magnitude grows for the more-damaged arm."* The first half
is right; the second is **backwards**:

| arm | pre-PPL | residual vs prediction |
|---|---:|---:|
| keep8 | 13.333 (**more** damaged) | **−3.86 pp** |
| keep10 | 12.816 (**less** damaged) | **−4.55 pp** |

The **less**-damaged arm has the **larger** miss. Its conclusion ("sub-linear/saturating") happens to
be right for other reasons, but the stated evidence for it does not hold, and the conclusion is
superseded anyway by the monotonicity violation above.

## Where this leaves the SFT result — three surviving statements, all between-groups

1. **SFT costs held-out PPL on every arm measured**: +4.46 (full32) / +8.51 (ShortGPT-16) / +9.43
   (keep14) / **+8.63 (keep10)** / +10.15 (keep8). Each individual number is solid — tight bootstrap
   CIs, one node per arm, byte-identical recipe, no NaN.
2. **The intact base is the outlier at +4.46%; all four pruned arms cluster in +8.5 to +10.2%.**
   Spread within the pruned group is 1.6 pp, i.e. the pruned arms are roughly interchangeable on this
   axis. The clean statement is a **two-group contrast (intact ~4.5% vs pruned ~9±1%)**, with **no
   ordering asserted inside the pruned group.**
3. **The "SFT damages memorised facts" fingerprint replicates across arms** (this part is robust and
   is the more interesting result — see below).

## The replicated fingerprint, which is the real finding here

keep8 and keep10 independently show the same pattern, both with large effects and tiny p-values:

| axis | keep8 Δ | keep10 Δ | direction |
|---|---:|---:|---|
| TriviaQA EM | −4.30 pp (p=6.8e−106) | **−5.19 pp** (p=1.1e−146) | large loss |
| PopQA EM | −0.88 pp (p=1.2e−12) | **−1.53 pp** (p=1.4e−41) | loss |
| MMLU letter | −0.60 pp (p=0.162) | **−1.82 pp** (p=5.6e−4) | loss |
| MMLU content-norm | −0.62 pp (p=0.005) | −0.43 pp (p=0.053) | small loss / marginal |
| core6 avg | −0.90 pp | **+0.03 pp** (wash) | inconsistent |

(all p two-sided.) **General instruction SFT degrades closed-book factual recall substantially while
leaving multiple-choice reasoning roughly intact.** That answers P2.4's actual question — "can general
SFT repair the pruning damage?" — with a clear **no, and it makes factual recall worse.** It is a
between-arm *replication*, needs no fitted slope, and no confounded axis.

⚠️ One caveat on core6: keep8 showed −0.90 pp of which `arc_easy` alone supplied −0.43 pp, and keep10
shows core6 **+0.03 pp (a wash)** while its `arc_easy` fell −3.45 pp (p=2.9e−9). So `arc_easy` moves a
lot in both arms while the core6 aggregate moves inconsistently — **report `arc_easy` separately and do
not lean on the core6 aggregate for this claim.**

## Bottom line for the writeup

Report the five ΔPPL% numbers as a table with the intact-vs-pruned contrast. **Assert no ordering
within the pruned group, no slope, no r², no saturation curve.** Lead the P2.4 section with the
factual-recall degradation, which replicates cleanly on two arms and does not depend on the confounded
axis. keep12 is still pending and will not change any of this — it is one more point on an axis that
cannot support a fit.
