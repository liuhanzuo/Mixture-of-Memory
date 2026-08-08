# The within-disk "floor" was harness drift, not runtime jitter — and it changes what cross-arch measures

**Date**: 2026-08-08 ~08:2x CST. **Verified by**: MAIN, from n=3 and n=2 same-arch same-disk repeats
committed to disk. **GPU cost**: ~40 min (variance-controls dispatch, agent `afe2a215`).
**This partially reverses `PAPERB_WITHIN_DISK_FLOOR.md` and revises `PAPERB_DAMAGE_SCALING_AUDIT.md`.**

## The measurement

Within-disk same-arch same-harness re-runs of pre-SFT eval batteries. Every pair below is on zwfy6
H20, same checkpoint, same code path:

| pair | Σ\|net flips\| across core6 |
|---|---:|
| keep14 v1 (Aug 8 01:31) vs v2 (Aug 8 08:02) | **0** (byte-identical) |
| keep8 v2 vs v3 | **0** (byte-identical) |
| shortgpt16 v2 vs v3 | **0** (byte-identical) |
| shortgpt16 v1 vs v3 | 20 |
| keep10 v1 vs v2 (from earlier tonight) | 18 |

Same-arch **same-harness-version** re-runs are **byte-identical** on all six core6 tasks. The 15–20
flip "floor" I documented in `PAPERB_WITHIN_DISK_FLOOR.md` was between **v1 (older harness) and v2+
(new harness added tonight)** — a **code-version boundary**, not runtime jitter.

## What partially reverses

- The claim in `PAPERB_WITHIN_DISK_FLOOR.md` that "core6 is reproducible only to ±0.2 pp under any
  re-run" and that this ~15-20 flip floor was intrinsic bf16/harness noise: **wrong**. Same-harness
  re-runs are bit-identical.
- The framing that cross-arch effects sit *inside* the noise floor: needs revision. Some of the
  cross-arch flip counts I measured earlier were **between old-harness zwfy6 and new-harness wzc1**,
  which conflated architecture with harness version.

## What survives, and what the actual cross-arch signal is

Recomputing cross-architecture with **matched (new) harness on both sides**:

| rung | wzc1 L20A (new) vs zwfy6 (new v2/v3) | per-task flip breakdown |
|---|---:|---|
| ShortGPT-16 | **7** | `hs +1, arc_c 0, arc_e +2, piqa −1, obqa −1, wino −2` |
| keep10 | **23** | `hs 0, arc_c +4, arc_e −6, piqa +2, obqa +2, wino +9` |
| keep8 | **29** | `hs +2, arc_c −4, arc_e −7, piqa +3, obqa −1, wino +12` |

So the honest cross-architecture picture at matched harness is:

1. **The floor now is 0**, not 18. Same-harness re-runs on the same GPU are deterministic to bit
   level.
2. **Cross-architecture flip counts range 7 to 29 across three rungs**, comfortably above the true
   floor.
3. The **winogrande task dominates**: it supplies +12, +9, and −2 of the three totals — the largest
   single component in all three rungs. If there is a genuine architecture effect on core6, it is
   plausibly a winogrande-specific one (winogrande option-scoring may involve near-tie likelihoods
   that bf16 reduction order can flip), and other tasks contribute little.
4. **The rungs are ordered ShortGPT-16 (7) < keep10 (23) < keep8 (29).** Suggestive of a damage
   effect, but n=3 with substantial per-task variability. Not enough to reinstate the damage-scaling
   claim; possibly enough to say "worth checking with per-task variance decomposition."

## What actually happened between v1 and v2+

I don't yet know which specific harness change introduced the boundary. Candidates: the
`assert_8shards` guard (v1 predates it; if a v1 silently kept a stale shard from a previous run
mixed with fresh ones, that alone would produce ~20-item drift); a change in how per-item
scoring is invoked; a change in tokenization/BOS handling. **I have not diagnosed which.** Until
diagnosed, treat v1 batteries as suspect for exact reproduction and use v2/v3 wherever possible.

The keep12 partial-merge finding (`PAPERB_TABLE4_KEEP12_PARTIAL_MERGE.md`) is likely *one instance*
of the general v1-fragility mechanism, not a separate defect. Paper's Table 4 was quoting v1
numbers for every rung it named; whether any others carry the same class of silent defect below
the `n_scored`-check threshold I've applied is now the open question.

## Consequences for the paper writeup

1. **The damage-scaling story remains dead** (it died on other grounds: within-pruned monotonicity
   fails at n=5, and the pre-SFT-PPL axis mixes depth with unequal healing budgets — see
   `PAPERB_SFT_FIT_CONFOUNDED.md`). This revision does not resurrect it.
2. **Cross-arch effect is real but modest**: ~7–29 flips out of ~13–17k scored items, dominated by
   winogrande on the three pruned rungs measured. Do not lead with this.
3. **The `_v2`/`_v3` batteries are the paper-quality numbers.** Whenever a v1 and a v2 exist, use
   v2. Table 4 rewrite should re-source every rung to a v2 measurement, not a v1.
4. **Paper B needs an explicit protocol note**: "held-out evaluations use harness version X, seeded
   shard assignment, `assert_8shards` on merge." This is a paragraph in the appendix, not a table
   change.

## Provenance

- keep14 v2: `zwfy6:olmo2_downstream_results/7B_keep14_step200000_v2` (mtime Aug 8 08:02, size differs
  from v1 by 3 bytes only in JSON formatting; per-task counts byte-identical to v1)
- keep8 v3, shortgpt16 v3: `zwfy6:*_v3/` (produced by agent `afe2a215` this heartbeat)
- Prior contradicted claim: `PAPERB_WITHIN_DISK_FLOOR.md` (commit `af6d869`) — the section titled
  "the within-disk flip count (18) exceeds the cross-architecture flip count (17)" was true as
  written for keep10 v1 vs v2, but the 18-flip figure now reads as harness-version drift, not
  runtime jitter, and the "exceeds" comparison should be pulled from paper writeup.
- Related: `PAPERB_TABLE4_KEEP12_PARTIAL_MERGE.md` (a known v1-side bug consistent with this).

## Retraction accounting for tonight

Five framings now retracted or revised:
1. dLLM sampler-audit generalization to MBPP+ (killed by G1)
2. Cross-arch damage-scaling of flip count (killed by within-disk floor claim — **which itself is now revised as harness drift**, though the damage-scaling claim is not resurrected)
3. Linear ΔPPL vs pre-PPL fit (killed by keep8)
4. Monotone-saturating fallback (killed by keep10)
5. Any within-pruned SFT ordering (killed by n=5 spread within 1.6 pp)

Now plus this partial revision: **the intra-disk floor was itself an artifact of harness drift,
not a physical noise floor.** The cross-arch numbers with matched harness are still small and don't
reinstate any earlier claim; they just need to be re-cited carefully.
