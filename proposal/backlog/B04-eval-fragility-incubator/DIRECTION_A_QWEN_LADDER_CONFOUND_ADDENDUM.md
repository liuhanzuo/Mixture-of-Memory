---
verdict_status: DOWNGRADED 2026-08-10
supersedes_claim_in: DIRECTION_A_QWEN_VERDICT.md ("DIRECTION_A_IS_OLMO_2_FAMILY_SPECIFIC")
downgraded_to: NON_MATCHED_INCONCLUSIVE
gpu_used: none (re-analysis of existing evidence JSONs)
---

# B04 addendum — the Qwen cross-family "kill" rests on a non-matched ladder

## 1. One line

The Qwen ladder **confounds damage level with training budget**: among its five
damaged rungs, Spearman(core6, heal_steps) = **+0.8944**, and **66.1%** of the
damaged-rung core6 spread is reproducible **with the damage held completely
constant** (the f12k2/14L cell alone spans core6 0.3848→0.4632 purely by training
2,000 vs 200,000 steps). A correlation computed across that ladder therefore
cannot be attributed to damage, so its collapse cannot be read as damage failing
to compress margins. **The Qwen result is downgraded from
`GENERAL_CLAIM_KILLED` to `NON_MATCHED_INCONCLUSIVE`.**

This is a downgrade of a **kill**, i.e. it removes a negative result. It does
**not** restore the general claim (see §5).

## 2. The confound, from the numbers on disk

`evidence/B04_Qwen_6rung_bs16_analysis.json` gives core6 and margins; the rung
names give the design factors. Laid out as a factor table:

| rung | layers kept | heal steps | fresh-init | core6 | median margin | frac<.005 |
|---|---:|---:|---|---:|---:|---:|
| Qwen3-8B base | 36 | — | pretrained | 0.6645 | 0.1550 | 1.78% |
| f12k2 @ 200000 | 14 | 200,000 | f12k2 | 0.4632 | 0.1339 | 2.08% |
| f12k2 @ 20000 | 14 | 20,000 | f12k2 | 0.4466 | 0.0952 | 3.58% |
| f12k2 @ 2000 | 14 | 2,000 | f12k2 | 0.3848 | 0.1074 | 3.41% |
| f12k4 @ 2000 | 16 | 2,000 | f12k4 | 0.3842 | 0.1075 | 3.55% |
| scratch14L @ 2000 | 14 | 2,000 | scratch | 0.3447 | 0.1258 | 3.09% |

Among the **five damaged rungs**:

- Spearman(core6, **heal_steps**) = **+0.8944**
- Spearman(core6, **layers_kept**) = **−0.3536**

Budget explains the core6 ordering **better than damage does**, and with the wrong
sign for damage. `layers_kept` takes only two values across the damaged rungs
(**14 at four rungs, 16 at one**), so there is essentially no damage-depth
variation to correlate against in the first place.

Held-constant check — inside the single f12k2/14L cell (identical damage, identical
fresh-init, only budget varies):

```
steps=  2,000  core6=0.3848  median_margin=0.1074
steps= 20,000  core6=0.4466  median_margin=0.0952
steps=200,000  core6=0.4632  median_margin=0.1339
core6 spread with damage FIXED      = 0.0783
core6 spread across 5 damaged rungs = 0.1185
                        ratio       = 0.661
```

So two thirds of the variation the correlation was computed over comes from a
factor that is not damage. Note also that the **non-monotonicity the verdict
identified as the story-breaker is located exactly on the budget axis**: the
step20000 rung (median 0.0952) is the outlier, and it differs from its
neighbours only in training budget. The verdict document itself listed
"heal-step count acts differently on Qwen" as a candidate mechanism; the
arithmetic above promotes that from candidate to the leading explanation, because
that axis carries 66% of the signal.

## 3. What the ladder can and cannot support

The only damage-attributable contrast is the **equal-budget slice** at 2,000
steps, where budget is held fixed and architecture/init varies:

| rung | layers | init | core6 | median margin | frac<.005 |
|---|---:|---|---:|---:|---:|
| f12k2 @ 2000 | 14 | f12k2 | 0.3848 | 0.1074 | 3.41% |
| f12k4 @ 2000 | 16 | f12k4 | 0.3842 | 0.1075 | 3.55% |
| scratch14L @ 2000 | 14 | scratch | 0.3447 | 0.1258 | 3.09% |

This slice has **n=3**, whose exact-permutation two-sided minimum p is
**2/6 = 0.3333**. It cannot reach significance at any conventional alpha *no
matter what the data say* — so it can neither confirm nor kill. (Its observed
Spearman(core6, median_margin) is −1.0, i.e. pointing the *opposite* way to the
OLMo finding, but at n=3 that is not evidence, and it is reported here only so
that nobody later mistakes it for a positive result. It also mixes two
architectural changes at once, 14→16 layers and f12k2→f12k4/scratch init, so even
with n it would not isolate damage.)

## 4. The OLMo ladder is also budget-varying — but not collinearly enough to void it

Honesty requires stating that the OLMo ladder is not perfectly matched either.
From `evidence/B04_6rung_bs16_analysis.json` plus the rung labels, among its five
damaged rungs:

- Spearman(core6, **heal_steps**) = **+0.8721** (steps ∈ {83,500; 121,000; 124,000; 200,000})
- Spearman(core6, **layers_kept**) = **+1.0000**

The difference that matters: on OLMo, **damage depth is a perfect rank predictor
of core6 and spans five distinct values (8/10/12/14/16)**, whereas budget is only
an imperfect one. On Qwen, damage depth spans **two** values and is a *worse*
predictor than budget. So the OLMo correlation has a real damage axis to ride on
and the Qwen one does not. The OLMo result is *not* claimed here to be
budget-clean — it is claimed to be **less confounded**, and its own budget
correlation of +0.8721 should be disclosed wherever the ρ=+1.00 is quoted.

## 5. What this does and does not change

**Downgraded**: the Qwen leg's verdict string `GENERAL_CLAIM_KILLED` →
`NON_MATCHED_INCONCLUSIVE`. The specific inference "damage does not compress
per-item margins in Qwen" is withdrawn, because the ladder cannot separate damage
from budget.

**NOT restored**: B04's status stays **`NARROWED_TO_OLMO_2_ONLY`**. Removing a
failed kill does not manufacture a successful replication. Before the downgrade,
cross-family generality was *refuted*; after it, cross-family generality is
*untested*. Both are equally far from *established*, and "untested" is not a
promotion argument. B04 remains backlog, and the `hold_in_backlog` novelty verdict
of 2026-08-09 is unaffected.

**Resurrection conditions are unchanged in substance but sharpened**: the existing
condition "a second confirming family (NOT Qwen; that has already failed)" is
amended. Qwen has **not** been fairly tested, so Qwen is **re-admitted** as a
candidate family — but only via a matched ladder (§6). The other condition, an
a-priori mechanism-level hypothesis, still stands and is untouched.

## 6. What a matched ladder would require

For a damage↔fragility correlation to be damage-attributable, the ladder needs:

1. **Budget held constant across rungs.** Every damaged rung trained for the
   *same* number of steps on the *same* data order — ideally each at its own
   canonical apex, or all at one fixed step count. The design must be able to
   state Spearman(core6, heal_steps) ≈ 0 across damaged rungs, and that number
   should be reported alongside the headline ρ.
2. **≥4 distinct damage levels**, so `layers_kept` has real rank variation. Two
   values (14/16) cannot carry a rank correlation. OLMo's 8/10/12/14/16 is the
   shape to match.
3. **One damage axis at a time.** Do not vary fresh-block width (f12k2 vs f12k4)
   and initialisation (inherited vs scratch) inside the same ladder used for the
   depth correlation; those belong in separate controls.
4. **n ≥ 6 rungs** to keep the exact-permutation two-sided floor at 2/720 =
   0.0028, matching the OLMo leg's inferential resolution. n=5 gives 2/120 =
   0.0167; n=3 gives 0.3333 and is useless.
5. **Identical scoring harness and identical item set** across rungs (the existing
   runs already satisfy this: all six rungs report n=17,195 with per-item
   `norm_scores` asserted present).
6. **Pre-registration of the primary metric before the rungs are scored** —
   `frac(margin<0.005)` was already designated PRIMARY on the OLMo leg, so reuse
   it rather than choosing post hoc among the four thresholds.

Cost note: requirement 1 is the expensive one, because it means re-healing Qwen
rungs to a common step count rather than reusing whatever checkpoints exist. This
addendum does **not** authorise that spend; it records what the spend would have
to buy. B04 stays in backlog either way, so there is no reason to run it before a
mechanism hypothesis exists (resurrection condition 2).

## 7. Threshold and judgement disclosure

No new threshold was invented. The two numbers used as evidence —
Spearman(core6, heal_steps) = +0.8944 and the 0.661 within-cell spread ratio — are
descriptive statistics of the existing design, not tests against a cutoff, and no
"collinearity is too high above X" line was drawn. The **judgement** being made is
that a correlation whose ordering is better explained by a nuisance factor than by
the factor of interest cannot license a claim about the factor of interest. That
judgement was made **after** seeing the data (2026-08-10), which is disclosed
here because it is the honest ordering: the confound was found by re-reading an
already-published verdict, not anticipated in the original gate. The gate design
should have required §6's matching conditions up front, and did not.

## 8. Provenance

- `proposal/backlog/B04-eval-fragility-incubator/evidence/B04_Qwen_6rung_bs16_analysis.json`
  (core6 + fragility_stats for all 6 Qwen rungs, n=17,195 each)
- `proposal/backlog/B04-eval-fragility-incubator/evidence/B04_6rung_bs16_analysis.json`
  (the OLMo comparison, n=17,195 each)
- `proposal/backlog/B04-eval-fragility-incubator/DIRECTION_A_QWEN_VERDICT.md`
  (the verdict being downgraded; retained in full, banner added at its head)
- Design factors (layers kept, heal steps, fresh-init) read from the rung labels
  recorded in both JSONs and in the verdict's provenance section.
- All statistics in this file recomputed this session from the two JSONs. No GPU,
  no re-scoring, no checkpoint touched.
