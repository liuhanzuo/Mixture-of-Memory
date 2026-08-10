---
verdict: DIRECTION_A_IS_OLMO_2_FAMILY_SPECIFIC
date: 2026-08-08
n_rungs: 6 (Qwen3-8B), matched to the six-rung OLMo-2-7B ladder
superseded_in_part: 2026-08-10
superseded_by: DIRECTION_A_QWEN_LADDER_CONFOUND_ADDENDUM.md
---

> ## ⚠️ 2026-08-10 — THE "KILL" IN THIS DOCUMENT IS DOWNGRADED TO INCONCLUSIVE
>
> The verdict below reads the Qwen rho collapse (+0.43 / -0.49, n.s.) as a **kill**
> of the general claim. It is not one. This ladder **confounds damage level with
> training budget**: among its 5 damaged rungs Spearman(core6, heal_steps) =
> **+0.8944** while Spearman(core6, layers_kept) = **-0.3536**, and **66.1%** of the
> damaged-rung core6 spread is reproducible with damage held *completely constant*
> (the f12k2/14L cell spans core6 0.3848 -> 0.4632 on budget alone, 2k vs 200k steps).
> `layers_kept` takes only TWO values across the damaged rungs (14 at four of them,
> 16 at one). A correlation over that design cannot be attributed to damage, so its
> collapse cannot be attributed to damage either.
>
> **`GENERAL_CLAIM_KILLED` -> `NON_MATCHED_INCONCLUSIVE`.** Cross-family generality
> is now **untested**, not refuted. That is NOT a promotion: B04 stays
> `NARROWED_TO_OLMO_2_ONLY`. §5 of the addendum sets out why removing a failed kill
> does not create a replication, and §6 states what a matched ladder would require.
>
> One consequence for this document specifically: §5's claim that "a Qwen family
> attempt has *already failed*, so any resurrection needs a fresh model choice" is
> **withdrawn** — Qwen is re-admitted as a candidate family, via a matched ladder only.
>
> Everything else below (the per-rung numbers, the OLMo leg, the n=6 exact-p floor
> arithmetic) is unchanged and was re-verified against the evidence JSONs on
> 2026-08-10. Read `DIRECTION_A_QWEN_LADDER_CONFOUND_ADDENDUM.md` first.

# B04 cross-family (Qwen3-8B) verdict

## 1. One line

**Direction A does NOT cross-family generalize.** On the Qwen3-8B prune-heal ladder,
Spearman(core6, median_margin) drops from OLMo's +1.0000 (p=0.0028) to **+0.4286
(p=0.42)** and Spearman(core6, frac<0.005) drops from -1.0000 (p=0.0028) to **-0.4857
(p=0.36)**. Signs point the right way but effect strength collapses to non-significant.
**B04's SURVIVING status is revoked; it becomes NARROWED_TO_OLMO_2_ONLY.**

## 2. Full comparison

| Metric | OLMo-2-7B n=6 | Qwen3-8B n=6 | shift |
|---|---:|---:|---:|
| Spearman(core6, median_margin) | **+1.0000** exact p=0.0028 | **+0.4286** p=0.42 | ρ dropped by 0.57, p ×150 |
| Spearman(core6, frac<0.001) | -0.9429 p=0.0083 | -0.4857 p=0.36 | ρ dropped by 0.46 |
| Spearman(core6, frac<0.005) [PRIMARY] | **-1.0000** exact p=0.0028 | **-0.4857** p=0.36 | ρ dropped by 0.51, p ×130 |
| Spearman(core6, frac<0.010) | -0.9429 p=0.0167 | -0.4857 p=0.36 | same |

The n=6 exact-permutation two-sided lower bound is 2/720 = 0.00278. OLMo hits that
floor twice; Qwen is 150× away from it.

## 3. Where the compression fails

Per-rung margin distributions, sorted by core6:

| Qwen rung | core6 | median_margin | frac<.005 | monotone? |
|---|---:|---:|---:|---|
| base (36L, undamaged) | 0.6645 | 0.1550 | 1.78% | (top) |
| f12k2 @ step200000 (14L, healed) | 0.4632 | 0.1339 | 2.08% | ✓ |
| f12k2 @ step20000 (14L, mid-heal) | 0.4466 | **0.0952** | 3.58% | ✗ |
| f12k2 @ step2000 (14L, min-heal) | 0.3848 | 0.1074 | 3.41% | ✗ |
| f12k4 @ step2000 (16L, wider fresh) | 0.3842 | 0.1075 | 3.55% | ✗ |
| scratch14L @ step2000 (14L, from-scratch) | 0.3447 | **0.1258** | 3.09% | ✗ |

**Non-monotonicity locations:**
* step20000 has *smaller* median (0.0952) than both step2000 (0.1074) and step200000
  (0.1339) — a healed-more model has a *tighter* margin distribution. That inverts the
  OLMo story completely.
* scratch14L has *bigger* median (0.1258) than any 14-layer f12k2 rung — a
  from-scratch 14-layer model that never inherited weights ends up with margins
  wider than the inherited-then-healed variants at the same core6 range.

Both violations are on the fragility metric, not on core6. **The rungs themselves are
still core6-ordered** — Qwen and OLMo agree that damage degrades aggregate accuracy.
They disagree on whether damage compresses per-item decision margins in the way B04's
OLMo evidence suggests.

## 4. What kills the general claim

The OLMo evidence structurally combined two claims:

  (a) heavier structural damage → lower core6 aggregate
  (b) heavier structural damage → tighter per-item acc_norm margin distribution

The Qwen evidence keeps (a) and drops (b). So (b) is not a general property of
"pruned-then-healed 7-8B causal LMs on core6" — it is a property of the OLMo-2-7B
ladder specifically. Candidate reasons, none confirmed:

* **Different fresh-layer initialisation regime.** The Qwen ladder mixes f12k2, f12k4
  and from-scratch initialisation of the fresh block; OLMo's ladder uses a single
  keep-front + fresh-init recipe across rungs. If the fresh-init strategy shapes the
  logit distribution's dispersion (which it plausibly does, via the fresh residual
  block's variance), fragility metrics will not co-vary with damage cleanly across
  strategies.
* **Heal-step count acts differently on Qwen.** OLMo's ladder has all rungs healed
  to their canonical apex; Qwen mixes step200000 vs step20000 vs step2000. A model
  that has trained *more* on the same damaged trunk can end up with a *narrower*
  logit distribution as it becomes more confident (the step20000 outlier).
* **Tokenizer / vocabulary size.** Qwen3 vocab is much larger; a 4-way MC scoring
  scheme touches a very different slice of it, and length normalisation constants
  differ. The reciprocity between fragility and gold-label distribution is not
  guaranteed to preserve across tokenizations.

Any of these could be the mechanism. The paper cannot claim without evidence.

## 5. Consequences

**Must update:**
* `proposal/backlog/B04-eval-fragility-incubator/STATUS.json`: `SURVIVING` →
  `NARROWED_TO_OLMO_2_ONLY`. Remove `promotion_pending: novelty_check_only` — the
  paper (if any) can no longer claim generality.
* `DIRECTION_A_VERDICT.md`: add this Qwen result as the family-scoping caveat that
  was previously listed as the next kill gate.

**Survives, in narrowed form:**
* The OLMo-2-7B six-rung result is unchanged and remains at maximum significance
  (both PRIMARY metrics at exact-p floor 0.0028).
* Its interpretation must be scoped: "on the OLMo-2-7B keepN + shortgpt16 ladder,
  aggregate core6 degradation is accompanied by per-item margin compression". This
  is still a legitimate observation about **this specific model family's** damage
  response, useful for anyone doing prune-heal work with OLMo-2 or writing about
  OLMo-2 evaluation.

**Does not survive:**
* Any claim that per-item margin compression is a general damage response of
  causal LMs on core6.
* B04 as a stand-alone paper. Reframe: either fold it into an OLMo-2 methods
  appendix (Paper B or the null-cal spin-out) or park it. The direction can only
  return as a general claim after a second confirming family, and a Qwen family
  attempt has *already failed*, so any resurrection needs a fresh model choice and
  a mechanism-level hypothesis, not another downstream ladder.

**Not a mistake — a live save:**
This is why the cross-family kill test existed. B04's own verdict document said
"NOT established beyond OLMo-2-7B. Cross-family replication is the next kill test."
That gate ran and killed the general claim. The self-limiting protocol worked.

## 6. Provenance

* Qwen bs16 downstream: `.21:/apdcephfs_wzc1/.../qwen3_probe2_downstream_results/qwen_{base_full36,f12k2_step200000,f12k2_step20000,f12k2_step2000,f12k4_step2000,scratch14L_step2000}_bs16/` — all six with `per_example_*.jsonl` (n_scored asserted per task) + `norm_lens/norm_scores` enrichment verified via `[ENRICH CHECK] norm_scores present = True`.
* Analyzer: `proposal/backlog/B04-eval-fragility/analyze_b04_qwen_6rung.py` (exact-permutation p, identical margin definition to the OLMo analyzer).
* Analysis JSON: `proposal/backlog/B04-eval-fragility-incubator/evidence/B04_Qwen_6rung_bs16_analysis.json`.
* Driver: `scripts/_run_b04_qwen_xfamily_21.sh`; ~17 min wall on 8× L20A.
* Commit: 36ddb1e (harness + driver + OLMo analysis), this verdict follows.
