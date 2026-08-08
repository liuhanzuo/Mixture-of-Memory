---
gate: A01 gate-3 (full-fp32 forward -- is the bf16 exact tie the causal mechanism?)
date: 2026-08-09
node: .73 (8x H20, zwfy6)
verdict: MECHANISM_FALSIFIED -- ties are a bf16 artifact AND removing them changes nothing
---

# A01 gate-3 — the fp32-vs-bf16 causal test

## 1. Verdict

**Both hypotheses gate-3 was designed to separate are wrong, and the answer is a third
thing.** `H_artifact` said the ties are a bf16 representation artifact that, once removed,
would let the argmax become input-driven and move letter accuracy. `H_real` said the damaged
model genuinely puts identical mass on the four letters. What actually happens:

* The ties **are** a pure bf16 artifact — fp32 removes **100%** of them on both arms.
* Removing them changes **18.03% of the damaged arm's letter argmax decisions**.
* And letter accuracy **does not move at all**: Δ = −0.0015, CI95 [−0.0064, +0.0033],
  exact McNemar **p = 0.570**.
* The damaged arm stays **below its own best-constant floor in fp32 too**, in fact more
  significantly so (−1.54 pp, boot p = 0.0062) than in bf16 (−1.39 pp, p = 0.0192).

So **the exact ties are not the mechanism of the letter-interface failure.** They are a
downstream symptom of a logit distribution that has already collapsed. Reshuffling 2,532
coin-flips does not recover information that is not there.

## 2. The measurement

Single process, single model instance, single item list, single batching order. The **only**
difference between arms is the forward dtype: `autocast(bf16)` vs no autocast (weights are
already fp32 masters). Everything downstream — `log_softmax(logits.float())`, the
teacher-forced sum-logprob, the argmax, the length normalisation — is byte-identical code.
The bf16 arm reproduces the archived per-example scores, which is what makes the fp32 arm a
clean contrast rather than a new harness with unknown drift. n = 14,042, 0 nan, 8/8 shards
asserted before merge.

### Arm 1: OLMo-2-7B base (intact)

| | bf16 | fp32 |
|---|---:|---:|
| letter acc | 0.6054 | 0.6054 |
| content_norm acc | 0.4706 | 0.4704 |
| **exact-tie count** | **18** | **0** |
| exact-tie rate | 0.13% | 0.00% |
| all-4 tied | 0 | 0 |
| min positive gap | 2.441e-04 | 8.225e-05 |
| gap median | 1.1185 | 1.1167 |
| letter vs floor (0.2689) | +33.65 pp | +33.65 pp |
| letter residual fraction | 0.5558 | 0.5558 |

Contrast: 96 letter argmax changes (0.68%), of which only 12 were among the 18 bf16-tied
items. **letter acc Δ = +0.0000, CI95 [−0.0011, +0.0011], McNemar p = 1.000.**

### Arm 2: OLMo-2-7B keep8@step121000 (most damaged rung, highest tie rate)

| | bf16 | fp32 |
|---|---:|---:|
| letter acc | 0.2550 | 0.2535 |
| content_norm acc | 0.3423 | 0.3427 |
| **exact-tie count** | **4303** | **0** |
| exact-tie rate | **30.64%** | 0.00% |
| **all-4 tied** | **175** | **0** |
| strict-argmax acc | 0.2578 | 0.2535 |
| min positive gap | 8.225e-05 | 3.052e-05 |
| gap median | 0.2500 | 0.1781 |
| letter vs floor (0.2689) | **−1.39 pp** (boot p 0.0192) | **−1.54 pp** (boot p 0.0062) |
| letter residual fraction | −0.0545 | −0.0607 |
| letter verdict | **BELOW floor** | **BELOW floor** |
| content vs floor (0.2845) | +5.78 pp | +5.82 pp |

Contrast: **2,532 letter argmax changes (18.03%)**, of which 2,466 were among the 4,303
bf16-tied items and 66 among the untied. content_norm argmax changed on only 195 (1.39%).
**letter acc Δ = −0.0015, CI95 [−0.0064, +0.0033], boot p = 0.5598, exact McNemar p = 0.570.**

## 3. Triangulation with gate-1 — three independent lines, one conclusion

| evidence | tie rate | effect of removing / having ties |
|---|---:|---|
| gate-3, OLMo-2 base | 0.13% → 0 | letter acc unchanged (McNemar p = 1.000) |
| gate-3, OLMo-2 keep8 | **30.64% → 0** | 18% of argmaxes flip; letter acc unchanged (p = 0.570); still below floor |
| gate-1, Llama-2-7B | **15.79%**, untouched | fully healthy: +14.11 pp above floor, letter ≈ content (p = 0.51) |

A high tie rate is compatible with a perfectly usable interface (Llama-2). Eliminating a
30% tie rate does not repair a broken one (keep8). Therefore **"bf16 exact ties cause the
letter interface to fail" is falsified from both directions.**

What the ties actually are: the damaged model's four letter logits are so tightly packed
that bf16's mantissa cannot separate them — the bf16 **gap median collapses to 0.2500** on
keep8 versus **1.1185** on base, a 4.5× compression. The tie count is a *readout* of that
compression, not its cause. This is the same phenomenon B04 measures at the per-item level
on core6 (damage compresses decision margins), arriving via a completely different route.

## 4. Consequences for A01's writeup

**Must drop:** any causal claim that the tie-breaking-by-index is what makes the letter
interface unreliable. The correct statement is weaker and cleaner:

> On structurally damaged OLMo-2 arms the four letter logits become inseparable at bf16
> precision (tie rate 0.13% → 30.64% from base to keep8, gap median 1.12 → 0.25). The ties
> are a precision artifact — full-fp32 forward removes all of them — but removing them does
> not restore measurable accuracy (Δ = −0.0015, p = 0.57), and the arm remains below its own
> best-constant floor in both precisions. The interface has lost the information, not merely
> the ability to break a tie.

**Strengthened:** the "below its own floor" finding is now precision-robust. A reviewer
cannot dismiss keep8's sub-floor letter accuracy as a bf16 artifact, because it is *more*
significant in fp32 (p = 0.0062 vs 0.0192).

**Also worth reporting:** `strict-argmax acc` on keep8 is 0.2578 in bf16 vs 0.2535 in fp32.
The bf16 number is *higher* because index-order tie-breaking happens to be slightly lucky on
this item set — a reminder that a tie-heavy interface's reported accuracy partly measures the
gold-label ordering, not the model.

## 5. Provenance

* Harness: `proposal/active/A01-null-calibration-methodology/code/a01_gate3_fp32_vs_bf16.py`
* Driver: `scripts/_a01_gate3_driver_73.sh`
* Results: `.73:/apdcephfs_zwfy6/.../results/a01_gate3/dtype_runs/{7B_base_dtype,7B_keep8_step121000_dtype}/dtype_summary.json`
* Merge logs: `logs/a01_gate3_7B_{base,keep8_step121000}_dtype_merge.log`
* Cost: base arm 21 min, keep8 arm 8 min, 8 GPUs each.

## 6. Operational note (recorded so it is not repeated)

The first driver run declared `MERGE FAILED` on a merge that had actually **succeeded** —
it checked for `summary.json` while this harness writes `dtype_summary.json`. The driver
then exited and left 8 H20s idle for 20 minutes. Fixed in
`scripts/_a01_gate3_driver_73.sh`. Lesson: a driver's success predicate must be checked
against the harness's actual output filename, not the convention used by a sibling harness.
