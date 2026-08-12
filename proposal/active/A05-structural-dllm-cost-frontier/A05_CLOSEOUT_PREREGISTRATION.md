# A05 closeout — pre-registration for §2 (adjudicating the surviving *cost* claim)

**Written 2026-08-12, BEFORE computing any corrected cost ratio.** Committed before the
numbers exist so that the verdict cannot be fitted to whatever came out.

## What is under test

The claim the K1 agent left standing, verbatim from `A05_K1_CANVAS_SWEEP_VERDICT.md` §"Consequences":

> "at *matched quality*, Scaffold Medium is 6.2×/8.2× cheaper in forward passes."

This is a **cost** claim at matched quality. It is *not* A05's registered quality claim (that one
died at K1). I am deciding whether it is (i) publishable, (ii) needs more work, or (iii) dead.

## Falsification conditions — any TWO firing means the claim is not publishable as stated

**F1 — central-tendency robustness.** The headline ratio uses `nfe_mean`. If substituting the
**median** reverses the direction of the ratio, or reduces it below 2×, on **either** benchmark,
then "6.2×/8.2× cheaper" is a property of a distribution tail rather than of the method.
*Rationale*: `evidence/cells/he_c32.json` already shows `nfe_mean 393.7` vs `nfe_median 32.0` — a
13× mean/median gap. A ratio of means between two heavy-tailed distributions is not a cost fact
until the tail is understood. (I have read those two numbers; I have NOT yet computed the
Scaffold-side median or any ratio.)

**F2 — is "matched quality" actually matched.** DreamOn's own measured round-to-round movement is
1.9 pp (r1→r2, `PROPOSAL.md` §2 D-C). If on either benchmark the pass@1 gap exceeds that in
**DreamOn's** favour, then quality is not matched there: Scaffold is *worse and cheaper*, which is
an ordinary quality/cost trade-off available to any degraded system, not a matched-quality win.

**F3 — the AR control on the same axis.** Put Qwen2.5-Coder-7B on whichever axis carries the
headline. If AR is **cheaper than Scaffold on that same axis while also scoring higher**, the claim
is a within-diffusion-family internal point — which is exactly the description under which the
original Pareto claim was RETRACTED (`DLLM_SALVAGE_ROADMAP_20260808.md` §1.1). Then it cannot stand
as a cost contribution regardless of the DreamOn ratio.

**F4 — axis agreement.** Compute NFE **and** `tokens_fed` (invariant 2). If the two axes disagree in
direction, or by more than 2× in magnitude, then no single "X× cheaper" scalar may be quoted at all
and the claim must be restated as a per-axis statement.

**F5 — budget-cap confound.** DreamOn's tail is its own iteration cap
(`2*max_gen_len + 2*expand_budget`, ~2060 forwards); Scaffold's is `max_model_calls=512`. If >50%
of DreamOn's total NFE mass comes from <15% of items, the *mean* is largely a readout of an
arbitrary, unmatched budget knob rather than of decoding efficiency.

## Decision rule (fixed now)

| firing | disposition |
|---|---|
| 0–1 conditions | claim survives → deserves its own proposal with its own gates |
| **2 or more** | **not publishable as stated** |
| F3 fires alone | automatically caps the claim at "within-family internal point" — cannot be a standalone cost contribution even if F1/F2/F4/F5 all pass |

## Axes I will report (all of them, never NFE alone)

1. **NFE** (true counted `model.forward` calls) — mean *and* median, both arms.
2. **`tokens_fed`** — Scaffold's `cumulative_model_tokens`, DreamOn's `tokens_fed_effective`,
   AR's hook-measured `tokens_fed`. Mean and median.
3. **`attended_context_sum`** — for the two no-KV-cache diffusion arms this equals `tokens_fed`;
   for AR it is separately measured and much larger. Reported because it is the axis on which AR
   does *not* dominate Scaffold, and omitting it would be cherry-picking against AR.

## Provenance rules for this section

* Scaffold NFE/`tokens_fed`: **recomputed by me** from `runs/scaffold_medium_*/metrics.rank*.jsonl`
  using the `process` **or** `failure_process` accessor (Retraction 3's fix), not read from a table.
* DreamOn: recomputed from the K1 `runs/a05_k1/*/metrics.rank*.jsonl` per-item rows.
* AR: `outputs/ar_qwen25coder7b_base{,_greedy}/{humaneval,mbpp}/report.json` on zwfy6, which carry
  `measured_matches_analytic: true`.
* Scaffold pass@1 (.177/.354) stays **READ** — the 29 GB checkpoint is wzc1-only and is not
  re-scored here. Any conclusion that depends on re-scoring Scaffold is out of scope and will be
  labelled as such.

## Also pre-registered: what would make me reject the task's own premises

I was told (a) the archived NFE is not a forward count, (b) `mask_expansion` is inert, (c) the HE+
stitch double-indents. If any of these three turns out to be **wrong** on inspection, I report that
instead of propagating it. Specifically I will re-derive (c) myself from the raw `raw_output` rows
rather than trusting the 113/117 attribution.
