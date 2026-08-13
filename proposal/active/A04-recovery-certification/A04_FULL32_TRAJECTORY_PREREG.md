# A04 — full32_dolmino trajectory NI scan: PRE-REGISTRATION

**Written 2026-08-13, BEFORE any of the four intermediate checkpoints had been scored.**
The GPU jobs for `step5000` (.73) and `step15000` (.82) were launched at 11:36:51 and
11:40:36; this file was committed before any `summary.json` existed for them. Its purpose
is to fix the two readings in advance so that neither can be selected after seeing the
numbers.

Dispatch origin: `STATUS.json:shallow_rung_ni_discrimination_20260812
.implication_for_pilot_two.cheap_next_steps_dominate[1]`.

---

## 1. The question

`full32_dolmino@step25000` is the **only NI accept in all of A04**: `mmlu_content` margin
**+1.0495 pp** (97.7 % MMLU recovery), with `triviaqa` missing acceptance by only **1.86
bootstrap SE**. Every damaged rung ever measured is constant-REJECT at 16.5–72.4 SE.

So if A04 has an accept **boundary** anywhere, it is on this trajectory. Scoring
`step5000/10000/15000/20000` on the same four axes locates it, or shows there is none.

## 2. Pre-registered readings

**(a) Accept appears only at step25000; step5000–20000 all REJECT on `mmlu_content`.**
→ The boundary is **located in the interval (20000, 25000]**. A04 acquires its first
reportable discrimination point: the rule's output changes as a function of the amount of
continued pretraining, in the direction the rule intends.

**(b) One or more earlier checkpoints also ACCEPT — possibly with a LARGER margin.**
→ The step25000 accept is **not the product of convergence**. It would then be consistent
with non-monotone continued-pretraining drift, and using step25000's accept as evidence of
"recovery" would be **reading a fluctuation as an achievement**. This weakens A04's own
headline.

**(b) is the outcome that is worse for A04, and it will be reported exactly as readily as
(a).** A third possibility — accept at some middle step but not at its neighbours — is a
sharper form of (b) and is reported as such.

## 3. Secondary pre-registered questions

1. **`triviaqa`**: at step25000 it is 1.86 SE from accepting. Does it get *closer* at
   earlier steps (evidence against monotone approach) or *further* (evidence for it)?
2. **RATIO(0.85) vs NI**: `full32_rescore_v2_20260812` established that at step25000 RATIO
   ACCEPTS (mean ratio 0.8515, margin **+0.0015**) while NI REJECTS on 2 of 3 decision
   axes. Does that disagreement **widen, narrow, or vanish** along the trajectory? The
   RATIO margin is thin enough (+0.0015 ≈ 3.3 nq_open items) that earlier checkpoints may
   fall on the other side of ρ.

## 4. Committed in advance — what this can NEVER conclude

- **`full32` has ZERO structural damage** (`keep_front_layers=32, n_fresh_layers=0`, all 32
  pretrained layers present, nothing transplanted). Any accept here is about
  **continued-pretraining drift on the heal corpus**, *not* about recovery from structural
  injury. This is the standing caveat in
  `shallow_rung_ni_discrimination_20260812.the_load_bearing_new_finding.caveat` and it
  travels with every number produced here.
- **The four checkpoints are NOT replicates.** They are successive states of ONE
  optimisation run. Their spread is training progress plus data order — never seed
  variance. No 7B `sd_run` exists or is reconstructible (one seed per rung; the historical
  ladder's seeds are unrecorded, `--seed` postdates them).
- Therefore **no claim that any deficit is large or small "relative to seed variance"** may
  be made at 7B.
- **No K1/K2/K3 clause can fire.** Those are defined over the pre-registered 1B arm set; a
  7B trajectory cannot fire them.
- Same-arch/same-harness re-runs on this harness are **byte-identical**
  (`full32_rescore_v2_20260812.correction_to_the_jitter_premise`), so no difference measured
  here may be called "runtime noise". These are four *different* models, so bit-identity
  does not apply to them — but "noise" is equally unavailable as an explanation.

## 5. Frozen protocol (deviation ⇒ the cells are non-comparable and must not be published)

`cb_bs=32`, `mmlu_bs=16`, `add_bos=0`, `num_shards=8`, `max_new_tokens=32`,
`max_ctx_len=512`, greedy (`do_sample=False, num_beams=1`), base LM, `chat_template=False`,
`mode=pruned keep_front_layers=32 n_fresh_layers=0` (the archive's own label for this arm).

- **Anchor:** vanilla `../models/OLMo-2-1124-7B` — `olmo2_closedbook_results/base_full{,_nqopen}`
  + `olmo2_mmlu_content_results/7B_base`. Imported from `a04_shallow_rung_ni_7b.ANCHOR`,
  never redeclared. **Guard G2 forbids** using `full32_step25000` as the anchor: it scores
  below vanilla on all four axes, so substituting it would shrink every Δ *and* lower every
  target — manufacturing accepts. Δ is never substituted.
- `protocol_asserted()` runs **before** anything is scored, parses `cb_bs`/`mmlu_bs` out of
  the drivers' own echoed log lines (`summary.json:meta` records **neither** batch_size nor
  chat_template — `A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md`), and **fails closed**.
- Shard integrity: index set **exactly {0..7}**, merged n exactly
  triviaqa 17944 / popqa 14267 / nq_open 3610 / mmlu 14042, 0 duplicates, 0 nan.
- `add_bos` asserted `is False` — never `is not True`, which passes silently on `None`.

## 6. Neighbour precondition (`A04_GATE_DESIGN.md` §2.0.2)

Any accept reported here must be accompanied by **the same axis's margin at the immediately
adjacent saved checkpoints on both sides**, or a statement that none exist. On this
trajectory the spacing is 5000 steps, so the neighbours of `stepN` are `stepN±5000`;
`step5000` has no lower neighbour and `step25000` no upper one. Stated **per axis**, not
blanket — §2.0.2's measured effect was concentrated on `triviaqa`.

## 7. Bootstrap seeds

New arm indices **500–503** (form `97*arm_index + 13*axis_index`); the step25000 endpoint
keeps its **archived** offset (`arm_index=203`) so the reproduction assertion is exact.
Disjoint from every archived offset: pilot_zero {0,1}, step100k 100–102, shallow_rung
200–203, keep14 300–301, neighbour 400–408, guard offsets 700/900/1700/1900/2400.

## 8. Analysis node is pinned

`neighbour_variability_20260813.reproducibility_defect_found`: `Generator.multinomial`
differs in 19/10000 rows between numpy 2.5.1 (.73) and 2.4.6 (.82), max margin drift
**0.0053 pp** — an order of magnitude **larger** than the 5e-4 pp reproduction hard-fail
threshold. The archived shallow-rung cells this scan must reproduce were produced on **.73
(numpy 2.5.1)**, so the statistical analysis runs **on .73 only**. GPU scoring may be
split across nodes (it is deterministic and node-independent); the bootstrap may not.
