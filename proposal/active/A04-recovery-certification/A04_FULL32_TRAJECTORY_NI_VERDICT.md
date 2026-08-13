# A04 — full32_dolmino trajectory: the NI accept is NOT the endpoint

**verdict string:** `SOME_CHECKPOINT_MEETS_THE_2OF3_BAR__PREREG_B_ACCEPT_NOT_FROM_CONVERGENCE`

| | |
|---|---|
| **date** | 2026-08-13 |
| **cost** | **6.53 GPU-h** scoring (723+727 s on `.73`, 745+744 s on `.82`; 2939 s total × 8 GPUs) + 0 GPU-h analysis (CPU, 59 s) |
| **nodes** | `.73` (step5000, step10000) and `.82` (step15000, step20000), 8×H20 each, zwfy6. Analysis pinned to **`.73`**. **NOT touched:** `.104` (paperC Qwen3 heal), `LOCAL`/`.21` (SparseForge #246 — wzc1 read-only, as the ckpt source) |
| **training** | **zero** |
| **evidence** | `evidence/a04_full32_trajectory_ni.json` |
| **prereg** | `A04_FULL32_TRAJECTORY_PREREG.md`, commit `537d323`, written before any of the four new checkpoints had a `summary.json` |
| **code** | `code/a04_full32_trajectory_axes_driver.sh`, `code/a04_full32_trajectory_ni.py`, `code/a04_full32_stage_parallel.sh` |

---

## 1. Headline

The dispatch asked where on the `full32_dolmino` trajectory A04's single NI accept
appears, expecting (per `full32_rescore_v2_20260812.trajectory_scan_NOT_run`) that
*"earlier ckpts are LESS converged so should reject HARDER: the accept boundary is most
likely BEYOND step25000."*

**The opposite is true. The four earlier checkpoints all ACCEPT on 2 of 3 decision axes;
the endpoint is the only one that FAILS the bar.**

| step | triviaqa | popqa | mmlu_content | axes accepting | **NI verdict (≥2/3)** |
|---|---|---|---|---|---|
| 5 000 | **+3.6932 A** | −2.0509 R | **+0.5439 A** | 2/3 | **ACCEPT** |
| 10 000 | **+3.0467 A** | −2.1911 R | **+0.3730 A** | 2/3 | **ACCEPT** |
| 15 000 | **+2.3331 A** | −3.0742 R | **+0.9142 A** | 2/3 | **ACCEPT** |
| 20 000 | **+2.4504 A** | −3.5088 R | **+0.6222 A** | 2/3 | **ACCEPT** |
| 25 000 *(archived endpoint)* | **−0.6035 R** | −4.5391 R | **+1.0495 A** | 1/3 | **REJECT** |

Margins in pp, `split` convention, decision axes. **Identical under all 5 tie conventions**:
`split` / `first` / `last` / `wrong` all give 2-of-3 at the four earlier steps and 1-of-3 at
the endpoint; under `credit`, D6 retires MMLU (5 cells) so the family is 2 axes and the
pattern becomes 1-of-2 accepting vs 0-of-2 — **the ordering and the verdict flip are
unchanged in every convention.** `nq_open` is design-demoted and rejects at every step
(−2.85 to −3.66).

**Pre-registered reading (b) fires, in its strongest form.** The accept set on `triviaqa` is
`{5000, 10000, 15000, 20000}` — it accepts at the **first** checkpoint measured and **loses**
the accept by the endpoint. This is not a boundary being approached from below; it is a
capability being **walked away from**.

## 2. Direct answers to the three pre-registered questions

**Q1 — where does the accept appear?** It does not *appear* anywhere: it is present at
step5000, the earliest checkpoint that exists, and is **lost between 20000 and 25000**. The
one boundary this trajectory contains is a **REJECT boundary in (20000, 25000]**, not an
accept boundary. Nothing in A04's design anticipated that direction.

**Q2 — does `triviaqa` get closer to accepting at earlier steps?** It is not close at all —
it is **already accepting**, by up to **+3.69 pp**, i.e. **6.1× further past the threshold**
than the endpoint is short of it (−0.60 pp). The endpoint's celebrated "1.86 SE from
accepting" is the *worst* point on the trajectory, and the 12.39 SE of headroom at step5000
is what the rule saw at every earlier checkpoint.

**Q3 — RATIO vs NI along the trajectory.** The disagreement is **unique to the endpoint**:

| step | RATIO mean | margin over ρ=0.85 | RATIO | NI | disagree |
|---|---|---|---|---|---|
| 5 000 | 0.898062 | **+0.048062** | A | A | no |
| 10 000 | 0.893552 | +0.043552 | A | A | no |
| 15 000 | 0.886364 | +0.036364 | A | A | no |
| 20 000 | 0.879378 | +0.029378 | A | A | no |
| 25 000 | 0.851495 | **+0.001495** | A | **R** | **yes** |

RATIO decreases **monotonically** and is 32× further from its threshold at step5000 than at
the endpoint. So the rule disagreement that `shallow_rung_ni_discrimination_20260812` called
*"A04's target rule-disagreement"* exists at **exactly one** of five checkpoints of one run —
the last one — and it arises because **NI moves while RATIO merely decays**. It is not a
stable property of the arm; it is a property of that checkpoint.

## 3. What the trajectory is actually doing: it is getting WORSE, monotonically

`popqa` margin is **monotone non-increasing across all five points** (−2.05 → −4.54), and 3
of its 4 successive moves are *resolved* on the item sample. `triviaqa` accuracy falls
61.40 → 57.15. `nq_open` falls. Only `mmlu_content` drifts mildly upward (+0.54 → +1.05,
non-monotone, and its 20000→25000 move is **not resolved**: CI lower bound lands exactly at
0.0000 while p = 0.0508, `criteria_disagree=true`, treated as NOT resolved by the
conservative AND).

**This is continued pretraining making a healthy model worse on 3 of 4 axes** — measured, not
inferred. The resolved moves:

| interval | triviaqa | popqa | mmlu_content |
|---|---|---|---|
| 5000→10000 | −0.6409 **resolved** (p=0.0072) | −0.1332 ns | −0.1211 ns |
| 10000→15000 | −0.7078 **resolved** (p=0.0038) | −0.8691 **resolved** (p=0.0001) | +0.5056 **resolved** (p=0.0418) |
| 15000→20000 | +0.1170 ns | −0.4276 **resolved** (p=0.0212) | −0.2706 ns |
| 20000→25000 | **−3.0205 resolved** (p=0.0001) | −1.0023 **resolved** (p=0.0001) | +0.4202 **NOT resolved**, criteria disagree |

## 4. ⚠️ The endpoint's `triviaqa` REJECT is substantially a VERBOSITY artefact

This is the finding that most changes how the archived cell should be read, and it is a
**labelled diagnostic** — the decision metric is and stays EM.

Of the **1313** items that went EM right→wrong over 20000→25000, **622 (47.4 %) still
CONTAIN the gold answer**, and mean prediction length on exactly those items explodes
**10.9 → 50.4 characters**. Verbatim:

- gold `Rudolf Hess` — step20000: `"Rudolf Hess"` → step25000: `"Rudolf Hess, who was the last inmate of Spandau prison in Berlin, was found hang…"`
- gold `Dark Blood` — step20000: `"Dark Blood"` → step25000: `"He died during the filming of the movie \"Dark Blood\" in 1993."`
- gold `Vancouver, Canada` — step20000: `"Vancouver, British Columbia"` → step25000: `"The Lions Gate Bridge is a suspension bridge in Vancouver, British Columbia, Can…"`

The `contains` metric confirms it: across the whole trajectory `triviaqa` EM falls
61.40 → 57.15 (−4.25 pp) while `contains` moves only 69.87 → 68.39 (−1.48 pp), and the
contains−EM gap **widens 8.47 → 11.24 pp**, versus 3.87 pp for the vanilla anchor.

So the axis whose REJECT flips the endpoint's verdict is measuring, to nearly half its
movement, **a model that stopped emitting short answers** — expected behaviour for a base LM
continued-pretrained on raw text with no SFT — and **not** demonstrated forgetting. The
other 52.6 % are genuine content substitutions (`Richard Noble`→`Andy Green`,
`David Hockney`→`David Bowie`), so this is not the whole story either. Both halves are on the
record; **neither re-scores any cell.**

## 5. §2.0.2 neighbour precondition — applied per axis, as required

9 accepting cells were tested against their ±5000-step neighbours:

- **`mmlu_content`: all 5 accepts survive their neighbours.** Every present neighbour also
  accepts, at every step including the endpoint. This is the one axis where the accept is
  *not* checkpoint-selection dependent.
- **`triviaqa`: 3 of 4 accepts survive; `step20000` FAILS** — its upper neighbour
  (step25000) rejects. Flagged `ACCEPT_IS_CHECKPOINT_SELECTION_DEPENDENT`.
- `step5000` (both axes) and `step25000` have only ONE neighbour each — they are the
  trajectory endpoints. Stated, not silently treated as satisfied.

This is the first time §2.0.2 has had an actual accept to gate, and it **fired on a real
cell** (`triviaqa|step20000`) — the precondition is doing work, not decorating the design.

## 6. Integrity, protocol and provenance

- **Archived endpoint reproduced to 3.6e-07 pp** (threshold 5e-4): triviaqa −0.603544,
  popqa −4.539146, mmlu_content +1.049530, and RATIO `mean_ratio` 0.8514950516430542 with
  **abs diff exactly 0.0**. The imported guard/anchor/rule are provably the ones that
  produced the archive, so the four new points sit on the endpoint's scale.
- **Anchor never substituted**: vanilla `../models/OLMo-2-1124-7B`, imported from
  `a04_shallow_rung_ni_7b.ANCHOR`. Guard **G2** honoured — `full32_step25000` is an ARM, never
  the anchor. Δ never recomputed against a different intact.
- **Guard D1–D6**: all four axes CERTIFIABLE under `split` (and under `first`/`last`/`wrong`);
  **0 of 15** decision cells retired. Under `credit`, `mmlu_content` is `NOT_CERTIFIABLE` by
  D6 and **5 of 15** cells are retired — exactly as `A04_MARGIN_GUARD_PREREG.md` §2.2
  predicted for 7B, and as the archived shallow-rung pass already recorded. No verdict
  depends on it.
- **24 of 24 shard cells**: index set **exactly {0..7}** (not a file count), merged n exactly
  17944/14267/3610/14042, 0 duplicate item_ids, 0 nan, item_id sequences identical across all
  six arms. Per-shard row counts in the JSON.
- **Protocol asserted from the INVOCATION and fail-closed**, then asserted a **second,
  stricter** way (every `DRIVER START` line, not just the first, plus step coverage — the two
  invocations per node share one append-only log). **Negative-tested twice**: a doctored log
  reading `cb_bs=48` → `FATAL protocol deviation`, a missing log → `FATAL: driver log …
  absent`; in both cases **no output file was written**. `cb_bs=32`, `mmlu_bs=16`,
  `add_bos` asserted `is False` on all 12 new result dirs, `max_new_tokens=32`,
  `chat_template=False` established structurally (neither harness has a chat-template code
  path). `mode=pruned keep_front=32 n_fresh=0 num_hidden_layers=32`, matching the archive's own
  label verbatim.
- **Harness md5 identical on all three disks** and to the copies that produced the anchor and
  the endpoint: `eval_olmo2_closedbook_qa.py` `2ed41993…`, `eval_olmo2_mmlu_content.py`
  `fe4a62db…`. Same-CODE comparison.
- **Staging**: full-file **sha256 equal on both disks** for all 4 ckpts (`a206cb96…`,
  `244d4db4…`, `dacf11ea…`, `fc28e917…`), zip entries **1435 = source** on each, plus a
  `torch.load` probe asserting meta step / keep 32 / fresh 0 / 32 layers /
  `len(model_state)==355` before any GPU was spent. A prefix hash was deliberately **not**
  used: the known cluster failure mode is a truncated write, which a prefix hash cannot see.
- **Analysis pinned to `.73` (numpy 2.5.1)**, the node that produced the archive, because
  `Generator.multinomial` drifts 19/10000 rows vs numpy 2.4.6 on `.82` — up to 0.0053 pp,
  an order of magnitude larger than the 5e-4 pp reproduction hard-fail. GPU scoring was split
  across nodes (deterministic); statistics were not.
- **Bootstrap offsets** 500–503 for the new arms, endpoint keeps its archived 203, interval
  offset SEED+2900. Disjoint from {0,1} / 100–102 / 200–203 / 300–301 / 400–408 and the
  700/900/1700/1900/2400 guard offsets.

## 7. Premises of the dispatch that were WRONG

1. **"~5.7 h of transfer at a measured 16.3 MiB/s"** — the *measurement* is right (I
   reproduced 17.4 MiB/s single-stream) but the *inference* is wrong. **16–17 MiB/s is a
   PER-STREAM ceiling, not the link's capacity.** 8 concurrent streams to ONE node measured
   **130.7 MiB/s**; realised per-ckpt rate was **137–138 MiB/s**. All four checkpoints staged
   and sha256-verified in **~31 min**, not 5.7 h. The WAIT-for-B200 decision in
   `full32_trajectory_staging_remeasured_20260813` rested on that inference and was
   unnecessary — but note the conclusion (*stage now*) was already correct for a different
   reason, so this widens the margin rather than reversing the call.
2. **"Earlier ckpts should reject HARDER; the boundary is likely beyond step25000"**
   (`trajectory_scan_NOT_run.expectation_to_design_for`) — **falsified.** Earlier
   checkpoints accept; the endpoint is the worst point on 3 of 4 axes.
3. **"full32 sits at 97.7 % MMLU recovery … the accept BOUNDARY is on that trajectory"** —
   an accept boundary is *not* on this trajectory. A **reject** boundary is, in (20000, 25000].
4. **"4 ckpts × 81.6 GiB"** — the dispatch's own corrected count. Confirmed: 5 files exist,
   4 are new scan points, `step25000` was already scored and was **not** re-scored (0 wasted GPU).

Confirmed as described: `mode=pruned` in the archived meta despite zero damage; anchor =
vanilla; all four ckpts zip-OK with 1435 entries; `.73`/`.82` both idle before launch.

### 7.1 MAIN's interim note — direction right, arithmetic wrong (as MAIN predicted)

`A04_FULL32_READING_B_IS_FIRING.md` (MAIN, written 11:58 while the scan was still running,
from the first completed checkpoints) called reading **(b)** correctly and asked for exactly
the three things this document delivers. Its closing instruction was *"Do not let my hand
arithmetic into the record… Canonical output only."* That instruction was right, and it is
enforced here:

| quantity | MAIN's hand arithmetic | canonical (`build_nulls` + `ni_rule`) |
|---|---|---|
| mmlu_content margin @ step15000 | **+1.2775 pp** | **+0.9142 pp** |
| mmlu_content residual @ step15000 | 18.0299 pp | — (recovery fraction 0.9686) |
| is step15000's margin > step25000's? | **yes** (+1.2775 > +1.0495) | **no** (+0.9142 < +1.0495) |

MAIN derived the residual by subtracting a recorded null instead of importing `build_nulls`,
the same shortcut it flagged as having put its keep14 margins ~0.5 pp off; the error here is
+0.36 pp in the same direction. **The canonical numbers are the record.** Consequences: on
`mmlu_content` the endpoint IS the best checkpoint (+1.0495) and its accept set IS a suffix
(all five accept), so MAIN's stronger claim — *"the earlier one accepts by more"* — is **false
on mmlu_content**. It is emphatically **true on `triviaqa`** (+3.6932 at step5000 vs −0.6035
at the endpoint), which is the axis that actually flips the verdict. So reading (b) fires, but
via a different axis than MAIN's interim note inferred, and the "no boundary on this
trajectory" hypothesis resolves into "no *accept* boundary; a *reject* boundary in
(20000, 25000]".

Also asked for and answered: **`mmlu_content` is NOT monotone in step** (successive diffs
−0.1709, +0.5412, −0.2920, +0.4273), and that non-monotonicity is in the verdict string via
`ACCEPT_NOT_FROM_CONVERGENCE`.

## 8. Licensed vs NOT licensed

**Licensed.** The 7B accuracies, nulls, residuals, Δ, lo95 bounds and margins; the exact
reproduction of the archived endpoint; *"no realisable perturbation of the ITEM SAMPLE flips
these verdicts"* (1.54–16.39 bootstrap SE, measured at 7B); *"the four earlier checkpoints
accept on 2 of 3 decision axes and the endpoint does not"*; *"the popqa margin degrades
monotonically across the trajectory"*; *"the RATIO/NI disagreement occurs at exactly one of
five checkpoints"*; and *"47.4 % of the endpoint's triviaqa EM regressions still contain the
gold answer"*.

**NOT licensed.**

1. **Anything about recovery from STRUCTURAL DAMAGE.** `full32` = `keep_front_layers=32,
   n_fresh_layers=0`: all 32 pretrained layers, nothing transplanted, nothing pruned. This is
   a **continued-pretraining control**. Every statement here is about **CPT drift on the heal
   corpus**. The caveat in
   `shallow_rung_ni_discrimination_20260812.the_load_bearing_new_finding.caveat` continues to
   travel with it.
2. **Any claim that these deficits are large or small "relative to seed variance."** `sd_run`
   is **1B-only** (S=3, keep12@5000). Every 7B rung has **exactly one seed**, and the
   historical 7B seeds are **unrecorded** (`--seed` postdates them), so no 7B `sd_run` is
   computable or reconstructible. The `deficit/sd_run` column in the JSON is labelled
   cross-scale extrapolation.
3. **The five checkpoints are NOT replicates.** They are five states of ONE optimisation run;
   their spread is training progress plus data order. The `triviaqa` collapse may not be
   attributed to, or excused as, seed variance.
4. **"Harness noise" is not available as an explanation.**
   `full32_rescore_v2_20260812.correction_to_the_jitter_premise` established there is **no
   measured runtime-jitter floor** on this harness (same-code re-runs are bit-identical).
   These are five *different* models, so bit-identity does not apply — but neither does "noise".
5. **The mmlu 20000→25000 move is NOT established.** CI lower bound exactly 0.0000 with
   p = 0.0508; criteria disagree; recorded as NOT resolved. Picking the favourable criterion
   would turn a tie into a result.
6. **No K1/K2/K3 clause fires.** They are defined over the pre-registered 1B arm set; a 7B
   trajectory cannot fire them. `STATUS.json:warning`'s two-corpora / unequal-steps caveat
   still holds.
7. **`contains` may not be substituted for EM.** §4 characterises what an EM move consists
   of; it does not re-score a cell, and the verdict table is EM throughout.

## 9. What this means for A04

The question A04 posed — *"does the gate discriminate?"* — now has a sharper and less
comfortable answer than either pre-registered branch anticipated.

**NI does discriminate: it changes state along a trajectory, and the state change is
resolved on the item sample.** That was the thing that had never been demonstrated. But the
direction is the reverse of the design's expectation: NI accepts a model that has been
continued-pretrained for 5 000 steps and *revokes* the accept at 25 000. The rule is
tracking **drift away from the anchor**, not **approach to it**.

Consequences:

- **`full32@step25000` should stop being described as A04's accept.** It is the only
  checkpoint on its own trajectory that **fails** the bar, and it was selected as the arm
  simply because it was the last save. The accept, and the 97.7 % recovery framing, belong to
  the checkpoints nobody had scored.
- **The "RATIO is too permissive" claim weakens considerably.** It rested on a disagreement
  at one checkpoint whose RATIO margin was +0.0015. Four checkpoints earlier on the same
  trajectory show **both** rules accepting, RATIO with 20–32× more headroom. The endpoint is
  where RATIO's decay happens to cross NI's, not where the rules differ in principle.
- **The neighbour precondition (§2.0.2) is vindicated by use** — it fired on
  `triviaqa|step20000`, an accept that does not survive its upper neighbour. Had this scan
  been run with a single hand-picked checkpoint, the reported margin would have been an
  artefact of that choice.
- **"Heal longer" remains unsupported**, and this pass makes it worse: on a *zero-damage*
  model, more continued pretraining moved 3 of 4 axes **away** from the anchor. Combined with
  `keep14_trajectory_ni_20260813.accept_boundary_is_not_on_this_trajectory` (8–244× the whole
  heal budget away, negative popqa slope), the premise that additional heal steps approach
  certification is now falsified at 7B on **both** a damaged and an undamaged arm.

**Cheapest next step**, if any: `outputs/olmo2_probe2_7B_full32_dolmino/` contains only these
five saves, so the (20000, 25000] reject boundary cannot be narrowed without new training.
The trajectory scan asked for by `cheap_next_steps_dominate[1]` is now **complete on both
halves** (keep14 in `517c8d2`, full32 here) and no further EVAL-ONLY step on these arms is
outstanding.
