# A04 — σ_run from the post-`ce5c298` data-order triplet, and K2

**Date**: 2026-08-13. **GPU-h spent: 0.0** (`.73` used as a CPU host; all 8 cards
verified at 0 MiB at dispatch and at exit, refuse-guard armed at >8000 MiB).
**Pre-registration**: `A04_SIGMA_RUN_POSTFIX_K2_PREREG.md`, commit **`94839e8`**,
committed **before the first number was computed**.
**Canonical evidence**: `evidence/a04_sigma_run_postfix.json`
(md5 `5d4f353822a2191ce1d4e1f0dbe00f88`, byte-identical on both disks).

> **Every number below is from the canonical JSON.** Where this prose and the JSON
> disagree, **the JSON wins** (`PROPOSAL.md` §4). Nothing here is hand-arithmetic.

---

## 0. Answers, in the order asked

| Q | Answer |
|---|---|
| **premise** | **HOLDS.** All three arms are post-`ce5c298`, each with a *positive* per-run preflight assertion of the fixed sampler line, and identical in every logged config field except `seed`. Checkpoints align at `step{205000,210000,215000,220000}`; evals exist at **step220000 only**, which is all a σ_run needs. |
| **Q1 σ_run (df=2)** | triviaqa **0.2688 pp** χ²95 [0.1399, 1.6893]; popqa **0.2311** [0.1203, 1.4523]; mmlu_content **0.0583** [0.0304, 0.3663]; *nq_open 0.0800 [0.0416, 0.5026]*. 12.07× multiplicative width at df=2. |
| **Q2 vs A03's σ̂** | **The premise's framing needs one correction, and it matters.** A03's keep7 σ̂ is **not** a pre-fix estimate — it is a **contaminated** one (3 post-fix draws + 1 pre-fix draw). Clean/contaminated ratio: triviaqa **0.67×**, popqa **1.18×**, mmlu_content **1.05×**. **A clean pre-vs-post contrast is NOT COMPUTABLE in this repo** and is reported as such rather than proxied. |
| **Q3 K2** | **K2 DOES NOT FIRE.** 0 of 3 decision axes exceed Δ. Smallest headroom popqa **3.39×**. And it does not fire under **any** of the four defensible estimators. **But this is NOT a clearance** — see §5. |
| **Q4 1B→7B** | **The bound direction CANNOT BE SIGNED, and I decline to sign it.** Two unmeasured effects push opposite ways. Quantified in §6. |

---

## 1. Premise verification — the part that could have stopped the task

I was asked to verify the premise first and to stop if it failed. It did not fail,
but two of the checks changed what the result means, so they are not a formality.

### 1.1 Post-fix: established from the arms' own preflight assertions, not from mtime

Each arm's `logs/a03_dataorder_seed<S>_progress.log` records a `grep` of the **live
trainer line, printed before `torchrun` launched**:

```
[08-10 16:55:29] trainer post-ce5c298 OK: 869:        sampler = DistributedSampler(ds, shuffle=True, seed=args.seed)
[08-10 16:55:29] launched torchrun pid=2686820 seed=43
```

| arm | launch | vs `ce5c298` (2026-08-09 23:21:09) |
|---|---|---|
| seed43 | 2026-08-10 16:55:29 | **+17.6 h** |
| seed44 | 2026-08-10 16:57:29 | **+17.6 h** |
| seed45 | 2026-08-11 12:04:07 | **+36.7 h** |

Each log's line 1 is `[seed] set_seed(43|44|45) on all ranks`, and each
`arch_meta.json` carries the matching `"seed"`. The ckpt mtimes cited in the task
(08-10 19:47 / 19:48 / 08-11 14:57) are consistent, but nothing here rests on them.

### 1.2 Matched: from each arm's own log header

Identical across all three: `arm=healing_front7+fresh2`, `keep_front=7`,
`n_fresh=2`, `num_hidden_layers=9`, fp32 master weights,
`world_size=8 bs=8 gaccum=2 eff_bs=128 seq_len=2048`, `max_steps=300000`,
`dataset rows=15491607 from data/dolmino_now15b.npy`, `torch AdamW`,
`n_params=1015097344`, resume from `outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt`,
resume LR `6.504e-06`.

`[optim]` groups: `fresh_decay 339.7M / fresh_nodecay 0.0M / inh_decay 675.3M /
inh_nodecay 0.1M`, **all four at `2.00e-05`** → **LR is uniform. No differential-LR
claim is made.**

### 1.3 ⚠️ The step-alignment answer, and what it forbids

All three arms saved 4 checkpoints and they align. **But eval shards exist at
`step220000` only.** So σ_run is measured at **exactly one step**, which is
appropriate — a σ_run is a *level* comparison across runs at a common step and
needs no cross-step pairing. It does mean:

> **Nothing in this document is a trajectory, monotonicity, or neighbour statistic**,
> and **no `E[range of k]` constant is used anywhere.** σ here is a **sample sd
> (ddof=1)**. `1.6926` (= `c_3`) and `c_8 = 2.8475` are recorded in the JSON under
> `RANGE_CONSTANTS_DECLARED_UNUSED` **precisely so that nobody can later lift a
> `c_n` out of this document.**

### 1.4 ⚠️⚠️ The finding that changes the framing: this triplet varies data order ONLY

All three arms **resume from one common `step200000.pt`** (`restored 102 model
tensors (strict)`, Adam moments preserved). Therefore:

> **Fresh-tail init variance is identically ZERO in all three.** This is a **pure
> data-order σ_run**, not the full run-to-run σ_run a from-prune multi-seed gate arm
> would have.
>
> The pre-fix families are the exact mirror image (init only). **Neither is the full
> σ_run.** The keep12 Stage-B family (101/102/103) *is* full run-to-run variance,
> because those arms prune fresh from the HF base per seed.

**Direction of the bias: this σ is DOWNWARD-biased for full run-to-run variance,
i.e. OPTIMISTIC for K2.** This was pre-registered (`PREREG` §1.5, §4.2 item 3)
before any number was seen, and it is the single most important caveat on §4.

### 1.5 Protocol — from the actual driver and the scripts, not from `summary.json:meta`

Driver: `/tmp/a03_dataorder_ext_driver.sh` on `.73` (mtime 2026-08-10 19:13),
still on disk, invocations reproduced verbatim in the JSON's `protocol_audit`.

* **`chat_template` is False STRUCTURALLY**: `grep -c apply_chat_template` = **0** in
  *both* `eval_olmo2_closedbook_qa.py` and `eval_olmo2_mmlu_content.py`, and neither
  has a flag to enable one. The assertion in code is written
  **`chat_template is not False` → FAIL** (so `None` fails too), never
  `is not True`.
* `add_bos=false` in every cell; greedy `do_sample=False, num_beams=1`,
  `max_new_tokens=32`; no few-shot; `n_valid=n, nan=0, trunc=0`.
* Integrity per cell: shard index set **exactly `{0..7}`** (not a count), exact item
  counts 17944 / 14267 / 3610 / 14042, **0 duplicate item_ids, 0 nan**; MMLU read
  via nested `content_norm.correct`.

### 1.6 Independent corroboration of the per-seed means

My independently-loaded arm means reproduce A03's recorded `means_pct` for seeds
43/44/45 to **max |diff| = 0.000e+00 on all four axes** — byte-exact, consistent
with `memory/same-harness-runs-bit-identical`. My df=3 χ² CI on triviaqa also
reproduces A03's recorded `[0.22877984971402363, 1.505793867745346]`.

---

## 2. Q1 — σ_run, the first one whose seeds actually moved the data

`build_nulls` was **imported and called** on the pinned intact anchor
(`A03_1B_base`, rule G0, `split` tie convention). The Δ it produced was
**cross-checked against the canonical full-precision constants and matched on all
four axes within 1e-9** — so Δ is not copied from prose and not substituted (guard
G2). Nulls used: triviaqa `0.0025635309852875612`, popqa `0.0229200252330553`,
mmlu_content `0.28445022076627263`, nq_open `0.00554016620498615`.

| axis | seed43 | seed44 | seed45 | **σ_run (pp, df=2)** | χ² 95 % CI | `bound_3` | Δ | headroom |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| **triviaqa** | 9.6968 | 9.2399 | 9.2231 | **0.2688** | [0.1399, 1.6893] | 0.4531 | 4.0431 | **8.92×** |
| **popqa** | 4.1004 | 3.7149 | 3.6868 | **0.2311** | [0.1203, 1.4523] | 0.3896 | 1.3205 | **3.39×** |
| **mmlu_content** | 32.2319 | 32.1963 | 32.1179 | **0.0583** | [0.0304, 0.3663] | 0.0983 | 1.0239 | **10.42×** |
| *nq_open (DEMOTED)* | 2.9086 | 2.9086 | 2.7701 | *0.0800* | [0.0416, 0.5026] | 0.1348 | 0.9695 | *7.19×* |

`bound_3 = t_{0.05,df=2} · σ / √3`, `t = 2.9199855803537124`. χ² interval from the
df=2 closed form `ppf(p) = −2 ln(1−p)` (scipy absent on `.73`), asserted `df == 2`,
multiplicative width **12.0707×** — reproducing A03's recorded df=2 width exactly.

**Per the standing rule** (`STATUS.json:sigma_run_input_from_A03.standing_rule`,
A03 `DATAORDER_PREREG.md` §4): no σ_run point estimate above is quotable without
its d.o.f. and its χ² interval. **At df=2 the σ estimate is very imprecise.**

⚠️ `nq_open`'s seed43 and seed44 score **identically** (2.9086 = 105/3610 each).
That is the same coincidence A03 already flagged; it is a 3610-item axis and it is
demoted anyway. It is *not* evidence of zero variance.

---

## 3. Q2 — the comparison, with the premise's framing corrected

### 3.1 The correction: A03's keep7 σ̂ is contaminated, not pre-fix

The task's premise was that A03's σ̂ came from pre-fix arms. **What I found is
sharper and different**, and it is a defect in a committed evidence file:

> `a03_sigma_run_n3.json` → `families.keep7_20k_cpt.seeds = [0, 43, 44, 45]`.
> **Seed 0 is A03 Arm 3**, whose own `logs/a03_arm3_progress.log` records
> `[08-09 01:11:43] launched torchrun pid=3642559` — **22 h 09 m BEFORE `ce5c298`**
> — and whose log line 1 is `[seed] set_seed(42)`. Its progress log carries **no**
> `trainer post-ce5c298 OK` line, unlike all six seeded runs.

So that σ̂ is **3 post-fix draws pooled with 1 pre-fix draw** — which
`PROPOSAL.md` §7.2 forbids in exactly those words. `families.pooled_df5` (and
`STATUS.json:sigma_run_input_from_A03.pooled_df5`) inherit it.

**What is and is not affected:**

* **NOT affected: K2's pre-registered estimator.** That is the **keep12** family
  (101/102/103), all three post-fix with positive preflight assertions
  (`PILOT_ONE_PREREG.md` §2.2). K2's arithmetic is numerically untouched.
* **Affected**: the keep7 df=3 σ and the pooled df=5 σ are not 口径-clean as
  recorded.
* `STATUS.json:sampler_fix_and_pilot_one_disposition_20260812`'s sentence *"Every
  run A04 consumes as σ_run input is POST-fix"* is **true of the six runs it
  enumerates** and **false of the keep7 FAMILY as recorded**, which carries a 4th
  draw it did not enumerate. The archived A03 JSON is **not edited** — it is
  provenance.

### 3.2 ⚠️ …and then the exclusion turns out to be conservative, not necessary

§7.2 excludes pre-fix arms because pre-fix seeds varied *only* fresh-tail init. On
**this** arm that premise does not obtain: all four keep7 draws resume from one
common ckpt, so init variance is zero in all four and the **only** stochastic input
in every one of them is the sampler order. I tested this mechanically rather than
reasoning about it (`EXCHANGEABILITY_PROBE_…` in the JSON):

| probe | result |
|---|---|
| Arm 3 (`--seed 42`, pre-fix) sampler.seed | **0** |
| post-fix `seed=0` explicitly, first 12 indices | **BIT-IDENTICAL to Arm 3's** |
| distinct orders among {arm3, 43, 44, 45} at `set_epoch(1)` | **4 of 4** |
| rank-0 20k-step slice Jaccard, arm3 vs 43/44/45 | 0.0105 / 0.0104 / **0.0101** |
| same, among post-fix pairs 43-44 / 43-45 / 44-45 | 0.0104 / 0.0106 / 0.0104 |

Arm 3's slice is near-disjoint from every post-fix draw, and **indistinguishable
from how disjoint the post-fix draws are from each other**. So on this arm seed 0
is a **legitimate 4th draw from the same data-order family.**

**I still applied the exclusion for the headline σ.** §7.2 is a binding
pre-registered rule and this document does not get to reinterpret it after seeing
that the wider family would be convenient. But it is now known to be
**conservative here rather than necessary**, both readings are reported, and §4.2
shows the verdict is identical either way — so nothing rests on the choice.

> **This does NOT rehabilitate pre-fix seeds in general.** It holds only where a
> common resume ckpt zeroes the init variance. For any arm pruned fresh per seed —
> the keep12 family, and **every arm in the gate design** — pre-fix seeds genuinely
> carry init-only variance and §7.2 applies with full force.

### 3.3 The ratio, labelled for what it is

| axis | §7.2-clean df=2 | A03 keep7 **contaminated** df=3 | ratio clean/contaminated | keep12 post-fix df=2 | ratio keep7clean/keep12 |
|---|---:|---:|---:|---:|---:|
| triviaqa | 0.2688 | 0.4039 | **0.67×** | 0.3023 | 0.89× |
| popqa | 0.2311 | 0.1959 | **1.18×** | 0.3328 | 0.69× |
| mmlu_content | 0.0583 | 0.0555 | **1.05×** | 0.0783 | 0.74× |
| *nq_open* | *0.0800* | *0.0750* | *1.07×* | *0.2091* | *0.38×* |

**Direction: mixed, and on the one axis that moves materially the removal makes σ
SMALLER, not larger.** Why triviaqa is the exception: seed 0's triviaqa mean
(10.0646) sits **+0.6780 pp = +2.52 clean-σ** above the mean of the other three, so
it was carrying most of the df=3 spread. On the other three axes seed 0 is within
±1 clean-σ (+0.45 / +0.98 / +0.92) and removing it slightly *raises* σ (as expected
from df 3→2 with an unremarkable point removed).

### 3.4 ⚠️ The premise's actual question is NOT ANSWERABLE, and I will not proxy it

The task asks whether **real data-order variance** is bigger or smaller than the
**pre-fix** σ̂. That requires ≥2 *pre-fix seed replicates* of one arm with evals on
these axes. Searched **both disks**:

* `outputs/olmo2_probe2_7B_keep14fresh2_seed1234` — the only pre-fix multi-"seed"
  object in the repo. **7B**, **no eval shards on either disk**, already labelled
  `init-variance only`.
* A03 Arms 3/4/6 are pre-fix but are **different LR schedules** (`arm4=peaklr`,
  `arm6=lowerband`). Their spread is a schedule effect, not seed variance.

> **So "is real data-order variance larger or smaller than pre-fix init variance"
> cannot be answered from this repo's data.** Reported as not-computable rather
> than proxied. What *can* be said, and is said in §3.3, is the effect of
> **removing one contaminating draw** — a different and smaller claim.

**Consequence for the task's framing question** ("does this make the power analysis
optimistic or pessimistic?"): **neither, on this evidence.** The power analysis
that actually drives K2 uses the **keep12** family, which was never contaminated.
The keep7 numbers move by 0.67–1.18×, in both directions, and K2's verdict is
unchanged under every estimator (§4.2). **The premise that "the σ̂ measures the
wrong thing, so every power calculation is mis-specified" is not sustained** — the
defect is real but it is a 口径 bookkeeping error in an archived file, not a
change to any decision.

---

## 4. Q3 — K2

### 4.1 Verdict, on the pre-registered point-estimate rule

> ## **K2 DOES NOT FIRE.**
> **0 of 3** decision axes have `bound_3 > Δ`. The rule needs **≥ 2**.
> Tightest axis: **popqa, `bound_3` 0.3896 vs Δ 1.3205 = 3.39× headroom.**

### 4.2 It does not fire under ANY defensible estimator

Reported because a verdict that hinges on a contested inclusion decision (§3.2)
must be shown not to depend on it:

| estimator | S | df | σ triviaqa / popqa / mmlu | axes exceeding | verdict |
|---|---:|---:|---|---:|---|
| **keep12 df=2 — PRE-REGISTERED** | 3 | 2 | 0.3023 / 0.3328 / 0.0783 | **0/3** | **K2_DOES_NOT_FIRE** |
| keep7 §7.2-clean df=2 — *this document* | 3 | 2 | 0.2688 / 0.2311 / 0.0583 | **0/3** | K2_DOES_NOT_FIRE |
| keep7 contaminated df=3 — *as A03 recorded* | 4 | 3 | 0.4039 / 0.1959 / 0.0555 | **0/3** | K2_DOES_NOT_FIRE |
| pooled keep7clean+keep12 df=4 — *sensitivity only* | 3 | 4 | 0.2860 / 0.2865 / 0.0690 | **0/3** | K2_DOES_NOT_FIRE |

**All four agree.** Note the pooled row is reported **as a sensitivity only**: per
`STATUS.json:...K2_STATUS_UNCHANGED_BY_SEED45.tempting_but_NOT_LICENSED`,
substituting a pooled σ after seeing which answer each gives is a change of
estimator and remains **unlicensed**. It agrees, so it changes nothing.

### 4.3 The χ² upper bound — reported, NOT OR-ed in

At the χ² upper limit of σ, **popqa** would exceed Δ (`bound 2.4484` vs `Δ 1.3205`)
on the clean keep7 family; triviaqa and mmlu_content would not. That is **1 of 3**,
below the ≥2 rule. Identical shape on the pre-registered keep12 family (popqa
`3.5260` vs `1.3205`, 1 of 3) — matching
`STATUS.json:pilot_one.MAIN_correction_20260812_1630` exactly.

**This is not a second decision rule.** The pre-registered test is on the point
estimate. Per `PREREG` §4.3 I foreclose the failure mode **in both directions**:

* I do **not** write "K2 FIRES" because the χ² upper bound would exceed Δ on 1–2 axes.
* I do **not** write "K2 is cleared" because the point estimate does not fire.

The honest line, which must ship with any K2 statement: **at df=2 σ is imprecise
(12.07× CI width), and the verdict is FRAGILE on one of three decision axes
(popqa).** On the contaminated df=3 family — more d.o.f. — even the χ² upper fires
on 0/3, which is a point in favour of buying d.o.f. rather than reinterpreting.

### 4.4 K2 limb 1 is NOT EVALUABLE, and that is worth stating

K2 has two limbs joined by "equivalently". Limb 2 (the `bound > Δ` test) is what
§4.1 adjudicates. **Limb 1** — σ_run ≥ 50 % of *"the smallest between-arm residual
difference the paper wants to claim"* — requires a quantity **A04 has never
declared**: the 4-arm gate never ran and no between-arm difference is claimed
anywhere. **Limb 1 is therefore not evaluable**, on this family or any other. It is
not treated as satisfied, and it is not treated as failed.

### 4.5 ⚠️ K2 not firing is NECESSARY, NOT SUFFICIENT

Pre-committed in `PREREG` §4.2 before the data: **a large σ kills; a small σ does
not clear.** Three reasons, all specific to this family:

1. **Wrong arm.** `keep7+fresh2` = 56.2 % depth, a **confirmed constant-REJECT**
   rung. **A saturated deficit is highly reproducible, so low seed variance is
   exactly what saturation looks like.** K2 is a variance gate and is
   **structurally blind** to it. (`STATUS.json:pilot_one.CRITICAL_CAVEAT`: "Stage B
   passed its kill gate and still failed its purpose.")
2. **Wrong budget.** 20 000 warm-resume steps from step200000, not the gate's
   from-prune budget.
3. **Partial stochasticity (§1.4).** Common init ⇒ **downward-biased** σ ⇒
   optimistic for K2.

> **So this document can kill A04 but cannot license Pilot Two.** The 1,077–4,309
> GPU-h gate remains **BLOCKED**, for the reason already on the record: no rung is
> known where NI can be *observed to accept*. **That is a rung-selection problem,
> not a variance problem, and more seeds cannot fix it.**

---

## 5. Q4 — the cost of gating a 7B experiment with a 1B σ_run

7B Δ values are **canonical**, read from A04's own 7B evidence
(`a04_keep14_trajectory_ni.json` / `a04_control_arms_ni.json`,
`per_convention.split.delta_pp` — identical in both), not re-derived.

| axis | σ_run 1B | `bound_3` 1B | Δ **1B** | Δ **7B** | Δ7B/Δ1B | exceeds Δ7B? | σ inflation to fire vs Δ7B |
|---|---:|---:|---:|---:|---:|---|---:|
| triviaqa | 0.2688 | 0.4531 | 4.0431 | 6.3291 | 1.565 | no (14.0× headroom) | ×13.97 |
| popqa | 0.2311 | 0.3896 | 1.3205 | 2.2457 | 1.701 | no (5.8×) | ×5.76 |
| mmlu_content | 0.0583 | 0.0983 | 1.0239 | 1.8614 | 1.818 | no (18.9×) | ×18.94 |
| *nq_open* | *0.0800* | *0.1348* | *0.9695* | *1.9945* | *2.057* | *no (14.8×)* | *×14.79* |

**Structural fact that the extrapolation turns on:** every 7B Δ is **1.57–2.06×
LARGER** than its 1B counterpart, because `Δ = 0.10 × residual(intact)` and the 7B
intact residual is larger. **So a σ held constant in pp is MORE easily accommodated
at 7B, not less.** Note what that means: the K2 test gets *easier to pass* as the
anchor's residual grows. That is a property of a **data-dependent margin**
(`must_not_claim[22]`, arXiv:2603.16213), **not** a property of the model — and it
is a second, independent reason not to read a non-firing K2 as reassurance.

### 5.1 Bound direction: **CANNOT BE SIGNED** — and I decline to guess

Two unmeasured effects act in **opposite** directions:

1. Δ is **1.57–2.06× larger at 7B** ⇒ using the 1B σ against a 7B Δ is
   **conservative / pessimistic**.
2. This 1B σ is **downward-biased** (common init, §1.4) ⇒ **optimistic**.

There is **no measurement of how σ_run itself scales with parameter count on this
harness**, so the product has unknown sign. Writing "upper bound" or "lower bound"
would be a guess, so per `PREREG` §6 I write **cannot be signed**.

### 5.2 What the external literature can and cannot contribute here

arXiv:2508.13144 (**NeurIPS 2025 Spotlight**; OpenReview `sAFottNlra`,
`venueid=NeurIPS.cc/2025/Conference`, `Camera_Ready_Revision` present — DBLP has it
CoRR-only, so S2/DBLP alone would misread it) Table 4 publishes OLMo-2 per-task
noise at 1.5B/7B/13B/32B — the only published handle on the sign of the scale
effect for this family. But it is **intact-model** noise, a rel-std over 30
consecutive checkpoints of **one** run (a *checkpoint-selection* quantity), on
**their OLMES protocol**. It is not a cross-run σ on A04's base protocol. Per
`must_not_claim[20]` and that note's own prohibition it may be **discussed, never
tabulated** against these numbers.

### 5.3 No 7B σ_run exists

`must_not_claim[23]`: one seed per 7B rung, historical seeds unrecorded, `--seed`
postdates the trainer revision that produced them. **No 7B σ_run is computable or
reconstructible.** §5 is a **sensitivity**, never a 7B result.

### 5.4 How far from firing — with the caveat attached

K2 needs ≥2 of 3, so the **second-easiest axis sets the bar**: σ would have to be
**≈8.9× larger** to fire against the 1B Δ and **≈14.0×** against the 7B Δ. The
margin is not marginal. **But per §4.5 a large distance from firing is not evidence
the gate is safe** — a constant-REJECT rung is *expected* to have small σ.

---

## 6. What this document does and does not add

**Adds:**

1. The first σ_run in this project computed **only** from runs whose seeds actually
   moved the data — and a mechanical demonstration (bit-identical order, Jaccard
   0.0101–0.0105) of what that means on this arm.
2. A defect in a committed evidence file: `a03_sigma_run_n3.json`'s keep7 family
   (and the pooled df=5 value that consumes it) is **§7.2-noncompliant**.
3. Evidence that the §7.2 exclusion is **conservative rather than necessary on
   arms with a common resume ckpt** — with an explicit prohibition on
   generalising that.
4. A K2 verdict robust across **four** estimators, plus the observation that **K2
   limb 1 has never been evaluable** for lack of a declared between-arm claim.
5. A quantitative statement of the 1B→7B cost, and a refusal to sign its direction.

**Does not add / must not be read as:**

- ⛔ A clearance of K2 or any authorisation of Pilot Two (§4.5).
- ⛔ Any claim that keep7 or keep12 can be *observed to accept*. Both are
  constant-REJECT.
- ⛔ A trajectory, monotonicity, or neighbour statistic; no `c_n` is used or
  available here (§1.3).
- ⛔ A full run-to-run σ_run (§1.4) — data-order only, downward-biased.
- ⛔ Any 7B σ_run, or any pre-vs-post-fix variance contrast (§3.4, §5.3).
- ⛔ A differential-LR claim (§1.2: uniform 2e-5).
- ⛔ Any number quoted finer than 0.01 pp across nodes (`must_not_claim[24]`).
- ⛔ Rehabilitation of pre-fix seeds in general (§3.2).

## 7. Provenance

| claim | source | how |
|---|---|---|
| σ_run, χ² CIs, bounds, Δ | `evidence/a04_sigma_run_postfix.json` | `per_axis.<axis>.{sigma_run_pp, sigma_chi2_95ci_pp, bound_S3_pp, delta_pp}` |
| Δ cross-check vs canonical | same | `intact_anchor.delta_cross_check_vs_canonical` (all 4 axes, tol 1e-9) |
| post-fix + matched config | `logs/a03_dataorder_seed4{3,4,5}{,_progress}.log`, `arch_meta.json` | verbatim in §1.1–1.2; preflight assertion lines |
| Arm 3 is pre-fix | `logs/a03_arm3_progress.log`, `logs/a03_arm3_cpt20k.log` | launch `08-09 01:11:43`; `set_seed(42)`; no post-`ce5c298` line |
| exchangeability of seed 0 | same JSON | `EXCHANGEABILITY_PROBE_seed0_is_a_LEGITIMATE_4th_DRAW` (real `DistributedSampler`, `set_epoch(1)`) |
| K2 under 4 estimators | same JSON | `SENSITIVITY_K2_under_every_estimator.verdicts` + `ALL_ESTIMATORS_AGREE` |
| Q4 transfer | same JSON | `Q4_1B_sigma_against_7B_delta` |
| A03 comparison | `a03_sigma_run_n3.json` md5 `5fb6cd4c3d693831e50d0817bda93ab8` (asserted at runtime) | `Q2_comparison_to_A03` |
| integrity / protocol | same JSON | `per_seed_integrity`, `intact_anchor.integrity`, `protocol_audit` |
| environment | same JSON | `node`, `numpy_version` **2.5.1** (`.73`), `chi2_method`, `gpu_refuse_guard` |
| canonical imports | same JSON | `canonical_imports` (`build_nulls` imported+called; `SEED=0`, `N_BOOT=10000`) |
