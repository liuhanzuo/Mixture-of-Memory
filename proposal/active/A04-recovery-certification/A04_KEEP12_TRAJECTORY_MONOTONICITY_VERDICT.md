# A04 — is the NI margin monotone along a damaged arm's DENSE heal trajectory? (keep12+fresh2, 8 points, 5 000-step grid)

**Verdict string:** `P_NARROWED_KEEP12_MONOTONE_IMPROVING_NOT_A_GENERAL_LAW_Q3_DOES_NOT_REPLICATE`

**Date:** 2026-08-13 · **GPU:** **12.5333 GPU-h** (`.73` driver 12:54:46 → 14:24:41 =
5390 s × 8 = 11.9778; `.82` cross-node control 548 s × 8 = 1.2178 — wall clock
driver-start to driver-end; the 11 four-axis scorings themselves summed to 5377 s).
Analysis is CPU-only.
**Nodes:** all 11 grid/aux checkpoints on **`.73`** (8×H20, zwfy6); **`.82`** ran one
same-checkpoint control. Both verified 8×0 MiB before launch; the driver refuses to
start above 8000 MiB.
**Not touched:** `.104` (paperC Qwen3 heal), `LOCAL`/`.21` (SparseForge #246).
**Pre-registration:** `A04_KEEP12_TRAJECTORY_PREREG.md`, commit **`4840c10`**, committed
**before the first keep12 capability number was computed**.
**Evidence:** `evidence/a04_keep12_trajectory_monotonicity.json`
**Code:** `code/a04_keep12_trajectory_axes_driver.sh`,
`code/a04_keep12_trajectory_monotonicity.py`

---

## 0. Headline, in one paragraph

**Claim P does not replicate on keep12+fresh2, and this is the outcome the
pre-registration named as "P IS NARROWED HARD."** On a genuinely damaged recovery arm
measured at **8 points, exactly 5 000 steps apart, inside one uninterrupted training
process**, the NI margin is **monotone-improving in trend on 2 of 3 decision axes**
(triviaqa ρ = +0.9701, p = 0.0004; popqa ρ = +0.8095, p = 0.0213) and **has no
detectable diffusion at all on the third** (mmlu_content range 0.6623 pp against a
noise floor of 1.0494 pp). **Zero of three decision axes WANDER.** The margin does move
non-monotonically *point to point* — 2 and 4 sign reversals respectively, and the
largest single-step excursions are a material 0.328× and 0.387× of Δ — but the
**direction of travel is unambiguously upward**, which is precisely what P denies. And
keep14's signature leg fails to reproduce: on the pre-registered length-matched window
(130000→155000, 25 000 steps vs keep14's 25 500) popqa **improves by +1.3317 pp**
(p = 0.0001), the **opposite sign** to keep14's −0.6729 pp. So the popqa mid-heal
regression is now **0 for 2** on independent arms (shortgpt16 failed it in
`A04_NEIGHBOUR_VARIABILITY_VERDICT.md` §3; keep12 fails it here, more strongly).
**P must be restated as arm-specific and may not be sold as a general methodological
law.** One unplanned finding is arguably worth more than the planned one: keep12's
500-step neighbour range (0.1951 pp) is **5.74× smaller** than keep8's, and across the
three arms now measured the range **rank-orders exactly by the learning rate at the
triple** (keep10 1.24e-5 → 1.2149 pp; keep8 6.80e-6 → 1.1202 pp; keep12 3.25e-6 →
0.1951 pp) at near-identical noise floors — so checkpoint scatter may be a property of the
**LR schedule position**, not of the arm (§5.1). What survives of P is narrower and still
useful — see §7 and §8.

---

## 1. Premises I checked first, and what each one turned out to be

The dispatch invited me to reject its premises. Two of its framing claims were right and
needed only confirmation; two of my *own* implementation choices were wrong and had to be
corrected before any number was published.

### 1.1 ✅ The grid is genuinely uniform and genuinely seam-free — checked BEFORE spending GPU

`A04_NEIGHBOUR_VARIABILITY_VERDICT.md` §1.2 found a 500-step cluster that straddled a
**resume seam** (the trainer restores optimizer state and RNG but rebuilds the loader via
`sampler.set_epoch(epoch); data_iter = iter(loader)` **without fast-forwarding inside the
epoch**, so the crossing interval saw a different data order). That is the first thing I
looked for here, because a seam anywhere inside the grid would confound the trend with a
data-order discontinuity.

`logs/olmo2_7B_keep12fresh2_resume200k_v2.log` has **exactly one** process start —
`[seed] set_seed(42) on all ranks` at 2026-08-08 13:58:02, one
`[resume] loading ckpt … step124000.pt (saved at step 124000, has_optimizer=True)`, one
`[resume] sampler.set_epoch(1)` — and **every one of the eleven checkpoints is written by
that single process** (grid saves at log lines 348 / 626 / 904 / 1182 / 1460 / 1738 /
2016 / 2294; 165500 and 166000 at 2323 / 2351; the log's last line is step 166020).
Every checkpoint meta also reports **`epoch=1`**, so no epoch boundary — and hence no
sampler reshuffle — occurs anywhere inside the grid either.

**So this is the first A04 trajectory measurement where "adjacent checkpoints" means
what it says**, for both the 5 000-step grid and the 500-step Q4 triple.

### 1.2 ✅ The checkpoints are distinct weights, and size proves nothing

`ls -l` shows **steps 130000–166000 all sharing 43 867 049 986 B**, with only step124000
differing (43 867 047 810 B). Byte size is therefore useless as an identity check here —
and a sibling in this repo (`shortgpt16/step128000.pt` on zwfy6) is a **truncated zip**
that is present, non-zero and dated like its healthy siblings, so existence-based
inventories are not evidence of readability either.

Identity was proven by content: all eleven `torch.load` (exercising the zip central
directory), all report `keep_front=12 n_fresh=2 num_hidden_layers=14`, **157 tensors**,
fp32, `epoch=1`, `has_optimizer=True`; per-tensor sha256 of `lm_head.weight` /
`embed_tokens.weight` / `layers.0.q_proj.weight` differ pairwise; and the **float64 sum of
every parameter** is distinct at every step and **monotonically decreasing**
(68640.843 → 67443.382 → 66041.978 → 65032.452 → 63042.541 → 62267.044 → 61244.476 →
60618.238 → 59721.344 → 59683.796 → 59540.192). The driver re-asserts the loaded meta's
step and arch **before** launching 8 GPUs.

### 1.3 ⛔ MY OWN BUG #1 — the noise gate constant. Reusing keep8's 1.6926 on 8 points would have been wrong by 40.6 %

`A04_NEIGHBOUR_VARIABILITY_VERDICT.md` §2.3 gates every range on **`1.6926 · σ`**. That
constant is `E[range of THREE iid N(0,1)] = 3/√π` — correct for a 3-point cluster and
**wrong for an 8-point grid**. The expected range grows with *k*: measured by Monte Carlo
(n = 2 000 000, fixed seed), **`c_8 = 2.8475`**, i.e. **1.683× larger than `c_3`**. Using
`c_3` on this grid would have put every noise floor **40.6 % too low**.

That is not academic. mmlu_content's observed range is **0.6623 pp**. Against the correct
floor (`c_8 · σ̄` = 1.0494 pp) it **fails** the gate and is reported as *no detectable
diffusion*. Against the wrong `c_3` floor (0.6238 pp) it would have **passed**, and I
would have reported a "0.66 pp neighbour gap on mmlu_content" that is **an artefact of
taking a max-minus-min over 8 noisy cells** — the exact error that guard exists to
prevent, committed in the guard's own name.

The estimator is therefore Monte-Carlo'd for the actual *k* and **validated against the
closed forms** before use: `c_2` recovered to 4.1 × 10⁻⁵ relative error, `c_3` to
2.9 × 10⁻⁴. The run **aborts** if that check fails, because a mis-estimated noise floor is
worse than no gate — it looks rigorous.

### 1.4 ⛔ MY OWN BUG #2 — the Q3 flag OR'd the weak read over the pre-registered headline

The prereg says, verbatim, of the length-matched window: *"Its popqa Δacc + CI + p is the
**headline** Q3 number"*, and calls the any-interval scan *"the **weaker** but more honest
question."*

My first implementation computed `REPLICATES = span_is_regression OR
(n_resolved_regressions > 0)`. On this data those two disagree: the headline window is a
resolved **improvement** (+1.3317 pp), while the scan does find 2 resolved regressions
among 7 intervals. The OR therefore printed **`Q3_REPLICATES`** — a conclusion drawn from
the weaker read, **against** the designated headline. That is the "pick the favourable
criterion" failure this protocol forbids elsewhere (it is why adjacent-interval tests use
a conservative AND). **Fixed to read (a) alone; the verdict string flipped to
`Q3_DOES_NOT_REPLICATE`.** Read (b) is still reported, beside a note on why it cannot set
the flag: with 21 decision-axis tests ~1 resolved move is expected under a global null,
and here the resolved moves go **both ways** (3 improvements vs 2 regressions), so "a
regression exists somewhere" is not evidence of keep14's *directional* phenomenon.

### 1.5 ⛔ MY OWN BUG #3 — the seed-disjointness check aborted on its own output

Minor but worth recording because it is the failure mode that gets a safety check deleted:
on re-run, `assert_seeds_disjoint` found `arm_index` 600–610 in
`a04_keep12_trajectory_monotonicity.json` — **its own file, about to be overwritten** —
and refused to proceed. Fixed by excluding the output basename explicitly (and recording
the exclusion in the JSON), not by weakening the check.

### 1.6 ⚠️ Two premises that hold but bound the reading

- **keep12 has no step128000.** Checkpoint rotation removed it (`keep_last_n=3`,
  `keep_steps=[83500, 121000, 124000, 150000, 175000, 200000]`, `milestone_every=5000`).
  So Q3 is **length-matched (25 000 vs 25 500 steps), never step-matched**, and is a
  replication of a *phenomenon*, not a matched pairwise comparison.
- **The learning rate is decaying across the grid**, 7.694 × 10⁻⁶ at the resume to
  3.26 × 10⁻⁶ by step 165900 (cosine to `max_steps=200000`). Later grid points are taken
  at a smaller LR. **This is not controlled for**, and it is a plausible mechanism for
  "excursions shrink as the grid progresses" — which is what the data show.

---

## 2. The 8-point curve (`split` convention, pre-registered)

Anchor: vanilla `models/OLMo-2-1124-7B` (`base_full`, `base_full_nqopen`, `7B_base`),
**imported** from `a04_shallow_rung_ni_7b.ANCHOR`, never redeclared, never substituted
(G0/G2). Nulls from **imported `build_nulls`** — not hand-computed, because MAIN's own
subtraction of a recorded null was ~0.5 pp off twice
(`A04_FULL32_READING_B_IS_FIRING.md`). `Δ = 0.10 × residual(intact)`, never substituted.

The anchor reproduces the archive exactly: intact residuals **63.2914 / 22.4574 /
18.6138 pp** → **Δ = 6.3291 / 2.2457 / 1.8614 pp**, identical to
`a04_shallow_rung_ni_7b`. All four axes **CERTIFIABLE** under D1–D6; **0 of 44 cells
retired** on the decision axes.

| axis | 130000 | 135000 | 140000 | 145000 | 150000 | 155000 | 160000 | 165000 |
|---|---|---|---|---|---|---|---|---|
| **triviaqa** margin pp | −35.0719 | −32.9932 | −34.3031 | −32.8650 | −32.2579 | −31.4941 | −31.4941 | **−31.1764** |
| acc % | 22.782 | 24.877 | 23.557 | 24.994 | 25.596 | 26.377 | 26.354 | 26.677 |
| recovery % | 35.6 | 38.9 | 36.8 | 39.1 | 40.0 | 41.3 | 41.2 | 41.7 |
| **popqa** margin pp | −18.6486 | −17.7795 | −18.6276 | −18.2912 | −17.4921 | −17.3099 | −17.5061 | **−17.4571** |
| acc % | 4.423 | 5.285 | 4.437 | 4.766 | 5.586 | 5.755 | 5.565 | 5.600 |
| recovery % | 9.5 | 13.3 | 9.6 | 11.0 | 14.7 | 15.4 | 14.6 | 14.7 |
| **mmlu_content** margin pp | −9.1200 | −9.1556 | −9.3478 | −9.3265 | −9.1342 | −9.1128 | −8.6856 | **−8.7354** |
| acc % | 36.704 | 36.647 | 36.455 | 36.469 | 36.676 | 36.697 | 37.110 | 37.053 |
| recovery % | 44.4 | 44.1 | 43.0 | 43.1 | 44.2 | 44.3 | 46.6 | 46.2 |
| *nq_open (demoted)* margin pp | −14.7091 | −14.2936 | −14.9030 | −14.7368 | −14.4598 | −14.3490 | −14.4598 | −14.1551 |

**Every checkpoint rejects on every axis**: 0 of 33 decision cells accept, identical under
all five tie conventions (`split`/`first`/`last`/`wrong`, and `credit` where MMLU retires
to 22 cells). The off-grid resume anchor step124000 (reported, **excluded from the
trend** because it is 6 000 steps below 130000 and would break uniformity) sits at
−33.3944 / −18.4804 / −9.5116 pp.

Note the arm is **far** from certifiable — recovery 9.5–46.6 %, margins 8.7–35.1 pp
short. Nothing here is a near-accept, so no accept-boundary claim is made or possible.

---

## 3. Q1 — monotonicity. The answer is "monotone in trend, wobbly point-to-point, and never wandering"

Verdict labels and thresholds were fixed in the prereg and are generated **mechanically**
by the script; I did not choose them after seeing the table.

| axis | verdict | strictly monotone? | sign reversals | Spearman ρ | p (perm) | OLS pp/1k | R² |
|---|---|---|---|---|---|---|---|
| **triviaqa** | **MONOTONE_TREND** | no | 2 | **+0.9701** | **0.0004** | +0.10428 | 0.831 |
| **popqa** | **MONOTONE_TREND** | no | 4 | **+0.8095** | **0.0213** | +0.03443 | 0.586 |
| **mmlu_content** | **UNRESOLVED** | no | 2 | +0.6667 | 0.0818 | +0.01414 | 0.502 |
| *nq_open (demoted)* | *UNRESOLVED* | no | 4 | +0.4791 | 0.2288 | +0.01187 | 0.329 |

**Tally: 3 decision axes → 2 MONOTONE_TREND (both improving), 1 UNRESOLVED, 0 WANDER.**

This is the pre-registered `ge2_axes_monotone_improving_none_wander` branch, and it fired:
*"P IS NARROWED HARD. Non-monotonicity would then be a property of keep14/keep8/full32 and
NOT of healing at this damage level."*

### 3.1 The honest complication: point-to-point excursions ARE material

P is not simply wrong; it is **wrong about the direction and right about the amplitude**.

| axis | range pp | noise floor `c_8·σ̄` pp | clears gate? | max \|Δmargin\| pp | ÷ Δ | amplitude ≥ 0.25 Δ? |
|---|---|---|---|---|---|---|
| triviaqa | **3.8955** | 1.1084 | **YES (3.51×)** | **2.0787** | **0.328** | **yes** |
| popqa | **1.3388** | 0.9737 | **YES (1.37×)** | **0.8691** | **0.387** | **yes** |
| mmlu_content | 0.6623 | 1.0494 | **no** | 0.4273 | 0.230 | no |
| *nq_open* | 0.7479 | 1.8942 | no | 0.6094 | 0.306 | (demoted) |

So on the two axes where anything is measurable, **a single 5 000-step interval can move
the margin by a third of Δ**, and the total range over the grid is **62 % of triviaqa's Δ
and 60 % of popqa's**. The clearest single event is **triviaqa 135000→140000: −1.3208 pp
of accuracy, p = 0.0001, 594 items right→wrong vs 357 wrong→right of 17 944** — a
resolved backward step in the middle of a strongly upward trend.

**That is why the §2.0.2 neighbour precondition survives this result even though P's
central claim does not.** A hand-picked checkpoint on this arm can still overstate a
margin by ~1–2 pp. What it *cannot* do is make the trajectory look like it is going
nowhere: 12 of 21 decision-axis intervals are resolved under BH (q = 0.05), and on
triviaqa **6 of 7** are resolved, **5 of them upward**.

### 3.2 mmlu_content: the axis that "moves" only if you use the wrong constant

mmlu_content is the sharpest methodological case in this pass. Its range (0.6623 pp) is
**0.63× the noise floor** for an 8-point range — so *no detectable diffusion*. Only
**1 of 7** intervals is resolved (155000→160000, +0.4130 pp, p = 0.0072), and its Spearman
p = 0.0818 misses the trend criterion. Total accuracy movement across 35 000 steps is
36.704 % → 37.053 %.

This axis also confirms the keep14 finding it was tested against: keep14's mmlu_content
had **plateaued** (+0.0071 pp over its last 46 500 steps). keep12's mmlu_content is doing
the same thing at a different absolute level — the axis that looks closest to accepting is
the one that is moving least.

---

## 4. Q3 — keep14's popqa regression FAILS to replicate, and now it is 0 for 2

| | keep14fresh2 (archived) | shortgpt16 (`neighbour_variability`) | **keep12fresh2 (here)** |
|---|---|---|---|
| interval | 128000→153500 (25 500 steps) | 128000→153500 | **130000→155000 (25 000 steps)** |
| popqa Δacc | **−0.6729 pp** | +0.0841 pp | **+1.3317 pp** |
| CI95 | [−0.9252, −0.4206] | [−0.1542, +0.3224] | **[+1.1075, +1.5630]** |
| p | 0.0001 | 0.5084 | **0.0001** |
| flips (w→r / r→w) | 122 / 218 | 152 / 140 | **231 / 41** |
| resolved regression? | **yes** | no | **no — resolved IMPROVEMENT** |

keep12 does not merely fail to regress; it improves **twice as much in the opposite
direction as keep14 regressed**, resolved at p = 0.0001 with 231 items going wrong→right
against 41 the other way.

**The popqa mid-heal dip is now 0 for 2 on independent arms.** Combined with §3's
verdicts, keep14's popqa regression should be cited **only** as an existence proof about
checkpoint selection — never as evidence about healing. Anyone extending it to a general
property is over-reaching on 1 of 3 arms.

For completeness (read (b), which does **not** set the flag): keep12's popqa has **2 of 7**
resolved adjacent regressions (135000→140000, −0.8481 pp, p = 0.0001; 155000→160000,
−0.1892 pp, p = 0.0230) and **3 of 7** resolved improvements. Resolved moves go both ways,
which is what a wobbly-but-rising series looks like.

---

## 5. Q4 — the 500-step neighbour range does not reproduce here, and with keep10 that makes THREE arms whose ranges rank-order by learning rate

Same `k = 3` convention as `A04_NEIGHBOUR_VARIABILITY_VERDICT.md` §2.3, so the arms are
directly comparable. This triple is inside the same single process (§1.1), so unlike
keep8's cluster 1 there is **no seam caveat at all**.

| axis | margins at 165000 / 165500 / 166000 pp | range | `1.6926·σ̄` | clears gate? |
|---|---|---|---|---|
| triviaqa | −31.1764 / −30.9814 / −30.9981 | **0.1951** | 0.6516 | **no (0.30×)** |
| popqa | −17.4571 / −17.4921 / −17.6323 | 0.1752 | 0.5816 | no (0.30×) |
| mmlu_content | −8.7354 / −8.9352 / −8.7140 | 0.2211 | 0.6130 | no (0.36×) |
| *nq_open* | −14.1551 / −14.5152 / −14.2936 | 0.3601 | 1.1493 | no (0.31×) |

**0 of 3 decision axes clear the gate.** keep8's headline — a **1.1202 pp** triviaqa range
over 500 steps — does **not** reproduce on keep12, where the same axis over the same
spacing moves **0.1951 pp**, less than a third of its own noise floor.

### 5.1 ⚠️ This must be read against `keep10_neighbour_range_20260813`, which ran concurrently and DID replicate

While this scan was on `.73`, a **keep10+fresh2** 500-step triple (89000/89500/90000) was
scored on `.82` and **replicated keep8** — triviaqa range **1.2149 pp**, 1.84× its floor,
within 8 % of keep8's 1.1202 pp, with all four gate booleans agreeing. Its verdict
explicitly recommended *"reading the keep12 trajectory for the shape question."* So the
correct summary is **not** "the neighbour range is a keep8 artefact." It is:

| arm | keep_front | 500-step triple | LR at the triple | **triviaqa range pp** | floor pp | clears? |
|---|---|---|---|---|---|---|
| **keep10** | 10 | 89000–90000 | **1.24 × 10⁻⁵** | **1.2149** | 0.6595 | **yes (1.84×)** |
| **keep8** | 8 | 130000–131000 | **6.80 × 10⁻⁶** | **1.1202** | 0.6577 | **yes (1.70×)** |
| **keep12** | 12 | 165000–166000 | **3.25 × 10⁻⁶** | **0.1951** | 0.6516 | **no (0.30×)** |

**The three arms rank-order identically by learning rate and by neighbour range.** The
noise floors are nearly identical across the three (0.6516–0.6595 pp), so this is not an
artefact of differing item noise. keep12's LR is **2.09× lower** than keep8's and its range
is **5.74× smaller**; keep10's is **1.82× higher** and its range is **1.08× larger**.

**This reframes the whole neighbour-range question.** The available evidence is at least as
consistent with *"checkpoint-to-checkpoint margin scatter is governed by where you are on
the LR schedule"* as with *"it is governed by damage depth."* Note the depth ordering is
**not** monotone with range (keep10 > keep8 > keep12 by range, but keep_front 10 < 8 is
inverted relative to 12), whereas the LR ordering is exact.

**Stated with its limits, because n = 3 arms:** this is a **hypothesis generated by three
points**, not a fitted relationship. Three arms cannot separate LR from step-count,
epoch position, or depth — keep12's triple is also its *latest* triple in absolute steps.
It is, however, a **cheap and falsifiable** prediction: score a 500-step triple **early**
in keep12's own schedule (high LR, same arm, same depth, same corpus) and the range should
be large. If it is, the effect is the schedule, not the arm. **That single test would
replace all further 3-point arm-adding**, which
`keep10_neighbour_range_20260813.consequences_for_the_gate[4]` already declined to fund.

**Consequence for §2.0.2:** the precondition stays, and its per-axis phrasing is now
confirmed on three arms (the same one axis clears, or fails, and the same three never
clear). But its **tolerance is not a constant** — triviaqa's 500-step range spans
0.1951–1.2149 pp across three arms, a **6.2× spread**, and the low end is *this* arm. So
§2.5's "≈1.2 pp on triviaqa" is an **upper** bound observed at moderate-to-high LR, not a
universal figure, and the defensible form of the rule remains *"report the neighbours"*
rather than *"clear a fixed number."*

---

## 6. Verification performed

1. **Protocol confirmed from the invocation, fail-closed, and RE-VERIFIED for this arm
   rather than inherited.** `summary.json:meta` records **neither `batch_size` nor
   `chat_template`** (`A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md`), so both come from the
   driver's own echoed lines: `DRIVER START … mmlu_bs=16 cb_bs=32`, per-axis
   `closedbook/nq_open START … bs=32`, `mmlu START … bs=16`, `DRIVER END rc=0`. Driver
   source defaults corroborate. **Negative-tested on the compute node:** a doctored log
   reading `cb_bs=48` → `FATAL protocol deviation`; a missing log → `FATAL: driver log …
   absent`; the clean log passes. In both failure cases **no output file is written**.
2. **`add_bos is False` on all 33 result dirs**, asserted with **`is False`** — never
   `is not True`, which passes silently on `None`. `max_new_tokens == 32` asserted on all
   22 generative dirs. **`ckpt_step` asserted equal to the requested step on all 33 dirs**,
   so a mislabelled result dir cannot enter the curve. `chat_template=False` established
   **structurally**: neither harness contains a chat-template code path — the only
   occurrence of the string is a docstring — so it cannot have been switched on.
3. **Shard integrity: 44 of 44 cells clean.** Shard index set **exactly {0..7}** (a set,
   not a file count), merged *n* exactly `EXPECTED_N` (triviaqa 17944 / popqa 14267 /
   nq_open 3610 / mmlu 14042), **0 duplicate `item_id`, 0 `nan`**, identical `item_id`
   sequences across all 12 arms. **Negative-tested on the compute node:** a hand-built
   7-of-8 popqa set is refused with `shard index set [0,1,2,4,5,6,7] != {0..7}`, and a
   duplicated shard is refused with `1783 duplicate item_ids`.
4. **The statistics are self-tested in-process, every run**, and the run aborts before
   writing anything if any case fails: monotone increasing/decreasing give ρ = ±1;
   tie handling matches a hand-computed value to 10⁻¹²; **a symmetric V-shaped wander
   gives ρ = 0.0000 and must NOT satisfy the trend criterion** (if it did, the classifier
   would mislabel exactly the phenomenon P describes); OLS recovers a known line. All
   passed. The Spearman p is a **permutation** p (20 000 draws, n = 8 → a t-approximation
   would be the wrong instrument at this n); minimum attainable p = 5 × 10⁻⁵.
5. **Noise-floor estimator validated against closed forms** before being used to gate
   anything (§1.3): `c_2` to 4.1 × 10⁻⁵, `c_3` to 2.9 × 10⁻⁴ relative error.
6. **Seed disjointness EXECUTED, not asserted — and it caught a real collision.** The
   check reads every archive's own recorded offsets and intersects them. My first choice
   (`arm_index` 500.., guard 2700, interval 2900) collided **exactly** with
   `a04_full32_trajectory_ni.json`, written the **same day at 12:33** and therefore absent
   from every prose disjointness list in the repo. Moved to **600.. / 3700 / 3900**. Had
   the check been a comment, re-running the full32 archive would later have produced
   different numbers for reasons unrelated to science.
7. **Cross-node SCORING determinism control — a claim A04 had asserted but never tested.**
   Prior verdicts state that scoring is deterministic and only the *analysis* bootstrap
   drifts across numpy versions, but nobody had verified that for the **GPU harness across
   two nodes**. `.82` re-scored step150000 with the identical driver: **0 item
   disagreements out of 17944 / 14267 / 14042 / 3610, all four axes bit-identical,
   acc_diff = 0.000000 pp**. The claim is now measured, not assumed. This control enters
   **no** statistic.
8. **All statistics on ONE node** (`.73`, numpy **2.5.1**), because
   `Generator.multinomial` differs in 19 of 10 000 rows between 2.5.1 and 2.4.6
   (`A04_NEIGHBOUR_VARIABILITY_VERDICT.md` §4.1). No comparison here mixes samplers.
9. **A free bs8 → bs32 sensitivity measurement, and it is reassuring.** The archived
   `7B_keep12_step124000_v2` dirs were scored 2026-08-08 at **`--batch_size 8` on both
   axes** by `scripts/_run_olmo2_p24_eval_ladder_prev2_73.sh`; this dispatch re-scored the
   **same** `step124000.pt` at cb=32 / mmlu=16 with the same harness md5s and
   `add_bos=false`. Only batch size differs. Result: **+0.0223 pp triviaqa (104/17944
   items differ), +0.0280 pp popqa (20/14267), +0.0499 pp mmlu_content (87/14042)**.
   A04's only prior datum was bs32→bs48 (12/14267 popqa); **bs8→bs32 is a 4× wider gap and
   still moves accuracy by ≤ 0.05 pp** — an order of magnitude below every finding here.
   Labelled diagnostic; enters no verdict.
10. **Everything imported, nothing reimplemented.** `ni_rule`, `ratio_rule`,
    `load_shards`, `build_nulls`, `mmlu_content_norm_vec`, `qa_metric_vec`, `EXPECTED_N`,
    `AXES`, `DEMOTED_AXES`, `PREREG` from `pilot_zero_rule_disagreement`;
    `paired_bootstrap`, `bh_reject`, `TIE_CONVS`, `N_BOOT`, `SEED` from A03's
    `analyze_1b_knowledge_floor`; `ANCHOR`, `_load_arm`, `assert_aligned`,
    `d4_interface_degenerate`, `D2_RESIDUAL_FLOOR_PP`, `Z95_TWO_SIDED`, `D4_*` from
    `a04_shallow_rung_ni_7b`. No metric, null, rule, guard or anchor re-derived.
11. **Not output degeneracy.** Labelled diagnostic across the grid: popqa and triviaqa hold
    **0.000 % empty predictions** throughout, top-constant share stays low, distinct
    predictions do not collapse. The models are changing answers, not breaking format.
12. **Verdict is convention-invariant.** 0 NI accepts across all 5 tie conventions × 44
    cells. `RATIO(0.85)` also rejects every checkpoint.

---

## 7. Licensed vs NOT licensed

### Licensed
- The 11 checkpoints' accuracies, nulls, residuals, Δ, lo95 bounds and margins in §2, and
  the 0-accept verdict under all five tie conventions and RATIO(0.85).
- **"On keep12+fresh2, across 8 checkpoints spanning 35 000 steps of one uninterrupted
  heal process, the NI margin trend is monotone-improving on 2 of 3 decision axes
  (ρ = +0.9701 p = 0.0004; ρ = +0.8095 p = 0.0213), UNRESOLVED on the third, and WANDERING
  on none."**
- **"The margin nonetheless moves non-monotonically point to point, by up to 0.328× Δ
  (triviaqa) and 0.387× Δ (popqa) in a single 5 000-step interval, with a resolved
  backward step of −1.3208 pp accuracy (p = 0.0001, 594 right→wrong of 17 944)."**
- **"keep14's popqa 128000→153500 regression does NOT replicate on keep12: the
  length-matched window is a resolved IMPROVEMENT of +1.3317 pp (p = 0.0001, 231 w→r vs
  41 r→w), the opposite sign. With shortgpt16 also failing it, the dip is 0 for 2 on
  independent arms."**
- **"keep8's 1.1202 pp 500-step triviaqa neighbour range does NOT reproduce on keep12
  (0.1951 pp, 0.30× its own noise floor); 0 of 3 decision axes clear the gate."** Read
  together with `keep10_neighbour_range_20260813` (which DID replicate keep8 at
  1.2149 pp), the three arms' triviaqa ranges **rank-order exactly by the learning rate at
  the triple**, at near-identical noise floors (0.6516–0.6595 pp) — reported as a
  **hypothesis from n = 3 arms**, not a fitted relationship.
- **"mmlu_content shows no detectable diffusion over 35 000 steps (range 0.6623 pp vs a
  1.0494 pp floor), and only 1 of 7 intervals resolves."**
- **"Scoring is bit-identical across `.73` and `.82` for a fixed checkpoint"** (0/49 863
  item disagreements, 4 axes) — previously asserted, now measured.
- **"bs8 → bs32 moves accuracy by ≤ 0.0499 pp on identical weights"** (104/17944,
  20/14267, 87/14042 items).

### NOT licensed
- ⛔ **Claim P as a general law.** This is the point of the pass. On the only densely
  sampled damaged arm, the trend is upward on every axis where a trend is resolvable, and
  0 of 3 axes wander. P may be stated only as **arm-specific**.
- ⛔ **The converse over-correction: "recovery is monotone."** Two axes are *trend*
  monotone, **not strictly monotone** (2 and 4 sign reversals), and one axis resolves
  nothing at all. The excursions are real and material relative to Δ.
- ⛔ **Using the k=3 constant 1.6926 to gate an 8-point range.** `c_8 = 2.8475`; the k=3
  constant puts the floor 40.6 % too low and would have converted mmlu_content's
  sub-noise 0.6623 pp into a reported "gap". Constants are in
  `noise_floor_constants`.
- ⛔ **Reporting the 5 sub-noise ranges** (mmlu_content 0.6623, nq_open 0.7479, and all
  three Q4 decision ranges) **as measured gaps.** They fail
  `range_exceeds_item_noise`; a max-minus-min of *k* noisy cells is biased upward even at
  zero true spread.
- ⛔ **Setting Q3 from read (b).** The prereg designates the length-matched window as the
  headline; 2 resolved regressions among 7 intervals, with 3 resolved improvements
  alongside and ~1 expected under a global null, is not evidence of a directional
  phenomenon. See §1.4 — I made this error and corrected it.
- ⛔ **Treating the 8 grid checkpoints as replicates.** They are successive states of ONE
  optimisation at a **decaying learning rate** (7.694e-06 → 3.26e-06); their spread is
  heal progress + data order + LR schedule, **not** independent-run variance. The LR decay
  is uncontrolled and is a live alternative explanation for the shrinking excursions.
- ⛔ **Asserting the LR-vs-range relationship of §5.1 as CAUSAL or fitted.** It is a rank
  ordering over **n = 3 arms**, which cannot separate LR from step count, epoch position or
  depth (keep12's triple is also its latest in absolute steps). It is a hypothesis with a
  named falsification test — an early, high-LR triple **on keep12 itself** — and must be
  quoted as such. Depth, notably, does **not** order the ranges monotonically.
- ⛔ **"The neighbour range is a keep8 artefact."** `keep10_neighbour_range_20260813`
  reproduced it at 1.2149 pp (1.84× floor) on an independently damaged arm. keep12 not
  reproducing it makes the effect **conditional**, not spurious.
- ⛔ **Anything of the form "the 7B deficits are large relative to seed variance."**
  `sd_run` is a **1B-only** quantity. Every 7B rung has exactly **one** seed and the
  historical 7B ladder's seeds are unrecorded, so no 7B `sd_run` is computable.
- ⛔ **Calling any of this "harness noise."** There is no measured runtime-jitter floor on
  this harness, and §6.7 now shows scoring is bit-identical across nodes. Item-sampling
  variability is a different thing and **is** quantified.
- ⛔ **Comparing keep12 / keep14 / keep8 / shortgpt16 margins as rungs of one ladder.**
  Four different architectures (14 / 16 / 10 / 16 layers; 157 / 179 / 113 / 179 tensors);
  the two-corpora / unequal-steps `STATUS.json:warning` still applies.
- ⛔ **Reading Q3 as step-matched.** keep12 has no step128000; the window is
  length-matched (25 000 vs 25 500) only, and "same step" ≠ "same data seen" across arms.
- ⛔ **Any accept-boundary claim.** Every cell rejects by 8.7–35.1 pp at 9.5–46.6 %
  recovery. Nothing here is near the boundary.
- ⛔ Any K1/K2/K3 clause — defined over the pre-registered **1B** arm set.
- ⛔ **Quoting any margin here to better than 0.01 pp across nodes** (numpy multinomial
  drift; all statistics here are single-node).

---

## 8. What this changes

1. **The proposed `paperD` thesis, as written, is refuted on its best test case.** P said
   margins *wander* non-monotonically. On the only damaged arm with a dense uniform grid,
   they **rise**, resolvedly, on every axis where anything resolves. A paper built on
   "recovery margins wander" would be contradicted by its own arm #3. **Do not promote P
   to `paperD` in its current form.**
2. **What survives is narrower, true, and still worth writing.** Three findings hold up
   and are mutually consistent:
   - **Single-interval excursions reach ~⅓ of Δ even inside a strongly monotone trend**
     (triviaqa 0.328 Δ, popqa 0.387 Δ; a resolved −1.32 pp step mid-climb). So a
     single-checkpoint margin **is** perturbable by ~1–2 pp on this arm — the §2.0.2
     neighbour precondition is justified **without** needing P's directional claim.
   - **Both of P's headline phenomena are conditional, not general.** keep14's popqa dip
     is **0 for 2** on independent arms (shortgpt16, keep12). keep8's 500-step triviaqa
     range is **1 for 2** (keep10 reproduced it, keep12 did not) — and §5.1 supplies a
     candidate condition: the three arms' ranges rank-order exactly by LR at the triple.
     A methodology paper can legitimately say *"these effects exist, are conditional, and
     one plausible condition is schedule position, so certify with neighbours reported"* —
     it cannot say *"margins wander in general."*
   - **The axis nearest the threshold is the axis that moves least** — keep14's
     mmlu_content plateau reproduces on keep12 as *no detectable diffusion at all*.
     Apparent proximity to accepting is not evidence of approach. This replicates.
3. **A concrete methodological error is now documented with a counterfactual.** The
   `E[range of k]` gate must use the actual *k*. Reusing keep8's k=3 constant on 8 points
   would have manufactured a 0.66 pp mmlu_content "gap" from pure noise (§1.3). Any future
   pass that quotes 1.6926 on a series longer than 3 points is wrong, and
   `A04_NEIGHBOUR_VARIABILITY_VERDICT.md` §2.3 should be read as k-specific.
4. **Two infrastructure facts A04 had been assuming are now measured.** Cross-node GPU
   scoring is **bit-identical** (0/49 863 items), and **bs8→bs32 moves accuracy ≤ 0.05 pp**
   on identical weights. The first licenses splitting future scoring across nodes; the
   second bounds the slop in every A04 comparison that mixes bs8-era archive dirs with
   bs32-era ones. Neither had a number before.
5. **§2.0.2's tolerance is NOT a constant, and §2.5 should say so.** triviaqa's 500-step
   range spans **0.1951–1.2149 pp across three arms — a 6.2× spread** — at near-identical
   noise floors. `keep10_neighbour_range_20260813` recommended amending §2.5 to *"cite two
   arms with the tolerance unchanged in value"*; **keep12 makes that amendment wrong as
   stated.** Two arms agreeing at ~1.2 pp looked like convergence; the third shows ~1.2 pp
   is an **upper bound observed at moderate-to-high LR**, not a universal tolerance. The
   precondition's defensible form is *"report the neighbours"* (which is what §2.0.2
   already says), not *"clear 1.2 pp"*.
6. **The next experiment is now obvious, cheap, and falsifying rather than confirming.**
   §5.1's LR ordering makes a sharp prediction: an **early, high-LR 500-step triple on
   keep12 itself** should show a *large* triviaqa range. Same arm, same depth, same corpus,
   same tokeniser — the only thing that changes is schedule position, so it separates LR
   from depth in a way that no amount of arm-adding can. ~3.5 GPU-h by keep10's measured
   rate. If the range is large, checkpoint scatter is a **schedule** phenomenon and the
   gate's tolerance should be conditioned on LR rather than on the arm; if it stays small,
   the effect really is arm-specific and keep12 is simply quiet.

**Recommendation.** Do **not** spend GPU on a fourth arm to rescue P's directional claim —
it has now failed on two independent arms, and `keep10_neighbour_range_20260813` had
already declined to fund a fourth 3-point cluster for the *range* question. Spend the
~3.5 GPU-h on the **within-arm LR probe** in item 6 instead: it is the only cheap test that
can discriminate the two live explanations, and it can falsify the interpretation I am
offering rather than decorate it.

If the methodology direction is pursued, its thesis should be **"certification margins on
damaged arms carry checkpoint-selection uncertainty of order ⅓ Δ per save interval, the
size is conditional (plausibly on schedule position), and the direction of travel is
upward"** — supported by keep8 + keep10 + keep14 + keep12 together, needing no new compute,
and not depending on P's claim that margins wander.
