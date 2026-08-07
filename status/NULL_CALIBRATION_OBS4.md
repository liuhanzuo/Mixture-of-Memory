# Null-Calibration — the self-falsification case (Paper E Obs4), task #170

Owner: subagent dispatched 2026-08-07. Budget: **0 GPU-hours** (pure CPU recompute on
existing per-example jsonl). This file is the only status doc this agent may write;
`status/NULL_CALIBRATION_P1.md` is another agent's deliverable and was not touched.

Closes the one open row of `status/NULL_CALIBRATION_P1.md` §9:
*"self-falsification case (Paper E Obs4) — ⚠️ NOT yet written up here."*

Regenerator: `scripts/build_null_calibration_table.py`, new leg `leg_obs4()` = **row C5**.
Run log `logs/null_calibration_obs4.log`, machine-readable dump
`results/null_calibration/null_calibration_obs4_nperm2000.json` (key
`c5_obs4_ours_retracted`).

```bash
python3 scripts/build_null_calibration_table.py --n-perm 2000 --seed 0 --n-boot 10000 \
    --out results/null_calibration/null_calibration_obs4_nperm2000.json
```

**Regression gate: C1–C4 are bit-identical to P1's dump** (all numeric fields agree to
<1e-12 across `c1_mc`, `c1_arm_acc`, `c2_squad`, `c3_cka`, `c4_probe`, `table`,
`gate_span`, `gate_c4_variants`, `gate_pass`). Adding C5 changed nothing upstream.

---

## 1. Where Obs4 is recorded, and what it claimed

**The record exists.** It was not reconstructed. Three independent places:

| location | content |
|---|---|
| `paperE_research/MAIN_verified_interface_validity.md:80-93` | §1 "**Obs 4 ✗ 被我自己推翻**：排序翻转只发生在仪器已失效处" — the retraction itself, with the two flip pairs and their CIs |
| `paperE_research/MAIN_verified_interface_validity.md:206` | §4 status table: "~~interface 翻转模型排序~~ ✗ **MAIN 自己推翻**：0 个翻转发生在双口径有效的 arm 之间" |
| `status/TRAINER_ACTIVITY.jsonl:6078` (`2026-08-06T15:14:41Z`) | `"Obs4_ranking_flip": "arXiv:2402.01781 Alzahrani (ACL 2024) + attack-3-self-refuted (Obs4 pair both at chance floor)"` |
| `UPDATELOG.md:5927` (2026-08-06 23:15) | "Obs4 (ranking flip) 被 Alzahrani arXiv:2402.01781 (ACL 2024) 占；且 workflow attack-4 (FATAL) 独立发现原 flip pair 都在 chance floor 上——自己也把它推翻了" |

**The claim.** Obs4 asserted that **the MC scoring interface flips the model ranking** —
i.e. two arms rank one way under the *letter* interface (score the tokens "A"/"B"/"C"/"D")
and the opposite way under the *content* interface (score the option strings,
length-normalised), so the choice of a supposedly innocuous scoring convention decides
which model is "better".

- **Arms**: 10 OLMo-2-7B arms → C(10,2) = **45 arm-pairs** screened.
  Note this is 10 arms, not C1's 9: `keep14-reheal @67.5k` is a flip arm and C1 omits it.
- **Metric**: MMLU accuracy, `letter` vs `content_norm`, n = **14,042** items, one shared
  item set (verified item-by-item: `item_id` and `gold_letter` sequences identical across
  all 10 arms, `nan` count 0 in every arm).
- **Direction of the reported flip**: 7/45 pairs sign-opposite; **2 significant on both
  interfaces**, and the headline one was framed as the "inherited vs random-init" control:

  | pair | letter | content_norm |
  |---|---|---|
  | keep10 @83.5k − scratch16L @200k | **+2.51pp** [+1.47,+3.53] | **−1.53pp** [−2.23,−0.84] |
  | keep10 @83.5k − keep14-reheal @67.5k | **+2.29pp** [+1.16,+3.38] | **−2.13pp** [−2.73,−1.52] |

  So letter says keep10 **beats** scratch16L; content says keep10 **loses** to it.

- **Retraction reason as recorded**: the flip only occurs where the letter interface has
  already collapsed onto a constant predictor. Restricted to arms significantly above their
  floor on *both* interfaces, significant flips = 0.

**All of the above reproduces from raw per-example jsonl with zero drift** from the
`MAIN_verified_interface_validity.md` record (which itself was MAIN-recomputed, not
subagent-relayed). The 10-arm letter/content table reproduces to 4 d.p.

---

## 2. Per-arm absolute scores, n = 14,042 every row

Source: `olmo2_mmlu_content_results/<arm>/per_example_mmlu.jsonl`.
**Flip arms marked ◀.** Letter null = best constant letter **always-D = 0.2689**
(gold marginals A .2295 / B .2465 / C .2551 / **D .2689**).
Content null = longest-option, **tie convention `split` = 0.2845**.

| arm | letter | vs always-D | boot p | McNemar p | letter verdict | content_norm | vs longest-option |
|---|---|---|---|---|---|---|---|
| base (32L intact) | .6054 | +33.65 | 0.0001 | <1e-300 | above floor | .4706 | +18.61 |
| full32 @25k | .5877 | +31.88 | 0.0001 | <1e-300 | above floor | .4662 | +18.18 |
| keep8 @121k | .2550 | −1.39 | 0.0204 | 1.76e-02 | **BELOW floor (sig)** | .3423 | +5.78 |
| **keep10 @83.5k ◀** | **.2720** | **+0.31** | **0.4214** | 4.22e-01 | **AT floor (indistinguishable)** | **.3445** | **+6.00** |
| keep12 @124k | .2728 | +0.38 | 0.3792 | 3.78e-01 | AT floor | .3629 | +7.85 |
| keep14 @200k | .3184 | +4.95 | 0.0001 | 7.78e-17 | above floor | .3832 | +9.88 |
| freezefront @200k | .2624 | −0.66 | 0.2426 | 2.37e-01 | AT floor | .3604 | +7.60 |
| **scratch16L @200k ◀** | **.2470** | **−2.19** | **0.0001** | 1.27e-04 | **BELOW floor (sig)** | **.3598** | **+7.53** |
| shortgpt16 @200k | .4742 | +20.53 | 0.0001 | 1.18e-272 | above floor | .4012 | +11.67 |
| **keep14-reheal @67.5k ◀** | **.2492** | **−1.97** | **0.0008** | 1.13e-03 | **BELOW floor (sig)** | **.3658** | **+8.13** |

`letter` and `content_norm` are the fraction of the 14,042 items scored correct.
"vs null" columns are pp, from a **paired** bootstrap on the per-item difference vector.
Bootstrap p is floored at 1/n_boot = 0.0001 (an exact 0 is unattainable from 10,000
resamples); the two `<1e-300` McNemar values underflow float64 and print as `0.0` in the
JSON dump.

---

## 3. The null, and the tie convention (stated because it is load-bearing)

**Letter interface → best constant letter, always-D = 0.2689** (n=14,042). Pre-registered
as the *best* constant, not each arm's own modal letter — see §5 check 2 for why that
choice changes signs.

**Content interface → longest-option heuristic, tie convention `split` = 0.2845.**
**4,805 / 14,042 = 34.22%** of items have ≥2 maximal-length options, exactly as P1
established, so the convention is load-bearing. All five conventions, recomputed here:

| convention | value |
|---|---|
| **split** (fractional credit; pre-registered) | **0.2845** |
| first-of-maximal | 0.2811 |
| last-of-maximal (the `.2822` in `paperE_research/`) | 0.2822 |
| credit (optimistic, any tie = hit) | 0.4537 |
| wrong (pessimistic, any tie = miss) | 0.1961 |

Matches P1 exactly (.2845 / .2811 / .2822 / .4537 / .1961). **Quote .2845 and print the
convention.**

---

## 4. Both significance tests — the contrast IS the finding

**Test A — arm vs arm. The flip is REAL and survives multiplicity correction.**
Paired bootstrap (10,000 resamples, item-level) + exact-binomial McNemar; then
Benjamini-Hochberg q=0.05 applied across the **whole** 45-pair × 2-interface screen,
because the flip was *found* by screening 45 pairs and an uncorrected screen is precisely
the error this paper is about.

| | letter | content |
|---|---|---|
| pairs raw p<0.05 | 38/45 | 37/45 |
| pairs BH q=0.05 | 38/45 | 37/45 |

- sign-opposite pairs: **7/45**
- significant on **both** interfaces, raw: **2**
- significant on **both** interfaces, **after BH**: **2** (unchanged)

Both flips: bootstrap p = 0.0001 on all four legs; McNemar p = 3.55e-06 / 1.72e-05
(keep10 vs scratch16L) and 7.67e-05 / 3.13e-12 (keep10 vs keep14-reheal).
**Multiplicity does not rescue Obs4** — a reviewer who suspects p-hacking will find the
effect intact.

**Test B — each arm vs its null. Every flip arm is at or below the floor.**

| flip arm | letter | vs always-D | verdict |
|---|---|---|---|
| keep10 @83.5k | .2720 | +0.31pp, boot p **0.4214** | **AT** the floor |
| keep14-reheal @67.5k | .2492 | −1.97pp, boot p 0.0008 | **BELOW** the floor |
| scratch16L @200k | .2470 | −2.19pp, boot p 0.0001 | **BELOW** the floor |

**3/3 flip arms are at or below the letter floor. None is above it.**

**The contrast:** the ranking difference is real (p≈1e-4, survives BH), *and* the interface
that produces it cannot distinguish any of the three arms from a constant predictor.
A significant difference between two quantities that both fail to beat an input-blind
baseline is a difference in *how the instrument fails*, not in capability.

**The kill, quantified.** Restricting to the **4/10** arms significantly above their floor
on both interfaces (base, full32 @25k, keep14 @200k, shortgpt16 @200k): 6 pairs,
**0 sign-opposite, 0 significant flips.** The flip lives entirely outside the valid set.

**Seed stability** (seeds 0,1,2,3,1234 at 10,000 resamples): sign-opposite 7, flips raw 2,
flips BH 2, flips-within-valid 0, n_valid 4, all three flip-arm verdicts unchanged in every
seed. The verdict is not seed-dependent.

---

## 5. Honesty checks — four ways this could have been wrong

1. **Is the content side let off?** No. Content is tested against its own floor, and
   **10/10** arms clear it (+5.78 to +18.61pp). So the retraction is **not** "both
   interfaces are dead": the *content* ranking of the flip arms is above-floor and
   meaningful; only the letter side is noise. The flip is **one live instrument vs one dead
   one** — which is exactly why it cannot support "two valid interfaces disagree", the form
   the claim needed to take.

2. **Does the null choice decide the verdict?** Partly, and it must be disclosed. Against
   each arm's **own modal-prediction** constant instead of the best constant:

   | flip arm | vs always-D (.2689) | vs its own modal constant |
   |---|---|---|
   | keep10 @83.5k | +0.31pp (p 0.4214) | always-D (.2689) +0.31pp (p 0.4116) |
   | keep14-reheal @67.5k | −1.97pp (p 0.0008) | always-B (.2465) **+0.26pp** (p 0.5024) |
   | scratch16L @200k | −2.19pp (p 0.0001) | always-A (.2295) **+1.75pp (p 0.0001, above)** |

   Signs flip. We use the **best** constant, because a floor defined by the arm under test
   is not a floor. Reported both ways regardless.

3. **Mechanism.** bf16 exact-tie rate on the letter interface (when top1 == top2 exactly,
   `argmax` breaks the tie by **index** — input-blind): intact base **0.0013** vs
   keep10 **0.1380**, scratch16L 0.0579, keep14-reheal **0.2547**. The interface decays
   into a constant predictor by a documented numerical route, not by losing knowledge.

4. **★ THE RECORD'S OWN WORDING IS IMPRECISE, AND THE IMPRECISION MATTERS.**
   Both `TRAINER_ACTIVITY.jsonl` and `UPDATELOG.md` say the flip pairs "sat at the **chance
   floor**". Against the generic 1/4 = **0.2500** chance line that is **not what the data
   say**:

   | flip arm | letter | vs .2500 | |
   |---|---|---|---|
   | keep10 @83.5k | .2720 | **+2.20pp [+1.46,+2.94]** | **significantly ABOVE the chance line** |
   | keep14-reheal @67.5k | .2492 | −0.08pp [−0.80,+0.64] | at the chance line |
   | scratch16L @200k | .2470 | −0.30pp [−1.03,+0.42] | at the chance line |

   **1/3 flip arms is significantly above the naive chance line.** The retraction therefore
   does **not** follow from a chance-line comparison — it follows only from the
   construct-appropriate best-constant floor (.2689), which sits **+1.89pp above** the
   chance line because MMLU's gold letters are not uniform.

   This is the strongest possible form of the paper's own thesis: the self-falsification
   case does not merely *illustrate* that generic chance lines are inadequate, **it depends
   on it**. Anyone re-deriving our retraction with a chance line concludes we retracted
   without cause.

   **Action: correct "at the chance floor" to "at or below the best-constant-predictor
   floor (always-D, .2689)" wherever it appears** — `status/TRAINER_ACTIVITY.jsonl:6078`,
   `UPDATELOG.md:5927`, `status/revival_slate/*` ("retracted because both arms were at the
   chance floor"), and `status/NULL_CALIBRATION_P1.md:270`. Those files are MAIN's/other
   agents'; flagging, not editing. (Note `MAIN_verified_interface_validity.md` itself is
   *correct* — it says 无知识地板 / constant predictor, never "chance".)

---

## 6. Verdict

**The self-falsification case HOLDS.** Obs4's ranking flip is statistically real
(2/45 pairs, p≈1e-4 on both interfaces, survives BH across the full screen), yet all
**3/3** arms involved are at or below the letter interface's own input-blind floor
(keep10 .2720 indistinguishable from always-D .2689 at p=0.4214; scratch16L .2470 and
keep14-reheal .2492 significantly *below* it), and **0 of 6** pairs among the 4 arms valid
on both interfaces flips at all. The retraction was correct and was **not** over-cautious.

The paper can therefore carry a genuine, recomputed, self-directed row rather than
preaching — and the case is stronger than the record implied, because retracting it
**required** the construct-appropriate null: the generic chance line would have cleared
one of the two headline arms.

**Two corrections to our own record**, in the spirit of P1 §6:

1. "both arms at the **chance floor**" → **wrong floor named**. keep10 is +2.20pp
   significantly **above** the .2500 chance line. The correct statement is "at or below the
   **best-constant-predictor** floor (always-D, .2689)". (§5 check 4)
2. The Obs4 screen covers **10** arms / **45** pairs, not the 9 arms C1 uses —
   `keep14-reheal @67.5k` is one of the three flip arms and is absent from C1's arm list.
   Any writeup reusing C1's arm set will silently lose one of the two flip pairs.
