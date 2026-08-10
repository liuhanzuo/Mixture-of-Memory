---
scope: A04 Pilot One — the sd_run measurement that decides K2. PRE-REGISTRATION.
date: 2026-08-10 23:50 GMT+8
status: PRE-DATA. Written while A03 seeds 43/44 are still training (step ~215k of
        220000 on .73/.82); no step220000 checkpoint exists for either seed, no
        eval has run, and no sd_run number exists anywhere in this repo.
        Verified at time of writing.
decides: K2 ("disagreement drowned by seed variance") — the kill clause the design
         calls "the most likely killer"
---

# Pilot One pre-registration: measure `sd_run`, then let K2 fire or not

## 0. Why this file exists before the data

`A04_GATE_DESIGN.md` §5.3 commits to a rule that is only honest if it is fixed in
advance:

> Recommended: **S=3 for the gate**, with a pre-committed rule that if the measured
> `sd_run` implies a bound `> Δ` on ≥2 axes, **K2 fires** rather than seeds being
> added post hoc. Adding seeds after seeing the variance is the same
> selective-reporting move A01's gate-4 exists to prevent.

The measurement is imminent and the temptation it creates is specific: if `sd_run`
comes back just barely too large, the cheap move is to add a 4th seed (`t` falls
6.314 → 2.920 → 2.353) until the bound fits under `Δ`. That is exactly the move
this file forecloses.

---

## 1. The precondition question, resolved explicitly

§6.2 gates Pilot One on: *"Only if Pilot Zero shows the disagreement exists."*

On 2026-08-10 (commit `b93247f`) I **retracted** Pilot Zero's K1 verdict to
INDETERMINATE. So the precondition must be re-examined rather than assumed, and
the distinction is load-bearing:

| Pilot Zero produced | status after the retraction |
|---|---|
| A verdict on K1's **"≤1 of ≥24 evaluated cells"** clause | **RETRACTED.** Only 3 cells were evaluated (3 axes × 1 checkpoint). A ≥24-cell clause cannot be adjudicated from 3 cells. |
| The finding that a disagreement **of the required shape exists** | **STANDS.** At the PLATEAU-accept checkpoint, `NI(Δ=0.10·residual)` rejects on 3/3 decision axes by 6.6–9.1× the margin. |

§6.2's precondition names **existence**, and existence is the part that survived.
It is also the more robust part: `residual(arm) − residual(intact) =
reported(arm) − reported(intact)`, so the null cancels exactly and the measured
difference is convention-independent; the margin-sensitivity sweep further shows
the existence of ≥1 rejecting axis is invariant over Δ fractions 0.10 → 0.66 under
all five null conventions (`evidence/a04_margin_sensitivity_sweep.json`).

**Conclusion: Pilot One's precondition is satisfied.** It is not being smuggled
past a failed gate. What is *not* satisfied is K1's own ≥24-cell denominator, and
that is a reason to run the gate, not a reason to skip the variance measurement
that would make running it pointless.

The second, harder prerequisite is also now met: the `DistributedSampler(seed=)`
fix (`ce5c298`) reached zwfy6 on 2026-08-10 16:53 (md5
`284b286f90b526e4e8ad93a68e2a3b16` verified on both disks). Before that date every
"seed" in this repo varied **fresh-block init only**, so any `sd_run` borrowed from
a pre-fix run is an underestimate and a K2 pass using it would be unearned
(`SEED_SEMANTICS_DEFECT.md`).

---

## 2. STAGE A — the free `sd_run` bound, from checkpoints already training

**Cost: 0 additional GPU-h.** A03's data-order replication (seeds 43 and 44,
`task #236`) is, incidentally, the first true run-to-run measurement this repo has
ever had: same config, same data, **different data order**, under the fixed
sampler. Its manipulation check is already verified pre-data — over the common
step window 200020–202880 the training-loss correlations are

| pair | r |
|---|---|
| Arm 3 vs Arm 6 (both sampler seed 0) | **+0.99966** |
| Arm 3 vs seed 43 | **−0.0101** |
| seed 43 vs seed 44 | **+0.0041** |

so the phase-lock is genuinely broken and the two runs are independent draws.

### 2.1 The estimator, fixed now

For each axis `x` with two independent runs, the unbiased sd from `n=2` is

```
sd_run(x) = |m_43(x) − m_44(x)| / sqrt(2)
```

and the S=3 one-sided bound Pilot One would face is
`bound_3(x) = t_{0.05,2} · sd_run(x) / sqrt(3) = 2.920 · sd_run(x) / 1.7321`.

### 2.2 Pre-committed decision rule

Pre-registered Δ values (from `PILOT_ZERO_VERDICT.md` §1, `split` convention,
anchor pinned by rule G0):

| axis | Δ | decision weight |
|---|---:|---|
| TriviaQA EM | **4.043 pp** | primary |
| PopQA EM | **1.321 pp** | decision |
| MMLU-content | **1.024 pp** | decision |
| *NQ-open EM* | *0.970 pp* | **DEMOTED — descriptive only, no decision weight** (design §5.2) |

> **K2 FIRES** iff `bound_3(x) > Δ_x` on **≥ 2 of the 3 decision axes**.
> If K2 fires, **A04 dies here**, for 0 GPU-h beyond what A03 already spent.
> If K2 does not fire, proceed to Stage B. **No 4th seed may be added at this
> stage to rescue a bound**, and no axis may be re-weighted after seeing a number.

### 2.3 The inference is ONE-DIRECTIONAL, and this is the whole caveat

This must not be overread, so the asymmetry is fixed in advance:

* **A large `sd_run` KILLS.** Run-to-run noise at `keep7` is evidence about the
  measurement apparatus and the 1B/dolmino/8×H20 training setup, which Pilot One
  would share. If the noise already swamps Δ here, spending 135 GPU-h to
  re-measure it at a different depth is not justified.
* **A small `sd_run` does NOT clear K2.** Three mismatches make Stage A
  inadmissible as a K2 *pass*: (i) wrong arm — `keep7+fresh2` (56.2 % depth), not
  the recommended `keep12+fresh2` (87.5 %); (ii) wrong budget — 20,000 steps, not
  5,000, and variance across a longer run need not scale to a shorter one;
  (iii) `n=2` gives a **range, not a variance with a usable CI**, so a small
  observed spread is weak evidence of a small population sd.

**So Stage A can only kill, never clear.** Recording that now prevents a
convenient small number from being promoted into a clearance later.

### 2.4 Also pre-committed: what a MIXED reading means

If `bound_3 > Δ` on exactly **1** of 3 decision axes, that is **not** a K2 fire and
**not** a clearance — it is `K2_INDETERMINATE_AT_STAGE_A`, and Stage B proceeds.
Declaring this now removes the discretion that a 1-of-3 result would otherwise
hand me.

---

## 3. STAGE B — Pilot One proper, only if Stage A did not fire

**Arm**: `keep12+fresh2` at 1B — `--keep_front_layers 12 --n_fresh_layers 2`.

Chosen by `PILOT_ZERO_VERDICT.md` §5 for a reason that is about falsifiability, not
optimism: `keep7+fresh2` is a confirmed **constant-REJECT** rung (after 52.4 B heal
tokens it recovers 12–39 % of the intact residual, NI rejecting by 6–9× on every
axis), so a rule tested only there can never be *observed to accept* and the
disagreement is automatic and uninformative. `keep12` at 87.5 % depth is the
candidate most likely to let NI sometimes accept, which is what makes the
disagreement test falsifiable at all.

**Launchability — verified, and it contradicts a natural reading of §5.**
`PILOT_ZERO_VERDICT.md` §5 says "No 1B arm at keep12 or keep10 exists on **either**
disk." That is true and it is about *measured recovery*; it is **not** a
launchability blocker. `scripts/train_olmo2_arch_probe2.py:transplant_front()`
prunes from `--model_path` (the HF base) directly and `--resume_from` is optional,
so the arm is constructed at launch. Assets verified present on zwfy6 2026-08-10
23:40: `models/OLMo-2-0425-1B/config.json`, `data/dolmino_now15b.npy`
(126,907,244,672 B), `data/dolmino_now_val.npy` (33,554,560 B).

**Design**: S=3 seeds × 5,000 steps, `--milestone_every 2500`, seeds `{101,102,103}`
pinned here so they cannot be chosen after a first run's result is seen. All three
on **one disk** (zwfy6) per the design's checklist — note `dolmino_now15b.npy` is
**62,020,903,040 B on wzc1 vs 126,907,244,672 B on zwfy6**: same name, different
file, so mixing disks would silently mix corpora.

**Cost**: measured 1B median 2.02 s/step on 8×H20 → 2.81 h wall and 22.4 GPU-h per
run; **≈135 GPU-h total**, ~5.6 h wall as one wave across 3 nodes (or 2 waves on
the 2 nodes freeing at ~04:20).

**Decision**: recompute `sd_run` from S=3 (proper `sd`, df=2, `t=2.920`) and apply
the §2.2 rule unchanged, on the same 3 decision axes, with the same Δ. If the S=3
bound exceeds Δ on ≥2 axes, **K2 fires and A04 dies for ~135 GPU-h instead of
~2,900.**

**Pre-committed against the obvious escape hatch**: if the bound lands just above Δ,
K2 fires. Seeds are not added. `S=4` (`t=2.353`) and `S=5` (`t=2.132`) are
tabulated in design §5.3 and are therefore *known* to be available; choosing them
after seeing S=3 would be selecting the design that gives the wanted answer.

**Not authorised by this file**: the full 1,077–4,309 GPU-h gate (Pilot Two). It
requires separate user approval per design §6.3.

---

## 4. Integrity requirements carried over (not optional)

* `chat_template=False`, `--add_bos 0`, no few-shot, greedy decode, on every eval.
* Shard completeness asserted per cell: `n_shards == 8`, shard indices exactly
  `{0..7}`, **and** exact item counts (MMLU 14,042 / TriviaQA 17,944 / PopQA 14,267
  / NQ-open 3,610). A silent 5-of-8 merge has corrupted results in this repo before.
* Canonical scorers imported from `A03/code/analyze_1b_knowledge_floor.py`, never
  reimplemented.
* Margin guard D1–D6 evaluated **before** `NI(Δ)`; intact anchor pinned per rule G0.
* `[optim] group` lines captured per run, to document whether differential LR was
  actually active — the distill trainer's `_classify_param` has silently made
  `--lr` a no-op for fresh groups elsewhere in this project.
* Every number that reaches a verdict must exist in a committed JSON, not only in
  prose. A03 was caught with a `+0.48pp` headline living only in two `.md` files.

## 5. Known-unverified at time of writing

1. `sd_run` — the entire point; no value exists yet anywhere in the repo.
2. Recovery level at `keep12` at 1B — unmeasured; the `j` choice extrapolates from
   a **7B** ladder that itself spans two corpora and unequal step budgets and is
   evidence of **ordering only**.
3. Whether 5,000 steps at `keep12` produces enough recovery for NI to ever accept.
   If `keep12` turns out to be a **constant-ACCEPT** rung, the gate must bracket
   downward to `keep10` (design §5, second choice) — and that bracketing decision
   is a *design* decision, not a result.
4. Whether the disagreement replicates at 7B — out of scope, unfunded.
