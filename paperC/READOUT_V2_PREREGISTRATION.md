# Pre-registration — read-out v2: a competence criterion that letter collapse cannot satisfy

**Written 2026-08-13, 03:15–04:10 +08:00.** **0 GPU used** (CPU analysis on `.73`;
`.104`'s training and both SparseForge B200 arms untouched throughout).

> **This document was written BEFORE the re-judging of the existing cells was
> committed to prose.** The criterion, its four admissibility gates, the
> materiality constant, and the stratification decision are all fixed in §2–§5
> below. §7's re-judged table was produced by running the §5 code against the
> already-scored records. The ordering matters and is stated plainly:
> **the *rule* is pre-hoc; the *numbers* it produces are post-hoc by
> construction, because the cells were scored on 2026-08-12/13.**
>
> What that ordering CANNOT protect against, and I am not claiming it does: I had
> already seen `HEAL_TRAJECTORY_READOUT_1.md` §4, so I knew the *shape* of the
> defect (letter collapse, `always-A` == floor, independence residual ~0) before
> designing the fix. This is amendment-after-seeing-a-defect, which is legitimate
> and is what §8 of the original pre-registration asked for ("to be decided
> BEFORE step 121000 so it stays pre-hoc"). It is **not** the same as having
> fixed the criterion before any data existed. Anyone reading this must treat the
> *criterion* as pre-registered against the **step-121000 read-out** and every
> other unscored milestone, and as a **post-hoc re-analysis** of the 27 cells
> that already existed.

**Supersedes** `HEAL_CONFOUND_PREREGISTRATION.md` §8 as the *competence* read-out.
It does **not** supersede §8's arm identity, depth choice, corpus caveats, or the
step-121000 read-out point, all of which stand unchanged.

---

## 1. The defect, restated precisely — and narrowed against the brief

`HEAL_TRAJECTORY_READOUT_1.md` §4 established that §8's `H_heal` criterion is
satisfiable by changing which letter a degenerate model collapses onto. I
verified this independently and it holds. But two of the framings around it need
correcting, because the corrections change what the fix has to do.

### 1a. CORRECTION: the harness is already likelihood-based. This is not a generation artefact.

The task brief suggests "per-item likelihood-based scoring (`mc_ll` style)
**instead of** generated-letter matching". **There is no generated-letter matching
to replace.** `scripts/eval_olmo2_mc_letter_content.py:500-622` scores the
candidate continuations `" A"`, `" B"`, … by summed token log-probability and
takes the `argmax`. The per-item records carry the full log-prob vector — e.g.
`{"A": -3.322937, "B": -7.510437, ...}`. There is no sampling, no decoding, no
regex over generated text.

So "the model emits A for 90% of items" means **the letter `A` wins the
log-prob argmax on 90% of items**, which is already the `mc_ll` reading. Switching
to likelihood scoring is a no-op here. **The collapse is in the argmax of a
likelihood, and any criterion built on `argmax`-accuracy inherits it.** That is
why the fix below changes the *null*, not the *interface*.

### 1b. CORRECTION: the defect is NOT confined to floor-level scores, but it is close.

The brief invites me to say so if "it only bites at floor-level scores and the
intact end was never at risk". Checked, and the honest answer is in between:

* At the intact end the defect is **inert**. `qwen3 INTACT` has
  `Delta_perm = +34.889 pp` vs a v1 reading of `+34.450 pp` — a 0.44 pp
  difference on a 34 pp effect. The intact end was never at risk.
* At the floor it is **total**: it decides the verdict entirely.
* But there is **one cell strictly between the two where it flips a published
  label**: `qwen3_8b_base/k14` un-healed reads **`ABOVE floor` (+0.233 pp,
  p=0.0192)** under v1 and **`TRACE_SIGNAL`** under v2. That cell is quoted in
  `HEAL_CONFOUND_PREREGISTRATION.md` §8 as the live justification for the
  "Neither / ambiguous" outcome branch. So the defect reaches at least one
  load-bearing non-floor claim, and "floor-only" would be too generous.

### 1c. What the defect actually is, mechanically

`always-<L>` accuracy is a non-flat dataset property (A `.1166` … J `.0785`).
`always-A` is the argmax, i.e. **the floor is by construction the score of the
single best collapse**. Therefore:

| collapse letter | v1 statistic (`acc − floor`) | v1 verdict |
|---|---:|---|
| always-A | `0.000` pp | AT floor |
| always-E | `−2.111` pp | **BELOW floor** |
| always-J | `−3.807` pp | **BELOW floor** |

Same competence (none), three verdicts. `MEASURED`: across all 10 legal collapse
letters the v1 statistic spans **2.193 pp** (computed on the items where each
letter is a legal option; §6 self-test).

---

## 2. The replacement criterion

Promote the independence model from *diagnostic* to *null*.

> **Null.** `acc_hat := E[acc | the arm's own prediction vector permuted
> uniformly at random across items, WITHIN `n_opt` strata]`
> `= Σ_s (1/(n·n_s)) Σ_L cnt_pred[s,L]·cnt_gold[s,L]`
>
> **Statistic.** `Delta_perm := acc − acc_hat`.
>
> **Test.** Two-sided; `p` from both (a) a 10,000-draw within-stratum
> permutation of the prediction vector and (b) a 10,000-resample paired
> bootstrap that **recomputes `acc_hat` inside each resample** (an acc-only CI
> would treat an estimated null as fixed and understate the uncertainty). Both
> must agree at α=0.05; seed 7, matching #248/#250/#251.

### Why this satisfies requirement (1) as an identity, not a hope

For a pure `always-L` emitter, `P(pred) = δ_L`, so
`acc_hat = Σ_L' δ_{L'L}·m_{L'} = m_L`, and `acc = m_L` as well. Hence

```
Delta_perm == 0   EXACTLY,   for every collapse letter L.
```

This is algebra, not an empirical regularity. **Requirement (1) is discharged by
construction**: a model emitting 100% A and one emitting 100% E both score
exactly 0. `MEASURED` in §6: 0 for all 10 letters to `<1e-12`, asserted in code.

### Why it has a competence term — requirement (2), with the numeric check

The independence model that "explains every damaged cell" **is** this null. So
the check requested is direct: **can the independence model explain a passing
score?** By definition it cannot — a cell passes iff `acc` is significantly
*greater* than what independence predicts. The numeric demonstration on data
already held:

| cell | `acc` | `acc_hat` | `Delta_perm` | boot p | can independence explain it? |
|---|---:|---:|---:|---:|---|
| qwen3 INTACT | `.461104` | `.112212` | **+34.889 pp** | 0.0001 | **NO** — off by 311% of the null |
| olmo2 INTACT | `.271858` | `.112324` | **+15.953 pp** | 0.0001 | **NO** |
| llama3 INTACT | `.329205` | `.112014` | **+21.719 pp** | 0.0001 | **NO** |
| qwen3 heal@7000 | `.115775` | `.115995` | −0.022 pp | 0.8522 | **YES** — fully explained |
| qwen3 k8 un-healed | `.107796` | `.109185` | −0.139 pp | 0.0964 | **YES** |
| olmo2 keep8@121000 | `.115442` | `.113152` | +0.229 pp | 0.3122 | **YES** |

### Why it retains discrimination — requirement (3)

The three competent intact anchors sit at `+15.95 / +21.72 / +34.89 pp` with
`p=0.0001` against a damaged population spanning `−0.42 … +0.61 pp`. The
separation is **26× the largest damaged effect**, and it is *larger* in relative
terms than v1's, because v2 removes the ~0.4 pp of letter-prior credit that v1
was giving the damaged arms for free.

### ⚠️ The one thing this null must never be quoted as

`Σ_L p_L m_L ≤ max_L m_L`, so **`acc_hat ≤` the best-constant floor always**,
with equality iff all prediction mass sits on `A`. v2 is therefore a **lower
absolute bar** than paperC's headline floor. It answers a different question
(§3) and **must never be reported as "the arm cleared paperC's floor"**.

---

## 3. This does not repeal A01 / paperC's headline null. It partitions the scope.

`build_null_calibration_table.py:596-600` states the rule v2 appears to break:
"the pre-registered null is the BEST constant (always-D) **because a floor must
not depend on the arm being tested**". That rule is right for what it serves, and
v2 does not touch it. Two questions, two nulls:

| | question | null | must be |
|---|---|---|---|
| **v1** (paperC headline) | *Is this interface valid?* Does it beat the best input-blind predictor? | best-constant `always-A .116606` | **arm-INdependent**, else arms are not mutually comparable |
| **v2** (this document) | *Does THIS arm know anything?* Do its predictions carry item-level information? | permutation of the arm's own predictions | **arm-CONDITIONAL**, else the statistic confounds "which letter it collapsed onto" with "what it knows" |

§8's `H_heal` is a v2-type question ("did healing restore capability?") that was
operationalised with a v1-type statistic. **That mismatch is the defect.** Both
are reported for every cell, and disagreements are disclosed (gate G5), never
silently resolved.

---

## 4. Candidate directions evaluated — including two rejections that matter

### 4a. REJECTED: reuse A01's `own_modal` null. It is *worse* than v1 here.

The brief asks me to check whether existing machinery already solves this, and to
prefer reuse. A01 **does** already have an arm-conditional letter null:
`build_null_calibration_table.py:612-616` computes `letter_own_modal_null` =
`always-<the arm's own modal letter>`, exactly to catch collapse. Reuse was the
expected answer. **It fails on this data, and in the dangerous direction.**

`MEASURED` on the P1 cell (`qwen3_8b_base/k8` un-healed, modal `E` at 94.5%):

| null | value | `acc − null` | reading |
|---|---:|---:|---|
| v1 best-constant `always-A` | `.116606` | **−0.881 pp** | BELOW floor |
| **A01 `own_modal` = `always-E`** | `.095495` | **+1.230 pp** | **looks ABOVE null** |
| v2 permutation (stratified) | `.109185` | −0.139 pp | n.s. |

`own_modal` **over-credits by 1.37 pp and flips the sign**. The mechanism,
verified: `own_modal` collapses the arm's whole prediction distribution to its
single modal letter, but the arm's other 5.5% of mass sits on **higher**-marginal
letters (A `.1166`, B `.1124` vs E `.0955`), so the *mixture* beats `always-E` by
`+0.098 pp` **with zero item-level information**, and stratification adds the
rest. A01 itself flags this ("against its own modal letter an arm can come out
marginally ABOVE while being BELOW always-D, and that must be disclosed") — it
was built as a *disclosure diagnostic*, not a competence null, and it is correct
at that job. v2's permutation null is the same idea done over the **full
prediction distribution** instead of its mode, which is what removes the loophole.

**So: reuse was checked, and rejected on measured grounds, not overlooked.**

### 4b. REJECTED: switch to the content interface (paperC's letter-vs-content work).

`content_norm` is **below the letter floor on every damaged cell**
(`.0748`–`.1100` vs `.116606`), and under the *same* permutation null it is
**more negative than letter**, not less:

| cell | letter `Delta_perm` | content_norm `Delta_perm` |
|---|---:|---:|
| qwen3 heal@7000 | −0.022 pp | **−3.610 pp** |
| qwen3 k8 un-healed | −0.139 pp | **−1.076 pp** |
| olmo2 keep8@121000 | +0.229 pp | −0.086 pp |
| qwen3 INTACT | +34.889 pp | +17.735 pp |

Content is a *different* interface with its own null pathologies (A01 gate-3: the
longest-option null is under-specified in three ways — tie convention, length
unit, tokenizer — moving up to 10.6 pp). It does not rescue this read-out.
`HEAL_TRAJECTORY_READOUT_1.md` §4 already noted content is below letter on all
five healed cells; this confirms it under the new null too.

### 4c. ADOPTED AS GATES, NOT AS THE METRIC: entropy / distinct-letter counts.

Correct instinct, wrong slot — and paperC has already been burned here. README's
scope-discipline list contains: *"'damage turns letter into a **constant**
predictor' — NARROWED … **Modal share and floor verdict are DECOUPLED** (modal
share is only 43–45% on several below-floor arms)."* Confirmed in the new table:
`llama3/k12` has modal share 0.339 and entropy 0.614 yet `Delta_perm = +0.002 pp`
(p=0.997) — maximally *un*-degenerate-looking, and completely empty. **Modal
share is not a competence proxy in either direction.** So entropy and modal share
are retained as *descriptive* columns and the *capacity* gate G1/G2 does the
admissibility work instead, because capacity is defined in the units of the
statistic itself.

### 4d. REJECTED for now: balanced / permuted-option controls.

The clean fix for a letter prior is to re-run with permuted option order, which
destroys it at the source. Rejected here on the requirement-(4) constraint: it
needs **new GPU** (a re-score of every cell) and this task is 0-GPU. Recorded as
the natural extension if the letter prior ever becomes load-bearing for a
headline claim.

### 4e. Relation to A01's null calibration, which owns this territory.

A01 owns null calibration and its `always-<L>` argmax null is precisely what is
being gamed. v2 does not duplicate or replace it (§3): it adds an
**arm-conditional** null for **capability** questions, keeps A01's
arm-independent null for **instrument-validity** questions, and reports both.
Estimators are **imported** from `mmlu_pro_power_nulls.py`, which itself asserts
bit-identity against A01's, so no estimator is re-implemented.

---

## 5. Fixed BEFORE re-judging: stratification, gates, materiality

### 5a. Stratification by `n_opt` is the canonical form. Fixed in advance.

MMLU-Pro's `n_opt` is **not** constant: `{3:21, 4:606, 5:52, 6:93, 7:158, 8:320,
9:801, 10:9981}`. Unstratified permutation assigns `pred=J` to 4-option items
where J was never available, and the gold marginal differs sharply by stratum
(gold-A is 42.9% at `n_opt=3` vs 10.5% at `n_opt=10`). The canonical null is
therefore **within-`n_opt`**; unstratified is reported as a sensitivity, in the
same discipline A01 applies to tie conventions.

**This is load-bearing, not cosmetic.** On the P1 cell it moves the statistic by
**−1.271 pp** — and it is the reason last night's headline residual (`+1.132 pp`,
the largest in the table, read as "the un-healed arm has the most residual
signal") **is an artefact**. Verified mechanism: the arm's modal letter `E` is
only legal on 11405/12032 items; `P(gold=E)` is `.095495` overall but `.100745`
conditional on E being legal. The unstratified null uses the lower number and
credits the arm with signal it does not have.

### 5b. Admissibility gates, evaluated BEFORE any verdict (A04 D1–D6 style)

| gate | condition | consequence |
|---|---|---|
| **G1** RESOLUTION | `Delta_max ≤ hw95`, where `Delta_max = Σ_L min(n_pred_L, n_gold_L)/n − acc_hat` is the best `Delta_perm` any re-assignment of the observed prediction multiset could reach | `NOT_MEASURABLE` — the ceiling is under the instrument's resolution (D6-analogue) |
| **G2** DEGENERATE | `Delta_max == 0` | `NOT_MEASURABLE` — a single-letter emitter; the null *is* the observation, zero power by construction. Named separately from G1 because it is the historically real case: A03's barely-healed keep7@500 emitted `A` on **14042/14042** items |
| **G3** INTEGRITY | shard index set `== {0..7}`, `n == 12032`, 0 dup, 0 nan, 0 trunc, `chat_template is False` | raise. Asserted with `is False`; **`is not True` is banned** (it passes on `None`) |
| **G4** NULL ADMISSIBILITY | no cell on the axis clears the null | the null sits above the whole population and measures nothing (D4-analogue) |
| **G5** COHERENCE | v1 and v2 disagree on a cell | **disclose**, never silently resolve |
| **G6** ANCHOR | the family's intact `recovery_fraction < 0.10` | `SIGNAL_NOT_ANCHORED` — relative claims blocked for that family |

### 5c. Materiality: significance is not capability at n=12032

At `n=12032`, `p<0.05` is reachable at effects far below anything nameable as
competence. **`qwen3/k14` reaches `p=0.0066` at `+0.267 pp` while emitting `A` on
94.6% of items.** A significance-only criterion would re-import the defect one
level up. So the magnitude scale is fixed here, in advance:

> **`recovery_fraction := Delta_perm / Delta_max`** — the share of the item-level
> alignment the arm's own prediction marginal could express that it actually does.
>
> **Material** ⟺ `recovery_fraction ≥ 0.10 × recovery_fraction(intact, SAME
> family)`, i.e. A04's own pre-registered `Delta = 0.10 × <intact anchor>`
> (`A04_MARGIN_GUARD_PREREG.md` §4) applied **within family**.

Within-family is not stylistic: A01 established cross-family nulls are not
commensurable (content null moves up to 10.6 pp across tokenizers on identical
items) and paperC's README bans family-general orderings on these exact rungs.

**G6 exists because one anchor fails.** `llama2_7b` intact scores `.131981`
against a `.112392` null — `recovery_fraction = 0.0545`, i.e. the "anchor" is
itself nearly incompetent on MMLU-Pro. 10% of that is 0.0055, which noise clears.
So Llama-2 gets `SIGNAL_NOT_ANCHORED` for relative claims; its absolute
`Delta_perm` is still reported. **`MEASURED`, and it is a real finding about the
benchmark**: MMLU-Pro is 10-way and hard enough that a 7B intact model is barely
off its own permutation null.

### 5d. Verdict labels

`NOT_MEASURABLE` (G1/G2) · `NO_ITEM_LEVEL_SIGNAL` (p≥0.05) · `ANTI_SIGNAL`
(significant, negative) · `TRACE_SIGNAL` (significant, positive, immaterial) ·
`SIGNAL_NOT_ANCHORED` (significant, positive, G6 blocks scaling) ·
`ITEM_LEVEL_SIGNAL` (significant, positive, material).

---

## 6. RAN / READ

### RAN (this session, first-hand, 0 GPU)

| what | where | result |
|---|---|---|
| `nvidia-smi` self-check | `.73`, `.82` | 8×`0 MiB` / `0 %` before and after; **no GPU used at any point** |
| v2 permutation read-out, 27 cells | `.73` CPU | 27/27 pass G3; ~100 s wall |
| Collapse-invariance self-test | `.73` CPU | `Delta_perm == 0` to `<1e-12` for **all 10** `always-<L>`; v1 spans **2.193 pp** |
| A01 `own_modal` comparison | `.73` CPU | `+1.230 pp` on P1 — **sign flip vs v1**; §4a |
| Content-interface permutation null | `.73` CPU | content more negative than letter on all damaged cells; §4b |
| Stratification mechanism on P1 | `.73` CPU | E legal on 11405/12032; `P(gold=E | E legal) = .100745` vs `.095495`; §5a |
| `.104` training liveness (read-only) | `.104` | step **8220**/200000, loss 3.00, 8/8 GPUs 98–99%, `5.74 s/step` — **untouched** |

### READ (pre-existing, not produced here)

| source | taken from it |
|---|---|
| `HEAL_CONFOUND_PREREGISTRATION.md` §8 | floor `always-A .116606`; `MAXLEN=2048`, `add_bos 0`, `desc_style none`, `chat_template=False`; read-out step 121000; the P1/P2 contrasts |
| `HEAL_TRAJECTORY_READOUT_1.md` §4 | the defect, and the independence-model diagnostic v2 promotes to a null |
| `A01/.../build_null_calibration_table.py:596-616` | the arm-independent-floor design rule (§3) and the `own_modal` diagnostic (§4a) |
| `A01/GATE3_CONVENTIONS_VERDICT.md` §3 | cross-family nulls not commensurable → within-family anchoring (§5c) |
| `A04_MARGIN_GUARD_PREREG.md` §2, §4 | the D1–D6 gate pattern; `Delta = 0.10 × intact`; the MMLU-letter D4 precedent (A03 keep7@500 = `A` on 14042/14042) |
| `paperC/README.md` scope discipline | modal share ⊥ floor verdict → entropy demoted to descriptive (§4c) |
| `scripts/eval_olmo2_mc_letter_content.py:500-622` | the harness is already likelihood-based (§1a) |

**Not run, deliberately:** any GPU job; any re-score; the step-121000 read-out;
the P1/P2 verdicts (§8).

---

## 7. The 27 cells re-judged

`n = 12032` per cell. `recov` = `recovery_fraction`; `rel` = `recovery_fraction`
relative to that family's intact anchor. G4 **admissible** (5 cells clear the
null). G1/G2 fire on **0** cells.

| cell | acc | `acc_hat` | `Delta_perm` | hw | boot p | recov | rel | **v2 verdict** | v1 `Δ` | v1 verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| qwen3 heal@5000 | .115276 | .115993 | −0.072 | 0.231 | 0.5412 | −0.009 | −0.016 | NO_SIGNAL | −0.133 | AT floor |
| qwen3 heal@5500 | .115691 | .115654 | +0.004 | 0.316 | 0.9810 | 0.000 | 0.001 | NO_SIGNAL | −0.091 | AT floor |
| qwen3 heal@6000 | .114860 | .115857 | −0.100 | 0.281 | 0.4932 | −0.008 | −0.015 | NO_SIGNAL | −0.175 | AT floor |
| qwen3 heal@6500 | .114943 | .115864 | −0.092 | 0.287 | 0.5204 | −0.008 | −0.015 | NO_SIGNAL | −0.166 | AT floor |
| qwen3 heal@7000 | .115775 | .115995 | −0.022 | 0.241 | 0.8522 | −0.002 | −0.004 | NO_SIGNAL | −0.083 | AT floor |
| olmo2 keep8@45000 | .117271 | .114046 | +0.322 | 0.370 | 0.0856 | 0.016 | 0.051 | NO_SIGNAL | +0.066 | AT floor |
| **olmo2 keep8@121000 [P2]** | .115442 | .113152 | +0.229 | 0.446 | 0.3122 | 0.009 | 0.030 | **NO_SIGNAL** | −0.116 | AT floor |
| **qwen3 k8 un-healed [P1]** | .107796 | .109185 | −0.139 | 0.162 | 0.0964 | −0.064 | −0.118 | **NO_SIGNAL** | **−0.881** | **BELOW floor** |
| olmo2 keep10@83500 | .112450 | .111508 | +0.094 | 0.481 | 0.6966 | 0.003 | 0.010 | NO_SIGNAL | −0.416 | AT floor |
| olmo2 keep12@124000 | .113115 | .112219 | +0.090 | 0.471 | 0.7026 | 0.003 | 0.010 | NO_SIGNAL | −0.349 | AT floor |
| olmo2 keep14@200000 | .119847 | .113766 | +0.608 | 0.500 | 0.0172 | 0.020 | 0.067 | **TRACE_SIGNAL** | +0.324 | AT floor |
| olmo2 shortgpt16@200000 | .153341 | .112797 | +4.054 | 0.575 | 0.0001 | 0.100 | 0.327 | **ITEM_LEVEL_SIGNAL** | +3.674 | ABOVE floor |
| qwen3 k10 un-healed | .118185 | .116155 | +0.203 | 0.228 | 0.0804 | 0.025 | 0.046 | NO_SIGNAL | +0.158 | AT floor |
| qwen3 k12 un-healed | .115027 | .114880 | +0.015 | 0.379 | 0.9432 | 0.001 | 0.001 | NO_SIGNAL | −0.158 | AT floor |
| **qwen3 k14 un-healed** | .118933 | .116263 | +0.267 | 0.188 | 0.0066 | 0.049 | 0.091 | **TRACE_SIGNAL** | +0.233 | **ABOVE floor** ⚠️ |
| llama2 k8 un-healed | .107463 | .111620 | −0.416 | 0.495 | 0.1002 | −0.013 | n/a | NO_SIGNAL | **−0.914** | **BELOW floor** |
| llama2 k10 un-healed | .113198 | .114575 | −0.138 | 0.356 | 0.4510 | −0.008 | n/a | NO_SIGNAL | −0.341 | AT floor |
| llama2 k12 un-healed | .115858 | .116528 | −0.067 | 0.081 | 0.1052 | −0.057 | n/a | NO_SIGNAL | −0.075 | AT floor |
| llama2 k14 un-healed | .116772 | .116408 | +0.036 | 0.085 | 0.3950 | 0.041 | n/a | NO_SIGNAL | +0.017 | AT floor |
| llama3 k8 un-healed | .113531 | .115399 | −0.187 | 0.351 | 0.2894 | −0.016 | −0.047 | NO_SIGNAL | −0.308 | AT floor |
| llama3 k10 un-healed | .111951 | .112269 | −0.032 | 0.366 | 0.8746 | −0.002 | −0.005 | NO_SIGNAL | −0.465 | AT floor |
| llama3 k12 un-healed | .111868 | .111851 | +0.002 | 0.514 | 0.9968 | 0.000 | 0.000 | NO_SIGNAL | −0.474 | AT floor |
| llama3 k14 un-healed | .114694 | .110945 | +0.375 | 0.446 | 0.0970 | 0.014 | 0.042 | NO_SIGNAL | −0.191 | AT floor |
| **qwen3 INTACT** | .461104 | .112212 | **+34.889** | 0.844 | 0.0001 | 0.544 | 1.000 | **ITEM_LEVEL_SIGNAL** | +34.450 | ABOVE floor |
| **llama3 INTACT** | .329205 | .112014 | **+21.719** | 0.813 | 0.0001 | 0.328 | 1.000 | **ITEM_LEVEL_SIGNAL** | +21.260 | ABOVE floor |
| **olmo2 INTACT** | .271858 | .112324 | **+15.953** | 0.761 | 0.0001 | 0.305 | 1.000 | **ITEM_LEVEL_SIGNAL** | +15.525 | ABOVE floor |
| **llama2 INTACT** | .131981 | .112392 | +1.959 | 0.525 | 0.0001 | **0.055** | 1.000 | ITEM_LEVEL_SIGNAL ⚠️**G6** | +1.538 | ABOVE floor |

### What changed, and is the new criterion doing work?

**Yes — three things change, and all three matter.**

1. **P1's verdict dissolves.** `qwen3_8b_base/k8` un-healed goes from **`BELOW
   floor` (−0.881 pp, p=0.0362)** to **`NO_ITEM_LEVEL_SIGNAL` (−0.139 pp,
   p=0.0964)**. Its entire below-floor reading was the `always-E`-vs-`always-A`
   gap. **This is the single most consequential change**: P1 is *the* contrast
   §8 uses to identify `H_heal`, and the asymmetry it rests on ("un-healed sits
   below floor, healed sits at it") **does not exist under v2** — both sides are
   simply empty. The same happens to `llama2/k8` (−0.914 → −0.416, n.s.), the
   only other `BELOW floor` cell in the table. **`BELOW floor` count: v1 = 2,
   v2 = 0.**
2. **One published `ABOVE floor` label is withdrawn.** `qwen3/k14` un-healed →
   `TRACE_SIGNAL`. It is significant (p=0.0066) but at `+0.267 pp` with
   `rel = 0.091` — under A04's 0.10 bar — while emitting `A` on 94.6% of items.
   §8 cites this cell as the reason its "Neither / ambiguous" branch is "a live
   possibility, not a strawman". That justification is now weaker.
   `olmo2/keep14` moves the other way (`AT floor` → `TRACE_SIGNAL`), so v2 is not
   uniformly conservative — it re-sorts rather than deflates.
3. **A benchmark-level limitation surfaces that v1 could not see.**
   `llama2_7b` intact has `recovery_fraction = 0.0545`: MMLU-Pro barely separates
   an *intact* Llama-2-7B from its own permutation null. Under v1 that cell reads
   a comfortable `ABOVE floor +1.538 pp`. **G6 fires**, and any Llama-2 relative
   recovery claim on MMLU-Pro is blocked. This is a real constraint on the paper's
   cross-family leg, invisible to v1.

**What does NOT change:** all five healed Qwen3 milestones, both OLMo-2 keep8
points, and the intact/damaged separation. The trajectory is still **flat at no
signal** over 5000→7000, and the intact end still separates by 16–35 pp. So v2
is not a re-litigation of the trajectory — it is a correction to what the
trajectory's endpoints *mean*.

---

## 8. Can §8's outcome labels be applied? **NO. Two independent reasons.**

Last night's agent refused on trajectory grounds. That refusal was correct and
**still holds**, and there is now a second, stronger reason.

**Reason 1 — the trajectory is 6% of the read-out.** The arm is at step **8220**
of the pre-registered **121000** (measured this session, read-only). §8 fixed the
read-out step in advance precisely so it could not be re-chosen after seeing
intermediate numbers. Applying outcome labels at 6.8% would be exactly that.

**Reason 2 — and this is the new one: `H_heal` is no longer identifiable by its
own P1 contrast.** §8 defines `H_heal supported` as "the healed arm moves UP to
AT-floor **while its un-healed twin sits below**". Under v2 the un-healed twin
**does not sit below** (−0.139 pp, p=0.0964). **The antecedent of the criterion
is false.** So `H_heal` as literally pre-registered is not merely unmeasured —
its contrast has dissolved, and no amount of further training makes it
measurable, because the failure is in the *comparator*, not the arm.

The `H_family` branch is equally unavailable: it requires the healed arm to stay
"significantly BELOW floor", and **no cell in the table is significantly below
its permutation null**.

**Consequence, stated plainly.** The step-121000 read-out should be run on the v2
criterion, and the question it answers is the narrower, well-posed one:

> Does a healed front-8 Qwen3 at step 121000 show **material item-level signal**
> (`ITEM_LEVEL_SIGNAL`) on MMLU-Pro, or does it remain at
> `NO_ITEM_LEVEL_SIGNAL` like every damaged cell measured so far?

That is answerable, falsifiable, and immune to letter collapse. But it is **not**
`H_heal` vs `H_family`, and the pre-registration's binary must not be reported as
though it had been tested. Recording this now, at step 8220, is the point of
finding the defect early.

**Trajectory evidence, for what it is worth (and it is worth little):** flat at
`−0.100 … +0.004 pp`, all `p > 0.49`, over 5000→7000. If step 121000 is also
`NO_ITEM_LEVEL_SIGNAL`, that will be consistent with OLMo-2's own keep8 at both
45000 and 121000, which are `+0.322` (p=0.086) and `+0.229` (p=0.312) after
**45k and 121k** heal steps. **OLMo-2 keep8 never acquires material signal on
MMLU-Pro at any heal budget measured**, which is the closest thing to a
prediction the existing data supports.

---

## 9. Falsifiability of v2 itself

Stated in advance, so v2 is not unfalsifiable:

1. **If G4 ever fires** (no cell on an axis clears the permutation null), v2 is
   inadmissible on that axis and must be retired, exactly as A04 retires a D4
   cell. Currently 5 cells clear it, so G4 passes.
2. **If the two p-values disagree** (permutation vs bootstrap) at α=0.05 on any
   cell, the disagreement is reported and the cell is `NEEDS_RECHECK`. Currently
   **0/27** disagree; largest `|boot_p − perm_p|` gap is **0.0371**
   (`llama2/k14`, both far from 0.05).
3. **If stratified and unstratified `Delta_perm` differ in SIGN** on a cell whose
   verdict is `ITEM_LEVEL_SIGNAL`, the verdict is downgraded pending a decision on
   the stratum definition. Currently 0 such cells (the largest stratification
   effect, −1.271 pp on P1, does not change a signal verdict).
4. **v2 cannot rescue any retracted claim.** In particular it does not revive
   "letter is a family-general step function", "k14 is the last arm above its
   floor" (v2 *weakens* this further), or "damage turns letter into a constant
   predictor".
5. **v2 says nothing about instrument validity.** paperC's headline claim
   ("report against the best-constant null, not chance") is a v1 statement and is
   untouched. §3.

---

## 10. Provenance

| artefact | path |
|---|---|
| this pre-registration | `paperC/READOUT_V2_PREREGISTRATION.md` |
| v2 read-out code | `paperC/code/heal_readout_v2_permutation_null.py` (md5 `fb3c129a…`, identical on both disks) |
| v2 evidence, 27 cells | `paperC/evidence/heal_readout_v2_permutation_null.json` |
| driver log | zwfy6 `logs/paperC_readout_v2.out` |
| v1 trajectory (unchanged) | `paperC/evidence/heal_trajectory_mmlu_pro.json` |
| v1 degeneracy diagnostic | `paperC/evidence/heal_degeneracy_decomposition.json` |
| per-item records, 27 cells | **zwfy6-only**: `mmlu_pro_lc_paperC_heal_results/`, `mmlu_pro_lc_crossfamily_results_fix/`, `mmlu_pro_letter_content_results/` |
| pinned checkpoints | zwfy6 `outputs/paperC_qwen3base_heal_k8f2_pinned/` — **do not delete** |

**Two-disk note.** The per-item records were checked on **both** disks: they are
`zwfy6`-only (`find` on wzc1 returns nothing), which is expected — they are ~21 MB
per cell and the scoring ran on `.73`/`.82`. The code and this document are on
**wzc1** and were copied to zwfy6 with `scp -O` and md5-verified identical before
execution. Nothing is claimed missing on the basis of a one-disk search.

**Checkpoint rotation.** `keep_last_n=3` + `milestone_every=5000` makes
non-multiples of 5000 transient. `outputs/paperC_qwen3base_heal_k8f2_pinned/`
protects 5000–7000 by hardlink (same inodes, 0 extra bytes). **Pin before scoring
any new milestone.**

---

## ADDENDUM 2026-08-15 — the "always" at lines 144-146 is RETRACTED

**Lines 144-146 above are left byte-identical** (dated pre-registration provenance must not be
silently rewritten). This addendum records that one of their claims no longer holds.

Those lines assert `acc_hat <=` the best-constant floor **always**, and that "v2 is therefore a
lower absolute bar". **The word "always" is withdrawn.** The proof behind it,
`sum_L p_L m_L <= max_L m_L`, is *unstratified*, whereas the estimator permutes within `n_opt`
strata. Stratification only gives `acc_hat <= sum_s w_s max_L m_(s,L)`, whose right-hand side
dominates `f_const = max_L sum_s w_s m_(s,L)`.

Measured on MMLU-Pro (0 GPU; the 8 per-item shards of `7B_base`, n=12032, 0 dup, 0 nan;
integer counts independently reproduced twice):

- `f_const = 1403/12032 = 0.1166057180851064` (always-A)
- `sum_s w_s max_L m_(s,L) = 1439/12032 = 0.1195977393617021`
- gap = **exactly 36 gold items = 0.2992021277 pp**
- `argmax_L` differs from A in **4 of 8 strata** (`n_opt` = 5, 6, 7, 8 -> B, E, B, E),
  covering 623/12032 items

The ordering nevertheless holds in all 27 evaluated cells, but that count is weaker than it
looks: 13 of the 27 clear the floor by more than the entire 0.2992 pp gap and so could not have
violated it under any prediction vector. The inversion is realizable **on MMLU-Pro itself** by
the `n_opt`-conditional emitter A,A,B,E,B,E,A,A.

**What replaces it:** the ordering is a theorem under an explicit regularity condition ---
`p_{s,L} = p_L` for all `s` gives `acc_hat = sum_L p_L f_L <= f_const` exactly. The 27 cells are
evidence for *that condition*, not for the bound.

Evidence: `paperC/evidence/s2_02_stratified_ordering.json` (now carries all 8 strata rows and
the integer forms), `paperC/evidence/s2_02_strata_raw.json`,
`paperC/code/s2_02_stratified_ordering.py`. Manuscript wording: `03_method.tex` final paragraph
of §3.2. Adjudicated in `paperC/review_rounds/round_01/`.
