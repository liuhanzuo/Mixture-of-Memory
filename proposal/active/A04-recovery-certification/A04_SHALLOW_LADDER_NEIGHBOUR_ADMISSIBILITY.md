# A04 — RECONCILIATION: does §2.0.2's neighbour precondition bind the shallow-rung ladder?

**Purpose.** `A04_GATE_DESIGN.md` §2.0.2 imposes a PRECONDITION on *reporting an
accept*; `A04_SHALLOW_RUNG_LADDER_PREREG.md` §5.5 pre-declares that this pass makes
**no neighbour claim**. If Branch A (NI ACCEPT on ≥2/3 axes) fires tonight, those two
sentences are read by the same reader in the same hour and the verdict's admissibility
is undefined. This document fixes the reading **before the numbers exist**, so that
the label attached to a Branch A accept is not chosen after seeing it.

**Written and committed PRE-DATA.** Evidence, recorded at 2026-08-13 **14:06:10 UTC**
(22:06 +0800), immediately before the `git commit` of this file:

| arm | node | last logged step | of | `step5000.pt` | eval result dirs | analysis JSON |
|---|---|---|---|---|---|---|
| `keep14+fresh2` seed101 | `.73` | **3740** | 5000 | absent | `A04_1B_shallow_*` **absent** | absent |
| `keep13+fresh2` seed101 | `.82` | **3900** | 5000 | absent | `A04_1B_shallow_*` **absent** | absent |

`ls -d olmo2_mmlu_content_results/A04_1B_shallow_* olmo2_closedbook_results/A04_1B_shallow_*`
→ `No such file or directory` on both nodes; `evidence/` contains no
`*shallow_ladder*` file. **No margin, no accuracy, no accept/reject boolean for
either arm exists anywhere on either disk at commit time.** Nothing below was fitted
to a number; every threshold is either imported from an already-committed document or
derived in closed form.

**Zero GPU spent by this document.** CPU reads + one markdown + one commit. No process
was killed, no file owned by the ladder run (`a04_shallow_ladder_chain.sh`,
`a04_shallow_rung_ladder_ni.py`, `STATUS.json`) was modified.

---

## 0. Answer up front

1. **§2.0.2 DOES bind this ladder.** It is scoped to "any `NI(Δ)` accept reported by
   this gate", and this ladder reports `NI(Δ)` accepts produced by the same imported
   `ni_rule` under the same frozen Δ. Prereg §5.5 does **not** exempt it — §5.5 renounces
   a *claim*, §2.0.2 imposes a *duty*, and renouncing a claim cannot discharge a duty.
   There is no conflict of the kind that would make Branch A inadmissible; there **is** a
   documentation gap, which is what this file closes.
2. **A Branch A accept is admissible**, and its label is fixed in §2 below:
   `CERTIFIED` / `ACCEPT_IS_CHECKPOINT_SELECTION_DEPENDENT` / `ACCEPT_SINGLE_SIDED_NEIGHBOUR_ONLY`,
   decided by a mechanical rule with no free parameter.
3. **The gate constant is `c₂ = 2/√π = 1.1283791670955126`** (k=2, two checkpoints),
   **not** 1.6926. Its σ is the **mean of the two cells' own `bootstrap_se_pp`** on that
   axis, computed by the same estimator the ladder script already uses.
4. **step2500 → step5000 is 2500 steps and is NOT a neighbourhood.** It spans
   LR 1.144e-05 → 2.000e-06 (a factor **5.72**) and half the cosine. The k=2 gate is
   therefore reported as a **necessary but not sufficient** discriminator, and a range
   that clears it must **not** be read as instability.
5. **Per-axis, never blanket** — §2.0.2 says so in bold. `mmlu_content` is length-free
   and has never had a supra-noise neighbour range; the generative axes have.

---

## 1. Is §2.0.2 in force here? — verbatim reading

### 1.1 §2.0.2's own scope sentence

> **PRECONDITION.** Any `NI(Δ)` accept reported by this gate must be accompanied by the same
> axis's margin at the **immediately adjacent saved checkpoints on both sides** (or a statement
> that none exist). An accept whose axis moves by more than that axis's measured neighbour
> range, without the neighbours also accepting, is **reported as checkpoint-selection dependent,
> not as a certified recovery.**

Three scope terms, each checked against the ladder:

| §2.0.2 term | ladder status | verdict |
|---|---|---|
| "any `NI(Δ)` accept" | the ladder's accept is `margin_pp = diff_lower95_one_sided_pp + delta_pp > 0` via the **imported** `ni_rule` (`a04_shallow_rung_ladder_ni.py`, prereg §4) — the identical function §2.0.2 governs | **IN SCOPE** |
| "reported by this gate" | §2.0.2 sits inside `A04_GATE_DESIGN.md` §2 ("Kill condition"), whose `NI(Δ)`/`Δ` definitions the prereg §4 imports rather than redefines ("The rule, the axes, the margin and the anchor are all **imported, not re-derived**"). The ladder is an instrument of this gate, not a separate gate. | **IN SCOPE** |
| "must be accompanied by … (or a statement that none exist)" | a duty on the *report*, not on the *experiment design* | **binding on tonight's verdict** |

§2.0.2 is additive to §2 and predates the ladder (added 2026-08-13, empirical basis
`neighbour_variability_20260813` + `full32_trajectory_ni_20260813`). Nothing in the
ladder prereg amends or supersedes it, and the prereg **never cites** §2.0.2 —
`grep -n "GATE_DESIGN\|2\.0\.2" A04_SHALLOW_RUNG_LADDER_PREREG.md` returns no hit.
An un-cited precondition is not a repealed one.

### 1.2 Prereg §5.5's own scope sentence

> 5. **No trajectory / monotonicity / neighbour claim.** `save_every 2500` yields
>    step2500 and step5000 only; a 2-point series has one difference and cannot
>    support a trend.

§5 is titled **"What this pass will NOT claim"**. Every item in it is of the form
"we will not assert X" — a self-denial that *narrows* what the authors may say. §5.5's
stated reason is that 2 points "cannot support a **trend**", and its object is a
**claim**: trajectory, monotonicity, neighbour.

**MAIN's framing is that this collides with §2.0.2. It does not, and the distinction is
the whole answer:**

* §5.5 forbids the ladder from *asserting* something about neighbours (e.g. "the margin
  improves from step2500 to step5000", "the arm is converging", "the accept is stable").
* §2.0.2 requires the ladder, *if it accepts*, to **disclose** the adjacent-checkpoint
  margin — or state that none exists.

A disclosure is not a claim. Reporting "step2500's margin on this axis is *m*, and 2500
steps is too coarse to be called a neighbourhood" satisfies §2.0.2 (the number is
accompanied) **and** honours §5.5 (no trend, no monotonicity, no stability is asserted).
The ladder's own code already draws this line explicitly: `RANGE_CONSTANTS_DECLARED_UNUSED`
carries `"used_in_this_analysis": False` while still *recording and self-testing* c₂/c₃/c₈
"so nobody can later reuse a wrong c_k". That is disclosure-without-claim, already
implemented, already committed.

**Consequence.** §5.5 does **not** make a Branch A accept inadmissible. What §5.5 *does*
do is forbid the one escape that would have been tempting after the fact — quoting the
step2500→step5000 difference as evidence that the accept is *robust*. That direction stays
closed.

### 1.3 Does "(or a statement that none exist)" cover us?

**No — it covers exactly half of the requirement, and half is what we must report.**

§2.0.2 says "immediately adjacent saved checkpoints on **both sides**". For step5000:

* **lower neighbour: EXISTS.** `step2500.pt` is on disk on both nodes (verified: `.73`
  17,819,242,212 B written 20:56:43 +0800; `.82` 17,013,823,232 B written 20:51).
* **upper neighbour: DOES NOT EXIST**, and cannot. `max_steps=5000`, and the trainer's
  save condition is `step % save_every == 0 and step > 0` with a terminal
  `_save(..., final=True)`. `final.pt` is written at **the same step** as `step5000.pt`
  (`_save` names by `"final" if final else f"step{step}"`, same `step` value), so it is
  the *same point*, not an upper neighbour.

So the escape hatch applies **to the upper side only**. The lower side has a real
checkpoint and therefore a real duty — a "none exist" statement covering both sides
would be **false**. This is precedented, not invented: `a04_full32_trajectory_ni.json`
already emits, for the trajectory's terminal points,
`"lower"/"upper": {"exists": false, "why": "no saved checkpoint on this side of the
trajectory (last point)"}` with `n_neighbours_present: 1`, and the verdict prose records
"`step5000` (both axes) and `step25000` have only ONE neighbour each — they are the
trajectory endpoints. **Stated, not silently treated as satisfied.**"

**And the precondition is already implemented in code** —
`a04_full32_trajectory_ni.py:neighbour_precondition()` — with the one-sided case handled
by `n_neighbours_present` + `all_present_neighbours_also_accept`. The reconciliation
below is therefore a *reuse* of an existing, exercised implementation, not a new rule.

### 1.4 The one thing the existing implementation gets wrong for our case

`neighbour_precondition()` collapses to a **binary**:

```python
rec["verdict"] = ("ACCEPT_SURVIVES_ITS_NEIGHBOURS" if rec["all_present_neighbours_also_accept"]
                  else "ACCEPT_IS_CHECKPOINT_SELECTION_DEPENDENT")
```

Two defects when transplanted to a 2-point, single-sided, 2500-step-spaced arm:

1. **It never consults the movement magnitude.** §2.0.2's text conditions the adverse
   label on "an accept whose axis **moves by more than that axis's measured neighbour
   range**". The full32 implementation drops that clause and labels on the neighbour's
   accept/reject boolean alone. On a 5-point 5000-step trajectory with 4 accepting
   `triviaqa` cells that was harmless; here it would let a **sub-noise** move (a move
   smaller than its own k=2 floor) brand an accept as selection-dependent.
2. **It cannot distinguish one-sided from two-sided.** `n_neighbours_present == 1` still
   returns `ACCEPT_SURVIVES_ITS_NEIGHBOURS`, i.e. a *stronger* label than a two-sided
   check earned. For a terminal checkpoint on a 5-point curve that was defensible; for an
   arm whose **only** neighbour is on one side it silently claims a check that was never run.

§2 fixes both, without altering §2.0.2's text or any threshold.

---

## 2. THE LABEL RULE — mechanical, fixed now, no free parameter

Applies **per (arm, axis)**, only to cells where `ni_accept == True` on a **decision
axis** (`triviaqa`, `popqa`, `mmlu_content`; `nq_open` is design-demoted and carries zero
weight, so §2.0.2 is not invoked for it — it is reported descriptively as always).

### 2.1 Inputs (all already produced by the ladder's own estimator)

For accepting cell `(arm, axis)` at step5000, and the same `(arm, axis)` at step2500:

* `m5 = margin_pp(step5000)`, `m25 = margin_pp(step2500)` — from the imported `ni_rule`,
  `margin = diff_lower95_one_sided_pp + delta_pp`.
* `a5 = ni_accept(step5000) (== True by construction)`, `a25 = ni_accept(step2500)`.
* `se5`, `se25` = each cell's own `bootstrap_se_pp`, i.e.
  `(diff_mean_pp − diff_lower95_one_sided_pp) / 1.6448536269514722`
  (`a04_shallow_rung_ladder_ni.py` lines ~874-875 — the estimator already in the script).

### 2.2 The k=2 gate

```
c2      = 2 / sqrt(pi)          = 1.1283791670955126     # EXACT for the normal
sigma   = (se5 + se25) / 2                                # mean of the two cells' own SEs
gate_pp = c2 * sigma
move_pp = abs(m5 - m25)                                   # k=2: range == |difference|
move_exceeds_floor = (move_pp > gate_pp)
```

**Why k=2 and not k=3.** There are exactly **two** checkpoints (`save_every=2500`,
`max_steps=5000`). `E[range of k iid N(0,σ)]/σ` is a function of k:
k=2 → `2/√π = 1.1283791670955126`; k=3 → `3/√π = 1.6925687506432689`; k=8 → ≈2.8472
(Monte Carlo). Using 1.6926 for k=2 would inflate the floor by **50.0 %**
(1.6926/1.1284) and could suppress a real move; the mirror-image error (c₃ for k=8)
made a floor **40.6 % too low** and manufactured a finding
(`A04_KEEP12_TRAJECTORY_MONOTONICITY_VERDICT`). The constants are already
self-tested in the ladder script (`RANGE_CONSTANTS_DECLARED_UNUSED`); this document
turns `used_in_this_analysis` from `False` to `True` **for the c₂ entry only**, and only
inside the §2.0.2 disclosure — no other statistic in the pass becomes range-based.

**Why σ is the mean of the two cells' own SEs, and not a pooled cross-axis mean.**
`A04_GATE_DESIGN.md` §2.0.2 itself quotes a k=2 example — full32 `mmlu_content`
"**0.1353 pp against a `k=2` gate of 0.2540 pp → FAILS**" — and `PROPOSAL.md` §10 records
its recipe verbatim: `abs(0.914222 − 1.049530) ÷ (1.128379 × mean of the two SEs)`.
Recomputed from `evidence/a04_full32_trajectory_ni.json`: `|0.9142216208517306 −
1.0495299814841190| = 0.135308 pp`, `1.1283791670955126 × (0.22080134105902507 +
0.22946021717898685)/2 = 0.254033 pp`, ratio **0.5326 → FAIL**. Exact match. The
**per-cell-SE** convention is therefore the one already in the gate document, and it is
adopted here unchanged. The alternative that was tried and *retracted* is the pooled
one — `MARGIN_TRAJECTORY_INSTABILITY_NOVELTY_20260813.md` §C2's `1.12838 × 0.3723 =
0.4201 pp` against a hand-computed `0.2280 pp`, which `PROPOSAL.md` §4.3 corrects with
"Its conclusion is right; its inputs were **1.69× off**". Do not pool.

> **⚠️ Provenance note on the §2.0.2 example, so it is not mis-cited as a template.**
> The `0.1353 pp` pair is `step15000` vs `step25000` — **10 000 steps apart, i.e. NOT
> adjacent** on a 5000-step grid. §2.0.2 calls it "the comparable 2-point
> **endpoint-neighbour** value on mmlu_content", but full32's actual *adjacent*
> endpoint pair is `step20000`/`step25000`: range `|0.6222404215923658 −
> 1.0495299814841190| = 0.427290 pp` against gate `0.261361 pp` → ratio **1.6349,
> PASSES**. So §2.0.2's illustrative arithmetic is right, its *label* ("endpoint-neighbour")
> is not, and the truly adjacent pair points the **opposite** way. This does not change
> anything §2.0.2 requires — the example is decoration, the PRECONDITION is the rule —
> but it does mean **the recipe is imported from §2.0.2, the step pair is not.** Ours is
> `step2500`/`step5000`, which is genuinely adjacent by construction (they are the only
> two saves).

### 2.3 The three labels

```
if not a25:
    if move_exceeds_floor:  label = "ACCEPT_IS_CHECKPOINT_SELECTION_DEPENDENT"
    else:                   label = "ACCEPT_SINGLE_SIDED_NEIGHBOUR_ONLY"   # + sub_floor flag
elif a25:
    label = "ACCEPT_SINGLE_SIDED_NEIGHBOUR_ONLY"                            # + lower_also_accepts
```

Written as an exhaustive table over the two booleans:

| # | `a25` (step2500 accepts) | `move_exceeds_floor` | LABEL | may the verdict say "certified recovery"? |
|---|---|---|---|---|
| 1 | **True** | any | `ACCEPT_SINGLE_SIDED_NEIGHBOUR_ONLY` (+`lower_neighbour_also_accepts=true`) | **No** — strongest available, but one side unchecked |
| 2 | False | **True** | `ACCEPT_IS_CHECKPOINT_SELECTION_DEPENDENT` | **No** |
| 3 | False | False | `ACCEPT_SINGLE_SIDED_NEIGHBOUR_ONLY` (+`neighbour_rejects_but_move_is_sub_floor=true`) | **No** |

**`CERTIFIED` is UNREACHABLE for this ladder, and that is decided now, not later.**
§2.0.2 conditions a certified reading on "the immediately adjacent saved checkpoints on
**both sides**". The upper side does not exist and cannot be made to exist without new
training. Therefore **no cell in tonight's verdict may be labelled `CERTIFIED`**, whatever
the numbers are. Option (a) in MAIN's dispatch is pre-emptively excluded on structural
grounds — no datum can change it.

**Why row 1 is not `CERTIFIED` even though the neighbour also accepts.** Calling a
one-sided check by the same name as a two-sided one is exactly the defect in
`neighbour_precondition()` §1.4(2). full32's `step5000` accept was labelled
`ACCEPT_SURVIVES_ITS_NEIGHBOURS` on **one** neighbour; the JSON is honest
(`n_neighbours_present: 1`), the label is not. We do not repeat that.

**Why row 3 is not the adverse label.** §2.0.2's adverse clause is conjunctive: an accept
is selection-dependent when it "**moves by more than** that axis's measured neighbour
range, **without** the neighbours also accepting". Both limbs must hold. A move inside
its own k=2 floor is not a measured move at all — `A04_GATE_DESIGN.md` §2.0.2's own
empirical basis says "every range must be gated on `range_exceeds_item_noise` before it
may be quoted", and 7 of 8 (1 of 6 on decision axes) of the keep8/shortgpt16 ranges are
inside item noise and "are NOT evidence of anything". Firing the adverse label on a
sub-floor move would convert noise into a demotion — the mirror of the error that voided
the within-arm-LR pass (`within_arm_lr_refutation_20260813`: a ratio of two ranges,
neither clearing its floor, is UNDEFINED, not a direction).

### 2.4 The blanket rule that follows

If **any** decision-axis cell of the accepting arm carries
`ACCEPT_IS_CHECKPOINT_SELECTION_DEPENDENT`, that **axis** is flagged. The arm-level
`NI_ACCEPT` verdict (≥2/3) is **not** overturned by the flag — §2.0.2 is a precondition on
*reporting*, and it explicitly "does not alter K1/K2/K3 or any threshold". The verdict
stands with its label attached. Precedent: `triviaqa|full32@step20000` was flagged and its
margin `+2.4504 pp` remained in the record.

---

## 3. Is 2500 steps a "neighbourhood"? — the question MAIN asked, answered against us

**No. 2500 steps at `max_steps=5000` is NOT a neighbourhood, and the k=2 gate must not be
sold as if it were.**

§2.0.2 already warns about exactly this failure mode, and the warning is quoted here
because it applies to us more strongly than to the case it was written for:

> ⚠️ Do **not** convert the full32 5-point spread into a neighbour range: its `k=5` ranges
> … span **25 000 steps of training progress**, not a neighbourhood.

The measured evidence base is:

| source | spacing | as fraction of that run's horizon | LR change across the pair |
|---|---|---|---|
| keep8 cluster2 (130000/130500/131000) | **500** | ~0.24 % of 200k+ | negligible |
| keep10 (89000/89500/90000) | **500** | ~0.25 % | negligible |
| full32 trajectory | **5000** | 20 % of 25k | material |
| **this ladder (step2500/step5000)** | **2500** | **50 % of 5000** | **1.144e-05 → 2.000e-06, factor 5.72** |

Recomputed from the trainer's own schedule (`get_lr` imported into
`train_olmo2_arch_probe2.py:91`; cosine `min_lr + 0.5·(base−min)·(1+cos(π·progress))`,
`base=2e-5`, `min=2e-6`, `warmup=150`, `max_steps=5000`):

* step2500 → cosine progress `(2500−150)/(5000−150) = 0.484536`, LR **1.143706e-05**
  (57.19 % of peak)
* step5000 → LR **2.000000e-06** (10.00 % of peak, the floor)
* ratio **5.7185×**

The two checkpoints are therefore **not two samples of one converged state**. step2500 sits
mid-cosine at over half peak LR; step5000 sits at the LR floor after full decay. The
difference between them is dominated by **training progress and LR annealing**, which is
precisely the confound §2.0.2 forbids importing.

**Binding consequences, all in the conservative direction:**

1. **`move_exceeds_floor == True` is NOT evidence of instability.** At this spacing it is the
   *expected* signature of a half-cosine of additional annealed training. The label rule
   §2.3 uses it only as a **necessary** condition for the adverse verdict (it can never, by
   itself, produce one — row 2 also requires `a25 == False`), never as sufficient evidence
   of jitter.
2. **`move_exceeds_floor == False` is the informative direction.** If half a cosine of
   training moves the margin by less than its own k=2 item-noise floor, that is a genuine
   (if weak) stability statement about the endpoint.
3. **The 2500-step move may NOT be compared numerically to the 500-step keep8/keep10 ranges.**
   Different spacing, different scale (7B vs 1B), different corpus. Any sentence of the form
   "our 2500-step move is larger/smaller than keep8's 1.1202 pp" is **prohibited**. The
   keep8/keep10 numbers are 7B (`gate: ..._7B`, anchor `OLMo-2-1124-7B`); **no 1B neighbour
   range has ever been measured**, so no 1B empirical floor exists to compare against —
   the c₂ gate is a closed-form item-noise floor, not a measured 1B neighbour range.
4. **Do not write "wider spacing is more convincing".** It is not. Wider spacing makes the
   comparison *weaker* as a neighbour check, because it admits progress as an explanation.
   MAIN flagged this intuition as a trap; it is a trap, and it is banned here by name.

**One-process provenance verified** (§2.0.2 trap 2 — a neighbourhood must not straddle a
resume seam). `grep -c resume` on both progress logs returns **0**; both runs are single
uninterrupted processes from `step 20` to now. And `step2500.pt` will survive to analysis
time: `select_rotation_victims(['step2500.pt','step5000.pt'], keep_last_n=3,
milestone_every=5000, just_written='step5000.pt')` → `[]` (executed).

---

## 4. Per-axis, not blanket — and how the axes differ

§2.0.2, bold in the original:

> **Stated PER-AXIS, not blanket.** The effect is axis-concentrated: only triviaqa cleared the
> noise floor. Blanket distrust of single-checkpoint numbers is **not** supported by this
> evidence, and claiming it would overstate what was measured.

So the §2.0.2 disclosure runs **once per accepting decision-axis cell**, and a flag on one
axis says nothing about another. The measured axis-concentration, both arms, decision axes
only:

| axis | keep8 cluster2 (500 steps) | keep10 (500 steps) | clears its floor? |
|---|---|---|---|
| `triviaqa` | 1.2149 pp / floor 0.6595 → **1.84×** (keep10); 1.1202 / 0.6577 → **1.70×** (keep8) | same | **YES on both arms** |
| `popqa` | 0.2523 / 0.5818 → 0.43× | 0.3151 / 0.5843 → 0.54× | no |
| `mmlu_content` | 0.2208 / 0.6522 → 0.34× | 0.1852 / 0.6351 → 0.29× | no |

(`evidence/a04_neighbour_variability.json:leg_A_decision_axis_margin_ranges_clean_cluster`,
`evidence/a04_keep10_neighbour_range.json:Q1_decision_axis_summary`. Both 7B.)

**`mmlu_content` is treated no more leniently in the label rule, and no more harshly.** The
same §2.3 table runs on it. What differs is the **caveat carried alongside**:

* `mmlu_content` is **length-free by construction** (prereg §5.8) and is the axis whose
  accepts *survived* their neighbours on full32 — all 5 of 5, per
  `neighbour_precondition_2_0_2`. It has never cleared a neighbour floor on any arm. An
  `mmlu_content` accept flagged `ACCEPT_IS_CHECKPOINT_SELECTION_DEPENDENT` would be the
  **first** such event in A04 and must be reported as surprising, not as routine.
* `triviaqa` / `popqa` are **generative EM** and inherit A04's two verbosity
  demonstrations (47.37 % of an EM loss; 50.00 % of an EM gain, which **reordered two
  arms**). An accept on these axes carries the format caveat **in addition to** whatever
  §2.0.2 label it earns. The two caveats are independent and neither substitutes for the
  other.

**No axis is exempted, and no blanket distrust is licensed.** Specifically forbidden:
"single-checkpoint 1B margins are unreliable" (unmeasured at 1B), and
"`mmlu_content` needs no neighbour check" (§2.0.2 scopes by axis, not by trust).

---

## 5. One-sentence conclusion

> **Tonight's verdict may legitimately record `pilot_two_status`'s blocker as DISCHARGED if and
> only if (i) `keep13` or `keep14` shows `NI(Δ)` ACCEPT on ≥2 of the 3 decision axes under the
> pre-registered `split` convention, and (ii) every accepting decision-axis cell is reported
> together with its `step2500` margin on the same axis, labelled by §2.3 — with `CERTIFIED`
> structurally unavailable (no upper neighbour exists) — and (iii) no cell's label is
> `ACCEPT_IS_CHECKPOINT_SELECTION_DEPENDENT` on a number of axes that would drop the surviving
> accepting-axis count below 2.**

Restating (iii) mechanically, since it is the only clause that can bite: the flag does **not**
overturn `NI_ACCEPT` (§2.4 — §2.0.2 alters no threshold), so the blocker discharges on the
≥2/3 count as pre-registered. But a verdict in which **all** accepting axes are flagged must
say so in its headline, because "a rung exists where NI can be OBSERVED TO ACCEPT" is then true
only of a checkpoint chosen at the LR floor of a 5000-step run — which is a weaker claim than
the blocker's plain reading, and the difference must be visible to the reader who prices
1,077–4,309 GPU-h against it.

**And the symmetric statement, so this document cannot be read as pro-accept:** under
**Branch B** (both arms constant-REJECT) §2.0.2 is **not triggered at all** — it gates accepts
only. The full32 pass recorded the right phrasing for that case and it is adopted verbatim:
*"no cell accepts on any axis, so §2.0.2 has nothing to gate. The precondition is not vacuously
satisfied — it is not triggered."* Branch B needs nothing from this document.

---

## 6. What this document does NOT do

1. **Does not amend §2.0.2, §2's kill clauses, K1/K2/K3, Δ, the anchor, the axes, the ≥2/3
   bar, or any threshold.** It selects among readings that §2.0.2 already permits and fixes
   the choice pre-data.
2. **Does not license a trend, monotonicity, convergence or stability claim** from
   step2500→step5000. Prereg §5.5 stands in full. The step2500 margin is a **disclosure**
   under §2.0.2 and nothing else.
3. **Does not license a σ_run, K2, seed-variance or `sd_run` statement.** One seed per arm.
4. **Does not compare the 2500-step move to any 500-step or 5000-step range**, or to the 7B
   arms (§3 consequence 3).
5. **Does not spend GPU.** No step2500 eval is requested or authorised by this file. If the
   `step2500` cells are wanted, that is a separate ~5 min/arm eval on an allowed node and needs
   its own dispatch — and if it is **not** run, §2.3 is unevaluable and the verdict must then
   state, per §2.0.2's own escape-hatch grammar, that *the lower neighbour exists on disk but
   was not scored*. That is a weaker disclosure than scoring it, and it is the honest one.
   **Which of these two the verdict uses is decided by whether the eval is run, not by what the
   step5000 numbers turn out to be.**
6. **Does not touch** `a04_shallow_ladder_chain.sh`, `a04_shallow_rung_ladder_ni.py`,
   `a04_shallow_ladder_eval_driver.sh`, `STATUS.json`, or any process on `.73`/`.82`/`.104`/
   `LOCAL`/`.21`. The ladder run and its chain watchers are owned by another agent.

---

## 7. Provenance — every number above is recomputable

| claim | source | how |
|---|---|---|
| §2.0.2 verbatim text | `A04_GATE_DESIGN.md` lines 185-189 | read directly |
| prereg §5.5 verbatim text | `A04_SHALLOW_RUNG_LADDER_PREREG.md` lines 264-266 | read directly |
| prereg never cites §2.0.2 | same | `grep -n "GATE_DESIGN\|2\.0\.2" ` → no hit |
| precondition already implemented, one-sided case handled | `code/a04_full32_trajectory_ni.py:381-413` | `neighbour_precondition()`; `n_neighbours_present`, `all_present_neighbours_also_accept` |
| full32 one-sided cells reported as such | `evidence/a04_full32_trajectory_ni.json:neighbour_precondition_2_0_2` | `triviaqa\|step5000`, `mmlu_content\|step5000`, `mmlu_content\|step25000` → `n_neighbours_present: 1` |
| the live flag precedent (+2.4504 / −0.6035) | same | `triviaqa\|step20000` → `ACCEPT_IS_CHECKPOINT_SELECTION_DEPENDENT` |
| c₂ = 1.1283791670955126 | closed form `2/√π`; self-tested in `code/a04_shallow_rung_ladder_ni.py:RANGE_CONSTANTS_DECLARED_UNUSED` | `python -c "import math;print(2/math.sqrt(math.pi))"` |
| §2.0.2's k=2 example reproduces exactly (0.135308 / 0.254033 → 0.5326 FAIL) | `evidence/a04_full32_trajectory_ni.json:discrimination_curve.split.mmlu_content.per_step` | `abs(0.9142216208517306−1.0495299814841190)` ÷ (`1.1283791670955126` × mean(`0.22080134105902507`, `0.22946021717898685`)) |
| the truly adjacent full32 endpoint pair PASSES (1.6349×) | same | steps 20000/25000: `abs(0.6222404215923658−1.0495299814841190)=0.427290` ÷ `1.1283791670955126`×mean(`0.2337896552389677`,`0.22946021717898685`)=`0.261361` |
| pooled-SE recipe is the retracted one | `PROPOSAL.md` §4.3; `../../shared/literature/MARGIN_TRAJECTORY_INSTABILITY_NOVELTY_20260813.md` §C2 | "Its conclusion is right; its inputs were 1.69× off" |
| LR 1.143706e-05 @2500 vs 2.000000e-06 @5000, ratio 5.7185 | `scripts/train_continued_pretrain.py:481` `get_lr`, imported at `scripts/train_olmo2_arch_probe2.py:91`; hypers from prereg §3 | cosine `min_lr+0.5(base−min)(1+cos(π·progress))`, progress `(2500−150)/(5000−150)=0.484536` |
| `final.pt` is the same step as `step5000.pt`, not an upper neighbour | `scripts/train_olmo2_arch_probe2.py:521` + `:1076` | `name = "final" if final else f"step{step}"`; terminal `_save(..., final=True)` at the same `step` |
| step2500 survives rotation | `scripts/ckpt_rotation.py:select_rotation_victims` | executed: `(['step2500.pt','step5000.pt'], keep_last_n=3, milestone_every=5000, just_written='step5000.pt')` → `[]` |
| step2500.pt exists on both arms | `ls -la` on `.73` and `.82`, zwfy6 | `.73` 17,819,242,212 B @20:56:43; `.82` 17,013,823,232 B @20:51 |
| one-process provenance, no resume seam | both progress logs on zwfy6 | `grep -c resume logs/a04_shallow_keep1{3,4}_seed101.log` → `0`, `0` |
| axis-concentration table (7B) | `evidence/a04_neighbour_variability.json:leg_A_decision_axis_margin_ranges_clean_cluster`; `evidence/a04_keep10_neighbour_range.json:Q1_decision_axis_summary` | `range_pp`, `expected_range_if_pure_noise_pp`, `range_exceeds_item_noise` |
| no 1B neighbour range has ever been measured | all four neighbour/trajectory JSONs | every `gate` string ends `_7B`; every `intact_anchor.choice` is `vanilla models/OLMo-2-1124-7B` |
| Branch B phrasing | `code/a04_full32_trajectory_ni.py:717-720` | the `if not neighbour:` branch |
| pre-data step counts 3740 / 3900 | live logs, 2026-08-13 14:06:10 UTC | `grep -oE 'step +[0-9]+' … \| tail -1` on each node |
| `bootstrap_se_pp` estimator | `code/a04_shallow_rung_ladder_ni.py:874-875` | `(diff_mean_pp − diff_lower95_one_sided_pp)/1.6448536269514722` |
