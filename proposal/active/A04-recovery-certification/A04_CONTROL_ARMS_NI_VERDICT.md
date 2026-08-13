# A04 — the two never-tested REPAIR-MODE controls: P3 fires, but on the metric, not on the mechanism

**Verdict string:** `P1_VIOLATED_ON_POPQA_BUT_THE_VIOLATION_IS_FORMAT_NOT_KNOWLEDGE__ZERO_INHERITANCE_FLOOR_REACHES_32_TO_40_PCT_OF_THE_INTACT_RESIDUAL`

**Date:** 2026-08-13 · **GPU: 0 GPU-h.** Every input is a per-example shard set
already on zwfy6, written 2026-08-02. No model loaded, no CUDA context, no
scoring. Analysis is CPU-only, read-only on every input.
**Node of record:** `.73` (8×H20, zwfy6, **numpy 2.5.1**), verified 8×0 MiB before
launch; the driver refuses to start if any GPU holds > 8000 MiB and refuses to
run on `.104` or `.21` by IP.
**Independently re-run on `.82`** (numpy **2.4.6**) — all three P-verdicts and
the popqa margin difference identical to 4 dp.
**Not touched:** `LOCAL` / `.21` (SparseForge #246), `.104` (paperC Qwen3 heal).
**Pre-registration:** `A04_CONTROL_ARMS_NI_PREREG.md`, committed as its own
commit (`e51f390`) **before the first margin existed**.
**Evidence:** `evidence/a04_control_arms_ni.json`
(sha256 `268187ce6beca8585689c863f20a32e144fa6cd09202822d569065d9adc5952c`,
md5 `3178229782440993b9baeac43bd2b611`, verified byte-identical on both disks)
**Code:** `code/a04_control_arms_ni.py`, `code/a04_control_arms_ni_driver.sh`

---

## 0. Answers, in one block

| Q | Answer |
|---|---|
| **P1** — is FF ≤ train-all on every axis? | **`P1_VIOLATED`.** 2 of 3 decision axes satisfy it (triviaqa −4.62 pp, mmlu_content −2.30 pp), but **popqa violates it by +1.78 pp = 3.79 pooled SE**. Identical under **all five** tie conventions. The sharper paired item bootstrap agrees and is resolved: **+1.7663 pp, CI95 [+1.3107, +2.2289], p = 0.0001**, 674 wrong→right vs 422 right→wrong. |
| **P2** — is FS the worst arm and a floor? | **`P2_HOLDS`**, 5/5 conventions. FS is the lowest-margin arm on **3 of 3** decision axes and rejects **3 of 3**. |
| **P3** — does the falsifier fire? | **FIRES on popqa** — but split into its two clauses, **only clause 1 survives**. Clause 1 ("FF's margin exceeds train-all's on the axis the gate actually decides on") is a measured fact. Clause 2 ("freezing the trunk recovers more *knowledge*") **does NOT survive**: the +1.7663 pp EM move decomposes exactly into **FORMAT +1.5350 pp** (CI [+1.2546, +1.8296]) **+ CONTENT +0.2313 pp** (CI [−0.1121, +0.5818], **p = 0.1972, NOT resolved**). |
| **Q1** — 4-axis margins + neighbours | Table in §2. **0 accepts, 3 arms × 4 axes × 5 conventions.** Neighbours: **none exist** — and the checkpoint that looked like one is from a *different run* (§3). |
| **Q2** — how far is the FS floor from null? | **FS is resolvedly ABOVE its own best-constant null on all 4 axes** (p = 0.0001 each). It reaches **32.55 %** (triviaqa), **11.58 %** (popqa), **40.47 %** (mmlu_content), **28.89 %** (nq_open) of the *intact* calibrated residual — **with zero inheritance.** This **compresses A04's "recovery" space directly**: train-all's headline 53.06 % mmlu_content recovery contains only a **+12.59 pp** inheritance premium over that floor, and FF's mmlu_content premium is **+0.34 pp, p = 0.8852, NOT resolved**. |
| **Q3** — does the ordering support "current rules accept what we reject"? | **No — it supplies zero new support.** All three arms REJECT under **both** NI **and** `RATIO(0.85)` (mean ratios 0.4728 / 0.4478 / 0.3996 vs ρ = 0.85). **Zero rule-disagreement cells.** And `PLATEAU(T)` is **not computable** for FF/FS — no PPL on disk — which was registered as a design limitation *before* scoring. |

**The one-line scientific summary.** The two controls do **not** produce the
accept A04 needs, so **Pilot Two still should not be approved** — but they
produce two findings that are worth more than another rung would have been:
(a) the gate's own decision metric **reorders the arms** in a direction the gate
design assumes impossible, and the reordering is **~87 % output format**; and
(b) a zero-inheritance model of the same shape and budget already sits at
**32–40 %** of the intact residual on 3 of 4 axes, so a large part of what A04
has been calling "recovery" is not attributable to recovery at all.

---

## 1. What was tested, and why it is not another rung

Neither arm had ever entered an A04 evidence file (`grep -rilE
'freezefront|fromscratch|frozen_front|scratch16'` over `evidence/` + `code/` +
`*.md` → **0 hits**).

| arm | construction | trainable | effective LR |
|---|---|---|---|
| **train-all** `keep14fresh2` | front 14 transplanted + 2 fresh, all trained | 4060.4 M / 4060.4 M | uniform **2e-5** |
| **FF** `..._freezefront` | front 14 transplanted **and frozen** | **1226.9 M** / 4060.4 M | uniform **2e-5** |
| **FS** `..._fromscratch` | base ignored, all 16 layers random-init | 4060.4 M / 4060.4 M | uniform **1e-4** |

All three are `keep_front=14, n_fresh=2, 16 layers`. They vary **repair mode**,
not depth. **They are never tabulated as a depth ladder.**

`apply_freeze_front` (`scripts/train_olmo2_arch_probe2.py:397`) was read, not
assumed: it sets `requires_grad_(False)` on exactly `model.layers.{0..13}.*`, so
FF is *the same injury with strictly less repair capacity*. That is what makes P1
a real prediction rather than a guess.

### 1.1 The comparison is corpus- and budget-matched — verified from the authoritative logs

| arm | first banner | `dataset rows=` | geometry | steps | resumes |
|---|---|---|---|---|---|
| train-all | 2026-07-16 21:36:20 | **7 570 911** | `bs=16 gaccum=1 eff_bs=128` | 200 000 | 4 |
| FF | 2026-07-25 12:15:48 | **7 570 911** | `bs=16 gaccum=1 eff_bs=128` | 200 000 | 1 |
| FS | 2026-07-21 02:00:06 | **7 570 911** | `bs=16 gaccum=1 eff_bs=128` | 200 000 | 0 |

Same corpus, same `eff_bs`, same `seq_len`, same step count, same disk, same
fp32 AdamW, same node class. **`STATUS.json:warning`'s two-corpora confound is a
DEPTH-ladder confound and does not apply here** — these are one depth on one
corpus. Registered in the prereg before scoring so it cannot be claimed
post-hoc.

### 1.2 ⚠️ The LR confound, registered in advance

`_classify_param` (trainer:436) returns `"fresh"` **first** under
`--from_scratch`, so FS's entire parameter set landed in the `fresh` group at
`lr_fresh=1e-4`. Read from the `[optim] group` lines:

- train-all: `inh_decay 4060.1M @2e-5` + `inh_nodecay 0.3M @2e-5`
- FF: `inh_decay 1226.8M @2e-5` + `inh_nodecay 0.0M @2e-5`
- **FS: `fresh_decay 4060.1M @1e-4` + `fresh_nodecay 0.3M @1e-4`**

**P1 (train-all vs FF) is LR-matched at 2e-5 and therefore clean.** **P2 and Q2
are not**: FS's floor is jointly attributable to "no inheritance" and "5× LR",
in *both* directions. Every FS number in this document carries that caveat.

Incidentally this also shows the `fresh`-group bug of `PROPOSAL.md §7.2` does
**not** bite this trainer — FF's `inh_decay` is exactly 1226.8 M, i.e. exactly the
unfrozen set, so classification worked. But no arm ever ran *differential* LR:
all three are uniform, at two different values.

---

## 2. Q1 — the margins (`split` convention, pre-registered)

Anchor: vanilla `models/OLMo-2-1124-7B`, **imported** from
`a04_shallow_rung_ni_7b.ANCHOR`, never redeclared, never substituted (G0/G2).
`build_nulls` **imported and called** — no margin here is obtained by subtracting
a recorded null. All four axes CERTIFIABLE under D1–D6 on `split`; **0 of 12
cells retired**.

| arm | axis | acc | recovery | deficit pp | Δ pp | **margin pp** | boot SE | SE to flip | NI |
|---|---|---|---|---|---|---|---|---|---|
| **train-all** | triviaqa | 29.403 % | 46.05 % | 34.1451 | 6.3291 | **−28.4624** | 0.3930 | 72.4 | REJECT |
| | popqa | 7.976 % | 25.31 % | 16.7730 | 2.2457 | **−15.0810** | 0.3366 | 44.8 | REJECT |
| | mmlu_content | 38.321 % | 53.06 % | 8.7381 | 1.8614 | **−7.4749** | 0.3637 | 20.6 | REJECT |
| | *nq_open* | 5.983 % | 27.22 % | 14.5152 | 1.9945 | **−13.5734** | 0.6399 | 21.2 | REJECT |
| **FF** | triviaqa | 24.766 % | 38.73 % | 38.7818 | 6.3291 | **−33.0824** | 0.3828 | 86.4 | REJECT |
| | **popqa** | **9.743 %** | **33.18 %** | 15.0067 | 2.2457 | **−13.3006** | 0.3281 | 40.5 | REJECT |
| | mmlu_content | 36.042 % | 40.81 % | 11.0169 | 1.8614 | **−9.7751** | 0.3767 | 26.0 | REJECT |
| | *nq_open* | 4.958 % | 22.08 % | 15.5402 | 1.9945 | **−14.6537** | 0.6736 | 21.8 | REJECT |
| **FS** | triviaqa | 20.859 % | 32.55 % | 42.6884 | 6.3291 | **−37.0057** | 0.3930 | 94.2 | REJECT |
| | popqa | 4.892 % | 11.58 % | 19.8570 | 2.2457 | **−18.1860** | 0.3494 | 52.0 | REJECT |
| | mmlu_content | 35.978 % | 40.47 % | 11.0810 | 1.8614 | **−9.8748** | 0.3983 | 24.8 | REJECT |
| | *nq_open* | 6.316 % | 28.89 % | 14.1828 | 1.9945 | **−13.3241** | 0.6905 | 19.3 | REJECT |

**0 of 3 decision axes accept, on any arm, under any of the five tie
conventions.** Margins sit **19.3–94.2 bootstrap SE** from flipping: no
realisable perturbation of the item sample changes any verdict.

Under `credit`, `mmlu_content` retires on `D6_delta_finer_than_instrument` for
all three arms (decision family 9 → 6). The verdicts are unchanged; per guard G1
clause (i) the threshold rescales, and the retired cells are **not** reported as
"NI rejected".

### 2.1 Neighbours (§2.0.2 compliance)

**None exist.** Each arm has exactly one scored checkpoint (step 200 000). §2.0.2
explicitly permits *"or a statement that none exist"*. Since **no cell accepts**,
the precondition has nothing to protect here.

The checkpoint that *looked* like a far neighbour is disqualified — see §3, which
is the most important operational finding in this pass.

---

## 3. ⚠️ `freezefront/step23500.pt` is from a DIFFERENT, ABANDONED run — dropped, not demoted

The dispatch anticipated scoring `step23500` as a far neighbour with the caveat
that 176 500 steps is not a neighbourhood. **That caveat understates the
problem.** `logs/olmo2_7B_keep14fresh2_freezefront.log` exists on **both disks
with different content**:

| | zwfy6 copy (162 067 B) | **wzc1 copy (1 368 257 B) ← authoritative** |
|---|---|---|
| first banner | 2026-07-21 02:02:20 | **2026-07-25 12:15:48** |
| geometry | **`bs=4 gaccum=4`** | **`bs=16 gaccum=1`** |
| `dataset rows=` | **15 491 607** | **7 570 911** |
| last step | **23 640** (process dies) | **200 000 + `final.pt`** |
| last save | `step23500.pt` @ 07-23 13:45:20 | `step200000.pt` @ 07-28 21:24:53 |

`outputs/..._freezefront/step23500.pt`'s mtime is
**`2026-07-23 13:45:20.774755372`** — it matches the *abandoned* run's save line
to the nanosecond, and its 26 056 482 807 B size (with optimizer state) belongs
to that run. The wzc1 run wrote its own `step23500` at 07-25 22:40:21 and
**rotated it away**.

**So `step23500` and `step200000` are not two points on one trajectory — they are
checkpoints of two different runs on two different corpora at two different
micro-batch geometries.** Scoring them together would have silently crossed
both. It is **dropped**, not demoted; bootstrap offset 802 is reserved and left
unused.

**Generalisable warning for every future agent:** the two disks' same-named
`logs/*.log` files are **not copies**. Any provenance claim about these arms must
state which disk, and the **wzc1** log is the authoritative one for all three.

### 3.1 The zwfy6 checkpoints are slim eval copies

| arm | wzc1 `step200000.pt` | zwfy6 `step200000.pt` |
|---|---|---|
| train-all | 48 724 467 827 B | 16 241 486 089 B |
| **FF** | **26 056 479 363 B** | 16 241 487 014 B |
| FS | 48 724 467 699 B | 16 241 486 829 B |

FF's smaller wzc1 file is an **optimizer-state artefact** (AdamW moments for only
1226.9 M trainable params), **not** a different architecture — the eval logs show
all three zwfy6 copies loading `179 tensors, strict, num_hidden_layers=16`. The
launchers hard-assert those exact byte counts before running.

---

## 4. P1 / P3 — the violation is real on the decision metric, and it is ~87 % format

### 4.1 The pre-registered statistic

| axis | FF margin | train-all margin | FF − train-all | / pooled SE | P1 |
|---|---|---|---|---|---|
| triviaqa | −33.0824 | −28.4624 | **−4.6199** | −8.42 | OK |
| **popqa** | **−13.3006** | **−15.0810** | **+1.7803** | **+3.79** | **VIOLATION (beyond SE)** |
| mmlu_content | −9.7751 | −7.4749 | **−2.3002** | −4.39 | OK |

`P1_VIOLATED`, identically under all five conventions (the popqa difference is
convention-invariant: the null cancels in the difference).

### 4.2 The sharper paired test agrees

The pre-registered P1 pools two independent SEs; both arms are scored on the
**same items** (`assert_aligned` proves identical `item_id` sequences), so the
paired difference is sharper:

| axis | FF − train-all | CI95 | p | resolved? | flips (w→r / r→w) |
|---|---|---|---|---|---|
| triviaqa | −4.6366 pp | [−5.2497, −4.0125] | 0.0001 | yes | +1149 / −1981 |
| **popqa** | **+1.7663 pp** | **[+1.3107, +2.2289]** | **0.0001** | **yes** | **+674 / −422** |
| mmlu_content | −2.2789 pp | [−2.8986, −1.6593] | 0.0001 | yes | +859 / −1179 |
| *nq_open* | −1.0249 pp | [−1.8006, −0.2493] | 0.0108 | yes | +88 / −125 |

The violation is **not** an artefact of the pooled-SE approximation.

### 4.3 ★ But the move is FORMAT, not KNOWLEDGE — and this is an exact partition

Every EM change between two arms on the same item is one of four kinds. Label
each by whether the *losing* arm's own prediction already **contained** the gold
answer:

```
EM_move  =  [ gains the other arm already CONTAINED  −  losses this arm still CONTAINS ]   ← FORMAT
         +  [ genuinely new gains                    −  genuine content losses        ]   ← CONTENT
```

This is a **partition of the same items**, not a re-scoring; the script asserts
the two parts sum to the observed EM move to < 1e-9 pp and refuses to publish
otherwise. `contains` is used **only to label an item** — it is **never**
substituted for EM as the decision metric (banned by `A04_GATE_DESIGN.md` §4.1).

| axis | observed EM move | **FORMAT** | **CONTENT** | CONTENT p | CONTENT resolved? |
|---|---|---|---|---|---|
| triviaqa | −4.6366 pp | +0.7914 [+0.4737, +1.1090] | **−5.4280** [−5.9519, −4.9209] | 0.0001 | **yes** |
| **popqa** | **+1.7663 pp** | **+1.5350** [+1.2546, +1.8296] | **+0.2313** [−0.1121, +0.5818] | **0.1972** | **NO** |
| *nq_open* | −1.0249 pp | +0.2770 [−0.1108, +0.6371] | **−1.3019** [−1.9945, −0.6094] | 0.0006 | **yes** |

**86.9 % of popqa's violating move is format** (1.5350 / 1.7663), and the content
remainder is **not resolved**. The corroborating counts: of FF's **674** popqa EM
gains, **337 — exactly 50.00 %** — are items whose gold answer train-all's own
prediction *already contained*. FF's mean popqa prediction is **26.42 chars vs
train-all's 49.72** (ratio **0.531**), its `contains` is **1.3247 pp LOWER**, its
top-constant share is **6.932 % vs 1.213 %**, and its distinct-prediction count
**collapses 9190 → 5073**.

So FF did not learn more popqa facts. **FF stopped padding its answers** — and on
an EM axis that is worth +1.77 pp. This is the exact mirror image of
`PROPOSAL.md §4.4`, where full32's EM *fell* while `contains` barely moved.

### 4.4 What survives of P3, clause by clause

The prereg's P3 trigger is **unchanged** and it **fires**. But P3 bundled two
claims, and they separate:

| clause | survives? | why |
|---|---|---|
| **1. "The arm ordering is wrong."** `margin_pp(FF) > margin_pp(train-all)` on a decision axis, resolved. | ✅ **YES** | This is a fact about the statistic the gate **actually decides on**. `A04_GATE_DESIGN.md` §3.2's presumed ordering (A1 > A2 > A3 > A4) is contradicted on popqa **by the gate's own metric.** Mechanism is irrelevant to this clause: the gate reads EM, and EM reordered the arms. |
| **2. "Freezing the trunk recovers more knowledge."** | ❌ **NO** | Requires a positive *resolved* CONTENT part. popqa's is **+0.2313 pp, p = 0.1972**. **May not be claimed.** |

**Registered consequences that still bind (clause 1):**

1. `A04_GATE_DESIGN.md` §3.2's arm ordering is an **untested assumption**, not a
   design fact. A **rung is `(depth, repair mode)`**, not `depth`. `PROPOSAL.md`
   §8 item 2's "whether any damage depth admits an accept is unknown" must widen
   to *depth **and** repair mode*.
2. **A new `must_not_claim` entry is owed** (see §7): A04 may not assert that
   training all layers repairs better than freezing the inherited ones — it is
   **false on popqa** under the gate's own metric.

**Registered consequence that is now WITHDRAWN (clause 2):** the prereg's
speculative positive route — *"the route to an accept may be training the
inherited weights LESS"* — is **not supported**. It rested on P1 being violated
for a knowledge reason. It is not. Recording this explicitly because the prereg
committed to the inference in advance and the data does not license it.

### 4.5 ⚠️ The deeper reading: this is a finding about A04's METRIC

Three of the four axes are generative EM, and this pass now has **two**
independent demonstrations that a generative-EM decision axis in a base-LM regime
partly measures output format:

- `PROPOSAL.md §4.4` (full32): 47.37 % of an EM *loss* was verbosity.
- **here (FF vs train-all): 50.00 % of an EM *gain* was verbosity, and it is
  enough to REORDER TWO ARMS.**

The second is strictly worse for the rule than the first. §4.4's confound moved a
*magnitude*; this one moves an **ordering** — and an ordering is what a
certification rule is *for*. `A04_GATE_DESIGN.md` §4's design implication
("either add a format-insensitive axis whose null is equally well defined, or
pre-register a verbosity diagnostic as a reporting requirement alongside every
generative cell") is no longer optional-sounding: **without it, the gate can be
reordered by output length.**

---

## 5. Q2 — the zero-inheritance floor reaches 32–40 % of the intact residual

**FS is resolvedly above its own best-constant null on all four axes**
(p = 0.0001 each):

| axis | FS acc | null | **FS residual** | CI95 | **as % of intact residual** |
|---|---|---|---|---|---|
| triviaqa | 20.859 % | 0.256 % | **+20.6030 pp** | [+20.0067, +21.1993] | **32.55 %** |
| popqa | 4.892 % | 2.292 % | **+2.6004 pp** | [+2.1518, +3.0210] | **11.58 %** |
| mmlu_content | 35.978 % | 28.445 % | **+7.5328 pp** | [+6.5814, +8.4942] | **40.47 %** |
| *nq_open* | 6.316 % | 0.554 % | **+5.7618 pp** | [+4.9584, +6.6205] | **28.89 %** |

**This directly compresses what "recovery" can mean.** The right reference for an
*inheritance* claim is arm-vs-FS, not arm-vs-null:

| axis | FS floor | train-all recovered | **train-all premium** | FF recovered | **FF premium** |
|---|---|---|---|---|---|
| triviaqa | 32.55 % | 46.05 % | **+13.50 pp** (p = 0.0001, resolved) | 38.73 % | **+6.17 pp** (resolved) |
| popqa | 11.58 % | 25.31 % | **+13.73 pp** (resolved) | 33.18 % | **+21.60 pp** (resolved) |
| mmlu_content | 40.47 % | 53.06 % | **+12.59 pp** (resolved) | 40.81 % | **+0.34 pp, p = 0.8852, NOT resolved** |
| *nq_open* | 28.89 % | 27.22 % | **−1.67 pp**, p = 0.4468, not resolved | 22.08 % | **−6.81 pp**, p = 0.0022, resolved **NEGATIVE** |

Two consequences:

1. **"53.06 % recovered" on mmlu_content is 40.47 points of floor plus a
   12.59-point premium.** Any A04 writeup quoting a recovery *fraction* must ship
   the floor, or it credits inheritance with work that random init plus the same
   corpus and budget already does.
2. **On nq_open, both arms are at or BELOW the zero-inheritance floor**, and FF is
   resolvedly below it. Inheritance bought nothing there.

⚠️ **Caveat, registered before scoring:** FS ran at **5× the LR** of the other two
arms. The floor is LR-confounded in both directions, so every premium above is
correspondingly uncertain. **These are not clean isolations of inheritance.** A
clean version needs the gate design's arm **A3 (`--random_trunk`)** — same LR, same
transplanted embed/norm/lm_head, random trunk only — which **does not exist on
either disk**.

---

## 6. Q3 — the ordering supplies ZERO new support for "current rules accept what we reject"

| axis | descending by margin |
|---|---|
| triviaqa | train-all > FF > FS |
| **popqa** | **FF > train-all > FS** |
| mmlu_content | train-all > FF > FS |

Against `safe_residual_claim`'s second half:

| arm | NI | `RATIO(0.85)` mean ratio | RATIO |
|---|---|---|---|
| train-all | REJECT 3/3 | 0.4728 | REJECT |
| FF | REJECT 3/3 | 0.4478 | REJECT |
| FS | REJECT 3/3 | 0.3996 | REJECT |

**Zero rule-disagreement cells.** All three RATIO means are 0.38–0.45 **below** ρ,
so nothing here is marginal. And `PLATEAU(T)` is **not computable**:
`olmo2_ppl_results/` contains **no** `freezefront` or `fromscratch` directory
(only `7B_keep14_step{0,128000,153500,200000}(_v2)` plus an unrelated
`7B_scratch16L_lr2e5_*` LR-control run — a **different** run, on the **other**
corpus, at uniform 2e-5).

**So Q3's answer is: this experiment cannot support that half of the claim, and
it was registered as unable to before any number was seen.** After this pass the
disagreement evidence is *still* exactly **1 of 5 checkpoints of 1 zero-damage
arm**, with RATIO's margin over ρ of +0.0015 (`PROPOSAL.md §4.3`). Three more
arms, three more REJECT/REJECT agreements.

---

## 7. What this changes, and what A04 must now not claim

### 7.1 Owed `must_not_claim` entries

> **26.** ❌ **"Training all layers repairs a given injury better than freezing
> the inherited front."** **False on popqa** under the gate's own decision metric:
> FF's margin exceeds train-all's by +1.7803 pp (3.79 SE; paired +1.7663 pp,
> p = 0.0001), at matched depth, corpus, budget **and** LR. `A04_GATE_DESIGN.md`
> §3.2's arm ordering A1 > A2 > A3 > A4 is an **untested assumption**.
>
> **27.** ❌ **"Freezing the inherited trunk recovers more knowledge."** The
> converse of 26 is equally forbidden. popqa's CONTENT component is **+0.2313 pp,
> CI [−0.1121, +0.5818], p = 0.1972** — unresolved. **86.9 %** of the move is
> format, and **50.00 %** of FF's EM gains are items train-all already contained.
>
> **28.** ❌ **Quoting any recovery fraction without its zero-inheritance floor.**
> FS reaches **32.55 / 11.58 / 40.47 / 28.89 %** of the intact residual. On
> mmlu_content, train-all's "53.06 % recovered" is 40.47 points of floor plus a
> 12.59-point premium; FF's premium there is **+0.34 pp, p = 0.8852**.
>
> **29.** ❌ **Calling `freezefront/step23500.pt` a checkpoint of the run that
> produced `step200000.pt`.** It belongs to an abandoned run on the **other**
> corpus (15 491 607 vs 7 570 911 rows) at a different micro-batch geometry. See §3.
>
> **30.** ❌ **Any clean "inheritance is worth X" claim from FS.** FS ran at 1e-4,
> the other two arms at 2e-5. The clean control is arm **A3 `--random_trunk`**,
> which does not exist on either disk.

### 7.2 The status table, updated

| component of `safe_residual_claim` | state after this pass |
|---|---|
| NI test against a null-calibrated intact target | built, frozen, discriminating — **and now shown to be reorderable by output format** (§4.3) |
| pre-registered, multi-seed, matched-corpus/token | design exists, **not run** |
| "current rules accept models this rule rejects" | **still exactly 1 of 5 checkpoints of 1 zero-damage arm.** This pass adds 3 arms of REJECT/REJECT agreement and **no** disagreement |
| an accept on a **damaged** arm | **still none, anywhere.** Now **0 accepts across keep8 / keep10 / keep14 / shortgpt16 / FF / FS × 5 conventions** |
| the rung-selection problem | **widened, not solved**: it is `(depth, repair mode)`, and repair mode reorders arms |

### 7.3 Recommendation — **do not approve Pilot Two**

Unchanged in direction, and now for a **second, independent** reason.

1. **The original reason stands.** No damaged arm has ever accepted; shallower
   rungs do not exist; and this pass adds two more constant-REJECT arms. Pilot
   Two (1 077–4 309 GPU-h) would price a gate that has never been observed to
   accept under damage.
2. **A new reason.** The gate's decision metric can be **reordered by output
   length** (§4.3–4.5). Funding 8 more runs to feed a metric with that property
   buys 8 more cells of the same defect. **The metric problem is a design fix,
   not an *n* fix**, and it is now the binding constraint — exactly as
   `PROPOSAL.md §8` item 3 said, but with an ordering flip rather than a
   magnitude as evidence.

**The two cheap next steps this pass identifies** (both stated as candidates, not
as authorised work):

- **Arm A3 (`--random_trunk`) at matched LR 2e-5** — the only thing that turns
  §5's floor into a clean inheritance measurement. One 7B run; the trainer flag
  already exists (`--random_trunk`, line 586, mutually exclusive with
  `--from_scratch`).
- **A format-insensitive decision axis with a well-defined null**, or a
  pre-registered verbosity diagnostic as a *reporting requirement* on every
  generative cell. **0 GPU** for the design; it is the cheapest way to remove the
  defect that §4.3 just demonstrated can flip an ordering.

---

## 8. Verification performed

1. **Archived endpoint reproduced EXACTLY.** train-all@200k re-derived under its
   archived offset 201 returns **dev = 0.00e+00 pp** on all three decision axes
   against `evidence/a04_shallow_rung_ni_7b.json`. The script hard-fails
   otherwise, proving the imported guard/anchor/rule are the objects that
   produced the archive.
   > ⚠️ **This gate caught my own error, and the error is recorded.** The first
   > version of `a04_control_arms_ni.py` hardcoded three reference constants
   > transcribed from `A04_KEEP14_TRAJECTORY_NI_VERDICT.md` §2's 4-dp table
   > (`−28.4624`) with invented trailing digits. The gate fired at 8.82e-05 pp —
   > **the recomputation was right and my constants were wrong** (canonical is
   > `−28.462438698172093`). This is the **fifth** hand-transcription slip of
   > 2026-08-13. The fix was not to loosen the tolerance: the script now **reads
   > the canonical JSON at runtime**, removing the transcription step entirely.
2. **Cross-node reproduction.** Re-run on `.82` (**numpy 2.4.6**) vs `.73`
   (**2.5.1**): archived reproduction 0.00e+00 on both, and P1/P2/P3 plus the
   popqa margin difference (**+1.7803 pp, +3.79 SE**) identical to 4 dp. The
   `multinomial` drift (19/10000 rows, ≤ 0.005294 pp) does not reach any verdict
   here — margins are 19.3–94.2 SE from flipping. **No margin quoted finer than
   0.01 pp** (`must_not_claim[24]`).
3. **Shard integrity, asserted and recorded.** All 16 arm×axis cells: shard index
   set **exactly {0..7}** (not "8 files"), merged `n` exactly `EXPECTED_N`
   (triviaqa 17944 / popqa 14267 / nq_open 3610 / mmlu 14042), **0 duplicate
   `item_id`, 0 nan**, and item_id sequences identical across all four arms
   (`assert_aligned`) — without which the paired differences of §4.2/§4.3 would
   compare different items.
4. **Protocol confirmed from the INVOCATION, fail-closed.** `cb_bs = 32` from the
   launchers' own echoed `START <model> … bs=32` lines in `logs/cb_driver_104.out`
   (FF **and** FS), `logs/cb_driver_73.out` (the train-all endpoint **and** the
   anchor), `logs/nqopen_driver_104.log`, `logs/nqopen_scratch.log`.
   `mmlu_bs = 16` from the launcher **source**: `p06_run_104_transferred.sh` (FF)
   and `p06_run_transferred.sh` (FS/train-all) both leave `BS` unset, and the
   script asserts neither sets it, → `_run_olmo2_mmlu_content.sh:43`
   `BS="${BS:-16}"`. **`add_bos` asserted with `is False`, never `is not True`**
   (which passes silently on `None`); `max_new_tokens == 32` asserted.
   `chat_template` established **structurally** — neither harness has a
   chat-template code path.
   > A **dedicated** asserter was written rather than importing
   > `a04_neighbour_variability.protocol_asserted`: that one requires a
   > `DRIVER START … mmlu_bs=.. cb_bs=..` header only the 2026-08-13 drivers
   > emit. These cells were scored 2026-08-02 by different launchers. Reusing it
   > would have crashed, and "fixing" it by loosening the regex would have
   > weakened the gate for every future caller. **The frozen expectation
   > `{cb_bs: 32, mmlu_bs: 16}` is identical.**
5. **Harness identity.** md5 `2ed41993241226c795a3ca38375933f7` (closedbook) /
   `fe4a62dbdf884a1e2aedc6ed26887b4e` (mmlu_content) — **identical** to the values
   `A04_KEEP14_TRAJECTORY_NI_VERDICT.md` §5.1 item 5 pins for the copies that
   produced the anchor and the endpoint. Asserted, not assumed.
6. **Checkpoint identity confirmed from the eval logs**, which echo
   `[pruned] loaded ckpt step=200000 keep_front=14 n_fresh=2
   num_hidden_layers=16 (179 tensors, strict) from <path>` for **all three**
   arms — proving the evals loaded the step and arch they claim, and that FF's
   smaller wzc1 file is an optimizer-state artefact (§3.1).
7. **Bootstrap seeds disjoint**, checked mechanically against every archived
   block (0-1, 100-102, 200-203, 300-301, 400-408, 500-503, 700-702). New arms
   use 800/801; train-all keeps **archived** 201 so the reproduction assert is
   meaningful; **802 reserved and left unused** for the dropped `step23500`.
8. **The decomposition partition is asserted, not asserted-in-prose.** FORMAT +
   CONTENT must equal the observed EM move to < 1e-9 pp or the script refuses to
   publish.
9. **Refuse-guard exercised.** The driver read 8×0 MiB on `.73` and `.82` before
   starting and refuses on > 8000 MiB, on `.104` and on `.21` by IP. `LOCAL`,
   `.21` and `.104` were never contacted for compute.
10. **`STATUS.json` append verified as an append, not just as a key addition.**
    Final diff: **326 insertions, 0 deletions.** The writer snapshots every
    pre-existing key's serialised bytes, asserts `count == old + 1`, asserts the
    old key **order** is unchanged, asserts each old key is **byte-identical**,
    and asserts the whole new file is a **byte-prefix extension** of the old one
    — restoring the original if anything fails.
    > ⚠️ **Two things worth recording here.**
    > **(a) The key count was a moving target.** The dispatch said 41; my own
    > `json.load` said **42**; the file was at **43** when the key was appended,
    > because a **concurrent** A04 pass (commit `7e54376`,
    > `keep12_trajectory_monotonicity_20260813`) landed its own key while this
    > analysis was running. This is why the writer trusts **no** hardcoded count
    > and derives everything from the file it is about to modify. The new key is
    > the **44th**.
    > **(b) The first write silently reformatted the entire file.** It used
    > `indent=1` (copied from the evidence writers) against a file that is
    > `indent=2`. The per-key byte check **passed** — correctly, since no key's
    > *value* changed — but `git diff` showed **2643 deletions / 2965
    > insertions**: every line rewritten. Reverted from backup; the writer now
    > **derives the file's own `json.dumps` format by round-tripping the original
    > bytes**, and the whole-file byte-prefix assertion was added precisely
    > because the per-key check structurally cannot see a reformat. A per-key
    > equality check is **not** sufficient evidence that an append-only file was
    > appended to.

---

## 9. Licensed vs NOT licensed

### Licensed
- Every accuracy, null, residual, Δ, lo95 bound and margin in §2, and their exact
  reproduction of the archived train-all endpoint.
- **"No realisable perturbation of the item sample flips any of these 12
  verdicts"** — margins are 19.3–94.2 bootstrap SE from accepting.
- **"FF's popqa margin exceeds train-all's, resolved on the item sample"**
  (+1.7663 pp, CI [+1.3107, +2.2289], p = 0.0001) — at matched depth, corpus,
  budget and LR.
- **"That move is 86.9 % output format, and its content component is
  unresolved"** — the partition is exact and asserted.
- **"A zero-inheritance model of the same depth, corpus and budget reaches
  32.55 / 11.58 / 40.47 / 28.89 % of the intact calibrated residual"** — with the
  5×-LR caveat attached every time.
- "0 accepts across three repair modes × 4 axes × 5 conventions."

### NOT licensed
- ⛔ **Any `sd_run` / seed-variance statement.** One seed per arm; all three
  pre-`ce5c298`; no 7B `sd_run` exists or is reconstructible
  (`must_not_claim[23]`).
- ⛔ **Treating these three arms as a depth ladder.** All are `keep_front=14`.
- ⛔ **"Freezing the trunk repairs better."** §4.4 clause 2 does not survive.
- ⛔ **Any clean inheritance quantification from FS.** LR-confounded (5×).
- ⛔ **Any `PLATEAU(T)` statement about FF/FS.** No PPL on disk.
- ⛔ **Calling `freezefront/step23500.pt` a neighbour** of `step200000`. §3.
- ⛔ **Substituting `contains` for EM anywhere.** §4.3 uses it only to *label*
  items; the decision metric is and stays EM (`A04_GATE_DESIGN.md` §4.1).
- ⛔ **Any K1/K2/K3 clause.** Defined over the pre-registered **1B** arm set; no
  7B result can fire them. The gate remains unauthorised.
- ⛔ **Reading FF's popqa result as evidence that A04 should pivot to
  freeze-based repair.** That was the prereg's speculative clause and it is
  **withdrawn** (§4.4).
