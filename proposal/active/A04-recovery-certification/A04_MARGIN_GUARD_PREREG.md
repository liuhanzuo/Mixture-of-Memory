# A04 — Pre-registered guard on the non-inferiority margin `Δ`

**Status**: PRE-REGISTERED AMENDMENT to `A04_GATE_DESIGN.md` §2.
**Date**: 2026-08-10. **GPU spent producing this document: ZERO** (CPU re-analysis of
per-example records already on disk, plus read-only `ssh`/`scp -O` of three JSONL files).

> ### Why this is a pre-registration and not a post-hoc repair
>
> The gate's arms are **A1 prefix+fresh-tail, A2 contiguous-keep-only, A3 random-trunk,
> A4 from-scratch**, at the `j` Pilot Zero recommended (`j = 12`). **None of them exists.**
> Verified 2026-08-10 by `ls` on **both** disks: the only 1B recovery runs anywhere are
> `outputs/olmo2_probe2_1B_keep7fresh2{,_16card}`, `..._keep7f2_dolmino_cpt20k` and
> `..._keep7f2_dolmino_arm4_peaklr20k` — all `keep7`, all on zwfy6, none at `j=12` or
> `j=10`, and **no 1B recovery output at all on wzc1**. So every decision rule below is
> fixed **before any recovered-arm number that it will judge exists**.
>
> What *does* already exist is the **intact anchor**, and that is the point: `Δ` depends
> **only** on the intact arm, so the guard's classification of every cell is computable now
> and cannot be tuned later. Cells whose verdict genuinely depends on the unseen arm are
> labelled `NEEDS_RECHECK_AFTER_DATA` with the **numeric trigger fixed here**, so the
> recheck is arithmetic, not judgement.

**Reproduce everything in this document:**
```bash
cd proposal/active/A04-recovery-certification
python3 code/a04_intact_residual_ci.py \
  --zwfy6_intact <A03_1B_base/per_example_mmlu.jsonl> \
  --wzc1_intact  olmo2_mmlu_content_results/a01_1B_intact_base_full/per_example_mmlu.jsonl \
  --out_json evidence/a04_intact_residual_ci_1b_mmlu.json
python3 code/a04_measure_pdisc.py \
  --intact <A03_1B_base/per_example_mmlu.jsonl> \
  --arm keep7_step200k=<A03_1B_keep7_step200k/per_example_mmlu.jsonl> \
  --arm keep7_step500=<A03_1B_keep7_step500/per_example_mmlu.jsonl> \
  --out_json evidence/a04_pdisc_mmlu_1b.json
python3 code/a04_margin_guard_classify.py \
  --out_json evidence/a04_margin_guard_classification.json
```

---

## 1. The defect, stated precisely

`A04_GATE_DESIGN.md` §2 freezes:

> `NI(Δ)` accepts arm `a` at checkpoint `c` on axis `x` iff the **one-sided lower 95% bound**
> on `residual(a,c,x) − residual(intact,x)` is **> −Δ_x**, where `residual = reported − null_x`
> and `Δ_x = 0.10 · residual(intact, x)`.

The rule is a well-formed non-inferiority test **only if `residual(intact, x)` is a comfortably
positive number**. It is not always. When `residual(intact, x) < 0`, `Δ_x < 0`, so `−Δ_x > 0`
and the acceptance condition becomes "the lower bound must exceed a **positive** number" —
a **strict superiority test**.

**The direction of the error matters and must be stated correctly.** This cannot manufacture a
false ACCEPT: a superiority test is strictly harder than the non-inferiority test it replaced.
The damage is (i) a **silent false REJECT**, and (ii) a **semantic mislabel** — the writeup
would report "non-inferiority did not hold" when the hypothesis actually tested was a
different one. In a pre-registered gate, silently swapping the hypothesis is the more serious
of the two.

Pilot Zero found this for real and, correctly, **flagged rather than fixed** it
(`PILOT_ZERO_VERDICT.md` §4.1; `delta_degenerate_negative_margin` per cell in
`evidence/pilot_zero_rule_disagreement.json`). This document supplies the missing guard.

---

## 2. Exhaustive enumeration of the ways `Δ` fails

Six conditions, each checked against **data on disk**, not hypothesised.

| id | condition | why `Δ` stops being a margin | observed in A04's cells? |
|---|---|---|:--|
| **D1** | `residual(intact, x) < 0` | `Δ < 0` ⇒ NI becomes strict superiority. Also: the comparison target is itself sub-null. | **YES** — `credit`/MMLU-content |
| **D2** | `0 ≤ residual(intact, x) ≤ 1.0 pp` | `Δ → 0` ⇒ NI degenerates towards exact-equality; power → 0. | **NO** at 1B (min positive residual 9.6953 pp) |
| **D3** | CI on `residual(intact, x)` contains 0 | the **sign** of `Δ` is a sampling outcome ⇒ the hypothesis identity is random | **NO** at 1B (all 5 conventions, all 4 axes, p = 1e-4) |
| **D4** | `residual(intact, x) > 0` but the **null itself** is inadmissible | `Δ` is well-formed arithmetic on an invalid measurement | **YES** — `credit`/MMLU-content (co-fires with D1) |
| **D5** | `residual(intact, x)` differs between two admissible measurements of the same intact model | `Δ` not uniquely determined by the pre-registration | **measured, below threshold** (drift 0.0014 pp on `Δ`) |
| **D6** | `Δ_x` < the achievable item-level 95 % CI half-width | the margin is finer than the instrument ⇒ NI can essentially never accept | **YES** — NQ-open; **borderline** — MMLU-content |

### 2.1 D1 — `residual(intact) < 0`. CONFIRMED, one axis × one convention

The **intact** OLMo-2-1B on MMLU-content, `credit` (oracle tie-break) convention:

| quantity | value | source |
|---|---:|---|
| intact reported `content_norm` | **0.386839481555334** | `A03/evidence/a03_1b_floor_nulls_4axes.json` `cells[arm=intact, task=mmlu_content]`; independently recomputed from the per-example dump → `0.3868394816` |
| `credit` null | **0.4537102976783934** | `evidence/a04_intact_residual_ci_1b_mmlu.json`; matches `A01/evidence/gate3_content_null_conventions.json` to `<1e-12` |
| residual(intact) | **−6.687081612305939 pp** | recomputed |
| `Δ = 0.10 · residual` | **−0.6687081612305939 pp** | recomputed |
| CI95 on residual | **[−7.7199, −5.6402] pp**, `p = 1e-4` | measured here (new) |

So the negativity is **not** a marginal artefact: the CI is entirely below zero at the
bootstrap p-floor.

**No other axis or convention is negative.** Verified exhaustively (5 conventions × 4 axes =
20 combinations, `evidence/a04_margin_guard_classification.json:per_axis_convention`).
The three QA axes' nulls are convention-free — asserted in the classifier — so their residuals
are identical under all five conventions: TriviaQA **+40.4313 pp**, PopQA **+13.2053 pp**,
NQ-open **+9.6953 pp**.

### 2.2 D2 — `residual ≈ 0`. NOT observed at 1B, but the threshold is fixed now

Smallest positive intact residual anywhere in A04's cell family is **NQ-open 9.6953 pp**
(⇒ `Δ = 0.9695 pp`). Nothing is near zero, so **D2 does not fire at 1B**.

Threshold pre-registered anyway, because a shallower cut, a different scale, or a widened axis
set could land there: **`residual_floor_pp = 1.0 pp`**. Rationale, not taste: at 1.0 pp,
`Δ = 0.10 pp`, which is **below every item-level bootstrap half-width measured anywhere in
this project** (minimum observed **0.5429 pp**, PopQA; see §3). A margin below the
instrument's resolution is not a margin.

> **Where D2 nearly bites, recorded so it is not discovered later.** On the **7B** ladder the
> `credit` convention puts the intact `7B_base` residual at **+1.6878 pp** ⇒ `Δ = 0.1688 pp`,
> against a CI half-width of **1.0541 pp** — i.e. `Δ` is **6.2× smaller** than the
> half-width. That is above the 1.0 pp D2 floor but fails D6 decisively. Source:
> `A01/evidence/gate3_content_null_conventions.json:arms.7B_base.by_dtype.bf16.vs_null.content_norm.credit`
> (`delta_pp` 1.6877937615724254, `ci95_pp` [0.6338128471727674, 2.7419527132887027]).
> **A04 runs at 1B, so this is out of scope — but it is the reason D2 and D6 are both
> pre-registered rather than only D1.**

### 2.3 D3 — CI on `residual(intact)` straddles 0. NOT observed

Measured for MMLU-content under all five conventions (new, this document) and taken from A03's
own intact cells for the three QA axes:

| axis | convention | residual (pp) | CI95 (pp) | straddles 0? |
|---|---|---:|---|:--|
| TriviaQA EM | all | +40.4313 | [39.7069, 41.1391] | no |
| PopQA EM | all | +13.2053 | [12.6165, 13.7871] | no |
| NQ-open EM | all | +9.6953 | [8.6981, 10.7202] | no |
| MMLU-content | split | +10.2389 | [9.3149, 11.1802] | no |
| MMLU-content | first | +10.5754 | [9.5357, 11.5867] | no |
| MMLU-content | last | +10.4686 | [9.4431, 11.4798] | no |
| MMLU-content | **credit** | **−6.6871** | **[−7.7199, −5.6402]** | no (entirely **below** 0 → D1) |
| MMLU-content | wrong | +19.0714 | [18.1171, 20.0185] | no |

**0 of 20 combinations straddle zero.** D3 is enumerated because it is the condition under
which the guard would be *unresolvable* rather than merely restrictive — and it is worth
recording that A04 is not in that position.

### 2.4 D4 — the null itself is inadmissible. CONFIRMED, and it is the honest reason to retire the cell

D1 says the arithmetic breaks. D4 says something worse and more specific: under `credit` at 1B,
**every** arm is below the null.

| arm | `credit` residual (pp) | source |
|---|---:|---|
| intact 1B | **−6.6871** | measured here |
| keep7+fresh2 @200k | **−12.9326** | `evidence/pilot_zero_rule_disagreement.json:per_convention.credit.cells` |
| cpt20k @205k/210k/215k/220k | −13.0466 / −13.1676 / −13.0466 / −13.1320 | same |
| arm4_peaklr20k @205k/210k/215k | −13.2389 / −13.0822 / −13.2175 | same |

**8 of 8 damaged cells plus the intact arm are below this "floor".** A floor above the entire
arm population is not a floor. A01 found the same structure at 7B: under `credit`, **5 of 6**
arms flip to significantly BELOW null and only the intact base clears it, by 1.69 pp
(`A01/GATE3_CONVENTIONS_VERDICT.md` §4;
`gate3_content_null_conventions.json:convention_sensitivity_bf16_content_norm.credit` →
`n_arms_above: 1, n_arms_below: 5`).

**This is why option (c) — retire the cell — is the correct guard and not merely the
conservative one.** Any `Δ` recomputed from `|residual|` would be a well-formed number
measuring nothing.

#### 2.4.1 Other genuine D4 instances in the project (checked; all outside A04's axis set)

The user asked specifically whether A01's retired MMLU-**letter** interface is such a case.
**It is — and it is a D4, not a D1.** Verified numbers, `A03/evidence/a03_1b_floor_nulls_4axes.json`:

| arm | MMLU-letter reported | best-constant null (always-D) | residual | verdict |
|---|---:|---:|---:|---|
| intact 1B | 0.3807149978635522 | 0.2689075630252101 | **+11.1807 pp** | ABOVE floor |
| pruned+healed (keep7@200k) | 0.25117504628970233 | 0.2689075630252101 | **−1.7733 pp**, CI [−2.9768, −0.5840], `p = 0.0034` | **BELOW floor** |
| barely-healed (keep7@500) | 0.22945449366187154 | 0.2689075630252101 | **−3.9453 pp**, `p = 1e-4` | **BELOW floor** |

Note the structure: `residual(**intact**)` is **+11.18 pp**, comfortably positive, so **D1
does NOT fire and `Δ` looks perfectly healthy (`Δ = 1.1181 pp`)**. It is the *damaged* arms
that are sub-null, plus outright degeneracy — the barely-healed control emits letter `A` on
**14,042 / 14,042** items (`nulls.letter_degeneration.barely_healed.modal_pred_share: 1.0`),
and the pruned+healed arm is indistinguishable from its own modal-C constant
(`vs_own_modal_null_p: 0.2816`). **A `Δ`-only guard would have passed this cell.** That is
precisely why D4 is a separate condition from D1, and why the design's §4.1 ban on MMLU-letter
is load-bearing rather than stylistic. MMLU-letter is already BANNED as an A04 axis, so it
contributes **0 cells** — but it is the cleanest illustration that "`Δ` is well-formed" does
not imply "the cell is measurable".

Two more, recorded for the same reason (both **outside** A04's four axes, so 0 cells):

* **winogrande is structurally degenerate**: both options share a continuation, so
  `norm_lens` are identical (`{A:28,B:28}`), length normalisation is a no-op
  (`acc == acc_norm == 0.7459` exactly), raw-vs-norm agreement is **1.0000**, the
  longest-option null is exactly **0.5000** with a **100 % tie rate**
  (`status/scout_21/lane2_a01_gate2.md` §3, §4a). It can only ever be a control.
* **BoolQ raw interface**: against the always-B null **0.6217** (not 0.50), keep12 **0.6101**,
  keep10 **0.6086**, keep8 **0.5948** are all **below** floor and keep14 (+0.0165) is
  **n.s.** (McNemar `p = 0.20`) — **4 of 6 arms not distinguishable from a constant**
  (`lane2_a01_gate2.md` §4b). Also exhibits an interface-dependent verdict flip.
  ⚠️ **UNVERIFIED here**: these BoolQ/winogrande/OpenBookQA numbers are quoted from
  `status/scout_21/lane2_a01_gate2.md`; I did **not** re-open the underlying shards in this
  pass. They are recorded as *why the widened-axis list needs the same guard*, not as A04
  evidence.

### 2.5 D5 — is the intact anchor unique? MEASURED, and it is not exactly

`Δ` is `0.10 ·` a number measured on one artefact, so the artefact must be pinned. **Two
independent admissible measurements of the same intact 1B model exist**, and I compared them
item-by-item:

| | `A03_1B_base` (zwfy6) | `a01_1B_intact_base_full` (wzc1) |
|---|---:|---:|
| `content_norm` | 0.3868394816 | **0.3869819114** |
| meta | `mode=base, num_hidden_layers=16, base_model=../models/OLMo-2-0425-1B, add_bos=false, content_desc=full` | **identical** |
| item flips | — | **48 / 14,042** |

Consequences, all measured:
* the five nulls are **bit-identical** between the two dumps (asserted `<1e-12`) — the null is
  a tokenizer+dataset property, as A01 claims;
* residual drift = **+0.014243 pp** ⇒ **`Δ` drift = +0.001424 pp**, i.e. **0.14 % of
  `Δ_MMLU`**. Far too small to move any decision.

So **D5 does not bite numerically**, but it does bite **procedurally**: two artefacts exist and
they are not identical, so "`Δ = 0.10 · residual(intact)`" does not by itself name a number.
The guard therefore **pins the anchor** (§4, rule G0).

> **Honest note on the 48 flips.** The repo's standing rule is that same-arch/same-harness
> re-runs are **byte-identical (0 flips)**, so a nonzero difference must be attributed to a
> named cause, not excused as noise. The named cause here is **a harness code change between
> the two evals**: `A03_1B_base` was written 2026-08-08 20:49 and
> `a01_1B_intact_base_full` 2026-08-09 08:25, and `scripts/eval_olmo2_mmlu_content.py` was
> modified in between by commit **`7ac9653`** (2026-08-08 22:29:43 +0800, "damaged non-OLMo
> extension via no-heal front-N truncation"), which also touched the shared loader import.
> ⚠️ **UNVERIFIED**: I have **not** proven that `7ac9653` is *causally* responsible for the 48
> flips — the two dumps also sit on different disks/nodes. The claim made here is only the
> weaker, verified one: **a harness commit falls strictly between the two evals**, so this is
> a code-version difference and not a same-code runtime-jitter claim. Establishing causality
> would need a same-code control, which is exactly what the standing rule demands before any
> "noise floor" is asserted. **No such floor is asserted here.**

### 2.6 D6 — `Δ` finer than the instrument. CONFIRMED for NQ-open, BORDERLINE for MMLU-content

The design already used this reasoning once, to demote NQ-open (§5.2). The guard generalises it
and — importantly — makes it **checkable before the data**, by pre-registering a **critical
discordance rate** instead of a half-width that nobody can know yet.

For a paired accuracy difference, `SE = sqrt(p_disc/n)` and `hw₉₅ = 1.96 · SE`, so

```
D6 fires  ⟺  Δ_x < hw₉₅  ⟺  p_disc > p*_crit ,   p*_crit = n · (Δ_x / (100 · 1.95996))²
```

`p*_crit` depends only on `Δ_x` and `n`, both known now. Fixed values:

| axis | n | `Δ` (pp), pre-reg `split` | **`p*_crit`** | `p_disc` observed in this project | D6 |
|---|---:|---:|---:|---|:--|
| TriviaQA EM | 17,944 | 4.0431 | **7.6359** | 0.2273 – 0.2447 | clear (huge headroom) |
| PopQA EM | 14,267 | 1.3205 | **0.6476** | 0.1095 – 0.1306 | clear |
| MMLU-content | 14,042 | 1.0239 | **0.3832** | **0.1601 – 0.3011** | **borderline** |
| NQ-open EM | 3,610 | 0.9695 | **0.0883** | **0.0861 – 0.0984** | **FIRES** |

The observed ranges are recovered from Pilot Zero's own per-cell bootstrap gaps
(`diff_mean_pp − diff_lower95_one_sided_pp = 1.6449 · SE`), which the classifier
cross-checks against a **direct** shard-level measurement: MMLU-content `p_disc` for
keep7@200k = **0.168708**, inside the recovered range (asserted in code).

**Two things here are worth not glossing over.**

1. **NQ-open's `p*_crit` (0.0883) sits *inside* its observed range (0.0861–0.0984).** So
   NQ-open's demotion is not a rounding-margin call: the axis has already been observed on
   both sides of its own critical value. The design's §5.2 demotion is **confirmed
   independently here**, from a different statistic.
2. **MMLU-content's borderline status only appears once the barely-healed arm is included.**
   Pilot Zero scored only arms at ≥200k steps, whose `p_disc` is 0.1601–0.1727 — comfortably
   under `p*_crit = 0.3832`. But the gate's frozen grid **starts at 2,500 steps**, and the
   closest on-disk analogue (`A03_1B_keep7_step500`, the barely-healed control) has
   `p_disc = 0.301097` — **79 % of the way to `p*_crit`**. Its half-width is **0.8831 pp**
   against `Δ = 1.0239 pp`, so `Δ/hw = 1.16`: still valid, but with only 16 % headroom.
   Restricting attention to the well-healed arms would have produced a falsely comfortable
   CERTIFIABLE verdict. **The early checkpoints are the D6 risk, not the late ones.**

---

## 3. Candidate guards, and why (c) wins

Measured inputs for every option:
`residual(intact, MMLU, credit) = −6.6871 pp`, CI `[−7.7199, −5.6402]`;
`|Δ| = 0.6687 pp`; MMLU-content half-width **0.6619 – 0.6873 pp** (well-healed) / **0.8831 pp**
(barely healed); all 8 damaged `credit` cells at **−12.93 to −13.24 pp**.

### (a) Absolute margin floor — `Δ = max(0.10 · residual, ε)`, ε ≈ 1 pp

* **Restores** a positive margin, so NI stays a non-inferiority test. ✔
* **Fatal**: it does nothing about **what is being compared**. Under `credit` the target is
  **6.69 pp below its own null**; "recovered is within 1 pp of a sub-null target" is a
  sentence with no scientific content. It converts a visible degeneracy into an invisible one.
* Also **arbitrary in a load-bearing way**: `ε = 1 pp` is comparable to the *entire* legitimate
  `Δ_MMLU = 1.0239 pp`, so on the axis where it fires it would be the dominant term, i.e. the
  guard would silently set the margin for the axis it was meant to rescue.
* **Rejected.**

### (b) `Δ = 0.10 · |residual(intact)|`

* **Cheapest** to state, and gives "looks normal" margins everywhere. ✔
* **Fatal, and worse than (a)**: it is *specifically* the failure mode the user flagged. It
  yields `Δ = +0.6687 pp` under `credit` — a plausible-looking number — while the comparison
  target is sub-null and **all 8 damaged cells sit 6.2–6.6 pp below the target**. The report
  would read like a legitimate non-inferiority evaluation of an illegitimate quantity.
* It also **breaks monotonicity**: as the intact arm gets *worse* past its null, `|residual|`
  grows, so `Δ` grows, so the test gets *easier*. A margin that loosens as the reference
  degrades is not a margin.
* **Rejected.**

### (c) Retire the cell — `NOT_CERTIFIABLE` ✅ RECOMMENDED

* **Honest**: an arm below its own input-blind floor has no measured capability on that axis;
  "recover to be no worse than it" is vacuous. This is A01's own logic (a below-floor arm
  carries no signal), applied to the *reference* arm instead of the test arm.
* **Preserves the pre-registration**: `Δ`'s formula is untouched. The guard changes only the
  cell's **admissibility**, which is a scope statement, not a threshold.
* **Fail-loud**: a retired cell is visibly absent, so it cannot be mistaken for a passed or
  failed test. (a) and (b) both fail *silently*.
* **Cost, stated plainly**: it removes cells. Under `credit`, 24 of 96. §5 shows this does
  **not** empty A04 under the pre-registered convention.
* **Recommended.**

### (d) Change the anchor (damaged-arm residual, or intact absolute score)

* **Damaged-arm anchor** — inadmissible: `Δ` would then depend on the arm being judged, so a
  worse arm buys itself a wider margin. Directly circular.
* **Intact absolute score** (`Δ = 0.10 · reported(intact)`, i.e. `0.10 × 0.3868 = 3.87 pp`) —
  well-defined and never negative, but it **re-imports the exact error A01 exists to kill**:
  it credits the model for the part of its score any input-blind constant would achieve. On
  MMLU-content the null is **0.2845**, i.e. **73.5 %** of the intact score, so a
  null-blind margin would be **3.78×** the null-calibrated one — a margin dominated by
  label/length artefacts. It would also make A04 self-contradictory, since its whole claim is
  null-calibrated non-inferiority.
* **Rejected**, but recorded, because it is the one a reviewer will propose.

---

## 4. The guard — verbatim, binding, pre-registered

> ### MARGIN GUARD (verbatim, binding; amendment to `A04_GATE_DESIGN.md` §2, 2026-08-10)
>
> **G0 — Anchor pinning.** Before any gate GPU is spent, **one** intact-arm artefact is named
> by path + SHA256 for each axis, and `residual(intact, x)` is computed from that artefact
> only. The 2026-08-10 pinning is
> `zwfy6:olmo2_mmlu_content_results/A03_1B_base/per_example_mmlu.jsonl`
> (md5 `d1a7b1cefc0031afa84e7b9334a08bc5`) and the matching `A03_1B_base*` closed-book dumps —
> i.e. **A03's anchor, the one Pilot Zero used**. The second admissible measurement
> (`wzc1:olmo2_mmlu_content_results/a01_1B_intact_base_full/`) is recorded as a D5 sensitivity
> check (drift on `Δ`: **0.0014 pp**) and **is not used for `Δ`**.
>
> **G1 — Admissibility precedes testing.** For every cell `(arm, checkpoint, axis,
> convention)`, the six conditions **D1–D6** are evaluated **before** `NI(Δ)` is computed. A
> cell failing any of **D1, D2, D3, D4, D6** is **NOT_CERTIFIABLE**: `NI(Δ)` is **not run**,
> no accept/reject is reported for it, and it is **excluded from the BH family** (which
> therefore shrinks — declare the reduced family size).
>
> **G2 — No margin substitution, ever.** When a cell is NOT_CERTIFIABLE, `Δ` is **not**
> replaced by `max(·, ε)`, by `0.10·|residual|`, by a different anchor, or by anything else.
> Options (a), (b) and (d) of §3 are **prohibited for the remainder of A04.** The only
> admissible responses are: retire the cell, or (with explicit re-registration) change the
> *interface* so the null becomes admissible.
>
> **G3 — Frozen numeric triggers.**
> * **D1**: `residual(intact, x) < 0`.
> * **D2**: `0 ≤ residual(intact, x) ≤ 1.0 pp`.
> * **D3**: the two-sided 95 % paired-bootstrap CI on `residual(intact, x)` contains 0
>   (10,000 resamples, A03's `paired_bootstrap`, seed as in that module).
> * **D4**: the intact arm **and every** damaged arm evaluated on that axis are below the
>   null; **or** the interface is structurally degenerate (a single constant emitted on
>   >99 % of items; or a ≥99 % tie rate in the null's winner set).
> * **D5**: `|Δ` drift across admissible intact artefacts`| ≥ 0.10 · Δ`. (Recorded, not
>   fatal: it mandates G0 rather than retirement.)
> * **D6**: measured `p_disc > p*_crit = n · (Δ_x / (100 · 1.959964))²`. Pre-computed
>   `p*_crit`: TriviaQA **7.6359**, PopQA **0.6476**, MMLU-content **0.3832**, NQ-open
>   **0.0883**.
>
> **G4 — The only post-data step, and it is arithmetic.** A cell marked
> `NEEDS_RECHECK_AFTER_DATA` here is marked so **only** because D6's input `p_disc` is a
> property of the unseen arm. The recheck is the single comparison `p_disc > p*_crit` with
> `p*_crit` **already fixed above**. No threshold may be adjusted at that point.
>
> **G5 — Convention is part of `Δ`'s definition.** `Δ_MMLU` is reported **with** its tie
> convention. The pre-registered decision convention remains **`split`**
> (`A01`, unchanged). The other four are reported as sensitivity only and **cannot** promote a
> NOT_CERTIFIABLE cell to CERTIFIABLE.
>
> **G6 — Reporting duty.** Every NOT_CERTIFIABLE cell is listed in the writeup with the
> condition that retired it and the number that triggered it. Retired cells are **never**
> reported as "NI rejected".

---

## 5. Classification of every planned cell — fixed before the recovered arms exist

Cell family = **4 arms × 6 checkpoints × 4 axes = 96** (arms `A1..A4` of §3.2; grid
`{2500, 5000, 10000, 20000, 40000, 80000}` of §2). Decision cells exclude the
design-demoted NQ-open: **3 axes × 24 = 72**.

**The guard's verdict is a property of `(axis, convention)` only**, because
`residual(intact, x)` does not depend on the arm or the checkpoint — the intact anchor is the
same model in every cell. **That is what makes this pre-registrable today.**

### 5.1 Per `(axis, convention)`

Full machine-readable table: `evidence/a04_margin_guard_classification.json:per_axis_convention`.

| convention | axis | residual(intact) pp | CI95 pp | `Δ` pp | `p*_crit` | `p_disc` obs. max | verdict | trigger |
|---|---|---:|---|---:|---:|---:|---|---|
| **split** ★ | TriviaQA | +40.4313 | [39.7069, 41.1391] | 4.0431 | 7.6359 | 0.2447 | **CERTIFIABLE** | — |
| **split** ★ | PopQA | +13.2053 | [12.6165, 13.7871] | 1.3205 | 0.6476 | 0.1306 | **CERTIFIABLE** | — |
| **split** ★ | MMLU-content | +10.2389 | [9.3149, 11.1802] | 1.0239 | 0.3832 | **0.3011** | **NEEDS_RECHECK_AFTER_DATA** | D6 within 2× |
| **split** ★ | NQ-open | +9.6953 | [8.6981, 10.7202] | 0.9695 | **0.0883** | **0.0984** | **NOT_CERTIFIABLE** | **D6** |
| first | TriviaQA / PopQA | as above | | | | | CERTIFIABLE | — |
| first | MMLU-content | +10.5754 | [9.5357, 11.5867] | 1.0575 | 0.4088 | 0.3011 | NEEDS_RECHECK | D6 within 2× |
| first | NQ-open | +9.6953 | | 0.9695 | 0.0883 | 0.0984 | NOT_CERTIFIABLE | D6 |
| last | TriviaQA / PopQA | as above | | | | | CERTIFIABLE | — |
| last | MMLU-content | +10.4686 | [9.4431, 11.4798] | 1.0469 | 0.4006 | 0.3011 | NEEDS_RECHECK | D6 within 2× |
| last | NQ-open | +9.6953 | | 0.9695 | 0.0883 | 0.0984 | NOT_CERTIFIABLE | D6 |
| **credit** | TriviaQA / PopQA | as above | | | | | CERTIFIABLE | — |
| **credit** | **MMLU-content** | **−6.6871** | **[−7.7199, −5.6402]** | **−0.6687** | n/a | 0.3011 | **NOT_CERTIFIABLE** | **D1 + D4** |
| **credit** | NQ-open | +9.6953 | | 0.9695 | 0.0883 | 0.0984 | NOT_CERTIFIABLE | D6 |
| wrong | TriviaQA / PopQA | as above | | | | | CERTIFIABLE | — |
| wrong | MMLU-content | +19.0714 | [18.1171, 20.0185] | 1.9071 | 1.3295 | 0.3011 | **CERTIFIABLE** | — |
| wrong | NQ-open | +9.6953 | | 0.9695 | 0.0883 | 0.0984 | NOT_CERTIFIABLE | D6 |

★ = the pre-registered decision convention.

### 5.2 Cell counts

| convention | all 96 cells: CERT / NOT_CERT / RECHECK | 72 decision cells: CERT / NOT_CERT / RECHECK |
|---|---|---|
| **split** ★ | **48 / 24 / 24** | **48 / 0 / 24** |
| first | 48 / 24 / 24 | 48 / 0 / 24 |
| last | 48 / 24 / 24 | 48 / 0 / 24 |
| credit | 48 / **48** / 0 | 48 / **24** / 0 |
| wrong | 72 / 24 / 0 | 72 / 0 / 0 |

**Headline, pre-registered convention `split`:** **24 of 96 cells (25 %) are
NOT_CERTIFIABLE** — all 24 are NQ-open, and **all 24 were already excluded from decisions by
the design's own §5.2 demotion**. Of the **72 decision cells, 0 are NOT_CERTIFIABLE**; 24
(MMLU-content) are `NEEDS_RECHECK_AFTER_DATA` on a fixed arithmetic trigger.

---

## 6. Effect on the kill clauses K1 / K2 / K3 — including the case where the guard disables one

This is the part a guard can silently get wrong: all three clauses are phrased as
"**N of the 4** axes". If the guard retires axes, a literal count can become
**unsatisfiable**, which would **disable a kill clause** and destroy falsifiability. Computed,
not assumed (`evidence/a04_margin_guard_classification.json:kill_clause_impact`).

| convention | surviving decision axes | K1 clause (a) "≥3 of 4" satisfiable? | K1 "≥24 cells" precondition | K2 "≥2 of 4" satisfiable? | K3 axes with intact residual <5pp |
|---|---|:--|---|:--|---|
| **split** ★ | **3 / 3** (TriviaQA, PopQA, MMLU-content) | **YES** | **YES** (72 cells) | **YES** | **0** → K3 does not fire |
| first | 3 / 3 | YES | YES (72) | YES | 0 |
| last | 3 / 3 | YES | YES (72) | YES | 0 |
| **credit** | **2 / 3** | **NO — K1 clause (a) becomes UNSATISFIABLE** | YES (48) | YES | 1 (MMLU-content, and it is **negative**) |
| wrong | 3 / 3 | YES | YES (72) | YES | 0 |

**Under the pre-registered `split` convention, the guard costs A04 nothing in falsifiability.**
All three clauses remain satisfiable, all 3 decision axes survive, and K1's own
"≥ 24 evaluated cells" precondition is met with 72 — which, incidentally, is the precondition
Pilot Zero **could not** meet (only 3 cells had PLATEAU defined; `PILOT_ZERO_VERDICT.md`
§6.2.5).

**Two failure modes must nevertheless be pre-registered, because they are real under other
conventions and would be real at a different scale:**

* **K1 disabling.** Retiring one of three decision axes makes "NI accepts on ≥3 of the 4 axes"
  impossible to satisfy, so K1's clause (a) would be **permanently false** and K1 could never
  fire — A04 would then be *unkillable by K1*, which is strictly worse than being killed.
  **Rule (pre-registered): K1's threshold rescales to `ceil(0.75 · n_surviving)` and K2's to
  `ceil(0.50 · n_surviving)`, preserving the original 3/4 and 2/4 proportions.** Under `split`
  this is a no-op (3 of 3, 2 of 3). Under `credit` it would give 2 of 2 and 1 of 2. **The
  rescaling rule is fixed here, before the data, precisely so it cannot be chosen later to
  suit an outcome.**
* **K3 sign coupling — a genuine trap.** K3 fires when an axis's intact residual is
  "below 5pp", and a **negative** residual is trivially below 5pp. So a convention that pushes
  an axis under its null pushes K3 **towards firing for the wrong reason**: K3 is meant to
  detect *an unmeasurable scale*, not *an inadmissible null*.
  **Rule (pre-registered): axes retired by D1 or D4 are EXCLUDED from K3's count, not counted
  as at-floor.** Under `credit`, MMLU-content would otherwise contribute a spurious
  at-floor vote. With the exclusion, `credit` leaves 0 of the 2 surviving axes below 5pp, so
  **K3 does not fire under any convention** — consistent with Pilot Zero §2 but now for a
  stated reason rather than by luck.

### 6.1 Does the guard leave A04 with nothing to measure? — NO, and here is the honest accounting

The user asked to say so plainly if the guard empties A04. **It does not**, under the
pre-registered convention:

* **72 of 72 decision cells remain admissible** (48 CERTIFIABLE + 24 pending one arithmetic
  check). Zero decision cells are retired.
* The **primary axis (TriviaQA EM) is the safest cell in the family**: `Δ = 4.0431 pp` against
  a `p*_crit` of **7.6359** versus an observed `p_disc` of ≤0.2447 — a **31×** margin. It
  cannot be retired by any of D1–D6 at 1B.
* The 24 retired cells are **entirely NQ-open**, an axis the design had **already** demoted to
  descriptive on independent grounds. **The guard therefore removes no decision capacity that
  A04 possessed before it.**

**What the guard does cost, stated without softening:**

1. **A04 loses the ability to report a `credit`-convention MMLU non-inferiority result.** Since
   `residual(arm) − residual(intact) = reported(arm) − reported(intact)` (the null cancels
   exactly — `evidence/pilot_zero_rule_disagreement.json:null_invariance_of_the_difference`,
   and `ni_rule`'s `null_vec_*` arguments are provably unused: 0 references in the function
   body), the *measured difference* is convention-invariant and **A04's headline is unaffected**.
   Only the `credit` **margin** is lost. The convention-robustness claim in
   `PILOT_ZERO_VERDICT.md` §4 must now be stated as **"the verdict is unchanged under the 4
   admissible conventions; the 5th (`credit`) is inadmissible on this axis by the margin
   guard"** — rather than "survives all five". **That is a real narrowing of a claim already
   written down, and it is this document's doing.**
2. **MMLU-content's admissibility now depends on the unseen arm.** With only 16 % headroom at
   the barely-healed `p_disc`, an early-checkpoint MMLU cell may well retire. If it does, the
   pre-registered rescaling leaves K1 at `ceil(0.75 × 2) = 2 of 2` — still satisfiable, so
   still falsifiable, but on a thinner base.
3. **A 5pp-residual axis would be uncertifiable at a 10 % margin.** `Δ = 0.5 pp` is below
   every measured half-width (0.54–1.02 pp). So **K3's own 5pp threshold and the guard's D6
   are in tension**: an axis can pass K3 (residual ≥5pp, "measurable") yet fail D6
   (`Δ` < half-width, "not certifiable at 10 %"). At 1B no axis is in that window — the
   smallest surviving residual is 10.2389 pp — but **at a shallower cut or a smaller scale it
   would be, and the guard would then retire axes K3 had just declared measurable.** This is
   an unresolved design tension, recorded rather than papered over. **It is not resolved by
   this document and must not be resolved by adjusting `Δ` after data.**

**Bottom line: the guard is not a kill signal for A04, and A04's survival does not depend on
the guard being lenient.** K2 — untestable with anything on disk (`SEED_SEMANTICS_DEFECT.md`)
— remains the likeliest killer, exactly as Pilot Zero concluded. The guard changes nothing
about that.

---

## 7. Provenance — every number traces to a file I opened

| number | file |
|---|---|
| intact 1B `content_norm` 0.386839481555334; MMLU-letter cells; QA intact residuals + CIs | `proposal/active/A03-parametric-vs-external-memory/evidence/a03_1b_floor_nulls_4axes.json` |
| five MMLU nulls (`split` 0.28445022076627263 … `credit` 0.4537102976783934); 7B `credit` flips 5/6 | `proposal/active/A01-null-calibration-methodology/evidence/gate3_content_null_conventions.json` |
| per-cell `credit` residuals (−12.93 … −13.24 pp); `delta_degenerate_negative_margin`; bootstrap gaps → `p_disc` | `proposal/active/A04-recovery-certification/evidence/pilot_zero_rule_disagreement.json` |
| intact residual CIs under all 5 conventions; D5 anchor drift (48 flips, 0.0014 pp on `Δ`) | `proposal/active/A04-recovery-certification/evidence/a04_intact_residual_ci_1b_mmlu.json` **(new)** |
| measured `p_disc` 0.168708 (keep7@200k) / 0.301097 (keep7@500) | `proposal/active/A04-recovery-certification/evidence/a04_pdisc_mmlu_1b.json` **(new)** |
| the classification + kill-clause impact tables | `proposal/active/A04-recovery-certification/evidence/a04_margin_guard_classification.json` **(new)** |
| BoolQ 0.6217 always-B; OpenBookQA 0.3635; winogrande 100 % tie rate | `status/scout_21/lane2_a01_gate2.md` — **quoted, NOT re-verified here** |
| per-example dumps (read-only, md5-verified after `scp -O`) | `zwfy6:olmo2_mmlu_content_results/A03_1B_{base,keep7_step200k,keep7_step500}/per_example_mmlu.jsonl` — md5 `d1a7b1cefc0031afa84e7b9334a08bc5` / `098e342e08e3e9407e8c56f745688cc2` / `1b46a4b3032d66490c3bce5ef0bf6af0` |
| second intact artefact | `wzc1:olmo2_mmlu_content_results/a01_1B_intact_base_full/per_example_mmlu.jsonl` |
| no 1B arm at `j=12`/`j=10` on either disk | `ls outputs/` on wzc1 **and** on zwfy6 (via `.82`), 2026-08-10 |

**Method compliance**: scorers/nulls (`longest_option_vector`, `paired_bootstrap`) are
**imported** from `A03/code/analyze_1b_knowledge_floor.py`, never reimplemented; item counts
hard-asserted at 14,042 with `nan`-row and duplicate-`item_id` checks; item-ID and gold-letter
alignment asserted across every pair of dumps; the five nulls asserted bit-identical between
the two intact artefacts; the recovered `p_disc` range asserted to contain the directly
measured 0.168708; the pre-registered `T`/`ρ`/`Δ`-fraction asserted equal to the values in
`evidence/pilot_zero_rule_disagreement.json:preregistration`.

**GPU: ZERO.** No training, no eval, no `torchrun`, no model load. Remote access was
read-only (`ls`, `md5sum`, `scp -O` of three JSONL files).

### 7.1 Discrepancy found between prose and evidence JSON — reported, JSON wins

`A04_GATE_DESIGN.md` §4 and `STATUS.json:nulls_per_metric.NQ_open_EM` both give the NQ-open
null as **`0.0055`**. The evidence JSONs give **`0.00554016620498615`**
(`a03_1b_floor_nulls_4axes.json:cells[intact,nq_open,em].null`, and
`pilot_zero_rule_disagreement.json:nulls.nq_open.acc`). The design's own §4 also states that
`0.0053` "must not be quoted" (via `A03/GATE_FOURAXES_VERDICT.md`).

**Assessment: the prose value is a rounding of the JSON value, not a conflict** — `0.0055`
vs `0.00554016620498615`, a **0.00402 pp** difference on the null, which moves
`Δ_NQ-open` from `0.9695290858725762 pp` to `0.9699307479224377 pp`, i.e. by
**0.000402 pp**, and changes no verdict (NQ-open is retired by D6 under either value:
`p*_crit` moves from 0.08834 to 0.08841, still inside the observed 0.0861–0.0984 range).
**Correction for the record: the canonical NQ-open null is `0.00554016620498615`; `0.0055`
is a display rounding and must not be used in arithmetic.**
All numbers in this document use the JSON values throughout. No other prose/JSON
inconsistency was found in the quantities this guard depends on (the four intact residuals,
the five MMLU nulls, and the eight `credit` cells were each re-derived and matched to
`<1e-6`).

---

## 8. UNVERIFIED / not established by this document

1. **Whether MMLU-content survives D6 at the gate's early checkpoints.** `p*_crit = 0.3832`
   vs a barely-healed `p_disc` of `0.301097` leaves **16 % headroom on `Δ/hw`**. The arms that
   would settle it do not exist. This is the `NEEDS_RECHECK_AFTER_DATA` verdict, and the
   trigger is fixed.
2. **`p_disc` for TriviaQA / PopQA / NQ-open at a barely-healed checkpoint.** Only MMLU-content
   has a step-500 arm scored here; the QA ranges come from arms at ≥200k steps. Since `p_disc`
   rises as the arm degrades, the QA D6 verdicts are computed from an **optimistic** range.
   TriviaQA's 31× headroom is safe by any margin; **PopQA (`p*_crit` 0.6476 vs observed max
   0.1306, ≈5× headroom) is very likely safe but not proven at step 2,500.**
3. **The causal attribution of the 48-item D5 drift** to harness commit `7ac9653`. Verified
   only that the commit falls strictly between the two evals; a same-code control was not run.
   No "noise floor" is claimed.
4. **BoolQ 0.6217 / OpenBookQA 0.3635 / winogrande 100 % tie rate** — quoted from
   `status/scout_21/lane2_a01_gate2.md`; underlying shards not re-opened in this pass. They
   inform only the widened-axis discussion, not any A04 verdict.
5. **Whether `residual(intact)` stays positive at `j = 12`.** The intact anchor is the same
   model regardless of `j`, so `Δ` itself is unaffected — but K3's clearance is arm-dependent
   and remains **UNVERIFIED for `j = 12`**, exactly as `PILOT_ZERO_VERDICT.md` §2 states.
6. **7B behaviour.** The `credit`/`7B_base` near-D2 case (`Δ = 0.1688 pp` vs half-width
   `1.0541 pp`) is recorded from A01's JSON but 7B is **out of A04's scope and unfunded**.
7. **The K3-vs-D6 tension** (§6.1 item 3) is **identified, not resolved.** No axis at 1B falls
   in the window, so it does not bite here.
8. **SHA256 for the pinned anchors.** G0 asks for SHA256; this pass verified **md5** after
   `scp -O` (matched at both ends). SHA256 must be recorded at gate launch.
