---
doc: A01 response ledger for the TCODEX skeptical audit
audit: proposal/archive/A03-parametric-vs-external-memory/evidence/TCODEX_AUDIT_20260810.md
audit_date: 2026-08-10
audit_verdict_on_A01: "Major revision"
this_response: 2026-08-10
compute: CPU only, ZERO GPU. No checkpoint read, no node touched except one
  read-only ssh to .82 to list zwfy6 summary.json files.
recompute_script: code/a01_audit_response_recompute.py
recompute_output: evidence/a01_audit_response_recompute.json
---

# A01 — response to the TCODEX audit, item by item

The audit (§2.1 and §7) returned **Major revision** on A01 and A01's own directory
contained **zero** reference to it. This file is that reference. Every number below
was recomputed from on-disk per-item records in this session by
`code/a01_audit_response_recompute.py`; where the audit and a prior A01 file
disagree, the recompute decides and the A01 file is corrected, not the audit.

Disposition vocabulary: **ACCEPT** = the audit is right and A01 changes;
**NARROW** = the finding survives but with a smaller scope than A01 claimed;
**REJECT** = A01 keeps the claim and says why.

---

## 0. Summary of dispositions

| # | Audit item (§2.1) | Disposition | Where it lands |
|---|---|---|---|
| R-1 | "letter is a family-general step function / sharp phase transition" → retract | **ACCEPT (retract "family-general" and "step function")** | `GATE1_DEPTHCURVE_VERDICT.md` banner + §1/§3 rewritten; `STATUS.json:gate1_depth_curve.verdict`; `PROPOSAL.md` claim 2 |
| R-2 | `GATE1_DEPTHCURVE_VERDICT.md` says Llama-2 content is "strictly monotone" but it decreases | **ACCEPT** | word "strictly" removed; two decreases printed with their p-values |
| R-3 | "five equally defensible tie conventions flip 5/6 verdicts" → retract | **ACCEPT (demote to "three executable conventions + two bounds")** | `GATE3_CONVENTIONS_VERDICT.md` banner + §4/§5; `STATUS.json:gate3b...`; `PROPOSAL.md` §protocol; `novelty_check.strongest_remaining_novel_claim` |
| R-4 | "damage generally turns letter into a constant predictor" → retract when modal share is 45–47% | **ACCEPT** | modal share now printed per depth in the recompute JSON; see §4 below |
| R-5 | cross-C1–C4 residual span "7–10×" is not one estimand | **ACCEPT** — already partly conceded by gate-4; see §5 | `STATUS.json:gate4...` note |
| R-6 | `active_all_gates_passed` overclaims while an unincorporated Major-revision verdict exists | **ACCEPT** | `STATUS.json:status` changed |
| R-7 | `7B_base_dtype_summary.json: letter_acc_diff_boot_p = 1.042` is an illegal p-value | **ACCEPT as a real defect**; ~~not fixed in this pass~~ → **FIXED & CLOSED 2026-08-11** | §7 + update note; estimator patched (`two_sided_boot_p`) and all six summaries re-emitted, 0 of 24 verdicts changed — `paperG/evidence/R7_BOOTSTRAP_P_FIX.md` |
| R-8 | narrow "general structural damage response" to extreme front-truncation | **ACCEPT** (was already the wording in `GATE1_DAMAGED_VERDICT.md`; the overreach is in `STATUS.json`'s headline) | §8 |
| R-9 | the tokenizer-dependence leg | **not attacked by the audit; KEPT and strengthened** | §6 |

---

## 1. R-1 — the "family-general step function" claim is RETRACTED

### What A01 claimed

`STATUS.json:gate1_depth_curve.verdict`:
> "THE TWO INTERFACES HAVE DIFFERENT FUNCTIONAL FORMS -- letter is a **STEP function**
> of depth, content_norm is SMOOTH and MONOTONE"

`PROPOSAL.md` claim 2: "这个塌陷是深度上的 **SHARP PHASE TRANSITION**，不是渐进衰减。"

`STATUS.json:gate1_depth_curve.llama2_anomaly` said the disconfirming gap-fill was
**"in flight on .21"**. It is not in flight. **It finished**, and the five arms
`gate1_dmg_llama2_7b_depth_gap2_k{8,12,18,22,26}` have been sitting on wzc1
unreported. Leaving a claim standing on "the disconfirming run hasn't landed yet"
while the run had in fact landed is the defect, independent of the numbers.

### What the numbers say (recomputed this session)

Llama-2-7B, front-truncation, MMLU n=14042/arm, 0 nan, 8/8 shards asserted,
**15 unique keep-depths** (k=4,6,8,10,12,14,16,18,20,22,24,26,28,30,31). Seven of
the 15 depths have two directories (original `depth_k*` plus the `gap`/`gap2`
re-run); the pairs were asserted identical to 12 dp before one was kept, so they
are re-runs of the same arm, not extra arms.

Floor = letter best-constant null always-D = `0.268908`.

| k | letter_acc | vs floor (pp) | binom p | verdict | modal letter |
|---:|---:|---:|---|---|---|
| 4 | 0.230523 | −3.838 | 2.0e−25 | BELOW | A @ 93.3% |
| 6 | 0.253383 | −1.552 | 3.1e−05 | BELOW | D @ 43.5% |
| 8 | 0.241490 | −2.742 | 1.4e−13 | BELOW | A @ 44.9% |
| 10 | 0.249323 | −1.958 | 1.4e−07 | BELOW | B @ 57.2% |
| 12 | 0.229454 | −3.945 | 8.7e−27 | BELOW | A @ **100.0%** |
| 14 | 0.230736 | −3.817 | 3.7e−25 | BELOW | A @ 96.4% |
| **16** | 0.328942 | **+6.003** | 1.3e−55 | **above** | A @ 57.8% |
| 18 | 0.335850 | +6.694 | 2.0e−68 | above | D @ 80.8% |
| 20 | 0.305441 | +3.653 | 5.6e−22 | above | A @ 84.9% |
| **22** | 0.230238 | **−3.867** | 9.1e−26 | **BELOW** | A @ 99.9% |
| **24** | 0.272255 | +0.335 | 0.371 | **AT (n.s.)** | A @ 92.8% |
| **26** | 0.394531 | +12.562 | 1.8e−228 | **above** | A @ 40.1% |
| 28 | 0.431491 | +16.258 | ~0 | above | D @ 45.4% |
| 30 | 0.422945 | +15.404 | ~0 | above | C @ 37.4% |
| 31 | 0.387694 | +11.879 | 3.2e−205 | above | A @ 44.2% |

Paired exact McNemar on all 14 adjacent steps, BH at α=0.05 across the 14:

| step | Δpp | McNemar p | BH-sig |
|---|---:|---|---|
| k4→k6 | +2.286 | 2.2e−05 | ✔ |
| k6→k8 | **−1.189** | 0.0166 | ✔ |
| k8→k10 | +0.783 | 0.0595 | — |
| k10→k12 | **−1.987** | 1.4e−04 | ✔ |
| k12→k14 | +0.128 | 0.294 | — |
| k14→k16 | +9.821 | 3.8e−123 | ✔ |
| k16→k18 | +0.691 | 0.187 | — |
| k18→k20 | **−3.041** | 3.8e−08 | ✔ |
| k20→k22 | **−7.520** | 2.8e−184 | ✔ |
| k22→k24 | +4.202 | 1.8e−116 | ✔ |
| k24→k26 | +12.228 | 2.4e−157 | ✔ |
| k26→k28 | +3.696 | 1.5e−16 | ✔ |
| k28→k30 | **−0.855** | 0.0213 | ✔ |
| k30→k31 | **−3.525** | 5.9e−17 | ✔ |

**Measured counts (these are the numbers, use these and not any recalled figure):**

* **11 of 14** adjacent steps are BH-significant.
* **6 of 14** steps are *decreases*, and **all 6 are BH-significant.** There is no
  "it's just two adjacent noisy points" reading available.
* Direction reversals: **7 raw**, **5 among BH-significant steps only**.
* **3 maximal BH-significant descending runs** (k4→k12 region, k18→k22, k28→k31).
* **4 floor-verdict crossings** along the depth axis:
  BELOW×6 → above×3 → BELOW → AT → above×4.

### On the "FOUR times" figure

The task brief reports a prior recompute finding "letter reverses direction FOUR
times after the gap-fill". **That figure is not reproduced as a reversal count.**
Four is the number of **floor-verdict crossings** (BELOW→above→BELOW→AT→above).
The *direction*-reversal count is **5** (BH-significant steps) or **7** (raw). Both
numbers say the same thing qualitatively; the specific integer 4 belongs to a
different quantity and should be written as "the verdict crosses the floor four
times", not "the curve reverses four times".

### Disposition

**RETRACT** "family-general step function / sharp phase transition." The
retraction is scoped precisely:

* ✅ **KEPT (per-family, descriptive):** three of four families show a
  single-layer letter jump much larger than any content step at the same layer —
  Qwen3-8B k24→k25 **+48.02 pp** (`0.229739`→`0.709942`) while content moves
  **+1.35 pp** (`0.304515`→`0.317975`) on the same forward passes; Llama-3-8B
  k17→k18 **+30.34 pp**; OLMo-2-7B k18→k19 **+26.68 pp**. These are read from
  `summary.json` and stand.
* ❌ **RETRACTED:** that this is a *family-general* form. Llama-2-7B — the fourth
  family, same protocol, same items — is not a step: 6 BH-significant decreases,
  5 BH-significant direction reversals, and the floor verdict crossing 4 times.
  n=4 families with 1 clear counterexample cannot carry "family-general".
* ❌ **RETRACTED:** the word "step function" as a functional-form claim for any
  family. No change-point inference was run, and the audit is right that none of
  the three requirements (change-point inference, answer-order replication,
  subject-level replication) was met. Even the three "clean" families are not
  monotone: measured **raw** letter reversals over each family's full merged grid
  (reversal = sign change in the adjacent-delta sequence, exact-zero steps dropped)
  are Qwen3 **13** (20 depths), OLMo-2 **11** (22 depths), Llama-3 **9**
  (17 depths), Llama-2 **7** (15 depths); largest single letter decrease is
  Qwen3 **−7.49 pp** and Llama-2 **−7.52 pp**. The jump is much larger than the
  wiggle, but the wiggle is not zero, so "step function" is a cartoon of the
  curve, not the curve.
* ✅ **KEPT and unchanged:** the *use* A01 makes of the curve, which does not need
  the step at all — arms below the transition sit at/below their floor and are not
  measurements, so a damage-scaling regression on letter accuracy that mixes
  sub- and supra-transition rungs is not estimating one quantity. Llama-2
  strengthens this rather than weakening it: its floor verdict is
  non-monotone in depth, so on Llama-2 you cannot even order the rungs by
  "how measurable" they are.

---

## 2. R-2 — "Llama-2 content is strictly monotone" is FALSE as printed

`GATE1_DEPTHCURVE_VERDICT.md` §1 table row: "Llama-2-7B … monotone? **yes,
strictly**", and §3: "Its content curve over the same range is **strictly
monotone** (0.2650 → 0.2877 → 0.3139 → 0.3548)."

That parenthesis is the **k16/k20/k24/k28 subset only**. Over the full 15-depth
grid the content curve has **two decreases**:

| step | content_norm | Δpp | McNemar p |
|---|---|---:|---|
| k8→k10 | 0.254237 → 0.253027 | **−0.121** | 0.749 |
| k10→k12 | 0.253027 → 0.252884 | **−0.014** | 0.984 |

So the honest statement is: **content is monotone at every BH-significant step
(8/14 steps are BH-significant, 0 of them decreasing), and its only two decreases
are −0.12 pp and −0.01 pp, both indistinguishable from zero (p=0.75, p=0.98).**
That is a *stronger* and *true* claim than "strictly monotone", which is false.
The word "strictly" is removed. The contrast with letter survives intact and gets
sharper: letter has 6 decreases and **all 6 are BH-significant**; content has 2
decreases and **0 are**.

---

## 3. R-3 — the tie-convention claim is DEMOTED

### What A01 claimed

`STATUS.json:novelty_check.strongest_remaining_novel_claim` and
`GATE3_CONVENTIONS_VERDICT.md`'s front-matter verdict:
> "THE NULL A01 RECOMMENDS HAS ITS OWN UNDECLARED CONVENTION DEGREE OF FREEDOM,
> AND IT REVERSES 5/6 ARM VERDICTS"

with a headline "**five defensible readings** put the null between 0.196126 and
0.453710 — a 25.76 pp spread".

### Why the audit is right

The 25.76 pp spread and the 5/6 reversal are **entirely** produced by `credit` and
`wrong`, and neither is an executable input-blind policy:

* `credit` scores 1 whenever gold ∈ W (the token-count winner set). To attain it,
  the baseline must break the tie in gold's favour — i.e. **know the gold letter**.
  An input-blind baseline by construction cannot. `credit` is an **oracle upper
  bound** on the null, not a null.
* `wrong` scores 0 on every tie. A baseline that must emit an answer gets `1/|W|`
  in expectation, so no policy attains `wrong` either. It is a **pessimistic lower
  bound**.

Together they bound the *identified set* of the null when the tie policy is
unstated. That is a legitimate and worth-reporting object. It is **not** "five
equally defensible conventions", and "flips 5/6 verdicts" is a statement about an
oracle, not about a reader's defensible choice.

### The executable conventions, recomputed

Three conventions are executable by an input-blind baseline: `split` (uniform
random tie-break, the pre-registered canonical), `first` (lowest index — what
`argmax` does), `last` (highest index).

| convention | null (MMLU n=14042, OLMo-2 tokenizer) | 6-arm verdicts (bf16 content_norm) |
|---|---:|---|
| `split` (canonical) | **0.284450** | **6 above / 0 at / 0 below** |
| `first` | 0.281085 | **6 above / 0 at / 0 below** |
| `last` | 0.282154 | **6 above / 0 at / 0 below** |
| — bound — `credit` (oracle UB) | 0.453710 | 1 above / 0 at / 5 below |
| — bound — `wrong` (pessimistic LB) | 0.196126 | 6 above / 0 at / 0 below |

**Measured spread over the three executable conventions: `0.3365 pp`**
(0.281085 … 0.284450). Not 25.76 pp. **0 of 6 arm verdicts move.** All six arms
are above all three executable nulls.

Residual fractions, bf16 content_norm, executable-only vs all-five:

| arm | split | first | last | exec-only span | exec-only ratio | all-five ratio |
|---|---:|---:|---:|---:|---:|---:|
| `7B_base` | 0.3955 | 0.4027 | 0.4004 | 0.0072 | **1.018×** | 16.26× |
| `7B_shortgpt16_step200000` | 0.2909 | 0.2993 | 0.2966 | 0.0084 | **1.029×** | sign change |
| `7B_keep14_step200000` | 0.2577 | 0.2665 | 0.2637 | 0.0088 | **1.034×** | sign change |
| `7B_keep12_step124000` | 0.2162 | 0.2255 | 0.2225 | 0.0093 | **1.043×** | sign change |
| `7B_keep10_step83500` | 0.1742 | 0.1840 | 0.1809 | 0.0098 | **1.056×** | sign change |
| `7B_keep8_step121000` | 0.1689 | 0.1787 | 0.1756 | 0.0098 | **1.058×** | sign change |

(The five damaged/healed arms' `credit` residual fractions are negative, so a
"ratio" over all five conventions is not defined — printed as "sign change".
That is itself the correct way to describe what `credit` does: it changes the
sign of the residual, which is what a bound can do and a convention should not.)

### Rewritten claim

> **The longest-option null is under-specified on ties, and the reader must be told
> which tie policy was used.** On MMLU, 34.22% of items have several options tied
> on continuation-token count and 13.37% (1,877 items) have all four tied. Among
> the **three tie policies an input-blind baseline can actually execute**
> (split / first / last) the null moves **0.34 pp** and **no arm verdict changes
> (6/6 above under all three)**. The remaining sensitivity is a **bounds**
> statement, not a convention statement: an oracle tie-break (`credit`, 0.453710)
> and an always-wrong tie-break (`wrong`, 0.196126) bracket the null's identified
> set at **25.76 pp**, and under the oracle bound 5 of 6 arms would read BELOW.
> So the reporting requirement is real but modest for the point estimate, and
> becomes load-bearing only if the tie policy is left unstated — in which case the
> honest null is an interval, not a number.

This is **weaker** than what `STATUS.json` called A01's
`strongest_remaining_novel_claim`, and it is no longer the strongest claim.
`STATUS.json` and `PROPOSAL.md` are updated to say so.

### What did NOT change

Every A01 number ever published used `split`, said so, and reproduces the archived
`*_dtype_summary.json` to `<1e-12` (regression gate in
`code/a01_gate3_content_conventions.py`, re-asserted). The demotion is a claim
demotion, not a numeric correction.

---

## 4. R-4 — "damage turns letter into a constant predictor" — ACCEPT the narrowing

The audit: below-floor accuracy does not imply per-item constancy when the modal
share is only 45–47%. Correct, and the Llama-2 curve makes the point sharply,
because modal share and floor verdict **decouple**:

* k12: modal A @ **100.0%** — genuinely a constant emitter, letter 0.229454 BELOW floor.
* k22: modal A @ **99.9%** — same.
* k8: modal A @ **44.9%** — letter 0.241490, also BELOW floor, but this arm is
  **not** a constant emitter. 44.9% modal.
* k6: modal D @ **43.5%** — BELOW floor, not constant.
* k28: modal D @ **45.4%** — ABOVE floor by +16.26 pp, with the *same* modal share
  as k8. So modal share alone predicts neither the floor verdict nor its sign.

Correct wording, adopted: *"damage drives the letter readout to or below its
best-constant floor; on some arms this is accompanied by literal modal collapse
(Llama-2 k12 100.0%, k22 99.9%) and on others it is not (k8 44.9%). Being at or
below a constant predictor's accuracy is not the same property as being a constant
predictor, and A01 should only assert the former."* Modal share is now printed per
depth in `evidence/a01_audit_response_recompute.json`.

---

## 5. R-5 — cross-construct "7–10×" span — ACCEPT

The audit says accuracy / EM / CKA / depth-fraction have no common estimand, so a
span across C1–C4 is a descriptive sensitivity of one chosen aggregation and
nothing more. Gate-4 already reached half of this
(`GATE4_VERDICT.md`: "the correct headline is 'residual fractions span roughly
7–10×'"), but it framed the span as a property of the *aggregation choice* rather
than admitting the four constructs are not commensurable in the first place.
Accepted as stated; the number stays, the interpretation narrows to "descriptive
sensitivity of the listed aggregation across four non-commensurable constructs".
No recompute needed — this is an interpretation defect, not an arithmetic one.

---

## 6. R-9 — the tokenizer leg (NOT attacked by the audit) — KEPT and strengthened

The audit's A01 section never challenges this leg. It is the one place where the
null's under-specification produces a **verdict change under a policy a reader
would plausibly adopt** (reuse one published null across families), rather than
under an oracle.

"Longest option" is counted in **continuation tokens**, so the same `split`
convention on the same 14,042 MMLU items gives a different null per tokenizer
(`evidence/a01_gate1_third_family.json:longest_option_split_tie_null`):

| family | longest-option split-tie null |
|---|---:|
| Llama-2-7B | **0.275661** |
| Qwen3-8B-Base | 0.283346 |
| OLMo-2-7B | 0.284450 |
| Llama-3-8B | **0.284664** |

Span **0.9003 pp** — which is **2.68×** the entire executable tie-convention span
(0.3365 pp) from §3. **The tokenizer axis is the larger of the two, and it is the
one A01 had been treating as a footnote.**

New this session: does it flip anything? I re-tested all **63** non-OLMo damaged
depth arms (both disks) with their own tokenizer's null vs the shared OLMo-2
`0.284450`, exact two-sided binomial, α=0.05, A01's existing above/AT/BELOW
trichotomy:

* **2 of 63 arm verdicts flip.**
* **1 is robust:** `gate1_dmg_llama2_7b_depth_k20`, content_norm `0.287708`.
  Against Llama-2's own null `0.275661`: **above**, p=**0.00146**. Against the
  shared `0.284450`: **AT (n.s.)**, p=0.395. One side is p<0.005, so this is not a
  boundary artifact — borrowing another family's null erases a real above-null
  verdict.
* **1 is a boundary artifact and is reported as such:**
  `gate1_dmg_llama3_8b_depth_k17` straddles α (p=0.0507 own vs p=0.0443 shared).
  Counted, flagged `robust=false`, not used as evidence.

**Threshold disclosure (required by A01's own protocol):** α=0.05 and the
above/AT/BELOW trichotomy are reused unchanged from `code/a01_gate3_fp32_vs_bf16.py`
and `code/a01_gate1_verdict.py`, i.e. pre-existing. The **`robust` flag
(min(p) < 0.005) was defined in this session, AFTER seeing that one of the two
flips straddled α.** It is disclosed as post-hoc. It exists to stop A01 counting a
0.0507-vs-0.0443 pair as a finding, so it makes the claim weaker, not stronger.
The 63 per-arm tests are **uncorrected**; with BH across 63 the single robust flip
(p=0.00146) survives at α=0.05 and the boundary one does not.

Claim, as it should be written:

> A content readout compared against **another family's** longest-option null is
> mis-calibrated by up to 0.90 pp, which is enough to change a verdict: on
> Llama-2 truncated to 20 layers, content_norm 0.287708 is significantly above
> Llama-2's own null 0.275661 (p=0.0015) but indistinguishable from the OLMo-2
> null 0.284450 (p=0.39). Any cross-family content comparison therefore needs a
> **per-family, per-convention** null. This axis is independent of the tie
> convention and, measured on our arms, 2.68× larger.

---

## 7. R-7 — the illegal p-value is a REAL open defect (not fixed here)

`evidence/gate3_dtype_runs/7B_base_dtype_summary.json` carries
`letter_acc_diff_boot_p = 1.042`. A p-value cannot exceed 1. This is a bug in the
bootstrap two-sided p construction (almost certainly a doubled one-sided tail
without a `min(1, ·)` clamp), not a typo, and it means every `boot_p` produced by
that code path near p≈1 is suspect in *value* even where the *verdict* is
unaffected. This pass does **not** fix it — fixing it means patching the estimator
in `code/a01_gate3_fp32_vs_bf16.py` and re-emitting all six dtype summaries, which
is a separate change with its own regression gate.

Logged as the top open item. Note the affected arm's *conclusion* does not depend
on it: the base arm's letter accuracy is byte-identical across dtypes
(0.6053980914399658 both), McNemar p=1.000, CI95 [−0.0011, +0.0011] — the verdict
is carried by McNemar and the CI, not by the malformed bootstrap p.

> **UPDATE 2026-08-11 — R-7 is now FIXED and CLOSED** (the "not fixed here" above
> describes the 2026-08-10 pass and is retained for provenance).
> The audit's guessed cause ("missing `min(1,·)` clamp") was the symptom; the real
> mechanism is that `2*min((bs<=0).mean(),(bs>=0).mean())` **double-counts** the
> resamples whose mean is exactly 0, so the two tails sum to `1 + P(bs==0)`. On the
> base arm `d` is a difference of 0/1 correctness vectors with only 28+28 discordant
> items out of 14042, and **5.44%** of bootstrap means land exactly on 0 —
> reproducing `1.0420` bit-for-bit from the per-example shards. Fixed with a shared
> `two_sided_boot_p()` that splits the zero atom evenly between the tails, making
> `p ≤ 1` structural rather than a truncation; `paired_bootstrap()` (which feeds
> every `*_vs_null_boot_p`, i.e. the floor verdicts, and had the same bug clamped
> only from below) now uses it too. All six summaries re-emitted from disk, 0 GPU,
> 8/8 shards / n=14042 / nan=0 each. Base arm **1.042 → 0.9876**; **0 of 24 verdicts
> changed, 0 of 30 p-values crossed α=0.05**, all non-p fields byte-identical.
> Full writeup: `paperG/evidence/R7_BOOTSTRAP_P_FIX.md`.

---

## 8. R-8 — "general structural damage response" — scope was already narrower in the verdict file

`GATE1_DAMAGED_VERDICT.md` describes its arms accurately as front-N truncation
with no fresh block and no heal. The overreach is in `STATUS.json`'s summary field
calling it `A01_GENERAL_CLAIM_CONFIRMED`. Accepted: the damaged-arm result is
about **extreme front-truncation (k8/k12) with no heal**, plus one healed OLMo-2
keep8 arm from a different regime. The audit's own counterexample stands and is
recorded: OLMo-2 healed keep8 has content_norm **0.342259**, clearly above the
split null 0.284450 (+5.78 pp, boot p=0.0001), so "content cannot rescue a damaged
arm" is true of the six pure-truncation non-OLMo arms and **false** of that healed
arm. `STATUS.json`'s field is reworded.

---

## 9. What A01 still has after all of this

Not nothing, and this is the point of writing the ledger rather than deleting the
claims:

1. **Report a construct-appropriate input-blind null, not a generic chance line.**
   MMLU letter always-D `0.268908` not 0.25; BoolQ always-B `0.6217` not 0.50.
   Unaffected by every audit item.
2. **A floor failure disqualifies a readout from carrying a capability
   interpretation.** Six damaged non-OLMo arms at/below their floor; OLMo-2 healed
   keep8 letter `0.2550` below floor and *more* significantly so in fp32
   (−1.538 pp, p=0.0062). Unaffected.
3. **The tie/precision mechanism is falsified** — fp32 removes every exact tie and
   moves no verdict on 5/6 arms; intact Llama-2 has 15.79% ties while sitting
   +14.11 pp above floor. Unaffected (the audit agrees, §2.1 "站得住" item 3).
4. **Nulls must be reported with their tie policy AND their tokenizer.** Demoted
   in §3, strengthened in §6. The tokenizer half is now the stronger half.
5. **Self-falsification as method**: A01 retracted its own headline, then retracted
   that retraction for testing the wrong condition, and now — with this file —
   retracts a third claim on an outside audit's say-so. That is the paper's actual
   contribution and it costs nothing to keep.

## 10. Provenance

* Recompute: `code/a01_audit_response_recompute.py` → `evidence/a01_audit_response_recompute.json`
* Llama-2 inputs (wzc1): `olmo2_mmlu_content_results/gate1_dmg_llama2_7b{,_depth,_depth_gap,_depth_gap2}_k*/per_example_mmlu_shard{0..7}of8.jsonl`, 15 unique depths, 14042 items each, 0 nan
* Other three families: `olmo2_mmlu_content_results/gate1_dmg_{qwen3_8b,llama3_8b,olmo2_7b}_depth*_k*/summary.json` — mostly **zwfy6**, read read-only via `.82`; the 5 `llama3_8b_depth_fine_21_k*` arms are on **wzc1**
* Conventions: `evidence/gate3_content_null_conventions.csv` (120 rows), `evidence/a01_gate1_third_family.json`
* Estimators: exact two-sided binomial (arm vs null), exact McNemar (adjacent depths), BH at α=0.05 across the 14-step family. No bootstrap was used in this pass, deliberately — see §7.
* Cost: CPU only, ~2 min. No GPU. No checkpoint opened. One read-only ssh to `.82`.
