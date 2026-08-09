# Null-Calibration Methodology — P1 (task #170, experiment 1)

Owner: subagent dispatched 2026-08-07. Budget: 0 GPU-hours (pure recompute on existing artefacts).
This file is the only status doc this agent may write. `.tex`, `versions/*`, `*TODOList*`, other `status/*.md` are MAIN's.

---

## PRE-REGISTERED GATE (written before any computation was run)

> **Gate.** If the four residual fractions (calibrated_residual / reported_value) **do not span at least
> one order of magnitude** (i.e. max/min < 10), the cross-construct claim weakens to a single-construct
> note and the direction downgrades to a Paper B appendix.

Secondary pre-registered commitments, fixed before the numbers were recomputed:

1. **Nulls are construct-appropriate and named in advance**, not a generic chance line:
   - MC scoring interface -> best constant letter (always-D) **and** longest-option heuristic.
   - Generative (SQuAD) -> majority-label constant **and** empty string.
   - Representation similarity -> **layer-order shuffle**, explicitly NOT the random-init floor (~0.091),
     which is the wrong and self-flattering baseline.
   - Probe readout depth -> native readout knee.
2. **The z-CKA leg must be re-permuted from 200 to 2000 perms/pair over all 91 pairs with
   Benjamini-Hochberg correction at q=0.05.** The survivor count is reported honestly whatever it is;
   if it falls materially below the current 58/91 the headline is restated in this file.
3. **Any number that fails to reproduce is reported loudly**, in its own section, before the table.
4. The paper must carry the **self-falsification case** (Paper E Obs4) or the framing is preaching.

---

*(results appended below as they are produced)*
---

# RESULTS (run 2026-08-07, 0 GPU-hours)

Regenerator: `scripts/build_null_calibration_table.py` (house style of
`scripts/verify_interface_audit.py`; every printed number is recomputed from raw
per-example / per-pair artefacts). Run log `logs/null_calibration_p1_nperm2000.log`,
machine-readable dump `results/null_calibration/null_calibration_p1_nperm2000.json`.

```
python3 scripts/build_null_calibration_table.py --n-perm 2000 --seed 0 \
    --out results/null_calibration/null_calibration_p1_nperm2000.json
```

GPU usage: **zero**. The C3 re-permutation acts only on the cached z-CKA matrices
in `proposal/shared/representation/cka_matrices/` (109 JSON files, 1.7 MB,
originally copied `scp -O` from `.73`). No activations were
re-extracted; `align_acts/` (18 GB, 14 models) was **not** needed.

## 1. Four-row master table

| construct | reported | construct-appropriate null | residual | resid / reported |
|---|---|---|---|---|
| C1 MC scoring interface (n=14,042 × 9 arms) | 0.3598 | 0.2845 longest-option (content's own floor) | 0.0753 | **0.2094** |
| C2 Generative label prior (n=2,000) | 0.6590 | 0.4985 majority-label constant | 0.1605 | **0.2436** |
| C3 Representation similarity (n=91 pairs) | 0.4907 | 0.4529 layer-order shuffle | 0.0377 | **0.0769** |
| C4 Probe readout depth (2 families × 3 tasks × 5 splits) | 0.6610 | 0.1505 native readout knee | 0.5105 | **0.7724** |

Secondary nulls, reported because each interface has more than one floor:
- C1: naive chance line .2500 (residual frac .3051); best constant letter always-D
  **.2689** (residual frac .2527). The longest-option floor is the *primary* null
  because the reported number is a **content**-interface number, and .2845 > .2689.
- C2: empty string EM **0.0000** (residual frac 1.0000 — an uninformative null,
  which is exactly why the majority-label constant is the pre-registered one).
- C3: random-init floor **0.0912** — the WRONG null. Using it would report a
  residual of 0.3995 instead of 0.0377, a **10.6× self-flattering inflation**.
- C4: linear-probe knees Qwen3-8B 0.3926 [0.3193,0.4659] / OLMo-2-7B 0.2854
  [0.2138,0.3570]; native knees (3-task mean) 0.8241 / 0.8750.

## 2. The two headline numbers, restated precisely (both drift from the slate)

**Headline 1 — MC scoring-interface inflation.** The content interface hands
`scratch16L @200k` (a random 16-layer block healed 200k steps, whose *letter*
accuracy .2470 is **2.19pp BELOW** the always-D floor, i.e. at/below the letter
floor by construction) a content_norm of **.3598**. Structural inflation:
**+10.98pp** vs the .25 chance line, **+9.09pp** vs always-D, **+7.53pp** vs the
content interface's own longest-option floor. The arm-to-arm effect the interface
is used to measure is keep14@200k − scratch16L@200k = **.3832 − .3598 = 2.34pp**.

> ⚠️ **The ratio is 4.69×, not the 4.8× the slate states.** 0.10978 / 0.023430 =
> **4.6854**. Against the content interface's own floor it is **3.22×**. Both are
> the honest framings; 4.8× does not reproduce and must not be quoted.

**Headline 2 — layer-order-shuffle null.** At 2000 perms/pair the null mean is
0.452936, which is **92.31%** of the reported midband z-CKA 0.490672. Usable
layer-correspondence signal = **0.0377**, not ≈0.4. (At the shipped 200 perms it
is 92.25%; the slate's "92.3%" is right to 3 s.f. — this one reproduces.)

## 3. ★ The statistical repair: 200 → 2000 perms/pair + Benjamini-Hochberg

Verification first: **at n_perm=200 the script reproduces the shipped file to the
last digit** — observed mean `0.4906724527457283` (drift 0.00e+00), null mean
`0.4526328836126522` (drift 0.00e+00), identity gate max|M[i][i]−1| = 1.777e-07.
So the re-permutation runs on the same object the paper would cite.

| | shipped | this run |
|---|---|---|
| n_perm / pair | 200 | **2000** |
| total null draws | 18,200 | **182,000** |
| min attainable p | 4.98e-03 | **5.00e-04** |
| null mean | 0.4526329 | 0.4529357 |
| median per-pair p | 0.015 | **0.015492** |
| pairs raw p<0.05 | 58/91 | **57/91** |
| pairs BH q=0.05 | *not computed* | **52/91 (57.1%)** |
| pairs obs > own null mean | 77/91 | 76/91 |

p-values use the add-one estimator p = (#{null ≥ obs} + 1)/(K + 1), which is the
correct finite-permutation form (an exact-0 p is not attainable).

**Honest answer to "how many of 91 survive BH at q=0.05": 52/91.** That is
*above* the 58/91 raw figure's BH-corrected value at 200 perms (**49/91**, also
computed here), because more permutations resolve the small p-values away from
the 1/201 floor faster than BH shrinks them. So the repair **helped** rather than
hurt — but the number to report is 52/91 = 57.1%, and the earlier "58/91" was
uncorrected and must not be quoted.

**Seed stability** (5 seeds at 2000 perms): null mean 0.45294–0.45306, median p
0.01399–0.01599, raw p<0.05 **57–58**, **BH q=0.05 50–52**. So the survivor count
is 50–52/91; report as **≈52/91 (50–52 across seeds)**, never as a bare 52.

**Stratification** (2000 perms, seed 0): same-family 6/11 survive, cross-family
46/80. Survivors' mean residual 0.0621 vs non-survivors' 0.0052. **15/91 pairs
have an observed midband value BELOW their own shuffle-null mean** — for those
pairs the midband correspondence is worse than a random layer reordering.

## 4. Pre-registered gate: PASSES, but on a knife edge — report the sensitivity

Residual fractions span **0.0769 → 0.7724 = 10.04×**. The gate (≥10×) **PASSES**.

⚠️ It passes by 0.4%. The span turns entirely on C4, the leg with the most
operationalization freedom, and **4 of 5 reasonable C4 variants fail**:

| C4 variant | resid frac | span | gate |
|---|---|---|---|
| Qwen+OLMo, native 3-task mean, pooled (headline) | 0.7724 | **10.04×** | PASS |
| Qwen+OLMo, native 3-task mean, per-model then avg | 0.7677 | 9.98× | FAIL |
| all 3 models, native 3-task mean, per-model then avg | 0.7074 | 9.20× | FAIL |
| all 3 models, native = SST2 only (matched support) | 0.6852 | 8.91× | FAIL |
| Qwen+OLMo, native = SST2 only | 0.5278 | 6.86× | FAIL |

**Recommendation: do not headline "an order of magnitude."** The defensible claim
is "the residual fraction ranges from **8% to 77%**, a ~7–10× spread depending on
how the probe leg is aggregated" — which supports the cross-construct claim
without resting on a 0.4% margin. A reviewer who recomputes C4 per-model instead
of pooled gets 9.98× and the literal gate fails. Any paper claiming ≥10× must
pre-declare the pooled 3-task-mean aggregation *and* print this table.

## 5. Verification status of each leg

| # | claim | status |
|---|---|---|
| 1 | MC constant-D floor .2689, n=14042 | ✓ A .2295 / B .2465 / C .2551 / D **.2689**, n=14042 |
| 2 | SQuAD majority refusal 997/2000 = .4985 | ✓ exact; label `根据提供的信息无法回答这个问题`; field is `target_text` |
| 3 | z-CKA .4906724527 vs shuffle null .4526328836, n=91, gate 1.777e-7 | ✓ **bit-exact, drift 0.00e+00** |
| 4 | probe knee 0.393L Qwen / 0.285L OLMo vs native 0.824L / 0.875L | ✓ **REPRODUCES from data, not from the .tex** |

**Leg 4 verified independently of `tab_depth.tex`** from `results/p1_2/p1_2_summary.json`
(the .tex is downstream of it; `results/probe_linguistic_*.json` is a *different*
probe suite and does NOT contain these values — it reports POS/DEPREL/CoLA
saturation and next-token logit-lens, so anyone looking there will wrongly
conclude the numbers are missing):

- Qwen3-8B (L=36): `content_j_frac_mean` **0.3926** CI95 [0.3193, 0.4659], n_points 15;
  native per task RTE 0.9444 / SST2 0.6389 / WiC 0.8889 → mean **0.8241**.
- OLMo-2-7B (L=32): `content_j_frac_mean` **0.2854** CI95 [0.2138, 0.3570], n_points 15;
  native RTE 1.0000 / SST2 0.7500 / WiC 0.8750 → mean **0.8750**.
- Llama-3-8B (L=32): linear **0.2688** (the .tex prints 0.275, which is the
  SST2-only knee 0.2750, not the 3-task mean — both are in the data, the .tex
  caption already discloses the SST2-only restriction, so this is a labelling
  nuance, not an error).

So the paper keeps **four** constructs. The cross-construct claim is not weakened
on this axis.

## 6. Discrepancies found — must be fixed before anything ships

1. **4.8× → 4.69×** (headline 1). Does not reproduce. See §2.
2. **longest-option floor .2822 → .2845.** The `.2822` recorded in
   `mmlu_interface_initial_dossier.md:137` is the
   **last-of-maximal** tie convention. 4,805/14,042 items have ≥2 maximal-length
   options, so the convention is load-bearing: split **0.2845** / first 0.2811 /
   last **0.2822** / optimistic 0.4537 / pessimistic 0.1961. `split` is the
   defensible one and `scripts/verify_interface_audit.py` already uses it (it
   prints 0.2845). **Quote .2845, and print the convention.**
3. **58/91 was never BH-corrected.** BH at 200 perms is 49/91. See §3.
4. **The "order of magnitude" gate is a 0.4% margin.** See §4.
5. C2's second null (empty string) gives EM 0.0000 → residual fraction 1.0, an
   uninformative null. Report it only to show the majority-label null is the
   binding one, never as the calibration baseline.

## 7. Boundary paragraph vs arXiv:2602.14486 (genuine prior art)

**Venue verified via OpenReview, not S2/DBLP.** Note `uz0gAAYydl`,
`venueid = ICML.cc/2026/Conference`, `venue = "ICML 2026 regular"`, invitations
include `ICML.cc/2026/Conference/Submission15852/-/Camera_Ready_Revision`;
cdate 2026-01-22, pdate 2026-04-30. arXiv v1 **Mon 16 Feb 2026 06:01:23 UTC**
(v2 2026-06-25, "ICML 2026 camera-ready"). **Pre-cutoff → cite as prior art, not
concurrent.** Authors Gröger et al. (EPFL / Basel / HSLU), PMLR 306.

> **Boundary.** Gröger et al. (ICML 2026) introduce permutation null-calibration
> for representational similarity and, like us, find that apparent convergence
> "largely disappears after calibration." We adopt their framing and differ on
> three axes that are each load-bearing. (i) **What is permuted.** Their null
> permutes **sample/row correspondences** — "s(X, π(Y)), where π(Y) permutes the
> rows of Y" — and is designed to correct two **scale** confounds: a width
> confound (chance similarity grows with representation width d relative to n)
> and a depth confound in which the reported max over M = L_A·L_B layer pairs
> inflates as E_H0[T_max] ≤ µ + Cσ·sqrt(log M), i.e. a *search-space* effect.
> They explicitly decline to compare individual layer pairs "because it is
> unknown" which pair is correct. Our null permutes **B's layer ORDER**, leaving
> every CKA entry untouched and changing only which of B's layers count as
> midband; it therefore tests the question their aggregation-based null brackets
> away — *is layer i the right partner for layer j*, which is precisely the
> quantity a layer-stitching or layer-correspondence claim rests on. Their
> calibration cannot answer it and ours cannot correct their width confound; the
> two nulls are complementary, not competing. (ii) **Breadth.** They calibrate
> three metric families (spectral CKA/CCA, neighborhood mKNN, geometric RSA)
> within the **single** construct of representation similarity. We run
> construct-appropriate calibration across **four unrelated construct families** —
> MC scoring interface, generative label prior, representation similarity, probe
> readout depth — which is what converts a metric fix into a claim about
> evaluation practice. (iii) **Self-application.** Our worked examples include a
> case where the protocol falsified *our own* headline. We must also concede two
> points to them rather than claim them: they apply Benjamini-Hochberg FDR across
> model pairs (their §"we further apply Benjamini-Hochberg FDR correction ... to
> control for multiple comparisons across model pairs"), so BH is **their**
> practice that we adopt, not our contribution; and they use K = 200
> permutations, the same count whose resolution floor we criticise in our own
> earlier analysis — so the 2000-perm extension is a self-repair, not a
> criticism of them.

**Do not claim**: that we introduce null calibration for similarity; that we
introduce BH correction; that they omit multiple-testing correction. All three
would be false and are the kind of claim that gets retracted.

## 8. Positioning vs arXiv:2606.16897 (CONCURRENT)

arXiv v1 **Mon 15 Jun 2026 16:07:10 UTC** → after the 2026-05-07 cutoff →
**concurrent; cite and position, does not preempt.** "Contrastive-Difference CKA
Reveals Concept-Specific Structural Alignment Across Language Model
Architectures" (Gao et al.).

**Precision guard verified against the PDF, and the earlier reviewer was wrong.**
They **DO** run a permutation null: "Permutation testing (n=200) validates: real
CKA∆ = 0.727 vs. null = 0.689 ± 0.005 (p < 0.005; permutation z = 7.4)", with
Mann-Whitney U as a nonparametric confirmation, and n=1,000 shuffles for their SAE
multi-test analysis. So the criticism is **not** "they omitted the null."

The criticism that *is* available: the null is run, significance holds
(z = 7.4), and the calibrated residual is **0.727 − 0.689 = 0.0380 = 5.23% of the
reported value** (equivalently, the null accounts for **94.77%** of it), while the
abstract reports "moderate geometric convergence (CKA∆ = 0.727) with near-perfect
functional equivalence (≥94% affine transfer)." The absolute number carries the
rhetorical weight; the calibrated residual is 5% of it. **This is the mandatory
column we propose, and we apply it to ourselves first: our own 91-pair
measurement has a residual of 0.0377 — numerically the same 0.038 gap** (ours is
7.69% of reported, theirs 5.23%). Their own limitation L2 already concedes
absolute CKA∆ varies 0.727 → 0.366 with question-set coverage, which supports
our point rather than contradicting it.

## 9. Required controls — status

| control | status |
|---|---|
| MC null = best constant letter **and** longest-option | ✓ both, with all 5 tie conventions printed |
| generative null = majority-label constant **and** empty string | ✓ both (.4985 / .0000) |
| z-CKA null = layer-order shuffle, **NOT** random-init | ✓ shuffle 0.4529 used; random-init 0.0912 printed only as the named wrong baseline, with the 10.6× inflation it would produce |
| probe null = native readout | ✓ per-task and aggregated |
| nulls reported for OUR numbers as aggressively as others' | ✓ §6 lists five discrepancies in our own record, incl. two the slate got wrong |
| self-falsification case (Paper E Obs4) | ⚠️ **NOT yet written up here** — the retraction is recorded in the slate (both flip arms sat at the chance floor) but the per-example flip-pair recompute is not in this run. Must be added before submission or the paper is preaching. Cost: CPU only. |

## 10. Verdict

- Four constructs stand; leg 4 verified from data. No leg lost.
- The statistical repair is done and **improves** the picture (52/91 BH vs 49/91
  BH at 200 perms), and the honest number is ≈52/91 (50–52 across seeds), 57.1%.
- The order-of-magnitude gate technically passes at 10.04× but is 0.4% from
  failing and fails under 4/5 C4 variants → **recommend restating the claim as a
  7–10× spread (8%–77%) and printing the sensitivity table**, rather than leaning
  on "an order of magnitude."
- 2602.14486 is genuine prior art and already does permutation calibration + BH
  for similarity. Our differentiation is real (layer-order vs sample-order;
  layer-correspondence vs scale; four constructs vs one) but **narrower than the
  slate implies**, because BH is theirs and K=200 is theirs too.
