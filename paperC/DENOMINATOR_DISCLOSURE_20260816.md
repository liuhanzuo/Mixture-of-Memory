# Designated-damaged denominator disclosure (paperC, 2026-08-16)

> **STATUS: COMPLETE.** Implemented and committed as `387dc90`. Independent re-derivation by
> agent, 0 GPU. Scripts: `paperC/code/recount_designated_denominators.py`,
> `paperC/code/gate_designated_denominator.py`, `paperC/code/gate_build_record_matches_pdf.py`.
> Evidence: `paperC/evidence/designated_damaged_denominators.json`.
> The §"STOP-AND-REPORT" block below was written to disk before implementation began, per
> instruction; NEW-1 was retracted 5 minutes later and the retraction is kept in place.

## 🔴 STOP-AND-REPORT: findings BEYOND what MAIN adjudicated

**Headline: there IS a fourth same-class instance (NEW-2), and it is the one that decides the
remedy. The chance side DOES change (10/15 → 20/25 off MMLU), but LESS than the floor side, and
in a direction that leaves the flip claim intact.**

### ~~NEW-1~~ RETRACTED (my error, 5 min after writing it). MAIN's 14/16 is CORRECT.

I first wrote that MAIN's "14/16" was wrong arithmetic. **It is not.** 14/16 is exactly right for
restoring `shortgpt16` alone, which is what MAIN's option (a) proposed. I had conflated it with the
*both-arms* restore. Retained here rather than deleted, because a retraction inside the deliverable
is provenance and because it demonstrates the failure mode this whole document is about: I quoted a
ratio without stating which arm set it was over.

The full ladder of MMLU-Pro denominators, independently recomputed:

| denominator rule | at-or-below floor | above floor |
|---|---|---|
| as reported | **14/15** (93.3%) | 1/15 (qwen3/k14 +0.233) |
| + `shortgpt16` only (= MAIN's option (a)) | **14/16** (87.5%) | 2/16 |
| + `keep14` only (parity with non-OLMo `k14`, which IS counted) | **15/16** (93.8%) | 1/16 |
| + both (full restore) | **15/17** (88.2%) | 2/17 |
| drop ALL `k14`/`keep14` (the other parity fix) | **12/12** (100%) | 0/12 |

`keep14` is `AT the floor (n.s.)` on MMLU-Pro (+0.3241 pp, hw 0.636, p=0.3234), so restoring it adds
to numerator *and* denominator and the at-or-below rate goes **up**. Only `shortgpt16` is above.
Note the last row: the *other* internally-consistent fix (drop depth-14 everywhere) gives a
**stronger** headline than the one in the paper.

### 🔴 NEW-2 (FOURTH instance of the class). `k14` is counted as damaged for the three non-OLMo families but `keep14` is NOT counted for OLMo-2 -- inside the *same* 15-cell denominator.

`code/mmlu_pro_power_nulls.py:96,100`:
```python
DAMAGED_OLMO = {"keep8", "keep10", "keep12"}          # 3 rungs -- keep14 and shortgpt16 dropped
XF_DAMAGED   = {"k8", "k10", "k12", "k14"}            # 4 rungs -- k14 INCLUDED
```
The 15-cell denominator is `3 (OLMo) + 3 families x 4 (non-OLMo) = 15`. So depth-14 is damaged
if the model is Llama-2/Llama-3/Qwen3 and not damaged if it is OLMo-2. There is **no stated rule**
that distinguishes them, and `04_experiments.tex:23` says the designation "is fixed by
construction, never by a measured score." Depth 14/32 is the same construction in both cases.
This also propagates into the `0/60`: that count is `3 families x 4 rungs x 5 benchmarks`, and
**all three `k14` arms are inside it**.

### 🟠 NEW-3. The paper's own power denominator is **21**, and `21 - 4 intact bases = 17` designated damaged cells. The verdict denominator is 15.

`00_abstract.tex:2`: "On MMLU-Pro, all 21 evaluated cells are powered at the scale of the reference
effect." Verified: `power_verdict.n_cells = 21`, `n_cells_powered = 21`. Those 21 are
`6 OLMo-2 rungs (base, shortgpt16, keep14, keep12, keep10, keep8) + 3 families x 5 rungs`.
Removing the 4 intact bases leaves **17 damaged cells**. The paper counts 21 when the count is
favourable ("all 21 powered") and 15 when it is not ("14/15 at or below the floor"). `shortgpt16`
and `keep14` are inside the first denominator and outside the second.

### 🟡 Chance side DOES change (MAIN explicitly did not determine this)

Off-MMLU, `10/15 above chance` becomes **20/25**; MMLU-Pro's `13/15 above chance(mean)` becomes
`15/17`. Neither is as dramatic as the floor side (which goes 0/15 -> 9/25), because the three
counted arms already read above chance -- which is the whole point of the flip claim. Full table below.


---

## 1. Independent recomputation (I did not copy MAIN's numbers)

Script: `paperC/code/recount_designated_denominators.py`, plus
`paperC/evidence/designated_damaged_denominators.json` (85 off-MMLU + 21 MMLU-Pro cells,
every field re-derived). Every floor delta was recomputed as `(acc - floor) * 100` from the
raw `acc` and re-asserted against the stored `delta_pp` at `1e-6`; all 106 assertions held,
so the stored deltas are internally consistent with the accuracies.

### 1a. Off-MMLU, OLMo-2, five small benchmarks (Winogrande excluded as the declared control)

| retained | arm | above floor | median Δ | min Δ | max Δ | above chance |
|---|---|---|---|---|---|---|
| 16/32 | `shortgpt16` | **5/5** | +30.200 | +11.208 | +49.663 | 5/5 |
| 14/32 | `keep14` | **4/5** | +6.997 | +2.612 | +18.687 | 5/5 |
| 12/32 | `keep12` | 0/5 | −0.427 | −2.200 | +0.246 | 5/5 |
| 10/32 | `keep10` | 0/5 | −1.556 | −5.000 | +1.621 | 1/5 |
| 8/32 | `keep8` | 0/5 | −0.939 | −1.800 | +2.503 | 4/5 |

Per-cell above-floor detail for the two excluded arms (all recomputed):
`shortgpt16` arc_c +31.143, arc_e +49.663, obqa +30.200, csqa +24.324, piqa +11.208, all p=0.0001.
`keep14` arc_c +6.997 (p=0.0001), arc_e +18.687 (0.0001), obqa +9.400 (0.0005), csqa +6.470 (0.0001);
piqa +2.612 (p=0.1680) is at the floor.
**These reproduce MAIN's figures exactly.** The three counted arms give 15 cells all within
±5.0 pp of their floors, 14/15 n.s.; the single significant one is `keep10`/arc_easy at
−2.694 pp (p=0.0290), **below** the floor.

I confirmed the verdict rule is not doing the work: the `verdict` string, `ci95_pp` excluding
zero, and `boot_p < 0.05` give **identical** counts on every arm set tested (0/15 and 9/25
under all three).

### 1b. ★ THE CHANCE SIDE — the half MAIN did not compute

| arm set | above floor | above chance |
|---|---|---|
| as reported (keep12/10/8) | 0/15 | **10/15** |
| + `shortgpt16` | 5/20 | 15/20 |
| + `keep14` | 4/20 | 15/20 |
| + both (adopted) | **9/25** | **20/25** |

MMLU-Pro chance side: 13/15 → **15/17** under `mean(1/n_opt)`; 15/15 → **17/17** under naive 0.10.

**The chance side changes less than the floor side, and that is the important part.**
Both restored arms read above chance in every cell they clear (and `keep14` reads above chance
even in the piqa cell it does not clear). So restoring them *strengthens* the paper's actual
thesis — the gap between "reads above chance" and "clears its floor" — for the three severely
damaged arms, while adding two arms for which the two references agree. The defect damages the
*near-unanimity* claim, not the flip claim. Over all 85 cells: 45/85 above chance, 9/85 above floor.

### 1c. The `0/60`, recounted from scratch

`0/60` = 3 non-OLMo families × 4 rungs (`k14,k12,k10,k8`) × 5 benchmarks. **It reproduces
exactly**, as do the three companion counts: `7/60` significantly below, `25/60` above chance,
`52/60` underpowered. `arc_challenge` median damaged effect −3.8396 and median half-width 3.9249
also reproduce. Only 4/60 cells are even *point*-above their floor (largest +0.983), all n.s.
**No hidden exclusion inside the 60.** But see NEW-2: the 60 *includes* `k14`.

## 2. The other three families — checked, and this is where the decisive finding is

| family | rollup `damaged_rungs` | n | matches manuscript? |
|---|---|---|---|
| `olmo2_7b` | `keep12, keep10, keep8` | 3 | **NO** — 04_experiments.tex:8 designates 5 |
| `llama2_7b` | `k14, k12, k10, k8` | 4 | yes |
| `llama3_8b` | `k14, k12, k10, k8` | 4 | yes |
| `qwen3_8b_base` | `k14, k12, k10, k8` | 4 | yes |

3 + 12 = the quoted 15. The three non-OLMo families are internally consistent and complete.
**The defect is OLMo-2-only — but it is not a simple omission, it is an inconsistency**, because
`k14` (retained depth 14) is counted three times while `keep14` (retained depth 14) is not.
This is NEW-2 above and it is what settles the remedy.

## 3. Decision: option (a), full inclusion. Option (b) is not available.

MAIN leaned to (b) — named exclusion with a stated rule. **I am choosing (a), and the reason is
not aesthetic.** Option (b) requires an inclusion rule, and every candidate rule fails:

- *"ShortGPT-16 is a baseline pruning operator, not a rung of our ladder"* — plausible for
  `shortgpt16`, but says nothing about `keep14`, which **is** a rung of the ladder and is
  excluded anyway. A rule that covers only one of the two omissions is not a rule.
- *"depth-14 retains too much of the stack to count as damaged"* — **contradicted by the paper's
  own denominator**, in which depth-14 IS damaged for three families (NEW-2). Adopting it would
  require dropping all three `k14` arms, which changes 14/15 to 12/12 and 0/60 to 0/45. That is
  a *stronger* headline, which is precisely why it must not be chosen post hoc: it is selection
  on outcome in the opposite direction, and `04_experiments.tex:23` forbids it.
- *"the OLMo-2 damaged set is whatever #248 used"* — provenance, not a scientific criterion, and
  `09a_relocated.tex:24` requires a criterion.

Full inclusion needs no rule, satisfies the App A.4 promise as written, and is verifiable by a
gate. **The cost is real and I am not minimising it**: `14/15 → 15/17` on MMLU-Pro and
`0/15 → 9/85` off MMLU, and the near-unanimity that `09a_relocated.tex:26` offers *in place of*
a family-wise correction is genuinely weaker than before. I have said so explicitly in the
Multiplicity paragraph rather than letting the depth threshold paper over it.

## 4. The monotonicity — a real positive result, with the overclaim I refused to make

Off-MMLU, all 85 designated cells, sorted by retained depth:

| retained | regime | n | above floor | min Δ | max Δ |
|---|---|---|---|---|---|
| 16 | prune-then-heal | 5 | **5** | +11.208 | +49.663 |
| 14 | prune-then-heal | 5 | **4** | +2.612 | +18.687 |
| 14 | truncate-only | 15 | 0 | −3.925 | +0.491 |
| 12 | prune-then-heal | 5 | 0 | −2.200 | +0.246 |
| 12 | truncate-only | 15 | 0 | −4.400 | +0.983 |
| 10 | prune-then-heal | 5 | 0 | −5.000 | +1.621 |
| 10 | truncate-only | 15 | 0 | −7.400 | +0.000 |
| 8 | prune-then-heal | 5 | 0 | −1.800 | +2.503 |
| 8 | truncate-only | 15 | 0 | −4.778 | +0.054 |

- **Depth ≥14: 9/25 above floor (all nine OLMo-2; 0/15 for the three truncate-only families at
  the same depth). Depth ≤12, any family: 0/60 above floor.**
  (An earlier draft of this document said "0/35" here; that was my arithmetic slip — the correct
  count is 0/60, i.e. 15 OLMo-2 cells at depth ≤12 plus 45 non-OLMo cells at depth ≤12.)
- **The two groups do not overlap.** Smallest above-floor margin **+6.470 pp**; largest
  point estimate among the 76 at-or-below cells **+2.612 pp**. So the threshold is not an
  artefact of borderline cells, which is the objection I expected and checked for.
- Spearman ρ(retained depth, floor delta) over the 25 OLMo-2 cells = **+0.8158**.

**Three things I refused to claim, and why:**
1. **Not per-benchmark monotone.** I tested it: `strictly_decreasing = False` on all five
   benchmarks (e.g. arc_challenge +31.14, +7.00, −0.43, **+1.62**, −0.94 — keep10 above keep12).
   The monotonicity is a **threshold at depth 14**, not a gradient, and the paper's own
   Discussion already says "a cliff rather than a gradient" and forbids fitting a depth curve.
   I kept that wording consistent.
2. **The threshold is benchmark-dependent.** On MMLU-Pro it is depth ≥16: `keep14` is at the
   floor there (+0.324, p=0.3234) while clearing 4/5 small benchmarks. A single number would be
   an overclaim, so the main text states depth ≥14 for off-MMLU only.
3. **Regime and family are confounded** — OLMo-2 is the only prune-then-heal family and no
   truncation rung exceeds depth 14, so `0/60` is consistent with the threshold *without
   testing it*. I wrote this caveat into `09a_relocated.tex` explicitly. **The threshold is a
   property of the OLMo-2 ladder, not a family contrast.**

This is why I think disclosure is net-positive for the paper rather than merely honest: an
unexplained `0/15` invites "your floor test just fails everything." A threshold with a clean
6.47-vs-2.61 pp separation shows the test tracks capability. But it is a **weaker** substitute
for multiplicity control than an exceptionless aggregate, and I said that in the text.

## 5. Files changed

| file | change |
|---|---|
| `code/mmlu_pro_power_nulls.py` | `DAMAGED_OLMO` widened to all five arms + 10-line comment on why it must not be narrowed |
| `evidence/mmlu_scale_power/mmlu_pro_power_nulls_v2.json` | `rollup.olmo2_7b{,_naive_chance}` re-derived; `DENOMINATOR_FIX_20260816` block records the superseded counts. **Asserted: every other top-level block byte-identical.** No new measurement. |
| `evidence/designated_damaged_denominators.json` | NEW. 106 cells, both references, full ladder, monotonicity, all caveats |
| `code/recount_designated_denominators.py` | NEW. independent re-derivation |
| `code/gate_designated_denominator.py` | NEW. 5 checks + 4-case negative control |
| `code/gate_build_record_matches_pdf.py` | NEW. 3 checks + 4-case negative control |
| `gate/designated_denominator.json` | NEW. gate output, PASS |
| `sections/04_experiments.tex` | designated-set paragraph rewritten: set stated once, 17/85 given, forward-ref to the disclosure |
| `sections/05_analysis.tex` | 15 cells → 17; `shortgpt16` named with its delta and framed as a positive result; off-MMLU 9/85 + 45/85 + depth threshold |
| `sections/09a_relocated.tex` | App A.4 promise now *kept*: full set enumerated, prior violation named with both deltas, gate cited; Multiplicity states the substitute is weaker; A.6 gives all nine cells + the non-overlap + the confound caveat |
| `sections/06_discussion.tex` | 14/15 → 15/17 + one clause on why the exceptions are informative |
| `sections/tab_mmlupro.tex` | **two rows added** (`shortgpt16`, `keep14`); caption states all 17 |

Not touched (agent A / B / reviewers own them): `00_abstract.tex`, `01_introduction.tex`,
`03b_nulls.tex`, `09_appendix.tex`, `tab_nulls.tex`, `tab_construct_nulls.tex`,
`gate/build_record.json`, `code/freeze_round.py`, `review_rounds/**`.

## 6. ★ EXACT TEXT FOR MAIN TO MERGE INTO AGENT A's FILES

### 6a. `sections/00_abstract.tex` — two edits

REPLACE:
> Among 15 designated damaged cells in four model families, the null choice alone reverses the reading: damaged arms appear above a chance line while only one clears its best-constant floor, and the honest aggregate is 14/15 at or below the floor.

WITH:
> Among 17 designated damaged cells in four model families, the null choice alone reverses the reading: damaged arms appear above a chance line while only two clear their best-constant floor, and the honest aggregate is 15/17 at or below the floor.

REPLACE:
> Across five smaller benchmarks, 10/15 designated cells show the same wrong-null flip, while a mandatory power analysis explains why most per-benchmark significance tests are inconclusive.

WITH:
> Across five smaller benchmarks the same wrong-null flip appears in 45/85 designated cells while only 9/85 clear their floor, and every one of those nine retains at least 14 of 32 layers; a mandatory power analysis explains why most per-benchmark significance tests are inconclusive.

> Note for MAIN: "all 21 evaluated cells are powered" in the same abstract is CORRECT and
> verified (`power_verdict.n_cells=21, n_cells_powered=21`). Leave it. But see NEW-3: 21 − 4
> intact bases = 17, which is now the verdict denominator too, so the two numbers are finally
> consistent. Before this fix they were not, and that inconsistency was itself a symptom.

### 6b. `sections/01_introduction.tex:21` — replace the whole item

> \item At MMLU-scale power, 17 designated damaged cells across four families exhibit wrong-null readings: 10/12 non-OLMo cells exceed item-averaged chance (12/12 exceed naive 0.10), yet only 1/12 clears the floor; four of the five designated OLMo-2 cells exceed either chance line but not the floor. The honest statement is 15/17 at or below the arm-independent floor, with one statistically real but materially negligible exception and one arm, \texttt{shortgpt16}, that clears the floor by $+3.674$ points and retains more of the stack than any other damaged arm. Off MMLU, 45/85 designated cells flip and 9/85 clear their floor, all nine retaining at least 14 of 32 layers; the accompanying power table is essential.

### 6c. `sections/03b_nulls.tex:12` — the `4.6x` errata (CONFIRMED)

Independently recomputed: `0.532164 / 0.125914 = 4.2264`. **`4.6` is wrong.** Its likely origin
is `credit / letter_floor = 0.532164 / 0.116606 = 4.5638`, which rounds to 4.6. Note the other
near neighbour, `credit / item_avg_chance = 0.532164 / 0.110877 = 4.7996` — that is the
**"4.8×" family that row 16 of the paper's own ledger explicitly prohibits resurrecting**
(`tab_claims.tex:24`), so the fix must be `4.2264`, not any re-derivation against a chance line.
Note `evidence/POWER_WALL_VERDICT.md:265` *also* says "4.6×", so the error is upstream in the
evidence file, not a typo in the .tex — flag it there too.

The other two numbers in the same sentence are **CORRECT**: span `(0.532164−0.125914)×100 =
40.6250` → "40.6-point"; `(0.532164−0.207613)×100 = 32.4551` → "32.5 points". Also verified
`credit/base_norm = 2.5632` → the "factor of 2.6" in POWER_WALL_VERDICT is right.

REPLACE:
> The \texttt{credit} value is 4.6$\times$ the \texttt{wrong} value

WITH:
> The \texttt{credit} value is 4.23$\times$ the \texttt{wrong} value

### 6d. `sections/tab_nulls.tex` — ALREADY FIXED BY AGENT A, no action

MAIN's brief says line 34 reads `+0.43--+2.60`. **It does not any more.** Current line 44 reads
`the $+0.49$--$+2.60$~pp gaps`, and the five row values are `+0.490 / +0.885 / +1.520 / +1.625 /
+2.600` — so `+0.49` is correct and A already applied it. I grepped `0\.43` across all of
`sections/*.tex`: **zero hits.** Nothing to do. (The nearest live value is Winogrande's
`floor_minus_chance_pp = 0.4341`, which is the negative control and appears in no table row.)

### 6e. `gate/build_record.json` — do NOT hand-edit; run the gate

I read its current state: A has **not** refreshed it. It still certifies `22 pages / 355196 B /
sha 56a376e1…` while `main.pdf` is `25 pages / 375507 B` (and 26/9-body after my rebuild). Since
it is A's file, I did not touch it. I instead wrote the **assertion** MAIN asked for:
`code/gate_build_record_matches_pdf.py`, which fails non-zero on any of bytes/sha256/pages
disagreeing with the PDF beside it. Whoever regenerates `build_record.json` should then run it.
Its live output and negative control are in §8.

## 7. Downstream sites — every one I found, and its disposition

| site | old | new | done by |
|---|---|---|---|
| `04_experiments.tex:23` | `14/15`,`10/15`,`0/60` as multiplicity substitute | set stated once; 17/85; depth threshold | me |
| `05_analysis.tex:7` | "three designated OLMo-2 cells", "these 15 cells" | "four of the five", "these 17 cells" | me |
| `05_analysis.tex:27` | "Fourteen of the 15"; "a real but immaterial exception" | "Fifteen of the 17"; both exceptions named, `shortgpt16` given its own delta | me |
| `05_analysis.tex:29` | `10/15` + `0/15` + `0/60` | `9/85` + `45/85` + depth threshold; `7/60`/`52/60` kept (verified) | me |
| `09a_relocated.tex:24` | the App A.4 promise, violated | promise kept; prior violation disclosed with both deltas | me |
| `09a_relocated.tex:26` | `0/60` offered in place of multiplicity control | restated on the real aggregate; admits it is a weaker substitute | me |
| `09a_relocated.tex:44` | `10/15`,`0/15`,`0/60` | `45/85`,`9/85`, all nine enumerated, non-overlap, confound caveat | me |
| `06_discussion.tex:15` | `14/15` | `15/17` + informative-exceptions clause | me |
| `tab_mmlupro.tex` | 12 rows, 15 cells implied | 14 rows, caption says 17 | me |
| `00_abstract.tex` | `14/15`, `10/15` | §6a | **MAIN** |
| `01_introduction.tex:21` | `14/15`, `10/15`, `0/15` | §6b | **MAIN** |
| `03b_nulls.tex:12` | `4.6x` | §6c | **MAIN** |
| `03b_nulls.tex:35` | `14/15`,`10/15`,`3/12`,`1/12` | needs `15/17` for the `14/15`; the 3/12 and 1/12 are non-OLMo-only and **unaffected** | **MAIN** |

**Sites I checked and deliberately did NOT change:**
- `3/12` / `1/12` / `10/12` / `12/12` / `0/12` (05_analysis 17,18,23,25; 09a:37; 01:21;
  03b_nulls:18) — all explicitly **"damaged non-OLMo cells"**, denominator 12 = 3 families × 4
  rungs. Verified against `s2_03_symmetric_inference.json:counts_12_damaged_non_olmo` (n=12) and
  its 15-cell list (12 damaged + 3 intact). **Unaffected by an OLMo-2 exclusion.**
- `0/27` (07_limitations:13, 09_appendix:33) and `tab_v2_full` — the 27-cell v2 set **already
  contains both arms** (`tab_v2_full.tex:21,22`). No hidden exclusion. `keep14` is "trace signal"
  and `shortgpt16` "item-level signal" there, consistent with my depth reading.
- `0/24`, `0/30`, `0/14`, `5/8` (integrity table) — bootstrap/truncation/OOM audit denominators,
  unrelated to arm designation.
- `1/10` (03b_nulls:28, 03b_nulls_summary:4) — construct counts, not arms.
- `tab_power` — keep8-only power table; correct as-is.

## 8. Gate negative-control output (measured, not asserted)

```
$ python3 code/gate_designated_denominator.py          # BEFORE my evidence fix
  CHECK 1 FAIL: MMLU-Pro rollup omits declared damaged OLMo-2 rungs ['keep14', 'shortgpt16'].
  CHECK 3 FAIL: retained depth 14 is counted as designated damaged for ['llama2_7b',
                'llama3_8b', 'qwen3_8b_base'] but family 'olmo2_7b' has arm(s) ['keep14'] at
                the same retained depth and does NOT count them.
  GATE designated_denominator: FAIL (3 checks passed, 2 failed)      RC=2

$ python3 code/gate_designated_denominator.py          # AFTER
  CHECK 1..5 pass
  GATE designated_denominator: PASS (5 checks passed, 0 failed)      RC=0

$ python3 code/gate_designated_denominator.py --selftest_negative_control
  NC-1 drop keep12 from MMLU-Pro rollup     -> CAUGHT
  NC-2 drop keep10 from off-MMLU rollup     -> CAUGHT
  NC-3 drop k14 from one family/bench cell  -> CAUGHT   (rectangularity)
  NC-4 positive control (unmutated)         -> 0 spurious failures OK
  NEGATIVE CONTROL PASSED                                            RC=0

$ python3 code/gate_build_record_matches_pdf.py
  pdf_bytes  record=355196  actual=376227  MISMATCH
  pdf_sha256 record=56a376e1…  actual=77cd460b…  MISMATCH
  pdf_pages  record=22  actual=25  MISMATCH
  GATE build_record_matches_pdf: FAIL                                RC=2

$ python3 code/gate_build_record_matches_pdf.py --selftest_negative_control
  NC perturb pdf_bytes  -> CAUGHT
  NC perturb pdf_sha256 -> CAUGHT
  NC perturb pdf_pages  -> CAUGHT
  NEGATIVE CONTROL PASSED                                            RC=0
```

The gate is written to fail if the *declaration sentence* in `04_experiments.tex` is reworded
out of recognition, rather than silently passing — a `SystemExit` with an explicit message. It
does **not** hardcode the expected arm list; it parses whatever the .tex declares. Hardcoding
would reproduce the defect being fixed.

## 9. PDF page count (measured with PyMuPDF, by me)

Scratch build at `/tmp/pc5` (`.texlive/2026`, `latexmk -pdf -bibtex -norc -gg`, rc=0,
0 overfull hbox/vbox, 0 undefined refs/cites):

- **Body = 9 pages.** Verified structurally, not by eyeball: last body block ends p9
  y=732.3 ("Regime, pooling, and literature scope"), and page 10 contains **only**
  `REFERENCES` (y=82.8) plus bibliography entries. **Within the 9-page ICLR limit.**
- Total 26 pages (was 25 before my edits; the extra page is in the **appendix**, from the two
  new `tab_mmlupro` rows and the expanded App A.4/A.6 — appendix pages are unlimited).
- Route: my first pass pushed body text onto p10 (REFERENCES at y=270.7, ≈190 pt over). I
  measured the deficit against an A-only baseline build (`/tmp/pcbase`, REFERENCES p9 y=640.1),
  then compressed **only my own additions** in three rounds until body text cleared p9.
  Net main-text growth after compression: `04` +122 B, `05` +515 B, `06` +176 B. The substantive
  disclosure lives in the appendix, which is the pattern `09a_relocated.tex` was created for.

## 10. Writer scripts — clean-tree behaviour confirmed

Backed up both outputs first (`/tmp/bk_prose.json`, `/tmp/bk_static.json`).

- `check_prose_vs_evidence.py` → `n_checked=91 n_ok=91 n_mismatch=0 n_skipped=24
  n_uncovered=0`, verdict PASS, **rc=0**. (Baseline was 86/86/0; `n_checked` rose by 5 because
  I added numbers, and **`n_mismatch` stayed 0**, which was the requirement.)
- `validate_tex_static.py` → all files OK, **rc=0**.
- `mmlu_pro_power_nulls_v2.json`: I asserted programmatically that **every top-level block other
  than `rollup` is byte-identical** to the pre-edit backup. No per-cell number moved.

## 11. Mandated verbatim diff — the census-invisible sentence classes

Diffed every sentence **containing no digits** (the class a numeric census cannot see) across all
four files I touched. Exactly three came up, all in modified-but-present form, none lost:

1. `04`: *"Every denominator below is over designated damaged cells: arms structurally pruned and
   reported as damaged rungs of their family."* → the **definition survives** and is now
   *stronger* (it enumerates the set instead of describing it). Contribution-scoping intact.
2. `04`: *"Appendix~\ref{app:designated} states both."* → the cross-reference survives (2 hits to
   `app:designated` in the file). Pointer preserved.
3. `05`: *"The result is strong but not universal."* → now *"The result is strong but not
   universal, and the two exceptions differ in kind."* **This is the epistemic-limitation
   sentence and it survives verbatim as a prefix**, extended rather than weakened.

No falsification-condition or contribution-boundary sentence was deleted. I also confirmed the
Discussion's existing *"a cliff rather than a gradient; no depth curve should be fitted to the
sparse rungs"* is untouched — my depth-threshold wording was written to be consistent with it,
not to quietly supersede it.

## 12. ⚠️ WHAT I COULD NOT VERIFY (explicit, not empty)

1. **Whether `main.pdf` in the repo is 9-page body.** I built in `/tmp/pc5` and did **not**
   overwrite `paperC/main.pdf`, because agent A rebuilt it at 14:01 and is still working. My
   9-page result is for the current `sections/` tree; MAIN must rebuild after merging §6a–6c,
   since abstract/intro edits change line counts and body pages are a **7-byte step function**
   at this margin. **Do not assume my page count survives the merge — re-measure.**
2. **The `4.6x` upstream.** I fixed the arithmetic and traced the likely origin (`4.5638` from
   `credit/letter_floor`), but `evidence/POWER_WALL_VERDICT.md:265` also says "4.6×". I did not
   edit that file (out of scope) and could not determine whether other .md/.json carry it.
   `grep` for `4\.6` in `sections/` found only the one .tex site.
3. **Whether `#248` had a defensible reason** for `{keep8,keep10,keep12}`. I found no criterion
   in any evidence file and the source comment says only "#248's damaged set, verbatim". I did
   not read the #248 launch records. If such a reason exists, my §3 argument that no rule is
   available would need revisiting — but it would still fail the `k14` parity test (NEW-2).
4. **The v1 `+4.054` permutation delta for `shortgpt16`** in `tab_v2_full.tex:22` — I read it but
   did not re-derive it from per-item records; it comes from a different estimator
   (`heal_readout_v2_permutation_null`) than the bootstrap floor deltas I recomputed.
5. **Whether restoring the arms changes `s2_03`'s 3/12 vs 1/12.** I argued it does not
   (that comparison is explicitly non-OLMo, n=12, verified in the JSON) but did **not** re-run
   the paired bootstrap with OLMo-2 arms added. If MAIN wants a 5-arm symmetric-inference
   column, that is a new computation I did not do.
6. **Reviewer-facing consistency of the other round_04 findings.** I did not read
   `review_rounds/round_04/raw/` (reviewer territory), so I cannot say whether a reviewer
   raised NEW-2 or NEW-3 independently.
7. **`0/60`'s `7/60` and `52/60` under the restored set.** These are non-OLMo-only counts and I
   verified them unchanged, but I did not compute the analogous "significantly below" and
   "underpowered" counts over all 85 cells. The main text therefore still quotes them scoped to
   the non-OLMo replication, which is accurate but means the 85-cell aggregate lacks those two
   companion statistics.
