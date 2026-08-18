# k-span (multi-region) code infilling: diffusion vs AR-FIM

**Status: IN PROGRESS.** This file is written incrementally. The pre-registered
decision rule in §3 was committed BEFORE the decontamination leg was run; see
the git history of this file for proof of ordering.

Node/env provenance:
- diffusion arm: `.252` 8×B200, wzc1 disk, `.venv_b200/bin/python` (py3.11, torch 2.11, transformers 4.51.3)
- AR arms: `.73` 8×H20, zwfy6 disk, `.venv_dream/bin/python` (py3.11, torch 2.5.1, transformers 4.46.2)
- **ALL grading on wzc1 with evalplus 0.3.1.** `.73` carries evalplus
  0.1.0.dev1; mixing the two would make the arms incomparable, so AR solutions
  were copied wzc1-ward (`scp -O`, `md5sum` MATCH on both arms) and graded by
  the same grader binary as diffusion.

Frozen spec (`data/` is gitignored, so the contract is the hash + the
deterministic builder; a rebuild from scratch reproduces both files bit-identically):

```
data/kspan/kspan_spec_v1.jsonl     sha256 1cc12a50d1f4255f...  415 rows  md5 1638ce4068bd8704...
data/kspan/topology_spec_v1.jsonl  sha256 da66a3a1f7cdcb30...  165 rows  md5 2d719f88647e6856...
```

Both hashes verified byte-identical on the zwfy6 disk before the AR arms ran, so
all three arms consumed the same holes.

---

## 0. Two defects found in the hole-construction recipe I was handed

Both were found by re-deriving the spec instead of trusting it, and both are
recorded here because they change the numbers.

### Defect A — `prompt + canonical_solution + suffix` is NOT consistent within a task

The recipe says: reconstruct each file as `row['prompt'] + row['canonical_solution']
+ row['suffix']`, "verified parseable 164/164". Parseable, yes — but **not unique**.

For **123 of 164** base tasks the rows of that same task reconstruct **two
different files**, differing by exactly one blank line: the `L0` row's `prompt`
carries a trailing `'\n'` the other rows' do not. So "the" reconstructed file is
ambiguous, and any hole-line index computed against the wrong variant is off by
one — silently, because the file still parses.

`row['prompt'].count('\n')` (the recipe's rule) is correct **only relative to
that row's own variant**. Verified: it locates the gold line exactly 1033/1033
times *within the row's own reconstruction*. It is wrong for 123 rows once you
fix a single reference file per task, which you must do to punch k holes in one
file.

**Fix.** Never infer the hole line from an index. Admit a row only if it is
*provably* the "blank out line i of reference file F" task, byte-exactly:

```
F_lines[i] == row['canonical_solution']
and ''.join(F_lines[:i]) == row['prompt']
and ''.join(F_lines[i+1:]) == row['suffix']
```

Reference file F = majority vote over the task's own rows, deterministic
tie-break (shortest, then lexicographic). Result: **910/1033 rows admitted,
123 dropped** (they belong to the other blank-line variant). Zero guessed
indices. This is the only reason the caps below are reproducible at all.

### Defect B — the k=4 cap is 59, not 60

Reported caps were k=1 164, k=2 108, k=3 84, k=4 **60**. I reproduce
**164 / 108 / 84 / 59**. Three of four match exactly; k=4 is off by one.

The cap depends jointly on (i) whether the 123 blank-line-variant rows are
recovered or dropped, and (ii) which non-adjacent subset selector is used. Grid:

| row admission | selector | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|---|
| byte-exact only | greedy MIS | 164 | 108 | 84 | **59** |
| byte-exact only | farthest-point | 164 | 100 | 84 | 53 |
| + blank-line shift recovery | greedy MIS | 164 | 117 | 100 | 70 |
| + blank-line shift recovery | farthest-point | 164 | 108 | 99 | 60 |
| **reported** | — | 164 | 108 | 84 | 60 |

No single consistent spec reproduces the reported row `[164, 108, 84, 60]`: the
reported k=2/k=3 match *byte-exact + greedy MIS*, while the reported k=4 matches
*shift-recovery + farthest-point*. I therefore treat the reported k=4 n=60 as
not reproducible and use **n=59**, which is internally consistent with the other
three cells. This is a 1-task discrepancy with no bearing on any conclusion, but
it is logged rather than smoothed over.

### The docstring bug I was warned about: verified absent

Assert (a) is live in the builder, and independently: **0 of 1033** admitted
holes land inside a docstring or bare string expression (checked via `ast`,
covering `lineno..end_lineno` of every string-constant `Expr`). Blank lines are
also refused. The five asserts (a)-(e) all run on every build.

---

## 1. Null controls on the frozen spec (n = 164/108/84/59)

Run first, before any model arm, because a dead pipeline looks exactly like a
working one until the nulls are inspected.

| control | k=1 | k=2 | k=3 | k=4 | verdict |
|---|---|---|---|---|---|
| `null_gold` (refill gold) | **.994** | **.991** | **.988** | **.983** | PASS — spec is well-posed |
| `null_delete` (drop hole lines) | .024 | .000 | .000 | .000 | PASS — floor is at zero |
| `null_mutate` (mutate gold) | .457 | .287 | .143 | .119 | PASS — degrades with k |
| `null_mutate`, mutable-only | **.043** | **.025** | **.014** | **.000** | PASS |
| `null_delete` parseable rate | .744 | .324 | .286 | .220 | — |

`null_gold` at .983-.994 (not 1.000) is EvalPlus flakiness on a handful of
timing-sensitive tasks, not a spec defect: the program is byte-identical to the
canonical file.

**`null_mutate` needs the extra row, and this is a real trap.** My mutation is
deterministic (flip first comparison / boolean / `and`/`or` / `+`, else bump the
first int literal). Some gold lines have nothing to mutate — e.g. `return
result`. For those the "mutated" program **is the gold program** and passes by
construction, inflating the null. Unmutable-hole rate rises steeply with k
(43.3% → 73.1% → 81.0% → 89.8% of tasks have >=1 such hole), so the *inflated*
null is also the *k-dependent* one. Excluding tasks where zero holes changed
gives the true null: **.043 / .025 / .014 / .000**. Anyone reporting the .457
row as "the mutated-gold control" is reporting mostly gold.

---

## 2. Full-n ladder (frozen spec, official EvalPlus grader, all cells n as stated)

| k | n | diffusion | AR-FIM | AR-FIM-fair | delta (diff − AR-FIM) |
|---|---|---|---|---|---|
| 1 | 164 | .671 | **.866** | **.866** | −.195 |
| 2 | 108 | .796 | .870 | .852 | −.074 |
| 3 | 84 | .750 | .738 | .798 | +.012 |
| 4 | 59 | .746 | .644 | .644 | +.102 |

**The direction replicates; the magnitude does not.** Pilot reported
delta −.167 @k1 → +.358 @k4, interaction **+.525**. At full n I get
−.195 @k1 → +.102 @k4, interaction **+.297** — about 57% of the pilot estimate,
i.e. the pilot's point estimate sat roughly 2 SE above what full n supports.
Every individual cell moved: diffusion k=1 .783→.671, k=4 .883→.746; AR k=4
.525→.644.

### EM-to-gold — the confound does NOT reproduce, it reverses

| k | diffusion EM (exact) | diffusion EM (stripped) | AR-FIM EM (stripped) |
|---|---|---|---|
| 1 | .518 | .579 | .598 |
| 2 | .444 | .648 | .639 |
| 3 | .345 | .524 | .524 |
| 4 | .288 | .424 | .390 |

The pilot reported diffusion EM-to-gold **rising** .633 → .792 → .833 with k and
flagged it as "THE CONFOUND". At full n on the corrected spec, diffusion EM
**falls** with k (.518 → .288 exact; .579 → .424 stripped). The memorisation
story — "diffusion's rise with k is a fixed-canvas model recalling lines it has
seen" — therefore has no support in this data: diffusion's pass@1 goes *up*
from k=1→2 while its EM goes *down*, and at k=4 it beats AR while matching AR's
EM. I still run the decontamination leg (§3) rather than declare the issue
closed, because "EM did not rise" is weaker than "EM does not explain the slope".

---

## 3. ★ PRE-REGISTERED DECISION RULE (committed before the decontamination run)

> **If the pass@1-vs-k slope is not significant after conditioning on
> EM-to-gold (or on the decontaminated set), the "diffusion's home turf" claim
> is WITHDRAWN, and I report the k × family interaction for AR only.**

Operationalised, fixed in advance:

1. **Primary test** — logistic regression of `passed` on `k`, restricted to the
   EM-false subset (`em_all_stripped == False`), fit per arm. The claim survives
   only if the diffusion `k` coefficient is positive with p < 0.05 two-sided.
2. **Secondary test** — same on the decontaminated (renamed/paraphrased) set.
3. **Common-subset test** — the k=1..4 cells are *nested* task sets (the 59
   k=4-capable tasks are a subset of the 164 k=1 tasks), so a raw cell-to-cell
   slope confounds "more holes" with "which tasks survive to k=4". The slope
   must also hold on the n=59 tasks present at every k. If it does not, the
   ladder is a task-difficulty artefact and the claim is withdrawn regardless
   of tests 1-2.
4. If diffusion fails but AR's decline with k is significant, the reportable
   result is **"AR-FIM degrades with the number of independent regions"** — a
   statement about AR, with diffusion serving only as the control that shows the
   degradation is not intrinsic to the task.

No result below was inspected before this rule was written.

---

## 4. Verdict against the pre-registered rule

### 4.1 The k-cells are NESTED, and that is where the pilot's effect came from

Only tasks with >=4 available non-adjacent body lines can appear at k=4, and
those are the longer functions. So a cell-to-cell slope mixes "more holes" with
"which tasks survived to k=4". The size of that confound:

| | pass@1 at **k=1**, k4-capable tasks (n=59) | pass@1 at **k=1**, shorter tasks (n=105) |
|---|---|---|
| diffusion | **.898** | .543 |
| AR-FIM | **.949** | .819 |

The tasks that reach k=4 are *far easier at k=1* for both arms. Diffusion's
apparent "rise with k" (.671 → .746) is therefore mostly the ladder discarding
its own hard cases. Restricted to the 59 tasks present at **every** k, diffusion
does not rise — it **falls**: .898 → .847 → .831 → .746.

### 4.2 Gate-by-gate

| gate | test | result | verdict |
|---|---|---|---|
| 1 | diffusion `beta_k`, EM-false subset, main set | **+0.511, z=+3.66, p=2.5e-4** | PASS |
| 2 | diffusion `beta_k`, EM-false subset, decontaminated | **+0.373, z=+2.24, p=0.025** | PASS |
| **3** | **diffusion `beta_k`, tasks present at every k** | **−0.346, z=−2.14, p=0.032** | **FAIL — sign reversed** |
| 3 | same, decontaminated | **−0.490, z=−3.83, p=1.3e-4** | **FAIL — sign reversed** |

Gate 3 was pre-registered as decisive ("if it does not, the ladder is a
task-difficulty artefact and the claim is withdrawn **regardless of tests
1-2**"). It fails in both sets, with the slope significantly *negative* — the
opposite sign to the claim. Gates 1 and 2 pass only because the EM-false subset
is itself k-dependent (conditioning on EM-false at k=4 selects the tasks
diffusion got wrong, which is a collider, not a control).

### 4.3 ⇒ "Diffusion's home turf" is WITHDRAWN

Diffusion's absolute pass@1 **declines** in k on any within-task comparison. It
never gets better at multi-region infilling by having more regions.

### 4.4 What survives: the k × family INTERACTION

Per the pre-registered fallback, the reportable result is about the *relative*
degradation rate. Logistic `passed ~ k + arm + k:arm`, SE **clustered by
task_id** (each task contributes up to 4 correlated rows per arm):

| set | subset | `k:arm` interaction | z | p |
|---|---|---|---|---|
| main | full (nested) | −0.598 | −4.09 | 4.3e-5 |
| main | **common tasks (balanced)** | **−0.435** | **−2.89** | **0.0038** |
| decontaminated | full (nested) | −0.201 | −1.98 | 0.047 |
| decontaminated | **common tasks (balanced)** | **−0.208** | **−2.15** | **0.032** |

Negative `k:arm` = **AR-FIM degrades faster in k than diffusion**, and this
survives balancing, clustering, AND decontamination. Within-task slopes on the
common set:

| arm | main `beta_k` | decontaminated `beta_k` |
|---|---|---|
| diffusion | −0.346 (p=.032) | −0.490 (p=1.3e-4) |
| AR-FIM | **−0.781** (p=1.5e-5) | **−0.697** (p=2.8e-7) |
| AR-FIM-fair | −0.829 (p=1.6e-5) | −0.680 (p=5.7e-7) |

**Both families get worse with more regions. AR gets worse about twice as fast.**
That is a much weaker claim than "diffusion's home turf", and it is the only one
the design supports.

The naive 4-cell interaction is **+0.297 (SE .096, z=3.10, p=0.0019)** vs the
pilot's **+0.525 (SE .109)** — direction replicates, magnitude ~57%. But the
naive statistic is inflated by the nesting; the balanced clustered estimate is
the honest one.

### 4.5 Decontamination held the surface, and it cost both arms

Renaming every local identifier + replacing docstrings with `"""Solve the
task."""` (408/415 rows retained; gold-refill gate **1.000 at every k**) drops
both arms hard:

| | diffusion | AR-FIM |
|---|---|---|
| k=1 | .671 → **.544** (−.127) | .866 → **.606** (−.260) |
| k=4 | .746 → **.466** (−.280) | .644 → **.379** (−.265) |

Roughly 26 points of AR's k=1 score and 28 points of diffusion's k=4 score were
attributable to the memorisable surface form. Any absolute infilling number on
untransformed HumanEval should be read with that in mind. Notably the
interaction *shrinks* on the decontaminated set (−0.435 → −0.208), so part of
the family difference on raw HumanEval was also surface recall.

---

## 5. EM-to-gold, per (k, arm) — the reviewer's one-line check

| k | diffusion pass@1 | diffusion EM | AR-FIM pass@1 | AR-FIM EM |
|---|---|---|---|---|
| 1 | .671 | .579 | .866 | .598 |
| 2 | .796 | .648 | .870 | .639 |
| 3 | .750 | .524 | .738 | .524 |
| 4 | .746 | .424 | .644 | .390 |

(EM = whitespace-stripped exact match of ALL k holes to gold.)

The pilot's reported confound — diffusion EM **rising** .633/.792/.833 with k —
**does not reproduce**. On the corrected spec at full n, diffusion EM **falls**
(.579 → .424), and the two families' EM curves are nearly identical at every k,
so EM cannot explain a family difference. The memorisation reading of the
original ladder is therefore unsupported; the ladder's problem was nesting, not
recall. Conditioning on EM (gates 1/2) is reported for completeness but is a
collider and should not be treated as the primary test.

---

## 6. Topology-vs-length control (total masked LINES held at 4)

Rerun on corrected holes; n=55 at every cell, the same 55 tasks throughout
(`data/kspan/topology_spec_v1.jsonl`). One 4-line hole / two 2-line / four
1-line. The buggy docstring version of this control returned a flat .975.

| k (regions × lines) | diffusion | AR-FIM | AR-FIM-fair | diffusion tokens_fed | AR tokens_fed |
|---|---|---|---|---|---|
| 1 × 4 lines | .727 | .618 | .600 | 9,306 | 1,023 |
| 2 × 2 lines | .855 | .691 | .709 | 8,883 | 1,024 |
| 4 × 1 line | .800 | .545 | .600 | 8,693 | 1,004 |

Masked-token budget is essentially constant across the row (diffusion
tokens_fed varies only 8.7k-9.3k, i.e. ±3%), so this **does** isolate topology
from length. Findings:

- **Not flat** — the buggy version's .975/.975/.975 was an artefact. Corrected,
  both families move with topology at fixed total length.
- **Fragmentation hurts AR and not diffusion**: AR .618 → .545 from one 4-line
  hole to four 1-line holes, while diffusion goes .727 → .800.
- But the interaction here is **+0.145, SE .124, z=1.18, p=0.24 — NOT
  significant** at n=55. Per-arm slopes are also non-significant
  (diffusion p=.50, AR p=.31).

**So the topology control is directionally consistent with §4.4 but
underpowered.** It does *not* independently establish the effect. It does
establish that the effect is not purely "more masked tokens", since token count
is held fixed here. I am not claiming significance from this leg.

---

## 7. Cost axes

Means over **ALL** tasks including truncated/aborted (never conditioned on
successful termination — that was Retraction 3). `tokens_fed` and
`attended_context_sum` are the two comparable axes; `forward_passes` (NFE) is
reported but is NOT comparable across families.

| k | diffusion tokens_fed | diffusion attended | AR-FIM tokens_fed | AR-FIM attended | AR fwd passes |
|---|---|---|---|---|---|
| 1 | 2,026 | 2,026 | **192** | 2,350 | 12.9 |
| 2 | 3,487 | 3,487 | **434** | 3,963 | 18.9 |
| 3 | 5,366 | 5,366 | **676** | 6,370 | 29.3 |
| 4 | 8,229 | 8,229 | **979** | 11,563 | 48.4 |

- On `tokens_fed`, AR is **8.4×-10.6× cheaper** at every k (KV cache: it re-feeds
  one token per decode step; diffusion re-feeds the whole canvas every step).
- On `attended_context_sum` the two families **cross**: AR is slightly cheaper at
  k=1-2 and becomes **1.41× more expensive** at k=4, because k sequential FIM
  calls each re-attend the whole file. This crossing is the only place a genuine
  efficiency argument for the diffusion canvas exists in this data, and it is
  modest.
- Diffusion cost is exactly `steps × (canvas_len + 1)` in all 415 tasks.

**Instrumentation cross-check.** Both families were re-derived against the
closed-form AR-with-KV-cache prediction in `forward_cost.analytic_cost`:

```
AR:         415/415 tasks agree EXACTLY on all three axes
            (forward_passes, tokens_fed, attended_context_sum)
diffusion:  415/415 tasks satisfy tokens_fed == steps*(canvas_len+1)
```

Getting to 415/415 required modelling two real details rather than tolerating a
mismatch: (i) `diffusion_generate` with `max_new_tokens=1` appends one extra mask
to the canvas, so the width is `canvas_len+1`; (ii) HF `generate` spends
`gen_tokens+1` forwards on a natural EOS stop but exactly `max_new_tokens` on a
budget stop. An earlier version of this check "agreed 389/415" — that 26-task
residual was the budget-stop cases, i.e. exactly the expensive tail
Retraction 3 was about.

### Termination / abort (disclosed separately from grading failures)

Truncation is rare and cannot drive any conclusion:

| arm | set | truncated tasks | aborted | generation errors |
|---|---|---|---|---|
| diffusion | main, all k | **0** | 0 | 0 |
| AR-FIM | main | 1 (k=2) | 0 | 0 |
| AR-FIM-fair | main | 0 | 0 | 0 |
| diffusion | decontaminated | 0 | 0 | 0 |
| AR-FIM | decontaminated | 3 (k=2,3,4: one each) | 0 | 0 |
| AR-FIM-fair | decontaminated | 2 (k=3,4) | 0 | 0 |

Shard completeness was asserted for every run (415 or 408 unique spec_ids, 0
duplicates); no run was scored from a partial merge.

---

## 8. AR-FIM-fair: the fairness repair does not rescue AR

Keeping later holes as `pass  # TODO` (valid suffix) instead of deleting them
helps only in the middle of the ladder and not at the end:

| k | AR-FIM | AR-FIM-fair | delta |
|---|---|---|---|
| 1 | .866 | .866 | 0 (identical by construction at k=1) |
| 2 | .870 | .852 | −.018 |
| 3 | .738 | .798 | +.060 |
| 4 | .644 | .644 | **0** |

At k=4 the two are identical, and the fair arm's within-task slope is if
anything steeper (−0.829 vs −0.781). So AR's decline is **not** an artefact of
our suffix construction — consistent with the pilot's conclusion on this point.
Both arms are reported everywhere above.

---

## 9. Claims: survived vs withdrawn

**SURVIVED**

1. **AR-FIM degrades with the number of independent regions roughly twice as
   fast as masked diffusion**, at matched holes. Balanced within-task,
   task-clustered: `k:arm` = −0.435 (p=.0038) raw, −0.208 (p=.032)
   decontaminated. Robust to the fairness repair.
2. **The effect is not merely "more masked tokens."** At total masked lines
   fixed at 4, fragmenting into 4 regions hurts AR (.618→.545) and not diffusion
   (.727→.800) — directionally consistent, though underpowered (p=0.24).
3. **On `tokens_fed` AR is ~10× cheaper at every k; on `attended_context_sum`
   the families cross**, AR becoming 1.41× more expensive by k=4.
4. **HumanEval infilling scores are substantially surface-memorisation.**
   Identifier renaming + docstring removal costs diffusion .671→.544 at k=1 and
   AR .866→.606, on a set whose gold refill still scores a perfect 1.000.

**WITHDRAWN**

1. ❌ **"Diffusion's home turf" / "diffusion improves as k grows."** Gate 3
   fails in both sets with reversed sign; diffusion's within-task slope is
   −0.346 (raw) / −0.490 (decontaminated). The apparent rise was the nested
   cell design discarding hard tasks.
2. ❌ **Interaction magnitude +0.525.** Replicates in direction only; naive
   estimate at full n is +0.297, and the design-appropriate balanced/clustered
   estimate is smaller still.
3. ❌ **"Diffusion EM-to-gold rises with k" as the confound.** Does not
   reproduce; EM falls (.579→.424) and is near-identical across families.
4. ❌ **k=4 n=60.** Not reproducible under any single consistent spec; n=59.
5. ❌ **The raw mutated-gold null (.800/.433/.200 pilot; .457/.287/.143/.119
   mine).** Both are inflated by unmutable gold lines. The true null is
   .043/.025/.014/.000.

**Bottom line.** There is a real, decontamination-robust k × family interaction,
but it is a statement about **AR's fragility**, not about diffusion's strength —
diffusion also degrades, just more slowly. The headline the pilot pointed at
does not survive its own nesting.

