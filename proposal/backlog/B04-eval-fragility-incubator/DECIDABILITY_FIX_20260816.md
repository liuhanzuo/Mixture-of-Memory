# B04 — decidability fix for clause 5 (φ). 2026-08-16, 0 GPU, PRE-DATA

> **Headline, stated first because it is the load-bearing sentence.**
> The span defect the decidability lens found is **real, and revision 2 did not fix it — it
> inverted it.** But that is the smaller of the two problems. The larger one, which no lens
> and no revision has stated, is that **φ cannot be computed at all**: 4 of the 5 read-out
> arms φ names have **no evaluation on either physical disk**, so `max(y)−min(y)` and
> `OLS slope(y)` are both undefined. The gate is not biased. It **returns nothing**.
>
> Its verdict today is `READOUT_ABSENT`, and the honest sentence is: **with the data now on
> disk this gate cannot fire, in either direction.** Making it fire costs 1.61 GPU-h.

**Authority for this document.** `git 45980ad` (2026-08-14 23:21 +0800) records the adversarial
pass: 6/6 lenses `NEEDS_REVISION`, two `decidable=False`, B04's being *"φ uses rescale span
116500 while the read-out's own span is 175000, so the decision statistic is not the measured
quantity."* Revision 2 (`git 9659a0f`, 23:54 the same evening) responded. This document audits
that response against the artifacts and repairs what remains broken. **`lifecycle` stays
`ready_cpu`.** Promotion is the next independent adversarial pass's call, not this document's —
that self-promotion is exactly what went wrong in revision 1.

---

## 1. The two spans, located in the artifacts

Both numbers are computed by the analyzer at runtime; neither is a literal in the data files.
I confirmed them against the code that produces them, not against STATUS.json's prose.

### 116500 — the comparator's heal-step span

| what | where |
|---|---|
| emitted as | `evidence/B04_wzc1_floor_analysis.json` → `clause5_budget_discrimination.damaged_heal_step_span` = `116500` |
| computed at | `code/analyze_b04_wzc1_floor.py:357` — `"damaged_heal_step_span": max(dstep) - min(dstep)` |
| `dstep` built at | `code/analyze_b04_wzc1_floor.py:335` — `dstep = [st for _, _, _, st in dam]`, `dam` = the 5 rows of `LADDER` (lines 100-104) with `keep is not None` |
| the 5 heal steps | `200000, 200000, 111500, 83500, 121000` — read from `LADDER`, and independently confirmed against each rung's own `summary.json.meta.ckpt_step` on disk |

Verified on disk, per rung (`olmo2_downstream_results/<dir>/summary.json` → `meta`):

| rung dir | `keep_front_layers` | `n_fresh_layers` | `num_hidden_layers` | `ckpt_step` |
|---|---:|---:|---:|---:|
| `7B_shortgpt16_step200000_wzc1` | 16 | 0 | 16 | 200000 |
| `7B_keep14_step200000_wzc1_v2` | 14 | 2 | 16 | 200000 |
| `7B_keep12_step111500_wzc1` | 12 | 2 | 14 | 111500 |
| `7B_keep10_step83500_wzc1` | 10 | 2 | 12 | 83500 |
| `7B_keep8_step121000_wzc1` | 8 | 2 | 10 | 121000 |

`max − min = 200000 − 83500 = 116500`. **116500 is the width of the interval
`I = [83500, 200000]` over which heal budget actually varies among the models B04's ρ = +1.00
is computed on.**

### 175000 — the revision-2 read-out grid's own span

| what | where |
|---|---|
| defined at | `code/analyze_b04_wzc1_floor.py:121` — `READOUT_SPAN = max(G1_READOUT_STEPS) - min(G1_READOUT_STEPS)   # 175000` |
| grid at | `code/analyze_b04_wzc1_floor.py:120` — `G1_READOUT_STEPS = [25000, 50000, 100000, 128000, 200000]` |
| emitted as | `evidence/B04_wzc1_floor_analysis.json` → `clause5_budget_discrimination.readout_span_USED_IN_PHI` = `175000` |
| enters φ at | `code/analyze_b04_wzc1_floor.py:128` (`def phi_budget(..., span=READOUT_SPAN, ...)`), used at `:147` (`slope_term = abs(ols_slope(steps, y_readout)) * span`) |

`200000 − 25000 = 175000`; `175000 / 116500 = 1.502146`.

Reproduced (`analyze_b04_wzc1_floor.py`, full run, this session, 0 GPU): `damaged_heal_step_span
116500`, `readout_span_USED_IN_PHI 175000`, `damaged_range 0.021820`, `guard 0.003244 -> OK`,
`σ̂ = 0.000541`, `ρ = +1.0000 p = 0.0028`. Everything in the evidence JSON reproduces from disk.

---

## 2. Which span is correct — and the answer is **neither**

The decidability lens was right that revision 1 was wrong, and revision 2's fix went one step
too far in the same axis. Both revisions asked *"whose span is it?"*. The right question is
*"what does the statistic have to explain away?"*.

**What φ is for.** B04 attributes margin compression to **damage depth**. But heal budget also
orders the same rungs — `Spearman(core6, heal_steps) = +0.6669` on the wzc1 ladder (+0.8721 on
zwfy6). φ must answer: *could budget alone, at fixed damage, produce as much median_margin
movement as B04 attributes to depth?* The denominator `D = 0.021820` is the depth-attributed
range **measured over models whose budgets lie in `I = [83500, 200000]`**. So the numerator must
be a budget excursion **over that same interval**. That is the invariant both revisions missed.

**Why 116500 alone is wrong (revision 1).** 116500 is `|I|`, a property of the comparator.
Revision 1 measured a slope over the read-out's points and then multiplied it by a *foreign*
interval width. The lens's wording is exact: the printed number was not the measured quantity.
It understated the read-out's own excursion by 1.502146×, in the direction favouring B04.
**Confirmed and upheld.**

**Why 175000 alone is also wrong (revision 2).** Revision 2 replaced 116500 with 175000 — the
read-out grid's own span — and that makes the slope term self-consistent. But it changes the
grid's *meaning*: `[25000, 50000, 100000, 128000, 200000]` puts **2 of its 5 points (25000,
50000) below `I`'s floor of 83500**, in 58500 steps of budget territory **no damaged rung
occupies**. Only 3 of 5 land inside `I`. So φ_W is a legitimate measurement of *something* — the
budget response over `[25000, 200000]` — but that is **not** the nuisance the denominator is
built from. Concretely, the very-early-heal region 25k–50k is where a healing model moves
fastest, so including it inflates the numerator with movement that **cannot** explain any
damaged rung's position. Revision 2 traded a numerator that under-read the nuisance for one that
**over-reads** it. Both are the same class of error: a numerator and a denominator measured over
different intervals.

**So the correct span is neither 116500 nor 175000.** It is the span of a read-out grid whose
points all lie **inside `I`**, and it must equal that grid's own hull. Two consequences:

1. This is a **legitimate finding of the kind the task anticipated**: the right span belongs to
   a grid that **no existing artifact covers**. §5 states exactly what data is needed.
2. **Neither revision's grid can simply be re-spanned.** Rescaling revision 2's grid back to
   116500 reproduces revision 1's exact defect — measured on this x-grid, it also drops the
   slope-term sup ratio from 1.173627 to **0.781300**, i.e. it would silently disable the
   max-guard that the falsifiability lens demanded. The grid must change, not just the constant.

**Best available grid, and its honest shortfall.** From the 15 `seed1234` checkpoints on wzc1
(`ls outputs/olmo2_probe2_7B_keep14fresh2_seed1234/`), 13 lie inside `I`. The 5-point grid
maximising coverage while keeping k = 5 is

```
I' = {100000, 128000, 153500, 175000, 200000},   S_I' = 100000
```

`S_I' / |I| = 0.8584`. **It does not reach `I`'s floor**: `[83500, 100000]`, 16500 steps
(14.16% of `|I|`), is uncovered because no `seed1234` checkpoint exists below 100000 other than
25000 and 50000, both of which are *outside* `I` on the other side. Disclosed, not hidden: φ on
`I'` is a **lower bound** on the budget excursion across `I`, and the shortfall is exactly the
16500 steps nearest `keep10@83500`.

**Therefore the repair reports both, and takes the worse.** `I'` is interval-matched to the
denominator but 14.16% short; `W` reaches further but 2/5 of it is off-support. Rather than pick
and be wrong in one of the two ways already documented, the repaired gate computes φ on **both**
grids and takes the **more severe verdict**. Machine-checked over 20 000 random shapes: the
combined verdict is **never** less severe than either alone (0/20000, §4 Part C). This can only
tighten the gate; it cannot let anything through that either grid would have caught.

---

## 3. The rewritten gate, verbatim

```
CLAUSE 5 (REVISION 3, 2026-08-16, PRE-DATA, 0 GPU) — budget discrimination at fixed damage

CONSTANTS (all measured, all 0 GPU, provenance in §1 and §4)
  D        = 0.021820     damaged-ladder median_margin range   [DENOMINATOR]
                          evidence/B04_wzc1_floor_analysis.json
                          -> clause5_budget_discrimination.damaged_range_median_margin
  sigma_hat= 0.000541     seed-pair floor, k=2, divisor E[range 2]/sigma = 1.1284
                          -> per_metric_floor_analysis.median_margin.sigma_hat
  I        = [83500, 200000]   the comparator's heal-step interval, |I| = 116500
                          (DESCRIPTIVE: |I| MUST NOT be multiplied by any slope. That is
                           the revision-1 defect. |I| is used ONLY to test grid support.)
  PHI_KILL = 0.60   PHI_PASS = 0.30      unchanged from revision 2

READ-OUT: the arm olmo2_probe2_7B_keep14fresh2_seed1234, damage held EXACTLY at
  keep_front=14, n_fresh=2, seed=1234; base ../models/OLMo-2-1124-7B; harness
  scripts/eval_olmo2_probe2_downstream.py; 8 shards; batch_size 8; --save_per_example;
  arch sm_100; one driver invocation. ONLY ckpt_step varies.

  GRID_I  = {100000, 128000, 153500, 175000, 200000}   S_I  = 100000   [PRIMARY]
            interval-matched: all 5 points inside I. Covers 0.8584 of |I|.
            Uncovered: [83500, 100000] = 16500 steps = 14.16% of |I|. DISCLOSED.
  GRID_W  = {25000, 50000, 100000, 128000, 200000}     S_W  = 175000   [SECONDARY]
            revision 2's grid, retained verbatim so revision 2 is auditable, NOT dropped.
            2/5 of its points (25000, 50000) lie BELOW I and are off-support.

STATISTIC — computed on EACH grid at ITS OWN span, then combined:
  phi_G   = max( max(y_G) - min(y_G),  |OLS slope of y_G on heal_step| * S_G ) / D
  verdict_G = KILL if phi_G >= 0.60 ; PASS if phi_G <= 0.30 ; else NARROWED
  VERDICT  = the MORE SEVERE of verdict_I and verdict_W      (KILL > NARROWED > PASS)

  Rationale for max(range, |slope|*S): the affordability lens required a shape-agnostic
  range ratio; the falsifiability lens required a max-guard against a flat-but-noisy
  response. max() IS the range ratio whenever range binds, exceeds it by at most the
  grid's sup ratio otherwise, and can never fall below it -> strictly the more
  conservative of the two statistics the two live lenses asked for. Measured sup ratios
  on the two fixed x-grids: GRID_I 1.220390, GRID_W 1.173627 (§4 Part A).
  Rationale for taking the worse of two grids: neither grid is both interval-matched and
  fully supported; choosing one would commit the revision-1 error (foreign interval) or
  the revision-2 error (off-support points). Taking the worse is monotone in severity and
  machine-verified never to be laxer than either grid alone (0/20000 random shapes).

REPORTING (mandatory, refuse to report otherwise)
  Every phi MUST be printed with: its grid, its span, its binding term (range|slope),
  D, sigma_hat, and Spearman(core6, heal_steps) for the NAMED ladder
  (+0.6669 wzc1 / +0.8721 zwfy6). A phi without its span is the artefact the
  decidability lens caught: 116500 vs 175000 vs 100000 differ by up to 1.75x, so the
  same data can print as three different numbers.

HARD ABORTS — each returns a NON-PASS verdict and blocks the downstream spend
  READOUT_ABSENT          if any named arm of a grid lacks a margin-computable eval dir
                          (all 6 core6 per_example_{task}.jsonl present) on EITHER disk.
                          phi is then UNDEFINED -- not small, not large. THIS IS THE
                          VERDICT TODAY (§4 Part D). It is a NON-PASS: an undefined ratio
                          cannot license 244-2560 GPU-h.
  PROTOCOL_VIOLATION      if a grid's evaluated step set differs in ANY way from its
                          prereg set. Adding points biases toward PASS (the unused
                          seed1234 ckpts cluster near 200000, shrinking the range term);
                          dropping one breaks k=5-vs-k=5 matching (E[range]/sigma is
                          2.0588 at k=4 vs 2.3259 at k=5, -11.5%, also toward PASS).
                          Also fires if any arm is not 8/8 shards, or pooled
                          n_scored != 17195, or any per-task n_scored != its HF
                          cardinality, or n_nan != 0.
  FIELD_ASYMMETRY         if any arm lacks norm_scores/norm_lens on all 6 tasks. An
                          asymmetric-field paired comparison already produced a 56x
                          artefact once (status/PAPERF_ACCNORM_VERIFIED.md:43-67 -- a
                          reported 34.7% flip rate whose true value was 0.62%).
  DENOMINATOR_UNRESOLVED  if D <= 0 or D < 6*sigma_hat = 0.0032435.
                          Measured: D = 0.021820 = 6.73x the guard. Currently OK.
  FLOOR_UNMEASURABLE      if sigma_hat == 0 (the normal outcome for same-driver re-runs).
                          Does NOT pass: it means the contrast is not a real nuisance
                          contrast. The seed pair is safe (varies init seed; sigma_hat != 0
                          on all 4 metrics).
  SHARD_SAMPLES_ARE_NOT_A_READOUT
                          The samples[] arrays inside shard*of8.json MUST NOT be used to
                          stand in for per_example. They are capped at 6 rows per shard per
                          task (scripts/eval_olmo2_probe2_downstream.py:476) = 48/task = 288
                          pooled, and carry no item_id. Bootstrap SD of the median at
                          n=288 is 0.008202 = 15.2*sigma_hat = 37.6% of D, so a 5-point
                          range built from them is noise-dominated by construction.

METRIC: median_margin is PRIMARY (fixed 2026-08-14 pre-data on the noise-floor argument;
  R = 68.26 with 4/5 adjacent gaps clearing 2*sigma_hat, versus R = 3.88 and 0/5 for
  frac<0.005). A phi computed on any frac metric CANNOT overturn the primary.

SCOPE: this gate tests the LIVE, NARROWED, OLMo-2-ONLY claim. The general cross-family
  claim is already dead (Qwen rho = +0.4286, p = 0.42; DIRECTION_A_QWEN_VERDICT.md sec 2)
  and this gate does not re-litigate it. All arms are OLMo-2-7B, same base, same harness,
  same arch. The Qwen f12k2/14L cell appears in §5 ONLY as an illustration that KILL is
  reachable -- NOT as a prediction for OLMo-2.

NO RANK STATISTIC ON THIS READ-OUT: phi is a range ratio, deliberately. With k=3 points
  monotone in the same order Spearman is +1.000000 as an ARITHMETIC IDENTITY, not as
  evidence (exact two-sided p floor at n=3 is 2/6 = 0.3333, unreachable at any alpha).
  No Spearman may be quoted from any budget read-out of k < 5, and none is used here.
```

---

## 4. FIRABILITY PRECHECK — computed now, before accepting the rewrite

Command actually run (0 GPU, CPU only, wzc1):

```
python3 /tmp/b04_repair_precheck.py
```

It reads `evidence/B04_wzc1_floor_analysis.json` (itself regenerated this session by
`python3 proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py`) plus the
on-disk `per_example_*.jsonl` and `summary.json`.

### Part A — spans and sup ratios

| grid | span S | points inside `I` | `sup_y |β|·S / range(y)` |
|---|---:|---|---:|
| `GRID_W` = {25k,50k,100k,128k,200k} @ S=175000 | 175000 | 3 / 5 | **1.173627** |
| `GRID_I` = {100k,128k,153.5k,175k,200k} @ S=100000 | 100000 | **5 / 5** | **1.220390** |
| *(illegal: `GRID_W` rescaled to 116500 = revision 1)* | 116500 | 3 / 5 | 0.781300 |

Each sup was verified empirically against the sup-attaining step shape, not only from the
closed form `S · Σ_{w_i>0} w_i`. The third row shows why revision 1's combination also
**disabled the max-guard** (sup < 1 ⇒ the slope term can never bind).

### Part B — the constants the thresholds rest on

| quantity | value |
|---|---|
| `D` | 0.021820 |
| `σ̂` | 0.000541 |
| `6σ̂` guard | 0.0032435 → `D` is **6.73×** the guard → admissible |
| k-matching | numerator k=5 / denominator k=5 → `E[range 5]/σ = 2.3259` cancels |
| φ under pure run-to-run noise | **0.0576** → **5.2× below** the PASS line, so PASS is a real measurement of a small effect |
| excursion KILL / PASS | ≥ 0.013092 (24.2 σ̂) / ≤ 0.006546 (12.1 σ̂) |
| monotone-equiv \|β\| on `GRID_I` (S=100000) | KILL ≥ 1.3092e-07 /step, PASS ≤ 6.5460e-08 /step |
| monotone-equiv \|β\| on `GRID_W` (S=175000) | KILL ≥ 7.4811e-08 /step, PASS ≤ 3.7406e-08 /step |
| shape-safe single-number PASS boundary, `GRID_I` | min(y) ≥ **0.103137** (was 0.102923 on `GRID_W`; larger because `GRID_I`'s sup ratio is larger) |
| single-number KILL boundary (both grids) | min(y) ≤ **0.095408** (sufficient, since `max() ≥ range`) |

### Part C — all three verdicts still reachable after the repair

| constructed y at the 5 read-out points | φ on `GRID_I` | φ on `GRID_W` | combined |
|---|---:|---:|---|
| `[0.0902, 0.0951, 0.1005, 0.1042, 0.1085]` monotone | 0.8483 KILL | 0.8387 KILL | **KILL** |
| `[0.1085, 0.0905, 0.0885, 0.0975, 0.1090]` non-monotone V | 0.9395 KILL | 0.9395 KILL | **KILL** |
| `[0.1000, 0.1030, 0.1055, 0.1070, 0.1085]` mid | 0.3914 NARROWED | 0.3896 NARROWED | **NARROWED** |
| `[0.1062, 0.1071, 0.1078, 0.1081, 0.1085]` early convergence | 0.1054 PASS | 0.1054 PASS | **PASS** |
| `[0.10850, 0.10796, 0.10904, 0.10812, 0.10850]` noise wobble (2σ̂) | 0.0495 PASS | 0.0495 PASS | **PASS** |

Reachable: `{KILL, NARROWED, PASS}` — none unreachable. Conservatism of the combine rule:
over 20 000 random 5-point shapes, the combined verdict was less severe than either grid alone
**0 times**. These y-vectors are **hypothetical**, written before any arm is evaluated; they
prove the gate *can* return each verdict, not which it will.

### Part D — the statistic value, next to the threshold

Union of the two prereg grids = `{25000, 50000, 100000, 128000, 153500, 175000, 200000}`,
7 arms. For each I searched **both physical disks** for an eval dir whose `summary.json.meta.ckpt`
is `outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step<N>.pt` **and** which carries all six
`per_example_{task}.jsonl`:

| read-out step | margin-computable eval dir |
|---|---|
| 25000 | **none** (wzc1: 45 dirs scanned; zwfy6: 165 dirs scanned) |
| 50000 | **none** |
| 100000 | **none** |
| 128000 | **none** |
| 153500 | **none** |
| 175000 | **none** |
| 200000 | `olmo2_downstream_results/keep14_s1234_step200000_sv181` (median_margin **0.108500**, n = 17195) |

**6 of 7 absent. On `GRID_I`: 4 of 5 absent. On `GRID_W`: 4 of 5 absent.**

zwfy6 was searched from `.73` by running a metadata scan over
`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/olmo2_downstream_results`
(165 dirs): **no dir on that disk references any `..._seed1234/step*.pt` checkpoint at all.**
The checkpoints themselves are present and healthy on wzc1 —
`outputs/olmo2_probe2_7B_keep14fresh2_seed1234/` holds
`step{25000,50000,100000,128000,153500,165000,170000,175000,180000,185000,190000,195000,199000,199500,200000}.pt`
(sizes, MAIN's own `ls` census 2026-08-16: **12 x 48 724 474 298 B, 2 x 48 724 473 567 B
(step25000/step50000), and 1 x 48 724 468 275 B (step200000)**). An earlier draft of this line gave
the range as "48 724 473 567 – 48 724 474 298 B each", which **excludes step200000** — the one
checkpoint that does have an eval. Corrected. **Only the evaluation is missing, not the models.**

> ### THE STATISTIC, NEXT TO ITS THRESHOLD
>
> | | value |
> |---|---|
> | **φ on `GRID_I`** | **NOT COMPUTABLE** — `y` has 4 holes ⇒ `max(y)−min(y)` undefined, `ols_slope(steps,y)` undefined |
> | **φ on `GRID_W`** | **NOT COMPUTABLE** — same 4 holes |
> | threshold | KILL ≥ 0.60 / PASS ≤ 0.30 |
> | **verdict** | **`READOUT_ABSENT`** (a hard abort, and a **NON-PASS**) |
>
> ### **CAN THE GATE FIRE? NO — NOT WITH THESE DATA.**
>
> This is the sentence the task asked for, and it is worth being precise about *why*, because
> it is a stronger failure than the one the decidability lens named. The lens said the
> statistic **was not the quantity measured**. The truth is that **the quantity was never
> measured at all.** φ does not evaluate to a wrong number; it does not evaluate.
>
> `phi_budget()` (`code/analyze_b04_wzc1_floor.py:128-155`) would raise before reaching its
> own guards — and note what that means for the machinery already built around it: the
> `--selftest` at `:485-537` passes today, and passed on 2026-08-14 and 2026-08-15, because it
> feeds **hand-written hypothetical y-vectors**. A selftest over invented inputs proves the
> *function* is falsifiable. It says nothing about whether the *gate* can run. Revision 2's
> `verdict_reachability_machine_checked` field, and the 2026-08-15 shape-safety fix that
> pinned `PASS_MIN = 0.102923`, are both correct **and both about a function that has no data
> to consume.** That is the gap this document closes.

### Part E — what *is* on disk, and which way it leans

Reported as auxiliary evidence, explicitly **not** as the gate firing. Each carries the reason
it is not the gate.

**E1 — the only fixed-damage multi-budget OLMo ladder with core6 on disk.** A scan of both
disks for training runs with ≥ 2 distinct heal steps evaluated on core6 found exactly one
usable OLMo cell: `outputs/olmo2_probe2_7B_keep8fresh2` at steps 100000 / 110000 / 121000, all
three on wzc1 with 8/8 shards and `n_scored` exact on all 6 tasks.

| step | dir | core6 |
|---:|---|---:|
| 100000 | `7B_keep8_step100000` | 0.516788 |
| 110000 | `7B_keep8_step110000` | 0.521836 |
| 121000 | `7B_keep8_step121000` | 0.523775 |

φ-analogue **on core6** (own span 21000, `D_core6 = 0.098167`): range term 0.006987, binding
term = range, **φ_core6 = 0.0712 → leans PASS.**

Three caveats, all disqualifying it as the gate: (i) **core6 is not the primary** —
`median_margin` is, and these 3 dirs have **no `per_example`**, so the primary is not
computable on them; (ii) k = 3, so no rank statistic is admissible (Spearman would be +1 by
identity); (iii) its span 21000 is **18.03% of `|I|`**, so 0.0712 is a **lower bound** and
cannot be read as "budget is negligible across `I`".

**E2 — equal-budget damage contrast on the primary metric.** `shortgpt16@200k` and
`keep14@200k` are both at heal step 200000 — budget held *exactly* equal — and differ in
median_margin by **0.008709 = 16.1 σ̂ = 39.91% of `D`**. This is the one place on disk where
depth moves the primary metric with budget fixed, and it moves it well clear of the floor.
Caveat: those two arms differ by more than depth (fresh=0/16L versus fresh=2/16L, both totalling
16 layers), so it is "structural change at fixed budget", not "pure depth"; and n = 2.

**E3 — the out-of-family precedent.** The Qwen f12k2/14L fixed-damage cell scores φ = 1.7760 →
KILL at its own span 198000. It stays in the record as proof KILL is reachable. It is **not** a
prediction for OLMo-2: B04's live claim is OLMo-2-only, and Qwen's general claim is already dead
(ρ = +0.4286, p = 0.42).

**E4 — the tempting shortcut, closed.** `shard*of8.json` does contain per-sample `lls` /
`norm_lls`, which looks like it could substitute for `per_example`. It cannot:
`scripts/eval_olmo2_probe2_downstream.py:476` caps `samples` at 6 rows per shard per task
(measured: 48 per task, 288 pooled, versus 17 195), and the rows carry no `item_id`. Bootstrap
SD of the pooled median at n = 288 is **0.008202 = 15.2 σ̂ = 37.6% of `D`** — a 5-point range
built from that is noise-dominated by construction. Hence the `SHARD_SAMPLES_ARE_NOT_A_READOUT`
abort.

**Net lean, stated as a lean and nothing more.** E1 leans PASS but on the wrong metric over
18% of the interval; E2 says depth does move the primary metric at fixed budget, which is
mildly favourable to B04; E3 is out of family. **These do not sum to a verdict, and the gate
still returns `READOUT_ABSENT`.**

---

## 5. What data would make the gate fire

| grid | arms needed (none evaluated) | GPU-h |
|---|---|---:|
| `GRID_I` (primary) | `seed1234` step 100000, 128000, 153500, 175000 | 1.076 |
| `GRID_W` (secondary) | additionally step 25000, 50000 | +0.538 |
| **both** | **6 arms** | **1.613** |

Cost anchor, verified in the log: `logs/sv181_main.log` lines 5-6 —
`[2026-08-12 01:12:18] (2) core6 downstream -> keep14_s42_step200000_sv181` →
`[2026-08-12 01:14:19] OK 8/8 shards` = **121 s** on 8 GPUs = 0.268889 GPU-h per rung; × 6 =
**1.613 GPU-h**. Same harness, same arch (sm_100), same driver
(`scripts/_run_paperB_keep14_seedvar_local.sh:116-125`, only `--ckpt` changes) as the arm that
produced the one read-out point already in hand.

All 6 checkpoints are on wzc1 now (§4 Part D). **sm_100 is mandatory** —
`paperB/SEEDVAR_KEEP14_VERDICT.md` line 3 puts σ̂ and the whole comparator ladder on LOCAL only,
so a non-sm_100 arm confounds the run-to-run term with a hardware term and makes even a FAIL
uninterpretable. LOCAL or `.21`. `--save_per_example` is **required**, or the run produces
another `7B_keep8_step100000`: correct core6, and the primary metric still not computable.

**Closing the 14.16% gap (optional, and honestly out of reach cheaply).** `GRID_I` cannot reach
`I`'s floor of 83500 because no `seed1234` checkpoint exists between 50000 and 100000. Reaching
it would need a re-heal with denser saves in that window — the 244–2560 GPU-h class of spend
that this gate exists to *decide*, not to presuppose. So the gap is **disclosed and accepted**,
and φ_I is reported as a lower bound over `I`.

---

## 6. Honest expected outcome

**What I expect if the 6 arms are evaluated: `NARROWED`, with `PASS` the next most likely, and
`KILL` genuinely possible.** Reasoning, with each input's weight stated:

- The one on-disk fixed-damage OLMo budget excursion (E1) is small — 7.12% of the comparator
  range — but it is on core6, not the primary, and covers only 18% of `I`. **Weak, points PASS.**
- Depth *does* move the primary metric at fixed budget by 16.1 σ̂ (E2), so the effect B04 claims
  is not obviously a budget artefact. **Weak, points PASS.**
- Against that: healing curves are steepest early, and `GRID_W` deliberately includes step
  25000 and 50000. If median_margin moves materially between 25k and 100k, φ_W alone can carry
  the combined verdict into NARROWED or KILL **even if φ_I is small** — which is precisely why
  `GRID_W` is retained rather than discarded.
- The only real fixed-damage precedent anywhere (E3) scored KILL — but out of family, and the
  Qwen general claim is dead. **Very weak for OLMo-2.**

**I am not confident in that expectation and it is not part of the gate.** The single fact that
matters and that I *am* confident in: **today the gate returns `READOUT_ABSENT`, which is a
non-pass, so the 244–2560 GPU-h matched-family ladder stays unauthorised.** The cheap next step
is 1.613 GPU-h of *evaluation only* — no training, all 6 checkpoints already on disk.

**On what the previous two revisions got right.** The max-guard, the k-matching, the denominator
guard, the shape-safe boundary rounding and the `--selftest` are all sound and are kept intact
here. Revision 2's error was narrower than it looks: it fixed the constant and left the grid, so
its numerator and denominator still spoke about different intervals. The error underneath both
revisions is one worth naming for the next agent: **three separate passes checked whether φ's
*formula* was right, and none checked whether φ's *inputs existed*.** A pre-registered,
fully-disclosed, machine-selftested, adversarially-revised gate can still be incapable of
firing — because the selftest was fed invented numbers.

---

## 7. Provenance

Every number here is either (a) reproduced this session from files on disk, or (b) quoted with
its file:line. No GPU was used; nothing was written to `olmo2_downstream_results/`.

- `proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py` — lines 100-104
  (`LADDER`), 120-125 (`G1_READOUT_STEPS`, `READOUT_SPAN`, `SLOPE_TERM_SUP_RATIO`, `PHI_KILL`,
  `PHI_PASS`), 128-155 (`phi_budget`), 335/357 (`dstep`, `damaged_heal_step_span`), 338-339
  (guard), 485-537 (`selftest_phi`). Full run reproduced the evidence JSON.
- `proposal/backlog/B04-eval-fragility-incubator/evidence/B04_wzc1_floor_analysis.json` —
  `clause5_budget_discrimination.{damaged_heal_step_span, readout_span_USED_IN_PHI,
  damaged_range_median_margin}`, `per_metric_floor_analysis.median_margin.sigma_hat`.
- `scripts/eval_olmo2_probe2_downstream.py:404-484` — `samples` capped at 6/shard/task (line
  476); `per_example` writes `norm_lens`/`norm_scores` (lines 456-474).
- `olmo2_downstream_results/` on **wzc1** — 45 dirs enumerated; per-rung `summary.json.meta`
  read for every rung in §1's table; `keep14_s1234_step200000_sv181` median_margin 0.108500 over
  n = 17195 recomputed from its `per_example_*.jsonl`.
- `olmo2_downstream_results/` on **zwfy6** (via `.73`,
  `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`) — 165 dirs enumerated by
  metadata scan; no `seed1234` checkpoint reference found.
- `outputs/olmo2_probe2_7B_keep14fresh2_seed1234/` — 15 `step*.pt`, sizes listed in §4 Part D.
- `logs/sv181_main.log:5-6` — the 121 s/rung cost anchor.
- `git 45980ad` (2026-08-14 23:21 +0800) — the adversarial verdict, 6/6 `NEEDS_REVISION`, B04
  `decidable=False`. `git 9659a0f` (23:54) — revision 2. `git 8f75def` (2026-08-15 01:05) — the
  PASS-boundary shape-safety fix.
- Bootstrap SDs of the median (n = 48 / 288 / 17195 = 0.019956 / 0.008202 / 0.001000) computed
  by resampling the donor's own 17 195 pooled margins, 4000 replicates, `seed=20260816`.
  **Caveat, per `memory/numpy-version-split-breaks-cross-node-bootstrap.md`:** these used the
  stdlib `random` module on wzc1 and are used only for an order-of-magnitude argument (15.2 σ̂),
  never as a threshold.

**Unverified, and named as such:** whether `zwfy6` holds any `seed1234` *checkpoint* (I scanned
its `olmo2_downstream_results` metadata, not its `outputs/`) — immaterial, since the checkpoints
are confirmed present on wzc1 and the gate must run on sm_100 (wzc1) anyway.
