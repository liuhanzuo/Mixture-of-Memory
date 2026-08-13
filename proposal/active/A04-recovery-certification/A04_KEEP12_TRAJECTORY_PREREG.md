# PRE-REGISTRATION — keep12+fresh2 dense trajectory monotonicity (8-point, 5 000-step grid)

**Written 2026-08-13, BEFORE any keep12 checkpoint was scored on the capability axes.
Committed before the first number was looked at.** Nothing in this file may be edited
after the verdict exists; corrections go in the verdict document as a labelled amendment.

---

## 0. The claim under test (call it **P**)

> **P.** The NI margin of a *damaged, healing* arm, measured relative to the pinned
> vanilla-7B null, **wanders non-monotonically along the training trajectory**, with an
> amplitude comparable to the certification tolerance Δ itself. Therefore a single-point
> accept that does not report its neighbourhood is uninterpretable.

P currently rests on three legs, each defective:

| leg | evidence | defect |
|---|---|---|
| keep14+fresh2 trajectory | `A04_KEEP14_TRAJECTORY_NI_VERDICT.md` — popqa margin **−0.6939 pp** over 128k→153.5k, then **+0.4556 pp**; the accuracy move is resolved (p = 0.0001) | **3 points only**, spacing **25 500 steps**, uneven |
| neighbour variability | `A04_NEIGHBOUR_VARIABILITY_VERDICT.md` — keep8 triviaqa range **1.1202 pp** at 500-step spacing, 1.70× the noise floor | **7 of 8** decision-axis ranges fall **inside** the noise gate `E[range of 3] = 1.6926 σ`; only 1 of 8 clears it |
| full32 trajectory | `A04_FULL32_TRAJECTORY_NI_VERDICT.md` / `A04_FULL32_READING_B_IS_FIRING.md` — step15000 accepts by **more** than step25000 | full32 is a **zero-damage** CPT arm, **not a recovery arm** |

**What P is missing:** a genuinely damaged recovery arm, on a **dense, evenly spaced**
step grid, with **enough points to test monotonicity as a trend** rather than as a sign.

## 1. The measurement

`outputs/olmo2_probe2_7B_keep12fresh2/` on **zwfy6** supplies exactly that:

**Primary grid (8 points, exact 5 000-step spacing):**
`130000, 135000, 140000, 145000, 150000, 155000, 160000, 165000`

**Secondary points:**
- `124000` — the resume anchor; **outside** the equispaced grid, reported separately, and
  **excluded from the Spearman/OLS trend statistics** so the grid stays uniform.
- `165500, 166000` — a 500-step triple `{165000, 165500, 166000}` for **Q4**, an
  independent replication of the neighbour-range measurement (which currently exists on
  keep8 only).

keep12+fresh2 is `keep_front=12, n_fresh=2, num_hidden_layers=14, 157 tensors`
(`arch_meta.json`, verified on zwfy6). It is a **different damage level** from keep14+fresh2
(14+2 = 16 layers, 179 tensors), so this is a **cross-arm** test of P, not extra points on
the arm that generated P.

### 1.1 Single-process provenance — verified before this file was committed

`A04_NEIGHBOUR_VARIABILITY_VERDICT.md` §1.2 established that a 500-step neighbourhood can
straddle a **resume seam** (loader rebuilt without intra-epoch fast-forward ⇒ different data
order). That check was run **first** here:

`logs/olmo2_7B_keep12fresh2_resume200k_v2.log` contains exactly **one** process start
(`[seed] set_seed(42)` at 2026-08-08 13:58:02, one `[resume] loading ckpt … step124000.pt`,
one `[resume] sampler.set_epoch(1)`), and **all eleven** checkpoints in this dispatch are
saved by that single process (grid saves at log lines 348 / 626 / 904 / 1182 / 1460 / 1738 /
2016 / 2294, plus 2323 / 2351 for 165500 / 166000; last line = step 166020 at 2026-08-12
11:14). **There is no seam anywhere inside 124000 → 166000.** So — unlike keep8 cluster 1 —
the whole grid is one uninterrupted data order, and the neighbour triple in Q4 is clean by
construction.

If checkpoint verification later contradicts this (wrong meta step, corrupt zip, non-distinct
weights), the affected point is **dropped and recorded**, never silently substituted.

## 2. What counts as monotone, and what counts as wandering

Fixed now, per axis, on the pre-registered `split` tie convention, decision axes
`{triviaqa, popqa, mmlu_content}` (nq_open reported but **demoted**, per §5.2 of the design):

| label | criterion (on the 8-point grid) |
|---|---|
| **MONOTONE** | all 7 successive margin differences have the same sign (`np.diff` all ≥ 0 or all ≤ 0) |
| **MONOTONE_TREND** | not strictly monotone, **but** Spearman ρ(step, margin) has \|ρ\| ≥ 0.7 **and** p < 0.05 |
| **WANDER** | neither of the above, **and** the observed margin range clears the noise gate of §3 |
| **UNRESOLVED** | neither of the above, **and** the range is **inside** the noise gate — reported as *no detectable diffusion*, explicitly **not** as a finding |

Two more quantities, fixed now:

- **Largest single-step margin change** `max|Δmargin|` between adjacent grid points,
  reported as an absolute pp value **and** as a ratio to that axis's Δ. P's amplitude claim
  ("comparable to Δ") is operationalised as **`max|Δmargin| / Δ ≥ 0.25`**.
- **Non-monotone excursion count**: the number of sign reversals in `np.diff(margin)`.
  A monotone series has 0.

**Every adjacent-interval accuracy move additionally gets its own paired item bootstrap**
(imported `paired_bootstrap`) and is only called *resolved* under the **conservative AND**
(CI95 excludes zero **AND** p < 0.05), exactly as the two prior passes did. A sign is not a
finding; an unresolved sign flip is a wobble, not an excursion.

## 3. The noise gate — how it is used, and what it forbids

For *k* checkpoints and iid N(0, σ) item noise, the expectation of `max − min` is
**strictly positive even at zero true spread**: `E[range of k] = c_k · σ`, exact for the
normal. `c_3 = 3/√π = 1.6926` (the constant used by the keep8 pass, k = 3);
**for the 8-point grid the correct constant is `c_8 = 2.847` — NOT 1.6926.**

Using 1.6926 on 8 points would understate the noise floor by 1.68× and manufacture a
finding. So:

- **8-point grid** → `expected_range_if_pure_noise_pp = c_8 · mean(bootstrap SE)`, with
  `c_8` obtained by **direct Monte-Carlo of the standard-normal range** (n = 2 000 000,
  fixed seed, recorded in the JSON) rather than a table lookup, and cross-checked against
  the closed-form `c_3 = 1.6925687506432689` on k = 3 as a correctness test of the estimator.
- **Q4's 500-step triple** → `c_3 = 1.6926 · mean(SE)`, the **identical** convention as
  `A04_NEIGHBOUR_VARIABILITY_VERDICT.md` §2.3, so the two are directly comparable.
- **σ is the same item-level bootstrap SE used everywhere in A04** (derived from the
  imported `ni_rule`'s one-sided lower bound, `SE = (mean − lo95)/1.6449`). **No new
  σ estimator.**

**BINDING:** if an axis's range does **not** clear its gate, the verdict text must say
*"no detectable diffusion on this axis"* and the number may **not** be quoted as a measured
gap. This is the rule that retired 7 of 8 ranges last time and it is not being relaxed.

## 4. What would make me narrow or kill P

Committed in advance. Each row is a real possible outcome of this run.

| outcome | consequence for P |
|---|---|
| **≥ 2 of 3** decision axes are MONOTONE or MONOTONE_TREND **in the improving direction**, and no axis WANDERs | **P is narrowed hard.** Non-monotonicity would then be a property of keep14/keep8/full32 and **not** of healing at this damage level. P must be restated as arm-specific and may **not** be sold as a general methodological law. |
| all 3 axes UNRESOLVED (every range inside its gate) | **P's amplitude claim is dead on this arm.** The margin does not measurably move at 5 000-step spacing ⇒ the "single point is uninterpretable" argument loses its quantitative basis here, and the neighbour precondition must be re-scoped to arms/axes where a range was actually measured. |
| ≥ 1 axis WANDERs **and** `max\|Δmargin\|/Δ ≥ 0.25` | **P replicates on a second, independently damaged arm.** This is the outcome that would license promotion to `paperD`. |
| popqa specifically regresses resolvedly somewhere mid-grid | **Q3 replicates**; P's strongest leg becomes cross-arm rather than single-arm. |
| popqa is monotone-improving throughout | **Q3 fails to replicate.** Combined with the keep8→shortgpt16 replication failure already on record, the popqa dip becomes **keep14-specific** and P must be argued from *range* alone, never from *directional regression*. |

**I am explicitly authorised to conclude that P is wrong.** A narrowing or a kill here is
worth more than a confirmation, because P is one gate away from being promoted to a paper.

## 5. Q3 — the specific replication test

keep14's strongest leg is popqa **−0.6729 pp over 128000→153500** (CI95 [−0.9252, −0.4206],
p = 0.0001, 122 wrong→right vs 218 right→wrong).

**keep12 has no 128000 checkpoint** (rotated away; the arm's saves in this range are
124000 then 130000+). So the interval is **not** step-matched, and I will not pretend it is.
Q3 is therefore evaluated as a **phenomenon-level** replication with two pre-committed reads:

1. **Interval-matched-in-length:** the 25 500-step-wide window closest to keep14's, on
   keep12's grid = **130000 → 155000** (25 000 steps, within 2 % of keep14's width). Its
   popqa Δacc + CI + p is the headline Q3 number.
2. **Any-interval:** whether *any* adjacent 5 000-step popqa interval on the grid is a
   resolved regression, and if so how many. This is the weaker but more honest question,
   because with 7 intervals a resolved move somewhere is easier to find than in keep14's 2.

**Multiplicity is acknowledged in advance:** 7 intervals × 4 axes = 28 adjacent-interval
tests. At α = 0.05, ~1.4 false positives are expected under a global null. So the JSON
records, per axis, both the raw resolved count **and** a **Benjamini–Hochberg** pass over
the 21 decision-axis interval p-values, and **the verdict's monotonicity claims rest on the
Spearman trend + the range gate, not on counting resolved intervals.** A single resolved
interval among 21 will **not** be reported as an excursion on its own.

## 6. Protocol — frozen, and re-verified for THIS arm rather than inherited

| field | value | how established |
|---|---|---|
| closedbook batch size | **32** | driver echoes `DRIVER START … cb_bs=32` + per-axis `START … bs=32`; parsed by a **fail-closed** gate before scoring |
| mmlu_content batch size | **16** | same mechanism, `mmlu_bs=16` |
| `add_bos` | **False** | read from every `summary.json:meta`, asserted with **`is False`** — never `is not True` (which passes silently on `None`) |
| `max_new_tokens` | **32** | asserted on every generative dir |
| `chat_template` | **False** | **structural**: neither harness has a chat-template code path. Asserted `is not False`-style, i.e. the recorded value must literally be `False` |
| shards | **8**, index set exactly `{0,…,7}` | set equality, not a file count; exact item counts (triviaqa 17944 / popqa 14267 / nq_open 3610 / mmlu 14042), 0 duplicate `item_id`, 0 `nan` |
| anchor | vanilla `models/OLMo-2-1124-7B` via **imported** `ANCHOR` | never redeclared, never substituted (G0/G2); `full32_step25000` forbidden as anchor |
| Δ | `0.10 × residual(intact)`, **imported** | never substituted; `max(Δ,ε)` / `0.10·|residual|` prohibited |
| nulls | **imported `build_nulls`** | MAIN's hand arithmetic was ~0.5 pp off twice; no null is recomputed here |

`summary.json:meta` records **neither** `batch_size` **nor** `chat_template`
(`A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md`), so both are confirmed from the **invocation**,
and the batch sizes are re-verified from **this dispatch's own driver logs** — not assumed
from keep14's. Batch size is not free: bs32→bs48 flipped 12/14267 popqa and 10/3610 nq_open
items (`full32_rescore_v2_20260812.sensitivity_bs48_probe`).

## 7. Node discipline and the numpy hazard

- GPU work on **`.73`** and **`.82`** only (8×H20 each, zwfy6). **`.104`** (paperC Qwen3
  heal) and **`LOCAL`/`.21`** (SparseForge #246) are **out of budget**. Both target nodes
  verified 8×0 MiB before launch, and the driver **refuses to start** if > 8000 MiB is held.
- **The entire bootstrap analysis runs on ONE node.** `.73` = numpy **2.5.1**,
  `.82` = numpy **2.4.6**, and `Generator.multinomial` differs in **19 of 10 000 rows**
  between them (max margin drift 0.005294 pp, triviaqa only —
  `A04_NEIGHBOUR_VARIABILITY_VERDICT.md` §4.1). **All statistics for this verdict are
  computed on `.73` (numpy 2.5.1)** and the node + version are recorded in the JSON.
  Scoring may be split across both nodes — it is deterministic given the harness — but
  **no statistic may mix nodes.**
- Bootstrap seed offsets must be **disjoint** from every archived cell. In use already:
  pilot_zero `arm_index` {0,1}; step100k 100–102; shallow_rung 200–203; keep14 trajectory
  300–301 (+ endpoint 201); neighbour variability 400–408 with guard 1700 / interval
  1900 / 2400; full32 trajectory (to be read from its JSON and avoided). This run uses
  `arm_index` **500+**, guard offset **2700**, interval offset **2900**, so **no archived
  number can be perturbed.**

## 8. Deliverables

- `A04_KEEP12_TRAJECTORY_MONOTONICITY_VERDICT.md`
- `evidence/a04_keep12_trajectory_monotonicity.json` — per-cell raw numbers, per-axis
  margin sequence, Spearman ρ + p, OLS slope, `max|Δmargin|`/Δ, sign-reversal count, the
  noise-gate decision per axis, the 28 adjacent-interval tests + BH, and the Q4 triple.
- `code/a04_keep12_trajectory_axes_driver.sh`, `code/a04_keep12_trajectory_monotonicity.py`
- a **new** `STATUS.json` key `keep12_trajectory_monotonicity_20260813`; **no existing key
  is modified**; `gpu_h_spent` = driver wall (start→end) × 8, per node, summed.

**One last commitment.** The verdict string is generated **mechanically** from the criteria
in §2–§3 by the analysis script. I do not get to choose it after seeing the table.
