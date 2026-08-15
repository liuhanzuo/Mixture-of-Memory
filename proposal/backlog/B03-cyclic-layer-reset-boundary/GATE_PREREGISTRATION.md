# B03 — GATE PRE-REGISTRATION (the read-out that `next_gate: NOT_SPECIFIED` was waiting on)

**Authored 2026-08-15. 0 GPU, 0 SSH, no node touched.** Every number below is either
(a) read off a file on disk with its path and md5 quoted, or (b) computed on CPU from those
files by a hand-written, self-tested routine (no `scipy` on any of the five nodes), or
(c) explicitly labelled **ASSUMED / ANCHORED, NOT MEASURED**.

**This document is submitted BEFORE any B03 GPU run exists.** No B03 layer-reset run exists
at any scale on either disk (`gpu_cost_estimate.why`), so there is no B03 number that could
have informed a threshold here. The git timestamp of the commit that adds this file is the
proof of ordering. **Nothing below may be edited once the first B03 number lands**; the
verdict goes in a separate `*_VERDICT.md`, and if any constant here is changed the gate is
**VOID** and must be re-run from scratch (the same rule A04's `frozen_constants.note` uses).

**Completing this pre-registration does NOT authorise GPU.** See §10.

---

## 0. What was missing, and the shape of the fix

`STATUS.json:next_gate` is the sentinel string `NOT_SPECIFIED`, and it diagnoses itself
correctly: `PROPOSAL.md` §「1B 核心 gate」 fixes a **2×3 design** (data regime
{single-pass, repeated-data} × reset count `N ∈ {0,1,3}`, at 1B) and **no decision rule**.
Its own checklist,
`next_gate_design_extracted_from_PROPOSAL_20260815.what_is_missing_before_this_is_executable`,
names four holes. This document closes them in order:

| Hole (verbatim from `STATUS.json`) | Closed in |
|---|---|
| "n / sample count per cell and the null-floor construction" | §3 (n), §4 (nulls and floors) |
| "the statistic for '显著 interaction' and its alpha / multiple-comparison handling across 6 cells x 4 metric families" | §2 (statistic), §5 (α + multiplicity) |
| "the threshold separating clause 1 (interaction) from the stated closure '普通统一退化'" | §6 (Θ_int, Θ_sep) + §7 (`kill_if`) |
| "a GPU-h estimate: no 1B reset run exists in this repo to extrapolate a rate from" | §9 (derived from the **1B continue-train** anchor, with the extrapolation's weakness stated) |

### 0.1 A contradiction inside `PROPOSAL.md` that has to be resolved BEFORE data

`PROPOSAL.md` §「存活条件」 says **「至少满足一项」** of two clauses — survive if **either**
holds ⇒ die only if **both** fail (`¬C1 ∧ ¬C2`).
`PROPOSAL.md` §「关闭条件」 lists **three** clauses "**ANY** of which closes", and those three
are the negations of the same two survival clauses ⇒ die if **either** fails (`¬C1 ∨ ¬C2`).

**`¬C1 ∧ ¬C2` and `¬C1 ∨ ¬C2` are different gates**, and picking between them *after* seeing
which one a result passes is the exact failure mode this file exists to prevent. So it is
picked now, with the reason:

> **ADOPTED, PRIMARY: the `存活条件` reading — KILL iff BOTH clauses fail.**
> Reason: it is the reading `RELATED_WORK.md` §4's *safe residual claim* is written against
> (that sentence carries the interaction as "the discriminating measurement" **and** the
> recoverable-PPL/persistent-knowledge separation as co-equal content), and it is the
> **conservative** direction for a kill decision — the asymmetry matters because a
> direction wrongly killed produces no correcting evidence, whereas a direction wrongly
> kept costs the Tier-2 GPU-h and gets killed on the next gate.

> **ALSO REPORTED, MANDATORY, NOT OPTIONAL: the `关闭条件`-literal verdict** (`¬C1 ∨ ¬C2`).
> The verdict document must print **both** verdicts side by side in its first table. If they
> disagree, the headline sentence is *"B03 survives its `存活条件` gate and fails its
> `关闭条件` gate; the two sections of `PROPOSAL.md` are inconsistent and this result does
> not resolve which was intended."* **It may not be reported as a clean survival.**

### 0.2 One thing this gate deliberately does NOT do

It does **not** try to distinguish the reset operator from LLF. `RELATED_WORK.md` §2.1
establishes that `--keep_front_layers K_f --n_fresh_layers K` at constant total depth **is**
LLF's mask `M^l = 1[l < K_f]`, and §3 items 1–2 forbid any method claim. **For a
regime-boundary claim that is a feature, not a defect**: using LLF's *identical* operator
makes the result a statement about **LLF**, not about a B03 variant a reviewer can dismiss as
untested. The `design_constraints_the_literature_forces_on_the_future_readout[0]` warning
("must NOT be `keep_front+n_fresh` **if any structural distinction from LLF is wanted**") is
therefore satisfied by *not wanting one*. The mid-stack-excision alternative it mentions is
explicitly **out of scope** — it is new trainer code and it would break the LLF identity that
gives the negative result its force.

---

## 1. The measured quantities

Per cell `(regime, N)` and per axis, two families of numbers are produced.

**Knowledge retention** (the primary family), in *retention units*:

```
R_know(cell, axis) = [ acc(cell, axis) - null(axis) ] / [ acc(intact_1B, axis) - null(axis) ]
```

i.e. **calibrated residual as a fraction of the intact model's calibrated residual**. This is
A04's `nulls_per_metric` convention verbatim (`intact_residual`), not a new one. `R = 1` means
"as much above-null knowledge as the intact 1B"; `R = 0` means "at the construct-appropriate
null". `acc(intact_1B, axis)` and `null(axis)` are **pre-registered constants** from §4.1,
never re-estimated per cell.

**Perplexity retention**, same units, so the two axes are commensurable:

```
R_ppl(cell) = ppl(intact_1B) / ppl(cell)          (= 1 / PPL-tax)
```

`ppl(intact_1B) = 10.6416` on `n_windows = 4096`, `n_tokens = 8,384,512` — A04
`nulls_per_metric.in_domain_PPL`, measured on OLMo-2 **1B**, the same scale as this gate.
PPL gets **no null** and is **never a capability axis** (A04's standing rule); it enters only
through §6.2's dissociation gap.

**Dissociation gap:**  `D(regime, N) = R_ppl(regime,N) − R_know(regime,N,primary)`.
By construction `D(regime, 0) ≡ 0` and `R ≡ 1` at `N = 0` (the `N=0` cell *is* the
un-reset continue-train reference for its own regime). **That identity is load-bearing in
§4.4** — it is why the range floor uses `c_2`, not `c_3`.

---

## 2. `statistic` — executable, and named to the line

### 2.1 PRIMARY: exact randomization test on the difference-in-differences contrast

```
Psi = [ R_know(RD, 3) - R_know(RD, 0) ] - [ R_know(SP, 3) - R_know(SP, 0) ]
```

on the **primary axis only** (`mmlu_content`, §5.1). `RD` = repeated-data, `SP` = single-pass.
This is the `reset × data-regime` interaction of `PROPOSAL.md` §「存活条件」 clause 1, written
as one scalar.

* **Unit of replication = the RUN**, not the item. A04's `power_analysis.unit_of_replication`
  fixes this for the repo and the reason is B04's tombstone: item-level significance at
  `n = 6` runs is not evidence about runs.
* **Test = EXHAUSTIVE ENUMERATION**, not sampling. Within **each** `N` level independently,
  relabel which `S` of the `2S` runs are `RD`; enumerate all `C(2S,S)²` joint assignments,
  recompute `Psi`, and take the two-sided p as
  `#{|Psi_perm| >= |Psi_obs|} / C(2S,S)²`. At `S = 3` that is `20² = 400` assignments — small
  enough to enumerate exactly on CPU in milliseconds.
* **Why exhaustive and not bootstrap.** Exhaustive enumeration uses **no RNG at all**, so it
  is **bit-identical on every node regardless of numpy version**. This repo has three numpy
  versions across five nodes (LOCAL 2.5.1 measured 2026-08-15, `.82` 2.4.6, others 2.5.1) and
  same-seed `multinomial` draws already **differ across nodes**
  (`memory/numpy-version-split-breaks-cross-node-bootstrap.md`). A primary decision that
  depends on the node is not a decision. Enumeration removes that failure mode by
  construction rather than by pinning.
* **Paired or unpaired?** **Unpaired between regimes** (an `SP` run and an `RD` run share no
  data order and cannot be paired) and **paired within a regime** only in the trivial sense
  that `R_know(regime,0)` is a *pre-registered constant per regime* estimated from that
  regime's own `N=0` cell means. The permutation therefore permutes **regime labels within
  each `N` level**, which is the only exchangeability the design supports.
* **Implementation + self-check**: enumeration is exact, so the self-check is a **closed-form
  identity**: the number of assignments must equal `C(2S,S)²` exactly, and permuting with the
  identity assignment must reproduce `Psi_obs` to `0` bits. The adjudicator must `assert`
  both, and must `assert` the p-value is a multiple of `1/400` at `S=3`.
* **SECONDARY / descriptive only**: a paired-item bootstrap CI on each cell mean, `N_BOOT =
  10000`, `seed = 20260815`, and **`assert numpy.__version__ == "2.5.1"`** — so it may run on
  LOCAL / `.73` / `.104` and **may NOT run on `.82`** (2.4.6). If it runs anywhere else the CI
  is void. **No kill or proceed decision may rest on this number.**

### 2.2 A `t`-based bound is reported ALONGSIDE, never instead

Because the exact test gives a p but not an interval, the verdict must also report
`Psi_obs ± t_{.975,df} · SE(Psi)`, `SE(Psi) = 2σ̂_run/√S`, with `df` chosen by §2.3.
The `t` quantile comes from a hand-written regularized-incomplete-beta routine
(no scipy). **Its self-test is mandatory and PASSED pre-data on 2026-08-15** against six
independent authorities:

| # | Authority | Worst error |
|---|---|---|
| 1 | `df=1` exact Cauchy form `t_p = tan(π(p−½))`, 7 quantiles | **1.10e-13** |
| 2 | `df=2` exact form `t_p = √2·u/√(1−u²)`, `u=2p−1`, 6 quantiles | (same run) |
| 3 | `df→∞` equals the standard-normal quantile, 3 quantiles | 2.38e-07 |
| 4 | **INDEPENDENT** Simpson quadrature of the `t` pdf, 9 `df` × 2 `p` | **5.38e-13** |
| 5 | Textbook 3-dp table, 12 entries | 4.45e-04 |
| 6 | Round-trip `cdf(ppf(p))==p` + antisymmetry, `df ∈ {1..30}` | 5.36e-15 (rel) |

> Two of my first three "authorities" were **my own misremembered closed forms** (a wrong
> `t_{0.80,2}` table value, and a bogus `df=4` formula). The self-test caught both, and the
> resolution each time was that the *code* was right and my *reference* was wrong — verified
> against the `df=2` closed form, which is exact. **That is why authority #4 is an independent
> numerical path and not a fourth remembered formula.** Any re-implementation must reproduce
> the table above before it is allowed to produce a verdict.

The `F` quantile used by §2.3 comes from the same beta routine and self-tests against the
standard `F` table: `F_.95(2,2)=19.0000` (table 19.0), `F_.95(3,3)=9.2766` (9.28),
`F_.975(2,2)=39.0000` (39.0), `F_.95(1,1)=161.4476` (161.4), `F_.95(5,5)=5.0503` (5.05),
`F_.95(10,10)=2.9782` (2.98).

### 2.3 `df` for the variance pool — decided by a pre-registered precondition, not by taste

Pooling `σ̂_run` over all 6 cells gives `df = 6(S−1)`; pooling only the 4 corner cells that
enter `Psi` gives `df = 4(S−1)`. Which one is used is **not** a choice made after seeing the
data:

> **PRECONDITION (fixed now).** Compute the per-cell `s²` on the primary axis. If
> `max s² / min s² ≤ F_{.975}(S−1, S−1)` — **39.0000 at `S=3`** — pool all 6 cells
> (`df = 6(S−1) = 12`). Otherwise pool only the 4 corner cells (`df = 4(S−1) = 8`).
> Both branches are pre-committed and **both clear the bar** (§3.3), so this precondition
> cannot change the verdict's direction; it exists so that the reported interval is honest.

---

## 3. `n` — and why this exact `n`

### 3.1 The design

| | `S` (runs per cell) | cells | runs |
|---|---|---|---|
| **Tier 1** (the 4 corner cells, `N ∈ {0,3}` × 2 regimes) — decides clause 1 | **3** | 4 | **12** |
| **Tier 2** (the 2 `N=1` cells) — decides clause 2's growth-in-`N` sub-clause | **3** | 2 | **6** |
| total | | 6 | **18** |

Items per axis are **frozen sets**, not resampled: `mmlu_content n = 14042`,
`triviaqa n = 17944`, `popqa n = 14267`, `nq_open n = 3610` (§4.1). The item sets are the
ones A04 pinned by sha256 (`g0_anchor_sha256_20260810`), so "same items" is machine-checkable
rather than asserted.

### 3.2 `S = 2` is excluded by an EXACT argument that does not depend on any variance

At `S = 2`, the number of joint relabellings is `C(4,2)² = 36`, so the **smallest attainable
two-sided p is `2/36 = 0.055556 ≥ 0.05`**. **`S = 2` cannot reach `α = 0.05` no matter what
the effect is.** At `S = 3`: `C(6,3)² = 400`, p floor `2/400 = 0.005000`. At `S = 4`: `4900`,
floor `0.000408`.

This is the same wall that killed B04 (`n = 6`, exact-permutation floor `2/720 = 0.002778`,
hit **twice** and still not a real effect) — quoted here as a **precedent for respecting the
floor**, not as a licence to buy p with more rungs. A04's `power_analysis.S2_is_unusable`
reaches the same "S=2 is not viable" conclusion by a completely different route (`t`
d.o.f.), which is why `PROPOSAL.md`'s implicit "2 seeds" minimum is rejected.

### 3.3 `S = 3` clears the effect-size bar — at the **pessimistic** end of the variance interval

`MDE(two-sided α=.05, power .80) = (t_{.975,df} + t_{.80,df}) · 2σ_run/√S`, primary axis
`mmlu_content`, `Θ_int = 1.0230 pp` (§6.1):

| `S` | `df` (pool-6) | multiplier | MDE @ `σ̂` | MDE @ `χ²` upper | vs `Θ_int` | verdict |
|---|---|---|---|---|---|---|
| 2 | 6 | 4.7413 | 0.3109 pp | 0.7626 pp | 3.29× / 1.34× headroom | (excluded by §3.2) |
| **3** | **12** | **3.5235** | **0.2311 pp** | **0.5667 pp** | **4.43× / 1.81×** | **ADOPTED** |
| 4 | 18 | 2.9630 | 0.1943 pp | 0.4766 pp | 5.26× / 2.15× | not bought (§9.3) |
| 5 | 24 | 2.6124 | 0.1713 pp | 0.4202 pp | 5.97× / 2.43× | not bought |

Corner-4 fallback at `S=3` (`df=8`, multiplier 3.6891): MDE 0.2419 / 0.5934 pp — **also
clears**, which is what makes §2.3's precondition safe.

**Equivalently, and this is the form G1 tests**: at `S=3` pool-6 the design tolerates
`σ_run(K) ≤ Θ_int / 3.5235 = 0.2903 pp`, i.e. **4.43× the measured continue-train `σ̂` and
1.81× its `χ²` upper bound**. Corner-4: `≤ 0.2773 pp`.

### 3.4 ⚠️ The honest weakness of §3.3, stated before it can be discovered

`σ̂_run` above is **continue-train** variance from A03/A04 arms (`keep7`+CPT@20k,
`keep12`@5k), **not** reset-cycle variance. **No reset-cycle `σ_run` exists at any scale on
either disk.** A multi-cycle reset run plausibly has *larger* run-to-run spread than a
single continue-train leg, because each reset re-draws a fresh random block.

Two consequences, both binding:

1. **§3.3 is a POWER ARGUMENT FROM A PRIOR, not from the target regime.** It is stated as
   such and may not be quoted as "the design is powered".
2. **G1 (§8.2) exists specifically to measure `σ_run(K)` before Tier 1 is funded**, and its
   pass condition is exactly the `0.2903 pp` bound of §3.3. If G1 measures
   `σ_run(K) > 0.2903 pp`, **`S` is NOT increased post hoc** — the gate reports
   `UNDERPOWERED` and B03 stops. (A04's `power_analysis.no_post_hoc_seed_addition`: adding
   seeds after seeing the variance is the selective-reporting move A01's gate-4 exists to
   prevent.)

Additional caveat on the prior itself: the A03/A04 seed arms **straddle** commit `ce5c298`
(2026-08-09, `DistributedSampler(..., seed=args.seed)`; verified at
`scripts/train_olmo2_arch_probe2.py:869`). Before that fix a "seed" moved only the fresh-block
init, not data order. Seed 45 landed 2026-08-11, i.e. after; the `keep12` seeds 101/102/103
are not individually date-verified in this pass. So `σ̂_run` may **understate** the data-order
component. This makes §3.4's demand for G1 stronger, not weaker.

---

## 4. `null floor` — four distinct floors, each with a magnitude

"Null floor" is ambiguous and `PROPOSAL.md` uses it once, unqualified. Four different floors
are needed and they are numerically different.

### 4.1 Construct nulls (the denominator of `R`) — MEASURED, frozen

From `proposal/archive/A03-parametric-vs-external-memory/evidence/a03_1b_floor_nulls.json`
(md5 `a97a73bf802737601a6057f767b70853`), as transcribed in A04's `nulls_per_metric`:

| axis | `n` | null | kind | intact 1B | **residual** |
|---|---:|---:|---|---:|---:|
| `mmlu_content` **(PRIMARY)** | 14042 | **0.28445022076627263** | longest-option, **`split` tie convention** (pre-registered; 34.219 % of items have ≥2 maximal-length options) | 0.3868 | **10.230 pp** |
| `triviaqa_em` | 17944 | 0.0025635309852875612 | best-constant string `"australia commonwealth realm"` | 0.4069 | 40.430 pp |
| `popqa_em` | 14267 | 0.0229200252330553 | best-constant string `"association football"`, argmax over 303 candidates | 0.155 | 13.210 pp |
| `nq_open_em` | 3610 | 0.0055 | best-constant answer | 0.1025 | 9.700 pp |

**BANNED interfaces, inherited verbatim from A04 and binding here:**
`MMLU_letter` (at 1B the pruned arm scores 0.2512 vs best-constant always-D 0.2689 =
significantly **below** floor, `p = 3.4e-3`; a barely-healed control emits `A` on 14042/14042
items — **0.25 is never the MMLU null**), and **raw `contains`** on generative QA (healing
changes verbosity ~6× — mean predicted chars 13.4 intact → 80.8 pruned+healed — and
`contains` rewards length; PopQA `contains` residual reads 56.4 % naive but 17.0 %
length-matched → **`contains_lenmatched` only**).

### 4.2 Item-level floor of the contrast — MEASURED-`n`, binomial

`SE_item(Psi) = 2 · SE_item(cell)`, `SE_item(cell) = √(p(1−p)/n)`, at the accuracies these
1B arms actually reach:

| axis | `n` | `p` used | `SE_item(cell)` | **`SE_item(Psi)`** | `Θ` | headroom |
|---|---:|---:|---:|---:|---:|---:|
| `mmlu_content` | 14042 | 0.322 | 0.3943 pp | **0.7886 pp** | 1.0230 pp | **1.30×** ⚠️ |
| `triviaqa` | 17944 | 0.100 | 0.2240 pp | 0.4479 pp | 4.0430 pp | 9.03× |
| `popqa` | 14267 | 0.045 | 0.1736 pp | 0.3471 pp | 1.3210 pp | 3.81× |
| `nq_open` | 3610 | 0.029 | 0.2793 pp | 0.5586 pp | 0.9700 pp | 1.74× |

The primary axis's **1.30× item headroom is the tightest number in this document** and must
appear in the verdict. It does not invalidate the design (the *decision* floor is the
run-level `SE(Psi) = 2σ/√S`, which at `S=3` is 0.0757 pp @`σ̂` / 0.1857 pp @`χ²` upper — an
order of magnitude smaller), but it means **the item sets may not be shrunk** and it is why
`nq_open` is demoted (§5.2).

### 4.3 Run-level sampling floor — the floor the decision actually uses

`SE(Psi) = 2 σ_run/√S`. At `S = 3`: `mmlu_content` **0.0757 pp** (`σ̂`) / **0.1857 pp**
(`χ²` upper); `triviaqa` 0.4233 / 1.0383; `popqa` 0.2996 / 0.7348; `nq_open` 0.1668 / 0.4091.
Same §3.4 caveat: this is a continue-train prior, and G1 replaces it with a measured
`σ_run(K)`.

### 4.4 Separation floor for clause 2 — and the `c_k` trap, handled explicitly

`memory/a-range-is-not-a-measurement-until-it-clears-its-floor.md` is binding: a range is
**not a measurement** until it clears its own floor, and `E[range of k iid N(0,σ)]/σ` depends
strongly on `k`. So clause 2's "curve separation" is given an explicit floor:

```
floor(range of D over the free N levels) = c_k * SE(D per level) = c_k * sigma_D / sqrt(S)
```

> **`k = 2`, NOT `k = 3`.** The 2×3 has three `N` levels, but `D(regime, 0) ≡ 0` **by
> construction** (§1) — it is an identity, not a measurement, and it contributes no variance.
> Only `N ∈ {1,3}` are free. **`c_2 = 2/√π = 1.1283791670955126`** (exact closed form;
> Monte-Carlo cross-check 1.1291 at 4e6 draws, numpy 2.5.1, seed 20260815).
> Using `c_3 = 3/√π = 1.6925687506432689` would **inflate the floor by exactly 1.5000× =
> +50.0 %** — the mirror image of the 40.6 %-*under*-statement that voided two A04 tasks in
> one day. The verdict document **must print `k`, `c_k` and the floor**, per that memory's
> "report both floors and the multiple, not just the ratio".

Illustrative magnitude, using `σ_D ≥ σ_R(know) = 0.00641` retention units (the *knowledge*
component alone; the PPL component is unmeasured, so this is a **lower bound** on `σ_D`):
at `S = 3`, `SE(D) ≥ 0.00370` → `floor ≥ 0.00418` retention units. Since `Θ_sep = 0.10`
(§6.2) is **24×** that lower bound, the separation clause is not floor-limited *if* `σ_D` is
near its lower bound — but `σ_D` is **not measured** and G1 must produce it (§8.2).

**No ratio of two ranges may be quoted anywhere in the verdict unless BOTH ranges clear
their own floors.** If either fails, the verdict word is `UNRESOLVED_SUBNOISE`, never
"smaller than".

---

## 5. `alpha` and multiplicity — family declared NOW

### 5.1 One primary axis, declared in advance

**PRIMARY = `mmlu_content`, tested at a plain two-sided `α = 0.05`.** Chosen pre-data
because it has (i) the largest `n` among the multiple-choice interfaces at a construct-valid
null, (ii) the smallest `σ_run` of the four axes (0.0656 pp, `χ²` [0.0409, 0.1608]) and hence
the largest MDE headroom (4.43× / 1.81×), and (iii) the only axis with a **direct Paper B
prior** on the knowledge side (§6.2). It carries the decision alone.

### 5.2 Three secondary axes under Holm–Bonferroni; total family = 4

`triviaqa`, `popqa`, `nq_open` are **secondary**. Sort `p_(1) ≤ p_(2) ≤ p_(3)`; reject `p_(i)`
iff `p_(j) ≤ 0.05/(3−j+1)` for all `j ≤ i`. Thresholds, fixed now:
**`0.016667` / `0.025000` / `0.050000`**.

Holm rather than BH because these three are **corroboration**, where FWER control is the
right object; A04's whole-family `BH q=.05` is for its 96-cell certification grid, a different
job. **The family is declared as exactly these 4 axes. No axis may be added later**, and
`nulls_if_gate_widened` axes (boolq / openbookqa / arc_challenge / piqa / commonsense_qa /
winogrande) are **out of family** — adding one after seeing results is the selective-reporting
move this section prevents.

`nq_open` is **DEMOTED to descriptive-only** by inheritance: A04 records that its item-level
95 % CI half-width (1.459–2.063 pp at `n = 3610`) **already exceeds** its own `Θ = 0.970 pp`.
It is reported and it may **never** carry a kill or proceed decision.

### 5.3 Where multiplicity is NOT applied, and why

* **6 cells:** the 6 cells enter through **one** scalar `Psi` (4 of them) plus **one**
  increment `D(3)−D(1)` (the other 2). There is no per-cell test, so there is nothing to
  correct across cells. This is deliberate: "6 cells × 4 metric families = 24 tests" is how a
  2×3 turns into a fishing expedition.
* **Two clauses (C1, C2):** **not** corrected against each other. Under §0.1's adopted
  `¬C1 ∧ ¬C2` kill rule, requiring **both** to fail makes the kill decision *harder*, so a
  multiplicity correction between them would be correcting in the anti-conservative
  direction. The `关闭条件`-literal secondary verdict (`¬C1 ∨ ¬C2`) **is** an OR of two
  tests, so **that verdict — and only that one — is reported with each clause at
  `α/2 = 0.025`**, stated in its own line.

---

## 6. `effect-size threshold` — two, both anchored to measurements on disk

### 6.1 `Θ_int` for the interaction (clause 1)

```
Theta_int = 0.10 * residual(intact_1B, axis)
```

**`mmlu_content`: `Θ_int = 1.0230 pp = 0.10 retention units`.** Also
`triviaqa 4.0430 pp`, `popqa 1.3210 pp`, `nq_open 0.9700 pp` (descriptive).

Provenance: this is **A04's `delta_noninferiority` constant, transferred unchanged**
(`gate_design.frozen_constants.delta_noninferiority = "0.10 * residual(intact, axis),
per-axis"`). It is **not invented for B03**, and re-using it means the two proposals'
knowledge-side effect sizes are directly comparable. In retention units it is exactly
`0.10` on **every** axis, which is why the contrast is defined in retention units in §1.

**An interaction smaller than 10 % of the intact model's entire above-null knowledge is
declared, in advance, to be scientifically uninteresting for this claim** — because the
surviving claim (`RELATED_WORK.md` §4) is that a *regime boundary* exists, and a boundary
that moves knowledge retention by <10 % of the intact residual does not separate
"single-pass LM pretraining is on the far side" from "monotone uniform degradation".

### 6.2 `Θ_sep` for the dissociation (clause 2)

```
Theta_sep = 0.10 retention units, applied to the INCREMENT D(3) - D(1)
```

Anchoring, from a **measured** in-repo number: `status/PAPERB_KEEP14_200K_EVAL.md` lines
11 / 13 give, at 7B `keep14` step 200k, PPL tax `1.428×` → `R_ppl = 1/1.428 = 0.7005`, and
MMLU above-chance recovery `R_know = 0.1950`. So the dissociation **level** there is
`D = 0.5055` retention units. `Θ_sep = 0.10` is **0.198× that measured level**.

> ⚠️ **`Θ_sep` is ANCHORED, NOT MEASURED.** The quantity the gate tests is the **increment
> in `N`**, and **nobody — us or the literature — has ever measured a dissociation increment
> across reset counts.** `0.10` is set at ~1/5 of a measured *level* on the grounds that an
> increment one-fifth the size of the whole known effect is the smallest thing worth calling
> "widens with `N`". **This is the weakest constant in this pre-registration and is labelled
> as such here so it cannot later be presented as data-derived.**

Power condition, so the threshold is not decorative: `MDE(D(3)−D(1)) = (t_{.975,df} +
t_{.80,df})·√2·σ_D/√S ≤ Θ_sep` requires `σ_D ≤ 0.04014` retention units at `S=3, df=12`
(`0.02983` at `S=2`; `0.04773` at `S=4`). With the knowledge component at
`σ_R(know) = 0.00641`, that leaves a **PPL-side budget of `σ_R(ppl) ≤ 0.03962`** (`0.03693`
if `σ_R(know)` sits at its `χ²` upper 0.01572). **`σ_R(ppl)` is UNMEASURED** — G1 must
produce it, and if it exceeds the budget, clause 2 is reported `UNDERPOWERED` rather than
"no separation".

### 6.3 The direction of clause 2 is pre-committed

Clause 2 survives only if the separation **widens** with `N`:
`D(3) > D(1) > 0` in point estimate **and** `range(D over N∈{1,3})` clears §4.4's floor
**and** `D(3) − D(1) ≥ Θ_sep` with `p ≤ 0.05`. A separation that **shrinks** with `N`, or one
that is large but flat, **does not satisfy clause 2** — `PROPOSAL.md` says
「且随 N/损伤时点扩大」 and that conjunct is binding. It will not be re-read as "separation
exists, therefore survive".

---

## 7. `comparator` — who has to be beaten, and in what sense

Three, all mandatory. The gate has no verdict if any is missing.

1. **Within-regime `N=0` cell** — the *arithmetic* comparator. `R ≡ 1` at `N=0` by
   construction, so each regime's own un-reset continue-train leg is the denominator of its
   own damage. This is what makes `Psi` a difference-*in*-differences rather than a raw
   between-regime comparison, and it is what absorbs any main effect of data regime on
   knowledge (which Muennighoff et al., NeurIPS 2023 `conf/nips/MuennighoffRBST23`, already
   owns and B03 may not claim).
2. **The published-negative prior: `Psi = 0` with monotone degradation in `N`** — the
   *scientific* comparator, and the one the literature predicts. `RELATED_WORK.md`
   `published_negative_priors_that_must_be_stated_before_results` lists **five external**
   (2109.00267 §5 "for large datasets reinitialization does not seem to offer a benefit";
   LLF ICLR 2022 Table A8 losing to baseline as data/baseline strengthen; SEAL CVPR 2023
   `conf/cvpr/SarfiKCKRMB23`; DASH NeurIPS 2024 `conf/nips/ShinO0Y24` App. C.1 "cannot be a
   solution" under stationary data; LoRR full-layer reset "detrimental" at 7B-class) **plus
   one internal** (our Paper B keep14, read as a prior, predicts monotone worsening in `N`).
   **These six must be stated in the verdict document BEFORE its results**, per
   `RELATED_WORK.md` §3 item 11. Beating this comparator means `|Psi| ≥ Θ_int` at `p ≤ 0.05`.
3. **`--from_scratch` at matched depth and matched token budget** — the *ceiling-check*
   comparator. Paper B measured it at 7B (`outputs/olmo2_probe2_7B_keep14fresh2_fromscratch`,
   step200000, held-out PPL 11.498 = 1.554× base vs healed keep14's 1.446×;
   `status/OLMO2_PRUNEHEAL_PPL.md:37`). **No 1B `from_scratch` control is on disk.** So
   either it is added as a 7th cell (+`S`×35.91 GPU-h, §9) **or** the verdict must state
   verbatim: *"no matched-depth from-scratch floor was measured at 1B, so we cannot exclude
   that every cell including `N=0` is at or below what training the same 16-layer shell from
   scratch achieves."* **Pre-registered choice: add it at `S=1` in Tier 1** (one run, 35.91
   GPU-h) — `S=1` because it is a **floor check**, not a term in `Psi`, and it needs no
   variance estimate.

**NOT a comparator, explicitly:** Shrink-and-Perturb (superseded — DASH beats it), and DASH's
own numbers (its protocol is an incrementally-growing dataset trained to 99.9 % train
accuracy, **not** single-epoch streaming pretraining, so it is a prior and reviewer ammunition
only, never a cross-tabulated baseline). Both restrictions are
`related_work_status.design_constraints_the_literature_forces_on_the_future_readout` verbatim.

---

## 8. Protocol invariants, and the two 0-GPU preconditions

### 8.1 Invariants (any violation VOIDS the gate)

| # | Invariant | Why / how checked |
|---|---|---|
| I1 | **All 6+ cells on ONE architecture: `sm_90` (H20 `.73`/`.82`/`.104`).** | `LIFECYCLE_SCHEMA.md` §3: mixing `sm_100` and `sm_90` confounds the primary effect with hardware drift. The 2.02 s/step anchor is 8×H20. **`needs_arch = sm_90`.** |
| I2 | **Matched token presentations.** Every cell sees exactly `T_total = 8000` optimizer steps × 262,144 tok/step = **2.097 B tokens**, regardless of `N` or regime. | `PROPOSAL.md` shared condition. Segments: `N=3` → 4×2000 steps; `N=1` → 2×4000; `N=0` → 1×8000. |
| I3 | **Matched LR schedule and final architecture.** One `--max_steps` horizon for all cells so the cosine is identical; total depth `keep_front + n_fresh` constant. | `PROPOSAL.md`. Note the differential-LR flags: whether the fresh group is actually distinct must be **verified in the log** for each cell — the sibling distill trainer's fresh group is a silent no-op (`CLAUDE.md`), so "differential LR" may not be claimed without a log line. |
| I4 | **Optimizer moments of reset layers are reset with them, and so is the LR scheduler state where applicable.** | `PROPOSAL.md`; and it is **protocol hygiene copied from Active Forgetting (NeurIPS 2023) §3**, cited as such, **never as a contribution** (`RELATED_WORK.md` §2.4, must-not-claim #4). |
| I5 | **Frozen item sets, verified by sha256 against A04's `g0_anchor_sha256_20260810`** before any scoring. | Same-harness re-runs in this repo are **byte-identical** (`memory/same-harness-runs-bit-identical.md`), so any per-item flip needs a named cause. The aggregator must `assert n_scored == expected` **per task**, not merely count NaNs. |
| I6 | **`chat_template = False`, base-model protocol, no BOS, likelihood-based MC.** | Standing repo rule; OLMo-2 is a BASE LM with no SFT/RL. |
| I7 | **The primary p-value is produced by exhaustive enumeration on any node; the secondary bootstrap only on numpy 2.5.1 with `seed=20260815`.** | §2.1. `.82` (numpy 2.4.6) is excluded from the bootstrap. |

### 8.2 G0 — 0 GPU, BLOCKING. The reset operator does not exist yet.

**Measured 2026-08-15 by reading `scripts/train_olmo2_arch_probe2.py`:** the trainer can
*construct* `keep_front + n_fresh` (`transplant_front`, line 170) and can *resume* a
checkpoint (`--resume_from`, line 631; `load_state_dict(..., strict=True)`, line 776), but
**there is no flag that re-initialises the top `K` layers at a resume point.** A cyclic reset
is therefore **not** available from existing flags. It is, however, **not a trainer change**:

> **G0 deliverable (CPU only, no GPU, no node):** a standalone checkpoint-surgery script that
> reads a `step{N}.pt`, re-initialises `model.layers.{keep_front .. keep_front+n_fresh-1}`
> from the **same** `Olmo2ForCausalLM(cfg).post_init()` distribution the trainer uses (never
> hand-built — the docstring's own warning), zeroes `exp_avg`/`exp_avg_sq` and resets `step`
> for exactly those parameter indices in `optimizer_state`, leaves every other tensor
> **byte-identical**, and writes a new ckpt that `--resume_from` strict-loads.
>
> **Mandatory pre-data self-tests, all 0 GPU:**
> `(a)` every non-reset tensor is byte-identical to the input (sha256 per tensor);
> `(b)` the reset layers pass the trainer's own `_assert_fresh_init` (line 140:
> `post_attention_layernorm` all-ones, `q_norm` all-ones, `q_proj.weight` std in the Olmo2
> init band) — **`post_attention_layernorm`, not `input_layernorm`, because OLMo-2 is
> POST-norm**;
> `(c)` the surgical ckpt strict-loads under `--dry_run_build` with **zero** missing/unexpected
> keys;
> `(d)` the `optimizer_state` group count is preserved, so the trainer's 2-group→4-group
> compatibility shim (line ~915) is **not** silently triggered — if it is, the resume is not
> faithful and G0 FAILS.
>
> **If G0 cannot pass (d), B03 stops at 0 GPU** and the finding is "this trainer cannot
> express a faithful multi-cycle reset", which is a real, publishable protocol note and costs
> nothing.

Also in G0, 0 GPU: **data-budget feasibility, already verified.** `data/dolmino_now15b.npy`
is `7,570,911 × 2048 = 15.505 B` tokens. The `SP` cell needs 2.097 B **unique** tokens =
1,024,000 rows = **13.53 %** of the file. The `RD` cell at `E = 8` epochs needs 0.262 B unique
× 8 = 128,000 rows = **1.69 %**. Both fit, and both are expressible with existing flags:
`--max_rows` (line 857-858) plus the trainer's `sampler.set_epoch(epoch)` loop (lines
1015-1028). **No new tokenization, no new data.**

> **`E = 8`, pre-registered, and the reason is a citation.** Muennighoff et al.
> (NeurIPS 2023) find up to **4 epochs** of repeated data ≈ negligible loss change.
> `E = 4` therefore sits **inside their benign band** and would give a "repeated-data" cell
> that is not meaningfully repeated — the interaction would be null **by design choice**, not
> by nature. `E = 8` is past their band. Choosing `E` after seeing a null interaction would be
> unfalsifiable, so it is fixed here.

### 8.3 G1 — the ONE cell that must be timed and varied before Tier 1 is funded

`(SP, N=3)` at `S = 3` = **3 runs, 107.7 GPU-h** (§9). It delivers exactly three things:

* a **measured** s/step for the reset regime (replacing the continue-train anchor);
* the **first `σ_run(K)`** for a reset arm, on all 4 axes — checked against §3.3's
  `≤ 0.2903 pp` (pool-6) / `≤ 0.2773 pp` (corner-4) bound;
* the **first `σ_D`**, hence `σ_R(ppl)`, checked against §6.2's `≤ 0.03962` budget.

**G1 is 14.3 % of the ladder and can kill the gate on power alone.** If either bound fails →
`UNDERPOWERED`, `S` is **not** raised (§3.4), Tier 1/2 are **not** funded.

---

## 9. `gpu_cost_estimate` — derived, with the extrapolation's weakness named

### 9.1 The anchor and its arithmetic

`2.02 s/step` median, `n = 36`, `logs/olmo2_1B_keep7fresh2_1node.log`, `world_size=8 bs=16
gaccum=1 eff_bs=128 seq_len=2048`, `262,144 tok/step` — A04 `cost.measured_anchors.1B_8xH20`.
⇒ `1.712375e-08 GPU-h/token`, i.e. **0.0584 B tokens per GPU-h**.

At `T_total = 8000` steps = 2.097 B tokens: **35.91 GPU-h per run** = 4.49 h wall on one
8-GPU node.

| leg | runs | GPU-h | node-days (1×8-GPU) |
|---|---:|---:|---:|
| **G0** (ckpt surgery + self-tests + data feasibility) | 0 | **0** | 0 |
| **G1** `(SP,N=3)` `S=3` | 3 | **107.7** | 0.56 |
| **Tier 1** 4 corner cells `S=3` (G1's 3 runs reused) | 12 → 9 new | **323.2** new | 1.68 |
| **Tier 1b** `from_scratch` floor check `S=1` (§7.3) | 1 | **35.9** | 0.19 |
| **Tier 2** 2 × `N=1` cells `S=3` | 6 | **215.5** | 1.12 |
| **train total** | 22 | **682.3** | 3.55 |
| eval (4 axes × 22 runs) | — | **~66** (order of magnitude) | — |
| **GRAND TOTAL** | | **~748 GPU-h** | **~3.9 node-days** |

### 9.2 What is weak about this, stated plainly

* The anchor is a **continue-train** rate for a `keep7+fresh2` 1B arm under
  `train_olmo2_arch_probe2.py`. A reset run is the **same** trainer at the **same** shape,
  so the s/step should transfer closely — **but that has not been measured**, which is why
  G1's first deliverable is a measured s/step.
* The eval figure is scaled from A04's `pilot_zero_eval_only_GPU_h = 3` for a comparable
  4-axis pass. It is **not independently re-derived** and is quoted only to order of
  magnitude.
* `16` cards give only `1.36×` throughput for `2×` the GPUs (A04
  `cost.measured_anchors.scaling_note`), so **8 cards is the efficient unit** — extra nodes
  buy parallel cells, not faster cells. Three H20 nodes ⇒ the 22 runs fit in ~1.3 wall-days
  if all three are free.
* **Supersedes `gpu_cost_estimate.value = "UNKNOWN -- 需先做 1-cell 计时"` only in part:**
  that key's `first_action` ("time ONE cell, multiply by 6") is **preserved and adopted** —
  it *is* G1. The `UNKNOWN` string stays on disk unedited (append-only).

### 9.3 Why `S = 3` and not `S = 4`

`S = 4` buys MDE 0.1943 vs 0.2311 pp (headroom 5.26× vs 4.43× at `σ̂`; 2.15× vs 1.81× at the
`χ²` upper) for **+227 GPU-h** on Tier 1+2. Since `S = 3` already clears `Θ_int` at the
**pessimistic** end of the variance interval, the extra `S` buys knowledge of `σ` rather than
a different verdict — A04's `what_seed45_actually_bought` makes exactly this point. `S = 3` is
also the minimum that clears §3.2's exact combinatorial floor.

---

## 10. `kill_if` / `proceed` — the decision, as a boolean

### 10.1 `kill_if` (PRIMARY, the `存活条件` reading of §0.1)

Clause verdicts on the **primary axis** `mmlu_content`:

```
C1_FAILS  ==  p_exact(Psi) > 0.05  AND  |Psi_hat| < Theta_int (= 0.10 retention units
                                                              = 1.0230 pp)
C2_FAILS  ==  p(D(3)-D(1)) > 0.05  AND  |D(3)-D(1)| < Theta_sep (= 0.10 retention units)
              AND  range(D over N in {1,3}) <= c_2 * sigma_D/sqrt(S)      [c_2 = 1.1283791670955126]

KILL  ==  C1_FAILS  AND  C2_FAILS
```

Each clause needs **both** a non-significant p **and** a sub-threshold effect — the B10
`gate_1` form (`not significant at alpha=0.05 AND |delta| < threshold`) verbatim. A
non-significant *large* estimate is **NOT** a kill; it is `UNDERPOWERED`.

**Also reported, mandatory (§0.1):** the `关闭条件`-literal verdict
`KILL_strict == C1_FAILS OR C2_FAILS`, with each clause at `α/2 = 0.025` (§5.3). Both
verdicts appear in the verdict document's first table.

### 10.2 Verdicts that are neither KILL nor PROCEED

| condition | verdict | consequence |
|---|---|---|
| G0 self-test `(d)` fails (optimizer-group shim triggers) | `NOT_EXPRESSIBLE` | stop at **0 GPU**; write the protocol note |
| G1 `σ_run(K) > 0.2903 pp` (pool-6) or `> 0.2773 pp` (corner-4) | `UNDERPOWERED` | stop after 107.7 GPU-h. **`S` is NOT raised** (§3.4) |
| G1 `σ_R(ppl) > 0.03962` retention units | `UNDERPOWERED_C2` | clause 2 is undecidable; clause 1 may still proceed, and the verdict says so |
| `p ≤ 0.05` but `|effect| < Θ` | `SIGNIFICANT_BUT_BELOW_THRESHOLD` | **not** a survival. `Θ` was set in advance precisely to make this outcome uninteresting |
| `p > 0.05` and `|effect| ≥ Θ` | `UNDERPOWERED`, not KILL | the kill needs **both** conjuncts |
| either range fails its §4.4 floor | `UNRESOLVED_SUBNOISE` on that clause | **no ratio of ranges may be quoted** |

### 10.3 `proceed`, and exactly what the first step costs

```
G0   -> 0 GPU.        BLOCKING. Ckpt-surgery script + 4 self-tests + data feasibility (done).
G1   -> 107.7 GPU-h.  ONE cell (SP,N=3) at S=3. Measures s/step, sigma_run(K), sigma_D.
                      Can return UNDERPOWERED and end B03 for 14.3% of the ladder.
T1   -> 323.2 GPU-h new (+35.9 for the from_scratch floor). Completes the 4 corner cells.
                      Decides clause 1.
T2   -> 215.5 GPU-h.  The two N=1 cells. Decides clause 2's growth-in-N conjunct.
                      NOT funded unless clause 1 is decided (either way) at T1.
```

**FIRST STEP AFTER THIS PRE-REGISTRATION = G0, WHICH IS 0 GPU.**

### 10.4 If it survives

The claim is `RELATED_WORK.md` §4's safe residual sentence and **nothing more**: a
**regime-boundary / negative-result** claim. All 11 `must_not_claim` items stay binding, the
six published negative priors are stated **before** the results, and the operator is named
**LLF (Zhou et al., ICLR 2022, `conf/iclr/ZhouVLC22`)** in the method section's first
sentence.

---

## 11. Provenance of every constant in this document

| constant | value | source (path, and md5 where hashed) |
|---|---|---|
| `σ_run` pooled `df=5`, 4 axes | mmlu 0.0656 [0.0409, 0.1608]; triviaqa 0.3666 [0.2288, 0.8992]; popqa 0.2595 [0.1620, 0.6364]; nq_open 0.1445 [0.0902, 0.3543] pp | `proposal/archive/A03-parametric-vs-external-memory/evidence/a03_sigma_run_n3.json`, md5 **`5fb6cd4c3d693831e50d0817bda93ab8`** — **MEASURED (continue-train, see §3.4)** |
| construct nulls + intact 1B + residuals | §4.1 table | `.../evidence/a03_1b_floor_nulls.json`, md5 **`a97a73bf802737601a6057f767b70853`**, as transcribed in `proposal/active/A04-recovery-certification/STATUS.json:nulls_per_metric` — **MEASURED** |
| `ppl(intact_1B) = 10.6416`, `n_windows 4096`, `n_tokens 8,384,512` | — | A04 `nulls_per_metric.in_domain_PPL` — **MEASURED** |
| `Θ_int = 0.10 × residual` | 1.0230 pp on mmlu | A04 `gate_design.frozen_constants.delta_noninferiority` — **TRANSFERRED, pre-existing** |
| `Θ_sep = 0.10` retention units | — | **ANCHORED to** `status/PAPERB_KEEP14_200K_EVAL.md:11,13` (tax 1.428× → `R_ppl` 0.7005; `R_know` 0.1950; level `D = 0.5055`). **NOT MEASURED as an increment** (§6.2) |
| `c_2 = 2/√π` | 1.1283791670955126 | exact closed form; MC 1.1291 (4e6 draws, numpy **2.5.1**, seed 20260815) — **DERIVED + verified** |
| exact-p floors | `S=2: 2/36=0.055556`; `S=3: 2/400=0.005`; `S=4: 2/4900=0.000408` | `C(2S,S)²`, exact combinatorics — **DERIVED** |
| `t` and `F` quantiles | §2.2 tables | hand-written, **6-authority self-test PASSED pre-data 2026-08-15**; no scipy on any node — **DERIVED + verified** |
| `2.02 s/step`, `262,144 tok/step` | ⇒ 35.91 GPU-h per 8000-step run | A04 `cost.measured_anchors.1B_8xH20` (`n=36`, `logs/olmo2_1B_keep7fresh2_1node.log`) — **MEASURED, but for continue-train (§9.2)** |
| `data/dolmino_now15b.npy` = `7,570,911 × 2048` | 15.505 B tokens | `numpy.load(mmap_mode='r').shape`, run 2026-08-15 on LOCAL, CPU only — **MEASURED this session** |
| `E = 8` epochs for the RD cell | — | **CHOSEN**, justified by Muennighoff et al. NeurIPS 2023 `conf/nips/MuennighoffRBST23` ("≤4 epochs ≈ negligible") — a citation, not a measurement |
| trainer facts: no reset flag; `--max_rows` 857-858; `set_epoch` 1015-1028; `_assert_fresh_init` 140; strict resume 776; 2→4 group shim ~915; `seed=args.seed` 869 | — | `scripts/train_olmo2_arch_probe2.py`, **read 2026-08-15** — **MEASURED (source)** |
| `from_scratch` 7B floor: PPL 11.498 = 1.554× base | — | `status/OLMO2_PRUNEHEAL_PPL.md:37` — **MEASURED at 7B; absent at 1B (§7.3)** |
| numpy on LOCAL | **2.5.1** | `python -c "import numpy"` on LOCAL, 2026-08-15 — **MEASURED this session** |

### 11.1 What was NOT done in this pass

* **No GPU.** No `nvidia-smi`, no `ssh`, no job launched, no job killed. `.73` / `.82` /
  `.104` were not contacted; LOCAL and `.212` were not contacted.
* **No reset-regime variance was measured** — §3.4, and it is why G1 exists.
* **No `Θ_sep` increment was measured** — §6.2, labelled ANCHORED.
* **The `keep12` seed arms' position relative to `ce5c298` was not date-verified** per run
  (§3.4). Treated as a caveat that *weakens* the prior, i.e. in the conservative direction.
* **No 1B `from_scratch` control exists** — §7.3 gives the two admissible dispositions and
  pre-registers which one is taken.

---

## 12. Pre-registration statement

> **This gate was written on 2026-08-15, before any B03 GPU run existed at any scale on
> either disk, and before the reset operator it requires existed as code.** Total GPU spent
> designing it: **0**. Every threshold above (`α = 0.05`; Holm `0.016667/0.025/0.05`;
> `Θ_int = 0.10 × residual(intact) = 1.0230 pp`; `Θ_sep = 0.10` retention units; `S = 3`;
> `T_total = 8000` steps; `E = 8`; `c_2 = 1.1283791670955126`; the `F_{.975}(S−1,S−1) = 39.0`
> pooling precondition; the `σ_run(K) ≤ 0.2903 pp` and `σ_R(ppl) ≤ 0.03962` power bounds) is
> **frozen by the commit that adds this file**. If any of them is later changed, **the gate is
> VOID** and must be re-run from the beginning; a changed constant may not be presented as
> the original.
>
> **Completing this pre-registration does not authorise GPU.** B03 remains
> `lifecycle = ready_cpu`, `priority = low`, `status = hold_gate_only`. The next action is
> **G0, which costs 0 GPU**. See `STATUS.json:gpu_policy` for what would have to be true
> before any card is spent.
