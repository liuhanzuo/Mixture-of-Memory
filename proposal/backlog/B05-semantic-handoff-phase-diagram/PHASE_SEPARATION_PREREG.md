# B05 — PHASE SEPARATION PREREG (PRE-DATA)

**Written 2026-08-14. REVISED 2026-08-15 under adversarial refutation.
0 GPU spent. No B05 cell has been run — the phase diagram still has 0 cells.**

This file exists so that the separation criterion is fixed **before** any B05 cell is
inspected. `proposal/README.md` rule 1: 新方向先写 PROPOSAL.md 和 kill gate，再启动 GPU.

The thing this file fixes is the one thing `STATUS.json` admitted was unfixed:
its `exit_clause.unspecified` field says «'不清晰' is not operationalised». Below it is.

---

## REV-1 (2026-08-15) — what changed, and why

The 2026-08-14 draft of this file was **refuted 3/3 by adversarial review**
(`wf_c2a3b490-e40`), and **two lenses returned `decidable=False`** — i.e. the gate as
written could not be evaluated by any procedure. The drafting agent had also written
`lifecycle: ready_gpu` into `STATUS.json` **before** its own gate was reviewed; MAIN
reverted that to `ready_cpu`. This revision applies the fixes. It is still **PRE-DATA**:
`ls ruler_results/b05_native_ruler_*` → no match on 2026-08-15.

| lens | verdict | what it demanded | where it is now |
|---|---|---|---|
| `decidability` | NEEDS_REVISION, **decidable=False** | a CELL→RUNG rule, because a rung has no phase label when its 4 cells disagree | **§3.1(a)** + `code/b05_phase_assign.py:188` (`rung_label`) |
| `falsifiability` | NEEDS_REVISION, **decidable=False** | a monotonicity guard over the ordered rung list `[BELOW,6,9,12,18]` = indices 0..4, so a non-monotone scatter cannot be read as separation | **§3.3** + `code/b05_phase_assign.py:211` (`cell_is_monotone`), promoted to kill clause **K3** |
| `affordability` | NEEDS_REVISION, decidable=True | replace the flat `4 × 0.8144` with a j-weighted sum | **§6** + `code/b05_cost_model.py`; the lens's *mechanism* is confirmed, one *sub-clause* is refuted with measurements — see §6.1 |

**A fourth defect, which no lens raised, is also fixed here: the FLOOR was calibrated on
a false positive.** `FLOOR = 4.0 pp + bootstrap-percentile CI excluding 0` was derived
from A02 calling `−4.0 pp [−8,−1]` SIGNIFICANT. That cell's discordance table is
`n01=4, n10=0`, whose **exact** two-sided McNemar p is **0.125 — not significant.** A
floor calibrated on that call inherits the false positive. §2.1 is rewritten: **exact
McNemar, `FLOOR = 6.0 pp`**, which is the true minimum detectable one-directional effect
at n=100. Reproduce with `python code/b05_phase_assign.py --selftest`.

Every change is pre-data and **directional in both directions**, so this is not
gate-loosening: raising the floor makes the Phase-I-empty kill route *harder* to trigger,
while K3 adds a *new* kill route that the old gate did not have. Both are declared before
any cell exists.

---

## 0. What is already measured, and what B05 is actually adding

A02 already ran **one row** of B05's grid: it varied **split depth j** at a **fixed
readout** (distilled LoRA), on the **retrieval-closed** RULER cells where recall@12 = 99–100 %
so retrieval cannot confound depth.

| A02 arm | j | readout | RULER mean over the 4 retrieval-closed cells |
|---|---|---|---|
| A0 | 0 | native suffix (no adapter) | **99.75** ← anchor |
| A1 | 0 | LoRA r32, 87.29 M | 99.75 (0 flips / 400 vs A0) |
| A2 | 6 | LoRA r32, 72.74 M | 99.25 |
| A3 | 9 | LoRA r32, 65.47 M | 98.25 |
| A4 | 12 | LoRA r32, 58.20 M | 90.75 |
| A5 | 18 | LoRA r32, 43.65 M | 20.75 |
| A6 | 12 | LoRA r40, 72.74 M (cap-matched to A2) | 90.50 |

Source: `proposal/backlog/A02-comem-write-read-repair/evidence/read_tax_ruler/a02_read_tax_ruler.json`
(md5 `fcb32b6dddb90cc5ca64925b5740462b`), all fail-closed gates PASS, n=100/cell, 8/8 shards.

**The column A02 did NOT run is the native-suffix readout at j > 0.** That column is
precisely B05's `readout capacity` axis. At j = 0 the axis is measured and is
**degenerate by construction** — GATE 0 of A02's prereg establishes that at `resume_j=0`
teacher == student, so the "optimal j=0 adapter" is the identity; A1 vs A0 is 0 flips
out of 400 paired items. **j = 0 therefore carries no information about whether readout
capacity is a real axis, and is excluded from every phase-non-emptiness test below.**

### Free prior evidence, and why it is NOT reusable as a paired cell

`ruler_results/qcmem_8b_zeroshot_j9_chatFALSE/` on wzc1 holds an adapter-free
(`lora_adapter: null`, `zero_training_no_adapter: true`) j=9 run at
`selector=iter_bm25 / topk=12 / hop=4 / chunk=512 / sink=bos / chat_template=false / seed=42`
— i.e. the canonical protocol. Merging its 8/8 shards under the canonical
`recall >= 1.0` rule (`scripts/eval_ruler_qcmem.py:720`) gives:

| cell | native j=9 (this dir) | A02 LoRA j=9 (A3) |
|---|---|---|
| niah_multikey_1 16k | 44.00 | 99.0 |
| niah_multikey_1 32k | 59.00 | 95.0 |
| variable_tracking 16k | 18.00 | 99.0 |
| variable_tracking 32k | 16.00 | 100.0 |
| **mean** | **34.25** | **98.25** |

**This is recorded as MOTIVATION, not as a B05 cell**, for two fail-closed reasons:

1. **It emits no `records.json`.** `ls ruler_results/qcmem_8b_zeroshot_j9_chatFALSE/*.records.json`
   → no such file (the dir predates the per-item/`input_ids_sha256` emission at
   `scripts/eval_ruler_qcmem.py:710-754`, mtime 2026-07-24 vs script 2026-08-04). Without
   `input_ids_sha256` there is no GATE C2 pairing, so a paired delta against A02's arms
   would be an **assumed** pairing, not a constructed one. A02's own verdict records that
   its first analyzer version would have *silently passed* on a `None` config read; the
   same class of defect is not being re-admitted here.
2. It does not record `max_new_tokens`, so the `variable_tracking` 60-token override
   (`eval_ruler_qcmem.py:602-603`) cannot be confirmed identical to A02's arms.

**Consequence, stated pre-data:** the native column must be **re-run** with the current
harness. The 34.25-vs-98.25 numbers above are declared here so that nobody can later
claim the gate was designed after seeing them — they are visible, and §3 explains exactly
why they do **not** make the gate un-failable.

---

## 1. The grid B05 will actually run (single variable: readout capacity)

Fixed at the A02 values, byte-for-byte, so that pairing is by construction:

```
model            ../models/Qwen--Qwen3-8b   (L = 36; identical string to the A02 anchors)
selector         iter_bm25    topk 12    iter_hop_topk 4    sink_tokens bos
chunk_size       512          limit 100  num_shards 8       seed 42
chat_template    False        enable_thinking False         bf16 + sdpa + greedy
PRIMARY cells    niah_multikey_1 x {16k, 32k},  variable_tracking x {16k, 32k}
```

Four new arms, **no training, no adapter**:

| B05 arm | flags | output dir (pre-declared) | paired against (A02, on disk) |
|---|---|---|---|
| `N6`  | `--resume_j 6`  (no `--lora_adapter`) | `ruler_results/b05_native_ruler_N6_j6` | `A2` `a02_rtax_ruler_A2_j6` |
| `N9`  | `--resume_j 9`  (no `--lora_adapter`) | `ruler_results/b05_native_ruler_N9_j9` | `A3` `a02_rtax_ruler_A3_j9` |
| `N12` | `--resume_j 12` (no `--lora_adapter`) | `ruler_results/b05_native_ruler_N12_j12` | `A4` `a02_ruler_c2_j12_readlora` |
| `N18` | `--resume_j 18` (no `--lora_adapter`) | `ruler_results/b05_native_ruler_N18_j18` | `A5` `a02_rtax_ruler_A5_j18` |

Output dir names are pre-declared at `code/b05_phase_assign.py:99` (`NATIVE_ARMS`) and
comparator dirs at `:105` (`COMPARATOR_DIRS`), so a later reader can verify from `ls` that none of the four
existed when this gate was written (§4).

Ceiling anchor for both readouts: **`A0`** = `ruler_results/a02_dvr_ruler_j0_top12` (j=0, no adapter).

4 arms × 4 cells = **16 cells**. Zero training steps: the native readout has no
parameters to fit. This is the whole reason B05's first GPU spend is ~3 GPU-h and not ~192.

---

## 2. Floor, and the Δ guard (both fixed pre-data)

### 2.1 Noise floor, and the statistical test — REWRITTEN IN REV-1

Per `memory/same-harness-runs-bit-identical`, a re-run at **same arch + same disk + same
harness** in this repo is **byte-identical (0 flips)**. The "15–20 flip within-disk floor"
that circulated earlier was code-version drift between two harness versions, not runtime
jitter, and is **not** imported here. With arms paired on identical `input_ids_sha256`,
the only remaining variance is sampling error.

#### The test is EXACT McNemar, not the paired bootstrap

Two independent, measured reasons:

1. **The bootstrap is anti-conservative on sparse one-directional discordance, and that
   is exactly the regime here.** In **20 of 20** comparator cells in
   `evidence/read_tax_ruler/a02_read_tax_per_item_vectors.json`, `n10 = 0` — the arm never
   wins an item the anchor lost. On such a table the percentile bootstrap over-rejects.
   Concretely, `ruler|niah_multikey_1|32k` arm `A3`: `n01=4, n10=0`, exact two-sided
   p = **0.125 (ns)**, while A02's bootstrap reported `[−8, −1]` = **SIG**.
2. **The bootstrap is node-dependent, so a phase label could flip with the node.**
   `memory/numpy-version-split-breaks-cross-node-bootstrap` records three numpy versions
   across five nodes (LOCAL 2.3.5 / .82 2.4.6 / rest 2.5.1) and same-seed `multinomial`
   divergence. `math.comb` is exact integer arithmetic: node-invariant by construction.
   A gate whose verdict depends on which node ran the analyzer is not a gate.

The bootstrap CI is still **computed and emitted** for continuity with A02's published
table, but it **never decides a label**. `stat_decides_label` records the deciding test
in every emitted cell, so the substitution is auditable rather than silent.

#### The floor follows from the test, and is not invented

At n = 100 with all discordance one-directional, exact two-sided p = `2^(1−k)` for `k`
discordant pairs. `2^(1−k) < 0.05 ⟺ k ≥ 6`. So:

| \|Δ\| | 3 pp | 4 pp | **5 pp** | **6 pp** | 10 pp |
|---|---|---|---|---|---|
| exact p | 0.250 | 0.125 | 0.0625 | **0.03125** | 0.00195 |
| verdict | ns | ns | ns | **SIG** | SIG |

> **`FLOOR = 6.0 pp` per cell, AND exact two-sided McNemar p < 0.05.** Both conditions,
> not either. A delta below 6.0 pp is declared *indistinguishable* regardless of its
> p-value; a significant p on a sub-floor delta is likewise *indistinguishable*.

This is the **true** minimum detectable effect of the design, whereas 4.0 pp was the
boundary of a test that mis-called a 4/0 table. Verify:
`python code/b05_phase_assign.py --selftest` → asserts MDE == 6 and asserts the A02 cell
re-adjudicates to `ns` with p == 0.125.

**What this costs, stated honestly.** Under the 6.0 pp floor the A02 LoRA comparator
re-adjudicates as: `A2` (j=6) SAME in 4/4 cells, `A3` (j=9) SAME in 4/4 (the −4.0 pp cell
flips SIG→ns), `A4` (j=12) BELOW in 3/4, `A5` (j=18) BELOW in 4/4. So the LoRA column is
`[SAME, SAME, BELOW, BELOW]` over `j ∈ {6,9,12,18}` — Phase II remains *possible* at
j ∈ {6,9} and Phase III at j ∈ {12,18}, and **the gate's power still sits entirely in
whether the native readout survives at j = 6 or j = 9** (§3.4). Raising the floor did not
predetermine any phase.

### 2.1b Read-out is per-cell binary correctness only

No cell may be scored by any quantity other than the per-item `correct` the harness wrote
into `*.records.json`, loaded through the **imported** A02 loaders (GATE E, §4.4).

### 2.2 Δ guard — the denominator can be zero or negative

The anchor `A0` sits at **100.0** on 3 of the 4 retrieval-closed cells (Wilson lo 96.3),
and at 99.0 on the fourth. Therefore:

* **All primary quantities are ABSOLUTE paired percentage points.** No headline number is
  a ratio. Every kill clause in §3.2 is stated in absolute pp; **no clause depends on any
  ratio**, so an ill-defined `R_j` can never block or trigger the gate.
* A recovery fraction `R_j = (LoRA_j − native_j) / (A0 − native_j)` is computed **only if**
  the denominator `(A0 − native_j) >= 10.0 pp` **and** that denominator is itself
  significant under the §2.1 exact test (`p < 0.05`). Otherwise the analyzer must emit
  `{"recovery_fraction": null, "reason": "denominator <NN>pp < 10.0pp or exact p=<P> >= 0.05 -> Δ ill-defined"}`
  and the phase assignment falls back to the absolute-pp rule of §3. This is the case that
  actually bites at `j = 6`, where native may be at the ceiling and the denominator → 0.
  Enforced at `code/b05_phase_assign.py:250` (`recovery_fraction`); the selftest asserts `null` at
  a 0 pp denominator and at a 2 pp denominator, and a real value at 60 pp.
* **Ceiling censoring is declared, not discovered:** an arm scoring 100.0 on a cell whose
  anchor is also 100.0 has an unmeasurable positive Δ. Such cells are flagged
  `at_ceiling_positive_censored` and may support *"indistinguishable"* but may never
  support *"better than"*. Per `memory/a-range-is-not-a-measurement-until-it-clears-its-floor`.

### 2.3 Aggregation hygiene, inherited verbatim

`a02_read_tax_ruler.json`'s own `aggregation_hygiene` field: **"PER-CELL ONLY."** The only
permitted cross-cell number is the mean over the 4 retrieval-closed RULER cells, labelled
as such.

**BABILong / LongBench / LoCoMo are CONTRAST ONLY and may never enter a phase assignment.**
`A02_DEPTH_VS_RETRIEVAL_VERDICT.md` measured recall@12 = 22.9–63.2 % on those cells, so
54.9–78.6 % of the change there is a *retrieval* effect, not a read effect. B05's
"semantic QA" family is therefore **demoted to a contrast arm** by prior measurement, not
by preference. Likewise the "format/verbalization" family sits on a scorer already shown
(`code/analyze_a02_format_mechanism.py`) to order arms by first-period truncation; it is
out of the primary grid.

### 2.4 Two prohibitions carried over from dead work

* **Do not re-run the forward-probe-predicts-graft-depth regression.**
  `proposal/archive/paperC-v1-frozen-cap/POSTMORTEM.md` §4.1 killed it: four forward probes
  on one model span the entire depth domain (0.000L → 1.000L), and the knowledge logit-lens
  that P-C2 was built on regresses at only r=+0.7347 and is systematically too shallow.
  B05's `必须避免` item 3 is that epitaph. Violating it collapses B05 into the archive.
* **Do not report the bm25-era bracket/gap ladder as a B05 result.**
  `status/QCMEM_J_DETERMINATION.md` is `selector=bm25`; the standing rule
  (`memory/qcmem-eval-selector-iterbm25`, user 2026-07-17) voids old bm25 numbers.
  It motivates the grid; it is not evidence in it.

---

## 3. The phase definition, and the KILL GATE

**Everything in §3 is executable.** The reference implementation is
`code/b05_phase_assign.py`, written pre-data and self-tested with
`--selftest`. Where prose and code could drift, **the code is the spec** and the
function name is given inline.

### 3.0 The ordered rung index — the domain every rule below quantifies over

Introduced in REV-1 because the `falsifiability` lens's fix requires an ordered index, and
"ladder rungs" was ambiguous about whether `j=0` was in the domain.

| index | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| rung key | `BELOW` | `6` | `9` | `12` | `18` |
| `resume_j` | 0 | 6 | 9 | 12 | 18 |

`code/b05_phase_assign.py:81` (`RUNG_INDEX`). Index **0** is named `BELOW` and is **Phase I by
construction** (A0 vs A0 = 0.0 pp; A1 vs A0 = 0 flips out of 400 paired items, A02
GATE 0 — teacher==student at `resume_j=0`). It is **excluded from every
phase-non-emptiness test** (`LADDER_J = [6, 9, 12, 18]`) and exists **only** to anchor
the monotonicity check of §3.3, which is undefined without a known-Phase-I left endpoint.
It costs 0 GPU: it is read from the on-disk `A0`/`A1` cells, never re-run.

### 3.1 Phase labels — two levels, CELL then RUNG

#### (a) ★ CELL → RUNG, the rule the `decidability` lens demanded

> **A rung `j` is assigned phase `P` iff ≥ 3 of its 4 primary cells carry label `P`;
> otherwise the rung is `MIXED` and counts toward neither phase.**

Verbatim from the lens's `specific_fix`. Implemented at
`code/b05_phase_assign.py:188` (`rung_label`). Three consequences are pre-registered:

* a **2–2 tie is `MIXED`**, never resolved toward either phase (selftest asserts this);
* **a missing cell counts as disagreement, not as an abstention.** The denominator is
  always `len(PRIMARY_CELLS) == 4`. A cell that is dropped — e.g. because its comparator
  dir lacks `records.json` (§5) — therefore cannot silently lower the bar from 3-of-4 to
  2-of-3 (selftest asserts this too);
* `MIXED` is **reported**, not renamed. A ladder that is mostly `MIXED` is a real result
  and it fires K1.

#### (b) The per-cell labels

At each rung, from absolute paired pp vs the `A0` anchor, using the §2.1 exact test with
`FLOOR = 6.0 pp` (`code/b05_phase_assign.py:163` (`cell_label`)):

| phase | condition on ONE cell |
|---|---|
| **I — joint readability** | native_j **and** LoRA_j both indistinguishable from `A0`, **and** native_j vs LoRA_j itself indistinguishable |
| **II — handoff band** | LoRA_j indistinguishable from `A0`, **and** native_j **BELOW** `A0` (≥ 6.0 pp **and** exact p < 0.05) → readout capacity is load-bearing at this depth |
| **III — joint failure** | **both** native_j and LoRA_j BELOW `A0` |
| **`MIXED_CELL`** | anything else — notably LoRA BELOW while native is not, which is **not a named phase** and is recorded rather than renamed |

"BELOW" and "indistinguishable" are the two outputs of
`code/b05_phase_assign.py:137` (`classify_vs_anchor`), which requires **both** the floor and the
p-value and emits `stat_decides_label` for audit.

### 3.2 KILL GATE — THREE clauses, pre-registered, fires on the 16-cell read-out

> **B05 dies as a standalone paper if ANY of K1, K2, K3 trips, on the 4
> retrieval-closed RULER cells (`niah_multikey_1` × {16k,32k},
> `variable_tracking` × {16k,32k}) at Qwen3-8B:**
>
> **K1 — COUNT.** Fewer than **3** of the phases {I, II, III} are non-empty over
> `j ∈ {6, 9, 12, 18}`, where a rung's phase comes from the ≥3-of-4 rule of §3.1(a) and
> `MIXED` rungs count toward nothing.
>
> **K2 — CROSS-TASK.** The Phase I/II boundary index differs by **≥ 2** between
> `niah_multikey_1` and `variable_tracking`, measured in the §3.0 rung indices (0..4),
> over that task's **monotone cells only** (§3.3). **A task with no located boundary
> fails K2 rather than passing it by default** — either because none of its cells is
> monotone, or because its cells never leave Phase I, which means the boundary lies
> deeper than `j=18` and was therefore *not located inside the measured ladder*.
>
> **K3 — SCATTER (new in REV-1).** Fewer than **3** of the 4 primary cells are monotone
> in rung index per §3.3.
>
> On firing, B05's pre-existing exit clause executes: fold into a Paper A/B mechanism
> subsection, do not make it standalone. It does **not** mean the direction was wrong —
> it means readout capacity and split depth are not separable into named phases at this
> resolution, which is itself a reportable negative for Paper A.

Implemented at `code/b05_phase_assign.py:275` (`adjudicate`); the emitted evidence JSON carries
`clauses.{K1_count,K2_crosstask,K3_scatter}`, `kill_gate_fired`, and
`kill_gate_reasons`.

### 3.3 ★ MONOTONICITY GUARD — the fix the `falsifiability` lens demanded

The lens's objection: without it, "a non-monotone scatter [can] be read as separation."
Its `specific_fix` names the ordered index list `[BELOW, 6, 9, 12, 18]` with indices 0..4
and requires the phase assignment to be **monotone in that index**. §3.0 defines the
index; this is the requirement:

> **A cell is MONOTONE iff its phase SEVERITY is non-decreasing along rung indices 0→4,
> with `SEVERITY = {I: 1, II: 2, III: 3}`.**
> **A label with no severity (`MIXED_CELL`, `UNDEFINED`) makes the cell NON-MONOTONE.**

`code/b05_phase_assign.py:211` (`cell_is_monotone`). The second sentence is load-bearing and is
the reason the guard is not itself gameable: **an unrankable label is not skipped.**
Skipping it would let a hole in the ladder read as a clean ordering — precisely the
failure the lens named.

The guard does two jobs, and the second is why it also became a kill clause:

1. **It gates K2's inputs.** A non-monotone cell **votes on no boundary**
   (`adjudicate`, boundaries loop). A boundary read off a scatter is meaningless, so the
   scatter must not be allowed to supply one — including the case where it would have
   supplied an *agreeing* one and thus made K2 pass by accident.
2. **It is its own clause, K3.** A guard that only invalidated K2 would leave a total
   scatter able to satisfy K1 — each rung could still win a ≥3-of-4 majority while the
   *ordering across* rungs was pure noise, i.e. exactly "a phase diagram retrofitted onto
   two unrelated curves". K3 closes that.

Boundary index per task = **`min` over that task's monotone cells** of the first index at
which the cell leaves Phase I (`cell_boundary_index`, line 235). The `min` rule is pre-registered so
that a task whose 16k and 32k cells disagree cannot be silently averaged into a
half-rung.

### 3.4 Where the gate's discriminating power sits — stated honestly

I am required to construct a failing counterexample before believing this is a gate.
Taking each phase in turn, under the REV-1 floor (6.0 pp + exact):

* **Phase III is already non-empty and will not fail.** A02's `A5` (j=18, LoRA) is BELOW
  the anchor in **4/4** cells (−68/−57/−96/−95 pp, exact p < 1e-5 each). If LoRA is dead
  at j=18, native cannot be alive there. **This clause carries no power and I am not
  pretending it does.**
* **Phase II is very likely non-empty**, given §0 (native j=9 ≈ 34.25 vs LoRA j=9 = 98.25,
  and `A3` re-adjudicates as SAME-as-anchor in 4/4 under the new floor, so the LoRA side of
  the Phase II condition holds at j=9). Low power. Also declared.
* **Phase I is the coin flip, and it is where K1 can actually kill.**
  Phase I requires a rung `j ≥ 6` at which native readout is *still intact*. The evidence
  genuinely does not settle it:
  - `status/QCMEM_J_DETERMINATION.md` places the 8B zero-shot readout cliff at
    j9 = 100 / j10 = 81 / j12 = 9 on **niah_single 16k** → suggests native survives j=6, so
    Phase I would be non-empty.
  - But the §0 merge shows native j=9 on the **harder retrieval-closed** cells is already
    44 / 59 / 18 / 16 — far below `niah_single`'s 100. So the cliff on *these* cells is
    **shallower than j=9**, and whether it is shallower than **j=6** is unmeasured.
* **K3 is genuinely open in both directions**, and this is new information the old gate
  could not have used: A02's own LoRA ladder is **non-monotone in raw accuracy** at
  `niah_multikey_1 32k` — `j6 = 99.0, j9 = 95.0, j12 = 96.0` — i.e. it *recovers* by 1 pp
  from j9 to j12 (`a02_read_tax_ruler.json`, `PRIMARY_ruler_per_cell`). Under the 6.0 pp
  floor that wobble is *inside* the floor and both rungs label as `I`, so this particular
  cell stays monotone. But it demonstrates that per-cell non-monotonicity of the size K3
  cares about is a real behaviour of this measurement, not a hypothetical, and the native
  column has no reason to be smoother than the LoRA column.

#### Three concrete failing results, one per clause

Each of these is asserted in `code/b05_phase_assign.py:430` (`selftest_gate`), so they are not
rhetorical — they are executed:

* **K1 fires** if `N6` comes back at ≈ 78 / 85 / 61 / 58 (and the deeper rungs at or
  below that). Then every rung in {6,9,12,18} has native BELOW `A0` by ≥ 6.0 pp with
  exact p < 0.05, Phase I is **empty on the ladder**, only II and III are populated,
  `2 < 3`, kill. The read tax would be a two-regime cliff — which Paper A already reports
  — and B05 would have no third phase to name.
* **K2 fires** if `N6` = 99 / 99 / 62 / 59 with `N9` = 99 / 96 / 18 / 16. Then
  `niah_multikey_1` leaves Phase I at index 3 while `variable_tracking` leaves it at
  index 1 — a gap of 2 — and the cross-task clause fires **even though three phases exist
  on the pooled view.** This is the clause that stops "phase diagram" from being
  retrofitted onto two unrelated per-task curves.
* **K3 fires** if native *dips and recovers*, e.g. `N6` ≈ 40 / 42 / 38 / 41 but
  `N9` ≈ 99 / 99 / 99 / 100. Every cell then goes `I → II/III → I → …`, severity
  decreases somewhere in 4/4 cells, `0 < 3`, kill — and correctly so, because a
  non-monotone readout curve is not a phase boundary no matter how many phase labels
  appear in it.

All three outcomes are plausible on the on-disk numbers, so the gate is falsifiable.

---

## 4. Read-out point (pre-registered; this is the decision point, not a step count)

B05 is offline eval, so there is no training step. The read-out is a **completeness
condition**, and it is enforced fail-closed by the driver, not by inspection:

1. **All 16 cells at 8/8 shards and n == 100 each.** Fork of
   `proposal/backlog/A02-comem-write-read-repair/code/run_a02_read_tax_eval.sh:174` —
   `if [ "$have" -ne "$want" ]; then note "  ABORT ruler $NAME: only $have/$want records"; return 9; fi`
   — which aborts the arm rather than merging a partial cell. This gate was
   **negative-tested in A02**, not merely asserted (deleting a shard produced
   `G1_SHARD_INCOMPLETE 7/8`), and it fired in production on unfinished cells.
2. **GATE C2 `input_ids_sha256` pairing PASS across all 4 native arms and their 4 LoRA
   comparators and `A0`** (9 arms), joined by `sample_index`. Analyzer:
   `code/analyze_a02_read_tax.py` (asserts sha equality, fail-closed;
   `N_EXPECT = 100` at line 90).
3. **GATE D config identity PASS**: `resume_j` as expected per arm, `lora_adapter is None`
   for all four native arms, `selector=iter_bm25`, `topk=12`, `iter_hop_topk=4`,
   `chunk_size=512`, `chat_template=False`. Read from the level each field actually lives
   at (A02's recorded bug: RULER stores config flat and carries no `chat_template`; that
   lives in the sibling summary — comparing `is not False`, never `is not True`).
4. **GATE E**: scorers **imported** from `analyze_a02_depth_vs_retrieval`, never
   reimplemented. Enforced at `code/b05_phase_assign.py:550`
   (`import analyze_a02_depth_vs_retrieval as dvr`, inside a `try` that **exits 2** rather
   than falling back to a local reimplementation if the canonical loaders are not
   importable).
5. **GATE F (new in REV-1) — the adjudicator's own selftest must PASS on the run node
   before the read-out.** `python code/b05_phase_assign.py --selftest` asserts the 6.0 pp
   MDE derivation, the ≥3-of-4 CELL→RUNG rule including the 2–2-tie and dropped-cell
   cases, the monotonicity guard, the denominator guard, and one triggering result per
   kill clause. It needs **no B05 data and no GPU**, so there is no excuse for running it
   after the fact.

**The read-out point, precisely:** the phase assignment of §3 is produced by exactly one
invocation of `code/b05_phase_assign.py --out <evidence_dir>`, after conditions 1–5 all
PASS. `adjudicate()` (`code/b05_phase_assign.py:275`) is the single function that emits
`kill_gate_fired`, and it is called **once**. No intermediate peeking at partial cells
decides anything. This file must be committed with a hash whose timestamp precedes the
mtime of the first B05 result file; if it is not, the prereg is void and the run is
descriptive only.

**Pre-data status assertion, checkable by anyone:**
`ls ruler_results/b05_native_ruler_*` → *no match* (verified on wzc1, 2026-08-15). The
four output dir names are pre-declared at `code/b05_phase_assign.py:99` (`NATIVE_ARMS`),
so a later reader can confirm those dirs did not exist when the gate was written.

---

## 5. Node / architecture requirement (not a preference)

**sm_90 (H20: `.73` / `.82` / `.104`, zwfy6 disk). Not B200.** Three independent reasons:

1. **The comparator was measured on sm_90.** `A02_READ_TAX_RULER_VERDICT.md:3` —
   "78 min wall, `.82` only (8× H20, zwfy6)". B05's native column is paired *against those
   exact arms*. Same-harness paired comparison must not straddle architectures, or stack
   drift confounds with hardware drift.
2. **The comparator's raw cells live on zwfy6.** `ruler_results/a02_rtax_ruler_*` and
   `ruler_results/a02_dvr_ruler_j0_top12` are recorded as zwfy6-resident
   (`A02_READ_TAX_RULER_VERDICT.md:324`) and are **absent on wzc1** (verified from LOCAL:
   `ls -d ruler_results/a02_*` → no match).
3. **The driver and analyzer are sha256-verified byte-identical on zwfy6** after `scp -O`
   (verdict lines 14–18), so no code transfer step is needed there.

No adapters are needed for B05's own arms (native readout = no LoRA), so the fact that 5
of A02's 6 adapters are wzc1-absent does not block B05 — but the **comparator result dirs
do** need to be present, which is why the node must be zwfy6.

⚠️ **Verify-before-dispatch (I could not check this from LOCAL):** `/apdcephfs_zwfy6` is
not mounted on this node (`ls: cannot access '/apdcephfs_zwfy6'`). Every zwfy6 path above
is **recorded-only from here**. Per `memory/two-disk-rule-applies-to-main-too`, the
dispatching agent must `ls` on `.73`/`.82`/`.104` and confirm all five comparator dirs
(`a02_dvr_ruler_j0_top12`, `a02_rtax_ruler_A2_j6`, `a02_rtax_ruler_A3_j9`,
`a02_ruler_c2_j12_readlora`, `a02_rtax_ruler_A5_j18`) each carry 8/8 `*.records.json`
**before** launching. If any comparator dir lacks `records.json`, that rung's pairing is
impossible and the rung must be dropped from the ladder *and the drop recorded here*.

---

## 6. Cost — j-WEIGHTED, from measured anchors only (REWRITTEN IN REV-1)

The `affordability` lens refuted the flat `4 × 0.8144` model. It is right that a flat
multiply is the wrong model, and §6.2 replaces it. §6.1 first records the one sub-clause
of the lens that the measurements refute, because that refutation is what determines the
*sign* of the correction.

Everything below is reproduced by `python code/b05_cost_model.py` (0 GPU; reads only
on-disk JSON). `--json` emits the machine-readable form.

### 6.1 The lens's mechanism is confirmed; its stated cause is not

> lens `affordability`: *"replace the flat '4 arms x 0.8144 = 3.26 GPU-h' with the
> j-weighted sum, because write runs layers[0:j] over the full context so cost grows with
> j (approx_tokens differ per rung)."*

**CONFIRMED — write cost does grow steeply with j.** Measured, native readout, no adapter,
`chunk_size=512`, `topk=12`, Qwen3-8B, `ruler_results/pareto_jsweep/bench_j{0,6,9,12}.json`
→ `rows[].qcmem.write_s`:

| L | j=0 | j=6 | j=9 | j=12 | linear fit |
|---|---|---|---|---|---|
| 16k | 0.0482 | 0.4443 | 0.7277 | 0.9152 s | `0.0735·j + 0.038`, r = 0.9975 |
| 32k | 0.0602 | 1.2124 | 1.4289 | 2.0559 s | `0.1612·j + 0.101`, r = 0.9908 |

That is **19×** (16k) and **34×** (32k) growth from j=0 to j=12. A flat model does hide it.

**REFUTED — "approx_tokens differ per rung." They do not.** The same files record
`rows[].qcmem.seq_len == 6657` at **every** j ∈ {0,6,9,12} and **every** length ∈
{8k,16k,32k,64k,128k}. The read pack is a fixed `topk=12 × chunk=512` budget; `j` changes
**which layers run**, not **how many tokens**. So the per-rung ratio is *not* a token-count
ratio, and a cost model built on `approx_tokens` would be wrong. Asserted in
`code/b05_cost_model.py:95` (`bench_rows`) and reported as `fit.<L>.seq_len_invariant_across_j`.

**DECISIVE, and the reason the flat estimate did not blow up:** `write_s` is a *small
share* of the per-query total, and the *dominant* term, `decode_s`, **falls** with j —
resuming at layer j means each of the 20–60 decode steps runs only `layers[j:L]`. Measured
per-query totals (`write+select+read+decode`):

| L | j=0 | j=6 | j=9 | j=12 |
|---|---|---|---|---|
| 16k | 21.39 | 18.70 | 17.38 | **16.13 s** |
| 32k | 21.27 | 19.48 | 18.17 | **17.20 s** |

End-to-end cost per query **decreases monotonically in j on every measured rung.** So a
j-weighted sum over *measured totals* would come out **below** the refuted flat figure.

### 6.2 The model actually adopted — a deliberately conservative ceiling

Reporting a number *lower* than the refuted one would look like evading the lens, so the
adopted model does **not** credit the falling terms. Per rung and length:

```
k_ceiling(j, L) = [ write_s fitted at j (floored at its j=9 value)
                    + (select_s + read_s + decode_s) PINNED at their j=9 values ]
                  / (per-query total at j=9)
```

i.e. **write is allowed to grow on its measured slope** (including a linear extrapolation
to j=18, which is unmeasured and flagged as such), while **every term measured to fall
with j is pinned at j=9 instead of being credited.** That is an upper bound on each rung by
construction. The timing anchor sits at j=9 because that is the only on-disk *native* run
on exactly these 4 cells with the real harness.

**Anchor (direct, same 4 cells, same native readout, 8 shards):**
`ruler_results/qcmem_8b_zeroshot_j9_chatFALSE/{cell}_shard*of8.json` → `elapsed_seconds`,
max over the 8 shards × 8 GPUs:

| cell | max shard s | GPU-h / cell @ j=9 |
|---|---|---|
| niah_multikey_1 16k | 37.0 | 0.0822 |
| niah_multikey_1 32k | 70.2 | 0.1561 |
| variable_tracking 16k | 67.6 | 0.1502 |
| variable_tracking 32k | 191.6 | 0.4259 |
| **per arm (4 cells) @ j=9** | | **0.8143** |

**The j-weighted sum:**

| arm | j | k_ceiling 16k | k_ceiling 32k | GPU-h |
|---|---|---|---|---|
| `N6` | 6 | 1.0000 | 1.0000 | **0.8143** |
| `N9` | 9 | 1.0000 | 1.0068 | **0.8183** |
| `N12` | 12 | 1.0110 | 1.0334 | **0.8363** |
| `N18` | 18 | 1.0364 | 1.0866 † | **0.8732** |
| **total** | | | | **3.3421 GPU-h** |

† j=18 has no measured bench row; its multiplier extrapolates the fitted `write_s` slope.

**3.3421 GPU-h**, vs the refuted flat **3.2572** — the j-weighting *raises* the estimate by
**+0.085 GPU-h (+2.6 %)**, exactly as the lens's mechanism predicts, and it does so from a
model that is an upper bound rather than a best guess.

**Anchor 2 (independent cross-check, sm_90):** `A02_READ_TAX_RULER_VERDICT.md:3` — 78 min
wall × 8 H20 = 10.4 GPU-h for 5 arms × 10 cells = 50 cells → **0.208 GPU-h/cell**;
16 cells → **3.33 GPU-h**. Agrees with the j-weighted total to **0.4 %**.

### 6.3 Budget and headroom, from the harness's own token caps

The bench decodes exactly `n_decode = 20` tokens, but the harness decodes up to
`max_new_tokens = 48`, overridden to **60 for `variable_tracking`**
(`scripts/eval_ruler_qcmem.py:602-603`). A **collapsed** arm is the case most likely to run
to the cap without emitting a stop token, and `N18` is expected to be collapsed (its LoRA
comparator `A5` is at 4–42 % accuracy). Worst case, if every item burns its full cap:

| cell | cap | worst-case multiplier |
|---|---|---|
| niah_multikey_1 16k / 32k | 48 | 2.277 / 2.227 |
| variable_tracking 16k / 32k | 60 | **2.824** / 2.753 |

> **BUDGET = 9.5 GPU-h** (= 3.3421 × 2.824, the j-weighted ceiling × the worst per-cell
> decode-cap inflation). **Point estimate 3.34 GPU-h ≈ 25 min wall on one 8×H20 node.**

The budget is now derived from a measured harness parameter rather than a round "50 %
headroom". Compare `N1:6`'s pre-data guess of "1 node ~1 day" ≈ 192 GPU-h — still a
**~57×** overestimate, entirely because that estimate assumed the 4-readout × 4-task
64-cell grid rather than the one column that is actually decisive and requires no training.

**Training steps: ZERO.** The native-suffix readout has no parameters to fit.

---

## 7. What is explicitly NOT in this gate

* No `small decoder` readout column — it has no measured rate anywhere in this repo, so
  costing it would be an invented number. Dropped as a legitimate narrowing.
* No `affine/tuned lens` column in the *primary* gate. It is nearly free (`extract_sec: 5.9`
  in `results/knowledge_logit_lens_Qwen3-8b-local.json`) but it is a **descriptive band**,
  never a predictor of graft depth (§2.4).
* No second model family. Cross-family replication is a *post-gate* extension and would
  cost 2.91 GPU-h per LoRA comparator adapter (measured: `outputs/qcmem_distill_qwen_j12_r32_4k`,
  4000 steps, `world_size=8`, ~24.5 samp/s per `logs/qcmem_distill_qwen_j12_r32_4k.log:421`;
  mtimes 00:04:02 → 00:25:50 = 1308 s wall × 8 = 2.91 GPU-h). It is not in the kill gate.
* No `parametric knowledge` phase cell (dead-probe prohibition, §2.4).

---

## 8. Residual risks this gate does NOT eliminate (REV-1, stated pre-data)

Listing these is not hedging — each is a specific way the read-out could come back
uninterpretable, and each is declared now so it cannot be discovered later as an excuse.

1. **Power at n=100 is coarse, and the 6.0 pp MDE makes it coarser.** The gate can
   distinguish "intact" from "≥ 6 pp worse" but cannot resolve a genuine 3–5 pp handoff
   band. If the true native curve degrades gently rather than cliff-like, the ladder will
   read as `I, I, III, III` and K1 will fire on an *absence of resolution* rather than on
   an absence of structure. **Mitigation is pre-registered rather than post-hoc:** if K1
   fires with **zero** `MIXED` rungs and **all 4 cells monotone**, the verdict record must
   state that the failure mode was resolution, and the exit clause's Paper A/B subsection
   must report it as "no third regime detectable at 6 pp / n=100" rather than "no third
   regime exists". Increasing n is a **post-gate** decision and is not funded here.
2. **`j = 18` cost is extrapolated, not measured.** `pareto_jsweep` stops at j=12. §6.2's
   `k_ceiling(18, L)` extends the fitted `write_s` slope. If the real j=18 write cost is
   superlinear the arm could overrun; the 9.5 GPU-h budget of §6.3 absorbs a 2.8× overrun,
   which is far larger than any plausible superlinearity over 6 extra layers.
3. **The comparator dirs are recorded-only from wzc1.** §5's verify-before-dispatch step is
   mandatory and unresolved. **Dropping `j=6` or `j=9` would remove the Phase I/II boundary
   and therefore the gate itself** — in that case B05 must return to `ready_cpu` and a new
   gate must be written, not run a 3-rung version of this one.
4. **A02's per-item vectors are the pairing substrate, and they were emitted by an earlier
   analyzer version.** `md5 3044dbf9f9e9929921c359b1dffe1ced` is asserted before use; if it
   does not match on the run node, the pairing is not the one this prereg was written
   against and the run is descriptive only.
5. **This gate tests ONE model (Qwen3-8B) and ONE task family (retrieval-closed RULER).**
   A clean 3-phase result is a claim about Qwen3-8B on those cells, nothing wider.
   Cross-family replication is explicitly post-gate (§7) — per
   `memory/direction-a-eval-fragility-established`, a single-family result must be labelled
   as such from the start, not narrowed after a cross-family check fails.
