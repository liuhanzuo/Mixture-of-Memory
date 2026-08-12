# A02 — Is the BABILong misordering a finding? PRE-REGISTRATION

**Written**: 2026-08-12, **BEFORE any statistic in Jobs 1/2 was computed.** What was
inspected first is limited to *structure* (JSON key layout, index joinability, n per
cell, `gold_locatable` counts) and to *already-published* numbers from the two prior
verdicts. No A4-vs-A5 contrast, no conditional accuracy, no VT recall value was looked
at before this file was written.

**Node**: `.82` only (verified 8/8 cards 0 % / 0 MiB, no compute procs, at dispatch).
Jobs 1 and 2.1 are **pure CPU re-analysis of per-item vectors already on disk** — zero
GPU. Job 2.2 may spend GPU on `.82` only, budget ≤ 30 GPU-h.

---

## 0. What is already established and is NOT re-derived here

From `A02_READ_TAX_RULER_VERDICT.md` (commit `3c9f8f9`), independently re-verified by
the dispatcher:

| arm | j | RULER mean | tax vs A0 | LoRA params |
|---|---|---|---|---|
| A0 | 0 | 99.75 | anchor | 0 |
| A1 | 0 | 99.75 | 0.00 | 87.29 M |
| A2 | 6 | 99.25 | −0.50 | 72.74 M |
| A3 | 9 | 98.25 | −1.50 | 65.47 M |
| A4 | 12 | 90.75 | −9.00 | 58.20 M |
| A5 | 18 | 20.75 | **−79.00** | 43.65 M |
| A6 | 12 r40 | 90.50 | −9.25 | 72.74 M |

Cliff not slope (Spearman ρ = −1.000, p = 1.4e-24); survives exact capacity matching
(A2 vs A6, both 72,744,960 params, −8.75 pp). **These are inputs, not questions.**

## 0.1 Structural facts verified before pre-registering (no outcome content)

* Per-item vectors `a02_read_tax_per_item_vectors.json`: 10 cells × 7 arms × n=100.
* dvr per-item `a02_depth_vs_retrieval_per_item.json`: same 10 cells, each with
  `recall_per_sample[{sample_index, gold_chunks, n_ctx, hit, n_sel}]`.
* **Index joinability is exact**: |common_idx ∩ recall sample_index| = 100/100 on
  **all 10 cells**. So the conditioning join needs no imputation.
* `gold_locatable` per cell: qa1 95/100, qa1×32k 100, qa2×16k 95, qa2×32k 96,
  qa5×16k 92, qa5×32k 95, niah_mk1×16k 97, niah_mk1×32k 100, **VT 0 / 0**.
* The retrieval pack is **arm-independent by construction**: all 7 arms ran the same
  `--selector iter_bm25 --topk 12 --iter_hop_topk 4 --chunk_size 512` on prompts whose
  `input_ids_sha256` are asserted equal across arms (GATE C2, 0 failures). Therefore a
  HIT/MISS label is a property of *(cell, sample)*, not of the arm — which is what makes
  conditioning on it legitimate rather than post-hoc arm selection.

---

## 1. JOB 1 — the claim under test

The read-tax verdict §4 asserts, without inference:

> BABILong **misorders** the arms: it ranks A5 (j=18, RULER 20.75) at or above A4
> (j=12, RULER 90.75) on **4 of 6 cells**. Any depth conclusion drawn from these cells
> would have been not merely attenuated but **wrong in sign**.

"4 of 6" is a **count of point estimates**. Three distinct claims hide inside it, and
they have very different strengths. I pre-commit to reporting which one holds.

| # | claim | what would establish it |
|---|---|---|
| **S (strong)** | BABILong *significantly inverts* the ordering | ≥1 cell where A5 > A4 with the paired CI95 excluding 0 (equivalently McNemar exact p < 0.05 in the A5-favouring direction) |
| **M (medium)** | BABILong *fails to recover* the true ordering that RULER resolves decisively | the ladder rank-correlation is destroyed on BABILong (per-cell Spearman(j, acc) not significantly negative) while RULER's is ρ = −1.00; and the A4-vs-A5 contrast is n.s. on the cells where the point estimate inverts |
| **W (weak / ordinary)** | BABILong is merely *underpowered* | the inversion is n.s. everywhere **and** dissolves (point estimate returns to the true sign) once conditioned on retrieval-HIT items |

**S is a benchmark-validity finding. M is a benchmark-validity finding of a weaker
kind (a benchmark in wide use cannot order a 70 pp manipulation). W is ordinary and
not publishable on its own.**

### 1.1 Statistics (fixed now)

1. **Per-cell A4-vs-A5 paired contrast**, all 10 cells (4 RULER as positive control,
   6 BABILong as the claim): **McNemar exact** (two-sided binomial on discordant
   pairs b vs c, exact, no continuity correction, no chi-square approximation) **plus**
   the project-canonical paired bootstrap CI95 (n_boot=5000, seed=42) so the number is
   comparable to every other A02 table. Report b, c, n_discordant per cell.
2. **Per-cell ladder rank test**: Spearman ρ between j ∈ {0, 6, 9, 12, 18} (arms
   A0, A2, A3, A4, A5 — the r=32 ladder, A1/A6 excluded as controls) and per-cell
   accuracy. n=5 ⇒ **exact permutation** p over all 120 orderings (min attainable
   two-sided p = 2/120 = 0.0167). Same computation on RULER cells as control.
3. **Multiplicity**: 6 BABILong cells per family. I report both raw p and
   Holm–Bonferroni-adjusted p within each family of 6, and state which claims
   survive adjustment.

### 1.2 Mechanism test — retrieval domination

dvr measured recall@12 = 22.9–63.2 % on these BABILong cells vs 99–100 % on RULER.
Proposed mechanism: **if retrieval fails first, the read tax cannot express itself** —
on a MISS item the gold chunk is absent from the pack, so every arm is answering from
a pack that cannot support the answer, and arms become indistinguishable regardless of
their read quality.

**Testable consequence, pre-registered**: the A4-vs-A5 separation should be
**larger on HIT items than on MISS items** in every cell, and the point-estimate
inversion should be **confined to (or much stronger on) the MISS subset**.

4. **Conditional contrast**: recompute statistic 1 on the HIT subset and the MISS
   subset separately, per cell. Report n per subset.
5. **Recall-vs-misordering association across cells**: Spearman between per-cell
   recall@12 and per-cell (A4 − A5) signed separation, over the 6 BABILong cells,
   exact permutation p. Prediction: **positive** (higher recall ⇒ the true ordering
   re-emerges). RULER's two locatable cells are plotted as the recall≈100 % endpoint
   but excluded from the 6-cell test to keep the family homogeneous.

### 1.3 Falsification conditions — committed BEFORE computing

* **F1 — the misordering is not significant anywhere.** If no BABILong cell has A5 > A4
  with CI95 excluding 0 (McNemar exact p ≥ 0.05 in the A5-favouring direction on all 6),
  then **claim S is REFUTED** and I will say so in exactly those words. I will *not*
  restate a non-significant inversion as "misorders" without qualification.
* **F2 — the misordering dissolves under conditioning.** If, on the retrieval-HIT
  subset, the point estimate (A4 − A5) returns to the true (positive) sign in the cells
  that inverted, then **BABILong is not invalid — it is underpowered on the wrong
  subset** (claim W), a weaker and more ordinary claim. I will report W as the finding
  and will *not* dress it as benchmark invalidity.
* **F2′ — power guard on F2 (anti-self-deception).** The HIT subsets are small by
  construction (as low as ~22 items at qa2×32k). **A non-significant HIT-subset result
  is NOT evidence that the misordering dissolved.** F2 is adjudicated on the
  **sign of the point estimate and the CI**, and I will additionally report, per cell,
  the minimum effect the HIT subset could have detected. If the HIT subset is too small
  to distinguish "dissolved" from "unresolvable", I will record **INCONCLUSIVE** for
  that cell rather than claiming either.
* **F3 — floor effect as an alternative explanation.** If the inverting cells are
  exactly the ones where A0's own accuracy is near the floor (qa2×32k spans 1–11 %),
  the mechanism is **floor compression**, not retrieval domination per se. I will test
  the two explanations separately (statistic 5 for retrieval; per-cell A0 accuracy vs
  separation for floor) and report if they are **mutually confounded** — with 6 cells
  and both regressors, they may well be, and if so I will say the mechanism is
  **not identified** rather than picking the one I prefer.
* **F4 — the ordering claim is vacuous if BABILong cannot order *anything*.** Control:
  if BABILong also fails to order the *large, uncontested* A0-vs-A5 contrast (RULER
  −79 pp), that reframes the finding from "misorders the depth ladder" to "these cells
  have no ordering power at all in this regime" — related but distinct. Reported either way.

### 1.4 Ownership decision (criteria fixed now, so the answer is not chosen by taste)

Candidates: **A02**, **B04-eval-fragility**, **new backlog proposal**. Decided against
what each proposal *claims*, not who produced the number:

* → **B04** iff the finding is an instance of B04's claim, i.e. *evaluation metric
  fragility under model damage on the same measured construct* (B04: `acc_norm`
  decision-margin compression / near-tie density under damage rungs, currently
  `NARROWED_TO_OLMO_2_ONLY`). Requires the mechanism to be the same (margin/near-tie),
  not merely the theme ("evals are fragile").
* → **A02** iff it is inseparable from the depth-tax result and has no reach beyond it.
* → **new backlog proposal** iff it is a general, separable claim about
  benchmark selection for a *specific class* of manipulation, whose mechanism
  (retrieval saturation gating the expressible effect) is different from B04's
  (margin compression under damage).

---

## 2. JOB 2.1 — can `variable_tracking`'s retrieval-closed status be measured directly?

Current status: **inherited, not measured.** dvr §7.3 records VT recall as `n/a` with
**0 gold-locatable** items, because `_locate_needle_chunks` cannot localise a VT chain.
Two of the four primary cells are VT.

**Structural finding that makes a direct measurement plausible** (verified before
pre-registering, from source only): `eval_ruler_mem_space._make_vt` **returns the chain
sentences and the variable names** (`vars_all, chain, value`), but `_build_sample`
**discards them for VT**, returning `gold_needle=None`. The dvr locator therefore got
`None` and recorded 0 locatable — *the information exists in the generator and was
dropped at the interface*, it is not intrinsically unavailable. And the dvr analyzer
already regenerates every RULER sample **bit-identically** (sha256 pairing across all
7 arms, 0 failures), so the same regeneration can recover the chain.

**Pre-registered VT recall definition** (fixed now, before measuring):
a VT sample is a **HIT iff every one of its 5 chain sentences** (`VAR A = 12345`,
`VAR B = VAR A`, …) **lands in the top-12 pack**. This is the *same strict all-in-pack
rule* the dvr used for NIAH/BABILong, so the number is protocol-comparable. It is a
**multi-hop chain**, so all-in-pack is strict, and partial-chain recall is reported as
a secondary, more lenient number (fraction of chain sentences in pack).

**Falsification / decision rule:**

* If measured VT recall@12 is **≥ 95 %**, VT's retrieval-closed status is
  **established directly** and caveat 1 is discharged; the 4-cell primary statistic
  stands as written.
* If it is **materially below** the niah_mk1 level (< 95 %), VT is **not**
  retrieval-closed, and the **primary statistic must be restated as niah_mk1-only**
  (−68 / −57 pp at j=18) **with VT demoted to support**. Restating is an acceptable
  outcome and I will implement it rather than argue around it.
* If regeneration cannot be made bit-identical (sha mismatch vs recorded
  `input_ids_sha256`), the measurement is **refused** — no approximate recall number
  will be reported. Fail-closed.

## 3. JOB 2.2 — is there a harder retrieval-closed cell that de-saturates the shallow end?

A0–A3 sit at 95–100 % on all four cells, so "tax ≈ 0 at shallow j" is partly a
statement about a saturated benchmark. Requirement for a de-saturating cell: it must be
**harder** (base accuracy meaningfully below ceiling) **while staying retrieval-closed**
(recall@12 ≥ 95 %), otherwise it re-introduces exactly the confound the primary read-out
was designed to exclude.

Available knobs in `scripts/eval_ruler_mem_space.py` (read from source):
`niah_single_3` (essay haystack, **36-char UUID values** — harder value copying),
`niah_multivalue` (1 key, 4 values, retrieve all), `niah_multiquery` (4 keys all
queried), and lengths up to `64k` / `128k` / `256k`.

**Pre-registered gate before spending GPU**: a candidate cell is worth running only if
it is plausibly *both* harder and retrieval-closed. Longer length attacks retrieval
recall (dvr showed recall degrades with length: qa2 49.5 % → 22.9 % from 16k→32k), so
**length is the wrong knob** — it de-saturates by breaking retrieval, which is
disqualifying. **Task difficulty at fixed 16k/32k is the right knob.** I will run at
most a small cell set, on `.82`, and will report `RAN` vs `READ` separately. If nothing
cheap satisfies both conditions, I record what would be needed and spend no GPU.

## 4. Invariants binding every number below

1. `chat_template=False` asserted for every eval, using `is not False` (never
   `is not True`, which silently passes on `None`). RULER stores config **flat** with
   `chat_template` in the **sibling summary**; BABILong nests it under **`prompt`**.
2. `selector=iter_bm25` (one-shot `bm25` results are void).
3. Sharded eval: assert **every shard present AND exact expected item count**, 0 dups,
   0 NaN, before any merge.
4. **Per-cell only. No pooled BABILong/LongEval figure** (banned −17.89 / +2.00 pp).
5. Canonical scorers **imported, never reimplemented**.
6. `_eval_taskpool_2group.sh` is hardcoded to `run_babilong_mem_space.py` and has **zero
   RULER support** — the dvr/read-tax driver pattern is used instead.
7. Remote kills by PID with `setsid` + `kill -9`; never `pkill -f`; eval procs excluded.
8. **RAN vs READ separated in every document.**
9. Two-disk rule: nothing called missing until checked on **both** wzc1 and zwfy6.
