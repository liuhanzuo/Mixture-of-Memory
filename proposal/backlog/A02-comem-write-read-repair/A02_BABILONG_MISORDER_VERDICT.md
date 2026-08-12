# A02 — Is the BABILong misordering a finding? VERDICT + A02 closeout

**Executed**: 2026-08-12, 22:45 → 23:50 CST, `.82` only (8× H20, zwfy6).
**Pre-registration**: `A02_BABILONG_MISORDER_PREREG.md`, written **before** any statistic
below was computed (only JSON key structure, index joinability and already-published
numbers were inspected first).
**GPU spent**: **~1.5 GPU-h** of the ≤30 budget. Jobs 1 and 2.1 were **pure CPU**
re-analysis of per-item vectors already on disk; the only GPU was Job 2.2's 4-arm
`niah_single_3` eval (11 min wall on 8 cards). `.82` verified 8/8 at 0 MiB / 0 % before
launch and released to 0 MiB / 0 % after. **`LOCAL`/`.21`/`.73`/`.104` were never
contacted** — no ssh, no process, no file write.

---

## 0. RAN vs READ

**RAN (new GPU spent in this gate)** — one cell, four arms:

| arm | j | adapter | `niah_single_3` × {16k,32k} |
|---|---|---|---|
| A0 | 0 | none | **RAN** (new task, never evaluated) |
| A2 | 6 | `qcmem_distill_qwen_j6_r32_4k` | **RAN** |
| A3 | 9 | `qcmem_distill_qwen_j9_r32_4k` | **RAN** |
| A4 | 12 | flagship `..._j12_r32_4k` | **RAN** |

**RAN (new CPU, zero GPU)** — 4 analyses over artefacts already on disk:
per-cell McNemar/bootstrap/ladder statistics; the format diagnosis; the truncation
ablation; the **direct `variable_tracking` recall measurement**; the 0-GPU
retrieval-closure screen of 3 candidate tasks × 2 lengths.

**READ (reused, not re-run)** — every accuracy number in the read-tax table
(`a02_read_tax_per_item_vectors.json`, 10 cells × 7 arms × n=100), the dvr per-item
recall labels, and the dvr recall@12 percentages. **No arm was re-evaluated on any cell
that already existed.** GATE J3 re-derived all 70 published per-cell accuracies from the
per-item vectors and reproduced **70/70 exactly**.

### Gate status

```
J1 integrity      n=100/arm/cell, 0 NaN, 0 dup, binary-only          PASS
J2 exact join     read-tax idx == dvr idx, 100/100 on all 10 cells   PASS (no imputation)
J3 reproduce      all 70 published accuracies re-derived             PASS (70/70 exact)
J4 refuse-cond    conditioning REFUSED where gold_locatable==0       FIRED on both VT cells
GATE C/C2/D (Job 2.2)  shards, n==100, 0 dup/NaN, sha pairing,
                       chat_template=False, iter_bm25/topk12/chunk512 PASS (0 errors)
selftest          McNemar vs closed form, Spearman ρ=−1/p=2/120,
                  Holm, J3 negative test                             PASS
```

**`chat_template` was verified with `is not False`, never `is not True`.** Confirmed on
disk that RULER's `records.json` is **flat and carries no `chat_template`** (it lives in
the sibling summary, where it reads `False`), exactly the trap the previous agent hit.

---

## 1. JOB 1 — the answer: **the misordering is real as a RANKING FAILURE, but it is NOT
a significant inversion, and the pre-registered mechanism is REFUTED**

### 1.1 The count "4 of 6 cells" survives; the *inference* does not

Per-cell A4 (j=12, RULER 90.75) vs A5 (j=18, RULER 20.75). True effect **+70 pp**.

| cell | A4 | A5 | Δ pp | CI95 | b/c | McNemar exact p | inverted? | **sig. inverted?** |
|---|---|---|---|---|---|---|---|---|
| ruler niah_mk1 16k | 90.0 | 32.0 | **+58.00** | [+48, +68] | 58/0 | <1e-4 | no | no |
| ruler niah_mk1 32k | 96.0 | 42.0 | **+54.00** | [+44, +64] | 54/0 | <1e-4 | no | no |
| ruler var_track 16k | 88.0 | 4.0 | **+84.00** | [+77, +91] | 84/0 | <1e-4 | no | no |
| ruler var_track 32k | 89.0 | 5.0 | **+84.00** | [+76, +91] | 84/0 | <1e-4 | no | no |
| babilong qa1 16k | 17.0 | 19.0 | −2.00 | [−11, +8] | 11/13 | 0.8388 | **YES** | no |
| babilong qa1 32k | 12.0 | 12.0 | +0.00 | [−9, +9] | 11/11 | 1.0000 | **YES** (tie) | no |
| babilong qa2 16k | 8.0 | 9.0 | −1.00 | [−8, +6] | 6/7 | 1.0000 | **YES** | no |
| babilong qa2 32k | 1.0 | 7.0 | −6.00 | [−12, −1] | 1/7 | 0.0703 | **YES** | **CI yes, McNemar no** |
| babilong qa5 16k | 58.0 | 49.0 | +9.00 | [−1, +19] | 18/9 | 0.1221 | no | no |
| babilong qa5 32k | 58.0 | 48.0 | +10.00 | [+0, +20] | 18/8 | 0.0755 | no | no |

**Adjudication of the pre-registered claims (§1 of the PREREG):**

* **Claim S (significant inversion) — NOT ESTABLISHED. F1 essentially fires.**
  Only `qa2×32k` has a CI95 excluding 0 in the A5-favouring direction (−6.00,
  [−12,−1]), and its **exact McNemar p = 0.0703 does not reach 0.05** (the contrast
  rests on **8 discordant items: b=1, c=7**). Under Holm within the 6-cell family the
  adjusted p is **0.4219**. Two tests disagree on the one cell that could carry S, and
  the significant-looking one is the *less* conservative. **I will not call this a
  significant inversion.** The premise's phrase "would be wrong in sign" is correct
  about the **point estimates** and unsupported as an **inferential** claim.
* **Claim M (fails to recover an ordering RULER resolves decisively) — ESTABLISHED.**
  This is the defensible finding, and it is not weak. On the r=32 ladder
  (j = 0/6/9/12/18), exact-permutation Spearman:

  | cell | accuracies (A0,A2,A3,A4,A5) | ρ | exact p | recovers ordering? |
  |---|---|---|---|---|
  | ruler niah_mk1 16k | 100, 99, 99, 90, 32 | −0.975 | 0.0333 | **YES** |
  | ruler var_track 16k | 100, 99, 99, 88, 4 | −0.975 | 0.0333 | **YES** |
  | ruler niah_mk1 32k | 99, 99, 95, 96, 42 | −0.872 | 0.1000 | no |
  | ruler var_track 32k | 100, 100, 100, 89, 5 | −0.894 | 0.1000 | no |
  | babilong qa1 32k | 35, 21, 14, 12, 12 | −0.975 | 0.0333 | **YES** |
  | babilong qa1 16k | 33, 23, 21, 17, 19 | −0.900 | 0.0833 | no |
  | babilong qa2 16k | 17, 13, 8, 8, 9 | −0.667 | 0.2667 | no |
  | babilong qa5 32k | 61, 62, 57, 58, 48 | −0.800 | 0.1333 | no |
  | babilong qa2 32k | 11, 3, 2, 1, 7 | **−0.400** | 0.5167 | no |
  | babilong qa5 16k | 53, 62, 63, 58, 49 | **−0.300** | 0.6833 | no |

  On a manipulation whose true magnitude is **70–84 pp**, four BABILong cells put
  A5 **at or above** A4, and two cells (`qa2×32k` ρ=−0.40, `qa5×16k` ρ=−0.30) retain
  almost no rank information at all. **Note honestly: n=5 caps the attainable exact p at
  0.0167, so 4 of 10 cells miss significance including two RULER cells** — the ladder
  test is underpowered by construction and is reported as descriptive rank recovery, not
  as a powered test.
* **F4 control — BABILong is NOT globally orderless.** On the uncontested A0-vs-A5
  contrast it is significant on 3 of 6 cells (qa1×16k +14.00 [+2.98,+26], qa1×32k
  +23.00 [+12,+34], qa5×32k +13.00 [+3,+24]) and n.s. on the other three. So these cells
  *can* detect a 79-pp collapse relative to *base*; what they cannot do is order the
  **j=12 vs j=18** step. That is a narrower and more defensible statement than
  "BABILong is invalid".

### 1.2 The pre-registered mechanism (retrieval domination) is **REFUTED**

This is the sharpest reversal in this gate, and it goes **against** the premise.

Conditioning on the dvr retrieval labels (legitimate: the pack is arm-independent —
identical selector config and `input_ids_sha256` asserted equal across arms, so HIT/MISS
is a pre-treatment covariate of *(cell, sample)*):

| cell | HIT n | HIT Δ(A4−A5) | HIT inverted? | MISS n | MISS Δ | MISS inverted? |
|---|---|---|---|---|---|---|
| qa1 16k | 60 | **−6.67** | **YES** | 35 | +8.57 | no |
| qa1 32k | 57 | **−5.26** | **YES** | 43 | +6.98 | no |
| qa2 16k | 47 | **−2.13** | **YES** | 48 | +0.00 | tie |
| qa2 32k | 22 | **−13.64** | **YES** | 74 | −5.41 | YES |
| qa5 16k | 59 | +5.08 | no | 33 | +21.21 | no |
| qa5 32k | 55 | +9.09 | no | 40 | +10.00 | no |

**The inversion does not dissolve on retrieval-HIT items — it is LARGER there** (qa1×16k
−6.67 on HIT vs **+8.57** on MISS; qa2×32k −13.64 vs −5.41). The pre-registered
prediction was the exact opposite ("separation larger on HIT than on MISS"). So:

* **F2 does not fire in the direction it was written for.** The premise's fallback
  ("BABILong is merely underpowered on the wrong subset") is **not** what the data show:
  the misordering is concentrated on the subset where retrieval **succeeded**.
* **Retrieval domination is therefore NOT the mechanism.** Low recall@12 (22.9–63.2 %)
  is real and it *is* why these cells cannot support depth inference in general, but it
  does **not** explain the A4/A5 inversion.
* **Statistic 5 agrees**: across the 6 cells, Spearman(recall@12, signed A4−A5
  separation) = **+0.543, exact p = 0.2972** — the right sign but not significant, and
  with 6 cells it cannot be.
* **F3 (floor) fires instead, and it is confounded.** Spearman(A0 accuracy, signed
  separation) = **+0.943, exact p = 0.0167** — at the n=6 exact-permutation lower bound,
  i.e. the strongest available. But Spearman(recall, A0 accuracy) = **+0.714**, so
  recall and floor are **substantially collinear across these 6 cells**. Per F3 I
  therefore record: **the retrieval-vs-floor mechanism is NOT IDENTIFIED by 6 cells.**
  Floor has the better sign, the better p, and the better story — but I cannot separate
  the two regressors here and I will not pretend otherwise.

### 1.3 The actual mechanism found: **the metric scores OUTPUT FORMAT, not just reading**

Following the conditioning result, I inspected the generations. `compare_answers`
requires the target to be the **only** task label surviving `preprocess_output`, and
`preprocess_output` **truncates at the first period**. A base LM that answers with a
multiple-choice enumeration —

```
A4 raw output: 'Choices: A. In the kitchen B. In the room C. In the street'
  preprocess_output -> 'choices: a'      -> 0 labels survive -> scored WRONG
A5 raw output: 'Answer: It is in the kitchen.  Question: What is the name of...'
  preprocess_output -> 'answer: it is in the kitchen' -> scored RIGHT
```

— is auto-zeroed **whether or not it located the fact**. The two arms emit that format
at very different rates, and the asymmetry is overwhelming:

| cell | A4 list-format | A5 list-format | McNemar p | A4 trunc-kills | A5 trunc-kills | McNemar p |
|---|---|---|---|---|---|---|
| qa1 16k | **62 %** | 31 % | <1e-5 | 40 % | 12 % | <1e-5 |
| qa1 32k | **75 %** | 42 % | <1e-5 | 43 % | 16 % | <1e-5 |
| qa2 16k | **60 %** | 25 % | <1e-5 | 25 % | 9 % | 0.0004 |
| qa2 32k | **74 %** | 32 % | <1e-5 | 22 % | 5 % | 0.0002 |
| qa5 16k | 2 % | 6 % | 0.2891 | 4 % | 4 % | 1.0000 |
| qa5 32k | 2 % | 5 % | 0.2500 | 3 % | 4 % | 1.0000 |

"trunc-kills" = the target **is present in the raw output** but is removed by the
first-period truncation. A4 loses **22–43 %** of its items that way; A5 loses 5–16 %.

**The dissociation is perfect**: the 4 cells whose ordering inverts
(qa1×{16k,32k}, qa2×{16k,32k}) are **exactly** the 4 cells with high list-format rates;
the 2 cells that order correctly (qa5×{16k,32k}, list format 2 %) are exactly the low
ones. Contingency table `[[4,0],[0,2]]`, **Fisher exact p = 0.0667** — the minimum
attainable with 6 cells, so this is a **descriptive dissociation, not a powered test**.

**The one-operation ablation.** Removing *only* the first-period truncation, keeping the
uniqueness requirement (so multiple-choice lists still score 0 and chance inflation is
removed by construction), on **all 100 items per cell with no conditioning**:

| cell | canonical Δ | inverted? | **ablated Δ** | inverted? | ladder ρ canonical → ablated |
|---|---|---|---|---|---|
| qa1 16k | −2.00 | YES | **+5.00** | **no — REPAIRED** | −0.900 (p .083) → **−0.975 (p .033)** |
| qa1 32k | +0.00 | YES | **+5.00** | **no — REPAIRED** | −0.975 → **−1.000 (p .0167)** |
| qa2 16k | −1.00 | YES | −2.00 | YES | −0.667 → −0.600 |
| qa2 32k | −6.00 | YES | −1.00 | YES | −0.400 → −0.205 |
| qa5 16k | +9.00 | no | +10.00 | no | −0.300 → −0.300 |
| qa5 32k | +10.00 | no | +9.00 | no | −0.800 → −0.800 |

**Deleting one line of metric code repairs the inversion on 2 of 6 cells and turns
qa1×32k into a perfect ladder (ρ = −1.000).** It does **not** repair qa2, where the
floor (A4 at 1 %) leaves nothing to recover. So the honest decomposition is:
**format/truncation accounts for the qa1 inversions; the qa2 inversions are floor.**

I explicitly retract a stronger intermediate number: a first, *lenient*
format-insensitive read (`target_in_raw`, target substring anywhere) showed **4/6 sign
flips**, but it is inflated by chance for multiple-choice outputs (a list naming 3 of 6
rooms hits the target ~50 % of the time by luck) and A4 emits lists far more often. The
strict ablation gives **2/6**. **2/6 is the number that stands.**

### 1.4 So what is the finding, stated at the strength the evidence supports

> On a depth manipulation whose true effect on retrieval-closed RULER is **+70 to +84 pp**,
> six widely-used BABILong cells fail to recover the ordering, and on four of them the
> point estimate is **inverted**. The inversion is **not** significant (best exact
> McNemar p = 0.0703, Holm 0.4219), so the correct claim is **ranking failure, not
> demonstrated sign error**. Its mechanism is **not** the retrieval bottleneck one would
> expect — the inversion is *larger* on retrieval-HIT items. Two of the four inversions
> are repaired by deleting a single operation from the metric (`preprocess_output`'s
> first-period truncation), which auto-zeroes **22–43 %** of one arm's items in which the
> target is actually present; the other two sit on a 1 % floor.

That is a **metric-implementation** finding plus a **floor** finding, not a
"benchmark inverts conclusions" finding. It is real, it is mechanistically explained,
and it is smaller than the premise proposed.

### 1.5 Ownership — **new backlog proposal `B11`, not A02 and not B04**

Decided against the pre-registered criteria (§1.4 of the PREREG), not by provenance:

* **NOT B04.** B04's construct is *per-item `acc_norm` decision-margin compression under
  model damage* — a **likelihood-ranking** metric over multiple-choice options, no
  generation, no retrieval, no string matching, currently `NARROWED_TO_OLMO_2_ONLY`.
  This finding is about a **generative string-matching scorer's text preprocessing**
  interacting with output format. Same umbrella word ("eval fragility"), **different
  measured construct, different mechanism, different failure surface**. Filing it under
  B04 would let the OLMo-2-only margin claim borrow support from an unrelated
  mechanism — precisely the conflation B04's own narrowing was meant to prevent. Per
  `memory/direction-a-eval-fragility-established.md` this is **not** a cross-family
  extension of B04's claim.
* **NOT A02.** A02 produced it, but it is **separable**: it concerns any base LM scored
  by `babilong.metrics` on a generative task, with or without CoMem, with or without a
  depth knob. Keeping it inside a dead-thesis proposal buries a reusable methodological
  result. A02 keeps it as *provenance*, not as *owner*.
* **→ New backlog proposal.** Its claim: *generative long-context benchmark scorers can
  encode output-format conventions strongly enough to destroy the ranking of a large
  true effect, and the failure is localised to specific, auditable lines of the metric.*
  Kill gate must include (a) cross-family replication (a second model family, since a
  single family's format habits could explain everything), (b) a novelty check against
  the LM-eval-harness / answer-extraction-robustness literature, which is **large** —
  this may well be already known, and that must be checked before any GPU is spent.

**Created**: `proposal/backlog/B11-generative-scorer-format-fragility/` with
`PROPOSAL.md`, `STATUS.json`, `SOURCES.md`. It starts at
`status: BACKLOG_UNVERIFIED_NOVELTY_UNCHECKED` — **not** "established".

---

## 2. JOB 2 — the two caveats, closed

### 2.1 Caveat 1 (`variable_tracking` recall) — **DISCHARGED by direct measurement, no
restatement needed**

The dvr gate reported VT recall as `n/a` with 0 gold-locatable items, and the read-tax
verdict carried VT's retrieval-closed status as *inherited from an accuracy step*. The
premise offered "restate the primary statistic as niah_mk1-only" as an acceptable
answer. **It is not necessary, because VT recall is directly measurable** and I measured it.

**Why it was `n/a` — an interface bug, not an intrinsic limit.**
`eval_ruler_mem_space._make_vt()` **returns** the chain sentences and variable names, but
`_build_sample()` **discards them for VT**, returning `gold_needle=None` (its own
docstring: *"NIAH tasks only; None for variable_tracking"*). The dvr locator received
`None` and correctly reported 0 locatable. The information existed in the generator and
was dropped at an interface.

**Measurement** (`measure_a02_vt_recall.py`, **0 GPU**): regenerate each sample with the
eval's RNG sequence, reconstruct the 5 chain sentences, apply the **same strict
all-in-pack rule** dvr used for NIAH, with a fail-closed sha gate.

| cell | recall@12 (strict, all 5 chain sentences in pack) | Wilson CI95 | fully locatable | sha pairing |
|---|---|---|---|---|
| ruler var_track 16k | **100.0 %** | [95.72, 100.0] | 86/100 | **PASS** |
| ruler var_track 32k | **100.0 %** | [96.07, 100.0] | 94/100 | **PASS** |

**Every regenerated prompt's `input_ids_sha256` matched what all 7 arms recorded** — the
regeneration is bit-identical, so this measures the packs the eval actually used.

**Honest coverage caveat**: 14/100 (16k) and 6/100 (32k) samples had exactly **one**
chain sentence my locator could not place (whitespace normalisation across a decode
boundary), so they are excluded from the strict denominator. In **all 20** of those
samples, *every* chunk that *was* located is in the pack (`frac_gold_chunks_in_pack` =
1.0), so under either treatment recall is 100 % across all 200 samples. The number does
not depend on how the partials are handled.

**Consequence**: VT is retrieval-closed **directly measured**, not inherited. The
4-cell primary statistic (read tax = −0.50 / −1.50 / −9.00 / −79.00 pp) **stands as
written**. Read-tax verdict caveat 1 is discharged, and the dvr `n/a` should be
superseded rather than carried forward.

*(Two locator bugs of my own were found and fixed en route, both recorded because both
initially produced a confident-looking wrong answer: (i) a per-chunk substring test
silently lost every chain sentence straddling a 512-token boundary → 0/100 locatable;
(ii) `prompt.rfind("value ")` landed in `VT_ANSWER_PREFIX` and returned the value
**with a trailing comma**, making exactly one sentence per sample unfindable while
`frac_in_pack` read a reassuring 1.0. Both now fail closed.)*

### 2.2 Caveat 2 (ceiling at shallow j) — **a de-saturating cell exists, was run, and it
CHANGES the shallow-end conclusion at 16k**

Screen first (`probe_a02_desaturation_candidates.py`, **0 GPU**, criterion recall@12 ≥ 95 %):

| candidate | 16k recall@12 | 32k recall@12 | verdict |
|---|---|---|---|
| `niah_single_3` (36-char UUID values) | 97.5 % | 95.0 % | **PASS**, and single-needle ⇒ screen is **exact** |
| `niah_multiquery` (4 keys queried) | 100 % | 95.0 % | passes, but **untrustworthy** |
| `niah_multivalue` (1 key, 4 values) | 100 % | **92.3 %** | **FAILS** |

`_make_niah` returns **only the first** queried needle as gold, so recall for the
multi-needle tasks is an **upper bound** — I rejected them rather than quote a number my
locator cannot verify. **Length was rejected as the difficulty knob a priori** (dvr
measured recall degrading with length, 49.5 → 22.9 %), because de-saturating by breaking
retrieval re-introduces the confound the primary read-out exists to exclude.

**Result** (`niah_single_3` × {16k, 32k}, n=100/cell, 8 shards, all gates 0 errors,
0 sha failures, ~1.5 GPU-h):

| arm | j | 16k | tax vs A0 @16k | 32k | tax vs A0 @32k |
|---|---|---|---|---|---|
| A0 | 0 | 97.0 [91.6, 99.0] | anchor | 96.0 [90.2, 98.4] | anchor |
| A2 | 6 | 89.0 [81.4, 93.8] | **−8.00 [−14, −3] SIG** | 95.0 | −1.00 [−3, +0] ns |
| A3 | 9 | 89.0 [81.4, 93.8] | **−8.00 [−14, −3] SIG** | 96.0 | +0.00 [+0, +0] ns |
| A4 | 12 | 90.0 [82.6, 94.5] | **−7.00 [−13, −2] SIG** | 95.0 | −1.00 [−3, +0] ns |

**This partially falsifies the read-tax verdict's "the knob is nearly free to j≈9".**
On the harder cell at 16k, j=6 already costs **−8.00 pp (significant)**, against
**−0.50 pp** on the saturated primary cell. The shallow end was **not** free; it was
**unmeasurable**, exactly as caveat 2 warned.

Two things must be said honestly against over-reading this:

1. **The cell did not fully de-saturate** (A0 = 97.0 / 96.0, still high). So this is
   "a harder cell that resolves a difference the easier cell could not", not "a cell
   with headroom".
2. **At 16k the depth *ordering* also flattens**: A2 89.0, A3 89.0, A4 90.0 — the tax
   appears **immediately and then stops growing** to j=12, unlike the primary cell's
   clean −0.5 → −1.5 → −9.0 progression. And at 32k nothing is significant at all. So
   `niah_single_3` **adds a shallow-end tax while weakening the monotone-cliff picture**.
   The cliff at j=18 is untouched (A5 was not run here; it is at 4–42 % everywhere).

**Net**: the read-tax verdict's headline shape (free → −9 → −79) is a **`niah_multikey_1`
+ `variable_tracking` statement**, not a depth law. The claim "tax ≈ 0 for j ≤ 9" must be
**narrowed** to those two tasks and **cannot** be generalised — one harder retrieval-closed
task at one length already contradicts it.

---

## 3. JOB 3 — A02's disposition: **`backlog`, not `active`, not archived**

The previous agent's closing line was: *"Nothing here is a win. A0 — no adapter at all —
is the best arm. The result prices a knob; it does not find a thesis. A02's storage form
stays dead."* **I take it literally and it survives this gate intact** — and this gate
made it slightly worse, since the shallow end is now known to carry a real tax at 16k.

**Is a priced knob + a benchmark finding a paper?** No — and the honest reason is that
after this gate they are not even two facts of equal quality:

* **The priced knob is CoMem-internal.** It quantifies the cost of *our own* dead
  mechanism. There is no baseline anyone else would want to beat, and A0 (do nothing)
  wins. Nobody needs the price of a knob on a discarded design.
* **The benchmark finding is real but reduced and, critically, is now owned elsewhere**
  (B11) with novelty **unchecked** against a large answer-extraction literature. It also
  no longer says what made it exciting: the inversion is not significant, and its
  mechanism is a metric-preprocessing line plus a floor, not a benchmark-validity
  catastrophe.
* Binding them into one paper would produce exactly the "two orphan facts" the premise
  warned about — with the added defect that the interesting one is not A02's to keep.

**Not archived either.** Archiving asserts the direction is dead *and* its evidence is
spent. A02's evidence is actively load-bearing: the read-tax ladder, the capacity-matched
pair, the depth-vs-retrieval decomposition, the per-item vectors, and now the **directly
measured VT recall** and the **de-saturation cell** are the reference artefacts for any
future depth/read claim, and B11 depends on A02's generations as its source data. An
`archive/` move with a `POSTMORTEM.md` would signal "do not reuse".

**Decision: A02 sits in `proposal/backlog/`, status
`CLOSED_NO_THESIS_DIAGNOSTIC_ASSETS_RETAINED`.** No further A02 GPU. Resurrection
requires a *new mechanism*, not another read-out of the same ladder.

> **Provenance correction — do not let this be misread later.** I first wrote that this
> gate *performed* the `active/ → backlog/` move. It did not. **Commit `135707b` (not
> mine, landed after the read-tax commit `3c9f8f9`) had already moved A02 to
> `proposal/backlog/` in git.** The working tree still showed it under `active/`, so my
> `git mv` only re-applied a move git had already recorded — which is why this commit
> contains no deletions under `active/`. Verified nothing was lost: all **50**
> HEAD-tracked A02 files are present in the worktree (plus this gate's 18 new files) and
> `evidence/read_tax_ruler/` is intact. **This gate's contribution is the status change
> and the closeout argument, not the directory move.**

**What A02 leaves behind, itemised** (its actual value):

1. A 5-point depth-tax curve on retrieval-closed RULER with retrieval byte-identical
   (free → −9 → −79 pp), **now correctly scoped to two tasks**.
2. Depth ≠ capacity, at exactly 72,744,960 params (−8.75 pp) — independently re-verified.
3. The j=0 control is a **null adapter** (0/400 correctness flips), so "optimal j=0
   adapter ≡ base model" is measured, not argued.
4. `variable_tracking` recall@12 = **100 %**, directly measured, superseding dvr's `n/a`
   — plus the reusable method for measuring it.
5. A harder retrieval-closed cell (`niah_single_3`) and the 0-GPU screen that found it.
6. The generations and per-item vectors that B11 is built on.

**Retractions/narrowings this gate forces on prior A02 documents** (all recorded in
`STATUS.json`, none hidden):

* Read-tax verdict §4's "**misorders** … would have been **wrong in sign**" →
  **narrow to "fails to recover the ordering; point estimates invert on 4/6 cells;
  the inversion is not statistically significant"**.
* Read-tax verdict caveat 1 (VT recall unmeasured) → **discharged**, recall = 100 %.
* Read-tax verdict §5 "no claim that j≤9 is free in general" → **strengthened from a
  caveat to a measured limitation**: on `niah_single_3`×16k, j=6 costs −8.00 pp (SIG).
* The premise's proposed mechanism (retrieval domination) → **refuted for the
  inversion**; retrieval domination remains the correct reason these cells cannot
  support depth inference *in general*.

---

## 4. Provenance

| artefact | location |
|---|---|
| pre-registration | `A02_BABILONG_MISORDER_PREREG.md` (this dir) |
| Job 1 statistics | `evidence/babilong_misorder/a02_babilong_misorder.json` md5 `403f5dc7439293436ad48ee165cf68da` |
| format diagnosis | `evidence/babilong_misorder/a02_babilong_format_diagnosis.json` md5 `615eaab67751dc138bd919596b1fd29c` |
| format per-item stats | `evidence/babilong_misorder/a02_format_mechanism.json` md5 `d6d1fa554f464cbad99ab2bcf06ae66b` |
| truncation ablation | `evidence/babilong_misorder/a02_truncation_ablation.json` md5 `db850dd98331ad9c8b1eaf47f7061af0` |
| **VT recall (direct)** | `evidence/babilong_misorder/a02_vt_recall_direct.json` md5 `48677352b54f6f9bc599f1684ff23484` |
| de-saturation screen | `evidence/babilong_misorder/a02_desaturation_screen.json` md5 `d3cc5ab7f2343e00e1d74098a67d0616` |
| de-saturation result | `evidence/babilong_misorder/a02_desaturation_result.json` md5 `874fdc4dc033583a5c9590989c5364d8` |
| analyzers (5) | `code/analyze_a02_babilong_misorder.py`, `diagnose_a02_babilong_format.py`, `analyze_a02_format_mechanism.py`, `analyze_a02_truncation_ablation.py`, `measure_a02_vt_recall.py`, `probe_a02_desaturation_candidates.py`, `analyze_a02_desaturation.py` |
| de-sat eval driver | `code/run_a02_desaturation_eval.sh` |
| new RULER cells | `ruler_results/a02_desat_ruler_{A0_j0,A2_j6,A3_j9,A4_j12}/` (**zwfy6**) |
| eval logs | `logs/a02_desat_eval_progress.log`, `logs/a02_vt_recall.out`, `logs/a02_desat_screen.out` (**zwfy6**) |

**All 7 evidence JSONs are md5-identical on wzc1 and zwfy6** (verified after `scp -O`).

**Node discipline**: `.82` only. `LOCAL`/`.21` (SparseForge #246, 4571/7500 + 4319/7500),
`.104` (paperC Qwen3 heal) and `.73` (another agent) were **never contacted**. `.82`
verified idle before launch, released to 0 MiB / 0 % after. No job needed killing.
