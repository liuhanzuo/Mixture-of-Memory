# A04 — PLATEAU(T) repair, G0 anchor pinning, and Δ margin-sensitivity

**Date**: 2026-08-10. **GPU spent: ZERO** (three CPU scripts; remote access limited to
`ls` / `md5sum` / `sha256sum` on `.82`, read-only).

This document closes three open items that were blocking A04's gate, and reports one
finding that **narrows** a claim the Pilot Zero verdict makes. Every number below was read
from a file in this pass. Where prose and JSON disagreed, the JSON won and the prose was
corrected.

| item | status before | status now |
|---|---|---|
| `PLATEAU(T)` dimensionally incoherent — **blocked running the gate at all** | both implemented readings broken on a real grid | **REPAIRED** (rule R3, `evidence/a04_plateau_rule_repair.json`) |
| G0 anchor pinning by SHA256 (`A04_MARGIN_GUARD_PREREG.md` §8 item 8) | md5 only | **CLOSED** (`evidence/a04_g0_anchor_sha256_pinning.json`) |
| Δ = 0.10·residual looks arbitrary (live reviewer objection) | unanswered | **ANSWERED, with a caveat** (`evidence/a04_margin_sensitivity_sweep.json`) |

---

## 1. `PLATEAU(T)` — the defect, characterised precisely

`A04_GATE_DESIGN.md:69-71` defines the rule as: accept iff the relative in-domain
validation PPL improvement over the preceding grid interval is `< T`, with
**`T = 2.0 % per 5 000 steps`**. `code/pilot_zero_rule_disagreement.py:168-202` evaluates
**two** readings of that sentence. Both are broken on A04's own frozen grid.

### 1.1 Reading R1 — "unscaled": compare `rel` over `d` steps against `2.0`

The threshold's stated units are `%/5k`; the measured quantity is an improvement over `d`
steps. Since relative improvement grows with the interval, R1's real stringency is set by
the checkpoint spacing. Expressed back in `%/5k` units, R1's **effective** threshold on
A04's frozen grid `{2500, 5000, 10000, 20000, 40000, 80000}` (`A04_GATE_DESIGN.md:155`) is:

| checkpoint | interval `d` | R1's effective threshold (`%/5k`) | vs the stated `T = 2.0` |
|---|---:|---:|---:|
| 2 500 | 2 500 | 3.9600 | **1.980× too lenient** |
| 5 000 | 2 500 | 3.9600 | 1.980× too lenient |
| 10 000 | 5 000 | 2.0000 | 1.000× (the only correct point) |
| 20 000 | 10 000 | 1.0051 | 0.503× (too strict) |
| 40 000 | 20 000 | 0.5038 | 0.252× (too strict) |
| 80 000 | 40 000 | 0.2522 | **0.126× — 7.93× too strict** |

**15.70× stringency spread across the six pre-frozen checkpoints.** The rule is only
correct at exactly one of them. This is not a rounding issue: the same underlying training
trajectory would be declared plateaued at step 5 000 and still-improving at step 80 000.

R1 is also **grid-dependent on real data**. Holding the terminal checkpoint at step
200 000 and varying only *which* earlier checkpoint counts as "preceding":

| preceding | `d` | `rel` (%) | R1 accepts? | `rate_5k` (%/5k) | R3 accepts? |
|---|---:|---:|:--|---:|:--|
| 50 000 | 150 000 | 12.5305 | **False** | 0.44527 | True |
| 100 000 | 100 000 | 4.6386 | **False** | 0.23720 | True |
| 147 000 | 53 000 | 1.3875 | **True** | 0.13173 | True |

`rel` spans **9.03×** (1.3875 → 12.5305) and **R1's verdict is not unanimous** — the same
checkpoint is plateaued or not depending on grid bookkeeping. `rate_5k` spans 3.38× and
**R3's verdict is unanimous**.

### 1.2 Reading R2 — "linear-scaled": compare `rel` against `2.0 · (d/5000)`

R2 fixes the units and gets the algebra wrong: relative improvement **compounds**, it does
not add. A trajectory decaying at *exactly* the threshold rate `2.0 %/5k` produces
`rel = 100·(1 − 0.98^(d/5000))`, which is **below** `2.0·(d/5000)` for every `d > 5000`.
So R2 **accepts a run that is still improving at precisely the rate the pre-registration
calls "not plateaued"**, and does so more readily the wider the interval:

| `d` | `T_linear` (%) | true compounded (%) | R2 over-allowance (pp) |
|---|---:|---:|---:|
| 5 000 | 2.0000 | 2.0000 | 0.0000 (exact) |
| 20 000 | 8.0000 | 7.7632 | +0.2368 |
| 53 000 | 21.2000 | 19.2772 | **+1.9228** |
| 80 000 | 32.0000 | 27.6202 | **+4.3798** |
| 250 000 | 100.0000 | 63.5830 | +36.4170 |

At the boundary rate, R2 accepts at `d ∈ {5000, 10000, 20000, 40000, 53000, 80000}` and
rejects only at `d = 2500`. Two further failures, both measured:

* **R2 becomes vacuous** at `d ≥ 250 000` steps: its threshold reaches 100 %, which `rel`
  can never attain, so the rule **cannot reject** — unfalsifiable, not merely lenient.
* **R2 is not composition-consistent.** Over 200 000 random `(d1, d2)` checkpoint pairs
  drawn from the frozen grid's own spacings, R2 **accepts a merged interval while
  accepting neither half** in **174** cases (seed 0; **161** at seed 1). The verdict
  depends on how the operator happened to subdivide the grid. R3 has **0** violations at
  both seeds, in both directions.

This is why R1-vs-R2 is not a matter of taste: R1 is grid-dependent in stringency, R2 is
grid-dependent in *logic*.

### 1.3 The repair — rule R3

> **`PLATEAU(T)` accepts at checkpoint `c` iff**
> ```
> rate_5k(c) = 100 · ( 1 − (ppl_c / ppl_prev) ** (5000 / d) )   <   T,     T = 2.0
> ```
> where `d` = steps since the preceding grid checkpoint. `T` is **unchanged**.

**Why this and not something else — on principle, not by fit:**

1. **It is the unique reading under which the pre-registered sentence names the same
   quantity at every spacing.** `rate_5k` is invariant to how an interval is subdivided,
   so a threshold stated in `%/5k` becomes grid-free. That is what "per 5 000 steps"
   means.
2. **At `d = 5000` it is bit-identical to the pre-registered arithmetic** — asserted
   `|R1 − R3| < 1e-12` on four trajectories, including the real
   `17.6194 → 16.1613` pair. So R3 does not retune `T`; it stops mis-applying it off the
   5k grid. **No new numeric constant is introduced anywhere.**
3. **Composition-consistent in both directions** (0/200 000 violations, two seeds), which
   is exactly the property an irregular grid requires and R2 lacks.
4. **Never vacuous**: `rate_5k` has the same range at every `d`, so no spacing makes the
   rule unfalsifiable.

**Pre-registration status, stated plainly.** `T = 2.0 %/5k` is pre-registered (git
`d1ba737`, 2026-08-09) and is **not changed**. The **choice of R3 over R1/R2 is
POST-HOC** — decided 2026-08-10, after the pilot's PPL trajectory was on screen. That is
disclosed here, not buried.

**Why the post-hoc choice is nonetheless defensible.** R3 is **strictly stricter than R2 at
every `d > 5000`** (the over-allowance column above is the proof), so the repair makes
`PLATEAU` *harder* to accept — which makes the `PLATEAU`-vs-`NI` disagreement A04 needs
*harder to find*. **The repair costs A04 rather than paying it.** A units fix that
happened to help the hypothesis would deserve suspicion; this one does not, and the
justification above appeals only to invariance and composition properties that hold for
**any** `T`, not to which arm it accepts.

### 1.4 What the repair costs Pilot Zero — reported because it is unfavourable

| reading | first checkpoint that accepts on the real trajectory |
|---|---|
| R1 (unscaled, the pilot's pre-registered reading) | **200 000** |
| R2 (linear-scaled) | 100 000 |
| **R3 (repaired)** | **100 000** |

Under R3, `PLATEAU` first accepts at **step 100 000** (`rate_5k = 0.86012 %/5k`), not step
200 000. **Pilot Zero scored capability axes only at step 200 000.** So:

* The repaired rule's own *earliest* accept checkpoint had **no capability measurement** —
  the `PLATEAU`-vs-`NI` cell at step 100 000 was **UNMEASURED, not resolved**. There is
  **4.6386 %** further relative PPL improvement between step 100 000 and step 200 000, so
  this is not a negligible relocation.
* Step 200 000 **still accepts** under R3 (`rate_5k = 0.13173 %/5k` ≪ 2.0), so the pilot's
  single measured cell survives the repair and the disagreement it found is not withdrawn.
* But **any claim about *where* the earliest disagreement lies now requires step-100 000
  capability scoring**, which is GPU work and was **not done here**.

> **★ CLOSED 2026-08-12 — see `A04_STEP100K_PLATEAU_VS_NI_VERDICT.md`.**
> Step 100 000 was scored on all four axes (`.73`, 8×H20, **1.17 GPU-h measured**), same
> harness and same protocol as the step-200000 cells (`--add_bos 0`, base LM, no chat
> template; archived cells reproduce to **0.000e+00 pp**). Result: **PLATEAU(R3) accepts and
> NI rejects on 3/3 decision axes by 7.34–9.54× Δ, unanimously across all five null
> conventions.** The rules **DISAGREE** at R3's own earliest accept point, so the
> earliest-disagreement claim **moves from step 200 000 to step 100 000**.
> Steps 50 000 and 150 000 were also scored, but PLATEAU is **UNDEFINED** at both
> (150 000 has no in-domain PPL on disk; 50 000 is the trajectory's first point), so the
> intended bracket around the accept boundary is only half-built.
> **This does not rescue A04**: `keep7+fresh2` is confirmed a CONSTANT-REJECT rung across
> all four measured checkpoints (16 cells, **zero** NI accepts), K1 stays INDETERMINATE, K2
> is untouched, and RATIO(ρ=0.85) still *agrees* with NI everywhere.

The gate is now runnable on an irregular grid. That was the blocker.

---

## 2. G0 — anchor pinning by SHA256, and a discovery about the recorded md5

`A04_MARGIN_GUARD_PREREG.md:344` requires each intact anchor be named **by path + SHA256
before any gate GPU**; §8 item 8 admitted only md5 was recorded. Now closed.

### 2.1 The md5 in the prereg pinned a *different byte stream* than the analysis read

The prereg records md5 `d1a7b1cefc0031afa84e7b9334a08bc5` for
`zwfy6:olmo2_mmlu_content_results/A03_1B_base/per_example_mmlu.jsonl`. But Pilot Zero and
the guard classifier both read the **eight per-shard files**, not the merged file. A naive
shard-order concatenation of those eight gives md5 `759e8d639d388e652323d29f15cff197` —
**not** the recorded value.

Resolved: the canonical merged file is the shards merged **and re-sorted by `item_id`**.
Reconstructing that byte stream reproduces the recorded md5 **exactly**, and sha256
`8cbbb008f2f606b039d61f6fedfb54ae4b44aedf7430f9322fab3150027e2378`, which equals the
`sha256sum` computed remotely on `.82` against the canonical zwfy6 merged file. So the
recorded md5 does pin what it claims to pin — but that required demonstrating, and it was
not documented before. Same reconstruction verified for all four axes.

### 2.2 Pinned anchors (full shard-level hashes in the evidence JSON)

| axis | `n` | canonical merged path (zwfy6) | SHA256 |
|---|---:|---|---|
| MMLU-content | 14 042 | `olmo2_mmlu_content_results/A03_1B_base/per_example_mmlu.jsonl` | `8cbbb008f2f606b0…027e2378` |
| TriviaQA EM | 17 944 | `olmo2_closedbook_results/A03_1B_base/per_example_triviaqa.jsonl` | `c30b9072e139b682…1cbf9d00` |
| PopQA EM | 14 267 | `olmo2_closedbook_results/A03_1B_base/per_example_popqa.jsonl` | `500e45070bd9dea2…e3f736a5` |
| NQ-open EM (demoted) | 3 610 | `olmo2_closedbook_results/A03_1B_base_nq/per_example_nq_open.jsonl` | `58ced6e6bdd34d3d…1a104c6d` |

All **24 anchor shard files** additionally hashed; local staging copies in `/tmp/a04src`
are **byte-identical** to the canonical zwfy6 originals on all 24. The D5 second
measurement (`wzc1:olmo2_mmlu_content_results/a01_1B_intact_base_full/`) is pinned too —
sha256 `8797e77a3141…f200029` — and remains **excluded from Δ** per G0.

### 2.3 The "D5 48-flip drift" — does it reproduce? **YES, exactly.**

| quantity | prereg (`:210-212`) | recomputed here |
|---|---:|---:|
| `content_norm`, `A03_1B_base` (zwfy6) | 0.3868394816 | **0.386839481555334** |
| `content_norm`, `a01_1B_intact_base_full` (wzc1) | 0.3869819114 | **0.38698191140863125** |
| item flips | 48 / 14 042 | **48 / 14 042** |
| residual drift | +0.014243 pp | **+0.014242985329726565 pp** |
| Δ drift | +0.001424 pp | **+0.0014242985329726565 pp** |

**Reproduces to full printed precision. No correction needed.**

**New, and it sharpens the prereg's own reading.** The prereg reported flips on
`content_norm` only. The same two dumps differ on the other two interfaces too, by
*different* amounts:

| interface | acc (A03, zwfy6) | acc (a01, wzc1) | item flips |
|---|---:|---:|---:|
| MMLU-**letter** | 0.3807149978635522 | 0.38164079190998434 | **191** |
| `content_raw` | 0.35992023928215355 | 0.3594217347956132 | **33** |
| `content_norm` | 0.386839481555334 | 0.38698191140863125 | **48** |

Also: **80** items change *predicted option* under `content_norm` while only 48 change
correctness — 32 prediction changes are wrong-to-wrong. Three interfaces disagreeing by
33 / 48 / 191 items rules out "one bad shard" or "one truncated file"; it is a scoring-path
difference that touches all three interfaces with different sensitivity, consistent with
the prereg's "harness commit `7ac9653` falls strictly between the two evals".

**Still UNVERIFIED, unchanged:** causality. No same-code control was run (that needs GPU).
The prereg's refusal to call this a "noise floor" **stands and is reinforced** — the repo's
standing rule is that same-arch/same-harness re-runs are byte-identical, so a nonzero flip
count requires a named cause and "jitter" is not available as an explanation.

---

## 3. Margin sensitivity — is the disagreement an artefact of `Δ = 0.10 · residual`?

The reviewer objection: 0.10 is unmotivated. Recomputed the K1-shaped comparison over Δ
fractions **0.10 → 0.66** (the assigned range, 29 points at 0.02) and, because two
crossings sit above 0.66, **extended to 1.00** (46 points) so crossings could be reported
at all. Everything at or below 0.66 is unchanged by the extension; points above are flagged
`in_assigned_range: false`.

**`Δ` is NOT re-registered.** Guard G2 prohibits substituting it. The pre-registered
fraction stays 0.10. This is a sensitivity report *on* that number. Verified: the sweep's
`0.10` column reproduces the pilot's cells **bit-for-bit** (all four
`diff_lower95_one_sided_pp` identical to 10 decimal places; all four nulls identical).

Arm = `keep7f2_step200000`, the **only** cell where `PLATEAU` is defined and accepts —
`olmo2_ppl_results/` carries in-domain PPL only for the keep7+fresh2 trajectory, so no
other cell can form a `PLATEAU`-vs-`NI` disagreement at all.

### 3.1 The answer, split into two questions that have different answers

**Q1 — does a disagreement of the required shape still EXIST (≥1 decision axis rejects)?**
This is the question that bears on K1, because **K1 fires only if the rules agree**.

| range | split | first | last | credit | wrong |
|---|:--|:--|:--|:--|:--|
| assigned 0.10 → 0.66 | **True** | **True** | **True** | **True** | **True** |
| extended 0.10 → 1.00 | False | False | False | False | False |

**Across the entire assigned 0.10 → 0.66 range, under all five null conventions, at least
one decision axis rejects. The conclusion is invariant over the assigned range.** That is a
real robustness result: the K1-shaped disagreement is not an artefact of the 0.10 margin.

**Q2 — do ALL THREE decision axes still reject?** (the stronger 3/3 phrasing the pilot
verdict uses):

| range | split | first | last | credit | wrong |
|---|:--|:--|:--|:--|:--|
| assigned 0.10 → 0.66 | **True** | False | False | True (2/2 surviving) | **False** |
| extended 0.10 → 1.00 | False | False | False | False | False |

**Q2 is NOT invariant.** First grid fraction where any axis flips to accept:

| convention | first accept |
|---|---:|
| `wrong` | **0.36** |
| `first` | 0.66 |
| `last` | 0.66 |
| **`split`** ★ pre-registered | **0.68** |
| `credit` | 0.80 |

### 3.2 Exact crossing fractions (one-sided lower 95 % bound, `split`)

| axis | residual(intact) | lower-95 % of diff | crossing `f*` | in assigned range? |
|---|---:|---:|---:|:--|
| MMLU-content | 10.2389 pp | −6.8010 pp | **0.6642** | no, by **0.0042** |
| TriviaQA EM | 40.4313 pp | −31.6986 pp | 0.7840 | no |
| PopQA EM | 13.2053 pp | −12.0207 pp | 0.9103 | no |

Under `credit`, MMLU-content has **no** crossing at any `f > 0`: `residual(intact)` is
**−6.6871 pp**, so `Δ < 0`, `NI` degenerates to strict superiority, and *widening* the
fraction makes it **harder**, not easier. Guard D1/G1 marks that cell **NOT_CERTIFIABLE**
and `NI` is not run for it — the sweep records it and reports no accept/reject, exactly as
G2 requires.

### 3.3 A near-miss worth recording, and a pre-run estimate of mine that was wrong

Before running the bootstrap I estimated MMLU-content's crossing at **0.610** from the
**point estimate** (6.2455 / 10.2389). The rule uses the **one-sided lower 95 % bound**,
giving **0.6642** — i.e. **0.0042 above the assigned range's upper endpoint**. Point-estimate
crossings are systematically smaller than bound-based ones (TriviaQA 0.7693 vs 0.7840;
PopQA 0.8753 vs 0.9103; MMLU 0.6100 vs 0.6642), because the CI bound is the more demanding
quantity.

**Consequence, stated because it matters:** a sweep truncated at exactly 0.66 would have
reported "invariant" for the pre-registered `split` convention while sitting **0.4 points
of fraction** from a flip. The extension to 1.00 exists so that this near-miss is visible
rather than hidden by the range endpoint. Both the point-estimate and bound-based crossings
are in the evidence JSON.

### 3.4 What may and may not be said

* ✅ **MAY say**: "over the assigned 0.10 → 0.66 range, under all five null conventions, a
  disagreement of the K1-required shape persists; K1 does not become fireable by widening
  the margin sixfold."
* ✅ **MAY say**: "under the pre-registered `split` convention the earliest axis-level flip
  is at Δ-fraction 0.6642 (MMLU-content), i.e. a margin **6.6× the pre-registered one**."
* ❌ **MUST NOT say**: "all three axes reject for any plausible margin." Under `wrong` that
  fails at **0.36**, and under `first`/`last` at **0.66** — inside the assigned range.
* ❌ **MUST NOT say** the range was chosen to be safe. It was assigned before the script
  existed, and it came within 0.0042 of concealing the `split` flip.
* ⚠️ **Unchanged by all of this**: none of it touches **K1's ≥24-cell denominator**
  (3 cells exist, retraction `b93247f` stands: **K1 = INDETERMINATE**) or **K2**, which
  still needs seeds. The margin objection is answered; the gate is still not run.

---

## 4. Reproduce

```bash
# 1. PLATEAU repair (seconds)
python proposal/active/A04-recovery-certification/code/a04_plateau_rule_repair.py \
  --ppl_json  proposal/active/A04-recovery-certification/evidence/a04_1b_keep7f2_ppl_trajectory.json \
  --out_json  proposal/active/A04-recovery-certification/evidence/a04_plateau_rule_repair.json

# 2. margin sweep (~10 min; needs the per-example dumps staged read-only)
python proposal/active/A04-recovery-certification/code/a04_margin_sensitivity_sweep.py \
  --raw_root  <dir with olmo2_{mmlu_content,closedbook}_results/> \
  --out_json  proposal/active/A04-recovery-certification/evidence/a04_margin_sensitivity_sweep.json
```

Anchor hashes: `evidence/a04_g0_anchor_sha256_pinning.json`. Scorers/nulls are **imported**
from `A03/code/analyze_1b_knowledge_floor.py`, never reimplemented; 8/8 shard completeness,
exact item counts, duplicate-`item_id`, `nan`-row, and cross-arm `item_id` alignment are all
hard-asserted.

## 5. What this document does NOT establish

1. ~~**Capability measurement at step 100 000**, the repaired rule's own first-accept
   checkpoint. GPU required. The `PLATEAU`-vs-`NI` cell there is unmeasured.~~
   **CLOSED 2026-08-12** (`A04_STEP100K_PLATEAU_VS_NI_VERDICT.md`): measured, the rules
   DISAGREE (3/3 axes), earliest disagreement moves to step 100 000. Still open in its
   place: **no in-domain PPL exists at step 150 000**, so PLATEAU has no verdict there and
   the bracket around the accept boundary is half-built.
2. **Causality of the 48-item D5 drift.** Needs a same-code control (GPU). No noise floor
   is claimed.
3. **K1's ≥24-cell clause.** Still INDETERMINATE (`b93247f`). Nothing here changes it.
4. **K2.** Still open; the A03 data-order seeds (43/44) are the intended input.
5. **Whether R3 is the *only* defensible repair.** It is the only one that is
   grid-invariant, composition-consistent, exactly equal to the pre-registration at
   `d = 5000`, and never vacuous — but no exhaustive search over rule families was done.
