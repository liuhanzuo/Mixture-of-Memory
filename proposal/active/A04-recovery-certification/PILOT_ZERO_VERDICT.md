# A04 Pilot Zero — VERDICT

**Date:** 2026-08-09. **GPU spent: ZERO.** Pure CPU re-analysis of per-example shards already on
disk (zwfy6), plus read-only `ssh`/`scp -O` of small JSON files.

**Raw numbers:** `evidence/pilot_zero_rule_disagreement.json` (+ `.csv`).
**Runnable script:** `code/pilot_zero_rule_disagreement.py`.
**Reproduce:**
```
python code/pilot_zero_rule_disagreement.py \
  --raw_root <dir with olmo2_{mmlu_content,closedbook}_results/> \
  --ppl_json <[[step,ppl],...] recomputed from olmo2_ppl_results/*/shard*of8.json> \
  --ladder_json ../A01-null-calibration-methodology/evidence/gate3_content_null_conventions.json \
  --out_json evidence/pilot_zero_rule_disagreement.json \
  --out_csv  evidence/pilot_zero_rule_disagreement.csv
```

---

## 1. K1 VERDICT (lead) — **K1 DOES NOT FIRE.**

**A disagreement of the required shape EXISTS.** At the checkpoint where `PLATEAU(T=2.0%/5k)`
accepts (step 200,000), `NI(Δ=0.10·residual(intact,x))` **rejects on 3 of 3 decision axes**, each
by 5–30× the margin.

| axis | frac of intact residual recovered | diff vs intact | one-sided lower 95% bound | −Δ | NI accepts? |
|---|---:|---:|---:|---:|:--:|
| **TriviaQA EM** (primary) | 23.07 % | −31.102 pp | **−31.699 pp** | −4.043 pp | **NO** (7.8× margin) |
| PopQA EM | 12.47 % | −11.558 pp | **−12.021 pp** | −1.321 pp | **NO** (9.1× margin) |
| MMLU-content | 39.00 % | −6.246 pp | **−6.801 pp** | −1.024 pp | **NO** (6.6× margin) |
| *NQ-open EM (DEMOTED, descriptive)* | *23.71 %* | *−7.396 pp* | *−8.255 pp* | *−0.970 pp* | *NO* |

Both K1 clauses fail:
* **clause (a)** "NI also accepts on ≥ 3 of the 4 axes" → NI accepts on **0/4**.
* **clause (b)** "disagreement cells ≤ 1" → **3** disagreement cells.

Since K1 requires **both**, and both fail by a wide margin, **A04 is not killed by K1.**
The direction survives its cheapest gate.

**This is not the same as A04 being alive.** See §6: the surviving claim is narrower than the one
in `A04_GATE_DESIGN.md` §1, and **K2 — the most likely killer — remains untested and is now known
to be untestable with any data currently on disk.**

## 2. K3 — does not fire; provisionally cleared, but not for the gate's arm

Intact 1B calibrated residuals (pre-registered `split` convention):
TriviaQA **40.431 pp**, PopQA **13.205 pp**, MMLU-content **10.239 pp**, NQ-open 9.695 pp.
K3 needs ≥ 3 of 4 axes below 5 pp; **0 of 4** are. So the scale has ample dynamic range.

**Caveat, load-bearing:** K3 is cleared for the arm measured here (`keep7+fresh2`), not for the
`j` the gate would actually use. A shallower cut (recommended `j=12`) has a *higher* absolute
capability and therefore *more* headroom, so K3 is very unlikely to fire there — but that is an
inference, not a measurement. **UNVERIFIED for j=12.**

## 3. K2 — out of scope, and now known to be worse than the design assumed

Pilot Zero **cannot** clear or fire K2: K2 is a between-seed variance statement and Pilot Zero has
one seed. The design already said this.

**New, worse:** the `DistributedSampler(seed=)` defect (fixed today; see
`SEED_SEMANTICS_DEFECT.md`) means that **no true run-to-run variance measurement exists anywhere
in this repo**, and none can be extracted retroactively — the pre-fix trainer gave every seed a
byte-identical data order, so the only multi-"seed" evidence is fresh-block **init** variance,
a strict subset of run-to-run variance. Any `sd_run` borrowed from pre-fix runs is an
**underestimate**, so a K2 "pass" using it would be unearned. Pilot One must run under the fixed
code. This makes Pilot One strictly more necessary than the design's §6.2 assumed.

---

## 4. Convention sensitivity — the verdict survives **all five** conventions

A01 established that the longest-option content null is itself convention-dependent, with a
**25.76 pp** spread, and that under `credit` 5 of 6 arms flip from "above null" to significantly
**below** null. So this check is mandatory, not decorative.

I recomputed all five conventions **on the 1B item set** (n = 14,042) with A01's canonical
`longest_option_vector`, and reproduced A01's values bit-for-bit; my measured spread is
**25.758438968807862 pp**, matching A01's reported 25.76 pp.

| convention | MMLU null | intact residual | Δ_MMLU | K1 fires? |
|---|---:|---:|---:|:--:|
| **split** (pre-registered) | 0.284450 | 10.239 pp | 1.024 pp | **NO** |
| first | 0.281085 | 10.575 pp | 1.058 pp | **NO** |
| last | 0.282154 | 10.469 pp | 1.047 pp | **NO** |
| credit | 0.453710 | **−6.687 pp** | **−0.669 pp** | **NO** |
| wrong | 0.196126 | 19.071 pp | 1.907 pp | **NO** |

**The K1 verdict does not flip under any convention**, and it also survives dropping the
degenerate axis entirely (`K1_survives_excluding_degenerate_axes: false` under every convention,
i.e. K1 still does not fire).

**Why it is robust — and this is the important structural point:** residual = reported − null, and
the *same* input-blind null applies to both arms on the *same* item set, so

```
residual(arm) − residual(intact)  =  reported(arm) − reported(intact)
```

The null **cancels exactly** in the quantity NI tests. The convention therefore moves only `Δ`
(= 0.10 · residual(intact)), never the measured difference. Since the observed gaps exceed every
convention's `Δ` by 6–9×, no admissible convention can rescue NI acceptance. A convention flip
would have needed the gaps to be *near* the margin; they are nowhere near it.

### 4.1 A degeneracy the convention check exposes (worth recording)

Under the **`credit`** (oracle tie-break) convention the **intact** 1B model itself scores
**below** its own MMLU null (0.3868 vs 0.4537), so residual(intact) = **−6.687 pp** and
`Δ = 0.10 · residual(intact)` is **negative**. A negative margin makes `NI(Δ)` demand that the
lower bound exceed a *positive* number — i.e. it silently becomes a **strict-superiority** test and
is no longer a non-inferiority test at all. This is flagged per-cell in the JSON
(`delta_degenerate_negative_margin`) rather than scored silently.

**Implication for the gate design, beyond Pilot Zero:** the frozen rule
`Δ_x = 0.10 · residual(intact, x)` is **ill-defined whenever residual(intact, x) ≤ 0**. I have
**not** changed it (it is pre-registered), but the gate must either state the convention as part of
the definition of `Δ` or add a documented guard. Flagging, not fixing, per instructions.

---

## 5. Recommended `j` — **j = 12** (`keep12+fresh2`, 14 of 16 layers, 87.5 % depth)

`keep7+fresh2` (9/16 = 56.25 % depth) is confirmed a **constant-REJECT rung**: after 52.43 B heal
tokens it recovers 12–39 % of the intact residual and NI rejects by 6–9× the margin on every axis.
A rule tested only there can never be *observed to accept*, so the disagreement is automatic and
carries no information about the rule's discrimination.

Evidence for the choice — the only measured depth→recovery curve in the repo
(`A01/evidence/gate3_content_null_conventions.json`, 7B, MMLU-content, bf16, split-tie null,
recomputed here):

| arm | depth kept | content_norm | residual | frac of intact residual |
|---|---:|---:|---:|---:|
| 7B base | 100 % | 0.470588 | 18.614 pp | 1.0000 |
| 7B shortgpt16 @200k | 50.0 % | 0.401154 | 11.670 pp | 0.6270 |
| 7B keep14 @200k | 50.0 % | 0.383208 | 9.876 pp | 0.5306 |
| 7B keep12 @124k | 43.8 % | 0.362911 | 7.846 pp | 0.4215 |
| 7B keep10 @83.5k | 37.5 % | 0.344467 | 6.002 pp | 0.3224 |
| 7B keep8 @121k | 31.2 % | 0.342259 | 5.781 pp | 0.3106 |

Recovery rises monotonically with depth kept over the whole measured range **with no sign of
saturating**, and the deepest rung measured keeps only 50 % of depth. So the shallowest available
cut is the one most likely to reach non-inferiority. `keep12+fresh2` at 1B keeps **87.5 %** —
far shallower than anything on that ladder — and is therefore the candidate most likely to let NI
sometimes accept, which is precisely what makes the disagreement test falsifiable. It also makes
the affordable 20,000-step (5.24 B token) budget plausible, whereas at keep7 even 52.43 B tokens
were insufficient.

**Second choice `j = 10`** (12/16 = 75 %): if keep12 turns out to be a *constant-ACCEPT* rung
(damage too mild ever to trip NI), keep10 brackets the interesting region from below. The gate
should be prepared to bracket.

**Explicitly not recommended: `j = 7`** — the arm just measured, a constant-REJECT rung.

**UNVERIFIED (do not paper over):**
* No 1B arm at keep12 or keep10 exists on **either** disk, so recovery at those depths at 1B is
  **unmeasured**. The recommendation extrapolates from a **7B** ladder.
* That ladder's own confounds are real and documented in `A04_GATE_DESIGN.md` §3: it spans **two
  corpora** (7,570,911 vs 15,491,607 rows) and **unequal steps** (keep14 200k / keep12 124k /
  keep10 83.5k / keep8 121k), and has one draw per rung. It is evidence of **ordering only**,
  never of an absolute recovery level.
* Whether keep12 at 1B is instead a constant-ACCEPT rung is **UNKNOWN**.

---

## 6. What the surviving claim may and may not say

### 6.1 A first-order narrowing: RATIO(ρ=0.85) does **not** disagree with NI

`A04_GATE_DESIGN.md` §1 names **two** incumbent rules: (a) a likelihood/perplexity plateau and
(b) an aggregate retained-accuracy-ratio threshold. I evaluated both.

`RATIO(ρ = 0.85)` at step 200,000: mean over-axes retention ratio = **0.4017**, far below 0.85, so
**RATIO rejects** — it **agrees** with NI. Per-axis ratios: TriviaQA 0.2356, PopQA 0.2542,
MMLU-content 0.8385, NQ-open 0.2784. It rejects at every one of the 8 arm×checkpoint cells
evaluated (range 0.3694–0.4185).

> **A04 must therefore claim the disagreement for the PLATEAU rule ONLY, not for the
> retained-ratio rule.** The design's §1 wording ("the stopping rules currently in use — (a) …
> and (b) … **accept** recovery runs that a non-inferiority test … rejects") is **too broad** and
> is falsified for (b) by this data. This is a narrowing of A04's own thesis, discovered by its own
> gate, and it should be carried into any writeup.

Note that only MMLU-content is anywhere near ρ (0.8385), which is exactly the axis whose null is
convention-fragile — a coincidence worth watching, not a result.

### 6.2 Honest caveats carried forward (not dropped)

1. **The §6.1 band `(1.388%, 3.297%]` is post-hoc.** The design says so and I repeat it: that band
   was computed *after* seeing the numbers, so it is **illustrative, not confirmatory**. What is
   confirmatory here is narrower and I state it precisely: `T = 2.0 %/5k` was committed in git as
   `d1ba737` (2026-08-09 23:02:27 +0800, verified with `git show`; the design file is byte-unchanged
   since that commit — `git diff d1ba737 -- <A04 dir>` shows only today's new note), and with that
   pre-committed `T` the comparison was then run. That ordering is satisfied. But `T` **lands inside
   a band that was chosen with knowledge of the data**, so the pre-registration is *procedurally*
   correct and *epistemically* weak on this specific point. It does not become strong evidence by
   being re-run.
2. **The grid cannot exercise the rule as written.** 4 points at 47k–53k spacing cannot test a
   5,000-step-resolution rule. I therefore report **both** readings:
   * *unscaled* (the design's own §6.1 arithmetic — compare raw interval improvement to `T`):
     accepts at step 200,000 only (1.388 % < 2.0 %). Rejects at 100k (8.276 %) and 147k (3.297 %).
   * *scaled* (literal units — `T` is "% per 5k steps", so scaled to the interval, e.g. 21.2 % for
     a 53k gap): accepts at **every** checkpoint, including 100k, which is obviously wrong as a
     plateau detector on this grid.
   **K1 does not fire under either reading**, so the conclusion is insensitive to this ambiguity.
   But the disagreement is only *sharp* under the unscaled reading, and a 5,000-step grid
   (`--milestone_every 2500`, as the gate specifies) is required to test the rule as written.
3. **`arm4_peaklr20k` intermediate cells are not settled capability numbers.** A03's STATUS says
   `NOT_YET_JUDGED -- do not cite the intermediate cells`. I independently reproduced the
   documented instability from the shards: TriviaQA EM moves **−1.404 pp** at step205000 then
   **+1.259 pp** at step210000 relative to the step-200000 baseline — non-monotone, both
   directions, consistent with the Adam-moment mismatch in `ARM4_DESIGN.md`. Used for **grid
   coverage only**; every such cell is tagged `arm_capability_unsettled: true` in the JSON. Notably,
   the K1 verdict does not depend on them: NI rejects at all 8 arm×checkpoint cells, including the
   4 stable `cpt20k` ones (TriviaQA frac recovered 0.2187–0.2426).
4. **NQ-open is DEMOTED and excluded from decision cells**, per design §5.2 (item-level 95 % CI
   half-width 1.459–2.063 pp at n = 3,610 exceeds its own Δ = 0.970 pp). Reported descriptively; it
   happens to reject too, but it carries no decision weight.
5. **K1's own "≥ 24 evaluated cells" precondition is NOT met by Pilot Zero.** Pilot Zero evaluates
   24 decision cells in total, but PLATEAU is *defined* at only **3** of them (1 arm × 1 checkpoint
   × 3 decision axes), because no in-domain PPL exists on disk for `cpt20k` or `arm4_peaklr20k`
   (`olmo2_ppl_results/` holds only `1B_base_full` and `1B_keep7_step{50000,100000,147000,200000}`
   — verified by `ls` on zwfy6). Cells with no PPL **cannot** form a PLATEAU-vs-NI disagreement and
   are not counted as such. The honest statement is: K1's clause (a) fails decisively on the cells
   that exist, which suffices **not** to fire K1, but this is not the 4-arm × 6-checkpoint family
   the real gate would evaluate.

---

## 7. Provenance — every number traces to a file I opened

* **PPL trajectory**: recomputed as `exp(Σ sum_nll / Σ n_tokens)` from
  `zwfy6:olmo2_ppl_results/{1B_base_full,1B_keep7_step50000,1B_keep7_step100000,1B_keep7_step147000,1B_keep7_step200000}/shard{0..7}of8.json`.
  Asserted per dir: **8/8 shards, shard indices exactly {0..7}, `n_tokens = 8,384,512`,
  `n_windows = 4,096`, `n_shards = 8`**. Recomputed PPL agrees with each `summary.json` to
  < 1e-9 (intact 10.641583879388614; 50k 17.619441896079884; 100k 16.161295049729876;
  147k 15.628480830626273; 200k 15.411630407090653). `val_path = data/dolmino_now_val.npy`,
  ckpts `outputs/olmo2_probe2_1B_keep7fresh2_16card/step*.pt`.
* **Capability**: merged from per-example shards, **8/8 asserted with exact item counts**
  (MMLU 14,042 / TriviaQA 17,944 / PopQA 14,267 / NQ-open 3,610), duplicate-`item_id` check, and
  shard-index-gap check. Any shortfall raises. Dirs:
  `olmo2_mmlu_content_results/{A03_1B_base,A03_1B_keep7_step200k,A03_1B_arm3_cpt_step{205,210,215}000,A03_1B_arm4_peaklr_step{205,210,215}000}`
  and `olmo2_closedbook_results/{A03_1B_base,A03_1B_keep7_step200k,A03_1B_arm3_cpt_step{205,210,215,220}000,A03_1B_arm4_peaklr_step{205,210,215}000}` (+ `_nq` variants).
* **Scorers/nulls imported, never reimplemented**: `best_constant_qa`, `longest_option_vector`,
  `best_constant_letter`, `paired_bootstrap` imported from
  `A03/code/analyze_1b_knowledge_floor.py`. MMLU `content_norm` correctness is recomputed from the
  stored per-option scores and **asserted equal** to the harness's own stored `correct` flag on all
  14,042 items.
* **Cross-check against A03's published cells**: my recomputed `reported`, `reported_intact` and
  `null` match `A03/evidence/a03_1b_floor_nulls_4axes.json` **exactly (Δ = 0.00e+00)** on all four
  axes; all five MMLU convention nulls match to < 1e-12.
* **Pre-registration**: `git show d1ba737` confirms `T = 2.0 %`, `ρ = 0.85`,
  `Δ_x = 0.10 · residual(intact, x)` present at 2026-08-09 23:02:27 +0800.
  `T`, `ρ`, `Δ` are module-level constants in the script with **no CLI override** — they cannot be
  tuned by an invocation. **Nothing was tuned to obtain this result.**
* **Construct-appropriate nulls used throughout; "above chance" never used.** MMLU-letter
  always-D **0.2689** (not 0.25) is recorded but the interface is BANNED as an axis by the design.
  BoolQ always-B **0.6217** (not 0.50) is recorded for the record and marked
  `UNVERIFIED_here: true` — it is quoted from `A04_GATE_DESIGN.md` §4.2 and was **not** recomputed
  in this pass (no BoolQ 1B shards were opened).

## 8. Complete list of UNVERIFIED / not-established items

1. **`sd_run` for any 1B recovery arm** — no multi-seed 1B run exists on either disk, and (new
   today) no *true* seed-variance run exists anywhere in the repo at all. K2 untested.
2. **Recovery level at `j=12` or `j=10` at 1B** — no such arm on either disk. The `j`
   recommendation is a 7B extrapolation.
3. **Whether 20,000 steps (5.24 B tokens) suffices** at the recommended `j` — unknown.
4. **PPL for `cpt20k` / `arm4_peaklr20k`** — does not exist, so PLATEAU is undefined for those
   7 checkpoints; they contribute NI/RATIO evidence and grid coverage only.
5. **BoolQ null 0.6217** — quoted from the design doc's §4.2, not recomputed here.
6. **Whether the disagreement replicates at 7B** — out of scope, unfunded.
7. **`arm4_peaklr20k` intermediate capability numbers** — A03 STATUS `NOT_YET_JUDGED`; instability
   independently reproduced here (±1.3 pp, both directions). Not settled.
8. **A04's §1 claim for the retained-ratio rule (b)** — actively **falsified** at this arm (RATIO
   rejects, agreeing with NI), not merely unverified.

---

## 9. Recommendation

**Do not archive A04 — K1 did not fire.** But do not authorise the full gate either.

1. **Narrow the claim now** to the PLATEAU rule only (§6.1). The two-rule version is falsified.
2. **Pilot One is now strictly mandatory and strictly more valuable than the design assumed**,
   because K2 is the most likely killer *and* the repo has no usable `sd_run` at all. It must run
   **under the seed fix** committed today, at the recommended `j = 12`, S = 3 seeds. Est. 135 GPU-h
   (design §6.2), ≈ 3–5 % of the full gate cost, and it buys the K2 answer.
3. **Do not authorise the 1,077–4,309 GPU-h gate until `sd_run` is known.**
4. Before Pilot One, decide the `Δ` guard for `residual(intact) ≤ 0` (§4.1) — a pre-registration
   repair, to be made explicitly and recorded, not silently.
