---
gate: A01 gate-1 depth-curve (all four families, thresholds pinned to a single layer)
date: 2026-08-09
revised: 2026-08-10 — verdict PARTIALLY RETRACTED, see banner below
nodes: .21 (Llama-2, wzc1) / .73 (Llama-3 + OLMo-2) / .82 (Qwen3 + OLMo-2 fine grain), all zwfy6 except .21
n: 14042 per arm, 0 nan, 8/8-shard + exact-n asserted before every merge
verdict: ⚠️ SUPERSEDED IN PART. The per-family single-layer jumps stand. "letter is a
  STEP function of depth" as a FAMILY-GENERAL functional-form claim is RETRACTED
  (2026-08-10), and "Llama-2 content is strictly monotone" is CORRECTED.
  Replacement: TCODEX_AUDIT_RESPONSE.md §1 and §2.
---

> # ⚠️ RETRACTION BANNER — 2026-08-10
>
> An independent skeptical audit
> (`proposal/active/A03-parametric-vs-external-memory/evidence/TCODEX_AUDIT_20260810.md`
> §2.1, verdict **Major revision**) attacked two things in this file. Both attacks
> are **accepted**. The replacement analysis, with the recomputed numbers, is
> **`TCODEX_AUDIT_RESPONSE.md` §1 (R-1) and §2 (R-2)**; the machine-readable
> recompute is `evidence/a01_audit_response_recompute.json`, produced by
> `code/a01_audit_response_recompute.py`.
>
> **1. "letter is a STEP function of depth" is RETRACTED as a family-general claim.**
> §3 below says the Llama-2 gap-fill at k=8,12,18,22,26 was "running on `.21` to
> confirm the dip is real". **It finished.** All five arms are on wzc1
> (`gate1_dmg_llama2_7b_depth_gap2_k{8,12,18,22,26}`) and they confirm the dip is
> real and larger than described. On the full 15-depth Llama-2 grid, with exact
> McNemar on every adjacent step and BH at α=0.05 across the 14 steps:
> **6 of 14 steps are decreases and all 6 are BH-significant**; there are
> **5 BH-significant direction reversals** (7 raw); and the floor verdict
> **crosses the floor 4 times** (BELOW×6 → above×3 → BELOW → AT → above×4).
> One of four families is a clean counterexample, so "family-general" is dropped.
> The three per-family jumps themselves (Qwen3 k24→k25 **+48.02 pp**, Llama-3
> k17→k18 **+30.34 pp**, OLMo-2 k18→k19 **+26.68 pp**) are unaffected and stand.
>
> **2. "Llama-2's content curve is strictly monotone" is FALSE as printed.** The
> §1 table row said "yes, **strictly**". Over the full k4…k31 grid the content
> curve has **two decreases**: k8→k10 `0.254237 → 0.253027` (−0.121 pp, McNemar
> p=0.749) and k10→k12 `0.253027 → 0.252884` (−0.014 pp, p=0.984). The word
> "strictly" is wrong. The true and stronger statement: **content decreases twice,
> both by <0.13 pp and both indistinguishable from zero, and 0 of its 8
> BH-significant steps is a decrease — while letter's 6 decreases are ALL
> BH-significant.** The letter-vs-content contrast survives and sharpens.
>
> Nothing else in this file is retracted. The numbers below are as measured; the
> §1 "monotone? yes, strictly" cell and the §3 "in flight" note are the two
> factual errors, corrected in place with markers.


# A01 gate-1 — depth curve, four families, thresholds pinned

Damage = front-N layer truncation of the intact base. **No fresh block, no heal** —
deliberately the worst case, so the curve isolates depth from training. Floor = the
MMLU letter best-constant null **0.2689** (always-D), not chance 0.25.

## 1. The headline: two interfaces, two functional forms

This is the finding, and it was not visible until the curve was filled in at
single-layer resolution.

**Letter accuracy is a step function.** It sits pinned at the floor across a wide
depth range, then jumps 30–48pp at ONE layer, then plateaus:

> ⚠️ **"is a step function" RETRACTED as a family-general form (2026-08-10).** Read
> this table as **"three of four families show a single-layer jump far larger than
> anything the content readout does at the same layer"** — that is what it measures
> and that stands. It does NOT license "letter is a step function of depth".
> Llama-2 (row 4) is a counterexample, and even the three clean families have
> 9–13 raw letter reversals over their grids (box below §1's second table).

| family | L | pinned-at-floor range | **transition** | jump | post-jump plateau |
|---|---:|---|---|---:|---|
| **Qwen3-8B-Base** | 36 | k=4…24, letter **0.2295–0.2301** (span 0.06pp over 20 layers) | **k24→k25** | **+48.0pp** | 0.647–0.730 |
| **Llama-3-8B** | 32 | k=4…16, 0.2319–0.2698 | **k17→k18** | **+30.3pp** | 0.573–0.620 |
| **OLMo-2-7B** | 32 | k=4…17, 0.2312–0.2695 | **k18→k19** | **+26.7pp** | 0.588–0.601 |
| Llama-2-7B | 32 | k=4…14, 0.2305–0.2534 | non-monotone, see §3 | — | 0.388–0.432 |

Qwen3 is the extreme case: **twenty consecutive depths agree to within 0.06pp**, then
one layer moves the metric 48 points. That is not a capability curve; that is a
switch.

**Content accuracy is smooth and monotone.** Same models, same items, same forward
passes — only the readout differs:

| family | content_norm at min depth → max depth | monotone? | largest single-layer jump |
|---|---|---|---:|
| Qwen3-8B | 0.2430 (k4) → 0.4691 (k34) | yes, no reversals | +5.0pp (k29→30… k32→34) |
| Llama-3-8B | 0.2535 (k4) → 0.4168 (k30) | yes (one −0.4pp blip k8→12) | +3.3pp |
| OLMo-2-7B | 0.2545 (k4) → 0.4419 (k30) | yes | +3.5pp |
| Llama-2-7B | 0.2441 (k4) → 0.3902 (k31) | ~~**yes, strictly**~~ **CORRECTED 2026-08-10: NOT strictly** — two decreases, k8→k10 −0.121pp (p=0.749) and k10→k12 −0.014pp (p=0.984); 0 of 8 BH-significant steps decreases. See banner + `TCODEX_AUDIT_RESPONSE.md` §2 | +2.9pp |

> ⚠️ The "yes, no reversals" / "yes" cells for the other three families are also
> loose as printed. Measured **raw** reversal counts (reversal = sign change in the
> sequence of adjacent-step deltas, exact-zero steps dropped; no significance
> filter), over each family's full merged grid:
>
> | family | n depths | letter reversals | letter decreases | content reversals | content decreases | max content decrease |
> |---|---:|---:|---:|---:|---:|---:|
> | Qwen3-8B | 20 | **13** | 7 | **4** | 2 | 0.171 pp |
> | Llama-3-8B | 17 | **9** | 8 | **3** | 3 | 0.897 pp |
> | OLMo-2-7B | 22 | **11** | 9 | **5** | 4 | 0.256 pp |
> | Llama-2-7B | 15 | **7** | 6 | **2** | 2 | 0.121 pp |
>
> So "content is smooth and monotone" is true of the *magnitudes* (every content
> decrease anywhere is ≤0.90 pp, most ≤0.26 pp) and of the significant structure,
> but **not literally of every adjacent pair in any family**. Letter's largest
> single decreases by contrast are Qwen3 −7.49 pp, Llama-2 −7.52 pp,
> OLMo-2 −3.83 pp, Llama-3 −2.96 pp. Write the contrast as
> "content's reversals are within noise, letter's are not", never as
> "content never decreases".



At the very layer where Qwen3's letter metric moves 48pp (k24→k25), its content
metric moves **+1.4pp** (0.3045 → 0.3180). Same weights, same data, same items.

## 2. Why this matters for A01

A01's protocol claim is "report every construct against a construct-appropriate null."
This curve supplies the mechanism-free version of that argument:

* **The letter interface is not measuring a graded quantity.** Below threshold it is
  a constant predictor (a degenerate readout), above threshold it is a competent one.
  Averaging across arms that straddle the threshold mixes two regimes; interpolating
  between them is meaningless.
* **The content interface tracks something graded.** Its monotone rise with depth is
  what a "capability degrades with damage" claim needs in order to be a claim about
  capability at all.
* **Consequence for damage-scaling studies (incl. our own Paper B):** any arm whose
  depth lands below its family's threshold contributes a floor value, not a
  measurement. A regression of letter-accuracy on damage that includes sub-threshold
  rungs is fitting a step, and its slope is an artifact of how many rungs fell on
  each side.

This strengthens gate-1's damaged-arm result (`GATE1_DAMAGED_VERDICT.md`, 6/6 below
floor): those arms were at keep8/keep12, i.e. **deep inside every family's
sub-threshold range**. Their being at floor is now explained, not just observed.

## 3. Llama-2 is the anomaly, and the anomaly is informative

> **⚠️ UPDATED 2026-08-10 — the gap-fill LANDED and it is worse than described here.**
> The paragraph below ("Gap-fill … is running on `.21`") was stale: the five arms
> `gate1_dmg_llama2_7b_depth_gap2_k{8,12,18,22,26}` are on wzc1 and were never
> reported. With them, the Llama-2 grid is 15 depths and the curve is not
> "one dip at k20/k24" but **6 BH-significant decreases and 5 BH-significant
> direction reversals**, with the floor verdict crossing 4 times:
>
> | k | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 | 31 |
> |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
> | letter | .2305 | .2534 | .2415 | .2493 | .2295 | .2307 | .3289 | .3359 | .3054 | **.2302** | .2723 | .3945 | .4315 | .4229 | .3877 |
> | vs floor | BELOW | BELOW | BELOW | BELOW | BELOW | BELOW | above | above | above | **BELOW** | AT | above | above | above | above |
>
> **k22 = 0.230238 (−3.867 pp, p=9.1e−26) falls all the way back BELOW the floor
> after three consecutive above-floor depths (k16/k18/k20)**, and k24 is only AT
> the floor (p=0.371) before k26 jumps to +12.56 pp. This is the fact that kills
> "family-general step function". See `TCODEX_AUDIT_RESPONSE.md` §1.
>
> Also: "Its content curve … is **strictly monotone**" — see the retraction banner,
> the word "strictly" is false over the full grid. The (0.2650 → 0.2877 → 0.3139 →
> 0.3548) parenthesis is the k16/k20/k24/k28 subset only.

Llama-2's letter curve is **non-monotone**: it rises off the floor at k16 (0.3289,
+6.0pp), then *falls back* toward the floor at k20 (0.3054) and k24 (0.2723,
+0.3pp — essentially floor again), then rises to 0.4315 at k28.

Its content curve over the same range is ~~strictly monotone~~ **monotone at every
significant step** (0.2650 → 0.2877 → 0.3139 → 0.3548). So the non-monotonicity
lives entirely in the letter readout, not in the model's underlying competence —
which is exactly the interface-artifact reading. ~~Gap-fill at k=8,12,18,22,26 is
running on `.21` to confirm the dip is real and not two adjacent noisy points.~~
**Gap-fill DONE (see box above): the dip is real, BH-significant, and there are
three separate descending runs, not one.**


Note also that Llama-2's post-threshold plateau (0.39–0.43) is far below the other
three families' (0.57–0.73), consistent with Llama-2-7B simply being a weaker model
on MMLU — its *intact* letter accuracy is 0.4100 (per `GATE1_VERDICT.md`'s intact
leg), so its truncated-k28 value 0.4315 is already at intact level.

## 4. Threshold location vs relative depth

| family | L | threshold k | k/L |
|---|---:|---:|---:|
| Qwen3-8B | 36 | 25 | **0.69** |
| Llama-3-8B | 32 | 18 | **0.56** |
| OLMo-2-7B | 32 | 19 | **0.59** |
| Llama-2-7B | 32 | (non-monotone) | — |

The three clean families put the switch at 56–69% of depth. Do **not** over-read this
as a law from n=3; report it as "the transition sits past the midpoint in all three
families we could pin," not as a constant fraction.

## 5. What this does NOT show

* **Not a mechanism.** The curve says nothing about *why* the letter readout flips at
  one layer. The tie-based explanation was already triple-falsified
  (`STATUS.json:must_drop`); this curve does not resurrect it, and does not replace it.
* **Not a statement about healed models.** Every arm here is truncation-only. A healed
  arm at the same depth can sit elsewhere; Paper B's keep8 healed arm reaches 0.2550,
  which is still sub-floor, but that is a separate measurement.
* **Not a per-task claim.** These are full-MMLU aggregates. Whether the step is
  uniform across the 57 subjects is unexamined.

## 6. Provenance

* Driver: `scripts/_a01_gate1_depth_curve.sh` (generic, env-parameterised by
  `FAMILY`/`MODEL_PATH`/`KEEPS`/`TAG`)
* Truncation loader: `scripts/eval_olmo2_probe2_ppl.py::load_truncated_any_family`
  (slices `model.model.layers[:N]`, resets `config.num_hidden_layers`)
* Harness: `scripts/eval_olmo2_mmlu_content.py --any_family --keep_front_layers N`,
  `chat_template=False`, no system prompt, no few-shot
* Result dirs: `olmo2_mmlu_content_results/gate1_dmg_{qwen3_8b,llama3_8b,olmo2_7b}_depth*_k*/`
  on **zwfy6**; `gate1_dmg_llama2_7b_depth*_k*/` on **wzc1** (.21)
* Arm count: Qwen3 14, OLMo-2 22, Llama-3 12, Llama-2 10 (+5 in flight) = 58 arms × 14042 items
* A note on node python: `.21`'s `.venv/bin/python` is a broken symlink to a
  torch-less py3.11; use `/opt/conda/envs/torch-base/bin/python` there too (first
  launch attempt died on `ModuleNotFoundError: numpy` across all 8 shards).
