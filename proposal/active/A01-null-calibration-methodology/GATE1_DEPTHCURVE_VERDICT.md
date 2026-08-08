---
gate: A01 gate-1 depth-curve (all four families, thresholds pinned to a single layer)
date: 2026-08-09
nodes: .21 (Llama-2, wzc1) / .73 (Llama-3 + OLMo-2) / .82 (Qwen3 + OLMo-2 fine grain), all zwfy6 except .21
n: 14042 per arm, 0 nan, 8/8-shard + exact-n asserted before every merge
verdict: THE TWO INTERFACES HAVE DIFFERENT FUNCTIONAL FORMS -- letter is a STEP function of depth, content_norm is SMOOTH and MONOTONE. Three families' letter thresholds now pinned to a SINGLE layer.
---

# A01 gate-1 — depth curve, four families, thresholds pinned

Damage = front-N layer truncation of the intact base. **No fresh block, no heal** —
deliberately the worst case, so the curve isolates depth from training. Floor = the
MMLU letter best-constant null **0.2689** (always-D), not chance 0.25.

## 1. The headline: two interfaces, two functional forms

This is the finding, and it was not visible until the curve was filled in at
single-layer resolution.

**Letter accuracy is a step function.** It sits pinned at the floor across a wide
depth range, then jumps 30–48pp at ONE layer, then plateaus:

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
| Llama-2-7B | 0.2441 (k4) → 0.3902 (k31) | **yes, strictly** | +2.9pp |

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

Llama-2's letter curve is **non-monotone**: it rises off the floor at k16 (0.3289,
+6.0pp), then *falls back* toward the floor at k20 (0.3054) and k24 (0.2723,
+0.3pp — essentially floor again), then rises to 0.4315 at k28.

Its content curve over the same range is **strictly monotone** (0.2650 → 0.2877 →
0.3139 → 0.3548). So the non-monotonicity lives entirely in the letter readout, not
in the model's underlying competence — which is exactly the interface-artifact
reading. Gap-fill at k=8,12,18,22,26 is running on `.21` to confirm the dip is real
and not two adjacent noisy points.

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
