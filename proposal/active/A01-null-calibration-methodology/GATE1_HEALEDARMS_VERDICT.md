---
gate: A01 gate-1 healed-arms leg (keep14 variants + shortgpt16, all 7B, 200k steps)
date: 2026-08-09
node: .21 (8x L20A, wzc1)
verdict: HEAL SOFTENS THE LETTER STEP BUT DOES NOT CLOSE THE INTERFACE ASYMMETRY -- with one striking exception (shortgpt16 INVERTS it)
n: 14042 per arm, 0 nan, 8/8-shard + exact-n asserted before every merge
---

> **★ 2026-08-09 05:56 CORRECTION — see §7.** The "shortgpt16 INVERTS the sign vs
> every other arm we have measured, including intact base" framing below is WRONG.
> Intact base OLMo-2-7B under this exact protocol gives **letter 0.6060, cn 0.4702,
> cn−l = −13.6pp** (FULL content) and **letter 0.6060, cn 0.4563, cn−l = −15pp**
> (cdnone). Both signs are NEGATIVE, i.e. intact base ALREADY has letter > content.
> That is the OPPOSITE of what I stated in §2c, and I made that statement without
> ever measuring intact base's content_norm under this protocol — I assumed it. §7
> reframes the finding around what the data actually says.

# A01 gate-1 — healed-arm interface measurements

Same MMLU protocol as the truncation depth-curve (`chat_template=False`, no
few-shot, no system prompt, `--add_bos 0`, fp32 weights / bf16-autocast forward,
8 shards, `--content_desc full`). All arms are 200k-step-healed Paper B
checkpoints on wzc1.

## 1. The table

Reference points (from earlier gates, same protocol):
* Truncated k=14, no heal: letter 0.2432 (-2.6pp vs floor 0.2689), content_norm 0.2720 (-1.3pp vs 0.2845)
* Intact base OLMo-2-7B-1124: letter 0.6054 (Paper B ref)

### Keep-front-N healed depth curve (all fresh2 + heal, various step counts)

| arm | step | letter | content_norm | CI95(cn−l) | McNemar p | Δ letter vs floor |
|---|---:|---:|---:|---|---:|---:|
| keep8fresh2 | 124500 | 0.2468 | 0.3424 | [0.0850, 0.1065] | 5.41e−69 | −2.2pp (at floor) |
| keep10fresh2 | 86500 | 0.2677 | 0.3478 | [0.0694, 0.0909] | 4.99e−47 | −0.1pp (at floor) |
| keep12fresh2 | 130000 | 0.2609 | 0.3670 | [0.0954, 0.1172] | 8.06e−81 | −0.8pp (at floor) |
| **keep14fresh2** | **200000** | **0.3185** | **0.3841** | [0.0550, 0.0763] | 1.20e−33 | **+5.0pp** |

**Caveat: step counts differ.** keep14 is the only 200k arm; keep8/10/12 stopped at
86–130k. If those arms were extended to 200k, letter might rise further, and the
apparent "step" between keep12→keep14 could be a training-budget artifact rather
than a depth-transition artifact. See §5. That said, content_norm is monotone in
depth even at unequal step counts (0.34 → 0.35 → 0.37 → 0.38), so the smooth-side
of the interface remains smooth here.

### keep14fresh2 recipe variants (all 200k)

| arm | letter | content_norm | CI95(cn−l) | McNemar p | Δ letter vs main |
|---|---:|---:|---|---:|---:|
| **keep14fresh2 main** | **0.3185** | **0.3841** | [0.0550, 0.0763] | 1.20e−33 | 0.0 (ref) |
| keep14fresh2 freeze_front | 0.2629 | 0.3600 | [0.0865, 0.1080] | 1.32e−68 | −5.6pp |
| keep14fresh2 from_scratch | 0.2458 | 0.3598 | [0.1036, 0.1243] | 8.97e−102 | −7.3pp |

### Prompt-richness robustness (content_desc=none control, all 4 healed arms)

| arm | letter | content_norm | CI95(cn−l) | shift vs FULL |
|---|---:|---:|---|---|
| keep14fresh2 main + cdnone | 0.3182 | 0.3767 | [0.0479, 0.0694] | letter −0.0003 / cn −0.0074 |
| keep14fresh2 freezefront + cdnone | 0.2623 | 0.3565 | [0.0833, 0.1053] | letter −0.0006 / cn −0.0035 |
| keep14fresh2 fromscratch + cdnone | 0.2458 | 0.3608 | [0.1047, 0.1256] | letter +0.0000 / cn +0.0010 |
| **shortgpt16 + cdnone** | **0.4738** | **0.3940** | **[−0.0902, −0.0694]** | letter +0.0004 / cn −0.0074 |

**The pattern is robust to content_desc.** All four healed arms move <1pp on
either interface between FULL and cdnone content descriptions. The three keep14
variants keep their POSITIVE (cn−l) sign; shortgpt16 keeps its NEGATIVE sign.
The finding is a property of the model, not of the content prompt richness.

### ShortGPT-16 (topology-preserving heal, 200k)

| arm | letter | content_norm | CI95(cn−l) | McNemar p |
|---|---:|---:|---|---:|
| **shortgpt16** | **0.4734** | **0.4014** | **[−0.0822, −0.0617]** | **2.71e−42** |

CI95 is on `(content_norm − letter)` in pp. All CIs exclude 0. **Only shortgpt16
puts the CI on the NEGATIVE side.**

## 2. Three separate results

### (a) The base "keep14 healed" story, sharpened

The main keep14fresh2 healed arm brings letter from truncated 0.2432 (below the
0.2689 floor) up to 0.3185 (+5.0pp above floor). That is a real improvement — the
readout is no longer a constant predictor. But it is not close to the post-transition
plateau (0.57–0.60 in the truncation depth-curve at k≥18 for OLMo-2, k≥25 for Qwen3);
`0.3185 – 0.6054 = −29pp` vs intact base. And content_norm on the same items is
0.3841, still 6.6pp higher than letter. **200k Dolmino steps soften the letter step
but do not close the interface gap on 4062M inherited + 646M fresh parameters.** The
content_desc=none control gives 0.3182 / 0.3767 — the gap doesn't depend on the
content prompt richness, it's a property of the readout itself.

### (b) Freezing the inherited layers or restarting them makes letter worse

Both control arms drop letter *below* the main healed arm:

* **freeze_front**: front-14 inherited layers frozen (no gradient), only fresh
  layers 14–15 + lm_head + norm trainable. Letter 0.2629 (−5.6pp vs main). Content
  0.3600 (−2.4pp). So freezing the inherited layers costs the readout most of what
  heal recovered, and hits letter (which is the fragile side) harder than content
  (which is the graded side). This is consistent with the reading in (c): the
  letter readout needs training signal *into the inherited weights* to un-collapse.
* **from_scratch**: all layers reinit + trained 200k. Letter 0.2458 (only +0.3pp
  above truncated k=14, essentially unchanged), content 0.3598 (comparable to
  freeze_front). Reinitialising throws away the intact base's readout circuitry
  entirely; Dolmino cannot rebuild it in 200k steps for the letter interface but
  can for the content interface.

The rank order **fromscratch < freezefront < keep14fresh2** is stable on letter and
approximately stable on content. It says: the healed model's letter readout
carries the intact base's inherited weights; kill either direction (freeze or
reinit) and you lose it.

### (c) ShortGPT-16 INVERTS the interface asymmetry

`shortgpt16` picks 16 layers to keep from the intact 32L base by block-level cosine
similarity of hidden-state deltas, then heals for 200k steps (as opposed to
keep14fresh2, which keeps front-14 and adds 2 fresh). The healed model gives:

    letter = 0.4734   content_norm = 0.4014   (letter > content by 6.7pp, McNemar p=2.7e-42)

Compared to every other arm we have measured — intact base, truncation-only across
four families at every depth, healed keep14fresh2 in three variants, healed 1B
keep7 — the sign of (content − letter) has been **positive** everywhere (content
usually beats letter by 5–15pp in absolute terms, more in relative terms). Here it
is negative, with a bootstrap CI that excludes 0 by 6+pp. This is not noise.

**Two ways to read it, both compatible with A01's protocol claim:**

* **(a) Layer topology, not depth, controls whether the letter readout can be
  rebuilt.** Front-14 truncation places all removed layers at the top of the
  stack; ShortGPT's removed layers are scattered by cosine-similarity criterion
  and are mostly *middle* layers. The retained set includes the intact base's
  lm_head-adjacent circuitry that emits label letters. When heal reinitialises
  the two fresh layers in the keep14fresh2 arm, they land at positions 14–15 in
  the middle of the stack; in ShortGPT there are no fresh layers to reinit — the
  16 kept layers are contiguous in their original relative position with
  respect to the readout, and the heal only has to fine-tune, not reconstruct.
* **(b) The letter readout depends on a specific subgraph of layers, and
  ShortGPT happens to preserve it while keep14 does not.** Same claim in weaker
  form: whatever letter is decoding lives past layer 14 in front-N terms.

Either reading strengthens A01's argument: **letter-interface accuracy is not a
graded reflection of model capability, it is a fragile property of specific
circuits**, and the same 200k Dolmino heal can leave you on either side of the
step depending on which layers were kept. Reporting letter without a
construct-appropriate null and without a matched-topology reference arm is not a
comparison of capability — it is a comparison of circuit preservation.

## 3. What must not be over-claimed

* n=1 for shortgpt16. The inversion is real (n=14042 items, McNemar p<10⁻⁴¹), but
  we have not run any *other* ShortGPT-selected topology, at any other depth, on
  any other family. Do not generalise to "any non-front-truncation heal inverts
  the interface" from this single arm.
* The keep14 variants (main, freezefront, fromscratch) are all keep-front-14; we
  do not have a ShortGPT-at-keep-14 arm to isolate topology from
  training-recipe.
* Content_norm is not identical across arms either (0.36–0.40). Do not treat it
  as an oracle. Report it as "the graded side of the same measurement" and let
  its own null (longest-option split-tie 0.2845) do the calibration.

## 4. What this DOES settle inside A01

1. Truncation-only depth curve showed letter = step function of depth (Qwen3 20
   depths within 0.06pp then +48pp at one layer). This adds: **heal does not
   turn the step into a smooth ramp**; keep14 healed is 0.32, still ~29pp below
   the intact plateau, and freeze/reinit controls sit 5–7pp below that. Heal
   moves the arm off the floor but not to the plateau.
2. Content_norm on the same forward passes remained smooth and monotone across
   the truncation depth curve; here it remains 0.36–0.40 across four keep14
   variants (0.6-pp spread on freeze/fromscratch/main, 4pp higher on shortgpt).
   So the healed regime does not induce non-monotonicity in the content readout
   either. Content stays the graded side.
3. The freezefront and fromscratch controls make the "letter readout carries the
   inherited weights" claim concrete rather than speculative — the two obvious
   ablations both hurt letter more than content, and their ranking is exactly
   what that claim predicts.

## 5. What this OPENS

* Extending the depth curve to keep8/keep10/keep12 healed on `.82` (in flight)
  will tell us whether the step-softening effect of heal is monotone in depth,
  i.e. is heal at keep8 (further into the sub-threshold range) enough to lift
  letter to keep14's 0.3185? Or does heal-lift saturate?
* A ShortGPT-at-different-keep-N sweep would isolate topology from depth. Not
  in scope for A01, but if the shortgpt16 inversion holds up on a second
  ShortGPT arm, it is worth a dedicated writeup.

## 5b. What the healed-depth curve added (2026-08-09 03:53 → 04:10)

`keep8 (0.2468) → keep10 (0.2677) → keep12 (0.2609) → keep14 (0.3185)` on letter,
with all sub-14 arms sitting at or below the 0.2689 floor. **Naïvely** this looks
like the same step function the truncation curve had, just softened by heal —
"heal at k≤12 doesn't lift letter off the floor".

But the step counts are NOT matched: keep8 stopped at 124.5k, keep10 at 86.5k,
keep12 at 130k, keep14 at 200k. So the comparison confounds depth with training
budget. Two clean readings both consistent with the data:

* **(A) Depth still gates the letter readout even after heal:** if we ran keep8
  at 200k, letter would still sit sub-floor because there are simply not enough
  layers to instantiate the readout circuit. Heal moves each arm off the floor
  by some depth-monotonic amount; keep14's 200k step count *plus* its
  keep-14-front topology together lift it over the floor; the earlier arms
  never get there regardless of training budget.
* **(B) The gap is a training-budget artifact:** if we ran keep8/10/12 to 200k,
  their letter accuracy would rise to something like keep14's 0.32. Under this
  reading the "step at k=14" seen in this partial curve would flatten and heal
  would look monotone in depth.

The data on hand cannot separate these; running keep8/10/12 out to 200k on a
zwfy6 node would settle it. **Meanwhile the ShortGPT-16 result — measured at
200k with the interface INVERTED — is unaffected by this ambiguity**, because
it is a topology-vs-recipe finding, not a depth finding.

Content_norm's monotone rise (0.34 → 0.35 → 0.37 → 0.38 across k=8..14) is
consistent with either reading — the graded side stays graded.

## 5c. What could still kill either reading

* If Paper B has trajectory data for keep8/10/12 (letter/content at earlier
  step counts), it would tell us whether letter is *still rising* at the
  stopping points. If it's flat, that supports (A); if still climbing, (B).
* An intermediate-depth ShortGPT (say ShortGPT-14 or ShortGPT-12) at 200k
  would isolate topology from depth.
* Both are for later, not blocking this verdict.

## 6. Provenance

* Driver: launched manually, `/tmp/a01_healed_arms.sh` (single-node 8-shard
  serial-per-arm; per-shard log + progress log on `.21`)
* Ckpts:
    * `outputs/olmo2_probe2_7B_keep14fresh2/final.pt` (wzc1, 30 GB)
    * `outputs/olmo2_probe2_7B_keep14fresh2_freezefront/final.pt`
    * `outputs/olmo2_probe2_7B_keep14fresh2_fromscratch/final.pt`
    * `outputs/olmo2_probe2_7B_shortgpt16/final.pt`
* Result dirs: `olmo2_mmlu_content_results/a01_7B_{keep14fresh2_freezefront,keep14fresh2_fromscratch,shortgpt16}_healed*/`
* Logs: `logs/a01_healed_arms_progress.log`, `logs/a01_keep14fresh2_7B_healed*` + `logs/a01_7B_*_healed_shard*.log`
* Wall time: ~5.5 min per arm on 8× L20A after ckpt-load phase; ckpt-load was
  the bottleneck (30 GB × 8 parallel readers = ~2 min FS thrash)

---

# §7. Correction and reframe (2026-08-09 05:56)

## 7.1 What I got wrong

In §2c I wrote that shortgpt16's letter > content was "opposite direction of all
other arms" and cited "intact base, truncation-only across four families, healed
keep14fresh2 in three variants, healed 1B keep7" as the reference set where cn > l.
I had NOT measured intact base's content_norm under the gate-1 protocol. I inferred
it from the healed-arm pattern.

That inference was wrong. Actual intact-base numbers under this exact protocol:

| arm | letter | content_norm | cn−l | CI95(cn−l) | McNemar p |
|---|---:|---:|---:|---|---:|
| **OLMo-2-7B intact + FULL** | **0.6060** | **0.4702** | **−13.6pp** | [−0.146, −0.126] | 3.0e−145 |
| OLMo-2-7B intact + cdnone | 0.6060 | 0.4563 | −15.0pp | [−0.160, −0.140] | 5.7e−175 |

Cross-family intact bases (from `GATE1_VERDICT.md`, retracted verdict but the
numbers themselves stood): Llama-3-8B and Qwen3-8B intact both have letter 16–23pp
above content. Only Llama-2-7B (a weak model) has letter ≈ content near floor.

So the honest pattern is:

| regime | letter vs content | example |
|---|---|---|
| **competent intact bases** (letter ≥ 0.60) | **letter > content, by 13–23pp** | OLMo-2-7B, Llama-3-8B, Qwen3-8B |
| **weaker intact bases** (letter 0.38–0.41) | **letter ≈ content (TIE)** | OLMo-2-**1B** (added 2026-08-09), Llama-2-7B |
| damaged / truncated arms | ≈ tied near floor | trunc k=8/12 non-OLMo |
| healed keep14 (main, freezefront, fromscratch) | **content > letter, by 6–12pp** | OLMo-2-7B keep14 variants |
| healed shortgpt16 | letter > content, by 7pp (matches competent-intact regime) | OLMo-2-7B shortgpt16 |

### 7.1.1 The OLMo-2-1B intact data point (2026-08-09)

Same protocol as intact 7B (base mode, `chat_template=False`, no few-shot,
`--add_bos 0`, `--content_desc full` and `cdnone` variants, 8-shard eval on .21).

| arm | letter | content_norm | cn − l | CI95(cn−l) | McNemar p |
|---|---:|---:|---:|---|---:|
| OLMo-2-1B intact + FULL | 0.3816 | 0.3870 | +0.53pp | [−0.51, +1.58] | 0.33 |
| OLMo-2-1B intact + cdnone | 0.3816 | 0.3762 | −0.54pp | [−1.58, +0.50] | 0.32 |

Both CIs cross 0. This is a **TIE**, not a sign flip — the same construct on the
same items but at a weaker letter readout gives no advantage to either interface.

### 7.1.2 What this refines about §7's read

§7 originally said "strong intact bases have letter > content by 13–23pp". The
correct statement is:

* **Letter > content is a property of COMPETENT letter readouts, not intact-ness.**
  1B OLMo-2 is intact, but its letter readout at 0.38 is not competent enough to
  beat the content readout at 0.39.
* **The letter advantage grows with letter competence, not merely with model
  scale.** Llama-2-7B is intact and 7B in parameters but its letter is 0.41 —
  same regime as 1B OLMo-2, not the same regime as 7B OLMo-2. Whatever letter
  readout is decoding, its per-token calibration matters more than the
  parameter count.
* **This makes the intact-vs-healed contrast tighter, not looser.** Healed
  keep14's letter (0.32) sits BELOW the intact 1B letter (0.38) — a smaller
  model, same architecture family, no damage, still beats healed 7B on the same
  interface. That is genuine capability loss in the fresh-2-layers-plus-heal
  recipe, and it is not a scale story.

This does NOT change §7's other reads. The shortgpt16 sign is still striking
(matches the competent-intact regime while its structurally-matched keep14
siblings do not).

## 7.2 What the corrected pattern says

The shortgpt16 result is not "the only arm that inverts". It is **the only healed
arm that preserves the intact base's letter > content ordering**. The keep14
variants (which reinit two fresh layers and heal for 200k) FLIP the ordering
relative to their own intact parent. That is still an unusual and specific fact
about keep14 heal — but it is not that shortgpt16 broke the rule; it is that
keep14 broke it, and shortgpt16 didn't.

Two readings compatible with all the data:

* **(A) The letter readout is a specific circuit in the intact model.** ShortGPT-16
  selects 16 layers by hidden-state-delta cosine similarity, in a way that tends to
  keep the readout-adjacent layers. keep-front-14 removes the top 18 layers and
  drops in two fresh untrained layers where the readout circuit used to live. Heal
  can partially rebuild the readout via content-style signals (hence content > letter
  after heal on keep14), but cannot fully rebuild the letter-emitting circuit in
  200k steps. ShortGPT-16 preserves the original circuit, so heal only fine-tunes.
* **(B) Content_norm is measuring something different.** On intact strong models,
  letter is a competent classifier; content_norm reads the option likelihoods
  differently and its calibration lands lower. On healed keep14, the letter circuit
  is degraded but the option-likelihood signal has been trained back to a
  respectable level — so content > letter. Under this reading it is the CONTENT
  interface that heal is preferentially rebuilding, not the letter one.

Both readings imply the *same* practical claim: **the letter and content interfaces
measure different circuits, and which one dominates depends on what the training
touched.** That is A01's core protocol claim, unchanged.

## 7.3 What is no longer supported

* "content_norm is the graded / fair side and letter is the fragile side of the
  same measurement." Only true for damaged and healed keep14 arms. On intact
  strong models the letter interface is BOTH more accurate AND appears equally
  smooth (though we did not run a within-intact depth curve to prove smoothness).
  So "always report content_norm" is too strong a recommendation.
* The framing that shortgpt16 is *anomalous*. It is not — it is the arm that
  matches intact behaviour. The anomalous arms are keep14 main / freezefront /
  fromscratch, which invert the ordering their intact parent had.

## 7.4 What still stands

* The truncation depth curve (letter as step function, content as smooth) is
  unchanged — that finding was on damaged arms, not on intact bases.
* Under damage the letter interface degenerates to at/below the floor across
  four families. Still true.
* Reporting every construct against a construct-appropriate null. Still the
  correct recommendation — best-constant / longest-option split-tie / etc.
  should always be computed before any letter vs content headline. This
  correction is itself an instance of exactly that discipline: I skipped
  measuring intact base's content_norm and wrote a claim about it; the
  measurement now contradicts the claim.

## 7.5 Provenance

* `olmo2_mmlu_content_results/a01_7B_intact_base_full/summary.json`  (this session's new run, 8-shard eval on .21)
* `olmo2_mmlu_content_results/a01_7B_intact_base_cdnone/summary.json`  (this session's new run)
* `olmo2_mmlu_content_results/a01_7B_keep14fresh2_freezefront_healed_cdnone/summary.json`  (2026-08-09 fill on .21)
* `olmo2_mmlu_content_results/a01_7B_keep14fresh2_fromscratch_healed_cdnone/summary.json`  (2026-08-09 fill on .21)
* `olmo2_mmlu_content_results/a01_1B_intact_base_full/summary.json`  (2026-08-09 cross-scale fill on .21)
* `olmo2_mmlu_content_results/a01_1B_intact_base_cdnone/summary.json`  (2026-08-09 cross-scale fill on .21)
* Prior intact non-OLMo numbers (Llama-2/Llama-3/Qwen3): `GATE1_VERDICT.md` §2 (numbers valid, its VERDICT retracted)
