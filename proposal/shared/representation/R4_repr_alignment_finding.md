# R4 — Layer-alignment measurement scaled to 14 models / n=91 pairs

**Purpose**: R3 (`R3_feasibility_ours.md` §3.2) measured layer-wise z-CKA on **5 model
pairs** and drew three conclusions from them. Five pairs is not enough to support any of
them, and one of the three (H2) was drawn from a **single** data point. This note scales
the same measurement to **14 pretrained models / 7 families / depths 12–48L**, giving
**n = 91 pairs**, and tests H1/H2/H3 statistically.

**Script**: `code/repr_alignment_multimodel.py` (`smoke_stitch_cpu.py`
untouched).
**Machine-readable output**: `repr_alignment_results.json`.
**Compute**: 16 GPU jobs on .73 + .82 (8×H20 each); total wall time ≈ 20 min.

---

## 0. Verdicts up front

| | Hypothesis (as stated in R3) | Verdict | The number that decides it |
|---|---|---|---|
| **H1** | z-CKA along relative depth is **U-shaped**: ends high, middle collapses | **HOLDS** | quadratic coef > 0 in **72/91** pairs (binomial p = 2.0e-8); both diagonal ends above the diagonal minimum in **87/91**; median U depth **0.505** z-CKA |
| **H2** | **Depth mismatch hurts more than family mismatch** ("優先同深度 not 同家族") | **DOES NOT HOLD — sign is reversed** | `same_family` **β = +0.171**, p = 0.0012 (QAP p = 0.0022) vs `log(depth_ratio)` **β = −0.056**, p = **0.273** (QAP p = 0.47). Standardised effects **0.341** vs **−0.113** — the *opposite* of H2's criterion. And even the family effect is **not robust** (dies without GPT-2, see §3) |
| **H3** | Cross-family mid-layer CKA sits in an awkward middle band, far above a random floor and far below a same-model ceiling | **HOLDS in substance; the specific 0.35–0.61 interval was too narrow** | observed midband z-CKA: **median 0.503, range 0.087–0.840** (n=91) vs random-init floor **0.091** and same-model adjacent-layer ceiling **0.977** (min 0.923). But only **58%** of pairs (53/91) fall inside R3's quoted 0.35–0.61 |

**Net effect on Paper D**: H1 survives and is the *strongest* result (and it is the one
that hurts a stitching paper the most). H2 — R3's headline "most informative unexpected
finding," which R3 explicitly proposed promoting to a standalone measurement claim — is
**an artefact of n=1** and must be retracted. H3's qualitative reading survives with a
wider band.

---

## 1. Protocol (identical to R3 unless noted) — the rebuttal-proof section

* **Corpus**: WikiText-103 **test** windows from `data/ood_ppl/wikitext103_test.npy`,
  decoded back to raw text with the OLMo-2 tokenizer, split into 60-word chunks,
  `seed=0`, 300 chunks scanned per model.
* **Row alignment across different tokenizers**: rows are **whitespace words**. Each
  model's subword hidden states are **mean-pooled** over the tokens whose character span
  overlaps the word's span (fast-tokenizer `offset_mapping`). This yields two
  `[N_words, D]` matrices whose rows refer to the same words even though the tokenizers
  differ. `add_special_tokens=False`; **one text per forward pass, so no padding token
  ever enters a pool**; `use_cache=False`.
* **N**: 4500 words extracted per model → **global intersection over all 16 extracted
  models** → **N = 4000 words used for every pair** (uniform, so pairs are comparable).
  All 16 models produced **identical** word-key sets, so the intersection cost nothing.
* **Representations**: `hidden_states[0..L]`, i.e. **L+1** points; index 0 = embedding
  output, index L = final layer *before* the final norm. Models in **fp32**.
* **Linear CKA**: `||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F · ||Y_c^T Y_c||_F)` with both
  matrices **column-centred**. Accumulated in fp64; **TF32 disabled**.
* **"z-CKA"** (R3's headline metric, reused verbatim): each `[N,D]` matrix is
  **per-dimension z-scored** (mean/std over the word axis) *before* CKA. This is
  mandatory: LLM residual streams have 2–3 "massive activation" dims carrying >70% of
  total variance, so raw CKA is mostly a statement about those dims. Raw CKA is also
  recorded in the JSON (`midband_raw_cka`) but never quoted.
* **"midband z-CKA"**: mean of the z-CKA matrix over the block
  `{i : i/L_A ∈ [0.25,0.75]} × {j : j/L_B ∈ [0.25,0.75]}`. (This definition was
  *reverse-engineered from and verified against* R3's `smoke_out/*.json` — see §4.)
* **Relative-depth diagonal**: `min(L_A,L_B)+1` points at `i = round(s·L_A/n)`,
  `j = round(s·L_B/n)`, `n = min(L_A,L_B)`.

### Deliberate differences from R3 (both are in the JSON `meta.differences_from_R3`)
1. R3 *asserted* that the word keys of every model matched exactly. Across 7 tokenizer
   families that assert would simply crash, so R4 takes the **global intersection**
   instead. In practice all 16 models agreed, so this changes nothing numerically.
2. R3 used N=4000 for its 1B triple but **N=3000** for its 7B/8B pairs. R4 uses **N=4000
   everywhere**. This is the sole source of the 0.0078 drift on `olmo2_7b:olmo2_1b` (§4).

### Hard gates (both passed before any curve was plotted)
* **CKA identity gate**: every one of the 14 models was run against **itself**;
  `max |M[i][i] − 1|` = **1.78e-7** (z) / **3.76e-7** (raw). PASS.
* **fp32-GPU vs fp64-CPU cross-check** on a mid entry of every pair:
  `max abs diff = 5.08e-8`. PASS.
* Both gates are `assert`s inside `--stage stats`, so a violation stops the pipeline
  rather than producing a plot.

### Model pool (n=14 → 91 pairs)

| family | models (L) |
|---|---|
| olmo2 | OLMo-2-0425-1B (16), OLMo-2-1124-7B (32) |
| llama3 | Llama-3.2-1B (16), Llama3-8B (32) |
| llama2 | Llama2-7B (32) |
| qwen3 | Qwen3-0.6B (28), Qwen3-1.7B (28), Qwen3-4B (36) |
| gpt2 | GPT-2 (12), GPT-2-medium (24), GPT-2-large (36), GPT-2-XL (48) |
| opt | OPT-2.7B (32) |
| openllama | OpenLLaMA-3B-v2 (26) |

Depth ratio spans **1.0–4.0**; the `same_family × same_depth` design is genuinely
crossed (cells n = 72 / 8 / 10 / 1), which is what lets H2 be tested at all.
Controls: 2 random-init models (OLMo-2-1B, Llama-3.2-1B) → 4 floor pairs; 14 self-pairs
→ identity gate + adjacent-layer ceiling.

*One model was skipped*: `openllama_3b` initially failed with `tiktoken` missing; fixed
by `pip install tiktoken` on .73 and re-extracted, so it **is** in the final n=91. No
model was dropped from the analysis.

---

## 2. H1 — U shape: **HOLDS**, and it is the strongest of the three

Fit `zCKA(t) = a + b·t + c·t²` along the relative-depth diagonal (`t` = relative depth,
`min(L_A,L_B)+1` points per pair). U-shape ⟺ `c > 0`.

| statistic | value |
|---|---|
| pairs with `c > 0` | **72 / 91 (79%)**, binomial p = **2.0e-8** |
| pairs with `c > 0` **and** `p_c < 0.05` | **49 / 91** |
| median `c` | **+0.777** (range −0.99 … +2.91) |
| median vertex position | `t` = **0.484** (IQR 0.350–0.703) — i.e. the minimum sits mid-depth |
| **non-parametric** check: argmin of the diagonal is *interior* (0.05 < t < 0.95) | **84 / 91** |
| **non-parametric** check: *both* ends strictly above the diagonal minimum | **87 / 91** |
| U depth (mean of the two ends − minimum) | median **0.505** z-CKA (IQR 0.425–0.612) |
| mean end-point z-CKA vs mean diagonal-minimum z-CKA | **0.746 vs 0.251** |

R3's 5-pair claim ("ends 0.63–0.89, middle collapses to 0.17–0.27") **generalises**.
Median diagonal minimum over 91 pairs is **0.205**, and for **2 pairs it is actually
below the random-init floor** (0.091) — i.e. at mid depth those two real model pairs are
*no more similar than a random network*.

### Excluding the trivial "CKA just decays with layer distance" explanation
Two independent arguments, both required:

1. **Geometric**: the relative-depth diagonal is *constructed* so that
   `|i/L_A − j/L_B| ≈ 0` at every point. Measured max distance anywhere on any diagonal
   is **0.0208**. A model in which CKA is a function of that distance alone therefore
   predicts a **flat** diagonal, not a U. The U cannot be a distance artefact.
2. **Statistical**: regress *every* matrix entry on a **cubic in `|i/L_A − j/L_B|`**
   (fits the matrices with R² ≈ 0.1–0.6), take residuals, and refit the quadratic on the
   diagonal residuals. Result is **identical**: `c > 0` in **72/91** (binomial
   p = 2.0e-8), 49/91 significant, median residual `c` = **+0.776**.

### Leave-one-family-out (H1 is not carried by any one family)
| dropped family | pairs with c>0 | binomial p |
|---|---|---|
| none | 72/91 (0.79) | 2.0e-8 |
| gpt2 | 33/45 (0.73) | 2.5e-3 |
| llama2 | 65/78 (0.83) | 1.8e-9 |
| llama3 | 52/66 (0.79) | 2.8e-6 |
| olmo2 | 47/66 (0.71) | 7.6e-4 |
| opt | 63/78 (0.81) | 3.8e-8 |
| qwen3 | 44/55 (0.80) | 8.7e-6 |

### The 19 non-U pairs are informative, not noise
`corr(c, midband z-CKA) = −0.833`: the U is **deepest exactly where overall alignment is
worst**. The 19 pairs with `c ≤ 0` are the *high*-CKA pairs (midband 0.39–0.83) whose
diagonals are near-flat-and-high — GPT-2↔OPT (0.77–0.80), Qwen3↔Qwen3 (0.63–0.70),
Llama2↔GPT-2 (0.56–0.61). Only 4 of the 19 reach p < 0.05, so most are simply flat.

**Consequence for Paper D (this is the load-bearing one)**: the layer band with the most
abstract semantics — the reason to stitch mid-network at all — is systematically the
band where the two residual streams are *least* alignable, and this now holds across 91
pairs, not 5.

---

## 3. H2 — "depth mismatch > family mismatch": **DOES NOT HOLD. The sign is reversed.**

Main regression, `midband_zCKA ~ same_family + log(depth_ratio)`, n = 91, R² = 0.114:

| term | β | SE | t | p | 95% CI |
|---|---|---|---|---|---|
| const | +0.4932 | 0.0264 | +18.68 | 2.3e-32 | [+0.441, +0.546] |
| **same_family** | **+0.1708** | 0.0512 | +3.34 | **0.0012** | [+0.069, +0.273] |
| **log(depth_ratio)** | **−0.0557** | 0.0505 | −1.10 | **0.273** | [−0.156, +0.045] |

Standardised effect sizes (β·SD(x)/SD(y)): `same_family` **+0.341**,
`log(depth_ratio)` **−0.113**.

R3's own stated criterion was: *H2 holds iff the `log(depth_ratio)` coefficient is
negative and significant **and** its effect size ≥ that of `same_family`.* Observed:
depth is **not** significant (p = 0.27) and its effect is **3× smaller** than family's.
**H2 fails on both clauses.** The depth coefficient does have the predicted negative
sign, so the honest phrasing is "depth mismatch may hurt a little, but family mismatch
demonstrably hurts about 3× more" — the reverse of R3's recommendation to prefer
same-depth over same-family.

Because pairs share models (dyadic non-independence, so the OLS p-value is
anti-conservative), two additional tests:
* **QAP node-label permutation** (5000 relabellings of the 14 models):
  `same_family` p = **0.0022**, `log(depth_ratio)` p = **0.466**. Same conclusion.
* **Node bootstrap** (1000 resamples of models): `same_family` 95% CI
  **[−0.035, +0.323]**, `log(depth_ratio)` **[−0.289, +0.111]**. Note the family CI now
  **includes 0** — see the robustness table below.

### Where R3's H2 came from: a single pair
R3 compared **one** same-family pair (OLMo-2-7B↔1B, midband 0.346) against **one**
cross-family same-depth pair (OLMo-2-1B↔Llama-3.2-1B, 0.467) and concluded depth beats
family. With all 11 same-family pairs visible, **OLMo-2-7B↔1B is the single lowest**:

| same-family pair | midband z-CKA | depth ratio |
|---|---|---|
| gpt2 ↔ gpt2-medium | 0.840 | 2.00 |
| gpt2-medium ↔ gpt2-large | 0.830 | 1.50 |
| gpt2 ↔ gpt2-large | 0.815 | 3.00 |
| qwen3-0.6B ↔ qwen3-4B | 0.701 | 1.29 |
| qwen3-1.7B ↔ qwen3-4B | 0.660 | 1.29 |
| qwen3-0.6B ↔ qwen3-1.7B | 0.633 | 1.00 |
| llama3.2-1B ↔ llama3-8B | 0.589 | 2.00 |
| gpt2-large ↔ gpt2-XL | 0.525 | 1.33 |
| gpt2-medium ↔ gpt2-XL | 0.508 | 2.00 |
| gpt2 ↔ gpt2-XL | 0.505 | 4.00 |
| **olmo2-1B ↔ olmo2-7B** | **0.338** | 2.00 |

`gpt2 ↔ gpt2-large` has depth ratio **3.0** and midband **0.815**; `gpt2 ↔ gpt2-medium`
has ratio **2.0** and **0.840**. Large depth mismatch is plainly compatible with high
alignment. Within same-family pairs alone, depth ratio explains nothing
(β = −0.044, p = 0.743, n = 11); within cross-family pairs alone, likewise
(β = −0.058, p = 0.297, n = 80). **The low 0.346 is an OLMo-2 property, not a depth
property** — OLMo-2-7B has the lowest mean midband z-CKA of all 14 models (0.329,
averaged over its 13 pairs), consistent with it being the only post-norm model here
(its layer RMS stays at 0.07→0.94 while pre-norm models sit at 2–20).

### Robustness: the *family* effect is itself fragile
| subset | n | same_family β (p) | log(depth_ratio) β (p) |
|---|---|---|---|
| all | 91 | **+0.171 (0.0012)** | −0.056 (0.273) |
| drop gpt2 | 45 | **+0.084 (0.280)** | −0.011 (0.906) |
| drop llama2 | 78 | +0.181 (0.0013) | −0.039 (0.494) |
| drop llama3 | 66 | +0.162 (0.0015) | −0.010 (0.846) |
| drop olmo2 | 66 | +0.185 (0.0014) | −0.092 (0.128) |
| drop openllama | 78 | +0.185 (0.0006) | −0.040 (0.448) |
| drop opt | 78 | +0.179 (0.0006) | −0.064 (0.223) |
| drop qwen3 | 55 | +0.182 (0.0089) | −0.105 (0.127) |
| + `log(width_ratio)` control | 91 | +0.171 (0.0015) | −0.056 (0.278) (width β=+0.002, p=0.96) |
| `same_lineage` instead of `same_family`¹ | 91 | +0.178 (5.4e-5) | −0.048 (0.319) |
| `abs_depth_diff` instead of `log ratio` | 91 | +0.182 (0.0005) | −0.0042 (**0.048**) |

¹ `same_lineage` merges llama2/llama3/openllama into one "llama-architecture" group.

Two things to state honestly:
* **The family effect loses significance when GPT-2 is removed** (β +0.084, p = 0.28).
  GPT-2 supplies **6 of the 11** same-family pairs (composition: gpt2 6, qwen3 3,
  llama3 1, olmo2 1). So "same family helps" is significant on n=91 but is
  disproportionately a GPT-2-scaling-ladder fact. Leave-one-model-out keeps
  `same_family` β in [+0.133, +0.211] (always positive) and `log(depth_ratio)` β in
  [−0.123, −0.007] (always negative), so no single *model* drives it, but a single
  *family* does.
* **Under `abs_depth_diff` instead of `log(depth_ratio)`, depth becomes marginally
  significant** (β = −0.0042 per layer, p = 0.048) while family stays 4× larger in
  standardised terms. This is the strongest form in which any part of H2 survives, and
  it is marginal and specification-dependent. It does **not** rescue H2's ordering
  claim.

**Bottom line for H2**: retract it. The defensible statement is *"neither family nor
depth explains much — total R² = 0.11. Family is the larger of the two effects, and
depth mismatch is not significant. R3's contrary conclusion rested on one pair whose low
score is an OLMo-2 (post-norm) idiosyncrasy."* R3's proposed follow-up — promoting
"layer alignment is set by depth, not family" to a publishable measurement claim — must
**not** be pursued.

---

## 4. H3 — the awkward middle band: **HOLDS in substance, narrow interval refuted**

All four reference levels on one scale (z-CKA, midband definition throughout):

| level | n | value |
|---|---|---|
| **floor** — real model vs **random-init** model | 4 | mean **0.091** (0.051 / 0.051 / 0.126 / 0.136) |
| **observed, all pairs** | 91 | min 0.087 · p25 0.388 · **median 0.503** · p75 0.585 · max 0.840 (mean 0.491, sd 0.164) |
| **observed, cross-family only** | 80 | min 0.087 · median 0.478 · max 0.827 (mean 0.471) |
| **null — B's layer order shuffled** (200 perms/pair) | 18 200 | mean **0.453**, 95% range [0.168, 0.754] |
| **ceiling** — same model, adjacent layers | 14 | min **0.923** · **median 0.977** · max 0.995 |

Separation: `observed max (0.840) < ceiling min (0.923)` — **no pair, cross-family or
not, reaches the same-model adjacent-layer ceiling** (0 of 91). At the bottom, only
**1 of 91** falls below the random-init floor mean. So the qualitative claim — *real but
far from substitutable* — is confirmed at n=91.

Two corrections to R3's phrasing:
1. **The interval 0.35–0.61 is too narrow**: only **53/91 (58%)** of pairs, and
   **49/80 (61%)** of cross-family pairs, fall inside it. Deciles are
   0.087 / 0.278 / 0.347 / 0.434 / 0.458 / 0.503 / 0.524 / 0.554 / 0.625 / 0.706 / 0.840.
   The correct statement is **"cross-family midband z-CKA spans ≈0.09–0.83, median
   ≈0.48"**.
2. **The floor that matters is not the block mean but the mid-depth minimum.** Median
   diagonal minimum is **0.205** and 2/91 pairs dip **below the random-init floor**.
   Anyone stitching mid-network faces 0.20, not 0.50.

### Caveat that must be reported (the layer-order-shuffle null is weak)
Permuting B's layer order gives a null mean of **0.453** against an observed **0.491** —
a gap of only **+0.038**. Per-pair permutation p-values: median 0.015, and 58/91 pairs
reach p < 0.05; 77/91 have observed above their own null mean. So the midband block *is*
mildly special, but **most of the midband z-CKA magnitude is not layer-correspondence
information** — it survives destroying the layer ordering. The honest reading: the
correct baseline for "is layer *i* the right partner for layer *j*" is this shuffle null
(≈0.45), **not** the random-init floor (≈0.09). Measured against the shuffle null, the
alignment signal available to a stitch layer is far thinner than the raw 0.50 suggests.
This *strengthens* the pessimistic conclusion R3 reached about stitching.

---

## 5. Consistency with R3's 5 pairs

Same script family, same protocol → the numbers reproduce:

| pair | R3 | R4 | abs diff |
|---|---|---|---|
| OLMo-2-1B ↔ Llama-3.2-1B | 0.467 | **0.4672** | 0.0002 |
| OLMo-2-1B ↔ Qwen3-1.7B | 0.517 | **0.5166** | 0.0004 |
| Llama-3.2-1B ↔ Qwen3-1.7B | 0.606 | **0.6062** | 0.0002 |
| OLMo-2-7B ↔ Llama3-8B | 0.383 | **0.3818** | 0.0012 |
| OLMo-2-7B ↔ OLMo-2-1B | 0.346 | **0.3382** | 0.0078 |

Four pairs agree to ≤1.2e-3. This also **confirms the reverse-engineered midband
definition** (block mean over the 25–75% relative-depth box) is exactly what R3 used.
The one visible drift, 0.0078 on `OLMo-2-7B↔OLMo-2-1B`, is explained: R3 computed its
7B/8B pairs on **N=3000** words, R4 uses **N=4000** for all pairs. (`OLMo-2-7B↔Llama3-8B`
came from the same N=3000 file and moved 0.0012, so this is sampling noise of the
expected size, not a protocol change.)

**So R3's measurements were correct; R3's *inference* from 5 of them was not.** All five
R3 pairs remain in the R4 table unchanged; what changed is that 86 more pairs show
OLMo-2-7B↔1B to be an outlier rather than a representative same-family pair.

---

## 6. What Paper D should take from this

1. **Keep and lead with H1.** "The mid-network band that carries transferable abstract
   semantics is systematically the least alignable band" now rests on 91 pairs with a
   distance-controlled test and a non-parametric confirmation. Median mid-depth z-CKA
   **0.205** against a same-model bridging norm of **0.977**.
2. **Delete the depth-vs-family claim.** Do not write "prefer same depth over same
   family." If a pairing heuristic is needed, the data say **same family** (β +0.171),
   with the caveat that it is GPT-2-driven and R² is only 0.11 — i.e. *neither* axis
   predicts alignment well.
3. **Re-baseline all "CKA is decent (≈0.5)" reasoning against the layer-shuffle null
   (≈0.45), not the random-init floor (≈0.09).** The usable layer-correspondence signal
   is ≈0.04, not ≈0.4. This is the single most damaging number in this note for a
   stitching architecture.
4. **OLMo-2 is the worst possible donor/host in this pool** (mean midband 0.329 for the
   7B, lowest of 14), plausibly because it is post-norm. If Paper D keeps OLMo-2 as its
   base, the alignment numbers will be near the bottom of the achievable range; the
   *architectural* choice of base model is worth an explicit sentence.

---

## 7. Reproduction

```bash
# per model (one GPU each) -- 14 real + 2 random-init
python proposal/shared/representation/code/repr_alignment_multimodel.py --stage extract --model <KEY> \
       [--random_init] --device cuda:0 --n_texts 300 --max_words 4500
# global word-key intersection -> N=4000
python proposal/shared/representation/code/repr_alignment_multimodel.py --stage align --target_words 4000
# 95 pair CKA jobs, LPT-balanced over 16 GPUs (see _r4_pair_slots.txt / _r4_cka_node.sh)
python proposal/shared/representation/code/repr_alignment_multimodel.py --stage cka --pairs a:b ... --device cuda:0
# identity gate + adjacent-layer ceiling, one per model
python proposal/shared/representation/code/repr_alignment_multimodel.py --stage selfcka --model <KEY> --device cuda:0
# all statistics -> repr_alignment_results.json
python proposal/shared/representation/code/repr_alignment_multimodel.py --stage stats \
       --n_perm 200 --n_qap 5000 --n_boot 1000
```

`scipy` is unavailable on the H20 nodes, so Student-t tails/quantiles (continued-fraction
incomplete beta), OLS with SEs/CIs, and the exact two-sided binomial test are implemented
in pure numpy inside the script; all were validated against published t critical values
(df=5,10,17,20,100) and against a closed-form correlation t-test before use.

---

## MAIN 独立复核（2026-08-06 18:xx，不用 agent 的报告数字，直接从 results.json 重算）

我自己写 OLS/binomial 复算，与 agent 报告**逐项吻合**：

| 检验 | MAIN 复算 | agent 报告 | 一致 |
|---|---|---|---|
| H1 `quad_c>0` | 72/91, binom p=9.84e-09, 显著 49 | 72/91, p=2e-8, 49 | ✓ |
| H1 两端>最低点 | **91/91** | 87/91 | ⚠ 我算 91（用 `diag_end_mean_zcka > diag_min_zcka`）；agent 可能用了更严格的"两端各自"而非均值 |
| H1 排除距离后 | 72/91（与原始**完全相同**） | 72/91 | ✓ |
| H1 对角线最大距离 | 0.0208 | 0.021 | ✓ |
| H2 `same_family` | β=+0.1708 t=+3.34 p=0.0009 标准化 +0.341 | β=+0.171 p=0.0012 +0.341 | ✓ |
| H2 `log(depth_ratio)` | β=−0.0557 t=−1.10 **p=0.2701** 标准化 −0.113 | β=−0.056 **p=0.273** −0.113 | ✓ |
| H2 总 R² | 0.1143 | 0.11 | ✓ |
| H3 实测 | min 0.087 / median 0.503 / max 0.840 / mean 0.491 | 同 | ✓ |
| H3 shuffle null | mean 0.453 → **实测−null = +0.038** | 0.453 vs 0.491 | ✓ |
| H3 落在 R3 区间 | 53/91 | 53/91 | ✓ |

**H2 被推翻的机制我也复核了**：同家族 11 个 pair 的 midband 由低到高是
`olmo2_1b:olmo2_7b 0.338` → `gpt2:gpt2_xl 0.505` → `gpt2_medium:gpt2_xl 0.508`
→ `gpt2_large:gpt2_xl 0.525` → … → `gpt2:gpt2_medium 0.840`。
**R3 的 H2 完全建立在 `olmo2_1b:olmo2_7b`=0.338 这一个点上，而它恰是全部同家族 pair 里最低的那个。**
反例 `gpt2:gpt2_large` depth_ratio=3.0 却 midband=0.815。

**最重要的新发现（agent 主动加的，我确认）**：H3 的 shuffle-层序 null 是 **0.453**，
实测 **0.491** —— **差值只有 +0.038**。也就是说 midband CKA 的绝对数值里
**绝大部分不是"层与层对应"的信息**，而是两个模型激活的общ体几何相似性。
这比 R3 原本的悲观结论**更强**：连"中间带 0.35-0.61 说明有部分可对齐性"这个读法都站不住。

**结论对 Paper D 的意义**：
- H1（U 型）**稳**，且排除了 trivial 距离解释 → 唯一可发表的观察
- H2（深度差 > 家族差）**证伪，符号还反了** → R3 建议的"publish 深度不是家族"必须撤掉
- H3 实质成立但区间说法要改，且 null 校准使其变成**更强的否证**
→ 三条里两条是负面/被推翻。Paper D 作为独立论文的最后一点希望（H2 那个反直觉点）没了。
