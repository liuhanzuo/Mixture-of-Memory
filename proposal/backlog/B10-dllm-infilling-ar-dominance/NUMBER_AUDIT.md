# B10 NUMBER AUDIT — every headline value re-derived from disk

**Date:** 2026-08-11. **GPU used: none.** All values below were recomputed from
per-task records already on disk; nothing was copied out of `SLATE.md`.

**Audit rule applied.** For each number: (a) the exact file + JSON key it came
from, (b) the value I got, (c) whether it matches SLATE, (d) `NOT ON DISK` if it
cannot be found. Where a value exists in both a summary file and the raw
per-task records, the **raw records are authoritative** and the summary is
treated as a claim to be checked.

---

## 0. Executive summary of the audit

| | verdict |
|---|---|
| Four headline pass@1 values | **ALL FOUND, ALL EXACT** (to 10 d.p.) — reproduced twice, independently |
| The true n | **1033, confirmed** — and it is genuinely SingleLine, not RandomSpan |
| The true task surface | **HumanEval-SingleLineInfilling**, `--which plus` |
| SLATE's own flagged correction (attended_context 0.88×) | **CONFIRMED** |
| SLATE's claim of a "matched lineage" to Dream-Coder-**Base** | **WRONG MODEL** — the run used Dream-Coder-v0-**Instruct**-7B |
| SLATE's claim (b) "kwargs are swallowed by `**kwargs`" | **CONFIRMED as a fact, but its IMPLICATION IS REFUTED** — see §6 |
| The headline ordering "AR beats the best diffusion arm" | **NOT STATISTICALLY SUPPORTED** — see §4 |
| Comparability to DreamOn's published numbers | **FAILS** — our harness reproduces neither arm; see §7 |

Two numbers in SLATE are *arithmetically* perfect and *scientifically*
unusable as stated. That combination is the main finding of this audit.

---

## 1. Provenance chain (what I recomputed from, and why it is independent)

The headline arms live in **one** file that SLATE never names:

```
zwfy6: /apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft_104/results/infilling/single_line_summary.json
       sha256 576cf5d3f046e57e9af6273d584d2ad564004e8a5ab5ee43d8d078d848bbebb8   6168 B
```

But I did **not** score from that summary. Each of the six arms also has raw
per-task output, and I recomputed from those:

```
zwfy6: .../dllm_draft_104/outputs/infilling_single_line/<ARM>/score.json     (has per_task[], 1033 rows)
zwfy6: .../dllm_draft_104/outputs/infilling_single_line/<ARM>/metrics.jsonl  (1033 lines, per-task cost)
```

So there are **three** independent levels: `single_line_summary.json` (a
roll-up), `score.json:pass_at_1` (the scorer's own aggregate), and
`score.json:per_task[]` + `metrics.jsonl` (raw). I recomputed pass@1 as
`sum(per_task[].pass)/len(per_task)` and both cost units as
`mean(metrics.jsonl[].cost[...])`. **All three levels agree exactly for all six
arms.** No rounding drift, no arm mismatch, no partial-shard merge.

Per-arm hashes (first 16 hex of sha256) recorded in `SOURCES.md`.

---

## 2. The four headline pass@1 values — RE-DERIVED

`which_tests = "plus"` for every arm. `n = 1033` for every arm. Every arm has
`generation_errors = 0` and `grader_self_test.trustworthy = true`
(`canonical_pass 10/10`, `stub_fail 10/10`).

| arm | SLATE says | **I recomputed** | n_pass / n | match? |
|---|---|---|---|---|
| `qwen_fim` (AR native FIM) | .7638 | **0.7637947725072604** | 789 / 1033 | ✅ EXACT |
| `dreamon_oracle` | .7590 | **0.7589545014520813** | 784 / 1033 | ✅ EXACT |
| `dream_fim` | .7115 | **0.7115198451113263** | 735 / 1033 | ✅ EXACT |
| `dreamon_fim` (own-prediction) | .7018 | **0.7018393030009681** | 725 / 1033 | ✅ EXACT |
| `qwen_prefix` (AR L→R control) | .5324 | **0.5324298160696999** | 550 / 1033 | ✅ EXACT |
| `dream_prefix` (diffusion unidir. control) | .4124 | **0.4123910939012584** | 426 / 1033 | ✅ EXACT |

Recompute method: `n_pass_recount` from `per_task[].pass` equals
`score.json:n_pass` equals the summary's `n_pass`, for all six arms. Task ids
are unique (1033 distinct) and are a strict subset of the benchmark's own 1033
ids, so no duplication and no cross-arm id drift.

**Arithmetic verdict: SLATE's four pass@1 values are exactly right.** This is
the one part of SLATE that survives untouched.

---

## 3. True n and true task surface — SLATE IS CORRECT, my brief's suspicion was wrong

The task brief asked me to check whether `n=1033` was actually the **RandomSpan**
surface mislabelled as SingleLine. **It is not.** SingleLine really is 1033:

```
zwfy6 .../dllm_draft/data/humaneval_infilling/HumanEval-SingleLineInfilling.jsonl
   1033 lines   sha256 6fffc71ec2f1674372fcc177511f92312f1a27a9eacd8e43255c9f5ee9eca8c8
zwfy6 .../HumanEval-RandomSpanInfilling.jsonl   1640 lines
zwfy6 .../HumanEval-MultiLineInfilling.jsonl    5815 lines
```

The launcher pins `SPLIT_FILE=...HumanEval-SingleLineInfilling.jsonl` and
`SPLIT_NAME=single_line`, and every `per_task[].task_id` has the form
`SingleLineInfilling/HumanEval/<k>/L<i>`.

**The `1033` grep hits my brief listed are all false positives** — every one is a
digit substring inside a float, not a task count. Verified by printing the
surrounding context:

| file the brief flagged | what the "1033" actually is |
|---|---|
| `runs/kspan_decon_diffusion/score.json` | `"pass@1": 0.46551724137931033` |
| `runs/kspan_decon_ar_fim/score.json` | `"attended": 10336` |
| `runs/spanlen/score_RandomSpan_dream_fim.json` | `"pass_at_1_plus": 0.21033210332103322` |
| `runs/spanlen/spanlen_summary_full1640.json` | same float as above |
| `runs/kspan_diffusion_nonoracle/score.json` | (n_solutions is 415, spec `kspan_spec_v1.jsonl`) |

So the grep trail did **not** point at a mislabelled surface; it pointed at
nothing. Independently, `build_kspan_infilling.py` does read
`HumanEval-SingleLineInfilling.jsonl` as its `POOL_REL` and admits 910/1033 rows,
so the k-span line is *derived from* the same pool — but its scored arms are
n=415/408/236/165, never 1033.

**Why MAIN could not find the arms.** MAIN searched `dllm_draft/runs/spanlen/`,
which holds a *different, later* experiment (span-length stratification on
**RandomSpan** n=1640 and **MultiLine_sub** n=420). The headline arms are not in
`runs/` at all — they are in `outputs/infilling_single_line/` on the **zwfy6
`dllm_draft_104` checkout only**. SLATE's paths are stale; corrected paths are in
`SOURCES.md`. **Nothing is missing and nothing needs re-running.**

---

## 4. ★ THE HEADLINE ORDERING IS NOT STATISTICALLY SUPPORTED

The values are exact; the *claim built on them* is not. Every arm scored the same
1033 items, so the comparison is **paired** and must use a paired test. SLATE
reports a bare ordering. Recomputing with McNemar's exact test (two-sided) plus
a paired bootstrap (20 000 resamples, seed 0) on the 1033 common ids:

| contrast | Δpass@1 | discordant b / c | McNemar exact p | paired 95% CI |
|---|---|---|---|---|
| **`qwen_fim` vs `dreamon_oracle`** | **+0.0048** | 38 / 33 | **p = 0.635** | **[−0.0116, +0.0213]** |
| `qwen_fim` vs `dream_fim` | +0.0523 | 86 / 32 | 6.9e-07 | [+0.0319, +0.0726] |
| `qwen_fim` vs `dreamon_fim` | +0.0620 | 90 / 26 | 1.9e-09 | [+0.0416, +0.0823] |
| `dreamon_oracle` vs `dreamon_fim` | +0.0572 | 64 / 5 | 4.1e-14 | [+0.0416, +0.0726] |
| `qwen_fim` vs `qwen_prefix` | +0.2314 | 254 / 15 | 3.2e-57 | [+0.2033, +0.2594] |
| `dream_fim` vs `dream_prefix` | +0.2991 | 332 / 23 | 2.5e-71 | [+0.2682, +0.3301] |

**The single most important line in this audit:** AR's win over the strongest
diffusion arm is **5 tasks out of 1033**, `p = 0.635`, and the confidence
interval **straddles zero**. SLATE's phrase "**ABOVE** DreamOn's oracle-assisted
.7590" describes a difference indistinguishable from noise.

The two large, robust effects are (i) AR beats the two *non-oracle-equivalent*
diffusion arms by 5–6 pp, and (ii) suffix visibility is worth ~23–30 pp to both
families. Those survive. "A plain AR model beats masked diffusion on diffusion's
home turf" — as a *ranking* claim against the best diffusion arm — does not.

Note this cuts against SLATE's own remedy. SLATE says to demote the oracle arm to
an upper bound and headline the non-oracle arm. Doing that makes the AR win
*significant* (p=1.9e-09) but also makes it a comparison against a **weakened**
diffusion configuration, which is exactly the kind of favourable framing the
direction exists to criticise. Both versions cannot be used at once.

---

## 5. Cost figures — RE-DERIVED, and SLATE's self-flagged correction CONFIRMED

Recomputed as the mean over all 1033 tasks of `metrics.jsonl[].cost[...]`.
Matches `score.json:cost_tokens` exactly for all six arms.

| arm | tokens_fed / task | attended_context_sum / task | forward_passes / task |
|---|---|---|---|
| `qwen_fim` | **238.90416263310746** | **2313.862536302033** | 9.977 |
| `qwen_prefix` | 248.972894482091 | 13023.093901258471 | 58.900 |
| `dream_fim` | **2035.027105517909** | **2035.027105517909** | 8.579 |
| `dream_prefix` | 1739.13068731849 | 1739.13068731849 | 8.579 |
| `dreamon_fim` | **4922.613746369797** | **4922.613746369797** | 16.470 |
| `dreamon_oracle` | **5826.759922555663** | **5826.759922555663** | 20.711 |

All four cost values SLATE quotes (238.90 / 2035.03 / 4922.61 / 5826.76) are
**EXACT**. Ratios relative to `qwen_fim`:

| arm | tokens_fed | attended_context_sum |
|---|---|---|
| `dream_fim` | 8.52× | **0.88× — CHEAPER THAN AR** |
| `dream_prefix` | 7.28× | **0.75× — CHEAPER THAN AR** |
| `dreamon_fim` | 20.60× | 2.13× |
| `dreamon_oracle` | 24.39× | 2.52× |

**SLATE's self-flagged correction is CONFIRMED.** "20.6–24.4× fewer tokens than
either DreamOn arm" is true *only for the two DreamOn arms and only under
`tokens_fed`*. Under `attended_context_sum`, **Dream-FIM is cheaper than AR**, so
"AR dominates on both cost units" is FALSE. Carry the correction; do not publish
the two-unit table without it.

One more caveat SLATE does not state: the two units are not two views of the same
quantity. For every diffusion arm `tokens_fed == attended_context_sum` exactly
(each denoising step re-feeds the whole canvas), whereas for AR they differ by
~10× because of KV caching. So the choice of unit *is* the choice of who wins,
and neither unit is neutral. This must be disclosed, not arbitrated.

---

## 6. ★ Claim (b) — the fact is TRUE, the conclusion drawn from it is REFUTED

This is the claim SLATE said it could not verify because "DreamOn weights are
absent from both disks." **The weights are present**, at
`zwfy6 .../dllm_draft/models/DreamOn-v0-7B/` (4 safetensors shards, 15 GB, plus
`generation_utils.py`, sha256 `9ef97ad61d77cf...`). `models` is a **symlink** from
`dllm_draft_104/models` → `dllm_draft/models`, which is likely why a naive `find`
missed it. **No re-download is needed.** I verified everything below statically
and on CPU — no GPU, no model load.

**Verified TRUE (the mechanical fact):**
1. `mask_expansion` and `delete_eos_token` appear **0 times** anywhere in
   DreamOn's `generation_utils.py`.
2. `DreamGenerationConfig.__init__` `kwargs.pop`s a fixed whitelist; the two
   names are not in it. `cfg.update(mask_expansion=True, delete_eos_token=True)`
   returns them as **unused**: `{'mask_expansion': True, 'delete_eos_token': True}`,
   and `hasattr(cfg,'mask_expansion')` is `False` before *and* after.
3. Our own call site does pass both:
   `dllm_draft/scripts/generate_evalplus_dreamon.py:133-134` (verified on wzc1,
   sha256 `579d1e0a9ec7...`). So they were indeed silently dropped.

**But the inference "therefore length-elasticity was never exercised" is FALSE.**
Expansion/deletion in DreamOn is **not flag-gated at all** — it is driven by
*token identity*, so it is **live by default**:
- sampler decides expansion by `x[0] == expand_token_id` and deletion by
  `== delete_token_id`; there is no `if generation_config.mask_expansion` anywhere;
- `config.json` ships `expand_token_id: 151667` (`<|expand|>` in
  `added_tokens.json`) and `mask_token_id: 151666`; via `from_model_config` these
  resolve to `151667` / `151666` / `delete=151643`;
- `expand_budget` is `None`, which the sampler turns into `max_gen_len * 2`
  (`generation_utils.py:383-384`) — i.e. a *generous* budget, not zero.

**Empirical confirmation from our own recorded telemetry** (`metrics.jsonl`,
`info.middle_tokens` vs `info.initial_masks`):

| arm | initial masks | final middle length | length CHANGED |
|---|---|---|---|
| `dreamon_fim` | 4 (constant) | min 0, median 6, max 36 | **871/1033 = 84.3 %** (803 longer, 68 shorter) |
| `dreamon_oracle` | 2–9 (35 distinct) | min 0, median 7, max 39 | 161/1033 = 15.6 % |

A mechanism that changed the output length on **84.3 %** of tasks was
unambiguously running. The two kwargs are **decorative no-ops that happened to
request behaviour that was already on by default.**

Further: the DreamOn README's own "Parameters" section **never advertises** these
two kwargs. It documents `max_new_tokens`, `alg`, `alg_temp`, `temperature`,
`top_p`, `top_k`, `number_transfer_tokens`, `output_history`,
`return_dict_in_generate` — and nothing else. So SLATE's framing ("DreamOn's
**advertised** kwargs … its headline length-elasticity is **not exercised**") is
wrong on both halves: not advertised, and exercised anyway.

**Consequence: claim (b) is DEAD as a finding about DreamOn.** What remains is a
much smaller, purely internal point: *our own* harness passed two kwargs that do
nothing, and any wording of ours claiming "with `mask_expansion=True`" must be
retracted (which `DLLM_RESULTS_20260807.md` already does). That is a lab-notebook
correction, not a defect in a public model, and **must not be published as one.**
The "functional defect in a public model's headline capability" — which SLATE
called half the paper — does not exist.

The `**kwargs` swallowing is inherited HF `GenerationConfig` behaviour that
silently accepts unknown keys. Writing that up as a DreamOn defect would be an
error of the same species this direction exists to criticise.

---

## 7. ★ Our numbers do not reproduce DreamOn's published numbers

DreamOn (ICLR 2026 Poster, camera-ready Table 1, arXiv:2602.01326v1, PDF read
directly) reports on HumanEval-Infilling **single-line pass@1**:

| | published | ours (this harness) | gap |
|---|---|---|---|
| DreamCoder-7B + DreamOn | **92.1** | 70.18 (`dreamon_fim`) | **−21.9 pp** |
| Qwen2.5-Coder-7B | **92.6** | 76.38 (`qwen_fim`) | **−16.2 pp** |
| DreamCoder-7B (no DreamOn) | 55.5 | 71.15 (`dream_fim`, oracle len) | +15.7 pp |

**Both** of our arms land far below the published values, so this is a harness
difference, not a model finding. The dominant identified cause is the **grading
axis**: we grade `--which plus` (HumanEval**+** extended tests), while the
benchmark's official protocol is the base HumanEval tests. Our own measured gold
ceiling on this exact split proves the axis is the problem:

```
wzc1 dllm_draft/runs/spanlen/gold_ceiling_SingleLine.json  (sha256 007baa0924f9e750...)
  n_rows 1033   gold_ceiling_base = 0.9894   gold_ceiling_plus = 0.8025
```

**Splicing the benchmark's own gold middle back in scores only 0.8025 on the axis
we graded.** So 19.75 % of items are unpassable *by construction* for every arm,
and a raw 0.7638 is not comparable to a published 92.6 at all. Restricting to the
829 gold-feasible items:

| arm | raw | gold-feasible only | raw / ceiling |
|---|---|---|---|
| `qwen_fim` | 0.7638 | **0.9324** | 0.9517 |
| `dreamon_oracle` | 0.7590 | **0.9337** | 0.9457 |
| `dream_fim` | 0.7115 | 0.8733 | 0.8866 |
| `dreamon_fim` | 0.7018 | 0.8625 | 0.8745 |
| `qwen_prefix` | 0.5324 | 0.6454 | 0.6634 |
| `dream_prefix` | 0.4124 | 0.5030 | 0.5139 |

Two consequences, both fatal to the headline as SLATE states it:

1. **On the feasible subset the sign FLIPS.** `dreamon_oracle` 0.9337 >
   `qwen_fim` 0.9324 (McNemar b=31, c=32, **p = 1.000**, Δ = −0.0012). The
   ordering that the whole direction rests on is an artefact of including items
   that nobody could pass.
2. Once ceiling-normalised, `qwen_fim` reaches 93.2 and `dreamon_oracle` 93.4 —
   both **close to DreamOn's published 92.6 / 92.1**. The apparent 16–22 pp
   "failure to reproduce" is mostly the plus-axis ceiling, which is reassuring
   about the harness and damning for the headline.

**This is the single largest correction in the audit.** Any future write-up must
either grade on the base axis or report ceiling-normalised numbers; the raw plus
numbers cannot be compared to anything in the literature.

---

## 8. Model lineage — SLATE NAMES THE WRONG MODEL

SLATE: "Qwen2.5-Coder-7B is a true matched lineage for **Dream-Coder-v0-Base-7B**
(identical hidden 3584 / 28 layers / 28 heads / 4 KV heads / intermediate 18944 /
vocab 152064 / rope_theta 1e6, differing only in `mask_token_id`)."

The **architectural facts are correct** — I verified all six fields from
`config.json` on disk for all four models, and they are identical across
Qwen2.5-Coder-7B, Dream-Coder-v0-Base-7B, Dream-Coder-v0-Instruct-7B and
DreamOn-v0-7B (all `hidden_size 3584`, `num_hidden_layers 28`,
`num_attention_heads 28`, `num_key_value_heads 4`, `intermediate_size 18944`,
`vocab_size 152064`, `rope_theta 1e6`; the Dream family adds
`mask_token_id 151666`, Qwen has `null`).

**But Dream-Coder-v0-Base-7B was never run.** The launcher maps
`dream_fim|dream_prefix → models/Dream-Coder-v0-**Instruct**-7B`. So the arm
compared against a *base* AR model is an **instruction-tuned** diffusion model.
The lineage sentence is true of a model that is not in the experiment.

This matters in both directions and is not merely pedantic: `qwen_fim` is
**Qwen2.5-Coder-7B base** (no instruct tuning) while `dream_fim` is
**Instruct**. Post-training differs across the arms, so "differing only in
`mask_token_id`" is false *as a description of the comparison actually run*. The
Base checkpoint **is on disk**, so this is fixable by re-running one arm — but it
is currently a confound, not a matched control.

---

## 9. The mechanism claim — re-derived, and it is the strongest surviving result

SLATE: "suffix-visibility gain is +.2314 for AR vs +.2991 for diffusion —
comparable — so bidirectional context is an affordance of the task FRAMING."

Recomputed from `per_task[].pass`:

```
gain_AR   = qwen_fim  − qwen_prefix  = 0.7638 − 0.5324 = +0.2314   ✅ EXACT
gain_DIFF = dream_fim − dream_prefix = 0.7115 − 0.4124 = +0.2991   ✅ EXACT
difference of gains (DIFF − AR)      = +0.0678
paired bootstrap 95 % CI             = [+0.0407, +0.0949]
```

Both numbers EXACT. One correction to the wording: the CI on the *difference of
gains* **excludes zero**, so the two gains are **not statistically
indistinguishable** — diffusion's suffix gain is reliably ~6.8 pp larger. "Both
families gain substantially from suffix visibility, diffusion somewhat more" is
supportable; "**comparable**" (implying no difference) overstates it.

Caveat that must travel with this comparison: `dream_prefix` still receives the
**oracle span length** while `qwen_prefix` does not, so the diffusion side of the
gain is measured under a length handout. This is a genuine asymmetry in the arm
design, disclosed in the harness docstring itself.

---

## 10. Line-by-line verdict on SLATE's `#3` entry

| SLATE assertion | verdict |
|---|---|
| n=1033, official evalplus, zero generation errors | ✅ TRUE (`generation_errors=0` all arms; grader self-test passes) |
| the four pass@1 values | ✅ EXACT to 10 d.p. |
| the two unidirectional control values | ✅ EXACT |
| tokens_fed/task 238.90 / 2035.03 / 4922.61 / 5826.76 | ✅ EXACT |
| "20.6–24.4× fewer tokens" | ✅ TRUE for the two DreamOn arms under `tokens_fed` only |
| "AR dominates on both cost units" | ❌ FALSE (Dream-FIM is 0.88×) — SLATE flags this itself |
| suffix-gain +.2314 / +.2991 | ✅ EXACT; "comparable" overstated (CI on difference excludes 0) |
| "ABOVE DreamOn's oracle-assisted .7590" | ❌ **NOT SIGNIFICANT** (p=0.635, CI straddles 0); **SIGN FLIPS** on gold-feasible subset |
| "matched lineage for Dream-Coder-v0-Base-7B" | ❌ **WRONG MODEL** — Instruct was run |
| "kwargs are absorbed by `**kwargs`, not parameters" | ✅ TRUE mechanically |
| "…so length-elasticity is not exercised" | ❌ **REFUTED** — active on 84.3 % of tasks |
| "advertised … kwargs" | ❌ FALSE — README never documents them |
| "every DreamOn number in circulation was produced with the toggle inert" | ❌ misleading — the *capability* was on; only the *kwargs* were inert |
| "DreamOn ships no matched-lineage AR native-FIM control on its own surface" | ❌ **FALSE** — Table 1 has Qwen2.5-Coder-7B at 92.6 |
| "DreamOn scores 0.122 pass@1 on from-scratch HumanEval+" | ✅ consistent with repo records (a *from-scratch* number, not infilling) |
| "24 GPU-h" cost estimate | ⚠️ wrong shape — see §11 |
| "six fully scored arms … marginal cost is prose" | ❌ FALSE — see §11 |

---

## 11. Honest cost to first *publishable* result

SLATE says the arms are scored so "the marginal cost is prose." That is only true
if the existing numbers answer the question, and §4/§7/§8 show they do not. The
arms are **not** unusable — they are a complete, well-instrumented pilot with
three specific defects. Fixing them is cheap but is **not zero**:

| # | required work | why | GPU |
|---|---|---|---|
| 1 | Re-score all 6 existing arms on the **base** axis | plus-axis ceiling 0.8025 makes raw numbers incomparable and flips the sign | **0 GPU** — `score_infilling.py --which base`, solutions already on disk (CPU only) |
| 2 | Re-run `dream_fim`/`dream_prefix` with Dream-Coder-**Base**-7B | current arm is Instruct; kills the lineage confound | ~2–4 GPU-h (2 arms × 1033 items, 8-way shardable) |
| 3 | Give `qwen_prefix` the same length handout as `dream_prefix`, or remove it from both | suffix-gain asymmetry | ~1–2 GPU-h |
| 4 | Paired tests + ceiling normalisation as the reporting standard | §4, §7 | 0 GPU |

**Step 1 alone is free and decides the direction's fate**, because it determines
whether the AR-vs-oracle ordering survives at all. It should be run before any
GPU is spent. Realistic total: **~4–8 GPU-h**, not 24 — but with a materially
higher chance that the headline dissolves than SLATE implies.

---

## 12. What I could not verify

| item | status |
|---|---|
| DreamOn's own eval scripts / exact base-axis protocol | **NOT ON DISK.** The published 92.6/92.1 come from the paper PDF, not from a re-run. Step 1 above closes this. |
| Whether DreamOn's `Lmax=128` expansion cap (paper §4.1) matched our `max_new_tokens=64` | **not established.** Ours caps the canvas at 64, the paper caps expansion at 128 — a possible additional source of the reproduction gap, untested. |
| DiffuCoder-7B / LLaDA-8B / Seed-Coder-8B / Deepseek-Coder-6.7B arms | not on disk; only in the paper. |
| RandomSpan / MultiLine DreamOn arms on wzc1 | `runs/spanlen/RandomSpan_dreamon_fim/` exists **only** on zwfy6 `dllm_draft_104`; wzc1 has the other 8 arm dirs. Not needed for B10. |

---

## 13. Reproduction commands

All read-only, CPU-only. `.73` (`configs/password_h20_853573.txt`, omit `-p`):

```bash
# pass@1 + cost, recomputed from per_task[] and metrics.jsonl for all 6 arms
/opt/conda/envs/torch-base/bin/python /tmp/audit_infill.py     # see SOURCES.md §5
/opt/conda/envs/torch-base/bin/python /tmp/cost_infill.py
# paired McNemar + bootstrap
/opt/conda/envs/torch-base/bin/python /tmp/mcnemar_infill.py
# gold-ceiling normalisation (needs gold_ceiling_SingleLine.json copied from wzc1)
/opt/conda/envs/torch-base/bin/python /tmp/ceiling_norm.py
# claim (b): static + CPU config probe, no weights loaded
cd /apdcephfs_zwfy6/.../dllm_draft_104 && ./.venv_dream/bin/python /tmp/dreamon_live_probe.py
```

The five audit scripts are transient (`/tmp`). Their logic is fully specified
above and each is <60 lines; they are re-derivable from this document. Nothing in
this audit depends on a script that is not reconstructible from the stated
file+key pairs.
