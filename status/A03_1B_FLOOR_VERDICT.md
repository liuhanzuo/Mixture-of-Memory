# A03 — 1B knowledge-floor KILL test: VERDICT

**Date**: 2026-08-08 · **Node**: `.82` (8×H20, all 8 GPUs) · **Commit**: `4ef07fa`
**Question tested**: A03's third Kill condition — *"1B pilot 所有知识指标均处于 floor，无法测量"* —
and, equivalently, A03's first key control — *"每个接口必须高于自己的 null floor"*.

## VERDICT: `1B_PILOT_VIABLE` (A03's 1B-pilot kill condition is **NOT** met)

The pruned+healed 1B arm is **significantly above its own construct-appropriate null on 4 of the
5 measurable knowledge interfaces**, so the knowledge axes are measurable at 1B and A03's 1B pilot
does not die. But the result comes with **two mandatory caveats that change A03's design**:

1. **MMLU *letter* must be dropped as an A03 interface at 1B.** The pruned arm scores `0.2512`
   against a best-constant floor of `0.2689` — i.e. **significantly BELOW its own floor**
   (`-1.77pp`, CI `[-2.98, -0.58]pp`, bootstrap `p=3.4e-3`, McNemar `p=3.6e-3`). It is also
   indistinguishable from *its own* modal-constant predictor (`always-C`, `p=0.28`): it emits
   `C` on 64.4% of items. This is exactly A01's documented letter-interface degeneration, and it
   reproduces at 1B. **`MMLU-content` (label-free, choice-text scoring) is the interface that works.**
2. **`contains` on generative QA needs a length-matched null**, because healing changes decoding
   verbosity (~6× longer predictions), and `contains` rewards length. Uncontrolled, the pruned
   arm's PopQA `contains` residual fraction looks like **56.4%**; against an input-blind null
   *matched to that arm's own output length* it is **17.0%**. The signal survives, but is 3.3×
   smaller than the naive number.

**Most discriminative axis for A03's 6-arm design: TriviaQA (EM).** Pruned-arm residual fraction
**97.3%** (`0.0959` vs `0.0026`), and the intact→pruned gap is huge (`0.4069` → `0.0959`), i.e. a
large, floor-free dynamic range for the 6 interventions to move in. **Recommended primary axis =
TriviaQA EM; secondary = PopQA EM; MC axis = MMLU-content only.**

---

## 1. What was run

Both harnesses are **pre-existing and reused verbatim** (no new eval code, no arch drift — both
import `load_base_model` / `load_pruned_model` from `scripts/eval_olmo2_probe2_ppl.py`):

| Axis | Harness | Items | Metric(s) |
|---|---|---:|---|
| MMLU letter + MMLU-content | `scripts/eval_olmo2_mmlu_content.py` | 14,042 | `letter_acc`, `content_norm_acc` |
| PopQA | `scripts/eval_olmo2_closedbook_qa.py` | 14,267 | `em`, `contains` |
| TriviaQA (`rc.nocontext`) | `scripts/eval_olmo2_closedbook_qa.py` | 17,944 | `em`, `contains` |

Driver: `scripts/_run_a03_1b_floor_82.sh` — 8 shards/8 GPUs per job, **8/8 shard files asserted
present AND exact item count asserted before any merge is trusted** (`n==14042` / `14267` / `17944`;
MMLU also `n_valid+n_nan==n`). All 6 jobs passed; driver exited `rc=0`.

Protocol (both harnesses, all arms): **`chat_template=False`**, `add_special_tokens=False`
(`--add_bos 0`), no system prompt, no few-shot, greedy decode for QA, fp32 weights /
bf16-autocast forward. OLMo-2 1B is a BASE LM with no SFT/RL.

### Arms

| Label | Model | Layers |
|---|---|---|
| `intact_1B_16L` | `../models/OLMo-2-0425-1B`, no ckpt | 16 |
| `pruned_healed_keep7f2_200k` | `outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt` | 9 (keep-front 7 + 2 fresh) |
| `pruned_healed_keep7f2_step500` | `outputs/olmo2_probe2_1B_keep7fresh2/step500.pt` | 9 (barely-healed lower bound) |

The third arm is not required by the task but is the load-bearing sanity control: a model that is
genuinely *at* the floor must show up as at/below floor. It does (see §4), which is what licenses
reading the `step200k` arm's residuals as real rather than as an artefact of the null being too weak.

## 2. Nulls — computed here, not copied

Every null is **input-blind** (a fixed rule/string emitted regardless of the question) and
**construct-appropriate** (same metric, same items as the arm it calibrates).

| Interface | Null | Value |
|---|---|---:|
| MMLU letter | best-constant letter = **always-D** (gold dist. A 3222 / B 3462 / C 3582 / **D 3776**) | **0.2689** |
| MMLU-content | longest-option heuristic, **split-tie** (pre-registered) | **0.2845** |
| PopQA EM | best-constant answer string, argmax over 303 candidates → `"association football"` | **0.0229** |
| PopQA contains | best single-word constant → `"american football"` | 0.0488 |
| PopQA contains (len-matched) | verbose input-blind constant, matched to each arm's mean pred length | 0.0259 / **0.0928** / 0.1030 |
| TriviaQA EM | best-constant answer string → `"australia commonwealth realm"` | **0.0026** |
| TriviaQA contains | best single-word constant | 0.0086 |
| TriviaQA contains (len-matched) | verbose input-blind constant, per arm | 0.0062 / **0.0164** / 0.0270 |

Notes that matter:

* **`.25` was never used as the MMLU floor.** The best constant is `always-D = 0.2689`,
  independently recomputed here from the gold distribution of this 14,042-item set. It reproduces
  A01's recorded value exactly — but it is *this run's* number, not a copied one.
* **Longest-option tie convention is load-bearing**: 34.22% of MMLU items have ≥2 maximal-length
  options. All five conventions: `split 0.28445` (used), `first 0.28109`, `last 0.28215`,
  `credit 0.45371`, `wrong 0.19613`. `split` = unbiased expectation under uniform tie-breaking.
* **QA nulls are best-constant, not the harness's `majority_em`.** The harness fixes the majority
  gold string a priori; a floor must be the *maximum* over input-blind constants, so all 303
  candidates (top-300 gold strings + empty + two refusal strings) are scored and the max is taken.
  On PopQA EM this raises the floor from `majority_em` to the same `0.0229` (the majority string
  *is* the argmax there); the empty/refusal constant scores ≈0 because neither benchmark has
  unanswerable items — so unlike A01's SQuAD case, **refusal is not the binding floor here; the
  majority-prior constant is**.
* **The length-matched `contains` null is new and necessary** (see §3.2).

Statistics follow A01's conventions exactly: paired bootstrap over items on the per-item difference
vector (`n_boot=10000`, multinomial representation, two-sided `p` floored at `1/n_boot=1e-4`),
exact-binomial McNemar on discordant items where the null is binary, and **Benjamini–Hochberg
`q=.05` across the whole 24-cell family** (all 24 cells rejected at BH except the one noted below).

## 3. Four-tuple table — `reported / null / calibrated residual / residual fraction`

`frac = residual / reported`. `CI` = 95% paired-bootstrap CI on the residual, in percentage points.

### 3.1 MMLU (n = 14,042)

| Arm | Interface | reported | null | residual | frac | CI (pp) | boot p | McNemar p | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| intact 16L | letter | 0.3807 | 0.2689 | **+0.1118** | 29.4% | [+10.00, +12.39] | 1e-4 | 5.8e-74 | ABOVE |
| intact 16L | **content** | 0.3868 | 0.2845 | **+0.1024** | 26.5% | [+9.32, +11.18] | 1e-4 | — | ABOVE |
| keep7+f2 @200k | letter | 0.2512 | 0.2689 | **−0.0177** | −7.1% | [−2.98, −0.58] | 3.4e-3 | 3.6e-3 | **BELOW** |
| keep7+f2 @200k | **content** | 0.3244 | 0.2845 | **+0.0399** | 12.3% | [+3.10, +4.87] | 1e-4 | — | **ABOVE** |
| keep7+f2 @500 | letter | 0.2295 | 0.2689 | −0.0395 | −17.2% | [−5.12, −2.80] | 1e-4 | 3.7e-11 | BELOW |
| keep7+f2 @500 | content | 0.2632 | 0.2845 | −0.0212 | −8.1% | [−3.00, −1.26] | 1e-4 | — | BELOW |

### 3.2 PopQA (n = 14,267)

| Arm | Interface | reported | null | residual | frac | CI (pp) | boot p | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| intact 16L | **em** | 0.1550 | 0.0229 | **+0.1321** | 85.2% | [+12.62, +13.79] | 1e-4 | ABOVE |
| keep7+f2 @200k | **em** | 0.0394 | 0.0229 | **+0.0165** | 41.8% | [+1.25, +2.06] | 1e-4 | **ABOVE** |
| keep7+f2 @500 | em | 0.0000 | 0.0229 | −0.0229 | n/a | [−2.54, −2.05] | 1e-4 | BELOW |
| intact 16L | contains | 0.1678 | 0.0488 | +0.1190 | 70.9% | [+11.31, +12.49] | 1e-4 | ABOVE |
| keep7+f2 @200k | contains | 0.1119 | 0.0488 | +0.0631 | 56.4% | [+5.81, +6.81] | 1e-4 | ABOVE |
| keep7+f2 @500 | contains | 0.0271 | 0.0488 | −0.0217 | −80.3% | [−2.47, −1.87] | 1e-4 | BELOW |
| intact 16L | contains (len-matched) | 0.1678 | 0.0259 | +0.1419 | 84.6% | [+13.61, +14.78] | 1e-4 | ABOVE |
| keep7+f2 @200k | **contains (len-matched)** | 0.1119 | **0.0928** | **+0.0191** | **17.0%** | [+1.33, +2.47] | 1e-4 | **ABOVE** |
| keep7+f2 @500 | contains (len-matched) | 0.0271 | 0.1030 | −0.0760 | −280.8% | [−8.05, −7.15] | 1e-4 | BELOW |

### 3.3 TriviaQA (n = 17,944)

| Arm | Interface | reported | null | residual | frac | CI (pp) | boot p | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| intact 16L | **em** | 0.4069 | 0.0026 | **+0.4043** | 99.4% | [+39.71, +41.14] | 1e-4 | ABOVE |
| keep7+f2 @200k | **em** | 0.0959 | 0.0026 | **+0.0933** | **97.3%** | [+8.91, +9.77] | 1e-4 | **ABOVE** |
| keep7+f2 @500 | em | 0.0002 | 0.0026 | −0.0023 | −1050% | [−0.31, −0.16] | 1e-4 | BELOW |
| intact 16L | contains | 0.4519 | 0.0086 | +0.4432 | 98.1% | [+43.59, +45.06] | 1e-4 | ABOVE |
| keep7+f2 @200k | contains | 0.2285 | 0.0086 | +0.2199 | 96.2% | [+21.38, +22.61] | 1e-4 | ABOVE |
| keep7+f2 @500 | contains | 0.0079 | 0.0086 | −0.0007 | −9.2% | [−0.23, +0.09] | **0.399** | **AT floor** |
| intact 16L | contains (len-matched) | 0.4519 | 0.0062 | +0.4456 | 98.6% | [+43.82, +45.28] | 1e-4 | ABOVE |
| keep7+f2 @200k | **contains (len-matched)** | 0.2285 | 0.0164 | **+0.2122** | 92.8% | [+20.60, +21.86] | 1e-4 | **ABOVE** |
| keep7+f2 @500 | contains (len-matched) | 0.0079 | 0.0270 | −0.0191 | −240.8% | [−2.16, −1.66] | 1e-4 | BELOW |

BH `q=.05`: **23/24 cells rejected**; the single non-rejection is `keep7@500 × TriviaQA contains`
(BH-adj `p=0.399`) — i.e. the only cell that is *statistically at* its floor rather than above or
below it.

## 4. Why the verdict is not KILL

A03's kill condition requires the pruned arm to be at floor on **all** knowledge indicators. It is
above floor, with BH-significant positive residuals, on:

* MMLU-**content** (+3.99pp, 12.3%),
* PopQA EM (+1.65pp, 41.8%) and PopQA `contains` even under the strict length-matched null (+1.91pp, 17.0%),
* TriviaQA EM (+9.33pp, **97.3%**) and TriviaQA `contains` length-matched (+21.2pp, 92.8%).

The `step500` arm is the control that makes this credible: it is **below or at** floor on *every*
interface (5 BELOW + 1 AT). So "at floor" is a state these nulls can actually detect at 1B — the
`step200k` arm's residuals are not an artefact of a null that is too easy to beat.

The intact arm is far above floor everywhere (26.5%–99.4%), so **the interfaces themselves are
valid at 1B**: where the pruned arm fails it is a *pruning* effect, not a task/interface artefact.
This is the intact-vs-pruned distinction the task asked for, and it comes out cleanly on 5 of 6
interfaces. The one exception is MMLU-letter, where the intact arm *is* above floor (+11.18pp) but
the pruned arms are below it, i.e. that interface breaks only for the pruned models — a
**model-dependent interface failure**, which is precisely why it must be dropped rather than
reported.

## 5. Two findings that change A03's design

**(a) Drop MMLU-letter at 1B; keep MMLU-content.** Letter-prediction distributions:

| Arm | A | B | C | D | modal share | vs own modal null | bf16 top-2 exact-tie rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| intact 16L | 1930 | 2418 | 8091 | 1603 | 57.6% (C) | +12.56pp (p=1e-4) | 6.98% |
| keep7+f2 @200k | 102 | 4838 | 9043 | 59 | 64.4% (C) | **−0.39pp (p=0.28)** | 10.95% |
| keep7+f2 @500 | **14042** | 0 | 0 | 0 | **100.0% (A)** | 0.00pp (p=1.0) | 0.00% |

The pruned arm is a near-constant `C`-emitter and is statistically **indistinguishable from its own
constant predictor**; the barely-healed arm is a *literal* constant predictor (always-A on all
14,042 items). Any A03 arm ranking read off MMLU-letter at 1B would be ranking degeneracy, not
knowledge. (Note `always-A = 0.2295` is below `always-D = 0.2689`, which is exactly how the
step500 arm lands "below chance" — a constant predictor that picked the *wrong* constant.)

**(b) `contains` must be length-controlled.** Mean prediction length: intact **13.4** chars →
keep7@200k **80.8** → keep7@500 **109.6** on PopQA (TriviaQA: 15.0 → 72.4 → 116.4). Healing makes
the model verbose, and `contains` gives credit for any gold substring, so verbosity alone buys
`contains` points: a purely input-blind 90-char string of frequent gold answers scores **0.0928**
on PopQA `contains` — nearly double the pruned arm's naive floor of `0.0488`, and it is *not*
knowledge. Under that control the pruned arm's PopQA `contains` residual fraction falls
**56.4% → 17.0%**. TriviaQA EM is immune to this (exact match, `frac` 97.3%), which is a second
reason to make it A03's primary axis.

## 6. Skipped / not run

* **Nothing was skipped for missing data.** All three datasets A03 names for the "old parametric
  knowledge" axis were present and loaded fully offline on `.82`
  (`cais/mmlu` 14,042 · `akariasai/PopQA` 14,267 · `mandarjoshi/trivia_qa rc.nocontext` 17,944),
  cache at `data/hf_datasets_cache/` on the zwfy6 disk.
* NQ-open is supported by the same harness and cached, but A03 names only MMLU-content / TriviaQA /
  PopQA, so it was not run.
* A03's other three knowledge axes (new injected facts, updated/conflicting knowledge,
  multi-evidence) are **not** tested here — this experiment only addresses the "old parametric
  knowledge" axis named in the kill condition. They still need their own floor checks before the
  6-arm build.

## 7. Reproduce

```bash
# on .82 (zwfy6 disk), 8 GPUs, ~7 min wall for all 3 arms x 3 tasks
PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
bash scripts/_run_a03_1b_floor_82.sh

# null calibration + significance (CPU, ~2 min)
python proposal/archive/A03-parametric-vs-external-memory/code/analyze_1b_knowledge_floor.py \
  --arms intact_1B_16L=A03_1B_base \
         pruned_healed_keep7f2_200k=A03_1B_keep7_step200k \
         pruned_healed_keep7f2_step500=A03_1B_keep7_step500 \
  --out proposal/archive/A03-parametric-vs-external-memory/evidence/a03_1b_floor_nulls.json
```

Artefacts: `proposal/archive/A03-parametric-vs-external-memory/evidence/a03_1b_floor_nulls.json`
(all 24 cells + null diagnostics) and the 6 harness `summary.json` under
`evidence/olmo2_{mmlu_content,closedbook}_results/`. Per-example dumps (needed to recompute any
null) live on `.82` at `olmo2_mmlu_content_results/A03_1B_*/per_example_mmlu.jsonl` and
`olmo2_closedbook_results/A03_1B_*/per_example_{popqa,triviaqa}.jsonl`.
