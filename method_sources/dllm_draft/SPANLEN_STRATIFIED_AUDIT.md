# Span-length stratified HumanEval-Infilling audit (Track B)

**Node:** `.73` (zwfy6, 8x H20) for generation. **All grading on wzc1** with
**evalplus 0.3.1** (`HUMANEVAL_PLUS_VERSION v0.1.10`, dataset hash
`fe585eb4df8c88d844eeb463ea4d0302`). Repo commit at start: `7a0275e`.
Total GPU cost **8.31 GPU-h** (6.84 accounted in shipped metrics + 1.47 in an
aborted run, §4). Ceiling was 8.

Grading is delegated **entirely** to the official EvalPlus sandbox
(`evalplus.evaluate` for whole-dataset runs, `evalplus.eval.untrusted_check` for
per-item runs). No test runner is hand-rolled anywhere in this track. Every
scoring invocation runs a grader self-test (split's own gold must pass, a `pass`
stub must fail) and refuses to emit numbers if it fails; all 9 scoring runs
returned `{"gold_pass": 12, "stub_fail": 12, "trustworthy": true}`. Every merge
asserts full shard coverage; all 9 merges reported `expected == merged`.

---

## 0. PIN RESULT: the record's blocking claim does **NOT** reproduce

The candidate record claimed, as its REQUIRED CONTROL:

> "I ran the canonical HumanEval solution for all 164 parents through the exact
> grader and got base=164/164 but **plus=143/164** — 21 parents FAIL the plus
> tests with their OWN reference solution", justifying a **"~+15 point
> correction, 1.5x the entire effect."**

### Re-derived

`runs/pin_canonical/canonical_solutions.jsonl` = `prompt + canonical_solution`
for all 164 HumanEval+ tasks, graded by `python -m evalplus.evaluate --dataset
humaneval` (full 164 coverage, so no assertion-lifting needed):

| program graded | base pass@1 | plus pass@1 |
|---|---|---|
| **HumanEval+ canonical** (`prompt + canonical_solution`) | **0.994** (163/164) | **0.982** (161/164) |
| same, `prompt + contract + canonical_solution` | 0.988 | 0.976 |
| same, at `EVALPLUS_MAX_MEMORY_BYTES=16GiB` | 0.994 | **0.994** (163/164) |

**plus = 161/164, not 143/164.** The default-cap configuration was re-run and
gave bit-identical 0.994/0.982 twice, so this is not sampling noise.

All three deviations are diagnosed, and **none** is "the reference solution is
semantically wrong":

1. **HumanEval/32 is unpassable by ANY program** — a genuine EvalPlus grader bug.
   In `evalplus.eval.unsafe_execute` the `find_zero` special oracle does
   `assert abs(_poly(*inp, out)) <= atol; continue`, and that `continue` skips
   the `progress.value += 1` below it, so `details` returns **empty** and
   `untrusted_check` forces `stat = FAIL` via `if len(details) != len(inputs)`.
   Confirmed by grading an independently written perfect bisection solver:
   also `base=fail`, `details len=0`. The attainable ceiling is **163**, not 164.
2. **HumanEval/15 and HumanEval/130 are memory-cap artifacts, not wrong answers.**
   Both fail only on the ~10^6-magnitude plus stress inputs (`[1000001]`,
   `[999999]`, …; 20 and 7 such inputs). Executing the canonical function on
   **every** plus input standalone gives 0 wrong answers and 0 timeouts. Raising
   `EVALPLUS_MAX_MEMORY_BYTES` from the 4 GiB default to 16 GiB makes both pass.
   They are OOM under `reliability_guard`.

**Excluding both artifacts, the ceiling is 163/163 base and 163/163 plus.**

### Where 143 came from — reproduced exactly

The record graded **the wrong program**. Reconstructing each parent from the
*infilling* split (`row['prompt'] + row['canonical_solution'] + row['suffix']`)
and grading THAT against HumanEval+ gives:

| program graded | base | plus | plus count |
|---|---|---|---|
| HumanEval+ canonical | 0.994 | **0.982** | 161/164 |
| infilling-reconstructed parents (first row per task) | 0.994 | **0.854** | 140/164 |
| infilling-reconstructed parents (majority-vote reference file) | 0.994 | **0.854** | 140/164 |
| majority-vote parents at 16 GiB cap | 0.994 | **0.866** | **142/164** |

142 + the 1 task lost to the HumanEval/32 grader bug = **143**. The record's
number is an artifact of grading infilling-split gold against HumanEval+ tests,
plus the two environment-dependent flakes.

**Mechanism:** the infilling splits ship the **ORIGINAL** OpenAI HumanEval
reference solutions; HumanEval+ ships **corrected** ones plus a far larger,
stricter input set. Reconstructed parents are byte-identical to the HumanEval+
canonical for **0 of 164** tasks (e.g. HumanEval/0: infilling gold is the O(n^2)
double loop, HumanEval+ canonical is the sorted-adjacent-pair version).

### Decision

**The claimed "~+15 point correction" must NOT be applied.** It does not measure
a broken grader ceiling; it measures a program/test-suite mismatch. Applying it
would have inflated every arm ~15 points against a phantom ceiling.

The underlying concern is real but **relocated** — see §1, the control this track
actually needs.

---

## 1. The control that IS needed: the splits' own gold ceiling is not 1.0, and it is length-dependent

Because infilling gold != HumanEval+ canonical, the attainable maximum per item
must be measured. I spliced each split's **own** gold middle back in
(`prompt + canonical_solution + suffix`) and graded it, all rows, both axes
(`scripts/spanlen_gold_ceiling.py`):

| split | n | gold ceiling **base** | gold ceiling **plus** |
|---|---|---|---|
| SingleLine | 1033 | 0.9894 | **0.8025** |
| MultiLine | 5815 | 0.9887 | **0.7620** |
| RandomSpan | 1640 | 0.9939 | **0.8500** |

Per stratum (gt middle length in Qwen2.5-Coder tokens), base / plus:

| stratum | SingleLine (n) | MultiLine (n) | RandomSpan (n) |
|---|---|---|---|
| 0-4 | .9898 / .7614 (197) | .9898 / .7614 (197) | .9972 / .8729 (354) |
| 5-8 | .9916 / .8126 (475) | .9887 / .8034 (529) | 1.0000 / .8598 (271) |
| 9-16 | .9891 / .8073 (275) | .9918 / .7951 (849) | .9944 / .8663 (359) |
| 17-32 | .9740 / .8182 (77) | .9916 / .7955 (1423) | .9946 / .8474 (367) |
| 33-64 | 1.0000 / .8889 (9) | .9870 / .7515 (1698) | .9909 / .8136 (220) |
| 65-128 | — | .9819 / **.6617** (940) | .9545 / **.7273** (66) |
| 129+ | — | 1.0000 / .8436 (179) | 1.0000 / 1.0000 (3) |

**(a) On the base axis the splits are clean.** Gold reaches .989-.994, and the
residual is dominated by HumanEval/32's grader bug (all 11 of SingleLine's
gold-base failures are HumanEval/32 rows). Independent check: every
reconstructed parent passes **its own shipped HumanEval test, 164/164**. The
benchmark is internally well-posed.

**(b) On the plus axis the ceiling drops to .76-.85 and is NOT flat in length.**
MultiLine 65-128 has ceiling **.6617** vs 9-16's **.7951** — a 13.3-point
ceiling swing *between strata of the same benchmark*. The loss traces to exactly
**23 parent tasks** (of 164) whose original HumanEval reference is buggy w.r.t.
HumanEval+ (top: HumanEval/95 21 rows, /140 20, /132 16, /39 15, /124 15,
/127 15, /123 12, /111 11). Their holes are not uniformly spread over lengths,
so they load unevenly onto strata.

**Consequence:** a raw per-stratum pass@1 curve on the plus axis **confounds
model degradation with a stratum-dependent ceiling**. Every table below reports
raw pass@1, the stratum ceiling, and ceiling-conditioned pass@1 (restricted to
items whose own gold passes). This is not cosmetic — for `qwen_fim` on RandomSpan
65-128 it moves plus pass@1 from **.3333 to .4583** (+12.5 points).

---

## 2. HEADLINE: the hypothesis is **REVERSED**. DreamOn's variable-length mechanism *collapses* at long spans

The candidate's premise was that DreamOn's advantage is concentrated where gt
middle length is far from the canvas prior, and vanishes when it is close.
**The opposite is what happens.**

### MultiLine_sub (n=420, line-aligned holes, 60 per stratum, all 60 long items drawn from the 179 gt_len>128 cases)

Ceiling-conditioned pass@1, **plus** axis (raw and ceiling in the JSON):

| stratum | n | ceil(plus) | dreamon_fim | dream_fim (oracle len) | qwen_fim (AR FIM) |
|---|---|---|---|---|---|
| 0-4 | 60 | .750 | **.9778** | .9556 | .9333 |
| 5-8 | 60 | .833 | **.9200** | .9000 | .9200 |
| 9-16 | 60 | .717 | .7442 | **.8605** | .7442 |
| 17-32 | 60 | .850 | .4706 | .6863 | **.7647** |
| 33-64 | 60 | .733 | .2727 | .5682 | **.6818** |
| 65-128 | 60 | .633 | **.0526** | .4737 | **.5000** |
| 129+ | 60 | .883 | **.0189** | .7170 | **.7547** |
| **OVERALL** | **420** | .771 | **.4969** | .7438 | **.7654** |

DreamOn is the **best** arm at 0-4 (.978) and ties at 5-8, then falls off a
cliff: **.0189 at 129+** against qwen_fim's .7547 and dream_fim's .7170.

**Ceiling-conditioned short-vs-long gap** (short={0-4,5-8,9-16},
long={33-64,65-128,129+}; bootstrap 10k):

| axis | arm | gap | 95% CI |
|---|---|---|---|
| plus | **dreamon_fim** | **+0.7729** | [+0.6990, +0.8462] |
| plus | dream_fim | +0.3058 | [+0.2098, +0.4016] |
| plus | qwen_fim | +0.2103 | [+0.1140, +0.3058] |
| base | **dreamon_fim** | **+0.7443** | [+0.6759, +0.8125] |
| base | dream_fim | +0.3944 | [+0.3093, +0.4795] |
| base | qwen_fim | +0.2811 | [+0.1907, +0.3662] |

DreamOn's length sensitivity is **~3.7x the AR FIM control's** on the plus axis,
and the CIs do not overlap. **Paired McNemar** (ceiling-conditioned, plus,
byte-identical items): dreamon vs qwen 17 vs 104 discordant, p=2.0e-16; the
per-stratum breakdown shows DreamOn is statistically indistinguishable at 0-4
(3 vs 1, p=0.625), 5-8 (2 vs 2, p=1) and 9-16 (5 vs 5, p=1), then loses
decisively from 17-32 onward (p=7.3e-4) through 129+ (1 vs 40, p=3.8e-11).

### Mechanism — verified, not inferred

DreamOn under-generates catastrophically at long spans. Mean emitted middle
tokens vs mean gt length, MultiLine_sub:

| stratum | mean gt_len | mean emitted middle_tokens |
|---|---|---|
| 0-4 | 4 | 3.7 |
| 9-16 | 12 | 9.2 |
| 65-128 | 84 | **27.9** |
| 129+ | 159 | **22.7** |

Emitted length *saturates near ~25 tokens* and then decreases. Concrete case,
`MultiLineInfilling/HumanEval/124/L0_L13`, gt_len=149: DreamOn emitted **4
tokens**, namely `'    # write here'`. It emits a placeholder comment instead of
expanding the canvas. This is not a recovery/parsing artifact — `recovered`
status is `{ok: 51, ok_empty_suffix: 9}` at 129+ with **zero**
`suffix_not_found`, only 5/420 programs unparseable, and 0 truncations and
0 generation errors across all arms.

### RandomSpan_sub (n=363, mid-token holes) — all three arms collapse

| stratum | n | ceil(plus) | dreamon_fim | dream_fim | qwen_fim |
|---|---|---|---|---|---|
| 0-4 | 60 | .867 | .3846 | .2692 | **.9615** |
| 5-8 | 60 | .917 | .2727 | .2545 | **.8727** |
| 9-16 | 60 | .867 | .2308 | .2308 | **.8269** |
| 17-32 | 60 | .900 | .1852 | .1296 | **.7963** |
| 33-64 | 60 | .750 | .1556 | .1333 | **.5556** |
| 65-128 | 60 | .733 | .0455 | .1818 | **.4318** |
| **OVERALL** | **363** | .840 | **.2164** | **.2000** | **.7574** |

Both diffusion arms sit at ~.20-.22 at every length, versus .76 for AR FIM
(dreamon vs qwen: 8 vs 173 discordant, p=1.7e-41). The two diffusion arms are
**statistically indistinguishable from each other** (31 vs 26, p=0.597), i.e.
on mid-token holes the oracle length hint is worth nothing. Full-split
confirmation at n=1640 (`score_RandomSpan_{qwen,dream}_fim.json`): qwen_fim
.7956 vs dream_fim .2059 ceiling-conditioned plus, McNemar p=3e-201.

Diffusion failures here are genuine model errors, not harness bugs: dream_fim
emits the correct *number* of tokens with wrong content (gold `' if '` ->
`' if ('`; gold `'          current'` -> `' , current'`), which is why 726/1640
of its programs do not parse. RandomSpan cuts holes *inside* tokens, which is
maximally hostile to a tokenizer-aligned canvas.

### Reading

The narrow claim "DreamOn is on par with SOTA AR at variable-length infilling"
survives only in the **short-span, line-aligned** corner (gt_len <= 8 on
MultiLine, where it is actually best at .978). It **fails** exactly where its
variable-length mechanism is supposed to matter most — long spans far from the
canvas prior — because the mechanism does not in fact expand that far. And it
fails everywhere on mid-token holes.

---

## 3. Secondary verified numbers

- **"179 MultiLine cases have gt_len > 128"** reproduces **exactly**: 179/5815
  in Qwen2.5-Coder tokens, max 239. So `max_new=256` suffices and **0 items are
  budget-truncated** in any arm (confirmed empirically: truncated=0 everywhere).
  RandomSpan has 3 such items (max 154), SingleLine 0 (max 46).
- Length distributions: SingleLine median 7 tokens (p95 20), MultiLine median 31
  (p95 110), RandomSpan median 12 (p95 62).

---

## 4. Budget: the per-case latencies I was handed were wrong by 29x for DreamOn

Handed telemetry: `dreamon_fim 0.89 s/case`, `dreamon_oracle 1.07`,
`dream_fim 0.39`, `qwen_fim 0.21`, total ~6 GPU-h. Measured on `.73`, 8-way
sharded, `max_new_tokens=256`:

| arm | handed s/case | measured s/case | ratio |
|---|---|---|---|
| qwen_fim | 0.21 | 0.74-1.22 | ~4-6x |
| dream_fim | 0.39 | 1.53-1.92 | ~4-5x |
| dreamon_fim | 0.89 | **26.0 mean / 2.3 median / 144 p99** (RandomSpan) | **29x on the mean** |

**Root cause, and it is a real trap for anyone re-running this.** In
`run_dreamon`, `max_new_tokens` is DreamOn's *canvas headroom*, not a stopping
budget (`max_length = max_new_tokens + input_len - num_masks`). Its
variable-length mechanism then operates over that whole canvas, so raising
`max_new` from the harness default 64 to the 256 required to fit the 179
`gt_len>128` MultiLine items multiplies its step count. Cost is also violently
heavy-tailed (median 2.3 s, mean 26.0 s), so a small pilot's mean under-predicts
badly. Full RandomSpan for this one arm projected to ~12.1 GPU-h.

**Action taken:** I killed the over-budget full-split DreamOn run at 1.47 GPU-h
(verifying by `grep -v score` that no scoring process was in the kill set — a
known hazard in this repo) and rebuilt the design as a **paired stratified
subsample shared by all three arms**, so every McNemar comparison is paired on
byte-identical items, with 60 per stratum so the rare long strata are not n=3:

- `data/infilling/spanlen_RandomSpan_sub.jsonl` n=363, md5 `1619a9b82f05ae6e3bef3202dcce94f9`
- `data/infilling/spanlen_MultiLine_sub.jsonl` n=420, md5 `eed57c6ff6a4a35954a54af5609bb6d1`

All subsets and all 6 arm solution files verified md5-identical across the
wzc1/zwfy6 disk boundary after `scp -O`.

---

## 5. What would kill these claims

- **§2 headline (DreamOn collapses at long spans)** dies if the collapse is a
  sampler-configuration artifact rather than a model property. The specific risk
  is that `max_new=256` interacts badly with DreamOn's expansion schedule — the
  same coupling that caused the 29x cost blow-up. **This is the single most
  important follow-up: re-run 129+ with a per-item `max_new` set just above
  gt_len, and with `--initial-masks` > 4.** If DreamOn's emitted length still
  saturates at ~25 tokens, the finding is a model property; if it tracks gt_len,
  the finding is about this harness's default and must be restated. I have NOT
  run that control, so §2 should be read as "DreamOn as invoked by this harness
  at max_new=256, initial_masks=4, T=0" rather than "DreamOn the model".
  > **★ 2026-08-12 — THE CONTROL HAS NOW BEEN RUN, and this bullet's own
  > "must be restated" branch is the one that fired.** A05's K1 gate swept
  > `initial_masks ∈ {8,32,128}` on **full-program HE+/MBPP+** (not on the
  > infilling strata, which stay untested):
  > * Emitted length **tracks the canvas, it does not saturate at ~25**: HE+ mean
  >   emitted goes 2.35 → 12.87 → 48.53 tokens and empty outputs go 128/164 →
  >   75/164 → **0/164**. So the "collapse" was **this harness's default**, and
  >   §2's phrasing must be read as a statement about `initial_masks=8`, exactly
  >   as this bullet hedged.
  > * Quality moves with it: MBPP+ **.085 → .3545**, HE+ **.122 → .2134**
  >   (**.2561** after a stitch-bug fix; **.4817** at canvas=128).
  > * **However** the ≥0.8 median emitted/gold criterion is still NOT met
  >   (peak 0.46), so "DreamOn is a calibrated length controller" is also
  >   unsupported. Both the collapse claim and its converse fail.
  > * `max_new` per-item headroom was **NOT** varied (only the initial canvas),
  >   and the oracle arms were not run — so the `max_new=256` half of this
  >   bullet's hypothesis remains formally untested.
  > Evidence: `Mixture-of-Memory/proposal/archive/A05-structural-dllm-cost-frontier/`.
- **§1(b) ceiling non-flatness** is measured at the 4 GiB default cap. Raising
  the cap recovered exactly 2/164 on the HumanEval+ canonical, so cap effects
  are small but nonzero; per-stratum ceilings should carry +-1-2 points. The
  23-parent attribution is robust (those are wrong answers on ordinary inputs,
  not OOM on 10^6 inputs).
- **§2 RandomSpan diffusion floor** is confounded with the mid-token hole
  boundary by construction. The MultiLine leg (line-aligned) is the control and
  it shows dream_fim recovering to .744, so the RandomSpan floor should be
  attributed to tokenizer misalignment, not to diffusion per se.
- **dreamon_oracle was not run.** The clean single-variable test of "does DreamOn
  benefit from a length hint" is missing; §2 compares DreamOn (non-oracle) to
  dream_fim (oracle, different model), so model and oracle-ness are entangled.

## 6. Artifacts

```
scripts/spanlen_gold_ceiling.py           gold-ceiling measurement (official sandbox, both axes)
scripts/spanlen_stratified_score.py       per-stratum scoring + ceiling conditioning + termination accounting
scripts/spanlen_assemble.py               tables + bootstrap gap CIs + paired McNemar
scripts/_run_spanlen_stratified_8gpu.sh   8-way sharded runner, HARD coverage assertion on merge
runs/spanlen/gold_ceiling_{SingleLine,MultiLine,RandomSpan,RandomSpan_sub,MultiLine_sub}.json
runs/spanlen/score_{RandomSpan,RandomSpan_sub,MultiLine_sub}_{qwen_fim,dream_fim,dreamon_fim}.json
runs/spanlen/spanlen_summary_{full1640,sub,ML_sub}.json
runs/pin_canonical/                       PIN sample files (canonical, +contract, reconstructed parents)
```
