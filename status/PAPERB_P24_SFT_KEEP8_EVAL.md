# Paper B P2.4 — keep8 post-SFT eval battery (.73)

**Date**: 2026-08-08 | **Node**: `.73` (8×H20 cc9.0, zwfy6) | **Commit**: `fd1633c`
**Driver**: `scripts/_run_olmo2_p24_eval_sft_keep8_73.sh` | **Log**: `logs/p24_eval_sft_keep8_73.log` | **PID** 3071993 (wrapper) / 3071994 (driver)
**Analysis**: `scripts/paired_analysis_p24_sft_keep8.py` → `results/paperb_p24_sft_keep8_paired.json`
**Runtime**: 06:05:22 → 06:21:37 = **16 min** (est. was ~90 min)

Pre/post pair, single-variable, **same node + same arch** (both H20 / zwfy6):

| | ckpt | battery name |
|---|---|---|
| pre  | `outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt` | `7B_keep8_step121000_v2{,_know}` |
| post | `outputs/olmo2_p24_sft_keep8fresh2/final.pt` (34.15 GB, step=842 keep=8 fresh=2) | `7B_p24_sft_keep8fresh2_final{,_know}` |

Protocol: `chat_template=False`, `--add_bos 0`, per-shard `CUDA_VISIBLE_DEVICES=$g` + `LOCAL_RANK=0 RANK=$g`, 8-way shard + merge.

---

## 1. THE FALSIFIABLE PREDICTION — observed vs predicted

The n=3 fit (`ΔPPL% = 1.60149·prePPL − 7.34121`, r=0.998) predicted **ΔPPL% ≈ +14.0%**, post-PPL ≈ **15.20**.

| | value |
|---|---|
| pre-SFT PPL | **13.3329** |
| post-SFT PPL | **14.6857** |
| **observed ΔPPL%** | **+10.15%** |
| predicted ΔPPL% | +14.01% |
| **residual** | **−3.86 pp** (post-PPL undershoots by 0.515) |

Shard-level bootstrap (8 shards, 20k resamples, identical per-shard token counts pre vs post — verified):
**ΔPPL% 95% CI = [+10.05, +10.23]**. Per-shard spread is tight (+9.96 … +10.29).
**The +14.0% prediction lies far outside this CI.** The undershoot is not measurement noise.

### Verdict: the relationship saturates — but the pre-registered "materially lower" bar was NOT met

This lands **between** the two pre-registered outcomes, so it must be stated precisely rather than
forced into either bucket:

- It is **not** "lands near 15.2": +10.15% vs +14.0% is a 3.86 pp miss, statistically decisive.
- It is **not** "materially lower" as pre-registered either: that bar was post-PPL < 14.5 (Δ < 9%),
  and we observe post-PPL 14.686 / Δ 10.15% — **just above the bar on both counts.**

The honest reading: **the linear n=3 fit was extrapolating past its range, and damage-sensitivity
is sub-linear (concave/saturating) rather than either linear-continuous or sharply flattened.**
keep8's Δ is still the largest of the four arms (monotone in pre-PPL is preserved), but it grows
much more slowly than linear extrapolation implied.

| arm | pre-PPL | ΔPPL% | resid vs n=4 fit |
|---|---:|---:|---:|
| full32 | 7.398 | +4.46% | −0.97 |
| shortgpt16 | 9.780 | +8.51% | +0.83 |
| keep14 | 10.561 | +9.43% | +1.02 |
| **keep8** | **13.333** | **+10.15%** | −0.88 |

- n=4 linear refit: slope **1.6015 → 0.9417** (−41%), and **r 0.998 → 0.907** (r² 0.996 → 0.822).
  A single new point at the extrapolated end destroys most of the apparent linearity.
- Quadratic fit leading coefficient = **−0.215 (negative ⇒ concave/saturating)**.
- log-linear fit `9.88·ln(pre) − 14.66` gives r=0.944 (r²=0.892), **better than the n=4 linear fit** —
  consistent with saturation.

⚠️ With n=4 all of these are descriptive, not inferential. The load-bearing claim is the narrow one:
**the n=3 linear law over-predicts keep8 by 3.86 pp, well outside the ±0.09 pp measurement CI.**

---

## 2. Downstream pre → post deltas

McNemar is **exact binomial on discordant pairs, TWO-SIDED** (no direction was pre-registered for
downstream metrics). `b` = post-correct/pre-wrong (SFT gained), `c` = post-wrong/pre-correct (SFT lost).
Paired bootstrap = 10k resamples, seed 0, item_id-aligned.

### core6 (acc_norm, the paper metric)

| task | n | pre | post | Δ | 95% CI | b/c | McNemar p (2-sided) |
|---|---:|---:|---:|---:|---|---|---:|
| hellaswag | 10042 | 0.5167 | 0.5144 | −0.0023 | [−0.0072, +0.0026] | 301/324 | 0.379 |
| arc_challenge | 1172 | 0.3635 | 0.3490 | −0.0145 | [−0.0316, +0.0026] | 46/63 | 0.125 |
| **arc_easy** | 2376 | 0.6549 | 0.6124 | **−0.0425** | [−0.0543, −0.0303] | 57/158 | **3.63e−12** |
| piqa | 1838 | 0.7149 | 0.7198 | +0.0049 | [−0.0049, +0.0147] | 47/38 | 0.386 |
| winogrande | 1267 | 0.5217 | 0.5241 | +0.0024 | [−0.0182, +0.0229] | 89/86 | 0.880 |
| openbookqa | 500 | 0.3680 | 0.3660 | −0.0020 | [−0.0260, +0.0240] | 20/21 | 1.000 |
| **MACRO** | — | **0.52328** | **0.51428** | **−0.00901** | **[−0.01579, −0.00235]** | — | — |

core6 macro drops by **0.90 pp**, CI excludes 0. **arc_easy alone supplies −0.43 pp of that** and is
the only individually significant task; the other five are all non-significant.

### know5 (acc_norm)

| task | n | pre | post | Δ | 95% CI | b/c | p |
|---|---:|---:|---:|---:|---|---|---:|
| mmlu | 14042 | 0.2545 | 0.2491 | −0.0053 | [−0.0140, +0.0033] | 1756/1831 | 0.217 |
| lambada_openai | 5153 | 0.4461 | 0.4411 | −0.0050 | [−0.0138, +0.0041] | 256/282 | 0.281 |
| boolq | 3270 | 0.6284 | 0.6196 | −0.0089 | [−0.0159, −0.0018] | 56/85 | 0.0181 |
| commonsense_qa | 1221 | 0.4210 | 0.3989 | −0.0221 | [−0.0360, −0.0082] | 26/53 | 0.00318 |
| social_iqa | 1954 | 0.4365 | 0.4447 | +0.0082 | [−0.0026, +0.0189] | 66/50 | 0.163 |
| **MACRO** | — | **0.43731** | **0.43067** | **−0.00664** | **[−0.01116, −0.00202]** | — | — |

### MMLU dual protocol (14042 items, fully paired)

| protocol | pre | post | Δ | 95% CI | b/c | p |
|---|---:|---:|---:|---|---|---:|
| letter | 0.25431 | 0.24833 | −0.0060 | [−0.0145, +0.0026] | 1723/1807 | 0.162 |
| content_norm | 0.34269 | 0.33649 | −0.0062 | [−0.0105, −0.0019] | 431/518 | 0.00522 |
| content_raw | 0.32225 | 0.31584 | −0.0064 | [−0.0105, −0.0022] | 402/492 | 0.00289 |

Letter protocol stays pinned near chance (0.25) both before and after — SFT does not buy
letter-protocol competence at keep8. Content protocol is meaningfully above chance and degrades
by a small but significant 0.62 pp. Note the letter test has ~4× the discordant pairs (3530 vs 949),
i.e. it is far noisier per item, which is why a similar Δ is non-significant there.

### Closed-book QA

| task/metric | n | pre | post | Δ | 95% CI | b/c | p |
|---|---:|---:|---:|---:|---|---|---:|
| popqa/em | 14267 | 0.04227 | 0.03343 | −0.0088 | [−0.0113, −0.0064] | 96/222 | 1.22e−12 |
| popqa/contains | 14267 | 0.12525 | 0.12673 | +0.0015 | [−0.0022, +0.0051] | 362/341 | 0.451 |
| popqa/f1 | 14267 | 0.08697 | 0.08700 | +0.00003 | [−0.0026, +0.0026] | — | — |
| **triviaqa/em** | 17944 | 0.15721 | 0.11419 | **−0.0430** | [−0.0469, −0.0390] | 277/1049 | **6.77e−106** |
| triviaqa/contains | 17944 | 0.30461 | 0.27497 | −0.0296 | [−0.0344, −0.0249] | 691/1223 | 2.53e−34 |
| triviaqa/f1 | 17944 | 0.24200 | 0.20704 | −0.0350 | [−0.0385, −0.0313] | — | — |

**TriviaQA EM is the single largest casualty (−4.30 pp, p≈1e−106).** PopQA splits informatively:
EM falls but `contains` and F1 are flat (CI straddles 0) — the SFT'd model still surfaces the right
string but stops matching it exactly, i.e. **a formatting/verbosity shift, not pure knowledge loss**.
TriviaQA degrades on `contains` and F1 too, so there the loss is real and not only formatting.

---

## 3. Integrity checks

- **Shard assertion: 5/5 stages asserted 8/8 before merge** (PPL 06:09:22, core6 06:11:45,
  know5 06:14:41, MMLU 06:17:30, closedbook 06:21:34). No `SHARD MISSING` / `ABORT merge` in the log.
- **Per-item predictions retained** on both sides: core6 `per_example_<task>.jsonl` (via
  `--save_per_example`), MMLU `per_example_mmlu.jsonl` (14042), closedbook popqa (14267) /
  triviaqa (17944). No re-run needed for pairing.
- **Pre-flight anchor guard** added to the driver: aborts if the pre-SFT anchor lacks summaries or
  per-item preds (pairing would be impossible). It passed.
- **Independent cross-validation**: every paired pre/post value reproduces the harness's own
  `summary.json` to <1e−9 — all 12 core6 cells, MMLU letter/content_norm/content_raw both sides,
  closedbook em/contains both sides. Item alignment is therefore sound (not a silent misjoin).
  Pre-side reproduces the MAIN-verified anchors exactly: **PPL 13.3329 ✓, core6 0.52328 ✓**.
- **PPL comparability**: pre and post both 8,384,512 tokens / 4096 windows, and per-shard token
  counts are identical — the ΔPPL% is over exactly the same held-out Dolmino windows.
- **Exact-test implementation validated**: no scipy on `.73`, so the exact two-sided binomial is
  computed with `math.comb` + `Fraction`. Checked against scipy 1.18 on 73 cases → **0 mismatches**
  (worst rel. err 7.4e−14), and it reproduces the harness's own recorded value
  `p=6.4055693680e−58`. `Fraction` was required: the naive `2.0**n` float form **overflows** once
  discordant pairs exceed ~1024, which happens on MMLU (n=3530).
- SFT training log: 0 NaN, step 840 loss=1.3947, saved 05:56:44.

## 4. Caveat (do not drop)

keep8's pre-SFT anchor is **step121000, NOT 200k** — keep8 never reached 200k
(`status/PAPERB_TABLE4_BUDGET_DEFECT.md`). This is fine for the present experiment, which holds the
arm's **own** pre-SFT ckpt fixed and measures the SFT delta on it. It is **NOT valid for
compute-matched depth comparisons against keep14 (200k)**. The caveat is also recorded in the
`caveat` field of `results/paperb_p24_sft_keep8_paired.json`.

A second-order consequence for §1: keep8's pre-PPL of 13.333 is itself partly a *less-trained*
number, not purely a *more-damaged* one. So the x-axis of the damage-scaling fit mixes depth damage
with training budget at the keep8 point, which is an additional reason to treat the saturation
reading as descriptive.

## 5. Sibling arms (not touched)

- `.104` keep12 — **still training** at check time (step 780/842, 8/8 GPUs at 100%). Untouched.
- `.82` keep10 — **finished 06:16:12** (`final.pt`, step=842) and the node is now **idle 0/8**.
  Untouched per instructions; flagging for MAIN as a ready-to-eval arm.
