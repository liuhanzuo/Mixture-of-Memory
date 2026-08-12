# A04 — capability scoring at the repaired PLATEAU rule's own earliest accept checkpoint

**Date**: 2026-08-12. **GPU cost, measured not estimated**: 8×H20 on `.73`, wall
14:19:03 → 14:27:49 = **526 s** for three checkpoints × four axes = **1.17 GPU-h**
(526 s × 8 cards). Step-100000 alone: **162 s** (150 000: 159 s; 50 000: 161 s).

This closes the defect `A04_PLATEAU_REPAIR_AND_MARGIN_SENSITIVITY.md` §1.4 / §5 item 1 named
as blocking a claim, and it closes it **against** A04 rather than for it.

| item | status before | status now |
|---|---|---|
| PLATEAU-vs-NI at step 100 000, R3's own earliest accept | **UNMEASURED** ("GPU work, not done here") | **MEASURED — the rules DISAGREE, 3/3 decision axes** |
| "where the earliest disagreement lies" | claimable only at step 200 000 | **moves to step 100 000** |
| step 150 000 / step 50 000 capability | unscored | **scored** (bracket around the accept boundary) |

---

## 1. The defect, verbatim

> "any claim about *where* the earliest disagreement lies now requires step-100 000
> capability scoring, which is GPU work and is **not done here**."
> — `A04_PLATEAU_REPAIR_AND_MARGIN_SENSITIVITY.md:142-143`

The repaired rule **R3** first accepts at step **100 000** (`rate_5k = 0.86012 %/5k` ≪ T = 2.0),
not step 200 000. Pilot Zero scored capability **only** at step 200 000, and there is **4.6386 %**
further relative PPL improvement between the two, so the relocation is not negligible.

---

## 2. What was scored, and the protocol identity argument

Arm `keep7f2` = `outputs/olmo2_probe2_1B_keep7fresh2_16card/` (OLMo-2-0425-1B, front-7 inherited
+ 2 fresh, 9 of 16 layers). Three checkpoints newly scored on all four A04 axes:

| new result dir | ckpt | steps |
|---|---|---|
| `olmo2_mmlu_content_results/A04_1B_keep7f2_step{50000,100000,150000}` | `step{...}.pt` | MMLU-content, n=14 042 |
| `olmo2_closedbook_results/A04_1B_keep7f2_step{...}` | same | TriviaQA n=17 944, PopQA n=14 267 |
| `olmo2_closedbook_results/A04_1B_keep7f2_step{...}_nq` | same | NQ-open n=3 610 (demoted) |

**Protocol is identical to the step-200000 baseline, and that is demonstrated, not asserted.**
The baseline cells were produced by `scripts/_run_a03_1b_floor_82.sh` (MMLU + popqa/triviaqa) and
`scripts/_run_a03_axes_floor_82.sh` (nq_open). Their exact runtime parameters were recovered from
the progress logs still on zwfy6 — `logs/a03_1b_floor_progress.log:1` reads
`ngpu=8 mmlu_bs=64 cb_bs=48`, `logs/a03_axes_floor_progress.log:2` reads `nq_bs=48` — and
reproduced verbatim in `code/a04_step100k_axes_driver.sh`, together with
`--content_desc full --add_bos 0 --max_new_tokens 32 --merge --n_boot 10000`.

`--add_bos 0` is the **base-LM protocol**: `chat_template=False`, no BOS. OLMo-2-0425-1B has no
SFT/RL, so a chat template would be both unfair and incomparable with every existing cell. Both
new summaries record `add_bos: false`, confirming it took effect rather than being silently
dropped.

**The harness is the same code, verified two ways.** `scripts/eval_olmo2_mmlu_content.py` and
`scripts/eval_olmo2_closedbook_qa.py` are byte-identical across wzc1 and zwfy6 (md5
`fe4a62db…`, `2ed41993…`). The only harness commit between the baseline eval (2026-08-08 20:51)
and this pass is **`7ac9653`** (2026-08-08 22:29) — the same commit `A04_MARGIN_GUARD_PREREG.md`
names as falling between the two D5 dumps. `git show 7ac9653 -- scripts/eval_olmo2_mmlu_content.py`
touches **only** the `--any_family` base-mode routing branch (adding
`load_truncated_any_family`). The OLMo `--ckpt` → `load_pruned_model` path this driver takes is
**unchanged**. That distinction matters: the repo's standing rule is that same-arch/same-harness
re-runs are **byte-identical**, so a code delta on the *live* path would have made these cells
non-comparable rather than merely noisy.

`keep_front_layers=7 / n_fresh_layers=2` were passed explicitly, but `load_pruned_model` reads them
**from the ckpt meta and raises if the CLI disagrees** — so passing them is a free assertion that
the checkpoint really is keep7+fresh2, not a guess. All three summaries confirm
`keep=7 fresh=2 num_hidden_layers=9`.

### 2.1 Integrity — asserted per cell, not inferred from a summary

For **all 20 cells** (5 arms × 4 axes), via `load_shards` in the imported Pilot Zero code plus an
additional alignment pass:

| assertion | result |
|---|---|
| exactly 8 shard files | **8/8, all 20 cells** |
| every shard index 0..7 present exactly once | **pass** |
| no duplicate `item_id` after merge | **pass** |
| exact item count (14 042 / 17 944 / 14 267 / 3 610) | **pass** |
| zero rows with `nan: true` | **0, all 20 cells** |
| `item_id` sequence identical across arms per axis | **pass** — so the paired difference is genuinely item-paired |

The last one is not decoration: `load_shards` sorts by `item_id`, so two arms covering different
id sets would pair item *k* of one against a **different** item *k* of the other and produce a
silently wrong difference. ⚠️ `nq_open` per-example files live in the **separate `_nq`-suffixed
dir**; a glob of only the main arm dir would have wrongly reported the axis missing.

### 2.2 Regression guard — the archived cells reproduce EXACTLY

Before reporting anything new, the archived step-200000 cells were recomputed through the same code
path and compared to `evidence/pilot_zero_rule_disagreement.json`:

| axis | `diff_mean_pp` | `lower95_one_sided_pp` | `delta_pp` | `ni_accept` | max abs diff |
|---|---|---|---|---|---|
| TriviaQA | reproduces | reproduces | reproduces | False = False | **0.000e+00 pp** |
| PopQA | reproduces | reproduces | reproduces | False = False | **0.000e+00 pp** |
| MMLU-content | reproduces | reproduces | reproduces | False = False | **0.000e+00 pp** |
| NQ-open | reproduces | reproduces | reproduces | False = False | **0.000e+00 pp** |

Likewise every `rate_5k` was cross-checked against `evidence/a04_plateau_rule_repair.json` rather
than trusting a second implementation of the formula: **max abs diff 0.0** at tol 1e-9
(0.8601172082 / 0.3560056973 / 0.1317291830). A new cell computed by a path that cannot reproduce
the published one would not be comparable to it, so the script exits non-zero instead of
publishing in that case.

Nothing was reimplemented. `ni_rule`, `ratio_rule`, `build_nulls`, `build_axis_data` and the frozen
`PREREG` dict (`T=2.0`, `rho=0.85`, `Δ=0.10·residual`, git `d1ba737`) are **imported** from
`pilot_zero_rule_disagreement.py`, which in turn imports A03's canonical scorers/nulls.

---

## 3. The four axes at step 100 000

Pre-registered convention `split`; null-calibrated residuals; `Δ = 0.10 · residual(intact)`;
one-sided lower 95 % bound of the paired item bootstrap (n_boot = 5000).

| axis | n | reported | null | residual (pp) | frac of intact residual | lower-95 % of diff (pp) | Δ (pp) | NI |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| TriviaQA EM | 17 944 | 0.0777419 | 0.0025635 | 7.5178 | **18.59 %** | **−33.5377** | 4.0431 | **REJECT** (8.29× Δ) |
| PopQA EM | 14 267 | 0.0337843 | 0.0229200 | 1.0864 | **8.23 %** | **−12.5955** | 1.3205 | **REJECT** (9.54× Δ) |
| MMLU-content | 14 042 | 0.3174049 | 0.2844502 | 3.2955 | **32.19 %** | **−7.5132** | 1.0239 | **REJECT** (7.34× Δ) |
| NQ-open EM *(demoted)* | 3 610 | 0.0243767 | 0.0055402 | 1.8837 | 19.43 % | −8.6427 | 0.9695 | REJECT (descriptive only) |

Raw harness accuracies at step 100 000: `letter=0.2582253240`, `content_raw=0.3005269905`,
`content_norm=0.3174049281`, `n_valid=14042`, `n_nan=0`.

---

## 4. The answer: at step 100 000 the rules DISAGREE

**PLATEAU(R3) ACCEPTS** at step 100 000 (`rate_5k = 0.86012 %/5k` < T = 2.0).
**NI(Δ) REJECTS on 3/3 decision axes**, each by **7.3–9.5× its own Δ**.

So the two rules **disagree at the repaired rule's own earliest accept point**, and the cell that
was UNMEASURED is now measured.

| step | R3 | rate_5k (%/5k) | NI reject / decision axes | rules disagree? |
|---:|:--|---:|:--|:--|
| 50 000 | **UNDEFINED** (first trajectory point, no preceding interval) | — | 3/3 | not evaluable |
| **100 000** | **ACCEPT** | **0.86012** | **3/3** | **YES** |
| 147 000 | ACCEPT | 0.35601 | *no capability scored* | — |
| 150 000 | **UNDEFINED** (no in-domain PPL on disk at this step) | — | 3/3 | not evaluable |
| 200 000 | ACCEPT | 0.13173 | 3/3 | YES (the pilot's cell, unchanged) |

> ⚠️ **SUPERSEDED IN PART, 2026-08-12 (`ac70809`) — every `rate_5k` in this table is
> conditional on the 4-point PPL grid `{50k, 100k, 147k, 200k}`, and two entries changed
> when a 150 000 PPL point was measured.** See
> [`A04_STEP150K_PPL_CLOSES_PLATEAU_GRID.md`](A04_STEP150K_PPL_CLOSES_PLATEAU_GRID.md).
> On the 5-point grid: step 150 000 becomes **ACCEPT** (`rate_5k = 0.22602`) and therefore
> **evaluable and disagreeing**, no longer "not evaluable"; and step 200 000's `rate_5k`
> moves **`0.13173 → 0.12607`** because its predecessor interval changes from
> `147k→200k` (d=53 000) to `150k→200k` (d=50 000). **Both values are correct on their own
> grid** — `rate_5k` is a function of the grid, not of the run, so it must always be quoted
> with its grid. No verdict changes (all ≪ T = 2.0) and the earliest disagreement stays at
> **100 000**, since the new point is later and cannot move it earlier.

**Two checkpoints where PLATEAU is UNDEFINED, reported because it limits the bracket.**
`olmo2_ppl_results/` carries the keep7f2 in-domain PPL at steps **{50 000, 100 000, 147 000,
200 000}** only. Step **150 000 has no PPL**, and step **50 000 is the first trajectory point**
(no preceding interval). PLATEAU therefore has no verdict at either, exactly as Pilot Zero ruled
for cpt20k/arm4_peaklr20k, and neither can form a disagreement — a rule evaluation no measurement
supports would be invented. Their NI cells are still reported: they extend the *capability*
trajectory, not the *rule* trajectory. **So the bracket I set out to build is only half-built:
I scored the capability side at 50k/150k, but the PPL side does not exist there, and no amount
of capability scoring fixes that.** Closing the 150k side would need an in-domain PPL run at
step 150 000 (cheap, ~8 shards), which is *not* done here.

### 4.1 Robustness — unanimous across all five null conventions

| convention | intact MMLU residual (pp) | Δ_MMLU (pp) | NI reject at 100k | disagree? |
|---|---:|---:|:--|:--|
| **split** ★ pre-registered | 10.2389 | 1.0239 | 3/3 | **True** |
| first | 10.5754 | 1.0575 | 3/3 | **True** |
| last | 10.4686 | 1.0469 | 3/3 | **True** |
| credit | **−6.6871** | **−0.6687** | 3/3 | **True** |
| wrong | 19.0714 | 1.9071 | 3/3 | **True** |

The disagreement at step 100 000 is **convention-invariant**. Under `credit` the intact model
falls *below* its own MMLU null, so `Δ < 0` and NI degenerates to demanding strict superiority —
flagged `delta_degenerate_negative_margin: true` rather than silently scored, consistent with
guard D1/G1. The verdict does not depend on that axis: TriviaQA and PopQA reject under every
convention (their nulls are convention-free).

### 4.2 RATIO(ρ = 0.85) still AGREES with NI — the disagreement stays PLATEAU-specific

| step | mean retained-accuracy ratio | RATIO accepts? |
|---:|---:|:--|
| 50 000 | 0.321481 | No |
| **100 000** | **0.366854** | **No** |
| 150 000 | 0.392508 | No |
| 200 000 | 0.401674 | No |

A04's claim names **two** incumbent rules. Only PLATEAU disagrees with NI; RATIO rejects at every
checkpoint, i.e. it **agrees** with NI. Pilot Zero's narrowing stands unchanged and must not be
widened to the ratio rule.

---

## 5. Does the "earliest disagreement" claim move? YES — to step 100 000

Over the checkpoints where **both** rules have a verdict (`{100 000, 200 000}`), the earliest
disagreement is now step **100 000**, not step 200 000.

**What may now be said:**

* ✅ "Under the repaired rule R3, the earliest checkpoint at which PLATEAU accepts while NI rejects
  is step **100 000**, and it rejects there on **3/3** decision axes by 7.3–9.5× Δ, under all five
  null conventions."
* ✅ "The relocation §1.4 flagged is **resolved, not merely disclosed**: the cell at R3's own
  earliest accept point is measured and it disagrees."
* ✅ "Between step 100 000 and step 200 000 the arm gains 8.23 → 12.47 % (PopQA), 18.59 → 23.07 %
  (TriviaQA), 32.19 → 39.00 % (MMLU-content) of intact residual — monotone, and nowhere near NI."

**What must NOT be said:**

* ❌ "Step 100 000 is *the* earliest disagreement." It is the earliest **measured** one on a
  4-point PPL grid. Steps 50 000 and 150 000 have no PLATEAU verdict at all (§4), and no checkpoint
  earlier than 100 000 can have one, because step 50 000 is the trajectory's first point.
* ❌ Anything about reading **R1**. R1's earliest accept remains step 200 000 and is untouched.
* ❌ That this changes **K1** (still INDETERMINATE, `b93247f`; the ≥24-cell denominator is
  unaffected — 12 decision cells exist now, up from 3, still < 24) or **K2**.

---

## 6. This does NOT rescue A04 — it hardens the finding that kills the rung

Stated plainly because the temptation runs the other way: **the standing conclusion is unchanged
and these numbers reinforce it.**

`keep7+fresh2` at 1B is a **CONSTANT-REJECT rung**. NI rejects on 4/4 axes at **every** one of the
four checkpoints now measured (50k / 100k / 150k / 200k), by **6.64–9.88× Δ** throughout (min:
MMLU-content @ 200k; max: PopQA @ 50k), with recovery never exceeding **39.00 %** of intact
residual on any axis and as low as **4.78 %** (PopQA @ 50k). Sixteen cells — twelve of them
decision cells — and **zero** NI accepts. A rung where NI can never accept cannot demonstrate
that a rule **discriminates** — only that two rules differ somewhere. That was Pilot Zero's own
`explicitly_NOT_recommended: j=7` finding and it is now quantified across the whole trajectory
rather than at one point.

Equally: the separate standing finding that **keep12 is a constant-reject rung** (NI rejects 4/4,
recovery 22–32 %) and that **K2 not firing is necessary-not-sufficient** is untouched by anything
here. Nothing in this pass moves a kill condition.

**Net effect on A04's claims: exactly one claim changes** — the *location* of the earliest
disagreement, from step 200 000 to step 100 000. Every verdict, every kill condition, and every
gate status is unchanged.

---

## 7. Reproduce

```bash
# 1. GPU: score the three checkpoints (8×H20 on a zwfy6 node; ~162 s per ckpt)
PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
bash proposal/active/A04-recovery-certification/code/a04_step100k_axes_driver.sh

# 2. CPU: the PLATEAU-vs-NI analysis (seconds; hard-fails if the archived
#    step-200000 cells do not reproduce to 1e-9 pp)
python proposal/active/A04-recovery-certification/code/a04_step100k_plateau_vs_ni.py \
  --raw_root        /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
  --ppl_json        proposal/active/A04-recovery-certification/evidence/a04_1b_keep7f2_ppl_trajectory.json \
  --repair_json     proposal/active/A04-recovery-certification/evidence/a04_plateau_rule_repair.json \
  --pilot_zero_json proposal/active/A04-recovery-certification/evidence/pilot_zero_rule_disagreement.json \
  --out_json        proposal/active/A04-recovery-certification/evidence/a04_step100k_plateau_vs_ni.json
```

Machine-readable output: `evidence/a04_step100k_plateau_vs_ni.json`.
Per-example dumps: `zwfy6:olmo2_{mmlu_content,closedbook}_results/A04_1B_keep7f2_step{50000,100000,150000}[_nq]/`.
Driver log: `zwfy6:logs/a04_step100k_axes_progress.log`.

## 8. What this document does NOT establish

1. **A PLATEAU verdict at step 150 000 or 50 000.** No in-domain PPL exists at 150 000; 50 000 is
   the trajectory's first point. The intended bracket around the accept boundary is therefore
   only half-built. An in-domain PPL run at step 150 000 would close the upper side.
2. **Anything about K1's ≥24-cell clause** (12 decision cells now, still short) **or K2.**
3. **Causality of the D5 48-item drift.** Still needs a same-code control; untouched here.
4. **That a rung exists where NI can accept.** Still unmeasured at 1B, and this pass makes the
   keep7 rung's unsuitability more certain, not less.
