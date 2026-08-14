# A04 — the shallow-rung ladder: does NI ever accept a *damaged* 1B model?

**Verdict string:** `NI_ACCEPT_REGION_AT_1B_CONTAINS_NO_DAMAGED_RUNG -- both new rungs are constant-REJECT down to a 12.5% cut, the lightest the family admits`

**Branch:** **B** — selected by the pre-registered rule in `A04_SHALLOW_RUNG_LADDER_PREREG.md` §4.1, not chosen after the fact. All three branches (A/B/C) were written before any number existed.

**Date:** 2026-08-13 · **GPU: 73.6 GPU-h** (training 72.3 + eval 1.32; this analysis is **0 GPU**, CPU-only and read-only on every input).
**Training nodes:** `.73` (keep14) and `.82` (keep13), 8×H20 each, zwfy6. **Node of record for every statistic:** `TENCENT64.site` (numpy **2.5.1**, python 3.14.6).
**Not touched:** `LOCAL` / `.21` (SparseForge #246), `.104` (paperC Qwen3-8B heal). Both the launcher and the analysis refuse those nodes **by IP**.
**Pre-registration:** `A04_SHALLOW_RUNG_LADDER_PREREG.md`, commit `a2e1a95`, committed **before the first margin existed** — both runs ~20 min into 5000 steps; no ckpt, no eval dir, no accuracy, no margin existed.
**Evidence:** `evidence/a04_shallow_rung_ladder.json` (sha256 `39322b964418fe02d3a7c4f4327f15a39d42e6ad236b27497310529e60011783`)
**§2.0.2 disposition companion:** `evidence/a04_shallow_ladder_neighbour_disposition.json` — read-only on the file above (whose sha256 `STATUS.json` pins), so the pre-registered analysis output is never rewritten to bolt on a disclosure.
**Code:** `code/a04_shallow_rung_ladder_ni.py`, `code/a04_shallow_ladder_eval_driver.sh`, `code/a04_shallow_ladder_chain.sh`, `scripts/_run_a04_shallow_ladder.sh`, `code/a04_shallow_ladder_neighbour_disposition.py`, `code/_a04_shallow_integrity_probe.py`

> **Every number in this document is rendered from the evidence JSON by `code/a04_render_shallow_ladder_verdict.py`.** Nothing is hand-transcribed. There were five hand-transcription slips in this proposal on 2026-08-13 alone; the fix adopted then was to delete the transcription step, and this document is that fix applied to prose.

---

## 0. The question, and why nothing on disk could answer it

`STATUS.json:pilot_one.pilot_two_status`, verbatim:

> **BLOCKED.** 1,077–4,309 GPU-h must not be committed until a NEW pre-data doc shows a rung exists where NI can be **OBSERVED TO ACCEPT**; otherwise the gate can only ever confirm rejection.

Same key: *"it is a **rung-selection problem, not a variance problem**."*

NI's discrimination curve had an **empty gap**. Damaged arms cluster at 11–63 % recovery and reject by tens of SE — 1B `keep12+fresh2` rejects on 4/4 axes by **27.0–90.4 × `sd_run`** at 22–32 % recovery. The **only** NI accept in all of A04 is `full32_dolmino`, which has **zero structural damage**. `keep12` was the lightest damaged 1B rung in existence, and shallower rungs had **0 checkpoints on either disk** (independently re-verified for this pass: `outputs/olmo2_probe2_7B_keep16fresh2/` holds only `arch_meta.json` on zwfy6 and does not exist on wzc1; no 1B `keep13`–`keep20` directory existed on either disk before today).

So the blocker was **not dischargeable by any re-analysis**. It required the two lightest damaged rungs the family admits, which is what this pass trained.

## 1. The two new arms

Protocol is **Pilot One Stage B, verbatim** — every hyper-parameter read out of `stageB_seed101/step5000.pt`'s own `train_args` dict, with **only** `keep_front_layers` changed. Same seed (101), same corpus (`dolmino_now15b.npy`, 126,907,244,672 B on zwfy6 — asserted, because wzc1's same-named file is a **different corpus**), same 5,000 steps, same uniform LR 2e-5, same eff_bs 128, both **post-`ce5c298`**.

| arm | `keep_front` | `n_fresh` | depth | cut | recovery-space position |
|---|---|---|---|---|---|
| `keep14+fresh2` | 14 | 2 | 16 | 2/16 = n/a | **new** |
| `keep13+fresh2` | 13 | 2 | 15 | 3/16 = n/a | **new** |
| `keep12+fresh2` *(Stage B reference)* | 12 | 2 | 14 | 4/16 = n/a | published |

**`keep14+fresh2` has depth 16 = the base's depth, and is still DAMAGED.** Base layers 14 and 15 are **discarded** and replaced by random-init Olmo2 layers, so 14 of 16 pretrained layers are inherited. The zero-damage control is `n_fresh_layers=0` continued-pretraining (the `full32` construction) and is a **different** arm. Reporting `keep14+fresh2` as zero-damage would be a category error, and using a continued-pretrained arm as the **anchor** is forbidden by guard G2 (§4).

**`keep15+fresh2` is why `keep14` is the boundary:** it would be 17 layers, **deeper than the 16-layer base**, so it is not a cut of the base at all and "recovery from damage" would have no referent.

## 2. GATE0 — no degeneracy at `keep_front + n_fresh == base depth`

Run **before** any 8-GPU commitment (1 GPU, 20 steps, `/tmp` output, 18:27–18:30), because a special trainer branch at that boundary would have invalidated the whole design.

| probe | tensors copied | expected `3+11·keep` | fresh ids | `max｜model−base｜` | fresh norms all-ones | fresh `q_proj` std | reached |
|---|---|---|---|---|---|---|---|
| keep14+fresh2 | **157** | 157 ✓ | `[14, 15]` | `0.000e+00` | True / True | 0.020001 | step 20, exit 0 |
| keep13+fresh2 | **146** | 146 ✓ | `[13, 14]` | `0.000e+00` | True / True | 0.019997 | step 20, exit 0 |

All 6 trainer asserts pass on both, and the live 8-GPU runs reproduce them (`keep14`: *copied 157 tensors … fresh tail layer-ids [14, 15] … ALL 6 CHECKS PASS*). **Source reading confirms why:** `transplant_front()` selects base keys by `lid < keep_front_layers` against the **base** state dict, and the expected fresh set is `range(keep, keep+n_fresh)` on the **new** cfg. There is **no branch** for `keep+fresh == base_layers`; the only conditional is `if n_fresh_layers > 0`, which skips the fresh-init assert for the `n_fresh=0` CPT control. Both arms have `n_fresh=2`.

**Optimizer groups observed** (so no differential-LR claim can be retrofitted): `keep14` → fresh 339.7 M / inherited 1145.0 M / 0.1 M; `keep13` → 339.7 M / 1077.9 M / 0.1 M — **all at 2.00e-05**. Uniform LR, as in Stage B.

## 3. Results

### 3.1 The intact anchor and Δ (convention `split`)

| axis | null | intact | residual(intact) | **Δ = 0.10 × residual** |
|---|---:|---:|---:|---:|
| `triviaqa` | 0.2564 | 40.6877 | 40.4313 | **4.0431** |
| `popqa` | 2.2920 | 15.4973 | 13.2053 | **1.3205** |
| `mmlu_content` | 28.4450 | 38.6839 | 10.2389 | **1.0239** |
| `nq_open` | 0.5540 | 10.2493 | 9.6953 | **0.9695** |

Δ was **built at runtime** by calling the imported `build_nulls()` on the G0-pinned anchor and then **cross-checked** against the canonical full-precision constants: max |diff| = **2.220e-16** pp, tolerance 1e-09. **Δ is never substituted** (guard G2).

### 3.2 NI margins — `margin_pp = lower95(diff) + Δ`; **> 0 means ACCEPT**

| arm | `triviaqa` | `popqa` | `mmlu_content` | `nq_open` | decision axes accepting |
|---|---:|---:|---:|---:|:--:|
| `keep14f2_step5000` | -16.5543 | -7.1255 | -3.5623 | -5.4848 | **0/3** |
| `keep13f2_step5000` | -25.3316 | -8.4993 | -5.2288 | -6.8698 | **0/3** |
| `keep12f2_step5000_REF` | -28.0010 | -9.0534 | -6.6531 | -7.3961 | **0/3** |

`nq_open` is **DEMOTED** by design §5.2 (its item-level 95 % CI half-width already exceeds its own Δ at n=3610) and carries **zero decision weight**; it is shown for completeness only.

### 3.3 How far each margin is from flipping (item bootstrap SE)

| arm | `triviaqa` | `popqa` | `mmlu_content` | `nq_open` |
|---|---:|---:|---:|---:|
| `keep14f2_step5000` | 47.9 SE | 28.3 SE | 11.8 SE | 11.2 SE |
| `keep13f2_step5000` | 70.5 SE | 31.7 SE | 15.1 SE | 14.1 SE |
| `keep12f2_step5000_REF` | 75.1 SE | 32.7 SE | 17.9 SE | 14.6 SE |

This is the **item-sample** SE only. It is **not** `sd_run` and says nothing about seed variance — one seed (101) per arm, so no `sd_run` is computable here.

### 3.4 Recovered fraction of the intact calibrated residual

| arm | cut | `triviaqa` | `popqa` | `mmlu_content` | `nq_open` |
|---|---:|---:|---:|---:|---:|
| `keep14` | n/a | 50.46% | 39.17% | 60.08% | 41.71% |
| `keep13` | n/a | 28.81% | 28.98% | 44.50% | 27.43% |
| `keep12` | n/a | 22.26% | 24.89% | 31.00% | 22.29% |

**No recovery fraction here may be read as "inheritance is worth X".** No 1B zero-inheritance floor (`--from_scratch` or `--random_trunk`) exists on either disk — verified for this pass — so these are fractions of the intact residual **only**. At 7B the zero-inheritance floor already reaches **32.6 / 11.6 / 40.5 / 28.9 %** of the intact residual (`control_arms_ni_20260813` Q2), i.e. a large part of what looks like "recovery" is work random init already does. `must_not_claim` item 28.

## 4. The verdict, and what it does to the blocker

* **`keep14f2_step5000`** → `NI_CONSTANT_REJECT` (0/3 decision axes accept; axes accepting: none). Identical under **all five** MMLU tie conventions: **True**.
* **`keep13f2_step5000`** → `NI_CONSTANT_REJECT` (0/3 decision axes accept; axes accepting: none). Identical under **all five** MMLU tie conventions: **True**.
* **`keep12f2_step5000_REF`** → `NI_CONSTANT_REJECT` (0/3 decision axes accept; axes accepting: none). Identical under **all five** MMLU tie conventions: **True**.

### → BRANCH B

**both constant-REJECT -> NI's accept region at 1B contains no damaged rung; negative but publishable verdict on the certification rule; pilot_two_status stays BLOCKED and the blocker is recorded as undischargeable by rung selection at 1B**

**This is a negative result and it is reported as one.** It is not dressed as a success and it is not a failure of the experiment: it is a measured fact about the certification rule A04 is trying to write.

What it establishes: **NI's accept region at 1B contains no damaged rung at all**, down to a cut of 2 of 16 layers — the lightest cut the family admits. The accept region is therefore bounded to lie strictly between "discard 2 of 16 layers" and "discard none". A rule with that property distinguishes **damaged from intact**, not **recovered from unrecovered** — and the latter is what a recovery-certification rule is for.

`pilot_two_status` **stays BLOCKED**, and the blocker is now recorded as **undischargeable by rung selection at 1B**: there is no shallower damaged rung left to try. Escaping it needs a different Δ, a different decision metric, or far more heal tokens — and A03 already showed 10× the token budget (52.43 B tokens at `keep7`) does not close the gap.

### 4.1 `A04_GATE_DESIGN.md` §2.0.2 — the neighbour precondition, and why it is **not triggered**

§2.0.2 **does bind this ladder** (`binds_this_ladder = True`): §2.0.2 is scoped to 'any NI(Delta) accept reported by this gate', and this ladder's accepts would come from the same IMPORTED ni_rule under the same frozen Delta. Prereg §5.5 renounces a CLAIM; §2.0.2 imposes a DISCLOSURE duty, and renouncing a claim cannot discharge a duty. That reading was fixed **pre-data** in `A04_SHALLOW_LADDER_NEIGHBOUR_ADMISSIBILITY.md` (commit `46ea84d`) — committed 2026-08-13 14:06:10 UTC with keep14 at step 3740-3800/5000 and keep13 at 3900-3960/5000, no step5000.pt, no A04_1B_shallow_* eval dir and no analysis JSON on either disk.

But §2.0.2 **gates accepts only**, and this pass has **0** accepting decision-axis cells, so `TRIGGERED = False`:

> NOT_TRIGGERED -- no cell accepts on any decision axis. Per the reconciliation document §5 (fixed pre-data, phrasing adopted verbatim from the full32 pass): 'under Branch B (both arms constant-REJECT) §2.0.2 is NOT triggered at all -- it gates accepts only ... The precondition is not vacuously satisfied -- it is not triggered.' Branch B needs nothing from that document.

**The precondition is NOT vacuously satisfied — it is not triggered.** The distinction matters: a vacuous satisfaction would let a later reader infer that a neighbour check was passed. None was run, because none was owed.

`CERTIFIED` was in any case **structurally unreachable** for this ladder, decided pre-data: §2.0.2 conditions a certified reading on the immediately adjacent saved checkpoints on BOTH sides. No upper neighbour of step5000 can exist (max_steps=5000; final.pt is the SAME step). Decided pre-data in the admissibility document §2.3 -- no datum can change it.

**The lower neighbour exists on disk and was NOT scored.** Per the admissibility document §6.5 that is the weaker of the two available disclosures and it is the honest one; **which one applies is decided by whether the eval was run, not by what the step5000 numbers turned out to be** — and under Branch B it is not owed at all.

| arm | lower neighbour `step2500.pt` | scored? | upper neighbour | `n_neighbours_present` |
|---|---|:--:|---|---:|
| `keep13` | exists, 17,013,823,232 B | **no** | **cannot exist** (`final.pt` is the *same* step) | 1 |
| `keep14` | exists, 17,819,242,212 B | **no** | **cannot exist** (`final.pt` is the *same* step) | 1 |

**2500 steps is NOT a neighbourhood** (`True`): 2500 steps is 50 % of this run's 5000-step horizon and spans LR 1.143706e-05 -> 2.000000e-06, a factor 5.7185 (half a cosine of annealed training). Admissibility §3 answers this question against us: the k=2 gate is NECESSARY-but-not-sufficient and a move that clears it must NOT be read as instability.

Forbidden by name, so the temptation is closed in writing:
* comparing a 2500-step move to the 500-step keep8/keep10 ranges (7B, different spacing, different scale, different corpus)
* writing 'wider spacing is more convincing' -- wider spacing makes a neighbour check WEAKER because it admits training progress as an explanation (banned by name in admissibility §3 consequence 4)

`grep -c resume` on both trainer logs returns 0: each arm is a single uninterrupted process, so no range here straddles a resume seam (§2.0.2 trap 2).

## 5. The 1B depth ladder, now four points instead of two

| rung | cut | depth | verdict | recovery `triviaqa` | recovery `popqa` | recovery `mmlu_content` |
|---|---:|---:|:--:|---:|---:|---:|
| `keep14` | 12.50% | 16 | CONSTANT_REJECT | 50.46% | 39.17% | 60.08% |
| `keep13` | 18.75% | 15 | CONSTANT_REJECT | 28.81% | 28.98% | 44.50% |
| `keep12` | 25.00% | 14 | CONSTANT_REJECT | 22.26% | 24.89% | 31.00% |
| *zero damage* (`full32`, **7B**) | 0.00% | 32 | **ACCEPT 1/3** | — | — | 97.7% |

The `full32` row is **7B and is not comparable as a matched experiment** — it is shown only to locate the one accept A04 has. keep12/13/14 share corpus, step count, protocol, LR, eff_bs and SEED (101) and are all post-ce5c298, so they ARE mutually comparable as a 1B depth ladder. They are NOT comparable to the 7B ladder (STATUS.json:warning's two-corpora confound) and 5000 steps is not a converged heal.

**Monotonicity is DESCRIPTIVE ONLY.** 3 points give 2 differences. 'Non-monotone' on 2 differences means ONE sign flip and cannot be distinguished from noise without a per-point sigma_run, which does not exist here (one seed per arm). NOT a trend fit and NOT decision-bearing.

| axis | recovered fraction across rungs | successive diffs | all same sign | sign reversals |
|---|---|---|:--:|---:|
| `triviaqa` | keep12=22.26%, keep13=28.81%, keep14=50.46% | +6.55pp, +21.65pp | True | 0 |
| `popqa` | keep12=24.89%, keep13=28.98%, keep14=39.17% | +4.09pp, +10.19pp | True | 0 |
| `mmlu_content` | keep12=31.00%, keep13=44.50%, keep14=60.08% | +13.49pp, +15.58pp | True | 0 |
| `nq_open` | keep12=22.29%, keep13=27.43%, keep14=41.71% | +5.14pp, +14.29pp | True | 0 |

### 5.1 Every range with **its own** noise floor and **its own** *k*

`E[range of k iid N(0,σ)]/σ` is **k-dependent**. E[range of k]/sigma is k-dependent. c_3 at k=8 makes a floor 40.6 % TOO LOW and manufactured a finding once; c_3 at k=2 inflates a floor by 50.0 % and can suppress a real move. Each range above records its own k, its own constant, AND what the wrong constant would have done.

| k | constant | closed form | used for |
|---:|---|---|---|
| 2 | `1.1283791670955126` | `2/sqrt(pi)` | the k=2 adjacent pair |
| 3 | `1.6925687506432689` | `3/sqrt(pi)` | the k=3 3-rung ladder |
| 8 | `2.8477` | Monte Carlo (no closed form) | **recorded, unused** |

σ is the **mean of the participating cells' own `bootstrap_se_pp`** — the per-cell recipe that reproduces `A04_GATE_DESIGN.md` §2.0.2's worked example exactly. **The pooled variant is the one `PROPOSAL.md` §4.3 retracted as "1.69× off". It is not used.**

**`ladder_k3_margin_range_keep12_keep13_keep14`** — 3-rung 1B depth-ladder margin range (keep12/keep13/keep14), **k=3** (`k_matches_n_cells = True`), c_k = `1.6925687506432689`

| axis | range | its floor (c_k·σ) | range/floor | clears its floor? | floor if the **wrong** c_k had been used |
|---|---:|---:|---:|:--:|---|
| `triviaqa` | 11.4467 pp | 0.6079 pp | 18.831× | **YES** | `c_2` → 0.4052 pp (-33.3%, would clear=True) |
| `popqa` | 1.9279 pp | 0.4497 pp | 4.287× | **YES** | `c_2` → 0.2998 pp (-33.3%, would clear=True) |
| `mmlu_content` | 3.0907 pp | 0.5765 pp | 5.361× | **YES** | `c_2` → 0.3843 pp (-33.3%, would clear=True) |
| `nq_open` | 1.9114 pp | 0.8361 pp | 2.286× | **YES** | `c_2` → 0.5574 pp (-33.3%, would clear=True) |

**NOT decision-bearing.** prereg §5.3 / §5.5: the 3-rung spread is across arms of DIFFERENT DEPTH -- it is the quantity the design VARIES, not a repeat measurement, so it is not a neighbour range and not a sigma_run. The verdict is decided by per-cell NI margins (all 12 decision-axis cells reject), never by this range.

`is_a_neighbour_range = False`, `is_a_sigma_run = False`. sigma here is the ITEM-sample bootstrap SE. ONE seed (101) per arm, so no sd_run exists at these rungs and no range here may be read as resolved against run-to-run variance.

**`adjacent_pair_k2_margin_range_keep13_keep14`** — adjacent-rung margin range (keep13 vs keep14), **k=2** (`k_matches_n_cells = True`), c_k = `1.1283791670955126`

| axis | range | its floor (c_k·σ) | range/floor | clears its floor? | floor if the **wrong** c_k had been used |
|---|---:|---:|---:|:--:|---|
| `triviaqa` | 8.7773 pp | 0.3976 pp | 22.076× | **YES** | `c_3` → 0.5964 pp (+50.0%, would clear=True) |
| `popqa` | 1.3738 pp | 0.2933 pp | 4.684× | **YES** | `c_3` → 0.4400 pp (+50.0%, would clear=True) |
| `mmlu_content` | 1.6664 pp | 0.3664 pp | 4.548× | **YES** | `c_3` → 0.5496 pp (+50.0%, would clear=True) |
| `nq_open` | 1.3850 pp | 0.5511 pp | 2.513× | **YES** | `c_3` → 0.8266 pp (+50.0%, would clear=True) |

**NOT decision-bearing.** prereg §5.7 makes the PAIRED item bootstrap the test of record for a keep13-vs-keep14 difference (see the paired CI table), not a range-vs-floor heuristic. This k=2 block is a disclosure so that the floor is stated with the constant that matches k=2, and so that 1.6926 can never be silently reused here.

`is_a_neighbour_range = False`, `is_a_sigma_run = False`. sigma here is the ITEM-sample bootstrap SE. ONE seed (101) per arm, so no sd_run exists at these rungs and no range here may be read as resolved against run-to-run variance.

**no ratio of two ranges appears anywhere in this pass. A ratio of two ranges neither of which clears its own floor is UNDEFINED, not a direction -- the error that voided within_arm_lr_refutation_20260813.**

prereg §4.2 / §5.5. The verdict is the per-cell NI margin table; every range above is a disclosure carrying its own floor.

## 6. Are the rungs even distinguishable from each other?

Paired item bootstrap on the **same** item set (alignment asserted). A difference whose CI straddles 0 is **UNRESOLVED — not "a direction"**.

**`keep14_minus_keep13`**

| axis | diff | CI95 | boot p | resolved |
|---|---:|---|---:|:--:|
| `triviaqa` | +8.7550 pp | [+8.2535, +9.2454] | 0.0001 | **yes** |
| `popqa` | +1.3458 pp | [+1.0093, +1.6822] | 0.0001 | **yes** |
| `mmlu_content` | +1.5952 pp | [+1.0682, +2.1295] | 0.0001 | **yes** |
| `nq_open` | +1.3850 pp | [+0.7756, +1.9945] | 0.0001 | **yes** |

**`keep12_minus_keep13`**

| axis | diff | CI95 | boot p | resolved |
|---|---:|---|---:|:--:|
| `triviaqa` | -2.6471 pp | [-3.0318, -2.2570] | 0.0001 | **yes** |
| `popqa` | -0.5397 pp | [-0.8411, -0.2453] | 0.0001 | **yes** |
| `mmlu_content` | -1.3816 pp | [-1.9299, -0.8332] | 0.0001 | **yes** |
| `nq_open` | -0.4986 pp | [-1.0249, +0.0277] | 0.0774 | no |

## 7. `RATIO(ρ=0.85)` — the rule-disagreement ledger

| arm | mean ratio | ρ = 0.85 | RATIO | NI | disagree? |
|---|---:|---:|:--:|:--:|:--:|
| `keep14f2_step5000` | 0.5831 | 0.85 | REJECT | REJECT | no |
| `keep13f2_step5000` | 0.4635 | 0.85 | REJECT | REJECT | no |
| `keep12f2_step5000_REF` | 0.4174 | 0.85 | REJECT | REJECT | no |

**0** rule-disagreement cell(s) from this pass: none.

## 8. Verification — every item below is an executed assertion

* **keep12 reproduction gate** — this pipeline reproduces `keep12` seed101's **own published** accuracy on all four axes to **0.000e+00 pp** (tolerance 1e-09). The canonical values are **read at runtime** from `evidence/pilot_one_stage_b_s3_verdict.json`, never transcribed. Without this the new rungs would not be measured on the same instrument as the arm they are compared against.
* **guard G2 — anchor must be VANILLA** — **executed**, two independent ways: the anchor tag is checked against a list of CPT/pruning markers, **and** the anchor cell's own `summary.json` meta must report `mode=base`, `num_hidden_layers=16` and **no `ckpt`**. Why it matters: at 7B, `full32_step25000` scores *below* vanilla base on all four axes, so substituting it would shrink every Δ **and** lower every target = **manufactured accepts**.
* **shard integrity** — every (arm × axis) cell: shard index **SET** exactly `{0..7}` (a set, not "8 files"), merged *n* exactly `EXPECTED_N` (17944 / 14267 / 3610 / 14042), **0** duplicate `item_id`, **0** nan in the metric vector, and `item_id` sequences **identical across every arm AND the anchor** — without which the paired differences would compare different items.
* **arm architecture verified from eval meta** — each arm's eval `summary.json` meta must report `keep_front` / `n_fresh` / `num_hidden_layers` matching its tag or the analysis **aborts**. An eval that rebuilt the wrong shell would otherwise be scored silently. The eval loader additionally reads keep/fresh from the **ckpt's own meta** and raises if the CLI disagrees, then `strict`-loads.
* **protocol** — `add_bos` asserted **`is False`** — never `is not True`, so `None` or missing **FAILS**. `chat_template` asserted `is not False` → FAIL, **plus structurally**: neither eval script contains an `apply_chat_template` call site, so no flag can enable one. `max_new_tokens == 32`. `mmlu_bs=16`, `cb_bs=32` — the Stage B driver's own values, so the new cells are protocol-identical to `keep12`. These are BASE LMs (no SFT/RL); any chat=True number is void.
* **canonical code imported, never reimplemented** — `ni_rule` / `ratio_rule` / `load_shards` / `build_nulls` / `mmlu_content_norm_vec` / `qa_metric_vec` / `EXPECTED_N` / `AXES` / `DEMOTED_AXES` / `PREREG` from `pilot_zero_rule_disagreement`; `assert_aligned` / `d4_interface_degenerate` from `a04_shallow_rung_ni_7b`; `paired_bootstrap` / `TIE_CONVS` / `N_BOOT` / `SEED` from A03's `analyze_1b_knowledge_floor`. **The null is never hand-computed** — MAIN's own subtraction of a recorded null was ~0.5 pp off twice.
* **bootstrap seed disjointness** — `arm_index` 1100/1101/1102, guard offset `SEED+9700+13·axis`. Disjoint from every archived block (0–1, 100–102, 200–204, 300–301, 400–408, 500–503, 600–610, 700–702, 800–801, 900–902, 1000–1005). The check is **EXECUTED** by `assert_seeds_disjoint` (reads each archive's own recorded offsets and raises on intersection): **8 archives scanned, no clash**. Prose claims of disjointness in this repo have already been wrong once, and the executed check has caught a real collision.
* **one node for every statistic** — numpy `Generator.multinomial` differs in **19/10000** rows between `.82`'s 2.4.6 and `.73`'s 2.5.1. Every statistic here comes from **TENCENT64.site** (numpy 2.5.1), pinned with `--expect_numpy`. Training is unaffected; only bootstrap statistics are.
* **gate constants self-tested but DECLARED UNUSED** — `E[range of k iid N(0,σ)]/σ` is **k-dependent**: k=2 → 1.1283791670955126, k=3 → 1.6925687506432689 (both closed form, **re-derived not trusted**), k=8 → 2.8477 (Monte Carlo, validated by reproducing the k=3 closed form to 2.87e-03). Reusing k=3's constant at k=8 makes a floor **40.6 % too low**. **None is used here** (one seed per arm, 2 checkpoints per arm) and they are recorded as `DECLARED_UNUSED` so nobody can reuse a wrong `c_k` from this document. **A ratio of two ranges neither of which clears its own floor is UNDEFINED, not a direction** — the error that voided `within_arm_lr_refutation_20260813`.
* **pipeline validated BEFORE the arms landed** — `--preflight_only` ran every guard, the anchor build, the Δ cross-check and the `keep12` reproduction gate **while both trainings were in flight**, writing nothing; `keep12` seed101 reproduced to **0.000e+00 pp**. `--preflight_ignore_own_training` (needed because our own training held the cards) is **hard-refused outside preflight mode**, so the GPU refuse-guard can never be bypassed by a run that writes an evidence file.
* **zwfy6's evidence dir was incomplete and was repaired first** — the first preflight scanned only **7** offset ledgers on zwfy6 vs **8** on wzc1 — **14 evidence files existed only on wzc1**, so the disjointness check was running against a **partial** archive set and could have missed a real collision. All 14 were `scp -O`'d and **md5-verified 14/14** before the analysis ran. Generalisable: the two disks' `proposal/` trees are not automatically in sync (zwfy6's is a hand copy, not a git checkout), and `assert_seeds_disjoint` must be pointed at a **complete** evidence dir.
* **positive preflight assertions printed before launch** — both progress logs carry, **before** the launch line: `PREFLIGHT-ASSERT trainer post-ce5c298: 869: sampler = DistributedSampler(ds, shuffle=True, seed=args.seed)`, the trainer md5 `284b286f90b526e4e8ad93a68e2a3b16`, `base num_hidden_layers=16`, the exact dolmino byte count, and `GPUs clear (0MiB held)`. Both arms are **post-fix**, the same side of the `ce5c298` break as the Stage B family (`PROPOSAL.md` §7.2).

## 9. Cost

| item | value |
|---|---:|
| keep13 training | 3.160 s/step → 4.39 h wall → **35.1 GPU-h** |
| keep14 training | 3.348 s/step → 4.65 h wall → **37.2 GPU-h** |
| training total | **72.3 GPU-h** |
| eval total | **1.32 GPU-h** |
| this analysis | **0 GPU-h** (CPU-only, read-only) |
| **grand total** | **73.6 GPU-h** |
| Pilot Two, for comparison | 1,077–4,309 GPU-h |

Wall time per arm is **measured** from each arm's own trainer log as `(t_last − t_first) / (step_last − step_first)` over the whole run — elapsed/iter, **not** an instantaneous `s/step` sample. This pass is **1.7–6.8 %** of Pilot Two, and it is the only expenditure that could decide whether Pilot Two's blocker is dischargeable at all.

## 10. What this pass does NOT license

* any sigma_run, seed-variance or K2 statement -- ONE seed (101) per arm; a sigma over arms of DIFFERENT depth is not a run-to-run sigma
* any PLATEAU(T) comparison -- no in-domain val PPL trajectory was produced for either new arm, so the NI-vs-PLATEAU disagreement cannot be evaluated
* any trajectory / monotonicity / neighbour-range claim -- save_every 2500 gives 2 checkpoints per arm; 2 points = 1 difference
* any differential-LR claim -- all four optimizer groups ran at 2.00e-05 (GATE0-measured, prereg s3.2)
* treating keep14+fresh2 as a zero-damage control -- depth 16 == base depth, but base layers 14-15 are DISCARDED and random-re-initialised; the zero-damage control is n_fresh_layers=0 CPT (full32-style)
* comparing these 1B rungs to the 7B ladder as a matched experiment (STATUS.json:warning two-corpora / unequal-steps confound)
* any recovery FRACTION read as a clean 'inheritance is worth X' -- no 1B zero-inheritance floor (--from_scratch or --random_trunk) exists on either disk, so the fractions here are fractions of the intact residual ONLY (must_not_claim item 28)
* a format-free reading of triviaqa / popqa / nq_open -- A04 has two demonstrations that generative EM partly measures output length (PROPOSAL.md 4.4: 47.37% of an EM loss; control_arms P3: 50.00% of an EM gain, which REORDERED two arms). mmlu_content is length-free by construction
* any claim that a keep13-vs-keep14 margin difference is 'measured' unless its paired CI excludes 0 (section 9)
* quoting any margin finer than 0.01 pp across nodes (numpy multinomial drift)

