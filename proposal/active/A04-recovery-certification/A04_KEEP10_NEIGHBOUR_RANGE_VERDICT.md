# A04 — does the neighbour range replicate on a second arm? (keep10+fresh2, 500-step triple)

**Verdict string:** `KEEP10_NEIGHBOUR_RANGE_REPLICATES_MATERIAL_TRIVIAQA`

**Date:** 2026-08-13 · **GPU:** **3.4911 GPU-h** (driver wall 13:43:16 → 14:09:27 =
1571 s × 8 GPUs; per-checkpoint 509 / 522 / 527 s). Analysis is CPU-only.
**Node:** `.82` only (8×H20, zwfy6, **numpy 2.4.6**), verified 8×0 MiB / 0 % / no
compute processes before launch; the driver refuses to start if > 8000 MiB is held
**and** refuses to run on a node whose numpy ≠ 2.4.6.
**Not touched:** `.73` (keep12 11-ckpt trajectory, still running), `.104` (paperC
Qwen3 heal), `LOCAL`/`.21` (SparseForge #246).
**Pre-registration:** `A04_KEEP10_NEIGHBOUR_RANGE_PREREG.md`, committed as its own
commit (`0e889cd`) **before the first checkpoint was scored**.
**Evidence:** `evidence/a04_keep10_neighbour_range.json`
(sha256 `0838a67e…`, byte-identical on both disks)
**Code:** `code/a04_keep10_neighbour_range_driver.sh`, `code/a04_keep10_neighbour_range.py`

---

## 0. Answers, in one block

| Q | Answer |
|---|---|
| **Q1** — keep10 per-axis range, gated on `1.6926·σ` | **triviaqa 1.2149 pp — CLEARS (1.84× floor).** popqa 0.3151 (0.54×, fails), mmlu_content 0.1852 (0.29×, fails), *nq_open 0.7479 (0.67×, fails, demoted)*. **1 of 3 decision axes clears.** |
| **Q2** — does keep8's 1.1202 pp triviaqa range replicate? | **YES, in magnitude — 1.2149 pp, 1.08× keep8's.** Same axis, same gate verdict, same side of 0.5 pp, on a differently-damaged arm. **But the SHAPE differs**: keep8 was a monotone terminal drop, keep10 is a **V** (−1.21 pp then **+0.72 pp**, both resolved at p = 0.0001). So *"one adjacent 500-step interval can move triviaqa by a pp"* is arm-independent; *"the last checkpoint is the worst one"* is not. |
| **Q3** — Heineman et al. rel.std | keep10 triviaqa **0.0349** vs their intact OLMo-2 7B-4T **0.003** = **11.65× raw**. But **the raw ratio is inflated by the denominator**: our arm sits at 17.42 % accuracy vs their ~0.6–0.7, and rel.std divides by the mean. Decomposed, the **absolute-SD ratio is 2.9–5.1×** across a plausible grid for their mean. On mmlu_content we are **9–17× QUIETER** than their MMLU (0.0026 vs 0.023). **n=3 vs n=30, damaged vs intact, zero-shot vs few-shot, base protocol vs OLMES — hypothesis only, not a result.** |

**The one-line scientific summary:** the neighbour-range loophole **is real on a
second arm and is worth about the same amount (~1.2 pp on triviaqa)**, so
`A04_GATE_DESIGN.md` §2.5's tolerance is no longer a one-arm number. But the
*mechanism* did **not** replicate — keep10's excursion **recovers**, which means
"average the neighbours" would fix keep10 and would **not** have fixed keep8, and
the two arms therefore support the **precondition** without supporting any single
story about *why*.

---

## 1. What I checked before spending GPU, including the trap that killed keep8's cluster 1

### 1.1 ✅ NO RESUME SEAM — cleared **in advance**, not discovered afterwards

keep8's cluster 1 was retroactively demoted because 124000/124500/125000 straddled
a process boundary. The dispatch asked for the same triple to be checked here, and
it is clean:

| ckpt | written by | when |
|---|---|---|
| — | `[resume] loading ckpt … step86500.pt`, `has_optimizer=True`, 135 tensors restored, `continue @ step=86500 epoch=0 warmup=150 max_steps=200000` | 2026-08-12 **03:57:09** |
| step89000 | same `.73` process (log line 154) | 08:44:52 |
| step89500 | same process (line 182) | 09:42:19 |
| step90000 | same process (line 210) | 10:39:43 |
| — | process dies, TCPStore/NCCL heartbeat — **after all three saves** | 11:15 |

`grep -c '\[resume\] loading ckpt'` on `logs/olmo2_7B_keep10fresh2_resume200k_73.log`
is **1**. One process, one loader, continuous data order. This is a **clean
500-step neighbourhood**, directly comparable to keep8's cluster 2.

### 1.2 ✅ Three genuinely distinct checkpoints — byte size proves nothing here

All three files are **exactly 39,009,621,855 B**. That is not identity: keep8's
130000/130500/131000 also shared one size, and `shortgpt16/step128000.pt` was a
7.7 GB **truncated** write that `ls -l` could not distinguish from healthy. So
identity was proven:

| | f64 sum of all params | `lm_head` sha256 | `embed_tokens` | `layers.0.q_proj` |
|---|---|---|---|---|
| step89000 | 4.3489697564e+04 | `fc500facfd5de7b9` | `45b40ec395f28011` | `ab4e5befcaf50f7f` |
| step89500 | 4.2742295066e+04 | `d2a3c049545eda35` | `0f6f3237705ac73f` | `8755d195564cb63b` |
| step90000 | 4.2535548743e+04 | `6830c33632aea269` | `9979e25d6ca7552b` | `c064553358fbc1d4` |

All distinct; all fp32, 135 tensors, meta `step/keep_front=10/n_fresh=2/depth=12`.
The driver additionally asserts the **loaded meta's** step and arch per checkpoint
**before** spending 8 GPUs, and `torch.load` exercises the zip central directory.

### 1.3 ⚠️ keep10 is a **third architecture** — only RANGES are compared

| arm | keep_front | n_fresh | layers | tensors |
|---|---|---|---|---|
| **keep10fresh2 (here)** | 10 | 2 | **12** | **135** |
| keep8fresh2 (the replicated arm) | 8 | 2 | 10 | 113 |
| keep12fresh2 | 12 | 2 | 14 | 163 |
| keep14fresh2 | 14 | 2 | 16 | 179 |

Absolute margins from these arms are **never tabulated as rungs of one ladder** —
different depths, and the keepN ladder additionally spans **two corpora** and
**unequal step counts** (`STATUS.json:warning`). A **range** is a *within-arm*
statistic, which is precisely why it is the thing that can be compared across arms.

### 1.4 ⚠️ `arch_meta.json:seed = 42` is **not** a training seed

`--seed` moves only the fresh-tail initialisation; `DistributedSampler(ds,
shuffle=True)` has no `seed=`, so data order is identical across seeds. No 7B
`sd_run` follows from it, and none is computable.

---

## 2. Q1 — the range, per axis (`split` convention)

Anchor: vanilla `models/OLMo-2-1124-7B`, **imported** from
`a04_shallow_rung_ni_7b.ANCHOR`, never redeclared, never substituted (G0/G2).
`Δ = 0.10 × residual(intact)`, imported. All four axes **CERTIFIABLE** under guard
D1–D6, **0 cells retired** on the `split` convention.

| axis | margins at 89000 / 89500 / 90000 (pp) | **range** | E[range \| pure noise] | ratio | > noise? | shape |
|---|---|---|---|---|---|---|
| **triviaqa** | −39.8702 / **−41.0850** / −40.3661 | **1.2149** | 0.6595 | **1.84×** | **YES** | **V (middle worst)** |
| popqa | −18.4317 / −18.7468 / −18.7257 | 0.3151 | 0.5843 | 0.54× | no | V |
| mmlu_content | −11.1282 / **−10.9431** / −11.0356 | 0.1852 | 0.6351 | 0.29× | no | inverted V |
| *nq_open (demoted)* | −15.8726 / −16.2604 / −15.5125 | 0.7479 | 1.1211 | 0.67× | no | V |

`Δ` for scale: triviaqa 6.3291, popqa 2.2457, mmlu_content 1.8614, nq_open 1.9945 pp.
So the triviaqa range is **19.2 % of its own Δ**, but **54.1 % of popqa's Δ** and
**65.3 % of mmlu_content's** — i.e. the same absolute wobble would be decisive on
the axes with tighter Δ.

**The noise gate did real work again: 3 of 4 ranges fail it.** A `max − min` of 3
noisy cells is biased upward even at zero true spread
(`E[range of 3 iid N(0,σ)] = 3/√π · σ = 1.6926 σ`), so the 0.185–0.748 pp ranges
are **not** measured neighbour gaps and are not reported as such. Had they been
quoted raw, the nq_open one (0.7479 pp) would have looked like the second-largest
finding in the file; it is 0.67× its own noise floor.

### 2.1 What produces the one real range: **two** resolved intervals, in opposite directions

| interval | Δacc | CI95 | p | resolved? | flips (w→r / r→w) |
|---|---|---|---|---|---|
| **triviaqa 89000→89500** | **−1.2093 pp** | [−1.5270, −0.8972] | **0.0001** | **YES** | +288 / **−505** |
| **triviaqa 89500→90000** | **+0.7245 pp** | [+0.4235, +1.0310] | **0.0001** | **YES** | **+445** / −315 |
| popqa 89000→89500 | −0.2944 pp | [−0.4766, −0.1121] | 0.0014 | YES | +66 / −108 |
| popqa 89500→90000 | +0.0000 pp | [−0.1682, +0.1682] | 1.0000 | no | +74 / −74 |
| mmlu 89000→89500 | +0.1780 pp | [−0.1496, +0.4985] | 0.2990 | no | +282 / −257 |
| mmlu 89500→90000 | −0.0855 pp | [−0.4273, +0.2494] | 0.6350 | no | +281 / −293 |
| *nq_open 89000→89500* | −0.4155 pp | [−0.7756, −0.0277] | 0.0362 | YES | +16 / −31 |
| *nq_open 89500→90000* | +0.7479 pp | [+0.3878, +1.1357] | 0.0001 | YES | +37 / −10 |

**popqa 89500→90000 is an exact zero with 74 items flipping each way** — a useful
reminder that "no net change" is not "nothing happened".

### 2.2 It is not output degeneracy

Labelled diagnostic across the triple (triviaqa): **0.000 % empty predictions** at
all three checkpoints; `top_constant_frac` **0.529 % → 0.340 % → 0.334 %**
(falling); distinct predictions 9208 → 9599 → 9358. The model is not collapsing.
But only **37.0 % / 37.8 %** of triviaqa prediction *strings* are identical across
the two intervals — so, as on keep8, **EM is a coarse read on a model whose outputs
are being rewritten wholesale between adjacent saves.** On popqa only **24.3 %** of
strings survive.

### 2.3 Zero accepts, all conventions

**0 NI accepts across all 5 tie conventions × 12 cells.** `RATIO(0.85)` mean ratios
0.3356–0.3474. Recovery fractions at step90000: triviaqa 27.2 %, popqa 9.1 %,
mmlu_content 34.0 %. This arm is nowhere near an accept — which is exactly why the
range measured here is a **methodological** quantity and not an accept being
defended.

---

## 3. Q2 — the replication verdict: **yes on size, no on shape**

| axis | keep8 range | gate | **keep10 range** | gate | k10/floor | k10/k8 | gates agree? |
|---|---|---|---|---|---|---|---|
| **triviaqa** | 1.1202 | **True** | **1.2149** | **True** | 1.84× | **1.08×** | **YES** |
| popqa | 0.2523 | False | 0.3151 | False | 0.54× | 1.25× | YES |
| mmlu_content | 0.2208 | False | 0.1852 | False | 0.29× | 0.84× | YES |
| *nq_open* | 0.3324 | False | 0.7479 | False | 0.67× | 2.25× | YES |

**All four gate booleans agree between the two arms.** That is a stronger result
than the headline axis alone: not only did the one supra-noise range reproduce, the
three sub-noise ones **stayed** sub-noise. The pre-registered
`replicates_and_material` branch fired.

### 3.1 But the mechanism is different, and this is the honest caveat

| | keep8 cluster 2 | keep10 |
|---|---|---|
| triviaqa margin shape | **monotone non-increasing** | **V, middle is the minimum** |
| resolved intervals | **1 of 2** (only the last: −1.09 pp) | **2 of 2** (−1.21 then **+0.72**) |
| worst checkpoint | **the last one** | **the middle one** |
| `best_minus_last` | **+1.1202 pp** | **+0.4960 pp** |

**This matters for the remedy, not for the requirement.** On keep8 the endpoint was
the worst point, so averaging the final k would *not* have rescued a reader — the
drift was terminal. On keep10 the excursion **recovers**, so Heineman et al.'s
prescription (average the final k) *would* have removed most of it. **The two arms
agree that a single adjacent checkpoint can misstate triviaqa by ~1.2 pp; they
disagree about which direction the error points.** Any claim that "later
checkpoints are worse" is therefore **keep8-specific** and must not be generalised
— the arm-independent claim is the weaker, symmetric one: *the margin at a
hand-picked checkpoint is unreliable at the ~1.2 pp level on triviaqa.*

Note also that `best_minus_last` — the advantage **actually realised** by reporting
the best of three instead of the last — is **2.26× smaller on keep10 (0.4960 pp)
than on keep8 (1.1202 pp)**, precisely because keep10 recovered. So §2.5's
**tolerance** (a range) replicates at 1.08×, while the **realised advantage**
does not. The gate design's precondition is phrased on the range, which is the
conservative choice and is now the better-supported one.

---

## 4. Q3 — restated in Heineman et al.'s unit

`arXiv:2508.13144`, *Signal and Noise: A Framework for Reducing Uncertainty in
Language Model Evaluation* (Heineman, Hofmann, Magnusson, Gu, Smith, Hajishirzi,
Lo, Dodge). **NeurIPS 2025 Spotlight** — re-verified live this session:
OpenReview note `sAFottNlra`, `venue="NeurIPS 2025 spotlight"`,
`venueid=NeurIPS.cc/2025/Conference`, invitation
`Submission26329/-/Camera_Ready_Revision` present. **DBLP has CoRR only**
(`journals/corr/abs-2508-13144`), so DBLP or S2 alone would misread this as a
preprint.

Their definition, re-extracted from the v1 PDF rather than taken from the repo's
earlier summary:
`Rel. Std.(m) = sqrt( (1/(n−1)) Σᵢ (mᵢ − m̄)² ) / m̄` — **ddof = 1**, matching the
`relstd` used here — computed over **the final 30 intermediate checkpoints, one per
1000 training steps**.

> ⚠️ **Table 4's cells are `SNR_{signal/noise}`.** TriviaQA 7B-4T reads
> `47.03_{0.135/0.003}` → noise **0.003**; MMLU reads `3.39_{0.078/0.023}` → noise
> **0.023**. The bare integers are SNR. Misreading that column would inflate the
> comparator by ~4 orders of magnitude. Both values match the repo's prior
> extraction exactly.

### 4.1 ⛔ THIS TABLE IS A CROSS-PROTOCOL HYPOTHESIS, NOT AN EQUAL-FOOTING COMPARISON

| | **ours (this file)** | **theirs (Table 4)** |
|---|---|---|
| n checkpoints | **3** | **30** |
| spacing | **500 steps** | **1000 steps** |
| model condition | **layer-pruned, mid-heal, 12 layers** | **intact OLMo-2 7B, 32 layers** |
| shots | **strict zero-shot** | **few-shot** (their App. A.1) |
| protocol | this repo's base (`cb_bs=32`, `add_bos=False`, `max_new_tokens=32`, greedy) | OLMES |
| MMLU interface | **content-continuation** | **letter-choice (RC/MC)** |

These may **not** be tabulated as if measured together, and no ratio may be quoted
without "n=3 vs n=30, different protocol" in the same sentence. Two of these
asymmetries were found by reading their PDF in this session and were **not** in the
repo's prior summary — **zero-shot vs few-shot**, and their TriviaQA metric being
short-form generation EM (the one axis where the metric family does match). Both
cut **against** reading a gap as evidence about injury: a zero-shot base-LM EM is a
less stabilised measurement than a few-shot one, independent of damage.

### 4.2 The numbers, with the asymmetry attached

| axis | keep10 rel.std | keep10 **abs SD** | keep8 rel.std | their OLMo-2 7B-4T | raw ratio |
|---|---|---|---|---|---|
| **triviaqa** | **0.0349** | 0.6086 pp | 0.0395 | **0.003** | **11.65×** |
| popqa | 0.0383 | 0.1700 pp | 0.0253 | *not in their suite* | — |
| **mmlu_content** | **0.0026** | 0.0890 pp | 0.0030 | **0.023** | **0.11×** |
| *nq_open* | 0.1010 | 0.3747 pp | 0.0445 | *not in their suite* | — |
| *mmlu **LETTER** (interface-matched)* | *0.0134* | *0.3530 pp* | — | *0.023* | *0.58×* |

PopQA and NQ-open are **not in their 30-benchmark suite**; those cells are **left
blank**, not filled with a nearby task.

### 4.3 ⛔ The raw 11.65× is mostly arithmetic — decomposed, not asserted

`rel.std = sd / mean`, so
`relstd_ours/relstd_theirs = (sd_ours/sd_theirs) × (mean_theirs/mean_ours)`.
Our arm sits at **17.42 %** TriviaQA; an intact OLMo-2-class 7B is in the 0.6–0.7
range. Their mean is **not published in Table 4** (only Rel.Dispersion and
Rel.Std), so it is **not guessed** — instead a sensitivity strip:

| if their mean acc = | their implied SD | **absolute SD ratio (ours/theirs)** | denominator factor |
|---|---|---|---|
| 0.40 | 0.1200 pp | **5.07×** | 2.30× |
| 0.50 | 0.1500 pp | **4.06×** | 2.87× |
| 0.60 | 0.1800 pp | **3.38×** | 3.44× |
| 0.65 | 0.1950 pp | **3.12×** | 3.73× |
| 0.70 | 0.2100 pp | **2.90×** | 4.02× |

So the defensible statement is **"2.9–5.1× larger absolute checkpoint-to-checkpoint
SD on TriviaQA"**, not 11.65×. And even that is bounded above by the estimator
itself: **at n = 3 the sample SD's own relative SD is 52.3 %**
(`sqrt(1−c₄²)/c₄`, c₄ = 0.886227), computed and self-tested in the code. A 2.9×
point estimate from 3 points, against a 30-point published value, under a different
shot setting, is **a hypothesis worth a controlled test and nothing more.**

### 4.4 What Q3 licenses, and the part that argues the other way

**Licensed as a hypothesis:** *damaged mid-heal arms may have several-fold larger
absolute TriviaQA checkpoint noise than intact OLMo-2 at the same scale.* Both of
our arms point the same way (keep10 0.6086 pp, keep8 0.6339 pp absolute SD — within
4 % of each other, which is the most reassuring number in this section).

**The counter-evidence, stated plainly:** on **MMLU we are 9–17× QUIETER than they
are** (0.0026 vs 0.023 rel.std; 0.0134 vs 0.023 even on the interface-matched
letter variant). If "damage makes everything noisier" were the story, mmlu_content
would not be the quietest cell in the file. So the honest reading is
**axis-specific**: short-form generative EM on a damaged arm is unstable;
multiple-choice-style scoring on the same damaged arm is *more* stable than the
published intact value. That is interesting, but it is **not** the clean
"damaged ⇒ noisier" claim, and it must not be written as one.

**Consequence for the paper's positioning:** the empirical foothold survives but
narrows to *one axis*, with a *decomposed* ratio of 2.9–5.1× rather than 11.65×,
from *n = 3*, under a *different shot setting* than the comparator. It is **not**
strong enough to carry a section on its own. The **normative** contribution —
that in a non-inferiority test neighbour noise is a *one-sided free option* rather
than a power loss, which is a thing Heineman et al. structurally cannot say because
they have no accept — remains the load-bearing part.

---

## 5. Verification performed

1. **Estimator self-test, executed before publishing.** `range_report` fed known
   inputs must reproduce the closed form `E[range of 3] = 3/√π = 1.6925687506`
   and must *fire* when range/SE = 2.0; `relstd([1,2,3]) = 0.5`; `c₄(2) =
   0.7978845608`, `c₄(3) = 0.8862269255` (textbook). `all_ok = True`; a failure
   aborts before any number is written.
2. **The gate applied to keep10 is the SAME CODE OBJECT that gated keep8.**
   `range_report`, `adjacent_interval_tests`, `guard_cell`, `protocol_asserted`,
   `shard_integrity_report`, `EXPECTED_RANGE_OVER_SD` are **imported from
   `a04_neighbour_variability`**; `ni_rule`, `build_nulls`, `ratio_rule`,
   `EXPECTED_N`, `AXES`, `PREREG` from `pilot_zero_rule_disagreement`;
   `paired_bootstrap`, `TIE_CONVS`, `N_BOOT`, `SEED` from A03; `ANCHOR`,
   `_load_arm`, `assert_aligned` from `a04_shallow_rung_ni_7b`. **No metric, null,
   rule, gate or anchor re-derived, and no margin obtained by subtracting a
   recorded null from a recorded accuracy.**
3. **keep8's numbers were READ from its archive, not quoted from prose.**
   `keep8_archive_readback` hard-fails if the archive's ranges differ from the
   documented 1.1202 / 0.2523 / 0.2208 / 0.3324 by > 5e-4 pp or if any gate
   boolean moved. It passed, and it confirmed from the archive itself that keep8
   was **published on `.82`/numpy 2.4.6** — the same node and sampler as this run,
   so the comparison is **within one multinomial**.
4. **Protocol confirmed from the INVOCATION, fail-closed.** `protocol_asserted`
   parses the driver's own echoed `DRIVER START … mmlu_bs=16 cb_bs=32` and
   per-axis `START … bs=…` lines out of `logs/a04_keep10_nbr_82.out`
   (`{closedbook: [32], nq_open: [32], mmlu: [16]}`) **before** anything is
   scored; driver source defaults agree as corroboration. `summary.json:meta`
   records **neither** batch size **nor** chat_template
   (`A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md`), which is why the log is the
   evidence. Batch size is not free: bs32→bs48 flipped 12/14267 popqa and
   10/3610 nq_open (`full32_rescore_v2_20260812.sensitivity_bs48_probe`).
5. **`add_bos is False` on all 9 result dirs**, asserted with **`is False`** —
   never `is not True`, which passes silently on `None`. `max_new_tokens == 32`
   on all 6 generative dirs. `chat_template=False` established **structurally**:
   neither harness contains a chat-template code path.
6. **Shard integrity: 16 of 16 cells clean.** Index set **exactly {0..7}** (not a
   file count), merged *n* exactly `EXPECTED_N` (triviaqa 17944 / popqa 14267 /
   nq_open 3610 / mmlu 14042), **0 duplicate item_ids, 0 nan**, identical
   `item_id` sequences across all four arms including the anchor.
7. **Checkpoint identity proven, not assumed** (§1.2), and single-process
   provenance verified from the training log **before** scoring (§1.1).
8. **Bootstrap seeds disjoint, checked as code.** `arm_index` 700–702, guard
   `SEED+4700`, intervals `SEED+4900`, intersected against every archived
   `bootstrap_offsets` block (`a04_full32_trajectory_ni` 203/500–503,
   `a04_keep14_trajectory_ni` 201/300–301, `a04_neighbour_variability` 400–408).
   **No archived number can be perturbed by this run.**
9. **Analysis is byte-identical on re-run** on the same node: sha256
   `0838a67edda0380e98355b5c723b3be59101d16e3366611dbd383933f8eddc52` twice.
   The evidence JSON is **byte-identical on both disks** after `scp -O`.
10. **`nvidia-smi` verified 8×0 MiB before launch and 8×0 MiB after**; `.73`'s
    keep12 job was never touched and no output name collides with
    `A04_7B_keep12f2_*` (prefix `A04_7B_keep10f2_NBR_*` verified to collide with
    nothing on disk before launch).
11. **`STATUS.json` append is mechanical and asserted**
    (`code/a04_status_add_keep10_neighbour_range_key.py`): it writes to a temp
    path and only `os.replace`s after confirming every pre-existing key still
    exists, **in the same order**, with a **byte-identical**
    `json.dumps(sort_keys=True)` value, and that exactly one key was added. **0 of
    41 pre-existing keys changed.**
    ⚠️ **The file had 41 keys at append time, not the 39 my own pre-registration
    recorded**: `full32_endpoint_is_not_the_accept_20260813` and
    `blockers_discharged_20260813` were added by **concurrent A04 work** (commits
    `7ec11d7` / `6e08c6d`) between prereg commit `0e889cd` and this append. The
    invariant is unaffected — it is *"no pre-existing key is modified"*, checked
    against the file as read — but the prereg's count was stale by two, which is a
    small illustration of the same lesson this whole document is about: **read the
    count from the artefact, not from a document.**

### 5.1 Three defects found in the inherited tooling, all fixed here and worth porting

1. **The seed-disjointness check FIRES ON ITS OWN OUTPUT.** The first run wrote
   `a04_keep10_neighbour_range.json` recording `arm_index 700–702`; the second run
   read that file back and aborted with *"arm_index [700,701,702] already used by
   a04_keep10_neighbour_range.json"*. **This is a false positive that will hit
   every idempotent re-run of every A04 analysis carrying this check**, and it
   arrives as a `FATAL` that looks like a real collision. Fixed by excluding the
   output path **by `os.path.realpath`** (not by name match, which would let a
   genuinely different file slip through), with the exclusion recorded in the JSON
   so it cannot hide a real clash. **`a04_keep12_trajectory_monotonicity.py` has
   the unfixed version.**
2. **`hostname -I` returns TEN addresses on these nodes** and `28.82.250.82` is
   **not** the first (`28.86.53.217 28.86.81.221 … 28.82.250.82 …`). A node guard
   written as `hostname -I | awk '{print $1}'` would **refuse to run on the
   correct node**. The guard here matches the whole list *and* requires
   `numpy == 2.4.6`, which is `.82`-specific in this cluster and therefore pins
   the node of record even if the IP layout changes.
3. **A `STATUS.json` append can silently reflow the whole file — twice over.**
   Writing the file back with `json.dump`'s defaults produced a **4517-line diff
   for a one-key append**, with 0 of 41 pre-existing values semantically changed.
   Two independent causes, both found only because a **textual** guard was added
   on top of the semantic one:
   - `indent=1` vs the file's committed `indent=2`;
   - **`ensure_ascii=True`** (the default) rewriting the file's raw UTF-8 CJK —
     `'提案'` inside `blockers_discharged_20260813` — as `提案`, which
     reflows every line after the first CJK character.

   A semantic "no pre-existing key changed" assertion **passes** in both cases,
   so it is not sufficient: a reviewer cannot spot a real edit inside a 4500-line
   reflow, and neither can `git log -p`. Both are now **detected from the file**
   and the script refuses to write unless the new text literally begins with the
   old text minus its closing brace. Result: **197 insertions, 0 deletions.**
   Any future agent appending to a `STATUS.json` should assume this trap exists.

---

## 6. Licensed vs NOT licensed

### Licensed
- The 12 cells' accuracies, nulls, residuals, Δ, lo95 bounds and margins in §2,
  and **0 accepts under all 5 tie conventions** and `RATIO(0.85)`.
- **"On a second, differently-damaged arm (keep10+fresh2, 12 layers), three
  checkpoints 500 steps apart in one uninterrupted heal segment show a triviaqa
  margin range of 1.2149 pp — 1.84× the range item noise alone would produce, and
  1.08× the range measured on keep8 — while popqa (0.3151), mmlu_content (0.1852)
  and nq_open (0.7479) remain indistinguishable from item noise, matching keep8's
  gate verdict on all four axes."**
- **"Two adjacent 500-step intervals were each resolved at p = 0.0001 and in
  OPPOSITE directions: −1.2093 pp (505 right→wrong vs 288 wrong→right of 17 944)
  then +0.7245 pp (445 vs 315)."** Not output degeneracy: 0 % empty, top-constant
  falling 0.529 → 0.334 %, distinct predictions not collapsing.
- **"The margin RANGE replicates (1.08×) but the SHAPE does not: keep8's triviaqa
  excursion was a monotone terminal drop, keep10's is a V that recovers, so the
  realised best-minus-last advantage is 2.26× smaller on keep10 (0.4960 vs
  1.1202 pp)."**
- **"keep10 and keep8 have absolute triviaqa checkpoint SDs within 4 % of each
  other (0.6086 vs 0.6339 pp)."**
- Q3 **as an explicitly cross-protocol hypothesis only**, with the decomposition:
  raw rel.std ratio 11.65×, **absolute-SD ratio 2.9–5.1×** depending on their
  unpublished mean, and the opposite sign on MMLU.

### NOT licensed
- ⛔ **"Later checkpoints are worse" / "the last checkpoint is the worst."** That
  is keep8's shape and it did **not** replicate. keep10's worst checkpoint is the
  **middle** one.
- ⛔ **Treating the three checkpoints as REPLICATES.** They are successive states
  of **one** optimisation; their spread is heal progress + data order. This is a
  **checkpoint-SELECTION** quantity, never seed variance. No 7B `sd_run` exists or
  is reconstructible (`--seed` moves only fresh-tail init; historical 7B seeds
  unrecorded).
- ⛔ **Comparing keep10 / keep8 / keep12 / keep14 ABSOLUTE margins as rungs of one
  ladder.** Four depths, two corpora, unequal steps.
- ⛔ **Reporting the three sub-noise ranges (0.185 / 0.315 / 0.748 pp) as measured
  neighbour gaps.** They fail `range_exceeds_item_noise`.
- ⛔ **Quoting "11.65× noisier than published OLMo-2"** without the decomposition
  in §4.3. The defensible figure is **2.9–5.1× on absolute SD**, from n=3, and
  **only on triviaqa** — MMLU points the other way.
- ⛔ **Tabulating our rel.std next to Heineman et al.'s as if co-measured**
  (§4.1): n=3 vs n=30, 500 vs 1000-step spacing, damaged vs intact, **zero-shot
  vs few-shot**, base protocol vs OLMES, content-continuation vs letter-choice.
- ⛔ **"Damaged arms are noisier"** as a general claim. True direction on triviaqa,
  **reversed** on MMLU (we are 9–17× quieter).
- ⛔ **Calling any of this "harness noise."** Same-code re-runs on a fixed
  checkpoint are bit-identical
  (`full32_rescore_v2_20260812.correction_to_the_jitter_premise`).
- ⛔ Any **K1/K2/K3** clause — defined over the pre-registered **1B** arm set.
- ⛔ **Quoting any margin here to better than 0.01 pp across nodes.** numpy's
  multinomial sampler differs between 2.4.6 (`.82`, which produced this file) and
  2.5.1; max observed drift 0.005294 pp, triviaqa only. Within a node the output
  is byte-identical. **This may not explain away any move larger than ~0.006 pp**
  — every resolved interval here is 55–230× larger.

---

## 7. What this changes for the certification rule

1. **§2.5's tolerance is no longer a one-arm number, and it did not need
   widening.** It proposed ≈1.2 pp on triviaqa and ≲0.35 pp elsewhere. keep10
   independently gives **1.2149 pp** on triviaqa and **0.185–0.315 pp** on the
   other decision axes. The proposed tolerance is **within 8 % on the axis that
   matters and correct in magnitude elsewhere.** Recommend amending §2.5 to cite
   **two arms** and to state the tolerance as **≈1.2 pp (triviaqa) / ≈0.35 pp
   (popqa, mmlu_content)**, unchanged in value.
2. **§2.0.2's "stated PER-AXIS, not blanket" was the right call, and is now
   confirmed twice.** The same one axis cleared, and the same three did not, on two
   independently damaged arms. Blanket distrust of single-checkpoint numbers
   remains unsupported.
3. **The precondition should be phrased on the RANGE, not on "the last checkpoint
   is worst."** §2.0.2 already is. keep10 shows why that matters: had the rule been
   written as "later checkpoints are worse", keep10 would have refuted it.
4. **A third arm is still not worth funding.** Two arms now agree on all four gate
   booleans and on the triviaqa magnitude to 8 %. What is *not* settled is the
   **shape/mechanism**, and a third 3-point cluster cannot settle that — that
   needs a *denser* trajectory on **one** arm, which `.73`'s concurrent keep12
   11-checkpoint scan is already producing. **Recommend: read the keep12
   trajectory for the shape question rather than adding a fourth 3-point cluster.**
5. **The Heineman comparison is a narrow, axis-specific hypothesis, not a
   headline.** §4.4. The paper's defensible novelty on this topic remains the
   **equivalence-decision** argument (neighbour noise as a one-sided free option in
   a non-inferiority test), which is unaffected by this run and which their work
   structurally cannot make.
6. **Two tooling defects are on the record** (§5.1) and one of them —
   the self-colliding seed check — will produce a spurious `FATAL` for the next
   agent who re-runs any A04 analysis idempotently.
