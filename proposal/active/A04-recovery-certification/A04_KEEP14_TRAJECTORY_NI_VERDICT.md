# A04 — the NI(Δ) discrimination curve along the keep14+fresh2 7B heal trajectory

**Verdict string:** `CONSTANT_REJECT_MARGIN_NON_MONOTONE_AND_RESOLVED`

**Date:** 2026-08-13 · **GPU:** 2.0756 GPU-h (`.73` 449 s × 8 + `.82` 485 s × 8, wall
clock driver-start to driver-end; the 4-axis scoring itself was 427 s + 434 s =
1.9133 GPU-h). Analysis is CPU-only.
**Nodes:** `.73` (step 128 000) and `.82` (step 153 500), 8×H20 each, zwfy6.
**Not touched:** `.104` (paperC Qwen3 heal), `LOCAL`/`.21` (SparseForge #246).
**Evidence:** `evidence/a04_keep14_trajectory_ni.json`
**Code:** `code/a04_keep14_trajectory_axes_driver.sh`, `code/a04_keep14_trajectory_ni.py`

---

## 1. What was asked, and what this is not

`STATUS.json:shallow_rung_ni_discrimination_20260812.implication_for_pilot_two
.cheap_next_steps_dominate[1]` asked for the intermediate 7B checkpoints to be
scored, because *"locating [the accept boundary] is what 'the gate discriminates'
would actually mean."*

This is **not** a re-test of whether keep14+fresh2 accepts. That was already
answered NO at the endpoint (REJECT 3/3, margins −28.4624 / −15.0810 / −7.4749 pp).
The question here is strictly different: **is the NI margin monotone in heal step,
and does its slope locate an accept boundary?**

The answer to the first half is **no, not on every axis** — and that is the
finding.

### The expectation was fixed in advance and was met

`full32_rescore_v2_20260812.trajectory_scan_NOT_run.expectation_to_design_for`
had already committed, for the analogous full32 scan, to *"earlier checkpoints
are LESS converged so should reject HARDER … the scan is worth having as the
discrimination curve but should not be sold in advance as likely to find an
accept."* No accept was found, on any checkpoint, on any of the five tie
conventions. **Nothing here was repackaged after the fact.**

### The full32 half was NOT run

Its four intermediate ckpts (step 5000/10000/15000/20000, 81.6 GiB each) are
**wzc1-only**; measured cross-disk rate is 16 MiB/s ≈ 89 min per ckpt before any
GPU work, and `LOCAL`/`.21` are outside this dispatch's node budget. That half
remains open exactly as `trajectory_scan_NOT_run` describes it.

---

## 2. The curve (`split` convention, pre-registered)

Anchor: vanilla `models/OLMo-2-1124-7B` (`base_full`, `base_full_nqopen`,
`7B_base`), **imported** from `a04_shallow_rung_ni_7b.ANCHOR`, never redeclared,
never substituted (guards G0/G2). `Δ = 0.10 × residual(intact)`, never
substituted. All four axes CERTIFIABLE under guard D1–D6; **0 of 9 decision
cells retired**; decision family 9 → 9.

| axis | step | acc | recovery | deficit pp | lo95 pp | Δ pp | **margin pp** | boot SE | SE to flip | NI |
|---|---|---|---|---|---|---|---|---|---|---|
| **triviaqa** | 128 000 | 28.238 % | 44.21 % | 35.3099 | −35.9396 | 6.3291 | **−29.6105** | 0.3828 | 77.3 | REJECT |
| | 153 500 | 28.767 % | 45.05 % | 34.7804 | −35.4157 | 6.3291 | **−29.0866** | 0.3862 | 75.3 | REJECT |
| | 200 000 | 29.403 % | 46.05 % | 34.1451 | −34.7916 | 6.3291 | **−28.4624** | 0.3930 | 72.4 | REJECT |
| **popqa** | 128 000 | 8.208 % | 26.34 % | 16.5417 | −17.0884 | 2.2457 | **−14.8426** | 0.3324 | 44.7 | REJECT |
| | 153 500 | 7.535 % | 23.35 % | 17.2146 | −17.7823 | 2.2457 | **−15.5366** | 0.3452 | 45.0 | REJECT |
| | 200 000 | 7.976 % | 25.31 % | 16.7730 | −17.3267 | 2.2457 | **−15.0810** | 0.3366 | 44.8 | REJECT |
| **mmlu_content** | 128 000 | 37.922 % | 50.91 % | 9.1369 | −9.7280 | 1.8614 | **−7.8666** | 0.3593 | 21.9 | REJECT |
| | 153 500 | 38.299 % | 52.94 % | 8.7594 | −9.3434 | 1.8614 | **−7.4820** | 0.3550 | 21.1 | REJECT |
| | 200 000 | 38.321 % | 53.06 % | 8.7381 | −9.3363 | 1.8614 | **−7.4749** | 0.3637 | 20.6 | REJECT |
| *nq_open (demoted)* | 128 000 | 5.651 % | 25.56 % | 14.8476 | −15.9280 | 1.9945 | **−13.9335** | 0.6568 | 21.2 | REJECT |
| | 153 500 | 5.956 % | 27.08 % | 14.5429 | −15.6233 | 1.9945 | **−13.6288** | 0.6568 | 20.8 | REJECT |
| | 200 000 | 5.983 % | 27.22 % | 14.5152 | −15.5679 | 1.9945 | **−13.5734** | 0.6399 | 21.2 | REJECT |

Every checkpoint: **0 of 3 decision axes accepting**, threshold 2. Identical under
all five tie conventions (`split`/`first`/`last`/`wrong`, and `credit` where MMLU
retires to 0 of 2). `RATIO(0.85)` also rejects all three (mean ratio 0.4644 /
0.4654 / 0.4728) — no rule disagreement anywhere on this trajectory.

---

## 3. Monotonicity — the actual finding

**2 of 3 decision axes are monotone; popqa is not, and its regression is
statistically resolved, not noise.**

| axis | 128k→153.5k | 153.5k→200k | monotone ↑ |
|---|---|---|---|
| triviaqa | **+0.5239** pp | **+0.6242** pp | yes |
| popqa | **−0.6939 pp** | +0.4556 pp | **NO** |
| mmlu_content | +0.3846 pp | +0.0071 pp | yes |
| *nq_open* | +0.3047 pp | +0.0554 pp | yes |

Because a *sign* is not a finding, each adjacent interval got its own paired item
bootstrap (imported `paired_bootstrap`, seeds `SEED+900+13·axis+7·pair`, disjoint
from every archived offset). A move counts as resolved only if the CI excludes
zero **and** p < 0.05 (conservative AND — see §6):

| interval | Δacc pp | CI95 pp | p | resolved? | flips |
|---|---|---|---|---|---|
| triviaqa 128k→153.5k | +0.5294 | [+0.1616, +0.8972] | 0.0058 | **yes** | +621 / −526 |
| triviaqa 153.5k→200k | +0.6353 | [+0.3121, +0.9641] | 0.0002 | **yes** | +503 / −389 |
| **popqa 128k→153.5k** | **−0.6729** | **[−0.9252, −0.4206]** | **0.0001** | **yes** | **+122 / −218** |
| popqa 153.5k→200k | +0.4416 | [+0.2313, +0.6519] | 0.0001 | **yes** | +158 / −95 |
| mmlu 128k→153.5k | +0.3774 | [+0.0000, +0.7691] | 0.0514 | no | +406 / −353 |
| mmlu 153.5k→200k | +0.0214 | [−0.2991, +0.3490] | 0.9264 | no | +259 / −256 |
| nq_open 128k→153.5k | +0.3047 | [−0.1662, +0.7756] | 0.2384 | no | +45 / −34 |
| nq_open 153.5k→200k | +0.0277 | [−0.3601, +0.4432] | 0.9298 | no | +27 / −26 |

**PopQA got resolvedly WORSE over 25 500 steps of healing** (p = 0.0001,
218 right→wrong vs 122 wrong→right), then partially recovered. So "more healing
⇒ closer to certifiable" is **false as a per-axis statement**, on the very arm
whose endpoint verdict A04 has been quoting.

### It is not an output-degeneracy artefact

Labelled diagnostic (never enters a verdict): across all three checkpoints popqa
has **0.000 % empty predictions**, top-constant share *falling* 2.481 % → 1.703 %
→ 1.213 %, and distinct predictions *rising* 8436 → 8556 → 9205. The model is not
collapsing; it is genuinely changing its answers. Sampled right→wrong flips are
content substitutions, not format breakage (`Moscow.` → `Nekrasov was born in
Moscow.`; `New York City.` → `Chicago, Illinois`; `Actor` → `Teacher`).

Also worth recording: only **30.17 %** of popqa prediction strings are identical
between 128k and 153.5k (triviaqa 47.18 %). The generation churns wholesale while
EM moves by fractions of a point — **EM is a very coarse read on these
checkpoints**, which is itself relevant to a rule that certifies on EM.

---

## 4. Where the accept boundary is: not here

Naive linear extrapolation of the last interval to margin = 0 — reported as a
**distance**, never as a forecast:

| axis | slope (last interval) | extra heal steps to margin 0 | × the whole 200k run |
|---|---|---|---|
| triviaqa | +0.01342 pp/1k | 2 120 442 | 10.6× |
| popqa | +0.00980 pp/1k | 1 539 222 | 7.7× |
| mmlu_content | +0.00015 pp/1k | 48 807 563 | **244.0×** |
| *nq_open* | +0.00119 pp/1k | 11 392 500 | 57.0× |

Under the OLS slope through all three points, **popqa's slope is negative
(−0.001709 pp/1k) and no forward extrapolation reaches zero at all.**

A straight line through 3 checkpoints of one run is not a healing model (heal
curves are concave), so these are order-of-magnitude statements only. But the
order of magnitude is the point: **the accept boundary is 8–244× the entire heal
budget away, and on one axis it is in the wrong direction.** It is not on this
trajectory, and no plausible extension of this trajectory reaches it.

MMLU-content is the sharpest case. It looks closest to accepting (53.06 % recovery,
smallest deficit), yet it is the axis whose margin has **stopped moving**: the last
46 500 steps bought +0.0071 pp, within item noise (p = 0.9264). It is not
converging on the accept threshold; it has plateaued 7.47 pp below it.

---

## 5. Verification performed

### 5.0 Protocol confirmed from the invocation (closes `A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md`)

MAIN filed `A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md` **while this scan was still
running**, noting a real artefact defect: `summary.json:meta` records
`mode / keep_front_layers / n_fresh_layers / num_hidden_layers / ckpt_step / ckpt /
base_model / add_bos / max_new_tokens` and **neither `batch_size` nor
`chat_template`** — so the two most decision-critical fields cannot be
reconstructed from the result dirs after the fact. That note is correct, and its
requirement (confirm both from the actual invocation; record them and *how*, in a
`protocol_asserted` field) is now satisfied **mechanically, as a hard gate**, not
by assertion in prose:

`a04_keep14_trajectory_ni.protocol_asserted()` runs **before anything is scored**
and parses the driver's own echoed lines out of the launch logs:

| source | evidence | value |
|---|---|---|
| `logs/a04_keep14_traj_128000.out` (`.73`) | `DRIVER START … mmlu_bs=16 cb_bs=32`; `closedbook START … bs=32`; `nq_open START … bs=32`; `mmlu START … bs=16` | **cb 32 / mmlu 16** |
| `logs/a04_keep14_traj_153500.out` (`.82`) | same four lines | **cb 32 / mmlu 16** |
| driver source defaults | `CB_BS="${CB_BS:-32}"`, `MMLU_BS="${MMLU_BS:-16}"` | corroboration only |
| all 9 result dirs | `summary.json:meta.add_bos` | **`False`**, asserted with `is False` |

The logs are the evidence (the driver echoes the variables it actually passes to
the harness); grepping my own source would not be. **The gate fails closed** —
negative-tested: a doctored log reading `cb_bs=48` produces
`FATAL protocol deviation … != {'cb_bs': 32, 'mmlu_bs': 16}` and **no output file
is written**; a missing log produces `FATAL: driver log … absent — … Refusing to
publish cells whose protocol cannot be established.`

**`chat_template=False` is established structurally, not by a flag** — because
there is no flag. Neither `scripts/eval_olmo2_closedbook_qa.py` nor
`scripts/eval_olmo2_mmlu_content.py` contains a chat-template code path; the only
occurrence of the string in either file is a docstring line. A protocol that
cannot be switched on cannot have been switched on. `add_bos is False` (the other
half of the base protocol) *is* in the artefacts and is asserted with `is False`,
**never** `is not True`, which passes silently on `None`.

The **step 200 000 endpoint predates this driver** (2026-08-02) and so has no log
from it. Its protocol is re-verified from the archive's own launch logs:
`logs/cb_driver_73.out` echoes `START base_full … bs=32` *and*
`START keep14_step200k … bs=32`; `logs/nqopen_driver_73.log` echoes
`START base_full_nqopen … bs=32`. So **the new cells match both the endpoint and
the anchor.** (This is also why §5.1 item 5 below matters: MMLU-content's 16 comes from
`p06_run_transferred.sh` leaving `BS` unset, and `git log -p --follow` shows one
revision of that line.)

MAIN's suggested durable fix — have the harness write `batch_size` and
`chat_template` into `summary.json:meta` — is **not** done here: it would modify a
shared scoring harness whose md5 is pinned as identical across both disks and to
the copies that produced the anchor and the endpoint. Touching it would break the
same-code guarantee that §5.1 item 5 relies on. It should be filed as a separate change
that re-baselines nothing.

### 5.1 Other checks

1. **Archived endpoint reproduced.** Re-deriving keep14fresh2@200k under its own
   archived bootstrap offset (`arm_index = 201`) returns −28.4624 / −15.0810 /
   −7.4749 pp, matching `evidence/a04_shallow_rung_ni_7b.json` to **< 5 × 10⁻⁵ pp**
   on all three decision axes. The script **hard-fails** otherwise. This proves the
   imported guard/anchor/rule are the ones that produced the archive, so the new
   points are on the same scale as the endpoint they are compared to.
2. **Shard integrity, asserted and recorded.** For all 16 arm×axis cells: shard
   index set **exactly {0..7}** (not merely "8 files"), merged n exactly equal to
   `EXPECTED_N` (triviaqa 17944 / popqa 14267 / nq_open 3610 / mmlu 14042),
   **0 duplicate item_ids, 0 nan**, and identical item_id sequences across arms.
   Per-shard row counts are in the JSON. Anchor dirs on zwfy6 confirmed 8/8
   (16 files in `base_full`: 8 popqa + 8 triviaqa).
3. **Independent harness cross-check.** Pre-existing letter-protocol MMLU runs
   (`olmo2_downstream_results/*_know`, a *different* script at bs=8, run
   2026-07-20 and 2026-08-02) give letter acc **0.30117 / 0.31242 / 0.31797** for
   128k/153.5k/200k. This dispatch's harness gives **0.302094 / 0.311423 /
   0.318402** — agreeing within 0.10 pp and reproducing the same ordering, from
   code and runs that predate this analysis.
4. **Checkpoints are distinct weights.** `step128000.pt` and
   `keep14_step153500.pt` have **identical byte size** (48 724 473 850 — both carry
   `optimizer_state`; the 200k endpoint lacks it, hence 16.2 GiB). Size is *not*
   identity: head-400 MiB md5 `29474e2b…` vs `ec9ed051…`, mid-200 MiB-at-20 GiB
   `dfe33568…` vs `2c6eb15a…`; `torch.load` metas report step 128000 vs 153500,
   both `keep_front=14 n_fresh=2`, 179 tensors, strict load. The driver asserts
   the meta step equals the requested step before spending GPU.
5. **Protocol recovered from the archive's own logs**, not from prose:
   `logs/cb_driver_73.out` echoes `bs=32` for both `base_full` and
   `keep14_step200k`; `scripts/p06_run_transferred.sh` leaves `BS` unset so
   MMLU-content ran at the driver default **16**, and `git log -p --follow` shows
   exactly one revision of that line (commit `d2e28f2`), so 16 is not later drift.
   Harness md5s (`2ed41993…`, `fe4a62db…`) are identical on both disks and to the
   copies that produced the archive.
6. **Batch size was not treated as free.** `full32_rescore_v2_20260812
   .sensitivity_bs48_probe` showed bs32→bs48 flips 12/14267 popqa and 10/3610
   nq_open items, so CB_BS=32 and MMLU_BS=16 were pinned to match the anchor and
   the endpoint.
7. **`add_bos is False` asserted with `is False`**, not `is not True` (the latter
   passes silently on `None`). Neither harness has a chat-template code path at
   all; these are BASE LMs with no SFT/RL.
8. **Bootstrap seeds disjoint** from every archived cell (new arms
   `arm_index` 300–301; pilot_zero 0–1, step100k 100–102, shallow_rung 200–203).
   No archived number can be perturbed by this run.

---

## 6. Licensed vs NOT licensed

### Licensed
- The 7B accuracies, nulls, residuals, Δ, lo95 bounds and margins in §2, and
  their reproduction of the archived endpoint.
- **"No realisable perturbation of the ITEM SAMPLE flips any of these
  verdicts"** — margins sit 20.6–77.3 bootstrap SE from accepting, measured at 7B.
- **"The popqa margin regressed between 128 000 and 153 500 by more than item
  sampling can explain"** (p = 0.0001, CI [−0.9252, −0.4206] pp) — this is a
  statement about the item sample only.
- "The accept boundary is not on this trajectory": every axis needs 7.7–244× the
  entire heal budget under a naive line, and popqa's all-points slope is negative.

### NOT licensed
- ⛔ **Anything of the form "the 7B deficits are large relative to seed
  variance."** `sd_run` is a **1B-only** quantity (S=3, keep12@5000). Every 7B
  rung — including all three points here — has **exactly ONE seed**, and the
  historical 7B ladder's seeds are **unrecorded** (`--seed` postdates them;
  trainer `afdfa66` called no seeding function), so **no 7B `sd_run` is computable
  or retrospectively reconstructible.** The deficit/sd_run column in the JSON is
  labelled cross-scale extrapolation and is not a 7B variance statement.
- ⛔ **Treating the three checkpoints as replicates of each other.** They are
  three checkpoints of **one run**; their spread is heal progress plus data order,
  not independent-run variance. A trend across them is a within-run trajectory,
  not a sampling distribution. In particular the popqa dip **cannot** be
  attributed to (or excused as) seed variance, because there is no seed variance
  estimate at this scale — it is only established as larger than *item* noise.
- ⛔ **Calling the popqa dip "harness noise."**
  `full32_rescore_v2_20260812.correction_to_the_jitter_premise` established there
  is **no measured runtime-jitter floor** on this harness (same-code re-runs are
  bit-identical; every previously reported "jitter" was a code-version or
  batch-size difference). These are three *different models*, so bit-identity does
  not apply — but it also removes "noise" as an available explanation.
- ⛔ **Any claim that the mmlu 128k→153.5k move is real.** Its CI lower bound
  landed *exactly* at 0.0000 pp while p = 0.0514: the two criteria disagree
  because the bootstrap of a 0/1 metric is discrete. Recorded as
  `criteria_disagree: true` and treated as **not resolved** (conservative AND).
  Picking the favourable criterion would turn a tie into a result.
- ⛔ Any K1/K2/K3 clause. Those are defined over the pre-registered **1B** arm
  set; a 7B ladder cannot fire them. Also unchanged: the two-corpora / unequal-steps
  `STATUS.json:warning` still holds, so this ladder is not a controlled scaling law.
- ⛔ Any statement about the **full32** trajectory. It was not run (§1).

---

## 7. What this changes for the certification rule

The blocker `shallow_rung_ni_discrimination_20260812` was written to address was
*"NI has never been observed to accept, so it has not been shown to
discriminate."* That was answered by showing NI accepts on the **zero-damage**
control. This pass adds the complementary fact, and it is not a comfortable one:

1. **The accept boundary is not reachable by healing longer on a damaged arm.**
   At 50 % depth kept and 200k heal steps — the shallowest damage and largest heal
   budget in the repository — the three decision margins are 7.5–28.5 pp short and
   moving at 0.00015–0.01342 pp per 1000 steps. The gap is 8–244 heal-runs wide.
   **"Heal it longer" is not a route to certification**, and any future tranche
   priced on that assumption is mispriced.
2. **Recovery is not monotone per axis, and the non-monotonicity is resolved.**
   A certification rule evaluated at a single arbitrary checkpoint can therefore
   return a *better* verdict than a later checkpoint of the same run. On popqa,
   step 128 000 certifies strictly better than step 153 500 (margin −14.8426 vs
   −15.5366 pp, both REJECT here — but the ordering is the wrong way round and
   resolved at p = 0.0001). **Any future accept obtained at a hand-picked
   checkpoint must be shown to survive its neighbours**, or the reported margin is
   partly a checkpoint-selection artefact. This is a concrete, cheap-to-run
   requirement that the current gate design does not contain.
3. **MMLU-content, the axis closest to accepting, has plateaued rather than
   converged** (+0.0071 pp over the last 46 500 steps, p = 0.9264). Its apparent
   proximity to the threshold is not evidence of approach.
4. **EM is coarse relative to the churn.** Only 30–47 % of generative predictions
   are string-identical between adjacent checkpoints while EM moves ≲ 0.7 pp. The
   rule is reading a stable-looking scalar off a model whose outputs are being
   rewritten wholesale.

**Recommendation (unchanged in direction, sharper in content):** do not fund a
new damaged-arm tranche on the premise that more heal steps approach the
boundary — this pass measures that premise and it is false at 7B. The remaining
cheap step is the **full32 trajectory** (wzc1-resident; run on `LOCAL`/`.21` when
SparseForge frees them, zero transfer), because full32 is the only arm where
"recovered" is defensible and therefore the only place the boundary has been seen
from. Add a **checkpoint-neighbourhood robustness requirement** to the gate design
before any accept is reported from a single checkpoint.
