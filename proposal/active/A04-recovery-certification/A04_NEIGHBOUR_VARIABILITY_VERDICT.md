# A04 — how much can a hand-picked checkpoint buy? (neighbour variability) and does the popqa mid-heal dip replicate?

**Verdict string:** `NEIGHBOUR_RANGE_MATERIAL_GE_0p5PP_TOLERANCE_REQUIRED_LEGB_DOES_NOT_REPLICATE`

**Date:** 2026-08-13 · **GPU:** **8.6556 GPU-h** wall-clock (`.73` 2317 s × 8 = 5.1489;
`.82` 1131 s × 8 = 2.5133 and 447 s × 8 = 0.9933). The 4-axis scoring itself was
2298 s + 1567 s = **8.5889 GPU-h** across 9 checkpoints. Analysis is CPU-only.
**Nodes:** `.73` (Leg A, 6 ckpts) and `.82` (Leg B, 3 ckpts), 8×H20 each, zwfy6.
Both were verified 0 MiB / 0 % before launch and the driver refuses to start if
> 8000 MiB is held.
**Not touched:** `.104` (paperC Qwen3 heal), `LOCAL`/`.21` (SparseForge #246).
**Evidence:** `evidence/a04_neighbour_variability.json`
**Code:** `code/a04_neighbour_variability_driver.sh`, `code/a04_neighbour_variability.py`

---

## 0. Headline, in one paragraph

The claim under test was §7.2 of `A04_KEEP14_TRAJECTORY_NI_VERDICT.md`: *"any future
accept obtained at a hand-picked checkpoint must be shown to survive its
neighbours."* It rested on **one arm at 25 500-step spacing** and carried **no
tolerance**. Measured at realistic spacing (500 steps, one arm, two clusters):
**the range is material on exactly one axis of one cluster — triviaqa,
1.1202 pp, which is 1.70× the range that pure item noise would produce — and is
indistinguishable from item noise on every other axis.** So the requirement is
**vindicated but narrow**: it is a real, quantifiable loophole, worth
**≈1.1 pp on the widest axis and ≲0.35 pp elsewhere**, and it is driven by a
single **resolved 1.09 pp accuracy drop over 500 steps** (p = 0.0001). Separately,
the cross-arm replication **failed**: shortgpt16's popqa does **not** regress over
128k→153.5k (**+0.0841 pp**, p = 0.5084, CI straddles zero), so the popqa dip is
**not** a general property of healing and that half of the claim must be
downgraded to *"there exists at least one arm on which this happens."*

---

## 1. Premises I checked, and the four that were wrong

The dispatch invited me to reject its premises. Four needed correcting, and two of
them would have silently corrupted the result.

### 1.1 ⛔ `shortgpt16/step128000.pt` on zwfy6 is **CORRUPT** — Leg B could not have run as specified

| | bytes | `zipfile` | `torch.load` |
|---|---|---|---|
| zwfy6 `step128000.pt` | **7 755 268 096** | **FAIL** | **FAIL** |
| zwfy6 `step153500.pt` | 48 724 473 978 | OK (731 entries) | OK |
| zwfy6 `step200000.pt` | 48 724 473 978 | OK (731 entries) | OK |
| **wzc1** `step128000.pt` | **48 724 473 978** | **OK (731 entries)** | OK |

The error is `PytorchStreamReader failed reading zip archive: failed finding
central directory ... high likelihood that your checkpoint file is corrupted` — a
**truncated write at 15.9 % of full size**, not a smaller dtype.

The dispatch's own ledger listed this arm as `n_ckpt=6` with these three steps
available, and warned me about a *different* shortgpt16 hazard (the merged-without-shards
**eval results**). The actual blocker was the **checkpoint itself**. Note that
`ls -l` cannot catch this: the file is present, non-zero, and dated like its
siblings, and this arm's ckpts do not all share one size anyway.

**Repair, with provenance:** staged the intact wzc1 copy to zwfy6 via `scp -O`
(2592 s, 18.8 MiB/s) into `outputs/a04_staged/sg16_step128000_from_wzc1.pt`, then
verified **full-file sha256 on both disks**:
`858eb32f389b5fd9b95fa551f296cab329feb176efb49eb70247eecec386643c` — **identical**.
A head/tail sample would have been the wrong check, since truncation is exactly
the failure mode. The driver additionally asserts the *loaded meta's* `step ==
128000` before spending GPU, and it **refuses to fall back** to the corrupt file
(the staged path must be passed explicitly via `SG16_CKPT_128000`).
**The corrupt original was NOT deleted** — it is the evidence, and the same defect
may exist in other arms.

### 1.2 ⛔ Leg A's cluster 1 **straddles a resume seam** — it is not an uninterrupted 500-step neighbourhood

This is the one the dispatch specifically asked me to look for, and it is real for
**cluster 1 only**:

| ckpt | written by | when |
|---|---|---|
| step124000 | `.73` process, resumed from `step121000_full.pt` | 2026-08-08 19:35 |
| step124500 | **same** `.73` process — then it **died** at 20:26 (TCPStore error) | 2026-08-08 20:24 |
| step125000 | **a DIFFERENT process on `.82`**, resumed **from** `step124500.pt` | 2026-08-12 01:27 |

The trainer restores optimizer state and RNG but rebuilds the loader
(`sampler.set_epoch(epoch); data_iter = iter(loader)`) **without fast-forwarding
within the epoch**, so the 124500→125000 interval saw a **different data order**
than an uninterrupted 500 steps would have.

**Cluster 2 (130000/130500/131000) is entirely inside the single `.82` process**
(log lines 319/347/375, no boundary, continuous data order). Cluster 2 is
therefore **the clean measurement and the one the headline uses**; cluster 1 is
reported separately. This turned out to matter: the material range appears in
**cluster 2**, the *clean* one — so it cannot be dismissed as a resume artefact.

### 1.3 ⚠️ keep8fresh2 is a **different architecture** from keep14fresh2 — Leg A is a within-arm range, not another rung

| arm | keep_front | n_fresh | layers | tensors | front contiguous? |
|---|---|---|---|---|---|
| keep8fresh2 (Leg A) | 8 | 2 | **10** | **113** | yes |
| keep14fresh2 (the archived claim) | 14 | 2 | 16 | 179 | yes |
| shortgpt16 (Leg B) | 16 | 0 | 16 | 179 | **NO** — `[0..12, 16, 17, 31]` |

So absolute margins from the three arms are **not** rungs of one ladder and are
never tabulated as such. This does not weaken Leg A: the claim is about
**within-run** checkpoint selection, which is exactly what a within-arm range
measures.

### 1.4 ⚠️ "the same step" is **not** synonymous across arms

`step` is an optimizer-step count only. shortgpt16 is at **epoch 2** by 153500 and
**epoch 3** by 200000; keep14fresh2 is at **epoch 1** at 128000 and **epoch 2** at
153500. Same step ⇒ **not** the same data seen. Leg B is therefore a replication
of a **phenomenon**, never a matched pairwise comparison — recorded in the JSON's
`replication_verdict.caveat`.

### 1.5 ✅ Premises that held

All six Leg A ckpts load, are fp32/113-tensor, and are **distinct weights** —
verified by per-tensor sha256 of `lm_head` / `embed_tokens` / `layers.0.q_proj`
**and** the float64 sum of every parameter. **Byte size is not identity here**:
124000/124500 share a size (34 152 197 522) and 125000/130000/130500/131000 share
another (34 152 196 306). The dispatch was right to insist on this check.

---

## 2. LEG A — the neighbour range (`split` convention)

Anchor: vanilla `models/OLMo-2-1124-7B`, **imported** from
`a04_shallow_rung_ni_7b.ANCHOR`, never redeclared, never substituted (G0/G2).
`Δ = 0.10 × residual(intact)`, never substituted. All four axes **CERTIFIABLE**
under guard D1–D6; **0 of 36 cells retired** on the decision axes (the 9 retired
cells are `credit`-convention MMLU only).

### 2.1 Cluster 2 — **clean, single process** (the headline measurement)

| axis | margins at 130000 / 130500 / 131000 (pp) | **range** | E[range \| pure noise] | > noise? | best ≠ last? |
|---|---|---|---|---|---|
| **triviaqa** | −41.4141 / −41.4529 / **−42.5343** | **1.1202** | 0.6577 | **YES (1.70×)** | **YES** |
| popqa | −18.0599 / −18.1370 / −18.3122 | 0.2523 | 0.5818 | no | yes |
| mmlu_content | −11.7905 / −11.6623 / −11.5698 | 0.2208 | 0.6522 | no | no |
| *nq_open (demoted)* | −15.8449 / −15.7064 / −16.0388 | 0.3324 | 1.1401 | no | yes |

### 2.2 Cluster 1 — **resume seam** (reported, not headline)

| axis | margins at 124000 / 124500 / 125000 (pp) | range | E[range \| noise] | > noise? |
|---|---|---|---|---|
| triviaqa | −43.9328 / −44.1223 / −43.8548 | 0.2675 | 0.6442 | no |
| popqa | −19.1743 / −19.1743 / −19.2935 | 0.1192 | 0.5914 | no |
| mmlu_content | −11.7763 / −11.5626 / −11.6837 | 0.2136 | 0.6449 | no |
| *nq_open* | −16.1496 / −16.3712 / −16.2604 | 0.2216 | 1.1496 | no |

**The resume seam did NOT widen the range** — cluster 1's ranges are *smaller*
than cluster 2's on every axis. So the seam is a caveat on cluster 1's
interpretation, not a driver of the finding.

### 2.3 The guard that stops a range from being a fake finding

A range is `max − min` of *k* noisy numbers and is **biased upward by noise even
when the true spread is zero**: for iid N(0, s), **E[range of 3] = 3/√π · s =
1.6926 s** (exact for the normal). The JSON therefore reports, for every cell,
`expected_range_if_pure_noise_pp = 1.6926 × mean(bootstrap SE)` next to the
observed range, plus a boolean `range_exceeds_item_noise`, and **the headline
cannot claim a measured gap unless that boolean is true**.

This guard did real work: **7 of the 8 decision-axis ranges across both clusters
fail it.** Had I reported ranges without it, I would have announced a
"0.22–0.33 pp neighbour gap" on popqa/mmlu/nq_open that is **smaller than the
noise floor of the statistic itself** — precisely the error the dispatch asked me
to avoid. Only triviaqa/cluster-2 clears it, at 1.70×.

### 2.4 What actually produces the one real range

It is not diffuse jitter across all three points — it is **one resolved drop in
the final 500 steps**:

| interval | Δacc | CI95 | p | resolved? | flips |
|---|---|---|---|---|---|
| triviaqa 130000→130500 | −0.0223 pp | [−0.2675, +0.2173] | 0.8798 | no | +249 / −253 |
| **triviaqa 130500→131000** | **−1.0867 pp** | **[−1.3319, −0.8359]** | **0.0001** | **YES** | **+160 / −355** |
| popqa 130000→130500 | −0.0911 pp | [−0.2383, +0.0561] | 0.2244 | no | +50 / −63 |
| popqa 130500→131000 | −0.1542 pp | [−0.3224, +0.0140] | 0.0772 | no | +63 / −85 |
| mmlu 130000→130500 | +0.1211 pp | [−0.1709, +0.4130] | 0.4374 | no | +230 / −213 |
| mmlu 130500→131000 | +0.0855 pp | [−0.2208, +0.3846] | 0.6092 | no | +234 / −222 |

So **a single 500-step step cost 1.09 pp of TriviaQA EM, resolved at p = 0.0001**
(355 right→wrong vs 160 wrong→right of 17 944). Cluster 1 also contains a resolved
interval (triviaqa 124500→125000, **+0.2786 pp**, p = 0.0242) — in the *opposite*
direction, and that one is the seam-crossing interval.

**It is not output degeneracy.** Labelled diagnostic across 130000/130500/131000:
**0.000 % empty predictions** throughout, `top_constant_frac` **falling**
0.440 % → 0.435 % → 0.368 %, distinct predictions **rising** 9519 → 9457 → 9686.
The model is not collapsing. Only **45.50 %** of triviaqa prediction *strings* are
identical across that interval, so — as with keep14 — **EM is a coarse read on a
model whose outputs are being rewritten wholesale between adjacent saves.**

### 2.5 The tolerance this supplies

`best_minus_last_pp` is the false advantage **as actually realised** by reporting
the best of three neighbours instead of the last:

| axis | cluster 2 | cluster 1 |
|---|---|---|
| triviaqa | **+1.1202 pp** | 0.0000 pp |
| popqa | +0.2523 pp | +0.1192 pp |
| mmlu_content | 0.0000 pp | +0.1211 pp |

**Proposed neighbour tolerance: an accept must clear its threshold by more than
≈1.2 pp on triviaqa and ≈0.35 pp on popqa/mmlu_content, or be demonstrated on
its ±500-step neighbours.** These are one-arm numbers and should be widened if a
second arm is ever measured.

---

## 3. LEG B — the cross-arm replication **FAILS**

Both readings were pre-committed in the module docstring and in the JSON's
`prereg_readings_fixed_in_advance` **before the numbers existed**. The
non-replicating branch is the one that fired.

| interval | keep14fresh2 (archived) | **shortgpt16 (here)** |
|---|---|---|
| **popqa 128000→153500** | **−0.6729 pp**, CI [−0.9252, −0.4206], p = 0.0001, +122/−218 | **+0.0841 pp**, CI [−0.1542, +0.3224], **p = 0.5084**, +152/−140 |

**Opposite sign, and not resolved.** `REPLICATES = False`.

shortgpt16's full curve, for the record:

| axis | 128000 | 153500 | 200000 | margin monotone ↑? |
|---|---|---|---|---|
| triviaqa | −25.1076 | −25.1633 | −24.8292 | no |
| popqa | −15.6277 | −15.5295 | −15.7749 | no |
| mmlu_content | −5.8085 | −5.8654 | −5.6518 | no |
| *nq_open* | −13.4072 | −13.1856 | −12.8532 | yes |

shortgpt16 **does** have a resolved popqa regression — but on the **other**
interval (153500→200000, **−0.2383 pp**, CI [−0.4416, −0.0350], p = 0.0264,
+95/−129), and 2.8× smaller. So *mid-heal popqa regressions are not unique to
keep14*, but **the specific 128k→153.5k dip is**, and it does not generalise.

**Consequence, as pre-registered:** the rule-level claim must be **downgraded**
from a property of healing to *"there exists at least one arm on which a later
checkpoint scores worse."* That is still enough to make single-checkpoint accepts
unsafe — **one counterexample suffices for that** — but it may **not** be stated
as a general property. Leg A independently supplies a *second*, stronger
counterexample (a resolved 1.09 pp drop over 500 steps on a different arm), so the
**requirement survives even though this replication failed**.

Also worth noting: shortgpt16 is by far the **strongest** of these damaged arms
(mmlu_content recovery 61.8–62.7 %, margin only −5.65 pp) and still rejects.

---

## 4. Verification performed

1. **Protocol confirmed from the invocation, fail-closed.** `protocol_asserted()`
   runs **before anything is scored** and parses the drivers' own echoed lines out
   of **all three** launch logs (Leg B ran as two invocations; `--legB_driver_log`
   takes a comma-separated list and **every** entry is gated, so a per-invocation
   drift among the three Leg B cells cannot hide). All three echo
   `mmlu_bs=16 cb_bs=32` and per-axis `bs=32/32/16`; driver source defaults agree
   as corroboration. **Negative-tested twice:** a doctored log reading `cb_bs=48`
   produces `FATAL protocol deviation ... != {'cb_bs': 32, 'mmlu_bs': 16}` and a
   missing log produces `FATAL: driver log ... absent`; in **both** cases **no
   output file is written** (confirmed by `ls`).
2. **`add_bos is False` on all 27 result dirs**, asserted with `is False` —
   **never** `is not True`, which passes silently on `None`. `max_new_tokens == 32`
   asserted on all 18 generative dirs. `chat_template=False` established
   **structurally**: neither harness contains a chat-template code path (the only
   occurrence of the string is a docstring), so it cannot have been switched on.
3. **Shard integrity: 40 of 40 cells clean.** Index set **exactly {0..7}** (not a
   file count), merged *n* exactly `EXPECTED_N` (triviaqa 17944 / popqa 14267 /
   nq_open 3610 / mmlu 14042), **0 duplicate item_ids, 0 nan**, and identical
   `item_id` sequences across all 10 arms. **Negative-tested:** a hand-built
   7-of-8 popqa set is refused with
   `shard index set [0,1,2,4,5,6,7] != {0..7}`.
4. **Everything imported, nothing reimplemented.** `ni_rule`, `ratio_rule`,
   `load_shards`, `build_nulls`, `mmlu_content_norm_vec`, `qa_metric_vec`,
   `EXPECTED_N`, `AXES`, `DEMOTED_AXES`, `PREREG` from `pilot_zero_rule_disagreement`;
   `paired_bootstrap`, `TIE_CONVS`, `N_BOOT`, `SEED` from A03's
   `analyze_1b_knowledge_floor`; `ANCHOR`, `_load_arm`, `assert_aligned`,
   `d4_interface_degenerate`, `D2_RESIDUAL_FLOOR_PP`, `Z95_TWO_SIDED`, `D4_*`,
   `SD_RUN_1B_PP` from `a04_shallow_rung_ni_7b`. No metric, null, rule, guard or
   anchor re-derived.
5. **Bootstrap seeds disjoint from every archived cell.** New `arm_index`
   400–408; guard `SEED+1700+13·axis`; intervals `SEED+1900+13·axis+7·pair`
   (Leg A) and `SEED+2400+...` (Leg B). Archived offsets avoided: pilot_zero
   {0,1}, step100k 100–102, shallow_rung 200–203, keep14 trajectory 300–301 +
   endpoint 201, and the keep14 guard/interval offsets 700/900. **No archived
   number can be perturbed by this run.**
6. **Checkpoint identity proven, not assumed** — see §1.5. The driver asserts the
   loaded meta's `step`, `keep_front_layers`, `n_fresh_layers` and
   `num_hidden_layers` **before** spending 8 GPUs, and the merge-time assertion
   re-checks `ckpt_step` against the requested step from `summary.json`.
7. **Verdict is convention-invariant.** **0 NI accepts across all 5 tie
   conventions × 36 cells.** `RATIO(0.85)` also rejects all 9 checkpoints (mean
   ratios 0.315–0.343 keep8, 0.491–0.499 shortgpt16).

### 4.1 A reproducibility defect I found and did **not** paper over

Running this analysis on `.73` and `.82` — **same code, same shards, same
seeds** — gives margins differing in the 4th decimal. Diagnosed precisely:

- the per-item difference vector `d` is **bit-identical** on both nodes
  (sha256 `4d0d81b9…`, identical `(vals, counts) = ([-1,0,1], [8653,9092,199])`,
  identical `d.mean()` to 20 significant figures);
- the raw RNG **bit stream is identical** (`default_rng(seed).integers` and
  `.binomial` agree exactly);
- but **`Generator.multinomial(n, p, size=10000)` differs in 19 of 10 000 rows**
  between **numpy 2.5.1** (`.73`) and **numpy 2.4.6** (`.82`) — first divergence at
  row 2598: `[8655,9082,207]` vs `[8655,9115,174]`. The sampler changed between
  those versions.

**Measured effect:** 3 of 24 published NI cells move, **max |margin| drift
0.005294 pp**, **triviaqa only** (the axis whose *p*-vector has a rare third
category, 199/17944). **No verdict boolean, no `range_exceeds_item_noise`, and
not the headline changes.** The drift is **211× smaller** than Leg A's 1.1202 pp
finding and does not touch popqa or mmlu_content at all. **Within** a node the
output is **byte-identical** on re-run (sha256 `1f88d6eb…` twice), so this is
purely cross-version, never run-to-run.

Two reasons this is recorded rather than ignored:
1. The standing `same-harness-runs-bit-identical` rule is about the **scoring
   harness**; this is the **analysis** layer, and it is a genuine counter-example
   there. A future agent re-running on a different node must not read the
   4th-decimal disagreement as a data problem.
2. **`a04_keep14_trajectory_ni.py` hard-fails if the archived endpoint margins do
   not reproduce to 5e-4 pp — which is TIGHTER than this 5.3e-3 pp drift.** That
   assertion is therefore only guaranteed to pass on a node whose numpy matches
   the one that produced the archive. **This is a latent tooling hard-fail, not
   scientific drift.** Fixing it means pinning numpy cluster-wide, which is
   outside this dispatch.

**This may not be cited to explain away any move larger than ~0.006 pp.** Every
resolved interval reported here is 15–200× larger.

---

## 5. Licensed vs NOT licensed

### Licensed
- The 9 checkpoints' accuracies, nulls, residuals, Δ, lo95 bounds and margins in
  §2–§3, and the 0-accept verdict under all 5 tie conventions and RATIO(0.85).
- **"Across three adjacent (500-step) checkpoints of one clean, uninterrupted
  heal segment, the NI margin ranges by 1.1202 pp on triviaqa — 1.70× what item
  noise alone would produce — and by ≤0.34 pp, indistinguishable from noise, on
  every other axis."**
- **"A single 500-step interval cost 1.0867 pp of TriviaQA EM, resolved at
  p = 0.0001 (355 right→wrong vs 160 wrong→right of 17 944), and it is not output
  degeneracy"** (0 % empty, top-constant falling, distinct rising).
- **"The keep14 popqa 128k→153.5k regression does NOT replicate on shortgpt16"**
  (+0.0841 pp, p = 0.5084, opposite sign, CI straddles zero).
- "shortgpt16 has its own resolved popqa regression on 153.5k→200k
  (−0.2383 pp, p = 0.0264)" — a *different* interval, 2.8× smaller.
- The proposed neighbour tolerance in §2.5, **as a one-arm estimate**.

### NOT licensed
- ⛔ **Anything of the form "the 7B deficits are large relative to seed
  variance."** `sd_run` is a **1B-only** quantity (S = 3, keep12@5000). **Every**
  7B rung here has **exactly ONE seed**, and the historical 7B ladder's seeds are
  **unrecorded** (`--seed` postdates them; trainer `afdfa66` called no seeding
  function), so **no 7B `sd_run` is computable or retrospectively
  reconstructible.**
- ⛔ **Treating the checkpoints within a cluster as replicates of each other.**
  They are successive states of **one** optimisation; their spread is heal
  progress + data order. **The range measured here is a CHECKPOINT-SELECTION
  quantity, not an estimate of seed variance,** and the two must never be
  conflated — the whole point of §2.3 is that even *item* noise had to be
  subtracted before the range meant anything.
- ⛔ **Calling any of this "harness noise."**
  `full32_rescore_v2_20260812.correction_to_the_jitter_premise` established there
  is **no measured runtime-jitter floor** on this harness (same-code re-runs are
  bit-identical). These are different models, so bit-identity does not apply —
  but it also removes "noise" as an available explanation. Item-sampling
  variability is a different thing and **is** quantified here.
- ⛔ **Reporting the 7 sub-noise ranges (0.119–0.332 pp) as measured neighbour
  gaps.** They fail `range_exceeds_item_noise`; a max-minus-min of 3 noisy cells
  is biased upward even at zero true spread.
- ⛔ **Reading cluster 1 as an uninterrupted 500-step neighbourhood** (§1.2).
- ⛔ **Comparing keep8fresh2 / shortgpt16 / keep14fresh2 margins as rungs of one
  ladder** (§1.3). Three architectures; the two-corpora / unequal-steps
  `STATUS.json:warning` also still holds.
- ⛔ **Treating Leg B as a matched pairwise comparison with keep14** (§1.4) — same
  step number ≠ same data seen.
- ⛔ **Quoting any margin here to better than 0.01 pp across nodes** (§4.1).
- ⛔ Any K1/K2/K3 clause — defined over the pre-registered **1B** arm set.
- ⛔ Generalising the tolerance in §2.5 beyond one arm / two clusters.

---

## 6. What this changes for the certification rule

1. **The neighbour-robustness requirement is vindicated, and now has a number.**
   §7.2 asked for it on the strength of one 25 500-step interval. At the spacing
   where hand-picking actually happens (500 steps), the loophole is worth
   **1.1202 pp** on the widest decision axis — **49.9 % of popqa's entire Δ
   (2.2457 pp), and 60.2 % of mmlu_content's (1.8614 pp)** — and it is **not**
   noise (1.70× the noise-range floor). A future accept reported at a single
   checkpoint can be overstated by roughly this much.
2. **But it is axis-concentrated, not diffuse.** 7 of 8 decision-axis ranges are
   inside item noise. So the requirement should be stated as *"an accept must
   clear its threshold by more than the measured neighbour range on the axis it
   accepts on"*, not as blanket distrust of all single-checkpoint numbers. The
   per-axis tolerances are in §2.5.
3. **The mechanism is worse than "the margin wobbles."** The 1.09 pp TriviaQA drop
   is a **resolved, one-interval event** with 355 items going right→wrong, while
   only 45.5 % of prediction strings are even stable across that interval. The
   rule is reading a scalar off a model whose outputs churn wholesale between
   adjacent saves — so **which** checkpoint you grab is a live degree of freedom,
   not a rounding question.
4. **The popqa dip is arm-specific; do not build on it.** Leg B refutes the
   generalisation. Anyone citing keep14's popqa non-monotonicity as evidence about
   *healing* is over-reaching; cite it as **an existence proof about
   checkpoint selection**, which is all that is needed and all that is supported.
5. **Two infrastructure defects are now on the record** and both cost real time:
   a **corrupt zwfy6 checkpoint** that no existence-based inventory would flag
   (§1.1), and a **cross-node numpy bootstrap drift tighter than an existing
   hard-fail tolerance** (§4.1). The second one means the archived keep14
   reproduction assertion is node-dependent and will eventually fire for a reason
   that has nothing to do with science.

**Recommendation:** adopt the per-axis neighbour tolerance from §2.5 into
`A04_GATE_DESIGN.md` as a **pre-condition on any reported accept**, phrased
per-axis rather than blanket. Do **not** spend GPU widening this to more arms
until there is an accept to protect — the requirement is already established by
two independent counterexamples (keep14 popqa at 25.5k spacing, keep8 triviaqa at
500-step spacing), and a third would not change the gate. Separately, **pin numpy
across the cluster** before the next analysis that relies on the 5e-4 pp
reproduction assertion.

---

## 7. CORRECTION NOTE (appended 2026-08-13, after the fact — §6.1's original sentence is left INTACT above)

⚠️ **§6.1's "49.9 % of popqa's entire Δ (2.2457 pp), and 60.2 % of
mmlu_content's (1.8614 pp)" is a CROSS-AXIS division and must not be reused.**

Both figures divide a **triviaqa** range (1.1202 pp) by a **different axis's**
Δ. `ni_rule` only ever compares a margin to *its own* axis's `delta_pp` — the
per-cell `delta_pp` in `evidence/a04_neighbour_variability.json` confirms this
(`per_convention.split.delta_pp`: triviaqa 6.3291, popqa 2.2457,
mmlu_content 1.8614 pp). A triviaqa range is never weighed against popqa's Δ by
any rule in this gate.

**Same-axis (correct) values:**

| cell | axis | range / move (pp) | that axis's OWN Δ (pp) | % of own Δ |
|---|---|---|---|---|
| keep8 c2 (clean, 500-step) | triviaqa | 1.1202 | 6.3291 | **17.70 %** |
| keep8 c2 | popqa | 0.2523 | 2.2457 | 11.24 % *(sub-noise, not a measured gap)* |
| keep8 c2 | mmlu_content | 0.2208 | 1.8614 | 11.86 % *(sub-noise, not a measured gap)* |
| keep14 (25 500-step, §3 reference) | popqa | 0.6939 (margin) | 2.2457 | **30.90 %** |

Recompute: `1.1202/6.3291 = 0.1770`; `0.6939/2.2457 = 0.3090`.

**Scope of this correction.** It changes a *magnitude*, not a finding. Every
verdict boolean, `range_exceeds_item_noise`, the 1.703× noise-gate ratio, the
§2.5 tolerance in pp, the resolved 130500→131000 drop, and the Leg B replication
failure are **all unaffected** — none of them involves a Δ ratio. 17.7–30.9 % of
the quantity under test is still material.

Two downstream texts inherit the error and are corrected in place of record
rather than by editing this file:
- `A04_GATE_DESIGN.md` §2.0.2's empirical-basis bullet ("= 49.9 % of popqa's
  entire Δ") — same cross-axis error, same substitution.
- `PROPOSAL.md` §5 (rewritten 2026-08-13) now carries the same-axis numbers and
  retires 49.9 % / 60.2 % explicitly.

First flagged in
`../../shared/literature/MARGIN_TRAJECTORY_INSTABILITY_NOVELTY_20260813.md` §C1
(commit `75cf173`); independently recomputed from the canonical JSON on
2026-08-13 before this note was written.
