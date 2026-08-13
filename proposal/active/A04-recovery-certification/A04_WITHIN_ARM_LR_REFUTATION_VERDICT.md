# A04 — does the LR hypothesis survive a WITHIN-ARM contrast? (keep8+fresh2, cluster1 vs cluster2)

**Verdict string:** `UNRESOLVED_SUBNOISE__INADMISSIBLE_SEAM`

**Date:** 2026-08-13 · **GPU: 0.0000 GPU-h.** Pure CPU re-analysis of per-example shards
already on disk from `neighbour_variability_20260813` (whose 8.6556 GPU-h is counted
*there*, not here). No model was loaded. `nvidia-smi` on `.82` read **8 × 0 MiB before and
after**. **Not touched:** `LOCAL`/`.21` (SparseForge #246, 8 cards each), `.104` (paperC
Qwen3 heal), `.73` (idle, but numpy 2.5.1 — see below).
**Node:** `.82` (8×H20, zwfy6, **numpy 2.4.6**) — deliberately the *same node and numpy
version that published* `evidence/a04_neighbour_variability.json`, so the archive's own
`bootstrap_cross_node_drift` cannot be operating. `.73` was idle and available but has
numpy 2.5.1 and would have introduced the documented 19-of-10 000-row `multinomial` split.
**Pre-registration:** `A04_WITHIN_ARM_LR_PREREG.md`, commit **`5e9b6fb`**, its own commit,
**before the first canonical range for this comparison was computed.**
**Evidence:** `evidence/a04_within_arm_lr.json` (sha256 `2c354972…`, byte-identical on both
disks)
**Code:** `code/a04_within_arm_lr.py`

---

## 0. Answers, in one block

| Q | Answer |
|---|---|
| **Q1** — canonical **NI-margin** range, cluster1 (higher LR) vs cluster2 (lower LR), gated on `1.6926·σ` per cluster | **triviaqa: c1 0.2842 pp (0.44× its floor, FAILS) vs c2 1.1034 pp (1.68×, CLEARS).** popqa 0.1192 (0.20×) vs 0.2383 (0.40×), both fail. mmlu_content 0.1852 (0.29×) vs 0.2136 (0.34×), both fail. *nq_open 0.1939 (0.17×) vs 0.3047 (0.26×), both fail, demoted.* **1 of 6 decision-axis cells clears — and it is in cluster2, not cluster1.** |
| **Q2** — is H_LR refuted? | **`UNRESOLVED_SUBNOISE`.** cluster1's triviaqa range is **inside its own item-noise floor** (0.2842 < 0.6518), so it is **not a measurement** and the ratio of the two ranges is **undefined**. This **neither supports nor refutes** H_LR. Two further, independent blockers: the LR contrast between the clusters is only **1.112×** (vs the **3.804×** H_LR was fitted on — **3.99 %** of it on the excess-ratio scale), and **cluster1 straddles a resume seam**, so it was never admissible as a clean control. **MAIN's premise that this pair can answer Q2 does not hold.** |
| **Q3** — how should §2.5's tolerance read? | **Split the number's two jobs.** As a *reporting trigger* ≈1.2 pp (triviaqa) / ≈0.35 pp (popqa, mmlu_content) stands as a **measured upper bound over 3 arms and 3 clean clusters**. As a *threshold an accept must clear*, it is **unavailable** — the same statistic, same arm, same depth, ranges 0.28–1.10 pp across two schedule positions and 0.1951–1.2149 pp across arms. Full wording in §7. |

**The one-line scientific summary:** MAIN's within-arm contrast is a **good idea that this
particular pair of clusters cannot execute** — the higher-LR cluster's range is sub-noise,
its LR contrast is 4 % of the effect size H_LR was fitted on, and it is the cluster with the
resume seam. What the pair *does* establish, without any LR interpretation, is that
**checkpoint-selection exposure is not a constant of an arm**: on one arm, at two positions
6 000 steps apart, the 500-step triviaqa margin range differs by **3.88×** (0.2842 → 1.1034)
— which is itself the reason §2.5 cannot be a threshold.

---

## 1. Premises I was asked to check — and the two that killed the test

The dispatch explicitly invited rejection of its premises. Two of the four are fatal to Q2,
and **both were verifiable before any capability number was recomputed.**

### 1.1 ⛔ FATAL — cluster1's LR is only **1.112×** cluster2's, not a high-vs-low contrast

This is the decisive one, and it is arithmetic, not measurement. LR **measured from the
trainer's own log lines** (`[step N/200000] ... lr=`) and independently recomputed from the
schedule the trainer actually uses (`train_semantic_bottleneck_1b.get_lr`, cosine,
`base_lr=2e-5`, `min_lr=2e-6`, `warmup=150`, `max_steps=200000`, all four param groups
identical per the run's own `[optim] group` banner):

| step | logged `lr=` | recomputed from the cosine | log |
|---|---|---|---|
| 124000 | 7.690e-06 | 7.694378e-06 | `..._73.log` |
| 124500 | 7.630e-06 | 7.628686e-06 | `..._73.log` |
| 125000 | 7.560e-06 | 7.563203e-06 | `..._82.log` |
| 130000 | 6.920e-06 | 6.920705e-06 | `..._82.log` |
| 130500 | 6.860e-06 | 6.857776e-06 | `..._82.log` |
| 131000 | 6.800e-06 | 6.795104e-06 | `..._82.log` |

- **cluster1 mean LR 7.6288e-6; cluster2 mean 6.8579e-6 → ratio 1.1118×** (1.1124× on the
  recomputed values, 1.1309× taking max/min over all six points).
- The cross-arm contrast H_LR was generated on, **re-measured from those arms' own logs**
  (not copied): keep10 1.24e-5 at step90000, keep12 **3.26e-6** at step166000 → **3.8037×**,
  against a published range spread of **6.2271×**.
- On the excess-ratio scale the within-arm contrast is **(1.1118−1)/(3.8037−1) = 3.99 %** of
  the cross-arm one; on the log scale **7.93 %**; on the raw-ratio scale 29.23 %.

**Consequence, and it is a design fact, not a result:** these two clusters are **6 000 steps
apart on a 200 000-step cosine that has already decayed to ~⅓ of base LR**. They are not a
high-LR vs low-LR pair; they are two nearly-identical points on the tail of the schedule.
Under any smooth monotone LR effect the *predicted* range difference is small, so **a null
here cannot refute H_LR**, and a *large* difference would be far too large for the LR gap and
would indicate a **third factor** rather than confirm the hypothesis. **This was written into
the pre-registration §3 before the ranges were read.**

> ⚠️ **My own pre-registration got this number wrong and I am not quietly fixing it.**
> `A04_WITHIN_ARM_LR_PREREG.md` §3 says the within-arm contrast is "**~13 %**" of the
> cross-arm one. That matches **none** of the three defensible scalings (3.99 % / 7.93 % /
> 29.23 %). The prereg's *qualitative* conclusion — underpowered — is unaffected and is in
> fact **strengthened**, since the true fraction on both natural scales is *smaller* than
> 13 %. Recorded in `evidence/a04_within_arm_lr.json:power_statement.prereg_correction`
> rather than edited, because the prereg is committed and its arithmetic is part of the
> record.

### 1.2 ⛔ FATAL — cluster1 **does** straddle a resume seam, and the seam sits on the one interval that matters

The dispatch asked me to check this and to **stop if it held**. It holds. Re-verified from
the training logs directly (not from the archive's flag), by reconstructing which process
wrote each save:

| ckpt | written by | timestamp |
|---|---|---|
| step124000 | `.73` process (1 `[resume]` banner, from `step121000_full.pt`) | 2026-08-08 19:35:45 |
| step124500 | **same** `.73` process — which then **died at 20:26** (TCPStore) | 2026-08-08 20:24:40 |
| step125000 | **a DIFFERENT `.82` process**, resumed **from `step124500.pt`** | 2026-08-12 01:27:16 |
| step130000 / 130500 / 131000 | **all inside that one `.82` process** | 09:36:24 / 10:25:18 / 11:14:10 |

`.73`'s log saves `[121500 … 124500]`; `.82`'s saves `[125000 … 131000]`. Each log has
**exactly one** `[resume] loading ckpt` banner, and cluster2's three saves have **no banner
between the first and the last**. So **cluster2 is clean and cluster1 is not**, and my
independently-computed seam verdict **agrees with the archive's own flag on both clusters**
(asserted; a disagreement would have aborted).

Why a seam disqualifies: `train_olmo2_arch_probe2.py:1011-1019` does
`sampler.set_epoch(epoch); data_iter = iter(loader)` on resume with **no intra-epoch
fast-forward**. Optimizer state and RNG are restored; **the loader position is not**. The
124500→125000 interval therefore saw a **different data order** than an uninterrupted 500
steps.

**And the seam is not a generic caveat here — it lands exactly on the load-bearing number.**
cluster1's **only** resolved interval on any axis is **124500→125000** (+0.2786 pp,
p = 0.0338) — *the seam-crossing one*
(`post_hoc_supplement_resolved_intervals.per_axis.triviaqa.hi_lr_largest_resolved_is_the_seam_interval
= true`). So the single datum that could have carried a refutation is precisely the one the
seam invalidates.

### 1.3 ✅ Premise that held — the shards exist and are complete, and no new scoring was needed

MAIN was right that cluster1's per-checkpoint numbers are already on disk and unreported:
all 6 checkpoints × 4 axes are present in `olmo2_closedbook_results/` +
`olmo2_mmlu_content_results/` under `A04_7B_keep8f2_step{124000,124500,125000,130000,130500,131000}`.
**28 of 28 shard cells clean**: index set **exactly `{0..7}`** (a set, not a count), merged
`n` exactly `EXPECTED_N` (triviaqa 17944 / popqa 14267 / nq_open 3610 / mmlu 14042),
**0 duplicate `item_id`, 0 nan**, identical `item_id` sequences across all 7 arms. So the
comparison genuinely costs **0 GPU**.

### 1.4 ✅ Premise that held — MAIN's hand-computed **accuracy** ranges all reproduce

Every one of MAIN's eight hand-computed accuracy ranges reproduces the canonical value to
< 1e-3 pp:

| axis | MAIN c1 / canonical | MAIN c2 / canonical |
|---|---|---|
| triviaqa | 0.2786 / **0.278645** | 1.1090 / **1.109006** |
| popqa | 0.1192 / **0.119156** | 0.2453 / **0.245321** |
| mmlu_content | 0.1852 / **0.185159** | 0.2065 / **0.206523** |
| nq_open | 0.1939 / **0.193906** | 0.3324 / **0.332410** |

**But this does not rescue the inference**, and MAIN was right to flag it: the accuracy range
is **a different statistic from the decision-bearing margin range**, and — more importantly —
**MAIN's version carried no noise gate at all.** Gated, three of MAIN's four "c1 vs c2"
comparisons are **undefined** and the fourth (triviaqa) is undefined *because c1 is
sub-noise*. The direction MAIN reported (0.25× on triviaqa) is arithmetically right and
**inferentially void**.

---

## 2. Q1 — the canonical margin ranges, per cluster, each gated by its OWN σ

`split` convention. `margin_pp = diff_lower95_one_sided_pp + delta_pp`, from the **imported**
`ni_rule`; nulls from the **imported** `build_nulls`; Δ from the imported guard, **never
substituted**; anchor = vanilla `models/OLMo-2-1124-7B` imported from
`a04_shallow_rung_ni_7b.ANCHOR`. All four axes **CERTIFIABLE** under guard D1–D6 (0 of 28
decision cells retired). Δ (split): triviaqa 6.3291, popqa 2.2457, mmlu_content 1.8614,
nq_open 1.9945 pp.

### 2.1 cluster1 = 124000 / 124500 / 125000 — **higher LR (7.63e-6), resume seam**

| axis | margins (pp) | **range** | floor `1.6926·σ` | ratio | clears? | % of own Δ |
|---|---|---|---|---|---|---|
| **triviaqa** | −43.9384 / −44.1390 / −43.8548 | **0.2842** | 0.6518 | 0.44× | **no** | 4.49 % |
| popqa | −19.1743 / −19.1743 / −19.2935 | 0.1192 | 0.5914 | 0.20× | no | 5.31 % |
| mmlu_content | −11.7763 / −11.5911 / −11.6695 | 0.1852 | 0.6497 | 0.29× | no | 9.95 % |
| *nq_open* | −16.1496 / −16.3435 / −16.2881 | 0.1939 | 1.1496 | 0.17× | no | 9.72 % |

### 2.2 cluster2 = 130000 / 130500 / 131000 — **lower LR (6.86e-6), clean, the archived headline**

| axis | margins (pp) | **range** | floor `1.6926·σ` | ratio | clears? | % of own Δ |
|---|---|---|---|---|---|---|
| **triviaqa** | −41.4250 / −41.4473 / −42.5284 | **1.1034** | 0.6575 | **1.68×** | **YES** | **17.43 %** |
| popqa | −18.0739 / −18.1440 / −18.3122 | 0.2383 | 0.5890 | 0.40× | no | 10.61 % |
| mmlu_content | −11.7692 / −11.6481 / −11.5555 | 0.2136 | 0.6351 | 0.34× | no | 11.48 % |
| *nq_open* | −15.8726 / −15.7341 / −16.0388 | 0.3047 | 1.1591 | 0.26× | no | 15.28 % |

### 2.3 The noise-gate constant, re-derived rather than trusted

**Both clusters are k = 3, so `1.6925687506432689 = 3/√π` is the correct constant for both**
and no k=5 or k=8 constant enters anywhere. Because a k-mismatch flipped a boolean elsewhere
in this repo on 2026-08-13, the script **re-derives** the constants instead of asserting
them, and aborts before writing anything if any check fails:

- closed forms checked against the table: `2/√π = 1.1283791670955126`,
  `3/√π = 1.6925687506432689`;
- **Monte Carlo, 600 000 draws each**: k=2 → 1.1284, k=3 → **1.6929**, k=5 → 2.3266
  (lit. 2.325929), k=8 → 2.8465. (k=5/k=8 are computed **only** to demonstrate that 1.6926
  is specific to k=3.)
- `range_report` is shown to take k **from its input length** (a 2-point call uses 1.1284);
- and the mistake is made **explicit and quantified**: a range of 1.50 against mean SE 1.00
  gives `gate=False` under the correct k=3 floor (1.6926) and `gate=True` under a wrong k=2
  floor (1.1284) — **`the_boolean_flips = true`**.

**σ is per cluster, and pooling is impossible by construction.** cluster1's mean bootstrap SE
is 0.385101 pp → floor 0.651809; cluster2's is 0.388489 → floor 0.657544. `range_report` is
called **once per cluster with that cluster's own SE list**, so one cluster's σ can never gate
the other. Proven by an executed self-test: two triples with *identical* ranges but different
SEs must yield **different floors and different booleans** — they do (gate `True` vs `False`).

---

## 3. Q2 — the verdict: `UNRESOLVED_SUBNOISE`, with an `INADMISSIBLE_SEAM` modifier

The label is emitted **mechanically** by the script from the criteria frozen in the prereg;
it was not chosen after seeing the table.

```
R = range(cluster1_HIGHER_LR) / range(cluster2_LOWER_LR)   on triviaqa
R = 0.2842 / 1.1034 = 0.2576
both_clusters_clear_their_own_gate = FALSE   (cluster1: 0.2842 < 0.6518)
=> UNRESOLVED_SUBNOISE          (prereg: ">=1 cluster fails its own gate => the
                                 ratio of two ranges is undefined; NEITHER
                                 support NOR refutation")
+ INADMISSIBLE_SEAM             (cluster1's resume seam confirmed from the logs)
```

**Read literally, as the pre-registration requires:** `R = 0.2576` *is* in H_LR's opposite
direction — the higher-LR cluster is 3.88× **narrower**. But **cluster1's range is inside its
own item noise**, and a max-minus-min of 3 noisy cells is biased upward *even at zero true
spread*, so 0.2842 pp is not an estimate of anything. **Dividing a measurement by a
non-measurement does not yield a refutation.** Per the prereg's `banned_rewordings`, this may
**not** be written up either as "consistent with noise, so H_LR is fine" or as "the direction
is reversed, so H_LR is dead".

**Three independent reasons the refutation MAIN hoped for is unavailable**, any one of which
suffices:

1. **Sub-noise.** cluster1 fails its own gate on **all four** axes (0.17–0.44× floor).
2. **Underpowered by construction.** 1.112× LR contrast vs the 3.804× H_LR was fitted on
   (**3.99 %** on the excess-ratio scale). Even a real, strong LR effect would be
   near-invisible at this contrast.
3. **Seam.** cluster1 is not a clean neighbourhood, and its **only** resolved interval **is**
   the seam-crossing one.

**So MAIN's design premise does not hold, and this is the answer to "tell me if it can't
answer Q2".** The two clusters are the right *idea* — same arm, same depth, same corpus, only
schedule position differs — but this specific pair supplies a 1.11× contrast on the decayed
tail of the cosine, and the earlier member is the seam-damaged one. **H_LR remains an
untested n=3 hypothesis. It is neither strengthened nor weakened by this run.**

### 3.1 The label is not an artefact of the bootstrap seed

The verdict is recomputed under **both** available bootstrap seed choices (this run's mandated
`arm_index` 1000–1005 and the archive's 400–405): `R = 0.2576` vs `0.2388`, `both_clear =
False` in both, **label identical**. `label_is_seed_invariant = true`; the script **aborts**
if it is not.

### 3.2 POST-HOC supplement — flagged as post-hoc, and it does not change the verdict

The pre-registered statistic is the *range*, which is undefined here. The **adjacent-interval
paired bootstrap** is a proper test (own CI, own p) and *is* defined regardless of the gate,
so suppressing it merely because the pre-registered statistic came out undefined would itself
be a selection effect. It is therefore reported — and **flagged `IS_POST_HOC: true`, not in
the prereg, does not set the label**:

| cluster | resolved 500-step triviaqa intervals | largest |
|---|---|---|
| cluster1 (hi LR) | 1 of 2 — **124500→125000**, +0.2786 pp, p = 0.0338 | 0.2786 pp |
| cluster2 (lo LR) | 1 of 2 — 130500→131000, **−1.0867 pp**, p = 0.0001 | 1.0867 pp |

No other axis has a resolved interval in either cluster. `R = 0.2564`, again opposite to
H_LR — **and again inadmissible**, because cluster1's one resolved interval is exactly the
seam-crossing one. The two defects are **not** redundant: post-hocness alone blocks it, and
the seam alone blocks it.

**What the supplement does license, with no LR interpretation:** within one arm, at two
positions 6 000 steps apart on one schedule, the largest resolved adjacent-500-step triviaqa
move differs by **~3.9×**. That is a statement about the **spread of the statistic across
positions**, not about which position is larger, so it survives the seam caveat — and it is
the empirical basis for Q3.

---

## 4. A reproducibility defect I found and did not paper over

**The margin range is reproducible to ~0.03 pp across bootstrap seed choices, not to the
5e-4 pp that an existing hard-fail assertion demands.** This is a *second*, independent
latent tooling hard-fail alongside the numpy one already recorded in
`A04_NEIGHBOUR_VARIABILITY_VERDICT.md` §4.1.

Recomputing both clusters on the same node, same numpy, same shards, same code:

| cell | recomputed range | archive range | drift |
|---|---|---|---|
| c1 triviaqa | 0.284218 | 0.267499 | 0.016719 |
| c1 mmlu_content | 0.185159 | 0.213645 | **0.028486** |
| c1 nq_open | 0.193906 | 0.221607 | 0.027701 |
| c2 triviaqa | 1.103433 | 1.120152 | 0.016719 |
| c2 popqa | 0.238312 | 0.252331 | 0.014018 |
| c2 mmlu_content | 0.213645 | 0.220766 | 0.007121 |
| c2 nq_open | 0.304709 | 0.332410 | 0.027701 |
| c1 popqa | 0.119156 | 0.119156 | 0.000000 |

**Diagnosed exactly, and it is not the numpy split.** Both this run and the archive are on
`.82` / numpy 2.4.6, so the documented cross-version drift (0.005294 pp max, **triviaqa
only**) cannot be operating — and these drifts reach 0.0285 pp and touch `mmlu_content` and
`nq_open`. The cause is the **mandated bootstrap seed change**:

- The **bootstrap-free** statistic reproduces **exactly**: all **8 of 8** accuracy ranges
  match the archive to < 5e-4 pp (see §1.4). This proves the item set, metric, null, Δ and
  loaded shards are identical. **The script hard-fails if this is not exact** — that failure
  would mean the data changed, which is not a seed effect and would block publication.
- `ni_rule`'s seed is `SEED + 97·arm_index + 13·axis`. `assert_seeds_disjoint` **forbids**
  reusing the archive's `arm_index` 400–405 (if this run took them, later re-running the
  archive could silently perturb published numbers), so this run must use 1000–1005 — a
  different bootstrap draw, hence a different 5th percentile.
- **Executed proof, not assertion:** re-running the same 24 cells with the **archive's own**
  `arm_index` reproduces the archive **exactly — 24 of 24, max drift 0.0 pp.**

**So the disjointness rule and bit-exact margin reproduction are mutually exclusive for this
estimator.** The rule wins; the cost is ~0.03 pp of margin precision. **Every gate boolean is
unchanged**, and the quantities compared here (0.2842 vs 1.1034 pp) are **10–40× the seed
drift**. But any future assertion that hard-fails on a 5e-4 pp *margin* reproduction is valid
only at **fixed `arm_index`**, and will eventually fire for a reason that has nothing to do
with science.

**This may not be cited to explain away any move larger than ~0.03 pp.**

### 4.1 A weaker-than-it-looks check I strengthened

`assert_seeds_disjoint` scans one `evidence/` directory. **zwfy6's copy is missing 12 JSON
archives that wzc1 has** — including `a04_sigma_run_postfix.json` (holds `arm_index`
900–902) and `a04_step100k_plateau_vs_ni.json` (100–102). Scanning only the compute node's
disk would have passed on a collision with an archive that exists only on the other disk. The
script therefore takes `--extra_evidence_dirs` and **both disks' evidence sets were scanned**
(the 12 wzc1-only files staged to `.82` with md5 verified), with **`arm_index` 1000–1005,
guard 6700, interval 6900** confirmed disjoint from all of `{0,1}`, 100–102, 200–203,
300–301, 400–408, 500–503, 600–610, 700–702, 800–801, 900–902.

---

## 5. Verification performed

1. **Protocol from the INVOCATION, fail-closed.** `mmlu_bs=16 cb_bs=32` parsed from the
   `DRIVER START` header of `logs/a04_nbr_keep8_legA.out` plus every per-axis
   `... START ... bs=` line; driver source defaults as corroboration only. **Never** from
   `summary.json:meta`, which records neither `batch_size` nor `chat_template`
   (`A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md`). Any deviation → no output file.
2. **`add_bos is False` on all 18 result dirs**, asserted with **`is False`** — never
   `is not True`, which passes silently on `None`. `max_new_tokens == 32` on all 12
   generative dirs. `chat_template=False` established **structurally**: neither harness
   contains a chat-template code path (the only occurrence of the string is a docstring),
   so it cannot have been switched on. OLMo-2 is a BASE LM.
3. **Shard integrity: 28 of 28 cells clean.** Index set **exactly `{0..7}`** as a *set*,
   merged `n` exactly `EXPECTED_N`, **0 duplicate `item_id`, 0 nan**, identical `item_id`
   sequences across all 7 arms.
4. **Everything imported, nothing re-implemented.** `build_nulls`, `ni_rule`, `ratio_rule`,
   `AXES`, `DEMOTED_AXES`, `EXPECTED_N`, `PREREG` from `pilot_zero_rule_disagreement`;
   `paired_bootstrap`, `TIE_CONVS`, `N_BOOT`, `SEED` from A03's
   `analyze_1b_knowledge_floor`; `ANCHOR`, `_load_arm`, `assert_aligned` from
   `a04_shallow_rung_ni_7b`; and `range_report`, `guard_cell`, `protocol_asserted`,
   `shard_integrity_report`, `adjacent_interval_tests`, `output_shape_and_flips`,
   `EXPECTED_RANGE_OVER_SD`, `LEG_A_CLUSTERS`, `ARM_ARCH` from `a04_neighbour_variability`
   — the same code objects that produced the archived keep8 numbers. The **fixed,
   self-excluding** `assert_seeds_disjoint` is taken from
   `a04_keep12_trajectory_monotonicity` **unweakened** (and it correctly refused a re-run to
   a *different* filename, which is the collision it is for).
5. **Estimator self-tests run before anything is read**, and abort on failure: the gate
   constants (§2.3), the k-mismatch boolean flip, and per-cluster σ non-pooling.
6. **The archive is read back and asserted, not quoted.** Both clusters' 8 margin ranges and
   8 gate booleans are checked against recorded literals (±5e-4 pp), and
   `cluster2.resume_seam is False` is asserted. A drifted archive raises.
7. **LR measured, never copied.** Parsed from the trainer's own `lr=` lines in **both**
   training logs, cross-checked against the trainer's own cosine at the logs' 3-sig-fig
   precision (tolerance *derived* from the printed precision, not guessed), and checked
   against recorded literals. A step logging two different LRs in two logs would be fatal.
   Cross-arm LRs likewise re-measured from keep10's and keep12's own logs.
8. **Seam reconstructed from the logs**, not read from the archive's flag, and the two are
   asserted to agree.
9. **Within-node determinism:** re-running to the same output path gives **0 statistical
   differences** (the only delta is the self-exclusion bookkeeping entry naming its own file).

---

## 6. Licensed vs NOT licensed

### Licensed
- The 6 checkpoints' accuracies, nulls, residuals, Δ, lo95 bounds, margins and the per-axis
  ranges + gate booleans in §2, under the `split` convention, on `.82`/numpy 2.4.6.
- **"cluster1's 500-step margin range fails its own item-noise gate on all four axes
  (0.17–0.44× floor), so the higher-LR cluster supplies no measurement to compare."**
- **"The two clusters differ in LR by only 1.112×, which is 3.99 % (excess-ratio scale) of
  the 3.804× cross-arm contrast H_LR was fitted on."**
- **"cluster1 straddles a resume seam, and its only resolved interval IS the seam-crossing
  one."**
- **"Within one arm, at two positions 6 000 steps apart, the 500-step triviaqa margin range
  differs by 3.88× (0.2842 → 1.1034 pp) — so checkpoint-selection exposure is not a constant
  of an arm."** (Note: only the larger of the two is above noise; the claim is about the
  *spread of the statistic*, and the smaller value is an **upper bound** on cluster1's true
  range, which is what makes the spread statement safe.)
- The seed-mechanism diagnosis in §4, including the executed 24-of-24 reproduction.

### NOT licensed
- ⛔ **"H_LR is refuted."** The verdict is `UNRESOLVED_SUBNOISE`. Three independent blockers
  (§3).
- ⛔ **"H_LR is confirmed" or "supported."** n=3 arms plus a 1.11× within-arm contrast cannot
  confirm a schedule law.
- ⛔ **Quoting `R = 0.2576` (or the post-hoc `0.2564`) as an effect size.** Its numerator is
  sub-noise; the ratio is **undefined** and is reported only because the prereg fixed the
  formula in advance.
- ⛔ **Treating the two clusters, or the checkpoints within one, as REPLICATES.** They are
  successive states of one optimisation; the range is a **checkpoint-SELECTION** quantity,
  never seed variance. No 7B `sd_run` exists or is reconstructible (one seed per rung;
  historical seeds unrecorded).
- ⛔ **Promoting cluster1 to a clean 500-step neighbourhood** (§1.2).
- ⛔ **Using the ACCURACY range in place of the MARGIN range** for any gate-design statement,
  even though MAIN's accuracy arithmetic checks out (§1.4).
- ⛔ **Reporting the 7 sub-noise ranges (0.119–0.305 pp) as measured neighbour gaps.**
- ⛔ **Citing the numpy multinomial split to explain the drifts in §4** — both runs are on the
  same node and numpy; the cause is the seed, and it is proven.
- ⛔ **Quoting any margin to better than 0.01 pp across nodes, or better than ~0.03 pp across
  seed choices.**
- ⛔ Any K1/K2/K3 clause — those are defined over the pre-registered **1B** arm set.
- ⛔ Comparing keep8 / keep10 / keep12 / keep14 / shortgpt16 margins as rungs of one ladder
  (different architectures).

---

## 7. Q3 — how `A04_GATE_DESIGN.md` §2.5's tolerance should read

MAIN listed three mutually inconsistent candidates. They are reconcilable, and the
reconciliation is **not** a compromise number — it is the observation that **§2.5's number is
being asked to do two different jobs, and it is valid for one and invalid for the other.**

The three candidates and what each is actually right about:

| candidate | source | what it is right about |
|---|---|---|
| "≈1.2 pp, unchanged, now on two arms" | `keep10_neighbour_range_20260813` §7.1 | the **largest** clean measured range replicates across arms to 8 % (1.1202 → 1.2149 pp). As an estimate of *how much hand-picking can buy at worst*, it is solid. |
| "conditional — 6.2× cross-arm spread" | `keep12_trajectory_monotonicity_20260813` §5 | the **range is not a constant**: 0.1951–1.2149 pp across three arms. So it cannot be a floor an accept must clear. |
| "even within one arm, two positions differ ~4×" | **this run**, §3.2 / §6 | the variation is **not even a property of the arm**. It is positional. So no per-arm constant is defensible either. |

All three are true simultaneously, because the first is about an **upper bound** and the
other two are about **dispersion**. Proposed wording:

> ### §2.5 (proposed replacement)
>
> **The neighbour precondition is a REPORTING requirement, not a threshold.** Any `NI(Δ)`
> accept must be reported together with the same axis's margin at the immediately adjacent
> saved checkpoints on both sides (or a statement that none exist). An accept whose axis
> moves by more than the measured neighbour range on that axis, without the neighbours also
> accepting, is reported as **checkpoint-selection dependent, not as a certified recovery.**
>
> **The measured neighbour range is an UPPER BOUND, and it may not be used as a
> pass/fail threshold.** Largest clean 500-step values observed, per axis, over **three arms
> and three seam-free clusters** (keep8 c2, keep10, keep12 Q4): **≈1.2 pp on triviaqa** and
> **≲0.35 pp on popqa / mmlu_content**. These are the *maxima*, and the statistic is
> **dispersed on at least two axes of variation**:
> * **across arms** — triviaqa 0.1951 pp (keep12) to 1.2149 pp (keep10), a **6.2× spread**;
> * **within one arm, across schedule position** — keep8 triviaqa 0.2842 pp (step124–125k)
>   to 1.1034 pp (step130–131k), a **3.9× spread**, at only a 1.11× LR difference.
>
> Consequently **no single number — per gate, per arm, or per LR — is defensible as a
> tolerance to clear.** The rule in force is *"report the neighbours and their gate
> verdicts"*. The ≈1.2 / ≈0.35 pp figures may be used **only** (a) to decide *whether a
> reported move is worth flagging*, and (b) to bound *how much* a hand-picked accept could
> have been overstated. They may **not** be used to license an accept that clears them.
>
> **Every range quoted anywhere in this gate must carry its own item-noise gate**
> `range > E[range of k iid N(0,σ)] · mean(bootstrap SE)`, with the constant matched to **k**
> (k=2 → 1.128379, k=3 → 1.692569, k=5 → 2.325929) and σ computed from **that cluster's own**
> SEs. A range below its gate is **not a measurement** and may not be quoted as one, compared
> to another range, or used in a ratio. **Census over the 9 decision-axis range cells from
> the three CLEAN 500-step clusters (keep8 c2, keep10, keep12 Q4): exactly 2 clear their
> gate, and both are triviaqa** (keep8 c2 1.1202 pp @1.70×; keep10 1.2149 pp @1.84×).
> Adding keep8's seam cluster c1 makes it **2 of 12**. The phenomenon is real but
> **axis-concentrated on triviaqa** and at or below the noise floor everywhere else, so
> blanket distrust of single-checkpoint numbers remains unsupported.
>
> **Not conditioned on LR.** The LR hypothesis of
> `keep12_trajectory_monotonicity_20260813` §5 remains an **untested n=3 observation**: the
> only available within-arm contrast (`within_arm_lr_refutation_20260813`) is
> `UNRESOLVED_SUBNOISE` at a 1.11× LR contrast and is additionally seam-damaged. An LR column
> in this section would not be supported.

**Why this can be adopted without new GPU:** it *weakens* what the gate may claim (removes a
threshold reading that was never measured) and *keeps* every number that was measured. It
also removes the temptation the three-candidate conflict creates — picking whichever
magnitude is convenient for a given accept.

---

## 8. What this changes, and what to do next

1. **§2.5 should be rewritten per §7** — the split between "reporting trigger (upper bound)"
   and "certification threshold (unavailable)". This is the main deliverable.
2. **H_LR should be recorded as UNTESTED, not as "supported by 3 arms".** The keep12 verdict's
   §5 already says n=3 is a hypothesis; this run adds that **the cheap within-arm test does
   not exist on disk** — the only candidate pair supplies a 1.11× contrast and is
   seam-damaged. Testing H_LR properly requires **training**: a 500-step triple early in some
   arm's schedule (LR ≥ ~1.5e-5). Given that §2.5's wording no longer depends on the answer,
   **that training is not worth funding now** — the gate text in §7 is correct whether H_LR is
   true or false.
3. **Two latent hard-fails are now on the record**, both node/seed dependent and neither
   scientific: numpy-`multinomial` at 5.3e-3 pp (already recorded) and **bootstrap-seed choice
   at ~3e-2 pp (new, §4)**. Any assertion demanding 5e-4 pp margin reproduction is valid only
   at fixed node **and** fixed `arm_index`. Pinning numpy cluster-wide fixes only the first.
4. **`assert_seeds_disjoint` should always be given both disks' evidence dirs** (§4.1). zwfy6
   is missing 12 archives; a single-disk scan is weaker than it appears.
5. **MAIN's own dispatch numbers are all arithmetically correct** (§1.4) — the failure was not
   arithmetic but **the missing noise gate**: three of the four comparisons are undefined once
   gated, and the fourth is undefined because its numerator is sub-noise. Worth recording as
   the recurring failure mode: *the gate, not the arithmetic, is what these range comparisons
   keep getting wrong.*

---

## 9. Provenance — every number above is recomputable

| claim | source | how |
|---|---|---|
| all per-cluster margin ranges + gates | `evidence/a04_within_arm_lr.json` | `per_cluster.<cluster>.per_axis.<axis>.margin_range` (`range_pp`, `expected_range_if_pure_noise_pp`, `range_exceeds_item_noise`) |
| Q1 side-by-side + R | same | `per_axis_range_comparison`, `primary_axis_decision_inputs`, `R_hi_lr_over_lo_lr` |
| the verdict label and its mechanical derivation | same | `headline_verdict`, `verdict_label`, `prereg.criteria` |
| label is seed-invariant | same | `verdict_label_seed_sensitivity.per_seed`, `.label_is_seed_invariant` |
| LR per step, logged **and** recomputed | same | `lr_measured_from_logs.per_step.<step>` |
| LR contrast 1.1118× / power fractions | same | `lr_by_cluster.contrast`, `power_statement` |
| prereg's "~13 %" retracted | same | `power_statement.prereg_correction` |
| cross-arm LR 3.8037× (re-measured) | same | `crossarm_lr_measured`, `power_statement.crossarm_lr_ratio_keep10_over_keep12_measured` |
| seam reconstruction | same | `seam_verification.per_log`, `.per_cluster` |
| MAIN's accuracy ranges reproduce | same | `MAIN_hand_arithmetic_check.per_axis` |
| the ~0.03 pp seed drift + its proof | same | `reproduction_vs_archive`, `seed_mechanism_control` (24/24 at 0.0 pp) |
| gate constants re-derived | same | `selftest_gate_constants` (closed forms + MC k=2,3,5,8 + the boolean-flip demo) |
| σ is per cluster | same | `selftest_sigma_per_cluster`, `per_cluster.*.per_axis.*.sigma_is_this_clusters_own` |
| post-hoc resolved intervals | same | `post_hoc_supplement_resolved_intervals` |
| protocol / shards / guard | same | `protocol_asserted`, `shard_integrity_explicit`, `guard_D1_D6` |
| archive literals asserted | same | `archive_readback` |
| §7's 6.2× cross-arm spread | `A04_KEEP12_TRAJECTORY_MONOTONICITY_VERDICT.md` §5 + `evidence/a04_keep10_neighbour_range.json` | 1.2149 / 0.1951 |
