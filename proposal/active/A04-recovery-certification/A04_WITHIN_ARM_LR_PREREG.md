# A04 — PRE-REGISTRATION: does the LR hypothesis survive a WITHIN-ARM contrast? (keep8+fresh2, cluster1 vs cluster2)

**Written and committed BEFORE the first canonical margin range for this comparison was
recomputed.** GPU budget: **0**. CPU-only re-analysis of per-example shards already on
`zwfy6`. No model is loaded; no node's GPU is touched.

**Node of record for the statistics:** `.82` (8×H20, zwfy6, **numpy 2.4.6**) — the *same
node and numpy version that published* `evidence/a04_neighbour_variability.json`
(`bootstrap_cross_node_drift.published_on_node = ".82 (numpy 2.4.6)"`). Chosen so this
recomputation is comparable **bit-for-bit** with the archive it re-reads, rather than
inheriting the known 19-of-10 000-row `multinomial` split between numpy 2.4.6 and 2.5.1.
`.73` is available and idle but has numpy 2.5.1 and is therefore **wrong for this job**.

---

## 1. The hypothesis under test, stated so it can be false

`A04_KEEP12_TRAJECTORY_MONOTONICITY_VERDICT.md` §5 generated, from **n = 3 arms**:

> **LR hypothesis (H_LR).** "Checkpoint-to-checkpoint margin scatter is governed by where
> you are on the LR schedule." Supporting table: keep10 (LR 1.24e-5, triviaqa 500-step
> range 1.2149 pp) > keep8 (6.80e-6, 1.1202 pp) > keep12 (3.25e-6, 0.1951 pp) — the three
> arms rank-order **identically** by LR and by range, while `keep_front` **does not**
> order them.

That verdict recorded the confound itself: in these runs "later" is the same thing as
"lower LR", so LR cannot be separated from step count, epoch position or depth. It
proposed a ~3.5 GPU-h probe: an *early, high-LR* 500-step triple on keep12.

**MAIN has established that probe cannot be run without new training.** Verified
2026-08-13 by `ls` on zwfy6: keep12's earliest checkpoint is step124000; keep8's only
early checkpoint is step45000 and has no neighbour; keep10 starts at 83500. **No arm has
an early 500-step triple.**

**What this pre-registration tests instead.** `evidence/a04_neighbour_variability.json`
Leg A scored **two** keep8 clusters, not one:

| cluster | steps | reported as |
|---|---|---|
| `cluster1_124000_125000` | 124000 / 124500 / 125000 | reported, **not** headline (resume seam) |
| `cluster2_130000_131000` | 130000 / 130500 / 131000 | `leg_A_clean_cluster`, the headline |

Both are the **same arm, same depth (keep_front=8, 10 layers, 113 tensors), same corpus,
same repair mode, same protocol, same harness** — differing only in **position on the
cosine schedule**. If H_LR is a schedule effect, the *earlier / higher-LR* cluster1 must
show the **larger** range.

---

## 2. What is fixed before any number is looked at

### 2.1 Imports — nothing re-implemented

`build_nulls`, `ni_rule`, `ratio_rule`, `AXES`, `DEMOTED_AXES`, `EXPECTED_N`, `PREREG`
from `pilot_zero_rule_disagreement`; `paired_bootstrap`, `TIE_CONVS`, `N_BOOT`, `SEED`
from A03's `analyze_1b_knowledge_floor`; `ANCHOR`, `_load_arm`, `assert_aligned` from
`a04_shallow_rung_ni_7b`; **and, critically, `range_report`, `guard_cell`,
`protocol_asserted`, `shard_integrity_report`, `adjacent_interval_tests`,
`EXPECTED_RANGE_OVER_SD`, `LEG_A_CLUSTERS` from `a04_neighbour_variability`** — i.e. the
*same code objects* that produced the archived keep8 numbers. Δ and the anchor are
**never substituted** (guards G0/G2).

### 2.2 The statistic is the **NI margin** range, not the accuracy range

`margin_pp = diff_lower95_one_sided_pp + delta_pp`. MAIN's dispatch note computed
**accuracy** ranges by hand; those are a *different statistic* (keep10's triviaqa is
margin 1.2149 vs accuracy 1.2093) and are **not** what §2.0.2 or §2.5 are written on.
Both are reported here; **only the margin range is decision-bearing.**

### 2.3 Noise gate constant — pinned to k, and re-derived

`E[range of k iid N(0,σ)] / σ`: **k=3 → 3/√π = 1.6925687506432689**; k=2 → 2/√π =
1.1283791670955126. Both clusters are **k=3**, so **1.6926 is the correct constant for
both** and no 5-point/8-point constant enters. This is re-derived by Monte Carlo in the
script's self-test (not merely asserted), because on 2026-08-13 an 8-point grid was
gated with the k=3 constant and that mistake moved a floor by 40.6 % and flipped a
boolean. **σ is computed per cluster from that cluster's own bootstrap SEs** — one
cluster's σ may never gate the other.

### 2.4 Protocol, integrity, seeds

- Protocol read from the driver log `logs/a04_nbr_keep8_legA.out` header
  (`mmlu_bs=16 cb_bs=32`) plus per-axis `START ... bs=` lines, **never from
  `summary.json:meta`** (which records neither `batch_size` nor `chat_template`).
  Fail-closed: any deviation → no output file.
- `add_bos` asserted with **`is False`** (never `is not True`, which passes on `None`);
  `max_new_tokens == 32` on generative dirs. `chat_template=False` established
  **structurally** (no chat-template code path exists in either harness).
- Shard **index set exactly `{0..7}`** (a set, not a count), merged `n` exactly
  `EXPECTED_N`, **0 duplicate `item_id`, 0 nan** — for all 6 checkpoints × 4 axes.
- Bootstrap seeds: `arm_index` base **1000**, guard offset **6700**, interval offset
  **6900** — verified disjoint from every archived offset by the **self-excluding**
  `assert_seeds_disjoint` from `a04_keep12_trajectory_monotonicity.py`, unweakened.
  Archived: 0/1, 100–102, 200–203, 300–301, 400–408, 500–503, 600–610, 700–702,
  800–801, 900–902.

### 2.5 Seam check is a HARD PRECONDITION, verified before the range is read

`cluster1` is **known** to straddle a resume seam
(`a04_neighbour_variability.json:leg_A_neighbour_variability.cluster1_124000_125000.resume_seam
= true`). This pre-registration therefore **re-verifies it from the training logs
directly** and treats the outcome as a gate:

- `logs/olmo2_7B_keep8fresh2_resume200k_73.log` — one `[resume]` banner (from
  `step121000_full.pt`), saves 124000 (19:35:45) and **124500 (20:24:40)**, then dies
  20:26 on a TCPStore error.
- `logs/olmo2_7B_keep8fresh2_resume200k_82.log` — a **different process**, 2026-08-12
  00:37:41, resumes **from `step124500.pt`**, saves **125000** (01:27:16), and later
  130000 / 130500 / 131000 (09:36 / 10:25 / 11:14) **all inside that one process**.

So the interval **124500→125000 crosses a process boundary**, and
`train_olmo2_arch_probe2.py:1011-1019` rebuilds the loader (`sampler.set_epoch(epoch);
data_iter = iter(loader)`) **without intra-epoch fast-forward**, so that interval saw a
different data order than an uninterrupted 500 steps.

**Consequence, fixed in advance:** cluster1 is **not a clean 500-step neighbourhood**,
and any verdict built on it inherits that defect. It is reported, but it **cannot be
promoted to a clean within-arm control**. See §4 — this is why the primary verdict is
expected to be `UNRESOLVED`, and the pre-registration says so *before* the numbers.

### 2.6 LR must be measured, never assumed

The LR at each of the six checkpoints is taken from the **training logs' own
`[step N/200000] ... lr=` lines** and independently recomputed from the trainer's actual
schedule (`train_semantic_bottleneck_1b.get_lr`, cosine, `base_lr=2e-5`, `min_lr=2e-6`,
`warmup=150`, `max_steps=200000`, confirmed from the `[optim] group ...` banner). The
recomputation must match the logged value to the logged 3 significant figures or the run
aborts. **The keep12-verdict table's numbers are not copied.**

---

## 3. THE DECISIVE PRE-REGISTERED OBSERVATION (recorded before any range is read)

The LR contrast between the two clusters is **measured from the logs and is tiny**:

| cluster | steps | logged LR | mean LR |
|---|---|---|---|
| cluster1 | 124000 / 124500 / 125000 | 7.69e-6 / 7.63e-6 / 7.56e-6 | 7.6288e-6 |
| cluster2 | 130000 / 130500 / 131000 | 6.92e-6 / 6.86e-6 / 6.80e-6 | 6.8579e-6 |

**Ratio = 1.1124× (mean), 1.1323× (max/min over all six points).**

Compare the cross-arm spread H_LR was generated on: keep10/keep12 = 1.24e-5 / 3.25e-6 =
**3.82×**, against a range spread of **6.23×**.

**This is written down BEFORE the ranges are read**, because it determines what the test
can possibly conclude:

* A **1.11× LR contrast** is **~13 % of the 3.82× contrast** the hypothesis was fitted
  on. If H_LR is any smooth monotone function of LR, the *predicted* range difference
  between these two clusters is **small**, and this design has **almost no power** to
  detect it.
* Therefore **a null result here does NOT refute H_LR**, and a large difference here
  would, if anything, *embarrass* H_LR by being far too large for the LR gap — i.e. it
  would indicate a **third factor**, not confirm the hypothesis.

**This is a stated design limitation of MAIN's proposal, registered in advance rather
than discovered afterwards.** MAIN's dispatch asked to be told if the design cannot
answer Q2. On the arithmetic above, **it cannot answer it in the affirmative or the
negative**; what it *can* do is bound the size of the within-arm positional effect and
show whether the "range" statistic is stable across two positions of one arm.

---

## 4. Verdict labels, their criteria, and which of them can fire

Fixed here. The verdict string is emitted **mechanically** by the script from these
criteria; it is not chosen after seeing the table. Primary axis = **triviaqa** (the only
axis that has ever cleared the gate). Decision axes = triviaqa, popqa, mmlu_content;
`nq_open` demoted by design §5.2.

**Direction convention, stated once so it cannot be flipped later.** cluster1 is the
**earlier / HIGHER-LR** cluster (mean 7.63e-6); cluster2 is the **later / LOWER-LR**
cluster (mean 6.86e-6). H_LR says *higher LR ⇒ larger range*, so H_LR predicts
`range(cluster1) > range(cluster2)`. Define `R = range(cluster1) / range(cluster2)`.

| label | criterion (on triviaqa margin range, `split` convention) |
|---|---|
| **`REFUTED_WITHIN_ARM`** | both clusters clear their own `range_exceeds_item_noise` gate **and** `R ≤ 0.83` — the higher-LR cluster is detectably **narrower**, the reverse of H_LR's ordering. |
| **`SUPPORTED_WITHIN_ARM`** | both clusters clear their gates **and** `R ≥ 1.20` — higher LR ⇒ wider, the same direction as the cross-arm table. |
| **`UNRESOLVED_SUBNOISE`** | **≥1 of the two clusters fails its own noise gate.** A range that is inside item noise is not a measurement, so the comparison of the two ranges is **undefined** — neither support nor refutation. |
| **`UNRESOLVED_UNDERPOWERED`** | both clear, but `0.83 < R < 1.20` — no detectable difference at a contrast this small. |
| **`INADMISSIBLE_SEAM`** | cluster1's resume seam is confirmed (it is, §2.5) → **whatever the ratio, the result may not be reported as a clean within-arm LR contrast.** This label is a **modifier appended to whichever of the above fires**, not a substitute for it. |

The `1.20 / 0.83` boundary is `±20 %`, chosen to be **looser** than the 8 % agreement
keep8-vs-keep10 already showed on this statistic (1.2149 / 1.1202 = 1.085) so that
ordinary replication noise cannot manufacture a direction.

**Explicitly pre-committed:** if `UNRESOLVED_SUBNOISE` fires, the report must say
**"no detectable difference — neither supports nor refutes H_LR"** and must **not** be
re-described as "consistent with noise, therefore H_LR is fine" or as "the direction is
reversed, therefore H_LR is dead". MAIN's dispatch asked for exactly this honesty and it
is fixed here before the first number.

---

## 5. Q3 — how §2.5's tolerance must be phrased, decided by which label fires

| label that fires | what §2.5 may say |
|---|---|
| `REFUTED_WITHIN_ARM` | the tolerance is **positional within a single arm**, so no per-arm constant is defensible; §2.5 must become "report the neighbours", full stop. |
| `SUPPORTED_WITHIN_ARM` | the tolerance may be **conditioned on LR**, and §2.5 should carry an LR column. |
| either `UNRESOLVED_*` | §2.5 keeps its **measured upper bound in pp** (~1.2 pp triviaqa / ~0.35 pp elsewhere, two arms) and adds an explicit statement that the **lower** end (keep12's 0.1951 pp) is real and unexplained, so the number is an **upper bound and not a threshold to clear**. The three candidate phrasings MAIN lists are then reconciled by **separating the two uses of the number**: as a *reporting trigger* it is an upper bound; as a *certification threshold* it is unavailable. |

---

## 6. What this run may NOT claim, fixed in advance

- ⛔ Any statement that the two clusters are **replicates**. They are successive states of
  one optimisation; the range is a **checkpoint-SELECTION** quantity, never seed variance.
  No 7B `sd_run` exists or is reconstructible.
- ⛔ Any promotion of cluster1 to a clean neighbourhood (§2.5).
- ⛔ Any claim that H_LR is **confirmed**. n=3 arms + a 1.11× within-arm contrast cannot
  confirm a schedule law.
- ⛔ Quoting any margin to better than **0.01 pp across nodes** (numpy multinomial split).
- ⛔ Any K1/K2/K3 clause — those are defined over the pre-registered **1B** arm set.
- ⛔ Re-deriving any null, Δ, metric, rule or anchor.

---

## 7. Outputs fixed in advance

- `A04_WITHIN_ARM_LR_REFUTATION_VERDICT.md`
- `evidence/a04_within_arm_lr.json`
- `code/a04_within_arm_lr.py`
- `STATUS.json` += one new key `within_arm_lr_refutation_20260813` with
  `gpu_h_spent: 0`, **append-only, verified at the text level** (byte prefix of the old
  file preserved), key count 45 → 46.
