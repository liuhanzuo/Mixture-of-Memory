# A04 — Kill Gate Design

**Date**: 2026-08-09. **Status**: DESIGN ONLY. **GPU spent producing this document: zero.**
**This document does not authorise any GPU run.** The full experiment is the most expensive gate in
the proposal tree and requires explicit user approval (see §7 for the honest number).

Prerequisite: `RELATED_WORK.md` (same directory), which narrows what A04 may claim. This design
implements only that narrowed claim.

---

## 1. The claim, stated so it can be false

> **A04 claim.** For a structurally injured LM undergoing recovery training, the stopping rules
> currently in use — (a) a likelihood/perplexity plateau, and (b) an aggregate
> retained-accuracy-ratio threshold — **accept** recovery runs that a non-inferiority test against
> the intact target on null-calibrated capability axes **rejects**. The disagreement is (i) large
> enough to change the decision, (ii) reproducible across independent recovery seeds, and
> (iii) not explainable by run-to-run variance.

Three ways this is false, each individually sufficient:

* **F1 (no disagreement).** The rules agree. Whenever PPL plateaus, the capability axes are also
  non-inferior. Then there is nothing to certify and the whole direction is a null result.
* **F2 (disagreement is noise).** The rules disagree on individual runs, but the disagreement is
  inside the seed-to-seed spread, so no rule can be stated that survives replication. This is
  exactly how B04's Direction A died (see `DIRECTION_A_QWEN_VERDICT.md`), and it is the most likely
  failure mode.
* **F3 (unmeasurable).** The capability axes at the target scale are at their own best-constant
  floor for every arm, so "non-inferior to intact" is undefined for lack of dynamic range.

A04 survives only if **none** of F1/F2/F3 holds.

---

## 2. Kill condition — verbatim, quantitative, pre-registered

Fixed before any data is seen. Style matched to A02's `kill_gate_verbatim` and A01's `Kill 条件`.

> ### KILL CONDITION (verbatim, binding)
>
> **A04 is killed if ANY of the following three clauses fires.**
>
> **K1 — no rule disagreement.** Across the arms, at every checkpoint on the pre-frozen grid where
> the **PPL-plateau rule** `PLATEAU(T)` accepts, the **non-inferiority rule** `NI(Δ)` also accepts,
> on **at least 3 of the 4** capability axes; **and** the count of `(arm, checkpoint)` cells where
> the two rules disagree is **≤ 1 out of ≥ 24 evaluated cells**. Consequence: there is no
> certification problem; archive with a POSTMORTEM.
>
> **K2 — disagreement drowned by seed variance.** For the primary axis (TriviaQA EM), the
> **between-seed standard deviation of the calibrated residual** at the pre-frozen apex checkpoint,
> pooled across arms with `S ≥ 3` seeds each, is **≥ 50% of the smallest between-arm residual
> difference the paper wants to claim**; equivalently, the one-sided 95% run-level bound
> `t_{0.05,S-1} · sd_run / sqrt(S)` **exceeds the pre-registered non-inferiority margin
> Δ = 10% of the intact arm's own calibrated residual** on **≥ 2 of the 4** axes. Consequence:
> stop; report the variance floor as the finding and do not claim a stopping rule.
>
> **K3 — axes at floor.** At the target scale, **≥ 3 of the 4** capability axes have the
> **intact** arm's calibrated residual (reported − construct-appropriate best-constant null)
> **below 5pp**, or have **every** damaged arm indistinguishable from its own best-constant
> predictor at BH `q = .05`. Consequence: the scale cannot support the measurement; either move
> scale (needs new approval) or archive.
>
> **Definitions frozen with this document:**
> * `NI(Δ)` accepts arm `a` at checkpoint `c` on axis `x` iff the **one-sided lower 95% bound** on
>   `residual(a,c,x) − residual(intact,x)` is **> −Δ_x**, where `residual = reported − null_x` and
>   `Δ_x = 0.10 · residual(intact, x)`. This is the TOST/non-inferiority direction, **not** a
>   superiority test, and **not** a two-sided difference test.
> * `PLATEAU(T)` accepts at checkpoint `c` iff the relative in-domain validation PPL improvement
>   over the preceding grid interval is `< T`, with **`T = 2.0 %` per 5 000 steps** — a single
>   number, fixed here, never re-tuned after seeing data.
> * `RATIO(ρ)` accepts iff `mean_over_axes(reported_a / reported_intact) ≥ ρ`, with
>   **`ρ = 0.85`**, chosen to match the published style of retention headline (CoMe reports
>   "retains 83% of original average accuracy", arXiv:2510.15304 abstract) rather than to fit our
>   data.
> * Both `T` and `ρ` and `Δ` are **frozen by this document's git commit**. If any is later
>   changed, the gate is void and must be re-run.

#### 2.0.1 AMENDMENT 2026-08-10 — admissibility guard on `Δ` (`A04_MARGIN_GUARD_PREREG.md`)

**The pre-registered text above is UNCHANGED. Nothing in it is edited, retuned, or replaced.**
This is an *additive* amendment, registered **before any recovered-arm datum exists** (verified
2026-08-10 by `ls outputs/` on **both** disks: the only 1B recovery runs anywhere are `keep7`
variants on zwfy6; there is no `j=12` or `j=10` arm, and no 1B recovery output at all on wzc1).

The `Δ_x = 0.10 · residual(intact, x)` definition above is a well-formed **non-inferiority**
margin only while `residual(intact, x)` is comfortably positive. Pilot Zero found a real case
where it is not: under the `credit` MMLU tie convention the **intact** 1B model scores
`0.386839481555334` against its own null `0.4537102976783934`, so
`residual(intact) = −6.687081612305939 pp` and `Δ` is **negative** — which makes `−Δ` positive
and silently converts `NI(Δ)` into a **strict-superiority** test (see §4.1 of
`PILOT_ZERO_VERDICT.md`; flagged per-cell as `delta_degenerate_negative_margin` in
`evidence/pilot_zero_rule_disagreement.json`). This cannot create a false accept — a
superiority test is strictly harder — but it creates a **silent false reject plus a swapped
hypothesis**, which in a pre-registered gate is the more serious failure.

**→ See `A04_MARGIN_GUARD_PREREG.md` for the binding guard.** In brief, and binding here:

* Six degeneracy conditions **D1–D6** (residual negative / at-zero / CI straddling zero /
  inadmissible null / unstable intact anchor / `Δ` below the item-level CI half-width) are
  evaluated **before** `NI(Δ)` is computed for any cell.
* A cell failing D1, D2, D3, D4 or D6 is **NOT_CERTIFIABLE**: `NI(Δ)` is **not run** for it, no
  accept/reject is reported, and it is **excluded from the BH family** (declare the reduced
  family size).
* **`Δ` is never substituted.** `max(Δ, ε)`, `0.10·|residual|`, and any change of anchor are
  **prohibited** for the remainder of A04 — a plausible-looking margin over a sub-null target
  measures nothing.
* Because `residual(intact, x)` depends on neither the arm nor the checkpoint, the guard's
  verdict is a property of `(axis, convention)` alone and is therefore **fully fixed today**.
  Under the pre-registered `split` convention: **0 of the 72 decision cells are retired**
  (the 24 NOT_CERTIFIABLE cells are all NQ-open, already demoted by §5.2), and 24
  MMLU-content cells are `NEEDS_RECHECK_AFTER_DATA` on the **fixed** arithmetic trigger
  `p_disc > p*_crit = 0.3832`.
* Two clarifications the guard adds to the kill clauses in this section, both fixed before
  data: (i) if the guard retires a decision axis, K1's "≥3 of 4" and K2's "≥2 of 4" rescale to
  `ceil(0.75·n_surviving)` and `ceil(0.50·n_surviving)` — otherwise retiring one of three axes
  would make K1 **unsatisfiable and thus unfireable**; (ii) axes retired by D1/D4 are
  **excluded** from K3's "residual below 5pp" count rather than counted as at-floor, since a
  *negative* residual is trivially "below 5pp" and would make K3 fire for an inadmissible null
  rather than for an unmeasurable scale.
* One consequence for a claim already written down: `PILOT_ZERO_VERDICT.md` §4's
  "survives **all five** conventions" must now read "**the verdict is unchanged under the four
  admissible conventions; `credit` is inadmissible on MMLU-content by the margin guard**".
  The measured *difference* is unaffected — the null cancels exactly in
  `residual(arm) − residual(intact)` — so only the `credit` **margin** is lost.


### 2.1 What the gate is NOT allowed to conclude

* Not "our rule is better" — it can only conclude the rules **disagree** and quantify by how much.
* Not a scaling law in depth. Only the arms actually trained under this design may be compared.
* Not anything about 7B. The gate runs at 1B; transfer is a separate, unapproved question.

---

## 3. Arms and what is held fixed

The confound in `STATUS.json:warning` is real and verified in our own data — see
`RELATED_WORK.md` §5: the existing keepN ladder spans **two different corpora** (7,570,911 rows on
wzc1 vs 15,491,607 rows on zwfy6, ratio **2.0462×**, re-derived here from `ls -l` byte counts on
both disks) **and** unequal steps (keep14 200k / keep12 124k / keep10 83.5k / keep8 121k). This
design exists to not repeat that.

### 3.1 Frozen-identical across every arm (the anti-confound contract)

| Held fixed | Value / mechanism |
|---|---|
| training array | **one** physical file, pinned by SHA256 **recorded before launch**; every arm's log must print the same `dataset rows=` |
| disk | all arms on **one** disk (zwfy6), because the same-named `dolmino_now15b.npy` differs between disks |
| token budget | matched in **token presentations** (`eff_bs × seq_len × steps`), not optimizer steps. At `eff_bs=128, seq_len=2048` → **262,144 tokens/step** |
| base model | `models/OLMo-2-0425-1B` (`num_hidden_layers=16, hidden_size=2048, vocab_size=100352`), verified present on **both** disks |
| optimizer | `adamw` fp32 master weights. **Not** `bnb_adamw8bit` (unavailable on `.82`, and changes optimizer state format) |
| LR schedule | identical `--lr / --min_lr / --lr_inherited / --min_lr_inherited / --warmup_steps` across arms; **and the writeup must state the LR is effectively uniform**, because `fresh_*` param groups were empty in all historical runs (`status/PAPERB_DIFFERENTIAL_LR_NEVER_ACTIVE.md`) — re-verify per arm from the `[optim] group` lines |
| batch | `eff_bs=128` via `batch_size × grad_accumulation_steps × world_size` — **must be recorded**, since the same `eff_bs` was reached as `8×1×16` (16-card) and `4×4×8` (8-card) historically |
| checkpoint grid | frozen **in this document**: `{2500, 5000, 10000, 20000, 40000, 80000}` steps. `--milestone_every 2500` to retain them |
| no resume | every arm runs straight through. Warm-restart destroys Adam moments and adds a confound (`status/PAPERB_RESUME_WARM_RESTART_DEFECT.md`) |
| node class | all arms on H20 (`.73/.82/.104`). L20A vs H20 must not be mixed within the gate — measured 7B medians are 1.56 s/step (L20A, bs16×1) vs 6.79–7.81 s/step (H20, bs4×4); different bs/gaccum, so this is **not** a clean hardware ratio, which is precisely why arms must not straddle it |

### 3.2 Arms

Damage is held at a **single depth** so that depth is not a variable. `keep_front j` is chosen so
the arm is measurable but not saturated; **j must be chosen by Pilot Zero (§6), not here.**

| # | Arm | Trainer flags | Role |
|---|---|---|---|
| A1 | prefix + fresh tail | `--keep_front_layers j --n_fresh_layers 2` | the canonical construction; the one A04's claim is about |
| A2 | contiguous keep-only | `--keep_front_layers j --n_fresh_layers 0` | isolates the contribution of the fresh tail |
| A3 | random trunk, inherited interface | `--keep_front_layers j --n_fresh_layers 2 --random_trunk` | same depth/shape as A1, only trunk provenance differs; embed/norm/lm_head still transplanted |
| A4 | from scratch | `--from_scratch` at depth `j+2` | the "did inheritance matter at all" floor |

`--random_trunk` and `--from_scratch` are mutually exclusive and both exist in
`scripts/train_olmo2_arch_probe2.py` (verified: `--random_trunk` at line 586, `--from_scratch` at
584, with an explicit `p.error` on combination).

**Arms deliberately NOT included**, and why:
* ShortGPT / non-contiguous drop (in the current `PROPOSAL.md` MVP list): it changes **which**
  layers are removed, i.e. it is a second damage variable. It belongs in a follow-up, not in a gate
  whose whole point is one-variable-at-a-time. Dropping it saves 25% of the spend.
* A "3 token budgets" arm set: **the three budgets are checkpoints of one run, not three runs.**
  Reading `PROPOSAL.md`'s "4 structures × 3 budgets × 2 seeds" as 24 independent trainings costs
  3,771 GPU-h; reading them as 6 checkpoints of 8 runs costs 2,873 GPU-h (§7). There is no
  scientific reason to retrain for a shorter budget when a checkpoint at that budget already exists
  on the same trajectory.

### 3.3 Seeds — and a defect that must be fixed first

`--seed` in `scripts/train_olmo2_arch_probe2.py` moves **only the fresh-tail random init**. Verified
live at line 863:

```python
sampler = DistributedSampler(ds, shuffle=True)
```

— **no `seed=` argument**, so its private generator uses `self.seed = 0` regardless of `--seed`;
`attention_dropout: 0.0` and `grep -c dropout` on the trainer is 0. So:

> **Every historical and current "seed" arm in this repo is fresh-block-initialisation variance,
> NOT training-seed variance.** Data order is identical across seeds.

Consequences for this gate, both mandatory:
1. **Blocking code change (CPU, ~1 line)**: pass `seed=args.seed` to `DistributedSampler` so that
   seeds actually vary data order. Without it, K2's variance estimate is an **underestimate** —
   it omits the data-order component — and a K2 "pass" would be unearned.
2. If the change is not made, the writeup must say "fresh-block-init variance" everywhere it says
   seed, and **K2's threshold must be treated as optimistic**, i.e. a K2 pass is not licensed.

Arm A2 has `n_fresh_layers 0`, so it has **no fresh block at all** — with the current trainer its
seeds are byte-identical and its `sd_run` is exactly 0. That is not a variance estimate; it is a
missing measurement. This alone makes fix (1) load-bearing rather than nice-to-have.

---

## 4. Metrics and their construct-appropriate nulls

A01 established that "above chance" is the wrong reference. Every axis gets its **own
best-constant floor**, and every number below is copied from a file in this repo, not recomputed
from memory.

Source: `proposal/active/A03-parametric-vs-external-memory/evidence/a03_1b_floor_nulls.json` and
`GATE_FOURAXES_VERDICT.md`, both measured on **OLMo-2 1B**, the same scale as this gate.

| Axis | n | Null (construct-appropriate) | Null value | Intact 1B | Intact calibrated residual | Δ = 10% |
|---|---:|---|---:|---:|---:|---:|
| **TriviaQA EM** (PRIMARY) | 17,944 | best-constant answer string `"australia commonwealth realm"` | **0.0025635** | 0.4069 | **0.4043** | **4.043 pp** |
| PopQA EM | 14,267 | best-constant string `"association football"` (argmax over 303 candidates) | **0.0229200** | 0.1550 | **0.1321** | **1.321 pp** |
| MMLU-**content** | 14,042 | longest-option, **split-tie** convention (pre-registered) | **0.2844502** | 0.3868 | **0.1023** | **1.023 pp** |
| NQ-open EM | 3,610 | best-constant answer | **0.0055** | 0.1025 | **0.0970** | **0.970 pp** |
| in-domain PPL | 4,096 windows / 8,384,512 tok | *no null* — used only by `PLATEAU(T)`, never as a capability axis | — | 10.6416 | — | — |

### 4.1 Interfaces explicitly BANNED from this gate

* **MMLU-letter.** At 1B the pruned arm scores 0.2512 against its best-constant floor
  **0.2689 (always-D)** — significantly **BELOW** floor (`p=3.4e-3`), and indistinguishable from its
  own modal-C constant (`p=0.28`); the barely-healed control emits `A` on **14,042/14,042** items.
  Verified in `a03_1b_floor_nulls.json:nulls.letter_degeneration`. It measures nothing here.
  **`0.25` is never the MMLU null.**
* **Raw `contains` on generative QA**, unless length-matched. Healing changes verbosity ~6×
  (mean pred chars 13.4 intact → 80.8 pruned+healed, from the same JSON), and `contains` rewards
  length. PopQA `contains` residual fraction reads 56.4% naive but **17.0%** against a
  length-matched input-blind null. If `contains` is reported at all, it is
  `contains_lenmatched` only.
* Any MC axis whose own best-constant floor was not recomputed on **this** item set.

### 4.2 Nulls for axes if the gate is ever widened (recorded so they are not invented later)

Verified in `status/scout_21/lane2_a01_gate2.md`: BoolQ null is **0.6217 (always-B)**, not 0.50;
OpenBookQA longest-option null is **0.3635**, not 0.25; `winogrande` is structurally degenerate
(identical norm_lens, 100% tie rate) and may only be a control. `arc_challenge` 0.2654,
`piqa` 0.5049, `commonsense_qa` 0.2088 (all always-B).

### 4.3 Protocol invariants (non-negotiable)

* **`chat_template=False` for every eval.** These are BASE LMs with no SFT/RL; a chat template is
  unfair and any chat=True number is void.
* Prune-heal OLMo-2 is a **BASE** LM: base-protocol only — PPL + LL-based MC, **no chat template,
  no BOS** (`--add_bos 0`), no system prompt, no few-shot, greedy decode for QA, compared against
  **vanilla OLMo-2 BASE**.
* **Sharded eval must assert every shard is present before merging.** A silently merged 5-of-8
  shard set has corrupted results in this repo before. Reuse the existing pattern from
  `scripts/_run_a03_1b_floor_82.sh`, which hard-asserts both `n_shards==8` **and** the exact
  expected item count (`assert MMLU n==14042`, `assert s["n_valid"]+s["n_nan"]==exp`,
  `assert v["n"]==e`) before any merge is trusted. Per-axis expected counts are the `n` column in
  §4.
* **Do not budget for or excuse anything as "runtime noise."** Same-arch/same-harness re-runs in
  this repo are **byte-identical (0 flips)**. Any nonzero difference must be attributed to a named
  cause or be treated as a real effect. If a "noise floor" is ever claimed, a same-code control
  must be run first.
* Every scoring step must **import the canonical scorer** from the harness that produced the
  predictions, never reimplement it. Two separate subagents reimplemented a metric in one session
  and both times the reimplementation produced a significant result where canonical produced a tie
  (`A02/STATUS.json:meta_note_for_future_scoring`).

---

## 5. Statistical power — why not n=6

### 5.1 The B04 lesson

B04 Direction A reached the **exact-permutation floor** for n=6: two-sided min
`p = 2/6! = 2/720 = 0.002778`, hit twice (Spearman `+1.0000` and `−1.0000` on OLMo-2-7B). **It
still died** on cross-family replication: on Qwen3-8B the same statistics fell to `+0.4286
(p=0.42)` and `−0.4857 (p=0.36)`, ρ dropping 0.51–0.57 and p by ×130–150
(`DIRECTION_A_QWEN_VERDICT.md`). So maximal significance at n=6 is **not** evidence of a real
effect; a rank correlation over 6 rungs can be a perfect monotone artefact of how the rungs were
chosen. Verified permutation floors: n=4 → 0.0833, n=5 → 0.0167, n=6 → 0.002778, n=7 → 0.000397,
n=8 → 0.0000496.

**Design consequence: A04's primary statistic must NOT be a rank correlation over rungs.** Adding
rungs to buy permutation p-value is exactly the move that failed. A04's primary statistic is a
**per-arm non-inferiority bound**, whose power comes from `S` (seeds) and item count `n`, not from
the number of rungs.

### 5.2 Item-level power is not the binding constraint

Upper-bound SE of a paired accuracy difference, `sqrt(p_disc/n)`:

| Axis | n | 95% CI half-width at p_disc=0.20 | at p_disc=0.40 |
|---|---:|---:|---:|
| TriviaQA | 17,944 | 0.654 pp | 0.925 pp |
| PopQA | 14,267 | 0.734 pp | 1.038 pp |
| MMLU | 14,042 | 0.740 pp | 1.046 pp |
| NQ-open | 3,610 | **1.459 pp** | **2.063 pp** |

Against `Δ_TriviaQA = 4.043 pp` the item-level CI is comfortably inside the margin. Against
`Δ_NQ-open = 0.970 pp` it is **not** — the item-level CI half-width alone (1.46–2.06 pp) already
exceeds the margin. **NQ-open therefore cannot carry a 10%-of-residual non-inferiority decision at
n=3,610 and must be demoted to a secondary/descriptive axis.** This is a design finding, not a
number to hide.

### 5.3 Run-level power is the binding constraint

The unit of replication is the **run**, so the decision statistic is a run-level one-sided bound
`t_{0.05,S-1} · sd_run / sqrt(S)`:

| S (seeds) | df | t | bound at sd_run=0.3pp | 0.5pp | 1.0pp |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 6.314 | 1.339 pp | 2.232 pp | 4.465 pp |
| 3 | 2 | 2.920 | 0.506 pp | 0.843 pp | 1.686 pp |
| 4 | 3 | 2.353 | 0.353 pp | 0.588 pp | 1.177 pp |
| 5 | 4 | 2.132 | 0.286 pp | 0.477 pp | 0.953 pp |

**S=2 is unusable.** With one degree of freedom, `t=6.314`, so even a tiny `sd_run` of 0.3 pp gives
a 1.34 pp bound — larger than `Δ` on three of the four axes (1.023 / 1.321 / 0.970 pp). A
2-seed design can only ever certify on TriviaQA (Δ=4.043 pp), and even then only if
`sd_run < 0.9 pp`. **`PROPOSAL.md`'s "至少 2 seeds" is therefore not a viable minimum for the
non-inferiority decision** — it was written for a superiority comparison, and equivalence is harder.

**S=3 is the minimum**, and it only works if `sd_run ≤ ~0.5 pp`. **`sd_run` is currently
UNVERIFIED** — no multi-seed 1B recovery run exists on either disk (the only 1B arm is
`keep7fresh2`, one draw, and its seed was not even recorded since `--seed` postdates it).
Measuring `sd_run` is therefore itself a gate output, and it is the reason Pilot One (§6.2) exists.

Recommended: **S=3 for the gate**, with a pre-committed rule that if the measured `sd_run` implies
a bound `> Δ` on ≥2 axes, **K2 fires** rather than seeds being added post hoc. Adding seeds after
seeing the variance is the same selective-reporting move A01's gate-4 exists to prevent.

### 5.4 Multiplicity

4 axes × 4 arms × 6 checkpoints = 96 cells. Apply **Benjamini–Hochberg `q=.05` across the whole
declared family**, matching A01/A03 convention. Declare the family size in advance. Note that
non-inferiority bounds are one-sided; do not silently mix them into a two-sided BH family.

---

## 6. Cheaper pilots — and why the first one may kill A04 for ~3 GPU-hours

### 6.1 Pilot Zero — CPU/eval only, uses checkpoints already on disk (≈3 GPU-h, or 0 with reuse)

**This is the recommended immediate next step and the reason the full spend should not be
authorised yet.**

A complete 1B recovery trajectory already exists and is verified on disk (zwfy6,
`olmo2_ppl_results/*/summary.json`, all with `n_shards=8`, `n_tokens=8384512`, `n_windows=4096`):

| checkpoint | in-domain PPL | gap to intact | rel. improvement over previous interval |
|---|---:|---:|---:|
| intact 1B base, 16L | **10.6416** | — | — |
| keep7+fresh2 @ 50,000 | 17.6194 | +6.9778 (+65.6%) | — |
| keep7+fresh2 @ 100,000 | 16.1613 | +5.5197 (+51.9%) | 8.276% |
| keep7+fresh2 @ 147,000 | 15.6285 | +4.9869 (+46.9%) | 3.297% |
| keep7+fresh2 @ 200,000 | 15.4116 | +4.7700 (+44.8%) | **1.388%** |

And the capability side at step 200,000 is already measured, null-calibrated, and BH-tested
(`GATE_FOURAXES_VERDICT.md`), with per-example shards on disk under
`olmo2_closedbook_results/A03_1B_keep7_step200k/` and `olmo2_mmlu_content_results/A03_1B_keep7_step200k/`:

| axis | arm residual | intact residual | **fraction of intact residual recovered** |
|---|---:|---:|---:|
| MMLU-content | 0.0399 | 0.1023 | **39.0%** |
| TriviaQA EM | 0.0933 | 0.4043 | **23.1%** |
| NQ-open EM | 0.0230 | 0.0970 | **23.7%** |

**What Pilot Zero tests.** Pre-register `T` and `ρ`, then ask whether they accept at step 200,000
while `NI(Δ)` rejects. The arithmetic is already available: the last interval's relative PPL
improvement is **1.388%**, the previous is **3.297%**. So **any** plateau threshold `T` in
`(1.388%, 3.297%]` accepts at 200,000 — a wide and entirely plausible band — while the arm has
recovered only **23–39%** of the intact null-calibrated residual, i.e. `NI(Δ=10%)` rejects by an
enormous margin on all three measured axes.

> **Honesty note, load-bearing:** the band `(1.388, 3.297]` was computed **after** looking at these
> numbers, so Pilot Zero as described is *illustrative, not confirmatory*. To be evidence, `T` must
> be committed in git **before** the comparison is re-run, and the grid is only 4 points with 47k–53k
> spacing, which cannot exercise a 5,000-step-resolution rule. Pilot Zero's honest output is
> therefore: **"does a disagreement of the required shape exist at all, on one arm, one seed?"**
> It can fire **K1** (if no disagreement exists, stop for ~0 GPU) and **K3** (the axes are already
> known measurable at 1B — A03 verified 4/5 interfaces above floor, so K3 is already
> provisionally cleared). It **cannot** clear K2, which needs seeds.

Cost: the per-example shards are on disk, so the disagreement analysis is **pure CPU, minutes**.
If any axis must be re-scored, A03 measured ~4–5 min per checkpoint per axis on 8×H20, so ≤10
checkpoints × 4 axes ≈ **3 GPU-h**. Also available for free: `cpt20k` steps
205k/210k/215k/220k and `arm4_peaklr20k` steps 205k/210k/215k, all verified present on zwfy6.

**Additional free output**: Pilot Zero picks `j`. The only existing 1B arm is `keep7+fresh2` = 9 of
16 layers = **56.2% depth**, and after **52.4 B heal tokens** it recovers only 23–39% of the intact
residual. A rule tested only there is a constant-REJECT and proves nothing. The gate needs a `j`
where a well-run recovery plausibly *does* approach non-inferiority — candidates
`keep12+fresh2` (14L, 87.5% depth) or `keep10+fresh2` (12L, 75%). Choosing `j` from the
literature-free evidence of Pilot Zero, rather than by assumption, is worth its cost alone.

### 6.2 Pilot One — the variance measurement, 3 runs (≈135 GPU-h, ~5.6 h wall on 3 nodes)

Only if Pilot Zero shows the disagreement exists. **Blocking prerequisite: the
`DistributedSampler(seed=)` fix from §3.3**, else this measures the wrong variance.

One arm (A1) at the `j` chosen by Pilot Zero, **S=3 seeds, 5,000 steps each**. Output: the first
real `sd_run` for a 1B recovery arm in this project.

Cost, from the measured 1B median **2.02 s/step on 8×H20** (n=36 log samples,
`logs/olmo2_1B_keep7fresh2_1node.log`): 3 runs × 5,000 steps = **2.81 h wall each**,
**22.4 GPU-h each**, **≈135 GPU-h total**, one wave per node on 3 nodes ≈ **5.6 h wall**.

Decision: plug `sd_run` into §5.3. If the S=3 bound exceeds `Δ` on ≥2 axes, **K2 fires and A04 dies
for ~135 GPU-h instead of ~2,900.** This is the single highest-value GPU purchase in the design.

### 6.3 Pilot Two — full gate. Do not launch without user approval.

---

## 7. Cost — honest numbers

All from measured medians in this repo's own logs, not estimates:

* **1B, 8×H20**: median **2.02 s/step** (n=36, `logs/olmo2_1B_keep7fresh2_1node.log`, header
  `world_size=8 bs=16 gaccum=1 eff_bs=128 seq_len=2048`).
* **1B, 16×H20**: median **1.48 s/step**, mean 1.5218 (n=10,000 log samples,
  `logs/olmo2_1B_keep7fresh2_16card_node0.log`, `world_size=16 bs=8 gaccum=1 eff_bs=128`).
  Wall-clock cross-check: 2026-07-16 22:40:29 → 2026-07-20 11:18:11 = **84.63 h for 200,000
  steps** = 1.5233 s/step average, consistent.
* 16 cards give **1.36× throughput for 2× the GPUs** — *scaling is poor*, so **8-card runs are the
  efficient unit**; use extra nodes for more arms in parallel, not for faster single arms.
* `eff_bs=128 × seq_len=2048` = **262,144 tokens/step**; 80,000 steps = **20.97 B tokens**;
  200,000 steps = **52.43 B tokens**.

| Configuration | runs | GPU-h | wall, 1×8-GPU node | wall, 3×8-GPU nodes |
|---|---:|---:|---:|---:|
| **Pilot Zero** (eval only) | 0 | **≈3** (or 0 with reuse) | minutes–hours | — |
| **Pilot One** (1 arm × 3 seeds × 5k) | 3 | **135** | 8.4 h | **5.6 h** (1 wave) |
| Gate, 4 arms × 3 seeds × 20k steps | 12 | **1,077** | 134.7 h (5.6 d) | **44.9 h** (4 waves) |
| Gate, 4 arms × 2 seeds × 80k steps | 8 | **2,873** | 359 h (15.0 d) | **135 h (5.6 d)** |
| Gate, 4 arms × 3 seeds × 80k steps | 12 | **4,309** | 538 h (22.4 d) | **202 h (8.4 d)** |
| `PROPOSAL.md` as literally written (24 independent trainings: 4 struct × 3 budgets × 2 seeds) | 24 | **3,771** | 471 h (19.6 d) | **157 h (6.5 d)** |

### 7.1 The honest verdict on affordability

* The literal 24-training reading costs **3,771 GPU-h**. Treating the three budgets as checkpoints
  of one trajectory instead of three separate runs removes **898 GPU-h (24%)** for **zero**
  scientific loss.
* The statistically defensible version — **S=3 seeds** (§5.3 shows S=2 cannot certify on 3 of 4
  axes) at the full 80k budget — costs **4,309 GPU-h ≈ 8.4 days on all three H20 nodes running
  nothing else**. Given that all five nodes are currently occupied (`.21` CAST, `.73` A03 Arm 4,
  `.104` Paper B keep12, LOCAL Paper B keep14, `.82` another agent's A02), and that proposal work
  competes with Paper A/B, **this is not affordable as a speculative gate.**
* The affordable defensible version is **4 arms × 3 seeds × 20,000 steps = 1,077 GPU-h ≈ 45 h on
  three nodes**. 20,000 steps = 5.24 B tokens. Whether that is enough recovery for any arm to
  approach non-inferiority is **UNVERIFIED** and is exactly what Pilot Zero's `j` selection is for
  — at `keep7` (56.2% depth) even 52.4 B tokens left a 44.8% PPL gap and 61–77% capability
  shortfall, so at a shallower cut like `keep12` the 20k budget may be adequate while at `keep7`
  it certainly is not.

**Recommendation on spend: authorise Pilot Zero (≈3 GPU-h) now, and Pilot One (135 GPU-h) only if
Pilot Zero fires positively. Do not authorise the 1,000–4,300 GPU-h gate until `sd_run` is
known**, because K2 is the most likely killer and Pilot One buys the K2 answer for 3–5% of the full
cost.

---

## 8. Pre-registration checklist (must all be true before any gate GPU is spent)

- [ ] `T = 2.0%/5k`, `ρ = 0.85`, `Δ_x = 0.10 · residual(intact, x)` committed to git, unchanged.
- [ ] **Margin guard applied (`A04_MARGIN_GUARD_PREREG.md`, amendment §2.0.1)**: D1–D6 evaluated
      per cell *before* `NI(Δ)`; NOT_CERTIFIABLE cells excluded from the BH family and the
      reduced family size declared; intact anchor pinned by path + SHA256 (rule G0); K1/K2
      thresholds rescaled and D1/D4-retired axes excluded from K3's count if any axis is
      retired.
- [ ] Checkpoint grid `{2500, 5000, 10000, 20000, 40000, 80000}` committed; `--milestone_every 2500`.
- [ ] Training array SHA256 + row count recorded; every arm's log shows the **same** `dataset rows=`.
- [ ] All arms on **one** disk; `dolmino_now15b.npy` size recorded (wzc1 62,020,903,040 B vs zwfy6
      126,907,244,672 B — **same name, different file**).
- [ ] `DistributedSampler(seed=)` fix landed, **or** every "seed" relabelled
      "fresh-block-init draw" and K2 declared optimistic.
- [ ] `[optim] group` lines captured per arm to document whether differential LR was actually active.
- [ ] `chat_template=False`, `--add_bos 0`, no few-shot, greedy decode, on every eval.
- [ ] Shard-completeness assertions wired (`n_shards==8` **and** exact item count) per §4.3.
- [ ] BH family size declared; one-sided NI bounds not mixed into a two-sided family.
- [ ] Canonical scorers imported, not reimplemented.
- [ ] `RELATED_WORK.md` §4.2 must-not-claim list pasted into the writeup's scope section.

## 9. Known-unverified items in this design

1. **`sd_run` for any 1B recovery arm.** No multi-seed 1B run exists on either disk. Pilot One's
   entire purpose.
2. **Whether 20,000 steps suffices** for any arm to approach non-inferiority at a shallower cut.
   Depends on `j`; only `keep7` (52.4 B tokens, 44.8% residual PPL gap) is measured.
3. **PPL for any 1B arm other than `keep7+fresh2`.** `olmo2_ppl_results/` contains only
   `1B_base_full` and `1B_keep7_step{50000,100000,147000,200000}` (+2 sanity dirs).
4. **Whether the disagreement replicates at 7B.** Out of scope; unfunded.
5. **arXiv:2408.11796 (Minitron Approach) venue** — not verified in this pass.
6. **Pilot Zero's `T` band** `(1.388, 3.297]` is post-hoc, as stated in §6.1. Illustrative only
   until `T` is committed first.
7. **(added 2026-08-10, see `A04_MARGIN_GUARD_PREREG.md` §8)** Whether **MMLU-content** survives
   the guard's D6 condition at the gate's *early* checkpoints. `p*_crit = 0.3832` against a
   barely-healed discordance of `p_disc = 0.301097` leaves only **16 % headroom on `Δ/hw`**; the
   arm that would settle it does not exist. The early checkpoints, not the late ones, are the D6
   risk.
8. **(added 2026-08-10)** `p_disc` at a barely-healed checkpoint for **TriviaQA / PopQA /
   NQ-open** — only MMLU-content has a step-500 arm scored. Since `p_disc` rises as the arm
   degrades, those three D6 verdicts rest on an **optimistic** range. TriviaQA's ~31× headroom
   is safe by any margin; PopQA's ~5× is very likely safe but unproven at step 2,500.
9. **(added 2026-08-10)** An **unresolved tension between K3 and D6**: an axis can pass K3
   (residual ≥ 5 pp ⇒ "measurable") yet fail D6 (`Δ = 0.10 × 5 pp = 0.5 pp` is below every
   item-level half-width measured here, 0.54–1.02 pp) ⇒ "not certifiable at a 10 % margin". No
   1B axis falls in that window (smallest surviving residual 10.2389 pp), so it does not bite
   here — but it is **not resolved**, and it must **not** be resolved by adjusting `Δ` after
   data.
