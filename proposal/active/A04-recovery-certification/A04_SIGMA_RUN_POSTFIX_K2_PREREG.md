---
scope: A04 — the FIRST σ_run whose seeds actually varied the data, and what it does to K2.
date: 2026-08-13
status: PRE-DATA for the σ_run numbers. Premise verification (§1) is ALREADY DONE
        and is reported here because it is a *precondition*, not a result: if the
        three arms had not been post-fix or not been matched, this document would
        have terminated at §1 with no σ_run computed at all.
        No σ_run / bound / K2 verdict existed at the time §3–§6 were written.
decides: K2 ("disagreement drowned by seed variance") — re-adjudicated on a
         data-order-varying σ_run instead of one contaminated by a pre-fix draw.
---

# Pre-registration — σ_run from the post-`ce5c298` keep7 data-order triplet, and K2

## 0. Why this file exists

`STATUS.json:power_analysis` and `PILOT_ONE_PREREG.md` §2.2 adjudicate **K2** using
σ̂ values that come from
`proposal/archive/A03-parametric-vs-external-memory/evidence/a03_sigma_run_n3.json`
(md5 `5fb6cd4c3d693831e50d0817bda93ab8`, verified on wzc1 today).

That file's **`keep7_20k_cpt` family is `seeds [0, 43, 44, 45]`** — and seed `0` is
A03 **Arm 3**, whose own progress log records launch at **2026-08-09 01:11:43**,
i.e. **22 h 09 m BEFORE `ce5c298`** (2026-08-09 23:21:09 +0800). Its training log's
first line is `[seed] set_seed(42)`, and its `DistributedSampler` had no `seed=`.

`PROPOSAL.md` §7.2 states the binding rule verbatim:

> **A pre-fix seed arm and a post-fix seed arm are therefore not draws from the
> same distribution, and must never enter the same `σ_run` estimate.**

So the **keep7 df=3 family violates §7.2**: it pools one pre-fix draw (whose only
stochastic input is fresh-tail init) with three post-fix draws (which additionally
see near-disjoint data subsets, measured rank-0 Jaccard 0.0102). This document
computes the σ_run that §7.2 actually licenses — **seeds 43/44/45 only, df=2** —
and re-runs K2's arithmetic on it.

**This is not a claim that the keep12 family is contaminated.** Seeds 101/102/103
are all post-fix (verified §1.4) and are K2's pre-registered estimator per
`PILOT_ONE_PREREG.md` §2.2. The keep7 family is a *second*, independent σ_run
family. What this document adds is (a) a §7.2-clean keep7 σ_run, and (b) an
explicit statement of what changes and what does not change about K2.

---

## 1. PREMISE VERIFICATION (done before any σ was computed; would have terminated the task)

The task's premise was that the three arms are post-fix and mutually matched. Each
leg below was checkable independently of the σ_run outcome, and any failure was
pre-declared to **stop the analysis and report the failure instead**.

### 1.1 (a) Different `seed=` actually passed — from each arm's OWN log

| arm | `logs/a03_dataorder_seed<S>.log` line 1 | `arch_meta.json:seed` |
|---|---|---|
| seed43 | `[seed] set_seed(43) on all ranks` | `43` |
| seed44 | `[seed] set_seed(44) on all ranks` | `44` |
| seed45 | `[seed] set_seed(45) on all ranks` | `45` |

### 1.2 (b) Launched after `ce5c298` — with a POSITIVE per-run assertion of the fixed line

Not inferred from ckpt mtime (the task correctly warns mtime has already misled
once today). Each arm's own `*_progress.log` carries a preflight `grep` of the
live trainer line, printed **before** `torchrun` was launched:

| arm | preflight line, verbatim | launch |
|---|---|---|
| seed43 | `[08-10 16:55:29] trainer post-ce5c298 OK: 869:        sampler = DistributedSampler(ds, shuffle=True, seed=args.seed)` | `08-10 16:55:29` |
| seed44 | same assertion | `08-10 16:57:29` |
| seed45 | same assertion | `08-11 12:04:07` |

`ce5c298` = 2026-08-09 23:21:09 +0800. All three launches postdate it by ≥17 h.

### 1.3 (c) Config identical except seed — from each arm's OWN log header

| field | seed43 | seed44 | seed45 |
|---|---|---|---|
| arm | `healing_front7+fresh2` | same | same |
| `keep_front_layers` / `n_fresh_layers` | 7 / 2 | 7 / 2 | 7 / 2 |
| `num_hidden_layers` | 9 | 9 | 9 |
| dtype | fp32 master | same | same |
| `world_size / bs / gaccum / eff_bs` | 8 / 8 / 2 / **128** | identical | identical |
| `seq_len` | 2048 | 2048 | 2048 |
| `lr_fresh / lr_inh` | 2e-05 / 2e-05 | identical | identical |
| `max_steps` (cosine horizon) | 300000 | 300000 | 300000 |
| corpus | `dataset rows=15491607 from data/dolmino_now15b.npy` | identical | identical |
| `[optim]` groups | `fresh_decay 339.7M / fresh_nodecay 0.0M / inh_decay 675.3M / inh_nodecay 0.1M`, **all at 2.00e-05** | identical | identical |
| optimizer | `torch AdamW (fp32 optimizer state)` | identical | identical |
| resume source | `outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt` | identical | identical |
| resume LR | `lr_fresh(now)=6.504e-06 lr_inh(now)=6.504e-06` | identical | identical |
| `n_params` | 1015097344 | identical | identical |

**LR is uniform (2e-5 across all four groups). No differential-LR claim is made**
(`A04_GATE_DESIGN.md` §3.1's standing requirement).

### 1.4 Step alignment — the pairing question the task asked about

All three arms have exactly 4 checkpoints: `step{205000,210000,215000,220000}.pt`.
So checkpoints align. **But eval shards exist at `step220000` only** for all three
seeds (`olmo2_{closedbook,mmlu_content}_results/A03_1B_dataorder_seed4{3,4,5}_step220000`).
Therefore:

> **σ_run is computed at exactly ONE step: 220000.** This is a *level* comparison at
> a common step across three runs — which is exactly what a σ_run is — and it needs
> no cross-step pairing. No trajectory / monotonicity / neighbour statistic is
> computed here, and none may be read off this document.

### 1.5 A material asymmetry, stated up front because it BOUNDS the whole result

All three arms **resume from the same `step200000.pt`** — one common fresh-tail
initialisation, restored `strict` (102 tensors) with Adam moments preserved. So:

> **This triplet varies the data subset/order and NOTHING ELSE.** It is a
> *pure data-order* σ_run, not the full run-to-run σ_run that a from-scratch
> multi-seed gate arm would have (which also carries fresh-block-init variance).
>
> The pre-fix families are the mirror image: they varied init only. **Neither is
> the full σ_run.** The keep12 family (101/102/103) *is* full run-to-run variance,
> because those arms prune fresh from the HF base per seed.

This asymmetry is pre-registered here as a **limitation of the keep7 triplet**, and
it is why §6 will not present this σ as a drop-in replacement for K2's estimator.

### 1.6 Protocol, from each arm's own eval logs and the actual driver (not `summary.json:meta`)

The driver that produced all three arms' evals is `/tmp/a03_dataorder_ext_driver.sh`
(still present on `.73`, mtime 2026-08-10 19:13). Its eval invocations, verbatim:

```
scripts/eval_olmo2_mmlu_content.py --base_model $BASE --ckpt $CK --keep_front_layers 7 \
    --n_fresh_layers 2 --output_name $TAG --num_shards 8 --shard_index $g \
    --batch_size 16 --content_desc full
scripts/eval_olmo2_closedbook_qa.py --base_model $BASE --ckpt $CK --keep_front_layers 7 \
    --n_fresh_layers 2 --tasks popqa,triviaqa --output_name $TAG --num_shards 8 --shard_index $g
scripts/eval_olmo2_closedbook_qa.py ... --tasks nq_open --output_name $NQTAG ...
```

* `chat_template`: **the eval scripts contain ZERO `apply_chat_template` call sites**
  (`grep -c apply_chat_template` = 0 in both), and no CLI flag exists to enable one.
  The closed-book script's docstring pins `base protocol (chat_template=False,
  add_special_tokens=False / no BOS by default)`. So `chat_template` is False
  **structurally**, not by flag. The assertion in code is written
  `assert chat_template is not False → FAIL` — i.e. **`is not False`**, never
  `is not True` (a `None` must fail, not pass).
* `add_bos`: `false` in every arm's merged `summary.json` and default `--add_bos 0`.
* greedy: `do_sample=False, num_beams=1`, `max_new_tokens=32`. No few-shot.
* `n_valid=…, nan=0, trunc=0` in every shard log inspected.

### 1.7 Verdict on the premise

**PREMISE HOLDS.** Three arms, post-fix, mutually matched, aligned at step220000,
one common protocol. The analysis proceeds. **Additional finding not in the task's
premise: the *existing* `a03_sigma_run_n3.json` keep7 family is §7.2-noncompliant**
because it includes pre-fix seed 0 — see §0 and §5.

---

## 2. Canonical-import discipline (fixed here, before any number)

1. `build_nulls` is **imported** from
   `code/pilot_zero_rule_disagreement.py` and **called** on the pinned intact
   anchor. No null, Δ, or residual is copied from any `.md` prose. `PROPOSAL.md` §4:
   canonical JSON > prose.
2. Δ is derived as `0.10 · residual(intact, x)` from the freshly-built nulls, then
   **cross-checked** against the canonical Δ recorded in
   `code/a04_sigma_run_independent_recompute.py` (`triviaqa 4.043134195274186`,
   `popqa 1.3205298941613512`, `mmlu_content 1.0238926078906136`,
   `nq_open 0.9695290858725762`) at **full float precision**, tolerance `1e-9`.
   A mismatch aborts. **Δ is never substituted, floored, or re-derived** (guard G2).
3. Intact anchor = `A03_1B_base` / `A03_1B_base_nq` (rule G0), MMLU tie convention
   **`split`** (pre-registered).
4. Shard integrity per cell: shard index set **exactly `{0..7}`** (not a count),
   exact item counts `triviaqa 17944 / popqa 14267 / nq_open 3610 / mmlu 14042`,
   **0 duplicate item_ids, 0 nan**; MMLU read via nested `content_norm.correct`.
5. Seed-disjointness executed with the **self-excluding** `assert_seeds_disjoint`
   from `a04_keep12_trajectory_monotonicity.py`, unweakened. This run claims
   `arm_index` base **900** and guard offset **`SEED+8700`** — disjoint from the
   archived `0,1 / 100..102 / 200..204 / 300,301 / 400..408 / 500..503 /
   600..610 / 700..702 / 800,801`.
6. **All arithmetic on `.73`, numpy 2.5.1** (recorded in the JSON). No number is
   quoted finer than 0.01 pp across nodes (`must_not_claim[24]`).
7. `scipy` is absent on `.73`; the χ² df=2 quantile is the **closed form**
   `CDF(x) = 1 − exp(−x/2)` ⇒ `ppf(p) = −2·ln(1−p)`, asserted `df == 2`. Verified to
   reproduce A03's recorded multiplicative width **12.0707×** at df=2.

### 2.1 Range-statistic constants — declared unused

This analysis computes **no range statistic**. `E[range of 3] = 1.6926 σ` and
`c_8 = 2.8475` are therefore **not used anywhere**, and the σ reported here is a
**sample standard deviation (ddof=1)**, not a range/√-something. Recorded so that
nobody can later reuse a `c_n` from this document.

---

## 3. Estimator, fixed now

For axis `x`, over the S=3 post-fix seeds `{43,44,45}` at step220000:

```
m_s(x)      = 100 · mean(per-item metric)          # arm mean, absolute accuracy
σ_run(x)    = sample sd of {m_43, m_44, m_45}, ddof=1        # df = 2
bound_3(x)  = t_{0.05, df=2} · σ_run(x) / sqrt(3),  t = 2.9199855803537124
χ² 95% CI   = [ σ·sqrt(df/χ²_{0.975,df}),  σ·sqrt(df/χ²_{0.025,df}) ]
            = [ 0.520658·σ,  6.284735·σ ]  at df = 2   (12.0707× multiplicative)
```

**Estimator is the per-axis ARM MEAN per seed (absolute accuracy), NOT a paired
delta** — matching `STATUS.json:sigma_run_input_from_A03.source_of_truth`
verbatim, because a paired delta shares the baseline term across seeds and
understates single-arm spread.

Per `STATUS.json:sigma_run_input_from_A03.standing_rule` (A03 `DATAORDER_PREREG.md`
§4): **no σ_run point estimate is quoted without its d.o.f. AND its χ² interval.**

---

## 4. K2, verbatim, and how it is applied

From `A04_GATE_DESIGN.md` §2:

> **K2 — disagreement drowned by seed variance.** … the one-sided 95% run-level
> bound `t_{0.05,S-1} · sd_run / sqrt(S)` **exceeds the pre-registered
> non-inferiority margin Δ = 10% of the intact arm's own calibrated residual** on
> **≥ 2 of the 4** axes.

Operationalised per `PILOT_ONE_PREREG.md` §2.2, unchanged:

* Decision axes = **triviaqa, popqa, mmlu_content** (3). `nq_open` is **DEMOTED**
  by design §5.2 — computed and reported, **zero decision weight**.
* **K2 FIRES iff `bound_3(x) > Δ_x` on ≥ 2 of the 3 decision axes.**
* Exactly **1 of 3** ⇒ `K2_INDETERMINATE` (`PILOT_ONE_PREREG.md` §2.4). Neither a
  fire nor a clearance.
* **0 of 3** ⇒ does not fire.
* **No 4th seed may be added** to rescue a bound. No axis re-weighting after seeing
  a number. No change of estimator after seeing which answer each gives
  (`STATUS.json:...K2_STATUS_UNCHANGED_BY_SEED45.tempting_but_NOT_LICENSED`).

### 4.1 What result would make me write "K2 FIRES" — declared before looking

| observed | verdict I will write |
|---|---|
| `bound_3 > Δ` on **≥2** of {triviaqa, popqa, mmlu_content} | **K2 FIRES.** A04 dies here, at ~135 GPU-h of already-spent cost instead of 1,077–4,309. |
| on **exactly 1** | **K2_INDETERMINATE** on this family. Not a clearance. Report the fragile axis by name. |
| on **0** | **K2 does not fire on this family** — and per §4.2 that is still **not a clearance**. |

### 4.2 Pre-committed asymmetry (carried verbatim in spirit from `PILOT_ONE_PREREG.md` §2.3)

> **A large σ_run KILLS. A small σ_run does NOT clear K2.**

Three reasons, fixed before the data, specific to *this* family:

1. **Wrong arm.** `keep7+fresh2` = 56.2 % depth, a confirmed **constant-REJECT**
   rung. K2 is a variance gate and is structurally blind to a saturated deficit —
   a reproducibly-terrible arm has *low* variance. (`pilot_one.CRITICAL_CAVEAT`.)
2. **Wrong budget.** 20 000 steps from a warm 200k checkpoint, not the gate's
   from-prune budget.
3. **Partial stochasticity (§1.5).** Common init ⇒ this σ omits the fresh-block-init
   component, so as an estimate of *full* run-to-run variance it is a
   **DOWNWARD-BIASED** estimate, i.e. **optimistic for K2**. A non-firing K2 on a
   downward-biased σ is worth very little; a firing K2 on one is worth a lot.

**Therefore this document can kill A04 but cannot license Pilot Two.**

### 4.3 The χ² upper bound is reported but is NOT a second decision rule

The pre-registered K2 test is on the **point estimate**. `bound_3` at the χ² upper
limit is reported for honesty (df=2 σ is very imprecise: 12.07× multiplicative
width) and **must ship with any K2 statement**, exactly as
`STATUS.json:pilot_one.MAIN_correction_20260812_1630` requires. It is **not** OR-ed
into the verdict. Explicitly pre-committing this because the task flags that a
previous agent OR-ed a weak reading on top of a pre-registered main criterion
today; the failure mode is symmetric and I am foreclosing it in **both**
directions:

* I will **not** write "K2 FIRES" because the χ² upper bound would exceed Δ on ≥2.
* I will **not** write "K2 is cleared" because the point estimate does not fire.

---

## 5. Q2 — the comparison to the pre-fix σ̂, and how it is framed

`a03_sigma_run_n3.json` values that will be compared (canonical, full precision):

| family | df | triviaqa | popqa | nq_open | mmlu_content |
|---|---:|---:|---:|---:|---:|
| `keep7_20k_cpt` seeds {0,43,44,45} — **§7.2-NONCOMPLIANT (mixes pre-fix seed 0)** | 3 | 0.40385537753737 | 0.19587079438308358 | 0.0750142050289564 | 0.055468458650466713 |
| `keep12_5k` seeds {101,102,103} — post-fix, K2's estimator | 2 | 0.30229201489958313 | 0.33279470298495445 | 0.20913668795763832 | 0.0783414… |
| `pooled_df5` — **pools the noncompliant family** | 5 | 0.3666 | 0.2595 | 0.1445 | 0.0656 |

**Pre-registered framing constraint.** The task asks whether real data-order
variance is larger or smaller than the pre-fix σ̂. The honest comparison is
**not** "post-fix σ vs pre-fix σ", because the mixed df=3 family is **not** a
pre-fix estimate — it is a *contaminated* one (3 post-fix + 1 pre-fix draw).
So the ratio I will report is:

```
ratio(x) = σ_run^{43,44,45, df=2}(x)  /  σ̂^{0,43,44,45, df=3}(x)
```

and it will be labelled for what it is: **"§7.2-clean vs contaminated"**, i.e. the
effect of *removing one pre-fix draw*, not a clean pre-vs-post contrast.

> ⚠️ **A clean pre-vs-post contrast is NOT COMPUTABLE and I pre-commit to saying so
> rather than manufacturing one.** It would need ≥2 *pre-fix* seed draws of the same
> arm with evals on the same axes. Checked: the only pre-fix multi-"seed" object in
> the repo is `outputs/olmo2_probe2_7B_keep14fresh2_seed1234`, which is 7B, has **no
> eval shards on either disk**, and is labelled `init-variance only`. A03 Arms 3/4/6
> are pre-fix but are **different LR schedules**, not seed replicates (Arm 4
> `peaklr`, Arm 6 `lowerband`), so their spread is a schedule effect. **Therefore
> "is real data-order variance bigger or smaller than pre-fix init variance" cannot
> be answered from this repo's data, and I will report that, not a proxy.**

Direction of the removal effect is *not* predicted here; both signs are reportable.

---

## 6. Q3/Q4 obligations fixed in advance

**Q3.** K2 verdict per §4 + §4.1, on the point estimate, on 3 decision axes, with
the χ² upper reported alongside per §4.3 and NOT OR-ed in.

**Q4 — the 1B→7B extrapolation, pre-committed as a *bound direction* question.**
Every A04 rung is 7B; this σ is 1B. I pre-commit to stating:

* whether the extrapolation is an **upper** or **lower** bound on the 7B σ_run, **or
  that it is neither/indeterminate** — and if the evidence does not settle it, to
  writing "**cannot be signed**" rather than picking the convenient sign;
* that no 7B σ_run exists or is reconstructible (`must_not_claim[23]`): one seed per
  rung, historical seeds unrecorded, `--seed` postdates the trainer revision;
* that the relevant external evidence (`arXiv:2508.13144` Table 4, OLMo-2 noise at
  1.5B/7B/13B/32B) is **intact**-model noise on a **different harness**, so it may
  inform the *direction* discussion but may **not** be tabulated against A04's
  numbers (`must_not_claim[20]`, and the same cross-harness prohibition the
  literature note itself imposes).

---

## 7. Deliverables and integrity

* `evidence/a04_sigma_run_postfix.json` — canonical; every number in the verdict
  must exist there. Records numpy version, node, `bootstrap_offsets`,
  `seed_disjointness_checked`, per-axis integrity block, and the Δ cross-check.
* `A04_SIGMA_RUN_POSTFIX_K2_VERDICT.md` — prose, subordinate to the JSON.
* `code/a04_sigma_run_postfix_k2.py` (+ driver) — CPU only, no GPU, no model load.
* `STATUS.json` — **pure append** of ONE new key `sigma_run_postfix_k2_20260813`
  (44 → 45), with `gpu_h_spent`. Verified by a **text-level append-only guard**:
  the pre-edit file must be a byte-exact prefix of the post-edit file after
  stripping the closing brace region, AND all 44 prior keys byte-identical when
  re-serialised, AND `len(keys) == 45`.
* `output_name` prefix reserved: **`A04_1B_k7f2_SEED4{3,4,5}_*`** — but note this
  analysis **re-scores nothing** and writes **no eval output dirs**, so no
  collision with existing dirs is possible.
* **GPU: 0.** `.73` is used as a CPU host. `nvidia-smi` asserted clear at dispatch
  (refuse-guard: >8000 MiB held on any card ⇒ exit non-zero).

## 8. Known-unverified / declined

1. Full run-to-run σ_run at keep7 (init + data order jointly) — not measurable from
   this triplet (§1.5). Not funded.
2. σ_run at any rung where NI can be **observed to accept** — still unmeasured; a
   rung-selection problem, not a variance problem (`pilot_one`). Unchanged by this
   document.
3. σ_run at any step other than 220000 for these arms — no eval shards exist.
   Declined; re-scoring 3 arms × 3 further checkpoints × 4 axes is GPU spend that
   buys a trajectory, not a σ.
4. 7B σ_run — non-existent and non-reconstructible. Permanent.
