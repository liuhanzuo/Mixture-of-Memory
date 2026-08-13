# A04 — PRE-REGISTRATION: the shallow-rung ladder (1B `keep13`/`keep14`+fresh2)

**Committed BEFORE any margin, deficit, accuracy or recovery number for either
new arm existed.** At the moment of this commit the two arms are ~20 minutes into
a 5,000-step training run and have produced no checkpoint and no eval shard, so
there is literally nothing to have peeked at. §7 records the full inventory of
what *was* visible in advance (all of it pre-existing keep7/keep12/full32
material, already published in A04) and why it does not launder the predictions
below.

**Date:** 2026-08-13 · **Training launched:** `.73` 18:36:30, `.82` 18:39:30
(both +0800) · **Planned GPU:** ~66 GPU-h total (2 arms × 8×H20 × ~4.15 h)
**Nodes:** `.73` (keep14) and `.82` (keep13), 8×H20 each, zwfy6 disk.
**Not touched:** `.104` (paperC Qwen3-8B heal, PID 3343471), `LOCAL` + `.21`
(SparseForge #246).

---

## 1. The blocker this is written against, verbatim

`STATUS.json:pilot_one.pilot_two_status`:

> **BLOCKED.** 1,077–4,309 GPU-h must not be committed until a NEW pre-data doc
> shows a rung exists where NI can be **OBSERVED TO ACCEPT**; otherwise the gate
> can only ever confirm rejection.

`STATUS.json:pilot_one.what_remains_genuinely_unmeasured` (same key):

> NOT measured: `sd_run` for any 1B arm at a rung where NI can be observed to
> ACCEPT. Neither keep7 nor keep12 is such a rung (both constant-REJECT) … More
> seeds on either family cannot fix that — **it is a rung-selection problem, not
> a variance problem.**

This document is that new pre-data doc. It does **not** attempt to clear the
blocker by argument; it commits, in advance, to both possible readings of a
measurement that has not been made.

## 2. The hole in NI's discrimination curve

Every NI verdict A04 has ever produced, with its recovery fraction:

| arm | scale | damage | recovery (fraction of intact residual) | NI |
|---|---|---|---|---|
| `keep7+fresh2` @220k | 1B | 9/16 kept = **43.75 % cut** | 11.88–36.64 % | REJECT 4/4 |
| `keep12+fresh2` @5k | 1B | 14/16 kept = **25 % cut** | 20.23–31.77 % | REJECT 4/4, by **27.0–90.4 × `sd_run`** |
| `keep14+fresh2` @200k | 7B | 16/32 kept = 50 % cut | 25.3–53.1 % | REJECT 3/3 |
| `shortgpt16` @200k | 7B | 16/32 = 50 % cut | 22.2–62.7 % | REJECT 3/3 |
| `..._freezefront` @200k | 7B | 16/32 = 50 % cut | 22.1–41.0 % | REJECT 3/3 |
| `..._fromscratch` @200k | 7B | 16/32, zero inheritance | 11.6–40.5 % | REJECT 3/3 |
| **`full32_dolmino` @25k** | 7B | **ZERO structural damage** | 97.7 % (mmlu) | **ACCEPT 1/3** ← the only accept |

Sources: `evidence/pilot_one_stage_b_falsifiability.json`,
`evidence/a04_shallow_rung_ni_7b.json`, `evidence/a04_control_arms_ni.json`,
`STATUS.json:shallow_rung_ni_discrimination_20260812`.

**The gap is enormous and empty.** On one side, damaged arms cluster at 11–63 %
recovery and reject by tens of standard errors. On the other, a zero-damage arm
accepts. Between "25 % cut → 22–32 % recovery → REJECT by ≥27 SE" and "0 % cut →
97.7 % recovery → ACCEPT" there is **not a single measured point**.

**And no shallower rung exists on either disk.** Verified 2026-08-13:
`STATUS.json:control_arms_ni_20260813.what_was_tested` records that "shallower
rungs (keep16/20/24/28+fresh2) have **0 checkpoints on either disk**"; MAIN
independently confirmed `outputs/olmo2_probe2_7B_keep16fresh2/` holds only
`arch_meta.json` on zwfy6 and does not exist at all on wzc1. At 1B, `keep12` is
already the lightest damaged rung ever trained, and the base has only 16 layers
(`models/OLMo-2-0425-1B/config.json` → `num_hidden_layers: 16`, re-asserted by
the launcher's preflight before either run started).

So the blocker cannot be discharged by any re-analysis. It requires new training,
and this is the cheapest new training that can address it.

## 3. The two arms, and why exactly these two

Protocol is **Pilot One Stage B, verbatim**. Every hyper-parameter was read out of
`outputs/olmo2_probe2_1B_keep12f2_dolmino_stageB_seed101/step5000.pt`'s own
`train_args` dict — not from any prose table. The **only** quantity that differs
is `--keep_front_layers`.

| node | arm | `keep_front` | `n_fresh` | depth | cut | seed | `output_dir` |
|---|---|---|---|---|---|---|---|
| `.73` | shallow rung 1 | **14** | 2 | 16 | **2/16 = 12.5 %** | 101 | `outputs/olmo2_probe2_1B_keep14f2_dolmino_shallow_seed101` |
| `.82` | shallow rung 2 | **13** | 2 | 15 | **3/16 = 18.75 %** | 101 | `outputs/olmo2_probe2_1B_keep13f2_dolmino_shallow_seed101` |

Frozen config (identical on both, identical to Stage B):
`data_path=data/dolmino_now15b.npy` (zwfy6, 126,907,244,672 B — asserted, because
wzc1's same-named file is a **different corpus** at 62,020,903,040 B) ·
`model_path=/apdcephfs_zwfy6/.../models/OLMo-2-0425-1B` · `n_fresh_layers=2` ·
`freeze_front=false` · `from_scratch=false` · `random_trunk=false` ·
`max_steps=5000` · `seq_len=2048` · `batch_size=8` · `grad_accumulation_steps=2`
(⇒ eff_bs 128 at world=8) · `lr=lr_inherited=2e-5` → `min_lr=min_lr_inherited=2e-6`
(**uniform LR — no differential-LR claim is made or licensed**) ·
`warmup_steps=150` · `optimizer=adamw` (fp32 master weights) · `weight_decay=0.1`
· `grad_clip=1.0` · `save_every=2500` · `milestone_every=5000` · `keep_last_n=3`
· `log_every=20` · `gradient_checkpointing=1`.

**Why keep14.** It is the lightest damaged rung that exists as a *concept*, not
merely the lightest one not yet run. `keep15+fresh2` would be 17 layers —
**deeper than the 16-layer base** — at which point the arm is no longer a cut of
the base and "recovery from damage" has no referent. So keep14 is the boundary of
the damaged family.

**Why keep13.** One intermediate point, so the curve has **four** rungs
(keep12 / keep13 / keep14 / zero-damage) rather than two. A two-point curve
cannot distinguish "NI's accept region begins somewhere below 12.5 % damage"
from "NI's accept region is a single point at 0 % damage".

**Why seed 101 on both.** Same seed as Stage B seed101, so if a future pass wants
to extend either arm to S=3 it inherits a seed already in the pre-registered set
{101, 102, 103} and the σ_run family stays coherent. **No σ_run is computed here**
(§6.4).

### 3.1 keep14+fresh2 is 16 layers = base depth, and is STILL DAMAGED

This must not be misread. `keep_front=14, n_fresh=2` gives
`cfg.num_hidden_layers = 16`, the same depth as the base — but base layers **14
and 15 are discarded** and replaced by **random-init Olmo2 layers**. The arm
inherits 14 of 16 pretrained layers, not 16.

The zero-damage control is a *different* construction: `n_fresh_layers=0`, every
layer transplanted, continued-pretraining only — that is what `full32_dolmino`
is. **keep14+fresh2 is not that, and must never be reported as a zero-damage
arm.** Equally, `full32`-style CPT must never be substituted as this ladder's
anchor (guard G2, §6.1).

### 3.2 GATE0: no degeneracy at `keep_front + n_fresh == base_layers`

Run **before** any 8-GPU commitment (1 GPU, 20 steps, `/tmp` output, 2026-08-13
18:27–18:30), because a degeneracy at the boundary would have invalidated the
whole design:

| probe | copied tensors | expected `3 + 11·keep` | fresh layer ids | `max｜model−base｜` | fresh `post_attn_ln` / `q_norm` all-ones | fresh `q_proj` std | reached |
|---|---|---|---|---|---|---|---|
| keep14+fresh2 | **157** | 3+11·14 = 157 ✓ | `[14, 15]` | `0.000e+00` | True / True | 0.020001 | step 20, exit 0 |
| keep13+fresh2 | **146** | 3+11·13 = 146 ✓ | `[13, 14]` | `0.000e+00` | True / True | 0.019997 | step 20, exit 0 |

All 6 trainer asserts pass on both. Source reading confirms why:
`transplant_front()` (`scripts/train_olmo2_arch_probe2.py:170`) selects base keys
by `lid < keep_front_layers` against the **base** state dict, and the expected
fresh set is `range(keep, keep+n_fresh)` computed on the **new** cfg — so 14+2
behaves exactly like 12+2 with two more inherited layers. There is **no special
branch** for `keep + fresh == base_layers`; the only conditional in that function
is `if n_fresh_layers > 0` (which skips the fresh-init assert for the
`n_fresh=0` CPT control), and both arms here have `n_fresh=2`.

Optimizer groups observed at GATE0 (recorded so no differential-LR claim can be
retrofitted): keep14 → `fresh_decay` 339.7 M @2.00e-05, `inh_decay` 1145.0 M
@2.00e-05, `inh_nodecay` 0.1 M @2.00e-05; keep13 → 339.7 M / 1077.9 M / 0.1 M,
all at 2.00e-05. **Uniform LR across all groups**, as in Stage B.

### 3.3 Positive preflight assertions, printed before launch

Both progress logs carry, *before* the launch line (Stage B convention — an
assertion that is printed is checkable; one that is claimed in prose is not):

```
PREFLIGHT-ASSERT trainer post-ce5c298: 869:  sampler = DistributedSampler(ds, shuffle=True, seed=args.seed)
PREFLIGHT-ASSERT trainer md5: 284b286f90b526e4e8ad93a68e2a3b16
PREFLIGHT-ASSERT base num_hidden_layers=16; cut = {2,3}/16; total depth = {16,15}
preflight OK: dolmino=119G (126,907,244,672 B asserted exactly)
GPUs clear (0MiB held)
```

Without the `seed=args.seed` fix a "seed" moves only the fresh-tail init and not
the data order (`PROPOSAL.md` §7.2), so any future σ_run over these arms would be
inadmissible. Both arms are **post-fix**, on the same side of the ce5c298 break
as the Stage B family.

---

## 4. THE DECISION RULE, fixed now

The rule, the axes, the margin and the anchor are all **imported, not
re-derived** (§6.1). Nothing in this section is a free parameter at analysis time.

* **Decision axes (3):** `triviaqa`, `popqa`, `mmlu_content`. `nq_open` is
  **DEMOTED** by design §5.2 (its item-level 95 % CI half-width already exceeds
  its own Δ at n=3610) and carries **zero decision weight**. It is reported
  descriptively.
* **Δ = 0.10 × residual(intact)**, never substituted, never floored (guard G2):
  `triviaqa` 4.043134195274186 pp · `popqa` 1.3205298941613512 pp ·
  `mmlu_content` 1.0238926078906136 pp · `nq_open` 0.9695290858725762 pp.
  These are re-derived at runtime by calling the imported `build_nulls()` on the
  pinned anchor and **cross-checked** against the above at 1e-9; a drift aborts.
* **NI(Δ) ACCEPTS** iff the one-sided lower 95 % bound on
  `residual(arm) − residual(intact)` is `> −Δ`, i.e.
  `margin_pp = diff_lower95_one_sided_pp + delta_pp > 0`. Computed by the
  imported `ni_rule`, which is the same function that produced every archived A04
  margin. The null cancels exactly in the paired difference, so the deficit is
  convention-independent; only Δ depends on the MMLU tie convention.
* **MMLU tie convention:** pre-registered `split`. All five of
  `{split, first, last, credit, wrong}` are reported as a sensitivity, and a
  verdict that holds under `split` but not the others must be flagged.
* **The bar is ≥ 2 of the 3 decision axes**, matching every prior A04 accept/
  reject statement (`full32` was reported as "1 of 3 → below the 2/3 bar").

### 4.1 The two branches, both written now

**Branch A — ACCEPT.** If `keep14` **or** `keep13` shows NI ACCEPT on **≥2 of 3**
decision axes, then a rung exists at which NI can be *observed to accept under
structural damage*, and
`STATUS.json:pilot_one.pilot_two_status`'s blocker — "a NEW pre-data doc shows a
rung exists where NI can be OBSERVED TO ACCEPT" — is **DISCHARGED**. The
consequence is *specifically* that: the blocker lifts. It is **not** an approval
of Pilot Two, which has an independent binding objection
(`control_arms_ni_20260813.recommendation.reason_2_new`: the decision metric can
be reordered by output length, and that is a design fix, not an *n* fix). Both
must clear before 1,077–4,309 GPU-h is priced.

**Branch B — REJECT.** If **both** new rungs are constant-REJECT, then the
recorded finding is: **NI's accept region at 1B is narrow enough that it contains
no damaged rung at all, down to a 12.5 % cut — the lightest cut the family
admits.** Under Branch B the accept region is bounded to lie strictly between
"discard 2 of 16 layers" and "discard none", which means NI as written
distinguishes *damaged* from *intact* rather than *recovered* from
*unrecovered*. That is a **negative but publishable** verdict on the A04
certification rule, and it is exactly what §4.2 of the promotion criteria calls a
finding that changes a scientific conclusion. **It is not a failure and it will
not be dressed as a success.** Under Branch B, `pilot_two_status` stays BLOCKED
and the blocker is recorded as **undischargeable by rung selection at 1B**.

**Branch C — mixed (exactly 1 of 3 axes accepts on one or both arms).**
Pre-committed as **INDETERMINATE**, by the same convention that put `full32`'s
1-of-3 below the bar. An INDETERMINATE result does **not** discharge the blocker.
Recording this branch in advance is what stops a single-axis accept from being
promoted after the fact.

### 4.2 The noise gate, with the constant that matches *k*

No range or spread statistic is decision-bearing in this pass (§6.4), but if any
is reported it must clear
`E[range of k iid N(0,σ)]/σ`, which is **k-dependent**:

| k | constant | source |
|---|---|---|
| 2 | **1.1283791670955126** | `2/√π`, closed form |
| 3 | **1.6925687506432689** | `3/√π`, closed form |
| 8 | **≈ 2.8472** | Monte Carlo, no closed form |

Using k=3's 1.6926 for k=8 makes the floor **40.6 % too low** and can manufacture
a finding (this is recorded in
`A04_KEEP12_TRAJECTORY_MONOTONICITY_VERDICT`). Whichever *k* is used must be
stated with its constant. **A ratio of two ranges is UNDEFINED when neither range
clears its own floor** — it is not "a direction". This is the exact error that
voided the within-arm-LR pass earlier today
(`within_arm_lr_refutation_20260813`), and it is pre-committed here so it cannot
recur.

---

## 5. What this pass will NOT claim, fixed in advance

1. **No σ_run, no seed-variance, no K2 statement.** One seed per arm. A σ over
   two arms of *different depth* is not a run-to-run σ.
2. **No PLATEAU(T) comparison.** No in-domain val PPL trajectory is produced for
   either arm, so the NI-vs-PLATEAU disagreement cannot be evaluated here.
3. **No depth-ladder scaling law.** keep12 (Stage B), keep13 and keep14 share
   corpus, step count, protocol and seed, so keep12/13/14 **are** mutually
   comparable as a 3-point 1B depth ladder — but they are not comparable to the
   7B ladder (`STATUS.json:warning`'s two-corpora confound), and 5,000 steps is
   not a converged heal.
4. **No differential-LR claim.** All four optimizer groups run at 2.00e-05
   (measured at GATE0, §3.2).
5. **No trajectory / monotonicity / neighbour claim.** `save_every 2500` yields
   step2500 and step5000 only; a 2-point series has one difference and cannot
   support a trend.
6. **No recovery FRACTION quoted without its zero-inheritance floor** where a
   floor exists (`must_not_claim` item 28). No 1B `--from_scratch` floor exists,
   so 1B recovery fractions are reported **as fractions of the intact residual
   only**, with that limitation stated.
7. **No claim that any margin difference between keep13 and keep14 is
   "measured"** unless it clears the item-level bootstrap SE. A difference inside
   the SE is unresolved, not a direction.
8. **No format/verbosity-free reading of the generative axes.** A04 has two
   independent demonstrations that a generative-EM axis partly measures output
   length (`PROPOSAL.md` §4.4: 47.37 % of an EM *loss* was verbosity;
   `control_arms_ni_20260813` P3: 50.00 % of an EM *gain*, and it **reordered two
   arms**). Any `triviaqa`/`popqa`/`nq_open` statement here inherits that
   caveat. `mmlu_content` is length-free by construction.

---

## 6. Method discipline (each item is an executed assertion, not a promise)

### 6.1 Anchor and Δ are pinned; the canonical code is imported

* Intact anchor = **`A03_1B_base`** (`mmlu`, `cb`) + **`A03_1B_base_nq`** (`nq`),
  the **vanilla** `models/OLMo-2-0425-1B`, pinned by guard **G0**
  (`evidence/a04_g0_anchor_sha256_pinning.json`).
* **Guard G2 forbids substituting a continued-pretrained model as anchor.** At 7B
  this was shown to *manufacture accepts*: `full32_step25000` scores **below**
  vanilla base on all four axes, so substituting it would shrink every Δ **and**
  lower every target. No CPT arm may be the anchor here either.
* `ni_rule`, `ratio_rule`, `load_shards`, `build_nulls`, `mmlu_content_norm_vec`,
  `qa_metric_vec`, `EXPECTED_N`, `AXES`, `DEMOTED_AXES`, `PREREG` are **imported
  from `pilot_zero_rule_disagreement`**; `paired_bootstrap`, `TIE_CONVS`,
  `N_BOOT`, `SEED` from A03's `analyze_1b_knowledge_floor` via
  `proposal_paths.a03_code_dir()`. **No metric, null, rule or guard is
  reimplemented** — two subagents have already produced spurious significance by
  reimplementing a metric, and MAIN's own hand-subtraction of a recorded null was
  ~0.5 pp off twice.
* **No constant is transcribed by hand.** Reference values are read from the
  canonical JSON **at runtime**. The `control_arms` pass caught its own
  hand-transcribed constant at 8.82e-05 pp and the fix was to remove the
  transcription step, not to loosen the tolerance.

### 6.2 Shard integrity, asserted before any statistic

Per (arm × axis): shard index set **exactly `{0..7}`** (a *set*, not a count of 8
files), merged `n` **exactly** `EXPECTED_N` (`triviaqa` 17944 / `popqa` 14267 /
`nq_open` 3610 / `mmlu` 14042), **0** duplicate `item_id`, **0** nan, and
`item_id` sequences **identical across all arms and the anchor** (`assert_aligned`)
— without which the paired difference silently compares different items. A
silently merged 5-of-8 set has corrupted results in this repo before.

### 6.3 Protocol invariants

`add_bos` asserted **`is False`** — never `is not True`, so a missing or `None`
value FAILS. `chat_template` asserted **`is not False` → FAIL**, and additionally
structurally: neither eval script contains an `apply_chat_template` call site.
These are BASE LMs (no SFT/RL); any chat=True number is void.
`max_new_tokens == 32`. Eval batch sizes: closed-book `bs=32`, mmlu_content
`bs=16` (the Stage B driver's own values, read from the driver source, not
assumed).

### 6.4 One node for all statistics

`.73` numpy **2.5.1**, `.82` numpy **2.4.6**. `Generator.multinomial` differs in
**19 of 10,000** rows between them (max margin drift 0.005294 pp), so **every
statistic is computed on ONE node** and the node + numpy version are recorded in
the evidence JSON. **Node of record: `.73`.** Training is unaffected by this;
only bootstrap statistics are.

### 6.5 Bootstrap seed offsets, checked mechanically

Archived and in use: `{0,1}` (pilot_zero) · `100–102` (step100k) · `200–204`
(shallow_rung) · `300,301` (keep14 traj) · `400–408` (neighbour var) · `500–503`
(full32 traj) · `600–610` (keep12 traj) · `700–702` (keep10 neighbour) ·
`800,801` (control arms, +archived 201) · `900–902` (σ_run postfix) ·
`1000–1005` (within-arm LR). This pass claims **`arm_index 1100+`** and
**guard offset 9700**. The disjointness is **EXECUTED** by
`assert_seeds_disjoint`, which reads every archive's own recorded offsets and
raises on intersection — prose claims of disjointness in this repo have been
wrong before, and the executed check has already caught one real collision.

### 6.6 Append-only STATUS.json

The new key is the **47th** (46 at the time of writing; a concurrent A04 pass
could land one, and the `control_arms` pass observed exactly that). The writer
therefore **trusts no hardcoded count**: it snapshots every pre-existing key's
serialised bytes, asserts `count == old + 1`, asserts the old key **order** is
unchanged, asserts each old key is **byte-identical**, **and** asserts the whole
new file is a **byte-PREFIX extension** of the old one — then restores from
backup if any check fails. The file is **`indent=2`**; the writer **derives the
format by round-tripping the original bytes** rather than assuming it, because
`indent=1` against an `indent=2` file rewrote all 2,643 lines of an append-only
record while every per-key byte check correctly passed.

---

## 7. Everything that was visible before this commit

Full inventory, so the predictions cannot be laundered:

* **Nothing at all about either new arm.** At commit time both runs are ~20 min
  into 5,000 steps: no `step2500.pt`, no `step5000.pt`, no eval directory, no
  accuracy, no margin. The training logs' loss values were **not** read, and loss
  is not a decision statistic here in any case.
* **Everything about the prior arms**, all already published in A04 and quoted in
  §2 with its source: keep7/keep12 recovery and REJECT margins, the 7B ladder,
  and `full32`'s single accept. These are precisely why keep13/keep14 were
  chosen; §2 is the *motivation*, and the motivation is public.
* **GATE0's structural output** (§3.2): tensor counts, layer ids, init asserts.
  These are *architecture* facts, not capability facts. GATE0 wrote to `/tmp`,
  ran 20 steps, and produced no eval.
* **Δ and the anchor** were fixed in `PILOT_ZERO_VERDICT.md` §1 long before any
  keep13/keep14 datum existed, and are quoted here to prevent later
  re-margining, not chosen here.

The predictions in §4.1 are stated over **margins and orderings** of quantities
that do not yet exist.

---

## 8. Cost, and why it does not need approval

Stage B measured **4.15 h/seed on 8×H20 = 33.2 GPU-h/arm**; two arms in parallel
on two nodes ⇒ **~4.2 h wall, ~66 GPU-h total**, plus ~5 min/arm for the 4-axis
eval (Stage B's measured eval cost). That is **1.5–6.1 %** of Pilot Two's
1,077–4,309 GPU-h, and it is the only expenditure that can decide whether Pilot
Two's blocker is dischargeable at all. Standing autonomy applies (2026-08-09
user directive); this is neither an irreversible outward action nor a large
speculative tranche.

The forbidden nodes are protected by an **IP-based refuse guard** in
`scripts/_run_a04_shallow_ladder.sh` (`.104` → exit 11, `.21` → exit 11) plus a
`>8000 MiB` GPU-held guard, so the budget is enforced by the launcher rather than
by the operator's memory.
