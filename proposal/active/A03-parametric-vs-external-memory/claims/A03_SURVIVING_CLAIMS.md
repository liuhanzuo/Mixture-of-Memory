---
scope: A03 — THE authoritative statement of what may and may not be claimed
date: 2026-08-11 00:00 GMT+8
status: AUTHORITATIVE. This file overrides every other .md in this proposal directory
        on questions of "what does A03 claim". If any other file's headline conflicts
        with §B here, that other file is wrong and §B says what replaced it.
read_this_first: yes. The directory contains a verdict that was written, retracted,
        withdrawn and replaced within 12 hours. Reading the wrong file resurrects a
        retracted claim.
canonical_trajectory_evidence: evidence/arm3_arm4_arm6_cpt_trajectory_paired_full.json
        md5 28584639f120aaff07bd1a52120f983e (verified identical on wzc1 AND zwfy6)
canonical_floor_evidence: evidence/a03_1b_floor_nulls_4axes.json (wzc1 copy; see §D-4,
        the two disks differ)
protocol_trajectory: per-item paired difference bootstrap, n_boot=5000, seed=42,
        CI95 percentile; SIG = CI excludes 0; baseline = A03_1B_keep7_step200k
protocol_floor: input-blind best-constant / length-matched nulls, n_boot=10000, seed=0,
        BH q=0.05 across 33 cells, plus McNemar
---

# A03 — surviving claims, retracted claims, open questions

## §0. How every number in this file was produced

**Every numeric value below was read programmatically out of the evidence JSONs, not
transcribed from any `.md` prose.** Several `.md` files in this directory contain
numbers that were later corrected; prose is not a source of truth here. The two
sources are:

| file | md5 (wzc1) | what it holds |
|---|---|---|
| `evidence/arm3_arm4_arm6_cpt_trajectory_paired_full.json` | `28584639f120aaff07bd1a52120f983e` | 3 arms × 4 dose points × {popqa, triviaqa, nq_open} × {em, contains, f1} = 108 paired-diff cells. **No MMLU** (see §D-3). |
| `evidence/a03_1b_floor_nulls_4axes.json` | `5e443b424bfde44397bc497a39062504` | 33 arm×axis×interface level-vs-floor cells for the Gate-1 pilot |

Supporting, used only where cited: `evidence/arm3_arm4_arm6_cpt_trajectory_paired_full_MMLUFIXED.json`
(md5 `d64f27b29f5e0126a481c2c82798f1fa`, **untracked at time of writing**),
`evidence/a03_mmlu_family_bh_signtest.json` (md5 `b5383cb1ab809b0bd6b51749e0b30086`,
**untracked**), `evidence/a03_1b_floor_nulls.json` (the earlier 24-cell subset).

Reproduction check performed while writing this file: the closed-book cells of the
canonical file and of `..._MMLUFIXED.json` are **identical across all 108 cells (0
diffs)**, and the 24 cells shared between `a03_1b_floor_nulls.json` and
`a03_1b_floor_nulls_4axes.json` agree on `reported`/`null`/`residual`/`verdict` with
**0 diffs**. So the numbers below are stable across every generation of the analyzer.

---

## §A. SURVIVING claims

Exactly **two** things survive, plus one weakly-suggestive observation. Nothing else.

### A-1. (STRONGEST) The Gate-1 pilot: a pruned+healed 1B retains measurable parametric knowledge above its own construct-appropriate floor

**Claim.** OLMo-2-0425-1B pruned to front-7 inherited layers + 2 freshly-initialised
layers (9 layers total, `keep7+fresh2`), healed for 200k Dolmino steps, scores
**BH-significantly ABOVE its own input-blind null floor on 4 of the 5 knowledge
interfaces that A03 certified as usable at 1B**, while a barely-healed step-500
control of the same topology is at or below floor on every interface.

**Numbers (read from `a03_1b_floor_nulls_4axes.json`, arm `pruned_healed`):**

| interface | reported | null (name) | residual | BH-adj p | verdict |
|---|---:|---:|---:|---:|---|
| MMLU content_norm | 0.3244 | 0.2845 (longest-option, split-tie) | **+3.99 pp** | 1.14e-04 | ABOVE |
| PopQA EM | 0.0394 | 0.0229 (best-constant `association football`) | **+1.65 pp** | 1.14e-04 | ABOVE |
| PopQA contains, length-matched | 0.1119 | 0.0928 (90-char verbose constant) | **+1.91 pp** | 1.14e-04 | ABOVE |
| TriviaQA EM | 0.0959 | 0.0026 (best-constant `australia commonwealth realm`) | **+9.33 pp** | 1.14e-04 | ABOVE |
| NQ-open EM | 0.0285 | 0.0055 (best-constant `2017`) | **+2.30 pp** | 1.14e-04 | ABOVE |
| *MMLU letter* | *0.2512* | *0.2689 (best-constant always-D)* | ***−1.77 pp*** | *3.74e-03* | ***BELOW*** — interface **retired at 1B** |

n: MMLU 14042, PopQA 14267, TriviaQA 17944, NQ-open 3610.
Control (`barely_healed`, keep7+fresh2 @ step500): 10 of its 11 cells are **BELOW**
floor and 1 is **AT** floor (`triviaqa contains`, boot p = 0.399, BH-adj p = 0.399 —
the single non-rejecting cell out of 33). That is what makes "at floor" a detectable
state rather than an untested assumption.

**Exact scope conditions — all of them are load-bearing:**
* one model (OLMo-2-0425-1B), one topology (`keep7+fresh2`, 9 layers from a 16-layer
  base, fresh layer ids 7 and 8 — verified in `outputs/olmo2_probe2_1B_keep7fresh2_16card/arch_meta.json`
  on zwfy6), one healing corpus (Dolmino), one checkpoint (step200000);
* **closed-book parametric knowledge only.** The other three axes A03 originally named
  (conflicting knowledge, multi-evidence, newly-injected facts) are out of scope; see
  `STATUS.json:remaining_axes_status` for the per-axis reason (three retired by
  pathological null floor or construct mismatch, one blocked by design);
* the claim is **"above its own floor"**, i.e. the model is doing more than an
  input-blind constant predictor. It is **not** "recovered", "preserved", or
  "comparable to intact". Intact 16L scores TriviaQA EM 0.4069 vs the pruned+healed
  arm's 0.0959 — a 4.2× gap;
* "4 of 5" counts the certified interfaces {MMLU-content, PopQA EM, PopQA
  contains-lenmatched, TriviaQA EM, NQ-open EM} and excludes MMLU-letter, which is
  retired because the pruned arm falls **below** its floor there and is
  indistinguishable from its own modal-C constant predictor (p = 0.28, emitting C on
  64.4% of items). If you instead count all 11 `pruned_healed` cells in the JSON
  (which double-counts raw and length-matched `contains`), it is 10 ABOVE / 1 BELOW.
  Quote "4/5 certified interfaces", and say which 5.

**★ Why A-1 does NOT inherit the trajectory oscillation defect (§B-1).** This is the
single most important structural point in this document.

A-1 is a **LEVEL** comparison: one checkpoint's score against a floor **computed from
that same checkpoint's own predictions and the label distribution**. The oscillation
defect of §B-1 is a property of **DIFFERENCES between adjacent checkpoints** — the
quantity `score(step_k) − score(step200000)`, which swings by up to 2.66 pp across
5000-step intervals. Three concrete reasons the defect cannot propagate here:

1. **No second checkpoint enters the estimator.** A-1's contrast is
   `score(step200000) − null(step200000)`. There is no `step_k` term, so
   checkpoint-to-checkpoint variance has nothing to act on.
2. **The effect sizes are large relative to both sources of uncertainty.** Against
   bootstrap uncertainty, A-1's residual/half-width ratios are TriviaQA EM **21.6×**
   (+9.33 pp, CI [+8.91, +9.77]), MMLU-content **4.5×** (+3.99 pp), NQ-open EM 3.9×,
   PopQA EM 4.1×, PopQA contains-lenmatched 3.4×. Against the *oscillation* amplitude:
   the largest checkpoint-to-checkpoint swing seen anywhere in §B-1 is 2.66 pp
   (Arm 4 TriviaQA EM), so TriviaQA EM's residual is 3.5× that swing and MMLU-content's
   is 1.5×. The three smallest residuals (+1.65, +1.91, +2.30 pp) are below 2.66 pp,
   so if one pessimistically charged A-1 the full oscillation amplitude those three
   would need the company they have — they point the same way as each other and as the
   two larger cells, on three different benchmarks.
3. **The direction is certified by a negative control, not by a p-value alone.** The
   step-500 arm is BELOW floor on the same interfaces with the same nulls. An
   oscillation artefact would have to move two arms in opposite directions across the
   same set of interfaces, which is not a thing oscillation does.

**Evidence path:** `evidence/a03_1b_floor_nulls_4axes.json` (wzc1);
analyzer `code/analyze_1b_knowledge_floor.py`; per-item inputs on zwfy6 at
`olmo2_closedbook_results/A03_1B_{base,keep7_step200k,keep7_step500}/per_example_{popqa,triviaqa,nq_open}.jsonl`
and `olmo2_mmlu_content_results/` for the same three arms. Narrative:
`GATE_FOURAXES_VERDICT.md` (mind its ERRATUM, §B-4 below) and
`status/A03_1B_FLOOR_VERDICT.md`.

---

### A-2. ❌ **RETRACTED 2026-08-11** (was: WEAKER, HEAVILY CONDITIONED) At step220000, under one fixed sampler seed, two low-LR CPT arms both read TriviaQA EM ≈ +0.5 pp SIG

> **RETRACTION NOTICE.** C-1 (§C-1 below) ran and returned the **ARTIFACT** branch of
> `DATAORDER_PREREG.md` §3.4: **zero** of the two landed seeds is CONFIRM (seed 43 = +0.1115 TIE,
> seed 44 = −0.3455 SIG-negative; band was [+0.20,+0.80]pp). Per that branch's pre-registered
> disposition, the positive reading is **RETRACTED and A03 retains only A-1**.
> Verdict doc: `DATAORDER_VERDICT.md`. Prereg: commit `a25d780` (2026-08-10 19:20:02 GMT+8).
>
> ⚠️ Two wording corrections that also apply to the text below: (a) the manipulation varied
> the **sampler seed**, which at 20k steps × eff-bs 128 = 16.53 % of the 15,491,607-row epoch means
> each seed saw a **different data subset**, not merely a different order — say "sampler-seed /
> data-subset variation"; (b) **no run-to-run variance floor may be quoted** from n=2 — prereg §4
> forbids it.
>
> The text below is preserved unedited as the historical record of what was claimed.

**Claim, stated with every condition attached.** On the **single minibatch sequence
that all three CPT arms shared**, at the pre-registered dose point step220000, both
low-LR arms show a positive significant TriviaQA EM delta versus the
`A03_1B_keep7_step200k` baseline, with `contains` and `f1` also positive and
significant:

| arm | LR band (× peak) | TriviaQA em Δ | CI95 | contains Δ | f1 Δ |
|---|---|---:|---|---:|---:|
| Arm 3 (cosine tail) | [0.325, 0.249] | **+0.4793 SIG** | [+0.2675, +0.6910] | +0.4737 SIG | +0.4678 SIG |
| Arm 6 (lower band) | [0.499, 0.425] | **+0.5016 SIG** | [+0.2564, +0.7356] | +0.3790 SIG | +0.3582 SIG |
| Arm 4 (peak-anchored) | [0.998, 0.559] | −0.9307 SIG | [−1.1927, −0.6576] | +0.8805 SIG | −0.8242 SIG |

n = 17944 for all cells.

**What "phase-locked" means, precisely — this is the sentence that is easiest to get
wrong in either direction.**

*The mechanism.* `ce5c298` ("pass `seed=args.seed` to every shuffling
DistributedSampler") was written on wzc1 and was **not on zwfy6 while Arms 3/4/6
ran**, so all three used `DistributedSampler(ds, shuffle=True)` with the library
default sampler seed 0 and consumed the **identical** minibatch sequence. Training-loss
Pearson r Arm3–Arm6 = **0.99982** (Arm3–Arm4 to step215 = 0.99187). *(Status note
verified 2026-08-11: both disks now carry the fixed line at
`scripts/train_olmo2_arch_probe2.py:869` at identical md5 `284b286f90b526e4e8ad93a68e2a3b16`
— the fix has since been propagated. That does not retroactively unlock Arms 3/4/6,
which ran before the sync.)*

*Error 1 to avoid — overstating.* "Arm 6 replicates Arm 3" is **NOT independent
replication.** Two runs that see the same batches in the same order at the same step
are not two draws from the population of training runs; they are one data path
observed twice at two LR settings. A-2 therefore **cannot** support: "the effect
replicates", "LR is the causal variable" (Arms 3 and 6 differ 1.7× in LR yet land
within 0.02 pp), or "the effect generalises".

*Error 2 to avoid — understating.* The oscillation is **NOT bootstrap noise.** It is
**reproducible**: the arm trajectories trace the same curve. Computed from the
canonical JSON:

| pair, cells used | Pearson r |
|---|---:|
| Arm3 vs Arm6, TriviaQA em, all 4 dose points | **+0.9642** |
| Arm3 vs Arm4, TriviaQA em, first 3 points (before Arm 4's data path broke) | **+0.9974** |
| Arm6 vs Arm4, TriviaQA em, first 3 points | **+0.9992** |
| Arm3 vs Arm4, TriviaQA em, all 4 (incl. Arm 4's broken step220) | +0.5304 |
| Arm3 vs Arm6, **all 36 (step × axis × metric) cells**, all 4 points | **+0.9172** |

So the swings are real signal that is a **deterministic function of data order**, not
sampling error being over-resolved. That is a stronger and more specific statement
than "it's noise", and it is why the decisive experiment is a data-order manipulation
(§C-1) rather than more seeds-of-initialisation or more dose points.

**Exact scope conditions:** one base checkpoint
(`outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt`), one corpus slice
(Dolmino), **one data order**, one dose point (step220000), n = 1 realisation per LR
band, no run-to-run variance floor measured anywhere in this proposal.

**~~A-2 is provisional and is the thing §C-1 is testing.~~ SUPERSEDED 2026-08-11:
§C-1 ran and A-2 is RETRACTED** via the ARTIFACT branch. It may no longer be stated at
all — not with scope conditions, not as a footnote, not as "suggestive". The +0.4793 /
+0.5016 numbers remain in the record only as an example of what a phase-locked sampler
seed can manufacture.

**Evidence path:** `evidence/arm3_arm4_arm6_cpt_trajectory_paired_full.json`;
`code/recompute_cpt_trajectory_paired.py` (hard-fails on <8/8 shards). Narrative:
`ARM6_FINAL_VERDICT.md` (current), with the phase-locking caveat in
`STATUS.json:arm6_midlowLR_cpt.phase_locked_defect`.

---

### A-3. (SUGGESTIVE ONLY — do not headline) Early-CPT damage at step205000 orders monotonically in LR magnitude

At the 5k-step dose point, the damage relative to baseline orders
**Arm 3 < Arm 6 < Arm 4** (i.e. larger LR → larger early loss) on 4 axes, read from
the canonical JSON:

| axis | Arm 3 (0.325×) | Arm 6 (0.499×) | Arm 4 (0.998×) |
|---|---:|---:|---:|
| triviaqa em | −0.4848 SIG | −0.7300 SIG | −1.4044 SIG |
| triviaqa f1 | −0.6300 SIG | −0.8540 SIG | −1.1893 SIG |
| popqa em | −0.3505 SIG | −0.5888 SIG | −0.9673 SIG |
| popqa f1 | −0.8228 SIG | −1.0830 SIG | −1.4400 SIG |

It has a mechanism (larger LR → larger early overshoot), and Arms 3 and 6 share
`warmup=150` so neither carries Arm 4's Adam-moment mismatch. But it is **one dose
point that reverses at the next** (all three arms swing positive at step210000), it is
**3 LR values**, and it is measured on the **same single data order** as A-2 — so it
is at most consistent with a mechanism, not established as a functional form. Do not
promote it, do not fit a curve to it.

---

## §B. RETRACTED claims — what replaced each, and when

Anything in this section is **dead**. If you find it asserted in another file, that
file is stale. Do not resurrect any of it without new data.

### B-1. "20k-step Dolmino CPT recovers parametric knowledge in pruned+healed 1B" (as a general claim)

* **Where it lived:** `ARM3_CPT_TRAJECTORY_INTERIM_VERDICT.md` headline
  ("TRIVIAQA_COHERENT_GAIN…", "3 co-moving SIG cells … are not chance").
* **Replaced by:** A-2 above — the same numbers, but the general reading is gone
  because (a) the trajectory oscillates, and (b) all arms share one data order.
* **Numeric basis (recomputed from the canonical JSON while writing this file):**
  median (checkpoint-to-checkpoint swing) / (mean bootstrap CI half-width) across the
  27 arm×axis×metric combinations = **2.39×**; worst = `arm4_peaklr triviaqa.em` at
  **10.01×** (swing 2.6638 pp vs half-width 0.2661 pp); the headline axis
  `arm3_cosine_tail triviaqa.em` = **4.68×** (swing 0.9641 pp vs half-width
  0.2062 pp). Arm 3's TriviaQA-EM trajectory is
  **[−0.4848, +0.3734, −0.0167, +0.4793] pp, mean +0.0878 pp** — the reported headline
  is the maximum of a series whose mean is **5.5× smaller**.
* **Date / commit:** retracted 2026-08-10, commit `9897a76`; the *interpretation* of
  the retraction was then corrected at commit `5083484` (see B-2).
* **Also killed by the same defect:** any A03 statement of the form "CPT
  recovers/harms parametric knowledge by X pp" derived from a single-checkpoint paired
  diff on this trajectory.

### B-2. "The trajectory-CPT claim is RETRACTED because Arm 6 nulled" (the step-215 verdict)

* **Where it lived:** `ARM6_STEP215_VERDICT.md`, status line "DECIDED — the
  trajectory-CPT claim is RETRACTED. Not 'narrowed', retracted."
* **WITHDRAWN 2026-08-10 16:30, commit `5083484`.** It called the verdict at
  step215000, but `ARM6_LOWERBAND_INTERIM.md` had pre-registered **step220000** as the
  decision point, and step215000 being null was already *expected* (canonical JSON:
  Arm3 −0.0167 TIE, Arm6 −0.0557 TIE, Arm4 −0.1170 TIE). Arm 6 then read
  **+0.5016 SIG** at step220000, firing the `arm6_step220_positive_SIG` branch.
* **Replaced by:** `ARM6_FINAL_VERDICT.md`, summarised as A-2.
* **The banner on `ARM6_STEP215_VERDICT.md` must stay.** Its *arithmetic* (swing/CI
  ratios, sign-contradiction counts) is broadly correct; its *interpretation* ("the
  bootstrap measures the wrong variance / tight CIs were false confidence") is
  superseded — the oscillation is reproducible at r ≈ 0.92–0.999 across arms, so it is
  real data-order-driven signal, not mis-measured noise. **Do not** cite this file as
  the current A03 position on anything.

### B-3. "Peak-LR CPT actively harms parametric knowledge"

* **Where it lived:** `ARM4_PEAKLR_VERDICT.md` §0 ("it is a sign-flip on the primary
  axis").
* **RETRACTED 2026-08-10, commit `9897a76`; the retraction banner is on the file and
  it stands.** Two independent reasons, both still valid after B-2's withdrawal:
  1. **Oscillation.** Arm 4's step220000 −0.9307 SIG sits in a series that also
     contains **+1.2595 SIG at step210000** — a 2.6638 pp swing, 10.01× the bootstrap
     half-width; Arm 4's TriviaQA-EM trajectory mean is **−0.2981 pp**.
  2. **Broken data path.** Arm 4's first step220000.pt was watcher-truncated to
     5,956,287,104 B of 12,181,311,650 B (48.9%) and was redone by resuming from
     step215000. The checkpoint stores `epoch` but **not** the within-epoch batch
     offset, so its last 5k steps replayed the epoch opening; original-vs-redo loss
     correlation over global steps 215020–220000 = **−0.0667**. Arm 4 step220000 is
     therefore **not** a matched-20k-exposure endpoint and must not be compared
     head-to-head against Arm 3/Arm 6 step220000.
* **Replaced by:** nothing. There is no surviving claim about peak-LR CPT. Arm 4's
  step220000 cell may be reported as an *observation on a known-broken data path*,
  never as evidence about LR.

### B-4. "Pruned+healed captures 97.3% of what TriviaQA had" / any headroom reading of the `frac` column

* **Where it lived:** `GATE_FOURAXES_VERDICT.md` defined `frac` as
  `(reported − null) / (1 − null)` ("share of available headroom").
* **ERRATUM already applied in that file, 2026-08-10.** The analyzer computed and the
  JSON published `(reported − null) / reported` — share of **the arm's own reported
  score** that sits above floor. Verified cell-by-cell: all 33 published values
  reproduce under `/reported` and none under `/(1 − null)`.
* **No published number changed; only the prose definition was wrong.** Both columns
  are now in the JSON as `residual_fraction_of_reported` and
  `residual_fraction_of_headroom`. The two readings diverge by an order of magnitude
  for low-scoring arms:

  | arm × axis | `/reported` (published) | `/(1 − null)` (headroom) |
  |---|---:|---:|
  | pruned+healed × TriviaQA EM | 97.33% | **9.35%** |
  | pruned+healed × NQ-open EM | 80.58% | **2.31%** |
  | pruned+healed × PopQA EM | 41.81% | **1.69%** |
  | pruned+healed × MMLU content_norm | 12.31% | **5.58%** |
  | intact × TriviaQA EM | 99.37% | 40.54% |

* **Rule going forward:** `/reported` answers "is this arm doing more than a constant
  predictor" and is the right statistic for A-1's ABOVE/BELOW verdict. It **must not**
  be read as recovery of lost capability. When quoting a recovery-flavoured number,
  quote `residual_fraction_of_headroom`, or better, quote raw scores (pruned+healed
  TriviaQA EM 0.0959 vs intact 0.4069).
* **`frac` is undefined when `reported == 0`.** `barely_healed × popqa × em` has
  `reported = 0.0000` and the JSON correctly emits `residual_fraction = null`; older
  prose printed "n/a". Do not compute a percentage there.

### B-5. "Arm 6's +0.77 SIG at step210000 already exceeds Arm 3's step220000 headline"

* **Where it lived:** `ARM6_LOWERBAND_INTERIM.md` §"step210000".
* **RETRACTED**, and self-retracted in the same file's own §"Why step210000 is NOT
  evidence". All three arms spike positive at step210000 (canonical JSON, TriviaQA em:
  Arm3 +0.3734, Arm6 +0.7746, Arm4 +1.2595) and Arm 4 — the *largest* spike — ends
  significantly negative. Ranking arms by step210000 would have picked Arm 4.
  Superseded by A-2's use of step220000 as the pre-registered point.

### B-6. "NQ-open's best-constant EM floor is 0.0053"

* **Where it lived:** `GATE_NQOPEN_VERDICT.md`, quoted from the eval script's own
  summary.
* **Corrected to 0.0055** (JSON: `nulls.nq_open.em.acc = 0.00554016620498615`,
  best constant `2017`, n = 3610). No verdict changes — intact 0.1025 and
  pruned+healed 0.0285 are ABOVE, barely-healed 0.0017 is BELOW under either value —
  but the two numbers must not both circulate.

### B-7. "The Arm 6 eval watcher's v3 sibling-size guard protected its 4 dose points"

* **Where it lived:** `STATUS.json`'s own description of the Arm 6 watcher.
* **Corrected 2026-08-10** (`DATAORDER_PREREG.md` §5.4 and
  `STATUS.json:dataorder_replication.arm6_size_guard_was_never_armed`): the Arm 6
  watcher resolved `REF_SIZE` **once at startup** (04:58:33) and its first sibling
  checkpoint landed at 07:47:51, so it logged `REF_SIZE=unknown bytes` and guard (b)
  was **silently disabled** for all 4 dose points (0 `REFUSE` lines). Arm 6's points
  were validated on mtime-age + size-stability + `torch.load` only. Arm 4's watcher is
  the one where the sibling-size guard actually fired (12 `REFUSE` lines).
* This does not invalidate Arm 6's numbers — all four of its checkpoints are on disk at
  the identical full size **12,181,311,650 B**, and the ext-driver carries its own
  internal `torch.load` probe — but the *guarantee* that was claimed did not exist.
  Fixed for the data-order runs in commit `097105b` (re-resolve `REF_SIZE` every loop).

### B-8. "CounterFact's in-sample 2AFC role-frequency prior is 64.3%"

* Retracted 2026-08-10, commit `ea9422c`: unreproducible; MAIN measured 57.0%
  in-sample. The canonical CounterFact floor for A03 is the held-out
  **0.5364** (test→train), with 0.5429 train→test. CounterFact remains
  `MEASURABLE_BUT_DEFERRED` (no 2AFC path in the harness), not retired.

### B-9. "BLOCKED_NO_DATA / BLOCKED_WRONG_FORMAT" for the three non-closed-book axes

* Retracted 2026-08-10, commit `314ae49`. All four candidate datasets are on disk
  (`data/knowledge_axes/{counterfact,mquake,hotpotqa,zsre}/`, 197 MB, sha256 in
  `MANIFEST.md`). The real reasons three are unused are **null-floor pathology and
  construct mismatch**, not data availability: MQuAKE best-constant EM floor 0.17733,
  zsRE 2AFC role-frequency prior 0.8565, HotpotQA 10.2% answer-in-question leakage
  plus a reading-comprehension construct. Do not re-file these as "we could not get
  the data".

---

## §C. OPEN questions — no verdict exists

### C-1. **THE** open question: does A-2 survive a change in data order?

**Status: RUNNING, no result. Do not write a verdict for this.** As of 2026-08-11
00:00 GMT+8 neither seed has produced a `step220000.pt`; read-only check on zwfy6
shows `outputs/olmo2_probe2_1B_keep7f2_dolmino_dataorder_seed{43,44}/` each containing
only `step205000.pt` and `step210000.pt`, and **no** `A03_1B_dataorder_seed*`
result directory exists under `olmo2_closedbook_results/` or
`olmo2_mmlu_content_results/`. The three `dataorder_seed*` cells in
`..._MMLUFIXED.json` all read `pending … absent`.

**Pre-registration:** `DATAORDER_PREREG.md` (md5 `ca44ec2c5d01bfc6539cbe9baa22f636`,
identical on both disks), written **pre-data**. It is **BINDING**. Do not re-derive
the rule after seeing numbers.

**The rule, restated (verified against the canonical JSON):**
* Primary endpoint: **TriviaQA `em` ONLY**, at step220000, paired vs
  `A03_1B_keep7_step200k`, same protocol (n_boot=5000, seed=42, CI95).
* θ_ref = (0.479269 + 0.501560)/2 = **+0.4905 pp**; σ_item = half-width/1.96 =
  0.21177/1.96 = **0.108 pp**; assuming σ_run ≤ σ_item, predictive SE ≤ 0.156 pp,
  giving 95% predictive half-width 0.305 pp.
* **REPLICATION BAND = [+0.20, +0.80] pp.** CONFIRM_i ⟺ CI_i excludes 0 **and**
  θ_i > 0 **and** θ_i ∈ [+0.20, +0.80].
* All landed seeds CONFIRM → **REPLICATES** (A-2's narrow reading stands, with its
  scope conditions).
* Zero landed seeds CONFIRM → **ARTIFACT** (A-2 is retracted, this time grounded;
  A03 keeps only A-1).
* **≥1 CONFIRM and ≥1 NOT-CONFIRM → MIXED, which was PRE-DECLARED A FAILURE of the
  claim.** Replication logic here is conjunctive, not majority-vote: one
  non-confirming data order is a direct counterexample. Disposition becomes
  `hold_in_backlog` with a written requirement of n ≥ 5 data orders.
* **NO tie-breaking seed may be added after seeing a split.** That is optional
  stopping and it is forbidden. Seeds 43/44/45 are the whole pre-registered set;
  seed 45 counts only because it was declared in advance (it is queued for .104,
  blocked).
* popqa / nq_open / MMLU and TriviaQA `contains`/`f1` are **SECONDARY** and **may not
  rescue a failed primary**. `contains`/`f1` agreeing with `em` is not extra
  replication: item-level delta correlations on Arm 3 step220000 are
  corr(em,f1)=0.781, corr(contains,f1)=0.497, corr(em,contains)=0.380 → effective
  **1.8–1.9** independent tests, not 3.
* Even the best case is capped: **if exactly 2 land and both CONFIRM, record
  "consistent with a real effect; NOT established"** — n = 2 cannot bound σ_run
  (1 dof, E[s] = 0.80σ). A03 **stays at proposal stage**; the n = 2 caveat is
  mandatory in STATUS.json and in any write-up. Fewer than 2 landing →
  `INCONCLUSIVE_INSUFFICIENT_N`, no verdict.

**Live risk to be aware of before interpreting a null result:** the running driver
`scripts/_run_a03_dataorder_repl.sh` carries the v1 bare-`[ -f ]` trainer stop guard
(kill -TERM; sleep 20; kill -9), the exact race that truncated Arm 4's checkpoint;
measured save window 13 s against a 60 s poll ⇒ ~1-in-5 truncation risk per run. The
watcher **detects** rather than prevents (writes `TRUNCATED_step220000.ALARM` and
scores nothing). Pre-registered remedy: **re-run the full 20k from step200000** with
the v3 guard; **never resume from step215000** (that reproduces Arm 4's
dataloader-offset defect).

### C-2. Is there any run-to-run variance floor at all?

**No.** Nothing in this proposal has a null arm — same data, same LR, same steps, run
twice. Every cross-arm difference in A-2/A-3 is therefore uncalibrated against the
floor it would have to clear. C-1 will supply two draws at n = 2, which falsifies but
does not estimate. Any future positive claim needs this.

### C-3. Does the MMLU axis move at all across the CPT trajectory?

Newly recovered and **not yet integrated into the canonical evidence file**; treat as
provisional. `..._MMLUFIXED.json` (untracked) gives 12 arm×step MMLU cells per
interface. All 12 `letter` cells and 11 of 12 `content_norm` cells are TIE; the single
nominal SIG is `arm6_lowerband step205000 content_norm` at −0.2849 pp
(CI [−0.5697, −0.0071]) and it **does not survive BH** over the post-hoc 24-cell
family (`a03_mmlu_family_bh_signtest.json`: minimum BH-adj p = 0.534, **zero** cells
reject). A post-hoc sign test over the 12 `content_norm` deltas gives 11/12 negative,
two-sided p = 0.00635 (letter: 4/10 negative, p = 0.754) — but the family was
**declared post-hoc, after observing the 1/24 nominal SIG**, so this is a robustness
note, not a finding. Provisional reading: **MMLU is flat**, with a possible small
uniformly-negative content_norm drift that is not established. Do not claim an MMLU
effect in either direction.

### C-4. The 6-arm study A03 was actually proposed to run does not exist

Per `GATE_FOURAXES_VERDICT.md` §6 and the tcodex audit: Arms 1 (intact) and 2
(pruned+healed) are ready; "Arm 3" here is a **CPT trajectory control**, not the
proposal's Arm 3. The proposal's core arms — **raw-text RAG, residual memory, and
joint CPT+memory** — have **never been run**. A03's central
parametric-vs-external-memory question is therefore **untested**. The instrument
(A-1) is certified; the experiment is not built. Note the naming collision:
`ARM4_PEAKLR_VERDICT.md`'s "Arm 4" is the peak-LR control, **not** the 6-arm study's
Arm 4.

---

## §D. Discrepancies found while writing this file

Recorded, not silently fixed, per the instruction that prose is not a source.

### D-1. NEW — the sign-contradiction denominator is now 7/13, not 7/12

`ARM6_STEP215_VERDICT.md` §2 and `STATUS.json:defect_quantified.sign_contradictions`
both say **"7/12 arm-axis combos with ≥2 SIG cells contain two SIG cells of opposite
sign"**, and `ARM6_FINAL_VERDICT.md`'s correction table carries "7/12" forward as
"stands". Recomputed from the canonical JSON (md5 `28584639…`, all 4 dose points × 3
arms): there are **13** combos with ≥2 SIG cells, of which **7** are
sign-contradictory → **7/13**.

The "12" is a fossil of the pre-step220000 evidence: with Arm 6 truncated at 3 dose
points the count is exactly **7/12** (verified by rerunning the same computation on
the Arm-6-first-3-points subset). Arm 6's step220000 cell promoted
`arm6_lowerband triviaqa.contains` from 1 SIG cell to 2 SIG cells (+0.4848 at
step205000, +0.3790 at step220000), adding a 13th combo which is **not**
sign-contradictory. Nobody updated the denominator after withdrawing the step-215
verdict. **Numerator unchanged; the correct figure is 7/13. Interpretation
unaffected** (it gets marginally less severe, 53.8% vs 58.3%).

### D-2. NEW — `frac_combos_over_2x` = 0.5556, not 0.52; `median = 2.39`, not 2.4

`STATUS.json:defect_quantified` records `median_swing_over_bootstrap_halfwidth: 2.4`,
`frac_combos_over_2x: 0.52`, `n_arm_axis_combos: 27`. Recomputed from the canonical
JSON over all 27 combos: median = **2.388** (matches 2.4 to the stated precision) but
**15/27 = 0.5556** exceed 2×, not 0.52. `0.52` is again the pre-step220000 figure
(**14/27 = 0.5185** with Arm 6 at 3 dose points — verified). Same root cause as D-1:
stale denominators carried across the withdrawal. Worst case (10.01×,
`arm4 triviaqa.em`) and headline axis (4.68×, `arm3 triviaqa.em`) both reproduce; the
`.md` prose rounds them to 10.0× and 4.7×.

### D-3. NEW — the canonical trajectory JSON has NO MMLU axis, yet three files call it a "four-axis" trajectory

`ARM3_CPT_TRAJECTORY_INTERIM_VERDICT.md` prints an MMLU-content table with per-cell
CI95 values, `STATUS.json:arm3_cpt_trajectory.mmlu_across_trajectory` states "letter
delta in [−0.23,+0.26]pp all TIE; cn delta in [−0.24,−0.11]pp all TIE", and
`ARM6_LOWERBAND_INTERIM.md` repeats those ranges. **The canonical evidence file
`arm3_arm4_arm6_cpt_trajectory_paired_full.json` contains no `mmlu` key on any of its
12 arm-step cells** (axes present: `popqa`, `triviaqa`, `nq_open` only) — and neither
does the older `arm3_arm4_cpt_trajectory_paired_full.json`. Root cause (per the
`..._MMLUFIXED.json` header): `load_mmlu` read flat key names the harness never
writes, so the axis was **silently dropped**.

Those prose MMLU numbers do now reproduce from the recovered file — Arm 3 letter
[−0.228, +0.263], content_norm [−0.235, −0.114], all TIE — so the *claim* was right.
But it was, for two days, **prose without machine-readable backing**, exactly the
failure mode that `evidence_provenance_fix_20260809` was written to close. The
recovered file is **untracked in git** at time of writing. **Anyone citing an MMLU
trajectory number must cite `..._MMLUFIXED.json` (md5 `d64f27b29f5e0126a481c2c82798f1fa`),
not the canonical file, and should first commit it.**

### D-4. NEW — the two disks' copies of `a03_1b_floor_nulls_4axes.json` differ

wzc1: md5 `5e443b424bfde44397bc497a39062504`. zwfy6:
`816144f45ab9fbb959d42c6aca8e5565`. Diffed field-by-field: the zwfy6 copy is the
**pre-erratum** version — it lacks the top-level `residual_fraction_definition` key
and the per-cell `residual_fraction_of_reported` / `residual_fraction_of_headroom`
fields (66 field-level differences, all of that shape). **All 33 cells agree on
`reported`, `null`, `residual`, `residual_fraction`, `boot_p`, `bh_adj_p` and
`verdict` — no verdict differs.** But a reader on zwfy6 has no way to discover the
B-4 erratum from the JSON, which is precisely the trap that erratum exists to close.
Also note the wzc1 file is **modified but uncommitted** in git. Recommended: commit
the wzc1 version, then `scp -O` it to zwfy6 and re-verify md5. (Not done here — this
lane is documentation-only.)

### D-5. Pre-existing, minor — `bh_k` is the count of *rejections*, not the family size

Both floor JSONs carry `n_cells` and `bh_k` differing by exactly 1
(24/23 and 33/32). Reconstructing the BH-adjusted p-values shows the family size used
in the arithmetic is `n_cells` (min BH-adj p = 1e-4 × 33/29 = 1.1379e-04 for the
33-cell file, and 1e-4 × 24/22 = 1.0909e-04 for the 24-cell one), while `bh_k` equals
the number of cells that reject at q = 0.05. The field name invites reading it as the
family size. **BH was applied over 33 cells** (and 24 in the earlier file). No number
changes; do not quote "BH over 32 tests".

### D-6. Pre-existing, resolved — `evidence_md5` in STATUS.json points at a superseded file

`STATUS.json:arm6_midlowLR_cpt.evidence_md5` = `36fed7ad8cce952c2c406c4abad80da7`,
which the sibling key `evidence_file` in the same block correctly describes as
**superseded** by `28584639f120aaff07bd1a52120f983e`. Two keys in one block disagree.
The canonical value is **`28584639f120aaff07bd1a52120f983e`** (verified present at
that md5 on **both** wzc1 and zwfy6). Left in place rather than edited, since
`STATUS.json`'s retraction history is itself the record; flagged here so nobody uses
the stale hash as a provenance check.

---

## §E. One-screen summary

| # | claim | strength | may be published as-is? |
|---|---|---|---|
| A-1 | pruned+healed 1B `keep7+fresh2` @200k is BH-significantly above its own construct-appropriate null on 4/5 certified closed-book interfaces; step-500 control at/below floor | **solid**; level-vs-floor, immune to §B-1 | yes, with its scope conditions and with `/reported` ≠ recovery (B-4) |
| A-2 | ~~at step220000, on one fixed sampler seed, Arm 3 (+0.4793 SIG) and Arm 6 (+0.5016 SIG) both show TriviaQA EM ≈ +0.5 pp~~ | ❌ **RETRACTED 2026-08-11** — C-1 returned the ARTIFACT branch (0/2 seeds CONFIRM; −0.3455 SIG-negative on seed 44) | **never** — see `DATAORDER_VERDICT.md` |
| A-3 | early-CPT (step205000) damage orders Arm3 < Arm6 < Arm4 on 4 axes | **suggestive only** | no |
| B-1..B-9 | see §B | **dead** | never |
| C-1 | sampler-seed replication, seeds 43/44 | **CLOSED 2026-08-11 → ARTIFACT** (0/2 CONFIRM) | yes, as a negative result: `DATAORDER_VERDICT.md` |
| C-2 | run-to-run variance floor | **does not exist** | — |
| C-3 | MMLU trajectory | **provisional**, untracked evidence | no |
| C-4 | the 6-arm parametric-vs-external-memory study | **never run** | — |
