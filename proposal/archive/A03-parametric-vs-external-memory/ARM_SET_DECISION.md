---
scope: A03 — execution of `STATUS.json:next_gate[0]`, the no-GPU decision gate.
       "Is the 6-arm plan still worth ~N GPU-h now that A-2 is retracted?
        Re-derive the minimum arm set that can still produce a publishable result."
date: 2026-08-11 21:10 GMT+8 (σ / MDE / CPT-increment numbers updated 2026-08-12 00:10 after sampler seed 45 landed — every superseded value is struck through in place, never silently replaced)
gpu_spent: ZERO by this document. Every number here is recomputed from JSONs already on
       disk, plus read-only `ls`/schema inspection on zwfy6. No job launched, no weights
       downloaded. (The 2026-08-12 update consumed seed 45, whose ~91 GPU-h was already
       committed before this gate ran, plus ~4 min of 8×H20 eval.)
recommendation: ARCHIVE (see §3) — **UNCHANGED by seed 45; reinforced from both sides**
       (§3.6, and `SEED45_VERDICT.md` §4.1: the MDE bar barely moved while the effect
       being chased crossed zero). Three assets migrate rather than being discarded (§3.4);
       the physical move was BLOCKED on two hardcoded path dependencies (§3.5) and has
       since been unblocked by lifting the loaders to `proposal/shared/code/`.
authority: this file decides the arm-set question only. It does NOT re-open
       `DATAORDER_VERDICT.md` (branch ARTIFACT, standing — now 0/3 CONFIRM) and does NOT decide A04 (§4).
reads_as_fact: claims/A03_SURVIVING_CLAIMS.md (A-1 survives, A-2 retracted),
       DATAORDER_VERDICT.md, SEED45_PREDECLARATION.md, SEED45_VERDICT.md, DATAORDER_PREREG.md §3.4/§3.6/§4
---

# A03 — minimum arm set decision

## §0. Executive summary

**Recommendation: `ARCHIVE`.**

The 6-arm study cannot be rescued by removing the dead arms, because the arms that
die are precisely the ones that carried the *scientific* question, and the arms that
survive are the ones whose answer is either (a) already measured at zero, (b)
guaranteed-large-but-uninformative, or (c) dependent on assets that do not exist and
on a thesis a sibling proposal (A02) has already measured negative.

The decisive quantity is not novelty and not cost. It is the **minimum detectable
effect (MDE) of this apparatus versus the effect size each candidate arm set targets.**
A03's own instability is now measured on two different arms:

| arm | axis | S | s (pp) | d.o.f. | χ² 95 % CI for σ (pp) |
|---|---|---:|---:|---:|---|
| keep7+fresh2, 20k CPT (sampler seeds 0 / 43 / 44 / **45**) | triviaqa em | **4** | **0.4039** | **3** | **[0.229, 1.506]** |
| keep12+fresh2, 5k (seeds 101 / 102 / 103) | triviaqa em | 3 | **0.3023** | 2 | [0.157, 1.900] |
| keep12+fresh2, 5k | popqa em | 3 | 0.3328 | 2 | [0.173, 2.092] |
| keep12+fresh2, 5k | mmlu content_norm | 3 | 0.0783 | 2 | [0.041, 0.492] |
| keep12+fresh2, 5k | nq_open em | 3 | 0.2091 | 2 | [0.109, 1.314] |
| **pooled, triviaqa em, both arms** | triviaqa em | 4+3 | **0.3666** | **5** | **[0.229, 0.899]** |

> **UPDATED 2026-08-12 00:10 (seed 45).** ~~keep7 20k triviaqa s = **0.4132**, df = **2**,
> χ² **[0.215, 2.597]**; pooled σ = **0.3620**, df = **4**, χ² **[0.217, 1.040]**~~
> **SUPERSEDED** — sampler seed 45 landed as the 4th keep7 draw. The old values are struck
> through, not deleted, because several other files quoted them and the retraction history
> is part of the record. Seed 45 also converted keep7's popqa/nq_open/mmlu_content from
> df=1 *pairwise ranges* (0.2726 / 0.0000 / 0.0252 — never quotable as σ) into real df=3
> estimates: **0.1959 / 0.0750 / 0.0555 pp**. See `SEED45_VERDICT.md` §3.

At the pooled estimate, a two-arm comparison at S = 3 seeds/arm can detect
**1.11 pp** (α = 0.05 two-sided, power 0.80); at the honest upper end of that σ
interval it can detect only **2.73 pp**. Every training-recipe arm A03 has left
targets an effect *smaller* than that. That is the whole decision. (At df=4 these read
1.10 pp and 3.16 pp — the point-estimate MDE is unchanged to 2 s.f.; the pessimistic
end improved 14 %. The decision is unaffected: see `SEED45_VERDICT.md` §4.1.)

> **Note on σ, per `DATAORDER_PREREG.md` §4.** No point estimate above may be quoted
> bare. Each carries its d.o.f. and its χ² interval. The interval is wide because
> d.o.f. is small: the multiplicative 95 % width of a χ²-based σ interval is
> **71.5× at df = 1**, **12.1× at df = 2**, 6.6× at df = 3, 4.8× at df = 4,
> **3.9× at df = 5**. This is
> also why the retracted 0.3231 pp "noise floor" (df = 1) was unusable, and why the
> numbers above are still weak — they are usable for *design* decisions (which is a
> one-directional use: a wide σ cannot license a small-effect experiment) and not for
> any claim of the form "the apparatus is ±Y pp".

**Seed 45 status — LANDED 2026-08-12 00:10 GMT+8. This decision is UNCHANGED.**
Primary axis **θ = −0.3622 pp, CI95 [−0.5517, −0.1838], SIG negative → NOT-CONFIRM**;
aggregate **0/3 CONFIRM → ARTIFACT**, exactly as `SEED45_PREDECLARATION.md` §3 fixed in
advance (both reachable branches retract A-2). Its two effects on this document are
folded in above and in §2: keep7 20k goes to **S = 4, df = 3** (σ-interval width
12.1× → 6.6×) and the pooled estimate to **df = 5** (4.8× → 3.9×). It also moved the
descriptive CPT increment in §1.1 from +0.0818 pp to **−0.0293 pp**, i.e. *further from*
the effect any surviving arm would need. Full analysis: `SEED45_VERDICT.md`.
~~(pre-landing text: "still training, step 215460/300000 … it has **not** landed, so
there is nothing to fold in")~~

---

## §1. What did each of the original 6 arms exist to show, and which died?

Arms enumerated from `PROPOSAL.md` lines 27-32 (the "最小 pilot" list), with
implementation status from `status/proposal_prep/A03_6ARM_DESIGN.md` §1 and §4.

| # | arm | what it existed to show | status now |
|---|---|---|---|
| 1 | intact full-depth | the upper reference: what parametric knowledge exists before damage | **ALIVE, already scored.** TriviaQA EM 0.4069, PopQA EM 0.1550, MMLU-content 0.3868, NQ-open EM 0.1025 |
| 2 | pruned / shallow (+heal) | the damaged reference: how much is left, and is it above its own floor | **ALIVE, already scored** = claim A-1 |
| 3 | pruned + CPT | **"knowledge can be written back into parameters"** — the parametric leg of the parametric-vs-external question | **DEAD as measured; UNPOWERED as designed.** See §1.1 |
| 4 | pruned + raw-text RAG | **"knowledge can be supplied externally as text"** — the external leg | **NEVER BUILT; assets absent.** See §1.2 |
| 5 | pruned + residual/KV memory | **"knowledge can be supplied as reusable residual memory, cheaper per repeat query than text"** | **NEVER BUILT; thesis already measured negative at 8B by A02.** See §1.3 |
| 6 | pruned + CPT + memory | **"the two are complementary; a joint system Pareto-dominates either"** — the interaction | **DEGENERATE.** See §1.4 |

Arms 1 and 2 are the only two that ever ran as 6-arm-study arms. Everything A03 has
published came from them (A-1) plus a set of *CPT trajectory controls* that are not
the same objects as arms 3/4/6 — the naming collision is recorded in
`claims/A03_SURVIVING_CLAIMS.md` §C-4 and `STATUS.json:arm4_peaklr_cpt.naming_warning`.

### §1.1 Arm 3 (+CPT) — the brief says it "loses its headline". Verified, with a correction.

**Verified dead in its measured instantiation.** A-2 *was* arm 3's headline. It is
retracted (`DATAORDER_VERDICT.md`, branch ARTIFACT). But the retraction is stronger
than "the headline is gone": pooling the three sampler-seed draws that exist for that
exact arm gives an effect estimate centred on zero.

The three draws are the same arm at the same dose point under three sampler seeds
(**four**, since sampler seed 45 landed 2026-08-12 — see the addendum below this table).
`DATAORDER_VERDICT.md` line 20 states this explicitly — "Arm 3, trained
pre-`ce5c298`, i.e. **sampler seed 0**" — and `scripts/_run_a03_dataorder_repl.sh`
is, by its own header, config-identical to `scripts/_run_a03_arm3_cpt.sh` apart from
`--seed` and the output path. Mechanically the pooling is sound: these are *resumed*
runs (all weights load from `step200000.pt`, `train_olmo2_arch_probe2.py:752-775`)
and the trainer contains **no dropout** (`grep -c dropout` = 0), so `--seed`'s only
material channel is `DistributedSampler(..., seed=args.seed)` at line 869 — which is
what the fix's own load-bearing comment says.

| sampler seed | triviaqa em Δ vs `A03_1B_keep7_step200k` (pp) | CI95 (pp) | sig |
|---|---:|---|---|
| 0 (the original Arm 3) | **+0.4793** | [+0.2675, +0.6910] | SIG |
| 43 | **+0.1115** | [−0.0947, +0.3177] | TIE |
| 44 | **−0.3455** | [−0.5517, −0.1393] | SIG (negative) |
| **45** *(added 2026-08-12)* | **−0.3622** | [−0.5517, −0.1838] | **SIG (negative)** |

**Aggregate at n = 4 draws: mean −0.0293 pp, s = 0.4039 pp, df = 3, t₀.₉₇₅ = 3.182 →
CI95 = [−0.672, +0.613] pp.** Against the quantity CPT is supposed to recover — the
intact-minus-pruned TriviaQA EM deficit of **31.10 pp** — that is
**−0.09 %, CI [−2.16 %, +1.97 %]** of the gap, for 20,000 steps = **5.243 B tokens**
= **90.7 GPU-h** per seed.

> ~~**Aggregate at n = 3: mean +0.0818 pp, s = 0.4132 pp, df = 2, t₀.₉₇₅ = 4.303 →
> CI95 = [−0.945, +1.108] pp** = **+0.26 %, CI [−3.04 %, +3.56 %]** of the gap~~
> **SUPERSEDED 2026-08-12** by the n=4 row above. The point estimate **crossed zero**
> (+0.0818 → −0.0293 pp) and the CI tightened ~38 %. Struck through rather than deleted
> because `POSTMORTEM.md`, `STATUS.json`, `A04/STAGE_B_DECISION.md` and
> `proposal/README.md` all quoted the n=3 figure. The scientific reading is unchanged —
> **indistinguishable from zero** — and the "mildly harmful" alternative is *not*
> established either (the CI still contains 0 comfortably). What is now excluded is
> recovery of more than ~2 % of the deficit at this budget.

> This aggregate is **post-hoc and descriptive**, not a verdict. Prereg §3.6 makes
> triviaqa em the sole primary endpoint and §4 bars σ claims from tiny d.o.f.; the
> pooled mean is reported here for the *design* question (what effect can we afford
> to look for), which is exactly the one-directional use that is legitimate. It does
> not revive, restate or amend A-2.

**The correction to the brief.** The brief says arms 3 and 6 "lose their headline".
For arm 3 that is right, but the sharper statement has two parts, because the
*measured* arm 3 is not the *designed* arm 3:

* **Measured arm 3 = "more of the same heal corpus"** (20k additional Dolmino steps
  from the healed checkpoint). Dead, per the table above.
* **Designed arm 3 = CPT on a knowledge-targeted corpus** — sub-question 1 of
  `A03_6ARM_DESIGN.md` §Arm 3 was never settled ("Dolmino vs Wikipedia subset vs
  knowledge-dense Dolmino"). This variant never had a headline to lose. It retains a
  hypothesis and **zero** results. It does not escape the decision, though: it is
  killed instead by §2's MDE bar plus one datum already in hand — the heal run itself
  spent **200,000 steps = 52.43 B tokens = 906.7 GPU-h** of Dolmino and still left a
  31.10 pp TriviaQA EM gap. "Add more generic pretraining tokens" is the intervention
  with the largest budget ever tried here, and it is the one that visibly failed.

### §1.2 Arm 4 (raw-text RAG) — never built, and the assets are not on either disk

Checked on zwfy6 (`.73`) and wzc1, read-only:

* **PopQA has no passages.** Arrow schema of the cached `akariasai___pop_qa` test
  split: `id, subj, prop, obj, subj_id, prop_id, obj_id, s_aliases, o_aliases, s_uri,
  o_uri, s_wiki_title, o_wiki_title, s_pop, o_pop, question, possible_answers`. There
  is no context/passage/supporting-document field. `A03_6ARM_DESIGN.md` §Arm 4 flagged
  this as "unverified"; it is now verified **absent**.
* **TriviaQA is cached in `rc.nocontext` only** (120 MB). The `entity_pages` /
  `search_results` configs that carry evidence documents are not on disk.
* **NQ-open carries no passages by construction.**
* **No Wikipedia passage corpus on either disk.** `data/` holds only
  `rmt_train_wikitext.jsonl`, `rmt_train_wiki_zh_10k.jsonl`,
  `wikitext_{2048_qwen3,chunks_llama3_4096,tokenized_1024}.npy` — a language-modelling
  corpus, not a retrieval index keyed to these questions.
* **No RAG harness for this model family.** `scripts/eval_olmo2_closedbook_qa.py` is
  strict closed-book by design (its own docstring line 16: "The model must answer from
  parametric memory alone (no passages/context)"). The existing retrieval scripts
  (`eval_p1_9_dense_rag.py`, `bench_iter_bm25_overhead.py`, …) are QCMem/Qwen3-8B.

So arm 4 costs a **passage-corpus download through the proxy (~13-20 GB for a
DPR-style `psgs_w100`) + a BM25/dense index over ~21 M passages + a new eval harness**
before it produces its first number. GPU cost is small (eval only); the acquisition
and engineering cost is the real price, and it is a project, not a gate.

And it would show what everyone expects: injecting the gold passage lifts EM by tens
of pp. That is §2's "guaranteed-large-but-uninformative" case.

### §1.3 Arm 5 (residual memory) — never built, and A02 already measured its thesis negative

`A03_6ARM_DESIGN.md` §Arm 5: no OLMo-2-1B QCMem read-LoRA exists on either disk; the
canonical one (`outputs/qcmem_distill_qwen_j12_r32_4k/final`) is **Qwen3-8B, layers
12..35 of 36, hidden 4096** and is inapplicable to a 9-layer, hidden-2048 model.
Building it = design a 1B recipe + adapt the write/read pipeline for OLMo-2 + train
(estimated 2-4 days of research engineering there, and the estimate predates the
architecture check).

More decisive: **the hypothesis this arm exists to test has already been measured, at
8B, with better assets, by A02** (`A02/STATUS.json:storage_readcompute_reframe_gate`):

* storage form **DEAD** — CoMem h12 is **2048× the bytes/token of raw text** (exact
  and constant across L); total-vs-RAG 632-1129×;
* read-compute form survives only **weakly** — 1.03-1.37× per query vs a
  *matched-pack text-RAG control*, needing N\* = 8-226 repeat queries to repay the
  Write, and N\* **grows** with corpus length (185-226 at 1 M), i.e. break-even gets
  worse exactly where the method is pitched;
* and **86-93 % of the apparent win is the retrieval axis**, which plain text-RAG
  gets for free.

That is A03's **kill condition #1** — "raw-text RAG 在所有质量/成本点严格支配 residual
memory" — in all but the formal firing. It is measured at a different scale and model
family, so it does not *legally* fire A03's clause. But the honest reading is that
A03 would spend days of engineering to re-derive, on a weaker 1B model, an answer its
sibling proposal already obtained on the stronger one.

### §1.4 Arm 6 (joint) — degenerate, and strictly harder than the arms it combines

Arm 6 = arm 3 + arm 5. Its scientific content is the **interaction**: is CPT+memory
better than the best of either? Two things kill it:

1. **Its CPT main effect is ~0** (§1.1: +0.08 pp, CI [−0.95, +1.11]). An interaction
   test on top of a null main effect needs *more* power than the main-effect test that
   already failed, not less — for a 2×2 design the interaction contrast has ~2× the
   variance of a main effect at the same S.
2. **It was never runnable**: blocked on arm 5, which is blocked on assets and on a
   thesis A02 measured negative.

So the brief is right that arm 6 loses its headline; the precise mechanism is that
arm 6 **degenerates into arm 5** once the CPT increment is zero, and inherits all of
arm 5's blockers plus a harder statistical target.

---

## §2. Is there a minimum arm set that still yields a publishable claim?

Cost model, measured not estimated: **2.04 s/step** on 8×H20 (median over seeds 43/44,
re-confirmed on the live seed-45 log). One 20k-step CPT run = 11.33 h wall = **90.7
GPU-h**; one 5k-step run = **22.7 GPU-h**; the 200k heal = **906.7 GPU-h**. Effective
batch = 8 × grad_accum 2 × 8 ranks = **128** (`train_olmo2_arch_probe2.py:598, 739`),
so 20k steps = 2,560,000 sequences = **5.243 B tokens** = 16.53 % of the
15,491,607-row `dolmino_now15b.npy` epoch (full epoch = 121,028 steps).

**The bar every candidate must clear.** Two-arm comparison, α = 0.05 two-sided,
power 0.80, S seeds/arm, pooled triviaqa σ̂ = **0.3666 pp (df = 5, χ² 95 % CI
[0.229, 0.899])** — updated 2026-08-12 from ~~σ̂ = 0.3620 pp (df = 4, χ² [0.217, 1.040])~~:

| S | MDE at σ̂ = 0.3666 (df 5) | MDE at σ = 0.899 (χ² 95 % upper) | training GPU-h for 2 arms × S at 20k steps | *(old: MDE @ 0.362 / 1.040)* |
|---:|---:|---:|---:|---|
| 3 | **1.11 pp** | **2.73 pp** | 544 | *1.10 / 3.16* |
| 4 | 0.87 pp | 2.13 pp | 726 | *0.86 / 2.47* |
| 5 | 0.74 pp | 1.82 pp | 907 | *0.73 / 2.10* |
| 8 | 0.55 pp | 1.35 pp | 1,451 | *0.55 / 1.57* |

Read that against what there is to detect. The intact-minus-pruned deficits — the
total amount of knowledge available to be recovered — are TriviaQA EM **31.10 pp**,
PopQA EM 11.56, NQ-open EM 7.40, MMLU-content 6.24. So at S = 3 an arm must recover
**≥3.6 %** of the TriviaQA gap to be detectable at σ̂, or **≥8.8 %** at the honest
upper end of σ. The measured CPT arm recovered **−0.09 % [−2.16 %, +1.97 %]** (n=4
draws; at n=3 it read +0.26 % [−3.04 %, +3.56 %]).

### Candidate set A — {1, 2} only: no new arms

*Claim it supports:* A-1, verbatim as in `claims/A03_SURVIVING_CLAIMS.md` §A-1 —
a pruned+healed 1B is BH-significantly above its own construct-appropriate null on
4/5 certified closed-book interfaces, while a barely-healed control of the same
topology is at/below floor on every one.
*Arms required:* none new. *GPU-h:* **0** (already scored).
*Effect-vs-spread:* this is the one comparison that is *comfortably* clear of the
apparatus instability, and for a structural reason, not a lucky one — it is a **LEVEL**
against a floor computed from the same checkpoint's own predictions, so no second
checkpoint and no second training run enters the estimator. Residuals: TriviaQA EM
+9.329 pp (CI [+8.905, +9.769]) = **25.4× the pooled σ̂** (σ̂ = 0.3666, df 5); MMLU-content +3.993
([+3.104, +4.874]) = 10.9×; NQ-open +2.299 = 6.3×; PopQA EM +1.647 = 4.5×. Even at
the χ² upper bound σ = 0.899 the smallest of these is 1.8×. *(At the superseded df = 4
σ̂ = 0.3620 these read 25.8× / 11.0× / 6.4× / 4.5×, and 1.6× at the old upper bound
1.040 — the argument is unchanged and marginally stronger.)*
*Verdict:* **sound, and already in hand — but it is not an A03 claim.** A03's thesis
is the parametric-vs-external *comparison*. Set A contains no external arm at all. It
is a floor-calibration case study, i.e. **A01's** thesis applied to generative QA —
which A03's own `GATE_FOURAXES_VERDICT.md` §2 already concedes ("this is also the
definition A01 uses in both code and prose … the two proposals agree on the number").

### Candidate set B — {1, 2, 3-targeted}: keep the parametric leg, retarget the corpus

*Claim it would support:* "knowledge-targeted CPT recovers parametric facts that
generic CPT does not."
*Arms required:* a new data pipeline (subsample Dolmino by PopQA/TriviaQA entity
overlap, or acquire a Wikipedia subset), then 2 conditions × S seeds × 20k steps.
*GPU-h:* **544 at S = 3**, plus pipeline engineering, plus (for the Wikipedia variant)
a download.
*Effect-vs-spread:* **fails the bar.** To beat generic CPT detectably it must recover
≥1.11 pp of TriviaQA EM (≥3.6 % of the gap) at σ̂, or ≥2.73 pp (≥8.8 %) at the χ²
upper bound. The prior for that is poor and is set by evidence already in hand: 200k
steps / 52.43 B tokens of the same corpus family — **10× this design's per-arm token
budget** — left a 31.10 pp gap, and the 20k top-up moved it by **−0.09 %** (n=4 draws;
+0.26 % at n=3). Proposing to
detect a ~1 pp difference between two low-dose CPT recipes on an apparatus whose own
σ interval reaches 0.90 pp is a restatement of the mistake that killed A-2: a
sub-1 pp target inside the apparatus's own instability.
*Verdict:* **rejected.**

### Candidate set C — {1, 2, 4}: the closed-book vs open-book contrast

*Claim it would support:* "a layer-pruned+healed model recovers most of its lost
closed-book accuracy when the evidence is supplied as retrieved text."
*Arms required:* passage corpus + index + new harness (§1.2). *GPU-h:* small for eval;
the cost is acquisition + 1-2 days engineering.
*Effect-vs-spread:* **passes the spread bar trivially and that is the problem.**
Gold-passage RAG on these benchmarks moves EM by tens of pp — order 100× σ̂. An
experiment whose effect is 100× the noise and whose sign nobody doubts is a **demo**,
not a test. It also cannot answer A03's actual question, because with no arm 5 there
is nothing to compare the external leg *against*: "text beats no-text" is not
"parametric vs external".
*Verdict:* **rejected — passes on power, fails on informativeness.**

### Candidate set D — {1, 2, 4, 5}: the real comparison, minus the joint arm

*Claim it would support:* A03's actual thesis — at matched evidence and matched total
cost, which interface (raw text vs reusable residual memory) recovers structurally
lost knowledge more economically.
*Arms required:* everything in set C, **plus** designing + training a 1B QCMem
read-LoRA that has never existed, plus adapting the write/read pipeline to OLMo-2.
*GPU-h:* training is the small part; **2-4 days of research engineering** is the
price, and the architecture check in §1.3 makes that estimate optimistic.
*Effect-vs-spread:* the *quality* contrast would be large enough to measure. But the
outcome is largely foreknown from A02's 8B measurement — storage form 2048× worse in
bytes, read-compute 1.03-1.37×, and 86-93 % of the apparent quality win attributable
to retrieval rather than to the memory interface. Reproducing that at 1B on a
9-layer model, where every capability is nearer its floor (pruned+healed TriviaQA EM
is 0.0959; PopQA EM 0.0394), makes the memory arm *less* likely to show an advantage,
not more.
*Verdict:* **rejected.** Highest scientific value of any candidate, and the only one
that is actually A03; but it is a multi-day build to test a hypothesis whose nearest
measurement is already negative, and A03's own kill condition #1 is the clause that
anticipates exactly this outcome.

### §2.1 Why no reduced set survives — the general form

Sort A03's remaining arms by target effect size and there are only two classes:

* **Training-recipe arms** (3, 3-targeted, 6): target effects are *marginal
  differences between pretraining recipes*, empirically ≤1 pp on this model and this
  budget. The apparatus's σ interval reaches **0.90 pp** on the primary axis at df = 5
  (1.04 pp at the superseded df = 4).
  These arms are **unresolvable at any S we can afford** — S = 8 (1,451 GPU-h for two
  arms) still only reaches **1.35 pp** at the honest end of σ (1.57 pp at df = 4).
* **Evidence-injection arms** (4, 5): target effects are tens of pp, far above the
  noise — but the interesting comparison among them (4 vs 5) requires building an
  artefact that does not exist, and its answer is already indicated negative by A02.

There is no third class. That is why "drop the dead arms and keep the rest" does not
produce a viable study: the dead arms were the ones carrying the question.

---

## §3. The honest recommendation: `ARCHIVE`

### §3.1 Why ARCHIVE rather than `PROCEED_WITH_REDUCED_SET`

No candidate set in §2 both (a) targets an effect comfortably larger than the
apparatus's measured spread and (b) answers A03's question. Sets B fails (a); set C
fails (b); set D passes both on paper but is gated on a multi-day build whose
hypothesis A02 already measured negative and whose result would land at a scale where
the memory interface is *less* favoured. Any of them would be GPU-h spent to reach a
conclusion we can already write down.

### §3.2 Why ARCHIVE rather than `NARROW_TO_A1_ONLY`

`NARROW_TO_A1_ONLY` is the tempting answer and it is subtly wrong. A-1 is a genuine,
verified, well-controlled result — but it is not a *narrowed A03*. A03 is defined
(`PROPOSAL.md` §核心问题) as the four-way comparison of where lost knowledge should
live. A-1 contains **no external-memory arm, no RAG arm, and no CPT arm**. A proposal
whose sole surviving claim is a floor-calibration case study is not a narrower version
of that proposal; it is a dead proposal with one salvageable part whose natural home is
**A01** (priority 1, `active/A01-null-calibration-methodology/`, whose thesis is
exactly cross-construct input-blind null calibration, and whose code A03's analyzer
already agrees with numerically).

Keeping A03 `active` with A-1 as its headline would also do positive harm. This
directory contains a verdict that was "written, retracted, withdrawn and replaced
within 12 hours" (`claims/A03_SURVIVING_CLAIMS.md` frontmatter), nine retracted
claims in §B, and six prose-vs-JSON discrepancies in §D. `proposal/README.md`'s own
definition of `archive/` — "保留必要 provenance，防止旧 claim 被误复活" — describes
precisely what this directory needs.

### §3.3 The kill conditions, checked against A03's own text

`PROPOSAL.md` §Kill 条件, in order:

1. *"raw-text RAG 在所有质量/成本点严格支配 residual memory"* — **effectively
   satisfied, at a different scale.** A02 measured storage 2048× worse, read-compute
   1.03-1.37×, N\* growing with L, 86-93 % of the win attributable to retrieval. Not a
   formal firing (8B/Qwen, not 1B/OLMo-2), and it is recorded here as such.
2. *"CPT 能以更低总成本恢复全部目标能力"* — **does NOT fire, and its failure is
   informative in the opposite direction:** CPT recovers ~nothing (**−0.09 %** of the
   31.10 pp gap, CI [−2.16 %, +1.97 %], n=4 draws; +0.26 % [−3.04 %, +3.56 %] at n=3)
   at 5.243 B tokens, and 52.43 B tokens of the
   same corpus left the gap open. The parametric leg does not win; it does not
   measurably move.
3. *"1B pilot 所有知识指标均处于 floor"* — **does NOT fire.** This is exactly what A-1
   refutes, and it is why A-1 is worth migrating rather than discarding.

So A03 is not being archived because a kill clause fired. It is being archived because
**the leg that was supposed to win was measured at zero, the leg that would win is
uninformative, and the comparison that would be interesting needs an artefact that does
not exist and whose nearest measurement is negative.**

### §3.4 What migrates (three assets, none discarded)

1. **A-1 → A01 as a generative-QA case study.** Four closed-book axes with
   construct-appropriate nulls (best-constant EM, longest-option split-tie for MMLU
   content, and per-arm **length-matched** `contains` nulls), each with a demonstrated
   at/below-floor control. The length-matched-null result is the strongest transferable
   piece: the naive null would have credited pruned+healed × NQ-open × `contains` with
   81.7 % versus the correct 43.3 % — a 1.9× inflation from the wrong null on one cell
   (`GATE_FOURAXES_VERDICT.md` §4). Also carry the B-4 erratum
   (`/reported` ≠ recovery-of-headroom; TriviaQA 97.3 % vs 9.4 %) — it is a
   general trap in exactly A01's family.
2. **The negative CPT result + the phase-locking diagnostic → a methods note.** "Three
   arms agreed at r = 0.99982 because they shared one minibatch sequence; the +0.48 pp
   effect they agreed on vanished (mean +0.08 pp, CI [−0.95, +1.11]) once the sampler
   seed varied" is a reusable, self-contained cautionary result about mistaking a
   deterministic data path for replication. It belongs wherever A01/A04's
   methodology is written up, and it is worth more than the claim it killed.
3. **The σ_run draws → A04** (§4). They are the repo's only run-to-run measurements
   and A04's K2 arithmetic consumes them directly.

### §3.5 The physical move is BLOCKED — do not `git mv` this directory yet

Ran the project's mandatory pre-archive reference check
(`CLAUDE.md` §研究方向命名与晋升规则):

```
grep -rl "A03-parametric-vs-external-memory" --include='*.py' --include='*.sh' \
     --include='*.md' --include='*.json' --include='*.tex' .
```

Two **code** dependencies would break on a move:

* `proposal/active/A04-recovery-certification/code/pilot_one_stage_a_sd_run.py:81`
  builds a path through `"A03-parametric-vs-external-memory" / "code"` to import
  A03's canonical `load_cb` / `load_mmlu` loaders. Those loaders carry the 8/8-shard,
  exact-item-count, duplicate-id and NaN assertions that every A04 Stage-A/Stage-B
  number rests on. Moving A03 silently breaks A04's driver.
* `proposal/active/A01-null-calibration-methodology/code/a01_audit_response_recompute.py:5,310`
  cites `A03/evidence/TCODEX_AUDIT_20260810.md` as the audit it responds to.

Plus documentation references from A01 (7 files), A02 (1) and A04 (6).

**Therefore:** the disposition is `ARCHIVE`, recorded in `STATUS.json`, with
`POSTMORTEM.md` written in place, and the directory left at its current path until the
two hardcoded paths are repointed (the clean fix is to lift the shared loaders into
`proposal/shared/`, which is what `shared/` exists for). This is deliberate: the
project's own rule says a directory referenced by live code is **禁删**, and half-moving
it is worse than not moving it.

### §3.6 What to do when seed 45 lands — **✅ DONE 2026-08-12 00:10**

Mechanical, and pre-committed by `SEED45_PREDECLARATION.md` §"Pre-declared analysis
rules". Nothing here reopens this decision. All four items executed; see
`SEED45_VERDICT.md`.

1. ✅ Classified by prereg §3.3 (CONFIRM ⟺ CI excludes 0 **and** θ > 0 **and**
   θ ∈ [+0.20, +0.80] pp). Result **θ = −0.3622 pp, CI95 [−0.5517, −0.1838] →
   NOT-CONFIRM** (fails θ > 0 and the band) → **ARTIFACT stands (0/3)**.
   **A-2 remains retracted** — and both branches would have retracted it.
2. ✅ keep7 20k CPT family recomputed at n = 4 draws (sampler seeds 0/43/44/45),
   **df = 3, s = 0.4039 pp, χ² [0.229, 1.506]**, width 12.1× → 6.6× as expected.
   Pooled: **0.3666 pp, df = 5, χ² [0.229, 0.899]**, width 4.8× → 3.9×.
   Bonus: keep7's popqa/nq_open/mmlu_content went from unusable df=1 pairwise ranges to
   real df=3 σ estimates (0.1959 / 0.0750 / 0.0555 pp).
3. ✅ Handed to A04 — `A04/STAGE_B_DECISION.md` addendum item 4 and
   `A04/STATUS.json:next_gate[4]` restated at df = 5 (MDE 1.11 pp at S=3 and σ̂;
   2.73 pp at the χ² upper bound). **A04's own pre-registered K2 arithmetic is
   numerically untouched** — it uses the *keep12* family's σ at df=2, and seed 45 is a
   keep7 draw — so popqa's fire-at-the-pessimistic-end still stands and the "more
   keep12 seeds before Pilot Two" recommendation is unchanged (see `SEED45_VERDICT.md` §4.3).
4. ✅ No seed 46 launched. Prereg: seed 45 is the last run under it.

---

## §4. Cross-proposal consequence for A04's 135 GPU-h

### §4.1 First, a fact that changes the question: the 135 GPU-h is already spent

`A04/STAGE_B_DECISION.md` asks whether to spend it. **It was spent, today**, under
standing autonomy — the doc's own correction banner records Path A launching at
05:53 GMT+8, and I verified completion on zwfy6 (read-only):

* all three pre-registered seeds trained: `outputs/…stageB…seed{101,102,103}/`
  each with `step2500.pt`, `step5000.pt`, `final.pt`;
* all three evaluated on all four axes, 17,944-row merged TriviaQA per-example files
  present for each;
* `A04/evidence/stageB_S3_verdict.json` exists (md5
  `7145d569f46ec0fa10dd56368071adf2`, written 14:53 GMT+8), verdict
  `STAGE_A_DOES_NOT_FIRE`, `n_decision_axes_exceeding: 0`, plus two pairwise
  files (`stageB_pair_101_102.json`, `stageB_pair_101_103.json`).

So the live A04 question is no longer "spend 135 GPU-h?" but **"spend the next
tranche — Pilot Two at 1,077-4,309 GPU-h, which `A04/STATUS.json:next_gate[4]`
correctly marks as requiring explicit user approval?"** I answer that question's
*dependency*, not the question.

### §4.2 Does my A03 decision strengthen or weaken the case? **Weaken — but not for the reason STAGE_B_DECISION.md gave.**

**Where the retracted argument was wrong, and stays wrong.** That doc's FOR-case leant
on "< 0.5 pp, i.e. inside the noise floor we just measured". Its own banner retracts
that (no floor was measured; the 0.3231 pp figure is df = 1 with a χ² σ interval of
≈[0.14, 10.3] pp). Independently, its prediction that keep12 spread would be *smaller*
than keep7's is **refuted**, and now with keep7 at df = 3 rather than df = 1 pairwise
ranges on three of the four axes:

| axis | keep7 20k, S = 4 (sampler seeds 0/43/44/**45**) | keep12 5k, S = 3 (seeds 101/102/103) | direction |
|---|---:|---:|---|
| triviaqa em | s = **0.4039** (df 3, χ² [0.229, 1.506]) | s = 0.3023 (df 2, χ² [0.157, 1.900]) | keep12 smaller |
| popqa em | s = **0.1959** (df 3, χ² [0.111, 0.730]) | s = 0.3328 (df 2, χ² [0.173, 2.092]) | **keep12 LARGER (1.7×)** |
| mmlu content_norm | s = **0.0555** (df 3, χ² [0.031, 0.207]) | s = 0.0783 (df 2, χ² [0.041, 0.492]) | **keep12 LARGER (1.4×)** |
| nq_open em | s = **0.0750** (df 3, χ² [0.042, 0.280]) | s = 0.2091 (df 2, χ² [0.109, 1.314]) | **keep12 LARGER (2.8×)** |

**Spread is not monotone in damage** — three of four axes go the wrong way, and the
one that goes the "right" way has overlapping σ intervals. Any A04 design that budgets
seeds by assuming less damage ⇒ less variance is mis-budgeted.

> **UPDATED 2026-08-12 (seed 45).** The keep7 row for triviaqa was
> ~~s = 0.4132 (df 2, χ² [0.215, 2.597])~~, and popqa/mmlu/nq_open were
> ~~df = 1 *pairwise ranges* 0.2726 / 0.0252 / 0.0000~~ that this section explicitly
> warned "must not be quoted as σ". Seed 45 replaced all four with df = 3 σ estimates,
> so the non-monotonicity claim now rests on real σ estimates on **both** sides instead
> of σ-vs-range. Note nq_open's old pairwise 0.0000 was a coincidence of two draws
> scoring identically (105/3610 each); its true s is 0.0750 pp. The multiplier on
> mmlu_content changes from "3.1×" (range-based) to **1.4×** (σ-based) — the direction
> is the same, the magnitude was overstated by the df=1 artefact.

**The real dependency, stated plainly.** A04 certifies that a structurally damaged
model has *recovered*. Its value is proportional to the size of the recovery effect
its rule adjudicates. A03 has now measured the **CPT recovery increment** on the shared
apparatus at **−0.0293 pp, CI95 [−0.672, +0.613] pp — −0.09 % of the 31.10 pp deficit,
a CI that contains zero** (n = 4 draws; ~~+0.0818 pp, CI [−0.945, +1.108], 0.26 %~~ at
n = 3) — at 5.243 B tokens/run, and the 200k-step heal shows 52.43 B
tokens of the same corpus does not close the gap either. If the increment a
certification rule is meant to adjudicate is statistically indistinguishable from zero
at the budgets we can afford, then the rule's *discriminative* content is untested:
NI and PLATEAU can only disagree informatively about a recovery trajectory that moves.
**That weakens the case for the next tranche** — and seed 45 weakened it slightly
further, by moving the point estimate from marginally positive to marginally negative.

**Two things that fairly push the other way, and must be recorded with the above:**

1. **A04's Pilot Zero is untouched by A03's retraction**, for the same structural
   reason A-1 is (`claims/A03_SURVIVING_CLAIMS.md` §A-1, "why A-1 does NOT inherit the
   trajectory oscillation defect"): it is a **level** at one checkpoint against intact,
   with no second-checkpoint and no second-training-run term in the estimator.
   `DATAORDER_VERDICT.md` §"Does NOT mean" #1 says this explicitly. A03's ARTIFACT
   verdict is **not** evidence against A04's finding, and must not be cited as such.
2. **A04's K2 arithmetic is now better supported than when Stage B launched** — S = 3
   instead of a pairwise range, on the correct arm (keep12) and the correct budget
   (5k steps). At the point estimates, `bound₃ = 2.920·s/√3` sits well under Δ on all
   three decision axes: triviaqa 0.510 vs 4.043 (**7.9× margin**), popqa 0.561 vs
   1.321 (2.4×), mmlu_content 0.132 vs 1.024 (7.8×). **The caveat A04 should carry:**
   evaluated at the χ² 95 % *upper* bound of each σ (df = 2), triviaqa (3.203 vs
   4.043) and mmlu_content (0.829 vs 1.024) still do not fire, but **popqa would
   (3.527 vs 1.321)** and demoted nq_open would (2.215 vs 0.970). So the honest
   statement is "K2 does not fire at the point estimate, and one decision axis would
   fire at the pessimistic end of a df = 2 σ interval." That argues for **more seeds
   before Pilot Two**, not for Pilot Two.
   **★ Seed 45 does NOT close this** (checked 2026-08-12): K2's pre-registered estimator
   is the *keep12* family's own `sd_run` at df = 2, and seed 45 is a *keep7* draw, so
   every number in this item is numerically unchanged — popqa still fires at
   3.526 vs 1.321. Substituting the pooled df = 5 σ would make nothing fire
   (popqa 0.740 vs 1.321), but that is a **change of estimator** after seeing which
   answer each gives, and it is not licensed. The way to close popqa's pessimistic-end
   trigger is d.o.f. on the **keep12** family, which only keep12 seeds buy. See
   `SEED45_VERDICT.md` §4.3.

**Net, and it is A04's call not mine:** my A03 decision **weakens** the case for
committing the next A04 tranche. Before it is committed, A04 should have to state — in
its own prereg, pre-data — **what recovery magnitude its certification is meant to
adjudicate, and show that magnitude exceeds the MDE its chosen S implies** (§2's table:
**1.11 pp at S = 3 and σ̂ = 0.3666 (df 5); 2.73 pp at the χ² upper bound** —
~~1.10 / 3.16 at df = 4~~). That is the discipline
A03 lacked, and it is why A03 is being archived.

### §4.3 A free correction for A04: Path B is already fully bought

`STAGE_B_DECISION.md` Path B proposed spending ~135 GPU-h on "two extra keep7 seeds at
20,000 steps to nail down the noise floor at n = 4 (df = 3)". **Do not buy that.** A
keep7 20k family at **n = 4, df = 3** now exists at zero marginal cost — the original
Arm 3 *is* the sampler-seed-0 draw of the identical config (§1.1: same checkpoint,
same schedule, config-identical driver, no dropout, `--seed`'s only channel is the
sampler), and `DATAORDER_VERDICT.md` line 20 already labels it that way; seeds 43/44
were already on disk, and **seed 45 landed 2026-08-11 23:29 on GPU-h already committed**.
Its σ is **0.4039 pp, df = 3, χ² [0.229, 1.506]** (~~0.4132, df = 2, χ² [0.215, 2.597]~~
before seed 45). Path B's stated deliverable — *exactly* n = 4 / df = 3 — therefore cost
**0 additional GPU-h**, not 135, and it is **delivered**, not pending.

---

## §5. Provenance

Every number recomputed this session; no GPU.

| quantity | source |
|---|---|
| A-1 residuals, nulls, BH p, intact-vs-pruned deficits | `evidence/a03_1b_floor_nulls_4axes.json` (md5 `5e443b424bfde44397bc497a39062504`, wzc1) |
| Arm 3 step220000 Δ = +0.4792688363798484, CI, n = 17944 | `evidence/arm3_arm4_arm6_cpt_trajectory_paired_full.json` (md5 `28584639f120aaff07bd1a52120f983e`) |
| **seed 45 Δ = −0.36223807400802494, CI, n = 17944** *(added 2026-08-12)* | `evidence/a03_cpt_trajectory_paired_full_with_seed45.json` (md5 `7b5cc4c7040561d9cdb8bd9d4916ad83`) |
| **σ_run at n=3 draws/family, pooled df = 5, MDE table, K2 sensitivity** *(added 2026-08-12)* | `evidence/a03_sigma_run_n3.json` (md5 `5fb6cd4c3d693831e50d0817bda93ab8`), produced by `code/recompute_sigma_run_n3.py` |
| **seed 45 shard integrity (8/8 × 4 axes × arm+baseline, 0 dup, 0 nan)** *(added 2026-08-12)* | `evidence/a03_seed45_integrity.json` (md5 `df1535f0bab24f4ebdeade806935b9fb`) |
| seeds 43/44 Δ and absolute means | `evidence/pilot_one_stage_a_verdict.json` (md5 `4ced4582cce6772a797a7f41e94e2a7a`) |
| keep12 S = 3 σ_run, bound₃, Δ per axis | `A04/evidence/stageB_S3_verdict.json` (md5 `7145d569f46ec0fa10dd56368071adf2`, zwfy6) |
| keep12 S = 2 pairwise | `A04/evidence/pilot_one_stageB_S2_verdict.json` |
| eff_bs 128, token counts, epoch fraction 16.53 % | `scripts/train_olmo2_arch_probe2.py:598,739` + row count 15,491,607 |
| 2.04 s/step, seed-45 progress `step 215460/300000` | live read-only tail of `logs/a03_dataorder_seed45.log` on `.73`, 20:54 GMT+8 |
| Arm 3 ≡ dataorder driver up to `--seed` | `scripts/_run_a03_arm3_cpt.sh` vs `scripts/_run_a03_dataorder_repl.sh` header; `train_olmo2_arch_probe2.py:863-869` (sampler), `:752-775` (resume), `grep -c dropout` = 0 |
| PopQA has no passage field | Arrow schema of `akariasai___pop_qa/…/pop_qa-test.arrow` read on `.73` |
| TriviaQA cached as `rc.nocontext` only, 120 MB | `du -sh` on `.73` HF cache |
| A02 storage 2048× / read 1.03-1.37× / 86-93 % retrieval | `A02/STATUS.json:storage_readcompute_reframe_gate` |
| σ CI widths (71.5× / 12.1× / 6.6× / 4.8×), MDE table | χ²(df) and t(df) quantiles, scipy |

**Statistical conventions used.** σ intervals are χ²-based:
`s·√(df/χ²_{0.975,df})` to `s·√(df/χ²_{0.025,df})`. `bound₃ = 2.920·s/√3` is A04's
pre-registered one-sided t(df = 2) form, reproduced not redefined. MDE is the standard
two-sample form `(t_{0.975,2S−2} + t_{0.80,2S−2})·σ·√(2/S)`. Pooled σ is
`√((df₁s₁² + df₂s₂²)/(df₁+df₂))`.
