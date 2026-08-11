---
scope: A03 — postmortem. Why the parametric-vs-external-memory direction is archived.
date: 2026-08-11 21:10 GMT+8 (σ / CPT-increment numbers updated 2026-08-12 00:10 after sampler seed 45 landed; superseded values struck through in place, not deleted)
decision_doc: ARM_SET_DECISION.md (the no-GPU gate that decided this)
seed45: SEED45_VERDICT.md — the last pre-registered run. NOT-CONFIRM → aggregate 0/3 → ARTIFACT stands, A-2 stays retracted, and the archiving argument is reinforced on both sides.
disposition: ARCHIVE. **The move HAS happened** (commit `d09d220`, 2026-08-11): the two
       blocking code paths in §5 were repointed by lifting the loaders to
       `proposal/shared/code/canonical_eval_loaders.py` (`3949f14`), and the directory now
       lives at `proposal/archive/A03-parametric-vs-external-memory/`. §5's "do NOT git mv"
       warning is retained below as a record of the pre-move state.
       ⚠️ zwfy6's `proposal/` is a hand-copied tree, not a git checkout, so on `.73`/`.82`/`.104`
       this directory is still at `proposal/active/...` — same bytes, different path.
authoritative_claims: claims/A03_SURVIVING_CLAIMS.md remains the single authority on
       what may and may not be said. This file explains the death; that file bounds
       the claims. Neither supersedes the other.
---

# A03 postmortem — where lost knowledge should live, and why we stopped asking

## §1. What A03 set out to do

After structural compression (layer pruning), a model loses parametric knowledge.
A03 asked where that knowledge should be put back: into the **parameters** (continued
pretraining), into the **prompt** (raw-text retrieval), into a **reusable residual /
KV memory**, or into some **joint** arrangement — and how the answer depends on whether
the knowledge is old, new, updated, or multi-evidence. Six arms, four knowledge axes,
a mandatory cost ledger, and a set of controls (each interface above its own null;
matched evidence; same tokenizer/family; closed- and open-book kept separate).

The design was good. That is worth saying plainly, because it is not why it died.

## §2. What actually happened, in order

1. **Gate 1 passed, and passed well.** A pruned+healed 1B (`keep7+fresh2`, 9 layers
   from 16, healed 200k Dolmino steps) is BH-significantly above its own
   construct-appropriate input-blind null on 4 of 5 certified closed-book interfaces,
   while a barely-healed step-500 control of the same topology is at or below floor on
   every one of them. The measurement instrument was certified. This is claim **A-1**
   and it still stands.
2. **The instrument then got used on the wrong kind of quantity.** A-1 is a *level*
   against a floor. The next thing measured was a *difference between checkpoints* — a
   20k-step CPT top-up, scored at step220000, +0.4793 pp SIG on TriviaQA EM.
3. **A second arm "replicated" it** at a different learning rate: +0.5016 pp, within
   0.02 pp. That agreement was read as validation.
4. **The agreement was engineered.** The `seed=args.seed` fix on
   `DistributedSampler` existed on wzc1 but had never been copied to zwfy6, so every
   arm ran with the library default sampler seed 0 and consumed a **byte-identical
   minibatch sequence**. Training-loss Pearson r between the two arms: **0.99982**.
   They were not two draws from the population of training runs; they were one data
   path observed twice.
5. **A data-order replication was pre-registered before any number existed**
   (`DATAORDER_PREREG.md`, commit `a25d780`, 19:20:02; earliest replication checkpoint
   on disk 19:47:23 — 27 minutes later) and run at **three** new sampler seeds. Results:
   **+0.1115 (TIE)**, **−0.3455 (SIG negative)** and **−0.3622 (SIG negative — seed 45,
   landed 2026-08-11 23:29)**, against a pre-registered band of
   [+0.20, +0.80] pp. Branch **ARTIFACT** fired: **0/3** landed seeds CONFIRM. A-2
   retracted. Both reachable branches would have retracted it, as
   `SEED45_PREDECLARATION.md` established before seed 45 ran.
6. **Pooling the four draws of that same arm gives an effect centred on zero:**
   mean **−0.0293 pp**, s = 0.4039, df = 3, **CI95 [−0.672, +0.613] pp**. Against the
   31.10 pp intact-minus-pruned TriviaQA EM deficit that is **−0.09 %,
   CI [−2.16 %, +1.97 %]** — for 5.243 B tokens per run.
   *(~~n = 3 draws: mean +0.0818 pp, s = 0.4132, df = 2, CI95 [−0.945, +1.108] =
   +0.26 %, CI [−3.04 %, +3.56 %]~~ — **SUPERSEDED 2026-08-12** by seed 45. The point
   estimate crossed zero and the CI tightened ~38 %; the reading is unchanged —
   indistinguishable from zero — and the "mildly harmful" alternative is *not*
   established either.)*
7. **The no-GPU arm-set gate then ran** (`ARM_SET_DECISION.md`) and found no reduced
   arm set that both targets an effect larger than the apparatus's own spread and
   answers A03's question.

## §3. The actual cause of death

Not the retraction. The retraction was the *symptom*. Three things had to be true
together, and they were:

**(a) The parametric leg was measured at zero.** CPT — the intervention A03 was built
to champion — recovers **−0.09 % of the gap [−2.16 %, +1.97 %]** at 5.243 B tokens
(4 sampler-seed draws; ~~+0.26 % [−3.04 %, +3.56 %]~~ at 3 draws). And the
one budget larger than that which was actually spent (the 200k-step heal, 52.43 B
tokens, 906.7 GPU-h) still left a 31.10 pp gap. The largest dose ever tried is the one
that visibly failed. There is no reason to think a 10×-smaller dose of a slightly
different corpus closes it.

**(b) The external leg was never built, and its thesis is already measured negative
elsewhere.** The RAG arm needs a passage corpus that is on neither disk — verified:
cached PopQA has no context field at all (17-column schema, none of them a passage),
TriviaQA is cached `rc.nocontext`-only, and `data/` holds no Wikipedia passage index.
The residual-memory arm needs a 1B QCMem read-LoRA that has never existed (the
canonical one is Qwen3-8B, layers 12..35 of 36, hidden 4096 — inapplicable to a
9-layer hidden-2048 model). And **A02 already measured the memory-vs-text question at
8B**: storage form dead at 2048× the bytes/token of raw text, read-compute form
1.03-1.37× per query against a matched-pack text-RAG control, N\* growing with corpus
length, and 86-93 % of the apparent win attributable to retrieval — which text-RAG
gets for free. That is A03's own kill condition #1 in all but the formal firing.

**(c) The remaining measurable effects are smaller than the apparatus's own spread.**
Pooled run-to-run σ on the primary axis is **0.3666 pp (df = 5, χ² 95 % CI
[0.229, 0.899])** — updated 2026-08-12 from ~~0.3620 pp (df = 4, χ² [0.217, 1.040])~~
after sampler seed 45 landed. A two-arm comparison at S = 3 seeds detects **1.11 pp** at
that point estimate and only **2.73 pp** at the interval's upper end. Every
training-recipe arm A03 has left targets ≤1 pp. **They are unresolvable at any S we can
afford** — S = 8, at 1,451 GPU-h for two arms, still only reaches **1.35 pp** at the
honest end of σ. *(At df = 4 these read 1.10 / 3.16 / 1.57 pp. Seed 45 moved the
point-estimate MDE by 0.01 pp and improved the pessimistic end by 14 % — while
simultaneously moving the effect being chased from +0.08 pp to −0.03 pp. The archiving
argument is reinforced from both sides, not weakened; see `SEED45_VERDICT.md` §4.1.)*

Put together: **the leg that was supposed to win was measured at zero; the leg that
would win is uninformative; the comparison that would be interesting needs an artefact
that does not exist and whose nearest measurement is already negative.**

## §4. Lessons that generalise (the part worth keeping)

1. **Agreement between two runs is not replication if they share a data path.** Two
   runs at r = 0.99982 training-loss correlation are one observation. Before quoting
   cross-arm agreement as evidence, check that the thing you varied is the only thing
   that varied — and check it *on the disk the runs actually ran on*. The fix existed;
   it was on the other disk. **Two disks, and a fix that exists on one of them, is the
   same as no fix.**
2. **Levels and differences have different failure modes.** A-1 (a level vs a floor
   computed from the same checkpoint's own predictions) survived everything that killed
   A-2 (a difference between checkpoints), and it survived *structurally*, not luckily:
   no second checkpoint and no second training run enters its estimator. When a
   proposal has both kinds of claim, expect them to live or die separately, and say
   which kind each one is.
3. **A bootstrap CI over items does not bound run-to-run variance.** A-2's CI
   half-width was 0.21 pp while the checkpoint-to-checkpoint swing on the same axis was
   up to 2.66 pp — 10× larger. Tight CIs on the wrong variance component are false
   confidence. If you are comparing training runs, the relevant σ needs ≥3 runs.
4. **Do not quote σ without its d.o.f. and its interval.** The multiplicative 95 %
   width of a χ²-based σ interval is **71.5× at df = 1**, 12.1× at df = 2, 6.6× at
   df = 3, 4.8× at df = 4, 3.9× at df = 5. The retracted "~0.3 pp noise floor" was a
   df = 1 point
   estimate with a σ interval of ≈[0.14, 10.3] pp — it bounded nothing. The fix is not
   a larger n; it is always reporting the interval. **And the corollary, visible in this
   proposal's own numbers:** going from df = 4 to df = 5 moved the σ *point estimate* by
   +1.3 % (0.3620 → 0.3666 pp) while shrinking its upper bound by 14 % (1.040 → 0.899).
   One more run buys you *knowledge of σ*, not a different σ — which is exactly why a
   bare point estimate was never the useful object.
5. **Pre-register the target effect size, not just the analysis.** A03 pre-registered
   its replication band, protocol and branches — genuinely, before data, verifiably by
   commit timestamps — and that pre-registration is what let the retraction be clean
   and fast. What it never pre-registered was **"what effect size is this apparatus
   able to resolve, and is the effect we are hunting bigger than that?"** Had it, the
   +0.48 pp hunt would never have started: it was inside the instability all along.
6. **A "seed" that does not reach the sampler is not a seed.** `set_seed()` /
   `torch.manual_seed()` cannot reach `DistributedSampler`, which builds its own
   generator from `self.seed + self.epoch` with `self.seed` defaulting to 0. Every
   "seed variance" arm in this repo before the fix was fresh-block-init variance only.
7. **A sampler-seed change at partial-epoch scale changes *which data* is seen, not
   just its order.** 20,000 steps × eff-batch 128 = 2,560,000 sequences against
   15,491,607 rows = **16.53 % of one epoch**. The seeds trained on largely different
   data. Say "sampler-seed / data-subset variation"; "data order" understates it and
   makes the original result sound more general than it is.

## §5. What survives, and where it goes

| asset | disposition |
|---|---|
| **A-1** (pruned+healed 1B above its own construct-appropriate null on 4/5 certified closed-book interfaces; barely-healed control at/below floor) | **Migrate to A01** as a generative-QA case study of null calibration. Carry the **length-matched `contains` null** result with it — the naive null inflates one cell 1.9× (81.7 % vs the correct 43.3 %) — and carry the **B-4 erratum** (`(reported−null)/reported` ≠ recovery of headroom: TriviaQA 97.3 % vs 9.4 %). |
| **The phase-locking diagnostic + the null CPT result** | **Migrate as a methods note.** "Three arms agreed at r = 0.99982 because they shared one minibatch sequence; the effect they agreed on vanished once the sampler seed varied" is a reusable cautionary result, and it is worth more than the claim it killed. |
| **The σ_run draws** (keep7 20k: sampler seeds 0/43/44/**45**, s = **0.4039 pp df = 3**, χ² [0.229, 1.506]; pooled with keep12 → **0.3666 pp df = 5**, χ² [0.229, 0.899]. ~~0.4132 df = 2~~ / ~~0.3620 df = 4~~ superseded 2026-08-12) | **Migrate to A04.** They are the repo's only run-to-run measurements and A04's K2 arithmetic consumes them directly. See `ARM_SET_DECISION.md` §4.3: A04's proposed "Path B" deliverable (n = 4, df = 3 at keep7) is **delivered** at 0 additional GPU-h. ⚠️ But note (`SEED45_VERDICT.md` §4.3) that K2's *pre-registered* estimator is the **keep12** family's own σ at df = 2, which seed 45 does not touch — popqa still fires at that σ's χ² upper bound, so "more keep12 seeds before Pilot Two" stands. |
| **The four certified axes + `code/analyze_1b_knowledge_floor.py` + `code/recompute_cpt_trajectory_paired.py`** | **Keep and reuse.** The loaders carry the 8/8-shard, exact-item-count, duplicate-id and NaN assertions that A04's numbers depend on. They should be lifted into `proposal/shared/` — see below. |
| Everything in `claims/A03_SURVIVING_CLAIMS.md` §B (B-1 … B-9) | **Dead. Do not resurrect.** That file is the authority; this one does not re-litigate it. |

**⚠️ HISTORICAL — the block below described the state on 2026-08-11 21:10, before the
move. It has since been resolved: the loaders were lifted to
`proposal/shared/code/canonical_eval_loaders.py` (commit `3949f14`, md5
`2ccce419839b17f0d8f29233b4b569ff` on **both** disks), both importers were repointed,
and the directory was moved to `proposal/archive/` in commit `d09d220`. Kept verbatim
because the reference check itself is the reusable part.**

~~**The directory has NOT been moved to `proposal/archive/`, deliberately.**~~ The
project's archive rule requires a reference check before moving, and it found two
**code** dependencies that a move would silently break:

* `A04/code/pilot_one_stage_a_sd_run.py:81` builds a path through
  `"A03-parametric-vs-external-memory" / "code"` to import A03's canonical
  `load_cb` / `load_mmlu`;
* `A01/code/a01_audit_response_recompute.py:5,310` cites
  `A03/evidence/TCODEX_AUDIT_20260810.md`.

Plus documentation references from A01 (7 files), A02 (1) and A04 (6). The clean fix
was to lift the shared loaders into `proposal/shared/` and repoint both importers; the
physical move followed that. A half-moved directory is worse than an unmoved one.

## §6. What would legitimately revive this direction

Not a cheaper version of the same experiment. Three things would have to change:

1. **A model scale where the effects are large enough to see.** At 1B with 9 layers,
   pruned+healed TriviaQA EM is 0.0959 and PopQA EM 0.0394 — near the floor, so every
   recovery increment is compressed into a range narrower than the apparatus's spread.
   A 7B/8B pruned arm has room for a recovery effect that clears 1-3 pp.
2. **The external arms actually existing.** A passage corpus + index for these
   benchmarks, and a residual-memory adapter for the *same* model family as the
   damaged arm — not a Qwen3-8B adapter compared against an OLMo-2-1B baseline, which
   is not the same-model comparison the proposal requires.
3. **A pre-registered target effect size with a power calculation attached**, of the
   form "we are hunting an X pp effect; our σ is Y with df = Z; at S seeds our MDE is
   W < X". A03 never wrote that sentence. It is the sentence that would have prevented
   all of this.

Condition 2's cost is dominated by asset acquisition and engineering, not GPU. If
someone builds those assets for another reason, the parametric-vs-external question
becomes cheap to ask again — and it is still, for what it is worth, a good question.
