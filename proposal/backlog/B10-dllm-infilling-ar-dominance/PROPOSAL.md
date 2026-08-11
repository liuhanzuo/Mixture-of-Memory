# B10 — dLLM infilling: does the matched AR control overturn the diffusion infilling case?

**Status:** `backlog`. **Not** active. No GPU is authorised by this document.
**Created:** 2026-08-11, by promoting a direction that until now existed only
inside `proposal/archive/revival-slate/SLATE.md` (rank 3 of 5) — which violated
the rule that `proposal/` is the single index and `archive/` is for dead or
superseded work.

**This promotion deliberately does NOT carry SLATE's claims forward as stated.**
Every number was re-derived from disk (`NUMBER_AUDIT.md`). The arithmetic held
perfectly; two of the three headline *claims* did not. What follows is the
surviving, defensible version — which is considerably narrower than SLATE's.

---

## 1. What was claimed, and what is left after audit

SLATE proposed two coupled results: **(a)** a matched-lineage AR model beats
masked diffusion on diffusion's own turf (single-line code infilling) while
feeding 20.6–24.4× fewer tokens; **(b)** DreamOn's advertised
`mask_expansion` / `delete_eos_token` kwargs are inert, so its headline
length-elasticity was never exercised.

Audit outcome:

| claim | status after re-derivation |
|---|---|
| the six pass@1 values, the four cost values, the suffix-gain values | ✅ **EXACT** to 10 d.p., reproduced twice from raw per-task records |
| **(a)** as a *ranking* claim ("AR is ABOVE the best diffusion arm") | ❌ **NOT SUPPORTED.** Paired McNemar `p = 0.635`; margin = 5 tasks / 1033; CI [−0.0116, +0.0213] straddles zero. On the gold-feasible subset the **sign flips** (diffusion 0.9337 vs AR 0.9324) |
| **(a)** as a *cost* claim ("fewer tokens on both units") | ❌ **FALSE** under `attended_context_sum`, where Dream-FIM is **0.88× = cheaper than AR**. True only for the two DreamOn arms under `tokens_fed` |
| **(b)** kwargs are silently swallowed | ✅ mechanically TRUE (verified statically, weights on disk after all) |
| **(b)** ⇒ "length-elasticity not exercised" | ❌ **REFUTED.** Expansion is token-driven, not flag-gated, and live by default: output length changed on **84.3 %** of `dreamon_fim` tasks. Also, the README never advertises those kwargs |
| "DreamOn ships no matched-lineage AR native-FIM control" — the stated motivation | ❌ **FALSE.** DreamOn's own Table 1 (ICLR 2026 camera-ready) reports **Qwen2.5-Coder-7B = 92.6** on this exact split |
| "matched lineage for Dream-Coder-v0-**Base**-7B" | ❌ **wrong model.** The run used Dream-Coder-v0-**Instruct**-7B; the AR arm is a *base* model. Post-training differs across arms |

**Claim (b) is dead. Claim (a)'s headline is dead. The premise that no AR control
exists is dead.** What remains is below, and it is a methodology contribution,
not a conclusion-reversal.

---

## 2. The surviving scientific question

> On a task surface where diffusion is claimed to be structurally advantaged, how
> much of the reported comparison is decided by **choices that are not the model**
> — the grading axis and its unmeasured feasibility ceiling, the cost unit, and
> the oracle-length handout — rather than by the model class?

This is worth asking because on our own data each of those three choices is
individually decisive, and they point in **different directions**:

1. **Grading axis / ceiling.** Splicing the benchmark's *own gold middle* back in
   scores only **0.8025** on the axis we graded (`--which plus`), vs 0.9894 on
   base. So ~19.75 % of items are unpassable by construction. Restricting to the
   829 feasible items moves `qwen_fim` 0.7638 → **0.9324** and `dreamon_oracle`
   0.7590 → **0.9337** — i.e. **the ranking inverts**, and both land near
   DreamOn's published 92.6 / 92.1. A 16–22 pp apparent "reproduction failure"
   was mostly an unmeasured ceiling.
2. **Cost unit.** Under `tokens_fed` AR is 8.5–24.4× cheaper. Under
   `attended_context_sum` Dream-FIM is **cheaper than AR** (0.88×). For every
   diffusion arm the two units are *identical by construction* (each step
   re-feeds the canvas) whereas for AR they differ ~10× because of KV caching. So
   the unit choice *is* the winner choice, and neither unit is neutral.
3. **Oracle handout.** `dreamon_oracle` (given true middle length) beats
   `dreamon_fim` (self-predicted) by +5.7 pp, `p = 4.1e-14`. Whether AR "wins"
   depends entirely on which diffusion configuration is the comparator.

Each factor is worth more than the ranking it produces. The defensible
contribution is **the sensitivity structure**, with the ranking reported as
*conditional on disclosed choices* — never as a bare headline.

### What is genuinely robust in the existing data

- Suffix visibility is worth **+0.2314** (AR) and **+0.2991** (diffusion), both
  overwhelmingly significant. The *difference* of gains is +0.0678 with paired CI
  [+0.0407, +0.0949] — excludes zero, so "comparable" is too strong, but the
  qualitative point stands: **bidirectional context is an affordance of the FIM
  task framing, available to AR too, not a property of the model class.** This is
  the single most defensible result in the pilot.
- AR beats the two non-oracle-equivalent diffusion arms by 5–6 pp (p ≤ 1.9e-09),
  and this survives ceiling restriction (p ≤ 3.7e-06).
- Zero generation errors in all six arms; grader self-test (canonical passes,
  stub fails) passes on every invocation.

---

## 3. Task surface, arms, protocol

**Surface:** official HumanEval-SingleLineInfilling, **n = 1033**
(`loubnabnl/humaneval_infilling`; Bavarian et al. 2022). Verified: 1033 lines,
sha256 `6fffc71e…`. Not RandomSpan (1640) and not MultiLine (5815).

**Grading:** `evalplus.eval.untrusted_check`, splicing
`prompt + middle + suffix`, with a per-invocation self-test asserting canonical
middles PASS and stub middles FAIL (this guard exists because a hand-rolled
grader that ignored return values caused an earlier retraction in this line of
work).

**Existing arms (all scored, greedy T=0, `max_new_tokens=64`, `initial_masks=4`):**

| arm | model | context | length | pass@1 (raw) | pass@1 (gold-feasible) |
|---|---|---|---|---|---|
| `qwen_fim` | Qwen2.5-Coder-7B (base) | prefix+suffix, native FIM sentinels | self-terminated | .7638 | **.9324** |
| `dreamon_oracle` | DreamOn-v0-7B | prefix+suffix | **oracle** mask count | .7590 | **.9337** |
| `dream_fim` | Dream-Coder-v0-**Instruct**-7B | prefix+suffix, fixed canvas | **oracle** | .7115 | .8733 |
| `dreamon_fim` | DreamOn-v0-7B | prefix+suffix | self-predicted | .7018 | .8625 |
| `qwen_prefix` | Qwen2.5-Coder-7B | prefix only | self, first line | .5324 | .6454 |
| `dream_prefix` | Dream-Coder-v0-Instruct-7B | prefix only | **oracle** | .4124 | .5030 |

**Three known defects in this arm set** (all fixable, listed in §6):
(i) grading axis carries a 0.8025 ceiling; (ii) `dream_*` uses Instruct while
`qwen_*` is base; (iii) `dream_prefix` gets an oracle length but `qwen_prefix`
does not, so the two suffix-gains are not measured under matched conditions.

---

## 4. Related Work / novelty boundary  *(mandatory before any GPU)*

### 4.1 Named collisions

| work | venue (verified) | overlap with B10 | residual gap |
|---|---|---|---|
| **DreamOn** — Diffusion LMs For Code Infilling Beyond Fixed-size Canvas (arXiv:2602.01326) | **ICLR 2026 Poster** (OpenReview `venueid=ICLR.cc/2026/Conference`; PDF footer "Published as a conference paper at ICLR 2026") | **Maximal.** Same benchmark, same split, and **already contains the matched AR control**: Table 1 gives Qwen2.5-Coder-7B **92.6** single-line / 58.7 multi-line vs DreamCoder+DreamOn 92.1 / 63.8. Also has an explicit **oracle-length** column (Table 2) and an initial-mask-length sweep | Reports **no** compute/token/NFE accounting vs AR; no ceiling/feasibility analysis of the grading axis; no RandomSpan; Python-only |
| **A3** — Autoregressive Models Rival Diffusion Models at ANY-ORDER Generation (arXiv:2601.13228) | **ICLR 2026 Poster** (OpenReview verified) | **Thesis-level.** Argues AR matches/beats diffusion at any-order generation incl. infilling — i.e. the *conclusion* B10 wanted to claim is already peer-reviewed | Proposes a new AR training scheme; does not audit *evaluation protocol* sensitivity on a fixed public benchmark |
| Diffusion LMs Can Approximate Optimal Infilling Lengths Implicitly (arXiv:2602.00476) | preprint (no OpenReview venue found) | training-free adaptive length for DLM infilling; +47.7 % over fixed-length | competes with DreamOn's premise; does not do AR-vs-diffusion cost or ceiling accounting |
| From Interface to Inference (arXiv:2607.26504) | preprint | insertion-based / latent masked diffusion; positional-uncertainty framing for code infilling | method paper, no protocol audit |
| Diffusion LMs Are Natively Length-Aware (arXiv:2603.06123) | preprint | **reports FLOPs** for DLM efficiency | the cost-accounting niche is being actively occupied — B10's "fair cost unit" angle is *not* unclaimed territory |
| Improving Variable-Length Generation via Length Regularization (2602.07546); Any-Order Flexible Length Masked Diffusion / FlexMDM (2509.01025) | preprints | variable-length masked diffusion | — |
| Bavarian et al., Efficient Training of LM to Fill in the Middle (arXiv:2207.14255) | — | defines FIM + this benchmark; establishes AR models *can* infill via sentinels | B10's `qwen_fim` arm is an application of this, not a discovery |
| LLaDA-8B, Dream-7B, DiffuCoder-7B, Deepseek-Coder-6.7B, Seed-Coder-8B | — | all already baselines **inside DreamOn Table 1** | — |

### 4.2 The overlap, stated plainly

SLATE's motivation was "DreamOn ships no matched-lineage AR native-FIM control on
its own evaluation surface, and the missing control reverses the conclusion."
**Both halves are false.** The control is in Table 1 of the camera-ready, and our
own repo note `TASK_SURFACE_LIT_GAPS.md` had already recorded it (including the
92.6). Worse, DreamOn's Table 1 shows Qwen at **92.6 vs 92.1** — the AR model is
*already ahead on single-line in the authors' own numbers*, and they say so
("on par with state-of-the-art autoregressive models"; they claim to *surpass* AR
only on **multi-line**). Our pilot therefore reproduces a published ordering
rather than reversing a conclusion.

Second, a peer-reviewed ICLR 2026 paper (**A3**) already argues the general
thesis that AR rivals diffusion at any-order generation.

### 4.3 Residual gap (what is actually unclaimed)

Narrow, but real, and consistent across the collision set:

1. **No paper in the set measures the benchmark's own feasibility ceiling on the
   axis it grades.** Our 0.8025 plus-axis gold ceiling, traced to 23 buggy parent
   tasks, silently changes the arm ordering. Reporting infilling pass@1 without a
   gold-refill control is an unexamined protocol defect *in the benchmark's use*.
2. **No paper reports cost in two units and shows the winner changes.** DreamOn
   reports none; 2603.06123 reports FLOPs within the diffusion family only.
3. **No paper isolates the oracle-length handout as a first-class factor** with
   the AR/non-oracle comparator held fixed (DreamOn's Table 2 oracle column is a
   reference point, not a controlled contrast against AR).

### 4.4 MUST NOT CLAIM

- ❌ "AR beats masked diffusion on diffusion's home turf." Not significant
  (p=0.635); sign flips on the feasible subset; **and it is DreamOn's own
  published result on single-line**, not ours.
- ❌ "The missing matched AR control reverses DreamOn's conclusion." The control
  is not missing.
- ❌ "AR dominates on both cost units." False (0.88×).
- ❌ "DreamOn's advertised length-elasticity toggle is inert / a functional defect
  in a public model." The kwargs are inert, but they are **not advertised**, and
  the capability was **active on 84.3 % of tasks**. Publishing this as a model
  defect would be a false accusation.
- ❌ "Every DreamOn number in circulation was produced with the toggle inert."
- ❌ "Matched lineage differing only in `mask_token_id`" for the arms as run
  (Instruct vs base).
- ❌ Any comparison of our raw plus-axis numbers to published values.
- ❌ Priority claims of any kind. DreamOn and A3 are ICLR 2026; if anything is
  written it is a follow-up audit.

### 4.5 Why the original death was still a bad kill

The direction died on a scan reporting "DreamOn covers this capability"
(`DLLM_RESULTS_20260807.md`: "Length-elastic 被 DreamOn 吃掉"). Per the standing
instruction that the bar is *identical*, not *overlapping*, that kill was
methodologically wrong — a capability is covered when a model demonstrably
performs it, and DreamOn measured **0.122** pass@1 on from-scratch HumanEval+ in
our hands. **But re-examination vindicates the kill's conclusion while rejecting
its reasoning**: DreamOn *does* demonstrably perform on its own surface (92.1
published; 0.8625 in our hands on the feasible subset), and its paper *does*
carry the AR control. The honest record is: the reasoning was lazy, and the answer
was right anyway. Both halves belong in any write-up.

---

## 5. ★ KILL GATE

Pre-registered. **Gate 1 costs 0 GPU and must run first.**

### Gate 1 — base-axis re-score (0 GPU, CPU only, hours)

Re-score all six existing arms with `score_infilling.py --which base`. Solutions
are already on disk; nothing is regenerated.

**KILL if:** on the base axis (gold ceiling 0.9894, so ≥98 % of items feasible),
the `qwen_fim` vs `dreamon_oracle` paired contrast is **not** significant at
α=0.05 **and** |Δ| < 0.02 — i.e. AR and the best diffusion arm are
indistinguishable on the axis the benchmark was designed for.

**Rationale, stated up front:** this is the *likely* outcome. On the plus axis
Δ = +0.0048 (p=0.635) and on the feasible subset Δ = −0.0012 (p=1.000). If the
base axis agrees, then **"AR vs diffusion" has no measurable answer on this
surface at n=1033**, the conclusion-reversal framing is dead for good, and the
only survivors are the two protocol observations (ceiling, cost unit) plus the
suffix-gain result. In that case B10 must be **rewritten as a protocol note or
archived** — it must NOT be re-framed to hunt a different ranking.

**PROCEED only if** the base axis produces a significant, ceiling-robust, and
*directionally stable* AR advantage over the strongest diffusion arm. Given the
evidence, treat proceeding as the unlikely branch.

### Gate 2 — lineage repair (~2–4 GPU-h, only if Gate 1 says PROCEED)

Re-run `dream_fim` / `dream_prefix` on Dream-Coder-v0-**Base**-7B (on disk).
**KILL if** the AR advantage over the best diffusion arm does not survive with
the post-training confound removed.

### Gate 3 — matched suffix-gain (~1–2 GPU-h)

Either give `qwen_prefix` the oracle length or remove it from `dream_prefix`.
**KILL the mechanism claim if** the two suffix gains stop being of comparable
magnitude once the handout is matched (this would mean the "affordance of task
framing" reading was an artefact of the asymmetry).

### Gate 4 — memorisation (design only; not authorised)

`KSPAN_INFILLING_RESULTS.md` §4.5 measured that identifier renaming + docstring
replacement costs **26–28 pp** on HumanEval infilling for both families, on a set
whose gold refill still scores 1.000. Any absolute number here is therefore
substantially surface recall. **No absolute pass@1 from this surface may be
reported as a capability measurement without a decontaminated companion.**

### Standing rule

If any gate kills, write `POSTMORTEM.md` and move to `proposal/archive/`. Do not
re-derive a new headline from the same six arms — that is the nested-ladder error
this repo already retracted twice (Retractions 6 and 7).

---

## 6. Cost to first result

| step | GPU | note |
|---|---|---|
| Gate 1 base-axis re-score, 6 arms | **0** | CPU; decides the direction |
| Paired stats + ceiling normalisation | 0 | already implemented for this audit |
| Gate 2 Base-7B re-run, 2 arms | ~2–4 h | 8-way shardable; ckpt on disk |
| Gate 3 matched handout | ~1–2 h | |

**Total ~4–8 GPU-h, not SLATE's 24** — but SLATE's "the marginal cost is prose"
is wrong: the arms as they stand do not support the claim, and the cheapest step
is the one most likely to end the direction.

---

## 7. Why this stays in `backlog`

1. The headline is not significant, and on the most defensible subset it reverses.
2. The stated motivation (missing AR control) is factually false.
3. The strongest thesis-level statement is already an ICLR 2026 poster (A3).
4. Claim (b), which SLATE called half the paper, does not exist.
5. What survives — ceiling sensitivity, cost-unit non-neutrality, suffix-gain
   symmetry — is a protocol note, and the protocol-audit niche is being occupied.

Promotion to `paper<X>/` requires, at minimum: Gate 1 passing on the base axis,
Gate 2 removing the lineage confound, and a novelty re-check against DreamOn +
A3 establishing that the *surviving* contribution is not just their Table 1
re-plotted.

**Honest one-line summary:** the numbers are real and exactly reproducible; the
story SLATE hung on them is not; the cheapest next experiment is free and will
probably end it.
