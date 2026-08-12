# A05 POSTMORTEM — structural dLLM cost frontier

**Direction**: does an explicit structural decoding runtime (Scaffold) beat its own model family
(DreamOn / Dream-Coder) on full-program code generation under a token-cost budget?
**Opened** 2026-08-12. **Closed** 2026-08-12. **Total cost**: ~21 GPU-h (K1) + 0 GPU (closeout).
**Killed by**: K1, a pre-registered experimental gate. Not by a literature collision, not by opinion.

---

## What was claimed

> On full-program code generation under a token-cost budget, an explicit structural decoding runtime
> reaches quality that its own model family does not reach at any comparable cost — while remaining
> well below autoregressive models, which dominate both axes.

Built on: Scaffold Medium `.177`/`.354` (HE+/MBPP+) vs DreamOn-v0-7B `.122`/`.085`, i.e. a
**+26.9 pp** MBPP+ margin, plus a matched plain-SFT control at `.000` to rule out "just trained more".

## What killed it

**K1** (registered in `PROPOSAL.md` §3 and `A05_K1_PREREGISTRATION.md`, committed `8f48ac1` before
any cell was graded): *fires if DreamOn at its best non-oracle canvas comes within 5.0 pp of Scaffold
Medium on both benchmarks.*

Sweeping **one integer** — `initial_masks` from 8 to 32, every other sampler knob frozen at the
archived values — moved DreamOn from `.122`/`.085` to `.2134`/`.3545`. After also fixing an HE+
post-processing bug found during closeout, DreamOn's best non-oracle HE+ is **`.4817`**.

| benchmark | Scaffold Medium | DreamOn best non-oracle | gap |
|---|---:|---:|---:|
| HE+ | .177 | **.4817** (canvas=128, corrected stitch) | **−30.5 pp** |
| MBPP+ | .354 | **.3545** (canvas=32) | −0.05 pp |

K1 fired with room to spare: DreamOn does not merely come *within* 5 pp, it **matches or beats**
Scaffold on both benchmarks. The `+26.9 pp` margin the whole direction rested on was **an artifact of
how the baseline was invoked.**

## The cost claim that briefly survived, and why it also died

K1's verdict left one claim standing: *"at matched quality Scaffold is 6.2×/8.2× cheaper in
forwards."* Closeout adjudicated it against 5 falsification conditions registered in advance
(`A05_CLOSEOUT_PREREGISTRATION.md`, commit `32f4e96`). **Four fired:**

* **Ratio of means reverses on the median.** DreamOn's median item uses **fewer** resources than
  Scaffold's on all four benchmark×axis combinations (0.56×-0.96×). "6.2×" describes tails.
* **The tail is an unmatched budget knob.** 12-13% of DreamOn's items sit at its own iteration cap
  (~2060-2180 forwards) and carry **57-61%** of all its NFE mass; Scaffold's cap is a different
  number (512). The mean-ratio largely measures two arbitrary settings.
* **Quality is not matched on HE+** — Scaffold is 30.5 pp *worse*, so "cheaper at matched quality"
  is only arguable on MBPP+ (where the gap is .354 vs .3545, i.e. the cost ratio was doing all the
  work).
* **The AR control dominates.** Qwen2.5-Coder-7B is **~70× cheaper than Scaffold on `tokens_fed`**
  *and* +.29 to +.35 more accurate — strict Pareto domination on the axis the repo itself designates
  as the cross-family-comparable one. Scaffold is competitive only on NFE, which
  `scripts/forward_cost.py` explicitly documents as *not* comparable across families.

So the surviving claim was a within-diffusion-family internal point — **exactly the description
under which this repo's original Pareto claim was RETRACTED in 2026-08.** It was not promoted.

## Three harness defects found along the way (all fixed at source)

1. **The logged `nfe` was never a forward count.** It was `len(output.history)`; DreamOn appends to
   `histories` at three sites, and it is `None` when `output_history=False` (hence r2's all-null).
   Archived 265.88/135.65 are `mean(len(history))`; true counted NFE is **172.3/153.4** — and MBPP+
   moves the *opposite* way, so the old numbers cannot be rescaled, only recounted.
2. **`mask_expansion` / `delete_eos_token` were always inert** (confirmed by execution, with a
   positive control). The r2 launcher advertised them as round-2 "fixes"; that fix never took effect,
   so r2 differed from r1 only in temperature/top_p.
3. **The HE+ stitch double-indented already-indented bodies**, understating *every* HE+ number this
   repo produced, worse at larger canvases: pass@1 plus `.1707 → .4817` at canvas=128, parseability
   `.287 → .963`. All 117 unparseable items at c128 were indentation errors. This also flipped the
   HE+ canvas curve from non-monotone to monotone, retiring a "DreamOn degrades at large canvases"
   sub-conclusion that was pure harness artifact.

## Lessons

1. **A baseline's config is part of your claim.** One integer moved MBPP+ by 26.6 pp — larger than
   most method gains anyone reports. Before claiming a margin over a baseline, sweep the baseline's
   most load-bearing knob. A05's margin did not survive the *first* such sweep.
2. **Report medians and tail mass, never a ratio of means alone.** The mean-ratio and the
   median-ratio here point in **opposite directions**. A single "N× cheaper" scalar concealed that.
3. **A cost claim needs the same baseline discipline as a quality claim.** The retracted Pareto claim
   died from a missing AR control; the "surviving" cost claim died from the *same* missing control.
   The lesson had been written down and was still re-learned — put the AR control in before the
   claim, not after.
4. **Fix mechanisms, not symptoms — and verify your fix changes something.** The stitch bug's
   published description ("it indents an already-indented body") yields a fix that is a **no-op**,
   because the de-indent happens inside `extract_python`'s trailing `.strip()`. My first fix altered
   0/164 outputs. Always assert your fix actually changed the data.
5. **A test that covers only the easy input is why the bug survived.** The stitch had a unit test —
   for the *unindented* case only. Two regression tests added.
6. **Post-processing bugs are re-gradeable for free.** Fixing the stitch needed **0 GPU**: the raw
   generations were on disk, so re-grading is not an approximation of a re-run, it *is* the re-run.
   Store raw model output, always.
7. **Killing a direction in one evening for ~21 GPU-h is a good outcome.** The gate was cheap, ran
   first, and fired. The alternative was writing up a paper on a baseline that never engaged.
8. **"Missing file" claims need both disks.** The roadmap's P1-C blocker
   (`scripts/generate_infilling.py` is missing) was **false** and had blocked that leg since
   2026-08-08; the file is on zwfy6 in a third checkout. Corrected in place.

## What outlives A05

The **evaluation-practice finding**, which is real and is *not* a claim about structural decoders:

> A widely-reported diffusion baseline's apparent weakness on full-program generation is a
> canvas-budget artifact. One sampler-config integer moves its MBPP+ from `.085` to `.3545`; a
> post-processing bug additionally understated its HE+ by up to 31 pp. A substantial fraction of the
> gap between a proposed method and its baseline was produced by how the baseline was invoked and
> post-processed, not by either model.

**Owner should be A01 (null-calibration methodology)**, not a new proposal and not a revived A05 —
it is the same species of finding as A01's, on a new surface. Deliberately **not** auto-promoted:
recorded in `STATUS.json:finding_that_outlives_a05` so the next agent picks it up on purpose.

**Also outliving A05**: the three source fixes (wzc1 `58bbb20`, zwfy6 `9651406`, `_104` `d214d37c`),
the corrections in `DLLM_RESULTS_20260807.md` / `DLLM_SALVAGE_ROADMAP_20260808.md` /
`SPANLEN_STRATIFIED_AUDIT.md`, and a reusable cost-audit harness that puts DreamOn, Scaffold and AR
on NFE + `tokens_fed` + `attended_context_sum` with mean/median/tail-share
(`code/a05_cost.py`, `evidence/a05_closeout_cost_audit.json`).

## What was NOT resolved (do not record these as settled)

* **Scaffold's `.177`/`.354` was never recomputed by A05.** Both verdicts lean on it and it is READ
  from `DLLM_RESULTS_20260807.md`, single round, 29 GB checkpoint wzc1-only. This is the weakest
  provenance link in the whole direction.
* **Oracle (per-item headroom) arms were never graded** — dropped for budget; excluded from the
  headline by invariant anyway.
* **`he_c512` / `mbpp_c128` / `mbpp_c512` never completed.** The corrected HE+ curve is *monotone up
  to c128*, so DreamOn's true ceiling on HE+ is **unknown and at least `.4817`** — larger canvases
  might go higher. This cannot change K1 (which takes the max) but it does mean nobody knows where
  DreamOn's full-program peak is.
* **`max_new` per-item headroom was never varied** — only the initial canvas. SPANLEN's original
  hypothesis was about `max_new=256`; that half remains formally untested.
* **K2 / K3 were never run** and should not be: K2 defends a margin whose sign flips between mean and
  median, K3 defends a margin that no longer exists.
* **The novelty check was never done.** It was gated on "before any GPU beyond K1", and K1 killed the
  direction first, so it is moot for A05 — but if the evaluation-practice finding is taken up under
  A01, that check still has to happen there.
