# Pre-registration — blast radius of the A05 harness defects (§1)

**Written before any grep of the three `dllm_draft` checkouts.** Committed prior to evidence
collection so the disposition below cannot be back-fitted.

## Hypothesis under test (the closeout's implicit claim)

> H0 (narrow scope): the stitch defect (c) is confined to the DreamOn full-program HumanEval path.
> No other arm's or benchmark's published number in `DLLM_RESULTS_20260807.md` is affected.

The closeout asserted narrow scope only for defect **(a)** (the `nfe = len(history)` mis-count),
and never asked the question for **(c)**. This pre-registration tests (c), and re-tests (a).

## Mechanism-derived prediction (stated before looking)

`combine_humaneval_prompt(prompt, generated)` applies `textwrap.indent(body, "    ")` **only on the
`else` branch** — i.e. only when `extract_python(generated)` contains **no top-level `def`/`async def`**.
Therefore the defect can bite an arm **only if** that arm's raw output is a bare, already-indented
function *body*. An arm that emits a complete `def ...` returns early and is **untouched**.

Prediction: arms differ, and the split is by **output shape**, not by model family per se. I expect
AR arms (which usually re-emit the full signature) to be mostly or entirely inert, and mask-diffusion
arms (which in-fill the body under the given prompt) to be affected. **If that holds, the asymmetry is
itself the finding, and it is a confound, not a courtesy: the shared post-processor silently penalises
exactly the arm shape that diffusion models produce.**

## Falsification conditions

**F1 — narrow scope FALSIFIED** if >=1 arm other than A05's own DreamOn cells (i.e. any arm whose
number is published in `DLLM_RESULTS_20260807.md` or a sibling results doc) both
(i) routes its stored `raw_output` through an extract-then-indent stitch, and
(ii) changes >=1 item's pass status when the corrected stitch is applied.
**Materiality threshold**: >=1.0 pp change in `pass@1 plus` on any published arm.

**F2 — narrow scope CONFIRMED (nil blast radius)** if for every non-A05 arm, either
(i) the code path provably never reaches the `textwrap.indent` branch, or
(ii) 0 items change pass status under the corrected stitch.
Confirming F2 is a legitimate and reportable outcome: it would mean the finding is about
*one driver*, and the "published numbers are affected" premise in my brief is **wrong**.

**F3 — Scaffold contamination (directly load-bearing for whether A05 deserved to die).**
If Scaffold's `.177`/`.354` themselves went through the buggy stitch, they are **understated**, the
K1 margin moves in the direction *favourable to A05*, and A05 may have been killed on a mis-measured
comparison. Threshold: if corrected Scaffold HE+ >= `.4817 - 5.0 pp = .4317`, K1's HE+ clause no
longer fires on the corrected numbers and **A05 must be re-opened**. (MBPP+ has no stitch, so its
`-0.05 pp` clause is unaffected — note this means K1 can only be undone on the HE+ leg, and K1
required *both*.)

**F4 — defect (a) re-test.** Narrow scope for the NFE mis-count is falsified if any driver other
than the DreamOn full-program driver writes an `nfe` field sourced from `len(...history)` rather
than a counted forward.

## Method (bind myself to it now)

1. CPU-first, all three checkouts (wzc1 `dllm_draft`, zwfy6 `dllm_draft`, zwfy6 `dllm_draft_104`).
2. Enumerate every definition **and** every call site of `combine_humaneval_prompt` / `extract_python`
   / any `textwrap.indent` on model output. Per arm, record: does it reach the indent branch?
3. For every arm with stored `raw_output`, re-grade as-run vs corrected with **evalplus +
   per-invocation self-test** (canonical PASS / stub FAIL). No hand-rolled verifier (invariant 1).
4. Arms whose `raw_output` was **not** stored are reported as *undecidable from disk*, never as
   "unaffected".
5. Every number labelled **RAN** (I computed it) or **READ** (from disk, someone else computed it).

## Registered in advance: what I will NOT allow myself to conclude

* Not "unaffected" from a null grep alone — absence of the function name in one checkout is not
  absence of the code path (it may be inlined or renamed).
* Not "affected" from a parseability change alone — parseability is not pass@1; only a graded
  pass-status flip counts toward F1.
