# A05 closeout — corrections propagated, surviving cost claim adjudicated, disposition set

**Written 2026-08-12.** Follows `A05_K1_CANVAS_SWEEP_VERDICT.md` (K1 FIRED).
**Pre-registration for §2 is `A05_CLOSEOUT_PREREGISTRATION.md`, committed `32f4e96` before any
ratio in this document was computed.**
**GPU used by this closeout: 0.** Everything here is CPU: re-grading reuses the `raw_output` rows
already on disk, and every cost number is re-aggregated from per-item JSONL.

---

## Verdict in one paragraph

K1's kill of A05's **quality** claim is confirmed and survives independent re-derivation — and gets
*stronger* once a third harness bug is fixed. The **cost** claim the K1 agent left standing
("Scaffold is 6.2×/8.2× cheaper in forwards at matched quality") is **dead on the same grounds as
the quality claim**: 4 of my 5 pre-registered falsification conditions fired, and the decisive one
is the AR control, which places Scaffold on the wrong side of the same Pareto argument that got the
original frontier claim RETRACTED in 2026-08. A05 is therefore **archived** with a POSTMORTEM. The
one genuinely new, publishable thing K1 produced is an *evaluation-practice* result — and it does
not belong to A05.

---

## 1. Corrections propagated (RAN — I verified each defect myself before propagating it)

I was asked to propagate three corrections. I re-derived all three from source rather than trusting
the report. All three are real; one had its **mechanism described imprecisely**, which mattered
because the imprecise version yields a fix that silently does nothing.

### (a) The archived `nfe` is not a forward count — CONFIRMED

* `scripts/generate_evalplus_dreamon.py:152-155` logged `len(output.history)`.
* `models/DreamOn-v0-7B/generation_utils.py` calls `histories.append(...)` at **three** distinct
  sites — lines **445** (transfer step), **476** (delete batch), **495** (expand batch). I read all
  three. So `len(history)` can reach ~3× the model-call count.
* I re-aggregated the archived rows: r1 non-null `nfe` = 164/164 and 378/378, means **265.88** and
  **135.65** — i.e. the numbers in A05's `PROPOSAL.md` §1 are exactly `mean(len(history))`. r2
  non-null = **0/164** and **0/378** (all null, because `output_history=False`).
* True counted NFE at the same archived setting (canvas=8), from K1's `model.forward` wrapper:
  **172.3** (HE+) / **153.4** (MBPP+).
* ⚠️ **A detail worth stating because it defeats the obvious workaround**: the error is *not* a
  uniform inflation. HE+ goes 265.88 → 172.3 (down) but MBPP+ goes 135.65 → **153.4 (up)**. You
  cannot rescale the old numbers; they must be recounted.
* **Consequence**: PROPOSAL.md §1's "**4.17× / 2.39× cheaper**" is void. Recomputed against true
  NFE at the archived canvas it would be 172.3/64.1 = 2.69× and 153.4/56.7 = 2.71× — but see §2,
  because *that comparison itself* is not a valid cost claim.

### (b) `mask_expansion` / `delete_eos_token` inert — CONFIRMED BY EXECUTION

Ran on `.73` against the real checkpoint config class (`code/verify_inert.py`):

```
has mask_expansion attr BEFORE: False
update() returned as UNUSED: {'mask_expansion': True, 'delete_eos_token': True}
has mask_expansion attr AFTER : False
control -> update(temperature=0.2) unused: {} -> cfg.temperature = 0.2
```

The control line is the part that makes this conclusive: a *real* parameter is consumed and returns
`{}`, while these two come straight back as unused. **Consequence**: the r2 launcher's header
claimed these as round-2 "fixes"; that fix never took effect, so **r2 differs from r1 only in
temperature/top_p**. Corrected in `scripts/_run_baselines_r2_wzc1.sh` and the kwargs removed from
the driver.

### (c) HE+ stitch double-indents — CONFIRMED, but the stated mechanism was imprecise

The K1 agent's claim (113 of 117 unparseable at c128; 158/164 restored) reproduces **exactly**.
But its description — "`combine_humaneval_prompt()` indents the raw output by 4 spaces … DreamOn
already emits a 4-space-indented body" — omits *where* the de-indentation happens, and my first
attempt at the fix therefore **did nothing at all** (0/164 outputs changed).

The precise mechanism: `combine_humaneval_prompt` calls `extract_python()` **first**, and
`extract_python` ends in `.strip()`. That removes leading whitespace from the **first line only**.
So line 1 lands at column 0 while lines 2..n keep their original 4-space depth. `textwrap.indent`
then adds 4 uniformly → line 1 at 4, line 2 at 8 → `IndentationError: unexpected indent`.

> **The trap**: a `textwrap.dedent` applied **after** `extract_python` is a **no-op**, because line 1
> has already been stripped so the common leading prefix is already 0. The dedent must be applied
> **before** extraction. Anyone implementing this from the K1 description alone would write the
> no-op version and conclude the bug was not real.

Observed error strings at c128 confirm the geometry: **all 117** unparseable items fail with an
indentation error — **86** `unexpected indent` and **31** `unindent does not match any outer
indentation level` — at source lines 10-38, i.e. in the body, never at the signature. Not one is a
different syntax error. (Note this means even the 4 items my fix does not rescue fail *for the same
class of reason*, e.g. `HumanEval/50`, where DreamOn emitted two functions with the second body
mis-indented relative to its own `def`; the harness cannot repair that.)

**Fix + regression tests**: `combine_humaneval_prompt` now dedents the raw text before extraction.
The pre-existing unit test only covered an **unindented** body, which is precisely why the bug
survived; I added `test_combines_already_indented_dreamon_body` and
`test_stitch_preserves_relative_indentation_depth` (asserting depths `[4, 8, 4]`). All 4 stitch
tests pass on both zwfy6 checkouts. (The full test module cannot import on zwfy6 due to a
**pre-existing, unrelated** stale `scaffold_coder.tokenizer_utils` missing `edit_source_token_ids`;
I ran the stitch functions in isolation rather than "fixing" an unrelated checkout.)

### (c') Corrected cells — RAN, 0 GPU, labelled as corrected, not as replacements

Generation is **byte-identical** to K1 (same `raw_output` rows, no model loaded). Only
post-processing changed. Same grader with the mandatory self-test (canonical PASS / stub FAIL).
Assertions: 164 items, 0 dups, task_id set equals the dataset, 0 generation errors.

| HE+ cell | parseability as-run → corrected | pass@1 **plus** as-run → corrected | Δ | items gained / lost |
|---|---|---|---|---|
| `he_c8` (= archived setting) | .988 → **1.000** | .1280 → **.1341** | +0.61 pp | 1 / 0 |
| `he_c32` | .860 → **.982** | .2134 → **.2561** | +4.27 pp | 7 / 0 |
| `he_c128` | .287 → **.963** | .1707 → **.4817** | **+31.10 pp** | 52 / 1 |

**Was re-running the c128 cell warranted?** No — and it would have been the wrong move. The bug is
**purely post-processing**, so re-grading the stored `raw_output` is not an approximation of a
re-run, it is *exactly* what a re-run would produce, at **0 GPU instead of ~6 GPU-h**. Spending GPU
here would have bought nothing. (I did spot-check the outputs by hand: the rescued programs are
ordinary correct code — e.g. `HumanEval/0` is a clean `sort`-then-scan solution — not degenerate
passes. The single lost item, `HumanEval/50`, is a genuinely malformed two-function output.)

**Two consequences that go beyond a number changing:**

1. **The HE+ canvas curve was non-monotone only because of the bug.** As-run: .128 → .213 → .171,
   which the K1 verdict read as "canvas=32 is the peak, c128 degrades". Corrected: **.134 → .256 →
   .482**, monotone increasing. "DreamOn degrades at large canvases" was a harness artifact.
   `STATUS.json`'s `unverified` entry claiming the HE+ curve is non-monotone is now **wrong** and
   has been corrected.
2. **K1 fires harder.** Best non-oracle HE+ is **.4817**, versus Scaffold Medium's `.177` — a
   **−30.5 pp** gap in DreamOn's favour, not −3.64 pp. K1's threshold was "within 5.0 pp"; it is
   now exceeded by a factor of six.

### Where the corrections landed (two-disk rule applied)

`dllm_draft` exists on **both** disks with **different contents**, and `dllm_draft_104` is a third
checkout. I checked all three and treated them differently on purpose:

| target | disk | action | commit |
|---|---|---|---|
| `dllm_draft` | wzc1 | docs + code + tests corrected | `58bbb20` |
| `dllm_draft` | zwfy6 | code + tests only (**it has no `DLLM_RESULTS`/`SPANLEN`/launcher**) | `9651406` |
| `dllm_draft_104` | zwfy6 | code + tests + launcher + **correction pointer** in its `DLLM_RESULTS` | `d214d37c` |

`dllm_draft_104/DLLM_RESULTS_20260807.md` is an **earlier 275-line branch** of the wzc1 739-line
file, not a stale copy of it (wzc1 carries 9 later retractions). Overwriting it would have destroyed
provenance, so I inserted a dated correction pointer naming wzc1 as authoritative and listing the
four specific wrong statements *by line number in that file*.

Documents corrected inline, each with the correction and its evidence stated (nothing silently
overwritten): `DLLM_RESULTS_20260807.md` (new CORRECTION BLOCK + the `+mask_expansion` table row +
the 2026-08-07 bullet whose own conclusion C4 overturns), `DLLM_SALVAGE_ROADMAP_20260808.md`
(§1.1 "DreamOn long-span advantage" row + §P1-C, incl. striking the false "missing script" blocker),
`SPANLEN_STRATIFIED_AUDIT.md` (§2's "single most important follow-up" bullet — the control it asked
for has now been run and the branch it hedged for is the one that fired).

**Blast radius, checked rather than assumed**: defect (a) is confined to
`generate_evalplus_dreamon.py`. `generate_evalplus_dream.py` and `generate_evalplus_dream_alg.py`
log `nfe: args.steps` (a fixed schedule, correct by construction) and `generate_kspan.py` /
`generate_infilling.py` use the hook-based `ForwardCostTracker`. Defect (c) touches only
`combine_humaneval_prompt`, used by that one driver (and its test). So **the infilling / k-span
numbers do not inherit either defect** — a point the task's framing left open.

---

## 2. Adjudicating the surviving cost claim: **DEAD**

The claim: *"at matched quality Scaffold is 6.2×/8.2× cheaper in forwards."*
Falsification conditions were fixed in advance (`A05_CLOSEOUT_PREREGISTRATION.md`, commit `32f4e96`);
the decision rule was "**2 or more fire → not publishable as stated**", with F3 alone capping the
claim at "within-family internal point".

**Result: F1, F2, F3, F5 fire. F4 partially. That is 4 of 5.**

### F1 — the ratio is a tail artifact: **FIRES**

| benchmark | axis | DreamOn c32 | Scaffold | ratio of **means** | ratio of **medians** |
|---|---|---:|---:|---:|---:|
| HE+ | NFE | 393.7 | 64.1 | **6.14×** | 32 / 57 = **0.56×** |
| HE+ | `tokens_fed` | 124,348 | 13,980 | **8.89×** | 8,048 / 11,166 = **0.72×** |
| MBPP+ | NFE | 466.0 | 56.7 | **8.21×** | 32 / 47 = **0.68×** |
| MBPP+ | `tokens_fed` | 101,202 | 7,080 | **14.29×** | 5,059 / 5,280 = **0.96×** |

**The direction reverses on every single axis.** On the *median* item, DreamOn is **cheaper than
Scaffold** (0.56×–0.96×). The headline "6.2×/8.2×" describes a ratio of means between two
heavy-tailed distributions, and it says nothing about the typical item. The pre-registered trigger
was "reverses direction, or falls below 2×, on either benchmark" — it reverses on all four.

### F5 — the tail is an unmatched budget knob, not decoding efficiency: **FIRES**

| cell | items at DreamOn's iteration cap | share of **all** NFE mass held by those items | items where NFE == canvas exactly |
|---|---:|---:|---:|
| `he_c32` | 19/164 (11.6%) | **61.3%** | 97/164 |
| `mbpp_c32` | 48/378 (12.7%) | **56.8%** | 196/378 |
| `he_c8` | 10/164 (6.1%) | **72.9%** | 133/164 |

Pre-registered trigger: ">50% of total NFE mass from <15% of items". Measured: **57-73% of the mass
from 6-13% of items**. Those items are sitting at DreamOn's own cap
(`2*max_gen_len + 2*expand_budget` ≈ 2060-2180 forwards), while Scaffold's cap is
`max_model_calls=512` — a *different, unmatched knob*. So the mean-ratio is substantially a readout
of two arbitrary budget settings. Meanwhile 59-84% of items terminate at exactly `NFE == canvas`,
i.e. one pass per initial mask with no expansion at all.

### F2 — "matched quality" is not matched in Scaffold's favour: **FIRES**

| benchmark | Scaffold | DreamOn best non-oracle | gap |
|---|---:|---:|---:|
| HE+ | .177 | **.4817** (c128, corrected) | **−30.5 pp** |
| MBPP+ | .354 | .3545 (c32) | −0.05 pp |

On MBPP+ the two are genuinely indistinguishable (−0.05 pp, well inside DreamOn's own 1.9 pp
round-to-round movement) — so "matched" is fair there, and this is the case the task asked me to
scrutinise: *is it comparing a `.354` to a `.3545` and letting the cost ratio do all the work?*
Yes — and that is the **charitable** reading. On HE+ it is not matched at all: DreamOn is **30.5 pp
better**. "Cheaper at matched quality" cannot be asserted for a system that is 30 pp *worse* on one
of the two benchmarks; that is a quality/cost trade-off, and a steep one.

### F3 — the AR control: **FIRES, and this one is fatal on its own**

Qwen2.5-Coder-7B, same architecture family, `measured_matches_analytic: true`, 0 errors, exact item
counts. All three arms on all three axes (means):

| benchmark | arm | pass@1 plus | NFE | `tokens_fed` | `attended_context_sum` |
|---|---|---:|---:|---:|---:|
| **HE+** | **AR (Qwen2.5-Coder-7B)** | **.5244** | **73.3** | **206** | 20,093 |
| | Scaffold Medium | .177 | 64.1 | 13,980 | 13,980 |
| | DreamOn c32 | .2134 (.4817 @c128) | 393.7 | 124,348 | 124,348 |
| **MBPP+** | **AR** | **.6481** | 51.4 | **101** | 6,078 |
| | Scaffold Medium | .354 | 56.7 | 7,080 | 7,080 |
| | DreamOn c32 | .3545 | 466.0 | 101,202 | 101,202 |

* On **`tokens_fed`** — the axis the repo itself designates as the cross-family-comparable one — AR
  is **67.7× / 70.2× cheaper than Scaffold** *and* **+.347 / +.294 more accurate**. Strict Pareto
  domination. This reproduces `DLLM_SALVAGE_ROADMAP` §1.1's "dramatically cheaper under
  `tokens_fed`" from per-item rows, at ~70×, exactly as the task anticipated.
* On **NFE**, Scaffold is nominally competitive (0.87× / 1.10×) — but NFE is explicitly **not
  cross-family comparable** (`scripts/forward_cost.py`: "one diffusion step re-feeds a whole canvas,
  one AR decode step feeds a single token"). Quoting the one axis on which Scaffold looks fine,
  while that axis is documented as invalid across families, is exactly the error behind Retraction 3.
* On **`attended_context_sum`**, Scaffold is 0.70× on HE+ / 1.16× on MBPP+ — genuinely
  comparable-to-slightly-better. This is the *only* axis where Scaffold is not dominated, and it is
  reported here rather than dropped. It still buys ~0.30 less accuracy.

**So the cost claim beats DreamOn (on means only) and loses to AR by ~70× while also being ~30 pp
less accurate.** That is precisely the "diffusion-family internal point" description under which
the Pareto claim was retracted in the first place. Per the pre-registered rule, F3 alone caps this
at a within-family observation; it cannot be a standalone cost contribution.

### F4 — axis disagreement: **PARTIALLY FIRES**

NFE and `tokens_fed` disagree in *magnitude* by more than the 2× tolerance on MBPP+ (8.21× vs
14.29×), and they disagree in *direction* against AR (Scaffold "wins" on NFE at 1.10×, loses 70× on
`tokens_fed`). So no single scalar "X× cheaper" may be quoted at all — every statement must name its
axis. I count this as fired for the AR comparison, and note it as a magnitude-only failure for the
DreamOn comparison.

### §2 verdict

**(iii) Dead on the same grounds as the quality claim.** Not "needs more work": the failure is not
missing controls, it is that the available controls point the other way. A third and fourth round
(the old K2) would refine the noise floor on a margin whose *sign flips* when you switch from mean
to median, and would not touch F3 at all. **No further GPU should be spent on it**, and it should
**not** be promoted into its own proposal. The honest one-line statement is:

> On MBPP+, Scaffold Medium and DreamOn (canvas=32) reach statistically indistinguishable pass@1;
> Scaffold uses ~8× fewer forward passes **in the mean** but ~1.5× *more* on the median item,
> because 13% of DreamOn's items sit at an iteration cap and carry 57% of its total cost. On HE+
> Scaffold is 30.5 pp worse. Both are dominated by a matched AR control by ~70× on `tokens_fed`.

### One premise of the task I should push back on

The framing "with the corrected NFE (172.3/153.4), **recompute the ratio** — does 6.2×/8.2×
survive?" contains a mismatch I should not have silently resolved: 172.3/153.4 are the **canvas=8**
figures, while 6.2×/8.2× came from **canvas=32** (393.7/466.0). They are different cells, so
substituting one into the other would produce a number describing no experiment. Corrected NFE at
canvas=8 vs Scaffold gives 2.69×/2.71×, but that pairs Scaffold against the *crippled* canvas — the
very comparison K1 killed. The defensible recomputation is the canvas=32 one shown above, and it
fails for the reasons in F1/F3/F5 rather than because 6.2 becomes some other number.

---

## 3. What is actually publishable here — and it is not A05's

K1's real result is about **evaluation practice**: *one sampler-config integer moves a widely-cited
diffusion baseline's MBPP+ from .085 to .3545 (+26.6 pp) and its HE+ from .122 to .4817 (+35.9 pp
with the stitch fixed), reversing a published-looking comparison.* Add the three harness defects and
the pattern is: **a substantial fraction of the apparent gap between a proposed method and its
baseline was produced by how the baseline was invoked and post-processed, not by either model.**

That is a useful, honest contribution. But **it is not a contribution about a structural decoding
runtime**, which is what A05 registered. Judged against the project's own ownership rules:

* **A01 (null-calibration methodology)** is the natural home for "what survives when you calibrate
  the baseline properly" — this is the same species of finding, on a new surface (code generation
  instead of QA), and A01 is already in major revision with an established methodological frame.
* **B10** owns infilling; untouched here.
* A05's own contribution, honestly stated, is **a negative result plus a set of harness bug fixes**.
  That is worth preserving as provenance, and it is not worth a paper section of its own.

I have **not** created a new proposal for this. Per the promotion rules a new direction needs its own
`PROPOSAL.md` + kill gate, and the right owner is A01 rather than a new ID; I have recorded the
finding and the pointer in A05's `STATUS.json` (`finding_that_outlives_a05`) so the next agent can
pick it up deliberately instead of inheriting it by accident.

---

## Disposition: **archive**

A05 moves to `proposal/archive/A05-structural-dllm-cost-frontier/` with this document and
`POSTMORTEM.md`. Nothing is deleted: `PROPOSAL.md`, the pre-registrations, `evidence/` (per-item pass
maps, grader self-tests, the corrected cells, the cost audit), and `code/` all travel with it.

**The kill is attributed to K1's experimental gate**, a pre-registered canvas sweep that ran on GPU
and fired — not to a literature argument and not to an opinion. The cost claim's death is likewise
experimental: the AR control and the median/tail decomposition are measurements, and my
falsification conditions were registered before I computed any of them.

---

## RAN vs READ

**RAN (produced by this closeout, 0 GPU):**
* Independent re-derivation of the stitch bug from raw `raw_output` (`code/fix_stitch.py`), including
  the finding that the naive post-extraction dedent is a no-op.
* Re-grading of all three HE+ cells with the corrected stitch, evalplus self-tested
  (`code/a05_regrade_stitch.py` → `evidence/cells_corrected/a05_closeout_stitch_regrade.json`).
* Execution-level proof that `mask_expansion`/`delete_eos_token` are inert, with a positive control
  (`code/verify_inert.py`).
* Re-aggregation of NFE / `tokens_fed` / `attended_context_sum` (mean, median, p90, max, tail share)
  for DreamOn (5 cells), Scaffold Medium (2 runs, `process`-or-`failure_process` accessor), and the
  AR control (4 runs) → `evidence/a05_closeout_cost_audit.json` (`code/a05_cost.py`).
* Iteration-cap concentration analysis (`code/cap_check.py`).
* Confirmation that r1 `nfe` non-null is 164/164 & 378/378 with means 265.88/135.65, and r2 is
  0/164 & 0/378.
* Two new regression tests, run on both zwfy6 checkouts.

**READ (taken from disk, not recomputed here):**
* **Scaffold Medium pass@1 .177 / .354** — `DLLM_RESULTS_20260807.md:447/:456`. **Never recomputed
  by A05**; the 29 GB `Scaffold-v0-stage1-7B` checkpoint is wzc1-only and re-scoring it was out of
  scope. Its *costs* I did recompute from per-item rows. This remains A05's weakest provenance link
  and it is load-bearing for both the K1 and the cost verdicts.
* **AR pass@1** `.5244`/`.6481` — `report.json` of the two AR runs (I recomputed their costs but not
  their grades).
* DreamOn's 1.9 pp round-to-round noise — `PROPOSAL.md` §2 D-C (r1→r2, n=2 rounds, cannot estimate sd).
* The three `histories.append` line numbers and the `max_model_calls`/iteration-cap constants — read
  from source.

**Never mixed**: no r1 NFE is combined with an r2 pass@1 anywhere above. The corrected HE+ cells are
labelled corrected and their as-run values are shown alongside. No oracle arm appears in any
comparison (both oracle cells were never graded).
