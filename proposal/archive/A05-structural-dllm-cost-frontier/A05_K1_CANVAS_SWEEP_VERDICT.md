# A05 K1 — canvas sweep verdict: **K1 FIRES**

**Gate**: K1 (pre-registered in `PROPOSAL.md` §3 and `A05_K1_PREREGISTRATION.md`, both committed
before any A05 cell was graded).
**Ran**: 2026-08-12, node `.73` (8×H20, zwfy6). **Cost**: 14.39 GPU-h compute / 20.66 GPU-h billed.
**Model**: `DreamOn-v0-7B` bf16. **Grader**: `evalplus.eval.untrusted_check`, self-tested per invocation.

---

## Verdict

> **K1 fires if** DreamOn at its best *non-oracle* canvas setting reaches within **5.0 pp** of
> Scaffold Medium on **both** benchmarks.

| benchmark | Scaffold Medium | DreamOn best non-oracle | gap (Scaffold − DreamOn) | within 5.0 pp? |
|---|---:|---:|---:|:--:|
| **HE+** | .177 | **.2134** @ canvas=32 | **−3.64 pp** | ✅ |
| **MBPP+** | .354 | **.3545** @ canvas=32 | **−0.05 pp** | ✅ |

Both clauses satisfied — and satisfied in the strong direction: DreamOn does not merely come *within*
5 pp, it **matches or exceeds** Scaffold Medium on both benchmarks. **K1 FIRES. A05's headline is dead.**

The +26.9 pp MBPP+ margin that A05 was built on was **a canvas artifact**. Changing one number in the
baseline's sampler config — `initial_masks` from 8 to 32, nothing else — moves DreamOn from .085 to
.3545 and erases the entire margin. Scaffold's measured advantage over its own family was an artifact
of how the family member was invoked.

**This is a successful gate, not a failed experiment.** A05 cost ~21 GPU-h and one evening to reach a
clean disposition, instead of being written up on a baseline that never engaged.

---

## RAN — measurements produced by this gate

All cells: 8 shards, `index % 8` sharding, exactly 164 (HE+) / 378 (MBPP+) items, 0 duplicate
`task_id`, 0 nan, 0 generation errors, shard completeness asserted before every merge.
Every sampler knob except `initial_masks` is frozen at the archived r2 values
(T=0.2, top_p=0.9, `alg=entropy`, `alg_temp=0`, `number_transfer_tokens=1`, `max_new_tokens=512`).

| cell | canvas | pass@1 base | **pass@1 plus** | NFE mean | tokens_fed (eff) | tokens_fed (padded) | gen tok mean | median emitted/gold | parseability | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `he_c8` | 8 | .1402 | **.1280** | 172.3 | 39,944 | 115,989 | 2.35 | 0.000 | .988 | 0.78 |
| `he_c32` | 32 | .2256 | **.2134** | 393.7 | 124,348 | — | 12.87 | 0.038 | .860 | 1.80 |
| `he_c128` | 128 | .1829 | **.1707** | 593.4 | 240,414 | — | 48.53 | 0.227 | .287 † | 6.07 |
| `mbpp_c8` | 8 | .0979 | **.0899** | 153.4 | 23,367 | — | 1.57 | 0.000 | .989 | 1.41 |
| `mbpp_c32` | 32 | .3968 | **.3545** | 466.0 | 101,202 | 279,393 | 11.43 | 0.460 | .984 | 4.33 |

† harness artifact, not model behaviour — see "The HE+ stitch bug" below.

> **★ 2026-08-12 — the three HE+ rows above are SUPERSEDED by corrected cells.** The stitch bug was
> fixed and the same `raw_output` re-graded (generation byte-identical, 0 GPU, evalplus self-tested):
> `he_c8` **.1280 → .1341**, `he_c32` **.2134 → .2561**, `he_c128` **.1707 → .4817**; parseability
> `.988 → 1.000`, `.860 → .982`, `.287 → .963`. **The HE+ canvas curve is therefore MONOTONE
> (.134 → .256 → .482), not the non-monotone .128 → .213 → .171 shown above** — so the reading that
> "canvas=32 is the peak and c128 degrades" was itself the harness artifact, and DreamOn's best
> non-oracle HE+ is **.4817 at c128**, a **−30.5 pp** gap versus Scaffold rather than −3.64 pp.
> K1 fires harder, not weaker. MBPP+ rows are unaffected (no stitch). See
> `evidence/cells_corrected/a05_closeout_stitch_regrade.json`.

**Not run** (budget): `he_c512`, `mbpp_c128`, `mbpp_c512`, and both oracle cells. `mbpp_c128` was
killed at 30/378 and is not graded. **This does not affect the verdict**: K1 takes the *max* over
non-oracle settings, and the ≤5.0 pp clause is already satisfied on both benchmarks at canvas=32, so
additional settings can only keep K1 fired. Oracle cells are excluded from the K1 decision by
invariant 4 regardless, which is why they were the cells I dropped.

### Reproduction check (my pre-registered falsification condition #1) — PASSES

`canvas=8` is the archived setting, so it doubles as a reproduction check. Pre-registered tolerance
was ±3 pp:

| quantity | mine | archive r2 | Δ |
|---|---:|---:|---:|
| HE+ pass@1 plus | .1280 | .122 | +0.6 pp |
| HE+ pass@1 base | .1402 | .140 | +0.0 pp |
| MBPP+ pass@1 plus | .0899 | .085 | +0.49 pp |
| HE+ `generated_tokens` mean | 2.35 | 2.29 | — |
| MBPP+ `generated_tokens` mean | 1.57 | 1.52 | — |
| HE+ empty raw output | 128/164 | 128/164 | exact |
| MBPP+ empty raw output | 332/378 | 333/378 | 1 item |

Well inside tolerance, with the behavioural telemetry matching almost exactly. The harness reproduces
the archive, so the other cells are trustworthy against the .177/.354 reference. MBPP+ grading uses
the byte-identical groundtruth pickle from the archived run (`ee43ecab…pkl`, md5 verified after a
cross-disk copy), so this is the same grader, not merely the same grader *version*.

### Degenerate-pass check (falsification condition #5) — PASSES

0 of 134 passing MBPP+ items at canvas=32 had `generated_tokens == 0` or empty raw output
(passers' emitted length: min 9, median 18, max 73 tokens). The gain is real emitted code, not the
HE+ `prompt + "pass"` fallback happening to satisfy tests. Same result for all five cells.

---

## The mechanism question: does under-generation go away when the canvas grows?

**Partly — and not enough to meet the roadmap's own criteria.**

| | canvas=8 | canvas=32 | canvas=128 | criterion |
|---|---:|---:|---:|---|
| HE+ median emitted/gold | 0.000 | 0.038 | 0.227 | ≥ 0.80 ❌ |
| HE+ parseability | .988 | .860 | .287 † | ≥ 0.90 |
| HE+ empty outputs | 128/164 | 75/164 | **0/164** | — |
| HE+ gen tokens mean | 2.35 | 12.87 | 48.53 | — |
| MBPP+ median emitted/gold | 0.000 | 0.460 | not run | ≥ 0.80 ❌ |
| MBPP+ parseability | .989 | .984 | not run | ≥ 0.90 ✅ |
| MBPP+ empty outputs | 332/378 | 165/378 | not run | — |

On the **65+ token spans** the roadmap singles out: HE+ median ratio 0.231 and parseability .264 at
canvas=128 (n=159); MBPP+ long spans (n=51) stay at ratio 0.000 even at canvas=32.

**Answer**: the *pathology* is a canvas artifact — empty output goes 128 → 75 → **0** on HE+ and
332 → 165 on MBPP+, and mean emitted length grows 2.35 → 12.87 → 48.53. So "DreamOn emits ~2 tokens
and nothing on 80% of items" is purely an artifact of `initial_masks=8`, exactly as D-A suspected.
But the *capability* claim still fails: median emitted/gold peaks at 0.46 (MBPP+ @32) and 0.23
(HE+ @128), both far below the 0.80 continuation criterion, and neither benchmark meets both criteria
at any measured setting. **DreamOn under a bigger canvas is no longer crippled, but it is still not a
well-calibrated length controller on full-program generation.** It is good enough to erase A05's
margin, which is all K1 needed to decide.

---

## Cost: the parity is bought at 6–8× Scaffold's cost (falsification condition #3)

> ### ⛔ SUPERSEDED 2026-08-12 — this section's "surviving" cost claim was subsequently FALSIFIED.
> Do **not** revive the 6.2×/8.2× figure from here. `A05_CLOSEOUT_VERDICT.md` §2 tested it against
> five conditions registered in advance (`A05_CLOSEOUT_PREREGISTRATION.md`, commit `32f4e96`) and
> **four fired**:
> * The ratio of **means** below **reverses on the median** — DreamOn's median item is *cheaper*
>   than Scaffold's on all four benchmark×axis pairs (0.56×–0.96×). Note `nfe_median` is 32 versus
>   Scaffold's 57; the table below quotes means only.
> * 12–13% of DreamOn's items sit at its own **iteration cap** (~2060–2180 forwards) and carry
>   **57–61%** of all its NFE mass. Scaffold's cap is a different, unmatched number (512).
> * Quality is matched only on **MBPP+**; on HE+ Scaffold is **30.5 pp worse** once the stitch bug
>   is fixed (DreamOn reaches **.4817**, not .2134).
> * The **AR control** — absent from this section — is ~**70× cheaper than Scaffold on
>   `tokens_fed`** *and* +.29/+.35 more accurate, i.e. strict Pareto domination. Scaffold looks
>   competitive only on NFE, which `scripts/forward_cost.py` documents as **not** comparable across
>   families.
>
> Net: the claim is a diffusion-family internal point — exactly the description under which the
> Pareto claim was already RETRACTED. It was **not** promoted to its own proposal, and **no further
> GPU** is authorised for it.

The A05 claim is explicitly *"under a token-cost budget"*, so parity at any cost does not preserve it —
but the cost must be stated, because it is the one thing that partially survives:

| | DreamOn @ canvas=32 | Scaffold Medium | ratio |
|---|---:|---:|---:|
| HE+ NFE mean | 393.7 | 63.8 | **6.2× more** |
| MBPP+ NFE mean | 466.0 | 56.7 | **8.2× more** |
| HE+ tokens_fed (effective) | 124,348 | 13,980 | **8.9× more** |

So the honest post-K1 statement is: **Scaffold Medium reaches the same quality as DreamOn at roughly
one-sixth to one-eighth the forward passes.** That is a *cost* result at matched quality, not the
*quality* result A05 claimed ("reaches quality its own family does not reach at any comparable cost").
The quality claim is dead. Whether the cost claim is worth a paper is a separate question, and it
would need K2/K3-style controls of its own plus an AR comparison — Qwen2.5-Coder-7B reaches .707/.680,
so both diffusion arms remain far below the AR ceiling on quality regardless.

---

## READ — facts I took from disk, not measured here

* **Scaffold Medium .177 / .354** — `dllm_draft/DLLM_RESULTS_20260807.md:447` (HE+ tier table) and
  `:456` (MBPP+ tier table). **Not recomputed**: the 29 GB `Scaffold-v0-stage1-7B` checkpoint is
  wzc1-only and K1 does not need it.
* **Archived DreamOn .122 / .085** — `runs/dreamon_heplus_r2/evalplus.out`,
  `runs/dreamon_mbppplus_r2/evalplus.out` (evalplus's own stdout). r1 was .110 / .066.
* **Scaffold NFE 63.8 / 56.7**, **Scaffold HE+ cost 13,980 tok** — same tier tables / `BASELINE_STATS.md`.
* **AR ceiling .707 / .680** — Dream-Coder-v0-Instruct-7B, `DLLM_RESULTS_20260807.md`.
* **Archived per-item telemetry** (`initial_masks=8` for every item, `generated_tokens` mean 1.90/2.29,
  ~80% empty, `nfe: null` throughout r2) — re-aggregated by me from
  `runs/dreamon_*/metrics.rank*.jsonl`, confirming §2 D-A's numbers exactly.

---

## Corrections to the record (things PROPOSAL.md and the archive got wrong)

1. **The archived `nfe` is not a forward-pass count, even where it is non-null.** The driver recorded
   `nfe = len(output.history)`. In `generation_utils.py::_sample`, `history` is appended once after the
   transfer step **and again** after any delete batch **and again** after any expand batch — so it can
   reach ~3× the number of model calls; and it is `None` whenever `output_history=False`, which is why
   every r2 item logs `nfe: null`. The r1 "NFE mean 265.9 / 135.7" quoted in `PROPOSAL.md` §1 is
   therefore **not an NFE**. My harness counts `model.forward` calls directly: the true value at the
   archived setting is **172.3** (HE+) / **153.4** (MBPP+). PROPOSAL.md §1's cost column and the
   "4.17× / 2.39× cheaper" figures derived from it should be corrected.
2. **`mask_expansion=True` / `delete_eos_token=True` are inert — now verified by execution, not just
   by reading.** `GenerationConfig.update()` returns exactly
   `{'mask_expansion': True, 'delete_eos_token': True}` as *unused* kwargs and the attributes never
   exist on the config. Invariant 7 is confirmed; I omitted them, which changes nothing behaviourally.
   Note this means the archived runs' `_run_baselines_r2_wzc1.sh` header comment ("Fixes applied…
   mask_expansion=True + delete_eos_token=True") describes a fix that never took effect.
3. **The HE+ stitch bug** (new, found here). `combine_humaneval_prompt()` indents the raw output by 4
   spaces under the prompt whenever the raw contains no top-level `def`. DreamOn already emits a
   4-space-indented body, so the stitch **double-indents** it → `SyntaxError`. This, not model
   degradation, is what drives HE+ parseability to .287 at canvas=128 (MBPP+, which uses
   `extract_python` with no stitching, stays at .984). Attribution: of 117 unparseable HE+ items at
   canvas=128, **113 are caused by the double indent** and only 4 are genuinely malformed; a
   dedent-aware stitch restores **158/164** parseable. The bug is shared with the archived runs so
   comparability holds, but **HE+ pass@1 at large canvases is understated** — i.e. the true HE+ curve
   is even more favourable to DreamOn, which strengthens K1.
4. **The cost estimate in PROPOSAL.md §4 was low.** It budgeted ~14 GPU-h for 10 cells; 5 cells cost
   20.66 GPU-h billed. Cause: shard stragglers. A few items hit DreamOn's own iteration cap
   (`2*max_gen_len + 2*expand_budget`) at ~2060 forwards / ~200 s while the median item finishes in
   under 1 s, so the slowest shard sets the wall clock (1.3–2.7× the summed compute).

---

## Consequences for A05

* **K1 fired → A05 is dead as proposed.** The claim "an explicit structural runtime reaches quality
  its own model family does not reach at any comparable cost" is refuted: the family member reaches
  the same quality once its canvas is set sensibly.
* **K2 and K3 are moot for the original headline** and should not be run for it. K2 would have
  measured whether a margin survives round noise; there is no margin left to test (−0.05 pp on
  MBPP+). Spending the ~13 GPU-h earmarked for K2+K3 on this claim would be waste.
* **What survives, and it is a different claim**: at *matched quality*, Scaffold Medium is 6.2×/8.2×
  cheaper in forward passes. That is a cost-efficiency finding about a structural decoder, not a
  capability finding, and it is still bounded above by AR on both axes. If anyone wants to pursue it,
  it needs its own proposal and its own gates — it must not be smuggled in under A05's registered claim.
* **D-A is resolved**: the DreamOn baseline *was* harness-crippled, the crippling *was* the canvas,
  and fixing it destroyed the headline rather than confirming it. **D-B and D-C are now irrelevant to
  the original claim** — there is no margin left for a reverse cell or a noise floor to defend.
* `dllm_draft`'s open **STOP** on Scaffold full-program generation should stand, and can now be closed
  with a reason rather than left ambiguous: the internal-point framing has been tested and it failed.

---

## Reproducing this

```bash
# on .73 (zwfy6)
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft
bash a05_k1/a05_k1_run_sweep.sh                    # generation, 8 shards/cell
.venv_dream/bin/python a05_k1/a05_k1_merge_and_grade.py \
    --run-dir runs/a05_k1/mbpp_c32 --dataset mbpp --label mbpp_c32 \
    --out-json runs/a05_k1/_cells/mbpp_c32.json    # merge + assert + grade
.venv_dream/bin/python a05_k1/a05_k1_build_table.py \
    --cells-dir runs/a05_k1/_cells --out-json runs/a05_k1/a05_k1_canvas_sweep.json
```

Code: `proposal/archive/A05-structural-dllm-cost-frontier/code/` (A05 was ARCHIVED 2026-08-12 -- see `A05_CLOSEOUT_VERDICT.md` and `POSTMORTEM.md`; the cost claim recorded below as "surviving" was subsequently falsified).
Evidence: `evidence/a05_k1_canvas_sweep.json` (+ `evidence/cells/*.json` per-cell, with per-item pass
maps and the grader self-test result recorded in each).
