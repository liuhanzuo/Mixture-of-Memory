# Table 4's "all rungs ran to 200k" claim is FALSE — keep8/keep10/keep12 never reached 200k

**Date**: 2026-08-08 CST. **Found by**: agent `a50df6cdeaf7b6a1d`, refusing MAIN's dispatch premise.
**Verified by**: MAIN, independently, by listing checkpoints on both disks and matching Table 4's
quoted core6 values back to their source directories. **GPU cost**: 0.

## The false claim

`paperB/TODOList.md:267` states:

> 所有深度阶梯 arm 均已完整运行至 200k step，论文 Table 4 的 Budget 列统一为 200k。
> ("All depth-ladder arms ran fully to 200k step; Table 4's Budget column is uniformly 200k.")

**Three of the five rungs never reached 200k.** Highest checkpoint actually on disk (identical on
wzc1 and zwfy6):

| rung | max step on disk | Table 4 claims |
|---|---:|---:|
| keep8 | **121,000** | 200k |
| keep10 | **83,500** | 200k |
| keep12 | **124,000** | 200k |
| keep14 | 200,000 | 200k ✅ |
| ShortGPT-16 | 200,000 | 200k ✅ |

keep10 reached only **42%** of the claimed budget.

## Which numbers Table 4 actually quotes

I matched each Table 4 core6 value back to its source directory by recomputing core6
(`acc_norm` ×5 + WinoGrande `acc`) over every candidate and finding the ones that round to the
quoted figure:

| rung | Table 4 core6 | matching source dir | recomputed | real step |
|---|---:|---|---:|---:|
| keep10 | `.5303` | `7B_keep10_step83500` | .53029 | **83,500** |
| keep12 | `.5669` | `7B_keep12_step124000` | .56694 | **124,000** |
| ShortGPT-16 | `.6215` | `7B_shortgpt16_step200000` | .62149 | 200,000 |

(keep10 also matches `step80000`/`step81500` and keep12 matches several neighbours within 6e-4 —
the trajectory is flat near its end, so core6 alone cannot pin the exact step. The **step ceiling**
is the load-bearing fact and it is unambiguous: no 200k checkpoint exists for these arms.)

## Why this matters — it is a compute-confound, not a typo

Table 4 is the paper's **depth ladder**, and its entire argument is that PPL tax and MMLU
degradation increase monotonically as you prune deeper. That comparison is only clean if every rung
got **the same healing budget**. It did not:

- keep14 (best rung) got 200k steps.
- keep10 (a middling rung) got 83.5k — **2.4× less healing**.

So part of the measured gap between rungs is *unequal healing compute*, not depth. The direction of
the bias is unfavourable: the shallower rungs that look worst also got **less** healing, which
inflates the apparent depth effect. The paper cannot claim "compute-matched" for this table as it
stands.

This is separate from, and larger than, the cross-architecture core6 floor documented in
`PAPERB_CORE6_CROSSARCH_FLOOR.md` (0.03–0.16 pp). A 2.4× budget difference is a first-order
confound.

## How the error was caught

MAIN's dispatch brief told the agent to eval `step200000.pt` for all four rungs. The agent checked,
found the file absent on **both** disks, and **refused the premise** rather than substituting
silently or failing. It proceeded with the true headline checkpoints and encoded the real step in
each output name (`7B_keep8_step121000_v2`, `7B_keep10_step83500_v2`, `7B_keep12_step124000_v2`,
`7B_shortgpt16_step200000_v2`).

Had it silently used whatever `step200000.pt` resolved to, or fabricated the path, this would have
surfaced as an unexplained eval failure or — worse — quietly evaluated the wrong checkpoints.
**Second time tonight that an agent refusing MAIN's stated premise was the right call** (the first
was the `final.pt` symlink, `PAPERB_P24_SHORTGPT16_ARM.md`).

## Required fixes, in priority order

1. **Correct `paperB/TODOList.md:267`** and every Table 4 caption / Budget column that says 200k.
   Report the real per-rung step. (Not done here — MAIN owns the `.tex`; and #189's audit should land
   first so all corrections go in one pass.)
2. **Decide the honest framing.** Two options, and the choice is the user's:
   - (a) Report the true steps and drop any "compute-matched" language; add the budget as a column so
     a reader sees keep10 got 83.5k. Cheapest, fully honest, weakens the ladder claim.
   - (b) Resume keep8/keep10/keep12 to a genuine 200k and re-eval. Costs GPU-weeks
     (keep10 needs 116.5k more steps) but makes the compute-matched claim true.
3. **Check whether the PPL / MMLU / aux5_raw columns carry the same defect** — they come from the
   same batteries, so they are quoting the same non-200k checkpoints. Fold into #189.
4. Note that P0.7's audit (`paperB/P0_7_AGGREGATE_AUDIT.md`) apparently already recorded these as the
   "headline steps" — so the numbers themselves are the intended ones. The defect is the **Budget
   column and the 200k claim**, not the measurements.

## Provenance

- Checkpoint listings: `outputs/olmo2_probe2_7B_keep{8,10,12}fresh2/`, `outputs/olmo2_probe2_7B_{keep14fresh2,shortgpt16}/` on wzc1 and zwfy6.
- core6 recomputation over all `zwfy6:olmo2_downstream_results/7B_*/summary.json`.
- False claim: `paperB/TODOList.md:267`.
- Agent report: `status/PAPERB_P24_LADDER_PREV2_EVAL.md`; driver `scripts/_run_olmo2_p24_eval_ladder_prev2_73.sh`, commits `6c3e329`, `82feb86`.
