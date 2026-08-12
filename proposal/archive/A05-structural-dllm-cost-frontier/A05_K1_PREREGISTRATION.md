# A05 K1 -- pre-registered analysis plan

Written **before** any A05 cell was graded. Timestamp fixed by the git commit that
introduces this file; the sweep was launched 2026-08-12 18:09 CST on `.73` and the
first cell had not been scored when this was written.

## The registered rule (from PROPOSAL.md §3, not restated loosely)

> **K1 fires if** DreamOn at its best *non-oracle* canvas setting reaches within
> **5.0 pp** of Scaffold Medium on **both** benchmarks.

Reference, not recomputed: Scaffold Medium HE+ **.177**, MBPP+ **.354**
(`dllm_draft/DLLM_RESULTS_20260807.md:447` / `:456`).

Decision arithmetic, fixed now:

* `gap_pp = (scaffold_plus - dreamon_best_nonoracle_plus) * 100`
* K1 fires iff `gap_pp <= 5.0` on **HE+ AND** MBPP+ simultaneously.
* A negative `gap_pp` (DreamOn above Scaffold) also satisfies `<= 5.0`, i.e. it
  fires. That is intended: it would mean the margin was budget, not structure.
* The oracle cell is excluded from `best_nonoracle`, per invariant 4.
* Metric is `pass@1` on the **plus** test suites, matching how .177/.354 were
  quoted. Base pass@1 is recorded alongside but is not the decision metric.

## My own falsification conditions (added on top of the gate)

These are the ways I can be wrong that the gate alone would not catch. I commit to
reporting any of them as a defect in my own result, not smoothing them over.

1. **Reproduction failure.** `canvas=8` is the archived setting. If it does not land
   near HE+ .122 / MBPP+ .085 (I pre-commit to a tolerance of **±3 pp**, roughly
   1.5x the observed r1→r2 movement of 1.2/1.9 pp), then my harness differs from the
   archive and no other cell in my table is trustworthy against the .177/.354
   reference. I would report the discrepancy and its cause **before** any verdict.
2. **Ceiling not reached.** If pass@1 is still rising monotonically at `canvas=512`,
   then "best non-oracle canvas" is an artifact of where I stopped sweeping, and a
   non-firing K1 is provisional rather than a clean survival. I must say so.
3. **Cost blowup invalidating the comparison.** If the best non-oracle canvas costs
   so much more than Scaffold's 63.8/56.7 mean NFE that the two are no longer at
   "comparable cost", then a non-firing K1 does not rescue the claim -- the claim is
   explicitly *under a token-cost budget*. I must report the cost ratio, not just
   the pp gap.
4. **Clamping.** Canvas is bounded by `max_new_tokens=512`; the oracle arm requests
   `gold+32`, which exceeds 512 for a few HE+ items. Any clamped item must be counted
   and disclosed, because a clamped "oracle" is not an oracle.
5. **Degenerate passes.** If passes at large canvases come from items where the model
   emitted nothing and the HE+ prompt-stitching fallback (`prompt + "    pass"`)
   happened to satisfy the tests, the score is a harness artifact. I check
   `generated_tokens == 0` against the per-item pass set, as MAIN did for the archive.

## Mechanism question, registered separately

Does under-generation go away as the canvas grows? Reported as the trend in
`emitted/gold` median ratio and parseability across canvases, judged against the
roadmap's own continuation criteria: **median ratio >= 0.8** and **parseability >= 90%
on 65+ token spans**. This is reported whether or not K1 fires, per §3's
"Recorded either way".

## What K1 cannot decide

K1 is a statement about the *canvas*, not about A05's headline. A non-firing K1
leaves K2 (round noise) and K3 (one-sidedness) live, and does not by itself make
the +26.9 pp margin publishable. Conversely, if K1 fires, A05 is dead regardless of
what K2/K3 would have said.
