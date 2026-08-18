# Six-review synthesis

## Scores

| Group | Soundness | Excitement | Overall | Confidence | Reproducibility |
|---|---:|---:|---:|---:|---:|
| Strict | 3.33 [3.0, 3.5] | 2.83 [2.5, 3.0] | 3.00 [3.0, 3.0] | 4.50 [4.5, 4.5] | 3.50 [3.5, 3.5] |
| Normal | 3.67 [3.5, 4.0] | 3.00 [3.0, 3.0] | 3.17 [3.0, 3.5] | 4.17 [4.0, 4.5] | 3.67 [3.5, 4.0] |
| All six | 3.50 [3.0, 4.0] | 2.92 [2.5, 3.0] | 3.08 [3.0, 3.5] | 4.33 [4.0, 4.5] | 3.58 [3.5, 4.0] |

## Individual reviews

- `v5_strict_1_GPT56.md` (strict): S=3.5, E=3.0, O=3.0, C=4.5, R=3.5
- `v5_strict_2_GPT56.md` (strict): S=3.5, E=2.5, O=3.0, C=4.5, R=3.5
- `v5_strict_3_GPT56.md` (strict): S=3.0, E=3.0, O=3.0, C=4.5, R=3.5
- `v5_normal_1_GPT56.md` (normal): S=4.0, E=3.0, O=3.5, C=4.5, R=4.0
- `v5_normal_2_GPT56.md` (normal): S=3.5, E=3.0, O=3.0, C=4.0, R=3.5
- `v5_normal_3_GPT56.md` (normal): S=3.5, E=3.0, O=3.0, C=4.0, R=3.5

## Critique extraction note

Consensus critique must be synthesized manually from the six evidence-anchored reports. Treat issues raised by >=3 reviewers as consensus; retain one-review concerns as outliers rather than deleting them.

## Consensus critique (manual synthesis)

### Consensus major issues

1. **No matched nearest reusable-context baseline (6/6).** The internal
   `j=0→12` endpoint is well controlled, but all reviewers still find practical
   significance unresolved without one same-backbone, same-pack, same-hardware,
   same-storage-boundary PIC/chunk-KV/modular-cache comparison. This remains the
   primary main-conference barrier and cannot be fixed by prose alone.
2. **The equal-latency uncertainty model does not respect the heterogeneous
   mixture (6/6).** v5 successfully makes the protocol auditable, which removes
   the v4 reproducibility criticism. However, the CI pools 900 example
   differences IID across nine cells, including 100 LoCoMo items from one
   conversation. Reviewers ask for cell-respecting/hierarchical resampling,
   conversation clustering, task-level dispersion, and leave-one-cell-out
   sensitivity. The point estimates remain valid; the inferential label is too
   strong.
3. **Clean run-level uncertainty is missing for the exact headline adapter
   (6/6).** The added runs confound seed with effective batch and cover reduced
   cohorts, not the exact 15-cell RULER-B/natural-task headline. This limits
   stable claims about the learned interface but does not invalidate the retained
   operating point.

### Consensus secondary issues

4. **Natural-task and repaired end-to-end validation remain limited (3/6).** The
   overlap repair is synthetic-only; natural-task overlap audits are incomplete;
   production-facing p95/concurrency/tails and a repaired end-to-end frontier are
   absent. Keep these results as scoped diagnostics.
5. **Distillation support is under-characterized (2/6).** The top-64 teacher
   support is renormalized, but retained teacher mass was not logged. A focused
   top-k mass/sensitivity audit would strengthen interpretation.
6. **Several local audit inconsistencies should be fixed before the next freeze.**
   Reviewers identified: the abstract's 32k break-even range (it summarizes only
   CPU-pinned values unless relabeled); matched LongEval/LongBench prose numbers
   whose rows are absent from cited tables, including a 12.31 versus displayed
   12.17 ambiguity; an unsupported MemoryLLM prompt-sensitivity assertion; and a
   TurboRAG page-range metadata error. These are editorial/audit issues, not
   reasons to reject the matched core.

### Strength consensus

- v5 fully repaired the prior equal-latency **auditability** problem: cohort,
  calibration, timing boundary, absolute latency, generation limits, under-fill,
  selector asymmetry, and bootstrap unit are all explicit.
- The same-pack/same-adapter `j=0→12` endpoint remains a strong causal internal
  measurement.
- Negative results and selector dependence are reported honestly; CoMem is not
  presented as a universal winner.
- Reproducibility/configuration/statistical documentation is unusually extensive,
  and claim scope is now disciplined.

## Score trend and decision

v5 standardized Overall is **3.08**, versus v4 **3.17**. This is not a clear
manuscript regression: v5 eliminated a consensus major auditability flaw, while
reviewers shifted attention to the deeper experiment-level barriers. More
importantly, the score distribution tightened to **3.0–3.5** (v4: 2.5–3.5), and
strict mean improved from **2.83 to 3.00**. Normal mean fell from 3.50 to 3.17
because two normal reviewers treated the remaining nearest-baseline/statistical
issues more conservatively. The stable interpretation is **strong Findings,
not yet stable ACL main**.

## Next iteration target

1. Fix all local numeric/table/bibliographic inconsistencies without new runs.
2. Recompute equal-latency uncertainty from saved artifacts with a dependence-aware
   analysis (cell-stratified or hierarchical; conversation cluster for LoCoMo;
   leave-one-cell-out), preserving the original pooled CI as a sensitivity view.
3. If new experiments are authorized, prioritize one matched nearest reusable-cache
   baseline and same-batch seed replications of the exact headline adapter.
4. Do not spend the next cycle expanding unrelated benchmarks or model families.
