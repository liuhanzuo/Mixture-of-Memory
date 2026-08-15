# SparseForge token-matched ±SLoRB union-9 — CLOSEOUT ADJUDICATION

**Written**: 2026-08-15 ~21:30 CST. **GPU used by this adjudication**: 0 (pure artifact
reading + arithmetic; no ssh, no card touched). **Every number below was re-derived from raw
`results_*.json` / `samples_*.jsonl` / `ppl_metrics.json` on wzc1, not transcribed from a
summary.**

---

## BOTTOM LINE

**NOT reportable as the `+SLoRB` headline it was framed as — and the reason is no longer a
caveat, it is a measured sign reversal.** MAIN's concern was correct and is now settled by data
that did not exist when this task was dispatched: the repairing cell (`slorb` checkpoint exported
`hard_drop`, i.e. variant-matched and genuinely 2:4) **was run today at 21:05–21:10** and is on
disk at `outputs/cast_eval_spec/sparseforge_tokenmatched_slorb/gapfill_union9_summary_hard_drop.json`.
With the export variant held fixed at `hard_drop`, the ±SLoRB contrast **flips from +1.9878 pp to
−2.4515 pp union-9** (paired item-level bootstrap 95% CI [−3.2432, −1.6187], excludes zero).
The originally-reported +1.9878 pp was therefore **not a ±SLoRB effect**: on one and the same
`slorb` checkpoint, changing only the export variant (`hard_drop` → `hard_fold`) moves union-9 by
**+4.4393 pp** [+3.5298, +5.3203], i.e. the dense-fold export accounts for more than twice the
headline. The **honest, variant-matched, 2:4-legal statement is that +SLoRB is 2.45 pp WORSE than
−SLoRB on union-9** at matched tokens (both arms `source_iter=7501`), and the only claim the
`hard_fold` number supports is about a **dense** model that has forfeited 2:4 entirely. A
`+SLoRB`-favourable sentence can be written **only** with the qualifier "when the SLoRB branch is
folded into the weights, producing a dense model that is no longer 2:4 and carries +848,429,056
(+26.2%) live parameters" — which is not the comparison the ±SLoRB training ablation was for.
Reportability: **`with_qualifier`** for the dense-fold row, **`as_is`** for the newly-available
variant-matched −2.45 pp result, and **the +1.99 pp framing must be retracted, not footnoted.**

---

## Reproduced numbers (all three cells, re-derived independently)

Harness string **byte-identical across all three cells**:
`lm_eval 0.4.8, --model hf, dtype=bfloat16, parallelize=True, add_bos_token=False, --batch_size auto, --num_fewshot 0, --seed 0`.
Per-task `n` **identical across all three** (boolq 3270, rte 277, hellaswag 10042, race 1045,
piqa 1838, winogrande 1267, arc_easy 2376, arc_challenge 1172, openbookqa 500), and the
`doc_id` sets in `samples_*.jsonl` are **identical set-for-set across all three cells for all 9
tasks** — so the contrasts below are genuinely paired.

| cell | arm | variant | 2:4? | union9 | cast7 | ast7 | ppl@4096 | ppl@2048 |
|---|---|---|---|---:|---:|---:|---:|---:|
| **A** | noslorb | `hard_drop` | **TRUE 2:4** | **59.5535** | 56.7603 | 60.6531 | **6.679524** | 7.147421 |
| **B** | slorb | `hard_fold` | **DENSE** | **61.5413** | 58.5539 | 62.6299 | **6.193779** | 6.604782 |
| **C** | slorb | `hard_drop` | **TRUE 2:4** | **57.1020** | 55.4004 | 57.5695 | **8.103113** | 8.552948 |

Reference point, same slice definitions, same lm_eval 0.4.8, genuinely 2:4
(`linear_zero_ratio` 0.5, `exact_2of4_tile_ratio` 1.0), same 7500-step budget —
**CAST-repro step7500: union9 62.0919 / cast7 59.2661 / ast7 63.2818, ppl@4096 6.137204**
(`outputs/cast_eval_spec/cast_7500/zeroshot_metrics.json` + `zeroshot_boolq_rte.json` +
`ppl_metrics.json`). Note this is a *different harness build* (`harness_version b86c479`) from the
three cells above, so treat it as context, not as a same-harness row.

### Re-derivation fidelity

I recomputed union9/cast7/ast7 from the raw `results_*.json` by independently applying the
primary-metric map (`acc_norm` for hellaswag/arc_easy/arc_challenge/openbookqa; `acc` for
boolq/rte/race/piqa/winogrande, confirmed against
`baselines/cast_repro/tools/aggregate_zeroshot_union9.py:34-39` for the slice tuples and `:41`
for `PRIMARY_METRIC`, which self-asserts coverage at `:52`):

- All **27 per-task primary values** (9 tasks × 3 cells) match their summaries exactly.
- Cell C: `union9` recomputed − summary = **0.0 exactly**.
- Cells A and B: recomputed − summary = **1.11e-16** (one ULP). **Correction to the Phase-1
  claim**: Phase-1 reported cells A and B as reproducing "bit-exactly ... not merely <1e-9 ...
  exact float equality". That is not reproducible as stated — a plain left-to-right
  `sum(values)/9` differs from the summary by 1 ULP for A and B. This is a floating-point
  summation-order artifact, scientifically irrelevant (1e-16 pp), but "exact float equality"
  is the wrong word and should not be repeated.
- Independently, the mean of the **per-item** `samples_*.jsonl` labels reproduces each headline
  to the same 1-ULP tolerance, confirming the summaries are means over the item files.
- `ppl@4096` / `ppl@2048` re-read from the independent `ppl{2048,4096}/ppl_metrics.json` files:
  exact match to the summaries in all three cells.

### The three contrasts (paired item-level bootstrap, B=4000, seed 0, stratified by task)

The same bootstrap resample is applied to all cells simultaneously, so per-item correlation is
retained. This is the correct unit of analysis; the unpaired stderr-based SE is ~1.8× looser and
is given for comparison.

| contrast | Δ union9 (pp) | paired 95% CI | unpaired SE (σ) | verdict |
|---|---:|---|---:|---|
| **B − A**  `slorb`(fold,DENSE) − `noslorb`(drop,2:4) — *the original headline* | **+1.9878** | [+1.2527, +2.7074] | 0.7218 (2.75σ) | excludes zero, **but confounded** |
| **C − A**  `slorb`(drop,2:4) − `noslorb`(drop,2:4) — **variant-matched ±SLoRB** | **−2.4515** | [−3.2432, −1.6187] | 0.7371 (3.33σ) | excludes zero, **sign REVERSED** |
| **B − C**  `slorb`(fold,DENSE) − `slorb`(drop,2:4) — *same checkpoint, variant only* | **+4.4393** | [+3.5298, +5.3203] | 0.7321 (6.06σ) | excludes zero, **dominates the headline** |

**Noise floor, and how it was computed.** Two floors apply and both are cleared:
1. *Sampling noise* — the paired bootstrap above resamples items within task; all three CIs
   exclude zero. The B−C leg (4.44 pp) is 2.23× the headline (1.99 pp), so the variant axis is
   not a small perturbation on the ±SLoRB axis, it is the larger term.
2. *Cross-architecture noise* — **not applicable here, by construction, and verified.** All three
   cells were scored on `compute_cap 10.0` (sm_100/B200): each `lm_eval.log` records
   `max memory per GPU to {0..3: ~190.84e9}` = 177.7 GiB/GPU, which is B200-class and cannot be
   H20 (95 GB) or a real L20A (48 GB); the gap-fill additionally records `compute_caps: "10.0"`
   from a hard preflight guard (`logs/union9_gapfill_slorb_hard_drop.log`,
   `P4 compute_cap(s)=[10.0] name(s)=[NVIDIA L20A] (name is NOT authoritative)`). Had any cell
   been scored on H20, `status/PAPERB_CORE6_CROSSARCH_FLOOR.md` measures a **0.03–0.16 pp**
   cc10.0-vs-cc9.0 offset on **bit-identical weights** (28 net item flips, symmetric in sign),
   which would sit underneath the effect. Since all three are cc10.0, that floor is 0 here.
   *(That doc carries its own partial revision: the "28 flips" figure was v1-harness on one side;
   matched-harness counts are 7/23/29 per rung. Either way the magnitude is ≤0.16 pp.)*

**Per-task decomposition of the variant-matched contrast (C − A, pp, +ve = +SLoRB better):**

| task | Δ | σ (unpaired) | n |
|---|---:|---:|---:|
| boolq | +1.896 | 1.70 | 3270 |
| rte | **−14.440** | 3.52 | 277 |
| hellaswag | −1.723 | 2.62 | 10042 |
| race | −0.478 | 0.23 | 1045 |
| piqa | +0.000 | 0.00 | 1838 |
| winogrande | −0.947 | 0.50 | 1267 |
| arc_easy | **−5.008** | 3.73 | 2376 |
| arc_challenge | −1.962 | 0.97 | 1172 |
| openbookqa | +0.600 | 0.20 | 500 |

7 of 9 tasks move against +SLoRB. The two largest movers are `rte` (n=277, the smallest task —
so −14.44 pp is only ~40 items and should not be quoted alone) and `arc_easy` (n=2376, −5.01 pp,
3.73σ, the most trustworthy single-task signal in the table).

---

## The sparsity asymmetry, stated plainly

**The two originally-completed arms differed in TWO things at once, not one.** `noslorb` was
exported `hard_drop` and is genuinely 2:4; `slorb` was exported `hard_fold` and is **dense**,
because folding `SLoRB_Weight @ x_proj` into `W` is exact linear algebra that fills the zeroed
positions. So the +1.9878 pp was a **sparse-vs-dense** comparison wearing a ±SLoRB label.

Deciding evidence, three independent measurements agreeing:

| quantity | A `noslorb/hard_drop` | B `slorb/hard_fold` | C `slorb/hard_drop` |
|---|---|---|---|
| `linear_zero_ratio` | **0.500000000** | **1.0809132472221097e-09** | **0.500000000** |
| `exact_2of4_tile_ratio` | **1.0** | **0.0** | **1.0** |
| 2:4 violations (of 1,619,001,344 tiles) | **0** | **1,619,001,344** (all) | **0** |
| `verify_2of4_hf_export.py` VERDICT | **PASS** | **FAIL** | **PASS** |
| `verify_pre_rc` / `verify_post_rc` | 0 / 0 | **2 / 2** | 0 / 0 |
| `2of4_eligible` | `"true"` | **`"false"`** | `"true"` |

B's export log states it outright: `[export] in-scope zero_fraction=0.000000001 exact-2:4
violations=1619001344` and `folded SLoRB into 224 tensors; max |SLoRB_eff| = 2.929961e-02`. Seven
zeros survive out of 6,476,005,376 elements. The dense-ness is confirmed by a *second, separate*
harness: `eval_hf_sparse_model.py` independently reports the same `1.0809132472221097e-09`.

**Why the matrix was asymmetric, and why it is now repaired.** The asymmetry was forced, not
chosen: the `noslorb` checkpoint contains no SLoRB tensors at all, so `--slorb fold` on it
hard-exits at `baselines/cast_repro/tools/export_sparseforge_to_hf.py:181`
(`"--slorb fold requested but {sk}/{xk} missing"`) — the missing cell `noslorb/hard_fold` is
**impossible in principle**, there is nothing to fold. The *other* missing cell,
`slorb/hard_drop`, was always possible and is the minimal sufficient repair; it is the one that
ran today. The exporter also refuses to write a fake result in either direction: `:213` refuses
`hard`+`drop` that is not exactly 2:4, and `:215` refuses a fold that *left* the weight 2:4. So
the PASS/FAIL gates above are real gates, not rubber stamps.

**Capacity, the second confound, which the gap-fill also fixes.** B carries the folded low-rank
branch as extra effective capacity: `SLoRB_Weight` 404,750,336 + `x_proj` 443,678,720 =
**848,429,056 params on top of 3,238,002,688 surviving weights = +26.2%**
(`status/SPARSEFORGE_TOKENMATCHED_PAIR_COMPLETE.md:120-125`). Cells A and C are both
3,238,002,688 surviving weights with the branch discarded, so **C − A is matched on capacity as
well as on variant** — which is precisely why it is the trustworthy contrast.

**Training-side isolation is clean.** Diffing the two arms' actual export-source `args.json`
end-to-end: **exactly 3 keys differ** — `SLoRB` (False vs True), `out_dir`, `resume_dir`. All of
`max_iters=7500`, `seed=1234`, `sparsity_ratio=0.5`, `mask_penalty_mode=nm_2_4`,
`mask_hardening_start=5294`, `srste_decay=0.0`, `global_batch_size=256`, `block_size=4096` are
identical. Both exports read `iter_num=7501 finalization_done=True`. The residual training-side
caveat recorded in `SPARSEFORGE_TOKENMATCHED_PAIR_COMPLETE.md` still stands: the two arms
resumed from different iters (noslorb from 6700, slorb from 6500), so slorb ran 200 extra
iterations of that segment, and this pair cannot separate those 200 iterations from the SLoRB
effect. **Note the direction: that residual asymmetry favours slorb, and slorb still loses by
2.45 pp on the variant-matched contrast.**

---

## MUST NOT CLAIM

1. **Do not claim "+SLoRB improves union-9 by +1.99 pp".** That number is
   dense-fold-vs-true-2:4. Held at matched variant it becomes **−2.45 pp**. This is a retraction,
   not a caveat.
2. **Do not put the `slorb/hard_fold` row in any 2:4 column, table, or sentence.** The run log
   says so verbatim: `logs/sparseforge_tm_union9_slorb_progress.log:314` —
   `** 2:4 COLUMN: BARRED. This arm's export is not exact 2:4 (SLoRB folded). Report it, but
   never in a 2:4 column. **`, and `:288` — `This arm will be scored and reported but is BARRED
   from any 2:4 column.` **Caveat for MAIN**: this bar is currently **advisory only**.
   `2of4_eligible` is a log print + an unread JSON string; it has **zero programmatic
   consumers**. Nothing in the codebase will stop a future aggregator from pulling that row into
   a 2:4 table. If the 2:4 column matters, the bar needs to become an assertion.
3. **Do not claim the ±SLoRB pair is a single-variable comparison** without also stating the
   200-iteration resume asymmetry (noslorb@6700 vs slorb@6500).
4. **Do not describe cell C (`slorb/hard_drop`) as "the deployable +SLoRB model" without
   qualification.** It is a **post-hoc amputation** of a branch the model trained to depend on.
   Its ppl@4096 = 8.103 vs B's 6.194 shows the damage. This is the same error class as
   `baselines/cast_repro/SPARSEFORGE_SAME_HARNESS.md`'s 2026-08-13 "Defect 1", which **retracted
   a headline** built on exactly this confusion, and the gap-fill launcher warns about it at
   `scripts/launch_union9_gapfill_212.sh:71-83`.
5. **Do not use the `sparseforge_5b` / `dolmino_link2` triplets as a "variant-only" control.**
   They are rank-deficient for this purpose: in each triplet all three variants are exports of a
   *single* SLoRB-trained checkpoint, `soft_fold` and `hard_fold` are **both dense**, and only
   `hard_drop` is 2:4 — so the one leg that moves density is the same leg that removes SLoRB, and
   no density coefficient is identifiable. Measured: mask-hardening leg is free (5b
   `soft_fold`→`hard_fold` = 62.5211→62.4335 = **−0.0877 pp**), the whole triplet effect is the
   SLoRB-removal leg (5b `hard_fold`→`hard_drop` = 62.4335→57.0678 = **−5.3656 pp**). Also: when
   quoting link2, keep the metric consistent — mixing link2's `plain_acc` with 5b's
   `union9_primary` produced a bogus ratio in an earlier pass.
6. **Do not claim "a dense model is expected to beat a 50%-sparse model, so this is fine".**
   Empirically false on this disk at matched budget: genuinely-2:4 CAST-repro-7500 scores union9
   **62.0919**, *above* the dense-fold B at **61.5413**.
7. **Do not claim the CAST-repro-7500 row is same-harness with the three cells.** It is
   `harness_version b86c479`; the three cells carry no such field. Cross-harness drift has
   already cost this project a retraction (−0.346 pp AST-7).
8. **Do not quote the `rte` −14.44 pp per-task swing as a standalone finding.** n=277; that is
   ~40 items.
9. **Do not say cells A/B/C "reproduce bit-exactly" or with "exact float equality".** A and B
   differ from their summaries by 1 ULP (1.11e-16) under a plain summation order.

---

## Remaining work order (measured per-cell cost; arch specified)

**All measured costs come from the runs' own stage timestamps, not estimates.** The gap-fill cell
that completed today took **10 m 62 s wall end-to-end on 4 GPUs** (`21:00:24` → `21:11:26`,
`logs/union9_gapfill_slorb_hard_drop.log`), decomposing as: export 2 m 03 s, verify-PRE 50 s, PPL
@4096+@2048 43 s, union-9 zero-shot on 4 GPUs 5 m 07 s, aggregate 2 s, verify-POST 41 s. That is
**≈0.73 GPU-h per cell** (11 min × 4 GPUs). The two original arms measured 9 m 15 s and
10 m 15 s for the identical 6-stage pipeline, so the cost is stable and reproducible.

**Arch requirement for every cell below: B200 / sm_100 (`compute_cap 10.0`) — NOT H20 sm_90.**
Reason, and it is a comparability requirement rather than a capability one: all three existing
cells were scored on cc10.0 (memory fingerprint 190.84e9/GPU, verified above). Scoring a new cell
on H20 sm_90 would add the **0.03–0.16 pp** cross-architecture offset measured in
`status/PAPERB_CORE6_CROSSARCH_FLOOR.md` on **bit-identical** weights, on top of the effect being
estimated — the exact cross-harness/cross-arch error class that already forced one retraction
here. `scripts/launch_union9_gapfill_212.sh` enforces this with `REQUIRE_SM=10.0` and refuses to
proceed otherwise. **The reserved `.212` (8×B200, sm_100, wzc1) is the correct and only home for
these.** All 32 cards on LOCAL/.73/.82/.104 are carrying four 200k-step trainings and must not be
touched.

| # | cell | why | measured cost | arch |
|---|---|---|---|---|
| **R1** | *(none — the blocking cell is DONE)* | `slorb/hard_drop` completed 21:10 today; the variant-matched ±SLoRB contrast now exists. **No GPU is required to fix the headline.** | **0** | — |
| **R2** | `noslorb` + `slorb`, `soft_drop` (2 cells) | The one variant combination never run anywhere on either disk (`find` over wzc1 `outputs/` returns nothing; zwfy6 has no `cast_eval_spec/` directory at all). It separates *mask-hardening* from *SLoRB-removal* on a 2:4-legal export, i.e. it is the only way to confirm on the token-matched pair that the hardening leg is as free as it was in the 5b/link2 triplets (−0.09 pp). Supported by the exporter (`--mask soft --slorb drop`). **Optional / robustness, not blocking.** | **≈0.73 GPU-h each → ≈1.5 GPU-h total** | **B200 sm_100** — must match the 3 existing cells |
| **R3** | 2nd seed of the token-matched pair (±SLoRB), then re-export both `hard_drop` | The −2.45 pp is n=1 run per arm. There is **no same-arch run-to-run replicate anywhere** for these arms, so run-level (as opposed to item-level) variance is unmeasured — see "what I did NOT verify". This is *training*, not eval: cost is the 7500-step SparseForge run, **not** 0.73 GPU-h. | **eval 2 × 0.73 = 1.5 GPU-h; training cost NOT measured by me** | **B200 sm_100** for the evals; training node per SparseForge ownership |
| **R4** | make the 2:4 bar programmatic (0 GPU) | `2of4_eligible` has no consumers; add an assertion in the aggregator/table builder so a dense row cannot silently enter a 2:4 column. | **0 GPU** | n/a — CPU |
| **R5** | retract/rewrite the +1.99 pp framing wherever it appears (0 GPU) | `status/SPARSEFORGE_TOKENMATCHED_PAIR_COMPLETE.md:~95-110` presents the +0.0199 union-9 delta as the ±SLoRB result. It now has a variant-matched refutation. **I did not edit it** (out of scope per instructions); MAIN should. | **0 GPU** | n/a — CPU |

**Priority**: R5 and R4 are free and should happen before anything is written up. R2 is cheap
robustness. R3 is the only genuinely expensive item and is only needed if a run-level variance
claim is wanted.

---

## What I did NOT verify

1. **Run-to-run (same-arch, same-weights) variance for these arms.** No replicate exists. The
   bootstrap CIs quantify **item sampling** noise only. `memory/same-harness-runs-bit-identical.md`
   records that same-arch/same-harness re-runs have been byte-identical elsewhere in this project,
   which *suggests* run-level jitter ≈ 0 — but I did not measure it here and do not assert it.
2. **The physical node identity of cells A and B.** I verified their `compute_cap` **indirectly**,
   via the 190.84e9-bytes/GPU memory fingerprint in their `lm_eval.log`. Only cell C records
   `compute_caps: "10.0"` explicitly. I did not (and could not, at 0 GPU) confirm which host ran
   A or B. `SPARSEFORGE_TOKENMATCHED_PAIR_COMPLETE.md` says LOCAL and `.212` respectively, both
   sm_100 — consistent, but that is a ledger claim, not my measurement.
3. **Bit-identity of the two `slorb` HF exports' shared inputs.** I confirmed both cells B and C
   name the same source `model.pt` and both report `iter_num=7501`, and the gap-fill preflight
   asserts `ckpt bytes=41078402630 (matches the arm scored on 2026-08-15)`. I did **not** hash the
   checkpoint myself.
4. **Whether SLoRB is deployable as "2:4 weight + separate low-rank branch at inference".** This
   is the scientifically interesting question the artifacts cannot answer. I found that
   `deploy_sparse_24/convert.py:186-190` lists `.SLoRB_Weight` and `.x_proj` under
   "训练辅助张量，不需要导出" (training auxiliary tensors, not exported) — i.e. the deployment
   path **drops** the branch, matching cell C, not cell B. I did **not** trace the training-time
   forward pass to confirm the branch is a genuine parallel path (which would make a
   sparse-weight + dense-branch deployment conceivable) versus a reparameterisation. **Do not
   claim SLoRB "cannot" be deployed with 2:4 on my authority** — I only verified that the two
   exporters on disk either fold it (destroying 2:4) or discard it.
5. **The 200-extra-iteration effect size.** I confirmed the resume asymmetry exists from
   `args.json`; I did not quantify what 200 iterations at that point in the schedule is worth.
6. **zwfy6 for these artifacts.** I checked and
   `/apdcephfs_zwfy6/share_304376610/pighzliu_code/outputs/cast_eval_spec/` **does not exist**,
   so all SparseForge union-9 evidence is wzc1-resident. This satisfies the two-disk rule for the
   "no `soft_drop` cell exists" claim, but I did not search zwfy6 subtrees other than that path.
7. **`sparseforge_5b` / `dolmino_link2` per-item data.** My triplet-leg numbers came from their
   summary JSONs (`sparseforge_same_harness_table.json` headline block, `link2_summary.json`),
   which I did not re-derive from raw results the way I did for cells A/B/C.
8. **Novelty / prior-art status** of any claim here. Out of scope.
