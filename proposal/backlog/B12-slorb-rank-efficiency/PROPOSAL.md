# B12 — Is SparseForge's SLoRB branch over-provisioned?

**Status**: BACKLOG. Novelty NOT checked. `RELATED_WORK.md` NOT on disk. **0 GPU spent so far.**
**Created**: 2026-08-16. **Disk**: wzc1-only (see §8).

---

## 1. The question, in the user's own terms

> "SLoRB's effect doesn't feel like it needs to be this big — try a SMALLER SLoRB."

SparseForge deploys a genuinely 2:4-sparse `W` **plus** an additive low-rank bypass called SLoRB:

```
out = masked_linear(x, W, mask) + (x @ x_proj.T) @ SLoRB_Weight.T
```
(`baselines/cast_repro/tools/probe_slorb_branch.py:8`; forward at `sparse_modeling.py:818-822`,
so the effective added operator is `E = SLoRB_Weight @ x_proj`, shape `(out, in)`.)

That branch is **not free**. MEASURED on the 5B headline checkpoint
(`/apdcephfs_wzc1/share_304376610/pighzliu_code/outputs/cast_eval_spec/sparseforge_5b/sparseforge_same_harness_table.json`,
key `source_checkpoint.slorb_branch`; every count below re-derived independently from the tensor
shapes in this pass and reproduced **exactly**):

| quantity | value |
|---|---|
| `SLoRB_Weight` elements | 404,750,336 |
| `x_proj` elements | 443,678,720 |
| extra live params vs pure 2:4 | **848,429,056** |
| surviving 2:4 weights | 3,238,002,688 |
| in-scope elements | 6,476,005,376 |
| extra as % of surviving weights | **26.2 %** |

So the branch costs **13.1011 density points** (63.1011 % → 50.0 %). The question is whether it has
to cost that much. **This needs no training**: the branch is a low-rank product already on disk, so
a smaller branch can be *constructed* from the trained checkpoint and evaluated.

---

## 2. What the premise got right, and three things it got wrong

Everything numeric in the brief reproduces. Three framing errors were found and are repaired here.
They are recorded because each one, uncorrected, would have produced a wrong headline.

### 2.1 CORRECTION — "SLoRB is worth +5.37 pp" is the *export-variant* axis, not the SLoRB axis

`62.4335 − 57.0678 = 5.3657 pp` is arithmetically right (verified), but the two rows are
`sparseforge_hard_fold` and `sparseforge_hard_drop`. Per `status/sparseforge_union9_closeout.json`:

* `hard_fold` has `linear_zero_ratio = 1.0809132472221097e-09`, `exact_2of4_tile_ratio = 0.0`,
  `exact_2of4_violations = 1,619,001,344` (**all** tiles), `zeros_surviving = 7` out of
  6,476,005,376. **It is a DENSE model.** `barred_from_2of4_column: true`.
* On the token-matched pair, the *same-checkpoint, variant-only* contrast
  (`B_minus_C_SAME_CKPT_VARIANT_ONLY`) is **+4.4393 pp [3.5298, 5.3203]** — i.e. `2.23×` the
  entire originally-reported ±SLoRB effect — and the honest variant-matched ±SLoRB contrast
  (`C_minus_A_VARIANT_MATCHED`) is **−2.4515 pp [−3.2432, −1.6187]**: **+SLoRB is WORSE**.
  The `+1.99 pp` headline is already formally RETRACTED in that file.

**Consequence for B12.** The 5.3657 pp window is still the correct *y*-axis span for this ladder,
because every rung here starts from the same 5B `hard_fold` model and removes branch capacity from
it — R0 and R1 are the two endpoints of exactly the transformation we are titrating. But the window
must be labelled **"folded-dense → true-2:4"**, never "the value of SLoRB". A rung that recovers
most of it has recovered most of *the fold*, and the fold is dense.

### 2.2 CORRECTION — "x_proj carries <6 % of the energy" is a SQUARED fraction; the real perturbation is ~24 %

`frac_energy_from_learned_delta` (min 0.0114 / median 0.0188 / max 0.0585) is a squared-Frobenius
share. The quantity that matters is the *relative error*, its square root, and it is in the same
file: `rel_fro_err_if_xproj_reverted_to_init` = **min 0.1067 / median 0.1368 / max 0.2418** over the
12 sampled projections. "<6 %" understates the branch-level damage by ~4×.

This does **not** kill rung A, because what the model sees is not `E` but `W_eff = W*mask + E`.
Computed exactly over **all 224 tensors** (`branch_importance_all224.json` × `global_operator_basis_all224.json`,
no sampling) in this pass:

| operator | global `‖ΔW_eff‖_F / ‖W_eff‖_F` | as % of deleting the branch |
|---|---|---|
| delete branch (= R1) | 0.200010 | 100 % |
| blocksum, **LS-refit** coefficients | **0.024523** | **12.26 %** |
| blocksum, naive revert (keep `SLoRB_Weight` as-is) | 0.025603 | 12.80 % |

So rung A is still the cheap rung — but state it as **12.3 % of the delete-branch perturbation**
(that is a `W_eff` fact), not as "<6 % of the energy" (a squared branch-level fact).
The LS refit is 4.22 % better than the naive revert and costs nothing, so **the protocol uses LS**.

### 2.3 CORRECTION — the "SVD is the weak mode" justification is refuted by the repo's own density-matched file

The brief's central argument for basis-coarsening over SVD is that phase 1 measured the spectrum of
`E` to be flat (**this replicates**: `r90/r90_null` median 0.995, ≥0.95 in 32/35 projections, only
L0.q_proj and L0.k_proj concentrated). And its arithmetic that half-rank SVD (424,214,528 params)
costs 4.81 % *more* than the full-rank blocksum branch (404,750,336) is **correct** — verified.

But `outputs/slorb_rank/op_matched_density_sample.json` already compares the two operators **at
matched parameter budget** (`svd_t` chosen so `(out+in)·t ≈ out·r_eff`; I verified the match holds
to <2 % on 160/175 cells, the 15 exceptions being `ke256` on the 11008-wide projections). At that
matched budget, on the deployed `W_eff`:

| `k_eff` | coarsen median pert | SVD median pert | SVD wins |
|---|---|---|---|
| 16 (c=1) | 0.0242 | 0.1217 | 1/35 |
| 32 (c=2) | 0.1512 | 0.1625 | 2/35 |
| 64 (c=4) | 0.1841 | 0.1828 | **12/35** |
| 128 (c=8) | 0.1987 | 0.1944 | **15/35** |
| 256 (c=16) | 0.2055 | 0.2015 | **24/35** |

**SVD is dominated only at c=1–2. From c=4 down it is a coin flip, and by c=16 it WINS 24/35.**
Mechanism: coarsening saturates (0.1841 → 0.2055, i.e. it hits the delete-branch ceiling of 0.2000
almost immediately) whereas SVD degrades gracefully. So `Dctl` **cannot** be pre-registered as
"deliberately included to be beaten". It is a genuine competitor at aggressive compression, and this
ladder is designed to stop *before* the regime where SVD takes over — which is a further reason the
rungs below are shallow (§4).

---

## 3. The measured spectrum evidence (why "shrink the rank" is the wrong knob)

`outputs/slorb_rank/slorb_rank_5b_headline.json` — exact `svdvals` of `E = SLoRB_Weight @ x_proj`
in float64 via QR of both factors (an `r×r` core; the identity
`svdvals(S@B) == sqrt(k)·svdvals(S)` holds to 1.5e-15, an independent correctness check on the QR
path). 35 projections = layers {0,7,15,23,31} × 7 projections.

* **`r` is not a free hyperparameter.** `r = in_features // SLoRB_k` with `SLoRB_k = 16` MEASURED
  from the checkpoint's own `args` (`SLoRB_k=16`, `SLoRB=true`, `SLoRB_init_type='sum'`,
  `trainable_projection=true`); `sparse_modeling.py:415-427`. Every projection is already at its
  maximum rank for `k=16`: `q/k/v/o` → 256, `gate/up` → 256, `down` → 688. **The only knob the
  trainer exposes for "a smaller SLoRB" is `k`.**
* **The spectrum is flat.** `r90/r` median 0.836; vs a Gaussian product null of identical shapes,
  `r90(measured)/r90(null)` median **0.995**, ≥0.95 in **32/35**. Stable rank median 83.2 (r=256
  family) against a null of 154.9. Only L0.q_proj (0.400) and L0.k_proj (0.237) are genuinely
  low-rank. **Asking SVD to exploit concentration measured not to exist is the wrong move for
  most of the model** — which is what §2.3's c=1–2 columns confirm.
* **Block structure DOES exist and is exact.** `x_proj` on-block entries are **exactly 1.0** in
  12/12 sampled projections (`x_proj_on_block_n_exactly_one == n_total`, min = max = 1.0), and
  off-block mass is tiny (`meanabs` 0.0030–0.0061). The trained `x_proj` is its block-sum init `B`
  plus a small learned delta. So `x @ B^T` is a **segment-sum** — it needs **zero stored
  parameters**. That is where the 443,678,720 params go.

---

## 4. Chosen protocol — basis coarsening (`SLoRB_k`), asymmetric, with an LS refit

**Operator** (all from the trained checkpoint; **no training**):

1. **blocksum-LS**: replace trained `x_proj` by its exact init `B` (`B[i, i·k:(i+1)·k] = 1`,
   `sparse_modeling.py:883-886`) and **re-fit `SLoRB_Weight` by least squares in the new basis**.
   `x_proj` becomes parameter-free. Deletes 443,678,720 params.
2. **coarsen**: additionally `k → k·c`, which is *exactly* the branch you would have obtained by
   training with `SLoRB_k = k·c`. Rank `r → r/c`. The LS-optimal coefficient for the coarser basis
   is the block **mean** of the `c` adjacent columns (**not** the sum — summing inflates scale by `c`).
3. **asymmetric**: a **different `c` per projection family**. This is the repair that makes the
   ladder efficient, and it is the one part of the brief's design I am keeping and extending.

**Why asymmetric.** Per-family cost and damage are strongly non-uniform (measured, all 224 tensors):
`down_proj` holds 22.28 % of `SLoRB_Weight` **and** has the *lowest* `relE/W_eff` (median 0.1546)
and the lowest coarsening perturbation at every `c` (0.1122 at c=2 vs 0.1533 for `q_proj`).
`o_proj` is next (0.1806 / 0.1256). So coarsening `down` and `o` buys density far more cheaply than
coarsening `q`/`k`. An exhaustive scan of all `5^7 = 78,125` coarsening maps over
`c ∈ {1,2,4,8,16}` (scoring global `‖ΔW_eff‖_F/‖W_eff‖_F` with per-family medians weighted by
per-family `‖W_eff‖²_F`) confirms the frontier **never** contains a uniform map above ψ≈0.52:
at ψ≥0.76 the uniform `c=2` ladder costs 0.1423 where the frontier costs 0.1241 (**+14.7 %
unnecessary damage**), and at ψ≥0.88 uniform `c=4` costs 0.1734 vs 0.1623.

### 4.1 The ladder

ψ ("psi") = fraction of the branch's 13.1011 density points **given back**;
ψ = (63.1011 − density)/(63.1011 − 50.0). R0 → ψ=0, R1 → ψ=1.
Perturbation column = global `‖ΔW_eff‖_F/‖W_eff‖_F`, and `%DEL` is it as a fraction of
deleting the branch outright (0.200010).

| rung | `c` per (q,k,v,o,gate,up,down) | branch_live | density % | ψ | pert | %DEL |
|---|---|---|---|---|---|---|
| **R0** ANCHOR `hard_fold` (ON DISK, 0 GPU) | — | 848,429,056 | 63.1011 | 0.0000 | 0.0000 | 0 |
| **A** blocksum-LS | 1,1,1,1,1,1,1 | 404,750,336 | 56.2500 | 0.5229 | 0.0245 | 12.3 |
| **P** ★ **PILOT** | 1,1,1,1,1,1,**8** | 325,844,992 | 55.0316 | 0.6159 | 0.0676 | 33.8 |
| **Q** | 1,1,1,**8**,1,1,**16** | 290,848,768 | 54.4912 | 0.6572 | 0.0802 | 40.1 |
| **R** | 1,1,**4**,**8**,1,1,**16** | 265,682,944 | 54.1026 | 0.6869 | 0.0913 | 45.7 |
| **S** | 1,1,**16**,**16**,1,**4**,**16** | 189,661,184 | 52.9287 | 0.7765 | 0.1241 | 62.0 |
| **Dctl** density-matched SVD at P | `t`=128,128,128,128,187,187,23 | 326,098,944 | 55.0355 | 0.6156 | 0.1170 | 58.5 |
| **R1** ANCHOR `hard_drop` (ON DISK, 0 GPU) | — | 0 | 50.0000 | 1.0000 | 0.2000 | 100 |

`Dctl` is matched to **P** to 0.0039 density points (0.08 % of params) — a genuine
operator-vs-operator comparison at fixed density, replacing the brief's `t=r/2` control which was
*dominated on both axes* by rung A and therefore uninformative.

**PILOT = rung P** (`down_proj` at `c=8`, everything else `c=1`). Rationale: it is the single
cheapest ψ-per-perturbation step on the whole frontier (ψ 0.523 → 0.616 for pert 0.0245 → 0.0676,
still only ⅓ of the delete-branch damage), it exercises **both** new code paths (LS refit **and**
per-projection coarsening) in one export, and if it fails, the ladder is dead at its cheapest
point and rungs Q/R/S are never launched.

### 4.2 Held fixed

* **checkpoint**: ONE file, `out_llama/models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260413_201320/model_best_lm_eval.pt`,
  `iter_num = 17900`.
* **mask**: `nm_2_4` hard, exact top-2-per-4, computed once from the same soft mask for every rung,
  never re-derived per rung. `emit_small_slorb_variants.py:79-92` is a line-for-line port of
  `export_sparseforge_to_hf.py:83-104`.
* **`W`**: untouched. Only the additive `E` changes. `W_eff = W*mask + E` for every rung
  including both anchors.
* **harness**: lm_eval **0.4.8** + transformers **4.57.6** + torch 2.13.x, asserted
  programmatically (the driver refuses to add a row otherwise). lm_eval 0.4.8 has **no**
  `__version__`; use `importlib.metadata.version`.
* **model args**: `dtype=bfloat16`, `parallelize=True`, `add_bos_token=False`, `--num_fewshot 0`,
  `--seed 0`, `--batch_size auto`.
* **item set**: union-9 with `n =` boolq 3270 / rte 277 / hellaswag 10042 / race 1045 / piqa 1838 /
  winogrande 1267 / arc_easy 2376 / arc_challenge 1172 / openbookqa 500. Primary metric =
  `acc_norm` for hellaswag/arc_easy/arc_challenge/openbookqa, `acc` for the other five
  (`aggregate_zeroshot_union9.py:41`).
* **arch**: `sm_100` / B200 ONLY, `REQUIRE_SM=10.0`. All rows in the 5B table were scored on
  compute_cap 10.0; H20 `sm_90` would inject the 0.03–0.16 pp cross-arch term.

### 4.3 Density axis — the honesty requirement

Every rung is **FOLDED** (`W*mask + E`) and therefore **DENSE ON DISK**. That is forced: lm_eval
only reads standard HF weights, and folding is exact algebra
(`export_sparseforge_to_hf.py:62,:123,:283`). The density column is the **two-matmul DEPLOYMENT**
count. **Any table must carry that distinction in the row, not a footnote** — reporting a folded
export in a 2:4 column is precisely the defect that already forced a retraction on this project
(`status/sparseforge_union9_closeout.json.must_not_claim[1]`).

No SLoRB rung can go below 50.0 % — that is pure 2:4 with zero branch.

---

## 5. Pre-registered threshold, with its noise floor

**Noise floor** (MEASURED, not assumed): paired item-level bootstrap within task, B=4000, seed 0,
same resample applied to all cells simultaneously. The three completed contrasts have paired CI95
half-widths of 0.7274 / 0.8123 / 0.8953 pp. **Floor = 0.8953 pp** (the widest).
`τ = 2 × 0.8953 ≈ 1.79 pp`.

Context: `τ = 1.79 pp` is **33.4 %** of the whole R0→R1 window (5.3657 pp). This ladder is
`n=1 run per rung`, so a rung inside `(τ, 2τ)` is 1–2 CI widths and **must not be called either way**.

**Pass bar on union-9 primary**: `score ≥ 62.4335 − 1.79 = 60.64`.

**Qualifying rungs** = ψ ≥ 0.60 (gives back ≥ 60 % of the 13.1011 density points):
**P (0.616), Q (0.657), R (0.687), S (0.777)**. Rung A (ψ=0.523) does **not** qualify.

| verdict | condition |
|---|---|
| **OVER-PROVISIONED** | at least one qualifying rung scores ≥ 60.64 |
| **NOT over-provisioned** | every qualifying rung loses > τ **AND** loss is monotone in ψ |
| **INDETERMINATE** | a qualifying rung lands in the (1.79, 3.58) pp loss band |

### 5.1 Pre-registered third outcome (recorded NOW so it cannot be reframed post hoc)

§2.2 and §3 make this **likely**: **rung A passing while P/Q/R/S all fail is NOT "over-provisioned
rank".** It is **"`x_proj` is free, rank is not"**. That is still a real result — 443,678,720
parameters deleted for a global `W_eff` perturbation of 12.3 % of the delete-branch level — but it
must be reported as an **`x_proj`/basis result**, and rung A's ψ=0.523 is below the qualifying bar
by construction so it cannot be laundered into a rank claim.

### 5.2 Pre-registered fourth outcome — the operator could lose to its own control

§2.3 shows SVD wins 12/35 at c=4 and 24/35 at c=16 on `W_eff` perturbation. If **Dctl beats P**,
the conclusion is **"basis coarsening is the wrong operator beyond c≈2"**, not "SLoRB is
over-provisioned". Recorded now because the brief pre-registered Dctl as a foregone loss.

### 5.3 THE PARETO CLAIM IS UNREACHABLE BY CONSTRUCTION — do not attempt it

CAST-repro pure 2:4 scores **62.0919 at density exactly 50.0 %** — the hard floor, *below every
rung in this ladder*, and only 0.3415 pp under R0 itself. **No SLoRB rung can dominate it**: every
rung has strictly higher density and would need to beat 62.0919 to win on quality. R0, with the
entire branch and 13.1 extra density points, only manages +0.3415 pp — inside τ. So
"SparseForge beats CAST at lower density" is **not on the table** and must not be claimed from
this ladder regardless of outcome.

Caveat on that comparison: the closeout warns CAST-repro is `harness_version b86c479` and *not*
same-harness with the token-matched cells A/B/C. **Within the 5B table used here it is same-harness**
(`harness.note: "byte-identical across all seven rows; only pretrained differs"`), so the
comparison above is legitimate — but the `b86c479` commit **is not present in this checkout**
(`git merge-base --is-ancestor` → `fatal: Not a valid object name`), so the harness build cannot be
re-verified from git and must be treated as an unverified provenance pointer.

---

## 6. Falsification condition

The direction dies if **either** holds:

1. **PILOT rung P loses more than τ = 1.79 pp** (score < 60.64) — the cheapest step on the measured
   frontier already fails, so no deeper rung can succeed; **stop, do not launch Q/R/S**. Cost to
   reach this verdict: **1.46 GPU-h** (P + Dctl).
2. **`Dctl` ≥ P** — basis coarsening is not the right operator, and the ladder as designed is
   answering the wrong question.

Additional stop rule: if **rung A itself** loses > τ, then even the parameter-free `x_proj`
substitution is unaffordable and every claim in §5.1 is void too.

---

## 7. MUST-NOT-CLAIM

1. **A post-hoc truncation of a trained branch is NOT a training-time rank ablation.** Every rung
   here is surgery on `model_best_lm_eval.pt` at `iter_num 17900`. A model *trained from scratch*
   with `SLoRB_k = 128` could reallocate capacity during training and land anywhere — plausibly
   *better* than this ladder's rung at the same density. **This experiment cannot bound that.**
   The claim it can support is: *"the trained branch contains this much redundancy, extractable
   without retraining."* It may **never** be phrased as "SLoRB only needs rank r", "SparseForge
   should be trained with `SLoRB_k = ...`", or any statement about the training-time design.
2. **Never** call the 5.3657 pp window "the value of SLoRB". It is folded-dense vs true-2:4. The
   honest variant-matched ±SLoRB contrast is **−2.4515 pp** (+SLoRB WORSE).
3. **Never** place any rung, or R0, in a 2:4 column, table, or sentence. All are folded and dense
   on disk (R0's `linear_zero_ratio = 1.08e-9`; the bar at
   `logs/sparseforge_tm_union9_slorb_progress.log:314` is advisory-only and has **zero**
   programmatic enforcement, so this is a human obligation).
4. **Never** claim any rung is Pareto-superior to CAST-repro pure 2:4 (§5.3).
5. **Never** report `frac_energy_from_learned_delta` (<6 %) as if it were a perturbation. It is a
   squared quantity; the relative error is ~13.7 % median at branch level, 12.3 %-of-delete at
   `W_eff` level.
6. **Never** claim the spectrum of `E` is concentrated, or that SVD "does not work". Flat vs the
   null in 32/35 (so SVD cannot exploit concentration), **but** SVD wins the density-matched
   `W_eff` comparison in 12/35 at c=4 and 24/35 at c=16.
7. **Never** call `Dctl` a control that was expected to lose (§5.2).
8. **Never** report a rung as a measurement without stating that it is `n=1` and that
   τ is 33.4 % of the entire available window.
9. **Never** claim these rungs are "what training with `SLoRB_k=k·c` would give". The *basis* is
   identical; the *coefficients* are an LS refit of a `k=16`-trained tensor, not trained in the
   coarse basis.
10. **Never** describe the differential-LR / node story from the SparseForge trainer here — no
    training happens in B12 at all.

---

## 8. Provenance and disk facts

* **The brief's path `outputs/cast_eval_spec/...` is NOT in this repo.** It resolves against
  `/apdcephfs_wzc1/share_304376610/pighzliu_code/` (one level above the checkout). Verified:
  `/apdcephfs_wzc1/share_304376610/pighzliu_code/outputs/cast_eval_spec/sparseforge_5b/sparseforge_same_harness_table.json` exists.
* **wzc1-only.** `/apdcephfs_zwfy6/share_304376610/pighzliu_code/outputs/cast_eval_spec` does
  **not** exist, nor does `.../Mixture-of-Memory/outputs/cast_eval_spec` on either disk. All
  SparseForge union-9 evidence and the 41 GB checkpoint are wzc1-resident → **LOCAL or .212 only**.
* Spectrum/structure evidence: `outputs/slorb_rank/{slorb_rank_5b_headline,slorb_xproj_struct_5b,
  branch_importance_all224,global_operator_basis_all224,block_energies_all224,
  op_matched_density_sample}.json` (in-repo).
* **Measured cell cost**: 0.73 GPU-h, from the completed gap-fill's own stage timestamps
  (11m02s wall on 4 GPUs; the two prior arms took 9m15s and 10m15s for the identical 6-stage
  pipeline). **Full ladder = 6 cells × 0.73 = 4.38 GPU-h. Pilot-only decision = 1.46 GPU-h.**

### 8.1 A defect in the tool this ladder depends on

`emit_small_slorb_variants.py` was **written but never executed** (its own docstring says so). It
does `torch.save(new_sd, out/"pytorch_model.bin")` — whereas the working exporter
(`export_sparseforge_to_hf.py:258`) does `model.save_pretrained(out, safe_serialization=True)`
**after** a `load_state_dict` round-trip against the dense reference's
`model.safetensors.index.json` key set (`:217-235`) and a post-cast 2:4 re-verification
(`:238-253`). A bare `torch.save` of a hand-built dict skips all of that, so a key-naming or
dtype error would surface as a silently mis-scored eval. **The new tool
(`baselines/cast_repro/tools/emit_slorb_ladder.py`) therefore reuses the exporter's verified
save path and key-set assertion, and is opt-in via new flags rather than a rewrite of the
existing exporter.**

---

## 9. GENERALISATION LEG — Qwen3.5-9B (QUEUED, **not** ready)

Same direction, different family: does the redundancy finding transfer off Llama-2? **This leg is
QUEUED and must not be launched with the Llama-2 ladder.** It is not a ladder rung; it needs a
new SparseForge *training* run before any rung exists, so it is a different order of cost.

**Asset** (MEASURED): `/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen3.5-9B`,
4 shards, 19,306,310,880 bytes on disk (index declares 19,306,216,416), 775 tensors.
**wzc1-ONLY** — `ls -d /apdcephfs_zwfy6/.../models/*Qwen3.5*` returns nothing.
(~27 min to `scp -O` at the measured 12 MB/s single-stream if zwfy6 is ever needed.)

**Blockers, in order of severity:**

1. **IT IS MULTIMODAL.** `architectures: ["Qwen3_5ForConditionalGeneration"]`,
   `model_type: qwen3_5`, has a `vision_config` (27-layer SigLIP-style tower, hidden 1152) plus
   `image_token_id`/`video_token_id`; README `pipeline_tag: image-text-to-text`. It is **not** a
   plain text decoder. The whole SparseForge/CAST pipeline assumes `LlamaForCausalLM`.
2. **24 of 32 layers are NOT attention.** `layer_types = 8 × [linear_attention ×3,
   full_attention]` → 24 `Qwen3_5GatedDeltaNet` layers (Gated DeltaNet / linear attention with
   `conv1d` + `A_log` + `dt_bias` recurrent state). **A Mamba-class recurrent op, not a transformer
   block.** SLoRB is defined on `nn.Linear` `q/k/v/o/gate/up/down`; that projection set does not
   exist in the same form in 24/32 layers. The ladder's per-family asymmetry (§4.1) has **no**
   meaning here without redefinition.
3. **GQA changes every shape.** 16 Q heads / 4 KV heads, `head_dim=256` →
   `q_proj (8192,4096)`, `k_proj`/`v_proj` `(1024,4096)`. Since `r = in_features // k`, `k_proj`
   and `v_proj` would get `r=256` against `out=1024` — a branch **wider than the tensor it
   corrects**. The density bookkeeping in §4.1 must be re-derived from scratch.
4. **It is INSTRUCT / post-trained**, not a base LM. Comparing it to a base-protocol union-9
   number would repeat the chat-vs-base error this project has already made.
5. **An `mtp.*` multi-token-prediction head** (15 tensors incl. a full self-attn+MLP block) is
   silently discarded by `AutoModelForCausalLM` — a silent-capacity-loss trap.
6. **Non-standard shard filenames**: `model.safetensors-0000N-of-00004.safetensors` (a **dash**).
   Any tool globbing `model-*.safetensors` finds **zero** shards.
7. `models--Qwen--Qwen3.5-9b/` is an **empty stub**, not a second snapshot: `du -sb` = 40 bytes,
   3 entries total (`refs/main` holding commit `c202236235762e1c871ad0ccb60c8ee5ba337b9a`), **no
   `blobs/`, no `snapshots/`, zero weight bytes**. A failed download; nothing to compare.

**Honest cost.** Not `6 × 0.73` GPU-h. It requires (a) a `Qwen3_5` port of `sparse_modeling.py`
including a decision about what SLoRB even means on a GatedDeltaNet layer, (b) a full SparseForge
training run to produce a checkpoint with a trained branch, (c) only *then* a ladder. The training
leg alone is the dominant term and is **not estimated here** — deliberately, because estimating it
would imply the port is scoped, and it is not. **Prerequisite before any GPU: a 0-GPU design note
answering blocker 2.**

---

## 10. Open blockers before ANY GPU

1. **Novelty not checked; `RELATED_WORK.md` not written.** `proposal/README.md` requires Related
   Work before new GPU. Post-hoc low-rank-branch truncation, structured-basis coarsening of a
   low-rank adapter, and "is the adapter rank over-provisioned" are all well-trodden areas
   (LoRA-rank ablations, AdaLoRA, post-training SVD compression). **0-GPU and blocking.**
2. The `b86c479` harness commit is not resolvable in this checkout (§5.3).
3. `emit_slorb_ladder.py` has never been executed (§8.1). Its first run must be the CPU-only
   export of rung A, whose manifest density must equal **56.2500 %** and whose `branch_live` must
   equal **404,750,336** exactly — a self-check with a known answer, before any GPU is requested.
