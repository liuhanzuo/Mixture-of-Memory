# B12 G0 LEG 2 — tool self-check of `emit_slorb_ladder.py` (rung A)

## STATUS: COMPLETE (leg 2 discharged; see VERDICT at the end)

Independent re-execution / verification pass, 2026-08-16. GPU budget for this pass: **ZERO**
(all 40 cards on all 5 nodes were verified busy with multi-day training before this pass began;
nothing here touches a GPU, every invocation is under `CUDA_VISIBLE_DEVICES=''`).

Gate text being discharged (verbatim from `STATUS.json.next_gate`, LEG 2):

> run `baselines/cast_repro/tools/emit_slorb_ladder.py` with `CUDA_VISIBLE_DEVICES=''`
> `--mode ladder --rung A` on the 5B checkpoint and ASSERT the emitted
> `slorb_variant_manifest.json` reports `density_two_matmul_deployment_form == 0.5625`
> exactly and `live_branch_params == 404750336` exactly.

---

## F1. The gate text names two things that do not exist in the tool (read BEFORE running)

Read `baselines/cast_repro/tools/emit_slorb_ladder.py:220-242` (the argparse block) and `:483`.

| gate text says | tool actually has | verdict |
|---|---|---|
| `--mode ladder` | **no `--mode` flag at all.** Flags are `--ckpt --output --rung --coeffs --model --project-root --dtype --dry-run --allow-density-mismatch` | gate text is WRONG; `--mode ladder` would `error: unrecognized arguments` |
| manifest `slorb_variant_manifest.json` | `emit_slorb_ladder.py:483` writes **`slorb_ladder_manifest.json`** | gate text carries the PREDECESSOR's filename: `slorb_variant_manifest.json` is written by `emit_small_slorb_variants.py:236` |

`--rung A` is spelled correctly. So the executable form of the gate is
`--rung A` (no `--mode`), and the asserted file is `slorb_ladder_manifest.json`.
Neither substitution changes what is being asserted; both are recorded rather than smoothed.

Consumer cross-check: `scripts/launch_slorb_rank_sweep.sh:248` and `:379` both read
`slorb_ladder_manifest.json`, i.e. the DRIVER agrees with the tool and the GATE TEXT is
the odd one out. This is a defect in the gate string, not in the tool or the driver.

## F2. Units of the asserted density: the gate's `0.5625` is the FRACTION, and that is right

`emit_slorb_ladder.py:350` computes `density = (surviving + branch_live) / scope_elems`
and `:388` stores that raw fraction under `density_two_matmul_deployment_form`.
The tool's own `EXPECTED_DENSITY_PCT["A"] = 56.2500` is in PERCENT and is a different
quantity from the manifest key. The gate asserts `== 0.5625`, i.e. against the manifest
key in its own units. Consistent.

Exact arithmetic: (3,238,002,688 + 404,750,336) / 6,476,005,376 = 3,642,753,024 / 6,476,005,376
= 9/16 = 0.5625 EXACTLY. Both operands and the quotient are exactly representable in
binary64, so `== 0.5625` is an achievable exact float equality, not an approximation.

## F3. The 5B checkpoint — path resolved, size-verified, wzc1-ONLY (both disks searched)

The tool takes `--ckpt` with no default; the path comes from
`scripts/launch_slorb_rank_sweep.sh:59` (`CK=...`).

```
/apdcephfs_wzc1/share_304376610/pighzliu_code/out_llama/models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260413_201320/model_best_lm_eval.pt
```

- **wzc1: PRESENT**, 41,078,444,091 bytes (mtime 2026-04-13 22:38). Byte-for-byte equal to
  the driver's own identity guard `EXPECT_CK_BYTES=41078444091` (`launch_slorb_rank_sweep.sh:60`),
  so this is the checkpoint the union-9 rows were scored from.
- **zwfy6: ABSENT** (`ls` -> No such file or directory on the same relative path).
  Confirms `STATUS.json.needs_disk` ("wzc1 ONLY"). Both disks were searched; no `find /` was used.
- Dense reference `models/Llama--Llama2-7b` present on wzc1 with
  `model-0000{1,2}-of-00002.safetensors` + `model.safetensors.index.json`, needed by the
  non-dry-run save path (`emit_slorb_ladder.py:422-434`).

## F4. ⚠️ REAL DEFECT: `--mode ladder` does not error — argparse SILENTLY prefix-matches it to `--model`

I ran the gate **literally as written**, including the non-existent `--mode ladder`, expecting
`error: unrecognized arguments`. It did NOT error. rc=0.

```bash
CUDA_VISIBLE_DEVICES='' /opt/conda/envs/torch-base/bin/python \
  baselines/cast_repro/tools/emit_slorb_ladder.py \
  --mode ladder --rung A \
  --ckpt /apdcephfs_wzc1/.../model_best_lm_eval.pt \
  --output /apdcephfs_wzc1/.../outputs/slorb_rank_ladder_hf/verify_rung_A \
  --dry-run > logs/b12_g0_leg2_VERIFY_literal_gate_flags.log 2>&1
rc=$?   # rc=0
```

Root cause, isolated in a standalone reproduction of the exact parser
(`emit_slorb_ladder.py:220-242`, which does not pass `allow_abbrev=False`):

```
args.model = 'ladder'
MISPARSE: --mode was prefix-matched to --model
```

`--mode` is a unique prefix of `--model` among this parser's options, so Python argparse's
default `allow_abbrev=True` binds it. The literal gate command therefore ran with
`--model ladder`, i.e. the dense reference model path was silently overwritten to
`$PROJECT_ROOT/ladder` (which **does not exist** — verified, `ls` rc=2).

**Why the dry run still passed and why that is the dangerous part:** `--model` is only
consumed in the non-dry-run save path (`:422-434`, reading
`model.safetensors.index.json` and `from_pretrained`). `--dry-run` returns at `:415`,
BEFORE any use of `model_path`. So:

- with `--dry-run`, `--mode ladder` is a silent no-op and the assertions pass anyway;
- **without** `--dry-run`, the same command line would crash at `:423`
  (`FileNotFoundError` on `$PROJECT_ROOT/ladder/model.safetensors.index.json`)
  AFTER having spent the full ~2 min checkpoint pass and all the branch algebra.

This is exactly the class of bug this gate exists to catch: a wrong flag that produces a
*green* self-check. Fix options (NOT applied in this pass — reported only): pass
`allow_abbrev=False` to `ArgumentParser`, and/or correct the gate string to drop `--mode`.

## F5. LEG 1 (`RELATED_WORK.md`) — REAL, not a stub

215 lines, read in full. It is a substantive survey, not a placeholder: 30 numbered candidates each
with an id, a venue **plus the authority that established it**, a mechanism, and an explicit
preempts?-with-reason column; a §0 table of which authority was used and **what each refused**
(api.openreview.net v1 403 ChallengeRequiredError; AdaLoRA's OR note not locatable by title because
the camera-ready drops the prefix; Semantic Scholar not consulted, 429); verbatim `totalResults` for
5 negative search surfaces; and a §"Honest limits" listing 9 entries it could NOT raise above preprint.

Surfaces (a)-(d) from `novelty_status_detail`, all four addressed with named works:
- **(a)** post-hoc SVD/low-rank truncation of trained adapters -> §2, 6 works (#3-#8: PARA,
  LoRA-Squeeze, Spectral Surgery, PHLoRA, SpectralLoRA, LoRA-drop). All dense-base.
- **(b)** adapter-rank over-provisioning ablations -> §3, 10 works (#9-#18: AdaLoRA, DyLoRA, SoRA,
  VeRA, NOLA, Tied-LoRA, LoRA-FA, DoRA, Shears, NAS-companion). All training-time or dense.
- **(c)** structured/block-basis coarsening as opposed to SVD -> §4 sub-section, plus verbatim empty
  surfaces (`abs:"coarsening" AND abs:"low-rank" AND abs:"adapter"` -> 3, all irrelevant). Nearest
  named operators are base-WEIGHT clustering (eDKM, HPCA 2025) and quantisation grouping.
- **(d)** low-rank bypasses attached to N:M-sparse weights -> §4, SLoPe (ICLR 2025, OR
  `venueid=ICLR.cc/2025/Conference` id `lqHv6dxBkj`) and SLiM (ICML 2025, id `4UfRP8MopP`), both
  ADD/CONSTRUCT rather than REDUCE.

It also goes *against* its own author's interest in three places, which is the strongest signal it
is not a rubber stamp: it names a **citation gap in PROPOSAL.md** (neither AST nor CAST is named
anywhere, yet AST/AAAI-2025 is the ORIGIN of the SLoRB branch B12 operates on); it declares
**Dctl is NOT a novel control** because PARA (arXiv:2604.27796, concurrent) already does post-hoc
SVD of a trained adapter with non-uniform per-layer rank; and it flags a **genuine tension** with
SpectralLoRA's concentrated-spectrum finding, forbidding the general claim.

Caveat worth recording (does not make it a stub): §6 conjunct 1 cites
`evidence/g0_leg2_rungA_selfcheck_20260816.json` — i.e. the LEG-1 document leans on a LEG-2
artefact for its "exactly 2:4, measured over all 224" claim, so the two legs are not fully
independent. The 2:4 measurement is re-verified below on its own terms.

**One-line verdict: REAL, and it clears (a)-(d) with named works and stated authorities.**
(Leg 1 was not my assignment; this is a read-only assessment.)

## F6. ★ THE GATE ASSERTION — canonical run, both numbers EXACT. **PASS**

The gate asserts against the **emitted manifest**, so a `--dry-run` (which writes nothing, `:410-415`)
cannot discharge it. The full export was run. Correct flags (no `--mode`), CPU only:

```bash
CUDA_VISIBLE_DEVICES='' /opt/conda/envs/torch-base/bin/python \
  baselines/cast_repro/tools/emit_slorb_ladder.py \
  --rung A \
  --ckpt /apdcephfs_wzc1/share_304376610/pighzliu_code/out_llama/models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260413_201320/model_best_lm_eval.pt \
  --output /apdcephfs_wzc1/share_304376610/pighzliu_code/outputs/slorb_rank_ladder_hf/verify_rung_A \
  --project-root /apdcephfs_wzc1/share_304376610/pighzliu_code \
  --dtype bfloat16 > logs/b12_g0_leg2_VERIFY_fullexport_correctflags.log 2>&1
rc=$?     # rc=0
```
`rc` taken on the line immediately after the command, no pipe and no `$(...)` between them.

Assertion script `/tmp/b12_assert_leg2.py`, log `logs/b12_g0_leg2_VERIFY_assertions.log`, **rc=0**
(the script `sys.exit(1)` on any mismatch, so rc=0 is itself the assertion, not a report):

| asserted key | expected | **actual** | test | result |
|---|---|---|---|---|
| `density_two_matmul_deployment_form` | `0.5625` | **`0.5625`** (`repr` = `0.5625`, type `float`) | `==` exact float equality | **PASS** |
| `live_branch_params` | `404750336` | **`404750336`** (type `int`) | `==` integer equality | **PASS** |

Manifest read from the file the tool actually writes:
`.../outputs/slorb_rank_ladder_hf/verify_rung_A/slorb_ladder_manifest.json` (112,640 B, 224 per-tensor rows).

Corroborating values in the same manifest, none asserted by the gate but all checked:
`scope_elements 6,476,005,376` (== `EXPECTED_SCOPE_ELEMENTS`), `surviving_2of4 3,238,002,688`
(== `EXPECTED_SURVIVING`), `psi 0.5229408217630581`, `prereg_match true`, `source_iter_num 17900`,
`SLoRB_k 16`, `operator coarsen`, `coefficients ls`.

Two independent internal consistency checks I added (not in the tool):
- density **recomputed from the manifest's own integer fields**: `(3238002688 + 404750336)/6476005376`
  = `0.5625`, exactly equal to the stored float. So the stored density is not a separately-rounded
  number that happens to print right.
- **sum of the 224 per-tensor `live_branch_params` = 404,750,336**, exactly equal to the reported
  total. So the total is not a scalar accumulated on a different path from the rows.

Export artefacts written (13.48 GB `model.safetensors` + 5 tokenizer/config files), and the reused
exporter guards all passed: key-set invariant vs the dense reference index (`:422-430`),
`load_state_dict` round-trip (`:435-442`), tokenizer.model requirement (`:475-477`).

**Note the export is DENSE ON DISK and that is the pre-registered, correct outcome:**
`post_cast_zero_fraction = 0.0` and **all** 1,619,001,344 tiles "violating" 2:4, `is_dense_on_disk true`.
The rung is folded (`W*mask + E`) and `E` is dense, so every masked-out entry is overwritten. The
tool's `:460-464` guard (which fires if the folded export IS exactly 2:4, i.e. the branch was a no-op)
correctly did NOT fire. Rung A's `zero_fraction` (0.0) is even lower than R0 hard_fold's (1.70e-9), so
**zero_fraction does not order this ladder** and must never be quoted as a sparsity claim for any rung.

## F7. Negative control on MY assertion harness — it is not vacuous

Before trusting rc=0 I perturbed a copy of the manifest by the smallest possible amount and re-ran
the identical harness (`logs/b12_g0_leg2_VERIFY_negctl.log`):

| perturbation | harness rc | verdict |
|---|---|---|
| `live_branch_params` 404750336 -> **404750337** (+1) | **1** | `FAIL` — fires correctly |
| `density` 0.5625 -> **0.5625000001** (+1e-10) | **1** | `FAIL` — fires correctly |

So the PASS in F6 is a real test, not a harness that returns 0 unconditionally.

## F8. ⚠ HOW MUCH the rung-A assertion actually tests: it is ARITHMETICALLY FORCED, given the two guards

This weakens (does not void) the gate's claim to be a strong correctness check, and it should be
recorded rather than left for someone to discover at rung P.

Measured over all 224 rows of the emitted manifest (`logs/b12_g0_leg2_VERIFY_tautology.log`):

```
all c==1                     : True          (rung A's map is c=1 everywhere)
all in_dim % 16 == 0         : True
live == out*(in//16) per row : True          (matches :325 live = S_eff.numel())
sum(out*in) == scope         : True
scope//16 == live_branch_params : True       (404,750,336)
scope//2  == surviving          : True       (3,238,002,688)
```

For rung A, `c=1` so `k_eff = 16` and `live = out * (in/16) = W.numel()/16` for every tensor. Hence
`branch_live == scope/16` and `density == surviving/scope + 1/16`. But `scope` and `surviving` are
**already hard-asserted earlier in the same function** (`:344-349`, `SystemExit` if either differs).
Once those two guards pass, `branch_live = 6,476,005,376/16 = 404,750,336` and
`density = 1/2 + 1/16 = 0.5625` are *forced by integer arithmetic*, not measured.

**What the rung-A assertion therefore DOES and DOES NOT test:**
- **DOES test**: that the checkpoint is the right one; that the mask reconstruction (`nm_2_4_hard`)
  yields exactly 50% survivors over all 224 tensors; the aux-suffix filter and projection selection
  (a mis-selected tensor set would move `scope`); the whole reused save path with its three guards;
  and that the tool runs at all on this checkpoint (its first-ever execution — which is the real value).
- **DOES NOT test**: the coarsening arithmetic at `c>1`, i.e. `_fit_blocksum_ls` at `k_eff>16`
  (rung A takes the `c==1` degenerate branch), `_materialise_blocksum`'s tail handling, the
  `r_eff2 == r_eff` assert at `:323`, the entire SVD path (`_branch_svd`, `_svd_t_density_matched`),
  and the non-divisible-`in_dim` `SystemExit` at `:178-180`. `STATUS.json.g0_result_20260816
  .does_NOT_discharge` already says this; F8 quantifies *why* rung A cannot reach it.

Consequence for the pilot: **rung P is the first cell whose density is not forced**, so its
pre-registration assertion (`P: branch_live 325,844,992 / density 55.0316%`) is the first genuinely
falsifiable one. It should be run as a `--dry-run` FIRST (~2 min CPU, 0 GPU) before any GPU is
committed, because a `c>1` bookkeeping error would otherwise surface only after the export.

## F9. EXTRA (beyond the gate, 0 GPU): rung P dry-run — the FIRST non-forced assertion also PASSES

Because F8 showed rung A cannot reach the `c>1` code, I spent ~6 min of CPU (0 GPU) exercising it via
a dry run of the pilot rung. This is bookkeeping only; **no export was written and NOTHING was launched.**

```bash
CUDA_VISIBLE_DEVICES='' /opt/conda/envs/torch-base/bin/python \
  baselines/cast_repro/tools/emit_slorb_ladder.py --rung P \
  --ckpt /apdcephfs_wzc1/.../model_best_lm_eval.pt \
  --output /apdcephfs_wzc1/.../outputs/slorb_rank_ladder_hf/dryrun_rung_P \
  --project-root /apdcephfs_wzc1/share_304376610/pighzliu_code \
  --dry-run > logs/b12_g0_leg2_EXTRA_rungP_dryrun.log 2>&1
rc=$?     # rc=0
```

```
[b12] coarsen map: q=1 k=1 v=1 o=1 gate=1 up=1 down=8
[b12] scope=6,476,005,376 surviving=3,238,002,688 branch_live=325,844,992
[b12] density=55.0316%  psi=0.6159  (expected 55.0316% / 0.6159)
[b12] OK pre-registration match: branch_live and density both exact
```

**Why this is the stronger result**, verified by independent arithmetic from Llama-2-7B shapes
(32 layers x {q,k,v,o (4096,4096); gate,up (11008,4096); down (4096,11008)}):

```
A live = 404,750,336  == scope/16   -> TRUE   (forced, per F8)
P live = 325,844,992  == scope/16   -> FALSE  (NOT forced)
P density = 55.031574%,  |55.031574 - 55.0316| = 2.6e-05  (inside the tool's 5e-4 tolerance)
```

Rung P's number is **not** obtainable from the two guarded integers; it requires the `c=8` path with
`k_eff = 128`, `r_eff = 11008//128 = 86` on down_proj only. The tool's `_fit_blocksum_ls` +
`r_eff2 == r_eff` assert (`:323`) and the `in_dim % k_eff` refusal (`:178-180`) all executed, and the
independently pre-registered `EXPECTED_BRANCH_LIVE["P"]` matched exactly. So the `c>1` coarsening
bookkeeping is now verified too, at zero GPU, and the pilot's own prereg assertion is de-risked.

Still NOT exercised by anything run here (unchanged from `STATUS.json.does_NOT_discharge`): the
**entire SVD path** (`_branch_svd`, `_svd_t_density_matched`, i.e. rung Dctl), and the non-dry-run
export at `c>1` (only rung A has actually been materialised and saved).

## F10. Independent re-verification of the exact-2:4 mask claim (all 224 tensors, no sampling)

Re-ran the prior pass's own checker `evidence/g0_leg2_verify_mask_exact2of4.py`
(log `logs/b12_g0_leg2_maskverify.log`) and read its output rather than trusting the summary:

```
n_in_scope_tensors 224 | mask_elements 6,476,005,376 | mask_ones 3,238,002,688
mask_ones_fraction 0.5 | mask_exact_2of4_violations 0 | columns_outside_any_group_of_4 0
W*mask zero_fraction 0.5 | W*mask exact_2of4_violations 0
```

So the 2:4 property holds EXACTLY where it lives (the mask / two-matmul deployment form), over all
224 in-scope tensors with no sampling, while the folded on-disk export is dense (F6). Both statements
are true simultaneously and must be reported together.

---

# VERDICT

**LEG 2: PASS.** `emit_slorb_ladder.py` ran on the 5B checkpoint under `CUDA_VISIBLE_DEVICES=''`,
rc=0, and the emitted manifest reports:

- `density_two_matmul_deployment_form` = **0.5625** — expected 0.5625 — **exact float equality**
- `live_branch_params` = **404750336** — expected 404750336 — **exact integer equality**

**Defects found and reported (NOT fixed here):**
1. **`--mode ladder` in the gate text is silently prefix-matched to `--model` by argparse**
   (`allow_abbrev` not disabled). Green under `--dry-run`, but would crash the *real* export at `:423`
   after ~2 min of wasted work. Fix: `ArgumentParser(..., allow_abbrev=False)` and drop `--mode` from
   the gate string. (F4)
2. **The gate names the wrong manifest filename** — `slorb_variant_manifest.json` is the PREDECESSOR's
   (`emit_small_slorb_variants.py:236`); this tool writes `slorb_ladder_manifest.json` (`:483`), which
   is also what the driver reads (`launch_slorb_rank_sweep.sh:248,:379`). (F1)
3. **The rung-A assertion is arithmetically FORCED once the `scope`/`surviving` guards pass**, so it
   tests the checkpoint + mask + save path but *not* the coarsening algebra. Mitigated in F9 by
   additionally dry-running rung P, whose value is not forced and also matched. (F8, F9)
4. Pre-existing, independently confirmed and unchanged: `scripts/launch_slorb_rank_sweep.sh` is
   **un-runnable as written** (`PY`/`LM_EVAL` default to `$REPO/venv_union9`, but the venv is at
   `$PROJECT_ROOT/venv_union9`, one level above the checkout; the guard at `:114` fires immediately).

**LEG 1: `RELATED_WORK.md` is REAL, not a stub** (215 lines, 30 candidates with per-family
authorities, surfaces (a)-(d) each addressed with named works, verbatim empty-search `totalResults`,
and three self-adverse findings). Assessed read-only; adjudicating leg 1 is not this pass's mandate.

**G0 as a whole: now clearable** on the evidence — leg 2 passes on its own terms and leg 1 exists and
is substantive. **This document does NOT set `ready_gpu` and does NOT launch the pilot**; that
decision, and the formal adjudication of leg 1, are reserved to the operator.

**GPU used by this pass: 0.0 GPU-h.** Every invocation carried `CUDA_VISIBLE_DEVICES=''`
(`torch.cuda.is_available() == False` verified); no node was ssh'd for compute; no card was touched.

---

## Appendix — bookkeeping hygiene of this pass

`STATUS.json`: appended exactly ONE dated sibling key,
`g0_leg2_independent_reverification_20260816`, matching the file's existing convention
(`g0_result_20260816`, `g0_result_correction_20260816`, `next_gate_gpu_20260816`, `lifecycle_20260816`).
Append-only discipline verified against `git show HEAD:...`:

```
added  : ['g0_leg2_independent_reverification_20260816']
removed: []      changed: []      ORDER of old keys preserved: True
```

`proposal/ready_queue.py` was run before and after with the ONLY difference being this key: the
report is **byte-identical** (`diff` empty). So the append is **inert to the scheduler** — it records
the verification without moving any lifecycle. B12 was ALREADY `ready_gpu` from the prior pass
(`lifecycle_20260816`), and **this pass did not set that and did not touch it**; deciding whether that
declaration is warranted, and adjudicating leg 1, remain the operator's.

Independent note: the prior pass's leg-2 claim could NOT simply be inherited — its full export went to
`/tmp/b12_rungA_export`, which no longer exists (`/tmp` was cleared, cf.
`memory/persist-artifacts-on-wzc1-or-diskb.md`), so its manifest was not re-readable. This pass's
export was therefore written to the project disk (wzc1) at
`outputs/slorb_rank_ladder_hf/verify_rung_A/` so the asserted manifest survives a reboot.
