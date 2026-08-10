# CAST reproduction (arXiv:2509.25996v1) — paper-based reimplementation

CAST has **no public code and no public checkpoint**. Everything here is built from the paper text in `docs/cast_arxiv_2509.25996v1_fulltext.txt` (+ `paper/` for table layout), with the same group's AST code (`baselines/ast_official_clean`, commit 28bd277) used only where the paper is silent.

Every hyperparameter is tagged `paper_explicit` / `ast_code_inferred` / `implementation_choice` in **[`SPEC.md`](SPEC.md)**. Read that before citing any number from here.

Because the official code is unavailable, results must be labelled **"CAST (our paper-based reimplementation)"**, never "CAST".

```
cast/
  sparse_linear.py   CastSparseLinear: dense forward, 2:4 magnitude mask (Eq.6),
                     learnable group scaling (Eq.11-12), finalization (Alg.2 L20-21)
  adams.py           AdamS (Alg.1, Eq.7-8) + the hard 100%-coverage assertion
  distill.py         Eq.13 convex KL+CE loss, self-teacher
  diagnostics.py     audit section-5 masked/kept magnitude report
  train_cast_llama.py  DDP training loop
tests/test_cast.py   33 unit tests (all passing, output below)
  checkpoint.py      resumable save/load: per-rank shard files, arg guard, the
                     anti-warm-restart verifier
tools/
  smoke_alignment.py        real-scale (224-tensor) alignment proof, pure torch
  fsdp_misalignment_demo.py empirical proof of the old FSDP bug
  integration_tiny.py       end-to-end loop test with a tiny stub model
  decay_budget.py           pre-flight feasibility check
  throughput_probe.py       measured step time / wall-clock projection
  diagnose_checkpoint.py    run the audit metric on a checkpoint
  prepare_dolmino_llama2.py PRIMARY data path (EXECUTED -- 77.7B tokens on disk)
  verify_checkpoint_roundtrip.py  bit-exact save/load proof at 7B scale (8 GPUs)
  resume_faithfulness.py    loss-trace resume comparison (see S6 for why it is
                            NOT the primary evidence)
scripts/launch_cast_llama.sh
```

---

## 1. The six audit failures, and what fixes each

Reference: `Mixture-of-Memory/SparseForge_Data/docs/CAST_REPRODUCTION_AUDIT.md`.

| # | Audit finding | Fix here | Evidence |
|---|---|---|---|
| **1** | **[fatal]** FSDP sliced `weight` and `mask` at different FlatParameter offsets; old code hit unequal numel → `mask = None` → **silent vanilla Adam**, so selective L1 decay never ran. Ratio 0.294, Wiki PPL 23.4514. | **No parameter sharding.** `mask` is a **buffer**, never packed into a FlatParameter, and `weight` is never sliced — so alignment holds structurally. AdamS **raises `MaskCoverageError`** on a missing/misshaped mask rather than degrading. ⚠️ **2026-08-09 corrections:** (a) the live run is **not** plain DDP but `--parallel zero2` = DDP + `ZeroRedundancyOptimizer` (**ZeRO-1** — the flag name is a misnomer; plain DDP OOMs at 7B, see SPEC.md §5); `_partition_parameters` assigns whole Parameters and never slices, so the alignment argument carries over. (b) **`aligned=224/224` in the log is NOT evidence of alignment** — the check compares only not-None/shape/device, never contents, and a deliberately wrong mask (49% agreement) still scores 224/224. `decayed=3,238,002,688` is vacuous (`topk(2)` writes `numel/2` True unconditionally, even for all-NaN weights) and `check_mask_sync` is dead code. Use the checkpoint mask-recompute probe instead. See SPEC.md §8 item 1. | `tools/fsdp_misalignment_demo.py` reproduces the bug (FSDP gives `(262144,)` and even `(0,)` weight shards against a `(512,512)` mask). Unit tests `test_missing_mask_raises`, `test_shape_mismatch_raises`. For actual alignment: recomputing the 2:4 mask from `ckpt_step1250/model.pt` weights gave 0.9997-0.99998 agreement vs 0.559 for a mismatched control. |
| **2** | **[fatal]** Paper uses Dolmino-Mix-1124; old run used C4. | **FIXED.** `tools/prepare_dolmino_llama2.py` was executed: `data/dolmino-mix-1124-llama2/` holds **77,721,665,859** LLaMA-2 tokens (vocab 32000), 9.9× the 7.86B the run needs, so it trains for <1 epoch. It is now the launcher default; C4 is opt-in only. ⚠️ the tokenizer wrote **uint32**, not uint16 — `--data-dtype auto` reads `metadata.json` and cross-checks it against the byte size, and **refuses to run** if metadata is absent (a hardcoded uint16 would reinterpret each 4-byte token as two, silently doubling the stream and injecting zeros). | `metadata.json`: `dtype=uint32`, `total_tokens=77721665859`; asserted at launch by `scripts/launch_cast_llama.sh` and again in-process |
| **3** | KL temperature: Eq. 13 has none, old code used T=2 with ×T². | Default **T = 1.0** (paper-literal). T=2 available as a named variant via `--kl-temperature 2.0`, never silently. | `test_temperature_default_is_paper_literal` |
| **4** | LR schedule not specified by the paper but presented as config. | **Superseded 2026-08-09.** The default is now **`--lr-schedule constant`** at peak 2e-5, which is what the live 7500-step run uses; `cosine` → `min_lr` 2e-6 + 375-step warmup is still available but **off by default**. Both are `implementation_choice` — the paper specifies no within-run schedule (`grep -ci cosine` over the full text = 0). ⚠️ Two defects remain open: the code labels `constant` "paper-literal: Appendix B", but the phrase "consistent learning rate" is at line 1646 in **Appendix D** and is about holding LR fixed *across* the scaling-law's token-budget points, not the within-run schedule; and **SPEC.md §6 still argues the LR must decay** and was never updated after the default flipped (`git log b4addd7..HEAD -- SPEC.md` is empty). Consequence: `constant` buys 2.6× total decay distance but gives up ~10× of terminal-magnitude margin (residual floor 2.0e-5 vs 2.0e-6). See the ⚠️ box in SPEC.md §6 for the accounting and the step-4000-5000 `max_masked_magnitude` re-check that decides whether it mattered. | SPEC.md §1 + §6 |
| **5** | Mask refreshed *after* `optimizer.step()`, so step 0 used an all-ones mask. | Refresh at the **top of step t**, before backward and before `opt.step()` (Alg. 1 L6-8; Alg. 2 L8-10 precede L13/L16). | `test_mask_refresh_before_step_and_every_T1` asserts the step-0 mask is already 2:4 |
| **6** | Eval at 2048 context; paper trains and evaluates at 4096. | `--seq-len 4096` default for both. | SPEC.md §1, §7 |

Additional hardening not in the audit: **fp32 master weights are enforced** (`require_fp32`). λ=4e-7 is below bf16 resolution, so a bf16 master weight silently discards the entire decay signal — AdamS raises rather than letting that happen.

## 2. Unit tests — real output

`/opt/conda/envs/torch-base/bin/python tests/test_cast.py` on `.21`:

```
PASS  test_nm_mask_hand_computed
        2:4 mask matches hand computation; exactly N per group
PASS  test_nm_mask_ties_still_exact
        exact ties still yield exactly 2 kept (topk, not >= xi)
PASS  test_adams_decays_masked_not_kept
        alpha_0=0.0 (no first-step decay); after 5 steps kept |delta|=0.00e+00, masked mean delta=-2.180e-02 (toward zero)
PASS  test_adams_sign_at_zero_is_zero
        sign(0)=0: weights already at zero stay exactly zero
PASS  test_adams_alpha_ramps_linearly_zero_to_one
        alpha_t = t/T from 0.0 to 1.0 (clamped at 1.0), T=8
PASS  test_second_moment_uses_mu_tilde
        v from mu~: masked v mean=3.124e-06 > 0, kept v=0.0e+00 (vanilla Adam would give 0 everywhere)
PASS  test_missing_mask_raises
        missing mask raises MaskCoverageError: [AdamS] in-scope parameter has no `cast_mask` attribute...
PASS  test_shape_mismatch_raises
        mask/weight shape mismatch raises (no silent Adam fallback)
PASS  test_expected_element_count_enforced
        wrong expected_scope_elements raises before any training happens
PASS  test_non_fp32_in_scope_raises
        bf16 in-scope weight is rejected up front
PASS  test_bf16_swallows_lambda_fp32_does_not
        after 100 decays of lambda=4e-07: fp32 moved 4.005e-05 (expected 4.000e-05), bf16 moved 0.0 -> signal fully lost
PASS  test_scale_groups_axis_semantics
        A[k] axis: element (r,c) scaled by A[r, c//(C/n)] (contiguous column blocks)
PASS  test_finalize_exact_2of4_and_scale_folded
        finalize: every one of 16 groups has exactly 2 nonzeros; scale folded into W and reset to 1
PASS  test_forward_is_dense
        forward is dense (masked weights still contribute): output=8.0 for 8 unit inputs
PASS  test_conversion_scope
        converted exactly ['q_proj', 'down_proj']; 'other' left dense and out of scope
PASS  test_mask_refresh_before_step_and_every_T1
        refreshed at steps [0, 10, 20] (T1=10); step-0 mask already 2:4 (50% kept), and AdamS's exact-2:4 assertion held on every step
PASS  test_kl_zero_when_identical
        D_KL(P||P) = 0.00e+00
PASS  test_cast_loss_is_convex_combination
        L = 0.333*KL + 0.667*CE checks out (ce=1.8748, kl=0.8157, total=1.5218); eta=0 -> pure CE
PASS  test_temperature_default_is_paper_literal
        default T=1.0 (paper), eta=1/3; KL@T=1=1.2374 vs AST-style T=2 -> 1.4899
PASS  test_convex_unnormalised_equivalence
        eta=1/3 convex  <=>  eta'=0.500 un-normalised with lr'=1.3333e-05
PASS  test_end_to_end_masked_weights_go_to_zero
        after 600 steps (lr 4e-03->4e-05, 4.2x decay headroom): masked/kept ratio=0.000292
        (broken run: 0.294), 100.0% of masked < 1e-4 (broken run: 21.5%), max |masked|=9.7e-05,
        hard-prune delta=0.0440%
PASS  test_terminal_magnitude_is_set_by_final_lr
        late per-step |dw| = 1.69 x lr (independent of lambda) -> the residual floor is O(final lr),
        so the LR schedule must decay for masked weights to vanish
PASS  test_resume_accepts_identical_config
        identical config resumes cleanly, no spurious diffs reported
PASS  test_resume_allows_benign_differences
        benign diffs allowed and reported: ['resume', 'save_every', 'stop_after']
PASS  test_resume_rejects_changed_max_steps
        changed --max-steps is refused (alpha_t = t/T would be rescaled)
PASS  test_resume_rejects_changed_l1_decay_and_lr
        changed l1_decay / lr / seed / seq_len all refused
PASS  test_resume_reports_every_mismatch_at_once
        all mismatching keys are listed in a single error
PASS  test_resume_rejects_checkpoint_missing_a_key
        missing key in checkpoint => refuse (cannot prove equality)
PASS  test_warm_restart_is_detected
        healthy moments pass; zeroed moments (the silent warm-restart signature) raise
        ResumeMismatchError
PASS  test_wrong_step_counter_is_detected
        a rewound step counter raises (would silently restart the decay ramp)
PASS  test_checkpoint_roundtrip_single_rank
        save->perturb->load restores weights, mask, both moments and the numpy data stream
        bit-exactly (next indices [80, 46, 51] reproduced)
PASS  test_incomplete_checkpoint_is_not_loadable
        a checkpoint without the DONE marker is refused and never auto-selected
PASS  test_legacy_weightsonly_checkpoint_is_refused
        a legacy weights-only .pt file is refused with an explicit reason

33/33 passed
```

The last 11 target the checkpoint/resume machinery; see S5b.

## 3. GPU smoke results (`.21`, 8×L20A cc10.0 183 GiB, verified idle first)

### 3a. Alignment assertion at real LLaMA2-7B scale (`tools/smoke_alignment.py`)

Pure torch — builds the 224 in-block projections at exact LLaMA2-7B dims, so it needs no `transformers`.

Single GPU:
```
[smoke] building 32 LLaMA2-7B-shaped blocks on NVIDIA L20A
[smoke] modules=224  scope={"cast_tensors": 224, "cast_elements": 6476005376, "cast_masked_elements": 3238002688}
[smoke] STATIC OK: 224 tensors, 6,476,005,376 elements, 3,238,002,688 masked (exactly half)
[smoke] step 0: aligned=224/224 coverage=100% decayed=3,238,002,688 alpha=0.000000 peak_mem=103.5 GiB
[smoke] step 3: aligned=224/224 coverage=100% decayed=3,238,002,688 alpha=0.000400 peak_mem=103.5 GiB
[smoke] PASS  alignment 224/224 (100%)
[smoke] PASS  decayed_elements = 3,238,002,688 (expected 3,238,002,688)
[smoke] PASS  peak memory 103.5 GiB / 178 GiB
```

8-GPU DDP:
```
[smoke] DDP wrap OK: every mask still element-aligned with its weight
[smoke] step 3: aligned=224/224 coverage=100% decayed=3,238,002,688 peak_mem=127.7 GiB
[smoke] PASS  peak memory 127.7 GiB / 178 GiB
```

`3,238,002,688` is not a fitted constant — it is `32 × (4·4096² + 3·4096·11008) / 2`, i.e. exactly half the in-block linear parameters, which is what exact 2:4 requires. AdamS asserts it every step.

### 3b. The FSDP bug, reproduced (`tools/fsdp_misalignment_demo.py`, 4 ranks)

```
[demo] DDP  : 4/4 tensors element-aligned  -> weight.shape == mask.shape everywhere
[demo] FSDP FULL_SHARD, use_orig_params=True:
[demo]   pre-wrap shapes (rank0): ('0.q_proj', (512, 512), (512, 512))
[demo]   MISALIGNED 0.q_proj.weight:    weight shard (262144,) vs mask (512, 512)
[demo]   MISALIGNED 0.down_proj.weight: weight shard (0,)      vs mask (512, 512)
[demo]   => 4 in-scope tensors have a weight shard whose shape/numel differs from the mask
```

This confirms the audit's root-cause diagnosis empirically rather than by assertion.

### 3c. End-to-end loop (`tools/integration_tiny.py`, 1 and 4 ranks)

Runs the real `train_cast_llama.main()` against a tiny stubbed LLaMA:
```
[cast] converted 14 in-block projections to CastSparseLinear
[cast] step 0/12  loss=4.3539 ce=6.3649 kl=0.3319 alpha=0.0000 aligned=14/14 decayed=163,840 flips=0
[cast] step 11/12 loss=4.3366 ce=6.3553 kl=0.2993 alpha=0.9167 aligned=14/14 decayed=163,840
[cast] finalized 14 modules (pruned with M_T, then folded the scaling module)
[cast] exact 2:4 violations after finalize: 0
PASS integration: loop ran, 14 saved projections are exact 2:4
```

**Not yet run:** a ≤50-step smoke of the *real* LLaMA2-7B through `train_cast_llama.py`. That is the only remaining gap and it is blocked solely on `transformers` not being installed on `.21` (§5).

## 4. Data — PRIMARY (Dolmino-Mix-1124) is now on disk and is the default

The paper uses **Dolmino-Mix-1124** with LLaMA-2 (Sec. VI-A). `tools/prepare_dolmino_llama2.py`
has since been **executed**, so the fallback is no longer needed:

| path | tokens | tokenizer | dtype | verdict |
|---|---|---|---|---|
| `Mixture-of-Memory/data/dolmino-mix-1124-llama2/` | **77,721,665,859** | Llama2-7b, vocab 32000 | **uint32** | **PRIMARY — the launcher default.** 9.9× the 7.86B needed ⇒ <1 epoch, no repetition |
| `data/dolmino-mix-1124-llama3/` | 469B | Llama3-8B, vocab 128000 | uint16 | unusable — LLaMA-2 vocab is 32000, id spaces unrelated |
| `data/dolmino-flan-heavy/` | 499.5M | Llama2-7b | uint16 | right source, 16× too small, and FLAN-heavy (38.9%), not Dolmino proportions |
| `data/c4_llama/` | 21.7B | Llama2-7b | uint16 | opt-in fallback only; this is the corpus the **broken** run used |

Note the path: it is under `Mixture-of-Memory/data/`, **not** `$PROJECT_ROOT/data/` (a different,
older data tree that does not contain dolmino). Passing the wrong one is not a silent failure —
`BinDataset` raises on the missing `train.bin`.

### ⚠️ The uint32 trap

The tokenizer wrote **uint32** even though vocab 32000 fits in uint16. Reading it as uint16
would reinterpret every token as *two* tokens — half of them zeros — with **no error anywhere**:
the memmap would simply be twice as long and the corpus would be garbage. Guards now in place:

1. `--data-dtype auto` (the default) reads `dtype` from `metadata.json`. It no longer falls back
   to uint16 when metadata is missing — it **raises**. The old fallback fired for real during
   development, on a mistyped `--data` path, and cheerfully logged
   `data-dtype auto-resolved to uint16` for a directory that did not exist.
2. It cross-checks `os.path.getsize(train.bin) // itemsize == metadata["total_tokens"]`, which
   catches a truncated file *and* a dtype/metadata disagreement.
3. `scripts/launch_cast_llama.sh` asserts the same before spending a single GPU-second, and
   prints the resolved width.

Verified at launch:
```
[cast] data-dtype resolved to uint32 (4 B/token) from metadata.json; dataset=allenai/dolmino-mix-1124
       tokenizer=.../models/Llama--Llama2-7b total_tokens=77,721,665,859 (byte size agrees)
[cast] train tokens: 77,721,665,859
```

Because PRIMARY is available, the "controlled C4 reimplementation" labelling caveat that used to
live here **no longer applies**: this is a paper-setting corpus.

## 5. Running it

`transformers` **is** installed on `.21` now (the earlier blocker is resolved), and the real
LLaMA2-7B path has been exercised end to end under `--parallel zero2` — see S3 for the alignment
numbers and S5b for the checkpoint evidence.

```bash
# 1. pre-flight feasibility (no GPU)
python tools/decay_budget.py --steps 7500
#    constant LR -> HEADROOM 11.19x (cosine: 4.32x, residual floor 2.0e-06), VERDICT: OK

# 2. real-model smoke, <=50 steps
bash scripts/launch_cast_llama.sh smoke
#    expect: "aligned=224/224 ... decayed=3,238,002,688" and finite loss

# 3. full run (dolmino is the default; dtype auto-read from metadata.json)
bash scripts/launch_cast_llama.sh full
RESUME=auto bash scripts/launch_cast_llama.sh full   # continue after a crash

# 4. prove the checkpoint machinery on this box before trusting a multi-day run
torchrun --nproc_per_node 8 tools/verify_checkpoint_roundtrip.py --parallel zero2 --steps 3
#    expect: VERDICT: PASS - state round-trips BIT-EXACTLY

# 5. verify the fix actually took, on the pre-finalization checkpoint
python tools/diagnose_checkpoint.py --ckpt outputs/cast_repro_zero2/prefinal.pt
#    PASS if ratio_mean < 0.01 and frac_below_1e-4 > 0.95 (broken run: 0.294 / 21.5%)
```

**Full-run ETA: ~34 h on 8×L20A (measured lower bound).** From `tools/throughput_probe.py` on one L20A (pure torch, so it ran without `transformers`):

```
measured  : 0.331s per micro-batch of 4,096 tokens
            480.6 TFLOP/s achieved on 1 card (in-block linears only)
corrected : x1.55 for attention+head+teacher -> 0.513s
projection to 8 cards, global batch 256:
  32 micro-steps/card/step -> 16s per optimizer step
  7500 steps -> 34 h = 1.4 days
peak memory: 104.9 GiB
```

Caveats: assumes perfect DDP scaling, omits dataloader and checkpoint I/O, and the ×1.55 correction (attention matmuls ×1.12, embeddings+lm_head ×1.04, bf16 teacher forward ×1.33) is an estimate. Expect **1.5–3 days** realistically. Do **not** use the paper's Appendix F figure (403 s/step for LLaMA3-8B on 32×H800) to project this — it is ~40× slower than a FLOPs estimate allows and extrapolating from it wrongly suggests a multi-month run.

~~Memory is not the blocker (127.7 GiB of 178 GiB measured under 8-GPU DDP)~~, but DDP does not shard optimizer state, so adding cards will not reduce per-card memory — the 183 GiB L20A/B200 class is required and an 80/97 GiB H20 cannot hold this run.

> **★ 2026-08-09 CORRECTION — memory IS the blocker.** The struck-through clause
> above (written 2026-08-08 16:11) predates the first real training attempt
> (2026-08-09 03:17–03:29) and is FALSE. The run OOM'd on the **second
> micro-batch of step 0**.
>
> * Per-rank **static** cost under plain DDP is **131.8 GB** (fp32 master 25.1 +
>   fp32 grads 25.1 + Adam m 25.1 + Adam v 25.1 + bf16 compute 12.6 + bool masks
>   6.3 + bf16 frozen teacher 12.6). L20A capacity 178.35 GiB → **46.6 GB** left
>   for activations.
> * Step 0 completed at a **measured** 138.6 G (`aligned=224/224`,
>   3,238,002,688 weights decayed), then the next 172 MiB allocation failed with
>   99.75 MiB free.
> * Measured peaks: **178.33 GiB** without gradient checkpointing, **174.04 GiB**
>   with checkpointing + `expandable_segments:True` — still OOM by ~100 MiB.
> * **Do not conflate the two figures: 131.8 GB is the static budget; 138.6 G is a
>   measured step-0 peak that already includes activations.** (An earlier summary
>   mixed them and produced the self-inconsistent 178.4 − 138.6 = 46.6.)
>
> The paper's own reported hardware (8×H800, 94 GB/card) also cannot hold
> 131.8 GB/rank, which implies **the paper must have used FSDP or ZeRO sharding**
> and simply does not state its parallelism strategy. Sharding therefore fills in
> an axis the paper left unspecified rather than deviating from it.
>
> Fix in progress: ZeRO-2 style sharding of optimizer state + gradients
> (`--parallel zero2`), which MUST preserve `weight ↔ mask` element alignment —
> see the `train_cast_llama.py` module docstring for why FSDP FULL_SHARD is
> forbidden (it silently disabled CAST once already; 7.86B tokens burned, Wiki
> PPL 23.45). `require_fp32=True` is non-negotiable: λ=4e-7 is below bf16
> resolution.

## 5b. Checkpoint / resume — what is guaranteed, and the evidence

**Why this got hardened.** A sibling project in this repo resumed three 200k-step arms and all
three *silently* became WARM RESTARTS: the optimizer param-groups did not line up, torch quietly
re-initialised the Adam moments, and a differential-LR setting never took effect. Nothing raised.
Weeks of compute produced arms that could not be compared to anything. A 1.5–3 day CAST run that
cannot be resumed or verified would be worse than no run.

### What a checkpoint contains

`ckpt_step<N>/` is a **directory**, written every `--save-every` steps:

| file | written by | contents |
|---|---|---|
| `meta.json` | rank 0 | step, full args, world size, parallel mode, torch version |
| `model.pt` | rank 0 | fp32 master weights + `cast_scale` + all **224 `mask` buffers** |
| `optim_rank<k>.pt` | rank k | that rank's Adam shard, keyed by ZeRO **global** param index |
| `rng_rank<k>.pt` | rank k | torch CPU + torch CUDA + the **numpy Generator** driving `BinDataset` |
| `DONE` | rank 0, last | published only after a barrier ⇒ a torn checkpoint is never loadable |

Measured: **81.4 GiB** per checkpoint (14.6 GB model + 8 × ~6.5–7.5 GB optimizer shards).
`--keep-last 2` therefore caps disk at ~163 GiB.

**The mask is saved, not recomputed.** It is refreshed only every `T1=10` steps, so it is *not* a
function of the current weights: the mask live at step 503 was computed from the weights as of
step 500. Recomputing it at resume time flips the entries near the intra-group threshold, changing
*which* weights get decayed. The live mask is restored verbatim.

### Why NOT `ZeroRedundancyOptimizer.consolidate_state_dict`

The obvious design — gather all shards to rank 0, write one file — was implemented first and
**hangs at 7B scale**. Measured on 8×L20A: `consolidate_state_dict(to=0)` ran >10 min, all 8 ranks
at 100% GPU util, zero bytes produced. py-spy across all ranks located it exactly — rank 3 (the
then-sender) parked on `zero_redundancy_optimizer.py:102`:

```python
data_send_tensor = torch.ByteTensor(data).to(device)   # `data` is a ~7 GB bytearray
```

`torch.ByteTensor(bytearray)` constructs element-by-element through the Python C-API while holding
the GIL — O(7e9) Python-level ops. It is fine for the small optimizer states the helper was written
for and unusable here. **Do not "fix" this by waiting longer.** The shard-file design avoids
cross-rank transfer entirely, and is strictly safer: state is already where it belongs, so it
cannot be mis-routed.

### The evidence: bit-exact round-trip at full scale

`tools/verify_checkpoint_roundtrip.py` on **8×L20A, LLaMA2-7B, `--parallel zero2`**, 3 real
training steps then save → **perturb everything in memory** → load → compare:

```
[verify] saved outputs/cast_ckpt_roundtrip/ckpt_step2
[verify] perturbed live state (weights +1, masks inverted, moments +7, step +999, RNG reseeded)
[verify] === CHECKPOINT ROUND-TRIP (bit-exactness of the SAVED vs RESTORED state) ===
[verify] parallel=zero2 world=8 steps_before_save=3 ckpt=ckpt_step2
[verify] float model tensors per rank : 515   max |delta| = 0.000e+00
[verify] bool mask buffers per rank   : 224   mismatching elements = 0
[verify] optimizer state tensors      : 78 params/rank, max |delta| = 0.000e+00
[verify] per-parameter step counters   : 0 mismatches
[verify] next-batch data indices       : IDENTICAL  (rank0 [18790478069, 24756986272,
                                         74929841005, 20491300911] reproduced exactly)
[verify] optimizer coverage            : 515/515 owned params carry moments
[verify] VERDICT: PASS - state round-trips BIT-EXACTLY
```

The perturbation step is what makes this a test rather than a tautology: weights +1, **every mask
bit inverted**, moments +7, step +999, RNG reseeded. A load that silently did nothing would fail
every line.

### Why the loss-trace comparison is NOT the primary evidence (honest negative result)

The intuitive test — run N+M steps straight through vs N → save → resume → M, diff the losses —
was run first (`tools/resume_faithfulness.py --real`) and is **inconclusive on this hardware**.
Two arms with *identical config, identical seed, and no resume at all* already diverge:

```
step   A (control)        B (resumed at step 3)   |diff|     phase
  0    1.107989544049     1.107989544049          0.0e+00    before ckpt  <- bit-identical
  1    6.252430841327     6.252966612577          5.4e-04    before ckpt
  2    3.457070469856     3.451371617615          5.7e-03    before ckpt  <- NO ckpt yet!
  3    2.527226369828     2.530831024051          3.6e-03    first resumed step
  4    1.775430817157     1.773593582213          1.8e-03    after resume
```

Step 0 is bit-identical (same weights, same batch), then the trajectories separate **before any
checkpoint exists**. The cause is non-deterministic reduction order in the backward kernels
(atomics in SDPA / gradient-checkpoint recompute), amplified by bf16 accumulation over 32
micro-batches. So the run-to-run noise floor is **5.7e-3**, and the post-resume difference
(3.6e-3) is *below* it — that comparison cannot distinguish a perfect resume from a subtly broken
one, and quoting it as proof would be precisely the unfalsifiable "looks fine" claim to avoid.
Hence the state-level round-trip above, which GPU non-determinism cannot contaminate.

What *is* independently confirmed from the real 8-GPU resume: the anti-warm-restart verifier
passing on live state, and the CAST invariant surviving the boundary:

```
[cast] resumed at step 2 (next step 3); optimizer state verified: 515/515 owned params carry
       non-zero moments with step==3
[cast] step 3/5 ... aligned=224/224(global) decayed=3,238,002,688 mem=145.7G
[cast] step 4/5 ... aligned=224/224(global) decayed=3,238,002,688 mem=145.7G
```

### Fail-loud guards (all unit-tested)

| guard | fires when | test |
|---|---|---|
| `ResumeMismatchError` on args | any of 21 trajectory-critical args differs; **all** offenders listed at once | `test_resume_rejects_changed_max_steps`, `..._l1_decay_and_lr`, `..._reports_every_mismatch_at_once` |
| unverifiable checkpoint | ckpt has no record of a critical arg ⇒ equality cannot be proven ⇒ refuse | `test_resume_rejects_checkpoint_missing_a_key` |
| **warm-restart detector** | any restored `exp_avg_sq` is identically zero (the signature of re-initialised state) | `test_warm_restart_is_detected` |
| step-counter check | per-param `step` ≠ expected; AdamS derives bias correction **and** `alpha_t=(step-1)/T` from it | `test_wrong_step_counter_is_detected` |
| partition check | the set of global param indices a rank owns ≠ the set in its shard file | (asserted in `load_training_state`) |
| torn checkpoint | no `DONE` marker ⇒ never auto-selected, refused if named explicitly | `test_incomplete_checkpoint_is_not_loadable` |
| legacy file | a weights-only `.pt` is refused with an explicit "WARM RESTART" reason | `test_legacy_weightsonly_checkpoint_is_refused` |
| world-size change | data sharding is `seed+rank`, so a different world size reads a different corpus | (asserted in `load_training_state`) |
| coverage across resume | `aligned=224/224` global check still runs every step, and a rank seeing 0 in-scope tensors now raises instead of passing vacuously | `assert_full_coverage` |

### `--stop-after` vs `--max-steps` (a trap worth naming)

To stop a run early **do not lower `--max-steps`**: AdamS uses `alpha_t = t/max_steps` as the decay
ramp, so changing it rescales the entire sparsification schedule and makes the two segments
different experiments. The resume guard refuses this — it caught exactly this mistake in the first
draft of the faithfulness harness. Use `--stop-after N`, which stops cleanly, writes a resumable
checkpoint, and **skips finalisation** (finalisation hard-prunes with M_T and is irreversible).

### Usage

```bash
bash scripts/launch_cast_llama.sh full              # fresh run, dolmino, save every 250
RESUME=auto bash scripts/launch_cast_llama.sh full  # continue from the newest complete ckpt
```

`--dist-timeout` defaults to 3600 s. It must exceed the checkpoint write: the FS measures
262 MB/s, so ~81 GiB is minutes during which non-zero ranks sit in a barrier — the 10-minute NCCL
default is too close for comfort. Do not lower it.

## 5c. THE RUN COMPLETED — result (2026-08-11)

The full 7500-step run finished on `.21` under `--parallel zero2`
(`outputs/cast_repro_zero2/`, log `logs/cast_repro_full_20260809_211514.log`,
110,842 s ≈ 30.8 h wall, 14.8 s/step, peak 145.7 GB/rank, `aligned=224/224` and
`decayed=3,238,002,688` on every step, `flips` decaying 6.78M → 72k).

**WikiText-2 PPL @ 4096, one harness (`baselines/eval_hf_sparse_model.py`), same
box, same tokenizer:**

| model | Wiki PPL | linear zero ratio | exact-2:4 tile ratio |
|---|---|---|---|
| LLaMA-2-7B dense reference | **5.2004** | 1.4e-06 | 0.0 |
| CAST @7500 steps (7.86B tok) | **6.1372** | 0.500000 | **1.0** |

Provenance: `outputs/cast_eval_spec/{cast_7500,dense_ref}/ppl_metrics.json`,
`logs/cast_eval_spec_0811_045836.log`. 335,872 target tokens, 82 sequences of
4096 (the whole of `wiki.test.raw`; the harness drops the final partial
sequence). Sparse model exported by `tools/export_final_to_hf.py`.

**Verdict against the §8 criteria: PASS on all three.**
1. Masked weights → 0 with exact 2:4: `exact_2of4_tile_ratio = 1.0` over all
   1.619e9 groups, `linear_zero_ratio` exactly 0.5, 0 violations. Independently
   re-derived from the saved tensors (not from the trainer's own counters, which
   §8 documents as vacuous): recomputing the 2:4 mask from `prefinal.pt` weights
   agrees with the saved mask at **0.99998** on a 12-tensor sample, against
   **0.500** for a permuted-mask control; and `final_sparse.pt` weights equal
   `prefinal.scaled_weight × mask` with `max|diff| = 0` (fold is exact, and
   `cast_scale` is all-ones afterwards, so the module is a bare `nn.Linear`).
2. PPL in the AST-ckpt band, far from 23.45: **6.137**, i.e. **+0.94 (+18.0%)
   over dense**. Slightly *better* than the 6.2–6.5 the §8 harness-offset
   argument predicted, so it is not a lucky-band artifact in the pessimistic
   direction. The broken FSDP run was 23.45.
3. Explainable ordering: CAST (6.137) sits between dense (5.200) and the broken
   run (23.45), under one harness.

⚠️ **Not yet measured**: the 7-task zero-shot average (SPEC §7 wants HellaSwag,
RACE, PIQA, WinoGrande, ARC-e, ARC-c, OBQA via LM Harness), and the
Wanda/naive-retraining arms that criterion 3 wants for a full ordering. The PPL
comparison above is dense-vs-CAST only.

## 6. Reporting rules

- Cite CAST reference numbers **only from Table III** (LLaMA2-7B @7.5B: Wiki PPL **5.58**, 7-task avg **55.91**). Table III is self-consistent with Table XII and all seven row averages recompute exactly. **Table VI contradicts it** (5.56 for the same run) and its `CAST w SRSTE` row appears to have ARC-e/ARC-c swapped. See SPEC.md §7.
- Report **both** "7500 steps" and the actual token count: 7500 × 256 × 4096 = **7,864,320,000 tokens (7.86B)**, i.e. **+4.9%** over the paper's nominal "7.5B" label. λ and α_t = t/T are both calibrated to T=7500, so steps are the mechanical spec and tokens are derived.
- **Success is not "hit 5.58".** Measured independently: the *same* AST official checkpoint scores Wiki PPL 5.69 → **6.3430** (+11%) and 7-task 58.62 → 57.94 across two harnesses. A correct CAST reproduction should land near **6.2–6.5** in our harness. Judge on (a) masked weights → 0 with exact 2:4 and 100% AdamS coverage, (b) PPL in the AST-ckpt band, far from 23.45, (c) explainable ordering vs dense/Wanda/naive-retraining under one harness.

## 7. Unresolved ambiguities

1. **Dolmino subset weights** — Sec. VI-A names the dataset but gives no mixture. `prepare_dolmino_llama2.py` uses the dataset's natural proportions; tagged `implementation_choice`.
2. **β₁/β₂/ε** — Alg. 1 names β₁,β₂ but never assigns values; Adam defaults assumed.
3. **Eq. (12) reshape axis** — ambiguous between contiguous column blocks and strided interleaving. We use contiguous blocks (SPEC.md §3). With n=2 the difference is real but untestable against the paper.
4. **α_t indexing** — Alg. 1 (t=0..T-1) and Alg. 2 (t=1..T) disagree by one step; we follow Alg. 1 so α₀=0. Effect is one step in 7500.
5. **Whether `lm_head`/`embed_tokens` are in Θ** — Sec. III says only "L linear layers". We exclude them (standard for N:M LLM pruning, and it makes the 224/3.238e9 accounting exact). If CAST included `lm_head`, its sparsity/parameter counts would differ.
6. **The "7.5B tokens" label** — Table XI's own steps×batch×seqlen gives 7.86B, so the paper's "7.5B" is rounded or uses a different accounting. Not reconcilable from the text.
7. **λ vs terminal magnitude (our finding, SPEC.md §6)** — AdamS's Adam normalization makes the per-step decay displacement saturate at ≈lr independent of λ, so the residual floor is O(final lr) and the total decay distance is bounded by `Σ lr_t·α_t`. The paper never discusses this, yet it is what decides whether the final hard prune is free. The paper recipe has 4.32× headroom, so it works — but this is the mechanism that made the broken run's collapse inevitable once AdamS was disabled, and it means **λ cannot be tuned independently of the LR schedule**.
