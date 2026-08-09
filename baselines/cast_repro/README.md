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
tests/test_cast.py   22 unit tests (all passing, output below)
tools/
  smoke_alignment.py        real-scale (224-tensor) alignment proof, pure torch
  fsdp_misalignment_demo.py empirical proof of the old FSDP bug
  integration_tiny.py       end-to-end loop test with a tiny stub model
  decay_budget.py           pre-flight feasibility check
  throughput_probe.py       measured step time / wall-clock projection
  diagnose_checkpoint.py    run the audit metric on a checkpoint
  prepare_dolmino_llama2.py PRIMARY data path (script only, NOT executed)
scripts/launch_cast_llama.sh
```

---

## 1. The six audit failures, and what fixes each

Reference: `Mixture-of-Memory/SparseForge_Data/docs/CAST_REPRODUCTION_AUDIT.md`.

| # | Audit finding | Fix here | Evidence |
|---|---|---|---|
| **1** | **[fatal]** FSDP sliced `weight` and `mask` at different FlatParameter offsets; old code hit unequal numel → `mask = None` → **silent vanilla Adam**, so selective L1 decay never ran. Ratio 0.294, Wiki PPL 23.4514. | **Plain DDP**, nothing sharded: `weight`/`mask` are full same-shape same-device tensors. `mask` is a **buffer**, so it is never packed into a FlatParameter. AdamS **raises `MaskCoverageError`** on any missing/misshaped mask — silent fallback is now impossible. | `tools/fsdp_misalignment_demo.py` reproduces the bug (FSDP gives `(262144,)` and even `(0,)` weight shards against a `(512,512)` mask); `smoke_alignment.py` shows 224/224 aligned under DDP. Unit tests `test_missing_mask_raises`, `test_shape_mismatch_raises`. |
| **2** | **[fatal]** Paper uses Dolmino-Mix-1124; old run used C4. | Unresolved by code — it is a data-availability problem. See §4; the honest options are spelled out and the fallback is explicitly *not* callable a paper reproduction. | `tools/prepare_dolmino_llama2.py` (PRIMARY path, written, **not run**) |
| **3** | KL temperature: Eq. 13 has none, old code used T=2 with ×T². | Default **T = 1.0** (paper-literal). T=2 available as a named variant via `--kl-temperature 2.0`, never silently. | `test_temperature_default_is_paper_literal` |
| **4** | LR schedule not specified by the paper but presented as config. | cosine → `min_lr` 2e-6 + 375-step warmup, tagged **`implementation_choice`** in SPEC.md §1. Table XI's only LR claim (peak 2e-5) is used verbatim. | SPEC.md §1 |
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

22/22 passed
```

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

## 4. Data — FALLBACK for now, PRIMARY not executed

The paper uses **Dolmino-Mix-1124** with LLaMA-2 (Sec. VI-A). What is actually on disk:

| path | tokens | tokenizer | verdict |
|---|---|---|---|
| `data/dolmino-mix-1124-llama3/` | 469B | **Llama3-8B, vocab 128000** | **unusable** — LLaMA-2 has vocab 32000 and the id spaces are unrelated |
| `data/dolmino-flan-heavy/` | 499.5M | Llama2-7b, vocab 32000 | right tokenizer, right *source*, but 16× too small for 7.86B and a **FLAN-heavy custom mix** (38.9% FLAN), not the Dolmino default proportions |
| `data/c4_llama/` | 21.7B | Llama2-7b | usable, 2.76 epochs at 7.86B — **the fallback** |
| `data/dolmino-mix-1124-raw/` | — | — | **does not exist**; raw download required |

**Recommendation: run FALLBACK (C4) now, and label it precisely.** Reasons:

1. It unblocks the baseline immediately and is a *controlled* comparison: C4 is what the previous run used, so a C4 result isolates the algorithmic fix (0.294 → ~0 masked ratio) from the data change. Changing both at once would confound the one thing we most need to verify.
2. It is defensible on its own terms — the paper itself uses C4 for OPT/GPT-2 "to remain consistent with their original pretraining data" (Sec. VI-A), and C4 is closer to LLaMA-2's actual pretraining mix than Dolmino is.
3. Dolmino is a deliberately *stronger*, knowledge-dense annealing corpus. Table IV shows LLaMA2-7B sparse **beating** dense on MMLU (45.74 → 52.34), which the paper attributes to "a pretraining corpus more focused on knowledge-intensive tasks". A C4 run will therefore **understate** CAST on knowledge benchmarks — so as a *baseline for SparseForge* it is conservative in the safe direction (it does not flatter our own method by handicapping CAST on perplexity, which is the headline metric, but it does mean MMLU-style gains will not reproduce).

**Mandatory labelling if FALLBACK is used** — this distinction matters a lot to a reviewer:

> CAST (our paper-based reimplementation), **controlled C4 reimplementation** — trained on C4 rather than the paper's Dolmino-Mix-1124 because a LLaMA-2-tokenized Dolmino corpus of sufficient size was unavailable. **This is not a paper-setting reproduction**, and knowledge-intensive results are expected to be lower than the paper's.

For the PRIMARY path, `tools/prepare_dolmino_llama2.py` is written and dry-run-checked but **deliberately not executed**:

```
python tools/prepare_dolmino_llama2.py --stage plan       # prints the accounting
python tools/prepare_dolmino_llama2.py --stage download --raw-dir data/dolmino-mix-1124-raw
python tools/prepare_dolmino_llama2.py --stage tokenize  --out-dir data/dolmino-mix-1124-llama2 \
       --target-tokens 9000000000 --workers 64
```
ETA ≈ **2–4 h wall, download-dominated** (~35–40 GB compressed; tokenization of 9B tokens is only ~20–40 min given this box did 469B tokens in 3306 s per the existing `metadata.json`). Output ≈ 18 GB uint16.

## 5. What is still needed for the full 7500-step run

**Blocker: `transformers` is not installed on `.21`.** Bare env has only `/opt/conda/envs/torch-base/bin/python` (py3.14, torch 2.13.0, numpy 2.5.1); `transformers`/`datasets`/`safetensors` are all `ModuleNotFoundError`. Per instructions I did **not** install anything — the real-model smoke and the full run both need approval:

```bash
# on .21, pinned as instructed
/opt/conda/envs/torch-base/bin/pip install 'transformers==4.57.6' 'datasets==2.21.0' safetensors
```
(Note: `transformers` also needs `safetensors` to read `models/Llama--Llama2-7b/*.safetensors`. The pin on `datasets==2.21.0` matters — a bare install pulls 5.0.1, whose cache layout differs from the existing 2.x cache and would trigger a full re-download.)

Then, in order:

```bash
# 1. pre-flight feasibility (no GPU)
python tools/decay_budget.py --steps 7500
#    -> HEADROOM 4.32x, residual floor 2.0e-06, VERDICT: OK

# 2. real-model smoke, <=50 steps  (the remaining unproven step)
bash scripts/launch_cast_llama.sh smoke
#    expect: "aligned=224/224 ... decayed=3,238,002,688" and finite loss

# 3. full run
DATA=data/c4_llama bash scripts/launch_cast_llama.sh full          # FALLBACK
# or, after the PRIMARY data build:
DATA=data/dolmino-mix-1124-llama2 DATA_DTYPE=uint16 bash scripts/launch_cast_llama.sh full

# 4. verify the fix actually took, on the pre-finalization checkpoint
python tools/diagnose_checkpoint.py --ckpt outputs/cast_repro_ddp/prefinal.pt
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
