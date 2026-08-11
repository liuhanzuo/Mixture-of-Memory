# CAST reproduction spec — source level for every decision

Paper: **CAST: Continuous and Differentiable Semi-Structured Sparsity-Aware Training for Large Language Models**, arXiv:2509.25996v1.
Full text on disk: `/apdcephfs_wzc1/share_304376610/pighzliu_code/baselines/cast_repro_paper_refs/docs/cast_arxiv_2509.25996v1_fulltext.txt`; layout/PDF in the sibling `paper/`.

> ⚠️ Those are **not** under `baselines/cast_repro/` — this file previously pointed at a bare `docs/` and `paper/`, which do not exist here. Anyone verifying a `paper_explicit` tag against a dead path silently verifies nothing; that is most likely how the Appendix B/D misattribution in §1's LR row survived. Always cite the full path plus a line number.

Source levels:

| level | meaning |
|---|---|
| `paper_explicit` | stated in the paper; the citation (Eq./Table/Algorithm line) is given |
| `ast_code_inferred` | not in the paper; taken from the same group's AST code (`baselines/ast_official_clean`, commit 28bd277) |
| `implementation_choice` | in neither; our decision, and we say so |

---

## 1. Hyperparameters (LLaMA2-7B)

| item | value | level | citation |
|---|---|---|---|
| Learning rate | 2e-5 | `paper_explicit` | Table XI, LLaMA "2-7B" column |
| Decay coefficient λ | 4e-7 | `paper_explicit` | Table XI, "Decay Coefficient" |
| Global batch size | 256 | `paper_explicit` | Table XI, "Batch Size" |
| Sequence length | 4096 | `paper_explicit` | Table XI, "Seqlen"; Appendix B: "4096 for all LLaMA models during both training and evaluation" |
| Training steps | 7500 | `paper_explicit` | Table XI, "Training Steps" = 7.5k |
| Tokens trained | 7.5B (nominal) | `paper_explicit` | Table XI, "Tokens Trained" |
| Mask update period T₁ | 10 | `paper_explicit` | Sec. IV-A: "updated every T₁ = 10 iterations"; Appendix C: "computed every 10 batches" |
| Scaling groups n | 2 | `paper_explicit` | Sec. VI-A: "We select n = 2 for weight scaling module" |
| KL coefficient η | 1/3 | `paper_explicit` | Table XI, "Kl Coefficient"; Sec. VI-C2: "we adopt η = 1/3 ... for the LLaMA models" |
| Sparsity pattern | 2:4 | `paper_explicit` | Sec. III, Eq. (4) |
| Forward pass | dense throughout | `paper_explicit` | Sec. IV, Fig. 2 (right) |
| β₁, β₂ | 0.9, 0.999 | `implementation_choice` | Alg. 1 names β₁/β₂ but never gives values; Adam defaults |
| ε | 1e-8 | `implementation_choice` | Adam default |
| LR schedule | **`constant` at peak 2e-5 is what the code defaults to and what the live run uses** (see the ⚠️ box in §6). `cosine` to min_lr 2e-6 is available via `--lr-schedule cosine` | `implementation_choice` | **Table XI gives only the peak LR, and the paper never specifies a within-run schedule** (`grep -ci cosine` = 0 over the full text). `constant` is therefore a defensible literal reading, but it is OUR choice, not `paper_explicit` — and the code's "Appendix B" citation for it is wrong (the phrase is in Appendix D, about the scaling-law sweep). min_lr = lr/10 mirrors AST official `alpha_f=0.1` (`ast_code_inferred` for the 0.1 ratio) |
| Warmup | 375 steps (5% of 7500) **when `--lr-schedule cosine`; ignored under `constant`, which is the default** | `implementation_choice` | not in the paper |
| Weight decay (L2) | 0 | `implementation_choice` | Table XI lists none; CAST's regularizer is the selective L1 |
| Grad clip | 1.0 | `implementation_choice` | not in the paper |
| KL temperature | **1.0** | `paper_explicit` | Eq. (13) has **no** temperature term ⇒ T=1 is the literal reading |
| — AST-style variant | 2.0 with ×T² | `ast_code_inferred` | `ast_official_clean/sparse_modeling.py:240` hardcodes `temperature=2` and `*(temperature**2)`. Available via `--kl-temperature 2.0`; **not** the default |
| Master weight dtype | fp32 | `implementation_choice` | forced by numerics: λ=4e-7 is below bf16 resolution — see `tests/test_cast.py::test_bf16_swallows_lambda_fp32_does_not` |
| Parallelism | **DDP + `ZeroRedundancyOptimizer` (ZeRO-1) via `--parallel zero2`**; plain DDP via `--parallel ddp` no longer fits at 7B (see §5) | `implementation_choice` | the paper says nothing; what matters is that neither option shards the *parameter*, so mask↔weight alignment is preserved. **Never FSDP** — it flattens the Parameter and breaks alignment (§5) |
| Micro-batch | 1 (+grad accum) | `implementation_choice` | memory-driven |

## 2. AdamS (Algorithm 1, Eq. 7–8)

Implemented in `cast/adams.py`. Per in-scope scalar θ, citing Alg. 1 lines:

```
g_t     = ∇_θ L                                              (line 10)
μ_t     = β₁·μ_{t-1} + (1-β₁)·g_t                             (line 11)   raw gradient only
α_t     = t / T                                               (line 12)
μ̃_t     = (1-α_t)·μ_t + α_t·λ·sign(θ_{t-1})   if m_t = 0      (line 14)
        = μ_t                                  if m_t = 1      (line 16)
v_t     = β₂·v_{t-1} + (1-β₂)·μ̃_t²                            (line 18)
μ̂_t     = μ̃_t / (1-β₁ᵗ),   v̂_t = v_t / (1-β₂ᵗ)                (lines 19-20)
θ_t     = θ_{t-1} - γ_t·μ̂_t / (√v̂_t + ε)                      (line 21)
```

### Resolved ambiguities

**(a) Is `v_t` built from μ̃² for kept weights too?** — **Yes**, `paper_explicit`.
Alg. 1 line 18 sits *outside* the if/else of lines 13–17, and line 16 sets μ̃ = μ for kept weights, so the two readings coincide there. Sec. IV-A3 states the intent directly: "we apply the decay to the first-order momentum and use the resulting sum to compute the second-order moment". Note this is *not* textbook Adam (which uses g²) — using μ² for everything is CAST's third modification. Verified by `test_second_moment_uses_mu_tilde`.

**(b) `α_t` indexing: Alg. 1 says t = 0..T-1, Alg. 2 says t = 1..T.** — We use **Alg. 1's t = 0..T-1**, so α₀ = 0 and the first step is pure Adam with no decay. Rationale: Alg. 1 is the definition of AdamS itself, and α₀ = 0 is what makes lines 12–14 self-consistent (at t=0 the decay term vanishes). With PyTorch's 1-based `state['step']`, α = (step−1)/T. Verified by `test_adams_alpha_ramps_linearly_zero_to_one`.

**(c) `sign(0)`** — `torch.sign`, i.e. sign(0) = 0, `implementation_choice`. A weight already at exactly zero gets no decay kick. Verified by `test_adams_sign_at_zero_is_zero`.

**(d) μ_t accumulates the raw gradient, never the decayed value.** `paper_explicit` — line 11 is unconditional and Sec. IV-A3's whole point is decoupling: "the decay signal remains accurate and uninfluenced by historical information". The decayed μ̃ is used for the step and then discarded.

**(e) Which tensors are in Θ?** — the **224 in-block projections** (q,k,v,o,gate,up,down × 32 layers); `embed_tokens` and `lm_head` stay dense. `implementation_choice`: Sec. III says only "L linear layers" without enumerating them. This is what Wanda/SparseGPT/MaskLLM all do, and it makes the accounting exact:

```
per layer = 4·4096·4096 + 3·4096·11008 = 202,375,168
× 32      = 6,476,005,376 in-scope elements
÷ 2       = 3,238,002,688 masked (exact 2:4)   <-- hard runtime assertion
```

`cast_scale` (224 × 4096 × 2 ≈ 1.8M params) and norms get plain Adam, same LR (Table XI lists a single LR; no differential LR is claimed).

## 3. Mask (Eq. 6) and weight scaling (Eq. 11–12)

**Mask** — `nm_magnitude_mask` in `cast/sparse_linear.py`. Groups are contiguous along the input/column axis (Eq. 3). Eq. (6) defines keep as `|W| ≥ ξ` with ξ the 2nd-largest absolute value in the group; taken literally, an exact tie keeps 3 or 4 of 4 and violates Eq. (4). We use `topk(2)`, which keeps exactly N and breaks ties by index — the only reading consistent with Eq. (4). `implementation_choice` for tie-breaking only. Verified by `test_nm_mask_hand_computed`, `test_nm_mask_ties_still_exact`.

Mask dtype is `torch.bool`: Sec. IV-A's Remark budgets the mask at "1/32" of optimizer state (1 bit/param); a float mask would be 32× over budget (26 GB/rank at 7B).

**Refresh timing** — at the **top of step t, before the backward and before `optimizer.step()`** (Alg. 1 lines 6–8; Alg. 2 lines 8–10 precede the gradient at line 13 and the AdamS update at line 16). The old code refreshed *after* `optimizer.step()`, so step 0 ran with an all-ones mask (audit §4.5). Verified by `test_mask_refresh_before_step_and_every_T1`.

**Scaling** — `A^k ∈ R^{R_k×n}` init to ones (Eq. 11–12; Alg. 2 line 4). Element (r,c) is scaled by `A[r, c // (C/n)]`, implemented as `W.view(R,n,C/n) * A.unsqueeze(-1)` → `view(R,C)`. `implementation_choice` for the axis convention: Eq. (12)'s reshape to `(R·n)×(C/n)` is ambiguous between contiguous column blocks and strided interleaving; contiguous blocks are the natural reading of "partition each row into n groups". Verified by `test_scale_groups_axis_semantics`.

**Finalization** — Alg. 2 line 20 (prune with M_T) **then** line 21 (fold the scaling in). Folding is element-wise so it cannot resurrect a pruned entry; afterwards `cast_scale` is reset to 1 and the module is numerically a bare `nn.Linear`. Verified by `test_finalize_exact_2of4_and_scale_folded`.

## 4. Distillation (Eq. 13)

`L = η·L_kl + (1−η)·L_ce`, forward KL `D_KL(P_teacher ‖ P_student)`, teacher = the frozen dense model itself (Sec. IV-C "dense model as a self-teacher").

**Normalization** — the **convex** form is `paper_explicit` (Eq. 13 is literally `η·L_kl + (1−η)·L_ce`). This matters for the LR: Table XI's 2e-5 is only meaningful if the total loss has plain-LM-loss scale. The un-normalized alternative `CE + η′·KL` is equivalent with `η′ = η/(1−η) = 0.5` and `lr′ = lr·(1−η) = 1.333e-5`; `convex_to_unnormalised()` records this and `test_convex_unnormalised_equivalence` checks it. We use the convex form, so **lr = 2e-5 verbatim**.

**Intermediate-layer distillation is deliberately absent** — Sec. IV-C and Appendix H: hidden/attention-based losses (TinyBERT, MobileBERT, Sparse-Finetuning) are *worse* than plain KL (Table XV), so CAST uses logit KL only.

## 5. Why not FSDP (and why the live run is ZeRO-1, not plain DDP)

The previous attempt (audit §4.1) ran FSDP FULL_SHARD. FSDP packs `weight` and `mask` into a FlatParameter and slices them at different global offsets, so a rank's weight shard and mask shard are not element-aligned. The old optimizer set `mask = None` on mismatch and **silently ran vanilla Adam** — the selective L1 decay never executed on most tensors. Result: masked/kept magnitude ratio 0.294 (should be ≈0), only 21.5% of masked weights below 1e-4, Wiki PPL 23.4514 after the final prune.

Empirically confirmed by `tools/fsdp_misalignment_demo.py` on 4×L20A:

```
DDP  : 4/4 tensors element-aligned  -> weight.shape == mask.shape everywhere

FSDP FULL_SHARD, use_orig_params=True:
  pre-wrap shapes (rank0): ('0.q_proj', (512, 512), (512, 512))
  MISALIGNED 0.q_proj.weight:    weight shard (262144,) vs mask (512, 512)
  MISALIGNED 0.down_proj.weight: weight shard (0,)      vs mask (512, 512)
```

Under DDP nothing is sharded: `weight` and `mask` are full, same-shape, same-device tensors, so alignment holds structurally. AdamS additionally asserts it every step and raises `MaskCoverageError` rather than degrading.

**⚠️ 2026-08-09 — the live run is ZeRO-1, not plain DDP; the memory figure below is also wrong.** Plain DDP does not fit: the per-rank static budget is **131.8 GiB** (fp32 params 26.9 + grads 26.9 + exp_avg 26.9 + exp_avg_sq 26.9 + bool masks 6.5 + bf16 teacher 13.5 + fp32 master 4.2), and measured step-0 peak was **178.33 GiB** without checkpointing / **174.04 GiB** with `expandable_segments` — OOM by ~100 MiB on a 183 GiB L20A. (The "127.7 GiB measured peak" previously stated here was from a smaller configuration and does not apply.)

The run therefore uses **`--parallel zero2`**, which is **DDP + `torch.distributed.optim.ZeroRedundancyOptimizer(AdamS)` = ZeRO-1**: only the *Adam state* is sharded, and `_partition_parameters` assigns **whole Parameters** greedily and never slices them, so weight↔mask element alignment is structurally preserved exactly as under plain DDP. Measured: `adam_state = 7.0 G/rank` over 29 tensors vs 50.2 G unsharded; steady peak **145.7 GiB/rank**.

**The flag name `zero2` is a misnomer** — grads are reduced-and-freed by DDP's own bucketing rather than sharded, so the mechanism is ZeRO-1. Do not write "ZeRO-2" in the paper. (Real ZeRO-2 or ZeRO-3 would need FSDP, which is forbidden above.)

## 6. A mechanism the paper does not discuss (our finding)

AdamS is Adam-normalized, so in the decay-dominated regime (α→1, small gradient) μ̃ → α·λ·sign(θ) and v → (α·λ)², hence `μ̂/√v̂ → sign(θ)` and the per-step displacement **saturates at ≈lr regardless of λ**. Measured: 1.69×lr late in training (`test_terminal_magnitude_is_set_by_final_lr`).

Two consequences for the full run:

1. **The total decay distance is bounded by `Σ_t lr_t·α_t`.** If that is smaller than typical |W|, masked weights cannot reach zero however correct the code is, and the final prune collapses the model. For the paper recipe (2e-5→2e-6 cosine, 7500 steps, |W|≈0.0067 from audit §5) the budget is 0.0289 ⇒ **4.32× headroom**. `tools/decay_budget.py` computes this; run it before any long run.
2. **The residual floor is O(final lr)**, so a *decaying* LR is what drives masked weights to zero. min_lr = 2e-6 gives a floor ≈4e-6, safely below the 1e-4 target. A constant LR leaves every masked weight parked at ≈lr.

> **⚠️ 2026-08-09 — the live 7500-step run does NOT follow point 2, and this section was never updated to say so.**
>
> `cast/train_cast_llama.py` changed its `--lr-schedule` default to **`constant`** in commit `b4addd7`, and this file has had **zero commits since** (`git log b4addd7..HEAD -- SPEC.md` is empty). So the paragraph above is stale relative to the code, not a description of it. The run on `.21` (`outputs/cast_repro_zero2`, argv verified from `/proc/266500/cmdline`) is `--lr-schedule constant --lr 2e-5`; the `--min-lr 2e-6 --warmup 375` it also passes are **dead flags** under `constant` (`train_cast_llama.py:275-276` returns `args.lr` directly) yet still get recorded into `run_manifest.json`, which reads as though cosine were active.
>
> **The justification given in the code for `constant` is a misattribution.** `--lr-schedule`'s help text calls `constant` "paper-literal: Appendix B". In `docs/cast_arxiv_2509.25996v1_fulltext.txt` the phrase "consistent learning rate" occurs **once**, at line 1646, inside **Appendix D "Details on Scaling Law Experiments"** (header at line 1645) — not Appendix B (line 1335). The full sentence is: *"To ensure effective mask learning, we maintain a consistent learning rate for each model **and adjust the decay factor based on the training token budget**."* That is about holding LR fixed **across** the scaling-law's token-budget points, **not** about the within-run schedule. Separately, `grep -ci cosine` and `grep -ci 'warm.?up'` over the full text both return **0** — the paper genuinely never specifies a within-run schedule. So `constant` is a *defensible* reading of a silent paper, but it is an `implementation_choice`, **not** `paper_explicit`, and the Appendix B citation is wrong.
>
> Cost of the deviation, via `tools/decay_budget.py` and the 1.69×lr result above:
>
> | schedule | α-weighted decay distance | headroom vs \|W\|≈0.0067 | residual floor | terminal \|dw\| | margin to the 1e-4 target |
> |---|---|---|---|---|---|
> | `constant` (**what is running**) | 0.0750 | 11.19× | 2.0e-5 | 3.38e-5 | **2.96×** |
> | `cosine` (what this section prescribes) | 0.0289 | 4.32× | 2.0e-6 | 3.38e-6 | 29.6× |
>
> `constant` buys **2.6× more total decay distance** but gives up **~10× of terminal-magnitude margin** — and terminal magnitude is precisely the axis §6 exists to reason about. It is not obviously fatal (3.38e-5 still sits under the 1e-4 target, and the implied `ratio_mean ≈ 3.38e-5/0.0224 ≈ 0.0015` clears §8's <0.01), but it is also **not** what §6 prescribes, and Appendix C's claim that masked weights converge to zero with sparse-weight-ratio → 1 is unreachable under a constant LR: it can only settle at O(lr).
>
> The open empirical question is the heavy tail, not the mean: `max_masked_magnitude` has risen monotonically across all five DIAG points (1.06329 → 1.06725 at step 1250). A weight at 1.067 is ~14× beyond the 0.0750 reachable decay distance, so the final Alg. 2 line-20 hard prune may not be free. **Re-check at step 4000-5000** — if `max_masked_magnitude` is still climbing then, `constant` is doing real damage and the run should be redone under `cosine`.
>
> ⚠️ `lr_schedule` is in `RESUME_CRITICAL_ARGS` (`checkpoint.py:151`), so this **cannot** be changed by resuming an existing checkpoint — switching to `cosine` requires a fresh run with a new `--out`.
>
> Whatever the outcome, any reported result must state the schedule explicitly and tag it `implementation_choice`.

λ therefore controls *when* decay starts dominating the gradient, not the terminal magnitude. This is consistent with the paper's claim that λ is "robust and easy to tune" (Appendix B) and that λ should be "the same order of magnitude as the gradient g_t" (Sec. IV-A2).

## 7. Evaluation

- WikiText-2 perplexity at **4096** context, train and eval (Appendix B). The audit's first debug used 2048 — not comparable.
- Zero-shot: HellaSwag, RACE, PIQA, WinoGrande, ARC-e, ARC-c, OBQA via LM Harness (Sec. VI-A).
- **Reference numbers must come from Table III only**: CAST @7.5B = Wiki PPL **5.58**, **CAST-7** average **55.91**. Table III is internally consistent with Table XII's scaling row (2-7B @7.5B = 5.58) and all seven row averages recompute exactly. Table VI's "5.56" for the same run contradicts Table III, and Table VI's `CAST w SRSTE` row has ARC-e/ARC-c apparently swapped (41.89/76.39 vs 76.52/43.68 elsewhere) — do not cite Table VI.

  > ⚠️ **2026-08-11 correction — 55.91 is a CAST-7 average, NOT an "AST-7" average.** This line
  > previously mislabelled it while line 159 above lists the CAST seven. The two suites differ and
  > are not interchangeable:
  > * **CAST-7** (CAST Table III) = HellaSwag, RACE, PIQA, WinoGrande, ARC-e, ARC-c, OBQA
  > * **AST-7** (AST Table 2)   = BoolQ, RTE, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA
  >
  > They intersect in only **5** tasks: RACE/PIQA are CAST-only, BoolQ/RTE are AST-only. Comparing a
  > CAST-7 mean against an AST-7 mean is invalid. Use the Union-9 table
  > (`outputs/cast_eval_spec_union9/union9_four_arm_table.json`, 9 tasks x 4 arms, one harness),
  > which lets each subset mean be sliced from the same run.
  >
  > **Also: 55.91 is a PLAIN-ACC average.** Table III's row `[54.50, 40.48, 77.09, 68.27, 76.52,
  > 43.68, 30.80]` recomputes to 55.9057 exactly, and HellaSwag 54.50 / OBQA 30.80 are on the
  > plain-acc scale (acc_norm would read ~73 / ~40). AST-7 is plain acc too. So the mixed
  > acc_norm/acc convention behind our internal `zeroshot_avg_primary` **cannot** be compared to
  > either paper's headline: under the mixed map our CAST repro reads 59.27 (+3.36 "better" than
  > 55.91), but on the papers' own plain-acc convention it is 54.37 — **1.54 pp worse**.

## 8. Success criteria (not "hit 5.58")

Independently measured: the *same* AST official checkpoint scores AST-7 58.62→57.94 and Wiki PPL 5.69→**6.3430** (+11%) across two harnesses. So even a perfect CAST reproduction lands near ~6.2 in our harness, not 5.58. Judge instead on:

> ⚠️⚠️ **2026-08-11: the 6.3430 anchor above is measured at seqlen 2048, so it must NOT be compared
> with the 4096-context PPLs this SPEC mandates in §7.** Measured on `.21`, same harness
> (`baselines/eval_hf_sparse_model.py`), same 335,872 target tokens, same AST checkpoint
> (`models/AST-official-LLaMA2-7B-2of4`, exact-2:4 verified):
> * seqlen **2048** → **6.342995328699181** — reproduces the archived value *bit-identically*
>   (`rebuttal_artifacts/2026-07-27/ast_official/ppl_metrics.json`, whose own `"seqlen": 2048` field
>   confirms it), so seqlen is the sole cause of the gap.
> * seqlen **4096** → **5.9125** (`outputs/cast_eval_spec/ast_official/ppl_metrics.json`).
>
> **Consequence: the "+11% harness tax" framing is wrong, and so is the "~6.2 is the realistic
> target" reasoning built on it.** At the SPEC-mandated 4096 the AST checkpoint reads **5.9125**,
> which is *better* than our CAST reproduction's **6.1372** — the opposite of the direction implied
> when 6.3430 is placed next to 4096-context numbers. `appendix.tex:324` ("AST official deployable
> … 6.3430") has the same defect and must be re-measured at 4096 or explicitly labelled 2048.
>
> **★ MEASURED 2026-08-11 12:55 on `.21` (32 s, both sparse arms at 2048) — this settles which
> direction to normalise, and it is the OPPOSITE of "re-measure everything at 4096".**
>
> | arm | @2048 | @4096 | rel |
> |---|---:|---:|---:|
> | dense LLaMA-2-7B | 5.5637 | 5.2004 | +6.99 % |
> | CAST-repro (ours) | **6.5268** | 6.1372 | +6.35 % |
> | Wanda 2:4 | 12.4749 | 11.7733 | +5.96 % |
> | AST official | 6.3430 | 5.9125 | +7.28 % |
>
> The offset is **+6.0…+7.3 % on all four arms** with identical token counts (335,872 =
> 164×2048 = 82×4096), so it is purely the window. New provenance:
> `outputs/cast_eval_spec_ppl2048/{cast_7500,wanda}/ppl_metrics.json`.
>
> **Normalise to 2048, not 4096.** SparseForge's entire PPL column is 2048, and SparseForge's own
> headline is **6.2179**. At 2048 the ordering is
>
>     SparseForge 6.2179  <  AST official 6.3430  <  CAST-repro 6.5268
>
> i.e. **AST official is 0.18 better than our CAST reproduction, and SparseForge is best** — which
> is the honest and internally consistent story. Normalising to 4096 instead would require
> re-running dense, AST, Wanda, SparseGPT, ALPS, ELSA, ProxSparse *and SparseForge itself*, and
> would put CAST-repro's 6.1372 above SparseForge's 6.2179 — a 7 % protocol artifact beating a
> 2 % real claim.
>
> **The trap this closes:** pasting the 4096 value 6.1372 into the existing 2048 column would have
> shown our CAST reimplementation beating SparseForge on perplexity, purely from a longer context
> window. The artifact is **3.5×** the size of the headline PPL claim it would have contaminated.
>
> §8 criterion 3 (relative ordering) is unaffected: dense < CAST < Wanda holds in both conventions.
>
> ⚠️ §8.2's "~6.2–6.5 band" still does not state its own seqlen. Our 6.5268 sits at the band's
> upper edge under 2048; 6.1372 is below the lower edge under 4096. **Attach a seqlen to that band
> before using it as pass/fail.**

1. **Algorithmic correctness** — masked weights → 0; exact 2:4; AdamS actually running on the in-scope weights.

   ⚠️ **2026-08-09: do NOT use the per-step log counters as this evidence.** Adversarial testing on `.21` showed all three are weaker than they read:
   * `aligned=224/224` — `adams.py:184-201` compares only `mask is not None`, shape and device, **never contents**. Attaching a *deliberately wrong* mask (49% agreement) still reported `aligned: 1/1, coverage: 1.0`. The counter is blind to the exact failure it is named after.
   * `decayed=3,238,002,688` — **vacuous by construction**. `nm_magnitude_mask` uses `topk(2)+scatter_`, writing exactly 2 True per 4-group unconditionally; `mask.sum()` stays `numel/2` even for all-zero, all-one, 1e30, or **all-NaN** weights. The assertion cannot fail.
   * `check_mask_sync` — **dead code**. Its cross-rank checksum is `sum(mask.sum())`, which by the above is always `numel/2`; two masks agreeing on only 50.5% of entries both checksum to the same value and pass `all_reduce(MIN)`. Ranks could decay disjoint weight sets undetected.

   What *is* load-bearing: the global tensor/element counts under zero2's SUM-reduce (224 / 6,476,005,376), the `cast_scope == 0` guard, and — for alignment specifically — **recomputing the 2:4 mask from the saved weights in a checkpoint**. That probe gave 0.999745-0.999979 agreement on an 8-tensor sample of `ckpt_step1250`, against 0.559 for a deliberately mismatched control. Use that, not the log line.

   The strongest *cheap* signal that AdamS is running at all is the DIAG pair: `masked_mean_magnitude` must fall while `kept_mean_magnitude` holds (observed 0.00740 → 0.00344 vs 0.02233 → 0.02243). Vanilla Adam moves both.
2. **PPL in the same band as the AST official ckpt under our harness** (~6.2–6.5), far from 23.45.
3. **Explainable relative ordering** vs dense / Wanda / naive-retraining under one harness.

## 9. Union-9 four-arm zero-shot table (2026-08-11, node .21)

Provenance: `outputs/cast_eval_spec_union9/` (per-arm `zeroshot_union9.json` +
`union9_four_arm_table.json`); driver `scripts/_union9_eval_spec_21.sh`; aggregator
`baselines/cast_repro/tools/aggregate_zeroshot_union9.py`. One harness for all four arms
(lm-eval 0.4.8, bf16, `add_bos_token=False`, no chat template, `num_fewshot=0`,
`batch_size auto`→64, `seed 0`); only `pretrained` differs.

**Why Union-9 exists**: CAST-7 and AST-7 share only 5 tasks, so a CAST-7 mean can never be compared
to an AST-7 mean. Running all 9 lets both subset means be sliced from the *same* numbers.

Primary metric map (identical across arms, asserted in the aggregator): `acc_norm` for
hellaswag/arc_easy/arc_challenge/openbookqa; `acc` for piqa/winogrande/race/**boolq**/**rte**.
BoolQ/RTE are binary classification / entailment, where option length is not a confound, so
`acc_norm` is meaningless — the harness emits none for them.

| task (n) | dense | CAST@7500 | Wanda | AST official |
|---|---|---|---|---|
| boolq (3270) | 77.7676 | 69.2355 | 68.1957 | 72.9052 |
| rte (277) | 63.1769 | 74.7292 | 53.4296 | 66.4260 |
| hellaswag\* (10042) | 75.9709 | 72.9337 | 55.1384 | 72.7146 |
| race (1045) | 39.7129 | 39.5215 | 35.4067 | 39.6172 |
| piqa (1838) | 77.8564 | 76.3330 | 70.2938 | 76.9859 |
| winogrande (1267) | 69.2976 | 66.2983 | 62.9834 | 67.3244 |
| arc_easy\* (2376) | 74.6633 | 74.2424 | 57.1128 | 71.3805 |
| arc_challenge\* (1172) | 46.2457 | 45.7338 | 31.7406 | 42.4061 |
| openbookqa\* (500) | 44.2000 | 39.8000 | 35.8000 | 40.8000 |
| **Union-9 mean** | **63.2101** | **62.0919** | **52.2335** | **61.1733** |
| CAST-7 slice | 61.1353 | 59.2661 | 49.7822 | 58.7470 |
| AST-7 slice | 64.4746 | 63.2818 | 52.0572 | 61.9938 |
| WikiText-2 PPL @4096 | 5.2004 | 6.1372 | 11.7733 | **5.9125** |

\* = `acc_norm`. **Plain-acc slice** (the convention both papers use, for paper-facing comparisons):
Union-9 59.5625 / 58.2837 / 49.4569 / 57.7540; CAST-7 56.4454 / 54.3698 / 46.2123 / 54.3507;
AST-7 59.7847 / 58.3856 / 48.4873 / 57.5976.

**Harness integrity**: re-running the 7 previously-published tasks reproduced the on-disk
`zeroshot_metrics.json` for all three existing arms with worst |Δ| = **0.000000 pp** (3 arms × 7
tasks × {acc, acc_norm}) — bit-identical, so the new BoolQ/RTE cells are on the same footing as the
old ones. The three pre-existing `zeroshot_metrics.json` / `ppl_metrics.json` files were not
modified; BoolQ+RTE were added as `zeroshot_boolq_rte.json`.

**External validation**: our dense arm reproduces AST Table 2's dense LLaMA-2-7B row to
**0.004 pp** on the AST-7 plain-acc mean (59.7847 vs the paper's 59.78; mean per-task |Δ| 0.21 pp),
which independently validates the whole harness including the new BoolQ/RTE cells. Our CAST
reproduction lands **1.54 pp below** CAST Table III's 55.91 on the same plain-acc convention.

A **second, independent** confirmation that the source convention is plain acc, via the AST official
checkpoint as a cross-harness anchor. §8 records that the same official ckpt scores AST-7 58.62 →
**57.94** across two harnesses. Ours, on that same ckpt:

| our convention, AST-official AST-7 | value | vs 57.94 |
|---|---:|---:|
| primary-metric mix (`acc_norm` where available) | 61.9938 | **+4.05 pp** |
| **plain `acc`** | **57.5976** | **−0.34 pp** ✅ |

Two anchors of different kinds — a dense row from AST's own table, and a fixed sparse checkpoint
measured under two harnesses — both land within 0.34 pp under plain acc and 4 pp off under the mixed
convention. The convention question is settled empirically, not by preference.

**Why this matters beyond bookkeeping**: MAIN reported on 2026-08-11 that our CAST reproduction
(59.27) *beat* CAST's paper (55.91) and began constructing an explanation for the "anomaly". Both
inputs were wrong — the mixed `acc_norm` convention inflates the average (HellaSwag alone moves
+18.8 pp: 53.92 acc → 72.93 acc_norm), and `SPEC.md` itself mislabelled 55.91's task set. The two
errors compounded in the same direction and manufactured a 3.36 pp "win" out of a real 1.54 pp
shortfall. **Never compare an `acc_norm`-mixed average to a published number.** The mixed convention
is valid only for internal arm-vs-arm ordering, where all arms share it.

**The headline is a tie, not a win**: on the CAST-7 plain-acc slice our CAST repro scores 54.3698 vs
the AST official checkpoint's 54.3507 — a **+0.02 pp** difference. On Union-9 primary the gap is
+0.92 pp, but leave-one-task-out shows it is carried *entirely by RTE*: dropping RTE collapses the
gap to **−0.0045 pp**, while dropping any other task leaves it at +0.62…+1.49 pp. RTE is the
smallest task (n=277, worst-case stderr 3.0 pp) and behaves anomalously — CAST scores 74.73 there,
**11.55 pp above dense** (McNemar exact p=0.00053), which is not a credible sparsification gain;
Wanda's 53.43 is at the 52.71% majority-class floor. Per-task McNemar (CAST vs AST, paired):
significant on rte (+23, p=0.017) and arc-nothing else in CAST's favour, while AST significantly
*wins* boolq (−120, p<1e-5) and hellaswag (−59, p=0.024). **Do not claim our CAST reproduction beats
the AST checkpoint.** Report the tie and footnote RTE.
