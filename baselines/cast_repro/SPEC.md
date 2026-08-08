# CAST reproduction spec — source level for every decision

Paper: **CAST: Continuous and Differentiable Semi-Structured Sparsity-Aware Training for Large Language Models**, arXiv:2509.25996v1.
Full text on disk: `docs/cast_arxiv_2509.25996v1_fulltext.txt`; layout/PDF in `paper/`.

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
| LR schedule | cosine to min_lr 2e-6 | `implementation_choice` | **Table XI gives only the peak LR.** min_lr = lr/10 mirrors AST official `alpha_f=0.1` (`ast_code_inferred` for the 0.1 ratio) |
| Warmup | 375 steps (5% of 7500) | `implementation_choice` | not in the paper |
| Weight decay (L2) | 0 | `implementation_choice` | Table XI lists none; CAST's regularizer is the selective L1 |
| Grad clip | 1.0 | `implementation_choice` | not in the paper |
| KL temperature | **1.0** | `paper_explicit` | Eq. (13) has **no** temperature term ⇒ T=1 is the literal reading |
| — AST-style variant | 2.0 with ×T² | `ast_code_inferred` | `ast_official_clean/sparse_modeling.py:240` hardcodes `temperature=2` and `*(temperature**2)`. Available via `--kl-temperature 2.0`; **not** the default |
| Master weight dtype | fp32 | `implementation_choice` | forced by numerics: λ=4e-7 is below bf16 resolution — see `tests/test_cast.py::test_bf16_swallows_lambda_fp32_does_not` |
| Parallelism | plain DDP | `implementation_choice` | the paper says nothing; DDP is required for mask↔weight alignment (§4 below) |
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

## 5. Why plain DDP, not FSDP

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

Cost: DDP does not shard optimizer state, so per-rank memory is ~128 GiB (fp32 params 26.9 + grads 26.9 + exp_avg 26.9 + exp_avg_sq 26.9 + bool masks 6.5 + bf16 teacher 13.5). Measured peak **127.7 GiB** on an L20A (183 GiB) — fits with headroom, but **not** on an 80/97 GiB card. Adding nodes does not reduce per-card memory.

## 6. A mechanism the paper does not discuss (our finding)

AdamS is Adam-normalized, so in the decay-dominated regime (α→1, small gradient) μ̃ → α·λ·sign(θ) and v → (α·λ)², hence `μ̂/√v̂ → sign(θ)` and the per-step displacement **saturates at ≈lr regardless of λ**. Measured: 1.69×lr late in training (`test_terminal_magnitude_is_set_by_final_lr`).

Two consequences for the full run:

1. **The total decay distance is bounded by `Σ_t lr_t·α_t`.** If that is smaller than typical |W|, masked weights cannot reach zero however correct the code is, and the final prune collapses the model. For the paper recipe (2e-5→2e-6 cosine, 7500 steps, |W|≈0.0067 from audit §5) the budget is 0.0289 ⇒ **4.32× headroom**. `tools/decay_budget.py` computes this; run it before any long run.
2. **The residual floor is O(final lr)**, so the LR *must* decay. min_lr = 2e-6 gives a floor ≈4e-6, safely below the 1e-4 target. A constant LR would leave every masked weight parked at ≈lr.

λ therefore controls *when* decay starts dominating the gradient, not the terminal magnitude. This is consistent with the paper's claim that λ is "robust and easy to tune" (Appendix B) and that λ should be "the same order of magnitude as the gradient g_t" (Sec. IV-A2).

## 7. Evaluation

- WikiText-2 perplexity at **4096** context, train and eval (Appendix B). The audit's first debug used 2048 — not comparable.
- Zero-shot: HellaSwag, RACE, PIQA, WinoGrande, ARC-e, ARC-c, OBQA via LM Harness (Sec. VI-A).
- **Reference numbers must come from Table III only**: CAST @7.5B = Wiki PPL **5.58**, AST-7 average **55.91**. Table III is internally consistent with Table XII's scaling row (2-7B @7.5B = 5.58) and all seven row averages recompute exactly. Table VI's "5.56" for the same run contradicts Table III, and Table VI's `CAST w SRSTE` row has ARC-e/ARC-c apparently swapped (41.89/76.39 vs 76.52/43.68 elsewhere) — do not cite Table VI.

## 8. Success criteria (not "hit 5.58")

Independently measured: the *same* AST official checkpoint scores AST-7 58.62→57.94 and Wiki PPL 5.69→**6.3430** (+11%) across two harnesses. So even a perfect CAST reproduction lands near ~6.2 in our harness, not 5.58. Judge instead on:

1. **Algorithmic correctness** — masked weights → 0; exact 2:4; AdamS on 100% of in-scope weights (asserted every step, expected 3,238,002,688 decayed elements).
2. **PPL in the same band as the AST official ckpt under our harness** (~6.2–6.5), far from 23.45.
3. **Explainable relative ordering** vs dense / Wanda / naive-retraining under one harness.
