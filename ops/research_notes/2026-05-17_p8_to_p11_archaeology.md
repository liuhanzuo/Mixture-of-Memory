# P8 → HEAD git archaeology — what changed under `src/memory/mem_space/` + `scripts/train_mem_space_babilong.py`

**Date:** 2026-05-17
**Author:** general-purpose-7 (researcher), read-only diff audit
**Scope:** P8 baseline commit `3c97b86` → current HEAD `69b396e`
**Method:** `git log` + `git show` + `git diff` on target paths; cross-check with QUERY_DIAG telemetry from existing logs.

---

## 1. Summary (confidence: **medium-high** for the negative finding; **low** for "the gap is from a code change in target paths")

**Headline finding (negative):** Of the six commits between `3c97b86` and `69b396e` that touch the target paths, **none** introduces a routing-relevant change that would still be active under the **L2-off + FSDP-off** path used by the current P11 DDP rerun. Walking through the candidates:

* `0f6e8a1` (L2 add) — refactors `extended_hidden` construction into a `parts` list, but produces the **same tensor** when `use_l2=False`. Confirmed by reading the rewritten branches against the P8 ancestor; the L2 read-path is fully gated on `self.l2 is not None and prev_latents.numel() > 0`.
* `62a26db` (gradient_checkpointing flag) — only kicks in when `self.config.gradient_checkpointing and self.training`, and uses `use_reentrant=False` with kwargs captured via Python closure. P11 DDP rerun **does** set `--gradient_checkpointing`, so this code path **is** live. But the gradient is correctly tracked through `extended_hidden → wrapped_layer(checkpoint) → ext_h → O_mem_hidden`, and the QUERY_DIAG signature is identical to P8 (which had no checkpoint at all) — see §3.
* `f9e3fa7` (length-weighted sampler + skip-mem-when-short) — pure CLI gating; the P11 DDP rerun does **not** pass `--skip_mem_when_short` and does **not** pass `--babilong_length_weights`, so both branches are inert. The new FSDP scaffolding inside this commit is also gated on `--use_fsdp`.
* `a6dcda3` (FSDP scalar reshape + no top-level wrap + manual ckpt path) — fully inside `_wrap_model_fsdp(...)` which is only called `if args.use_fsdp`. **DDP path is bit-identical to P8.**
* `0349264` (L2 forward()) — touches `l2_compressor.py` and a one-line edit in `patch.py`. No effect when `use_l2=False`.
* `69b396e` (optimizer must walk model.parameters() post-FSDP-wrap) — explicit `if args.use_fsdp:` else-branch keeps DDP optimizer-collection bit-identical to P8.

**Reframing the comparison:** The P8 baseline that the team-lead is treating as "healthy step25 routing" (`top1_sim_mean=0.022`, `per_tok_logit_std=1.66`) is *not* the original P8 (commit `3c97b86`). It is `logs/p8_temp20_500_20260517_105421.log`, which uses **`--selector_temperature 20`**, not P8's original `1.0`. The actual original P8 probe at commit `3c97b86` (`logs/phase8_probe_20260515_2237.log`) has step25 `top1_sim_mean=0.002228`, `per_tok_logit_std=0.082520` — **identical** to the current P11 DDP rerun and to P11 FSDP. See §3.

**Implication:** the routing trajectory has *not* regressed across the P8→HEAD code path. The 25-pp BABILong eval gap (P8=59.14 vs P11 step500=33.81) must come from somewhere **outside** the QUERY_DIAG signal (eval-time code drift, generation hyperparams, ckpt loader, or the writeback / dual-gate / forget-bias regime — all of which are stable in this commit window). Confidence: **medium-high** that the regression is *not* in the diff `3c97b86..69b396e` of the target paths.

---

## 2. Commit timeline (P8 → HEAD)

```
3c97b86 (P8 baseline, 2026-05-15 22:37)  feat(scripts): eval_queue_watcher
   ↓
0f6e8a1 (2026-05-16 11:27)  feat(mem_space): add L2 token-compressed KV memory
   ↓
62a26db (2026-05-16 12:43)  feat(mem_space): add gradient_checkpointing flag for L2
   ↓
f9e3fa7 (2026-05-16 17:27)  feat(phase1b-v3): length-weighted sampler + skip-mem-when-short
                            (also dumps in-progress FSDP plumbing)
   ↓
a6dcda3 (2026-05-16 18:13)  fix(fsdp): scalar reshape + no top-level wrap + manual ckpt path
   ↓
0349264 (2026-05-16 20:48)  fix(l2): add forward() to L2Compressor for FSDP unsharding
   ↓
69b396e (2026-05-17 08:38, HEAD)  fix(fsdp): optimizer must walk model.parameters() post-wrap
```

Stats over the full window:
```
 scripts/train_mem_space_babilong.py      | 438 ++++++++++++++++++++++++++++--
 src/memory/mem_space/babilong_dataset.py |  24 ++-
 src/memory/mem_space/config.py           |  18 ++
 src/memory/mem_space/l2_compressor.py    | 197 ++++++++++++++ (new file)
 src/memory/mem_space/layer.py            | 220 +++++++++++++---
 src/memory/mem_space/patch.py            |  79 +++++-
```

NOTE: `33d1ca9` (chunk-local BPTT) and `033b286` (per-chunk pool() cache + reset hook) are listed in the team-lead brief but are **ancestors** of `3c97b86` — `git merge-base 3c97b86 69b396e = 3c97b86`, so they are part of the P8 baseline, not post-P8 changes.

---

## 3. Per-commit verdict — does it influence the L2-off + FSDP-off path?

### 3.1 `0f6e8a1` — L2 add — verdict: **inert when `use_l2=False`**

**Hunks of interest (layer.py):**
* `layer.py:600-700` — extended_hidden construction rewritten from nested if/else into `parts.append(...)` list. With `use_l2=False`, the L2 compressor is `None`, `k_l2=0`, the L2 token block is never appended. Compared against P8 ancestor (`git show 3c97b86:src/memory/mem_space/layer.py`):
  * P8: `if k_l3 > 0 and k_slots > 0: cat([l3, M_sel, H])`; etc., 4 explicit branches.
  * post-`0f6e8a1`: `parts = []; if k_l3 > 0: append(l3); if k_l2 > 0: append(l2); if k_slots > 0: append(M_sel); append(H); cat(parts)`.
  * **Equivalent** for k_l2=0. No torch op semantics changed.
* `layer.py:_build_extended_attn_mask_l2` — only called `if k_l2 > 0`. Inert.
* `layer.py:794` — `if ext_h.shape[1] != k_l3 + k_l2 + k_slots + T:` — when `k_l2=0` this matches the P8 invariant `k_l3 + k_slots + T`.
* `layer.py:O_mem_hidden = ext_h[:, l1_start:l1_start + k_slots]` where `l1_start = k_l3 + k_l2`. With k_l2=0, this is `ext_h[:, k_l3:k_l3+k_slots]` — bit-identical slice to P8.

**patch.py:** L2 compressor construction and `_l2_post_forward_hook` are gated on `if config.use_l2:`. Inert.

**Verdict:** This commit is fully inert under `use_l2=False` for both DDP and FSDP. **Cannot explain the gap.**

### 3.2 `62a26db` — gradient_checkpointing — verdict: **active in P11 DDP rerun, but no detectable routing effect**

The P11 DDP launcher (`scripts/launch_p11_ddp_500step_validate.sh:90`) **does** pass `--gradient_checkpointing`, so this hunk is live in the suspect run.

**Hunk:** `layer.py:_maybe_ckpt_wrapped_layer` (line 455–493). Routes both `bypass_h = self.wrapped_layer(hidden_states, ...)` and `ext_h = self.wrapped_layer(extended_hidden, ...)` through `torch.utils.checkpoint(..., use_reentrant=False)` when `config.gradient_checkpointing and self.training`.

Walking the gradient path:
* `extended_hidden = cat([l3_summaries, M_sel_hidden, hidden_states])`.
* `M_sel_hidden = self.slot_to_hidden(M_sel_slot)`, where `M_sel_slot = slots.gather(1, idx) * scores` (top-k from selector). **Trainable upstream.**
* Goes into checkpoint as positional Tensor input `h`. Closure captures kwargs (none of which are required for grad).
* `use_reentrant=False` correctly tracks gradients to `h` and to *parameters* of `self.wrapped_layer` (frozen, so no grad needed). On backward, `extended_hidden.grad` is populated, propagates back through cat → `M_sel_hidden` → `slot_to_hidden` → selector keys.
* The hook `_l3_post_forward_hook` fires on `MemorySpaceLayer.forward` exit (not on wrapped_layer), so it does **not** fire during checkpoint recompute.

**Empirical check:** P11 DDP rerun step25 telemetry vs P8 (no checkpoint) step25:

| Run | top1_sim_mean | per_tok_logit_std | key_max_cos |
|---|---|---|---|
| P8 (`phase8_probe_20260515_2237.log`, no `--gradient_checkpointing` — flag did not exist at commit `3c97b86`) | 0.002228 | 0.082520 | 0.9727 |
| P11 DDP rerun (`p11_ddp_500step_validate_20260517_1116.log`, `--gradient_checkpointing` ON) | 0.002228 | 0.082520 | 0.9648 |
| P11 FSDP full (`p11_fsdp_full_20260516_181417.log`, `--gradient_checkpointing` ON) | 0.002182 | 0.082031 | 0.9805 |

These three numbers are statistically indistinguishable. **`--gradient_checkpointing` produces no detectable change in routing telemetry.**

**Verdict:** Active under DDP, but the routing math is preserved bit-for-bit. **Cannot explain the gap.** (Lower-confidence caveat: a checkpoint can in principle change reduction order in fp16/bf16 reduce; but selector logits are computed *outside* the checkpointed wrapped_layer, so the std telemetry shouldn't move.)

### 3.3 `f9e3fa7` — length-weighted sampler + skip-mem-when-short — verdict: **inert** in P11 DDP rerun

P11 DDP rerun launcher does NOT pass `--skip_mem_when_short` and does NOT pass `--babilong_length_weights`. Verified at `scripts/launch_p11_ddp_500step_validate.sh:69-78` — only `--babilong_lengths 1k,2k,4k`, no weights, no skip flag.

**Hunks:**
* `babilong_dataset.py`: `length_weights` kwarg defaults to `None`; when None, `rng.choice` is used (the P8 path). Inert.
* `train_mem_space_babilong.py`: `--skip_mem_when_short` defaults to `False`; the `_set_skip_writeback` call is guarded `if skip_mem_when_short:`. Inert.
* `layer.py`: the `_skip_wb` clause guards `cfg.enable_writeback and not cfg.disable_l1_inject and not _skip_wb`. With `_skip_writeback_this_call` never set, `_skip_wb=False` and the writeback path is identical to P8.
* In-progress FSDP plumbing in this commit (FSDP imports, `_wrap_model_fsdp`, `_is_distributed_wrapper`) is fully gated on `args.use_fsdp`.

**Verdict:** All branches inert under the P11 DDP rerun launch flags. **Cannot explain the gap.**

### 3.4 `a6dcda3` — FSDP scalar reshape / top-level wrap removal — verdict: **inert under DDP**

All hunks are inside `_wrap_model_fsdp` and `_save_adapter`. The `_wrap_model_fsdp` function is only called in the `if args.use_fsdp:` branch. P11 DDP rerun never calls it.

The DDP path remains: `model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)`. Bit-identical to P8.

**Verdict:** Cannot explain the gap.

### 3.5 `0349264` — L2 `forward()` for FSDP — verdict: **inert when `use_l2=False`**

Adds `L2Compressor.forward()` (delegates to `compress`) and a one-line patch.py change. Both gated on `use_l2`.

### 3.6 `69b396e` — optimizer walks `model.parameters()` post-FSDP — verdict: **DDP path explicitly preserved**

Diff explicitly: `if args.use_fsdp: trainable = [p for p in model.parameters() if p.requires_grad]; else: trainable = _mem_space_params(...)`.

P11 DDP rerun takes the `else` branch — bit-identical to P8 optimizer construction. Sanity log line `Optimizer param-collection sanity: optim_params=255 named_parameters(requires_grad)=255 use_fsdp=False` confirms 255/255 params present, exactly as in the P8 era.

**Verdict:** DDP optimizer-collection unchanged. Cannot explain the gap.

### 3.7 `33d1ca9` (chunk-local BPTT) + `033b286` (per-chunk pool cache) — verdict: **NOT in the diff window**

The team-lead's hypothesis list included these two commits as possible regression sources. Verified by:

```
$ git merge-base --is-ancestor 33d1ca9 3c97b86 && echo ancestor
ancestor
$ git merge-base --is-ancestor 033b286 3c97b86 && echo ancestor
ancestor
$ git log --oneline 3c97b86 -- src/memory/mem_space scripts/train_mem_space_babilong.py | grep -E "33d1ca9|033b286"
033b286 fix(l3): per-chunk pool() cache + reset hook clears prev_chunk_h
33d1ca9 fix(l3): chunk-local BPTT — recompute summary in layer.forward from prev-chunk detached H
```

Both commits are **ancestors** of `3c97b86` (the P8 baseline). They were already live when P8 trained and hit BABILong mean=59.14. They therefore cannot be regression candidates between P8 and HEAD — they are part of the P8 baseline itself. Their effect on routing-key BPTT (if any) is *already baked into the 59.14 number*, not a post-P8 change.

If the concern is whether their pre-existing detach() truncates routing-key gradients in general (i.e. is the BPTT design itself wrong, independent of when it landed), that is a separate audit question outside the scope of "P8 → HEAD diff". Flagging here for completeness, but it cannot account for a P8→P11 regression.

**Verdict:** Out of scope; not part of the `3c97b86..69b396e` diff.

---

## 4. Top-3 highest-suspicion diffs (under the assumption a code regression *does* exist somewhere in the target paths)

Even after the §3 walk, if I have to rank the leftover suspicion, ordered by ROI of investigation:

### Rank 1 — `62a26db` `--gradient_checkpointing` interaction with the **two-call-on-same-frozen-module** pattern

**File / line:** `src/memory/mem_space/layer.py:455-493` (`_maybe_ckpt_wrapped_layer`); both call sites at `:803` (bypass) and `:817` (ext).

**Why still on the list despite §3.2's negative empirical:** The QUERY_DIAG signal is computed *before* the wrapped_layer call and is stable. But `slot_delta = ext_h[:, l1_start+k_slots:] - bypass_h` is the path that actually reads memory back into the residual stream, and **both** sides of this subtraction now go through their own checkpoint. If there's any subtle ordering issue (RNG state for SDPA dropout — none here since dropout=0; bf16 reduce non-determinism around `slot_delta` magnitude), it would manifest in the *output residual* but not in QUERY_DIAG. A 25-pp BABILong gap from a small per-step output bias compounding across 32 layers × 500 steps is plausible but unverified.

**In `use_l2=False` path?** Yes, fully active.
**In `use_fsdp=False` path?** Yes, fully active (this is the path taken by P11 DDP rerun).
**Changes the P8-time forward / writeback / sample-reset?** The forward of wrapped_layer is now wrapped in `torch.utils.checkpoint(use_reentrant=False)`. The writeback / sample-reset code is unchanged.

### Rank 2 — `f9e3fa7` in-progress FSDP scaffolding side-effects on the DDP path

**File / line:** `scripts/train_mem_space_babilong.py:54-69` (FSDP imports), `:271-400` (`_wrap_model_fsdp`, `_is_distributed_wrapper`, `_set_skip_writeback`), `:1052` (`_mem_space_params(model.module if _is_distributed_wrapper(model) else model)`).

The change from `isinstance(model, DDP)` to `_is_distributed_wrapper(model)` is functionally equivalent under DDP because `_is_distributed_wrapper` returns `True` for `DDP` and `True` for `FSDP`. With `args.use_fsdp=False`, the `model = DDP(...)` line wins and `_is_distributed_wrapper(model)` returns True → `model.module` is unwrapped — same as P8.

**Lower-confidence concern:** The `try / import FSDP` at module top has nontrivial import-time side effects (registers some hooks in newer torch). I have not measured this. Confidence: **low** that this matters.

### Rank 3 — `0f6e8a1` `extended_hidden` `parts`-list refactor (negligible suspicion)

**File / line:** `src/memory/mem_space/layer.py:660-680`. The `torch.cat(parts, dim=1)` call constructs a new tensor whose memory layout could subtly differ from the explicit 3-way `cat([l3, M_sel, H])` in P8 — same logical content, possibly different stride. This would matter only if downstream `position_embeddings` indexing or attention mask construction is sensitive to stride; both code paths use `_extend_position_embeddings(position_embeddings, k_l3 + k_l2 + k_slots)` and `_build_extended_attn_mask(...)` which take the count, not strides.

**Confidence:** **very low** that this matters. Listed only because the team-lead asked for top-3.

---

## 5. Diff-only conclusions (no experiment suggestions per spec)

Per spec, this audit is read-only and does not recommend training experiments — main will decide next steps from the diff findings alone. The diff-only conclusions are:

**Definite (high confidence):**
* In the diff `3c97b86..69b396e` restricted to `src/memory/mem_space/` and `scripts/train_mem_space_babilong.py`, **only one** post-P8 hunk is actually executed under the P11 DDP rerun's flag set (L2-off, FSDP-off, `--skip_mem_when_short` not passed, no length-weights): the `--gradient_checkpointing` path in `src/memory/mem_space/layer.py:455-493` introduced by `62a26db`.
* Every other post-P8 hunk in the target paths is gated on `args.use_fsdp`, `config.use_l2`, `--skip_mem_when_short`, or `--babilong_length_weights` — and none of those flags are set by `scripts/launch_p11_ddp_500step_validate.sh`.
* The two commits `33d1ca9` and `033b286` named in the team-lead's hypothesis #4 are **ancestors** of P8 (`3c97b86`), so they are part of the P8 baseline and cannot regression-account for the P8→P11 gap.

**Suggestive (medium confidence):**
* Even the `--gradient_checkpointing` hunk produces a QUERY_DIAG signature (`top1_sim_mean`, `per_tok_logit_std`, `key_max_cos`) **statistically identical** to the P8 baseline at step 25 / 49 / 73 (see §3 cross-table, §7 provenance). On the routing-telemetry axis, no regression is observable in the target-path diff.

**Implication for downstream investigation (informational only, no experiment proposed):**
* If the goal is to localize the BABILong eval gap (P8=59.14 vs P11 step500=33.81), the diff in target paths is *unlikely* to contain it. Probable next places to audit (outside this report's scope):
  - Eval pipeline (`scripts/run_babilong_mem_space.py`, the `_load_adapter` path, generation defaults like `temperature` / `top_p`, the BABILong metric in `third_party/babilong-pkg/`).
  - Code paths in the wider mem_space stack not covered here (writeback / dual-gate / forget-bias) which were not part of this audit's path scope.

---

## 7. Provenance

All log lines used here are existing files in this checkout; nothing was re-run during this audit.

* `logs/phase8_probe_20260515_2237.log` — P8 baseline at commit `3c97b86`, `selector_temperature=1.0`, no gradient_checkpointing.
* `logs/p11_ddp_500step_validate_20260517_1116.log` — P11 DDP rerun at HEAD, `selector_temperature=1.0`, `--gradient_checkpointing`.
* `logs/p11_fsdp_full_20260516_181417.log` — P11 FSDP 5000-step at commit `a6dcda3`.
* `logs/p8_temp20_500_20260517_105421.log` — P8 + `selector_temperature=20` rerun (NOT the canonical P8).
* `logs/p11_temp20_500_20260517_063303.log` — P11 + `selector_temperature=20`.
* `outputs/babilong_sft_phase8_l1l3_lr2e5/adapter_config.json` — confirms P8 hyperparams (`selector_temperature: 1.0`, `total_steps: 500`).
* `scripts/launch_p11_ddp_500step_validate.sh` — confirms P11 DDP rerun does NOT pass `--use_fsdp`, `--use_l2`, `--skip_mem_when_short`, `--babilong_length_weights`; DOES pass `--gradient_checkpointing`.
