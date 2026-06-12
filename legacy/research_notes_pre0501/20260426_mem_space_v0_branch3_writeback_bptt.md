# Memory-Space v0 — Branch-3 writeback-BPTT pivot plan

**Date:** 2026-04-26
**Author:** /researcher (dispatched from main after Stage-2a + Stage-2b both FAIL with PPL > 100)
**Status:** plan only — no code change performed; coder + trainer not yet dispatched
**Red-line trigger:** `CLAUDE.md` "PPL > 100 → /researcher before /trainer"

Upstream failures that triggered this pivot (see `status/gpu_runs.jsonl`):

| Run | config | PPL | verdict |
|---|---|---|---|
| Tier-3 held-out (8G × 200, skip=200) | N=512, k=64, **writeback detached**, h2s frozen | **2.1278** | **best — Branch 1 PASS** |
| Stage-2a unfreeze h2s (8G × 200, skip=0) | N=512, k=64, writeback detached, **h2s trainable** | 322.5094 | FAIL |
| Stage-2b N=1024 k=128 smoke (1G × 10) | N=1024, k=128, writeback detached, h2s frozen | 426.3595 | FAIL |

Both Stage-2 failures share a common structural cause: **the slot-write pathway is severed from the loss**, so every knob that talks to the write path (unfreeze h2s, more capacity) wiggles a broken-gradient channel. This note pins down the exact severance points and proposes the minimal change (Option A) to restore end-to-end gradient flow for intra-chunk writebacks while preserving the inter-chunk gradient break.

---

## 1. Root-cause audit — every gradient-severing op in the memory path

I walked `src/memory/mem_space/{layer.py,memory_bank.py,selector.py,config.py}` and
`scripts/train_mem_space_pg19.py` line by line. Here is the full inventory of
`.detach()` / `requires_grad=False` / non-Parameter state, with verdict per item.

| # | Location | Code | Effect | Verdict for Branch-3 |
|---|---|---|---|---|
| D1 | `layer.py:425` | `self.memory_bank.write(idx, O_mem_slot.detach(), beta_val)` | **kills** ∂L/∂O_mem_hidden via the write path; also kills ∂L/∂slot_to_hidden via layer i→i+1 (the 32-layer shared-bank chain) | **REMOVE** — this is the single load-bearing severance |
| D2 | `layer.py:420` | `beta_val = float(beta_t.detach().item())` | β is now a **Python float** → no grad into `gate_param` from the write path (gate only sees grad from `aux["beta"]` which nothing consumes) | **REMOVE** — pass `beta_t` as tensor into `write()` |
| D3 | `memory_bank.py:150` (init) | `self.slots = slots.detach()` | intentional cold-start break; slots do not carry grad to H used for init | **KEEP** — this is the graph root for the chunk |
| D4 | `memory_bank.py:143` (hidden_pool init) | `H_l.detach().mean(dim=1, keepdim=True)` | ditto — init is a fresh leaf | **KEEP** |
| D5 | `memory_bank.py:93` (`detach_()`) | `self.slots = self.slots.detach()` | explicit inter-segment break for callers | **KEEP** — but note it is not currently called anywhere (relied on by `reset` alone; see D7) |
| D6 | `selector.py:131` | `idx = idx.detach()` | STE — gradient flows via `ste_weights` instead. Correct. | **KEEP** |
| D7 | `scripts/train_mem_space_pg19.py` | `_reset_banks(model)` before every chunk (both `_evaluate_chunks` and training rollout) | wipes bank to None, forces re-init next forward → inter-chunk gradient is **structurally broken** (the next chunk's slots are a brand-new leaf) | **KEEP** — this is the *only* thing that makes BPTT tractable in memory/time |
| F1 | `layer.py:224-226` | `if config.hidden_to_slot_frozen: p.requires_grad = False` | h2s excluded from optimizer | **KEEP for first Branch-3 run** — isolate the gradient-path flip from the capacity flip; re-enable in a follow-up if Option A PASSes |
| F2 | `train_mem_space_pg19.py:_mem_space_params` | only harvests `selector`, `gate_param`, `slot_output_gate`, `slot_to_hidden` | h2s not passed to AdamW (belt-and-suspenders with F1) | **KEEP** |
| S1 | `memory_bank.py:72` | `self.slots` is a **plain attribute**, not a `nn.Parameter` and not `register_buffer` | state is per-forward, DDP does not sync it, state_dict does not checkpoint it | **KEEP** — we want per-sample per-chunk state, not a global weight |
| S2 | `memory_bank.py:218` | `self.slots = self.slots.scatter(1, idx_exp, updated)` — out-of-place scatter | each write creates a new autograd leaf rooted at `updated`; `updated` depends on `new_repr` | works naturally with D1 removed: the tensor at layer i+1's `get()` call carries grad back into layer i's `O_mem_hidden` |

**Conclusion of the audit.** The write path is inert because of D1 + D2 together. Everything else (D3–D7, F1–F2, S1) is either correct, intentional, or belongs to the inter-chunk boundary (which we explicitly want to keep broken). Unfreezing h2s (F1 → False) as Stage-2a did *without* fixing D1+D2 simply adds a 540 M-parameter channel that only the load-balance aux loss can reach; AdamW then pushes those weights along a pathological direction that is orthogonal to LM loss, destroying the h2s init that was implicitly frozen-into at PPL=2.1278. That matches the Stage-2a training-log signature exactly: aux_loss descends 28.3 → 22.1 (selector *is* learning load balance) while lm_loss climbs 5.85 → 6.99 (LM backbone is being steered against its own loss).

---

## 2. Proposed code change — **Option A: remove D1 + D2 only**

Of the three options the dispatch brief enumerated:

* **A. Remove `.detach()` on `O_mem_slot` + pass `beta_t` as tensor (writeback-BPTT through shared bank)** ← **recommended**
* **B. Differentiable surrogate** (e.g. stop-gradient on cross-chunk, but explicit grad route via `slot_delta`) — overkill, adds new surface
* **C. Per-sample `nn.Parameter`** (bank as Parameter, updated by AdamW) — breaks per-sample semantics and requires rewriting `_reset_banks`; fundamentally wrong for pg19 chunks where each sample is a different book

Option A is the minimum sufficient change. The intra-chunk BPTT path that opens up is the **32-layer shared-bank chain**: `memory_bank` is per-layer, but *within a single forward pass over one chunk*, each `MemorySpaceLayer.forward` executes sequentially and `get()` / `write()` on its own bank. There is no cross-layer slot sharing in the current design — each layer l has its own bank, its own selector, its own projections. So Option A's BPTT horizon is one forward over one chunk, one layer at a time. That is **exactly** the horizon we want: short enough to avoid vanishing/exploding, rich enough to teach h2s "if you write X into the selected slot, the next time this slot is re-selected, the LM loss decreases".

**(correction after re-reading `layer.py:319` `slots = self.memory_bank.get()`)** Because each layer has its own `memory_bank`, the BPTT chain is actually even shorter: for a given layer l, the gradient flows from layer-l's `lm_loss → O_mem_hidden_l` only if *within the same chunk forward* some downstream op at layer l reads the slots we wrote. In the current design, the `write` happens **after** that layer's `get()` (line 319) and `scores` computation, so **there is no same-layer self-read in one chunk**. The gradient path that Option A actually opens is:

  `lm_loss  →  bypass_h + α · slot_delta  →  slot_delta  →  ext_h[k:]`
  (which is already unbroken today — it is the α-gated Flamingo path that produced PPL=2.1278)

and **additionally**:

  `lm_loss  →  O_mem_hidden  (via ext_h[:k])  →  hidden_to_slot  →  O_mem_slot  →  memory_bank.write  →  slots scatter`

Today the second path is cut by D1. Option A reconnects it. The catch is that, at the layer where the write happens, the *written* slot is not re-read in the same chunk (the bank is reset on the next chunk). **So at first glance Option A appears to give h2s a gradient route that still leads nowhere.**

That reasoning is wrong on a subtle point. `self.slots = self.slots.scatter(1, idx_exp, updated)` **rebinds** `self.slots` to a new autograd node whose inputs include `O_mem_slot`. If *anything* downstream in the same chunk reads `self.memory_bank.get()` again, a gradient path exists. Tracing `layer.py` confirms that within a single chunk's forward, each `MemorySpaceLayer.forward` calls `memory_bank.get()` exactly once (line 319, before the write). So within one chunk the written slots are never re-read by the same layer.

**However**, the write affects the *selector's scores* through a side door: `ste_weights = scores + (one_hot_scores - scores).detach()` (selector.py:146) backprops into `Q_sel` and `K_sel` via `scores`, which is a softmax over `torch.einsum("bs,bns->bn", q, k)` where `k = self.K_sel(slots)`. `slots` here is `memory_bank.get()` — the *pre-write* snapshot. So the selector's gradient into slots is also taken before the write. The write is a pure sink within one chunk.

**This is a real limitation of Option A as described.** To make writeback-BPTT actually teach h2s, we need the written slots to feed back into something still-differentiable. There are three clean ways:

* **A.1 (simplest, recommended for the first Branch-3 run):** Keep Option A's change (remove D1, pass β as tensor), but also add a *second* selector query on the post-write slots at each layer and mix it into `ste_weights` with a small fixed weight (e.g. 0.1). That creates a self-consistent "the bank I just wrote is the bank I'll route through" loop within one chunk.
* **A.2:** Carry the bank across layers (layer l's post-write bank becomes layer l+1's pre-read bank). This turns the 32 decoder layers into one long BPTT chain through the memory bank, which is **exactly** the "writeback-BPTT" semantics the dispatch brief asked for. Requires a small wiring change in `apply_mem_space_to_model` to share a single `MemoryBank` across layers, or to thread it manually.
* **A.3:** Add a second H-query at the end of each chunk that reads the post-write bank and uses it in an auxiliary "reconstruction" loss (e.g. reconstruct the next chunk's first N tokens from slots). Speculative; deferred.

**Recommendation for the first Branch-3 run: A.2 — share the memory bank across the 32 layers.** This is what "writeback-BPTT" means structurally: write at layer l, read at layer l+1, compound 32× within one chunk, inter-chunk break preserved by `_reset_banks`. Below is the full code spec the coder subagent should implement.

### 2.1 Code spec for A.2 (single shared bank, intra-chunk BPTT)

Files touched: `src/memory/mem_space/layer.py`, `src/memory/mem_space/patch.py` (or wherever `apply_mem_space_to_model` lives), `scripts/train_mem_space_pg19.py` (CLI flag only; no logic change). **No change to `memory_bank.py` or `selector.py`.**

1. **`apply_mem_space_to_model(model, cfg)`:** construct **one** `MemoryBank` per model (not per layer) with `num_slots, slot_dim` from cfg. Pass that single instance into every `MemorySpaceLayer` as a new `shared_bank` arg. Each `MemorySpaceLayer` sets `self.memory_bank = shared_bank` instead of building its own. Add a new cfg flag `shared_memory_bank: bool = True` to keep the old per-layer behaviour accessible via `=False` for ablation.
2. **`MemorySpaceLayer.forward` (layer.py:279)** — three edits:
   * Line 425: change `self.memory_bank.write(idx, O_mem_slot.detach(), beta_val)` → `self.memory_bank.write(idx, O_mem_slot, beta_t)`.
   * Line 420: delete `beta_val = float(beta_t.detach().item())` and its use site on line 425 (replaced above).
   * `MemoryBank.write` signature: change `gate: float` → `gate: torch.Tensor | float`; inside, replace `(1.0 - gate) * current + gate * new_repr.to(self.slots.dtype)` with code that handles both. Minimal patch:
     ```python
     gate_t = gate if isinstance(gate, torch.Tensor) else torch.tensor(gate, device=self.slots.device, dtype=self.slots.dtype)
     updated = (1.0 - gate_t) * current + gate_t * new_repr.to(self.slots.dtype)
     ```
     and drop the `if gate <= 0.0: return` short-circuit (it would decouple the graph when β is still warming; we now want grad-flow even at β≈0 because that's exactly when the selector/projections need to *start* learning to write).
3. **`_reset_banks(model)` in `train_mem_space_pg19.py`:** with a shared bank there is only *one* bank to reset; walk `root._mem_space_layers[0].memory_bank.reset()` or (cleaner) expose `root._mem_space_shared_bank` and reset that. Preserve idempotence.
4. **`scripts/train_mem_space_pg19.py` CLI:** add `--shared_memory_bank / --no_shared_memory_bank` (default True). Log `shared_memory_bank` in the output JSON.

Line-count estimate: ~60 lines across 3 files. Below the 200-line threshold in `CLAUDE.md` — main can write this directly, no coder subagent strictly needed, but dispatching /coder with this spec is also fine.

### 2.2 Why A.2 and not A.1 first

A.1 (second selector query on post-write slots) adds a new trainable surface and couples the selector update to the write update, creating a potential positive-feedback loop in the first few steps. A.2 is structurally cleaner: it gives exactly one BPTT path per chunk (slots are threaded through 32 layers), zero new trainable params, and matches the "memory bank is shared across depth, per-sample-per-chunk-per-model" intuition from the design doc §2.3. If A.2 PASSes (PPL ≤ 2.5), we stop and reopen the Stage-2a / Stage-2b knobs on top of A.2. If A.2 FAILs (PPL > 10), we retreat to A.1.

### 2.3 What stays frozen

* `hidden_to_slot_frozen=True` for the first Branch-3 run. This isolates the gradient-path flip (D1+D2 removed) from the capacity flip (h2s trainable). If Branch-3 A.2 PASSes, Stage-2a is re-opened *on top of* A.2 as a proper ablation. The Stage-2a failure on the current detached-write branch is void as data.
* All decoder-layer backbone weights frozen (unchanged).
* `slot_to_hidden` stays trainable (unchanged from Tier-3).
* `slot_output_gate` (α) stays trainable (unchanged from Tier-3).
* `gate_param` (β logit) stays trainable; Option A gives it a real gradient for the first time.

---

## 3. Gradient-scope discipline

| Scope | Current state | Branch-3 A.2 |
|---|---|---|
| Intra-layer | `get()` → selector → extended forward → writeback — writeback severed by D1 | unchanged topology, D1 removed → post-write bank carries grad forward |
| Intra-chunk, cross-layer (layer l → l+1) | **not possible** — per-layer banks | **ENABLED** — shared bank threads grad across 32 layers |
| Inter-chunk | broken by `_reset_banks` (D7) + `slots.detach()` at init (D3) | **unchanged** — same break preserved |
| Cross-sample | per-sample bank (batch dim B), reset each chunk | **unchanged** |

**Invariant we promise the optimiser:** a backward over chunk i produces gradients that depend only on activations inside chunk i. Chunk i+1 sees a fresh leaf (re-init). That is what `_reset_banks` guarantees, and A.2 does not touch it. No risk of unbounded BPTT chains, no memory-blow-up on long rollouts, no "graph does not contain X" AutogradErrors as long as `_reset_banks` is called between chunks (already the case).

**Gradient clipping:** `torch.nn.utils.clip_grad_norm_(trainable, 1.0)` stays. The BPTT chain is 32 writes deep but each β ≤ 0.3, so the effective Jacobian per hop is (1 − β) I + β · W_h2s · J_attn_writeback ≈ I + O(0.3) at most. Empirically, Tier-3 held-out with α=tanh(slot_output_gate) and the 32-layer Flamingo chain trained cleanly with clip=1.0; expect the same here.

---

## 4. Predicted PPL range + pass/kill gates

Anchors:

* Bypass parity (no memory): PPL 16.50 on pg19-llama3 chunks (skip=40000).
* Tier-3 full-train-eval (skip=0): PPL 1.5751.
* **Tier-3 held-out (skip=200): PPL 2.1278** ← the baseline Branch-3 must match or beat.
* Stage-2a broken-grad + trainable h2s: PPL 322.5094 (useless as reference).

**Prediction for Branch-3 A.2 (held-out, skip=200, 8G × 200):**

| Quantile | PPL | reasoning |
|---|---|---|
| p10 (best case) | 1.8 | gradient-bearing write tightens the slot routing → slightly better than the accidentally-good Tier-3 held-out |
| p50 (likely) | 2.0–2.3 | same order as Tier-3; writeback-BPTT's value is retrieval quality, which at k=64 and one-book-per-chunk has limited headroom |
| p90 (worst acceptable) | 2.5 | still within 20 % of 2.1278 |
| kill-gate | > 10 | anywhere along training; immediate kill |
| catastrophic | NaN in any grad | immediate kill |

**PASS gate:** final held-out PPL ≤ 2.5 **and** monotone-ish training curve (step-200 lm_ppl within 2× of step-1 lm_ppl). Under this we declare Branch-3 A.2 the new baseline and re-open Stage-2a (unfreeze h2s) on top of it.

**KILL gates (any one kills the run):**

| # | Trigger | Action |
|---|---|---|
| K1 | step-1 lm_ppl > 20 on 8G full | abort; this means the intra-chunk BPTT immediately destabilises |
| K2 | NaN in `gate_param.grad` or `slot_to_hidden.grad` at any step | abort; dump activations for debug |
| K3 | any lm_ppl > 100 at any step | red-line PPL>100 hit → /researcher again |
| K4 | step-200 lm_ppl > 2 × step-1 lm_ppl | diverging training; abort |
| K5 | aux_loss > 30 at step 20 | selector destabilised by the richer signal |

**Why PPL could still be bad despite the fix:** if k=64 is already saturating retrieval headroom (i.e. the 2.1278 held-out is close to the task's "memory-free on this chunk" lower bound, not a floor set by broken writeback), Branch-3 A.2 will match but not beat. That would still be a PASS (we unblocked the write path; Stage-2a can now be retested cleanly) but it would mean the next research direction is *longer context* (multi-chunk rollouts with deliberate cross-chunk carryover — requires dropping D3/D7 selectively) rather than *more capacity*.

---

## 5. Direct 8-GPU run config (no 1-GPU smoke; saturate B200)

Per the 2026-04-26 CLAUDE.md rule: **"更改之后不需要单卡 smoke. 直接多卡运行"** + **"选取好 batch size 最大化显卡效率"**.

```
node:            b200-1 (28.89.17.143)
canonical cwd:   /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
script:          scripts/train_mem_space_pg19.py

torchrun --nproc_per_node=8 scripts/train_mem_space_pg19.py \
    --model  /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
    --data   /apdcephfs_wzc1/share_303098609/pighzliu_code/data/pg19_chunks_llama3.npy \
    --max_chunks 200 \
    --seq_len 4096 \
    --skip_chunks 200 \
    --batch_size 1 \
    --num_slots 512 \
    --top_k 64 \
    --selector_dim 128 \
    --writeback_gate_max 0.3 \
    --writeback_warmup_steps 0 \
    --load_balance_weight 0.01 \
    --slot_init random --slot_init_noise 1.0 \
    --max_train_steps 200 \
    --lr 3e-4 \
    --attn_impl sdpa \
    --dtype bfloat16 \
    --output_dir outputs/branch3_writeback_bptt_A2_heldout
    # NEW FLAGS introduced by the coder for A.2:
    --shared_memory_bank        # default True after the change
    # hidden_to_slot stays frozen (default hidden_to_slot_frozen=True; do NOT pass --unfreeze_hidden_to_slot)
```

**Batch-size rationale (seq_len-bound, cannot raise).**

* Per-GPU memory budget on L20A = 183 GiB.
* bf16 Llama-3-8B fwd state: 8B params × 2 bytes ≈ 16 GiB weights (frozen) + 32 decoder layers × 4096 hidden × 4096 seq_len × (2× wrapped forward bypass+extended) × ≈ 4 activations/tensor × 2 bytes bf16 ≈ 33 GiB activations with k=64 prepended → ~49 GiB per sample. Ext forward runs on (k+T)=4160 tokens, so extended is ~2.5 % larger than bypass; total ≈ 50 GiB per sample.
* Add the 32-layer shared bank: B × 512 × 4096 × 2 bytes × (saved for backward) ≈ 4 MiB per sample — negligible.
* A.2 intra-chunk BPTT through 32 layers adds 32 × write activations × 2 bytes = ~32 × k × slot_dim = 32 × 64 × 4096 × 2 ≈ 16 MiB per sample — negligible.
* AdamW moments on 570 M trainable params (selector + s2h + gates + β): 570M × 4 bytes × 2 moments × fp32 = ~4.6 GiB. DDP gradient bucket adds ~1 GiB peak.
* Total per-GPU peak estimate: 50 + 5 + 5 NCCL buffer ≈ **60 GiB / 183 GiB** ≈ 33 %.

This leaves headroom, **but `batch_size=1` is already what the Tier-3 held-out 2.1278 used and what pg19 chunking assumes** (one book per sample). Raising `--batch_size` to 2 is feasible on a single GPU memory-wise (~100 GiB), but the *global* batch with 8 GPUs becomes 16 which deviates from Tier-3 and muddies the A/B. **Recommendation: keep `--batch_size=1` for the Branch-3 validation run** so PPL comparison to Tier-3 2.1278 is apples-to-apples. Record `batch_size=1` in `gpu_runs.jsonl` with the note "seq_len-bound; A/B parity with Tier-3 held-out baseline; headroom available for follow-ups".

If the trainer wants to push a follow-up run after PASS, `--batch_size 2 --seq_len 4096` should fit and would ~2× throughput at the cost of losing parity. Note that for follow-ups after pass.

**seq_len is fixed at 4096** (the pg19_chunks_llama3.npy layout; do not touch).

**lr=3e-4** — matches the Tier-3 held-out config. Stage-2a used lr=1e-3, which the bypass-parity unit test had already flagged as over-aggressive for this param count. Do not use 1e-3.

**skip_chunks=200** — held-out (unseen during Tier-3 training). Same as the 2.1278 baseline. This is the only honest A/B against Tier-3.

**max_train_steps=200** — one epoch over the 200 held-out chunks; matches Tier-3.

**output_dir: `outputs/branch3_writeback_bptt_A2_heldout`** under the canonical wzc1 workdir.

**Logging cadence:** existing `logger.info` at every step in `_evaluate_chunks` (every 20 steps during eval) and every step during training is sufficient. The trainer subagent should tail the log and watch for K1–K5 in real time.

**ACTIVE_SWEEPS.jsonl entry format** (to be added by the trainer when dispatching):
```
{"ts":"<ISO>","sweep":"branch3_writeback_bptt_A2","node":"b200-1","status":"running","config":{"num_slots":512,"top_k":64,"batch_size":1,"seq_len":4096,"lr":3e-4,"shared_memory_bank":true,"hidden_to_slot_frozen":true,"slot_init":"random","slot_init_noise":1.0,"skip_chunks":200,"max_train_steps":200}}
```

---

## 6. Failure modes in the first ~20 steps

Concrete debug playbook if anything goes wrong. Steps are numbered by the
priority the trainer should apply them.

| Symptom | Likely cause | First response | Second response |
|---|---|---|---|
| Step-1 lm_ppl >> 20 (e.g. 50+) | Shared-bank wiring wrong: layer-l writes clobber layer-(l+1)'s pre-read incoherently (slot_dim or B mismatch) | abort; unit-test the shared-bank apply on a 2-layer tiny-Llama (`scripts/test_mem_space_bypass_parity.py` pattern) | if unit test passes, fall back to A.1 (per-layer banks + grad-bearing write + 0.1-weighted second selector query on post-write slots) |
| Step-1 lm_ppl ≈ 2.3 but climbs by step 20 to > 5 | gradient through shared bank is too strong; β=0.15 effective too large at the start of training | drop `--writeback_gate_max 0.3` → 0.1 (halves β) | add `--writeback_warmup_steps 50` to ramp β from 0 |
| NaN in gate_param.grad | β went to numerical extreme (σ(big_number)·0.3 → 0.3 with huge inv-sigmoid grad) | abort; clip β gradient to 0.1 via param group in AdamW | rethink gate parameterisation; consider `β = 0.3 · sigmoid(gate_param)` with `init=logit(β_target/0.3)` ≈ 0 already |
| NaN in slot_to_hidden.grad | 4096×4096 Linear overflow under bf16 with the new through-path | abort; switch s2h only to fp32 master weights (keeps activations bf16) | reduce s2h init std 0.02 → 0.01 |
| aux_loss > 30 at step 20 (was 28 at step 1) | selector now has a competing signal (write wants different routing than LM) | drop `--load_balance_weight 0.01` → 0.003 | add an LM-loss-weighted slow-decay on aux |
| step-200 ppl ≈ step-1 ppl ≈ 2.2, no improvement | A.2's writeback-BPTT is signal-free on this task — 2.1278 was already the floor at k=64 | PASS at 2.1278 ± 0.3 is acceptable; note that write path is unblocked but doesn't help at this capacity; follow-up is longer-context not more-capacity | |
| any per-layer write raises shape error | shared-bank refactor forgot that per-layer `hidden_to_slot` outputs are per-layer (each layer has its own 4096×4096 h2s, so O_mem_slot is per-layer); shared-bank must accept per-layer writes | confirm `MemoryBank.write` is invariant to which layer called it (it is: just scatter into slot positions) | the bank's content IS a mix across layers, which is fine — that's the whole point of depth-shared memory |
| `find_unused_parameters=True` complains about `gate_param` | writeback never fires (β=0 fast path still somewhere) | confirm `if gate <= 0.0: return` was deleted | set `writeback_gate_init=0.1` to give nonzero β at step 0 |
| training pauses forever at step 1 | DDP grad sync stalls on the shared bank's `.scatter` chain | abort; force the shared bank's state to **not** participate in DDP by leaving it as a plain Python attribute (it already is — verify) | as a last resort, wrap the 32-write chain in `torch.utils.checkpoint` to rematerialise |

**Not a failure mode (expected):** `n_trainable` will drop from ~570 M (Tier-3) to ~570 M (unchanged — A.2 touches no parameter count; it only re-wires gradient flow). `gpu_runs.jsonl` should log n_trainable ≈ 570.43 M, matching Tier-3.

---

## 7. One-paragraph summary (for RESEARCHER_REPORTS.jsonl)

The Stage-2a (PPL 322.5) and Stage-2b (PPL 426.4) failures share a single root cause: the slot writeback pathway is severed by `O_mem_slot.detach()` and `beta_val = float(...)` in `layer.py:420-425`, so `hidden_to_slot` and `gate_param` receive no LM-loss gradient and any Stage-2 knob that touches the write path (unfreeze h2s, scale N/k) wiggles a broken channel. The Branch-3 cure (Option A.2) is: (i) remove `.detach()` on `O_mem_slot`, (ii) pass `beta_t` as a tensor not a Python float into `MemoryBank.write`, and (iii) share a single `MemoryBank` across all 32 decoder layers so intra-chunk writeback-BPTT threads through depth — while keeping `_reset_banks` and `init`-time detach, so inter-chunk gradient remains broken. `hidden_to_slot_frozen=True` stays for the first Branch-3 run to isolate the gradient-flip from the capacity-flip; re-opening Stage-2a is a follow-up *after* Branch-3 PASSes. Predicted held-out PPL 1.8–2.5 vs Tier-3 baseline 2.1278 (PASS gate ≤ 2.5, KILL gate > 10 or NaN). Run config: b200-1, 8× L20A, `--shared_memory_bank --num_slots 512 --top_k 64 --batch_size 1 --skip_chunks 200 --max_train_steps 200 --lr 3e-4 --slot_init random --slot_init_noise 1.0 --writeback_gate_max 0.3 --writeback_warmup_steps 0` (no 1-GPU smoke per new CLAUDE.md rule; batch_size seq_len-bound, per-GPU peak ~60 GiB/183 GiB). First-20-step failure playbook covers K1–K5 kill gates and retreat paths to A.1 (per-layer banks + second selector query).

---

## 8. Dispatch readiness checklist

- [x] root-cause audit complete (D1 + D2 load-bearing; everything else preserved)
- [x] concrete code change specified (Option A.2; ~60 LOC across 3 files)
- [x] gradient-scope discipline documented (intra-chunk BPTT enabled; inter-chunk break preserved)
- [x] PPL prediction + pass/kill gates set
- [x] direct 8-GPU config written with batch-size rationale
- [x] failure-mode playbook for first 20 steps
- [x] one-paragraph summary for JSONL
- [ ] **main**: apply code change (main can do this in <60 LOC, or dispatch /coder)
- [ ] **main**: rsync local zwfy6 → b200-1 canonical wzc1 workdir
- [ ] **main**: dispatch /trainer with the command in §5
- [ ] **trainer**: append ACTIVE_SWEEPS.jsonl row (§5) when dispatching
- [ ] **trainer**: tail log for K1–K5; auto-abort on K1/K2/K3
- [ ] **trainer**: on run complete, append gpu_runs.jsonl + compare to Tier-3 held-out 2.1278

**Branch-3 is dispatch-ready.** No more research needed before code change; next blocking step is the code edit itself.
