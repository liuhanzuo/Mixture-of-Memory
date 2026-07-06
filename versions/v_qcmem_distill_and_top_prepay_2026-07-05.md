# QCMem extensions — self-distillation (A) + non-contiguous top-prepay resume (B)

Date: 2026-07-05
Backbone: Qwen3-8B (`models/Qwen--Qwen3-8b`, L=36), zero-training QCMem resume
mechanism already validated (self_test byte-exact, j-sweep run).

Prior result (context): zero-training QCMem j-sweep on Qwen3-8B (official
judge, n=100, qa5 16k): j0=57, j9=49, j12=51, j15=40, j18=19 — sweet spot
j12-15 saves 33-42% of read compute while holding 40-51. But qa1 (precise
localisation) collapses early (j12=11) and the Llama backbone collapses on qa5
(j12=3). These two extensions attack those two failure modes.

---

## Direction A — LoRA self-distillation (`scripts/train_qcmem_distill.py`)

### Architecture (forward pass)

```
# ONE shared Qwen3-8B, frozen; LoRA r16 (all-linear) on layers[j:] only.
# Per PG19 window -> [sink(BOS) ; ctx_0 .. ctx_{n_ctx-1} ; query], chunk_size each.

# ---- TEACHER (j=0, adapters OFF, no_grad) : RAG upper bound ----
with peft_model.disable_adapter(), torch.no_grad():
    qc0 = QCMemModel(model, resume_j=0)               # full-depth caches
    t_sink = qc0.write_chunk(BOS)                      # embed only (j=0)
    t_ctx  = [qc0.write_chunk(c) for c in ctx]         # embed only
    t_q    = qc0.write_chunk(query)
    t_logits = qc0.read_core(t_sink, t_ctx, t_q, logits_tail=T_q)  # == full fwd
    teacher_idx, teacher_val = topk(t_logits[query_seg], k=64)     # cache constant

# ---- STUDENT (j=resume_j, adapters ON) ----
with torch.no_grad():                                  # bottom layers[0:j] frozen
    s_sink = qc.write_chunk(BOS)                       # depth-j cache (query-blind)
    s_ctx  = [qc.write_chunk(c) for c in ctx]          # depth-j caches
    s_q    = qc.write_chunk(query)
s_logits = qc.read_core(s_sink, s_ctx, s_q, logits_tail=T_q)   # layers[j:] w/ LoRA (GRAD)

loss = distill_logits_kl(s_logits[query_seg], teacher_idx, teacher_val, lam=0.6)
       (+ ce_weight * CE(s_logits, teacher_argmax))    # optional
loss.backward();  allreduce_grads_mean();  clip;  adamw.step()
```

- The teacher and student share ONE model instance (adapters toggled), so no
  second 8B copy in memory. Only the ~29.1M LoRA params (+ AdamW state) train.
- Bottom `layers[0:j]` have NO adapters and are run under `no_grad` at write —
  the student learns to reconstruct the j=0 teacher's read-out purely from the
  shallow depth-j cache by adapting `layers[j:]`.
- Loss on the QUERY-segment tokens only (`logits_tail=T_q` avoids materialising
  the full `[1,|H|,V=151936]` logits — ~1.2GB/2k-pack in bf16 — byte-identical
  to slicing).

### Initialization / hyperparameters

| param | value | reason |
|---|---|---|
| LoRA r / alpha | 16 / 32 | matches QCMem paper "29M LoRA"; verified 29.10M trainable |
| LoRA targets | all-linear (q,k,v,o,gate,up,down) on layers[j:] | full adaptation of the resume band |
| resume_j (student) | 12 | zero-train sweet spot on Qwen (qa5 j12=51); qa1 cliff to fix |
| teacher j | 0 | RAG upper bound (qa1 j0=81) |
| distill_lambda | 0.6 | matches dolmino_cpt bidirectional-KL default |
| teacher_topk | 64 | matches existing distill cache convention |
| n_ctx / chunk | 7 / 512 | 4096-tok training window (raise to attack longer lengths) |
| lr / steps / warmup | 1e-4 / 1000 / 50 | few-hundred-to-1000-step LoRA fit |

### Relationship to prior work
- QCMem paper's LoRA self-distillation (their Qwen j12 qa1 .14 -> .67). This is
  the PURE-PG19 arm (no needle-mix). The paper reports pure-LM distillation was
  insufficient on Llama (§4.6/§4.9); the open question is whether Qwen's better
  zero-training resume behaviour lets pure PG19 suffice. If not, a needle-mix
  arm is the follow-up (would require a synthetic-recall generator — NOT
  babilong, to respect the red line).
- Distinct from mem_space distillation (`train_mem_space_dolmino_cpt.py`): there
  the student is a memory-bank/x-attn architecture; here the student is the SAME
  transformer resumed at depth j — no new params beyond LoRA, no memory module.

### Known issues / caveats
- Pure PG19 LM distillation may not transfer to precise-localisation qa1 (the
  paper's warning). Measure qa1 recovery vs zero-train j12=11 before scaling.
- Manual gradient all-reduce (not DDP.forward) because the student graph runs
  `layers[...]` directly, bypassing `DistributedDataParallel.forward`'s reduce
  hooks — wrapping in DDP would silently de-sync ranks. Params broadcast from
  rank 0 at start; grads SUM-then-/world_size each step.

---

## Direction B — non-contiguous "top-prepay" resume (`QCMemModel.top_prepay_b`)

### Feasibility analysis (the gating question: can the TOP b layers be prepaid?)

User hypothesis: cache "front a layers + back b layers", recompute only the
middle `[a : L-b]`. Bottom prepay (`a` layers at write) is already exact — it IS
`resume_j`. The NEW question is whether the TOP `b` layers can be prepaid.

**Verdict: exact top-prepay is IMPOSSIBLE.** The top band's input `h_{L-b}` is
the OUTPUT of the middle integration band, which is query-conditioned (the query
attends to the context there). So `h_{L-b}` cannot be materialised chunk-local
before the query is present. Measured on Qwen3-8B (2026-07-05): a query-blind
(chunk-local) `h_{L-b}` vs the query-aware value diverges by

| b | top-input depth L-b | cos(blind, aware) | rel L2 diff |
|---|---|---|---|
| 12 | 24 | 0.865 | 11.3 |
| 8  | 28 | 0.902 | 5.8  |
| 6  | 30 | 0.916 | 4.2  |

i.e. NOT close — a cached top-band output would be wrong. Top layers sit ABOVE
the query-conditioning band in the compute DAG; "front + back caching" in the
strict sense does not hold.

**Tractable approximation implemented instead:** recompute the middle band
`layers[a : L-b]` over the FULL packed sequence (query-aware, exact), then run
only the TOP band `layers[L-b : L]` QUERY-LOCAL — over the query tail alone with
fresh contiguous positions, NOT over the context. This saves running the top `b`
layers across the (long) context. It leans on the hypothesis that the top layers
are query-blind "output/format" layers whose context-attention is dispensable —
which is exactly the ablation question this arm answers. No KV injection, no new
params (composes with Direction A's LoRA).

### Forward pass (`read_core`, b>0)

```
packed = [sink ; ctx... ; query]                       # depth-a caches (write side)
mid = layers[a : L-b](packed)          over |H|, fresh positions, causal   # query-AWARE
top_in = mid[:, -T_q:, :]              # query tail only
top = layers[L-b : L](top_in)          over T_q, fresh positions 0:T_q     # query-LOCAL
logits = lm_head(norm(top))            # [1, T_q, V]
```

- `b == 0` (default) -> EXACT connective resume == current QCMem read; returns
  `[1,|H|,V]`. This is the byte-exact self_test path.
- `resume_forward_ids` (single contiguous sequence) is exact for ALL `(a,b)`
  (one chunk => top-prepay degenerates), so the self_test gate holds at any
  config. Verified: max|logit diff| = 0.000e+00 for a in {0,12,36} × b in {0,6,12}.

### Compute saving
Top `b` layers run over `T_q` (query tail, e.g. 512) instead of `|H|` (full pack,
up to ~thousands). Saving grows with context length: at 16k pack, T_q=512, b=8
skips ~8 layers over ~15.5k tokens.

### Known issues / caveats
- APPROXIMATE by construction (see feasibility). Quality depends on how
  query-blind the top layers really are — measure qa1/qa2/qa5 vs b=0 across
  lengths. Expect degradation to grow with b and with context length (more
  context the top band no longer sees).
- Combine with Direction A: distill a student whose read uses b>0 so LoRA can
  compensate for the query-local top band (`--top_prepay_b` in the trainer).
```
```

---

## Launch commands

### A — self-distillation (Qwen, pure PG19), local B200 8×L20A
```bash
PROJECT_ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory \
setsid nohup bash scripts/launch_qcmem_distill_qwen_pg19.sh \
  >logs/qcmem_distill_qwen_pg19.out 2>&1 &
# -> outputs/qcmem_distill_qwen_j12b0_pg19_nctx7/{step*,final}/adapter_model.safetensors
```
Then eval the distilled adapter vs zero-training (official judge):
```bash
RESUME_J=12 B_VALUES="0" \
LORA_ADAPTER=outputs/qcmem_distill_qwen_j12b0_pg19_nctx7/final RUN_TAG=_distilled \
TASKS="qa1 qa2 qa5" LENGTHS="0k 1k 2k 4k 8k 16k" \
PROJECT_ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory \
setsid nohup bash scripts/_eval_qcmem_bsweep_taskpool.sh >logs/qcmem_distilled_eval.out 2>&1 &
```

### B — top-prepay ablation (zero-training first), on a second node (.52 diskB)
```bash
RESUME_J=12 B_VALUES="0 4 8 12" TASKS="qa1 qa2 qa5" LENGTHS="0k 1k 2k 4k 8k 16k" \
PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
PYTHON_BIN=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/.venv/bin/python \
setsid nohup bash scripts/_eval_qcmem_bsweep_taskpool.sh >logs/qcmem_bsweep.out 2>&1 &
```
(Note: .52 is diskB — Qwen weights + code must be present there; local wzc1 B200
node has them already.)
