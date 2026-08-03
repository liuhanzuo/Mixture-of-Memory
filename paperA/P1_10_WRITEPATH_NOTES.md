# P1.10 — Write-path distillation (CoMem document-contextual Write, trained)

**Status:** launched 2026-08-03 on node `.104` (28.83.24.104, 8× H20, diskB/torch-base).
**Task:** Paper A #142 / P1.10 (user-approved). Trained upper bound of the P0.18 finding
that a *document-contextual Write* closes the deployable gap (Arm B 92.5 → E0 100),
and of P0.17's zero-training overlap-Write (recovers 80–87% of it).

## Goal
Train CoMem's **Write path** — the lower `j=12` layers (indices `0..11`) that produce
the cached `h12` for each 512-token chunk — so the cheap, deployable **chunk-local**
Write emits an `h12` that is **document-contextual** (aware of the preceding document
context), distilled token-for-token against the (non-deployable) document-contextual
Write teacher. Only the Write LoRA is trained; the trained Read is frozen.

## Design

### Student (deployable, trained)
- Qwen3-8B, `resume_j=12`, **chunk-local Write**: each of sink / ctx_k / query is
  encoded to depth 12 in isolation (RoPE `0:T`), per-chunk `h12` packed into fresh
  contiguous pack positions `0:H`, `layers[12:36]` resume → query-tail logits.
- **WRITE LoRA on layers `0..11`** (targets `q,k,v,o,gate,up,down`, r32/α64) — the
  *inverse* of the flagship READ trainer, which puts LoRA on `12..35`.
- Write is **grad-bearing** (NOT under `no_grad`, unlike the flagship); gradient flows
  query-tail-logits → frozen Read (`12:36`) → packed `h12` → WRITE LoRA (`0:12`).

### Teacher (document-contextual Write; P0.18 E0 "closes-to-100" construction)
- `layers[0:12]` run **once, continuously, full-causally** over the whole packed window
  `[sink ; ctx_0 … ctx_{n-1} ; query]` (contiguous positions `0:N`) with the **WRITE
  LoRA disabled** (`peft.disable_adapter`) → frozen base lower-12 = E0's stock lower-12
  (adapter-independent, since the flagship LoRA lives on `12:35`). `layers[12:36]`
  resume over the pack → query-tail logits. No grad.
- In this PG19 regime EVERY chunk is used in order ⇒ document positions == pack
  positions ⇒ the store→read **repositioning gap that E0 isolates against Arm B is
  ZERO**. So the teacher is exactly "document-contextual Write + trained Read", and the
  student's ONLY deficiency vs. it is the chunk-local **isolation** of its Write — which
  is precisely what the WRITE LoRA is trained to overcome.

### Shared frozen Read
- The flagship READ LoRA `outputs/qcmem_distill_qwen_j12_r32_4k/final` (layers `12..35`,
  r32/α64) is loaded and **`merge_and_unload`-ed into the base** as frozen weights.
  Both teacher and student therefore use the **identical trained Read**; the ONLY
  difference between them is how `h12` is produced (continuous vs chunk-local). This
  mirrors the P0.18 E0 harness, where E0 closes to ~100 **with the flagship READ LoRA
  present on all arms** — the decision to keep it frozen-in-base is taken directly from
  that config.

### Objective
- Bidirectional **top-64 logit KL** on the query-chunk tokens (`distill_logits_kl`,
  reused verbatim from `train_mem_space_dolmino_cpt.py`), `λ=0.6`, `ce_weight=0`.
  Chosen (over an `h12` MSE) to match the flagship READ-path objective family and
  because the mechanism — "make the two Reads agree" — is a logit-level statement; a
  frozen Read can re-weight `h12` dimensions, so matching outputs is the faithful target.

## Code isolation
- **New files only** — did NOT edit `train_qcmem_distill.py` or `qcmem_model.py`.
  - `scripts/train_qcmem_writepath_distill.py`
  - `scripts/launch_qcmem_writepath_distill_diskB.sh`
- The stock `QCMemModel.read_core`/`write_chunk` are batch-1 (write is `@no_grad`), so
  the trainer reproduces the SAME math **batched along the batch axis** and grad-bearing
  using QCMem's public low-level accessors (`embed_tokens`, `rotary_emb`, `_run_layers`,
  `norm`, `lm_head`) — SDPA implicit-causal masking (`attention_mask=None`, `q_len>1`),
  RoPE positions `0:S`. Batching is how the H20 is filled (all windows fixed length).
- `--self_test`: with the WRITE LoRA disabled, the batched contiguous Write(0:12)+
  Read(12:36) reproduces the merged model's full forward to fp tolerance (validates the
  batched pipeline + SDPA implicit-causal == explicit causal). Zero-init LoRA-B ⇒
  identity at step 0.

## Hyperparameters (mirror flagship `distill_args.json` unless noted)
`resume_j=12`, LoRA `r32/α64` on layers `0..11`, targets `q,k,v,o,gate,up,down`;
`chunk_size=512`, `n_ctx=3` (2048-tok window); `teacher_topk=64`, `λ=0.6`, `ce=0`;
`total_steps=4000`, `lr=8e-5`, `warmup=100`, `grad_accum=1`, `save_interval=500`;
bf16, sdpa, seed 42; **`gradient_checkpointing` ON** (grad now spans all 36 layers).
- **`batch_size = 24`** (NEW, windows/step) — chosen to fill the 97.8 GB H20 near ~80%;
  measured peak **~75 GB/card (~77%)** at 100% GPU util, no OOM. Probed batch=8 → ~40 GB
  (~41%, under-utilised) and batch=24 → ~75 GB; both give the same **6.7 win/s** throughput
  (larger batch is free util), so 24 was kept for the full run (safe headroom vs peaks on a
  ~4 h run; batch=28/32 risked OOM spikes for marginal gain).

## Run
- Node: `.104` (28.83.24.104), diskB, `PYTHON_BIN=/opt/conda/envs/torch-base/bin/python`
  (node `.venv` lacks torch; torch-base = torch 2.13.0 / peft 0.19.1 / tf 5.5.4, CUDA OK).
- Model path on `.104`: `models/Qwen3-8b-local` (repo symlink → zwfy6 Qwen--Qwen3-8b).
- Launch: `PROJECT_ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
  PYTHON_BIN=/opt/conda/envs/torch-base/bin/python setsid nohup bash
  scripts/launch_qcmem_writepath_distill_diskB.sh >logs/qcmem_writepath_distill.out 2>&1 &`
- output_dir `outputs/qcmem_writepath_distill_qwen_j12_r32/`, log
  `logs/qcmem_writepath_distill_qwen_j12_r32.log` (launcher writes `logs/$RUN.log`;
  outer wrapper stdout at `logs/qcmem_writepath_distill.out`).
- Sanity: `MAX_STEPS_SMOKE=30` (batch 8) + `MAX_STEPS_SMOKE=12` (batch 24) — both showed
  finite decreasing loss, no OOM/NaN, and `--self_test` max|diff|=0.000e+00.
- **Full run launched 2026-08-03 on `.104`, pid `1113844`, commit `8cf49ea`.**
  Step-10: `loss 0.1429  lr 8.80e-06  6.7 win/s` (batch=8 smoke step10 0.1945→step20 0.1437;
  batch=24 probe step10 0.1431 — consistent). ETA ≈ **4.0 h** for 4000 steps
  (4000 × 24 / 6.7 ≈ 1.43e4 s); saves every 500 steps to the output_dir.
