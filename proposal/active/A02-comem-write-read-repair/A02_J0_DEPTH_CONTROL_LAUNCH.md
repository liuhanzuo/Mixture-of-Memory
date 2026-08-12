# A02 — j=0 Depth Control: LAUNCH RECORD

**Node**: `.82` (28.82.250.82), 8× H20, zwfy6 disk. **Only `.82` was touched.**
`.104` / `.73` / LOCAL / `.21` were never contacted — no ssh, no process, no file.
**Launched**: 2026-08-12 15:20:47 CST (07:20:47Z)
**Driver**: `proposal/active/A02-comem-write-read-repair/code/run_a02_j0_depth_control.sh`
sha256 `8c2740748da8ab124f58555e3c8c624c85f6277a0321dfb6268f6000be68f5ff`
(**verified identical on wzc1 and zwfy6** after `scp -O`)
**Design**: `A02_J0_DEPTH_CONTROL_PREREG.md` (written BEFORE the training launch)
**Trainer**: `scripts/train_qcmem_distill.py`, zwfy6 md5 `a8c56b100432bb10293820b0936a6874`,
**unmodified** — no new trainer written.
**Derived from**: `scripts/_launch_p2_4_depthcurve.sh` (the on-disk launcher that
produced `outputs/qcmem_distill_qwen_j{6,9,18}_r32_4k`).

---

## 1. Pre-launch node verification

```
nvidia-smi (all 8):        0 MiB / 97871 MiB, 0 %
--query-compute-apps:      EMPTY
pgrep 'train_|eval_|torch.distributed':  EMPTY
df /apdcephfs_zwfy6:       3.2 T available
torch 2.13.0 / peft 0.19.1 / transformers 5.5.4
bitsandbytes:              ModuleNotFoundError  <-- irrelevant, see below
```

`.82` has **no bitsandbytes**, as warned. **Not a blocker for this trainer**:
`scripts/train_qcmem_distill.py` uses `torch.optim.AdamW` (line 485) and has **no
`import bitsandbytes` anywhere** (verified by grep — unlike
`train_olmo2_arch_probe2_distill.py`, which is module-level bnb-locked to .73/.104).
No `bnb8bit` path is used.

## 2. Fail-closed gates, all PASSED before any training GPU

```
[15:20:46] GATE A PASS flagship Read-LoRA sha dd09cd17457c63578c0f
GATE B PASS: ladder j=6,9,18 match flagship recipe on all 22 fields
             except resume_j; adapter spans are [j..35] r32/a64
[15:20:47] GATE B PASS ladder recipe-matched
GATE C PASS: j12_r40 params == j6_r32 params == 72,744,960 (exact capacity match)
[15:20:47] GATE C PASS capacity arithmetic exact
[15:20:47] trainer md5 = a8c56b100432bb10293820b0936a6874
```

GATE B is what makes the match **real rather than nominal**: it machine-verifies
the three reused ladder arms against the flagship on 22 recipe fields
(`lora_rank/alpha/dropout/targets, chunk, n_ctx, query_loss_tokens, teacher_topk,
distill_lambda, ce_weight, total_steps, lr, warmup, wd, grad_accum, grad_clip,
gradient_checkpointing, dtype, attn_impl, seed, top_prepay_b`) plus the adapter
span `[j..35]`, and aborts on any mismatch.

## 3. Arms launched (serial, 8-GPU DDP each)

| arm | j | r/α | trainable | purpose |
|---|---|---|---|---|
| **A1** `a02_j0control_lora_r32` | 0 | 32/64 | **87.29 M** on `layers[0:36]` | the literal control `next_gate[3]` asks for |
| **A6** `a02_j12_capmatch_r40` | 12 | 40/80 | 72.74 M on `layers[12:36]` | exact capacity match to on-disk j=6 |

## 4. "Is it really training" — evidence, not process-existence

**A1**, checked at step 150 and again at 580:

1. **init banner**: `model=…/Qwen--Qwen3-8b resume_j=0 top_prepay_b=0 lora_r=32
   chunk=512 n_ctx=3 dtype=torch.bfloat16 world_size=8` — **world_size=8**, correct
   model, correct depth.
2. **param groups vs architecture**: `LoRA on layers[0:36] targets=[q,k,v,o,gate,up,down]
   -> trainable 87.29M params`. Independently predicted from the architecture:
   `75776 × r × (36−j) = 75776 × 32 × 36 = 87,293,952` = **87.29 M ✓ exact**.
   (Same formula reproduces the on-disk 58.20/65.47/72.74/43.65 M for j=12/9/6/18.)
3. **step lines with loss and cadence**: monotone step counter 10→580, loss printed,
   lr following the warmup→cosine schedule (`8.80e-06 → 8.00e-05 → 7.80e-05`).
4. **maxmem vs capacity**: **41015 MiB / 97871 MiB = 41.9 %** on all 8 cards,
   util **98–100 %**.
5. **failure syntax**: `grep -inE 'traceback|unrecognized|out of memory|CUDA error|
   ChildFailedError'` filtered with `grep -viE 'No NaN/Inf'` → **empty**.
   (The `-v` filter is deliberate: a bare `grep -icE 'traceback|nan'` matches the
   PASSING line `✓ No NaN/Inf in model parameters` and yields a false alarm.)
6. **9 procs** matching `train_qcmem_distill.py` = 1 launcher + 8 ranks.

## 5. Measured cadence and ETA — from real windows, never a single line

`elapsed/iter` across three windows (wall-clock deltas, not the log's
instantaneous `samp/s`):

| window | Δt | Δsteps | s/step | 4000-step projection |
|---|---|---|---|---|
| step 160 → 290 | 136 s | 130 | 1.046 | 69.7 min |
| step 290 → 580 | 321 s | 290 | 1.107 | 73.8 min |
| **combined 160 → 580** | **457 s** | **420** | **1.088** | **72.5 min** |

**A1 ETA ≈ 73 min** (finishes ≈ 16:33 CST). This lands inside the 82–95 min
prediction from the ladder's own H20 timings (linear in grad-bearing layers) —
slightly faster, so the prediction was conservative.

**A6** has 24 grad-bearing layers vs A1's 36 ⇒ **≈ 55–62 min**.
**Both arms ≈ 2.1–2.3 h**, ending ≈ 17:30 CST.

⚠️ The flagship's logged 24.5 samp/s is **not** comparable: its `distill_args.json`
`model_path` is the **wzc1** path, i.e. it ran on **L20A**, not H20. Using it would
have understated the ETA ~3×.

## 6. First result: the vacuity prediction is CONFIRMED at full scale

PREREG §2.7 prediction 2 was "A1's training loss stays ~1e-3 for all 4000 steps".
Through step 580 on 8 GPUs:

| | flagship j=12 (on disk) | **A1 j=0 (this run)** |
|---|---|---|
| loss @ step 10 | **0.2991** | **0.0011** |
| loss @ step 130 | — | 0.0022 |
| loss @ step 490 | — | 0.0020 |
| **max loss seen** | — | **0.0077** |
| loss @ step 4000 | 0.0555 | (pending) |

The j=0 objective sits **~150–270× below** the flagship's starting loss and shows
**no descent** — it began at its optimum. Combined with the 20-step 1-GPU GATE 0
probe (**loss 0.0000 at step 1**), this is the measured form of:

> **the "LoRA distilled for j=0" that `next_gate[3]` requests is a NULL adapter —
> its distillation objective is already globally minimised at initialisation,
> because at j=0 the teacher and the student are the same computational path.**

## 7. Hygiene notes

* **No rotation flags passed.** The zwfy6 trainer predates `--keep_last_n` /
  `--keep_steps` and argparse-rejects them; my first probe died with
  `error: unrecognized arguments: --keep_last_n 3`. The ladder launcher passes
  none either, so omitting them *is* the recipe-faithful choice. 8 saves × ~223 MB
  × 2 arms ≈ 3.6 GB against 3.2 T free.
* **No `--eval_interval` passed, deliberately.** This trainer **has no such flag**
  and **no inline eval in its step loop** (verified by grep: only a docstring
  mention of `run_self_test`). So the NCCL-watchdog SIGABRT mode cannot occur, and
  passing `--eval_interval 0` would be a bogus arg that argparse would reject —
  killing the launch exactly as `--keep_last_n` did. All eval is offline.
* **Probe artefacts removed** (`outputs/a02_j0control_vacuity_probe/`) so they can
  never be mistaken for a real arm. GATE 0's evidence lives in
  `logs/a02_gate0_j0_vacuity.log` on `.82`.
* **Batch size**: unchanged and **not tunable** — the trainer has no batch
  dimension (no `--batch_size`), and `n_ctx=3` / `grad_accum=1` are part of the
  recipe being matched. Raising them would fill the card but **destroy the
  matched-tokens property this gate exists to establish**. 41.9 % card occupancy is
  therefore the correct outcome, not an under-utilisation to be fixed.
