---
scope: A03 Arm 4 (peak-LR CPT) design; task #199 territory
date: 2026-08-09
status: LAUNCHED 2026-08-09 ~13:24 GMT+8 on .73 (Config B). step205000/210000 already evaluated. Was DESIGN_ONLY before launch -- do not read that stale value.
run: outputs/olmo2_probe2_1B_keep7f2_dolmino_arm4_peaklr20k (watcher self-stops at step220000.pt)
driver: scripts/_run_a03_arm4_peaklr.sh
evidence: evidence/arm3_arm4_cpt_trajectory_paired_full.json
naming_warning: "This 'Arm 4' is the peak-LR CPT CONTROL for Arm 3. It is NOT the Arm 4 of the 6-arm study (pruned+heal+raw-text RAG), which is still unimplemented. Two different experiments share the label -- always say 'Arm 4 (peak-LR)' or 'Arm 4 (raw-text RAG)'."
---

# A03 Arm 4 design — how to restart at peak LR after resume

## Motivation

Arm 3 (interim VERDICT.md: no coherent trajectory at step205/210/215k) ran at
0.28–0.33× peak LR because Arm 2's cosine was fully consumed at step200000 and
Arm 3 kept the same schedule with `--max_steps 300000`. A null result at this
LR band cannot license the claim "Dolmino CPT saturates on A03's axes". Arm 4
tests the alternative: a **fresh peak-LR CPT arm** starting from Arm 2's
`step200000.pt`.

## Trainer LR mechanics (verified from source)

`get_lr` at `scripts/train_semantic_bottleneck_1b.py:76` (imported by
`train_olmo2_arch_probe2.py:91`):

```python
def get_lr(step, warmup, max_steps, base_lr, min_lr):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    prog = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(prog, 1.0)))
```

Uses `step` as an ABSOLUTE step number and `max_steps` as an absolute stopping
point. Resume keeps `step=step_saved` (train_olmo2_arch_probe2.py:988-1002).
So the effective LR at resume depends entirely on `(step, warmup_steps,
max_steps)` triple, not on any resume-flag semantics.

## Options for Arm 4

Given resume step = 200000 and 20k target CPT window (stop at step 220000):

| config | max_steps | warmup_steps | LR at step200000 | LR at step220000 |
|---|---:|---:|---:|---:|
| **A: warmup-hack, 220k horizon** | 220000 | 200500 | 1.995e-5 (≈peak) | ~2e-6 (min_lr, cosine consumed) |
| **B: warmup-hack, 240k horizon** | 240000 | 200500 | 1.995e-5 (≈peak) | ~1.12e-5 (0.56× peak) |
| C: standard config | 220000 | 150 | ~2e-6 (past warmup, near end of cosine) | ~2e-6 | ← Arm 3 pattern, DO NOT USE |

**Recommendation: Config B (`--max_steps 240000 --warmup_steps 200500`).** This
holds LR near peak for the first ~500 steps (warmup ramp 200000→200500), then
cosines from peak (2e-5) down to 0.56× peak (1.12e-5) over the 20k window
(step200500→220000). Matches typical CPT recipes better than a peak-to-min
sweep.

If tighter budget: Config A does peak-to-min over 20k, still peak-anchored.

## Adam moment implications

Arm 2's Adam moments were adapted to LR trajectory 2e-5→2e-6 across 200k steps,
ending at min_lr. Restarting at peak LR means moments briefly under-adapted for
the first few hundred steps. Expected overhead: ~500-1000 steps of "warm-up"
before moment stabilization. Not a bug, but note it in the Arm 4 verdict:
"Arm 4's first 500 steps are Adam-moment-mismatched; interpret step205k
onward."

## Assets required

* **Ckpt: `outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt`** — 12 GB,
  **zwfy6-only** as of last check. Cross-disk (wzc1) is 5–16h at measured
  12–37 MB/s. → Arm 4 must run on `.73`, `.82`, or `.104`.
* Data: `data/dolmino_now15b.npy` — present on both disks.
* Base: `../models/OLMo-2-0425-1B` — present on both disks.

## When to launch

* NOT before Arm 3 finishes (both use step200000.pt from same trainer script; a
  parallel run would create a heal-recipe collision on the trainer's global
  state).
* Preferred node: `.73` immediately after Arm 3's step220000.pt is emitted and
  the watcher completes step220000 eval. Arm 3 will be done, .73's 55 GB/card
  budget is exactly what Arm 4 needs.
* Alternative: `.82`, same disk.

## Command (draft)

```bash
PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
CKPT=outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt \
OUT=outputs/olmo2_probe2_1B_keep7f2_dolmino_arm4_peaklr20k \
python -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
  --model_path ../models/OLMo-2-0425-1B \
  --resume_from "$CKPT" \
  --keep_front_layers 7 --n_fresh_layers 2 \
  --data_path data/dolmino_now15b.npy \
  --output_dir "$OUT" \
  --max_steps 240000 \
  --lr 2e-5 --min_lr 2e-6 \
  --lr_inherited 2e-5 --min_lr_inherited 2e-6 \
  --seq_len 2048 --batch_size 8 \
  --warmup_steps 200500 \
  --save_every 5000 --gradient_checkpointing 1
```

Wall time: 20k steps × 2.05 s = 11.4 h.

## Expected result and its interpretation

Two clean readings compatible with A03's four certified axes:

* **Arm 4 shows gain where Arm 3 did not** → Arm 3's null was an LR artifact;
  peak-LR CPT can still move parametric knowledge past 200k. This licenses a
  "CPT saturation is not automatic" claim.
* **Arm 4 also shows no coherent trajectory** → the plateau is real; Dolmino
  CPT past 200k saturates on these axes at 1B-keep7 regardless of LR schedule.

Either outcome is scientifically informative. Anti-outcome: both arms
show idiosyncratic per-axis wobbles with no cross-axis pattern — same as Arm 3
alone, useless.

## Kill / promote

Not a kill gate for A03 (A03's kill was cleared at gate-1 pilot). Arm 4 is a
methodology improvement that makes the Arm 3 result interpretable.
