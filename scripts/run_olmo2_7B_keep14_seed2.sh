#!/usr/bin/env bash
# Paper B P1.2 -- SECOND SEED of the keep14+fresh2 7B healing arm.
#
# Purpose: the paper currently reports ONE training trajectory for this arm
# (outputs/olmo2_probe2_7B_keep14fresh2, 2026-07-16..07-21). This run is an
# independent replicate whose ONLY intended difference is the random draw for
# the 2 fresh tail layers' initialisation, so P1.2 can report seed-level spread.
#
# ---------------------------------------------------------------------------
# WHY THIS IS A SEPARATE SCRIPT AND NOT `KEEP=14 bash scripts/run_olmo2_7B_keepN.sh`
# ---------------------------------------------------------------------------
# (1) CLOBBER. run_olmo2_7B_keepN.sh derives OUT_DIR from KEEP/N_FRESH only, so
#     KEEP=14 N_FRESH=2 resolves to outputs/olmo2_probe2_7B_keep14fresh2 -- the
#     ORIGINAL run's directory, whose step200000.pt/final.pt are paper-bearing.
# (2) LR EQUIVALENCE (the important one -- see the block below). Reproducing the
#     original's ACTUAL optimisation requires --lr 2e-5 --min_lr 2e-6, which the
#     shared launcher hardcodes to 1e-4 / (default) 1e-5.
#
# ---------------------------------------------------------------------------
# ★ THE UNIFORM-LR EQUIVALENCE -- READ BEFORE CHANGING --lr / --min_lr
# ---------------------------------------------------------------------------
# The original run was launched with --lr 1e-4 --lr_inherited 2e-5, i.e. it ASKED
# for a differential LR. It did not get one. At that time (trainer @ afdfa66)
# build_param_groups() ran AFTER the DDP wrap, and _classify_param() did not
# strip the 'module.' prefix that DDP.named_parameters() prepends -- so EVERY
# trainable param fell through to 'inherited'. logs/olmo2_7B_keep14fresh2.log
# confirms it, only two groups were ever created:
#     [optim] group inh_decay:   4060.1M params base_lr=2.00e-05 min_lr=2.00e-06
#     [optim] group inh_nodecay:    0.3M params base_lr=2.00e-05 min_lr=2.00e-06
# -> the original arm trained at a UNIFORM 2e-5 (cosine to 2e-6). --lr 1e-4 was
#    a silent no-op, and arch_meta.json's "lr_fresh": 1e-4 is aspirational, not
#    what ran. (The 'module.' strip landed later, in 7a330ce, 2026-08-03.)
#
# So on today's FIXED trainer, re-issuing the original command verbatim would
# create real fresh_* groups at 1e-4 and change the optimisation -- a SECOND
# changed variable on top of the seed, which would make the seed-variance number
# worthless. Passing --lr 2e-5 --min_lr 2e-6 instead gives every param the same
# (base_lr, min_lr, weight_decay, betas, eps) it had in the original run. AdamW
# is per-parameter, so 4 groups with identical hyper-parameters is numerically
# the same optimiser as the original's 2 groups. The log will now print four
# groups (fresh_decay/fresh_nodecay/inh_decay/inh_nodecay) ALL at 2.00e-05 /
# 2.00e-06 -- that is the expected, correct signature of this equivalence.
#
# DO NOT "restore" --lr 1e-4 here thinking it matches the original. It does not.
#
# ---------------------------------------------------------------------------
# ★ WHAT --seed ACTUALLY CONTROLS (scope of the P1.2 claim)
# ---------------------------------------------------------------------------
# set_seed() seeds python/numpy/torch/CUDA before model construction. In THIS
# trainer that means the seed controls exactly ONE thing that matters:
#   * the random init of the 2 fresh tail layers (the 14 inherited layers are
#     overwritten by the transplant; embed/norm/lm_head are inherited too).
# It does NOT control data order: the loader uses DistributedSampler(ds,
# shuffle=True) with NO seed= argument, so its permutation is torch.randperm
# under g.manual_seed(0 + epoch) -- a private generator, fixed at seed 0,
# independent of --seed and of the global torch seed. Both runs therefore see
# the SAME data in the SAME order.
# It does NOT control dropout either: OLMo-2-1124-7B config has
# attention_dropout=0.0 and the trainer adds none.
# => P1.2 must describe this as FRESH-BLOCK INITIALISATION variance under a
#    fixed data order, NOT as full training-seed (init x data-order) variance.
#
# ---------------------------------------------------------------------------
# ★ THE ORIGINAL RUN HAD NO SEED AT ALL
# ---------------------------------------------------------------------------
# --seed did not exist until c57c4cb (2026-08-03); the original launched
# 2026-07-16 on trainer afdfa66, which never called any *.manual_seed(). Its
# fresh-block init was drawn from torch's default nondeterministic seed and was
# never recorded (hence no "seed" key in its arch_meta.json). Consequences:
#   * seed 1 is an UNKNOWN, UNREPRODUCIBLE draw. It cannot be re-run bit-exactly.
#   * the (seed1, seed2) pair is still a valid 2-sample look at init spread,
#     because both are independent draws from the same init distribution.
#   * only THIS run's seed is reproducible. Report seed 1 as "unseeded/unrecorded".
# DDP consistency was never at risk in either run: DDP's _sync_module_states
# broadcasts rank-0 parameters at wrap time.
#
# Usage:  bash scripts/run_olmo2_7B_keep14_seed2.sh
set -euo pipefail

PROJECT_ROOT="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"

# .venv/bin/python on the wzc1 nodes no longer has torch (verified 2026-08-07);
# the conda env carries torch 2.13.0, matching the 2.13 stack the original ran on.
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"

SEED="${SEED:-1234}"          # seed 2. Trainer default is 42; 1234 is used so a
                              # default-seeded run can never be mistaken for this one.
KEEP=14
N_FRESH=2
RESUME_FROM="${RESUME_FROM:-}"

DATA_PATH="/dev/shm/dolmino_now15b.npy"      # md5 e4da8db79c264da70f5b5be5a26f342d,
                                             # verified identical on LOCAL and .252
MODEL_PATH="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B"
OUT_DIR="outputs/olmo2_probe2_7B_keep14fresh2_seed${SEED}"
LOG_FILE="logs/olmo2_7B_keep14fresh2_seed${SEED}.log"

# Checkpoint retention. This is the one difference from the original that CANNOT
# affect the trained weights -- it only decides which ckpts survive on disk. The
# original ran under the old unbounded-milestone policy (~1.8 TB/arm); wzc1 is at
# 90% use, so this run is bounded instead. KEEP_STEPS pins the P1.2 pre-registered
# evaluation grid (TODOList: 0/25k/50k/100k/128k/153.5k/200k) so retention can
# never delete a paper-bearing point.
SAVE_EVERY="${SAVE_EVERY:-500}"
MILESTONE_EVERY="${MILESTONE_EVERY:-5000}"
KEEP_LAST_N="${KEEP_LAST_N:-3}"
KEEP_MILESTONES="${KEEP_MILESTONES:-8}"
KEEP_STEPS="${KEEP_STEPS:-25000,50000,100000,128000,153500,200000}"

mkdir -p "$OUT_DIR" logs

echo "[keep14_seed2] SEED=$SEED KEEP=$KEEP N_FRESH=$N_FRESH -> $OUT_DIR (log $LOG_FILE)"
echo "[keep14_seed2] uniform LR 2e-5 -> 2e-6 (reproduces the original's ACTUAL optimisation; see header)"
echo "[keep14_seed2] retention: save_every=$SAVE_EVERY milestone_every=$MILESTONE_EVERY keep_last_n=$KEEP_LAST_N keep_milestones=$KEEP_MILESTONES keep_steps=$KEEP_STEPS"

[ -z "$RESUME_FROM" ] && : > "$LOG_FILE"

nohup "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --model_path "$MODEL_PATH" \
    --keep_front_layers "$KEEP" \
    --n_fresh_layers "$N_FRESH" \
    --batch_size 16 \
    --grad_accumulation_steps 1 \
    --seq_len 2048 \
    --lr 2e-5 \
    --min_lr 2e-6 \
    --lr_inherited 2e-5 \
    --max_steps 200000 \
    --gradient_checkpointing 1 \
    --seed "$SEED" \
    --save_every "$SAVE_EVERY" \
    --milestone_every "$MILESTONE_EVERY" \
    --keep_last_n "$KEEP_LAST_N" \
    --keep_milestones "$KEEP_MILESTONES" \
    --keep_steps "$KEEP_STEPS" \
    ${RESUME_FROM:+--resume_from "$RESUME_FROM"} \
  >>"$LOG_FILE" 2>&1 &

echo "[keep14_seed2] launched pid=$! ; tail -f $LOG_FILE"
