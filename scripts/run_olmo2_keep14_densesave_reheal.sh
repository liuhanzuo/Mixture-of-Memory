#!/usr/bin/env bash
# Paper B #103 -- keep14 dense-save RE-HEAL (matched-PPL crossing-point capture).
#
# WHY: the existing keep14 checkpoints (step128000 PPL10.827 / step153500 10.693 /
# final@200k 10.561) are ALL below the two matched-PPL targets needed for the
# NLL-vs-MMLU leg (random-front endpoint PPL 11.498, frozen-front endpoint 12.797).
# keep14 heal PPL is monotone-decreasing, so those two crossing points sit EARLY
# (~0-80k). This run reproduces the keep14 heal EXACTLY (same init / LR / batch /
# schedule / seed) and ONLY densifies checkpoint retention into a NEW output_dir so
# the early ckpts near PPL 11.5 / 12.8 are durably kept for later MMLU eval.
#
# ALIGNMENT with original keep14 (source of truth: scripts/run_olmo2_7B_keepN.sh
# launched with KEEP=14 N_FRESH=2; verified against logs/olmo2_7B_keep14fresh2.log
# header + scripts/train_olmo2_arch_probe2.py argparse defaults):
#   init      : inherited -- transplant OLMo-2-7B front14 + embed/norm/lm_head, add 2
#               fresh random-init layers -> 16L 4.06B, ALL trainable (train-all).
#               NOT --freeze_front, NOT --from_scratch.
#   batch     : bs=16 gaccum=1 world_size=8 -> eff_bs=128 ; seq_len=2048
#   lr        : UNIFORM 2e-5 for ALL params (min 2e-6).  *** IMPORTANT ***
#               The reference keep14 run (and the whole keep12/10/8 ladder) trained
#               EVERY param at lr_inherited=2e-5 -- NOT the differential fresh=1e-4
#               the header text claims.  Cause: at that code version _classify_param
#               ran on DDP-wrapped names ('module.<...>') and did NOT strip the
#               'module.' prefix, so fresh tail + lm_head fell through to 'inherited'
#               and got 2e-5 (see logs/olmo2_7B_keep14fresh2.log line 15:
#               '[optim] group inh_decay: 4060.1M params base_lr=2.00e-05' -- ALL
#               4060.1M params, no fresh group). The 'module.'-strip fix was added
#               LATER, so the current code would put fresh@1e-4. To reproduce the
#               EXACT keep14 PPL-vs-step trajectory that the matched-PPL crossing
#               points must align to, we force uniform 2e-5 via --lr 2e-5 --min_lr
#               2e-6 (numerically identical to the original single-group behavior;
#               the fresh/inh group split is cosmetic when both lrs are equal).
#   schedule  : warmup=150 then cosine to min over max_steps ; max_steps=200000
#               (FIXED -- the cosine horizon MUST match the original trajectory,
#                otherwise the PPL-vs-step curve diverges and crossing steps are wrong)
#   optim     : torch AdamW fp32, betas(0.9,0.95) eps1e-8 ; weight_decay=0.1 ; grad_clip=1.0
#   precision : fp32 master weights + bf16 autocast forward ; gradient_checkpointing=1
#   seed      : 42
#   data      : /dev/shm/dolmino_now15b.npy (7,570,911 rows, seq_len 2048)
#
# ONLY DIFFERENCE vs the original keep14 run:
#   save_every       500  -> 2500   (dense early saves)
#   milestone_every  5000 -> 2500   (durably retain the every-2500 ckpts; without this
#                                     the rolling-retention prunes non-5000 multiples)
#   keep_last_n      3    -> 0      (*** ROTATION DISABLED ***, see below)
#   output_dir       -> outputs/olmo2_keep14_densesave_reheal  (new; never clobbers keep14)
#
# *** WHY --keep_last_n 0 IS LOAD-BEARING HERE (added 2026-08-05) ***
# The trainers now rotate checkpoints (keep the newest N + a capped number of
# milestones); that is what stops a 200k-step 7B run from writing ~1.8 TB. THIS
# RUN MUST OPT OUT: its entire purpose is retaining EVERY dense save so the
# matched-PPL crossing points can be bracketed (step27500 is the only STRICT
# frozen-front match at dPPL 0.0386; step25000/30000 are its two-sided bracket;
# the random-front match at PPL 11.498 is expected near step63000 and has not been
# reached yet). --keep_last_n 0 disables rotation entirely, so no _save call can
# delete step2500..step27500 and destroy the bracket. Do NOT pass
# --keep_milestones here, and do NOT drop --keep_last_n 0 when resuming.
#
# NOTE: no wandb -- train_olmo2_arch_probe2.py has no wandb integration and the
# original keep14 run did not use it; progress is in the log (step/loss/ppl lines).
set -euo pipefail

PROJECT_ROOT="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
DATA_PATH="/dev/shm/dolmino_now15b.npy"
MODEL_PATH="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B"
OUT_DIR="outputs/olmo2_keep14_densesave_reheal"

mkdir -p "$OUT_DIR" logs

echo "[keep14-densesave-reheal] PYTHON=$PYTHON_BIN OUT=$OUT_DIR save_every=2500 milestone_every=2500 keep_last_n=0 (ROTATION DISABLED - every save retained for PPL-crossing bracketing) eff_bs=128 (bs16 ga1 x8)"

exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --model_path "$MODEL_PATH" \
    --keep_front_layers 14 \
    --n_fresh_layers 2 \
    --batch_size 16 \
    --grad_accumulation_steps 1 \
    --seq_len 2048 \
    --lr 2e-5 \
    --min_lr 2e-6 \
    --lr_inherited 2e-5 \
    --min_lr_inherited 2e-6 \
    --warmup_steps 150 \
    --weight_decay 0.1 \
    --grad_clip 1.0 \
    --seed 42 \
    --optimizer adamw \
    --max_steps 200000 \
    --gradient_checkpointing 1 \
    --save_every 2500 \
    --milestone_every 2500 \
    --keep_last_n 0
