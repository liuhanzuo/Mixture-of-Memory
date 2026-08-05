#!/usr/bin/env bash
# PaperB P0.5 Arm A (ShortGPT contiguous-16, keep layers 0-15, n_fresh=0) — MIGRATED to LOCAL 8xB200 (wzc1)
# for ~9x speedup over H20 (user 2026-08-02: put TODOList tasks on faster B200). From-scratch prune from base
# (build_shortgpt_model transplants from base, no ckpt needed). Was step2060 on .104 H20 (killed, ~5.4h discarded).
# B200 ~1s/step -> 200k ~2.5d vs H20 9.5s/step ~22d. Identical recipe to .104 launch (seed42, bs2xga8=eff128, 200k).
set -u
WD=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$WD" || exit 1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs outputs
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_shortgpt.py \
  --data_path /dev/shm/dolmino_now15b.npy \
  --output_dir outputs/olmo2_p05_armA_contig16 \
  --model_path ../models/OLMo-2-1124-7B \
  --keep_layer_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --lr_inherited 2e-5 --min_lr_inherited 2e-6 \
  --batch_size 2 --grad_accumulation_steps 8 --seq_len 2048 \
  --max_steps 200000 --warmup_steps 150 --weight_decay 0.1 --grad_clip 1.0 \
  --save_every 5000 --extra_save_steps 50000,100000,150000 \
  --keep_last_n ${KEEP_LAST_N:-3} --keep_milestones ${KEEP_MILESTONES:-8} \
  --gradient_checkpointing 1 --seed 42
