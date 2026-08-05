#!/usr/bin/env bash
# PaperB P0.5 Arm B (ShortGPT retained-final14+2fresh, keep [0-12,31]+2fresh, split LR) — on .252 B200 (wzc1)
# for ~5x speedup + recipe-parity with Arm A on LOCAL B200 (both .venv torch2.13). From-base prune (no ckpt).
# Was step2240 on .73 H20 (killed, discarded). Identical recipe to .73 launch (seed42, bs2xga8=eff128, 200k,
# split LR inherited 2e-5 / fresh 1e-4). Run on .252 via SSH; wzc1 shared FS so outputs land on shared disk.
set -u
WD=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$WD" || exit 1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs outputs
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_shortgpt_fresh.py \
  --data_path /dev/shm/dolmino_now15b.npy \
  --output_dir outputs/olmo2_p05_armB_final14_fresh2 \
  --model_path ../models/OLMo-2-1124-7B \
  --keep_layer_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,31 --n_fresh_layers 2 \
  --lr_inherited 2e-5 --min_lr_inherited 2e-6 --lr_fresh 1e-4 --min_lr_fresh 1e-5 \
  --batch_size 2 --grad_accumulation_steps 8 --seq_len 2048 \
  --max_steps 200000 --warmup_steps 150 --weight_decay 0.1 --grad_clip 1.0 \
  --save_every 5000 --extra_save_steps 50000,100000,150000 \
  --keep_last_n ${KEEP_LAST_N:-3} --keep_milestones ${KEEP_MILESTONES:-8} \
  --gradient_checkpointing 1 --seed 42
