#!/usr/bin/env bash
# Memory/speed probe for keep14-distill on ONE B200, run while the step5000 ckpt
# transfers. Question: can we drop --gradient_checkpointing (a mathematically
# EXACT change -- recomputation vs stored activations yield identical gradients)
# and buy ~30% throughput over the 200k-step run?
#
# The static footprint is bs-independent and identical to the H20 run:
#   student fp32 15.13 + grads fp32 15.13 + AdamW8bit 7.56 + teacher bf16 13.59
#   = 51.41 GiB.  H20 peaked 94.6 of 97.8 GiB at bs=16 WITH checkpointing.
# A single-GPU peak is a good proxy for per-rank DDP peak (plain DDP does NOT
# shard params/grads/optim; it only adds ~small bucket + NCCL buffers).
#
# Writes to a THROWAWAY output_dir and is killed before the unconditional final
# save, so it cannot touch outputs/olmo2_probe2_7B_keep14fresh2_distill/.
set -u
R=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
cd $R
GC=${GC:-0}
BS=${BS:-16}
PY=/opt/conda/envs/torch-base/bin/python
TMP=/dev/shm/probe_gc${GC}_bs${BS}
mkdir -p $TMP logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${DEV:-0}

$PY scripts/train_olmo2_arch_probe2_distill.py \
  --data_path /dev/shm/dolmino_now15b_wzc1.npy \
  --output_dir $TMP \
  --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
  --keep_front_layers 14 --n_fresh_layers 2 \
  --distill_teacher_model /apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
  --distill_lambda 0.6 --teacher_topk 64 \
  --lr 1e-4 --lr_inherited 2e-5 --min_lr 1e-5 --min_lr_inherited 2e-6 \
  --batch_size $BS --grad_accumulation_steps 1 --seq_len 2048 \
  --max_steps 40 --warmup_steps 150 --weight_decay 0.1 --grad_clip 1.0 \
  --save_every 100000 --log_every 5 \
  --max_rows 20000 \
  --gradient_checkpointing $GC 2>&1 | tail -25
