#!/bin/bash
set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
mkdir -p logs
export PYTHONUNBUFFERED=1
torchrun --nproc_per_node=8 scripts/train_mem_space_pg19.py \
  --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
  --data /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
  --use_infini_attention \
  --infini_beta_init -1.0 \
  --infini_beta_lr 0.1 \
  --infini_memory_scale_init 5.0 \
  --infini_trainable_proj \
  --lr 1e-3 \
  --max_steps 3000 \
  --max_chunks 500 \
  --skip_chunks 0 \
  --seq_len 4096 \
  --batch_size 1 \
  --num_slots 128 \
  --top_k 64 \
  --slot_init random \
  --slot_init_noise 0.02 \
  --output_dir outputs/infini_v5_trainable_proj
