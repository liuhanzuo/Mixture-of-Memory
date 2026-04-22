#!/bin/bash
set -e
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

python scripts/train_gated_sparse_memory.py \
  --model_name_or_path models/Llama--Llama2-7b \
  --data_path data/pg19_train.jsonl \
  --output_dir outputs/sparse_memory_concat_fusion_5k \
  --max_steps 2 \
  --num_slots 128 --top_k 8 --window_size 256 \
  --bypass_bias_init -2.0 --bypass_gate_lr_multiplier 10.0 \
  --bf16 --gradient_checkpointing --logging_steps 1 \
  2>&1
