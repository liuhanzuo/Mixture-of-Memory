#!/bin/bash
set -e
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 --master_port 29501 scripts/train_gated_sparse_memory.py \
  --model_name_or_path models/Llama--Llama2-7b \
  --data_path data/pg19_train.jsonl \
  --output_dir outputs/sparse_memory_smoke_test \
  --overwrite_output_dir \
  --max_steps 5 \
  --save_steps 3 \
  --num_slots 128 --top_k 8 --window_size 256 \
  --bypass_bias_init -2.0 --bypass_gate_lr_multiplier 10.0 \
  --bf16 --gradient_checkpointing --logging_steps 1 \
  2>&1
