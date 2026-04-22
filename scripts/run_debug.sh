#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/train_gated_sparse_memory.py \
  --output_dir outputs/sparse_memory_concat_fusion_v1 \
  --model_name_or_path models/Llama--Llama2-7b \
  --data_path /tmp/pg19_tiny.jsonl \
  --max_steps 5 \
  --num_slots 128 \
  --top_k 8 \
  --window_size 256 \
  --bypass_bias_init -2.0 \
  --bypass_gate_lr_multiplier 10.0 \
  --bf16 \
  --gradient_checkpointing \
  --logging_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --max_seq_length 512 \
  --overwrite_output_dir
