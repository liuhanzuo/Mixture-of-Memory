#!/bin/bash
set -e
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate

echo "=== Recovery7: resuming from checkpoint-4000 (steps 4017-5000) ==="
echo "=== write_top_k=8 (FIXED mode), cublasLt divide error was root cause ==="
echo "=== Start: $(date) ==="

torchrun --nnodes=1 --nproc_per_node=8 scripts/train_gated_sparse_memory.py \
  --model_name_or_path models/Llama--Llama2-7b \
  --data_path /tmp/pg19_small.jsonl \
  --output_dir outputs/sparse_memory_concat_fusion_v1_fixed \
  --max_seq_length 4096 \
  --num_slots 128 \
  --window_size 256 \
  --top_k 8 \
  --write_top_k 8 \
  --ema_alpha 0.1 \
  --bypass_bias_init -2.0 \
  --bypass_gate_lr_multiplier 10.0 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --max_steps 5000 \
  --warmup_steps 100 \
  --logging_steps 50 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --bf16 \
  --gradient_checkpointing \
  --ddp_find_unused_parameters \
  > outputs/sparse_memory_concat_fusion_v1_fixed/recovery7.log 2>&1

echo "=== End: $(date) ==="
