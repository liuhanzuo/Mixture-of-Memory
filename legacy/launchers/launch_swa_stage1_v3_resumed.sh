#!/bin/bash
set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

LOG="logs/swa_stage1_v3_resumed_node0_$(date +%Y%m%d_%H%M).log"
echo "[launch] Starting swa_stage1_v3_resumed at $(date)" | tee "$LOG"

torchrun \
  --nproc_per_node 8 \
  --master_addr 127.0.0.1 \
  --master_port 29501 \
  scripts/train_mem_space_pg19.py \
  --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
  --data data/pg19_chunks_llama3.npy \
  --max_chunks 5916 \
  --swa_window 512 \
  --seq_len 4096 \
  --niah_mix_fraction 0.10 \
  --niah_max_N 16 \
  --max_steps 20000 \
  --lr 3e-4 \
  --num_slots 512 \
  --top_k 64 \
  --shared_memory_bank \
  --unfreeze_hidden_to_slot \
  --writeback_warmup_steps 500 \
  --writeback_gate_max 0.3 \
  --load_balance_weight 0.01 \
  --batch_size 2 \
  --skip_chunks 0 \
  --init_from outputs/swa_stage1_v3/mem_space_adapter_step010000.pt \
  --output_dir outputs/swa_stage1_v3_resumed \
  2>&1 | tee -a "$LOG"
