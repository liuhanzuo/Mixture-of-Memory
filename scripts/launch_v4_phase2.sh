#!/bin/bash
set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
mkdir -p logs outputs/v4_phase2
export PYTHONUNBUFFERED=1
export PYTHONPATH=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:$PYTHONPATH

torchrun --nproc_per_node=8 scripts/train_v4_chunk_memory.py \
  --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
  --data /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
  --num_slots 8 \
  --top_k 4 \
  --chunks_per_doc 16 \
  --lora_rank 16 \
  --lr 1e-4 \
  --max_steps 500 \
  --max_chunks 500 \
  --skip_chunks 0 \
  --seq_len 4096 \
  --batch_size 1 \
  --resume_checkpoint outputs/v4_chunk_memory/step_200.pt \
  --output_dir outputs/v4_phase2 \
  2>&1 | tee logs/v4_phase2_$(date +%Y%m%d_%H%M%S).log
