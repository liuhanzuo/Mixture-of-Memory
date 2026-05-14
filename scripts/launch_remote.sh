#!/bin/bash
# Generic remote launcher - called with: bash scripts/launch_remote.sh <tag> <extra_args...>
set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
TAG=$1
shift
mkdir -p logs outputs/v4_ablation_${TAG}
export PYTHONUNBUFFERED=1
torchrun --nproc_per_node=8 scripts/train_v4_chunk_memory.py \
  --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
  --data /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
  --max_chunks 500 --skip_chunks 0 --seq_len 4096 --batch_size 1 \
  --output_dir outputs/v4_ablation_${TAG} \
  "$@" \
  2>&1 | tee logs/v4_ablation_${TAG}_$(date +%Y%m%d_%H%M%S).log
