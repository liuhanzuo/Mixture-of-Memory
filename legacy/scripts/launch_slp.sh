#!/bin/bash
set -e
cd /root/Mixture-of-Memory
export TORCH_ELASTIC_TRACEBACK=1

LOGDIR=/root/Mixture-of-Memory/logs
OUTDIR=/root/Mixture-of-Memory/outputs
MODEL=/root/Mixture-of-Memory/models/Llama--Llama2-7b/
DATA=/root/Mixture-of-Memory/data/slimpajama_chunks_4096.npy

EXP=$1
PORT=$2
shift 2
EXTRA="$@"

mkdir -p "$LOGDIR" "$OUTDIR/$EXP"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MASTER_PORT=$PORT torchrun \
  --nproc_per_node=8 \
  scripts/train_sparse_memory.py \
  --base_model "$MODEL" \
  --data_path "$DATA" \
  --output_dir "$OUTDIR/$EXP" \
  --lr 2e-5 --batch_size 2 --grad_accumulation_steps 4 \
  --max_steps 5000 --log_every 50 --save_every 1000 \
  --mixed_precision bf16 --seq_len 4096 \
  $EXTRA \
  2>&1 | tee "$LOGDIR/train_${EXP}.log"

echo "TRAINING_EXIT_CODE=$?"
