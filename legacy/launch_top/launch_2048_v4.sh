#!/bin/bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory

export PROJECT_DIR=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
export CUDA_VISIBLE_DEVICES=2,3,4,5
export NUM_GPUS=4
export MEMORY_SLOTS=2048
export MAX_STEPS=5000
export BATCH_SIZE=1
export GRAD_ACCUM=8
export SEQ_LEN=2048
export OUTPUT_DIR=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/sparse_memory_ablation_2048_v4

mkdir -p "$OUTPUT_DIR"
bash scripts/run_train_sparse_memory.sh > "$OUTPUT_DIR/train.log" 2>&1
