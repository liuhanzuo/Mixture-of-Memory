#!/bin/bash
# SFT-unfreeze step500 in-attn oracle probe vs OFF baseline (2026-06-18).
# Decisive break-the-32K-wall probe: did unfreezing the backbone for 500 steps
# teach the reader to CONSUME injected in-attn oracle KV?
#   OFF arm    (GPU0-3): no injection. baseline (~22 frozen).
#   ORACLE arm (GPU4-7): in-attn L16 topk64 oracle-only (~21 frozen).
# niah_single_1 / 4k / n=100, chunk=512 (training-aligned). 4 shards/arm = 25/card.
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
cd "$PROJECT_ROOT"

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

CKPT="outputs/sft_unfreeze_inattn_full/full_model_step000500.pt"
ADAPTER="outputs/sft_unfreeze_inattn_full/adapter_config.json"
MODEL="models/Meta-Llama-3-8B"

COMMON="--model_type mem_space --model_path $MODEL \
  --checkpoint $CKPT --adapter_config $ADAPTER \
  --tasks niah_single_1 --lengths 4k --num_samples 100 \
  --chunk_size 512 --num_shards 4"

mkdir -p logs babilong_results

# ---- OFF arm: GPU0-3 ----
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s nohup $PYTHON_BIN -m scripts.eval_ruler_mem_space \
    $COMMON --shard_index $s --device cuda:0 \
    --results_folder babilong_results --output_name sft_step500_probe_off \
    > logs/sft_step500_probe_off_shard${s}.log 2>&1 &
done

# ---- ORACLE in-attn arm: GPU4-7 (shard 0..3) ----
for s in 0 1 2 3; do
  gpu=$((s+4))
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYTHON_BIN -m scripts.eval_ruler_mem_space \
    $COMMON --shard_index $s --device cuda:0 \
    --use_inattn_kv --inattn_kv_layer 16 --inattn_kv_topk 64 \
    --oracle_evidence --oracle_layers 16 --inattn_oracle_only \
    --results_folder babilong_results --output_name sft_step500_probe_oracle \
    > logs/sft_step500_probe_oracle_shard${s}.log 2>&1 &
done

wait
echo "ALL_DONE sft_step500_inattn_probe"
