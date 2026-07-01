#!/usr/bin/env bash
# Efficiency micro-benchmark: single stage1 chunk128 run, parameterized for
# 16-GPU (2x8) vs 8-GPU (1x8) fair comparison at IDENTICAL eff batch 32.
#   Arm B (16-GPU): NNODES=2 NPROC_PER_NODE=8 GA=2  -> 16 x bs1 x ga2 = eff 32
#   Arm A ( 8-GPU): NNODES=1 NPROC_PER_NODE=8 GA=4  ->  8 x bs1 x ga4 = eff 32
# Same chunk128 / 8B / grad-ckpt config as launch_progressive_chunk_h800.sh stage1.
# Each step processes the same amount of data -> steps/s directly comparable.
#
# USAGE (16-GPU, run on BOTH nodes, only NODE_RANK differs):
#   node0: NNODES=2 GA=2 OUT=outputs/effi_test_16gpu_c128 NODE_RANK=0 MASTER_ADDR=30.203.138.213 MASTER_PORT=29820 bash scripts/effi_test_chunk128_h800.sh
#   node1: NNODES=2 GA=2 OUT=outputs/effi_test_16gpu_c128 NODE_RANK=1 MASTER_ADDR=30.203.138.213 MASTER_PORT=29820 bash scripts/effi_test_chunk128_h800.sh
# USAGE (8-GPU, node0 only):
#   node0: NNODES=1 GA=4 OUT=outputs/effi_test_8gpu_c128 NODE_RANK=0 MASTER_ADDR=30.203.138.213 MASTER_PORT=29821 bash scripts/effi_test_chunk128_h800.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_jn2/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-30.203.138.213}"
MASTER_PORT="${MASTER_PORT:-29820}"
GA="${GA:-2}"
OUT="${OUT:-outputs/effi_test_16gpu_c128}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_SL="${NCCL_IB_SL:-3}"
export NCCL_IB_TC="${NCCL_IB_TC:-160}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
LOG="logs/effi_test_c128_$(basename "$OUT")_node${NODE_RANK}.log"
mkdir -p logs "$OUT"

echo "[effi_test] NNODES=$NNODES NPROC=$NPROC_PER_NODE GA=$GA OUT=$OUT node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT total_steps=$TOTAL_STEPS $(date)" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="$GPUS" "$PYBIN" -m torch.distributed.run \
  --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir "$OUT" --total_steps "$TOTAL_STEPS" --lr 1e-4 --warmup_steps 100 \
  --chunk_size 128 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
  --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps "$GA" \
  --curriculum 0:3 --bptt_window 2 --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l3_diversity_threshold 0.5 \
  --l_recon_weight 0.0 --route_aux_weight 1.0 \
  --use_memory_xattn --memory_xattn_gate_init 0.4 \
  --save_interval 100000 --eval_interval 0 --eval_samples 30 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name "effi_test_$(basename "$OUT")" --dtype bfloat16 --attn_impl sdpa --seed 42 \
  >>"$LOG" 2>&1

echo "[effi_test] DONE $(date)" | tee -a "$LOG"
