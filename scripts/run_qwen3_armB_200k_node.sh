#!/usr/bin/env bash
# 2-node 16-GPU DDP: resume Qwen3-8B armB (keep12+fresh2 heal) step20000 -> 200000.
# Invoke on BOTH nodes; $1 = node_rank (0 on .24.104 master, 1 on .85.73 worker).
# Verified NCCL recipe: bond1 (28.83.x mgmt seg, cross-pod routable) + IB disabled,
#   master_addr = .24.104 bond1 = 28.83.24.104. 2-node all_reduce smoke PASSED.
# eff_bs = bs4 x accum2 x 16 ranks = 128 (matches original armB schedule).
set -euo pipefail
NODE_RANK="${1:?usage: run_qwen3_armB_200k_node.sh <node_rank>}"

PROJECT_ROOT="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
PY_DIR="/opt/conda/envs/torch-base/bin"
TORCHRUN="$PY_DIR/torchrun"
MASTER_IP="28.83.24.104"
PORT="29562"

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=1
export WANDB_MODE=offline

LOG="$PROJECT_ROOT/logs/qwen3_armB_200k_node${NODE_RANK}.log"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

setsid nohup "$TORCHRUN" \
  --nnodes 2 --node_rank "$NODE_RANK" --nproc_per_node 8 \
  --master_addr "$MASTER_IP" --master_port "$PORT" \
  scripts/train_qwen3_arch_probe2.py \
  --data_path data/slimpajama_chunks_2048_qwen3.npy \
  --output_dir outputs/qwen3_minarch_armB_f12k2_200k \
  --model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
  --keep_front_layers 12 --n_fresh_layers 2 \
  --resume_from outputs/qwen3_minarch_armB_f12k2_20k/step20000.pt \
  --max_steps 200000 \
  --batch_size 4 --grad_accumulation_steps 2 --seq_len 2048 \
  --lr 1e-4 --min_lr 1e-5 --lr_inherited 2e-5 --min_lr_inherited 2e-6 \
  --gradient_checkpointing 1 --save_every 500 --log_every 20 \
  --device auto > "$LOG" 2>&1 &
echo "NODE${NODE_RANK}_LAUNCHED pid=$! log=$LOG"
