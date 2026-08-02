#!/usr/bin/env bash
# Paper B P1.3 — LR-matched init control. 2-node 16-card DDP over IB.
#   Fully random-init 16L (keep14+fresh2 SHELL, --from_scratch) at UNIFORM peak
#   LR 2e-5 (both --lr and --lr_inherited = 2e-5). Removes the init x LR confound
#   vs the existing fromscratch arm (lr_fresh=1e-4) and matches keep14's 2e-5.
#   from_scratch puts ALL params in the 'fresh' bucket at --lr, so --lr 2e-5 =>
#   uniform 2e-5 across the whole model.
#
#   eff-batch = 128 = per_device2 x accum4 x 16 ranks (2 nodes x 8 GPU).
#   seq_len 2048, max_steps 200000, warmup 150, wd 0.1, grad_clip 1.0,
#   save_every 5000 (rolling-retention keeps every-5000 milestone permanently ->
#   50000/100000/150000 preserved automatically), gradient_checkpointing on,
#   seed 42, fp32 master weights.
#
# Runs on BOTH nodes with $1=node_rank ($2=log path). .73=rank0 (master), .82=rank1.
# PROJECT_ROOT is derived from THIS script's own location, so each node uses its
# own alias mount (.73 wzc1 / .82 zwfy6) automatically.
#
# IB (user requires RoCE): NCCL_IB_DISABLE=0, NCCL_IB_GID_INDEX=3 (RoCE v2),
# NCCL_IB_HCA=mlx5_bond_1; rdzv control plane on bond1. Override NCCL_IB_DISABLE=1
# to fall back to TCP-over-bond if IB bring-up fails.
set -euo pipefail

NODE_RANK="${1:?usage: run_olmo2_p13_node.sh <node_rank> <log>}"
LOG="${2:?usage: run_olmo2_p13_node.sh <node_rank> <log>}"

# Resolve PROJECT_ROOT from this script's directory (scripts/..), so the correct
# per-node alias path is used without hardcoding wzc1 vs zwfy6.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

case "$LOG" in
  /*) : ;;
  *) LOG="$PROJECT_ROOT/$LOG" ;;
esac
mkdir -p "$(dirname "$LOG")"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
MODEL_PATH="${MODEL_PATH:-../models/OLMo-2-1124-7B}"
DATA_PATH="${DATA_PATH:-/dev/shm/dolmino_now15b.npy}"
OUT_DIR="${OUT_DIR:-outputs/olmo2_p13_scratch16_lr2e5_uniform}"
MASTER_IP="${MASTER_IP:-28.85.35.73}"
PORT="${PORT:-29517}"
NPROC="${NPROC:-8}"
BS="${BS:-2}"
GA="${GA:-4}"          # 16 ranks x 2 x 4 = 128 eff-batch
SEED="${SEED:-42}"
RESUME_FROM="${RESUME_FROM:-}"

mkdir -p "$OUT_DIR" logs

# NCCL: RoCE/IB (user requirement). rdzv control plane on bond1.
# NCCL_NET_GDR_LEVEL=0 + PCI_RELAXED_ORDERING=1: at 8-ranks/node the GPU-Direct
# RDMA MR-registration path (ibv_reg_mr_iova2) fails with "Invalid argument" on
# these H20+mlx5_bond_1 boxes; disabling GDR keeps IB transport (data still rides
# RoCE, just staged via host memory) and the 16-rank all-reduce smoke PASSES.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_bond_1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[run_olmo2_p13_node] rank=$NODE_RANK master=$MASTER_IP:$PORT nproc=$NPROC \
eff_bs=$((BS*GA*NPROC*2)) PROJECT_ROOT=$PROJECT_ROOT OUT=$OUT_DIR IB_DISABLE=$NCCL_IB_DISABLE"

CMD=(
  "$PYTHON_BIN" -m torch.distributed.run
    --nnodes 2 --node_rank "$NODE_RANK" --nproc_per_node "$NPROC"
    --master_addr "$MASTER_IP" --master_port "$PORT"
  scripts/train_olmo2_arch_probe2.py
    --data_path "$DATA_PATH"
    --output_dir "$OUT_DIR"
    --model_path "$MODEL_PATH"
    --from_scratch
    --keep_front_layers 14
    --n_fresh_layers 2
    --lr 2e-5
    --min_lr 2e-6
    --lr_inherited 2e-5
    --min_lr_inherited 2e-6
    --batch_size "$BS"
    --grad_accumulation_steps "$GA"
    --seq_len 2048
    --max_steps 200000
    --warmup_steps 150
    --weight_decay 0.1
    --grad_clip 1.0
    --save_every 5000
    --gradient_checkpointing 1
    --seed "$SEED"
)
[ -n "$RESUME_FROM" ] && CMD+=(--resume_from "$RESUME_FROM")

printf '  %q' "${CMD[@]}"; echo
exec "${CMD[@]}" >>"$LOG" 2>&1
