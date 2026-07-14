#!/usr/bin/env bash
# 2-node 16-GPU FSDP: Hunyuan-A13B-Pretrain keep13+fresh2 minimal-arch continue-train.
# Invoke on BOTH nodes; $1 = node_rank (0 on lhz master, 1 on lhz2 worker).
#
# Why 16-card: the 37.9B (keep13+fresh2) model with FULL_SHARD + fp32 master + AdamW
#   (backbone NOT frozen -> all params trainable) does NOT fit seq_len=2048 on a single
#   8x H200 node (OOM'd at ~136/140GB even with BACKWARD_POST). Two nodes halve the
#   per-rank shard (fp32 master 9.5 + grad 9.5 + AdamW m/v 19 = ~38GB/rank persistent),
#   leaving ample room for seq_len=2048 activations -> restores Qwen/Hy3 seq_len alignment.
#
# NCCL recipe (verified 2026-07-14): eth0 (172.16.x, cross-node routable; 2-node TCP
#   smoke on 172.16.141.84<->172.16.206.31:29562 PASSED) + IB disabled + c10d rdzv.
#   master_addr = lhz eth0 = 172.16.141.84.
#
# keep13 / fresh2 / backbone-unfrozen / seq_len=2048 are FIXED. eff_bs = bs1 x accum8 x
#   16 ranks = 128.
set -euo pipefail
NODE_RANK="${1:?usage: run_hunyuan_a13b_keep13_fresh2_2node.sh <node_rank 0|1>}"

# Both nodes share the gpfs mount /volume/haru (verified 2026-07-14: lhz2 also mounts
# it), so model / venv / data / code are the SAME path on both -> zero rsync.
PROJECT_ROOT="${PROJECT_ROOT:-/volume/haru/Mixture-of-Memory}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv_hy3/bin/python}"
MASTER_IP="${MASTER_IP:-172.16.141.84}"
PORT="${PORT:-29562}"

cd "$PROJECT_ROOT"
export WANDB_MODE=offline OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 NCCL_TIMEOUT=7200
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1

LOG="$PROJECT_ROOT/logs/hunyuan_a13b_keep13_fresh2_16card_node${NODE_RANK}.log"
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs/hunyuan_a13b_keep13_fresh2_16card"

"$PYTHON_BIN" -m torch.distributed.run \
  --nnodes 2 --node_rank "$NODE_RANK" --nproc_per_node 8 \
  --master_addr "$MASTER_IP" --master_port "$PORT" \
  scripts/train_hunyuan_a13b_probe2.py \
  --model_path models/Hunyuan-A13B-Pretrain \
  --data_path data/slimpajama_chunks_2048_hunyuan.npy \
  --output_dir outputs/hunyuan_a13b_keep13_fresh2_16card \
  --keep_front_layers 13 --n_fresh_layers 2 \
  --max_steps 20000 --seq_len 2048 \
  --batch_size 1 --grad_accumulation_steps 8 \
  --lr 1e-4 --lr_inherited 2e-5 --warmup_steps 100 \
  --save_every 1000 --log_every 10 --gradient_checkpointing 1 \
  --fsdp_cpu_offload 0 \
  > "$LOG" 2>&1
echo "A13B_KEEP13F2_16CARD_NODE${NODE_RANK}_EXIT_$?" >> "$LOG"
