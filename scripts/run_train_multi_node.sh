#!/bin/bash
# ============================================================
# MAG 多节点分布式训练启动脚本
# ============================================================
# 用法 (4 机 32 卡示例):
#   在每台机器上分别运行, 仅 NODE_RANK 不同:
#     NODE_RANK=0 bash scripts/run_train_multi_node.sh   # 主节点
#     NODE_RANK=1 bash scripts/run_train_multi_node.sh   # 工作节点 1
#     NODE_RANK=2 bash scripts/run_train_multi_node.sh   # 工作节点 2
#     NODE_RANK=3 bash scripts/run_train_multi_node.sh   # 工作节点 3
#
# 或者通过调度系统 (如 TJM/SLURM) 自动分配 NODE_RANK:
#   srun --nodes=4 --ntasks-per-node=1 bash scripts/run_train_multi_node.sh
# ============================================================

set -euo pipefail

# ==================== 集群配置 ====================
# 节点数和每节点 GPU 数
NNODES=${NNODES:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# 主节点地址 (填主节点 IP, 所有节点必须一致)
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

# 当前节点编号 (0=主节点, 1~N-1=工作节点)
# 方式1: 手动设置 NODE_RANK=0/1/2/3
# 方式2: SLURM 环境自动读取 SLURM_NODEID
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}

echo "============================================================"
echo "MAG 多节点训练 - 节点 ${NODE_RANK}/${NNODES}"
echo "  MASTER: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  GPUs/Node: ${NPROC_PER_NODE}"
echo "  Total GPUs: $((NNODES * NPROC_PER_NODE))"
echo "============================================================"

# ==================== 路径配置 ====================
MODEL_PATH="../models/Qwen--Qwen3-8b/"
DATA_PATH="data/mag_train_generated.jsonl"
OUTPUT_DIR="outputs/mag_anti_tf_multi"

# ==================== 训练超参 ====================
NUM_EPOCHS=3
LR=1e-4
MAX_SEQ_LEN=4096
DEEP_ENCODE_LAYERS=8
MAX_REAL_SAMPLES=27832
SLIDING_WINDOW=4096

# ==================== Anti-Teacher-Forcing 配置 ====================
# 这些参数是为了解决 teacher forcing 下 CrossAttention 
# 直接从 V 中"抄"答案导致 lm_loss → 0 的问题
LABEL_SMOOTHING=0.1           # Label Smoothing 系数 (阻止 loss → 0)
KL_BETA=0.5                   # KL 约束系数 (限制记忆对输出分布的影响幅度)
KL_TEMPERATURE=2.0            # KL 温度 (较高温度使分布更平滑)
DETACH_VALUE="--detach_value"  # V 分支 stop-gradient (阻止"抄答案"的梯度路径)
# 如果要禁用 detach_value, 改为: DETACH_VALUE="--no_detach_value"

# Scheduled Sampling 配置 (训练后期逐步引入 token corruption)
SCHEDULED_SAMPLING=""  # 默认关闭, 首次训练建议先不开
# 如果要启用, 取消下面的注释:
# SCHEDULED_SAMPLING="--scheduled_sampling --ss_start_epoch 1 --ss_max_ratio 0.3"

# ==================== 启动训练 ====================
torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/train_mag.py \
    --model_path ${MODEL_PATH} \
    --data_source jsonl \
    --data_path ${DATA_PATH} \
    --deep_encode_layers ${DEEP_ENCODE_LAYERS} \
    --max_real_samples ${MAX_REAL_SAMPLES} \
    --sliding_window ${SLIDING_WINDOW} \
    --output_dir ${OUTPUT_DIR} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --label_smoothing ${LABEL_SMOOTHING} \
    --kl_beta ${KL_BETA} \
    --kl_temperature ${KL_TEMPERATURE} \
    ${DETACH_VALUE} \
    ${SCHEDULED_SAMPLING}

echo "============================================================"
echo "训练完成! 输出目录: ${OUTPUT_DIR}"
echo "============================================================"
