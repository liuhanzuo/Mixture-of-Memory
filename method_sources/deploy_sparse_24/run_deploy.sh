#!/usr/bin/env bash
set -euo pipefail
#
# run_deploy.sh — 2:4 稀疏模型部署一键脚本
#
# 完整流程：
#   1. 从训练 checkpoint 中导出 2:4 稀疏权重 → HuggingFace 模型
#   2. 运行 Dense vs Sparse 性能对比 benchmark
#   3. 文本生成演示
#
# 用法：
#   bash deploy_sparse_24/run_deploy.sh [CHECKPOINT] [BASE_MODEL] [OUTPUT_DIR]
#
# 示例：
#   # 使用默认路径
#   bash deploy_sparse_24/run_deploy.sh
#
#   # 指定 checkpoint
#   bash deploy_sparse_24/run_deploy.sh /path/to/model.pt models/Qwen--Qwen3-1.7B
#
# 环境要求：
#   - PyTorch >= 2.1.0 (with CUDA)
#   - GPU: SM80+ (A100, H100, H800, ...)
#   - transformers, safetensors, datasets
#

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# 默认参数（可通过命令行覆盖）
CHECKPOINT=${1:-""}
BASE_MODEL=${2:-"models/Qwen--Qwen3-1.7B"}
OUTPUT_DIR=${3:-"${SCRIPT_DIR}/exported_model"}

# 自动查找最新的 checkpoint（如果未指定）
if [ -z "$CHECKPOINT" ]; then
    echo "[INFO] 未指定 checkpoint，正在自动查找..."
    # 按照优先级查找
    SEARCH_DIRS=(
        "${PROJECT_DIR}/out_llama"
        "${PROJECT_DIR}/outputs"
    )
    for dir in "${SEARCH_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            # 查找最新的 model.pt
            found=$(find "$dir" -name "model.pt" -type f 2>/dev/null | head -1)
            if [ -n "$found" ]; then
                CHECKPOINT="$found"
                echo "[INFO] 找到 checkpoint: $CHECKPOINT"
                break
            fi
            # 也查找 retrain_best.pt
            found=$(find "$dir" -name "retrain_best.pt" -type f 2>/dev/null | head -1)
            if [ -n "$found" ]; then
                CHECKPOINT="$found"
                echo "[INFO] 找到 checkpoint: $CHECKPOINT"
                break
            fi
        fi
    done
    
    if [ -z "$CHECKPOINT" ]; then
        echo "❌ 未找到训练 checkpoint！"
        echo ""
        echo "用法: bash deploy_sparse_24/run_deploy.sh <CHECKPOINT> [BASE_MODEL] [OUTPUT_DIR]"
        echo ""
        echo "示例:"
        echo "  bash deploy_sparse_24/run_deploy.sh outputs/qwen-AST-nm_2_4/final_iter17000/model.pt"
        exit 1
    fi
fi

# 确认 base model 路径是绝对路径
if [[ ! "$BASE_MODEL" = /* ]]; then
    BASE_MODEL="${PROJECT_DIR}/${BASE_MODEL}"
fi

echo "============================================================"
echo "2:4 Sparse Model Deployment Pipeline"
echo "============================================================"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Base Model:  $BASE_MODEL"
echo "  Output Dir:  $OUTPUT_DIR"
echo "  PyTorch:     $(python3 -c 'import torch; print(torch.__version__)')"
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  GPU:         $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    echo "  GPU:         N/A (will fail on benchmark)"
fi
echo "============================================================"

# Step 1: 导出模型
echo ""
echo "=============================="
echo "Step 1: 导出 2:4 稀疏模型"
echo "=============================="
python3 "${SCRIPT_DIR}/convert.py" \
    --checkpoint "$CHECKPOINT" \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --verify \
    --save_report

# 检查是否有 GPU
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo ""
    echo "⚠️ 没有可用的 GPU，跳过 benchmark 和推理步骤"
    echo ""
    echo "在 GPU 节点上运行以下命令："
    echo ""
    echo "  # 性能对比"
    echo "  python3 deploy_sparse_24/benchmark.py --model_dir $OUTPUT_DIR"
    echo ""
    echo "  # PPL 评估"
    echo "  python3 deploy_sparse_24/eval_ppl.py --model_dir $OUTPUT_DIR --base_model $BASE_MODEL"
    echo ""
    echo "  # 文本生成"
    echo "  python3 deploy_sparse_24/inference.py --model_dir $OUTPUT_DIR --prompt 'Hello world'"
    exit 0
fi

# Step 2: 性能对比 Benchmark
echo ""
echo "=============================="
echo "Step 2: Dense vs Sparse Benchmark"
echo "=============================="
python3 "${SCRIPT_DIR}/benchmark.py" \
    --model_dir "$OUTPUT_DIR" \
    --batch_sizes "1,4,8,16,32,64" \
    --seq_lens "128,256,512,1024,2048" \
    --num_runs 20 \
    --save_results "${OUTPUT_DIR}/benchmark_results.json"

# Step 3: 文本生成演示
echo ""
echo "=============================="
echo "Step 3: 文本生成演示"
echo "=============================="
echo "[Dense 推理]"
python3 "${SCRIPT_DIR}/inference.py" \
    --model_dir "$OUTPUT_DIR" \
    --prompt "The future of artificial intelligence is" \
    --max_new_tokens 64 \
    --no_sparse_accel

echo ""
echo "[2:4 Sparse 加速推理]"
python3 "${SCRIPT_DIR}/inference.py" \
    --model_dir "$OUTPUT_DIR" \
    --prompt "The future of artificial intelligence is" \
    --max_new_tokens 64

echo ""
echo "============================================================"
echo "✅ 部署完成！"
echo ""
echo "模型已导出到: $OUTPUT_DIR"
echo "Benchmark 结果: ${OUTPUT_DIR}/benchmark_results.json"
echo ""
echo "可用命令："
echo "  # 推理生成"
echo "  python3 deploy_sparse_24/inference.py --model_dir $OUTPUT_DIR --prompt 'your prompt' --benchmark"
echo ""
echo "  # 性能对比"  
echo "  python3 deploy_sparse_24/benchmark.py --model_dir $OUTPUT_DIR"
echo ""
echo "  # PPL 评估"
echo "  python3 deploy_sparse_24/eval_ppl.py --model_dir $OUTPUT_DIR --base_model $BASE_MODEL"
echo "============================================================"
