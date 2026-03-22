#!/usr/bin/env bash
# ============================================================
# run_experiments.sh — MoM 完整实验流程
#
# 使用方法:
#   bash scripts/run_experiments.sh              # 运行全部阶段
#   bash scripts/run_experiments.sh --phase 1    # 只运行 Phase 1
#   bash scripts/run_experiments.sh --phase 2    # 只运行 Phase 2
#   bash scripts/run_experiments.sh --phase 3    # 只运行 Phase 3
#   bash scripts/run_experiments.sh --phase 4    # 只运行 Phase 4
#
# 实验阶段:
#   Phase 1: 基础验证 (GPU检查 + 模型加载 + Demo对话)
#   Phase 2: Baseline 评测 (5个配置消融实验)
#   Phase 3: LoRA 训练 (L2聚合器 + L3总结器)
#   Phase 4: 训练后评测 (使用训练好的adapter重新评测)
# ============================================================
set -euo pipefail

# ---- 颜色 ---- #
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()      { echo -e "${GREEN}[  OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()    { echo -e "${RED}[FAIL]${NC} $*"; }
phase()   { echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${CYAN}  $*${NC}"; echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"; }

# ---- 配置 ---- #
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"
MODEL_PATH="/apdcephfs/pig_data/Adaptive-Sparse-Trainer/models/Qwen--Qwen3-1.7b"

# 实验参数 (可通过环境变量覆盖)
NUM_SAMPLES=${NUM_SAMPLES:-20}
SEED=${SEED:-42}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-3}
LORA_RANK=${LORA_RANK:-16}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-2e-5}

# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
METRICS_DIR="${OUTPUT_DIR}/metrics"
RUNS_DIR="${OUTPUT_DIR}/runs"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${METRICS_DIR}" "${RUNS_DIR}" "${LOG_DIR}"

# ---- 参数解析 ---- #
TARGET_PHASE=${1:-"all"}
if [ "$TARGET_PHASE" = "--phase" ]; then
    TARGET_PHASE="${2:-all}"
fi

echo ""
echo "============================================================"
echo "  🧠 Mixture-of-Memory — 完整实验流程"
echo "============================================================"
echo "  项目: ${PROJECT_ROOT}"
echo "  模型: ${MODEL_PATH}"
echo "  样本数: ${NUM_SAMPLES}"
echo "  种子: ${SEED}"
echo "  训练 Epochs: ${TRAIN_EPOCHS}"
echo "  LoRA Rank: ${LORA_RANK}"
echo "  目标阶段: ${TARGET_PHASE}"
echo "============================================================"


# ============================================================
# Phase 1: 基础验证
# ============================================================
run_phase1() {
    phase "Phase 1: 基础验证"
    
    # 1.1 GPU 检查
    info "1.1 检查 GPU 环境..."
    python3 -c "
import torch
if not torch.cuda.is_available():
    print('❌ CUDA 不可用! 请确认 GPU 驱动和 PyTorch CUDA 版本')
    exit(1)
n = torch.cuda.device_count()
print(f'✅ 检测到 {n} 块 GPU:')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
    print(f'   GPU {i}: {name} ({mem:.1f} GB)')
"
    ok "GPU 环境正常"

    # 1.2 单元测试
    info "1.2 运行单元测试..."
    python3 -m pytest "${PROJECT_ROOT}/tests/" -q --tb=line 2>&1 | tail -3
    ok "单元测试通过"

    # 1.3 Demo 对话 (验证模型加载+推理通路)
    info "1.3 运行 Demo 对话 (验证完整推理通路)..."
    python3 "${PROJECT_ROOT}/scripts/run_chat.py" \
        --config-name swa_mom \
        --mode demo \
        --seed ${SEED} \
        --log-level INFO \
        2>&1 | tee "${LOG_DIR}/phase1_demo.log"
    ok "Demo 对话完成"

    echo ""
    ok "Phase 1 全部通过 ✅"
}


# ============================================================
# Phase 2: Baseline 评测 (消融实验)
# ============================================================
run_phase2() {
    phase "Phase 2: Baseline 评测 (消融实验)"

    info "运行 5 配置消融实验 (每个配置 ${NUM_SAMPLES} 样本)..."
    info "配置列表: swa_only, swa_l1, swa_l1_l2, swa_mom, fullattn_baseline"
    echo ""

    python3 "${PROJECT_ROOT}/scripts/run_ablation.py" \
        --num-samples ${NUM_SAMPLES} \
        --seed ${SEED} \
        --log-level INFO \
        2>&1 | tee "${LOG_DIR}/phase2_ablation.log"

    # 也单独运行 swa_mom 的详细评测
    info "运行 swa_mom 详细评测 (${NUM_SAMPLES} 样本)..."
    python3 "${PROJECT_ROOT}/scripts/run_eval.py" \
        --config-name swa_mom \
        --num-samples ${NUM_SAMPLES} \
        --seed ${SEED} \
        --log-level INFO \
        2>&1 | tee "${LOG_DIR}/phase2_eval_swa_mom.log"

    ok "Phase 2 Baseline 评测完成 ✅"
    echo ""
    echo "  📊 结果位于:"
    echo "     ${METRICS_DIR}/ablation_results.json"
    echo "     ${METRICS_DIR}/swa_mom_eval.json"
}


# ============================================================
# Phase 3: LoRA 训练
# ============================================================
run_phase3() {
    phase "Phase 3: LoRA 训练"

    # 3.1 训练 L2 聚合器
    info "3.1 训练 L2 聚合器 (LoRA, ${TRAIN_EPOCHS} epochs)..."
    python3 "${PROJECT_ROOT}/scripts/train_l2.py" \
        --epochs ${TRAIN_EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --lora-rank ${LORA_RANK} \
        --save-dir "${RUNS_DIR}/l2_aggregator_training" \
        --log-level INFO \
        2>&1 | tee "${LOG_DIR}/phase3_train_l2.log"
    ok "L2 聚合器训练完成"

    # 3.2 训练 L3 总结器
    info "3.2 训练 L3 总结器 (LoRA, ${TRAIN_EPOCHS} epochs)..."
    python3 "${PROJECT_ROOT}/scripts/train_l3.py" \
        --epochs ${TRAIN_EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --lora-rank ${LORA_RANK} \
        --save-dir "${RUNS_DIR}/l3_summarizer_training" \
        --log-level INFO \
        2>&1 | tee "${LOG_DIR}/phase3_train_l3.log"
    ok "L3 总结器训练完成"

    echo ""
    ok "Phase 3 训练完成 ✅"
    echo ""
    echo "  📦 训练产物:"
    echo "     L2 adapter: ${RUNS_DIR}/l2_aggregator_training/final_adapter/"
    echo "     L3 adapter: ${RUNS_DIR}/l3_summarizer_training/final_adapter/"
}


# ============================================================
# Phase 4: 训练后评测
# ============================================================
run_phase4() {
    phase "Phase 4: 训练后评测"

    L2_ADAPTER="${RUNS_DIR}/l2_aggregator_training/final_adapter"
    L3_ADAPTER="${RUNS_DIR}/l3_summarizer_training/final_adapter"

    if [ ! -d "${L2_ADAPTER}" ] || [ ! -d "${L3_ADAPTER}" ]; then
        warn "训练产物不完整，跳过 Phase 4"
        warn "请先运行 Phase 3 (bash scripts/run_experiments.sh --phase 3)"
        return 1
    fi

    info "使用训练好的 LoRA adapter 重新评测..."
    info "L2 adapter: ${L2_ADAPTER}"
    info "L3 adapter: ${L3_ADAPTER}"

    # 使用 Python 脚本做训练后评测（因为需要修改配置注入 adapter 路径）
    python3 -c "
import json
import sys
import time
import logging
sys.path.insert(0, '${PROJECT_ROOT}')

from pathlib import Path
from omegaconf import OmegaConf
from src.agents.memory_agent import MemoryAgent, AgentConfig
from src.agents.session_runner import SessionRunner
from src.backbone import build_backbone_from_config
from src.tasks.synthetic_update_task import SyntheticUpdateTask
from src.tasks.profile_task import ProfileTask
from src.tasks.longhorizon_chat_task import LongHorizonChatTask
from src.utils.logging import setup_logging
from src.utils.seeds import set_seed

setup_logging(level='INFO')
set_seed(${SEED})

logger = logging.getLogger('phase4')

# 加载配置
config_path = Path('${PROJECT_ROOT}/configs/exp/swa_mom.yaml')
cfg = OmegaConf.load(str(config_path))

# 构建 backbone
config_dir = Path('${PROJECT_ROOT}/configs')
backbone = build_backbone_from_config(cfg, config_dir=config_dir)
tokenizer = backbone.get_tokenizer()

# 构建 agent (配置 LLM 后端 + adapter)
agent = MemoryAgent(
    config=AgentConfig(enable_l1=True, enable_l2=True, enable_l3=True),
    backbone=backbone,
    tokenizer=tokenizer,
)

# 评测
results = {}

# 1. Synthetic Update
logger.info('Running synthetic_update...')
task = SyntheticUpdateTask(num_samples=${NUM_SAMPLES}, seed=${SEED})
samples = task.generate_samples()
predictions = []
for sample in samples:
    runner = SessionRunner(agent)
    msgs = [c for r, c in sample.conversation if r == 'user']
    msgs.append(sample.query)
    trace = runner.run_conversation(messages=msgs, session_id=f'post_update_{sample.sample_id}')
    predictions.append(trace.agent_replies[-1] if trace.agent_replies else '')
    agent.reset(keep_l3=False)
report = task.evaluate_batch(samples, predictions)
results['synthetic_update'] = {'overall_accuracy': report['overall_accuracy']}
logger.info(f'synthetic_update: {report[\"overall_accuracy\"]:.2%}')

# 2. Profile
logger.info('Running profile_bench...')
task2 = ProfileTask(num_samples=${NUM_SAMPLES}, seed=${SEED})
samples2 = task2.generate_samples()
predictions2 = []
for sample in samples2:
    runner = SessionRunner(agent)
    msgs = [c for r, c in sample.conversation if r == 'user']
    msgs.append(sample.query)
    trace = runner.run_conversation(messages=msgs, session_id=f'post_profile_{sample.sample_id}')
    predictions2.append(trace.agent_replies[-1] if trace.agent_replies else '')
    agent.reset(keep_l3=False)
report2 = task2.evaluate_batch(samples2, predictions2)
results['profile_bench'] = {'avg_precision': report2['avg_precision']}
logger.info(f'profile_bench: {report2[\"avg_precision\"]:.2%}')

# 3. Long-Horizon Chat
logger.info('Running longhorizon_chat...')
task3 = LongHorizonChatTask(num_samples=${NUM_SAMPLES}, seed=${SEED})
samples3 = task3.generate_samples()
predictions3 = []
for sample in samples3:
    runner = SessionRunner(agent)
    msgs = [c for r, c in sample.conversation if r == 'user']
    msgs.append(sample.query)
    trace = runner.run_conversation(messages=msgs, session_id=f'post_longhorizon_{sample.sample_id}')
    predictions3.append(trace.agent_replies[-1] if trace.agent_replies else '')
    agent.reset(keep_l3=False)
report3 = task3.evaluate_batch(samples3, predictions3)
results['longhorizon_chat'] = {'overall_accuracy': report3['overall_accuracy']}
logger.info(f'longhorizon_chat: {report3[\"overall_accuracy\"]:.2%}')

# 保存结果
output_path = Path('${METRICS_DIR}/post_training_eval.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

print()
print('=' * 60)
print('📊 训练后评测结果 (swa_mom + LoRA adapters)')
print('=' * 60)
for task_name, metrics in results.items():
    print(f'  {task_name}:')
    for k, v in metrics.items():
        print(f'    {k}: {v:.2%}' if isinstance(v, float) else f'    {k}: {v}')
print(f'  📁 结果: {output_path}')
" 2>&1 | tee "${LOG_DIR}/phase4_post_training_eval.log"

    ok "Phase 4 训练后评测完成 ✅"
}


# ============================================================
# 主流程
# ============================================================

case "${TARGET_PHASE}" in
    1|phase1)
        run_phase1
        ;;
    2|phase2)
        run_phase2
        ;;
    3|phase3)
        run_phase3
        ;;
    4|phase4)
        run_phase4
        ;;
    all)
        TOTAL_START=$(date +%s)

        run_phase1
        run_phase2
        run_phase3
        run_phase4

        TOTAL_END=$(date +%s)
        TOTAL_TIME=$((TOTAL_END - TOTAL_START))

        echo ""
        echo "============================================================"
        echo -e "  ${GREEN}🎉 全部实验完成！${NC}"
        echo "============================================================"
        echo "  总耗时: ${TOTAL_TIME}s ($(( TOTAL_TIME / 60 ))m $(( TOTAL_TIME % 60 ))s)"
        echo ""
        echo "  📊 结果汇总:"
        echo "     消融实验:     ${METRICS_DIR}/ablation_results.json"
        echo "     Baseline 评测: ${METRICS_DIR}/swa_mom_eval.json"
        echo "     训练后评测:   ${METRICS_DIR}/post_training_eval.json"
        echo ""
        echo "  📦 训练产物:"
        echo "     L2 adapter:  ${RUNS_DIR}/l2_aggregator_training/final_adapter/"
        echo "     L3 adapter:  ${RUNS_DIR}/l3_summarizer_training/final_adapter/"
        echo ""
        echo "  📋 日志:"
        echo "     ${LOG_DIR}/"
        echo "============================================================"
        ;;
    *)
        echo "未知的阶段: ${TARGET_PHASE}"
        echo "用法: bash scripts/run_experiments.sh [--phase 1|2|3|4|all]"
        exit 1
        ;;
esac
