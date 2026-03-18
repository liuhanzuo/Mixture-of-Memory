#!/usr/bin/env bash
# ============================================================
# LongMemEval 基准评估脚本 (scaffold — 待完善)
# ============================================================
set -euo pipefail

CHECKPOINT="checkpoints/checkpoint_best.pt"
CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/model/tiny.yaml"
LONGMEMEVAL_DATA_DIR="./data/longmemeval"
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --model) MODEL_CONFIG="$2"; shift 2 ;;
        --data-dir) LONGMEMEVAL_DATA_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo " Mixture-of-Memory v0 — LongMemEval Evaluation (Scaffold)"
echo "============================================================"
echo " 注意: LongMemEval 评估在 v0 中为 scaffold 实现"
echo " 需要先准备 LongMemEval 数据集到: ${LONGMEMEVAL_DATA_DIR}"
echo "============================================================"

python -c "
import sys
sys.path.insert(0, '.')

from src.common.logging import setup_logging
setup_logging(level='INFO')

from src.eval.evaluate_longmemeval import LongMemEvalEvaluator

print('LongMemEval evaluator scaffold 已就绪。')
print('请实现 LongMemEvalEvaluator.evaluate() 后重新运行。')
print('')
print('使用方式:')
print('  evaluator = LongMemEvalEvaluator(cfg, device)')
print('  evaluator.load_checkpoint(checkpoint_path)')
print('  results = evaluator.evaluate(data_dir)')
"

echo ""
echo "LongMemEval scaffold 完成。"
