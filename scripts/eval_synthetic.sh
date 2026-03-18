#!/usr/bin/env bash
# ============================================================
# 合成数据集评估脚本
# ============================================================
set -euo pipefail

CHECKPOINT="checkpoints/checkpoint_best.pt"
CONFIG="configs/synthetic.yaml"
MODEL_CONFIG="configs/model/tiny.yaml"
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --model) MODEL_CONFIG="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo " Mixture-of-Memory v0 — Synthetic Evaluation"
echo "============================================================"
echo " Checkpoint: ${CHECKPOINT}"
echo " Config:     ${CONFIG}"
echo " Device:     ${DEVICE}"
echo "============================================================"

python -c "
import sys
sys.path.insert(0, '.')

from omegaconf import OmegaConf
from src.common.config import load_config
from src.common.logging import setup_logging
from src.common.seed import set_seed
from src.eval.evaluate_synthetic import SyntheticEvaluator
import torch

cfg = load_config('${CONFIG}', '${MODEL_CONFIG}')
setup_logging(level='INFO')
set_seed(cfg.training.seed)
device = torch.device('${DEVICE}' if torch.cuda.is_available() or '${DEVICE}' == 'cpu' else 'cpu')

evaluator = SyntheticEvaluator(cfg=cfg, device=device)
evaluator.load_checkpoint('${CHECKPOINT}')
results = evaluator.evaluate()
evaluator.print_results(results)
"

echo ""
echo "评估完成!"
