#!/usr/bin/env bash
# ============================================================
# Stage 1 训练脚本
# 冻结 backbone，训练 evaluator / gather / write / readout / fusion
# ============================================================
set -euo pipefail

# --- 默认参数 ---
CONFIG="configs/synthetic.yaml"
MODEL_CONFIG="configs/model/tiny.yaml"
ABLATION=""
EXTRA_ARGS=""
DEVICE="cuda"
SEED=42

# --- 解析命令行参数 ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"; shift 2 ;;
        --model)
            MODEL_CONFIG="$2"; shift 2 ;;
        --ablation)
            ABLATION="$2"; shift 2 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

echo "============================================================"
echo " Mixture-of-Memory v0 — Stage 1 Training"
echo "============================================================"
echo " Config:   ${CONFIG}"
echo " Model:    ${MODEL_CONFIG}"
echo " Ablation: ${ABLATION:-none}"
echo " Device:   ${DEVICE}"
echo " Seed:     ${SEED}"
echo "============================================================"

# --- 构建配置覆盖 ---
OVERRIDES="training.seed=${SEED}"

if [ -n "${ABLATION}" ]; then
    ABLATION_CONFIG="configs/ablations/${ABLATION}.yaml"
    if [ ! -f "${ABLATION_CONFIG}" ]; then
        echo "错误: 找不到 ablation 配置 ${ABLATION_CONFIG}"
        exit 1
    fi
    echo " 使用 ablation 配置: ${ABLATION_CONFIG}"
fi

# --- 运行训练 ---
CMD="python -m src.training.trainer"
CMD="${CMD} --config ${CONFIG}"
CMD="${CMD} --model-config ${MODEL_CONFIG}"

if [ -n "${ABLATION}" ]; then
    CMD="${CMD} --ablation-config ${ABLATION_CONFIG}"
fi

CMD="${CMD} --device ${DEVICE}"
CMD="${CMD} --overrides ${OVERRIDES}"
CMD="${CMD} ${EXTRA_ARGS}"

echo ""
echo "执行命令: ${CMD}"
echo ""

# 如果作为模块无法直接运行，则退回到脚本方式
python -c "
import sys
sys.path.insert(0, '.')

from omegaconf import OmegaConf
from src.common.config import load_config
from src.common.logging import setup_logging
from src.common.seed import set_seed
from src.backbone.lm_wrapper import FrozenLMWrapper
from src.data.synthetic_dataset import build_synthetic_from_config
from src.training.trainer import MemoryTrainer
from torch.utils.data import DataLoader
import torch

# 加载配置
configs_to_merge = ['${CONFIG}', '${MODEL_CONFIG}']
ablation = '${ABLATION}'
if ablation:
    configs_to_merge.append('configs/ablations/' + ablation + '.yaml')

cfg = load_config(*configs_to_merge)
cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(['training.seed=${SEED}']))

# 初始化
setup_logging(level='INFO')
set_seed(cfg.training.seed)
device = torch.device('${DEVICE}' if torch.cuda.is_available() or '${DEVICE}' == 'cpu' else 'cpu')

# 构建 backbone
backbone = FrozenLMWrapper(
    model_name=cfg.backbone.model_name,
    freeze=cfg.backbone.freeze,
)
backbone = backbone.to(device)

# 构建数据集
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(cfg.backbone.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_ds, val_ds, test_ds = build_synthetic_from_config(cfg, tokenizer=tokenizer)

train_loader = DataLoader(
    train_ds,
    batch_size=cfg.training.batch_size,
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    val_ds,
    batch_size=cfg.training.batch_size,
    shuffle=False,
    num_workers=0,
)

# 构建 Trainer 并训练
trainer = MemoryTrainer(cfg=cfg, backbone=backbone, device=device)
history = trainer.train(train_loader=train_loader, val_loader=val_loader)

print('训练完成！')
print(f'最终 train_loss: {history[\"train_loss\"][-1]:.4f}')
if history[\"val_loss\"]:
    print(f'最佳 val_loss: {min(history[\"val_loss\"]):.4f}')
"

echo ""
echo "Stage 1 训练完成!"
