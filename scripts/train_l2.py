#!/usr/bin/env python3
"""
train_l2.py — 便捷的 L2 聚合器 LoRA 训练入口脚本。

Usage::

    # 默认配置 (合成数据，3 epoch)
    python scripts/train_l2.py

    # 自定义参数
    python scripts/train_l2.py --epochs 5 --lr 3e-5 --lora-rank 32 --batch-size 8

    # 使用真实数据
    python scripts/train_l2.py --train-data data/processed/l2_train.jsonl
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.training.train_l2_aggregator import L2AggregatorTrainer, L2AggregatorTrainConfig
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="L2 Aggregator LoRA Training")
    parser.add_argument("--base-model", type=str, default=None, help="基础模型路径")
    parser.add_argument("--train-data", type=str, default=None, help="训练数据路径 (JSONL)")
    parser.add_argument("--val-data", type=str, default=None, help="验证数据路径 (JSONL)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--save-dir", type=str, default="outputs/runs/l2_aggregator_training")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    config = L2AggregatorTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        grad_accumulation_steps=args.grad_accum,
        max_seq_len=args.max_seq_len,
        save_dir=args.save_dir,
    )

    if args.base_model:
        config.base_model = args.base_model
    if args.train_data:
        config.train_data_path = args.train_data
    if args.val_data:
        config.val_data_path = args.val_data

    logger.info(f"L2 训练配置: epochs={config.epochs}, lr={config.lr}, "
                f"lora_rank={config.lora_rank}, batch_size={config.batch_size}")

    trainer = L2AggregatorTrainer(config)
    results = trainer.train()

    print(f"\n✅ L2 聚合器训练完成!")
    print(f"   状态: {results['status']}")
    print(f"   总步数: {results['total_steps']}")
    print(f"   最终 loss: {results['final_train_loss']:.4f}")
    print(f"   保存路径: {results['save_dir']}")


if __name__ == "__main__":
    main()
