"""
训练阶段管理器。

管理不同训练阶段下各模块的冻结/解冻策略。

Stage 1 (v0):
  - 冻结: backbone（始终）
  - 训练: evaluator, selector, gather, write heads, readout, fusion, retention

Stage 2 (未来):
  - 可选: 用 LoRA 或 partial unfreezing 微调 backbone

Stage 3 (未来):
  - 端到端联合训练
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


class TrainingStageManager:
    """训练阶段管理器。

    根据当前阶段配置各模块的训练/冻结状态。

    Args:
        stage: 当前训练阶段（1, 2, 3）。
    """

    def __init__(self, stage: int = 1) -> None:
        self.stage = stage

    def configure_stage(
        self,
        backbone: nn.Module,
        trainable_modules: Dict[str, nn.Module],
    ) -> List[nn.Parameter]:
        """根据当前阶段配置模块冻结状态，返回可训练参数。

        Args:
            backbone: 冻结的 backbone 模型。
            trainable_modules: {名称: 模块} 字典，包含所有外部模块。

        Returns:
            trainable_params: 当前阶段需要训练的参数列表。
        """
        if self.stage == 1:
            return self._configure_stage1(backbone, trainable_modules)
        elif self.stage == 2:
            return self._configure_stage2(backbone, trainable_modules)
        else:
            raise ValueError(f"未支持的训练阶段: {self.stage}")

    def _configure_stage1(
        self,
        backbone: nn.Module,
        trainable_modules: Dict[str, nn.Module],
    ) -> List[nn.Parameter]:
        """Stage 1: 冻结 backbone，训练所有外部模块。

        训练的模块包括:
          - block evaluator
          - anchor selector (如果有可学习参数)
          - retrospective gather
          - write / route / retain heads
          - memory readout projections
          - fusion head
          - retention scheduler (如果是 learned 模式)
        """
        logger.info("=" * 60)
        logger.info("配置 Stage 1: 冻结 backbone，训练外部模块")
        logger.info("=" * 60)

        # 确保 backbone 冻结
        frozen_count = 0
        for param in backbone.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
        if frozen_count > 0:
            logger.info(f"  额外冻结了 {frozen_count} 个 backbone 参数")

        # 解冻所有外部模块
        trainable_params = []
        for name, module in trainable_modules.items():
            module_params = 0
            for param in module.parameters():
                param.requires_grad = True
                trainable_params.append(param)
                module_params += param.numel()
            logger.info(f"  训练模块 [{name}]: {module_params:,} 参数")

        logger.info(f"  总可训练参数: {sum(p.numel() for p in trainable_params):,}")
        return trainable_params

    def _configure_stage2(
        self,
        backbone: nn.Module,
        trainable_modules: Dict[str, nn.Module],
    ) -> List[nn.Parameter]:
        """Stage 2 (预留): 支持 LoRA / partial unfreezing。

        当前仅是占位实现，与 Stage 1 相同。
        后续可以在此处添加:
          - LoRA adapter
          - 仅解冻 backbone 的最后 N 层
          - 仅解冻 lm_head
        """
        logger.info("=" * 60)
        logger.info("配置 Stage 2: 预留（当前与 Stage 1 相同）")
        logger.info("=" * 60)

        # 目前与 Stage 1 行为相同
        return self._configure_stage1(backbone, trainable_modules)

    @staticmethod
    def get_stage_description(stage: int) -> str:
        """获取阶段描述。"""
        descriptions = {
            1: "Stage 1: 冻结 backbone，训练 evaluator / gather / write / readout / fusion",
            2: "Stage 2 (预留): 可选 LoRA / partial unfreezing",
            3: "Stage 3 (预留): 端到端联合训练",
        }
        return descriptions.get(stage, f"未知阶段: {stage}")
