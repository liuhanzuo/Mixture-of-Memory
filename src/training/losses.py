"""
损失函数模块。

包含:
  1. 主任务损失: 基于 fused hidden states 的 next-token prediction CE loss
  2. Utility 辅助损失: 训练 evaluator scores 与 drop-write utility 目标相关
  3. MemoryAugmentedLoss: 组合上述损失的统一接口
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class NextTokenLoss(nn.Module):
    """标准的 next-token prediction 交叉熵损失。

    从 fused hidden states 通过 lm_head 得到 logits，
    然后与 shifted labels 计算 CE loss。
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """计算 next-token CE loss。

        Args:
            logits: [B, T, V] 模型预测的 logits。
            labels: [B, T] 目标 token ids，padding 处为 -100。

        Returns:
            loss: 标量 CE loss。
        """
        # Shift: logits[:-1] 预测 labels[1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
        )
        return loss

    def forward_per_token(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """计算逐 token 的 CE loss（不 reduce），用于 utility 计算。

        Args:
            logits: [B, T, V]
            labels: [B, T]

        Returns:
            per_token_loss: [B, T-1]
        """
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        B, T_minus_1, V = shift_logits.shape
        per_token = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction="none",
        )
        return per_token.view(B, T_minus_1)


class EvaluatorUtilityLoss(nn.Module):
    """Utility-based 辅助损失，训练 evaluator scores 与 utility targets 相关。

    目标: evaluator 对每个 token 的打分应与该 token 作为 anchor 时的
    写入 utility 正相关。

    实现方式:
    - 如果有精确的 per-anchor utility target，使用 MSE 回归
    - 也支持 ranking loss（margin-based）
    """

    def __init__(
        self,
        loss_type: str = "mse",
        margin: float = 0.1,
    ) -> None:
        """
        Args:
            loss_type: "mse" (回归) 或 "ranking" (margin-based ranking)。
            margin: ranking loss 的 margin。
        """
        super().__init__()
        self.loss_type = loss_type
        self.margin = margin

    def forward(
        self,
        evaluator_scores: torch.Tensor,
        anchor_indices: torch.Tensor,
        utility_targets: torch.Tensor,
    ) -> torch.Tensor:
        """计算 evaluator 辅助损失。

        Args:
            evaluator_scores: [B, L] block 内每个 token 的评分。
            anchor_indices: [B, K] 被选中的 anchor 位置。
            utility_targets: [B, K] 每个 anchor 的 utility target。

        Returns:
            loss: 标量损失。
        """
        # 提取被选中 anchor 位置的 evaluator scores
        # anchor_indices: [B, K]
        selected_scores = torch.gather(
            evaluator_scores, dim=1, index=anchor_indices
        )  # [B, K]

        if self.loss_type == "mse":
            # 归一化 utility targets 到 [0, 1]
            # 对 evaluator scores 做 sigmoid 映射到 [0, 1]
            pred = torch.sigmoid(selected_scores)
            target = utility_targets.detach()
            # 将 target 也归一化到 [0, 1]（如果还没有）
            t_min = target.min(dim=-1, keepdim=True).values
            t_max = target.max(dim=-1, keepdim=True).values
            denom = (t_max - t_min).clamp(min=1e-8)
            target_norm = (target - t_min) / denom
            loss = F.mse_loss(pred, target_norm)

        elif self.loss_type == "ranking":
            # Pairwise ranking loss: 高 utility 的 anchor 应该有更高的 score
            loss = self._pairwise_ranking_loss(selected_scores, utility_targets)

        else:
            raise ValueError(f"未知损失类型: {self.loss_type}")

        return loss

    def _pairwise_ranking_loss(
        self,
        scores: torch.Tensor,
        utilities: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise margin ranking loss。

        对所有 (i, j) 对，如果 utility[i] > utility[j]，
        则要求 score[i] > score[j] + margin。

        Args:
            scores: [B, K]
            utilities: [B, K]

        Returns:
            loss: 标量。
        """
        B, K = scores.shape
        if K < 2:
            return torch.tensor(0.0, device=scores.device)

        # 构造所有对
        # scores_i: [B, K, 1], scores_j: [B, 1, K]
        scores_i = scores.unsqueeze(2)
        scores_j = scores.unsqueeze(1)
        util_i = utilities.unsqueeze(2)
        util_j = utilities.unsqueeze(1)

        # 只考虑 utility[i] > utility[j] 的对
        mask = (util_i > util_j + 1e-6).float()  # [B, K, K]

        # hinge loss: max(0, margin - (score_i - score_j))
        diff = scores_i - scores_j  # [B, K, K]
        loss_pairs = F.relu(self.margin - diff)  # [B, K, K]
        loss_pairs = loss_pairs * mask

        # 平均有效对的损失
        num_valid = mask.sum().clamp(min=1.0)
        loss = loss_pairs.sum() / num_valid

        return loss


class MemoryAugmentedLoss(nn.Module):
    """Memory-augmented 系统的组合损失函数。

    组合:
      1. 主 LM 损失 (next-token prediction)
      2. Evaluator utility 辅助损失
      3. 可选的正则化项

    Args:
        lm_weight: LM 损失权重。
        utility_weight: utility 辅助损失权重。
        utility_loss_type: utility 损失类型 ("mse" / "ranking")。
        ignore_index: LM loss 中忽略的 label id。
    """

    def __init__(
        self,
        lm_weight: float = 1.0,
        utility_weight: float = 0.1,
        utility_loss_type: str = "mse",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.lm_weight = lm_weight
        self.utility_weight = utility_weight

        self.lm_loss = NextTokenLoss(ignore_index=ignore_index)
        self.utility_loss = EvaluatorUtilityLoss(loss_type=utility_loss_type)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        evaluator_scores: Optional[torch.Tensor] = None,
        anchor_indices: Optional[torch.Tensor] = None,
        utility_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算组合损失。

        Args:
            logits: [B, T, V] fused logits。
            labels: [B, T] 目标 labels。
            evaluator_scores: [B, L] 可选的 evaluator 分数。
            anchor_indices: [B, K] 可选的 anchor 索引。
            utility_targets: [B, K] 可选的 utility targets。

        Returns:
            total_loss: 加权后的总损失（标量）。
            loss_dict: 各项损失分量的字典。
        """
        loss_dict: Dict[str, float] = {}

        # --- 1. 主 LM 损失 ---
        lm = self.lm_loss(logits, labels)
        loss_dict["lm_loss"] = lm.item()
        total_loss = self.lm_weight * lm

        # --- 2. Evaluator utility 辅助损失 ---
        if (
            self.utility_weight > 0
            and evaluator_scores is not None
            and anchor_indices is not None
            and utility_targets is not None
        ):
            util_loss = self.utility_loss(
                evaluator_scores, anchor_indices, utility_targets
            )
            loss_dict["utility_loss"] = util_loss.item()
            total_loss = total_loss + self.utility_weight * util_loss
        else:
            loss_dict["utility_loss"] = 0.0

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict

    def compute_per_token_lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """计算逐 token LM loss（用于 utility target 生成）。

        Args:
            logits: [B, T, V]
            labels: [B, T]

        Returns:
            per_token_loss: [B, T-1]
        """
        return self.lm_loss.forward_per_token(logits, labels)
