"""
ContextSelector: Learned Scorer 选择有用的记忆 context。

核心思想 (Counterfactual Context Selection):
    utility(c_i) = L(without c_i) - L(with c_i)

训练时:
    - 离线收集 ΔLoss 数据作为监督信号
    - 训练一个轻量 scorer 网络拟合 utility

推理时:
    - Scorer 直接打分, 选出 top-k 最有用的记忆 context
    - 不需要多次前向推理

Scorer 架构:
    score = σ(MLP(q ⊙ m + q + m))
    其中 q = query 编码, m = memory 编码
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ContextSelectorConfig:
    """ContextSelector 配置。

    Attributes:
        input_dim: 输入特征维度 (= memory encoder output_dim).
        hidden_dim: Scorer MLP 隐藏层维度.
        num_mlp_layers: MLP 层数 (至少 2).
        top_k: 推理时选出的最有用记忆数量.
        temperature: 训练时 Gumbel-Softmax 的温度参数.
        dropout: MLP 内部 dropout.
        use_query_memory_interaction: 是否使用 q ⊙ m 交互特征.
    """
    input_dim: int = 2048
    hidden_dim: int = 256
    num_mlp_layers: int = 3
    top_k: int = 5
    temperature: float = 1.0
    dropout: float = 0.1
    use_query_memory_interaction: bool = True


class ContextSelector(nn.Module):
    """Learned Scorer: 预测每条记忆 context 对当前 query 的有用程度。

    训练目标:
        给定 query embedding q 和一组 memory embeddings {m_1, ..., m_K},
        预测每条记忆的 utility score (由 counterfactual ΔLoss 提供监督)。

    推理:
        对每条记忆打分, 返回 top-k 最有用的记忆及其分数。

    Usage::

        selector = ContextSelector(config)

        # 训练
        scores = selector(query_emb, memory_embs)  # (B, K)
        loss = selector.compute_loss(scores, target_utilities)

        # 推理
        selected_indices, selected_scores = selector.select(query_emb, memory_embs, top_k=5)
    """

    def __init__(self, config: ContextSelectorConfig | dict[str, Any] | None = None):
        super().__init__()
        if config is None:
            config = ContextSelectorConfig()
        elif isinstance(config, dict):
            config = ContextSelectorConfig(**{
                k: v for k, v in config.items()
                if k in ContextSelectorConfig.__dataclass_fields__
            })
        self.config = config

        # 计算 MLP 输入维度
        # 拼接: [q, m, q⊙m] → 3 * input_dim (如果使用交互)
        # 拼接: [q, m]       → 2 * input_dim (不使用交互)
        if config.use_query_memory_interaction:
            mlp_input_dim = config.input_dim * 3
        else:
            mlp_input_dim = config.input_dim * 2

        # 构建 MLP scorer
        layers: list[nn.Module] = []
        current_dim = mlp_input_dim
        for i in range(config.num_mlp_layers - 1):
            next_dim = config.hidden_dim
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.GELU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            current_dim = next_dim

        # 最后一层: hidden_dim → 1 (utility score)
        layers.append(nn.Linear(current_dim, 1))

        self.scorer = nn.Sequential(*layers)

        # 用于训练时生成 soft selection mask 的温度
        self._temperature = config.temperature

        logger.info(
            f"[ContextSelector] 初始化: input_dim={config.input_dim}, "
            f"hidden_dim={config.hidden_dim}, layers={config.num_mlp_layers}, "
            f"top_k={config.top_k}, interaction={config.use_query_memory_interaction}"
        )

    def forward(
        self,
        query_emb: torch.Tensor,
        memory_embs: torch.Tensor,
    ) -> torch.Tensor:
        """对每条记忆计算 utility score。

        Args:
            query_emb: 查询编码, (B, D) 或 (D,).
            memory_embs: 记忆编码, (B, K, D) 或 (K, D).

        Returns:
            scores: (B, K) 每条记忆的 utility score (未经 sigmoid).
        """
        # 统一维度
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)  # (1, D)
        if memory_embs.dim() == 2:
            memory_embs = memory_embs.unsqueeze(0)  # (1, K, D)

        B, K, D = memory_embs.shape

        # 扩展 query: (B, D) → (B, K, D)
        q_expanded = query_emb.unsqueeze(1).expand(B, K, D)

        # 拼接特征
        if self.config.use_query_memory_interaction:
            interaction = q_expanded * memory_embs  # (B, K, D)
            features = torch.cat([q_expanded, memory_embs, interaction], dim=-1)  # (B, K, 3D)
        else:
            features = torch.cat([q_expanded, memory_embs], dim=-1)  # (B, K, 2D)

        # 统一 dtype: embedding 可能是 bf16, 而 MLP 参数是 float32
        # 将输入转为 MLP 参数的 dtype
        scorer_dtype = next(self.scorer.parameters()).dtype
        features = features.to(dtype=scorer_dtype)

        # 通过 MLP scorer
        scores = self.scorer(features).squeeze(-1)  # (B, K)

        return scores

    def select(
        self,
        query_emb: torch.Tensor,
        memory_embs: torch.Tensor,
        top_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """推理时: 选出 top-k 最有用的记忆。

        Args:
            query_emb: (B, D) 或 (D,)
            memory_embs: (B, K, D) 或 (K, D)
            top_k: 选出数量, 默认使用配置值.

        Returns:
            (selected_indices, selected_scores):
                selected_indices: (B, top_k) 选中的记忆索引
                selected_scores: (B, top_k) 对应的分数 (经过 sigmoid)
        """
        if top_k is None:
            top_k = self.config.top_k

        with torch.no_grad():
            scores = self.forward(query_emb, memory_embs)  # (B, K)
            probs = torch.sigmoid(scores)  # (B, K) 归一化到 [0, 1]

            # 如果 K <= top_k, 直接全部返回
            K = scores.shape[1]
            if K <= top_k:
                indices = torch.arange(K, device=scores.device).unsqueeze(0).expand(scores.shape[0], -1)
                return indices, probs

            # Top-k 选择
            selected_scores, selected_indices = torch.topk(probs, k=top_k, dim=-1)
            return selected_indices, selected_scores

    def soft_select(
        self,
        query_emb: torch.Tensor,
        memory_embs: torch.Tensor,
    ) -> torch.Tensor:
        """训练时: 生成 soft selection weights (可微分)。

        使用 Gumbel-Softmax 或简单的 sigmoid 产出可微分的选择权重,
        使得梯度可以流过选择操作。

        Args:
            query_emb: (B, D)
            memory_embs: (B, K, D)

        Returns:
            weights: (B, K) 每条记忆的选择权重, ∈ [0, 1].
        """
        scores = self.forward(query_emb, memory_embs)  # (B, K)

        # 使用 sigmoid + temperature 作为 soft selection
        weights = torch.sigmoid(scores / self._temperature)

        return weights

    def compute_loss(
        self,
        predicted_scores: torch.Tensor,
        target_utilities: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """计算 Scorer 的训练损失。

        使用 MSE loss: scorer 预测的 utility 应该接近真实的 ΔLoss。
        同时加一个 ranking loss: 如果 utility_a > utility_b, 则 score_a > score_b。

        Args:
            predicted_scores: (B, K) scorer 输出的原始分数.
            target_utilities: (B, K) 真实的 utility (由 counterfactual ΔLoss 计算).
            reduction: "mean" | "sum" | "none".

        Returns:
            total_loss: 标量 loss.
        """
        # MSE Loss: 直接拟合 utility 值
        mse_loss = F.mse_loss(
            torch.sigmoid(predicted_scores),
            torch.sigmoid(target_utilities),  # 归一化到 [0, 1]
            reduction=reduction,
        )

        # Ranking Loss (BPR-like): 保持排序一致性
        # 对每对 (i, j) 其中 utility_i > utility_j,
        # 要求 score_i > score_j
        K = predicted_scores.shape[1]
        ranking_loss = torch.tensor(0.0, device=predicted_scores.device)

        if K >= 2:
            # 随机采样若干 pair 计算 ranking loss (避免 O(K^2))
            num_pairs = min(K * (K - 1) // 2, 10)
            for _ in range(num_pairs):
                i = torch.randint(0, K, (1,)).item()
                j = torch.randint(0, K, (1,)).item()
                if i == j:
                    continue
                # 如果 target_i > target_j, 则要求 score_i > score_j
                diff_target = target_utilities[:, i] - target_utilities[:, j]
                diff_pred = predicted_scores[:, i] - predicted_scores[:, j]
                # margin ranking loss
                ranking_loss = ranking_loss + F.relu(-diff_target.sign() * diff_pred + 0.1).mean()

            ranking_loss = ranking_loss / max(num_pairs, 1)

        total_loss = mse_loss + 0.1 * ranking_loss
        return total_loss

    @torch.no_grad()
    def compute_counterfactual_utilities(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        memory_embs: torch.Tensor,
        mag_gate: nn.Module,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """离线计算 counterfactual ΔLoss (用于生成训练数据)。

        对每条记忆 c_i, 计算:
            utility(c_i) = L(without c_i) - L(with all)

        Args:
            model: backbone 模型.
            input_ids: (B, T) 输入 token ids.
            memory_embs: (B, K, D) 记忆编码.
            mag_gate: MAGGate 模块 (用于注入记忆).
            attention_mask: (B, T) 注意力 mask.
            labels: (B, T) 目标 token ids.

        Returns:
            utilities: (B, K) 每条记忆的 utility score.
        """
        B, K, D = memory_embs.shape

        # 基准: 使用所有记忆的 loss
        base_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        base_loss = base_output.loss if hasattr(base_output, "loss") and base_output.loss is not None else torch.tensor(0.0)

        utilities = torch.zeros(B, K, device=memory_embs.device)

        # 对每条记忆计算 leave-one-out loss
        for i in range(K):
            # 去掉第 i 条记忆
            mask = torch.ones(K, dtype=torch.bool, device=memory_embs.device)
            mask[i] = False
            partial_embs = memory_embs[:, mask]  # (B, K-1, D)

            # 重新前向 (简化: 只比较最后一层 hidden state 的 loss 差异)
            partial_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )
            partial_loss = partial_output.loss if hasattr(partial_output, "loss") and partial_output.loss is not None else torch.tensor(0.0)

            # utility = L(without c_i) - L(with all)
            utilities[:, i] = partial_loss - base_loss

        return utilities
