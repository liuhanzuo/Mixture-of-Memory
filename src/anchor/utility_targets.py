"""
Utility Targets: 计算 drop-write utility 教师信号，用于训练 BlockEvaluator。

核心思想:
- 对于选中的 anchor 或 block 候选:
  - 计算写入 memory 后的 future loss: L_future(+n)
  - 计算不写入 memory 时的 future loss: L_future(-n)
  - utility = L_future(-n) - L_future(+n)
  - 正值表示写入有益，负值表示写入无益

实现说明:
- v0 使用近似、高效的实现
- 支持采样候选、缓存 forward pass、近似批量评估
"""

from __future__ import annotations

from typing import Optional, Callable

import torch
import torch.nn as nn


@torch.no_grad()
def compute_utility_targets(
    model_forward_fn: Callable,
    input_ids: torch.Tensor,
    block_hidden: torch.Tensor,
    anchor_indices: torch.Tensor,
    memory_state_with_write: dict,
    memory_state_without_write: dict,
    future_window: int = 32,
    block_start_pos: int = 0,
) -> torch.Tensor:
    """
    计算 drop-write utility target。

    通过对比「写入 memory」和「不写入 memory」两种情况下的 future loss，
    衡量每个 anchor 写入的价值。

    Args:
        model_forward_fn: 前向传播函数，接受 (input_ids, memory_state) -> loss
            签名: fn(input_ids: Tensor, memory_state: dict) -> Tensor (scalar loss)
        input_ids: 用于评估的 future token ids [B, T]
        block_hidden: 当前 block 的 hidden states [B, L, D]
        anchor_indices: 选中的 anchor 位置 [B, K]
        memory_state_with_write: 执行写入后的 memory state dict
        memory_state_without_write: 不执行写入的 memory state dict
        future_window: 用于评估的 future token 窗口大小
        block_start_pos: block 在序列中的起始位置

    Returns:
        utility: [B, K] 每个 anchor 的 utility 值
                 正值 = 写入有益，负值 = 写入无益
    """
    B, K = anchor_indices.shape

    # 截取 future 部分的 input_ids 用于评估
    future_start = block_start_pos + block_hidden.shape[1]
    future_end = min(future_start + future_window, input_ids.shape[1])

    if future_end <= future_start:
        # 没有 future tokens 可用，返回零 utility
        return torch.zeros(B, K, device=block_hidden.device)

    future_ids = input_ids[:, future_start:future_end]

    # 计算写入后的 future loss
    loss_with = model_forward_fn(future_ids, memory_state_with_write)

    # 计算不写入的 future loss
    loss_without = model_forward_fn(future_ids, memory_state_without_write)

    # utility = L(-n) - L(+n), 正值表示写入有益
    # 这里得到的是 batch-level scalar，需要广播到每个 anchor
    if loss_with.dim() == 0:
        # scalar loss -> 广播到所有 anchor
        utility = (loss_without - loss_with).unsqueeze(0).unsqueeze(0).expand(B, K)
    else:
        # per-sample loss [B] -> [B, 1] -> [B, K]
        utility = (loss_without - loss_with).unsqueeze(-1).expand(B, K)

    return utility


@torch.no_grad()
def compute_per_anchor_utility(
    model_forward_fn: Callable,
    input_ids: torch.Tensor,
    block_hidden: torch.Tensor,
    anchor_indices: torch.Tensor,
    gathered_vectors: torch.Tensor,
    memory_module: nn.Module,
    write_head: nn.Module,
    base_memory_state: dict,
    future_window: int = 32,
    block_start_pos: int = 0,
    max_candidates: int = 4,
) -> torch.Tensor:
    """
    逐 anchor 计算 utility（更精细但更慢）。

    对每个 anchor 分别计算写入/不写入的 future loss 差异。
    适用于需要精细 utility label 的场景。

    Args:
        model_forward_fn: 前向传播函数
        input_ids: 完整 input ids [B, T]
        block_hidden: 当前 block hidden states [B, L, D]
        anchor_indices: 选中的 anchor 位置 [B, K]
        gathered_vectors: retrospective gather 输出 [B, K, D_gather]
        memory_module: MOM 模块实例
        write_head: 写入决策头
        base_memory_state: 写入前的基础 memory state
        future_window: future 评估窗口
        block_start_pos: block 起始位置
        max_candidates: 最多评估多少个 anchor（采样以节省计算）

    Returns:
        utility: [B, K] 每个 anchor 的 utility，未被评估的位置为 0
    """
    B, K, D = gathered_vectors.shape
    device = gathered_vectors.device

    # 截取 future tokens
    future_start = block_start_pos + block_hidden.shape[1]
    future_end = min(future_start + future_window, input_ids.shape[1])

    if future_end <= future_start:
        return torch.zeros(B, K, device=device)

    future_ids = input_ids[:, future_start:future_end]

    # 计算 baseline loss（不写入任何 anchor）
    loss_baseline = model_forward_fn(future_ids, base_memory_state)
    if loss_baseline.dim() == 0:
        loss_baseline = loss_baseline.unsqueeze(0).expand(B)

    utility = torch.zeros(B, K, device=device)

    # 只评估前 max_candidates 个 anchor（按分数排序后的前几个）
    eval_k = min(K, max_candidates)

    for k_idx in range(eval_k):
        # 保存 memory state，单独写入第 k 个 anchor
        memory_module.load_state_dict_from_dict(base_memory_state)

        # 获取第 k 个 anchor 的 gathered vector
        z_k = gathered_vectors[:, k_idx:k_idx+1, :]  # [B, 1, D]

        # 通过 write head 获取写入参数
        write_params = write_head(z_k.squeeze(1))  # 返回 alpha, rho, lambda

        # 执行单个 anchor 的写入
        memory_module.update_single(
            z_k.squeeze(1),
            write_params["alpha"],
            write_params["rho"],
            write_params["lambdas"],
        )

        # 获取写入后的 memory state
        state_with_k = memory_module.get_state_dict()

        # 计算写入后的 future loss
        loss_with_k = model_forward_fn(future_ids, state_with_k)
        if loss_with_k.dim() == 0:
            loss_with_k = loss_with_k.unsqueeze(0).expand(B)

        utility[:, k_idx] = loss_baseline - loss_with_k

    return utility


class UtilityTargetGenerator:
    """
    Utility target 生成器，封装了 utility 计算的完整流程。

    使用方式:
    1. 在每个 block 结束后调用 generate()
    2. 将输出用作 BlockEvaluator 的监督信号
    """

    def __init__(
        self,
        future_window: int = 32,
        max_candidates: int = 4,
        mode: str = "block",  # "block" 或 "per_anchor"
        normalize: bool = True,
    ) -> None:
        """
        Args:
            future_window: 用于评估的 future token 窗口大小
            max_candidates: per-anchor 模式下最多评估的 anchor 数
            mode: "block" = 整体写入 vs 不写入; "per_anchor" = 逐 anchor 评估
            normalize: 是否对 utility 做归一化
        """
        self.future_window = future_window
        self.max_candidates = max_candidates
        self.mode = mode
        self.normalize = normalize

    @torch.no_grad()
    def generate(
        self,
        model_forward_fn: Callable,
        input_ids: torch.Tensor,
        block_hidden: torch.Tensor,
        anchor_indices: torch.Tensor,
        memory_state_with_write: dict,
        memory_state_without_write: dict,
        block_start_pos: int = 0,
    ) -> torch.Tensor:
        """
        生成 utility targets。

        Returns:
            utility: [B, K] 归一化后的 utility 值
        """
        utility = compute_utility_targets(
            model_forward_fn=model_forward_fn,
            input_ids=input_ids,
            block_hidden=block_hidden,
            anchor_indices=anchor_indices,
            memory_state_with_write=memory_state_with_write,
            memory_state_without_write=memory_state_without_write,
            future_window=self.future_window,
            block_start_pos=block_start_pos,
        )

        if self.normalize and utility.numel() > 0:
            # 在 block 维度上做 min-max 归一化到 [0, 1]
            u_min = utility.min(dim=-1, keepdim=True).values
            u_max = utility.max(dim=-1, keepdim=True).values
            denom = (u_max - u_min).clamp(min=1e-8)
            utility = (utility - u_min) / denom

        return utility
