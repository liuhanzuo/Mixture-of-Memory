"""
训练工具函数。

包含:
  - 可训练参数统计
  - 优化器构建
  - 学习率调度器构建
  - 梯度裁剪
  - 检查点保存/加载
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

logger = logging.getLogger(__name__)


# ============================================================
# 参数统计
# ============================================================

def count_trainable_params(model: nn.Module) -> Dict[str, int]:
    """统计模型中可训练 / 总参数量。

    Args:
        model: PyTorch 模型。

    Returns:
        dict: 包含 total, trainable, frozen 三个键。
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def log_param_summary(modules: Dict[str, nn.Module]) -> None:
    """逐模块打印参数统计。

    Args:
        modules: {名称: 模块} 字典。
    """
    total_trainable = 0
    for name, module in modules.items():
        stats = count_trainable_params(module)
        total_trainable += stats["trainable"]
        logger.info(
            f"  [{name}] total={stats['total']:,}  "
            f"trainable={stats['trainable']:,}  frozen={stats['frozen']:,}"
        )
    logger.info(f"  [总计可训练参数] {total_trainable:,}")


# ============================================================
# 优化器
# ============================================================

def get_optimizer(
    params: Iterable[torch.Tensor],
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
) -> AdamW:
    """构建 AdamW 优化器。

    Args:
        params: 可训练参数迭代器。
        lr: 学习率。
        weight_decay: 权重衰减。
        betas: Adam 的 beta 参数。
        eps: Adam 的 epsilon。

    Returns:
        AdamW 优化器。
    """
    optimizer = AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    logger.info(f"创建 AdamW 优化器: lr={lr}, wd={weight_decay}")
    return optimizer


def collect_trainable_params(
    modules: Dict[str, nn.Module],
) -> List[Dict[str, Any]]:
    """从多个模块中收集可训练参数，支持按模块分组设置学习率。

    Args:
        modules: {名称: 模块} 字典。

    Returns:
        参数组列表，可直接传给优化器。
    """
    param_groups = []
    for name, module in modules.items():
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            param_groups.append({"params": params, "name": name})
    return param_groups


# ============================================================
# 学习率调度器
# ============================================================

def get_scheduler(
    optimizer: Optimizer,
    warmup_steps: int = 200,
    max_steps: int = 5000,
    min_lr_ratio: float = 0.1,
) -> _LRScheduler:
    """构建带 warmup 的余弦退火调度器。

    学习率变化：
      - [0, warmup_steps): 线性 warmup 从 0 到 lr
      - [warmup_steps, max_steps]: 余弦退火到 min_lr_ratio * lr

    Args:
        optimizer: 优化器。
        warmup_steps: warmup 步数。
        max_steps: 总训练步数。
        min_lr_ratio: 最终学习率与初始学习率的比值。

    Returns:
        LambdaLR 调度器。
    """
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # 线性 warmup
            return float(current_step) / float(max(1, warmup_steps))
        # 余弦退火
        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

    scheduler = LambdaLR(optimizer, lr_lambda)
    logger.info(
        f"创建余弦退火调度器: warmup={warmup_steps}, "
        f"max_steps={max_steps}, min_lr_ratio={min_lr_ratio}"
    )
    return scheduler


# ============================================================
# 梯度裁剪
# ============================================================

def clip_grad_norm(
    params: Iterable[torch.Tensor],
    max_norm: float = 1.0,
) -> float:
    """梯度裁剪并返回裁剪前的梯度范数。

    Args:
        params: 参数迭代器。
        max_norm: 最大梯度范数。

    Returns:
        裁剪前的总梯度范数。
    """
    params_list = [p for p in params if p.grad is not None]
    if not params_list:
        return 0.0
    total_norm = torch.nn.utils.clip_grad_norm_(params_list, max_norm)
    return total_norm.item() if isinstance(total_norm, torch.Tensor) else float(total_norm)


# ============================================================
# 检查点
# ============================================================

def save_checkpoint(
    path: Union[str, Path],
    step: int,
    modules: Dict[str, nn.Module],
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """保存训练检查点。

    Args:
        path: 保存路径。
        step: 当前训练步数。
        modules: {名称: 模块} 字典。
        optimizer: 可选的优化器状态。
        scheduler: 可选的调度器状态。
        metrics: 可选的指标字典。
        extra: 可选的额外信息。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {"step": step}

    # 保存各模块的 state_dict
    for name, module in modules.items():
        checkpoint[f"module_{name}"] = module.state_dict()

    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if metrics is not None:
        checkpoint["metrics"] = metrics
    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, str(path))
    logger.info(f"检查点已保存: {path} (step={step})")


def load_checkpoint(
    path: Union[str, Path],
    modules: Dict[str, nn.Module],
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """加载训练检查点。

    Args:
        path: 检查点路径。
        modules: {名称: 模块} 字典，会被就地更新。
        optimizer: 可选的优化器，会被就地更新。
        scheduler: 可选的调度器，会被就地更新。
        strict: 是否严格匹配 state_dict。

    Returns:
        checkpoint: 包含 step, metrics 等信息的字典。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"检查点不存在: {path}")

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

    # 恢复各模块
    for name, module in modules.items():
        key = f"module_{name}"
        if key in checkpoint:
            module.load_state_dict(checkpoint[key], strict=strict)
            logger.info(f"  已恢复模块: {name}")
        else:
            logger.warning(f"  检查点中未找到模块: {name}")

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("  已恢复优化器状态")

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info("  已恢复调度器状态")

    step = checkpoint.get("step", 0)
    logger.info(f"检查点已加载: {path} (step={step})")
    return checkpoint
