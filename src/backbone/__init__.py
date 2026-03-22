"""
Backbone 模块 —— 提供统一的模型前向接口。

支持:
- Full Attention 基线 (上界参照)
- SWA / Local Attention 骨干 (主实验)
- Memory-Readable 扩展接口 (用于 L1 记忆读出集成)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.backbone.hidden_state_types import BackboneOutput
from src.backbone.interfaces import BackboneModel, MemoryReadableBackbone
from src.backbone.swa_model import SWABackbone
from src.backbone.full_attention_model import FullAttentionBackbone

logger = logging.getLogger(__name__)

__all__ = [
    "BackboneOutput",
    "BackboneModel",
    "MemoryReadableBackbone",
    "SWABackbone",
    "FullAttentionBackbone",
    "build_backbone_from_config",
]


def build_backbone_from_config(
    cfg: DictConfig | dict[str, Any],
    config_dir: str | Path | None = None,
) -> BackboneModel:
    """从实验配置构建 backbone 模型的统一工厂函数。

    支持两种配置输入形式:
    1. 完整实验配置 (含 defaults 引用) — 自动解析 model 配置
    2. 直接的 backbone 配置字典

    配置解析优先级:
    - cfg.backbone → 直接使用
    - cfg.model → 作为 model 配置文件名，从 configs/model/ 加载
    - 自动检测 debug 模式 (model_name_or_path 不存在时 fallback)

    Args:
        cfg: 实验配置或 backbone 配置。
        config_dir: 配置文件根目录 (用于解析 defaults 引用)。

    Returns:
        构建好的 BackboneModel 实例。

    Example::

        # 从实验配置
        exp_cfg = OmegaConf.load("configs/exp/swa_mom.yaml")
        backbone = build_backbone_from_config(exp_cfg, config_dir="configs")

        # 从 backbone 配置
        model_cfg = OmegaConf.load("configs/model/swa_qwen.yaml")
        backbone = build_backbone_from_config(model_cfg)
    """
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    # ---- 提取 backbone 配置 ---- #
    backbone_cfg = _resolve_backbone_config(cfg, config_dir)

    if backbone_cfg is None:
        logger.warning(
            "[build_backbone] 未找到有效的 backbone 配置，"
            "将使用 debug tiny 模型。"
        )
        backbone_cfg = OmegaConf.create({
            "type": "swa",
            "debug": True,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "vocab_size": 1000,
            "max_seq_len": 512,
            "device": "cpu",
            "dtype": "float32",
            "window_size": 64,
        })

    # ---- 自动检测是否需要 debug 模式 ---- #
    if not backbone_cfg.get("debug", False):
        model_path = backbone_cfg.get("model_name_or_path", "")
        if model_path and not Path(model_path).exists():
            logger.warning(
                f"[build_backbone] 模型路径不存在: {model_path}，"
                f"自动切换到 debug 模式。"
            )
            backbone_cfg = OmegaConf.merge(backbone_cfg, {"debug": True})

    # ---- 根据 type 选择构建器 ---- #
    backbone_type = backbone_cfg.get("type", "swa").lower()

    if backbone_type in ("swa", "sliding_window"):
        logger.info(f"[build_backbone] 构建 SWA backbone (debug={backbone_cfg.get('debug', False)})")
        backbone = SWABackbone.from_config(backbone_cfg)
    elif backbone_type in ("full_attention", "fullattn", "full"):
        logger.info(f"[build_backbone] 构建 Full Attention backbone (debug={backbone_cfg.get('debug', False)})")
        backbone = FullAttentionBackbone.from_config(backbone_cfg)
    else:
        raise ValueError(
            f"未知的 backbone 类型: {backbone_type!r}。"
            f"支持: 'swa', 'full_attention'"
        )

    # 日志
    logger.info(
        f"[build_backbone] Backbone 构建完成: "
        f"type={backbone_type}, dim={backbone.get_hidden_dim()}, "
        f"layers={backbone.get_num_layers()}, device={backbone.get_device()}, "
        f"debug={backbone.is_debug()}, "
        f"tokenizer={'✓' if backbone.get_tokenizer() else '✗'}"
    )
    return backbone


def _resolve_backbone_config(
    cfg: DictConfig,
    config_dir: str | Path | None = None,
) -> DictConfig | None:
    """从实验配置中解析出 backbone 配置。

    尝试以下路径 (按优先级):
    1. cfg.backbone — 直接的 backbone 配置
    2. cfg 本身包含 backbone 字段 (如直接传入 model yaml)
    3. 从 defaults 中的 /model 引用加载 model yaml

    Returns:
        解析出的 backbone DictConfig，若无则返回 None。
    """
    # 路径 1: cfg.backbone 直接存在
    if "backbone" in cfg:
        return cfg.backbone

    # 路径 2: cfg 本身就是 model yaml (含 backbone.type 等)
    if "type" in cfg and "model_name_or_path" in cfg:
        return cfg

    # 路径 3: 从 defaults 引用加载 model yaml
    if config_dir is not None:
        config_dir = Path(config_dir)
        defaults = cfg.get("defaults", [])
        for item in defaults:
            if isinstance(item, dict):
                # defaults 格式: {"/model": "swa_qwen"}
                for key, value in item.items():
                    if "model" in key:
                        model_yaml_path = config_dir / "model" / f"{value}.yaml"
                        if model_yaml_path.exists():
                            logger.info(f"[build_backbone] 从 defaults 加载 model 配置: {model_yaml_path}")
                            model_cfg = OmegaConf.load(str(model_yaml_path))
                            if "backbone" in model_cfg:
                                return model_cfg.backbone
                            return model_cfg
            elif isinstance(item, str) and "model" in item:
                # defaults 格式: "/model: swa_qwen" (字符串形式)
                parts = item.split(":")
                if len(parts) == 2:
                    value = parts[1].strip()
                    model_yaml_path = config_dir / "model" / f"{value}.yaml"
                    if model_yaml_path.exists():
                        model_cfg = OmegaConf.load(str(model_yaml_path))
                        if "backbone" in model_cfg:
                            return model_cfg.backbone
                        return model_cfg

    # 路径 4: 尝试从实验配置的 model 字段获取模型名
    model_ref = cfg.get("model", None)
    if isinstance(model_ref, str) and config_dir is not None:
        model_yaml_path = Path(config_dir) / "model" / f"{model_ref}.yaml"
        if model_yaml_path.exists():
            model_cfg = OmegaConf.load(str(model_yaml_path))
            if "backbone" in model_cfg:
                return model_cfg.backbone
            return model_cfg

    return None
