"""
配置管理模块。

基于 OmegaConf 实现层级化 YAML 配置加载与合并。
支持：
  - 单文件加载
  - 多文件层级合并（后面的覆盖前面的）
  - CLI 参数覆盖 (dotlist 格式)
  - defaults 关键字自动继承基础配置
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf


# 项目根目录 & 默认配置目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "configs"


def load_config(
    path: Union[str, Path],
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """加载单个 YAML 配置文件，并可选地用 dotlist 覆盖字段。

    如果配置文件中包含 ``defaults`` 键（列表形式），会自动先加载基础配置
    再合并当前文件。

    Args:
        path: YAML 文件路径（绝对路径或相对于 configs/ 的路径）。
        overrides: OmegaConf dotlist 格式的覆盖列表，如
            ``["training.lr=3e-4", "memory.num_memories=5"]``。

    Returns:
        合并后的 DictConfig。
    """
    path = _resolve_config_path(path)
    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig), f"配置文件应为 dict 格式, 得到 {type(cfg)}"

    # --- 处理 defaults 继承 ---
    if "defaults" in cfg:
        defaults = cfg.pop("defaults")
        if isinstance(defaults, (list, tuple)):
            base_cfgs = [load_config(d) for d in defaults]
        else:
            base_cfgs = [load_config(defaults)]
        # 依次合并基础配置，然后合并当前配置
        merged = base_cfgs[0]
        for bc in base_cfgs[1:]:
            merged = OmegaConf.merge(merged, bc)
        cfg = OmegaConf.merge(merged, cfg)

    # --- CLI 覆盖 ---
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def merge_configs(*configs: Union[DictConfig, Dict[str, Any]]) -> DictConfig:
    """合并多个配置，后面的覆盖前面的。

    Args:
        *configs: 任意数量的 DictConfig 或 dict。

    Returns:
        合并后的 DictConfig。
    """
    result = OmegaConf.create({})
    for c in configs:
        if isinstance(c, dict):
            c = OmegaConf.create(c)
        result = OmegaConf.merge(result, c)
    return result


def config_to_dict(cfg: DictConfig) -> Dict[str, Any]:
    """将 DictConfig 转为普通 Python dict（递归）。"""
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def _resolve_config_path(path: Union[str, Path]) -> Path:
    """将路径解析为绝对路径。如果是相对路径则基于 configs/ 目录。"""
    path = Path(path)
    if path.is_absolute():
        return path
    # 尝试 configs/ 下
    candidate = _CONFIG_DIR / path
    if candidate.exists():
        return candidate
    # 尝试加 .yaml 后缀
    candidate_yaml = _CONFIG_DIR / f"{path}.yaml"
    if candidate_yaml.exists():
        return candidate_yaml
    raise FileNotFoundError(
        f"找不到配置文件: {path} (搜索路径: {_CONFIG_DIR})"
    )
