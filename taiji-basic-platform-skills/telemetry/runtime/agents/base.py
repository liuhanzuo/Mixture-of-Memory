# -*- coding: utf-8 -*-
"""Base types for agent detection and hook configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class AgentSpec:
    variant: str
    family: str
    binary_names: List[str]
    home_candidates: List[str]
    config_relpath: str
    enabled_by_default: bool = True


@dataclass
class AgentInstall:
    variant: str
    family: str
    binary: Optional[str]
    version: Optional[str]
    home: Path
    config_path: Path
    detected: bool
