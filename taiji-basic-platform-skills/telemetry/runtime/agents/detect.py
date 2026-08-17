# -*- coding: utf-8 -*-
"""Detect installed agent variants."""

import shutil
import subprocess
from typing import Iterable, List, Optional

from ..paths import expand
from .base import AgentInstall, AgentSpec
from .registry import AGENT_SPECS


def _first_binary(binary_names):
    for name in binary_names:
        path = shutil.which(name)
        if path:
            return path
    return None


def _read_version(binary):
    if not binary:
        return None
    try:
        proc = subprocess.Popen(
            [binary, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            universal_newlines=True,
        )
        try:
            out, _ = proc.communicate(timeout=3)
        except TypeError:
            out, _ = proc.communicate()
        except subprocess.TimeoutExpired:
            proc.kill()
            return None
        return (out or "").strip().splitlines()[0] if out else None
    except Exception:
        return None


def detect_one(spec, force=False, read_version=True):
    # Prefer the first existing home, otherwise the first configured home.
    homes = [expand(p) for p in spec.home_candidates]
    existing_homes = [p for p in homes if p.exists()]
    home = existing_homes[0] if existing_homes else homes[0]
    config_path = home / spec.config_relpath
    binary = _first_binary(spec.binary_names)
    detected = bool(binary or home.exists() or config_path.exists())
    if not detected and not force:
        return None
    return AgentInstall(
        variant=spec.variant,
        family=spec.family,
        binary=binary,
        version=_read_version(binary) if read_version else None,
        home=home,
        config_path=config_path,
        detected=detected,
    )


def detect_installs(selected_variants=None, read_versions=True):
    selected = set(selected_variants or [])
    installs = []
    for spec in AGENT_SPECS:
        if selected and spec.variant not in selected:
            continue
        install = detect_one(
            spec, force=bool(selected), read_version=read_versions
        )
        if install:
            installs.append(install)
    return installs
