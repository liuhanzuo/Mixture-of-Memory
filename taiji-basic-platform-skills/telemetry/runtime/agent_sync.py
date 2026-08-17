# -*- coding: utf-8 -*-
"""Throttled background reconciliation for newly installed Agent hooks."""

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from . import constants, paths
from .agents.detect import detect_installs
from .agents.registry import ADAPTERS
from .consent import load_config, now_iso, save_config
from .queue_store import FileLock, log_error


def _recent(path, interval_seconds):
    try:
        return time.time() - Path(path).stat().st_mtime < interval_seconds
    except OSError:
        return False


def _touch(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _eligible(config):
    return bool(config.get("enabled"))


def _interval(config, key, default):
    runtime = config.get("runtime") or {}
    try:
        return max(1, int(runtime.get(key) or default))
    except Exception:
        return default


def agent_sync_due(config):
    """Cheap SessionStart check; no Agent filesystem scan happens here."""
    if not _eligible(config):
        return False
    success_interval = _interval(
        config, "agent_sync_interval_seconds", constants.AGENT_SYNC_INTERVAL_SECONDS
    )
    retry_interval = _interval(
        config, "agent_sync_retry_seconds", constants.AGENT_SYNC_RETRY_SECONDS
    )
    if _recent(paths.agent_sync_success_stamp_path(), success_interval):
        return False
    if _recent(paths.agent_sync_attempt_stamp_path(), retry_interval):
        return False
    return True


def try_start_agent_sync(hook_script_path, config):
    """Start detached reconciliation at most once per retry window."""
    try:
        if not agent_sync_due(config):
            return False
        _touch(paths.agent_sync_attempt_stamp_path())
        subprocess.Popen(
            [sys.executable, str(hook_script_path), "--sync-agents"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=(os.name != "nt"),
        )
        return True
    except Exception as e:
        try:
            paths.agent_sync_attempt_stamp_path().unlink()
        except OSError:
            pass
        log_error("failed to start agent sync: %r" % (e,))
        return False


def _hook_command(hook_path, agent_variant):
    return "python3 %s --agent %s" % (
        shlex.quote(str(hook_path)),
        shlex.quote(agent_variant),
    )


def _hook_markers(hook_path, agent_variant):
    return [
        constants.HOOK_MARKER,
        str(hook_path),
        _hook_command(hook_path, agent_variant),
    ]


def _agent_config(install, existing):
    existing = existing or {}
    return {
        "enabled": existing.get("enabled", True),
        "family": install.family,
        "binary": install.binary or existing.get("binary"),
        "version": install.version if install.version is not None else existing.get("version"),
        "home": str(install.home),
        "config_path": str(install.config_path),
        "detected": bool(install.detected),
    }


def merge_detected_agents(
    config, hook_script_path, read_versions=True, selected_variants=None
):
    """Merge hooks/config for currently installed agents; caller owns install lock."""
    installs = detect_installs(
        selected_variants, read_versions=read_versions
    )
    agents = config.setdefault("agents", {})
    changed = 0
    hook_path = Path(hook_script_path).resolve()
    successful_installs = []
    for install in installs:
        try:
            adapter = ADAPTERS[install.family]
            adapter.merge_hooks(
                install,
                _hook_command(hook_path, install.variant),
                paths.backups_dir(),
                hook_markers=_hook_markers(hook_path, install.variant),
            )
        except Exception as e:
            log_error(
                "agent sync skipped variant=%s config=%s: %r"
                % (install.variant, install.config_path, e)
            )
            continue
        if install.variant not in agents:
            changed += 1
        agents[install.variant] = _agent_config(
            install, agents.get(install.variant)
        )
        successful_installs.append(install)

    runtime = config.setdefault("runtime", {})
    runtime["last_agent_sync_at"] = now_iso()
    runtime.setdefault(
        "agent_sync_interval_seconds", constants.AGENT_SYNC_INTERVAL_SECONDS
    )
    runtime.setdefault(
        "agent_sync_retry_seconds", constants.AGENT_SYNC_RETRY_SECONDS
    )
    save_config(config)
    for install in successful_installs:
        _touch(paths.agent_ensure_stamp_path(install.variant))
    return changed, successful_installs


def sync_detected_agents(hook_script_path, read_versions=True):
    """Detect installed Agents and idempotently repair their telemetry hooks."""
    config = load_config(default={})
    if not _eligible(config):
        return 0

    try:
        with FileLock(paths.install_lock_path(), wait_ms=constants.AGENT_SYNC_LOCK_WAIT_MS):
            # A concurrent --disable/--decline may have changed config after
            # the cheap pre-lock check. Re-read while holding the shared lock
            # so stale sync state cannot re-enable telemetry on save.
            config = load_config(default={})
            if not _eligible(config):
                return 0
            changed, successful_installs = merge_detected_agents(
                config, hook_script_path, read_versions=read_versions
            )
            # Do not mark a failed reconciliation as successful. If every
            # detected Agent failed to accept its hook, leave the success stamp
            # absent so a later SessionStart can retry after the shorter retry
            # interval instead of waiting for the full success interval.
            if successful_installs:
                _touch(paths.agent_sync_success_stamp_path())
            return changed
    except Exception as e:
        log_error("agent sync failed: %r" % (e,))
        return 0
