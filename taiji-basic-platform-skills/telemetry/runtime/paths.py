# -*- coding: utf-8 -*-
"""Filesystem paths for the installed telemetry runtime."""

import os
from pathlib import Path

from . import constants


def expand(path):
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def manager_home():
    return expand(os.environ.get("TAIJI_SKILL_MANAGER_HOME", constants.DEFAULT_MANAGER_HOME))


def config_path():
    return manager_home() / "config.json"


def version_path():
    return manager_home() / "VERSION"


def hooks_dir():
    return manager_home() / "hooks"


def runtime_dir():
    return manager_home() / "runtime"


def queue_dir():
    return manager_home() / "queue"


def pending_dir():
    return queue_dir() / "pending"


def sending_dir():
    return queue_dir() / "sending"


def failed_dir():
    return queue_dir() / "failed"


def queue_locks_dir():
    return queue_dir() / "locks"


def logs_dir():
    return manager_home() / "logs"


def telemetry_log_path():
    return logs_dir() / "telemetry.log"


def locks_dir():
    return manager_home() / "locks"


def flush_lock_path():
    return locks_dir() / "flush.lock"


def install_lock_path():
    return locks_dir() / "install.lock"


def agent_sync_attempt_stamp_path():
    return state_dir() / "agent-sync-attempt.stamp"


def agent_sync_success_stamp_path():
    return state_dir() / "agent-sync-success.stamp"


def agent_ensure_dir():
    return state_dir() / "agent-ensure"


def agent_ensure_stamp_path(agent_variant):
    safe_variant = "".join(
        char if char.isalnum() or char in "_.-" else "_"
        for char in str(agent_variant)
    )
    return agent_ensure_dir() / (safe_variant + ".stamp")


def backups_dir():
    return manager_home() / "backups"


def state_dir():
    return manager_home() / "state"


def flush_state_path():
    return state_dir() / "flush_state.json"


def file_retry_state_path():
    return state_dir() / "file_retry_state.json"


def quarantine_dir():
    return queue_dir() / "quarantine"


def ensure_runtime_dirs():
    for d in [
        manager_home(),
        hooks_dir(),
        runtime_dir(),
        pending_dir(),
        sending_dir(),
        failed_dir(),
        quarantine_dir(),
        queue_locks_dir(),
        logs_dir(),
        locks_dir(),
        backups_dir(),
        state_dir(),
        agent_ensure_dir(),
    ]:
        d.mkdir(parents=True, exist_ok=True)
