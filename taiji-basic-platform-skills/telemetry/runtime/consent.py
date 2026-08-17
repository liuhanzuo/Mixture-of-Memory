# -*- coding: utf-8 -*-
"""Telemetry config and consent helpers."""

import datetime
import os
import uuid

from . import constants
from .json_utils import (
    JsonFileCorruptionError,
    append_text_best_effort,
    atomic_write_json,
    read_json,
)
from .queue_store import FileLock
from . import paths


def load_config(default=None):
    fallback = default if default is not None else {}
    try:
        return read_json(paths.config_path(), default=fallback) or {}
    except JsonFileCorruptionError as e:
        append_text_best_effort(
            paths.telemetry_log_path(),
            "[telemetry] ERROR config JSON corrupted; fail-closed with default: %s\n" % e,
        )
        return fallback


def save_config(config):
    paths.ensure_runtime_dirs()
    atomic_write_json(paths.config_path(), config)


def is_enabled(config):
    return bool(config and config.get("enabled"))


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()


def build_config(installs, endpoint=None, existing=None, installed_from=None):
    existing = existing or {}
    install_id = existing.get("install_id") or str(uuid.uuid4())
    existing_backend = existing.get("backend") or {}
    endpoint_value = endpoint
    if endpoint_value is None:
        endpoint_value = os.environ.get("TAIJI_SKILL_MANAGER_ENDPOINT") or existing_backend.get("endpoint") or constants.DEFAULT_ENDPOINT
    agents = {}
    for install in installs:
        agents[install.variant] = {
            "enabled": True,
            "family": install.family,
            "binary": install.binary,
            "version": install.version,
            "home": str(install.home),
            "config_path": str(install.config_path),
            "detected": bool(install.detected),
        }
    return {
        "enabled": True,
        "install_id": install_id,
        "consent": {
            "accepted": True,
            "consent_version": constants.CONSENT_VERSION,
            "accepted_at": now_iso(),
        },
        "backend": {
            "endpoint": endpoint_value or "",
            "timeout_ms": constants.FLUSH_TOTAL_TIMEOUT_MS,
        },
        "runtime": {
            "hook_runtime_version": constants.HOOK_RUNTIME_VERSION,
            "installed_at": now_iso(),
            "installed_from": str(installed_from) if installed_from else None,
            "agent_sync_interval_seconds": constants.AGENT_SYNC_INTERVAL_SECONDS,
            "agent_sync_retry_seconds": constants.AGENT_SYNC_RETRY_SECONDS,
        },
        "queue": {
            "max_file_size_mb": constants.MAX_FILE_SIZE_MB,
            "max_events_per_file": constants.MAX_EVENTS_PER_FILE,
            "max_event_size_mb": constants.MAX_EVENT_SIZE_MB,
            "max_http_batch_size_mb": constants.MAX_HTTP_BATCH_SIZE_MB,
            "pending_max_mb": constants.PENDING_MAX_MB,
            "failed_retention_days": constants.FAILED_RETENTION_DAYS,
            "failed_max_mb": constants.FAILED_MAX_MB,
        },
        "agents": agents,
    }



def consent_status(config=None):
    config = config if config is not None else load_config(default={})
    consent = config.get("consent") or {}
    if consent.get("accepted") is True:
        return "accepted"
    if consent.get("accepted") is False:
        return "declined"
    return "undecided"


def record_decline():
    with FileLock(paths.install_lock_path(), wait_ms=constants.INSTALL_LOCK_WAIT_MS):
        config = load_config(default={})
        config["enabled"] = False
        config["consent"] = {
            "accepted": False,
            "consent_version": constants.CONSENT_VERSION,
            "declined_at": now_iso(),
        }
        save_config(config)
        return config


def disable_config():
    with FileLock(paths.install_lock_path(), wait_ms=constants.INSTALL_LOCK_WAIT_MS):
        config = load_config(default={})
        config["enabled"] = False
        save_config(config)
        return config
