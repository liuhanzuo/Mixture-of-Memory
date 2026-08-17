# -*- coding: utf-8 -*-
"""Constants for Taiji skill telemetry."""

SKILL_NAME = "taiji-basic-platform-skills"
SKILL_VERSION = "5.0.0"
HOOK_RUNTIME_VERSION = "0.4.7"
CONSENT_VERSION = "2026-07-22"
SCHEMA_VERSION = "1.0"

DEFAULT_MANAGER_HOME = "~/.taiji-skill-manager"
DEFAULT_ENDPOINT = "http://taiji-skills-manager.woa.com/api/telemetry/hook-traces"

HOOK_TIMEOUT_SECONDS = 3
HOOK_MAX_RUNTIME_MS = 1500
LOCK_WAIT_MS = 50

MAX_FILE_SIZE_MB = 10
MAX_EVENTS_PER_FILE = 1000
MAX_EVENT_SIZE_MB = 2
MAX_HTTP_BATCH_SIZE_MB = 10
PENDING_MAX_MB = 50
FAILED_RETENTION_DAYS = 3
FAILED_MAX_MB = 20

FLUSH_CONNECT_TIMEOUT_MS = 300
FLUSH_READ_TIMEOUT_MS = 800
FLUSH_TOTAL_TIMEOUT_MS = 1200
FLUSH_BACKOFF_SECONDS = 60
FLUSH_MAX_RETRY = 5
FLUSH_MAX_DRAIN_SECONDS = 10
FLUSH_IDLE_GRACE_MS = 100
EVENT_ACK_MODE = "event_v1"

AGENT_SYNC_INTERVAL_SECONDS = 24 * 60 * 60
AGENT_SYNC_RETRY_SECONDS = 5 * 60
AGENT_ENSURE_INTERVAL_SECONDS = 24 * 60 * 60
AGENT_SYNC_LOCK_WAIT_MS = 100
INSTALL_LOCK_WAIT_MS = 3000

HOOK_MARKER = ".taiji-skill-manager/hooks/hook.py"
# Codex support for PostToolUseFailure has not been verified, so keep its
# registration set conservative. Claude-family agents (including tclaude)
# support PostToolUseFailure and emit structured error/is_interrupt fields.
CODEX_HOOK_EVENTS = [
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "Stop",
]

CLAUDE_HOOK_EVENTS = [
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "Stop",
]
