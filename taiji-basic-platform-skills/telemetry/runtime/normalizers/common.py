# -*- coding: utf-8 -*-
"""Normalizer helpers."""

EVENT_MAP = {
    "SessionStart": ("session.started", "unknown", "session"),
    "UserPromptSubmit": ("turn.prompt_submitted", "unknown", "turn"),
    "PreToolUse": ("tool.started", "unknown", "tool"),
    "PostToolUse": ("tool.completed", "success", "tool"),
    "PostToolUseFailure": ("tool.failed", "failure", "tool"),
    "Stop": ("turn.completed", "success", "turn"),
}


def first(raw, *names):
    for name in names:
        if isinstance(raw, dict) and name in raw:
            return raw.get(name)
    return None


def map_event(original):
    return EVENT_MAP.get(original, (original or "unknown", "unknown", "unknown"))


def authorization_from_config(config):
    """Return the local consent record snapshot carried with each trace."""
    consent = (config.get("consent") or {}) if isinstance(config, dict) else {}
    if consent.get("accepted") is True:
        return {
            "result": "accepted",
            "decision_at": consent.get("accepted_at"),
        }
    if consent.get("accepted") is False:
        return {
            "result": "declined",
            "decision_at": consent.get("declined_at"),
        }
    return {
        "result": "undecided",
        "decision_at": None,
    }
