# -*- coding: utf-8 -*-
"""Unified event schema construction."""

import datetime
import uuid

from . import constants


def now_local():
    return datetime.datetime.now(datetime.timezone.utc).astimezone()


def now_iso():
    return now_local().isoformat()


def date_key_from_iso(iso_text):
    try:
        dt = datetime.datetime.fromisoformat(iso_text)
    except Exception:
        dt = now_local()
    return dt.strftime("%Y%m%d")


def build_event(
    raw_event,
    agent_family,
    agent_variant,
    agent_version,
    install_id,
    hook_runtime_version,
    original_event_name,
    normalized_event_name,
    outcome,
    scope,
    session,
    tool,
    conversation,
    agent_extra,
    consent_version=constants.CONSENT_VERSION,
    authorization=None,
):
    recorded_at = now_iso()
    return {
        "schema_version": constants.SCHEMA_VERSION,
        "event_id": str(uuid.uuid4()),
        "recorded_at": recorded_at,
        "date_key": date_key_from_iso(recorded_at),
        "source": {
            "agent_family": agent_family,
            "agent_variant": agent_variant,
            "agent_version": agent_version,
            "hook_runtime_version": hook_runtime_version,
            "skill_name": constants.SKILL_NAME,
            "skill_version": constants.SKILL_VERSION,
            "install_id": install_id,
        },
        "session": {
            "session_id": session.get("session_id"),
            "turn_id": session.get("turn_id"),
            "cwd": session.get("cwd"),
            "transcript_path": session.get("transcript_path"),
            "model": session.get("model"),
            "permission_mode": session.get("permission_mode"),
        },
        "event": {
            "original_event_name": original_event_name,
            "normalized_event_name": normalized_event_name,
            "outcome": outcome,
            "scope": scope,
        },
        "tool": {
            "tool_name": tool.get("tool_name"),
            "tool_use_id": tool.get("tool_use_id"),
            "tool_input": tool.get("tool_input"),
            "tool_response": tool.get("tool_response"),
            "error": tool.get("error"),
        },
        "conversation": {
            "prompt": conversation.get("prompt"),
            "last_assistant_message": conversation.get("last_assistant_message"),
        },
        "agent_extra": agent_extra or {},
        "raw": {
            "event": raw_event,
        },
        "privacy": {
            "consent_version": consent_version,
            "authorization": authorization or {
                "result": "undecided",
                "decision_at": None,
            },
        },
    }
