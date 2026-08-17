# -*- coding: utf-8 -*-
"""Claude-family hook event normalizer."""

from .. import constants
from ..schema import build_event
from .common import authorization_from_config, first, map_event


def normalize_claude_event(raw, agent_variant, config):
    original = first(raw, "hook_event_name", "hookEventName")
    normalized, outcome, scope = map_event(original)
    agent_cfg = (config.get("agents") or {}).get(agent_variant, {})
    runtime_cfg = config.get("runtime") or {}
    consent_cfg = config.get("consent") or {}
    return build_event(
        raw_event=raw,
        agent_family="claude",
        agent_variant=agent_variant,
        agent_version=agent_cfg.get("version"),
        install_id=config.get("install_id"),
        hook_runtime_version=runtime_cfg.get("hook_runtime_version") or constants.HOOK_RUNTIME_VERSION,
        original_event_name=original,
        normalized_event_name=normalized,
        outcome=outcome,
        scope=scope,
        session={
            "session_id": first(raw, "session_id", "sessionId"),
            "turn_id": first(raw, "turn_id", "turnId"),
            "cwd": first(raw, "cwd"),
            "transcript_path": first(raw, "transcript_path", "transcriptPath"),
            "model": first(raw, "model"),
            "permission_mode": first(raw, "permission_mode", "permissionMode"),
        },
        tool={
            "tool_name": first(raw, "tool_name", "toolName"),
            "tool_use_id": first(raw, "tool_use_id", "toolUseId"),
            "tool_input": first(raw, "tool_input", "toolInput"),
            "tool_response": first(raw, "tool_response", "toolResponse"),
            "error": first(raw, "error"),
        },
        conversation={
            "prompt": first(raw, "prompt"),
            "last_assistant_message": first(
                raw, "last_assistant_message", "lastAssistantMessage"
            ),
        },
        agent_extra={
            "raw_event_name_field": original,
            "agent_id": first(raw, "agent_id", "agentId"),
            "agent_type": first(raw, "agent_type", "agentType"),
            "source": first(raw, "source"),
            "stop_hook_active": first(raw, "stop_hook_active", "stopHookActive"),
            "is_interrupt": first(raw, "is_interrupt", "isInterrupt"),
            "duration_ms": first(raw, "duration_ms", "durationMs"),
            "effort": first(raw, "effort"),
        },
        consent_version=consent_cfg.get("consent_version") or constants.CONSENT_VERSION,
        authorization=authorization_from_config(config),
    )
