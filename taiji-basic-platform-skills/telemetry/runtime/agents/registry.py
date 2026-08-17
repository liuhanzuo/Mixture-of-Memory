# -*- coding: utf-8 -*-
"""Registry of supported agent variants and family adapters."""

from .base import AgentSpec
from .codex import CodexAdapter
from .claude import ClaudeAdapter

AGENT_SPECS = [
    AgentSpec("codex", "codex", ["codex"], ["~/.codex"], "hooks.json"),
    AgentSpec("tcodex", "codex", ["tcodex"], ["~/.tcodex"], "hooks.json"),
    AgentSpec("codex-internal", "codex", ["codex-internal"], ["~/.codex-internal"], "hooks.json"),
    AgentSpec("claude", "claude", ["claude"], ["~/.claude"], "settings.json"),
    AgentSpec("tclaude", "claude", ["tclaude"], ["~/.tclaude"], "settings.json"),
    AgentSpec("claude-internal", "claude", ["claude-internal"], ["~/.claude-internal"], "settings.json"),
]

ADAPTERS = {
    "codex": CodexAdapter(),
    "claude": ClaudeAdapter(),
}


def get_spec(variant):
    for spec in AGENT_SPECS:
        if spec.variant == variant:
            return spec
    return None


def get_adapter(family):
    return ADAPTERS[family]
