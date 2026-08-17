# -*- coding: utf-8 -*-
"""Codex-family hook configuration support."""

from copy import deepcopy

from .. import constants
from ..json_utils import atomic_write_json, backup_file, read_json


class CodexAdapter(object):
    family = "codex"
    hook_events = constants.CODEX_HOOK_EVENTS

    def matcher_for_event(self, event_name):
        if event_name == "SessionStart":
            return "startup|resume|clear|compact"
        if event_name in ("PreToolUse", "PostToolUse"):
            return "*"
        return None

    def build_hook_group(self, event_name, hook_command):
        handler = {
            "type": "command",
            "command": hook_command,
            "timeout": constants.HOOK_TIMEOUT_SECONDS,
            "statusMessage": "Taiji Skill telemetry",
        }
        group = {"hooks": [handler]}
        matcher = self.matcher_for_event(event_name)
        if matcher is not None:
            group["matcher"] = matcher
        return group

    def _remove_own_hooks(self, data, hook_markers=None):
        hook_markers = [m for m in (hook_markers or [constants.HOOK_MARKER]) if m]
        hooks = data.get("hooks")
        if not isinstance(hooks, dict):
            return data
        for event_name, groups in list(hooks.items()):
            if not isinstance(groups, list):
                continue
            new_groups = []
            for group in groups:
                if not isinstance(group, dict):
                    new_groups.append(group)
                    continue
                handlers = group.get("hooks")
                if not isinstance(handlers, list):
                    new_groups.append(group)
                    continue
                kept = []
                for h in handlers:
                    cmd = h.get("command") if isinstance(h, dict) else None
                    if not (isinstance(cmd, str) and any(marker in cmd for marker in hook_markers)):
                        kept.append(h)
                if kept:
                    new_group = deepcopy(group)
                    new_group["hooks"] = kept
                    new_groups.append(new_group)
            hooks[event_name] = new_groups
        return data

    def merge_hooks(self, install, hook_command, backup_dir, hook_markers=None):
        data = read_json(install.config_path, default={}) or {}
        if not isinstance(data, dict):
            data = {}
        original = deepcopy(data)
        data = self._remove_own_hooks(data, hook_markers=hook_markers)
        hooks = data.setdefault("hooks", {})
        for event_name in self.hook_events:
            groups = hooks.setdefault(event_name, [])
            if not isinstance(groups, list):
                groups = []
                hooks[event_name] = groups
            groups.append(self.build_hook_group(event_name, hook_command))
        if data != original:
            install.config_path.parent.mkdir(parents=True, exist_ok=True)
            backup_file(install.config_path, backup_dir, install.variant)
            atomic_write_json(install.config_path, data)
        return install.config_path

    def uninstall_hooks(self, install, backup_dir, hook_markers=None):
        data = read_json(install.config_path, default=None)
        if not isinstance(data, dict):
            return False
        original = deepcopy(data)
        data = self._remove_own_hooks(data, hook_markers=hook_markers)
        if data != original:
            backup_file(install.config_path, backup_dir, install.variant)
            atomic_write_json(install.config_path, data)
            return True
        return False
