#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Taiji telemetry hook entrypoint."""

import argparse
import json
import os
import sys
import threading
from pathlib import Path

# Make both source-tree and installed runtime imports work:
# - source: telemetry/hook.py + telemetry/runtime/
# - installed: ~/.taiji-skill-manager/hooks/hook.py + ~/.taiji-skill-manager/runtime/
_RUNTIME_PARENT = Path(__file__).resolve().parent
if (_RUNTIME_PARENT.parent / "runtime").exists():
    _RUNTIME_PARENT = _RUNTIME_PARENT.parent
if str(_RUNTIME_PARENT) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_PARENT))

from runtime import constants, paths  # noqa: E402
from runtime.consent import load_config  # noqa: E402
from runtime.queue_store import append_event, log_error  # noqa: E402
from runtime.sender import flush_pending, try_start_flusher  # noqa: E402
from runtime.agent_sync import sync_detected_agents, try_start_agent_sync  # noqa: E402
from runtime.normalizers.codex import normalize_codex_event  # noqa: E402
from runtime.normalizers.claude import normalize_claude_event  # noqa: E402


AGENT_FAMILY = {
    "codex": "codex",
    "tcodex": "codex",
    "codex-internal": "codex",
    "claude": "claude",
    "tclaude": "claude",
    "claude-internal": "claude",
}


def _start_watchdog():
    def _exit():
        os._exit(0)

    timer = threading.Timer(constants.HOOK_MAX_RUNTIME_MS / 1000.0, _exit)
    timer.daemon = True
    timer.start()
    return timer


def _read_stdin_json():
    raw = sys.stdin.read()
    if not raw:
        return {}
    return json.loads(raw)


def _normalize(raw_event, agent_variant, config):
    family = AGENT_FAMILY.get(agent_variant)
    if family == "codex":
        return normalize_codex_event(raw_event, agent_variant, config)
    if family == "claude":
        return normalize_claude_event(raw_event, agent_variant, config)
    # Unknown fallback: keep raw under a Claude-like permissive normalizer shape.
    return normalize_claude_event(raw_event, agent_variant or "unknown", config)


def hook_main(agent_variant):
    try:
        config = load_config(default={})
        if not config.get("enabled"):
            return 0
        if agent_variant:
            agent_cfg = (config.get("agents") or {}).get(agent_variant)
            if agent_cfg is not None and not agent_cfg.get("enabled", True):
                return 0
        raw_event = _read_stdin_json()
        event = _normalize(raw_event, agent_variant or "unknown", config)
        queued = append_event(event, config)
        if queued:
            try_start_flusher(Path(__file__).resolve(), config)
        if ((event.get("event") or {}).get("normalized_event_name") == "session.started"):
            try_start_agent_sync(Path(__file__).resolve(), config)
    except Exception as e:
        log_error("hook_main failed: %r" % (e,))
    return 0


def sync_agents_main():
    try:
        sync_detected_agents(Path(__file__).resolve())
    except Exception as e:
        log_error("sync_agents_main failed: %r" % (e,))
    return 0


def flush_main():
    try:
        config = load_config(default={})
        if not config.get("enabled"):
            return 0
        flush_pending(config)
    except Exception as e:
        log_error("flush_main failed: %r" % (e,))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Taiji telemetry hook")
    parser.add_argument("--agent", default="unknown", help="agent variant")
    parser.add_argument("--flush", action="store_true", help="run detached queue flusher")
    parser.add_argument("--sync-agents", action="store_true", help="reconcile detected Agent hooks")
    args = parser.parse_args(argv)
    if args.sync_agents:
        return sync_agents_main()
    if args.flush:
        # The flusher is detached from the Agent request path and has bounded
        # HTTP timeouts. Do not apply the 1.5s hook watchdog to batch uploads.
        return flush_main()

    timer = _start_watchdog()
    try:
        return hook_main(args.agent)
    finally:
        try:
            timer.cancel()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
