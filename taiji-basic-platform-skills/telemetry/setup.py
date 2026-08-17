#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup Taiji skill telemetry hooks after explicit user consent."""

import argparse
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

# Import local runtime from the skill telemetry directory.
_TELEMETRY_DIR = Path(__file__).resolve().parent
if str(_TELEMETRY_DIR) not in sys.path:
    sys.path.insert(0, str(_TELEMETRY_DIR))

from runtime import constants, paths  # noqa: E402
from runtime.agents.detect import detect_installs  # noqa: E402
from runtime.agents.registry import ADAPTERS, AGENT_SPECS  # noqa: E402
from runtime.agent_sync import merge_detected_agents  # noqa: E402
from runtime.queue_store import FileLock  # noqa: E402
from runtime.consent import (  # noqa: E402
    build_config,
    consent_status,
    disable_config,
    load_config,
    record_decline,
    save_config,
)


def _parse_agents(text):
    if not text:
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def _ignore_copy(dirpath, names):
    ignored = []
    for name in names:
        if name == "__pycache__" or name.endswith(".pyc") or name.endswith(".pyo"):
            ignored.append(name)
    return ignored


def _source_runtime_version():
    version_file = _TELEMETRY_DIR / "VERSION"
    try:
        return version_file.read_text(encoding="utf-8").strip()
    except OSError:
        return constants.HOOK_RUNTIME_VERSION


def _runtime_needs_update():
    try:
        installed_version = paths.version_path().read_text(encoding="utf-8").strip()
        return installed_version != _source_runtime_version()
    except OSError:
        return True


def _infer_current_agent_variant():
    source_path = _TELEMETRY_DIR.resolve()
    for spec in AGENT_SPECS:
        for candidate in spec.home_candidates:
            try:
                source_path.relative_to(paths.expand(candidate).resolve())
                return spec.variant
            except ValueError:
                continue
    return None


def _stamp_is_recent(path, interval_seconds):
    try:
        return time.time() - path.stat().st_mtime < interval_seconds
    except OSError:
        return False


def _copy_runtime():
    paths.ensure_runtime_dirs()
    # Runtime versions before 0.3.8 kept an unbounded full event mirror.
    # Remove it during upgrade; delivery uses queue/pending, not this legacy log.
    try:
        (paths.logs_dir() / "events.jsonl").unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
    src_runtime = _TELEMETRY_DIR / "runtime"
    src_hook = _TELEMETRY_DIR / "hook.py"
    src_version = _TELEMETRY_DIR / "VERSION"

    tmp_runtime = paths.manager_home() / "runtime.tmp"
    if tmp_runtime.exists():
        shutil.rmtree(str(tmp_runtime))
    shutil.copytree(str(src_runtime), str(tmp_runtime), ignore=_ignore_copy)
    if paths.runtime_dir().exists():
        shutil.rmtree(str(paths.runtime_dir()))
    os.replace(str(tmp_runtime), str(paths.runtime_dir()))

    paths.hooks_dir().mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src_hook), str(paths.hooks_dir() / "hook.py"))
    try:
        os.chmod(str(paths.hooks_dir() / "hook.py"), 0o755)
    except OSError:
        pass
    if src_version.exists():
        shutil.copy2(str(src_version), str(paths.version_path()))
    else:
        paths.version_path().write_text(constants.HOOK_RUNTIME_VERSION + "\n", encoding="utf-8")


def _hook_command(agent_variant):
    hook_path = paths.hooks_dir() / "hook.py"
    return "python3 %s --agent %s" % (shlex.quote(str(hook_path)), shlex.quote(agent_variant))


def _hook_markers(agent_variant):
    # Remove both normal production hooks and test/runtime-home-specific hooks.
    return [
        constants.HOOK_MARKER,
        str(paths.hooks_dir() / "hook.py"),
        _hook_command(agent_variant),
    ]


def _print_detected(installs):
    if not installs:
        print("未检测到可注册的 Agent。")
        return
    print("检测到以下 Agent：")
    for install in installs:
        marker = "detected" if install.detected else "forced"
        print("- %-15s family=%-6s config=%s (%s)" % (install.variant, install.family, install.config_path, marker))


def _install(args):
    selected = _parse_agents(args.agents)
    installs = detect_installs(selected)
    if not installs:
        print("没有检测到 Agent；可用 --agents 指定要强制注册的 variant。")
        return 1
    try:
        with FileLock(paths.install_lock_path(), wait_ms=constants.INSTALL_LOCK_WAIT_MS):
            _copy_runtime()
            existing = load_config(default={})
            config = build_config(
                installs, endpoint=args.endpoint, existing=existing, installed_from=_TELEMETRY_DIR
            )
            # Do not advertise telemetry as enabled until at least one Agent
            # hook has been written successfully. This prevents a partial or
            # fully failed installation from looking healthy in --status.
            config["enabled"] = False

            successful_installs = []
            for install in installs:
                try:
                    adapter = ADAPTERS[install.family]
                    adapter.merge_hooks(
                        install,
                        _hook_command(install.variant),
                        paths.backups_dir(),
                        hook_markers=_hook_markers(install.variant),
                    )
                except Exception as e:
                    config.get("agents", {}).pop(install.variant, None)
                    print("跳过 %-15s：配置损坏或写入失败：%r" % (install.variant, e))
                    continue
                successful_installs.append(install)
                print("已注册 %-15s -> %s" % (install.variant, install.config_path))

            if not successful_installs:
                if existing.get("enabled"):
                    # A repair/reinstall must not disable a previously working
                    # telemetry setup merely because this attempt could not
                    # update any Agent configuration.
                    print("Telemetry 安装失败：未能成功注册任何 Agent hook；保留原有 Telemetry 配置。")
                else:
                    # First-time installation: no hook was registered, so keep
                    # telemetry disabled rather than reporting a false success.
                    save_config(config)
                    print("Telemetry 安装失败：未能成功注册任何 Agent hook。")
                return 1

            config["enabled"] = True
            save_config(config)
    except Exception as e:
        print("Telemetry 安装失败：%r" % (e,))
        return 1
    print()
    print("Telemetry 已开启。Codex family 可能需要在交互界面运行 /hooks 并 trust 新 hook。")
    print("运行时目录：%s" % paths.manager_home())
    print("上报地址：%s" % ((config.get("backend") or {}).get("endpoint") or ""))
    return 0


def _sync_agents():
    config = load_config(default={})
    if not config.get("enabled"):
        print("Telemetry 未开启，跳过 Agent hook 同步。")
        return 0
    try:
        with FileLock(paths.install_lock_path(), wait_ms=constants.AGENT_SYNC_LOCK_WAIT_MS):
            if _runtime_needs_update():
                _copy_runtime()
            config = load_config(default={})
            if not config.get("enabled"):
                print("Telemetry 未开启，跳过 Agent hook 同步。")
                return 0
            changed, installs = merge_detected_agents(
                config, paths.hooks_dir() / "hook.py", read_versions=False
            )
    except Exception as e:
        print("Telemetry Agent hook 同步失败：%r" % (e,))
        return 1
    print("Telemetry Agent hook 同步完成：detected=%d newly_registered=%d" % (len(installs), changed))
    return 0


def _ensure_current_agent():
    config = load_config(default={})
    if not config.get("enabled"):
        print("Telemetry 未开启，跳过当前 Agent hook 检查。")
        return 0
    variant = _infer_current_agent_variant()
    if not variant:
        print("无法从当前 Skill 路径推断 Agent；请使用 --sync-agents。")
        return 0
    stamp = paths.agent_ensure_stamp_path(variant)
    if not _runtime_needs_update() and _stamp_is_recent(
        stamp, constants.AGENT_ENSURE_INTERVAL_SECONDS
    ):
        print("Telemetry 当前 Agent hook 已在同步周期内：%s" % variant)
        return 0
    try:
        with FileLock(paths.install_lock_path(), wait_ms=constants.AGENT_SYNC_LOCK_WAIT_MS):
            if _runtime_needs_update():
                _copy_runtime()
            config = load_config(default={})
            if not config.get("enabled"):
                print("Telemetry 未开启，跳过当前 Agent hook 检查。")
                return 0
            changed, installs = merge_detected_agents(
                config,
                paths.hooks_dir() / "hook.py",
                read_versions=False,
                selected_variants=[variant],
            )
    except Exception as e:
        print("Telemetry 当前 Agent hook 检查失败：%r" % (e,))
        return 1
    if not installs:
        print("未检测到当前 Agent：%s" % variant)
        return 0
    print("Telemetry 当前 Agent hook 已确保：agent=%s newly_registered=%d" % (variant, changed))
    return 0


def _current_agent_support_status():
    # This is intentionally read-only. Skill hosts use it before checking or
    # modifying telemetry consent so unsupported Agents never see the consent
    # flow and do not create telemetry state files.
    variant = _infer_current_agent_variant()
    if variant:
        print("supported:%s" % variant)
    else:
        print("unsupported")
    return 0


def _consent_status():
    status = consent_status()
    print(status)
    return 0


def _decline():
    record_decline()
    print("Telemetry 授权已拒绝，不会注册或修改任何 Agent hook。")
    return 0


def _status():
    config = load_config(default={})
    print("Telemetry config: %s" % paths.config_path())
    print("enabled: %s" % bool(config.get("enabled")))
    print("endpoint: %s" % ((config.get("backend") or {}).get("endpoint") or ""))
    print("installed agents:")
    for name, agent in sorted((config.get("agents") or {}).items()):
        print("- %-15s enabled=%s config=%s" % (name, agent.get("enabled"), agent.get("config_path")))
    print()
    _print_detected(detect_installs())
    return 0


def _disable():
    disable_config()
    print("Telemetry 已关闭（hook 仍保留，但会直接退出）。")
    return 0


def _all_for_uninstall(selected=None):
    # Force selected variants so uninstall can remove hooks even if binary is absent.
    variants = selected or [spec.variant for spec in AGENT_SPECS]
    return detect_installs(variants)


def _uninstall_hooks(args):
    selected = _parse_agents(args.agents)
    installs = _all_for_uninstall(selected)
    changed = 0
    for install in installs:
        adapter = ADAPTERS[install.family]
        if adapter.uninstall_hooks(install, paths.backups_dir(), hook_markers=_hook_markers(install.variant)):
            changed += 1
            print("已移除 %-15s -> %s" % (install.variant, install.config_path))
    print("完成，修改配置数：%d" % changed)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Setup Taiji skill telemetry")
    parser.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--agents", help="comma-separated variants to register")
    parser.add_argument("--endpoint", help="skill-manager telemetry endpoint")
    parser.add_argument("--status", action="store_true", help="show telemetry status")
    parser.add_argument("--sync-agents", action="store_true", help="quickly register missing hooks for detected agents")
    parser.add_argument("--ensure-current-agent", action="store_true", help="ensure only the current skill host agent hook")
    parser.add_argument("--current-agent-support-status", action="store_true", help="print whether the current Skill host Agent is supported")
    parser.add_argument("--consent-status", action="store_true", help="print accepted, declined, or undecided")
    parser.add_argument("--decline", action="store_true", help="record telemetry consent refusal without installing hooks")
    parser.add_argument("--disable", action="store_true", help="disable telemetry without removing hooks")
    parser.add_argument("--uninstall-hooks", action="store_true", help="remove telemetry hooks")
    args = parser.parse_args(argv)

    if args.current_agent_support_status:
        return _current_agent_support_status()
    if args.consent_status:
        return _consent_status()
    if args.decline:
        return _decline()
    if args.ensure_current_agent:
        return _ensure_current_agent()
    if args.sync_agents:
        return _sync_agents()
    if args.status:
        return _status()
    if args.disable:
        return _disable()
    if args.uninstall_hooks:
        return _uninstall_hooks(args)
    return _install(args)


if __name__ == "__main__":
    raise SystemExit(main())
