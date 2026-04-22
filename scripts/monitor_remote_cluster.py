#!/usr/bin/env python3
"""
monitor_remote_cluster.py — Health-check and auto-recover remote L20A training cluster.

Checks all 4 nodes (28.89.17.143, .144, .85, 19.134) for:
  • SSH reachability
  • GPU utilisation & memory
  • Expected training processes alive (from remote_experiments.json)
  • Output dir progress (latest checkpoint / log lines)
  • Stale / zombie processes
  • Auto-recovery: kill stale process → relaunch via dispatch_remote_training.py

Usage:
  python scripts/monitor_remote_cluster.py [--check-only] [--recover] [--loop INTERVAL_SECS] [--json-config PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "remote_experiments.json"
LOG_DIR = PROJECT_ROOT / "logs" / "cluster_monitor"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSH helper
# ---------------------------------------------------------------------------

def ssh_run(host: str, cmd: str, timeout: int = 60) -> str:
    """Run a command via SSH using password auth (same approach as remote_cluster.py)."""
    ssh_key = PROJECT_ROOT / "configs" / "password.txt"
    # Build sshpass-based command for non-interactive use
    if ssh_key.exists():
        password = ssh_key.read_text().strip()
        ssh_cmd = [
            "sshpass", "-p", password,
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "PubkeyAuthentication=no",
            f"root@{host}",
            cmd,
        ]
    else:
        # Fallback: rely on key-based auth
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            f"root@{host}",
            cmd,
        ]
    try:
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "SSHPASS": ssh_key.read_text().strip() if ssh_key.exists() else ""},
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"exit code {result.returncode}")
        return result.stdout.strip()
    except FileNotFoundError:
        # sshpass not installed — fall back to pexpect
        return _ssh_run_pexpect(host, cmd, timeout)


def _ssh_run_pexpect(host: str, cmd: str, timeout: int = 60) -> str:
    """Fallback SSH via pexpect (same as remote_cluster.py)."""
    import pexpect
    password_file = PROJECT_ROOT / "configs" / "password.txt"
    password = password_file.read_text().strip() if password_file.exists() else "4fS6h9nHdbICfm6,"

    child = pexpect.spawn(
        f'ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password root@{host}',
        timeout=timeout, encoding='utf-8', codec_errors='replace',
    )
    child.expect('assword:')
    child.sendline(password)
    time.sleep(3)
    child.sendline(cmd)
    time.sleep(2)
    child.sendline('echo __ENDOFOUTPUT__')
    child.expect('__ENDOFOUTPUT__', timeout=timeout - 10)
    output = child.before
    child.sendline('exit')
    child.close()
    return output.strip()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GPUStatus:
    index: int
    util_pct: float
    mem_used_mb: float
    mem_total_mb: float
    temperature: float = 0.0
    processes: list[str] = field(default_factory=list)


@dataclass
class NodeStatus:
    host: str
    name: str
    reachable: bool = False
    ssh_error: str = ""
    gpus: list[GPUStatus] = field(default_factory=list)
    expected_pid: Optional[int] = None
    expected_experiment: str = ""
    expected_output_dir: str = ""
    pid_alive: bool = False
    pid_cmdline: str = ""
    gpu_busy: bool = False
    anomaly: str = ""  # human-readable problem description
    needs_recovery: bool = False
    last_log_lines: list[str] = field(default_factory=list)
    latest_checkpoint: Optional[str] = None


# ---------------------------------------------------------------------------
# Monitoring functions
# ---------------------------------------------------------------------------

def check_ssh(host: str) -> tuple[bool, str]:
    """Quick SSH connectivity test."""
    try:
        out = ssh_run(host, "echo ok", timeout=15)
        return True, ""
    except Exception as e:
        return False, str(e)


def parse_nvidia_smi(host: str) -> list[GPUStatus]:
    """Parse nvidia-smi output from remote host."""
    out = ssh_run(host, "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits", timeout=30)
    gpus = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append(GPUStatus(
                index=int(parts[0]),
                util_pct=float(parts[1]),
                mem_used_mb=float(parts[2]),
                mem_total_mb=float(parts[3]),
                temperature=float(parts[4]),
            ))
    return gpus


def check_pid(host: str, pid: int) -> tuple[bool, str]:
    """Check if a PID is alive and get its cmdline."""
    try:
        out = ssh_run(host, f"ps -p {pid} -o pid,cmd --no-headers 2>/dev/null || echo DEAD", timeout=15)
        if "DEAD" in out:
            return False, ""
        return True, out.strip()
    except Exception:
        return False, ""


def check_gpu_processes(host: str) -> list[str]:
    """List GPU-bound processes (simplified nvidia-smi pmon)."""
    try:
        out = ssh_run(host, "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null || echo NONE", timeout=15)
        return [l.strip() for l in out.splitlines() if l.strip() and l.strip() != "NONE"]
    except Exception:
        return []


def check_training_progress(host: str, output_dir: str) -> tuple[list[str], Optional[str]]:
    """Grab last few lines of the latest log file and find latest checkpoint."""
    project = "/root/Mixture-of-Memory"
    try:
        # Find most recent log
        find_out = ssh_run(
            host,
            f"ls -t {project}/{output_dir}/*.log 2>/dev/null | head -1",
            timeout=15,
        )
        if not find_out.strip():
            # Try .txt or no extension
            find_out = ssh_run(
                host,
                f"ls -t {project}/{output_dir}/log* 2>/dev/null | head -1",
                timeout=15,
            )
        log_path = find_out.strip()
        if log_path and not log_path.startswith("ls:") and not log_path.startswith("No such"):
            tail = ssh_run(host, f"tail -20 {log_path}", timeout=15)
            lines = tail.splitlines()
        else:
            lines = []

        # Check checkpoints
        ckpt_out = ssh_run(
            host,
            f"ls -t {project}/{output_dir}/checkpoint* 2>/dev/null | head -3",
            timeout=15,
        )
        checkpoints = [l.strip() for l in ckpt_out.splitlines() if l.strip()]
        latest_ckpt = checkpoints[0] if checkpoints else None
        return lines, latest_ckpt
    except Exception as e:
        log.warning("Failed to check progress on %s: %s", host, e)
        return [], None


def assess_node(status: NodeStatus) -> None:
    """Set anomaly and needs_recovery flags."""
    if not status.reachable:
        status.anomaly = "UNREACHABLE"
        status.needs_recovery = False  # can't recover if unreachable
        return

    gpu_avg_util = sum(g.util_pct for g in status.gpus) / max(len(status.gpus), 1)
    gpu_total_mem = sum(g.mem_used_mb for g in status.gpus)
    gpu_processes = check_gpu_processes(status.host)

    status.gpu_busy = gpu_avg_util > 5 or gpu_total_mem > 1000

    # Case 1: Expected PID is dead
    if status.expected_pid and not status.pid_alive:
        if status.gpu_busy:
            status.anomaly = f"PID {status.expected_pid} DEAD but GPUs busy (orphaned process?)"
            status.needs_recovery = True
        else:
            status.anomaly = f"PID {status.expected_pid} DEAD, GPUs idle — WASTED"
            status.needs_recovery = True
        return

    # Case 2: PID alive but GPUs idle
    if status.expected_pid and status.pid_alive and not status.gpu_busy:
        status.anomaly = f"PID {status.expected_pid} alive but GPUs idle (stuck/failed training?)"
        # Don't auto-recover yet — could be loading data
        status.needs_recovery = False
        return

    # Case 3: No expected PID but GPUs busy (untracked experiment)
    if not status.expected_pid and status.gpu_busy:
        status.anomaly = f"GPUs busy but no expected PID in config (untracked experiment)"
        status.needs_recovery = False
        return

    # Case 4: Everything looks good
    if status.pid_alive and status.gpu_busy:
        status.anomaly = ""
        status.needs_recovery = False
        return

    # Case 5: GPUs idle, no PID — idle node
    status.anomaly = "Node idle (no expected PID, GPUs free)"
    status.needs_recovery = False


def kill_and_relaunch(node_status: NodeStatus, config: dict) -> bool:
    """Kill any stale GPU processes and relaunch the experiment."""
    host = node_status.host
    node_name = node_status.name

    log.warning("[%s] Starting recovery...", node_name)

    # Step 1: Kill stale processes using GPUs
    try:
        ssh_run(host, "nvidia-smi --query-compute-apps=pid --format=csv,noheader -x 2>/dev/null | grep -oP '\\d+' | xargs -r kill -9 2>/dev/null; sleep 2; echo CLEANED", timeout=30)
    except Exception as e:
        log.error("[%s] Failed to clean GPU processes: %s", node_name, e)
        return False

    # Step 2: Kill expected PID if it's in a zombie state
    if node_status.expected_pid:
        try:
            ssh_run(host, f"kill -9 {node_status.expected_pid} 2>/dev/null; echo KILLED", timeout=10)
        except Exception:
            pass

    # Step 3: Verify GPUs are free
    gpus = parse_nvidia_smi(host)
    any_busy = any(g.util_pct > 5 or g.mem_used_mb > 500 for g in gpus)
    if any_busy:
        log.error("[%s] GPUs still busy after cleanup — aborting recovery", node_name)
        return False

    # Step 4: Relaunch using dispatch_remote_training.py
    dispatch_script = SCRIPTS_DIR / "dispatch_remote_training.py"
    if not dispatch_script.exists():
        log.error("[%s] dispatch_remote_training.py not found at %s", node_name, dispatch_script)
        return False

    experiment = node_status.expected_experiment
    if not experiment:
        log.error("[%s] No experiment name in config — can't relaunch", node_name)
        return False

    log.info("[%s] Relaunching experiment: %s", node_name, experiment)

    try:
        result = subprocess.run(
            [sys.executable, str(dispatch_script), node_name, experiment],
            capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            log.error("[%s] Dispatch failed: %s", node_name, result.stderr)
            return False
        log.info("[%s] Dispatch output: %s", node_name, result.stdout.strip())
        return True
    except Exception as e:
        log.error("[%s] Dispatch exception: %s", node_name, e)
        return False


# ---------------------------------------------------------------------------
# Main check loop
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def run_check(config: dict, do_recover: bool = False) -> list[NodeStatus]:
    """Run a full check across all nodes. Optionally auto-recover."""
    results = []

    for host, node_cfg in config["nodes"].items():
        name = node_cfg.get("name", host)
        status = NodeStatus(
            host=host,
            name=name,
            expected_pid=node_cfg.get("pid"),
            expected_experiment=node_cfg.get("experiment", ""),
            expected_output_dir=node_cfg.get("output_dir", ""),
        )

        # SSH check
        reachable, ssh_err = check_ssh(host)
        status.reachable = reachable
        status.ssh_error = ssh_err

        if not reachable:
            assess_node(status)
            results.append(status)
            continue

        # GPU status
        try:
            status.gpus = parse_nvidia_smi(host)
        except Exception as e:
            status.anomaly = f"Failed to query GPUs: {e}"
            results.append(status)
            continue

        # PID check
        if status.expected_pid:
            alive, cmdline = check_pid(host, status.expected_pid)
            status.pid_alive = alive
            status.pid_cmdline = cmdline

        # Training progress
        if status.expected_output_dir:
            lines, ckpt = check_training_progress(host, status.expected_output_dir)
            status.last_log_lines = lines[-5:]  # keep last 5
            status.latest_checkpoint = ckpt

        # Assess
        assess_node(status)

        # Recovery
        if do_recover and status.needs_recovery:
            success = kill_and_relaunch(status, config)
            if success:
                log.info("[%s] ✅ Recovery successful — experiment relaunched", name)
            else:
                log.error("[%s] ❌ Recovery failed", name)

        results.append(status)

    return results


def print_report(results: list[NodeStatus]) -> None:
    """Print a human-readable summary."""
    print("\n" + "=" * 80)
    print(f"  CLUSTER STATUS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    total_gpus = 0
    used_gpus = 0
    problems = 0

    for s in results:
        total_gpus += len(s.gpus)
        gpu_busy = sum(1 for g in s.gpus if g.util_pct > 5 or g.mem_used_mb > 1000)
        used_gpus += gpu_busy

        icon = "🟢" if not s.anomaly else ("🔴" if s.needs_recovery else "🟡")
        print(f"\n{icon} {s.name} ({s.host})")
        if not s.reachable:
            print(f"   ❌ SSH UNREACHABLE: {s.ssh_error}")
            problems += 1
            continue

        if s.gpus:
            gpu_strs = [f"GPU{g.index}:{g.util_pct:.0f}%/{g.mem_used_mb:.0f}MB" for g in s.gpus[:4]]
            print(f"   GPUs: {' | '.join(gpu_strs)}")
            print(f"   Busy: {gpu_busy}/{len(s.gpus)}")

        if s.expected_pid:
            alive_str = "✅ alive" if s.pid_alive else "❌ dead"
            print(f"   PID {s.expected_pid}: {alive_str}")

        if s.expected_experiment:
            print(f"   Experiment: {s.expected_experiment}")

        if s.latest_checkpoint:
            print(f"   Latest checkpoint: {s.latest_checkpoint}")

        if s.last_log_lines:
            print(f"   Last log: {s.last_log_lines[-1][:100]}")

        if s.anomaly:
            print(f"   ⚠️  {s.anomaly}")
            if s.needs_recovery:
                problems += 1
        elif s.anomaly == "" and s.pid_alive:
            print("   Status: Running normally ✅")

    print(f"\n{'─' * 80}")
    print(f"  Summary: {used_gpus}/{total_gpus} GPUs active | {problems} nodes need attention")
    print(f"  Log: {log_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Monitor remote L20A training cluster")
    parser.add_argument("--check-only", action="store_true", help="Only check, no recovery")
    parser.add_argument("--recover", action="store_true", help="Auto-recover failed experiments")
    parser.add_argument("--loop", type=int, default=0, help="Loop interval in seconds (0 = one-shot)")
    parser.add_argument("--json-config", type=Path, default=DEFAULT_CONFIG, help="Path to remote_experiments.json")
    parser.add_argument("--alert-cmd", type=str, default="", help="Command to run on alert (e.g. send notification)")
    args = parser.parse_args()

    config = load_config(args.json_config)
    do_recover = args.recover and not args.check_only
    iteration = 0

    while True:
        iteration += 1
        log.info("Check iteration #%d (recover=%s)", iteration, do_recover)
        results = run_check(config, do_recover=do_recover)
        print_report(results)

        # Alert on problems
        problem_nodes = [s for s in results if s.anomaly and s.needs_recovery]
        if problem_nodes and args.alert_cmd:
            msg = f"⚠️ Cluster alert: {len(problem_nodes)} node(s) need attention: " + ", ".join(s.name for s in problem_nodes)
            try:
                subprocess.run(args.alert_cmd, shell=True, capture_output=True, timeout=10)
                log.info("Alert sent: %s", msg)
            except Exception as e:
                log.error("Alert failed: %s", e)

        if args.loop <= 0:
            break
        log.info("Sleeping %d seconds...", args.loop)
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
