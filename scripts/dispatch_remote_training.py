#!/usr/bin/env python3
"""Dispatch RMT v10 training across 4 remote L20A nodes."""

import pexpect
import time
import sys
import json
from datetime import datetime

NODES = {
    "28.89.17.143": "l0",
    "28.89.17.144": "l0l1",
    "28.89.17.85":  "l0l2",
    "28.89.19.134": "l0l1l2",
}

SSH_OPTS = "-o PubkeyAuthentication=no -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
PASSWORD = "4fS6h9nHdbICfm6,"
REMOTE_WORKDIR = "/root/Mixture-of-Memory"
LOG_FILE = f"dispatch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def ssh_cmd(ip, cmd, timeout=30):
    """Run a command on remote node via pexpect, return output."""
    full_cmd = f"ssh {SSH_OPTS} root@{ip} '{cmd}'"
    child = pexpect.spawn(full_cmd, timeout=timeout, encoding="utf-8")
    child.expect("password:", timeout=15)
    child.sendline(PASSWORD)
    child.expect(pexpect.EOF, timeout=timeout)
    return child.before


def start_training(ip, config):
    """Start training on a remote node. Returns PID."""
    # Kill any existing training on this node
    print(f"[{ip}] Cleaning up previous runs...")
    ssh_cmd(ip, f"pkill -f train_rmt_v10.py; sleep 1; echo done", timeout=15)

    # Launch training in background with nohup
    remote_cmd = (
        f"cd {REMOTE_WORKDIR} && "
        f"nohup bash scripts/remote_train_v10.sh --config {config} "
        f"> logs/train_{config}.log 2>&1 & "
        f"echo $!"
    )
    print(f"[{ip}] Starting config={config}...")
    output = ssh_cmd(ip, remote_cmd, timeout=20)
    pid = output.strip().split("\n")[-1].strip()
    print(f"[{ip}] PID={pid}, config={config}")
    return pid


def verify_training(ip, config, pid, wait=15):
    """Check GPU utilization to confirm training started."""
    print(f"[{ip}] Waiting {wait}s for training to initialize...")
    time.sleep(wait)

    # Check if process still alive
    output = ssh_cmd(ip, f"ps -p {pid} -o pid,comm --no-headers", timeout=10)
    if not output.strip():
        print(f"[{ip}] ⚠ PID {pid} not found — training may have crashed. Check logs/train_{config}.log")
        return False

    # Check GPU usage
    gpu_output = ssh_cmd(ip, "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader", timeout=15)
    print(f"[{ip}] GPU status:\n{gpu_output}")
    return True


def main():
    results = {}
    print(f"=== Dispatching RMT v10 training to {len(NODES)} nodes ===")
    print(f"Log: {LOG_FILE}\n")

    for ip, config in NODES.items():
        try:
            pid = start_training(ip, config)
            results[ip] = {"config": config, "pid": pid, "status": "started"}
        except Exception as e:
            print(f"[{ip}] ❌ Failed: {e}")
            results[ip] = {"config": config, "pid": None, "status": f"error: {e}"}

    print("\n=== Verifying training started ===\n")
    for ip, info in results.items():
        if info["pid"]:
            ok = verify_training(ip, info["config"], info["pid"])
            info["status"] = "running" if ok else "crashed"

    # Summary
    print("\n=== Summary ===")
    for ip, info in results.items():
        print(f"  {ip}  config={info['config']}  pid={info['pid']}  status={info['status']}")

    # Save log
    with open(LOG_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetails saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
