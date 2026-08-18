#!/usr/bin/env python3
"""Inspect or advance the registered Scaffold-Coder remote heartbeat safely."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(
    "/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft"
)


def parse_access(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    patterns = {
        "host": r"^- Host: `([^`]*)`",
        "user": r"^- SSH user: `([^`]*)`",
        "password": r"^- Password: `([^`]*)`",
        "remote_root": r"^- Remote workspace:\s*\n\s*`([^`]*)`",
    }
    values: dict[str, str] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            raise ValueError(f"missing {name} in {path}")
        values[name] = match.group(1)
    return values


REMOTE_INSPECTOR = r'''
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
from pathlib import Path

root = Path(os.environ["SCAFFOLD_REMOTE_ROOT"])
trigger = os.environ.get("SCAFFOLD_TRIGGER") == "1"
ensure_daemon = os.environ.get("SCAFFOLD_ENSURE_DAEMON") == "1"
tail_lines = int(os.environ.get("SCAFFOLD_TAIL_LINES", "80"))

def run(command, timeout=30):
    result = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {"returncode": result.returncode, "output": result.stdout}

if ensure_daemon:
    tmux = run(["tmux", "has-session", "-t", "scaffold-heartbeat"])
    if tmux["returncode"] != 0:
        run([str(root / "ops" / "start_heartbeat.sh")])

heartbeat_result = None
if trigger:
    heartbeat_result = run([str(root / "ops" / "heartbeat.py")], timeout=60)

gpu_result = run([
    "nvidia-smi",
    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
    "temperature.gpu,power.draw",
    "--format=csv,noheader,nounits",
])
gpus = []
for row in csv.reader(gpu_result["output"].splitlines()):
    if len(row) != 7:
        continue
    values = [value.strip() for value in row]
    try:
        gpus.append({
            "index": int(values[0]),
            "name": values[1],
            "memory_used_mib": int(values[2]),
            "memory_total_mib": int(values[3]),
            "utilization_percent": int(values[4]),
            "temperature_c": int(values[5]),
            "power_w": float(values[6]),
        })
    except ValueError:
        pass

active_path = root / "ops" / "state" / "active_run.json"
active = json.loads(active_path.read_text()) if active_path.exists() else None
active_log_tail = ""
if active and active.get("log_path"):
    log_path = Path(active["log_path"])
    if log_path.exists():
        active_log_tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace")
            .splitlines()[-tail_lines:]
        )

queue_rows = []
queue_path = root / "ops" / "queue.tsv"
if queue_path.exists():
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        row = next(csv.reader([line], delimiter="\t"))
        if len(row) >= 3:
            queue_rows.append({
                "status": row[0],
                "id": row[1],
                "resource": row[2],
            })

history_path = root / "ops" / "history.tsv"
history_tail = ""
if history_path.exists():
    history_tail = "\n".join(
        history_path.read_text(encoding="utf-8", errors="replace")
        .splitlines()[-tail_lines:]
    )
heartbeat_path = root / "ops" / "logs" / "heartbeat.log"
heartbeat_tail = ""
if heartbeat_path.exists():
    heartbeat_tail = "\n".join(
        heartbeat_path.read_text(encoding="utf-8", errors="replace")
        .splitlines()[-tail_lines:]
    )

git_head = run(["git", "log", "-1", "--oneline"])
git_status = run(["git", "status", "--short", "--branch"])
tmux = run(["tmux", "ls"])
disk = shutil.disk_usage(root)
ready = [row for row in queue_rows if row["status"] == "READY"]
blocked = [
    row for row in queue_rows
    if row["status"] in {"BLOCKED", "NEEDS_DEBUG"}
]
gpu_idle = bool(gpus) and all(
    gpu["memory_used_mib"] <= 2048
    and gpu["utilization_percent"] <= 10
    for gpu in gpus
)

report = {
    "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    "hostname": os.uname().nodename,
    "remote_root": str(root),
    "triggered_heartbeat": trigger,
    "heartbeat_result": heartbeat_result,
    "heartbeat_tmux": tmux,
    "git_head": git_head["output"].strip(),
    "git_status": git_status["output"].strip(),
    "disk_free_gib": round(disk.free / 2**30, 2),
    "gpu_idle": gpu_idle,
    "gpus": gpus,
    "active_run": active,
    "active_log_tail": active_log_tail,
    "queue_counts": {
        "total": len(queue_rows),
        "ready": len(ready),
        "blocked": len(blocked),
        "done": sum(row["status"] == "DONE" for row in queue_rows),
    },
    "first_ready": ready[0] if ready else None,
    "blocked_items": blocked,
    "history_tail": history_tail,
    "heartbeat_tail": heartbeat_tail,
}
print(json.dumps(report, indent=2, sort_keys=True))
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--trigger",
        action="store_true",
        help="run one registered heartbeat before collecting the report",
    )
    parser.add_argument(
        "--ensure-daemon",
        action="store_true",
        help="start the remote tmux heartbeat if it is absent",
    )
    parser.add_argument("--tail-lines", type=int, default=80)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    access = parse_access(project_root / "SERVER_ACCESS.md")
    sshpass = shutil.which("sshpass")
    ssh = shutil.which("ssh")
    if not sshpass or not ssh:
        raise RuntimeError("ssh and sshpass are required")

    known_hosts = project_root / "ops" / "ssh" / "known_hosts"
    command = [
        sshpass,
        "-e",
        ssh,
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=15",
        f"{access['user']}@{access['host']}",
        (
            f"SCAFFOLD_REMOTE_ROOT={access['remote_root']!r} "
            f"SCAFFOLD_TRIGGER={'1' if args.trigger else '0'} "
            f"SCAFFOLD_ENSURE_DAEMON={'1' if args.ensure_daemon else '0'} "
            f"SCAFFOLD_TAIL_LINES={args.tail_lines} "
            "python3 -"
        ),
    ]
    environment = dict(os.environ)
    environment["SSHPASS"] = access["password"]
    result = subprocess.run(
        command,
        input=REMOTE_INSPECTOR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        timeout=args.timeout,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr, end="")
        return result.returncode

    report = json.loads(result.stdout)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
