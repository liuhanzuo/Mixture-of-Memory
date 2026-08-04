#!/usr/bin/env python3
"""GPU cluster monitor — collects nvidia-smi from local + .196 + B200, serves a
light-themed dashboard (real-time mem/util/power + history sparklines).

Run:  python monitor/gpu_monitor_server.py [--port 8088] [--interval 5]
Open: http://<this-host>:8088/
Stops with Ctrl-C. No external deps (stdlib only).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)

# --- node definitions ------------------------------------------------------
# Each node: how to run nvidia-smi. local runs directly; remote via sshpass.
QUERY = "index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit"
SMI = (
    "nvidia-smi --query-gpu=" + QUERY + " --format=csv,noheader,nounits"
)
# Running tasks: training AND offline eval/bench drivers. The roster of active
# scripts changed after the mem_space era (OLMo-2 prune-heal, QCMem distill,
# paperC graft-FT, patching/eval harnesses), so match the current families too
# or the frontend's task panel silently reports "idle" on a busy node.
TRAIN_PAT = (
    'scripts/train_mem_space|scripts/train_olmo2|scripts/train_qcmem'
    '|scripts/train_shortgpt|scripts/eval_|scripts/bench_|scripts/patch_olmo2'
    '|scripts/probe_linguistic|_run_p0_20|_run_paperB|_run_paperC|paperC_pc1'
)
PS_CMD = (
    "ps -eo pid,etimes,cmd | grep -E '" + TRAIN_PAT + "' "
    "| grep -v grep || true"
)
# Latest training metrics: for each train proc, follow its stdout (fd/1) to the
# log file and grab the last `INFO - [step ...]` line. Dedup by log path so each
# distinct run reports once. Output lines: "<logpath>@@<metric line>".
METRIC_CMD = (
    "for pid in $(pgrep -f 'scripts/train_mem_space|scripts/train_olmo2|scripts/train_qcmem' 2>/dev/null); do "
    "log=$(readlink /proc/$pid/fd/1 2>/dev/null); "
    "[ -f \"$log\" ] || continue; "
    "line=$(grep -aE 'INFO - \\[step|^\\[step' \"$log\" 2>/dev/null | tail -1); "
    "[ -n \"$line\" ] && echo \"${log}@@${line}\"; "
    "done 2>/dev/null | sort -u || true"
)
# Full training curves: read directly from the run log every poll and cache them
# to JSON so frontend curves survive monitor restarts and do not depend on
# in-memory polling cadence. Output format is "<logpath>@@<metric line>" for
# up to the last METRIC_HISTORY_LEN training steps per active run.
CURVE_CMD = (
    "for pid in $(pgrep -f 'scripts/train_mem_space|scripts/train_olmo2|scripts/train_qcmem' 2>/dev/null); do "
    "log=$(readlink /proc/$pid/fd/1 2>/dev/null); "
    "[ -f \"$log\" ] || continue; "
    "grep -aE 'INFO - \\[step|^\\[step' \"$log\" 2>/dev/null | tail -1000 | sed \"s#^#${log}@@#\"; "
    "done 2>/dev/null || true"
)

NODES = [
    {"id": "local", "label": "本机 LOCAL (wzc1, 8xL20A/B200级 183GB)", "mode": "local"},
    {
        "id": "252",
        "label": ".252 (28.89.19.252, wzc1, 8xB200)",
        "mode": "ssh",
        "host": "28.89.19.252",
        "port": "22",
        "pwfile": os.path.join(PROJECT, "configs/password_b200_19252.txt"),
    },
    {
        "id": "73",
        "label": ".73 (28.85.35.73:36000, zwfy6, 8xH20)",
        "mode": "ssh",
        "host": "28.85.35.73",
        "port": "36000",
        "pwfile": os.path.join(PROJECT, "configs/password_h20_853573.txt"),
    },
    {
        "id": "82",
        "label": ".82 (28.82.250.82:36000, zwfy6, 8xH20)",
        "mode": "ssh",
        "host": "28.82.250.82",
        "port": "36000",
        "pwfile": os.path.join(PROJECT, "configs/password_h20_82250.txt"),
    },
    {
        "id": "104",
        "label": ".104 (28.83.24.104:36000, zwfy6, 8xH20)",
        "mode": "ssh",
        "host": "28.83.24.104",
        "port": "36000",
        "pwfile": os.path.join(PROJECT, "configs/password_h20_24104.txt"),
    },
]

HISTORY_LEN = 720  # ~1h at 5s interval
# node_id -> deque of {t, total_mem_used, total_mem, avg_util, total_power}
history: dict[str, deque] = {n["id"]: deque(maxlen=HISTORY_LEN) for n in NODES}
# node_id -> latest per-gpu snapshot
latest: dict[str, dict] = {n["id"]: {} for n in NODES}
# node_id -> list of running training task dicts
tasks: dict[str, list] = {n["id"]: [] for n in NODES}
# node_id -> list of latest training-metric dicts (per run)
metrics: dict[str, list] = {n["id"]: [] for n in NODES}
# run_name -> deque of {t, step, lm, t2_needle, distill_kl, distill_hid, ...}
# keyed by run name (not node) so a run that migrates nodes keeps its curve.
METRIC_HISTORY_LEN = 1000
CURVE_CACHE = os.path.join(PROJECT, "status", "GPU_TRAINING_CURVES.json")
metric_history: dict[str, deque] = {}
lock = threading.Lock()


def run_cmd(node: dict, cmd_str: str, timeout: int = 25) -> str | None:
    """Run a shell command locally or over ssh; return stdout or None."""
    try:
        if node["mode"] == "local":
            out = subprocess.run(
                cmd_str, shell=True, capture_output=True, text=True, timeout=timeout
            )
        else:
            cmd = ["sshpass", "-f", node["pwfile"], "ssh"]
            if node.get("port") and str(node["port"]) != "22":
                cmd += ["-p", str(node["port"])]
            cmd += [
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=12",
                "-o", "PreferredAuthentications=password",
                f"root@{node['host']}", cmd_str,
            ]
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def run_smi(node: dict, timeout: int = 25) -> str | None:
    return run_cmd(node, SMI, timeout)


def _fmt_elapsed(secs: int) -> str:
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, _ = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m"


def _extract_arg(cmd: str, flag: str) -> str:
    """Return token following `flag` in a command string, or ''."""
    toks = cmd.split()
    if flag in toks:
        i = toks.index(flag)
        if i + 1 < len(toks):
            return toks[i + 1]
    return ""


def parse_tasks(text: str) -> list[dict]:
    """Parse `ps -eo pid,etimes,cmd` lines into compact training-task dicts.
    Only keep the torchrun parent (has --nproc or master_port) to avoid
    listing every rank worker."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, etimes, cmd = parts
        try:
            etimes_i = int(etimes)
        except ValueError:
            continue
        # one entry per run: prefer the launcher / rank0 (avoid N duplicates)
        run_name = _extract_arg(cmd, "--wandb_run_name")
        out_dir = _extract_arg(cmd, "--output_dir")
        name = run_name or (out_dir.rstrip("/").split("/")[-1] if out_dir else "")
        if not name:
            continue
        out.append({
            "pid": pid,
            "name": name,
            "out_dir": out_dir,
            "elapsed": _fmt_elapsed(etimes_i),
            "etimes": etimes_i,
        })
    # dedup by run name, keep the one with largest elapsed (the parent)
    best: dict[str, dict] = {}
    for t in out:
        k = t["name"]
        if k not in best or t["etimes"] > best[k]["etimes"]:
            best[k] = t
    return sorted(best.values(), key=lambda x: -x["etimes"])


# metrics we surface on the dashboard, in display order
METRIC_KEYS = [
    "step", "lm", "t2_needle", "distill_kl", "distill_hid",
    "aux", "lr", "nf", "speed",
]


def _parse_metric_line(logpath: str, line: str) -> dict | None:
    """Parse one train log metric line into a metric dict."""
    run = logpath.rstrip("/").split("/")[-1].replace(".log", "")
    m = {"run": run, "log": logpath}
    sm = re.search(r"\[step\s+(\d+)\s*/\s*(\d+)\]", line)
    if not sm:
        return None
    m["step"] = int(sm.group(1))
    m["total_steps"] = int(sm.group(2))
    # key=value pairs (value may be float / int / -1.0 sentinel)
    for k, v in re.findall(r"(\w+)=(-?\d+\.?\d*(?:e[-+]?\d+)?)", line):
        if k in ("step",):
            continue
        try:
            m[k] = float(v) if ("." in v or "e" in v) else int(v)
        except ValueError:
            pass
    return m


def parse_metrics(text: str) -> list[dict]:
    """Parse METRIC_CMD output ("<logpath>@@<line>") into latest per-run metrics.

    A train log line looks like:
      ... INFO - [step 115/2000] lm=0.0000 t2_needle=0.0000 aux=0.0000 ... nf=0 ...
    """
    runs = []
    seen_logs = set()
    for raw in text.splitlines():
        if "@@" not in raw:
            continue
        logpath, line = raw.split("@@", 1)
        if logpath in seen_logs:
            continue
        seen_logs.add(logpath)
        m = _parse_metric_line(logpath, line)
        if m is not None:
            runs.append(m)
    return runs


def parse_log_curves(text: str) -> dict[str, list[dict]]:
    """Parse CURVE_CMD output into full per-run metric curves from logs."""
    by_run: dict[str, dict[int, dict]] = {}
    for raw in text.splitlines():
        if "@@" not in raw:
            continue
        logpath, line = raw.split("@@", 1)
        m = _parse_metric_line(logpath, line)
        if m is None:
            continue
        run = m["run"]
        pt = {"step": m["step"]}
        for k in ("lm", "t2_needle", "distill_kl", "distill_hid", "aux", "lr", "speed", "nf"):
            if k in m:
                pt[k] = m[k]
        by_run.setdefault(run, {})[m["step"]] = pt
    return {
        run: [points[s] for s in sorted(points)[-METRIC_HISTORY_LEN:]]
        for run, points in by_run.items()
    }


def load_curve_cache() -> None:
    """Load persisted curves from JSON cache at monitor startup."""
    try:
        with open(CURVE_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    for run, pts in data.items():
        if isinstance(pts, list):
            metric_history[run] = deque(pts[-METRIC_HISTORY_LEN:], maxlen=METRIC_HISTORY_LEN)


def save_curve_cache() -> None:
    """Persist current curves for frontend/restarter continuity."""
    try:
        os.makedirs(os.path.dirname(CURVE_CACHE), exist_ok=True)
        tmp = CURVE_CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: list(v) for k, v in metric_history.items()}, f)
        os.replace(tmp, CURVE_CACHE)
    except Exception:
        pass


def parse_smi(text: str) -> list[dict]:
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "mem_used": float(parts[2]),
                "mem_total": float(parts[3]),
                "util": float(parts[4]),
                "power": float(parts[5]) if parts[5] not in ("[N/A]", "") else 0.0,
                "power_limit": float(parts[6]) if parts[6] not in ("[N/A]", "") else 0.0,
            })
        except ValueError:
            continue
    return gpus


def poll_loop(interval: int):
    while True:
        for node in NODES:
            text = run_smi(node)
            ts = time.time()
            if text is None:
                with lock:
                    latest[node["id"]] = {"ok": False, "t": ts, "gpus": []}
                    tasks[node["id"]] = []
                    metrics[node["id"]] = []
                continue
            gpus = parse_smi(text)
            tot_used = sum(g["mem_used"] for g in gpus)
            tot_mem = sum(g["mem_total"] for g in gpus)
            avg_util = (sum(g["util"] for g in gpus) / len(gpus)) if gpus else 0.0
            tot_pow = sum(g["power"] for g in gpus)
            # running training tasks (best-effort; don't fail the node on error)
            ps_text = run_cmd(node, PS_CMD, timeout=20)
            node_tasks = parse_tasks(ps_text) if ps_text else []
            # latest training metrics + full training curves read directly from
            # each active run's log file (not from the monitor's polling history).
            mt_text = run_cmd(node, METRIC_CMD, timeout=20)
            node_metrics = parse_metrics(mt_text) if mt_text else []
            curve_text = run_cmd(node, CURVE_CMD, timeout=25)
            node_curves = parse_log_curves(curve_text) if curve_text else {}
            with lock:
                latest[node["id"]] = {"ok": True, "t": ts, "gpus": gpus}
                tasks[node["id"]] = node_tasks
                metrics[node["id"]] = node_metrics
                # Replace per-run curves from the underlying log so the frontend
                # displays the actual run history and survives monitor restarts.
                for run, pts in node_curves.items():
                    metric_history[run] = deque(pts, maxlen=METRIC_HISTORY_LEN)
                if node_curves:
                    save_curve_cache()
                history[node["id"]].append({
                    "t": ts,
                    "mem_used": round(tot_used),
                    "mem_total": round(tot_mem),
                    "util": round(avg_util, 1),
                    "power": round(tot_pow, 1),
                })
        time.sleep(interval)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            with open(os.path.join(ROOT, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path.startswith("/api/data"):
            with lock:
                payload = {
                    "nodes": [{"id": n["id"], "label": n["label"]} for n in NODES],
                    "latest": latest,
                    "tasks": tasks,
                    "metrics": metrics,
                    "metric_history": {k: list(v) for k, v in metric_history.items()},
                    "history": {k: list(v) for k, v in history.items()},
                    "server_time": time.time(),
                }
            self._send(200, json.dumps(payload))
        else:
            self._send(404, "{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--interval", type=int, default=5)
    args = ap.parse_args()

    load_curve_cache()
    t = threading.Thread(target=poll_loop, args=(args.interval,), daemon=True)
    t.start()
    ThreadingHTTPServer.allow_reuse_address = True
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"GPU monitor serving on http://0.0.0.0:{args.port}/  (poll {args.interval}s)",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
