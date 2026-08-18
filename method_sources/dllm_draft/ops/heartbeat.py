#!/usr/bin/env python3
"""Auditable 30-minute watchdog for Scaffold-Coder experiments.

Only runs registered in ``ops/state/active_run.json`` may be terminated or
restarted. Runs listed under ``external_runs`` in config.json are read-only.
"""

from __future__ import annotations

import base64
import csv
import datetime as dt
import fcntl
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
STATE_DIR = OPS / "state"
LOG_DIR = OPS / "logs"
SNAPSHOT_DIR = OPS / "snapshots"
CONTROL_DIR = OPS / "control"
CONFIG_PATH = OPS / "config.json"
QUEUE_PATH = OPS / "queue.tsv"
HISTORY_PATH = OPS / "history.tsv"
ACTIVE_PATH = STATE_DIR / "active_run.json"
HEARTBEAT_LOG = LOG_DIR / "heartbeat.log"

ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)"
    r"|CUDA out of memory"
    r"|OutOfMemoryError"
    r"|NCCL[^\n]*(?:error|failed|timeout)"
    r"|No space left on device"
    r"|Segmentation fault"
    r"|(?:^|\s)Killed(?:\s|$)"
    r"|loss\s*=\s*(?:nan|inf)\b",
    re.IGNORECASE | re.MULTILINE,
)
STEP_RE = re.compile(r"\[step\s+(\d+)\s*/\s*(\d+)\]")
TRAIN_STEP_RE = re.compile(
    r"step:(\d+)\s+-\s+train/loss:([0-9.eE+-]+)"
)
TOTAL_TRAIN_STEPS_RE = re.compile(r"Total training steps:\s*(\d+)")
EPOCH_PROGRESS_RE = re.compile(r"Epoch\s+(\d+)/(\d+):\s+(\d+)%")
RESUME_CHECKPOINT_RE = re.compile(
    r"Resuming [^\n]* from ([^\n]+/global_step_\d+)"
)
CHECKPOINT_STEP_RE = re.compile(r"global_step_(\d+)$")
CHECKPOINT_REQUIRED_FILES = (
    "config.json",
    "configuration_dream.py",
    "modeling_dream.py",
    "tokenization_dream.py",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "optimizer_state.pt",
    "training_state.pt",
)
PROGRESS_MILESTONES = (25, 50, 75)
TRAINING_CHECKPOINT_POINTERS = {
    "SCAFFOLD-SFT-STAGE1-8GPU-001": (
        OPS / "artifacts" / "scaffold_stage1_latest_checkpoint.txt"
    ),
    "SCHEDULE-ONLY-SFT-STAGE1-001": (
        OPS / "artifacts" / "schedule_only_stage1_latest_checkpoint.txt"
    ),
    "PLAIN-SFT-STAGE1-001": (
        OPS / "artifacts" / "plain_stage1_latest_checkpoint.txt"
    ),
    "RECOVERY-BASE-PLAIN-1EP-TRAIN-001": (
        OPS / "artifacts" / "recovery_base_plain_1ep_checkpoint.txt"
    ),
    "RECOVERY-INSTRUCT-PLAIN-1EP-TRAIN-001": (
        OPS / "artifacts" / "recovery_instruct_plain_1ep_checkpoint.txt"
    ),
    "RECOVERY-INSTRUCT-HIGHNOISE-1EP-TRAIN-001": (
        OPS / "artifacts"
        / "recovery_instruct_highnoise_1ep_checkpoint.txt"
    ),
}


def tracks_training_checkpoint(run_id: str) -> bool:
    return run_id in TRAINING_CHECKPOINT_POINTERS


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).astimezone()


def stamp() -> str:
    return now().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for path in (STATE_DIR, LOG_DIR, SNAPSHOT_DIR, CONTROL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def history(status: str, run_id: str, message: str) -> None:
    clean = message.replace("\t", " ").replace("\n", " ")
    append_line(HISTORY_PATH, f"{stamp()}\t{status}\t{run_id}\t{clean}")


def log(message: str) -> None:
    line = f"{stamp()} {message}"
    append_line(HEARTBEAT_LOG, line)
    print(line, flush=True)


def run_command(
    command: list[str], *, timeout: int = 30, cwd: Path | None = None
) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return result.returncode, result.stdout
    except Exception as exc:  # watchdog must record failures, not crash silently
        return 255, f"{type(exc).__name__}: {exc}"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def tail_text(path: Path, max_bytes: int = 256_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def extract_training_progress(
    text: str,
    *,
    known_total_steps: int | None = None,
) -> dict[str, Any] | None:
    step_matches = TRAIN_STEP_RE.findall(text)
    if not step_matches:
        return None
    step, loss = step_matches[-1]
    total_matches = TOTAL_TRAIN_STEPS_RE.findall(text)
    total = (
        int(total_matches[-1])
        if total_matches
        else known_total_steps
    )
    epoch_matches = EPOCH_PROGRESS_RE.findall(text)
    result: dict[str, Any] = {
        "step": int(step),
        "loss": float(loss),
    }
    if total:
        result["total_steps"] = int(total)
        result["fraction"] = min(1.0, (int(step) + 1) / int(total))
    if epoch_matches:
        epoch, epochs, percent = epoch_matches[-1]
        result.update(
            {
                "epoch": int(epoch),
                "epochs": int(epochs),
                "epoch_percent": int(percent),
            }
        )
    return result


def pending_progress_milestones(
    fraction: float,
    recorded: list[int] | tuple[int, ...],
) -> list[int]:
    seen = {int(value) for value in recorded}
    percent = 100 * fraction
    return [
        milestone
        for milestone in PROGRESS_MILESTONES
        if milestone <= percent and milestone not in seen
    ]


def latest_complete_checkpoint(
    outputs_root: Path,
    *,
    launched_at: str | None,
    explicit_paths: tuple[Path, ...] = (),
) -> dict[str, Any] | None:
    """Find a resumable checkpoint created by, or named in, this launch."""

    launch_timestamp: float | None = None
    if launched_at:
        try:
            launch_timestamp = dt.datetime.fromisoformat(
                launched_at
            ).timestamp()
        except ValueError:
            pass
    explicit = {path.resolve() for path in explicit_paths}
    candidates = explicit | {
        path.resolve()
        for path in outputs_root.glob("*/global_step_*")
        if path.is_dir()
    }

    complete: list[tuple[float, int, Path, dict[str, Any]]] = []
    for path in candidates:
        integrity = checkpoint_integrity(path)
        if integrity is None:
            continue
        modified = float(integrity["mtime_timestamp"])
        if (
            path not in explicit
            and launch_timestamp is not None
            and modified < launch_timestamp - 60
        ):
            continue
        complete.append(
            (
                modified,
                int(integrity["step"]),
                path,
                integrity,
            )
        )
    if not complete:
        return None
    modified, step, path, integrity = max(complete)
    return {
        "path": str(path.resolve()),
        "step": step,
        "mtime": dt.datetime.fromtimestamp(
            modified,
            tz=dt.timezone.utc,
        ).astimezone().isoformat(timespec="seconds"),
        "integrity": "complete",
        "model_shards": integrity["model_shards"],
        "model_bytes": integrity["model_bytes"],
        "optimizer_bytes": integrity["optimizer_bytes"],
        "training_state_bytes": integrity["training_state_bytes"],
        "checkpoint_bytes": integrity["checkpoint_bytes"],
    }


def checkpoint_integrity(path: Path) -> dict[str, Any] | None:
    """Validate the lightweight on-disk invariants needed for resume."""

    match = CHECKPOINT_STEP_RE.search(path.name)
    if match is None:
        return None
    required = [path / name for name in CHECKPOINT_REQUIRED_FILES]
    if not all(file.is_file() and file.stat().st_size > 0 for file in required):
        return None
    try:
        index = json.loads(
            (path / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            return None
        shard_names = sorted(set(weight_map.values()))
        if not shard_names or not all(
            isinstance(name, str) and name.endswith(".safetensors")
            for name in shard_names
        ):
            return None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    root = path.resolve()
    shards: list[Path] = []
    for name in shard_names:
        shard = (path / name).resolve()
        if shard.parent != root:
            return None
        if not shard.is_file() or shard.stat().st_size <= 0:
            return None
        shards.append(shard)
    files = required + shards
    modified = max(file.stat().st_mtime for file in files)
    model_bytes = sum(shard.stat().st_size for shard in shards)
    optimizer_bytes = (path / "optimizer_state.pt").stat().st_size
    training_state_bytes = (path / "training_state.pt").stat().st_size
    return {
        "step": int(match.group(1)),
        "mtime_timestamp": modified,
        "model_shards": len(shards),
        "model_bytes": model_bytes,
        "optimizer_bytes": optimizer_bytes,
        "training_state_bytes": training_state_bytes,
        "checkpoint_bytes": sum(file.stat().st_size for file in files),
    }


def validate_training_success(
    pointer_path: Path,
    *,
    expected_total_steps: int | None,
) -> dict[str, Any]:
    if not pointer_path.is_file():
        return {
            "ok": False,
            "reason": f"missing checkpoint pointer {pointer_path}",
        }
    raw = pointer_path.read_text(encoding="utf-8").strip()
    if not raw:
        return {
            "ok": False,
            "reason": f"empty checkpoint pointer {pointer_path}",
        }
    checkpoint = Path(raw)
    integrity = checkpoint_integrity(checkpoint)
    if integrity is None:
        return {
            "ok": False,
            "reason": f"incomplete checkpoint {checkpoint}",
        }
    step = int(integrity["step"])
    if expected_total_steps is not None and step < expected_total_steps:
        return {
            "ok": False,
            "reason": (
                f"checkpoint step {step} is below expected "
                f"{expected_total_steps}"
            ),
            "checkpoint": str(checkpoint),
            "integrity": integrity,
        }
    return {
        "ok": True,
        "checkpoint": str(checkpoint.resolve()),
        "integrity": integrity,
    }


def collect_gpu_snapshot() -> dict[str, Any]:
    query = (
        "index,uuid,name,temperature.gpu,power.draw,power.limit,"
        "memory.total,memory.used,memory.free,utilization.gpu,utilization.memory"
    )
    code, output = run_command(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, Any]] = []
    if code == 0:
        for row in csv.reader(output.splitlines()):
            if len(row) != 11:
                continue
            values = [value.strip() for value in row]
            try:
                gpus.append(
                    {
                        "index": int(values[0]),
                        "uuid": values[1],
                        "name": values[2],
                        "temperature_c": int(values[3]),
                        "power_w": float(values[4]),
                        "power_limit_w": float(values[5]),
                        "memory_total_mib": int(values[6]),
                        "memory_used_mib": int(values[7]),
                        "memory_free_mib": int(values[8]),
                        "utilization_gpu_percent": int(values[9]),
                        "utilization_memory_percent": int(values[10]),
                    }
                )
            except ValueError:
                continue
    return {"returncode": code, "raw": output, "gpus": gpus}


def filesystem_snapshot() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    return {
        "path": str(ROOT),
        "total_gib": round(usage.total / 2**30, 2),
        "used_gib": round(usage.used / 2**30, 2),
        "free_gib": round(usage.free / 2**30, 2),
    }


def inspect_run(spec: dict[str, Any]) -> dict[str, Any]:
    pid = int(spec.get("pid", 0) or 0)
    log_path = Path(spec["log_path"]) if spec.get("log_path") else None
    success_path = spec.get("success_path")
    text = tail_text(log_path) if log_path else ""
    steps = STEP_RE.findall(text)
    progress = None
    if steps:
        current, total = map(int, steps[-1])
        progress = {
            "current": current,
            "total": total,
            "fraction": round(current / total, 6) if total else None,
        }
    stat = None
    if log_path and log_path.exists():
        info = log_path.stat()
        stat = {
            "path": str(log_path),
            "size": info.st_size,
            "mtime": dt.datetime.fromtimestamp(
                info.st_mtime, tz=dt.timezone.utc
            ).astimezone().isoformat(timespec="seconds"),
            "age_seconds": round(time.time() - info.st_mtime, 1),
        }
    return {
        "id": spec.get("id", "unknown"),
        "pid": pid,
        "alive": pid_alive(pid) if pid else False,
        "read_only": bool(spec.get("read_only", False)),
        "success": bool(success_path and glob.glob(success_path)),
        "progress": progress,
        "log": stat,
        "recent_error_signature": bool(ERROR_RE.search(text)),
    }


def gpus_idle(snapshot: dict[str, Any], config: dict[str, Any]) -> bool:
    gpus = snapshot["gpus"]
    if len(gpus) != int(config["expected_gpu_count"]):
        return False
    memory_limit = int(config["gpu_idle_memory_threshold_mib"])
    util_limit = int(config["gpu_idle_utilization_threshold_percent"])
    return all(
        gpu["memory_used_mib"] <= memory_limit
        and gpu["utilization_gpu_percent"] <= util_limit
        for gpu in gpus
    )


def load_queue() -> tuple[list[str], list[list[str]]]:
    comments: list[str] = []
    rows: list[list[str]] = []
    if not QUEUE_PATH.exists():
        return comments, rows
    for line in QUEUE_PATH.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            comments.append(line)
            continue
        row = next(csv.reader([line], delimiter="\t"))
        if len(row) != 7:
            log(f"QUEUE_INVALID expected=7_fields row={row!r}")
            continue
        rows.append(row)
    return comments, rows


def save_queue(comments: list[str], rows: list[list[str]]) -> None:
    temporary = QUEUE_PATH.with_suffix(".tsv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for line in comments:
            handle.write(line + "\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)
    os.replace(temporary, QUEUE_PATH)


def set_queue_status(run_id: str, status: str) -> None:
    comments, rows = load_queue()
    changed = False
    for row in rows:
        if row[1] == run_id:
            row[0] = status
            changed = True
            break
    if changed:
        save_queue(comments, rows)


def launch_run(
    *,
    run_id: str,
    resource: str,
    max_retries: int,
    cwd: str,
    success_path: str,
    command: str,
    retries: int,
) -> dict[str, Any]:
    workdir = Path(cwd)
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{run_id}.log"
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    log_handle.write(
        f"\n===== launch {stamp()} retry={retries}/{max_retries} =====\n"
    )
    process = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=workdir,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )
    state = {
        "id": run_id,
        "resource": resource,
        "max_retries": max_retries,
        "retries": retries,
        "cwd": str(workdir),
        "success_path": success_path,
        "command": command,
        "log_path": str(log_path),
        "pid": process.pid,
        "process_group": process.pid,
        "launched_at": stamp(),
        "last_observed_log_size": log_path.stat().st_size,
        "consecutive_stall_checks": 0,
    }
    atomic_json(ACTIVE_PATH, state)
    history("LAUNCHED", run_id, f"pid={process.pid} resource={resource}")
    log(f"RUN_LAUNCHED id={run_id} pid={process.pid} resource={resource}")
    return state


def archive_active(state: dict[str, Any], status: str) -> None:
    archive = STATE_DIR / f"{state['id']}.{status.lower()}.{int(time.time())}.json"
    atomic_json(archive, state)
    try:
        ACTIVE_PATH.unlink()
    except FileNotFoundError:
        pass


def restart_active(state: dict[str, Any], reason: str) -> None:
    retries = int(state.get("retries", 0)) + 1
    max_retries = int(state.get("max_retries", 0))
    if retries > max_retries:
        state["status"] = "NEEDS_DEBUG"
        state["failure_reason"] = reason
        state["updated_at"] = stamp()
        atomic_json(ACTIVE_PATH, state)
        set_queue_status(state["id"], "BLOCKED")
        history("NEEDS_DEBUG", state["id"], reason)
        log(f"RUN_NEEDS_DEBUG id={state['id']} reason={reason}")
        return
    archive_active(state, "retry")
    history("RETRYING", state["id"], f"retry={retries} reason={reason}")
    launch_run(
        run_id=state["id"],
        resource=state["resource"],
        max_retries=max_retries,
        cwd=state["cwd"],
        success_path=state.get("success_path", ""),
        command=state["command"],
        retries=retries,
    )


def inspect_active(
    config: dict[str, Any],
    gpu_snapshot: dict[str, Any] | None = None,
) -> bool:
    """Return True when an active run remains after inspection."""
    state = load_json(ACTIVE_PATH, None)
    if not state:
        return False
    run_id = state["id"]
    if state.get("status") == "NEEDS_DEBUG":
        log(f"RUN_BLOCKED_NEEDS_DEBUG id={run_id}")
        return True

    success_path = state.get("success_path", "")
    if success_path and glob.glob(success_path):
        pointer_path = TRAINING_CHECKPOINT_POINTERS.get(run_id)
        if pointer_path is not None:
            expected_total = (
                int(state["progress"]["total_steps"])
                if state.get("progress", {}).get("total_steps")
                else None
            )
            validation = validate_training_success(
                pointer_path,
                expected_total_steps=expected_total,
            )
            state["success_validation"] = validation
            if not validation["ok"]:
                reason = str(validation["reason"])
                state["status"] = "NEEDS_DEBUG"
                state["failure_reason"] = reason
                state["updated_at"] = stamp()
                atomic_json(ACTIVE_PATH, state)
                set_queue_status(run_id, "BLOCKED")
                history("NEEDS_DEBUG", run_id, reason)
                log(f"RUN_SUCCESS_VALIDATION_FAILED id={run_id} reason={reason}")
                return True
        state["status"] = "COMPLETED"
        state["completed_at"] = stamp()
        archive_active(state, "completed")
        set_queue_status(run_id, "DONE")
        history("COMPLETED", run_id, f"success_path={success_path}")
        log(f"RUN_COMPLETED id={run_id}")
        return False

    pid = int(state["pid"])
    if not pid_alive(pid):
        restart_active(state, "registered process exited without success artifact")
        return ACTIVE_PATH.exists()

    log_path = Path(state["log_path"])
    recent = tail_text(log_path)
    previous_progress = state.get("progress") or {}
    progress = extract_training_progress(
        recent,
        known_total_steps=previous_progress.get("total_steps"),
    )
    if progress is not None:
        fraction = float(progress.get("fraction", 0.0))
        launched_at = state.get("launched_at")
        if 0 < fraction < 1 and launched_at:
            try:
                elapsed = (
                    now() - dt.datetime.fromisoformat(launched_at)
                ).total_seconds()
                eta = elapsed * (1 - fraction) / fraction
                progress["elapsed_seconds"] = round(elapsed, 1)
                progress["eta_seconds"] = round(eta, 1)
                progress["estimated_completion"] = (
                    now() + dt.timedelta(seconds=eta)
                ).isoformat(timespec="seconds")
            except ValueError:
                pass
        state["progress"] = progress
        recorded = [
            int(value)
            for value in state.get("milestones_recorded", [])
        ]
        milestones = pending_progress_milestones(
            float(progress.get("fraction", 0.0)),
            recorded,
        )
        for milestone in milestones:
            history(
                "MILESTONE",
                run_id,
                (
                    f"progress={milestone}% step={progress.get('step')} "
                    f"total={progress.get('total_steps')}"
                ),
            )
            log(
                f"RUN_MILESTONE id={run_id} progress={milestone}% "
                f"step={progress.get('step')}"
            )
        if milestones:
            state["milestones_recorded"] = sorted(
                set(recorded) | set(milestones)
            )
    if tracks_training_checkpoint(run_id):
        explicit_checkpoints = tuple(
            Path(path.strip())
            for path in RESUME_CHECKPOINT_RE.findall(recent)
        )
        checkpoint = latest_complete_checkpoint(
            ROOT / "outputs",
            launched_at=state.get("launched_at"),
            explicit_paths=explicit_checkpoints,
        )
        if checkpoint is not None:
            state["latest_checkpoint"] = checkpoint
    else:
        state.pop("latest_checkpoint", None)
    current_size = log_path.stat().st_size if log_path.exists() else 0
    previous_size = int(state.get("last_observed_log_size", 0))
    new_text = ""
    if log_path.exists() and current_size > previous_size:
        with log_path.open("rb") as handle:
            handle.seek(previous_size)
            new_text = handle.read(min(current_size - previous_size, 2_000_000)).decode(
                "utf-8", errors="replace"
            )

    log_age = time.time() - log_path.stat().st_mtime if log_path.exists() else 1e12
    if ERROR_RE.search(new_text):
        history("ERROR_DETECTED", run_id, "fatal signature in newly written log")
        log(f"RUN_ERROR_SIGNATURE id={run_id}")
        # Most fatal distributed errors cause the launcher to exit. We do not
        # kill a still-running job immediately; one more heartbeat confirms a
        # stall before a registered process group is terminated.
        state["consecutive_stall_checks"] = max(
            1, int(state.get("consecutive_stall_checks", 0))
        )

    gpu_computing = bool(
        state.get("resource") == "gpu8"
        and gpu_snapshot is not None
        and any(
            gpu["utilization_gpu_percent"]
            > int(config["gpu_idle_utilization_threshold_percent"])
            for gpu in gpu_snapshot["gpus"]
        )
    )
    if log_age > int(config["stall_seconds"]) and not gpu_computing:
        state["consecutive_stall_checks"] = (
            int(state.get("consecutive_stall_checks", 0)) + 1
        )
        log(
            f"RUN_STALL_SUSPECT id={run_id} log_age_seconds={int(log_age)} "
            f"checks={state['consecutive_stall_checks']}"
        )
    else:
        state["consecutive_stall_checks"] = 0
        if log_age > int(config["stall_seconds"]) and gpu_computing:
            log(
                f"RUN_LOG_STALE_GPU_ACTIVE id={run_id} "
                f"log_age_seconds={int(log_age)}"
            )

    if int(state["consecutive_stall_checks"]) >= 2:
        process_group = int(state.get("process_group", pid))
        log(f"RUN_TERMINATING_REGISTERED_GROUP id={run_id} pgid={process_group}")
        history("TERMINATING", run_id, "two consecutive stall/error checks")
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.time() + int(config["terminate_grace_seconds"])
        while time.time() < deadline and pid_alive(pid):
            time.sleep(2)
        restart_active(state, "stalled or fatal-error registered run")
        return ACTIVE_PATH.exists()

    state["last_observed_log_size"] = current_size
    state["last_observed_at"] = stamp()
    state["recent_tail"] = recent[-4000:]
    atomic_json(ACTIVE_PATH, state)
    progress_text = (
        f" progress={state['progress']}"
        if state.get("progress")
        else ""
    )
    log(
        f"RUN_HEALTHY id={run_id} pid={pid} "
        f"log_age_seconds={int(log_age)}{progress_text}"
    )
    return True


def maybe_launch_next(
    config: dict[str, Any], gpu_snapshot: dict[str, Any]
) -> None:
    if not config.get("auto_launch_queue", False):
        return
    comments, rows = load_queue()
    waiting_for_gpu: list[str] = []
    for index, row in enumerate(rows):
        status, run_id, resource, max_retries, cwd, success_path, encoded = row
        if status != "READY":
            continue
        if success_path and glob.glob(success_path):
            reason = (
                "READY item has a pre-existing success artifact: "
                f"{success_path}"
            )
            rows[index][0] = "BLOCKED"
            save_queue(comments, rows)
            history("NEEDS_DEBUG", run_id, reason)
            log(f"QUEUE_STALE_SUCCESS id={run_id} path={success_path}")
            return
        if resource == "gpu8" and not gpus_idle(gpu_snapshot, config):
            log(f"QUEUE_WAIT_GPU id={run_id}")
            waiting_for_gpu.append(run_id)
            continue
        try:
            command = base64.b64decode(encoded).decode("utf-8")
        except Exception as exc:
            rows[index][0] = "INVALID"
            save_queue(comments, rows)
            history("QUEUE_INVALID", run_id, f"command decode failed: {exc}")
            return
        rows[index][0] = "RUNNING"
        save_queue(comments, rows)
        launch_run(
            run_id=run_id,
            resource=resource,
            max_retries=int(max_retries),
            cwd=cwd,
            success_path=success_path,
            command=command,
            retries=0,
        )
        return
    if waiting_for_gpu:
        log(f"QUEUE_NO_RUNNABLE_ITEMS gpu_wait={waiting_for_gpu}")
    else:
        log("QUEUE_NO_READY_ITEMS")


def heartbeat() -> None:
    ensure_dirs()
    config = load_json(CONFIG_PATH, {})
    gpu = collect_gpu_snapshot()
    filesystem = filesystem_snapshot()
    external = [inspect_run(spec) for spec in config.get("external_runs", [])]
    code, process_output = run_command(
        ["ps", "-eo", "pid,ppid,etime,%cpu,%mem,args", "--sort=-%cpu"],
        timeout=15,
    )
    snapshot = {
        "timestamp": stamp(),
        "gpu": gpu,
        "filesystem": filesystem,
        "external_runs": external,
        "process_returncode": code,
        "process_top": "\n".join(process_output.splitlines()[:100]),
    }
    snapshot_name = now().strftime("%Y%m%dT%H%M%S%z") + ".json"
    atomic_json(SNAPSHOT_DIR / snapshot_name, snapshot)

    gpu_count = len(gpu["gpus"])
    max_temp = max(
        (item["temperature_c"] for item in gpu["gpus"]), default=-1
    )
    used = [item["memory_used_mib"] for item in gpu["gpus"]]
    utils = [item["utilization_gpu_percent"] for item in gpu["gpus"]]
    log(
        f"HEALTH gpu_count={gpu_count} gpu_mem_used_mib={used} "
        f"gpu_util_percent={utils} max_temp_c={max_temp} "
        f"project_free_gib={filesystem['free_gib']}"
    )

    if gpu_count != int(config["expected_gpu_count"]):
        log(
            f"ALERT_GPU_COUNT expected={config['expected_gpu_count']} "
            f"observed={gpu_count}"
        )
    if max_temp > int(config["max_gpu_temperature_c"]):
        log(
            f"ALERT_GPU_TEMPERATURE limit={config['max_gpu_temperature_c']} "
            f"observed={max_temp}"
        )
    if filesystem["free_gib"] < float(config["minimum_project_free_gib"]):
        log(
            f"ALERT_LOW_DISK limit_gib={config['minimum_project_free_gib']} "
            f"observed_gib={filesystem['free_gib']}"
        )

    for item in external:
        log(
            f"EXTERNAL_RUN id={item['id']} alive={item['alive']} "
            f"success={item['success']} progress={item['progress']} "
            f"log={item['log']} recent_error={item['recent_error_signature']}"
        )

    active = inspect_active(config, gpu)
    if not active:
        maybe_launch_next(config, gpu)


def main() -> int:
    ensure_dirs()
    lock_path = CONTROL_DIR / "heartbeat.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("HEARTBEAT_SKIPPED another invocation holds the lock")
            return 0
        try:
            heartbeat()
        except Exception as exc:
            log(f"HEARTBEAT_EXCEPTION type={type(exc).__name__} message={exc}")
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
