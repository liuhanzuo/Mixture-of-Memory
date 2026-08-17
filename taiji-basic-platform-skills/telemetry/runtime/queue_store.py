# -*- coding: utf-8 -*-
"""Local JSONL queue storage for telemetry events."""

import json
import os
import re
import time
from pathlib import Path

try:
    import fcntl
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None

from . import constants, paths
from .json_utils import append_text_best_effort

_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_FILE_RE = re.compile(
    r"^(?P<agent>.+)__(?P<date>\d{8})__(?P<part>\d{4})"
    r"(?:__(?P<suffix>\d+))?\.jsonl$"
)


def _safe(s):
    return _SAFE_RE.sub("_", str(s or "unknown"))[:160]


def log_error(message):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    append_text_best_effort(paths.telemetry_log_path(), "[%s] ERROR %s\n" % (ts, message))


def log_info(message):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    append_text_best_effort(paths.telemetry_log_path(), "[%s] INFO %s\n" % (ts, message))


class FileLock(object):
    def __init__(self, path, wait_ms=constants.LOCK_WAIT_MS):
        self.path = Path(path)
        self.wait_ms = wait_ms
        self.fh = None
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a+")
        if fcntl is None:
            self.acquired = True
            return self
        deadline = time.time() + (self.wait_ms / 1000.0)
        while True:
            try:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.acquired = True
                return self
            except (IOError, OSError):
                if time.time() >= deadline:
                    raise TimeoutError("lock timeout: %s" % self.path)
                time.sleep(0.005)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fh and fcntl is not None and self.acquired:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        finally:
            if self.fh:
                self.fh.close()
        return False


def pending_total_bytes():
    total = 0
    pdir = paths.pending_dir()
    if not pdir.exists():
        return 0
    for p in pdir.glob("*.jsonl"):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def _parse_part(path):
    m = _FILE_RE.match(path.name)
    if not m:
        return None
    return int(m.group("part"))


def _file_for(directory, agent, date_key, part):
    return directory / ("%s__%s__%04d.jsonl" % (_safe(agent), _safe(date_key), part))


def _lock_for(agent, date_key, part):
    return paths.queue_locks_dir() / ("%s__%s__%04d.lock" % (_safe(agent), _safe(date_key), part))


def _count_lines(path, limit=None):
    count = 0
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            for _ in f:
                count += 1
                if limit is not None and count >= limit:
                    return count
    except OSError:
        return 0
    return count


def _within_threshold(path, max_size_bytes, max_events, incoming_size=0):
    p = Path(path)
    if not p.exists():
        return True
    try:
        current_size = p.stat().st_size
        if current_size >= max_size_bytes:
            return False
        if incoming_size and current_size + incoming_size > max_size_bytes:
            return False
    except OSError:
        return False
    return _count_lines(p, limit=max_events) < max_events


def _existing_parts(agent, date_key):
    parts = set()
    for d in [paths.pending_dir(), paths.sending_dir(), paths.failed_dir()]:
        if not d.exists():
            continue
        pattern = "%s__%s__*.jsonl" % (_safe(agent), _safe(date_key))
        for p in d.glob(pattern):
            part = _parse_part(p)
            if part is not None:
                parts.add(part)
    return sorted(parts)


def choose_pending_file(agent, date_key, config, incoming_size=0):
    queue_cfg = config.get("queue") or {}
    max_mb = int(queue_cfg.get("max_file_size_mb") or constants.MAX_FILE_SIZE_MB)
    max_events = int(queue_cfg.get("max_events_per_file") or constants.MAX_EVENTS_PER_FILE)
    max_size_bytes = max_mb * 1024 * 1024
    parts = _existing_parts(agent, date_key)
    part = parts[-1] if parts else 1
    while True:
        pending = _file_for(paths.pending_dir(), agent, date_key, part)
        sending = _file_for(paths.sending_dir(), agent, date_key, part)
        failed = _file_for(paths.failed_dir(), agent, date_key, part)
        if sending.exists() or failed.exists():
            part += 1
            continue
        if _within_threshold(
            pending, max_size_bytes, max_events, incoming_size=incoming_size
        ):
            return pending, _lock_for(agent, date_key, part)
        part += 1


def append_event(event, config):
    try:
        paths.ensure_runtime_dirs()
        queue_cfg = config.get("queue") or {}
        pending_max_mb = int(queue_cfg.get("pending_max_mb") or constants.PENDING_MAX_MB)
        if pending_total_bytes() > pending_max_mb * 1024 * 1024:
            log_error("pending queue exceeds limit; dropping event %s" % event.get("event_id"))
            return False
        event_line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        event_size = len(event_line.encode("utf-8"))
        max_event_mb = int(
            queue_cfg.get("max_event_size_mb") or constants.MAX_EVENT_SIZE_MB
        )
        if event_size > max_event_mb * 1024 * 1024:
            log_error(
                "event exceeds size limit; dropping event_id=%s size_bytes=%d limit_mb=%d"
                % (event.get("event_id"), event_size, max_event_mb)
            )
            return False
        agent = ((event.get("source") or {}).get("agent_variant")) or "unknown"
        date_key = event.get("date_key") or time.strftime("%Y%m%d")
        pending, lock_path = choose_pending_file(
            agent, date_key, config, incoming_size=event_size
        )
        try:
            with FileLock(lock_path, wait_ms=constants.LOCK_WAIT_MS):
                pending.parent.mkdir(parents=True, exist_ok=True)
                with pending.open("a", encoding="utf-8") as f:
                    f.write(event_line)
            return True
        except Exception as e:
            log_error("failed to append pending: %r" % (e,))
            return False
    except Exception as e:
        log_error("append_event crashed: %r" % (e,))
        return False


def pending_files():
    paths.ensure_runtime_dirs()
    return sorted(paths.pending_dir().glob("*.jsonl"))


def sending_files():
    paths.ensure_runtime_dirs()
    return sorted(paths.sending_dir().glob("*.jsonl"))


def failed_files():
    paths.ensure_runtime_dirs()
    return sorted(paths.failed_dir().glob("*.jsonl"))


def parse_queue_filename(path):
    m = _FILE_RE.match(Path(path).name)
    if not m:
        return None
    return m.group("agent"), m.group("date"), int(m.group("part"))


def claim_pending_file(pending_path):
    parsed = parse_queue_filename(pending_path)
    if not parsed:
        return None
    agent, date_key, part = parsed
    lock_path = _lock_for(agent, date_key, part)
    try:
        with FileLock(lock_path, wait_ms=constants.LOCK_WAIT_MS):
            if not pending_path.exists():
                return None
            sending = _file_for(paths.sending_dir(), agent, date_key, part)
            sending.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(pending_path), str(sending))
            return sending
    except Exception as e:
        log_error("failed to claim pending %s: %r" % (pending_path, e))
        return None


def read_jsonl(path, return_bad_lines=False):
    events = []
    bad_lines = []
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception as e:
                    record = {
                        "source": "local_jsonl_parse",
                        "line_number": line_number,
                        "raw_line": raw_line.rstrip("\n"),
                        "error": repr(e),
                    }
                    bad_lines.append(record)
                    log_error("bad jsonl line in %s line=%d: %r" % (path, line_number, e))
    except Exception as e:
        log_error("failed to read jsonl %s: %r" % (path, e))
    if return_bad_lines:
        return events, bad_lines
    return events


def _write_event_lines(path, events):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def replace_sending_with_pending_events(path, events):
    """Persist only unsent events back to pending, then remove sending safely."""
    path = Path(path)
    parsed = parse_queue_filename(path)
    if not parsed:
        return False
    agent, date_key, part = parsed
    lock_path = _lock_for(agent, date_key, part)
    pending = _file_for(paths.pending_dir(), agent, date_key, part)
    try:
        with FileLock(lock_path, wait_ms=constants.LOCK_WAIT_MS):
            if not path.exists():
                return True
            if events:
                _write_event_lines(pending, events)
            path.unlink()
        return True
    except Exception as e:
        log_error("failed to checkpoint sending %s: %r" % (path, e))
        return False


def replace_sending_with_failed_events(path, events):
    """Move only retry-exhausted events to failed, preserving accepted events."""
    path = Path(path)
    parsed = parse_queue_filename(path)
    if not parsed:
        return False
    agent, date_key, part = parsed
    lock_path = _lock_for(agent, date_key, part)
    failed = _file_for(paths.failed_dir(), agent, date_key, part)
    try:
        with FileLock(lock_path, wait_ms=constants.LOCK_WAIT_MS):
            if not path.exists():
                return True
            if failed.exists():
                failed = failed.with_name(
                    failed.stem + "__" + str(time.time_ns()) + failed.suffix
                )
            _write_event_lines(failed, events)
            try:
                os.utime(str(failed), None)
            except OSError:
                pass
            path.unlink()
        return True
    except Exception as e:
        log_error("failed to checkpoint failed events %s: %r" % (path, e))
        return False


def quarantine_events(path, records):
    """Persist non-retryable event rejections for operator inspection."""
    if not records:
        return True
    path = Path(path)
    parsed = parse_queue_filename(path)
    if not parsed:
        return False
    agent, date_key, part = parsed
    lock_path = _lock_for(agent, date_key, part)
    out = paths.quarantine_dir() / (
        "%s__%s__%04d__%d.jsonl" % (_safe(agent), _safe(date_key), part, time.time_ns())
    )
    try:
        with FileLock(lock_path, wait_ms=constants.LOCK_WAIT_MS):
            _write_event_lines(out, records)
        return True
    except Exception as e:
        log_error("failed to quarantine events from %s: %r" % (path, e))
        return False


def delete_sending(path):
    try:
        Path(path).unlink()
        return True
    except FileNotFoundError:
        return True
    except Exception as e:
        log_error("failed to delete sending %s: %r" % (path, e))
        return False


def move_sending_back(path):
    path = Path(path)
    parsed = parse_queue_filename(path)
    if not parsed:
        return False
    agent, date_key, part = parsed
    lock_path = _lock_for(agent, date_key, part)
    pending = _file_for(paths.pending_dir(), agent, date_key, part)
    try:
        with FileLock(lock_path, wait_ms=constants.LOCK_WAIT_MS):
            if not path.exists():
                return True
            if pending.exists():
                # Merge content back if a same-name pending file somehow exists.
                with path.open("r", encoding="utf-8") as src, pending.open("a", encoding="utf-8") as dst:
                    for line in src:
                        dst.write(line)
                path.unlink()
            else:
                os.replace(str(path), str(pending))
        return True
    except Exception as e:
        log_error("failed to move sending back %s: %r" % (path, e))
        return False


def move_sending_failed(path):
    path = Path(path)
    parsed = parse_queue_filename(path)
    if not parsed:
        return False
    agent, date_key, part = parsed
    lock_path = _lock_for(agent, date_key, part)
    failed = _file_for(paths.failed_dir(), agent, date_key, part)
    try:
        with FileLock(lock_path, wait_ms=constants.LOCK_WAIT_MS):
            if not path.exists():
                return True
            failed.parent.mkdir(parents=True, exist_ok=True)
            if failed.exists():
                failed = failed.with_name(
                    failed.stem + "__" + str(time.time_ns()) + failed.suffix
                )
            os.replace(str(path), str(failed))
            try:
                os.utime(str(failed), None)
            except OSError:
                pass
        return True
    except Exception as e:
        log_error("failed to move sending failed %s: %r" % (path, e))
        return False


def recover_sending_files():
    """Return interrupted in-flight files to pending before a new flush run."""
    recovered = 0
    for path in sending_files():
        if move_sending_back(path):
            recovered += 1
    return recovered


def _queue_limit(config, key, default):
    queue_cfg = config.get("queue") or {}
    value = queue_cfg.get(key)
    if value is None:
        value = default
    try:
        return max(0, int(value))
    except Exception:
        return default


def cleanup_failed_files(config, now_epoch=None):
    """Best-effort failed retention by age first, then total directory size."""
    now_epoch = float(now_epoch if now_epoch is not None else time.time())
    retention_days = _queue_limit(
        config, "failed_retention_days", constants.FAILED_RETENTION_DAYS
    )
    max_mb = _queue_limit(config, "failed_max_mb", constants.FAILED_MAX_MB)
    cutoff = now_epoch - retention_days * 24 * 60 * 60
    entries = []
    stats = {
        "expired_deleted": 0,
        "quota_deleted": 0,
        "reclaimed_bytes": 0,
        "remaining_files": 0,
        "remaining_bytes": 0,
    }

    for path in failed_files():
        try:
            stat = path.stat()
        except Exception as e:
            log_error("failed cleanup stat %s: %r" % (path, e))
            continue
        if retention_days > 0 and stat.st_mtime < cutoff:
            try:
                path.unlink()
                stats["expired_deleted"] += 1
                stats["reclaimed_bytes"] += stat.st_size
            except FileNotFoundError:
                pass
            except Exception as e:
                log_error("failed cleanup unlink expired %s: %r" % (path, e))
                entries.append((stat.st_mtime, stat.st_size, path))
        else:
            entries.append((stat.st_mtime, stat.st_size, path))

    total_bytes = sum(entry[1] for entry in entries)
    max_bytes = max_mb * 1024 * 1024
    if max_mb > 0 and total_bytes > max_bytes:
        kept = []
        for mtime, size, path in sorted(entries, key=lambda item: item[0]):
            if total_bytes <= max_bytes:
                kept.append((mtime, size, path))
                continue
            try:
                path.unlink()
                total_bytes -= size
                stats["quota_deleted"] += 1
                stats["reclaimed_bytes"] += size
            except FileNotFoundError:
                total_bytes -= size
            except Exception as e:
                log_error("failed cleanup unlink quota %s: %r" % (path, e))
                kept.append((mtime, size, path))
        entries = kept

    stats["remaining_files"] = len(entries)
    stats["remaining_bytes"] = max(0, total_bytes)
    return stats
