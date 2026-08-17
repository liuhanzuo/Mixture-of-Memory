# -*- coding: utf-8 -*-
"""Detached flusher and HTTP sender."""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None

try:
    from urllib import request as urlrequest
except ImportError:  # pragma: no cover
    import urllib.request as urlrequest

from . import constants, paths
from .json_utils import atomic_write_json, read_json_recover
from .queue_store import (
    FileLock,
    claim_pending_file,
    cleanup_failed_files,
    delete_sending,
    log_error,
    log_info,
    move_sending_back,
    parse_queue_filename,
    pending_files,
    quarantine_events,
    read_jsonl,
    recover_sending_files,
    replace_sending_with_failed_events,
    replace_sending_with_pending_events,
)


def _load_flush_state():
    # Delivery state is disposable runtime metadata. A malformed file is
    # backed up by read_json_recover and treated as an empty state so it cannot
    # prevent pending telemetry events from being retried.
    return read_json_recover(paths.flush_state_path(), default={}) or {}


def _save_flush_state(state):
    try:
        paths.state_dir().mkdir(parents=True, exist_ok=True)
        atomic_write_json(paths.flush_state_path(), state)
    except Exception as e:
        log_error("failed to save flush state: %r" % (e,))


def _load_file_retry_state():
    # Per-file retry counts are also recoverable runtime metadata. Reset them
    # after corruption rather than blocking the entire delivery pipeline.
    return read_json_recover(paths.file_retry_state_path(), default={}) or {}


def _save_file_retry_state(state):
    try:
        paths.state_dir().mkdir(parents=True, exist_ok=True)
        atomic_write_json(paths.file_retry_state_path(), state)
    except Exception as e:
        log_error("failed to save file retry state: %r" % (e,))


def _file_retry_key(path):
    return Path(path).name


def note_file_failure(path):
    state = _load_file_retry_state()
    key = _file_retry_key(path)
    entry = state.get(key) or {}
    count = int(entry.get("retry_count") or 0) + 1
    state[key] = {"retry_count": count, "last_failure_at_epoch": time.time()}
    _save_file_retry_state(state)
    return count


def clear_file_failure(path):
    state = _load_file_retry_state()
    if state.pop(_file_retry_key(path), None) is not None:
        _save_file_retry_state(state)


def in_backoff():
    state = _load_flush_state()
    until = state.get("backoff_until_epoch")
    try:
        return until is not None and time.time() < float(until)
    except Exception:
        return False


def note_success():
    _save_flush_state({"failure_count": 0})


def note_failure():
    """Record endpoint-level health failure; file retries are tracked separately."""
    state = _load_flush_state()
    count = int(state.get("failure_count") or 0) + 1
    state.update(
        {
            "failure_count": count,
            "last_failure_at_epoch": time.time(),
            "backoff_until_epoch": time.time() + constants.FLUSH_BACKOFF_SECONDS,
        }
    )
    _save_flush_state(state)
    return count


def _try_lock_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a+")
    if fcntl is None:
        return fh
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fh
    except (IOError, OSError):
        fh.close()
        return None


def _unlock_file(fh):
    try:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def try_start_flusher(hook_script_path, config):
    try:
        endpoint = ((config.get("backend") or {}).get("endpoint") or "").strip()
        if not endpoint:
            return False
        if in_backoff():
            return False
        fh = _try_lock_path(paths.flush_lock_path())
        if fh is None:
            return False
        _unlock_file(fh)
        subprocess.Popen(
            [sys.executable, str(hook_script_path), "--flush"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=(os.name != "nt"),
        )
        return True
    except Exception as e:
        log_error("failed to start flusher: %r" % (e,))
        return False


def _encode_json(value):
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":")).encode("utf-8")


def _ack_from_response(payload, batch):
    event_ids = {event.get("event_id") for event in batch.get("events") or []}
    event_ids.discard(None)
    if payload.get("ack_version") == constants.EVENT_ACK_MODE:
        accepted = set(payload.get("accepted_event_ids") or []) & event_ids
        rejected = {
            item.get("event_id"): item
            for item in (payload.get("rejected_events") or [])
            if isinstance(item, dict) and item.get("event_id") in event_ids
            and item.get("retryable") is False
        }
        retryable = {
            item.get("event_id")
            for item in (payload.get("retryable_events") or [])
            if isinstance(item, dict) and item.get("event_id") in event_ids
        }
        unresolved = event_ids - accepted - set(rejected) - retryable
        retryable.update(unresolved)
        return {"accepted": accepted, "rejected": rejected, "retryable": retryable}

    # Legacy servers are all-or-nothing. Accept their response only when the
    # published count confirms that every event in this batch was accepted.
    if payload.get("status") == "published" and payload.get("published_events") == len(event_ids):
        return {"accepted": event_ids, "rejected": {}, "retryable": set()}
    return None


def post_batch(endpoint, batch, timeout_ms):
    data = _encode_json(batch)
    req = urlrequest.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = max(float(timeout_ms or constants.FLUSH_TOTAL_TIMEOUT_MS) / 1000.0, 0.1)
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            if not 200 <= int(code) < 300:
                return None
            try:
                payload = json.loads(resp.read().decode("utf-8") or "{}")
            except Exception:
                return None
            return _ack_from_response(payload, batch)
    except Exception:
        raise


def _queue_file_metadata(sending_path):
    parsed = parse_queue_filename(sending_path)
    agent_variant = None
    date_key = None
    if parsed:
        agent_variant, date_key, _part = parsed
    return agent_variant, date_key


def batch_for_file(sending_path, events, batch_id=None):
    agent_variant, date_key = _queue_file_metadata(sending_path)
    return {
        "schema_version": constants.SCHEMA_VERSION,
        "ack_mode": constants.EVENT_ACK_MODE,
        "batch_id": batch_id or str(uuid.uuid4()),
        "agent_variant": agent_variant,
        "date_key": date_key,
        "events": events,
    }


def batches_for_file(sending_path, events, config):
    queue_cfg = config.get("queue") or {}
    max_mb = int(queue_cfg.get("max_http_batch_size_mb") or constants.MAX_HTTP_BATCH_SIZE_MB)
    max_bytes = max_mb * 1024 * 1024
    batches = []
    current_events = []
    current_batch_id = str(uuid.uuid4())
    base_size = len(_encode_json(batch_for_file(sending_path, [], current_batch_id)))
    current_size = base_size

    for event in events:
        event_size = len(_encode_json(event))
        additional = event_size + (1 if current_events else 0)
        if current_events and current_size + additional > max_bytes:
            batches.append(batch_for_file(sending_path, current_events, current_batch_id))
            current_events = []
            current_batch_id = str(uuid.uuid4())
            base_size = len(_encode_json(batch_for_file(sending_path, [], current_batch_id)))
            current_size = base_size
            additional = event_size
        if current_size + additional > max_bytes:
            raise ValueError(
                "single event exceeds HTTP batch limit: event_id=%s size_bytes=%d limit_mb=%d"
                % (event.get("event_id"), event_size, max_mb)
            )
        current_events.append(event)
        current_size += additional

    if current_events:
        batches.append(batch_for_file(sending_path, current_events, current_batch_id))
    return batches


def _cleanup_failed(config):
    try:
        stats = cleanup_failed_files(config)
        deleted = stats["expired_deleted"] + stats["quota_deleted"]
        if deleted:
            log_info(
                "failed cleanup expired_deleted=%d quota_deleted=%d reclaimed_bytes=%d remaining_files=%d remaining_bytes=%d"
                % (stats["expired_deleted"], stats["quota_deleted"], stats["reclaimed_bytes"], stats["remaining_files"], stats["remaining_bytes"])
            )
    except Exception as e:
        log_error("failed cleanup crashed: %r" % (e,))


def _checkpoint_delivery(sending, retry_events, quarantine_records, config):
    """Apply an ACK result without reintroducing accepted events into pending."""
    if quarantine_records and not quarantine_events(sending, quarantine_records):
        retry_events.extend(record["event"] for record in quarantine_records)
    if retry_events:
        retries = note_file_failure(sending)
        if retries >= constants.FLUSH_MAX_RETRY:
            if replace_sending_with_failed_events(sending, retry_events):
                clear_file_failure(sending)
                _cleanup_failed(config)
                return True
            return False
        return replace_sending_with_pending_events(sending, retry_events)
    clear_file_failure(sending)
    return delete_sending(sending)


def flush_pending(config):
    endpoint = ((config.get("backend") or {}).get("endpoint") or "").strip()
    if not endpoint:
        return 0
    fh = _try_lock_path(paths.flush_lock_path())
    if fh is None:
        return 0
    sent_count = 0
    deadline = time.time() + constants.FLUSH_MAX_DRAIN_SECONDS
    empty_scans = 0
    try:
        _cleanup_failed(config)
        recovered = recover_sending_files()
        if recovered:
            log_info("recovered %d interrupted sending file(s)" % recovered)

        while time.time() < deadline:
            files = pending_files()
            if not files:
                empty_scans += 1
                if empty_scans >= 2:
                    break
                time.sleep(constants.FLUSH_IDLE_GRACE_MS / 1000.0)
                continue
            empty_scans = 0

            for pending in files:
                if time.time() >= deadline:
                    break
                sending = claim_pending_file(pending)
                if not sending:
                    continue
                events, bad_lines = read_jsonl(sending, return_bad_lines=True)
                if bad_lines and not quarantine_events(sending, bad_lines):
                    # Keep the original file intact if preserving malformed
                    # local records fails; never silently discard evidence.
                    move_sending_back(sending)
                    continue
                if not events:
                    delete_sending(sending)
                    clear_file_failure(sending)
                    continue
                retry_events = []
                quarantine_records = []
                endpoint_failed = False
                try:
                    batches = batches_for_file(sending, events, config)
                    for index, batch in enumerate(batches):
                        try:
                            ack = post_batch(endpoint, batch, (config.get("backend") or {}).get("timeout_ms"))
                        except Exception as e:
                            # Earlier batches may already have returned event-level
                            # ACKs. Only retry this unacknowledged batch and the
                            # batches that have not been attempted yet; do not
                            # requeue events that were explicitly accepted.
                            log_error("post batch failed for %s: %r" % (sending, e))
                            endpoint_failed = True
                            retry_events.extend(batch["events"])
                            for later in batches[index + 1:]:
                                retry_events.extend(later["events"])
                            break
                        if ack is None:
                            endpoint_failed = True
                            retry_events.extend(batch["events"])
                            for later in batches[index + 1:]:
                                retry_events.extend(later["events"])
                            break
                        if ack["retryable"]:
                            # The HTTP ingress was reachable, but at least one
                            # event was not durably accepted by Pulsar. Apply the
                            # endpoint backoff before retrying that remainder.
                            endpoint_failed = True
                        for event in batch["events"]:
                            event_id = event.get("event_id")
                            if event_id in ack["accepted"]:
                                sent_count += 1
                            elif event_id in ack["rejected"]:
                                quarantine_records.append({"event": event, "rejection": ack["rejected"][event_id]})
                            else:
                                retry_events.append(event)
                except Exception as e:
                    # Preserve the original sending file on unexpected local
                    # processing/checkpoint preparation failures. The targeted
                    # per-batch exception above handles network failures without
                    # requeueing already acknowledged batches.
                    log_error("flush preparation failed for %s: %r" % (sending, e))
                    endpoint_failed = True
                    retry_events = list(events)

                if not _checkpoint_delivery(sending, retry_events, quarantine_records, config):
                    # Preserve the original file on checkpoint errors rather than lose events.
                    move_sending_back(sending)
                if endpoint_failed:
                    note_failure()
                    return sent_count
                note_success()
        return sent_count
    finally:
        _unlock_file(fh)
