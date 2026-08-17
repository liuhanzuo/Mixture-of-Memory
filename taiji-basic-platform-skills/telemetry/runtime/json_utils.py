# -*- coding: utf-8 -*-
"""Small JSON and file helpers."""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path


class JsonFileCorruptionError(ValueError):
    """A JSON file could not be decoded and was copied aside for recovery."""

    def __init__(self, path, backup_path, cause):
        self.path = Path(path)
        self.backup_path = Path(backup_path) if backup_path else None
        self.cause = cause
        super().__init__(
            "invalid JSON in %s; backup=%s; cause=%r"
            % (self.path, self.backup_path, cause)
        )


def backup_corrupt_json(path):
    path = Path(path)
    if not path.exists():
        return None
    ts = time.strftime("%Y%m%dT%H%M%S")
    dest = path.with_name(path.name + "." + ts + ".corrupt")
    suffix = 1
    while dest.exists():
        dest = path.with_name(path.name + "." + ts + "." + str(suffix) + ".corrupt")
        suffix += 1
    try:
        shutil.copy2(str(path), str(dest))
        return dest
    except OSError:
        return None


def read_json(path, default=None):
    """Read JSON strictly; malformed files are preserved and reported to caller."""
    path = Path(path)
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise JsonFileCorruptionError(path, backup_corrupt_json(path), e)


def read_json_recover(path, default=None):
    """Best-effort JSON read for runtime state that can safely reset to defaults."""
    try:
        return read_json(path, default=default)
    except JsonFileCorruptionError:
        return default


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def atomic_write_json(path, data):
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def backup_file(src, backup_dir, prefix):
    src = Path(src)
    if not src.exists():
        return None
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    dest = backup_dir / (prefix + "__" + src.name + "__" + ts + ".bak")
    shutil.copy2(str(src), str(dest))
    return dest


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def append_text_best_effort(path, text):
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass
