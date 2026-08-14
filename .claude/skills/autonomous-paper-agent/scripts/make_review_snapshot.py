#!/usr/bin/env python3
"""Create an immutable, hashed paper-review snapshot using only the standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                yield child
        return
    raise FileNotFoundError(path)


def copy_input(source: Path, destination: Path, label: str) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    if source.is_file():
        target = destination / label / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "snapshot_path": target.relative_to(destination).as_posix(),
                "source_path": source.resolve().as_posix(),
                "sha256": sha256_file(target),
                "size_bytes": target.stat().st_size,
            }
        )
        return records

    root = source.resolve()
    for item in iter_files(source):
        relative = item.resolve().relative_to(root)
        target = destination / label / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        records.append(
            {
                "snapshot_path": target.relative_to(destination).as_posix(),
                "source_path": item.resolve().as_posix(),
                "sha256": sha256_file(target),
                "size_bytes": target.stat().st_size,
            }
        )
    return records


def snapshot_digest(records: list[dict[str, str | int]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: str(row["snapshot_path"])):
        digest.update(str(record["snapshot_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True, help="Zero-based review round number.")
    parser.add_argument("--paper", type=Path, required=True, help="Compiled manuscript PDF.")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        type=Path,
        help="Additional file or directory to copy. Repeat for multiple inputs.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("review"))
    parser.add_argument("--force", action="store_true", help="Replace an existing round snapshot.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.round < 0:
        raise ValueError("--round must be non-negative")
    if not args.paper.is_file():
        raise FileNotFoundError(f"Paper PDF not found: {args.paper}")

    round_dir = args.output_root / f"round_{args.round:02d}"
    destination = round_dir / "submission"
    if destination.exists():
        if not args.force:
            raise FileExistsError(f"Snapshot already exists: {destination}; use --force to replace it")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, str | int]] = []
    records.extend(copy_input(args.paper, destination, "manuscript"))

    used_labels: set[str] = {"manuscript"}
    for index, source in enumerate(args.include, start=1):
        if not source.exists():
            raise FileNotFoundError(f"Included path not found: {source}")
        base = source.name or f"input_{index}"
        label = base
        suffix = 2
        while label in used_labels:
            label = f"{base}_{suffix}"
            suffix += 1
        used_labels.add(label)
        records.extend(copy_input(source, destination, label))

    manifest = {
        "schema_version": "1.0.0",
        "round": args.round,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_sha256": snapshot_digest(records),
        "files": sorted(records, key=lambda row: str(row["snapshot_path"])),
    }
    manifest_path = destination / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"snapshot_dir": destination.as_posix(), **manifest}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
