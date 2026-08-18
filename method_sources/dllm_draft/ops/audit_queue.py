#!/usr/bin/env python3
"""Statically audit the live heartbeat queue without executing commands."""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_RE = re.compile(r"(?:^|\s)(\./scripts/[A-Za-z0-9_.-]+)")
VALID_STATUS = {
    "READY",
    "RUNNING",
    "DONE",
    "BLOCKED",
    "INVALID",
}


def read_rows(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        row = next(csv.reader([line], delimiter="\t"))
        rows.append(row)
    return rows


def audit_rows(rows: list[list[str]], root: Path) -> dict[str, Any]:
    issues = []
    ids: dict[str, int] = {}
    success_paths: dict[str, str] = {}
    statuses = {}
    for index, row in enumerate(rows):
        if len(row) != 7:
            issues.append(
                {"row": index, "issue": f"expected 7 fields, got {len(row)}"}
            )
            continue
        status, run_id, resource, retries, cwd, success_path, encoded = row
        statuses[status] = statuses.get(status, 0) + 1
        if status not in VALID_STATUS:
            issues.append({"id": run_id, "issue": f"unknown status {status}"})
        if run_id in ids:
            issues.append(
                {
                    "id": run_id,
                    "issue": f"duplicate ID, first row {ids[run_id]}",
                }
            )
        ids[run_id] = index
        if resource not in {"cpu", "gpu8"}:
            issues.append(
                {"id": run_id, "issue": f"unknown resource {resource}"}
            )
        try:
            if int(retries) < 0:
                raise ValueError
        except ValueError:
            issues.append(
                {"id": run_id, "issue": f"invalid max_retries {retries}"}
            )
        if not Path(cwd).is_dir():
            issues.append({"id": run_id, "issue": f"missing cwd {cwd}"})
        if success_path:
            previous = success_paths.get(success_path)
            if previous is not None and previous != run_id:
                issues.append(
                    {
                        "id": run_id,
                        "issue": (
                            f"success path also used by {previous}: "
                            f"{success_path}"
                        ),
                    }
                )
            success_paths[success_path] = run_id
            if status == "READY" and glob.glob(success_path):
                issues.append(
                    {
                        "id": run_id,
                        "issue": (
                            "READY item has pre-existing success artifact "
                            f"{success_path}"
                        ),
                    }
                )
        try:
            command = base64.b64decode(encoded, validate=True).decode("utf-8")
        except Exception as exc:
            issues.append(
                {"id": run_id, "issue": f"invalid command encoding: {exc}"}
            )
            continue
        for script in (
            SCRIPT_RE.findall(command)
            if status != "DONE"
            else ()
        ):
            path = (Path(cwd) / script).resolve()
            if not path.is_file():
                issues.append(
                    {"id": run_id, "issue": f"missing command script {path}"}
                )
            elif not path.stat().st_mode & 0o111:
                issues.append(
                    {
                        "id": run_id,
                        "issue": f"command script is not executable {path}",
                    }
                )
    return {
        "rows": len(rows),
        "statuses": statuses,
        "issues": issues,
        "ok": not issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=ROOT / "ops" / "queue.tsv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_rows(read_rows(args.queue), ROOT)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
