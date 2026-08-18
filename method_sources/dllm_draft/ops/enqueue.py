#!/usr/bin/env python3
"""Append a deterministic command to the heartbeat queue."""

from __future__ import annotations

import argparse
import base64
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "ops" / "queue.tsv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--resource", choices=("cpu", "gpu8"), required=True)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--success-path", default="")
    parser.add_argument("--command", required=True)
    args = parser.parse_args()

    encoded = base64.b64encode(args.command.encode("utf-8")).decode("ascii")
    with QUEUE.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "READY",
                args.id,
                args.resource,
                args.max_retries,
                args.cwd,
                args.success_path,
                encoded,
            ]
        )
    print(f"queued {args.id}")


if __name__ == "__main__":
    main()

