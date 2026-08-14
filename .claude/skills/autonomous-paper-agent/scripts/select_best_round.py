#!/usr/bin/env python3
"""Select the strongest verified review-round checkpoint, not merely the latest one."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROUND_PATTERN = re.compile(r"round_(\d+)$")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def round_number(path: Path) -> int:
    match = ROUND_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Invalid round directory name: {path.name}")
    return int(match.group(1))


def rank_tuple(round_dir: Path) -> tuple[Any, ...]:
    panel = read_json(round_dir / "panel_summary.json")
    if not isinstance(panel, dict):
        raise ValueError(f"Missing panel_summary.json in {round_dir}")
    gate = read_json(round_dir / "gate_status.json", {})
    meta = read_json(round_dir / "meta_review.json", {})

    integrity_pass = bool(gate.get("integrity_pass", False))
    unresolved_critical = int(gate.get("unresolved_critical", panel.get("issue_counts", {}).get("by_severity", {}).get("critical", 999)))
    unresolved_major = int(gate.get("unresolved_major_technical_evidence", panel.get("issue_counts", {}).get("by_severity", {}).get("major", 999)))
    overall = panel.get("overall", {})
    quality = panel.get("quality_index", {})

    return (
        1 if integrity_pass else 0,
        -unresolved_critical,
        -unresolved_major,
        float(overall.get("median", -1)),
        float(overall.get("lower_quartile", -1)),
        float(meta.get("meta_score", -1)),
        float(quality.get("median", -1)),
        -round_number(round_dir),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_root", type=Path, nargs="?", default=Path("review"))
    parser.add_argument("--output", type=Path, default=Path("review/best_checkpoint.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.review_root.is_dir():
        raise FileNotFoundError(f"Review root not found: {args.review_root}")

    candidates: list[tuple[Path, tuple[Any, ...]]] = []
    for round_dir in sorted(args.review_root.glob("round_*")):
        if not round_dir.is_dir() or not (round_dir / "panel_summary.json").exists():
            continue
        candidates.append((round_dir, rank_tuple(round_dir)))

    if not candidates:
        raise ValueError("No review rounds with panel_summary.json found")

    best_dir, best_rank = max(candidates, key=lambda item: item[1])
    payload = {
        "schema_version": "1.0.0",
        "selected_round": round_number(best_dir),
        "selected_path": best_dir.as_posix(),
        "rank_tuple": list(best_rank),
        "candidates": [
            {
                "round": round_number(path),
                "path": path.as_posix(),
                "rank_tuple": list(rank),
            }
            for path, rank in sorted(candidates, key=lambda item: round_number(item[0]))
        ],
        "ranking_order": [
            "integrity_pass",
            "fewer_unresolved_critical",
            "fewer_unresolved_major_technical_evidence",
            "higher_panel_median",
            "higher_panel_lower_quartile",
            "higher_meta_score",
            "higher_quality_index",
            "earlier_round_as_tie_breaker"
        ]
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
