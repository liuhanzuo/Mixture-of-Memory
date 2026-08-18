#!/usr/bin/env python3
"""Select the best calibrated LoRA scale and emit a checkpoint pointer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--pointer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-screen-plus", type=float, default=0.50)
    args = parser.parse_args()

    report = json.loads(
        (args.calibration_root / "report.json").read_text(encoding="utf-8")
    )
    selected = dict(report["selected"])
    label = str(selected["label"])
    checkpoint = (args.calibration_root / f"checkpoint_{label}").resolve()
    if not (checkpoint / "model.safetensors.index.json").is_file():
        raise SystemExit(f"selected checkpoint is incomplete: {checkpoint}")
    plus = float(selected["plus_pass1"])
    parse_rate = float(selected["parse_rate"])
    errors = int(selected["errors"])
    if plus < args.minimum_screen_plus:
        raise SystemExit(
            f"best screening HE+ {plus:.6f} is below "
            f"{args.minimum_screen_plus:.6f}"
        )
    if parse_rate < 0.95 or errors:
        raise SystemExit(
            f"selected scale is unreliable: parse={parse_rate:.6f} "
            f"errors={errors}"
        )

    payload = {
        "calibration_root": str(args.calibration_root.resolve()),
        "checkpoint": str(checkpoint),
        "selection_rule": "max plus_pass1, then parse_rate",
        "minimum_screen_plus": args.minimum_screen_plus,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.pointer.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.pointer.write_text(str(checkpoint) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
