#!/usr/bin/env python3
"""Remove local filesystem metadata from merged evaluation summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PATH_FIELDS = {
    "ckpt", "val_path", "data_path", "output_dir", "cache_dir",
    "base_model_path", "model_path",
}


def sanitize(path: Path) -> None:
    data = json.loads(path.read_text())
    output_name = str(data.get("output_name", path.parent.name))
    model_id = (
        "allenai/OLMo-2-0425-1B"
        if output_name.startswith("1B_")
        else "allenai/OLMo-2-1124-7B"
    )
    meta = data.get("meta")
    if isinstance(meta, dict):
        for key in PATH_FIELDS:
            meta.pop(key, None)
        meta["base_model"] = model_id
        if "scratch" in output_name.lower():
            meta["initialization"] = "fully_random"
        elif "base_full" in output_name.lower():
            meta["initialization"] = "pretrained_base"
        else:
            meta["initialization"] = "inherited_front"
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="directory containing summary.json files")
    args = parser.parse_args()
    paths = sorted(args.root.rglob("summary.json"))
    if not paths:
        raise FileNotFoundError(f"no summary.json files under {args.root}")
    for path in paths:
        sanitize(path)
    print(f"sanitized {len(paths)} summaries under {args.root}")


if __name__ == "__main__":
    main()
