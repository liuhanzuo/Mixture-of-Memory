#!/usr/bin/env python3
"""Build a deterministic HumanEval+ screen stratified by depth and length."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from analyze_eval_by_depth import depth_group, maximum_compound_depth


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def token_length(source: str) -> int:
    # Stable tokenizer-independent lexical complexity proxy.
    return len(
        [
            node
            for node in ast.walk(ast.parse(source))
            if not isinstance(node, (ast.Load, ast.Store, ast.Del))
        ]
    )


def build(rows: list[dict[str, Any]], *, size: int, seed: int):
    enriched = []
    for row in rows:
        source = str(row.get("prompt", "")) + str(
            row["canonical_solution"]
        )
        depth = maximum_compound_depth(source)
        enriched.append(
            {
                **row,
                "capacity_compound_depth": depth,
                "capacity_depth_group": depth_group(depth),
                "capacity_source_lines": len(source.splitlines()),
                "capacity_ast_nodes": token_length(source),
            }
        )
    if size > len(enriched):
        raise ValueError("requested screen exceeds dataset size")
    # Round-robin over depth groups after independent deterministic shuffles.
    groups = {}
    for row in enriched:
        groups.setdefault(row["capacity_depth_group"], []).append(row)
    for offset, key in enumerate(sorted(groups)):
        groups[key].sort(key=lambda row: row["task_id"])
        random.Random(seed + offset).shuffle(groups[key])
    selected = []
    keys = sorted(groups)
    while len(selected) < size:
        advanced = False
        for key in keys:
            if groups[key] and len(selected) < size:
                selected.append(groups[key].pop())
                advanced = True
        if not advanced:
            break
    selected.sort(key=lambda row: row["task_id"])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    selected = build(
        read_jsonl(args.input),
        size=args.size,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, sort_keys=True) + "\n"
        for row in selected
    )
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "size": len(selected),
        "seed": args.seed,
        "task_ids": [row["task_id"] for row in selected],
        "depth_counts": {
            key: sum(
                row["capacity_depth_group"] == key for row in selected
            )
            for key in sorted(
                {row["capacity_depth_group"] for row in selected}
            )
        },
        "line_range": [
            min(row["capacity_source_lines"] for row in selected),
            max(row["capacity_source_lines"] for row in selected),
        ],
        "ast_node_range": [
            min(row["capacity_ast_nodes"] for row in selected),
            max(row["capacity_ast_nodes"] for row in selected),
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
