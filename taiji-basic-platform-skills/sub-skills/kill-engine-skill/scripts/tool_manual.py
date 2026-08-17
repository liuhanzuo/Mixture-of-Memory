#!/usr/bin/env python3
"""按需提取当前 sub-skill 的单工具 API 手册。"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

HEADER_RE = re.compile(r"^## ([A-Za-z_][A-Za-z0-9_.:-]*)\s*$")
H2_RE = re.compile(r"^##(?:\s|$)")


@dataclass(frozen=True)
class Manual:
    tool: str
    source: str
    line_start: int
    line_end: int
    content: str


def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


def api_files(root: Path) -> list[Path]:
    refs = root / "references"
    seen = set()
    files = []
    for p in sorted(refs.glob("**/*_api.md")):
        if str(p) not in seen:
            files.append(p)
            seen.add(str(p))
    api_dir = refs / "api"
    if api_dir.is_dir():
        for p in sorted(api_dir.glob("**/*.md")):
            if str(p) not in seen:
                files.append(p)
                seen.add(str(p))
    return files


def in_fenced_code(lines: Iterable[str]) -> list[bool]:
    states: list[bool] = []
    fenced = False
    for line in lines:
        states.append(fenced)
        if line.lstrip().startswith("```"):
            fenced = not fenced
    return states


def load_manuals(root: Path) -> list[Manual]:
    manuals: list[Manual] = []
    for path in api_files(root):
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        fenced = in_fenced_code(lines)
        headers: list[tuple[int, re.Match[str]]] = []
        for index, line in enumerate(lines):
            if fenced[index] or not H2_RE.match(line):
                continue
            match = HEADER_RE.match(line)
            if match is None:
                # 跳过非裸工具名的 H2（如中文章节标题、emoji 标题等）
                continue
            headers.append((index, match))
        for position, (start, match) in enumerate(headers):
            end = headers[position + 1][0] if position + 1 < len(headers) else len(lines)
            manuals.append(
                Manual(
                    tool=match.group(1),
                    source=str(path.relative_to(root)),
                    line_start=start + 1,
                    line_end=end,
                    content="".join(lines[start:end]).rstrip() + "\n",
                )
            )
    return manuals


def index_manuals(manuals: list[Manual]) -> dict[str, Manual]:
    grouped: dict[str, list[Manual]] = {}
    for manual in manuals:
        grouped.setdefault(manual.tool, []).append(manual)
    duplicates = {tool: values for tool, values in grouped.items() if len(values) > 1}
    if duplicates:
        details = "\n".join(
            f"{tool}: {', '.join(item.source for item in values)}"
            for tool, values in sorted(duplicates.items())
        )
        raise ValueError("发现重复工具手册：\n" + details)
    return {tool: values[0] for tool, values in grouped.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tools", nargs="*", help="要读取的工具名，可传多个")
    parser.add_argument("--list", action="store_true", help="列出全部工具手册")
    parser.add_argument("--json", action="store_true", help="输出单个 JSON 文档")
    args = parser.parse_args()
    if args.list and args.tools:
        parser.error("--list 不能与工具名同时使用")
    if not args.list and not args.tools:
        parser.error("请提供工具名，或使用 --list")
    return args


def print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    root = skill_root()
    try:
        manuals = load_manuals(root)
        by_tool = index_manuals(manuals)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"tool_manual.py: {exc}", file=sys.stderr)
        return 2

    if args.list:
        items = [
            {"tool": item.tool, "source": item.source, "line_start": item.line_start, "line_end": item.line_end}
            for item in sorted(manuals, key=lambda value: value.tool)
        ]
        if args.json:
            print_json({"tools": items})
        else:
            for item in items:
                print(f"{item['tool']}\t{item['source']}:{item['line_start']}-{item['line_end']}")
        return 0

    missing = [tool for tool in args.tools if tool not in by_tool]
    if missing:
        choices = sorted(by_tool)
        for tool in missing:
            suggestion = difflib.get_close_matches(tool, choices, n=3)
            suffix = f"；可能是：{', '.join(suggestion)}" if suggestion else ""
            print(f"tool_manual.py: 未找到工具 `{tool}`{suffix}", file=sys.stderr)
        return 2

    selected = [by_tool[tool] for tool in args.tools]
    if args.json:
        print_json({"manuals": [asdict(item) for item in selected]})
    else:
        for index, item in enumerate(selected):
            if index:
                print("\n" + "=" * 80 + "\n")
            print(item.content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
