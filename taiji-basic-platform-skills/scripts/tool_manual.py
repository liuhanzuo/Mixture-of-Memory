#!/usr/bin/env python3
"""顶层兜底 tool_manual：跨所有子 skill 扫描 _api.md，按需提取工具手册。

与各子 skill 内的 tool_manual.py 输出格式完全一致，区别在于：
  - 扫描范围：所有 sub-skills/*/references/*_api.md（排除 helper_api.md 与 helpers/ 目录）
  - 同名工具会标注所属子 skill；同名时不报错，返回第一个匹配项 + 警告

用法：
    python3 scripts/tool_manual.py <tool_name> [<tool_name> ...]
    python3 scripts/tool_manual.py --list
    python3 scripts/tool_manual.py --json <tool_name>
"""

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

SKILL_ROOT = Path(__file__).resolve().parents[1]
SUB_SKILLS_DIR = SKILL_ROOT / "sub-skills"


@dataclass(frozen=True)
class Manual:
    tool: str
    sub_skill: str
    source: str       # 子 skill 内的相对路径
    line_start: int
    line_end: int
    content: str


def in_fenced_code(lines: Iterable[str]) -> list[bool]:
    states: list[bool] = []
    fenced = False
    for line in lines:
        states.append(fenced)
        if line.lstrip().startswith("```"):
            fenced = not fenced
    return states


def is_helper(filename: str) -> bool:
    """排除 helper 手册（helper_api.md 或 helpers/ 目录下的文件）。"""
    return filename == "helper_api.md" or filename.startswith("helpers/")


def api_files_in_skill(skill_dir: Path) -> list[Path]:
    """收集一个子 skill 下所有非 helper 的 _api.md。"""
    ref_dir = skill_dir / "references"
    if not ref_dir.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(ref_dir.rglob("*_api.md")):
        rel = str(path.relative_to(ref_dir))
        if not is_helper(rel):
            files.append(path)
    return files


def load_all_manuals() -> list[Manual]:
    """跨所有子 skill 收集工具手册。"""
    manuals: list[Manual] = []
    for skill_dir in sorted(SUB_SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        sub_name = skill_dir.name
        ref_dir = skill_dir / "references"
        for path in api_files_in_skill(skill_dir):
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            fenced = in_fenced_code(lines)
            headers: list[tuple[int, re.Match[str]]] = []
            for idx, line in enumerate(lines):
                if fenced[idx] or not H2_RE.match(line):
                    continue
                m = HEADER_RE.match(line)
                if m:
                    headers.append((idx, m))
            for pos, (start, m) in enumerate(headers):
                end = headers[pos + 1][0] if pos + 1 < len(headers) else len(lines)
                manuals.append(
                    Manual(
                        tool=m.group(1),
                        sub_skill=sub_name,
                        source=str(path.relative_to(ref_dir)),
                        line_start=start + 1,
                        line_end=end,
                        content="".join(lines[start:end]).rstrip() + "\n",
                    )
                )
    return manuals


def index_manuals(manuals: list[Manual]) -> tuple[dict[str, Manual], dict[str, list[str]]]:
    """构建 {tool_name: Manual} 索引。

    Returns:
        (by_tool, duplicates)：同名工具取第一个匹配项，duplicates 记录同名警告。
    """
    by_tool: dict[str, Manual] = {}
    dupes: dict[str, list[str]] = {}
    for m in manuals:
        if m.tool in by_tool:
            dupes.setdefault(m.tool, [by_tool[m.tool].sub_skill]).append(m.sub_skill)
        else:
            by_tool[m.tool] = m
    return by_tool, dupes


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


def _manuals_to_json(manuals: list[Manual]) -> list[dict]:
    """序列化 Manual 列表（兼容子 skill 版 tool_manual.py 输出字段）。"""
    return [
        {
            "tool": m.tool,
            "sub_skill": m.sub_skill,
            "source": f"sub-skills/{m.sub_skill}/references/{m.source}",
            "line_start": m.line_start,
            "line_end": m.line_end,
            "content": m.content,
        }
        for m in manuals
    ]


def main() -> int:
    args = parse_args()
    try:
        manuals = load_all_manuals()
    except (OSError, UnicodeError) as exc:
        print(f"tool_manual.py: 加载手册失败: {exc}", file=sys.stderr)
        return 2

    by_tool, dupes = index_manuals(manuals)

    if dupes:
        for tool, skills in dupes.items():
            print(
                f"tool_manual.py: 工具 `{tool}` 在多个子 skill 中出现（{', '.join(skills)}），"
                f"已使用第一个匹配项",
                file=sys.stderr,
            )

    if args.list:
        items = sorted(
            _manuals_to_json(manuals),
            key=lambda x: x["tool"],
        )
        if args.json:
            print_json({"tools": [{k: v for k, v in m.items() if k != "content"} for m in items]})
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
        print_json({"manuals": _manuals_to_json(selected)})
    else:
        for idx, item in enumerate(selected):
            if idx:
                print("\n" + "=" * 80 + "\n")
            print(item.content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
