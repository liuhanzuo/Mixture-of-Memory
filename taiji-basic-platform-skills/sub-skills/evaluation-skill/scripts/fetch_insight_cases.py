#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insight Case 明细对比数据拉取脚本.

用法:
    # 按 task_ids 拉取
    python3 fetch_insight_cases.py --task-ids 103674,103673 --output ./cases.jsonl

    # 按 insight_id 自动解析 task 列表
    python3 fetch_insight_cases.py --insight-id 2550 --output ./cases.jsonl

    # 指定 MCP URL 和 Token（默认从 ~/.config/taiji/credentials.json 读取）
    python3 fetch_insight_cases.py --task-ids 103674,103673 --output ./cases.jsonl \\
        --mcp-url http://taiji-openapi.woa.com \\
        --token i1M3uXrnHFROZu_RaMPglA

    # 按维度筛选 + 分数过滤
    python3 fetch_insight_cases.py --task-ids 103674,103673 --output ./math_cases.jsonl \\
        --dimension-filter '{"task_lv1":["数学"]}' \\
        --task-score-filters '[{"task_id":103674,"score_lte":0.5}]'

    # 指定分页大小（默认每页 50 条）
    python3 fetch_insight_cases.py --task-ids 103674,103673 --output ./cases.jsonl --page-size 100

数据格式:
    每行一个 JSON 对象，包含 question_id / exercise_name / exercise_version_id / common / task_case_details.
    文件末尾追加 meta 行: {"_meta": {"total": 120, "task_ids": [...], "insight_id": 2550, "fetched_at": "..."}}

API:
    POST /v1/hunyuan/evaluation/insight/list_cases
    GET  /v1/hunyuan/evaluation/insight/detail (insight_id 模式自动调用)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# -----------------------------
# MCP 调用（通过 connect_mcp.py）
# -----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CONNECT_MCP = str(SCRIPT_DIR / "connect_mcp.py")


def mcp_call(tool: str, args: dict) -> dict:
    """调用 MCP 工具，返回解析后的 JSON."""
    args_json = json.dumps(args)
    result = subprocess.run(
        ["python3", CONNECT_MCP, "call", tool, args_json],
        capture_output=True, text=True, timeout=120, cwd=str(SCRIPT_DIR),
    )
    if result.returncode != 0:
        print(f"[ERROR] MCP 调用失败: {result.stderr[:500]}", file=sys.stderr)
        sys.exit(1)
    output = result.stdout
    # 提取 📤 Result: ... 🎉 Done! 之间的 JSON
    marker_start = "📤 Result:\n" + "=" * 80 + "\n"
    marker_end = "\n🎉 Done!"
    start = output.find(marker_start)
    if start < 0:
        print(f"[ERROR] 无法解析 MCP 返回: {output[:500]}", file=sys.stderr)
        sys.exit(1)
    json_start = start + len(marker_start)
    end = output.find(marker_end, json_start)
    json_str = output[json_start:end] if end > 0 else output[json_start:]
    return json.loads(json_str)


def fetch_insight_detail(insight_id: int) -> dict | None:
    """通过 MCP 拉取 Insight detail."""
    try:
        resp = mcp_call("get_taiji_eval_insight_detail", {"insight_id": insight_id})
        data = resp.get("data", {})
        return {
            "insight_id": insight_id,
            "insight_name": data.get("name", ""),
            "base_line_task_id": (data.get("conf") or {}).get("base_line_task_id"),
            "weight_nodes": data.get("weight_nodes", []),
        }
    except Exception as e:
        print(f"[WARN] 获取 Insight detail 失败: {e}", file=sys.stderr)
        return None


def get_insight_task_ids(insight_id: int) -> list[int]:
    """通过 MCP 获取 insight 关联的 task_ids."""
    resp = mcp_call("get_taiji_eval_insight_detail", {"insight_id": insight_id})
    data = resp.get("data", {})
    tasks = data.get("tasks", [])
    task_ids = [t["id"] for t in tasks if "id" in t]
    print(f"[INFO] Insight {insight_id} 关联 {len(task_ids)} 个任务: {task_ids}", file=sys.stderr)
    return task_ids


def fetch_all_pages(
    task_ids: list[int],
    page_size: int = 50,
    exercise_version_ids: list[int] | None = None,
    dimension_filter: dict | None = None,
    task_score_filters: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """
    通过 MCP 分页拉取全量数据.
    返回 (items, meta) 其中 meta 含 total / task_info / resolved_task_ids.
    """
    all_items: list[dict] = []
    task_info: dict = {}
    resolved_task_ids: list[int] = []
    page = 1
    total = 0

    body_base: dict[str, Any] = {
        "task_ids": task_ids,
        "page_size": page_size,
    }
    if exercise_version_ids:
        body_base["exercise_version_ids"] = exercise_version_ids
    if dimension_filter:
        body_base["dimension_filter"] = dimension_filter
    if task_score_filters:
        body_base["task_score_filters"] = task_score_filters

    while True:
        body = {**body_base, "page_index": page}
        resp = mcp_call("list_taiji_eval_insight_cases", body)
        data = resp.get("data", {})

        if page == 1:
            task_info = data.get("task_info", {})
            resolved_task_ids = data.get("resolved_task_ids", [])
            total = data.get("total", 0)
            print(f"[INFO] 共 {total} 条数据，开始拉取...", file=sys.stderr)

        items = data.get("data", [])
        if not items:
            break

        all_items.extend(items)
        print(f"[INFO] 第 {page} 页: {len(items)} 条，累计 {len(all_items)} 条", file=sys.stderr)

        if len(all_items) >= total or len(items) < page_size:
            break

        page += 1

    return all_items, {
        "total": total,
        "task_ids": task_ids,
        "task_info": task_info,
        "resolved_task_ids": resolved_task_ids,
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# -----------------------------
# 主入口
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insight Case 明细对比数据拉取（通过 MCP 工具调用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --task-ids 103674,103673 --output ./cases.jsonl
  %(prog)s --insight-id 2550 --output ./cases.jsonl
  %(prog)s --task-ids 103674,103673 --output ./cases.jsonl --dimension-filter '{"task_lv1":["数学"]}'
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--task-ids",
        help="任务 ID 列表，逗号分隔，如 103674,103673"
    )
    group.add_argument(
        "--insight-id", type=int,
        help="Insight ID，自动获取关联的 task_ids 并拉取对比数据"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="输出文件路径 (.jsonl)"
    )
    parser.add_argument(
        "--page-size", type=int, default=50,
        help="每页数量，默认 50"
    )
    parser.add_argument(
        "--dimension-filter", type=json.loads,
        help='维度过滤，JSON 字符串，如 \'{"task_lv1":["数学"]}\''
    )
    parser.add_argument(
        "--task-score-filters", type=json.loads,
        help='按 task 分数过滤，JSON 字符串'
    )
    parser.add_argument(
        "--exercise-version-ids",
        help="评测集版本 ID 列表，逗号分隔"
    )
    args = parser.parse_args()

    # 解析 task_ids（支持 --task-ids 和 --insight-id 两种方式）
    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
    else:
        task_ids = get_insight_task_ids(args.insight_id)
    if not task_ids:
        print("[ERROR] task_ids 不能为空", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Task IDs: {task_ids}", file=sys.stderr)

    # 拉取数据
    exercise_version_ids = None
    if args.exercise_version_ids:
        exercise_version_ids = [int(x.strip()) for x in args.exercise_version_ids.split(",") if x.strip()]

    start = time.time()
    items, meta = fetch_all_pages(
        task_ids,
        page_size=args.page_size,
        exercise_version_ids=exercise_version_ids,
        dimension_filter=args.dimension_filter,
        task_score_filters=args.task_score_filters,
    )
    elapsed = time.time() - start

    # 如果使用了 --insight-id，额外拉取 insight detail（weight_nodes + 基线信息）
    if args.insight_id:
        detail = fetch_insight_detail(args.insight_id)
        if detail:
            meta["insight_detail"] = detail
            print(f"[INFO] Insight detail 已获取: {len(detail.get('weight_nodes', []))} 个 weight_nodes", file=sys.stderr)

    # 写入 JSONL
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")

    file_size = output_path.stat().st_size
    print(f"[INFO] 完成！{len(items)} 条数据 → {output_path} ({file_size / 1024 / 1024:.1f} MB)，耗时 {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
