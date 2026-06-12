#!/usr/bin/env python3
"""
build_l3_from_l2.py — 从 L2 记忆对象生成 L3 画像/语义记忆。

从 build_l2_from_messages.py 产出的 L2 对象 JSON 中,
使用 L3 Summarizer 生成长期画像条目, 并通过 L3 Reviser 处理冲突,
最终导出 Markdown 和 JSON 格式的画像文件。

Usage::

    python scripts/build_l3_from_l2.py --input data/processed/l2_objects.json --output-dir data/processed/
    python scripts/build_l3_from_l2.py --demo
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.memory.l2.types import L2MemoryObject
from src.memory.l3.summarizer import L3Summarizer, L3ProfileEntry
from src.memory.l3.profile_store import L3ProfileStore
from src.memory.l3.reviser import L3Reviser
from src.memory.l3.formatter import L3Formatter
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def load_l2_objects(input_path: str) -> list[L2MemoryObject]:
    """从 JSON 文件加载 L2 对象。"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    objects_data = data.get("objects", [])
    objects: list[L2MemoryObject] = []

    for obj_dict in objects_data:
        obj = L2MemoryObject(
            object_id=obj_dict.get("object_id", ""),
            object_type=obj_dict.get("object_type", ""),
            summary_text=obj_dict.get("summary_text", ""),
            confidence=obj_dict.get("confidence", 1.0),
            source_turn_ids=obj_dict.get("source_turn_ids", []),
            created_at=obj_dict.get("created_at", ""),
            last_updated_at=obj_dict.get("last_updated_at", ""),
            metadata=obj_dict.get("metadata", {}),
            is_archived=obj_dict.get("is_archived", False),
        )
        objects.append(obj)

    logger.info(f"加载了 {len(objects)} 个 L2 对象 from {input_path}")
    return objects


# ---- 内置 demo L2 对象 ----
def create_demo_l2_objects() -> list[L2MemoryObject]:
    """创建 demo L2 对象 (模拟 build_l2_from_messages 的输出)。"""
    return [
        L2MemoryObject(
            object_id="demo_topic_001",
            object_type="topic",
            summary_text="Discussion topic: 大语言模型稀疏化训练, SparseGPT, Wanda对比",
            confidence=0.7,
            source_turn_ids=["t001", "t002"],
        ),
        L2MemoryObject(
            object_id="demo_pref_001",
            object_type="preference",
            summary_text="technical, structured explanations",
            confidence=0.8,
            source_turn_ids=["t003"],
        ),
        L2MemoryObject(
            object_id="demo_task_001",
            object_type="task",
            summary_text="MoM-Agent项目, 层次化Agent记忆系统",
            confidence=0.9,
            source_turn_ids=["t004", "t005"],
        ),
        L2MemoryObject(
            object_id="demo_state_001",
            object_type="state",
            summary_text="working on hierarchical memory for local-attention agents",
            confidence=0.75,
            source_turn_ids=["t006"],
        ),
        L2MemoryObject(
            object_id="demo_topic_002",
            object_type="topic",
            summary_text="Discussion topic: FSDP分布式训练, HYBRID_SHARD策略, NCCL通信优化",
            confidence=0.65,
            source_turn_ids=["t007", "t008"],
        ),
        L2MemoryObject(
            object_id="demo_pref_002",
            object_type="preference",
            summary_text="Python and PyTorch for deep learning research",
            confidence=0.85,
            source_turn_ids=["t009"],
        ),
    ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build L3 Profile from L2 Objects")
    parser.add_argument("--input", type=str, default=None, help="L2 对象 JSON 文件路径")
    parser.add_argument("--output-dir", type=str, default="data/processed/", help="输出目录")
    parser.add_argument("--demo", action="store_true", help="使用内置 demo 数据")
    parser.add_argument("--max-entries", type=int, default=100, help="L3 store 最大条目数")
    parser.add_argument("--conflict-threshold", type=float, default=0.8, help="冲突检测阈值")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    # 加载 L2 对象
    if args.demo:
        logger.info("使用内置 demo L2 对象")
        l2_objects = create_demo_l2_objects()
    elif args.input:
        l2_objects = load_l2_objects(args.input)
    else:
        print("❌ 请指定 --input 或 --demo 参数。")
        sys.exit(1)

    if not l2_objects:
        print("❌ 没有可处理的 L2 对象。")
        sys.exit(1)

    # 构建 L3 组件
    summarizer = L3Summarizer(config={"summarizer_backend": "rule_based"})
    store = L3ProfileStore(max_entries=args.max_entries)
    reviser = L3Reviser(conflict_threshold=args.conflict_threshold)
    formatter = L3Formatter()

    # L3 总结
    logger.info(f"从 {len(l2_objects)} 个 L2 对象生成 L3 profile entries...")
    new_entries = summarizer.summarize(l2_objects)
    logger.info(f"生成了 {len(new_entries)} 条 L3 entries")

    # 修订 & 写入 store
    revised_count = 0
    for entry in new_entries:
        was_revised = reviser.revise_or_add(entry, store)
        if was_revised:
            revised_count += 1

    logger.info(f"修订: {revised_count}/{len(new_entries)} 条 entries 触发了修订/合并")

    # 导出
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Markdown
    md_path = output_dir / "profile.md"
    formatter.export_markdown(store, str(md_path))

    # JSON
    json_path = output_dir / "profile.json"
    formatter.export_json(store, str(json_path))

    # 汇总文本
    summary_text = formatter.format_summary(store)

    # 打印结果
    print(f"\n✅ L3 画像构建完成:")
    print(f"   输入 L2 对象数: {len(l2_objects)}")
    print(f"   生成 L3 entries: {len(new_entries)}")
    print(f"   修订合并数: {revised_count}")
    print(f"   最终 store 大小: {len(store)}")
    print(f"   Markdown 导出: {md_path}")
    print(f"   JSON 导出: {json_path}")

    print(f"\n📋 画像摘要:")
    print(summary_text)

    # 详细列表
    all_entries = store.get_all()
    print(f"\n📝 L3 条目列表:")
    for entry in all_entries:
        print(f"   [{entry.category}] {entry.key}: {entry.value[:80]}... (conf={entry.confidence:.2f})")


if __name__ == "__main__":
    main()
