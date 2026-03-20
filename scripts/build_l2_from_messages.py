#!/usr/bin/env python3
"""
build_l2_from_messages.py — 从对话消息构建 L2 记忆对象。

从 JSON 格式的对话消息文件中, 使用 L2 Aggregator 提取记忆对象,
并保存到 JSON 文件中。

支持两种输入格式:
1. 单个对话: {"messages": [{"role": "user", "content": "..."}, ...]}
2. 多个对话: [{"session_id": "s1", "messages": [...]}, ...]

Usage::

    python scripts/build_l2_from_messages.py --input data/raw/conversations.json --output data/processed/l2_objects.json
    python scripts/build_l2_from_messages.py --demo  # 使用内置 demo 数据
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

from src.memory.l2.aggregator import L2Aggregator
from src.memory.l2.object_store import L2ObjectStore
from src.memory.l2.merger import L2Merger
from src.memory.l2.types import ChatMessage, L2MemoryObject
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# ---- 内置 demo 对话 ----
DEMO_MESSAGES = [
    {"role": "user", "content": "我最近在研究大语言模型的稀疏化训练。", "turn_id": "t001"},
    {"role": "assistant", "content": "稀疏化训练是一个很有前景的方向。", "turn_id": "t001"},
    {"role": "user", "content": "I prefer technical, structured explanations.", "turn_id": "t002"},
    {"role": "assistant", "content": "好的，我会用结构化的方式回答。", "turn_id": "t002"},
    {"role": "user", "content": "我现在在做MoM-Agent项目，帮我记一下。", "turn_id": "t003"},
    {"role": "assistant", "content": "已记录，MoM-Agent项目。", "turn_id": "t003"},
    {"role": "user", "content": "I am currently working on hierarchical memory for agents.", "turn_id": "t004"},
    {"role": "assistant", "content": "了解，层次化Agent记忆。", "turn_id": "t004"},
    {"role": "user", "content": "帮我解释一下Flash Attention的原理。", "turn_id": "t005"},
    {"role": "assistant", "content": "Flash Attention通过分块计算来减少IO开销...", "turn_id": "t005"},
]


def load_messages(input_path: str) -> list[list[dict[str, Any]]]:
    """加载消息数据, 返回按会话分组的消息列表。"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # 单个对话
        return [data.get("messages", [])]
    elif isinstance(data, list):
        if data and isinstance(data[0], dict) and "messages" in data[0]:
            # 多个对话
            return [conv.get("messages", []) for conv in data]
        else:
            # 直接是消息列表
            return [data]
    else:
        logger.error(f"无法解析的输入格式: {type(data)}")
        return []


def messages_to_chat(messages: list[dict[str, Any]]) -> list[ChatMessage]:
    """将字典消息转为 ChatMessage。"""
    chat_msgs = []
    for msg in messages:
        chat_msgs.append(ChatMessage(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            turn_id=msg.get("turn_id", ""),
            timestamp=msg.get("timestamp", ""),
        ))
    return chat_msgs


def objects_to_dicts(objects: list[L2MemoryObject]) -> list[dict[str, Any]]:
    """将 L2MemoryObject 列表转为可序列化字典列表。"""
    result = []
    for obj in objects:
        result.append({
            "object_id": obj.object_id,
            "object_type": obj.object_type,
            "summary_text": obj.summary_text,
            "confidence": obj.confidence,
            "source_turn_ids": obj.source_turn_ids,
            "created_at": obj.created_at,
            "last_updated_at": obj.last_updated_at,
            "metadata": obj.metadata,
            "is_archived": obj.is_archived,
        })
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build L2 Memory Objects from Messages")
    parser.add_argument("--input", type=str, default=None, help="输入 JSON 文件路径")
    parser.add_argument("--output", type=str, default="data/processed/l2_objects.json", help="输出 JSON 文件路径")
    parser.add_argument("--demo", action="store_true", help="使用内置 demo 数据")
    parser.add_argument("--chunk-size", type=int, default=5, help="每个 chunk 的消息数")
    parser.add_argument("--similarity-threshold", type=float, default=0.75, help="合并相似度阈值")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    # 加载消息
    if args.demo:
        logger.info("使用内置 demo 数据")
        all_sessions = [DEMO_MESSAGES]
    elif args.input:
        logger.info(f"从文件加载: {args.input}")
        all_sessions = load_messages(args.input)
    else:
        print("❌ 请指定 --input 或 --demo 参数。")
        sys.exit(1)

    # 构建 L2 组件
    aggregator = L2Aggregator(config={"aggregator_backend": "rule_based"})
    store = L2ObjectStore(max_objects=200)
    merger = L2Merger(similarity_threshold=args.similarity_threshold)

    total_messages = 0
    total_objects = 0

    for session_idx, messages in enumerate(all_sessions):
        logger.info(f"处理会话 {session_idx + 1}/{len(all_sessions)}, {len(messages)} 条消息")
        total_messages += len(messages)

        # 按 chunk_size 分块聚合
        chat_msgs = messages_to_chat(messages)
        for i in range(0, len(chat_msgs), args.chunk_size):
            chunk = chat_msgs[i:i + args.chunk_size]
            new_objects = aggregator.aggregate(chunk)

            for obj in new_objects:
                merged = merger.merge_or_add(obj, store)
                if not merged:
                    pass  # merge_or_add 内部已处理 add

            total_objects += len(new_objects)

    # 获取最终的 store 内容
    all_objects = store.get_all()

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "total_sessions": len(all_sessions),
        "total_messages": total_messages,
        "total_extracted_objects": total_objects,
        "final_store_size": len(all_objects),
        "objects": objects_to_dicts(all_objects),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ L2 记忆对象构建完成:")
    print(f"   会话数: {len(all_sessions)}")
    print(f"   消息数: {total_messages}")
    print(f"   提取对象数: {total_objects}")
    print(f"   合并后 store 大小: {len(all_objects)}")
    print(f"   输出文件: {output_path}")

    # 打印对象摘要
    print(f"\n📋 L2 对象列表:")
    for obj in all_objects[:20]:
        print(f"   [{obj.object_type}] {obj.summary_text[:80]}... (conf={obj.confidence:.2f})")
    if len(all_objects) > 20:
        print(f"   ... 共 {len(all_objects)} 个对象")


if __name__ == "__main__":
    main()
