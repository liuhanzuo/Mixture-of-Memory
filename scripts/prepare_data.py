#!/usr/bin/env python3
"""
prepare_data.py — 从 HuggingFace 真实数据集下载并转换为 MoM 训练格式。

支持的数据源:
  - msc: Facebook Multi-Session Chat (多轮对话 + persona)
  - locomo: LoCoMo 长期对话记忆数据集 (需本地 JSON)
  - jsonl: 自定义 JSONL 文件

Usage::

    # 从 MSC 下载并生成 L2 + L3 训练数据
    python scripts/prepare_data.py --source msc --max-samples 2000

    # 从本地 LoCoMo JSON 转换
    python scripts/prepare_data.py --source locomo --locomo-path data/raw/locomo10.json

    # 指定输出目录
    python scripts/prepare_data.py --source msc --output-dir data/processed --max-samples 1000

    # 仅生成 L2 数据
    python scripts/prepare_data.py --source msc --target l2

    # 仅生成 L3 数据
    python scripts/prepare_data.py --source msc --target l3
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ======================================================================
# MSC (Multi-Session Chat) 数据集转换
# ======================================================================

# L2 object_type 关键词映射表 (用于从对话文本中启发式识别记忆类型)
PREFERENCE_KEYWORDS = [
    "prefer", "like", "love", "enjoy", "favorite", "favourite", "hate", "dislike",
    "always", "usually", "rather", "fond of", "passion", "interest",
    "喜欢", "偏好", "习惯", "爱好", "讨厌",
]
IDENTITY_KEYWORDS = [
    "my name", "i am", "i'm", "i work", "i live", "my job", "my age",
    "born in", "graduated", "married", "single",
    "我叫", "我是", "我住在", "我的工作", "我在",
]
TASK_KEYWORDS = [
    "working on", "planning to", "going to", "need to", "want to",
    "currently", "project", "deadline", "goal",
    "正在做", "计划", "打算", "项目",
]


def classify_utterance(text: str) -> str:
    """根据关键词启发式分类一条对话为 L2 object_type。"""
    text_lower = text.lower()
    for kw in IDENTITY_KEYWORDS:
        if kw in text_lower:
            return "entity"
    for kw in PREFERENCE_KEYWORDS:
        if kw in text_lower:
            return "preference"
    for kw in TASK_KEYWORDS:
        if kw in text_lower:
            return "task"
    return "topic"


def persona_to_profile_entries(personas: list[str]) -> list[dict[str, Any]]:
    """将 MSC 的 persona 描述转换为 L3 profile entries。"""
    entries = []
    for persona in personas:
        persona = persona.strip()
        if not persona:
            continue

        # 根据内容分类
        persona_lower = persona.lower()
        if any(kw in persona_lower for kw in ["prefer", "like", "love", "enjoy", "favorite", "hate"]):
            category = "preference"
            key = "preference"
        elif any(kw in persona_lower for kw in ["i am", "i'm", "my name", "i work", "i live"]):
            category = "identity"
            key = "identity"
        elif any(kw in persona_lower for kw in ["hobby", "interest", "study", "research"]):
            category = "research_interest"
            key = "interest"
        else:
            category = "factual"
            key = "factual_info"

        entries.append({
            "key": key,
            "value": persona,
            "confidence": 0.9,
            "category": category,
        })
    return entries


def utterances_to_l2_objects(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    """从对话消息中启发式提取 L2 记忆对象。

    策略: 对 user 发言进行关键词分类，生成对应的记忆对象。
    """
    objects = []
    seen_texts = set()  # 去重

    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg["content"].strip()
        if not content or len(content) < 10:
            continue

        # 简单去重 (避免重复的记忆对象)
        content_key = content[:50].lower()
        if content_key in seen_texts:
            continue
        seen_texts.add(content_key)

        obj_type = classify_utterance(content)
        # 截断过长的文本
        summary = content[:200] if len(content) > 200 else content

        objects.append({
            "object_type": obj_type,
            "summary_text": summary,
            "confidence": round(random.uniform(0.7, 0.95), 2),
        })

    return objects


def convert_msc_to_l2(
    dataset,
    max_samples: int = 500,
    min_turns: int = 4,
) -> list[dict[str, Any]]:
    """将 MSC 数据集转换为 L2 训练数据。

    Args:
        dataset: HuggingFace MSC dataset split
        max_samples: 最大样本数
        min_turns: 最少对话轮数

    Returns:
        L2 训练样本列表, 每条: {"messages": [...], "objects": [...]}
    """
    l2_samples = []

    for i, example in enumerate(dataset):
        if len(l2_samples) >= max_samples:
            break

        # MSC 格式: dialog 是一个列表，包含 [speaker1_utterance, speaker2_utterance, ...]
        dialog = example.get("dialog", example.get("dialogue", []))
        if not dialog:
            continue

        # 转为 messages 格式
        messages = []
        for j, utterance in enumerate(dialog):
            if isinstance(utterance, str):
                role = "user" if j % 2 == 0 else "assistant"
                messages.append({"role": role, "content": utterance})
            elif isinstance(utterance, dict):
                messages.append({
                    "role": utterance.get("role", "user" if j % 2 == 0 else "assistant"),
                    "content": utterance.get("text", utterance.get("content", "")),
                })

        if len(messages) < min_turns:
            continue

        # 提取记忆对象
        objects = utterances_to_l2_objects(messages)
        if not objects:
            continue

        l2_samples.append({
            "messages": messages,
            "objects": objects,
        })

    logger.info(f"[MSC→L2] 转换了 {len(l2_samples)} 条 L2 训练样本")
    return l2_samples


def convert_msc_to_l3(
    dataset,
    max_samples: int = 500,
) -> list[dict[str, Any]]:
    """将 MSC 数据集转换为 L3 训练数据。

    利用 MSC 中的 persona 标注作为 L3 profile 的 ground truth。

    Args:
        dataset: HuggingFace MSC dataset split
        max_samples: 最大样本数

    Returns:
        L3 训练样本列表, 每条: {"l2_objects": [...], "profile_entries": [...]}
    """
    l3_samples = []

    for i, example in enumerate(dataset):
        if len(l3_samples) >= max_samples:
            break

        # 提取 persona
        personas = example.get("personas", example.get("persona", []))
        # MSC 中 personas 可能是嵌套列表 (两个说话人各有一组)
        if personas and isinstance(personas[0], list):
            # 取第一个说话人的 persona
            personas = personas[0]

        if not personas:
            continue

        # 从对话中提取 L2 对象 (作为 L3 的输入)
        dialog = example.get("dialog", example.get("dialogue", []))
        messages = []
        for j, utterance in enumerate(dialog):
            if isinstance(utterance, str):
                role = "user" if j % 2 == 0 else "assistant"
                messages.append({"role": role, "content": utterance})
            elif isinstance(utterance, dict):
                messages.append({
                    "role": utterance.get("role", "user" if j % 2 == 0 else "assistant"),
                    "content": utterance.get("text", utterance.get("content", "")),
                })

        l2_objects = utterances_to_l2_objects(messages)
        if not l2_objects:
            continue

        # Persona → L3 profile entries
        profile_entries = persona_to_profile_entries(personas)
        if not profile_entries:
            continue

        l3_samples.append({
            "l2_objects": l2_objects,
            "profile_entries": profile_entries,
        })

    logger.info(f"[MSC→L3] 转换了 {len(l3_samples)} 条 L3 训练样本")
    return l3_samples


# ======================================================================
# LoCoMo 数据集转换
# ======================================================================

def convert_locomo_to_l2(
    locomo_data: list[dict],
    max_samples: int = 500,
) -> list[dict[str, Any]]:
    """将 LoCoMo 数据集转换为 L2 训练数据。"""
    l2_samples = []

    for conversation in locomo_data:
        if len(l2_samples) >= max_samples:
            break

        sessions = conversation.get("conversation", [])
        for session in sessions:
            if len(l2_samples) >= max_samples:
                break

            dialog = session.get("dialog", session.get("dialogue", []))
            if not dialog:
                continue

            messages = []
            for turn in dialog:
                if isinstance(turn, dict):
                    messages.append({
                        "role": turn.get("role", "user"),
                        "content": turn.get("text", turn.get("content", "")),
                    })
                elif isinstance(turn, str):
                    role = "user" if len(messages) % 2 == 0 else "assistant"
                    messages.append({"role": role, "content": turn})

            if len(messages) < 4:
                continue

            objects = utterances_to_l2_objects(messages)
            if objects:
                l2_samples.append({"messages": messages, "objects": objects})

    logger.info(f"[LoCoMo→L2] 转换了 {len(l2_samples)} 条 L2 训练样本")
    return l2_samples


def convert_locomo_to_l3(
    locomo_data: list[dict],
    max_samples: int = 500,
) -> list[dict[str, Any]]:
    """将 LoCoMo 数据集转换为 L3 训练数据。

    利用 LoCoMo 中的 observation/event_summary 作为 profile ground truth。
    """
    l3_samples = []

    for conversation in locomo_data:
        if len(l3_samples) >= max_samples:
            break

        # 收集所有 session 的 L2 对象
        all_l2_objects = []
        sessions = conversation.get("conversation", [])
        for session in sessions:
            dialog = session.get("dialog", session.get("dialogue", []))
            messages = []
            for turn in dialog:
                if isinstance(turn, dict):
                    messages.append({
                        "role": turn.get("role", "user"),
                        "content": turn.get("text", turn.get("content", "")),
                    })
            all_l2_objects.extend(utterances_to_l2_objects(messages))

        if not all_l2_objects:
            continue

        # 从 observation / event_summary 提取 profile
        observations = conversation.get("observation", conversation.get("observations", []))
        if isinstance(observations, str):
            observations = [observations]

        profile_entries = []
        for obs in observations:
            if isinstance(obs, str) and obs.strip():
                profile_entries.append({
                    "key": "observation",
                    "value": obs.strip(),
                    "confidence": 0.85,
                    "category": "factual",
                })
            elif isinstance(obs, dict):
                profile_entries.append({
                    "key": obs.get("key", "observation"),
                    "value": obs.get("value", obs.get("text", "")),
                    "confidence": obs.get("confidence", 0.85),
                    "category": obs.get("category", "factual"),
                })

        if profile_entries:
            l3_samples.append({
                "l2_objects": all_l2_objects[:10],  # 限制 L2 对象数量
                "profile_entries": profile_entries,
            })

    logger.info(f"[LoCoMo→L3] 转换了 {len(l3_samples)} 条 L3 训练样本")
    return l3_samples


# ======================================================================
# 保存工具函数
# ======================================================================

def save_jsonl(data: list[dict], path: Path, val_ratio: float = 0.1) -> tuple[int, int]:
    """保存数据为 JSONL 格式，并自动分出验证集。

    Returns:
        (train_count, val_count)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # 随机打乱
    random.shuffle(data)

    # 分割
    val_count = max(1, int(len(data) * val_ratio))
    train_data = data[val_count:]
    val_data = data[:val_count]

    # 保存训练集
    with open(path, "w", encoding="utf-8") as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 保存验证集
    val_path = path.with_name(path.stem.replace("train", "val") + path.suffix)
    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"[保存] 训练集: {len(train_data)} → {path}")
    logger.info(f"[保存] 验证集: {len(val_data)} → {val_path}")
    return len(train_data), len(val_data)


# ======================================================================
# 主入口
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="下载并转换真实数据集为 MoM 训练格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", type=str, default="msc",
        choices=["msc", "locomo"],
        help="数据源: msc=Multi-Session Chat, locomo=LoCoMo",
    )
    parser.add_argument(
        "--target", type=str, default="all",
        choices=["l2", "l3", "all"],
        help="目标: l2=仅L2, l3=仅L3, all=两者",
    )
    parser.add_argument("--max-samples", type=int, default=2000, help="每种数据的最大样本数")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # LoCoMo 专用
    parser.add_argument("--locomo-path", type=str, default=None, help="LoCoMo JSON 文件路径")

    # MSC 专用
    parser.add_argument("--msc-subset", type=str, default="session_1",
                        help="MSC 子集: session_1, session_2, session_3, session_4, session_5")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(name)-30s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  📦 MoM 数据准备工具")
    print(f"{'='*60}")
    print(f"  数据源: {args.source}")
    print(f"  目标: {args.target}")
    print(f"  最大样本数: {args.max_samples}")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*60}\n")

    # =========================================================
    # MSC 数据源
    # =========================================================
    if args.source == "msc":
        try:
            from datasets import load_dataset
        except ImportError:
            print("❌ 请先安装 datasets: pip install datasets")
            sys.exit(1)

        print("[INFO] 正在从 HuggingFace 下载 MSC 数据集...")
        print("[INFO] (首次下载可能需要几分钟)")

        try:
            dataset = load_dataset("facebook/msc", args.msc_subset, split="train",
                                   trust_remote_code=True)
            print(f"[OK] MSC {args.msc_subset} 加载成功: {len(dataset)} 条对话")
        except Exception as e:
            print(f"❌ MSC 数据集加载失败: {e}")
            print("\n[提示] 如果无法访问 HuggingFace，请尝试:")
            print("  1. 设置代理: export HF_ENDPOINT=https://hf-mirror.com")
            print("  2. 手动下载后使用 --source locomo --locomo-path <path>")
            sys.exit(1)

        if args.target in ("l2", "all"):
            print("\n[INFO] 转换 L2 训练数据...")
            l2_data = convert_msc_to_l2(dataset, max_samples=args.max_samples)
            if l2_data:
                train_n, val_n = save_jsonl(l2_data, output_dir / "l2_train.jsonl", args.val_ratio)
                print(f"  ✅ L2 数据: 训练={train_n}, 验证={val_n}")
            else:
                print("  ⚠️ 未能生成 L2 数据")

        if args.target in ("l3", "all"):
            print("\n[INFO] 转换 L3 训练数据...")
            l3_data = convert_msc_to_l3(dataset, max_samples=args.max_samples)
            if l3_data:
                train_n, val_n = save_jsonl(l3_data, output_dir / "l3_train.jsonl", args.val_ratio)
                print(f"  ✅ L3 数据: 训练={train_n}, 验证={val_n}")
            else:
                print("  ⚠️ 未能生成 L3 数据")

    # =========================================================
    # LoCoMo 数据源
    # =========================================================
    elif args.source == "locomo":
        locomo_path = args.locomo_path
        if not locomo_path:
            # 尝试默认路径
            default_paths = [
                Path("data/raw/locomo10.json"),
                Path("data/raw/locomo.json"),
                _PROJECT_ROOT / "data" / "raw" / "locomo10.json",
            ]
            for p in default_paths:
                if p.exists():
                    locomo_path = str(p)
                    break

        if not locomo_path or not Path(locomo_path).exists():
            print("❌ LoCoMo 数据文件不存在。请先下载:")
            print("   git clone https://github.com/snap-research/locomo.git")
            print("   cp locomo/data/locomo10.json data/raw/")
            print("   或指定路径: --locomo-path <path>")
            sys.exit(1)

        print(f"[INFO] 加载 LoCoMo 数据: {locomo_path}")
        with open(locomo_path, "r", encoding="utf-8") as f:
            locomo_data = json.load(f)

        if isinstance(locomo_data, dict):
            # 某些版本的 LoCoMo 数据是 dict 格式
            locomo_data = list(locomo_data.values()) if not isinstance(list(locomo_data.values())[0], dict) else [locomo_data]

        print(f"[OK] LoCoMo 加载成功: {len(locomo_data)} 条对话")

        if args.target in ("l2", "all"):
            print("\n[INFO] 转换 L2 训练数据...")
            l2_data = convert_locomo_to_l2(locomo_data, max_samples=args.max_samples)
            if l2_data:
                train_n, val_n = save_jsonl(l2_data, output_dir / "l2_train.jsonl", args.val_ratio)
                print(f"  ✅ L2 数据: 训练={train_n}, 验证={val_n}")

        if args.target in ("l3", "all"):
            print("\n[INFO] 转换 L3 训练数据...")
            l3_data = convert_locomo_to_l3(locomo_data, max_samples=args.max_samples)
            if l3_data:
                train_n, val_n = save_jsonl(l3_data, output_dir / "l3_train.jsonl", args.val_ratio)
                print(f"  ✅ L3 数据: 训练={train_n}, 验证={val_n}")

    print(f"\n{'='*60}")
    print(f"  ✅ 数据准备完成！")
    print(f"  📁 输出目录: {output_dir}")
    print(f"{'='*60}")
    print(f"\n  后续使用:")
    print(f"    python scripts/train_l2.py --train-data {output_dir}/l2_train.jsonl --val-data {output_dir}/l2_val.jsonl")
    print(f"    python scripts/train_l3.py --train-data {output_dir}/l3_train.jsonl --val-data {output_dir}/l3_val.jsonl")
    print(f"\n  或直接在训练时使用 --dataset msc 自动加载:")
    print(f"    python scripts/train_l2.py --dataset msc")
    print(f"    python scripts/train_l3.py --dataset msc")


if __name__ == "__main__":
    main()
