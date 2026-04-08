#!/usr/bin/env python3
"""预处理脚本: 将 MAG 格式数据因果化 (Causal Memory Training)

用法:
    python scripts/prepare_causal_data.py \
        --input data/mag_train_generated.jsonl \
        --output data/mag_train_causal.jsonl \
        --num_memories 15 --num_hard_negatives 3

说明:
    将对话前文的原始轮次作为记忆, 使 target 天然依赖记忆内容.
    这样训练时只需要纯 NTP loss, 不需要额外的 gap/contrastive loss.
"""

import argparse
import json
import logging
import random
import time
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="因果化 MAG 训练数据")
    parser.add_argument("--input", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--num_memories", type=int, default=15, help="每个样本的候选记忆总数")
    parser.add_argument("--num_hard_negatives", type=int, default=3, help="硬负例数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    # ========== 1. 加载数据 ==========
    logger.info(f"加载数据: {args.input}")
    t0 = time.time()
    data = []
    n_parse_err = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if isinstance(d, dict) and "input_text" in d and "memory_texts" in d:
                    data.append(d)
            except json.JSONDecodeError:
                n_parse_err += 1
    t1 = time.time()
    logger.info(f"加载完成: {len(data)} 条有效数据 ({n_parse_err} 条解析失败), 耗时 {t1-t0:.1f}s")

    # ========== 2. 按 dialogue_id + session_idx 分组 ==========
    logger.info("按对话分组...")
    dialogue_groups: dict[str, list[dict]] = defaultdict(list)
    for d in data:
        dial_id = d.get("dialogue_id", "unknown")
        sess_idx = d.get("session_idx", 0)
        key = f"{dial_id}_sess{sess_idx}"
        dialogue_groups[key].append(d)

    group_keys_list = list(dialogue_groups.keys())
    logger.info(f"共 {len(group_keys_list)} 个对话组")

    # 统计组大小分布
    sizes = [len(g) for g in dialogue_groups.values()]
    logger.info(f"组大小: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")

    # ========== 3. 预构建每个组的轮次文本 ==========
    logger.info("构建轮次文本...")
    group_turn_texts: dict[str, list[str]] = {}
    for group_key, group in dialogue_groups.items():
        turns = []
        for d in group:
            turn_text = f"用户: {d['input_text']}"
            if d.get("target_text"):
                turn_text += f" 助手: {d['target_text']}"
            turns.append(turn_text)
        group_turn_texts[group_key] = turns

    # ========== 4. 因果化构造 ==========
    logger.info("开始因果化构造...")
    t2 = time.time()

    group_key_to_idx = {k: i for i, k in enumerate(group_keys_list)}
    samples = []
    n_skipped = 0

    for group_key, group in dialogue_groups.items():
        if len(group) < 2:
            n_skipped += 1
            continue

        turns = group_turn_texts[group_key]
        my_idx = group_key_to_idx[group_key]

        for t in range(1, len(group)):
            current = group[t]
            if not current.get("target_text"):
                continue

            query = current["input_text"]
            target = current["target_text"]

            # 相关记忆: 前面轮次的对话原文
            causal_memories = turns[:t]
            max_causal = args.num_memories - args.num_hard_negatives
            if len(causal_memories) > max_causal:
                causal_memories = causal_memories[-max_causal:]

            relevant_count = len(causal_memories)

            # 硬负例: 按组索引随机采样
            hard_negs = []
            n_hard = min(args.num_hard_negatives, len(group_keys_list) - 1)
            if n_hard > 0:
                neg_indices = []
                while len(neg_indices) < n_hard:
                    idx = random.randint(0, len(group_keys_list) - 1)
                    if idx != my_idx:
                        neg_indices.append(idx)
                for gi in neg_indices:
                    neg_turns = group_turn_texts[group_keys_list[gi]]
                    if neg_turns:
                        hard_negs.append(neg_turns[random.randint(0, len(neg_turns) - 1)])

            # 组装 + 打乱
            all_memories = causal_memories + hard_negs
            relevant_indices = list(range(relevant_count))

            indices = list(range(len(all_memories)))
            random.shuffle(indices)
            shuffled_memories = [all_memories[i] for i in indices]
            index_map = {old: new for new, old in enumerate(indices)}
            shuffled_relevant = [index_map[r] for r in relevant_indices]

            samples.append({
                "input_text": query,
                "target_text": target,
                "memory_texts": shuffled_memories,
                "relevant_indices": shuffled_relevant,
                "dialogue_id": current.get("dialogue_id", "unknown"),
                "session_idx": current.get("session_idx", 0),
            })

    random.shuffle(samples)
    t3 = time.time()

    avg_mem = sum(len(s["memory_texts"]) for s in samples) / max(len(samples), 1)
    avg_rel = sum(len(s["relevant_indices"]) for s in samples) / max(len(samples), 1)
    logger.info(
        f"因果化完成: {len(data)} → {len(samples)} 条样本 "
        f"(跳过 {n_skipped} 个过短对话组), 耗时 {t3-t2:.1f}s"
    )
    logger.info(f"  平均 {avg_mem:.1f} 条记忆, {avg_rel:.1f} 条相关")

    # ========== 5. 保存 ==========
    logger.info(f"保存到: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"完成! 共 {len(samples)} 条因果化训练样本")


if __name__ == "__main__":
    main()
