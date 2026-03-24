#!/usr/bin/env python3
"""
LoCoMo → MAG JSONL 转换脚本

将 LoCoMo 长对话数据集转换为 MAG 训练格式。
核心策略:
  1. 跨 session 记忆: 当前 session 的回复需要依赖之前 session 的对话历史作为记忆
  2. QA evidence: 利用 LoCoMo 自带的 QA 标注 + evidence (dia_id) 构造精确的记忆标签
  3. Observation: 利用 LoCoMo 生成的 observation 作为额外记忆候选

输出格式 (每行一个 JSON):
{
    "input_text": "当前用户发言 (query)",
    "target_text": "对应的回复 (Phase 2 预测目标)",
    "memory_texts": ["记忆候选1", "记忆候选2", ...],
    "relevant_indices": [0, 2, 5]  // 相关记忆的索引
}

用法:
    python scripts/convert_locomo.py \
        --input locomo/data/locomo10.json \
        --output data/raw/locomo_train.jsonl \
        --num_memories 15 \
        --strategy cross_session
"""

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("convert_locomo")


def parse_locomo(data_path: str) -> list[dict]:
    """解析 LoCoMo JSON 文件, 返回结构化的对话数据。

    Returns:
        list of dicts, 每个 dict 包含:
        - sample_id: 样本ID
        - speaker_a, speaker_b: 两个说话人名字
        - sessions: [{"session_id": "session_1", "date_time": "...", "turns": [...]}]
        - dia_id_map: {dia_id: turn_dict}  用于 evidence 查找
        - qa: QA 标注列表
        - observations: {session_key: [obs_list]}
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw = list(raw.values())

    parsed = []
    for conv in raw:
        conv_data = conv.get("conversation", {})
        if not isinstance(conv_data, dict):
            logger.warning(f"跳过非法 conversation 字段: type={type(conv_data)}")
            continue

        speaker_a = conv_data.get("speaker_a", "Speaker A")
        speaker_b = conv_data.get("speaker_b", "Speaker B")

        # 提取所有 session 并排序
        session_keys = sorted(
            [k for k in conv_data.keys() if re.match(r"^session_\d+$", k)],
            key=lambda k: int(k.split("_")[1])
        )

        sessions = []
        dia_id_map = {}  # dia_id -> {"speaker": ..., "text": ..., "session_idx": ...}

        for si, sk in enumerate(session_keys):
            turns_raw = conv_data[sk]
            if not isinstance(turns_raw, list):
                continue

            date_key = f"{sk}_date_time"
            date_time = conv_data.get(date_key, "")

            turns = []
            for turn in turns_raw:
                if not isinstance(turn, dict) or "text" not in turn:
                    continue
                text = turn["text"].strip()
                if not text:
                    continue
                dia_id = turn.get("dia_id", "")
                t = {
                    "speaker": turn.get("speaker", ""),
                    "text": text,
                    "dia_id": dia_id,
                    "session_idx": si,
                    "session_id": sk,
                }
                turns.append(t)
                if dia_id:
                    dia_id_map[dia_id] = t

            sessions.append({
                "session_id": sk,
                "session_idx": si,
                "date_time": date_time,
                "turns": turns,
            })

        # 提取 observations
        # 结构: conv["observation"] = {
        #   "session_1_observation": {"SpeakerA": [[text, dia_id], ...], "SpeakerB": [...]},
        #   ...
        # }
        observations = {}  # {session_num: [obs_text, ...]}
        obs_raw = conv.get("observation", {})
        if isinstance(obs_raw, dict):
            for obs_key, speaker_dict in obs_raw.items():
                match = re.search(r"session_(\d+)", obs_key)
                if not match or not isinstance(speaker_dict, dict):
                    continue
                session_num = int(match.group(1))
                obs_texts = []
                for speaker_name, obs_list in speaker_dict.items():
                    if isinstance(obs_list, list):
                        for item in obs_list:
                            if isinstance(item, list) and len(item) >= 1:
                                text = item[0].strip() if isinstance(item[0], str) else str(item[0])
                                if text:
                                    obs_texts.append(text)
                            elif isinstance(item, str) and item.strip():
                                obs_texts.append(item.strip())
                if obs_texts:
                    observations[session_num] = obs_texts

        # 提取 QA 标注
        qa_list = conv.get("qa", [])

        parsed.append({
            "sample_id": conv.get("sample_id", f"conv_{len(parsed)}"),
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "sessions": sessions,
            "dia_id_map": dia_id_map,
            "qa": qa_list,
            "observations": observations,
        })

    logger.info(f"解析完成: {len(parsed)} 个对话, "
                f"共 {sum(len(c['sessions']) for c in parsed)} 个 session, "
                f"共 {sum(sum(len(s['turns']) for s in c['sessions']) for c in parsed)} 轮对话, "
                f"共 {sum(len(c['qa']) for c in parsed)} 个 QA 标注")
    return parsed


def build_cross_session_samples(
    conv: dict,
    num_memories: int = 15,
    num_hard_negatives: int = 3,
    min_history_sessions: int = 1,
    all_utterances: list[str] | None = None,
) -> list[dict]:
    """策略 1: 跨 session 记忆构建。

    对于第 N 个 session 中的每个 turn:
    - query = 当前 user 发言
    - target = 对应的另一个说话人的回复
    - 相关记忆 = 前 N-1 个 session 中的 turn (历史对话片段)
    - 硬负例 = 同 session 中的其他 turn
    - 随机负例 = 其他对话的 turn

    关键: 只为 session >= 2 的轮次生成样本, 这样模型必须依赖记忆
    """
    sessions = conv["sessions"]
    speaker_a = conv["speaker_a"]

    if len(sessions) < 2:
        return []

    samples = []

    for si in range(min_history_sessions, len(sessions)):
        current_session = sessions[si]
        turns = current_session["turns"]

        if len(turns) < 4:
            continue

        # 收集所有历史 session 的 turn 文本 (作为记忆池)
        history_turns = []
        for prev_si in range(si):
            for t in sessions[prev_si]["turns"]:
                text = t["text"]
                if len(text) >= 10:
                    history_turns.append(text)

        if not history_turns:
            continue

        # 收集当前 session 的所有 turn (用于硬负例)
        current_all_texts = [t["text"] for t in turns if len(t["text"]) >= 10]

        # 遍历当前 session 中的 turn pair
        i = 0
        while i < len(turns) - 1:
            speaker_turn = turns[i]
            reply_turn = turns[i + 1]

            # 确保是一个 "问-答" 对
            if speaker_turn["speaker"] == reply_turn["speaker"]:
                i += 1
                continue

            query = speaker_turn["text"]
            target = reply_turn["text"]

            if len(query) < 5 or len(target) < 5:
                i += 2
                continue

            # --- 相关记忆: 从历史 session 中选取 ---
            # 优先选最近的历史, 再随机补充
            max_relevant = min(len(history_turns), num_memories // 2)
            if max_relevant <= 0:
                i += 2
                continue

            # 最近的历史优先 (取后 max_relevant 条)
            if len(history_turns) <= max_relevant:
                relevant_memories = list(history_turns)
            else:
                # 一半取最近, 一半随机
                half = max(1, max_relevant // 2)
                recent = history_turns[-half:]
                older = random.sample(history_turns[:-half], min(max_relevant - half, len(history_turns[:-half])))
                relevant_memories = recent + older

            # --- 硬负例: 当前 session 中的其他 turn ---
            hard_neg_candidates = [
                t for t in current_all_texts
                if t != query and t != target
            ]
            hard_negs = random.sample(
                hard_neg_candidates,
                min(num_hard_negatives, len(hard_neg_candidates))
            ) if hard_neg_candidates else []

            # --- 随机负例: 其他对话的 turn ---
            remaining_slots = num_memories - len(relevant_memories) - len(hard_negs)
            random_negs = []
            if remaining_slots > 0 and all_utterances:
                # 过滤掉当前对话的发言
                avoid_set = set(relevant_memories + hard_negs + [query, target])
                candidates = [u for u in all_utterances if u not in avoid_set]
                if candidates:
                    random_negs = random.sample(
                        candidates,
                        min(remaining_slots, len(candidates))
                    )

            # 组装
            all_memories = relevant_memories + hard_negs + random_negs
            relevant_indices = list(range(len(relevant_memories)))

            # 打乱
            indices = list(range(len(all_memories)))
            random.shuffle(indices)
            shuffled_memories = [all_memories[idx] for idx in indices]
            shuffled_relevant = sorted([indices.index(r) for r in relevant_indices])

            samples.append({
                "input_text": query,
                "target_text": target,
                "memory_texts": shuffled_memories,
                "relevant_indices": shuffled_relevant,
            })

            i += 2  # 跳过已处理的 pair

        # 将当前 session 的 turn 也加入历史 (供后续 session 使用)
        # (不需要, 因为外层循环 si 递增, 历史会自动增长)

    return samples


def build_qa_evidence_samples(
    conv: dict,
    num_memories: int = 15,
    num_hard_negatives: int = 3,
    all_utterances: list[str] | None = None,
) -> list[dict]:
    """策略 2: 利用 QA evidence 标注构建精确的记忆标签。

    LoCoMo 的 QA 标注包含:
    - question: 需要回忆的问题
    - answer: 正确答案
    - evidence: [dia_id, ...] 指向包含答案的对话 turn

    转换:
    - input_text = question
    - target_text = answer
    - 相关记忆 = evidence 指向的对话 turn 的 text
    - 负例 = 其他随机 turn
    """
    dia_id_map = conv["dia_id_map"]
    qa_list = conv["qa"]
    sessions = conv["sessions"]

    if not qa_list:
        return []

    # 收集所有 turn 文本用于负例
    all_conv_turns = []
    for s in sessions:
        for t in s["turns"]:
            if len(t["text"]) >= 10:
                all_conv_turns.append(t["text"])

    samples = []

    for qa in qa_list:
        question = qa.get("question", "").strip()
        answer = qa.get("answer", "")
        if isinstance(answer, (int, float)):
            answer = str(answer)
        answer = answer.strip() if isinstance(answer, str) else str(answer)
        evidence_ids = qa.get("evidence", [])

        if not question or not answer:
            continue

        # 解析 evidence dia_ids, 找到对应的 turn text
        evidence_texts = []
        for eid_raw in evidence_ids:
            # evidence 可能是 "D1:3" 或 "D8:9; D9:17" 格式
            if isinstance(eid_raw, str):
                for eid in re.split(r"[;,]\s*", eid_raw):
                    eid = eid.strip()
                    if eid in dia_id_map:
                        text = dia_id_map[eid]["text"]
                        if text not in evidence_texts:
                            evidence_texts.append(text)

        if not evidence_texts:
            continue  # 没有 evidence 的 QA 跳过

        # 相关记忆 = evidence turn texts
        relevant_memories = evidence_texts[:num_memories // 2]

        # 硬负例: 同对话中非 evidence 的 turn
        evidence_set = set(evidence_texts)
        hard_neg_candidates = [t for t in all_conv_turns if t not in evidence_set and t != question]
        hard_negs = random.sample(
            hard_neg_candidates,
            min(num_hard_negatives, len(hard_neg_candidates))
        ) if hard_neg_candidates else []

        # 随机负例
        remaining_slots = num_memories - len(relevant_memories) - len(hard_negs)
        random_negs = []
        if remaining_slots > 0 and all_utterances:
            avoid_set = set(relevant_memories + hard_negs + [question, answer])
            candidates = [u for u in all_utterances if u not in avoid_set]
            if candidates:
                random_negs = random.sample(
                    candidates,
                    min(remaining_slots, len(candidates))
                )

        # 组装
        all_memories = relevant_memories + hard_negs + random_negs
        relevant_indices = list(range(len(relevant_memories)))

        # 打乱
        indices = list(range(len(all_memories)))
        random.shuffle(indices)
        shuffled_memories = [all_memories[idx] for idx in indices]
        shuffled_relevant = sorted([indices.index(r) for r in relevant_indices])

        samples.append({
            "input_text": question,
            "target_text": answer,
            "memory_texts": shuffled_memories,
            "relevant_indices": shuffled_relevant,
        })

    return samples


def build_observation_samples(
    conv: dict,
    num_memories: int = 15,
    num_hard_negatives: int = 3,
    all_utterances: list[str] | None = None,
) -> list[dict]:
    """策略 3: 利用 observation 构建 "事实回忆" 训练样本。

    思路: observation 是对某个 session 的事实总结,
    后续 session 中的对话可能需要这些事实才能合理回复。

    - 对于 session N 中的每个 turn pair, 将之前 session 的 observations 作为记忆候选
    - 与 cross_session 类似, 但使用 observation (更精炼) 而非原始 turn
    """
    sessions = conv["sessions"]
    observations = conv["observations"]

    if len(sessions) < 2 or not observations:
        return []

    # observations 已经是 {session_num: [obs_text, ...]} 格式
    obs_by_session = observations

    samples = []

    for si in range(1, len(sessions)):
        current_session = sessions[si]
        turns = current_session["turns"]

        if len(turns) < 4:
            continue

        # 收集之前所有 session 的 observations
        history_obs = []
        for prev_si in range(si):
            session_num = prev_si + 1  # session_1, session_2, ...
            if session_num in obs_by_session:
                history_obs.extend(obs_by_session[session_num])

        if not history_obs:
            continue

        # 每个 turn pair
        i = 0
        while i < len(turns) - 1:
            if turns[i]["speaker"] == turns[i + 1]["speaker"]:
                i += 1
                continue

            query = turns[i]["text"]
            target = turns[i + 1]["text"]

            if len(query) < 5 or len(target) < 5:
                i += 2
                continue

            # 相关记忆 = 历史 observations (精炼事实)
            max_relevant = min(len(history_obs), num_memories // 2)
            relevant_memories = random.sample(history_obs, max_relevant) if len(history_obs) > max_relevant else list(history_obs)

            # 硬负例 + 随机负例
            current_texts = [t["text"] for t in turns if len(t["text"]) >= 10 and t["text"] != query and t["text"] != target]
            hard_negs = random.sample(
                current_texts,
                min(num_hard_negatives, len(current_texts))
            ) if current_texts else []

            remaining_slots = num_memories - len(relevant_memories) - len(hard_negs)
            random_negs = []
            if remaining_slots > 0 and all_utterances:
                avoid_set = set(relevant_memories + hard_negs + [query, target])
                candidates = [u for u in all_utterances if u not in avoid_set]
                if candidates:
                    random_negs = random.sample(candidates, min(remaining_slots, len(candidates)))

            all_memories = relevant_memories + hard_negs + random_negs
            relevant_indices = list(range(len(relevant_memories)))

            indices = list(range(len(all_memories)))
            random.shuffle(indices)
            shuffled_memories = [all_memories[idx] for idx in indices]
            shuffled_relevant = sorted([indices.index(r) for r in relevant_indices])

            samples.append({
                "input_text": query,
                "target_text": target,
                "memory_texts": shuffled_memories,
                "relevant_indices": shuffled_relevant,
            })

            i += 2

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="将 LoCoMo 数据集转换为 MAG 训练格式 JSONL"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="LoCoMo JSON 数据文件路径 (locomo10.json)")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 JSONL 文件路径")
    parser.add_argument("--num_memories", type=int, default=15,
                        help="每个样本的候选记忆数量 (default: 15)")
    parser.add_argument("--num_hard_negatives", type=int, default=3,
                        help="硬负例数量 (default: 3)")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["cross_session", "qa_evidence", "observation", "all"],
                        help="构建策略: cross_session=跨session记忆, "
                             "qa_evidence=QA标注, observation=事实观察, all=全部")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--min_history_sessions", type=int, default=1,
                        help="cross_session 策略中, 最少需要多少个历史 session (default: 1)")
    args = parser.parse_args()

    random.seed(args.seed)

    # 1. 解析 LoCoMo
    logger.info(f"正在解析 LoCoMo 数据: {args.input}")
    conversations = parse_locomo(args.input)

    if not conversations:
        logger.error("未解析到任何对话数据!")
        sys.exit(1)

    # 2. 收集全局 utterance 池 (用于跨对话随机负例)
    all_utterances: list[str] = []
    for conv in conversations:
        for s in conv["sessions"]:
            for t in s["turns"]:
                if len(t["text"]) >= 10:
                    all_utterances.append(t["text"])
    logger.info(f"全局 utterance 池: {len(all_utterances)} 条")

    # 3. 按策略生成样本
    all_samples = []
    strategy = args.strategy

    for conv_idx, conv in enumerate(conversations):
        conv_samples = []

        if strategy in ("cross_session", "all"):
            cs_samples = build_cross_session_samples(
                conv,
                num_memories=args.num_memories,
                num_hard_negatives=args.num_hard_negatives,
                min_history_sessions=args.min_history_sessions,
                all_utterances=all_utterances,
            )
            conv_samples.extend(cs_samples)
            if cs_samples:
                logger.info(f"  对话 {conv_idx} [{conv['sample_id']}]: "
                            f"cross_session 生成 {len(cs_samples)} 个样本")

        if strategy in ("qa_evidence", "all"):
            qa_samples = build_qa_evidence_samples(
                conv,
                num_memories=args.num_memories,
                num_hard_negatives=args.num_hard_negatives,
                all_utterances=all_utterances,
            )
            conv_samples.extend(qa_samples)
            if qa_samples:
                logger.info(f"  对话 {conv_idx} [{conv['sample_id']}]: "
                            f"qa_evidence 生成 {len(qa_samples)} 个样本")

        if strategy in ("observation", "all"):
            obs_samples = build_observation_samples(
                conv,
                num_memories=args.num_memories,
                num_hard_negatives=args.num_hard_negatives,
                all_utterances=all_utterances,
            )
            conv_samples.extend(obs_samples)
            if obs_samples:
                logger.info(f"  对话 {conv_idx} [{conv['sample_id']}]: "
                            f"observation 生成 {len(obs_samples)} 个样本")

        all_samples.extend(conv_samples)

    # 4. 全局打乱
    random.shuffle(all_samples)

    # 5. 统计
    n_total = len(all_samples)
    n_with_target = sum(1 for s in all_samples if "target_text" in s)
    avg_memories = sum(len(s["memory_texts"]) for s in all_samples) / max(n_total, 1)
    avg_relevant = sum(len(s["relevant_indices"]) for s in all_samples) / max(n_total, 1)

    logger.info("=" * 60)
    logger.info(f"转换完成!")
    logger.info(f"  总样本数: {n_total}")
    logger.info(f"  含 target_text: {n_with_target} ({100 * n_with_target / max(n_total, 1):.1f}%)")
    logger.info(f"  平均记忆数: {avg_memories:.1f}")
    logger.info(f"  平均相关记忆数: {avg_relevant:.1f}")
    logger.info("=" * 60)

    # 6. 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"已保存到: {output_path} ({file_size_mb:.2f} MB, {n_total} 行)")

    # 7. 打印几个样本示例
    logger.info("\n--- 样本示例 ---")
    for i, s in enumerate(all_samples[:3]):
        logger.info(f"\n样本 {i+1}:")
        logger.info(f"  query: {s['input_text'][:80]}...")
        logger.info(f"  target: {s.get('target_text', 'N/A')[:80]}...")
        logger.info(f"  记忆数: {len(s['memory_texts'])}, 相关: {s['relevant_indices']}")
        for ri in s["relevant_indices"][:2]:
            logger.info(f"    相关记忆[{ri}]: {s['memory_texts'][ri][:60]}...")


if __name__ == "__main__":
    main()
