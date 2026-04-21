#!/usr/bin/env python3
"""
MAG 评估脚本: 评估训练好的 ContextSelector + MAGGate 的效果。

评估维度:
1. Selector 准确率: Hit@K, Precision@K (选出的记忆是否真的相关)
2. Gate 激活分析: 各注入层的 gate sigmoid 均值/方差
3. PPL 对比: 有记忆注入 vs 无记忆注入的困惑度差异
4. 生成质量对比: 给定 query + 记忆, 对比有/无 MAG 的生成结果

用法:
    python scripts/eval_mag.py \
        --model_path ../models/Qwen--Qwen3-8b/ \
        --mag_weights_dir outputs/mag_trained \
        --data_path data/raw/dailydialog_train.jsonl \
        --num_eval_samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# 将项目根目录加入 sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.memory.mag.memory_encoder import MemoryEncoder, MemoryEncoderConfig
from src.memory.mag.context_selector import ContextSelector, ContextSelectorConfig
from src.memory.mag.mag_gate import MAGGate, MAGGateConfig
from src.memory.mag.kv_memory_injector import (
    KVMemoryInjector, KVMemoryInjectorConfig,
    KVAdapterInjector, RawKVInjector,
    create_kv_injector,
    compress_memory_for_kv_injection,
    extract_raw_kv_for_injection,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("eval_mag")


# ======================================================================
# 参数解析
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAG 评估脚本")

    # 模型与权重
    parser.add_argument("--model_path", type=str, required=True,
                        help="Backbone 模型路径")
    parser.add_argument("--mag_weights_dir", type=str, required=True,
                        help="MAG 训练权重目录 (含 context_selector.pt, mag_gate.pt, mag_config.json)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    parser.add_argument("--data_path", type=str, required=True,
                        help="评估数据路径 (JSONL, 与训练数据格式相同)")
    parser.add_argument("--num_eval_samples", type=int, default=100,
                        help="评估样本数量 (从数据文件中取前 N 条)")
    parser.add_argument("--max_seq_len", type=int, default=1024)

    # 评估选项
    parser.add_argument("--eval_selector", action="store_true", default=True,
                        help="评估 Selector 准确率")
    parser.add_argument("--eval_gate", action="store_true", default=True,
                        help="评估 Gate 激活分析")
    parser.add_argument("--eval_ppl", action="store_true", default=True,
                        help="评估 PPL 对比")
    parser.add_argument("--eval_generation", action="store_true", default=True,
                        help="评估生成质量对比")
    parser.add_argument("--num_generate_samples", type=int, default=5,
                        help="生成对比的样本数")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="生成时最大 token 数")
    parser.add_argument("--inference_gate_scale", type=float, default=1.0,
                        help="推理时记忆注入强度缩放 (0~1, 默认1.0 即不缩放. "
                             "直接缩放残差增量 gate*m_agg, 而非仅缩放 selection_weights. "
                             "推荐 0.3~0.5 以防自回归生成时 hidden state 偏移累积)")
    parser.add_argument("--layer_gate_scales", type=str, default="",
                        help="逐层差异化 gate_scale, 格式: 'layer_idx:scale,...' "
                             "例如 '9:0.8,18:0.5,27:0.3,35:0.2' "
                             "高层注入更强, 推理时应用更小的 scale. "
                             "未指定的层使用 --inference_gate_scale 的值")
    parser.add_argument("--mag_inject_steps", type=int, default=10,
                        help="自回归生成时最大 MAG 注入步数 (默认10). "
                             "超过此步数后完全停止注入, backbone 接管生成. "
                             "设为 0 表示不限制(仅靠衰减). 推荐 5~15")

    parser.add_argument("--deep_encode_layers", type=int, default=8,
                        help="记忆深层编码使用的 backbone 层数 (0=仅用 embedding, 推荐 8~12)")
    parser.add_argument("--sliding_window", type=int, default=0,
                        help="加载模型时启用 SWA 并设置窗口大小 (0=不启用, 需与训练时一致)")

    # 输出
    parser.add_argument("--output_file", type=str, default="",
                        help="评估结果 JSON 输出路径 (默认自动生成到 mag_weights_dir 下)")
    parser.add_argument("--output_report", type=str, default="",
                        help="评估详细报告 TXT 输出路径 (默认自动生成到 mag_weights_dir 下, 包含所有样本的实际输出)")
    parser.add_argument("--seed", type=int, default=42)

    # LLM-as-Judge 评估
    parser.add_argument("--eval_llm_judge", action="store_true", default=False,
                        help="使用 LLM-as-Judge 评估记忆利用质量 (用 backbone 自身打分)")
    parser.add_argument("--judge_model_path", type=str, default="",
                        help="Judge 模型路径 (默认空=使用 backbone 自身)")

    return parser.parse_args()


# ======================================================================
# 模型加载
# ======================================================================

def load_backbone(model_path: str, device: str, dtype_str: str, sliding_window: int = 0):
    """加载 backbone 模型和 tokenizer。

    Args:
        sliding_window: SWA 窗口大小 (0=不启用, 需与训练时一致).
    """
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    dtype = getattr(torch, dtype_str, torch.float32)

    # 解析路径中的 '..' 等相对引用，避免新版 transformers/huggingface_hub 报错
    model_path = os.path.realpath(model_path)

    logger.info(f"加载 backbone: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 设置 SWA (需与训练时一致)
    if sliding_window > 0:
        if hasattr(config, "sliding_window"):
            config.sliding_window = sliding_window
            logger.info(f"★ SWA 已启用: sliding_window={sliding_window}")
        if hasattr(config, "max_window_layers"):
            config.max_window_layers = config.num_hidden_layers
            logger.info(f"  max_window_layers={config.max_window_layers}")
    else:
        logger.info("SWA 未启用 (sliding_window=0)")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 冻结 backbone
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    hidden_dim = config.hidden_size
    num_layers = config.num_hidden_layers
    logger.info(f"backbone 加载完成: hidden_dim={hidden_dim}, num_layers={num_layers}")

    return model, tokenizer, hidden_dim, num_layers


def load_mag_weights(
    weights_dir: str,
    hidden_dim: int,
    device: str,
    dtype_str: str,
    backbone_config: Any = None,
):
    """加载训练好的 MAG 权重 (支持 svd_only / svd_adapter / raw_kv 三种模式)。"""
    weights_path = Path(weights_dir)

    # 加载配置
    config_path = weights_path / "mag_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到 MAG 配置文件: {config_path}")

    with open(config_path, "r") as f:
        mag_config = json.load(f)

    logger.info(f"MAG 配置: {json.dumps(mag_config, indent=2, ensure_ascii=False)}")

    # 初始化 ContextSelector
    sel_cfg = ContextSelectorConfig(
        input_dim=mag_config.get("hidden_dim", hidden_dim),
        hidden_dim=mag_config.get("selector_hidden_dim", 256),
        top_k=mag_config.get("selector_top_k", 5),
    )
    selector = ContextSelector(sel_cfg).to(device)

    # 加载 selector 权重
    sel_ckpt = weights_path / "context_selector.pt"
    if sel_ckpt.exists():
        selector.load_state_dict(torch.load(sel_ckpt, map_location=device))
        logger.info(f"Selector 权重已加载: {sel_ckpt} ({sel_ckpt.stat().st_size / 1024:.1f} KB)")
    else:
        logger.warning(f"找不到 Selector 权重: {sel_ckpt}")

    # 初始化 KV 注入器 (根据 injection_mode 自动选择类型)
    injection_mode = mag_config.get("injection_mode", "svd_only")
    num_attn_heads = mag_config.get("num_attention_heads", 32)
    num_kv_heads = mag_config.get("num_key_value_heads", 8)
    head_dim_val = mag_config.get("head_dim", hidden_dim // num_attn_heads)
    if backbone_config is not None:
        num_attn_heads = getattr(backbone_config, 'num_attention_heads', num_attn_heads)
        num_kv_heads = getattr(backbone_config, 'num_key_value_heads', num_kv_heads)
        head_dim_val = getattr(backbone_config, 'head_dim', head_dim_val)

    injector_cfg = KVMemoryInjectorConfig(
        hidden_dim=mag_config.get("hidden_dim", hidden_dim),
        num_layers=mag_config.get("num_layers", 36),
        injection_layers=mag_config.get("injection_layers", []),
        num_attention_heads=num_attn_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim_val,
        init_alpha=mag_config.get("kv_init_alpha", 0.1),
        max_alpha=mag_config.get("kv_max_alpha", 0.5),
        injection_mode=injection_mode,
        lora_rank=mag_config.get("lora_rank", 16),
        lora_share_params=mag_config.get("lora_share_params", True),
        max_raw_kv_tokens=mag_config.get("max_raw_kv_tokens", 128),
    )
    kv_injector = create_kv_injector(injector_cfg).to(device)
    logger.info(f"注入模式: {injection_mode}")

    # 加载 injector 权重 (strict=False: 允许 checkpoint 与模型结构部分不匹配)
    injector_ckpt = weights_path / "kv_injector.pt"
    if injector_ckpt.exists():
        ckpt_state = torch.load(injector_ckpt, map_location=device)
        load_result = kv_injector.load_state_dict(ckpt_state, strict=False)
        logger.info(f"KVInjector 权重已加载: {injector_ckpt} ({injector_ckpt.stat().st_size / 1024:.1f} KB)")
        if load_result.missing_keys:
            logger.warning(f"KVInjector 缺失的 key (将使用初始值): {load_result.missing_keys}")
        if load_result.unexpected_keys:
            logger.warning(f"KVInjector 多余的 key (已忽略): {load_result.unexpected_keys}")
    else:
        logger.warning(f"找不到 KVInjector 权重: {injector_ckpt}")

    selector.eval()
    kv_injector.eval()

    return selector, kv_injector, mag_config


# ======================================================================
# 数据加载
# ======================================================================

def load_eval_data(data_path: str, num_samples: int) -> list[dict]:
    """加载评估数据 (与训练数据格式一致)。"""
    import random

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    data = []

    # 如果是 JSONL 且含有 messages 字段, 先转换为 MAG 格式
    raw_data = []
    _decoder = json.JSONDecoder()
    n_skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    raw_data.append(obj)
                # 非 dict 类型直接跳过
            except json.JSONDecodeError as e:
                if "Extra data" in e.msg:
                    # 多个 JSON 对象拼在同一行, 用 raw_decode 逐个拆分
                    pos = 0
                    try:
                        while pos < len(line):
                            while pos < len(line) and line[pos] in " \t":
                                pos += 1
                            if pos >= len(line):
                                break
                            obj, end = _decoder.raw_decode(line, pos)
                            if isinstance(obj, dict):
                                raw_data.append(obj)
                            pos = end
                    except json.JSONDecodeError:
                        n_skipped += 1
                else:
                    n_skipped += 1
                continue
            if len(raw_data) >= num_samples * 5:  # 多读一些, 转换后筛选
                break
    if n_skipped > 0:
        logger.warning(f"JSONL 加载时跳过 {n_skipped} 行格式错误的数据")

    # 检查格式
    if raw_data and "input_text" in raw_data[0] and "memory_texts" in raw_data[0]:
        # 已经是 MAG 格式
        data = raw_data

        # ★ 检测数据是否已因果化 (记忆以 "用户: " 开头)
        is_causal = False
        for d in data[:10]:
            if d.get("memory_texts") and d["memory_texts"][0].startswith("用户: "):
                is_causal = True
                break
        if is_causal:
            logger.info(f"数据已因果化 (Causal Memory Training), {len(data)} 条样本")
        else:
            logger.info(f"数据为原始 MAG 格式, {len(data)} 条样本, 自动执行因果化...")
            data = _causalize_mag_data(data)
            logger.info(f"因果化完成: {len(data)} 条样本")

        data = data[:num_samples]
    elif raw_data and "messages" in raw_data[0]:
        # 对话格式, 需要转换
        # 从 train_mag.py 导入转换函数
        try:
            from scripts.train_mag import conversations_to_mag_samples
        except ImportError:
            # 如果导入失败, 用简化版本
            data = _simple_conversation_to_mag(raw_data, num_samples)
        else:
            data = conversations_to_mag_samples(
                raw_data,
                num_memories_per_sample=10,
                num_hard_negatives=3,
                seed=42,
            )[:num_samples]
    else:
        raise ValueError(f"数据格式不正确, 需要 'input_text'+'memory_texts' 或 'messages' 字段")

    logger.info(f"加载了 {len(data)} 条评估样本")
    return data


def _causalize_mag_data(
    data: list[dict],
    num_hard_negatives: int = 3,
    seed: int = 42,
) -> list[dict]:
    """将原始 MAG 格式数据因果化: 用对话前文原文替代摘要记忆。

    与 prepare_causal_data.py 逻辑一致:
    - 按 dialogue_id + session_idx 分组
    - 每个样本的记忆 = 同组前面轮次的 "用户: ... 助手: ..." 原文
    - 加入少量硬负例 (其他对话组的轮次)
    """
    import random as _rng
    from collections import defaultdict

    _rng.seed(seed)

    # 0. 过滤: 只保留有 input_text 的条目
    data = [d for d in data if d.get("input_text")]

    # 1. 按对话分组
    dialogue_groups: dict[str, list[dict]] = defaultdict(list)
    for d in data:
        dial_id = d.get("dialogue_id", "unknown")
        sess_idx = d.get("session_idx", 0)
        key = f"{dial_id}_sess{sess_idx}"
        dialogue_groups[key].append(d)

    group_keys_list = list(dialogue_groups.keys())
    logger.info(f"  因果化: {len(group_keys_list)} 个对话组")

    # 2. 构建轮次文本
    group_turn_texts: dict[str, list[str]] = {}
    for group_key, group in dialogue_groups.items():
        turns = []
        for d in group:
            turn_text = f"用户: {d['input_text']}"
            if d.get("target_text"):
                turn_text += f" 助手: {d['target_text']}"
            turns.append(turn_text)
        group_turn_texts[group_key] = turns

    # 3. 因果化构造
    group_key_to_idx = {k: i for i, k in enumerate(group_keys_list)}
    samples = []

    for group_key, group in dialogue_groups.items():
        if len(group) < 2:
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
            max_causal = 15 - num_hard_negatives
            if len(causal_memories) > max_causal:
                causal_memories = causal_memories[-max_causal:]

            relevant_count = len(causal_memories)

            # 硬负例
            hard_negs = []
            n_hard = min(num_hard_negatives, len(group_keys_list) - 1)
            if n_hard > 0:
                neg_indices = []
                while len(neg_indices) < n_hard:
                    idx = _rng.randint(0, len(group_keys_list) - 1)
                    if idx != my_idx:
                        neg_indices.append(idx)
                for gi in neg_indices:
                    neg_turns = group_turn_texts[group_keys_list[gi]]
                    if neg_turns:
                        hard_negs.append(neg_turns[_rng.randint(0, len(neg_turns) - 1)])

            # 组装 + 打乱
            all_memories = causal_memories + hard_negs
            relevant_indices = list(range(relevant_count))

            indices = list(range(len(all_memories)))
            _rng.shuffle(indices)
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

    _rng.shuffle(samples)
    avg_mem = sum(len(s["memory_texts"]) for s in samples) / max(len(samples), 1)
    avg_rel = sum(len(s["relevant_indices"]) for s in samples) / max(len(samples), 1)
    logger.info(f"  因果化: 平均 {avg_mem:.1f} 条记忆, {avg_rel:.1f} 条相关")
    return samples


def _simple_conversation_to_mag(
    conversations: list[dict],
    max_samples: int,
) -> list[dict]:
    """简化版对话→MAG 格式转换。"""
    import random
    random.seed(42)

    all_utterances = []
    for conv in conversations:
        for msg in conv.get("messages", []):
            if msg["role"] == "user" and len(msg["content"].strip()) >= 10:
                all_utterances.append(msg["content"].strip())

    samples = []
    for conv in conversations:
        messages = conv.get("messages", [])
        turns = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            content = msg["content"].strip()
            if msg["role"] == "user" and len(content) >= 5:
                user_text = content
                assistant_text = None
                if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                    a_content = messages[i + 1]["content"].strip()
                    if len(a_content) >= 5:
                        assistant_text = a_content
                        i += 1
                turns.append((user_text, assistant_text))
            i += 1

        if len(turns) < 3:
            continue

        for t in range(2, len(turns)):
            query = turns[t][0]
            target = turns[t][1]
            relevant = [tp[0] for tp in turns[:t]]
            # 加随机负例
            negs = random.sample(
                [u for u in all_utterances if u != query and u not in relevant],
                min(5, len(all_utterances)),
            )
            memories = relevant + negs
            relevant_indices = list(range(len(relevant)))
            # 打乱
            indices = list(range(len(memories)))
            random.shuffle(indices)
            shuffled = [memories[i] for i in indices]
            shuffled_rel = [indices.index(r) for r in relevant_indices]

            sample = {
                "input_text": query,
                "memory_texts": shuffled,
                "relevant_indices": shuffled_rel,
            }
            if target:
                sample["target_text"] = target
            samples.append(sample)

            if len(samples) >= max_samples:
                return samples

    return samples


# ======================================================================
# 评估 1: Selector 准确率
# ======================================================================

def eval_selector_accuracy(
    selector: ContextSelector,
    encoder: MemoryEncoder,
    data: list[dict],
    device: str,
) -> dict:
    """评估 Selector 的记忆选择准确率。

    指标:
    - Hit@K: top-K 中是否至少命中了 1 条相关记忆
    - Precision@K: top-K 中命中的相关记忆比例
    - NDCG@K: 归一化折损累积增益
    - 相关记忆平均得分 vs 不相关记忆平均得分 (区分度)
    """
    logger.info("=" * 60)
    logger.info("评估 1: ContextSelector 准确率")
    logger.info("=" * 60)

    top_k = selector.config.top_k
    hit_at_k = 0
    precision_at_k_sum = 0.0
    ndcg_at_k_sum = 0.0
    relevant_score_sum = 0.0
    irrelevant_score_sum = 0.0
    relevant_count = 0
    irrelevant_count = 0
    total = 0

    # ★ 精准打分分析: 收集所有分数用于 AUC-ROC、分布统计、统计检验
    all_relevant_scores: list[float] = []    # 所有相关记忆的 sigmoid 分数
    all_irrelevant_scores: list[float] = []  # 所有不相关记忆的 sigmoid 分数
    all_labels: list[int] = []               # 二分类标签 (1=相关, 0=不相关)
    all_probs: list[float] = []              # 对应的 sigmoid 概率
    per_sample_gaps: list[float] = []        # 每个样本内的 score_gap
    per_sample_auc: list[float] = []         # 每个样本内的 AUC

    selector.eval()
    with torch.no_grad():
        for sample in data:
            if "relevant_indices" not in sample or not sample["relevant_indices"]:
                continue

            query_emb = encoder.encode_texts([sample["input_text"]])  # (1, D)
            memory_embs = encoder.encode_texts_deep(sample["memory_texts"])  # (K, D) 深层编码
            memory_embs = memory_embs.unsqueeze(0)  # (1, K, D)

            scores = selector(query_emb, memory_embs)  # (1, K)
            probs = torch.sigmoid(scores).squeeze(0)  # (K,)

            relevant_set = set(sample["relevant_indices"])
            K = len(sample["memory_texts"])

            # 相关 vs 不相关得分
            sample_rel_scores = []
            sample_irrel_scores = []
            for idx in range(K):
                p = probs[idx].item()
                if idx in relevant_set:
                    relevant_score_sum += p
                    relevant_count += 1
                    all_relevant_scores.append(p)
                    sample_rel_scores.append(p)
                    all_labels.append(1)
                else:
                    irrelevant_score_sum += p
                    irrelevant_count += 1
                    all_irrelevant_scores.append(p)
                    sample_irrel_scores.append(p)
                    all_labels.append(0)
                all_probs.append(p)

            # 每个样本内的 gap
            if sample_rel_scores and sample_irrel_scores:
                s_gap = sum(sample_rel_scores) / len(sample_rel_scores) - sum(sample_irrel_scores) / len(sample_irrel_scores)
                per_sample_gaps.append(s_gap)

                # 每个样本内的 AUC (相关分数 > 不相关分数的比例)
                correct_pairs = 0
                total_pairs = 0
                for rs in sample_rel_scores:
                    for irs in sample_irrel_scores:
                        total_pairs += 1
                        if rs > irs:
                            correct_pairs += 1
                        elif rs == irs:
                            correct_pairs += 0.5
                per_sample_auc.append(correct_pairs / max(total_pairs, 1))

            # Top-K 指标
            actual_k = min(top_k, K)
            _, top_indices = torch.topk(probs, k=actual_k)
            top_indices_set = set(top_indices.cpu().tolist())

            # Hit@K
            if top_indices_set & relevant_set:
                hit_at_k += 1

            # Precision@K
            hits = len(top_indices_set & relevant_set)
            precision_at_k_sum += hits / actual_k

            # NDCG@K
            dcg = 0.0
            for rank, idx in enumerate(top_indices.cpu().tolist()):
                if idx in relevant_set:
                    dcg += 1.0 / math.log2(rank + 2)  # rank 从 0 开始
            # 理想 DCG
            ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_set), actual_k)))
            ndcg_at_k_sum += dcg / max(ideal_dcg, 1e-9)

            total += 1

    if total == 0:
        logger.warning("没有包含 relevant_indices 的评估样本!")
        return {}

    avg_rel = relevant_score_sum / max(relevant_count, 1)
    avg_irrel = irrelevant_score_sum / max(irrelevant_count, 1)

    results = {
        f"Hit@{top_k}": hit_at_k / total,
        f"Precision@{top_k}": precision_at_k_sum / total,
        f"NDCG@{top_k}": ndcg_at_k_sum / total,
        "avg_relevant_score": avg_rel,
        "avg_irrelevant_score": avg_irrel,
        "score_gap": avg_rel - avg_irrel,
        "num_samples": total,
    }

    logger.info(f"  样本数: {total}")
    logger.info(f"  Hit@{top_k}:       {results[f'Hit@{top_k}']:.4f}")
    logger.info(f"  Precision@{top_k}: {results[f'Precision@{top_k}']:.4f}")
    logger.info(f"  NDCG@{top_k}:      {results[f'NDCG@{top_k}']:.4f}")
    logger.info(f"  相关记忆平均分:    {results['avg_relevant_score']:.4f}")
    logger.info(f"  不相关记忆平均分:  {results['avg_irrelevant_score']:.4f}")
    logger.info(f"  区分度 (gap):      {results['score_gap']:.4f}")

    # ★ 精准打分分析
    logger.info(f"\n  ---- 精准打分分析 ----")

    # 1. 分数分布统计
    import numpy as np
    rel_arr = np.array(all_relevant_scores) if all_relevant_scores else np.array([0.0])
    irrel_arr = np.array(all_irrelevant_scores) if all_irrelevant_scores else np.array([0.0])

    logger.info(f"  [分数分布]")
    logger.info(f"    相关记忆:   mean={rel_arr.mean():.4f}  std={rel_arr.std():.4f}  "
                f"min={rel_arr.min():.4f}  max={rel_arr.max():.4f}  "
                f"median={np.median(rel_arr):.4f}  (N={len(all_relevant_scores)})")
    logger.info(f"    不相关记忆: mean={irrel_arr.mean():.4f}  std={irrel_arr.std():.4f}  "
                f"min={irrel_arr.min():.4f}  max={irrel_arr.max():.4f}  "
                f"median={np.median(irrel_arr):.4f}  (N={len(all_irrelevant_scores)})")

    results["relevant_score_std"] = float(rel_arr.std())
    results["irrelevant_score_std"] = float(irrel_arr.std())
    results["relevant_score_median"] = float(np.median(rel_arr))
    results["irrelevant_score_median"] = float(np.median(irrel_arr))

    # 2. 分数分布直方图 (文本形式)
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rel_hist, _ = np.histogram(rel_arr, bins=bins)
    irrel_hist, _ = np.histogram(irrel_arr, bins=bins)
    logger.info(f"  [分数分布直方图]")
    logger.info(f"    区间        相关    不相关")
    for i in range(len(bins) - 1):
        bar_rel = "█" * min(rel_hist[i], 30)
        bar_irrel = "█" * min(irrel_hist[i], 30)
        logger.info(f"    [{bins[i]:.1f}-{bins[i+1]:.1f})  {rel_hist[i]:>5d}   {irrel_hist[i]:>5d}  "
                     f"R|{bar_rel}  I|{bar_irrel}")

    # 3. 全局 AUC-ROC (Wilcoxon-Mann-Whitney 统计量)
    if all_relevant_scores and all_irrelevant_scores:
        correct_pairs = 0
        total_pairs = len(all_relevant_scores) * len(all_irrelevant_scores)
        # 高效计算: 排序法
        combined = [(s, 1) for s in all_relevant_scores] + [(s, 0) for s in all_irrelevant_scores]
        combined.sort(key=lambda x: x[0])
        # 计算 AUC: 相关分数排名之和
        rank_sum = 0.0
        for rank_idx, (score, label) in enumerate(combined):
            if label == 1:
                rank_sum += rank_idx + 1  # 1-based rank
        n_pos = len(all_relevant_scores)
        n_neg = len(all_irrelevant_scores)
        auc_roc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0.5

        results["auc_roc"] = auc_roc
        logger.info(f"  [AUC-ROC]: {auc_roc:.4f}  (0.5=随机, 1.0=完美)")
        if auc_roc < 0.55:
            logger.info(f"    🔴 AUC ≈ 0.5, Selector 打分接近随机, 未学到有效区分能力")
        elif auc_roc < 0.70:
            logger.info(f"    🟡 AUC 偏低, Selector 有微弱区分能力但不足")
        elif auc_roc < 0.85:
            logger.info(f"    🟢 AUC 尚可, Selector 有一定区分能力")
        else:
            logger.info(f"    ✅ AUC 优秀, Selector 区分能力强")

    # 4. 统计显著性检验 (Mann-Whitney U 检验)
    if len(all_relevant_scores) >= 5 and len(all_irrelevant_scores) >= 5:
        try:
            from scipy import stats as sp_stats
            u_stat, p_value = sp_stats.mannwhitneyu(
                all_relevant_scores, all_irrelevant_scores,
                alternative="greater",  # 检验: 相关分数 > 不相关分数
            )
            results["mannwhitney_u"] = float(u_stat)
            results["mannwhitney_p"] = float(p_value)
            logger.info(f"  [Mann-Whitney U 检验]: U={u_stat:.1f}, p={p_value:.6f}")
            if p_value < 0.001:
                logger.info(f"    ✅ p < 0.001, 相关/不相关分数差异极显著")
            elif p_value < 0.05:
                logger.info(f"    🟢 p < 0.05, 差异显著")
            else:
                logger.info(f"    🔴 p ≥ 0.05, 差异不显著 — Selector 未学到有效区分")

            # Cohen's d 效应量
            pooled_std = np.sqrt((rel_arr.std()**2 + irrel_arr.std()**2) / 2)
            if pooled_std > 1e-9:
                cohens_d = (rel_arr.mean() - irrel_arr.mean()) / pooled_std
                results["cohens_d"] = float(cohens_d)
                logger.info(f"  [Cohen's d 效应量]: {cohens_d:.4f}")
                if abs(cohens_d) < 0.2:
                    logger.info(f"    🔴 效应量极小 (|d| < 0.2), 实际区分能力可忽略")
                elif abs(cohens_d) < 0.5:
                    logger.info(f"    🟡 效应量小 (0.2 ≤ |d| < 0.5)")
                elif abs(cohens_d) < 0.8:
                    logger.info(f"    🟢 效应量中等 (0.5 ≤ |d| < 0.8)")
                else:
                    logger.info(f"    ✅ 效应量大 (|d| ≥ 0.8), 区分能力强")
        except ImportError:
            logger.info(f"  [统计检验]: scipy 未安装, 跳过 Mann-Whitney U 检验")

    # 5. 每样本 AUC 分布
    if per_sample_auc:
        ps_auc_arr = np.array(per_sample_auc)
        results["per_sample_auc_mean"] = float(ps_auc_arr.mean())
        results["per_sample_auc_std"] = float(ps_auc_arr.std())
        results["per_sample_auc_min"] = float(ps_auc_arr.min())
        n_good = int((ps_auc_arr >= 0.7).sum())
        n_bad = int((ps_auc_arr <= 0.5).sum())
        logger.info(f"  [每样本 AUC]: mean={ps_auc_arr.mean():.4f}  std={ps_auc_arr.std():.4f}  "
                     f"min={ps_auc_arr.min():.4f}  max={ps_auc_arr.max():.4f}")
        logger.info(f"    AUC≥0.7 的样本: {n_good}/{len(per_sample_auc)} ({n_good/len(per_sample_auc)*100:.1f}%)")
        logger.info(f"    AUC≤0.5 的样本: {n_bad}/{len(per_sample_auc)} ({n_bad/len(per_sample_auc)*100:.1f}%) ← 等于或差于随机")

    # 6. 每样本 gap 分布
    if per_sample_gaps:
        ps_gap_arr = np.array(per_sample_gaps)
        n_positive = int((ps_gap_arr > 0).sum())
        n_negative = int((ps_gap_arr <= 0).sum())
        results["per_sample_gap_mean"] = float(ps_gap_arr.mean())
        results["per_sample_gap_std"] = float(ps_gap_arr.std())
        results["pct_positive_gap"] = float(n_positive / len(per_sample_gaps) * 100)
        logger.info(f"  [每样本 Gap]: mean={ps_gap_arr.mean():.4f}  std={ps_gap_arr.std():.4f}")
        logger.info(f"    gap>0 (相关>不相关): {n_positive}/{len(per_sample_gaps)} ({n_positive/len(per_sample_gaps)*100:.1f}%)")
        logger.info(f"    gap≤0 (相关≤不相关): {n_negative}/{len(per_sample_gaps)} ({n_negative/len(per_sample_gaps)*100:.1f}%) ← 排序错误")

    # 判断效果
    gap = results["score_gap"]
    if gap > 0.15:
        logger.info("  ✅ Selector 区分度良好, 能有效区分相关/不相关记忆")
    elif gap > 0.05:
        logger.info("  🟡 Selector 有一定区分能力, 但效果有限")
    else:
        logger.info("  🔴 Selector 区分度差, 几乎无法区分相关和不相关记忆")

    return results


# ======================================================================
# 评估 2: Gate 激活分析
# ======================================================================

def eval_gate_activation(
    model: nn.Module,
    tokenizer: Any,
    encoder: MemoryEncoder,
    selector: ContextSelector,
    kv_injector: KVMemoryInjector,
    data: list[dict],
    device: str,
    max_seq_len: int = 1024,
) -> dict:
    """分析 KVMemoryInjector 在各注入层的 alpha 值。

    指标:
    - 各层 alpha 值 (期望 > 0.05, 说明在注入)
    """
    logger.info("=" * 60)
    logger.info("评估 2: Alpha 注入强度分析")
    logger.info("=" * 60)

    injector_module = kv_injector.module if hasattr(kv_injector, "module") else kv_injector

    sorted_injection = sorted(injector_module.injection_layers)
    results = {"per_layer": {}, "overall_alpha_mean": 0.0}
    all_alphas = []

    for layer_idx in sorted_injection:
        alpha = injector_module.get_alpha(layer_idx).item()
        all_alphas.append(alpha)
        results["per_layer"][str(layer_idx)] = {"alpha": alpha}
        logger.info(f"  Layer {layer_idx}: alpha={alpha:.4f}")

    if all_alphas:
        results["overall_alpha_mean"] = sum(all_alphas) / len(all_alphas)

    overall = results["overall_alpha_mean"]
    logger.info(f"  整体 alpha 均值: {overall:.4f}")
    if overall > 0.1:
        logger.info("  ✅ Alpha 已学到有意义的注入强度")
    elif overall > 0.03:
        logger.info("  🟡 Alpha 较小, 注入较弱")
    else:
        logger.info("  🔴 Alpha 几乎为 0, 注入未生效")

    logger.info(f"  评估样本数: {min(len(data), 50)}")

    return results


# ======================================================================
# 记忆压缩工具函数 (根据 injection_mode 选择压缩方式)
# ======================================================================

def _compress_memory_for_eval(
    per_layer_hs_flat: dict[int, torch.Tensor],
    backbone_layers: nn.ModuleList,
    flat_mask: torch.Tensor | None,
    injection_mode: str,
    svd_rank: int = 8,
    svd_normalize: bool = True,
    max_raw_kv_tokens: int = 128,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """根据 injection_mode 选择对应的记忆压缩方式。"""
    if injection_mode == "raw_kv":
        return extract_raw_kv_for_injection(
            memory_hidden_states=per_layer_hs_flat,
            backbone_layers=backbone_layers,
            attention_mask=flat_mask,
            max_tokens=max_raw_kv_tokens,
        )
    else:
        # svd_only / svd_adapter: 先做 SVD 压缩
        return compress_memory_for_kv_injection(
            memory_hidden_states=per_layer_hs_flat,
            backbone_layers=backbone_layers,
            attention_mask=flat_mask,
            svd_rank=svd_rank,
            normalize_keys=svd_normalize,
        )


# ======================================================================
# 评估 3: PPL 对比 (有记忆 vs 无记忆)
# ======================================================================

def eval_ppl_comparison(
    model: nn.Module,
    tokenizer: Any,
    encoder: MemoryEncoder,
    selector: ContextSelector,
    kv_injector: KVMemoryInjector,
    data: list[dict],
    device: str,
    max_seq_len: int = 1024,
    svd_rank: int = 8,
    svd_normalize: bool = True,
    injection_mode: str = "svd_only",
    max_raw_kv_tokens: int = 128,
    inference_scale: float = 0.3,
) -> dict:
    """对比有记忆注入和无记忆注入时的困惑度 (PPL)。

    核心: 如果 KV 注入学到了有用信息, 有记忆时 PPL 应该更低。
    """
    logger.info("=" * 60)
    logger.info("评估 3: PPL 对比 (有记忆 vs 无记忆)")
    logger.info("=" * 60)

    injector_module = kv_injector.module if hasattr(kv_injector, "module") else kv_injector
    selector_module = selector.module if hasattr(selector, "module") else selector

    backbone_layers, final_norm, lm_head, rotary_emb = _get_backbone_layers(model)
    sorted_injection = sorted(injector_module.injection_layers)
    first_injection_layer = sorted_injection[0] if sorted_injection else 0

    # 只评估有 target_text 的样本
    samples_with_target = [s for s in data if "target_text" in s and s["target_text"]]
    if not samples_with_target:
        logger.warning("没有包含 target_text 的评估样本, 跳过 PPL 对比!")
        return {}

    logger.info(f"  有 target_text 的样本: {len(samples_with_target)}")

    total_loss_with_mem = 0.0
    total_loss_no_mem = 0.0
    total_tokens = 0
    num_evaluated = 0

    selector.eval()
    kv_injector.eval()

    with torch.no_grad():
        for sample in samples_with_target[:min(len(samples_with_target), 100)]:
            # Tokenize: [query | target]
            query_enc = tokenizer(sample["input_text"], add_special_tokens=True,
                                  truncation=True, max_length=max_seq_len // 2)
            target_enc = tokenizer(sample["target_text"], add_special_tokens=False,
                                   truncation=True, max_length=max_seq_len // 2)

            query_ids = query_enc["input_ids"]
            target_ids = target_enc["input_ids"]
            combined_ids = query_ids + target_ids
            if len(combined_ids) > max_seq_len:
                combined_ids = combined_ids[:max_seq_len]

            input_ids = torch.tensor([combined_ids], device=device)
            query_len = len(query_ids)

            # Labels: 只在 target 部分计算 loss
            labels = torch.full_like(input_ids, -100)
            labels[0, query_len:] = input_ids[0, query_len:]

            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            # 编码记忆 (per-layer deep encoding + SVD 压缩)
            query_emb = encoder.encode_texts([sample["input_text"]])
            memory_embs = encoder.encode_texts_deep(sample["memory_texts"]).unsqueeze(0)

            # Per-layer SVD 压缩
            per_layer_hs, per_layer_mask = encoder.encode_texts_deep_all_layers(
                sample["memory_texts"],
                target_layers=sorted_injection,
            )
            virtual_kv_cache = {}
            if per_layer_hs:
                per_layer_hs_flat = {}
                flat_mask = None
                for l_idx, hs in per_layer_hs.items():
                    K_mem, T_mem, D_mem = hs.shape
                    per_layer_hs_flat[l_idx] = hs.view(1, K_mem * T_mem, D_mem)
                    if per_layer_mask is not None and flat_mask is None:
                        flat_mask = per_layer_mask.view(1, K_mem * T_mem)
                virtual_kv_cache = _compress_memory_for_eval(
                    per_layer_hs_flat=per_layer_hs_flat,
                    backbone_layers=backbone_layers,
                    flat_mask=flat_mask,
                    injection_mode=injection_mode,
                    svd_rank=svd_rank,
                    svd_normalize=svd_normalize,
                    max_raw_kv_tokens=max_raw_kv_tokens,
                )

            # --- Forward: 有记忆 ---
            loss_with = _forward_with_kv_injection(
                model, input_ids, labels,
                backbone_layers, final_norm, lm_head, rotary_emb,
                injector_module, virtual_kv_cache,
                first_injection_layer, device,
                inference_scale=inference_scale,
            )

            # --- Forward: 无记忆 (纯 backbone) ---
            loss_without = _forward_without_mag(
                model, input_ids, attention_mask, labels,
                backbone_layers, final_norm, lm_head, rotary_emb,
                device,
            )

            # 有效 token 数
            n_tokens = (labels != -100).sum().item()
            if n_tokens > 0:
                total_loss_with_mem += loss_with * n_tokens
                total_loss_no_mem += loss_without * n_tokens
                total_tokens += n_tokens
                num_evaluated += 1

    if total_tokens == 0:
        logger.warning("没有有效的评估 token!")
        return {}

    avg_loss_with = total_loss_with_mem / total_tokens
    avg_loss_without = total_loss_no_mem / total_tokens
    ppl_with = math.exp(min(avg_loss_with, 100))  # 防溢出
    ppl_without = math.exp(min(avg_loss_without, 100))

    results = {
        "ppl_with_memory": ppl_with,
        "ppl_without_memory": ppl_without,
        "ppl_reduction": ppl_without - ppl_with,
        "ppl_reduction_pct": (ppl_without - ppl_with) / max(ppl_without, 1e-9) * 100,
        "avg_loss_with_memory": avg_loss_with,
        "avg_loss_without_memory": avg_loss_without,
        "total_tokens": total_tokens,
        "num_samples": num_evaluated,
    }

    logger.info(f"  评估样本数: {num_evaluated}, 总 token: {total_tokens}")
    logger.info(f"  PPL (有记忆):   {ppl_with:.4f}  (avg_loss={avg_loss_with:.4f})")
    logger.info(f"  PPL (无记忆):   {ppl_without:.4f}  (avg_loss={avg_loss_without:.4f})")
    logger.info(f"  PPL 降低:       {results['ppl_reduction']:.4f} ({results['ppl_reduction_pct']:.2f}%)")

    if results["ppl_reduction"] > 0.5:
        logger.info("  ✅ 有记忆注入时 PPL 显著降低, MAG 有效!")
    elif results["ppl_reduction"] > 0:
        logger.info("  🟡 有记忆注入时 PPL 略有降低, MAG 有一定效果")
    else:
        logger.info("  🔴 有记忆注入时 PPL 未降低甚至升高, MAG 未生效或有害")

    return results


def _forward_with_kv_injection(
    model, input_ids, labels,
    backbone_layers, final_norm, lm_head, rotary_emb,
    injector_module, virtual_kv_cache,
    first_injection_layer, device,
    inference_scale: float = 0.3,
) -> float:
    """有 KV 注入的 forward, 返回 loss 标量。"""
    if hasattr(model.model, "embed_tokens"):
        h = model.model.embed_tokens(input_ids)
    else:
        h = model.get_input_embeddings()(input_ids)

    B_seq, T_seq, D_seq = h.shape
    position_ids = torch.arange(T_seq, device=h.device).unsqueeze(0)
    position_embeddings = _compute_position_embeddings(model, h, position_ids, rotary_emb)

    # 构造 4D causal mask
    causal_mask_4d = torch.zeros(B_seq, 1, T_seq, T_seq, device=h.device, dtype=h.dtype)
    causal_mask_4d.masked_fill_(
        torch.triu(torch.ones(T_seq, T_seq, device=h.device, dtype=torch.bool), diagonal=1)
        .unsqueeze(0).unsqueeze(0),
        float("-inf"),
    )

    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    layer_kwargs = {"attention_mask": causal_mask_4d}
    if position_embeddings is not None:
        layer_kwargs["position_embeddings"] = position_embeddings
    else:
        layer_kwargs["position_ids"] = position_ids

    for layer_idx in range(len(backbone_layers)):
        if layer_idx in injector_module.injection_layers and virtual_kv_cache:
            h, _ = injector_module.forward_decoder_layer(
                layer_idx=layer_idx,
                decoder_layer=backbone_layers[layer_idx],
                hidden_states=h,
                attention_mask=causal_mask_4d,
                position_embeddings=position_embeddings,
                virtual_kv_cache=virtual_kv_cache,
                inference_scale=inference_scale,
            )
        else:
            layer_output = backbone_layers[layer_idx](h, **layer_kwargs)
            h = layer_output[0] if isinstance(layer_output, tuple) else layer_output

    h = final_norm(h)
    logits = lm_head(h)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss.item()


def _forward_without_mag(
    model, input_ids, attention_mask, labels,
    backbone_layers, final_norm, lm_head, rotary_emb,
    device,
) -> float:
    """无 MAG 注入的 forward (纯 backbone), 返回 loss 标量。"""
    if hasattr(model.model, "embed_tokens"):
        h = model.model.embed_tokens(input_ids)
    else:
        h = model.get_input_embeddings()(input_ids)

    B_seq, T_seq, D_seq = h.shape
    position_ids = torch.arange(T_seq, device=h.device).unsqueeze(0)
    position_embeddings = _compute_position_embeddings(model, h, position_ids, rotary_emb)

    # 构造 4D causal mask (与 _forward_with_kv_injection 保持一致)
    causal_mask_4d = torch.zeros(B_seq, 1, T_seq, T_seq, device=h.device, dtype=h.dtype)
    causal_mask_4d.masked_fill_(
        torch.triu(torch.ones(T_seq, T_seq, device=h.device, dtype=torch.bool), diagonal=1)
        .unsqueeze(0).unsqueeze(0),
        float("-inf"),
    )

    layer_kwargs = {"attention_mask": causal_mask_4d}
    if position_embeddings is not None:
        layer_kwargs["position_embeddings"] = position_embeddings
    else:
        layer_kwargs["position_ids"] = position_ids

    for layer_idx in range(len(backbone_layers)):
        layer_output = backbone_layers[layer_idx](h, **layer_kwargs)
        h = layer_output[0] if isinstance(layer_output, tuple) else layer_output

    h = final_norm(h)
    logits = lm_head(h)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss.item()


# ======================================================================
# 评估 4: 生成质量对比
# ======================================================================

def eval_generation_comparison(
    model: nn.Module,
    tokenizer: Any,
    encoder: MemoryEncoder,
    selector: ContextSelector,
    kv_injector: KVMemoryInjector,
    data: list[dict],
    device: str,
    num_samples: int = 5,
    max_new_tokens: int = 128,
    max_seq_len: int = 1024,
    svd_rank: int = 8,
    svd_normalize: bool = True,
    max_inject_steps: int = 10,
    injection_mode: str = "svd_only",
    max_raw_kv_tokens: int = 128,
    inference_scale: float = 0.3,
    layer_gate_scales: dict[int, float] | None = None,
) -> dict:
    """对比有记忆 vs 无记忆的生成结果。

    选取几个有代表性的样本, 分别:
    - 无记忆: 纯 backbone 生成
    - 有记忆: KV 注入后生成
    展示两者的差异, 供人工评估。
    """
    logger.info("=" * 60)
    logger.info("评估 4: 生成质量对比")
    logger.info("=" * 60)
    logger.info(f"  最大注入步数 = {max_inject_steps} (超过后 backbone 完全接管)")
    logger.info(f"  inference_scale = {inference_scale} (注入强度缩放, 逐步衰减)")

    injector_module = kv_injector.module if hasattr(kv_injector, "module") else kv_injector
    selector_module = selector.module if hasattr(selector, "module") else selector

    backbone_layers, final_norm, lm_head, rotary_emb = _get_backbone_layers(model)
    sorted_injection = sorted(injector_module.injection_layers)

    samples_with_target = [s for s in data if "target_text" in s and s["target_text"]]
    if not samples_with_target:
        logger.warning("没有包含 target_text 的样本, 跳过生成对比!")
        return {}

    # 选取样本 (取有较多记忆的)
    selected = sorted(samples_with_target, key=lambda s: len(s.get("memory_texts", [])), reverse=True)
    selected = selected[:num_samples]

    generation_results = []

    selector.eval()
    kv_injector.eval()

    for i, sample in enumerate(selected):
        logger.info(f"\n--- 样本 {i + 1}/{num_samples} ---")
        logger.info(f"  Query: {sample['input_text'][:100]}...")
        logger.info(f"  Ground Truth: {sample['target_text'][:100]}...")
        logger.info(f"  记忆数: {len(sample['memory_texts'])}")
        if "relevant_indices" in sample:
            logger.info(f"  相关记忆索引: {sample['relevant_indices']}")

        # 展示部分记忆内容
        for j, mem in enumerate(sample["memory_texts"][:3]):
            rel_mark = "✓" if j in sample.get("relevant_indices", []) else "✗"
            logger.info(f"  记忆[{j}] ({rel_mark}): {mem[:80]}...")

        with torch.no_grad():
            # --- 无记忆生成 ---
            gen_no_mem = _generate_text(
                model, tokenizer, sample["input_text"],
                max_new_tokens=max_new_tokens,
                max_seq_len=max_seq_len,
                device=device,
            )

            # --- 有记忆生成 ---
            # Per-layer SVD 压缩
            per_layer_hs, per_layer_mask = encoder.encode_texts_deep_all_layers(
                sample["memory_texts"],
                target_layers=sorted_injection,
            )
            virtual_kv_cache = {}
            if per_layer_hs:
                per_layer_hs_flat = {}
                flat_mask = None
                for l_idx, hs in per_layer_hs.items():
                    K_mem, T_mem, D_mem = hs.shape
                    per_layer_hs_flat[l_idx] = hs.view(1, K_mem * T_mem, D_mem)
                    if per_layer_mask is not None and flat_mask is None:
                        flat_mask = per_layer_mask.view(1, K_mem * T_mem)
                virtual_kv_cache = _compress_memory_for_eval(
                    per_layer_hs_flat=per_layer_hs_flat,
                    backbone_layers=backbone_layers,
                    flat_mask=flat_mask,
                    injection_mode=injection_mode,
                    svd_rank=svd_rank,
                    svd_normalize=svd_normalize,
                    max_raw_kv_tokens=max_raw_kv_tokens,
                )

            gen_with_mem = _generate_text_with_kv_injection(
                model, tokenizer, sample["input_text"],
                injector_module=injector_module,
                virtual_kv_cache=virtual_kv_cache,
                max_new_tokens=max_new_tokens,
                max_seq_len=max_seq_len,
                device=device,
                max_inject_steps=max_inject_steps,
                inference_scale=inference_scale,
                layer_gate_scales=layer_gate_scales,
            )

        logger.info(f"  生成 (无记忆): {gen_no_mem[:150]}...")
        logger.info(f"  生成 (有记忆): {gen_with_mem[:150]}...")

        # 简单文本相似度 (字符 bigram 重叠率, 兼容中英文)
        import re as _re
        def _char_bigrams(text: str) -> set[str]:
            t = _re.sub(r'\s+', '', text.lower().strip())
            if len(t) < 2:
                return {t} if t else set()
            return {t[i:i+2] for i in range(len(t) - 1)}

        gt_ngrams = _char_bigrams(sample["target_text"])
        no_mem_ngrams = _char_bigrams(gen_no_mem)
        with_mem_ngrams = _char_bigrams(gen_with_mem)

        overlap_no = len(gt_ngrams & no_mem_ngrams) / max(len(gt_ngrams), 1)
        overlap_with = len(gt_ngrams & with_mem_ngrams) / max(len(gt_ngrams), 1)

        logger.info(f"  词重叠 (无记忆 vs GT): {overlap_no:.4f}")
        logger.info(f"  词重叠 (有记忆 vs GT): {overlap_with:.4f}")

        # 记忆关键词命中率 (Memory Keyword Recall)
        # 从记忆文本中提取关键 bigram, 检查生成文本是否包含
        selected_mems = sample.get("memory_texts", [])
        relevant_indices = sample.get("relevant_indices", list(range(len(selected_mems))))
        relevant_mems = [selected_mems[j] for j in relevant_indices if j < len(selected_mems)]
        mem_text_combined = " ".join(relevant_mems[:5])  # 取前5条相关记忆
        mem_ngrams = _char_bigrams(mem_text_combined)

        mem_recall_no = len(mem_ngrams & no_mem_ngrams) / max(len(mem_ngrams), 1) if mem_ngrams else 0
        mem_recall_with = len(mem_ngrams & with_mem_ngrams) / max(len(mem_ngrams), 1) if mem_ngrams else 0
        logger.info(f"  记忆关键词召回 (无记忆): {mem_recall_no:.4f}")
        logger.info(f"  记忆关键词召回 (有记忆): {mem_recall_with:.4f}")

        # 检查生成是否有差异
        is_different = gen_no_mem.strip() != gen_with_mem.strip()
        logger.info(f"  两种生成是否不同: {'是 ✅' if is_different else '否 🔴'}")

        # ★ 精准打分: ROUGE-L (最长公共子序列)
        def _rouge_l(hypothesis: str, reference: str) -> float:
            """计算 ROUGE-L F1 (基于字符级 LCS, 兼容中英文)。"""
            hyp = _re.sub(r'\s+', '', hypothesis.strip())
            ref = _re.sub(r'\s+', '', reference.strip())
            if not hyp or not ref:
                return 0.0
            m, n = len(hyp), len(ref)
            # 空间优化的 LCS
            prev = [0] * (n + 1)
            for i in range(1, m + 1):
                curr = [0] * (n + 1)
                for j in range(1, n + 1):
                    if hyp[i-1] == ref[j-1]:
                        curr[j] = prev[j-1] + 1
                    else:
                        curr[j] = max(curr[j-1], prev[j])
                prev = curr
            lcs_len = prev[n]
            precision = lcs_len / m if m > 0 else 0
            recall = lcs_len / n if n > 0 else 0
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)

        rouge_l_no = _rouge_l(gen_no_mem, sample["target_text"])
        rouge_l_with = _rouge_l(gen_with_mem, sample["target_text"])
        logger.info(f"  ROUGE-L (无记忆 vs GT): {rouge_l_no:.4f}")
        logger.info(f"  ROUGE-L (有记忆 vs GT): {rouge_l_with:.4f}")

        # ★ 精准打分: 实体/关键词精确匹配
        # 从 GT 和记忆中提取关键实体 (中文: 2-6字连续非标点; 英文: 完整单词)
        def _extract_entities(text: str) -> set[str]:
            """提取文本中的关键实体 (人名、地名、数字、专有名词等)。"""
            entities = set()
            # 中文: 提取 2-6 字的连续汉字片段 (作为候选实体)
            cn_matches = _re.findall(r'[\u4e00-\u9fff]{2,6}', text)
            # 过滤掉常见停用词
            cn_stopwords = {'我们', '你们', '他们', '这个', '那个', '什么', '怎么', '可以', '已经',
                           '但是', '因为', '所以', '如果', '虽然', '而且', '或者', '不过', '还是',
                           '一个', '一些', '这些', '那些', '自己', '现在', '时候', '问题', '方面',
                           '需要', '应该', '能够', '可能', '比较', '非常', '确实', '觉得', '认为',
                           '建议', '考虑', '关于', '方面', '情况', '进行', '提供', '使用', '通过'}
            for w in cn_matches:
                if w not in cn_stopwords:
                    entities.add(w)
            # 英文: 提取完整单词 (排除常见停用词)
            en_words = _re.findall(r'[a-zA-Z]{3,}', text)
            en_stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
                           'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
                           'will', 'with', 'this', 'that', 'from', 'they', 'were', 'some'}
            for w in en_words:
                if w.lower() not in en_stopwords:
                    entities.add(w.lower())
            # 数字 (含小数、百分比)
            numbers = _re.findall(r'\d+(?:\.\d+)?%?', text)
            entities.update(numbers)
            return entities

        gt_entities = _extract_entities(sample["target_text"])
        mem_entities = _extract_entities(mem_text_combined)
        no_mem_entities = _extract_entities(gen_no_mem)
        with_mem_entities = _extract_entities(gen_with_mem)

        # GT 实体命中率
        gt_entity_hit_no = len(gt_entities & no_mem_entities) / max(len(gt_entities), 1) if gt_entities else 0
        gt_entity_hit_with = len(gt_entities & with_mem_entities) / max(len(gt_entities), 1) if gt_entities else 0
        logger.info(f"  GT实体命中 (无记忆): {gt_entity_hit_no:.4f} ({len(gt_entities & no_mem_entities)}/{len(gt_entities)})")
        logger.info(f"  GT实体命中 (有记忆): {gt_entity_hit_with:.4f} ({len(gt_entities & with_mem_entities)}/{len(gt_entities)})")

        # ★ 精准打分: 记忆信息渗透率 (Memory Penetration Rate)
        # 记忆中的实体有多少出现在了生成文本中 (但不在 query 中)
        query_entities = _extract_entities(sample["input_text"])
        # 记忆独有实体 = 记忆实体 - query实体 (排除 query 本身就包含的)
        mem_unique_entities = mem_entities - query_entities
        if mem_unique_entities:
            penetration_no = len(mem_unique_entities & no_mem_entities) / len(mem_unique_entities)
            penetration_with = len(mem_unique_entities & with_mem_entities) / len(mem_unique_entities)
        else:
            penetration_no = 0.0
            penetration_with = 0.0
        logger.info(f"  记忆渗透率 (无记忆): {penetration_no:.4f} ({len(mem_unique_entities & no_mem_entities)}/{len(mem_unique_entities)} 独有实体)")
        logger.info(f"  记忆渗透率 (有记忆): {penetration_with:.4f} ({len(mem_unique_entities & with_mem_entities)}/{len(mem_unique_entities)} 独有实体)")

        # 构造记忆摘要文本 (供 LLM judge 使用)
        memories_summary = "\n".join(
            f"- {mem[:100]}" for mem in relevant_mems[:5]
        )

        generation_results.append({
            "query": sample["input_text"],
            "ground_truth": sample["target_text"],
            "gen_no_memory": gen_no_mem,
            "gen_with_memory": gen_with_mem,
            "overlap_no_mem": overlap_no,
            "overlap_with_mem": overlap_with,
            "mem_recall_no": mem_recall_no,
            "mem_recall_with": mem_recall_with,
            "rouge_l_no": rouge_l_no,
            "rouge_l_with": rouge_l_with,
            "gt_entity_hit_no": gt_entity_hit_no,
            "gt_entity_hit_with": gt_entity_hit_with,
            "mem_penetration_no": penetration_no,
            "mem_penetration_with": penetration_with,
            "is_different": is_different,
            "memories_text": memories_summary,
        })

    # 汇总
    n_different = sum(1 for r in generation_results if r["is_different"])
    n = max(len(generation_results), 1)
    avg_overlap_no = sum(r["overlap_no_mem"] for r in generation_results) / n
    avg_overlap_with = sum(r["overlap_with_mem"] for r in generation_results) / n
    avg_mem_recall_no = sum(r["mem_recall_no"] for r in generation_results) / n
    avg_mem_recall_with = sum(r["mem_recall_with"] for r in generation_results) / n
    avg_rouge_l_no = sum(r["rouge_l_no"] for r in generation_results) / n
    avg_rouge_l_with = sum(r["rouge_l_with"] for r in generation_results) / n
    avg_entity_hit_no = sum(r["gt_entity_hit_no"] for r in generation_results) / n
    avg_entity_hit_with = sum(r["gt_entity_hit_with"] for r in generation_results) / n
    avg_penetration_no = sum(r["mem_penetration_no"] for r in generation_results) / n
    avg_penetration_with = sum(r["mem_penetration_with"] for r in generation_results) / n

    results = {
        "num_samples": len(generation_results),
        "num_different_outputs": n_different,
        "pct_different": n_different / n * 100,
        "avg_word_overlap_no_mem": avg_overlap_no,
        "avg_word_overlap_with_mem": avg_overlap_with,
        "avg_mem_recall_no_mem": avg_mem_recall_no,
        "avg_mem_recall_with_mem": avg_mem_recall_with,
        "avg_rouge_l_no_mem": avg_rouge_l_no,
        "avg_rouge_l_with_mem": avg_rouge_l_with,
        "avg_gt_entity_hit_no_mem": avg_entity_hit_no,
        "avg_gt_entity_hit_with_mem": avg_entity_hit_with,
        "avg_mem_penetration_no_mem": avg_penetration_no,
        "avg_mem_penetration_with_mem": avg_penetration_with,
        "samples": generation_results,
    }

    logger.info(f"\n  ---- 生成结果汇总 ----")
    logger.info(f"  样本数:               {len(generation_results)}")
    logger.info(f"  两种生成有差异比例:   {results['pct_different']:.1f}%")

    logger.info(f"\n  {'指标':<22s} {'无记忆':>8s}  {'有记忆':>8s}  {'Δ':>8s}  {'判断'}")
    logger.info(f"  {'-'*70}")

    def _fmt_row(name: str, v_no: float, v_with: float) -> str:
        delta = v_with - v_no
        sign = "+" if delta > 0 else ""
        if delta > 0.01:
            verdict = "✅ 提升"
        elif delta < -0.01:
            verdict = "🔴 下降"
        else:
            verdict = "— 持平"
        return f"  {name:<22s} {v_no:>8.4f}  {v_with:>8.4f}  {sign}{delta:>7.4f}  {verdict}"

    logger.info(_fmt_row("字符Bigram重叠", avg_overlap_no, avg_overlap_with))
    logger.info(_fmt_row("ROUGE-L", avg_rouge_l_no, avg_rouge_l_with))
    logger.info(_fmt_row("GT实体命中率", avg_entity_hit_no, avg_entity_hit_with))
    logger.info(_fmt_row("记忆关键词召回", avg_mem_recall_no, avg_mem_recall_with))
    logger.info(_fmt_row("记忆信息渗透率", avg_penetration_no, avg_penetration_with))

    # 综合判断
    improvements = 0
    regressions = 0
    for v_no, v_with in [
        (avg_rouge_l_no, avg_rouge_l_with),
        (avg_entity_hit_no, avg_entity_hit_with),
        (avg_mem_recall_no, avg_mem_recall_with),
        (avg_penetration_no, avg_penetration_with),
    ]:
        if v_with > v_no + 0.005:
            improvements += 1
        elif v_with < v_no - 0.005:
            regressions += 1

    logger.info(f"\n  综合: {improvements}/4 项指标提升, {regressions}/4 项下降")
    if improvements >= 3:
        logger.info(f"  ✅ 记忆注入对生成质量有明显正面影响")
    elif improvements >= 2 and regressions == 0:
        logger.info(f"  🟢 记忆注入有一定正面影响")
    elif regressions >= 2:
        logger.info(f"  🔴 记忆注入对生成质量有负面影响, 需要调优")
    else:
        logger.info(f"  🟡 记忆注入效果不明显, 需要更多样本或调优参数")

    return results


# ======================================================================
# 评估 5: LLM-as-Judge 记忆利用质量评估
# ======================================================================

# --- Prompt 模板 (RAGAS 风格) ---

_JUDGE_PROMPT_MEMORY_FAITHFULNESS = """你是一个严格的评估专家。请评估以下回答是否利用了提供的记忆信息。

【记忆内容】
{memories}

【用户问题】
{query}

【模型回答】
{answer}

请从以下维度打分 (1-5分):
1. 记忆利用度: 回答是否引用或利用了记忆中的具体信息(人名、事件、数据、建议等)?
   - 1分: 完全没有利用记忆, 回答是通用的
   - 2分: 可能间接涉及记忆内容, 但不明确
   - 3分: 利用了部分记忆信息
   - 4分: 较好地利用了记忆中的关键信息
   - 5分: 充分利用了记忆, 回答明显基于记忆内容

2. 回答质量: 回答是否流畅、有条理、有实质内容?
   - 1分: 乱码/重复/无意义
   - 2分: 能读懂但质量差
   - 3分: 基本合格
   - 4分: 质量较好
   - 5分: 质量优秀

请严格按以下JSON格式输出, 不要输出其他内容:
{{"memory_utilization": <1-5>, "answer_quality": <1-5>}}"""

_JUDGE_PROMPT_COMPARATIVE = """你是一个严格的评估专家。请对比以下两个回答, 判断哪个更好地利用了记忆信息来回答问题。

【记忆内容】
{memories}

【用户问题】
{query}

【回答A (无记忆辅助)】
{answer_no_mem}

【回答B (有记忆辅助)】
{answer_with_mem}

请评估:
1. 哪个回答更好地利用了记忆中的信息? (A/B/平局)
2. 哪个回答整体质量更高? (A/B/平局)

请严格按以下JSON格式输出, 不要输出其他内容:
{{"memory_winner": "<A|B|tie>", "quality_winner": "<A|B|tie>", "reason": "<简短理由>"}}"""


def _llm_judge_score(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 200,
) -> str:
    """用 LLM 生成评估结果 (JSON 格式)。"""
    # 构造 chat 格式 (如果 tokenizer 支持)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,  # Qwen3 关闭思考模式, 直接输出
        )
    else:
        input_text = prompt

    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][encoded["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _parse_judge_scores(response: str) -> dict:
    """从 LLM judge 的回复中解析 JSON 分数。"""
    import re as _re
    # 尝试提取 JSON
    json_match = _re.search(r'\{[^{}]+\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {}


def eval_llm_judge(
    model: nn.Module,
    tokenizer: Any,
    generation_results: list[dict],
    device: str = "cuda",
) -> dict:
    """使用 LLM-as-Judge 评估记忆利用质量。

    评估维度 (参考 RAGAS 框架):
    1. Memory Faithfulness (记忆忠实度): 回答是否利用了记忆信息
    2. Answer Quality (回答质量): 回答是否流畅有条理
    3. Comparative (对比评估): 有记忆 vs 无记忆哪个更好

    使用 backbone 模型自身作为 judge, 无需外部 API。
    """
    logger.info("=" * 60)
    logger.info("评估 5: LLM-as-Judge 记忆利用质量")
    logger.info("=" * 60)

    if not generation_results:
        logger.warning("没有生成结果可供评估!")
        return {}

    scores_with_mem = []   # 有记忆的 faithfulness 分数
    scores_no_mem = []     # 无记忆的 faithfulness 分数
    quality_with = []      # 有记忆的质量分数
    quality_no = []        # 无记忆的质量分数
    comparative_results = []  # 对比结果

    for i, result in enumerate(generation_results):
        query = result["query"][:300]
        gt = result.get("ground_truth", "")[:200]
        gen_no = result["gen_no_memory"][:500]
        gen_with = result["gen_with_memory"][:500]

        # 从 result 中获取记忆文本 (如果有的话)
        # 注意: generation_results 中可能没有 memory_texts, 需要从原始数据获取
        memories_text = result.get("memories_text", "")
        if not memories_text and gt:
            # 用 GT 作为记忆参考 (因为 GT 就是基于记忆生成的)
            memories_text = f"参考答案: {gt}"

        logger.info(f"\n  --- Judge 样本 {i+1}/{len(generation_results)} ---")

        # 1. 评估有记忆的回答
        prompt_with = _JUDGE_PROMPT_MEMORY_FAITHFULNESS.format(
            memories=memories_text,
            query=query,
            answer=gen_with,
        )
        resp_with = _llm_judge_score(model, tokenizer, prompt_with, device)
        scores_w = _parse_judge_scores(resp_with)

        mem_util_with = scores_w.get("memory_utilization", 0)
        qual_with = scores_w.get("answer_quality", 0)
        scores_with_mem.append(mem_util_with)
        quality_with.append(qual_with)
        logger.info(f"    有记忆: 记忆利用={mem_util_with}/5, 质量={qual_with}/5")

        # 2. 评估无记忆的回答
        prompt_no = _JUDGE_PROMPT_MEMORY_FAITHFULNESS.format(
            memories=memories_text,
            query=query,
            answer=gen_no,
        )
        resp_no = _llm_judge_score(model, tokenizer, prompt_no, device)
        scores_n = _parse_judge_scores(resp_no)

        mem_util_no = scores_n.get("memory_utilization", 0)
        qual_no = scores_n.get("answer_quality", 0)
        scores_no_mem.append(mem_util_no)
        quality_no.append(qual_no)
        logger.info(f"    无记忆: 记忆利用={mem_util_no}/5, 质量={qual_no}/5")

        # 3. 对比评估
        prompt_cmp = _JUDGE_PROMPT_COMPARATIVE.format(
            memories=memories_text,
            query=query,
            answer_no_mem=gen_no,
            answer_with_mem=gen_with,
        )
        resp_cmp = _llm_judge_score(model, tokenizer, prompt_cmp, device)
        cmp_result = _parse_judge_scores(resp_cmp)
        comparative_results.append(cmp_result)

        mem_winner = cmp_result.get("memory_winner", "?")
        qual_winner = cmp_result.get("quality_winner", "?")
        reason = cmp_result.get("reason", "")
        logger.info(f"    对比: 记忆利用胜={mem_winner}, 质量胜={qual_winner}")
        if reason:
            logger.info(f"    理由: {reason[:100]}")

    # 汇总
    valid_with = [s for s in scores_with_mem if s > 0]
    valid_no = [s for s in scores_no_mem if s > 0]
    valid_q_with = [s for s in quality_with if s > 0]
    valid_q_no = [s for s in quality_no if s > 0]

    avg_mem_util_with = sum(valid_with) / max(len(valid_with), 1)
    avg_mem_util_no = sum(valid_no) / max(len(valid_no), 1)
    avg_qual_with = sum(valid_q_with) / max(len(valid_q_with), 1)
    avg_qual_no = sum(valid_q_no) / max(len(valid_q_no), 1)

    # 对比统计
    n_mem_win_b = sum(1 for r in comparative_results if r.get("memory_winner") == "B")
    n_mem_win_a = sum(1 for r in comparative_results if r.get("memory_winner") == "A")
    n_mem_tie = sum(1 for r in comparative_results if r.get("memory_winner") == "tie")
    n_qual_win_b = sum(1 for r in comparative_results if r.get("quality_winner") == "B")
    n_qual_win_a = sum(1 for r in comparative_results if r.get("quality_winner") == "A")
    n_qual_tie = sum(1 for r in comparative_results if r.get("quality_winner") == "tie")

    total = len(generation_results)

    results = {
        "num_samples": total,
        "avg_memory_utilization_with_mem": avg_mem_util_with,
        "avg_memory_utilization_no_mem": avg_mem_util_no,
        "memory_utilization_lift": avg_mem_util_with - avg_mem_util_no,
        "avg_quality_with_mem": avg_qual_with,
        "avg_quality_no_mem": avg_qual_no,
        "quality_lift": avg_qual_with - avg_qual_no,
        "comparative_memory_win_B": n_mem_win_b,
        "comparative_memory_win_A": n_mem_win_a,
        "comparative_memory_tie": n_mem_tie,
        "comparative_quality_win_B": n_qual_win_b,
        "comparative_quality_win_A": n_qual_win_a,
        "comparative_quality_tie": n_qual_tie,
        "comparative_details": comparative_results,
    }

    logger.info(f"\n  LLM Judge 汇总:")
    logger.info(f"  记忆利用度 (有记忆): {avg_mem_util_with:.2f}/5")
    logger.info(f"  记忆利用度 (无记忆): {avg_mem_util_no:.2f}/5")
    logger.info(f"  记忆利用提升:        {avg_mem_util_with - avg_mem_util_no:+.2f}")
    logger.info(f"  回答质量 (有记忆):   {avg_qual_with:.2f}/5")
    logger.info(f"  回答质量 (无记忆):   {avg_qual_no:.2f}/5")
    logger.info(f"  质量提升:            {avg_qual_with - avg_qual_no:+.2f}")
    logger.info(f"  对比-记忆利用: 有记忆胜={n_mem_win_b}, 无记忆胜={n_mem_win_a}, 平局={n_mem_tie}")
    logger.info(f"  对比-整体质量: 有记忆胜={n_qual_win_b}, 无记忆胜={n_qual_win_a}, 平局={n_qual_tie}")

    if avg_mem_util_with > avg_mem_util_no + 0.5:
        logger.info("  ✅ 有记忆注入时模型更好地利用了记忆信息!")
    elif avg_mem_util_with < avg_mem_util_no:
        logger.info("  🔴 有记忆注入反而降低了记忆利用度, MAG 注入可能有害")
    else:
        logger.info("  ⚠️ 记忆利用度提升不明显, 需要进一步调优")

    return results


def _detect_ngram_repeat(token_ids: list[int], n: int = 4) -> bool:
    """检测是否出现 n-gram 重复循环。"""
    if len(token_ids) < n * 3:
        return False
    # 检查最后 n 个 token 构成的 n-gram 是否在之前出现过 >= 2 次
    last_ngram = tuple(token_ids[-n:])
    count = 0
    for i in range(len(token_ids) - n):
        if tuple(token_ids[i:i+n]) == last_ngram:
            count += 1
    return count >= 3  # 同一个 n-gram 出现 3+ 次 = 循环


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float = 1.3,
) -> torch.Tensor:
    """对已生成的 token 施加重复惩罚。"""
    if not generated_ids or penalty == 1.0:
        return logits
    unique_ids = set(generated_ids)
    for token_id in unique_ids:
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def _generate_text(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    max_seq_len: int = 1024,
    device: str = "cuda",
    repetition_penalty: float = 1.3,
) -> str:
    """纯 backbone 生成文本 (无 MAG)。"""
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding, 便于对比
            temperature=1.0,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 只返回新生成的部分
    new_tokens = outputs[0][encoded["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _generate_text_with_kv_injection(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    injector_module: KVMemoryInjector,
    virtual_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
    max_new_tokens: int = 128,
    max_seq_len: int = 1024,
    device: str = "cuda",
    repetition_penalty: float = 1.5,
    max_inject_steps: int = 10,
    inference_scale: float = 0.3,
    layer_gate_scales: dict[int, float] | None = None,
) -> str:
    """有 KV 注入的文本生成 — 直接拼接虚拟 KV 到 self-attention。

    ★ 核心策略:
      1. 每步做 full sequence forward (无 KV cache, 简单可靠)
      2. 在注入层, 虚拟 KV 直接拼到 backbone self-attention 的 KV 上
      3. 只在前 max_inject_steps 步注入, 之后完全停止 (backbone 接管)
      4. 注入强度随步数线性衰减: scale * (1 - step/max_inject_steps)
      5. alpha 由 KVMemoryInjector 的可学习参数控制

    额外保护:
    - inference_scale: 全局注入强度缩放 (推荐 0.2~0.5)
    - 逐步衰减: 注入强度随生成步数线性递减, 防止累积偏移
    - Repetition penalty (1.5): 对已生成 token 降权, 防止循环
    - N-gram 重复检测: 4-gram 重复 3 次则强制终止
    - 硬截断: 超过 max_inject_steps 步后, 不注入
    """
    backbone_layers, final_norm, lm_head, rotary_emb = _get_backbone_layers(model)
    num_layers = len(backbone_layers)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    )
    input_ids = encoded["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    generated_ids = input_ids.clone()
    generated_token_list: list[int] = []
    eos_token_id = tokenizer.eos_token_id

    effective_inject_steps = max_inject_steps if max_inject_steps > 0 else max_new_tokens

    with torch.no_grad():
        for gen_step in range(max_new_tokens):
            cur_ids = generated_ids
            if cur_ids.shape[1] > max_seq_len:
                cur_ids = cur_ids[:, -max_seq_len:]

            # Embedding
            if hasattr(model.model, "embed_tokens"):
                h = model.model.embed_tokens(cur_ids)
            else:
                h = model.get_input_embeddings()(cur_ids)

            B_seq, T_seq, D_seq = h.shape
            attention_mask = torch.ones(1, T_seq, device=device, dtype=torch.bool)
            position_ids = torch.arange(T_seq, device=device).unsqueeze(0)
            position_embeddings = _compute_position_embeddings(model, h, position_ids, rotary_emb)

            layer_kwargs: dict[str, Any] = {"attention_mask": attention_mask}
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings
            else:
                layer_kwargs["position_ids"] = position_ids

            # 构造 4D causal mask
            causal_mask_4d = torch.zeros(B_seq, 1, T_seq, T_seq, device=h.device, dtype=h.dtype)
            causal_mask_4d.masked_fill_(
                torch.triu(torch.ones(T_seq, T_seq, device=h.device, dtype=torch.bool), diagonal=1)
                .unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )

            # 是否注入 + 逐步衰减
            do_inject = gen_step < effective_inject_steps and virtual_kv_cache

            # 计算当前步的注入强度: 基础 scale × 线性衰减
            if do_inject:
                decay = 1.0 - gen_step / effective_inject_steps  # 从 1.0 线性衰减到 0
                step_scale = inference_scale * decay
            else:
                step_scale = 0.0

            for layer_idx in range(num_layers):
                if do_inject and layer_idx in injector_module.injection_layers:
                    # 逐层差异化缩放
                    layer_scale = step_scale
                    if layer_gate_scales and layer_idx in layer_gate_scales:
                        layer_scale = layer_gate_scales[layer_idx] * decay

                    h, _ = injector_module.forward_decoder_layer(
                        layer_idx=layer_idx,
                        decoder_layer=backbone_layers[layer_idx],
                        hidden_states=h,
                        attention_mask=causal_mask_4d,
                        position_embeddings=position_embeddings,
                        virtual_kv_cache=virtual_kv_cache,
                        inference_scale=layer_scale,
                    )
                else:
                    # ★ 非注入层也使用 4D causal mask, 保持一致性
                    layer_output = backbone_layers[layer_idx](
                        h,
                        attention_mask=causal_mask_4d,
                        position_embeddings=position_embeddings,
                    )
                    h = layer_output[0] if isinstance(layer_output, tuple) else layer_output

            h = final_norm(h)
            logits = lm_head(h)

            # 取最后一个 token 的 logits
            next_logits = logits[0, -1, :].clone()

            # 施加 repetition penalty
            next_logits = _apply_repetition_penalty(
                next_logits, generated_token_list, penalty=repetition_penalty
            )

            # Greedy decoding
            next_token_id = next_logits.argmax(dim=-1).item()
            generated_token_list.append(next_token_id)

            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            generated_ids = torch.cat([generated_ids, next_token_tensor], dim=-1)

            # 终止条件 1: EOS
            if next_token_id == eos_token_id:
                break

            # 终止条件 2: N-gram 重复检测
            if _detect_ngram_repeat(generated_token_list, n=4):
                break

    # 只返回新生成的部分
    new_tokens = generated_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ======================================================================
# 工具函数 (从 train_mag.py 复用)
# ======================================================================

def _get_backbone_layers(model: nn.Module):
    """获取 backbone 的 transformer 层列表、final norm、lm_head 和 rotary_emb。"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        rotary_emb = getattr(model.model, "rotary_emb", None)
        return model.model.layers, model.model.norm, model.lm_head, rotary_emb
    raise ValueError("不支持的模型结构，请检查 backbone 的层结构")


def _compute_position_embeddings(model, h, position_ids, rotary_emb=None):
    """计算 rotary position embeddings。"""
    if rotary_emb is not None:
        return rotary_emb(h, position_ids)
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        return model.model.rotary_emb(h, position_ids)
    return None


# ======================================================================
# 详细报告保存
# ======================================================================

def _save_detailed_report(
    report_path: str,
    all_results: dict,
    total_time: float,
    args: argparse.Namespace,
) -> None:
    """保存人类可读的详细评估报告, 包含所有样本的实际输出。"""
    lines: list[str] = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("MAG 评估详细报告")
    lines.append(sep)
    lines.append(f"模型路径:       {args.model_path}")
    lines.append(f"MAG 权重目录:   {args.mag_weights_dir}")
    lines.append(f"数据路径:       {args.data_path}")
    lines.append(f"评估样本数:     {args.num_eval_samples}")
    lines.append(f"生成样本数:     {args.num_generate_samples}")
    lines.append(f"inference_scale: {args.inference_gate_scale}")
    lines.append(f"mag_inject_steps: {args.mag_inject_steps}")
    lines.append(f"总耗时:         {total_time:.1f}s")
    lines.append("")

    # ---- Selector 评估 ----
    if "selector" in all_results and all_results["selector"]:
        sel = all_results["selector"]
        lines.append(sep)
        lines.append("评估 1: ContextSelector 准确率")
        lines.append(sep)
        for k, v in sel.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")

    # ---- Gate/Alpha 评估 ----
    if "gate" in all_results and all_results["gate"]:
        gate = all_results["gate"]
        lines.append(sep)
        lines.append("评估 2: Alpha 注入强度分析")
        lines.append(sep)
        for k, v in gate.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")

    # ---- PPL 评估 ----
    if "ppl" in all_results and all_results["ppl"]:
        ppl = all_results["ppl"]
        lines.append(sep)
        lines.append("评估 3: PPL 对比 (有记忆 vs 无记忆)")
        lines.append(sep)
        for k, v in ppl.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")

    # ---- 生成质量对比 (逐样本详细输出) ----
    if "generation" in all_results and all_results["generation"]:
        gen = all_results["generation"]
        lines.append(sep)
        lines.append("评估 4: 生成质量对比 (逐样本详细输出)")
        lines.append(sep)
        lines.append(f"  样本数:           {gen.get('num_samples', 0)}")
        lines.append(f"  生成差异比例:     {gen.get('pct_different', 0):.1f}%")
        lines.append(f"  平均词重叠 (无记忆): {gen.get('avg_word_overlap_no_mem', 0):.4f}")
        lines.append(f"  平均词重叠 (有记忆): {gen.get('avg_word_overlap_with_mem', 0):.4f}")
        lines.append(f"  平均记忆召回 (无记忆): {gen.get('avg_mem_recall_no_mem', 0):.4f}")
        lines.append(f"  平均记忆召回 (有记忆): {gen.get('avg_mem_recall_with_mem', 0):.4f}")
        lines.append("")

        samples = gen.get("samples", [])
        for i, s in enumerate(samples):
            lines.append("-" * 80)
            lines.append(f"样本 {i + 1}/{len(samples)}")
            lines.append("-" * 80)
            lines.append(f"[Query]")
            lines.append(f"  {s.get('query', '')}")
            lines.append("")
            lines.append(f"[Ground Truth]")
            lines.append(f"  {s.get('ground_truth', '')}")
            lines.append("")
            lines.append(f"[记忆内容]")
            lines.append(f"  {s.get('memories_text', '(无)')}")
            lines.append("")
            lines.append(f"[生成 - 无记忆]")
            lines.append(f"  {s.get('gen_no_memory', '')}")
            lines.append("")
            lines.append(f"[生成 - 有记忆]")
            lines.append(f"  {s.get('gen_with_memory', '')}")
            lines.append("")
            lines.append(f"  词重叠 (无记忆 vs GT): {s.get('overlap_no_mem', 0):.4f}")
            lines.append(f"  词重叠 (有记忆 vs GT): {s.get('overlap_with_mem', 0):.4f}")
            lines.append(f"  记忆关键词召回 (无记忆): {s.get('mem_recall_no', 0):.4f}")
            lines.append(f"  记忆关键词召回 (有记忆): {s.get('mem_recall_with', 0):.4f}")
            lines.append(f"  两种生成是否不同: {'是 ✅' if s.get('is_different') else '否 🔴'}")
            lines.append("")

    # ---- LLM Judge ----
    if "llm_judge" in all_results and all_results["llm_judge"]:
        judge = all_results["llm_judge"]
        lines.append(sep)
        lines.append("评估 5: LLM-as-Judge")
        lines.append(sep)
        for k, v in judge.items():
            if k == "samples":
                continue
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")

        judge_samples = judge.get("samples", [])
        for i, js in enumerate(judge_samples):
            lines.append(f"  --- Judge 样本 {i + 1} ---")
            for jk, jv in js.items():
                lines.append(f"    {jk}: {jv}")
            lines.append("")

    # ---- 汇总 ----
    lines.append(sep)
    lines.append("汇总")
    lines.append(sep)

    if "selector" in all_results and all_results["selector"]:
        sel = all_results["selector"]
        lines.append(f"  Selector Hit@5: {sel.get('Hit@5', sel.get('Hit@3', 'N/A'))}")
        lines.append(f"  Selector 区分度: {sel.get('score_gap', 'N/A')}")
    if "gate" in all_results and all_results["gate"]:
        lines.append(f"  Alpha 均值: {all_results['gate'].get('overall_alpha_mean', 'N/A')}")
    if "ppl" in all_results and all_results["ppl"]:
        ppl = all_results["ppl"]
        lines.append(f"  PPL (有记忆): {ppl.get('ppl_with_memory', 'N/A')}")
        lines.append(f"  PPL (无记忆): {ppl.get('ppl_without_memory', 'N/A')}")
        lines.append(f"  PPL 降低: {ppl.get('ppl_reduction_pct', 'N/A')}%")
    if "generation" in all_results and all_results["generation"]:
        gen = all_results["generation"]
        lines.append(f"  记忆召回 (有记忆): {gen.get('avg_mem_recall_with_mem', 'N/A')}")
        lines.append(f"  记忆召回 (无记忆): {gen.get('avg_mem_recall_no_mem', 'N/A')}")

    lines.append("")

    report_p = Path(report_path)
    report_p.parent.mkdir(parents=True, exist_ok=True)
    with open(report_p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ======================================================================
# 主函数
# ======================================================================

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        args.device = "cuda:0"

    # 1. 加载 backbone
    model, tokenizer, hidden_dim, num_layers = load_backbone(
        args.model_path, args.device, args.dtype,
        sliding_window=args.sliding_window,
    )

    # 2. 加载 MAG 训练权重
    backbone_config = model.config
    selector, kv_injector, mag_config = load_mag_weights(
        args.mag_weights_dir, hidden_dim, args.device, args.dtype,
        backbone_config=backbone_config,
    )

    # 3. 初始化 MemoryEncoder
    enc_cfg = MemoryEncoderConfig(
        max_memory_tokens=mag_config.get("max_memory_tokens", 64),
        pooling="mean",
        deep_encode_layers=args.deep_encode_layers,
    )
    encoder = MemoryEncoder(enc_cfg)
    encoder.set_backbone(
        backbone_model=model,
        tokenizer=tokenizer,
        hidden_dim=hidden_dim,
        device=args.device,
        dtype=getattr(torch, args.dtype, torch.float32),
    )

    # SVD 配置
    svd_rank = mag_config.get("svd_rank", 8)
    svd_normalize = mag_config.get("svd_normalize", True)
    injection_mode = mag_config.get("injection_mode", "svd_only")
    max_raw_kv_tokens = mag_config.get("max_raw_kv_tokens", 128)

    # 4. 加载评估数据
    data = load_eval_data(args.data_path, args.num_eval_samples)

    # 5. 执行各项评估
    all_results = {}
    t0 = time.time()

    if args.eval_selector:
        try:
            all_results["selector"] = eval_selector_accuracy(
                selector, encoder, data, args.device,
            )
        except Exception as e:
            logger.error(f"Selector 评估失败: {e}", exc_info=True)

    if args.eval_gate:
        try:
            all_results["gate"] = eval_gate_activation(
                model, tokenizer, encoder, selector, kv_injector, data,
                args.device, args.max_seq_len,
            )
        except Exception as e:
            logger.error(f"Alpha 评估失败: {e}", exc_info=True)

    if args.eval_ppl:
        try:
            all_results["ppl"] = eval_ppl_comparison(
                model, tokenizer, encoder, selector, kv_injector, data,
                args.device, args.max_seq_len,
                svd_rank=svd_rank, svd_normalize=svd_normalize,
                injection_mode=injection_mode,
                max_raw_kv_tokens=max_raw_kv_tokens,
                inference_scale=args.inference_gate_scale,
            )
        except Exception as e:
            logger.error(f"PPL 评估失败: {e}", exc_info=True)

    if args.eval_generation:
        try:
            # 解析逐层 gate_scale
            layer_gate_scales_dict = None
            if args.layer_gate_scales:
                layer_gate_scales_dict = {}
                for item in args.layer_gate_scales.split(","):
                    parts = item.strip().split(":")
                    if len(parts) == 2:
                        layer_gate_scales_dict[int(parts[0])] = float(parts[1])

            all_results["generation"] = eval_generation_comparison(
                model, tokenizer, encoder, selector, kv_injector, data,
                args.device,
                num_samples=args.num_generate_samples,
                max_new_tokens=args.max_new_tokens,
                max_seq_len=args.max_seq_len,
                svd_rank=svd_rank,
                svd_normalize=svd_normalize,
                max_inject_steps=args.mag_inject_steps,
                injection_mode=injection_mode,
                max_raw_kv_tokens=max_raw_kv_tokens,
                inference_scale=args.inference_gate_scale,
                layer_gate_scales=layer_gate_scales_dict,
            )
        except Exception as e:
            logger.error(f"生成评估失败: {e}", exc_info=True)

    # LLM-as-Judge 评估 (依赖 generation 结果)
    if args.eval_llm_judge:
        gen_samples = []
        if "generation" in all_results and all_results["generation"]:
            gen_samples = all_results["generation"].get("samples", [])

        if gen_samples:
            try:
                # 使用 backbone 自身或指定的 judge 模型
                if args.judge_model_path:
                    from transformers import AutoModelForCausalLM, AutoTokenizer as AT
                    judge_dtype = getattr(torch, args.dtype, torch.bfloat16)
                    logger.info(f"加载 Judge 模型: {args.judge_model_path}")
                    judge_model = AutoModelForCausalLM.from_pretrained(
                        args.judge_model_path, torch_dtype=judge_dtype,
                        device_map="auto", trust_remote_code=True,
                    )
                    judge_tokenizer = AT.from_pretrained(
                        args.judge_model_path, trust_remote_code=True,
                    )
                    all_results["llm_judge"] = eval_llm_judge(
                        judge_model, judge_tokenizer, gen_samples, args.device,
                    )
                    del judge_model  # 释放显存
                else:
                    all_results["llm_judge"] = eval_llm_judge(
                        model, tokenizer, gen_samples, args.device,
                    )
            except Exception as e:
                logger.error(f"LLM Judge 评估失败: {e}", exc_info=True)
        else:
            logger.warning("没有生成结果, 跳过 LLM Judge 评估 (需要先启用 --eval_generation)")

    total_time = time.time() - t0

    # 6. 汇总报告
    logger.info("\n" + "=" * 60)
    logger.info("评估汇总报告")
    logger.info("=" * 60)

    if "selector" in all_results and all_results["selector"]:
        sel = all_results["selector"]
        top_k = selector.config.top_k if not hasattr(selector, "module") else selector.module.config.top_k
        logger.info(f"  Selector Hit@{top_k}: {sel.get(f'Hit@{top_k}', 'N/A'):.4f}")
        logger.info(f"  Selector 区分度:    {sel.get('score_gap', 'N/A'):.4f}")
        if "auc_roc" in sel:
            logger.info(f"  Selector AUC-ROC:   {sel['auc_roc']:.4f}")
        if "cohens_d" in sel:
            logger.info(f"  Selector Cohen's d: {sel['cohens_d']:.4f}")
        if "per_sample_auc_mean" in sel:
            logger.info(f"  Selector 样本AUC:   {sel['per_sample_auc_mean']:.4f} ± {sel.get('per_sample_auc_std', 0):.4f}")

    if "gate" in all_results and all_results["gate"]:
        gate = all_results["gate"]
        logger.info(f"  Alpha 整体均值:     {gate.get('overall_alpha_mean', 'N/A'):.4f}")

    if "ppl" in all_results and all_results["ppl"]:
        ppl = all_results["ppl"]
        logger.info(f"  PPL (有记忆):       {ppl.get('ppl_with_memory', 'N/A'):.4f}")
        logger.info(f"  PPL (无记忆):       {ppl.get('ppl_without_memory', 'N/A'):.4f}")
        logger.info(f"  PPL 降低:           {ppl.get('ppl_reduction_pct', 'N/A'):.2f}%")

    if "generation" in all_results and all_results["generation"]:
        gen = all_results["generation"]
        logger.info(f"  生成差异比例:       {gen.get('pct_different', 'N/A'):.1f}%")
        logger.info(f"  记忆召回 (有记忆):  {gen.get('avg_mem_recall_with_mem', 'N/A'):.4f}")
        logger.info(f"  记忆召回 (无记忆):  {gen.get('avg_mem_recall_no_mem', 'N/A'):.4f}")
        if "avg_rouge_l_with_mem" in gen:
            logger.info(f"  ROUGE-L (有记忆):   {gen['avg_rouge_l_with_mem']:.4f}")
            logger.info(f"  ROUGE-L (无记忆):   {gen['avg_rouge_l_no_mem']:.4f}")
        if "avg_gt_entity_hit_with_mem" in gen:
            logger.info(f"  GT实体命中 (有记忆):{gen['avg_gt_entity_hit_with_mem']:.4f}")
            logger.info(f"  GT实体命中 (无记忆):{gen['avg_gt_entity_hit_no_mem']:.4f}")
        if "avg_mem_penetration_with_mem" in gen:
            logger.info(f"  记忆渗透率 (有记忆):{gen['avg_mem_penetration_with_mem']:.4f}")
            logger.info(f"  记忆渗透率 (无记忆):{gen['avg_mem_penetration_no_mem']:.4f}")

    if "llm_judge" in all_results and all_results["llm_judge"]:
        judge = all_results["llm_judge"]
        logger.info(f"  [Judge] 记忆利用 (有记忆): {judge.get('avg_memory_utilization_with_mem', 0):.2f}/5")
        logger.info(f"  [Judge] 记忆利用 (无记忆): {judge.get('avg_memory_utilization_no_mem', 0):.2f}/5")
        logger.info(f"  [Judge] 记忆利用提升:      {judge.get('memory_utilization_lift', 0):+.2f}")
        logger.info(f"  [Judge] 质量 (有记忆):     {judge.get('avg_quality_with_mem', 0):.2f}/5")
        logger.info(f"  [Judge] 质量 (无记忆):     {judge.get('avg_quality_no_mem', 0):.2f}/5")
        n_b = judge.get('comparative_memory_win_B', 0)
        n_a = judge.get('comparative_memory_win_A', 0)
        n_t = judge.get('comparative_memory_tie', 0)
        logger.info(f"  [Judge] 对比胜率:          有记忆胜={n_b}, 无记忆胜={n_a}, 平局={n_t}")

    logger.info(f"  总耗时: {total_time:.1f}s")

    # 7. 保存结果
    # 自动生成输出路径
    output_json_path = args.output_file or str(Path(args.mag_weights_dir) / "eval_results.json")
    output_report_path = args.output_report or str(Path(args.mag_weights_dir) / "eval_report.txt")

    # 7a. 保存 JSON 结果
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 移除不可序列化的字段
    save_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            save_results[k] = {
                kk: vv for kk, vv in v.items()
                if not isinstance(vv, (torch.Tensor,))
            }
        else:
            save_results[k] = v

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  JSON 结果已保存到: {output_path}")

    # 7b. 保存详细文本报告 (包含所有样本的实际输出)
    _save_detailed_report(output_report_path, all_results, total_time, args)
    logger.info(f"  详细报告已保存到: {output_report_path}")

    # 8. 综合判断
    logger.info("\n" + "=" * 60)
    logger.info("综合判断")
    logger.info("=" * 60)

    issues = []
    good_signs = []

    if "selector" in all_results and all_results["selector"]:
        sel = all_results["selector"]
        gap = sel.get("score_gap", 0)
        auc = sel.get("auc_roc", 0.5)
        cohens_d = sel.get("cohens_d", 0)

        if auc < 0.55:
            issues.append(f"Selector AUC-ROC={auc:.4f} ≈ 随机, 完全没有学到区分能力")
        elif auc < 0.70:
            issues.append(f"Selector AUC-ROC={auc:.4f} 偏低, 区分能力不足")
        elif auc >= 0.85:
            good_signs.append(f"Selector AUC-ROC={auc:.4f} 优秀")
        else:
            good_signs.append(f"Selector AUC-ROC={auc:.4f} 尚可")

        if gap < 0.05:
            issues.append(f"Selector 区分度差 (gap={gap:.4f} < 0.05)")
        elif gap > 0.15:
            good_signs.append(f"Selector 区分度良好 (gap={gap:.4f})")

        if abs(cohens_d) < 0.2 and cohens_d != 0:
            issues.append(f"Cohen's d={cohens_d:.4f}, 效应量极小, 实际区分可忽略")

        # 检查分数是否坍缩到同一区间
        rel_std = sel.get("relevant_score_std", 0)
        irrel_std = sel.get("irrelevant_score_std", 0)
        if rel_std < 0.05 and irrel_std < 0.05:
            issues.append(f"分数方差极小 (rel_std={rel_std:.4f}, irrel_std={irrel_std:.4f}), "
                         f"所有记忆得分坍缩到同一区间, Selector 未学到有效打分")

    if "gate" in all_results and all_results["gate"]:
        gate = all_results["gate"]
        alpha_mean = gate.get("overall_alpha_mean", 0)
        # 检查是否所有层 alpha 一样 (未学到层间差异)
        per_layer = gate.get("per_layer", {})
        if per_layer:
            alphas = [v.get("alpha", 0) for v in per_layer.values()]
            alpha_range = max(alphas) - min(alphas) if alphas else 0
            if alpha_range < 0.01 and len(alphas) > 1:
                issues.append(f"所有层 Alpha 完全一致 ({alphas[0]:.4f}), Gate 未学到层间差异化注入")

        if alpha_mean < 0.03:
            issues.append(f"Alpha 几乎为 0 (mean={alpha_mean:.4f}), KV 注入未生效")
        elif alpha_mean > 0.05:
            good_signs.append(f"Alpha 已学到有意义的值 (mean={alpha_mean:.4f})")

    if "ppl" in all_results and all_results["ppl"]:
        ppl_diff = all_results["ppl"].get("ppl_reduction", 0)
        if ppl_diff <= 0:
            issues.append(f"PPL 未降低 (Δ={ppl_diff:.4f}), 记忆注入无效或有害")
        elif ppl_diff > 0.5:
            good_signs.append(f"PPL 显著降低 ({ppl_diff:.4f})")

    if "generation" in all_results and all_results["generation"]:
        gen = all_results["generation"]
        mr_with = gen.get("avg_mem_recall_with_mem", 0)
        mr_no = gen.get("avg_mem_recall_no_mem", 0)
        if mr_with > mr_no + 0.02:
            good_signs.append(f"记忆关键词召回提升 ({mr_no:.4f} → {mr_with:.4f})")
        elif mr_with < mr_no:
            issues.append(f"记忆关键词召回反而下降 ({mr_no:.4f} → {mr_with:.4f})")

        # ROUGE-L 对比
        rl_with = gen.get("avg_rouge_l_with_mem", 0)
        rl_no = gen.get("avg_rouge_l_no_mem", 0)
        if rl_with > rl_no + 0.01:
            good_signs.append(f"ROUGE-L 提升 ({rl_no:.4f} → {rl_with:.4f})")
        elif rl_with < rl_no - 0.01:
            issues.append(f"ROUGE-L 下降 ({rl_no:.4f} → {rl_with:.4f})")

        # 记忆渗透率
        pen_with = gen.get("avg_mem_penetration_with_mem", 0)
        pen_no = gen.get("avg_mem_penetration_no_mem", 0)
        if pen_with > pen_no + 0.02:
            good_signs.append(f"记忆渗透率提升 ({pen_no:.4f} → {pen_with:.4f})")
        elif pen_with < pen_no:
            issues.append(f"记忆渗透率下降 ({pen_no:.4f} → {pen_with:.4f})")

        # GT 实体命中
        ent_with = gen.get("avg_gt_entity_hit_with_mem", 0)
        ent_no = gen.get("avg_gt_entity_hit_no_mem", 0)
        if ent_with > ent_no + 0.02:
            good_signs.append(f"GT实体命中提升 ({ent_no:.4f} → {ent_with:.4f})")

    if "llm_judge" in all_results and all_results["llm_judge"]:
        judge = all_results["llm_judge"]
        lift = judge.get("memory_utilization_lift", 0)
        if lift > 0.5:
            good_signs.append(f"[Judge] 记忆利用度提升 ({lift:+.2f})")
        elif lift < -0.5:
            issues.append(f"[Judge] 记忆利用度下降 ({lift:+.2f})")
        q_lift = judge.get("quality_lift", 0)
        if q_lift < -0.5:
            issues.append(f"[Judge] 回答质量下降 ({q_lift:+.2f})")
        elif q_lift > 0.5:
            good_signs.append(f"[Judge] 回答质量提升 ({q_lift:+.2f})")

    if good_signs:
        logger.info("  ✅ 积极信号:")
        for sign in good_signs:
            logger.info(f"    - {sign}")
    if issues:
        logger.info("  🔴 问题:")
        for issue in issues:
            logger.info(f"    - {issue}")
    if not issues and not good_signs:
        logger.info("  评估数据不足, 无法做出判断")

    # 详细诊断和改进建议
    if issues:
        logger.info(f"\n  ---- 诊断与改进建议 ----")

        # 诊断 1: Selector 问题
        sel_issues = [i for i in issues if "Selector" in i or "区分度" in i or "Cohen" in i or "坍缩" in i]
        if sel_issues:
            logger.info(f"\n  📋 Selector 诊断:")
            logger.info(f"    问题: Selector 无法区分相关/不相关记忆")
            logger.info(f"    可能原因:")
            logger.info(f"      1. 训练数据中的硬负例不够难 (随机负例太容易区分)")
            logger.info(f"      2. 深层编码层数不够 (当前 deep_encode_layers 可能太浅)")
            logger.info(f"      3. Selector MLP 容量不足 (hidden_dim 太小)")
            logger.info(f"    建议:")
            logger.info(f"      - 增加 --num_hard_negatives (从同一 session 采样更多硬负例)")
            logger.info(f"      - 增加 --deep_encode_layers 到 12 (更深的语义编码)")
            logger.info(f"      - 增加 --selector_hidden_dim 到 512")

        # 诊断 2: Gate/Alpha 问题
        gate_issues = [i for i in issues if "Alpha" in i or "层间" in i]
        if gate_issues:
            logger.info(f"\n  📋 Gate 诊断:")
            logger.info(f"    问题: Gate 未学到有效的注入强度控制")
            logger.info(f"    可能原因:")
            logger.info(f"      1. 训练数据太简单, backbone 不需要记忆就能预测 (lm_loss ≈ 0)")
            logger.info(f"      2. 注入层选择不合理 (层间距太均匀)")
            logger.info(f"    建议:")
            logger.info(f"      - 使用更长的对话数据, 让 target 真正依赖记忆中的信息")
            logger.info(f"      - 手动指定 --mag_injection_layers (如只在高层注入: 21 27 35)")

        # 诊断 3: PPL 问题
        ppl_issues = [i for i in issues if "PPL" in i]
        if ppl_issues:
            logger.info(f"\n  📋 PPL 诊断:")
            logger.info(f"    问题: 记忆注入后 PPL 反而上升")
            logger.info(f"    可能原因:")
            logger.info(f"      1. 注入强度过大, 干扰了 backbone 的正常推理")
            logger.info(f"      2. SVD 压缩损失了关键信息")
            logger.info(f"    建议:")
            logger.info(f"      - 降低 --inference_gate_scale (如 0.1~0.3)")
            logger.info(f"      - 增加 --svd_rank (如 16 或 32)")

        # 诊断 4: 生成质量问题
        gen_issues = [i for i in issues if "召回" in i or "ROUGE" in i or "渗透" in i]
        if gen_issues:
            logger.info(f"\n  📋 生成质量诊断:")
            logger.info(f"    问题: 记忆注入未改善生成质量")
            logger.info(f"    可能原因:")
            logger.info(f"      1. Backbone 本身生成能力差 (base 模型没有对话能力)")
            logger.info(f"      2. 记忆信息未被有效编码到 KV cache 中")
            logger.info(f"    建议:")
            logger.info(f"      - 使用 chat/instruct 版本的 backbone (如 Qwen3-1.7B-Instruct)")
            logger.info(f"      - 增加 --max_inject_steps (让记忆影响更多生成步)")

        if len(issues) >= 3 and len(good_signs) == 0:
            logger.info(f"\n  ⚠️ 多项指标均不理想, 建议优先排查:")
            logger.info(f"    1. 确认 backbone 是否为 chat 模型 (base 模型无法正常对话)")
            logger.info(f"    2. 确认训练数据质量 (target 是否真正依赖记忆信息)")
            logger.info(f"    3. 考虑增加训练轮数或调整学习率")

    elif good_signs and not issues:
        logger.info("\n  MAG 训练效果良好! 可以集成到 MemoryAgent 使用。")
    else:
        logger.info("\n  训练效果有好有坏, 建议针对性调优上述问题项。")


if __name__ == "__main__":
    main()
xian