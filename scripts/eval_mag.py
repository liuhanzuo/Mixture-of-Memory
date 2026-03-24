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
    parser.add_argument("--max_seq_len", type=int, default=512)

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
                        help="评估结果 JSON 输出路径 (可选)")
    parser.add_argument("--seed", type=int, default=42)

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
) -> tuple[ContextSelector, MAGGate, dict]:
    """加载训练好的 MAG 权重。"""
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

    # 初始化 MAGGate
    gate_cfg = MAGGateConfig(
        hidden_dim=mag_config.get("hidden_dim", hidden_dim),
        num_heads=mag_config.get("mag_num_heads", 8),
        memory_dim=mag_config.get("hidden_dim", hidden_dim),
        injection_layers=mag_config.get("injection_layers", []),
        share_parameters=mag_config.get("mag_share_parameters", True),
        gate_init_bias=mag_config.get("mag_gate_init_bias", -2.0),
    )
    mag_gate = MAGGate(gate_cfg).to(device)

    # 加载 gate 权重
    gate_ckpt = weights_path / "mag_gate.pt"
    if gate_ckpt.exists():
        mag_gate.load_state_dict(torch.load(gate_ckpt, map_location=device))
        logger.info(f"MAGGate 权重已加载: {gate_ckpt} ({gate_ckpt.stat().st_size / 1024:.1f} KB)")
    else:
        logger.warning(f"找不到 MAGGate 权重: {gate_ckpt}")

    selector.eval()
    mag_gate.eval()

    return selector, mag_gate, mag_config


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
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_data.append(json.loads(line))
            if len(raw_data) >= num_samples * 5:  # 多读一些, 转换后筛选
                break

    # 检查格式
    if raw_data and "input_text" in raw_data[0] and "memory_texts" in raw_data[0]:
        # 已经是 MAG 格式
        data = raw_data[:num_samples]
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
            for idx in range(K):
                if idx in relevant_set:
                    relevant_score_sum += probs[idx].item()
                    relevant_count += 1
                else:
                    irrelevant_score_sum += probs[idx].item()
                    irrelevant_count += 1

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

    results = {
        f"Hit@{top_k}": hit_at_k / total,
        f"Precision@{top_k}": precision_at_k_sum / total,
        f"NDCG@{top_k}": ndcg_at_k_sum / total,
        "avg_relevant_score": relevant_score_sum / max(relevant_count, 1),
        "avg_irrelevant_score": irrelevant_score_sum / max(irrelevant_count, 1),
        "score_gap": (relevant_score_sum / max(relevant_count, 1)) - (irrelevant_score_sum / max(irrelevant_count, 1)),
        "num_samples": total,
    }

    logger.info(f"  样本数: {total}")
    logger.info(f"  Hit@{top_k}:       {results[f'Hit@{top_k}']:.4f}")
    logger.info(f"  Precision@{top_k}: {results[f'Precision@{top_k}']:.4f}")
    logger.info(f"  NDCG@{top_k}:      {results[f'NDCG@{top_k}']:.4f}")
    logger.info(f"  相关记忆平均分:    {results['avg_relevant_score']:.4f}")
    logger.info(f"  不相关记忆平均分:  {results['avg_irrelevant_score']:.4f}")
    logger.info(f"  区分度 (gap):      {results['score_gap']:.4f}")

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
    mag_gate: MAGGate,
    data: list[dict],
    device: str,
    max_seq_len: int = 512,
) -> dict:
    """分析 MAGGate 在各注入层的 gate 激活情况。

    关注指标:
    - 各层 gate sigmoid 均值 (期望 > 0.2, 说明 gate 在打开)
    - gate 方差 (期望 > 0, 说明 gate 在学习区分性注入)
    - gate bias 当前值
    """
    logger.info("=" * 60)
    logger.info("评估 2: Gate 激活分析")
    logger.info("=" * 60)

    mag_gate_module = mag_gate.module if hasattr(mag_gate, "module") else mag_gate
    selector_module = selector.module if hasattr(selector, "module") else selector

    backbone_layers, final_norm, lm_head, rotary_emb = _get_backbone_layers(model)
    sorted_injection = sorted(mag_gate_module.injection_layers)
    first_injection_layer = sorted_injection[0] if sorted_injection else 0

    gate_stats = {layer_idx: {"means": [], "stds": []} for layer_idx in sorted_injection}

    selector.eval()
    mag_gate.eval()

    num_evaluated = 0
    with torch.no_grad():
        for sample in data[:min(len(data), 50)]:  # 最多评估 50 个样本
            query_emb = encoder.encode_texts([sample["input_text"]])
            memory_embs = encoder.encode_texts_deep(sample["memory_texts"]).unsqueeze(0)
            selection_weights = selector_module.soft_select(query_emb, memory_embs)

            # Tokenize
            if "target_text" in sample and sample["target_text"]:
                query_enc = tokenizer(sample["input_text"], add_special_tokens=True,
                                      truncation=True, max_length=max_seq_len // 2)
                target_enc = tokenizer(sample["target_text"], add_special_tokens=False,
                                       truncation=True, max_length=max_seq_len // 2)
                combined_ids = query_enc["input_ids"] + target_enc["input_ids"]
                if len(combined_ids) > max_seq_len:
                    combined_ids = combined_ids[:max_seq_len]
                input_ids = torch.tensor([combined_ids], device=device)
            else:
                encoded = tokenizer(sample["input_text"], return_tensors="pt",
                                    truncation=True, max_length=max_seq_len)
                input_ids = encoded["input_ids"].to(device)

            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            # Forward through backbone with MAG injection
            if hasattr(model.model, "embed_tokens"):
                h = model.model.embed_tokens(input_ids)
            else:
                h = model.get_input_embeddings()(input_ids)

            position_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
            position_embeddings = _compute_position_embeddings(model, h, position_ids, rotary_emb)

            layer_kwargs = {"attention_mask": attention_mask}
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings
            else:
                layer_kwargs["position_ids"] = position_ids

            # Forward 到第一个注入层之前
            for layer_idx in range(first_injection_layer):
                layer_output = backbone_layers[layer_idx](h, **layer_kwargs)
                h = layer_output[0] if isinstance(layer_output, tuple) else layer_output

            # 从第一个注入层开始
            for layer_idx in range(first_injection_layer, len(backbone_layers)):
                layer_output = backbone_layers[layer_idx](h, **layer_kwargs)
                h = layer_output[0] if isinstance(layer_output, tuple) else layer_output

                if layer_idx in mag_gate_module.injection_layers:
                    h, gate_val = mag_gate_module.inject(
                        layer_idx, h, memory_embs,
                        selection_weights=selection_weights,
                        return_gate=True,
                    )
                    if gate_val is not None:
                        gate_stats[layer_idx]["means"].append(gate_val.mean().item())
                        gate_stats[layer_idx]["stds"].append(gate_val.std().item())

            num_evaluated += 1

    # 汇总统计
    results = {"per_layer": {}, "overall_gate_mean": 0.0}
    all_means = []

    for layer_idx in sorted_injection:
        means = gate_stats[layer_idx]["means"]
        stds = gate_stats[layer_idx]["stds"]
        if means:
            avg_mean = sum(means) / len(means)
            avg_std = sum(stds) / len(stds)
            all_means.append(avg_mean)
            results["per_layer"][str(layer_idx)] = {
                "avg_gate_mean": avg_mean,
                "avg_gate_std": avg_std,
            }
            logger.info(f"  Layer {layer_idx}: gate_mean={avg_mean:.4f}, gate_std={avg_std:.4f}")

    if all_means:
        results["overall_gate_mean"] = sum(all_means) / len(all_means)

    # Gate bias 诊断
    try:
        if mag_gate_module.config.share_parameters and mag_gate_module._shared_block is not None:
            bias = mag_gate_module._shared_block.gate_proj.bias.data.mean().item()
            results["gate_bias"] = bias
            logger.info(f"  Gate bias (共享): {bias:.4f}")
    except Exception:
        pass

    # 判断效果
    overall = results["overall_gate_mean"]
    logger.info(f"  整体 gate 均值: {overall:.4f}")
    if overall > 0.3:
        logger.info("  ✅ Gate 已打开, MAG 在积极注入记忆")
    elif overall > 0.15:
        logger.info("  🟡 Gate 部分打开, MAG 在谨慎注入")
    else:
        logger.info("  🔴 Gate 几乎关闭 (≈ sigmoid(-2)=0.12), MAG 未生效")

    logger.info(f"  评估样本数: {num_evaluated}")

    return results


# ======================================================================
# 评估 3: PPL 对比 (有记忆 vs 无记忆)
# ======================================================================

def eval_ppl_comparison(
    model: nn.Module,
    tokenizer: Any,
    encoder: MemoryEncoder,
    selector: ContextSelector,
    mag_gate: MAGGate,
    data: list[dict],
    device: str,
    max_seq_len: int = 512,
) -> dict:
    """对比有记忆注入和无记忆注入时的困惑度 (PPL)。

    核心: 如果 MAG 学到了有用信息, 有记忆时 PPL 应该更低。
    """
    logger.info("=" * 60)
    logger.info("评估 3: PPL 对比 (有记忆 vs 无记忆)")
    logger.info("=" * 60)

    mag_gate_module = mag_gate.module if hasattr(mag_gate, "module") else mag_gate
    selector_module = selector.module if hasattr(selector, "module") else selector

    backbone_layers, final_norm, lm_head, rotary_emb = _get_backbone_layers(model)
    sorted_injection = sorted(mag_gate_module.injection_layers)
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
    mag_gate.eval()

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

            # 编码记忆 (深层编码)
            query_emb = encoder.encode_texts([sample["input_text"]])
            memory_embs = encoder.encode_texts_deep(sample["memory_texts"]).unsqueeze(0)
            selection_weights = selector_module.soft_select(query_emb, memory_embs)

            # --- Forward: 有记忆 ---
            loss_with = _forward_with_mag(
                model, input_ids, attention_mask, labels,
                backbone_layers, final_norm, lm_head, rotary_emb,
                mag_gate_module, memory_embs, selection_weights,
                first_injection_layer, device,
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


def _forward_with_mag(
    model, input_ids, attention_mask, labels,
    backbone_layers, final_norm, lm_head, rotary_emb,
    mag_gate_module, memory_embs, selection_weights,
    first_injection_layer, device,
) -> float:
    """有 MAG 注入的 forward, 返回 loss 标量。"""
    if hasattr(model.model, "embed_tokens"):
        h = model.model.embed_tokens(input_ids)
    else:
        h = model.get_input_embeddings()(input_ids)

    position_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
    position_embeddings = _compute_position_embeddings(model, h, position_ids, rotary_emb)

    layer_kwargs = {"attention_mask": attention_mask}
    if position_embeddings is not None:
        layer_kwargs["position_embeddings"] = position_embeddings
    else:
        layer_kwargs["position_ids"] = position_ids

    for layer_idx in range(len(backbone_layers)):
        layer_output = backbone_layers[layer_idx](h, **layer_kwargs)
        h = layer_output[0] if isinstance(layer_output, tuple) else layer_output

        if layer_idx in mag_gate_module.injection_layers:
            h = mag_gate_module.inject(
                layer_idx, h, memory_embs,
                selection_weights=selection_weights,
            )

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

    position_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
    position_embeddings = _compute_position_embeddings(model, h, position_ids, rotary_emb)

    layer_kwargs = {"attention_mask": attention_mask}
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
    mag_gate: MAGGate,
    data: list[dict],
    device: str,
    num_samples: int = 5,
    max_new_tokens: int = 128,
    max_seq_len: int = 512,
    gate_scale: float = 0.1,
    layer_gate_scales: dict[int, float] | None = None,
    max_inject_steps: int = 10,
) -> dict:
    """对比有记忆 vs 无记忆的生成结果。

    选取几个有代表性的样本, 分别:
    - 无记忆: 纯 backbone 生成
    - 有记忆: MAG 注入后生成 (使用 gate_scale 缩放注入强度)
    展示两者的差异, 供人工评估。
    """
    logger.info("=" * 60)
    logger.info("评估 4: 生成质量对比")
    logger.info("=" * 60)
    logger.info(f"  推理 gate_scale = {gate_scale} (1.0=训练时强度, 越小注入越轻)")
    logger.info(f"  最大注入步数 = {max_inject_steps} (超过后 backbone 完全接管)")
    if layer_gate_scales:
        logger.info(f"  逐层 gate_scale: {layer_gate_scales}")

    mag_gate_module = mag_gate.module if hasattr(mag_gate, "module") else mag_gate
    selector_module = selector.module if hasattr(selector, "module") else selector

    samples_with_target = [s for s in data if "target_text" in s and s["target_text"]]
    if not samples_with_target:
        logger.warning("没有包含 target_text 的样本, 跳过生成对比!")
        return {}

    # 选取样本 (取有较多记忆的)
    selected = sorted(samples_with_target, key=lambda s: len(s.get("memory_texts", [])), reverse=True)
    selected = selected[:num_samples]

    generation_results = []

    selector.eval()
    mag_gate.eval()

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
            query_emb = encoder.encode_texts([sample["input_text"]])
            memory_embs = encoder.encode_texts_deep(sample["memory_texts"]).unsqueeze(0)
            selection_weights = selector_module.soft_select(query_emb, memory_embs)

            gen_with_mem = _generate_text_with_mag(
                model, tokenizer, sample["input_text"],
                mag_gate_module=mag_gate_module,
                memory_embs=memory_embs,
                selection_weights=selection_weights,
                max_new_tokens=max_new_tokens,
                max_seq_len=max_seq_len,
                device=device,
                gate_scale=gate_scale,
                layer_gate_scales=layer_gate_scales,
                max_inject_steps=max_inject_steps,
            )

        logger.info(f"  生成 (无记忆): {gen_no_mem[:150]}...")
        logger.info(f"  生成 (有记忆): {gen_with_mem[:150]}...")

        # 简单文本相似度 (词重叠率)
        gt_words = set(sample["target_text"].lower().split())
        no_mem_words = set(gen_no_mem.lower().split())
        with_mem_words = set(gen_with_mem.lower().split())

        overlap_no = len(gt_words & no_mem_words) / max(len(gt_words), 1)
        overlap_with = len(gt_words & with_mem_words) / max(len(gt_words), 1)

        logger.info(f"  词重叠 (无记忆 vs GT): {overlap_no:.4f}")
        logger.info(f"  词重叠 (有记忆 vs GT): {overlap_with:.4f}")

        # 检查生成是否有差异
        is_different = gen_no_mem.strip() != gen_with_mem.strip()
        logger.info(f"  两种生成是否不同: {'是 ✅' if is_different else '否 🔴'}")

        generation_results.append({
            "query": sample["input_text"],
            "ground_truth": sample["target_text"],
            "gen_no_memory": gen_no_mem,
            "gen_with_memory": gen_with_mem,
            "overlap_no_mem": overlap_no,
            "overlap_with_mem": overlap_with,
            "is_different": is_different,
        })

    # 汇总
    n_different = sum(1 for r in generation_results if r["is_different"])
    avg_overlap_no = sum(r["overlap_no_mem"] for r in generation_results) / max(len(generation_results), 1)
    avg_overlap_with = sum(r["overlap_with_mem"] for r in generation_results) / max(len(generation_results), 1)

    results = {
        "num_samples": len(generation_results),
        "num_different_outputs": n_different,
        "pct_different": n_different / max(len(generation_results), 1) * 100,
        "avg_word_overlap_no_mem": avg_overlap_no,
        "avg_word_overlap_with_mem": avg_overlap_with,
        "samples": generation_results,
    }

    logger.info(f"\n  生成结果汇总:")
    logger.info(f"  两种生成有差异的比例: {results['pct_different']:.1f}%")
    logger.info(f"  平均词重叠 (无记忆): {avg_overlap_no:.4f}")
    logger.info(f"  平均词重叠 (有记忆): {avg_overlap_with:.4f}")

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
    max_seq_len: int = 512,
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


def _generate_text_with_mag(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    mag_gate_module: MAGGate,
    memory_embs: torch.Tensor,
    selection_weights: torch.Tensor,
    max_new_tokens: int = 128,
    max_seq_len: int = 512,
    device: str = "cuda",
    repetition_penalty: float = 1.5,
    gate_scale: float = 0.3,
    layer_gate_scales: dict[int, float] | None = None,
    max_inject_steps: int = 10,
) -> str:
    """有 MAG 注入的文本生成 — Last-token-only + 硬截断策略。

    ★ 核心策略: Last-token-only + 硬截断
      1. 每步仍做 full sequence forward (无 KV cache, 简单可靠)
      2. MAG 只注入到**最后一个 token 位置** (当前预测位置)
         - 历史 token 的 hidden state 不被 MAG 修改, 保持 backbone 原始表征
      3. 只在前 max_inject_steps 步注入, 之后完全停止 (backbone 接管)
      4. 在注入窗口内, gate_scale 随步数线性衰减至 0:
           scale(step) = gate_scale × (1 - step / max_inject_steps)
         - 比指数衰减更可控, 到截断点刚好为 0

      公式 (step < max_inject_steps):
        对于位置 t < last_pos:  h'[t] = h[t]                              # 不注入
        对于位置 t = last_pos:  h'[t] = h[t] + scale(step) × (g ⊙ W_o m)  # 仅此处注入

      公式 (step >= max_inject_steps):
        h' = h  (所有位置, 完全不注入)

    额外保护:
    - Repetition penalty (1.5): 对已生成 token 降权, 防止循环
    - N-gram 重复检测: 4-gram 重复 3 次则强制终止
    - 硬截断: 超过 max_inject_steps 步后, 零注入
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

    # max_inject_steps=0 表示不限制, 只靠衰减 (兼容旧逻辑)
    effective_inject_steps = max_inject_steps if max_inject_steps > 0 else max_new_tokens

    with torch.no_grad():
        for gen_step in range(max_new_tokens):
            # 当前序列
            cur_ids = generated_ids
            if cur_ids.shape[1] > max_seq_len:
                cur_ids = cur_ids[:, -max_seq_len:]

            # Embedding
            if hasattr(model.model, "embed_tokens"):
                h = model.model.embed_tokens(cur_ids)
            else:
                h = model.get_input_embeddings()(cur_ids)

            attention_mask = torch.ones(1, h.shape[1], device=device, dtype=torch.bool)
            position_ids = torch.arange(h.shape[1], device=device).unsqueeze(0)
            position_embeddings = _compute_position_embeddings(model, h, position_ids, rotary_emb)

            layer_kwargs: dict[str, Any] = {"attention_mask": attention_mask}
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings
            else:
                layer_kwargs["position_ids"] = position_ids

            # 当前步的 gate_scale: 线性衰减 + 硬截断
            if gen_step < effective_inject_steps:
                # 线性衰减: gate_scale × (1 - step/max_steps)
                cur_scale = gate_scale * (1.0 - gen_step / effective_inject_steps)
            else:
                cur_scale = 0.0  # 硬截断: 完全不注入

            # Forward all layers, MAG 只注入最后一个 token 位置
            for layer_idx in range(num_layers):
                layer_output = backbone_layers[layer_idx](h, **layer_kwargs)
                h = layer_output[0] if isinstance(layer_output, tuple) else layer_output

                if layer_idx in mag_gate_module.injection_layers and cur_scale > 0.005:
                    # 确定该层的 scale
                    layer_scale = cur_scale
                    if layer_gate_scales and layer_idx in layer_gate_scales:
                        layer_scale = layer_gate_scales[layer_idx] * (1.0 - gen_step / effective_inject_steps)

                    # ★ 只对最后一个 token 位置注入 MAG
                    h_last = h[:, -1:, :]  # (B, 1, D)
                    h_last = mag_gate_module.inject(
                        layer_idx, h_last, memory_embs,
                        selection_weights=selection_weights,
                        gate_scale=layer_scale,
                    )
                    h = torch.cat([h[:, :-1, :], h_last], dim=1)

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
    selector, mag_gate, mag_config = load_mag_weights(
        args.mag_weights_dir, hidden_dim, args.device, args.dtype,
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

    # 4. 加载评估数据
    data = load_eval_data(args.data_path, args.num_eval_samples)

    # 5. 执行各项评估
    all_results = {}
    t0 = time.time()

    # 解析逐层 gate_scale
    layer_gate_scales: dict[int, float] | None = None
    if args.layer_gate_scales:
        layer_gate_scales = {}
        for part in args.layer_gate_scales.split(","):
            part = part.strip()
            if ":" in part:
                layer_str, scale_str = part.split(":", 1)
                layer_gate_scales[int(layer_str)] = float(scale_str)
        if layer_gate_scales:
            logger.info(f"逐层 gate_scale: {layer_gate_scales}")
        else:
            layer_gate_scales = None

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
                model, tokenizer, encoder, selector, mag_gate, data,
                args.device, args.max_seq_len,
            )
        except Exception as e:
            logger.error(f"Gate 评估失败: {e}", exc_info=True)

    if args.eval_ppl:
        try:
            all_results["ppl"] = eval_ppl_comparison(
                model, tokenizer, encoder, selector, mag_gate, data,
                args.device, args.max_seq_len,
            )
        except Exception as e:
            logger.error(f"PPL 评估失败: {e}", exc_info=True)

    if args.eval_generation:
        try:
            all_results["generation"] = eval_generation_comparison(
                model, tokenizer, encoder, selector, mag_gate, data,
                args.device,
                num_samples=args.num_generate_samples,
                max_new_tokens=args.max_new_tokens,
                max_seq_len=args.max_seq_len,
                gate_scale=args.inference_gate_scale,
                layer_gate_scales=layer_gate_scales,
                max_inject_steps=args.mag_inject_steps,
            )
        except Exception as e:
            logger.error(f"生成评估失败: {e}", exc_info=True)

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

    if "gate" in all_results and all_results["gate"]:
        gate = all_results["gate"]
        logger.info(f"  Gate 整体均值:      {gate.get('overall_gate_mean', 'N/A'):.4f}")
        if "gate_bias" in gate:
            logger.info(f"  Gate bias:          {gate['gate_bias']:.4f}")

    if "ppl" in all_results and all_results["ppl"]:
        ppl = all_results["ppl"]
        logger.info(f"  PPL (有记忆):       {ppl.get('ppl_with_memory', 'N/A'):.4f}")
        logger.info(f"  PPL (无记忆):       {ppl.get('ppl_without_memory', 'N/A'):.4f}")
        logger.info(f"  PPL 降低:           {ppl.get('ppl_reduction_pct', 'N/A'):.2f}%")

    if "generation" in all_results and all_results["generation"]:
        gen = all_results["generation"]
        logger.info(f"  生成差异比例:       {gen.get('pct_different', 'N/A'):.1f}%")

    logger.info(f"  总耗时: {total_time:.1f}s")

    # 7. 保存结果
    if args.output_file:
        output_path = Path(args.output_file)
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
        logger.info(f"  结果已保存到: {output_path}")

    # 8. 综合判断
    logger.info("\n" + "=" * 60)
    logger.info("综合判断")
    logger.info("=" * 60)

    issues = []
    good_signs = []

    if "selector" in all_results and all_results["selector"]:
        gap = all_results["selector"].get("score_gap", 0)
        if gap < 0.05:
            issues.append("Selector 区分度差 (gap < 0.05), 无法有效区分相关/不相关记忆")
        elif gap > 0.15:
            good_signs.append(f"Selector 区分度良好 (gap={gap:.4f})")

    if "gate" in all_results and all_results["gate"]:
        gate_mean = all_results["gate"].get("overall_gate_mean", 0)
        if gate_mean < 0.15:
            issues.append(f"Gate 未打开 (mean={gate_mean:.4f} ≈ 初始值 0.12), MAG 未生效")
        elif gate_mean > 0.25:
            good_signs.append(f"Gate 已打开 (mean={gate_mean:.4f})")

    if "ppl" in all_results and all_results["ppl"]:
        ppl_diff = all_results["ppl"].get("ppl_reduction", 0)
        if ppl_diff <= 0:
            issues.append("PPL 未降低, 记忆注入无效或有害")
        elif ppl_diff > 0.5:
            good_signs.append(f"PPL 显著降低 ({ppl_diff:.4f})")

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

    if issues and not good_signs:
        logger.info("\n  建议: 训练效果不佳, 可能原因:")
        logger.info("    1. 训练数据太简单 (DailyDialog 对话太短, backbone 不需要记忆)")
        logger.info("    2. 训练轮数不够 (lm_loss ≈ 0 说明数据太简单)")
        logger.info("    3. 建议换用更有挑战性的数据 (如 LoCoMo, MSC 长对话)")
    elif good_signs and not issues:
        logger.info("\n  MAG 训练效果良好! 可以集成到 MemoryAgent 使用。")


if __name__ == "__main__":
    main()
