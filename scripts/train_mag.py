#!/usr/bin/env python3
"""
MAG 训练脚本: 联合训练 ContextSelector + MAGGate。

训练流程:
1. 加载 backbone (Qwen3) 并冻结其参数
2. 初始化 MemoryEncoder (共享 backbone embedding, 不训练)
3. 初始化 ContextSelector (可训练)
4. 初始化 MAGGate (可训练)
5. 使用 MoM benchmark 数据作为训练数据:
   - Phase 1: 收集 counterfactual ΔLoss 数据, 预训练 Scorer
   - Phase 2: 联合训练 Scorer + Gate, 端到端优化 LM loss
6. 保存训练好的 Scorer + Gate 权重

Usage::

    python scripts/train_mag.py \\
        --model_path ../models/Qwen--Qwen3-1.7b \\
        --output_dir outputs/mag_trained \\
        --num_epochs 3 \\
        --batch_size 4 \\
        --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

# 将项目根目录加入 sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.memory.mag.memory_encoder import MemoryEncoder, MemoryEncoderConfig
from src.memory.mag.context_selector import ContextSelector, ContextSelectorConfig
from src.memory.mag.mag_gate import MAGGate, MAGGateConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("train_mag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAG (Memory-Augmented Generation) 训练脚本")

    # 模型配置
    parser.add_argument("--model_path", type=str, default="../models/Qwen--Qwen3-1.7b",
                        help="Backbone 模型路径")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # MAG 架构配置
    parser.add_argument("--mag_num_heads", type=int, default=8, help="CrossAttention 头数")
    parser.add_argument("--mag_injection_layers", type=int, nargs="+", default=None,
                        help="注入的 Transformer 层索引 (如 6 12 18 23)")
    parser.add_argument("--mag_share_parameters", action="store_true", default=True,
                        help="注入层之间是否共享参数")
    parser.add_argument("--mag_gate_init_bias", type=float, default=-2.0,
                        help="Gate 偏置初始值 (负值使初始 gate 接近 0)")
    parser.add_argument("--selector_hidden_dim", type=int, default=256,
                        help="Selector MLP 隐藏层维度")
    parser.add_argument("--selector_top_k", type=int, default=5, help="选出的 top-k 记忆数")
    parser.add_argument("--max_memory_tokens", type=int, default=64, help="每条记忆最大 token 数")

    # 训练配置
    parser.add_argument("--output_dir", type=str, default="outputs/mag_trained")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512, help="训练序列最大长度")
    parser.add_argument("--grad_accumulation_steps", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)

    # 数据配置
    parser.add_argument("--num_synthetic_samples", type=int, default=1000,
                        help="合成训练样本数量")
    parser.add_argument("--num_memories_per_sample", type=int, default=10,
                        help="每个训练样本的候选记忆条数")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_backbone(model_path: str, device: str, dtype_str: str) -> tuple[nn.Module, Any, int, int]:
    """加载 backbone 模型和 tokenizer。

    Returns:
        (model, tokenizer, hidden_dim, num_layers)
    """
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    dtype = getattr(torch, dtype_str, torch.float32)

    logger.info(f"加载模型: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 冻结 backbone 参数
    for param in model.parameters():
        param.requires_grad = False

    hidden_dim = config.hidden_size
    num_layers = config.num_hidden_layers
    logger.info(f"模型加载完成: hidden_dim={hidden_dim}, num_layers={num_layers}")

    return model, tokenizer, hidden_dim, num_layers


def generate_synthetic_data(
    tokenizer: Any,
    num_samples: int,
    num_memories: int,
    max_seq_len: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """生成合成训练数据。

    每个样本包含:
    - input_text: 一段对话文本 (query)
    - memory_texts: 一组候选记忆文本
    - relevant_indices: 与 query 相关的记忆索引 (ground truth)

    Returns:
        训练样本列表.
    """
    import random
    random.seed(seed)

    # 预定义话题和记忆模板
    topics = [
        ("NLP", "自然语言处理", ["BERT", "GPT", "Transformer", "预训练", "微调"]),
        ("CV", "计算机视觉", ["ResNet", "ViT", "目标检测", "图像分类", "分割"]),
        ("推荐系统", "推荐算法", ["协同过滤", "深度推荐", "CTR预测", "召回", "排序"]),
        ("强化学习", "RL", ["PPO", "DQN", "策略梯度", "奖励函数", "环境"]),
        ("数据库", "SQL", ["索引", "查询优化", "事务", "分布式", "缓存"]),
    ]

    memory_templates = [
        "用户正在研究{topic}领域的{keyword}",
        "用户的项目涉及{keyword}技术",
        "用户偏好使用{keyword}方法",
        "上次讨论了{topic}的{keyword}问题",
        "用户提到了{keyword}的最新进展",
    ]

    query_templates = [
        "关于{keyword}，你能告诉我最新进展吗？",
        "我之前说过关于{keyword}的什么？",
        "{keyword}的原理是什么？",
        "帮我回忆一下我们讨论过的{topic}",
        "我的研究方向是什么来着？",
    ]

    noise_memories = [
        "用户喜欢喝咖啡",
        "今天天气不错",
        "用户使用 macOS 系统",
        "上次对话是两天前",
        "用户说下周要开会",
        "用户提到了周末计划",
        "最近在看一本小说",
        "用户的时区是 UTC+8",
    ]

    samples: list[dict[str, Any]] = []

    for _ in range(num_samples):
        # 随机选择一个话题
        topic_name, topic_desc, keywords = random.choice(topics)
        keyword = random.choice(keywords)

        # 生成 query
        query = random.choice(query_templates).format(
            topic=topic_name, keyword=keyword
        )

        # 生成记忆列表 (部分相关, 部分不相关)
        num_relevant = random.randint(1, min(3, num_memories))
        memories: list[str] = []
        relevant: list[int] = []

        # 相关记忆
        for i in range(num_relevant):
            kw = keywords[i % len(keywords)]
            mem = random.choice(memory_templates).format(
                topic=topic_name, keyword=kw
            )
            memories.append(mem)
            relevant.append(i)

        # 不相关记忆 (噪声)
        for _ in range(num_memories - num_relevant):
            # 有一定概率来自其他话题
            if random.random() < 0.3:
                other_topic, _, other_kws = random.choice([t for t in topics if t[0] != topic_name])
                mem = random.choice(memory_templates).format(
                    topic=other_topic, keyword=random.choice(other_kws)
                )
            else:
                mem = random.choice(noise_memories)
            memories.append(mem)

        # 打乱记忆顺序
        indices = list(range(len(memories)))
        random.shuffle(indices)
        shuffled_memories = [memories[i] for i in indices]
        shuffled_relevant = [indices.index(r) for r in relevant]

        samples.append({
            "input_text": query,
            "memory_texts": shuffled_memories,
            "relevant_indices": shuffled_relevant,
        })

    logger.info(f"生成了 {len(samples)} 个合成训练样本")
    return samples


def train_phase1_scorer(
    model: nn.Module,
    tokenizer: Any,
    encoder: MemoryEncoder,
    selector: ContextSelector,
    train_data: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Phase 1: 预训练 Scorer (使用合成数据的 ground truth relevance)。"""
    logger.info("=" * 60)
    logger.info("Phase 1: 预训练 ContextSelector (Scorer)")
    logger.info("=" * 60)

    optimizer = optim.AdamW(
        selector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    selector.train()
    total_loss = 0.0
    num_steps = 0

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for i, sample in enumerate(train_data):
            # 编码 query
            query_emb = encoder.encode_texts([sample["input_text"]])  # (1, D)

            # 编码记忆
            memory_embs = encoder.encode_texts(sample["memory_texts"])  # (K, D)
            memory_embs = memory_embs.unsqueeze(0)  # (1, K, D)

            # 构建 target utilities (相关=1.0, 不相关=0.0)
            K = len(sample["memory_texts"])
            target_utilities = torch.zeros(1, K, device=query_emb.device)
            for idx in sample["relevant_indices"]:
                target_utilities[0, idx] = 1.0

            # Forward
            scores = selector(query_emb, memory_embs)  # (1, K)

            # Loss
            loss = selector.compute_loss(scores, target_utilities)

            # 梯度累积
            loss = loss / args.grad_accumulation_steps
            loss.backward()

            if (i + 1) % args.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                num_steps += 1

            epoch_loss += loss.item() * args.grad_accumulation_steps
            epoch_steps += 1

            if (i + 1) % args.log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                logger.info(
                    f"  Epoch {epoch+1}, Step {i+1}/{len(train_data)}: "
                    f"loss={avg_loss:.4f}"
                )

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch+1} 完成: avg_loss={avg_epoch_loss:.4f}")

    logger.info(f"Phase 1 完成: total_steps={num_steps}")


def train_phase2_joint(
    model: nn.Module,
    tokenizer: Any,
    encoder: MemoryEncoder,
    selector: ContextSelector,
    mag_gate: MAGGate,
    train_data: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Phase 2: 联合训练 Selector + MAGGate (端到端 LM loss)。"""
    logger.info("=" * 60)
    logger.info("Phase 2: 联合训练 Selector + MAGGate (端到端)")
    logger.info("=" * 60)

    # 只训练 selector + gate 的参数
    trainable_params = list(selector.parameters()) + list(mag_gate.parameters())
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr * 0.1,  # Phase 2 用更小的学习率
        weight_decay=args.weight_decay,
    )

    selector.train()
    mag_gate.train()
    total_loss = 0.0
    num_steps = 0

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for i, sample in enumerate(train_data):
            # 编码 query
            query_emb = encoder.encode_texts([sample["input_text"]])  # (1, D)

            # 编码记忆
            memory_embs = encoder.encode_with_grad(sample["memory_texts"])  # (K, D)
            memory_embs = memory_embs.unsqueeze(0)  # (1, K, D)

            # Selector soft selection
            selection_weights = selector.soft_select(query_emb, memory_embs)  # (1, K)

            # Tokenize query
            encoded = tokenizer(
                sample["input_text"],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_len,
            )
            input_ids = encoded["input_ids"].to(args.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(args.device)

            # Forward: backbone → 获取 hidden states
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    labels=input_ids,  # self-supervised
                )
                base_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0)
                all_hidden = list(outputs.hidden_states)

            # MAG 注入: 在指定层注入记忆
            # 注意: 这里需要让梯度流过 mag_gate 和 selector
            modified_hidden = mag_gate.inject_into_all_hidden_states(
                all_hidden_states=all_hidden,
                memory_vectors=memory_embs,
                selection_weights=selection_weights,
            )

            # 用修改后的 last hidden state 重新计算 logits 和 loss
            h_final = modified_hidden[-1]

            # 通过 lm_head (冻结的)
            if hasattr(model, "lm_head"):
                logits = model.lm_head(h_final)
            else:
                logits = h_final  # fallback

            # 计算 LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # 梯度累积
            loss = loss / args.grad_accumulation_steps
            loss.backward()

            if (i + 1) % args.grad_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                num_steps += 1

            epoch_loss += loss.item() * args.grad_accumulation_steps
            epoch_steps += 1

            if (i + 1) % args.log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                logger.info(
                    f"  Epoch {epoch+1}, Step {i+1}/{len(train_data)}: "
                    f"loss={avg_loss:.4f}, base_loss={base_loss.item():.4f}"
                )

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch+1} 完成: avg_loss={avg_epoch_loss:.4f}")

    logger.info(f"Phase 2 完成: total_steps={num_steps}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载 backbone
    model, tokenizer, hidden_dim, num_layers = load_backbone(
        args.model_path, args.device, args.dtype
    )

    # 2. 初始化 MemoryEncoder
    enc_cfg = MemoryEncoderConfig(max_memory_tokens=args.max_memory_tokens, pooling="mean")
    encoder = MemoryEncoder(enc_cfg)
    encoder.set_backbone(
        backbone_model=model,
        tokenizer=tokenizer,
        hidden_dim=hidden_dim,
        device=args.device,
        dtype=getattr(torch, args.dtype, torch.float32),
    )

    # 3. 初始化 ContextSelector
    sel_cfg = ContextSelectorConfig(
        input_dim=hidden_dim,
        hidden_dim=args.selector_hidden_dim,
        top_k=args.selector_top_k,
    )
    selector = ContextSelector(sel_cfg).to(args.device)

    # 4. 初始化 MAGGate
    injection_layers = args.mag_injection_layers
    if injection_layers is None:
        # 默认: 均匀选择 4 个层
        injection_layers = [
            num_layers // 4,
            num_layers // 2,
            num_layers * 3 // 4,
            num_layers - 1,
        ]

    gate_cfg = MAGGateConfig(
        hidden_dim=hidden_dim,
        num_heads=args.mag_num_heads,
        memory_dim=hidden_dim,
        injection_layers=injection_layers,
        share_parameters=args.mag_share_parameters,
        gate_init_bias=args.mag_gate_init_bias,
    )
    mag_gate = MAGGate(gate_cfg).to(args.device)

    # 打印参数量
    total_trainable = sum(p.numel() for p in selector.parameters()) + sum(p.numel() for p in mag_gate.parameters())
    logger.info(f"可训练参数量: {total_trainable:,}")
    logger.info(f"  Selector: {sum(p.numel() for p in selector.parameters()):,}")
    logger.info(f"  MAGGate: {sum(p.numel() for p in mag_gate.parameters()):,}")
    logger.info(f"注入层: {injection_layers}")

    # 5. 生成训练数据
    train_data = generate_synthetic_data(
        tokenizer=tokenizer,
        num_samples=args.num_synthetic_samples,
        num_memories=args.num_memories_per_sample,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )

    t0 = time.time()

    # Phase 1: 预训练 Scorer
    train_phase1_scorer(model, tokenizer, encoder, selector, train_data, args)

    # Phase 2: 联合训练 Selector + Gate
    train_phase2_joint(model, tokenizer, encoder, selector, mag_gate, train_data, args)

    total_time = time.time() - t0

    # 6. 保存
    save_path = Path(args.output_dir)
    torch.save(selector.state_dict(), save_path / "context_selector.pt")
    torch.save(mag_gate.state_dict(), save_path / "mag_gate.pt")

    # 保存配置
    config_info = {
        "model_path": args.model_path,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "injection_layers": injection_layers,
        "mag_num_heads": args.mag_num_heads,
        "mag_share_parameters": args.mag_share_parameters,
        "mag_gate_init_bias": args.mag_gate_init_bias,
        "selector_hidden_dim": args.selector_hidden_dim,
        "selector_top_k": args.selector_top_k,
        "max_memory_tokens": args.max_memory_tokens,
        "total_trainable_params": total_trainable,
        "training_time_s": total_time,
    }
    with open(save_path / "mag_config.json", "w") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)

    logger.info(f"训练完成! 总耗时: {total_time:.1f}s")
    logger.info(f"权重已保存到: {args.output_dir}")
    logger.info(f"  context_selector.pt: {(save_path / 'context_selector.pt').stat().st_size / 1024:.1f} KB")
    logger.info(f"  mag_gate.pt: {(save_path / 'mag_gate.pt').stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
